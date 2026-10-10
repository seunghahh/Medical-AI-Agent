#!/usr/bin/env python3
"""Bounded development-only prompt search; does not train weights or edit the base prompt."""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from clinic.client import ChatClient, ModelError
from clinic.agent import DOCTOR_PROMPT
from clinic.data import ROOT, DATA
from clinic.evaluation import load_split, read_run, summarize
from run import positive

OPTIMIZER_PROMPT = """Improve generic workflow guidance for a simulated diagnostic agent.
Return JSON {"instructions":"...", "reason":"..."}. Instructions must be concise,
case-independent, and supplement existing action, evidence, and stopping rules.
Use aggregate failure categories to improve information gathering, specificity, stopping,
and response validation. Do not name diseases, memorize cases, request hidden labels,
change evaluation criteria, or weaken base rules. No patient or reference labels are provided.
Read base_rules carefully. UNKNOWN means unavailable recorded data; clarification cannot
recover it. Each encounter is independent: never compare diagnoses across patients.
Do not recommend weight training. This is prompt search on public development cases.
"""


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def failure_analysis(results, manifest):
    rows, summary = summarize(results, manifest)
    categories, feedback = Counter(), Counter()
    cases = []
    for result, row in zip(results, rows):
        if not row["completed"]:
            categories["incomplete"] += 1
        if row["diagnosis_match"] is not True:
            categories["diagnosis_" + row["diagnosis_relation"]] += 1
        categories["rejected_actions"] += row["rejected_actions"]
        categories["unknown_requests"] += row["unknown_requests"]
        categories["requests_without_new_information"] += sum(
            h["accepted"] and h.get("new_information") is False
            and (h["action"].get("action") in ("EXAM", "TEST") or h["action"].get("intent") == "question")
            for h in result["history"])
        for h in result["history"]:
            if not h["accepted"]:
                feedback[h.get("feedback", "Unknown validation error")] += 1
        cases.append({"case_index": row["case_index"], "prediction": result["diagnosis"],
                      "reference": result["gold_diagnosis"], "assessment": result["diagnosis_assessment"],
                      "completed": row["completed"], "turns": row["turns"], "error": row["error"]})
    # Only this aggregate is sent to the optimizer; case labels stay in the local audit.
    return {"summary": summary, "workflow_failures": dict(categories),
            "validation_feedback": dict(feedback), "cases": cases}


def training_feedback(results, cases):
    """Detailed failures for TRAIN-only reflection; never used as doctor instructions."""
    feedback = []
    for result in results:
        case = cases[result["case_index"]]
        if case.gold != result["gold_diagnosis"]:
            raise ValueError("training feedback case/reference mismatch")
        if result["diagnosis_assessment"]["match"] is True:
            continue
        kinds = ("SAY", "EXAM", "TEST") if result["round"] == "final" else ("SAY", "EXAM")
        observed = [e for e in result["evidence"] if e["status"] == "OBSERVED"]
        aliases = {"opening.demographics": "Demographics", "opening.chief_complaint": "Symptoms.Primary_Symptom"}
        acquired = {(aliases.get(e["source"], e["source"]), json.dumps(e["value"], sort_keys=True, ensure_ascii=False)) for e in observed}
        missed = [{"source": p, "value": v} for k in kinds for p, v in case.sources[k].items()
                  if (p, json.dumps(v, sort_keys=True, ensure_ascii=False)) not in acquired]
        feedback.append({"prediction": result["diagnosis"], "reference": case.gold,
            "assessment": result["diagnosis_assessment"],
            "observed_facts": [{"id": e["id"], "source": e["source"], "value": str(e["value"])[:600]} for e in observed[:16]],
            "unacquired_recorded_facts": [{**e, "value": str(e["value"])[:600]} for e in missed[:12]],
            "trajectory": [{"action": h["action"], "accepted": h["accepted"],
                            "feedback": h.get("feedback"), "new_information": h.get("new_information")}
                           for h in result["history"][-10:]]})
    return feedback


def candidate_wins(candidate, incumbent):
    # shortcut: dev scores select prompts in-sample; only a later frozen test run measures generalization.
    for key in ("attempted_coverage", "completion_rate_all_requested", "diagnosis_assessment_coverage_all_requested"):
        if candidate[key] < incumbent[key]:
            return False
    if candidate["unobserved_basis_rejections"] > incumbent["unobserved_basis_rejections"]:
        return False
    def rank(summary):
        return (summary["diagnosis_matches"], summary["completion_rate_all_requested"],
                -summary["unobserved_basis_rejections"], -summary["rejected_action_rate"],
                -summary["repeat_requests"], -summary["mean_turns"])
    return rank(candidate) > rank(incumbent)


def validate_candidate(reply, results):
    instructions, reason = reply.get("instructions"), reply.get("reason")
    if not isinstance(instructions, str) or not 1 <= len(instructions.strip()) <= 4000 or not isinstance(reason, str) or not reason.strip():
        raise ValueError("optimizer must return nonempty instructions (<=4000 chars) and reason")
    from clinic.runner import normalize_diagnosis
    text = " " + normalize_diagnosis(instructions) + " "
    for result in results:
        for label in (result.get("diagnosis"), result.get("gold_diagnosis")):
            if label and " " + normalize_diagnosis(label) + " " in text:
                raise ValueError("candidate contains a development diagnosis label; rejected")
    return instructions


def optimize(args):
    split = load_split(args.split_file, args.dataset)
    if args.limit > len(split["dev"]):
        raise ValueError("--limit exceeds development split size")
    if args.candidates > 3 or args.max_turns > 50 or args.max_seconds > 1200:
        raise ValueError("use <=3 candidates, <=50 turns and <=1200 seconds per case")
    instructions = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else ""
    if len(instructions) > 8000:
        raise ValueError("initial additional prompt must be <=8000 characters")
    output = args.output or ROOT / "outputs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-optimize")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    client = ChatClient(args.base_url, args.model, args.max_tokens, args.reasoning)
    report = {"dev_case_indices": split["dev"][:args.limit], "test_used": False,
              "selection_scope": "In-sample development selection; not held-out performance",
              "simulator_cache": str((output / "simulator-cache").resolve()),
              "timing_note": "Shared simulator cache affects elapsed time and tokens; compare diagnostic/behavior metrics, not speed.",
              "weights_updated": False, "base_prompt_modified": False, "runs": []}
    save_json(output / "optimization.json", report)
    best = None
    best_instructions = instructions
    analysis, best_results = None, None
    try:
        for index in range(args.candidates + 1):
            label = "baseline" if index == 0 else f"candidate-{index}"
            prompt_path = output / f"{label}-prompt.txt"
            if index:
                generation = client.complete(OPTIMIZER_PROMPT,
                    {"base_rules": DOCTOR_PROMPT, "current_guidance": best_instructions, "workflow_failures": analysis["workflow_failures"],
                     "validation_feedback": analysis["validation_feedback"],
                     "summary": {k: analysis["summary"][k] for k in
                         ("diagnosis_matches", "diagnosis_needs_review", "completion_rate_all_requested", "mean_turns")}},
                    "prompt_optimizer", time.monotonic() + 180)
                save_json(output / f"{label}-generation.json", generation)
                instructions = validate_candidate(generation, best_results)
            prompt_path.write_text(instructions, encoding="utf-8")
            folder = output / label
            command = [sys.executable, str(ROOT / "run.py"), "eval", "--split-file", str(args.split_file.resolve()),
                       "--split", "dev", "--limit", str(args.limit), "--dataset", str(args.dataset.resolve()),
                       "--output", str(folder.resolve()), "--prompt-file", str(prompt_path.resolve()),
                       "--simulator-cache", str((output / "simulator-cache").resolve())]
            for option in ("engine", "base_url", "model", "max_turns", "max_seconds", "max_tokens", "reasoning", "judge_mode"):
                command.extend(["--" + option.replace("_", "-"), str(getattr(args, option))])
            if args.verbose:
                command.append("--verbose")
            print(f"Evaluating {label} on {args.limit} fixed dev case(s)", flush=True)
            completed = subprocess.run(command, cwd=ROOT, timeout=args.limit * (args.max_seconds + 360) + 60)
            manifest, results = read_run(folder)
            current = failure_analysis(results, manifest)
            save_json(output / f"{label}-failure-analysis.json", current)
            summary = current["summary"]
            eligible = completed.returncode == 0 and summary["attempted_coverage"] == 1
            report["runs"].append({"name": label, "eligible": eligible, "summary": summary,
                                   "prompt_file": str(prompt_path.resolve())})
            if index == 0 and (summary["attempted_coverage"] != 1 or any(
                    (r.get("error") or "").startswith("Model endpoint unavailable") for r in results)):
                raise ValueError("baseline has missing cases or an unavailable endpoint; fix execution before prompt search")
            if best is None or (eligible and candidate_wins(summary, best["summary"])):
                best, analysis, best_results = report["runs"][-1], current, results
                best_instructions = instructions
            report["selected"] = best["name"]
            report["selection_rule"] = "No lower coverage/completion or more unobserved citations; rank diagnosis matches, completion, violations, rejections, repeats, turns. Ties keep incumbent."
            (output / "best-prompt.txt").write_text(best_instructions, encoding="utf-8")
            save_json(output / "optimization.json", report)
    except (ModelError, ValueError, OSError, subprocess.TimeoutExpired) as exc:
        report["error"] = str(exc)
        raise
    finally:
        report["seconds"] = time.monotonic() - started
        report["optimizer_calls"] = client.calls
        save_json(output / "optimization.json", report)
    return output, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-file", type=Path, default=ROOT / "eval/split.json")
    parser.add_argument("--dataset", type=Path, default=DATA)
    parser.add_argument("--limit", type=positive, default=2, help="number of fixed development cases")
    parser.add_argument("--candidates", type=positive, default=1, help="number of proposed prompts (maximum 3)")
    parser.add_argument("--engine", choices=["plain", "langgraph"], default="langgraph")
    parser.add_argument("--base-url", default=os.getenv("CLINIC_BASE_URL", "http://localhost:11434/v1"))
    parser.add_argument("--model", default=os.getenv("CLINIC_MODEL", "gpt-oss:20b"))
    parser.add_argument("--max-turns", type=positive, default=10)
    parser.add_argument("--max-seconds", type=positive, default=180)
    parser.add_argument("--max-tokens", type=positive, default=4096)
    parser.add_argument("--reasoning", choices=["low", "medium", "high"], default="low")
    parser.add_argument("--judge-mode", choices=["aliases", "llm"], default="llm")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    try:
        output, report = optimize(args)
    except (ModelError, ValueError, OSError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
        parser.error(str(exc))
    print(f"Selected: {report['selected']} (dev only; base prompt unchanged)")
    print(f"Artifacts: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
