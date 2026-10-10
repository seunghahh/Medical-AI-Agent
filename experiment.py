#!/usr/bin/env python3
"""Frozen baseline versus working-memory workflow: development, then unused test cases."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from clinic.data import ROOT, DATA, provenance
from clinic.evaluation import compare_runs, load_split, read_run, summarize
from optimize import candidate_wins
from run import positive


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def fingerprint(paths):
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-prompt", type=Path, default=ROOT/"outputs/prompt-search-dev/best-prompt.txt")
    parser.add_argument("--previous-evaluation", type=Path, default=ROOT/"outputs/20261009-heldout-comparison/evaluation-plan.json")
    parser.add_argument("--split-file", type=Path, default=ROOT/"eval/split.json")
    parser.add_argument("--dev-limit", type=positive, default=20)
    parser.add_argument("--max-turns", type=positive, default=10)
    parser.add_argument("--max-seconds", type=positive, default=180)
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--model", default="gpt-oss:20b")
    args = parser.parse_args()
    try:
        split = load_split(args.split_file, DATA)
        previous = json.loads(args.previous_evaluation.read_text(encoding="utf-8"))
        if previous["provenance"] != split["provenance"]:
            raise ValueError("previous evaluation dataset mismatch")
        used = previous["test_case_indices"]
        if not set(used).issubset(split["test"]):
            raise ValueError("previous test cases are not in the fixed split")
        remaining = [i for i in split["test"] if i not in used]
        if not remaining or args.dev_limit > len(split["dev"]) or args.max_turns > 50 or args.max_seconds > 1200:
            raise ValueError("invalid development count, remaining test cases, or case budgets")
        instructions = args.candidate_prompt.read_text(encoding="utf-8")
        if not 1 <= len(instructions) <= 8000:
            raise ValueError("candidate guidance must contain 1..8000 characters")
        args.output.mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    output = args.output.resolve()
    frozen = output/"frozen-prompts"
    frozen.mkdir()
    (frozen/"baseline.txt").write_text("", encoding="utf-8")
    (frozen/"candidate.txt").write_text(instructions, encoding="utf-8")
    evaluated_split = {**split, "test": remaining}
    save(output/"remaining-split.json", evaluated_split)
    sources = [ROOT/n for n in ("run.py", "eval.py", "optimize.py", "experiment.py")] + sorted((ROOT/"clinic").glob("*.py"))
    hashes = fingerprint(sources)
    for path in sources:
        dest = output/"frozen-code"/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    plan = {"frozen_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
            "provenance": provenance(), "code_sha256": hashes, "dev_case_indices": split["dev"][:args.dev_limit],
            "previous_test_cases_excluded": used, "test_case_indices": remaining,
            "prompt_sha256": {n: hashlib.sha256((frozen/(n+".txt")).read_bytes()).hexdigest() for n in ("baseline", "candidate")},
            "arms": {"baseline": {"working_memory": False, "additional_guidance": False},
                     "candidate": {"working_memory": True, "additional_guidance": True}},
            "settings": {"model": args.model, "engine": "langgraph", "max_turns": args.max_turns,
                         "max_seconds": args.max_seconds, "reasoning": "low", "max_tokens": 4096},
            "candidate_regenerated": False, "test_used_for_prompt_search": False,
            "timing_note": "Shared matching cache changes simulator time and tokens; not a controlled speed experiment.",
            "runs": {}}
    save(output/"evaluation-plan.json", plan)
    try:
        for phase, count in (("dev", args.dev_limit), ("test", len(remaining))):
            for arm in ("baseline", "candidate"):
                if fingerprint(sources) != hashes or provenance() != plan["provenance"]:
                    raise ValueError("code/dataset changed after freeze")
                if any(hashlib.sha256((frozen/(n+".txt")).read_bytes()).hexdigest() != h for n,h in plan["prompt_sha256"].items()):
                    raise ValueError("prompt changed after freeze")
                folder = output/phase/arm
                plan["active_stage"] = phase+"-"+arm
                save(output/"evaluation-plan.json", plan)
                command = [sys.executable, "-u", str(ROOT/"run.py"), "eval", "--engine", "langgraph",
                    "--split-file", str(output/"remaining-split.json"), "--split", phase, "--limit", str(count),
                    "--base-url", args.base_url, "--model", args.model, "--max-turns", str(args.max_turns),
                    "--max-seconds", str(args.max_seconds), "--reasoning", "low", "--max-tokens", "4096",
                    "--judge-mode", "llm", "--simulator-cache", str(output/phase/"shared-simulator-cache"),
                    "--prompt-file", str(frozen/(arm+".txt")), "--output", str(folder), "--verbose"]
                if arm == "candidate":
                    command.append("--working-memory")
                print(f"START {phase}-{arm}: {count} cases", flush=True)
                with (output/(phase+"-"+arm+".log")).open("w", encoding="utf-8") as log:
                    process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                             timeout=count*(args.max_seconds+360)+60)
                manifest, results = read_run(folder)
                _, summary = summarize(results, manifest)
                plan["runs"][phase+"-"+arm] = {"exit_code": process.returncode, "summary": summary}
                save(output/"evaluation-plan.json", plan)
                if summary["attempted_cases"] != count:
                    raise ValueError("incomplete cohort; see run log before continuing")
                print(f"DONE {phase}-{arm}: completed={summary['completed_cases']}/{count}, diagnosis_matches={summary['diagnosis_matches']}/{count}", flush=True)
            compare_command = [sys.executable, str(ROOT/"eval.py"), "compare", str(output/phase/"baseline"),
                               str(output/phase/"candidate"), "--output", str(output/phase/"comparison")]
            subprocess.run(compare_command, cwd=ROOT, check=True, capture_output=True, text=True, timeout=30)
            if phase == "dev":
                baseline, candidate = [plan["runs"]["dev-"+n]["summary"] for n in ("baseline", "candidate")]
                plan["dev_selected"] = "candidate" if candidate_wins(candidate, baseline) else "baseline"
                plan["selection_note"] = "Dev selects the preferred arm. Both predeclared frozen arms are evaluated on unused test cases, including a dev-rejected candidate; no regeneration."
                plan["selection_frozen_utc"] = datetime.now(timezone.utc).isoformat()
                save(output/"evaluation-plan.json", plan)
                print("DEV SELECTION", plan["dev_selected"], flush=True)
        if fingerprint(sources) != hashes:
            raise ValueError("code changed during evaluation")
        plan.update(status="completed", active_stage=None)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        plan.update(status="failed", error=str(exc))
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        plan["updated_utc"] = datetime.now(timezone.utc).isoformat()
        save(output/"evaluation-plan.json", plan)
    print(f"Artifacts: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
