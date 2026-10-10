#!/usr/bin/env python3
"""Local development entry point. Not an official NOVA submission interface."""
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from clinic.agent import DemoDoctor, DemoSimulator, Doctor, Simulator, resolver_config
from clinic.client import ChatClient, ModelError
from clinic.data import ROOT, DATA, load_cases, provenance
from clinic.runner import run_case
from clinic.evaluation import load_split, write_report
from clinic.live import Journal, serve, open_monitor
from clinic.diagnosis import assess_result, evaluator_config


def positive(value):
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["demo", "run", "check", "eval", "serve"])
    parser.add_argument("--port", type=positive, default=8767, help="localhost monitor port")
    parser.add_argument("--no-browser", action="store_true", help="do not start/open the live browser for run (headless use)")
    parser.add_argument("--engine", choices=["plain", "langgraph"], default="plain")
    parser.add_argument("--round", choices=["preliminary", "final"], default="preliminary")
    parser.add_argument("--base-url", default=os.getenv("CLINIC_BASE_URL", "http://localhost:11434/v1"))
    parser.add_argument("--model", default=os.getenv("CLINIC_MODEL", "gpt-oss:20b"))
    parser.add_argument("--simulator-base-url", default=None)
    parser.add_argument("--simulator-model", default=None)
    parser.add_argument("--simulator-cache", type=Path, help="shared validated record selections for reproducible paired runs")
    parser.add_argument("--working-memory", action="store_true", help="case-local facts, unknowns, grounded hypotheses and next-information notes")
    parser.add_argument("--diagnosis-review", action="store_true", help="compare evidence-linked candidates once before submitting each valid draft diagnosis")
    parser.add_argument("--information-mode", choices=["interactive", "full"], default="interactive",
                        help="full is an offline diagnostic ablation using all records allowed in this round")
    parser.add_argument("--max-tokens", type=positive, default=4096)
    parser.add_argument("--reasoning", choices=["low", "medium", "high"], default="low")
    parser.add_argument("--max-turns", type=positive, default=50)
    parser.add_argument("--max-seconds", type=positive, default=1200)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=positive, default=1)
    parser.add_argument("--dataset", type=Path, default=DATA)
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true", help="show each action, acquired response and rejection live")
    parser.add_argument("--prompt-file", type=Path, help="additional workflow instructions; base action/evidence rules remain")
    parser.add_argument("--judge-mode", choices=["aliases", "llm"], default="aliases",
                        help="post-encounter diagnosis assessment; llm uses the configured model")
    args = parser.parse_args()
    if args.port > 65535:
        parser.error("port must be <=65535")
    if args.command == "serve":
        serve(args.port)
        return 0
    instructions = ""
    if args.prompt_file:
        try:
            instructions = args.prompt_file.read_text(encoding="utf-8")
        except OSError as exc:
            parser.error(str(exc))
        if len(instructions) > 8000:
            parser.error("additional prompt must be at most 8000 characters")
    if args.max_turns > 50 or args.max_seconds > 1200:
        parser.error("practice budgets must be <=50 turns and <=1200 seconds")
    client = ChatClient(args.base_url, args.model, args.max_tokens, args.reasoning)
    if args.command == "check":
        try:
            reply = client.complete('Return exactly {"ok":true} as JSON.', {"test": "connection"}, "probe", time.monotonic()+180)
            if reply != {"ok": True}:
                raise ModelError("Unexpected connection-test JSON")
        except ModelError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(f"Connected: {args.base_url} / {args.model}")
        return 0
    if args.engine == "langgraph":
        try:
            from clinic.langgraph_runner import build_graph
        except ImportError:
            parser.error("LangGraph engine requires requirements-langgraph.txt in the active environment")
    cases = load_cases(args.dataset)
    split = None
    if args.command == "eval":
        if not args.split_file or args.start_index != 0:
            parser.error("eval requires --split-file and uses its case indices (no --start-index)")
        try:
            split = load_split(args.split_file, args.dataset)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(str(exc))
        if args.split not in split or args.limit > len(split[args.split]):
            parser.error("--limit exceeds the selected split size")
        by_index = {case.index: case for case in cases}
        selected = [by_index[i] for i in split[args.split][:args.limit]]
    else:
        if args.split_file:
            parser.error("--split-file is only available with eval")
        if args.start_index < 0 or args.start_index+args.limit > len(cases):
            parser.error(f"requested case range must be inside 0..{len(cases)-1}")
        selected = cases[args.start_index:args.start_index+args.limit]
    demo = args.command == "demo"
    if demo and args.diagnosis_review:
        parser.error("diagnosis review requires a model run")
    if demo and (args.start_index != 0 or args.limit != 1 or args.dataset.resolve() != DATA.resolve()):
        parser.error("scripted demo is restricted to bundled MedQA case 0")
    sim_client = client
    if args.simulator_base_url or args.simulator_model:
        sim_client = ChatClient(args.simulator_base_url or args.base_url,
            args.simulator_model or args.model, args.max_tokens, args.reasoning)
    output = args.output or ROOT / "outputs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + args.command)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "provenance": provenance(args.dataset),
        "mode": "scripted_demo" if demo else "model", "doctor_model": None if demo else args.model,
        "simulator_model": None if demo else sim_client.model, "round": args.round,
        "simulator_resolver": None if demo else resolver_config(sim_client),
        "simulator_cache": str(args.simulator_cache.resolve()) if args.simulator_cache else None,
        "max_turns": args.max_turns, "max_seconds": args.max_seconds,
        "reasoning_effort": args.reasoning, "max_tokens": args.max_tokens,
        "case_indices": [case.index for case in selected],
        "engine": args.engine, "official_submission_ready": False,
        "working_memory_enabled": args.working_memory, "information_mode": args.information_mode,
        "diagnosis_review_enabled": args.diagnosis_review,
        "split_name": args.split if split else None, "fixed_split": split,
        "additional_instructions": instructions,
        "prompt_sha256": hashlib.sha256(instructions.encode()).hexdigest(),
        "diagnosis_evaluator": evaluator_config(args.judge_mode, args.model, args.max_tokens, args.reasoning),
        "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("run.py", "clinic/agent.py", "clinic/client.py", "clinic/runner.py", "clinic/langgraph_runner.py", "clinic/evaluation.py", "clinic/diagnosis.py", "clinic/matching.py", "clinic/memory.py", "clinic/live.py")}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    results = []
    print(f"Live monitor: http://127.0.0.1:{args.port}/", flush=True)
    with Journal(output, {"model": None if demo else args.model, "mode": manifest["mode"], "engine": args.engine, "case_indices": manifest["case_indices"], "review_enabled": args.diagnosis_review, "memory_enabled": args.working_memory, "judge_mode": args.judge_mode}) as live, (output / "results.jsonl").open("w") as stream:
        if args.command == "run" and not args.no_browser:
            try:
                open_monitor(args.port)
            except OSError as exc:
                print(f"Live monitor unavailable (agent continues): {exc}", file=sys.stderr)
        for case in selected:
            if args.command == "eval":
                # Local evaluation treats each case as an independent session.
                client = ChatClient(args.base_url, args.model, args.max_tokens, args.reasoning)
                sim_client = client if not (args.simulator_base_url or args.simulator_model) else ChatClient(
                    args.simulator_base_url or args.base_url, args.simulator_model or args.model,
                    args.max_tokens, args.reasoning)
            simulator = DemoSimulator() if demo else Simulator(sim_client, args.simulator_cache)
            result = run_case(case, DemoDoctor() if demo else Doctor(client, instructions, args.diagnosis_review, observer=live.emit),
                simulator, args.round,
                args.max_turns, args.max_seconds, demo, None if demo else client,
                engine=args.engine, verbose=args.verbose, working_memory=args.working_memory,
                information_mode=args.information_mode, observer=live.emit)
            if not demo and sim_client is not client:
                result["simulator_calls"] = sim_client.calls[:]
                sim_client.calls.clear()
            result["simulator_resolutions"] = [] if demo else simulator.resolutions
            judge = ChatClient(args.base_url, args.model, args.max_tokens, args.reasoning) if args.judge_mode == "llm" and not demo else None
            live.emit("evaluation_start", case_index=case.index, method=args.judge_mode)
            result["diagnosis_assessment"] = assess_result(result, manifest["provenance"], judge, case)
            result["diagnosis_judge_calls"] = judge.calls if judge else []
            live.emit("evaluation", case_index=case.index, diagnosis=result["diagnosis"], gold=result["gold_diagnosis"], assessment=result["diagnosis_assessment"])
            results.append(result)
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            if result["soap"]:
                (output / f"case-{case.index}-soap.json").write_text(json.dumps(result["soap"], indent=2, ensure_ascii=False))
            print(f"case={case.index} completed={result['completed']} turns={result['turns']} diagnosis={result['diagnosis']}")
            if result["error"]:
                print(result["error"], file=sys.stderr)
                if result["error"].startswith(("Model endpoint unavailable", "Simulator cache", "Simulator returned invalid", "Invalid simulator rejection")):
                    break  # Infrastructure errors stop the cohort; case-local failures remain scored.
        write_report(output, results, manifest)
    print(f"Artifacts: {output.resolve()}")
    return 0 if len(results) == args.limit and all(r["completed"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
