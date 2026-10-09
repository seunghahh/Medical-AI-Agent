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
from clinic.agent import DemoDoctor, DemoSimulator, Doctor, Simulator
from clinic.client import ChatClient, ModelError
from clinic.data import ROOT, DATA, load_cases, provenance
from clinic.runner import run_case
from clinic.evaluation import load_split, write_report


def positive(value):
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["demo", "run", "check", "eval"])
    parser.add_argument("--engine", choices=["plain", "langgraph"], default="plain")
    parser.add_argument("--round", choices=["preliminary", "final"], default="preliminary")
    parser.add_argument("--base-url", default=os.getenv("CLINIC_BASE_URL", "http://localhost:11434/v1"))
    parser.add_argument("--model", default=os.getenv("CLINIC_MODEL", "gpt-oss:20b"))
    parser.add_argument("--simulator-base-url", default=None)
    parser.add_argument("--simulator-model", default=None)
    parser.add_argument("--max-tokens", type=positive, default=4096)
    parser.add_argument("--reasoning", choices=["low", "medium", "high"], default="low")
    parser.add_argument("--max-turns", type=positive, default=50)
    parser.add_argument("--max-seconds", type=positive, default=1200)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=positive, default=1)
    parser.add_argument("--dataset", type=Path, default=DATA)
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true", help="show each action, acquired response and rejection live")
    args = parser.parse_args()
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
        if args.limit > len(split[args.split]):
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
        "max_turns": args.max_turns, "max_seconds": args.max_seconds,
        "reasoning_effort": args.reasoning, "max_tokens": args.max_tokens,
        "case_indices": [case.index for case in selected],
        "engine": args.engine, "official_submission_ready": False,
        "split_name": args.split if split else None, "fixed_split": split,
        "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("run.py", "clinic/agent.py", "clinic/client.py", "clinic/runner.py", "clinic/langgraph_runner.py", "clinic/evaluation.py")}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    results = []
    with (output / "results.jsonl").open("w") as stream:
        for case in selected:
            if args.command == "eval":
                # Local evaluation treats each case as an independent session.
                client = ChatClient(args.base_url, args.model, args.max_tokens, args.reasoning)
                sim_client = client if not (args.simulator_base_url or args.simulator_model) else ChatClient(
                    args.simulator_base_url or args.base_url, args.simulator_model or args.model,
                    args.max_tokens, args.reasoning)
            result = run_case(case, DemoDoctor() if demo else Doctor(client),
                DemoSimulator() if demo else Simulator(sim_client), args.round,
                args.max_turns, args.max_seconds, demo, None if demo else client,
                engine=args.engine, verbose=args.verbose)
            if not demo and sim_client is not client:
                result["simulator_calls"] = sim_client.calls[:]
                sim_client.calls.clear()
            results.append(result)
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            if result["soap"]:
                (output / f"case-{case.index}-soap.json").write_text(json.dumps(result["soap"], indent=2, ensure_ascii=False))
            print(f"case={case.index} completed={result['completed']} turns={result['turns']} diagnosis={result['diagnosis']}")
            if result["error"]:
                print(result["error"], file=sys.stderr)
                break  # Do not repeat endpoint/budget errors across the cohort.
    write_report(output, results, manifest)
    print(f"Artifacts: {output.resolve()}")
    return 0 if len(results) == args.limit and all(r["completed"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
