#!/usr/bin/env python3
"""Create fixed practice splits, regenerate reports, or compare paired runs."""
import argparse
import csv
import json
import shutil
from pathlib import Path
from clinic.data import DATA, load_cases, provenance
from clinic.evaluation import compare_runs, make_split, read_run, write_report
from clinic.client import ChatClient
from clinic.diagnosis import assess_result, evaluator_config
from run import positive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    split = commands.add_parser("split", help="create disjoint public practice dev/test case lists")
    split.add_argument("--dataset", type=Path, default=DATA)
    split.add_argument("--dev", type=positive, default=20)
    split.add_argument("--test", type=positive, default=30)
    split.add_argument("--seed", type=int, default=20261009)
    split.add_argument("--output", type=Path, default=Path("eval/split.json"))
    report = commands.add_parser("report", help="regenerate cases.csv and summary.json from saved results")
    report.add_argument("run", type=Path)
    assess = commands.add_parser("assess", help="assess saved diagnosis labels in a new run folder; originals remain intact")
    assess.add_argument("run", type=Path)
    assess.add_argument("--output", type=Path, required=True)
    assess.add_argument("--judge-mode", choices=["aliases", "llm"], default="aliases")
    assess.add_argument("--base-url", default="http://localhost:11434/v1")
    assess.add_argument("--model", default="gpt-oss:20b")
    assess.add_argument("--dataset", type=Path, default=DATA)
    compare = commands.add_parser("compare", help="compare the same cases under matching model/budget settings")
    compare.add_argument("before", type=Path)
    compare.add_argument("after", type=Path)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--allow-resolver-change", action="store_true",
                         help="explicitly compare a changed simulator; result measures the combined workflow")
    args = parser.parse_args()
    try:
        if args.command == "split":
            data = make_split(args.dataset, args.dev, args.test, args.seed)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(data, indent=2, ensure_ascii=False))
            print(f"Fixed split: {args.output.resolve()} (dev={args.dev}, test={args.test})")
        elif args.command == "report":
            manifest, results = read_run(args.run)
            print(json.dumps(write_report(args.run, results, manifest), indent=2, ensure_ascii=False))
        elif args.command == "assess":
            manifest, results = read_run(args.run)
            if manifest["provenance"] != provenance(args.dataset):
                raise ValueError("assessment dataset provenance mismatch")
            cases = {c.index: c for c in load_cases(args.dataset)}
            args.output.mkdir(parents=True, exist_ok=False)
            manifest["assessment_source_run"] = str(args.run.resolve())
            manifest["diagnosis_evaluator"] = evaluator_config(args.judge_mode, args.model)
            (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
            with (args.output / "results.jsonl").open("w", encoding="utf-8") as stream:
                for result in results:
                    judge = ChatClient(args.base_url, args.model) if args.judge_mode == "llm" else None
                    result["diagnosis_assessment"] = assess_result(result, manifest["provenance"], judge, cases[result["case_index"]])
                    result["diagnosis_judge_calls"] = judge.calls if judge else []
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                    stream.flush()
            for soap in args.run.glob("case-*-soap.json"):
                shutil.copyfile(soap, args.output / soap.name)
            print(json.dumps(write_report(args.output, results, manifest), indent=2, ensure_ascii=False))
        else:
            paired, comparison = compare_runs(args.before, args.after, allow_resolver_change=args.allow_resolver_change)
            args.output.mkdir(parents=True, exist_ok=False)
            (args.output / "comparison.json").write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
            with (args.output / "paired.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
                writer.writeheader()
                writer.writerows(paired)
            print(json.dumps(comparison, indent=2, ensure_ascii=False))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
