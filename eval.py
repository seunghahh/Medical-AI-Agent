#!/usr/bin/env python3
"""Create fixed practice splits, regenerate reports, or compare paired runs."""
import argparse
import csv
import json
from pathlib import Path
from clinic.data import DATA
from clinic.evaluation import compare_runs, make_split, read_run, write_report
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
    compare = commands.add_parser("compare", help="compare the same cases under matching model/budget settings")
    compare.add_argument("before", type=Path)
    compare.add_argument("after", type=Path)
    compare.add_argument("--output", type=Path, required=True)
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
        else:
            paired, comparison = compare_runs(args.before, args.after)
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
