#!/usr/bin/env python3
"""Measure guideline retrieval on labelled queries, so a retrieval change can be judged by numbers.

Cases file (JSON list):
  [{"id": "C10", "query": "체중이 줄고 손이 떨려요", "expected": ["graves", "갑상선기능항진증"],
    "extra": ["hyperthyroidism"], "thought": "Weight loss and tremor suggest hyperthyroidism (Graves disease) ..."}]
`expected` entries match case-insensitively against a hit's title or path; any one counts.
`extra` and `thought` are optional stand-ins for the model's search terms and its Thought-RAG reasoning
(save them from a real run's retrievals.json). 'thought' is the variant that searches with the reasoning too.

  python eval_retrieval.py cases.json [--k 3] [--guidelines DIR]
Prints hit@k and MRR for the old OR-only search, the current search and the Thought-RAG search. No model calls.
"""
import argparse
import json
import sys
from pathlib import Path

from rag import GuidelineIndex, guideline_root

VARIANTS = {
    'or-only (old)': dict(strict=False, expand=False, use_extra=False),
    'current': dict(strict=True, expand=True, use_extra=True),
    'thought': dict(strict=True, expand=True, use_extra=True, use_thought=True),
}


def rank_of(hits, expected):
    wanted = [e.lower() for e in expected]
    for rank, hit in enumerate(hits, 1):
        text = (hit['title'] + ' ' + hit['path']).lower()
        if any(w in text for w in wanted):
            return rank
    return None


def evaluate(index, cases, k):
    report = {}
    for name, options in VARIANTS.items():
        ranks = []
        for case in cases:
            extra = case.get('extra', ()) if options['use_extra'] else ()
            thought = case.get('thought', '') if options.get('use_thought') else ''
            hits = index.search(case['query'], k, extra=extra, strict=options['strict'], expand=options['expand'],
                                thought=thought)
            ranks.append(rank_of(hits, case['expected']))
        found = [r for r in ranks if r]
        report[name] = {'cases': len(cases), f'hit@{k}': round(len(found) / len(cases), 3),
                        'mrr': round(sum(1 / r for r in found) / len(cases), 3),
                        'missed': [c['id'] for c, r in zip(cases, ranks) if not r]}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('cases')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--guidelines')
    args = parser.parse_args()
    cases = json.loads(Path(args.cases).read_text(encoding='utf-8'))
    if not cases:
        sys.exit('no cases')
    index = GuidelineIndex(Path(args.guidelines) if args.guidelines else guideline_root())
    print(json.dumps(evaluate(index, cases, args.k), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
