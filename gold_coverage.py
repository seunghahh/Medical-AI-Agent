#!/usr/bin/env python3
"""Which findings of the reference record did the agent's consultation actually go looking for?

A local development tool (not part of the submission). For each result folder written by develop.py it asks a model
to compare the consultation log with the S and O sections of samples/<case>/model_record.md and to mark every
documented finding as asked/examined or missed, so a change can be judged per case instead of through the
leaderboard's noisy total.

  python gold_coverage.py results/C01-... results/C03-... [--gold samples/C01/model_record.md]

The model comes from the same NOVA_LOCAL_DOCTOR_URL / _MODEL / _KEY variables that develop.py uses; set them yourself,
in your own terminal. The reference record is a worked example, not the official answer key, and the judge is an LLM,
so treat the numbers as a guide to what is missing rather than a score.
"""
import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

JUDGE_PROMPT = """당신은 진료 기록 감수자입니다. 참고 초진 기록(모범 예시)의 S(문진)와 O(진찰) 부분과, 에이전트의 진료 로그를 비교합니다.
참고 기록이 적은 서로 다른 임상 소견을 하나씩 뽑으세요: 문진 항목(현병력, 과거력, 약물, 가족력, 사회력, 계통 문진)과 진찰 소견.
각 항목마다, 에이전트가 그것을 알아낼 수 있는 질문을 했거나 진찰 동작을 요청했는지 판정합니다.
로그에 맞는 질문이나 진찰이 있어야 covered=true입니다. 에이전트의 기록에 우연히 같은 내용이 있어도 안 물었으면 false입니다.
검사(혈액·영상) 결과와 평가·계획(A/P)은 제외합니다. 항목은 15자 안팎의 짧은 한국어로 씁니다.
JSON 하나만: {"items":[{"item":"항목","kind":"history 또는 exam","covered":true 또는 false,"turn":로그의 턴 번호 또는 null}]}"""


def build_log(transcript):
    """The consultation as numbered lines: questions with answers and exam requests with results (no final record)."""
    lines = []
    for event in transcript.get('events', []):
        action, response = event.get('action'), event.get('response') or {}
        if action == 'SAY':
            lines.append(f"T{event.get('turn')} 질문: {event.get('content')} -> 답: {response.get('content', '')}")
        elif action == 'EXAM':
            lines.append(f"T{event.get('turn')} 진찰: {event.get('content')} -> 결과: {response.get('result') or '(응답 없음)'}")
    return '\n'.join(lines)


def gold_sections(markdown):
    """The S and O sections of a reference record (the part that records what a doctor finds out)."""
    parts = re.split(r'\n(?=## )', markdown)
    return '\n\n'.join(p.strip() for p in parts if p.startswith(('## [S]', '## [O]')) and '[O] 검사' not in p[:12])


def parse_items(text):
    decoder = json.JSONDecoder()
    for match in re.finditer(r'\{', text or ''):
        try:
            value, _ = decoder.raw_decode(text, match.start())
        except ValueError:
            continue
        if isinstance(value, dict) and isinstance(value.get('items'), list):
            return [i for i in value['items'] if isinstance(i, dict) and str(i.get('item', '')).strip()]
    return []


def summarize(items):
    result = {}
    for kind in ('history', 'exam'):
        group = [i for i in items if i.get('kind') == kind]
        covered = [i for i in group if i.get('covered') is True]
        result[kind] = {'total': len(group), 'covered': len(covered),
                        'share': round(len(covered) / len(group), 3) if group else None,
                        'missed': [i['item'] for i in group if i.get('covered') is not True]}
    total = sum(r['total'] for r in result.values())
    result['all'] = {'total': total, 'covered': sum(r['covered'] for r in result.values()),
                     'share': round(sum(r['covered'] for r in result.values()) / total, 3) if total else None}
    return result


def judge(endpoint, gold, log):
    """Ask the model; two tries because a long list occasionally comes back malformed."""
    messages = [{'role': 'system', 'content': JUDGE_PROMPT},
                {'role': 'user', 'content': f'참고 초진 기록(S, O)\n{gold}\n\n에이전트의 진료 로그\n{log}'}]
    for _ in range(2):
        text = endpoint.complete(messages, max_tokens=6000)[0]
        items = parse_items(text)
        if items:
            return items
    return []


def report(folder, endpoint, gold_path=None):
    folder = Path(folder)
    transcript = json.loads((folder / 'transcript.json').read_text(encoding='utf-8'))
    case_id = (transcript.get('summary') or {}).get('case_id') or folder.name.split('-')[0]
    path = Path(gold_path) if gold_path else HERE / 'samples' / case_id / 'model_record.md'
    gold = gold_sections(path.read_text(encoding='utf-8'))
    items = judge(endpoint, gold, build_log(transcript))
    summary = summarize(items)
    (folder / 'gold_coverage.json').write_text(json.dumps({'case': case_id, 'summary': summary, 'items': items},
                                                         ensure_ascii=False, indent=2), encoding='utf-8')
    return case_id, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('folders', nargs='+', help='result folders written by develop.py')
    parser.add_argument('--gold', help='reference record to use instead of samples/<case>/model_record.md (one folder only)')
    args = parser.parse_args()
    if args.gold and len(args.folders) != 1:
        parser.error('--gold works with a single result folder')
    from nova_local import Endpoint
    endpoint = Endpoint.from_env('NOVA_LOCAL_DOCTOR')
    shares = []
    for folder in args.folders:
        try:
            case_id, summary = report(folder, endpoint, args.gold)
        except Exception as exc:  # keep going: one broken folder should not hide the others
            print(f'{folder}: {type(exc).__name__}: {exc}', file=sys.stderr)
            continue
        print(f"\n{case_id}  문진 {summary['history']['covered']}/{summary['history']['total']}  "
              f"진찰 {summary['exam']['covered']}/{summary['exam']['total']}  전체 {summary['all']['share']}")
        for kind, title in (('history', '놓친 문진'), ('exam', '놓친 진찰')):
            if summary[kind]['missed']:
                print(f"  {title}: " + ', '.join(summary[kind]['missed']))
        if summary['all']['share'] is not None:
            shares.append(summary['all']['share'])
    if len(shares) > 1:
        print(f'\n평균 커버리지 {sum(shares) / len(shares):.2f} ({len(shares)}개 증례)')


if __name__ == '__main__':
    main()
