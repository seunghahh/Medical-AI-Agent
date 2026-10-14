#!/usr/bin/env python3
"""Which findings of the reference record did the agent's consultation actually go looking for?

A local development tool (not part of the submission). For each result folder written by develop.py it checks the
consultation log against the findings that samples/<case>/model_record.md documents in S and O, and says which ones
the agent asked about or examined and which it missed, so a change can be judged per case instead of through the
leaderboard's noisy total.

  python gold_coverage.py results/C01-... results/C03-...  [--refresh] [--gold samples/C01/model_record.md]

The list of findings is extracted from the reference record ONCE per case (and cached in .cache/), so the
denominator is the same for every run you compare; only the "was it asked?" judgement is made per run. The model
comes from the same NOVA_LOCAL_DOCTOR_URL / _MODEL / _KEY variables that develop.py uses, set in your own terminal.
The reference record is a worked example, not the official answer key, and both steps are done by an LLM, so treat
the numbers as a guide to what is missing rather than a score.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / '.cache'

EXTRACT_PROMPT = """당신은 진료 기록 감수자입니다. 참고 초진 기록(모범 예시)의 S(문진)와 O(진찰) 부분에서, 서로 다른 임상 소견을 하나씩 뽑으세요.
문진 항목(현병력, 과거력, 약물, 가족력, 사회력, 계통 문진)은 kind를 history, 진찰 소견은 exam으로 합니다.
같은 질문이나 같은 진찰로 함께 알아낼 수 있는 소견은 하나로 묶습니다. 환자의 첫 진술에 이미 나온 정보(나이, 성별, 주호소)는 제외합니다.
검사(혈액·영상) 결과와 평가·계획(A/P)도 제외합니다. 항목은 15자 안팎의 짧은 한국어로 씁니다.
JSON 하나만: {"items":[{"item":"항목","kind":"history 또는 exam"}]}"""

JUDGE_PROMPT = """당신은 진료 기록 감수자입니다. 번호가 붙은 임상 소견 목록과 에이전트의 진료 로그를 비교합니다.
각 소견마다, 에이전트가 그것을 알아낼 수 있는 질문을 했거나 진찰 동작을 요청했는지 판정합니다.
로그에 맞는 질문이나 진찰이 있어야 covered=true입니다. 에이전트가 우연히 같은 정보를 얻었더라도 묻지 않았으면 false입니다.
turn에는 근거가 된 로그의 턴 번호를 씁니다(false이면 null).
JSON 하나만: {"results":[{"n":번호,"covered":true 또는 false,"turn":번호 또는 null}]}"""


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


def first_json(text, key):
    """The first JSON object in a model answer that has a list under `key` (tolerates chatter around it)."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r'\{', text or ''):
        try:
            value, _ = decoder.raw_decode(text, match.start())
        except ValueError:
            continue
        if isinstance(value, dict) and isinstance(value.get(key), list):
            return value[key]
    return []


def ask(endpoint, system, user, key):
    """Two tries, because a long list occasionally comes back malformed. Temperature 0 where the model allows it."""
    messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
    for _ in range(2):
        found = first_json(endpoint.complete(messages, max_tokens=6000, temperature=0)[0], key)
        if found:
            return found
    return []


def gold_items(endpoint, gold, opening, case_id, refresh=False, cache=CACHE):
    """The findings to look for, extracted once per (case, reference text) and reused so runs stay comparable."""
    cache = Path(cache)
    stamp = hashlib.sha256((gold + opening).encode('utf-8')).hexdigest()[:12]
    path = cache / f'gold_items_{case_id}.json'
    if path.exists() and not refresh:
        saved = json.loads(path.read_text(encoding='utf-8'))
        if saved.get('stamp') == stamp and saved.get('items'):
            return saved['items']
    raw = ask(endpoint, EXTRACT_PROMPT, f'첫 진술(제외할 정보)\n{opening}\n\n참고 초진 기록(S, O)\n{gold}', 'items')
    items = [{'item': str(i['item']).strip(), 'kind': i.get('kind') if i.get('kind') in ('history', 'exam') else 'history'}
             for i in raw if isinstance(i, dict) and str(i.get('item', '')).strip()]
    if items:
        cache.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'stamp': stamp, 'items': items}, ensure_ascii=False, indent=2), encoding='utf-8')
    return items


def judge(endpoint, items, log):
    """Mark each fixed finding as asked/examined or missed. Anything the model leaves out counts as missed."""
    numbered = '\n'.join(f"{n}. [{i['kind']}] {i['item']}" for n, i in enumerate(items, 1))
    answers = {r.get('n'): r for r in ask(endpoint, JUDGE_PROMPT, f'소견 목록\n{numbered}\n\n에이전트의 진료 로그\n{log}', 'results')
               if isinstance(r, dict)}
    return [{**item, 'covered': answers.get(n, {}).get('covered') is True, 'turn': answers.get(n, {}).get('turn')}
            for n, item in enumerate(items, 1)]


def summarize(items):
    result = {}
    for kind in ('history', 'exam'):
        group = [i for i in items if i.get('kind') == kind]
        covered = [i for i in group if i.get('covered') is True]
        result[kind] = {'total': len(group), 'covered': len(covered),
                        'share': round(len(covered) / len(group), 3) if group else None,
                        'missed': [i['item'] for i in group if i.get('covered') is not True]}
    total = sum(r['total'] for r in result.values())
    done = sum(r['covered'] for r in result.values())
    result['all'] = {'total': total, 'covered': done, 'share': round(done / total, 3) if total else None}
    return result


def report(folder, endpoint, gold_path=None, refresh=False, cache=CACHE):
    folder = Path(folder)
    transcript = json.loads((folder / 'transcript.json').read_text(encoding='utf-8'))
    case_id = (transcript.get('summary') or {}).get('case_id') or folder.resolve().name.split('-')[0]
    path = Path(gold_path) if gold_path else HERE / 'samples' / case_id / 'model_record.md'
    opening = str((transcript.get('start_information') or {}).get('opening', ''))
    findings = gold_items(endpoint, gold_sections(path.read_text(encoding='utf-8')), opening, case_id, refresh, cache)
    if not findings:
        raise RuntimeError('모범 기록에서 항목을 뽑지 못했습니다')
    items = judge(endpoint, findings, build_log(transcript))
    summary = summarize(items)
    (folder / 'gold_coverage.json').write_text(json.dumps({'case': case_id, 'summary': summary, 'items': items},
                                                         ensure_ascii=False, indent=2), encoding='utf-8')
    return case_id, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('folders', nargs='+', help='result folders written by develop.py')
    parser.add_argument('--gold', help='reference record to use instead of samples/<case>/model_record.md (one folder only)')
    parser.add_argument('--refresh', action='store_true', help='extract the reference findings again instead of using the cache')
    args = parser.parse_args()
    if args.gold and len(args.folders) != 1:
        parser.error('--gold works with a single result folder')
    from nova_local import Endpoint
    endpoint = Endpoint.from_env('NOVA_LOCAL_DOCTOR')
    shares = []
    for folder in args.folders:
        try:
            case_id, summary = report(folder, endpoint, args.gold, args.refresh)
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
