#!/usr/bin/env python3
"""Local benchmark runner; submission contains the agent, retrieval, grounding and the SDK."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import time
import zipfile

import run
from benchmarks import ACI_SPLITS, load_rows
from local import BenchmarkSimulator
from nova_local import save_result
from rag import GuidelineIndex, guideline_root
from live import Journal, open_monitor
from monitor import LiveClient

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT.parent / 'nova-benchmarks-2026'


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root):
    results = []
    for name in ('agentclinic_medqa_extended.jsonl', 'agentclinic_medqa.jsonl'):
        path = root / 'AgentClinic-MedQA' / name
        with path.open() as stream:
            count = sum(bool(json.loads(line)['OSCE_Examination']) for line in stream if line.strip())
        results.append({'file': str(path), 'rows': count, 'sha256': sha256(path)})
    for split in ('train', 'validate', 'test'):
        path = root / 'DDXPlus' / f'release_{split}_patients.zip'
        with zipfile.ZipFile(path) as z:
            members = [n for n in z.namelist() if not n.endswith('/') and not n.startswith('__MACOSX/')]
            if len(members) != 1:
                raise ValueError(f'Expected one CSV in {path}')
            with z.open(members[0]) as binary, io.TextIOWrapper(binary) as text:
                # Streaming reads also check the member's CRC on reaching EOF.
                count = sum(1 for _ in csv.DictReader(text))
        results.append({'file': str(path), 'rows': count, 'sha256': sha256(path)})
    for split, name in ACI_SPLITS.items():
        path = root / 'ACI-Bench/aci-bench-corpus/challenge_data' / name
        with path.open(newline='') as stream:
            count = sum(1 for _ in csv.DictReader(stream))
        results.append({'file': str(path), 'rows': count, 'sha256': sha256(path)})
    return results


def normalized(text):
    return re.sub(r'\W+', '', text.casefold())


def execute(args):
    destination = ROOT / 'results' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    destination.mkdir(parents=True)
    state = {'rag': None} if args.no_rag else run.prepare()
    index = state.get('rag')
    manifest = {'arguments': vars(args), 'created': datetime.now().astimezone().isoformat(),
                'official_score': False, 'code_sha256': {p.name: sha256(p) for p in ROOT.glob('*.py')},
                'rag': index.info if index else None, 'source_files': {}}
    write_json(destination / 'manifest.json', manifest)
    summaries = []
    journal = Journal(destination, {'model': 'NOVA_LOCAL_DOCTOR', 'mode': 'model', 'engine': 'NOVA benchmark + RAG',
                                   'case_indices': [args.dataset], 'case_count': args.limit, 'rag_enabled': bool(index),
                                   'memory_enabled': False, 'review_enabled': False, 'judge_mode': 'none'},
                      directory=destination.parent / '.live')
    if not args.no_browser:
        try:
            open_monitor(args.port)
        except OSError as exc:
            print(f'Monitor unavailable: {exc}', file=sys.stderr)
    try:
        for row in load_rows(args.data_root, args.dataset, args.split, args.offset, args.limit):
            case = row['case']
            if args.dataset == 'aci':
                case['task'] = 'note-generation'
            source = row['provenance']['file']
            if source not in manifest['source_files']:
                manifest['source_files'][source] = sha256(source)
                for extra in ('release_evidences.json', 'release_conditions.json') if args.dataset == 'ddxplus' else ():
                    path = Path(source).parent / extra
                    manifest['source_files'][str(path)] = sha256(path)
                write_json(destination / 'manifest.json', manifest)
            out = destination / case['case_id']
            out.mkdir()
            encounter_state = {'max_turns': args.max_turns, 'rag': index, 'retrievals': []}
            # Only guideline retrieval data are reused; patient state starts afresh.
            sim = BenchmarkSimulator(case, minutes=args.minutes, log=lambda line: print(line, flush=True))
            client = LiveClient(sim, journal)
            encounter_state['observer'] = client.emit
            started = time.time()
            error = None
            print(f'\n[{case["case_id"]}] {case["start_information"]["opening"]}', flush=True)
            try:
                client.ready()
                if args.dataset == 'aci':
                    session = client.session()
                    transcript = [row['dialogue']]
                    (out / 'source_dialogue.txt').write_text(row['dialogue'], encoding='utf-8')
                    references = run.retrieve(client, session, transcript, encounter_state, preliminary=False)
                    note, diagnosis = run.write_record(client, session, transcript, preliminary=False, references=references)
                    client.diagnose(note=note, diagnosis=diagnosis)
                else:
                    run.consult(client, encounter_state)
            except KeyboardInterrupt:
                error = 'KeyboardInterrupt: interrupted'
                raise
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
                print(error, file=sys.stderr, flush=True)
            finally:
                summary = save_result(sim, out, started)
                summary['error'] = error
                summary['task'] = 'note-generation' if args.dataset == 'aci' else 'preliminary-consultation'
                summary['local_contract_pass'] = (sim.status == 'completed' and sim.doctor_calls >= 1
                    and error is None and sim.turns <= args.max_turns and summary['seconds'] <= args.minutes * 60 and (args.dataset == 'aci' or sim.record['O']['tests'] == ''))
                summary['official_score'] = False
                summary['max_turns'] = args.max_turns
                summary['time_limit_minutes'] = args.minutes
                evaluation = {'provenance': row['provenance'], 'summary': summary}
                if 'gold' in row:
                    evaluation.update(gold_diagnosis=row['gold'], predicted_diagnosis=sim.diagnosis,
                                      normalized_exact_match=normalized(sim.diagnosis or '') == normalized(row['gold']),
                                      assessment='String equality only; translation/synonyms require independent review.')
                else:
                    (out / 'reference_note.txt').write_text(row['reference_note'], encoding='utf-8')
                    evaluation['assessment'] = 'Note-generation only. No gold diagnosis or automatic clinical quality score.'
                write_json(out / 'evaluation.json', evaluation)
                write_json(out / 'retrievals.json', encounter_state['retrievals'])
                write_json(out / 'grounding.json', encounter_state.get('grounding', {}))
                write_json(out / 'summary.json', summary)
                summaries.append(summary)
                client.emit('case_end', completed=sim.status == 'completed', error=error, turn=sim.turns)
                write_json(destination / 'summary.json', summaries)
                print('Saved:', out, flush=True)
        if not summaries:
            raise ValueError('No rows selected: check --offset/--limit')
    finally:
        failure = sys.exc_info()[1]
        journal.emit('run_end', status='interrupted' if isinstance(failure, KeyboardInterrupt) else
                     'failed' if failure or any(s['error'] for s in summaries) else 'finished',
                     error=str(failure) if failure else None)
        journal.stream.close()
        if index:
            index.close()
    print('Artifacts:', destination, flush=True)
    return 1 if any(s['error'] or s['status'] != 'completed' for s in summaries) else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    check = sub.add_parser('check', help='Validate all benchmark files, counts and ZIP CRCs without model calls')
    check.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    execute_parser = sub.add_parser('run')
    execute_parser.add_argument('--data-root', type=str, default=str(DEFAULT_DATA))
    execute_parser.add_argument('--dataset', required=True, choices=('agentclinic', 'ddxplus', 'aci'))
    execute_parser.add_argument('--split', default='validate')
    execute_parser.add_argument('--offset', type=int, default=0)
    execute_parser.add_argument('--limit', type=int, default=1)
    execute_parser.add_argument('--max-turns', type=int, choices=range(1, 51), default=50, metavar='1..50')
    execute_parser.add_argument('--minutes', type=float, default=30)
    execute_parser.add_argument('--no-rag', action='store_true', help='Ablation run without guideline retrieval')
    execute_parser.add_argument('--port', type=int, default=8768)
    execute_parser.add_argument('--no-browser', action='store_true', help='Do not auto-start/open monitor; still write events')
    search = sub.add_parser('search', help='Build/reuse offline guideline index and show source metadata')
    search.add_argument('query')
    sub.add_parser('package', help='Build submission.zip from a fixed allowlist, excluding datasets, keys and outputs')
    args = parser.parse_args()
    if args.command == 'check':
        result = inventory(args.data_root)
        write_json(ROOT / 'data-inventory.json', result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == 'search':
        index = GuidelineIndex(guideline_root())
        try:
            print(json.dumps({'index': index.info, 'hits': [{k: v for k, v in h.items() if k != 'text'}
                                for h in index.search(args.query)]}, ensure_ascii=False, indent=2))
        finally:
            index.close()
    elif args.command == 'package':
        names = ['run.py', 'rag.py', 'grounding.py', 'nova_sdk.py', 'requirements.txt', 'SOURCES.md']
        with zipfile.ZipFile(ROOT / 'submission.zip', 'w', zipfile.ZIP_DEFLATED) as z:
            for name in names:
                z.write(ROOT / name, name)
        print(ROOT / 'submission.zip')
    else:
        if args.offset < 0 or args.limit < 1 or not 3 < args.minutes <= 30:
            parser.error('--offset >= 0, --limit >= 1, 3 < --minutes <= 30 are required')
        return execute(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
