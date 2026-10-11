"""공개 샘플 증례로 내 에이전트를 내 컴퓨터에서 실행합니다(채점 없음).

  python develop.py --list                              # 들어 있는 샘플 증례
  python develop.py --case samples/C10                  # ./run.py의 prepare()·consult()로 진료
  python develop.py --case samples/C10 --agent ../my_agent/run.py
  python develop.py --case samples/C10 --manual         # 내가 직접 의사가 되어 가상 환자와 대화

모델 연결(OpenAI 호환 주소)은 환경변수로 정합니다. 자세한 내용은 README.md를 보세요.
  NOVA_LOCAL_DOCTOR_URL / _MODEL / _KEY   Doctor 모델(대회의 gpt-oss-20b 자리)  기본: Ollama gpt-oss:20b
  NOVA_LOCAL_SIM_URL / _MODEL / _KEY      가상 환자·진찰 판정 모델                기본: Doctor와 같음
"""
import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from nova_local import NovaError, load_case, save_result  # noqa: E402
from local import SessionSimulator as Simulator
from live import Journal, open_monitor
from monitor import LiveClient


def load_agent(path):
    path = Path(path).resolve()
    sys.path.insert(0, str(path.parent))  # 에이전트 폴더의 모듈(nova_sdk 등)을 먼저 찾도록
    spec = importlib.util.spec_from_file_location('nova_agent', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ('prepare', 'consult'):
        if not callable(getattr(module, name, None)):
            raise SystemExit(f'{path}에 {name}() 함수가 없습니다.')
    return module


def manual(client):
    print('\n직접 진료: "s 질문", "e 진찰 동작", "q" 종료 (한 번에 30자 이하)')
    while True:
        line = input('> ').strip()
        if line in ('q', 'quit', 'exit'):
            return
        kind, _, content = line.partition(' ')
        try:
            if kind == 's':
                client.say(content)
            elif kind == 'e':
                client.exam(content)
            else:
                print('s 또는 e로 시작하세요.')
        except NovaError as exc:
            print(f'(오류 {exc.code}) 남은 턴: {client.session()["turns_remaining"]}')
        except ValueError as exc:
            print(f'({exc})')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--case', help='샘플 폴더 또는 case.json (예: samples/C10)')
    parser.add_argument('--agent', default='run.py', help='prepare()와 consult(client, state)가 있는 파일 (기본 ./run.py)')
    parser.add_argument('--manual', action='store_true', help='에이전트 대신 직접 질문·진찰')
    parser.add_argument('--minutes', type=float, default=30, help='진료 제한 시간(분, 대회와 같은 30분). 0이면 제한 없음')
    parser.add_argument('--out', default='results', help='결과 폴더')
    parser.add_argument('--quiet', action='store_true', help='대화를 화면에 출력하지 않음')
    parser.add_argument('--list', action='store_true', help='samples/ 아래 증례 목록')
    parser.add_argument('--max-turns', type=int, choices=range(1, 51), default=50, metavar='1..50')
    parser.add_argument('--port', type=int, default=8768, help='병원 모니터 포트')
    parser.add_argument('--no-browser', action='store_true', help='브라우저/서버 자동 열기 생략; 이벤트는 저장')
    args = parser.parse_args()

    if args.list or not args.case:
        cases = sorted(p.parent for p in (HERE / 'samples').glob('*/case.json'))
        for folder in cases:
            case = load_case(folder)
            print(f'{folder.relative_to(HERE)}  {case["start_information"]["opening"][:60]}')
        if not cases:
            print('samples/ 폴더에 공개 샘플이 없습니다. 개발 가이드에서 샘플 ZIP을 받아 samples/에 푸세요.')
        return

    agent = None if args.manual else load_agent(args.agent)
    state = agent.prepare() if agent else None
    if isinstance(state, dict):
        state['max_turns'] = args.max_turns
    case = load_case(args.case)
    simulator = Simulator(case, minutes=args.minutes, log=None if args.quiet else lambda line: print(line, flush=True))
    out = Path(args.out) / f'{case["case_id"]}-{datetime.now():%Y%m%d-%H%M%S-%f}'
    out.mkdir(parents=True)
    journal = Journal(out, {'model': simulator.doctor.model, 'mode': 'model', 'engine': 'NOVA local + RAG',
                           'case_indices': [case['case_id']], 'rag_enabled': bool(isinstance(state, dict) and state.get('rag')),
                           'review_enabled': False, 'memory_enabled': False, 'judge_mode': 'none'})
    client = LiveClient(simulator, journal)
    if isinstance(state, dict):
        state['observer'] = client.emit
    if not args.no_browser:
        try:
            open_monitor(args.port)
        except OSError as exc:
            print(f'모니터 자동 열기 실패: {exc}', file=sys.stderr)
    print(f'증례 {case["case_id"]} · Doctor {simulator.doctor.model} · 최대 {args.max_turns}턴', flush=True)
    print(f'시작 정보: {case["start_information"]}', flush=True)
    started = time.time()
    status, error = 'finished', None
    try:
        client.ready()
        if args.manual:
            manual(client)
        else:
            agent.consult(client, state)
    except KeyboardInterrupt:
        status, error = 'interrupted', '사용자 중단'
        print('\n중단했습니다.')
    except Exception as exc:
        status, error = 'failed', f'{type(exc).__name__}: {exc}'
        print(error, file=sys.stderr)
    finally:
        summary = save_result(simulator, out, started)
        summary.update(error=error, max_turns=args.max_turns, official_score=False)
        (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        if isinstance(state, dict):
            (out / 'retrievals.json').write_text(json.dumps(state.get('retrievals', []), ensure_ascii=False, indent=2))
            (out / 'grounding.json').write_text(json.dumps(state.get('grounding', {}), ensure_ascii=False, indent=2))
            if state.get('rag'):
                state['rag'].close()
        journal.emit('case_end', case_index=case['case_id'], completed=simulator.status == 'completed',
                     error=error, turn=simulator.turns)
        journal.emit('run_end', status=status, error=error)
        journal.stream.close()
    print(f'상태 {summary["status"]} · {summary["turns"]}턴 · Doctor {summary["doctor_calls"]}회 · {summary["seconds"]}초')
    print(f'결과: {out.resolve()}/')
    print(f'모범 기록(채점 정답 아님): samples/{case["case_id"]}/model_record.md')
    return 130 if status == 'interrupted' else 1 if status == 'failed' else 0


if __name__ == '__main__':
    sys.exit(main())
