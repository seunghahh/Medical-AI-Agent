"""기초 제출 코드(베이스라인). prepare()와 consult()를 고쳐 자신의 에이전트를 만드세요.

흐름
1. 지정 Doctor 모델(gpt-oss-20b, client.chat)이 매 턴 다음 행동 하나를 JSON으로 고릅니다.
   SAY = 환자에게 질문·설명 한 마디, EXAM = 신체 진찰 동작 하나, DONE = 진료 종료.
2. 행동 결과(환자 대답, 진찰 소견)를 진료 기록에 붙여 다음 턴에 다시 보여 줍니다.
3. 마지막에 같은 모델이 진료 기록만 근거로 S·O·A·P 초진 기록과 주진단을 작성해 제출합니다.

예선 전용 예제입니다(client.chat은 예선 지정 모델). 본선은 prepare()에서 직접 불러온 모델로 바꿉니다.
"""
import json
import re
import time
import unicodedata
from datetime import datetime

from nova_sdk import RECORD_FIELDS, NovaError, empty_record, run_agent

MAX_TURNS = 30      # 대회 한도 50턴 중 이 예제가 쓰는 턴 수
TIME_MARGIN = 180   # 세션 마감 3분 전부터는 기록 작성으로 넘어갑니다(초)
MAX_CHARS = 30      # SAY·EXAM 한 번에 30자(공백 포함, NFC) 이하
# Doctor 모델 추론 강도: 'low'(싸고 빠름, 빈 답 많음)·'medium'·'high'(비싸고 느림, max_tokens 넉넉히).
# 같은 요청 실험에서 medium을 명시하면 지정하지 않을 때보다 빈 답이 적었습니다.
REASONING_EFFORT = 'medium'
STOP_CODES = {'TURN_LIMIT_REACHED', 'SESSION_EXPIRED', 'SESSION_CLOSED', 'RUN_NOT_ACTIVE'}
# 모델·네트워크의 일시 오류. PROVIDER_RESPONSE_INVALID: Doctor 모델이 빈 답을 냄(요금 없음).
# PATIENT_/EXAMINER_RESPONSE_INVALID: 가상 환자·진찰 판정 모델이 형식에 맞는 답을 못 냄(턴 안 씀).
RETRY_CODES = {'PROVIDER_UNAVAILABLE', 'PROVIDER_RESPONSE_INVALID', 'PATIENT_RESPONSE_INVALID',
               'EXAMINER_RESPONSE_INVALID', 'API_UNAVAILABLE'}

STEP_PROMPT = f"""당신은 외래 진료실의 의사입니다. 가상 환자를 진료합니다.
매번 아래 JSON 객체 하나만 출력하세요.
{{"action": "SAY", "content": "환자에게 할 말"}}  질문 하나 또는 설명 한 마디({MAX_CHARS}자 이내)
{{"action": "EXAM", "content": "진찰 동작 하나"}}  예: "복부 압통 확인하겠습니다"({MAX_CHARS}자 이내)
{{"action": "DONE"}}  진료를 마칠 때
진행 순서: 주호소 → 현병력(시작 시점, 양상, 위치, 심해지거나 나아지는 요인, 동반 증상) → 과거력·복용 약·알레르기
→ 가족력 → 사회력(흡연, 음주, 직업) → 필요한 계통 문진 → 관련 신체 진찰 → 환자에게 가능성 높은 진단과 다음 계획 설명 → DONE.
질문은 한 번에 하나만 하고, 이미 물은 것은 다시 묻지 않습니다. 진찰은 한 번에 동작 하나만 요청합니다."""

RECORD_PROMPT = """진료 기록만 근거로 초진 기록을 작성합니다. 확인하지 않은 사실은 쓰지 않습니다.
시작 정보(첫 진술, 활력징후)는 확인된 사실입니다. 병원 입원·초진 기록 말투로 간결하게 씁니다.
(+)/(-)는 대화나 진찰로 확인한 항목에만 쓰고, 확인하지 않은 항목은 ( )로 비워 둡니다.
JSON 객체 하나만 출력하세요. 모든 값은 한국어 문자열(줄바꿈은 \\n)이고, 해당 내용이 없으면 빈 문자열입니다.
{"S": {"cc": "C.C : 주호소 / Onset : 시작 시점 / 통증이면 통증평가 : NRS 점수, 부위, 양상, 기간",
       "hpi": "P.I : 경과를 간결하게, '~하여 내원함'",
       "pmh": "DM / HTN (-/-), Op. Hx (-), Medicine history, Allergy (-)처럼 항목별",
       "fh": "DM / HTN (-/-), Stroke (-) 등", "sh": "Smoking (-) p/d yrs, Alcohol 회/주, 직업",
       "ros": "Fever/Chill (-/-), Chest pain (-), Headache (+)처럼 (+)/(-) 목록"},
 "O": {"exam": "V/S, H/E/E/N/T, Chest, Abdomen 등 부위별 진찰 소견", "tests": "검사 결과"},
 "A": {"evidence": "#1. 주진단 다음 줄에 '- 근거 :' 수집한 사실", "differential": "'r/o 진단명 : 이유'를 한 줄씩"},
 "P": {"tests": "추가 검사 계획", "treatment": "치료", "disposition": "처치 결정", "follow_up": "추적 계획", "education": "환자 교육"},
 "diagnosis": "가장 가능성 높은 주진단 하나"}"""


def prepare():
    # 본선: 등록한 로컬 모델을 여기서 불러옵니다(증례 접근 전, 인터넷 없음).
    # model_dir = os.environ['NOVA_MODEL_DIR']
    # return YourModel.load(model_dir, local_files_only=True)
    return None


def parse_json(text):
    """모델 출력에서 첫 JSON 객체를 꺼냅니다(코드 블록이나 앞뒤 설명이 붙어도 됩니다)."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r'\{', text or ''):
        try:
            value, _ = decoder.raw_decode(text, match.start())
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def with_retry(call, *args, **kwargs):
    """일시 오류는 잠시 뒤 다시 시도합니다. 실패한 요청은 턴을 쓰지 않고, 재시도는 새 요청으로 보냅니다."""
    for wait in (2, 5, 10, None):
        try:
            return call(*args, **kwargs)
        except NovaError as exc:
            if exc.code not in RETRY_CODES or wait is None:
                raise
            time.sleep(wait)


def ask_doctor(client, system, user, max_tokens):
    # gpt-oss-20b는 추론 토큰도 max_tokens에 포함되므로 여유 있게 둡니다.
    completion = with_retry(client.chat, [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
                            max_tokens=max_tokens, reasoning_effort=REASONING_EFFORT)
    return completion['choices'][0]['message']['content'] or ''


def situation(session, transcript):
    initial = session['initial']
    return (f"시작 정보\n첫 진술: {initial['opening']}\n활력징후: {initial['vital_signs']}\n\n"
            '진료 기록\n' + ('\n'.join(transcript) or '(아직 없음)'))


def seconds_left(session):
    if not session.get('deadline'):
        return float('inf')
    return datetime.fromisoformat(session['deadline']).timestamp() - time.time()


def next_step(client, session, transcript, turns_left):
    """다음 행동 하나. 형식이 틀리면(빈 답, JSON 아님, 30자 초과) 바로 다시 묻고, 세 번 모두 틀리면 None."""
    request = situation(session, transcript) + f'\n\n남은 턴: {turns_left}'
    if turns_left <= 2:
        request += '\n이제 환자에게 가능성 높은 진단과 다음 계획을 쉬운 말로 설명하세요(SAY).'
    elif turns_left <= 10:
        request += '\n문진을 마무리하고 필요한 신체 진찰(EXAM)을 하세요.'
    request += '\n다음 행동:'
    for _ in range(3):
        step = parse_json(ask_doctor(client, STEP_PROMPT, request, 2048)) or {}
        action = str(step.get('action', '')).upper()
        content = unicodedata.normalize('NFC', str(step.get('content', ''))).strip()
        if action == 'DONE' or (action in ('SAY', 'EXAM') and 0 < len(content) <= MAX_CHARS):
            return action, content
        request += f'\n(직전 출력은 형식에 맞지 않았습니다. {MAX_CHARS}자 이내의 JSON 하나만 출력하세요.)'
    return None


def consult(client, model):
    session = client.session()
    transcript, turns_left, refused = [], min(MAX_TURNS, session['turns_remaining']), 0
    while turns_left > 0 and seconds_left(session) > TIME_MARGIN and refused < 3:
        try:
            step = next_step(client, session, transcript, turns_left)
        except NovaError:
            break  # 호출 한도·계속된 장애: 지금까지의 기록으로 마무리합니다.
        if step is None or step[0] == 'DONE':
            break
        action, content = step
        try:
            if action == 'SAY':
                reply = with_retry(client.say, content)
                transcript += [f'의사: {content}', f"환자: {reply['content']}"]
            else:
                reply = with_retry(client.exam, content)
                transcript.append(f"진찰({content}): {reply['result']}")
        except NovaError as exc:
            if exc.code in STOP_CODES or exc.code in RETRY_CODES:
                break
            # EXAM_NOT_SPECIFIC(진찰을 뭉뚱그려 요청), EXAM_NOT_RECOGNIZED(신체진찰이 아닌 요청).
            # 잘못 보낸 요청이라 답은 없지만 1턴을 씁니다.
            refused += 1
            turns_left -= 1
            hint = '신체진찰이 아닙니다. 진찰 동작 하나를 요청하세요' if exc.code == 'EXAM_NOT_RECOGNIZED' else '하나만 구체적으로 다시 요청하세요'
            transcript.append(f'(알림: {exc.code}, 1턴 사용. {hint})')
            continue
        refused = 0
        turns_left = min(turns_left - 1, reply.get('turns_remaining', turns_left))
    note, diagnosis = write_record(client, session, transcript)
    return client.diagnose(note=note, diagnosis=diagnosis)


def write_record(client, session, transcript):
    result = {}
    for _ in range(3):  # 형식이 틀리거나 주진단이 빠지면 다시 작성합니다.
        try:
            result = parse_json(ask_doctor(client, RECORD_PROMPT, situation(session, transcript), 4096)) or {}
        except NovaError:
            break
        if str(result.get('diagnosis') or '').strip():
            break
    note = empty_record()
    for section, fields in RECORD_FIELDS.items():
        values = result.get(section) if isinstance(result.get(section), dict) else {}
        for field in fields:
            value = values.get(field) or ''
            note[section][field] = ', '.join(map(str, value)) if isinstance(value, list) else str(value)
    if not any(note['S'].values()):
        note['S']['hpi'] = '\n'.join(line for line in transcript if line.startswith(('의사:', '환자:')))
    diagnosis = ' '.join(str(result.get('diagnosis') or '').split()) or '진단 미정'
    return note, diagnosis


if __name__ == '__main__':
    run_agent(prepare, consult)
