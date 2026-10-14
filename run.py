"""N.O.V.A. 2026 예선 제출 코드: 공감형 Doctor Agent.

흐름
1. 문진: Doctor 모델이 매 턴 SAY(질문+짧은 공감) / EXAM(진찰 하나) / DONE 중 하나를 JSON으로 고릅니다.
2. 진찰: 남은 턴이 줄면 신체 진찰로 넘어가고, 진찰을 한 번도 안 했으면 강제로 안내합니다.
3. 판단: 진료 기록을 보고 주진단·감별진단과 환자에게 할 설명 5마디를 한 번에 정리합니다(추론 강도 high).
4. 설명: 마지막 턴에 진단 설명 → 검사·치료 계획 → 환자 교육 → 경고 증상 → 공감 마무리를 SAY로 전달합니다.
   (기록에 교육 계획을 쓰는 것만으로는 대화 평가에 반영되지 않기 때문입니다.)
5. 기록: 턴 번호가 붙은 진료 기록만 근거로 S·O·A·P 초진 기록과 주진단을 작성해 DIAGNOSE로 제출합니다.

예선 전용입니다(client.chat은 예선 지정 Doctor 모델). 증례마다 client.chat을 여러 번 호출합니다.
"""
import json
import re
import time
import unicodedata
import uuid
from datetime import datetime

from nova_sdk import RECORD_FIELDS, NovaError, empty_record, run_agent

# ---------------------------------------------------------------- 설정
MAX_TURNS = 44        # 대회 한도 50턴 중 사용할 턴(여유 6턴)
WRAP_TURNS = 5        # 마지막에 환자 설명용으로 남겨 두는 턴
EXAM_WINDOW = 12      # 설명 턴을 뺀 남은 턴이 이 값 이하가 되면 진찰 단계로 안내
MIN_HISTORY = 8       # 이 횟수의 SAY 전에는 DONE을 받지 않습니다(문진 부족 방지)
TIME_MARGIN = 300     # 세션 마감 5분 전에는 문진을 멈추고 마무리(초)
MAX_CHARS = 30        # SAY·EXAM 한 번에 30자(공백 포함, NFC) 이하

STEP_EFFORT = 'medium'     # 매 턴 행동 선택
ASSESS_EFFORT = 'high'     # 진단 판단은 한 번만 깊게
RECORD_EFFORT = 'medium'   # 기록 작성

STOP_CODES = {'TURN_LIMIT_REACHED', 'SESSION_EXPIRED', 'SESSION_CLOSED', 'RUN_NOT_ACTIVE'}
RETRY_CODES = {'PROVIDER_UNAVAILABLE', 'PROVIDER_RESPONSE_INVALID', 'PATIENT_RESPONSE_INVALID',
               'EXAMINER_RESPONSE_INVALID', 'API_UNAVAILABLE', 'REQUEST_PENDING'}
# 같은 request_id로 다시 보내야 하는 오류(서버가 끝낸 답을 그대로 돌려줍니다). 나머지는 새 요청으로 보냅니다.
SAME_ID_CODES = {'PROVIDER_UNAVAILABLE', 'REQUEST_PENDING', 'API_UNAVAILABLE'}
TURN_USED_CODES = {'EXAM_NOT_SPECIFIC', 'EXAM_NOT_RECOGNIZED'}  # 답은 없지만 1턴을 씁니다.

# ---------------------------------------------------------------- 프롬프트
STEP_PROMPT = """당신은 따뜻하고 노련한 외래 진료 의사입니다. 가상 환자를 진료합니다.
환자가 존중받고 안심한다고 느끼게 하면서, 진단에 필요한 정보를 빠짐없이 모읍니다.
매번 아래 JSON 객체 하나만 출력하세요.
{"action": "SAY", "content": "환자에게 할 말"}   공감 + 질문 하나({MAX}자 이내)
{"action": "EXAM", "content": "진찰 동작 하나"}  예: "복부 청진하겠습니다"({MAX}자 이내)
{"action": "DONE"}  문진과 진찰이 충분할 때(그러면 환자 설명 단계로 넘어갑니다)

대화 원칙
- 존댓말, 부드러운 말투, 쉬운 말(의학용어 피하기). 한 번에 {MAX}자 이내(공백·문장부호 포함).
- 첫 마디는 인사 + 열린 질문. 예: "안녕하세요. 어디가 불편하세요?"
- 환자가 통증·불안·걱정·괴로움을 말하면 질문 앞에 짧은 공감을 붙입니다.
  예: "많이 힘드셨겠어요. 언제부터였나요?" 공감 문구는 매번 다르게, 과장하지 않고 진심으로.
  감정을 드러낸 직후에는 반드시 공감하고, 그 밖에는 2~3번에 한 번 정도만 붙입니다.
- 질문은 한 번에 하나만. 이미 물은 것은 다시 묻지 않습니다. 답이 모호하면 짧게 되묻습니다.
- 환자의 말을 받아서 이어 묻습니다(대화가 끊기지 않게).

문진 항목(기록에 없는 것부터): 주호소 → 시작 시점·경과 → 위치·양상·세기(통증이면 0~10점)
→ 악화·완화 요인 → 동반 증상 → 과거 병력·수술력 → 복용 약 → 알레르기 → 가족력
→ 사회력(흡연·음주·직업) → 진단 감별에 필요한 계통 문진(열·오한, 기침, 호흡곤란, 가슴 통증,
구토·설사, 배뇨·배변 변화, 체중 변화 등. 없다는 답도 중요한 정보입니다).
질문은 가능성 높은 진단을 가르는 것(위험 신호 포함)을 우선합니다.

진찰 원칙
- 활력징후는 시작 정보에 있으므로 다시 요청하지 않습니다.
- 진단에 도움이 되는 진찰을 동작 하나씩 요청합니다. 예: "복부 촉진하겠습니다", "폐 청진하겠습니다",
  "인두 확인하겠습니다". 여러 진찰을 묶거나("다 해 주세요"), 결과를 골라 말하게 하는 요청은 금지입니다.
- 검사(혈액·영상)는 예선에서 받을 수 없으니 요청하지 않습니다."""

ASSESS_PROMPT = """진료 기록을 보고 의사로서 판단을 정리합니다. 기록에 없는 사실은 지어내지 않습니다.
JSON 객체 하나만 출력하세요.
{"diagnosis": "가장 가능성 높은 주진단 하나(진단명만)",
 "differential": ["감별진단 1", "감별진단 2", "감별진단 3"],
 "explain": ["1번 말", "2번 말", "3번 말", "4번 말", "5번 말"]}

explain은 환자에게 순서대로 한 마디씩 건넬 말입니다. 모두 존댓말, 쉬운 말, 의학용어 최소화,
각각 {MAX}자 이내(공백·문장부호 포함)로 반드시 지키세요. 말투는 따뜻하게.
 1) 진단 설명: 가장 가능성 높은 상태를 쉬운 말로. 확정이 아니라 '의심된다'는 겸손한 표현.
 2) 필요한 검사·치료 계획 (이 환자에 맞게 구체적으로)
 3) 환자 교육·생활 수칙 (이 환자에 맞게 구체적으로)
 4) 바로 다시 오거나 응급실에 와야 하는 경고 증상
 5) 환자의 걱정에 공감하는 마무리 + 궁금한 점이 있는지 묻기"""

RECORD_PROMPT = """진료 기록만 근거로 초진 기록을 작성합니다.
규칙
- 진료 기록에 있는 사실만 씁니다. 묻지 않은 것·하지 않은 진찰을 있는 것처럼 쓰지 않고, 환자 답과 다르게 쓰지 않습니다.
- 시작 정보(첫 진술, 활력징후)는 확인된 사실입니다. 활력징후는 '(시작 정보)'로 표기합니다.
- (+)/(-)는 대화나 진찰로 확인한 항목에만 씁니다. 확인하지 않은 항목은 Allergy ( )처럼 괄호를 비웁니다.
- 병원 입원·초진 기록 말투로 간결하게. 진찰 결과에는 받은 턴 번호를 붙입니다. 예: 'Abdomen : tenderness (+) RLQ (턴 21)'
- 예선에는 검사 결과가 없으므로 O.tests는 빈 문자열입니다. 필요한 검사는 P.tests에 이유와 함께 씁니다.
- 아직 하지 않은 일은 계획으로만 씁니다(시행한 결과처럼 쓰지 않기).
- A.evidence는 '#1. 주진단' 다음 줄에 '- 근거 :'로 수집한 사실과 그 해석을 씁니다.
- A.differential은 'r/o 진단명 : 이유(배제·유지 근거)'를 한 줄씩 2~4개 씁니다.
- P.education에는 진료 중 환자에게 실제로 설명한 내용(턴 번호 포함)과 추가로 필요한 교육을 씁니다.
- 아래 '잠정 판단'이 있으면 일관되게 쓰되, 이후 대화에서 새 정보가 나왔으면 반영해 수정합니다.
JSON 객체 하나만 출력하세요. 모든 값은 한국어(영어 의학 약어 허용) 문자열(줄바꿈은 \\n)이고, 해당 내용이 없으면 빈 문자열입니다.
{"S": {"cc": "C.C : 주호소\\nOnset : 시작 시점\\n통증이면 통증평가 : NRS 점수, 부위, 양상, 기간",
       "hpi": "P.I : 경과를 간결하게, '~하여 내원함'",
       "pmh": "DM / HTN (-/-), Op. Hx (-), Medicine history, Allergy (-)처럼 항목별",
       "fh": "DM / HTN (-/-), Stroke (-) 등", "sh": "Smoking (-) p/d yrs, Alcohol 회/주, 직업",
       "ros": "Fever/Chill (-/-), Chest pain (-), Headache (+)처럼 (+)/(-) 목록"},
 "O": {"exam": "V/S (시작 정보), H/E/E/N/T, Chest, Abdomen 등 부위별 소견 (턴 N)", "tests": ""},
 "A": {"evidence": "#1. 주진단\\n- 근거 : ...", "differential": "r/o 진단명 : 이유"},
 "P": {"tests": "필요한 검사 : 이유", "treatment": "약물·처치 계획", "disposition": "귀가·외래 추적·의뢰·응급 처치 등",
       "follow_up": "재진 시점, 바로 다시 와야 하는 경고 증상", "education": "환자 교육(설명한 턴 포함)"},
 "diagnosis": "가장 가능성 높은 주진단 하나(진단명만)"}"""

# 마지막 설명 단계: 남은 턴 수 → 전달할 말의 순서(0=진단 설명, 1=검사·치료, 2=교육, 3=경고 증상, 4=공감 마무리)
EXPLAIN_PRIORITY = {5: [0, 1, 2, 3, 4], 4: [0, 1, 3, 4], 3: [0, 3, 4], 2: [0, 4], 1: [0]}
GENERIC_EXPLAIN = [
    '진찰과 문진을 종합해 설명드릴게요.',
    '필요한 검사와 치료를 함께 정할게요.',
    '증상이 어떻게 변하는지 살펴봐 주세요.',
    '심해지면 바로 다시 오셔야 해요.',
    '많이 걱정되셨죠? 궁금한 점 있으세요?',
]


# ---------------------------------------------------------------- 공통 도구
def prepare():
    # 예선은 준비할 자산이 없습니다(본선: NOVA_MODEL_DIR에서 모델 로드).
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
    """일시 오류는 잠시 뒤 다시 시도합니다.
    PROVIDER_UNAVAILABLE·REQUEST_PENDING·API_UNAVAILABLE은 같은 request_id로, 빈 답 등은 새 request_id로 보냅니다."""
    request_id = str(uuid.uuid4())
    for wait in (2, 5, 10, 20, None):
        try:
            return call(*args, request_id=request_id, **kwargs)
        except NovaError as exc:
            if exc.code not in RETRY_CODES or wait is None:
                raise
            time.sleep(wait)
            if exc.code not in SAME_ID_CODES:
                request_id = str(uuid.uuid4())


def ask_doctor(client, system, user, max_tokens, effort):
    # 추론 토큰도 max_tokens에 포함되므로 여유 있게 둡니다.
    completion = with_retry(client.chat, [{'role': 'system', 'content': system.replace('{MAX}', str(MAX_CHARS))},
                                          {'role': 'user', 'content': user}],
                            max_tokens=max_tokens, reasoning_effort=effort)
    try:
        return completion['choices'][0]['message']['content'] or ''
    except (KeyError, IndexError, TypeError):
        return ''


def clean(text):
    return unicodedata.normalize('NFC', str(text or '')).strip()


def situation(session, transcript):
    initial = session.get('initial') or {}
    return (f"시작 정보(턴 0)\n첫 진술: {initial.get('opening', '')}\n활력징후: {initial.get('vital_signs', '')}\n\n"
            '진료 기록\n' + ('\n'.join(transcript) or '(아직 없음)'))


def seconds_left(session):
    if not session.get('deadline'):
        return float('inf')
    try:
        return datetime.fromisoformat(session['deadline']).timestamp() - time.time()
    except ValueError:
        return float('inf')


# ---------------------------------------------------------------- 진료 단계
def act(client, state, action, content):
    """SAY/EXAM 한 번. 진료 기록과 턴 번호를 갱신합니다. 실패하면 NovaError를 그대로 올립니다."""
    n = state['turn'] + 1
    try:
        if action == 'SAY':
            reply = with_retry(client.say, content)
            state['say'] += 1
            state['transcript'] += [f'[턴 {n}] 의사: {content}', f"[턴 {n}] 환자: {clean(reply.get('content'))}"]
        else:
            reply = with_retry(client.exam, content)
            state['exam'] += 1
            state['transcript'].append(f"[턴 {n}] 진찰({content}): {clean(reply.get('result'))}")
    except NovaError as exc:
        if exc.code in TURN_USED_CODES:
            state['turn'] = n  # 잘못 보낸 요청이지만 1턴을 씁니다.
        raise
    state['turn'] = n
    if isinstance(reply.get('turns_remaining'), int):
        state['server_left'] = reply['turns_remaining']
    return reply


def turns_left(state):
    return min(state['budget'] - state['turn'], state['server_left'])


def next_step(client, session, state):
    """다음 행동 하나. 형식이 틀리면 바로 다시 묻고, 세 번 모두 틀리면 None."""
    usable = turns_left(state) - WRAP_TURNS
    request = situation(session, state['transcript']) + f'\n\n문진·진찰에 쓸 수 있는 남은 턴: {usable}'
    if state['say'] == 0:
        request += '\n첫 마디입니다. 따뜻한 인사와 함께 어디가 불편한지 물으세요(SAY).'
    elif state['say'] < MIN_HISTORY:
        request += '\n아직 문진이 부족합니다. 기록에 없는 문진 항목을 이어서 물으세요(SAY).'
    if state['exam'] == 0 and usable <= EXAM_WINDOW + 4:
        request += '\n아직 신체 진찰을 하지 않았습니다. 지금 진단에 가장 중요한 진찰을 하세요(EXAM).'
    elif usable <= EXAM_WINDOW:
        request += '\n문진을 마무리하고, 진단 감별에 필요한 신체 진찰(EXAM)을 하나씩 하세요. 충분하면 DONE.'
    request += '\n다음 행동:'
    allow_done = state['say'] >= MIN_HISTORY
    for _ in range(3):
        step = parse_json(ask_doctor(client, STEP_PROMPT, request, 2048, STEP_EFFORT)) or {}
        action = str(step.get('action', '')).upper()
        content = clean(step.get('content'))
        if action == 'DONE' and allow_done:
            return 'DONE', ''
        if action in ('SAY', 'EXAM') and 0 < len(content) <= MAX_CHARS:
            return action, content
        request += f'\n(직전 출력은 형식에 맞지 않았습니다. {MAX_CHARS}자 이내의 JSON 하나만 출력하세요.)'
    return None


def assess(client, session, state):
    """주진단·감별진단과 환자에게 할 설명 5마디. 실패해도 빈 값으로 진행합니다."""
    effort = ASSESS_EFFORT if seconds_left(session) > 600 else 'medium'
    data = {}
    for attempt in range(2):
        try:
            data = parse_json(ask_doctor(client, ASSESS_PROMPT, situation(session, state['transcript']),
                                         8000, effort if attempt == 0 else 'medium')) or {}
        except NovaError:
            data = {}
        if clean(data.get('diagnosis')):
            break
    explain = data.get('explain') if isinstance(data.get('explain'), list) else []
    differential = data.get('differential') if isinstance(data.get('differential'), list) else []
    return {'diagnosis': clean(data.get('diagnosis')),
            'differential': [clean(d) for d in differential if clean(d)],
            'explain': [clean(e) for e in explain]}


def explain_to_patient(client, session, state, assessment):
    """마지막 턴에 진단·계획·교육·경고 증상·공감 마무리를 환자에게 말합니다."""
    count = min(len(GENERIC_EXPLAIN), turns_left(state))
    for index in EXPLAIN_PRIORITY.get(count, []):
        if seconds_left(session) < 60:
            break
        sentences = assessment['explain']
        text = sentences[index] if index < len(sentences) else ''
        if not 0 < len(text) <= MAX_CHARS:
            text = GENERIC_EXPLAIN[index]
        try:
            act(client, state, 'SAY', text)
        except NovaError:
            break


# ---------------------------------------------------------------- 기록
def clip_note(note, limit=95_000):
    """기록 전체 UTF-8 100,000바이트 한도를 넘지 않게 긴 칸부터 줄입니다."""
    def size():
        return len(json.dumps(note, ensure_ascii=False).encode('utf-8'))
    while size() > limit:
        section, field = max(((s, f) for s in note for f in note[s]), key=lambda k: len(note[k[0]][k[1]]))
        note[section][field] = note[section][field][:len(note[section][field]) // 2]
    return note


def write_record(client, session, state, assessment):
    request = situation(session, state['transcript'])
    if assessment['diagnosis']:
        request += (f"\n\n잠정 판단(진료 중 환자에게 설명한 방향)\n주진단: {assessment['diagnosis']}\n"
                    f"감별진단: {', '.join(assessment['differential']) or '(없음)'}")
    low_time = seconds_left(session) < 240
    attempts = [('low', 4096)] * 3 if low_time else [(RECORD_EFFORT, 8000), (RECORD_EFFORT, 8000), ('low', 4096)]
    result = {}
    for effort, tokens in attempts:  # 형식이 틀리거나 주진단이 빠지면 다시 작성합니다.
        try:
            result = parse_json(ask_doctor(client, RECORD_PROMPT, request, tokens, effort)) or {}
        except NovaError:
            result = {}
            continue
        if clean(result.get('diagnosis')):
            break
    note = empty_record()
    for section, fields in RECORD_FIELDS.items():
        values = result.get(section) if isinstance(result.get(section), dict) else {}
        for field in fields:
            value = values.get(field) or ''
            note[section][field] = '\n'.join(map(str, value)) if isinstance(value, list) else str(value)
    note['O']['tests'] = ''  # 예선에는 검사 결과가 없습니다.
    if not any(note['S'].values()):
        note['S']['hpi'] = '\n'.join(line for line in state['transcript'] if '의사:' in line or '환자:' in line)
    diagnosis = ' '.join(clean(result.get('diagnosis')).split()) or assessment['diagnosis'] or '진단 미정'
    return clip_note(note), diagnosis.encode('utf-8')[:500].decode('utf-8', 'ignore')


# ---------------------------------------------------------------- 진입점
def consult(client, model):
    session = client.session()
    server_left = session.get('turns_remaining', MAX_TURNS)
    state = {'turn': 0, 'say': 0, 'exam': 0, 'transcript': [],
             'budget': min(MAX_TURNS, server_left), 'server_left': server_left}
    refused = bad = 0
    while turns_left(state) > WRAP_TURNS and seconds_left(session) > TIME_MARGIN and refused < 3 and bad < 2:
        try:
            step = next_step(client, session, state)
        except NovaError:
            break  # 호출 한도·계속된 장애: 지금까지의 기록으로 마무리합니다.
        if step is None:
            bad += 1
            continue
        if step[0] == 'DONE':
            break
        action, content = step
        try:
            act(client, state, action, content)
        except NovaError as exc:
            if exc.code in STOP_CODES or exc.code not in TURN_USED_CODES:
                break
            # EXAM_NOT_SPECIFIC·EXAM_NOT_RECOGNIZED: 잘못 보낸 요청(답 없음, 1턴 사용). 다음 호출에 알려 줍니다.
            refused += 1
            hint = '신체진찰이 아닙니다. 진찰 동작 하나를 요청하세요' if exc.code == 'EXAM_NOT_RECOGNIZED' else '하나만 구체적으로 다시 요청하세요'
            state['transcript'].append(f"[턴 {state['turn']}] (알림: {exc.code}, 1턴 사용. {hint})")
            continue
        refused = bad = 0
    assessment = assess(client, session, state)
    explain_to_patient(client, session, state, assessment)
    note, diagnosis = write_record(client, session, state, assessment)
    return with_retry(client.diagnose, note=note, diagnosis=diagnosis)


if __name__ == '__main__':
    run_agent(prepare, consult)
