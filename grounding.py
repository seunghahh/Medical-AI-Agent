"""Source-bound patient facts: no model paraphrases in S/O or diagnostic evidence."""
import re
import unicodedata
from nova_sdk import empty_record

ONES = {'일': 1, '이': 2, '삼': 3, '사': 4, '오': 5, '육': 6, '칠': 7, '팔': 8, '구': 9}
NATIVE = {'한': 1, '하나': 1, '두': 2, '둘': 2, '세': 3, '셋': 3, '석': 3, '네': 4, '넷': 4,
          '넉': 4, '다섯': 5, '여섯': 6, '일곱': 7, '여덟': 8, '아홉': 9, '열': 10,
          '스무': 20, '스물': 20, '서른': 30, '마흔': 40, '쉰': 50, '예순': 60,
          '일흔': 70, '여든': 80, '아흔': 90}
UNITS = {'살': '세', '세': '세', '킬로그램': 'kg', '키로그램': 'kg', '키로그람': 'kg',
         '키로': 'kg', '킬로': 'kg', 'kg': 'kg', '㎏': 'kg', '센티미터': 'cm', 'cm': 'cm'}
MEASURE = re.compile(r'(?<![가-힣\w])([가-힣]+|\d+(?:\.\d+)?)\s*(' + '|'.join(UNITS) +
                     r')(?=$|[^가-힣a-zA-Z]|에서|으로|이다|이고|이며|입니다|이었|였|에|의|은|는|을|를|이|가|도|로|와|과|만|부터|까지)')
MISSING = re.compile(r'미확인|UNKNOWN|모르|몰라|기억.{0,5}안|자료.{0,8}없|미제공|not (?:recorded|provided|known)', re.I)


def korean_number(word):
    if word in NATIVE:
        return NATIVE[word]
    for tens, value in NATIVE.items():
        rest = word[len(tens):] if word.startswith(tens) else ''
        if value >= 10 and rest in NATIVE and NATIVE[rest] < 10:
            return value + NATIVE[rest]
    if not word or any(c not in '일이삼사오육칠팔구십백천' for c in word):
        return None
    total = digit = 0
    previous = 10000
    for char in word:
        if char in ONES:
            if digit:
                return None
            digit = ONES[char]
        else:
            unit = {'십': 10, '백': 100, '천': 1000}[char]
            if unit >= previous:
                return None
            total += (digit or 1) * unit
            previous, digit = unit, 0
    return total + digit


def normalize_measurements(text):
    def replace(match):
        value, unit = match.groups()
        number = value if value[0].isdigit() else korean_number(value)
        return f'{number}{UNITS[unit]}' if number is not None else match.group()
    return MEASURE.sub(replace, unicodedata.normalize('NFC', text))


ASKS = re.compile(r'나요|습니까|세요|한가요|알려|말씀해|말해')


def asks(content):
    """True when a doctor line reads as a question, so the patient's reply is kept as an observation."""
    return '?' in content or bool(ASKS.search(content))


def sources(session, transcript):
    initial = session['initial']
    items = [{'id': 'opening', 'kind': 'opening', 'question': '', 'raw': initial['opening']},
             {'id': 'vitals', 'kind': 'exam', 'question': '시작 활력징후', 'raw': initial['vital_signs']}]
    question = None
    for line in transcript:
        match = re.match(r'턴 (\d+) 의사: (.*)', line, re.S)
        if match:
            question = match.groups()
        elif line.startswith('환자: ') and question:
            turn, content = question
            asked = asks(content)
            consent = re.search(r'검사|치료|처방|의뢰|복용', content) and re.search(
                r'할까요|하시겠|하실래|하겠습니다|진행|요청|추가|검사\?$', content)
            # shortcut: Intent uses wording patterns; keep Q/A provenance for ambiguous questions.
            asked = asked and not consent
            items.append({'id': f'turn{turn}', 'kind': 'say' if asked else 'communication',
                          'question': content, 'raw': line[4:]})
            question = None
        else:
            match = re.match(r'턴 (\d+) 진찰\((.*?)\): (.*)', line, re.S)
            if match:
                turn, content, answer = match.groups()
                items.append({'id': f'turn{turn}', 'kind': 'exam', 'question': content, 'raw': answer})
            question = None
    for item in items:
        item['value'] = unicodedata.normalize('NFC', item['raw'])
    return items


def render(item):
    prefix = f'[{item["id"]}] '
    return prefix + (f'질문/요청: {item["question"]}\n답변: ' if item['question'] else '') + item['value']


def source_record(items):
    note = empty_record()
    note['S']['cc'] = render(items[0])
    note['S']['hpi'] = render(items[0])
    buckets = {'pmh': [], 'fh': [], 'sh': [], 'ros': []}
    exams = []
    for item in items[1:]:
        if item['kind'] == 'exam':
            exams.append(render(item))
        elif item['kind'] == 'say':
            question = item['question']
            if re.search(r'가족|부모|어머니|아버지|family|mother|father', question, re.I):
                key = 'fh'
            elif re.search(r'과거|병력|약|복용|알레르기|수술|past|medication|allerg|surgery', question, re.I):
                key = 'pmh'
            elif re.search(r'흡연|담배|음주|술|직업|운동|smok|alcohol|occupation', question, re.I):
                key = 'sh'
            else:
                key = 'ros'
                note['S']['hpi'] += '\n' + render(item)
            buckets[key].append(render(item))
    for field, values in buckets.items():
        note['S'][field] = '\n'.join(values) or '미확인 (확보된 응답 없음)'
    if not any(re.search(r'알레르기|allerg', item['question'], re.I) for item in items if item['kind'] == 'say'):
        note['S']['pmh'] += '\n알레르기: 미확인 (질문하지 않음)'
    note['O']['exam'] = '\n'.join(exams)
    if not any(item['kind'] == 'exam' and item['id'] != 'vitals' for item in items):
        note['O']['exam'] += '\n추가 신체진찰: 미시행'
    return note


def measurements(text):
    return {(number, UNITS[unit]) for number, unit in
            re.findall(r'(\d+(?:\.\d+)?)\s*(세|살|kg|cm)', normalize_measurements(text))}


def unsafe_generated(text, items):
    # shortcut: Pattern checks are not clinical verification; A/P still need independent review.
    observed = '\n'.join(item['value'] for item in items)
    if measurements(text) - measurements(observed):
        return '관찰에 없는 나이/체중/키 수치'
    if re.search(r'\(-\)|음성|무병력|(?:흡연|알레르기|발열|과거력|가족력|식욕|소견).{0,15}(?:없|정상)', text):
        return '계획에 포함된 환자 음성/정상 단정'
    return ''


def plan_clauses(value):
    """Keep safe plan clauses when a separate clause contains an unsupported claim."""
    return [part.strip() for part in re.split(r'(?<=[.!?。])\s+|[;；\n]+', value) if part.strip()]


def unsafe_plan_clause(text, items):
    # Conditional future growth rates are decision thresholds, not measured patient height.
    checked = re.sub(
        r'\b(?:if|when|unless)\b[^.!?;\n]*\b(?:growth|height)\s+(?:velocity|rate)'
        r'[^.!?;\n]*?\d+(?:\.\d+)?\s*cm\s*(?:/|per\s+)(?:year|yr|month)s?\b',
        '', text, flags=re.I)
    checked = re.sub(
        r'(?:성장\s*속도|성장률)[^.!?;\n]*?\d+(?:\.\d+)?\s*cm\s*/\s*(?:년|연|월)'
        r'\s*(?:미만|이하|이상|초과)(?:이면|라면|일 때|인 경우)', '', checked)
    return unsafe_generated(checked, items)


def finalize(candidate, items):
    note = source_record(items)
    blocked = []
    diagnosis = candidate.get('diagnosis', '')
    diagnosis = ' '.join(diagnosis.split()) if isinstance(diagnosis, str) else ''
    if not diagnosis or len(diagnosis.encode()) > 500 or unsafe_generated(diagnosis, items):
        diagnosis = '진단 미정'
    # Assessment quotes observations verbatim; differential names remain hypotheses.
    facts = [i for i in items if i['kind'] != 'communication']
    note['A']['evidence'] = '#1. ' + diagnosis + ' (추정)\n관찰 근거:\n' + '\n'.join(render(i) for i in facts)
    differential = candidate.get('differential', [])
    by_id = {i['id']: i for i in facts}
    if isinstance(differential, list):
        for entry in differential[:5]:
            if not isinstance(entry, dict):
                continue
            name, ids = entry.get('name'), entry.get('source_ids')
            if (isinstance(name, str) and name.strip() and not unsafe_generated(name, items)
                    and isinstance(ids, list) and ids and all(isinstance(i, str) and i in by_id for i in ids)):
                note['A']['differential'] += f'r/o {name.strip()} (가설)\n' + '\n'.join(render(by_id[i]) for i in ids) + '\n'
    plans = candidate.get('P', {})
    if isinstance(plans, dict):
        for field in note['P']:
            value = plans.get(field, '')
            if not isinstance(value, str):
                continue
            accepted = []
            for clause in plan_clauses(value):
                reason = unsafe_plan_clause(clause, items)
                if reason:
                    blocked.append({'field': 'P.' + field, 'reason': reason, 'rejected': clause})
                else:
                    accepted.append(clause)
            if accepted:
                note['P'][field] = '[계획/제안, 시행 결과 아님] ' + ' '.join(accepted)
    return note, diagnosis, blocked


def retrieval_sources(items):
    catalog = []
    for item in items:
        if item['kind'] == 'communication' or MISSING.search(item['value']):
            continue
        parts = re.split(r'(?<=[.!?。])\s+', item['value']) if item['kind'] == 'opening' else [item['value']]
        for n, value in enumerate(parts):
            if value.strip():
                catalog.append({**item, 'id': item['id'] + f'.s{n}', 'value': value})
    return catalog


def query_spans(candidate, items):
    by_id = {i['id']: i for i in items}
    accepted, rejected = [], []
    for entry in candidate if isinstance(candidate, list) else []:
        if not isinstance(entry, dict):
            continue
        source_id, quote = entry.get('source_id'), entry.get('quote')
        item = by_id.get(source_id) if isinstance(source_id, str) else None
        # Keep each full sentence/response so a selected span cannot drop negation or qualifiers.
        if (item and isinstance(quote, str) and quote.strip()
                and not MISSING.search(item['value']) and quote == item['value']):
            accepted.append({'source_id': source_id, 'quote': quote})
        else:
            rejected.append(entry)
    return accepted[:8], rejected
