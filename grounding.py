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
        # '세' is an age only after a digit or a Sino-Korean number (삼십이 세); after a native number it counts things
        # ('두세 번' = two or three times, '세 개' = three), so leave those alone.
        if unit == '세' and not value[0].isdigit() and any(c not in '일이삼사오육칠팔구십백천' for c in value):
            return match.group()
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
        item['value'] = normalize_measurements(item['raw'])
    return items


def render(item):
    """One compact line for an observation: what was asked, what came back, and where it came from."""
    value = item['value'].replace('\n', ' / ')
    body = f'{item["question"]} → {value}' if item['question'] else value
    return f'- {body} ({label(item["id"])})'


def label(source_id):
    """Short citation for a source id: opening.s1 -> 첫 진술, vitals -> 활력징후, turn12 -> T12."""
    base = source_id.split('.')[0]
    return '첫 진술' if base == 'opening' else '활력징후' if base == 'vitals' else 'T' + base[4:] if base.startswith('turn') else base


def readable_ids(text):
    """Models sometimes write internal ids into plans ('근거: turn2, vitals'); show them as the record's citations."""
    text = re.sub(r'\bturn(\d+)\b', r'T\1', text)
    text = re.sub(r'\bopening(?:\.s\d+)?\b', '첫 진술', text)
    return re.sub(r'\bvitals\b', '활력징후', text)


def cite(ids):
    return ', '.join(dict.fromkeys(label(i) for i in ids))


def record_sources(items):
    """What the record-writing model sees: observations only, the opening split into sentences, no duplicate raw text."""
    shown = []
    for item in items:
        if item['kind'] == 'communication':
            continue
        if item['id'] == 'opening':
            parts = [p for p in re.split(r'(?<=[.!?。])\s+', item['value']) if p.strip()]
            shown += [{'id': f'opening.s{n}', 'kind': 'opening', 'value': p} for n, p in enumerate(parts)]
        else:
            shown.append({k: item[k] for k in ('id', 'kind', 'question', 'value') if item.get(k)})
    return shown


def source_record(items, plan=None):
    """S and O copied from the observations. `plan` (already validated) only chooses which opening sentences are the
    chief complaint and supplies a short history summary; every other word comes from what the patient said."""
    plan = plan or {}
    note = empty_record()
    opening = items[0]
    note['S']['cc'] = plan.get('cc') or render(opening)
    buckets = {'pmh': [], 'fh': [], 'sh': [], 'ros': []}
    exams, hpi_lines = [], []
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
                hpi_lines.append(render(item))
            buckets[key].append(render(item))
    if plan.get('hpi'):
        # A summary replaces the repeated Q/A in the history; the questions stay once, in ROS.
        note['S']['hpi'] = plan['hpi'] + '\n' + render(opening)
    else:
        note['S']['hpi'] = '\n'.join([render(opening)] + hpi_lines)
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


NEGATIVE_TEXT = re.compile(r'\(-\)|음성|무병력|(?:흡연|알레르기|발열|과거력|가족력|식욕|소견).{0,15}(?:없|정상)')
NEGATIVE_ANSWER = re.compile(r'없|않|아니|못|안 |\(-\)')


def unsupported_claim(text, ids, by_id, items):
    """Why a model-written sentence cannot go into the record ('' = fine). It must cite real observations, introduce
    no measurement or value the patient did not give (numbers above 12 and decimals must appear in the sources),
    and may call something absent or normal only if a cited answer says so. Meaning is not checked."""
    if not isinstance(text, str) or not text.strip() or len(text) > 400:
        return '형식'
    if not isinstance(ids, list) or not ids or not all(isinstance(i, str) and i.split('.')[0] in by_id for i in ids):
        return '출처 ID'
    observed = '\n'.join(item['value'] for item in items if item['kind'] != 'communication')
    if measurements(text) - measurements(observed):
        return '관찰에 없는 나이/체중/키 수치'
    seen = set(re.findall(r'\d+(?:\.\d+)?', normalize_measurements(observed)))
    for number in re.findall(r'\d+(?:\.\d+)?', text):
        if number not in seen and ('.' in number or int(number) > 12):
            return '관찰에 없는 수치'
    cited = [by_id[i.split('.')[0]] for i in ids]
    if NEGATIVE_TEXT.search(text) and not any(NEGATIVE_ANSWER.search(c['value']) for c in cited):
        return '근거 없는 음성/정상 단정'
    return ''


def writing_plan(candidate, items, blocked):
    """Accept the parts of the model's record that pass the checks; the rest falls back to the plain observations."""
    facts = [i for i in items if i['kind'] != 'communication']
    by_id = {i['id']: i for i in facts}
    plan = {}
    opening = {f'opening.s{n}': p for n, p in enumerate(
        p for p in re.split(r'(?<=[.!?。])\s+', items[0]['value']) if p.strip())}
    cc = candidate.get('cc')
    if isinstance(cc, list) and cc and all(isinstance(i, str) and i in opening for i in cc):
        plan['cc'] = ' '.join(opening[i] for i in dict.fromkeys(cc)) + ' (첫 진술)'
    hpi = candidate.get('hpi')
    if isinstance(hpi, dict):
        reason = unsupported_claim(hpi.get('text'), hpi.get('source_ids'), by_id, items)
        if reason:
            blocked.append({'field': 'S.hpi', 'reason': reason, 'rejected': hpi})
        else:
            plan['hpi'] = f'요약: {hpi["text"].strip()} ({cite(hpi["source_ids"])})'
    return plan


def finalize(candidate, items):
    blocked = []
    note = source_record(items, writing_plan(candidate, items, blocked))
    diagnosis = candidate.get('diagnosis', '')
    diagnosis = ' '.join(diagnosis.split()) if isinstance(diagnosis, str) else ''
    if not diagnosis or len(diagnosis.encode()) > 500 or unsafe_generated(diagnosis, items):
        diagnosis = '진단 미정'
    facts = [i for i in items if i['kind'] != 'communication']
    by_id = {i['id']: i for i in facts}
    # Assessment: interpreted bullets that cite their observations; if none survive the checks, the observations themselves.
    bullets = []
    for entry in candidate.get('evidence', []) if isinstance(candidate.get('evidence'), list) else []:
        if not isinstance(entry, dict):
            continue
        reason = unsupported_claim(entry.get('text'), entry.get('source_ids'), by_id, items)
        if reason:
            blocked.append({'field': 'A.evidence', 'reason': reason, 'rejected': entry})
        else:
            bullets.append(f'- {entry["text"].strip()} ({cite(entry["source_ids"])})')
    body = '\n'.join(bullets[:8]) if bullets else '관찰 근거:\n' + '\n'.join(render(i) for i in facts)
    note['A']['evidence'] = '#1. ' + diagnosis + ' (추정)\n' + body
    differential = candidate.get('differential', [])
    if isinstance(differential, list):
        lines = []
        for entry in differential[:5]:
            if not isinstance(entry, dict):
                continue
            name, ids = entry.get('name'), entry.get('source_ids')
            if not (isinstance(name, str) and name.strip() and not unsafe_generated(name, items)
                    and isinstance(ids, list) and ids and all(isinstance(i, str) and i.split('.')[0] in by_id for i in ids)):
                continue
            why = entry.get('reason')
            if isinstance(why, str) and why.strip():
                problem = unsupported_claim(why, ids, by_id, items)
                if not problem:
                    lines.append(f'r/o {name.strip()} : {why.strip()} ({cite(ids)})')
                    continue
                blocked.append({'field': 'A.differential', 'reason': problem, 'rejected': entry})
            lines.append(f'r/o {name.strip()} (가설)\n' + '\n'.join(render(by_id[i.split('.')[0]]) for i in dict.fromkeys(ids)))
        note['A']['differential'] = '\n'.join(lines)
    plans = candidate.get('P', {})
    if isinstance(plans, dict):
        for field in note['P']:
            value = plans.get(field, '')
            if not isinstance(value, str):
                continue
            reason = unsafe_generated(value, items)
            if reason:
                blocked.append({'field': 'P.' + field, 'reason': reason, 'rejected': value})
            elif value.strip():
                note['P'][field] = '[계획/제안, 시행 결과 아님] ' + readable_ids(value.strip())
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
