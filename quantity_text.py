"""Format explicit quantities without inferring values, rates, or missing facts."""
import re
import unicodedata

_DIGITS = dict(zip('영공일이삼사오육칠팔구', '00123456789'))
_NATIVE = dict(zip(('한', '하나', '두', '둘', '세', '셋', '네', '넷', '다섯', '여섯',
                   '일곱', '여덟', '아홉', '열', '스무', '스물', '서른', '마흔', '쉰',
                   '예순', '일흔', '여든', '아흔'),
                  (1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8, 9, 10, 20, 20, 30, 40, 50,
                   60, 70, 80, 90)))
_NATIVE.update({'석':3,'넉':4})
_NATIVE_WORDS = set(_NATIVE) | {a+b for a,x in _NATIVE.items() for b,y in _NATIVE.items() if x>=10 and y<10}
_KOREAN_NUMBER = r'(?:'+'|'.join(sorted(_NATIVE_WORDS,key=len,reverse=True))+r'|[영공일이삼사오육칠팔구십백천]+)'
_UNITS = {'센티미터': 'cm', '킬로그램': 'kg', '키로그램': 'kg', '키로그람': 'kg',
          '밀리그램': 'mg', '밀리리터': 'mL', '㎏': 'kg', '㎝': 'cm',
          '주일':'주',
          **{unit: unit for unit in ('cm', 'kg', 'mg', 'mL', '세', '살', '년', '개월',
                                     '달', '해', '주', '일', '시간', '시', '분', '초', '회', '번', '알', '정', '잔')}}
_BOUNDARY = r'(?=$|[^가-힣A-Za-z]|예요|에요|라고|라는|씩|정도|쯤|에서|으로|이고|이며|입니다|인데|임|이었|였|에|의|은|는|을|를|이|가|도|로|와|과|만|부터|까지|간|동안|째|경|가량|내외|전|후)'
_QUANTITY = re.compile(r'(?<![가-힣\w.])(?P<number>'+_KOREAN_NUMBER+r'(?:\s*점\s*[영공일이삼사오육칠팔구]+)?|[+-]?\d+(?:\.\d+)?)\s*'
                       r'(?P<unit>' + '|'.join(sorted(_UNITS, key=len, reverse=True)) + ')' + _BOUNDARY)


def _integer(word):
    if word in _NATIVE:
        return _NATIVE[word]
    for prefix, value in _NATIVE.items():
        if value >= 10 and word.startswith(prefix) and word[len(prefix):] in _NATIVE:
            tail = _NATIVE[word[len(prefix):]]
            if tail < 10:
                return value + tail
    if word in ('영', '공'):
        return 0
    if not word or any(c not in '일이삼사오육칠팔구십백천' for c in word):
        return None
    total = digit = 0
    previous = 10000
    for char in word:
        if char in _DIGITS:
            if digit:
                return None
            digit = int(_DIGITS[char])
        else:
            scale = {'십': 10, '백': 100, '천': 1000}[char]
            if scale >= previous:
                return None
            total += (digit or 1) * scale
            previous, digit = scale, 0
    return total + digit


def _number(word):
    if re.fullmatch(r'[+-]?\d+(?:\.\d+)?', word):
        return word
    parts = re.split(r'\s*점\s*', word)
    whole = _integer(parts[0])
    if whole is None:
        return None
    if len(parts) == 1:
        return str(whole)
    if len(parts) == 2 and parts[1] and all(c in _DIGITS for c in parts[1]):
        return str(whole) + '.' + ''.join(_DIGITS[c] for c in parts[1])
    return None


def normalize_quantities(text):
    def replace(match):
        # '이번에는' means 'this time', not a count of two repetitions.
        if match[0]=='이번':
            return match[0]
        value = _number(match['number'])
        return value + _UNITS[match['unit']] if value is not None else match[0]
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'(?<![가-힣])반\s*(?=알|정)', '0.5', text)
    # Protect approximate native counts before '두세 번' can be misread as '두 세'.
    ranges={'한두':'1~2','두세':'2~3','서너':'3~4','네다섯':'4~5','대여섯':'5~6'}
    text = re.sub(r'(?<![가-힣])('+'|'.join(ranges)+r')\s*(?=잔|알|정|번|회|달|개월|일|주|시간|시|세|살)',
                  lambda m: ranges[m[1]], text)
    return _QUANTITY.sub(replace, text)


def quantities(text):
    return {(m['number'], _UNITS[m['unit']]) for m in _QUANTITY.finditer(normalize_quantities(text))
            if re.fullmatch(r'[+-]?\d+(?:\.\d+)?', m['number'])}
