"""NOVA SDK 1.0: Python standard library only, one session per invocation."""
from __future__ import annotations

import json
import os
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import uuid

__version__ = '1.0.1'
# Cloudflare's browser integrity check in front of nova.snubhai.org refuses urllib's default User-Agent
# (HTTP 403, "error code: 1010"); every request carries this one instead (2026-10-12).
USER_AGENT = 'nova-sdk/' + __version__


# The initial clinical record: free text in each of these S/O/A/P fields (an empty string is allowed).
# The main diagnosis is the separate `diagnosis` argument of Client.diagnose.
RECORD_FIELDS = {
    'S': ('cc', 'hpi', 'pmh', 'fh', 'sh', 'ros'),      # 주호소, 현병력, 과거력·약물·알레르기, 가족력, 사회력, 계통 문진
    'O': ('exam', 'tests'),                              # 진찰 소견, 검사 결과
    'A': ('evidence', 'differential'),                   # 근거, 감별진단
    'P': ('tests', 'treatment', 'disposition', 'follow_up', 'education'),  # 추가 검사, 치료, 처치 결정, 추적, 환자 교육
}


def empty_record():
    """A record with every field present and empty, to fill in."""
    return {section: {field: '' for field in fields} for section, fields in RECORD_FIELDS.items()}


class NovaError(RuntimeError):
    def __init__(self, code, status=None):
        self.code = code
        self.status = status
        super().__init__(code)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Client:
    def __init__(self, api_base=None, token=None, session_id=None, *, sample=False, timeout=240):
        self.api_base = (api_base or os.environ['NOVA_API_BASE']).rstrip('/')
        self.token = token or os.environ['NOVA_RUN_TOKEN']
        self.session_id = str(session_id or os.getenv('NOVA_SESSION_ID', ''))
        self.sample = sample
        self.timeout = timeout
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
        parsed = urllib.parse.urlsplit(self.api_base)
        if parsed.scheme not in ('http', 'https') or not parsed.netloc or parsed.query or parsed.fragment or parsed.username:
            raise ValueError('NOVA_API_BASE must be an HTTP origin')

    def _open(self, request):
        return self._opener.open(request, timeout=self.timeout)

    def _request(self, method, path, body=None):
        request = urllib.request.Request(
            self.api_base + path, method=method,
            data=json.dumps(body, ensure_ascii=False).encode('utf-8') if body is not None else None,
            headers={'Authorization': 'Bearer ' + self.token, 'Content-Type': 'application/json',
                     'User-Agent': USER_AGENT},
        )
        try:
            response = self._open(request)
        except urllib.error.HTTPError as exc:
            response = exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise NovaError('API_UNAVAILABLE') from exc
        with response:
            raw = response.read(2_000_001)
            if len(raw) > 2_000_000:
                raise NovaError('RESPONSE_TOO_LARGE')
            try:
                payload = json.loads(raw)
            except (ValueError, UnicodeError) as exc:
                raise NovaError('INVALID_API_RESPONSE', response.status) from exc
            if not 200 <= response.status < 300:
                error = payload.get('error', {}) if isinstance(payload, dict) else {}
                raise NovaError(error.get('code', 'API_ERROR'), response.status)
            return payload

    def ready(self):
        """Sandbox handshake, called after prepare() and before case access."""
        if self.sample:
            return {'ready': True}
        return self._request('POST', '/_nova/ready', {})

    def create_sample(self, case_id, *, request_id=None):
        if not self.sample:
            raise ValueError('Sample sessions require a development client')
        session = self._request('POST', '/v1/sample/sessions', {'case_id': str(case_id), 'request_id': request_id or str(uuid.uuid4())})
        self.session_id = str(session['session_id'])
        return session

    def session(self):
        if self.sample:
            if not self.session_id:
                raise ValueError('Create a public sample session first')
            path = '/v1/sample/sessions/' + str(uuid.UUID(self.session_id))
        else:
            path = '/v1/session'
        return self._request('GET', path)

    def action(self, action, *, request_id=None, **fields):
        body = {'request_id': request_id or str(uuid.uuid4()), 'action': action, **fields}
        if action in ('SAY', 'EXAM', 'TEST'):
            text = unicodedata.normalize('NFC', str(body.get('content', '')))
            if not text or len(text) > 30:
                raise ValueError('SAY, EXAM and TEST require 1–30 NFC characters')
            body['content'] = text
        path = '/v1/session/actions'
        if self.sample:
            path = '/v1/sample/sessions/' + str(uuid.UUID(self.session_id)) + '/actions'
        return self._request('POST', path, body)

    def say(self, content, *, request_id=None):
        return self.action('SAY', content=content, request_id=request_id)

    def exam(self, content, *, request_id=None):
        """One specific maneuver in words, e.g. '복부 청진하겠습니다'. There is no exam code list."""
        return self.action('EXAM', content=content, request_id=request_id)

    def test(self, content, *, request_id=None):
        """Finals only. One specific test in words, e.g. '간기능검사 하겠습니다'. There is no test list."""
        return self.action('TEST', content=content, request_id=request_id)

    def diagnose(self, *, note, diagnosis, request_id=None):
        """Submit the S/O/A/P record (see empty_record) and one main diagnosis together."""
        if (not isinstance(note, dict) or set(note) != set(RECORD_FIELDS) or any(
                not isinstance(note[s], dict) or set(note[s]) != set(f) or not all(isinstance(v, str) for v in note[s].values())
                for s, f in RECORD_FIELDS.items())):
            raise ValueError('note must have every S/O/A/P field as a string; start from empty_record()')
        if not isinstance(diagnosis, str) or not diagnosis.strip():
            raise ValueError('One main diagnosis is required')
        return self.action('DIAGNOSE', note=note, diagnosis=diagnosis, request_id=request_id)

    def chat(self, messages, *, max_tokens=2048, reasoning_effort=None, temperature=None, model=None, request_id=None):
        # The Doctor model counts its reasoning in max_tokens; a small budget can end before the answer starts.
        # reasoning_effort: 'low', 'medium' or 'high' (None = model default). Higher thinks longer and costs more.
        # temperature: 0-2 (None = model default).
        # model: None = the Doctor (openai/gpt-6-luna since 2026-10-13; the old id openai/gpt-oss-20b still means the Doctor). With a development key and no session, also
        # 'openai/gpt-6.1-sol' or 'openai/gpt-6-luna' (reasoning_effort none..xhigh, no temperature), paid from team credit.
        body = {'request_id': request_id or str(uuid.uuid4()), 'messages': messages, 'max_tokens': max_tokens}
        if model is not None:
            body['model'] = model
        if reasoning_effort is not None:
            body['reasoning_effort'] = reasoning_effort
        if temperature is not None:
            body['temperature'] = temperature
        if self.session_id:
            body['session_id'] = self.session_id
        return self._request('POST', '/v1/chat/completions', body)


def run_agent(prepare, consult):
    """prepare() -> state; consult(client, state) -> result. Preparation has no case access."""
    client = Client()
    state = prepare()
    client.ready()
    return consult(client, state)
