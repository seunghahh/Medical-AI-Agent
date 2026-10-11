"""Adapt NOVA client events to the copied Live Clinic UI. Never receives gold data."""
from nova_local import LocalClient, NovaError


class LiveClient(LocalClient):
    def __init__(self, simulator, journal):
        super().__init__(simulator)
        self.journal = journal
        self.attempt = 0
        self.cache = {}
        self.emit('case_start', evidence=[{'id': 'START', 'status': 'OBSERVED', 'source': 'initial',
                                         'value': simulator.public_session()['initial']}])

    def emit(self, kind, **data):
        self.journal.emit(kind, case_index=self.simulator.case['case_id'], **data)

    def chat(self, messages, **kwargs):
        self.attempt += 1
        system = messages[0]['content']
        kind = 'rag_query' if '검색어' in system else 'record_start' if '초진 기록을 작성' in system else 'thinking'
        self.emit(kind, attempt=self.attempt, turn=self.simulator.turns)
        return super().chat(messages, **kwargs)

    def action(self, action, **kwargs):
        import json
        request_id = kwargs.get('request_id')
        serialized = json.dumps({'action': action, **kwargs}, ensure_ascii=False, sort_keys=True)
        if request_id and request_id in self.cache:
            previous, result = self.cache[request_id]
            if previous != serialized:
                raise NovaError('REQUEST_ID_CONFLICT', 409)
            return result
        item = {'action': action, 'text': kwargs.get('content', '')}
        self.emit('action', action=item, turn=self.simulator.turns)
        if action != 'DIAGNOSE':
            self.emit('resolving', action=item, turn=self.simulator.turns)
        try:
            reply = super().action(action, **kwargs)
        except NovaError as exc:
            self.emit('rejected', reason=exc.code, turn=self.simulator.turns)
            raise
        if request_id:
            self.cache[request_id] = (serialized, reply)
        if action == 'DIAGNOSE':
            self.emit('diagnosis', action={'action': action, 'diagnosis': kwargs['diagnosis'],
                'basis': [kwargs['note']['A']['evidence']], 'note': kwargs['note']}, turn=self.simulator.turns,
                status='submitted')
        else:
            value = reply.get('content', reply.get('result', ''))
            unknown = '미확인' in value or 'UNKNOWN' in value
            self.emit('observation', turn=self.simulator.turns, new_information=not unknown,
                      evidence=[{'id': f'T{self.simulator.turns}', 'status': 'UNKNOWN' if unknown else 'OBSERVED',
                                 'source': action + ': ' + kwargs['content'], 'value': value}])
        return reply
