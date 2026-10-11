"""Current public contract and offline evaluator export regressions."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run
from local import SessionSimulator
from nova_local import LocalClient
from nova_sdk import NovaError
from test_contract import Endpoint, case
from export_review import export


class AuditTests(unittest.TestCase):
    def test_refusal_consumes_last_turn_exactly_once(self):
        sim = SessionSimulator(case(), doctor=Endpoint(), sim=Endpoint())
        sim.turns = 49
        body = {'action': 'EXAM', 'content': '모두 진찰', 'request_id': 'refused'}
        with patch.object(sim, 'examine', side_effect=NovaError('EXAM_NOT_SPECIFIC', 422)) as examiner:
            for _ in range(2):
                with self.assertRaisesRegex(NovaError, 'EXAM_NOT_SPECIFIC'):
                    sim.action(body)
            self.assertEqual(examiner.call_count, 1)
        self.assertEqual(sim.turns, 50)
        self.assertEqual(sim.events[-1]['refused'], 'EXAM_NOT_SPECIFIC')
        with self.assertRaisesRegex(NovaError, 'TURN_LIMIT_REACHED'):
            LocalClient(sim).say('새 질문')

    def test_consult_respects_small_budget_after_refused_exam(self):
        doctor = Endpoint([{'action': 'EXAM', 'content': '모두 진찰'}, {'diagnosis': '미정'}])
        sim = SessionSimulator(case(), doctor=doctor, sim=Endpoint())
        with patch.object(sim, 'examine', side_effect=NovaError('EXAM_NOT_SPECIFIC', 422)):
            result = run.consult(LocalClient(sim), {'max_turns': 1})
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(sim.turns, 1)
        self.assertEqual(sim.doctor_calls, 2)

    def test_model_failure_does_not_consume_turn_and_gets_new_request(self):
        ids = []
        def call(**kwargs):
            ids.append(kwargs['request_id'])
            if len(ids) == 1:
                raise NovaError('PATIENT_RESPONSE_INVALID', 503)
            return 'ok'
        with patch.object(run.time, 'sleep'):
            self.assertEqual(run.with_retry(call, request_id='first'), 'ok')
        self.assertNotEqual(ids[0], ids[1])
        sim = SessionSimulator(case(), doctor=Endpoint(), sim=Endpoint())
        with patch.object(sim, 'patient', side_effect=NovaError('PATIENT_RESPONSE_INVALID', 503)):
            with self.assertRaises(NovaError):
                LocalClient(sim).say('질문')
        self.assertEqual(sim.turns, 0)

    def test_blank_doctor_counts_call_but_not_tokens(self):
        sim = SessionSimulator(case(), doctor=Endpoint(), sim=Endpoint())
        with patch.object(sim.doctor, 'complete', return_value=('', 'length', 100, 200)):
            with self.assertRaisesRegex(NovaError, 'PROVIDER_RESPONSE_INVALID'):
                LocalClient(sim).chat([{'role': 'user', 'content': 'hello'}])
        self.assertEqual((sim.doctor_calls, sim.doctor_input, sim.doctor_output), (1, 0, 0))

    def test_sample_path_enforces_preliminary_record(self):
        from nova_sdk import empty_record
        sim = SessionSimulator(case(), doctor=Endpoint(), sim=Endpoint())
        with self.assertRaisesRegex(NovaError, 'DOCTOR_CALL_REQUIRED'):
            LocalClient(sim).diagnose(note=empty_record(), diagnosis='미정')
        sim.doctor_calls = 1
        note = empty_record(); note['O']['tests'] = 'invented'
        with self.assertRaisesRegex(NovaError, 'PRELIMINARY_TEST_RESULTS_FORBIDDEN'):
            LocalClient(sim).diagnose(note=note, diagnosis='미정')

    def test_diagnose_wire_format_and_record_limits(self):
        from nova_sdk import Client, empty_record
        note = empty_record(); note['S']['cc'] = '체중 감소'
        client = Client(api_base='http://localhost', token='offline-test')
        with patch.object(client, '_request', return_value={'status': 'completed'}) as request:
            client.diagnose(note=note, diagnosis='진단 가설', request_id='final')
            method, path, body = request.call_args.args
            self.assertEqual((method, path), ('POST', '/v1/session/actions'))
            self.assertEqual(set(body), {'action', 'request_id', 'note', 'diagnosis'})
            self.assertEqual(body['action'], 'DIAGNOSE')
            with self.assertRaises(ValueError):
                client.diagnose(note={**note, 'diagnosis': '잘못된 위치'}, diagnosis='진단 가설')
            self.assertEqual(request.call_count, 1)
        sim = SessionSimulator(case(), doctor=Endpoint(), sim=Endpoint())
        sim.doctor_calls = 1
        for diagnosis, error in [('가' * 167, 'RECORD_TOO_LARGE'), ('진단1\n진단2', 'SINGLE_DIAGNOSIS_REQUIRED')]:
            with self.assertRaisesRegex(NovaError, error):
                LocalClient(sim).diagnose(note=note, diagnosis=diagnosis)
        note['S']['cc'] = '가' * 34_000
        with self.assertRaisesRegex(NovaError, 'RECORD_TOO_LARGE'):
            LocalClient(sim).diagnose(note=note, diagnosis='진단 가설')

    def test_review_bundle_requires_evidence_and_keeps_reference_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'Missing'):
                export(root)
            for name, data in [('record.json', {'diagnosis': 'hypothesis'}), ('transcript.json', {'events': []}),
                               ('summary.json', {'official_score': False})]:
                (root / name).write_text(json.dumps(data))
            reference = root / 'reference.txt'; reference.write_text('REFERENCE_ONLY')
            text = export(root, reference=reference).read_text()
            self.assertIn('Evaluator-only reference', text)
            self.assertIn('not official scores', text)
            self.assertIn('REFERENCE_ONLY', text)


if __name__ == '__main__':
    unittest.main()
