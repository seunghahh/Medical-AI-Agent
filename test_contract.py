"""Offline regression tests: boundaries, adapters, retries, retrieval and record output."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run
from benchmarks import load_rows, decode_ddx
from local import BenchmarkSimulator, UNKNOWN
from nova_local import LocalClient
from nova_sdk import empty_record, NovaError
from rag import GuidelineIndex, metadata


class Endpoint:
    model = 'test-only'

    def __init__(self, replies=()):
        self.replies = iter(replies)
        self.messages = []

    def complete(self, messages, **kwargs):
        self.messages.append(messages)
        return json.dumps(next(self.replies), ensure_ascii=False), 'stop', 10, 10


class PhaseEndpoint(Endpoint):
    """Offline replies for each public consultation phase; no hidden case access."""
    def complete(self, messages, **kwargs):
        self.messages.append(messages)
        system = messages[0]['content']
        if system.startswith('You are interviewing a patient'):
            reply = next(self.replies, {'action': 'DONE'})
        elif 'catalog=' in system:
            reply = {'ids': []}
        elif system.startswith(('Choose the most likely causal', 'Review the draft causal')):
            reply = {'diagnosis': 'provisional', 'reason': 'Observed complaint; cause needs confirmation.',
                     'differential': [], 'O': {'tests': 'invented test'}}
        elif system.startswith('Write a concise prospective clinical plan'):
            reply = {'P': {'tests': 'Tests to confirm the cause', 'treatment': 'Review medicines before treatment',
                           'disposition': 'Review', 'follow_up': 'Reassess in 2 weeks.',
                           'education': 'Severe pain requires urgent care'}}
        elif system.startswith('You translate a single clinical message'):
            if '위험 증상' in system:
                text = '심해지면 바로 진료를 받아요.'
            elif '검사 하나' in system:
                text = '검사로 원인을 확인해요.'
            else:
                text = '원인은 더 확인이 필요해요.'
            reply = {'text': text}
        else:
            raise AssertionError('Unexpected consultation phase: ' + system[:80])
        return json.dumps(reply, ensure_ascii=False), 'stop', 10, 10


def case():
    return {'case_id': 'fixture', 'start_information': {'opening': '배가 아파요', 'vital_signs': '미제공'},
            'patient_script': {'History': 'HIDDEN_HISTORY'},
            'exam_results': [{'exam': 'abdominal palpation', 'result': 'HIDDEN_EXAM'}]}


def simulator(replies=()):
    return BenchmarkSimulator(case(), doctor=Endpoint(), sim=Endpoint(replies), minutes=20)


class ContractTests(unittest.TestCase):
    def test_manual_sample_commands(self):
        from develop import manual
        from unittest.mock import Mock
        client = Mock()
        with patch('builtins.input', side_effect=['s 언제부터 아팠나요?', 'e 복부 촉진', 'q']), patch('builtins.print'):
            manual(client)
        client.say.assert_called_once_with('언제부터 아팠나요?')
        client.exam.assert_called_once_with('복부 촉진')
        client.diagnose.assert_not_called()

    def test_live_monitor_events_do_not_expose_hidden_case(self):
        from live import Journal, snapshot
        from monitor import LiveClient
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            with Journal(directory, {'model': 'test-only'}, directory=directory / '.live') as journal:
                sim = simulator([{'content': '어제부터요'}])
                client = LiveClient(sim, journal)
                client.say('언제부터 아팠나요?', request_id='one')
                client.say('언제부터 아팠나요?', request_id='one')
                events = snapshot(directory=directory / '.live')['events']
                self.assertEqual([e['type'] for e in events], ['run_start', 'case_start', 'action', 'resolving', 'observation'])
                self.assertNotIn('HIDDEN', json.dumps(events))
                cursor = events[-1]['seq']
                self.assertEqual(snapshot(journal.run_id, cursor, directory=directory / '.live')['events'], [])

    @unittest.skipUnless((Path(__file__).resolve().parent.parent / 'nova-benchmarks-2026').is_dir(), 'external benchmark integration data not installed')
    def test_cli_outputs_for_each_benchmark(self):
        import bench
        with tempfile.TemporaryDirectory() as tmp:
            for dataset in ('agentclinic', 'ddxplus', 'aci'):
                note = empty_record(); note['S']['cc'] = 'Observed complaint'
                doctor = Endpoint([{**note, 'diagnosis': 'provisional'}]) if dataset == 'aci' else PhaseEndpoint()
                def factory(case, **kwargs):
                    sim = BenchmarkSimulator(case, doctor=doctor, sim=Endpoint(), **kwargs)
                    sim.patient = lambda utterance: ('미확인: 원자료에 해당 정보가 없습니다.', utterance)
                    return sim
                args = SimpleNamespace(command='run', data_root=str(bench.DEFAULT_DATA), dataset=dataset,
                                       split='validate', offset=0, limit=1, max_turns=50, minutes=20, no_rag=True,
                                       port=8768, no_browser=True)
                with patch.object(bench, 'ROOT', Path(tmp)), patch.object(bench, 'BenchmarkSimulator', side_effect=factory), patch('builtins.print'):
                    self.assertEqual(bench.execute(args), 0)
            outputs = list(Path(tmp).glob('results/*/*/evaluation.json'))
            self.assertEqual(len(outputs), 3)
            self.assertEqual(len(list(Path(tmp).glob('results/*/*/reference_note.txt'))), 1)
            for path in outputs:
                result = json.loads(path.read_text())
                self.assertTrue(result['summary']['local_contract_pass'])
                self.assertFalse(result['summary']['official_score'])

    def test_public_boundary_and_test_rejection(self):
        sim = simulator()
        visible = json.dumps(LocalClient(sim).session())
        self.assertNotIn('HIDDEN', visible)
        with self.assertRaisesRegex(NovaError, 'ACTION_NOT_AVAILABLE'):
            LocalClient(sim).test('혈액 검사')
        self.assertEqual(sim.turns, 0)

    def test_nfc_length_turn_limit_and_final_at_50(self):
        sim = simulator()
        client = LocalClient(sim)
        with patch.object(sim, 'patient', return_value=('응답', '질문')):
            client.say('가' * 30)
            with self.assertRaisesRegex(ValueError, '30'):
                client.say('가' * 31)
            sim.turns = 50
            with self.assertRaisesRegex(NovaError, 'TURN_LIMIT_REACHED'):
                client.say('질문')
        sim.doctor_calls = 1
        note = empty_record(); note['S']['cc'] = '배가 아파요'
        self.assertEqual(client.diagnose(note=note, diagnosis='진단 미정')['status'], 'completed')
        self.assertEqual(sim.turns, 50)

    def test_idempotent_action_and_conflict(self):
        sim = simulator(); client = LocalClient(sim)
        with patch.object(sim, 'patient', return_value=('응답', '질문')) as patient:
            first = client.say('질문', request_id='same')
            self.assertEqual(client.say('질문', request_id='same'), first)
            self.assertEqual(sim.turns, 1); self.assertEqual(patient.call_count, 1)
            with self.assertRaisesRegex(NovaError, 'REQUEST_ID_CONFLICT'):
                client.say('다른 질문', request_id='same')

    def test_missing_exam_is_unknown_not_generated_normal(self):
        sim = simulator([{'id': None, 'reason': 'unlisted'}])
        self.assertEqual(LocalClient(sim).exam('머피 징후 확인')['result'], UNKNOWN)
        self.assertEqual(sim.sim_calls, 1)

    @unittest.skipUnless((Path(__file__).resolve().parent.parent / 'nova-benchmarks-2026').is_dir(), 'external benchmark integration data not installed')
    def test_gold_and_tests_withheld_from_all_adapters(self):
        root = Path(__file__).resolve().parent.parent / 'nova-benchmarks-2026'
        for dataset in ('agentclinic', 'ddxplus', 'aci'):
            rows = list(load_rows(root, dataset, limit=2))
            self.assertEqual(len(rows), 2)
            for row in rows:
                serialized = json.dumps(row['case'])
                for key in ('Correct_Diagnosis', 'Test_Results', 'PATHOLOGY', 'DIFFERENTIAL_DIAGNOSIS', 'reference_note'):
                    self.assertNotIn(key, serialized)
                self.assertNotIn('dialogue', row['case'])

    def test_ddx_value_decoding_and_default_absence(self):
        meta = {'E_1': {'question_en': 'Fever?', 'data_type': 'B', 'default_value': 0},
                'E_2': {'question_en': 'Pain type?', 'data_type': 'M', 'default_value': 'V_0',
                        'value_meaning': {'V_2': {'en': 'burning'}, 'V_3': {'en': 'sharp'}}},
                'E_3': {'question_en': 'Severity?', 'data_type': 'C', 'default_value': 0},
                'E_4': {'question_en': 'Cough?', 'data_type': 'B', 'default_value': 0}}
        result = decode_ddx({'EVIDENCES': "['E_1', 'E_2_@_V_2', 'E_2_@_V_3', 'E_3_@_5']"}, meta)
        self.assertEqual(result['E_1']['answer'], '예 (Yes)')
        self.assertEqual(result['E_2']['answer'], 'burning, sharp')
        self.assertEqual(result['E_3']['answer'], '5')
        self.assertEqual(result['E_4']['answer'], '아니요 (No)')

    def test_consult_duplicate_guard_and_observed_only_prompt(self):
        doctor = PhaseEndpoint([
            {'action': 'SAY', 'topic': 'course', 'content': '언제부터 아팠나요?'},
            {'action': 'DONE'}])
        sim = BenchmarkSimulator(case(), doctor=doctor, sim=Endpoint())
        client = LocalClient(sim)
        spoken = []
        original = client.say
        def capture(content, **kwargs):
            spoken.append(content)
            return original(content, **kwargs)
        with patch.object(run, 'history_queue', return_value=[('onset', '언제부터 아팠나요?')]), \
             patch.object(sim, 'patient', return_value=('어제부터요', '질문')), \
             patch.object(client, 'say', side_effect=capture):
            result = run.consult(client, {})
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(spoken.count('언제부터 아팠나요?'), 1)
        self.assertEqual(sim.record['O']['tests'], '')
        self.assertNotIn('HIDDEN', json.dumps(doctor.messages))
        self.assertIn('어제부터요', json.dumps(doctor.messages, ensure_ascii=False))

    def test_record_only_path_preserves_observed_tests(self):
        note = empty_record(); note['O']['tests'] = 'Hb 8 g/dL'; note['S']['cc'] = 'fatigue'
        sim = simulator(); sim.doctor = Endpoint([{**note, 'diagnosis': 'anemia'}])
        record, _ = run.write_record(LocalClient(sim), sim.public_session(), ['Hb 8 g/dL'], preliminary=False)
        self.assertEqual(record['O']['tests'], 'Hb 8 g/dL')

    def test_doctor_required_and_prelim_tests_empty(self):
        sim = simulator(); client = LocalClient(sim)
        note = empty_record(); note['S']['cc'] = '증상'
        with self.assertRaisesRegex(NovaError, 'DOCTOR_CALL_REQUIRED'):
            client.diagnose(note=note, diagnosis='진단 미정')
        sim.doctor_calls = 1; note['O']['tests'] = 'unseen'
        with self.assertRaisesRegex(NovaError, 'PRELIMINARY_TEST_RESULTS_FORBIDDEN'):
            client.diagnose(note=note, diagnosis='진단 미정')

    def test_transient_action_reuses_request_id(self):
        sim = simulator(); sim.doctor = PhaseEndpoint()
        client = LocalClient(sim)
        session = client.session(); session['turns_remaining'] = 3
        original = client.say; ids = []
        def lost_response(content, **kwargs):
            ids.append(kwargs['request_id'])
            answer = original(content, **kwargs)
            if len(ids) == 1:
                raise NovaError('API_UNAVAILABLE', 503)
            return answer
        with patch.object(client, 'say', side_effect=lost_response), patch('run.time.sleep'), \
             patch.object(client, 'session', return_value=session), \
             patch.object(sim, 'patient', return_value=('네', '설명')):
            result = run.consult(client, {})
        self.assertEqual(ids[0], ids[1])
        self.assertLessEqual(sim.turns, 3)
        self.assertEqual(result['status'], 'completed')


class RetrievalTests(unittest.TestCase):
    def test_korean_compound_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'guidelines'; (root / 'source').mkdir(parents=True)
            (root / 'source/doc.md').write_text('---\ntitle: "갑상선기능항진증"\n---\n치료 지침', encoding='utf-8')
            index = GuidelineIndex(root, Path(tmp) / 'cache')
            self.assertEqual(index.search('갑상선 기능 항진')[0]['title'], '갑상선기능항진증')
            index.close()

    def test_full_body_retrieval_cache_and_invalidation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'guidelines'; (root / 'source').mkdir(parents=True)
            path = root / 'source/doc.md'
            path.write_text('---\ntitle: "Example"\nsource_url: "https://example.org"\nlicense: "CC BY"\n---\n'
                            + ('unrelated text ' * 1000) + 'diplopia ptosis fatigable weakness')
            index = GuidelineIndex(root, Path(tmp) / 'cache')
            fingerprint = index.fingerprint
            hits = index.search('diplopia ptosis')
            self.assertEqual(len(hits), 1); self.assertGreater(hits[0]['offset'], 10000)
            self.assertEqual(hits[0]['license'], 'CC BY'); self.assertEqual(hits[0]['source_url'], 'https://example.org')
            self.assertEqual(index.search('!!!'), [])
            index.close()
            cached = GuidelineIndex(root, Path(tmp) / 'cache')
            self.assertEqual(cached.fingerprint, fingerprint); cached.close()
            path.write_text(path.read_text() + '\nnew content')
            updated = GuidelineIndex(root, Path(tmp) / 'cache')
            self.assertNotEqual(updated.fingerprint, fingerprint); updated.close()

    def test_rag_keeps_patient_and_reference_separate(self):
        from rag import context
        text = context([{'title': 'Guide', 'text': 'example finding'}])
        self.assertIn('S/O에는 시작 정보와 실제 응답만', text)
        self.assertIn('참고 문헌', text)


if __name__ == '__main__':
    unittest.main()
