"""Required history, turn budgeting and actual patient explanations."""
import json
import unittest
from unittest.mock import patch

import run
from grounding import asks, sources
from nova_local import LocalClient
from nova_sdk import NovaError
from test_contract import Endpoint, PhaseEndpoint, simulator


def run_consult(max_turns, fail_explanation=False):
    sim = simulator(); sim.doctor = PhaseEndpoint()
    client = LocalClient(sim)
    session = client.session(); session['turns_remaining'] = max_turns
    spoken = []
    original = client.say
    def capture(content, **kwargs):
        spoken.append(content)
        return original(content, **kwargs)
    with patch.object(client, 'session', return_value=session), \
         patch.object(client, 'say', side_effect=capture), \
         patch.object(sim, 'patient', return_value=('미확인: 원자료에 해당 정보가 없습니다.', '응답')):
        if fail_explanation:
            with patch.object(run, 'patient_explanations', side_effect=NovaError('PROVIDER_UNAVAILABLE', 503)):
                result = run.consult(client, {})
        else:
            result = run.consult(client, {})
    return sim, spoken, result


class FlowTests(unittest.TestCase):
    def test_fixed_history_precedes_model_done_and_actual_explanations(self):
        sim, spoken, result = run_consult(20)
        required = [q for _, q in run.history_queue(sim.public_session()['initial']['opening'])]
        self.assertEqual(spoken[:len(required)], required)
        closing = spoken[len(required):]
        self.assertEqual(len(closing), 5)
        self.assertTrue(all(not asks(text) for text in closing))
        self.assertIn('2주 뒤 결과와 경과를 확인해요.', closing)
        self.assertEqual(result['status'], 'completed')
        self.assertNotIn('HIDDEN', json.dumps(sim.doctor.messages))
        self.assertIn('[실제 설명/', sim.record['P']['education'])

    def test_small_budget_reserves_closing_and_final_record(self):
        sim, spoken, result = run_consult(8)
        self.assertLessEqual(sim.turns, 8)
        self.assertEqual(len(spoken), 5)
        self.assertTrue(all(not asks(text) for text in spoken))
        self.assertEqual(result['status'], 'completed')

    def test_invalid_explanation_is_repaired_without_asking_a_question(self):
        sim = simulator()
        sim.doctor = Endpoint([{'text': '증상이 더 있나요?'}, {'text': '검사로 원인을 확인해요.'}, {'text': ''}])
        note = {'P': {'tests': 'Check the cause', 'treatment': '', 'follow_up': '', 'education': ''}}
        output = run.patient_explanations(LocalClient(sim), note, '')
        self.assertEqual(output, [('next', '검사로 원인을 확인해요.')])
        self.assertTrue(all(len(text) <= run.MAX_CHARS and not asks(text) for _, text in output))

    def test_patient_replies_to_explanation_are_not_observations(self):
        session = {'initial': {'opening': '배가 아파요', 'vital_signs': '미제공'}}
        transcript = ['턴 1 의사: 검사로 원인을 확인해요.', '환자: 네, 알겠습니다']
        self.assertEqual([item['kind'] for item in sources(session, transcript)[2:]], ['communication'])

    def test_failed_explanation_does_not_lose_final_record(self):
        sim, spoken, result = run_consult(20, fail_explanation=True)
        self.assertIsNotNone(sim.record)
        self.assertEqual(result['status'], 'completed')
        self.assertTrue(all(asks(text) for text in spoken))

    def test_rag_can_be_switched_off(self):
        with patch.dict('os.environ', {'NOVA_RAG': '0'}):
            self.assertIsNone(run.prepare()['rag'])


if __name__ == '__main__':
    unittest.main()
