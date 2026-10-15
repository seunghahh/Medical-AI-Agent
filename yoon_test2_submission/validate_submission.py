"""Offline regression runner; excluded from the upload ZIP. No API/key use."""
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if not (REPO / 'test_contract.py').is_file():
    REPO = HERE.parent / 'Medical-AI-Agent'
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
import run
from test_contract import PhaseEndpoint, simulator
from test_flow import run_consult
from test_policy import Scripted
from nova_local import LocalClient
from nova_sdk import NovaError


class SubmissionTests(unittest.TestCase):
    def test_batch_rejects_duplicate_unknown_topics_and_long_questions(self):
        client = Scripted([{'questions': [
            {'topic': 'onset', 'content': '언제부터 아팠나요?'},
            {'topic': 'onset', 'content': '증상이 언제 시작됐나요?'},
            {'topic': 'invented', 'content': '어떤가요?'},
            {'topic': 'severity', 'content': '아' * 31 + '?'}]}])
        self.assertEqual(run.question_batch(client, {'initial': {'opening': '배가 아파요', 'vital_signs': ''}}, [], set()),
                         [('onset', '언제부터 아팠나요?')])
        self.assertEqual(len(client.calls), 1)

    def test_batch_respects_pending_basic_history_and_prior_questions(self):
        client = Scripted([{'questions': [
            {'topic': 'past', 'content': '앓았던 병이 있나요?'},
            {'topic': 'onset', 'content': '언제부터 아팠나요?'},
            {'topic': 'impact', 'content': '일상생활에 어떤 불편이 있나요?'}]}])
        result = run.question_batch(client, {'initial': {'opening': '통증', 'vital_signs': ''}},
                                    ['턴 1 의사: 언제부터 아팠나요?'], set(), ['past'])
        self.assertEqual(result, [('impact', '일상생활에 어떤 불편이 있나요?')])

    def test_feedback_and_history_have_distinct_record_boundaries(self):
        sim, spoken, result = run_consult(20)
        required = [q for _, q in run.history_queue(sim.public_session()['initial']['opening'])]
        self.assertEqual(spoken[:len(required)], required)
        self.assertEqual(len(spoken[len(required):]), 7)
        self.assertTrue(all(not run.asks(q) for q in spoken[len(required):-2]))
        self.assertEqual(spoken[-2:], ['다음에 할 일을 어떻게 이해하셨나요?', '계획에서 걱정되거나 궁금한 점이 있나요?'])
        self.assertIn('[실제 확인/understanding]', sim.record['P']['education'])
        self.assertNotIn('다음에 할 일을', sim.record['S']['hpi'])
        self.assertEqual(sim.record['O']['tests'], '')
        self.assertEqual(result['status'], 'completed')

    def test_small_budget_and_all_50_turn_budgets_finish(self):
        for budget in range(1, 51):
            with self.subTest(budget=budget):
                sim, spoken, result = run_consult(budget)
                self.assertLessEqual(sim.turns, budget)
                self.assertTrue(all(len(q) <= 30 for q in spoken))
                self.assertEqual(result['status'], 'completed')

    def test_feedback_does_not_restate_a_dose_or_invent_interval(self):
        for text in ['10mg을 드시면 돼요.', '3일 뒤 다시 와요.', '증상이 더 있나요?']:
            with self.subTest(text=text):
                client = Scripted([{'text': text}])
                out = run.feedback_response(client, {'P': {}}, 'working', '질문', '약을 더 먹나요?')
                self.assertEqual(out, '궁금한 점은 담당의와 더 상의해요.')

    def test_assent_or_unknown_needs_no_model_call(self):
        for reply in ['네.', '알겠습니다', '미확인: 자료가 없습니다.']:
            self.assertEqual(run.feedback_response(Scripted([]), {'P': {}}, '', '', reply), '')

    def test_focused_questions_precede_basic_history_and_followups_react(self):
        sim = simulator()
        class FocusedEndpoint(PhaseEndpoint):
            def complete(self, messages, **kwargs):
                system = messages[0]['content']
                if 'For this call return' in system:
                    return json.dumps({'questions': [{'topic': 'onset', 'content': '언제부터 아팠나요?'},
                                                     {'topic': 'severity', 'content': '통증은 0~10 중 몇 점인가요?'}]}), 'stop', 10, 10
                if system.startswith('Respond to patient feedback'):
                    return json.dumps({'text': '검사로 원인을 확인해요.'}), 'stop', 10, 10
                return super().complete(messages, **kwargs)
        sim.doctor = FocusedEndpoint()
        client = LocalClient(sim)
        spoken = []
        original = client.say
        def say(text, **kwargs):
            spoken.append(text)
            return original(text, **kwargs)
        def patient(text):
            return ('검사를 왜 하는지 걱정돼요.' if text.startswith('계획에서') else '어제부터 아파요.', 'reply')
        with patch.object(client, 'say', side_effect=say), patch.object(sim, 'patient', side_effect=patient):
            result = run.consult(client, {})
        self.assertEqual(spoken[1:3], ['언제부터 아팠나요?', '통증은 0~10 중 몇 점인가요?'])
        self.assertIn('검사를 왜 하는지 걱정돼요.', sim.record['P']['education'])
        self.assertNotIn('검사를 왜 하는지 걱정돼요.', sim.record['A']['evidence'])
        self.assertEqual(result['status'], 'completed')

    def test_feedback_outage_preserves_final_submission(self):
        sim = simulator(); sim.doctor = PhaseEndpoint()
        with patch.object(sim, 'patient', return_value=('잘 모르겠어요.', 'reply')), \
             patch.object(run, 'feedback_response', side_effect=NovaError('PROVIDER_UNAVAILABLE', 503)):
            self.assertEqual(run.consult(LocalClient(sim), {})['status'], 'completed')
        self.assertIsNotNone(sim.record)


def flatten(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from flatten(test)
        else:
            yield test


if __name__ == '__main__':
    # The old fake endpoint predates feedback. Teach it this new phase, keeping
    # the original assertions for duplicate history and observed-only records.
    original_complete = PhaseEndpoint.complete
    def complete_with_feedback(self, messages, **kwargs):
        if messages[0]['content'].startswith('Respond to patient feedback'):
            self.messages.append(messages)
            return json.dumps({'text': '검사로 원인을 확인해요.'}), 'stop', 10, 10
        return original_complete(self, messages, **kwargs)
    PhaseEndpoint.complete = complete_with_feedback
    external = REPO.parent / 'nova-benchmarks-2026' / 'AgentClinic-MedQA' / 'agentclinic_medqa_extended.jsonl'
    if not external.is_file():
        from test_contract import ContractTests
        for name in ('test_cli_outputs_for_each_benchmark', 'test_gold_and_tests_withheld_from_all_adapters'):
            setattr(ContractTests, name, unittest.skip('External AgentClinic dataset file is not installed')(getattr(ContractTests, name)))
    # Two original tests require exactly five closing statements and no feedback.
    # Replace those obsolete expectations with boundary/budget tests above.
    replaced = {'test_flow.FlowTests.test_fixed_history_precedes_model_done_and_actual_explanations',
                'test_flow.FlowTests.test_small_budget_reserves_closing_and_final_record'}
    suite = unittest.TestSuite()
    for name in ('test_audit', 'test_contract', 'test_flow', 'test_grounding', 'test_policy', 'test_rag'):
        suite.addTests(test for test in flatten(unittest.defaultTestLoader.loadTestsFromName(name))
                       if test.id() not in replaced)
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(SubmissionTests))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    sys.exit(not result.wasSuccessful())
