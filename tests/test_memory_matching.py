import copy
import json
import time
import unittest
from dataclasses import replace
from unittest.mock import patch
from clinic.agent import DemoDoctor, DemoSimulator, Doctor, Simulator
from clinic.data import load_cases
from clinic.matching import allowed_source_ids, bundled_request, source_facts
from clinic.memory import apply_update, build_memory
from clinic.runner import Encounter, run_case
from test_clinic import FakeClient


class MemoryMatchingTests(unittest.TestCase):
    def test_action_plan_expires_after_observation_rejection_and_intervening_exam(self):
        question = {'action': 'SAY', 'intent': 'question', 'text': '출생 체중은 몇 그램인가요?'}
        exam = {'action': 'EXAM', 'text': '호흡음을 청진합니다.'}
        different = {**question, 'text': '산모의 병력이 있나요?'}
        notes = {'hypotheses': [{'diagnosis': 'Candidate', 'basis': ['e1'], 'against': [],
                                'missing': 'birth weight'}],
                 'next_information_needed': 'birth weight', 'next_action_reason': 'Distinguish candidates'}
        final = DemoDoctor().act({'force_diagnose': True, 'evidence': [], 'round': 'preliminary'}, 0)
        class Spy(FakeClient):
            def __init__(self, replies):
                super().__init__(replies)
                self.prompts = []
            def complete(self, system, payload, role, deadline):
                self.prompts.append(system)
                return super().complete(system, payload, role, deadline)
        for engine in ('plain', 'langgraph'):
            for facts in ([], [{'path': 'History', 'value': 'Recorded birth weight'}]):
                with self.subTest(engine=engine, facts=facts):
                    client = Spy([{**a, 'working_memory': copy.deepcopy(notes)}
                                  for a in (question, question, exam, different)] + [final])
                    simulator = DemoSimulator()
                    with patch.object(simulator, 'resolve', side_effect=[
                        {'accepted': True, 'facts': facts, 'reason': '' if facts else 'not_recorded'},
                        {'accepted': True, 'facts': [{'path': 'Exam', 'value': 'New finding'}], 'reason': ''},
                        {'accepted': True, 'facts': [], 'reason': 'not_recorded'}]) as resolve:
                        result = run_case(load_cases()[96], Doctor(client), simulator,
                                          engine=engine, working_memory=True)
                    self.assertTrue(result['completed'])
                    self.assertEqual(result['turns'], 3)
                    self.assertEqual(result['rejections'], 1)
                    self.assertEqual(resolve.call_count, 3)
                    self.assertTrue(result['history'][1]['feedback'].startswith('Repeated question'))
                    for payload in client.payloads[1:]:
                        self.assertEqual(payload['working_memory']['next_information_needed'], '')
                        self.assertEqual(payload['working_memory']['next_action_reason'], '')
                        self.assertEqual(payload['working_memory']['hypotheses'], notes['hypotheses'])
                        self.assertIn(question['text'], payload['asked_questions'])
                    self.assertIn('SAY', client.payloads[3]['action_policy']['allowed_actions'])
                    self.assertIn(question['text'], client.prompts[3])
                    self.assertIn('An intervening EXAM does not reopen it', client.prompts[3])
                    self.assertEqual(result['memory_updates'][1]['next_information_needed'], 'birth weight')
                    self.assertNotIn(load_cases()[96].gold, json.dumps(client.payloads))

    def test_sentence_selection_preserves_original_and_does_not_return_neighbors(self):
        history = 'Symptoms started 3.5 weeks ago. Dr. Smith prescribed a drug. No injury is reported.'
        facts = source_facts({'History': history}, 'SAY')
        self.assertEqual([f['value'] for f in facts.values()],
                         ['Symptoms started 3.5 weeks ago.', 'Dr. Smith prescribed a drug.', 'No injury is reported.'])
        self.assertTrue(all(f['value'] in history for f in facts.values()))
        self.assertEqual(source_facts({'Exam': history}, 'EXAM')['f0']['value'], history)
        case = replace(load_cases()[0], sources={'SAY': {'History': history}})
        simulator = Simulator(FakeClient([{'accepted': True, 'source_ids': ['f0'], 'reason': ''}]))
        result = simulator.resolve(case, {'action': 'SAY', 'intent': 'question', 'text': '언제 시작됐나요?'}, time.monotonic()+30)
        self.assertEqual(result['facts'], [facts['f0']])
        self.assertNotIn('prescribed', json.dumps(result))

    def test_injury_question_cannot_fetch_unrelated_medication_or_symptom_history(self):
        case = replace(load_cases()[0], sources={'SAY': {'History':
            'Difficulty walking began a month ago. The patient takes a medication.'}})
        action = {'action': 'SAY', 'intent': 'question', 'text': '최근 머리 부상이 있었나요?'}
        simulator = Simulator(FakeClient([{'accepted': True, 'source_ids': ['f0', 'f1'], 'reason': ''}]))
        result = simulator.resolve(case, action, time.monotonic()+30)
        self.assertEqual(result['facts'], [])
        self.assertEqual(result['reason'], 'not_recorded')
        self.assertEqual(simulator.resolutions[0]['discarded_source_ids'], ['f0', 'f1'])
        facts = source_facts({'History': 'No head injury is reported. A drug was prescribed.'}, 'SAY')
        self.assertEqual(allowed_source_ids(action, facts), {'f0'})

    def test_memory_opposition_must_be_observed_and_missing_stays_separate(self):
        evidence = [{'id': 'e1', 'status': 'OBSERVED', 'source': 'History', 'value': 'A finding', 'turn': 1},
                    {'id': 'e2', 'status': 'UNKNOWN', 'source': 'Another finding', 'value': 'Not recorded', 'turn': 2}]
        update = {'hypotheses': [{'diagnosis': 'Candidate', 'basis': ['e1'], 'against': ['e2'], 'missing': 'Another finding'}],
                  'next_information_needed': 'Another exam', 'next_action_reason': 'Distinguish candidates'}
        memory, error = apply_update(update, evidence, build_memory(evidence))
        self.assertIn('OBSERVED', error)
        update['hypotheses'][0]['against'] = []
        memory, error = apply_update(update, evidence, memory)
        self.assertIsNone(error)
        self.assertEqual(memory['hypotheses'][0]['missing'], 'Another finding')
        self.assertEqual(memory['hypotheses'][0]['against'], [])

    def test_eyelid_and_strength_bundle_is_rejected_without_turn_charge(self):
        client = FakeClient([])
        simulator = Simulator(client)
        case = load_cases()[0]
        for text in ['눈꺼풀 처짐과 근력 저하를 확인합니다.',
                     'Check ptosis and muscle strength.', '안검하수 및 근력을 확인합니다.']:
            action = {'action': 'EXAM', 'text': text}
            encounter = Encounter(case, DemoDoctor(), simulator, 'preliminary', 50, 1200, None)
            state = encounter.initial_state()
            state['action'] = action
            after = encounter.execute(state)
            self.assertEqual(after['turns'], 0)
            self.assertEqual(after['rejections'], 1)
            self.assertEqual(after['history'][-1]['feedback'], 'bundled')
            self.assertEqual(after['evidence'], state['evidence'])
        self.assertEqual(client.payloads, [])
        for text in ['지속 상방 주시로 안검하수를 확인합니다.',
                     '눈꺼풀 처짐을 확인합니다.', 'Check upper extremity strength.',
                     'Inspect ptosis and eyelid position.']:
            self.assertFalse(bundled_request({'action': 'EXAM', 'text': text}), text)

    def test_physical_methods_cannot_return_each_others_structured_findings(self):
        case = replace(load_cases()[77], sources={"EXAM": {
            "Chest_Examination.Inspection": "Visible finding",
            "Chest_Examination.Palpation": "Palpable finding",
            "Chest_Examination.Auscultation": "Crackles",
            "Chest_Examination.Percussion": "Dullness",
            "Chest_Examination.Other": "Unstructured finding"}})
        for text, expected in [("흉부 촉진을 시행합니다.", 1), ("Palpate the chest.", 1),
                               ("폐음 청진을 시행합니다.", 2), ("Auscultate the lungs.", 2),
                               ("흉부 타진을 시행합니다.", 3), ("Percuss the chest.", 3),
                               ("피부를 시각적으로 검사합니다.", 0), ("Inspect the chest.", 0)]:
            simulator = Simulator(FakeClient([{"accepted": True,
                "source_ids": ["f0", "f1", "f2", "f3", "f4"], "reason": ""}]))
            result = simulator.resolve(case, {"action": "EXAM", "text": text}, time.monotonic()+30)
            self.assertEqual([f['path'] for f in result['facts']],
                             [list(case.sources['EXAM'])[expected], 'Chest_Examination.Other'])
        actual = load_cases()[77]
        wrong = list(actual.sources['EXAM']).index('Chest_Examination.Percussion')
        simulator = Simulator(FakeClient([{"accepted": True, "source_ids": [f'f{wrong}'], "reason": ""}]))
        result = simulator.resolve(actual, {"action": "EXAM", "text": "흉부 촉진을 시행합니다."}, time.monotonic()+30)
        self.assertEqual(result, {"accepted": True, "facts": [], "reason": "not_recorded"})
        facts = {"f0": {"path": "Exam.Percussion", "value": "Dullness"}}
        self.assertIsNone(allowed_source_ids({"action": "TEST", "text": "Percuss"}, facts))
        self.assertIsNone(allowed_source_ids({"action": "EXAM", "text": "Chest exam"}, facts))
        self.assertIsNone(allowed_source_ids({"action": "EXAM", "text": "palpation versus percussion"}, facts))
        self.assertIsNone(allowed_source_ids({"action": "EXAM", "text": "왼쪽 폐 기저부를 두드려 소리를 청진합니다."}, facts))

    def test_clear_bundles_are_rejected_before_model_call_but_one_maneuver_is_allowed(self):
        bad = ['Inspect eyes and palpate abdomen.', 'Check active and passive range of motion.',
               'Abduction and external rotation of shoulder.', '황달을 확인하고 복부를 촉진합니다.']
        good = ['보행을 관찰합니다.', 'Check active forward flexion of shoulder.',
                'Palpate abdomen for tenderness and a mass.', 'Inspect patellar reflex.']
        for text in bad:
            action = {"action": "EXAM", "text": text}
            self.assertTrue(bundled_request(action), text)
            client = FakeClient([])
            result = Simulator(client).resolve(load_cases()[27], action, time.monotonic()+30)
            self.assertEqual(result, {"accepted": False, "facts": [], "reason": "bundled"})
            self.assertEqual(client.payloads, [])
        for text in good:
            self.assertFalse(bundled_request({"action": "EXAM", "text": text}), text)

    def test_selected_records_are_filtered_to_requested_scope(self):
        case = load_cases()[27]
        keys = list(case.sources['EXAM'])
        forward = keys.index('Shoulder_Examination.Range_of_Motion.Forward_Flexion')
        abduction = keys.index('Shoulder_Examination.Range_of_Motion.Abduction')
        reply = {"accepted": True, "source_ids": [f'f{forward}', f'f{abduction}'], "reason": ""}
        simulator = Simulator(FakeClient([reply]))
        result = simulator.resolve(case, {"action": "EXAM", "text": "Check forward flexion."}, time.monotonic()+30)
        self.assertEqual([f['path'] for f in result['facts']], [keys[forward]])
        self.assertEqual(simulator.resolutions[0]['discarded_source_ids'], [f'f{abduction}'])
        simulator = Simulator(FakeClient([reply]))
        result = simulator.resolve(case, {"action": "EXAM", "text": "Neer impingement test."}, time.monotonic()+30)
        self.assertEqual(result['facts'], [])
        self.assertEqual(result['reason'], 'not_recorded')

    def test_memory_facts_are_deduplicated_and_unknown_is_not_negative_evidence(self):
        evidence = [{"id": "e1", "source": "History", "value": "Recorded history", "status": "OBSERVED", "turn": 1},
                    {"id": "e2", "source": "History", "value": "Recorded history", "status": "OBSERVED", "turn": 2},
                    {"id": "e3", "source": "Pain?", "value": "Not recorded", "status": "UNKNOWN", "turn": 3}]
        memory = build_memory(evidence)
        self.assertEqual(len(memory['observed_facts']), 1)
        self.assertEqual(memory['unavailable_requests'], ['Pain?'])
        update = {"hypotheses": [{"diagnosis": "Candidate", "basis": ['e3']}],
                  "next_information_needed": "A discriminating finding", "next_action_reason": "Focused exam"}
        rejected, error = apply_update(update, evidence, memory)
        self.assertIn('OBSERVED', error)
        self.assertEqual(rejected['hypotheses'], [])
        update['hypotheses'][0]['basis'] = ['e1']
        update['observed_facts'] = ['Injected fabricated finding']
        accepted, error = apply_update(update, evidence, memory)
        self.assertIsNone(error)
        self.assertEqual(accepted['observed_facts'], memory['observed_facts'])
        self.assertEqual(accepted['hypotheses'][0]['basis'], ['e1'])
        self.assertEqual(memory['hypotheses'], [])

    def test_visual_vulvar_scope_handles_inspection_schema(self):
        case = load_cases()[184]
        keys = list(case.sources['EXAM'])
        target = 'Genitourinary_Examination.Inspection'
        selected = f'f{keys.index(target)}'
        reply = {"accepted": True, "source_ids": [selected], "reason": ""}
        simulator = Simulator(FakeClient([reply]))
        result = simulator.resolve(case, {"action": "EXAM", "text": "Inspect the vulvar skin visually."}, time.monotonic()+30)
        self.assertEqual([f['path'] for f in result['facts']], [target])

    def test_memory_is_case_local_and_plain_graph_parity_preserves_history(self):
        case = replace(load_cases()[0], gold='HIDDEN_GOLD_MARKER')
        question = {"action": "SAY", "text": "증상은 언제 시작됐나요?", "intent": "question",
                    "working_memory": {"hypotheses": [{"diagnosis": "Candidate", "basis": ['e2']}],
                                       "next_information_needed": "History", "next_action_reason": "Acquire onset"}}
        final = DemoDoctor().act({"force_diagnose": True, "evidence": [], "round": "preliminary"}, 0)
        results = []
        for engine in ['plain', 'langgraph']:
            spy = FakeClient([copy.deepcopy(question), copy.deepcopy(final)])
            result = run_case(case, Doctor(spy), DemoSimulator(), engine=engine, working_memory=True)
            self.assertTrue(result['completed'])
            self.assertEqual(result['memory_updates'][0]['accepted'], True)
            self.assertNotIn('HIDDEN_GOLD_MARKER', json.dumps(spy.payloads))
            self.assertNotIn('working_memory', result['history'][0]['action'])
            self.assertEqual(spy.payloads[1]['working_memory']['hypotheses'][0]['diagnosis'], 'Candidate')
            result.pop('seconds')
            result.pop('orchestration')
            results.append(result)
        self.assertEqual(results[0], results[1])
        fresh = Encounter(case, DemoDoctor(), DemoSimulator(), 'preliminary', 10, 180, None, working_memory=True)
        self.assertEqual(fresh.initial_state()['working_memory']['hypotheses'], [])
        self.assertEqual(fresh.initial_state()['memory_updates'], [])

    def test_flat_model_notes_are_validated_and_bad_notes_do_not_pollute_state(self):
        case = load_cases()[0]
        notes = {"hypotheses": [{"diagnosis": "Candidate", "basis": ['e2']}],
                 "next_information_needed": "History", "next_action_reason": "Clarify onset"}
        question = {"action": "SAY", "text": "증상은 언제 시작됐나요?", "intent": "question", **notes}
        final = DemoDoctor().act({"force_diagnose": True, "evidence": [], "round": "preliminary"}, 0)
        final['working_memory'] = {**notes, 'hypotheses': [{'diagnosis': 'Ungrounded', 'basis': ['e999']}]}
        spy = FakeClient([question, final])
        result = run_case(case, Doctor(spy), DemoSimulator(), working_memory=True)
        self.assertEqual([u['accepted'] for u in result['memory_updates']], [True, False])
        self.assertEqual(result['working_memory']['hypotheses'][0]['diagnosis'], 'Candidate')
        self.assertNotIn('hypotheses', result['history'][0]['action'])


if __name__ == '__main__':
    unittest.main()
