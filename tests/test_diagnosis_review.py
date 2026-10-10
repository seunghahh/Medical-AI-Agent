import json
import time
import unittest
from dataclasses import replace
from clinic.agent import DemoSimulator, Doctor, validate_review
from clinic.client import ResponseFormatError
from clinic.data import load_cases
from clinic.runner import run_case
from test_clinic import FakeClient


def final(diagnosis="Candidate A", basis=None):
    return {"action": "DIAGNOSE", "diagnosis": diagnosis, "basis": basis or ["e1"],
            "differential": ["Candidate B"], "plan": {k: "Further evaluation required"
            for k in ("further_tests", "treatment", "disposition", "follow_up", "education")}}


def review(case_index=0):
    opening = load_cases()[case_index].opening
    return {"candidates": [
        {"diagnosis": "Candidate A", "support": [{"id": "e1", "quote": opening['demographics']}],
         "against": [{"id": "e2", "quote": opening['chief_complaint']}]},
        {"diagnosis": "Candidate B", "support": [{"id": "e2", "quote": opening['chief_complaint']}], "against": []}],
        "final": final("Candidate B", ["e2"])}


class DiagnosisReviewTests(unittest.TestCase):
    def test_review_is_opt_in_and_grounded_in_both_engines_without_hidden_records(self):
        case = replace(load_cases()[0], gold="HIDDEN_GOLD_MARKER", sources={
            "SAY": {"History": "HIDDEN_HISTORY_MARKER"}, "EXAM": {}, "TEST": {"Lab": "HIDDEN_LAB_MARKER"}})
        for engine in ("plain", "langgraph"):
            baseline = FakeClient([final()])
            before = run_case(case, Doctor(baseline), DemoSimulator(), engine=engine)
            self.assertEqual(before['diagnosis'], "Candidate A")
            self.assertEqual(before['diagnosis_reviews'], [])
            client = FakeClient([final(), review()])
            after = run_case(case, Doctor(client, diagnosis_review=True), DemoSimulator(), engine=engine)
            self.assertEqual(after['diagnosis'], "Candidate B")
            self.assertEqual(after['turns'], 0)
            self.assertTrue(after['diagnosis_reviews'][0]['accepted'])
            self.assertEqual(len(client.payloads), 2)
            self.assertTrue(all(e['status'] == 'OBSERVED' for e in client.payloads[-1]['observed_evidence']))
            text = json.dumps(client.payloads)
            for marker in ("HIDDEN_GOLD_MARKER", "HIDDEN_HISTORY_MARKER", "HIDDEN_LAB_MARKER", "gold_diagnosis"):
                self.assertNotIn(marker, text)

    def test_unknown_fabricated_or_malformed_review_keeps_valid_draft(self):
        state = {'round': 'preliminary', 'evidence': [
            {'id': 'e1', 'status': 'OBSERVED', 'value': load_cases()[0].opening['demographics']},
            {'id': 'e2', 'status': 'OBSERVED', 'value': load_cases()[0].opening['chief_complaint']},
            {'id': 'e3', 'status': 'UNKNOWN', 'source': 'Unrecorded exam'}]}
        self.assertIsNone(validate_review(review(), state))
        same_record = review(); same_record['candidates'][0]['against'] = same_record['candidates'][0]['support'][:]
        self.assertIsNone(validate_review(same_record, state))
        unsupported_alternative = review(); unsupported_alternative['candidates'][0]['support'] = []
        self.assertIsNone(validate_review(unsupported_alternative, state))
        bad = []
        for field in ('support', 'against'):
            for reference in ('e3', 'e999', 3):
                reply = review(); reply['candidates'][0][field] = [{'id': reference, 'quote': 'Original value'}]; bad.append(reply)
        reply = review(); reply['final']['basis'] = ['e3']; bad.append(reply)
        reply = review(); reply['final']['action'] = 'TEST'; bad.append(reply)
        reply = review(); reply['final']['diagnosis'] = 'Not listed'; bad.append(reply)
        reply = review(); reply['candidates'][1]['diagnosis'] = 'Candidate A'; bad.append(reply)
        reply = review(); reply['selection_reason'] = 'No bloody stool'; bad.append(reply)
        reply = review(); reply['candidates'][0]['note'] = 'No bloody stool'; bad.append(reply)
        reply = review(); reply['candidates'][1]['support'][0]['quote'] = 'No bloody stool'; bad.append(reply)
        reply = review(); reply['final']['basis'] = ['e1']; bad.append(reply)
        bad.extend([{}, [], {'candidates': None}])
        for reply in bad:
            doctor = Doctor(FakeClient([reply]), diagnosis_review=True)
            draft = final()
            self.assertEqual(doctor.review(state, draft, time.monotonic()+60), draft)
            self.assertFalse(doctor.reviews[0]['accepted'])
            self.assertTrue(doctor.reviews[0]['error'])
            self.assertEqual(doctor.reviews[0]['rejected_response'], reply)

    def test_review_format_failure_and_short_deadline_preserve_draft_and_tag_cost(self):
        class Broken:
            calls = []
            def complete(self, *args):
                self.calls.append({'role': 'doctor'})
                raise ResponseFormatError('Invalid final JSON')
        state = {'round': 'preliminary', 'evidence': [{'id': 'e1', 'status': 'OBSERVED'}]}
        client = Broken(); doctor = Doctor(client, diagnosis_review=True)
        self.assertEqual(doctor.review(state, final(), time.monotonic()+60), final())
        self.assertEqual(client.calls[0]['stage'], 'diagnosis_review')
        self.assertFalse(doctor.reviews[0]['accepted'])
        doctor.review(state, final(), time.monotonic()+10)
        self.assertEqual(len(client.calls), 1)
        self.assertIn('skipped', doctor.reviews[1]['error'])

    def test_invalid_draft_not_reviewed_and_reviews_remain_case_local(self):
        malformed = final(); malformed['basis'] = ['e999']
        doctor = Doctor(FakeClient([malformed]), diagnosis_review=True)
        state = {'round': 'preliminary', 'evidence': [{'id': 'e1', 'status': 'OBSERVED'}]}
        self.assertEqual(doctor.act(state, time.monotonic()+60), malformed)
        self.assertEqual(doctor.reviews, [])
        client = FakeClient([final(), review(0), final(), review(1)])
        doctor = Doctor(client, diagnosis_review=True)
        results = [run_case(load_cases()[i], doctor, DemoSimulator()) for i in (0, 1)]
        self.assertEqual([len(r['diagnosis_reviews']) for r in results], [1, 1])
        self.assertEqual(len(doctor.reviews), 2)

    def test_negation_cannot_be_fabricated_or_trimmed_and_explicit_negative_is_allowed(self):
        state = {'round': 'preliminary', 'evidence': [
            {'id': 'e1', 'status': 'OBSERVED', 'value': 'Abdomen is distended. No vomiting is reported.'},
            {'id': 'e2', 'status': 'OBSERVED', 'value': 'No fever or bloody stools.'}]}
        reply = review()
        reply['candidates'][0]['support'] = [{'id': 'e1', 'quote': 'Abdomen is distended.'}]
        reply['candidates'][0]['against'] = [{'id': 'e1', 'quote': 'No vomiting is reported.'}]
        reply['candidates'][1]['support'] = [{'id': 'e2', 'quote': 'No fever or bloody stools.'}]
        self.assertIsNone(validate_review(reply, state))
        for quote in ['fever or bloody stools.', 'No diarrhea.', 'No vomiting is reported. No diarrhea.']:
            reply['candidates'][1]['support'][0]['quote'] = quote
            self.assertIn('quotation', validate_review(reply, state))

    def test_review_receives_latest_valid_memory_and_cannot_drop_tracked_alternative(self):
        opening = load_cases()[0].opening
        state = {'round': 'preliminary', 'evidence': [
            {'id': 'e1', 'status': 'OBSERVED', 'source': 'Demographics', 'turn': 0, 'value': opening['demographics']},
            {'id': 'e2', 'status': 'OBSERVED', 'source': 'Complaint', 'turn': 0, 'value': opening['chief_complaint']}],
            'working_memory': {'hypotheses': []}}
        draft = final(); draft['differential'] = []
        draft['working_memory'] = {'hypotheses': [{'diagnosis': 'Candidate B', 'basis': ['e2'],
            'against': ['e1'], 'missing': 'One distinguishing exam'}],
            'next_information_needed': 'One exam', 'next_action_reason': 'Compare alternatives'}
        client = FakeClient([draft, review()]); doctor = Doctor(client, diagnosis_review=True)
        result = doctor.act(state, time.monotonic()+60)
        self.assertEqual(client.payloads[-1]['candidate_pool'], ['Candidate A', 'Candidate B'])
        self.assertEqual(client.payloads[-1]['working_memory']['hypotheses'][0]['against'], ['e1'])
        self.assertEqual(result['working_memory']['hypotheses'][0]['diagnosis'], 'Candidate B')
        dropped = review(); dropped['candidates'][1]['diagnosis'] = 'Candidate C'; dropped['final']['diagnosis'] = 'Candidate C'
        doctor = Doctor(FakeClient([dropped]), diagnosis_review=True)
        self.assertEqual(doctor.review({**state, 'working_memory': result['working_memory']}, draft, time.monotonic()+60), draft)
        self.assertIn('dropped', doctor.reviews[0]['error'])
        draft['working_memory']['hypotheses'][0]['basis'] = ['e999']
        client = FakeClient([draft, review()]); doctor = Doctor(client, diagnosis_review=True)
        result = doctor.act(state, time.monotonic()+60)
        self.assertEqual(client.payloads[-1]['working_memory']['hypotheses'], [])
        self.assertEqual(result['working_memory']['hypotheses'][0]['basis'], ['e999'])


if __name__ == '__main__':
    unittest.main()
