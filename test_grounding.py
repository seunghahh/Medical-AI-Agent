"""Regressions for source-bound SOAP and retrieval, without model calls."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import run
from grounding import finalize, normalize_measurements, sources, retrieval_sources, query_spans


class GroundingTests(unittest.TestCase):
    def setUp(self):
        self.session = {'initial': {
            'opening': '나는 서른두 살 여자이다. 오십육 키로그람에서 칠 키로그람 빠져 사십구 키로그람 정도이다.',
            'vital_signs': 'BP 138/72, HR 102'}}
        self.transcript = ['턴 1 의사: 식욕이 어떻게 되나요?', '환자: 식욕이 좋아졌어요',
                           '턴 2 의사: 발열이나 오한, 야간 땀이 있나요?', '환자: 밤에 땀 많이 나요.',
                           '턴 3 의사: 과거 질환이 있었나요?', '환자: 과거 질환 없었어요.',
                           '턴 4 의사: 갑상선 항진증 가능성이 있어요.', '환자: 네, 알겠습니다.']
        self.items = sources(self.session, self.transcript)

    def test_wrong_model_demographics_and_unasked_negatives_cannot_enter_facts(self):
        candidate = {'diagnosis': '갑상선 기능 항진증',
                     'S': {'cc': '42세 14 kg 감소', 'pmh': '알레르기 (-)', 'sh': '흡연 (-)'},
                     'O': {'exam': '정상', 'tests': '정상'},
                     'A': {'evidence': '42세 14 kg 감소, 발열 없음'},
                     'P': {'education': '42세이므로 주의', 'tests': '갑상선 기능 검사 계획'}}
        note, diagnosis, blocked = finalize(candidate, self.items)
        rendered = json.dumps(note, ensure_ascii=False)
        for value in ('32세', '56kg', '7kg', '49kg', '미확인', '미시행', '식욕이 좋아졌어요'):
            self.assertIn(value, rendered)
        for value in ('42세', '14 kg', '알레르기 (-)', '흡연 (-)', '발열 없음', '네, 알겠습니다'):
            self.assertNotIn(value, rendered)
        self.assertEqual(note['O']['tests'], '')
        self.assertEqual(diagnosis, candidate['diagnosis'])
        self.assertEqual(blocked[0]['field'], 'P.education')
        self.assertIn('시행 결과 아님', note['P']['tests'])

    def test_native_numbers_before_se_are_counts_not_ages(self):
        for text in ('하루에 두세 번 보고', '세 번 먹어요', '두세 개'):
            self.assertEqual(normalize_measurements(text), text)
        self.assertEqual(normalize_measurements('삼십이 세 여자'), '32세 여자')
        self.assertEqual(normalize_measurements('서른두 살, 32세'), '32세, 32세')

    def test_number_normalization_is_not_case_specific(self):
        self.assertEqual(normalize_measurements('마흔두 살, 육십일 키로그람, 2.5 kg'), '42세, 61kg, 2.5kg')
        self.assertEqual(normalize_measurements('지난 살, 여러 키로'), '지난 살, 여러 키로')

    def test_unknown_exam_and_response_remain_unknown(self):
        items = sources(self.session, ['턴 1 의사: 알레르기가 있나요?', '환자: 몰라요',
                                       '턴 2 진찰(갑상선 촉진): UNKNOWN'])
        note, _, _ = finalize({}, items)
        self.assertIn('몰라요', note['S']['pmh'])
        self.assertIn('UNKNOWN', note['O']['exam'])
        self.assertNotIn('미시행', note['O']['exam'])
        self.assertNotIn('정상', json.dumps(note, ensure_ascii=False))

    def test_queries_reject_paraphrases_fabrication_and_dropped_negation(self):
        catalog = retrieval_sources(self.items)
        picks = [{'source_id': 'turn1.s0', 'quote': '식욕이 좋아졌어요'},
                 {'source_id': 'turn1.s0', 'quote': '정상 식욕'},
                 {'source_id': 'turn2.s0', 'quote': '발열 없음'},
                 {'source_id': 'turn3.s0', 'quote': '과거 질환'},
                 {'source_id': 'turn4.s0', 'quote': '네, 알겠습니다.'},
                 {'source_id': 'fake', 'quote': 'hypertension'}]
        accepted, rejected = query_spans(picks, catalog)
        self.assertEqual(accepted, picks[:1])
        self.assertEqual(rejected, picks[1:])

    def test_retrieval_fallback_and_aci_dialogue_path(self):
        queries = []
        state = {'rag': SimpleNamespace(search=lambda query: queries.append(query) or [])}
        with patch.object(run, 'ask_doctor', return_value='{"selections":[]}'):
            run.retrieve(None, self.session, [], state)
            run.retrieve(None, self.session, ['Patient: I have fatigue.'], state, preliminary=False)
        self.assertIn('32세', queries[0])
        self.assertEqual(queries[1], 'Patient: I have fatigue.')
        self.assertTrue(all(entry['fallback'] for entry in state['retrievals']))

    def test_model_search_terms_only_widen_the_search(self):
        calls = []
        state = {'rag': SimpleNamespace(search=lambda query, **options: calls.append((query, options)) or [])}
        reply = ('{"selections":[],"thought":" Weight loss with tachycardia suggests\\n hyperthyroidism. ",'
                 '"terms":["hyperthyroidism"," goiter ",7,"hyperthyroidism",""]}')
        with patch.object(run, 'ask_doctor', return_value=reply):
            run.retrieve(None, self.session, [], state)
        query, options = calls[0]
        self.assertEqual(options['extra'], ['hyperthyroidism', 'goiter'])
        self.assertEqual(options['thought'], 'Weight loss with tachycardia suggests hyperthyroidism.')
        self.assertNotIn('hyperthyroidism', query)
        logged = state['retrievals'][-1]
        self.assertEqual((logged['terms'], logged['thought']), (options['extra'], options['thought']))
        self.assertEqual(run.search_terms('not a list'), [])
        self.assertEqual((run.search_thought(None), len(run.search_thought('x ' * 900))), ('', 800))

    def test_plain_search_call_when_model_gives_no_thought_or_terms(self):
        calls = []
        state = {'rag': SimpleNamespace(search=lambda query, **options: calls.append(options) or [])}
        with patch.object(run, 'ask_doctor', return_value='{"selections":[]}'):
            run.retrieve(None, self.session, [], state)
        self.assertEqual(calls, [{}])

    def test_model_summary_and_interpretation_are_cited_and_checked(self):
        candidate = {'diagnosis': '갑상선 기능 항진증', 'cc': ['opening.s1'],
                     'hpi': {'text': '32세 여자, 체중 7kg 감소, 식욕 증가, 야간 발한.', 'source_ids': ['opening.s0', 'opening.s1', 'turn1', 'turn2']},
                     'evidence': [
                         {'text': '식욕 증가에도 체중 감소 → 대사 항진 시사', 'source_ids': ['turn1']},
                         {'text': '빈맥 HR 120', 'source_ids': ['vitals']},                 # value the patient never gave
                         {'text': '과거력 없음', 'source_ids': ['turn1']},                  # cited answer does not say so
                         {'text': '과거력 없음', 'source_ids': ['turn3']},                  # cited answer does ("없었어요")
                         {'text': '발열 소견', 'source_ids': ['turn99']}],                  # no such observation
                     'differential': [{'name': '갑상선염', 'reason': '야간 발한이 있으나 TSH 수용체 항체로 감별', 'source_ids': ['turn2']}]}
        note, _, blocked = finalize(candidate, self.items)
        evidence = note['A']['evidence']
        self.assertIn('- 식욕 증가에도 체중 감소 → 대사 항진 시사 (T1)', evidence)
        self.assertIn('- 과거력 없음 (T3)', evidence)
        self.assertNotIn('HR 120', evidence); self.assertNotIn('T99', evidence)
        self.assertEqual(sorted(b['reason'] for b in blocked),
                         ['관찰에 없는 수치', '근거 없는 음성/정상 단정', '출처 ID'])
        self.assertIn('r/o 갑상선염 : 야간 발한이 있으나 TSH 수용체 항체로 감별 (T2)', note['A']['differential'])
        self.assertTrue(note['S']['cc'].startswith('56kg에서 7kg'))
        self.assertTrue(note['S']['hpi'].startswith('요약: 32세 여자'))
        self.assertIn('나는 32세 여자이다', note['S']['hpi'])             # the patient's own statement stays
        self.assertNotIn('식욕이 좋아졌어요', note['S']['hpi'])             # questions are listed once, in ROS
        self.assertIn('식욕이 좋아졌어요', note['S']['ros'])
        self.assertNotIn('질문/요청', json.dumps(note, ensure_ascii=False))

    def test_internal_ids_in_plans_are_shown_as_citations(self):
        note, _, _ = finalize({'diagnosis': '가설', 'P': {'tests': 'TSH 검사(근거: turn2, vitals, opening.s1, turn12)'}}, self.items)
        self.assertIn('(근거: T2, 활력징후, 첫 진술, T12)', note['P']['tests'])
        self.assertNotIn('turn2', note['P']['tests'])

    def test_no_internal_id_survives_anywhere_in_the_record(self):
        candidate = {'diagnosis': '갑상선기능항진증(turn1)', 'cc': ['opening.s1'],
                     'hpi': {'text': '식욕 증가(turn1), 발한(turn2)', 'source_ids': ['turn1', 'turn2']},
                     'evidence': [{'text': '식욕 증가(turn1)는 대사 항진 시사 [opening.s1]', 'source_ids': ['turn1']}],
                     'differential': [{'name': '감별(vitals)', 'reason': '발한은 설명하나(turn2) 검사로 구분', 'source_ids': ['turn2']}],
                     'P': {'tests': 'TSH(근거: turn2, vitals)', 'education': '안내(opening.s1)'}}
        note, diagnosis, _ = finalize(candidate, self.items)
        text = json.dumps(note, ensure_ascii=False) + diagnosis
        for leaked in ('turn1', 'turn2', 'opening.', 'vitals', 'source_id', '[turn', '[opening'):
            self.assertNotIn(leaked, text)
        self.assertIn('식욕 증가(T1)는 대사 항진 시사 [첫 진술]', note['A']['evidence'])
        self.assertIn('갑상선기능항진증(T1)', diagnosis)

    def test_unusable_model_parts_fall_back_to_the_plain_observations(self):
        candidate = {'diagnosis': '가설', 'cc': ['opening.s9'], 'hpi': {'text': '체중 14kg 감소', 'source_ids': ['turn1']},
                     'evidence': [{'text': '해석', 'source_ids': []}], 'differential': 'oops'}
        note, _, blocked = finalize(candidate, self.items)
        self.assertIn('관찰 근거:', note['A']['evidence'])
        self.assertIn('식욕이 좋아졌어요', note['A']['evidence'])
        self.assertIn('식욕이 좋아졌어요', note['S']['hpi'])
        self.assertFalse(note['S']['hpi'].startswith('요약'))
        self.assertIn('나는 32세 여자이다', note['S']['cc'])
        self.assertEqual({b['field'] for b in blocked}, {'S.hpi', 'A.evidence'})

    def test_differentials_require_real_source_ids(self):
        note, _, _ = finalize({'diagnosis': '가설', 'differential': [
            {'name': '정상 식욕', 'source_ids': ['missing']},
            {'name': '감별 가설', 'source_ids': ['turn1']} ]}, self.items)
        self.assertNotIn('정상 식욕', note['A']['differential'])
        self.assertIn('식욕이 좋아졌어요', note['A']['differential'])

    def test_consent_is_not_a_symptom_or_retrieval_fact(self):
        items = sources(self.session, ['턴 1 의사: TSH, FT4, T3 검사를 할까요?', '환자: 네, 검사를 해요.',
                                       '턴 2 의사: TSH 수용체 항체 검사 추가하실래요?', '환자: 네, 추가하겠습니다.',
                                       '턴 3 의사: 과거 검사 결과가 있나요?', '환자: 몰라요'])
        self.assertEqual([i['kind'] for i in items[2:]], ['communication', 'communication', 'say'])
        note, _, _ = finalize({}, items)
        self.assertNotIn('검사를 해요', json.dumps(note, ensure_ascii=False))
        self.assertFalse(any(i['id'].startswith(('turn1.', 'turn2.')) for i in retrieval_sources(items)))


if __name__ == '__main__':
    unittest.main()
