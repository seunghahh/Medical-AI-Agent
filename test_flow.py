"""Consultation flow: required history before DONE, the per-patient plan, and the closing conversation."""
import json
import unittest
from unittest.mock import patch

import run
from grounding import asks, sources
from nova_local import LocalClient
from test_contract import Endpoint, simulator

NOTE = {'diagnosis': '진단 미정'}
MESSAGES = ['걱정 많으셨겠어요.', '검사로 원인을 확인하겠습니다.', '심해지면 바로 오셔야 해요.']
CLOSING = [{'message': m} for m in MESSAGES]


def say(text):
    return {'action': 'SAY', 'content': text}


def run_consult(replies, max_turns, patient=20):
    sim = simulator([{'content': '네, 그래요'}] * patient)
    sim.doctor = Endpoint(replies)
    client = LocalClient(sim)
    spoken = []
    original = client.say
    client.say = lambda content, **kw: spoken.append(content) or original(content, **kw)
    run.consult(client, {'max_turns': max_turns})
    return sim, spoken


@patch.object(run, 'PLAN_VISIT', False)   # the scripted replies below are for the generic checklist; planning has its own tests
class FlowTests(unittest.TestCase):
    def test_missing_topics_reads_the_doctors_questions(self):
        self.assertEqual(run.missing_topics([]), ['과거력', '복용 약·알레르기', '가족력', '사회력', '신체 진찰'])
        transcript = ['턴 1 의사: 과거에 앓은 병이 있나요?', '환자: 없어요', '턴 2 의사: 드시는 약이나 알레르기는요?', '환자: 없어요',
                      '턴 3 의사: 가족 중에 아픈 분 계세요?', '환자: 없어요', '턴 4 의사: 수술 받은 적 있나요?', '환자: 없어요',
                      '턴 5 진찰(복부 촉진): 압통 없음']
        self.assertEqual(run.missing_topics(transcript), ['사회력'])  # '수술' must not count as drinking
        self.assertEqual(run.missing_topics(transcript + ['턴 6 의사: 담배나 술을 하시나요?']), [])

    def test_done_is_sent_back_until_history_is_taken_then_patient_gets_an_explanation(self):
        questions = ['과거에 앓은 병이 있나요?', '복용하는 약이나 알레르기는요?', '가족 중에 같은 병이 있나요?', '담배나 술을 하시나요?']
        replies = [{'action': 'DONE'}, *map(say, questions), {'action': 'DONE'}, {'action': 'DONE'}, *CLOSING, NOTE]
        sim, spoken = run_consult(replies, max_turns=12)
        self.assertEqual(spoken, questions + MESSAGES)
        self.assertEqual(sim.turns, 7)
        self.assertIn('확인할 항목', json.dumps(sim.doctor.messages, ensure_ascii=False))

    def test_gathering_stops_early_enough_to_leave_turns_for_the_explanation(self):
        replies = [*[say(f'{n}번째 증상이 어떤가요?') for n in range(6)], CLOSING[0], CLOSING[2], NOTE]
        sim, spoken = run_consult(replies, max_turns=8)
        self.assertEqual(sim.turns, 8)
        self.assertEqual(spoken[-2:], [MESSAGES[0], MESSAGES[2]])

    def test_the_closing_reads_the_patients_last_reply_and_is_tagged(self):
        sim = simulator([{'content': '혹시 큰 병은 아닌가요?'}, {'content': '네'}, {'content': '네'}])
        sim.doctor = Endpoint(CLOSING)
        client = LocalClient(sim)
        transcript = []
        run.explain(client, sim.public_session(), transcript, 5, 0)
        asked = json.dumps(sim.doctor.messages, ensure_ascii=False)
        self.assertIn('혹시 큰 병은 아닌가요?', json.dumps(sim.doctor.messages[1], ensure_ascii=False))  # 2nd line saw the reply
        self.assertEqual([l for l in transcript if l.startswith('턴')],
                         [f'턴 {n} 의사(설명): {m}' for n, m in enumerate(MESSAGES, 1)])
        self.assertNotIn('의사:', ' '.join(transcript))
        self.assertIn('이번 말(3/3)', asked)

    def test_the_closing_may_ask_a_question_without_polluting_the_record(self):
        session = {'initial': {'opening': '배가 아파요', 'vital_signs': '미제공'}}
        transcript = ['턴 1 의사(설명): 궁금한 점 있으세요?', '환자: 네, 약은 얼마나 먹나요?',
                      '턴 2 의사: 언제부터 아팠나요?', '환자: 어제요']
        kinds = [(i['id'], i['kind']) for i in sources(session, transcript)[2:]]
        self.assertEqual(kinds, [('turn1', 'communication'), ('turn2', 'say')])

    def test_the_last_closing_line_always_carries_warning_signs(self):
        def last(*replies):
            sim = simulator(); sim.doctor = Endpoint(list(replies))
            return run.next_explanation(LocalClient(sim), sim.public_session(), [], 3, 3)
        self.assertEqual(last({'message': '궁금한 점 있으세요?'}, {'message': '카페인을 줄이세요.'}), run.FALLBACK_EXPLANATION[2])
        self.assertEqual(last({'message': '심해지면 응급실로 오세요.'}), '심해지면 응급실로 오세요.')
        self.assertEqual(last({'message': 'x' * 31}, {'nothing': 1}), run.FALLBACK_EXPLANATION[2])   # unusable twice
        self.assertTrue(all(0 < len(m) <= run.MAX_CHARS for m in run.FALLBACK_EXPLANATION))

    def test_only_the_first_of_several_questions_is_sent(self):
        sim = simulator(); sim.doctor = Endpoint([say('흡연을 하시나요? 어떤 일을 하시나요?')])
        transcript = []
        step = run.next_step(LocalClient(sim), sim.public_session(), transcript, 20)
        self.assertEqual(step, ('SAY', '흡연을 하시나요?'))
        self.assertIn('보류한 질문: 어떤 일을 하시나요?', transcript[-1])
        sim.doctor = Endpoint([say('어디가 아픈가요?')])
        quiet = []
        self.assertEqual(run.next_step(LocalClient(sim), sim.public_session(), quiet, 20), ('SAY', '어디가 아픈가요?'))
        self.assertEqual(quiet, [])

    def test_too_long_empathy_plus_question_is_cut_to_the_question(self):
        self.assertEqual(run.fit_to_limit('잠을 못 주무셔서 많이 힘드셨겠어요. 언제부터 잠들기 어려우셨나요?'), '언제부터 잠들기 어려우셨나요?')
        self.assertEqual(run.fit_to_limit('x' * 45), '')
        sim = simulator(); sim.doctor = Endpoint([say('잠을 못 주무셔서 많이 힘드셨겠어요. 언제부터 잠들기 어려우셨나요?')])
        self.assertEqual(run.next_step(LocalClient(sim), sim.public_session(), [], 20), ('SAY', '언제부터 잠들기 어려우셨나요?'))

    def test_a_visit_does_not_end_without_a_single_question(self):
        garbage = [{'oops': 1}] * 3                      # next_step gives up: no action in the right format
        replies = [*garbage, {'action': 'DONE'}, {'action': 'DONE'}, {'action': 'DONE'}, *CLOSING, NOTE]
        sim, spoken = run_consult(replies, max_turns=12)
        self.assertEqual(spoken[0], run.FIRST_QUESTION)
        self.assertEqual(spoken[1:], MESSAGES)
        self.assertEqual(sim.turns, 4)

    def test_children_are_not_asked_about_smoking_drinking_or_work(self):
        session = {'initial': {'opening': '이름은 김도현이고, 여섯 살 남자아이이다.', 'vital_signs': ''}}
        self.assertEqual(run.patient_age(session), 6)
        self.assertEqual(run.patient_age({'initial': {'opening': '저는 예순두 살 여자이다.'}}), 62)
        self.assertIsNone(run.patient_age({'initial': {'opening': '배가 아파요'}}))
        self.assertNotIn('사회력', run.missing_topics([], age=6))
        self.assertIn('사회력', run.missing_topics([], age=40))
        self.assertIn('사회력', run.missing_topics([]))              # unknown age: ask

    def test_patient_replies_to_the_explanation_are_not_observations(self):
        session = {'initial': {'opening': '배가 아파요', 'vital_signs': '미제공'}}
        transcript = []
        for number, message in enumerate(MESSAGES, 1):
            transcript += [f'턴 {number} 의사(설명): {message}', '환자: 네, 알겠습니다']
        kinds = [item['kind'] for item in sources(session, transcript)[2:]]
        self.assertEqual(kinds, ['communication'] * 3)

    def test_a_broken_explanation_does_not_lose_the_record(self):
        replies = [say('과거에 앓은 병이 있나요?'), {'action': 'DONE'}, {'action': 'DONE'}, {'action': 'DONE'}, NOTE]
        with patch.object(run, 'explain', side_effect=RuntimeError('boom')):
            sim, spoken = run_consult(replies, max_turns=12)
        self.assertEqual(spoken, ['과거에 앓은 병이 있나요?'])
        self.assertIsNotNone(sim.record)  # the record was still submitted

    def test_rag_can_be_switched_off_for_a_comparison_run(self):
        with patch.dict('os.environ', {'NOVA_RAG': '0'}):
            self.assertIsNone(run.prepare()['rag'])


class PlanTests(unittest.TestCase):
    PLAN = {'history': [{'name': '수면제 용량 변화', 'kw': ['용량', '늘']}, {'name': '', 'kw': ['x']}, 'junk',
                        {'name': '금단 증상', 'kw': []}],
            'exams': [{'name': '기립성 혈압', 'kw': ['혈압']}]}

    def plan(self, *replies):
        sim = simulator(); sim.doctor = Endpoint(list(replies))
        return run.plan_visit(LocalClient(sim), sim.public_session())

    def test_the_plan_keeps_only_usable_items_and_always_asks_about_the_main_worry(self):
        items = self.plan(self.PLAN)
        self.assertEqual([i['name'] for i in items], ['수면제 용량 변화', '기립성 혈압', '가장 걱정되는 점'])
        self.assertEqual(items[1]['kind'], '진찰')
        self.assertEqual(self.plan({'history': 'nope'}, {'exams': []}), [])      # nothing usable twice: no plan

    def test_items_count_as_done_when_the_doctor_used_their_words(self):
        items = self.plan(self.PLAN)
        transcript = ['턴 1 의사: 수면제 용량이 늘었나요?', '환자: 네', '턴 2 진찰(혈압을 재겠습니다): 120/80']
        self.assertEqual(run.remaining_items(items, transcript), ['가장 걱정되는 점'])
        self.assertEqual(run.remaining_items(items, transcript + ['턴 3 의사(설명): 가장 걱정되는 건 이거죠']),
                         ['가장 걱정되는 점'])                       # closing lines never count
        self.assertEqual(run.remaining_items(items, transcript + ['턴 3 의사: 가장 걱정되시는 점은 무엇인가요?']), [])

    def test_the_next_step_sees_what_is_left_and_done_waits_for_it(self):
        replies = [self.PLAN, {'action': 'DONE'}, say('수면제 용량이 늘었나요?'), say('가장 걱정되시는 점은 무엇인가요?'),
                   say('혈압을 재겠습니다'), {'action': 'DONE'}]
        replies[4] = {'action': 'EXAM', 'content': '혈압을 재겠습니다'}
        sim = simulator([{'content': '네'}] * 8)
        sim.doctor = Endpoint([*replies, *CLOSING, NOTE])
        client = LocalClient(sim)
        spoken = []
        original = client.say
        client.say = lambda content, **kw: spoken.append(content) or original(content, **kw)
        with patch.object(client, 'exam', return_value={'result': '120/80', 'turn': 4, 'turns_remaining': 40}), \
                patch.object(run, 'PLAN_VISIT', True):
            run.consult(client, {'max_turns': 12})
        seen = json.dumps(sim.doctor.messages, ensure_ascii=False)
        self.assertIn('아직 안 한 필수 항목', seen)
        self.assertIn('수면제 용량 변화', seen)
        self.assertEqual(spoken[:2], ['수면제 용량이 늘었나요?', '가장 걱정되시는 점은 무엇인가요?'])
        self.assertEqual(spoken[2:], MESSAGES)

    def test_the_plan_and_what_was_left_undone_are_kept_for_the_logs(self):
        plan = {'history': [{'name': '수면제 용량 변화', 'kw': ['용량']}], 'exams': []}
        sim = simulator([{'content': '네'}] * 8)
        sim.doctor = Endpoint([plan, {'action': 'DONE'}, {'action': 'DONE'}, {'action': 'DONE'}, {'action': 'DONE'}, *CLOSING, NOTE])
        state = {'max_turns': 12}
        with patch.object(run, 'PLAN_VISIT', True):
            run.consult(LocalClient(sim), state)
        self.assertEqual([i['name'] for i in state['visit_plan']['items']], ['수면제 용량 변화', '가장 걱정되는 점'])
        self.assertEqual(state['visit_plan']['left_undone'], ['수면제 용량 변화', '가장 걱정되는 점'])   # DONE won 3 nudges later

    def test_the_model_reports_which_planned_item_an_action_covers(self):
        items = self.plan(self.PLAN)
        covered = set()
        sim = simulator()
        sim.doctor = Endpoint([{'action': 'SAY', 'content': '그 약을 어떻게 드세요?', 'item': 1}])
        step = run.next_step(LocalClient(sim), sim.public_session(), [], 20, '', items, covered)
        self.assertEqual((step, covered), (('SAY', '그 약을 어떻게 드세요?'), {1}))
        self.assertEqual([n for n, _ in run.remaining_numbered(items, [], covered)], [2, 3])      # keyword-free progress
        seen = json.dumps(sim.doctor.messages, ensure_ascii=False)
        self.assertIn('1. 수면제 용량 변화, 2. 기립성 혈압, 3. 가장 걱정되는 점', seen)               # numbered list offered
        for bad in ('x', 0, 9, True and 99):                                                     # nonsense numbers are ignored
            sim.doctor = Endpoint([{'action': 'SAY', 'content': '질문입니다?', 'item': bad}])
            run.next_step(LocalClient(sim), sim.public_session(), [], 20, '', items, covered)
        self.assertEqual(covered, {1})

    def test_action_selection_gets_a_short_hint_not_the_guideline_text(self):
        state = {'retrievals': [{'thought': '수면제 의존 가능성. ' * 80, 'hits': [{'title': 'Insomnia CPG'}, {'title': 'Zolpidem'}]}]}
        hint = run.step_hint(state)
        self.assertTrue(hint.startswith('참고(지침 검색 요약): 수면제 의존 가능성'))
        self.assertIn('찾은 문서: Insomnia CPG; Zolpidem', hint)
        self.assertLess(len(hint), 600)
        self.assertEqual(run.step_hint({}), '')
        self.assertEqual(run.step_hint(None), '')

    def test_tiny_budgets_skip_the_plan(self):
        # With 4 turns a plan call would eat the first scripted reply (DONE); the closing and the record follow it.
        with patch.object(run, 'PLAN_VISIT', True):
            sim, spoken = run_consult([{'action': 'DONE'}, *CLOSING, NOTE], max_turns=4)
        self.assertEqual(spoken, MESSAGES)
        self.assertIsNotNone(sim.record)


if __name__ == '__main__':
    unittest.main()
