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



class ReplyEndpoint(PhaseEndpoint):
    """PhaseEndpoint that also answers the closing-reply calls with a chosen text."""
    def __init__(self, text='약 걱정을 함께 확인해 가요.', summary_text='석 달 동안 살이 빠지셨군요.', exam_ids=()):
        super().__init__([])
        self.text, self.summary_text, self.exam_ids, self.reply_calls = text, summary_text, list(exam_ids), 0

    def complete(self, messages, **kwargs):
        system = messages[0]['content']
        if system.startswith('You write one short Korean reply'):
            self.messages.append(messages)
            self.reply_calls += 1
            text = self.summary_text if 'Reflect back' in messages[1]['content'] else self.text
            return json.dumps({'text': text}, ensure_ascii=False), 'stop', 10, 10
        if 'catalog=' in system:
            self.messages.append(messages)
            return json.dumps({'ids': self.exam_ids}), 'stop', 10, 10
        return super().complete(messages, **kwargs)


def run_closing(says, max_turns=40, text='약 걱정을 함께 확인해 가요.', exam_ids=(), **endpoint):
    """A whole consultation where the patient's answers come from `says` (question text -> answer, else default)."""
    sim = simulator(); sim.doctor = ReplyEndpoint(text, exam_ids=exam_ids, **endpoint)
    client = LocalClient(sim)
    session = client.session(); session['turns_remaining'] = max_turns
    spoken = []
    original = client.say
    def capture(content, **kwargs):
        spoken.append(content)
        return original(content, **kwargs)
    def patient(utterance):
        return says(utterance), utterance
    exams = []
    def examine(content, **kwargs):
        exams.append(content)
        return {'result': '특이 소견 없음', 'turn': len(exams), 'turns_remaining': 99}
    with patch.object(client, 'session', return_value=session), patch.object(client, 'say', side_effect=capture), \
         patch.object(client, 'exam', side_effect=examine), patch.object(sim, 'patient', side_effect=patient):
        run.consult(client, {})
    sim.exams = exams
    return sim, spoken


UNKNOWN = '미확인: 원자료에 해당 정보가 없습니다.'


class ClosingReplyTests(unittest.TestCase):
    def test_the_worry_is_answered_right_after_the_cause(self):
        sim, spoken = run_closing(lambda q: '약이 자꾸 늘어서 걱정돼요.' if q == run.CONCERN_QUESTION else UNKNOWN)
        closing = spoken[len(run.history_queue(sim.public_session()['initial']['opening'])):]
        self.assertEqual(closing[0], run.TRANSITION)                           # too little history to reflect back
        at = closing.index('약 걱정을 함께 확인해 가요.')
        self.assertEqual(closing[at - 1], '원인은 더 확인이 필요해요.')        # the cause line comes first
        self.assertEqual(closing[-1], run.INVITATION)
        self.assertIn('[실제 설명/concern]', sim.record['P']['education'])
        self.assertEqual(sim.doctor.reply_calls, 1)

    def test_no_worry_line_without_a_usable_answer(self):
        sim, spoken = run_closing(lambda q: UNKNOWN)
        self.assertEqual(sim.doctor.reply_calls, 0)
        self.assertNotIn('[실제 설명/concern]', sim.record['P']['education'])
        self.assertEqual(run.concern_words(['턴 1 의사: ' + run.CONCERN_QUESTION, '환자: ' + UNKNOWN]), '')
        self.assertEqual(run.concern_words(['턴 1 의사: ' + run.CONCERN_QUESTION, '환자: 네']), '')       # too short to use
        self.assertEqual(run.concern_words(['턴 1 의사: 아무 질문', '환자: 걱정돼요 많이']), '')

    def test_a_question_during_the_closing_gets_an_answer_but_at_most_twice(self):
        asked = {text for _, text in run.history_queue('배가 아파요')}
        sim, spoken = run_closing(lambda q: UNKNOWN if q in asked else '혹시 큰 병은 아닌지 걱정돼요.',
                                  text='걱정하시는 점도 검사로 확인해요.')
        self.assertEqual(spoken.count('걱정하시는 점도 검사로 확인해요.'), 1)    # the same sentence is never sent twice
        self.assertEqual(spoken.count(run.SAFE_REPLIES['answer']), 1)         # the second answer fell back to the fixed line
        self.assertEqual(sim.doctor.reply_calls, 3)                           # 1st answer, then 2nd answer plus one retry
        self.assertEqual(spoken[-1], run.INVITATION)

    def test_an_unsafe_or_unusable_reply_is_replaced_by_a_fixed_neutral_line(self):
        for bad in ('이 병은 반드시 완치돼요.', '혹시 다른 증상이 있나요?', '약 20mg을 드세요.', '이건 한 문장으로 말하기에는 너무 길고 복잡한 설명이라서요.',
                    '암은 아니에요.', '큰 병은 없어요.'):
            sim, spoken = run_closing(lambda q: '약이 걱정돼요.' if q == run.CONCERN_QUESTION else UNKNOWN, text=bad)
            self.assertIn(run.SAFE_REPLIES['concern'], spoken, bad)
            self.assertNotIn(bad, spoken)

    def test_a_tight_budget_keeps_the_planned_lines_and_skips_the_extras(self):
        sim, spoken = run_closing(lambda q: '이게 괜찮을까요?', max_turns=5)    # a worried patient, but only five turns
        self.assertEqual(sim.doctor.reply_calls, 0)                           # no room for a lead line or an answer
        self.assertEqual(len(spoken), 5)
        self.assertIn('검사로 원인을 확인해요.', spoken)
        self.assertTrue(any('진료' in m for m in spoken))                      # the warning line is still sent

    def test_a_short_empathy_phrase_follows_a_distressed_answer_only(self):
        question = '이전에 앓은 병이나 수술받은 적이 있나요?'            # 22 characters
        def prefix(answer, text=question, used=0):
            return run.empathy_prefix(['턴 1 의사: 어디가 불편한가요?', '환자: ' + answer], text, used)
        self.assertEqual(prefix('배가 너무 아파요.'), '아프시죠. ')                 # the longer phrase would not fit in 30
        self.assertEqual(prefix('배가 너무 아파요.', '언제부터인가요?'), '많이 아프시죠. ')   # a short question leaves room
        self.assertEqual(prefix('앞으로가 걱정돼요.'), '걱정되시죠. ')
        self.assertEqual(prefix('요즘 잠을 못 자서 힘들어요.'), '힘드시죠. ')
        for none in ('아프지 않아요.', '괜찮아요.', UNKNOWN, '네.'):
            self.assertEqual(prefix(none), '', none)
        self.assertEqual(prefix('배가 너무 아파요.', used=run.MAX_EMPATHY), '')       # never more than three per visit
        self.assertEqual(prefix('걱정돼요.', run.CONCERN_QUESTION), '')                 # that question already opens with empathy
        self.assertEqual(run.empathy_prefix([], question, 0), '')
        self.assertEqual(run.empathy_prefix(['턴 1 의사: 아프시죠. 언제부터인가요?', '환자: 많이 아파요.'], question, 0), '')   # not twice in a row
        self.assertTrue(all(len(prefix('배가 아파요.', 'x' * n) + 'x' * n) <= run.MAX_CHARS or prefix('배가 아파요.', 'x' * n) == ''
                            for n in range(1, 31)))                                   # whatever the question, never over 30

    def test_the_consultation_uses_the_empathy_phrase_at_most_three_times(self):
        asked = {text for _, text in run.history_queue('배가 아파요')}
        sim, spoken = run_closing(lambda q: '배가 아파서 힘들어요.' if any(q.endswith(a) for a in asked) else UNKNOWN)
        prefixed = [m for m in spoken if m.startswith(('아프시죠.', '많이 아프시죠.', '걱정되시죠.', '힘드시죠.'))]
        self.assertEqual(len(prefixed), run.MAX_EMPATHY)
        self.assertTrue(all(len(m) <= run.MAX_CHARS for m in spoken))
        self.assertTrue(all(any(m.endswith(a) for a in asked) for m in prefixed))   # still one of the planned questions

    def test_the_summary_opens_the_exam_and_the_worry_is_answered_in_the_closing(self):
        queue = run.history_queue('배가 아파요')
        said = {queue[0][1]: '약이 자꾸 늘어서 걱정돼요.', queue[1][1]: '석 달 전부터 살이 빠졌어요.', queue[2][1]: '두근거림도 있어요.'}
        def answer(question):
            return next((said[known] for known in said if question.endswith(known)), UNKNOWN)
        sim, spoken = run_closing(answer)
        closing = spoken[len(queue):]
        self.assertEqual(closing[0], '석 달 동안 살이 빠지셨군요.' + run.EXAM_CUE)
        self.assertIn('약 걱정을 함께 확인해 가요.', closing[1:])
        self.assertEqual(sim.doctor.reply_calls, 2)                            # the summary and the worry answer
        self.assertEqual(closing[-1], run.INVITATION)

    def test_a_one_line_summary_opens_the_exam_when_there_is_no_usable_worry(self):
        queue = run.history_queue('배가 아파요')
        said = {queue[1][1]: '석 달 전부터 살이 빠졌어요.', queue[2][1]: '두근거림도 있어요.'}
        def answer(question):
            return next((said[known] for known in said if question.endswith(known)), UNKNOWN)
        sim, spoken = run_closing(answer)
        self.assertEqual(spoken[len(queue)], '석 달 동안 살이 빠지셨군요.' + run.EXAM_CUE)
        self.assertEqual(sim.doctor.reply_calls, 1)

    def test_no_summary_when_the_patient_told_us_too_little(self):
        sim, spoken = run_closing(lambda q: UNKNOWN)
        self.assertEqual(sim.doctor.reply_calls, 0)

    def test_up_to_max_exams_are_done_and_the_closing_still_fits(self):
        ids = ['heart', 'lungs', 'abdomen', 'thyroid', 'eyes', 'tremor']
        sim, spoken = run_closing(lambda q: UNKNOWN, exam_ids=ids)
        self.assertEqual(sim.exams, [run.EXAM_CATALOG[k] for k in ids[:run.MAX_EXAMS]])
        self.assertTrue(any('진료' in m for m in spoken))                            # the warning line is still sent

    def test_exams_stop_early_enough_to_keep_the_closing_lines(self):
        ids = ['heart', 'lungs', 'abdomen', 'thyroid', 'eyes']
        sim, spoken = run_closing(lambda q: UNKNOWN, max_turns=12, exam_ids=ids)   # nine of twelve turns belong to the closing
        self.assertEqual(len(sim.exams), 3)
        self.assertEqual(len(spoken), 6)
        self.assertEqual(spoken[-1], run.INVITATION)
        self.assertTrue(any('진료' in m for m in spoken))

    def test_what_counts_as_a_question_or_worry(self):
        for yes in ('그럼 약을 줄일 방법도 있을까요?', '혹시 큰 병은 아닌지 걱정돼요.', '무서워요', '괜찮을까요'):
            self.assertTrue(run.needs_answer(yes), yes)
        for no in ('네, 알겠습니다.', '네, 검사받고 원인을 알고 싶어요.', UNKNOWN):
            self.assertFalse(run.needs_answer(no), no)


class FlowTests(unittest.TestCase):
    def test_fixed_history_precedes_model_done_and_actual_explanations(self):
        sim, spoken, result = run_consult(30)
        required = [q for _, q in run.history_queue(sim.public_session()['initial']['opening'])]
        self.assertEqual(spoken[:len(required)], required)
        self.assertEqual(spoken[len(required)], run.TRANSITION)   # the exam is announced before it starts
        closing = spoken[len(required) + 1:]
        self.assertEqual(len(closing), 6)                       # five planned lines and the closing invitation
        self.assertEqual(closing[-1], run.INVITATION)
        self.assertTrue(all(not asks(text) for text in closing[:-1]))   # only the last line asks (understanding check)
        self.assertIn('2주 뒤 결과와 경과를 확인해요.', closing)
        self.assertEqual(result['status'], 'completed')
        self.assertNotIn('HIDDEN', json.dumps(sim.doctor.messages))
        self.assertIn('[실제 설명/', sim.record['P']['education'])

    def test_small_budget_reserves_closing_and_final_record(self):
        sim, spoken, result = run_consult(8)
        self.assertLessEqual(sim.turns, 8)
        self.assertEqual(len(spoken), 6)
        self.assertEqual(spoken[-1], run.INVITATION)
        self.assertTrue(all(not asks(text) for text in spoken[:-1]))
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
        self.assertIn(run.TRANSITION, spoken)
        self.assertTrue(all(asks(text) for text in spoken if text != run.TRANSITION))

    def test_rag_can_be_switched_off(self):
        with patch.dict('os.environ', {'NOVA_RAG': '0'}):
            self.assertIsNone(run.prepare()['rag'])


if __name__ == '__main__':
    unittest.main()


class ClosingContentTests(unittest.TestCase):
    """The general-purpose closing fixes: follow-up times, answers from the plan, no repeated lines, a self-care line."""

    def note(self, **plan):
        base = {'tests': 'TSH test', 'treatment': 'Review first', 'follow_up': 'Review within 1–2 weeks.',
                'education': 'Do not stop the medicine abruptly.'}
        base.update(plan)
        return {'P': base, 'S': {'hpi': '', 'fh': '', 'pmh': '스틸녹스를 먹어요'}}

    def client(self, *replies):
        sim = simulator(); sim.doctor = Endpoint(list(replies))
        return LocalClient(sim), sim

    def test_follow_up_times_in_hours_and_in_words(self):
        plan = 'Reassess within 24 hours, then within one week.'
        self.assertEqual(run.follow_up_time(plan), ('24', '시간'))
        self.assertEqual(run.followup_sentence(plan), '24시간 안에 상태를 다시 확인해요.')
        self.assertEqual(run.follow_up_time('See you again in a week.'), ('1', '주'))
        self.assertEqual(run.followup_sentence('Return in two weeks.'), '2주 뒤 결과와 경과를 확인해요.')
        self.assertEqual(run.followup_sentence('Return in 6–12 months.'), '6~12개월 뒤 결과와 경과를 확인해요.')
        self.assertIsNone(run.follow_up_time('Return when needed.'))

    def test_a_when_question_gets_the_time_the_plan_states_without_a_model_call(self):
        client, sim = self.client({'text': '결과가 나오면 함께 살펴봐요.'})      # used only for the repeated question below
        text = run.reply_message(client, 'answer', '검사 결과는 언제쯤 알 수 있을까요?', self.note(), 'dx', set())
        self.assertEqual(text, '결과는 1~2주 안에 함께 봐요.')
        self.assertEqual(sim.doctor.messages, [])                     # answered from the plan: no model call
        again = run.reply_message(client, 'answer', '결과는 언제 나오나요?', self.note(), 'dx', {text})
        self.assertEqual(again, '결과가 나오면 함께 살펴봐요.')           # the time was already said: a different sentence

    def test_numbers_are_allowed_only_when_the_plan_or_the_patient_used_them(self):
        client, sim = self.client({'text': '3주 뒤에 함께 정해요.'}, {'text': '2주 안에 함께 정해요.'})
        self.assertEqual(run.reply_message(client, 'answer', '어떻게 하나요?', self.note(), 'dx', set()), '2주 안에 함께 정해요.')
        client, sim = self.client({'text': '3주 뒤에 함께 정해요.'}, {'text': '3개월 걸려요.'})
        self.assertEqual(run.reply_message(client, 'answer', '어떻게 하나요?', self.note(), 'dx', set()),
                         run.SAFE_REPLIES['answer'])         # neither attempt used a number from the plan

    def test_the_summary_request_has_no_plan_tail_and_the_answer_request_sees_the_education_text(self):
        client, sim = self.client({'text': '석 달 동안 살이 빠지셨군요.'}, {'text': '갑자기 끊지 말고 천천히 줄여요.'})
        run.reply_message(client, 'summary', '석 달 전부터 살이 빠졌어요', self.note(), 'dx', set())
        run.reply_message(client, 'answer', '약을 줄일 방법이 있을까요?', self.note(), 'dx', set())
        summary_request, answer_request = (m[1]['content'] for m in sim.doctor.messages)
        self.assertIn('Do not mention the plan', summary_request)
        self.assertNotIn('decided together', summary_request)
        self.assertIn('Do not stop the medicine abruptly', answer_request)
        self.assertIn('gradually', answer_request)

    def test_a_line_that_repeats_an_earlier_one_is_not_sent_again(self):
        said = {'큰 병일까 걱정되셨죠, 원인 확인 후 함께 정해요.'}
        client, sim = self.client({'text': '석 달간 살 빠져 걱정되셨죠, 원인 확인해 함께 정해요.'}, {'text': '석 달 동안 살이 빠지셨군요.'})
        self.assertEqual(run.reply_message(client, 'summary', '살이 빠졌어요', self.note(), 'dx', said), '석 달 동안 살이 빠지셨군요.')
        self.assertTrue(run.similar('큰 병일까 걱정되셨죠, 원인 확인 후 함께 정해요.', '석 달간 살 빠져 걱정되셨죠, 원인 확인해 함께 정해요.'))
        self.assertFalse(run.similar('검사로 원인을 확인해요.', '검사 확인 후 치료 방향을 함께 정해요.'))
        self.assertFalse(run.similar('금단·중독 상태를 평가해 안전한 치료 수준을 정해요.', '검사로 정확히 확인하고 함께 정해요.'))

    def test_the_explanation_gets_a_self_care_line_and_may_name_a_drug_the_patient_mentioned(self):
        client, sim = self.client(*[{'text': t} for t in (
            '스틸녹스를 오래 먹어 몸이 약에 익숙해졌을 수 있어요.', '피검사로 원인을 확인해요.',
            '확인 후 약을 천천히 줄이는 계획을 세워요.', '갑자기 끊지 말고 천천히 줄여요.', '심해지면 바로 진료받아요.')])
        output = run.patient_explanations(client, self.note(), 'Zolpidem use disorder')
        self.assertEqual([topic for topic, _ in output], ['cause', 'next', 'treatment', 'follow_up', 'self_care', 'warning'])
        first = sim.doctor.messages[0]
        self.assertIn('may be used exactly as they said it', first[0]['content'])
        self.assertIn('스틸녹스를 먹어요', first[1]['content'])               # the cause call sees the patient's medication answer
        self.assertIn('금단', first[0]['content'])                              # and is told to use everyday words
        self.assertEqual(output[2], ('treatment', '확인 후 약을 천천히 줄이는 계획을 세워요.'))  # explained, not a fixed line


class GeneralTests(unittest.TestCase):
    """Fixes found by running diseases other than the three samples."""

    def test_ages_are_read_from_korean_english_and_months(self):
        for text, expected in (('저는 서른두 살 여자이다.', 32.0), ('나는 62세 남자이다.', 62.0), ('35-year-old female. Double vision', 35.0),
                               ('8-month-old boy. Crying', 8 / 12), ('생후 8개월 남아가 울어요.', 8 / 12), ('8개월 된 아기가 배가 불러요.', 8 / 12)):
            self.assertAlmostEqual(run.age_in_years(text), expected, msg=text)
        self.assertIsNone(run.age_in_years('3개월 전부터 살이 빠졌다.'))      # a duration, not an age
        self.assertIsNone(run.age_in_years('배가 아파요'))

    def test_an_infant_is_never_asked_about_smoking_or_drinking(self):
        for opening in ('8-month-old boy. Crying', '생후 8개월 남아가 울어요.', '여섯 살 남자아이이다.'):
            queue = dict(run.history_queue(opening))
            self.assertNotIn('alcohol', queue, opening)
            self.assertEqual(queue['smoking'], '가정에서 담배 연기 노출이 있나요?', opening)
            self.assertIn('birth', queue, opening)
        adult = dict(run.history_queue('35-year-old female. Double vision'))
        self.assertIn('alcohol', adult)
        self.assertIn('menstrual', adult)                                     # an adult woman, even when the opening is English

    def test_no_exam_request_bundles_two_things_and_the_new_ones_fit_the_limit(self):
        for key, phrase in run.EXAM_CATALOG.items():
            self.assertLessEqual(len(phrase), run.MAX_CHARS, key)
            self.assertNotIn('상하지', phrase, key)                            # arms and legs were refused as "not specific"
        for key in ('arm_strength', 'leg_strength', 'eye_movement', 'ptosis', 'pupils', 'neck_stiffness', 'cva', 'ears', 'romberg'):
            self.assertIn(key, run.EXAM_CATALOG)
        self.assertNotIn('strength', run.EXAM_CATALOG)

    def test_dont_know_and_not_confirmed_are_not_answers(self):
        for no in ('무엇이 가장 걱정되는지는 잘 모르겠어요.', '가족력은 확인되지 않았어요.', '확인 안 됐어요', '그건 모르겠어요', '미확인입니다.'):
            self.assertTrue(run.UNKNOWN_REPLY.search(no), no)
            self.assertEqual(run.concern_words(['턴 1 의사: ' + run.CONCERN_QUESTION, '환자: ' + no]), '', no)
        self.assertIsNone(run.UNKNOWN_REPLY.search('살이 계속 빠지는 이유가 큰 병일까 봐 걱정돼요.'))

    def test_the_summary_facts_leave_out_the_answer_about_the_worry(self):
        session = {'initial': {'opening': '배가 아파요', 'vital_signs': ''}}
        transcript = ['턴 1 의사: ' + run.CONCERN_QUESTION, '환자: 큰 병일까 봐 걱정돼요.',
                      '턴 2 의사: 언제부터인가요?', '환자: 석 달 전부터 시작됐어요.',
                      '턴 3 의사: 다른 증상이 있나요?', '환자: 두근거림이 있어요.']
        facts = run.patient_facts(session, transcript)
        self.assertIn('석 달 전부터', facts)
        self.assertNotIn('큰 병일까', facts)
        self.assertEqual(run.patient_facts(session, transcript[:4]), '')        # only one usable answer left: too little
