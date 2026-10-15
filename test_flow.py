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
            text = self.summary_text if 'Summarize' in messages[1]['content'] else self.text
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
    def test_the_closing_opens_with_a_line_about_the_patients_own_worry(self):
        sim, spoken = run_closing(lambda q: '약이 자꾸 늘어서 걱정돼요.' if q == run.CONCERN_QUESTION else UNKNOWN)
        closing = spoken[len(run.history_queue(sim.public_session()['initial']['opening'])):]
        self.assertEqual(closing[0], '약 걱정을 함께 확인해 가요.')
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
        for bad in ('이 병은 반드시 완치돼요.', '혹시 다른 증상이 있나요?', '약 20mg을 드세요.', '이건 한 문장으로 말하기에는 너무 길고 복잡한 설명이라서요.'):
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

    def test_a_one_line_summary_comes_first_when_the_patient_told_us_enough(self):
        queue = run.history_queue('배가 아파요')
        said = {queue[0][1]: '약이 자꾸 늘어서 걱정돼요.', queue[1][1]: '석 달 전부터 살이 빠졌어요.'}
        def answer(question):
            for known in said:
                if question.endswith(known):          # a question may carry a short empathy phrase in front
                    return said[known]
            return UNKNOWN
        sim, spoken = run_closing(answer)
        closing = spoken[len(queue):]
        self.assertEqual(closing[:2], ['석 달 동안 살이 빠지셨군요.', '약 걱정을 함께 확인해 가요.'])
        self.assertEqual(closing[-1], run.INVITATION)

    def test_no_summary_when_the_patient_told_us_too_little(self):
        sim, spoken = run_closing(lambda q: UNKNOWN)
        self.assertEqual(sim.doctor.reply_calls, 0)

    def test_up_to_five_exams_are_done_and_the_closing_still_fits(self):
        ids = ['heart', 'lungs', 'abdomen', 'thyroid', 'eyes', 'tremor']
        sim, spoken = run_closing(lambda q: UNKNOWN, exam_ids=ids)
        self.assertEqual(sim.exams, [run.EXAM_CATALOG[k] for k in ids[:run.MAX_EXAMS]])
        self.assertTrue(any('진료' in m for m in spoken))                            # the warning line is still sent

    def test_exams_stop_early_enough_to_keep_the_five_closing_lines(self):
        ids = ['heart', 'lungs', 'abdomen', 'thyroid', 'eyes']
        sim, spoken = run_closing(lambda q: UNKNOWN, max_turns=9, exam_ids=ids)    # five of nine turns belong to the closing
        self.assertEqual(len(sim.exams), 4)
        self.assertEqual(len(spoken), 5)
        self.assertTrue(any('진료' in m for m in spoken))

    def test_what_counts_as_a_question_or_worry(self):
        for yes in ('그럼 약을 줄일 방법도 있을까요?', '혹시 큰 병은 아닌지 걱정돼요.', '무서워요', '괜찮을까요'):
            self.assertTrue(run.needs_answer(yes), yes)
        for no in ('네, 알겠습니다.', '네, 검사받고 원인을 알고 싶어요.', UNKNOWN):
            self.assertFalse(run.needs_answer(no), no)


class FlowTests(unittest.TestCase):
    def test_fixed_history_precedes_model_done_and_actual_explanations(self):
        sim, spoken, result = run_consult(20)
        required = [q for _, q in run.history_queue(sim.public_session()['initial']['opening'])]
        self.assertEqual(spoken[:len(required)], required)
        closing = spoken[len(required):]
        self.assertEqual(len(closing), 6)                       # five planned lines and the closing invitation
        self.assertEqual(closing[-1], run.INVITATION)
        self.assertTrue(all(not asks(text) for text in closing))
        self.assertIn('2주 뒤 결과와 경과를 확인해요.', closing)
        self.assertEqual(result['status'], 'completed')
        self.assertNotIn('HIDDEN', json.dumps(sim.doctor.messages))
        self.assertIn('[실제 설명/', sim.record['P']['education'])

    def test_small_budget_reserves_closing_and_final_record(self):
        sim, spoken, result = run_consult(8)
        self.assertLessEqual(sim.turns, 8)
        self.assertEqual(len(spoken), 6)
        self.assertEqual(spoken[-1], run.INVITATION)
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
