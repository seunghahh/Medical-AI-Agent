"""Consultation flow: required history before DONE, and the closing explanation to the patient."""
import json
import unittest
from unittest.mock import patch

import run
from grounding import asks, sources
from nova_local import LocalClient
from test_contract import Endpoint, simulator

NOTE = {'diagnosis': '진단 미정'}
EXPLANATION = {'messages': ['걱정 많으셨겠어요.', '검사로 원인을 확인하겠습니다.', '심해지면 바로 오셔야 해요.']}


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
        replies = [{'action': 'DONE'}, *map(say, questions), {'action': 'DONE'}, {'action': 'DONE'}, EXPLANATION, NOTE]
        sim, spoken = run_consult(replies, max_turns=12)
        self.assertEqual(spoken, questions + EXPLANATION['messages'])
        self.assertEqual(sim.turns, 7)
        self.assertIn('확인할 항목', json.dumps(sim.doctor.messages, ensure_ascii=False))

    def test_gathering_stops_early_enough_to_leave_turns_for_the_explanation(self):
        replies = [*[say(f'{n}번째 증상이 어떤가요?') for n in range(6)], EXPLANATION, NOTE]
        sim, spoken = run_consult(replies, max_turns=8)
        self.assertEqual(sim.turns, 8)
        self.assertEqual(spoken[-2:], [EXPLANATION['messages'][0], run.FALLBACK_EXPLANATION[2]])  # no room: cause + safety line

    def test_explanation_never_asks_a_question(self):
        doctor = Endpoint([{'messages': ['증상이 더 있나요?', '또 궁금한 게 있으세요?']}, {'messages': ['검사로 확인하겠습니다.', '질문 있으세요?']}])
        sim = simulator(); sim.doctor = doctor
        self.assertEqual(run.explanation_messages(LocalClient(sim), sim.public_session(), [], 3),
                         ['검사로 확인하겠습니다.', run.FALLBACK_EXPLANATION[2]])  # the only valid line, plus the fixed safety line
        junk = Endpoint([{'messages': ['x' * 31]}, {'nothing': 1}])
        sim.doctor = junk
        fallback = run.explanation_messages(LocalClient(sim), sim.public_session(), [], 3)
        self.assertEqual(fallback, list(run.FALLBACK_EXPLANATION))
        self.assertTrue(all(0 < len(m) <= run.MAX_CHARS and not asks(m) for m in run.FALLBACK_EXPLANATION))

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

    def test_the_explanation_always_includes_warning_signs(self):
        def messages(*texts):
            sim = simulator(); sim.doctor = Endpoint([{'messages': list(texts)}])
            return run.explanation_messages(LocalClient(sim), sim.public_session(), [], 3)
        none_mention = messages('가능성이 있어요.', '검사 후 치료를 정해요.', '질문 있으세요?')   # third is a question: dropped
        self.assertEqual(none_mention, ['가능성이 있어요.', '검사 후 치료를 정해요.', run.FALLBACK_EXPLANATION[2]])
        full = messages('가능성이 있어요.', '검사 후 치료를 정해요.', '카페인은 줄입니다.')
        self.assertEqual(full[-1], run.FALLBACK_EXPLANATION[2]); self.assertEqual(len(full), 3)
        covered = messages('가능성이 있어요.', '심해지면 응급실에 가야 해요.')
        self.assertEqual(covered, ['가능성이 있어요.', '심해지면 응급실에 가야 해요.'])

    def test_patient_replies_to_the_explanation_are_not_observations(self):
        session = {'initial': {'opening': '배가 아파요', 'vital_signs': '미제공'}}
        transcript = []
        for number, message in enumerate(EXPLANATION['messages'], 1):
            transcript += [f'턴 {number} 의사: {message}', '환자: 네, 알겠습니다']
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


if __name__ == '__main__':
    unittest.main()
