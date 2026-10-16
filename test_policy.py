import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
import run
from nova_sdk import NovaError

class Scripted:
    def __init__(self,results): self.results=iter(results);self.calls=[]
    def chat(self,messages,**kwargs):
        self.calls.append((messages,kwargs))
        x=next(self.results)
        if isinstance(x,Exception):raise x
        return {'choices':[{'message':{'content':json.dumps(x,ensure_ascii=False)}}]}

class PolicyTests(unittest.TestCase):
    def test_topic_reuse_rejected(self):
        c=Scripted([{'action':'SAY','topic':'onset','content':'언제 시작됐나요?'},
                    {'action':'SAY','topic':'course','content':'점점 심해지나요?'}])
        session={'initial':{'opening':'오래 아파요.','vital_signs':'미제공'}}
        self.assertEqual(run.choose_question(c,session,[],{'onset'}),('course','점점 심해지나요?'))
        self.assertEqual(len(c.calls),2)

    def test_completed_invalid_gets_new_request_id(self):
        c=Scripted([NovaError('PROVIDER_RESPONSE_INVALID',503),{'ok':True}])
        run.brief_call(c,'s','u',effort='medium')
        self.assertNotEqual(c.calls[0][1]['request_id'],c.calls[1][1]['request_id'])
        self.assertEqual(c.calls[1][1]['reasoning_effort'],'low')

    def test_transport_preserves_request(self):
        c=Scripted([NovaError('PROVIDER_UNAVAILABLE',503),{'ok':True}])
        run.brief_call(c,'s','u',effort='medium')
        self.assertEqual(c.calls[0],c.calls[1])

    def test_explicit_compression_not_silent_drop(self):
        too_long='검사로 원인을 정확하게 확인한 다음에 치료 방향을 함께 결정할 수 있어요.'
        c=Scripted([{'text':too_long},{'text':'피검사로 원인을 확인해요.'}])
        note={'P':dict.fromkeys(('tests','treatment','follow_up','education'),'')}
        note['P']['tests']='Blood tests to assess cause'
        self.assertEqual(run.patient_explanations(c,note,''),[('next','피검사로 원인을 확인해요.')])
        self.assertEqual(len(c.calls),2)

    def test_treatment_and_followup_cannot_be_reinvented(self):
        c=Scripted([{'text':'약을 바로 시작해요.'},{'text':'약을 바로 시작해요.'},{'text':''}])
        note={'P':{'tests':'','education':'','treatment':'Review first','follow_up':'Reassess in 2 weeks.'}}
        output=run.patient_explanations(c,note,'')
        # An unconditional treatment line is refused twice, so the safe fixed line is used; follow-up is rendered by code.
        self.assertEqual(output,[('treatment',run.TREATMENT_FALLBACK),('follow_up','2주 뒤 결과와 경과를 확인해요.')])
        self.assertEqual(len(c.calls),3)
        self.assertIn('위험 증상',c.calls[2][0][0]['content'])
        self.assertEqual(run.followup_sentence('Return in 6–12 months.'),'6~12개월 뒤 결과와 경과를 확인해요.')
        self.assertIsNone(run.followup_sentence('Return when needed.'))

    def test_content_duplicate_cannot_hide_behind_new_topic(self):
        self.assertTrue(run.repeated_question('가족 중 키가 작은 사람은 있나요?', ['턴 1 의사: 가족 중 키가 작은 사람이 있나요?']))
        self.assertTrue(run.repeated_question('담배 피우시나요?', ['턴 1 의사: 담배 피우시나요?']))
        self.assertFalse(run.repeated_question('약 한 알에 몇 mg이라고 적혀 있나요?', ['턴 1 의사: 그 약을 하루 언제 몇 번 드시나요?']))
        self.assertFalse(run.repeated_question('약을 줄이려다 실패한 적이 있나요?', ['턴 1 의사: 그 약을 한 번에 실제로 얼마나 드시나요?']))

    def test_observations_stay_verbatim(self):
        from grounding import sources
        raw='새벽 두세 시에 복용했고 몸무게는 십육 점 오 키로그람이다.'
        items=sources({'initial':{'opening':raw,'vital_signs':'미제공'}},[])
        self.assertEqual(items[0]['value'],raw)
        for value in run.EXAM_CATALOG.values():self.assertLessEqual(len(value),30)

    def test_unsafe_dose_repair_does_not_change_diagnosis(self):
        c=Scripted([{'treatment':'현재 복용약을 확인한 후 의료진과 치료 방향을 정합니다.'}])
        note={'P':{'treatment':'Start drug 10 mg daily'},'S':{'hpi':'관찰'}}
        trace={};session={'initial':{'opening':'약을 복용해요.','vital_signs':'미제공'}}
        run.check_treatment(c,note,'unchanged',session,[],trace)
        self.assertNotRegex(note['P']['treatment'],run.DOSE_PATTERN)
        self.assertEqual(note['S'],{'hpi':'관찰'})
        self.assertTrue(trace['treatment_review']['fallback'])
        self.assertEqual(c.calls,[])

    def test_basic_history_is_not_dependent_on_model_retry(self):
        topics=dict(run.history_queue('여섯 살 남자아이이고 키가 작아요.'))
        self.assertTrue({'past','medications','allergy','family','smoking'}<=set(topics))
        self.assertNotIn('alcohol',topics)
        self.assertNotIn('med_name',topics)
        self.assertTrue({'birth','development','daily_life'}<=set(topics))
        self.assertNotIn('pregnancy',topics)
        reproductive=dict(run.history_queue('서른두 살 여자이고 체중이 줄었어요.'))
        self.assertTrue({'menstrual','pregnancy'}<=set(reproductive))
        adult=dict(run.history_queue('예순두 살 여자이고 약을 처방받고 싶어요.'))
        self.assertIn('med_name',adult)
        self.assertIn('alcohol',adult)
        for question in [*topics.values(),*adult.values()]:self.assertLessEqual(len(question),30)

    def test_review_questions_drop_long_repeated_and_duplicate_lines(self):
        c=Scripted([{'questions':['식욕은 어떤가요?','식욕은 어떤가요?','잠은 잘 주무시나요?',
            '가'*31+'?','언제부터 아팠나요?','더위 때문에 힘들어요.','목 앞이 붓거나 이물감이 있나요?',3]}])
        session={'initial':{'opening':'살이 빠져요.','vital_signs':'미제공'}}
        got=run.review_questions(c,session,['턴 1 의사: 언제부터 아팠나요?','환자: 석 달 전부터요.'])
        self.assertEqual(got,[('review_1','식욕은 어떤가요?'),('review_2','잠은 잘 주무시나요?'),
                              ('review_3','목 앞이 붓거나 이물감이 있나요?')])

    def test_a_plan_with_real_line_breaks_inside_strings_is_still_read(self):
        # A real line break (not the two characters backslash-n) inside a JSON string, as the model writes it.
        text='{"P":{"tests":"- TSH : confirm'+chr(10)+'- ECG : tachycardia","treatment":"x","disposition":"y","follow_up":"z","education":"w"}}'
        self.assertEqual(run.parse_plan(text)['P']['tests'],'- TSH : confirm'+chr(10)+'- ECG : tachycardia')
        self.assertEqual(run.parse_json(text)['P']['follow_up'],'z')

    def test_adults_get_one_merged_past_and_family_question_and_work_history_but_children_do_not(self):
        adult=dict(run.history_queue('서른두 살 여자이고 체중이 줄었어요.'))
        self.assertIn('고혈압',adult['past']); self.assertIn('고혈압',adult['family'])
        self.assertIn('occupation',adult)
        self.assertFalse({'chronic','family_chronic'}&set(adult))              # no near-duplicate second question
        child=dict(run.history_queue('여섯 살 남자아이이고 키가 작아요.'))
        self.assertNotIn('고혈압',child['past'])
        self.assertFalse({'occupation','alcohol'}&set(child))
        self.assertEqual(len(run.family_followups('어머니가 같은 병으로 약을 드세요.')),1)
        from grounding import source_record
        items=[{'id':'opening','kind':'opening','question':'','value':'x','raw':'x'},
               {'id':'turn1','kind':'say','question':adult['past'],'value':'없어요','raw':'없어요'},
               {'id':'turn2','kind':'say','question':adult['family'],'value':'없어요','raw':'없어요'}]
        note=source_record(items)
        self.assertIn('turn1',note['S']['pmh']); self.assertIn('turn2',note['S']['fh'])   # still filed in the right sections

    def test_a_reply_sees_the_whole_plan_field_and_what_was_already_said(self):
        c=Scripted([{'text':'대부분 치료로 좋아질 수 있어요.'}])
        late='Return immediately for chest pain or fainting.'
        note={'P':{'tests':'[계획/제안, 시행 결과 아님] - TSH : confirm','treatment':'','education':'',
                   'follow_up':'[계획/제안, 시행 결과 아님] Review in 1 week. '+'x'*600+late}}
        text=run.reply_message(c,'concern','큰 병일까 걱정돼요.',note,'Graves disease',{'갑상샘이 활발한 병일 가능성이 있어요.'})
        self.assertEqual(text,'대부분 치료로 좋아질 수 있어요.')
        request=json.loads(c.calls[0][0][1]['content'])
        self.assertIn(late,request['plan']['follow_up'])                       # past the old 400-character cut
        self.assertFalse(request['plan']['tests'].startswith('[계획'))         # the record prefix is not sent
        self.assertEqual(request['lines_already_said'],['갑상샘이 활발한 병일 가능성이 있어요.'])

    def test_planned_questions_are_cut_to_the_limit(self):
        c=Scripted([{'questions':['기침이 있나요?','열이 나나요?','가래가 나오나요?','숨이 차나요?','가슴이 아픈가요?']}])
        session={'initial':{'opening':'기침해요.','vital_signs':'미제공'}}
        self.assertEqual(len(run.review_questions(c,session,[],limit=3)),3)

    def test_unknown_family_answers_get_no_follow_up(self):
        for unknown in ('가족 중 비슷한 증상이 있는지는 확인되지 않았습니다.','가족력은 잘 모르겠어요.','가족 얘기는 확인 안 됐어요.'):
            self.assertEqual(run.family_followups(unknown),[],unknown)
        self.assertTrue(run.family_followups('어머니가 같은 병으로 약을 드세요.'))

    def test_the_cause_line_never_sends_the_patient_to_a_doctor(self):
        c=Scripted([{'text':'신장 결석일 수 있어 진료를 받아봐요.'},{'text':'신장에 돌이 생겼을 가능성이 있어요.'}])
        note={'P':dict.fromkeys(('tests','treatment','follow_up','education'),''),'S':{'hpi':'','fh':'','pmh':''}}
        self.assertEqual(run.patient_explanations(c,note,'Nephrolithiasis'),[('cause','신장에 돌이 생겼을 가능성이 있어요.')])

    def test_every_new_exam_is_one_action_within_the_limit(self):
        for key in ('pelvic','rectal','breast','hernia','testes','skin_lesion','foot_pulse','calf','spo2'):
            self.assertIn(key,run.EXAM_CATALOG)
            self.assertLessEqual(len(run.EXAM_CATALOG[key]),run.MAX_CHARS)

    def test_positive_relative_not_erased_by_other_relative_negative(self):
        self.assertTrue(run.family_followups('아빠가 어릴 때 키가 작았다. 다른 가족에는 없다.'))
        self.assertFalse(run.family_followups('가족 중 비슷한 증상은 없다.'))
        self.assertFalse(run.family_followups('미확인: 원자료에 해당 정보가 없습니다.'))
        self.assertTrue(run.repeated_question('지난 6개월 동안 키가 얼마나 자랐나요?',['턴 1 의사: 최근 6개월간 키가 얼마나 자랐나요?']))

    def test_numeric_hints_preserve_decimal_subject_and_ambiguous_time(self):
        raw='아빠 키는 백칠십팔 센티미터이고 엄마 키는 백육십오 센티미터이다. 몸무게는 십육 점 오 키로그람이다. 새벽 두세 시다.'
        item={'id':'x','raw':raw,'value':raw}
        out=run.numeric_context([item])[0]
        self.assertEqual(out['raw'],raw)
        self.assertEqual(out['value'],raw)
        self.assertNotIn('numeric_hints',item)
        self.assertEqual([(x['numeric'],x['unit']) for x in out['numeric_hints']],[('178','cm'),('165','cm'),('16.5','kg')])
        for h in out['numeric_hints']: self.assertEqual(raw[h['start']:h['end']],h['raw_span'])
        self.assertNotIn('numeric_hints',run.numeric_context([{'raw':'두세 시에 먹고 한두 킬로그램 변했어요.'}])[0])

    def test_plan_outer_brace_repair_preserves_only_complete_values(self):
        complete={'P':{'tests':'Check cause','treatment':'Observe','disposition':'Review','follow_up':'2 weeks','education':'Return if worse'}}
        encoded=json.dumps(complete)
        self.assertEqual(run.parse_plan(encoded),complete)
        self.assertEqual(run.parse_plan(encoded[:-1]),complete)
        self.assertEqual(run.parse_plan(json.dumps(complete['P'])),complete)
        cut=dict(complete['P']); del cut['education']       # a reply cut off inside the last value keeps the finished ones
        self.assertEqual(run.parse_plan(encoded[:-8]),{'P':cut})
        self.assertIsNone(run.parse_plan('{"P":{"tests":"unclosed'))

    def test_medication_questions_are_specific_and_within_action_limit(self):
        topics=[topic for topic,_ in run.MEDICATION_REVIEW]
        self.assertEqual(len(topics),len(set(topics)))
        self.assertTrue({'med_name','med_amount','med_pattern','med_strength','med_missed'} <= set(topics))
        for _,question in run.MEDICATION_REVIEW:
            self.assertLessEqual(len(question),30)
            self.assertTrue(run.asks(question))

if __name__=='__main__':unittest.main(verbosity=2)
