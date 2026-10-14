"""The gold-coverage dev tool: log building, reference parsing, fixed finding lists, judging, and a full report.
Everything here uses temporary folders: the tool writes next to the consultation it reports on."""
import json
import tempfile
import unittest
from pathlib import Path

import gold_coverage as gc
from test_contract import Endpoint

RAW = [{'item': '식욕 변화', 'kind': 'history'}, {'item': '월경 변화', 'kind': 'history'},
       {'item': '안구 돌출', 'kind': 'exam'}, {'item': '갑상선 bruit', 'kind': 'exam'}, {'item': '', 'kind': 'exam'}, 'junk']
TRANSCRIPT = {'summary': {'case_id': 'C10'}, 'start_information': {'opening': '32세 여자, 체중 감소'}, 'events': [
    {'turn': 1, 'action': 'SAY', 'content': '식욕은 변했나요?', 'response': {'content': '늘었어요'}},
    {'turn': 2, 'action': 'EXAM', 'content': '목을 만져 보겠습니다', 'response': {}},
    {'turn': 3, 'action': 'EXAM', 'content': '눈을 살펴보겠습니다', 'response': {'result': '돌출(+/+)'}},
    {'turn': 3, 'action': 'DIAGNOSE', 'diagnosis': '갑상선기능항진증'}]}
VERDICT = {'results': [{'n': 1, 'covered': True, 'turn': 1}, {'n': 3, 'covered': True, 'turn': 3}, 'junk']}


class GoldCoverageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self.tmp.name) / 'cache'
        self.folder = Path(self.tmp.name) / 'C10-run'
        self.folder.mkdir()
        (self.folder / 'transcript.json').write_text(json.dumps(TRANSCRIPT, ensure_ascii=False), encoding='utf-8')

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_log_has_questions_and_exams_but_not_the_final_record(self):
        log = gc.build_log(TRANSCRIPT)
        self.assertIn('T1 질문: 식욕은 변했나요? -> 답: 늘었어요', log)
        self.assertIn('T2 진찰: 목을 만져 보겠습니다 -> 결과: (응답 없음)', log)
        self.assertNotIn('갑상선기능항진증', log)

    def test_only_history_and_exam_sections_of_the_reference_are_kept(self):
        text = gc.gold_sections((Path(__file__).parent / 'samples' / 'C10' / 'model_record.md').read_text(encoding='utf-8'))
        self.assertIn('## [S] 현병력', text)
        self.assertIn('## [O] 진찰 소견', text)
        for dropped in ('## [O] 검사 결과', '## [A]', '## [P]', '## 주진단'):
            self.assertNotIn(dropped, text)

    def test_a_chatty_or_malformed_answer_is_still_parsed(self):
        self.assertEqual(len(gc.first_json('설명 ' + json.dumps({'items': RAW[:2]}, ensure_ascii=False) + ' 끝', 'items')), 2)
        self.assertEqual(gc.first_json('not json', 'items'), [])
        self.assertEqual(gc.first_json(json.dumps({'other': [1]}), 'items'), [])

    def test_the_finding_list_is_extracted_once_and_then_reused(self):
        endpoint = Endpoint([{'items': RAW}])                  # a second call would raise StopIteration
        items = gc.gold_items(endpoint, 'gold', '32세 여자', 'C10', cache=self.cache)
        self.assertEqual([i['item'] for i in items], ['식욕 변화', '월경 변화', '안구 돌출', '갑상선 bruit'])
        self.assertEqual(gc.gold_items(endpoint, 'gold', '32세 여자', 'C10', cache=self.cache), items)
        # a changed reference text, or --refresh, extracts again
        again = gc.gold_items(Endpoint([{'items': RAW[:1]}]), 'gold v2', '32세 여자', 'C10', cache=self.cache)
        self.assertEqual(len(again), 1)
        refreshed = gc.gold_items(Endpoint([{'items': RAW[:2]}]), 'gold v2', '32세 여자', 'C10', refresh=True, cache=self.cache)
        self.assertEqual(len(refreshed), 2)

    def test_judging_marks_unmentioned_findings_as_missed(self):
        findings = [{'item': i['item'], 'kind': i['kind']} for i in RAW[:4]]
        judged = gc.judge(Endpoint([VERDICT]), findings, 'log')
        self.assertEqual([i['covered'] for i in judged], [True, False, True, False])
        self.assertEqual(judged[2]['turn'], 3)

    def test_summary_counts_and_lists_the_missed_findings(self):
        findings = [{'item': i['item'], 'kind': i['kind']} for i in RAW[:4]]
        summary = gc.summarize(gc.judge(Endpoint([VERDICT]), findings, 'log'))
        self.assertEqual((summary['history']['covered'], summary['history']['total']), (1, 2))
        self.assertEqual(summary['exam']['missed'], ['갑상선 bruit'])
        self.assertEqual(summary['all']['share'], 0.5)
        self.assertIsNone(gc.summarize([])['all']['share'])

    def test_the_model_is_asked_for_temperature_zero_and_retried_once(self):
        calls = []

        class Spy(Endpoint):
            def complete(self, messages, **kwargs):
                calls.append(kwargs)
                return super().complete(messages, **kwargs)

        self.assertEqual(len(gc.ask(Spy([{'oops': 1}, {'items': RAW[:2]}]), 'sys', 'user', 'items')), 2)
        self.assertTrue(all(c.get('temperature') == 0 for c in calls))
        self.assertEqual(len(calls), 2)

    def test_report_writes_the_result_next_to_the_consultation_and_keeps_the_cache_out_of_the_repo(self):
        case_id, summary = gc.report(self.folder, Endpoint([{'items': RAW}, VERDICT]), cache=self.cache)
        self.assertEqual((case_id, summary['all']['covered'], summary['all']['total']), ('C10', 2, 4))
        saved = json.loads((self.folder / 'gold_coverage.json').read_text(encoding='utf-8'))
        self.assertEqual(len(saved['items']), 4)
        self.assertTrue((self.cache / 'gold_items_C10.json').exists())
        # a second run reuses the finding list: only the judging call is needed
        case_id, summary = gc.report(self.folder, Endpoint([VERDICT]), cache=self.cache)
        self.assertEqual(summary['all']['total'], 4)

    def test_a_reference_without_findings_is_an_error_not_a_zero(self):
        with self.assertRaises(RuntimeError):
            gc.report(self.folder, Endpoint([{'nothing': 1}, {'nothing': 1}]), cache=self.cache)


if __name__ == '__main__':
    unittest.main()
