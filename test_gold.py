"""The gold-coverage dev tool: log building, reference parsing, summaries and a full report with a fake judge."""
import json
import tempfile
import unittest
from pathlib import Path

import gold_coverage as gc
from test_contract import Endpoint

ITEMS = [{'item': '식욕 변화', 'kind': 'history', 'covered': True, 'turn': 1},
         {'item': '월경 변화', 'kind': 'history', 'covered': False, 'turn': None},
         {'item': '안구 돌출', 'kind': 'exam', 'covered': True, 'turn': 3},
         {'item': '갑상선 bruit', 'kind': 'exam', 'covered': False, 'turn': None}]
TRANSCRIPT = {'summary': {'case_id': 'C10'}, 'events': [
    {'turn': 1, 'action': 'SAY', 'content': '식욕은 변했나요?', 'response': {'content': '늘었어요'}},
    {'turn': 2, 'action': 'EXAM', 'content': '목을 만져 보겠습니다', 'response': {}},
    {'turn': 3, 'action': 'EXAM', 'content': '눈을 살펴보겠습니다', 'response': {'result': '돌출(+/+)'}},
    {'turn': 3, 'action': 'DIAGNOSE', 'diagnosis': '갑상선기능항진증'}]}


class GoldCoverageTests(unittest.TestCase):
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

    def test_summary_counts_and_lists_the_missed_findings(self):
        summary = gc.summarize(ITEMS)
        self.assertEqual((summary['history']['covered'], summary['history']['total']), (1, 2))
        self.assertEqual(summary['exam']['missed'], ['갑상선 bruit'])
        self.assertEqual(summary['all']['share'], 0.5)
        self.assertIsNone(gc.summarize([])['all']['share'])

    def test_a_malformed_or_chatty_answer_is_still_parsed(self):
        self.assertEqual(len(gc.parse_items('설명입니다 ' + json.dumps({'items': ITEMS}, ensure_ascii=False) + ' 끝')), 4)
        self.assertEqual(gc.parse_items('not json'), [])
        self.assertEqual(gc.parse_items(json.dumps({'items': ['x', {'item': ' '}, {'item': '유효'}]})), [{'item': '유효'}])

    def test_report_writes_the_result_next_to_the_consultation(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / 'transcript.json').write_text(json.dumps(TRANSCRIPT, ensure_ascii=False), encoding='utf-8')
            case_id, summary = gc.report(folder, Endpoint([{'items': ITEMS}]))
            self.assertEqual((case_id, summary['all']['covered']), ('C10', 2))
            saved = json.loads((folder / 'gold_coverage.json').read_text(encoding='utf-8'))
            self.assertEqual(len(saved['items']), 4)

    def test_the_judge_retries_once_when_the_first_answer_has_no_items(self):
        endpoint = Endpoint([{'oops': 1}, {'items': ITEMS}])
        self.assertEqual(len(gc.judge(endpoint, 'gold', 'log')), 4)


if __name__ == '__main__':
    unittest.main()
