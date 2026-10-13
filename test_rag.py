"""Retrieval behaviour on a tiny synthetic corpus (no guideline data or model needed)."""
import tempfile
import unittest
from pathlib import Path

from rag import GuidelineIndex, chunk_sections, synonyms, tokenize

DOCS = {
    'graves': '---\ntitle: "Graves disease"\n---\n# Overview\nAutoimmune hyperthyroidism.\n## Diagnosis\n'
              'Weight loss, tremor, palpitation, goiter. Check TSH and free T4.\n## Treatment\nAntithyroid drugs.',
    'pain': '---\ntitle: "General pain guide"\n---\n# Pain\nPain pain pain assessment with NRS score and pain history.',
    'myasthenia': '---\ntitle: "Myasthenia gravis"\n---\n# Features\nDiplopia and ptosis that worsen with fatigue.',
    '갑상선': '---\ntitle: "갑상선기능항진증"\n---\n# 증상\n체중감소 손떨림 심계항진이 나타난다.',
}


class RagTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name) / 'guidelines'
        (root / 'source').mkdir(parents=True)
        for name, text in DOCS.items():
            (root / 'source' / f'{name}.md').write_text(text, encoding='utf-8')
        self.index = GuidelineIndex(root, Path(self.tmp.name) / 'cache')

    def tearDown(self):
        self.index.close()
        self.tmp.cleanup()

    def titles(self, hits):
        return [h['title'] for h in hits]

    def test_rare_words_beat_a_generic_word(self):
        hits = self.index.search('pain diplopia ptosis')
        self.assertEqual(hits[0]['title'], 'Myasthenia gravis')

    def test_everyday_korean_reaches_clinical_terms(self):
        hits = self.index.search('요즘 손이 떨려요 그리고 가슴이 두근거려요')
        self.assertEqual(hits[0]['title'], 'Graves disease')

    def test_extra_terms_widen_search_only(self):
        self.assertEqual(self.index.search('윗눈꺼풀 처짐')[:1], [])
        hits = self.index.search('윗눈꺼풀 처짐', extra=['ptosis', 'diplopia'])
        self.assertEqual(self.titles(hits)[:1], ['Myasthenia gravis'])

    def test_heading_is_kept_and_fusion_ranks_shared_hits_first(self):
        self.assertEqual(self.index.search('weight loss tremor')[0]['heading'], 'Overview')
        fused = self.index.search_many(['weight loss', 'tremor palpitation', 'diplopia'], limit=2)
        self.assertEqual(self.titles(fused)[0], 'Graves disease')

    def test_thought_finds_what_the_patients_words_cannot(self):
        query = '윗눈꺼풀 처짐'
        self.assertEqual(self.index.search(query), [])
        thought = 'Fatigable ptosis with diplopia that worsens with activity points to neuromuscular junction involvement.'
        hits = self.index.search(query, thought=thought)
        self.assertEqual(hits[0]['title'], 'Myasthenia gravis')
        self.assertEqual(self.index.search(query, thought='the of and is a'), [])  # generic words alone retrieve nothing

    def test_particles_and_synonyms(self):
        self.assertEqual(tokenize('체중이 줄었어요 목이'), ['체중', '줄었', '목이'])  # particles and endings are dropped from longer words only
        self.assertIn('goiter', synonyms(tokenize('목이 부어요')))
        self.assertIn('edema', synonyms(tokenize('목이 부어요')))
        self.assertIn('tremor', synonyms(['손떨림']))
        self.assertEqual(synonyms(['완전무관']), [])

    def test_chunks_follow_headings_and_long_sections_still_split(self):
        body = '# A\n' + 'x' * 100 + '\n# B\n' + 'y' * 100
        self.assertEqual([c[1] for c in chunk_sections(body)], ['A'])  # short sections merge into one chunk
        long_body = '# A\n' + 'z ' * 4000
        chunks = list(chunk_sections(long_body))
        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(len(c[2]) <= 2500 for c in chunks))
        self.assertEqual(chunks[1][0], 2200)

    def test_evaluator_compares_old_and_current_search(self):
        from eval_retrieval import evaluate
        cases = [{'id': 'a', 'query': '손이 떨려요 가슴이 두근거려요', 'expected': ['graves']},
                 {'id': 'b', 'query': '윗눈꺼풀 처짐', 'expected': ['myasthenia'], 'extra': ['ptosis']}]
        report = evaluate(self.index, cases, 3)
        self.assertEqual(report['current']['hit@3'], 1.0)
        self.assertLess(report['or-only (old)']['hit@3'], 1.0)
        self.assertEqual(report['or-only (old)']['missed'], ['a', 'b'])
        with_thought = evaluate(self.index, [{'id': 'c', 'query': '윗눈꺼풀 처짐', 'expected': ['myasthenia'],
                                              'thought': 'fatigable ptosis and diplopia'}], 3)
        self.assertEqual(with_thought['thought']['hit@3'], 1.0)
        self.assertEqual(with_thought['current']['hit@3'], 0.0)


if __name__ == '__main__':
    unittest.main()
