import os
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from clinic.agent import DemoDoctor
from clinic.data import ROOT, provenance


class ExperimentTests(unittest.TestCase):
    def test_pipeline_freezes_prompts_excludes_used_test_cases_and_runs_both_arms(self):
        final = DemoDoctor().act({"force_diagnose": True, "evidence": [], "round": "preliminary"}, 0)
        final['diagnosis'] = 'Mock diagnosis'
        payloads = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                payloads.append(payload)
                system = payload['messages'][0]['content']
                if system.startswith('Assess two diagnosis labels'):
                    content = {'relation': 'different', 'reason': 'Different mock labels'}
                else:
                    content = dict(final)
                    if 'Use working_memory' in system:
                        content['working_memory'] = {'hypotheses': [], 'next_information_needed': '', 'next_action_reason': 'Submit a mock result'}
                body = {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(content)}}],
                        'usage': {'prompt_tokens': 50, 'completion_tokens': 20}}
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(body).encode())
            def log_message(self, *args):
                pass
        server = HTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                output, previous, prompt = tmp/'experiment', tmp/'previous.json', tmp/'prompt.txt'
                split = json.loads((ROOT/'eval/split.json').read_text())
                used = split['test'][:-1]
                previous.write_text(json.dumps({'provenance': provenance(), 'test_case_indices': used}))
                prompt.write_text('Use observed evidence only.')
                command = [sys.executable, 'experiment.py', '--output', str(output), '--dev-limit', '1',
                           '--candidate-prompt', str(prompt), '--previous-evaluation', str(previous),
                           '--base-url', f'http://127.0.0.1:{server.server_port}/v1']
                result = subprocess.run(command, cwd=ROOT, env={**os.environ, "CLINIC_LIVE_DIR": str(Path(tempfile.gettempdir()) / "clinic-test-live")}, capture_output=True, text=True, timeout=40)
                self.assertEqual(result.returncode, 0, result.stderr)
                plan = json.loads((output/'evaluation-plan.json').read_text())
                self.assertEqual(plan['status'], 'completed')
                self.assertEqual(plan['test_case_indices'], split['test'][-1:])
                self.assertEqual(plan['dev_case_indices'], split['dev'][:1])
                self.assertFalse(plan['candidate_regenerated'])
                self.assertFalse(plan['test_used_for_prompt_search'])
                self.assertEqual((output/'frozen-prompts/candidate.txt').read_text(), prompt.read_text())
                for phase in ['dev', 'test']:
                    for arm in ['baseline', 'candidate']:
                        manifest = json.loads((output/phase/arm/'manifest.json').read_text())
                        self.assertEqual(manifest['working_memory_enabled'], arm=='candidate')
                        records = [json.loads(x) for x in (output/phase/arm/'results.jsonl').read_text().splitlines()]
                        self.assertEqual(len(records), 1)
                        self.assertEqual(len(records[0]['memory_updates']), int(arm=='candidate'))
                    self.assertTrue((output/phase/'comparison/comparison.json').exists())
                doctors = [p for p in payloads if p['messages'][0]['content'].startswith('You are a doctor')]
                self.assertEqual(len(doctors), 4)
                self.assertTrue(all('gold_diagnosis' not in p['messages'][1]['content'] for p in doctors))
                self.assertIn('dev_selected', plan)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
