import json
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen
from http.server import ThreadingHTTPServer
from unittest.mock import patch
from clinic.live import Journal, snapshot, Handler, open_monitor, monitor_ready
from clinic.data import load_cases
from clinic.agent import DemoDoctor, DemoSimulator, Doctor
from clinic.runner import run_case
from test_clinic import FakeClient


class LiveTests(unittest.TestCase):
    def test_journal_incremental_restart_partial_write_and_interrupt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertIsNone(snapshot(directory=root)['run_id'])
            output = root / 'run1'; output.mkdir()
            with self.assertRaises(KeyboardInterrupt):
                with Journal(output, {'mode': 'model'}, root) as log:
                    log.emit('thinking', turn=0)
                    first = snapshot(directory=root)
                    self.assertEqual([e['seq'] for e in first['events']], [1, 2])
                    self.assertEqual(snapshot(log.run_id, 2, root)['events'], [])
                    raise KeyboardInterrupt()
            self.assertEqual(snapshot(log.run_id, 2, root)['events'][0]['status'], 'interrupted')
            with (output / 'events.jsonl').open('a') as file:
                file.write('{"unfinished":')
            self.assertEqual(len(snapshot(directory=root)['events']), 3)
            second = root / 'run2'; second.mkdir()
            with Journal(second, {}, root):
                self.assertEqual(snapshot(log.run_id, 99, root)['events'][0]['seq'], 1)

    def test_live_observations_match_result_and_no_hidden_data(self):
        case = replace(load_cases()[0], gold='SECRET_GOLD')
        for engine in ('plain', 'langgraph'):
            events = []
            def emit(kind, **data):
                events.append(json.loads(json.dumps(dict(type=kind, **data))))
            result = run_case(case, DemoDoctor(), DemoSimulator(), demo=True,
                              working_memory=True, engine=engine, observer=emit)
            self.assertNotIn('SECRET_GOLD', json.dumps(events))
            acquired = [e for event in events if event['type'] in ('case_start', 'observation') for e in event['evidence']]
            self.assertEqual(acquired, result['evidence'])
            self.assertEqual(events[-1]['turn'], result['turns'])
            self.assertEqual(sum(e['type']=='thinking' for e in events), len(result['history']))

    def test_rejection_does_not_emit_fake_response_or_charge_turn(self):
        question = {'action':'SAY','text':'증상은 언제 시작됐나요?','intent':'question'}
        final = DemoDoctor().act({'force_diagnose':True,'evidence':[],'round':'preliminary'}, 0)
        events = []
        result = run_case(load_cases()[0], Doctor(FakeClient([question, question, final])), DemoSimulator(),
            observer=lambda kind, **data: events.append(dict(type=kind, **data)))
        self.assertEqual(result['rejections'], 1)
        self.assertEqual(sum(e['type']=='observation' for e in events), 1)
        self.assertEqual(next(e['turn'] for e in events if e['type']=='rejected'), 1)

    def test_server_exposes_only_web_assets_and_validated_cursor(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        base = f'http://127.0.0.1:{server.server_port}'
        try:
            with urlopen(base+'/motion.mjs') as response:
                self.assertIn('javascript', response.headers['Content-Type'])
                self.assertIn(b'export class ClinicMotion', response.read())
            with patch('clinic.live.snapshot', return_value={'run_id':None,'events':[]}):
                self.assertEqual(json.load(urlopen(base+'/api/events'))['events'], [])
            for path, code in [('/../run.py',404),('/outputs/.live/latest.json',404),('/api/events?after=no',400)]:
                with self.assertRaises(HTTPError) as caught:
                    urlopen(base+path)
                self.assertEqual(caught.exception.code, code)
        finally:
            server.shutdown(); server.server_close(); thread.join()

    def test_browser_launch_reuses_server_and_requests_new_window(self):
        with patch('clinic.live.monitor_ready', return_value=True), patch('clinic.live.subprocess.Popen') as spawn, patch('clinic.live.webbrowser.open', return_value=True) as browser:
            self.assertEqual(open_monitor(8767), 'http://127.0.0.1:8767/')
            spawn.assert_not_called()
            browser.assert_called_once_with('http://127.0.0.1:8767/', new=1, autoraise=True)

    def test_browser_launch_starts_detached_server_and_waits_for_health(self):
        with tempfile.TemporaryDirectory() as temp, patch('clinic.live.LIVE', Path(temp)), patch('clinic.live.monitor_ready', side_effect=[False, True]), patch('clinic.live.subprocess.Popen') as spawn, patch('clinic.live.webbrowser.open', return_value=True):
            open_monitor(8768)
            self.assertTrue(spawn.call_args.kwargs['start_new_session'])
            self.assertEqual(spawn.call_args.args[0][-3:], ['serve', '--port', '8768'])

    def test_other_project_server_is_not_reused(self):
        from io import BytesIO
        with patch('clinic.live.urlopen', return_value=BytesIO(b'{"app":"other"}')):
            with self.assertRaisesRegex(OSError, 'another project'):
                monitor_ready('http://127.0.0.1:8767/')

    def test_missing_browser_does_not_fail_the_agent(self):
        import webbrowser
        with patch('clinic.live.monitor_ready', return_value=True), patch('clinic.live.webbrowser.open', side_effect=webbrowser.Error('unavailable')), patch('sys.stderr'):
            self.assertEqual(open_monitor(8767), 'http://127.0.0.1:8767/')
