"""Local monitor startup must be bounded and must never signal Windows processes."""
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import URLError

import live


class MonitorRuntimeTests(unittest.TestCase):
    def test_health_check_bypasses_proxies_and_has_short_timeout(self):
        identity = {'app': 'nova-live-clinic', 'root': str(live.ROOT), 'live': str(live.LIVE)}
        response = io.StringIO(json.dumps(identity))
        opener = Mock()
        opener.open.return_value.__enter__ = Mock(return_value=response)
        opener.open.return_value.__exit__ = Mock(return_value=False)
        with patch.object(live, 'build_opener', return_value=opener) as build:
            self.assertTrue(live.monitor_ready('http://127.0.0.1:8768/'))
        self.assertEqual(build.call_args.args[0].proxies, {})
        opener.open.assert_called_once_with('http://127.0.0.1:8768/api/health', timeout=0.25)

    def test_timeout_or_refused_connection_is_not_ready(self):
        for error in (TimeoutError(), URLError('refused'), ConnectionRefusedError()):
            opener = Mock(); opener.open.side_effect = error
            with self.subTest(error=type(error).__name__), patch.object(live, 'build_opener', return_value=opener):
                self.assertFalse(live.monitor_ready('http://127.0.0.1:8768/'))

    def test_unrelated_service_is_not_reused(self):
        opener = Mock()
        opener.open.return_value.__enter__ = Mock(return_value=io.StringIO('{"app":"other"}'))
        opener.open.return_value.__exit__ = Mock(return_value=False)
        with patch.object(live, 'build_opener', return_value=opener), self.assertRaises(OSError):
            live.monitor_ready('http://127.0.0.1:8768/')

    def test_failed_startup_has_deadline_and_does_not_launch_browser(self):
        with tempfile.TemporaryDirectory() as directory:
            child = Mock(); child.poll.return_value = None
            with patch.object(live, 'LIVE', Path(directory)), \
                 patch.object(live, 'monitor_ready', return_value=False), \
                 patch.object(live.subprocess, 'Popen', return_value=child), \
                 patch.object(live.time, 'monotonic', side_effect=[0, 0, 3.1]), \
                 patch.object(live.time, 'sleep'), patch.object(live.webbrowser, 'open') as browser:
                with self.assertRaisesRegex(OSError, 'did not start'):
                    live.open_monitor(8768)
                browser.assert_not_called()

    def test_current_process_liveness_never_sends_signal(self):
        with patch.object(live.os, 'kill', side_effect=AssertionError('must not signal self')):
            self.assertTrue(live.process_alive(os.getpid()))
            self.assertFalse(live.process_alive(-1))
            self.assertFalse(live.process_alive(None))

    @unittest.skipUnless(os.name == 'nt', 'Windows process query')
    def test_windows_child_liveness_does_not_kill_child(self):
        child = subprocess.Popen([os.sys.executable, '-c', 'import time; time.sleep(10)'],
                                 creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            with patch.object(live.os, 'kill', side_effect=AssertionError('must not send a Windows signal')):
                self.assertTrue(live.process_alive(child.pid))
                self.assertIsNone(child.poll())
        finally:
            child.terminate()
            child.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
