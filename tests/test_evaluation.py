import copy
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from clinic.agent import DemoDoctor, DemoSimulator
from clinic.data import DATA, ROOT, load_cases, provenance
from clinic.evaluation import compare_runs, load_split, make_split, summarize, write_report
from clinic.runner import run_case


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = load_cases()[0]

    def fixture(self, model=False):
        result = run_case(self.case, DemoDoctor(), DemoSimulator(), demo=True)
        if model:
            result["mode"], result["exact_match"] = "model", True
        manifest = {"provenance": provenance(), "case_indices": [0], "mode": result["mode"],
            "round": "preliminary", "doctor_model": "mock" if model else None,
            "simulator_model": "mock" if model else None, "max_turns": 50, "max_seconds": 1200,
            "reasoning_effort": "low", "max_tokens": 4096, "engine": "plain"}
        return result, manifest

    def save(self, output, result, manifest):
        output.mkdir()
        (output / "manifest.json").write_text(json.dumps(manifest))
        (output / "results.jsonl").write_text(json.dumps(result) + "\n")

    def test_split_reproducible_disjoint_and_bound_to_dataset(self):
        split = make_split(DATA)
        self.assertEqual(split, make_split(DATA))
        self.assertEqual((len(split["dev"]), len(split["test"])), (20, 30))
        self.assertFalse(set(split["dev"]) & set(split["test"]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.json"
            path.write_text(json.dumps(split))
            self.assertEqual(load_split(path, DATA), split)
            split["test"][0] = split["dev"][0]
            path.write_text(json.dumps(split))
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_split(path, DATA)
            split = make_split(DATA)
            split["provenance"]["sha256"] = "tampered"
            path.write_text(json.dumps(split))
            with self.assertRaisesRegex(ValueError, "provenance"):
                load_split(path, DATA)
        with self.assertRaises(ValueError):
            make_split(DATA, dev_count=200, test_count=30)

    def test_partial_run_keeps_requested_denominator_and_missing_usage_null(self):
        result, manifest = self.fixture(model=True)
        manifest["case_indices"] = [0, 1]
        result["model_calls"] = [{"role": "doctor", "input_tokens": 0, "output_tokens": 0, "usage_reported": False}]
        rows, summary = summarize([result], manifest)
        self.assertEqual(summary["attempted_coverage"], .5)
        self.assertEqual(summary["completion_rate_all_requested"], .5)
        self.assertEqual(summary["exact_match_accuracy_all_requested"], .5)
        self.assertEqual(summary["exact_match_accuracy_all_attempted"], 1)
        self.assertIsNone(summary["total_doctor_input_tokens"])
        self.assertIsNone(rows[0]["simulator_output_tokens"])

    def test_rejected_actions_repeats_unknown_and_usage(self):
        result, manifest = self.fixture(model=True)
        action = {"action": "TEST", "text": "CBC"}
        result["history"].append({"action": action, "accepted": False, "feedback": "unavailable", "turn": 5})
        result["history"].append(copy.deepcopy(result["history"][0]))
        result["evidence"].append({"status": "UNKNOWN", "turn": 1})
        result["model_calls"] = [{"role": "doctor", "input_tokens": 12, "output_tokens": 3, "usage_reported": True}]
        result["simulator_calls"] = [{"role": "simulator", "input_tokens": 8, "output_tokens": 2, "usage_reported": True}]
        rows, summary = summarize([result], manifest)
        self.assertEqual(rows[0]["repeat_requests"], 1)
        self.assertEqual(rows[0]["unknown_requests"], 1)
        self.assertEqual(summary["rejected_action_rate"], 1 / 8)
        self.assertEqual(summary["schema_valid_action_rate"], 7 / 8)
        self.assertEqual(summary["total_doctor_input_tokens"], 12)
        self.assertEqual(summary["total_simulator_output_tokens"], 2)

    def test_demo_report_and_comparison_do_not_claim_accuracy(self):
        result, manifest = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            before, after = Path(tmp)/"before", Path(tmp)/"after"
            self.save(before, result, manifest)
            self.save(after, result, manifest)
            summary = write_report(before, [result], manifest)
            self.assertTrue((before / "cases.csv").exists())
            self.assertIsNone(summary["exact_match_accuracy_all_requested"])
            paired, comparison = compare_runs(before, after)
            self.assertIsNone(paired[0]["after_exact_match"])
            self.assertIsNone(comparison["delta_after_minus_before"]["exact_match_accuracy_all_requested"])

    def test_comparison_refuses_changed_controls_and_partial_runs(self):
        result, manifest = self.fixture(model=True)
        with tempfile.TemporaryDirectory() as tmp:
            before, after = Path(tmp)/"before", Path(tmp)/"after"
            self.save(before, result, manifest)
            changed = {**manifest, "doctor_model": "different"}
            self.save(after, result, changed)
            with self.assertRaisesRegex(ValueError, "doctor_model"):
                compare_runs(before, after)
            manifest["case_indices"] = [0, 1]
            for output in (before, after):
                (output / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "every requested"):
                compare_runs(before, after)

    def test_duplicate_result_cannot_inflate_accuracy(self):
        result, manifest = self.fixture(model=True)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize([result, result], manifest)

    def test_eval_cli_uses_split_order_and_independent_case_sessions(self):
        final = DemoDoctor().act({"force_diagnose": True, "evidence": [], "round": "preliminary"}, 0)
        final["diagnosis"] = "Mock diagnosis"
        payloads = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payloads.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                reply = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(final)}}],
                    "usage": {"prompt_tokens": 260001, "completion_tokens": 7}}
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(reply).encode())
            def log_message(self, *args):
                pass
        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                output, split_path = Path(tmp)/"run", Path(tmp)/"split.json"
                split = make_split(DATA, dev_count=2, test_count=1)
                split_path.write_text(json.dumps(split))
                completed = subprocess.run([sys.executable, "run.py", "eval", "--split-file", str(split_path),
                    "--limit", "2", "--base-url", f"http://127.0.0.1:{server.server_port}/v1",
                    "--output", str(output)], cwd=ROOT, capture_output=True, text=True, timeout=20)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                records = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
                self.assertEqual([r["case_index"] for r in records], split["dev"])
                self.assertEqual([len(r["model_calls"]) for r in records], [1, 1])
                self.assertTrue(all("gold" not in p["messages"][1]["content"] for p in payloads))
                summary = json.loads((output / "summary.json").read_text())
                self.assertEqual(summary["total_doctor_input_tokens"], 520002)
                self.assertTrue((output / "cases.csv").exists())
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
