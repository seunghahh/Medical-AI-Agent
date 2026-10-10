import os
import copy
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from clinic.agent import DemoDoctor, DemoSimulator, Doctor
from clinic.client import ModelError
from clinic.data import ROOT, load_cases, provenance
from clinic.diagnosis import assess_result, assessment_for_report
from clinic.evaluation import summarize
from clinic.runner import run_case
from optimize import candidate_wins, validate_candidate


class Judge:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.payloads = reply, error, []

    def complete(self, system, payload, role, deadline):
        self.payloads.append((system, payload, role))
        if self.error:
            raise self.error
        return self.reply


class PromptSearchTests(unittest.TestCase):
    def result(self, prediction="PML", reference="Progressive multifocal encephalopathy (PML)"):
        return {"mode": "model", "completed": True, "diagnosis": prediction, "gold_diagnosis": reference}

    def test_alias_and_dataset_error_are_not_generic_acronym_or_fuzzy_matching(self):
        self.assertTrue(assess_result(self.result(), provenance())["match"])
        self.assertEqual(assess_result(self.result(), provenance())["method"], "dataset_label_variant")
        self.assertIsNone(assess_result(self.result(), {"sha256": "different"})["match"])
        self.assertIsNone(assess_result(self.result("Another disease (PML)"), provenance())["match"])
        self.assertTrue(assess_result(self.result("PML", "Progressive multifocal leukoencephalopathy"), {})["match"])
        self.assertIsNone(assess_result(self.result("Lymphoma", "Diffuse large B-cell lymphoma"), {})["match"])

    def test_semantic_judge_related_and_unknown_are_not_correct(self):
        judge = Judge({"relation": "related_but_not_equivalent", "reason": "Parent disease is less specific"})
        score = assess_result(self.result("Lymphoma", "Diffuse large B-cell lymphoma"), {}, judge)
        self.assertFalse(score["match"])
        self.assertEqual(set(judge.payloads[0][1]), {"prediction", "reference"})
        for reply in ({"relation": "uncertain", "reason": "Ambiguous"}, {"relation": "equivalent", "reason": ""}, {"match": True}):
            self.assertIsNone(assess_result(self.result("X", "Y"), {}, Judge(reply))["match"])
        score = assess_result(self.result("X", "Y"), {}, Judge(error=ModelError("offline")))
        self.assertEqual(score["method"], "judge_error")
        self.assertIsNone(score["match"])

    def test_unresolved_and_unattempted_cases_remain_in_denominator(self):
        result = run_case(load_cases()[0], DemoDoctor(), DemoSimulator(), demo=True)
        result["mode"] = "model"
        result["diagnosis_assessment"] = assess_result(result, {})
        other = copy.deepcopy(result)
        other.update(case_index=1, diagnosis="Unresolved disease", exact_match=False)
        other.pop("diagnosis_assessment")
        rows, summary = summarize([result, other], {"mode": "model", "case_indices": [0, 1, 2]})
        self.assertEqual(summary["diagnosis_matches"], 1)
        self.assertEqual(summary["diagnosis_match_accuracy_lower_bound_all_requested"], 1/3)
        self.assertEqual(summary["diagnosis_assessment_coverage_all_requested"], 1/3)
        self.assertEqual(summary["diagnosis_needs_review"], 1)
        self.assertEqual(summary["total_diagnosis_judge_input_tokens"], 0)
        other["diagnosis_judge_calls"] = [{"role": "diagnosis_judge", "input_tokens": 31,
                                          "output_tokens": 9, "usage_reported": True}]
        self.assertEqual(summarize([result, other], {"mode": "model", "case_indices": [0, 1, 2]})[1]["total_diagnosis_judge_input_tokens"], 31)
        result["diagnosis"] = "Changed label"
        with self.assertRaisesRegex(ValueError, "stale"):
            assessment_for_report(result, {})

    def test_prompt_supplements_rules_and_mandatory_policy(self):
        client = Judge({"action": "DIAGNOSE"})
        Doctor(client, "Focus on information gain.").act({"action_policy": {
            "reason": "No further useful information", "allowed_actions": ["DIAGNOSE"]}}, 0)
        prompt = client.payloads[0][0]
        self.assertIn("missing does not mean normal", prompt)
        self.assertLess(prompt.index("Focus on information gain"), prompt.index("MANDATORY current action policy"))

    def test_selection_rejects_coverage_drops_and_memorized_labels(self):
        baseline = {"diagnosis_matches": 1, "attempted_coverage": 1, "completion_rate_all_requested": 1,
                    "diagnosis_assessment_coverage_all_requested": 1, "unobserved_basis_rejections": 0,
                    "rejected_action_rate": .2, "repeat_requests": 1, "mean_turns": 10}
        improved = {**baseline, "mean_turns": 7}
        self.assertTrue(candidate_wins(improved, baseline))
        self.assertFalse(candidate_wins(baseline, baseline))
        for key in ("attempted_coverage", "completion_rate_all_requested", "diagnosis_assessment_coverage_all_requested"):
            self.assertFalse(candidate_wins({**improved, key: .5}, baseline))
        self.assertFalse(candidate_wins({**improved, "unobserved_basis_rejections": 1}, baseline))
        with self.assertRaisesRegex(ValueError, "diagnosis label"):
            validate_candidate({"instructions": "Always diagnose PML", "reason": "cheat"}, [self.result()])

    def test_assess_cli_preserves_original_run_and_soap(self):
        result = run_case(load_cases()[1], DemoDoctor(), DemoSimulator(), demo=True)
        result.update(self.result(), exact_match=False)
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp)/"old", Path(tmp)/"new"
            source.mkdir()
            manifest = {"mode": "model", "case_indices": [1], "provenance": provenance()}
            (source/"manifest.json").write_text(json.dumps(manifest))
            raw = json.dumps(result)+"\n"
            (source/"results.jsonl").write_text(raw)
            (source/"case-0-soap.json").write_text('{"original":true}')
            command = [sys.executable, "eval.py", "assess", str(source), "--output", str(output)]
            reply = subprocess.run(command, cwd=ROOT, env={**os.environ, "CLINIC_LIVE_DIR": str(Path(tempfile.gettempdir()) / "clinic-test-live")}, capture_output=True, text=True, timeout=20)
            self.assertEqual(reply.returncode, 0, reply.stderr)
            self.assertEqual((source/"results.jsonl").read_text(), raw)
            updated = json.loads((output/"results.jsonl").read_text())
            self.assertFalse(updated["exact_match"])
            self.assertTrue(updated["diagnosis_assessment"]["match"])
            self.assertEqual((output/"case-0-soap.json").read_text(), '{"original":true}')
            again = subprocess.run(command, cwd=ROOT, env={**os.environ, "CLINIC_LIVE_DIR": str(Path(tempfile.gettempdir()) / "clinic-test-live")}, capture_output=True, text=True, timeout=20)
            self.assertNotEqual(again.returncode, 0)

    def test_optimization_cli_evaluates_candidates_without_test_or_label_leakage(self):
        payloads = []
        split = json.loads((ROOT/"eval/split.json").read_text())
        case = load_cases()[split["dev"][0]]
        final = DemoDoctor().act({"force_diagnose": True, "evidence": [], "round": "preliminary"}, 0)
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                payloads.append(payload)
                system = payload["messages"][0]["content"]
                if "Improve generic workflow" in system:
                    content = {"instructions": "Compare observed findings before selecting the most specific supported diagnosis.", "reason": "Improve specificity"}
                else:
                    content = {**final, "diagnosis": case.gold if "most specific supported" in system else "Unrelated diagnosis"}
                body = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(content)}}],
                        "usage": {"prompt_tokens": 50, "completion_tokens": 20}}
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(body).encode())
            def log_message(self, *args):
                pass
        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp)/"search"
                command = [sys.executable, "optimize.py", "--engine", "plain", "--limit", "1", "--candidates", "1",
                           "--judge-mode", "aliases", "--base-url", f"http://127.0.0.1:{server.server_port}/v1", "--output", str(output)]
                reply = subprocess.run(command, cwd=ROOT, env={**os.environ, "CLINIC_LIVE_DIR": str(Path(tempfile.gettempdir()) / "clinic-test-live")}, capture_output=True, text=True, timeout=30)
                self.assertEqual(reply.returncode, 0, reply.stderr)
                report = json.loads((output/"optimization.json").read_text())
                self.assertEqual(report["selected"], "candidate-1")
                self.assertEqual(report["dev_case_indices"], [case.index])
                self.assertFalse(report["test_used"])
                self.assertFalse(report["weights_updated"])
                self.assertTrue((output/"baseline-failure-analysis.json").exists())
                self.assertIn("most specific supported", (output/"best-prompt.txt").read_text())
                for payload in payloads:
                    user = payload["messages"][1]["content"]
                    self.assertNotIn(case.gold, user)
                    self.assertNotIn("gold_diagnosis", user)
                    self.assertNotIn("reference", user)
                self.assertEqual(len(payloads), 3)  # doctor baseline, optimizer, doctor candidate
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
