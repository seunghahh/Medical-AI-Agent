import os
import copy
import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from clinic.agent import DemoDoctor, DemoSimulator, Doctor
from clinic.data import DATA, ROOT, load_cases, provenance
from clinic.diagnosis import assess_result, assessment_for_report
from clinic.evaluation import compare_runs, load_split
from clinic.runner import Encounter, run_case
from research_pipeline import reserve_split
from optimize import training_feedback
from test_clinic import FakeClient


class ResearchPipelineTests(unittest.TestCase):
    def test_full_information_is_round_scoped_and_gold_free_in_both_engines(self):
        original = load_cases()[0]
        case = replace(original, gold="HIDDEN_REFERENCE_MARKER", sources={
            "SAY": {"History": "RECORDED_HISTORY_MARKER"},
            "EXAM": {"Physical": "RECORDED_EXAM_MARKER"},
            "TEST": {"Laboratory": "HIDDEN_LAB_MARKER"}})
        final = DemoDoctor().act({"force_diagnose": True, "evidence": [], "round": "preliminary"}, 0)
        for engine in ("plain", "langgraph"):
            client = FakeClient([copy.deepcopy(final)])
            result = run_case(case, Doctor(client), DemoSimulator(), engine=engine, information_mode="full")
            text = json.dumps(client.payloads)
            self.assertIn("RECORDED_HISTORY_MARKER", text)
            self.assertIn("RECORDED_EXAM_MARKER", text)
            self.assertNotIn("HIDDEN_REFERENCE_MARKER", text)
            self.assertNotIn("HIDDEN_LAB_MARKER", text)
            self.assertTrue(result["completed"])
            self.assertEqual(result["turns"], 0)
            self.assertTrue(client.payloads[0]["force_diagnose"])
        encounter = Encounter(case, None, None, "final", 10, 180, None, information_mode="full")
        self.assertIn("HIDDEN_LAB_MARKER", json.dumps(encounter.initial_state()))

    def test_specificity_requires_context_and_explicit_support_and_stale_scores_fail(self):
        case = replace(load_cases()[0], gold="Reference disease")
        result = {"case_index": case.index, "mode": "model", "completed": True,
                  "diagnosis": "Specific reference disease", "gold_diagnosis": case.gold}
        reply = {"relation": "compatible_specific", "reason": "Recorded age supports refinement", "specificity_supported": True}
        client = FakeClient([copy.deepcopy(reply)])
        accepted = assess_result(result, {}, client, case)
        self.assertTrue(accepted["match"])
        self.assertEqual(client.payloads[0]["case_context"]["recorded_findings"], case.sources)
        self.assertIsNone(assess_result(result, {}, FakeClient([reply]))["match"])
        self.assertIsNone(assess_result(result, {}, FakeClient([{**reply, "specificity_supported": False}]), case)["match"])
        result["diagnosis_assessment"] = accepted
        self.assertTrue(assessment_for_report(result, {})["match"])
        result["diagnosis_assessment"]["specificity_supported"] = False
        with self.assertRaisesRegex(ValueError, "unsupported"):
            assessment_for_report(result, {})
        with self.assertRaisesRegex(ValueError, "mismatch"):
            assess_result(result, {}, FakeClient([reply]), replace(case, gold="Different reference"))

    def test_reservation_excludes_requested_cases_and_validates_train_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = Path(tmp)/"old"
            old.mkdir()
            stamp = {**provenance(), "dataset": "renamed-copy.jsonl", "upstream_commit": None}
            (old/"manifest.json").write_text(json.dumps({"provenance": stamp, "case_indices": [0, 1, 2]}))
            split = reserve_split(DATA, tmp, 3, 3, 3, 42)
            self.assertEqual(split, reserve_split(DATA, tmp, 3, 3, 3, 42))
            self.assertFalse(set(sum([split[k] for k in ("train", "dev", "test")], [])) & {0, 1, 2})
            path = Path(tmp)/"split.json"
            path.write_text(json.dumps(split))
            self.assertEqual(load_split(path, DATA), split)
            split["train"][0] = split["dev"][0]
            path.write_text(json.dumps(split))
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_split(path, DATA)
            (old/"manifest.json").write_text(json.dumps({"provenance": stamp, "case_indices": ["0"]}))
            with self.assertRaisesRegex(ValueError, "previous case indices"):
                reserve_split(DATA, tmp, 3, 3, 3, 42)

    def test_case_binding_applies_even_to_fast_exact_and_alias_paths(self):
        case = load_cases()[0]
        for prediction in (case.gold, "PML"):
            result = {"case_index": case.index, "mode": "model", "completed": True,
                      "diagnosis": prediction, "gold_diagnosis": prediction}
            with self.assertRaisesRegex(ValueError, "mismatch"):
                assess_result(result, provenance(), case=replace(case, index=99))

    def test_training_feedback_does_not_call_opening_age_and_complaint_unacquired(self):
        case = load_cases()[0]
        result = run_case(case, DemoDoctor(), DemoSimulator(), demo=True)
        result["diagnosis_assessment"] = {"match": False}
        feedback = training_feedback([result], {case.index: case})[0]
        paths = {x["source"] for x in feedback["unacquired_recorded_facts"]}
        self.assertNotIn("Demographics", paths)
        self.assertNotIn("Symptoms.Primary_Symptom", paths)

    def test_full_mode_explicit_action_instruction_does_not_fabricate_a_model_action(self):
        systems = []
        class Spy:
            def complete(self, system, payload, role, deadline):
                systems.append(system)
                return {"diagnosis": "Candidate"}
        doctor = Doctor(Spy())
        answer = doctor.act({"information_mode": "full"}, time.monotonic()+10)
        self.assertEqual(answer, {"diagnosis": "Candidate"})
        self.assertIn('literal field "action":"DIAGNOSE"', systems[-1])
        doctor.act({"information_mode": "interactive"}, time.monotonic()+10)
        self.assertNotIn("Offline diagnostic mode", systems[-1])

    def test_complete_pipeline_uses_train_feedback_then_frozen_dev_selection_before_test(self):
        payloads = []
        fixtures = json.loads((ROOT/"eval/diagnosis-rubric.json").read_text())["fixtures"]
        final = DemoDoctor().act({"force_diagnose": True, "evidence": [], "round": "preliminary"}, 0)
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                payloads.append(payload)
                system = payload["messages"][0]["content"]
                user = json.loads(payload["messages"][1]["content"])
                if system.startswith("Assess two diagnosis labels"):
                    fixture = next((f for f in fixtures if f["prediction"] == user["prediction"]
                        and f["reference"] == user["reference"] and
                        {k: f["record"].get(k, {}) for k in ("SAY", "EXAM", "TEST")} == user["case_context"]["recorded_findings"]), None)
                    expected = fixture["expected"] if fixture else user["prediction"] == user["reference"]
                    content = {"relation": "equivalent" if expected is True else "different" if expected is False else "uncertain",
                               "reason": "Synthetic test judgment"}
                elif "TRAIN-only reflection" in system:
                    content = {"instructions": "REFINED_GUIDANCE: review the observed pattern before committing.", "reason": "Interpret acquired evidence"}
                else:
                    index = user["opening"]["chief_complaint"].split("-")[-1]
                    content = {**final, "diagnosis": "Disease-"+index if "REFINED_GUIDANCE" in system else "Wrong disease"}
                body = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(content)}}],
                        "usage": {"prompt_tokens": 20, "completion_tokens": 20}}
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
                tmp = Path(tmp)
                dataset, output = tmp/"synthetic.jsonl", tmp/"study"
                raw = [{"OSCE_Examination": {"Patient_Actor": {"History": "History-"+str(i),
                    "Symptoms": {"Primary_Symptom": "Symptom-"+str(i)}}, "Physical_Examination_Findings": {"Skin": "Exam-"+str(i)},
                    "Test_Results": {"Lab": "LAB_MARKER-"+str(i)}, "Objective_for_Doctor": "OBJECTIVE_MARKER",
                    "Correct_Diagnosis": "Disease-"+str(i)}} for i in range(6)]
                dataset.write_text("\n".join(json.dumps(x) for x in raw))
                process = subprocess.run([sys.executable, "research_pipeline.py", "--output", str(output),
                    "--dataset", str(dataset), "--history", str(tmp/"history"), "--engine", "plain",
                    "--train-count", "1", "--dev-count", "1", "--test-count", "1",
                    "--base-url", f"http://127.0.0.1:{server.server_port}/v1"], cwd=ROOT, env={**os.environ, "CLINIC_LIVE_DIR": str(Path(tempfile.gettempdir()) / "clinic-test-live")}, capture_output=True, text=True, timeout=40)
                self.assertEqual(process.returncode, 0, process.stdout+process.stderr)
                plan = json.loads((output/"pipeline.json").read_text())
                self.assertEqual(plan["status"], "completed")
                self.assertEqual(plan["dev_selected"], "optimized")
                self.assertFalse(plan["test_used_for_optimization"])
                self.assertEqual(plan["rubric_validation"]["passed"], len(fixtures))
                reflections = [p for p in payloads if "TRAIN-only reflection" in p["messages"][0]["content"]]
                self.assertEqual(len(reflections), 1)
                reflection = reflections[0]["messages"][1]["content"]
                self.assertIn("Disease-"+str(plan["split"]["train"][0]), reflection)
                self.assertNotIn("Disease-"+str(plan["split"]["test"][0]), reflection)
                doctors = [p for p in payloads if p["messages"][0]["content"].startswith("You are a doctor")]
                self.assertEqual(len(doctors), 8)
                for payload in doctors:
                    user = payload["messages"][1]["content"]
                    self.assertNotIn("Disease-", user)
                    self.assertNotIn("OBJECTIVE_MARKER", user)
                    self.assertNotIn("LAB_MARKER", user)
                with self.assertRaisesRegex(ValueError, "information_mode"):
                    compare_runs(output/"dev/baseline", output/"dev/full")
                self.assertTrue(compare_runs(output/"dev/baseline", output/"dev/full", True)[1]["diagnostic_ablation"])
                self.assertTrue((output/"scorecard.csv").exists())
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
