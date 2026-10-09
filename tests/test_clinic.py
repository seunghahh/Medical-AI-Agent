import json
import threading
import time
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import BytesIO, StringIO
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, HTTPServer
from clinic.agent import DemoDoctor, DemoSimulator, Doctor, Simulator
from clinic.client import ChatClient, ModelError, ResponseFormatError, parse_json
from clinic.data import load_cases
from clinic.runner import Encounter, run_case, validate_action


class FakeClient:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.payloads = []

    def complete(self, system, payload, role, deadline):
        # Serialize now, preventing later ledger mutations from hiding leakage.
        self.payloads.append(json.loads(json.dumps(payload)))
        return next(self.replies)


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_cases()

    def test_dataset_shape(self):
        self.assertEqual(len(self.cases), 215)
        for case in self.cases:
            self.assertTrue(case.gold)
            self.assertEqual(set(case.sources), {"SAY", "EXAM", "TEST"})

    def test_demo_has_no_accuracy_claim(self):
        result = run_case(self.cases[0], DemoDoctor(), DemoSimulator(), demo=True)
        self.assertTrue(result["completed"])
        self.assertIsNone(result["exact_match"])
        self.assertEqual(result["turns"], 5)
        self.assertFalse(any(e["action"] == "TEST" for e in result["evidence"]))
        self.assertTrue(all("turn" in e for e in result["soap"]["O"]))

    def test_final_round_test(self):
        result = run_case(self.cases[0], DemoDoctor(), DemoSimulator(), round_name="final", demo=True)
        self.assertEqual(result["turns"], 6)
        self.assertTrue(any(e["action"] == "TEST" for e in result["soap"]["O"]))

    def test_turn_limit_zero_cost_diagnose(self):
        result = run_case(self.cases[0], DemoDoctor(), DemoSimulator(), max_turns=1, demo=True)
        self.assertTrue(result["completed"])
        self.assertEqual(result["turns"], 1)
        self.assertEqual(result["history"][-1]["action"]["action"], "DIAGNOSE")

    def test_say_limits_and_test_gate(self):
        self.assertIsNotNone(validate_action({"action":"SAY", "text":"가"*31,"intent":"question"}, "preliminary"))
        self.assertIsNotNone(validate_action({"action":"SAY", "text":"두통? 복통?","intent":"question"}, "preliminary"))
        self.assertIsNotNone(validate_action({"action":"TEST", "text":"CBC"}, "preliminary"))
        self.assertIsNone(validate_action({"action":"TEST", "text":"CBC"}, "final"))

    def test_doctor_does_not_receive_hidden_fields(self):
        final = DemoDoctor().act({"force_diagnose": True,"evidence":[],"round":"preliminary"}, 0)
        spy = FakeClient([final])
        run_case(self.cases[0], Doctor(spy), DemoSimulator())
        payload = json.dumps(spy.payloads[0])
        self.assertNotIn(self.cases[0].gold, payload)
        self.assertNotIn("Acetylcholine_Receptor_Antibodies", payload)
        self.assertNotIn("History", payload)
        self.assertNotIn("Objective_for_Doctor", payload)

    def test_missing_result_unknown(self):
        fake = FakeClient([{"accepted":True,"source_ids":[],"reason":"not_recorded"}])
        observation = Simulator(fake).resolve(self.cases[0], {"action":"TEST", "text":"unknown test"}, time.monotonic()+30)
        self.assertEqual(observation["facts"], [])
        self.assertNotIn(self.cases[0].gold, json.dumps(fake.payloads))

    def test_repeated_question_is_reselected_without_simulator_call_or_turn(self):
        question = {"action": "SAY", "text": "눈 움직임이 제한되나요?", "intent": "question"}
        variant = {**question, "text": "눈움직임이 제한되나요？"}
        final = DemoDoctor().act({"force_diagnose": True, "evidence": [], "round": "preliminary"}, 0)
        for facts in ([], [{"path": "History", "value": "Recorded answer"}]):
            with self.subTest(facts=facts):
                spy = FakeClient([question, variant, final])
                simulator = DemoSimulator()
                with patch.object(simulator, "resolve", return_value={"accepted": True,
                                  "facts": facts, "reason": "" if facts else "not_recorded"}) as resolve:
                    result = run_case(self.cases[0], Doctor(spy), simulator)
                self.assertTrue(result["completed"])
                self.assertEqual(result["turns"], 1)
                self.assertEqual(result["rejections"], 1)
                self.assertEqual(resolve.call_count, 1)
                self.assertTrue(result["history"][1]["feedback"].startswith("Repeated question"))
                self.assertEqual(spy.payloads[2]["asked_questions"], [question["text"]])
                self.assertEqual(spy.payloads[2]["unavailable_requests"], [] if facts else [question["text"]])
                self.assertEqual(spy.payloads[1]["evidence"], spy.payloads[2]["evidence"])

    def test_repeated_facts_and_unknowns_switch_to_exam_and_new_facts_reset_policy(self):
        simulator = DemoSimulator()
        encounter = Encounter(self.cases[0], DemoDoctor(), simulator, "preliminary", 50, 120, None)
        state = encounter.initial_state()
        replies = [{"accepted": True, "facts": [{"path": "History", "value": "same record"}], "reason": ""}] * 2
        replies += [{"accepted": True, "facts": [], "reason": "not_recorded"}] * 2
        with patch.object(simulator, "resolve", side_effect=replies):
            for text in ("언제 시작됐나요?", "언제부터 시작됐나요?", "수직 복시인가요?", "눈 통증이 있나요?"):
                state = encounter.execute({**state, "action": {"action": "SAY", "intent": "question", "text": text}})
        self.assertEqual([h["new_information"] for h in state["history"]], [True, False, False, False])
        self.assertEqual(encounter.action_policy(state)["allowed_actions"], ["EXAM", "DIAGNOSE"])
        with patch.object(simulator, "resolve") as resolve:
            state = encounter.execute({**state, "action": {"action": "SAY", "intent": "question", "text": "다른 질문이 있나요?"}})
            resolve.assert_not_called()
        self.assertEqual(state["turns"], 4)
        self.assertIn("Action policy", state["history"][-1]["feedback"])
        with patch.object(simulator, "resolve", return_value={"accepted": True,
                          "facts": [{"path": "new_exam", "value": "recorded finding"}], "reason": ""}):
            state = encounter.execute({**state, "action": {"action": "EXAM", "text": "단일 진찰"}})
        self.assertEqual(encounter.action_policy(state)["consecutive_no_new_information"], 0)
        self.assertIn("SAY", encounter.action_policy(state)["allowed_actions"])

    def test_repeat_recovery_policy_survives_other_rejections(self):
        encounter = Encounter(self.cases[0], DemoDoctor(), DemoSimulator(), "final", 50, 120, None)
        state = encounter.initial_state()
        action = {"action": "SAY", "text": "언제 시작됐나요?", "intent": "question"}
        state = encounter.execute({**state, "action": action})
        state = encounter.execute({**state, "action": action})
        state = encounter.execute({**state, "action": {"action": "INVALID"}})
        self.assertEqual(encounter.action_policy(state)["allowed_actions"], ["EXAM", "TEST", "DIAGNOSE"])

    def test_hallucinated_source_id_rejected(self):
        fake = FakeClient([{"accepted":True,"source_ids":["f999"],"reason":""}])
        with self.assertRaises(ModelError):
            Simulator(fake).resolve(self.cases[0], {"action":"EXAM","text":"ptosis"}, time.monotonic()+30)

    def test_bundled_exam_no_turn_charge(self):
        class Bundled:
            def resolve(self, *args):
                return {"accepted":False,"facts":[],"reason":"bundled"}
        final = DemoDoctor().act({"force_diagnose":True,"evidence":[],"round":"preliminary"}, 0)
        spy = FakeClient([{"action":"EXAM","text":"several exams"}, final])
        result = run_case(self.cases[0], Doctor(spy), Bundled())
        self.assertEqual(result["turns"], 0)
        self.assertEqual(result["rejections"], 1)

    def test_unknown_not_added_to_objective(self):
        class Missing:
            def resolve(self, *args):
                return {"accepted":True,"facts":[],"reason":"not_recorded"}
        result = run_case(self.cases[0], DemoDoctor(), Missing(), max_turns=1, demo=True)
        self.assertEqual(len(result["soap"]["unknown_requests"]), 1)
        self.assertTrue(all(e["status"] == "OBSERVED" for e in result["soap"]["S"] + result["soap"]["O"]))

    def test_unseen_evidence_reference_rejected(self):
        final = DemoDoctor().act({"force_diagnose":True,"evidence":[],"round":"preliminary"}, 0)
        final["basis"] = ["e999"]
        spy = FakeClient([final]*10)
        result = run_case(self.cases[0], Doctor(spy), DemoSimulator())
        self.assertFalse(result["completed"])
        self.assertEqual(result["rejections"], 10)

    def test_deadline(self):
        result = run_case(self.cases[0], DemoDoctor(), DemoSimulator(), max_seconds=0, demo=True)
        self.assertFalse(result["completed"])
        self.assertIn("deadline", result["error"])

    def test_verbose_shows_actions_before_observations_without_hidden_data(self):
        output = StringIO()
        class Missing:
            def resolve(self, case, action, deadline):
                self_test.assertIn(action["text"], output.getvalue())
                return {"accepted": True, "facts": [], "reason": "not_recorded"}
        self_test = self
        case = replace(self.cases[0], gold="HIDDEN_GOLD_FOR_TEST")
        final = DemoDoctor().act({"force_diagnose": True, "evidence": [], "round": "preliminary"}, 0)
        replies = [{"action": "SAY", "text": "가" * 31, "intent": "question"},
                   {"action": "SAY", "text": "두통이 있나요?", "intent": "question"}, final]
        with redirect_stdout(output):
            result = run_case(case, Doctor(FakeClient(replies)), Missing(), verbose=True)
        text = output.getvalue()
        self.assertTrue(result["completed"])
        self.assertIn("거절 (1/10): SAY exceeds 30 characters", text)
        self.assertIn("turn=1", text)
        self.assertIn("UNKNOWN", text)
        self.assertIn('"action": "DIAGNOSE"', text)
        self.assertNotIn(case.gold, text)
        self.assertNotIn("Acetylcholine_Receptor_Antibodies", text)
        with redirect_stdout(output := StringIO()):
            run_case(case, DemoDoctor(), DemoSimulator(), demo=True)
        self.assertEqual(output.getvalue(), "")

    def test_json_parser(self):
        self.assertEqual(parse_json('```json\n{"action":"SAY"}\n```'), {"action":"SAY"})
        with self.assertRaises(ValueError):
            parse_json('[1, 2]')

    def test_http_transport_final_channel_and_usage(self):
        captured = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                captured.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                body = {"choices":[{"finish_reason":"stop","message":{
                    "reasoning_content":"not final JSON", "content":'{"ok":true}'}}],
                    "usage":{"prompt_tokens":11,"completion_tokens":7}}
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
            client = ChatClient(f"http://127.0.0.1:{server.server_port}/v1", "gpt-oss:20b")
            self.assertEqual(client.complete("test", {}, "doctor", time.monotonic()+5), {"ok":True})
            self.assertEqual(client.calls[0]["output_tokens"], 7)
            self.assertEqual(captured[0]["model"], "gpt-oss:20b")
            self.assertEqual(captured[0]["reasoning_effort"], "low")
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_assistant_arguments_are_json_data_and_other_tools_are_rejected(self):
        def complete(message, finish="tool_calls"):
            body = {"choices": [{"finish_reason": finish, "message": message}],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 7}}
            client = ChatClient("http://localhost:11434/v1", "gpt-oss:20b")
            with patch("clinic.client.urllib.request.urlopen", return_value=BytesIO(json.dumps(body).encode())):
                answer = client.complete("test", {}, "doctor", time.monotonic() + 5)
            return answer, client.calls[0]
        call = {"type": "function", "function": {"name": "assistant", "arguments": '{"action":"SAY","text":"증상은 언제 시작됐나요?","intent":"question"}'}}
        message = {"content": "", "reasoning": '{"action":"DIAGNOSE"}', "tool_calls": [call]}
        answer, usage = complete(message)
        self.assertEqual(answer["action"], "SAY")
        self.assertEqual(usage["final_source"], "assistant_arguments")
        self.assertEqual(usage["finish_reason"], "tool_calls")
        action_call = {**call, "function": {**call["function"], "name": "ACTION"}}
        answer, usage = complete({"content": "", "tool_calls": [action_call]})
        self.assertIsNone(validate_action(answer, "preliminary"))
        self.assertEqual(usage["final_source"], "ACTION_arguments")
        overlong = {"action": "SAY", "text": "가" * 31, "intent": "question"}
        action_call["function"]["arguments"] = json.dumps(overlong)
        answer, _ = complete({"content": "", "tool_calls": [action_call]})
        self.assertEqual(validate_action(answer, "preliminary"), "SAY exceeds 30 characters")
        for calls in ([{**call, "function": {"name": "python", "arguments": '{"ok":true}'}}], [call, call]):
            with self.subTest(calls=calls), self.assertRaisesRegex(ModelError, "Unexpected tool call"):
                complete({"content": "", "tool_calls": calls})
        with self.assertRaisesRegex(ModelError, "No final answer"):
            complete({"content": "", "reasoning": '{"ok":true}'}, "stop")
        with self.assertRaisesRegex(ModelError, "Invalid final JSON"):
            complete({"content": "", "tool_calls": [{**call, "function": {"name": "assistant", "arguments": "not json"}}]})
        with self.assertRaisesRegex(ModelError, "Generation truncated"):
            complete(message, "length")

    def test_format_error_reselects_but_endpoint_error_stops(self):
        client = ChatClient("http://localhost:11434/v1", "gpt-oss:20b")
        final = DemoDoctor().act({"force_diagnose": True, "evidence": [], "round": "preliminary"}, 0)
        with patch.object(client, "complete", side_effect=[ResponseFormatError("No final answer"), final]) as request:
            result = run_case(self.cases[0], Doctor(client), DemoSimulator(), client=client)
        self.assertTrue(result["completed"])
        self.assertEqual(result["rejections"], 1)
        self.assertEqual(result["turns"], 0)
        self.assertEqual(request.call_count, 2)
        self.assertEqual(result["history"][0]["action"]["format_error"], "No final answer")
        with patch.object(client, "complete", side_effect=ResponseFormatError("No final answer")) as request:
            result = run_case(self.cases[0], Doctor(client), DemoSimulator(), client=client)
        self.assertFalse(result["completed"])
        self.assertEqual(result["rejections"], 10)
        self.assertEqual(request.call_count, 10)
        with patch.object(client, "complete", side_effect=ModelError("Endpoint unavailable")) as request:
            result = run_case(self.cases[0], Doctor(client), DemoSimulator(), client=client)
        self.assertEqual(result["error"], "Endpoint unavailable")
        self.assertEqual(result["rejections"], 0)
        self.assertEqual(request.call_count, 1)


if __name__ == "__main__":
    unittest.main()
