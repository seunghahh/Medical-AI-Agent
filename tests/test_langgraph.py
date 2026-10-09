import importlib.util
import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from clinic.agent import DemoDoctor, DemoSimulator, Doctor
from clinic.data import load_cases
from clinic.runner import Encounter, run_case


class Replies:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.inputs = []

    def complete(self, system, payload, role, deadline):
        self.inputs.append(json.loads(json.dumps(payload)))
        return next(self.replies)


@unittest.skipUnless(importlib.util.find_spec("langgraph"), "optional LangGraph engine not installed")
class LangGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = load_cases()[0]

    def test_plain_and_graph_have_identical_encounter_results(self):
        for round_name in ("preliminary", "final"):
            with self.subTest(round=round_name):
                results = [run_case(self.case, DemoDoctor(), DemoSimulator(),
                    round_name=round_name, demo=True, engine=e) for e in ("plain", "langgraph")]
                for result in results:
                    result.pop("seconds")
                    result.pop("orchestration")
                self.assertEqual(*results)
                self.assertIsNone(results[1]["exact_match"])

    def test_invalid_action_feedback_drives_next_action(self):
        final = DemoDoctor().act({"force_diagnose": True, "evidence": [], "round": "preliminary"}, 0)
        replies = Replies([{"action": "TEST", "text": "CBC"}, final])
        result = run_case(self.case, Doctor(replies), DemoSimulator(), engine="langgraph")
        self.assertTrue(result["completed"])
        self.assertEqual(result["turns"], 0)
        self.assertEqual(result["rejections"], 1)
        self.assertIn("unavailable", replies.inputs[1]["history"][-1]["feedback"])
        self.assertNotIn(self.case.gold, json.dumps(replies.inputs[0]))

    def test_repeated_question_is_blocked_in_graph(self):
        question = {"action": "SAY", "text": "증상은 언제 시작됐나요?", "intent": "question"}
        final = DemoDoctor().act({"force_diagnose": True, "evidence": [], "round": "preliminary"}, 0)
        replies = Replies([question, question, final])
        result = run_case(self.case, Doctor(replies), DemoSimulator(), engine="langgraph")
        self.assertTrue(result["completed"])
        self.assertEqual((result["turns"], result["rejections"]), (1, 1))
        self.assertTrue(result["history"][1]["feedback"].startswith("Repeated question"))

    def test_graph_stops_after_five_requests_without_new_information(self):
        class Missing:
            def resolve(self, *args):
                return {"accepted": True, "facts": [], "reason": "not_recorded"}
        final = DemoDoctor().act({"force_diagnose": True, "evidence": [], "round": "preliminary"}, 0)
        actions = [{"action": "SAY", "intent": "question", "text": f"증상 {i}이 있나요?"} for i in range(3)]
        actions += [{"action": "EXAM", "text": f"단일 진찰 {i}"} for i in range(2)]
        replies = Replies(actions + [final])
        result = run_case(self.case, Doctor(replies), Missing(), engine="langgraph")
        self.assertTrue(result["completed"])
        self.assertEqual((result["turns"], result["rejections"]), (5, 0))
        self.assertTrue(replies.inputs[-1]["force_diagnose"])
        self.assertEqual(replies.inputs[-1]["action_policy"]["allowed_actions"], ["DIAGNOSE"])

    def test_unobserved_basis_stops_after_rejection_budget(self):
        final = DemoDoctor().act({"force_diagnose": True, "evidence": [], "round": "preliminary"}, 0)
        final["basis"] = ["e999"]
        replies = Replies([final] * 10)
        result = run_case(self.case, Doctor(replies), DemoSimulator(), engine="langgraph")
        self.assertFalse(result["completed"])
        self.assertEqual(result["rejections"], 10)
        self.assertEqual(result["turns"], 0)

    def test_checkpoints_exclude_hidden_sources_and_keep_observation_order(self):
        from clinic.langgraph_runner import build_graph
        encounter = Encounter(self.case, DemoDoctor(), DemoSimulator(), "preliminary", 50, 1200, None)
        graph, config = build_graph(encounter)
        graph.invoke(encounter.initial_state(), config)
        snapshots = list(graph.get_state_history(config))
        self.assertGreater(len(snapshots), 3)
        for snapshot in snapshots:
            values = snapshot.values
            if not values:
                continue
            self.assertFalse({"gold", "gold_diagnosis", "sources", "case", "client"} & values.keys())
            self.assertTrue(all(e["turn"] <= values["turns"] for e in values["evidence"]))
            self.assertNotIn("Acetylcholine_Receptor_Antibodies", json.dumps(values))
        earliest = next(s for s in reversed(snapshots) if "evidence" in s.values)
        self.assertEqual(earliest.values["turns"], 0)
        self.assertEqual(earliest.values["history"], [])

    def test_case_memory_is_isolated_and_unknown_stays_unknown(self):
        from clinic.langgraph_runner import build_graph
        class Missing:
            def resolve(self, *args):
                return {"accepted": True, "facts": [], "reason": "not_recorded"}
        first = Encounter(self.case, DemoDoctor(), Missing(), "preliminary", 1, 1200, None)
        graph1, config1 = build_graph(first)
        state1 = graph1.invoke(first.initial_state(), config1)
        self.assertTrue(any(e["status"] == "UNKNOWN" for e in state1["evidence"]))
        second = Encounter(self.case, DemoDoctor(), DemoSimulator(), "preliminary", 1, 1200, None)
        graph2, config2 = build_graph(second)
        self.assertNotEqual(config1["configurable"]["thread_id"], config2["configurable"]["thread_id"])
        self.assertEqual(graph2.get_state(config2).values, {})
        self.assertEqual(second.initial_state()["history"], [])
        soap = first.result(state1, True)["soap"]
        self.assertTrue(all(e["status"] == "OBSERVED" for e in soap["S"] + soap["O"]))

    def test_turn_and_deadline_limits(self):
        result = run_case(self.case, DemoDoctor(), DemoSimulator(), max_turns=1,
                          demo=True, engine="langgraph")
        self.assertTrue(result["completed"])
        self.assertEqual(result["turns"], 1)
        expired = run_case(self.case, DemoDoctor(), DemoSimulator(), max_seconds=0,
                           demo=True, engine="langgraph")
        self.assertFalse(expired["completed"])
        self.assertIn("deadline", expired["error"])

    def test_verbose_graph_shows_acquired_response_and_diagnosis(self):
        with redirect_stdout(output := StringIO()):
            result = run_case(self.case, DemoDoctor(), DemoSimulator(), max_turns=1,
                              demo=True, engine="langgraph", verbose=True)
        self.assertTrue(result["completed"])
        self.assertIn("증상은 언제 시작됐나요?", output.getvalue())
        self.assertIn("OBSERVED History", output.getvalue())
        self.assertIn('"action": "DIAGNOSE"', output.getvalue())
