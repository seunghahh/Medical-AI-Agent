import json
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from clinic.agent import RESOLVER_PROMPT, Simulator
from clinic.client import ModelError
from clinic.data import load_cases
from test_clinic import FakeClient


class SimulatorCacheTests(unittest.TestCase):
    def setUp(self):
        self.case = load_cases()[27]
        self.action = {"action": "EXAM", "text": "Check forward flexion."}
        self.reply = {"accepted": True, "source_ids": ["f0"], "reason": ""}
        self.deadline = time.monotonic() + 30

    def test_same_request_cannot_change_within_or_across_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeClient([self.reply, {"accepted": True, "source_ids": [], "reason": "not_recorded"}])
            simulator = Simulator(client, tmp)
            first = simulator.resolve(self.case, self.action, self.deadline)
            second = simulator.resolve(self.case, {**self.action, "text": " CHECK  forward flexion. "}, self.deadline)
            other = Simulator(FakeClient([]), tmp)
            third = other.resolve(self.case, self.action, self.deadline)
            self.assertEqual(first, second)
            self.assertEqual(first, third)
            self.assertEqual(len(client.payloads), 1)
            self.assertEqual([r["cache_hit"] for r in simulator.resolutions], [False, True])
            self.assertTrue(other.resolutions[0]["cache_hit"])

    def test_unknown_and_bundle_rejections_are_also_stable(self):
        for reply in ({"accepted": True, "source_ids": [], "reason": "not_recorded"},
                      {"accepted": False, "source_ids": [], "reason": "bundled"}):
            with self.subTest(reply=reply):
                client = FakeClient([reply, self.reply])
                simulator = Simulator(client)
                self.assertEqual(simulator.resolve(self.case, self.action, self.deadline),
                                 simulator.resolve(self.case, self.action, self.deadline))
                self.assertEqual(len(client.payloads), 1)

    def test_cache_is_bound_to_sources_model_and_prompt_but_never_gold(self):
        client = FakeClient([self.reply] * 4)
        client.model = "mock"
        simulator = Simulator(client)
        simulator.resolve(self.case, self.action, self.deadline)
        simulator.resolve(replace(self.case, gold="HIDDEN_LABEL"), self.action, self.deadline)
        sources = {**self.case.sources, "EXAM": {**self.case.sources["EXAM"], "extra": "different"}}
        simulator.resolve(replace(self.case, sources=sources), self.action, self.deadline)
        client.model = "changed"
        simulator.resolve(self.case, self.action, self.deadline)
        with patch("clinic.agent.RESOLVER_PROMPT", RESOLVER_PROMPT + " changed"):
            simulator.resolve(self.case, self.action, self.deadline)
        self.assertEqual(len(client.payloads), 4)
        self.assertNotIn("HIDDEN_LABEL", json.dumps(client.payloads))
        self.assertNotIn("HIDDEN_LABEL", json.dumps(simulator.resolutions))

    def test_invalid_selections_and_corrupt_cache_fail_without_entering_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            simulator = Simulator(FakeClient([{"accepted": True, "source_ids": ["f999"], "reason": ""}]), tmp)
            with self.assertRaises(ModelError):
                simulator.resolve(self.case, self.action, self.deadline)
            self.assertEqual(list(Path(tmp).iterdir()), [])
            self.assertEqual(simulator.cache, {})
            simulator = Simulator(FakeClient([self.reply]), tmp)
            simulator.resolve(self.case, self.action, self.deadline)
            path = next(Path(tmp).iterdir())
            path.write_text('{"accepted":true,"source_ids":["f999"]}')
            with self.assertRaises(ModelError):
                Simulator(FakeClient([]), tmp).resolve(self.case, self.action, self.deadline)
            path.write_text('broken JSON')
            with self.assertRaisesRegex(ModelError, "cache unavailable"):
                Simulator(FakeClient([]), tmp).resolve(self.case, self.action, self.deadline)

    def test_communication_does_not_fetch_facts_and_cache_does_not_bypass_deadline(self):
        simulator = Simulator(FakeClient([]))
        action = {"action": "SAY", "intent": "empathy", "text": "힘드셨겠어요."}
        self.assertEqual(simulator.resolve(self.case, action, self.deadline)["facts"], [])
        self.assertEqual(simulator.client.payloads, [])
        with self.assertRaisesRegex(ModelError, "deadline"):
            simulator.resolve(self.case, action, time.monotonic() - 1)


if __name__ == "__main__":
    unittest.main()
