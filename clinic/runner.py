import json
import re
import time
from .client import ModelError
from .data import flatten


def validate_action(action, round_name, force=False):
    if not isinstance(action, dict) or action.get("action") not in {"SAY", "EXAM", "TEST", "DIAGNOSE"}:
        return "Invalid action"
    kind = action["action"]
    if force and kind != "DIAGNOSE":
        return "Stopping policy requires DIAGNOSE"
    if kind == "TEST" and round_name == "preliminary":
        return "TEST is unavailable in the preliminary round"
    if kind != "DIAGNOSE":
        text = action.get("text")
        if not isinstance(text, str) or not text.strip():
            return "Action text is required"
        if kind == "SAY":
            if len(text) > 30:
                return "SAY exceeds 30 characters"
            if action.get("intent") not in {"question", "explanation", "empathy"}:
                return "Invalid SAY intent"
            if text.count("?") + text.count("？") > 1:
                return "Only one question per SAY"
    else:
        if not isinstance(action.get("diagnosis"), str) or not action["diagnosis"].strip():
            return "One primary diagnosis is required"
        if not isinstance(action.get("basis"), list) or any(not isinstance(x, str) for x in action["basis"]):
            return "basis must be a list of evidence IDs"
        if not isinstance(action.get("differential"), list) or any(not isinstance(x, str) for x in action["differential"]):
            return "differential must be a list of strings"
        plan = action.get("plan")
        if not isinstance(plan, dict) or any(not isinstance(plan.get(k), str) or not plan[k].strip()
            for k in ("further_tests", "treatment", "disposition", "follow_up", "education")):
            return "All five plan fields must contain text"
    return None


def normalize_diagnosis(text):
    return re.sub(r"[^\w]+", " ", text.casefold()).strip()


def soap_from_evidence(final, evidence):
    def records(kinds):
        return [e for e in evidence if e["action"] in kinds and e["status"] == "OBSERVED"]
    return {"S": records({"START", "SAY"}), "O": records({"VITAL", "EXAM", "TEST"}),
        "A": {"primary_diagnosis": final["diagnosis"], "basis": final["basis"],
              "differential": final["differential"]},
        "P": final["plan"],
        "unknown_requests": [e for e in evidence if e["status"] == "UNKNOWN"]}


def add_evidence(evidence, kind, path, value, turn, status="OBSERVED"):
    evidence.append({"id": f"e{len(evidence)+1}", "action": kind,
        "source": path, "value": value, "turn": turn, "status": status})


class Encounter:
    """Shared action semantics for the plain loop and LangGraph; hidden case stays here."""
    def __init__(self, case, doctor, simulator, round_name, max_turns, max_seconds, client, verbose=False):
        self.case, self.doctor, self.simulator = case, doctor, simulator
        self.round_name, self.max_turns, self.client = round_name, max_turns, client
        self.start = time.monotonic()
        self.deadline = self.start + max_seconds
        self.call_start = len(client.calls) if client else 0
        self.verbose = verbose

    def log(self, text):
        if self.verbose:
            print(text, flush=True)

    def initial_state(self):
        evidence = []
        opening = self.case.opening
        add_evidence(evidence, "START", "opening.demographics", opening["demographics"], 0,
            "UNKNOWN" if opening["demographics"] == "Not provided" else "OBSERVED")
        add_evidence(evidence, "START", "opening.chief_complaint", opening["chief_complaint"], 0)
        for path, value in flatten(opening["vital_signs"]).items():
            if path:
                add_evidence(evidence, "VITAL", "Vital_Signs." + path, value, 0)
        return {"evidence": evidence, "history": [], "turns": 0, "rejections": 0,
                "action": None, "final": None, "error": None}

    def check_deadline(self):
        if time.monotonic() >= self.deadline:
            raise ModelError("Case deadline exceeded")

    def choose(self, state):
        try:
            self.check_deadline()
            public = {"round": self.round_name, "opening": self.case.opening,
                "evidence": state["evidence"], "history": state["history"],
                "turns_used": state["turns"], "turns_remaining": self.max_turns-state["turns"],
                "force_diagnose": self.force_diagnose(state),
                "action_policy": self.action_policy(state),
                "asked_questions": [h["action"]["text"] for h in state["history"]
                    if h["accepted"] and h["action"].get("action") == "SAY"
                    and h["action"].get("intent") == "question"],
                "unavailable_requests": [e["source"] for e in state["evidence"]
                    if e["status"] == "UNKNOWN" and e["turn"] > 0]}
            if "SAY" not in public["action_policy"]["allowed_actions"]:
                self.log("진행 정책: " + public["action_policy"]["reason"])
            action = self.doctor.act(public, self.deadline)
            self.check_deadline()
            if self.client:
                own = [c for c in self.client.calls if c["role"] == "doctor"]
                if sum(c["input_tokens"] for c in own) > 500000 or sum(c["output_tokens"] for c in own) > 100000:
                    raise ModelError("Doctor session token budget exceeded")
            return {**state, "action": action}
        except ModelError as exc:
            return {**state, "error": str(exc)}

    def action_policy(self, state):
        stalled = 0
        for item in reversed(state["history"]):
            action = item["action"]
            if not item["accepted"] or (action["action"] == "SAY" and action.get("intent") != "question"):
                continue
            if item.get("new_information", True):
                break
            stalled += 1
        allowed = ["SAY", "EXAM", "DIAGNOSE"]
        if self.round_name == "final":
            allowed.insert(2, "TEST")
        reason = "Choose a useful action using acquired evidence."
        repeated = False
        for item in reversed(state["history"]):
            if item["accepted"]:
                break
            repeated |= item.get("feedback", "").startswith("Repeated question")
        # shortcut: local toy thresholds; validate clinical quality before competition use.
        if stalled >= 5:
            allowed = ["DIAGNOSE"]
            reason = "Five requests added no new recorded information. Submit a working diagnosis with uncertainty and further evaluation in the plan."
        elif stalled >= 3 or repeated:
            allowed.remove("SAY")
            reason = "Questioning is stalled. Choose a new specific EXAM (or TEST in the final round), or DIAGNOSE. Do not ask another SAY question in this step."
        return {"allowed_actions": allowed, "consecutive_no_new_information": stalled, "reason": reason}

    def force_diagnose(self, state):
        return (state["turns"] >= self.max_turns or self.deadline-time.monotonic() < 30
                or self.action_policy(state)["allowed_actions"] == ["DIAGNOSE"])

    def execute(self, state):
        # Copy lists so updating observations cannot mutate earlier checkpoints.
        state = {**state, "evidence": list(state["evidence"]), "history": list(state["history"])}
        action, evidence, history = state["action"], state["evidence"], state["history"]
        self.log(f"\n[case={self.case.index} 시도={len(history)+1} 사용턴={state['turns']}/{self.max_turns}]\n"
                 f"의사: {json.dumps(action, ensure_ascii=False)}")
        def reject(reason):
            state["rejections"] += 1
            history.append({"action": action, "accepted": False, "feedback": reason, "turn": state["turns"]})
            self.log(f"거절 ({state['rejections']}/10): {reason}")
            if state["rejections"] >= 10:
                state["error"] = "Too many rejected actions"
            return state
        try:
            self.check_deadline()
            reason = validate_action(action, self.round_name, self.force_diagnose(state))
            policy = self.action_policy(state)
            if not reason and action["action"] not in policy["allowed_actions"]:
                reason = "Action policy: " + policy["reason"]
            if not reason and action.get("intent") == "question" and action["action"] == "SAY":
                key = re.sub(r"[\W_]+", "", action["text"].casefold())
                if any(h["accepted"] and h["action"].get("action") == "SAY"
                       and h["action"].get("intent") == "question"
                       and re.sub(r"[\W_]+", "", h["action"]["text"].casefold()) == key
                       for h in history):
                    # shortcut: text matching catches literal repeats; paraphrases need semantic checks.
                    reason = "Repeated question: already asked. Choose a different useful question, EXAM or DIAGNOSE; UNKNOWN will not change by asking again."
            if not reason and action["action"] == "DIAGNOSE":
                known = {e["id"] for e in evidence if e["status"] == "OBSERVED"}
                if not set(action["basis"]).issubset(known):
                    reason = "Diagnosis cites unknown or unobserved evidence"
            if reason:
                return reject(reason)
            if action["action"] == "DIAGNOSE":
                state["final"] = action
                history.append({"action": action, "accepted": True, "turn": state["turns"]})
                return state
            observation = self.simulator.resolve(self.case, action, self.deadline)
            self.check_deadline()
            if not observation["accepted"]:
                return reject(observation["reason"])
            new_information = any(not any(e["status"] == "OBSERVED" and e["source"] == fact["path"]
                and e["value"] == fact["value"] for e in evidence) for fact in observation["facts"])
            state["turns"] += 1
            new_ids = []
            for fact in observation["facts"]:
                add_evidence(evidence, action["action"], fact["path"], fact["value"], state["turns"])
                new_ids.append(evidence[-1]["id"])
            if not observation["facts"] and action.get("intent") not in {"explanation", "empathy"}:
                add_evidence(evidence, action["action"], action["text"],
                    "Not recorded; cannot infer absence or normality", state["turns"], "UNKNOWN")
                new_ids.append(evidence[-1]["id"])
            history.append({"action": action, "accepted": True, "turn": state["turns"],
                            "evidence_ids": new_ids, "feedback": observation["reason"],
                            "new_information": new_information})
            if action.get("intent") == "question" or action["action"] in {"EXAM", "TEST"}:
                self.log("정보 갱신: " + ("새 근거 확보" if new_information else "새 정보 없음 (UNKNOWN 또는 기존 근거 재조회)"))
            responses = evidence[-len(new_ids):] if new_ids else []
            for item in responses:
                self.log(f"응답 [turn={item['turn']} {item['id']} {item['status']} {item['source']}]: "
                         f"{json.dumps(item['value'], ensure_ascii=False)}")
            if not new_ids:
                self.log("응답: 추가 관찰 없음 (설명/공감 발화)")
            return state
        except ModelError as exc:
            self.log(f"오류: {exc}")
            return {**state, "error": str(exc)}

    def result(self, state, demo):
        final = state["final"]
        # Gold is consulted only AFTER the encounter ends, outside graph state.
        score = normalize_diagnosis(final["diagnosis"]) == normalize_diagnosis(self.case.gold) if final else False
        return {"case_index": self.case.index, "mode": "scripted_demo" if demo else "model",
            "round": self.round_name, "completed": final is not None, "error": state["error"],
            "diagnosis": final["diagnosis"] if final else None,
            "gold_diagnosis": self.case.gold, "exact_match": None if demo else score,
            "score_note": "Exact text match only; synonyms may count as false. No official NOVA score.",
            "turns": state["turns"], "rejections": state["rejections"], "seconds": time.monotonic()-self.start,
            "history": state["history"], "evidence": state["evidence"],
            "soap": soap_from_evidence(final, state["evidence"]) if final else None,
            "model_calls": self.client.calls[self.call_start:] if self.client else []}


def run_case(case, doctor, simulator, round_name="preliminary", max_turns=50,
             max_seconds=1200, demo=False, client=None, engine="plain", verbose=False):
    if engine not in {"plain", "langgraph"}:
        raise ValueError("engine must be plain or langgraph")
    encounter = Encounter(case, doctor, simulator, round_name, max_turns, max_seconds, client, verbose)
    state = encounter.initial_state()
    orchestration = {"engine": engine}
    if engine == "langgraph":
        from .langgraph_runner import build_graph
        graph, config = build_graph(encounter)
        state = graph.invoke(state, config)
        orchestration["checkpoint_count"] = sum(1 for _ in graph.get_state_history(config))
    else:
        while not (state["final"] or state["error"]):
            state = encounter.choose(state)
            if not state["error"]:
                state = encounter.execute(state)
    return {**encounter.result(state, demo), "orchestration": orchestration}
