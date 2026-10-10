import json
import re
import time
from .client import ModelError
from .data import flatten
from .memory import apply_update, build_memory


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
    def __init__(self, case, doctor, simulator, round_name, max_turns, max_seconds, client, verbose=False, working_memory=False, information_mode="interactive", observer=None):
        self.case, self.doctor, self.simulator = case, doctor, simulator
        self.round_name, self.max_turns, self.client = round_name, max_turns, client
        self.start = time.monotonic()
        self.deadline = self.start + max_seconds
        self.call_start = len(client.calls) if client else 0
        self.review_start = len(getattr(doctor, "reviews", []))
        self.observer = observer
        self.verbose = verbose
        self.working_memory = working_memory
        if information_mode not in ("interactive", "full"):
            raise ValueError("information_mode must be interactive or full")
        self.information_mode = information_mode

    def emit(self, kind, **data):
        if self.observer:
            self.observer(kind, case_index=self.case.index, **data)

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
        if self.information_mode == "full":
            # Diagnostic ablation: reveal only records accessible in this round, never the gold.
            kinds = ("SAY", "EXAM", "TEST") if self.round_name == "final" else ("SAY", "EXAM")
            for kind in kinds:
                for path, value in self.case.sources[kind].items():
                    if not any(e["source"] == path and e["value"] == value for e in evidence):
                        add_evidence(evidence, kind, path, value, 0)
        state = {"evidence": evidence, "history": [], "turns": 0, "rejections": 0,
                 "action": None, "final": None, "error": None}
        if self.working_memory:
            state.update(working_memory=build_memory(evidence), memory_updates=[])
        self.emit("case_start", evidence=evidence, max_turns=self.max_turns, memory_enabled=self.working_memory)
        return state

    def check_deadline(self):
        if time.monotonic() >= self.deadline:
            raise ModelError("Case deadline exceeded")

    def choose(self, state):
        try:
            self.check_deadline()
            public = {"round": self.round_name, "opening": self.case.opening,
                "information_mode": self.information_mode,
                "evidence": state["evidence"], "history": state["history"],
                "turns_used": state["turns"], "turns_remaining": self.max_turns-state["turns"],
                "force_diagnose": self.force_diagnose(state),
                "action_policy": self.action_policy(state),
                "asked_questions": [h["action"]["text"] for h in state["history"]
                    if h["accepted"] and h["action"].get("action") == "SAY"
                    and h["action"].get("intent") == "question"],
                "unavailable_requests": [e["source"] for e in state["evidence"]
                    if e["status"] == "UNKNOWN" and e["turn"] > 0]}
            memory = build_memory(state["evidence"], state.get("working_memory")) if self.working_memory else None
            if memory is not None:
                public["working_memory"] = memory
            if "SAY" not in public["action_policy"]["allowed_actions"]:
                self.log("진행 정책: " + public["action_policy"]["reason"])
            self.emit("thinking", turn=state["turns"], attempt=len(state["history"])+1)
            action = self.doctor.act(public, self.deadline)
            updates = list(state.get("memory_updates", []))
            if self.working_memory and isinstance(action, dict):
                action = dict(action)
                update = action.pop("working_memory", None)
                if update is None and any(k in action for k in ("hypotheses", "next_information_needed", "next_action_reason")):
                    update = {k: action.pop(k) for k in ("hypotheses", "next_information_needed", "next_action_reason") if k in action}
                if update is not None:
                    memory, error = apply_update(update, state["evidence"], memory)
                    updates.append({"turn": state["turns"], "accepted": error is None,
                                    "error": error, "hypotheses": memory["hypotheses"],
                                    "next_information_needed": memory["next_information_needed"],
                                    "next_action_reason": memory["next_action_reason"]})
                    if error:
                        self.log("메모리 갱신 제외: " + error)
            self.check_deadline()
            if self.client:
                own = [c for c in self.client.calls if c["role"] == "doctor"]
                if sum(c["input_tokens"] for c in own) > 500000 or sum(c["output_tokens"] for c in own) > 100000:
                    raise ModelError("Doctor session token budget exceeded")
            return {**state, "action": action, **({"working_memory": memory, "memory_updates": updates} if self.working_memory else {})}
        except ModelError as exc:
            return {**state, "error": str(exc)}

    def action_policy(self, state):
        if self.information_mode == "full":
            return {"allowed_actions": ["DIAGNOSE"], "consecutive_no_new_information": 0,
                    "reason": "Diagnostic ablation: all recorded information available in this round is already supplied. Submit DIAGNOSE using those observed facts."}
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
        if self.working_memory:
            # A plan describes this attempt, not a pending task after its outcome.
            state = {**state, "working_memory": {**state["working_memory"],
                "next_information_needed": "", "next_action_reason": ""}}
        # Copy lists so updating observations cannot mutate earlier checkpoints.
        state = {**state, "evidence": list(state["evidence"]), "history": list(state["history"])}
        action, evidence, history = state["action"], state["evidence"], state["history"]
        self.log(f"\n[case={self.case.index} 시도={len(history)+1} 사용턴={state['turns']}/{self.max_turns}]\n"
                 f"의사: {json.dumps(action, ensure_ascii=False)}")
        self.emit("action", action=action, turn=state["turns"], attempt=len(history)+1)
        def reject(reason):
            state["rejections"] += 1
            history.append({"action": action, "accepted": False, "feedback": reason, "turn": state["turns"]})
            self.emit("rejected", reason=reason, turn=state["turns"])
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
                self.emit("diagnosis", action=action, turn=state["turns"])
                state["final"] = action
                history.append({"action": action, "accepted": True, "turn": state["turns"]})
                return state
            self.emit("resolving", action=action, turn=state["turns"])
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
            if self.working_memory:
                state["working_memory"] = build_memory(evidence, state.get("working_memory"))
            if action.get("intent") == "question" or action["action"] in {"EXAM", "TEST"}:
                self.log("정보 갱신: " + ("새 근거 확보" if new_information else "새 정보 없음 (UNKNOWN 또는 기존 근거 재조회)"))
            responses = evidence[-len(new_ids):] if new_ids else []
            self.emit("observation", evidence=responses, turn=state["turns"], new_information=new_information)
            if self.working_memory:
                self.emit("memory", memory=state["working_memory"], turn=state["turns"])
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
            "information_mode": self.information_mode,
            "round": self.round_name, "completed": final is not None, "error": state["error"],
            "diagnosis": final["diagnosis"] if final else None,
            "gold_diagnosis": self.case.gold, "exact_match": None if demo else score,
            "score_note": "Exact text match only; synonyms may count as false. No official NOVA score.",
            "turns": state["turns"], "rejections": state["rejections"], "seconds": time.monotonic()-self.start,
            "history": state["history"], "evidence": state["evidence"],
            "soap": soap_from_evidence(final, state["evidence"]) if final else None,
            "diagnosis_reviews": getattr(self.doctor, "reviews", [])[self.review_start:],
            "model_calls": self.client.calls[self.call_start:] if self.client else []}



def run_case(case, doctor, simulator, round_name="preliminary", max_turns=50,
             max_seconds=1200, demo=False, client=None, engine="plain", verbose=False, working_memory=False, information_mode="interactive", observer=None):
    if engine not in {"plain", "langgraph"}:
        raise ValueError("engine must be plain or langgraph")
    encounter = Encounter(case, doctor, simulator, round_name, max_turns, max_seconds, client, verbose, working_memory, information_mode, observer)
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
    encounter.emit("case_end", completed=state["final"] is not None, error=state["error"], turn=state["turns"])
    return {**encounter.result(state, demo), "orchestration": orchestration,
            "working_memory": state.get("working_memory"), "memory_updates": state.get("memory_updates", [])}
