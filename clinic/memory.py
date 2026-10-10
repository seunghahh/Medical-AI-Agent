"""Case-local memory built only from acquired evidence and grounded model hypotheses."""
import json


def build_memory(evidence, previous=None):
    previous = previous or {}
    unique = {}
    unknown = []
    for item in evidence:
        if item["status"] == "OBSERVED":
            key = (item["source"], json.dumps(item["value"], sort_keys=True, ensure_ascii=False))
            if key not in unique:
                unique[key] = {"id": item["id"], "source": item["source"], "value": item["value"]}
        elif item["turn"] > 0 and item["source"] not in unknown:
            unknown.append(item["source"])
    return {"observed_facts": list(unique.values()), "unavailable_requests": unknown,
            "hypotheses": previous.get("hypotheses", []),
            "next_information_needed": previous.get("next_information_needed", ""),
            "next_action_reason": previous.get("next_action_reason", "")}


def validate_update(update, evidence):
    if not isinstance(update, dict):
        return "working_memory must be an object"
    hypotheses = update.get("hypotheses")
    if not isinstance(hypotheses, list) or len(hypotheses) > 3:
        return "memory hypotheses must be a list of at most 3 candidates"
    observed = {e["id"] for e in evidence if e["status"] == "OBSERVED"}
    for hypothesis in hypotheses:
        if not isinstance(hypothesis, dict) or not isinstance(hypothesis.get("diagnosis"), str) or not hypothesis["diagnosis"].strip() or len(hypothesis["diagnosis"]) > 160:
            return "memory hypothesis diagnosis must be nonempty text (<=160 characters)"
        basis = hypothesis.get("basis")
        if not isinstance(basis, list) or len(basis) > 8 or any(not isinstance(i, str) or i not in observed for i in basis):
            return "memory hypotheses may cite only existing OBSERVED evidence IDs (<=8)"
        against = hypothesis.get("against", [])
        if not isinstance(against, list) or len(against) > 8 or any(not isinstance(i, str) or i not in observed for i in against):
            return "memory opposing evidence must cite OBSERVED IDs (<=8)"
        if not isinstance(hypothesis.get("missing", ""), str) or len(hypothesis.get("missing", "")) > 300:
            return "memory missing information must be text (<=300 characters)"
    for field in ("next_information_needed", "next_action_reason"):
        if not isinstance(update.get(field), str) or len(update[field]) > 300:
            return f"memory {field} must be text (<=300 characters)"
    return None


def apply_update(update, evidence, previous):
    error = validate_update(update, evidence)
    if error:
        return build_memory(evidence, previous), error
    clean = {"hypotheses": [{"diagnosis": h["diagnosis"], "basis": h["basis"],
                             "against": h.get("against", []), "missing": h.get("missing", "")}
                            for h in update["hypotheses"]],
             "next_information_needed": update["next_information_needed"],
             "next_action_reason": update["next_action_reason"]}
    return build_memory(evidence, clean), None
