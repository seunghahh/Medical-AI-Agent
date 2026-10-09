"""Local practice metrics, fixed case splits, and paired run comparisons."""
import csv
import json
import random
from pathlib import Path
from .data import load_cases, provenance
from .runner import validate_action


def make_split(dataset, dev_count=20, test_count=30, seed=20261009):
    indices = [case.index for case in load_cases(dataset)]
    if dev_count < 1 or test_count < 1 or dev_count + test_count > len(indices):
        raise ValueError("dev/test counts must be positive and fit the dataset")
    random.Random(seed).shuffle(indices)
    return {"schema_version": 1, "provenance": provenance(dataset), "seed": seed,
            "dev": indices[:dev_count], "test": indices[dev_count:dev_count + test_count],
            "note": "Public practice cases; not official NOVA cases or a clinical benchmark."}


def load_split(path, dataset):
    split = json.loads(Path(path).read_text(encoding="utf-8"))
    if split.get("schema_version") != 1 or split.get("provenance") != provenance(dataset):
        raise ValueError("split schema/dataset provenance mismatch; use the original dataset")
    available = {case.index for case in load_cases(dataset)}
    selected = []
    for name in ("dev", "test"):
        values = split.get(name)
        if not isinstance(values, list) or not values:
            raise ValueError(f"{name} must be a nonempty list of case indices")
        if any(type(i) is not int or i not in available for i in values):
            raise ValueError(f"invalid {name} case index")
        selected.extend(values)
    if len(selected) != len(set(selected)):
        raise ValueError("duplicate cases or dev/test overlap")
    return split


def case_metrics(result):
    history = result["history"]
    accepted = [h for h in history if h["accepted"]]
    requests = [h["action"] for h in accepted if h["action"].get("action") != "DIAGNOSE"]
    signatures = [json.dumps(a, sort_keys=True, ensure_ascii=False) for a in requests]
    calls = result.get("model_calls", []) + result.get("simulator_calls", [])
    row = {"case_index": result["case_index"], "completed": result["completed"],
        "exact_match": result["exact_match"], "turns": result["turns"],
        "seconds": result["seconds"], "action_attempts": len(history),
        "rejected_actions": sum(not h["accepted"] for h in history),
        "schema_valid_actions": sum(validate_action(h["action"], result["round"]) is None for h in history),
        "repeat_requests": len(signatures) - len(set(signatures)),
        "unknown_requests": sum(e["status"] == "UNKNOWN" and e["turn"] > 0 for e in result["evidence"]),
        "unobserved_basis_rejections": sum(h.get("feedback") == "Diagnosis cites unknown or unobserved evidence" for h in history),
        "doctor_calls": sum(c["role"] == "doctor" for c in calls),
        "simulator_calls": sum(c["role"] == "simulator" for c in calls),
        "diagnosis": result["diagnosis"], "error": result["error"]}
    for role in ("doctor", "simulator"):
        own = [c for c in calls if c["role"] == role]
        # shortcut: providers may omit usage; require reported usage before showing token totals.
        reported = bool(own) and all(c.get("usage_reported", False) for c in own)
        for kind in ("input_tokens", "output_tokens"):
            row[f"{role}_{kind}"] = sum(c[kind] for c in own) if reported else None
    return row


def summarize(results, manifest):
    rows = [case_metrics(r) for r in results]
    requested = len(manifest["case_indices"])
    attempted = len(rows)
    if requested < 1 or not attempted:
        raise ValueError("at least one requested/attempted case is required")
    indices = [r["case_index"] for r in rows]
    if len(set(indices)) != attempted or not set(indices).issubset(manifest["case_indices"]):
        raise ValueError("results contain duplicate or unrequested cases")
    if any(r["mode"] != manifest["mode"] for r in results):
        raise ValueError("result/manifest mode mismatch")
    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None
    attempts = sum(r["action_attempts"] for r in rows)
    model = manifest["mode"] == "model"
    correct = sum(r["exact_match"] is True for r in rows)
    summary = {"mode": manifest["mode"], "requested_cases": requested,
        "attempted_cases": attempted, "completed_cases": sum(r["completed"] for r in rows),
        "attempted_coverage": attempted / requested,
        "completion_rate_all_requested": sum(r["completed"] for r in rows) / requested,
        "exact_match_accuracy_all_attempted": correct / attempted if model else None,
        "exact_match_accuracy_all_requested": correct / requested if model else None,
        "schema_valid_action_rate": ratio(sum(r["schema_valid_actions"] for r in rows), attempts),
        "rejected_action_rate": ratio(sum(r["rejected_actions"] for r in rows), attempts),
        "mean_turns": sum(r["turns"] for r in rows) / attempted,
        "mean_seconds": sum(r["seconds"] for r in rows) / attempted,
        "repeat_requests": sum(r["repeat_requests"] for r in rows),
        "unknown_requests": sum(r["unknown_requests"] for r in rows),
        "unobserved_basis_rejections": sum(r["unobserved_basis_rejections"] for r in rows),
        "official_nova_score": None, "official_agentclinic_score": None,
        "notes": ["Text exact match does not measure clinical appropriateness or recognize synonyms.",
                  "Unattempted and incomplete cases remain in the all-requested denominator.",
                  "Schema checks do not establish semantic single-question/exam compliance.",
                  "Repeated identical requests may be appropriate; review the history."]}
    for role in ("doctor", "simulator"):
        for kind in ("input_tokens", "output_tokens"):
            key = f"{role}_{kind}"
            values = [r[key] for r in rows]
            summary[f"total_{key}"] = sum(values) if all(v is not None for v in values) else None
    return rows, summary


def write_report(output, results, manifest):
    rows, summary = summarize(results, manifest)
    output = Path(output)
    with (output / "cases.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def read_run(output):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    results = [json.loads(line) for line in (output / "results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return manifest, results


def compare_runs(before, after):
    manifests, results = zip(read_run(before), read_run(after))
    controls = ("provenance", "case_indices", "mode", "round", "doctor_model", "simulator_model",
                "max_turns", "max_seconds", "reasoning_effort", "max_tokens")
    for key in controls:
        if manifests[0][key] != manifests[1][key]:
            raise ValueError(f"comparison setting mismatch: {key}")
    reports = [summarize(r, m) for r, m in zip(results, manifests)]
    expected = set(manifests[0]["case_indices"])
    for rows, _ in reports:
        if {r["case_index"] for r in rows} != expected:
            raise ValueError("paired comparison requires every requested case to be attempted in both runs")
    left, right = [{r["case_index"]: r for r in rows} for rows, _ in reports]
    paired = [{"case_index": i, "before_completed": left[i]["completed"],
        "after_completed": right[i]["completed"], "before_exact_match": left[i]["exact_match"],
        "after_exact_match": right[i]["exact_match"], "turns_delta": right[i]["turns"]-left[i]["turns"],
        "rejections_delta": right[i]["rejected_actions"]-left[i]["rejected_actions"],
        "before_diagnosis": left[i]["diagnosis"], "after_diagnosis": right[i]["diagnosis"]}
        for i in manifests[0]["case_indices"]]
    a, b = [summary for _, summary in reports]
    metrics = ("completion_rate_all_requested", "exact_match_accuracy_all_requested",
               "schema_valid_action_rate", "rejected_action_rate", "mean_turns", "mean_seconds")
    return paired, {"before": str(before), "after": str(after),
        "before_code_sha256": manifests[0].get("code_sha256"),
        "after_code_sha256": manifests[1].get("code_sha256"),
        "before_engine": manifests[0].get("engine"), "after_engine": manifests[1].get("engine"),
        "delta_after_minus_before": {k: b[k]-a[k] if a[k] is not None and b[k] is not None else None for k in metrics},
        "note": "Descriptive paired comparison only; inspect failures and repeat model runs before claiming improvement."}
