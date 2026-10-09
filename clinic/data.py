import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "vendor/AgentClinic/agentclinic_medqa_extended.jsonl"


def flatten(value, prefix=""):
    if isinstance(value, dict):
        return {p: v for key, child in value.items()
                for p, v in flatten(child, f"{prefix}.{key}" if prefix else key).items()}
    # Lists are kept as one source item, preserving the original text exactly.
    return {prefix: value}


@dataclass
class Case:
    index: int
    opening: dict
    sources: dict
    gold: str


def load_cases(path=DATA):
    cases = []
    for index, line in enumerate(Path(path).read_text().splitlines()):
        if not line.strip():
            continue
        exam = json.loads(line)["OSCE_Examination"]
        actor = exam["Patient_Actor"]
        physical = exam["Physical_Examination_Findings"]
        symptoms = actor.get("Symptoms", {})
        chief = symptoms.get("Primary_Symptom") if isinstance(symptoms, dict) else symptoms
        # Do not use Objective_for_Doctor: it can expose more than an opening utterance.
        opening = {"demographics": actor.get("Demographics", "Not provided"),
                   "chief_complaint": chief or "I am here for a medical consultation.",
                   "vital_signs": physical.get("Vital_Signs", {}) if isinstance(physical, dict) else {}}
        cases.append(Case(index, opening, {
            "SAY": flatten(actor), "EXAM": flatten(physical),
            "TEST": flatten(exam["Test_Results"])}, str(exam["Correct_Diagnosis"])))
    return cases


def provenance(path=DATA):
    return {"dataset": Path(path).name, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "upstream_commit": "b6fbe22300e99a267a7ac94eaa465ab552eef741"
                if Path(path).resolve() == DATA.resolve() else None}
