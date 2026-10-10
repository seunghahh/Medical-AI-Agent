"""Conservative request/scope checks; no disease labels or invented findings."""
import hashlib
import re
from pathlib import Path


def matching_fingerprint():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def source_facts(source, action):
    """Expose original SAY sentences separately; never generate or summarize facts."""
    facts = {}
    for path, value in source.items():
        parts = [value]
        if action == "SAY" and isinstance(value, str):
            parts, start = [], 0
            for boundary in re.finditer(r"[.!?][\"')\]]*\s+", value):
                end = boundary.start() + len(boundary.group().rstrip())
                # shortcut: common abbreviations only; upgrade if corpus boundary audits find errors.
                if re.search(r"\b(?:Mr|Mrs|Ms|Dr|Prof|St|vs|etc|e\.g|i\.e|[A-Z])\.$", value[start:end]):
                    continue
                parts.append(value[start:end])
                start = boundary.end()
            if value[start:]:
                parts.append(value[start:])
        for index, part in enumerate(parts):
            facts[f"f{len(facts)}"] = {"path": f"{path}.sentence_{index+1}" if len(parts) > 1 else path,
                                      "value": part}
    return facts


def bundled_request(action):
    if action["action"] not in ("EXAM", "TEST"):
        return False
    text = action["text"].casefold()
    if re.search(r"active\s+(?:and|&)\s+passive|능동.*(?:및|와|과).*수동", text):
        return True
    if not re.search(r"\band\b|&|;|\bthen\b|followed by|및|하고|한 뒤|동시에|추가로|[와과]\s", text):
        return False
    # shortcut: high-confidence maneuver words; ambiguous phrasing still needs LLM review.
    maneuvers = (r"inspect|inspection|visual|시진|육안|시각|관찰|황달.{0,6}확인|\bptosis\b|안검하수|눈꺼풀\s*처짐",
                 r"palpat|촉진", r"auscultat|청진", r"percuss|타진",
                 r"speculum|질경", r"bimanual|양손.{0,3}검사",
                 r"range.of.motion|운동.?범위", r"strength|근력",
                 r"reflex|반사", r"gait|보행", r"forward.flexion|전방.{0,3}굴곡",
                 r"\babduction\b|외전", r"external.rotation|외회전", r"internal.rotation|내회전")
    return sum(bool(re.search(pattern, text)) for pattern in maneuvers) > 1


def allowed_source_ids(action, facts):
    """None means no deterministic restriction; an empty set means unrecorded."""
    if action["action"] == "SAY":
        # shortcut: explicit injury questions only; other relevance remains resolver-assessed.
        injury = r"trauma|injur|accident|head.{0,10}(?:hit|strike)|(?:hit|struck).{0,10}head|외상|부상|사고|다쳤|다친|머리.{0,5}부딪"
        if re.search(injury, action["text"], re.I):
            return {i for i, fact in facts.items()
                    if re.search(injury, str(fact["value"]), re.I)}
        return None
    if action["action"] not in ("EXAM", "TEST"):
        return None
    text = action["text"].casefold()
    for name in ("neer", "hawkins", "romberg", "murphy"):
        if re.search(r"\b" + name + r"\b", text):
            return {i for i, fact in facts.items()
                    if name in (fact["path"] + " " + str(fact["value"])).casefold()}
    movements = ((r"forward.flexion|전방.{0,3}굴곡", "Forward_Flexion"),
                 (r"\babduction\b|외전", "Abduction"),
                 (r"external.rotation|외회전", "External_Rotation"),
                 (r"internal.rotation|내회전", "Internal_Rotation"))
    targets = [suffix for pattern, suffix in movements if re.search(pattern, text)]
    structured = {i for i, fact in facts.items() if ".Range_of_Motion." in fact["path"]}
    if targets and structured and not re.search(r"strength|근력", text):
        return {i for i in structured if any(facts[i]["path"].endswith("." + t) for t in targets)}
    if re.search(r"vulv|외음", text) and re.search(r"visual|inspect|육안|시진|관찰", text):
        visible = {i for i, fact in facts.items()
                   if "External_Genitalia" in fact["path"] or
                   re.search(r"(?:Genitourinary|Gynecologic|Pelvic)_Examination\.Inspection$", fact["path"])}
        # shortcut: restrict recognized schemas only; other record names need LLM review.
        return visible if visible else None
    if action["action"] == "EXAM":
        methods = {"inspection": r"inspect|visual|시진|육안|시각",
                   "palpation": r"palpat|촉진",
                   "auscultation": r"auscultat|청진",
                   "percussion": r"percuss|타진|두드"}
        requested = [name for name, pattern in methods.items() if re.search(pattern, text)]
        if len(requested) == 1:
            # shortcut: exclude explicit mismatched method fields; unstructured findings still need LLM review.
            return {i for i, fact in facts.items()
                    if fact["path"].rsplit(".", 1)[-1].casefold() not in methods
                    or fact["path"].rsplit(".", 1)[-1].casefold() == requested[0]}
    return None
