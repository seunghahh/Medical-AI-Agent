# Medical AI Agent

A Python agent for clinical conversations. It gathers patient history, requests focused physical examination findings, and produces a SOAP note with a provisional diagnosis and follow-up plan. Optional local medical references support diagnosis review.

![Clinical conversation interface demo](docs/nova-demo.gif)

## Runtime

The repository root contains the original cand103b agent (`lb33`), copied without runtime changes from `hsh_test 1022_cand103b_lb33.zip`.

Submission ZIP SHA-256:

```text
b72f588d33a18b78c0162ab89c1c14f7f743a4433cd712dbfa96ffd0d4c224b1
```

The eight runtime files are:

| File | Purpose |
| --- | --- |
| `run.py` | Consultation flow, model requests, history and examination selection, diagnosis review, plans, patient explanations, and final SOAP formatting through `write_record()` |
| `grounding.py` | Observation sources and validation of generated record content |
| `record_sections.py` | Routing observations into SOAP sections and linking planned warning advice to follow-up |
| `care_references.py` | Selecting local guideline passages to support care planning |
| `rag.py` | Local guideline indexing, retrieval, and source metadata |
| `quantity_text.py` | Normalization of Korean quantities and units |
| `nova_sdk.py` | NOVA session API client and agent entry point |
| `requirements.txt` | Dependency declaration; this release uses the Python standard library |

SOAP formatting is part of `run.py`; this release does not require `workflow.py` or `soap_writer.py`. The consultation targets 33 interaction turns, with room for patient responses within the session's 50-turn limit. This is the submitted implementation, not a revised version of its planning, validation, or closing behavior.

Experiments, saved encounters, tests, reference cases, provider presets, and development self-grading tools are kept outside the tracked runtime. The demo shows a separate local interface, not part of the submitted agent.

## Run

Use Python 3.12 or newer. The agent uses the Python standard library.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Connect to a running NOVA-compatible clinical session service that provides patient conversations, examination findings, and model responses. Set its endpoint and session token, then start the agent:

```bash
export NOVA_API_BASE="http://127.0.0.1:8000"
export NOVA_RUN_TOKEN="<session-token>"
python run.py
```

Replace the endpoint and token with values supplied by your service. This repository contains the agent; the session service and the interface shown above run separately.

To use local medical references, set `NOVA_GUIDELINES_DIR` to their directory before starting. To disable reference retrieval, set `NOVA_RAG=0`.
