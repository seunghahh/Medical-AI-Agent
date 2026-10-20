# Medical AI Agent

A Python agent for clinical conversations. It gathers patient history, requests focused physical examination findings, and produces a SOAP note with a provisional diagnosis and follow-up plan. Optional local medical references support diagnosis review.

![Clinical conversation interface demo](docs/nova-demo.gif)

## Runtime

The repository root contains the cand23 agent. Its eight runtime files are `run.py`, `workflow.py`, `soap_writer.py`, `grounding.py`, `quantity_text.py`, `nova_sdk.py`, `rag.py`, and `requirements.txt`. The workflow collects observations and coordinates the consultation; the SOAP writer formats and checks the final record against those observations.

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
