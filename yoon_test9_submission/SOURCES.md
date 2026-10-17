# Source notes

Runtime files in this package are taken from the repository's `origin/hsh-test` branch at build time.

- `run.py`: NOVA agent; adaptive follow-up questioning, observation-based examination selection, grounded SOAP finalization, patient-facing explanations and responses.
- `grounding.py`: evidence-bound observations, record construction and generated-plan checks.
- `rag.py`: offline SQLite FTS5 retrieval for the mounted guideline corpus.
- `nova_sdk.py`: official-compatible NOVA client and submission helpers.
- `requirements.txt`: standard-library-only runtime; no third-party dependencies.

Public evaluation references used to guide this package:
- https://nova.snubhai.org/evaluation/
- https://nova.snubhai.org/rules/
- https://nova.snubhai.org/docs/

The `self_grade.py` developer-side rubric tool and all sample/evaluator data are intentionally excluded from the runtime ZIP.
