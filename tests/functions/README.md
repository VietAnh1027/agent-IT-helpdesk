# Function tests

## Confidence evaluation with manual labels

```powershell
python -m eval.harness.eval_confidence_scorer --limit 3
python -m eval.harness.eval_confidence_scorer --ticket-id TC_SYNC_009
```

Output defaults to a new timestamped JSON in `eval/results`. Fill each
`assessment.human_review` with `verdict` (`acceptable`, `unacceptable`, or
`uncertain`) and `reason`. Leave verdict null until reviewed. The harness does
not run a correctness judge or infer human labels. It evaluates RAG using the
dataset category, not the full production classifier/guardrail routing.

Component scores and weighted contributions are saved for later weight tuning;
this script does not automatically tune weights. Current weights are shared with
`src/confidence.py`. Summary means include only confidence status `ok`.
Existing output files are rejected to preserve manually entered reviews.
`test_eval_confidence_scorer.py` tests scoring, failures, JSON output, selection,
reference isolation, and preservation of existing output.

Run all tests from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Install `requirements-dev.txt`, then enforce 100% branch coverage for the three
refactored modules:

```powershell
.\.venv\Scripts\python.exe -m pytest --cov=src.rag.retriever --cov=src.confidence --cov=src.agent.llm_client --cov-branch --cov-report=term-missing --cov-fail-under=100
```

These tests use mocked database/model responses and do not require API keys,
downloads, a running database, or Ollama.

- `test_confidence.py`: BM25, independent channel candidates, parent deduplication,
  RRF normalization, stable ordering, cosine compatibility, consensus, confidence
  arithmetic, evidence validation, judge fallback, provider selection, context
  budget, settings, and Mistral request/response handling.
- `test_eval_generation.py`: generation/provider compatibility, golden-data
  isolation, partial answers, backend context sources, top-k context selection, summary
  denominators, and writing JSON without losing answers on generation errors.

## Verification on 2026-10-12

25 tests passed. Eval CLI `--help` and `git diff --check` passed.
The previous harness test file has been moved here and updated.

Live checks: Ollama returned ConnectError and PostgreSQL returned OperationalError.
No live generation, Mistral request, ticket baseline comparison, or human review
of judge labels was performed. Unit tests validate contracts and arithmetic;
they do not establish semantic accuracy, latency, or calibrated routing thresholds.

After services are available, run individual ticket comparisons:

```powershell
python -m eval.harness.eval_generation --ticket-id TC_SYNC_002
python -m eval.harness.eval_generation --ticket-id TC_SYNC_009
```

Groundedness uses Mistral when `MISTRAL_API_KEY` is nonblank; otherwise it uses
Ollama. Generation and correctness retain their existing local models.
Inspect the saved claims against the evidence before trusting the score.

## Sources v2 (15/10/2026)

Both reports now declare schema_version=2. Generation contains answer/sources/complete. Sources are parent IDs and titles from exactly the contexts supplied to generation, not evidence that each claim is supported. Groundedness uses short context references (`E1`, `E2`, ...) and Python maps valid references back to backend source metadata. Reports without schema_version are historical v1; do not overwrite manual reviews. Tests cover plain answers, arbitrary bracket text, source ordering/deduplication, invalid metadata, retained sources on partial answers, context-reference validation, and bounded judge output.

Verification for v2 (2026-10-15): 32 tests passed, both eval CLI help commands and git diff --check passed. Live smoke tests blocked: Ollama ConnectError, PostgreSQL OperationalError. Historical verification above describes the earlier implementation.
