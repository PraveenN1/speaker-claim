# Speaker Claim Evaluator

This service evaluates factual product claims in long AI answers using live model tool calls and the supplied speaker catalogue.

## Setup

1. Create a virtual environment and install `pip install -e ".[dev]"`.
2. Copy `.env.example` to `.env` and set `DATABASE_URL`, `GEMINI_API_KEY`, and `GEMINI_MODEL`. The application loads this local file at startup; it stays excluded from Git.
3. Start PostgreSQL, then run `uvicorn app.main:app --reload`.
4. Run `pytest` for deterministic checks.

The CSVs are deliberately read with Python's CSV parser using UTF-8 and `newline=""`. Do not edit or re-save `answers.csv`; its text is the offset authority.

## Current architecture

- `app/api/routes/`: thin FastAPI transport endpoints.
- `app/services/`: classification and evaluation workflows.
- `app/repositories/`: SQLAlchemy persistence adapters.
- `app/models/`: API/domain model exports.
- `app/core/`: environment configuration and app composition.

## Evaluation workflow

The application evaluates two frozen agent versions using identical data and scoring rules.

1. Load the supplied CSVs with a proper UTF-8 CSV parser. Preserve every `ai_answer` character because source offsets are part of the contract.
2. Run `v1` on the 40-answer development split. For every answer, the agent uses the controlled `get_brand_facts` tool, extracts complete factual sentences, and returns validated exact spans, labels, evidence fact IDs, and reasons.
3. Save all accepted predictions and terminal answer errors. The development scorer then performs exact span matching against the gold file and reports extraction metrics, joint metrics, per-label/macro F1, confusion matrix, missed claims, extra claims, wrong labels, and errors.
4. Inspect the baseline evidence, make one documented behavior change, and run `v2` on the same development inputs with the same provider, model, data, and evaluator settings. Compare improvements and regressions.
5. Freeze `v1` and `v2`, then run both on the 10-answer check split. Save predictions, errors, timing, and settings, but leave all quality metrics null because check labels are private.
6. Preserve four live-run reports: `v1` development, `v2` development, `v1` check, and `v2` check. Include dataset hashes, sanitized model/tool exchanges, test output, and trace evidence.

Gold labels, spans, explanations, difficulty tags, and scores are never provided to the model or tool path. They are used only by the deterministic scorer after a development prediction has been saved.

The Gemini provider uses Google's OpenAI-compatible endpoint for function calling. Before submission, run the four required live evaluations (v1/v2 × development/check) and commit only secret-free report artifacts and CSV hashes.
