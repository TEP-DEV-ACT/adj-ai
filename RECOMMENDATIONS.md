# adj-ai — Top 3 Recommendations

A code review of the package (source, tests, packaging, CI). The package is cleanly
set up (src-layout, `pyproject.toml`, MIT license, GitHub Actions for PR tests + PyPI
release) and the public API is small and sensible (`Orchestrator(...).run_task()`).
The weaknesses cluster in three areas, listed in priority order below.

> Note: an automated scan flagged a committed `.env` with a live key. Verified false —
> `.env` and `dist/` are both in `.gitignore` and untracked. No action needed there.

Primary file: `src/adj_ai/orchestrator.py` (~365 lines, monolithic). Tests in `tests/`.

---

## Recommendation 1 — Add a mocked unit-test layer + coverage (highest value)

**Problem.** Every meaningful test calls real provider APIs (`tests/conftest.py`
session-scoped fixtures build live clients). Consequences: tests are skipped entirely
without API keys (only `test_no_client_raises_error` always runs), they are slow/flaky,
non-deterministic (judge ranking + regex parsing of LLM output), and none of the 13
private methods or ~18 error paths are exercised. Estimated coverage ~15–20%.

**Change.**
- Split tests into two tiers:
  - `tests/unit/` — fast, mocked, always run in CI. Mock the SDK clients with
    `unittest.mock.MagicMock` (stub `anthropic_client.messages.create(...)` and
    `models.list()` to return canned objects). Add `pytest-mock` to the `dev` extra.
  - Keep existing live tests as `tests/integration/`, tagged with the already-declared
    `integration` marker (currently defined in `pyproject.toml` but never applied).
- Cover with unit tests: model-selection filtering + fallback, candidate generation
  count (verify `candidate_number` controls list length), judge ranking parse, and each
  error path (no models, no haiku/mini match, empty judged list, `OrchestratorError`).
- Add input-validation tests (see Rec 3).
- Add `pytest-cov`; run unit tests on every PR, gate at a starting threshold (e.g. 70%).
  Run integration tests only on a schedule or manual dispatch so PRs don't depend on
  external APIs / secrets.

**Files.** `tests/conftest.py`, new `tests/unit/*`, move `tests/test_provider_*.py` +
`test_multi_provider.py` into `tests/integration/`, `pyproject.toml`
(`[tool.pytest.ini_options]`, dev deps), `.github/workflows/pr-tests.yml`.

---

## Recommendation 2 — Collapse the 3× provider duplication and fix the bugs it hides

**Problem.** Model selection, candidate generation, and judging are each implemented
three nearly-identical times (~30–40 duplicated lines per provider). The duplication
both inflates the file and conceals real defects:

- **Dead guard (confirmed, `orchestrator.py:129-132`):** the `elif not (chosen_...)`
  binds only to `if self.deepseek_client:`. If a DeepSeek client is present the guard
  never runs; the "no model selected" case is effectively unreachable. It also raises
  `NotImplementedError` rather than `OrchestratorError`.
- **Single-candidate path skips judging (confirmed, `orchestrator.py:57-60`):** when
  `candidate_number == 1` but multiple clients are supplied, every client still
  generates a candidate, then `candidates[0]` is returned with no judging — the result
  is just "whichever provider ran first." Behavior should be defined and intentional.
- **Fragile ranking parse (`orchestrator.py:264, 310, 354`):** `re.findall(r"\d+", text)`
  mis-handles multi-digit indices (`candidate_number >= 10`) and silently yields
  wrong/empty rankings on prose output.
- Magic strings (`'haiku'`, `'mini'`, `'sonnet'`, …) and hardcoded `max_tokens`
  scattered throughout.

**Change.** Introduce a small provider abstraction — a `Provider` protocol/base with
`select_model()`, `generate(prompt)`, `judge(candidates, judge_prompt)` — and
`AnthropicProvider`, `OpenAIProvider`, `DeepSeekProvider` implementations. `Orchestrator`
iterates over the configured providers instead of branching per-client. As part of the
refactor: fix the model-selection guard, define the single-candidate semantics
explicitly, harden ranking parsing (parse explicit indices / validate against candidate
count, fall back deterministically), and lift filter keywords + token limits into named
constants/config.

**Files.** `src/adj_ai/orchestrator.py` (split into a `providers.py` module),
`src/adj_ai/exceptions.py`. Covered by the new unit tests from Rec 1.

---

## Recommendation 3 — Robustness: validation, response checks, logging, and lint tooling

**Problem.** No constructor validation (`candidate_number`, `max_task_tokens`, non-empty
prompts all unchecked), no validation of API response shape before indexing
(`response.content[0].text` / `response.choices[0].message.content` raise opaque
`IndexError`s), zero logging (can't see which model was selected or which candidate won),
and no automated formatting/linting (no ruff/black/mypy/pre-commit).

**Change.**
- Add a `__post_init__` validating `candidate_number >= 1`, `max_task_tokens >= 1`,
  non-empty prompts, and at least one client — raising `OrchestratorError` with clear
  messages.
- Guard response access: check content exists before indexing; raise `OrchestratorError`
  with provider context on malformed/empty responses.
- Add `logging` (module-level logger) at INFO for model selection + winner, DEBUG for
  candidate counts, WARNING on silent fallbacks.
- (Optional but recommended) consider returning a small result object (winner + chosen
  models + all candidates) instead of a bare `str`, and document that `run_task()` can
  currently return `None`.
- Add `ruff` (lint + format) and `mypy` to the `dev` extra and a `ruff`/`mypy` step in
  `pr-tests.yml`; optionally a `.pre-commit-config.yaml`. This also surfaces the
  unused-`i`-parameter and missing-return-type findings automatically.

**Files.** `src/adj_ai/orchestrator.py`, `src/adj_ai/exceptions.py`, `pyproject.toml`
(dev deps + `[tool.ruff]`/`[tool.mypy]`), `.github/workflows/pr-tests.yml`, optional
`.pre-commit-config.yaml`.

---

## Verification

- `pip install -e ".[dev]"`
- `pytest tests/unit` — fast, no keys needed; should pass and report coverage
  (`pytest --cov=adj_ai`).
- `pytest -m integration` locally with keys set (and on the scheduled CI job) to confirm
  live behavior unchanged.
- `ruff check src tests` and `mypy src` clean.
- Manual smoke: run `examples/example.py` with at least one provider key to confirm
  `run_task()` still returns a sensible answer end-to-end.

Suggested order: Rec 1 first (safety net for the Rec 2 refactor), then Rec 2, then Rec 3.
