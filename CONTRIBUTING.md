# Contributing

## Setup

```bash
git clone https://github.com/aaron-seq/ML-voice-lead-analysis.git
cd ML-voice-lead-analysis/backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

pytest                                  # 298 tests, ~10s
pytest tests/test_benchmarks.py -s      # performance figures
ruff check app tests                    # lint
mypy app                                # types
bandit -q -r app                        # security static analysis
pip-audit -r requirements.txt --strict  # dependency advisories
```

### Running against PostgreSQL

Tests use a per-test SQLite file by default. Set `TEST_DATABASE_URL` to run the
integration suite against the database production uses instead:

```bash
pip install -r requirements-postgres.txt
TEST_DATABASE_URL=postgresql+asyncpg://callsense:callsense@localhost:5432/callsense_test \
  pytest tests/integration
```

A shared server has no per-test file isolation, so every table is dropped and
recreated around each test. `TestDatabaseUnderTest` asserts the live engine is
really speaking the dialect you asked for — the variable was previously
exported by CI and read by nothing, so that job ran SQLite twice while
reporting that it covered PostgreSQL.

Frontend:

```bash
cd frontend && npm install
npm run lint && npm run type-check && npm run build
npm audit --audit-level=high
```

Note that `next lint` no longer exists (removed in Next 16); `npm run lint` runs
`eslint .` against `eslint.config.mjs`.

## The highest-value contribution: lexicon coverage

CallSense favours precision over recall, which means it misses phrasings it has
never seen. If your team says something the engine does not detect, that is a
bug worth reporting and an easy fix.

1. Confirm the miss:

   ```bash
   python -m app.cli.main analyse your-call.txt
   ```

   A component sitting at `0.0` with a phrase in the transcript that should have
   fired is the signature.
2. Add the phrase to `backend/app/analysis/lexicons.py`:

   ```python
   SignalType.BUDGET: [
       ("we have budget", 0.95),
       ("your new phrase", 0.80),   # weight = how diagnostic, not how loud
   ],
   ```
3. Add a test in `tests/unit/test_analysis.py`.
4. Run `pytest`. `TestLexiconHygiene` will fail if the phrase already exists
   under another signal type.

Weight guidance:

| Range      | Meaning                                              |
| ---------- | ---------------------------------------------------- |
| 0.90–0.95 | Near-certain ("budget is approved", "I can approve") |
| 0.70–0.85 | Strong ("we need", "next steps")                     |
| 0.50–0.65 | Suggestive ("pricing", "timeline")                   |
| 0.35–0.50 | Weak context only                                    |

## Code standards

- **Domain and analysis code is strictly typed.** `mypy` enforces
  `disallow_untyped_defs` on `app/analysis/*` and `app/domain/*`.
- **The engine performs no I/O.** Anything in `app/analysis/` must be a pure
  function of its inputs. This is what keeps it testable and deterministic.
- **Every finding carries evidence.** A new detector must populate `Evidence`
  with the transcript span that produced it.
- **Comment the *why*.** The codebase explains non-obvious decisions and
  trade-offs, not what the next line does.
- `ruff format` is intentionally not enforced — `lexicons.py` is a hand-formatted
  data table and auto-formatting makes it much harder to review.
- **Every setting must do something.** A config field nothing reads is
  fabricated functionality; `SECRET_KEY`, `SENTRY_DSN` and `ANALYSIS_TIMEOUT_S`
  were removed for exactly that reason (see
  [ADR 0005](docs/adr/0005-no-signing-secret.md)). If you add one, add the code
  that consumes it in the same change.

## Testing standards

Tests exercise the real engine, a real database and real storage. **Do not mock
the service layer.** The previous suite mocked everything and passed while the
product was broken end to end.

- New analyzer behaviour → a test in `tests/unit/test_analysis.py` stating the
  behaviour in its name.
- New endpoint → a test in `tests/integration/test_api.py`.
- New field the frontend reads → add it to `UI_CONTRACT` in
  `tests/integration/test_contract.py`.

### Changes that move scores

`tests/golden/` pins the band, outcome and score of every transcript in
`examples/transcripts/` plus targeted cases in `tests/golden/cases/`. The rest
of the suite checks *properties*; this checks *values*, so a weight tweak that
moves every call by eight points fails here instead of shipping unnoticed.

When a change is **meant** to move scores:

1. Bump `ENGINE_VERSION` in `backend/app/domain/models.py`, with a comment
   saying what changed.
2. Run `pytest tests/golden --update-golden`.
3. Commit the regenerated `tests/golden/expected.json` **in the same commit**
   as the change. The fixture diff then shows a reviewer exactly which calls
   moved, and by how much.

The update refuses to record moved scores if `ENGINE_VERSION` was not bumped.
Stored analyses carry the version that produced them, and a scoring change
under an unchanged version makes old and new results impossible to tell apart.

Adding a case needs no bump: drop a transcript into `tests/golden/cases/` and
run the update. Prefer a case that pins one specific behaviour, and say which
in its filename.

## Adding a transcript format

1. Write a parser in `app/analysis/parsing.py` returning a `Transcript`.
2. Dispatch to it from `parse_transcript` (detect from **content**, not the file
   extension — `.json` covers at least four incompatible shapes).
3. Add a round-trip test in `tests/unit/test_parsing.py`.

## Architecture decisions

Non-obvious decisions live in [docs/adr/](docs/adr/). If a change reverses one,
update the ADR's *Revisit if* section rather than silently contradicting it. If a
change makes a new non-obvious decision, add an ADR — five short sections, not a
design document.

## Pull requests

- Conventional commit messages (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`).
- CI must be green: lint, types, tests, Postgres integration, frontend build,
  end-to-end smoke.
- If a change alters scores, bump `ENGINE_VERSION` in `app/domain/models.py` and
  say so in the PR — stored analyses record the version that produced them.
- Be honest in documentation. If something is partial, say it is partial.
