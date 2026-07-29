# Contributing

## Setup

```bash
git clone https://github.com/aaron-seq/ML-voice-lead-analysis.git
cd ML-voice-lead-analysis/backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

pytest                 # 159 tests, ~10s
ruff check app tests
mypy app
```

Frontend:

```bash
cd frontend && npm install
npm run type-check && npm run build
```

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

| Range | Meaning |
|---|---|
| 0.90–0.95 | Near-certain ("budget is approved", "I can approve") |
| 0.70–0.85 | Strong ("we need", "next steps") |
| 0.50–0.65 | Suggestive ("pricing", "timeline") |
| 0.35–0.50 | Weak context only |

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

## Testing standards

Tests exercise the real engine, a real database and real storage. **Do not mock
the service layer.** The previous suite mocked everything and passed while the
product was broken end to end.

- New analyzer behaviour → a test in `tests/unit/test_analysis.py` stating the
  behaviour in its name.
- New endpoint → a test in `tests/integration/test_api.py`.
- New field the frontend reads → add it to `UI_CONTRACT` in
  `tests/integration/test_contract.py`.

## Adding a transcript format

1. Write a parser in `app/analysis/parsing.py` returning a `Transcript`.
2. Dispatch to it from `parse_transcript` (detect from **content**, not the file
   extension — `.json` covers at least four incompatible shapes).
3. Add a round-trip test in `tests/unit/test_parsing.py`.

## Pull requests

- Conventional commit messages (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`).
- CI must be green: lint, types, tests, Postgres integration, frontend build,
  end-to-end smoke.
- If a change alters scores, bump `ENGINE_VERSION` in `app/domain/models.py` and
  say so in the PR — stored analyses record the version that produced them.
- Be honest in documentation. If something is partial, say it is partial.
