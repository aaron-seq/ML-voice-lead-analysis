# Roadmap

CallSense today does one thing well: transcript in, explainable qualification
out. This is the path from that to a product a sales org runs day to day.

Each phase is ordered by **value per unit of effort**, and each states what
would have to be true to call it done.

---

## Shipped — Phase 0: Explainable analysis

- Transcript parsing: AWS Transcribe, Whisper, WebVTT, SRT, diarised and plain text
- Speaker-role inference
- Lexicon sentiment with negation, intensifiers, downtoners
- 10 buying-signal types, 7 objection categories with playbooks
- Explainable lead score with evidence, penalties and separate confidence
- Call outcome classification and next-best-action
- Talk metrics, topics, extractive summaries
- REST API, CLI, dashboard, CSV export
- API-key auth with three roles, audit log, rate limiting
- 194 tests; CI running lint, types, tests, Postgres, and an end-to-end smoke

---

## Phase 1 — Make it trustworthy *(next)*

The score is currently unvalidated. Nothing else matters as much as fixing that.

| Item | Why | Done when |
|---|---|---|
| **Calibrate against real outcomes** | The rubric is untested against conversion | A public evaluation on ≥ 200 labelled calls, with AUC and per-band conversion published in `SCORING.md` |
| **Golden-transcript regression suite** | Prevent silent scoring drift | 30 annotated transcripts with expected bands; CI fails on unexplained change |
| **Lexicon coverage report** | Quantify the recall limit honestly | A command reporting what fraction of prospect turns matched nothing |
| **Inter-rater baseline** | Know what "good" is | Two humans label 50 calls; report human–human agreement alongside engine–human |

**Risk:** calibration may show the weights are poorly tuned for some funnels.
That is the point of measuring.

---

## Phase 2 — Fit into a real workflow

Analysis nobody sees changes nothing.

| Item | Notes |
|---|---|
| **CRM sync** | HubSpot and Salesforce first: push score, outcome and next actions onto the contact/opportunity |
| **Audio ingestion** | Accept media, run Whisper, feed the existing pipeline. Keep ASR pluggable |
| **Dialer/meeting integrations** | Zoom, Gong, Aircall, Google Meet transcript webhooks |
| **Async job queue** | Needed once audio (minutes, not milliseconds) is in scope; API returns 202 + status polling |
| **Scheduled digests** | "Your 5 hottest calls and 3 unaddressed objections this week" |

---

## Phase 3 — Team and coaching

The manager's use case, which is where retention lives.

| Item | Notes |
|---|---|
| **Multi-call comparison** | Diff two calls; track one account across a sequence |
| **Rep scorecards** | Talk ratio, discovery-question rate, objection-handling rate over time |
| **Account timeline** | Every touch on one opportunity, with score trajectory |
| **Coaching flags** | "This rep's talk ratio exceeded 70% on 8 of 12 calls" |
| **Annotations** | Humans correct the engine; corrections become evaluation labels — closes the loop with Phase 1 |
| **Saved views and alerts** | "Notify me when a hot lead has no next step" |

---

## Phase 4 — Scale and multi-tenancy

Only worth doing once Phases 1–3 have users.

| Item | Notes |
|---|---|
| **Normalised signals table** | Removes the analytics scaling limit documented in `API.md` |
| **SSO / OIDC** | Replaces static API keys |
| **Row-level tenancy** | Per-team data isolation |
| **Shared rate limiting** | Redis-backed; the current limiter is per-process |
| **Retention policies** | Automatic purge — recorded customer conversations are regulated data |
| **PII redaction** | Detect and mask card numbers, emails, addresses before storage |

---

## Phase 5 — Better models, carefully

Deliberately last. Every item here trades explainability for nuance, and the
current architecture's main asset is that a rep can check the reasoning.

| Item | Trade-off |
|---|---|
| **Embedding-based signal detection** | Catches novel phrasing; evidence becomes "similar to" rather than "matched" |
| **Learned scoring model** | Only defensible once Phase 1 has produced a real dataset. Ship *alongside* the rubric with both scores visible |
| **LLM-generated summaries** | Better prose, real hallucination risk. Would need per-claim grounding against transcript spans |
| **Real speaker diarisation** | Replaces the heuristic; needs audio, so it follows Phase 2 |
| **Multi-language** | Lexicons per language, or a multilingual embedding approach |

---

## Explicitly not planned

- **A hosted SaaS offering.** Self-hosted is the point.
- **Real-time in-call coaching.** A different product with different latency
  constraints.
- **Replacing the CRM.** CallSense should feed one, not become one.

## Contributing to the roadmap

The most valuable contributions right now, in order:

1. **Labelled call data** for Phase 1 calibration (anonymised).
2. **Lexicon coverage** — phrases your team uses that the engine misses.
3. **Transcript format support** — a parser for a dialer we do not handle.

See [CONTRIBUTING.md](../CONTRIBUTING.md).
