# Architecture Decision Records

Short records of decisions that were not obvious and would otherwise be
re-litigated. Each states the decision, why it was taken, what it costs, and
what would make us revisit it.

| ADR | Decision | Status |
|-----|----------|--------|
| [0001](0001-single-domain-contract.md) | One canonical domain contract shared by every layer | Accepted |
| [0002](0002-pure-python-analysis-engine.md) | No heavyweight ML on the default analysis path | Accepted |
| [0003](0003-explainable-rubric-over-learned-model.md) | Explainable weighted rubric instead of a trained classifier | Accepted |
| [0004](0004-sql-plus-blob-storage.md) | SQL for analyses, blob storage for transcripts | Accepted |
| [0005](0005-no-signing-secret.md) | No `SECRET_KEY` until something signs something | Accepted |
