# Anti-Fraud Strategy Project

This project is built for the `Case Anti-Fraud 3` competition task.

Primary objective:
- maximize `f1-score` on the hidden test evaluation

Business objective:
- detect risky users without building a brittle one-off solution

Core deliverables:
- `submission.csv` with `id_user,is_fraud`
- written explanation of the approach, top fraud signals, and business integration logic

## Current Status

- executable Python pipeline is implemented in `src/`
- automated test suite currently covers loaders, feature blocks, validation, training, runner logic, and submission formatting
- current implemented feature groups: user, transaction aggregate, risk, and temporal
- current dependency-light baseline can run on real data and previously reached validation `f1-score` around `0.1139` before the latest runtime-heavy model upgrade
- current blocker: the upgraded full-data pipeline times out, so runtime optimization is the immediate next priority

## Main Entry Points

- `src/pipeline.py`: end-to-end load, split, train, validate, predict, and artifact generation
- `src/config.py`: artifact directory management
- `tests/`: regression and contract tests for the pipeline

## Working Rules For Agents

- Read `context.md` before making major changes.
- If any error, failed experiment, or regression happens, write it to `bags.md`.
- If you replace an approach, model, rule set, or implementation choice, write it to `context.md`.
- Prefer changes that improve validation `f1-score` and preserve reproducibility.

## Recommended Repository Layout

```text
.
|-- bags.md
|-- context.md
|-- README.md
|-- docs/
|   |-- project-structure.md
|   |-- technical-design.md
|   |-- modeling-strategy.md
|   `-- delivery-workflow.md
|-- data/
|   |-- raw/
|   `-- processed/
|-- notebooks/
|-- src/
|   |-- config.py
|   |-- pipeline.py
|   |-- data/
|   |-- features/
|   |-- models/
|   |-- evaluation/
|   |-- explain/
|   `-- submission/
|-- artifacts/
|   |-- features/
|   |-- models/
|   |-- reports/
|   `-- submissions/
|-- tests/
|-- train_users.csv
|-- train_transactions.csv
|-- test_users.csv
`-- test_transactions.csv
```

Documentation lives in `docs/`.
