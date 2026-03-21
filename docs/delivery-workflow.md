# Delivery Workflow

## Session Start Checklist For Any Agent

1. Read `context.md`
2. Read `bags.md`
3. Check current docs in `docs/`
4. Continue from the latest active direction instead of restarting old ideas

## During Work

- if an error occurs, append it to `bags.md`
- if a previous approach is replaced, append the decision to `context.md`
- if a metric changes, note the before/after values in the relevant file
- keep all important paths, assumptions, and model choices documented
- if a full run times out, document the timeout, avoid blind reruns, and create a faster benchmark path first

## Before Ending A Session

- update `context.md` with current active approach
- update `bags.md` with unresolved blockers or failed attempts
- make sure the next agent can understand what is done, what failed, and what to try next

## Final Competition Deliverables

- prediction file with columns `id_user,is_fraud`
- concise written explanation covering:
  - selected approach
  - top 5 features or rules
  - why those features matter
  - how the solution can be integrated into business operations

## Recommended Additional Documents

- validation notes with best metric and threshold
- experiment log for major model runs
- final summary for Notion submission

## Current Operational Notes

- use `src/pipeline.py` as the default executable path
- check `artifacts/reports/baseline_metrics.json` after each successful run
- treat `artifacts/models/baseline_model.json` as the latest saved training artifact
- if the upgraded model is too slow on full data, benchmark on validation-first runs before generating a new submission
