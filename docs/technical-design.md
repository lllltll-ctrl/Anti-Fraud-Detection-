# Technical Design

## Problem Framing

We predict one label per user, not per transaction. The target `is_fraud` exists in `train_users.csv`, while transactions provide behavioral evidence.

## Data Sources

- `train_users.csv`: user-level attributes and target label
- `train_transactions.csv`: historical transaction behavior for train users
- `test_users.csv`: user-level test set without target
- `test_transactions.csv`: historical transaction behavior for test users

## Modeling Strategy

Best initial path for score maximization:
- build a user-level feature table
- aggregate all transaction history per user
- train a strong tabular model
- tune threshold for best validation `f1-score`

Preferred model order:
1. CatBoost baseline
2. LightGBM comparison
3. optional hybrid with post-model business rules only if validation improves

Current implemented fallback:
- because `catboost`, `lightgbm`, and `sklearn` are not available in the current environment, the repository currently uses a dependency-light baseline implemented with `numpy` and `pandas`
- this baseline uses target-mean encoding for categorical features, normalized numeric features, a simple logistic-style learner, and `f1`-driven threshold selection
- this is an execution bridge, not the intended final competition model

## Feature Groups

### User Profile Features
- gender
- registration country
- traffic source
- email domain
- registration timestamp features such as hour, day, and weekday

### Transaction Volume Features
- total transaction count
- success count and fail count
- fail ratio
- amount sum, mean, max, std

### Card Features
- unique card count
- card reuse ratio
- users per card hash
- brand and type distributions

### Geography Features
- registration country vs card country match
- registration country vs payment country match
- unique countries used
- mismatch counts and ratios

### Error And Risk Features
- counts of `antifraud`, `fraud`, `do not honor`, `insufficient funds error`
- ratios of failed `card_init`, `card_recurring`, and `google-pay`

### Temporal Features
- time from registration to first transaction
- time from registration to first successful transaction
- average and median time between transactions
- burst activity metrics per hour or day

Implemented temporal features currently include:
- `minutes_to_first_tx`
- `minutes_to_first_success`
- `tx_span_minutes`
- `mean_gap_minutes`
- `tx_in_first_24h`
- `fail_in_first_24h`
- `max_tx_per_day`

## Validation Strategy

- use user-level validation only
- start with stratified split because the target is imbalanced
- track `f1-score` as the main metric
- later compare with cross-validation if time permits
- avoid leakage from any future-derived features or validation contamination

Current runner behavior:
- uses user-level stratified train/validation split
- writes model artifact, validation metrics, and submission file through `src/pipeline.py`
- has already exposed a runtime limitation on the upgraded baseline when run end-to-end on full competition data

## Current Technical Risks

- full-data runtime is too slow after the model upgrade
- the current fallback learner is useful for iteration, but is still weaker than the desired boosted-tree model family
- feature generation is repeated on every full run, so caching or staged execution will likely be needed

## Explainability Requirements

The final documentation must clearly explain:
- top 5 features or rules
- why they matter for fraud detection
- what trade-offs they introduce

Use feature importance and sample-based interpretation to support the explanation.

## Production-Like Decision Policy

For business integration, expose a risk score and optionally map it into three bands:
- low risk: approve
- medium risk: manual review
- high risk: block

This is usually stronger than a single hard block threshold in a fraud case presentation.
