# Project Structure

## Goal

Build a fraud-detection pipeline that predicts `is_fraud` at the user level and is optimized for `f1-score`.

## Directory Plan

### `data/`
- `raw/`: untouched source files if we later move CSVs into a dedicated folder
- `processed/`: cached user-level feature tables and cleaned datasets

### `notebooks/`
- EDA, feature ideation, and one-off experiments
- not the source of truth for final training or inference

### `src/`
- production-style code used for reproducible runs

Recommended modules:
- `src/config.py`: shared paths, seeds, target name, thresholds, and model presets
- `src/pipeline.py`: end-to-end training, validation, artifact writing, and submission generation
- `src/data/loaders.py`: typed CSV readers and dataset joins
- `src/features/user_features.py`: features from `train_users.csv` and `test_users.csv`
- `src/features/transaction_features.py`: aggregate transaction features per user
- `src/features/temporal_features.py`: time-gap and burst features
- `src/features/risk_features.py`: mismatch and high-risk behavior features
- `src/features/build_dataset.py`: final train/test feature matrix builder
- `src/models/train.py`: train loop and experiment runner
- `src/models/predict.py`: test inference
- `src/models/threshold.py`: threshold selection for best `f1-score`
- `src/evaluation/metrics.py`: evaluation metrics
- `src/evaluation/validation.py`: split strategy and cross-validation logic
- `src/explain/feature_importance.py`: top signals and explainability outputs
- `src/submission/make_submission.py`: final competition CSV creation

### `artifacts/`
- stores generated assets that should not be hand-edited
- examples: fitted models, feature importance exports, validation reports, submission files

Current generated files:
- `artifacts/models/baseline_model.json`
- `artifacts/reports/baseline_metrics.json`
- `artifacts/submissions/submission.csv`

### `tests/`
- unit checks for feature builders, data joins, and submission formatting
- integration checks for config, train/predict pipeline, and artifact generation

Current implemented `tests/` coverage:
- loaders and timestamp parsing
- user, transaction, risk, and temporal features
- dataset assembly
- metrics and stratified validation
- threshold selection
- train/predict artifact contracts
- config and end-to-end runner behavior

### `docs/`
- human-readable project documentation for current and future sessions

## Development Principle

Use notebooks for discovery and `src/` for repeatable implementation. If a notebook experiment becomes useful, move it into `src/` and record the change in `context.md`.
