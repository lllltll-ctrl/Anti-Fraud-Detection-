# Modeling Strategy

## Objective Priority

1. maximize validation `f1-score`
2. keep the pipeline reproducible
3. keep the result explainable enough for the written case submission

## Experiment Order

### Stage 1 - Strong Baseline
- merge users with simple transaction aggregates
- train a first working baseline
- tune threshold on validation
- save baseline metrics

Current status:
- completed with executable fallback baseline in `src/models/train.py`
- initial real-data validation was about `0.0728`

### Stage 2 - High-Value Features
- add mismatch features
- add card reuse features
- add error-group ratios
- add temporal burst features
- compare against baseline after each feature block

Current status:
- risk and temporal features were added
- real-data validation improved to about `0.1139` before the latest runtime-heavy training upgrade

### Stage 3 - Model Comparison
- compare CatBoost and LightGBM on the same feature set
- keep the model with the best stable `f1-score`

Current status:
- blocked by missing external ML libraries in the current environment
- if dependency installation becomes available, this should become the highest-priority modeling step

### Stage 4 - Rule Layer
- add only a few post-model rules for extreme-risk cases
- keep rules only if they improve validation `f1-score`

Current status:
- not started yet; defer until the statistical model is stable and runtime is acceptable

## Immediate Next Priorities

1. add a faster validation-only training path
2. reduce runtime of the upgraded fallback learner
3. cache or persist feature tables before full inference
4. move to CatBoost or LightGBM as soon as dependencies can be installed

## What To Avoid

- optimizing for accuracy instead of `f1-score`
- using raw high-cardinality values without aggregation
- adding rules with no measurable validation gain
- relying only on notebook logic for final submission

## Suggested Success Criteria

- reproducible feature build for train and test
- one-command training flow
- one-command submission generation
- clear top-5 feature explanation for the final write-up
- runtime short enough to complete full-data validation and test inference in practice
