import pandas as pd

from src.models.threshold import find_best_threshold
from src.submission.make_submission import build_submission


def test_find_best_threshold_returns_f1_optimized_cutoff():
    y_true = [0, 0, 1, 1]
    y_proba = [0.1, 0.4, 0.6, 0.9]

    result = find_best_threshold(y_true, y_proba)

    assert set(["threshold", "f1_score"]).issubset(result.keys())
    assert 0.4 <= result["threshold"] <= 0.6
    assert result["f1_score"] == 1.0


def test_build_submission_creates_expected_columns(sample_test_users_df):
    predictions = [1, 0]

    submission = build_submission(sample_test_users_df, predictions)

    assert list(submission.columns) == ["id_user", "is_fraud"]
    assert submission.to_dict(orient="records") == [
        {"id_user": 201, "is_fraud": 1},
        {"id_user": 202, "is_fraud": 0},
    ]
    assert pd.api.types.is_integer_dtype(submission["is_fraud"])


def test_build_submission_rejects_length_mismatch(sample_test_users_df):
    predictions = [1]

    try:
        build_submission(sample_test_users_df, predictions)
    except ValueError as exc:
        assert "same length" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched prediction length")
