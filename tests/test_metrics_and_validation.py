import pandas as pd

from src.evaluation.metrics import binary_f1_score, precision_recall_summary
from src.evaluation.validation import stratified_split_users


def test_binary_f1_score_returns_expected_value():
    score = binary_f1_score([0, 0, 1, 1], [0, 1, 1, 1])

    assert round(score, 4) == 0.8


def test_precision_recall_summary_returns_expected_counts():
    summary = precision_recall_summary([0, 0, 1, 1], [0, 1, 1, 1])

    assert summary == {
        "true_positive": 2,
        "false_positive": 1,
        "false_negative": 0,
        "precision": 2 / 3,
        "recall": 1.0,
        "f1_score": 0.8,
    }


def test_stratified_split_users_preserves_target_and_has_no_overlap():
    users = pd.DataFrame(
        {
            "id_user": list(range(20)),
            "is_fraud": [0] * 16 + [1] * 4,
        }
    )

    train_df, valid_df = stratified_split_users(users, target_column="is_fraud", valid_size=0.25, random_state=7)

    assert len(train_df) == 15
    assert len(valid_df) == 5
    assert set(train_df["id_user"]).isdisjoint(set(valid_df["id_user"]))
    assert valid_df["is_fraud"].sum() == 1
