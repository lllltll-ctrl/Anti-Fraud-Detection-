from __future__ import annotations

from typing import Iterable

import pandas as pd


def build_submission(users: pd.DataFrame, predictions: Iterable[int]) -> pd.DataFrame:
    prediction_list = [int(prediction) for prediction in predictions]
    if len(users) != len(prediction_list):
        raise ValueError("Users and predictions must have the same length")

    submission = pd.DataFrame(
        {
            "id_user": users["id_user"].astype("int64"),
            "is_fraud": pd.Series(prediction_list, dtype="int64"),
        }
    )
    return submission
