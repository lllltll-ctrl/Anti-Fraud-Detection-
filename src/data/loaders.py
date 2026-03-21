from __future__ import annotations

from pathlib import Path

import pandas as pd


USER_TIMESTAMP_COLUMN = "timestamp_reg"
TRANSACTION_TIMESTAMP_COLUMN = "timestamp_tr"


def load_users(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data[USER_TIMESTAMP_COLUMN] = pd.to_datetime(
        data[USER_TIMESTAMP_COLUMN],
        utc=True,
        format="mixed",
    )
    return data


def load_transactions(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data[TRANSACTION_TIMESTAMP_COLUMN] = pd.to_datetime(
        data[TRANSACTION_TIMESTAMP_COLUMN],
        utc=True,
        format="mixed",
    )
    data["amount"] = data["amount"].astype(float)
    return data
