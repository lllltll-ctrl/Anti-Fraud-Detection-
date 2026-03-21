from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.loaders import load_transactions, load_users
from src.features.build_dataset import build_model_dataset


def load_or_build_feature_table(
    users_path: str | Path,
    transactions_path: str | Path,
    cache_path: str | Path,
    is_train: bool,
) -> pd.DataFrame:
    cache_path = Path(cache_path)
    if cache_path.exists():
        return pd.read_csv(cache_path)

    users = load_users(users_path)
    transactions = load_transactions(transactions_path)
    feature_table = build_model_dataset(users, transactions, is_train=is_train)
    feature_table.to_csv(cache_path, index=False)
    return feature_table
