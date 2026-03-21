from pathlib import Path

import pandas as pd

from src.feature_cache import load_or_build_feature_table


def test_load_or_build_feature_table_writes_processed_csv(tmp_path: Path, sample_users_df, sample_transactions_df):
    users_path = tmp_path / "train_users.csv"
    transactions_path = tmp_path / "train_transactions.csv"
    cache_path = tmp_path / "cached_features.csv"

    sample_users_df.to_csv(users_path, index=False)
    sample_transactions_df.to_csv(transactions_path, index=False)

    table = load_or_build_feature_table(
        users_path=users_path,
        transactions_path=transactions_path,
        cache_path=cache_path,
        is_train=True,
    )

    assert cache_path.exists()
    assert len(table) == len(sample_users_df)
    assert "is_fraud" in table.columns


def test_load_or_build_feature_table_reuses_existing_cache(tmp_path: Path, sample_users_df, sample_transactions_df):
    users_path = tmp_path / "train_users.csv"
    transactions_path = tmp_path / "train_transactions.csv"
    cache_path = tmp_path / "cached_features.csv"

    sample_users_df.to_csv(users_path, index=False)
    sample_transactions_df.to_csv(transactions_path, index=False)

    original = load_or_build_feature_table(
        users_path=users_path,
        transactions_path=transactions_path,
        cache_path=cache_path,
        is_train=True,
    )
    original_id_users = original["id_user"].tolist()

    mutated_users = sample_users_df.copy()
    mutated_users.loc[:, "id_user"] = [999, 998, 997]
    mutated_users.to_csv(users_path, index=False)

    reused = load_or_build_feature_table(
        users_path=users_path,
        transactions_path=transactions_path,
        cache_path=cache_path,
        is_train=True,
    )

    assert reused["id_user"].tolist() == original_id_users
    assert pd.read_csv(cache_path)["id_user"].tolist() == original_id_users
