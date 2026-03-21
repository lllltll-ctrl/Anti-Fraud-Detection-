import pandas as pd

from src.data.loaders import load_transactions, load_users


def test_load_users_parses_expected_columns(sample_users_df, temp_csv_dir):
    path = temp_csv_dir / "users.csv"
    sample_users_df.to_csv(path, index=False)

    loaded = load_users(path)

    assert list(loaded.columns) == list(sample_users_df.columns)
    assert pd.api.types.is_datetime64tz_dtype(loaded["timestamp_reg"])
    assert loaded["id_user"].tolist() == [101, 102, 103]
    assert loaded["is_fraud"].tolist() == [0, 1, 0]


def test_load_transactions_parses_timestamp_and_amount(sample_transactions_df, temp_csv_dir):
    path = temp_csv_dir / "transactions.csv"
    sample_transactions_df.to_csv(path, index=False)

    loaded = load_transactions(path)

    assert pd.api.types.is_datetime64tz_dtype(loaded["timestamp_tr"])
    assert loaded["amount"].sum() == 51.0
    assert loaded["id_user"].value_counts().to_dict()[101] == 2


def test_load_users_supports_mixed_iso_timestamp_formats(temp_csv_dir):
    path = temp_csv_dir / "mixed_users.csv"
    pd.DataFrame(
        {
            "id_user": [1, 2],
            "timestamp_reg": [
                "2025-07-16 19:01:23.868869+00:00",
                "2025-09-07 18:24:51+00:00",
            ],
            "email": ["a@gmail.com", "b@gmail.com"],
            "gender": ["female", "male"],
            "reg_country": ["Portugal", "Austria"],
            "traffic_type": ["organic", "ppc"],
            "is_fraud": [0, 1],
        }
    ).to_csv(path, index=False)

    loaded = load_users(path)

    assert pd.api.types.is_datetime64tz_dtype(loaded["timestamp_reg"])
    assert loaded["timestamp_reg"].isna().sum() == 0
