import pandas as pd

from src.features.risk_features import build_risk_features


def test_build_risk_features_creates_country_mismatch_and_shared_card_metrics(sample_users_df):
    transactions = pd.DataFrame(
        {
            "id_user": [101, 101, 102, 103],
            "timestamp_tr": [
                "2025-01-01 11:00:00+00:00",
                "2025-01-01 12:00:00+00:00",
                "2025-01-02 16:00:00+00:00",
                "2025-01-03 23:00:00+00:00",
            ],
            "amount": [10.0, 20.0, 5.0, 7.5],
            "status": ["success", "fail", "fail", "success"],
            "transaction_type": ["card_init", "google-pay", "card_init", "card_recurring"],
            "error_group": ["", "antifraud", "fraud", ""],
            "currency": ["EUR", "EUR", "USD", "EUR"],
            "card_brand": ["VISA", "VISA", "MASTERCARD", "VISA"],
            "card_type": ["DEBIT", "DEBIT", "DEBIT", "CREDIT"],
            "card_country": ["Portugal", "Spain", "Brazil", "Germany"],
            "card_holder": ["alice", "alice", "bob", "carol"],
            "card_mask_hash": ["shared_card", "shared_card", "shared_card", "card_c"],
            "payment_country": ["Portugal", "Italy", "Brazil", "Germany"],
        }
    )

    features = build_risk_features(sample_users_df, transactions)
    row_101 = features.loc[features["id_user"] == 101].iloc[0]
    row_102 = features.loc[features["id_user"] == 102].iloc[0]

    assert set(
        [
            "id_user",
            "card_country_mismatch_count",
            "payment_country_mismatch_count",
            "unique_card_countries",
            "unique_payment_countries",
            "max_users_per_card",
            "card_reuse_flag",
        ]
    ).issubset(features.columns)
    assert row_101["card_country_mismatch_count"] == 1
    assert row_101["payment_country_mismatch_count"] == 1
    assert row_101["unique_card_countries"] == 2
    assert row_102["max_users_per_card"] == 2
    assert row_102["card_reuse_flag"] == 1


def test_build_risk_features_returns_zeroes_for_users_without_transactions(sample_test_users_df, sample_transactions_df):
    features = build_risk_features(sample_test_users_df, sample_transactions_df)

    assert len(features) == len(sample_test_users_df)
    assert features["card_country_mismatch_count"].sum() == 0
    assert features["max_users_per_card"].sum() == 0
