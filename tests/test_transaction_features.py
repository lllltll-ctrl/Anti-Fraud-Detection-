from src.features.transaction_features import build_transaction_aggregates


def test_build_transaction_aggregates_creates_core_metrics(sample_transactions_df):
    aggregated = build_transaction_aggregates(sample_transactions_df)
    row_101 = aggregated.loc[aggregated["id_user"] == 101].iloc[0]

    assert set(
        [
            "id_user",
            "tx_count",
            "success_count",
            "fail_count",
            "fail_ratio",
            "amount_sum",
            "amount_mean",
            "amount_std",
            "amount_max",
            "unique_cards",
            "antifraud_count",
            "fraud_error_count",
            "fraud_error_ratio",
            "has_prepaid_card",
            "success_ratio",
            "tx_count_per_card",
        ]
    ).issubset(aggregated.columns)
    assert row_101["tx_count"] == 2
    assert row_101["success_count"] == 1
    assert row_101["fail_count"] == 1
    assert row_101["fail_ratio"] == 0.5
    assert row_101["amount_sum"] == 30.0
    assert row_101["amount_max"] == 20.0
    assert row_101["unique_cards"] == 1
    assert row_101["antifraud_count"] == 1
    assert row_101["fraud_error_ratio"] == 0.0
    assert row_101["has_prepaid_card"] == 0
    assert row_101["success_ratio"] == 0.5
    assert row_101["tx_count_per_card"] == 2.0


def test_build_transaction_aggregates_returns_one_row_per_user(sample_transactions_df):
    aggregated = build_transaction_aggregates(sample_transactions_df)

    assert aggregated["id_user"].is_unique
    assert sorted(aggregated["id_user"].tolist()) == [101, 102, 103]
