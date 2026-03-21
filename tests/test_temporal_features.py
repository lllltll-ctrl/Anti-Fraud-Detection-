from src.features.temporal_features import build_temporal_features


def test_build_temporal_features_creates_expected_delay_gap_and_burst_columns(sample_users_df, sample_transactions_df):
    features = build_temporal_features(sample_users_df, sample_transactions_df)
    row_101 = features.loc[features["id_user"] == 101].iloc[0]
    row_102 = features.loc[features["id_user"] == 102].iloc[0]

    assert set(
        [
            "id_user",
            "minutes_to_first_tx",
            "minutes_to_first_success",
            "tx_span_minutes",
            "mean_gap_minutes",
            "max_tx_per_day",
            "tx_in_first_24h",
            "fail_in_first_24h",
        ]
    ).issubset(features.columns)
    assert row_101["minutes_to_first_tx"] == 60.0
    assert row_101["minutes_to_first_success"] == 60.0
    assert row_101["tx_span_minutes"] == 60.0
    assert row_101["mean_gap_minutes"] == 60.0
    assert row_101["tx_in_first_24h"] == 2
    assert row_101["fail_in_first_24h"] == 1
    assert row_102["max_tx_per_day"] == 1


def test_build_temporal_features_returns_zeroes_for_users_without_transactions(sample_test_users_df, sample_transactions_df):
    features = build_temporal_features(sample_test_users_df, sample_transactions_df)

    assert len(features) == len(sample_test_users_df)
    assert features["minutes_to_first_tx"].sum() == 0
    assert features["tx_in_first_24h"].sum() == 0
