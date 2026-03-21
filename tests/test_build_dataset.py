from src.features.build_dataset import build_model_dataset


def test_build_model_dataset_merges_user_and_transaction_features(sample_users_df, sample_transactions_df):
    dataset = build_model_dataset(sample_users_df, sample_transactions_df, is_train=True)

    assert len(dataset) == len(sample_users_df)
    assert "is_fraud" in dataset.columns
    assert "email_domain" in dataset.columns
    assert "email_is_freemail" in dataset.columns
    assert "tx_count" in dataset.columns
    assert "amount_std" in dataset.columns
    assert "amount_max" in dataset.columns
    assert "success_ratio" in dataset.columns
    assert "tx_count_per_card" in dataset.columns
    assert "card_country_mismatch_count" in dataset.columns
    assert "max_users_per_card" in dataset.columns
    assert "card_reuse_flag" in dataset.columns
    assert "fraud_error_ratio" in dataset.columns
    assert "has_prepaid_card" in dataset.columns
    assert "minutes_to_first_tx" in dataset.columns
    assert "tx_in_first_24h" in dataset.columns


def test_build_model_dataset_fills_missing_transaction_values(sample_test_users_df, sample_transactions_df):
    dataset = build_model_dataset(sample_test_users_df, sample_transactions_df, is_train=False)
    row = dataset.loc[dataset["id_user"] == 201].iloc[0]

    assert "is_fraud" not in dataset.columns
    assert row["tx_count"] == 0
    assert row["fail_count"] == 0
