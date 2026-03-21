from __future__ import annotations

import pandas as pd

from src.features.temporal_features import build_temporal_features
from src.features.risk_features import build_risk_features
from src.features.transaction_features import build_transaction_aggregates
from src.features.user_features import build_user_features


TRANSACTION_FILL_COLUMNS = [
    "tx_count",
    "success_count",
    "fail_count",
    "fail_ratio",
    "amount_sum",
    "amount_mean",
    "amount_std",
    "amount_max",
    "amount_min",
    "amount_range",
    "unique_cards",
    "antifraud_count",
    "fraud_error_count",
    "fraud_error_ratio",
    "has_prepaid_card",
    "success_ratio",
    "tx_count_per_card",
    "antifraud_ratio",
    "unique_currencies",
    "unique_card_brands",
    "unique_tx_types",
    "card_init_count",
    "card_recurring_count",
    "card_init_ratio",
    "unique_error_groups",
    "unique_card_holders",
    "card_holder_per_card",
    "card_switch_count",
    "card_switch_ratio",
    "small_amount_count",
    "amount_changed_after_fail_count",
]

RISK_FILL_COLUMNS = [
    "card_country_mismatch_count",
    "payment_country_mismatch_count",
    "unique_card_countries",
    "unique_payment_countries",
    "max_users_per_card",
    "card_reuse_flag",
]

TEMPORAL_FILL_COLUMNS = [
    "minutes_to_first_tx",
    "minutes_to_first_success",
    "tx_span_minutes",
    "mean_gap_minutes",
    "max_tx_per_day",
    "tx_in_first_24h",
    "fail_in_first_24h",
    "tx_first_hour",
    "fail_first_hour",
    "unique_cards_first_hour",
    "max_fail_streak",
    "tx_first_6h",
    "fail_first_6h",
]


def build_model_dataset(
    users: pd.DataFrame,
    transactions: pd.DataFrame,
    is_train: bool,
) -> pd.DataFrame:
    user_features = build_user_features(users)
    transaction_features = build_transaction_aggregates(transactions)
    risk_features = build_risk_features(users, transactions)
    temporal_features = build_temporal_features(users, transactions)
    dataset = user_features.merge(transaction_features, on="id_user", how="left")
    dataset = dataset.merge(risk_features, on="id_user", how="left")
    dataset = dataset.merge(temporal_features, on="id_user", how="left")

    for column in TRANSACTION_FILL_COLUMNS:
        dataset[column] = dataset[column].fillna(0)

    for column in RISK_FILL_COLUMNS:
        dataset[column] = dataset[column].fillna(0)

    for column in TEMPORAL_FILL_COLUMNS:
        dataset[column] = dataset[column].fillna(0)

    if not is_train and "is_fraud" in dataset.columns:
        dataset = dataset.drop(columns=["is_fraud"])

    return dataset
