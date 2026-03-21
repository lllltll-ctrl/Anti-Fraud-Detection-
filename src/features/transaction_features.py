from __future__ import annotations

import pandas as pd


def build_transaction_aggregates(transactions: pd.DataFrame) -> pd.DataFrame:
    data = transactions[
        ["id_user", "timestamp_tr", "amount", "status", "error_group", "card_mask_hash", "card_type",
         "transaction_type", "currency", "card_brand", "card_holder"]
    ].copy()
    data["timestamp_tr"] = pd.to_datetime(data["timestamp_tr"], utc=True, format="mixed")
    data = data.sort_values(["id_user", "timestamp_tr"])
    data["amount"] = data["amount"].astype(float)
    data["is_success"] = (data["status"] == "success").astype("int64")
    data["is_fail"] = (data["status"] == "fail").astype("int64")
    data["is_antifraud"] = (data["error_group"] == "antifraud").astype("int64")
    data["is_fraud_error"] = (data["error_group"] == "fraud").astype("int64")
    data["is_prepaid"] = (data["card_type"] == "PREPAID").astype("int64")
    data["is_card_init"] = (data["transaction_type"] == "card_init").astype("int64")
    data["is_card_recurring"] = (data["transaction_type"] == "card_recurring").astype("int64")
    data["is_small_amount"] = (data["amount"] < 2.0).astype("int64")

    # Sequential features: card switching and amount change after fail
    data["prev_card"] = data.groupby("id_user")["card_mask_hash"].shift(1)
    data["card_switch"] = (
        (data["card_mask_hash"] != data["prev_card"]) & data["prev_card"].notna()
    ).astype("int64")
    data["prev_status"] = data.groupby("id_user")["status"].shift(1)
    data["prev_amount"] = data.groupby("id_user")["amount"].shift(1)
    data["amount_changed_after_fail"] = (
        (data["prev_status"] == "fail") & (data["amount"] != data["prev_amount"])
    ).astype("int64")

    aggregated = data.groupby("id_user", as_index=False).agg(
        tx_count=("id_user", "size"),
        success_count=("is_success", "sum"),
        fail_count=("is_fail", "sum"),
        amount_sum=("amount", "sum"),
        amount_mean=("amount", "mean"),
        amount_std=("amount", "std"),
        amount_max=("amount", "max"),
        amount_min=("amount", "min"),
        unique_cards=("card_mask_hash", "nunique"),
        antifraud_count=("is_antifraud", "sum"),
        fraud_error_count=("is_fraud_error", "sum"),
        has_prepaid_card=("is_prepaid", "max"),
        unique_currencies=("currency", "nunique"),
        unique_card_brands=("card_brand", "nunique"),
        unique_tx_types=("transaction_type", "nunique"),
        card_init_count=("is_card_init", "sum"),
        card_recurring_count=("is_card_recurring", "sum"),
        unique_error_groups=("error_group", "nunique"),
        unique_card_holders=("card_holder", "nunique"),
        card_switch_count=("card_switch", "sum"),
        small_amount_count=("is_small_amount", "sum"),
        amount_changed_after_fail_count=("amount_changed_after_fail", "sum"),
    )
    aggregated["fail_ratio"] = aggregated["fail_count"] / aggregated["tx_count"].clip(lower=1)
    aggregated["success_ratio"] = aggregated["success_count"] / aggregated["tx_count"].clip(lower=1)
    aggregated["fraud_error_ratio"] = aggregated["fraud_error_count"] / aggregated["tx_count"].clip(lower=1)
    aggregated["tx_count_per_card"] = aggregated["tx_count"] / aggregated["unique_cards"].clip(lower=1)
    aggregated["antifraud_ratio"] = aggregated["antifraud_count"] / aggregated["tx_count"].clip(lower=1)
    aggregated["card_init_ratio"] = aggregated["card_init_count"] / aggregated["tx_count"].clip(lower=1)
    aggregated["card_switch_ratio"] = aggregated["card_switch_count"] / aggregated["tx_count"].clip(lower=1)
    aggregated["amount_std"] = aggregated["amount_std"].fillna(0.0)
    aggregated["amount_range"] = aggregated["amount_max"] - aggregated["amount_min"]
    aggregated["card_holder_per_card"] = aggregated["unique_card_holders"] / aggregated["unique_cards"].clip(lower=1)
    return aggregated
