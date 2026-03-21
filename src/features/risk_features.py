from __future__ import annotations

import pandas as pd


RISK_FILL_COLUMNS = [
    "card_country_mismatch_count",
    "payment_country_mismatch_count",
    "unique_card_countries",
    "unique_payment_countries",
    "max_users_per_card",
    "card_reuse_flag",
]


def build_risk_features(users: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    user_base = users[["id_user", "reg_country"]].copy()
    tx = transactions[
        ["id_user", "card_country", "payment_country", "card_mask_hash"]
    ].copy()

    card_user_counts = (
        tx.groupby("card_mask_hash")["id_user"]
        .nunique()
        .rename("users_per_card")
        .reset_index()
    )
    tx = tx.merge(card_user_counts, on="card_mask_hash", how="left")
    tx = tx.merge(user_base, on="id_user", how="left")

    tx["card_country_mismatch"] = (tx["card_country"] != tx["reg_country"]).astype("int64")
    tx["payment_country_mismatch"] = (tx["payment_country"] != tx["reg_country"]).astype("int64")

    features = tx.groupby("id_user", as_index=False).agg(
        card_country_mismatch_count=("card_country_mismatch", "sum"),
        payment_country_mismatch_count=("payment_country_mismatch", "sum"),
        unique_card_countries=("card_country", "nunique"),
        unique_payment_countries=("payment_country", "nunique"),
        max_users_per_card=("users_per_card", "max"),
    )
    features["card_reuse_flag"] = (features["max_users_per_card"] > 1).astype("int64")

    features = user_base[["id_user"]].merge(features, on="id_user", how="left")
    for column in RISK_FILL_COLUMNS:
        features[column] = features[column].fillna(0)

    return features
