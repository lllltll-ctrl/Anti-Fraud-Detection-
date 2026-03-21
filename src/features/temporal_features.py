from __future__ import annotations

import numpy as np
import pandas as pd


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


def _compute_max_fail_streak(status_series: pd.Series) -> pd.DataFrame:
    is_fail = (status_series.values == "fail").astype(np.int8)
    user_ids = status_series.index.get_level_values("id_user") if isinstance(status_series.index, pd.MultiIndex) else None

    # Vectorized approach: group by user and compute max consecutive fails
    return is_fail


def build_temporal_features(users: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    user_base = users[["id_user", "timestamp_reg"]].copy()
    user_base["timestamp_reg"] = pd.to_datetime(user_base["timestamp_reg"], utc=True, format="mixed")

    tx = transactions[["id_user", "timestamp_tr", "status", "card_mask_hash"]].copy()
    tx["timestamp_tr"] = pd.to_datetime(tx["timestamp_tr"], utc=True, format="mixed")
    tx = tx.merge(user_base, on="id_user", how="left")
    tx = tx.sort_values(["id_user", "timestamp_tr"])

    tx["minutes_from_registration"] = (tx["timestamp_tr"] - tx["timestamp_reg"]).dt.total_seconds() / 60.0
    tx["success_minutes_from_registration"] = tx["minutes_from_registration"].where(tx["status"] == "success")
    tx["prev_timestamp_tr"] = tx.groupby("id_user")["timestamp_tr"].shift(1)
    tx["gap_minutes"] = (tx["timestamp_tr"] - tx["prev_timestamp_tr"]).dt.total_seconds() / 60.0
    tx["tx_day"] = tx["timestamp_tr"].dt.floor("D")
    tx["within_first_day"] = (tx["minutes_from_registration"] <= 24 * 60).astype("int64")
    tx["fail_within_first_day"] = ((tx["status"] == "fail") & (tx["within_first_day"] == 1)).astype("int64")
    tx["within_first_hour"] = (tx["minutes_from_registration"] <= 60).astype("int64")
    tx["fail_first_hour_flag"] = ((tx["status"] == "fail") & (tx["within_first_hour"] == 1)).astype("int64")
    tx["within_first_6h"] = (tx["minutes_from_registration"] <= 360).astype("int64")
    tx["fail_first_6h_flag"] = ((tx["status"] == "fail") & (tx["within_first_6h"] == 1)).astype("int64")
    tx["is_fail"] = (tx["status"] == "fail").astype("int64")

    # Max consecutive fail streak per user (vectorized with groupby)
    tx["prev_fail"] = tx.groupby("id_user")["is_fail"].shift(1).fillna(0).astype("int64")
    tx["streak_break"] = ((tx["is_fail"] == 1) & (tx["prev_fail"] == 0)).astype("int64")
    tx["streak_id"] = tx.groupby("id_user")["streak_break"].cumsum()
    fail_only = tx[tx["is_fail"] == 1]
    if len(fail_only) > 0:
        streak_lengths = fail_only.groupby(["id_user", "streak_id"]).size().reset_index(name="streak_len")
        max_streaks = streak_lengths.groupby("id_user")["streak_len"].max().reset_index()
        max_streaks.columns = ["id_user", "max_fail_streak"]
    else:
        max_streaks = pd.DataFrame(columns=["id_user", "max_fail_streak"])

    # First hour card diversity
    first_hour_tx = tx[tx["within_first_hour"] == 1]
    if len(first_hour_tx) > 0:
        first_hour_cards = first_hour_tx.groupby("id_user")["card_mask_hash"].nunique().reset_index()
        first_hour_cards.columns = ["id_user", "unique_cards_first_hour"]
    else:
        first_hour_cards = pd.DataFrame(columns=["id_user", "unique_cards_first_hour"])

    day_counts = tx.groupby(["id_user", "tx_day"]).size().rename("tx_per_day").reset_index()
    max_day_counts = day_counts.groupby("id_user")["tx_per_day"].max().rename("max_tx_per_day").reset_index()

    features = tx.groupby("id_user", as_index=False).agg(
        minutes_to_first_tx=("minutes_from_registration", "min"),
        minutes_to_first_success=("success_minutes_from_registration", "min"),
        tx_span_minutes=("minutes_from_registration", lambda values: float(values.max() - values.min()) if len(values) else 0.0),
        mean_gap_minutes=("gap_minutes", "mean"),
        tx_in_first_24h=("within_first_day", "sum"),
        fail_in_first_24h=("fail_within_first_day", "sum"),
        tx_first_hour=("within_first_hour", "sum"),
        fail_first_hour=("fail_first_hour_flag", "sum"),
        tx_first_6h=("within_first_6h", "sum"),
        fail_first_6h=("fail_first_6h_flag", "sum"),
    )
    features = features.merge(max_day_counts, on="id_user", how="left")
    features = features.merge(max_streaks, on="id_user", how="left")
    features = features.merge(first_hour_cards, on="id_user", how="left")
    features = user_base[["id_user"]].merge(features, on="id_user", how="left")

    for column in TEMPORAL_FILL_COLUMNS:
        features[column] = features[column].fillna(0)

    return features
