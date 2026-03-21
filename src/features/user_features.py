from __future__ import annotations

import pandas as pd


FREEMAIL_DOMAINS = {
    "gmail.com",
    "yahoo.com",
    "hotmail.com",
    "outlook.com",
    "live.com",
    "icloud.com",
    "aol.com",
    "proton.me",
    "protonmail.com",
}


def build_user_features(users: pd.DataFrame) -> pd.DataFrame:
    features = users.copy()
    features["timestamp_reg"] = pd.to_datetime(features["timestamp_reg"], utc=True)
    features["email_domain"] = features["email"].fillna("").str.split("@").str[-1].str.lower()
    features["email_is_freemail"] = features["email_domain"].isin(FREEMAIL_DOMAINS).astype("int64")
    features["reg_hour"] = features["timestamp_reg"].dt.hour.astype("int64")
    features["reg_weekday"] = features["timestamp_reg"].dt.weekday.astype("int64")
    return features
