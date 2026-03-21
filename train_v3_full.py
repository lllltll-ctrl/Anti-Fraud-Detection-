"""V3 Full Pipeline: comprehensive feature engineering + graph + ensemble.
Targets F1 0.85+ by exploiting ALL raw data columns.
"""
import time, json, warnings, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from scipy.stats import rankdata, skew, kurtosis
from collections import Counter

warnings.filterwarnings("ignore")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_best_threshold(y_true, y_proba):
    ranked = sorted(zip(y_proba, y_true), key=lambda x: x[0], reverse=True)
    total_pos = int(sum(y_true))
    tp = fp = 0
    fn = total_pos
    best_t, best_f1 = 0.5, 0.0
    i = 0
    while i < len(ranked):
        t = ranked[i][0]
        btp = bfp = 0
        while i < len(ranked) and ranked[i][0] == t:
            if ranked[i][1] == 1:
                btp += 1
            else:
                bfp += 1
            i += 1
        tp += btp
        fp += bfp
        fn -= btp
        pd_ = tp + fp
        rd = tp + fn
        if pd_ == 0 or rd == 0:
            continue
        p = tp / pd_
        r = tp / rd
        if p + r == 0:
            continue
        f1 = 2 * p * r / (p + r)
        if f1 > best_f1:
            best_t = t
            best_f1 = f1
    return best_t, best_f1


def kfold_target_encode(train_col, y, test_col, smoothing=50, n_splits=5, seed=42):
    """K-fold target encoding for a column."""
    gm = y.mean()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_te = np.full(len(train_col), gm)
    y_s = pd.Series(y)

    for tr_i, val_i in skf.split(np.zeros(len(y)), y):
        stats = pd.DataFrame({"c": train_col.iloc[tr_i], "y": y_s.iloc[tr_i]}).groupby("c")["y"].agg(["mean", "count"])
        s = (stats["count"] * stats["mean"] + smoothing * gm) / (stats["count"] + smoothing)
        train_te[val_i] = train_col.iloc[val_i].map(s).fillna(gm).values

    stats_all = pd.DataFrame({"c": train_col, "y": y_s}).groupby("c")["y"].agg(["mean", "count"])
    s_all = (stats_all["count"] * stats_all["mean"] + smoothing * gm) / (stats_all["count"] + smoothing)
    test_te = test_col.map(s_all).fillna(gm).values
    return train_te, test_te


# ============================================================
# 1. LOAD RAW DATA
# ============================================================
log("Loading raw data...")
tr_tx = pd.read_csv("train_transactions.csv")
tr_u = pd.read_csv("train_users.csv")
te_tx = pd.read_csv("test_transactions.csv")
te_u = pd.read_csv("test_users.csv")

fraud_users = set(tr_u[tr_u["is_fraud"] == 1]["id_user"])
all_tx = pd.concat([tr_tx, te_tx], ignore_index=True)
y = tr_u["is_fraud"].values

log(f"Train: {len(tr_u)} users, {len(tr_tx)} txs | Test: {len(te_u)} users, {len(te_tx)} txs")

# ============================================================
# 2. FEATURE ENGINEERING FROM RAW TRANSACTIONS
# ============================================================
log("Building features from raw transactions...")

# Parse timestamps
all_tx["ts"] = pd.to_datetime(all_tx["timestamp_tr"], format="ISO8601")
all_tx["is_success"] = (all_tx["status"] == "success").astype(int)
all_tx["is_fail"] = 1 - all_tx["is_success"]

# Parse user registration timestamps
for df_u in [tr_u, te_u]:
    df_u["ts_reg"] = pd.to_datetime(df_u["timestamp_reg"], format="ISO8601")

# Merge reg time into transactions for time-since-reg features
reg_map = pd.concat([tr_u[["id_user", "ts_reg"]], te_u[["id_user", "ts_reg"]]])
all_tx = all_tx.merge(reg_map, on="id_user", how="left")
all_tx["minutes_since_reg"] = (all_tx["ts"] - all_tx["ts_reg"]).dt.total_seconds() / 60.0


def build_user_features(tx_df):
    """Build comprehensive features per user from transactions."""
    g = tx_df.groupby("id_user")

    feats = pd.DataFrame({"id_user": g.ngroup().index if hasattr(g, 'ngroup') else tx_df["id_user"].unique()})
    feats = pd.DataFrame({"id_user": tx_df["id_user"].unique()})

    # --- BASIC COUNTS ---
    basic = g.agg(
        tx_count=("id_user", "count"),
        success_count=("is_success", "sum"),
        fail_count=("is_fail", "sum"),
        amount_sum=("amount", "sum"),
        amount_mean=("amount", "mean"),
        amount_std=("amount", "std"),
        amount_max=("amount", "max"),
        amount_min=("amount", "min"),
    ).reset_index()
    basic["amount_range"] = basic["amount_max"] - basic["amount_min"]
    basic["fail_ratio"] = basic["fail_count"] / basic["tx_count"].clip(lower=1)
    basic["success_ratio"] = basic["success_count"] / basic["tx_count"].clip(lower=1)
    basic["amount_cv"] = basic["amount_std"] / basic["amount_mean"].clip(lower=0.01)
    feats = feats.merge(basic, on="id_user", how="left")

    # --- AMOUNT DISTRIBUTION ---
    log("  Amount distribution...")
    amt_pct = g["amount"].quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).unstack()
    amt_pct.columns = [f"amount_p{int(q*100)}" for q in amt_pct.columns]
    amt_pct = amt_pct.reset_index()
    feats = feats.merge(amt_pct, on="id_user", how="left")

    amt_skew = g["amount"].apply(lambda x: skew(x, nan_policy="omit") if len(x) > 2 else 0).reset_index()
    amt_skew.columns = ["id_user", "amount_skewness"]
    feats = feats.merge(amt_skew, on="id_user", how="left")

    amt_kurt = g["amount"].apply(lambda x: kurtosis(x, nan_policy="omit") if len(x) > 2 else 0).reset_index()
    amt_kurt.columns = ["id_user", "amount_kurtosis"]
    feats = feats.merge(amt_kurt, on="id_user", how="left")

    # Round amounts
    round_amt = tx_df.copy()
    round_amt["is_round"] = (round_amt["amount"] % 1.0 == 0).astype(int)
    round_cnt = round_amt.groupby("id_user")["is_round"].agg(["sum", "mean"]).reset_index()
    round_cnt.columns = ["id_user", "round_amount_count", "round_amount_ratio"]
    feats = feats.merge(round_cnt, on="id_user", how="left")

    # Small amounts
    small = tx_df.copy()
    small["is_small"] = (small["amount"] < 2.0).astype(int)
    small_cnt = small.groupby("id_user")["is_small"].agg(["sum", "mean"]).reset_index()
    small_cnt.columns = ["id_user", "small_amount_count", "small_amount_ratio"]
    feats = feats.merge(small_cnt, on="id_user", how="left")

    # --- CARD DIVERSITY ---
    log("  Card diversity...")
    card_div = g.agg(
        unique_cards=("card_mask_hash", "nunique"),
        unique_card_holders=("card_holder", "nunique"),
        unique_card_brands=("card_brand", "nunique"),
        unique_card_types=("card_type", "nunique"),
        unique_currencies=("currency", "nunique"),
        unique_card_countries=("card_country", "nunique"),
        unique_payment_countries=("payment_country", "nunique"),
        unique_tx_types=("transaction_type", "nunique"),
        unique_error_groups=("error_group", "nunique"),
    ).reset_index()
    feats = feats.merge(card_div, on="id_user", how="left")

    # Card switch count
    sorted_tx = tx_df.sort_values(["id_user", "ts"])
    sorted_tx["prev_card"] = sorted_tx.groupby("id_user")["card_mask_hash"].shift(1)
    sorted_tx["card_switched"] = (sorted_tx["card_mask_hash"] != sorted_tx["prev_card"]).astype(int)
    sorted_tx.loc[sorted_tx["prev_card"].isna(), "card_switched"] = 0
    card_sw = sorted_tx.groupby("id_user")["card_switched"].sum().reset_index()
    card_sw.columns = ["id_user", "card_switch_count"]
    feats = feats.merge(card_sw, on="id_user", how="left")
    feats["card_switch_ratio"] = feats["card_switch_count"] / feats["tx_count"].clip(lower=1)

    # --- ERROR GROUP BREAKDOWN ---
    log("  Error group breakdown...")
    top_errors = ["antifraud", "fraud", "3ds error", "insufficient funds", "card problem",
                  "decline (general)", "cvv error", "issuer decline", "limit exceeded", "do not honor"]
    for err in top_errors:
        col_name = "err_" + err.replace(" ", "_").replace("(", "").replace(")", "")
        tx_df[f"_is_{col_name}"] = (tx_df["error_group"] == err).astype(int)
        err_cnt = tx_df.groupby("id_user")[f"_is_{col_name}"].agg(["sum", "mean"]).reset_index()
        err_cnt.columns = ["id_user", f"{col_name}_count", f"{col_name}_ratio"]
        feats = feats.merge(err_cnt, on="id_user", how="left")

    # --- CARD TYPE BREAKDOWN ---
    log("  Card type breakdown...")
    for ctype in ["DEBIT", "CREDIT", "PREPAID"]:
        tx_df[f"_is_{ctype}"] = tx_df["card_type"].str.upper().str.contains(ctype, na=False).astype(int)
        ct_agg = tx_df.groupby("id_user")[f"_is_{ctype}"].agg(["sum", "mean"]).reset_index()
        ct_agg.columns = ["id_user", f"{ctype.lower()}_count", f"{ctype.lower()}_ratio"]
        feats = feats.merge(ct_agg, on="id_user", how="left")

    # Card type fail ratios
    for ctype in ["DEBIT", "CREDIT", "PREPAID"]:
        subset = tx_df[tx_df[f"_is_{ctype}"] == 1]
        if len(subset) > 0:
            ct_fail = subset.groupby("id_user")["is_fail"].mean().reset_index()
            ct_fail.columns = ["id_user", f"{ctype.lower()}_fail_ratio"]
            feats = feats.merge(ct_fail, on="id_user", how="left")

    # --- TRANSACTION TYPE BREAKDOWN ---
    log("  Transaction type breakdown...")
    for ttype in ["card_init", "card_recurring", "google-pay", "apple-pay", "resign"]:
        safe_name = ttype.replace("-", "_")
        tx_df[f"_is_{safe_name}"] = (tx_df["transaction_type"] == ttype).astype(int)
        tt_agg = tx_df.groupby("id_user")[f"_is_{safe_name}"].agg(["sum", "mean"]).reset_index()
        tt_agg.columns = ["id_user", f"{safe_name}_count", f"{safe_name}_ratio"]
        feats = feats.merge(tt_agg, on="id_user", how="left")

    # Digital wallet aggregate
    tx_df["_is_digital_wallet"] = tx_df["transaction_type"].isin(["google-pay", "apple-pay"]).astype(int)
    dw = tx_df.groupby("id_user")["_is_digital_wallet"].agg(["sum", "mean"]).reset_index()
    dw.columns = ["id_user", "digital_wallet_count", "digital_wallet_ratio"]
    feats = feats.merge(dw, on="id_user", how="left")
    feats["has_digital_wallet"] = (feats["digital_wallet_count"] > 0).astype(int)

    # --- TEMPORAL FEATURES ---
    log("  Temporal features...")
    # Minutes to first tx, first success
    first_tx = g["minutes_since_reg"].min().reset_index()
    first_tx.columns = ["id_user", "minutes_to_first_tx"]
    feats = feats.merge(first_tx, on="id_user", how="left")

    success_tx = tx_df[tx_df["is_success"] == 1]
    first_success = success_tx.groupby("id_user")["minutes_since_reg"].min().reset_index()
    first_success.columns = ["id_user", "minutes_to_first_success"]
    feats = feats.merge(first_success, on="id_user", how="left")

    # Tx span
    tx_span = g["minutes_since_reg"].agg(lambda x: x.max() - x.min()).reset_index()
    tx_span.columns = ["id_user", "tx_span_minutes"]
    feats = feats.merge(tx_span, on="id_user", how="left")

    # Mean gap
    sorted_tx2 = tx_df.sort_values(["id_user", "ts"])
    sorted_tx2["prev_ts"] = sorted_tx2.groupby("id_user")["ts"].shift(1)
    sorted_tx2["gap_minutes"] = (sorted_tx2["ts"] - sorted_tx2["prev_ts"]).dt.total_seconds() / 60.0
    mean_gap = sorted_tx2.groupby("id_user")["gap_minutes"].mean().reset_index()
    mean_gap.columns = ["id_user", "mean_gap_minutes"]
    feats = feats.merge(mean_gap, on="id_user", how="left")

    # First 24h, first hour, first 6h
    tx_df["in_first_24h"] = (tx_df["minutes_since_reg"] <= 1440).astype(int)
    tx_df["in_first_hour"] = (tx_df["minutes_since_reg"] <= 60).astype(int)
    tx_df["in_first_6h"] = (tx_df["minutes_since_reg"] <= 360).astype(int)

    f24h = tx_df.groupby("id_user")["in_first_24h"].sum().reset_index()
    f24h.columns = ["id_user", "tx_in_first_24h"]
    feats = feats.merge(f24h, on="id_user", how="left")

    fail_f24h = tx_df[tx_df["is_fail"] == 1].groupby("id_user")["in_first_24h"].sum().reset_index()
    fail_f24h.columns = ["id_user", "fail_in_first_24h"]
    feats = feats.merge(fail_f24h, on="id_user", how="left")

    fh = tx_df.groupby("id_user")["in_first_hour"].sum().reset_index()
    fh.columns = ["id_user", "tx_first_hour"]
    feats = feats.merge(fh, on="id_user", how="left")

    fail_fh = tx_df[tx_df["is_fail"] == 1].groupby("id_user")["in_first_hour"].sum().reset_index()
    fail_fh.columns = ["id_user", "fail_first_hour"]
    feats = feats.merge(fail_fh, on="id_user", how="left")

    f6h = tx_df.groupby("id_user")["in_first_6h"].sum().reset_index()
    f6h.columns = ["id_user", "tx_first_6h"]
    feats = feats.merge(f6h, on="id_user", how="left")

    fail_f6h = tx_df[tx_df["is_fail"] == 1].groupby("id_user")["in_first_6h"].sum().reset_index()
    fail_f6h.columns = ["id_user", "fail_first_6h"]
    feats = feats.merge(fail_f6h, on="id_user", how="left")

    # Max tx per day
    tx_df["tx_day"] = tx_df["ts"].dt.date
    daily = tx_df.groupby(["id_user", "tx_day"]).size().reset_index(name="daily_count")
    max_daily = daily.groupby("id_user")["daily_count"].max().reset_index()
    max_daily.columns = ["id_user", "max_tx_per_day"]
    feats = feats.merge(max_daily, on="id_user", how="left")

    # Max fail streak
    sorted_tx3 = tx_df.sort_values(["id_user", "ts"])
    def max_fail_streak(group):
        fails = group["is_fail"].values
        max_s = cur = 0
        for f in fails:
            if f == 1:
                cur += 1
                max_s = max(max_s, cur)
            else:
                cur = 0
        return max_s
    mfs = sorted_tx3.groupby("id_user").apply(max_fail_streak).reset_index()
    mfs.columns = ["id_user", "max_fail_streak"]
    feats = feats.merge(mfs, on="id_user", how="left")

    # Unique cards in first hour
    first_h_tx = tx_df[tx_df["in_first_hour"] == 1]
    ucfh = first_h_tx.groupby("id_user")["card_mask_hash"].nunique().reset_index()
    ucfh.columns = ["id_user", "unique_cards_first_hour"]
    feats = feats.merge(ucfh, on="id_user", how="left")

    # --- VELOCITY / BURST ---
    log("  Velocity features...")
    # Max transactions per hour
    tx_df["tx_hour_bucket"] = tx_df["ts"].dt.floor("h")
    hourly = tx_df.groupby(["id_user", "tx_hour_bucket"]).size().reset_index(name="hourly_count")
    max_hourly = hourly.groupby("id_user")["hourly_count"].max().reset_index()
    max_hourly.columns = ["id_user", "tx_per_hour_max"]
    feats = feats.merge(max_hourly, on="id_user", how="left")

    # Amount per hour max
    hourly_amt = tx_df.groupby(["id_user", "tx_hour_bucket"])["amount"].sum().reset_index()
    max_hourly_amt = hourly_amt.groupby("id_user")["amount"].max().reset_index()
    max_hourly_amt.columns = ["id_user", "amount_per_hour_max"]
    feats = feats.merge(max_hourly_amt, on="id_user", how="left")

    # Min time between consecutive fails
    fail_tx = tx_df[tx_df["is_fail"] == 1].sort_values(["id_user", "ts"])
    fail_tx["prev_fail_ts"] = fail_tx.groupby("id_user")["ts"].shift(1)
    fail_tx["fail_gap_sec"] = (fail_tx["ts"] - fail_tx["prev_fail_ts"]).dt.total_seconds()
    min_fail_gap = fail_tx.groupby("id_user")["fail_gap_sec"].min().reset_index()
    min_fail_gap.columns = ["id_user", "min_fail_gap_seconds"]
    feats = feats.merge(min_fail_gap, on="id_user", how="left")

    # --- TIME-OF-DAY PATTERNS ---
    log("  Time-of-day patterns...")
    tx_df["tx_hour"] = tx_df["ts"].dt.hour
    tx_df["is_night"] = tx_df["tx_hour"].isin(range(22, 24)).astype(int) | tx_df["tx_hour"].isin(range(0, 6)).astype(int)
    tx_df["is_business"] = ((tx_df["tx_hour"] >= 9) & (tx_df["tx_hour"] <= 18)).astype(int)

    night = tx_df.groupby("id_user")["is_night"].mean().reset_index()
    night.columns = ["id_user", "night_tx_ratio"]
    feats = feats.merge(night, on="id_user", how="left")

    biz = tx_df.groupby("id_user")["is_business"].mean().reset_index()
    biz.columns = ["id_user", "business_hours_ratio"]
    feats = feats.merge(biz, on="id_user", how="left")

    mode_h = tx_df.groupby("id_user")["tx_hour"].agg(lambda x: x.mode().iloc[0] if len(x) > 0 else 12).reset_index()
    mode_h.columns = ["id_user", "mode_hour"]
    feats = feats.merge(mode_h, on="id_user", how="left")

    dist_h = tx_df.groupby("id_user")["tx_hour"].nunique().reset_index()
    dist_h.columns = ["id_user", "distinct_hours"]
    feats = feats.merge(dist_h, on="id_user", how="left")

    # --- COUNTRY MISMATCH ---
    log("  Country mismatch...")
    # Need reg_country from users
    rc_map = pd.concat([tr_u[["id_user", "reg_country"]], te_u[["id_user", "reg_country"]]])
    tx_rc = tx_df.merge(rc_map, on="id_user", how="left")
    tx_rc["card_country_mismatch"] = (tx_rc["card_country"] != tx_rc["reg_country"]).astype(int)
    tx_rc["payment_country_mismatch"] = (tx_rc["payment_country"] != tx_rc["reg_country"]).astype(int)
    cm = tx_rc.groupby("id_user").agg(
        card_country_mismatch_count=("card_country_mismatch", "sum"),
        payment_country_mismatch_count=("payment_country_mismatch", "sum"),
    ).reset_index()
    feats = feats.merge(cm, on="id_user", how="left")

    # Card reuse (max users per card)
    card_user_counts = tx_df.groupby("card_mask_hash")["id_user"].nunique().to_dict()
    tx_df["_card_n_users"] = tx_df["card_mask_hash"].map(card_user_counts)
    max_card_reuse = tx_df.groupby("id_user")["_card_n_users"].max().reset_index()
    max_card_reuse.columns = ["id_user", "max_users_per_card"]
    feats = feats.merge(max_card_reuse, on="id_user", how="left")
    feats["card_reuse_flag"] = (feats["max_users_per_card"] > 1).astype(int)

    # Amount changed after fail
    sorted_tx4 = tx_df.sort_values(["id_user", "ts"])
    sorted_tx4["prev_amount"] = sorted_tx4.groupby("id_user")["amount"].shift(1)
    sorted_tx4["prev_fail"] = sorted_tx4.groupby("id_user")["is_fail"].shift(1)
    sorted_tx4["amt_changed_after_fail"] = (
        (sorted_tx4["prev_fail"] == 1) &
        (sorted_tx4["amount"] != sorted_tx4["prev_amount"])
    ).astype(int)
    acaf = sorted_tx4.groupby("id_user")["amt_changed_after_fail"].sum().reset_index()
    acaf.columns = ["id_user", "amount_changed_after_fail_count"]
    feats = feats.merge(acaf, on="id_user", how="left")

    # Fail before first success
    def fails_before_success(group):
        for i, row in enumerate(group["is_success"].values):
            if row == 1:
                return i
        return len(group)
    sorted_tx5 = tx_df.sort_values(["id_user", "ts"])
    fbs = sorted_tx5.groupby("id_user").apply(fails_before_success).reset_index()
    fbs.columns = ["id_user", "fails_before_first_success"]
    feats = feats.merge(fbs, on="id_user", how="left")

    return feats


# Build features for all transactions
user_feats = build_user_features(all_tx)

# ============================================================
# 3. USER PROFILE FEATURES
# ============================================================
log("Adding user profile features...")
all_users = pd.concat([tr_u, te_u[te_u.columns.difference(["is_fraud"])]], ignore_index=True)
all_users["email_domain"] = all_users["email"].str.split("@").str[1].fillna("unknown")
all_users["email_is_freemail"] = all_users["email_domain"].isin(["gmail.com", "yahoo.com", "hotmail.com", "outlook.com"]).astype(int)
all_users["reg_hour"] = pd.to_datetime(all_users["timestamp_reg"], format="ISO8601").dt.hour
all_users["reg_weekday"] = pd.to_datetime(all_users["timestamp_reg"], format="ISO8601").dt.weekday
all_users["reg_is_night"] = all_users["reg_hour"].isin(list(range(0, 6)) + list(range(22, 24))).astype(int)
all_users["reg_is_weekend"] = (all_users["reg_weekday"] >= 5).astype(int)

user_profile_cols = ["id_user", "gender", "traffic_type", "reg_country", "email_domain",
                     "email_is_freemail", "reg_hour", "reg_weekday", "reg_is_night", "reg_is_weekend"]
user_feats = user_feats.merge(all_users[user_profile_cols], on="id_user", how="left")

# Gender-traffic interaction
user_feats["gender_traffic"] = user_feats["gender"].fillna("unk") + "_" + user_feats["traffic_type"].fillna("unk")

# ============================================================
# 4. DERIVED FEATURES
# ============================================================
log("Adding derived features...")
tx = user_feats["tx_count"].clip(lower=1)
uc = user_feats["unique_cards"].clip(lower=1)

user_feats["cards_per_tx"] = user_feats["unique_cards"] / tx
user_feats["holders_per_tx"] = user_feats["unique_card_holders"] / tx
user_feats["switches_per_card"] = user_feats["card_switch_count"] / uc
user_feats["tx_count_per_card"] = user_feats["tx_count"] / uc
user_feats["card_holder_per_card"] = user_feats["unique_card_holders"] / uc

fh = user_feats["tx_first_hour"].clip(lower=1)
user_feats["fail_ratio_first_hour"] = user_feats["fail_first_hour"].fillna(0) / fh
user_feats["cards_first_hour_ratio"] = user_feats["unique_cards_first_hour"].fillna(0) / fh
user_feats["tx_first_6h_ratio"] = user_feats["tx_first_6h"] / tx
user_feats["multi_holder_switch"] = (user_feats["unique_card_holders"] > 1).astype(int) * user_feats["card_switch_count"]
user_feats["country_mismatch_total"] = user_feats["card_country_mismatch_count"].fillna(0) + user_feats["payment_country_mismatch_count"].fillna(0)
user_feats["fail_streak_per_tx"] = user_feats["max_fail_streak"].fillna(0) / tx
user_feats["fast_starter"] = (user_feats["minutes_to_first_tx"] < 60).astype(int)
user_feats["very_fast_starter"] = (user_feats["minutes_to_first_tx"] < 10).astype(int)

for c in ["tx_count", "amount_sum", "card_switch_count", "minutes_to_first_tx", "tx_span_minutes", "mean_gap_minutes"]:
    if c in user_feats.columns:
        user_feats[f"log_{c}"] = np.log1p(user_feats[c].clip(lower=0).fillna(0))

user_feats["risk_combo"] = (
    (user_feats["card_switch_count"] > 3).astype(int)
    + (user_feats["unique_card_holders"] > 1).astype(int)
    + (user_feats["fail_ratio"] > 0.5).astype(int)
    + (user_feats["country_mismatch_total"] > 0).astype(int)
)

# ============================================================
# 5. GRAPH FEATURES (K-FOLD)
# ============================================================
log("Building card/holder graph...")
user_cards = all_tx.groupby("id_user")["card_mask_hash"].apply(set).to_dict()
card_users_map = all_tx.groupby("card_mask_hash")["id_user"].apply(set).to_dict()
user_holders = all_tx.groupby("id_user")["card_holder"].apply(lambda x: set(x.dropna().str.lower())).to_dict()
holder_users_map = {}
for uid, holders in user_holders.items():
    for h in holders:
        if h not in holder_users_map:
            holder_users_map[h] = set()
        holder_users_map[h].add(uid)

GRAPH_FEAT_NAMES = [
    "graph_card_fraud_n", "graph_card_total_n", "graph_card_fraud_ratio",
    "graph_card_max_density", "graph_n_fraud_cards", "graph_fraud_card_ratio",
    "graph_holder_fraud_n", "graph_holder_total_n", "graph_holder_fraud_ratio",
    "graph_any_fraud_connection", "graph_total_fraud_connections",
]


def get_graph_features(uid, known_fraud):
    cards = user_cards.get(uid, set())
    holders = user_holders.get(uid, set())
    card_neighbors = set()
    card_fraud_neighbors = set()
    max_card_density = 0.0
    n_fraud_cards = 0
    for card in cards:
        others = card_users_map.get(card, set()) - {uid}
        card_neighbors |= others
        fraud_on_card = others & known_fraud
        card_fraud_neighbors |= fraud_on_card
        if others:
            density = len(fraud_on_card) / len(others)
            max_card_density = max(max_card_density, density)
        if fraud_on_card:
            n_fraud_cards += 1
    holder_neighbors = set()
    holder_fraud_neighbors = set()
    for h in holders:
        others = holder_users_map.get(h, set()) - {uid}
        holder_neighbors |= others
        holder_fraud_neighbors |= (others & known_fraud)
    n_card_fraud = len(card_fraud_neighbors)
    n_card_total = len(card_neighbors)
    n_holder_fraud = len(holder_fraud_neighbors)
    n_holder_total = len(holder_neighbors)
    return [
        n_card_fraud, n_card_total, n_card_fraud / max(n_card_total, 1),
        max_card_density, n_fraud_cards, n_fraud_cards / max(len(cards), 1),
        n_holder_fraud, n_holder_total, n_holder_fraud / max(n_holder_total, 1),
        int(n_card_fraud > 0 or n_holder_fraud > 0),
        n_card_fraud + n_holder_fraud,
    ]


# FULL graph for train (no K-fold — use ALL labels)
log("Building FULL graph features for train (no K-fold)...")
train_ids = tr_u["id_user"].values
test_ids = te_u["id_user"].values
train_graph = np.zeros((len(train_ids), len(GRAPH_FEAT_NAMES)))

for i, uid in enumerate(train_ids):
    train_graph[i] = get_graph_features(uid, fraud_users)
log(f"  Done. Train users with fraud connection: {(train_graph[:, 9] > 0).sum()}")

# Graph for test (use all train labels)
log("Building graph features for test...")
test_graph = np.zeros((len(test_ids), len(GRAPH_FEAT_NAMES)))
for i, uid in enumerate(test_ids):
    test_graph[i] = get_graph_features(uid, fraud_users)

# Add graph features to user_feats
train_feat_df = user_feats[user_feats["id_user"].isin(set(train_ids))].copy()
test_feat_df = user_feats[user_feats["id_user"].isin(set(test_ids))].copy()

# Ensure order matches
train_feat_df = train_feat_df.set_index("id_user").loc[train_ids].reset_index()
test_feat_df = test_feat_df.set_index("id_user").loc[test_ids].reset_index()

for j, name in enumerate(GRAPH_FEAT_NAMES):
    train_feat_df[name] = train_graph[:, j]
    test_feat_df[name] = test_graph[:, j]

# ============================================================
# 6. TARGET ENCODING
# ============================================================
log("Target encoding...")
te_cols = ["reg_country", "traffic_type", "gender", "email_domain", "gender_traffic"]
for col in te_cols:
    train_col = train_feat_df[col].fillna("missing").astype(str)
    test_col = test_feat_df[col].fillna("missing").astype(str)
    tr_te, te_te = kfold_target_encode(train_col, y, test_col, smoothing=50)
    train_feat_df[f"{col}_te"] = tr_te
    test_feat_df[f"{col}_te"] = te_te

# Dominant card brand target encoding
dom_brand = all_tx.groupby("id_user")["card_brand"].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "unknown").to_dict()
train_feat_df["dominant_card_brand"] = train_feat_df["id_user"].map(dom_brand).fillna("unknown")
test_feat_df["dominant_card_brand"] = test_feat_df["id_user"].map(dom_brand).fillna("unknown")
tr_te, te_te = kfold_target_encode(
    train_feat_df["dominant_card_brand"].astype(str), y,
    test_feat_df["dominant_card_brand"].astype(str), smoothing=50
)
train_feat_df["card_brand_te"] = tr_te
test_feat_df["card_brand_te"] = te_te

# Dominant error group target encoding
dom_err = all_tx[all_tx["error_group"].notna()].groupby("id_user")["error_group"].agg(
    lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "none"
).to_dict()
train_feat_df["dominant_error"] = train_feat_df["id_user"].map(dom_err).fillna("none")
test_feat_df["dominant_error"] = test_feat_df["id_user"].map(dom_err).fillna("none")
tr_te, te_te = kfold_target_encode(
    train_feat_df["dominant_error"].astype(str), y,
    test_feat_df["dominant_error"].astype(str), smoothing=50
)
train_feat_df["error_group_te"] = tr_te
test_feat_df["error_group_te"] = te_te

# ============================================================
# 7. PREPARE FOR TRAINING
# ============================================================
log("Preparing for training...")
EXCLUDED = {"id_user", "timestamp_reg", "email", "is_fraud", "ts_reg",
            "dominant_card_brand", "dominant_error", "gender_traffic"}
feature_cols = [c for c in train_feat_df.columns if c not in EXCLUDED]
cat_cols = ["gender", "traffic_type", "reg_country", "email_domain"]
cat_cols = [c for c in cat_cols if c in feature_cols]
num_cols = [c for c in feature_cols if c not in cat_cols]

log(f"Total features: {len(feature_cols)}, Cat: {len(cat_cols)}, Num: {len(num_cols)}")

tr = train_feat_df.copy()
te = test_feat_df.copy()
for col in cat_cols:
    vals = pd.concat([tr[col], te[col]]).fillna("missing").astype(str)
    cats = sorted(vals.unique().tolist())
    tr[col] = pd.Categorical(tr[col].fillna("missing").astype(str), categories=cats)
    te[col] = pd.Categorical(te[col].fillna("missing").astype(str), categories=cats)
for col in num_cols:
    fill = float(tr[col].median()) if not tr[col].dropna().empty else 0.0
    tr[col] = tr[col].fillna(fill).astype(float)
    te[col] = te[col].fillna(fill).astype(float)

# ============================================================
# 8. TRAIN ENSEMBLE
# ============================================================
log("Training ensemble...")
all_oofs = []
all_tests = []
labels = []

configs = [
    ("gbdt_7f_42", 7, 42, "lgb", {"lr": 0.02, "leaves": 63, "depth": -1, "sub": 0.8, "col": 0.7}),
    ("gbdt_5f_42", 5, 42, "lgb", {"lr": 0.02, "leaves": 63, "depth": -1, "sub": 0.8, "col": 0.7}),
    ("gbdt_7f_123", 7, 123, "lgb", {"lr": 0.02, "leaves": 63, "depth": -1, "sub": 0.8, "col": 0.7}),
    ("gbdt_5f_123", 5, 123, "lgb", {"lr": 0.02, "leaves": 63, "depth": -1, "sub": 0.8, "col": 0.7}),
    ("gbdt_7f_999", 7, 999, "lgb", {"lr": 0.015, "leaves": 95, "depth": 8, "sub": 0.75, "col": 0.65}),
    ("dart_7f_42", 7, 42, "dart", {"lr": 0.05, "leaves": 63, "depth": -1, "sub": 0.8, "col": 0.7}),
    ("dart_5f_123", 5, 123, "dart", {"lr": 0.05, "leaves": 63, "depth": -1, "sub": 0.8, "col": 0.7}),
]

for name, n_folds, seed, mtype, params in configs:
    log(f"=== {name} ===")
    skf_m = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    tp_arr = np.zeros(len(te))

    for fold, (tr_idx, val_idx) in enumerate(skf_m.split(tr, y)):
        if mtype == "lgb":
            m = lgb.LGBMClassifier(
                objective="binary", n_estimators=5000, learning_rate=params["lr"],
                num_leaves=params["leaves"], max_depth=params["depth"],
                min_child_samples=50, subsample=params["sub"], colsample_bytree=params["col"],
                reg_alpha=0.1, reg_lambda=1.0, random_state=seed, n_jobs=-1, verbosity=-1,
            )
            m.fit(
                tr[feature_cols].iloc[tr_idx], y[tr_idx],
                eval_set=[(tr[feature_cols].iloc[val_idx], y[val_idx])],
                eval_metric="binary_logloss",
                callbacks=[lgb.early_stopping(200, verbose=False)],
                categorical_feature=cat_cols,
            )
        elif mtype == "dart":
            m = lgb.LGBMClassifier(
                boosting_type="dart", objective="binary", n_estimators=1000,
                learning_rate=params["lr"], num_leaves=params["leaves"], max_depth=params["depth"],
                min_child_samples=50, subsample=params["sub"], colsample_bytree=params["col"],
                reg_alpha=0.1, reg_lambda=1.0, drop_rate=0.1, skip_drop=0.5,
                random_state=seed, n_jobs=-1, verbosity=-1,
            )
            m.fit(
                tr[feature_cols].iloc[tr_idx], y[tr_idx],
                eval_set=[(tr[feature_cols].iloc[val_idx], y[val_idx])],
                eval_metric="binary_logloss",
                callbacks=[lgb.early_stopping(50, verbose=False)],
                categorical_feature=cat_cols,
            )

        oof[val_idx] = m.predict_proba(tr[feature_cols].iloc[val_idx])[:, 1]
        tp_arr += m.predict_proba(te[feature_cols])[:, 1] / n_folds

    t_, f1_ = find_best_threshold(y, oof)
    log(f"  F1={f1_:.4f}")
    all_oofs.append(oof)
    all_tests.append(tp_arr)
    labels.append(name)

    if name == "gbdt_7f_42":
        imp = dict(zip(feature_cols, m.feature_importances_))
        sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)
        log("  Top-30 feature importances:")
        for fn, fv in sorted_imp[:30]:
            log(f"    {fn}: {fv}")

# ============================================================
# 9. BLEND
# ============================================================
log("\n=== BLEND ===")
n_models = len(all_oofs)

avg_oof = np.mean(all_oofs, axis=0)
avg_test = np.mean(all_tests, axis=0)
t_eq, f1_eq = find_best_threshold(y, avg_oof)
log(f"Equal blend: F1={f1_eq:.4f}")

best_weights = np.ones(n_models) / n_models
best_blend_f1 = f1_eq
for _ in range(5):
    improved = False
    for i in range(n_models):
        for delta in [-0.15, -0.1, -0.05, -0.02, 0.02, 0.05, 0.1, 0.15]:
            w = best_weights.copy()
            w[i] = max(0, w[i] + delta)
            if w.sum() == 0:
                continue
            w /= w.sum()
            blend = sum(w[j] * all_oofs[j] for j in range(n_models))
            _, f1_b = find_best_threshold(y, blend)
            if f1_b > best_blend_f1:
                best_blend_f1 = f1_b
                best_weights = w.copy()
                improved = True
    if not improved:
        break

log(f"Optimized blend: F1={best_blend_f1:.4f}")
for i, lbl in enumerate(labels):
    if best_weights[i] > 0.001:
        log(f"  {lbl}: {best_weights[i]:.3f}")

rank_oof = np.mean([rankdata(o) for o in all_oofs], axis=0)
rank_test = np.mean([rankdata(t_) for t_ in all_tests], axis=0)
t_rk, f1_rk = find_best_threshold(y, rank_oof)
log(f"Rank blend: F1={f1_rk:.4f}")

individual_f1s = []
for i in range(n_models):
    _, f1_i = find_best_threshold(y, all_oofs[i])
    individual_f1s.append(f1_i)
sorted_idx = sorted(range(n_models), key=lambda x: individual_f1s[x], reverse=True)
best_topk_f1 = 0.0
best_topk_oof = None
best_topk_test = None
for k in [3, 4, 5, 6, 7]:
    if k > n_models:
        break
    top_idx = sorted_idx[:k]
    sub_oof = np.mean([all_oofs[i] for i in top_idx], axis=0)
    sub_test = np.mean([all_tests[i] for i in top_idx], axis=0)
    _, f1_k = find_best_threshold(y, sub_oof)
    log(f"  Top-{k}: F1={f1_k:.4f}")
    if f1_k > best_topk_f1:
        best_topk_f1 = f1_k
        best_topk_oof = sub_oof
        best_topk_test = sub_test

best_f1 = max(f1_eq, f1_rk, best_blend_f1, best_topk_f1)
if best_blend_f1 >= best_f1:
    final_test = sum(best_weights[j] * all_tests[j] for j in range(n_models))
    final_oof = sum(best_weights[j] * all_oofs[j] for j in range(n_models))
    approach = "optimized_blend"
elif best_topk_f1 >= best_f1:
    final_test = best_topk_test
    final_oof = best_topk_oof
    approach = "topk_blend"
elif f1_rk >= best_f1:
    final_test = rank_test
    final_oof = rank_oof
    approach = "rank_blend"
else:
    final_test = avg_test
    final_oof = avg_oof
    approach = "equal_blend"

t_final, _ = find_best_threshold(y, final_oof)
preds = (final_test >= t_final).astype(int)

log(f"\nFINAL: {approach}, F1={best_f1:.4f}, fraud={preds.sum()} ({preds.mean()*100:.2f}%)")

BASELINE = 0.6672
if best_f1 > BASELINE:
    sub = pd.DataFrame({"id_user": test_feat_df["id_user"].astype("int64"), "is_fraud": preds})
    sub.to_csv("artifacts/submissions/submission.csv", index=False)
    log(f"IMPROVED! Saved submission.csv (was {BASELINE}, now {best_f1:.4f})")
    metrics = {
        "oof_f1": round(best_f1, 6),
        "approach": f"v3_full_{approach}_{n_models}models",
        "n_features": len(feature_cols),
        "predicted_fraud": int(preds.sum()),
        "predicted_fraud_rate": round(float(preds.mean()), 6),
    }
    with open("artifacts/reports/baseline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
else:
    sub = pd.DataFrame({"id_user": test_feat_df["id_user"].astype("int64"), "is_fraud": preds})
    sub.to_csv("artifacts/submissions/submission_v3.csv", index=False)
    log(f"No improvement ({best_f1:.4f} vs {BASELINE}). Saved as submission_v3.csv")
