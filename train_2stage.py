"""Two-stage fraud detection: graph components + ML for mixed cases."""
import time, json, warnings, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict, deque

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


# ============================================================
# 1. LOAD DATA
# ============================================================
log("Loading data...")
tr_tx = pd.read_csv("train_transactions.csv")
tr_u = pd.read_csv("train_users.csv")
te_tx = pd.read_csv("test_transactions.csv")
te_u = pd.read_csv("test_users.csv")
all_tx = pd.concat([tr_tx, te_tx])

fraud_set = set(tr_u[tr_u["is_fraud"] == 1]["id_user"])
train_ids = set(tr_u["id_user"])
test_ids = set(te_u["id_user"])

# ============================================================
# 2. BUILD CARD GRAPH & FIND CONNECTED COMPONENTS
# ============================================================
log("Building card graph...")
card_to_users = defaultdict(set)
user_to_cards = defaultdict(set)
for uid, card in zip(all_tx["id_user"], all_tx["card_mask_hash"]):
    card_to_users[card].add(uid)
    user_to_cards[uid].add(card)

log("Finding connected components...")
all_users = train_ids | test_ids
visited = set()
components = []
for start in all_users:
    if start in visited:
        continue
    component = set()
    queue = deque([start])
    while queue:
        u = queue.popleft()
        if u in visited:
            continue
        visited.add(u)
        component.add(u)
        for card in user_to_cards.get(u, set()):
            for neighbor in card_to_users.get(card, set()):
                if neighbor not in visited:
                    queue.append(neighbor)
    components.append(component)

log(f"  Total components: {len(components)}")

# Classify components
pure_fraud_comps = []
pure_legit_comps = []
mixed_comps = []

for comp in components:
    train_in_comp = comp & train_ids
    fraud_in_comp = train_in_comp & fraud_set
    legit_in_comp = train_in_comp - fraud_set
    if len(fraud_in_comp) > 0 and len(legit_in_comp) == 0:
        pure_fraud_comps.append(comp)
    elif len(fraud_in_comp) == 0:
        pure_legit_comps.append(comp)
    else:
        mixed_comps.append(comp)

pure_fraud_users = set()
for c in pure_fraud_comps:
    pure_fraud_users |= c
pure_legit_users = set()
for c in pure_legit_comps:
    pure_legit_users |= c
mixed_users = set()
for c in mixed_comps:
    mixed_users |= c

train_pure_fraud = pure_fraud_users & train_ids
train_pure_legit = pure_legit_users & train_ids
train_mixed = mixed_users & train_ids
test_pure_fraud = pure_fraud_users & test_ids
test_pure_legit = pure_legit_users & test_ids
test_mixed = mixed_users & test_ids

log(f"  Train: {len(train_pure_fraud)} pure fraud, {len(train_pure_legit)} pure legit, {len(train_mixed)} mixed")
log(f"  Test:  {len(test_pure_fraud)} pure fraud, {len(test_pure_legit)} pure legit, {len(test_mixed)} mixed")

# Verify: pure fraud train users should all be fraud
pf_check = len(train_pure_fraud & fraud_set)
log(f"  Pure fraud check: {pf_check}/{len(train_pure_fraud)} are fraud")

# ============================================================
# 3. BUILD FEATURES FOR MIXED COMPONENT USERS
# ============================================================
log("Building features for mixed component users...")

# Parse timestamps
all_tx["ts"] = pd.to_datetime(all_tx["timestamp_tr"], format="ISO8601")
tr_u["ts_reg"] = pd.to_datetime(tr_u["timestamp_reg"], format="ISO8601")
te_u["ts_reg"] = pd.to_datetime(te_u["timestamp_reg"], format="ISO8601")

all_users_df = pd.concat([
    tr_u[["id_user", "email", "gender", "reg_country", "traffic_type", "ts_reg"]],
    te_u[["id_user", "email", "gender", "reg_country", "traffic_type", "ts_reg"]],
])

# Focus on ALL users (we still need features for all, ML handles mixed)
def build_features(user_ids, tx_df, users_df):
    """Build comprehensive features for a set of users."""
    user_list = sorted(user_ids)
    tx_sub = tx_df[tx_df["id_user"].isin(user_ids)].copy()
    tx_sub = tx_sub.sort_values(["id_user", "ts"])

    users_sub = users_df[users_df["id_user"].isin(user_ids)].set_index("id_user")

    feats = pd.DataFrame({"id_user": user_list})
    feats = feats.set_index("id_user")

    # Basic user features
    feats["gender"] = users_sub["gender"]
    feats["reg_country"] = users_sub["reg_country"]
    feats["traffic_type"] = users_sub["traffic_type"]
    feats["email_domain"] = users_sub["email"].str.split("@").str[1]
    feats["reg_hour"] = users_sub["ts_reg"].dt.hour
    feats["reg_weekday"] = users_sub["ts_reg"].dt.weekday

    # Transaction aggregates
    g = tx_sub.groupby("id_user")
    feats["tx_count"] = g.size().reindex(user_list, fill_value=0)
    feats["success_count"] = tx_sub[tx_sub.status == "success"].groupby("id_user").size().reindex(user_list, fill_value=0)
    feats["fail_count"] = tx_sub[tx_sub.status == "fail"].groupby("id_user").size().reindex(user_list, fill_value=0)
    feats["fail_ratio"] = feats["fail_count"] / feats["tx_count"].clip(lower=1)
    feats["success_ratio"] = feats["success_count"] / feats["tx_count"].clip(lower=1)

    # Amount features
    feats["amount_sum"] = g["amount"].sum().reindex(user_list, fill_value=0)
    feats["amount_mean"] = g["amount"].mean().reindex(user_list, fill_value=0)
    feats["amount_std"] = g["amount"].std().reindex(user_list, fill_value=0)
    feats["amount_max"] = g["amount"].max().reindex(user_list, fill_value=0)
    feats["amount_min"] = g["amount"].min().reindex(user_list, fill_value=0)
    feats["amount_range"] = feats["amount_max"] - feats["amount_min"]
    feats["amount_cv"] = feats["amount_std"] / feats["amount_mean"].clip(lower=0.01)

    # Card features
    feats["unique_cards"] = g["card_mask_hash"].nunique().reindex(user_list, fill_value=0)
    feats["unique_card_holders"] = g["card_holder"].nunique().reindex(user_list, fill_value=0)
    feats["unique_card_countries"] = g["card_country"].nunique().reindex(user_list, fill_value=0)
    feats["unique_card_brands"] = g["card_brand"].nunique().reindex(user_list, fill_value=0)
    feats["card_holder_per_card"] = feats["unique_card_holders"] / feats["unique_cards"].clip(lower=1)
    feats["cards_per_tx"] = feats["unique_cards"] / feats["tx_count"].clip(lower=1)
    feats["holders_per_tx"] = feats["unique_card_holders"] / feats["tx_count"].clip(lower=1)

    # Error features
    feats["fraud_error_count"] = tx_sub[tx_sub.error_group == "fraud"].groupby("id_user").size().reindex(user_list, fill_value=0)
    feats["antifraud_count"] = tx_sub[tx_sub.error_group == "antifraud"].groupby("id_user").size().reindex(user_list, fill_value=0)
    feats["fraud_error_ratio"] = feats["fraud_error_count"] / feats["tx_count"].clip(lower=1)
    feats["antifraud_ratio"] = feats["antifraud_count"] / feats["tx_count"].clip(lower=1)
    feats["unique_error_groups"] = g["error_group"].nunique().reindex(user_list, fill_value=0)

    # Transaction type features
    feats["unique_tx_types"] = g["transaction_type"].nunique().reindex(user_list, fill_value=0)
    feats["card_init_count"] = tx_sub[tx_sub.transaction_type == "card_init"].groupby("id_user").size().reindex(user_list, fill_value=0)
    feats["card_recurring_count"] = tx_sub[tx_sub.transaction_type == "card_recurring"].groupby("id_user").size().reindex(user_list, fill_value=0)
    feats["card_init_ratio"] = feats["card_init_count"] / feats["tx_count"].clip(lower=1)

    # Card type features
    feats["has_prepaid"] = tx_sub[tx_sub.card_type.str.contains("PREPAID", case=False, na=False)].groupby("id_user").size().reindex(user_list, fill_value=0).clip(upper=1)
    feats["has_credit"] = tx_sub[tx_sub.card_type.str.contains("CREDIT", case=False, na=False)].groupby("id_user").size().reindex(user_list, fill_value=0).clip(upper=1)

    # Country mismatch
    tx_sub["card_country_mismatch"] = (tx_sub["card_country"] != tx_sub["payment_country"]).astype(int)
    feats["country_mismatch_ratio"] = tx_sub.groupby("id_user")["card_country_mismatch"].mean().reindex(user_list, fill_value=0)

    # Card switch (different card from previous tx)
    tx_sub["prev_card"] = tx_sub.groupby("id_user")["card_mask_hash"].shift(1)
    tx_sub["card_switch"] = (tx_sub["card_mask_hash"] != tx_sub["prev_card"]).astype(int)
    tx_sub.loc[tx_sub["prev_card"].isna(), "card_switch"] = 0
    feats["card_switch_count"] = tx_sub.groupby("id_user")["card_switch"].sum().reindex(user_list, fill_value=0)
    feats["card_switch_ratio"] = feats["card_switch_count"] / feats["tx_count"].clip(lower=1)

    # Temporal features
    first_tx = g["ts"].min().reindex(user_list)
    last_tx = g["ts"].max().reindex(user_list)
    reg_ts = users_sub["ts_reg"].reindex(user_list)
    feats["minutes_to_first_tx"] = (first_tx - reg_ts).dt.total_seconds() / 60
    feats["tx_span_minutes"] = (last_tx - first_tx).dt.total_seconds() / 60

    # Time gaps
    tx_sub["gap_sec"] = tx_sub.groupby("id_user")["ts"].diff().dt.total_seconds()
    feats["mean_gap_sec"] = tx_sub.groupby("id_user")["gap_sec"].mean().reindex(user_list, fill_value=0)
    feats["min_gap_sec"] = tx_sub.groupby("id_user")["gap_sec"].min().reindex(user_list, fill_value=0)
    feats["log_mean_gap"] = np.log1p(feats["mean_gap_sec"].clip(lower=0))

    # Min gap for fail txs
    fail_tx = tx_sub[tx_sub.status == "fail"]
    feats["min_fail_gap_sec"] = fail_tx.groupby("id_user")["gap_sec"].min().reindex(user_list, fill_value=-1)

    # Time-of-day
    tx_sub["hour"] = tx_sub["ts"].dt.hour
    feats["mode_hour"] = tx_sub.groupby("id_user")["hour"].apply(lambda x: x.mode().iloc[0] if len(x) > 0 else 12).reindex(user_list, fill_value=12)
    feats["night_tx_ratio"] = tx_sub.assign(night=tx_sub.hour.isin([0,1,2,3,4,5]).astype(int)).groupby("id_user")["night"].mean().reindex(user_list, fill_value=0)
    feats["business_hours_ratio"] = tx_sub.assign(bh=tx_sub.hour.isin(range(9,18)).astype(int)).groupby("id_user")["bh"].mean().reindex(user_list, fill_value=0)

    # First hour/6h activity
    first_tx_per_user = first_tx.to_dict()
    tx_sub["since_first"] = tx_sub.apply(lambda r: (r["ts"] - first_tx_per_user.get(r["id_user"], r["ts"])).total_seconds() / 3600, axis=1)
    feats["tx_first_hour"] = tx_sub[tx_sub.since_first <= 1].groupby("id_user").size().reindex(user_list, fill_value=0)
    feats["fail_first_hour"] = tx_sub[(tx_sub.since_first <= 1) & (tx_sub.status == "fail")].groupby("id_user").size().reindex(user_list, fill_value=0)
    feats["tx_first_6h"] = tx_sub[tx_sub.since_first <= 6].groupby("id_user").size().reindex(user_list, fill_value=0)

    # Fail-then-success same amount pattern
    tx_sub["prev_status"] = tx_sub.groupby("id_user")["status"].shift(1)
    tx_sub["prev_amount"] = tx_sub.groupby("id_user")["amount"].shift(1)
    tx_sub["fail_then_success"] = ((tx_sub.prev_status == "fail") & (tx_sub.status == "success") & (tx_sub.amount == tx_sub.prev_amount)).astype(int)
    feats["fail_then_success_count"] = tx_sub.groupby("id_user")["fail_then_success"].sum().reindex(user_list, fill_value=0)

    # Max fail streak
    def max_fail_streak_fn(statuses):
        max_s = cur = 0
        for s in statuses:
            if s == "fail":
                cur += 1
                max_s = max(max_s, cur)
            else:
                cur = 0
        return max_s
    feats["max_fail_streak"] = g["status"].apply(max_fail_streak_fn).reindex(user_list, fill_value=0)

    # Small amounts ratio
    feats["small_amount_ratio"] = tx_sub.assign(small=(tx_sub.amount <= 5).astype(int)).groupby("id_user")["small"].mean().reindex(user_list, fill_value=0)

    # Amount per hour max
    tx_sub["tx_hour_key"] = tx_sub["ts"].dt.floor("h")
    hourly_amount = tx_sub.groupby(["id_user", "tx_hour_key"])["amount"].sum().reset_index()
    feats["amount_per_hour_max"] = hourly_amount.groupby("id_user")["amount"].max().reindex(user_list, fill_value=0)

    # Log transforms
    for c in ["tx_count", "amount_sum", "card_switch_count", "minutes_to_first_tx"]:
        if c in feats.columns:
            feats[f"log_{c}"] = np.log1p(feats[c].clip(lower=0))

    # Risk combo
    feats["risk_combo"] = (
        (feats["card_switch_count"] > 3).astype(int)
        + (feats["unique_card_holders"] > 1).astype(int)
        + (feats["fail_ratio"] > 0.5).astype(int)
        + (feats["country_mismatch_ratio"] > 0).astype(int)
    )

    # GRAPH FEATURES within mixed component
    for uid in user_list:
        cards = user_to_cards.get(uid, set())
        neighbors = set()
        fraud_neighbors = set()
        for card in cards:
            others = card_to_users.get(card, set()) - {uid}
            neighbors |= others
            fraud_neighbors |= (others & fraud_set)
        n_total = len(neighbors)
        n_fraud = len(fraud_neighbors)
        feats.at[uid, "graph_n_neighbors"] = n_total
        feats.at[uid, "graph_n_fraud_neighbors"] = n_fraud
        feats.at[uid, "graph_fraud_ratio"] = n_fraud / max(n_total, 1)
        # Fraction of user's cards that have fraud
        n_fraud_cards = sum(1 for c in cards if (card_to_users.get(c, set()) - {uid}) & fraud_set)
        feats.at[uid, "graph_fraud_cards"] = n_fraud_cards
        feats.at[uid, "graph_fraud_card_ratio"] = n_fraud_cards / max(len(cards), 1)
        # Max density on any card
        max_density = 0.0
        for card in cards:
            others = card_to_users.get(card, set()) - {uid}
            if others:
                density = len(others & fraud_set) / len(others)
                max_density = max(max_density, density)
        feats.at[uid, "graph_max_card_density"] = max_density

    feats = feats.reset_index()
    return feats


# Build features for ALL train and test users
log("Building features for all users...")
train_feats = build_features(train_ids, all_tx, all_users_df)
log(f"  Train features: {train_feats.shape}")
test_feats = build_features(test_ids, all_tx, all_users_df)
log(f"  Test features: {test_feats.shape}")

# Add target
train_feats = train_feats.merge(tr_u[["id_user", "is_fraud"]], on="id_user")

# ============================================================
# 4. TARGET ENCODING
# ============================================================
log("Target encoding...")
y_all = train_feats["is_fraud"].values
gm = y_all.mean()
smoothing = 50
te_cols_to_encode = ["reg_country", "gender", "traffic_type", "email_domain"]

for te_col in te_cols_to_encode:
    skf_te = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_te_vals = np.full(len(train_feats), gm)
    col_data = train_feats[te_col].fillna("missing").astype(str)
    y_s = pd.Series(y_all)
    for tr_i, val_i in skf_te.split(train_feats, y_all):
        stats = pd.DataFrame({"c": col_data.iloc[tr_i], "y": y_s.iloc[tr_i]}).groupby("c")["y"].agg(["mean", "count"])
        s = (stats["count"] * stats["mean"] + smoothing * gm) / (stats["count"] + smoothing)
        train_te_vals[val_i] = col_data.iloc[val_i].map(s).fillna(gm).values
    # Full stats for test
    stats_all = pd.DataFrame({"c": col_data, "y": y_s}).groupby("c")["y"].agg(["mean", "count"])
    s_all = (stats_all["count"] * stats_all["mean"] + smoothing * gm) / (stats_all["count"] + smoothing)
    test_te_vals = test_feats[te_col].fillna("missing").astype(str).map(s_all).fillna(gm).values
    train_feats[f"{te_col}_te"] = train_te_vals
    test_feats[f"{te_col}_te"] = test_te_vals

# ============================================================
# 5. PREPARE FOR TRAINING
# ============================================================
EXCLUDED = {"id_user", "is_fraud", "email", "ts_reg"}
cat_cols = ["gender", "reg_country", "traffic_type", "email_domain"]
feature_cols = [c for c in train_feats.columns if c not in EXCLUDED]
num_cols = [c for c in feature_cols if c not in cat_cols]

log(f"Total features: {len(feature_cols)}, Cat: {len(cat_cols)}, Num: {len(num_cols)}")

# Prepare data
tr = train_feats.copy()
te = test_feats.copy()
for col in cat_cols:
    vals = pd.concat([tr[col], te[col]]).fillna("missing").astype(str)
    cats = sorted(vals.unique().tolist())
    tr[col] = pd.Categorical(tr[col].fillna("missing").astype(str), categories=cats)
    te[col] = pd.Categorical(te[col].fillna("missing").astype(str), categories=cats)
for col in num_cols:
    fill = float(tr[col].median()) if not tr[col].dropna().empty else 0.0
    tr[col] = tr[col].fillna(fill).astype(float)
    te[col] = te[col].fillna(fill).astype(float)

# Component type labels
tr["comp_type"] = 0  # mixed
tr.loc[tr["id_user"].isin(train_pure_fraud), "comp_type"] = 1  # pure fraud
tr.loc[tr["id_user"].isin(train_pure_legit), "comp_type"] = -1  # pure legit

te["comp_type"] = 0
te.loc[te["id_user"].isin(test_pure_fraud), "comp_type"] = 1
te.loc[te["id_user"].isin(test_pure_legit), "comp_type"] = -1

# ============================================================
# 6. TWO-STAGE PREDICTION
# ============================================================
log("=== STAGE 1: Component-based classification ===")
y = train_feats["is_fraud"].values

# Stage 1: auto-classify pure components
train_stage1_pred = np.full(len(tr), -1.0)  # -1 = unknown
test_stage1_pred = np.full(len(te), -1.0)

# Pure fraud = 1.0, Pure legit = 0.0
train_stage1_pred[tr["comp_type"] == 1] = 1.0
train_stage1_pred[tr["comp_type"] == -1] = 0.0
test_stage1_pred[te["comp_type"] == 1] = 1.0
test_stage1_pred[te["comp_type"] == -1] = 0.0

n_auto_fraud_train = (train_stage1_pred == 1.0).sum()
n_auto_legit_train = (train_stage1_pred == 0.0).sum()
n_mixed_train = (train_stage1_pred == -1.0).sum()
log(f"  Train: {n_auto_fraud_train} auto-fraud, {n_auto_legit_train} auto-legit, {n_mixed_train} need ML")

n_auto_fraud_test = (test_stage1_pred == 1.0).sum()
n_auto_legit_test = (test_stage1_pred == 0.0).sum()
n_mixed_test = (test_stage1_pred == -1.0).sum()
log(f"  Test:  {n_auto_fraud_test} auto-fraud, {n_auto_legit_test} auto-legit, {n_mixed_test} need ML")

# Stage 1 OOF F1 (on train, auto-classified only)
auto_mask = train_stage1_pred >= 0
auto_preds = train_stage1_pred[auto_mask].astype(int)
auto_true = y[auto_mask]
from sklearn.metrics import f1_score, precision_score, recall_score
f1_auto = f1_score(auto_true, auto_preds)
log(f"  Stage 1 auto F1: {f1_auto:.4f} (P={precision_score(auto_true, auto_preds):.4f}, R={recall_score(auto_true, auto_preds):.4f})")

# ============================================================
# 7. STAGE 2: ML FOR MIXED COMPONENTS
# ============================================================
log("\n=== STAGE 2: ML for mixed component users ===")
mixed_train_mask = tr["comp_type"] == 0
mixed_test_mask = te["comp_type"] == 0

X_mixed_train = tr.loc[mixed_train_mask, feature_cols]
y_mixed_train = y[mixed_train_mask.values]
X_mixed_test = te.loc[mixed_test_mask, feature_cols]

log(f"  Mixed train: {len(X_mixed_train)} ({y_mixed_train.mean()*100:.1f}% fraud)")
log(f"  Mixed test: {len(X_mixed_test)}")

# Train ensemble on mixed users
from scipy.stats import rankdata

configs = [
    ("gbdt_7f_42", 7, 42, "lgb", {"lr": 0.02, "leaves": 63, "depth": -1, "sub": 0.8, "col": 0.7}),
    ("gbdt_5f_42", 5, 42, "lgb", {"lr": 0.02, "leaves": 63, "depth": -1, "sub": 0.8, "col": 0.7}),
    ("gbdt_7f_123", 7, 123, "lgb", {"lr": 0.02, "leaves": 63, "depth": -1, "sub": 0.8, "col": 0.7}),
    ("gbdt_5f_123", 5, 123, "lgb", {"lr": 0.02, "leaves": 63, "depth": -1, "sub": 0.8, "col": 0.7}),
    ("gbdt_7f_999", 7, 999, "lgb", {"lr": 0.015, "leaves": 95, "depth": 8, "sub": 0.75, "col": 0.65}),
]

all_oofs = []
all_tests = []
model_labels = []

for name, n_folds, seed, mtype, params in configs:
    log(f"  === {name} ===")
    skf_m = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(X_mixed_train))
    tp_arr = np.zeros(len(X_mixed_test))

    for fold, (tr_idx, val_idx) in enumerate(skf_m.split(X_mixed_train, y_mixed_train)):
        m = lgb.LGBMClassifier(
            objective="binary", n_estimators=5000, learning_rate=params["lr"],
            num_leaves=params["leaves"], max_depth=params["depth"],
            min_child_samples=50, subsample=params["sub"], colsample_bytree=params["col"],
            reg_alpha=0.1, reg_lambda=1.0, random_state=seed, n_jobs=-1, verbosity=-1,
        )
        m.fit(
            X_mixed_train.iloc[tr_idx], y_mixed_train[tr_idx],
            eval_set=[(X_mixed_train.iloc[val_idx], y_mixed_train[val_idx])],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(200, verbose=False)],
            categorical_feature=cat_cols,
        )
        oof[val_idx] = m.predict_proba(X_mixed_train.iloc[val_idx])[:, 1]
        tp_arr += m.predict_proba(X_mixed_test)[:, 1] / n_folds

    t_, f1_ = find_best_threshold(y_mixed_train, oof)
    log(f"    F1={f1_:.4f}")
    all_oofs.append(oof)
    all_tests.append(tp_arr)
    model_labels.append(name)

    if name == configs[0][0]:
        imp = dict(zip(feature_cols, m.feature_importances_))
        top_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:20]
        log("    Top-20 features:")
        for fn, fv in top_imp:
            log(f"      {fn}: {fv}")

# Blend
n_models = len(all_oofs)
avg_oof = np.mean(all_oofs, axis=0)
avg_test = np.mean(all_tests, axis=0)
t_eq, f1_eq = find_best_threshold(y_mixed_train, avg_oof)
log(f"  Equal blend (mixed): F1={f1_eq:.4f}")

# Greedy weight optimization
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
            _, f1_b = find_best_threshold(y_mixed_train, blend)
            if f1_b > best_blend_f1:
                best_blend_f1 = f1_b
                best_weights = w.copy()
                improved = True
    if not improved:
        break

log(f"  Optimized blend (mixed): F1={best_blend_f1:.4f}")

# Final mixed predictions
mixed_oof_final = sum(best_weights[j] * all_oofs[j] for j in range(n_models))
mixed_test_final = sum(best_weights[j] * all_tests[j] for j in range(n_models))
t_mixed, _ = find_best_threshold(y_mixed_train, mixed_oof_final)

# ============================================================
# 8. COMBINE STAGE 1 + STAGE 2
# ============================================================
log("\n=== COMBINING STAGES ===")

# Train OOF: combine auto + ML predictions
train_oof_combined = np.zeros(len(y))
# Pure fraud → 1
train_oof_combined[tr["comp_type"].values == 1] = 1.0
# Pure legit → 0
train_oof_combined[tr["comp_type"].values == -1] = 0.0
# Mixed → ML prediction
train_oof_combined[mixed_train_mask.values] = mixed_oof_final

# Find optimal threshold on full train OOF
t_final, f1_final = find_best_threshold(y, train_oof_combined)
log(f"Combined OOF F1: {f1_final:.4f} (threshold={t_final:.4f})")

# Also try: for mixed, sweep threshold independently
best_overall_f1 = 0
best_mixed_t = t_mixed
for t_try in np.arange(0.05, 0.95, 0.01):
    combined_pred = np.zeros(len(y))
    combined_pred[tr["comp_type"].values == 1] = 1
    combined_pred[tr["comp_type"].values == -1] = 0
    combined_pred[mixed_train_mask.values] = (mixed_oof_final >= t_try).astype(int)
    f1_try = f1_score(y, combined_pred)
    if f1_try > best_overall_f1:
        best_overall_f1 = f1_try
        best_mixed_t = t_try

log(f"Best combined F1: {best_overall_f1:.4f} (mixed threshold={best_mixed_t:.4f})")

# ============================================================
# 9. GENERATE SUBMISSION
# ============================================================
log("\n=== SUBMISSION ===")
test_preds = np.zeros(len(te))
test_preds[te["comp_type"].values == 1] = 1
test_preds[te["comp_type"].values == -1] = 0
test_preds[mixed_test_mask.values] = (mixed_test_final >= best_mixed_t).astype(int)
test_preds = test_preds.astype(int)

log(f"Test predictions: {test_preds.sum()} fraud ({test_preds.mean()*100:.2f}%)")
log(f"  Auto fraud: {n_auto_fraud_test}")
log(f"  Auto legit: {n_auto_legit_test}")
log(f"  ML fraud: {test_preds[mixed_test_mask.values].sum()}")

sub = pd.DataFrame({"id_user": te_u["id_user"].astype("int64"), "is_fraud": test_preds})
sub.to_csv("artifacts/submissions/submission.csv", index=False)
log(f"Saved submission.csv")

metrics = {
    "oof_f1": round(best_overall_f1, 6),
    "approach": "two_stage_components_ml",
    "predicted_fraud": int(test_preds.sum()),
    "predicted_fraud_rate": round(float(test_preds.mean()), 6),
    "mixed_f1": round(best_blend_f1, 6),
    "auto_f1": round(f1_auto, 6),
    "mixed_threshold": round(best_mixed_t, 4),
}
with open("artifacts/reports/baseline_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
log(f"FINAL F1: {best_overall_f1:.4f}")
