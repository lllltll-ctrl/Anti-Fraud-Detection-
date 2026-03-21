"""Train LightGBM/DART with card-graph fraud propagation features."""
import time, json, warnings, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

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
# 1. BUILD CARD GRAPH
# ============================================================
log("Loading transactions...")
tr_tx = pd.read_csv("train_transactions.csv")
tr_u = pd.read_csv("train_users.csv")
te_tx = pd.read_csv("test_transactions.csv")
te_u = pd.read_csv("test_users.csv")

fraud_users = set(tr_u[tr_u["is_fraud"] == 1]["id_user"])
all_tx = pd.concat([tr_tx, te_tx])

log("Building card graph...")
user_cards = all_tx.groupby("id_user")["card_mask_hash"].apply(set).to_dict()
card_users = all_tx.groupby("card_mask_hash")["id_user"].apply(set).to_dict()

# Also build card_holder graph
user_holders = all_tx.groupby("id_user")["card_holder"].apply(lambda x: set(x.dropna().str.lower())).to_dict()
holder_users = {}
for uid, holders in user_holders.items():
    for h in holders:
        if h not in holder_users:
            holder_users[h] = set()
        holder_users[h].add(uid)

y = tr_u["is_fraud"].values


def get_graph_features(uid, known_fraud, user_cards_map, card_users_map, user_holders_map, holder_users_map):
    """Get graph-based features for a user."""
    cards = user_cards_map.get(uid, set())
    holders = user_holders_map.get(uid, set())

    # Card graph
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

    # Holder graph
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
        n_card_fraud,
        n_card_total,
        n_card_fraud / max(n_card_total, 1),
        max_card_density,
        n_fraud_cards,
        n_fraud_cards / max(len(cards), 1),
        n_holder_fraud,
        n_holder_total,
        n_holder_fraud / max(n_holder_total, 1),
        # Combined: any fraud connection
        int(n_card_fraud > 0 or n_holder_fraud > 0),
        # Total fraud connections
        n_card_fraud + n_holder_fraud,
    ]


GRAPH_FEAT_NAMES = [
    "graph_card_fraud_n", "graph_card_total_n", "graph_card_fraud_ratio",
    "graph_card_max_density", "graph_n_fraud_cards", "graph_fraud_card_ratio",
    "graph_holder_fraud_n", "graph_holder_total_n", "graph_holder_fraud_ratio",
    "graph_any_fraud_connection", "graph_total_fraud_connections",
]

# ============================================================
# 2. K-FOLD GRAPH FEATURES FOR TRAIN (no leakage)
# ============================================================
log("Building K-fold graph features for train...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_graph = np.zeros((len(tr_u), len(GRAPH_FEAT_NAMES)))

for fold, (tr_idx, val_idx) in enumerate(skf.split(tr_u, y)):
    fold_fraud = set(tr_u.iloc[tr_idx][tr_u.iloc[tr_idx]["is_fraud"] == 1]["id_user"])
    for i in val_idx:
        uid = tr_u.iloc[i]["id_user"]
        train_graph[i] = get_graph_features(uid, fold_fraud, user_cards, card_users, user_holders, holder_users)
    log(f"  Fold {fold} done")

# ============================================================
# 3. GRAPH FEATURES FOR TEST (use all train labels)
# ============================================================
log("Building graph features for test...")
test_graph = np.zeros((len(te_u), len(GRAPH_FEAT_NAMES)))
for i, uid in enumerate(te_u["id_user"].values):
    test_graph[i] = get_graph_features(uid, fraud_users, user_cards, card_users, user_holders, holder_users)
log(f"  Done. Test users with fraud connection: {(test_graph[:, 9] > 0).sum()}")

# ============================================================
# 4. MERGE WITH EXISTING FEATURES
# ============================================================
log("Merging features...")
train_df = pd.read_csv("artifacts/processed/train_features.csv")
test_df = pd.read_csv("artifacts/processed/test_features.csv")

for j, name in enumerate(GRAPH_FEAT_NAMES):
    train_df[name] = train_graph[:, j]
    test_df[name] = test_graph[:, j]

# Target encoding
gm = y.mean()
smoothing = 50
skf_te = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_te = np.full(len(train_df), gm)
y_s = pd.Series(y)
for tr_i, val_i in skf_te.split(train_df, y):
    col = train_df["reg_country"].fillna("missing")
    stats = pd.DataFrame({"c": col.iloc[tr_i], "y": y_s.iloc[tr_i]}).groupby("c")["y"].agg(["mean", "count"])
    s = (stats["count"] * stats["mean"] + smoothing * gm) / (stats["count"] + smoothing)
    train_te[val_i] = col.iloc[val_i].map(s).fillna(gm).values
stats_all = pd.DataFrame({"c": train_df["reg_country"].fillna("missing"), "y": y_s}).groupby("c")["y"].agg(["mean", "count"])
s_all = (stats_all["count"] * stats_all["mean"] + smoothing * gm) / (stats_all["count"] + smoothing)
test_te = test_df["reg_country"].fillna("missing").map(s_all).fillna(gm).values
train_df["reg_country_te"] = train_te
test_df["reg_country_te"] = test_te


def add_feats(df):
    o = df.copy()
    tx = o["tx_count"].clip(lower=1)
    o["cards_per_tx"] = o["unique_cards"] / tx
    o["holders_per_tx"] = o["unique_card_holders"] / tx
    o["switches_per_card"] = o["card_switch_count"] / o["unique_cards"].clip(lower=1)
    fh = o["tx_first_hour"].clip(lower=1)
    o["fail_ratio_first_hour"] = o["fail_first_hour"] / fh
    o["tx_first_6h_ratio"] = o["tx_first_6h"] / tx
    o["multi_holder_switch"] = (o["unique_card_holders"] > 1).astype(int) * o["card_switch_count"]
    o["country_mismatch_total"] = o["card_country_mismatch_count"] + o["payment_country_mismatch_count"]
    o["amount_cv"] = o["amount_std"] / o["amount_mean"].clip(lower=0.01)
    o["small_amount_ratio"] = o["small_amount_count"] / tx
    o["fail_streak_per_tx"] = o["max_fail_streak"] / tx
    o["fast_starter"] = (o["minutes_to_first_tx"] < 60).astype(int)
    for c in ["tx_count", "amount_sum", "card_switch_count", "minutes_to_first_tx"]:
        o[f"log_{c}"] = np.log1p(o[c].clip(lower=0))
    o["risk_combo"] = (
        (o["card_switch_count"] > 3).astype(int)
        + (o["unique_card_holders"] > 1).astype(int)
        + (o["fail_ratio"] > 0.5).astype(int)
        + (o["country_mismatch_total"] > 0).astype(int)
    )
    return o


train_df = add_feats(train_df)
test_df = add_feats(test_df)

EXCLUDED = {"id_user", "timestamp_reg", "email", "is_fraud"}
feature_cols = [c for c in train_df.columns if c not in EXCLUDED]
cat_cols = [c for c in feature_cols if pd.api.types.is_string_dtype(train_df[c]) or pd.api.types.is_object_dtype(train_df[c])]
num_cols = [c for c in feature_cols if c not in cat_cols]

log(f"Features: {len(feature_cols)}, Cat: {len(cat_cols)}, Num: {len(num_cols)}")

tr, te = train_df.copy(), test_df.copy()
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
# 5. TRAIN MULTI-CONFIG ENSEMBLE
# ============================================================
from scipy.stats import rankdata

all_oofs = []
all_tests = []
labels = []

configs = [
    # (name, n_folds, seed, model_type, params)
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
            oof[val_idx] = m.predict_proba(tr[feature_cols].iloc[val_idx])[:, 1]
            tp_arr += m.predict_proba(te[feature_cols])[:, 1] / n_folds
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

    # Print graph feature importances (first GBDT model)
    if name == "gbdt_7f_42":
        imp = dict(zip(feature_cols, m.feature_importances_))
        log("  Graph feature importances:")
        for gf in GRAPH_FEAT_NAMES:
            log(f"    {gf}: {imp.get(gf, 0)}")

# ============================================================
# 6. BLEND
# ============================================================
log("\n=== BLEND ===")
n_models = len(all_oofs)

# Equal blend
avg_oof = np.mean(all_oofs, axis=0)
avg_test = np.mean(all_tests, axis=0)
t_eq, f1_eq = find_best_threshold(y, avg_oof)
log(f"Equal blend: F1={f1_eq:.4f}")

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

# Rank blend
rank_oof = np.mean([rankdata(o) for o in all_oofs], axis=0)
rank_test = np.mean([rankdata(t_) for t_ in all_tests], axis=0)
t_rk, f1_rk = find_best_threshold(y, rank_oof)
log(f"Rank blend: F1={f1_rk:.4f}")

# Top-k subsets
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

# Final: pick best approach
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

BASELINE = 0.6661
if best_f1 > BASELINE:
    sub = pd.DataFrame({"id_user": test_df["id_user"].astype("int64"), "is_fraud": preds})
    sub.to_csv("artifacts/submissions/submission.csv", index=False)
    log(f"IMPROVED! Saved submission.csv")
    metrics = {
        "oof_f1": round(best_f1, 6),
        "approach": f"graph_{approach}_{n_models}models",
        "predicted_fraud": int(preds.sum()),
        "predicted_fraud_rate": round(float(preds.mean()), 6),
    }
    with open("artifacts/reports/baseline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
else:
    sub = pd.DataFrame({"id_user": test_df["id_user"].astype("int64"), "is_fraud": preds})
    sub.to_csv("artifacts/submissions/submission_graph_v2.csv", index=False)
    log(f"No improvement ({best_f1:.4f} vs {BASELINE}). Saved as submission_graph_v2.csv")
