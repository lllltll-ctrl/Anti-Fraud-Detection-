"""Best blend: original features (from try_dart_7fold) + diverse models + weight optimization."""
import json, warnings, numpy as np, pandas as pd, lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from scipy.stats import rankdata
warnings.filterwarnings("ignore")

train_df = pd.read_csv("artifacts/processed/train_features.csv")
test_df = pd.read_csv("artifacts/processed/test_features.csv")

y_s = train_df["is_fraud"].astype(int)
y = y_s.values
gm = y_s.mean()
smoothing = 50

# Target encoding (same as try_dart_7fold)
skf_te = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_te = np.full(len(train_df), gm)
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

# Advanced features (same as try_dart_7fold)
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

print(f"Features: {len(feature_cols)}, Cat: {len(cat_cols)}, Num: {len(num_cols)}", flush=True)

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

# CatBoost prep
tr_cb, te_cb = train_df.copy(), test_df.copy()
for col in cat_cols:
    tr_cb[col] = tr_cb[col].fillna("missing").astype(str)
    te_cb[col] = te_cb[col].fillna("missing").astype(str)
for col in num_cols:
    fill = float(tr_cb[col].median()) if not tr_cb[col].dropna().empty else 0.0
    tr_cb[col] = tr_cb[col].fillna(fill).astype(float)
    te_cb[col] = te_cb[col].fillna(fill).astype(float)
cat_indices = [feature_cols.index(c) for c in cat_cols]


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
    ("cb_7f_42", 7, 42, "cb", {}),
    ("cb_5f_123", 5, 123, "cb", {}),
]

for name, n_folds, seed, mtype, params in configs:
    print(f"\n=== {name} ===", flush=True)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    tp = np.zeros(len(te))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(tr, y)):
        if mtype == "lgb":
            m = lgb.LGBMClassifier(
                objective="binary", n_estimators=5000, learning_rate=params["lr"],
                num_leaves=params["leaves"], max_depth=params["depth"],
                min_child_samples=50, subsample=params["sub"], colsample_bytree=params["col"],
                reg_alpha=0.1, reg_lambda=1.0, random_state=seed, n_jobs=-1, verbosity=-1,
            )
            m.fit(tr[feature_cols].iloc[tr_idx], y[tr_idx],
                  eval_set=[(tr[feature_cols].iloc[val_idx], y[val_idx])],
                  eval_metric="binary_logloss",
                  callbacks=[lgb.early_stopping(200, verbose=False)],
                  categorical_feature=cat_cols)
            oof[val_idx] = m.predict_proba(tr[feature_cols].iloc[val_idx])[:, 1]
            tp += m.predict_proba(te[feature_cols])[:, 1] / n_folds
        elif mtype == "dart":
            m = lgb.LGBMClassifier(
                boosting_type="dart", objective="binary", n_estimators=1000,
                learning_rate=params["lr"], num_leaves=params["leaves"], max_depth=params["depth"],
                min_child_samples=50, subsample=params["sub"], colsample_bytree=params["col"],
                reg_alpha=0.1, reg_lambda=1.0, drop_rate=0.1, skip_drop=0.5,
                random_state=seed, n_jobs=-1, verbosity=-1,
            )
            m.fit(tr[feature_cols].iloc[tr_idx], y[tr_idx],
                  eval_set=[(tr[feature_cols].iloc[val_idx], y[val_idx])],
                  eval_metric="binary_logloss",
                  callbacks=[lgb.early_stopping(50, verbose=False)],
                  categorical_feature=cat_cols)
            oof[val_idx] = m.predict_proba(tr[feature_cols].iloc[val_idx])[:, 1]
            tp += m.predict_proba(te[feature_cols])[:, 1] / n_folds
        elif mtype == "cb":
            m = CatBoostClassifier(
                iterations=2000, learning_rate=0.05, depth=6,
                l2_leaf_reg=3, random_seed=seed, verbose=0,
                early_stopping_rounds=100, cat_features=cat_indices,
                eval_metric="Logloss", task_type="CPU",
            )
            m.fit(tr_cb[feature_cols].iloc[tr_idx], y[tr_idx],
                  eval_set=(tr_cb[feature_cols].iloc[val_idx], y[val_idx]),
                  verbose=0)
            oof[val_idx] = m.predict_proba(tr_cb[feature_cols].iloc[val_idx])[:, 1]
            tp += m.predict_proba(te_cb[feature_cols])[:, 1] / n_folds

    t_, f1_ = find_best_threshold(y, oof)
    print(f"  F1={f1_:.4f}", flush=True)
    all_oofs.append(oof)
    all_tests.append(tp)
    labels.append(name)

# === Blending ===
print("\n=== BLEND RESULTS ===", flush=True)
n_models = len(all_oofs)
for i, lbl in enumerate(labels):
    t_, f1_ = find_best_threshold(y, all_oofs[i])
    print(f"  {lbl}: F1={f1_:.4f}")

# Equal average
avg_oof = np.mean(all_oofs, axis=0)
avg_test = np.mean(all_tests, axis=0)
t_eq, f1_eq = find_best_threshold(y, avg_oof)
print(f"\nEqual avg: F1={f1_eq:.4f}")

# Rank blend
rank_oof = np.mean([rankdata(o) for o in all_oofs], axis=0)
rank_test = np.mean([rankdata(t_) for t_ in all_tests], axis=0)
t_rk, f1_rk = find_best_threshold(y, rank_oof)
print(f"Rank avg: F1={f1_rk:.4f}")

# Greedy weight optimization
print("\nOptimizing weights...", flush=True)
best_weights = np.ones(n_models) / n_models
best_blend_f1 = f1_eq

for round_ in range(5):
    improved = False
    for i in range(n_models):
        for delta in [-0.15, -0.1, -0.05, -0.02, 0.02, 0.05, 0.1, 0.15]:
            w = best_weights.copy()
            w[i] = max(0, w[i] + delta)
            if w.sum() == 0:
                continue
            w = w / w.sum()
            blend = sum(w[j] * all_oofs[j] for j in range(n_models))
            _, f1_b = find_best_threshold(y, blend)
            if f1_b > best_blend_f1:
                best_blend_f1 = f1_b
                best_weights = w.copy()
                improved = True
    if not improved:
        break

print(f"Optimized weights:")
for i, lbl in enumerate(labels):
    if best_weights[i] > 0.001:
        print(f"  {lbl}: {best_weights[i]:.3f}")
print(f"Optimized blend F1: {best_blend_f1:.4f}")

# Also try: top-k models only
print("\nTrying top-k subsets...", flush=True)
individual_f1s = []
for i in range(n_models):
    _, f1_ = find_best_threshold(y, all_oofs[i])
    individual_f1s.append(f1_)

sorted_idx = sorted(range(n_models), key=lambda x: individual_f1s[x], reverse=True)
for k in [3, 4, 5, 6, 7]:
    if k > n_models:
        break
    top_idx = sorted_idx[:k]
    sub_oof = np.mean([all_oofs[i] for i in top_idx], axis=0)
    _, f1_k = find_best_threshold(y, sub_oof)
    print(f"  Top-{k}: F1={f1_k:.4f} ({', '.join(labels[i] for i in top_idx)})")

# Final: pick best
best_f1 = max(f1_eq, f1_rk, best_blend_f1)
if best_blend_f1 >= max(f1_eq, f1_rk):
    final_oof = sum(best_weights[j] * all_oofs[j] for j in range(n_models))
    final_test = sum(best_weights[j] * all_tests[j] for j in range(n_models))
    approach = "optimized_blend"
elif f1_eq >= f1_rk:
    final_oof = avg_oof
    final_test = avg_test
    approach = "mean_blend"
else:
    final_oof = rank_oof
    final_test = rank_test
    approach = "rank_blend"

t_final, _ = find_best_threshold(y, final_oof)
print(f"\nBest: {approach}, F1={best_f1:.4f}")

BASELINE = 0.6274
if best_f1 > BASELINE:
    preds = (final_test >= t_final).astype(int)
    sub = pd.DataFrame({"id_user": test_df["id_user"].astype("int64"), "is_fraud": preds})
    sub.to_csv("artifacts/submissions/submission.csv", index=False)
    print(f"IMPROVED! Saved. F1={best_f1:.4f}, fraud={preds.sum()} ({preds.mean()*100:.2f}%)")
    metrics = {
        "oof_f1": round(best_f1, 6),
        "approach": f"best_{approach}_{n_models}models",
        "predicted_fraud": int(preds.sum()),
        "predicted_fraud_rate": round(float(preds.mean()), 6),
    }
    with open("artifacts/reports/baseline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
else:
    preds = (final_test >= t_final).astype(int)
    sub = pd.DataFrame({"id_user": test_df["id_user"].astype("int64"), "is_fraud": preds})
    sub.to_csv("artifacts/submissions/submission_v6.csv", index=False)
    print(f"No improvement ({best_f1:.4f} vs {BASELINE}). Saved as v6.")
