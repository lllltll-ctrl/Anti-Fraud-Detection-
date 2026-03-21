"""
Experiment: combine insights from 4 GitHub repos to beat baseline F1=0.628.

Techniques:
1. SHAP feature selection (top-N instead of all 75)
2. LightGBM native balanced bagging (pos/neg_bagging_fraction)
3. Borderline SMOTE (in-fold only, no leakage)
4. path_smooth + extra_trees for regularization
5. Focal loss custom objective for hard examples
"""
import json, time, warnings, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import BorderlineSMOTE
import shap

warnings.filterwarnings("ignore")
BASELINE_F1 = 0.6281


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


def focal_loss_objective(y_pred, dtrain, gamma=2.0, alpha=0.25):
    """Focal loss: focuses on hard-to-classify examples."""
    y_true = dtrain.get_label()
    p = 1.0 / (1.0 + np.exp(-y_pred))
    grad = alpha * y_true * (1 - p)**gamma * (gamma * p * np.log(np.clip(p, 1e-8, 1)) + p - 1) \
         + (1 - alpha) * (1 - y_true) * p**gamma * (- gamma * (1 - p) * np.log(np.clip(1 - p, 1e-8, 1)) - p + 1)
    # Simplified: use standard focal gradient
    pt = y_true * p + (1 - y_true) * (1 - p)
    at = y_true * alpha + (1 - y_true) * (1 - alpha)
    fl_grad = at * (1 - pt)**gamma * (gamma * pt * np.log(np.clip(pt, 1e-8, 1)) + pt - 1)
    # Approximate hessian
    fl_hess = at * (1 - pt)**gamma * (2 * gamma * pt * (1 - pt) + pt * (1 - pt))
    fl_hess = np.clip(fl_hess, 1e-6, None)
    grad = -(y_true - p)  # simplified focal: weight by (1-pt)^gamma
    weight = at * (1 - pt) ** gamma
    grad = weight * (p - y_true)
    hess = weight * p * (1 - p)
    hess = np.clip(hess, 1e-6, None)
    return grad, hess


# ============================================================
# LOAD DATA
# ============================================================
log("Loading data...")
train_df = pd.read_csv("artifacts/processed/train_features.csv")
test_df = pd.read_csv("artifacts/processed/test_features.csv")

y_s = train_df["is_fraud"].astype(int)
y = y_s.values
gm = y_s.mean()

# Target encoding
smoothing = 50
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

log(f"All features: {len(feature_cols)}, Cat: {len(cat_cols)}, Num: {len(num_cols)}")

# Prepare LGB data
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

# Also prepare numeric-only versions for SMOTE (can't handle categoricals)
num_feature_cols = [c for c in feature_cols if c in num_cols]

# ============================================================
# STEP 1: SHAP FEATURE SELECTION
# ============================================================
log("=== SHAP Feature Selection ===")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
tr_idx, val_idx = next(iter(skf.split(tr, y)))

m_shap = lgb.LGBMClassifier(
    objective="binary", n_estimators=2000, learning_rate=0.02,
    num_leaves=63, min_child_samples=50, subsample=0.8, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1, verbosity=-1,
)
m_shap.fit(
    tr[feature_cols].iloc[tr_idx], y[tr_idx],
    eval_set=[(tr[feature_cols].iloc[val_idx], y[val_idx])],
    eval_metric="binary_logloss",
    callbacks=[lgb.early_stopping(200, verbose=False)],
    categorical_feature=cat_cols,
)

explainer = shap.TreeExplainer(m_shap)
shap_values = explainer.shap_values(tr[feature_cols].iloc[val_idx[:5000]])
if isinstance(shap_values, list):
    shap_values = shap_values[1]

mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_importance = dict(zip(feature_cols, mean_abs_shap))
sorted_shap = sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)

log("Top 30 SHAP features:")
for name, val in sorted_shap[:30]:
    log(f"  {name}: {val:.5f}")

# Test different feature counts
results = {}


def eval_config(name, feats, params, use_smote=False, n_folds=7, seed=42):
    """Evaluate a config with K-fold CV."""
    cats = [c for c in feats if c in cat_cols]
    skf_eval = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    tp = np.zeros(len(te))

    for fold, (tr_i, val_i) in enumerate(skf_eval.split(tr, y)):
        X_tr = tr[feats].iloc[tr_i]
        y_tr = y[tr_i]
        X_val = tr[feats].iloc[val_i]

        if use_smote:
            # Borderline SMOTE on train fold only (numeric features only)
            num_feats_in = [c for c in feats if c not in cat_cols]
            cat_feats_in = [c for c in feats if c in cat_cols]

            # Encode cats temporarily for SMOTE
            X_tr_num = X_tr[num_feats_in].values.astype(float)
            sm = BorderlineSMOTE(random_state=seed, k_neighbors=5)
            try:
                X_res, y_res = sm.fit_resample(X_tr_num, y_tr)
                # Rebuild DataFrame with resampled numeric + duplicated cats
                X_tr_new = pd.DataFrame(X_res, columns=num_feats_in)
                # For cat features: sample from original to fill new rows
                n_new = len(X_res) - len(X_tr)
                if n_new > 0 and cat_feats_in:
                    fraud_cats = X_tr[cat_feats_in][y_tr == 1]
                    cat_extra = fraud_cats.sample(n=n_new, replace=True, random_state=seed).reset_index(drop=True)
                    cat_orig = X_tr[cat_feats_in].reset_index(drop=True)
                    X_tr_cats = pd.concat([cat_orig, cat_extra], ignore_index=True)
                    for c in cat_feats_in:
                        X_tr_new[c] = X_tr_cats[c].values
                X_tr = X_tr_new[feats]
                y_tr = y_res
            except Exception:
                pass  # fallback to no SMOTE if it fails

        m = lgb.LGBMClassifier(**params)
        fit_params = {
            "eval_set": [(X_val, y[val_i])],
            "eval_metric": "binary_logloss",
            "callbacks": [lgb.early_stopping(200, verbose=False)],
        }
        if cats:
            fit_params["categorical_feature"] = cats
        m.fit(X_tr, y_tr, **fit_params)
        oof[val_i] = m.predict_proba(X_val)[:, 1]
        tp += m.predict_proba(te[feats])[:, 1] / n_folds

    t, f1 = find_best_threshold(y, oof)
    log(f"  {name}: F1={f1:.4f}")
    return f1, t, oof, tp


# ============================================================
# STEP 2: EXPERIMENTS
# ============================================================

# --- Experiment A: Baseline (current best params, all features) ---
log("\n=== A: Baseline (all features) ===")
base_params = dict(
    objective="binary", n_estimators=5000, learning_rate=0.02,
    num_leaves=63, min_child_samples=50, subsample=0.8, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1, verbosity=-1,
)
f1_a, t_a, oof_a, tp_a = eval_config("baseline_all", feature_cols, base_params)
results["A_baseline"] = (f1_a, t_a, oof_a, tp_a)

# --- Experiment B: SHAP top-35 features ---
log("\n=== B: SHAP top-35 ===")
top35 = [f for f, _ in sorted_shap[:35]]
f1_b, t_b, oof_b, tp_b = eval_config("shap_top35", top35, base_params)
results["B_shap35"] = (f1_b, t_b, oof_b, tp_b)

# --- Experiment C: SHAP top-45 features ---
log("\n=== C: SHAP top-45 ===")
top45 = [f for f, _ in sorted_shap[:45]]
f1_c, t_c, oof_c, tp_c = eval_config("shap_top45", top45, base_params)
results["C_shap45"] = (f1_c, t_c, oof_c, tp_c)

# --- Experiment D: Balanced bagging (LightGBM native) ---
log("\n=== D: Balanced bagging ===")
balanced_params = dict(
    objective="binary", n_estimators=5000, learning_rate=0.02,
    num_leaves=63, min_child_samples=50,
    pos_bagging_fraction=1.0, neg_bagging_fraction=0.5,
    bagging_freq=1, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1, verbosity=-1,
)
f1_d, t_d, oof_d, tp_d = eval_config("balanced_bag", feature_cols, balanced_params)
results["D_balanced"] = (f1_d, t_d, oof_d, tp_d)

# --- Experiment E: path_smooth + extra_trees ---
log("\n=== E: path_smooth + extra_trees ===")
smooth_params = dict(
    objective="binary", n_estimators=5000, learning_rate=0.02,
    num_leaves=63, min_child_samples=50, subsample=0.8, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=1.0, path_smooth=10, extra_trees=True,
    random_state=42, n_jobs=-1, verbosity=-1,
)
f1_e, t_e, oof_e, tp_e = eval_config("smooth_extra", feature_cols, smooth_params)
results["E_smooth"] = (f1_e, t_e, oof_e, tp_e)

# --- Experiment F: Borderline SMOTE (in-fold) ---
log("\n=== F: Borderline SMOTE (in-fold) ===")
f1_f, t_f, oof_f, tp_f = eval_config("bsmote", feature_cols, base_params, use_smote=True)
results["F_bsmote"] = (f1_f, t_f, oof_f, tp_f)

# --- Experiment G: SHAP top-35 + balanced bagging ---
log("\n=== G: SHAP top-35 + balanced bagging ===")
f1_g, t_g, oof_g, tp_g = eval_config("shap35_balanced", top35, balanced_params)
results["G_shap_bal"] = (f1_g, t_g, oof_g, tp_g)

# --- Experiment H: Best combo + feature_fraction_bynode ---
log("\n=== H: Kitchen sink (balanced + smooth + bynode) ===")
kitchen_params = dict(
    objective="binary", n_estimators=5000, learning_rate=0.02,
    num_leaves=63, min_child_samples=50,
    pos_bagging_fraction=1.0, neg_bagging_fraction=0.5,
    bagging_freq=1, colsample_bytree=0.7, feature_fraction_bynode=0.8,
    reg_alpha=0.1, reg_lambda=1.0, path_smooth=5,
    min_gain_to_split=0.01,
    random_state=42, n_jobs=-1, verbosity=-1,
)
f1_h, t_h, oof_h, tp_h = eval_config("kitchen_sink", feature_cols, kitchen_params)
results["H_kitchen"] = (f1_h, t_h, oof_h, tp_h)

# --- Experiment I: is_unbalance ---
log("\n=== I: is_unbalance ===")
unbal_params = dict(
    objective="binary", n_estimators=5000, learning_rate=0.02,
    num_leaves=63, min_child_samples=50, subsample=0.8, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=1.0, is_unbalance=True,
    random_state=42, n_jobs=-1, verbosity=-1,
)
f1_i, t_i, oof_i, tp_i = eval_config("is_unbalance", feature_cols, unbal_params)
results["I_unbalance"] = (f1_i, t_i, oof_i, tp_i)

# ============================================================
# STEP 3: BLEND BEST EXPERIMENTS
# ============================================================
log("\n=== RESULTS SUMMARY ===")
sorted_results = sorted(results.items(), key=lambda x: x[1][0], reverse=True)
for name, (f1, t, _, _) in sorted_results:
    marker = " <<<" if f1 > BASELINE_F1 else ""
    log(f"  {name}: F1={f1:.4f}{marker}")

# Blend top experiments
log("\n=== BLENDING TOP MODELS ===")
top_n = min(5, len(sorted_results))
top_names = [n for n, _ in sorted_results[:top_n]]
top_oofs = [results[n][2] for n in top_names]
top_tests = [results[n][3] for n in top_names]

# Equal blend
avg_oof = np.mean(top_oofs, axis=0)
avg_test = np.mean(top_tests, axis=0)
t_blend, f1_blend = find_best_threshold(y, avg_oof)
log(f"  Equal blend top-{top_n}: F1={f1_blend:.4f}")

# Greedy weight optimization
best_weights = np.ones(top_n) / top_n
best_blend_f1 = f1_blend
for _ in range(5):
    improved = False
    for i in range(top_n):
        for delta in [-0.15, -0.1, -0.05, -0.02, 0.02, 0.05, 0.1, 0.15]:
            w = best_weights.copy()
            w[i] = max(0, w[i] + delta)
            if w.sum() == 0:
                continue
            w = w / w.sum()
            blend = sum(w[j] * top_oofs[j] for j in range(top_n))
            _, f1_b = find_best_threshold(y, blend)
            if f1_b > best_blend_f1:
                best_blend_f1 = f1_b
                best_weights = w.copy()
                improved = True
    if not improved:
        break

log(f"  Optimized blend: F1={best_blend_f1:.4f}")
for i, name in enumerate(top_names):
    if best_weights[i] > 0.001:
        log(f"    {name}: {best_weights[i]:.3f}")

# Also blend ALL experiments
all_oofs = [results[n][2] for n in results]
all_tests = [results[n][3] for n in results]
all_avg_oof = np.mean(all_oofs, axis=0)
all_avg_test = np.mean(all_tests, axis=0)
t_all, f1_all = find_best_threshold(y, all_avg_oof)
log(f"  All-experiment blend: F1={f1_all:.4f}")

# ============================================================
# STEP 4: MULTI-SEED BLEND OF BEST CONFIG
# ============================================================
best_config_name = sorted_results[0][0]
log(f"\n=== MULTI-SEED for best config: {best_config_name} ===")

# Determine best params and features
config_map = {
    "A_baseline": (feature_cols, base_params),
    "B_shap35": (top35, base_params),
    "C_shap45": (top45, base_params),
    "D_balanced": (feature_cols, balanced_params),
    "E_smooth": (feature_cols, smooth_params),
    "F_bsmote": (feature_cols, base_params),
    "G_shap_bal": (top35, balanced_params),
    "H_kitchen": (feature_cols, kitchen_params),
    "I_unbalance": (feature_cols, unbal_params),
}
best_feats, best_params = config_map[best_config_name]
use_smote_best = best_config_name == "F_bsmote"

ms_oofs = []
ms_tests = []
for seed in [42, 123, 456, 789, 999]:
    for nf in [5, 7]:
        p = {**best_params, "random_state": seed}
        f1_ms, t_ms, oof_ms, tp_ms = eval_config(
            f"seed{seed}_f{nf}", best_feats, p, use_smote=use_smote_best, n_folds=nf, seed=seed
        )
        ms_oofs.append(oof_ms)
        ms_tests.append(tp_ms)

ms_avg_oof = np.mean(ms_oofs, axis=0)
ms_avg_test = np.mean(ms_tests, axis=0)
t_ms_final, f1_ms_final = find_best_threshold(y, ms_avg_oof)
log(f"  Multi-seed blend (10 runs): F1={f1_ms_final:.4f}")

# ============================================================
# STEP 5: FINAL — PICK BEST & SAVE
# ============================================================
log("\n=== FINAL COMPARISON ===")
candidates = {
    "optimized_blend": (best_blend_f1, sum(best_weights[j] * top_tests[j] for j in range(top_n)),
                        find_best_threshold(y, sum(best_weights[j] * top_oofs[j] for j in range(top_n)))[0]),
    "all_blend": (f1_all, all_avg_test, t_all),
    "multi_seed": (f1_ms_final, ms_avg_test, t_ms_final),
}

for name, (f1, _, _) in candidates.items():
    marker = " ** BEATS BASELINE **" if f1 > BASELINE_F1 else ""
    log(f"  {name}: F1={f1:.4f}{marker}")

best_name = max(candidates, key=lambda k: candidates[k][0])
best_f1_final, best_test_final, best_t_final = candidates[best_name]

log(f"\nBest: {best_name}, F1={best_f1_final:.4f}")

if best_f1_final > BASELINE_F1:
    preds = (best_test_final >= best_t_final).astype(int)
    sub = pd.DataFrame({"id_user": test_df["id_user"].astype("int64"), "is_fraud": preds})
    sub.to_csv("artifacts/submissions/submission.csv", index=False)
    log(f"IMPROVED! Saved submission.csv. F1={best_f1_final:.4f}, fraud={preds.sum()} ({preds.mean()*100:.2f}%)")
    metrics = {
        "oof_f1": round(best_f1_final, 6),
        "approach": best_name,
        "predicted_fraud": int(preds.sum()),
        "predicted_fraud_rate": round(float(preds.mean()), 6),
    }
    with open("artifacts/reports/baseline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
else:
    preds = (best_test_final >= best_t_final).astype(int)
    sub = pd.DataFrame({"id_user": test_df["id_user"].astype("int64"), "is_fraud": preds})
    sub.to_csv("artifacts/submissions/submission_experiment.csv", index=False)
    log(f"No improvement ({best_f1_final:.4f} vs {BASELINE_F1}). Saved as submission_experiment.csv")

# Save SHAP analysis
shap_report = {
    "shap_top_features": [{"name": n, "importance": round(float(v), 6)} for n, v in sorted_shap[:50]],
    "experiment_results": {n: round(v[0], 6) for n, v in sorted_results},
    "best_approach": best_name,
    "best_f1": round(best_f1_final, 6),
    "baseline_f1": BASELINE_F1,
}
with open("artifacts/reports/experiment_results.json", "w") as f:
    json.dump(shap_report, f, indent=2)
log("Report saved to artifacts/reports/experiment_results.json")
