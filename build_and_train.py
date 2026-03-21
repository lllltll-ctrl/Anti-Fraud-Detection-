"""
Full pipeline: rebuild features with target encoding + new features, retrain ensemble.
Run autonomously - no user input needed.
"""
from __future__ import annotations
import json, warnings, time, sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from pathlib import Path

warnings.filterwarnings("ignore")
BASE = Path(".")


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================
# 1. FEATURE ENGINEERING
# ============================================================

def target_encode_column(train_col, train_y, test_col, col_name, n_folds=5, smoothing=20):
    """K-fold target encoding to prevent leakage."""
    global_mean = train_y.mean()
    train_encoded = np.full(len(train_col), global_mean)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    for tr_idx, val_idx in skf.split(train_col, train_y):
        tr_vals = train_col.iloc[tr_idx]
        tr_targets = train_y.iloc[tr_idx]
        stats = pd.DataFrame({"col": tr_vals, "y": tr_targets}).groupby("col")["y"].agg(["mean", "count"])
        smoothed = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
        train_encoded[val_idx] = train_col.iloc[val_idx].map(smoothed).fillna(global_mean).values

    # For test: use all train data
    stats = pd.DataFrame({"col": train_col, "y": train_y}).groupby("col")["y"].agg(["mean", "count"])
    smoothed = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
    test_encoded = test_col.map(smoothed).fillna(global_mean).values

    return train_encoded, test_encoded


def add_advanced_features(df):
    """Add features on top of cached base features."""
    out = df.copy()
    tx = out["tx_count"].clip(lower=1)

    # Velocity ratios
    out["cards_per_tx"] = out["unique_cards"] / tx
    out["holders_per_tx"] = out["unique_card_holders"] / tx
    out["switches_per_card"] = out["card_switch_count"] / out["unique_cards"].clip(lower=1)

    # First hour intensity
    fh = out["tx_first_hour"].clip(lower=1)
    out["fail_ratio_first_hour"] = out["fail_first_hour"] / fh
    out["cards_first_hour_ratio"] = out["unique_cards_first_hour"] / fh

    # 6h ratio
    out["tx_first_6h_ratio"] = out["tx_first_6h"] / tx

    # Combined risk
    out["multi_holder_switch"] = (out["unique_card_holders"] > 1).astype(int) * out["card_switch_count"]
    out["country_mismatch_total"] = out["card_country_mismatch_count"] + out["payment_country_mismatch_count"]

    # Amount patterns
    out["amount_cv"] = out["amount_std"] / out["amount_mean"].clip(lower=0.01)
    out["small_amount_ratio"] = out["small_amount_count"] / tx

    # Fail streak
    out["fail_streak_per_tx"] = out["max_fail_streak"] / tx

    # Speed flags
    out["fast_starter"] = (out["minutes_to_first_tx"] < 60).astype(int)
    out["very_fast_starter"] = (out["minutes_to_first_tx"] < 10).astype(int)

    # Log transforms
    for col in ["tx_count", "amount_sum", "card_switch_count", "minutes_to_first_tx",
                 "tx_span_minutes", "mean_gap_minutes"]:
        if col in out.columns:
            out[f"log_{col}"] = np.log1p(out[col].clip(lower=0))

    # Interaction: high-risk combo
    out["risk_combo"] = (
        (out["card_switch_count"] > 3).astype(int) +
        (out["unique_card_holders"] > 1).astype(int) +
        (out["fail_ratio"] > 0.5).astype(int) +
        (out["country_mismatch_total"] > 0).astype(int)
    )

    return out


def find_best_threshold(y_true, y_proba):
    ranked = sorted(zip(y_proba, y_true), key=lambda x: x[0], reverse=True)
    total_pos = int(sum(y_true))
    tp = fp = 0; fn = total_pos
    best_t, best_f1 = 0.5, 0.0
    i = 0
    while i < len(ranked):
        t = ranked[i][0]; btp = bfp = 0
        while i < len(ranked) and ranked[i][0] == t:
            if ranked[i][1] == 1: btp += 1
            else: bfp += 1
            i += 1
        tp += btp; fp += bfp; fn -= btp
        pd_ = tp + fp; rd = tp + fn
        if pd_ == 0 or rd == 0: continue
        p = tp / pd_; r = tp / rd
        if p + r == 0: continue
        f1 = 2 * p * r / (p + r)
        if f1 > best_f1: best_t = t; best_f1 = f1
    return best_t, best_f1


# ============================================================
# 2. MAIN PIPELINE
# ============================================================

def main():
    log("Loading cached features...")
    train_df = pd.read_csv("artifacts/processed/train_features.csv")
    test_df = pd.read_csv("artifacts/processed/test_features.csv")

    y_series = train_df["is_fraud"].astype(int)
    y = y_series.values

    # --- Target encoding for reg_country ---
    log("Target encoding reg_country...")
    train_te, test_te = target_encode_column(
        train_df["reg_country"].fillna("missing"),
        y_series,
        test_df["reg_country"].fillna("missing"),
        "reg_country", smoothing=50
    )
    train_df["reg_country_te"] = train_te
    test_df["reg_country_te"] = test_te

    # --- Advanced features ---
    log("Adding advanced features...")
    train_df = add_advanced_features(train_df)
    test_df = add_advanced_features(test_df)

    EXCLUDED = {"id_user", "timestamp_reg", "email", "is_fraud"}
    feature_cols = [c for c in train_df.columns if c not in EXCLUDED]

    cat_cols = [c for c in feature_cols
                if pd.api.types.is_string_dtype(train_df[c]) or pd.api.types.is_object_dtype(train_df[c])]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    log(f"Features: {len(feature_cols)}, Cat: {len(cat_cols)}, Num: {len(num_cols)}")

    # --- Prepare data for each model ---
    # LGB
    tr_lgb, te_lgb = train_df.copy(), test_df.copy()
    for col in cat_cols:
        vals = pd.concat([tr_lgb[col], te_lgb[col]]).fillna("missing").astype(str)
        cats = sorted(vals.unique().tolist())
        tr_lgb[col] = pd.Categorical(tr_lgb[col].fillna("missing").astype(str), categories=cats)
        te_lgb[col] = pd.Categorical(te_lgb[col].fillna("missing").astype(str), categories=cats)
    for col in num_cols:
        fill = float(tr_lgb[col].median()) if not tr_lgb[col].dropna().empty else 0.0
        tr_lgb[col] = tr_lgb[col].fillna(fill).astype(float)
        te_lgb[col] = te_lgb[col].fillna(fill).astype(float)

    # CB
    tr_cb, te_cb = train_df.copy(), test_df.copy()
    for col in cat_cols:
        tr_cb[col] = tr_cb[col].fillna("missing").astype(str)
        te_cb[col] = te_cb[col].fillna("missing").astype(str)
    for col in num_cols:
        fill = float(tr_cb[col].median()) if not tr_cb[col].dropna().empty else 0.0
        tr_cb[col] = tr_cb[col].fillna(fill).astype(float)
        te_cb[col] = te_cb[col].fillna(fill).astype(float)

    # XGB (label encode cats)
    tr_xgb, te_xgb = train_df.copy(), test_df.copy()
    for col in cat_cols:
        vals = pd.concat([tr_xgb[col], te_xgb[col]]).fillna("missing").astype(str)
        mapping = {v: i for i, v in enumerate(sorted(vals.unique()))}
        tr_xgb[col] = tr_xgb[col].fillna("missing").astype(str).map(mapping).astype(float)
        te_xgb[col] = te_xgb[col].fillna("missing").astype(str).map(mapping).astype(float)
    for col in num_cols:
        fill = float(tr_xgb[col].median()) if not tr_xgb[col].dropna().empty else 0.0
        tr_xgb[col] = tr_xgb[col].fillna(fill).astype(float)
        te_xgb[col] = te_xgb[col].fillna(fill).astype(float)

    cat_indices = [feature_cols.index(c) for c in cat_cols]

    # ============================================================
    # 3. TRAINING — 5-fold, 3 models
    # ============================================================
    N_FOLDS = 5
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    oof_lgb = np.zeros(len(y))
    oof_cb = np.zeros(len(y))
    oof_xgb = np.zeros(len(y))
    tp_lgb = np.zeros(len(test_df))
    tp_cb = np.zeros(len(test_df))
    tp_xgb = np.zeros(len(test_df))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(train_df, y)):
        log(f"--- Fold {fold} ---")

        # LightGBM
        m = lgb.LGBMClassifier(
            objective="binary", n_estimators=5000, learning_rate=0.02,
            num_leaves=63, min_child_samples=50, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1, verbosity=-1,
        )
        m.fit(
            tr_lgb[feature_cols].iloc[tr_idx], y[tr_idx],
            eval_set=[(tr_lgb[feature_cols].iloc[val_idx], y[val_idx])],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(200, verbose=False)],
            categorical_feature=cat_cols,
        )
        oof_lgb[val_idx] = m.predict_proba(tr_lgb[feature_cols].iloc[val_idx])[:, 1]
        tp_lgb += m.predict_proba(te_lgb[feature_cols])[:, 1] / N_FOLDS
        t, f1 = find_best_threshold(y[val_idx], oof_lgb[val_idx])
        log(f"  LGB: F1={f1:.4f}, iter={m.best_iteration_}")

        # CatBoost
        m2 = CatBoostClassifier(
            iterations=2000, learning_rate=0.05, depth=6, l2_leaf_reg=5,
            random_seed=42, verbose=0, eval_metric="Logloss",
            early_stopping_rounds=100, cat_features=cat_indices, thread_count=-1,
        )
        m2.fit(
            tr_cb[feature_cols].iloc[tr_idx], y[tr_idx],
            eval_set=(tr_cb[feature_cols].iloc[val_idx], y[val_idx]),
        )
        oof_cb[val_idx] = m2.predict_proba(tr_cb[feature_cols].iloc[val_idx])[:, 1]
        tp_cb += m2.predict_proba(te_cb[feature_cols])[:, 1] / N_FOLDS
        t, f1 = find_best_threshold(y[val_idx], oof_cb[val_idx])
        log(f"  CB:  F1={f1:.4f}, iter={m2.best_iteration_}")

        # XGBoost
        m3 = xgb.XGBClassifier(
            objective="binary:logistic", n_estimators=3000, learning_rate=0.03,
            max_depth=6, min_child_weight=50, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1, verbosity=0,
            early_stopping_rounds=150, tree_method="hist",
        )
        m3.fit(
            tr_xgb[feature_cols].iloc[tr_idx], y[tr_idx],
            eval_set=[(tr_xgb[feature_cols].iloc[val_idx], y[val_idx])],
            verbose=False,
        )
        oof_xgb[val_idx] = m3.predict_proba(tr_xgb[feature_cols].iloc[val_idx])[:, 1]
        tp_xgb += m3.predict_proba(te_xgb[feature_cols])[:, 1] / N_FOLDS
        t, f1 = find_best_threshold(y[val_idx], oof_xgb[val_idx])
        log(f"  XGB: F1={f1:.4f}")

    # ============================================================
    # 4. INDIVIDUAL + BLEND
    # ============================================================
    log("=== INDIVIDUAL OOF ===")
    for name, oof in [("LGB", oof_lgb), ("CB", oof_cb), ("XGB", oof_xgb)]:
        t, f1 = find_best_threshold(y, oof)
        log(f"{name}: F1={f1:.4f}, thresh={t:.4f}")

    # Blend search
    log("=== BLEND SEARCH ===")
    best_f1_blend = 0; best_w = (0.34, 0.33, 0.33); best_t_blend = 0.5
    for w1 in np.arange(0.1, 0.85, 0.05):
        for w2 in np.arange(0.05, 0.85 - w1, 0.05):
            w3 = 1.0 - w1 - w2
            if w3 < 0.05: continue
            blend = w1 * oof_lgb + w2 * oof_cb + w3 * oof_xgb
            t, f1 = find_best_threshold(y, blend)
            if f1 > best_f1_blend:
                best_f1_blend = f1; best_w = (w1, w2, w3); best_t_blend = t

    log(f"Blend: LGB={best_w[0]:.2f} CB={best_w[1]:.2f} XGB={best_w[2]:.2f}")
    log(f"Blend F1: {best_f1_blend:.4f}, thresh: {best_t_blend:.4f}")

    # ============================================================
    # 5. STACKING — meta-model on OOF predictions
    # ============================================================
    log("=== STACKING ===")
    oof_stack = np.column_stack([oof_lgb, oof_cb, oof_xgb])
    test_stack = np.column_stack([tp_lgb, tp_cb, tp_xgb])

    # K-fold stacking
    oof_meta = np.zeros(len(y))
    test_meta = np.zeros(len(test_df))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(oof_stack, y)):
        meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        meta.fit(oof_stack[tr_idx], y[tr_idx])
        oof_meta[val_idx] = meta.predict_proba(oof_stack[val_idx])[:, 1]
        test_meta += meta.predict_proba(test_stack)[:, 1] / N_FOLDS

    t_stack, f1_stack = find_best_threshold(y, oof_meta)
    log(f"Stacking F1: {f1_stack:.4f}, thresh: {t_stack:.4f}")

    # ============================================================
    # 6. PICK BEST APPROACH
    # ============================================================
    approaches = {
        "blend": (best_f1_blend, best_t_blend, best_w[0]*tp_lgb + best_w[1]*tp_cb + best_w[2]*tp_xgb),
        "stack": (f1_stack, t_stack, test_meta),
    }

    # Also check individual models
    for name, oof, tp in [("lgb_only", oof_lgb, tp_lgb), ("cb_only", oof_cb, tp_cb), ("xgb_only", oof_xgb, tp_xgb)]:
        t, f1 = find_best_threshold(y, oof)
        approaches[name] = (f1, t, tp)

    best_name = max(approaches, key=lambda k: approaches[k][0])
    best_f1, best_thresh, best_test_pred = approaches[best_name]

    log(f"\nBest approach: {best_name}")
    log(f"Best F1: {best_f1:.4f}, thresh: {best_thresh:.4f}")

    # ============================================================
    # 7. PSEUDO-LABELING (if blend/stack improved)
    # ============================================================
    BASELINE_F1 = 0.6252  # previous best

    if best_f1 > BASELINE_F1 + 0.002:
        log("=== PSEUDO-LABELING ===")
        # Use high-confidence predictions on test
        high_fraud = best_test_pred >= 0.8
        high_legit = best_test_pred <= 0.05
        n_pseudo_fraud = int(high_fraud.sum())
        n_pseudo_legit = int(high_legit.sum())
        log(f"Pseudo labels: {n_pseudo_fraud} fraud, {n_pseudo_legit} legit")

        if n_pseudo_fraud > 100:
            # Add pseudo-labeled data to training
            pseudo_fraud_df = test_df[high_fraud].copy()
            pseudo_fraud_df["is_fraud"] = 1
            pseudo_legit_df = test_df[high_legit].sample(n=min(n_pseudo_legit, n_pseudo_fraud * 5), random_state=42).copy()
            pseudo_legit_df["is_fraud"] = 0

            augmented = pd.concat([train_df, pseudo_fraud_df, pseudo_legit_df], ignore_index=True)
            y_aug = augmented["is_fraud"].astype(int).values
            log(f"Augmented: {len(augmented)} rows ({y_aug.sum()} fraud)")

            # Retrain LGB only (fastest) on augmented data, predict on full test
            log("Retraining LGB on augmented data...")

            # Prepare augmented LGB data
            aug_lgb = augmented.copy()
            for col in cat_cols:
                vals = pd.concat([aug_lgb[col], te_lgb[col]]).fillna("missing").astype(str)
                cats = sorted(vals.unique().tolist())
                aug_lgb[col] = pd.Categorical(aug_lgb[col].fillna("missing").astype(str), categories=cats)
            for col in num_cols:
                fill = float(aug_lgb[col].median()) if not aug_lgb[col].dropna().empty else 0.0
                aug_lgb[col] = aug_lgb[col].fillna(fill).astype(float)

            # Train on full augmented, use original val for early stopping
            oof_pseudo = np.zeros(len(y))  # only for original data
            tp_pseudo = np.zeros(len(test_df))

            skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            for fold, (tr_idx, val_idx) in enumerate(skf2.split(train_df, y)):
                # Use original val_idx for validation but train on augmented - original_val
                aug_tr_mask = np.ones(len(augmented), dtype=bool)
                aug_tr_mask[val_idx] = False  # exclude original val from training

                m = lgb.LGBMClassifier(
                    objective="binary", n_estimators=5000, learning_rate=0.02,
                    num_leaves=63, min_child_samples=50, subsample=0.8, colsample_bytree=0.7,
                    reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1, verbosity=-1,
                )
                m.fit(
                    aug_lgb[feature_cols].iloc[np.where(aug_tr_mask)[0]], y_aug[aug_tr_mask],
                    eval_set=[(tr_lgb[feature_cols].iloc[val_idx], y[val_idx])],
                    eval_metric="binary_logloss",
                    callbacks=[lgb.early_stopping(200, verbose=False)],
                    categorical_feature=cat_cols,
                )
                oof_pseudo[val_idx] = m.predict_proba(tr_lgb[feature_cols].iloc[val_idx])[:, 1]
                tp_pseudo += m.predict_proba(te_lgb[feature_cols])[:, 1] / 5

            t_p, f1_p = find_best_threshold(y, oof_pseudo)
            log(f"Pseudo-labeled LGB F1: {f1_p:.4f}")

            if f1_p > best_f1:
                best_f1 = f1_p
                best_thresh = t_p
                best_test_pred = tp_pseudo
                best_name = "pseudo_lgb"
                log("Pseudo-labeling improved! Using it.")
            else:
                log("Pseudo-labeling did not improve. Keeping previous best.")
    else:
        log(f"Skipping pseudo-labeling (F1={best_f1:.4f} vs baseline={BASELINE_F1:.4f})")

    # ============================================================
    # 8. SAVE RESULTS
    # ============================================================
    preds = (best_test_pred >= best_thresh).astype(int)
    pred_fraud = int(preds.sum())
    pred_rate = float(preds.mean())

    log(f"\nFINAL: {best_name}, F1={best_f1:.4f}, fraud={pred_fraud} ({pred_rate*100:.2f}%)")

    # Only overwrite if improved
    if best_f1 > BASELINE_F1:
        sub = pd.DataFrame({"id_user": test_df["id_user"].astype("int64"), "is_fraud": preds})
        sub.to_csv("artifacts/submissions/submission.csv", index=False)
        log("Submission saved (improved!)")
    else:
        sub = pd.DataFrame({"id_user": test_df["id_user"].astype("int64"), "is_fraud": preds})
        sub.to_csv("artifacts/submissions/submission_v2.csv", index=False)
        log(f"Saved as submission_v2.csv (no improvement over {BASELINE_F1:.4f})")

    # Save OOF blend for potential stacking
    oof_final = best_w[0]*oof_lgb + best_w[1]*oof_cb + best_w[2]*oof_xgb
    labels = (oof_final >= best_t_blend).astype(int)
    tp_cnt = int(((labels==1)&(y==1)).sum())
    fp_cnt = int(((labels==1)&(y==0)).sum())
    fn_cnt = int(((labels==0)&(y==1)).sum())
    prec = tp_cnt/(tp_cnt+fp_cnt) if tp_cnt+fp_cnt else 0
    rec = tp_cnt/(tp_cnt+fn_cnt) if tp_cnt+fn_cnt else 0

    metrics = {
        "best_approach": best_name,
        "oof_f1": round(best_f1, 6),
        "threshold": round(best_thresh, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "feature_count": len(feature_cols),
        "n_folds": N_FOLDS,
        "predicted_fraud": pred_fraud,
        "predicted_fraud_rate": round(pred_rate, 6),
        "individual_f1": {},
    }
    for name, oof in [("lgb", oof_lgb), ("cb", oof_cb), ("xgb", oof_xgb)]:
        t, f1 = find_best_threshold(y, oof)
        metrics["individual_f1"][name] = round(f1, 6)
    metrics["blend_f1"] = round(best_f1_blend, 6)
    metrics["stack_f1"] = round(f1_stack, 6)

    with open("artifacts/reports/baseline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log(f"Metrics saved.\n{json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    main()
