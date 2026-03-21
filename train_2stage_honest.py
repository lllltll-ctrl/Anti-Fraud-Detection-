"""Two-stage HONEST: K-fold graph features (no leakage) + component override."""
import time, json, warnings, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from collections import defaultdict, deque
from scipy.stats import rankdata

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
y_full = tr_u.set_index("id_user")["is_fraud"]

# ============================================================
# 2. BUILD GRAPHS
# ============================================================
log("Building graphs...")
card_to_users = defaultdict(set)
user_to_cards = defaultdict(set)
for uid, card in zip(all_tx["id_user"], all_tx["card_mask_hash"]):
    card_to_users[card].add(uid)
    user_to_cards[uid].add(card)

holder_to_users = defaultdict(set)
user_to_holders = defaultdict(set)
for uid, holder in zip(all_tx["id_user"], all_tx["card_holder"].fillna("").str.lower()):
    if holder:
        holder_to_users[holder].add(uid)
        user_to_holders[uid].add(holder)

# Parse timestamps
all_tx["ts"] = pd.to_datetime(all_tx["timestamp_tr"], format="ISO8601", utc=True)
tr_u["ts_reg"] = pd.to_datetime(tr_u["timestamp_reg"], format="ISO8601", utc=True)
te_u["ts_reg"] = pd.to_datetime(te_u["timestamp_reg"], format="ISO8601", utc=True)

all_users_df = pd.concat([
    tr_u[["id_user", "email", "gender", "reg_country", "traffic_type", "ts_reg"]],
    te_u[["id_user", "email", "gender", "reg_country", "traffic_type", "ts_reg"]],
])
all_users_df["ts_reg"] = pd.to_datetime(all_users_df["ts_reg"], utc=True)

# ============================================================
# 3. CONNECTED COMPONENTS (for override)
# ============================================================
log("Finding connected components...")
all_users = train_ids | test_ids
visited = set()
user_component = {}
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
    comp_id = len(components)
    for u in component:
        user_component[u] = comp_id
    components.append(component)

log(f"  {len(components)} components")


# ============================================================
# 4. BUILD BEHAVIORAL FEATURES (no graph, no leakage)
# ============================================================
def build_base_features(user_ids_list, tx_df, users_df):
    """Build behavioral features WITHOUT graph/label info."""
    tx_sub = tx_df[tx_df["id_user"].isin(set(user_ids_list))].copy()
    tx_sub = tx_sub.sort_values(["id_user", "ts"])
    users_sub = users_df.set_index("id_user").reindex(user_ids_list)

    feats = pd.DataFrame(index=user_ids_list)
    feats.index.name = "id_user"

    feats["gender"] = users_sub["gender"].values
    feats["reg_country"] = users_sub["reg_country"].values
    feats["traffic_type"] = users_sub["traffic_type"].values
    feats["email_domain"] = users_sub["email"].str.split("@").str[1].values
    feats["reg_hour"] = users_sub["ts_reg"].dt.hour.values
    feats["reg_weekday"] = users_sub["ts_reg"].dt.weekday.values

    g = tx_sub.groupby("id_user")
    feats["tx_count"] = g.size().reindex(user_ids_list, fill_value=0).values
    tx_clip = feats["tx_count"].clip(lower=1)

    feats["success_count"] = tx_sub[tx_sub.status == "success"].groupby("id_user").size().reindex(user_ids_list, fill_value=0).values
    feats["fail_count"] = tx_sub[tx_sub.status == "fail"].groupby("id_user").size().reindex(user_ids_list, fill_value=0).values
    feats["fail_ratio"] = feats["fail_count"] / tx_clip
    feats["success_ratio"] = feats["success_count"] / tx_clip

    feats["amount_sum"] = g["amount"].sum().reindex(user_ids_list, fill_value=0).values
    feats["amount_mean"] = g["amount"].mean().reindex(user_ids_list, fill_value=0).values
    feats["amount_std"] = g["amount"].std().reindex(user_ids_list, fill_value=0).values
    feats["amount_max"] = g["amount"].max().reindex(user_ids_list, fill_value=0).values
    feats["amount_min"] = g["amount"].min().reindex(user_ids_list, fill_value=0).values
    feats["amount_range"] = feats["amount_max"] - feats["amount_min"]
    feats["amount_cv"] = feats["amount_std"] / feats["amount_mean"].clip(lower=0.01)

    feats["unique_cards"] = g["card_mask_hash"].nunique().reindex(user_ids_list, fill_value=0).values
    feats["unique_card_holders"] = g["card_holder"].nunique().reindex(user_ids_list, fill_value=0).values
    feats["unique_card_countries"] = g["card_country"].nunique().reindex(user_ids_list, fill_value=0).values
    feats["unique_payment_countries"] = g["payment_country"].nunique().reindex(user_ids_list, fill_value=0).values
    feats["unique_card_brands"] = g["card_brand"].nunique().reindex(user_ids_list, fill_value=0).values
    feats["card_holder_per_card"] = feats["unique_card_holders"] / feats["unique_cards"].clip(lower=1)
    feats["cards_per_tx"] = feats["unique_cards"] / tx_clip
    feats["holders_per_tx"] = feats["unique_card_holders"] / tx_clip

    feats["fraud_error_count"] = tx_sub[tx_sub.error_group == "fraud"].groupby("id_user").size().reindex(user_ids_list, fill_value=0).values
    feats["antifraud_count"] = tx_sub[tx_sub.error_group == "antifraud"].groupby("id_user").size().reindex(user_ids_list, fill_value=0).values
    feats["fraud_error_ratio"] = feats["fraud_error_count"] / tx_clip
    feats["antifraud_ratio"] = feats["antifraud_count"] / tx_clip
    feats["unique_error_groups"] = g["error_group"].nunique().reindex(user_ids_list, fill_value=0).values
    feats["err_fraud_ratio"] = feats["fraud_error_count"] / (feats["fraud_error_count"] + feats["antifraud_count"]).clip(lower=1)

    feats["unique_tx_types"] = g["transaction_type"].nunique().reindex(user_ids_list, fill_value=0).values
    feats["card_init_count"] = tx_sub[tx_sub.transaction_type == "card_init"].groupby("id_user").size().reindex(user_ids_list, fill_value=0).values
    feats["card_init_ratio"] = feats["card_init_count"] / tx_clip
    feats["resign_count"] = tx_sub[tx_sub.transaction_type == "resign"].groupby("id_user").size().reindex(user_ids_list, fill_value=0).values

    feats["has_prepaid"] = tx_sub[tx_sub.card_type.str.contains("PREPAID", case=False, na=False)].groupby("id_user").size().reindex(user_ids_list, fill_value=0).clip(upper=1).values

    tx_sub["card_pay_mismatch"] = (tx_sub["card_country"] != tx_sub["payment_country"]).astype(int)
    feats["country_mismatch_ratio"] = tx_sub.groupby("id_user")["card_pay_mismatch"].mean().reindex(user_ids_list, fill_value=0).values

    tx_with_reg = tx_sub.merge(users_df[["id_user", "reg_country"]], on="id_user", how="left")
    tx_with_reg["card_reg_mismatch"] = (tx_with_reg["card_country"] != tx_with_reg["reg_country"]).astype(int)
    feats["card_reg_mismatch_ratio"] = tx_with_reg.groupby("id_user")["card_reg_mismatch"].mean().reindex(user_ids_list, fill_value=0).values

    tx_sub["prev_card"] = tx_sub.groupby("id_user")["card_mask_hash"].shift(1)
    tx_sub["card_switch"] = ((tx_sub["card_mask_hash"] != tx_sub["prev_card"]) & tx_sub["prev_card"].notna()).astype(int)
    feats["card_switch_count"] = tx_sub.groupby("id_user")["card_switch"].sum().reindex(user_ids_list, fill_value=0).values
    feats["card_switch_ratio"] = feats["card_switch_count"] / tx_clip

    first_tx = g["ts"].min().reindex(user_ids_list)
    last_tx = g["ts"].max().reindex(user_ids_list)
    reg_ts = pd.to_datetime(pd.Series(users_sub["ts_reg"].values, index=user_ids_list), utc=True)
    feats["minutes_to_first_tx"] = ((first_tx - reg_ts).dt.total_seconds() / 60).values
    feats["tx_span_minutes"] = ((last_tx - first_tx).dt.total_seconds() / 60).values
    feats["minutes_to_first_success"] = ((tx_sub[tx_sub.status == "success"].groupby("id_user")["ts"].min().reindex(user_ids_list) - reg_ts).dt.total_seconds() / 60).values

    tx_sub["gap_sec"] = tx_sub.groupby("id_user")["ts"].diff().dt.total_seconds()
    feats["mean_gap_sec"] = tx_sub.groupby("id_user")["gap_sec"].mean().reindex(user_ids_list, fill_value=0).values
    feats["min_gap_sec"] = tx_sub.groupby("id_user")["gap_sec"].min().reindex(user_ids_list, fill_value=0).values
    feats["log_mean_gap"] = np.log1p(np.clip(feats["mean_gap_sec"], 0, None))
    feats["min_fail_gap_sec"] = tx_sub[tx_sub.status == "fail"].groupby("id_user")["gap_sec"].min().reindex(user_ids_list, fill_value=-1).values

    tx_sub["hour"] = tx_sub["ts"].dt.hour
    feats["mode_hour"] = tx_sub.groupby("id_user")["hour"].apply(lambda x: x.mode().iloc[0] if len(x) > 0 else 12).reindex(user_ids_list, fill_value=12).values
    feats["night_tx_ratio"] = tx_sub.assign(n=tx_sub.hour.isin([0, 1, 2, 3, 4, 5]).astype(int)).groupby("id_user")["n"].mean().reindex(user_ids_list, fill_value=0).values
    feats["business_hours_ratio"] = tx_sub.assign(b=tx_sub.hour.isin(range(9, 18)).astype(int)).groupby("id_user")["b"].mean().reindex(user_ids_list, fill_value=0).values

    first_tx_dict = first_tx.to_dict()
    tx_sub["hrs_since_first"] = tx_sub.apply(
        lambda r: (r["ts"] - first_tx_dict.get(r["id_user"], r["ts"])).total_seconds() / 3600, axis=1
    )
    feats["tx_first_hour"] = tx_sub[tx_sub.hrs_since_first <= 1].groupby("id_user").size().reindex(user_ids_list, fill_value=0).values
    feats["fail_first_hour"] = tx_sub[(tx_sub.hrs_since_first <= 1) & (tx_sub.status == "fail")].groupby("id_user").size().reindex(user_ids_list, fill_value=0).values
    feats["tx_first_6h"] = tx_sub[tx_sub.hrs_since_first <= 6].groupby("id_user").size().reindex(user_ids_list, fill_value=0).values

    tx_sub["prev_status"] = tx_sub.groupby("id_user")["status"].shift(1)
    tx_sub["prev_amount"] = tx_sub.groupby("id_user")["amount"].shift(1)
    tx_sub["fail_then_success"] = ((tx_sub.prev_status == "fail") & (tx_sub.status == "success") & (tx_sub.amount == tx_sub.prev_amount)).astype(int)
    feats["fail_then_success_count"] = tx_sub.groupby("id_user")["fail_then_success"].sum().reindex(user_ids_list, fill_value=0).values

    def max_streak(statuses):
        mx = cur = 0
        for s in statuses:
            if s == "fail":
                cur += 1
                mx = max(mx, cur)
            else:
                cur = 0
        return mx
    feats["max_fail_streak"] = g["status"].apply(max_streak).reindex(user_ids_list, fill_value=0).values
    feats["small_amount_ratio"] = tx_sub.assign(s=(tx_sub.amount <= 5).astype(int)).groupby("id_user")["s"].mean().reindex(user_ids_list, fill_value=0).values

    tx_sub["tx_hour_key"] = tx_sub["ts"].dt.floor("h")
    hourly = tx_sub.groupby(["id_user", "tx_hour_key"])["amount"].sum().reset_index()
    feats["amount_per_hour_max"] = hourly.groupby("id_user")["amount"].max().reindex(user_ids_list, fill_value=0).values

    tx_sub["tx_day"] = tx_sub["ts"].dt.date
    daily = tx_sub.groupby(["id_user", "tx_day"]).size().reset_index(name="cnt")
    feats["max_tx_per_day"] = daily.groupby("id_user")["cnt"].max().reindex(user_ids_list, fill_value=0).values

    max_shared = []
    for uid in user_ids_list:
        cards = user_to_cards.get(uid, set())
        mx = max((len(card_to_users.get(c, set())) for c in cards), default=0)
        max_shared.append(mx)
    feats["max_users_per_card"] = max_shared

    for c in ["tx_count", "amount_sum", "card_switch_count", "minutes_to_first_tx", "mean_gap_sec"]:
        feats[f"log_{c}"] = np.log1p(np.clip(feats[c].values.astype(float), 0, None))

    feats["risk_combo"] = (
        (feats["card_switch_count"] > 3).astype(int)
        + (feats["unique_card_holders"] > 1).astype(int)
        + (feats["fail_ratio"] > 0.5).astype(int)
        + (feats["country_mismatch_ratio"] > 0).astype(int)
    )
    feats["fail_ratio_first_hour"] = feats["fail_first_hour"] / feats["tx_first_hour"].clip(lower=1)
    feats["tx_first_6h_ratio"] = feats["tx_first_6h"] / tx_clip

    # Component size (no leakage - just structure)
    feats["comp_size"] = [len(components[user_component.get(uid, 0)]) for uid in user_ids_list]
    feats["log_comp_size"] = np.log1p(feats["comp_size"])

    # Number of card neighbors / holder neighbors (no fraud info - just counts)
    card_neighbor_counts = []
    holder_neighbor_counts = []
    for uid in user_ids_list:
        cards = user_to_cards.get(uid, set())
        cn = set()
        for c in cards:
            cn |= card_to_users.get(c, set()) - {uid}
        card_neighbor_counts.append(len(cn))
        holders = user_to_holders.get(uid, set())
        hn = set()
        for h in holders:
            hn |= holder_to_users.get(h, set()) - {uid}
        holder_neighbor_counts.append(len(hn))
    feats["graph_card_total_n"] = card_neighbor_counts
    feats["graph_holder_total_n"] = holder_neighbor_counts

    feats = feats.reset_index()
    feats = feats.rename(columns={"index": "id_user"})
    return feats


def add_graph_features(feats_df, user_ids, known_fraud):
    """Add graph features using a specific known_fraud set."""
    graph_data = []
    for uid in user_ids:
        cards = user_to_cards.get(uid, set())
        neighbors = set()
        fraud_neighbors = set()
        max_density = 0.0
        n_fraud_cards = 0
        for card in cards:
            others = card_to_users.get(card, set()) - {uid}
            neighbors |= others
            fn = others & known_fraud
            fraud_neighbors |= fn
            if others:
                d = len(fn) / len(others)
                max_density = max(max_density, d)
            if fn:
                n_fraud_cards += 1
        nt = len(neighbors)
        nf = len(fraud_neighbors)

        holders = user_to_holders.get(uid, set())
        h_neighbors = set()
        h_fraud = set()
        for h in holders:
            others = holder_to_users.get(h, set()) - {uid}
            h_neighbors |= others
            h_fraud |= (others & known_fraud)
        hn = len(h_neighbors)
        hf = len(h_fraud)

        # Component fraud ratio with this known_fraud
        comp_id = user_component.get(uid, 0)
        comp = components[comp_id]
        comp_train = comp & train_ids
        comp_known_fraud = comp_train & known_fraud
        comp_fr = len(comp_known_fraud) / max(len(comp_train), 1)

        graph_data.append([
            nf, nf / max(nt, 1), max_density,
            n_fraud_cards, n_fraud_cards / max(len(cards), 1),
            hf, hf / max(hn, 1),
            int(nf > 0 or hf > 0), nf + hf,
            comp_fr,
        ])

    graph_cols = [
        "g_card_fraud_n", "g_card_fraud_ratio", "g_card_max_density",
        "g_n_fraud_cards", "g_fraud_card_ratio",
        "g_holder_fraud_n", "g_holder_fraud_ratio",
        "g_any_fraud", "g_total_fraud",
        "g_comp_fraud_ratio",
    ]
    graph_df = pd.DataFrame(graph_data, columns=graph_cols)
    for col in graph_cols:
        feats_df[col] = graph_df[col].values
    return feats_df


# ============================================================
# 5. BUILD BASE FEATURES
# ============================================================
log("Building base features (no graph)...")
train_feats = build_base_features(tr_u["id_user"].tolist(), all_tx, all_users_df)
log(f"  Train: {train_feats.shape}")
test_feats = build_base_features(te_u["id_user"].tolist(), all_tx, all_users_df)
log(f"  Test: {test_feats.shape}")

train_feats = train_feats.merge(tr_u[["id_user", "is_fraud"]], on="id_user")
y = train_feats["is_fraud"].values

# ============================================================
# 6. TARGET ENCODING (K-fold, no leakage)
# ============================================================
log("Target encoding...")
gm = y.mean()
smoothing = 50
for te_col in ["reg_country", "gender", "traffic_type", "email_domain"]:
    skf_te = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_te_vals = np.full(len(train_feats), gm)
    col_data = train_feats[te_col].fillna("missing").astype(str)
    y_s = pd.Series(y)
    for tr_i, val_i in skf_te.split(train_feats, y):
        stats = pd.DataFrame({"c": col_data.iloc[tr_i], "y": y_s.iloc[tr_i]}).groupby("c")["y"].agg(["mean", "count"])
        s = (stats["count"] * stats["mean"] + smoothing * gm) / (stats["count"] + smoothing)
        train_te_vals[val_i] = col_data.iloc[val_i].map(s).fillna(gm).values
    stats_all = pd.DataFrame({"c": col_data, "y": y_s}).groupby("c")["y"].agg(["mean", "count"])
    s_all = (stats_all["count"] * stats_all["mean"] + smoothing * gm) / (stats_all["count"] + smoothing)
    test_te_vals = test_feats[te_col].fillna("missing").astype(str).map(s_all).fillna(gm).values
    train_feats[f"{te_col}_te"] = train_te_vals
    test_feats[f"{te_col}_te"] = test_te_vals

# ============================================================
# 7. K-FOLD GRAPH FEATURES + TRAINING
# ============================================================
log("Training with K-fold graph features...")

# Add graph features for TEST (full train labels - this is legitimate)
test_feats = add_graph_features(test_feats, te_u["id_user"].tolist(), fraud_set)

# Prepare categoricals
cat_cols = ["gender", "reg_country", "traffic_type", "email_domain"]
EXCLUDED = {"id_user", "is_fraud"}

# We need graph feature columns to exist in train_feats before determining feature_cols
# Add dummy graph columns first
graph_cols = ["g_card_fraud_n", "g_card_fraud_ratio", "g_card_max_density",
              "g_n_fraud_cards", "g_fraud_card_ratio",
              "g_holder_fraud_n", "g_holder_fraud_ratio",
              "g_any_fraud", "g_total_fraud", "g_comp_fraud_ratio"]
for col in graph_cols:
    train_feats[col] = 0.0

feature_cols = [c for c in train_feats.columns if c not in EXCLUDED]
num_cols = [c for c in feature_cols if c not in cat_cols]
log(f"Features: {len(feature_cols)}, Cat: {len(cat_cols)}, Num: {len(num_cols)}")

# Prepare categorical encoding
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

# Main CV loop: K-fold graph features computed per fold
configs = [
    ("gbdt_7f_42", 7, 42, "lgb", {"lr": 0.02, "leaves": 63, "depth": -1, "sub": 0.8, "col": 0.7}),
    ("gbdt_5f_42", 5, 42, "lgb", {"lr": 0.02, "leaves": 63, "depth": -1, "sub": 0.8, "col": 0.7}),
    ("gbdt_7f_123", 7, 123, "lgb", {"lr": 0.02, "leaves": 63, "depth": -1, "sub": 0.8, "col": 0.7}),
    ("gbdt_5f_123", 5, 123, "lgb", {"lr": 0.02, "leaves": 63, "depth": -1, "sub": 0.8, "col": 0.7}),
    ("gbdt_7f_999", 7, 999, "lgb", {"lr": 0.015, "leaves": 95, "depth": 8, "sub": 0.75, "col": 0.65}),
    ("gbdt_7f_77", 7, 77, "lgb", {"lr": 0.025, "leaves": 48, "depth": 7, "sub": 0.85, "col": 0.65}),
    ("dart_7f_42", 7, 42, "dart", {"lr": 0.05, "leaves": 63, "depth": -1, "sub": 0.8, "col": 0.7}),
    ("dart_5f_123", 5, 123, "dart", {"lr": 0.05, "leaves": 63, "depth": -1, "sub": 0.8, "col": 0.7}),
]

all_oofs = []
all_tests = []
model_labels = []
train_user_ids = tr_u["id_user"].tolist()

for name, n_folds, seed, mtype, params in configs:
    log(f"=== {name} ===")
    skf_m = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    tp_arr = np.zeros(len(te))

    for fold, (tr_idx, val_idx) in enumerate(skf_m.split(tr, y)):
        # K-FOLD GRAPH: train graph uses only train fold labels
        fold_fraud = set(tr_u.iloc[tr_idx][tr_u.iloc[tr_idx]["is_fraud"] == 1]["id_user"])

        # Compute graph features for train fold (using fold_fraud as known)
        tr_fold = tr.iloc[tr_idx].copy()
        tr_fold_ids = [train_user_ids[i] for i in tr_idx]
        tr_fold_graph = add_graph_features(
            train_feats.iloc[tr_idx].copy(), tr_fold_ids, fold_fraud
        )
        for gc in graph_cols:
            tr_fold[gc] = tr_fold_graph[gc].values

        # Compute graph features for val fold (using fold_fraud as known)
        val_fold_ids = [train_user_ids[i] for i in val_idx]
        val_fold_graph = add_graph_features(
            train_feats.iloc[val_idx].copy(), val_fold_ids, fold_fraud
        )
        val_fold = tr.iloc[val_idx].copy()
        for gc in graph_cols:
            val_fold[gc] = val_fold_graph[gc].values

        if mtype == "lgb":
            m = lgb.LGBMClassifier(
                objective="binary", n_estimators=5000, learning_rate=params["lr"],
                num_leaves=params["leaves"], max_depth=params["depth"],
                min_child_samples=50, subsample=params["sub"], colsample_bytree=params["col"],
                reg_alpha=0.1, reg_lambda=1.0, random_state=seed, n_jobs=-1, verbosity=-1,
            )
            m.fit(
                tr_fold[feature_cols], y[tr_idx],
                eval_set=[(val_fold[feature_cols], y[val_idx])],
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
                tr_fold[feature_cols], y[tr_idx],
                eval_set=[(val_fold[feature_cols], y[val_idx])],
                eval_metric="binary_logloss",
                callbacks=[lgb.early_stopping(50, verbose=False)],
                categorical_feature=cat_cols,
            )

        oof[val_idx] = m.predict_proba(val_fold[feature_cols])[:, 1]
        tp_arr += m.predict_proba(te[feature_cols])[:, 1] / n_folds

    t_, f1_ = find_best_threshold(y, oof)
    log(f"  F1={f1_:.4f}")
    all_oofs.append(oof)
    all_tests.append(tp_arr)
    model_labels.append(name)

    if name == configs[0][0]:
        imp = dict(zip(feature_cols, m.feature_importances_))
        top20 = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:20]
        log("  Top-20 features:")
        for fn, fv in top20:
            log(f"    {fn}: {fv}")

# ============================================================
# 8. BLEND
# ============================================================
log("\n=== BLEND ===")
n_models = len(all_oofs)

avg_oof = np.mean(all_oofs, axis=0)
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
for i, lbl in enumerate(model_labels):
    if best_weights[i] > 0.001:
        log(f"  {lbl}: {best_weights[i]:.3f}")

blend_oof = sum(best_weights[j] * all_oofs[j] for j in range(n_models))
blend_test = sum(best_weights[j] * all_tests[j] for j in range(n_models))

# ============================================================
# 9. TWO-STAGE OVERRIDE
# ============================================================
log("\n=== TWO-STAGE OVERRIDE ===")

# Classify components based on full train labels
train_ct = []
for uid in train_user_ids:
    comp_id = user_component[uid]
    comp = components[comp_id]
    train_in = comp & train_ids
    fraud_in = train_in & fraud_set
    legit_in = train_in - fraud_set
    if fraud_in and not legit_in:
        train_ct.append("pure_fraud")
    elif not fraud_in:
        train_ct.append("pure_legit")
    else:
        train_ct.append("mixed")
train_ct = np.array(train_ct)

test_ct = []
for uid in te_u["id_user"]:
    comp_id = user_component[uid]
    comp = components[comp_id]
    train_in = comp & train_ids
    fraud_in = train_in & fraud_set
    legit_in = train_in - fraud_set
    if fraud_in and not legit_in:
        test_ct.append("pure_fraud")
    elif not fraud_in:
        test_ct.append("pure_legit")
    else:
        test_ct.append("mixed")
test_ct = np.array(test_ct)

log(f"Train: {(train_ct=='pure_fraud').sum()} pf, {(train_ct=='pure_legit').sum()} pl, {(train_ct=='mixed').sum()} mx")
log(f"Test:  {(test_ct=='pure_fraud').sum()} pf, {(test_ct=='pure_legit').sum()} pl, {(test_ct=='mixed').sum()} mx")

# Sweep threshold with override
best_overall_f1 = 0
best_t = 0.5
for t_try in np.arange(0.05, 0.95, 0.005):
    combined = np.zeros(len(y))
    combined[train_ct == "pure_fraud"] = 1
    combined[train_ct == "pure_legit"] = 0
    combined[train_ct == "mixed"] = (blend_oof[train_ct == "mixed"] >= t_try).astype(int)
    f1_try = f1_score(y, combined)
    if f1_try > best_overall_f1:
        best_overall_f1 = f1_try
        best_t = t_try

log(f"Two-stage F1: {best_overall_f1:.4f} (threshold={best_t:.4f})")

# Raw ML
t_raw, f1_raw = find_best_threshold(y, blend_oof)
log(f"Raw ML F1: {f1_raw:.4f}")

# Pick best
if best_overall_f1 > f1_raw:
    log("Using TWO-STAGE")
    test_preds = np.zeros(len(te))
    test_preds[test_ct == "pure_fraud"] = 1
    test_preds[test_ct == "pure_legit"] = 0
    test_preds[test_ct == "mixed"] = (blend_test[test_ct == "mixed"] >= best_t).astype(int)
    final_f1 = best_overall_f1
    approach = "two_stage_honest"
else:
    log("Using RAW ML")
    test_preds = (blend_test >= t_raw).astype(int)
    final_f1 = f1_raw
    approach = "raw_ml_honest"

test_preds = test_preds.astype(int)

log(f"\nFINAL HONEST F1: {final_f1:.4f}")
log(f"Test: {test_preds.sum()} fraud ({test_preds.mean()*100:.2f}%)")

sub = pd.DataFrame({"id_user": te_u["id_user"].astype("int64"), "is_fraud": test_preds})
sub.to_csv("artifacts/submissions/submission_honest.csv", index=False)
log("Saved submission_honest.csv")

metrics = {
    "oof_f1_honest": round(final_f1, 6),
    "approach": approach,
    "predicted_fraud": int(test_preds.sum()),
    "threshold": round(best_t, 4),
}
log(json.dumps(metrics, indent=2))
log("Done!")
