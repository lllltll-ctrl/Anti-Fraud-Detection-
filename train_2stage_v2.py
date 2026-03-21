"""Two-stage v2: train on ALL data, override pure components in prediction."""
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

# ============================================================
# 2. BUILD GRAPHS & FIND CONNECTED COMPONENTS
# ============================================================
log("Building card graph...")
card_to_users = defaultdict(set)
user_to_cards = defaultdict(set)
for uid, card in zip(all_tx["id_user"], all_tx["card_mask_hash"]):
    card_to_users[card].add(uid)
    user_to_cards[uid].add(card)

log("Finding connected components (card only)...")
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

log(f"  Total components: {len(components)}")

# Classify components
comp_labels = {}  # comp_id -> 'pure_fraud', 'pure_legit', 'mixed'
for i, comp in enumerate(components):
    train_in = comp & train_ids
    fraud_in = train_in & fraud_set
    legit_in = train_in - fraud_set
    if len(fraud_in) > 0 and len(legit_in) == 0:
        comp_labels[i] = "pure_fraud"
    elif len(fraud_in) == 0:
        comp_labels[i] = "pure_legit"
    else:
        comp_labels[i] = "mixed"

# Map users to component type
def get_comp_type(uid):
    return comp_labels.get(user_component.get(uid, -1), "pure_legit")

train_comp_types = [get_comp_type(uid) for uid in tr_u["id_user"]]
test_comp_types = [get_comp_type(uid) for uid in te_u["id_user"]]

n_pf = sum(1 for t in train_comp_types if t == "pure_fraud")
n_pl = sum(1 for t in train_comp_types if t == "pure_legit")
n_mx = sum(1 for t in train_comp_types if t == "mixed")
log(f"  Train: {n_pf} pure_fraud, {n_pl} pure_legit, {n_mx} mixed")

n_pf_te = sum(1 for t in test_comp_types if t == "pure_fraud")
n_pl_te = sum(1 for t in test_comp_types if t == "pure_legit")
n_mx_te = sum(1 for t in test_comp_types if t == "mixed")
log(f"  Test:  {n_pf_te} pure_fraud, {n_pl_te} pure_legit, {n_mx_te} mixed")

# ============================================================
# 3. BUILD COMPREHENSIVE FEATURES
# ============================================================
log("Building features...")

# Also build holder graph
log("  Building holder graph...")
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


def build_features_for(user_ids_list, tx_df, users_df):
    """Build features for a list of user IDs."""
    tx_sub = tx_df[tx_df["id_user"].isin(set(user_ids_list))].copy()
    tx_sub = tx_sub.sort_values(["id_user", "ts"])
    users_sub = users_df.set_index("id_user").reindex(user_ids_list)

    feats = pd.DataFrame(index=user_ids_list)
    feats.index.name = "id_user"

    # User-level features
    feats["gender"] = users_sub["gender"].values
    feats["reg_country"] = users_sub["reg_country"].values
    feats["traffic_type"] = users_sub["traffic_type"].values
    feats["email_domain"] = users_sub["email"].str.split("@").str[1].values
    feats["reg_hour"] = users_sub["ts_reg"].dt.hour.values
    feats["reg_weekday"] = users_sub["ts_reg"].dt.weekday.values

    # Transaction aggregates
    g = tx_sub.groupby("id_user")

    feats["tx_count"] = g.size().reindex(user_ids_list, fill_value=0).values
    tx_clip = feats["tx_count"].clip(lower=1)

    feats["success_count"] = tx_sub[tx_sub.status == "success"].groupby("id_user").size().reindex(user_ids_list, fill_value=0).values
    feats["fail_count"] = tx_sub[tx_sub.status == "fail"].groupby("id_user").size().reindex(user_ids_list, fill_value=0).values
    feats["fail_ratio"] = feats["fail_count"] / tx_clip
    feats["success_ratio"] = feats["success_count"] / tx_clip

    # Amount
    feats["amount_sum"] = g["amount"].sum().reindex(user_ids_list, fill_value=0).values
    feats["amount_mean"] = g["amount"].mean().reindex(user_ids_list, fill_value=0).values
    feats["amount_std"] = g["amount"].std().reindex(user_ids_list, fill_value=0).values
    feats["amount_max"] = g["amount"].max().reindex(user_ids_list, fill_value=0).values
    feats["amount_min"] = g["amount"].min().reindex(user_ids_list, fill_value=0).values
    feats["amount_range"] = feats["amount_max"] - feats["amount_min"]
    feats["amount_cv"] = feats["amount_std"] / feats["amount_mean"].clip(lower=0.01)

    # Cards
    feats["unique_cards"] = g["card_mask_hash"].nunique().reindex(user_ids_list, fill_value=0).values
    feats["unique_card_holders"] = g["card_holder"].nunique().reindex(user_ids_list, fill_value=0).values
    feats["unique_card_countries"] = g["card_country"].nunique().reindex(user_ids_list, fill_value=0).values
    feats["unique_payment_countries"] = g["payment_country"].nunique().reindex(user_ids_list, fill_value=0).values
    feats["unique_card_brands"] = g["card_brand"].nunique().reindex(user_ids_list, fill_value=0).values
    feats["card_holder_per_card"] = feats["unique_card_holders"] / feats["unique_cards"].clip(lower=1)
    feats["cards_per_tx"] = feats["unique_cards"] / tx_clip
    feats["holders_per_tx"] = feats["unique_card_holders"] / tx_clip

    # Errors
    feats["fraud_error_count"] = tx_sub[tx_sub.error_group == "fraud"].groupby("id_user").size().reindex(user_ids_list, fill_value=0).values
    feats["antifraud_count"] = tx_sub[tx_sub.error_group == "antifraud"].groupby("id_user").size().reindex(user_ids_list, fill_value=0).values
    feats["fraud_error_ratio"] = feats["fraud_error_count"] / tx_clip
    feats["antifraud_ratio"] = feats["antifraud_count"] / tx_clip
    feats["unique_error_groups"] = g["error_group"].nunique().reindex(user_ids_list, fill_value=0).values

    # Transaction types
    feats["unique_tx_types"] = g["transaction_type"].nunique().reindex(user_ids_list, fill_value=0).values
    feats["card_init_count"] = tx_sub[tx_sub.transaction_type == "card_init"].groupby("id_user").size().reindex(user_ids_list, fill_value=0).values
    feats["card_recurring_count"] = tx_sub[tx_sub.transaction_type == "card_recurring"].groupby("id_user").size().reindex(user_ids_list, fill_value=0).values
    feats["card_init_ratio"] = feats["card_init_count"] / tx_clip
    feats["resign_count"] = tx_sub[tx_sub.transaction_type == "resign"].groupby("id_user").size().reindex(user_ids_list, fill_value=0).values

    # Card types
    feats["has_prepaid"] = tx_sub[tx_sub.card_type.str.contains("PREPAID", case=False, na=False)].groupby("id_user").size().reindex(user_ids_list, fill_value=0).clip(upper=1).values
    feats["has_credit"] = tx_sub[tx_sub.card_type.str.contains("CREDIT", case=False, na=False)].groupby("id_user").size().reindex(user_ids_list, fill_value=0).clip(upper=1).values

    # Country mismatch
    tx_sub["card_pay_mismatch"] = (tx_sub["card_country"] != tx_sub["payment_country"]).astype(int)
    feats["country_mismatch_ratio"] = tx_sub.groupby("id_user")["card_pay_mismatch"].mean().reindex(user_ids_list, fill_value=0).values
    feats["country_mismatch_count"] = tx_sub.groupby("id_user")["card_pay_mismatch"].sum().reindex(user_ids_list, fill_value=0).values

    # Card reg country mismatch
    tx_with_reg = tx_sub.merge(users_df[["id_user", "reg_country"]], on="id_user", how="left")
    tx_with_reg["card_reg_mismatch"] = (tx_with_reg["card_country"] != tx_with_reg["reg_country"]).astype(int)
    feats["card_reg_mismatch_ratio"] = tx_with_reg.groupby("id_user")["card_reg_mismatch"].mean().reindex(user_ids_list, fill_value=0).values

    # Card switch
    tx_sub["prev_card"] = tx_sub.groupby("id_user")["card_mask_hash"].shift(1)
    tx_sub["card_switch"] = ((tx_sub["card_mask_hash"] != tx_sub["prev_card"]) & tx_sub["prev_card"].notna()).astype(int)
    feats["card_switch_count"] = tx_sub.groupby("id_user")["card_switch"].sum().reindex(user_ids_list, fill_value=0).values
    feats["card_switch_ratio"] = feats["card_switch_count"] / tx_clip

    # Temporal
    first_tx = g["ts"].min().reindex(user_ids_list)
    last_tx = g["ts"].max().reindex(user_ids_list)
    reg_ts = pd.to_datetime(pd.Series(users_sub["ts_reg"].values, index=user_ids_list), utc=True)
    feats["minutes_to_first_tx"] = ((first_tx - reg_ts).dt.total_seconds() / 60).values
    feats["tx_span_minutes"] = ((last_tx - first_tx).dt.total_seconds() / 60).values
    feats["minutes_to_first_success"] = ((tx_sub[tx_sub.status == "success"].groupby("id_user")["ts"].min().reindex(user_ids_list) - reg_ts).dt.total_seconds() / 60).values

    # Time gaps
    tx_sub["gap_sec"] = tx_sub.groupby("id_user")["ts"].diff().dt.total_seconds()
    feats["mean_gap_sec"] = tx_sub.groupby("id_user")["gap_sec"].mean().reindex(user_ids_list, fill_value=0).values
    feats["min_gap_sec"] = tx_sub.groupby("id_user")["gap_sec"].min().reindex(user_ids_list, fill_value=0).values
    feats["log_mean_gap"] = np.log1p(np.clip(feats["mean_gap_sec"], 0, None))
    feats["min_fail_gap_sec"] = tx_sub[tx_sub.status == "fail"].groupby("id_user")["gap_sec"].min().reindex(user_ids_list, fill_value=-1).values

    # Time of day
    tx_sub["hour"] = tx_sub["ts"].dt.hour
    feats["mode_hour"] = tx_sub.groupby("id_user")["hour"].apply(lambda x: x.mode().iloc[0] if len(x) > 0 else 12).reindex(user_ids_list, fill_value=12).values
    feats["night_tx_ratio"] = tx_sub.assign(n=tx_sub.hour.isin([0, 1, 2, 3, 4, 5]).astype(int)).groupby("id_user")["n"].mean().reindex(user_ids_list, fill_value=0).values
    feats["business_hours_ratio"] = tx_sub.assign(b=tx_sub.hour.isin(range(9, 18)).astype(int)).groupby("id_user")["b"].mean().reindex(user_ids_list, fill_value=0).values

    # First hour/6h activity
    first_tx_dict = first_tx.to_dict()
    tx_sub["hrs_since_first"] = tx_sub.apply(
        lambda r: (r["ts"] - first_tx_dict.get(r["id_user"], r["ts"])).total_seconds() / 3600, axis=1
    )
    feats["tx_first_hour"] = tx_sub[tx_sub.hrs_since_first <= 1].groupby("id_user").size().reindex(user_ids_list, fill_value=0).values
    feats["fail_first_hour"] = tx_sub[(tx_sub.hrs_since_first <= 1) & (tx_sub.status == "fail")].groupby("id_user").size().reindex(user_ids_list, fill_value=0).values
    feats["tx_first_6h"] = tx_sub[tx_sub.hrs_since_first <= 6].groupby("id_user").size().reindex(user_ids_list, fill_value=0).values

    # Fail->success pattern
    tx_sub["prev_status"] = tx_sub.groupby("id_user")["status"].shift(1)
    tx_sub["prev_amount"] = tx_sub.groupby("id_user")["amount"].shift(1)
    tx_sub["fail_then_success"] = ((tx_sub.prev_status == "fail") & (tx_sub.status == "success") & (tx_sub.amount == tx_sub.prev_amount)).astype(int)
    feats["fail_then_success_count"] = tx_sub.groupby("id_user")["fail_then_success"].sum().reindex(user_ids_list, fill_value=0).values

    # Max fail streak
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

    # Small amount ratio
    feats["small_amount_ratio"] = tx_sub.assign(s=(tx_sub.amount <= 5).astype(int)).groupby("id_user")["s"].mean().reindex(user_ids_list, fill_value=0).values

    # Amount per hour max
    tx_sub["tx_hour_key"] = tx_sub["ts"].dt.floor("h")
    hourly = tx_sub.groupby(["id_user", "tx_hour_key"])["amount"].sum().reset_index()
    feats["amount_per_hour_max"] = hourly.groupby("id_user")["amount"].max().reindex(user_ids_list, fill_value=0).values

    # Max tx per day
    tx_sub["tx_day"] = tx_sub["ts"].dt.date
    daily = tx_sub.groupby(["id_user", "tx_day"]).size().reset_index(name="cnt")
    feats["max_tx_per_day"] = daily.groupby("id_user")["cnt"].max().reindex(user_ids_list, fill_value=0).values

    # Max users sharing any of this user's cards
    max_shared = []
    for uid in user_ids_list:
        cards = user_to_cards.get(uid, set())
        mx = 0
        for c in cards:
            mx = max(mx, len(card_to_users.get(c, set())))
        max_shared.append(mx)
    feats["max_users_per_card"] = max_shared

    # Log transforms
    for c in ["tx_count", "amount_sum", "card_switch_count", "minutes_to_first_tx", "mean_gap_sec"]:
        feats[f"log_{c}"] = np.log1p(np.clip(feats[c].values.astype(float), 0, None))

    # Risk combo
    feats["risk_combo"] = (
        (feats["card_switch_count"] > 3).astype(int)
        + (feats["unique_card_holders"] > 1).astype(int)
        + (feats["fail_ratio"] > 0.5).astype(int)
        + (feats["country_mismatch_ratio"] > 0).astype(int)
    )

    # Fail ratio first hour
    feats["fail_ratio_first_hour"] = feats["fail_first_hour"] / feats["tx_first_hour"].clip(lower=1)
    feats["tx_first_6h_ratio"] = feats["tx_first_6h"] / tx_clip

    # Component type as feature
    feats["comp_type_num"] = [
        1 if get_comp_type(uid) == "pure_fraud" else (-1 if get_comp_type(uid) == "pure_legit" else 0)
        for uid in user_ids_list
    ]

    # Component size
    feats["comp_size"] = [len(components[user_component.get(uid, 0)]) for uid in user_ids_list]
    feats["log_comp_size"] = np.log1p(feats["comp_size"])

    # Component fraud ratio (for train users in same component)
    comp_fraud_ratio = {}
    for i, comp in enumerate(components):
        train_in = comp & train_ids
        if train_in:
            fraud_in = train_in & fraud_set
            comp_fraud_ratio[i] = len(fraud_in) / len(train_in)
        else:
            comp_fraud_ratio[i] = 0.0
    feats["comp_fraud_ratio"] = [comp_fraud_ratio.get(user_component.get(uid, 0), 0.0) for uid in user_ids_list]

    # ---- GRAPH FEATURES (card) ----
    graph_feats = np.zeros((len(user_ids_list), 6))
    for idx, uid in enumerate(user_ids_list):
        cards = user_to_cards.get(uid, set())
        neighbors = set()
        fraud_neighbors = set()
        max_density = 0.0
        n_fraud_cards = 0
        for card in cards:
            others = card_to_users.get(card, set()) - {uid}
            neighbors |= others
            fn = others & fraud_set
            fraud_neighbors |= fn
            if others:
                d = len(fn) / len(others)
                max_density = max(max_density, d)
            if fn:
                n_fraud_cards += 1
        nt = len(neighbors)
        nf = len(fraud_neighbors)
        graph_feats[idx] = [
            nf, nt, nf / max(nt, 1), max_density,
            n_fraud_cards, n_fraud_cards / max(len(cards), 1)
        ]
    for j, name in enumerate(["graph_card_fraud_n", "graph_card_total_n", "graph_card_fraud_ratio",
                               "graph_card_max_density", "graph_n_fraud_cards", "graph_fraud_card_ratio"]):
        feats[name] = graph_feats[:, j]

    # ---- GRAPH FEATURES (holder) ----
    holder_feats = np.zeros((len(user_ids_list), 3))
    for idx, uid in enumerate(user_ids_list):
        holders = user_to_holders.get(uid, set())
        h_neighbors = set()
        h_fraud = set()
        for h in holders:
            others = holder_to_users.get(h, set()) - {uid}
            h_neighbors |= others
            h_fraud |= (others & fraud_set)
        hn = len(h_neighbors)
        hf = len(h_fraud)
        holder_feats[idx] = [hf, hn, hf / max(hn, 1)]
    for j, name in enumerate(["graph_holder_fraud_n", "graph_holder_total_n", "graph_holder_fraud_ratio"]):
        feats[name] = holder_feats[:, j]

    # Combined graph
    feats["graph_any_fraud"] = ((graph_feats[:, 0] > 0) | (holder_feats[:, 0] > 0)).astype(int)
    feats["graph_total_fraud"] = graph_feats[:, 0] + holder_feats[:, 0]

    feats = feats.reset_index()
    feats = feats.rename(columns={"index": "id_user"})
    return feats


log("  Building train features...")
train_feats = build_features_for(tr_u["id_user"].tolist(), all_tx, all_users_df)
log(f"  Train: {train_feats.shape}")

log("  Building test features...")
test_feats = build_features_for(te_u["id_user"].tolist(), all_tx, all_users_df)
log(f"  Test: {test_feats.shape}")

# Add target
train_feats = train_feats.merge(tr_u[["id_user", "is_fraud"]], on="id_user")
y = train_feats["is_fraud"].values

# ============================================================
# 4. TARGET ENCODING
# ============================================================
log("Target encoding...")
gm = y.mean()
smoothing = 50
te_encode_cols = ["reg_country", "gender", "traffic_type", "email_domain"]

for te_col in te_encode_cols:
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

# Also encode error_group cross-features
# err_fraud_ratio: ratio of fraud_error to all errors per user
train_feats["err_fraud_ratio"] = train_feats["fraud_error_count"] / (train_feats["fraud_error_count"] + train_feats["antifraud_count"]).clip(lower=1)
test_feats["err_fraud_ratio"] = test_feats["fraud_error_count"] / (test_feats["fraud_error_count"] + test_feats["antifraud_count"]).clip(lower=1)

# ============================================================
# 5. PREPARE FEATURES
# ============================================================
EXCLUDED = {"id_user", "is_fraud"}
cat_cols = ["gender", "reg_country", "traffic_type", "email_domain"]
feature_cols = [c for c in train_feats.columns if c not in EXCLUDED]
num_cols = [c for c in feature_cols if c not in cat_cols]

log(f"Total features: {len(feature_cols)}, Cat: {len(cat_cols)}, Num: {len(num_cols)}")

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

# ============================================================
# 6. TRAIN ENSEMBLE ON ALL DATA
# ============================================================
log("Training ensemble on ALL data...")

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
    model_labels.append(name)

    if name == configs[0][0]:
        imp = dict(zip(feature_cols, m.feature_importances_))
        top30 = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:30]
        log("  Top-30 features:")
        for fn, fv in top30:
            log(f"    {fn}: {fv}")

# ============================================================
# 7. BLEND
# ============================================================
log("\n=== BLEND ===")
n_models = len(all_oofs)

# Equal blend
avg_oof = np.mean(all_oofs, axis=0)
t_eq, f1_eq = find_best_threshold(y, avg_oof)
log(f"Equal blend (raw): F1={f1_eq:.4f}")

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

log(f"Optimized blend (raw): F1={best_blend_f1:.4f}")
for i, lbl in enumerate(model_labels):
    if best_weights[i] > 0.001:
        log(f"  {lbl}: {best_weights[i]:.3f}")

# Get blended OOF and test
blend_oof = sum(best_weights[j] * all_oofs[j] for j in range(n_models))
blend_test = sum(best_weights[j] * all_tests[j] for j in range(n_models))

# ============================================================
# 8. TWO-STAGE: OVERRIDE PURE COMPONENTS
# ============================================================
log("\n=== TWO-STAGE OVERRIDE ===")

# Component type arrays
train_ct = np.array(train_comp_types)
test_ct = np.array(test_comp_types)

# For train: find best threshold on mixed users only
mixed_mask_train = train_ct == "mixed"
mixed_oof = blend_oof[mixed_mask_train]
mixed_y = y[mixed_mask_train]
t_mixed, f1_mixed = find_best_threshold(mixed_y, mixed_oof)
log(f"Mixed-only OOF F1: {f1_mixed:.4f} (threshold={t_mixed:.4f})")

# Sweep mixed threshold for best OVERALL F1
best_overall_f1 = 0
best_t = t_mixed
for t_try in np.arange(0.05, 0.95, 0.005):
    combined = np.zeros(len(y))
    combined[train_ct == "pure_fraud"] = 1
    combined[train_ct == "pure_legit"] = 0
    combined[mixed_mask_train] = (blend_oof[mixed_mask_train] >= t_try).astype(int)
    f1_try = f1_score(y, combined)
    if f1_try > best_overall_f1:
        best_overall_f1 = f1_try
        best_t = t_try

log(f"Best combined F1: {best_overall_f1:.4f} (threshold={best_t:.4f})")

# Also try: use raw ML threshold for ALL users (no override)
t_raw, f1_raw = find_best_threshold(y, blend_oof)
log(f"Raw ML (no override): F1={f1_raw:.4f}")

# Pick best approach
if best_overall_f1 > f1_raw:
    log("Using TWO-STAGE approach (override pure components)")
    # Generate test predictions
    test_preds = np.zeros(len(te))
    test_preds[test_ct == "pure_fraud"] = 1
    test_preds[test_ct == "pure_legit"] = 0
    mixed_mask_test = test_ct == "mixed"
    test_preds[mixed_mask_test] = (blend_test[mixed_mask_test] >= best_t).astype(int)
    final_f1 = best_overall_f1
    approach = "two_stage"
else:
    log("Using RAW ML approach (no override)")
    test_preds = (blend_test >= t_raw).astype(int)
    final_f1 = f1_raw
    approach = "raw_ml"

test_preds = test_preds.astype(int)

# ============================================================
# 9. SAVE
# ============================================================
log(f"\nFINAL: {approach}, F1={final_f1:.4f}")
log(f"Test: {test_preds.sum()} fraud ({test_preds.mean()*100:.2f}%)")
if approach == "two_stage":
    log(f"  Pure fraud (auto): {(test_ct == 'pure_fraud').sum()}")
    log(f"  Pure legit (auto): {(test_ct == 'pure_legit').sum()}")
    log(f"  Mixed ML fraud: {test_preds[test_ct == 'mixed'].sum()}")

sub = pd.DataFrame({"id_user": te_u["id_user"].astype("int64"), "is_fraud": test_preds})
sub.to_csv("artifacts/submissions/submission.csv", index=False)
log("Saved submission.csv")

metrics = {
    "oof_f1": round(final_f1, 6),
    "approach": approach,
    "predicted_fraud": int(test_preds.sum()),
    "predicted_fraud_rate": round(float(test_preds.mean()), 6),
    "mixed_f1": round(f1_mixed, 6),
    "threshold": round(best_t, 4),
    "n_models": n_models,
}
with open("artifacts/reports/baseline_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

log("Done!")
