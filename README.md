# Anti-Fraud Detection — SKELAR x mono AI Competition

Two-stage fraud detection system combining graph analysis with ML ensemble.

## Results

| Metric | Value |
|--------|-------|
| Honest OOF F1 | **0.8176** |
| Test fraud rate | 3.78% (6,405 / 169,449) |
| Features | 65 |
| Models | 3 LightGBM GBDT |
| Training time | ~50 min |

## Approach

**Stage 1 — Graph Override:**
- Build card graph via `card_mask_hash` → 420K connected components
- Components with ≥90% known fraud → auto-fraud
- Components with 0% fraud → auto-legit
- Mixed → Stage 2

**Stage 2 — ML Classification:**
- 3 LightGBM GBDT models (5-fold CV, different seeds)
- 65 features: behavioral, graph (card + holder), target encoding, interactions
- K-fold graph features (no data leakage)
- Greedy weight blend optimization

**Stage 3 — Post-processing:**
- Fraud propagation through card + holder graph (threshold 60%, min 3 neighbors)
- Calibration cap at train fraud rate (3.78%)

## Files

```
train_2stage_honest.py    # Main pipeline (honest, no leakage)
SOLUTION.md               # Detailed solution writeup (UA)
artifacts/submissions/    # submission.csv output
```

## How to Run

```bash
# Requires: train_users.csv, train_transactions.csv, test_users.csv, test_transactions.csv
pip install numpy pandas lightgbm scikit-learn
python train_2stage_honest.py
# Output: artifacts/submissions/submission.csv
```

## Top-5 Features

1. **g_comp_fraud_ratio** (255) — fraud ratio in graph component
2. **comp_size** (172) — component size
3. **fail_x_cards** (159) — fail ratio × unique cards
4. **traffic_type_te** (131) — target-encoded traffic type
5. **cards_x_holders** (129) — unique cards × unique holders

See [SOLUTION.md](SOLUTION.md) for full analysis and business integration proposal.
