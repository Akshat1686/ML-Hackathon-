"""Step 10 - Feature selection evidence: permutation importance + feature-group ablation.

Uses ONLY training trips: fit the tuned HGB on 70 training trips, measure on 10 held-out
training trips (same held-out set as the decoder tuning).
  * Permutation importance: drop in observation-level Top-1 accuracy when one feature is shuffled.
  * Group ablation: retrain without each feature group and measure Top-1 on the held-out trips.
Outputs: results/feature_importance.csv, results/feature_ablation.csv, figures/feature_importance.png
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap.candidates import FEATURES  # noqa: E402
from pathsnap.config import FEAT_DIR, FIG_DIR, RESULTS_DIR, SEED  # noqa: E402
from pathsnap.evaluation import observation_metrics, top1_predictions  # noqa: E402
from pathsnap.models import make_hgb  # noqa: E402

bp = json.load(open(os.path.join(RESULTS_DIR, "best_params.json")))["best_params"]
df = pd.read_pickle(os.path.join(FEAT_DIR, "train.pkl"))
df = df[df.label_valid].reset_index(drop=True)
trips = sorted(df.trip_id.unique())
rng = np.random.default_rng(SEED)
hold = set(rng.choice(trips, 10, replace=False))
tr, va = df[~df.trip_id.isin(hold)], df[df.trip_id.isin(hold)].reset_index(drop=True)
meta = va[["trip_id", "obs", "arc", "y", "true_arc", "label_valid"]].copy()


def top1(model, X, cols):
    m = meta.copy()
    m["p"] = model.predict_proba(X[cols].values)[:, 1]
    return observation_metrics(top1_predictions(m))["top1_acc"]


model = make_hgb(bp).fit(tr[FEATURES].values, tr.y.values)
base = top1(model, va, FEATURES)
print("baseline held-out top1", base, flush=True)

rows = []
for f in FEATURES:
    drops = []
    for r in range(3):
        Xp = va[FEATURES].copy()
        Xp[f] = rng.permutation(Xp[f].values)
        drops.append(base - top1(model, Xp, FEATURES))
    rows.append(dict(feature=f, importance_mean=np.mean(drops), importance_std=np.std(drops)))
    print(rows[-1], flush=True)
imp = pd.DataFrame(rows).sort_values("importance_mean", ascending=False)
imp.to_csv(os.path.join(RESULTS_DIR, "feature_importance.csv"), index=False)

GROUPS = {
    "distance (dist, rank, minus_min, ratio)": ["dist", "dist_rank", "dist_minus_min", "dist_ratio"],
    "heading (diff, cos, valid)": ["heading_diff", "heading_cos", "heading_valid"],
    "along-arc motion": ["along_move", "along_move_norm"],
    "temporal context (ctx_dist_mean/max)": ["ctx_dist_mean", "ctx_dist_max"],
    "arc geometry (len, t_frac, endpoint)": ["arc_len", "t_frac", "endpoint_dist"],
    "network (degrees, reverse, n_cands)": ["deg_out_to", "deg_in_from", "has_reverse", "n_cands"],
    "kinematics (speed, time_gap)": ["speed", "time_gap"],
}
abl = [dict(removed="none (all 20 features)", n_features=len(FEATURES), heldout_top1=base)]
for g, cols in GROUPS.items():
    keep = [f for f in FEATURES if f not in cols]
    m = make_hgb(bp).fit(tr[keep].values, tr.y.values)
    abl.append(dict(removed=g, n_features=len(keep), heldout_top1=top1(m, va, keep)))
    print(abl[-1], flush=True)
abl = pd.DataFrame(abl)
abl["delta_vs_all"] = abl.heldout_top1 - base
abl.to_csv(os.path.join(RESULTS_DIR, "feature_ablation.csv"), index=False)

fig, ax = plt.subplots(1, 2, figsize=(15, 6))
ax[0].barh(imp.feature[::-1], imp.importance_mean[::-1], xerr=imp.importance_std[::-1], color="tab:blue")
ax[0].set_xlabel("drop in Top-1 accuracy when shuffled")
ax[0].set_title("Permutation importance (10 held-out training trips)")
a = abl.iloc[1:].sort_values("delta_vs_all")
ax[1].barh(a.removed, a.delta_vs_all, color="tab:red")
ax[1].set_xlabel("change in held-out Top-1 when group removed (retrained)")
ax[1].set_title(f"Feature-group ablation (all features: {base:.4f})")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "feature_importance.png"), dpi=120)
print("done")
