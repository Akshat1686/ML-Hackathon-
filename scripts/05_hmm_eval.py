"""Step 5 - HMM/Viterbi: decoder tuning + classifier-only vs classifier+HMM comparison.

Decoder tuning (no student-test leakage into the choice):
    hold out 10 training trips, fit the final configuration on the other 70 trips,
    score the 10 held-out trips at full 1 Hz, grid-search (beta, w_emit) on them.
Comparison:
    for every trained model, decode the 10 student-test trips with the chosen decoder and
    report Top-1 / continuity / route F1 before and after HMM.
Outputs: results/hmm_grid.csv, results/hmm_comparison.csv, results/hmm_per_trip.csv,
         models/inference_config.json, figures.
"""
import json
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap.candidates import DEFAULT_RADIUS, FEATURES, align_route_labels, build_candidate_table  # noqa: E402
from pathsnap.config import DATA_DIR, FEAT_DIR, FIG_DIR, MODEL_DIR, RESULTS_DIR, SEED  # noqa: E402
from pathsnap.data_io import list_trips, load_trip  # noqa: E402
from pathsnap.evaluation import continuity, route_set_scores  # noqa: E402
from pathsnap.hmm import HMMConfig, viterbi_decode  # noqa: E402
from pathsnap.models import make_hgb  # noqa: E402

best_params = json.load(open(os.path.join(RESULTS_DIR, "best_params.json")))["best_params"]


def decode_eval(trip, cand, cfg):
    lab, valid = align_route_labels(trip, return_valid=True)
    dec, st = viterbi_decode(trip, cand, cfg)
    obs = dec.obs.values
    clf = cand.loc[cand.groupby("obs").p.idxmax()].sort_values("obs").arc.values
    v = valid[obs]
    res = dict(trip_id=trip.trip_id, n_obs=len(obs))
    for name, pa in [("clf", clf), ("hmm", dec.pred_arc.values)]:
        res[f"{name}_top1"] = float((pa[v] == lab[obs][v]).mean())
        res[f"{name}_continuity"] = continuity(pa, trip.arcs)
        res[f"{name}_route_f1"] = route_set_scores(pa, trip.route)[2]
    res.update({f"hmm_{k}": v_ for k, v_ in st.items() if k != "n_obs"})
    return res


# ---------- 1. decoder tuning on held-out training trips ----------
train_ids = list_trips(os.path.join(DATA_DIR, "train"))
rng = np.random.default_rng(SEED)
hold = sorted(rng.choice(train_ids, 10, replace=False))
tr = pd.read_pickle(os.path.join(FEAT_DIR, "train.pkl"))
tr = tr[tr.label_valid & ~tr.trip_id.isin(hold)]
m = make_hgb(best_params).fit(tr[FEATURES].values, tr.y.values)
hold_data = []
for tid in hold:
    trip = load_trip(os.path.join(DATA_DIR, "train"), tid)
    c = build_candidate_table(trip, radius=DEFAULT_RADIUS)
    c["p"] = m.predict_proba(c[FEATURES].values)[:, 1]
    hold_data.append((trip, c))
grid = []
for beta in [5, 10, 20, 40, 80]:
    for w in [0.5, 1.0, 2.0]:
        cfg = HMMConfig(beta=beta, w_emit=w)
        rs = pd.DataFrame([decode_eval(t, c, cfg) for t, c in hold_data])
        grid.append(dict(beta=beta, w_emit=w, hmm_top1=rs.hmm_top1.mean(), clf_top1=rs.clf_top1.mean(),
                         hmm_continuity=rs.hmm_continuity.mean(), hmm_route_f1=rs.hmm_route_f1.mean()))
        print(grid[-1], flush=True)
grid = pd.DataFrame(grid)
grid.to_csv(os.path.join(RESULTS_DIR, "hmm_grid.csv"), index=False)
best = grid.sort_values(["hmm_top1", "hmm_route_f1"], ascending=False).iloc[0]
cfg = HMMConfig(beta=float(best.beta), w_emit=float(best.w_emit))
print("chosen decoder", cfg, flush=True)
json.dump(dict(candidate_radius_m=DEFAULT_RADIUS, features=FEATURES, hmm=cfg.to_dict(), model="HistGradientBoosting (BO-tuned)",
               best_params=best_params, decoder_tuning_trips=list(map(str, hold))),
          open(os.path.join(MODEL_DIR, "inference_config.json"), "w"), indent=2)

# ---------- 2. classifier-only vs classifier+HMM on student test, for every model ----------
test = pd.read_pickle(os.path.join(FEAT_DIR, "student_test.pkl"))
trips = {t: load_trip(os.path.join(DATA_DIR, "student_test"), t) for t in test.trip_id.unique()}
score_files = {f[len("test_scores_"):-4]: os.path.join(FEAT_DIR, f) for f in os.listdir(FEAT_DIR) if f.startswith("test_scores_")}
comp, per_trip = [], []
for name, path in sorted(score_files.items()):
    sc = pd.read_pickle(path)
    df = test.merge(sc, on=["trip_id", "obs", "arc"])
    rows = [decode_eval(trips[t], g, cfg) for t, g in df.groupby("trip_id")]
    rs = pd.DataFrame(rows)
    rs["model"] = name
    per_trip.append(rs)
    w = rs.n_obs / rs.n_obs.sum()
    comp.append(dict(model=name, clf_top1=(rs.clf_top1 * w).sum(), hmm_top1=(rs.hmm_top1 * w).sum(),
                     clf_continuity=rs.clf_continuity.mean(), hmm_continuity=rs.hmm_continuity.mean(),
                     clf_route_f1=rs.clf_route_f1.mean(), hmm_route_f1=rs.hmm_route_f1.mean(),
                     decode_s_total=rs.hmm_decode_s.sum()))
    print(comp[-1], flush=True)
comp = pd.DataFrame(comp)
comp.to_csv(os.path.join(RESULTS_DIR, "hmm_comparison.csv"), index=False)
pt = pd.concat(per_trip)
pt.to_csv(os.path.join(RESULTS_DIR, "hmm_per_trip.csv"), index=False)

# ---------- figures ----------
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
x = np.arange(len(comp))
for a, (c1, c2, t) in zip(ax, [("clf_top1", "hmm_top1", "Top-1 accuracy"),
                               ("clf_continuity", "hmm_continuity", "Route continuity"),
                               ("clf_route_f1", "hmm_route_f1", "Route arc-set F1")]):
    a.bar(x - .2, comp[c1], .4, label="classifier only")
    a.bar(x + .2, comp[c2], .4, label="classifier + HMM")
    a.set_xticks(x)
    a.set_xticklabels(comp.model, rotation=45, ha="right")
    a.set_title(f"{t} (student test)")
    a.set_ylim(max(0, min(comp[c1].min(), comp[c2].min()) - .1), 1)
    a.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "hmm_vs_classifier.png"), dpi=120)
plt.close()
fin = pt[pt.model == "final"].sort_values("trip_id")
plt.figure(figsize=(9, 4))
xx = np.arange(len(fin))
plt.bar(xx - .2, fin.clf_top1, .4, label="classifier only")
plt.bar(xx + .2, fin.hmm_top1, .4, label="classifier + HMM")
plt.xticks(xx, fin.trip_id, rotation=45)
plt.ylabel("Top-1 accuracy")
plt.title("Final model: per student-test trip")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "hmm_per_trip_final.png"), dpi=120)
pv = grid.pivot(index="beta", columns="w_emit", values="hmm_top1")
plt.figure(figsize=(5, 4))
plt.imshow(pv.values, cmap="viridis", aspect="auto")
plt.xticks(range(len(pv.columns)), pv.columns)
plt.yticks(range(len(pv.index)), pv.index)
plt.xlabel("w_emit")
plt.ylabel("beta (m)")
for i in range(pv.shape[0]):
    for j in range(pv.shape[1]):
        plt.text(j, i, f"{pv.values[i, j]:.3f}", ha="center", va="center", color="w", fontsize=8)
plt.title("Decoder grid: Top-1 on held-out train trips")
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "hmm_grid.png"), dpi=120)
print("done")
