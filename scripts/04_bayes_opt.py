"""Step 4 - Bayesian hyperparameter optimisation of the selected model (HistGradientBoosting).

Objective : mean observation-level Top-1 accuracy over 3-fold GroupKFold on the 80 training
            trips (groups = trips). Top-1 is our primary metric because the decoder consumes the
            per-observation ranking of candidates.
Search    : learning_rate, max_iter, max_leaf_nodes, min_samples_leaf, l2_regularization, max_features
Budget    : 5 initial + 15 GP-EI trials = 20 trials (max_iter is an upper bound; early stopping on a 10% internal split)
Afterwards: refit best params on all train trips, compare untuned vs tuned with the SAME 5-fold CV
            and the SAME student-test evaluation.
Outputs: results/bayes_opt_trials.csv, results/tuned_vs_untuned.csv, results/best_params.json,
         models/final_model.joblib, figures.
"""
import json
import os
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap.bayesopt import Space, bayes_opt  # noqa: E402
from pathsnap.candidates import FEATURES  # noqa: E402
from pathsnap.config import DATA_DIR, FEAT_DIR, FIG_DIR, MODEL_DIR, RESULTS_DIR  # noqa: E402
from pathsnap.data_io import load_trip  # noqa: E402
from pathsnap.evaluation import observation_metrics, row_metrics, sequence_metrics, top1_predictions  # noqa: E402
from pathsnap.models import MODEL_SPECS, make_hgb  # noqa: E402

train = pd.read_pickle(os.path.join(FEAT_DIR, "train.pkl"))
train = train[train.label_valid].reset_index(drop=True)
test = pd.read_pickle(os.path.join(FEAT_DIR, "student_test.pkl"))
X, y, groups = train[FEATURES].values, train.y.values, train.trip_id.values
meta = train[["trip_id", "obs", "arc", "y", "true_arc", "label_valid"]]


def cv_scores(factory, n_splits):
    rows = []
    for tr, va in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = factory()
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[va])[:, 1]
        r = row_metrics(y[va], p)
        v = meta.iloc[va].copy()
        v["p"] = p
        r.update(observation_metrics(top1_predictions(v)))
        rows.append(r)
    return pd.DataFrame(rows)


SPACE = Space({
    "learning_rate": (0.02, 0.3, "log", "float"),
    "max_iter": (100, 600, "lin", "int"),
    "max_leaf_nodes": (15, 127, "log", "int"),
    "min_samples_leaf": (10, 300, "log", "int"),
    "l2_regularization": (1e-4, 10.0, "log", "float"),
    "max_features": (0.4, 1.0, "lin", "float"),
})


def objective(params):
    return float(cv_scores(lambda: make_hgb(params), 3).top1_acc.mean())


t0 = time.time()
best, trials = bayes_opt(objective, SPACE, n_init=5, n_iter=15, seed=42, log=lambda s: print(s, flush=True))
bo_seconds = time.time() - t0
tr_df = pd.DataFrame(trials)
tr_df.to_csv(os.path.join(RESULTS_DIR, "bayes_opt_trials.csv"), index=False)
best_params = {k: best[k] for k in SPACE.names}
json.dump(dict(best_params=best_params, best_cv3_top1=best["objective"], n_trials=len(trials),
               total_seconds=bo_seconds), open(os.path.join(RESULTS_DIR, "best_params.json"), "w"), indent=2)

# ---- untuned vs tuned, same protocol ----
test_trips = {t: load_trip(os.path.join(DATA_DIR, "student_test"), t) for t in test.trip_id.unique()}
Xt = test[FEATURES].values
v = test.label_valid.values
rows = []
for name, factory in [("HistGB (untuned)", MODEL_SPECS["HistGB"][2]), ("HistGB (BO-tuned)", lambda: make_hgb(best_params))]:
    cv = cv_scores(factory, 5)
    m = factory()
    t1 = time.time()
    m.fit(X, y)
    fit_s = time.time() - t1
    t1 = time.time()
    p = m.predict_proba(Xt)[:, 1]
    inf_s = time.time() - t1
    te = test[["trip_id", "obs", "arc", "y", "true_arc", "label_valid"]].copy()
    te["p"] = p
    top = top1_predictions(te)
    rm = row_metrics(test.y.values[v], p[v])
    rows.append(dict(model=name, cv_macro_f1=cv.macro_f1.mean(), cv_macro_f1_std=cv.macro_f1.std(), cv_top1=cv.top1_acc.mean(),
                     cv_log_loss=cv.log_loss.mean(), test_top1=observation_metrics(top)["top1_acc"],
                     test_macro_f1=rm["macro_f1"], test_log_loss=rm["log_loss"],
                     test_route_continuity=sequence_metrics(top, test_trips)["route_continuity"],
                     train_time_s=fit_s, infer_ms_per_1k_obs=1e6 * inf_s / top.shape[0]))
    print(rows[-1], flush=True)
    if "tuned" in name and "untuned" not in name:
        joblib.dump(m, os.path.join(MODEL_DIR, "final_model.joblib"), compress=3)
        te[["trip_id", "obs", "arc", "p"]].to_pickle(os.path.join(FEAT_DIR, "test_scores_final.pkl"))
pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "tuned_vs_untuned.csv"), index=False)

# ---- figures ----
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(tr_df.trial, tr_df.objective, "o", label="trial objective")
ax[0].plot(tr_df.trial, tr_df.objective.cummax(), "-", label="best so far")
ax[0].axvline(4.5, ls="--", color="gray")
ax[0].text(0.2, tr_df.objective.min(), "random init", fontsize=8)
ax[0].set_xlabel("trial")
ax[0].set_ylabel("3-fold group-CV Top-1 accuracy")
ax[0].legend()
ax[0].set_title("Bayesian optimisation progress")
sc = ax[1].scatter(tr_df.learning_rate, tr_df.max_leaf_nodes, c=tr_df.objective, cmap="viridis", s=50)
ax[1].set_xscale("log")
ax[1].set_xlabel("learning_rate")
ax[1].set_ylabel("max_leaf_nodes")
plt.colorbar(sc, ax=ax[1], label="objective")
ax[1].set_title("Explored region")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "bayes_opt.png"), dpi=120)
print("best", best_params)
