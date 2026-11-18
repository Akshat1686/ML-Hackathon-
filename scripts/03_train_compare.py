"""Step 3 - Train and compare all candidate-scoring models.

Protocol
  * 5-fold GroupKFold over the 80 training trips (groups = trip_id, so no trip is split)
    -> CV macro F1 / Top-1 / log loss (mean +- std over folds)
  * Refit on all training trips -> evaluate on the 10 student-test trips
  * Only label-valid observations are used for fitting and scoring.
Outputs: results/model_comparison.csv, results/cv_folds.csv, results/training_histories.json,
         models/<name>.joblib, figures.
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
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap.candidates import FEATURES  # noqa: E402
from pathsnap.config import DATA_DIR, FEAT_DIR, FIG_DIR, MODEL_DIR, RESULTS_DIR, SEED  # noqa: E402
from pathsnap.data_io import load_trip  # noqa: E402
from pathsnap.evaluation import observation_metrics, row_metrics, sequence_metrics, top1_predictions  # noqa: E402
from pathsnap.models import MODEL_SPECS  # noqa: E402

os.makedirs(MODEL_DIR, exist_ok=True)
rng = np.random.default_rng(SEED)

train = pd.read_pickle(os.path.join(FEAT_DIR, "train.pkl"))
test = pd.read_pickle(os.path.join(FEAT_DIR, "student_test.pkl"))
train = train[train.label_valid].reset_index(drop=True)
X, y, groups = train[FEATURES].values, train.y.values, train.trip_id.values
Xt = test[FEATURES].values
test_trips = {t: load_trip(os.path.join(DATA_DIR, "student_test"), t) for t in test.trip_id.unique()}


def subsample_idx(idx, max_rows):
    if max_rows is None or len(idx) <= max_rows:
        return idx
    return np.sort(rng.choice(idx, max_rows, replace=False))


def fit(name, idx):
    family, cfg, factory, max_rows = MODEL_SPECS[name]
    m = factory()
    sidx = subsample_idx(idx, max_rows)
    t0 = time.time()
    m.fit(X[sidx], y[sidx])
    return m, time.time() - t0


only = sys.argv[1:] or list(MODEL_SPECS)
gkf = GroupKFold(n_splits=5)
fold_rows, summary, histories = [], [], {}
for name in only:
    family, cfg, _, _ = MODEL_SPECS[name]
    print(f"=== {name}", flush=True)
    for k, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        m, tfit = fit(name, tr_idx)
        p = m.predict_proba(X[va_idx])[:, 1]
        r = row_metrics(y[va_idx], p)
        va = train.iloc[va_idx][["trip_id", "obs", "arc", "y", "true_arc", "label_valid"]].copy()
        va["p"] = p
        r.update(observation_metrics(top1_predictions(va)))
        r.update(model=name, fold=k, fit_s=tfit)
        fold_rows.append(r)
        print(f"  fold {k}: macroF1={r['macro_f1']:.4f} top1={r['top1_acc']:.4f} ll={r['log_loss']:.4f} ({tfit:.1f}s)", flush=True)
    cv = pd.DataFrame([f for f in fold_rows if f["model"] == name])
    # refit on all training trips, evaluate on student test
    m, tfit = fit(name, np.arange(len(X)))
    t0 = time.time()
    pt = m.predict_proba(Xt)[:, 1]
    tinf = time.time() - t0
    te = test[["trip_id", "obs", "arc", "y", "true_arc", "label_valid"]].copy()
    te["p"] = pt
    v = test.label_valid.values
    rm = row_metrics(test.y.values[v], pt[v])
    top = top1_predictions(te)
    om = observation_metrics(top)
    sm = sequence_metrics(top, test_trips)
    n_obs = te.groupby(["trip_id", "obs"]).ngroups
    summary.append(dict(
        model=name, family=family, config=cfg,
        cv_macro_f1=cv.macro_f1.mean(), cv_macro_f1_std=cv.macro_f1.std(), cv_top1=cv.top1_acc.mean(),
        cv_top1_std=cv.top1_acc.std(), cv_log_loss=cv.log_loss.mean(),
        test_top1=om["top1_acc"], test_macro_f1=rm["macro_f1"], test_precision=rm["precision"], test_recall=rm["recall"],
        test_f1_pos=rm["f1_pos"], test_accuracy=rm["accuracy"], test_log_loss=rm["log_loss"],
        test_route_continuity=sm["route_continuity"], test_route_f1=sm["route_f1"],
        candidate_miss_rate=om["candidate_miss_rate"], train_time_s=tfit, infer_ms_per_1k_obs=1000 * tinf / n_obs * 1000,
    ))
    print(summary[-1], flush=True)
    joblib.dump(m, os.path.join(MODEL_DIR, f"{name}.joblib"), compress=3)
    te[["trip_id", "obs", "arc", "p"]].to_pickle(os.path.join(FEAT_DIR, f"test_scores_{name}.pkl"))
    # confusion matrix (row level, threshold 0.5) on student test
    cm = confusion_matrix(test.y.values[v], (pt[v] >= .5).astype(int))
    ConfusionMatrixDisplay(cm, display_labels=["not route arc", "route arc"]).plot(values_format="d")
    plt.title(f"{name} - student test (row level)")
    plt.savefig(os.path.join(FIG_DIR, f"cm_{name}.png"), dpi=100, bbox_inches="tight")
    plt.close()
    # real epoch-wise histories only where the algorithm produces them
    if name == "MLP":
        mlp = m.steps[-1][1]
        histories["MLP"] = dict(train_loss=list(map(float, mlp.loss_curve_)),
                                val_score=list(map(float, mlp.validation_scores_)))
    if name == "HistGB":
        histories["HistGB"] = dict(train_loss=list(map(float, -m.train_score_)),
                                   val_loss=list(map(float, -m.validation_score_)))

# merge with any previous runs (so models can be run separately)
cmp_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
fold_path = os.path.join(RESULTS_DIR, "cv_folds.csv")
hist_path = os.path.join(RESULTS_DIR, "training_histories.json")
new_s, new_f = pd.DataFrame(summary), pd.DataFrame(fold_rows)
if os.path.exists(cmp_path):
    old = pd.read_csv(cmp_path)
    new_s = pd.concat([old[~old.model.isin(only)], new_s])
    oldf = pd.read_csv(fold_path)
    new_f = pd.concat([oldf[~oldf.model.isin(only)], new_f])
new_s.to_csv(cmp_path, index=False)
new_f.to_csv(fold_path, index=False)
hist = json.load(open(hist_path)) if os.path.exists(hist_path) else {}
hist.update(histories)
json.dump(hist, open(hist_path, "w"))
print(new_s[["model", "cv_macro_f1", "cv_top1", "test_top1", "test_macro_f1", "test_log_loss", "test_route_continuity"]])
