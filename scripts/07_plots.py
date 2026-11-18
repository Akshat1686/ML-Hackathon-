"""Step 7 - Model comparison figures from the recorded results (no new numbers are created here)."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap.config import FIG_DIR, RESULTS_DIR  # noqa: E402

cmp_ = pd.read_csv(os.path.join(RESULTS_DIR, "model_comparison.csv")).sort_values("cv_macro_f1")
folds = pd.read_csv(os.path.join(RESULTS_DIR, "cv_folds.csv"))

fig, ax = plt.subplots(figsize=(9, 4.5))
order = cmp_.model.tolist()
ax.boxplot([folds[folds.model == m].macro_f1 for m in order], tick_labels=order)
ax.set_title("5-fold group CV macro F1 (80 training trips)")
ax.set_ylabel("macro F1")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cv_macro_f1.png"), dpi=120)
plt.close()

x = np.arange(len(cmp_))
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
ax[0].bar(x - .2, cmp_.test_top1, .4, label="Top-1 accuracy (obs level)")
ax[0].bar(x + .2, cmp_.test_macro_f1, .4, label="Macro F1 (candidate level)")
ax[0].set_ylim(.7, 1)
ax[0].set_xticks(x)
ax[0].set_xticklabels(cmp_.model, rotation=30)
ax[0].legend()
ax[0].set_title("Student-test performance")
ax[1].bar(x, cmp_.test_log_loss, color="tab:red")
ax[1].set_xticks(x)
ax[1].set_xticklabels(cmp_.model, rotation=30)
ax[1].set_title("Student-test log loss (lower is better)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "test_comparison.png"), dpi=120)
plt.close()

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(cmp_.infer_ms_per_1k_obs, cmp_.test_top1, s=60)
for _, r in cmp_.iterrows():
    ax.annotate(r.model, (r.infer_ms_per_1k_obs, r.test_top1), fontsize=8, xytext=(4, 4), textcoords="offset points")
ax.set_xscale("log")
ax.set_xlabel("inference time (ms per 1000 GPS observations, log scale)")
ax.set_ylabel("student-test Top-1 accuracy")
ax.set_title("Accuracy vs inference cost")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "accuracy_vs_speed.png"), dpi=120)
plt.close()

hist = json.load(open(os.path.join(RESULTS_DIR, "training_histories.json")))
fig, ax = plt.subplots(1, len(hist), figsize=(6 * len(hist), 4))
ax = np.atleast_1d(ax)
for a, (name, h) in zip(ax, hist.items()):
    if name == "MLP":
        a.plot(h["train_loss"], label="training loss (log loss)")
        a2 = a.twinx()
        a2.plot(h["val_score"], color="tab:orange", label="validation accuracy")
        a2.set_ylabel("validation accuracy")
        a2.legend(loc="center right")
    else:
        a.plot(h["train_loss"], label="training loss")
        a.plot(h["val_loss"], label="validation loss")
    a.set_xlabel("epoch / boosting iteration")
    a.set_title(f"{name} training history (real, from the final fit)")
    a.legend(loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "training_histories.png"), dpi=120)
print("plots done")
