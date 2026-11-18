"""Quick refit of ONLY the deployed model, using the recorded best hyperparameters.

Use this on a new laptop if loading models/final_model.joblib fails because of a different
scikit-learn version (pickles are version-specific). Needs features/train.pkl
(run scripts/02_build_features.py first, ~1 min). Takes ~1 min.
"""
import json
import os
import sys

import joblib
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap.candidates import FEATURES  # noqa: E402
from pathsnap.config import FEAT_DIR, MODEL_DIR, RESULTS_DIR  # noqa: E402
from pathsnap.models import make_hgb  # noqa: E402

bp = json.load(open(os.path.join(RESULTS_DIR, "best_params.json")))["best_params"]
tr = pd.read_pickle(os.path.join(FEAT_DIR, "train.pkl"))
tr = tr[tr.label_valid]
m = make_hgb(bp).fit(tr[FEATURES].values, tr.y.values)
joblib.dump(m, os.path.join(MODEL_DIR, "final_model.joblib"), compress=3)
print("refit done ->", os.path.join(MODEL_DIR, "final_model.joblib"), bp)
