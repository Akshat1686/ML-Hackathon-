"""End-to-end inference for one trip (used by the API and the evaluation scripts).

trip files -> candidates + features -> classifier probabilities -> HMM/Viterbi -> arc per observation
The reference route is NEVER used here; scoring is done separately in `score_prediction`.
"""
from __future__ import annotations

import json
import os
import time

import joblib
import numpy as np
import pandas as pd

from .candidates import FEATURES, align_route_labels, build_candidate_table
from .config import MODEL_DIR
from .data_io import Trip
from .evaluation import continuity, route_set_scores
from .hmm import HMMConfig, viterbi_decode


class PathSnapModel:
    def __init__(self, model_dir: str = MODEL_DIR):
        self.clf = joblib.load(os.path.join(model_dir, "final_model.joblib"))
        cfg = json.load(open(os.path.join(model_dir, "inference_config.json")))
        self.radius = cfg["candidate_radius_m"]
        self.features = cfg["features"]
        assert self.features == FEATURES, "feature list mismatch between training and inference"
        self.hmm = HMMConfig(**cfg["hmm"])
        self.meta = cfg

    def predict(self, trip: Trip) -> dict:
        t0 = time.time()
        cand = build_candidate_table(trip, radius=self.radius)            # no labels passed
        t1 = time.time()
        cand["p"] = self.clf.predict_proba(cand[self.features].values)[:, 1]
        t2 = time.time()
        clf_top = cand.loc[cand.groupby("obs").p.idxmax()].sort_values("obs")
        dec, stats = viterbi_decode(trip, cand, self.hmm)
        t3 = time.time()
        return dict(obs=dec.obs.values, hmm_arc=dec.pred_arc.values, clf_arc=clf_top.arc.values,
                    segment=dec.segment.values, n_candidates=int(len(cand)), hmm_stats=stats,
                    timing_s=dict(candidates_features=t1 - t0, classifier=t2 - t1, viterbi=t3 - t2, total=t3 - t0))


def score_prediction(trip: Trip, pred: dict) -> dict:
    """Server-side evaluation against the reference route (labels never leave the server)."""
    lab, valid = align_route_labels(trip, return_valid=True)
    obs = pred["obs"]
    v = valid[obs]
    truth = lab[obs]
    out = {}
    for key, name in [("hmm_arc", "hmm"), ("clf_arc", "classifier_only")]:
        pa = pred[key]
        p, r, f = route_set_scores(pa, trip.route)
        out[name] = dict(top1_accuracy=float((pa[v] == truth[v]).mean()),
                         route_continuity=continuity(pa, trip.arcs),
                         route_precision=p, route_recall=r, route_f1=f)
    out["n_scored_observations"] = int(v.sum())
    out["n_observations"] = int(len(obs))
    return out


def route_geometry(trip: Trip, arcs_seq: np.ndarray):
    """Distinct arcs in visiting order -> list of [[lat, lon], [lat, lon]] segments."""
    seq = pd.unique(arcs_seq)
    a, b = trip.nodes[trip.arcs[seq, 0]], trip.nodes[trip.arcs[seq, 1]]
    return [[[float(a[k, 1]), float(a[k, 0])], [float(b[k, 1]), float(b[k, 0])]] for k in range(len(seq))]
