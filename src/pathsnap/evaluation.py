"""Metrics used throughout PathSnap.

Row level (one row = one (observation, candidate arc) pair, binary y):
    accuracy, precision, recall, macro F1, log loss
Observation level:
    Top-1 accuracy  = argmax-probability candidate == reference arc (missed candidates count as wrong)
    candidate miss rate = reference arc not in the candidate set
Sequence level:
    route continuity = share of consecutive observation pairs whose predicted arcs are the same arc
                       or directly connected in the network (arc_a.to == arc_b.from)
    route precision / recall / F1 over the set of distinct arcs on the predicted vs reference route
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score


def row_metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    return dict(
        accuracy=accuracy_score(y, yhat),
        precision=precision_score(y, yhat, zero_division=0),
        recall=recall_score(y, yhat, zero_division=0),
        f1_pos=f1_score(y, yhat, zero_division=0),
        macro_f1=f1_score(y, yhat, average="macro", zero_division=0),
        log_loss=log_loss(y, pc, labels=[0, 1]),
    )


def top1_predictions(df: pd.DataFrame, score_col="p") -> pd.DataFrame:
    """Best-scoring candidate per observation -> DataFrame(trip_id, obs, pred_arc, true_arc, label_valid, hit_possible)."""
    idx = df.groupby(["trip_id", "obs"])[score_col].idxmax()
    top = df.loc[idx, ["trip_id", "obs", "arc"]].rename(columns={"arc": "pred_arc"})
    if "true_arc" in df:
        g = df.groupby(["trip_id", "obs"]).agg(true_arc=("true_arc", "first"), label_valid=("label_valid", "first"),
                                               in_cands=("y", "max"))
        top = top.merge(g.reset_index(), on=["trip_id", "obs"])
    return top.sort_values(["trip_id", "obs"]).reset_index(drop=True)


def continuity(pred_arcs: np.ndarray, arcs: np.ndarray) -> float:
    a, b = pred_arcs[:-1], pred_arcs[1:]
    if len(a) == 0:
        return 1.0
    ok = (a == b) | (arcs[a, 1] == arcs[b, 0])
    return float(ok.mean())


def route_set_scores(pred_arcs, route):
    P, R = set(map(int, pred_arcs)), set(map(int, route))
    tp = len(P & R)
    prec = tp / max(1, len(P))
    rec = tp / max(1, len(R))
    return prec, rec, 2 * prec * rec / max(1e-9, prec + rec)


def observation_metrics(top: pd.DataFrame) -> dict:
    v = top[top.label_valid]
    return dict(top1_acc=float((v.pred_arc == v.true_arc).mean()),
                candidate_miss_rate=float(1 - v.in_cands.mean()),
                n_obs=int(len(v)))


def sequence_metrics(top: pd.DataFrame, trips: dict) -> dict:
    """trips: trip_id -> Trip (for arcs + route). Averages over trips."""
    cont, f1s, precs, recs = [], [], [], []
    for tid, g in top.groupby("trip_id"):
        tr = trips[tid]
        pa = g.sort_values("obs").pred_arc.values
        cont.append(continuity(pa, tr.arcs))
        if tr.route is not None:
            p, r, f = route_set_scores(pa, tr.route)
            precs.append(p)
            recs.append(r)
            f1s.append(f)
    return dict(route_continuity=float(np.mean(cont)), route_precision=float(np.mean(precs)) if precs else np.nan,
                route_recall=float(np.mean(recs)) if recs else np.nan, route_f1=float(np.mean(f1s)) if f1s else np.nan)
