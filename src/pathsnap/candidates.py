"""Candidate generation, observation labels and feature engineering.

Pipeline for one trip
---------------------
1. RoadIndex: keep only arcs near the trajectory, densify every arc into sample
   points (~DENSIFY_M apart) and put them in a KD-tree for fast radius search.
2. Candidates: for each GPS point, all arcs whose true perpendicular distance is
   <= radius (fallback: the MIN_CANDS nearest arcs within FALLBACK_M).
3. Labels (training / evaluation only): the reference route is a list of arcs with
   NO timestamps, so we derive a per-observation label by a monotone alignment
   (dynamic programming) of GPS points to route positions that minimises the total
   point-to-arc distance. Label of point i = route arc aligned to it.
4. Features: geometric + kinematic + temporal-context features per (point, arc).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .data_io import Trip
from .geo import angle_diff, bearing_deg, point_segment

DENSIFY_M = 10.0
DEFAULT_RADIUS = 50.0
MIN_CANDS = 3
FALLBACK_M = 300.0
CTX_OFFSETS = (-6, -3, 3, 6)     # neighbouring observations used for context features

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio",
    "heading_diff", "heading_cos", "heading_valid", "speed",
    "arc_len", "t_frac", "endpoint_dist", "n_cands",
    "ctx_dist_mean", "ctx_dist_max", "along_move", "along_move_norm",
    "deg_out_to", "deg_in_from", "has_reverse", "time_gap",
]


class RoadIndex:
    def __init__(self, trip: Trip, margin_m: float = FALLBACK_M + 200):
        nxy = trip.nodes_xy
        a, b = nxy[trip.arcs[:, 0]], nxy[trip.arcs[:, 1]]
        lo = trip.track_xy.min(0) - margin_m
        hi = trip.track_xy.max(0) + margin_m
        seg_lo, seg_hi = np.minimum(a, b), np.maximum(a, b)
        near = (seg_hi >= lo).all(1) & (seg_lo <= hi).all(1)
        self.arc_ids = np.nonzero(near)[0]
        self.A, self.B = a, b                      # full arrays indexed by arc id
        self.length = np.linalg.norm(b - a, axis=1)
        self.bearing = bearing_deg(b[:, 0] - a[:, 0], b[:, 1] - a[:, 1])
        # densify
        pts, owner = [], []
        for e in self.arc_ids:
            n = max(2, int(np.ceil(self.length[e] / DENSIFY_M)) + 1)
            s = np.linspace(0, 1, n)[:, None]
            pts.append(a[e] + s * (b[e] - a[e]))
            owner.append(np.full(n, e))
        self.pts = np.vstack(pts) if pts else np.zeros((0, 2))
        self.owner = np.concatenate(owner) if owner else np.zeros(0, int)
        self.tree = cKDTree(self.pts) if len(self.pts) else None
        # degrees and reverse arcs
        V = len(trip.nodes)
        self.deg_out = np.bincount(trip.arcs[:, 0], minlength=V)
        self.deg_in = np.bincount(trip.arcs[:, 1], minlength=V)
        key = trip.arcs[:, 0].astype(np.int64) * (V + 1) + trip.arcs[:, 1]
        rkey = trip.arcs[:, 1].astype(np.int64) * (V + 1) + trip.arcs[:, 0]
        self.has_reverse = np.isin(rkey, key).astype(np.int8)
        self.arcs = trip.arcs

    def candidates(self, p, radius):
        """Return arc ids within `radius` of point p (with fallback)."""
        if self.tree is None:
            return np.zeros(0, int)
        idx = self.tree.query_ball_point(p, radius + DENSIFY_M / 2 + 1)
        arcs = np.unique(self.owner[idx]) if len(idx) else np.zeros(0, int)
        if len(arcs):
            d, _ = point_segment(np.repeat(p[None], len(arcs), 0), self.A[arcs], self.B[arcs])
            arcs = arcs[d <= radius]
        if len(arcs) < MIN_CANDS:
            k = min(len(self.pts), 200)
            dd, ii = self.tree.query(p, k=k, distance_upper_bound=FALLBACK_M)
            ii = ii[np.isfinite(dd)]
            more = pd.unique(self.owner[ii])
            arcs = np.unique(np.r_[arcs, more[:MIN_CANDS]]).astype(int)
        return arcs


LABEL_MAX_M = 100.0   # points farther than this from their aligned route arc get no reliable label


def align_route_labels(trip: Trip, cap_m: float = 200.0, return_valid: bool = False):
    """Monotone DP alignment of GPS points to the reference route.

    D[i, j] = c(i, j) + min_{k <= j} D[i-1, k]  (route position never goes backwards)
    where c(i, j) = distance of point i to route arc j (capped). Returns arc id per point.
    """
    route = trip.route
    P = trip.track_xy
    A = trip.nodes_xy[trip.arcs[route, 0]]
    B = trip.nodes_xy[trip.arcs[route, 1]]
    N, M = len(P), len(route)
    C = np.empty((N, M))
    for j in range(M):
        d, _ = point_segment(P, np.repeat(A[j][None], N, 0), np.repeat(B[j][None], N, 0))
        C[:, j] = np.minimum(d, cap_m)
    D = np.empty((N, M))
    back = np.empty((N, M), dtype=np.int32)
    D[0] = C[0]
    for i in range(1, N):
        prev = D[i - 1]
        pm = np.minimum.accumulate(prev)
        # argmin of each prefix: last position where the running minimum was set
        arg = np.maximum.accumulate(np.where(prev <= pm, np.arange(M), 0)).astype(np.int32)
        D[i] = C[i] + pm
        back[i] = arg
    j = int(np.argmin(D[-1]))
    pos = np.empty(N, dtype=np.int32)
    for i in range(N - 1, -1, -1):
        pos[i] = j
        if i:
            j = back[i, j]
    lab = route[pos]
    if not return_valid:
        return lab
    d = C[np.arange(N), pos]
    return lab, d < LABEL_MAX_M


def _kinematics(trip: Trip):
    P, t = trip.track_xy, trip.track[:, 2]
    N = len(P)
    i0 = np.clip(np.arange(N) - 2, 0, N - 1)
    i1 = np.clip(np.arange(N) + 2, 0, N - 1)
    dxy = P[i1] - P[i0]
    dt = np.maximum(t[i1] - t[i0], 1e-6)
    dist = np.linalg.norm(dxy, axis=1)
    speed = dist / dt
    heading = bearing_deg(dxy[:, 0], dxy[:, 1])
    valid = (dist > 3.0).astype(np.int8)          # heading meaningless when (almost) stopped
    gap = np.r_[0.0, np.diff(t)]
    return speed, heading, valid, gap


def build_candidate_table(trip: Trip, radius: float = DEFAULT_RADIUS, labels: np.ndarray | None = None,
                          step: int = 1, label_valid: np.ndarray | None = None) -> pd.DataFrame:
    """One row per (GPS observation, candidate arc) with features and optional label."""
    idx = RoadIndex(trip)
    P = trip.track_xy
    N = len(P)
    speed, heading, hvalid, gap = _kinematics(trip)
    obs_ids = np.arange(0, N, step)
    rows_obs, rows_arc = [], []
    for i in obs_ids:
        c = idx.candidates(P[i], radius)
        rows_obs.append(np.full(len(c), i))
        rows_arc.append(c)
    obs = np.concatenate(rows_obs) if rows_obs else np.zeros(0, int)
    arc = np.concatenate(rows_arc).astype(int) if rows_arc else np.zeros(0, int)
    A, B = idx.A[arc], idx.B[arc]
    dist, tf = point_segment(P[obs], A, B)
    L = idx.length[arc]
    df = pd.DataFrame({"trip_id": trip.trip_id, "obs": obs, "arc": arc, "dist": dist, "t_frac": tf, "arc_len": L})
    g = df.groupby("obs")["dist"]
    df["dist_rank"] = g.rank(method="first") - 1
    mn = g.transform("min")
    df["dist_minus_min"] = df["dist"] - mn
    df["dist_ratio"] = df["dist"] / (mn + 1.0)
    df["n_cands"] = g.transform("size")
    hd = angle_diff(heading[obs], idx.bearing[arc])
    hv = hvalid[obs]
    df["heading_diff"] = np.where(hv == 1, hd, 90.0)
    df["heading_cos"] = np.where(hv == 1, np.cos(np.radians(hd)), 0.0)
    df["heading_valid"] = hv
    df["speed"] = speed[obs]
    df["endpoint_dist"] = np.minimum(tf, 1 - tf) * L
    # temporal context: how close are neighbouring observations to this arc?
    ctx = []
    for o in CTX_OFFSETS:
        j = np.clip(obs + o, 0, N - 1)
        d, _ = point_segment(P[j], A, B)
        ctx.append(d)
    ctx = np.column_stack(ctx)
    df["ctx_dist_mean"] = ctx.mean(1)
    df["ctx_dist_max"] = ctx.max(1)
    # movement along the arc direction between obs-3 and obs+3 (unclipped projection)
    ab = B - A
    L2 = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-9)
    jm, jp = np.clip(obs - 3, 0, N - 1), np.clip(obs + 3, 0, N - 1)
    s_m = np.einsum("ij,ij->i", P[jm] - A, ab) / np.sqrt(L2)
    s_p = np.einsum("ij,ij->i", P[jp] - A, ab) / np.sqrt(L2)
    mv = s_p - s_m
    df["along_move"] = mv
    df["along_move_norm"] = mv / (np.linalg.norm(P[jp] - P[jm], axis=1) + 1.0)
    df["deg_out_to"] = idx.deg_out[idx.arcs[arc, 1]]
    df["deg_in_from"] = idx.deg_in[idx.arcs[arc, 0]]
    df["has_reverse"] = idx.has_reverse[arc]
    df["time_gap"] = gap[obs]
    if labels is not None:
        df["true_arc"] = labels[obs]
        df["y"] = (df["arc"].values == labels[obs]).astype(np.int8)
        df["label_valid"] = (label_valid[obs] if label_valid is not None else np.ones(len(obs), bool))
    return df
