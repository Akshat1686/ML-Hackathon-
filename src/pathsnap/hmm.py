"""HMM + Viterbi route reconstruction.

Hidden state at observation i : one of its candidate arcs (projection point on that arc).
Emission  log P(z_i | arc)    : w_emit * log( p_clf(arc) / sum_candidates p_clf )
                                 (classifier probability, renormalised within the observation)
Transition log P(arc_j | arc_i): -|d_net - d_gc| / beta
      d_gc  = straight-line distance between the two GPS observations
      d_net = shortest driving distance along the directed network from the projection on
              arc_i to the projection on arc_j (Dijkstra, arc lengths as weights)
      Same arc: d_net = |t_j - t_i| * length (small backwards jitter allowed)
      Unreachable within the search limit -> transition impossible (-inf).
Viterbi: delta_i(s) = emit_i(s) + max_s' [delta_{i-1}(s') + trans(s', s)]
Breaks  : if the time gap > gap_break_s, or every transition is impossible, the chain is cut and
          a new Viterbi segment starts (each segment decoded independently).
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .data_io import Trip

NEG = -1e18


@dataclass
class HMMConfig:
    beta: float = 30.0          # metres; tolerance of route-vs-straight-line distance mismatch
    w_emit: float = 1.0         # weight of classifier evidence
    gap_break_s: float = 60.0   # cut the chain if two observations are further apart in time
    limit_extra_m: float = 300.0  # Dijkstra search radius = d_gc * 2 + limit_extra_m
    min_prob: float = 1e-4

    def to_dict(self):
        return asdict(self)


class NetworkRouter:
    """Shortest-path distances on the trip network, with a per-source cache."""

    def __init__(self, trip: Trip, arc_subset: np.ndarray):
        arcs = trip.arcs[arc_subset]
        self.length = np.linalg.norm(trip.nodes_xy[trip.arcs[:, 1]] - trip.nodes_xy[trip.arcs[:, 0]], axis=1)
        nodes = np.unique(arcs)
        self.local = {int(n): i for i, n in enumerate(nodes)}
        u = np.array([self.local[int(a)] for a in arcs[:, 0]])
        v = np.array([self.local[int(b)] for b in arcs[:, 1]])
        w = np.maximum(self.length[arc_subset], 1e-3)
        n = len(nodes)
        # keep the shortest arc if parallel arcs exist
        order = np.argsort(-w)
        self.G = csr_matrix((w[order], (u[order], v[order])), shape=(n, n))
        self.cache: dict[tuple[int, int], np.ndarray] = {}

    def dist_from(self, node: int, limit: float) -> np.ndarray | None:
        ln = self.local.get(int(node))
        if ln is None:
            return None
        lim_key = int(np.ceil(limit / 250.0))           # bucket limits so the cache is reused
        key = (ln, lim_key)
        if key not in self.cache:
            self.cache[key] = dijkstra(self.G, indices=ln, limit=lim_key * 250.0)
        return self.cache[key]

    def node_index(self, node):
        return self.local.get(int(node))


def viterbi_decode(trip: Trip, cand: pd.DataFrame, cfg: HMMConfig, router: NetworkRouter | None = None):
    """cand: rows for ONE trip with columns obs, arc, t_frac, p (classifier probability).
    Returns DataFrame(obs, pred_arc, segment) and stats dict."""
    t0 = time.time()
    cand = cand.sort_values(["obs", "arc"])
    obs_ids = cand.obs.unique()
    groups = [g for _, g in cand.groupby("obs", sort=True)]
    if router is None:
        router = NetworkRouter(trip, np.unique(np.r_[cand.arc.values, _neighbour_arcs(trip, cand.arc.values)]))
    arcs = trip.arcs
    L = router.length
    P = trip.track_xy
    T = trip.track[:, 2]

    states = [g.arc.values.astype(int) for g in groups]
    tfr = [g.t_frac.values for g in groups]
    em = []
    for g in groups:
        p = np.clip(g.p.values, cfg.min_prob, None)
        em.append(cfg.w_emit * np.log(p / p.sum()))

    n = len(groups)
    delta = [None] * n
    back = [None] * n
    seg_start = np.zeros(n, dtype=bool)
    seg_start[0] = True
    delta[0] = em[0].copy()
    n_breaks_gap = n_breaks_unreach = 0
    for i in range(1, n):
        oi, oj = obs_ids[i - 1], obs_ids[i]
        dt = T[oj] - T[oi]
        d_gc = float(np.linalg.norm(P[oj] - P[oi]))
        if dt > cfg.gap_break_s:
            seg_start[i] = True
            n_breaks_gap += 1
            delta[i] = em[i].copy()
            continue
        A, B = states[i - 1], states[i]
        ta, tb = tfr[i - 1], tfr[i]
        limit = 2 * d_gc + cfg.limit_extra_m
        D = np.full((len(A), len(B)), np.inf)
        for a_k, a in enumerate(A):
            rem_a = (1 - ta[a_k]) * L[a]
            same = B == a
            if same.any():
                D[a_k, same] = np.abs(tb[same] - ta[a_k]) * L[a]
            dist = router.dist_from(arcs[a, 1], limit)
            if dist is None:
                continue
            for b_k, b in enumerate(B):
                if b == a:
                    continue
                nb = router.node_index(arcs[b, 0])
                if nb is None:
                    continue
                dn = dist[nb]
                if np.isfinite(dn):
                    D[a_k, b_k] = min(D[a_k, b_k], rem_a + dn + tb[b_k] * L[b])
        trans = np.where(np.isfinite(D), -np.abs(D - d_gc) / cfg.beta, NEG)
        score = delta[i - 1][:, None] + trans
        best_prev = score.argmax(0)
        best = score[best_prev, np.arange(len(B))]
        if (best <= NEG / 2).all():
            # no feasible transition: start a new segment
            seg_start[i] = True
            n_breaks_unreach += 1
            delta[i] = em[i].copy()
            continue
        delta[i] = best + em[i]
        back[i] = best_prev

    # backtrack segment by segment
    path = np.empty(n, dtype=int)
    seg_id = np.cumsum(seg_start) - 1
    i = n - 1
    while i >= 0:
        k = int(np.argmax(delta[i]))
        while True:
            path[i] = k
            if seg_start[i]:
                break
            k = int(back[i][k])
            i -= 1
        i -= 1
    pred = np.array([states[i][path[i]] for i in range(n)])
    out = pd.DataFrame({"obs": obs_ids, "pred_arc": pred, "segment": seg_id})
    stats = dict(n_obs=n, n_segments=int(seg_start.sum()), breaks_gap=n_breaks_gap,
                 breaks_unreachable=n_breaks_unreach, decode_s=time.time() - t0)
    return out, stats


def _neighbour_arcs(trip: Trip, cand_arcs: np.ndarray, hops: int = 0):
    """Arcs that touch the candidate arcs' nodes and lie near the track (for routing between candidates)."""
    # route search must be able to go through non-candidate arcs: use all arcs near the trajectory
    P = trip.track_xy
    lo, hi = P.min(0) - 500, P.max(0) + 500
    a, b = trip.nodes_xy[trip.arcs[:, 0]], trip.nodes_xy[trip.arcs[:, 1]]
    near = ((np.maximum(a, b) >= lo) & (np.minimum(a, b) <= hi)).all(1)
    return np.nonzero(near)[0]
