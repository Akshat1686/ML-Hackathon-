from collections import Counter
from functools import lru_cache
from heapq import heappop, heappush

import numpy as np

from .data import GAP, motion


BRIDGE_LIMIT_M = 160.0
MAX_SPEED_M_S = 45.0


def decode(trip, candidates, features, probability):
    """Maximize emissions and feasible directed transitions in log space."""
    order = np.argsort(trip.arcs[:, 0], kind="stable")
    offsets = np.searchsorted(trip.arcs[order, 0], np.arange(len(trip.nodes) + 1))

    @lru_cache(maxsize=4096)
    def shortest(source):
        distances, predecessor = {source: 0.0}, {}
        queue = [(0.0, source)]
        while queue:
            distance, node = heappop(queue)
            if distance != distances[node]:
                continue
            for arc in order[offsets[node]:offsets[node + 1]]:
                target = int(trip.arcs[arc, 1])
                total = distance + trip.lengths[arc]
                if total <= BRIDGE_LIMIT_M and total < distances.get(target, np.inf):
                    distances[target] = total
                    predecessor[target] = int(arc)
                    heappush(queue, (total, target))
        return distances, predecessor

    emissions = np.log(np.clip(probability, 1e-7, 1))
    emissions -= np.log(np.clip(probability.sum(axis=1, keepdims=True), 1e-7, None))
    fraction = features[:, :, 3]
    _, _, _, seconds = motion(trip)
    parents = np.zeros(candidates.shape, dtype=np.int16)
    best = np.zeros(len(candidates), dtype=np.int16)
    starts = np.zeros(len(candidates), dtype=bool)
    starts[0] = True
    score = emissions[0]
    best[0] = score.argmax()
    for t in range(1, len(candidates)):
        prev, cur = candidates[t - 1], candidates[t]
        transition = np.full((len(prev), len(cur)), -np.inf)
        if seconds[t] <= GAP:
            observed = np.linalg.norm(trip.track[t] - trip.track[t - 1])
            source = trip.arcs[prev, 1]
            target = trip.arcs[cur, 0]
            same = prev[:, None] == cur
            forward = (fraction[t][None, :] - fraction[t - 1][:, None]) * trip.lengths[prev, None]
            bridge = np.array([[shortest(int(node))[0].get(int(other), np.inf) for other in target]
                               for node in source])
            distance = (1 - fraction[t - 1, :, None]) * trip.lengths[prev, None]
            distance = distance + bridge + fraction[t] * trip.lengths[cur]
            distance = np.where(same, abs(forward), distance)
            plausible = (distance <= MAX_SPEED_M_S * seconds[t] + 20) & (~same | (forward >= -25))
            transition[plausible] = -abs(distance[plausible] - observed) / 18
        total = score[:, None] + transition
        parents[t] = total.argmax(axis=0)
        updated = total.max(axis=0) + emissions[t]
        if not np.isfinite(updated).any():
            starts[t] = True
            updated = emissions[t]
        score = updated - updated.max()
        best[t] = score.argmax()

    chosen = np.empty(len(candidates), dtype=np.int32)
    state = best[-1]
    for t in range(len(candidates) - 1, -1, -1):
        chosen[t] = candidates[t, state]
        state = best[t - 1] if starts[t] else parents[t, state]

    segments = []
    for t, arc in enumerate(chosen):
        arc = int(arc)
        if starts[t]:
            segments.append([arc])
        elif arc != chosen[t - 1]:
            source = int(trip.arcs[chosen[t - 1], 1])
            target = int(trip.arcs[arc, 0])
            _, predecessor = shortest(source)
            middle = []
            while target != source:
                connector = predecessor[target]
                middle.append(connector)
                target = int(trip.arcs[connector, 0])
            segments[-1].extend(reversed(middle))
            segments[-1].append(arc)
    return chosen, segments, starts


def direct_links(trip, chosen):
    return (chosen[:-1] == chosen[1:]) | (trip.arcs[chosen[:-1], 1] == trip.arcs[chosen[1:], 0])


def split_baseline(trip, chosen):
    links = direct_links(trip, chosen) & (np.diff(trip.gps[:, 2]) <= GAP)
    segments = np.split(chosen, np.flatnonzero(~links) + 1)
    return [[int(a) for a in part[np.r_[True, part[1:] != part[:-1]]]] for part in segments]


def sequence_metrics(segments, reference):
    predicted_edges = Counter(arc for part in segments for arc in part)
    reference_edges = Counter(reference)
    predicted_pairs = Counter(pair for part in segments for pair in zip(part, part[1:]))
    reference_pairs = Counter(zip(reference, reference[1:]))

    def f1(left, right):
        total = left.total() + right.total()
        return 2 * (left & right).total() / total if total else 1.0

    return {"arc_multiset_f1": f1(predicted_edges, reference_edges),
            "ordered_arc_pair_f1": f1(predicted_pairs, reference_pairs)}
