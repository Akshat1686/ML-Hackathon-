from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

K = 8
GAP = 35.0
FEATURES = (
    "distance_m", "heading_agreement", "heading_known", "position_on_arc",
    "arc_length_m", "gps_speed_m_s", "start_distance_m", "end_distance_m",
)


@dataclass
class Trip:
    directory: Path
    gps: np.ndarray
    nodes: np.ndarray
    arcs: np.ndarray
    xy: np.ndarray
    track: np.ndarray
    starts: np.ndarray
    vectors: np.ndarray
    lengths: np.ndarray


def load_trip(directory):
    directory = Path(directory)
    def read(extension, dtype=float):
        return np.loadtxt(directory / f"{directory.name}.{extension}", dtype=dtype, ndmin=2)

    gps, nodes, arcs = read("track"), read("nodes"), read("arcs", np.int32)
    if gps.shape[1] != 3 or nodes.shape[1] != 2 or arcs.shape[1] != 2:
        raise ValueError(f"Unexpected file shape in {directory.name}")
    if not np.isfinite(gps).all() or not np.isfinite(nodes).all():
        raise ValueError(f"Non-finite coordinates or timestamps in {directory.name}")
    if np.any(arcs < 0) or np.any(arcs >= len(nodes)):
        raise ValueError(f"Invalid road node ID in {directory.name}")
    if np.any(np.diff(gps[:, 2]) <= 0):
        raise ValueError(f"Non-increasing timestamps in {directory.name}")
    center = np.median(gps[:, :2], axis=0)
    scale = np.array([np.cos(np.deg2rad(center[1])), 1.0]) * (6371008.8 * np.pi / 180)
    xy = (nodes - center) * scale
    track = (gps[:, :2] - center) * scale
    starts = xy[arcs[:, 0]]
    vectors = xy[arcs[:, 1]] - starts
    return Trip(directory, gps, nodes, arcs, xy, track, starts, vectors, np.linalg.norm(vectors, axis=1))


def reference(trip):
    route = np.loadtxt(trip.directory / f"{trip.directory.name}.route", dtype=np.int32, ndmin=1)
    if np.any(route < 0) or np.any(route >= len(trip.arcs)):
        raise ValueError(f"Invalid reference arc ID in {trip.directory.name}")
    return route


def projection(trip, points, arc_ids):
    base = trip.starts[arc_ids]
    vector = trip.vectors[arc_ids]
    length = trip.lengths[arc_ids]
    delta = points - base
    fraction = np.clip(np.sum(delta * vector, axis=-1) / np.maximum(length**2, 1e-9), 0, 1)
    return np.linalg.norm(delta - fraction[..., None] * vector, axis=-1), fraction


def motion(trip):
    steps = np.diff(trip.track, axis=0, prepend=trip.track[:1])
    seconds = np.diff(trip.gps[:, 2], prepend=trip.gps[0, 2])
    steps[seconds > GAP] = 0
    if len(steps) > 1:
        steps[0] = steps[1]
    velocity = np.linalg.norm(steps, axis=1)
    speed = np.divide(velocity, seconds, out=np.zeros_like(velocity), where=seconds > 0)
    heading = steps / np.maximum(velocity[:, None], 1e-9)
    return heading, velocity >= 2, speed, seconds


class SegmentIndex:
    """Index samples no more than 20 m apart, then rank by exact segment distance."""

    def __init__(self, trip, arc_ids):
        self.trip = trip
        self.arc_ids = np.asarray(arc_ids)
        counts = np.maximum(1, np.ceil(trip.lengths[self.arc_ids] / 20).astype(int)) + 1
        self.rows = np.repeat(np.arange(len(arc_ids)), counts)
        offset = np.arange(len(self.rows)) - np.repeat(np.cumsum(counts) - counts, counts)
        fraction = offset / (counts[self.rows] - 1)
        sampled_arcs = self.arc_ids[self.rows]
        self.tree = cKDTree(trip.starts[sampled_arcs] + fraction[:, None] * trip.vectors[sampled_arcs])

    def nearby(self, points, count):
        count = min(count, len(self.arc_ids))
        _, nearest = self.tree.query(points, k=min(max(4 * count, 2), self.tree.n))
        candidates = np.empty((len(points), count), dtype=np.int32)
        for index, point in enumerate(points):
            rows = np.unique(self.rows[nearest[index]])
            distances, _ = projection(self.trip, point, self.arc_ids[rows])
            if len(rows) >= count:
                upper_bound = np.partition(distances, count - 1)[count - 1]
                rows = np.unique(self.rows[self.tree.query_ball_point(point, upper_bound + 10.001)])
            else:
                # Long segments can dominate the initial sample neighborhood.
                radius = max(20.0, float(distances.max()) + 10.001)
                while True:
                    rows = np.unique(self.rows[self.tree.query_ball_point(point, radius)])
                    if len(rows) >= count:
                        distances, _ = projection(self.trip, point, self.arc_ids[rows])
                        if np.partition(distances, count - 1)[count - 1] <= radius - 10:
                            break
                    radius *= 2
            distances, _ = projection(self.trip, point, self.arc_ids[rows])
            candidates[index] = rows[np.argsort(distances, kind="stable")[:count]]
        return candidates


def candidate_features(trip):
    index = SegmentIndex(trip, np.arange(len(trip.arcs), dtype=np.int32))
    arcs = index.nearby(trip.track, K)
    distance, position = projection(trip, trip.track[:, None, :], arcs)
    start = np.linalg.norm(trip.track[:, None] - trip.starts[arcs], axis=2)
    end = np.linalg.norm(trip.track[:, None] - trip.starts[arcs] - trip.vectors[arcs], axis=2)
    heading, known, speed, _ = motion(trip)
    agreement = (heading[:, None] * trip.vectors[arcs]).sum(axis=2) / np.maximum(trip.lengths[arcs], 1e-9)
    broadcast = np.broadcast_to
    features = np.stack([
        distance, agreement * known[:, None], broadcast(known[:, None], arcs.shape),
        position, trip.lengths[arcs], broadcast(speed[:, None], arcs.shape), start, end,
    ], axis=2)
    return arcs, features.astype(np.float32)


def align_reference(trip, route):
    # Preserve route occurrence indexes so repeated road IDs can be disambiguated.
    index = SegmentIndex(trip, route)
    occurrences = np.sort(index.nearby(trip.track, 12), axis=1)
    arcs = route[occurrences]
    distance, fraction = projection(trip, trip.track[:, None, :], arcs)
    progress = np.r_[0.0, np.cumsum(trip.lengths[route])][occurrences] + fraction * trip.lengths[arcs]
    heading, known, _, seconds = motion(trip)
    direction = (heading[:, None] * trip.vectors[arcs]).sum(axis=2) / np.maximum(trip.lengths[arcs], 1e-9)
    costs = distance / 10 + (1 - direction) * known[:, None] * 0.85
    back = np.zeros_like(occurrences)
    result = np.empty(len(arcs), dtype=np.int32)
    cuts = np.r_[0, np.flatnonzero(seconds > GAP), len(arcs)]
    for first, stop in zip(cuts[:-1], cuts[1:]):
        score = costs[first]
        for t in range(first + 1, stop):
            movement = np.linalg.norm(trip.track[t] - trip.track[t - 1])
            jump = progress[t, None, :] - progress[t - 1, :, None]
            penalty = np.maximum(-jump - 6, 0) / 6
            penalty += np.maximum(jump - max(20 + movement, 35 * seconds[t]), 0) / 15
            penalty += np.maximum(occurrences[t - 1, :, None] - occurrences[t, None, :], 0) * 0.4
            choices = score[:, None] + penalty
            back[t] = choices.argmin(axis=0)
            score = choices.min(axis=0) + costs[t]
        choice = int(score.argmin())
        for t in range(stop - 1, first - 1, -1):
            result[t] = arcs[t, choice] if distance[t, choice] <= 55 else -1
            choice = back[t, choice]
    return result


def inventory(trip, route):
    graph = csr_matrix((np.ones(len(trip.arcs)), trip.arcs.T), shape=(len(trip.nodes), len(trip.nodes)))
    steps = np.diff(trip.gps[:, 2])
    directed = trip.arcs[route]
    return {
        "gps": len(trip.gps), "nodes": len(trip.nodes), "arcs": len(trip.arcs),
        "reference_arcs": len(route), "missing_values": 0,
        "one_second_steps": int((steps == 1).sum()), "max_gap_s": float(steps.max(initial=0)),
        "gaps_over_35s": int((steps > GAP).sum()), "zero_length_arcs": int((trip.lengths == 0).sum()),
        "weak_components": connected_components(graph, directed=False, return_labels=False),
        "isolated_nodes": len(trip.nodes) - len(np.unique(trip.arcs)),
        "reference_breaks": int((directed[:-1, 1] != directed[1:, 0]).sum()),
        "lon_min": float(trip.gps[:, 0].min()), "lon_max": float(trip.gps[:, 0].max()),
        "lat_min": float(trip.gps[:, 1].min()), "lat_max": float(trip.gps[:, 1].max()),
        "file_bytes": sum(path.stat().st_size for path in trip.directory.iterdir() if path.is_file()),
    }
