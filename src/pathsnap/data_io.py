"""Reading the raw PathSnap trip folders.

Each trip folder <id>/ contains:
  <id>.track  lon lat t          (GPS observations, ~1 Hz)
  <id>.nodes  lon lat            (row index = node_id)
  <id>.arcs   from_node to_node  (row index = arc_id, directed straight segment)
  <id>.route  arc_id             (reference route, chronological; may be absent)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from .geo import LocalProjection


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray            # (N,3) lon, lat, t
    nodes: np.ndarray            # (V,2) lon, lat
    arcs: np.ndarray             # (E,2) int from,to
    route: np.ndarray | None     # (M,) int arc ids, or None
    proj: LocalProjection = field(init=False)
    track_xy: np.ndarray = field(init=False)   # (N,2) metres
    nodes_xy: np.ndarray = field(init=False)   # (V,2) metres

    def __post_init__(self):
        self.proj = LocalProjection.from_lonlat(self.track[:, 0], self.track[:, 1])
        self.track_xy = self.proj.to_xy(self.track[:, 0], self.track[:, 1])
        self.nodes_xy = self.proj.to_xy(self.nodes[:, 0], self.nodes[:, 1])


def list_trips(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        return []
    return sorted(d for d in os.listdir(folder)
                  if os.path.isdir(os.path.join(folder, d)) and
                  os.path.exists(os.path.join(folder, d, f"{d}.track")))


def _load(path, dtype=float, cols=None):
    arr = np.loadtxt(path, dtype=dtype, ndmin=2)
    if cols is not None and arr.size == 0:
        arr = arr.reshape(0, cols)
    return arr


def clean_track(track: np.ndarray) -> np.ndarray:
    """Drop NaN / implausible coordinates, sort by time, drop duplicate timestamps."""
    ok = np.isfinite(track).all(axis=1)
    ok &= (np.abs(track[:, 0]) <= 180) & (np.abs(track[:, 1]) <= 90)
    track = track[ok]
    track = track[np.argsort(track[:, 2], kind="stable")]
    keep = np.r_[True, np.diff(track[:, 2]) > 1e-6]
    return track[keep]


def load_trip(folder: str, trip_id: str, with_route: bool = True) -> Trip:
    base = os.path.join(folder, trip_id, trip_id)
    track = clean_track(_load(base + ".track", cols=3))
    nodes = _load(base + ".nodes", cols=2)
    arcs = _load(base + ".arcs", dtype=int, cols=2)
    route = None
    if with_route and os.path.exists(base + ".route"):
        route = np.loadtxt(base + ".route", dtype=int, ndmin=1)
    return Trip(trip_id, track, nodes, arcs, route)
