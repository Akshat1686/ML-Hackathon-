"""Geometry helpers.

Lon/lat are angles, so we never measure distance in degrees. Each trip covers at
most a few tens of km, so we use a local equirectangular (tangent-plane)
projection centred on the trip's GPS centroid:
    x = R * cos(lat0) * (lon - lon0)    y = R * (lat - lat0)     (radians, metres)
Within ~50 km of the centre the distance error is far below GPS noise (<0.1%),
and it works anywhere on the globe (the dataset spans many countries/UTM zones).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

R_EARTH = 6_371_008.8


@dataclass
class LocalProjection:
    lon0: float
    lat0: float

    @classmethod
    def from_lonlat(cls, lon, lat):
        return cls(float(np.mean(lon)), float(np.mean(lat)))

    def to_xy(self, lon, lat):
        lon = np.radians(np.asarray(lon, float) - self.lon0)
        lat = np.radians(np.asarray(lat, float) - self.lat0)
        return np.column_stack([R_EARTH * np.cos(np.radians(self.lat0)) * lon, R_EARTH * lat])

    def to_lonlat(self, xy):
        xy = np.asarray(xy, float)
        lon = np.degrees(xy[:, 0] / (R_EARTH * np.cos(np.radians(self.lat0)))) + self.lon0
        lat = np.degrees(xy[:, 1] / R_EARTH) + self.lat0
        return np.column_stack([lon, lat])


def haversine_m(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def point_segment(p, a, b):
    """Distance from points p (K,2) to segments a->b (K,2 each).
    Returns (distance, t) with t in [0,1] the projection fraction along the segment."""
    ab = b - a
    L2 = np.einsum("ij,ij->i", ab, ab)
    t = np.where(L2 > 0, np.einsum("ij,ij->i", p - a, ab) / np.where(L2 > 0, L2, 1), 0.0)
    t = np.clip(t, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.linalg.norm(p - proj, axis=1), t


def bearing_deg(dx, dy):
    """Compass-style bearing (0 = north, clockwise) from planar deltas."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def angle_diff(a, b):
    """Absolute smallest difference between two bearings, 0..180."""
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)
