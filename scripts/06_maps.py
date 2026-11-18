"""Step 6 - Geographical visualisation.

For selected student-test trips, builds an interactive Folium map with layers:
  raw GPS observations | nearby road network | reference route | classifier-only route | HMM route
(+ a static PNG of the same layers for the PDF report).
Usage: python scripts/06_maps.py [trip_id ...]      (default: 3 student-test trips)
Outputs: results/maps/trip_<id>.html, results/figures/map_<id>.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap.candidates import DEFAULT_RADIUS, build_candidate_table  # noqa: E402
from pathsnap.config import DATA_DIR, FIG_DIR, MAP_DIR, MODEL_DIR  # noqa: E402
from pathsnap.data_io import list_trips, load_trip  # noqa: E402
from pathsnap.geo import point_segment  # noqa: E402
from pathsnap.pipeline import PathSnapModel  # noqa: E402

os.makedirs(MAP_DIR, exist_ok=True)
model = PathSnapModel(MODEL_DIR)
ids = sys.argv[1:] or ["00000056", "00000009", "00000077"]
folder = os.path.join(DATA_DIR, "student_test")


def nearby_arcs(trip, max_m=120.0):
    """Arcs within max_m of any GPS point (keeps the HTML small)."""
    c = build_candidate_table(trip, radius=max_m, step=5)
    return np.unique(c.arc.values)


def segs(trip, arc_ids):
    a, b = trip.nodes[trip.arcs[arc_ids, 0]], trip.nodes[trip.arcs[arc_ids, 1]]
    return [[[a[k, 1], a[k, 0]], [b[k, 1], b[k, 0]]] for k in range(len(arc_ids))]


for tid in ids:
    trip = load_trip(folder, tid)
    pred = model.predict(load_trip(folder, tid, with_route=False))
    net = nearby_arcs(trip)
    hmm_arcs, clf_arcs = pd.unique(pred["hmm_arc"]), pd.unique(pred["clf_arc"])
    gps = trip.track[:, :2]

    # ---------- static PNG for the report ----------
    fig, ax = plt.subplots(figsize=(9, 9))
    for s in segs(trip, net):
        ax.plot([s[0][1], s[1][1]], [s[0][0], s[1][0]], color="#cccccc", lw=.6)
    for i, s in enumerate(segs(trip, trip.route)):
        ax.plot([s[0][1], s[1][1]], [s[0][0], s[1][0]], color="green", lw=5, alpha=.5, label="reference route" if i == 0 else None)
    for i, s in enumerate(segs(trip, clf_arcs)):
        ax.plot([s[0][1], s[1][1]], [s[0][0], s[1][0]], color="orange", lw=1.5, ls="--", label="classifier only" if i == 0 else None)
    for i, s in enumerate(segs(trip, hmm_arcs)):
        ax.plot([s[0][1], s[1][1]], [s[0][0], s[1][0]], color="red", lw=2, label="classifier + HMM" if i == 0 else None)
    ax.scatter(gps[:, 0], gps[:, 1], s=2, color="blue", label="raw GPS", zorder=5)
    ax.set_aspect(1 / np.cos(np.radians(gps[:, 1].mean())))
    ax.set_title(f"Student-test trip {tid}")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"map_{tid}.png"), dpi=110)
    plt.close()

    # ---------- interactive Folium map ----------
    try:
        import folium
    except ImportError:
        print("folium not installed - skipped HTML for", tid)
        continue
    center = [float(gps[:, 1].mean()), float(gps[:, 0].mean())]
    m = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")
    layers = [
        ("Road network (nearby arcs)", net, dict(color="#888888", weight=1, opacity=.6)),
        ("Reference route", trip.route, dict(color="#2ca02c", weight=8, opacity=.5)),
        ("Classifier-only route", clf_arcs, dict(color="#ff7f0e", weight=3, opacity=.8, dash_array="6")),
        ("Predicted route (classifier + HMM)", hmm_arcs, dict(color="#d62728", weight=4, opacity=.9)),
    ]
    for name, arcs_, style in layers:
        fg = folium.FeatureGroup(name, show=name != "Classifier-only route")
        for s in segs(trip, arcs_):
            folium.PolyLine(s, **style).add_to(fg)
        fg.add_to(m)
    fg = folium.FeatureGroup("Raw GPS observations")
    for lon, lat in gps[::2]:
        folium.CircleMarker([lat, lon], radius=2, color="#1f77b4", fill=True, fill_opacity=.8).add_to(fg)
    fg.add_to(m)
    folium.Marker([gps[0, 1], gps[0, 0]], tooltip="start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker([gps[-1, 1], gps[-1, 0]], tooltip="end", icon=folium.Icon(color="black")).add_to(m)
    m.fit_bounds([[gps[:, 1].min(), gps[:, 0].min()], [gps[:, 1].max(), gps[:, 0].max()]])
    folium.LayerControl(collapsed=False).add_to(m)
    legend = ("<div style='position:fixed;bottom:25px;left:25px;z-index:9999;background:white;padding:8px;"
              "border:1px solid #999;font-size:13px'><b>Trip " + tid + "</b><br>"
              "<span style='color:#1f77b4'>●</span> raw GPS<br>"
              "<span style='color:#888'>━</span> road network<br>"
              "<span style='color:#2ca02c'>━</span> reference route<br>"
              "<span style='color:#ff7f0e'>┅</span> classifier only<br>"
              "<span style='color:#d62728'>━</span> classifier + HMM</div>")
    m.get_root().html.add_child(folium.Element(legend))
    out = os.path.join(MAP_DIR, f"trip_{tid}.html")
    m.save(out)
    print("saved", out)
