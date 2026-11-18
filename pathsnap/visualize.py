import argparse
import csv
import json
from pathlib import Path

import folium
import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from .data import FEATURES, candidate_features, load_trip, reference
from .decode import decode

BLUE, ORANGE, GREEN, GRAY = "#2279b0", "#b85c1b", "#007e73", "#8a98a7"


def make_maps(directory, out):
    trip = load_trip(directory)
    route = reference(trip)
    arcs, features = candidate_features(trip)
    model = joblib.load(out / "model.joblib")
    score = model.predict_proba(features.reshape(-1, len(FEATURES)))[:, 1].reshape(arcs.shape)
    _, parts, _ = decode(trip, arcs, features, score)
    roads = trip.arcs[np.unique(arcs)]
    reference_segments = trip.arcs[route]
    predicted_segments = trip.arcs[np.concatenate(parts)]
    gps = trip.gps[:, :2]

    map_ = folium.Map(tiles=None, control_scale=True)
    map_.default_js = [resource for resource in map_.default_js if resource[0] == "leaflet"]
    map_.default_css = [resource for resource in map_.default_css if resource[0] == "leaflet_css"]
    for name, shape, color, weight in [
        ("Candidate road network", roads, GRAY, 2),
        ("Reference route", reference_segments, ORANGE, 7),
        ("Predicted route", predicted_segments, GREEN, 4),
    ]:
        layer = folium.FeatureGroup(name=name).add_to(map_)
        folium.PolyLine(trip.nodes[shape][:, :, ::-1].tolist(), color=color, weight=weight,
                        opacity=0.75).add_to(layer)
    observations = folium.FeatureGroup(name="GPS observations").add_to(map_)
    for coordinates in gps:
        folium.CircleMarker(coordinates[::-1].tolist(), radius=2, color=BLUE, fill=True,
                            fill_opacity=0.8, weight=1).add_to(observations)
    coords = np.vstack((gps, trip.nodes[reference_segments.ravel()]))
    map_.fit_bounds([coords.min(axis=0)[::-1].tolist(), coords.max(axis=0)[::-1].tolist()])
    folium.LayerControl(collapsed=False).add_to(map_)
    legend = f"""<div style="position:fixed;z-index:900;left:22px;bottom:30px;
        background:white;padding:12px 16px;border:1px solid #cbd5e1;border-radius:8px;
        font:14px system-ui;color:#152537"><b>Student-test trip {directory.name}</b><br>
        <span style="color:{BLUE}">GPS observations</span><br>
        <span style="color:{GRAY}">Candidate road network</span><br>
        <span style="color:{ORANGE}">Reference route</span><br>
        <span style="color:{GREEN}">Predicted route</span></div>"""
    map_.get_root().html.add_child(folium.Element(legend))
    html = map_.get_root().render()
    (out / "folium_map.html").write_text("\n".join(line.rstrip() for line in html.splitlines()) + "\n",
                                          encoding="utf-8", newline="\n")

    fig, axis = plt.subplots(figsize=(8, 4.6), layout="constrained")
    for name, shape, color, width in [
        ("Candidate road network", roads, GRAY, 0.85),
        ("Reference route", reference_segments, ORANGE, 2.8),
        ("Predicted route", predicted_segments, GREEN, 1.7),
    ]:
        axis.add_collection(LineCollection(trip.nodes[shape], colors=color, linewidth=width, label=name))
    axis.scatter(*gps.T, color=BLUE, s=7, label="GPS observations", zorder=3, alpha=0.7)
    axis.autoscale()
    axis.set_aspect(1 / np.cos(np.deg2rad(gps[:, 1].mean())))
    axis.set(xlabel="Longitude (degrees)", ylabel="Latitude (degrees)", title=f"Student-test trip {directory.name}")
    axis.ticklabel_format(useOffset=False)
    axis.legend(loc="best", fontsize=8)
    fig.savefig(out / "route_plot.png", dpi=170)
    plt.close(fig)


def make_charts(out):
    summary = json.loads((out / "results.json").read_text(encoding="utf-8"))
    with (out / "inventory.csv").open(encoding="utf-8") as file:
        inventory = list(csv.DictReader(file))
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3), layout="constrained")
    for part, color in [("train", BLUE), ("student_test", GREEN)]:
        subset = [row for row in inventory if row["partition"] == part]
        label = part.replace("_", " ")
        axes[0].hist([int(row["gps"]) for row in subset], bins=np.geomspace(250, 20000, 13),
                     color=color, alpha=0.6, label=label)
        axes[1].scatter([int(row["arcs"]) for row in subset], [float(row["max_gap_s"]) for row in subset],
                        color=color, alpha=0.6, label=label)
    axes[0].set(xscale="log", xlabel="GPS observations per trip", ylabel="Trips")
    axes[1].set(xscale="log", yscale="log", xlabel="Road arcs per trip", ylabel="Maximum GPS gap (s)")
    axes[1].legend(loc="upper left")
    fig.savefig(out / "eda.png", dpi=170)
    plt.close(fig)

    rows = summary["models"]
    names = [r["model"].replace("HistGradientBoosting", "Hist. Gradient Boosting") for r in rows]
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), layout="constrained", sharey=True)
    axes[0].barh(y, [r["cv_macro_f1"] for r in rows], color=BLUE)
    axes[1].barh(y - .17, [r["macro_f1"] for r in rows], .32, color=BLUE, label="Macro F1")
    axes[1].barh(y + .17, [r["top1"] for r in rows], .32, color=GREEN, label="Top-1")
    axes[2].barh(y, [r["log_loss"] for r in rows], color=ORANGE)
    axes[0].set(yticks=y, yticklabels=names, xlim=(0, 1), xlabel="Grouped CV macro F1")
    axes[0].invert_yaxis()
    axes[1].set(xlim=(0, 1), xlabel="Student-test scores")
    axes[1].legend(loc="lower right", fontsize=8)
    axes[2].set(xlabel="Student-test log loss")
    fig.savefig(out / "model_comparison.png", dpi=170)
    plt.close(fig)

    with (out / "bayesian_trials.csv").open(encoding="utf-8") as file:
        trials = list(csv.DictReader(file))
    fig, axis = plt.subplots(figsize=(7.6, 2.7), layout="constrained")
    values = [float(r["score"]) for r in trials]
    axis.scatter(range(len(values)), values, color=BLUE, label="Trial")
    axis.step(range(len(values)), np.maximum.accumulate(values), where="post", color=GREEN, label="Best so far")
    axis.set(xlabel="Trial", ylabel="Grouped CV macro F1")
    axis.legend()
    fig.savefig(out / "tuning.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trip", type=Path, default=Path("Student folder/student_test/00000052"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    make_charts(args.out)
    make_maps(args.trip, args.out)
