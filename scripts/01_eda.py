"""Step 1 - Dataset exploration (train + student_test partitions only).

Outputs (results/eda/ and results/figures/):
  inventory.csv            per-trip counts, sizes, time gaps, extent, network stats
  data_quality.md          short record of issues / assumptions
  fig_*.png                distributions, trajectory plots, network plot
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap.config import DATA_DIR, FIG_DIR, RESULTS_DIR  # noqa: E402
from pathsnap.data_io import list_trips, load_trip  # noqa: E402
from pathsnap.candidates import align_route_labels  # noqa: E402
from pathsnap.geo import point_segment  # noqa: E402

OUT = os.path.join(RESULTS_DIR, "eda")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

rows = []
for part in ["train", "student_test"]:
    folder = os.path.join(DATA_DIR, part)
    for tid in list_trips(folder):
        base = os.path.join(folder, tid, tid)
        raw = np.loadtxt(base + ".track", ndmin=2)
        n_raw = len(raw)
        n_nan = int((~np.isfinite(raw)).any(axis=1).sum())
        tr = load_trip(folder, tid)
        dt = np.diff(tr.track[:, 2])
        step = np.linalg.norm(np.diff(tr.track_xy, axis=0), axis=1)
        G = nx.DiGraph()
        G.add_nodes_from(range(len(tr.nodes)))
        G.add_edges_from(map(tuple, tr.arcs))
        wcc = nx.number_weakly_connected_components(G)
        isolated = sum(1 for v in G.nodes if G.degree(v) == 0)
        route_ok = sum(tr.arcs[tr.route[i], 1] == tr.arcs[tr.route[i + 1], 0] for i in range(len(tr.route) - 1))
        lab, valid = align_route_labels(tr, return_valid=True)
        d, _ = point_segment(tr.track_xy, tr.nodes_xy[tr.arcs[lab, 0]], tr.nodes_xy[tr.arcs[lab, 1]])
        route_len = np.linalg.norm(tr.nodes_xy[tr.arcs[tr.route, 1]] - tr.nodes_xy[tr.arcs[tr.route, 0]], axis=1).sum()
        rows.append(dict(
            partition=part, trip_id=tid, n_obs_raw=n_raw, n_obs_clean=len(tr.track), n_nan_rows=n_nan,
            n_nodes=len(tr.nodes), n_arcs=len(tr.arcs), n_route_arcs=len(tr.route),
            route_connected_frac=route_ok / max(1, len(tr.route) - 1),
            duration_s=tr.track[-1, 2] - tr.track[0, 2],
            dt_median=np.median(dt), frac_dt_1s=np.mean(np.isclose(dt, 1.0)), dt_max=dt.max(), n_gaps_gt10s=int((dt > 10).sum()),
            track_len_km=step.sum() / 1000, route_len_km=route_len / 1000,
            max_step_m=step.max(), speed_p95=np.percentile(step / np.maximum(dt, 1e-6), 95),
            lon_min=tr.track[:, 0].min(), lon_max=tr.track[:, 0].max(), lat_min=tr.track[:, 1].min(), lat_max=tr.track[:, 1].max(),
            extent_km=np.ptp(tr.track_xy, axis=0).max() / 1000,
            weak_components=wcc, isolated_nodes=isolated,
            gps_to_route_median_m=np.median(d), gps_to_route_p95_m=np.percentile(d, 95),
            frac_label_valid=valid.mean(),
            size_kb=sum(os.path.getsize(base + e) for e in [".track", ".nodes", ".arcs", ".route"]) / 1024,
        ))
        print(part, tid, len(tr.track))

inv = pd.DataFrame(rows)
inv.to_csv(os.path.join(OUT, "inventory.csv"), index=False)
summary = inv.groupby("partition").agg(
    trips=("trip_id", "count"), obs_total=("n_obs_clean", "sum"), obs_median=("n_obs_clean", "median"),
    nodes_median=("n_nodes", "median"), arcs_median=("n_arcs", "median"), nan_rows=("n_nan_rows", "sum"),
    gaps_gt10s=("n_gaps_gt10s", "sum"), frac_dt_1s=("frac_dt_1s", "mean"),
    gps_to_route_median_m=("gps_to_route_median_m", "median"), frac_label_valid=("frac_label_valid", "mean"))
summary.to_csv(os.path.join(OUT, "summary_by_partition.csv"))
print(summary.T)

# ---- figures ----
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
for p, c in [("train", "tab:blue"), ("student_test", "tab:orange")]:
    s = inv[inv.partition == p]
    ax[0, 0].hist(s.n_obs_clean, bins=20, alpha=.6, label=p, color=c)
    ax[0, 1].hist(s.track_len_km, bins=20, alpha=.6, label=p, color=c)
    ax[0, 2].hist(np.log10(s.n_arcs), bins=20, alpha=.6, label=p, color=c)
    ax[1, 0].hist(s.dt_max, bins=20, alpha=.6, label=p, color=c)
    ax[1, 1].hist(s.gps_to_route_median_m, bins=20, alpha=.6, label=p, color=c)
    ax[1, 2].scatter(s.lon_min, s.lat_min, label=p, color=c)
titles = ["GPS observations per trip", "Trajectory length (km)", "log10(#arcs) in trip network",
          "Max time gap per trip (s)", "Median GPS-to-route distance (m)", "Trip location (lon, lat)"]
for a, t in zip(ax.ravel(), titles):
    a.set_title(t)
    a.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "eda_distributions.png"), dpi=120)
plt.close()

# sampling interval distribution (train)
dts = []
for tid in inv[inv.partition == "train"].trip_id:
    tr = load_trip(os.path.join(DATA_DIR, "train"), tid, with_route=False)
    dts.append(np.diff(tr.track[:, 2]))
dts = np.concatenate(dts)
plt.figure(figsize=(6, 4))
plt.hist(np.clip(dts, 0, 20), bins=40)
plt.yscale("log")
plt.title("Sampling interval (train, clipped at 20 s)")
plt.xlabel("seconds")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "eda_sampling_interval.png"), dpi=120)
plt.close()

# trajectory + network plots for 4 example trips
ex = inv[inv.partition == "train"].trip_id.tolist()[:4]
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
for a, tid in zip(axs, ex):
    tr = load_trip(os.path.join(DATA_DIR, "train"), tid)
    lo, hi = tr.track_xy.min(0) - 200, tr.track_xy.max(0) + 200
    A, B = tr.nodes_xy[tr.arcs[:, 0]], tr.nodes_xy[tr.arcs[:, 1]]
    m = ((np.minimum(A, B) <= hi) & (np.maximum(A, B) >= lo)).all(1)
    for e in np.nonzero(m)[0]:
        a.plot([A[e, 0], B[e, 0]], [A[e, 1], B[e, 1]], color="lightgray", lw=.5)
    R = tr.route
    for e in R:
        a.plot([A[e, 0], B[e, 0]], [A[e, 1], B[e, 1]], color="green", lw=2)
    a.scatter(tr.track_xy[:, 0], tr.track_xy[:, 1], s=2, color="blue")
    a.set_title(f"train {tid}: GPS (blue), route (green)")
    a.set_aspect("equal")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "eda_trajectories.png"), dpi=110)
plt.close()

with open(os.path.join(OUT, "data_quality.md"), "w") as f:
    f.write("# Data quality notes (generated by scripts/01_eda.py)\n\n")
    f.write(f"- Trips: train={sum(inv.partition=='train')}, student_test={sum(inv.partition=='student_test')}\n")
    f.write(f"- Rows with NaN/invalid coords: {int(inv.n_nan_rows.sum())}\n")
    f.write(f"- Mean fraction of 1 s sampling steps: {inv.frac_dt_1s.mean():.3f}; gaps >10 s: {int(inv.n_gaps_gt10s.sum())}; largest gap {inv.dt_max.max():.0f} s\n")
    f.write(f"- Trips are spread worldwide (lon {inv.lon_min.min():.1f}..{inv.lon_max.max():.1f}) -> local tangent-plane projection per trip\n")
    f.write(f"- Network size per trip: {inv.n_arcs.min()}..{inv.n_arcs.max()} arcs; most arcs have a reverse twin (two-way roads)\n")
    f.write(f"- Reference route: consecutive arcs connected in {inv.route_connected_frac.mean()*100:.1f}% of steps on average\n")
    f.write(f"- Route has no timestamps -> per-observation labels derived by monotone DP alignment; "
            f"{(1-inv.frac_label_valid.mean())*100:.1f}% of points are >100 m from the aligned route and are marked label-unreliable\n")
    worst = inv.sort_values("frac_label_valid").head(5)[["partition", "trip_id", "frac_label_valid", "track_len_km", "route_len_km"]]
    f.write("\nTrips with the most label-unreliable points (route shorter than / not covering the track):\n\n")
    f.write(worst.to_markdown(index=False) if hasattr(worst, "to_markdown") else worst.to_string(index=False))
    f.write("\n")
print("EDA done")
