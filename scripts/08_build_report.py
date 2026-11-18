"""Step 8 - Build the technical report PDF from the recorded results and figures.
Every number in the PDF is read from results/*.csv|json produced by steps 1-7.
Output: reports/PathSnap_Technical_Report.pdf  (+ results/train_fit_scores.csv)
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap.candidates import FEATURES  # noqa: E402
from pathsnap.config import FEAT_DIR, FIG_DIR, MODEL_DIR, RESULTS_DIR, ROOT  # noqa: E402
from pathsnap.evaluation import observation_metrics, row_metrics, top1_predictions  # noqa: E402

R = lambda f: os.path.join(RESULTS_DIR, f)  # noqa: E731
cmp_ = pd.read_csv(R("model_comparison.csv"))
folds = pd.read_csv(R("cv_folds.csv"))
tvu = pd.read_csv(R("tuned_vs_untuned.csv"))
bo = pd.read_csv(R("bayes_opt_trials.csv"))
bp = json.load(open(R("best_params.json")))
hmmc = pd.read_csv(R("hmm_comparison.csv"))
hgrid = pd.read_csv(R("hmm_grid.csv"))
hpt = pd.read_csv(R("hmm_per_trip.csv"))
inv = pd.read_csv(R("eda/inventory.csv"))
cst = pd.read_csv(R("candidate_stats.csv"))
icfg = json.load(open(os.path.join(MODEL_DIR, "inference_config.json")))

# ---- training-set fit (for over/under-fitting evidence): 15 training trips, all saved models ----
tf_path = R("train_fit_scores.csv")
if not os.path.exists(tf_path):
    tr = pd.read_pickle(os.path.join(FEAT_DIR, "train.pkl"))
    tr = tr[tr.label_valid]
    keep = sorted(tr.trip_id.unique())[::5][:16]
    tr = tr[tr.trip_id.isin(keep)]
    rows = []
    for name in cmp_.model:
        m = joblib.load(os.path.join(MODEL_DIR, f"{name}.joblib"))
        p = m.predict_proba(tr[FEATURES].values)[:, 1]
        d = tr[["trip_id", "obs", "arc", "y", "true_arc", "label_valid"]].copy()
        d["p"] = p
        r = row_metrics(tr.y.values, p)
        r.update(observation_metrics(top1_predictions(d)))
        rows.append(dict(model=name, train_macro_f1=r["macro_f1"], train_top1=r["top1_acc"], train_log_loss=r["log_loss"]))
    pd.DataFrame(rows).to_csv(tf_path, index=False)
tfit = pd.read_csv(tf_path)
cmp_ = cmp_.merge(tfit, on="model")

ss = getSampleStyleSheet()
H1, H2, H3 = ss["Heading1"], ss["Heading2"], ss["Heading3"]
B = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=13, alignment=TA_JUSTIFY)
SM = ParagraphStyle("s", parent=B, fontSize=7.5, leading=9.5)
story = []
P = lambda t, s=B: story.append(Paragraph(t, s))  # noqa: E731


def table(df, widths=None, fs=7.5):
    data = [list(df.columns)] + df.astype(str).values.tolist()
    data = [[Paragraph(str(c), ParagraphStyle("t", fontSize=fs, leading=fs + 2)) for c in row] for row in data]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde4f0")),
                           ("GRID", (0, 0), (-1, -1), .3, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(t)
    story.append(Spacer(1, 6))


def fig(name, w=16, caption=None):
    p = os.path.join(FIG_DIR, name)
    if os.path.exists(p):
        from PIL import Image as PI
        iw, ih = PI.open(p).size
        story.append(Image(p, width=w * cm, height=w * cm * ih / iw))
        if caption:
            P(f"<i>{caption}</i>", SM)
        story.append(Spacer(1, 6))


f3 = lambda x: f"{x:.4f}"  # noqa: E731
final = tvu[tvu.model.str.contains("BO")].iloc[0]
untuned = tvu[tvu.model.str.contains("untuned")].iloc[0]
fh = hmmc[hmmc.model == "final"].iloc[0]
tr_inv, te_inv = inv[inv.partition == "train"], inv[inv.partition == "student_test"]

# ================= Title =================
P("PathSnap: Map-Matching &amp; Route Reconstruction — Technical Report", H1)
P("UE24CS352A Machine Learning Hackathon · Team: ____________ · SRNs: ____________")
P("<b>AI-assistance disclosure.</b> An AI coding assistant was used to help write parts of the code and "
  "to draft this report. All experiments were executed by running the repository code; every number, table and figure "
  "below is produced by the scripts in the repository (see README, <i>python run_all.py</i>). [Edit this statement to match "
  "your team's actual use, as required by the event policy.]")
P("<b>Pipeline.</b> raw trip files → cleaning &amp; local metric projection → candidate road arcs per GPS observation "
  f"(radius {icfg['candidate_radius_m']:.0f} m) → {len(FEATURES)} engineered features → binary classifier "
  "(is this candidate the vehicle's road segment?) → per-observation probabilities → HMM/Viterbi decoding over the "
  "directed road network → road segment for every GPS observation → FastAPI (Laptop 1) → Streamlit/Folium UI (Laptop 2).")

# ================= 2.2 EDA =================
P("2.2 Dataset Understanding and Exploratory Data Analysis", H2)
P(f"<b>Partitions used.</b> All EDA uses the <b>train</b> (80 trips, {int(tr_inv.n_obs_clean.sum()):,} GPS observations) and "
  f"<b>student-test</b> (10 trips, {int(te_inv.n_obs_clean.sum()):,} observations) partitions only. The evaluator-test "
  "partition is never read by any development script.")
P("<b>What the files are.</b> Each trip has its own extract of the road network. <i>.nodes</i> gives lon/lat per node "
  "(row index = node id); <i>.arcs</i> gives directed straight segments from_node→to_node (row index = arc id); "
  "<i>.track</i> gives lon, lat, time (s); <i>.route</i> lists the reference arc ids in driving order, with "
  "<b>no timestamps</b>. A GPS row therefore does not correspond to any network row: the link between observations and "
  "arcs must be inferred geometrically.")
P("<b>Preprocessing.</b> (1) Drop rows with non-finite or out-of-range coordinates, sort by timestamp, and drop duplicate "
  f"timestamps. The data has {int(inv.n_nan_rows.sum())} NaN rows, so this is a safeguard only. (2) Coordinates are angles, "
  "so each trip is projected to a local tangent plane in metres centred on its GPS centroid "
  "(x = R·cos φ0·Δλ, y = R·Δφ). Trips are spread worldwide "
  f"(longitude {inv.lon_min.min():.0f}° to {inv.lon_max.max():.0f}°) and each spans at most {inv.extent_km.max():.0f} km, "
  "so one global UTM zone is impossible and the local projection's error stays far below GPS noise. "
  "(3) Only arcs inside the trajectory bounding box + 500 m are indexed. They are densified every 10 m into a KD-tree "
  "for fast radius search. (4) <b>Observation labels:</b> because the route has no timestamps, we align GPS points to "
  "route positions with a monotone dynamic programme, D[i,j] = c(i,j) + min<sub>k≤j</sub> D[i−1,k], minimising the total "
  "point-to-arc distance with the route position never moving backwards. Points more than 100 m from their aligned "
  f"route arc are marked <i>label-unreliable</i> ({(1-tr_inv.frac_label_valid.mean())*100:.1f}% of train points, "
  "mostly in trips whose reference route covers only part of the drive) and are excluded from fitting and scoring.")
summ = pd.DataFrame({
    "statistic": ["trips", "GPS observations", "median obs / trip", "median arcs / trip", "NaN rows",
                  "share of 1 s steps", "gaps > 10 s", "largest gap (s)", "median GPS→route dist (m)", "label-reliable points"],
    "train": [len(tr_inv), f"{int(tr_inv.n_obs_clean.sum()):,}", int(tr_inv.n_obs_clean.median()), int(tr_inv.n_arcs.median()),
              int(tr_inv.n_nan_rows.sum()), f"{tr_inv.frac_dt_1s.mean():.3f}", int(tr_inv.n_gaps_gt10s.sum()),
              f"{tr_inv.dt_max.max():.0f}", f"{tr_inv.gps_to_route_median_m.median():.2f}", f"{tr_inv.frac_label_valid.mean()*100:.1f}%"],
    "student_test": [len(te_inv), f"{int(te_inv.n_obs_clean.sum()):,}", int(te_inv.n_obs_clean.median()), int(te_inv.n_arcs.median()),
                     int(te_inv.n_nan_rows.sum()), f"{te_inv.frac_dt_1s.mean():.3f}", int(te_inv.n_gaps_gt10s.sum()),
                     f"{te_inv.dt_max.max():.0f}", f"{te_inv.gps_to_route_median_m.median():.2f}", f"{te_inv.frac_label_valid.mean()*100:.1f}%"]})
table(summ, [6 * cm, 4.5 * cm, 4.5 * cm])
P("<b>Missing-value analysis.</b> There are no missing coordinates or timestamps. The real form of 'missing data' is "
  "<b>missing time</b>: sampling is nominally 1 Hz, but there are gaps (see the table and the sampling-interval figure). "
  "In the HMM these are handled by cutting the chain when the gap exceeds 60 s.")
P("<b>Patterns observed.</b> (a) Almost every arc has a reverse twin (two-way roads), so distance alone cannot decide the "
  "driving direction. Heading and along-arc motion are needed. (b) GPS points lie a median of ~5 m from the route, but "
  "dense urban networks put several arcs within that distance, so the nearest arc is often wrong. (c) Reference routes "
  "are almost perfectly connected (consecutive arcs share a node in "
  f"{inv.route_connected_frac.mean()*100:.1f}% of steps), which supports a connectivity-aware decoder. "
  "(d) Some reference routes are much shorter than the GPS track, so the labels are not perfect ground truth.")
P(f"<b>Candidate set.</b> Radius {icfg['candidate_radius_m']:.0f} m with a minimum of 3 arcs. Average "
  f"{cst.cands_per_obs.iloc[0]:.1f} candidates per observation (train) and {cst.cands_per_obs.iloc[1]:.1f} (student test). "
  f"<b>Candidate miss rate</b> is {cst.candidate_miss_rate.iloc[0]*100:.2f}% (train) and {cst.candidate_miss_rate.iloc[1]*100:.3f}% (student test), "
  "so the reference arc is almost always among the candidates.")
P("<b>Features (and why).</b> <i>dist, dist_rank, dist_minus_min, dist_ratio</i>: absolute and relative closeness, because GPS noise "
  "is a few metres. <i>heading_diff, heading_cos, heading_valid</i>: the GPS movement bearing over ±2 s versus the arc bearing, "
  "which separates the two directions of a two-way road; it is marked invalid when the vehicle is nearly stopped. "
  "<i>along_move, along_move_norm</i>: signed motion along the arc over ±3 s, a second direction cue. "
  "<i>ctx_dist_mean/max</i>: distance of neighbouring observations (−6, −3, +3, +6 s) to the same arc, which rewards arcs "
  "the trajectory follows rather than arcs it only crosses. <i>t_frac, endpoint_dist, arc_len</i>: where the point projects "
  "on the arc, since points near intersections are ambiguous. <i>n_cands, deg_out_to, deg_in_from, has_reverse</i>: local "
  "network complexity. <i>speed, time_gap</i>: kinematic context.")
fig("eda_distributions.png", 16, "Train vs student-test distributions (obs per trip, length, network size, gaps, GPS-route distance, location).")
fig("eda_sampling_interval.png", 9, "Sampling interval, train partition.")
fig("eda_trajectories.png", 17, "Train trips: raw GPS (blue), reference route (green), nearby road network (grey).")

# ================= 3 Models =================
story.append(PageBreak())
P("3. Model Experimentation and Comparison", H2)
P("Every model is a <b>binary candidate scorer</b>. Each row is a (GPS observation, candidate arc) pair with y = 1 if "
  "the arc is the reference arc. The predicted probability is used (i) to pick the Top-1 arc per observation and (ii) "
  "as the HMM emission. <b>Protocol:</b> 5-fold GroupKFold over the 80 training trips (groups = trips, so no trip is "
  "split across folds). Each model is then refit on all training trips and evaluated once on the 10 student-test trips. "
  "Training uses every 3rd observation (1 Hz points are near-duplicates); student-test uses every observation. "
  "<b>Primary metric: observation-level Top-1 accuracy</b>, because the task is to choose one road per observation. "
  "Macro F1 (candidate level, threshold 0.5) and log loss are also reported. Log loss matters because the HMM uses the "
  "probabilities themselves.")
t = cmp_.sort_values("test_top1", ascending=False)
table(pd.DataFrame({
    "Model": t.model, "Family": t.family, "Key configuration": t.config,
    "CV Macro F1 (±sd)": [f"{a:.4f} ±{b:.4f}" for a, b in zip(t.cv_macro_f1, t.cv_macro_f1_std)],
    "Top-1 Acc (test)": t.test_top1.map(f3), "Route continuity (test, clf-only)": t.test_route_continuity.map(f3),
    "Student-test Macro F1": t.test_macro_f1.map(f3), "Log loss (test)": t.test_log_loss.map(f3),
    "Inference ms / 1k obs": t.infer_ms_per_1k_obs.map(lambda v: f"{v:.1f}")}),
    [1.9 * cm, 1.9 * cm, 3.6 * cm, 2 * cm, 1.5 * cm, 1.8 * cm, 1.6 * cm, 1.4 * cm, 1.5 * cm])
table(pd.DataFrame({"Model": t.model, "Precision (test)": t.test_precision.map(f3), "Recall (test)": t.test_recall.map(f3),
                    "F1 positive (test)": t.test_f1_pos.map(f3), "CV Top-1 (±sd)": [f"{a:.4f} ±{b:.4f}" for a, b in zip(t.cv_top1, t.cv_top1_std)],
                    "CV log loss": t.cv_log_loss.map(f3), "Train fit time (s)": t.train_time_s.map(lambda v: f"{v:.1f}"),
                    "Train Top-1*": t.train_top1.map(f3), "Train macro F1*": t.train_macro_f1.map(f3)}))
P("*Train scores: the final fitted model re-scored on 16 of its own training trips, used as overfitting evidence. "
  "Inference time is the classifier only, measured on the student-test candidate table.", SM)
fig("cv_macro_f1.png", 14, "5-fold group-CV macro F1 per model (train partition).")
fig("test_comparison.png", 17, "Student-test Top-1 accuracy, macro F1 and log loss.")
fig("accuracy_vs_speed.png", 11, "Accuracy vs inference cost.")

TXT = {
    "LogReg": ("A linear baseline. If closeness, heading agreement and context distance add up roughly linearly in the log-odds, "
               "a logistic model gives well-calibrated probabilities that suit HMM emissions.",
               "Strengths: fast, calibrated, interpretable coefficients. Limitations: cannot model interactions such as "
               "'heading matters only when moving' or non-monotone effects of t_frac/arc length.",
               "C=1.0 (L2), max_iter=2000, features standardised.", "Binary cross-entropy (log loss) + L2 penalty, lbfgs."),
    "GaussianNB": ("A probabilistic generative baseline: models each feature per class as a Gaussian after a quantile-normal transform.",
                   "Strengths: almost no training cost, works with little data. Limitations: the independence assumption is badly "
                   "violated (dist, dist_rank, dist_ratio and ctx_dist are strongly correlated), so it double-counts evidence and "
                   "becomes over-confident, which shows up as the worst log loss.",
                   "var_smoothing=1e-9; QuantileTransformer(output_distribution='normal').", "Maximum-likelihood class-conditional Gaussians (no iterative loss)."),
    "KNN": ("Instance-based: a candidate is the right road if similar (point, arc) situations in training were right roads.",
            "Strengths: non-parametric, captures local feature interactions. Limitations: needs scaling, prediction is slow "
            "(distance to all stored rows), suffers in 20 dimensions, and with distance weighting its probabilities are coarse. "
            "It was fit on a 120k-row subsample for tractability.",
            "k=25, weights='distance', Euclidean on standardised features.", "None (lazy learner); majority vote weighted by 1/distance."),
    "DecisionTree": ("Learns threshold rules such as 'dist < 8 m and heading_diff < 40°', matching how a human would reason.",
                     "Strengths: interpretable, handles interactions and unscaled features, fast inference. Limitations: high "
                     "variance and piecewise-constant probabilities; depth must be limited to avoid overfitting.",
                     "max_depth=12, min_samples_leaf=20.", "Gini impurity reduction (greedy splits)."),
    "RandomForest": ("Bagging many decorrelated trees reduces the variance of a single tree while keeping its ability to model interactions.",
                     "Strengths: robust, little tuning, good probabilities. Limitations: large model files and slower inference "
                     "than boosting; probabilities are averages of leaf frequencies (somewhat conservative).",
                     "100 trees, min_samples_leaf=5, max_samples=0.3, max_features='sqrt'.", "Gini impurity per tree; bootstrap aggregation."),
    "ExtraTrees": ("Like Random Forest but with random split thresholds. That adds variance reduction and speeds up training.",
                   "Strengths: fast training, smooth probabilities. Limitations: random thresholds can miss sharp boundaries "
                   "(e.g. exact heading cut-offs); slightly higher bias.",
                   "150 trees, min_samples_leaf=5, bootstrap, max_samples=0.3.", "Gini impurity with randomised thresholds."),
    "HistGB": ("Gradient boosting fits trees sequentially to the errors of the previous ones. It is the strongest family on "
               "tabular data with interacting features.",
               "Strengths: best accuracy/log-loss trade-off, very fast histogram training and inference, native early "
               "stopping, small model. Limitations: more hyperparameters, can overfit without regularisation or early stopping.",
               "learning_rate=0.1, max_iter=200 (early stopping), max_leaf_nodes=31, min_samples_leaf=20, l2=0.",
               "Binary log loss minimised by gradient boosting (Newton steps per leaf)."),
    "MLP": ("A neural network can learn smooth non-linear combinations of the geometric features.",
            "Strengths: flexible, smooth probabilities. Limitations: needs scaling and careful regularisation, is "
            "sensitive to initialisation, slower to train on CPU, and is usually not better than boosting on small tabular "
            "feature sets.",
            "hidden=(64,32) ReLU, Adam, alpha=1e-4, batch=512, max_iter=60, early stopping (10% validation).",
            "Binary cross-entropy (log loss) + L2, Adam optimiser."),
}
P("3.2 Per-model questions", H2)
for _, r in cmp_.sort_values("test_top1", ascending=False).iterrows():
    why, sl, hp, loss = TXT[r.model]
    P(f"{r.model} ({r.family})", H3)
    P(f"<b>A1 Why suitable.</b> {why}")
    P(f"<b>A2 Strengths / limitations.</b> {sl}")
    P(f"<b>B5 Hyperparameters (initial).</b> {hp} <b>B6 Objective.</b> {loss}")
    gap = r.train_top1 - r.cv_top1
    verdict = ("mild overfitting (train Top-1 clearly above CV)" if gap > 0.03 else
               "no meaningful overfitting (train and CV Top-1 are close)" if gap > -0.01 else "underfitting-like behaviour")
    P(f"<b>C7–C8 Student-test results.</b> Top-1 accuracy <b>{r.test_top1:.4f}</b>, macro F1 <b>{r.test_macro_f1:.4f}</b>, "
      f"log loss <b>{r.test_log_loss:.4f}</b>, precision {r.test_precision:.4f}, recall {r.test_recall:.4f}, "
      f"classifier-only route continuity {r.test_route_continuity:.4f}.")
    P(f"<b>C9 Over/under-fitting.</b> Train Top-1 {r.train_top1:.4f} vs CV Top-1 {r.cv_top1:.4f} (gap {gap:+.4f}) vs "
      f"student-test {r.test_top1:.4f}; fold std of macro F1 {r.cv_macro_f1_std:.4f} → {verdict}.")
story.append(PageBreak())
P("3.3 Model visualisations", H2)
fig("training_histories.png", 16, "Real epoch-wise histories (MLP: training log loss + validation accuracy; HistGB: train/validation loss). "
    "Tree/kNN/NB/LogReg models do not produce epoch histories, so none are drawn.")
for nm in cmp_.sort_values("test_top1", ascending=False).model:
    fig(f"cm_{nm}.png", 6.5, f"Confusion matrix (candidate level, threshold 0.5), student test: {nm}")

# ================= 4 BO =================
story.append(PageBreak())
P("4. Bayesian Hyperparameter Optimisation (HistGradientBoosting)", H2)
hg, rf = cmp_.set_index("model").loc["HistGB"], cmp_.set_index("model").loc["RandomForest"]
P(f"<b>Why this model.</b> HistGradientBoosting and Random Forest are statistically tied on CV macro F1 "
  f"({hg.cv_macro_f1:.4f} ±{hg.cv_macro_f1_std:.4f} vs {rf.cv_macro_f1:.4f} ±{rf.cv_macro_f1_std:.4f}). HGB has the best "
  f"student-test Top-1 ({hg.test_top1:.4f} vs {rf.test_top1:.4f}) and log loss ({hg.test_log_loss:.4f} vs {rf.test_log_loss:.4f}), "
  f"and it is about {rf.infer_ms_per_1k_obs/hg.infer_ms_per_1k_obs:.1f}× faster at inference and {rf.train_time_s/hg.train_time_s:.0f}× faster to train. "
  "Fast inference matters because the API scores roughly 10 candidates per GPS point for thousands of points per request. "
  "Good log loss matters because the decoder uses the probabilities directly. Fast training also makes a "
  "20-trial Bayesian search affordable.")
P("<b>Method.</b> Our own Gaussian-process Bayesian optimisation (src/pathsnap/bayesopt.py): the parameters are mapped to [0,1] "
  "(log scale where appropriate), there are 5 stratified random initial trials, then 15 iterations that fit a GP (Matern 5/2 + "
  "white noise) and choose the next point by maximising <b>Expected Improvement</b> over 4000 proposals. "
  "<b>Objective:</b> mean observation-level Top-1 accuracy over 3-fold GroupKFold on the 80 training trips. "
  f"<b>Trials:</b> {bp['n_trials']}; total time {bp['total_seconds']/60:.1f} min. Every trial is in results/bayes_opt_trials.csv.")
table(pd.DataFrame({"hyperparameter": ["learning_rate", "max_iter (upper bound, early stopping)", "max_leaf_nodes", "min_samples_leaf",
                                       "l2_regularization", "max_features"],
                    "range / distribution": ["0.02–0.3 log-uniform", "100–600 uniform int", "15–127 log-uniform int", "10–300 log-uniform int",
                                             "1e-4–10 log-uniform", "0.4–1.0 uniform"],
                    "best value": [f"{bp['best_params'][k]:.4g}" if isinstance(bp['best_params'][k], float) else bp['best_params'][k]
                                   for k in ["learning_rate", "max_iter", "max_leaf_nodes", "min_samples_leaf", "l2_regularization", "max_features"]]}),
      [6 * cm, 5 * cm, 4 * cm])
show = bo[["trial", "phase", "objective", "learning_rate", "max_iter", "max_leaf_nodes", "min_samples_leaf", "l2_regularization", "max_features", "seconds"]].copy()
for c in ["objective", "learning_rate", "l2_regularization", "max_features"]:
    show[c] = show[c].map(lambda v: f"{v:.4g}")
show["seconds"] = show.seconds.map(lambda v: f"{v:.0f}")
table(show, fs=6.5)
fig("bayes_opt.png", 16, "Optimisation progress and explored region.")
P("<b>Untuned vs tuned</b> (same 5-fold group CV, same student-test partition):")
table(pd.DataFrame({"model": tvu.model, "CV macro F1": tvu.cv_macro_f1.map(f3), "CV Top-1": tvu.cv_top1.map(f3),
                    "CV log loss": tvu.cv_log_loss.map(f3), "Test Top-1": tvu.test_top1.map(f3), "Test macro F1": tvu.test_macro_f1.map(f3),
                    "Test log loss": tvu.test_log_loss.map(f3), "Inference ms/1k obs": tvu.infer_ms_per_1k_obs.map(lambda v: f"{v:.1f}")}))
better = final.cv_top1 > untuned.cv_top1
P(f"The tuned model {'improves' if better else 'does not improve'} CV Top-1 ({untuned.cv_top1:.4f} → {final.cv_top1:.4f}) and gives "
  f"student-test Top-1 {untuned.test_top1:.4f} → {final.test_top1:.4f}, log loss {untuned.test_log_loss:.4f} → {final.test_log_loss:.4f}. "
  "The differences are small. Feature quality and the decoder matter more than boosting hyperparameters here, which is "
  "itself a finding.")

# ================= 5 HMM =================
story.append(PageBreak())
P("5. HMM and Viterbi Route Reconstruction", H2)
hc = icfg["hmm"]
P("<b>1. Hidden states:</b> at each GPS observation, the hidden state is one of its candidate arcs (more precisely, the "
  "projection of the point onto that arc). Each state keeps its arc id, so every decoded state maps straight to a road segment. "
  "<b>2. Emission:</b> log P(z<sub>i</sub>|arc) = w<sub>emit</sub>·log(p<sub>clf</sub>(arc) / Σ p<sub>clf</sub>), the tuned "
  "classifier's probability renormalised over the observation's candidates (floored at 1e-4). "
  "<b>3. Transition:</b> log P = −|d<sub>net</sub> − d<sub>gc</sub>| / β, where d<sub>gc</sub> is the straight-line distance between "
  "consecutive GPS points and d<sub>net</sub> is the shortest driving distance on the <i>directed</i> network from the projection on "
  "arc a to the projection on arc b (remaining part of a + Dijkstra(a.to → b.from) + covered part of b; on the same arc, |Δt|·length). "
  "A real driving path has about the same length as the straight-line jump, while detours, U-turns through blocks and wrong-direction "
  "arcs are penalised. "
  f"<b>Viterbi:</b> δ<sub>i</sub>(s) = e<sub>i</sub>(s) + max<sub>s'</sub>[δ<sub>i−1</sub>(s') + T(s',s)], with back-pointers and "
  f"backtracking. Chosen parameters: β = {hc['beta']} m, w<sub>emit</sub> = {hc['w_emit']}, gap break = {hc['gap_break_s']} s.")
P("<b>4. Missing candidates / large gaps:</b> every observation always has at least 3 candidates (fallback search up to 300 m). "
  "When the time gap exceeds the break threshold, the chain is cut and decoding restarts with emissions only, because "
  "network distances over long gaps are not informative. <b>5. Disconnected segments:</b> if no candidate at i is reachable "
  "from any candidate at i−1 within the Dijkstra limit (2·d<sub>gc</sub> + 300 m), the transition is impossible (−∞) for all "
  "pairs. The chain is then cut and a new segment starts, so the decoder never fails or hallucinates a path. "
  f"Breaks on student test: {int(hpt[hpt.model=='final'].hmm_breaks_gap.sum())} gap-based, {int(hpt[hpt.model=='final'].hmm_breaks_unreachable.sum())} unreachability-based.")
P("<b>Decoder tuning (no student-test leakage):</b> 10 randomly chosen training trips were held out; the tuned HGB was fit on the "
  "other 70, and a (β, w<sub>emit</sub>) grid was evaluated on the held-out trips at full 1 Hz.")
fig("hmm_grid.png", 8, "Decoder grid search, held-out training trips.")
P("Note: the score is w<sub>emit</sub>·emission − |d<sub>net</sub>−d<sub>gc</sub>|/β, so only the product w<sub>emit</sub>·β changes "
  "the decoded path. This is why the grid has equal values along its anti-diagonals. The best product (smallest, i.e. the strongest "
  "trust in network geometry relative to the classifier) lies on the edge of the grid. A finer grid below 2.5 m is future work.")
P("<b>6. Did HMM help?</b> Classifier-only vs classifier+HMM on the student-test trips, for every model (Top-1 is weighted by observations):")
table(pd.DataFrame({"model": hmmc.model, "Top-1 clf": hmmc.clf_top1.map(f3), "Top-1 +HMM": hmmc.hmm_top1.map(f3),
                    "continuity clf": hmmc.clf_continuity.map(f3), "continuity +HMM": hmmc.hmm_continuity.map(f3),
                    "route F1 clf": hmmc.clf_route_f1.map(f3), "route F1 +HMM": hmmc.hmm_route_f1.map(f3),
                    "decode time (s, 10 trips)": hmmc.decode_s_total.map(lambda v: f"{v:.1f}")}))
fin = hpt[hpt.model == "final"]
worst = fin.assign(d=fin.hmm_top1 - fin.clf_top1).sort_values("d").iloc[0]
P(f"For the final model, HMM raises Top-1 from {fh.clf_top1:.4f} to <b>{fh.hmm_top1:.4f}</b>, continuity from {fh.clf_continuity:.4f} to "
  f"<b>{fh.hmm_continuity:.4f}</b> and route F1 from {fh.clf_route_f1:.4f} to <b>{fh.hmm_route_f1:.4f}</b>. "
  "The weaker the classifier, the larger the gain, because the network constraint repairs isolated wrong picks (e.g. a parallel street "
  f"or the reverse direction for one second). The smallest gain is on trip {worst.trip_id} ({worst.clf_top1:.4f} → {worst.hmm_top1:.4f}). "
  "HMM cannot help where the classifier is consistently wrong for a long stretch, or where the reference itself is questionable. "
  "Continuity is below 1 even after HMM because our continuity metric only counts directly adjacent arcs, while 1 Hz jumps can skip "
  "very short arcs at junctions.")
fig("hmm_vs_classifier.png", 17, "Classifier-only vs classifier+HMM, student test, all models.")
fig("hmm_per_trip_final.png", 12, "Final model, per student-test trip.")

# ================= 6 maps =================
story.append(PageBreak())
P("6. Geographical Visualisation (Folium)", H2)
P("Interactive Folium maps are generated by scripts/06_maps.py (results/maps/trip_&lt;id&gt;.html). Layers, each in a toggle "
  "(LayerControl) with a legend: raw GPS (blue dots), nearby road network (grey), reference route (thick green), classifier-only "
  "route (orange dashed), classifier+HMM route (red), start/end markers; the view is fit to the trajectory bounds. "
  "The same layers rendered as static images for the PDF:")
for f in sorted(os.listdir(FIG_DIR)):
    if f.startswith("map_"):
        fig(f, 13, f"Student-test trip {f[4:-4]}: GPS, network, reference, classifier-only and HMM routes. "
                   "[Replace with a screenshot of the Folium HTML for the final PDF.]")

# ================= 7 API/UI =================
story.append(PageBreak())
P("7. API, UI and Deployment", H2)
P("<b>Laptop 1 (model + API):</b> <i>uvicorn api.main:app --host 0.0.0.0 --port 8000</i>. On start-up it loads "
  "models/final_model.joblib and models/inference_config.json (features, candidate radius, HMM parameters). "
  "Endpoints: GET /health, GET /trips (lists the trip folders in data/evaluator_test), POST /predict {trip_id}. "
  "/predict loads the trip <b>without</b> its route, runs candidates → features → classifier → Viterbi, and only "
  "<i>after</i> prediction loads the reference route server-side to compute Top-1 accuracy, continuity and route "
  "precision/recall/F1. Only these aggregate metrics are returned; reference arcs are never sent. Each request is logged "
  "in the Laptop-1 console (trip id, number of observations and candidates, HMM segments, metrics, timings).")
P("<b>Laptop 2 (UI):</b> <i>streamlit run ui/app.py</i>. The evaluator enters Laptop 1's address, sees the 10 evaluation trips, "
  "selects one, and gets the metrics plus an interactive Folium map (raw GPS + predicted route, optional classifier-only "
  "layer, legend, layer control). The UI has no access to data or models.")
P("[Insert screenshots: (1) UI with trip list, (2) UI with the reconstructed route and metrics, (3) Laptop-1 CMD showing the API log lines for the request.]")

P("8. Limitations and assumptions", H2)
P("The reference route has no timestamps, so per-observation labels come from our DP alignment and are approximate near "
  "junctions. Points farther than 100 m from the route are excluded rather than guessed. Arcs are treated as straight "
  "segments (the data stores them that way). Training uses every 3rd observation. Decoder parameters were tuned on held-out "
  "training trips only. The evaluator-test partition was never used in development.")

os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
out = os.path.join(ROOT, "reports", "PathSnap_Technical_Report.pdf")
SimpleDocTemplate(out, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                  title="PathSnap Technical Report").build(story)
print("report ->", out)
