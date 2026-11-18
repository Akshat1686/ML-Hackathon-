import argparse
import ast
import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

INK, MUTED, TEAL, LINE = "#182537", "#476176", "#00675d", "#d8e4e9"
BODY = ParagraphStyle("body", fontName="Helvetica", fontSize=10, leading=14, spaceAfter=8,
                      textColor=colors.HexColor(INK), alignment=TA_LEFT)
H1 = ParagraphStyle("title", parent=BODY, fontName="Helvetica-Bold", fontSize=31, leading=36, spaceAfter=12)
H2 = ParagraphStyle("heading", parent=BODY, fontName="Helvetica-Bold", fontSize=18, leading=22, spaceAfter=10)
H3 = ParagraphStyle("subheading", parent=BODY, fontName="Helvetica-Bold", fontSize=11, leading=14,
                    spaceBefore=7, spaceAfter=3)
CAPTION = ParagraphStyle("caption", parent=BODY, fontSize=8, leading=11, textColor=colors.HexColor(MUTED))

NOTES = {
    "Logistic Regression": ("Binary log loss with regularization and fold-fitted standardization.",
        "A fast linear baseline. A single linear decision surface limits interactions between heading, proximity and road length."),
    "Gaussian Naive Bayes": ("Class-conditional Gaussian maximum likelihood with variance smoothing.",
        "A fast probabilistic baseline. Distance features share geometry and violate its conditional independence assumption."),
    "Decision Tree": ("Greedy Gini-impurity splits.",
        "A single tree can learn different proximity rules for moving and stationary observations. Its leaf probabilities depend on training samples."),
    "Extra Trees": ("Gini splits over randomized features and thresholds, with probabilities averaged across trees.",
        "An ensemble can model nonlinear interactions while averaging over different split points. More trees increase scoring cost."),
    "HistGradientBoosting": ("Binary log loss with regularized additive trees and histogram-binned split search.",
        "Boosting can refine errors at intersections and parallel roads using combinations of features. Tree depth, bins and iteration count affect cost."),
}


def generate(out, destination):
    data = json.loads((out / "results.json").read_text(encoding="utf-8"))
    models = data["models"]
    story = []

    def add(text, style=BODY):
        story.append(Paragraph(escape(text), style))

    def section(name):
        story.append(PageBreak())
        add(name, H2)

    def table(rows, widths):
        cell = ParagraphStyle("cell", parent=BODY, fontSize=8.2, leading=10)
        head = ParagraphStyle("head", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)
        wrapped = [[Paragraph(escape(str(value)), head if i == 0 else cell) for value in row]
                   for i, row in enumerate(rows)]
        grid = Table(wrapped, colWidths=widths, hAlign="LEFT")
        grid.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(TEAL)),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#eef5f6"), colors.white]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.extend([grid, Spacer(1, 12)])

    def picture(name, height, caption):
        path = out / name
        image = Image(str(path))
        scale = min(486 / image.imageWidth, height / image.imageHeight)
        image.drawWidth, image.drawHeight = image.imageWidth * scale, image.imageHeight * scale
        story.append(image)
        add(caption, CAPTION)

    add("PathSnap", H1)
    add("Technical report · Distributed map matching and route reconstruction", H3)
    add("Development experiment on the supplied PathSnap trip data")
    add("2. Dataset understanding and EDA", H2)
    with (out / "inventory.csv").open(encoding="utf-8") as stream:
        inventory = list(csv.DictReader(stream))
    overview = [["Partition", "Trips", "GPS points", "Road arcs", "Missing fields", "Largest GPS gap"]]
    for split in ("train", "student_test"):
        subset = [r for r in inventory if r["partition"] == split]
        overview.append([split, len(subset), f"{sum(int(r['gps']) for r in subset):,}",
                         f"{sum(int(r['arcs']) for r in subset):,}", sum(int(r["missing_values"]) for r in subset),
                         f"{max(float(r['max_gap_s']) for r in subset):,.0f} s"])
    table(overview, [92, 47, 83, 94, 77, 93])
    add("Each trip supplies timestamped GPS and a directed road graph. Headerless files were parsed with shape, finite-value, timestamp and index checks. A per-trip local equirectangular projection converts coordinate differences to metres.")
    steps = sum(int(row["gps"]) - 1 for row in inventory)
    regular = sum(int(row["one_second_steps"]) for row in inventory)
    fragmented = sum(int(row["weak_components"]) > 1 for row in inventory)
    broken = sum(int(row["reference_breaks"]) > 0 for row in inventory)
    add(f"Across the two development partitions, {regular / steps:.1%} of GPS intervals were one second, {fragmented} trip graphs had multiple weak components, and {broken} reference routes contained a break in directed adjacency.")
    add("The reference lists directed arcs in travel order. A separate alignment projects GPS to candidate occurrences along that route, then minimizes distance, heading and travel-order penalties. Alignments restart after gaps above 35 seconds. Points whose aligned arc is over 55 m away are omitted from observation-label metrics and counted in coverage.")
    add("The inference candidate search uses a KD-tree over sampled segment geometry. It retrieves the eight nearest arcs by exact projected segment distance. The classifier receives distance, heading agreement and validity, projected fraction, road length, GPS speed, and distances to both arc endpoints. Geometry separates close roads, while movement helps distinguish opposite directions.")
    picture("eda.png", 185, "Measured trip sizes, GPS gaps and graph sizes. Partition-level and per-trip statistics are retained in inventory.csv.")

    section("3. Model experimentation and comparison")
    add("Three grouped CV folds keep each training trip intact. The student-test partition is used for the held-out development comparison. Candidate precision, recall and macro F1 use a 0.5 decision threshold. Observation Top-1 includes candidate misses among aligned points. Inference time here covers scoring all student-test feature rows.")
    header = ["Model", "Family", "CV F1", "Student F1", "Top-1", "Continuity", "Log loss", "Score s"]
    model_rows = [[r["model"], r["family"], f"{r['cv_macro_f1']:.3f}", f"{r['macro_f1']:.3f}",
                   f"{r['top1']:.3f}", f"{r['continuity']:.3f}", f"{r['log_loss']:.3f}",
                   f"{r['inference_s']:.3f}"] for r in models]
    table([header, *model_rows], [106, 68, 45, 59, 43, 63, 50, 52])
    picture("model_comparison.png", 230, "Five trained classifiers and the tuned configuration. CV uses training trips, while the centre and right panels use student-test.")
    add(f"{data['selected']} had the strongest grouped CV macro F1 among the initial candidates. This metric weights the match class equally with the more numerous non-matches. The nearest-road baseline reached {data['nearest']['top1']:.1%} Top-1. Candidate miss rate was {data['classifier']['candidate_miss']:.2%} with the shared shortlist.")
    add("Model outputs, per-fold scores, training and inference times, and confusion matrices are recorded in model_comparison.csv and results.json.")

    section("3. Individual model experiments")
    for row in models[:-1]:
        name = row["model"]
        loss, note = NOTES[name]
        add(name, H3)
        add(f"{loss} {note} Settings were " + ", ".join(f"{k}={v}" for k, v in row["parameters"].items()) + ".")
        gap = row["train_macro_f1"] - row["cv_macro_f1"]
        diagnosis = ("The gap suggests sensitivity to unseen trips." if gap > 0.025
                     else "The scores show a small train-to-CV gap.")
        add(f"Candidate F1 was {row['train_macro_f1']:.3f} on training data and {row['cv_macro_f1']:.3f} in grouped CV (gap {gap:+.3f}). {diagnosis} Student-test Top-1, macro F1 and log loss were {row['top1']:.3f}, {row['macro_f1']:.3f} and {row['log_loss']:.3f}. Precision and recall were {row['precision']:.3f} and {row['recall']:.3f}.")
    add("A gap above 0.025 is used as a descriptive flag. Epoch-wise histories were unavailable for these runs.", CAPTION)

    section("4. Bayesian hyperparameter optimisation")
    best = models[-1]
    base = next(row for row in models if row["model"] == data["selected"])
    with (out / "bayesian_trials.csv").open(encoding="utf-8") as stream:
        trials = list(csv.DictReader(stream))
    ranges = ast.literal_eval(trials[0]["ranges"])
    add(f"Optuna TPE maximized three-fold grouped candidate macro F1 using the 80 training trips. The search ran {data['trials']} trials with seed {data['seed']}, four startup trials and the base configuration queued for the first trial. Trial {data['best_trial']} was best. Search runtime was {data['search_s']:.1f} seconds.")
    if data["selected"] == "HistGradientBoosting":
        add("Leaf count and minimum samples per leaf govern interaction complexity. Learning rate scales each boosting step; L2 regularization restrains leaf values. The ranges surround the base settings while keeping the tree and iteration budgets suitable for CPU training.")
    table([["Parameter", "Search distribution", "Best value"], *[
        [key, value, f"{best['parameters'][key]:.5g}"] for key, value in ranges.items()]], [124, 269, 93])
    table([["Metric", "Untuned", "Tuned"], *[
        [label, f"{base[key]:.4f}", f"{best[key]:.4f}"] for label, key in
        [("Grouped CV macro F1", "cv_macro_f1"), ("Student-test Top-1", "top1"),
         ("Student-test macro F1", "macro_f1"), ("Student-test log loss", "log_loss"),
         ("Student-test inference seconds", "inference_s")]]], [270, 108, 108])
    picture("tuning.png", 155, "Actual optimization trace. Trial parameters, scores and runtimes are retained in bayesian_trials.csv.")
    add(f"The search changed grouped CV macro F1 by {best['cv_macro_f1'] - base['cv_macro_f1']:+.4f} and student-test Top-1 by {100*(best['top1'] - base['top1']):+.2f} percentage points.")

    section("5. HMM and Viterbi route reconstruction")
    add("States are candidate directed arc IDs at each GPS observation. Normalized classifier probabilities supply emissions in log space. Transitions compare traveled network distance with observed displacement. Different arcs use a directed shortest path between their endpoints, bounded to 160 m for the connecting portion. Travel distance is also limited by elapsed time. The same-arc transition allows up to 25 m of backward jitter.")
    add("Viterbi adds the current log emission to the maximum previous log score plus transition score, retains the maximizing parent and backtracks. A gap over 35 seconds or the absence of a feasible transition starts a new route part. Every observation receives a nearest-road shortlist when the graph is nonempty. The decoder inserts connecting arcs and suppresses repeats within a route part.")
    decoder_rows = [["Method", "Top-1", "Continuity", "Arc F1", "Ordered pair F1", "Route parts"]]
    for label, key in [("Nearest road", "nearest"), ("Tuned classifier", "classifier"), ("Classifier + Viterbi", "decoded")]:
        row = data[key]
        decoder_rows.append([label, f"{row['top1']:.3f}", f"{row['continuity']:.3f}",
                             f"{row['arc_multiset_f1']:.3f}", f"{row['ordered_arc_pair_f1']:.3f}",
                             row["route_parts"]])
    table(decoder_rows, [123, 56, 72, 62, 98, 75])
    add(f"Top-1 pools {data['decoded']['aligned']:,} aligned observations from {data['decoded']['points']:,} student-test GPS points. Continuity measures consecutive observations linked by the produced route, excluding pairs across gaps over 35 seconds. Arc F1 compares directed arc counts and ordered pair F1 measures adjacent arc pairs. Sequence F1 is averaged across trips.")
    with (out / "route_comparison.csv").open(encoding="utf-8") as stream:
        routes = list(csv.DictReader(stream))
    prior = {r["trip"]: r for r in routes if r["method"] == "classifier"}
    per_trip = [(r["trip"], (int(r["correct"]) - int(prior[r["trip"]]["correct"])) / max(1, int(r["aligned"])))
                for r in routes if r["method"] == "hmm"]
    sample = min(per_trip, key=lambda pair: pair[1])
    add(f"Viterbi changed pooled point accuracy by {100*(data['decoded']['top1']-data['classifier']['top1']):+.2f} percentage points. The lowest per-trip Top-1 change was {sample[1]*100:+.2f} points on {sample[0]}. Strong continuity preferences can smooth away legitimate route changes and the finite shortlist limits recovery.")
    add(f"Route decoding plus metric calculation took {data['decoder_s']:.2f} seconds for the ten student-test trips with features and classifier scores already available.")

    section("6. Geographical visualisation")
    if (out / "folium_capture.png").exists():
        picture("folium_capture.png", 490, "Browser capture of the generated Folium map for student-test trip 00000052.")
    else:
        picture("route_plot.png", 490, "Exported geographic view using the same trip geometry as the generated Folium map.")
    add("The interactive Folium map contains GPS observations, candidate roads, reference arcs and the predicted route on separate layers. Its HTML and generator are included with the project. Development references appear in this offline comparison only.")

    section("7. API, UI and deployment")
    add("Laptop 1 serves FastAPI, the fitted classifier, raw trip inputs and server-side scoring. Laptop 2 serves the evaluator UI. The UI requests the trip list, posts the selected ID and displays the predicted geometry and aggregate metrics. The API logs each prediction and uses separate development and evaluation modes.")
    evidence = out / "api_smoke.json"
    if evidence.exists():
        smoke = json.loads(evidence.read_text(encoding="utf-8"))
        add(f"The local HTTP rehearsal returned {smoke['successful_predictions']} successful predictions for the ten student-test trips. It checked that an unknown trip returned HTTP 404 and responses included no reference geometry.")
    if (out / "ui_capture.png").exists():
        picture("ui_capture.png", 333, "Working evaluator UI using the development API and student-test data.")
    if (out / "api_capture.png").exists():
        picture("api_capture.png", 154, "Local receiving-terminal output captured during the development rehearsal.")
    elif evidence.exists():
        add("Recorded successful server response from the local rehearsal", H3)
        add(smoke["terminal_line"], CAPTION)
    add("The organizer-only trips and an independent second laptop were unavailable for this run. The LAN deployment and final evaluator scoring require the on-site rehearsal. The development evidence above uses the supplied student-test partition.")
    add("AI-use disclosure", H3)
    add("Codex assisted with implementation, experiment execution, visualization and drafting. The figures and tables use recorded runs. Review and reproduce the work before submission so the team can explain its own results.")

    def page(canvas, document):
        canvas.setStrokeColor(colors.HexColor(LINE))
        canvas.line(54, 43, A4[0] - 54, 43)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawString(54, 30, "PathSnap  /  Technical report")
        canvas.drawRightString(A4[0] - 54, 30, f"{document.page}")

    SimpleDocTemplate(str(destination), pagesize=A4, title="PathSnap | Technical Report",
                      leftMargin=54, rightMargin=54, topMargin=52, bottomMargin=56).build(
                          story, onFirstPage=page, onLaterPages=page)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--pdf", type=Path, default=Path("PathSnap_Technical_Report.pdf"))
    args = parser.parse_args()
    generate(args.out, args.pdf)
