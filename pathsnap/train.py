import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import optuna
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, log_loss, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from .data import FEATURES, GAP, K, align_reference, candidate_features, inventory, load_trip, reference
from .decode import BRIDGE_LIMIT_M, MAX_SPEED_M_S, decode, direct_links, sequence_metrics, split_baseline

SEED = 1729
CACHE_SCHEMA = "v1"
MODELS = {
    "Logistic Regression": ("Linear", {"C": 1.0, "max_iter": 150}),
    "Gaussian Naive Bayes": ("Probabilistic", {"var_smoothing": 1e-9}),
    "Decision Tree": ("Tree", {"max_depth": 9, "min_samples_leaf": 50}),
    "Extra Trees": ("Ensemble", {"n_estimators": 24, "max_depth": 12, "min_samples_leaf": 25}),
    "HistGradientBoosting": ("Boosting", {"learning_rate": 0.1, "max_iter": 55, "max_leaf_nodes": 23,
                                         "min_samples_leaf": 35, "l2_regularization": 0.1}),
}


def make_model(name, params):
    if name == "Logistic Regression":
        return make_pipeline(StandardScaler(), LogisticRegression(**params))
    if name == "Gaussian Naive Bayes":
        return GaussianNB(**params)
    if name == "Decision Tree":
        return DecisionTreeClassifier(random_state=SEED, **params)
    if name == "Extra Trees":
        return ExtraTreesClassifier(random_state=SEED, n_jobs=2, **params)
    return HistGradientBoostingClassifier(random_state=SEED, early_stopping=False, **params)


def suggest(name, trial, base):
    params = dict(base)
    if name == "Logistic Regression":
        params["C"] = trial.suggest_float("C", 0.01, 20, log=True)
    elif name == "Gaussian Naive Bayes":
        params["var_smoothing"] = trial.suggest_float("var_smoothing", 1e-11, 0.01, log=True)
    elif name in ("Decision Tree", "Extra Trees"):
        params["max_depth"] = trial.suggest_int("max_depth", 6, 15)
        params["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 15, 150, log=True)
    else:
        params["max_leaf_nodes"] = trial.suggest_int("max_leaf_nodes", 12, 36)
        params["learning_rate"] = trial.suggest_float("learning_rate", 0.035, 0.2, log=True)
        params["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 20, 140, log=True)
        params["l2_regularization"] = trial.suggest_float("l2_regularization", 0.02, 3.0, log=True)
    return params


def probabilities(model, features):
    return model.predict_proba(features.reshape(-1, len(FEATURES)))[:, 1].reshape(features.shape[:2])


def candidate_scores(labels, probability):
    truth, predicted = labels.ravel(), probability.ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predicted >= 0.5, labels=[False, True], average="macro", zero_division=0)
    return {"candidate_accuracy": float((truth == (predicted >= 0.5)).mean()),
            "precision": float(precision), "recall": float(recall), "macro_f1": float(f1),
            "log_loss": float(log_loss(truth, predicted, labels=[False, True])),
            "confusion": confusion_matrix(truth, predicted >= 0.5, labels=[False, True]).tolist()}


def evaluate(trip, arcs, features, labels, route, scores, use_decoder=False):
    valid = labels >= 0
    eligible = np.diff(trip.gps[:, 2]) <= GAP
    if use_decoder:
        chosen, parts, starts = decode(trip, arcs, features, scores)
        linked = ~starts[1:]
    else:
        chosen = arcs[np.arange(len(arcs)), scores.argmax(axis=1)]
        parts = split_baseline(trip, chosen)
        linked = direct_links(trip, chosen)
    result = {
        "aligned": int(valid.sum()), "points": len(labels), "correct": int((chosen[valid] == labels[valid]).sum()),
        "missed": int((~(arcs[valid] == labels[valid, None]).any(axis=1)).sum()),
        "linked": int((linked & eligible).sum()), "eligible_links": int(eligible.sum()),
        "route_parts": len(parts),
    }
    result.update(sequence_metrics(parts, route))
    return result


def summarize(rows):
    def ratio(numerator, denominator):
        return sum(row[numerator] for row in rows) / max(1, sum(row[denominator] for row in rows))
    return {**{key: sum(row[key] for row in rows) for key in ("points", "aligned", "route_parts")},
            "top1": ratio("correct", "aligned"), "candidate_miss": ratio("missed", "aligned"),
            "continuity": ratio("linked", "eligible_links"),
            **{key: float(np.mean([row[key] for row in rows]))
               for key in ("arc_multiset_f1", "ordered_arc_pair_f1")}}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(data, cache, out):
    all_trips, rows = {}, []
    cache.mkdir(parents=True, exist_ok=True)
    for split in ("train", "student_test"):
        all_trips[split] = []
        for directory in sorted((data / split).iterdir()):
            if not directory.is_dir():
                continue
            trip = load_trip(directory)
            route = reference(trip)
            source_times = [p.stat().st_mtime_ns for p in sorted(directory.glob(f"{directory.name}.*"))]
            cached = cache / f"{split}-{directory.name}-{CACHE_SCHEMA}.npz"
            if cached.exists():
                with np.load(cached) as previous:
                    if previous["timestamps"].tolist() == source_times:
                        arcs, features, labels = (previous[key] for key in ("arcs", "features", "labels"))
                    else:
                        cached.unlink()
            if not cached.exists():
                arcs, features = candidate_features(trip)
                labels = align_reference(trip, route)
                np.savez(cached, timestamps=source_times, arcs=arcs, features=features, labels=labels)
            rows.append({"partition": split, "trip": directory.name, **inventory(trip, route),
                         "aligned": int((labels >= 0).sum())})
            all_trips[split].append((directory, arcs, features, labels, route))
            print(f"Prepared {split}/{directory.name}", flush=True)
    write_csv(out / "inventory.csv", rows)
    return all_trips


def candidate_table(trips):
    samples = [(i, features[labels >= 0], arcs[labels >= 0] == labels[labels >= 0, None])
               for i, (_, arcs, features, labels, _) in enumerate(trips)]
    x = np.concatenate([features.reshape(-1, len(FEATURES)) for _, features, _ in samples])
    y = np.concatenate([target.ravel() for _, _, target in samples])
    groups = np.concatenate([np.full(target.size, i, dtype=np.int16) for i, _, target in samples])
    return x, y, groups


def run(data, cache, out, trials):
    if trials <= 4:
        raise ValueError("Use at least five trials so TPE runs after four startup trials")
    out.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    datasets = prepare(data, cache, out)
    prep_time = perf_counter() - start
    x, y, groups = candidate_table(datasets["train"])
    folds = list(GroupKFold(n_splits=3).split(x, y, groups))
    student = [(load_trip(directory), arcs, features, labels, route)
               for directory, arcs, features, labels, route in datasets["student_test"]]

    def cross_validation(name, params):
        values = []
        for fit, validation in folds:
            model = make_model(name, params).fit(x[fit], y[fit])
            predicted = model.predict(x[validation])
            values.append(float(precision_recall_fscore_support(
                y[validation], predicted, labels=[False, True], average="macro", zero_division=0)[2]))
        return values

    rows, detailed = [], {}

    def measure(name, params, cv, tuned=False):
        before = perf_counter()
        model = make_model(name, params).fit(x, y)
        fit_s = perf_counter() - before
        training_probability = model.predict_proba(x)[:, 1]
        before = perf_counter()
        scores = [probabilities(model, features) for _, _, features, _, _ in student]
        inference_s = perf_counter() - before
        route_rows, true, pred = [], [], []
        for (trip, arcs, features, labels, route), score in zip(student, scores):
            valid = labels >= 0
            true.extend((arcs[valid] == labels[valid, None]).ravel())
            pred.extend(score[valid].ravel())
            route_rows.append({"trip": trip.directory.name, **evaluate(trip, arcs, features, labels, route, score)})
        version = name + (" (tuned)" if tuned else "")
        row = {"model": version, "family": MODELS[name][0], "parameters": params,
               "cv_folds": cv, "cv_macro_f1": float(np.mean(cv)), "cv_std": float(np.std(cv)),
               "train_macro_f1": candidate_scores(y, training_probability)["macro_f1"],
               "training_s": fit_s, "inference_s": inference_s,
               **candidate_scores(np.asarray(true), np.asarray(pred)), **summarize(route_rows)}
        rows.append(row)
        detailed[version] = route_rows
        print(f"{version}: CV F1 {row['cv_macro_f1']:.4f}, student Top-1 {row['top1']:.4f}", flush=True)
        return model

    for name, (_, params) in MODELS.items():
        measure(name, params, cross_validation(name, params))
    selected = max(rows, key=lambda row: (row["cv_macro_f1"], row["top1"], -row["inference_s"]))["model"]
    base = MODELS[selected][1]
    search_start = perf_counter()
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=4))
    search_keys = {
        "Logistic Regression": ("C",), "Gaussian Naive Bayes": ("var_smoothing",),
        "Decision Tree": ("max_depth", "min_samples_leaf"), "Extra Trees": ("max_depth", "min_samples_leaf"),
        "HistGradientBoosting": ("max_leaf_nodes", "learning_rate", "min_samples_leaf", "l2_regularization"),
    }
    study.enqueue_trial({key: base[key] for key in search_keys[selected]})

    def objective(trial):
        cv = cross_validation(selected, suggest(selected, trial, base))
        trial.set_user_attr("cv_folds", cv)
        return float(np.mean(cv))

    study.optimize(objective, n_trials=trials)
    search_s = perf_counter() - search_start
    history = [{"trial": t.number, "state": t.state.name, "score": t.value, "parameters": t.params,
                "ranges": {key: str(value) for key, value in t.distributions.items()},
                "cv_folds": t.user_attrs["cv_folds"], "seconds": t.duration.total_seconds()} for t in study.trials]
    write_csv(out / "bayesian_trials.csv", history)
    model = measure(selected, {**base, **study.best_params}, study.best_trial.user_attrs["cv_folds"], True)
    joblib.dump(model, out / "model.joblib", compress=3)
    decoder_rows = []
    decode_s = 0.0
    for trip, arcs, features, labels, route in student:
        score = probabilities(model, features)
        before = perf_counter()
        decoded = evaluate(trip, arcs, features, labels, route, score, use_decoder=True)
        decode_s += perf_counter() - before
        decoder_rows.append({"trip": trip.directory.name, **decoded})
    plain = detailed[rows[-1]["model"]]
    baseline = [{"trip": trip.directory.name, **evaluate(trip, arcs, features, labels, route, -features[:, :, 0])}
                for trip, arcs, features, labels, route in student]
    write_csv(out / "model_comparison.csv", rows)
    write_csv(out / "route_comparison.csv", [
        {"method": method, **row} for method, data_rows in
        [("nearest", baseline), ("classifier", plain), ("hmm", decoder_rows)] for row in data_rows])
    result = {"seed": SEED, "features": FEATURES, "candidates": K, "gap_s": GAP,
              "bridge_limit_m": BRIDGE_LIMIT_M, "max_transition_speed_m_s": MAX_SPEED_M_S,
              "prep_s": prep_time, "search_s": search_s, "trials": trials, "best_trial": study.best_trial.number,
              "selected": selected, "models": rows, "nearest": summarize(baseline),
              "classifier": summarize(plain), "decoded": summarize(decoder_rows), "decoder_s": decode_s,
              "student_per_trip": detailed}
    (out / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Finished experiments. Tuned classifier + HMM Top-1: {result['decoded']['top1']:.4f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("Student folder"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/pathsnap"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--trials", type=int, default=12)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.data, args.cache, args.out, args.trials)
