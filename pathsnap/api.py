import argparse
import logging
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .data import FEATURES, GAP, align_reference, candidate_features, load_trip, reference
from .decode import decode, sequence_metrics

logger = logging.getLogger("uvicorn.error")


class PredictionRequest(BaseModel):
    trip_id: str


def create_app(trip_root, model_path, mode="development"):
    trip_root = Path(trip_root)
    allowed = {}
    for path in sorted(trip_root.iterdir()):
        if path.is_dir() and all((path / f"{path.name}.{ext}").is_file()
                                 for ext in ("track", "nodes", "arcs")):
            allowed[path.name] = path
    if len(allowed) != 10:
        raise ValueError(f"Expected 10 trip directories, found {len(allowed)} in {trip_root}")
    if mode == "evaluation" and any(not (p / f"{p.name}.route").is_file() for p in allowed.values()):
        raise ValueError("Final evaluation needs a server-side .route file for each trip")
    model = joblib.load(model_path)
    app = FastAPI(title="PathSnap")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type"])

    @app.get("/health")
    def health():
        return {"ok": True, "mode": mode, "trip_count": len(allowed)}

    @app.get("/trips")
    def trips():
        return {"mode": mode, "trip_ids": list(allowed)}

    @app.post("/predict")
    def predict(request: PredictionRequest):
        directory = allowed.get(request.trip_id)
        if directory is None:
            raise HTTPException(status_code=404, detail="Unknown trip ID")
        start = perf_counter()
        trip = load_trip(directory)
        arcs, features = candidate_features(trip)
        scores = model.predict_proba(features.reshape(-1, len(FEATURES)))[:, 1].reshape(arcs.shape)
        predicted, parts, starts = decode(trip, arcs, features, scores)
        inference_ms = round(1000 * (perf_counter() - start), 2)
        metrics = {"gps_observations": len(predicted), "route_parts": len(parts), "inference_ms": inference_ms,
                   "observation_accuracy": None, "alignment_coverage": None, "ordered_arc_pair_f1": None}
        eligible = np.diff(trip.gps[:, 2]) <= GAP
        metrics["route_continuity"] = float((~starts[1:] & eligible).sum() / eligible.sum()) if eligible.any() else None
        # Evaluation references enter only here, after the inference result is fixed.
        try:
            route = reference(trip)
        except FileNotFoundError:
            if mode == "evaluation":
                raise HTTPException(status_code=500, detail="Server-side scoring data unavailable") from None
        else:
            labels = align_reference(trip, route)
            valid = labels >= 0
            metrics["observation_accuracy"] = float((predicted[valid] == labels[valid]).mean()) if valid.any() else None
            metrics["alignment_coverage"] = float(valid.mean())
            metrics.update(sequence_metrics(parts, route))
        geometry = []
        for part in parts:
            nodes = trip.arcs[part]
            chain = np.r_[nodes[0, 0], nodes[:, 1]]
            geometry.append(trip.nodes[chain].tolist())
        gps_parts = np.split(trip.gps[:, :2], np.flatnonzero(~eligible) + 1)
        logger.info("trip=%s mode=%s gps=%d route_parts=%d inference_ms=%.2f top1=%s",
                    request.trip_id, mode, len(predicted), len(parts), inference_ms,
                    metrics["observation_accuracy"])
        return {"trip_id": request.trip_id, "geometry": {"gps": [x.tolist() for x in gps_parts],
                "predicted_route": geometry}, "metrics": metrics}

    return app


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--trips", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("results/model.joblib"))
    parser.add_argument("--mode", choices=["development", "evaluation"], default="development")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    options = parser.parse_args()
    uvicorn.run(create_app(options.trips, options.model, options.mode), host=options.host, port=options.port)
