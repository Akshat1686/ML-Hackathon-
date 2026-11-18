"""PathSnap FastAPI backend  (runs on LAPTOP 1).

Start:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
Endpoints:
    GET  /health            -> status + model info
    GET  /trips             -> the evaluation trip IDs available on this server
    POST /predict {trip_id} -> full pipeline on that trip + server-side scoring

The reference route stays on this machine: it is only used inside score_prediction() AFTER the
prediction has been made, and only aggregate metrics are returned (never the reference arcs).
Evaluation trips folder: data/evaluator_test/<trip_id>/  (override with env PATHSNAP_EVAL_DIR)
"""
import logging
import os
import sys
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from pathsnap.config import DATA_DIR  # noqa: E402
from pathsnap.data_io import list_trips, load_trip  # noqa: E402
from pathsnap.pipeline import PathSnapModel, route_geometry, score_prediction  # noqa: E402

EVAL_DIR = os.environ.get("PATHSNAP_EVAL_DIR", os.path.join(DATA_DIR, "evaluator_test"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | PATHSNAP-API | %(message)s")
log = logging.getLogger("pathsnap")

app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL = PathSnapModel()
log.info("Model loaded: %s | HMM %s | eval folder %s | trips %s",
         MODEL.meta.get("model"), MODEL.meta.get("hmm"), EVAL_DIR, list_trips(EVAL_DIR))


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL.meta.get("model"), "hmm": MODEL.meta.get("hmm"),
            "eval_dir": EVAL_DIR, "n_trips": len(list_trips(EVAL_DIR))}


@app.get("/trips")
def trips():
    ids = list_trips(EVAL_DIR)
    log.info("GET /trips -> %d trips", len(ids))
    return {"trips": ids}


@app.post("/predict")
def predict(req: PredictRequest):
    t0 = time.time()
    if req.trip_id not in list_trips(EVAL_DIR):
        log.warning("POST /predict unknown trip_id=%s", req.trip_id)
        raise HTTPException(404, f"unknown trip_id {req.trip_id}")
    log.info("POST /predict received trip_id=%s", req.trip_id)
    trip_pred = load_trip(EVAL_DIR, req.trip_id, with_route=False)   # prediction never sees the route
    pred = MODEL.predict(trip_pred)
    log.info("trip %s: %d GPS obs, %d candidates, %d HMM segments, %.2fs",
             req.trip_id, len(pred["obs"]), pred["n_candidates"], pred["hmm_stats"]["n_segments"], pred["timing_s"]["total"])

    metrics = None
    trip_ref = load_trip(EVAL_DIR, req.trip_id, with_route=True)       # server-side scoring only
    if trip_ref.route is not None:
        metrics = score_prediction(trip_ref, pred)
        log.info("trip %s: HMM top1=%.4f continuity=%.4f route_F1=%.4f | classifier-only top1=%.4f",
                 req.trip_id, metrics["hmm"]["top1_accuracy"], metrics["hmm"]["route_continuity"],
                 metrics["hmm"]["route_f1"], metrics["classifier_only"]["top1_accuracy"])

    gps = trip_pred.track[pred["obs"]]
    resp = {
        "trip_id": req.trip_id,
        "gps": [[float(lat), float(lon)] for lon, lat in gps[:, :2]],
        "predicted_route": route_geometry(trip_pred, pred["hmm_arc"]),
        "classifier_only_route": route_geometry(trip_pred, pred["clf_arc"]),
        "predicted_arc_per_observation": [int(a) for a in pred["hmm_arc"]],
        "metrics": metrics,
        "hmm_stats": pred["hmm_stats"],
        "timing_s": {**pred["timing_s"], "request_total": time.time() - t0},
    }
    log.info("POST /predict done trip_id=%s in %.2fs", req.trip_id, time.time() - t0)
    return resp
