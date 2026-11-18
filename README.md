# PathSnap — GPS map-matching & route reconstruction

Noisy GPS observations go through a candidate classifier and then an HMM/Viterbi decoder, which gives the road segment the vehicle was on for every GPS observation. The system is served by a FastAPI backend (Laptop 1) and a Streamlit + Folium UI (Laptop 2).

```
raw trip files ─► clean + local metric projection ─► candidate arcs (50 m radius, ≥3)
   ─► 20 features per (observation, arc) ─► HistGradientBoosting (Bayesian-optimised)
   ─► HMM / Viterbi on the directed road network ─► arc per observation ─► API ─► UI map
```

## Repository layout
| Folder / file | What it contains |
|---|---|
| `data/train`, `data/student_test` | Raw trips as provided (80 / 10) |
| `data/evaluator_test/` | **Put the 10 evaluator trip folders here on Laptop 1** (never used in development) |
| `src/pathsnap/data_io.py` | Reads and cleans `.track/.nodes/.arcs/.route` |
| `src/pathsnap/geo.py` | Local tangent-plane projection, point-to-segment distance, bearings |
| `src/pathsnap/candidates.py` | Spatial index, candidate generation, DP label alignment, feature engineering |
| `src/pathsnap/models.py` | The 8 compared models and the tunable final model |
| `src/pathsnap/bayesopt.py` | Gaussian-process Bayesian optimisation (Expected Improvement) |
| `src/pathsnap/hmm.py` | HMM states, emissions, transitions (Dijkstra), Viterbi, gap and break handling |
| `src/pathsnap/evaluation.py` | Top-1, macro F1, log loss, candidate miss rate, continuity, route F1 |
| `src/pathsnap/pipeline.py` | End-to-end inference + server-side scoring (used by the API) |
| `scripts/01_eda.py` … `08_build_report.py` | Every experiment, in order |
| `api/main.py` | FastAPI backend (Laptop 1) |
| `ui/app.py` | Streamlit evaluator UI (Laptop 2) |
| `models/` | `final_model.joblib`, `inference_config.json`, all compared models |
| `results/` | CSV logs (EDA, CV folds, comparison, BO trials, HMM), `figures/`, `maps/` (Folium HTML) |
| `reports/PathSnap_Technical_Report.pdf` | Technical report (generated from `results/`) |

## Install (both laptops)
```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
```
Use Python 3.10–3.13 (3.12 is recommended).

## If the saved model will not load
Model pickles depend on the scikit-learn version. If the API fails while loading `final_model.joblib`, rebuild it (about 2 min):
```bash
python scripts/02_build_features.py
python scripts/09_refit_final.py
```

## Reproduce all results
```bash
python run_all.py            # or run scripts/01 … 08 one by one
```
| Step | Script | Output |
|---|---|---|
| EDA | `01_eda.py` | `results/eda/inventory.csv`, `data_quality.md`, EDA figures |
| Candidates + features | `02_build_features.py` | `features/*.pkl`, `results/candidate_stats.csv` |
| 8-model comparison | `03_train_compare.py` | `results/model_comparison.csv`, `cv_folds.csv`, confusion matrices, `models/*.joblib` |
| Bayesian optimisation | `04_bayes_opt.py` | `results/bayes_opt_trials.csv`, `best_params.json`, `tuned_vs_untuned.csv`, `models/final_model.joblib` |
| HMM / Viterbi | `05_hmm_eval.py` | `results/hmm_grid.csv`, `hmm_comparison.csv`, `models/inference_config.json` |
| Folium maps | `06_maps.py` | `results/maps/trip_<id>.html` |
| Plots | `07_plots.py` | comparison figures |
| Report | `08_build_report.py` | `reports/PathSnap_Technical_Report.pdf` |

Random seed: 42 everywhere. Cross-validation is GroupKFold by trip.

## Deployment (two laptops on the same Wi-Fi/hotspot)

**Laptop 1 — model + API**
1. Copy the 10 evaluator trip folders into `data/evaluator_test/` (each as `<trip_id>/<trip_id>.track|.nodes|.arcs|.route`).
2. Find the IP: `ipconfig` → IPv4 address (e.g. `192.168.1.23`).
3. Start the API (keep this window visible, because it shows the request logs):
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
4. If Windows Firewall asks, click **Allow access** for Python on private networks.
5. Check it from a browser: `http://127.0.0.1:8000/health` and `http://127.0.0.1:8000/docs`.

**Laptop 2 — UI**
```bash
streamlit run ui/app.py
```
In the sidebar, set **API URL** to `http://<Laptop-1-IP>:8000`, pick a trip, then click **Reconstruct route**.

**API contract**
- `GET /trips` → `{"trips": ["...", ...]}`
- `POST /predict` with `{"trip_id": "00000012"}` → GPS points, the predicted route geometry, the arc per observation, server-side metrics (HMM vs classifier-only), HMM stats and timings.

Command-line test from Laptop 2:
```bash
curl -X POST http://<IP>:8000/predict -H "Content-Type: application/json" -d "{\"trip_id\": \"00000012\"}"
```

## Leakage rules we follow
- The prediction path loads trips with `with_route=False`. The reference route is read only by `score_prediction` after prediction, and only aggregate metrics leave the server.
- `data/evaluator_test` is read only by the API. No training, tuning or decoder setting uses it.
- Decoder parameters are tuned on 10 held-out *training* trips.
