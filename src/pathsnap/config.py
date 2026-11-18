import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.environ.get("PATHSNAP_DATA", os.path.join(ROOT, "data"))
RESULTS_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
MAP_DIR = os.path.join(RESULTS_DIR, "maps")
MODEL_DIR = os.path.join(ROOT, "models")
FEAT_DIR = os.path.join(ROOT, "features")
SEED = 42
