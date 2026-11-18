"""Reproduce every result in the report, in order.  Usage:  python run_all.py
(Total ~40-60 min on a 2-core laptop; each step can also be run on its own.)"""
import subprocess
import sys

STEPS = ["scripts/01_eda.py", "scripts/02_build_features.py", "scripts/03_train_compare.py",
         "scripts/04_bayes_opt.py", "scripts/05_hmm_eval.py", "scripts/06_maps.py",
         "scripts/07_plots.py", "scripts/08_build_report.py"]
for s in STEPS:
    print(f"\n===== {s} =====", flush=True)
    subprocess.run([sys.executable, s], check=True)
