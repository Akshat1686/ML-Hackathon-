"""Step 2 - Candidate generation + feature engineering + labels.

train        : every 3rd GPS observation (consecutive 1 Hz points are near-duplicates;
               subsampling keeps training tractable and reduces autocorrelation)
student_test : every observation (used for model comparison / decoder evaluation)

Output: features/<partition>.pkl and results/candidate_stats.csv
"""
import os
import sys
import time
from multiprocessing import Pool

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap.config import DATA_DIR, FEAT_DIR, RESULTS_DIR  # noqa: E402
from pathsnap.data_io import list_trips, load_trip  # noqa: E402
from pathsnap.candidates import DEFAULT_RADIUS, align_route_labels, build_candidate_table  # noqa: E402

STEP = {"train": 3, "student_test": 1}


def work(args):
    part, tid = args
    tr = load_trip(os.path.join(DATA_DIR, part), tid)
    lab, valid = align_route_labels(tr, return_valid=True)
    df = build_candidate_table(tr, radius=DEFAULT_RADIUS, labels=lab, label_valid=valid, step=STEP[part])
    df["partition"] = part
    return df


if __name__ == "__main__":
    os.makedirs(FEAT_DIR, exist_ok=True)
    stats = []
    for part in ["train", "student_test"]:
        t0 = time.time()
        jobs = [(part, t) for t in list_trips(os.path.join(DATA_DIR, part))]
        with Pool(min(8, os.cpu_count() or 2)) as pool:
            dfs = pool.map(work, jobs)
        df = pd.concat(dfs, ignore_index=True)
        df.to_pickle(os.path.join(FEAT_DIR, f"{part}.pkl"))
        v = df[df.label_valid]
        per_obs = v.groupby(["trip_id", "obs"]).agg(hit=("y", "max"), n=("y", "size"))
        stats.append(dict(partition=part, rows=len(df), observations=df.groupby(["trip_id", "obs"]).ngroups,
                          valid_observations=len(per_obs), cands_per_obs=per_obs.n.mean(),
                          candidate_miss_rate=1 - per_obs.hit.mean(), positive_rate=v.y.mean(),
                          build_seconds=time.time() - t0))
        print(stats[-1])
    pd.DataFrame(stats).to_csv(os.path.join(RESULTS_DIR, "candidate_stats.csv"), index=False)
