"""Small, dependency-free Bayesian optimisation (Gaussian-process surrogate + Expected Improvement).

Why our own instead of a library: it keeps the search fully transparent (every trial, the
surrogate's predicted mean/std and the EI value are logged) and needs only scikit-learn.

Algorithm
  1. Map every hyperparameter to [0,1] (log scale where the parameter spans orders of magnitude).
  2. Evaluate n_init random points (Latin-hypercube-like stratified sampling).
  3. Repeat: fit GP (Matern 5/2 + white noise) on (x, objective) -> pick the point among
     n_candidates random proposals that maximises Expected Improvement:
         EI(x) = (mu - f_best - xi) * Phi(z) + sigma * phi(z),  z = (mu - f_best - xi) / sigma
     -> evaluate the true objective there -> add to data.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


class Space:
    def __init__(self, spec: dict):
        # spec: name -> (low, high, "log"|"lin", "int"|"float")
        self.spec = spec
        self.names = list(spec)

    def decode(self, u):
        out = {}
        for k, name in enumerate(self.names):
            lo, hi, scale, kind = self.spec[name]
            if scale == "log":
                v = math.exp(math.log(lo) + u[k] * (math.log(hi) - math.log(lo)))
            else:
                v = lo + u[k] * (hi - lo)
            out[name] = int(round(v)) if kind == "int" else float(v)
        return out


def bayes_opt(objective, space: Space, n_init=6, n_iter=19, n_candidates=4000, xi=0.002, seed=42, log=print):
    rng = np.random.default_rng(seed)
    d = len(space.names)
    # stratified initial design
    U = (rng.permuted(np.tile(np.arange(n_init), (d, 1)), axis=1).T + rng.random((n_init, d))) / n_init
    X, Y, trials = [], [], []

    def run(u, phase, mu=None, sd=None, ei=None):
        params = space.decode(u)
        t0 = time.time()
        y = objective(params)
        rec = dict(trial=len(trials), phase=phase, objective=y, seconds=time.time() - t0, gp_mean=mu, gp_std=sd, ei=ei, **params)
        trials.append(rec)
        X.append(u)
        Y.append(y)
        log(f"trial {rec['trial']:2d} [{phase}] obj={y:.5f} best={max(Y):.5f} {params}")

    for u in U:
        run(u, "init")
    kernel = ConstantKernel(1.0) * Matern(length_scale=np.full(d, 0.3), nu=2.5) + WhiteKernel(1e-4)
    for _ in range(n_iter):
        Xa, Ya = np.array(X), np.array(Y)
        mu_y, sd_y = Ya.mean(), Ya.std() + 1e-9
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False, n_restarts_optimizer=3, random_state=seed)
        gp.fit(Xa, (Ya - mu_y) / sd_y)
        C = rng.random((n_candidates, d))
        # also probe around the current best point (local refinement)
        best_u = Xa[Ya.argmax()]
        C[: n_candidates // 4] = np.clip(best_u + rng.normal(0, 0.08, (n_candidates // 4, d)), 0, 1)
        m, s = gp.predict(C, return_std=True)
        fbest = (Ya.max() - mu_y) / sd_y
        z = (m - fbest - xi) / np.maximum(s, 1e-9)
        ei = (m - fbest - xi) * norm.cdf(z) + s * norm.pdf(z)
        k = int(ei.argmax())
        run(C[k], "bo", mu=float(m[k] * sd_y + mu_y), sd=float(s[k] * sd_y), ei=float(ei[k]))
    best = max(trials, key=lambda r: r["objective"])
    return best, trials
