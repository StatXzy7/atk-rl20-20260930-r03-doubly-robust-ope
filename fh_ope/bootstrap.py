"""Episode-level nonparametric bootstrap.

The resampling unit is the *whole episode* (estimates are i.i.d. at the
episode level), never individual transitions.  For DR the nuisance model is
refit inside every bootstrap draw with out-of-fold cross-fitting, so the
intervals reflect nuisance estimation noise as well as sampling noise.

These intervals are empirical (frequentist, plug-in) uncertainty statements.
With estimated/tilted weights or misspecified models they have no finite-sample
exact coverage guarantee; they are labelled accordingly in the report.
"""
from __future__ import annotations

import numpy as np

from . import estimators as E
from .models import _subset, TabularQModel


def percentile_ci(boot_estimates: np.ndarray, alpha: float = 0.1):
    lo = float(np.quantile(boot_estimates, alpha / 2))
    hi = float(np.quantile(boot_estimates, 1 - alpha / 2))
    return lo, hi


def _resample(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, n)


def bootstrap_is(ds, gamma: float, kind: str, clip, B: int, seed: int,
                 alpha: float = 0.1) -> dict:
    rng = np.random.default_rng(seed)
    n = ds.n_episodes
    if kind == "pdis":
        parts = E.pdis_parts(ds, gamma, clip)
        base = parts["estimate"]
        per = parts["per_episode"]
        draws = np.array([per[_resample(n, rng)].mean() for _ in range(B)])
    else:
        parts = E.snis_parts(ds, gamma, clip)
        base = parts["estimate"]
        draws = np.empty(B)
        for b in range(B):
            idx = _resample(n, rng)
            draws[b] = E.combine_snis(parts, idx)
    lo, hi = percentile_ci(draws, alpha)
    return dict(estimate=base, lo=lo, hi=hi, draws=draws)


def bootstrap_dr(ds, mdp, clip, B: int, seed: int, alpha: float = 0.1,
                 K: int = 5, action_blind: bool = False,
                 leakage: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    n = ds.n_episodes
    draws = np.empty(B)
    for b in range(B):
        idx = _resample(n, rng)
        sub = _subset(ds, idx)
        if leakage:
            model = TabularQModel(mdp.n_states, mdp.n_actions, mdp.H, mdp.gamma,
                                  action_blind=action_blind).fit(sub, mdp.target)
            out = E.dr_parts(sub, mdp, model, clip=clip)
        else:
            labels = sub.folds(K, seed=int(rng.integers(1 << 30)))
            per = np.full(sub.n_episodes, np.nan)
            for k in np.unique(labels):
                eval_pos = np.where(labels == k)[0]
                train_pos = np.where(labels != k)[0]
                train = _subset(sub, train_pos)
                model = TabularQModel(
                    mdp.n_states, mdp.n_actions, mdp.H, mdp.gamma,
                    action_blind=action_blind).fit(train, mdp.target)
                out = E.dr_parts(_subset(sub, eval_pos), mdp, model, clip=clip)
                per[eval_pos] = out["per_episode"]
            out = dict(estimate=float(per.mean()), per_episode=per)
        draws[b] = out["estimate"]
    # point estimate supplied externally (same pipeline on the original data)
    lo, hi = percentile_ci(draws, alpha)
    return dict(lo=lo, hi=hi, draws=draws)
