"""Episode-level nonparametric bootstrap.

Resample whole EPISODES with replacement (never time steps or segments), rerun
the entire pipeline (weight recomputation + K-fold cross-fitted nuisance + the
estimator). Percentile intervals summarize the empirical sampling distribution
of the estimator. They are empirical uncertainty statements: they do NOT carry
a finite-sample coverage guarantee, and for biased/heavy-tailed estimators
(SNIS/PDIS under tiny behavior probabilities) coverage can be far below nominal.
"""

from __future__ import annotations

import numpy as np

from .crossfit import compute_weights, crossfit_q
from .data import EpisodeBatch
from .estimators import dr, pdis, snis
from .mdp import FiniteHorizonMDP


def _point_on_batch(mdp, batch, folds, clip, q_mode):
    _, _, log_prefix_c = compute_weights(mdp, batch, clip)
    q_sa, q_next, v0, _ = crossfit_q(mdp, batch, folds, mode=q_mode)
    p = pdis(batch.rewards, log_prefix_c, mdp.discount)
    s, _ = snis(batch.rewards, log_prefix_c, mdp.discount)
    d = dr(batch.rewards, log_prefix_c, q_sa, q_next, v0, mdp.discount)
    return p, s, d


def bootstrap_estimates(
    mdp: FiniteHorizonMDP,
    data: EpisodeBatch,
    n_boot: int,
    rng: np.random.Generator,
    k_folds: int = 4,
    clip: float | None = None,
    q_mode: str = "empirical",
) -> dict:
    n = data.n
    samples = {"pdis": [], "snis": [], "dr": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot = data.take(idx)
        folds = np.arange(n) % k_folds
        p, s, d = _point_on_batch(mdp, boot, folds, clip, q_mode)
        samples["pdis"].append(p)
        samples["snis"].append(s)
        samples["dr"].append(d)
    out = {}
    for name, arr in samples.items():
        arr = np.asarray(arr, float)
        finite = arr[np.isfinite(arr)]
        if finite.size < 2:
            out[name] = {"se": np.nan, "lo": np.nan, "hi": np.nan, "finite_frac": finite.size / arr.size}
            continue
        lo, hi = np.percentile(finite, [2.5, 97.5])
        out[name] = {
            "se": float(np.std(finite, ddof=1)),
            "lo": float(lo),
            "hi": float(hi),
            "finite_frac": float(finite.size / arr.size),
        }
    return out
