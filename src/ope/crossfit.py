"""Episode-level K-fold cross-fitting of nuisance models and OPE estimation.

Splitting unit is the whole episode: for each evaluation fold k the Q/V model
is trained ONLY on episodes from the other folds. Segments of one episode are
never divided between a training fold and an evaluation fold, so no trajectory
contributes its own transitions when predicting itself (no train/eval leak).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data import EpisodeBatch
from .estimators import dr, effective_sample_size, pdis, snis
from .mdp import FiniteHorizonMDP
from .nuisance import TabularQ
from .weights import (
    clip_log_ratio,
    cumulative_log_ratio,
    log_policy_ratio,
    population_support_check,
    support_violations,
)


def episode_kfold(n: int, k: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Assign each of n episodes a fold id in 0..k-1 (whole-episode split)."""
    if rng is None:
        order = np.arange(n)
    else:
        order = rng.permutation(n)
    folds = np.empty(n, dtype=int)
    folds[order] = np.arange(n) % k
    return folds


@dataclass
class OPEEvaluation:
    estimates: dict
    clip: float | None
    support_violation: bool
    population_missing: list
    frac_violation_episodes: float
    ess: dict
    unseen_cells: list
    n_episodes: int
    note: str = ""


def crossfit_q(
    mdp: FiniteHorizonMDP,
    data: EpisodeBatch,
    folds: np.ndarray,
    mode: str = "empirical",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Fit Q on the K-1 other folds for every fold; return out-of-fold arrays."""
    k = int(folds.max()) + 1
    q_sa = np.empty((data.n, mdp.horizon))
    q_next = np.empty((data.n, mdp.horizon))
    v0 = np.empty(data.n)
    unseen = np.zeros(mdp.horizon, dtype=int)
    for f in range(k):
        train_idx = np.where(folds != f)[0]
        eval_idx = np.where(folds == f)[0]
        model = TabularQ(mdp, mode=mode).fit(data.take(train_idx))
        a, b, c = model.arrays_for(data.take(eval_idx))
        q_sa[eval_idx] = a
        q_next[eval_idx] = b
        v0[eval_idx] = c
        if model.unseen_cells:
            unseen += np.asarray(model.unseen_cells)
    return q_sa, q_next, v0, unseen.tolist()


def in_sample_q(
    mdp: FiniteHorizonMDP,
    data: EpisodeBatch,
    mode: str = "empirical",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """LEAKAGE counterexample: one model trained on ALL episodes and evaluated
    on those same episodes. Each trajectory's own transitions predict itself."""
    model = TabularQ(mdp, mode=mode).fit(data)
    q_sa, q_next, v0 = model.arrays_for(data)
    return q_sa, q_next, v0, list(model.unseen_cells)


def compute_weights(
    mdp: FiniteHorizonMDP,
    data: EpisodeBatch,
    clip: float | None = None,
):
    log_step = log_policy_ratio(mdp, data)
    log_prefix = cumulative_log_ratio(log_step)
    log_prefix_clipped = clip_log_ratio(log_prefix, clip)
    return log_step, log_prefix, log_prefix_clipped


def evaluate_all(
    mdp: FiniteHorizonMDP,
    data: EpisodeBatch,
    folds: np.ndarray,
    clip: float | None = None,
    q_mode: str = "empirical",
    crossfit: bool = True,
) -> OPEEvaluation:
    _, log_prefix, log_prefix_c = compute_weights(mdp, data, clip)
    if crossfit:
        q_sa, q_next, v0, unseen = crossfit_q(mdp, data, folds, mode=q_mode)
    else:
        q_sa, q_next, v0, unseen = in_sample_q(mdp, data, mode=q_mode)

    missing = population_support_check(mdp)
    viol = support_violations(mdp, data)
    identified = (not missing) and (not np.any(viol))

    est: dict = {}
    if identified:
        est["pdis"] = pdis(data.rewards, log_prefix_c, mdp.discount)
        est["snis"], mean_log_w = snis(data.rewards, log_prefix_c, mdp.discount)
        est["dr"] = dr(data.rewards, log_prefix_c, q_sa, q_next, v0, mdp.discount)
        est["plugin_v0"] = float(np.mean(v0))
        est["mean_log_weight"] = mean_log_w
        note = ""
    else:
        for key in ("pdis", "snis", "dr", "plugin_v0"):
            est[key] = np.nan
        est["mean_log_weight"] = np.nan
        note = (
            "OFF-SUPPORT: target places positive probability on (state,action) "
            "pairs with zero behavior probability. The target value is NOT "
            "identified by the logged data; no epsilon is added and no "
            "interval is reported. Any point estimate would rely on an "
            "untestable model extrapolation."
        )
    ess = effective_sample_size(log_prefix_c)
    return OPEEvaluation(
        estimates=est,
        clip=clip,
        support_violation=not identified,
        population_missing=missing,
        frac_violation_episodes=float(np.mean(viol)),
        ess=ess,
        unseen_cells=unseen,
        n_episodes=data.n,
        note=note,
    )
