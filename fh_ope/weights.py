"""Cumulative importance weights in log space, clipping and ESS diagnostics.

Definitions (episode i, stage t)
--------------------------------
step_ratio_{i,t} = pi(a_t|s_t) / mu(a_t|s_t)
cum_weight_{i,t} = prod_{k<=t} step_ratio_{i,k}

We carry cumulative weights as log values:
    ell_{i,t} = sum_{k<=t} (log pi - log mu)
with the conventions
    pi>0, mu=0 -> +inf  (an action required by pi is off-support; fatal)
    pi=0, mu=0 -> 0     (0/0: an action neither policy takes is irrelevant)
    pi=0, mu>0 -> -inf  (weight exactly 0 once the trajectory diverges)

Exponentiation only happens inside estimators.  ESS is computed on the set of
episodes with nonzero weight in the log domain (log-sum-exp); episodes with
exactly zero weight never enter it, and their share is reported separately --
an ESS computed over a handful of episodes out of thousands is the clearest
weight-degeneracy signal.
"""
from __future__ import annotations

import numpy as np


def _log_prob(p: np.ndarray) -> np.ndarray:
    out = np.full_like(p, -np.inf, dtype=float)
    pos = p > 0
    out[pos] = np.log(p[pos])
    return out


def step_log_ratios(ds) -> np.ndarray:
    valid = ds.mask
    lp_t = _log_prob(ds.p_target)
    lp_b = _log_prob(ds.p_behavior)
    ell = np.zeros_like(lp_t)
    has_b = ds.p_behavior > 0
    # pi>0, mu>0 : log ratio ; pi=0, mu>0 : -inf ; mu=0 handled below
    pb_pos = ds.p_behavior > 0
    pt_pos = ds.p_target > 0
    both_pos = pb_pos & pt_pos
    ell[both_pos] = lp_t[both_pos] - lp_b[both_pos]
    ell[pb_pos & ~pt_pos] = -np.inf
    ell[~pb_pos & pt_pos] = np.inf
    # both zero stays 0
    return np.where(valid, ell, 0.0)


def cumulative_log_weights(ds) -> np.ndarray:
    return np.cumsum(step_log_ratios(ds), axis=1)


def weights_from_log(cum_log: np.ndarray, clip: float | None = None) -> np.ndarray:
    cl = np.minimum(cum_log, np.log(clip)) if clip is not None else cum_log
    with np.errstate(over="ignore"):
        return np.exp(cl)


def ess_from_log(log_w: np.ndarray) -> float:
    """Kish ESS over finite log weights only; 0 if none carry nonzero weight."""
    x = np.asarray(log_w, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    m1 = x.max()
    s1 = np.exp(x - m1).sum()
    m2 = (2 * x).max()
    s2 = np.exp(2 * x - m2).sum()
    log_ess = 2 * (m1 + np.log(s1)) - (m2 + np.log(s2))
    with np.errstate(over="ignore"):
        return float(np.exp(min(log_ess, 700.0)))


def ess(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    nz = w[w > 0]
    s1, s2 = nz.sum(), (nz ** 2).sum()
    return float(s1 * s1 / s2) if s2 > 0 else 0.0


def stable_self_normalized(numer: np.ndarray, w: np.ndarray) -> float:
    total = w.sum()
    return float((numer * w).sum() / total) if total > 0 else float("nan")


def weight_diagnostics(ds, clip: float | None = None) -> dict:
    cum_ell = cumulative_log_weights(ds)
    log_last = cum_ell[:, -1]
    n = ds.n_episodes
    n_zero = int((log_last == -np.inf).sum())
    n_posinf = int(np.isposinf(log_last).sum())
    w_used = weights_from_log(cum_ell, clip=clip)[:, -1]
    finite_max = cum_ell[np.isfinite(cum_ell)].max() if np.isfinite(cum_ell).any() else -np.inf
    # share of raw weight held by top 10% of *nonzero* episodes
    wraw_nz = weights_from_log(cum_ell)[:, -1]
    wraw_nz = wraw_nz[wraw_nz > 0]
    if wraw_nz.size:
        top = max(1, wraw_nz.size // 10)
        sw = np.sort(wraw_nz)
        top_share = float(sw[-top:].sum() / sw.sum())
    else:
        top_share = float("nan")
    return dict(
        n_episodes=n,
        n_zero_weight_episodes=n_zero,
        frac_zero_weight=n_zero / n,
        n_posinf_weight_episodes=n_posinf,
        max_finite_log_weight=float(finite_max) if np.isfinite(finite_max) else None,
        ess_nonzero=ess_from_log(log_last),
        ess_fraction_of_nonzero=ess_from_log(log_last) / max(1, n - n_zero),
        ess_fraction_of_all=ess_from_log(log_last) / n,
        ess_after_clip=ess(w_used[w_used > 0]),
        frac_weight_top10_nz=top_share,
        n_zero_behavior_cells=int((ds.p_behavior[ds.mask] == 0).sum()),
        clipped=clip is not None,
    )
