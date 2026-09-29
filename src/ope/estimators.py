"""Core OPE estimators implemented from scratch (finite horizon, episodes).

Notation for episode i, times t = 0..H-1, discount gamma:

    g_i              = sum_t gamma^t r_t^i
    rho_{0:t}^i      = prod_{h=0..t} pi(a_h^i|s_h^i) / mu(a_h^i|s_h^i)
    rho_{-1} = 1,    Qhat_H(.) = 0

1. Per-decision IS (PDIS):
    J_PDIS = (1/n) sum_i sum_t gamma^t rho_{0:t}^i r_t^i

2. Trajectory-level self-normalized IS (SNIS):
    J_SNIS = sum_i rho_{0:H-1}^i g_i / sum_i rho_{0:H-1}^i

3. Sequential / weighted doubly robust (DR):
    J_DR = (1/n) sum_i [ Vhat_0(s_0^i)
        + sum_t gamma^t rho_{0:t}^i * delta_t^i ],
    delta_t^i = r_t^i + gamma Qhat_{t+1}(s_{t+1}^i) - Qhat_t(s_t^i, a_t^i).
    Here Vhat_t(s) = sum_a pi(a|s) Qhat_t(s,a) under the *target* policy.
"""

from __future__ import annotations

import numpy as np

_EPS_LOG = 1e-300


def signed_weighted_sum(values: np.ndarray, log_w: np.ndarray) -> np.ndarray:
    """Column-wise stable sum of w*v; values may be negative, log_w may be -inf.

    Returns a per-column vector (sum over episodes). Positive/negative parts
    combined by a single shared log-sum-exp scale. +inf weight times a nonzero
    value yields NaN (off-support: the contribution is not finite/identified).
    """
    values = np.asarray(values, float)
    log_w = np.asarray(log_w, float)
    finite = np.isfinite(log_w)
    v = np.where(finite, values, 0.0)
    nz = finite & (v != 0.0)
    log_abs = np.where(nz, np.log(np.abs(v) + _EPS_LOG), -np.inf)
    lp = np.where(nz & (v > 0), log_w + log_abs, -np.inf)
    lm = np.where(nz & (v < 0), log_w + log_abs, -np.inf)
    stacked = np.concatenate([lp, lm], axis=0)
    m = np.max(stacked, axis=0)
    m_eff = np.where(np.isfinite(m), m, 0.0)
    sp = np.sum(np.where(np.isneginf(lp), 0.0, np.exp(np.minimum(lp - m_eff, 0.0))), axis=0)
    sm = np.sum(np.where(np.isneginf(lm), 0.0, np.exp(np.minimum(lm - m_eff, 0.0))), axis=0)
    out = np.exp(m_eff) * (sp - sm)
    pos_hit = np.any(np.isposinf(log_w) & (values != 0.0), axis=0)
    out[pos_hit] = np.nan
    out[~np.isfinite(m)] = 0.0
    return out


def discounted_returns(rewards: np.ndarray, discount: float) -> np.ndarray:
    return rewards @ (discount ** np.arange(rewards.shape[1]))


def pdis(rewards: np.ndarray, log_rho_prefix: np.ndarray, discount: float) -> float:
    gammas = discount ** np.arange(rewards.shape[1])
    return float(np.sum(signed_weighted_sum(rewards * gammas, log_rho_prefix)) / rewards.shape[0])


def snis(rewards: np.ndarray, log_rho_traj: np.ndarray, discount: float):
    """Returns (estimate, mean_log_weight). NaN estimate if all weights zero."""
    g = discounted_returns(rewards, discount)
    log_w = log_rho_traj[:, -1]
    m = np.max(log_w)
    n = log_w.shape[0]
    if not np.isfinite(m):
        return float("nan"), float(m)
    w = np.exp(log_w - m)
    denom = np.sum(w)
    if denom == 0.0:
        return float("nan"), float(m - np.log(n))
    return float(np.sum(w * g) / denom), float(m - np.log(n))


def dr(rewards, log_rho_prefix, q_sa, q_next_s, v0, discount: float) -> float:
    """Sequential DR with pre-evaluated nuisance arrays."""
    gammas = discount ** np.arange(rewards.shape[1])
    bellman = (rewards + discount * q_next_s - q_sa) * gammas
    correction = np.sum(signed_weighted_sum(bellman, log_rho_prefix)) / rewards.shape[0]
    return float(np.mean(v0) + correction)


def effective_sample_size(log_rho_prefix: np.ndarray) -> dict:
    """ESS_t = (sum_i w_it)^2 / sum_i w_it^2 per time step, in log space.

    Returns normalized ESS/n in [0,1] for comparison with #episodes. Small ESS
    signals variance/bias trouble; it is not a coverage or correctness proof.
    """
    n, H = log_rho_prefix.shape
    m_all = np.max(log_rho_prefix, axis=0)
    ess_t = np.zeros(H)
    for t in range(H):
        m = m_all[t]
        if np.isfinite(m):
            w = np.exp(log_rho_prefix[:, t] - m)
            denom = np.sum(w**2)
            if denom > 0:
                ess_t[t] = (w.sum() ** 2) / denom
    frac = ess_t / n
    return {
        "ess": float(ess_t[-1]),
        "ess_per_t": ess_t,
        "ess_frac": frac,
        "min_ess": float(ess_t.min()),
        "min_ess_frac": float(frac.min()),
    }
