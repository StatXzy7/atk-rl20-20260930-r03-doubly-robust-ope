"""Importance ratios in log space: numerically stable even when products underflow.

For episode i and prefix 0..t define

    rho_{0:t}^i = prod_{h<=t} pi(a_h^i | s_h^i) / mu(a_h^i | s_h^i).

We store log rho and only exponentiate when strictly necessary (DR/SNIS use
log-sum-exp). Behavior probability mu is the *recorded logged* probability;
callers must not silently epsilon-patch zeros.
"""

from __future__ import annotations

import numpy as np

from .data import EpisodeBatch
from .mdp import FiniteHorizonMDP

NEG_INF = -np.inf


def safe_log(x: np.ndarray) -> np.ndarray:
    out = np.full_like(np.asarray(x, float), NEG_INF, dtype=float)
    np.log(np.asarray(x, float), out=out, where=np.asarray(x) > 0.0)
    return out


def log_policy_ratio(mdp: FiniteHorizonMDP, data: EpisodeBatch, use_recorded: bool = True) -> np.ndarray:
    """Per-step log pi(a|s) - log mu(a|s), shape (n, H).

    Behavior log-prob comes from the recorded value log_mu (logged at data
    collection time); zero support yields +inf ratio which we keep explicit.
    """
    n, H = data.n, data.H
    t_idx = np.arange(H)[None, :]
    log_pi = np.stack(
        [mdp.target.log_probs(h) for h in range(H)], axis=0
    )[t_idx, data.states, data.actions]
    if use_recorded:
        log_mu = data.log_mu
    else:
        log_mu_t = np.stack(
            [safe_log(mdp.behavior.probs[h]) for h in range(H)], axis=0
        )
        log_mu = log_mu_t[t_idx, data.states, data.actions]
    return log_pi - log_mu


def cumulative_log_ratio(log_rho_step: np.ndarray) -> np.ndarray:
    """log rho_{0:t} with explicit infinities, computed in O(H) vector ops.

    -inf at a step (target prob 0) fixes the prefix product to exactly 0 from
    that step onward (prefix log = -inf); +inf before that yields +inf. We
    never form inf + (-inf) (which would be NaN): the zero state dominates.
    """
    x = log_rho_step
    neg = np.isneginf(x)
    pos = np.isposinf(x)
    # state: 0 finite, +1 +inf seen, -1 zero seen. first occurrence time masks.
    zero_seen = np.cumsum(neg, axis=1) > 0
    pos_seen_before_zero = np.cumsum(np.where(zero_seen, 0, pos), axis=1) > 0
    finite_part = np.where(neg | pos, 0.0, x)
    acc = np.cumsum(np.where(zero_seen, 0.0, finite_part), axis=1)
    out = np.where(zero_seen, NEG_INF, np.where(pos_seen_before_zero, np.inf, acc))
    return out


def clip_log_ratio(log_rho_prefix: np.ndarray, clip: float | None) -> np.ndarray:
    """Clip prefix ratio to [0, clip] in probability space => log clip.

    Clipping is an explicit, documented bias source: it is not "robustness".
    Lower clipping at 0 only matters when a prefix log-ratio is negative;
    ratios cannot be negative, so only the upper bound binds in log space.
    """
    if clip is None:
        return log_rho_prefix
    clip = float(clip)
    if clip <= 0:
        raise ValueError("clip must be positive")
    return np.minimum(log_rho_prefix, np.log(clip))


def support_violations(mdp: FiniteHorizonMDP, data: EpisodeBatch) -> np.ndarray:
    """Boolean mask (n,) of episodes with pi(a|s)>0 but logged mu(a|s)==0.

    These are off-support actions: the target value is not identified by the
    logged distribution, and epsilon-patching the zero must not hide that.
    """
    t_idx = np.arange(mdp.horizon)[None, :]
    pi_t = np.stack([mdp.target.probs[h] for h in range(mdp.horizon)], axis=0)
    pi = pi_t[t_idx, data.states, data.actions]
    mu = np.exp(data.log_mu)
    return np.any((pi > 0.0) & (mu <= 0.0), axis=1)


def population_support_check(mdp: FiniteHorizonMDP) -> list[tuple[int, int, int]]:
    """Return list of (h, s, a) where target puts mass but behavior puts none,
    restricted to states reachable under the target policy at that time."""
    missing: list[tuple[int, int, int]] = []
    for h, states in enumerate(mdp.target_state_support()):
        for s in sorted(states):
            for a in range(mdp.n_actions):
                if mdp.target.probs[h][s, a] > 0 and mdp.behavior.probs[h][s, a] <= 0:
                    missing.append((h, int(s), int(a)))
    return missing
