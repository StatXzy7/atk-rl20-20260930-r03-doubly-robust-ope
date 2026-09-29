"""Core OPE estimators (implemented from scratch).

For episode i and stage t define the cumulative PDIS weight

    rho_{0:t}^i = prod_{k=0..t} pi(a_k^i|s_k^i) / mu(a_k^i|s_k^i).

PDIS : mean_i sum_t gamma^t rho_{0:t}^i r_t^i
SNIS : [sum_{i,t} gamma^t rho_{0:t}^i r_t^i]
       / [sum_{i,t} gamma^t rho_{0:t}^i]
       (per-decision pooled self-normalization; the denominator targets the
       discount normalizer sum_t gamma^t, so it estimates J, not a weighted
       average of the observed discounted return)
DR   : sequential doubly robust, see docs/derivations.md:
       Vhat_t(s) = sum_a pi(a|s) Qhat_t(s,a), Vhat_H = 0
       delta_t   = r_t + gamma (1-done) Vhat_{t+1}(s') - Qhat_t(s,a)
       DR        = mean_i [ Vhat_0(s0) + sum_t gamma^t rho_{0:t}^i delta_t ]

Cumulative-weight clipping (shared) is a bias/variance trade-off and is never
described as "robustness": the clipped estimator is biased under correct props.
"""
from __future__ import annotations

import numpy as np

from . import weights as W
from .models import _subset


def _discount(ds, gamma: np.ndarray | float):
    disc = np.float_power(gamma, np.arange(ds.H))[None, :]
    return np.broadcast_to(disc, (ds.n_episodes, ds.H)).copy()


def _stage_weights(ds, clip):
    cum_ell = W.cumulative_log_weights(ds)
    w = W.weights_from_log(cum_ell, clip=clip)
    return np.where(ds.mask, w, 0.0)


def pdis_parts(ds, gamma: float, clip: float | None = None) -> dict:
    w = _stage_weights(ds, clip)
    disc = _discount(ds, gamma)
    per_i = (disc * w * ds.reward).sum(axis=1)
    return dict(estimate=float(per_i.mean()), per_episode=per_i, n_nan=0)




def snis_parts(ds, gamma: float, clip: float | None = None) -> dict:
    """Per-decision self-normalized IS.

    Numerator  : (1/n) sum_i sum_t gamma^t rho_{0:t}^i r_t^i        (= PDIS)
    Denominator: (1/n) sum_i sum_t gamma^t rho_{0:t}^i
                 = (sum_t gamma^t) * mean_t(normalization ratio)
    The leading 1/n cancels in the ratio, but the discount normalizer
    G = sum_t gamma^t must divide the pooled weighted sum -- otherwise the
    result is shrunk by a factor of G (this was a classic off-by-normalizer
    bug).  Bootstrap redraws reuse the same two pooled sums.
    """
    w = _stage_weights(ds, clip)
    disc = _discount(ds, gamma)
    wd = disc * w
    numer = (wd * ds.reward).sum(axis=1)
    denom = wd.sum(axis=1)
    G = float(disc[0].sum())
    den_total = denom.sum() / G / ds.n_episodes
    est = float(numer.mean() / den_total) if den_total > 0 else float("nan")
    return dict(estimate=est, numer=numer, denom=denom, G=G,
                n=ds.n_episodes, n_nan=0, kind="snis")


def combine_snis(parts: dict, idx: np.ndarray | None = None) -> float:
    n = parts["numer"]
    d = parts["denom"]
    if idx is not None:
        n, d = n[idx], d[idx]
    nb = len(n)
    den = d.sum() / parts["G"] / nb
    return float(n.mean() / den) if den > 0 else float("nan")


def dr_parts(ds, mdp, model, clip: float | None = None) -> dict:
    """Vectorized sequential DR.

    Vhat_t(s) = sum_a pi(a|s) Qhat_t(s,a), Vhat_H = 0
    delta_{i,t} = r + gamma (1-done) Vhat_{t+1}(s') - Qhat_t(s,a)
    DR_i = Vhat_0(s0) + sum_t gamma^t rho_{0:t} delta_{i,t}
    """
    H, gamma, S = mdp.H, mdp.gamma, mdp.n_states
    V = np.full((H, S), np.nan)
    for t in range(H):
        with np.errstate(invalid="ignore"):
            V[t] = np.nansum(mdp.target[t] * model.Q[t], axis=1)
    w = _stage_weights(ds, clip)
    disc = np.float_power(gamma, np.arange(H))[None, :]

    n = ds.n_episodes
    ii, tt = np.nonzero(ds.mask)
    qsa = model.Q[tt, ds.state[ii, tt], ds.action[ii, tt]]
    cont = np.zeros(len(ii))
    alive = ~ds.done[ii, tt]
    has_next = (tt + 1) < H
    use = alive & has_next
    cont[use] = V[tt[use] + 1, ds.next_state[ii[use], tt[use]]]
    delta = ds.reward[ii, tt] + gamma * cont - qsa
    w_dec = disc[0, tt] * w[ii, tt]

    # Expose nuisance-support gaps: a NaN Bellman correction sitting under a
    # LARGE cumulative weight is the dangerous case (rare target cell poorly
    # estimated -> its bias is multiplied). Never silently pretend it is 0.
    nan_corr = ~np.isfinite(delta)
    weighted_nan_mass = float(np.abs(w_dec[nan_corr]).sum())
    weighted_total = float(np.abs(w_dec).sum())
    lead = V[0][ds.state[:, 0]]
    lead_nan = ~np.isfinite(lead)

    term = np.zeros((n, H))
    term[ii, tt] = w_dec * np.nan_to_num(delta, nan=0.0)
    per_i = np.nan_to_num(lead, nan=0.0) + term.sum(axis=1)
    return dict(
        estimate=float(per_i.mean()), per_episode=per_i,
        n_nan=int(nan_corr.sum() + lead_nan.sum()),
        weighted_nan_fraction=(weighted_nan_mass / weighted_total
                               if weighted_total > 0 else 0.0),
        max_weight_on_nan=float(np.abs(w_dec[nan_corr]).max())
        if nan_corr.any() else 0.0,
    )


def crossfit_dr(ds, mdp, models_by_fold: dict, labels: np.ndarray,
                clip: float | None = None) -> dict:
    """Fold-k episodes scored by a model trained only on episodes outside k.

    An episode is wholly inside one fold (EpisodeDataset.folds), so no prefix
    or suffix of an evaluated episode ever enters its nuisance training set.
    """
    per_i = np.full(ds.n_episodes, np.nan)
    total_nan = 0
    nan_mass, max_w = [], []
    for k in np.unique(labels):
        idx = np.where(labels == k)[0]
        out = dr_parts(_subset(ds, idx), mdp, models_by_fold[int(k)], clip=clip)
        per_i[idx] = out["per_episode"]
        total_nan += out["n_nan"]
        nan_mass.append(out["weighted_nan_fraction"])
        max_w.append(out["max_weight_on_nan"])
    return dict(estimate=float(per_i.mean()), per_episode=per_i,
                n_nan=total_nan,
                weighted_nan_fraction=float(np.mean(nan_mass)),
                max_weight_on_nan=float(np.max(max_w)))
