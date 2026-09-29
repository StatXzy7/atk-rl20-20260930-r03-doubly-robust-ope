"""Tabular finite-horizon Q/V nuisance models fit by backward recursion.

Correct model at stage t estimates
    E[r_t + gamma (1-done) V_{t+1}(s_{t+1}) | s_t, a_t]
on logged transitions, with V_{t+1} derived from the fitted Q_{t+1} under
the target policy.  Cells absent from the training folds are NaN, so an
estimator never silently substitutes zero.

``action_blind=True`` pools over the action when estimating conditional means
-- an intentionally wrong Q control with nonzero error wherever true Q depends
on the action.

All fits/estimation are vectorized over episodes for speed.
"""
from __future__ import annotations

import numpy as np

from .mdp import FiniteHorizonMDP, dp_policy_value


class TabularQModel:
    def __init__(self, n_states: int, n_actions: int, H: int, gamma: float,
                 action_blind: bool = False):
        self.S, self.A, self.H, self.gamma = n_states, n_actions, H, gamma
        self.action_blind = action_blind
        self.Q = np.full((H, n_states, n_actions), np.nan)

    def fit(self, ds, target_probs: np.ndarray) -> "TabularQModel":
        Vnext = np.zeros(self.S)
        for t in range(self.H - 1, -1, -1):
            m = ds.mask[:, t]
            s = ds.state[m, t]
            a = ds.action[m, t]
            cont = np.zeros(m.sum())
            alive = ~ds.done[m, t]
            cont[alive] = Vnext[ds.next_state[m, t][alive]]
            y = ds.reward[m, t] + self.gamma * cont

            if self.action_blind:
                sums_s = np.bincount(s, weights=y, minlength=self.S)
                cnt_s = np.bincount(s, minlength=self.S).astype(float)
                with np.errstate(invalid="ignore"):
                    mean_s = np.where(cnt_s > 0, sums_s / np.where(cnt_s > 0, cnt_s, 1),
                                      np.nan)
                self.Q[t] = mean_s[:, None]
            else:
                flat = s * self.A + a
                sums = np.bincount(flat, weights=y, minlength=self.S * self.A)
                cnt = np.bincount(flat, minlength=self.S * self.A).astype(float)
                with np.errstate(invalid="ignore"):
                    mean = np.where(cnt > 0, sums / np.where(cnt > 0, cnt, 1), np.nan)
                self.Q[t] = mean.reshape(self.S, self.A)

            with np.errstate(invalid="ignore"):
                Vnext = np.nansum(target_probs[t] * self.Q[t], axis=1)
            Vnext[~np.isfinite(self.Q[t]).any(axis=1)] = np.nan
        return self

    def q(self, t, s, a):
        return self.Q[t, s, a]


class OracleQModel:
    def __init__(self, mdp: FiniteHorizonMDP):
        _, _, self.Q = dp_policy_value(mdp, mdp.target)

    def q(self, t, s, a):
        return self.Q[t, s, a]

    def fit(self, *args, **kwargs):
        return self


def _subset(ds, idx):
    from .data import EpisodeDataset
    return EpisodeDataset(
        mdp_name=ds.mdp_name, H=ds.H, n_episodes=int(len(idx)),
        state=ds.state[idx], action=ds.action[idx], reward=ds.reward[idx],
        next_state=ds.next_state[idx], done=ds.done[idx], mask=ds.mask[idx],
        p_behavior=ds.p_behavior[idx], p_target=ds.p_target[idx],
    )


def fit_on_folds(ds, mdp: FiniteHorizonMDP, labels: np.ndarray,
                 action_blind: bool = False) -> dict:
    """Model for fold k is trained on episodes OUTSIDE fold k only.

    Folds partition whole episodes (see EpisodeDataset.folds), so a prefix or
    suffix of an evaluated episode can never enter its own training fold.
    """
    return {
        int(k): TabularQModel(
            mdp.n_states, mdp.n_actions, mdp.H, mdp.gamma,
            action_blind=action_blind
        ).fit(_subset(ds, labels != k), mdp.target)
        for k in np.unique(labels)
    }


def fit_in_sample(ds, mdp: FiniteHorizonMDP, action_blind: bool = False):
    m = TabularQModel(mdp.n_states, mdp.n_actions, mdp.H, mdp.gamma,
                      action_blind=action_blind).fit(ds, mdp.target)
    return m
