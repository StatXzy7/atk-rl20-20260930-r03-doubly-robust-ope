"""Tabular Q/V nuisance models fitted from logged transitions.

Q_h(s,a) is estimated by backward fitted-Q iteration with the *target* policy:

    y_i = r_h^i + gamma * Vhat_{h+1}(s_{h+1}^i),
    Vhat_{h+1}(s) = sum_a pi_{h+1}(a|s) Qhat_{h+1}(s,a),  Vhat_H = 0,
    Qhat_h(s,a) = mean{ y_i : (s_h^i,a_h^i)=(s,a) }  (transitions of TRAIN fold).

Modes:
  "empirical" : correct-spec tabular fitted-Q; unseen (s,a) cells fall back to
                the time-step mean target, which is still pure model extrapolation
                (counts of unseen cells are reported, not hidden).
  "oracle"    : exact Q from dynamic programming on the known model.
  "myopic"    : intentionally wrong: Q_h = cell mean immediate reward only.
  "zero"      : intentionally wrong: Q = 0; then DR algebraically equals PDIS.
"""

from __future__ import annotations

import numpy as np

from .data import EpisodeBatch
from .mdp import FiniteHorizonMDP, dp_evaluate


class TabularQ:
    def __init__(self, mdp: FiniteHorizonMDP, mode: str = "empirical"):
        if mode not in {"empirical", "oracle", "myopic", "zero"}:
            raise ValueError(f"unknown mode {mode}")
        self.mdp = mdp
        self.mode = mode
        self.H = mdp.horizon
        self.S = mdp.n_states
        self.A = mdp.n_actions
        self.Q: list[np.ndarray] = []
        self.V: list[np.ndarray] = []
        self.unseen_cells: list[int] = []
        self.fitted_ = False

    def fit(self, train: EpisodeBatch, oracle_q=None) -> "TabularQ":
        n_cell = self.S * self.A
        if self.mode == "oracle":
            if oracle_q is None:
                oracle_q = dp_evaluate(self.mdp)[1]
            self.Q = [np.array(q, copy=True) for q in oracle_q]
            self.unseen_cells = [0] * self.H
        elif self.mode == "zero":
            self.Q = [np.zeros((self.S, self.A)) for _ in range(self.H)]
            self.unseen_cells = [n_cell] * self.H
        else:
            gamma = self.mdp.discount
            pi = self.mdp.target.probs
            Vnext = np.zeros(self.S)
            self.Q = [None] * self.H
            self.unseen_cells = [0] * self.H
            for h in range(self.H - 1, -1, -1):
                if h + 1 < self.H:
                    Vnext = pi[h + 1] @ self.Q[h + 1].T if False else np.sum(
                        pi[h + 1] * self.Q[h + 1], axis=1)
                cont = Vnext[train.next_states[:, h]]
                y = train.rewards[:, h] + (0.0 if self.mode == "myopic" else gamma * cont)
                cell = train.states[:, h] * self.A + train.actions[:, h]
                sums = np.bincount(cell, weights=y, minlength=n_cell)
                cnt = np.bincount(cell, minlength=n_cell)
                glob = float(y.mean()) if train.n else 0.0
                q_flat = np.full(n_cell, glob)
                mask = cnt > 0
                q_flat[mask] = sums[mask] / cnt[mask]
                self.Q[h] = q_flat.reshape(self.S, self.A)
                self.unseen_cells[h] = int(np.sum(~mask))
        self.V = [np.sum(self.mdp.target.probs[h] * self.Q[h], axis=1)
                  for h in range(self.H)]
        self.fitted_ = True
        return self

    def arrays_for(self, batch: EpisodeBatch):
        """Return (q_sa, q_next_s, v0) evaluated on batch episodes."""
        if not self.fitted_:
            raise RuntimeError("model not fitted")
        n = batch.n
        q_sa = np.empty((n, self.H))
        q_next_s = np.zeros((n, self.H))
        for h in range(self.H):
            Qh = self.Q[h]
            q_sa[:, h] = Qh[batch.states[:, h], batch.actions[:, h]]
            if h + 1 < self.H:
                q_next_s[:, h] = self.V[h + 1][batch.next_states[:, h]]
        v0 = self.V[0][batch.initial_states]
        return q_sa, q_next_s, v0

    def plugin_value(self, batch: EpisodeBatch) -> float:
        _, _, v0 = self.arrays_for(batch)
        return float(np.mean(v0))
