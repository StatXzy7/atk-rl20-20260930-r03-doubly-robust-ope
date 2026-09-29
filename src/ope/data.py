"""Episode-level logged data. A complete episode is the independent unit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mdp import FiniteHorizonMDP

_FIELDS = ("states", "actions", "rewards", "next_states", "log_mu")


@dataclass
class EpisodeBatch:
    """Per-trajectory arrays, all shape (n_episodes, horizon).

    states/actions/rewards/next_states are per time step; log_mu[h] is the
    *recorded* log behavior probability of the logged action. `done` is fixed
    at the terminal step by construction and is not carried around. No segment
    of an episode is ever split across folds.
    """

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    log_mu: np.ndarray
    initial_states: np.ndarray

    @property
    def n(self) -> int:
        return self.states.shape[0]

    @property
    def H(self) -> int:
        return self.states.shape[1]

    @property
    def done(self) -> np.ndarray:
        out = np.zeros((self.n, self.H), dtype=bool)
        out[:, -1] = True
        return out

    def take(self, idx: np.ndarray) -> "EpisodeBatch":
        """Resample whole episodes. Flat-index fancy indexing in one pass each."""
        idx = np.asarray(idx)
        return EpisodeBatch(
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.log_mu[idx],
            self.initial_states[idx],
        )


def collect_episodes(mdp: FiniteHorizonMDP, n: int, rng: np.random.Generator) -> EpisodeBatch:
    """Sample n iid fixed-horizon episodes under the behavior policy."""
    H = mdp.horizon
    states = np.empty((n, H), dtype=np.int64)
    actions = np.empty((n, H), dtype=np.int64)
    rewards = np.empty((n, H), dtype=float)
    next_states = np.empty((n, H), dtype=np.int64)
    log_mu = np.empty((n, H), dtype=float)

    s = rng.choice(mdp.n_states, size=n, p=mdp.p0)
    initial = s.copy()
    for h in range(H):
        mu = mdp.behavior.probs[h]
        a = _categorical(mu[s], rng)
        r = mdp.reward.sample(h, s, a, rng)
        ns = _categorical(mdp.P[h][s, a], rng)
        states[:, h] = s
        actions[:, h] = a
        rewards[:, h] = r
        next_states[:, h] = ns
        probs_sa = mu[s, a]
        col = np.full(n, -np.inf)
        np.log(probs_sa, out=col, where=probs_sa > 0.0)
        log_mu[:, h] = col
        s = ns
    return EpisodeBatch(states, actions, rewards, next_states, log_mu, initial)


def _categorical(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    cdf = np.cumsum(probs, axis=-1)
    u = rng.random(probs.shape[:-1] + (1,))
    return np.argmax(u <= cdf, axis=-1)
