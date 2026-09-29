"""Episode-level data containers, K-fold splitting (no episode split), and
identifiability diagnostics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mdp import FiniteHorizonMDP, target_occupancy


@dataclass
class EpisodeDataset:
    mdp_name: str
    H: int
    n_episodes: int
    state: np.ndarray          # (n, H)
    action: np.ndarray         # (n, H)
    reward: np.ndarray         # (n, H)
    next_state: np.ndarray     # (n, H)
    done: np.ndarray           # (n, H) bool
    mask: np.ndarray           # (n, H) bool, valid timestep indicator
    p_behavior: np.ndarray     # (n, H), logged mu(a|s)
    p_target: np.ndarray       # (n, H)

    def folds(self, K: int, seed: int = 0) -> np.ndarray:
        """Return (n,) array of fold labels; each episode stays in one fold."""
        rng = np.random.default_rng(seed)
        idx = rng.permutation(self.n_episodes)
        labels = np.empty(self.n_episodes, dtype=np.int64)
        splits = np.array_split(idx, K)
        for k, group in enumerate(splits):
            labels[group] = k
        return labels


def _choice_p(probs2d: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Vectorized categorical draw for rows of ``probs2d`` (n, K)."""
    u = rng.random(probs2d.shape[0])
    cdf = np.cumsum(probs2d, axis=1)
    return (u[:, None] > cdf).sum(axis=1).clip(max=probs2d.shape[1] - 1)


def collect(mdp: FiniteHorizonMDP, n_episodes: int, seed: int) -> EpisodeDataset:
    """Vectorized rollout: all n episodes advance one stage per iteration."""
    rng = np.random.default_rng(seed)
    n, H = n_episodes, mdp.H
    S, A = mdp.n_states, mdp.n_actions
    state = np.zeros((n, H), np.int64)
    action = np.zeros((n, H), np.int64)
    reward = np.zeros((n, H))
    nxt = np.zeros((n, H), np.int64)
    done = np.zeros((n, H), bool)
    mask = np.zeros((n, H), bool)
    pb = np.zeros((n, H))
    pt = np.zeros((n, H))

    cur = _choice_p(np.broadcast_to(mdp.mu0, (n, S)).copy(), rng)
    alive = np.ones(n, bool)
    for t in range(H):
        live = np.where(alive)[0]
        state[live, t] = cur[live]
        mask[live, t] = True
        acts = _choice_p(mdp.behavior[t][cur[live]], rng)
        action[live, t] = acts
        kp = mdp.probs[t, cur[live], acts]           # (n_live, S)
        ns = _choice_p(kp, rng)
        p = mdp.reward_mean[t, cur[live], acts, ns]
        draw = rng.random(len(live))
        r = np.where((p > 0) & (p < 1), (draw < p).astype(float),
                     np.where(p >= 1, 1.0, 0.0))
        nxt[live, t] = ns
        reward[live, t] = r
        pb[live, t] = mdp.behavior[t, cur[live], acts]
        pt[live, t] = mdp.target[t, cur[live], acts]
        terminal = (ns == mdp.absorbing) | (t == H - 1)
        done[live, t] = terminal
        cur[live] = ns
        alive[live] = ~terminal
    return EpisodeDataset(
        mdp_name=mdp.name, H=H, n_episodes=n,
        state=state, action=action, reward=reward, next_state=nxt,
        done=done, mask=mask, p_behavior=pb, p_target=pt,
    )


def empirical_coverage(ds: EpisodeDataset, n_states: int, n_actions: int):
    """Count of observed (t, s, a) tuples. Zero count => no empirical support."""
    counts = np.zeros((ds.H, n_states, n_actions))
    valid = ds.mask
    np.add.at(counts,
              (np.broadcast_to(np.arange(ds.H), ds.state.shape)[valid],
               ds.state[valid], ds.action[valid]),
              1)
    return counts


def population_uncovered_target(mdp: FiniteHorizonMDP) -> np.ndarray:
    """Boolean (H, S, A): target puts mass where behaviour has *exact zero*.

    A True entry lying on a target-reachable (t,s) is a population
    non-identifiability witness for the policy value.
    """
    zero_b = mdp.behavior <= 0.0
    d = target_occupancy(mdp)                       # (H, S)
    target_mass = mdp.target > 0.0
    reachable = d[:, :, None] > 0
    return zero_b & target_mass & reachable


def empirical_target_gaps(ds: EpisodeDataset, mdp: FiniteHorizonMDP) -> np.ndarray:
    """Boolean (H, S, A): target-positive cells with zero empirical visits.

    This is the data-side flag: even where behaviour has epsilon support, a
    finite sample may contain zero episodes through a required cell.
    """
    counts = empirical_coverage(ds, mdp.n_states, mdp.n_actions)
    return (mdp.target > 0.0) & (counts == 0.0)


def identifiability_report(ds: EpisodeDataset, mdp: FiniteHorizonMDP) -> dict:
    pop_gap = population_uncovered_target(mdp)
    emp_gap = empirical_target_gaps(ds, mdp)
    d = target_occupancy(mdp)
    # weight of target occupancy that falls on population-zero cells
    pop_bad_mass = float((d[:, :, None] * mdp.target * pop_gap).sum())
    emp_bad_mass = float((d[:, :, None] * mdp.target * emp_gap).sum())
    return dict(
        population_identifiable=not pop_gap.any(),
        empirical_target_mass_supported=1.0 - emp_bad_mass,
        population_uncovered_mass=pop_bad_mass,
        empirical_uncovered_cells=np.argwhere(emp_gap),
        population_uncovered_cells=np.argwhere(pop_gap),
    )
