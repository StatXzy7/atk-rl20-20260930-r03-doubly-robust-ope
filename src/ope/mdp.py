"""Finite-horizon tabular MDP and independent ground-truth evaluation by DP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Policy:
    """Time-dependent policy probs[h][state, action], h = 0, ..., H-1."""

    probs: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "probs", tuple(np.asarray(p, float) for p in self.probs))

    @property
    def H(self) -> int:
        return len(self.probs)

    def log_probs(self, h: int) -> np.ndarray:
        out = np.full(self.probs[h].shape, -np.inf)
        np.log(self.probs[h], out=out, where=self.probs[h] > 0.0)
        return out


@dataclass(frozen=True)
class BernoulliReward:
    """Bernoulli reward with time-dependent mean rmean[h][state, action]."""

    rmean: tuple[np.ndarray, ...]

    @property
    def H(self) -> int:
        return len(self.rmean)

    def mean(self, h: int) -> np.ndarray:
        return self.rmean[h]

    def sample(self, h: int, s: np.ndarray, a: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        p = self.rmean[h][s, a]
        return rng.binomial(1, p).astype(float)


def _as_homo(tables, H: int, name: str) -> tuple[np.ndarray, ...]:
    if isinstance(tables, tuple):
        if len(tables) != H:
            raise ValueError(f"{name}: expected {H} time tables, got {len(tables)}")
        return tuple(np.asarray(t, float) for t in tables)
    arr = np.asarray(tables, float)
    return tuple(arr for _ in range(H))


@dataclass(frozen=True)
class FiniteHorizonMDP:
    """P[h][s, a, s'] and time-dependent (or homogeneous) policy/reward tables."""

    horizon: int
    n_states: int
    n_actions: int
    P: tuple[np.ndarray, ...]
    reward: BernoulliReward
    target: Policy
    behavior: Policy
    p0: np.ndarray
    discount: float

    @staticmethod
    def build(
        horizon: int,
        P,
        rmean,
        target,
        behavior,
        p0: np.ndarray,
        discount: float = 1.0,
    ) -> "FiniteHorizonMDP":
        P_t = _as_homo(P, horizon, "P")
        r_t = _as_homo(rmean, horizon, "rmean")
        pi_t = _as_homo(target, horizon, "target")
        mu_t = _as_homo(behavior, horizon, "behavior")
        n_states, n_actions = pi_t[0].shape
        for h in range(horizon):
            if P_t[h].shape != (n_states, n_actions, n_states):
                raise ValueError(f"P[{h}] has shape {P_t[h].shape}")
            if not np.allclose(P_t[h].sum(-1), 1.0, atol=1e-10):
                raise ValueError(f"P[{h}] rows must sum to 1")
            for nm, pol in (("target", pi_t), ("behavior", mu_t)):
                if not np.allclose(pol[h].sum(-1), 1.0, atol=1e-10):
                    raise ValueError(f"{nm}[{h}] rows must sum to 1")
                if np.any(pol[h] < 0):
                    raise ValueError(f"{nm}[{h}] has negative entries")
        if np.asarray(p0).shape != (n_states,) or not np.isclose(np.sum(p0), 1.0):
            raise ValueError("p0 must be a probability vector over states")
        return FiniteHorizonMDP(
            horizon=horizon,
            n_states=n_states,
            n_actions=n_actions,
            P=P_t,
            reward=BernoulliReward(r_t),
            target=Policy(pi_t),
            behavior=Policy(mu_t),
            p0=np.asarray(p0, float),
            discount=float(discount),
        )

    def target_state_support(self) -> list[set[int]]:
        """States reachable under the target policy at each step (from p0)."""
        reach = [{int(s) for s in np.flatnonzero(self.p0 > 0)}]
        for h in range(self.horizon):
            nxt: set[int] = set()
            for s in reach[h]:
                for a in np.flatnonzero(self.target.probs[h][s] > 0):
                    nxt.update(int(x) for x in np.flatnonzero(self.P[h][s, a] > 0))
            reach.append(nxt)
        reach.pop()
        return reach


def dp_evaluate(mdp: FiniteHorizonMDP):
    """Backward DP under the target policy on the *known* transition model.

    Returns (J = E_{s0~p0} V_0(s0), Q tables, V tables) indexed by time.
    This value is independent of any collected logged data.
    """
    H = mdp.horizon
    Q = [np.zeros((mdp.n_states, mdp.n_actions)) for _ in range(H)]
    V = [np.zeros(mdp.n_states) for _ in range(H + 1)]
    for h in range(H - 1, -1, -1):
        cont = mdp.P[h] @ V[h + 1]
        Q[h] = mdp.reward.mean(h) + mdp.discount * cont
        V[h] = np.sum(mdp.target.probs[h] * Q[h], axis=1)
    return float(np.dot(mdp.p0, V[0])), Q, V
