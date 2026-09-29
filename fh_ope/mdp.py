"""Finite-horizon tabular MDPs and dynamic-programming ground truth.

Conventions
-----------
* Stages are t = 0, ..., H-1.  An episode contains H rows; rows after the
  episode entered an absorbing state are marked ``mask = False``.
* ``probs[t, s, a, :]`` is the transition kernel at stage ``t``.
* Rewards are Bernoulli with mean ``reward_mean[t, s, a, s']`` (0/1 means are
  deterministic), so every environment has an analytically known mean MDP and
  ground-truth policy value computed by backward dynamic programming.
* ``behavior`` / ``target`` are policy tables of shape ``(H, S, A)``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FiniteHorizonMDP:
    name: str
    H: int
    gamma: float
    mu0: np.ndarray                 # (S,)
    probs: np.ndarray              # (H, S, A, S)
    reward_mean: np.ndarray        # (H, S, A, S')
    behavior: np.ndarray           # (H, S, A)
    target: np.ndarray             # (H, S, A)
    absorbing: int | None = None

    @property
    def n_states(self) -> int:
        return self.probs.shape[1]

    @property
    def n_actions(self) -> int:
        return self.probs.shape[2]

    def rollout(self, rng: np.random.Generator, policy: np.ndarray | str = "behavior") -> dict:
        if isinstance(policy, str):
            policy = self.behavior if policy == "behavior" else self.target
        S, A, H = self.n_states, self.n_actions, self.H
        s = np.empty(H, dtype=np.int64)
        a = np.empty(H, dtype=np.int64)
        r = np.zeros(H, dtype=np.float64)
        s_next = np.zeros(H, dtype=np.int64)
        done = np.zeros(H, dtype=bool)
        p_b = np.zeros(H, dtype=np.float64)
        p_t = np.zeros(H, dtype=np.float64)
        mask = np.zeros(H, dtype=bool)

        state = int(rng.choice(S, p=self.mu0))
        active = True
        for t in range(H):
            if not active:
                break
            mask[t] = True
            s[t] = state
            at = int(rng.choice(A, p=policy[t, state]))
            a[t] = at
            ns = int(rng.choice(S, p=self.probs[t, state, at]))
            s_next[t] = ns
            p = float(self.reward_mean[t, state, at, ns])
            r[t] = float(rng.binomial(1, p)) if 0.0 < p < 1.0 else float(p)
            p_b[t] = self.behavior[t, state, at]
            p_t[t] = self.target[t, state, at]
            terminal = ns == self.absorbing or t == H - 1
            done[t] = terminal
            state = ns
            active = not terminal
        return dict(s=s, a=a, r=r, s_next=s_next, done=done, mask=mask,
                    p_behavior=p_b, p_target=p_t)


def dp_policy_value(mdp: FiniteHorizonMDP, policy: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Backward dynamic programming.

    Returns (J(pi), V, Q) with V shape (H+1, S) [V[H] = 0] and Q shape (H, S, A).
    """
    H, S, A = mdpp = (mdp.H, mdp.n_states, mdp.n_actions)
    Q = np.zeros((H, S, A))
    V = np.zeros((H + 1, S))
    for t in range(H - 1, -1, -1):
        immediate = np.einsum("saz,saz->sa", mdp.probs[t], mdp.reward_mean[t])
        cont = mdp.probs[t] @ V[t + 1]
        Q[t] = immediate + mdp.gamma * cont
        V[t] = np.einsum("sa,sa->s", policy[t], Q[t])
    J = float(mdp.mu0 @ V[0])
    return J, V, Q


def target_occupancy(mdp: FiniteHorizonMDP) -> np.ndarray:
    """Target-policy marginal state occupancy d_pi(t, s), sum_s d = 1 each t."""
    d = np.zeros((mdp.H, mdp.n_states))
    d[0] = mdp.mu0
    for t in range(mdp.H - 1):
        joint = d[t, :, None, None] * mdp.target[t, :, :, None] * mdp.probs[t]
        d[t + 1] = joint.sum(axis=(0, 1))
    return d


# --------------------------------------------------------------------------- #
# Environment factories
# --------------------------------------------------------------------------- #
def _full_chain_probs(nS: int, nA: int, H: int) -> np.ndarray:
    """Random-ish but fixed stochastic kernel on a non-absorbing chain."""
    rng = np.random.default_rng(0)
    P = rng.random((H, nS, nA, nS))
    # make transitions local: self and neighbours dominate
    for s in range(nS):
        P[:, s, :, :] *= 0.15
        P[:, s, :, s] += 0.7
        P[:, s, :, (s + 1) % nS] += 0.15
    return P / P.sum(axis=-1, keepdims=True)


def make_good_coverage() -> FiniteHorizonMDP:
    """MDP A: full softmax-style support, moderate bounded ratios."""
    S, A, H, gamma = 3, 2, 4, 0.9
    P = _full_chain_probs(S, A, H)
    rmean = np.zeros((H, S, A, S))
    vals = np.array([
        # t=0
        [[0.2, 0.8], [0.5, 0.5], [0.7, 0.1]],
        # t=1
        [[0.1, 0.6], [0.9, 0.2], [0.3, 0.4]],
        # t=2
        [[0.4, 0.7], [0.2, 0.8], [0.6, 0.5]],
        # t=3 (terminal stage)
        [[0.1, 0.9], [0.8, 0.3], [0.5, 0.6]],
    ], dtype=float)
    for t in range(H):
        rmean[t] = vals[t][:, :, None] * np.ones((1, 1, S))
    rng = np.random.default_rng(1)
    behavior = rng.random((H, S, A)) + 0.6
    behavior /= behavior.sum(axis=-1, keepdims=True)
    target = np.array([[[0.2, 0.8]] * S for _ in range(H)], dtype=float)
    target = np.broadcast_to(np.array([0.25, 0.75]), (H, S, A)).copy()
    mu0 = np.array([0.5, 0.3, 0.2])
    return FiniteHorizonMDP("good_coverage", H, gamma, mu0, P, rmean, behavior, target)


def make_tail_risk(eps: float = 0.05) -> FiniteHorizonMDP:
    """MDP B: behaviour is almost deterministic while the target puts high
    mass on the rare action.  Per-step ratios are 0.2/(1-eps) and 0.8/eps
    (the rare-action ratio is 16 at eps=0.05); cumulative weights along a
    target-heavy trajectory explode as (0.8/eps)^t even though every
    trajectory keeps strictly positive (nonzero) weight.
    """
    S, A, H, gamma = 2, 2, 5, 0.95
    P = np.zeros((H, S, A, S))
    # from s=0: a0 -> s1, a1 -> s0 ; from s=1: a0 -> s0, a1 -> s1
    P[:, 0, 0, 1] = 1.0
    P[:, 0, 1, 0] = 1.0
    P[:, 1, 0, 0] = 1.0
    P[:, 1, 1, 1] = 1.0
    rmean = np.zeros((H, S, A, S))
    rmean[:, 0, 1, :] = 0.8   # reward on the target-preferred actions
    rmean[:, 1, 0, :] = 0.8
    rmean[:, 0, 0, :] = 0.1
    rmean[:, 1, 1, :] = 0.1
    behavior = np.empty((H, S, A))
    # rare action: a1 in s0 and a0 in s1 -> exactly the target-preferred acts
    behavior[:, 0, :] = [1 - eps, eps]
    behavior[:, 1, :] = [eps, 1 - eps]
    target = np.empty((H, S, A))
    target[:, 0, :] = [0.2, 0.8]
    target[:, 1, :] = [0.8, 0.2]
    mu0 = np.array([1.0, 0.0])
    return FiniteHorizonMDP("tail_risk", H, gamma, mu0, P, rmean, behavior, target)


def make_no_coverage(secret_reward: float) -> FiniteHorizonMDP:
    """MDP C: states {0:start, 1:branch, 2:absorbing}.

    Both policies take a0 at stage 0 and reach state 1.  At stage 1 the target
    takes a1 while the behaviour policy puts *zero* mass on a1.  The reward
    realised by (s=1, a=1) is never observed and is the ``secret_reward``
    argument, so two MDPs with different target values generate identical data.
    """
    S, A, H, gamma, ABS = 3, 2, 3, 0.9, 2
    P = np.zeros((H, S, A, S))
    P[:, 0, 0, 1] = 1.0
    P[:, 0, 1, ABS] = 1.0
    P[:, 1, :, ABS] = 1.0
    P[:, ABS, :, ABS] = 1.0
    rmean = np.zeros((H, S, A, S))
    rmean[:, 0, 0, :] = 0.3       # observed, identical for both variants
    rmean[:, 1, 0, :] = 0.0       # observed behaviour action
    rmean[:, 1, 1, :] = secret_reward   # NEVER observed under the behaviour policy
    behavior = np.zeros((H, S, A))
    behavior[:, 0, :] = [1.0, 0.0]
    behavior[:, 1, :] = [1.0, 0.0]   # a1 has strictly zero mass
    behavior[:, ABS, :] = 0.5
    target = np.zeros((H, S, A))
    target[:, 0, 0] = 1.0
    target[:, 1, 1] = 1.0
    target[:, ABS, :] = 0.5
    mu0 = np.array([1.0, 0.0, 0.0])
    tag = "secret1" if secret_reward > 0.5 else "secret0"
    return FiniteHorizonMDP(f"no_coverage_{tag}", H, gamma, mu0, P, rmean,
                            behavior, target, absorbing=ABS)
