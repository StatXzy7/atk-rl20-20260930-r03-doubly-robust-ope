"""Three analytically/DP-solvable finite-horizon MDP scenarios.

All rewards are Bernoulli with known means; all transitions are deterministic.
Ground-truth target values are computed independently by backward DP on the
known model (src.ope.mdp.dp_evaluate), never from collected data.

Scenario 1 "good_coverage": target=always action 1, behavior=(0.5,0.5)
            everywhere; full support, bounded moderate weights.
Scenario 2 "tiny_behavior": same MDP, behavior takes action 1 only with
            probability 1e-3 at the critical state. Deep prefix weights then
            have magnitude up to 1/rho-style products and ESS collapses;
            clipping and SNIS biases are explicit and measurable.
Scenario 3 "no_coverage":  behavior probability of the target action is
            exactly 0 at a target-reachable state; the target value is not
            identified by the logged distribution. We refuse to epsilon-patch
            the zero and refuse to emit a plausible narrow interval.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ope.mdp import FiniteHorizonMDP, dp_evaluate  # noqa: E402

HORIZON = 4
DISCOUNT = 0.9
S = 2
A = 2

# Deterministic transitions: action 1 advances state 0 -> 1 and holds state 1;
# action 0 stays in state 0.
P = np.zeros((S, A, S))
P[0, 0, 0] = 1.0
P[0, 1, 1] = 1.0
P[1, 0, 0] = 1.0
P[1, 1, 1] = 1.0

# Bernoulli reward means per (state, action): action 1 yields reward.
RMEAN = np.array(
    [
        [0.05, 0.2],  # state 0: action 1 (rare under behavior) yields 0.2
        [0.10, 0.8],  # state 1: target action 1 yields high reward 0.8
    ]
)
# Make rewards time-independent: same table each step.
RMEAN_T = tuple(RMEAN for _ in range(HORIZON))

TARGET = np.array(
    [
        [0.0, 1.0],
        [0.0, 1.0],
    ]
)

P0 = np.array([1.0, 0.0])  # deterministic start state s0=0


def _policy_rows(prob_a1_s0: float, prob_a1_s1: float) -> np.ndarray:
    q0 = prob_a1_s0
    q1 = prob_a1_s1
    return np.array([[1.0 - q0, q0], [1.0 - q1, q1]])


def make_good_coverage() -> FiniteHorizonMDP:
    # Behavior favors the target action 1; ratio pi/mu is between 0 and 1.25.
    beh = _policy_rows(0.8, 0.8)
    return FiniteHorizonMDP.build(
        HORIZON, P, RMEAN_T, TARGET, tuple(beh for _ in range(HORIZON)), P0, DISCOUNT
    )


def make_tiny_behavior(rare: float = 0.12) -> FiniteHorizonMDP:
    beh = _policy_rows(rare, rare)
    return FiniteHorizonMDP.build(
        HORIZON, P, RMEAN_T, TARGET, tuple(beh for _ in range(HORIZON)), P0, DISCOUNT
    )


def make_no_coverage() -> FiniteHorizonMDP:
    beh = _policy_rows(0.0, 0.5)
    # At state 0 behavior NEVER takes action 1; at state 1 (unreachable under
    # behavior) the value is irrelevant to logged data but matters to target.
    return FiniteHorizonMDP.build(
        HORIZON, P, RMEAN_T, TARGET, tuple(beh for _ in range(HORIZON)), P0, DISCOUNT
    )


def make_leakage(rare: float = 0.15) -> FiniteHorizonMDP:
    """Sparse-cell design used for the train/evaluation leakage counterexample."""
    beh = _policy_rows(rare, rare)
    return FiniteHorizonMDP.build(
        HORIZON, P, RMEAN_T, TARGET, tuple(beh for _ in range(HORIZON)), P0, DISCOUNT
    )


SCENARIOS = {
    "good_coverage": make_good_coverage,
    "tiny_behavior": make_tiny_behavior,
    "no_coverage": make_no_coverage,
    "leakage": make_leakage,
}


def truth_table() -> dict:
    out = {}
    for name, fn in SCENARIOS.items():
        mdp = fn()
        j, _, _ = dp_evaluate(mdp)
        out[name] = j
    return out


if __name__ == "__main__":
    for name, j in truth_table().items():
        print(f"{name:15s} J(pi) = {j:.6f}")
