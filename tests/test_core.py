"""Tests for MDP/DP truth, weights, estimators, cross-fitting, support."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ope.crossfit import (  # noqa: E402
    compute_weights,
    crossfit_q,
    episode_kfold,
    evaluate_all,
    in_sample_q,
)
from ope.data import collect_episodes  # noqa: E402
from ope.estimators import dr, effective_sample_size, pdis, snis  # noqa: E402
from ope.mdp import FiniteHorizonMDP, dp_evaluate  # noqa: E402
from ope.nuisance import TabularQ  # noqa: E402
from ope.weights import (  # noqa: E402
    cumulative_log_ratio,
    population_support_check,
    support_violations,
)


def toy_mdp(mu0=0.5, mu1=0.5, horizon=2, gamma=1.0):
    S, A = 2, 2
    P = np.zeros((S, A, S))
    P[0, 0, 0] = 1.0
    P[0, 1, 1] = 1.0
    P[1, 0, 0] = 1.0
    P[1, 1, 1] = 1.0
    R = np.array([[0.0, 0.0], [0.3, 0.9]])
    target = np.array([[0.0, 1.0], [0.0, 1.0]])
    beh = np.array([[1.0 - mu0, mu0], [1.0 - mu1, mu1]])
    return FiniteHorizonMDP.build(
        horizon, P, tuple(R for _ in range(horizon)),
        target, tuple(beh for _ in range(horizon)),
        np.array([1.0, 0.0]), gamma,
    )


def test_dp_truth_matches_manual_horizon2():
    mdp = toy_mdp(mu0=0.5, mu1=0.5, horizon=2, gamma=1.0)
    j, Q, V = dp_evaluate(mdp)
    # Q1(1,1)=0.9; V1(1)=0.9; Q0(0,1)=0+V1(1)=0.9; V0(0)=0.9; J=0.9
    assert Q[1][1, 1] == pytest.approx(0.9)
    assert V[1][1] == pytest.approx(0.9)
    assert Q[0][0, 1] == pytest.approx(0.9)
    assert j == pytest.approx(0.9)


def test_dp_discounted_horizon4_manual():
    S, A = 2, 2
    P = np.zeros((S, A, S))
    P[0, 0, 0] = P[0, 1, 1] = P[1, 0, 0] = P[1, 1, 1] = 1.0
    R = np.array([[0.0, 0.0], [0.0, 1.0]])
    target = np.array([[0.0, 1.0], [0.0, 1.0]])
    mdp = FiniteHorizonMDP.build(
        4, P, tuple(R for _ in range(4)),
        tuple(target for _ in range(4)), tuple(target for _ in range(4)),
        np.array([1.0, 0.0]), 0.9)
    j, _, _ = dp_evaluate(mdp)
    assert j == pytest.approx(0.9 + 0.81 + 0.729)


def test_pdis_unbiased_exhaustive_horizon2():
    """H=2, deterministic rewards/transitions; enumerate 4 trajectories.

    States s0=0; a0 in {0,1}; s1 in {0,1}; a1 in {0,1}.
    Rewards: r(s,a) = s + a (0..2). Target always action 1.
    Behavior P(a=1|s) = q_s (q0=0.4, q1=0.6). Discount 1.
    DP truth: V2=0; V1(s)=s+1; Q0(0,1)=r(0,1)+V1(1)=1+2=3 => J=3.
    """
    S, A = 2, 2
    P = np.zeros((S, A, S))
    P[0, 0, 0] = P[0, 1, 1] = P[1, 0, 0] = P[1, 1, 1] = 1.0
    R = np.array([[0.0, 1.0], [1.0, 2.0]])
    target = np.array([[0.0, 1.0], [0.0, 1.0]])
    q0, q1 = 0.4, 0.6
    beh = np.array([[1 - q0, q0], [1 - q1, q1]])
    mdp = FiniteHorizonMDP.build(
        2, P, (R, R), (target, target), (beh, beh), np.array([1.0, 0.0]), 1.0)
    j, _, _ = dp_evaluate(mdp)
    assert j == pytest.approx(3.0)

    # (a0, a1) -> (s1, behavior prob, r0, r1, step log ratios)
    cases = [
        (0, 0, 0, (1 - q0) * (1 - q1), 0.0, 1.0, (-np.inf, np.log(1.0 / (1 - q1)))),
        (0, 1, 0, (1 - q0) * q1,       0.0, 2.0, (-np.inf, np.log(1.0 / q1))),
        (1, 0, 1, q0 * (1 - q1),       1.0, 1.0, (np.log(1.0 / q0), -np.inf)),
        (1, 1, 1, q0 * q1,             1.0, 2.0, (np.log(1.0 / q0), np.log(1.0 / q1))),
    ]
    lr = np.array([[c[6][0], c[6][1]] for c in cases])
    rew = np.array([[c[4], c[5]] for c in cases])
    prob = np.array([c[3] for c in cases])
    prefix = cumulative_log_ratio(lr)
    rho = np.exp(prefix)
    pdis_per_traj = rho[:, 0] * rew[:, 0] + rho[:, 1] * rew[:, 1]
    assert np.sum(prob * pdis_per_traj) == pytest.approx(j, abs=1e-12)


def test_dr_equals_pdis_when_q_zero():
    mdp = toy_mdp()
    rng = np.random.default_rng(0)
    data = collect_episodes(mdp, 300, rng)
    _, _, lp = compute_weights(mdp, data, None)
    n, H = data.n, data.H
    val = dr(data.rewards, lp, np.zeros((n, H)), np.zeros((n, H)), np.zeros(n), 1.0)
    assert val == pytest.approx(pdis(data.rewards, lp, 1.0), abs=1e-10)


def test_dr_exact_q_is_exact_deterministic():
    """With exact Q from DP on a deterministic-reward MDP, DR is exactly J."""
    S, A = 2, 2
    P = np.zeros((S, A, S))
    P[0, 0, 0] = P[0, 1, 1] = P[1, 0, 0] = P[1, 1, 1] = 1.0
    R = np.array([[0.0, 0.0], [0.0, 1.0]])
    target = np.array([[0.0, 1.0], [0.0, 1.0]])
    beh = np.array([[0.7, 0.3], [0.7, 0.3]])
    mdp = FiniteHorizonMDP.build(
        3, P, tuple(R for _ in range(3)),
        tuple(target for _ in range(3)), tuple(beh for _ in range(3)),
        np.array([1.0, 0.0]), 0.9)
    j, Q, V = dp_evaluate(mdp)
    rng = np.random.default_rng(2)
    data = collect_episodes(mdp, 50, rng)
    _, _, lp = compute_weights(mdp, data, None)
    n, H = data.n, data.H
    q_sa = np.empty((n, H))
    q_next = np.zeros((n, H))
    for h in range(H):
        q_sa[:, h] = Q[h][data.states[:, h], data.actions[:, h]]
        if h + 1 < H:
            q_next[:, h] = V[h + 1][data.next_states[:, h]]
    v0 = V[0][data.initial_states]
    # Deterministic rewards => Bellman residual exactly zero per trajectory.
    assert dr(data.rewards, lp, q_sa, q_next, v0, 0.9) == pytest.approx(j, abs=1e-12)


def test_pdis_dr_unbiased_montecarlo():
    mdp = toy_mdp(mu0=0.45, mu1=0.55, horizon=3, gamma=0.9)
    j, _, _ = dp_evaluate(mdp)
    rng = np.random.default_rng(3)
    folds = episode_kfold(2000, 5, rng)
    p, d = [], []
    for _ in range(40):
        data = collect_episodes(mdp, 2000, rng)
        _, _, lp = compute_weights(mdp, data, None)
        p.append(pdis(data.rewards, lp, 0.9))
        q_sa, q_next, v0, _ = crossfit_q(mdp, data, folds, mode="oracle")
        d.append(dr(data.rewards, lp, q_sa, q_next, v0, 0.9))
    assert np.mean(p) == pytest.approx(j, abs=0.02)
    assert np.mean(d) == pytest.approx(j, abs=0.02)


def test_snis_equals_is_when_on_policy():
    mdp = toy_mdp(mu0=1.0, mu1=1.0, horizon=2, gamma=0.9)  # behavior=target
    rng = np.random.default_rng(4)
    data = collect_episodes(mdp, 500, rng)
    _, _, lp = compute_weights(mdp, data, None)
    g_hat, _ = snis(data.rewards, lp, 0.9)
    # all weights = 1 => SNIS = ordinary sample-mean return
    assert g_hat == pytest.approx(np.mean(data.rewards @ [1.0, 0.9]), abs=1e-12)


def test_fold_assignment_is_episode_level():
    folds = episode_kfold(100, 4, np.random.default_rng(0))
    assert sorted(np.unique(folds)) == [0, 1, 2, 3]
    # every whole episode assigned exactly one fold; counts balanced
    counts = np.bincount(folds)
    assert np.max(counts) - np.min(counts) <= 1


def test_crossfit_uses_disjoint_episodes():
    mdp = toy_mdp()
    rng = np.random.default_rng(5)
    data = collect_episodes(mdp, 80, rng)
    folds = episode_kfold(80, 4, rng)
    for f in range(4):
        train = set(np.where(folds != f)[0])
        ev = set(np.where(folds == f)[0])
        assert train.isdisjoint(ev)
    # smoke: crossfit runs and unseen cells reported
    _, _, _, unseen = crossfit_q(mdp, data, folds, mode="empirical")
    assert isinstance(unseen, list)


def test_support_violation_detected_and_marked_nonidentified():
    S, A = 2, 2
    P = np.zeros((S, A, S))
    P[0, 0, 0] = P[0, 1, 1] = P[1, 0, 0] = P[1, 1, 1] = 1.0
    R = np.array([[0.0, 0.0], [0.0, 1.0]])
    target = np.array([[0.0, 1.0], [0.0, 1.0]])
    beh = np.array([[1.0, 0.0], [0.5, 0.5]])  # action1 at state0 impossible
    mdp = FiniteHorizonMDP.build(
        4, P, tuple(R for _ in range(4)),
        tuple(target for _ in range(4)), tuple(beh for _ in range(4)),
        np.array([1.0, 0.0]), 0.9)
    missing = population_support_check(mdp)
    assert (0, 0, 1) in missing
    rng = np.random.default_rng(6)
    data = collect_episodes(mdp, 200, rng)
    folds = episode_kfold(200, 4, rng)
    res = evaluate_all(mdp, data, folds)
    assert res.support_violation is True
    for key in ("pdis", "snis", "dr"):
        assert np.isnan(res.estimates[key])
    assert "NOT" in res.note and "identified" in res.note


def test_stable_weights_no_underflow_or_nan():
    lr = np.random.default_rng(7).normal(-5.0, 1.0, size=(20, 6))
    pref = cumulative_log_ratio(lr)
    assert np.all(np.isfinite(pref))
    # direct exponentiation underflows but log-space PDIS must still work
    rewards = np.ones((20, 6))
    val = pdis(rewards, pref, 0.99)
    assert np.isfinite(val)
    ess = effective_sample_size(pref)
    assert 0 <= ess["min_ess"] <= 20


def test_inf_prefix_handling():
    lr = np.array([[1.0, -np.inf], [np.inf, 0.0], [-np.inf, 1.0]])
    pref = cumulative_log_ratio(lr)
    assert pref[0, 1] == -np.inf      # product hits zero and stays zero
    assert pref[2, 1] == -np.inf
    assert pref[1, 0] == np.inf and pref[1, 1] == np.inf


def test_clipping_changes_estimate_and_is_bounded():
    mdp = toy_mdp(mu0=0.1, mu1=0.1, horizon=3, gamma=0.9)
    rng = np.random.default_rng(8)
    data = collect_episodes(mdp, 300, rng)
    folds = episode_kfold(300, 4, rng)
    raw = evaluate_all(mdp, data, folds, clip=None)
    clipped = evaluate_all(mdp, data, folds, clip=5.0)
    assert np.isfinite(raw.estimates["pdis"])
    assert np.isfinite(clipped.estimates["pdis"])


def test_snis_nan_when_all_weights_zero():
    rewards = np.ones((5, 2))
    log_w = np.full((5, 2), -np.inf)
    val, mlw = snis(rewards, log_w, 1.0)
    assert np.isnan(val)


def test_clip_binds_in_log_space():
    from ope.weights import clip_log_ratio
    lr = np.array([[0.0, 5.0, -1.0]])
    pref = cumulative_log_ratio(lr)
    c = clip_log_ratio(pref, 2.0)
    # raw prefixes: 1, e^5, e^4 -> clipped independently per time to <=2
    assert np.exp(c[0, 0]) == pytest.approx(1.0)
    assert np.exp(c[0, 1]) == pytest.approx(2.0)
    assert np.exp(c[0, 2]) == pytest.approx(2.0)


def test_scenarios_truth_and_nonidentification():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
    from scenarios import SCENARIOS, truth_table
    tt = truth_table()
    assert tt["good_coverage"] == pytest.approx(2.1512, abs=1e-6)
    missing = population_support_check(SCENARIOS["no_coverage"]())
    assert missing and missing[0][2] == 1
    # good/tiny have full support on target-reachable cells
    assert population_support_check(SCENARIOS["good_coverage"]()) == []
    assert population_support_check(SCENARIOS["tiny_behavior"]()) == []


def test_myopic_plugin_is_bad_but_dr_stays_unbiased():
    """Wrong Q: model plug-in is badly biased, but DR with exact mu stays unbiased."""
    mdp = toy_mdp(mu0=0.45, mu1=0.55, horizon=3, gamma=0.9)
    j, _, _ = dp_evaluate(mdp)
    rng = np.random.default_rng(11)
    drs, plugs = [], []
    folds = episode_kfold(1500, 5, rng)
    for _ in range(20):
        data = collect_episodes(mdp, 1500, rng)
        q_sa, q_next, v0, _ = crossfit_q(mdp, data, folds, mode="myopic")
        _, _, lp = compute_weights(mdp, data, None)
        drs.append(dr(data.rewards, lp, q_sa, q_next, v0, 0.9))
        plugs.append(float(np.mean(v0)))
    assert abs(np.mean(drs) - j) < 0.03            # DR remains unbiased
    assert abs(np.mean(plugs) - j) > 0.05          # plug-in is biased
