import numpy as np
import pytest

from fh_ope.mdp import (make_good_coverage, make_tail_risk, make_no_coverage,
                        dp_policy_value)
from fh_ope import data as D
from fh_ope import weights as W
from fh_ope import estimators as E
from fh_ope.models import TabularQModel, OracleQModel, fit_on_folds


def test_dp_matches_monte_carlo_target():
    import dataclasses
    mdp = make_good_coverage()
    J, _, _ = dp_policy_value(mdp, mdp.target)
    # roll out UNDER the target policy (swap logged policy)
    target_mdp = dataclasses.replace(mdp, behavior=mdp.target)
    ds = D.collect(target_mdp, 60000, 7)
    disc = mdp.gamma ** np.arange(mdp.H)
    rets = (disc[None, :] * ds.reward * ds.mask).sum(axis=1)
    se = rets.std() / np.sqrt(len(rets))
    assert abs(rets.mean() - J) < 3 * se


def test_pdis_unbiased_good_coverage():
    mdp = make_good_coverage()
    truth, _, _ = dp_policy_value(mdp, mdp.target)
    ests = []
    for seed in range(120):
        ds = D.collect(mdp, 300, seed)
        ests.append(E.pdis_parts(ds, mdp.gamma)["estimate"])
    assert abs(np.mean(ests) - truth) < 0.05


def test_snis_near_truth_good_coverage():
    mdp = make_good_coverage()
    truth, _, _ = dp_policy_value(mdp, mdp.target)
    ests = []
    for seed in range(60):
        ds = D.collect(mdp, 400, seed)
        ests.append(E.snis_parts(ds, mdp.gamma)["estimate"])
    # SNIS is consistent and biased at finite n; with n=400 it is close
    assert abs(np.mean(ests) - truth) < 0.05


def test_dr_exact_in_expectation_with_oracle_q():
    """Exact Q => Bellman correction has conditional mean zero at every stage;
    DR is unbiased. Single-episode values fluctuate (rewards are random)."""
    mdp = make_good_coverage()
    truth, _, _ = dp_policy_value(mdp, mdp.target)
    ests = [E.dr_parts(D.collect(mdp, 500, s), mdp, OracleQModel(mdp))["estimate"]
            for s in range(80)]
    se = np.std(ests) / np.sqrt(len(ests))
    assert abs(np.mean(ests) - truth) < 3 * se


def test_dr_exact_with_oracle_q_deterministic():
    """Exact Q on a deterministic-reward horizon makes every DR value equal J
    (the Bellman correction is identically zero). Construct a one-stage
    deterministic-reward horizon so nothing is random."""
    from fh_ope.mdp import FiniteHorizonMDP
    S, A, H = 2, 2, 1
    P = np.zeros((H, S, A, S)); P[0, :, :, 1] = 1.0
    rm = np.zeros((H, S, A, S)); rm[0, 0, 0, 1] = 1.0; rm[0, 0, 1, 1] = 0.4
    pol = np.zeros((H, S, A)); pol[0, :, 0] = 1.0
    mdp = FiniteHorizonMDP("det", H, 1.0, np.array([1.0, 0.0]), P, rm, pol, pol,
                           absorbing=1)
    truth, _, _ = dp_policy_value(mdp, mdp.target)
    ds = D.collect(mdp, 50, 0)
    out = E.dr_parts(ds, mdp, OracleQModel(mdp))
    assert abs(out["estimate"] - truth) < 1e-9
    assert np.all(np.abs(out["per_episode"] - truth) < 1e-9)
    assert truth == 1.0


def test_dr_unbiased_correct_props_any_q():
    """With true behaviour props, DR is unbiased even with a wrong Q."""
    mdp = make_good_coverage()
    truth, _, _ = dp_policy_value(mdp, mdp.target)
    rng = np.random.default_rng(0)
    wrong_Q = TabularQModel(mdp.n_states, mdp.n_actions, mdp.H, mdp.gamma)
    wrong_Q.Q[:] = rng.normal(0.3, 0.2, size=wrong_Q.Q.shape)
    ests = []
    for seed in range(80):
        ds = D.collect(mdp, 400, seed)
        ests.append(E.dr_parts(ds, mdp, wrong_Q)["estimate"])
    assert abs(np.mean(ests) - truth) < 0.05


def test_crossfit_no_episode_leakage_and_consistency():
    mdp = make_good_coverage()
    truth, _, _ = dp_policy_value(mdp, mdp.target)
    ests = []
    for seed in range(40):
        ds = D.collect(mdp, 600, seed)
        labels = ds.folds(5, seed=0)
        # each episode assigned exactly one fold
        assert set(np.unique(labels, return_counts=True)[0]) == set(range(5))
        models = fit_on_folds(ds, mdp, labels)
        out = E.crossfit_dr(ds, mdp, models, labels)
        ests.append(out["estimate"])
    assert abs(np.mean(ests) - truth) < 0.05


def test_crossfit_fold_sizes():
    mdp = make_good_coverage()
    ds = D.collect(mdp, 103, 0)
    labels = ds.folds(5, seed=42)
    counts = np.bincount(labels)
    assert counts.sum() == 103
    assert counts.max() - counts.min() <= 1


def test_weights_numerically_stable():
    mdp = make_tail_risk(eps=0.05)
    ds = D.collect(mdp, 200, 1)
    ell = W.cumulative_log_weights(ds)
    w = W.weights_from_log(ell)
    # max ratio (1000)^5 = 1e15; no overflow/NaN
    assert np.isfinite(w).all()
    diag = W.weight_diagnostics(ds)
    assert diag["ess_fraction_of_all"] < 0.05      # effective sample ~1 episode
    assert diag["frac_weight_top10_nz"] > 0.9
    # clipping cuts the tail and raises the post-clip ESS
    clipped_diag = W.weight_diagnostics(ds, clip=50.0)
    assert clipped_diag["ess_after_clip"] > diag["ess_nonzero"]


def test_tail_risk_pdis_variance_huge():
    mdp = make_tail_risk(eps=0.05)
    truth, _, _ = dp_policy_value(mdp, mdp.target)
    ests = [E.pdis_parts(D.collect(mdp, 300, s), mdp.gamma)["estimate"]
            for s in range(60)]
    # unbiased in expectation but catastrophic RMSE relative to clipping
    rmse = np.sqrt(np.mean((np.array(ests) - truth) ** 2))
    clip_ests = [E.pdis_parts(D.collect(mdp, 300, s), mdp.gamma, clip=20.0)["estimate"]
                 for s in range(60)]
    rmse_clip = np.sqrt(np.mean((np.array(clip_ests) - truth) ** 2))
    bias_clip = np.mean(clip_ests) - truth
    assert rmse > rmse_clip
    # clipping introduces a sign-known bias: the target's rare path is
    # upweighted, so clipping biases downward
    assert bias_clip < 0


def test_no_coverage_is_flagged_not_repaired():
    mdp0 = make_no_coverage(0.0)
    mdp1 = make_no_coverage(1.0)
    # identical behaviour-policy data distribution under both variants
    ds0 = D.collect(mdp0, 500, 0)
    ds1 = D.collect(mdp1, 500, 0)
    assert np.allclose(ds0.reward, ds1.reward)
    rep0 = D.identifiability_report(ds0, mdp0)
    rep1 = D.identifiability_report(ds1, mdp1)
    assert rep0["population_identifiable"] is False
    assert rep1["population_identifiable"] is False
    # different truths, impossible to separate from the logged data
    J0, _, _ = dp_policy_value(mdp0, mdp0.target)
    J1, _, _ = dp_policy_value(mdp1, mdp1.target)
    assert J0 != J1
    # DR with a model trained on the data cannot see the secret reward:
    # the missing (t=1,s=1,a=1) cell is NaN
    labels = ds0.folds(5, seed=0)
    models = fit_on_folds(ds0, mdp0, labels)
    for m in models.values():
        assert not np.isfinite(m.Q[1, 1, 1])
    out = E.crossfit_dr(ds0, mdp0, models, labels)
    # The missing cell is a target action never logged, so no DR correction
    # visits it (n_nan corrections == 0); the non-identifiability is carried
    # by the report flags, not by a magically inflated point estimate.
    assert out["n_nan"] == 0
    assert rep0["population_uncovered_mass"] > 0
    assert not rep0["population_identifiable"]
    # every data-driven estimate equals the observable component (0.3), i.e.
    # it silently misses the secret gamma * secret_reward term
    assert abs(out["estimate"] - 0.3) < 0.05


def test_leakage_reduces_residual_correction():
    """In-sample fitted Q overfits cells: DR correction shrinks toward zero
    on training data, masking model error compared with cross-fitting."""
    mdp = make_good_coverage()
    ds = D.collect(mdp, 300, 5)
    labels = ds.folds(5, seed=0)

    # action-blind wrong model
    cf_models = fit_on_folds(ds, mdp, labels, action_blind=True)
    cf = E.crossfit_dr(ds, mdp, cf_models, labels)

    in_model = TabularQModel(mdp.n_states, mdp.n_actions, mdp.H, mdp.gamma,
                             action_blind=True).fit(ds, mdp.target)
    in_sample = E.dr_parts(ds, mdp, in_model)

    # same wrong Q: in-sample residual corrections are shrunk toward zero
    cf_resid = cf["per_episode"] - cf["per_episode"].mean()
    in_resid = in_sample["per_episode"] - in_sample["per_episode"].mean()
    assert np.var(in_resid) < np.var(cf_resid)


def test_dr_reports_weighted_nan_support_gap():
    """tail_risk: rare target cells are poorly estimated; DR must expose the
    fraction of cumulative weight that lands on NaN corrections rather than
    silently zeroing them."""
    mdp = make_tail_risk(eps=0.05)
    ds = D.collect(mdp, 300, 0)
    labels = ds.folds(5, seed=0)
    models = fit_on_folds(ds, mdp, labels)
    out = E.crossfit_dr(ds, mdp, models, labels)
    assert "weighted_nan_fraction" in out
    assert out["weighted_nan_fraction"] >= 0.0
    # oracle model has no missing cells -> zero weighted NaN mass
    o = E.dr_parts(ds, mdp, OracleQModel(mdp))
    assert o["weighted_nan_fraction"] == 0.0
