"""Repeated-sampling Monte Carlo experiment.

For every independent dataset replication we fit the three estimators (plus
controls), build episode-level bootstrap intervals, and then aggregate over
replications:

    bias  = mean_r est_r - J
    rmse  = sqrt(mean_r (est_r - J)^2)
    cover = mean_r 1{lo_r <= J <= hi_r}
    mcse(bias) = sd_r(est_r)/sqrt(R)
    mcse(rmse) = sd_r((est_r-J)^2) / (2 rmse sqrt(R))
    mcse(cover) = sqrt(cover(1-cover)/R)

Bootstrap intervals are empirical frequentist uncertainty intervals; with
tilted/estimated weights and estimated nuisance they are not exact finite-
sample confidence intervals (coverage is measured, not assumed).
"""
from __future__ import annotations

import numpy as np

from . import data as D
from . import estimators as E
from . import weights as W
from .mdp import dp_policy_value
from .models import TabularQModel, fit_on_folds


def _mcse(values: np.ndarray) -> float:
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def run_environment(mdp, *, n_episodes: int, n_reps: int, K: int = 5,
                    B: int = 200, clips=(None, 20.0), alpha: float = 0.1,
                    include_dr_wrong: bool = True,
                    include_leakage: bool = True,
                    seed0: int = 1000) -> dict:
    truth, _, _ = dp_policy_value(mdp, mdp.target)
    rows = []

    def record(name, ests, los, his, note=""):
        ests = np.asarray(ests, float)
        los = np.asarray(los, float)
        his = np.asarray(his, float)
        err = ests - truth
        cover = (los <= truth) & (truth <= his)
        row = dict(
            estimator=name,
            bias=float(err.mean()),
            rmse=float(np.sqrt((err ** 2).mean())),
            coverage=float(cover.mean()),
            mcse_bias=_mcse(err),
            mcse_rmse=_mcse_rmse(err),
            mcse_coverage=float(np.sqrt(cover.mean() * (1 - cover.mean())
                                        / len(cover))),
            mean_width=float((his - los).mean()),
            n_reps=len(ests),
            note=note,
        )
        rows.append(row)
        return row

    def _mcse_rmse(err):
        sq = err ** 2
        rmse = np.sqrt(sq.mean())
        return float(sq.std(ddof=1) / (2 * rmse * np.sqrt(len(sq)))
                     if rmse > 0 else 0.0)

    for clip in clips:
        tag = "raw" if clip is None else f"clip{clip:g}"
        pdis_est, pdis_lo, pdis_hi = [], [], []
        snis_est, snis_lo, snis_hi = [], [], []
        dr_est, dr_lo, dr_hi = [], [], []
        dr_nanfrac = []
        if include_dr_wrong:
            drw_est, drw_lo, drw_hi = [], [], []
            drw_nanfrac = []
        if include_leakage:
            leak_est, leak_lo, leak_hi = [], [], []
            leak_nanfrac = []
        ess_list = []
        for r in range(n_reps):
            seed = seed0 + r
            ds = D.collect(mdp, n_episodes, seed)
            labels = ds.folds(K, seed=seed)
            models = fit_on_folds(ds, mdp, labels)
            diag = W.weight_diagnostics(ds, clip=clip)
            ess_list.append(diag["ess_fraction_of_all"])

            p = E.pdis_parts(ds, mdp.gamma, clip)
            pdis_est.append(p["estimate"])
            s = E.snis_parts(ds, mdp.gamma, clip)
            snis_est.append(s["estimate"])
            drcf = E.crossfit_dr(ds, mdp, models, labels, clip=clip)
            dr_est.append(drcf["estimate"])
            dr_nanfrac.append(drcf.get("weighted_nan_fraction", 0.0))

            rng = np.random.default_rng(seed)
            n = n_episodes
            # bootstrap PDIS
            draws_p = np.array([p["per_episode"][rng.integers(0, n, n)].mean()
                                for _ in range(B)])
            pdis_lo.append(np.quantile(draws_p, alpha / 2))
            pdis_hi.append(np.quantile(draws_p, 1 - alpha / 2))
            # bootstrap SNIS (recompute ratio)
            draws_s = np.empty(B)
            for b in range(B):
                ix = rng.integers(0, n, n)
                draws_s[b] = E.combine_snis(s, ix)
            snis_lo.append(np.quantile(draws_s, alpha / 2))
            snis_hi.append(np.quantile(draws_s, 1 - alpha / 2))
            # bootstrap DR with refitting inside each draw
            draws_d = _bootstrap_dr(ds, mdp, rng, B, K, clip, wrong=False)
            dr_lo.append(np.quantile(draws_d, alpha / 2))
            dr_hi.append(np.quantile(draws_d, 1 - alpha / 2))

            if include_dr_wrong:
                models_w = fit_on_folds(ds, mdp, labels, action_blind=True)
                drw = E.crossfit_dr(ds, mdp, models_w, labels, clip=clip)
                drw_est.append(drw["estimate"])
                drw_nanfrac.append(drw.get("weighted_nan_fraction", 0.0))
                draws_w = _bootstrap_dr(ds, mdp, rng, B, K, clip, wrong=True)
                drw_lo.append(np.quantile(draws_w, alpha / 2))
                drw_hi.append(np.quantile(draws_w, 1 - alpha / 2))

            if include_leakage:
                in_model = TabularQModel(
                    mdp.n_states, mdp.n_actions, mdp.H, mdp.gamma).fit(ds, mdp.target)
                leak = E.dr_parts(ds, mdp, in_model, clip=clip)
                leak_est.append(leak["estimate"])
                leak_nanfrac.append(leak.get("weighted_nan_fraction", 0.0))
                draws_l = _bootstrap_leak(ds, mdp, rng, B, clip)
                leak_lo.append(np.quantile(draws_l, alpha / 2))
                leak_hi.append(np.quantile(draws_l, 1 - alpha / 2))

        note_clip = "cumulative-weight clipping: biased under correct props" \
            if clip is not None else "unclipped"
        record(f"PDIS_{tag}", pdis_est, pdis_lo, pdis_hi, note_clip)
        record(f"SNIS_{tag}", snis_est, snis_lo, snis_hi, note_clip)
        row_dr = record(f"DR_crossfit_{tag}", dr_est, dr_lo, dr_hi, note_clip)
        rows[-1]["mean_weighted_nan_fraction"] = float(np.mean(dr_nanfrac))
        if include_dr_wrong:
            record(f"DR_crossfit_actionBlind_{tag}", drw_est, drw_lo, drw_hi,
                   "intentional misspecified Q (actions pooled)")
            rows[-1]["mean_weighted_nan_fraction"] = float(np.mean(drw_nanfrac))
        if include_leakage:
            record(f"DR_insample_LEAKY_{tag}", leak_est, leak_lo, leak_hi,
                   "train==eval leakage; overconfident intervals expected")
            rows[-1]["mean_weighted_nan_fraction"] = float(np.mean(leak_nanfrac))
        # attach mean ESS only on first clip row set
        if clip is None:
            mean_ess = float(np.mean(ess_list))
            for row in rows:
                row.setdefault("mean_ess_fraction", mean_ess)

    return dict(mdp=mdp.name, truth=truth, n_episodes=n_episodes,
                n_reps=n_reps, B=B, alpha=alpha,
                mean_ess_fraction=float(np.mean(ess_list)),
                rows=rows)


def _bootstrap_dr(ds, mdp, rng, B, K, clip, wrong):
    n = ds.n_episodes
    draws = np.empty(B)
    for b in range(B):
        ix = rng.integers(0, n, n)
        sub = D.EpisodeDataset(
            mdp_name=ds.mdp_name, H=ds.H, n_episodes=n,
            state=ds.state[ix], action=ds.action[ix], reward=ds.reward[ix],
            next_state=ds.next_state[ix], done=ds.done[ix], mask=ds.mask[ix],
            p_behavior=ds.p_behavior[ix], p_target=ds.p_target[ix])
        labels = sub.folds(K, seed=int(rng.integers(1 << 30)))
        models = fit_on_folds(sub, mdp, labels, action_blind=wrong)
        draws[b] = E.crossfit_dr(sub, mdp, models, labels, clip=clip)["estimate"]
    return draws


def _bootstrap_leak(ds, mdp, rng, B, clip):
    n = ds.n_episodes
    draws = np.empty(B)
    for b in range(B):
        ix = rng.integers(0, n, n)
        sub = D.EpisodeDataset(
            mdp_name=ds.mdp_name, H=ds.H, n_episodes=n,
            state=ds.state[ix], action=ds.action[ix], reward=ds.reward[ix],
            next_state=ds.next_state[ix], done=ds.done[ix], mask=ds.mask[ix],
            p_behavior=ds.p_behavior[ix], p_target=ds.p_target[ix])
        model = TabularQModel(
            mdp.n_states, mdp.n_actions, mdp.H, mdp.gamma).fit(sub, mdp.target)
        draws[b] = E.dr_parts(sub, mdp, model, clip=clip)["estimate"]
    return draws
