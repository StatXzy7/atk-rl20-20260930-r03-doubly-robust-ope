"""Repeated-data Monte Carlo study of PDIS, SNIS and sequential DR.

For each independent data re-collection we:
  1. collect n iid EPISODES under the behavior policy,
  2. split episodes into K folds and cross-fit tabular Q/V nuisance models,
  3. compute PDIS, trajectory SNIS, sequential DR and the plain model plug-in,
  4. run an episode-level bootstrap (whole trajectories resampled) for CIs.

Reported across replicates: bias, RMSE, 95% interval coverage, median width,
fraction of finite estimates, minimum normalized ESS, and Monte Carlo standard
errors for bias/RMSE/coverage. Clipping level is explicit. The "no_coverage"
scenario is not estimable: we only report the identification failure rate and
never fabricate a narrow interval.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ope.bootstrap import bootstrap_estimates  # noqa: E402
from ope.crossfit import episode_kfold, evaluate_all  # noqa: E402
from ope.data import collect_episodes  # noqa: E402

from scenarios import SCENARIOS  # noqa: E402

RESULT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def summarize(values: np.ndarray, truth: float) -> dict:
    v = np.asarray(values, float)
    finite = v[np.isfinite(v)]
    n = v.size
    out = {
        "n_replicates": int(n),
        "finite_frac": float(finite.size / n),
    }
    if finite.size == 0:
        out.update({"mean": None, "bias": None, "rmse": None, "median": None})
        return out
    err = finite - truth
    out["mean"] = float(finite.mean())
    out["median"] = float(np.median(finite))
    out["bias"] = float(err.mean())
    out["rmse"] = float(np.sqrt(np.mean(err**2)))
    # Monte Carlo standard errors (iid replicates)
    out["bias_mc_se"] = float(err.std(ddof=1) / np.sqrt(finite.size))
    out["rmse_mc_se"] = float(
        np.sqrt(np.var(err**2, ddof=1) / (4 * finite.size * max(out["rmse"], 1e-12) ** 2))
    )
    return out


def coverage_stats(lo, hi, truth):
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    finite = np.isfinite(lo) & np.isfinite(hi)
    n = lo.size
    if finite.sum() == 0:
        return {"coverage": None, "coverage_mc_se": None, "median_width": None, "finite_frac": 0.0}
    cov = (lo[finite] <= truth) & (truth <= hi[finite])
    p = float(cov.mean())
    return {
        "coverage": p,
        "coverage_mc_se": float(np.sqrt(p * (1 - p) / finite.sum())),
        "median_width": float(np.median(hi[finite] - lo[finite])),
        "finite_frac": float(finite.sum() / n),
    }


def run_scenario(name, n_episodes=300, n_replicates=120, n_boot=199,
                 k_folds=4, clip=None, q_mode="empirical", seed=0):
    mdp = SCENARIOS[name]()
    from ope.mdp import dp_evaluate
    truth, _, _ = dp_evaluate(mdp)

    methods = ["pdis", "snis", "dr", "plugin_v0"]
    point = {m: [] for m in methods}
    ci = {m: {"lo": [], "hi": []} for m in methods}
    min_ess, seen_rare, ess_steps = [], [], []
    t0 = time.time()
    master = np.random.default_rng(seed)

    for rep in range(n_replicates):
        rng = np.random.default_rng(master.integers(0, 2**31 - 1))
        data = collect_episodes(mdp, n_episodes, rng)
        folds = episode_kfold(n_episodes, k_folds, rng)
        res = evaluate_all(mdp, data, folds, clip=clip, q_mode=q_mode)
        min_ess.append(res.ess["min_ess_frac"])
        ess_steps.append(res.ess["ess_frac"])
        seen_rare.append(float(np.mean(np.exp(data.log_mu) < 0.01)))
        if res.support_violation:
            for m in methods:
                point[m].append(np.nan)
                ci[m]["lo"].append(np.nan)
                ci[m]["hi"].append(np.nan)
            continue
        for m in methods:
            point[m].append(res.estimates[m])
        boots = bootstrap_estimates(
            mdp, data, n_boot, rng, k_folds=k_folds, clip=clip, q_mode=q_mode)
        for m in ("pdis", "snis", "dr"):
            ci[m]["lo"].append(boots[m]["lo"])
            ci[m]["hi"].append(boots[m]["hi"])
        ci["plugin_v0"]["lo"].append(np.nan)
        ci["plugin_v0"]["hi"].append(np.nan)

    result = {
        "scenario": name,
        "truth": truth,
        "config": {
            "n_episodes": n_episodes,
            "n_replicates": n_replicates,
            "n_boot": n_boot,
            "k_folds": k_folds,
            "clip": clip,
            "q_mode": q_mode,
            "elapsed_sec": round(time.time() - t0, 1),
        },
        "min_ess_frac_median": float(np.median(min_ess)),
        "ess_frac_per_t_median": [float(x) for x in np.median(np.asarray(ess_steps), axis=0)],
        "fraction_small_mu_median": float(np.median(seen_rare)),
        "methods": {},
    }
    for m in methods:
        stats = summarize(np.array(point[m]), truth)
        if m != "plugin_v0":
            stats.update(coverage_stats(np.array(ci[m]["lo"]), np.array(ci[m]["hi"]), truth))
        result["methods"][m] = stats
    return result


def run_no_coverage(n_episodes=300, n_replicates=60, seed=10):
    """Identification diagnostic only: no point estimates, no intervals."""
    mdp = SCENARIOS["no_coverage"]()
    from ope.mdp import dp_evaluate
    truth, _, _ = dp_evaluate(mdp)
    from ope.crossfit import population_support_check
    missing = population_support_check(mdp)
    master = np.random.default_rng(seed)
    flags = 0
    for _ in range(n_replicates):
        data = collect_episodes(mdp, n_episodes, np.random.default_rng(master.integers(0, 2**31 - 1)))
        folds = episode_kfold(n_episodes, 4, np.random.default_rng(0))
        res = evaluate_all(mdp, data, folds)
        flags += int(res.support_violation)
    return {
        "scenario": "no_coverage",
        "truth": truth,
        "config": {"n_episodes": n_episodes, "n_replicates": n_replicates},
        "missing_target_support_pairs": [list(x) for x in missing],
        "identification_fail_detected_rate": flags / n_replicates,
        "report": (
            "Target reaches (s=0,a=1) with probability 1 but behavior assigns "
            "it probability 0. J(pi) is not identifiable from logged data. "
            "Estimators returned NaN and no interval is constructed; there is "
            "no epsilon smoothing."
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        n_rep, n_boot, n_ep = 30, 99, 200
    else:
        n_rep, n_boot, n_ep = 120, 199, 300

    os.makedirs(RESULT_DIR, exist_ok=True)
    jobs = [
        ("good_coverage", dict(n_episodes=n_ep, n_replicates=n_rep, n_boot=n_boot, clip=None, q_mode="empirical")),
        ("good_coverage_myopicQ", None),
        ("good_coverage_zeroQ", None),
        ("tiny_behavior", dict(n_episodes=n_ep, n_replicates=n_rep, n_boot=n_boot, clip=None, q_mode="empirical")),
        ("tiny_behavior_clip20", dict(n_episodes=n_ep, n_replicates=n_rep, n_boot=n_boot, clip=20.0, q_mode="empirical")),
        ("no_coverage", None),
    ]
    all_results = []
    for name, cfg in jobs:
        print(f"=== {name} ===", flush=True)
        if name == "no_coverage":
            r = run_no_coverage(n_episodes=n_ep, n_replicates=max(40, n_rep // 2))
        elif name == "good_coverage_myopicQ":
            r = run_scenario("good_coverage", n_episodes=n_ep, n_replicates=n_rep,
                             n_boot=n_boot, q_mode="myopic", seed=100)
            r["scenario"] = name
        elif name == "good_coverage_zeroQ":
            r = run_scenario("good_coverage", n_episodes=n_ep, n_replicates=n_rep,
                             n_boot=n_boot, q_mode="zero", seed=200)
            r["scenario"] = name
        else:
            if name == "tiny_behavior_clip20":
                r = run_scenario("tiny_behavior", **cfg)
                r["scenario"] = name
            else:
                r = run_scenario(name, **cfg)
        all_results.append(r)
        with open(os.path.join(RESULT_DIR, f"{name}.json"), "w") as f:
            json.dump(r, f, indent=2)
        print(json.dumps(r, indent=2)[:1200], flush=True)
    with open(os.path.join(RESULT_DIR, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
