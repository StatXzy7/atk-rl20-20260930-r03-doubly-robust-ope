"""Train/evaluation leakage counterexample.

Honest pipeline: for fold k, Q is fitted on the other K-1 folds and evaluated
on fold k. Leaky pipeline: Q is fitted on ALL episodes and evaluated on those
SAME episodes, so each trajectory's own (s,a,r,s') transitions appear in the
cell means used to evaluate it. With sparse (s,a) cells (small behavior prob
of the target action) and random rewards, the in-sample Bellman residuals
shrink artificially: DR point estimates look less variable and bootstrap
intervals become too narrow => over-optimistic, under-coverage.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ope.crossfit import (  # noqa: E402
    compute_weights,
    crossfit_q,
    episode_kfold,
    in_sample_q,
)
from ope.data import collect_episodes  # noqa: E402
from ope.estimators import dr  # noqa: E402
from ope.mdp import dp_evaluate  # noqa: E402
from ope.bootstrap import bootstrap_estimates  # noqa: E402

from scenarios import make_leakage  # noqa: E402


def dr_point(mdp, data, q_arrays):
    _, _, lp = compute_weights(mdp, data, None)
    q_sa, q_next, v0 = q_arrays
    return dr(data.rewards, lp, q_sa, q_next, v0, mdp.discount)


def main(n_episodes=120, n_replicates=120, n_boot=199, k_folds=4, seed=7, rare=0.10):
    mdp = make_leakage(rare)
    truth, _, _ = dp_evaluate(mdp)
    master = np.random.default_rng(seed)
    rows = {"honest_crossfit": {"est": [], "lo": [], "hi": [], "se": []},
            "leaky_insample": {"est": [], "lo": [], "hi": [], "se": []}}
    for _ in range(n_replicates):
        rng = np.random.default_rng(master.integers(0, 2**31 - 1))
        data = collect_episodes(mdp, n_episodes, rng)
        folds = episode_kfold(n_episodes, k_folds, rng)
        honest_q = crossfit_q(mdp, data, folds, mode="empirical")[:3]
        leaky_q = in_sample_q(mdp, data, mode="empirical")[:3]
        rows["honest_crossfit"]["est"].append(dr_point(mdp, data, honest_q))
        rows["leaky_insample"]["est"].append(dr_point(mdp, data, leaky_q))

        # bootstrap both pipelines; bootstrap refits nuisance from scratch
        hb = {"est": []}
        lb = {"est": []}
        for _b in range(n_boot):
            idx = rng.integers(0, n_episodes, n_episodes)
            boot = data.take(idx)
            bfolds = np.arange(n_episodes) % k_folds
            hb["est"].append(dr_point(mdp, boot, crossfit_q(mdp, boot, bfolds, "empirical")[:3]))
            lb["est"].append(dr_point(mdp, boot, in_sample_q(mdp, boot, "empirical")[:3]))
        for key, store in (("honest_crossfit", hb), ("leaky_insample", lb)):
            arr_b = np.asarray(store["est"])
            lo, hi = np.percentile(arr_b, [2.5, 97.5])
            rows[key]["lo"].append(lo)
            rows[key]["hi"].append(hi)
            rows[key]["se"].append(float(np.std(arr_b, ddof=1)))

    out = {"scenario": "leakage", "truth": truth,
           "config": {"n_episodes": n_episodes, "n_replicates": n_replicates,
                      "n_boot": n_boot, "k_folds": k_folds},
           "pipelines": {}}
    for key, d in rows.items():
        est = np.array(d["est"])
        lo, hi = np.array(d["lo"]), np.array(d["hi"])
        cover = (lo <= truth) & (truth <= hi)
        out["pipelines"][key] = {
            "mean": float(est.mean()),
            "bias": float((est - truth).mean()),
            "rmse": float(np.sqrt(np.mean((est - truth) ** 2))),
            "sd_estimates": float(est.std(ddof=1)),
            "coverage": float(cover.mean()),
            "coverage_mc_se": float(np.sqrt(cover.mean() * (1 - cover.mean()) / est.size)),
            "median_ci_width": float(np.median(hi - lo)),
            "mean_bootstrap_se": float(np.mean(d["se"])),
            "sd_of_estimates": float(est.std(ddof=1)),
        }
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "leakage.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
