"""Entry point: run the three MDP experiments and write results to results/."""
import json
import time

import numpy as np

from fh_ope.mdp import (make_good_coverage, make_tail_risk, make_no_coverage)
from fh_ope.experiments import run_environment
from fh_ope import data as D
from fh_ope.mdp import dp_policy_value


def main():
    np.set_printoptions(suppress=True)
    summary = {}

    t0 = time.time()
    summary["good_coverage"] = run_environment(
        make_good_coverage(), n_episodes=300, n_reps=100, K=5, B=100)
    print(f"[good_coverage] done in {time.time()-t0:.0f}s", flush=True)

    t1 = time.time()
    summary["tail_risk"] = run_environment(
        make_tail_risk(eps=0.05), n_episodes=300, n_reps=100, K=5, B=100,
        clips=(None, 20.0, 5.0))
    print(f"[tail_risk] done in {time.time()-t1:.0f}s", flush=True)

    # no-coverage: do NOT estimate; document non-identifiability and the two
    # observationally-equivalent truths
    nc = {}
    for secret, tag in [(0.0, "secret0"), (1.0, "secret1")]:
        m = make_no_coverage(secret)
        J, _, _ = dp_policy_value(m, m.target)
        ds = D.collect(m, 500, 0)
        rep = D.identifiability_report(ds, m)
        nc[tag] = dict(
            target_truth=J,
            population_identifiable=rep["population_identifiable"],
            population_uncovered_mass=rep["population_uncovered_mass"],
            uncovered_cells=rep["population_uncovered_cells"].tolist(),
        )
    nc["note"] = (
        "Both variants generate IDENTICAL logged behaviour-policy data but "
        "have different target values; no weighting/modeling can separate "
        "them. No point estimate or narrow interval is produced.")
    summary["no_coverage"] = nc

    with open("results/results.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    # human-readable table
    lines = []
    lines.append("# Experiment results\n")
    lines.append("Bootstrap: episode-level, percentile, B per replication; "
                 "90% nominal. Coverage is empirical over independent "
                 "re-collections, not a finite-sample guarantee.\n")
    for name, res in summary.items():
        if name == "no_coverage":
            continue
        lines.append(f"\n## {name}  (J={res['truth']:.4f}, "
                     f"n={res['n_episodes']}, reps={res['n_reps']}, "
                     f"mean ESS/n={res['mean_ess_fraction']:.4f})\n")
        lines.append("| estimator | bias | RMSE | coverage | mean CI width "
                     "| wNaNfrac | MCSE bias | note |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in res["rows"]:
            wn = r.get("mean_weighted_nan_fraction", float("nan"))
            lines.append(
                f"| {r['estimator']} | {r['bias']:+.4f} | {r['rmse']:.4f} "
                f"| {r['coverage']:.2f} | {r['mean_width']:.3f} "
                f"| {wn:.3f} | {r['mcse_bias']:.4f} | {r['note']} |")
    lines.append("\n## no_coverage (non-identifiable)\n")
    for k, v in nc.items():
        lines.append(f"- {k}: {v}")
    with open("results/results.md", "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines[:4]))
    print("wrote results/results.json and results/results.md")


if __name__ == "__main__":
    main()
