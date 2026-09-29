# Experiment results

Bootstrap: episode-level, percentile, B per replication; 90% nominal. Coverage is empirical over independent re-collections, not a finite-sample guarantee.


## good_coverage  (J=1.8033, n=300, reps=100, mean ESS/n=0.4097)

| estimator | bias | RMSE | coverage | mean CI width | wNaNfrac | MCSE bias | note |
|---|---|---|---|---|---|---|---|
| PDIS_raw | +0.0066 | 0.0987 | 0.87 | 0.313 | nan | 0.0099 | unclipped |
| SNIS_raw | +0.0043 | 0.0640 | 0.84 | 0.206 | nan | 0.0064 | unclipped |
| DR_crossfit_raw | +0.0020 | 0.0572 | 0.86 | 0.183 | 0.000 | 0.0057 | unclipped |
| DR_crossfit_actionBlind_raw | +0.0037 | 0.0647 | 0.89 | 0.212 | 0.000 | 0.0065 | intentional misspecified Q (actions pooled) |
| DR_insample_LEAKY_raw | +0.0020 | 0.0565 | 0.87 | 0.185 | 0.000 | 0.0057 | train==eval leakage; overconfident intervals expected |
| PDIS_clip20 | +0.0066 | 0.0987 | 0.87 | 0.313 | nan | 0.0099 | cumulative-weight clipping: biased under correct props |
| SNIS_clip20 | +0.0043 | 0.0640 | 0.84 | 0.206 | nan | 0.0064 | cumulative-weight clipping: biased under correct props |
| DR_crossfit_clip20 | +0.0020 | 0.0572 | 0.86 | 0.183 | 0.000 | 0.0057 | cumulative-weight clipping: biased under correct props |
| DR_crossfit_actionBlind_clip20 | +0.0037 | 0.0647 | 0.89 | 0.212 | 0.000 | 0.0065 | intentional misspecified Q (actions pooled) |
| DR_insample_LEAKY_clip20 | +0.0020 | 0.0565 | 0.87 | 0.185 | 0.000 | 0.0057 | train==eval leakage; overconfident intervals expected |

## tail_risk  (J=2.9861, n=300, reps=100, mean ESS/n=0.0231)

| estimator | bias | RMSE | coverage | mean CI width | wNaNfrac | MCSE bias | note |
|---|---|---|---|---|---|---|---|
| PDIS_raw | -0.8813 | 1.8725 | 0.50 | 3.034 | nan | 0.1660 | unclipped |
| SNIS_raw | -0.1372 | 0.5269 | 0.78 | 1.040 | nan | 0.0511 | unclipped |
| DR_crossfit_raw | -1.3583 | 1.9031 | 0.04 | 1.009 | 0.076 | 0.1340 | unclipped |
| DR_crossfit_actionBlind_raw | -0.6564 | 1.7786 | 0.49 | 2.465 | 0.000 | 0.1661 | intentional misspecified Q (actions pooled) |
| DR_insample_LEAKY_raw | -1.2841 | 1.5786 | 0.07 | 0.970 | 0.000 | 0.0923 | train==eval leakage; overconfident intervals expected |
| PDIS_clip20 | -1.9081 | 1.9238 | 0.00 | 0.768 | nan | 0.0247 | cumulative-weight clipping: biased under correct props |
| SNIS_clip20 | -0.4121 | 0.4976 | 0.42 | 0.766 | nan | 0.0280 | cumulative-weight clipping: biased under correct props |
| DR_crossfit_clip20 | -1.4591 | 1.5386 | 0.02 | 0.740 | 0.030 | 0.0491 | cumulative-weight clipping: biased under correct props |
| DR_crossfit_actionBlind_clip20 | -1.5503 | 1.5663 | 0.00 | 0.687 | 0.000 | 0.0224 | intentional misspecified Q (actions pooled) |
| DR_insample_LEAKY_clip20 | -1.3296 | 1.4663 | 0.07 | 0.795 | 0.000 | 0.0621 | train==eval leakage; overconfident intervals expected |
| PDIS_clip5 | -2.5043 | 2.5059 | 0.00 | 0.278 | nan | 0.0090 | cumulative-weight clipping: biased under correct props |
| SNIS_clip5 | -0.9846 | 1.0032 | 0.00 | 0.524 | nan | 0.0193 | cumulative-weight clipping: biased under correct props |
| DR_crossfit_clip5 | -1.4596 | 1.5380 | 0.02 | 0.730 | 0.018 | 0.0487 | cumulative-weight clipping: biased under correct props |
| DR_crossfit_actionBlind_clip5 | -2.0423 | 2.0443 | 0.00 | 0.279 | 0.000 | 0.0090 | intentional misspecified Q (actions pooled) |
| DR_insample_LEAKY_clip5 | -1.3303 | 1.4660 | 0.08 | 0.786 | 0.000 | 0.0619 | train==eval leakage; overconfident intervals expected |

## no_coverage (non-identifiable)

- secret0: {'target_truth': 0.3, 'population_identifiable': False, 'population_uncovered_mass': 1.0, 'uncovered_cells': [[1, 1, 1]]}
- secret1: {'target_truth': 1.2, 'population_identifiable': False, 'population_uncovered_mass': 1.0, 'uncovered_cells': [[1, 1, 1]]}
- note: Both variants generate IDENTICAL logged behaviour-policy data but have different target values; no weighting/modeling can separate them. No point estimate or narrow interval is produced.