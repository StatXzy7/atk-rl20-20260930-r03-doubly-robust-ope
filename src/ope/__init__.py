"""Finite-horizon offline policy evaluation: PDIS, SNIS and sequential DR.

Sampling unit is the complete episode. Cross-fitting splits episodes, never
within-episode segments. All core estimators are implemented from scratch.
"""

from .mdp import FiniteHorizonMDP, Policy, dp_evaluate, BernoulliReward
from .data import EpisodeBatch, collect_episodes
from .weights import (
    log_policy_ratio,
    cumulative_log_ratio,
    clip_log_ratio,
    support_violations,
)
from .estimators import pdis, snis, dr, effective_sample_size
from .nuisance import TabularQ
from .crossfit import episode_kfold, crossfit_q, evaluate_all

__all__ = [
    "FiniteHorizonMDP",
    "Policy",
    "dp_evaluate",
    "BernoulliReward",
    "EpisodeBatch",
    "collect_episodes",
    "log_policy_ratio",
    "cumulative_log_ratio",
    "clip_log_ratio",
    "support_violations",
    "pdis",
    "snis",
    "dr",
    "effective_sample_size",
    "TabularQ",
    "episode_kfold",
    "crossfit_q",
    "evaluate_all",
]
