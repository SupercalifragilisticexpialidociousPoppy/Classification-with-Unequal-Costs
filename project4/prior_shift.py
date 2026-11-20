"""Prior-shift correction (T5, Member 5) for use by the C5 experiments.

A model trained where a fraction pi1 of emails is spam outputs p = P(y=1 | x).
If at deployment the spam fraction is pi1_new but what spam and real email
look like, p(x | y), is unchanged (label shift), Bayes' rule gives

    P'(1|x) / P'(0|x)  =  P(1|x) / P(0|x)  *  (pi1_new / pi0_new) / (pi1 / pi0).

So the log-odds move by the constant

    delta = log(pi1_new / pi0_new) - log(pi1 / pi0),     p' = sigma(logit(p) + delta).

Two equivalent ways to use it (they give the same actions; only a p lying
exactly on a threshold can come out differently, through rounding):
    adjust_posterior(p, ...)  then bayes_action(p', L)      - correct the probability
    bayes_action_shifted(p, L, ...)                         - keep p, move the thresholds
"""
from __future__ import annotations

import numpy as np

from .decision import LossMatrix, band_action, thresholds


def _check_prior(pi1, name):
    if not 0.0 < pi1 < 1.0:
        raise ValueError(f"{name} must lie strictly between 0 and 1")


def log_odds_shift(pi1_train: float, pi1_new: float) -> float:
    """delta = log(pi1_new / pi0_new) - log(pi1_train / pi0_train)."""
    _check_prior(pi1_train, "pi1_train")
    _check_prior(pi1_new, "pi1_new")
    return float(np.log(pi1_new / (1 - pi1_new)) - np.log(pi1_train / (1 - pi1_train)))


def adjust_posterior(p, pi1_train: float, pi1_new: float):
    """Corrected posterior p' = p r / (p r + 1 - p) with r = exp(delta).

    Same as sigma(logit(p) + delta), written with odds so that p = 0 and
    p = 1 stay exactly 0 and 1.
    """
    r = np.exp(log_odds_shift(pi1_train, pi1_new))
    p = np.asarray(p, dtype=float)
    if not np.all((p >= 0.0) & (p <= 1.0)):  # also catches NaN
        raise ValueError("probabilities must lie in [0, 1]")
    return p * r / (p * r + (1.0 - p))


def shifted_thresholds(L: LossMatrix, pi1_train: float, pi1_new: float) -> tuple[float, float]:
    """Thresholds on the *uncorrected* p that give the corrected Bayes rule.

    Solving adjust_posterior(p) = theta for p gives theta / (theta + r (1 - theta)).
    """
    r = np.exp(log_odds_shift(pi1_train, pi1_new))
    return tuple(float(t / (t + r * (1.0 - t))) for t in thresholds(L))


def bayes_action_shifted(p, L: LossMatrix, pi1_train: float, pi1_new: float):
    """Bayes action at the new prior, computed from the uncorrected p by moving thresholds."""
    return band_action(p, *shifted_thresholds(L, pi1_train, pi1_new))
