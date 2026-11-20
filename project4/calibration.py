"""Probability-quality metrics (C4, Member 5).

Every function takes true labels ``y`` in {0, 1} and predicted probabilities
``p = P(y=1 | x)`` as 1-D arrays (the output of any model's ``predict_proba``).

Scores (lower is better)
    log_loss(y, p)            mean cross-entropy, the same error logistic regression minimises
    brier(y, p)               mean squared distance between p and the 0/1 label
Calibration
    reliability_bins(y, p)    per bin: number of emails, mean predicted p, observed spam fraction
    ece(y, p)                 expected calibration error: weighted gap between the last two
Changing a model's probabilities without changing its 0.5-labels
    temperature_scale(p, T)   p_T = sigma(logit(p) / T); T < 1 more confident, T > 1 less
Re-calibrating on validation data (optional part of C4)
    platt_fit(y, p) -> (A, B) a 1-D logistic regression of y on logit(p), fitted by Newton / IRLS
    platt_apply(p, A, B)      p' = sigma(A logit(p) + B)
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit

EPS = 1e-12  # probabilities are clipped to [EPS, 1 - EPS] before taking logs


def _check(y, p):
    y = np.asarray(y).astype(int).ravel()
    p = np.asarray(p, dtype=float).ravel()
    if y.shape != p.shape:
        raise ValueError(f"y and p differ in length: {y.shape} vs {p.shape}")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0 or 1")
    if np.isnan(p).any() or (p < 0).any() or (p > 1).any():
        raise ValueError("probabilities must lie in [0, 1]")
    return y, p


def logit(p, eps: float = EPS):
    """log(p / (1 - p)) after clipping p to [eps, 1 - eps] (so it is always finite)."""
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)


# ------------------------------------------------------------------ scores
def log_loss(y, p, eps: float = EPS) -> float:
    """Mean cross-entropy  -(1/N) sum_n [ y_n ln p_n + (1 - y_n) ln(1 - p_n) ].

    This is the logistic-regression error E(w) divided by N.  p is clipped to
    [eps, 1 - eps] so that a confident wrong answer (p = 0 for a spam email)
    costs about -ln(eps) = 27.6 instead of infinity.
    """
    y, p = _check(y, p)
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def brier(y, p) -> float:
    """Brier score  (1/N) sum_n (p_n - y_n)^2  (0 is perfect; always predicting 0.5 gives 0.25)."""
    y, p = _check(y, p)
    return float(np.mean((p - y) ** 2))


# ------------------------------------------------------------- calibration
def reliability_bins(y, p, n_bins: int = 10, strategy: str = "uniform") -> dict:
    """Group the predictions into bins and compare predicted and observed spam rates.

    strategy:
      "uniform"  - equal-width bins [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]
      "quantile" - equal-count bins: sort p and cut into n_bins groups of
                   (almost) the same size.  Equal p values always share a bin
                   (a cut that would split them moves past them), so a bin can
                   be larger than N / n_bins and a later one smaller or empty.

    Returns a dict of arrays, one entry per bin:
      count     number of predictions in the bin
      mean_p    average predicted probability   (NaN if the bin is empty)
      frac_pos  fraction that really are spam  (NaN if the bin is empty)
      lo, hi    smallest and largest p the bin can hold (uniform) or holds (quantile)
    A perfectly calibrated model has mean_p == frac_pos in every bin.
    """
    y, p = _check(y, p)
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.clip((p * n_bins).astype(int), 0, n_bins - 1)  # p = 1 goes in the last bin
        groups = [np.flatnonzero(idx == b) for b in range(n_bins)]
        lo, hi = edges[:-1], edges[1:]
    elif strategy == "quantile":
        order = np.argsort(p, kind="stable")
        ps = p[order]
        sizes = [len(g) for g in np.array_split(order, n_bins)]
        cuts = np.cumsum(sizes)[:-1]
        cuts = np.where(cuts > 0, np.searchsorted(ps, ps[np.maximum(cuts - 1, 0)], side="right"), 0)
        groups = np.split(order, cuts)
        lo = np.array([p[g].min() if g.size else np.nan for g in groups])
        hi = np.array([p[g].max() if g.size else np.nan for g in groups])
    else:
        raise ValueError("strategy must be 'uniform' or 'quantile'")
    count = np.array([g.size for g in groups])
    mean_p = np.array([p[g].mean() if g.size else np.nan for g in groups])
    frac_pos = np.array([y[g].mean() if g.size else np.nan for g in groups])
    return {"count": count, "mean_p": mean_p, "frac_pos": frac_pos, "lo": lo, "hi": hi}


def ece(y, p, n_bins: int = 10, strategy: str = "uniform") -> float:
    """Expected calibration error  sum_b (n_b / N) |frac_pos_b - mean_p_b|.

    The average gap between what the model says and what happens, weighted by
    how many predictions fall in each bin.  0 means perfectly calibrated
    (at this bin resolution).
    """
    b = reliability_bins(y, p, n_bins, strategy)
    full = b["count"] > 0
    w = b["count"][full] / b["count"].sum()
    return float(np.sum(w * np.abs(b["frac_pos"][full] - b["mean_p"][full])))


# --------------------------------------------- reshaping the probabilities
def temperature_scale(p, T: float, eps: float = EPS):
    """p_T = sigma(logit(p) / T) for a temperature T > 0.

    The log-odds are divided by T, so their sign never changes: every email
    stays on the same side of 0.5 and the 0.5-labels (hence accuracy) are
    identical for every T.  T < 1 pushes probabilities towards 0 and 1
    (over-confident), T > 1 pulls them towards 0.5 (under-confident).
    """
    if not T > 0:
        raise ValueError("temperature must be positive")
    return expit(logit(p, eps) / T)


def platt_fit(y, p, l2: float = 1e-6, max_iter: int = 100, tol: float = 1e-10, eps: float = EPS):
    """Fit p' = sigma(A * logit(p) + B) to labels y by maximum likelihood.

    This is logistic regression with the two features (logit(p), 1), solved
    with the Newton / IRLS update of the lectures:
        w <- w - (Phi^T R Phi + l2 I)^{-1} (Phi^T (y_hat - y) + l2 w).
    The tiny L2 term keeps the solution finite if the classes are separable.
    Fit on validation data only.  Returns (A, B); (1, 0) leaves p unchanged.
    """
    y, p = _check(y, p)
    Phi = np.column_stack([logit(p, eps), np.ones_like(p)])
    w = np.array([1.0, 0.0])

    def objective(w):
        a = Phi @ w
        # sum of log(1 + exp(-s a)) with s = 2y - 1, computed stably
        return np.logaddexp(0.0, -(2 * y - 1) * a).sum() + 0.5 * l2 * w @ w

    f = objective(w)
    for _ in range(max_iter):
        y_hat = expit(Phi @ w)
        grad = Phi.T @ (y_hat - y) + l2 * w
        R = y_hat * (1.0 - y_hat)
        H = Phi.T @ (Phi * R[:, None]) + l2 * np.eye(2)
        step = np.linalg.solve(H, grad)
        # damped Newton: halve the step until the objective does not increase
        for _ in range(30):
            w_new = w - step
            f_new = objective(w_new)
            if f_new <= f + 1e-12:
                break
            step = step / 2
        else:
            break                       # no step lowers the objective: w is the minimum
        w, f = w_new, f_new
        if np.linalg.norm(grad) <= tol * max(1.0, abs(f)):
            break                       # gradient is zero: converged
    return float(w[0]), float(w[1])


def platt_apply(p, A: float, B: float, eps: float = EPS):
    """Apply a fitted Platt map: p' = sigma(A * logit(p) + B)."""
    return expit(A * logit(p, eps) + B)
