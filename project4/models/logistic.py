"""Logistic regression from scratch (C3, Member 2).

Binary logistic regression, P(y = 1 | x) = sigma(w^T x + b), fitted by
Newton's method, which is IRLS (iteratively reweighted least squares), with an
L2 penalty on the weights.  The intercept is NOT penalised.

Objective minimised (sum convention, not mean):

    E(w, b) = - sum_n [ y_n log p_n + (1 - y_n) log(1 - p_n) ] + (l2 / 2) ||w||^2

so ``l2`` equals 1 / C in scikit-learn's ``LogisticRegression(C=...)``.  This
makes the cross-check against scikit-learn a one-line change of argument.

With s_n = p_n (1 - p_n), the gradient and Hessian (T3) are

    grad = X~^T (p - y) + l2 * [w; 0]
    H    = X~^T diag(s) X~ + l2 * diag(1, ..., 1, 0)

where X~ = [X, 1].  The Newton / IRLS step is  theta <- theta - H^{-1} grad.
For l2 > 0 the objective is strictly convex, so the minimiser is unique and
exists even for linearly separable data.  For l2 = 0 and separable data the
weights diverge; ``fit`` then stops at ``max_iter`` and warns.

Interface (frozen in Section 2 of the work plan):

    LogisticRegression(l2).fit(X, y).predict_proba(X)   # 1-D array of P(y=1|x)

Notes
-----
* Standardise features before fitting (fit the scaler on training data only).
  Newton is invariant to feature scaling in exact arithmetic, but the L2 penalty
  is not: it penalises all weights equally, so unscaled features get
  uneven shrinkage.
* Linear systems are solved with ``np.linalg.solve``; no explicit inverse.
"""
from __future__ import annotations

import warnings

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class LogisticRegression:
    """L2-penalised logistic regression fitted by Newton / IRLS.

    Parameters
    ----------
    l2 : strength of the L2 penalty on the weights (>= 0). Equals 1 / C in
        scikit-learn. The intercept is not penalised.
    max_iter : maximum number of Newton iterations.
    tol : stop when the Newton decrement  g^T H^{-1} g / 2  falls below ``tol``.

    Attributes (after ``fit``)
    --------------------------
    coef_ : weights, shape (d,).
    intercept_ : bias, float.
    n_iter_ : number of Newton iterations used.
    converged_ : True if the tolerance was reached before ``max_iter``.
    loss_ : penalised negative log-likelihood at the solution.
    """

    def __init__(self, l2: float = 1.0, max_iter: int = 100, tol: float = 1e-10):
        if not l2 >= 0:
            raise ValueError("l2 must be non-negative")
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        self.l2 = float(l2)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    # ------------------------------------------------------------------ fit
    def _objective(self, theta: np.ndarray, X1: np.ndarray, y: np.ndarray) -> float:
        z = X1 @ theta
        # sum_n [log(1 + e^z_n) - y_n z_n]  is the negative log-likelihood
        nll = np.sum(np.logaddexp(0.0, z) - y * z)
        return float(nll + 0.5 * self.l2 * np.dot(theta[:-1], theta[:-1]))

    def fit(self, X, y) -> "LogisticRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        if X.ndim != 2:
            raise ValueError("X must be 2-D (n_samples, n_features)")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y differ in length: {X.shape[0]} vs {y.shape[0]}")
        if not np.isin(y, (0, 1)).all():
            raise ValueError("labels must be 0 or 1")
        y = y.astype(float)
        n, d = X.shape

        X1 = np.hstack([X, np.ones((n, 1))])  # last column = intercept
        reg = np.full(d + 1, self.l2)
        reg[-1] = 0.0  # do not penalise the intercept

        # Start from the intercept-only solution: log-odds of the base rate.
        theta = np.zeros(d + 1)
        base = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        theta[-1] = np.log(base / (1.0 - base))

        f = self._objective(theta, X1, y)
        self.converged_ = False
        self.n_iter_ = 0
        for it in range(1, self.max_iter + 1):
            p = _sigmoid(X1 @ theta)
            grad = X1.T @ (p - y) + reg * theta
            s = p * (1.0 - p)
            H = (X1 * s[:, None]).T @ X1
            H[np.diag_indices(d + 1)] += reg
            # Tiny jitter keeps the solve well-posed when l2 = 0 and the
            # data are (nearly) separable or collinear.
            H[np.diag_indices(d + 1)] += 1e-12 * max(1.0, np.trace(H) / (d + 1))
            step = np.linalg.solve(H, grad)
            decrement = 0.5 * float(grad @ step)
            self.n_iter_ = it
            if decrement < self.tol:
                self.converged_ = True
                break

            # Backtracking line search (Armijo) guards against overshooting
            # far from the optimum; near the optimum t = 1 is always accepted.
            t = 1.0
            while t > 1e-10:
                f_new = self._objective(theta - t * step, X1, y)
                if f_new <= f - 1e-4 * t * float(grad @ step):
                    break
                t *= 0.5
            theta = theta - t * step
            f = self._objective(theta, X1, y)

        # On separable data with l2 = 0 the probabilities saturate at 0 / 1, so
        # the Newton decrement becomes tiny without the weights having reached
        # a minimiser (none exists).  Treat that as non-convergence.
        if self.l2 == 0.0 and self.converged_:
            p = _sigmoid(X1 @ theta)
            if np.max(np.abs(p - y)) < 1e-6:
                self.converged_ = False

        if not self.converged_:
            warnings.warn(
                f"LogisticRegression did not converge in {self.max_iter} iterations "
                "or the data are linearly separable with l2 = 0 (no finite MLE); "
                "increase l2.",
                RuntimeWarning,
                stacklevel=2,
            )

        self.coef_ = theta[:-1].copy()
        self.intercept_ = float(theta[-1])
        self.loss_ = f
        return self

    # -------------------------------------------------------------- predict
    def _check_fitted(self):
        if not hasattr(self, "coef_"):
            raise RuntimeError("call fit before predict_proba / decision_function")

    def decision_function(self, X) -> np.ndarray:
        """Logit a(x) = w^T x + b, 1-D array."""
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.coef_.shape[0]:
            raise ValueError(
                f"X must have shape (n, {self.coef_.shape[0]}), got {X.shape}"
            )
        return X @ self.coef_ + self.intercept_

    def predict_proba(self, X) -> np.ndarray:
        """P(y = 1 | x) as a 1-D array (project-wide convention)."""
        return _sigmoid(self.decision_function(X))
