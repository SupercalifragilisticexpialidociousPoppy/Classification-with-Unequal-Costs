"""Gaussian generative classifiers (C2, Member 3; theory T2 by Member 2).

Model:  y ~ Bernoulli(pi1),  x | y=k ~ N(mu_k, S_k).

    shared_cov=True   one pooled covariance S for both classes (LDA): the
                      log-odds a(x) is linear in x.
    shared_cov=False  one covariance per class (QDA): a(x) is quadratic.

Fitting is by maximum likelihood (covariances divide by n_k, not n_k - 1),
followed by the T2 regularisation rule, shrinkage towards a scaled identity

    S_reg = (1 - reg) * S + reg * (tr(S) / d) * I,      0 <= reg <= 1,

with ``reg`` chosen on the validation part (``select_reg``).  Spambase needs
it: many features are almost always zero within a class, so the raw
covariances are singular or nearly so.

All densities are evaluated in the log domain through Cholesky factors
(no explicit matrix inverse) and the posterior is formed with
log-sum-exp / a stable sigmoid.

Interface (frozen):  GaussianClassifier(shared_cov, reg).fit(X, y).predict_proba(X)
returns P(y=1 | x) as a 1-D array.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import LinAlgError, cholesky, solve_triangular
from scipy.special import expit, logsumexp

LOG_2PI = np.log(2.0 * np.pi)
REG_GRID = (0.0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0)


class GaussianClassifier:
    """Two-class Gaussian generative classifier.

    Parameters
    ----------
    shared_cov : True for one pooled covariance (linear boundary),
        False for class-specific covariances (quadratic boundary).
    reg : shrinkage strength in [0, 1] towards (tr(S)/d) I.  0 is plain
        maximum likelihood; 1 replaces S by a multiple of the identity.

    Fitted attributes
    -----------------
    priors_ (2,), log_priors_ (2,), means_ (2, d), covariances_ (2, d, d)
    (both entries equal when shared_cov), chol_ (2, d, d) lower Cholesky
    factors, log_dets_ (2,) log-determinants of the covariances.
    """

    def __init__(self, shared_cov: bool = True, reg: float = 0.0):
        if not 0.0 <= reg <= 1.0:
            raise ValueError("reg must lie in [0, 1]")
        self.shared_cov = shared_cov
        self.reg = reg

    # ------------------------------------------------------------ fitting
    def _shrink(self, S: np.ndarray) -> np.ndarray:
        d = S.shape[0]
        nu = np.trace(S) / d
        return (1.0 - self.reg) * S + self.reg * nu * np.eye(d)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        y = np.asarray(y).astype(int).ravel()
        if X.shape[0] != y.size:
            raise ValueError("X and y differ in length")
        if not np.isin(y, (0, 1)).all():
            raise ValueError("labels must be 0 or 1")
        n, d = X.shape
        counts = np.array([(y == 0).sum(), (y == 1).sum()])
        if (counts == 0).any():
            raise ValueError("both classes must be present in the training data")

        self.priors_ = counts / n
        self.log_priors_ = np.log(self.priors_)
        self.means_ = np.vstack([X[y == k].mean(0) for k in (0, 1)])

        # Maximum-likelihood scatter of each class: S_k = (1/n_k) sum (x - mu_k)(x - mu_k)^T
        covs = []
        for k in (0, 1):
            D = X[y == k] - self.means_[k]
            covs.append(D.T @ D / counts[k])
        if self.shared_cov:
            pooled = (counts[0] * covs[0] + counts[1] * covs[1]) / n
            covs = [pooled, pooled]
        covs = [self._shrink(S) for S in covs]
        self.covariances_ = np.stack(covs)

        chols = []
        for k, S in enumerate(covs):
            try:
                chols.append(cholesky(S, lower=True))
            except LinAlgError as e:
                raise LinAlgError(
                    f"covariance of class {k} is singular; use reg > 0 "
                    "(shrinkage towards a scaled identity)"
                ) from e
        self.chol_ = np.stack(chols)
        self.log_dets_ = 2.0 * np.log(np.diagonal(self.chol_, axis1=1, axis2=2)).sum(1)
        self.n_features_ = d
        return self

    # ---------------------------------------------------------- inference
    def _check_X(self, X):
        if not hasattr(self, "chol_"):
            raise RuntimeError("call fit first")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None] if self.n_features_ == 1 else X[None, :]
        if X.shape[1] != self.n_features_:
            raise ValueError(f"expected {self.n_features_} features, got {X.shape[1]}")
        return X

    def log_likelihoods(self, X) -> np.ndarray:
        """log p(x | y=k) for k = 0, 1; shape (n, 2).

        With S = L L^T:  (x - mu)^T S^{-1} (x - mu) = ||L^{-1}(x - mu)||^2,
        computed by a triangular solve, and log|S| = 2 sum log diag(L).
        """
        X = self._check_X(X)
        d = self.n_features_
        out = np.empty((X.shape[0], 2))
        for k in (0, 1):
            z = solve_triangular(self.chol_[k], (X - self.means_[k]).T, lower=True)
            out[:, k] = -0.5 * (z**2).sum(0) - 0.5 * self.log_dets_[k] - 0.5 * d * LOG_2PI
        return out

    def joint_log_proba(self, X) -> np.ndarray:
        """log p(x, y=k) = log pi_k + log p(x | y=k); shape (n, 2)."""
        return self.log_likelihoods(X) + self.log_priors_

    def predict_log_proba(self, X) -> np.ndarray:
        """log P(y=k | x), shape (n, 2), normalised with log-sum-exp."""
        J = self.joint_log_proba(X)
        return J - logsumexp(J, axis=1, keepdims=True)

    def decision_function(self, X) -> np.ndarray:
        """Posterior log-odds a(x) = log P(y=1|x) - log P(y=0|x), 1-D.

        Linear in x when shared_cov=True, quadratic otherwise.  The rule
        "spam iff P(y=1|x) >= t" is "a(x) >= log(t / (1 - t))".
        """
        J = self.joint_log_proba(X)
        return J[:, 1] - J[:, 0]

    def predict_proba(self, X) -> np.ndarray:
        """P(y=1 | x) as a 1-D array (project-wide convention)."""
        return expit(self.decision_function(X))

    def predict(self, X, t: float = 0.5) -> np.ndarray:
        """Two-action labels: 1 iff P(y=1|x) >= t."""
        return (self.predict_proba(X) >= t).astype(int)

    # ------------------------------------------- linear form (shared cov)
    def linear_coefficients(self):
        """For shared_cov=True: (w, b) with a(x) = w^T x + b.

        w = S^{-1}(mu1 - mu0),  b = -1/2 mu1^T S^{-1} mu1 + 1/2 mu0^T S^{-1} mu0 + log(pi1/pi0).
        These play the role of the logistic-regression weights and bias (T2).
        Computed with two triangular solves, not an inverse.
        """
        if not self.shared_cov:
            raise ValueError("the log-odds is quadratic when shared_cov=False")
        Lc = self.chol_[0]

        def S_solve(v):
            return solve_triangular(Lc.T, solve_triangular(Lc, v, lower=True), lower=False)

        mu0, mu1 = self.means_
        w = S_solve(mu1 - mu0)
        b = -0.5 * mu1 @ S_solve(mu1) + 0.5 * mu0 @ S_solve(mu0) + self.log_priors_[1] - self.log_priors_[0]
        return w, float(b)


# ------------------------------------------------- choosing reg on validation
def _log_loss(y, p, eps=1e-12):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def select_reg(X_tr, y_tr, X_va, y_va, shared_cov: bool, grid=REG_GRID, score=None):
    """Choose the shrinkage strength on the validation part.

    Fits one model per value in ``grid`` on the training part and keeps the
    one with the lowest validation score.  The default score is log-loss,
    a strictly proper scoring rule, so this picks the regularisation that
    gives the best *probabilities* (which is what the cost-sensitive
    decisions need), not the best accuracy.  Values whose covariance stays
    singular are skipped.  Pass ``score(y, p)`` (lower is better) to use
    another criterion, e.g. validation realized loss.

    Returns (best_reg, fitted_model, scores) with scores a dict reg -> score.
    """
    score = score or _log_loss
    best, scores = None, {}
    for r in grid:
        try:
            m = GaussianClassifier(shared_cov=shared_cov, reg=r).fit(X_tr, y_tr)
        except LinAlgError:
            continue
        s = score(y_va, m.predict_proba(X_va))
        scores[r] = s
        if best is None or s < best[0]:
            best = (s, r, m)
    if best is None:
        raise LinAlgError("every value of reg gave a singular covariance")
    return best[1], best[2], scores
