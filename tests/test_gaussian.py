"""Tests for the Gaussian generative classifiers (C2)."""
import numpy as np
import pytest
from scipy.linalg import LinAlgError
from scipy.stats import multivariate_normal

from project4.data import gaussian_true_posterior, make_gaussian
from project4.models.gaussian import GaussianClassifier, select_reg

MU0, MU1 = np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.5, -0.5])
S_SHARED = np.array([[1.0, 0.3, 0.0], [0.3, 1.0, 0.2], [0.0, 0.2, 0.5]])
S1_OTHER = np.array([[2.0, -0.4, 0.1], [-0.4, 0.7, 0.0], [0.1, 0.0, 1.5]])
PI1 = 0.4


@pytest.fixture(scope="module")
def shared_data():
    return make_gaussian(60_000, PI1, MU0, MU1, S_SHARED, S_SHARED, seed=1)


@pytest.fixture(scope="module")
def separate_data():
    return make_gaussian(60_000, PI1, MU0, MU1, S_SHARED, S1_OTHER, seed=2)


# -------------------------------------------- recovers the true posterior
def test_shared_cov_matches_true_posterior(shared_data):
    X, y = shared_data
    p = GaussianClassifier(shared_cov=True).fit(X, y).predict_proba(X)
    p_true = gaussian_true_posterior(X, PI1, MU0, MU1, S_SHARED, S_SHARED)
    assert np.mean(np.abs(p - p_true)) < 0.005
    assert np.max(np.abs(p - p_true)) < 0.03


def test_separate_cov_matches_true_posterior(separate_data):
    X, y = separate_data
    p = GaussianClassifier(shared_cov=False).fit(X, y).predict_proba(X)
    p_true = gaussian_true_posterior(X, PI1, MU0, MU1, S_SHARED, S1_OTHER)
    err = np.abs(p - p_true)
    # Sampling error only; it is largest for the few points far out in the tails.
    assert err.mean() < 0.005 and np.quantile(err, 0.99) < 0.03


def test_parameters_are_maximum_likelihood(separate_data):
    X, y = separate_data
    m = GaussianClassifier(shared_cov=False).fit(X, y)
    assert np.isclose(m.priors_[1], y.mean())
    for k in (0, 1):
        Xk = X[y == k]
        assert np.allclose(m.means_[k], Xk.mean(0))
        assert np.allclose(m.covariances_[k], np.cov(Xk.T, bias=True))  # divides by n_k
    m2 = GaussianClassifier(shared_cov=True).fit(X, y)
    n0, n1 = (y == 0).sum(), (y == 1).sum()
    pooled = (n0 * m.covariances_[0] + n1 * m.covariances_[1]) / y.size
    assert np.allclose(m2.covariances_[0], pooled) and np.allclose(m2.covariances_[1], pooled)


def test_log_likelihood_matches_scipy(separate_data):
    X, y = separate_data
    m = GaussianClassifier(shared_cov=False, reg=0.1).fit(X[:5000], y[:5000])
    ll = m.log_likelihoods(X[:200])
    for k in (0, 1):
        ref = multivariate_normal(m.means_[k], m.covariances_[k]).logpdf(X[:200])
        assert np.allclose(ll[:, k], ref)


# --------------------------------------------------- scikit-learn cross-check
def test_shared_cov_matches_sklearn_lda(shared_data):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    X, y = shared_data[0][:3000], shared_data[1][:3000]
    p = GaussianClassifier(shared_cov=True).fit(X, y).predict_proba(X)
    # lsqr uses the ML pooled covariance and the empirical priors, as we do.
    ref = LinearDiscriminantAnalysis(solver="lsqr").fit(X, y).predict_proba(X)[:, 1]
    assert np.allclose(p, ref, atol=1e-8)


def test_separate_cov_matches_sklearn_qda(separate_data):
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

    X, y = separate_data[0][:3000], separate_data[1][:3000]
    q = QuadraticDiscriminantAnalysis(store_covariance=True).fit(X, y)
    ref = q.predict_proba(X)[:, 1]
    m = GaussianClassifier(shared_cov=False).fit(X, y)
    # scikit-learn estimates the class covariances slightly differently
    # (n_k - 1 denominator, SVD), so the fitted posteriors agree closely ...
    assert np.max(np.abs(m.predict_proba(X) - ref)) < 0.01
    # ... and with scikit-learn's own parameters plugged in they agree exactly,
    # which checks our log-density and posterior computation.
    for k in (0, 1):
        S = q.covariance_[k]
        m.means_[k] = q.means_[k]
        m.chol_[k] = np.linalg.cholesky(S)
        m.log_dets_[k] = 2 * np.log(np.diag(m.chol_[k])).sum()
    assert np.allclose(m.predict_proba(X), ref, atol=1e-6)


# ----------------------------------------------------------- structure
def test_shared_cov_log_odds_is_linear(shared_data):
    X, y = shared_data
    m = GaussianClassifier(shared_cov=True, reg=0.05).fit(X, y)
    w, b = m.linear_coefficients()
    assert np.allclose(m.decision_function(X[:500]), X[:500] @ w + b)
    with pytest.raises(ValueError):
        GaussianClassifier(shared_cov=False).fit(X, y).linear_coefficients()


def test_predict_proba_is_1d_probability(separate_data):
    X, y = separate_data
    p = GaussianClassifier(shared_cov=False).fit(X, y).predict_proba(X[:100])
    assert p.shape == (100,) and np.all((p >= 0) & (p <= 1))
    lp = GaussianClassifier(shared_cov=False).fit(X, y).predict_log_proba(X[:100])
    assert np.allclose(np.exp(lp).sum(1), 1) and np.allclose(np.exp(lp[:, 1]), p)


def test_one_dimensional_input():
    X, y = make_gaussian(5000, 0.3, -1.0, 1.0, 1.0, 1.0, seed=3)
    p = GaussianClassifier().fit(X[:, 0], y).predict_proba(X[:, 0])
    assert p.shape == (5000,)


def test_extreme_points_do_not_overflow(shared_data):
    X, y = shared_data
    m = GaussianClassifier(shared_cov=False).fit(X, y)
    far = np.array([[1e4, -1e4, 1e4], [-1e4, 1e4, -1e4]])
    p = m.predict_proba(far)
    lp = m.predict_log_proba(far)
    assert np.all(np.isfinite(p)) and np.all(np.isfinite(lp))


# ------------------------------------------------------- regularisation
def test_shrinkage_rule():
    S = np.array([[4.0, 1.0], [1.0, 2.0]])
    m = GaussianClassifier(reg=0.25)
    assert np.allclose(m._shrink(S), 0.75 * S + 0.25 * 3.0 * np.eye(2))
    assert np.allclose(GaussianClassifier(reg=1.0)._shrink(S), 3.0 * np.eye(2))
    with pytest.raises(ValueError):
        GaussianClassifier(reg=1.5)


def test_singular_covariance_needs_reg():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 4))
    X[:, 3] = 0.0                       # constant feature -> singular covariance
    y = (rng.random(400) < 0.5).astype(int)
    with pytest.raises(LinAlgError):
        GaussianClassifier(shared_cov=False, reg=0.0).fit(X, y)
    p = GaussianClassifier(shared_cov=False, reg=0.01).fit(X, y).predict_proba(X)
    assert np.all(np.isfinite(p))


def test_select_reg_uses_validation_and_skips_singular():
    rng = np.random.default_rng(1)
    X, y = make_gaussian(3000, 0.4, np.zeros(5), np.full(5, 0.6), np.eye(5), np.eye(5), seed=4)
    X = np.hstack([X, np.zeros((X.shape[0], 1))])   # singular without shrinkage
    r, model, scores = select_reg(X[:2000], y[:2000], X[2000:], y[2000:], shared_cov=False)
    assert 0.0 not in scores and r > 0
    assert scores[r] == min(scores.values())
    assert model.reg == r
