"""Unit tests for project4.models.logistic (C3).

Self-contained: synthetic data is generated here so the tests do not depend on
data.py.  The scikit-learn cross-check is skipped if scikit-learn is missing.
"""
import warnings

import numpy as np
import pytest

from project4.models.logistic import LogisticRegression, _sigmoid


def make_data(n=2000, d=4, seed=0, w_true=None, b_true=-0.5):
    """Data drawn from a true logistic model, so the fit has a known target."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    if w_true is None:
        w_true = np.linspace(1.5, -1.0, d)
    p = _sigmoid(X @ w_true + b_true)
    y = (rng.uniform(size=n) < p).astype(int)
    return X, y, np.asarray(w_true), b_true


def penalised_gradient(model, X, y):
    """Gradient of the penalised negative log-likelihood at the fitted solution."""
    p = _sigmoid(X @ model.coef_ + model.intercept_)
    g_w = X.T @ (p - y) + model.l2 * model.coef_
    g_b = np.sum(p - y)
    return np.append(g_w, g_b)


# ----------------------------------------------------------------- interface
def test_predict_proba_is_1d_in_unit_interval():
    X, y, _, _ = make_data()
    p = LogisticRegression(l2=1.0).fit(X, y).predict_proba(X)
    assert p.shape == (X.shape[0],)
    assert np.all((p >= 0) & (p <= 1))


def test_fit_returns_self():
    X, y, _, _ = make_data(n=200)
    m = LogisticRegression()
    assert m.fit(X, y) is m


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        LogisticRegression().predict_proba(np.zeros((3, 2)))


def test_input_validation():
    X, y, _, _ = make_data(n=50)
    with pytest.raises(ValueError):
        LogisticRegression(l2=-1.0)
    with pytest.raises(ValueError):
        LogisticRegression().fit(X, y + 1)  # labels in {1, 2}
    with pytest.raises(ValueError):
        LogisticRegression().fit(X, y[:-1])
    m = LogisticRegression().fit(X, y)
    with pytest.raises(ValueError):
        m.predict_proba(X[:, :-1])  # wrong number of features


def test_sigmoid_is_stable_for_extreme_logits():
    z = np.array([-1000.0, -50.0, 0.0, 50.0, 1000.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any overflow warning fails the test
        s = _sigmoid(z)
    assert np.allclose(s, [0.0, 0.0, 0.5, 1.0, 1.0])


# ------------------------------------------------------------- correctness
@pytest.mark.parametrize("l2", [0.0, 0.1, 1.0, 50.0])
def test_gradient_is_zero_at_solution(l2):
    """First-order optimality: independent of any reference implementation."""
    X, y, _, _ = make_data(n=1500, seed=1)
    m = LogisticRegression(l2=l2).fit(X, y)
    assert m.converged_
    # The stopping rule is on the Newton decrement (quadratic in the gradient),
    # so the gradient itself is only guaranteed to be small, not ~1e-10.
    assert np.max(np.abs(penalised_gradient(m, X, y))) < 1e-4


def test_solution_beats_random_perturbations():
    """The objective is convex, so the fitted point must be a global minimum."""
    X, y, _, _ = make_data(n=800, seed=2)
    m = LogisticRegression(l2=1.0).fit(X, y)
    X1 = np.hstack([X, np.ones((len(X), 1))])
    theta = np.append(m.coef_, m.intercept_)
    f_star = m._objective(theta, X1, y.astype(float))
    rng = np.random.default_rng(0)
    for _ in range(50):
        f_pert = m._objective(theta + 0.05 * rng.normal(size=theta.shape), X1, y.astype(float))
        assert f_pert > f_star


def test_newton_converges_in_few_iterations():
    X, y, _, _ = make_data(n=3000, seed=3)
    m = LogisticRegression(l2=1.0).fit(X, y)
    assert m.converged_
    assert m.n_iter_ <= 12  # quadratic convergence: typically 5-8 steps


def test_matches_scikit_learn():
    sk = pytest.importorskip("sklearn.linear_model")
    X, y, _, _ = make_data(n=2000, d=6, seed=4)
    for l2 in (0.1, 1.0, 10.0):
        ours = LogisticRegression(l2=l2).fit(X, y)
        theirs = sk.LogisticRegression(
            C=1.0 / l2, solver="lbfgs", tol=1e-12, max_iter=10000
        ).fit(X, y)
        assert np.allclose(ours.coef_, theirs.coef_.ravel(), atol=1e-5)
        assert np.isclose(ours.intercept_, theirs.intercept_[0], atol=1e-5)
        assert np.allclose(
            ours.predict_proba(X), theirs.predict_proba(X)[:, 1], atol=1e-6
        )


def test_recovers_true_model_with_large_sample():
    X, y, w_true, b_true = make_data(n=100_000, d=3, seed=5, w_true=[1.0, -2.0, 0.5])
    m = LogisticRegression(l2=1e-3).fit(X, y)
    assert np.allclose(m.coef_, w_true, atol=0.06)
    assert abs(m.intercept_ - b_true) < 0.06
    # fitted posteriors close to the true posteriors
    X_test = np.random.default_rng(99).normal(size=(2000, 3))
    p_true = _sigmoid(X_test @ w_true + b_true)
    assert np.max(np.abs(m.predict_proba(X_test) - p_true)) < 0.02


# ---------------------------------------------------- regularisation / bias
def test_intercept_is_not_penalised():
    """Huge l2 shrinks the weights to ~0, leaving the intercept at the base-rate logit."""
    X, y, _, _ = make_data(n=2000, seed=6)
    m = LogisticRegression(l2=1e9).fit(X, y)
    assert np.allclose(m.coef_, 0.0, atol=1e-4)
    base = y.mean()
    assert np.isclose(m.intercept_, np.log(base / (1 - base)), atol=1e-3)


def test_larger_l2_shrinks_weights():
    X, y, _, _ = make_data(n=1000, seed=7)
    norms = [np.linalg.norm(LogisticRegression(l2=l).fit(X, y).coef_) for l in (0.01, 1.0, 100.0)]
    assert norms[0] > norms[1] > norms[2]


def test_separable_data_needs_l2():
    """T3: on separable data the MLE does not exist; the L2 term restores it."""
    rng = np.random.default_rng(8)
    X = np.vstack([rng.normal(-3, 0.3, size=(50, 2)), rng.normal(3, 0.3, size=(50, 2))])
    y = np.r_[np.zeros(50), np.ones(50)].astype(int)

    # with a penalty: converges, finite weights, perfect separation
    m = LogisticRegression(l2=1.0).fit(X, y)
    assert m.converged_
    assert np.all(np.isfinite(m.coef_)) and np.linalg.norm(m.coef_) < 50
    assert np.all((m.predict_proba(X) >= 0.5) == y)

    # without: weights blow up and the model warns that it did not converge
    with pytest.warns(RuntimeWarning, match="separable"):
        m0 = LogisticRegression(l2=0.0, max_iter=30).fit(X, y)
    assert np.linalg.norm(m0.coef_) > 5 * np.linalg.norm(m.coef_)


def test_probabilities_are_calibrated_on_training_set():
    """With a tiny penalty, mean predicted probability equals the base rate
    (the intercept's gradient condition sum_n (p_n - y_n) = 0)."""
    X, y, _, _ = make_data(n=2000, seed=9)
    m = LogisticRegression(l2=1e-6).fit(X, y)
    assert np.isclose(m.predict_proba(X).mean(), y.mean(), atol=1e-8)
