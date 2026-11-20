"""Numerical checks of the T5 prior-shift correction."""
import numpy as np
import pytest
from scipy.special import expit

from project4.data import gaussian_true_posterior, make_gaussian, resample_to_prior
from project4.decision import LossMatrix, bayes_action
from project4.metrics import realized_loss
from project4.prior_shift import (adjust_posterior, bayes_action_shifted, log_odds_shift,
                                  shifted_thresholds)


def test_worked_example_in_report():
    # trained at pi1 = 0.5, deployed at pi1' = 0.1: odds are divided by 9
    assert log_odds_shift(0.5, 0.1) == pytest.approx(np.log(1 / 9))
    assert adjust_posterior(0.9, 0.5, 0.1) == pytest.approx(0.5)
    assert adjust_posterior(0.98, 0.5, 0.1) == pytest.approx(0.845, abs=5e-4)       # spam -> inspect
    L = LossMatrix(10, 1, 0.3)
    assert bayes_action([0.98], L)[0] == 1 and bayes_action([adjust_posterior(0.98, 0.5, 0.1)], L)[0] == 2
    t0, t1 = shifted_thresholds(L, 0.5, 0.1)
    assert (round(t0, 2), round(t1, 3)) == (0.79, 0.997)


def test_no_shift_and_round_trip():
    p = np.linspace(0, 1, 101)
    assert adjust_posterior(p, 0.4, 0.4) == pytest.approx(p)
    assert adjust_posterior(adjust_posterior(p, 0.4, 0.05), 0.05, 0.4) == pytest.approx(p)
    assert adjust_posterior([0.0, 1.0], 0.5, 0.2).tolist() == [0.0, 1.0]
    with pytest.raises(ValueError):
        adjust_posterior(p, 0.0, 0.5)
    for bad in ([1.5], [-0.1], [np.nan]):
        with pytest.raises(ValueError):
            adjust_posterior(bad, 0.5, 0.2)


@pytest.mark.parametrize("pi1_new", [0.05, 0.1, 0.2, 0.4, 0.6, 0.8])
def test_correction_is_exact_for_gaussian_classes(pi1_new):
    """Correcting the true posterior at pi1 = 0.5 gives the true posterior at pi1_new,
    both for shared and for class-specific covariances."""
    X, _ = make_gaussian(2000, 0.5, [0, 0], [1, 1], np.eye(2), [[2, 0.5], [0.5, 1]], seed=0)
    for S1 in (np.eye(2), np.array([[2, 0.5], [0.5, 1]])):
        p_train = gaussian_true_posterior(X, 0.5, [0, 0], [1, 1], np.eye(2), S1)
        p_new = gaussian_true_posterior(X, pi1_new, [0, 0], [1, 1], np.eye(2), S1)
        assert adjust_posterior(p_train, 0.5, pi1_new) == pytest.approx(p_new, rel=1e-9, abs=1e-12)


def test_logistic_regression_form_is_an_intercept_shift():
    rng = np.random.default_rng(1)
    Phi = np.column_stack([np.ones(500), rng.normal(size=(500, 3))])   # constant feature first
    w = rng.normal(size=4)
    delta = log_odds_shift(0.39, 0.1)
    w_new = w.copy()
    w_new[0] += delta                                                 # only w0 changes
    assert expit(Phi @ w_new) == pytest.approx(adjust_posterior(expit(Phi @ w), 0.39, 0.1))


def test_refitting_at_a_new_prior_moves_only_the_intercept():
    """Fit logistic regression on Gaussian data (shared covariance, so the model is
    correct) at two spam shares: the slopes agree and the intercepts differ by delta."""
    from sklearn.linear_model import LogisticRegression
    mu0, mu1, S = [0.0, 0.0], [1.0, 0.5], [[1.0, 0.3], [0.3, 1.0]]
    X, y = make_gaussian(400_000, 0.5, mu0, mu1, S, S, seed=5)
    fits = {}
    for pi1 in (0.5, 0.15):
        Xs, ys = resample_to_prior(X, y, pi1, seed=6)
        m = LogisticRegression(penalty=None, max_iter=1000).fit(Xs, ys)
        fits[pi1] = (m.intercept_[0], m.coef_[0], ys.mean())
    (b5, w5, s5), (b15, w15, s15) = fits[0.5], fits[0.15]
    assert w15 == pytest.approx(w5, abs=0.03)
    assert b15 - b5 == pytest.approx(log_odds_shift(s5, s15), abs=0.03)


@pytest.mark.parametrize("c_insp", [0.05, 0.3, 0.9, 2.0])   # with and without an inspect band
def test_moving_thresholds_equals_correcting_probabilities(c_insp):
    L = LossMatrix(10, 1, c_insp)
    p = np.random.default_rng(2).random(100_000)
    for pi1_train, pi1_new in ((0.5, 0.05), (0.39, 0.8), (0.3, 0.3)):
        a1 = bayes_action(adjust_posterior(p, pi1_train, pi1_new), L)
        a2 = bayes_action_shifted(p, L, pi1_train, pi1_new)
        assert np.array_equal(a1, a2)
        t0, t1 = shifted_thresholds(L, pi1_train, pi1_new)
        assert 0 < t0 <= t1 < 1


def test_correction_lowers_realized_loss_after_shift():
    """Monte Carlo: train-prior decisions lose more than corrected ones at a new prior."""
    mu0, mu1, S = [0.0], [1.5], [[1.0]]
    X, y = make_gaussian(200_000, 0.5, mu0, mu1, S, S, seed=3)
    L = LossMatrix(10, 1, 0.3)
    for pi1_new in (0.05, 0.8):
        Xd, yd = resample_to_prior(X, y, pi1_new, seed=4)
        p = gaussian_true_posterior(Xd, 0.5, mu0, mu1, S, S)          # model trained at 0.5
        oracle = gaussian_true_posterior(Xd, pi1_new, mu0, mu1, S, S)
        corrected = bayes_action(adjust_posterior(p, 0.5, pi1_new), L)
        assert np.array_equal(corrected, bayes_action(oracle, L))
        assert realized_loss(yd, corrected, L) < realized_loss(yd, bayes_action(p, L), L)
