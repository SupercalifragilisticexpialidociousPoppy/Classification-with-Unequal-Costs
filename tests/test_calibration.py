"""Checks of the C4 probability-quality metrics."""
import numpy as np
import pytest
from scipy.special import expit

from project4.calibration import (brier, ece, log_loss, logit, platt_apply, platt_fit,
                                  reliability_bins, temperature_scale)


def calibrated_sample(n, seed):
    """p ~ Uniform(0, 1) and y ~ Bernoulli(p): perfectly calibrated by construction."""
    rng = np.random.default_rng(seed)
    p = rng.random(n)
    y = (rng.random(n) < p).astype(int)
    return y, p


# ------------------------------------------------------------------ scores
def test_log_loss_hand_example_from_lecture_notes():
    # Notes, Section 13.2: y = (0.9, 0.2, 0.7), t = (1, 0, 1) gives E = 0.685 (a sum).
    assert log_loss([1, 0, 1], [0.9, 0.2, 0.7]) == pytest.approx(0.685 / 3, abs=1e-3)


def test_log_loss_clips_confident_mistakes():
    assert log_loss([1], [0.0]) == pytest.approx(-np.log(1e-12))
    assert np.isfinite(log_loss([0, 1], [1.0, 0.0]))


def test_brier_hand_values():
    assert brier([1, 0], [1.0, 0.0]) == 0.0
    assert brier([1, 0, 1, 0], [0.5] * 4) == pytest.approx(0.25)
    assert brier([1, 0, 1], [0.9, 0.2, 0.7]) == pytest.approx((0.01 + 0.04 + 0.09) / 3)


def test_scores_match_scikit_learn():
    sk = pytest.importorskip("sklearn.metrics")
    y, p = calibrated_sample(5000, seed=1)
    assert log_loss(y, p) == pytest.approx(sk.log_loss(y, p), rel=1e-9)
    assert brier(y, p) == pytest.approx(sk.brier_score_loss(y, p), rel=1e-12)


def test_input_checks():
    with pytest.raises(ValueError):
        log_loss([0, 1], [0.5])
    with pytest.raises(ValueError):
        brier([0, 2], [0.5, 0.5])
    with pytest.raises(ValueError):
        brier([0, 1], [0.5, 1.5])


# ------------------------------------------------------------- calibration
def test_uniform_bins_hand_example():
    y = [0, 0, 1, 1, 1]
    p = [0.05, 0.15, 0.15, 0.95, 1.0]       # p = 1 must land in the last bin
    b = reliability_bins(y, p, n_bins=10)
    assert b["count"].tolist() == [1, 2, 0, 0, 0, 0, 0, 0, 0, 2]
    assert b["mean_p"][1] == pytest.approx(0.15) and b["frac_pos"][1] == pytest.approx(0.5)
    assert np.isnan(b["mean_p"][2]) and np.isnan(b["frac_pos"][2])
    assert b["frac_pos"][9] == 1.0
    # ECE = (1/5)|0-0.05| + (2/5)|0.5-0.15| + (2/5)|1-0.975|
    assert ece(y, p) == pytest.approx(0.2 * 0.05 + 0.4 * 0.35 + 0.4 * 0.025)


def test_quantile_bins_have_equal_counts():
    y, p = calibrated_sample(1003, seed=2)
    b = reliability_bins(y, p, n_bins=10, strategy="quantile")
    assert b["count"].sum() == 1003 and b["count"].max() - b["count"].min() <= 1
    assert np.all(np.diff(b["mean_p"]) > 0)          # bins are ordered by p
    assert np.all(b["lo"] <= b["mean_p"]) and np.all(b["mean_p"] <= b["hi"])


def test_quantile_bins_keep_equal_p_together():
    # four emails all at p = 0.5, half spam: perfectly calibrated, whatever their order
    for y in ([1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0]):
        assert ece(y, [0.5] * 4, n_bins=2, strategy="quantile") == 0.0
    p = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.9])
    b = reliability_bins(np.zeros(6), p, n_bins=3, strategy="quantile")
    assert b["count"].tolist() == [5, 0, 1]           # the 0.2s are not split; middle bin empty
    assert b["hi"][0] < b["lo"][2]


def test_ece_small_for_calibrated_large_for_overconfident():
    y, p = calibrated_sample(200_000, seed=3)
    for strategy in ("uniform", "quantile"):
        assert ece(y, p, strategy=strategy) < 0.01
        assert ece(y, temperature_scale(p, 0.3), strategy=strategy) > 0.1


# --------------------------------------------- reshaping the probabilities
def test_temperature_keeps_labels_and_accuracy():
    y, p = calibrated_sample(10_000, seed=4)
    labels = p >= 0.5
    for T in (0.2, 0.5, 1.0, 2.0, 5.0):
        q = temperature_scale(p, T)
        assert np.array_equal(q >= 0.5, labels)
    assert temperature_scale(p, 1.0) == pytest.approx(p, abs=1e-9)
    assert np.all(np.abs(temperature_scale(p, 0.5) - 0.5) >= np.abs(p - 0.5) - 1e-12)


def test_log_loss_and_brier_are_best_at_the_true_probabilities():
    """Proper scoring rules: on calibrated data no temperature beats T = 1."""
    y, p = calibrated_sample(200_000, seed=5)
    Ts = np.exp(np.linspace(np.log(0.5), np.log(2.0), 41))
    for score in (log_loss, brier):
        values = [score(y, temperature_scale(p, T)) for T in Ts]
        assert Ts[int(np.argmin(values))] == pytest.approx(1.0, abs=0.08)


def test_platt_recovers_known_map():
    rng = np.random.default_rng(6)
    a = rng.normal(0, 2, 200_000)
    y = (rng.random(a.size) < expit(0.5 * a - 1.0)).astype(int)
    A, B = platt_fit(y, expit(a))
    assert A == pytest.approx(0.5, abs=0.02) and B == pytest.approx(-1.0, abs=0.02)
    assert log_loss(y, platt_apply(expit(a), A, B)) < log_loss(y, expit(a))


def test_platt_matches_scikit_learn_and_leaves_calibrated_alone():
    lm = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(7)
    a = rng.normal(0, 3, 5000)
    y = (rng.random(a.size) < expit(0.7 * a + 0.4)).astype(int)
    A, B = platt_fit(y, expit(a), l2=0.0)
    ref = lm.LogisticRegression(C=1e10, tol=1e-12, max_iter=10_000).fit(logit(expit(a))[:, None], y)
    assert A == pytest.approx(ref.coef_[0, 0], abs=1e-4)
    assert B == pytest.approx(ref.intercept_[0], abs=1e-4)
    y2, p2 = calibrated_sample(100_000, seed=8)
    A2, B2 = platt_fit(y2, p2)
    assert A2 == pytest.approx(1.0, abs=0.05) and B2 == pytest.approx(0.0, abs=0.05)
