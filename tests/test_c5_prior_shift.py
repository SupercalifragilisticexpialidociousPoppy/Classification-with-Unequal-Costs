"""C5 (Member 1): helpers of experiments/task4_prior_shift.py."""
import numpy as np
import pytest

from project4 import data
from project4.experiments import task4_prior_shift as c5

MU0, MU1, S = [0.0, 0.0], [1.5, 1.0], [[1.0, 0.3], [0.3, 1.0]]


@pytest.mark.parametrize("pi_new", [0.05, 0.2, 0.5, 0.8])
def test_em_recovers_the_deployment_prior_from_exact_posteriors(pi_new):
    # Exact posteriors at the training prior 0.5, evaluated on data drawn at pi_new.
    X, y = data.make_gaussian(50_000, pi_new, MU0, MU1, S, S, seed=1)
    p = data.gaussian_true_posterior(X, 0.5, MU0, MU1, S, S)
    pi_hat, n_iter = c5.em_prior(p, 0.5)
    assert pi_hat == pytest.approx(y.mean(), abs=0.01)
    assert n_iter < c5.EM_MAX_ITER


def test_em_stays_at_the_training_prior_without_shift():
    X, y = data.make_gaussian(50_000, 0.4, MU0, MU1, S, S, seed=2)
    p = data.gaussian_true_posterior(X, 0.4, MU0, MU1, S, S)
    assert c5.em_prior(p, 0.4)[0] == pytest.approx(0.4, abs=0.01)


def test_run_split_synthetic_smoke(monkeypatch):
    monkeypatch.setattr(c5, "DEPLOY_PRIORS", (0.05, 0.8))
    monkeypatch.setattr(c5, "CONFUSION_PRIOR", 0.05)
    records, confusions, em_rows = c5.run_split("synthetic", seed=0)
    assert {r["method"] for r in records} == set(c5.METHODS)
    assert len(records) == 2 * len(c5.MODELS) * len(c5.METHODS)
    assert len(em_rows) == 2 * len(c5.MODELS)
    assert set(confusions) == {(m, k) for m in c5.MODELS for k in c5.METHODS}
    for C in confusions.values():
        assert C.sum() == c5.SYNTH_N_DEPLOY
    loss = {(r["model"], r["pi1_deploy"], r["method"]): r["realized_loss"] for r in records}
    for m in c5.MODELS:
        for pi in (0.05, 0.8):
            # Far from the training prior 0.5, correcting must help, and nothing beats the oracle by much.
            assert loss[(m, pi, "corrected")] < loss[(m, pi, "uncorrected")]
            assert loss[(m, pi, "oracle")] <= loss[(m, pi, "corrected")] + 0.01
