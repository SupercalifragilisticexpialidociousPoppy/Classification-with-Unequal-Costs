"""Independent arithmetic and integration checks for the T4/C6 report evidence."""
import csv
import math
import statistics
from collections import defaultdict

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import expit, logit, rel_entr
from scipy.stats import norm

from project4 import data
from project4.analytic import GaussianModel1D
from project4.config import DEFAULT_LOSS, RESULTS_DIR, SEEDS
from project4.experiments.task5_inspection import DATASETS, INSPECTION_COSTS, MODELS, evaluate
from project4.models.gaussian import GaussianClassifier
from project4.models.logistic import LogisticRegression


def test_report_temperature_example():
    y = np.array([0] * 8 + [1] * 2 + [0] * 2 + [1] * 8)
    q = np.repeat([0.2, 0.8], 10)
    sharp = expit(3 * logit(q))
    assert sharp == pytest.approx(np.repeat([1 / 65, 64 / 65], 10))
    assert np.array_equal(q >= 0.5, sharp >= 0.5)
    for forecast, expected_cost in [(q, 0.25), (sharp, 1.10)]:
        rows = [r for r in evaluate(y, logit(forecast)) if r["c_insp"] == DEFAULT_LOSS.c_insp]
        assert next(r for r in rows if r["rule"] == "accuracy")["accuracy"] == 0.8
        assert next(r for r in rows if r["rule"] == "inspect")["cost"] == pytest.approx(expected_cost)


def test_expected_score_regrets_and_brier_grouping():
    for p in (0, 0.2, 0.75, 1):
        true_distribution = np.array([1 - p, p])
        mask = true_distribution > 0
        for q in (0.1, 0.2, 0.75, 0.9):
            forecast = np.array([1 - q, q])
            regret = true_distribution[mask] @ np.log(true_distribution[mask] / forecast[mask])
            assert regret == pytest.approx(rel_entr(true_distribution, forecast).sum())
            assert (regret > 0) == (q != p)
            assert (1 - p) * q**2 + p * (1 - q)**2 - p * (1 - p) == pytest.approx((p - q)**2)

    q = np.array([0.1, 0.2, 0.7, 0.8])
    y = np.array([0, 1, 0, 1])
    for groups in (q, q > 0.5):
        bins = [groups == g for g in np.unique(groups)]
        reliability = sum(b.mean() * (q[b].mean() - y[b].mean())**2 for b in bins)
        resolution = sum(b.mean() * (y[b].mean() - y.mean())**2 for b in bins)
        grouped_score = sum(np.sum((q[b].mean() - y[b])**2) for b in bins) / len(y)
        assert grouped_score == pytest.approx(reliability - resolution + y.var())
        if groups.dtype == bool:
            assert grouped_score != pytest.approx(np.mean((q - y)**2))


def test_c6_ties_and_free_inspection():
    # Include deterministic endpoints and ties at the positive-price log-odds cutoffs.
    logits = np.array([-np.inf, np.log(0.3) - np.log(0.7), 0, np.log(9.7) - np.log(0.3), np.inf])
    y = np.array([1, 1, 0, 0, 0])
    rows = {r["c_insp"]: r for r in evaluate(y, logits) if r["rule"] == "inspect"}
    assert (rows[0]["inspected"], rows[0]["cost"]) == (3, 11 / 5)
    assert (rows[0.3]["fp"], rows[0.3]["fn"], rows[0.3]["inspected"]) == (2, 2, 1)
    assert rows[10 / 11]["inspected"] == 0


def test_free_inspection_for_extreme_finite_logits():
    logits = np.array([-1000., -50., 50., 1000.])
    rows = evaluate([1, 1, 0, 0], logits)
    free = next(r for r in rows if r["rule"] == "inspect" and r["c_insp"] == 0)
    assert free["inspected"] == 4 and free["cost"] == 0


def test_synthetic_overlay_matches_integrated_class_densities():
    oracle = GaussianModel1D(mu=1, pi1=0.4)
    grid = np.array([0, 0.1, 0.3, 0.8, 10 / 11, 1.2])
    curves = oracle.curves_vs_c_insp(grid, 10, 1)
    for i, ci in enumerate(grid):
        def densities(x):
            real, spam = 0.6 * norm.pdf(x + 1), 0.4 * norm.pdf(x - 1)
            return spam, 10 * real, ci * (real + spam)

        # Split the integral at policy changes, where the rejection indicator jumps.
        thresholds = [ci, 1 - ci / 10] if ci < 10 / 11 else [10 / 11]
        points = [(logit(t) - math.log(2 / 3)) / 2 for t in thresholds if 0 < t < 1]

        def mass_inspected(x):
            deliver, spam, inspect = densities(x)
            return (0.6 * norm.pdf(x + 1) + 0.4 * norm.pdf(x - 1)
                    if inspect < min(deliver, spam) else 0)

        risk = quad(lambda x: min(densities(x)), -14, 14, points=points)[0]
        rejected = quad(mass_inspected, -14, 14, points=points)[0]
        assert curves["bayes_risk"][i] == pytest.approx(risk, abs=1e-10)
        assert curves["reject_rate"][i] == pytest.approx(rejected, abs=1e-10)


def test_saved_counts_paired_savings_and_error_bands():
    with (RESULTS_DIR / "c6_per_split.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    with (RESULTS_DIR / "c6_summary.csv").open(newline="") as f:
        summary = list(csv.DictReader(f))

    groups = defaultdict(list)
    baseline = {}
    for r in rows:
        key = r["dataset"], r["model"], r["rule"], float(r["c_insp"])
        groups[key].append(r)
        if r["rule"] == "two_action":
            baseline[r["dataset"], r["model"], r["seed"]] = float(r["cost"])
    assert len(rows) == len(DATASETS) * len(MODELS) * len(SEEDS) * (len(INSPECTION_COSTS) + 2)
    assert len(summary) == len(groups) == len(rows) // len(SEEDS)
    covered = set()
    for stats in summary:
        dataset, model, rule, ci = stats["dataset"], stats["model"], stats["rule"], float(stats["c_insp"])
        key = dataset, model, rule, ci
        assert key not in covered
        covered.add(key)
        group = groups[key]
        assert sorted(int(r["seed"]) for r in group) == list(SEEDS)
        assert int(stats["splits"]) == len(SEEDS)
        checked = defaultdict(list)
        for r in group:
            n, fp, fn, nr = (int(r[f]) for f in ("n", "fp", "fn", "inspected"))
            cost = (10 * fp + fn + ci * nr) / n
            reference = baseline[dataset, model, r["seed"]]
            checked["cost"].append(cost)
            checked["reject_rate"].append(nr / n)
            checked["saving"].append(reference - cost)
            if rule != "inspect":
                assert nr == 0
                checked["accuracy"].append((n - fp - fn) / n)
            elif ci >= 10 / 11:
                assert nr == 0 and cost == pytest.approx(reference)
            for field, values in checked.items():
                assert float(r[field]) == pytest.approx(values[-1], abs=1e-13)
        for field, values in checked.items():
            assert float(stats[field]) == pytest.approx(statistics.mean(values), abs=1e-13)
            assert float(stats[field + "_se"]) == pytest.approx(
                statistics.stdev(values) / math.sqrt(len(values)), abs=1e-13)


@pytest.mark.parametrize("dataset", DATASETS)
def test_saved_policies_against_refitted_conditional_loss_argmin(dataset):
    if dataset == "Spambase":
        if not (data.DATA_DIR / data.SPAMBASE_FILE).exists():
            pytest.skip("Spambase not cached")
        spambase = data.load_spambase()
    by_fit = defaultdict(list)
    with (RESULTS_DIR / "c6_per_split.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["dataset"] == dataset:
                by_fit[int(row["seed"]), row["model"]].append(row)
    assert set(by_fit) == {(seed, model) for seed in SEEDS for model in MODELS}
    for (seed, model), rows in by_fit.items():
        X, y = (data.make_gaussian(10_000, 0.4, -1, 1, 1, 1, 10_000 + seed)
                if dataset == "Synthetic" else spambase)
        ids = data.split(np.arange(len(y))[:, None], y, seed)
        train, valid, test = (part.ravel() for part in (ids.X_tr, ids.X_va, ids.X_te))
        np.testing.assert_array_equal(np.sort(np.r_[train, valid, test]), np.arange(len(y)))
        Z = (X - X[train].mean(0)) / np.maximum(X[train].std(0), 1e-12)
        penalties = {float(r["penalty"]) for r in rows}
        assert len(penalties) == 1
        penalty = penalties.pop()
        fit = GaussianClassifier(reg=penalty) if model == "Gaussian" else LogisticRegression(l2=penalty)
        z = fit.fit(Z[train], y[train]).decision_function(Z[test])
        log_deliver = -np.logaddexp(0, -z)
        log_spam = math.log(10) - np.logaddexp(0, z)
        for r in rows:
            if r["rule"] == "accuracy":
                actions = z >= 0
            elif r["rule"] == "two_action":
                actions = log_deliver >= log_spam
            else:
                price = float(r["c_insp"])
                log_review = math.log(price) if price else -math.inf
                log_costs = np.column_stack((log_deliver, log_spam, np.full_like(z, log_review)))
                actions = log_costs.argmin(axis=1)
            counts = (np.count_nonzero((actions == 1) & (y[test] == 0)),
                      np.count_nonzero((actions == 0) & (y[test] == 1)),
                      np.count_nonzero(actions == 2))
            assert counts == tuple(int(r[field]) for field in ("fp", "fn", "inspected"))
