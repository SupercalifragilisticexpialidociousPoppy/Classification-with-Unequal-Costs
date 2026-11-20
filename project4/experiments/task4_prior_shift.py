"""Task 4 experiment (C5, Member 1): decisions after a prior shift.

A filter is trained where a share pi1 of emails is spam and deployed where the
share is pi1' while p(x | y) stays the same (label shift).  For every split
(seeds 0-19), model and deployment prior this script compares

    uncorrected  Bayes action on the model's p (trained at pi1)
    corrected    Bayes action on p' = adjust_posterior(p, pi1, pi1')       (T5, known pi1')
    estimated    the same, with pi1' estimated from the unlabelled deployment
                 emails by EM (Saerens et al., 2002)                         (pi1' unknown)
    oracle       Bayes action on the true posterior at pi1'                 (synthetic only)

and records realized loss, accuracy, inspection rate and the confusion matrix.

Data
  synthetic  two 5-D Gaussians (same class densities as Task 2).  Train/validation
             come from a sample with pi1 = 0.5; every deployment set is a fresh
             sample of SYNTH_N_DEPLOY emails drawn at pi1', so p(x | y) is exactly unchanged.
  spambase   train/validation as usual (pi1 = 39.4%); deployment sets are the test
             part resampled within each class to pi1' (data.resample_to_prior,
             without replacement, largest set the test part allows).

Models: Gaussian with shared covariance (shrinkage chosen on validation log-loss,
models.gaussian.select_reg) and logistic regression (L2 chosen on validation
log-loss).  The test part is never used to choose anything.

Outputs (means with standard errors over the 20 splits)
  figures/c5_prior_shift.{pdf,png}       realized loss against deployment prior
  results/c5_per_split.csv               one row per dataset, seed, model, prior, method
  results/c5_summary.csv                 means and standard errors
  results/c5_loss_<dataset>.tex          LaTeX table of mean loss (tabular only)
  results/c5_confusion.tex               mean confusion matrices at CONFUSION_PRIOR
  results/c5_em_priors.csv               EM estimates of pi1' against the truth

Run:  python -m project4.experiments.task4_prior_shift  [--seeds 3]
"""
from __future__ import annotations

import argparse
import csv
import warnings
from collections import defaultdict

import numpy as np

from project4 import data
from project4.config import DEFAULT_LOSS, FIGURES_DIR, RESULTS_DIR, SEEDS
from project4.decision import bayes_action
from project4.metrics import confusion_with_reject, mean_and_se, summarize
from project4.models.gaussian import select_reg
from project4.models.logistic import LogisticRegression
from project4.plotting import COLORS, INK, INK_MUTED, apply_style, plt, save
from project4.prior_shift import adjust_posterior

# =============================================================== configuration
LOSS = DEFAULT_LOSS                              # c_FP = 10, c_FN = 1, c_I = 0.3
DEPLOY_PRIORS = (0.05, 0.1, 0.2, 0.4, 0.6, 0.8)
CONFUSION_PRIOR = 0.05                           # confusion matrices are reported here

# Synthetic data: the class densities of Task 2 (5-D, different covariances).
SYNTH_D = 5
SYNTH_MU0 = np.zeros(SYNTH_D)
SYNTH_MU1 = 0.9 * np.ones(SYNTH_D)
SYNTH_S0 = np.eye(SYNTH_D)
SYNTH_S1 = 0.5 * np.eye(SYNTH_D) + 0.5 * np.ones((SYNTH_D, SYNTH_D))
SYNTH_PI1_TRAIN = 0.5
SYNTH_N_TRAIN = 4601                             # same size as Spambase; 60/20 used for train/validation
SYNTH_N_DEPLOY = 5000                            # fresh deployment sample per prior and seed

SPAMBASE_LOG1P = True                            # log(1+x) before standardising (C2, C4)
L2_GRID = np.logspace(-2, 3, 11)                 # logistic penalty, chosen on validation log-loss
EM_MAX_ITER, EM_TOL = 1000, 1e-8

MODELS = ("gaussian", "logistic")
MODEL_LABELS = {"gaussian": "Gaussian (shared cov.)", "logistic": "Logistic regression"}
METHODS = ("uncorrected", "corrected", "estimated", "oracle")
METHOD_LABELS = {
    "uncorrected": "uncorrected",
    "corrected": r"corrected, known $\pi_1'$",
    "estimated": r"corrected, $\pi_1'$ by EM",
    "oracle": "true posterior (oracle)",
}
TABLE_LABELS = {"uncorrected": "none", "corrected": "known", "estimated": "EM", "oracle": "oracle"}
# ===========================================================================


def log_loss(y, p, eps=1e-12) -> float:
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def fit_logistic(s: data.Split, grid=L2_GRID):
    """Logistic regression with the L2 penalty of lowest validation log-loss."""
    best = (np.inf, None, None)
    for l2 in grid:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = LogisticRegression(l2=l2).fit(s.X_tr, s.y_tr)
        v = log_loss(s.y_va, m.predict_proba(s.X_va))
        if v < best[0]:
            best = (v, l2, m)
    return best[1], best[2]


def fit_models(s: data.Split) -> dict:
    """Both models, each tuned on the validation part only."""
    _, gauss, _ = select_reg(s.X_tr, s.y_tr, s.X_va, s.y_va, shared_cov=True)
    _, logi = fit_logistic(s)
    return {"gaussian": gauss, "logistic": logi}


def em_prior(p, pi1_train: float, max_iter: int = EM_MAX_ITER, tol: float = EM_TOL):
    """Estimate the deployment spam share from unlabelled posteriors (Saerens et al., 2002).

    Repeat:  p' = adjust_posterior(p, pi1_train, pi_hat);  pi_hat = mean(p').
    The fixed point is the maximum-likelihood prior when the model's p is calibrated.
    Returns (pi_hat, number of iterations).
    """
    p = np.asarray(p, dtype=float)
    pi_hat = float(pi1_train)
    for it in range(1, max_iter + 1):
        new = float(np.mean(adjust_posterior(p, pi1_train, pi_hat)))
        new = min(max(new, 1e-6), 1 - 1e-6)          # adjust_posterior needs 0 < pi < 1
        if abs(new - pi_hat) < tol:
            return new, it
        pi_hat = new
    return pi_hat, max_iter


def synthetic_true_posterior(X, pi1):
    return data.gaussian_true_posterior(X, pi1, SYNTH_MU0, SYNTH_MU1, SYNTH_S0, SYNTH_S1)


def deployment_sets(dataset: str, seed: int):
    """Yield (pi1_target, raw split, deployment X, y, true posterior or None) for one seed."""
    if dataset == "synthetic":
        X, y = data.make_gaussian(SYNTH_N_TRAIN, SYNTH_PI1_TRAIN, SYNTH_MU0, SYNTH_MU1, SYNTH_S0, SYNTH_S1, seed)
        raw = data.split(X, y, seed)
        for k, pi in enumerate(DEPLOY_PRIORS):
            Xd, yd = data.make_gaussian(SYNTH_N_DEPLOY, pi, SYNTH_MU0, SYNTH_MU1, SYNTH_S0, SYNTH_S1,
                                        seed=10_000 + 100 * seed + k)
            yield pi, raw, Xd, yd, synthetic_true_posterior(Xd, pi)
    elif dataset == "spambase":
        X, y = data.load_spambase()
        raw = data.split(X, y, seed)
        for k, pi in enumerate(DEPLOY_PRIORS):
            Xd, yd = data.resample_to_prior(raw.X_te, raw.y_te, pi, seed=10_000 + 100 * seed + k)
            yield pi, raw, Xd, yd, None
    else:
        raise ValueError(dataset)


def run_split(dataset: str, seed: int):
    """Per-split records and summed confusion matrices for one dataset and seed."""
    records, confusions, em_rows = [], {}, []
    fitted = None
    for pi, raw, Xd, yd, p_true in deployment_sets(dataset, seed):
        log1p = SPAMBASE_LOG1P and dataset == "spambase"
        if fitted is None:                                   # same split for every prior
            st = data.Standardizer(log1p=log1p).fit(raw.X_tr)
            s = data.Split(st.transform(raw.X_tr), raw.y_tr, st.transform(raw.X_va), raw.y_va,
                           st.transform(raw.X_te), raw.y_te)
            fitted = (st, fit_models(s), float(raw.y_tr.mean()))
        st, models, pi_train = fitted
        Z = st.transform(Xd)
        for name, model in models.items():
            p = model.predict_proba(Z)
            pi_em, n_iter = em_prior(p, pi_train)
            em_rows.append(dict(dataset=dataset, seed=seed, model=name, pi1_deploy=pi,
                                pi1_actual=float(yd.mean()), pi1_em=pi_em, em_iter=n_iter))
            probs = {
                "uncorrected": p,
                "corrected": adjust_posterior(p, pi_train, pi),
                "estimated": adjust_posterior(p, pi_train, pi_em),
            }
            if p_true is not None:
                probs["oracle"] = p_true
            for method, q in probs.items():
                a = bayes_action(q, LOSS)
                rec = dict(dataset=dataset, seed=seed, model=name, pi1_deploy=pi, method=method,
                           pi1_train=pi_train, n_deploy=int(yd.size), **summarize(yd, a, LOSS))
                records.append(rec)
                if np.isclose(pi, CONFUSION_PRIOR):
                    confusions[(name, method)] = confusion_with_reject(yd, a)
    return records, confusions, em_rows


# ---------------------------------------------------------------- summaries
SUMMARY_KEYS = ("realized_loss", "accuracy", "system_accuracy", "reject_rate", "fpr", "fnr")


def summarise(records):
    groups = defaultdict(list)
    for r in records:
        groups[(r["dataset"], r["model"], r["pi1_deploy"], r["method"])].append(r)
    rows = []
    for (ds, model, pi, method), rs in sorted(groups.items()):
        row = dict(dataset=ds, model=model, pi1_deploy=pi, method=method, n_splits=len(rs),
                   n_deploy=float(np.mean([r["n_deploy"] for r in rs])))
        for k in SUMMARY_KEYS:
            m, se = mean_and_se([r[k] for r in rs])
            row[k], row[k + "_se"] = float(m), float(se)
        rows.append(row)
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _cell(m, se):
    return rf"{m:.3f}\,{{\scriptsize$\pm${se:.3f}}}"


def write_loss_tables(summary, datasets):
    for ds in datasets:
        methods = [m for m in METHODS if any(r["dataset"] == ds and r["method"] == m for r in summary)]
        idx = {(r["model"], r["pi1_deploy"], r["method"]): r for r in summary if r["dataset"] == ds}
        cols = "l" + "r" * (len(MODELS) * len(methods))
        head1 = " & ".join(rf"\multicolumn{{{len(methods)}}}{{c}}{{{MODEL_LABELS[m]}}}" for m in MODELS)
        head2 = " & ".join(TABLE_LABELS[m] for _ in MODELS for m in methods)
        lines = [rf"\begin{{tabular}}{{{cols}}}", r"\toprule", rf" & {head1} \\",
                 r"\cmidrule(lr){2-%d}\cmidrule(lr){%d-%d}" % (1 + len(methods), 2 + len(methods), 1 + 2 * len(methods)),
                 rf"$\pi_1'$ & {head2} \\", r"\midrule"]
        for pi in DEPLOY_PRIORS:
            cells = [_cell(idx[(m, pi, k)]["realized_loss"], idx[(m, pi, k)]["realized_loss_se"])
                     for m in MODELS for k in methods]
            lines.append(f"{pi:g} & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (RESULTS_DIR / f"c5_loss_{ds}.tex").write_text("\n".join(lines) + "\n")


def write_confusion_table(confusions, datasets):
    """Mean confusion matrices (counts per deployment set) at CONFUSION_PRIOR."""
    lines = [r"\begin{tabular}{llll rrr}", r"\toprule",
             r"Data & Model & Correction & True label & Deliver & Spam folder & Inspect \\", r"\midrule"]
    for ds in datasets:
        for name in MODELS:
            for method in ("uncorrected", "corrected"):
                C = np.mean(confusions[(ds, name, method)], axis=0)
                lab = f"{ds.capitalize()} & {MODEL_LABELS[name]} & {TABLE_LABELS[method]}"
                lines.append(f"{lab} & real & {C[0, 0]:.1f} & {C[0, 1]:.1f} & {C[0, 2]:.1f} \\\\")
                lines.append(f" & & & spam & {C[1, 0]:.1f} & {C[1, 1]:.1f} & {C[1, 2]:.1f} \\\\[2pt]")
            lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    (RESULTS_DIR / "c5_confusion.tex").write_text("\n".join(lines) + "\n")


def fig_loss(summary, datasets, pi_train):
    fig, axes = plt.subplots(len(datasets), len(MODELS), figsize=(7.0, 2.5 * len(datasets)),
                             sharex=True, squeeze=False)
    for i, ds in enumerate(datasets):
        for j, name in enumerate(MODELS):
            ax = axes[i, j]
            for k, method in enumerate(("uncorrected", "corrected", "estimated")):
                rs = [r for r in summary if r["dataset"] == ds and r["model"] == name and r["method"] == method]
                x = np.array([r["pi1_deploy"] for r in rs])
                m = np.array([r["realized_loss"] for r in rs])
                se = np.array([r["realized_loss_se"] for r in rs])
                ax.plot(x, m, "o-", ms=3.5, color=COLORS[k], label=METHOD_LABELS[method])
                ax.fill_between(x, m - se, m + se, color=COLORS[k], alpha=0.2, lw=0)
            rs = [r for r in summary if r["dataset"] == ds and r["model"] == name and r["method"] == "oracle"]
            if rs:
                ax.plot([r["pi1_deploy"] for r in rs], [r["realized_loss"] for r in rs],
                        color=INK, lw=1.0, ls="--", label=METHOD_LABELS["oracle"])
            ax.axvline(pi_train[ds], color=INK_MUTED, lw=0.8, ls=":")
            ax.annotate(rf"train $\pi_1={pi_train[ds]:.2f}$", (pi_train[ds], 1.0), xycoords=("data", "axes fraction"),
                        xytext=(3, -10), textcoords="offset points", fontsize=7, color=INK_MUTED)
            ax.set_title(f"{ds.capitalize()}: {MODEL_LABELS[name]}", loc="left", fontsize=9)
            ax.set_ylim(bottom=0)
            if i == len(datasets) - 1:
                ax.set_xlabel(r"Deployment spam share $\pi_1'$")
            if j == 0:
                ax.set_ylabel("Realized loss per email")
    handles, labels = [], []
    for ax in axes.ravel():
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.suptitle(rf"$c_{{FP}}={LOSS.c_fp:g}$, $c_{{FN}}={LOSS.c_fn:g}$, $c_I={LOSS.c_insp:g}$; "
                 r"mean $\pm$ s.e. over the splits", x=0.02, ha="left", fontsize=9, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    save(fig, FIGURES_DIR / "c5_prior_shift")


# --------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=len(SEEDS), help="number of splits (seeds 0..n-1)")
    ap.add_argument("--datasets", nargs="*", default=["synthetic", "spambase"])
    # run_all calls main() with no arguments: use the defaults, never run_all's own command line.
    args = ap.parse_args([] if argv is None else argv)
    apply_style()

    records, em_rows = [], []
    confusions = defaultdict(list)
    for ds in args.datasets:
        for seed in SEEDS[: args.seeds]:
            rec, conf, em = run_split(ds, seed)
            records += rec
            em_rows += em
            for (name, method), C in conf.items():
                confusions[(ds, name, method)].append(C)
        print(f"  {ds}: {args.seeds} splits done")

    summary = summarise(records)
    write_csv(RESULTS_DIR / "c5_per_split.csv", records)
    write_csv(RESULTS_DIR / "c5_summary.csv", summary)
    write_csv(RESULTS_DIR / "c5_em_priors.csv", em_rows)
    write_loss_tables(summary, args.datasets)
    write_confusion_table(confusions, args.datasets)
    pi_train = {ds: float(np.mean([r["pi1_train"] for r in records if r["dataset"] == ds])) for ds in args.datasets}
    fig_loss(summary, args.datasets, pi_train)

    for ds in args.datasets:
        print(f"\n[{ds}] mean realized loss (uncorrected / corrected / EM" + (" / oracle)" if ds == "synthetic" else ")"))
        for name in MODELS:
            for pi in DEPLOY_PRIORS:
                vals = [r for r in summary if r["dataset"] == ds and r["model"] == name and r["pi1_deploy"] == pi]
                vals = {r["method"]: r["realized_loss"] for r in vals}
                print(f"  {name:9s} pi1'={pi:<4g} " + "  ".join(f"{vals[m]:.4f}" for m in METHODS if m in vals))
    return 0


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
