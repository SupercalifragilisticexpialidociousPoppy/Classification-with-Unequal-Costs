"""Task 2 experiment (C3, Member 2): cost-ratio sweep, decision-rule comparison.

For every dataset, model and train/validation/test split (seeds 0-19) this script

  1. fits the model on the training part, choosing its regularisation strength
     on the validation part (validation log-loss; the test part is never used
     to choose anything);
  2. sweeps the cost ratio c_FP / c_FN over a log grid with c_I fixed;
  3. at each ratio compares three decision rules on the test part
        accuracy        threshold 0.5, no inspection (accuracy-based rule)
        bayes_2action   threshold c_FP / (c_FP + c_FN), no inspection
        bayes_reject    Bayes rule with the inspect band (thresholds of T1)
  4. records average realized loss, accuracy, reject rate, FPR, FNR and the
     2 x 3 confusion matrix (true label x deliver / spam / inspect).

On the synthetic dataset the exact posterior is known, so the same three rules
applied to the true posterior ("oracle") give the Bayes-risk reference.

Outputs, relative to project4/ (means with standard errors over the splits)
    results/task2_sweep_summary.csv   one row per dataset, model, rule, ratio
    results/task2_confusion.csv       mean confusion matrices at REP_RATIOS
    results/task2_hyperparams.csv     chosen l2 / reg for every split
    results/task2_config.json         the configuration used
    results/task2_loss_<dataset>.tex          LaTeX table: mean loss +- s.e. at REP_RATIOS
    results/task2_confusion_<dataset>_<model>.tex   LaTeX tables: mean confusion matrices
                                              (tabular only; wrap in table + caption in the report)
    figures/task2_loss_<dataset>.png          loss vs cost ratio
    figures/task2_reject_<dataset>.png        reject rate vs cost ratio
    figures/task2_confusion_<dataset>.png     confusion matrices at REP_RATIOS

Run from the repository root:
    python -m project4.experiments.task2_cost_sweep            # full run
    python -m project4.experiments.task2_cost_sweep --seeds 3  # quick check
"""
from __future__ import annotations

import argparse
import csv
import json
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from project4 import data
from project4.decision import (
    ACTION_NAMES,
    LossMatrix,
    bayes_action,
    threshold_action,
)
from project4.metrics import confusion_with_reject, mean_and_se, summarize
from project4.models.gaussian import GaussianClassifier
from project4.models.logistic import LogisticRegression

# =============================================================== configuration
DATASETS = ("spambase", "synthetic")
N_SEEDS = 20                       # seeds 0 .. N_SEEDS-1

# Cost sweep.  c_I is fixed.  Two ways of turning a ratio r = c_FP / c_FN into
# costs (both keep the ratio exactly r):
#   "geometric": c_FP = sqrt(r), c_FN = 1 / sqrt(r), so c_FP * c_FN = 1.  The
#                critical inspection cost c_FP c_FN / (c_FP + c_FN) is then
#                symmetric in r and never below ~0.21 over the grid, so one
#                fixed c_I keeps the inspect band open at every ratio.
#   "fn_fixed":  c_FN = 1, c_FP = r (the plan's c_FP = 10, c_FN = 1 is r = 10).
#                The critical c_I shrinks to r / (1 + r), so for small r the
#                band closes unless c_I is tiny.
COST_SCHEME = "geometric"
C_INSP = 0.1
RATIO_MIN, RATIO_MAX, N_GRID = 1 / 20, 20.0, 17
REP_RATIOS = (0.1, 1.0, 10.0)      # confusion matrices are reported here

# Hyper-parameter grids (chosen on validation log-loss).
L2_GRID = np.logspace(-2, 3, 11)   # logistic: sum-convention penalty (= 1/C)
REG_GRID = (1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0)   # Gaussian covariance shrinkage

# Feature transform before the standardiser (log(1 + x) applies to Spambase only).
LOG1P_LOGISTIC = False
LOG1P_GAUSSIAN = False

# Synthetic data: two Gaussians with different covariances.
SYNTH_N, SYNTH_PI1, SYNTH_D = 4601, 0.39, 5
SYNTH_MU0 = np.zeros(SYNTH_D)
SYNTH_MU1 = 0.9 * np.ones(SYNTH_D)
SYNTH_S0 = np.eye(SYNTH_D)
SYNTH_S1 = 0.5 * np.eye(SYNTH_D) + 0.5 * np.ones((SYNTH_D, SYNTH_D))

CONFUSION_MODEL = "logistic"       # model shown in the confusion-matrix figure

PROJECT_DIR = Path(__file__).resolve().parents[1]      # .../project4
FIG_DIR, RES_DIR = PROJECT_DIR / "figures", PROJECT_DIR / "results"
# ===========================================================================

RULES = ("accuracy", "bayes_2action", "bayes_reject")
RULE_LABELS = {
    "accuracy": "accuracy rule (t = 0.5)",
    "bayes_2action": "Bayes, no inspection",
    "bayes_reject": "Bayes with inspection",
}
MODELS = {
    # name: (hyper-parameter name, grid, factory, use log1p)
    "logistic": ("l2", L2_GRID, lambda h: LogisticRegression(l2=h), LOG1P_LOGISTIC),
    "gauss_shared": ("reg", REG_GRID, lambda h: GaussianClassifier(shared_cov=True, reg=h), LOG1P_GAUSSIAN),
    "gauss_qda": ("reg", REG_GRID, lambda h: GaussianClassifier(shared_cov=False, reg=h), LOG1P_GAUSSIAN),
}
MODEL_LABELS = {
    "logistic": "Logistic regression",
    "gauss_shared": "Gaussian, shared cov.",
    "gauss_qda": "Gaussian, class-specific cov.",
    "oracle": "True posterior (oracle)",
}
METRICS = ("loss", "acc", "reject", "fpr", "fnr")


def ratio_grid() -> np.ndarray:
    base = np.logspace(np.log10(RATIO_MIN), np.log10(RATIO_MAX), N_GRID)
    return np.unique(np.round(np.concatenate([base, REP_RATIOS]), 10))


def make_loss(ratio: float) -> LossMatrix:
    if COST_SCHEME == "geometric":
        return LossMatrix(np.sqrt(ratio), 1.0 / np.sqrt(ratio), C_INSP)
    if COST_SCHEME == "fn_fixed":
        return LossMatrix(ratio, 1.0, C_INSP)
    raise ValueError(f"unknown COST_SCHEME {COST_SCHEME!r}")


def decide(rule: str, p: np.ndarray, L: LossMatrix) -> np.ndarray:
    if rule == "accuracy":
        return threshold_action(p, 0.5)
    if rule == "bayes_2action":
        return threshold_action(p, L.two_action_threshold)
    if rule == "bayes_reject":
        return bayes_action(p, L)
    raise ValueError(rule)


def log_loss(y, p, eps=1e-12) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_with_validation(factory, grid, s: data.Split):
    """Fit one model per grid value on train; return the best on validation log-loss."""
    best = (np.inf, None, None)
    for h in grid:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = factory(h).fit(s.X_tr, s.y_tr)
                v = log_loss(s.y_va, model.predict_proba(s.X_va))
        except (np.linalg.LinAlgError, FloatingPointError, ValueError):
            continue  # e.g. covariance not positive definite at this strength
        if np.isfinite(v) and v < best[0]:
            best = (v, h, model)
    if best[2] is None:
        raise RuntimeError("no hyper-parameter value gave a valid fit")
    return best[1], best[2], best[0]


def load_dataset(name: str, seed: int):
    """Raw (unstandardised) split for one seed, plus the true-posterior function if known."""
    if name == "spambase":
        X, y = data.load_spambase()
        return data.split(X, y, seed), None
    if name == "synthetic":
        X, y = data.make_gaussian(SYNTH_N, SYNTH_PI1, SYNTH_MU0, SYNTH_MU1, SYNTH_S0, SYNTH_S1, seed)

        def true_post(Xr):
            return data.gaussian_true_posterior(Xr, SYNTH_PI1, SYNTH_MU0, SYNTH_MU1, SYNTH_S0, SYNTH_S1)

        return data.split(X, y, seed), true_post
    raise ValueError(name)


def run_split(dataset: str, seed: int, grid: np.ndarray):
    """All records for one dataset and one seed."""
    raw, true_post = load_dataset(dataset, seed)
    probs, hypers = {}, []
    for name, (hname, hgrid, factory, use_log1p) in MODELS.items():
        s = data.preprocess(raw, log1p=use_log1p and dataset == "spambase")
        h, model, val_ll = fit_with_validation(factory, hgrid, s)
        probs[name] = model.predict_proba(s.X_te)
        hypers.append(dict(dataset=dataset, seed=seed, model=name, hyper=hname, value=h, val_logloss=val_ll))
    if true_post is not None:
        probs["oracle"] = true_post(raw.X_te)

    y = raw.y_te
    records = []
    for r in grid:
        L = make_loss(r)
        for name, p in probs.items():
            for rule in RULES:
                a = decide(rule, p, L)
                m = summarize(y, a, L)
                records.append(dict(
                    dataset=dataset, model=name, rule=rule, ratio=r, seed=seed,
                    loss=m["realized_loss"], acc=m["accuracy"], reject=m["reject_rate"],
                    fpr=m["fpr"], fnr=m["fnr"], C=confusion_with_reject(y, a),
                ))
    return records, hypers


# ------------------------------------------------------------------ outputs
def aggregate(records):
    groups = defaultdict(list)
    for rec in records:
        groups[(rec["dataset"], rec["model"], rec["rule"], rec["ratio"])].append(rec)
    summary = {}
    for key, recs in groups.items():
        row = {"n_splits": len(recs)}
        for m in METRICS:
            mean, se = mean_and_se([r[m] for r in recs])
            row[m + "_mean"], row[m + "_se"] = float(mean), float(se)
        row["C_mean"] = np.mean([r["C"] for r in recs], axis=0)
        summary[key] = row
    return summary


def write_csvs(summary, hypers):
    RES_DIR.mkdir(parents=True, exist_ok=True)
    cols = [f"{m}_{s}" for m in METRICS for s in ("mean", "se")]
    with open(RES_DIR / "task2_sweep_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "model", "rule", "ratio_cfp_cfn", "c_fp", "c_fn", "c_insp", "n_splits", *cols])
        for (ds, model, rule, r), row in sorted(summary.items()):
            L = make_loss(r)
            w.writerow([ds, model, rule, f"{r:.6g}", f"{L.c_fp:.6g}", f"{L.c_fn:.6g}", L.c_insp,
                        row["n_splits"], *[f"{row[c]:.6g}" for c in cols]])
    with open(RES_DIR / "task2_confusion.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "model", "rule", "ratio_cfp_cfn", "true_label",
                    *[f"mean_count_{n}" for n in ACTION_NAMES]])
        for (ds, model, rule, r), row in sorted(summary.items()):
            if any(np.isclose(r, q) for q in REP_RATIOS):
                for yv in (0, 1):
                    w.writerow([ds, model, rule, f"{r:.6g}", yv, *[f"{v:.3f}" for v in row["C_mean"][yv]]])
    with open(RES_DIR / "task2_hyperparams.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "seed", "model", "hyper", "value", "val_logloss"])
        w.writeheader()
        w.writerows(hypers)
    cfg = dict(
        datasets=DATASETS, cost_scheme=COST_SCHEME, c_insp=C_INSP, ratio_range=[RATIO_MIN, RATIO_MAX],
        rep_ratios=REP_RATIOS, l2_grid=list(map(float, L2_GRID)), reg_grid=list(REG_GRID),
        log1p_logistic=LOG1P_LOGISTIC, log1p_gaussian=LOG1P_GAUSSIAN,
        synthetic=dict(n=SYNTH_N, pi1=SYNTH_PI1, d=SYNTH_D, mu0=SYNTH_MU0.tolist(), mu1=SYNTH_MU1.tolist(),
                       S0=SYNTH_S0.tolist(), S1=SYNTH_S1.tolist()),
    )
    (RES_DIR / "task2_config.json").write_text(json.dumps(cfg, indent=2))


TEX_MODELS = {"logistic": "Logistic", "gauss_shared": "Gauss.\\ shared", "gauss_qda": "Gauss.\\ class-spec.", "oracle": "Oracle"}
TEX_RULES = {
    "accuracy": r"accuracy ($t=0.5$)",
    "bayes_2action": "Bayes, 2-action",
    "bayes_reject": "Bayes + insp.",
}


def _rep(grid):
    return [next(q for q in grid if np.isclose(q, r)) for r in REP_RATIOS]


def write_tex_tables(summary, datasets):
    """LaTeX tabulars for the report (\\input them; the report adds table/caption)."""
    RES_DIR.mkdir(parents=True, exist_ok=True)
    reps = _rep(ratio_grid())
    for ds in datasets:
        models = list(MODELS) + (["oracle"] if (ds, "oracle", "bayes_reject", reps[0]) in summary else [])
        # --- loss table: model x rule, one column per representative ratio
        lines = [r"\begin{tabular}{ll" + "c" * len(reps) + "}", r"\toprule",
                 rf"Model & Rule & \multicolumn{{{len(reps)}}}{{c}}{{$\cFP/\cFN$}} \\", rf"\cmidrule(l){{3-{2 + len(reps)}}}",
                 " & & " + " & ".join(f"{r:g}" for r in REP_RATIOS) + r" \\", r"\midrule"]
        for i, model in enumerate(models):
            rules = ("bayes_reject",) if model == "oracle" else RULES
            for j, rule in enumerate(rules):
                cells = []
                for rr in reps:
                    row = summary[(ds, model, rule, rr)]
                    cells.append(f"{row['loss_mean']:.3f}{{\\scriptsize$\\pm${row['loss_se']:.3f}}}")
                name = TEX_MODELS[model] if j == 0 else ""
                lines.append(f"{name} & {TEX_RULES[rule]} & " + " & ".join(cells) + r" \\")
            if i < len(models) - 1:
                lines.append(r"\midrule")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (RES_DIR / f"task2_loss_{ds}.tex").write_text("\n".join(lines) + "\n")

        # --- confusion tables: one per model, blocks per ratio, rows per rule x true label
        for model in models:
            rules = ("bayes_reject",) if model == "oracle" else RULES
            lines = [r"\begin{tabular}{llcrrr}", r"\toprule",
                     r"$\cFP/\cFN$ & Rule & True & Deliver & Spam & Inspect \\", r"\midrule"]
            for bi, (r, rr) in enumerate(zip(REP_RATIOS, reps)):
                first = True
                for rule in rules:
                    C = summary[(ds, model, rule, rr)]["C_mean"]
                    for yv, lab in ((0, "real"), (1, "spam")):
                        a = f"{r:g}" if first else ""
                        b = TEX_RULES[rule] if yv == 0 else ""
                        lines.append(f"{a} & {b} & {lab} & " + " & ".join(f"{v:.1f}" for v in C[yv]) + r" \\")
                        first = False
                if bi < len(REP_RATIOS) - 1:
                    lines.append(r"\midrule")
            lines += [r"\bottomrule", r"\end{tabular}"]
            (RES_DIR / f"task2_confusion_{ds}_{model}.tex").write_text("\n".join(lines) + "\n")


def _series(summary, ds, model, rule, metric, grid):
    mean = np.array([summary[(ds, model, rule, r)][metric + "_mean"] for r in grid])
    se = np.array([summary[(ds, model, rule, r)][metric + "_se"] for r in grid])
    return mean, se


def plot_curves(summary, ds, grid, metric, ylabel, fname, rules=RULES, logy=False):
    models = [m for m in MODELS] + (["oracle"] if (ds, "oracle", "bayes_reject", grid[0]) in summary else [])
    fig, axes = plt.subplots(1, len(MODELS), figsize=(5 * len(MODELS), 4), sharey=True)
    for ax, model in zip(np.atleast_1d(axes), MODELS):
        for rule in rules:
            m, se = _series(summary, ds, model, rule, metric, grid)
            ax.plot(grid, m, marker="o", ms=3, label=RULE_LABELS[rule])
            ax.fill_between(grid, m - se, m + se, alpha=0.25)
        if "oracle" in models:
            m, _ = _series(summary, ds, "oracle", "bayes_reject", metric, grid)
            ax.plot(grid, m, "k--", lw=1.2, label="oracle, Bayes with inspection")
        for q in REP_RATIOS:
            ax.axvline(q, color="grey", lw=0.6, ls=":")
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(r"cost ratio $c_{FP}/c_{FN}$")
        ax.set_title(MODEL_LABELS[model])
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel(ylabel)
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle(f"{ds}: test set, mean $\\pm$ s.e. over splits, $c_I$ = {C_INSP}", y=1.02)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusions(summary, ds):
    fig, axes = plt.subplots(len(REP_RATIOS), len(RULES), figsize=(3.6 * len(RULES), 2.6 * len(REP_RATIOS)))
    for i, r in enumerate(REP_RATIOS):
        rr = next(q for q in ratio_grid() if np.isclose(q, r))
        for j, rule in enumerate(RULES):
            C = summary[(ds, CONFUSION_MODEL, rule, rr)]["C_mean"]
            frac = C / np.maximum(C.sum(1, keepdims=True), 1e-12)
            ax = axes[i, j]
            ax.imshow(frac, vmin=0, vmax=1, cmap="Blues")
            for yv in (0, 1):
                for a in range(3):
                    ax.text(a, yv, f"{C[yv, a]:.0f}", ha="center", va="center",
                            color="white" if frac[yv, a] > 0.5 else "black", fontsize=9)
            ax.set_xticks(range(3), ACTION_NAMES, fontsize=8)
            ax.set_yticks((0, 1), ("real", "spam"), fontsize=8)
            if i == 0:
                ax.set_title(RULE_LABELS[rule], fontsize=9)
            if j == 0:
                ax.set_ylabel(f"$c_{{FP}}/c_{{FN}}$ = {r:g}\ntrue label", fontsize=9)
    fig.suptitle(f"{ds}, {MODEL_LABELS[CONFUSION_MODEL]}: mean test counts per split "
                 "(colour = row share)", fontsize=10)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"task2_confusion_{ds}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_headline(summary, datasets):
    for ds in datasets:
        print(f"\n[{ds}] mean test loss, 'accuracy' / 'bayes_2action' / 'bayes_reject'")
        for model in list(MODELS) + (["oracle"] if (ds, "oracle", "bayes_reject", ratio_grid()[0]) in summary else []):
            for r in REP_RATIOS:
                rr = next(q for q in ratio_grid() if np.isclose(q, r))
                v = [summary[(ds, model, rule, rr)]["loss_mean"] for rule in RULES]
                print(f"  {model:13s} ratio {r:>5g}:  " + "  /  ".join(f"{x:.4f}" for x in v))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=N_SEEDS, help="number of splits (seeds 0..n-1)")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    args = ap.parse_args(argv)

    grid = ratio_grid()
    records, hypers = [], []
    for ds in args.datasets:
        for seed in range(args.seeds):
            rec, hyp = run_split(ds, seed, grid)
            records += rec
            hypers += hyp
            print(f"{ds}: seed {seed} done", flush=True)

    summary = aggregate(records)
    write_csvs(summary, hypers)
    for ds in args.datasets:
        plot_curves(summary, ds, grid, "loss", "average realized loss", f"task2_loss_{ds}.png")
        plot_curves(summary, ds, grid, "reject", "reject rate", f"task2_reject_{ds}.png", rules=("bayes_reject",))
        plot_confusions(summary, ds)
    write_tex_tables(summary, args.datasets)
    print_headline(summary, args.datasets)
    print(f"\nWrote results to {RES_DIR} and figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
