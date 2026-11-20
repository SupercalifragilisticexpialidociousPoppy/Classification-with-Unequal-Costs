"""C2 (Member 3): Gaussian classifiers on Spambase, raw vs log(1+x) features.

For each of the 20 stratified splits and each preprocessing (standardise, or
log(1+x) then standardise) we fit the shared- and class-specific-covariance
classifiers, choose the shrinkage strength on the validation part by
log-loss (``select_reg``), and evaluate on the test part.

Outputs
  results/c2_gaussian_per_split.csv   one row per split x preprocessing x model
  results/c2_gaussian_summary.csv     mean and standard error over the 20 splits
  results/c2_gaussian_summary.tex     the table for the report
  figures/c2_gaussian_log1p.{pdf,png} test log-loss and realized loss, raw vs log(1+x)
"""
import csv

import numpy as np

from project4.config import DEFAULT_LOSS, FIGURES_DIR, RESULTS_DIR, SEEDS
from project4.data import load_spambase, preprocess, split
from project4.decision import bayes_action, threshold_action
from project4.metrics import accuracy, mean_and_se, realized_loss, reject_rate
from project4.models.gaussian import select_reg
from project4.plotting import COLORS, apply_style, plt, save

# ---- configuration --------------------------------------------------------
LOSS = DEFAULT_LOSS                       # c_fp = 10, c_fn = 1, c_I = 0.3
PREPROCESSING = (("raw", False), ("log1p", True))
MODELS = (("shared", True), ("separate", False))
MODEL_LABEL = {"shared": "Shared covariance", "separate": "Class-specific covariance"}
MODEL_TICK = {"shared": "Shared\ncovariance", "separate": "Class-specific\ncovariance"}
TEX_PREP = {"raw": "raw", "log1p": r"$\log(1+x)$"}
TEX_MODEL = {"shared": "shared", "separate": "class-specific"}
PREP_LABEL = {"raw": "standardised", "log1p": r"$\log(1+x)$, standardised"}
METRICS = ("reg", "val_log_loss", "test_log_loss", "test_brier", "test_accuracy",
           "test_loss_bayes", "test_reject_rate", "test_loss_two_action", "test_loss_acc_rule")
EPS = 1e-12


def log_loss(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def run():
    X, y = load_spambase()
    rows = []
    for seed in SEEDS:
        s0 = split(X, y, seed)
        for prep, use_log in PREPROCESSING:
            s = preprocess(s0, log1p=use_log)
            for name, shared in MODELS:
                r, model, scores = select_reg(s.X_tr, s.y_tr, s.X_va, s.y_va, shared_cov=shared)
                p = model.predict_proba(s.X_te)
                t_star = LOSS.two_action_threshold
                rows.append({
                    "seed": seed, "preprocessing": prep, "model": name,
                    "reg": r,
                    "val_log_loss": scores[r],
                    "test_log_loss": log_loss(s.y_te, p),
                    "test_brier": brier(s.y_te, p),
                    "test_accuracy": accuracy(s.y_te, threshold_action(p, 0.5)),
                    "test_loss_bayes": realized_loss(s.y_te, bayes_action(p, LOSS), LOSS),
                    "test_reject_rate": reject_rate(bayes_action(p, LOSS)),
                    "test_loss_two_action": realized_loss(s.y_te, threshold_action(p, t_star), LOSS),
                    "test_loss_acc_rule": realized_loss(s.y_te, threshold_action(p, 0.5), LOSS),
                })
    return rows


def summarize(rows):
    out = []
    for prep, _ in PREPROCESSING:
        for name, _ in MODELS:
            sel = [r for r in rows if r["preprocessing"] == prep and r["model"] == name]
            row = {"preprocessing": prep, "model": name}
            for k in METRICS:
                m, se = mean_and_se([r[k] for r in sel])
                row[k], row[k + "_se"] = float(m), float(se)
            row["reg_median"] = float(np.median([r["reg"] for r in sel]))
            out.append(row)
    return out


def paired_log1p_effect(rows, metric):
    """Mean and SE over splits of (log1p - raw) for each model (paired by seed)."""
    res = {}
    for name, _ in MODELS:
        raw = {r["seed"]: r[metric] for r in rows if r["model"] == name and r["preprocessing"] == "raw"}
        lg = {r["seed"]: r[metric] for r in rows if r["model"] == name and r["preprocessing"] == "log1p"}
        res[name] = mean_and_se([lg[s] - raw[s] for s in SEEDS])
    return res


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def write_tex(summary):
    def cell(r, k, fmt):
        return f"{r[k]:{fmt}} $\\pm$ {r[k + '_se']:{fmt}}"

    lines = [
        r"\begin{tabular}{llrrrrrr}", r"\toprule",
        r"Features & Model & $\alpha$ & Log-loss & Brier & Accuracy & Loss $t^\ast$ & Loss Bayes \\",
        r"\midrule",
    ]
    for r in summary:
        lines.append(
            f"{TEX_PREP[r['preprocessing']]} & {TEX_MODEL[r['model']]} & {r['reg_median']:g} & "
            f"{cell(r, 'test_log_loss', '.3f')} & {cell(r, 'test_brier', '.3f')} & "
            f"{cell(r, 'test_accuracy', '.3f')} & {cell(r, 'test_loss_two_action', '.3f')} & "
            f"{cell(r, 'test_loss_bayes', '.3f')} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (RESULTS_DIR / "c2_gaussian_summary.tex").write_text("\n".join(lines) + "\n")


def plot(summary):
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    panels = (("test_log_loss", "Test log-loss"),
              ("test_loss_bayes", f"Test realized loss (Bayes rule, $c_I={LOSS.c_insp:g}$)"))
    x = np.arange(len(MODELS))
    width = 0.36
    for ax, (k, title) in zip(axes, panels):
        for j, (prep, _) in enumerate(PREPROCESSING):
            sel = [r for r in summary if r["preprocessing"] == prep]
            ax.bar(x + (j - 0.5) * width, [r[k] for r in sel], width,
                   yerr=[r[k + "_se"] for r in sel], color=COLORS[j], capsize=3,
                   label=PREP_LABEL[prep])
        ax.set_xticks(x, [MODEL_TICK[n] for n, _ in MODELS])
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, 1.0))
    save(fig, FIGURES_DIR / "c2_gaussian_log1p")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = run()
    write_csv(RESULTS_DIR / "c2_gaussian_per_split.csv", rows)
    summary = summarize(rows)
    write_csv(RESULTS_DIR / "c2_gaussian_summary.csv", summary)
    write_tex(summary)
    plot(summary)
    for r in summary:
        print(f"  {r['preprocessing']:6s} {r['model']:8s} reg~{r['reg_median']:<6g} "
              f"logloss {r['test_log_loss']:.3f}+-{r['test_log_loss_se']:.3f}  "
              f"acc {r['test_accuracy']:.3f}  loss(Bayes) {r['test_loss_bayes']:.3f}+-{r['test_loss_bayes_se']:.3f}")
    for k in ("test_log_loss", "test_loss_bayes"):
        for name, (m, se) in paired_log1p_effect(rows, k).items():
            print(f"  log1p - raw, {k:16s} {name:8s}: {m:+.4f} +- {se:.4f}")


if __name__ == "__main__":
    main()
