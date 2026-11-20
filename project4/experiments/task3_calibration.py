"""Task 3 (C4, Member 5): how good are the models' probabilities, and does it matter?

For both models (Gaussian generative, logistic regression) on synthetic data and
Spambase, over the 20 splits:
  1. log-loss, Brier score, expected calibration error (ECE), accuracy and
     realized loss of the Bayes rule, on the test part;
  2. reliability diagrams (test predictions of all 20 splits pooled);
  3. temperature scaling of the logistic model's log-odds: accuracy stays fixed
     while the realized loss changes (Member 4's "same accuracy, different cost");
  4. optional: Platt re-calibration fitted on the validation part.

Outputs
  figures/c4_reliability.{pdf,png}     reliability diagrams with bin-count histograms
  figures/c4_temperature.{pdf,png}     accuracy, log-loss, cost and inspection rate vs T
  results/c4_scores.{csv,tex}          score table (mean and SE over 20 splits)
  results/c4_temperature.{csv,tex}     temperature table for selected T

Models: the team's from-scratch classes (models/gaussian.py, models/logistic.py)
are used when they exist. Until then scikit-learn stand-ins are used so the
pipeline runs; the outputs then say "PREVIEW" and must be regenerated.
"""
import csv
import importlib

import numpy as np

from project4 import calibration as cal
from project4.config import C_FN, C_FP, C_INSP, FIGURES_DIR, RESULTS_DIR, SEEDS
from project4.data import gaussian_true_posterior, load_spambase, make_gaussian, preprocess, split
from project4.decision import LossMatrix, bayes_action, threshold_action
from project4.metrics import accuracy, mean_and_se, realized_loss, reject_rate
from project4.plotting import COLORS, INK_MUTED, apply_style, plt, save

# ---- configuration --------------------------------------------------------
L = LossMatrix(C_FP, C_FN, C_INSP)
N_BINS = 10
LOG1P = True                         # log(1+x) before standardising Spambase (skewed counts)
GAUSS_SHARED_COV = True              # Gaussian model with one shared covariance
GAUSS_REG_GRID = (1e-3, 1e-2, 1e-1, 0.3)       # covariance shrinkage, chosen on validation
LOGISTIC_L2_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)  # L2 strength, chosen on validation
SYN = dict(n=4601, pi1=0.4, mu0=[0.0, 0.0], mu1=[1.5, 1.0], S=[[1.0, 0.3], [0.3, 1.0]], seed=2024)
T_GRID = np.exp(np.linspace(np.log(0.25), np.log(4.0), 33))   # temperatures for the sweep
T_TABLE = (0.4, 1.0, 2.5)            # rows of the temperature table
TEMPERATURE_MODEL = ("spambase", "Logistic")


# ---- models ---------------------------------------------------------------
class _SkGaussian:
    """Stand-in for models/gaussian.py: LDA/QDA with shrinkage towards a scaled identity."""

    def __init__(self, shared_cov, reg):
        from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                                   QuadraticDiscriminantAnalysis)
        self.m = (LinearDiscriminantAnalysis(solver="lsqr", shrinkage=reg) if shared_cov
                  else QuadraticDiscriminantAnalysis(reg_param=reg))

    def fit(self, X, y):
        self.m.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)[:, 1]


class _SkLogistic:
    """Stand-in for models/logistic.py: minimises E(w) + (l2/2)|w|^2 (intercept unpenalised)."""

    def __init__(self, l2):
        from sklearn.linear_model import LogisticRegression
        self.m = LogisticRegression(C=1.0 / l2, max_iter=10_000)

    def fit(self, X, y):
        self.m.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)[:, 1]


def _team_class(module, cls):
    """The team's model class, or None if its file does not exist yet.

    Only a missing module falls back to a stand-in; any other import error
    (a bug inside the team's file) is raised so it cannot go unnoticed.
    """
    name = f"project4.models.{module}"
    try:
        return getattr(importlib.import_module(name), cls)
    except ModuleNotFoundError as e:
        if e.name != name:
            raise
        return None


def get_models():
    """{name: (factory(hyperparameter), grid)} and a flag saying whether any stand-in is used."""
    Gaussian = _team_class("gaussian", "GaussianClassifier")
    Logistic = _team_class("logistic", "LogisticRegression")
    if Gaussian is not None:
        gauss = lambda reg: Gaussian(shared_cov=GAUSS_SHARED_COV, reg=reg)  # noqa: E731
    else:
        gauss = lambda reg: _SkGaussian(GAUSS_SHARED_COV, reg)  # noqa: E731
    logi = (lambda l2: Logistic(l2=l2)) if Logistic is not None else (lambda l2: _SkLogistic(l2))  # noqa: E731
    missing = [m for m, c in (("gaussian.py", Gaussian), ("logistic.py", Logistic)) if c is None]
    if missing:
        print(f"   PREVIEW: {', '.join(missing)} not found, using scikit-learn stand-ins")
    return {"Gaussian": (gauss, GAUSS_REG_GRID), "Logistic": (logi, LOGISTIC_L2_GRID)}, bool(missing)


def fit_on_validation(factory, grid, s):
    """Fit one model per hyperparameter on train; keep the one with the lowest validation log-loss."""
    best = None
    for h in grid:
        model = factory(h).fit(s.X_tr, s.y_tr)
        p_va = np.asarray(model.predict_proba(s.X_va), dtype=float)
        score = cal.log_loss(s.y_va, p_va)
        if best is None or score < best[0]:
            best = (score, h, model, p_va)
    return best[2], best[3], best[1]


# ---- data -----------------------------------------------------------------
def datasets():
    Xs, ys = make_gaussian(SYN["n"], SYN["pi1"], SYN["mu0"], SYN["mu1"], SYN["S"], SYN["S"], SYN["seed"])
    Xb, yb = load_spambase()
    return {"synthetic": (Xs, ys, False), "spambase": (Xb, yb, LOG1P)}


def true_posterior(X):
    return gaussian_true_posterior(X, SYN["pi1"], SYN["mu0"], SYN["mu1"], SYN["S"], SYN["S"])


# ---- metrics --------------------------------------------------------------
METRICS = ("accuracy", "log_loss", "brier", "ece", "ece_quantile", "cost", "reject_rate")


def score(y, p):
    a = bayes_action(p, L)
    return {
        "accuracy": accuracy(y, threshold_action(p, 0.5)),   # labels from p >= 0.5
        "log_loss": cal.log_loss(y, p),
        "brier": cal.brier(y, p),
        "ece": cal.ece(y, p, N_BINS, "uniform"),
        "ece_quantile": cal.ece(y, p, N_BINS, "quantile"),
        "cost": realized_loss(y, a, L),                        # Bayes rule with inspection
        "reject_rate": reject_rate(a),
    }


# ---- experiment -----------------------------------------------------------
def run():
    models, preview = get_models()
    scores = {}            # (data, model) -> list of score dicts over seeds
    pooled = {}            # (data, model) -> (y, p) test predictions pooled over seeds
    temp = []              # per seed: {T: score dict}
    chosen = {}            # (data, model) -> list of chosen hyperparameters
    for data_name, (X, y, log1p) in datasets().items():
        for seed in SEEDS:
            raw = split(X, y, seed)
            s = preprocess(raw, log1p=log1p)
            rows = {}
            if data_name == "synthetic":
                rows["Bayes (true $p$)"] = true_posterior(raw.X_te)
            for name, (factory, grid) in models.items():
                model, p_va, h = fit_on_validation(factory, grid, s)
                chosen.setdefault((data_name, name), []).append(h)
                p_te = np.asarray(model.predict_proba(s.X_te), dtype=float)
                A, B = cal.platt_fit(s.y_va, p_va)
                rows[name] = p_te
                rows[f"{name} + Platt"] = cal.platt_apply(p_te, A, B)
                pooled.setdefault((data_name, name), ([], []))
                pooled[(data_name, name)][0].append(s.y_te)
                pooled[(data_name, name)][1].append(p_te)
                if (data_name, name) == TEMPERATURE_MODEL:
                    temp.append({T: score(s.y_te, cal.temperature_scale(p_te, T))
                                 for T in sorted(set(T_GRID) | set(T_TABLE))})
            for name, p in rows.items():
                scores.setdefault((data_name, name), []).append(score(s.y_te, p))
        print(f"   {data_name}: done")
    for key, hs in chosen.items():
        vals, counts = np.unique(hs, return_counts=True)
        print(f"   chosen hyperparameter {key}: " + ", ".join(f"{v:g} x{c}" for v, c in zip(vals, counts)))
    pooled = {k: (np.concatenate(v[0]), np.concatenate(v[1])) for k, v in pooled.items()}
    return scores, pooled, temp, preview


def summarise(list_of_dicts):
    out = {}
    for m in METRICS:
        mean, se = mean_and_se([d[m] for d in list_of_dicts])
        out[m], out[m + "_se"] = float(mean), float(se)
    return out


# ---- tables ---------------------------------------------------------------
def _pm(mean, se, digits=3):
    return f"{mean:.{digits}f}\\,{{\\scriptsize$\\pm${se:.{digits}f}}}"


def _preview_row(preview, ncols):
    """A visible table line while the scikit-learn stand-ins are in use."""
    if not preview:
        return []
    return [rf"\multicolumn{{{ncols}}}{{l}}{{\scriptsize\textit{{PREVIEW: scikit-learn stand-in models;"
            r" regenerate after the team models are merged.}} \\"]


def write_scores(scores, preview):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = [{"data": d, "model": m, **summarise(v)} for (d, m), v in scores.items()]
    with open(RESULTS_DIR / "c4_scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + ["preview"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "preview": preview})
    cols = ("accuracy", "log_loss", "brier", "ece", "cost", "reject_rate")
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"Data & Model & Accuracy & Log-loss & Brier & ECE & Cost & Inspected \\", r"\midrule"]
    last = None
    for r in rows:
        if last is not None and r["data"] != last:
            lines.append(r"\midrule")
        data = r["data"].capitalize() if r["data"] != last else ""
        last = r["data"]
        lines.append(f"{data} & {r['model']} & " + " & ".join(_pm(r[c], r[c + '_se']) for c in cols) + r" \\")
    lines.append(r"\bottomrule")
    lines += _preview_row(preview, 8) + [r"\end{tabular}"]
    (RESULTS_DIR / "c4_scores.tex").write_text("\n".join(lines) + "\n")
    return rows


def write_temperature(temp, preview):
    Ts = sorted(temp[0])
    rows = [{"T": T, **summarise([t[T] for t in temp])} for T in Ts]
    with open(RESULTS_DIR / "c4_temperature.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + ["preview"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "preview": preview})
    cols = ("accuracy", "log_loss", "brier", "ece", "cost", "reject_rate")
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Temperature & Accuracy & Log-loss & Brier & ECE & Cost & Inspected \\", r"\midrule"]
    for r in rows:
        if any(np.isclose(r["T"], t) for t in T_TABLE):
            note = {1.0: " (model as fitted)"}.get(round(r["T"], 6), "")
            lines.append(f"$T={r['T']:g}${note} & " + " & ".join(_pm(r[c], r[c + '_se']) for c in cols) + r" \\")
    lines.append(r"\bottomrule")
    lines += _preview_row(preview, 7) + [r"\end{tabular}"]
    (RESULTS_DIR / "c4_temperature.tex").write_text("\n".join(lines) + "\n")
    return rows


# ---- figures --------------------------------------------------------------
def _preview_note(fig, preview):
    if preview:
        fig.text(0.99, 1.0, "PREVIEW: scikit-learn stand-in models", ha="right", va="bottom",
                 fontsize=7, color=INK_MUTED)


def fig_reliability(pooled, preview):
    keys = [("synthetic", "Gaussian"), ("synthetic", "Logistic"), ("spambase", "Gaussian"), ("spambase", "Logistic")]
    fig = plt.figure(figsize=(7.0, 7.0))
    outer = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)
    for i, (data, model) in enumerate(keys):
        r, c = i // 2, i % 2
        inner = outer[r, c].subgridspec(2, 1, height_ratios=(3, 1), hspace=0.08)
        ax = fig.add_subplot(inner[0])
        axh = fig.add_subplot(inner[1], sharex=ax)
        y, p = pooled[(data, model)]
        ax.plot([0, 1], [0, 1], color=INK_MUTED, lw=0.8, ls="--")
        for k, (strategy, lab) in enumerate((("uniform", "equal-width bins"), ("quantile", "equal-count bins"))):
            b = cal.reliability_bins(y, p, N_BINS, strategy)
            ok = b["count"] > 0
            ax.plot(b["mean_p"][ok], b["frac_pos"][ok], "o-" if k == 0 else "s:", ms=4 if k == 0 else 3.5,
                    color=COLORS[k], lw=1.6 if k == 0 else 1.2, label=lab)
        ax.set_title(f"{data.capitalize()}, {model}", loc="left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(labelbottom=False)
        if c == 0:
            ax.set_ylabel("Observed spam fraction")
        b = cal.reliability_bins(y, p, N_BINS, "uniform")
        axh.bar((b["lo"] + b["hi"]) / 2, b["count"], width=1 / N_BINS * 0.9, color=COLORS[0], alpha=0.35)
        axh.set_yscale("log")
        axh.set_ylim(1, None)
        axh.grid(axis="x", visible=False)
        if c == 0:
            axh.set_ylabel("Count", fontsize=8)
        axh.set_xlabel("Predicted probability $p$")
    fig.axes[0].legend(loc="upper left", fontsize=8)
    _preview_note(fig, preview)
    save(fig, FIGURES_DIR / "c4_reliability")


def fig_temperature(temp, preview):
    Ts = np.array(sorted(T_GRID))
    panels = (("accuracy", "Accuracy (labels from $p \\geq 0.5$)"), ("log_loss", "Log-loss"),
              ("cost", "Realized loss per email (Bayes rule)"), ("reject_rate", "Fraction inspected"))
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.4), sharex=True)
    for ax, (m, title) in zip(axes.ravel(), panels):
        mean, se = mean_and_se([[t[T][m] for T in Ts] for t in temp])
        ax.fill_between(Ts, mean - se, mean + se, color=COLORS[0], alpha=0.2, lw=0)
        ax.plot(Ts, mean, color=COLORS[0])
        ax.axvline(1.0, color=INK_MUTED, lw=0.8, ls="--")
        ax.set_title(title, loc="left")
        ax.set_xscale("log")
        ax.set_xticks([0.25, 0.5, 1, 2, 4])
        ax.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
        ax.minorticks_off()
    lo, hi = axes[0, 0].get_ylim()
    mid = (lo + hi) / 2
    axes[0, 0].set_ylim(mid - 0.02, mid + 0.02)          # show that accuracy is flat
    fig.supxlabel("Temperature $T$ (log scale).  $T<1$: over-confident,  $T=1$: model as fitted,  "
                  "$T>1$: under-confident", fontsize=9)
    d, m = TEMPERATURE_MODEL
    fig.suptitle(f"{m} model on {d.capitalize()}, $p_T=\\sigma(\\mathrm{{logit}}(p)/T)$, "
                 f"$c_{{FP}}={C_FP:g}$, $c_{{FN}}={C_FN:g}$, $c_I={C_INSP:g}$",
                 x=0.02, ha="left", fontsize=9, color=INK_MUTED)
    fig.tight_layout()
    _preview_note(fig, preview)
    save(fig, FIGURES_DIR / "c4_temperature")


def main():
    apply_style()
    scores, pooled, temp, preview = run()
    rows = write_scores(scores, preview)
    trows = write_temperature(temp, preview)
    fig_reliability(pooled, preview)
    fig_temperature(temp, preview)
    for r in rows:
        print(f"  {r['data']:9s} {r['model']:18s} acc={r['accuracy']:.3f} logloss={r['log_loss']:.3f} "
              f"brier={r['brier']:.3f} ece={r['ece']:.3f}/{r['ece_quantile']:.3f} cost={r['cost']:.3f} "
              f"insp={r['reject_rate']:.3f}")
    for r in trows:
        if any(np.isclose(r["T"], t) for t in T_TABLE):
            print(f"  T={r['T']:<4g} acc={r['accuracy']:.4f} logloss={r['log_loss']:.3f} "
                  f"cost={r['cost']:.3f} insp={r['reject_rate']:.3f}")


if __name__ == "__main__":
    main()
