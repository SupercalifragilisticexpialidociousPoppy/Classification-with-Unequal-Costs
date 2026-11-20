"""Member 4 (C6). Run with python -m project4.run_all --only task5_inspection."""
import csv
from collections import defaultdict

import numpy as np

from project4 import data, metrics
from project4.analytic import GaussianModel1D
from project4.config import DEFAULT_LOSS, FIGURES_DIR, RESULTS_DIR, SEEDS, T6_MU, T6_PI1
from project4.decision import band_action, threshold_action
from project4.models.gaussian import select_reg
from project4.models.logistic import LogisticRegression
from project4.plotting import COLORS, INK_MUTED, apply_style, plt, save


SYNTHETIC_SIZE = 10_000
SYNTHETIC_SEED_OFFSET = 10_000
DATASETS = ("Synthetic", "Spambase")
MODELS = ("Gaussian", "Logistic")
L2_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
INSPECTION_COSTS = np.unique(np.r_[
    np.round(np.linspace(0, 1.2, 61), 2), DEFAULT_LOSS.c_insp, DEFAULT_LOSS.critical_c_insp
])


def fit_models(X_tr, y_tr, X_va, y_va):
    """Return selected training fits. Both selection criteria use validation labels only."""
    reg, gaussian, _ = select_reg(X_tr, y_tr, X_va, y_va, shared_cov=True)
    yield "Gaussian", reg, gaussian

    best_score = np.inf
    for l2 in L2_GRID:
        model = LogisticRegression(l2=l2).fit(X_tr, y_tr)
        # Exact log-loss in logit space, including very confident predictions.
        signed_logits = (1.0 - 2.0 * y_va) * model.decision_function(X_va)
        score = np.mean(np.logaddexp(0, signed_logits))
        if score < best_score:
            best_score, best_l2, best_model = score, l2, model
    yield "Logistic", best_l2, best_model


def evaluate(y, logits):
    """Compare policies on held-out posterior log-odds before sigmoid rounding."""
    baseline = threshold_action(logits, np.log(DEFAULT_LOSS.c_fp) - np.log(DEFAULT_LOSS.c_fn))
    base_cost = metrics.realized_loss(y, baseline, DEFAULT_LOSS)

    def record(rule, actions, loss):
        counts = metrics.confusion_with_reject(y, actions)
        cost = metrics.realized_loss(y, actions, loss)
        return dict(rule=rule, c_insp=loss.c_insp, n=len(y),
                    fp=counts[0, 1], fn=counts[1, 0], inspected=counts[:, 2].sum(),
                    cost=cost, reject_rate=metrics.reject_rate(actions),
                    saving=base_cost - cost,
                    accuracy=metrics.accuracy(y, actions) if rule != "inspect" else None)

    yield record("accuracy", threshold_action(logits, 0.0), DEFAULT_LOSS)
    yield record("two_action", baseline, DEFAULT_LOSS)
    for c_insp in INSPECTION_COSTS:
        loss = DEFAULT_LOSS.with_c_insp(c_insp)
        if loss.has_reject_band:
            # Odds at the band edges are c_I/(c_FN-c_I) and (c_FP-c_I)/c_I.
            # Free inspection gives an open band over all finite log-odds.
            log_price = np.log(c_insp) if c_insp > 0 else -np.inf
            lower = log_price - np.log(loss.c_fn - c_insp)
            upper = np.log(loss.c_fp - c_insp) - log_price
            actions = band_action(logits, lower, upper)
        else:
            actions = baseline
        yield record("inspect", actions, loss)


def run():
    spambase = data.load_spambase()
    rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            if dataset == "Synthetic":
                X, y = data.make_gaussian(SYNTHETIC_SIZE, T6_PI1, -T6_MU, T6_MU,
                                          1, 1, SYNTHETIC_SEED_OFFSET + seed)
            else:
                X, y = spambase
            split = data.preprocess(data.split(X, y, seed))
            for name, penalty, model in fit_models(split.X_tr, split.y_tr, split.X_va, split.y_va):
                logits = model.decision_function(split.X_te)
                if logits.shape != split.y_te.shape or not np.isfinite(logits).all():
                    raise ValueError(f"Invalid log-odds from {name} on {dataset}, seed {seed}")
                for row in evaluate(split.y_te, logits):
                    rows.append(dict(dataset=dataset, seed=seed, model=name, penalty=penalty, **row))
    return rows


def aggregate(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["dataset"], r["model"], r["rule"], r["c_insp"]].append(r)
    summary = {}
    for key, group in groups.items():
        dataset, model, rule, c_insp = key
        result = dict(dataset=dataset, model=model, rule=rule, c_insp=c_insp, splits=len(group))
        for field in ("cost", "reject_rate", "saving", "accuracy"):
            if field == "accuracy" and rule == "inspect":
                result[field], result[field + "_se"] = None, None
            else:
                result[field], result[field + "_se"] = metrics.mean_and_se([r[field] for r in group])
        summary[key] = result
    return summary


def write_csv(filename, rows):
    rows = list(rows)
    with (RESULTS_DIR / filename).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_tables(summary):
    header = [r"\begin{tabular}{llrrrr}", r"\toprule"]
    rules = header + [r" & & \multicolumn{2}{c}{$t=0.5$} & \multicolumn{2}{c}{$t=\tstar$}\\",
                      r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
                      r"Data & Model & Accuracy & Cost & Accuracy & Cost\\", r"\midrule"]
    savings = header + [r"Data & Model & Two actions & With inspection & Saving & Inspected\\",
                        r"\midrule"]

    def cell(row, field):
        return rf"\({row[field]:.3f}\pm{row[field + '_se']:.3f}\)"

    for dataset in DATASETS:
        for model in MODELS:
            acc, two, insp = [summary[dataset, model, rule, DEFAULT_LOSS.c_insp]
                              for rule in ("accuracy", "two_action", "inspect")]
            label = f"{dataset} & {model} & "
            rules.append(label + " & ".join(cell(r, f) for r in (acc, two)
                                             for f in ("accuracy", "cost")) + r"\\")
            values = [cell(two, "cost"), cell(insp, "cost"), cell(insp, "saving"),
                      cell(insp, "reject_rate")]
            savings.append(label + " & ".join(values) + r"\\")
    for filename, lines in (("c6_rules.tex", rules), ("c6_savings.tex", savings)):
        lines += [r"\bottomrule", r"\end{tabular}"]
        (RESULTS_DIR / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_curves(summary):
    analytic = GaussianModel1D(T6_MU, T6_PI1).curves_vs_c_insp(
        INSPECTION_COSTS, DEFAULT_LOSS.c_fp, DEFAULT_LOSS.c_fn)
    for dataset in DATASETS:
        fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.1), sharex=True)
        for ax, field, oracle_field, ylabel in zip(
                axes, ("cost", "reject_rate"), ("bayes_risk", "reject_rate"),
                ("Realized cost per email", "Fraction inspected")):
            for i, model in enumerate(MODELS):
                rows = [summary[dataset, model, "inspect", ci] for ci in INSPECTION_COSTS]
                mean = np.array([r[field] for r in rows])
                se = np.array([r[field + "_se"] for r in rows])
                ax.fill_between(INSPECTION_COSTS, mean - se, mean + se, color=COLORS[i], alpha=0.23)
                ax.plot(INSPECTION_COSTS, mean, color=COLORS[i], label=model,
                        linestyle=("-", "--")[i], linewidth=1.7)
            if dataset == "Synthetic":
                ax.plot(INSPECTION_COSTS, analytic[oracle_field], ":", color="black",
                        linewidth=1.6, label="Analytic Bayes")
            ax.axvline(DEFAULT_LOSS.critical_c_insp, color=INK_MUTED, ls="-.", lw=1,
                       label=r"Cutoff $10/11$")
            ax.set(xlabel=r"Inspection cost $c_I$", ylabel=ylabel, xlim=(0, INSPECTION_COSTS[-1]))
            ax.set_ylim(bottom=0)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8)
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        save(fig, FIGURES_DIR / f"c6_inspection_{dataset.lower()}")


def main():
    rows = run()
    summary = aggregate(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv("c6_per_split.csv", rows)
    write_csv("c6_summary.csv", summary.values())
    write_tables(summary)
    apply_style()
    plot_curves(summary)


if __name__ == "__main__":
    main()
