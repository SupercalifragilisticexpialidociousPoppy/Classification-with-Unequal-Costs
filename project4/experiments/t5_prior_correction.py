"""T5 figure (Member 5): what the prior-shift correction does. Pure formula, no data.

Outputs
  figures/t5_prior_correction.{pdf,png}
    left : corrected posterior p' against the model's p for several deployment priors
    right: the decision thresholds on the model's (uncorrected) p that give the
           corrected Bayes rule, against the deployment prior
"""
import numpy as np

from project4.config import C_FN, C_FP, C_INSP, FIGURES_DIR
from project4.decision import LossMatrix
from project4.plotting import COLORS, INK_MUTED, apply_style, plt, save
from project4.prior_shift import adjust_posterior, shifted_thresholds

# ---- configuration --------------------------------------------------------
PI1_TRAIN = 0.5                      # prior the model was trained at (as in the C5 synthetic setup)
PI1_NEW = (0.05, 0.2, 0.8)           # deployment priors drawn on the left panel
PI1_GRID = np.linspace(0.01, 0.99, 197)


def main():
    apply_style()
    L = LossMatrix(C_FP, C_FN, C_INSP)
    p = np.linspace(0, 1, 401)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))

    ax = axes[0]
    ax.plot(p, p, color=INK_MUTED, lw=0.8, ls="--")
    for k, pi1 in enumerate(PI1_NEW):
        ax.plot(p, adjust_posterior(p, PI1_TRAIN, pi1), color=COLORS[k], label=rf"$\pi_1'={pi1:g}$")
    ax.annotate("no shift", (0.45, 0.45), xytext=(6, -12), textcoords="offset points",
                fontsize=8, color=INK_MUTED)
    ax.set_xlabel(r"Model's probability $p$ (trained at $\pi_1=%g$)" % PI1_TRAIN)
    ax.set_ylabel(r"Corrected probability $p'$")
    ax.set_title("Correcting the probability", loc="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    th = np.array([shifted_thresholds(L, PI1_TRAIN, q) for q in PI1_GRID])
    ax.fill_between(PI1_GRID, th[:, 0], th[:, 1], color=COLORS[0], alpha=0.15, lw=0)
    ax.fill_between(PI1_GRID, th[:, 1], 1.0, color=COLORS[1], alpha=0.15, lw=0)
    ax.plot(PI1_GRID, th[:, 0], color=COLORS[0], lw=1.6)
    ax.plot(PI1_GRID, th[:, 1], color=COLORS[0], lw=1.6)
    ax.axvline(PI1_TRAIN, color=INK_MUTED, lw=0.8, ls="--")
    ax.text(PI1_TRAIN + 0.01, 0.02, "no shift", fontsize=8, color=INK_MUTED)
    ax.text(0.33, 0.66, "inspect", fontsize=8, color=INK_MUTED)
    ax.text(0.12, 0.2, "deliver", fontsize=8, color=INK_MUTED)
    ax.text(0.985, 0.9, "spam", fontsize=8, color=INK_MUTED, ha="right")
    ax.set_xlabel(r"Deployment spam share $\pi_1'$")
    ax.set_ylabel(r"Threshold on the model's $p$")
    ax.set_title("Or: keep $p$, move the thresholds", loc="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.suptitle(rf"$c_{{FP}}={C_FP:g}$, $c_{{FN}}={C_FN:g}$, $c_I={C_INSP:g}$", x=0.02, ha="left",
                 fontsize=9, color=INK_MUTED)
    fig.tight_layout()
    save(fig, FIGURES_DIR / "t5_prior_correction")


if __name__ == "__main__":
    main()
