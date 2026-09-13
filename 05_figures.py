"""
Step 5: Fig. 1 -- the SI label against degree (vector PDF, one column wide).

Plots influence capability IC at the selected horizon against degree for every
gene, with the 75th-percentile threshold that defines the "key regulator"
label, so the reader can see that the label is close to a degree cut-off.

Writes: ../latex/figures/fig_ic_degree.pdf
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIG = os.path.join(HERE, "..", "latex", "figures")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "pdf.fonttype": 42,
})


def main():
    os.makedirs(FIG, exist_ok=True)
    d = pd.read_csv(os.path.join(DATA, "features.csv")).merge(
        pd.read_csv(os.path.join(DATA, "ic_scores.csv")), on="gene")
    sens = json.load(open(os.path.join(DATA, "label_sensitivity.json")))
    horizon = max(sens["horizons"], key=lambda k: sens["horizons"][k]["cv"])
    icv = 100 * d["ic_" + horizon].to_numpy()
    deg = d["degree"].to_numpy(float)
    thr = np.percentile(icv, 75)
    pos = icv > thr

    fig, ax = plt.subplots(figsize=(3.45, 2.05))
    ax.scatter(deg[~pos], icv[~pos], s=3, c="#9aa5b1", lw=0, label="label 0")
    ax.scatter(deg[pos], icv[pos], s=3, c="#1f4e79", lw=0, label="label 1 (top quartile)")
    ax.axhline(thr, color="#c0504d", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("Degree")
    ax.set_ylabel(f"IC (% of genes infected, $T={horizon[1:]}$)")
    rho = spearmanr(icv, deg)[0]
    ax.text(0.03, 0.95, rf"Spearman $\rho = {rho:.3f}$", transform=ax.transAxes, va="top")
    ax.legend(frameon=False, loc="lower right", markerscale=2.5, handletextpad=0.2)
    ax.tick_params(width=0.6, length=2)
    fig.tight_layout(pad=0.3)
    out = os.path.join(FIG, "fig_ic_degree.pdf")
    fig.savefig(out)
    print(f"wrote {out}: horizon {horizon}, threshold {thr:.3f}%, rho {rho:.3f}, "
          f"{pos.sum()} positives")


if __name__ == "__main__":
    main()
