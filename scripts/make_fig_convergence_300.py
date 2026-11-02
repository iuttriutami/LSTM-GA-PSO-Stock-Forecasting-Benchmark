"""Convergence of the main (paper) protocol up to 300 candidate positions (complements Fig. 6, which stops at 100).

Reads results/TableV_E7a_paper_protocol_runs.csv, which holds the best validation MSE and the test MAPE of the
configuration selected after 50, 100 and 300 candidate positions for five independent GA and PSO searches.
Three checkpoints are plotted, not a per-evaluation curve. For the full curve run scripts/export_search_history.py
first; this script then draws it from results/SearchHistory_E7a.csv when that file exists.

    python scripts/make_fig_convergence_300.py        ->  figures/FigS2_paper_protocol_convergence_300.png
"""
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
ROOT = pathlib.Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
RW, COL = 1.37, {"GA": "#1f77b4", "PSO": "#d62728"}
r = pd.read_csv(RES / "TableV_E7a_paper_protocol_runs.csv")
H = RES / "SearchHistory_E7a.csv"
hist = pd.read_csv(H) if H.exists() else None

fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.4))
for opt, c in COL.items():
    d = r[r.optimizer == opt]
    g = d.groupby("budget")
    b = np.array(sorted(d.budget.unique()))
    if hist is None:
        q = g.best_val_mse.quantile([.25, .5, .75]).unstack()
        axs[0].plot(b, q[.5], "o-", color=c, label=opt); axs[0].fill_between(b, q[.25], q[.75], color=c, alpha=.2)
    else:
        for _, h in hist[hist.optimizer == opt].groupby("run"):
            axs[0].plot(h.calls, h.best_val_mse, color=c, alpha=.25, lw=.8)
        m = hist[hist.optimizer == opt].groupby("calls").best_val_mse.median()
        axs[0].plot(m.index, m.values, color=c, lw=1.6, label=opt)
    axs[1].plot(b, g.test_MAPE_mean.mean(), "o-", color=c, label=opt)
    axs[1].fill_between(b, g.test_MAPE_mean.min(), g.test_MAPE_mean.max(), color=c, alpha=.15)
    axs[2].plot(b, g.distinct_at_budget.mean(), "o-", color=c, label=opt)
axs[1].axhline(RW, color="black", lw=1); axs[1].set_ylim(1.2, 2.7); axs[1].text(120, RW + .04, "Random walk (1.37)", va="bottom", fontsize=7)
axs[2].plot([0, 300], [0, 300], ":", color="grey"); axs[2].text(20, 250, "dotted: unique = positions", fontsize=7, color="grey")
for ax, t, yl in zip(axs, ["(a) Best validation MSE", "(b) Test MAPE (%) of selected config.", "(c) Unique configurations"],
                     ["median, IQR band" if hist is None else "median (bold)", "mean, min-max band", "mean over 5 runs"]):
    ax.set_title(t, fontsize=8); ax.set_xlabel("Candidate positions"); ax.set_ylabel(yl, fontsize=7); ax.set_xticks([50, 100, 300])
axs[0].legend(frameon=False)
fig.tight_layout(); fig.savefig(FIG / "FigS2_paper_protocol_convergence_300.png", dpi=600, bbox_inches="tight")
print("saved", FIG / "FigS2_paper_protocol_convergence_300.png")
