"""Dense-venue stress-test figure from stress_results.json."""
import sys; sys.path.insert(0, ".")
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.2, "mathtext.fontset": "cm",
    "axes.linewidth": 0.6, "axes.titlesize": 8.6, "axes.labelsize": 8.4,
    "legend.fontsize": 7.2, "xtick.labelsize": 7.4, "ytick.labelsize": 7.4,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
BLUE, RED, GREEN, ORANGE, GRAY = "#1f5fa8", "#c23b22", "#2e7d4f", "#e08214", "#666666"
COL, FULL = 3.45, 7.1
R = json.load(open("stress_results.json"))
KEYS = ["Salt_Tar", "ACC"]
NAME = {"Salt_Tar": "Salt \\& Tar (outdoor)", "ACC": "ACC Arena (indoor)"}
CC = {"Salt_Tar": BLUE, "ACC": RED}

def fig_stress():
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.05),
                             gridspec_kw={"wspace": 0.42})
    # (a) empirical per-tile bid (aggregate throughput) CDF
    ax = axes[0]
    for k in KEYS:
        b = np.array(R[k]["tile_thr"]).ravel()
        b = np.sort(b) / b.max()
        ax.plot(b, np.linspace(0, 1, len(b)), lw=1.4, color=CC[k], label=NAME[k])
    ax.set_xlabel("normalized per-tile bid (agg.\\ throughput)")
    ax.set_ylabel("CDF")
    ax.set_title("(a) empirical bid vector")
    ax.legend(frameon=False, fontsize=6.6, loc="lower right")
    # annotate CV
    ax.text(0.04, 0.9, f"CV: S\\&T {R['Salt_Tar']['raw']['cv_tile']:.2f}, "
            f"ACC {R['ACC']['raw']['cv_tile']:.2f}", fontsize=6.4)
    # (b) venue load / stress characterization
    ax = axes[1]
    metrics = ["active\nUEs", "mean PRB\n(/50)", "mean BLER\n(%)"]
    st = [R["Salt_Tar"]["raw"]["active"]*100,
          R["Salt_Tar"]["raw"]["mean_prb"]/50*100,
          R["Salt_Tar"]["raw"]["mean_bler"]*100]
    ac = [R["ACC"]["raw"]["active"]*100,
          R["ACC"]["raw"]["mean_prb"]/50*100,
          R["ACC"]["raw"]["mean_bler"]*100]
    x = np.arange(3); w = 0.36
    ax.bar(x - w/2, st, w, color=BLUE, label="Salt \\& Tar")
    ax.bar(x + w/2, ac, w, color=RED, label="ACC Arena")
    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=6.6)
    ax.set_ylabel("value [\\%]")
    ax.set_title("(b) measured venue load")
    ax.legend(frameon=False, fontsize=6.6, loc="upper left")
    for i, (a, c) in enumerate(zip(st, ac)):
        ax.text(i - w/2, a + 1, f"{a:.0f}", ha="center", fontsize=6.0)
        ax.text(i + w/2, c + 1, f"{c:.0f}", ha="center", fontsize=6.0)
    # (c) mechanism outcome
    ax = axes[2]
    freed = [R[k]["freed_frac"]*100 for k in KEYS]
    pot = [R[k]["PoT"] for k in KEYS]
    x = np.arange(2)
    ax.bar(x, freed, 0.5, color=GREEN)
    ax.set_ylabel("capacity retained [\\%]", color=GREEN)
    ax.tick_params(axis="y", labelcolor=GREEN)
    ax.set_ylim(0, 100)
    for i, v in enumerate(freed):
        ax.text(i, v + 1.5, f"{v:.0f}\\%", ha="center", fontsize=6.8, color=GREEN)
    ax2 = ax.twinx()
    ax2.plot(x, pot, "s", color=RED, ms=7)
    for i, v in enumerate(pot):
        ax2.text(i + 0.08, v, f"PoT {v:.2f}", fontsize=6.8, color=RED, va="center")
    ax2.axhline(1, color=GRAY, lw=0.6, ls=":")
    ax2.set_ylabel("price of truthfulness", color=RED)
    ax2.tick_params(axis="y", labelcolor=RED)
    ax2.set_ylim(1.0, 1.4)
    ax.set_xticks(x); ax.set_xticklabels(["Salt \\& Tar", "ACC Arena"], fontsize=7)
    ax.set_title("(c) procurement outcome")
    fig.savefig("fig_stress.pdf"); plt.close(fig)

fig_stress()
print("stress figure done")
