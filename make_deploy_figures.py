"""Deployment-growth figures from deploy_results.json."""
import sys; sys.path.insert(0, ".")
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.2, "mathtext.fontset": "cm",
    "axes.linewidth": 0.6, "axes.titlesize": 8.6, "axes.labelsize": 8.4,
    "legend.fontsize": 7.0, "xtick.labelsize": 7.4, "ytick.labelsize": 7.4,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
BLUE, RED, GREEN, ORANGE, GRAY = "#1f5fa8", "#c23b22", "#2e7d4f", "#e08214", "#666666"
COL, FULL = 3.45, 7.1
R = json.load(open("deploy_results.json"))
yrs = R["adoption"]["years"]
ETAS = ["7", "15", "25"]
ECOL = {"7": ORANGE, "15": BLUE, "25": GREEN}

# ======================================================================
# Fig 1: adoption+contamination | procurement response | PoT
# ======================================================================
def fig_growth():
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.1),
                             gridspec_kw={"wspace": 0.42})
    # (a) unprotected contamination vs year
    ax = axes[0]
    for e in ETAS:
        ax.plot(yrs, R["adoption"]["dT_unprot"][e], "-", color=ECOL[e],
                lw=1.3, label=f"$\\eta={e}$ bit/s/Hz")
    ax.axhline(R["adoption"]["dT_tol"], color="k", ls="--", lw=0.8)
    ax.text(2025.3, 1.3, "protection target 1 K", fontsize=6.6)
    ax.annotate("17 K", xy=(2040, 17), xytext=(2035, 14.5), fontsize=7,
                color=BLUE, arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.set_xlabel("deployment year"); ax.set_ylabel("unprotected $\\delta T$ [K]")
    ax.set_title("(a) contamination growth")
    ax.legend(frameon=False, loc="upper left")
    # (b) quiet fraction + freed capacity (eta=15)
    ax = axes[1]
    t = R["trajectory"]["15"]
    qf = [100*d["quiet_frac"] for d in t]
    ff = [100*d["freed_frac"] for d in t]
    l1 = ax.plot(yrs, qf, "s-", color=RED, ms=3, label="spectrum quieted")[0]
    ax.set_ylabel("band quieted [\\%]", color=RED)
    ax.tick_params(axis="y", labelcolor=RED); ax.set_ylim(60, 100)
    ax2 = ax.twinx()
    l2 = ax2.plot(yrs, ff, "o-", color=BLUE, ms=3, label="5G capacity retained")[0]
    ax2.set_ylabel("capacity retained [\\%]", color=BLUE)
    ax2.tick_params(axis="y", labelcolor=BLUE); ax2.set_ylim(0, 50)
    ax.set_xlabel("deployment year")
    ax.set_title("(b) procurement response ($\\eta{=}15$)")
    ax.legend(handles=[l1, l2], frameon=False, loc="center right", fontsize=6.6)
    # (c) achieved dT (flat at tol) + PoT
    ax = axes[2]
    for e in ETAS:
        t = R["trajectory"][e]
        ax.plot(yrs, [d["PoT"] for d in t], "-", color=ECOL[e], lw=1.3,
                label=f"$\\eta={e}$")
    ax.set_xlabel("deployment year"); ax.set_ylabel("price of truthfulness")
    ax.set_title("(c) procurement overhead")
    ax.axhline(1, color=GRAY, lw=0.6, ls=":")
    ax.legend(frameon=False, loc="upper right")
    fig.savefig("fig_deploy_growth.pdf"); plt.close(fig)

fig_growth()

# ======================================================================
# Fig 2: 2040 contamination-capacity frontier | permissible RFI level
# ======================================================================
def fig_frontier():
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.1),
                             gridspec_kw={"wspace": 0.34})
    ax = axes[0]
    fr = R["frontier_2040"]
    tol = [f["tol"] for f in fr]
    freed = [100*f["freed"] for f in fr]
    pot = [f["PoT"] for f in fr]
    l1 = ax.plot(tol, freed, "o-", color=BLUE, ms=4, label="capacity retained")[0]
    ax.set_xscale("log")
    ax.set_xlabel("contamination tolerance $\\delta T_{\\mathrm{tol}}$ [K]")
    ax.set_ylabel("capacity retained [\\%]", color=BLUE)
    ax.tick_params(axis="y", labelcolor=BLUE)
    ax2 = ax.twinx()
    l2 = ax2.plot(tol, pot, "s--", color=RED, ms=4, label="price of truthfulness")[0]
    ax2.set_ylabel("price of truthfulness", color=RED)
    ax2.tick_params(axis="y", labelcolor=RED)
    ax.axvline(1.0, color="k", lw=0.6, ls="--")
    ax.set_title("(a) contamination--capacity frontier (2040)")
    ax.legend(handles=[l1, l2], frameon=False, loc="center right", fontsize=6.8)
    # permissible RFI: largest per-station RFI needing no procurement
    ax = axes[1]
    for e in ETAS:
        p1 = np.array(R["permissible_dBW"][e])
        ax.plot(yrs, p1, "-", color=ECOL[e], lw=1.3,
                label=f"$\\eta_{{\\mathrm{{sp}}}}={e}$")
    ax.set_xlabel("deployment year")
    ax.set_ylabel("permissible single-BS RFI [dBW]")
    ax.set_title("(b) no-procurement RFI level ($\\delta T_{\\mathrm{tol}}{=}1$K)")
    ax.legend(frameon=False, loc="upper right", fontsize=6.8)
    ax.annotate("below: no procurement", xy=(2032, R["permissible_dBW"]["25"][7]),
                xytext=(2029.5, -181), fontsize=6.4,
                arrowprops=dict(arrowstyle="->", lw=0.6))
    fig.savefig("fig_deploy_frontier.pdf"); plt.close(fig)

fig_frontier()
print("deploy figures done")
