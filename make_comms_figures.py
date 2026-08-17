"""Communications figures from comms_results.json."""
import sys; sys.path.insert(0, ".")
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from comms_model import N_FREQ_SUB, N_TIME

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.2, "mathtext.fontset": "cm",
    "axes.linewidth": 0.6, "axes.titlesize": 8.6, "axes.labelsize": 8.4,
    "legend.fontsize": 7.2, "xtick.labelsize": 7.4, "ytick.labelsize": 7.4,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
BLUE, RED, GREEN, ORANGE, GRAY = "#1f5fa8", "#c23b22", "#2e7d4f", "#e08214", "#666666"
COL, FULL = 3.45, 7.1
R = json.load(open("comms_results.json"))

# ---- reconstruct quiet mask for the C1 allocation ----
m = R["maps"]
se = np.array(m["se"]); n_freq = m["n_freq"]; counts = m["greedy_counts"]
order = [np.array(o) for o in m["order"]]
quiet = np.zeros((n_freq, N_TIME), bool)
for j in range(3):
    block = order[j][:counts[j]]                 # raveled indices within block
    for flat in block:
        fr, tc = divmod(int(flat), N_TIME)
        quiet[j * N_FREQ_SUB + fr, tc] = True

CH = ["18.7 GHz", "23.8 GHz\n(EESS)", "36.5 GHz"]

# ======================================================================
# Fig A: SE map + allocation  |  per-channel NEDT bars
# ======================================================================
def figA():
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.5),
                             gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.34})
    ax = axes[0]
    im = ax.imshow(se, cmap="viridis", aspect="auto", origin="lower",
                   extent=[0, N_TIME, 0, n_freq], vmin=0, vmax=7.4)
    for i in range(n_freq):
        for t in range(N_TIME):
            if quiet[i, t]:
                ax.add_patch(Rectangle((t, i), 1, 1, fill=False,
                                       edgecolor="white", lw=0.9))
    for y in [5, 10]:
        ax.axhline(y, color="w", lw=0.8)
    ax.set_yticks([2.5, 7.5, 12.5]); ax.set_yticklabels(CH, fontsize=7)
    ax.set_xlabel("time slot")
    ax.set_title("(a) active-user opportunity cost + procured quiet tiles")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("bps/Hz", fontsize=6.8)
    ax.text(10.5, 7.5, "congestion\nhotspot", color="w", ha="center",
            va="center", fontsize=7, weight="bold")

    ax = axes[1]
    p = R["params"]; r = R["realistic"]
    x = np.arange(3); w = 0.27
    base = p["NEDT_base_K"]; ach = r["NEDT_achieved_K"]; spec = p["NEDT_spec_K"]
    ax.bar(x - w, base, w, color=GRAY, label="baseline (RFI)")
    ax.bar(x, ach, w, color=BLUE, label="after procurement")
    ax.bar(x + w, spec, w, color=GREEN, label="design spec")
    ax.set_xticks(x); ax.set_xticklabels(["18.7", "23.8", "36.5"])
    ax.set_xlabel("AMSR-2 channel [GHz]"); ax.set_ylabel("NEDT [K]")
    ax.set_title("(b) radiometric noise restored")
    ax.legend(frameon=False, fontsize=6.6, loc="upper left")
    fig.savefig("fig_comms_map.pdf"); plt.close(fig)

figA()

# ======================================================================
# Fig B: capacity-protection frontier
# ======================================================================
def figB():
    fr = R["frontier"]
    tol = [f["tpw_target"] for f in fr]
    freed = [100 * f["freed_frac"] for f in fr]
    gbps = [f["retained_Gbps"] for f in fr]
    pot = [f["PoT"] for f in fr]
    fig, ax = plt.subplots(figsize=(COL*2*0.86, 2.4))
    l1 = ax.plot(tol, freed, "o-", color=BLUE, ms=4,
                 label="secondary capacity retained")[0]
    ax.set_xlabel("IWV protection target (TPW RMS) [mm]")
    ax.set_ylabel("capacity retained [\\%]", color=BLUE)
    ax.tick_params(axis="y", labelcolor=BLUE)
    ax.set_ylim(60, 100)
    # secondary y for Gbps
    for x, gb, fp in zip(tol, gbps, freed):
        ax.annotate(f"{gb:.1f}", (x, fp), textcoords="offset points",
                    xytext=(0, 5), fontsize=6.2, color=BLUE, ha="center")
    ax2 = ax.twinx()
    l2 = ax2.plot(tol, pot, "s--", color=RED, ms=4,
                  label="price of truthfulness")[0]
    ax2.set_ylabel("price of truthfulness", color=RED)
    ax2.tick_params(axis="y", labelcolor=RED)
    ax2.set_ylim(1.0, 3.2)
    ax2.axhline(1, color=GRAY, lw=0.6, ls=":")
    ax.axvline(1.0, color="k", lw=0.6, ls="--")
    ax.text(1.01, 62, "operating\npoint", fontsize=6.6)
    ax.legend(handles=[l1, l2], frameon=False, loc="center right", fontsize=6.8)
    ax.set_title("Capacity--protection frontier (numbers: retained Gb/s)")
    fig.savefig("fig_comms_frontier.pdf"); plt.close(fig)

figB()

# ======================================================================
# Fig C: regime comparison + congestion sweep
# ======================================================================
def figC():
    r = R["realistic"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.35),
                             gridspec_kw={"wspace": 0.3})
    ax = axes[0]
    regimes = ["time-dynamic\n(mute window)", "static band\n(23.8 GHz)",
               "proposed\nprocurement"]
    vals = [r["dynamic_retained_Gbps"], r["static_retained_Gbps"],
            r["retained_Gbps"]]
    cols = [GRAY, ORANGE, BLUE]
    bars = ax.bar(range(3), vals, 0.6, color=cols)
    ax.set_xticks(range(3)); ax.set_xticklabels(regimes, fontsize=7)
    ax.set_ylabel("secondary throughput retained [Gb/s]")
    ax.set_ylim(0, 6.2)
    ax.axhline(r["total_Gbps"], color="k", lw=0.7, ls=":")
    ax.text(0.05, r["total_Gbps"]+0.06, f"total {r['total_Gbps']:.1f} Gb/s",
            fontsize=6.8)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.08, f"{v:.2f}", ha="center", fontsize=7.2)
    # feasibility annotations
    ax.text(1, 0.25, "TPW 1.39 mm\n(misses 1.0)", ha="center", color="white",
            fontsize=6.3, weight="bold")
    ax.text(2, 0.25, "TPW 1.00 mm\n(meets target)", ha="center", color="white",
            fontsize=6.3, weight="bold")
    ax.set_title("(a) capacity retained at equal sensing protection")

    ax = axes[1]
    cs = [c for c in R["congestion_sweep"] if c.get("feasible", True)]
    cf = [100 * c["congested_frac"] for c in cs]
    pot = [c["PoT"] for c in cs]
    freed = [100 * c["freed_frac"] for c in cs]
    l1 = ax.plot(cf, pot, "s-", color=RED, ms=4, label="price of truthfulness")[0]
    ax.set_xlabel("contested-band congested fraction [\\%]")
    ax.set_ylabel("price of truthfulness", color=RED)
    ax.tick_params(axis="y", labelcolor=RED); ax.set_ylim(1.0, 2.4)
    ax2 = ax.twinx()
    l2 = ax2.plot(cf, freed, "o--", color=BLUE, ms=4,
                  label="capacity retained")[0]
    ax2.set_ylabel("capacity retained [\\%]", color=BLUE)
    ax2.tick_params(axis="y", labelcolor=BLUE); ax2.set_ylim(70, 95)
    ax.annotate("in-band substitutes exhausted;\nreroute to 18.7 GHz",
                xy=(80, 2.16), xytext=(28, 1.95),
                arrowprops=dict(arrowstyle="->", lw=0.7), fontsize=6.6)
    ax.legend(handles=[l1, l2], frameon=False, loc="upper left", fontsize=6.8)
    ax.set_title("(b) network congestion vs. procurement overhead")
    fig.savefig("fig_comms_regimes.pdf"); plt.close(fig)

figC()
print("comms figures done")
