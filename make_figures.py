"""Generate all paper figures from results.json (PDF, IEEE two-column sizes)."""
import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow, FancyBboxPatch, FancyArrowPatch
from csp import (amsr_instance, trap_field, field_to_channels, CostField,
                 greedy_allocate)

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.2, "mathtext.fontset": "cm",
    "axes.linewidth": 0.6, "axes.titlesize": 8.6, "axes.labelsize": 8.4,
    "legend.fontsize": 7.4, "xtick.labelsize": 7.6, "ytick.labelsize": 7.6,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
BLUE, RED, GREEN, ORANGE, GRAY = "#1f5fa8", "#c23b22", "#2e7d4f", "#e08214", "#666666"
R = json.load(open("results.json"))
COL, FULL = 3.45, 7.1   # IEEE column / full width (in)

# ======================================================================
# Fig 1: sharing regimes schematic
# ======================================================================
def regimes():
    rng = np.random.default_rng(3)
    cost = rng.uniform(0.15, 0.75, (3, 12))
    cost[0, 4:9] = 0.97
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 1.85))
    titles = ["(a) Static partitioning", "(b) Time-dynamic sharing",
              "(c) Flexible market (proposed)"]
    quiet = [np.zeros((3, 12), bool)] * 3
    q0 = np.zeros((3, 12), bool); q0[0, :] = True
    q1 = np.zeros((3, 12), bool); q1[:, 4:8] = True
    q2 = np.zeros((3, 12), bool)
    order = np.argsort(cost.ravel())
    for idx in order[:14]:
        q2[np.unravel_index(idx, cost.shape)] = True
    for ax, q, t in zip(axes, [q0, q1, q2], titles):
        ax.imshow(cost, cmap="Reds", vmin=0, vmax=1, aspect="auto",
                  extent=[0, 12, 0, 3], origin="lower")
        for i in range(3):
            for k in range(12):
                if q[i, k]:
                    ax.add_patch(Rectangle((k, i), 1, 1, facecolor=BLUE,
                                           edgecolor="white", lw=0.5, alpha=0.92))
        ax.set_xticks([]); ax.set_yticks([0.5, 1.5, 2.5])
        ax.set_yticklabels(["Ch 1\n23.8 GHz", "Ch 2", "Ch 3"], fontsize=6.8)
        ax.set_xlabel("time $\\rightarrow$", fontsize=7.4)
        ax.set_title(t, fontsize=8.2)
        for s in ax.spines.values(): s.set_linewidth(0.6)
    h1 = Rectangle((0, 0), 1, 1, facecolor=BLUE)
    h2 = Rectangle((0, 0), 1, 1, facecolor="#f4a582")
    axes[1].legend([h1, h2], ["procured quiet tile", "secondary opportunity cost"],
                   loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2,
                   frameon=False, fontsize=7.4)
    fig.savefig("fig_sharing.pdf"); plt.close(fig)

regimes()

# ======================================================================
# Fig 2: architecture schematic
# ======================================================================
def architecture():
    fig, ax = plt.subplots(figsize=(FULL, 2.3)); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(-0.35, 3.4)
    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                    facecolor=fc, edgecolor="black", lw=0.7))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.6)
    def arr(x1, y1, x2, y2, text="", dy=0.13, color="black", style="-|>"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                     mutation_scale=9, lw=0.9, color=color))
        if text:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, text, ha="center",
                    fontsize=6.8, color=color)
    box(0.2, 1.9, 2.3, 1.2, "Active secondary users\n(hold tiles $\\Omega_i$,\nprivate costs $c_x$)", "#fce3dc")
    box(0.2, 0.25, 2.3, 1.2, "Passive radiometer\n(EESS incumbent)\ntolerances $\\varepsilon_k^2$", "#dceafc")
    box(4.0, 1.05, 2.6, 1.45, "Spectrum coordinator\n(SAS / database)\nruns $\\mathcal{M}_{\\mathrm{CSP}}$:\ngreedy cover + thresholds", "#e8f3e6")
    box(7.6, 1.9, 2.2, 1.2, "Winning quiet set $S_g$\n(scattered tiles)", "#dceafc")
    box(7.6, 0.25, 2.2, 1.2, "Threshold payments\n$\\theta_x=$ critical bids", "#fff3d6")
    arr(2.5, 2.6, 4.0, 2.2, "bids $b_x$ per tile")
    arr(2.5, 0.8, 4.0, 1.3, "valuation $G(S)$", dy=-0.30)
    arr(6.6, 2.15, 7.6, 2.45, "quiet schedule")
    arr(6.6, 1.35, 7.6, 0.95, "payments")
    ax.text(5.3, -0.12, "physics-grounded demand: $\\sigma_j^2 \\propto 1/B_j(S)$,"
            " $\\mathrm{Var}[\\hat y_k]=\\sum_j c_{k,j}^2\\sigma_j^2(S) \\leq \\varepsilon_k^2$",
            ha="center", fontsize=7.2, style="italic", color=GRAY)
    fig.savefig("fig_architecture.pdf"); plt.close(fig)

architecture()

# ======================================================================
# Fig 3: trap microstructure + payments vs cost
# ======================================================================
rad = amsr_instance()
fimg = trap_field(seed=0)
field = CostField(field_to_channels(fimg))
g = greedy_allocate(rad, field)
sel_mask = np.zeros((15, 20), bool)
for (j, r, b) in g["picked"]:
    flat = field.order[j][r]                       # index in channel raveled
    fr, tc = divmod(flat, 20)
    sel_mask[j * 5 + fr, tc] = True

def fig_trap():
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.45),
                             gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.45})
    ax = axes[0]
    im = ax.imshow(fimg, cmap="Reds", aspect="auto", origin="lower",
                   extent=[0, 20, 0, 15], norm=matplotlib.colors.LogNorm(vmin=1, vmax=50))
    for i in range(15):
        for t in range(20):
            if sel_mask[i, t]:
                ax.add_patch(Rectangle((t, i), 1, 1, fill=False,
                                       edgecolor=BLUE, lw=1.0))
    for y in [5, 10]:
        ax.axhline(y, color="k", lw=0.8)
    ax.set_yticks([2.5, 7.5, 12.5]); ax.set_yticklabels(["Ch 1\n23.8 GHz", "Ch 2", "Ch 3"])
    ax.set_xlabel("time slot"); ax.set_title("(a) cost field and greedy allocation")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    cb.ax.set_title("[\\$]", fontsize=7)
    ax = axes[1]
    mk = dict(zip(range(3), ["o", "s", "^"]))
    for j in range(3):
        c = R["trap"]["cost_by_ch"][f"ch{j+1}"]
        t = R["trap"]["theta_by_ch"][f"ch{j+1}"]
        ax.scatter(c, t, s=14, marker=mk[j], label=f"Ch {j+1}",
                   color=[BLUE, GREEN, ORANGE][j], alpha=0.8, lw=0)
    lim = [0.8, 60]
    ax.plot(lim, lim, "k--", lw=0.7, label="$\\theta=c$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("winner true cost $c_x$ [\\$]")
    ax.set_ylabel("threshold payment $\\theta_x$ [\\$]")
    ax.set_title("(b) payments vs. costs (greedy $\\mathcal{M}_{\\mathrm{CSP}}$)")
    ax.legend(frameon=False, loc="lower right")
    ax.annotate("trap sets the price of\ntiles it never sells",
                xy=(2.2, 31), xytext=(4.2, 6.0),
                arrowprops=dict(arrowstyle="->", lw=0.7), fontsize=7)
    fig.savefig("fig_trap.pdf"); plt.close(fig)

fig_trap()

# ======================================================================
# Fig 4: cost comparison + payment comparison
# ======================================================================
def fig_cost():
    t = R["trap"]
    fig, axes = plt.subplots(1, 2, figsize=(COL * 2, 2.1))
    ax = axes[0]
    bars = [t["C_static"], t["greedy_cost"]]
    ax.bar([0, 1], bars, 0.55, color=[GRAY, BLUE])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["fixed band\n(19 trap tiles)", "flexible CSP\n(0 trap tiles)"])
    ax.set_ylabel("procurement cost [\\$]")
    for i, b in enumerate(bars):
        ax.text(i, b + 12, f"\\${b:,.0f}", ha="center", fontsize=7.6)
    ax.set_title(f"(a) cross-channel substitution: $-{100*t['save']:.1f}\\%$")
    ax.set_ylim(0, 1200)
    ax = axes[1]
    x = np.arange(2); w = 0.38
    ax.bar(x - w / 2, [t["exact_cost"], t["greedy_cost"]], w, color=BLUE, label="cost $C$")
    ax.bar(x + w / 2, [t["P_pivot"], t["P_greedy"]], w, color=RED, label="payment $P$")
    ax.set_xticks(x); ax.set_xticklabels(["exact pivot\n(VCG benchmark)", "greedy $\\mathcal{M}_{\\mathrm{CSP}}$"])
    ax.set_ylabel("dollars")
    ax.text(0 + w / 2, t["P_pivot"] + 30, f"PoT {t['PoT_pivot']:.2f}", ha="center", fontsize=7.4)
    ax.text(1 + w / 2, t["P_greedy"] + 30, f"PoT {t['PoT_greedy']:.2f}", ha="center", fontsize=7.4)
    ax.legend(frameon=False); ax.set_title("(b) price of truthfulness (trap)")
    ax.set_ylim(0, 2300)
    fig.savefig("fig_costcompare.pdf"); plt.close(fig)

fig_cost()

# ======================================================================
# Fig 5: ensemble box plots
# ======================================================================
def fig_ens():
    e = R["ensemble"]
    fig, ax = plt.subplots(figsize=(COL, 1.9))
    data = [e["g_alloc"], e["rho"], e["PoT"], e["Phi"]]
    bp = ax.boxplot(data, vert=False, widths=0.55, showfliers=True,
                    flierprops=dict(marker=".", ms=3, alpha=0.4),
                    medianprops=dict(color=RED, lw=1.2),
                    boxprops=dict(lw=0.8), whiskerprops=dict(lw=0.8))
    ax.set_yticklabels(["$g_{\\mathrm{alloc}}$", "$\\rho$",
                        "$\\mathrm{PoT}$", "$\\Phi_{\\mathrm{e2e}}$"])
    ax.axvline(1, color=GRAY, lw=0.7, ls=":")
    ax.set_xlabel("ratio (200 random fields)")
    fig.savefig("fig_ensemble.pdf"); plt.close(fig)

fig_ens()

# ======================================================================
# Fig 6: dissociation sweeps
# ======================================================================
def fig_diss():
    a = R["sweep_scarcity"]; b = R["sweep_curvature"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.15))
    ax = axes[0]
    ax.plot(a["trap"], a["PoT"], "o-", color=RED, ms=4)
    ax.set_xlabel("trap cost [\\$/tile]"); ax.set_ylabel("PoT")
    ax.set_title(f"(a) scarcity sweep at constant curvature $c={a['curvature']:.4f}$")
    ax.axhline(1, color=GRAY, lw=0.7, ls=":")
    ax.annotate("saturation: marginal substitute\nbecomes Ch-2 reallocation",
                xy=(50, a["PoT"][5]), xytext=(28, 4.2),
                arrowprops=dict(arrowstyle="->", lw=0.7), fontsize=7)
    ax = axes[1]
    cf = np.array(b["c_full"]); med = np.array(b["PoT_med"])
    q1 = np.array(b["PoT_q1"]); q3 = np.array(b["PoT_q3"])
    o = np.argsort(cf)
    ax.plot(cf[o], med[o], "s-", color=BLUE, ms=4, label="PoT (median, 20 fields)")
    ax.fill_between(cf[o], q1[o], q3[o], color=BLUE, alpha=0.18, lw=0, label="IQR")
    ax.set_xlabel("total curvature $c$ (closed form, Thm. 6)")
    ax.set_ylabel("PoT")
    ax.set_ylim(0.9, 1.6)
    ax.axhline(1, color=GRAY, lw=0.7, ls=":")
    ax.set_title("(b) curvature sweep $c\\in[0.30,1.00]$ at flat PoT")
    ax.legend(frameon=False, loc="upper right")
    fig.savefig("fig_dissociation.pdf"); plt.close(fig)

fig_diss()

# ======================================================================
# Fig 7: curvature formula + cover gap
# ======================================================================
def fig_curv():
    cfm = R["curv_formula"]; b = R["sweep_curvature"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.15))
    ax = axes[0]
    ax.semilogx(cfm["DR"], cfm["c_exact"], color=BLUE, lw=1.2, label="closed form (Thm. 6)")
    ax.semilogx(cfm["DR"], cfm["c_lim"], "--", color=RED, lw=1.0,
                label="fine-tile limit $1-(B^{(0)}\\!/B^{\\max})^2$")
    ax.plot(cfm["amsr_DR"], cfm["amsr_c"], "k*", ms=9, label="23.8 GHz channel")
    ax.set_xlabel("bandwidth dynamic range $B^{\\max}/B^{(0)}$")
    ax.set_ylabel("total curvature $c$")
    ax.set_title("(a) curvature is bandwidth geometry")
    ax.legend(frameon=False, loc="lower right")
    ax = axes[1]
    cop = np.array(b["c_op"]); gap = np.array(b["gap"]); inv = np.array(b["inv1mc"])
    o = np.argsort(cop)
    ax.plot(cop[o], gap[o], "o-", color=GREEN, ms=4, label="measured $C(S_g)/C(S^\\star)$")
    ax.plot(cop[o], inv[o], "^--", color=GRAY, ms=4, label="$1/(1-\\bar c)$ multiplier")
    ax.set_yscale("log"); ax.set_xlabel("operating curvature $\\bar c$")
    ax.set_ylabel("cover-cost ratio")
    ax.set_title("(b) cover gap is flat in curvature")
    ax.legend(frameon=False, loc="upper left")
    fig.savefig("fig_curvature.pdf"); plt.close(fig)

fig_curv()

# ======================================================================
# Fig 8: tolerance frontier
# ======================================================================
def fig_front():
    fr = R["frontier"]
    e2 = [f["eps2"] for f in fr]; C = [f["C"] for f in fr]; P = [f["P"] for f in fr]
    PoT = [f["PoT"] for f in fr]
    fig, axes = plt.subplots(2, 1, figsize=(COL, 3.0), sharex=True,
                             gridspec_kw={"hspace": 0.12})
    ax = axes[0]
    ax.plot(e2, C, "o-", color=BLUE, ms=4, label="cost $C(S_g)$")
    ax.plot(e2, P, "s-", color=RED, ms=4, label="payment $P(S_g)$")
    ax.set_yscale("log"); ax.set_ylabel("dollars")
    ax.legend(frameon=False); ax.invert_xaxis()
    ax = axes[1]
    ax.plot(e2, PoT, "d-", color=GREEN, ms=4)
    ax.axhline(1, color=GRAY, lw=0.7, ls=":")
    ax.set_ylabel("PoT"); ax.set_xlabel("IWV tolerance $\\varepsilon^2$ [K$^2$]")
    ax.annotate("trap is marginal\nsubstitute (unbought)", xy=(0.25, PoT[3]),
                xytext=(0.43, 6.2), arrowprops=dict(arrowstyle="->", lw=0.7), fontsize=7)
    ax.annotate("trap must be bought:\ncost absorbs the rent", xy=(0.15, PoT[0]),
                xytext=(0.32, 3.6), arrowprops=dict(arrowstyle="->", lw=0.7), fontsize=7)
    fig.savefig("fig_frontier.pdf"); plt.close(fig)

fig_front()

# ======================================================================
# Fig 9: scalability
# ======================================================================
def fig_scal():
    s = R["scalability"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.1))
    ax = axes[0]
    ax.semilogy(s["n_bf"], s["t_bf"], "o-", color=RED, ms=4, label="exact brute force")
    nb = np.array(s["n_big"], float)
    ax.semilogy(s["n_bf"], np.array(s["t_gr_small"]) + 1e-6, "s-", color=BLUE, ms=4,
                label="greedy $\\mathcal{M}_{\\mathrm{CSP}}$")
    ax.set_xlabel("tiles $n$ (heterogeneous volumes)")
    ax.set_ylabel("runtime [s]"); ax.legend(frameon=False, loc="upper left")
    ax.set_title("(a) exponential vs. near-linear")
    axin = ax.inset_axes([0.58, 0.12, 0.4, 0.42])
    axin.loglog(s["n_big"], s["t_big"], "s-", color=BLUE, ms=3)
    axin.set_title("greedy to $n=10^6$", fontsize=6.4)
    axin.tick_params(labelsize=5.6)
    ax = axes[1]
    ax.semilogx(s["n_gap"], 100 * (np.array(s["gap"]) - 1), "o-", color=GREEN, ms=4)
    ax.set_xlabel("tiles $n$"); ax.set_ylabel("mean optimality gap [\\%]")
    ax.set_title("(b) gap vs. exact DP (20 seeds/point)")
    fig.savefig("fig_scalability.pdf"); plt.close(fig)

fig_scal()

# ======================================================================
# Fig 10 (new): reserve price + market thickness
# ======================================================================
def fig_res_thick():
    res = [r for r in R["reserve"] if r["feasible"]]
    rinf = [r["r"] for r in R["reserve"] if not r["feasible"]]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.15))
    ax = axes[0]
    rr = [min(r["r"], 60) if r["r"] != None else 60 for r in res]
    rr = [60 if not np.isfinite(r["r"]) else r["r"] for r in res]
    ax.plot(rr, [r["PoT"] for r in res], "o-", color=RED, ms=4, label="PoT$(r)$")
    ax.axhline(1, color=GRAY, lw=0.7, ls=":")
    if rinf:
        ax.axvspan(0, max(rinf) + 0.15, color=GRAY, alpha=0.2, lw=0)
        ax.text(max(rinf) / 2 + 0.4, 4.4, "infeasible", rotation=90, fontsize=7, color=GRAY)
    ax.set_xlabel("reserve price $r$ [\\$/tile]")
    ax.set_ylabel("PoT")
    ax.set_title("(a) reserve price caps scarcity rents (trap field)")
    ax.annotate("same allocation,\n$6.2\\times$ cheaper", xy=(2.6, res[0]["PoT"]),
                xytext=(13, 2.4), arrowprops=dict(arrowstyle="->", lw=0.7), fontsize=7)
    ax = axes[1]
    t = R["thickness"]
    ax.plot(t["N"], t["med"], "s-", color=BLUE, ms=4, label="median PoT")
    ax.fill_between(t["N"], t["q1"], t["q3"], color=BLUE, alpha=0.18, lw=0, label="IQR (30 fields)")
    ax.axhline(1, color=GRAY, lw=0.7, ls=":")
    ax.set_xlabel("tiles per channel $N_j$")
    ax.set_ylabel("PoT")
    ax.set_title("(b) market thickness drives payments to cost")
    ax.legend(frameon=False)
    fig.savefig("fig_reserve_thickness.pdf"); plt.close(fig)

fig_res_thick()

# ======================================================================
# Fig 11 (new): empirical DSIC verification
# ======================================================================
def fig_truth():
    fig, axes = plt.subplots(2, 1, figsize=(COL, 2.6), sharex=True,
                             gridspec_kw={"hspace": 0.14})
    labs = ["(a) Ch-1 winner: scarce substitutes",
            "(b) Ch-2 winner: abundant substitutes"]
    for ax, v, col, lab in zip(axes, R["truthful"], [BLUE, GREEN], labs):
        b = np.array(v["bids"]) / v["theta"]
        ax.step(b, v["util"], where="post", color=col, lw=1.3)
        ax.axvline(v["cost"] / v["theta"], color=GRAY, lw=0.7, ls=":")
        ax.plot(v["cost"] / v["theta"], v["theta"] - v["cost"], "k*", ms=9,
                label="truthful report $b_x=c_x$")
        ax.set_ylabel("utility [\\$]")
        m = v["theta"] - v["cost"]
        ax.set_ylim(-0.12 * m, 1.25 * m)
        ax.text(0.03, 0.86, lab + f"  ($c_x=\\${v['cost']:.2f}$, "
                f"$\\theta_x=\\${v['theta']:.2f}$)",
                transform=ax.transAxes, fontsize=7.2)
        ax.legend(frameon=False, loc="center right", fontsize=6.8)
    axes[1].set_xlabel("reported bid $b_x$ (normalized by $\\theta_x$)")
    fig.savefig("fig_truthful.pdf"); plt.close(fig)

fig_truth()
print("figures done")
