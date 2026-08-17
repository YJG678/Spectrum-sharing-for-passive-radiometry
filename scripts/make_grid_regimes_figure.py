"""Three-regime coexistence figure on the 3x5x20 grid (real allocation)."""
import sys; sys.path.insert(0, ".")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from csp import greedy_allocate
from comms_model import build_radiometer, comms_cost_field, N_FREQ_SUB, N_TIME

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.2, "mathtext.fontset": "cm",
    "axes.linewidth": 0.6, "axes.titlesize": 8.6, "axes.labelsize": 8.4,
    "legend.fontsize": 7.2, "xtick.labelsize": 7.4, "ytick.labelsize": 7.4,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
BLUE = "#1f5fa8"; FULL = 7.1

rad, meta = build_radiometer()
field = comms_cost_field(meta, seed=0)
g = greedy_allocate(rad, field)
se = field._se                                    # 15 x 20 spectral-efficiency
n_freq = 3 * N_FREQ_SUB

# proposed quiet mask from the real allocation (cheapest tiles per channel)
prop = np.zeros((n_freq, N_TIME), bool)
for j in range(3):
    for flat in field.order[j][:g["counts"][j]]:
        fr, tc = divmod(int(flat), N_TIME)
        prop[j * N_FREQ_SUB + fr, tc] = True

# static: 23.8 GHz group = channel index 1 -> rows 5..9, all columns
static = np.zeros((n_freq, N_TIME), bool); static[5:10, :] = True
# best fixed-window: optimized W* = seven lowest-throughput slots (all bands)
import json
bw = json.load(open("bestwindow.json"))
W = bw["bestW"]
window = np.zeros((n_freq, N_TIME), bool)
for t in W:
    window[:, t] = True

CH = ["18.7", "23.8\n(EESS)", "36.5"]
def draw(ax, mask, title, edge=True):
    ax.imshow(se, cmap="viridis", aspect="auto", origin="lower",
              extent=[0, N_TIME, 0, n_freq], vmin=0, vmax=7.4)
    for i in range(n_freq):
        for t in range(N_TIME):
            if mask[i, t]:
                ax.add_patch(Rectangle((t, i), 1, 1,
                             facecolor="white", edgecolor=BLUE,
                             lw=0.5 if edge else 0, alpha=0.75))
    for y in (5, 10):
        ax.axhline(y, color="w", lw=0.8)
    ax.set_yticks([2.5, 7.5, 12.5]); ax.set_yticklabels(CH, fontsize=6.8)
    ax.set_xlabel("time slot")
    ax.set_title(title, fontsize=8.2)

fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.05), gridspec_kw={"wspace": 0.28})
draw(axes[0], static,
     f"(a) static partitioning\n(100 tiles, TPW 1.39\\,mm: infeasible)")
draw(axes[1], window,
     f"(b) best fixed-window muting\n(105 tiles, TPW 0.97\\,mm, 4.06\\,Gb/s)")
draw(axes[2], prop,
     f"(c) proposed procurement\n(69 tiles, TPW 1.00\\,mm, 4.97\\,Gb/s)")
# shared colorbar
cb = fig.colorbar(axes[2].images[0], ax=axes, fraction=0.014, pad=0.01)
cb.set_label("secondary SE [bps/Hz]", fontsize=7)
fig.savefig("fig_grid_regimes.pdf")
plt.close(fig)
print("grid-regimes figure done; proposed counts", g["counts"].tolist())
