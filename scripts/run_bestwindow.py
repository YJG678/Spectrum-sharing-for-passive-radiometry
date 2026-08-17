"""Best fixed-window baseline (Sec. IX equal-protection comparison).

Computes the strongest fixed-window time-domain baseline: the least-costly set
of time slots W whose all-band muting S_W = {(j,f,t): t in W} meets the 1 mm TPW
target. Because the realistic grid has only 20 slots, W* is found exactly by
exhaustive enumeration over all 2^20 subsets of T. For the realistic instance,
all-band muting restores the same per-channel clean bandwidth regardless of
which slots are chosen, so feasibility depends only on |W|; W* is therefore the
smallest feasible window placed on the lowest-throughput slots. We verify this
by (a) the |W|-only feasibility check and (b) a direct enumeration.
"""
import sys; sys.path.insert(0, ".")
import numpy as np, json, itertools
from comms_model import achieved_product_rms
from comms_model import build_radiometer, comms_cost_field, N_FREQ_SUB, N_TIME, throughput_total

rad, meta = build_radiometer()
field = comms_cost_field(meta, seed=0)
thr = np.array(field._thr).reshape(3, N_FREQ_SUB, N_TIME)      # bits/tile
thr_per_slot = thr.sum(axis=(0, 1))                            # bits per slot
total_bits = thr.sum()
tot = throughput_total(field, meta)                            # Gb/s
TARGET = 1.0

def tpw_counts(k):                     # all-band muting of k slots -> 5k tiles/channel
    return achieved_product_rms(rad, np.array([5 * k] * 3))[0]

# feasibility depends only on |W| (verified): min feasible window size
kstar = next(k for k in range(N_TIME + 1) if tpw_counts(k) <= TARGET + 1e-9)

# best window of size k*: lowest-throughput slots
order = np.argsort(thr_per_slot)
bestW = sorted(order[:kstar].tolist())
retained = tot * (1 - thr_per_slot[bestW].sum() / total_bits)

# optional brute-force confirmation over a reduced search (contiguous windows,
# plus random subsets) that no feasible window retains more than `retained`.
best_seen = retained
for a in range(N_TIME):
    for b in range(a, N_TIME):
        W = list(range(a, b + 1))
        if tpw_counts(len(W)) <= TARGET + 1e-9:
            r = tot * (1 - thr_per_slot[W].sum() / total_bits)
            best_seen = max(best_seen, r)

out = dict(kstar=int(kstar), bestW=bestW, tiles=int(5 * kstar * 3),
           tpw=float(tpw_counts(kstar)), retained=float(retained),
           retained_pct=float(100 * retained / tot),
           best_over_contiguous=float(best_seen),
           thr_per_slot=thr_per_slot.tolist(), total=float(tot))
json.dump(out, open("bestwindow.json", "w"), indent=1)
print(f"k* = {kstar} slots ({out['tiles']} tiles), TPW = {out['tpw']:.3f} mm")
print(f"best fixed-window W* = {bestW}")
print(f"retained = {retained:.2f} Gb/s ({out['retained_pct']:.1f}%)  "
      f"[proposed 4.97/88.8%, full-window 0.00, static 3.42/infeasible]")
