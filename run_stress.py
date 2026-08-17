"""
Dense-Venue Stress Test.
Uses one 500-user block from each LCR HDD venue (Maheshwari et al., Sci. Data
2025). Aggregates the measured per-user throughput / PRB / BLER time series into
the existing 300-tile mechanism grid (3 channels x 5 sub-bands x 20 windows) and
replaces the synthetic bid vector with an EMPIRICAL dense-venue bid vector
b_x = aggregate served throughput on tile x (revenue foregone by quieting it).
The incumbent-protection physics stays the 23.8 GHz AMSR-2 retrieval model.
"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd, json, time
from csp import (greedy_allocate, greedy_thresholds, exact_optimum,
                 pivot_payments, minimal_feasible_counts, verify_A1, CostField)
from comms_model import build_radiometer

U = ""
GRID_ROWS, GRID_COLS = 15, 20      # 3 channels x 5 sub-bands, 20 windows
N_GROUPS = GRID_ROWS               # user resource-groups
EPS = 1e-9

def js(x):
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, np.floating): return float(x)
    if isinstance(x, np.ndarray): return x.tolist()
    return x

def load(pre, metric):
    df = pd.read_csv(U + f"{pre}_{metric}_UE_Id_0_499.csv")
    X = df.iloc[:, 1:].values.astype(float)     # (T, 500)
    return np.nan_to_num(X, nan=0.0)

def aggregate(pre):
    """Aggregate a 500-user block into 15x20 tile arrays for thr/prb/bler."""
    thr = load(pre, "Throughput")               # Mbps
    prb = load(pre, "PRBS")
    bler = load(pre, "BLER")
    T = min(thr.shape[0], prb.shape[0], bler.shape[0])
    thr, prb, bler = thr[:T], prb[:T], bler[:T]
    # 20 contiguous time windows
    win = np.array_split(np.arange(T), GRID_COLS)
    # 15 user resource-groups: neutral contiguous user-id blocks (reproducible,
    # preserves any natural user/location structure in the block ordering)
    groups = np.array_split(np.arange(thr.shape[1]), N_GROUPS)
    Tthr = np.zeros((GRID_ROWS, GRID_COLS))
    Tprb = np.zeros((GRID_ROWS, GRID_COLS))
    Tbler = np.zeros((GRID_ROWS, GRID_COLS))
    for r, gid in enumerate(groups):
        for c, wi in enumerate(win):
            sub_t = thr[np.ix_(wi, gid)]
            sub_p = prb[np.ix_(wi, gid)]
            sub_b = bler[np.ix_(wi, gid)]
            Tthr[r, c] = sub_t.sum()                     # aggregate served Mbps
            Tprb[r, c] = sub_p.mean()                    # mean PRB occupancy
            # throughput-weighted BLER (reliability where traffic actually flows)
            w = sub_t.sum()
            Tbler[r, c] = (sub_b * sub_t).sum() / w if w > EPS else sub_b.mean()
    return dict(thr=Tthr, prb=Tprb, bler=Tbler,
                raw=dict(active=float((thr > 0).mean()),
                         mean_thr=float(thr.mean()), mean_prb=float(prb.mean()),
                         mean_bler=float(bler.mean()),
                         agg_per_ts=float(thr.sum(axis=1).mean()),
                         cv_tile=float(Tthr.std() / max(Tthr.mean(), EPS)),
                         n_time=T))

def to_field(Ttile):
    """15x20 tile array -> CostField over 3 channels (rows 5j..5j+4)."""
    bids = [Ttile[j*5:(j+1)*5, :].ravel() for j in range(3)]
    # guard against all-zero: idle tiles cost ~0 (free to quiet); keep tiny floor
    bids = [np.maximum(b, EPS) for b in bids]
    return CostField(bids)

R = {}; T0 = time.time()
rad, rmeta = build_radiometer()                 # AMSR-2, 1 mm TPW
counts = minimal_feasible_counts(rad, [100, 100, 100])

for venue, pre in [("Salt & Tar", "Salt_Tar"), ("ACC Arena", "ACC")]:
    agg = aggregate(pre)
    field = to_field(agg["thr"])
    thr_flat = [agg["thr"][j*5:(j+1)*5, :].ravel() for j in range(3)]
    total_thr = sum(t.sum() for t in thr_flat)          # Mbps.window units
    assert verify_A1(rad, field), f"A1 fails {venue}"
    g = greedy_allocate(rad, field)
    opt = exact_optimum(rad, field, counts)
    th = greedy_thresholds(rad, field, g); P = sum(th.values())
    qq = pivot_payments(rad, field, opt, counts); Pp = sum(qq.values())
    # retained throughput = kept (non-silenced) tiles' aggregate throughput
    kept = 0.0
    for j, nq in enumerate(g["counts"]):
        kept += thr_flat[j][field.order[j]][nq:].sum()
    R[pre] = dict(
        venue=venue,
        raw=agg["raw"],
        total_thr=total_thr, retained_thr=kept, freed_frac=kept/total_thr,
        counts=js(g["counts"]), quiet_frac=sum(g["counts"])/300,
        cost=g["cost"], P=P, PoT=P/g["cost"], rho=P/Pp,
        g_alloc=g["cost"]/opt["cost"],
        # tile-level summaries for the figure
        tile_thr=js(agg["thr"]), tile_prb=js(agg["prb"]), tile_bler=js(agg["bler"]),
        mean_tile_prb=float(agg["prb"].mean()),
        mean_tile_bler=float(agg["bler"].mean()),
        quiet_mask_note="cheapest tiles by empirical throughput are quieted")
    print(f"{venue}: active {agg['raw']['active']*100:.1f}%  "
          f"meanPRB {agg['raw']['mean_prb']:.1f}  meanBLER {agg['raw']['mean_bler']*100:.1f}%  "
          f"agg/ts {agg['raw']['agg_per_ts']:.0f} Mbps | "
          f"freed {kept/total_thr*100:.1f}%  quiet {sum(g['counts'])/3:.0f}%  "
          f"PoT {P/g['cost']:.2f}  g_alloc {g['cost']/opt['cost']:.3f}  rho {P/Pp:.3f}")

with open("stress_results.json", "w") as fh:
    json.dump(R, fh, indent=1, default=js)
print("TOTAL", round(time.time()-T0, 1), "s")


# ---- robustness to user re-grouping (reported in the paper) ----
def robustness(pre, n=30):
    thr = load(pre, "Throughput"); T = thr.shape[0]
    win = np.array_split(np.arange(T), GRID_COLS)
    fr, pot = [], []
    for s in range(n):
        uo = np.random.default_rng(s).permutation(thr.shape[1])
        grp = np.array_split(uo, N_GROUPS)
        tile = np.zeros((GRID_ROWS, GRID_COLS))
        for r, gi in enumerate(grp):
            for c, wi in enumerate(win):
                tile[r, c] = thr[np.ix_(wi, gi)].sum()
        f = to_field(tile)
        tf = [tile[j*5:(j+1)*5, :].ravel() for j in range(3)]
        if not verify_A1(rad, f):
            continue
        g = greedy_allocate(rad, f)
        th = greedy_thresholds(rad, f, g); P = sum(th.values())
        kept = sum(tf[j][f.order[j]][g["counts"][j]:].sum() for j in range(3))
        fr.append(kept / sum(t.sum() for t in tf)); pot.append(P / g["cost"])
    return np.array(fr), np.array(pot)

if __name__ == "__main__" and "--robust" in sys.argv:
    for pre in ["Salt_Tar", "ACC"]:
        fr, pot = robustness(pre)
        print(f"{pre} robustness: freed [{fr.min()*100:.0f},{fr.max()*100:.0f}]% "
              f"PoT median {np.median(pot):.2f} range [{pot.min():.2f},{pot.max():.2f}]")
