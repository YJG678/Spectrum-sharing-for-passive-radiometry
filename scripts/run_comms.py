"""Communication-system experiments with realistic AMSR-2 + 5G-NR parameters."""
import sys; sys.path.insert(0, ".")
import numpy as np, json, time
from csp import *
from comms_model import *

R = {}; T0 = time.time()
def js(x):
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, np.floating): return float(x)
    if isinstance(x, np.ndarray): return x.tolist()
    return x

rad, meta = build_radiometer()
R["params"] = dict(
    channels=meta["channels"],
    B_full_MHz=js(meta["B_full"]), B0_MHz=js(np.round(meta["B0"],2)),
    NEDT_spec_K=js(meta["NEDT"]), kappa=js(meta["kappa"]),
    df_tile_MHz=js(np.round(meta["df_tile"]/1e6,2)),
    dt_tile_ms=meta["dt_tile"]*1e3, tau_ms=meta["tau"]*1e3,
    SE_cap=SE_CAP, eps2_mm2=js(rad.eps2),
    NEDT_base_K=js(np.round(achieved_nedt(rad,meta,[0,0,0]),3)),
    NEDT_full_K=js(np.round(achieved_nedt(rad,meta,[100,100,100]),3)),
    TPW_base_mm=float(achieved_product_rms(rad,[0,0,0])[0]),
    TPW_full_mm=float(achieved_product_rms(rad,[100,100,100])[0]),
    Nj=100, sens_IWV=[-0.40,0.90,-0.10])

# ======================================================================
# C1: realistic instance + three coexistence regimes
# ======================================================================
field = comms_cost_field(meta, seed=0)
counts = minimal_feasible_counts(rad, field.N)
g = greedy_allocate(rad, field); opt = exact_optimum(rad, field, counts)
th = greedy_thresholds(rad, field, g); P = sum(th.values())
R_tot = throughput_total(field, meta)

# per-channel total throughput (Gbps)
ch_thr = [field._thr[j].sum()/meta["tau"]/1e9 for j in range(3)]

# regime A: static partitioning -- protect the 23.8 band fully (ch index 1)
staticA_counts = [0, 100, 0]
staticA_feasible = rad.feasible_counts(np.array(staticA_counts))
R_staticA = throughput_retained(field, meta, staticA_counts)
# regime A': single-band procurement confined to 23.8 (min tiles there, else infeasible)
band_only = None
for n2 in range(101):
    if rad.feasible_counts(np.array([0, n2, 0])):
        band_only = [0, n2, 0]; break
# regime B: time-dynamic -- mute all channels in the window
R_dynamic = 0.0
# regime C: proposed flexible procurement
R_flex = throughput_retained(field, meta, g["counts"])

R["realistic"] = dict(
    total_Gbps=R_tot, ch_thr_Gbps=ch_thr,
    greedy_counts=js(g["counts"]), exact_counts=js(opt["counts"]),
    g_alloc=g["cost"]/opt["cost"],
    NEDT_achieved_K=js(np.round(achieved_nedt(rad,meta,g["counts"]),3)),
    TPW_achieved_mm=float(achieved_product_rms(rad,g["counts"])[0]),
    retained_Gbps=R_flex, freed_frac=R_flex/R_tot,
    PoT=P/g["cost"], P=P, C=g["cost"],
    static_counts=staticA_counts, static_feasible=bool(staticA_feasible),
    static_TPW_mm=float(achieved_product_rms(rad,staticA_counts)[0]),
    static_retained_Gbps=R_staticA, static_freed_frac=R_staticA/R_tot,
    bandonly_counts=band_only,
    bandonly_feasible=band_only is not None,
    bandonly_retained_Gbps=(throughput_retained(field,meta,band_only)
                            if band_only else 0.0),
    bandonly_TPW_mm=(float(achieved_product_rms(rad,band_only)[0])
                     if band_only else None),
    dynamic_retained_Gbps=R_dynamic,
    # freed area-time fraction in the contested 23.8 band
    contested_freed_flex=1 - g["counts"][1]/100,
    contested_freed_static=0.0,
    A1=verify_A1(rad, field))
print("C1 done", round(time.time()-T0,1))

# ======================================================================
# C2: capacity-vs-protection frontier (sweep TPW tolerance)
# ======================================================================
front = []
for eps2 in [0.64, 0.81, 1.00, 1.21, 1.44, 1.96, 2.56]:
    rs, _ = build_radiometer(eps2=[eps2])
    gg = greedy_allocate(rs, field)
    if gg is None:
        continue
    cf = minimal_feasible_counts(rs, field.N)
    oo = exact_optimum(rs, field, cf)
    tt = greedy_thresholds(rs, field, gg); Pp = sum(tt.values())
    Rf = throughput_retained(field, meta, gg["counts"])
    front.append(dict(
        eps2=eps2, tpw_target=float(np.sqrt(eps2)),
        tpw_achieved=float(achieved_product_rms(rs, gg["counts"])[0]),
        nedt238=float(achieved_nedt(rs, meta, gg["counts"])[1]),
        counts=js(gg["counts"]), retained_Gbps=Rf, freed_frac=Rf/R_tot,
        PoT=Pp/gg["cost"], C=gg["cost"], g_alloc=gg["cost"]/oo["cost"]))
R["frontier"] = front
print("C2 done", round(time.time()-T0,1))

# ======================================================================
# C3: congestion-extent sweep (fraction of contested band that is hot)
# As the high-cost congested region widens across the 23.8 GHz band, cheap
# substitutes in that band vanish, so the price of truthful procurement rises
# and freed capacity falls -- the 5G-network analog of substitute scarcity.
# ======================================================================
loads = []
for width in [0, 3, 6, 9, 12, 16, 20]:            # hot time-slots in 23.8 band
    cols = range(0, width)
    fl = comms_cost_field(meta, seed=0, hotspot_rows=range(5, 10),
                          hotspot_cols=cols, hotspot_load=4.0, hotspot_bump=18.0)
    gg = greedy_allocate(rad, fl)
    if gg is None:
        loads.append(dict(width=width, feasible=False)); continue
    tt = greedy_thresholds(rad, fl, gg); Pp = sum(tt.values())
    Rf = throughput_retained(fl, meta, gg["counts"])
    loads.append(dict(width=width, congested_frac=width/20.0,
                      feasible=True, counts=js(gg["counts"]),
                      retained_Gbps=Rf, freed_frac=Rf/throughput_total(fl,meta),
                      contested_freed=1 - gg["counts"][1]/100,
                      PoT=Pp/gg["cost"], C=gg["cost"]))
R["congestion_sweep"] = loads
print("C3 done", round(time.time()-T0,1))

# ======================================================================
# C4: random 5G SINR-field ensemble (100 layouts, random hotspot)
# ======================================================================
ens = dict(PoT=[], freed=[], rho=[], g_alloc=[], tpw=[])
cf = minimal_feasible_counts(rad, field.N)
for s in range(100):
    rng = np.random.default_rng(3000 + s)
    # random hotspot position in the 23.8 band
    c0 = rng.integers(2, 12); rows = range(5, 10)
    cols = range(int(c0), int(min(20, c0 + rng.integers(6, 12))))
    fl = comms_cost_field(meta, seed=3000 + s,
                          hotspot_rows=rows, hotspot_cols=cols,
                          hotspot_bump=rng.uniform(8, 20),
                          hotspot_load=rng.uniform(2, 8))
    if not verify_A1(rad, fl):
        continue
    gg = greedy_allocate(rad, fl)
    oo = exact_optimum(rad, fl, cf)
    tt = greedy_thresholds(rad, fl, gg); Pp = sum(tt.values())
    qq = pivot_payments(rad, fl, oo, cf); Qp = sum(qq.values())
    ens["PoT"].append(Pp/gg["cost"])
    ens["freed"].append(throughput_retained(fl,meta,gg["counts"])
                        / throughput_total(fl,meta))
    ens["rho"].append(Pp/Qp)
    ens["g_alloc"].append(gg["cost"]/oo["cost"])
    ens["tpw"].append(float(achieved_product_rms(rad, gg["counts"])[0]))
R["ensemble"] = {k: js(np.array(v, float)) for k, v in ens.items()}
print("C4 done", round(time.time()-T0,1))

# ======================================================================
# C5: full AMSR-2 channel set, multi-product
# ======================================================================
chs5 = ("6.9", "10.65", "18.7", "23.8", "36.5")
# sensitivities (representative weighted-regression magnitudes)
C_IWV  = [ 0.00,  0.00, -0.40, 0.90, -0.10]   # mm/K  (water vapour)
C_WIND = [ 0.35,  0.30,  0.10, 0.05,  0.25]   # (m/s)/K (surface wind)
C_SST  = [ 0.55,  0.40,  0.10, 0.05, -0.15]   # K/K  (sea-surface temp)
eps5 = [1.0, 1.2, 0.5]                        # mm^2, (m/s)^2, K^2
rad5, meta5 = build_radiometer(channels=chs5, table=AMSR2_FULL,
                               C_sens=[C_IWV, C_WIND, C_SST], eps2=eps5)
# build a 5-channel cost field: hotspot in the 23.8 band (channel index 3)
field5 = comms_cost_field(meta5, seed=0, hotspot_rows=range(15, 20),
                          hotspot_cols=range(5, 16))
cf5 = minimal_feasible_counts(rad5, field5.N) if rad5.J <= 3 else None
g5 = greedy_allocate(rad5, field5)
tt5 = greedy_thresholds(rad5, field5, g5); P5 = sum(tt5.values())
Rt5 = throughput_total(field5, meta5)
Rf5 = throughput_retained(field5, meta5, g5["counts"])
R["full_amsr2"] = dict(
    channels=list(chs5), counts=js(g5["counts"]),
    total_Gbps=Rt5, retained_Gbps=Rf5, freed_frac=Rf5/Rt5,
    PoT=P5/g5["cost"], C=g5["cost"],
    rms=js(np.round(achieved_product_rms(rad5, g5["counts"]),3)),
    eps=js(np.sqrt(np.array(eps5))),
    products=["IWV (mm)", "wind (m/s)", "SST (K)"],
    NEDT_achieved=js(np.round(achieved_nedt(rad5, meta5, g5["counts"]),3)))
print("C5 done", round(time.time()-T0,1))

# stash the C1 field maps for plotting
R["maps"] = dict(
    sinr=js(field._sinr), se=js(field._se), thr=js(field._thr_grid),
    n_freq=field._n_freq, n_time=N_TIME,
    greedy_counts=js(g["counts"]), order=[js(o) for o in field.order])

with open("comms_results.json", "w") as fh:
    json.dump(R, fh, indent=1, default=js)
print("TOTAL", round(time.time()-T0,1), "s")

# quick textual summary
r = R["realistic"]
print("\n== realistic instance ==")
print("total %.2f Gbps; flex retains %.2f Gbps (%.1f%% freed); "
      "static retains %.2f Gbps (%.1f%%)" % (
      r["total_Gbps"], r["retained_Gbps"], 100*r["freed_frac"],
      r["static_retained_Gbps"], 100*r["static_freed_frac"]))
print("TPW: base %.2f -> flex %.3f mm (tol 1.0); static-band %.3f mm (feasible=%s)" % (
      R["params"]["TPW_base_mm"], r["TPW_achieved_mm"],
      r["static_TPW_mm"], r["static_feasible"]))
print("PoT %.2f; NEDT@23.8 %.3f K" % (r["PoT"], r["NEDT_achieved_K"][1]))
print("ensemble freed median %.3f, PoT median %.3f" % (
      np.median(R["ensemble"]["freed"]), np.median(R["ensemble"]["PoT"])))
