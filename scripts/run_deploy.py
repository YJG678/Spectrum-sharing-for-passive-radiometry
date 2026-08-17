"""Deployment-growth experiment (Gompertz + aggregate-leakage contamination).
Runs the CSP mechanism year-by-year to hold AMSU-A ch-1 contamination within a
tolerance, reporting procurement cost, freed 5G capacity, price of truthfulness,
and achieved dT.  Built on the methods of Golparvar et al., DySPAN 2024."""
import sys; sys.path.insert(0, ".")
import numpy as np, json, time
from csp import (greedy_allocate, greedy_thresholds, exact_optimum,
                 pivot_payments, minimal_feasible_counts, verify_A1)
from comms_model import *

R = {}; T0 = time.time()
def js(x):
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, np.floating): return float(x)
    if isinstance(x, np.ndarray): return x.tolist()
    return x

YEARS = list(range(2025, 2041))
ETAS = [7, 15, 25]
Ps = -175.0; dT_tol = 1.0

# ---- D0: adoption + unprotected contamination curves (reproduce DySPAN) ----
R["adoption"] = dict(
    years=YEARS,
    adopt=[gompertz_adoption(y) for y in YEARS],
    dT_unprot={str(e): [dT_per_bs(Ps) * base_station_count(y, e) for y in YEARS]
               for e in ETAS},
    N_BS={str(e): [base_station_count(y, e) for y in YEARS] for e in ETAS},
    a_K_per_BS=dT_per_bs(Ps), Ps_dBW=Ps, dT_tol=dT_tol)

# ---- D1: procurement trajectory at eta=15 (central case) ----
def run_year(year, eta, seed=0):
    rad, meta = build_contam_instance(year, eta, Ps_rfi_dBW=Ps, dT_tol=dT_tol)
    field = contam_cost_field(meta, seed=seed)
    tot = contam_throughput_total(field)
    if rad.q_req <= 0:                     # already within tolerance
        return dict(year=year, eta=eta, procure=False, dT_unprot=meta["dT_unprot"],
                    dT_ach=meta["dT_unprot"], quiet_frac=0.0, cost=0.0, P=0.0,
                    PoT=1.0, retained=tot, total=tot, freed_frac=1.0,
                    counts=[0]*meta["J"])
    g = greedy_allocate(rad, field)
    th = greedy_thresholds(rad, field, g); P = sum(th.values())
    ret = contam_retained(field, g["counts"])
    return dict(year=year, eta=eta, procure=True, dT_unprot=meta["dT_unprot"],
                dT_ach=rad.dT(g["counts"]),
                quiet_frac=sum(g["counts"])/meta["n_tiles"],
                cost=g["cost"], P=P, PoT=P/g["cost"] if g["cost"]>0 else 1.0,
                retained=ret, total=tot, freed_frac=ret/tot,
                counts=js(g["counts"]),
                q_req=meta["q_req"], M=meta["M"], A1=verify_A1(rad, field))

traj = {str(e): [run_year(y, e) for y in YEARS] for e in ETAS}
R["trajectory"] = traj
print("D1 done", round(time.time()-T0,1))

# ---- D2: permissible single-BS interference sweep (like their Fig. 8) ----
# For each (year, eta), find the largest Ps (dBW) that needs no procurement
# to keep dT<=dT_tol (i.e. dT_unprot<=dT_tol): the permissible RFI level.
perm = {str(e): [] for e in ETAS}
for e in ETAS:
    for y in YEARS:
        N = base_station_count(y, e)
        # dT_unprot = (P/(kB))*N <= dT_tol  ->  P <= dT_tol*kB/N
        P_lim = dT_tol * K_BOLTZ * B_AMSU / N          # W
        perm[str(e)].append(10*np.log10(P_lim))        # dBW
R["permissible_dBW"] = perm
print("D2 done", round(time.time()-T0,1))

# ---- D3: contamination-vs-capacity frontier at 2040, eta=15 (sweep dT_tol) --
front = []
for tol in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
    rad, meta = build_contam_instance(2040, 15, Ps_rfi_dBW=Ps, dT_tol=tol)
    field = contam_cost_field(meta, seed=0)
    tot = contam_throughput_total(field)
    if rad.q_req <= 0:
        front.append(dict(tol=tol, quiet_frac=0, cost=0, PoT=1.0,
                          freed=1.0, retained=tot)); continue
    g = greedy_allocate(rad, field)
    th = greedy_thresholds(rad, field, g); P = sum(th.values())
    ret = contam_retained(field, g["counts"])
    front.append(dict(tol=tol, quiet_frac=sum(g["counts"])/meta["n_tiles"],
                      cost=g["cost"], PoT=P/g["cost"], freed=ret/tot,
                      retained=ret, dT_ach=rad.dT(g["counts"])))
R["frontier_2040"] = front
print("D3 done", round(time.time()-T0,1))

# ---- D4: verify contamination reduction F(S) is monotone submodular --------
# F(S) = a^2 (2 M q(S) - q(S)^2); numerically check diminishing returns.
rad, meta = build_contam_instance(2040, 15, dT_tol=1.0)
a, M, m = meta["a"], meta["M"], meta["m_per_tile"]
def Fq(q): return a*a*(2*M*q - q*q)
qs = np.linspace(0, meta["q_req"], 50)
marg = np.diff(Fq(qs))
R["submod_check"] = dict(monotone=bool(np.all(marg > -1e-9)),
                         diminishing=bool(np.all(np.diff(marg) < 1e-9)),
                         F0=float(Fq(0)), Fqreq=float(Fq(meta["q_req"])))
print("D4 done", round(time.time()-T0,1), R["submod_check"])

with open("deploy_results.json", "w") as fh:
    json.dump(R, fh, indent=1, default=js)

# summary
print("\n== eta=15 trajectory ==")
for d in traj["15"]:
    print(f' {d["year"]}: dT_unprot={d["dT_unprot"]:5.1f}K -> dT={d["dT_ach"]:.2f}K '
          f'quiet={d["quiet_frac"]*100:4.0f}% cost={d["cost"]:7.1f} PoT={d["PoT"]:.2f} '
          f'freed={d["freed_frac"]*100:4.0f}%')
print("\n== permissible single-BS RFI (dBW) to need no procurement ==")
for e in ETAS:
    print(f' eta={e}: 2025 {perm[str(e)][0]:.0f}  2030 {perm[str(e)][5]:.0f}  '
          f'2040 {perm[str(e)][-1]:.0f}')
print("TOTAL", round(time.time()-T0,1),"s")
