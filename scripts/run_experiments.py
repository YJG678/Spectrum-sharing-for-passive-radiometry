"""Run every experiment for the journal paper; dump results to JSON."""
import numpy as np, json, time, itertools
from csp import *

R = {}
T0 = time.time()


def js(x):
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, np.ndarray): return x.tolist()
    return x


# ======================================================================
# E1: trap instance
# ======================================================================
rad = amsr_instance()
fimg = trap_field(seed=0)
field = CostField(field_to_channels(fimg))
counts_feas = minimal_feasible_counts(rad, field.N)

g = greedy_allocate(rad, field)
opt = exact_optimum(rad, field, counts_feas)
th = greedy_thresholds(rad, field, g)
pp = pivot_payments(rad, field, opt, counts_feas)
P_g, P_p = sum(th.values()), sum(pp.values())

# fixed-band baseline: minimum n1 with channel-1 only
n1_min = None
for n1 in range(field.N[0] + 1):
    if rad.feasible_counts(np.array([n1, 0, 0])):
        n1_min = n1; break
pref1 = field.prefix_cost(0)
C_static = float(pref1[n1_min])
n_trap_static = max(0, n1_min - 45)            # 45 non-trap ch-1 tiles

tot = field.total()
R["trap"] = dict(
    var_base=js(rad.var_base[0]), Gtot=rad.Gtot,
    greedy_counts=js(g["counts"]), greedy_cost=g["cost"],
    exact_counts=js(opt["counts"]), exact_cost=opt["cost"],
    g_alloc=g["cost"] / opt["cost"],
    P_greedy=P_g, P_pivot=P_p,
    PoT_greedy=P_g / g["cost"], PoT_pivot=P_p / opt["cost"],
    rho=P_g / P_p,
    n1_min_static=n1_min, C_static=C_static, n_trap_static=n_trap_static,
    save=1 - g["cost"] / C_static,
    field_total=tot,
    U_tile_static=1 - n1_min / sum(field.N),
    U_tile_flex=1 - len(g["picked"]) / sum(field.N),
    U_value_static=1 - C_static / tot,
    U_value_flex=1 - g["cost"] / tot,
    q_static=[n1_min / field.N[0], 0, 0],
    q_flex=[int(c) / N for c, N in zip(g["counts"], field.N)],
    A1=verify_A1(rad, field),
    theta_by_ch={f"ch{j+1}": [th[(jj, r)] for (jj, r, b) in g["picked"] if jj == j]
                 for j in range(3)},
    cost_by_ch={f"ch{j+1}": [b for (jj, r, b) in g["picked"] if jj == j]
                for j in range(3)},
    pivot_theta={f"ch{j+1}": [pp[(j, r)] for r in range(opt["counts"][j])]
                 for j in range(3)},
    pivot_cost={f"ch{j+1}": [float(field.sorted[j][r]) for r in range(opt["counts"][j])]
                for j in range(3)},
)
print("E1 trap done", time.time() - T0)

# ======================================================================
# E2: random-field ensemble (200 instances)
# ======================================================================
ens = dict(PoT=[], Phi=[], rho=[], g_alloc=[], A1=[])
for s in range(200):
    rng = np.random.default_rng(1000 + s)
    f = smoothed_uniform_field(rng, 15, 20)
    fl = CostField(field_to_channels(f))
    gg = greedy_allocate(rad, fl)
    oo = exact_optimum(rad, fl, counts_feas)
    tt = greedy_thresholds(rad, fl, gg)
    qq = pivot_payments(rad, fl, oo, counts_feas)
    P = sum(tt.values()); Q = sum(qq.values())
    ens["PoT"].append(P / gg["cost"])
    ens["Phi"].append(P / oo["cost"])
    ens["rho"].append(P / Q)
    ens["g_alloc"].append(gg["cost"] / oo["cost"])
    ens["A1"].append(verify_A1(rad, fl))
R["ensemble"] = {k: js(np.array(v, float)) if k != "A1" else all(v)
                 for k, v in ens.items()}
print("E2 ensemble done", time.time() - T0)

# ======================================================================
# E3a: trap-value sweep (scarcity at constant curvature)
# ======================================================================
c_full = total_curvature(rad.B0, rad.beta, field.N)
sweep_a = dict(trap=[], PoT=[], cost=[], curvature=c_full)
for h in [3, 5, 10, 20, 35, 50, 80]:
    f = trap_field(seed=0, trap_value=h)
    fl = CostField(field_to_channels(f))
    gg = greedy_allocate(rad, fl)
    tt = greedy_thresholds(rad, fl, gg)
    sweep_a["trap"].append(h)
    sweep_a["PoT"].append(sum(tt.values()) / gg["cost"])
    sweep_a["cost"].append(gg["cost"])
R["sweep_scarcity"] = sweep_a
print("E3a done", time.time() - T0)

# ======================================================================
# E3b + E4b: baseline-scale sweep (curvature swing at flat PoT) and
#            cover gap vs operating curvature
# ======================================================================
sweep_b = dict(scale=[], c_full=[], c_op=[], PoT_med=[], PoT_q1=[], PoT_q3=[],
               gap=[], inv1mc=[])
for s_ in [1, 3, 10, 30, 100, 500]:
    B0 = (10.0 * s_, 250.0 * s_, 100.0 * s_)
    rs = Radiometer(B0=B0, beta=[10] * 3, kappa=[500] * 3, tau=1, phi=[0] * 3,
                    C_sens=[[0.45, -0.20, 0.05]], eps2=[0.0])
    # choose eps2 so the required reduction is 60% of the achievable one
    Fmax = float(rs.var_base[0] - rs.var_k(np.array([100] * 3))[0])
    eps2 = float(rs.var_base[0] - 0.6 * Fmax)
    rs = Radiometer(B0=B0, beta=[10] * 3, kappa=[500] * 3, tau=1, phi=[0] * 3,
                    C_sens=[[0.45, -0.20, 0.05]], eps2=[eps2])
    cf = minimal_feasible_counts(rs, [100] * 3)
    pots, gaps, cops = [], [], []
    for sd in range(20):
        rng = np.random.default_rng(5000 + sd)
        fl = CostField(field_to_channels(smoothed_uniform_field(rng, 15, 20)))
        gg = greedy_allocate(rs, fl)
        oo = exact_optimum(rs, fl, cf)
        tt = greedy_thresholds(rs, fl, gg)
        pots.append(sum(tt.values()) / gg["cost"])
        gaps.append(gg["cost"] / oo["cost"])
        cops.append(operating_curvature(B0, [10] * 3, gg["counts"]))
    q1, med, q3 = np.percentile(pots, [25, 50, 75])
    sweep_b["scale"].append(s_)
    sweep_b["c_full"].append(total_curvature(B0, [10] * 3, [100] * 3))
    sweep_b["c_op"].append(float(np.median(cops)))
    sweep_b["PoT_med"].append(med); sweep_b["PoT_q1"].append(q1)
    sweep_b["PoT_q3"].append(q3)
    sweep_b["gap"].append(float(np.median(gaps)))
    sweep_b["inv1mc"].append(1.0 / (1.0 - float(np.median(cops))))
R["sweep_curvature"] = {k: js(np.array(v)) if isinstance(v, list) else v
                        for k, v in sweep_b.items()}
print("E3b/E4b done", time.time() - T0)

# ======================================================================
# E4a: closed-form curvature vs dynamic range (analytic)
# ======================================================================
DR = np.logspace(0.05, 3, 60)            # Bmax/B0
beta_fine = 1e-3
c_exact, c_lim = [], []
for d in DR:
    B0 = 1.0; Bmax = d
    N = (Bmax - B0) / beta_fine
    c_exact.append(1 - (B0 * (B0 + beta_fine)) / ((Bmax - beta_fine) * Bmax))
    c_lim.append(1 - (B0 / Bmax) ** 2)
R["curv_formula"] = dict(DR=js(DR), c_exact=js(np.array(c_exact)),
                         c_lim=js(np.array(c_lim)),
                         amsr_DR=float((10 + 100 * 10) / 10),
                         amsr_c=total_curvature([10, 250, 100], [10] * 3, [100] * 3))

# ======================================================================
# E5: multi-product scenarios (trap field)
# ======================================================================
C_IWV = (0.45, -0.20, 0.05)
C_W = (0.10, 0.35, 0.25)
C_S = (0.25, 0.15, -0.30)
scen = [
    ("IWV only", [C_IWV], [0.25]),
    ("IWV + wind", [C_IWV, C_W], [0.25, 0.30]),
    ("IWV + wind + SST", [C_IWV, C_W, C_S], [0.25, 0.30, 0.50]),
    ("Tight SST tolerance", [C_IWV, C_W, C_S], [0.25, 0.30, 0.30]),
    ("Tight IWV tolerance", [C_IWV, C_W, C_S], [0.15, 0.30, 0.50]),
]
mp = []
for name, Cs, e2 in scen:
    rs = Radiometer(B0=[10, 250, 100], beta=[10] * 3, kappa=[500] * 3, tau=1,
                    phi=[0] * 3, C_sens=Cs, eps2=e2)
    gg = greedy_allocate(rs, field)
    tt = greedy_thresholds(rs, field, gg)
    cf = minimal_feasible_counts(rs, field.N)
    oo = exact_optimum(rs, field, cf)
    P = sum(tt.values())
    ratio = float(np.max(rs.var_k(gg["counts"]) / rs.eps2))
    mp.append(dict(name=name, C=gg["cost"], P=P, PoT=P / gg["cost"],
                   maxvar=ratio, counts=js(gg["counts"]),
                   exact_cost=oo["cost"], g_alloc=gg["cost"] / oo["cost"]))
R["multiproduct"] = mp
print("E5 done", time.time() - T0)

# ======================================================================
# E6: tolerance sweep (quality-cost frontier, trap field)
# ======================================================================
front = []
for e2 in [0.15, 0.18, 0.20, 0.25, 0.30, 0.40, 0.60, 0.80]:
    rs = amsr_instance(eps2=e2)
    gg = greedy_allocate(rs, field)
    tt = greedy_thresholds(rs, field, gg)
    cf = minimal_feasible_counts(rs, field.N)
    oo = exact_optimum(rs, field, cf)
    P = sum(tt.values())
    front.append(dict(eps2=e2, C=gg["cost"], P=P, PoT=P / gg["cost"],
                      nsel=len(gg["picked"]),
                      U_tile=1 - len(gg["picked"]) / 300,
                      counts=js(gg["counts"]), exact_cost=oo["cost"]))
R["frontier"] = front
print("E6 done", time.time() - T0)

# ======================================================================
# E8 (new): reserve-price sweep on the trap field
# ======================================================================
res = []
maxbg = float(max(np.max(field.sorted[j][field.sorted[j] < 49]) for j in range(3)))
for r in [2.0, 2.3, 2.6, 3.0, 5.0, 10.0, 20.0, 35.0, 50.0, np.inf]:
    excl = {(j, rk) for j in range(3) for rk in range(field.N[j])
            if field.sorted[j][rk] > r}
    gg = greedy_allocate(rad, field, exclude=excl)
    if gg is None:
        res.append(dict(r=js(r), feasible=False)); continue
    # thresholds in the restricted market, capped at r
    tt = {}
    for (j, rk, b) in gg["picked"]:
        run = greedy_allocate(rad, field, exclude=excl | {(j, rk)}, record=True)
        if run is None:
            tt[(j, rk)] = r          # pivotal within restricted market
            continue
        best = 0.0
        for (n_before, jj, rr2, gn, bb) in run["trace"]:
            gx = rad.gain(n_before, j)
            if gx > EPS:
                best = max(best, gx * bb / gn)
        tt[(j, rk)] = min(best, r)
    P = sum(tt.values())
    res.append(dict(r=js(r), feasible=True, C=gg["cost"], P=P,
                    PoT=P / gg["cost"], counts=js(gg["counts"])))
R["reserve"] = res
R["reserve_maxbg"] = maxbg
print("E8 done", time.time() - T0)

# ======================================================================
# E9 (new): market thickness — PoT vs tiles per channel
# ======================================================================
thick = dict(N=[], med=[], q1=[], q3=[], rho_med=[])
for Nt in [14, 17, 20, 26, 34, 44, 60]:        # time slots; N_j = 5*Nt
    Nj = 5 * Nt
    pots, rhos = [], []
    cf = minimal_feasible_counts(rad, [Nj] * 3)
    for sd in range(30):
        rng = np.random.default_rng(9000 + 100 * Nt + sd)
        fl = CostField(field_to_channels(smoothed_uniform_field(rng, 15, Nt)))
        gg = greedy_allocate(rad, fl)
        tt = greedy_thresholds(rad, fl, gg)
        oo = exact_optimum(rad, fl, cf)
        qq = pivot_payments(rad, fl, oo, cf)
        P = sum(tt.values())
        pots.append(P / gg["cost"]); rhos.append(P / sum(qq.values()))
    q1, med, q3 = np.percentile(pots, [25, 50, 75])
    thick["N"].append(Nj); thick["med"].append(med)
    thick["q1"].append(q1); thick["q3"].append(q3)
    thick["rho_med"].append(float(np.median(rhos)))
R["thickness"] = {k: js(np.array(v)) for k, v in thick.items()}
print("E9 done", time.time() - T0)

# ======================================================================
# E10 (new): truthfulness verification (utility vs reported bid)
# ======================================================================
# pick the ch-1 winner with the highest threshold and a ch-2 winner
w1 = max([(th[(j, r)], j, r, b) for (j, r, b) in g["picked"] if j == 0])
w2 = max([(th[(j, r)], j, r, b) for (j, r, b) in g["picked"] if j == 1])
ver = []
for (tstar, j, rk, c_true) in [w1, w2]:
    bids = np.linspace(0.05, 1.6 * tstar, 80)
    util = []
    for z in bids:
        run = greedy_allocate(rad, field, modified={(j, rk): z})
        win = any(jj == j and rr == rk for jj, rr, _ in run["picked"])
        util.append(tstar - c_true if win else 0.0)
    ver.append(dict(ch=j + 1, theta=tstar, cost=c_true,
                    bids=js(bids), util=js(np.array(util))))
R["truthful"] = ver
print("E10 done", time.time() - T0)

# ======================================================================
# E7: scalability under heterogeneous tile volumes (J=1)
# ======================================================================
def hetero_instance(rng, n):
    v = rng.uniform(5, 15, n)
    c = rng.uniform(1, 3, n) * (v / 10) ** 0.8
    D = 0.45 * v.sum()
    return v, c, D

def hetero_greedy(v, c, D):
    # G = min(sum v, D); ratio rule = v/c until residual, then min residual fix
    idx = np.argsort(-(v / c))
    tot, cost, sel = 0.0, 0.0, []
    for i in idx:
        if tot >= D: break
        sel.append(i); tot += v[i]; cost += c[i]
    return cost, sel

def hetero_dp(v, c, D, scale=10):
    vi = np.round(v * scale).astype(int)
    Di = int(np.ceil(D * scale))
    INF = 1e18
    dp = np.full(Di + 1, INF); dp[0] = 0.0
    for vv, cc in zip(vi, c):
        ndp = dp.copy()
        src = dp + cc
        # transition: j -> min(j+vv, Di)
        tgt = np.minimum(np.arange(Di + 1) + vv, Di)
        np.minimum.at(ndp, tgt, src)
        dp = ndp
    return float(dp[Di])

def hetero_brute(v, c, D):
    n = len(v); best = np.inf
    for mask in range(1 << n):
        s = [i for i in range(n) if mask >> i & 1]
        if v[s].sum() >= D:
            best = min(best, c[s].sum())
    return best

rngh = np.random.default_rng(7)
scal = dict(n_bf=[], t_bf=[], t_gr_small=[], n_gap=[], gap=[],
            n_big=[], t_big=[])
for n in [10, 12, 14, 16, 18, 20, 22]:
    v, c, D = hetero_instance(rngh, n)
    t0 = time.time(); hetero_brute(v, c, D); tb = time.time() - t0
    t0 = time.time(); hetero_greedy(v, c, D); tg = time.time() - t0
    scal["n_bf"].append(n); scal["t_bf"].append(tb); scal["t_gr_small"].append(tg)
for n in [50, 100, 200, 500, 1000, 2000]:
    gaps = []
    for sd in range(20):
        rng = np.random.default_rng(100 * n + sd)
        v, c, D = hetero_instance(rng, n)
        cg, _ = hetero_greedy(v, c, D)
        ce = hetero_dp(v, c, D)
        gaps.append(cg / ce)
    scal["n_gap"].append(n); scal["gap"].append(float(np.mean(gaps)))
for n in [10**3, 10**4, 10**5, 10**6]:
    rng = np.random.default_rng(n)
    v, c, D = hetero_instance(rng, n)
    t0 = time.time(); hetero_greedy(v, c, D); scal["n_big"].append(n)
    scal["t_big"].append(time.time() - t0)
R["scalability"] = scal
print("E7 done", time.time() - T0)

# ======================================================================
# Hardness-reduction verification (Thms 1-2 vs brute force)
# ======================================================================
def verify_thm1(trials=300):
    rng = np.random.default_rng(42); ok = 0
    for _ in range(trials):
        n = rng.integers(4, 11)
        a = rng.integers(1, 25, n)
        t = int(rng.integers(1, a.sum()))
        # radiometric instance per reduction
        kap, b = 1.0, 1.0
        eps2 = kap / (b + t)
        # min feasible cost by brute force on the radiometric side
        best = np.inf; subset_hits_t = False
        for mask in range(1 << int(n)):
            s = [i for i in range(n) if mask >> i & 1]
            ssum = a[s].sum()
            if kap / (b + ssum) <= eps2 + 1e-12:
                best = min(best, ssum)
            if ssum == t: subset_hits_t = True
        ok += (abs(best - t) < 1e-9) == subset_hits_t
    return ok, trials

def verify_thm2(trials=400):
    rng = np.random.default_rng(43); ok = 0
    for _ in range(trials):
        K = int(rng.integers(2, 7)); M = int(rng.integers(2, 8))
        sets = [set(np.flatnonzero(rng.random(K) < 0.5)) for _ in range(M)]
        # ensure coverage possible
        if not set().union(*sets) >= set(range(K)):
            sets[0] |= set(range(K))
        gam = rng.uniform(1, 5, M)
        # set-cover optimum by brute force
        bestSC = np.inf
        for mask in range(1 << M):
            cov = set().union(*[sets[i] for i in range(M) if mask >> i & 1]) \
                if mask else set()
            if cov >= set(range(K)):
                bestSC = min(bestSC, sum(gam[i] for i in range(M) if mask >> i & 1))
        # radiometric instance per reduction
        kap, b = 1.0, 1.0
        d = np.array([sum(k in sets[i] for i in range(M)) for k in range(K)], float)
        delta = kap / b - kap / (b + 1)
        eps2 = d * kap / b - 0.5 * delta
        bestR = np.inf
        for mask in range(1 << M):
            sel = [i for i in range(M) if mask >> i & 1]
            var = np.array([sum((kap / (b + 1) if i in sel else kap / b)
                                for i in range(M) if k in sets[i])
                            for k in range(K)])
            if np.all(var <= eps2 + 1e-12):
                bestR = min(bestR, sum(gam[i] for i in sel))
        ok += abs(bestSC - bestR) < 1e-9
    return ok, trials

R["verify_thm1"] = verify_thm1()
R["verify_thm2"] = verify_thm2()
print("reduction checks", R["verify_thm1"], R["verify_thm2"], time.time() - T0)

with open("results.json", "w") as fh:
    json.dump(R, fh, indent=1, default=js)
print("TOTAL", time.time() - T0, "s")
