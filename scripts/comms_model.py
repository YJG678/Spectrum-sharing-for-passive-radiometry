"""
Communication-system layer on top of csp.py.

Passive side (AMSR-2 grounded)
------------------------------
Real channels 18.7 / 23.8 / 36.5 GHz with published RF bandwidths and NEDT
specs [Imaoka2010]. The radiometric noise constant is derived from NEDT so the
model is dimensionally a brightness-temperature variance in K^2:
    var_j(B) = kappa_j / (B_j tau) + phi_j,  NEDT_j(B) = sqrt(var_j(B)).
Calibrating at the full RF bandwidth B_full,j over the procurement window tau,
    NEDT_spec,j^2 = kappa_j / (B_full,j tau)  =>  kappa_j = NEDT_spec,j^2 B_full,j tau.
Heavy in-band interference leaves only a residual clean bandwidth
B0_j = rho0 * B_full,j; procuring quiet tiles restores clean bandwidth toward
B_full,j. The retrieved product (integrated water vapour, IWV/TPW, in mm) is a
weighted linear combination of channel brightness temperatures, so its error
variance is the weighted sum of channel variances [Wentz2000,Rodgers2000].

Active side (5G-NR grounded)
----------------------------
Secondary 5G NR users in the adjacent n258 band. Each tile carries a
spectral efficiency SE = min(log2(1+SINR), SE_cap) (256-QAM cap 7.4 bps/Hz,
3GPP TS 38.214 CQI table). A tile's served throughput over the window is
SE * df_tile * dt_tile; its opportunity cost (the secondary's bid) is the
served *revenue*, load_x * SE_x * df_tile * dt_tile * w. Congestion hotspots
(dense urban / transit hubs) have both high SINR and high load and are thus
expensive to silence.
"""
import numpy as np
from csp import Radiometer, CostField

# ---- AMSR-2 channel table (Imaoka et al. 2010; JAXA GCOM-W AMSR2) ----
AMSR2 = {
    "18.7": dict(f=18.7, B_full=200.0,  NEDT=0.7),   # MHz, K
    "23.8": dict(f=23.8, B_full=400.0,  NEDT=0.6),
    "36.5": dict(f=36.5, B_full=1000.0, NEDT=0.7),
}
AMSR2_FULL = {
    "6.9":  dict(f=6.9,  B_full=350.0,  NEDT=0.34),
    "10.65":dict(f=10.65,B_full=100.0,  NEDT=0.7),
    "18.7": dict(f=18.7, B_full=200.0,  NEDT=0.7),
    "23.8": dict(f=23.8, B_full=400.0,  NEDT=0.6),
    "36.5": dict(f=36.5, B_full=1000.0, NEDT=0.7),
}

SE_CAP = 7.4          # bps/Hz, 256-QAM (3GPP TS 38.214 CQI 15)
TAU = 10e-3           # s, procurement / coordination window (a few NR slots)
N_FREQ_SUB = 5        # frequency sub-bands per channel
N_TIME = 20           # time slots per window
RHO0 = 0.05           # residual clean-bandwidth fraction under heavy RFI


def build_radiometer(channels=("18.7", "23.8", "36.5"),
                     C_sens=None, eps2=None, tau=TAU, rho0=RHO0,
                     table=AMSR2, products=None):
    """
    Returns (rad, meta) where rad is a csp.Radiometer in K^2 / mm^2 units and
    meta carries B_full, df_tile (Hz), dt_tile (s), beta (MHz/tile).
    """
    chs = [table[c] for c in channels]
    J = len(chs)
    B_full = np.array([c["B_full"] for c in chs])          # MHz
    NEDT = np.array([c["NEDT"] for c in chs])              # K
    kappa = NEDT ** 2 * B_full                              # K^2 * MHz  (tau folded into B units below)
    B0 = rho0 * B_full                                      # MHz
    beta = (B_full / N_FREQ_SUB)                            # MHz per freq sub-band tile? -> see below
    # Each tile occupies one freq sub-band for one time slot. Procuring a tile
    # for the whole window contributes (B_full/N_FREQ_SUB) * (1/N_TIME) of the
    # per-window-averaged clean bandwidth; summed over all N_FREQ_SUB*N_TIME
    # tiles it restores exactly B_full.
    beta_tile = (B_full / N_FREQ_SUB) / N_TIME             # MHz added to B_j per procured tile
    if C_sens is None:
        # IWV/TPW sensitivity (mm/K): 23.8 dominant, 18.7 negative, 36.5 small
        C_sens = [[-0.40, 0.90, -0.10]]
    if eps2 is None:
        eps2 = [1.0]                                        # mm^2  (1 mm RMS TPW)
    rad = Radiometer(B0=B0, beta=[beta_tile[j] for j in range(J)],
                     kappa=kappa, tau=1.0, phi=[0.0] * J,
                     C_sens=C_sens, eps2=eps2)
    df_tile = (B_full / N_FREQ_SUB) * 1e6                   # Hz
    dt_tile = tau / N_TIME                                  # s
    meta = dict(B_full=B_full, NEDT=NEDT, kappa=kappa, B0=B0,
                beta_tile=beta_tile, df_tile=df_tile, dt_tile=dt_tile,
                channels=list(channels), tau=tau, J=J)
    return rad, meta


# ---------------------------------------------------------------
# 5G-NR SINR / spectral-efficiency cost field
# ---------------------------------------------------------------
def sinr_field(rng, n_freq, n_time, base_db=10.0, base_sd=6.0, smooth=1.2,
               hotspot=None, freq_tilt_db=2.0):
    """SINR (dB) over the grid; optional hotspot bump. freq_tilt: higher
    sub-bands lose a little SINR (more path loss)."""
    from scipy.ndimage import gaussian_filter
    s = rng.normal(base_db, base_sd, (n_freq, n_time))
    s = gaussian_filter(s, sigma=smooth, mode="reflect")
    tilt = np.linspace(0, -freq_tilt_db, n_freq)[:, None]
    s = s + tilt
    if hotspot is not None:
        rows, cols, bump = hotspot
        for i in rows:
            for t in cols:
                s[i, t] += bump
    return s


def spectral_efficiency(sinr_db):
    se = np.log2(1.0 + 10 ** (sinr_db / 10.0))
    return np.clip(se, 0.1, SE_CAP)


def comms_cost_field(meta, seed=0, hotspot_bump=18.0, hotspot_load=3.0,
                     hotspot_rows=range(5, 10), hotspot_cols=range(5, 16),
                     w=1.0, load_scale=1.0):
    """
    Build a CostField whose per-tile bids are foregone served revenue, plus a
    parallel per-tile throughput array (bits over the window) for KPI reporting.
    Channel j occupies grid rows [5j, 5j+5); the contested 23.8 GHz band is the
    middle channel (rows 5..9), where we place the congestion hotspot.
    """
    J = meta["J"]
    n_freq = J * N_FREQ_SUB
    rng = np.random.default_rng(seed)
    hs = (list(hotspot_rows), list(hotspot_cols), hotspot_bump)
    sinr = sinr_field(rng, n_freq, N_TIME, hotspot=hs)
    se = spectral_efficiency(sinr)
    # load map
    load = np.ones((n_freq, N_TIME)) * load_scale
    for i in hotspot_rows:
        for t in hotspot_cols:
            load[i, t] = hotspot_load * load_scale
    # per-tile throughput (bits over window) and bid (revenue)
    thr = np.zeros((n_freq, N_TIME))
    bid = np.zeros((n_freq, N_TIME))
    for j in range(J):
        df = meta["df_tile"][j]; dt = meta["dt_tile"]
        rows = slice(j * N_FREQ_SUB, (j + 1) * N_FREQ_SUB)
        thr[rows, :] = se[rows, :] * df * dt
        bid[rows, :] = w * load[rows, :] * thr[rows, :]
    # channel-wise lists (row-major within channel block)
    bids = [bid[j * N_FREQ_SUB:(j + 1) * N_FREQ_SUB, :].ravel() for j in range(J)]
    field = CostField(bids)
    field._thr = [thr[j * N_FREQ_SUB:(j + 1) * N_FREQ_SUB, :].ravel() for j in range(J)]
    field._se = se
    field._sinr = sinr
    field._load = load
    field._thr_grid = thr
    field._n_freq = n_freq
    return field


# ---------------------------------------------------------------
# Communications KPIs
# ---------------------------------------------------------------
def throughput_total(field, meta):
    """Aggregate secondary throughput if ALL tiles stay active (Gbps)."""
    bits = sum(t.sum() for t in field._thr)
    return bits / meta["tau"] / 1e9


def throughput_retained(field, meta, quiet_counts):
    """
    Secondary throughput retained (Gbps) when the cheapest `quiet_counts[j]`
    tiles of each channel are silenced. The mechanism silences the cheapest
    tiles per channel (Prop. within-channel exactness), i.e. the lowest-revenue
    tiles; throughput uses the throughput (not revenue) of the *remaining*
    tiles. Returns (Gbps_retained, per-channel quiet bits).
    """
    total = 0.0
    for j, nq in enumerate(quiet_counts):
        thr = field._thr[j]
        # quiet tiles are the cheapest-by-bid; align throughput to that order
        order = field.order[j]
        thr_sorted = thr[order]
        retained = thr_sorted[nq:].sum()       # keep the non-silenced tiles
        total += retained
    return total / meta["tau"] / 1e9


def freed_fraction(field, meta, quiet_counts):
    """Fraction of total secondary capacity that remains active."""
    R = throughput_retained(field, meta, quiet_counts)
    return R / throughput_total(field, meta)


def achieved_nedt(rad, meta, counts):
    """Per-channel achieved NEDT (K) at the given counts."""
    return np.sqrt(rad.sigma2(np.asarray(counts)))


def achieved_product_rms(rad, counts):
    """Achieved retrieval RMS per product (sqrt of variance)."""
    return np.sqrt(rad.var_k(np.asarray(counts)))


# ======================================================================
# Deployment-growth layer (Gompertz diffusion + aggregate-leakage
# contamination), following Golparvar et al., IEEE DySPAN 2024.
# ======================================================================
K_BOLTZ = 1.380649e-23        # J/K
B_AMSU = 270e6                # Hz, AMSU-A channel-1 sensing band (23.665-23.935)
POP_METRO = 5.0e6            # aggregated metro population per footprint
DEMAND_PER_USER = 100e6      # bit/s, min per-user 5G mmWave demand [Golparvar'24]
B_5G = 500e6                 # Hz, n258 channel bandwidth

# Gompertz parameters fit so n258 adoption starts ~2021 and saturates ~2040
GOMP = dict(b1=32.0, b2=4.16, b3=0.219, t0=2021)


def gompertz_adoption(year):
    """Subscribers per 100 people in the n258 band (Gompertz S-curve)."""
    t = year - GOMP["t0"]
    return GOMP["b1"] * np.exp(-GOMP["b2"] * np.exp(-GOMP["b3"] * t))


def base_station_count(year, eta_sp, pop=POP_METRO):
    """N_BS in a footprint: N = D_t/(eta_sp*B), D_t = subscribers*demand."""
    subs = gompertz_adoption(year) / 100.0 * pop
    Dt = subs * DEMAND_PER_USER
    return Dt / (eta_sp * B_5G)


def dT_per_bs(Ps_rfi_dBW=-175.0, B=B_AMSU):
    """Induced brightness-temperature contamination per base station (K),
    dT = P_RFI/(k B), with P_RFI the per-BS received leaked power."""
    P_w = 10 ** (Ps_rfi_dBW / 10.0)
    return P_w / (K_BOLTZ * B)


class ContamRadiometer:
    """
    Contamination-cover incumbent (single 23.8 GHz band, AMSU-A ch-1).
    Aggregate contamination with quiet set S:
        dT(S) = a * (M - q(S)),  q(S) = sum of BS silenced by S,  a = dT/BS.
    Feasibility: dT(S) <= dT_tol  <=>  q(S) >= q_req = M - dT_tol/a.
    The coverage G(S)=min(q(S), q_req) is monotone submodular (a capped
    modular function), so the mechanism of Sec. VII applies verbatim.
    Interface mirrors csp.Radiometer so greedy/threshold/exact/pivot reuse it.
    """
    def __init__(self, M, m_per_tile, a, dT_tol, J=1):
        self.M = float(M); self.a = float(a); self.dT_tol = float(dT_tol)
        self.q_req = max(0.0, M - dT_tol / a)
        self.m = np.atleast_1d(np.asarray(m_per_tile, float))  # BS per tile, per channel
        self.J = J
        self.Gtot = float(self.q_req)
        self.var_base = np.array([self.q_req])   # placeholder for interface
        self.Gamma = np.array([self.q_req])
        self.eps2 = np.array([0.0])
        self.K = 1

    def _q(self, n):
        return float(np.dot(np.asarray(n, float), self.m if len(self.m)==self.J
                            else np.full(self.J, self.m[0])))

    def G(self, n):
        return min(self._q(n), self.q_req)

    def gain(self, n, j):
        q = self._q(n)
        mj = self.m[j] if len(self.m) == self.J else self.m[0]
        return min(self.q_req, q + mj) - min(self.q_req, q)

    def covered(self, n):
        return self._q(n) >= self.q_req - 1e-9

    def feasible_counts(self, n):
        return self.covered(n)

    def var_k(self, n):
        # residual contamination "variance" proxy (K): dT(S)
        return np.array([self.a * max(0.0, self.M - self._q(n))])

    def sigma2(self, n):
        return self.var_k(n)

    def dT(self, n):
        return self.a * max(0.0, self.M - self._q(n))


def build_contam_instance(year, eta_sp, Ps_rfi_dBW=-175.0, dT_tol=1.0,
                          n_tiles=300, J=1, pop=POP_METRO):
    """Return (rad, meta) for the deployment-year contamination instance."""
    M = base_station_count(year, eta_sp, pop)
    a = dT_per_bs(Ps_rfi_dBW)
    m_per_tile = M / n_tiles                       # equal BS per tile
    rad = ContamRadiometer(M=M, m_per_tile=[m_per_tile]*J, a=a,
                           dT_tol=dT_tol, J=J)
    meta = dict(M=M, a=a, dT_unprot=a*M, m_per_tile=m_per_tile,
                q_req=rad.q_req, year=year, eta=eta_sp, n_tiles=n_tiles,
                Ps_rfi_dBW=Ps_rfi_dBW, dT_tol=dT_tol, J=J)
    return rad, meta


def contam_cost_field(meta, seed=0, hotspot_frac=0.5):
    """
    5G opportunity-cost field for the contamination instance: per-tile bid =
    served revenue = load*SE*df*dt scaled by per-tile BS count; also stores
    per-tile served throughput (bits) for capacity KPIs. Single band (J tiles
    per row block). Total secondary throughput grows with deployment (more BS).
    """
    J = meta["J"]; n_tiles = meta["n_tiles"]
    per_ch = n_tiles // J
    n_freq = J * N_FREQ_SUB
    n_time = per_ch // N_FREQ_SUB
    rng = np.random.default_rng(seed)
    from scipy.ndimage import gaussian_filter
    sinr = gaussian_filter(rng.normal(10, 6, (n_freq, n_time)), 1.2, mode="reflect")
    # congestion hotspot
    hw = int(hotspot_frac * n_time)
    sinr[:, :hw] += 16.0
    se = np.clip(np.log2(1 + 10 ** (sinr / 10)), 0.1, SE_CAP)
    load = np.ones((n_freq, n_time)); load[:, :hw] = 4.0
    df = B_5G / N_FREQ_SUB; dt = 1e-3
    m = meta["m_per_tile"]
    thr = se * df * dt * m                        # bits/tile (BS reuse)
    bid = load * thr
    bids = [bid[j*N_FREQ_SUB:(j+1)*N_FREQ_SUB, :].ravel() for j in range(J)]
    field = CostField(bids)
    field._thr = [thr[j*N_FREQ_SUB:(j+1)*N_FREQ_SUB, :].ravel() for j in range(J)]
    field._se = se
    return field


def contam_throughput_total(field):
    return sum(t.sum() for t in field._thr) / 1e9    # Gb (per ms window ->Gb/s*1e-3? keep relative)


def contam_retained(field, quiet_counts):
    tot = 0.0
    for j, nq in enumerate(quiet_counts):
        order = field.order[j]
        tot += field._thr[j][order][nq:].sum()
    return tot / 1e9


# ======================================================================
# LCR HDD dense-deployment calibration (Maheshwari et al., Sci. Data 2025).
# Two validated venues supply a realistic, dense secondary opportunity-cost
# field: Salt & Tar (outdoor, 3000 users, 9 RUs) and ACC Arena (indoor,
# 12000 users, 33 RUs). We transfer the venues' DEMAND statistics (user/RU
# density, SINR distribution, traffic-type mix, throughput/PRB scale) to
# parametrize the contested secondary; the incumbent-protection physics
# remains the 23.8 GHz radiometer model.
# ======================================================================
LCR = {
    "salt_tar": dict(users=3000, ru=9,  indoor=False, sinr_mu=4.0, sinr_sd=8.0,
                     hotspot_frac=0.45, hotspot_gain=1.8, sample_s=1),
    "acc_arena": dict(users=12000, ru=33, indoor=True, sinr_mu=1.0, sinr_sd=9.0,
                      hotspot_frac=0.60, hotspot_gain=2.3, sample_s=3),
}
# LCR HDD radio configuration (Table 2)
LCR_CFG = dict(freq_GHz=3.8, B_MHz=100.0, mimo=4, eirp_dBm=49.0, ant_dBi=12.0,
               scs_kHz=30.0, n_prb=273, qam=256, se_cap_layer=7.4)
# traffic-type mix and per-type load weight
# states {0 off,1 idle,2 constant,3 video,4 gaming,5 http}
LCR_TRAFFIC_P = np.array([0.15, 0.25, 0.10, 0.25, 0.10, 0.15])
LCR_TRAFFIC_W = np.array([0.00, 0.05, 0.40, 1.00, 0.80, 0.30])
# validated reference points (Technical Validation section)
LCR_VALID = dict(salt_tar_thr_Mbps=244.36, salt_tar_thr_meas=235.7,
                 acc_prb=130, acc_prb_meas=126, prb_total=273)


def lcr_mimo_se(sinr_db, cfg=LCR_CFG):
    """Rank-adaptive 4x4-MIMO spectral efficiency (bps/Hz), 256-QAM capped.
    Number of usable layers grows with SINR; each layer capped at 7.4 bps/Hz."""
    se1 = np.clip(np.log2(1 + 10 ** (sinr_db / 10.0)), 0.1, cfg["se_cap_layer"])
    # heuristic rank: 1 layer below 3 dB, +1 layer per ~6 dB up to 4
    rank = np.clip(1 + np.floor((sinr_db - 3.0) / 6.0), 1, cfg["mimo"])
    return se1 * rank


def lcr_venue_field(venue="salt_tar", seed=0, grid=(15, 20)):
    """
    Build a CostField whose per-tile bids are the aggregate secondary revenue
    (foregone served throughput) contending for that time-frequency resource
    in the given LCR HDD venue. Denser venues (more users/RU, lower SINR)
    yield higher, more concentrated bids -> scarcer substitutes.
    Returns (field, meta).
    """
    v = LCR[venue]; cfg = LCR_CFG
    n_freq, n_time = grid
    rng = np.random.default_rng(seed)
    users_per_ru = v["users"] / v["ru"]
    # per-tile contention multiplier ~ users/RU normalized (density of demand)
    dens = users_per_ru / 333.0                       # ~1 for Salt & Tar
    # SINR field: spatially correlated, venue-calibrated
    from scipy.ndimage import gaussian_filter
    sinr = gaussian_filter(rng.normal(v["sinr_mu"], v["sinr_sd"],
                                      (n_freq, n_time)), 1.1, mode="reflect")
    sinr = np.clip(sinr, -20, 25)
    se = lcr_mimo_se(sinr, cfg)
    # traffic type per tile -> load weight (aggregate over contending users)
    ttype = rng.choice(6, size=(n_freq, n_time), p=LCR_TRAFFIC_P)
    load = LCR_TRAFFIC_W[ttype]
    # congestion hotspot (stage / main scenario)
    hw = int(v["hotspot_frac"] * n_time)
    load[:, :hw] *= v["hotspot_gain"]
    # resource size per tile: 100 MHz split over 5 sub-bands, 30 kHz-SCS slot
    df = cfg["B_MHz"] * 1e6 / N_FREQ_SUB               # Hz
    dt = 0.5e-3                                        # s (30 kHz SCS slot)
    thr = se * df * dt * dens                          # bits/tile (served if active)
    bid = load * thr                                   # revenue proxy
    # split into channel blocks (reuse incumbent grid J=3)
    J = n_freq // N_FREQ_SUB
    bids = [bid[j*N_FREQ_SUB:(j+1)*N_FREQ_SUB, :].ravel() for j in range(J)]
    field = CostField(bids)
    field._thr = [thr[j*N_FREQ_SUB:(j+1)*N_FREQ_SUB, :].ravel() for j in range(J)]
    field._se = se; field._sinr = sinr; field._load = load
    meta = dict(venue=venue, users=v["users"], ru=v["ru"],
                users_per_ru=users_per_ru, indoor=v["indoor"],
                mean_sinr=float(sinr.mean()), mean_se=float(se.mean()),
                df=df, dt=dt, dens=dens,
                total_Gbps=float(sum(t.sum() for t in field._thr))/dt/1e9)
    return field, meta


def lcr_prb_utilization(field, quiet_counts):
    """Fraction of resources still active (PRB-utilization proxy) after
    silencing the cheapest quiet_counts tiles per channel."""
    kept = 0; tot = 0
    for j, nq in enumerate(quiet_counts):
        n = len(field.sorted[j]); tot += n; kept += (n - nq)
    return kept / tot
