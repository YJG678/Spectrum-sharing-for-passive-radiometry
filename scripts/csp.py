"""
Reference implementation: Cognitive Spectrum Procurement (CSP) for passive
radiometer protection.

Model
-----
J channels, channel j has N_j equal-volume tiles, each contributing
beta_j = v_j / tau of effective bandwidth.  B_j(n) = B0_j + n * beta_j.
Channel variance  sigma_j^2(n) = kappa_j / (B_j(n) * tau) + phi_j.
Product k variance Var_k(n) = sum_j c_{k,j}^2 sigma_j^2(n)  <=  eps_k^2.

Capped aggregate  G(n) = sum_k mu_k * min(F_k(n), Gamma_k),
F_k(n) = Var_k(0) - Var_k(n), Gamma_k = Var_k(0) - eps_k^2.

Mechanism M_CSP: greedy ratio rule max DeltaG / bid, threshold payments.
"""
import numpy as np
import heapq
import itertools
import time

EPS = 1e-12


class Radiometer:
    """Equal-volume-tile radiometric coverage instance (count-based)."""

    def __init__(self, B0, beta, kappa, tau, phi, C_sens, eps2, mu=None):
        self.B0 = np.asarray(B0, float)            # (J,)
        self.beta = np.asarray(beta, float)        # (J,) bandwidth per tile
        self.kappa = np.asarray(kappa, float)      # (J,)
        self.tau = float(tau)
        self.phi = np.asarray(phi, float)          # (J,)
        self.C = np.atleast_2d(np.asarray(C_sens, float))  # (K, J)
        self.eps2 = np.atleast_1d(np.asarray(eps2, float))  # (K,)
        self.K, self.J = self.C.shape
        self.mu = np.ones(self.K) if mu is None else np.asarray(mu, float)
        self.var_base = self.var_k(np.zeros(self.J, int))   # (K,)
        self.Gamma = self.var_base - self.eps2               # (K,)
        if np.any(self.Gamma <= 0):
            raise ValueError("some tolerance already met at baseline")
        self.Gtot = float(np.sum(self.mu * self.Gamma))

    # ---- physics -----------------------------------------------------
    def bandwidth(self, n):
        return self.B0 + np.asarray(n, float) * self.beta

    def sigma2(self, n):
        return self.kappa / (self.bandwidth(n) * self.tau) + self.phi

    def var_k(self, n):
        return (self.C ** 2) @ self.sigma2(n)      # (K,)

    def feasible_counts(self, n):
        return bool(np.all(self.var_k(n) <= self.eps2 + 1e-9))

    # ---- capped aggregate -------------------------------------------
    def G(self, n):
        F = self.var_base - self.var_k(n)
        return float(np.sum(self.mu * np.minimum(F, self.Gamma)))

    def gain(self, n, j):
        """Marginal G-gain of one more channel-j tile at counts n."""
        n2 = np.array(n, int)
        n2[j] += 1
        return self.G(n2) - self.G(n)

    def covered(self, n):
        return self.Gtot - self.G(n) <= 1e-10 * max(1.0, self.Gtot)


class CostField:
    """Per-channel sorted bid lists (equal-volume tiles)."""

    def __init__(self, bids_per_channel):
        # list of 1-D arrays (unsorted, original tile order)
        self.raw = [np.asarray(b, float) for b in bids_per_channel]
        self.order = [np.argsort(b, kind="stable") for b in self.raw]
        self.sorted = [b[o] for b, o in zip(self.raw, self.order)]
        self.N = [len(b) for b in self.raw]

    def prefix_cost(self, j):
        return np.concatenate([[0.0], np.cumsum(self.sorted[j])])

    def total(self):
        return float(sum(b.sum() for b in self.raw))


# ----------------------------------------------------------------------
# Greedy allocation (counts + cheapest-prefix within channel)
# ----------------------------------------------------------------------
def greedy_allocate(rad, field, exclude=None, modified=None, record=False):
    """
    Greedy ratio rule.  exclude: set of (j, sorted_rank) tiles removed from
    the market.  modified: dict {(j, rank): new_bid}.  Returns dict with
    counts, picked tiles [(j, rank, bid)], cost, and (optionally) the
    step trace [(counts_before, j, rank, gain, bid)].
    """
    J = rad.J
    exclude = exclude or set()
    modified = modified or {}
    ptr = [0] * J                      # next candidate rank per channel
    n = np.zeros(J, int)
    picked, trace = [], []
    cost = 0.0

    def bid_of(j, r):
        return modified.get((j, r), field.sorted[j][r])

    def advance(j):
        while ptr[j] < field.N[j] and (j, ptr[j]) in exclude:
            ptr[j] += 1

    # candidate per channel = cheapest unpicked tile, but a *modified* bid
    # can change which unpicked tile in the channel is cheapest; handle by
    # keeping a per-channel heap of available (bid, rank).
    heaps = []
    for j in range(J):
        h = [(bid_of(j, r), r) for r in range(field.N[j])
             if (j, r) not in exclude]
        heapq.heapify(h)
        heaps.append(h)

    while not rad.covered(n):
        best = None
        for j in range(J):
            if not heaps[j]:
                continue
            g = rad.gain(n, j)
            if g <= EPS:
                continue
            b, r = heaps[j][0]
            ratio = g / b if b > 0 else np.inf
            if best is None or ratio > best[0] + 0:
                best = (ratio, j, r, g, b)
        if best is None:
            return None                # infeasible market
        _, j, r, g, b = best
        heapq.heappop(heaps[j])
        if record:
            trace.append((n.copy(), j, r, g, b))
        picked.append((j, r, b))
        cost += b
        n[j] += 1
    return {"counts": n, "picked": picked, "cost": cost, "trace": trace}


def greedy_thresholds(rad, field, base=None):
    """
    Threshold payment for every greedy winner, via one no-x greedy run per
    winner: theta_x = max_t  Delta_x(S_t) * b_{y_t} / Delta_{y_t}(S_t)
    over the prefixes S_t of the run with x removed.
    """
    if base is None:
        base = greedy_allocate(rad, field)
    thetas = {}
    for (j, r, b) in base["picked"]:
        run = greedy_allocate(rad, field, exclude={(j, r)}, record=True)
        assert run is not None, "Assumption A1 violated"
        best = 0.0
        for (n_before, jj, rr, g, bb) in run["trace"]:
            gx = rad.gain(n_before, j)
            if gx <= EPS:
                continue
            best = max(best, gx * bb / g)
        thetas[(j, r)] = best
    return thetas


def greedy_threshold_bisect(rad, field, j, r, hi=1e7, iters=60):
    """Binary-search threshold (validation only)."""
    lo = 0.0
    def wins(z):
        run = greedy_allocate(rad, field, modified={(j, r): z})
        return run is not None and any((jj == j and rr == r)
                                       for jj, rr, _ in run["picked"])
    if wins(hi):
        return np.inf
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if wins(mid):
            lo = mid
        else:
            hi = mid
    return lo


# ----------------------------------------------------------------------
# Exact min-cost cover on equal-volume grids (Prop. 1 enumeration)
# ----------------------------------------------------------------------
def minimal_feasible_counts(rad, Ns):
    """
    Enumerate minimal feasible count vectors for J=3 by scanning (n1, n2)
    and computing the least n3 meeting every product constraint.
    Returns array of count vectors (M, 3).  For J != 3 falls back to a
    full product scan.
    """
    J = rad.J
    if J == 3:
        out = []
        for n1 in range(Ns[0] + 1):
            for n2 in range(Ns[1] + 1):
                n3 = _min_last(rad, [n1, n2], Ns[2])
                if n3 is not None:
                    out.append((n1, n2, n3))
        return np.array(out, int)
    # generic fallback
    out = []
    for n in itertools.product(*[range(N + 1) for N in Ns]):
        if rad.feasible_counts(np.array(n)):
            out.append(n)
    return np.array(out, int)


def _min_last(rad, n_head, Nlast):
    lo, hi = 0, Nlast
    n = np.array(list(n_head) + [Nlast], int)
    if not rad.feasible_counts(n):
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        n[-1] = mid
        if rad.feasible_counts(n):
            hi = mid
        else:
            lo = mid + 1
    return lo


def exact_optimum(rad, field, counts=None):
    """Min-cost feasible selection: cheapest-prefix per channel over the
    minimal feasible count vectors."""
    if counts is None:
        counts = minimal_feasible_counts(rad, field.N)
    if len(counts) == 0:
        return None
    pref = [field.prefix_cost(j) for j in range(rad.J)]
    cost = sum(pref[j][counts[:, j]] for j in range(rad.J))
    i = int(np.argmin(cost))
    return {"counts": counts[i], "cost": float(cost[i]), "all_counts": counts}


def pivot_payments(rad, field, opt=None, counts=None):
    """
    Exact efficient (Clarke) pivot payments for the optimal allocation:
    theta*_x = OPT_{-x} - (OPT - c_x), for every tile x in S*.
    """
    if counts is None:
        counts = minimal_feasible_counts(rad, field.N)
    if opt is None:
        opt = exact_optimum(rad, field, counts)
    pref = [field.prefix_cost(j) for j in range(rad.J)]
    base_cost = opt["cost"]
    pay = {}
    for j in range(rad.J):
        nj = opt["counts"][j]
        if nj == 0:
            continue
        # tiles in S* on channel j are sorted ranks 0..nj-1
        for r in range(nj):
            c_x = field.sorted[j][r]
            # prefix costs of channel j with tile (j, r) deleted
            s = np.delete(field.sorted[j], r)
            pj = np.concatenate([[0.0], np.cumsum(s)])
            Nj_new = len(s)
            ok = counts[:, j] <= Nj_new
            cost = np.zeros(len(counts))
            for jj in range(rad.J):
                if jj == j:
                    cost += pj[np.minimum(counts[:, jj], Nj_new)]
                else:
                    cost += pref[jj][counts[:, jj]]
            cost[~ok] = np.inf
            opt_minus = float(cost.min())
            pay[(j, r)] = opt_minus - (base_cost - c_x)
    return pay


# ----------------------------------------------------------------------
# Cost-field generators
# ----------------------------------------------------------------------
def smoothed_uniform_field(rng, n_freq, n_time, lo=1.0, hi=3.0, sigma=1.0):
    from scipy.ndimage import gaussian_filter
    f = rng.uniform(lo, hi, size=(n_freq, n_time))
    f = gaussian_filter(f, sigma=sigma, mode="reflect")
    # gaussian smoothing shrinks variance; rescale back into [lo, hi]
    f = lo + (hi - lo) * (f - f.min()) / (f.max() - f.min())
    return f


def trap_field(seed=0, n_freq=15, n_time=20, trap_value=50.0,
               trap_rows=range(0, 5), trap_cols=range(5, 16)):
    rng = np.random.default_rng(seed)
    f = smoothed_uniform_field(rng, n_freq, n_time)
    for i in trap_rows:
        for t in trap_cols:
            f[i, t] = trap_value
    return f


def field_to_channels(f, rows_per_channel=5):
    J = f.shape[0] // rows_per_channel
    return [f[j * rows_per_channel:(j + 1) * rows_per_channel, :].ravel()
            for j in range(J)]


# ----------------------------------------------------------------------
# Curvature (Theorem: closed form)
# ----------------------------------------------------------------------
def total_curvature(B0, beta, N):
    B0 = np.asarray(B0, float); beta = np.asarray(beta, float)
    N = np.asarray(N, float)
    Bmax = B0 + N * beta
    r = (B0 * (B0 + beta)) / ((Bmax - beta) * Bmax)
    return 1.0 - float(r[N >= 1].min())


def operating_curvature(B0, beta, n_op):
    B0 = np.asarray(B0, float); beta = np.asarray(beta, float)
    Bop = B0 + np.asarray(n_op, float) * beta
    r = (B0 * (B0 + beta)) / ((Bop - beta) * Bop)
    r = np.where(np.asarray(n_op) >= 1, r, 1.0)
    return 1.0 - float(r.min())


# ----------------------------------------------------------------------
# Default AMSR-2-like instance
# ----------------------------------------------------------------------
def amsr_instance(eps2=0.25, B0=(10., 250., 100.), kappa=500., beta=10.,
                  C=((0.45, -0.20, 0.05),), phi=0.0, mu=None,
                  eps2_vec=None):
    J = len(B0)
    rad = Radiometer(B0=B0, beta=[beta] * J, kappa=[kappa] * J, tau=1.0,
                     phi=[phi] * J, C_sens=C,
                     eps2=eps2_vec if eps2_vec is not None else [eps2],
                     mu=mu)
    return rad


def verify_A1(rad, field):
    """Assumption A1: coverage holds after deleting any single tile.
    With equal volumes it suffices to delete one tile per channel."""
    N = np.array(field.N)
    for j in range(rad.J):
        n = N.copy(); n[j] -= 1
        if not rad.feasible_counts(n):
            return False
    return True
