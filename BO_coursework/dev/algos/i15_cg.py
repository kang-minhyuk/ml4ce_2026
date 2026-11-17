"""i2: informed screening funnel. Plates locate every clone's temperature/pH optimum, mini bioreactors screen and
refine clones, Pilot runs exploit.

1. Plates (rich medium) at space-filling, then targeted, temperature/pH conditions. A joint asymmetric-peak model
   (log titer = clone height - asymmetric quadratic drop around the clone's own optimum, shared widths, floored) gives
   each clone's estimated optimum (T_c, pH_c) and plate peak height.
2. Mini bioreactors screen the clones with the best plate peaks at their estimated optimum with a prior recipe
   (glucose at its upper bound, little glutamine early, more at feed 2), then refine the best clones.
3. A GP on log titer over (T - T_c, pH - pH_c, glutamine levels) with clone offsets and two correlated scales
   (mini bioreactor, Pilot) picks Pilot recipes (posterior mean / small UCB) among clones tested in a bioreactor.
Recipe coordinates are fractions of lab.bounds, so nothing depends on the absolute ranges.
"""

import numpy as np
import sobol_seq
from scipy.linalg import cho_solve, solve_triangular
from scipy.optimize import least_squares, minimize
from scipy.stats import norm

STUDENT_NAMES = ["TODO Firstname Lastname CID", "TODO Firstname Lastname CID"]

CFG = {
    # prior recipe (unit fractions of the bounds): glucose level for start + both feeds, glutamine start/feed1, feed2
    # glutamine start / feed-1 levels are searched on a log10 scale (sharp optimum in log concentration)
    "g0": 1.0, "la0": -2.0, "la1": -2.3, "b0": 0.6,
    "la_box": (-3.3, -0.5), "b_box": (0.1, 1.0), "log_sd_mult": 4.0,
    "plate_medium": (1.0, 0.05),     # plate glucose / glutamine (unit)
    "n_space_plates": 3,             # space-filling plates before targeted ones
    "plates_per_seg": [4, 2, 0],
    "mbr_per_seg": [1, 2, 1],
    "pilot_frac": [0.0, 0.5, 1.0],
    "order": ["plate", "mbr"],
    "mbr_before_plates": False,
    "mbr_refine_slots": [0, 0, 0],   # per segment: slots of an MBR batch spent on refinement (rest = screening)
    "pilot_acq": "ei", "beta": 0.3, "pilot_top": 2, "peak_model": "quad", "first_pilot": "prior", "pilot_rank": "observed", "pilot_tp_fixed": False, "pilot_gp": "all",
    "local_sd": (0.03, 0.08, 0.15),
    "n_local": 48,
    "restarts": 1, "maxiter": 100,
    "min_noise": 0.01,
    "log_eps": 0.01,
}

JITTER = 1e-6
LS0 = (0.3, 0.3, 0.6, 0.6, 0.3)  # length-scale prior centres: dT, dpH (unit), log10 gln start/feed1, feed2


def sobol(n, d, rng):
    return (sobol_seq.i4_sobol_generate(d, n, 1) + rng.random(d)) % 1.0


def ei(mu, sd, best):
    sd = np.maximum(sd, 1e-9)
    z = (mu - best) / sd
    return (mu - best) * norm.cdf(z) + sd * norm.pdf(z)


# ------------------------------------------------------------------------------------------------ space
class Space:
    def __init__(self, lab):
        self.lab = lab
        self.lo, self.hi = np.array(lab.bounds, float).T
        self.span = np.where(self.hi > self.lo, self.hi - self.lo, 1.0)
        self.clones = list(lab.celltypes)
        self.nc = len(self.clones)
        self.pilot, self.plate = lab.objective_reactor, lab.screening_reactor
        mids = [r for r in lab.reactor_types if r not in (self.pilot, self.plate)]
        self.mbr = mids[0] if mids else None

    def unit(self, row):
        return (np.array([float(v) for v in row[2:]]) - self.lo) / self.span

    def raw(self, t, p, g, la0, la1, b):
        lo, sp = self.lo, self.span
        T, P = lo[0] + t * sp[0], lo[1] + p * sp[1]
        r = [T, P, lo[2] + g * sp[2], lo[3] + 10 ** la0 * sp[3], lo[4] + g * sp[4], lo[5] + 10 ** la1 * sp[5],
             lo[6] + g * sp[6], lo[7] + b * sp[7], T, P, T, P]
        return np.clip(np.array(r, float), self.lo, self.hi)

    def bio_rows(self, reactor, recipes):
        out = np.empty((len(recipes), 14), dtype=object)
        out[:, 0] = reactor
        for i, (c, *rest) in enumerate(recipes):
            out[i, 1] = self.clones[int(c)]
            out[i, 2:] = [float(v) for v in self.raw(*rest)]
        return out

    def plate_rows(self, clones, t, p, g, a):
        lo, sp = self.lo, self.span
        return self.lab.plate([self.clones[int(c)] for c in clones], float(lo[0] + t * sp[0]),
                              float(lo[1] + p * sp[1]), float(lo[2] + g * sp[2]), float(lo[3] + a * sp[3]))


# ------------------------------------------------------------------------------------------------ plate peak model
def drop(d, wl, wr):
    return np.where(d < 0, (d / wl) ** 2, (d / wr) ** 2)


def fit_quad_peaks(T, P, C, Y, nc, rng, prev=None):
    """Joint asymmetric-peak fit on plate data. Returns per-clone height, optimum (t, p) in unit coordinates."""
    ly = np.log(np.maximum(Y, 1e-4 * max(Y.max(), 1e-12)))
    seen = np.zeros(nc, bool)
    seen[np.unique(C)] = True

    def unpack(th):
        return th[:nc], th[nc:2 * nc], th[2 * nc:3 * nc], np.exp(th[3 * nc:3 * nc + 4]), th[3 * nc + 4]

    def res(th):
        a, tc, pc, w, fl = unpack(th)
        pred = a[C] - np.minimum(drop(T - tc[C], w[0], w[1]) + drop(P - pc[C], w[2], w[3]), np.exp(fl))
        return np.concatenate([pred - ly, 0.1 * (tc - 0.5), 0.1 * (pc - 0.5)])

    n = len(ly)
    rows = np.arange(n)

    def jac(th):
        _, tc, pc, w, fl = unpack(th)
        dt, dp = T - tc[C], P - pc[C]
        lt, lp = dt < 0, dp < 0
        wt, wp = np.where(lt, w[0], w[1]), np.where(lp, w[2], w[3])
        act = (dt / wt) ** 2 + (dp / wp) ** 2 < np.exp(fl)
        J = np.zeros((n + 2 * nc, 3 * nc + 5))
        J[rows, C] = 1.0
        J[rows, nc + C] = np.where(act, 2 * dt / wt ** 2, 0.0)
        J[rows, 2 * nc + C] = np.where(act, 2 * dp / wp ** 2, 0.0)
        g_t, g_p = np.where(act, 2 * (dt / wt) ** 2, 0.0), np.where(act, 2 * (dp / wp) ** 2, 0.0)
        J[rows, 3 * nc] = np.where(lt, g_t, 0.0)
        J[rows, 3 * nc + 1] = np.where(lt, 0.0, g_t)
        J[rows, 3 * nc + 2] = np.where(lp, g_p, 0.0)
        J[rows, 3 * nc + 3] = np.where(lp, 0.0, g_p)
        J[rows, 3 * nc + 4] = np.where(act, 0.0, -np.exp(fl))
        J[n + np.arange(nc), nc + np.arange(nc)] = 0.1
        J[n + nc + np.arange(nc), 2 * nc + np.arange(nc)] = 0.1
        return J

    a0 = np.array([ly[C == c].max() if seen[c] else ly.mean() for c in range(nc)])
    t0 = np.array([T[C == c][np.argmax(ly[C == c])] if seen[c] else 0.5 for c in range(nc)])
    p0 = np.array([P[C == c][np.argmax(ly[C == c])] if seen[c] else 0.5 for c in range(nc)])
    lo = np.concatenate([np.full(nc, -60.0), np.full(nc, -0.2), np.full(nc, -0.2), [np.log(0.03)] * 4, [-3.0]])
    hi = np.concatenate([np.full(nc, 60.0), np.full(nc, 1.2), np.full(nc, 1.2), [np.log(5.0)] * 4, [3.0]])
    starts = [np.concatenate([a0, t0, p0, np.log([0.3, 0.6, 0.6, 0.2]), [np.log(3.0)]])]
    if prev is not None:
        starts.insert(0, prev)
    starts.append(np.concatenate([a0, t0 + rng.normal(0, 0.05, nc), p0 + rng.normal(0, 0.05, nc),
                                  np.log([0.3, 0.6, 0.6, 0.2]) + rng.normal(0, 0.3, 4), [np.log(3.0)]]))
    best = None
    for st in starts:
        try:
            r = least_squares(res, np.clip(st, lo + 1e-9, hi - 1e-9), jac=jac, bounds=(lo, hi), loss="soft_l1", f_scale=0.3,
                              max_nfev=400)
        except Exception:  # pylint: disable=broad-exception-caught
            continue
        if best is None or r.cost < best.cost:
            best = r
    a, tc, pc, w, fl = unpack(best.x)
    return {"height": a, "t": np.clip(tc, 0, 1), "p": np.clip(pc, 0, 1), "seen": seen, "theta": best.x}


def gauss(d, wl, wr):
    return np.exp(-np.where(d < 0, (d / wl) ** 2, (d / wr) ** 2))


def fit_gauss_peaks(T, P, C, ly, nc, rng, per_clone_k, prev=None):
    """log y = h_c + k_c (max(fT, .2) max(fP, .2) - 1); fT, fP asymmetric Gaussians around the clone optimum with
    widths shared by all clones (unit coordinates). Growth-rate factors act multiplicatively, so log titer saturates
    away from the optimum rather than dropping quadratically."""
    nk = nc if per_clone_k else 1
    n = len(ly)
    seen = np.zeros(nc, bool)
    seen[np.unique(C)] = True

    def unpack(th):
        k = np.exp(th[3 * nc:3 * nc + nk])
        return th[:nc], th[nc:2 * nc], th[2 * nc:3 * nc], k, np.exp(th[3 * nc + nk:3 * nc + nk + 4])

    def res(th):
        h, tc, pc, k, w = unpack(th)
        F = np.maximum(gauss(T - tc[C], w[0], w[1]), 0.2) * np.maximum(gauss(P - pc[C], w[2], w[3]), 0.2)
        kC = k[C] if per_clone_k else k[0]
        lk = np.log(k)
        return np.concatenate([h[C] + kC * (F - 1) - ly, 0.1 * (tc - 0.5), 0.1 * (pc - 0.5), 0.3 * (lk - lk.mean())])

    npar = 3 * nc + nk + 4
    rows = np.arange(n)
    kcol = 3 * nc + (C if per_clone_k else 0)

    def jac(th):
        h, tc, pc, k, w = unpack(th)
        dt, dp = T - tc[C], P - pc[C]
        lt, lp = dt < 0, dp < 0
        wt, wp = np.where(lt, w[0], w[1]), np.where(lp, w[2], w[3])
        gt, gp = np.exp(-(dt / wt) ** 2), np.exp(-(dp / wp) ** 2)
        at, ap = gt > 0.2, gp > 0.2
        GT, GP = np.maximum(gt, 0.2), np.maximum(gp, 0.2)
        kC = k[C] if per_clone_k else k[0]
        J = np.zeros((n + 2 * nc + nk, npar))
        J[rows, C] = 1.0
        J[rows, nc + C] = np.where(at, kC * GP * gt * 2 * dt / wt ** 2, 0.0)
        J[rows, 2 * nc + C] = np.where(ap, kC * GT * gp * 2 * dp / wp ** 2, 0.0)
        np.add.at(J, (rows, kcol), kC * (GT * GP - 1))
        dwt = np.where(at, kC * GP * gt * 2 * (dt / wt) ** 2, 0.0)
        dwp = np.where(ap, kC * GT * gp * 2 * (dp / wp) ** 2, 0.0)
        o = 3 * nc + nk
        J[rows, o] = np.where(lt, dwt, 0.0)
        J[rows, o + 1] = np.where(lt, 0.0, dwt)
        J[rows, o + 2] = np.where(lp, dwp, 0.0)
        J[rows, o + 3] = np.where(lp, 0.0, dwp)
        J[n + np.arange(nc), nc + np.arange(nc)] = 0.1
        J[n + nc + np.arange(nc), 2 * nc + np.arange(nc)] = 0.1
        J[n + 2 * nc:, 3 * nc:3 * nc + nk] = 0.3 * (np.eye(nk) - 1.0 / nk)
        return J

    h0 = np.array([ly[C == c].max() if seen[c] else ly.mean() for c in range(nc)])
    t0 = np.array([T[C == c][np.argmax(ly[C == c])] if seen[c] else 0.5 for c in range(nc)])
    p0 = np.array([P[C == c][np.argmax(ly[C == c])] if seen[c] else 0.5 for c in range(nc)])
    lo = np.concatenate([np.full(nc, -60.0), np.full(nc, -0.3), np.full(nc, -0.3), np.full(nk, np.log(0.05)),
                         [np.log(0.03)] * 4])
    hi = np.concatenate([np.full(nc, 60.0), np.full(nc, 1.3), np.full(nc, 1.3), np.full(nk, np.log(20.0)),
                         [np.log(5.0)] * 4])
    w0 = np.log([0.33, 0.8, 0.84, 0.2])
    starts = [np.concatenate([h0, t0, p0, np.full(nk, np.log(2.0)), w0])]
    starts.append(np.concatenate([h0, t0, p0, np.full(nk, np.log(2.0)), w0 + rng.normal(0, 0.3, 4)]))
    if prev is not None and len(prev) == npar:
        starts.insert(0, prev)
    best = None
    for st in starts:
        try:
            r_ = least_squares(res, np.clip(st, lo + 1e-9, hi - 1e-9), jac=jac, bounds=(lo, hi),
                               loss="soft_l1", f_scale=0.3, max_nfev=300)
        except Exception:  # pylint: disable=broad-exception-caught
            continue
        if best is None or r_.cost < best.cost:
            best = r_
    h, tc, pc, k, w = unpack(best.x)
    return h, np.clip(tc, 0, 1), np.clip(pc, 0, 1), best.x


def fit_peaks(T, P, C, Y, nc, rng, prev=None, model="quad"):
    if model == "quad":
        return fit_quad_peaks(T, P, C, Y, nc, rng, prev)
    ly = np.log(np.maximum(Y, 1e-4 * max(Y.max(), 1e-12)))
    seen = np.zeros(nc, bool)
    seen[np.unique(C)] = True
    prev = prev if isinstance(prev, dict) else {}
    _, tc, pc, th_k = fit_gauss_peaks(T, P, C, ly, nc, rng, True, prev.get("k"))
    h, _, _, th_1 = fit_gauss_peaks(T, P, C, ly, nc, rng, False, prev.get("1"))
    return {"height": h, "t": tc, "p": pc, "seen": seen, "theta": {"k": th_k, "1": th_1}}


# ------------------------------------------------------------------------------------------------ bio GP
class BioGP:
    """GP on log titer of bioreactor runs: inputs z (D dims); kernel
    B[s, s'] * (amp^2 RBF_ARD(z) + ampc^2 delta(c, c')) with s in {mbr, pilot}; per-scale mean and noise."""

    def __init__(self, cfg, rng, D):
        self.cfg, self.rng, self.D = cfg, rng, D
        self.theta = None

    def unpack(self, th):
        D = self.D
        ls = np.exp(th[:D])
        amp, ampc = np.exp(th[D]), np.exp(th[D + 1])
        l11, l21, l22 = np.exp(th[D + 2]), th[D + 3], np.exp(th[D + 4])
        means = th[D + 5:D + 7]
        noise = np.exp(th[D + 7:D + 9])
        lb = np.array([[l11, 0.0], [l21, l22]])
        return ls, amp, ampc, lb, means, noise

    def kmat(self, z1, c1, s1, z2, c2, s2, th):
        ls, amp, ampc, lb, _, _ = self.unpack(th)
        a, b = z1 / ls, z2 / ls
        sq = np.maximum(np.sum(a ** 2, 1)[:, None] + np.sum(b ** 2, 1)[None, :] - 2 * a @ b.T, 0)
        bm = lb @ lb.T
        return bm[np.ix_(s1, s2)] * (amp ** 2 * np.exp(-0.5 * sq) + ampc ** 2 * (c1[:, None] == c2[None, :]))

    def nll(self, th):
        """Negative log marginal likelihood (+ length-scale prior) and its analytic gradient."""
        D = self.D
        ls, amp, ampc, lb, means, noise = self.unpack(th)
        z, c, s = self.z, self.c, self.s
        a = z / ls
        R = np.exp(-0.5 * np.maximum(np.sum(a ** 2, 1)[:, None] + np.sum(a ** 2, 1)[None, :] - 2 * a @ a.T, 0))
        Dl = c[:, None] == c[None, :]
        bm = lb @ lb.T
        Bs = bm[np.ix_(s, s)]
        G = amp ** 2 * R + ampc ** 2 * Dl
        K = Bs * G + np.diag(noise[s] ** 2 + JITTER)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10, np.zeros_like(th)
        r = self.y - means[s]
        al = cho_solve((L, True), r)
        lp = np.log(self.ls0)
        val = 0.5 * r @ al + np.sum(np.log(np.diag(L))) + 0.5 * np.sum((th[:D] - lp) ** 2)
        W = cho_solve((L, True), np.eye(len(r))) - np.outer(al, al)
        g = np.zeros_like(th)
        BR = Bs * amp ** 2 * R
        for k in range(D):
            dz = (z[:, k][:, None] - z[:, k][None, :]) / ls[k]
            g[k] = 0.5 * np.sum(W * BR * dz ** 2)
        g[:D] += th[:D] - lp
        g[D] = np.sum(W * BR)
        g[D + 1] = np.sum(W * Bs * ampc ** 2 * Dl)
        l11, l21, l22 = lb[0, 0], lb[1, 0], lb[1, 1]
        for i, dB in ((D + 2, [[2 * l11 ** 2, l11 * l21], [l11 * l21, 0.0]]), (D + 3, [[0.0, l11], [l11, 2 * l21]]),
                      (D + 4, [[0.0, 0.0], [0.0, 2 * l22 ** 2]])):
            g[i] = 0.5 * np.sum(W * np.asarray(dB)[np.ix_(s, s)] * G)
        for k in (0, 1):
            g[D + 5 + k] = -np.sum(al[s == k])
            g[D + 7 + k] = np.sum(np.diag(W)[s == k]) * noise[k] ** 2
        return val, g

    def fit(self, z, c, s, y, ls0):
        D = self.D
        self.ls0 = np.asarray(ls0, float)
        self.z, self.c, self.s = np.asarray(z, float), np.asarray(c, int), np.asarray(s, int)
        self.ymu, self.ysd = float(np.mean(y)), max(float(np.std(y)), 1e-6)
        self.y = (np.asarray(y, float) - self.ymu) / self.ysd
        mn = min(np.log(self.cfg["min_noise"] / self.ysd), np.log(0.5))
        bnds = [(np.log(0.3 * l), np.log(10 * l)) for l in self.ls0] + [(np.log(0.05), np.log(5))] * 2 \
            + [(np.log(0.2), np.log(3)), (-3, 3), (np.log(0.05), np.log(3))] + [(-5, 5)] * 2 + [(mn, np.log(1.0))] * 2
        m = [float(self.y[self.s == k].mean()) if np.any(self.s == k) else 0.0 for k in (0, 1)]
        default = np.concatenate([np.log(self.ls0), [0.0, np.log(0.7), 0.0, 0.8, np.log(0.6)], m, [np.log(0.05)] * 2])
        starts = [self.theta] if self.theta is not None else []
        starts.append(default)
        for _ in range(self.cfg["restarts"]):
            starts.append(default + self.rng.normal(0, 0.4, len(default)))
        lo, hi = np.array(bnds).T
        best = None
        for st in starts:
            r = minimize(self.nll, np.clip(st, lo, hi), jac=True, method="L-BFGS-B", bounds=bnds,
                         options={"maxiter": self.cfg["maxiter"]})
            if best is None or r.fun < best.fun:
                best = r
        self.theta = best.x
        _, _, _, _, means, noise = self.unpack(self.theta)
        K = self.kmat(self.z, self.c, self.s, self.z, self.c, self.s, self.theta) + np.diag(noise[self.s] ** 2 + JITTER)
        self.L = np.linalg.cholesky(K)
        self.alpha = cho_solve((self.L, True), self.y - means[self.s])

    def predict(self, z, c, s):
        z = np.atleast_2d(z)
        c, s = np.asarray(c, int), np.asarray(s, int)
        ls, amp, ampc, lb, means, _ = self.unpack(self.theta)
        ks = self.kmat(self.z, self.c, self.s, z, c, s, self.theta)
        mu = means[s] + ks.T @ self.alpha
        v = solve_triangular(self.L, ks, lower=True)
        bm = lb @ lb.T
        var = bm[s, s] * (amp ** 2 + ampc ** 2) - np.sum(v ** 2, 0)
        return mu, np.sqrt(np.maximum(var, 1e-12))


# ------------------------------------------------------------------------------------------------ state
class State:
    def __init__(self, lab, cfg, rng):
        self.sp = Space(lab)
        self.cfg, self.rng = cfg, rng
        self.rows, self.ys = [], []
        self.peaks = None
        self.gp = None
        self.n_plates_own = 0

    def add(self, rows, ys):
        self.rows += list(rows)
        self.ys += [float(v) for v in ys]

    def split(self):
        sp = self.sp
        plate, bio = [], []
        for r, y in zip(self.rows, self.ys):
            (plate if r[0] == sp.plate else bio).append((r, y))
        return plate, bio

    def update_peaks(self):
        plate, _ = self.split()
        if not plate:
            self.peaks = None
            return
        if self.peaks is not None and self.peaks.get("n") == len(plate):
            return
        sp = self.sp
        U = np.array([sp.unit(r) for r, _ in plate])
        C = np.array([sp.clones.index(r[1]) for r, _ in plate])
        Y = np.array([y for _, y in plate])
        prev = self.peaks["theta"] if self.peaks is not None else None
        self.peaks = fit_peaks(U[:, 0], U[:, 1], C, Y, sp.nc, self.rng, prev, self.cfg["peak_model"])
        self.peaks["n"] = len(plate)

    def clone_opt(self, c):
        if self.peaks is None or not self.peaks["seen"][c]:
            return 0.5, 0.5
        return float(self.peaks["t"][c]), float(self.peaks["p"][c])

    def bio_data(self):
        sp = self.sp
        _, bio = self.split()
        Z, C, S, Y = [], [], [], []
        for r, y in bio:
            u = sp.unit(r)
            c = sp.clones.index(r[1])
            tc, pc = self.clone_opt(c)
            t = (120 * u[0] + 120 * u[8] + 96 * u[10]) / 336
            p = (120 * u[1] + 120 * u[9] + 96 * u[11]) / 336
            Z.append([t - tc, p - pc, np.log10(max(u[3], 1e-4)), np.log10(max(u[5], 1e-4)), u[7]])
            C.append(c)
            S.append(1 if r[0] == sp.pilot else 0)
            Y.append(y)
        return np.array(Z), np.array(C, int), np.array(S, int), np.array(Y)

    def fit_gp(self):
        Z, C, S, Y = self.bio_data()
        if len(Y) < 3:
            self.gp = None
            return
        y = np.maximum(Y, 0)
        z = np.log(y + self.cfg["log_eps"] * max(y.max(), 1e-9))
        if self.gp is None:
            self.gp = BioGP(self.cfg, self.rng, Z.shape[1])
        self.gp.fit(Z, C, S, z, LS0)


# ------------------------------------------------------------------------------------------------ policies
def plate_rank(st):
    """Clones ordered by plate peak height (unseen clones first get explored by the plate policy)."""
    if st.peaks is None:
        return list(range(st.sp.nc))
    h = np.where(st.peaks["seen"], st.peaks["height"], -np.inf)
    return list(np.argsort(-h))


def choose_plate(st):
    sp, cfg, rng = st.sp, st.cfg, st.rng
    size = sp.lab.batch_sizes[sp.plate]
    g, a = cfg["plate_medium"]
    plate, _ = st.split()
    counts = np.zeros(sp.nc)
    for r, _ in plate:
        counts[sp.clones.index(r[1])] += 1
    if st.n_plates_own < cfg["n_space_plates"] or st.peaks is None:
        pts = sobol(16, 2, np.random.default_rng(int(sp.lab.seed) + 3))
        t, p = pts[st.n_plates_own % 16]
        order = list(np.argsort(counts + rng.random(sp.nc) * 0.1))
        clones = [order[i % sp.nc] for i in range(size)]
        return sp.plate_rows(clones, t, p, g, a)
    # targeted: condition at the estimated optimum of the best clone whose optimum has no plate point nearby
    rank = plate_rank(st)
    U = np.array([sp.unit(r)[:2] for r, _ in plate])
    C = np.array([sp.clones.index(r[1]) for r, _ in plate])
    target = rank[0]
    for c in rank[:max(6, sp.nc // 3)]:
        tc, pc = st.clone_opt(c)
        d = np.sqrt(((U[C == c] - [tc, pc]) ** 2).sum(1)) if np.any(C == c) else np.array([np.inf])
        if d.min() > 0.06:
            target = c
            break
    t, p = st.clone_opt(target)
    t, p = float(np.clip(t + rng.normal(0, 0.02), 0, 1)), float(np.clip(p + rng.normal(0, 0.02), 0, 1))
    # wells: clones whose optimum is near this condition first, then by rank
    dist = np.array([np.hypot(*(np.array(st.clone_opt(c)) - [t, p])) for c in range(sp.nc)])
    score = dist - 0.02 * (sp.nc - np.argsort(np.argsort(rank)))
    order = list(np.argsort(score))
    k = min(size, sp.nc)
    clones = order[:k]
    while len(clones) < size:
        clones.append(clones[len(clones) % k])
    return sp.plate_rows(clones, t, p, g, a)


def prior_recipe(st, c):
    t, p = st.clone_opt(c)
    return (c, t, p, st.cfg["g0"], st.cfg["la0"], st.cfg["la1"], st.cfg["b0"])


def candidates(st, clones, tp_fixed=False, scale=None):
    """Recipes for the given clones: prior recipe at the estimated optimum plus local perturbations of the best
    observed bio recipes of each clone (of the given scale, if any). tp_fixed: perturb the glutamine levels only."""
    cfg, rng = st.cfg, st.rng
    out = []
    Z, C, S, Y = st.bio_data()
    for c in clones:
        base = [prior_recipe(st, c)]
        idx = np.where((C == c) & ((S == scale) if scale is not None else True))[0]
        tc, pc = st.clone_opt(c)
        for i in idx[np.argsort(-Y[idx])][:3]:
            base.append((c, float(np.clip(Z[i, 0] + tc, 0, 1)), float(np.clip(Z[i, 1] + pc, 0, 1)), cfg["g0"],
                         float(np.clip(Z[i, 2], *cfg["la_box"])), float(np.clip(Z[i, 3], *cfg["la_box"])),
                         float(np.clip(Z[i, 4], *cfg["b_box"]))))
        for b in base:
            out.append(b)
            for sd in cfg["local_sd"]:
                for _ in range(cfg["n_local"] // len(cfg["local_sd"])):
                    dt, dp = (0.0, 0.0) if tp_fixed else (rng.normal(0, sd), rng.normal(0, sd))
                    out.append((c, np.clip(b[1] + dt, 0, 1), np.clip(b[2] + dp, 0, 1),
                                cfg["g0"], np.clip(b[4] + rng.normal(0, sd * cfg["log_sd_mult"]), *cfg["la_box"]),
                                np.clip(b[5] + rng.normal(0, sd * cfg["log_sd_mult"]), *cfg["la_box"]),
                                np.clip(b[6] + rng.normal(0, sd * 2), *cfg["b_box"])))
    return out


def gp_eval(st, recs, scale, acq, inc=None, gp=None):
    gp = gp if gp is not None else st.gp
    if gp is None:
        return np.zeros(len(recs))
    Z = np.array([[t - st.clone_opt(c)[0], p - st.clone_opt(c)[1], a0, a1, b] for c, t, p, g, a0, a1, b in recs])
    C = np.array([r[0] for r in recs], int)
    mu, sd = gp.predict(Z, C, np.full(len(recs), scale))
    if acq == "mean":
        return mu
    if acq == "ei":
        return ei(mu, sd, inc)
    return mu + st.cfg["beta"] * sd


def bio_clones(st):
    _, bio = st.split()
    return sorted(set(st.sp.clones.index(r[1]) for r, _ in bio))


def too_close(st, rec, reactor):
    _, bio = st.split()
    sp = st.sp
    raw = sp.raw(*rec[1:])
    for r, _ in bio:
        if r[0] == reactor and sp.clones.index(r[1]) == rec[0]:
            if np.max(np.abs(np.array([float(v) for v in r[2:]]) - raw) / sp.span) < 0.01:
                return True
    return False


def choose_mbr(st, seg):
    sp, cfg = st.sp, st.cfg
    size = sp.lab.batch_sizes[sp.mbr]
    refine = min(cfg["mbr_refine_slots"][min(seg, len(cfg["mbr_refine_slots"]) - 1)], size)
    tested = set(bio_clones(st))
    if st.gp is None:
        refine = 0
    screen = [c for c in plate_rank(st) if c not in tested][:size - refine]
    refine = size - len(screen)
    recs = [prior_recipe(st, c) for c in screen]
    if refine > 0 and st.gp is not None:
        _, _, S, Y = st.bio_data()
        inc = float(np.max(st.gp.y[st.gp.s == 1])) if np.any(st.gp.s == 1) else float(np.max(st.gp.y))
        top = rank_bio_clones(st)[:2]
        cand = [r for r in candidates(st, top) if not too_close(st, r, sp.mbr)]
        v = gp_eval(st, cand, 1, "ei", inc)
        for i in np.argsort(-v):
            if len(recs) >= size:
                break
            if all(np.max(np.abs(np.array(cand[i][1:]) - np.array(r[1:]))) > 0.02 or cand[i][0] != r[0] for r in recs):
                recs.append(cand[i])
    while len(recs) < size:
        c = plate_rank(st)[len(recs) % sp.nc]
        recs.append(prior_recipe(st, c))
    return sp.bio_rows(sp.mbr, recs)


def rank_bio_clones(st):
    """Bio-tested clones ordered by GP-predicted Pilot value at their best observed/prior recipe."""
    clones = bio_clones(st)
    if st.gp is None or not clones:
        _, bio = st.split()
        best = {}
        for r, y in bio:
            c = st.sp.clones.index(r[1])
            best[c] = max(best.get(c, -1), y)
        return sorted(best, key=lambda c: -best[c])
    vals = []
    for c in clones:
        v = gp_eval(st, candidates(st, [c])[:8], 1, "mean")
        vals.append(v.max())
    return [clones[i] for i in np.argsort(-np.array(vals))]


def rank_observed(st):
    """Clones ordered by measured performance: best Pilot titer first (clones run at Pilot scale), then the rest by
    best measured mini-bioreactor titer. Within one run all measurements share one noise factor, so measured titers
    compare exactly."""
    sp = st.sp
    _, bio = st.split()
    best_p, best_m = {}, {}
    for r, y in bio:
        c = sp.clones.index(r[1])
        d = best_p if r[0] == sp.pilot else best_m
        d[c] = max(d.get(c, -np.inf), y)
    pil = sorted(best_p, key=lambda c: -best_p[c])
    rest = sorted([c for c in best_m if c not in best_p], key=lambda c: -best_m[c])
    if pil and rest:
        # a clone never run at Pilot scale competes with the Pilot incumbent through its MBR titer relative to the
        # incumbent's MBR titer
        inc = pil[0]
        if inc in best_m and best_m[rest[0]] > best_m[inc]:
            return [inc, rest[0]] + [c for c in pil[1:]] + rest[1:]
    return pil + rest


def prior_titers(st, reactor):
    """Best titer per clone among runs of `reactor` made at the prior glutamine recipe (same feeding, so clones
    compare directly)."""
    sp, cfg = st.sp, st.cfg
    _, bio = st.split()
    out = {}
    for r, y in bio:
        if r[0] != reactor:
            continue
        u = sp.unit(r)
        if abs(np.log10(max(u[3], 1e-6)) - cfg["la0"]) < 0.05 and abs(np.log10(max(u[5], 1e-6)) - cfg["la1"]) < 0.05 \
                and abs(u[7] - cfg["b0"]) < 0.02:
            c = sp.clones.index(r[1])
            out[c] = max(out.get(c, -np.inf), y)
    return out


def choose_pilot_race(st):
    """Pilot clone by racing: clones are compared by their mini-bioreactor titer at the prior recipe; a clone that
    beats the Pilot incumbent's gets one Pilot run at the prior recipe, then Pilot titers decide. Otherwise the
    Pilot run refines the glutamine levels of the incumbent clone (GP expected improvement)."""
    sp, cfg = st.sp, st.cfg
    size = sp.lab.batch_sizes[sp.pilot]
    mp = prior_titers(st, sp.mbr) if sp.mbr is not None else {}
    _, bio = st.split()
    pp = {}
    for r, y in bio:
        if r[0] == sp.pilot:
            c = sp.clones.index(r[1])
            pp[c] = max(pp.get(c, -np.inf), y)
    order = sorted(mp, key=lambda c: -mp[c])
    if not order:
        order = rank_observed(st) or plate_rank(st)
    if not pp:
        return sp.bio_rows(sp.pilot, [prior_recipe(st, order[0])] * size)
    inc = max(pp, key=pp.get)
    challengers = [c for c in order if c not in pp]
    if challengers and mp.get(challengers[0], -np.inf) > cfg["switch"] * mp.get(inc, -np.inf):
        return sp.bio_rows(sp.pilot, [prior_recipe(st, challengers[0])] * size)
    cand = [r for r in candidates(st, [inc], cfg["pilot_tp_fixed"]) if not too_close(st, r, sp.pilot)]
    gp = st.gp
    if cfg["race_gp"] == "clone":
        # model the incumbent clone alone (its mini-bioreactor and Pilot runs): other clones only blur its response
        Z, C, S, Y = st.bio_data()
        m = C == inc
        if m.sum() >= 2:
            y = np.maximum(Y[m], 0)
            gp = BioGP(cfg, st.rng, Z.shape[1])
            gp.fit(Z[m], C[m], S[m], np.log(y + cfg["log_eps"] * max(y.max(), 1e-9)), LS0)
    if gp is None or not cand:
        return sp.bio_rows(sp.pilot, [prior_recipe(st, inc)] * size)
    inc_y = float(np.max(gp.y[gp.s == 1])) if np.any(gp.s == 1) else float(np.max(gp.y))
    v = gp_eval(st, cand, 1, cfg["pilot_acq"], inc_y, gp)
    return sp.bio_rows(sp.pilot, [cand[i] for i in np.argsort(-v)[:size]])


def choose_pilot(st):
    sp, cfg = st.sp, st.cfg
    if cfg["pilot_policy"] == "race":
        return choose_pilot_race(st)
    size = sp.lab.batch_sizes[sp.pilot]
    ranker = rank_observed if cfg["pilot_rank"] == "observed" else rank_bio_clones
    clones = ranker(st)[:st.cfg["pilot_top"]] or plate_rank(st)[:1]
    n_pilot = sum(1 for r in st.rows if r[0] == sp.pilot)
    if n_pilot == 0 and cfg["first_pilot"] == "prior":
        return sp.bio_rows(sp.pilot, [prior_recipe(st, clones[0])] * size)
    acq = "mean" if n_pilot == 0 else cfg["pilot_acq"]
    gp = st.gp
    if cfg["pilot_gp"] == "pilot" and n_pilot > 0:
        # glutamine optima differ between scales: model the Pilot response from Pilot runs alone
        Z, C, S, Y = st.bio_data()
        m = S == 1
        y = np.maximum(Y[m], 0)
        gp = BioGP(cfg, st.rng, Z.shape[1])
        gp.fit(Z[m], C[m], S[m], np.log(y + cfg["log_eps"] * max(y.max(), 1e-9)), LS0)
    cand = [r for r in candidates(st, clones, cfg["pilot_tp_fixed"], 1 if cfg["pilot_gp"] == "pilot" else None)
            if not too_close(st, r, sp.pilot)]
    inc = None
    if gp is not None:
        inc = float(np.max(gp.y[gp.s == 1])) if np.any(gp.s == 1) else float(np.max(gp.y))
    v = gp_eval(st, cand, 1, acq, inc, gp)
    recs = []
    for i in np.argsort(-v):
        if len(recs) >= size:
            break
        recs.append(cand[i])
    while len(recs) < size:
        recs.append(prior_recipe(st, clones[0]))
    return sp.bio_rows(sp.pilot, recs)


# ------------------------------------------------------------------------------------------------ scale plan
def keeps_budget_usable(lab, reactor):
    pilot = lab.objective_reactor
    if reactor == pilot:
        return True
    budget_after = lab.remaining_budget - lab.batch_cost(reactor)
    exp_after = lab.remaining_experiments - lab.batch_sizes[reactor]
    pil = lab.batch_sizes[pilot]
    return exp_after >= pil and (budget_after // lab.batch_cost(pilot)) * pil <= exp_after


def next_scale(lab, sp, cfg, plan):
    pc = lab.batch_cost(sp.pilot)
    ckpts = sorted(lab.score_checkpoints)
    upcoming = [c for c in ckpts if c >= lab.spent + 1e-9]
    seg = min(len(ckpts) - len(upcoming), len(ckpts) - 1) if upcoming else len(ckpts) - 1
    seg_end = min(upcoming[0], lab.budget) if upcoming else lab.budget
    money = seg_end - lab.spent
    last = len(upcoming) <= 1
    pil = plan.setdefault("_pilots", {})
    if cfg["seg_advance"] and not last and pil.get(seg, 0) > 0 and money < pc:
        # this checkpoint already has a Pilot run and its leftover money cannot buy another: plan the next segment
        # (a Pilot fired now would spend the next segment's money before its screening)
        seg, seg_end = seg + 1, (min(upcoming[1], lab.budget) if len(upcoming) > 1 else lab.budget)
        money = seg_end - lab.spent
        last = len(upcoming) <= 2
    if seg not in plan:
        pick = lambda key: cfg[key][min(seg, len(cfg[key]) - 1)]  # noqa: E731
        max_p = int(money // pc)
        want = min(max(int(pick("pilot_frac") * money // pc), 1), max_p)
        plan[seg] = {"pilot": want, "plate": pick("plates_per_seg"), "mbr": pick("mbr_per_seg")}
    st = plan[seg]
    reserve = st["pilot"] * pc

    def cheap_ok(r):
        return r is not None and lab.can_run(r) and keeps_budget_usable(lab, r) and money - lab.batch_cost(r) >= reserve

    for key in cfg["order"]:
        r = sp.plate if key == "plate" else sp.mbr
        if st[key] > 0 and cheap_ok(r):
            if key == "plate" and cfg["mbr_before_plates"] and st["mbr"] > 0 and sp.mbr is not None \
                    and money - lab.batch_cost(r) - lab.batch_cost(sp.mbr) < reserve \
                    and money - lab.batch_cost(sp.mbr) >= reserve:
                continue
            st[key] -= 1
            return r, seg
    if lab.can_run(sp.pilot) and money >= pc:
        st["pilot"] = max(st["pilot"] - 1, 0)
        pil[seg] = pil.get(seg, 0) + 1
        return sp.pilot, seg
    if last or money < pc:
        for r in (sp.mbr, sp.plate):
            if r and lab.can_run(r) and keeps_budget_usable(lab, r) and money >= lab.batch_cost(r):
                return r, seg
    if lab.can_run(sp.pilot):
        pil[seg] = pil.get(seg, 0) + 1
        return sp.pilot, seg
    rs = lab.affordable_reactors()
    return (rs[-1] if rs else None), seg


# ------------------------------------------------------------------------------------------------ main
def my_algorithm(initial_x, initial_y, lab, seed, cfg=None):
    """Informed screening funnel; see module docstring."""
    cfg = dict(CFG if cfg is None else cfg)
    rng = np.random.default_rng(seed)
    st = State(lab, cfg, rng)
    st.add(initial_x[:len(initial_y)], initial_y)
    plan = {}
    while lab.affordable_reactors():
        reactor, seg = next_scale(lab, st.sp, cfg, plan)
        if reactor is None:
            break
        st.update_peaks()
        st.fit_gp()
        if reactor == st.sp.plate:
            batch = choose_plate(st)
            st.n_plates_own += 1
        elif reactor == st.sp.pilot:
            batch = choose_pilot(st)
        else:
            batch = choose_mbr(st, seg)
        y, _ = lab(batch)
        if len(y) == 0:
            break
        st.add(batch[:len(y)], y)


CFG.update({'race_gp': 'all'})
CFG.update({'pilot_policy': 'race', 'switch': 1.0})
CFG.update({'seg_advance': True})
CFG.update({'pilot_tp_fixed': True})


CFG.update({'mbr_before_plates': True})


CFG.update({'race_gp': 'clone'})
