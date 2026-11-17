"""a5: structured multi-fidelity funnel (plates -> mini bioreactors -> Pilot) with an additive GP.

Recipe family (unit coordinates relative to lab.bounds): constant temperature t and pH p, one glucose level g for the
start and both feeds, one glutamine level a for the start and feed 1, glutamine b at feed 2.
Model: GP on log-titer over (clone, t, p, g, a, b) and scale: additive 1-D components (feed-2 term only for
bioreactors), clone-specific (t, p) surface, clone offset, weak bioreactor interaction term; scales coupled by a learned
3x3 covariance. Plates locate each clone's (t, p) optimum; mini bioreactors screen distinct promising clones at their
predicted best recipe (Pilot EI); Pilot runs exploit (posterior mean / small UCB) among clones tested in a bioreactor.
Scale plan per scoring segment: plates, then mini-bioreactor batches, then the Pilot runs just before the checkpoint.
"""

import numpy as np
import sobol_seq
from scipy.linalg import cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.stats import norm

STUDENT_NAMES = ["TODO Firstname Lastname CID", "TODO Firstname Lastname CID"]

CFG = {
    "log_eps": 0.01,
    "restarts": 2,
    "maxiter": 120,
    "warm_maxiter": 40,
    "refit_every": 3,
    "ls_prior": (np.log(0.5), 1.5),
    "min_noise": 0.01,
    "interaction": True,
    "project_nonconforming": True,
    # scale plan
    "plates_per_seg": [3, 1, 0],
    "mbr_per_seg": [1, 1, 0],
    "pilot_frac": [0.0, 0.5, 1.0],
    "order": ["plate", "mbr"],
    "mbr_before_plates": True,
    # plates
    "plate_medium": (0.3, 0.3),      # (glucose, glutamine) unit levels of the plate medium; None = model choice
    "plate_beta": 2.0,
    # bioreactors
    "pool": 512,
    "top_clones": 8,
    "n_refine": 4,
    "mbr_distinct": True,
    "pilot_acq": "ucb",
    "first_pilot_acq": "mean",
    "beta": 0.3,
    "pilot_bio_seen": True,
    "snap": 0.0,
    "box": None,                     # per-dim unit search bounds [(lo, hi)] * 5 for bioreactor recipes; None = [0,1]
}

JITTER = 1e-6


# ------------------------------------------------------------------------------------------------ GP
class MFGP:
    """Additive multi-fidelity GP: K = B[s,s'] * sum_k comp_k; comp = RBF-ARD on a dim subset (or constant),
    optionally times delta(clone) and a fidelity mask. Per-fidelity constant mean and noise; ML-II with gradients."""

    def __init__(self, n_fid, comps, rng, cfg):
        self.n_fid, self.rng, self.cfg = n_fid, rng, cfg
        self.comps = []
        for c in comps:
            m = np.ones(n_fid)
            if c.get("mask") is not None:
                m = np.zeros(n_fid)
                m[list(c["mask"])] = 1.0
            self.comps.append({**c, "m": m})
        self.tril = np.tril_indices(n_fid)
        self.diag = self.tril[0] == self.tril[1]
        self.n_tri = len(self.tril[0])
        self.theta = None

    def n_params(self):
        return sum(len(c["dims"]) + 1 for c in self.comps) + self.n_tri + 2 * self.n_fid

    def unpack(self, th):
        i, out = 0, []
        for c in self.comps:
            d = len(c["dims"])
            out.append((np.exp(th[i:i + d]), np.exp(th[i + d])))
            i += d + 1
        tri = th[i:i + self.n_tri].copy()
        i += self.n_tri
        tri[self.diag] = np.exp(tri[self.diag])
        lb = np.zeros((self.n_fid, self.n_fid))
        lb[self.tril] = tri
        return out, lb, th[i:i + self.n_fid], np.exp(th[i + self.n_fid:i + 2 * self.n_fid])

    def ls_idx(self):
        idx, i = [], 0
        for c in self.comps:
            idx += list(range(i, i + len(c["dims"])))
            i += len(c["dims"]) + 1
        return np.array(idx, int)

    def default_theta(self, ls=0.5, rho=0.8, amp=1.0, noise=0.05):
        parts = []
        for c in self.comps:
            parts += [np.full(len(c["dims"]), np.log(ls)), [np.log(amp * c.get("amp0", 1.0))]]
        lb = np.linalg.cholesky((1 - rho) * np.eye(self.n_fid) + rho * np.ones((self.n_fid, self.n_fid)))
        tri = lb[self.tril].copy()
        tri[self.diag] = np.log(tri[self.diag])
        means = np.array([self.y_by_fid.get(s, 0.0) for s in range(self.n_fid)])
        return np.concatenate(parts + [tri, means, np.full(self.n_fid, np.log(noise))])

    def random_theta(self):
        th = self.default_theta(rho=self.rng.uniform(0.4, 0.95), amp=np.exp(self.rng.uniform(-0.7, 0.3)),
                                noise=np.exp(self.rng.uniform(np.log(0.02), np.log(0.2))))
        li = self.ls_idx()
        th[li] = self.rng.uniform(np.log(0.15), np.log(2.0), len(li))
        return th

    def bounds(self):
        b = []
        for c in self.comps:
            b += [(np.log(0.03), np.log(20.0))] * len(c["dims"]) + [(np.log(0.01), np.log(5.0))]
        b += [(np.log(0.05), np.log(5.0)) if m else (-5.0, 5.0) for m in self.diag]
        b += [(-6.0, 6.0)] * self.n_fid
        b += [(np.log(self.min_noise), np.log(1.0))] * self.n_fid
        return b

    @staticmethod
    def sqdist(a, b, ls):
        a, b = a / ls, b / ls
        return np.maximum(np.sum(a ** 2, 1)[:, None] + np.sum(b ** 2, 1)[None, :] - 2 * a @ b.T, 0.0)

    def comp_mats(self, x1, c1, s1, x2, c2, s2, th):
        cp, lb, _, _ = self.unpack(th)
        mats, same = [], None
        for comp, (ls, amp) in zip(self.comps, cp):
            d = comp["dims"]
            if len(d):
                k = amp ** 2 * np.exp(-0.5 * self.sqdist(x1[:, d], x2[:, d], ls))
            else:
                k = np.full((len(x1), len(x2)), amp ** 2)
            if comp["clone"]:
                if same is None:
                    same = (c1[:, None] == c2[None, :]).astype(float)
                k = k * same
            if comp.get("mask") is not None:
                k = k * comp["m"][s1][:, None] * comp["m"][s2][None, :]
            mats.append(k)
        return mats, lb @ lb.T

    def kern(self, x1, c1, s1, x2, c2, s2, th=None):
        mats, bm = self.comp_mats(x1, c1, s1, x2, c2, s2, self.theta if th is None else th)
        return bm[np.ix_(s1, s2)] * sum(mats)

    def prior_var(self, s):
        cp, lb, _, _ = self.unpack(self.theta)
        tot = np.zeros(len(s))
        for comp, (_, amp) in zip(self.comps, cp):
            tot += amp ** 2 * comp["m"][s]
        return np.sum(lb ** 2, 1)[s] * tot

    def nll(self, th):
        x, c, s, y = self.x, self.c, self.s, self.y
        n = len(y)
        cp, lb, means, noise = self.unpack(th)
        mats, bm = self.comp_mats(x, c, s, x, c, s, th)
        bss = bm[np.ix_(s, s)]
        R = sum(mats)
        ky = bss * R + np.diag(noise[s] ** 2 + JITTER)
        try:
            L = np.linalg.cholesky(ky)
        except np.linalg.LinAlgError:
            return 1e10, np.zeros_like(th)
        r = y - means[s]
        a = cho_solve((L, True), r)
        val = 0.5 * r @ a + np.sum(np.log(np.diag(L)))
        Linv = solve_triangular(L, np.eye(n), lower=True)
        W = np.outer(a, a) - Linv.T @ Linv
        WB = W * bss
        g = []
        for comp, (ls, amp), K in zip(self.comps, cp, mats):
            M = WB * K
            d = comp["dims"]
            if len(d):
                xs = x[:, d] / ls
                g.append(-0.5 * (2 * (xs ** 2).T @ M.sum(1) - 2 * np.sum((M @ xs) * xs, 0)))
            g.append([-np.sum(M)])
        oh = np.eye(self.n_fid)[s]
        gL = 2 * (-0.5 * oh.T @ (W * R) @ oh) @ lb
        gt = gL[self.tril]
        gt[self.diag] *= lb[self.tril][self.diag]
        g.append(gt)
        g.append(-np.bincount(s, weights=a, minlength=self.n_fid))
        g.append(-np.bincount(s, weights=np.diag(W), minlength=self.n_fid) * noise ** 2)
        grad = np.concatenate([np.atleast_1d(v) for v in g])
        li = self.ls_idx()
        mu0, sd0 = self.cfg["ls_prior"]
        val += 0.5 * np.sum((th[li] - mu0) ** 2) / sd0 ** 2
        grad[li] += (th[li] - mu0) / sd0 ** 2
        return val, grad

    def fit(self, x, c, s, y, theta0=None, restarts=2, maxiter=150):
        self.x, self.c, self.s = np.asarray(x, float), np.asarray(c, int), np.asarray(s, int)
        self.ymu, self.ysd = float(np.mean(y)), max(float(np.std(y)), 1e-6)
        self.y = (np.asarray(y, float) - self.ymu) / self.ysd
        self.y_by_fid = {f: float(self.y[self.s == f].mean()) for f in np.unique(self.s)}
        self.min_noise = self.cfg["min_noise"] / self.ysd
        bnds = self.bounds()
        lo, hi = np.array(bnds).T
        starts = []
        if theta0 is not None and len(theta0) == self.n_params():
            starts.append(theta0)
        if not starts or restarts > 0:
            starts.append(self.default_theta())
        starts += [self.random_theta() for _ in range(restarts)]
        best = None
        for st in starts:
            try:
                res = minimize(self.nll, np.clip(st, lo, hi), jac=True, method="L-BFGS-B", bounds=bnds,
                               options={"maxiter": maxiter})
            except Exception:  # pylint: disable=broad-exception-caught
                continue
            if best is None or res.fun < best.fun:
                best = res
        self.theta = best.x if best is not None else np.clip(self.default_theta(), lo, hi)
        self.cache()

    def cache(self):
        _, _, self.means, noise = self.unpack(self.theta)
        ky = self.kern(self.x, self.c, self.s, self.x, self.c, self.s) + np.diag(noise[self.s] ** 2 + JITTER)
        self.L = np.linalg.cholesky(ky)
        self.alpha = cho_solve((self.L, True), self.y - self.means[self.s])

    def predict(self, x, c, s):
        x = np.atleast_2d(x)
        c, s = np.asarray(c, int), np.asarray(s, int)
        ks = self.kern(self.x, self.c, self.s, x, c, s)
        mu = self.means[s] + ks.T @ self.alpha
        v = solve_triangular(self.L, ks, lower=True)
        return mu, np.sqrt(np.maximum(self.prior_var(s) - np.sum(v ** 2, 0), 1e-12))

    def add_fantasy(self, x, c, s, y):
        self.x = np.vstack([self.x, x])
        self.c = np.concatenate([self.c, c])
        self.s = np.concatenate([self.s, s])
        self.y = np.concatenate([self.y, y])
        self.cache()

    def snapshot(self):
        return self.x.copy(), self.c.copy(), self.s.copy(), self.y.copy()

    def restore(self, snap):
        self.x, self.c, self.s, self.y = snap
        self.cache()


# ------------------------------------------------------------------------------------------------ space
def ei(mu, sd, best):
    sd = np.maximum(sd, 1e-9)
    z = (mu - best) / sd
    return (mu - best) * norm.cdf(z) + sd * norm.pdf(z)


def sobol(n, d, rng):
    return (sobol_seq.i4_sobol_generate(d, n, 1) + rng.random(d)) % 1.0


class Space:
    """Structured recipe family in unit coordinates u = (t, p, g, a, b)."""

    D = 5

    def __init__(self, lab):
        self.lab = lab
        self.lo, self.hi = np.array(lab.bounds, float).T
        self.span = np.where(self.hi > self.lo, self.hi - self.lo, 1.0)
        self.clones = list(lab.celltypes)
        self.nc = len(self.clones)
        self.fids = list(lab.reactor_types)
        self.pilot, self.plate = lab.objective_reactor, lab.screening_reactor
        mids = [r for r in self.fids if r not in (self.pilot, self.plate)]
        self.mbr = mids[0] if mids else None
        self.f_pilot, self.f_plate = self.fids.index(self.pilot), self.fids.index(self.plate)
        self.f_mbr = self.fids.index(self.mbr) if self.mbr else None

    def raw(self, u):
        t, p, g, a, b = np.clip(u, 0, 1)
        lo, sp = self.lo, self.span
        T, P = lo[0] + t * sp[0], lo[1] + p * sp[1]
        r = [T, P, lo[2] + g * sp[2], lo[3] + a * sp[3], lo[4] + g * sp[4], lo[5] + a * sp[5],
             lo[6] + g * sp[6], lo[7] + b * sp[7], T, P, T, P]
        return np.clip(np.array(r, float), self.lo, self.hi)

    def encode(self, row, project):
        """Row -> (u, ok). Plates: (t, p, glc0, gln0, 0). Bioreactors in the family map exactly; others are
        projected (time-weighted T/pH, mean glucose, mean early glutamine) when `project`."""
        v = np.array([float(x) for x in row[2:]])
        w = (v - self.lo) / self.span
        if row[0] == self.plate:
            return np.array([w[0], w[1], w[2], w[3], 0.0]), True
        u = np.array([w[0], w[1], w[2], w[3], w[7]])
        if np.max(np.abs(self.raw(u) - v) / self.span) <= 0.02:
            return u, True
        if not project:
            return None, False
        t = (120 * w[0] + 120 * w[8] + 96 * w[10]) / 336
        p = (120 * w[1] + 120 * w[9] + 96 * w[11]) / 336
        return np.array([t, p, (w[2] + w[4] + w[6]) / 3, (w[3] + w[5]) / 2, w[7]]), True

    def bio_rows(self, reactor, clones, units):
        out = np.empty((len(clones), 14), dtype=object)
        out[:, 0] = reactor
        for i, (c, u) in enumerate(zip(clones, units)):
            out[i, 1] = self.clones[int(c)]
            out[i, 2:] = [float(x) for x in self.raw(u)]
        return out

    def plate_rows(self, clones, t, p, g, a):
        lo, sp = self.lo, self.span
        return self.lab.plate([self.clones[int(c)] for c in clones], float(lo[0] + t * sp[0]),
                              float(lo[1] + p * sp[1]), float(lo[2] + g * sp[2]), float(lo[3] + a * sp[3]))


# ------------------------------------------------------------------------------------------------ model
class Model:
    def __init__(self, sp, cfg, rng):
        self.sp, self.cfg = sp, cfg
        bio = [f for f in range(len(sp.fids)) if f != sp.f_plate]
        comps = [{"dims": [d], "clone": False, "amp0": 0.5} for d in range(4)]
        comps.append({"dims": [4], "clone": False, "mask": bio, "amp0": 0.5})
        comps.append({"dims": [0, 1], "clone": True, "amp0": 0.7})
        comps.append({"dims": [], "clone": True, "amp0": 0.7})
        if cfg["interaction"]:
            comps.append({"dims": [0, 1, 2, 3, 4], "clone": False, "mask": bio, "amp0": 0.2})
        self.gp = MFGP(len(sp.fids), comps, rng, cfg)
        self.theta = None
        self.n_fits = 0

    def fit(self, X, C, S, Y, time_ok):
        y = np.maximum(np.asarray(Y, float), 0.0)
        z = np.log(y + self.cfg["log_eps"] * max(float(y.max()), 1e-9))
        full = self.theta is None or self.n_fits % self.cfg["refit_every"] == 0
        if time_ok > 1:
            restarts = self.cfg["restarts"] if full else 0
            maxiter = self.cfg["maxiter"] if full else self.cfg["warm_maxiter"]
        elif time_ok > 0:
            restarts, maxiter = 0, 15
        else:
            restarts, maxiter = 0, 0
        if maxiter == 0 and self.theta is not None:
            self.gp.x, self.gp.c, self.gp.s = np.asarray(X, float), np.asarray(C, int), np.asarray(S, int)
            self.gp.ymu, self.gp.ysd = float(np.mean(z)), max(float(np.std(z)), 1e-6)
            self.gp.y = (z - self.gp.ymu) / self.gp.ysd
            self.gp.cache()
        else:
            self.gp.fit(X, C, S, z, self.theta, restarts, maxiter)
        self.theta = self.gp.theta
        self.n_fits += 1

    def predict(self, U, C, fid):
        return self.gp.predict(U, C, np.full(len(C), fid))


# ------------------------------------------------------------------------------------------------ acquisition
def pilot_incumbent(model, sp):
    gp = model.gp
    m = gp.s == sp.f_pilot
    if np.any(m):
        return float(np.max(gp.y[m]))
    mu, _ = model.predict(gp.x, gp.c, sp.f_pilot)
    return float(np.max(mu))


def centres(model, sp, clones=None, k=4):
    gp = model.gp
    bio = np.where(gp.s != sp.f_plate)[0]
    if clones is not None:
        bio = np.array([i for i in bio if int(gp.c[i]) in clones], int)
    if not len(bio):
        return []
    mu, _ = model.predict(gp.x[bio], gp.c[bio], sp.f_pilot)
    return [(gp.x[i].copy(), int(gp.c[i])) for i in bio[np.argsort(-mu)[:k]]]


def box(cfg, D):
    b = np.array(cfg["box"] if cfg["box"] is not None else [(0.0, 1.0)] * D, float)
    return b[:, 0], b[:, 1]


def maximise(model, sp, rng, cfg, fn, allowed, cen, penal=None):
    """Maximise fn(U, C) over allowed clones: Sobol pool on the most promising clones + local search + L-BFGS."""
    allowed = np.array(sorted(allowed), int)
    blo, bhi = box(cfg, sp.D)
    pool = blo + sobol(cfg["pool"], sp.D, rng) * (bhi - blo)
    if cfg["snap"] > 0:
        ps = pool.copy()
        ps[:, 2:] = np.where(ps[:, 2:] < cfg["snap"], 0.0, np.where(ps[:, 2:] > 1 - cfg["snap"], 1.0, ps[:, 2:]))
        pool = np.vstack([pool, ps])
    coarse = pool[:128]
    v = fn(np.tile(coarse, (len(allowed), 1)), np.repeat(allowed, len(coarse))).reshape(len(allowed), -1).max(1)
    top = list(allowed[np.argsort(-v)[:cfg["top_clones"]]])
    U, C = [np.tile(pool, (len(top), 1))], [np.repeat(top, len(pool))]
    for u, c in cen:
        if c in allowed:
            for sc in (0.03, 0.08, 0.2):
                U.append(np.clip(u + rng.normal(0, sc, (24, sp.D)), blo, bhi))
                C.append(np.full(24, c))
            U.append(np.clip(u, blo, bhi)[None, :])
            C.append(np.array([c]))
    U, C = np.vstack(U), np.concatenate(C).astype(int)

    def val(u, c):
        x = fn(u, c)
        return penal(u, c, x) if penal is not None else x

    vals = val(U, C)
    order = np.argsort(-vals)
    best = (int(C[order[0]]), U[order[0]].copy(), float(vals[order[0]]))
    done = 0
    for i in order[:cfg["n_refine"] * 4]:
        if done >= cfg["n_refine"]:
            break
        done += 1
        cc = int(C[i])
        res = minimize(lambda u: -float(val(u[None, :], np.array([cc]))[0]), U[i], method="L-BFGS-B",
                       bounds=list(zip(blo, bhi)), options={"maxiter": 30})
        if -res.fun > best[2]:
            best = (cc, np.clip(res.x, blo, bhi), float(-res.fun))
    return best


def dist_penalty(model, sp, fid, dmin=0.03):
    gp = model.gp
    done = gp.s == fid
    X, Cd = gp.x[done], gp.c[done]

    def pen(u, c, v):
        if not len(X):
            return v
        d = np.sqrt(((u[:, None, :] - X[None, :, :]) ** 2).sum(-1))
        d = np.where(np.asarray(c)[:, None] == Cd[None, :], d, np.inf).min(1)
        return np.where(d < dmin, v - 1e3, v)

    return pen


def bio_seen(model, sp):
    gp = model.gp
    return set(int(c) for c in gp.c[gp.s != sp.f_plate])


def choose_pilot(model, sp, rng, cfg):
    n_p = int(np.sum(model.gp.s == sp.f_pilot))
    acq = cfg["first_pilot_acq"] if n_p == 0 else cfg["pilot_acq"]
    allowed = bio_seen(model, sp) if cfg["pilot_bio_seen"] else set()
    if not allowed:
        allowed = set(range(sp.nc))
    inc = pilot_incumbent(model, sp)

    def fn(u, c):
        mu, sd = model.predict(u, c, sp.f_pilot)
        if acq == "mean":
            return mu
        if acq == "ei":
            return ei(mu, sd, inc)
        return mu + cfg["beta"] * sd

    size = sp.lab.batch_sizes[sp.pilot]
    snap = model.gp.snapshot()
    picks = []
    for _ in range(size):
        c, u, _ = maximise(model, sp, rng, cfg, fn, allowed, centres(model, sp, allowed),
                           dist_penalty(model, sp, sp.f_pilot))
        picks.append((c, u))
        if size > 1:
            mu, _ = model.predict(u[None, :], [c], sp.f_pilot)
            model.gp.add_fantasy(u[None, :], np.array([c]), np.array([sp.f_pilot]), mu)
    if size > 1:
        model.gp.restore(snap)
    return sp.bio_rows(sp.pilot, [p[0] for p in picks], [p[1] for p in picks])


def choose_mbr(model, sp, rng, cfg):
    size = sp.lab.batch_sizes[sp.mbr]
    snap = model.gp.snapshot()
    inc = pilot_incumbent(model, sp)
    picks, used = [], set()
    for _ in range(size):
        allowed = set(range(sp.nc)) - used if cfg["mbr_distinct"] and len(used) < sp.nc else set(range(sp.nc))

        def fn(u, c, inc=inc):
            mu, sd = model.predict(u, c, sp.f_pilot)
            return ei(mu, sd, inc)

        c, u, _ = maximise(model, sp, rng, cfg, fn, allowed, centres(model, sp),
                           dist_penalty(model, sp, sp.f_mbr))
        picks.append((c, u))
        used.add(c)
        fs = [sp.f_mbr, sp.f_pilot]
        mu, _ = model.gp.predict(np.tile(u, (2, 1)), [c, c], fs)
        model.gp.add_fantasy(np.tile(u, (2, 1)), np.array([c, c]), np.array(fs), mu)
        inc = max(inc, float(mu[-1]))
    model.gp.restore(snap)
    return sp.bio_rows(sp.mbr, [p[0] for p in picks], [p[1] for p in picks])


def choose_plate(model, sp, rng, cfg, n_plates_done):
    size = sp.lab.batch_sizes[sp.plate]
    med = cfg["plate_medium"]
    if model is None:
        tp = sobol(8, 2, np.random.default_rng(int(sp.lab.seed) + 11))[n_plates_done % 8]
        clones = [i % sp.nc for i in range(size)]
        g, a = med if med is not None else (0.5, 0.5)
        return sp.plate_rows(clones, tp[0], tp[1], g, a)
    conds = sobol(256, 4, rng)
    if med is not None:
        conds[:, 2], conds[:, 3] = med
    k = min(size, sp.nc)
    # value of each clone at a condition: EI of its plate titer over the clone's own best predicted plate titer
    grid = sobol(256, 2, rng)
    G = np.zeros((len(grid), 5))
    G[:, :2] = grid
    G[:, 2], G[:, 3] = (med if med is not None else (0.5, 0.5))
    mu_g, _ = model.predict(np.repeat(G, sp.nc, axis=0), np.tile(np.arange(sp.nc), len(G)), sp.f_plate)
    clone_best = mu_g.reshape(len(G), sp.nc).max(0)
    # weight clones by their promise at Pilot (UCB of best predicted Pilot titer)
    blo, bhi = box(cfg, sp.D)
    P = blo + sobol(128, 5, rng) * (bhi - blo)
    mu_p, sd_p = model.predict(np.repeat(P, sp.nc, axis=0), np.tile(np.arange(sp.nc), len(P)), sp.f_pilot)
    promise = (mu_p + 2 * sd_p).reshape(len(P), sp.nc).max(0)
    w = np.exp(np.clip((promise - promise.max()) / max(np.std(promise), 1e-6), -10, 0))
    U = np.zeros((len(conds), 5))
    U[:, :4] = conds
    mu, sd = model.predict(np.repeat(U, sp.nc, axis=0), np.tile(np.arange(sp.nc), len(U)), sp.f_plate)
    val = ei(mu, sd, np.tile(clone_best, len(U))).reshape(len(U), sp.nc) * w[None, :]
    j = int(np.argmax(-np.sort(-val, axis=1)[:, :k].sum(1)))
    clones = list(np.argsort(-val[j])[:k])
    while len(clones) < size:
        clones.append(clones[len(clones) % k])
    return sp.plate_rows(clones, *conds[j])


# ------------------------------------------------------------------------------------------------ scale plan
def keeps_budget_usable(lab, reactor):
    pilot = lab.objective_reactor
    if reactor == pilot:
        return True
    budget_after = lab.remaining_budget - lab.batch_cost(reactor)
    exp_after = lab.remaining_experiments - lab.batch_sizes[reactor]
    pil = lab.batch_sizes[pilot]
    return exp_after >= pil and (budget_after // lab.batch_cost(pilot)) * pil <= exp_after


def cheap_ok_after(lab, first, second, money, reserve):
    """Whether `second` still fits in this segment's money after running `first`."""
    return money - lab.batch_cost(first) - lab.batch_cost(second) >= reserve


def next_scale(lab, sp, cfg, plan):
    pc = lab.batch_cost(sp.pilot)
    ckpts = sorted(lab.score_checkpoints)
    upcoming = [c for c in ckpts if c >= lab.spent + 1e-9]
    seg = min(len(ckpts) - len(upcoming), len(ckpts) - 1) if upcoming else len(ckpts) - 1
    seg_end = min(upcoming[0], lab.budget) if upcoming else lab.budget
    money = seg_end - lab.spent
    last = len(upcoming) <= 1
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
            if key == "plate" and cfg["mbr_before_plates"] and st["mbr"] > 0 and r is not None \
                    and not cheap_ok_after(lab, r, sp.mbr, money, reserve):
                continue    # this plate would squeeze out a planned mini-bioreactor batch
            st[key] -= 1
            return r
    if lab.can_run(sp.pilot) and money >= pc:
        st["pilot"] = max(st["pilot"] - 1, 0)
        return sp.pilot
    if last or money < pc:
        for r in (sp.mbr, sp.plate):
            if r and lab.can_run(r) and keeps_budget_usable(lab, r) and money >= lab.batch_cost(r):
                return r
    if lab.can_run(sp.pilot):
        return sp.pilot
    rs = lab.affordable_reactors()
    return rs[-1] if rs else None


# ------------------------------------------------------------------------------------------------ main
def my_algorithm(initial_x, initial_y, lab, seed, cfg=None):
    """Structured multi-fidelity funnel; see module docstring."""
    cfg = dict(CFG if cfg is None else cfg)
    rng = np.random.default_rng(seed)
    sp = Space(lab)
    rows = list(initial_x[:len(initial_y)])
    ys = [float(v) for v in initial_y]
    model = Model(sp, cfg, rng)
    plan = {}
    n_plates = 0
    last_iter = 0.0
    while lab.affordable_reactors():
        reactor = next_scale(lab, sp, cfg, plan)
        if reactor is None:
            break
        t0 = lab.elapsed
        left, tl = lab.time_remaining, max(lab.time_limit, 1.0)
        time_ok = 2 if left > 0.4 * tl else (1 if left > max(0.1 * tl, 3 * last_iter) else 0)
        X, C, S, Y = [], [], [], []
        for r, y in zip(rows, ys):
            u, ok = sp.encode(r, cfg["project_nonconforming"])
            if ok:
                X.append(u); C.append(sp.clones.index(r[1])); S.append(sp.fids.index(r[0])); Y.append(y)
        have_model = len(Y) >= 6
        if have_model:
            model.fit(np.array(X), np.array(C), np.array(S), np.array(Y), time_ok)
        if reactor == sp.plate:
            batch = choose_plate(model if have_model else None, sp, rng, cfg, n_plates)
            n_plates += 1
        elif not have_model:
            blo, bhi = box(cfg, sp.D)
            batch = sp.bio_rows(reactor, rng.integers(sp.nc, size=lab.batch_sizes[reactor]),
                                blo + sobol(lab.batch_sizes[reactor], sp.D, rng) * (bhi - blo))
        elif reactor == sp.pilot:
            batch = choose_pilot(model, sp, rng, cfg)
        else:
            batch = choose_mbr(model, sp, rng, cfg)
        last_iter = max(lab.elapsed - t0, 0.0)
        y, _ = lab(batch)
        if len(y) == 0:
            break
        rows += list(batch[:len(y)])
        ys += [float(v) for v in y]


CFG.update({'box': [[0, 1], [0, 1], [1, 1], [0, 0.12], [0.15, 0.85]]})
