"""Multi-fidelity Bayesian optimisation for the digital cell-culture lab.

Model: one Gaussian process over all three scales (plate, mini bioreactor, Pilot) on sqrt(titer) with an additive
kernel K = B[s,s'] * ( k_shared(T,pH,glc0,gln0) + delta(c,c') k_clone(T,pH,glc0,gln0) + k_feed(all 6 dims; bioreactors) ),
B a learned 3x3 scale covariance, per-scale mean and noise, hyperparameters by maximum marginal likelihood.
Recipes: 6-D (T, pH, starting glc/gln, feed glc/gln used for both feeds, no shifts); a plate is the same space
with no feed. Budget is split between scoring checkpoints: cheap plates / MBR first, then Pilot runs.
"""

import numpy as np
import sobol_seq
from scipy.linalg import cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.stats import norm



JITTER = 1e-6


def ei(mu, sd, best):
    sd = np.maximum(sd, 1e-9)
    z = (mu - best) / sd
    return (mu - best) * norm.cdf(z) + sd * norm.pdf(z)


def sobol(n, d, rng):
    return (sobol_seq.i4_sobol_generate(d, n, 1) + rng.random(d)) % 1.0


class AdditiveMFGP:
    """K = B[s,s'] * sum_k A_k^2 * mask_k(s) mask_k(s') * [delta(c,c') if clone_k] * RBF_ARD(x[dims_k], x'[dims_k]).

    Per-scale constant mean and noise. Inputs are unit-scaled. theta packs, per component: log ls (len dims), log amp;
    then tri(L_B), means, log noise.
    """

    def __init__(self, n_fid, comps, seed=0, restarts=2, ls_prior=(np.log(0.5), 1.5)):
        self.n_fid = n_fid
        self.comps = [dict(c) for c in comps]  # dims, clone (bool), mask (list of fidelities or None)
        for c in self.comps:
            m = np.ones(n_fid)
            if c.get("mask") is not None:
                m = np.zeros(n_fid); m[list(c["mask"])] = 1.0
            c["m"] = m
        self.rng = np.random.default_rng(seed)
        self.restarts = restarts
        self.tril = np.tril_indices(n_fid)
        self.diag_mask = self.tril[0] == self.tril[1]
        self.n_tri = len(self.tril[0])
        self.theta = None
        self.y_floor = 0.0
        self.ls_prior = ls_prior
        self.min_noise_log = 0.01
        self.ls_max = 20.0
        self.maxiter = 150
        self.skip_opt = False
        self.matern = False
        self.transform = "log"

    def n_params(self):
        return sum(len(c["dims"]) + 1 for c in self.comps) + self.n_tri + 2 * self.n_fid

    def unpack(self, th):
        i = 0
        out = []
        for c in self.comps:
            d = len(c["dims"])
            out.append((np.exp(th[i:i + d]), np.exp(th[i + d])))
            i += d + 1
        tri = th[i:i + self.n_tri].copy(); i += self.n_tri
        tri[self.diag_mask] = np.exp(tri[self.diag_mask])
        lb = np.zeros((self.n_fid, self.n_fid)); lb[self.tril] = tri
        means = th[i:i + self.n_fid]; i += self.n_fid
        noise = np.exp(th[i:i + self.n_fid])
        return out, lb, means, noise

    def ls_idx(self):
        idx, i = [], 0
        for c in self.comps:
            idx += list(range(i, i + len(c["dims"])))
            i += len(c["dims"]) + 1
        return np.array(idx)

    def default_theta(self, ls=0.5, rho=0.8, amp=1.0, noise=0.05):
        parts = []
        for c in self.comps:
            parts += [np.full(len(c["dims"]), np.log(ls)), [np.log(amp * c.get("amp0", 1.0))]]
        bm = (1 - rho) * np.eye(self.n_fid) + rho * np.ones((self.n_fid, self.n_fid))
        lb = np.linalg.cholesky(bm)
        tri = lb[self.tril].copy(); tri[self.diag_mask] = np.log(tri[self.diag_mask])
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
            b += [(np.log(0.03), np.log(self.ls_max))] * len(c["dims"]) + [(np.log(0.02), np.log(5.0))]
        b += [(np.log(0.05), np.log(5.0)) if m else (-5.0, 5.0) for m in self.diag_mask]
        b += [(-6.0, 6.0)] * self.n_fid
        b += [(np.log(self.min_noise), np.log(1.0))] * self.n_fid
        return b

    @staticmethod
    def sqdist(a, b, ls):
        a, b = a / ls, b / ls
        return np.maximum(np.sum(a**2, 1)[:, None] + np.sum(b**2, 1)[None, :] - 2 * a @ b.T, 0.0)

    def comp_mats(self, x1, c1, s1, x2, c2, s2, th, with_grad=False):
        cp, lb, _, _ = self.unpack(th)
        mats, gmats = [], []
        same = None
        for comp, (ls, amp) in zip(self.comps, cp):
            d = comp["dims"]
            sq = self.sqdist(x1[:, d], x2[:, d], ls)
            if self.matern:
                r5 = np.sqrt(5.0 * sq)
                e = np.exp(-r5)
                k = amp**2 * (1 + r5 + 5.0 * sq / 3.0) * e
                g = amp**2 * (5.0 / 3.0) * (1 + r5) * e
            else:
                k = amp**2 * np.exp(-0.5 * sq)
                g = k
            mult = 1.0
            if comp["clone"]:
                if same is None:
                    same = (c1[:, None] == c2[None, :]).astype(float)
                mult = same
            if comp.get("mask") is not None:
                mult = mult * comp["m"][s1][:, None] * comp["m"][s2][None, :]
            mats.append(k * mult)
            if with_grad:
                gmats.append(g * mult)
        if with_grad:
            return mats, gmats, lb @ lb.T
        return mats, lb @ lb.T

    def kern(self, x1, c1, s1, x2, c2, s2, th=None):
        th = self.theta if th is None else th
        mats, bm = self.comp_mats(x1, c1, s1, x2, c2, s2, th)
        return bm[np.ix_(s1, s2)] * sum(mats)

    def prior_var(self, x, c, s):
        cp, lb, _, _ = self.unpack(self.theta)
        bd = np.sum(lb**2, 1)[s]
        tot = np.zeros(len(s))
        for comp, (_, amp) in zip(self.comps, cp):
            tot += amp**2 * comp["m"][s]
        return bd * tot

    def nll(self, th):
        x, c, s, y = self.x, self.c, self.s, self.y
        n = len(y)
        cp, lb, means, noise = self.unpack(th)
        mats, gmats, bm = self.comp_mats(x, c, s, x, c, s, th, with_grad=True)
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
        W = np.outer(a, a) - cho_solve((L, True), np.eye(n))
        WB = W * bss
        g = []
        for comp, (ls, amp), K, G in zip(self.comps, cp, mats, gmats):
            M = WB * G
            xs = x[:, comp["dims"]] / ls
            g.append(-0.5 * (2 * (xs**2).T @ M.sum(1) - 2 * np.sum((M @ xs) * xs, 0)))
            g.append([-np.sum(WB * K)])
        oh = np.eye(self.n_fid)[s]
        gB = -0.5 * oh.T @ (W * R) @ oh
        gL = 2 * gB @ lb
        gt = gL[self.tril]
        gt[self.diag_mask] *= lb[self.tril][self.diag_mask]
        g.append(gt)
        g.append(-np.bincount(s, weights=a, minlength=self.n_fid))
        g.append(-np.bincount(s, weights=np.diag(W), minlength=self.n_fid) * noise**2)
        grad = np.concatenate([np.atleast_1d(v) for v in g])
        li = self.ls_idx()
        mu0, sd0 = self.ls_prior
        val += 0.5 * np.sum((th[li] - mu0) ** 2) / sd0**2
        grad[li] += (th[li] - mu0) / sd0**2
        return val, grad

    def fit(self, x, c, s, y_titer, theta0=None):
        self.x = np.asarray(x, float); self.c = np.asarray(c, int); self.s = np.asarray(s, int)
        yt = np.maximum(np.asarray(y_titer, float), 0.0)
        if self.y_floor > 0:
            for f in np.unique(self.s):
                m = self.s == f
                yt[m] = np.maximum(yt[m], self.y_floor * yt[m].max())
        if self.transform == "log0":
            ly = np.log(np.maximum(yt, 1e-9))
        elif self.transform == "sqrt":
            ly = np.sqrt(yt)
        elif self.transform == "lin":
            ly = yt.copy()
        else:
            ly = np.log1p(yt)
        self.ymu, self.ysd = ly.mean(), max(ly.std(), 1e-6)
        self.y = (ly - self.ymu) / self.ysd
        self.y_by_fid = {f: float(self.y[self.s == f].mean()) for f in np.unique(self.s)}
        self.min_noise = self.min_noise_log / self.ysd
        bnds = self.bounds()
        lo, hi = np.array(bnds).T
        if self.skip_opt and theta0 is not None and len(theta0) == self.n_params():
            self.theta = np.clip(theta0, lo, hi)
            self.cache()
            return self
        starts = []
        if theta0 is not None and len(theta0) == self.n_params():
            starts.append(theta0)
        starts.append(self.default_theta())
        starts += [self.random_theta() for _ in range(self.restarts)]
        best = None
        for st in starts:
            st = np.clip(st, lo, hi)
            try:
                res = minimize(self.nll, st, jac=True, method="L-BFGS-B", bounds=bnds, options={"maxiter": self.maxiter})
            except Exception:  # pylint: disable=broad-exception-caught
                continue
            if best is None or res.fun < best.fun:
                best = res
        self.theta = best.x if best is not None else np.clip(self.default_theta(), lo, hi)
        self.cache()
        return self

    def cache(self):
        _, _, means, noise = self.unpack(self.theta)
        self.means, self.noise = means, noise
        ky = self.kern(self.x, self.c, self.s, self.x, self.c, self.s) + np.diag(noise[self.s] ** 2 + JITTER)
        self.L = np.linalg.cholesky(ky)
        self.alpha = cho_solve((self.L, True), self.y - means[self.s])

    def add_fantasy(self, x, c, s, y_model):
        self.x = np.vstack([self.x, x]); self.c = np.concatenate([self.c, c]); self.s = np.concatenate([self.s, s])
        self.y = np.concatenate([self.y, y_model])
        self.cache()

    def snapshot(self):
        return (self.x.copy(), self.c.copy(), self.s.copy(), self.y.copy())

    def restore(self, snap):
        self.x, self.c, self.s, self.y = snap
        self.cache()

    def predict(self, x, c, s):
        x = np.atleast_2d(x); c = np.asarray(c, int); s = np.asarray(s, int)
        ks = self.kern(self.x, self.c, self.s, x, c, s)
        mu = self.means[s] + ks.T @ self.alpha
        v = solve_triangular(self.L, ks, lower=True)
        return mu, np.maximum(self.prior_var(x, c, s) - np.sum(v**2, 0), 1e-12)


def joint_sample(gp, x, c, s, rng, n_samples=1):
    """Joint posterior samples of f at the given points (model space), shape (n_samples, n)."""
    ks = gp.kern(gp.x, gp.c, gp.s, x, c, s)
    mu = gp.means[np.asarray(s)] + ks.T @ gp.alpha
    v = solve_triangular(gp.L, ks, lower=True)
    cov = gp.kern(x, c, s, x, c, s) - v.T @ v
    cov = 0.5 * (cov + cov.T) + 1e-8 * np.eye(len(mu))
    try:
        lc = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        w, vec = np.linalg.eigh(cov)
        lc = vec * np.sqrt(np.maximum(w, 0))
    return mu[None, :] + rng.standard_normal((n_samples, len(mu))) @ lc.T


STUDENT_NAMES = ["TODO Firstname Lastname CID", "TODO Firstname Lastname CID"]


CFG = {
    "pilots_per_segment": [1, 2, 99],
    "plates_per_segment": [4, 1, 1],
    "mbr_per_segment": [1, 1, 1],
    "order": ["plate", "mbr"],
    "n_space_fill_plates": 0,
    "plate_acq": "pilot_ei",     # pilot_ei | plate_ei | plate_ucb
    "mbr_acq": "pilot_ei",       # pilot_ei | mbr_ei
    "pilot_acq": "ucb",          # ei | mean | ucb
    "first_pilot_acq": "mean",
    "ucb_beta": 0.5,
    "clone_dims": [0, 1, 2, 3],
    "y_floor": 0.0,
    "restarts": 2,
    "pool": 384,
    "n_refine": 5,
    "clone_topk": 30,
    "tie_tol": 0.02,
    "tr": 0.05,
    "fantasy_pilot": True,
    "plate_medium": "exploit",
    "medium_jitter": 0.08,
    "min_dist": 0.03,
    "ls_max": 20.0,
    "matern": False,
    "transform": "sqrt",
    "snap": 0.0,
    "blocks": None,
    "maximin": False,
    "min_data": 8,
    "plate_scale_nc": 0.0,
    "guard_frac": [0.5, 0.3, 0.15],    # below these fractions of the time limit: no restarts / short fits / reuse fit
    "block_clones": 3,
    "block_centres": 2,
    "peak": False,
    "ts_pool": 200,
    "ts_clones": 8,
    "ts_plate_conds": 64,
    "clone_feed": False,
    "plate_beta": 3.0,
    "sf_medium": None,
    "sf_medium_frac": None,
    "pilot_clones": "all",           # all | bio_seen | pilot_seen  (restrict Pilot recipes to tested clones)
    "pilot_clones_skip_first": False,
}


class Space:
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

    def raw12(self, u6):
        """Tied 12-D raw settings for a 6-D unit vector (T, pH, glc0, gln0, glc feed, gln feed)."""
        u6 = np.clip(u6, 0, 1)
        t, p, g, q = self.lo[:4] + u6[:4] * self.span[:4]
        gf, qf = self.lo[4:6] + u6[4:6] * self.span[4:6]
        raw = np.array([t, p, g, q, gf, qf, gf, qf, t, p, t, p], float)
        return np.clip(raw, self.lo, self.hi)

    def u6_of(self, v):
        return np.concatenate([(v[:4] - self.lo[:4]) / self.span[:4], (v[4:6] - self.lo[4:6]) / self.span[4:6]])

    def is_tied(self, v, tol):
        return np.all(np.abs(self.raw12(self.u6_of(v)) - v) / self.span <= tol)

    def encode(self, rows, ys, tol):
        U, C, S, Y = [], [], [], []
        for r, y in zip(rows, ys):
            v = np.array([float(x) for x in r[2:]])
            u = self.u6_of(v)
            if r[0] == self.plate:
                u[4:6] = 0.0
            elif not self.is_tied(v, tol):
                continue
            U.append(u); C.append(self.clones.index(r[1])); S.append(self.fids.index(r[0])); Y.append(y)
        return np.array(U), np.array(C, int), np.array(S, int), np.array(Y, float)

    def bio_rows(self, reactor, clones, units):
        out = np.empty((len(clones), 2 + len(self.lo)), dtype=object)
        out[:, 0] = reactor
        for i, (c, u) in enumerate(zip(clones, units)):
            out[i, 1] = self.clones[int(c)]
            out[i, 2:] = [float(v) for v in self.raw12(u)]
        return out

    def plate_rows(self, clones, u4):
        raw = self.lo[:4] + np.clip(u4, 0, 1) * self.span[:4]
        return self.lab.plate([self.clones[int(c)] for c in clones], *[float(v) for v in raw])


def acq_values(gp, U, C, fid, acq, inc, beta=1.0):
    mu, var = gp.predict(U, C, np.full(len(C), fid))
    sd = np.sqrt(var)
    if acq == "ei":
        return ei(mu, sd, inc)
    if acq == "ucb":
        return mu + beta * sd
    return mu


def best_point(gp, sp, rng, cfg, fid, acq, inc, clones, centres=(), beta=1.0, block=None):
    if block is not None and len(centres):
        U, C = [], []
        nb = max(cfg["pool"] // len(centres), 32)
        for u, c in centres:
            for cc in set([int(c)]) | (set(int(x) for x in clones[:cfg["block_clones"]]) if 0 in block else set()):
                base = np.tile(u, (nb, 1))
                base[:, block] = sobol(nb, len(block), rng)
                loc = np.tile(u, (16, 1))
                loc[:, block] = np.clip(u[block] + rng.normal(0, cfg["tr"], (16, len(block))), 0, 1)
                U.append(np.vstack([base, loc])); C.append(np.full(nb + 16, cc))
        U = np.vstack(U); C = np.concatenate(C).astype(int)
        fixed = np.setdiff1d(np.arange(6), block)
    else:
        fixed = None
        pool = sobol(cfg["pool"], 6, rng)
        U = [np.tile(pool, (len(clones), 1))]
        C = [np.repeat(clones, len(pool))]
        for u, c in centres:
            pert = np.clip(u + rng.normal(0, cfg["tr"], (16, 6)), 0, 1)
            for cc in set([int(c)]) | set(int(x) for x in clones[:5]):
                U.append(np.vstack([u, pert])); C.append(np.full(17, cc))
        U = np.vstack(U); C = np.concatenate(C)
    if cfg["snap"] > 0:
        Us = U.copy()
        nd = Us[:, 2:6]
        nd[nd < cfg["snap"]] = 0.0
        nd[nd > 1 - cfg["snap"]] = 1.0
        Us[:, 2:6] = nd
        U = np.vstack([U, Us]); C = np.concatenate([C, C])
    done = gp.s == fid

    def penal(u, c, v):
        if not np.any(done) or cfg["min_dist"] <= 0:
            return v
        X, Cd = gp.x[done], gp.c[done]
        d = np.sqrt(((u[:, None, :] - X[None, :, :]) ** 2).sum(-1))
        d = np.where(c[:, None] == Cd[None, :], d, np.inf).min(1)
        return np.where(d < cfg["min_dist"], -1e9, v)

    vals = penal(U, C, acq_values(gp, U, C, fid, acq, inc, beta))
    order = np.argsort(-vals)
    best = (int(C[order[0]]), U[order[0]].copy(), float(vals[order[0]]))
    for i in order[:cfg["n_refine"]]:
        cc = int(C[i])
        bnds = [(0, 1)] * 6
        if fixed is not None:
            bnds = [(float(U[i][d]), float(U[i][d])) if d in fixed else (0, 1) for d in range(6)]
        res = minimize(lambda u: -float(penal(u[None, :], np.array([cc]),
                                              acq_values(gp, u[None, :], np.array([cc]), fid, acq, inc, beta))[0]), U[i],
                       method="L-BFGS-B", bounds=bnds, options={"maxiter": 50})
        if -res.fun > best[2]:
            best = (cc, np.clip(res.x, 0, 1), float(-res.fun))
    return best


def pilot_incumbent(gp, fid_p):
    m = gp.s == fid_p
    if np.any(m):
        return float(np.max(gp.y[m]))
    mu, _ = gp.predict(gp.x, gp.c, np.full(len(gp.c), fid_p))
    return float(np.max(mu))


def best_feed(gp, sp):
    bio = np.where(gp.s != sp.fids.index(sp.plate))[0]
    if not len(bio):
        return np.full(2, 0.5)
    mu, _ = gp.predict(gp.x[bio], gp.c[bio], np.full(len(bio), sp.fids.index(sp.pilot)))
    return gp.x[bio[np.argmax(mu)], 4:6]


def choose_plate(gp, sp, rng, cfg, n_own, counts, best_medium=None, prev_tph=None):
    lab = sp.lab
    size = lab.batch_sizes[sp.plate]
    if gp is None:
        pts = sobol(16, 4, np.random.default_rng(int(lab.seed) + 7))
        u4 = pts[n_own % 16].copy()
        if cfg["maximin"] and prev_tph is not None and len(prev_tph):
            cand = sobol(512, 2, rng)
            d = np.sqrt(((cand[:, None, :] - prev_tph[None, :, :]) ** 2).sum(-1)).min(1)
            u4[:2] = cand[int(np.argmax(d))]
        if cfg["sf_medium"] == "best" and best_medium is not None:
            u4[2:] = np.clip(best_medium + rng.normal(0, 0.05, 2), 0, 1)
        order = np.argsort(counts + rng.random(sp.nc) * 0.1)
        return sp.plate_rows([order[i % sp.nc] for i in range(size)], u4)
    fid_plate, fid_p = sp.fids.index(sp.plate), sp.fids.index(sp.pilot)
    conds = sobol(256, 4, rng)
    if cfg["plate_medium"] == "exploit":
        pool = sobol(256, 4, rng)
        Up = np.hstack([np.repeat(pool, sp.nc, axis=0), np.zeros((256 * sp.nc, 2))])
        mu, _ = gp.predict(Up, np.tile(np.arange(sp.nc), 256), np.full(256 * sp.nc, fid_plate))
        med = pool[int(np.argmax(mu.reshape(256, sp.nc).max(1))), 2:4]
        conds[:, 2:4] = np.clip(med + rng.normal(0, cfg["medium_jitter"], (256, 2)), 0, 1)
    U4 = np.repeat(conds, sp.nc, axis=0)
    C = np.tile(np.arange(sp.nc), len(conds))
    if cfg["plate_acq"] == "ts":
        nconds = cfg["ts_plate_conds"]
        cs = conds[:nconds]
        U = np.hstack([np.repeat(cs, sp.nc, axis=0), np.zeros((nconds * sp.nc, 2))])
        Cc = np.tile(np.arange(sp.nc), nconds)
        f = joint_sample(gp, U, Cc, np.full(len(Cc), fid_plate), rng, 1)[0].reshape(nconds, sp.nc)
        j = int(np.argmax(f.max(1)))
        k = min(size, sp.nc)
        clones = list(np.argsort(-f[j])[:k])
        while len(clones) < size:
            clones.append(clones[len(clones) % k])
        return sp.plate_rows(clones, cs[j])
    if cfg["plate_acq"] == "pilot_ei":
        U = np.hstack([U4, np.tile(best_feed(gp, sp), (len(U4), 1))])
        val = acq_values(gp, U, C, fid_p, "ei", pilot_incumbent(gp, fid_p))
    else:
        U = np.hstack([U4, np.zeros((len(U4), 2))])
        if cfg["plate_acq"] == "plate_ucb":
            val = acq_values(gp, U, C, fid_plate, "ucb", 0.0, cfg["plate_beta"])
        else:
            inc = float(np.max(gp.y[gp.s == fid_plate])) if np.any(gp.s == fid_plate) else 0.0
            val = acq_values(gp, U, C, fid_plate, "ei", inc)
    val = val.reshape(len(conds), sp.nc)
    k = min(size, sp.nc)
    j = int(np.argmax(-np.sort(-val, axis=1)[:, :k].sum(1)))
    clones = list(np.argsort(-val[j])[:k])
    while len(clones) < size:
        clones.append(clones[len(clones) % k])
    return sp.plate_rows(clones, conds[j])


def choose_bio(gp, sp, rng, cfg, reactor, acq_name):
    fid_p = sp.fids.index(sp.pilot)
    fid = sp.fids.index(reactor)
    size = sp.lab.batch_sizes[reactor]
    inc = pilot_incumbent(gp, fid_p)
    bio = np.where(gp.s != sp.fids.index(sp.plate))[0]
    if len(bio):
        mu, _ = gp.predict(gp.x[bio], gp.c[bio], np.full(len(bio), fid_p))
        top = bio[np.argsort(-mu)[:4]]
        centres = [(gp.x[i], gp.c[i]) for i in top]
    else:
        centres = []
    if cfg["clone_topk"] < sp.nc:
        pool = sobol(256, 6, rng)
        U = np.repeat(pool, sp.nc, axis=0); C = np.tile(np.arange(sp.nc), len(pool))
        mu, _ = gp.predict(U, C, np.full(len(C), fid_p))
        clones = np.argsort(-mu.reshape(len(pool), sp.nc).max(0))[:cfg["clone_topk"]]
    else:
        clones = np.arange(sp.nc)
    if reactor == sp.pilot and cfg.get("pilot_clones", "all") != "all" and len(bio):
        first = not np.any(gp.s == fid_p)
        if not (first and cfg.get("pilot_clones_skip_first", False)):
            if cfg["pilot_clones"] == "bio_seen":
                allowed = set(int(v) for v in gp.c[bio])
            else:  # "pilot_seen": clones already run at Pilot (else bio-seen)
                pm = gp.s == fid_p
                allowed = set(int(v) for v in gp.c[pm]) if np.any(pm) else set(int(v) for v in gp.c[bio])
            kept = [int(c) for c in clones if int(c) in allowed]
            clones = np.array(kept if kept else sorted(allowed), int)
            centres = [(u, c) for u, c in centres if int(c) in allowed] or centres
    if size == 1:
        c, u, _ = best_point(gp, sp, rng, cfg, fid_p, acq_name, inc, clones, centres, cfg["ucb_beta"])
        return sp.bio_rows(reactor, [c], [u])
    if acq_name == "ts":
        pool = sobol(cfg["ts_pool"], 6, rng)
        U = [np.tile(pool, (len(clones[:cfg["ts_clones"]]), 1))]
        C = [np.repeat(clones[:cfg["ts_clones"]], len(pool))]
        for u, c in centres:
            U.append(np.clip(u + rng.normal(0, cfg["tr"], (40, 6)), 0, 1)); C.append(np.full(40, c))
            U.append(np.clip(u + rng.normal(0, 3 * cfg["tr"], (40, 6)), 0, 1)); C.append(np.full(40, c))
        U = np.vstack(U); C = np.concatenate(C).astype(int)
        f = joint_sample(gp, U, C, np.full(len(C), fid_p), rng, size)
        picks, used = [], set()
        for k in range(size):
            order = np.argsort(-f[k])
            j = next((i for i in order if i not in used), order[0])
            used.add(j); picks.append((C[j], U[j]))
        return sp.bio_rows(reactor, [p[0] for p in picks], [p[1] for p in picks])
    block = None
    if cfg["blocks"] and len(centres):
        n_batches = int(np.sum(gp.s == fid)) // size
        block = cfg["blocks"][n_batches % len(cfg["blocks"])]
        centres = centres[:cfg["block_centres"]]
    saved = gp.snapshot()
    picks = []
    for _ in range(size):
        if acq_name == "mbr_ei":
            m = gp.s == fid
            inc_m = float(np.max(gp.y[m])) if np.any(m) else inc
            c, u, _ = best_point(gp, sp, rng, cfg, fid, "ei", inc_m, clones, centres)
        else:
            c, u, _ = best_point(gp, sp, rng, cfg, fid_p, "ei", inc, clones, centres, block=block)
        picks.append((c, u))
        fs = [fid, fid_p] if cfg["fantasy_pilot"] else [fid]
        mu, _ = gp.predict(np.tile(u, (len(fs), 1)), [c] * len(fs), fs)
        gp.add_fantasy(np.tile(u, (len(fs), 1)), np.array([c] * len(fs)), np.array(fs), mu)
        if cfg["fantasy_pilot"]:
            inc = max(inc, float(mu[-1]))
    gp.restore(saved)
    return sp.bio_rows(reactor, [p[0] for p in picks], [p[1] for p in picks])


def next_ckpt(lab):
    up = [c for c in lab.score_checkpoints if c >= lab.spent]
    return min(up) if up else lab.budget


def fallback_bio(sp, rng, reactor, U, C, S, Y):
    """Bioreactor batch without a model: best observed (clone, conditions), else Sobol recipes over all clones."""
    size = sp.lab.batch_sizes[reactor]
    if len(Y):
        order = np.argsort(-Y)
        picks = []
        for k in range(size):
            i = order[k % len(order)]
            u = U[i].copy()
            if S[i] == sp.fids.index(sp.plate):
                u[4:6] = 0.5
            if k >= len(order):
                u = np.clip(u + rng.normal(0, 0.1, 6), 0, 1)
            picks.append((C[i], u))
    else:
        pts = sobol(size, 6, rng)
        picks = [(int(rng.integers(sp.nc)), pts[k]) for k in range(size)]
    return sp.bio_rows(reactor, [p[0] for p in picks], [p[1] for p in picks])


def keeps_budget_usable(lab, reactor):
    pilot = lab.objective_reactor
    if reactor == pilot:
        return True
    budget_after = lab.remaining_budget - lab.batch_cost(reactor)
    exp_after = lab.remaining_experiments - lab.batch_sizes[reactor]
    pil = lab.batch_sizes[pilot]
    return exp_after >= pil and (budget_after // lab.batch_cost(pilot)) * pil <= exp_after


def next_scale(lab, cfg, sp, plan):
    """Segment plan: between scoring checkpoints run the planned cheap batches first, then the Pilot runs."""
    pilot, pc = sp.pilot, lab.batch_cost(sp.pilot)
    ckpts = sorted(lab.score_checkpoints)
    nseg = len(cfg["pilots_per_segment"])
    upcoming = [c for c in ckpts if c >= lab.spent + 1e-9]
    seg = min(len(ckpts) - len(upcoming), nseg - 1) if upcoming else nseg - 1
    last = len(upcoming) <= 1
    seg_end = min(upcoming[0], lab.budget) if upcoming else lab.budget
    if seg not in plan:
        money = seg_end - lab.spent
        max_p = int(money // pc)
        cheap = cfg["mbr_per_segment"][seg] * (lab.batch_cost(sp.mbr) if sp.mbr else 0) \
            + cfg["plates_per_segment"][seg] * lab.batch_cost(sp.plate)
        want = cfg["pilots_per_segment"][seg]
        if last or want >= 99:
            want = max(min(max_p, int((money - cheap) // pc)), 1 if max_p else 0)
        else:
            want = min(max(want, 1), max_p)
        n_pl = cfg["plates_per_segment"][seg]
        if cfg["plate_scale_nc"] and n_pl > 0:
            n_pl = int(round(n_pl * max(1.0, sp.nc / (lab.batch_sizes[sp.plate] * cfg["plate_scale_nc"]))))
        plan[seg] = {"pilot": want, "mbr": cfg["mbr_per_segment"][seg], "plate": n_pl}
    st = plan[seg]
    money_left = seg_end - lab.spent
    reserve = st["pilot"] * pc

    def ok(r):
        return r is not None and lab.can_run(r) and keeps_budget_usable(lab, r) \
            and money_left - lab.batch_cost(r) >= reserve

    for key in cfg["order"]:
        r = sp.plate if key == "plate" else sp.mbr
        if st[key] > 0 and ok(r):
            return r, st, key
    if lab.can_run(pilot) and (st["pilot"] > 0 or money_left >= pc or last):
        return pilot, st, "pilot"
    for r in (sp.mbr, sp.plate):
        if r and lab.can_run(r) and keeps_budget_usable(lab, r) and money_left >= lab.batch_cost(r):
            return r, st, None
    if lab.can_run(pilot):
        return pilot, st, "pilot"
    rs = [r for r in lab.affordable_reactors()]
    return (rs[-1], st, None) if rs else (None, st, None)


def my_algorithm(initial_x, initial_y, lab, seed, cfg=None):
    cfg = dict(CFG if cfg is None else cfg)
    rng = np.random.default_rng(seed)
    sp = Space(lab)
    data_x = list(initial_x[:len(initial_y)])
    data_y = [float(v) for v in initial_y]
    fid_p = sp.fids.index(sp.pilot)
    bio_f = [i for i, r in enumerate(sp.fids) if r != sp.plate]
    comps = [{"dims": [0, 1, 2, 3], "clone": False}, {"dims": cfg["clone_dims"], "clone": True},
             {"dims": [0, 1, 2, 3, 4, 5], "clone": False, "mask": bio_f, "amp0": 0.5}]
    if cfg["clone_feed"]:
        comps.append({"dims": [4, 5], "clone": True, "mask": bio_f, "amp0": 0.3})
    gp = AdditiveMFGP(len(sp.fids), comps, seed=seed, restarts=cfg["restarts"])
    gp.y_floor = cfg["y_floor"]
    gp.ls_max = cfg["ls_max"]
    gp.matern = cfg["matern"]
    gp.transform = cfg["transform"]
    n_own_plates = 0
    theta = None
    n_plates = sum(1 for r in data_x if r[0] == sp.plate) // max(lab.batch_sizes[sp.plate], 1)
    plan = {}
    last_iter = 0.0
    base_pool = cfg["pool"]
    while lab.affordable_reactors():
        reactor, st, key = next_scale(lab, cfg, sp, plan)
        if reactor is None:
            break
        # time guard: effort scales down as the clock runs out (grading machines may be slower)
        t_iter0 = lab.elapsed
        left = lab.time_remaining
        tl = max(lab.time_limit, 1.0)
        g_in = gp.g if hasattr(gp, "g") else gp
        gp.restarts = cfg["restarts"] if left > cfg["guard_frac"][0] * tl else 0
        g_in.maxiter = 150 if left > cfg["guard_frac"][1] * tl else 30
        g_in.skip_opt = left < max(cfg["guard_frac"][2] * tl, 4 * last_iter)
        cfg["pool"] = base_pool if left > cfg["guard_frac"][1] * tl else max(64, base_pool // 4)
        U, C, S, Y = sp.encode(data_x, data_y, cfg["tie_tol"])
        model_needed = reactor != sp.plate or n_own_plates >= cfg["n_space_fill_plates"]
        enough = len(Y) >= cfg["min_data"] and np.any(S == sp.fids.index(sp.plate)) if len(Y) else False
        if not enough:
            if reactor != sp.plate and sp.plate in lab.affordable_reactors() and \
                    lab.spent + lab.batch_cost(sp.plate) + lab.batch_cost(sp.pilot) <= next_ckpt(lab):
                reactor, key = sp.plate, None   # no usable data yet: screen first
            model_needed = False
        if model_needed:
            gp.fit(U, C, S, Y, theta0=theta)
            theta = gp.theta
        if reactor == sp.plate:
            counts = np.bincount([sp.clones.index(r[1]) for r in data_x if r[0] == sp.plate], minlength=sp.nc)
            bm = None
            if len(Y):
                pl = S == sp.fids.index(sp.plate)
                if np.any(pl):
                    bm = U[pl][np.argmax(Y[pl]), 2:4]
            pt = U[S == sp.fids.index(sp.plate)][:, :2] if len(U) else None
            batch = choose_plate(gp if model_needed else None, sp, rng, cfg, n_own_plates, counts, bm, pt)
            n_own_plates += 1
        elif not model_needed:
            batch = fallback_bio(sp, rng, reactor, U, C, S, Y)
        elif reactor == sp.pilot:
            acq = cfg["pilot_acq"] if np.any(S == fid_p) else cfg["first_pilot_acq"]
            batch = choose_bio(gp, sp, rng, cfg, reactor, acq)
        else:
            batch = choose_bio(gp, sp, rng, cfg, reactor, cfg["mbr_acq"])
        if key:
            st[key] = max(st[key] - 1, 0)
        last_iter = max(lab.elapsed - t_iter0, 0.0)
        y, _ = lab(batch)
        if len(y) == 0:
            break
        data_x += list(batch[:len(y)])
        data_y += [float(v) for v in y]
        if batch[0][0] == sp.plate:
            n_plates += 1


CFG.update({'clone_dims': [0, 1]})


CFG.update({'plates_per_segment': [6, 1, 1]})
