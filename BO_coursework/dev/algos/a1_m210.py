"""a1: multi-fidelity GP BO over (clone, full 12-D recipe), checkpoint-segmented scale plan.

Model: log-titer GP, K = B[s,s'] * (k_shared(all 12 dims) + delta_cc' k_clone(T/pH dims) + delta_cc' a^2), per-scale
mean and noise, ML-II hyperparameters (analytic gradients). Scale plan: in each scoring segment, cheap batches first,
then Pilot run(s) just before the checkpoint. Pilot: maximise posterior UCB. MBR: batch Pilot-EI (kriging believer).
Plates: condition + clones maximising summed Pilot-EI proxy, space-filling before a model exists.
"""

import numpy as np
import sobol_seq
from scipy.linalg import cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.stats import norm

STUDENT_NAMES = ["TODO Firstname Lastname CID", "TODO Firstname Lastname CID"]

CFG = {
    "transform": "log",          # log | sqrt
    "log_eps": 0.01,             # log(y + eps * max(y))
    "tph_dims": [0, 1, 8, 9, 10, 11],
    "clone_offset": True,
    "pilot_dev": False,          # extra Pilot-only component (all dims)
    "restarts": 2,
    "maxiter": 150,
    "warm_maxiter": 40,
    "refit_every": 4,
    "pilot_frac": [0.0, 0.5, 1.0],   # fraction of segment money for Pilot runs (min 1)
    "plates_per_seg": [2, 0, 0],
    "mbr_per_seg": [99, 99, 99],     # max MBR batches per segment (rest of the money goes to Pilot)
    "mbr_first": True,
    "pilot_acq": "ucb",
    "first_pilot_acq": "mean",
    "beta": 1.0,
    "mbr_acq": "pilot_ei",       # pilot_ei | mbr_ucb
    "pool": 384,
    "top_clones": 8,
    "n_local": 64,
    "n_refine": 4,
    "min_data_bio": 4,
    "ls_prior": (np.log(0.5), 1.5),
    "min_noise": 0.01,
    "tied": False,               # search only recipes with feed2 = feed1 and no T/pH shifts
}

JITTER = 1e-6


# ----------------------------------------------------------------------------------------------- GP
class MFGP:
    """Additive multi-fidelity GP. comps: list of dict(dims, clone, mask)."""

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
        means = th[i:i + self.n_fid]
        noise = np.exp(th[i + self.n_fid:i + 2 * self.n_fid])
        return out, lb, means, noise

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
        bm = (1 - rho) * np.eye(self.n_fid) + rho * np.ones((self.n_fid, self.n_fid))
        lb = np.linalg.cholesky(bm)
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
            b += [(np.log(0.03), np.log(20.0))] * len(c["dims"]) + [(np.log(0.02), np.log(5.0))]
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
            k = amp ** 2 * np.exp(-0.5 * self.sqdist(x1[:, d], x2[:, d], ls)) if len(d) else \
                np.full((len(x1), len(x2)), amp ** 2)
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
        W = np.outer(a, a) - cho_solve((L, True), np.eye(n))
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
        gB = -0.5 * oh.T @ (W * R) @ oh
        gL = 2 * gB @ lb
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
        return self

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


# ----------------------------------------------------------------------------------------------- helpers
def ei(mu, sd, best):
    sd = np.maximum(sd, 1e-9)
    z = (mu - best) / sd
    return (mu - best) * norm.cdf(z) + sd * norm.pdf(z)


def sobol(n, d, rng):
    return (sobol_seq.i4_sobol_generate(d, n, 1) + rng.random(d)) % 1.0


class Space:
    def __init__(self, lab):
        self.lab = lab
        self.lo, self.hi = np.array(lab.bounds, float).T
        self.span = np.where(self.hi > self.lo, self.hi - self.lo, 1.0)
        self.d = len(self.lo)
        self.clones = list(lab.celltypes)
        self.nc = len(self.clones)
        self.fids = list(lab.reactor_types)
        self.pilot, self.plate = lab.objective_reactor, lab.screening_reactor
        mids = [r for r in self.fids if r not in (self.pilot, self.plate)]
        self.mbr = mids[0] if mids else None
        self.f_pilot = self.fids.index(self.pilot)
        self.f_plate = self.fids.index(self.plate)
        self.f_mbr = self.fids.index(self.mbr) if self.mbr else None

    def to_unit(self, row):
        return (np.array([float(v) for v in row[2:]]) - self.lo) / self.span

    def to_raw(self, u):
        return np.clip(self.lo + np.clip(u, 0, 1) * self.span, self.lo, self.hi)

    def bio_rows(self, reactor, clones, units):
        out = np.empty((len(clones), 2 + self.d), dtype=object)
        out[:, 0] = reactor
        for i, (c, u) in enumerate(zip(clones, units)):
            out[i, 1] = self.clones[int(c)]
            out[i, 2:] = [float(v) for v in self.to_raw(u)]
        return out

    def plate_unit(self, u4):
        """Full 12-D unit vector a plate well with shared settings u4 corresponds to."""
        raw = np.zeros(self.d)
        raw[:4] = self.lo[:4] + np.clip(u4, 0, 1) * self.span[:4]
        raw[4:8] = 0.0
        raw[8], raw[9], raw[10], raw[11] = raw[0], raw[1], raw[0], raw[1]
        return (raw - self.lo) / self.span

    def plate_rows(self, clones, u4):
        raw = self.lo[:4] + np.clip(u4, 0, 1) * self.span[:4]
        return self.lab.plate([self.clones[int(c)] for c in clones], *[float(v) for v in raw])


# ----------------------------------------------------------------------------------------------- data/model
class Model:
    def __init__(self, sp, cfg, rng):
        self.sp, self.cfg, self.rng = sp, cfg, rng
        comps = [{"dims": list(range(sp.d)), "clone": False},
                 {"dims": [d for d in cfg["tph_dims"] if d < sp.d], "clone": True, "amp0": 0.7}]
        if cfg["clone_offset"]:
            comps.append({"dims": [], "clone": True, "amp0": 0.7})
        if cfg["pilot_dev"]:
            comps.append({"dims": list(range(sp.d)), "clone": False, "mask": [sp.f_pilot], "amp0": 0.3})
        self.gp = MFGP(len(sp.fids), comps, rng, cfg)
        self.theta = None

    def transform(self, y):
        y = np.maximum(np.asarray(y, float), 0.0)
        if self.cfg["transform"] == "sqrt":
            return np.sqrt(y)
        self.eps = self.cfg["log_eps"] * max(float(np.max(y)), 1e-9)
        return np.log(y + self.eps)

    def fit(self, X, C, S, Y, restarts, maxiter):
        self.gp.fit(X, C, S, self.transform(Y), self.theta, restarts, maxiter)
        self.theta = self.gp.theta

    def predict(self, X, C, fid):
        mu, sd = self.gp.predict(X, C, np.full(len(C), fid))
        return mu, sd


# ----------------------------------------------------------------------------------------------- acquisition
def tie(u, cfg):
    """Project unit recipes onto the tied subspace (feed 2 = feed 1, constant T/pH) if cfg['tied']."""
    if not cfg["tied"] or u.shape[-1] != 12:
        return u
    u = np.array(u, float, copy=True)
    u[..., 6:8] = u[..., 4:6]
    u[..., 8] = u[..., 0]
    u[..., 10] = u[..., 0]
    u[..., 9] = u[..., 1]
    u[..., 11] = u[..., 1]
    return u


def candidate_pool(model, sp, rng, cfg, fid, centres):
    """Candidates over top clones: Sobol pool + local perturbations around centres (unit x, clone)."""
    gp = model.gp
    pool = sobol(cfg["pool"], sp.d, rng)
    U = np.repeat(pool, sp.nc, axis=0)
    C = np.tile(np.arange(sp.nc), len(pool))
    mu, sd = model.predict(U, C, fid)
    clone_score = (mu + sd).reshape(len(pool), sp.nc).max(0)
    top = list(np.argsort(-clone_score)[:cfg["top_clones"]])
    for _, c in centres:
        if int(c) not in top:
            top.append(int(c))
    Ul = [np.tile(pool, (len(top), 1))]
    Cl = [np.repeat(top, len(pool))]
    for u, c in centres:
        for scale in (0.03, 0.1, 0.25):
            pert = np.clip(u + rng.normal(0, scale, (cfg["n_local"] // 3, sp.d)), 0, 1)
            Ul.append(np.vstack([u[None, :], pert]))
            Cl.append(np.full(len(pert) + 1, int(c)))
    return np.vstack(Ul), np.concatenate(Cl).astype(int)


def centres_from_data(model, sp, k=4):
    gp = model.gp
    bio = np.where(gp.s != sp.f_plate)[0]
    if not len(bio):
        return []
    mu, _ = model.predict(gp.x[bio], gp.c[bio], sp.f_pilot)
    top = bio[np.argsort(-mu)[:k]]
    return [(gp.x[i].copy(), int(gp.c[i])) for i in top]


def maximise(model, sp, rng, cfg, fid, acq_fn, centres, penal=None):
    U, C = candidate_pool(model, sp, rng, cfg, fid, centres)

    def val(u, c):
        u = tie(u, cfg)
        v = acq_fn(u, c)
        return penal(u, c, v) if penal is not None else v

    v = val(U, C)
    order = np.argsort(-v)
    best = (int(C[order[0]]), U[order[0]].copy(), float(v[order[0]]))
    seen = set()
    for i in order[:cfg["n_refine"] * 3]:
        if len(seen) >= cfg["n_refine"]:
            break
        key = (int(C[i]), tuple(np.round(U[i], 2)))
        if key in seen:
            continue
        seen.add(key)
        cc = int(C[i])
        res = minimize(lambda u: -float(val(u[None, :], np.array([cc]))[0]), U[i], method="L-BFGS-B",
                       bounds=[(0, 1)] * sp.d, options={"maxiter": 40})
        if -res.fun > best[2]:
            best = (cc, np.clip(res.x, 0, 1), float(-res.fun))
    return best[0], tie(best[1][None, :], cfg)[0], best[2]


def min_dist_penalty(model, sp, fid, dmin=0.02):
    gp = model.gp
    done = gp.s == fid

    def pen(u, c, v):
        if not np.any(done):
            return v
        X, Cd = gp.x[done], gp.c[done]
        d = np.sqrt(((u[:, None, :] - X[None, :, :]) ** 2).sum(-1))
        d = np.where(np.asarray(c)[:, None] == Cd[None, :], d, np.inf).min(1)
        return np.where(d < dmin, v - 1e3, v)

    return pen


def pilot_incumbent(model, sp):
    gp = model.gp
    m = gp.s == sp.f_pilot
    if np.any(m):
        return float(np.max(gp.y[m]))
    mu, _ = model.predict(gp.x, gp.c, sp.f_pilot)
    return float(np.max(mu))


def choose_pilot(model, sp, rng, cfg, n_pilots_done):
    acq = cfg["first_pilot_acq"] if n_pilots_done == 0 else cfg["pilot_acq"]
    beta = cfg["beta"]

    def fn(u, c):
        mu, sd = model.predict(u, c, sp.f_pilot)
        if acq == "mean":
            return mu
        if acq == "ei":
            return ei(mu, sd, pilot_incumbent(model, sp))
        return mu + beta * sd

    size = sp.lab.batch_sizes[sp.pilot]
    centres = centres_from_data(model, sp)
    snap = model.gp.snapshot()
    picks = []
    for _ in range(size):
        c, u, _ = maximise(model, sp, rng, cfg, sp.f_pilot, fn, centres, min_dist_penalty(model, sp, sp.f_pilot))
        picks.append((c, u))
        if size > 1:
            mu, _ = model.predict(u[None, :], [c], sp.f_pilot)
            model.gp.add_fantasy(u[None, :], np.array([c]), np.array([sp.f_pilot]), mu)
    if size > 1:
        model.gp.restore(snap)
    return sp.bio_rows(sp.pilot, [p[0] for p in picks], [p[1] for p in picks])


def choose_mbr(model, sp, rng, cfg):
    size = sp.lab.batch_sizes[sp.mbr]
    centres = centres_from_data(model, sp)
    snap = model.gp.snapshot()
    inc = pilot_incumbent(model, sp)
    picks = []
    for _ in range(size):
        if cfg["mbr_acq"] == "mbr_ucb":
            def fn(u, c):
                mu, sd = model.predict(u, c, sp.f_mbr)
                return mu + 2.0 * sd
        else:
            def fn(u, c, inc=inc):
                mu, sd = model.predict(u, c, sp.f_pilot)
                return ei(mu, sd, inc)
        c, u, _ = maximise(model, sp, rng, cfg, sp.f_pilot, fn, centres, min_dist_penalty(model, sp, sp.f_mbr))
        picks.append((c, u))
        fs = [sp.f_mbr, sp.f_pilot]
        mu, _ = model.gp.predict(np.tile(u, (2, 1)), [c, c], fs)
        model.gp.add_fantasy(np.tile(u, (2, 1)), np.array([c, c]), np.array(fs), mu)
        inc = max(inc, float(mu[-1]))
    model.gp.restore(snap)
    return sp.bio_rows(sp.mbr, [p[0] for p in picks], [p[1] for p in picks])


def choose_plate(model, sp, rng, cfg, plate_count, counts):
    size = sp.lab.batch_sizes[sp.plate]
    if model is None:
        pts = sobol(16, 4, np.random.default_rng(int(sp.lab.seed) + 7))
        u4 = pts[plate_count % 16]
        order = np.argsort(counts + rng.random(sp.nc) * 0.1)
        return sp.plate_rows([order[i % sp.nc] for i in range(size)], u4)
    conds = sobol(128, 4, rng)
    U = np.array([sp.plate_unit(u4) for u4 in conds])
    UU = np.repeat(U, sp.nc, axis=0)
    C = np.tile(np.arange(sp.nc), len(conds))
    mu, sd = model.predict(UU, C, sp.f_plate)
    val = (mu + 2.0 * sd).reshape(len(conds), sp.nc)
    k = min(size, sp.nc)
    j = int(np.argmax(-np.sort(-val, axis=1)[:, :k].sum(1)))
    clones = list(np.argsort(-val[j])[:k])
    while len(clones) < size:
        clones.append(clones[len(clones) % k])
    return sp.plate_rows(clones, conds[j])


# ----------------------------------------------------------------------------------------------- scale plan
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
    if seg not in plan:
        max_p = int(money // pc)
        frac = cfg["pilot_frac"][min(seg, len(cfg["pilot_frac"]) - 1)]
        want = int(frac * money // pc)
        want = min(max(want, 1), max_p)
        plan[seg] = {"pilot": want, "plate": cfg["plates_per_seg"][min(seg, len(cfg["plates_per_seg"]) - 1)],
                     "mbr": cfg["mbr_per_seg"][min(seg, len(cfg["mbr_per_seg"]) - 1)]}
    st = plan[seg]
    reserve = st["pilot"] * pc

    def cheap_ok(r):
        return r is not None and lab.can_run(r) and keeps_budget_usable(lab, r) and money - lab.batch_cost(r) >= reserve

    if st["plate"] > 0 and cheap_ok(sp.plate):
        st["plate"] -= 1
        return sp.plate
    if st["mbr"] > 0 and cheap_ok(sp.mbr):
        st["mbr"] -= 1
        return sp.mbr
    if lab.can_run(sp.pilot) and (st["pilot"] > 0 or last):
        st["pilot"] = max(st["pilot"] - 1, 0)
        return sp.pilot
    if lab.can_run(sp.pilot) and money >= pc:
        return sp.pilot
    for r in (sp.mbr, sp.plate):
        if r and lab.can_run(r) and keeps_budget_usable(lab, r) and money >= lab.batch_cost(r):
            return r
    if lab.can_run(sp.pilot):
        return sp.pilot
    rs = lab.affordable_reactors()
    return rs[-1] if rs else None


# ----------------------------------------------------------------------------------------------- main
def my_algorithm(initial_x, initial_y, lab, seed, cfg=None):
    cfg = dict(CFG if cfg is None else cfg)
    rng = np.random.default_rng(seed)
    sp = Space(lab)
    rows = list(initial_x[:len(initial_y)])
    ys = [float(v) for v in initial_y]
    model = Model(sp, cfg, rng)
    plan = {}
    plate_count = 0
    n_fits = 0
    last_iter = 0.0
    while lab.affordable_reactors():
        reactor = next_scale(lab, sp, cfg, plan)
        if reactor is None:
            break
        t0 = lab.elapsed
        left, tl = lab.time_remaining, max(lab.time_limit, 1.0)
        full = model.theta is None or n_fits % cfg["refit_every"] == 0
        restarts = cfg["restarts"] if (full and left > 0.5 * tl) else 0
        maxiter = cfg["maxiter"] if (full and left > 0.3 * tl) else cfg["warm_maxiter"]
        X = np.array([sp.to_unit(r) for r in rows]) if rows else np.zeros((0, sp.d))
        C = np.array([sp.clones.index(r[1]) for r in rows], int)
        S = np.array([sp.fids.index(r[0]) for r in rows], int)
        Y = np.array(ys)
        have_model = len(Y) >= 6
        if have_model and left > max(0.05 * tl, 3 * last_iter):
            model.fit(X, C, S, Y, restarts, maxiter)
            n_fits += 1
        elif have_model and model.theta is not None:
            model.gp.fit(X, C, S, model.transform(Y), model.theta, 0, 5)
        else:
            have_model = False
        if reactor == sp.plate:
            counts = np.bincount(C[S == sp.f_plate], minlength=sp.nc) if len(C) else np.zeros(sp.nc)
            batch = choose_plate(model if have_model else None, sp, rng, cfg, plate_count, counts)
            plate_count += 1
        elif not have_model:
            batch = fallback_bio(sp, rng, reactor, X, C, S, Y)
        elif reactor == sp.pilot:
            batch = choose_pilot(model, sp, rng, cfg, int(np.sum(S == sp.f_pilot)))
        else:
            batch = choose_mbr(model, sp, rng, cfg)
        last_iter = max(lab.elapsed - t0, 0.0)
        y, _ = lab(batch)
        if len(y) == 0:
            break
        rows += list(batch[:len(y)])
        ys += [float(v) for v in y]


def fallback_bio(sp, rng, reactor, X, C, S, Y):
    size = sp.lab.batch_sizes[reactor]
    if len(Y):
        order = np.argsort(-Y)
        picks = []
        for k in range(size):
            i = order[k % len(order)]
            u = X[i].copy()
            if S[i] == sp.f_plate:
                u[4:8] = 0.5
            if k >= len(order) or S[i] == sp.f_plate:
                u = np.clip(u + rng.normal(0, 0.05, sp.d), 0, 1)
            picks.append((C[i], u))
    else:
        pts = sobol(size, sp.d, rng)
        picks = [(int(rng.integers(sp.nc)), pts[k]) for k in range(size)]
    return sp.bio_rows(reactor, [p[0] for p in picks], [p[1] for p in picks])


CFG.update({'mbr_per_seg': [2, 1, 0]})
