"""Informed screening funnel: Bayesian T/pH localisation with value-of-information plate design, clone racing, and
a simulated glutamine-response library (transfer BO).

Plates (MTP) -> mini bioreactors (MBR) -> Pilot. All recipe coordinates are fractions of `lab.bounds`, and every
setting (budget, costs, batch sizes, checkpoints, clones, bounds, experiment limit) is read from `lab`.

1. Plates at a rich medium (glucose at its upper bound, little glutamine). Plate titer around a clone's optimum
   follows a fixed temperature/pH template (asymmetric quadratic in each, additive, saturating at a floor; widths
   TMPL_W from an offline simulator study). For each clone a grid posterior over its optimum is computed (uniform
   prior, height profiled out, heavy-tailed residuals). The operating T/pH is the Bayes action: the point that
   maximises the posterior-expected relative titer under a wider (bioreactor-scale) template. Clones are ranked by
   their posterior-mean peak height. The first two plates are space-filling; every later plate condition maximises
   the expected value of information, i.e. the expected gain in that decision utility summed over the top-ranked
   clones (cf. knowledge-gradient / decision-theoretic experimental design). With fewer than three of our own
   plates, the clone's best observed plate condition is used instead.
2. Mini bioreactors screen the best plate clones at their estimated optimum with one common prior recipe (glucose at
   its upper bound in every feed; glutamine start / feed 1 / feed 2 from the portfolio's first entry). Clones are
   screened in order of plate height plus 0.5 x log of the expected relative titer at the chosen T/pH, so clones
   whose optimum is still uncertain wait for more plate data. Within one run all measurements share one noise
   factor, so screened clones compare directly.
3. Pilot runs race the clones on measured titers. The best screened clone gets the first Pilot run. A clone whose MBR
   titer beats the Pilot incumbent's gets one Pilot run, and after that Pilot titers decide. The incumbent's
   glutamine levels are then searched because the titer optimum is a sharp, clone-specific cusp in log glutamine.
   A library of simulated Pilot glutamine-response surfaces (pre-computed offline for many simulated clones,
   embedded below in compressed form) acts as the prior. Its members are first weighted by how close their plate
   titer at the standard plate medium is to the clone's (a clone's absolute plate titer predicts how much
   glutamine it needs; the clone's value is the best standard-medium plate titer corrected by the template penalty
   at its posterior mode, correction capped). A clone's first Pilot run uses the recipe with the best weighted
   mean; from its 2nd run on, its Pilot titers re-weight the members (likelihood with a free log offset per member;
   cf. ranking-weighted GP ensembles / pre-trained GP priors in transfer-learning BO) and the next recipe maximises
   the mixture's expected improvement. A Gaussian process on log titer (temperature/pH offsets from the
   clone optimum, log10 glutamine at start and feed 1, glutamine at feed 2, clone offsets, correlated MBR/Pilot
   scales) ranks clones and is the fallback when the library has no usable candidate.
4. The money between score checkpoints is planned so that each reachable checkpoint gets a Pilot run, preceded by
   plates and an MBR screen when affordable. Money that cannot buy a Pilot run before a checkpoint is planned with
   the next segment; plates a segment planned but could not afford are carried into the next one. Model refits stop when less than `time_guard` seconds remain.
"""

import numpy as np
import sobol_seq
from scipy.linalg import cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.stats import norm

STUDENT_NAMES = ["TODO Firstname Lastname CID", "TODO Firstname Lastname CID"]

CFG = {
    'carry_plates': 9,
    'feat_s': 0.5,
    'feat_first': True,
    'feat_lib': True,
    'feat_h': 'cap',
    'cap_pen': 2.0,
    'time_guard': 20.0,
    'g0': 1.0,
    'la0': -1.7959,
    'la1': -2.699,
    'b0': 0.6,
    'la_box': (-3.3, -0.5),
    'b_box': (0.1, 1.0),
    'log_sd_mult': 4.0,
    'plate_medium': (1.0, 0.05),
    'n_space_plates': 2,
    'plates_per_seg': [5, 2, 0],
    'mbr_per_seg': [2, 2, 1],
    'pilot_frac': [0.0, 0.5, 1.0],
    'order': ['plate', 'mbr'],
    'mbr_before_plates': True,
    'mbr_refine_slots': [0, 0, 0],
    'pilot_acq': 'ei',
    'beta': 0.3,
    'pilot_top': 2,
    'first_pilot': 'prior',
    'pilot_rank': 'observed',
    'pilot_tp_fixed': True,
    'pilot_gp': 'all',
    'local_sd': (0.03, 0.08, 0.15),
    'n_local': 48,
    'restarts': 1,
    'maxiter': 100,
    'min_noise': 0.01,
    'log_eps': 0.01,
    'advance_unreachable': True,
    'blind_copy': False,
    'min_fit_plates': 3,
    'plan_by_stage': True,
    'fallback_plates': True,
    'portfolio': [(-1.7959, -2.699, 0.6), (-1.5229, -1.699, 0.85), (-2.0969, -2.699, 0.85)],
    'pilot_policy': 'race',
    'switch': 1.0,
    'seg_advance': True,
    'use_lib': True,
    'lib_sig': 0.1,
    'lib_until': 99,
    'lib_from': 1,
    'lib_acq': 'ei',
    'bayes_sig': 0.35,
    'bayes_wscale': 1.3,
    'voi_k': 10,
    'screen_eu': 0.5,
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
# T/pH response template of plate titer around a clone's optimum (simulator study: log titer falls as an asymmetric
# quadratic in temperature and in pH, additively, saturating at a floor). Widths in absolute units (C, pH):
# below/above the temperature optimum, below/above the pH optimum.
TMPL_W = (1.25, 3.08, 0.65, 0.15)


def template_pen(dt, dp, w, fl):
    d = (dt / np.where(dt < 0, w[0], w[1])) ** 2 + (dp / np.where(dp < 0, w[2], w[3])) ** 2
    return fl * (1 - np.exp(-d / fl))


def bayes_locations(T, P, C, Y, nc, w, sig=0.35, fl=5.0, ng=31, na=21, wscale=1.3, lo=-0.05, hi=1.05):
    """For every clone with plate data: grid posterior over its optimum (uniform prior on the box, height profiled
    out), then the operating point maximising the expected relative titer under a Pilot-scale template (widths x
    wscale). Returns dict with t, p (unit), map t/p and the posterior spread."""
    ly = np.log(np.maximum(Y, 1e-4 * max(Y.max(), 1e-12)))
    g = np.linspace(lo, hi, ng)
    GT, GP = np.meshgrid(g, g, indexing="ij"); GT, GP = GT.ravel(), GP.ravel()
    a = np.linspace(0.0, 1.0, na)
    AT, AP = np.meshgrid(a, a, indexing="ij"); AT, AP = AT.ravel(), AP.ravel()
    wp = tuple(x * wscale for x in w)
    # utility of action (rows) given optimum (cols): relative Pilot titer
    U = np.exp(-template_pen(AT[:, None] - GT[None, :], AP[:, None] - GP[None, :], wp, fl))
    t_out = np.full(nc, 0.5); p_out = np.full(nc, 0.5); spread = np.full(nc, 1.0)
    tm = np.full(nc, 0.5); pm = np.full(nc, 0.5)
    hgt = np.full(nc, float(np.mean(ly)))
    eu_best = np.full(nc, 0.5)
    for c in np.unique(C):
        m = C == c
        pen = template_pen(T[m][None, :] - GT[:, None], P[m][None, :] - GP[:, None], w, fl)  # grid x wells
        h = np.median(ly[m][None, :] + pen, axis=1)                                         # profiled height
        r = ly[m][None, :] - (h[:, None] - pen)
        nll = np.sum(np.log1p((r / sig) ** 2), axis=1)                                      # Cauchy-robust
        post = np.exp(-(nll - nll.min())); post /= post.sum()
        eu = U @ post
        k = int(np.argmax(eu))
        eu_best[c] = eu[k]
        t_out[c], p_out[c] = AT[k], AP[k]
        j = int(np.argmax(post)); tm[c], pm[c] = GT[j], GP[j]
        spread[c] = np.sqrt(post @ ((GT - post @ GT) ** 2 + (GP - post @ GP) ** 2))
        hgt[c] = post @ h
    return {"t": t_out, "p": p_out, "t_map": tm, "p_map": pm, "spread": spread, "height": hgt, "eu": eu_best}


def clone_posterior(t, p, ly, GT, GP, w, sig=0.35, fl=5.0):
    pen = template_pen(t[None, :] - GT[:, None], p[None, :] - GP[:, None], w, fl)
    h = np.median(ly[None, :] + pen, axis=1)
    r = ly[None, :] - (h[:, None] - pen)
    nll = np.sum(np.log1p((r / sig) ** 2), axis=1)
    post = np.exp(-(nll - nll.min()))
    return post / post.sum(), h


def voi_condition(T, P, C, Y, clones, weights, w, sig=0.35, fl=5.0, wscale=1.3, ng=15, na=11, nx=11, support=40,
                  lo=-0.05, hi=1.05):
    """Plate condition (unit t, p) maximising the summed value of information about the operating points of
    `clones` (weighted)."""
    ly = np.log(np.maximum(Y, 1e-4 * max(Y.max(), 1e-12)))
    g = np.linspace(lo, hi, ng)
    GT, GP = [v.ravel() for v in np.meshgrid(g, g, indexing="ij")]
    a = np.linspace(0, 1, na)
    AT, AP = [v.ravel() for v in np.meshgrid(a, a, indexing="ij")]
    x = np.linspace(0, 1, nx)
    XT, XP = [v.ravel() for v in np.meshgrid(x, x, indexing="ij")]
    wp = tuple(v * wscale for v in w)
    U = np.exp(-template_pen(AT[:, None] - GT[None, :], AP[:, None] - GP[None, :], wp, fl))   # A x G
    total = np.zeros(len(XT))
    for c, wt in zip(clones, weights):
        m = C == c
        if not np.any(m):
            post, h = np.full(len(GT), 1.0 / len(GT)), np.zeros(len(GT))
        else:
            post, h = clone_posterior(T[m], P[m], ly[m], GT, GP, w, sig, fl)
        base = np.max(U @ post)
        sup = np.argsort(-post)[:support]
        ps = post[sup] / post[sup].sum()
        # predicted log titer at each candidate x under each hypothesis g: (X, G)
        mx = h[None, :] - template_pen(XT[:, None] - GT[None, :], XP[:, None] - GP[None, :], w, fl)
        for i in range(len(XT)):
            d = (mx[i][None, :] - mx[i][sup][:, None]) / sig                     # support x G
            L = post[None, :] / (1 + d ** 2)
            L /= L.sum(1, keepdims=True)
            V = np.max(L @ U.T, axis=1)                                         # support
            total[i] += wt * (ps @ V - base)
    k = int(np.argmax(total))
    return float(XT[k]), float(XP[k]), total[k]


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
        self.best_cond_cache = None

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
        # optimum: grid posterior per clone under the fixed T/pH template; operating point = Bayes action;
        # height (ranking) = posterior mean of the clone's peak height
        w = (TMPL_W[0] / sp.span[0], TMPL_W[1] / sp.span[0], TMPL_W[2] / sp.span[1], TMPL_W[3] / sp.span[1])
        bl = bayes_locations(U[:, 0], U[:, 1], C, Y, sp.nc, w, sig=self.cfg["bayes_sig"], wscale=self.cfg["bayes_wscale"])
        seen = np.zeros(sp.nc, bool)
        seen[np.unique(C)] = True
        # capped-max height: best plate titer of the clone at the standard medium, corrected by the template
        # penalty at the posterior mode (correction capped: only near wells are trusted). Tracks the clone's true
        # plate peak titer far better than the posterior-mean height, which far wells compress (corr .82 vs .38).
        ly = np.log(np.maximum(Y, 1e-4 * max(Y.max(), 1e-12)))
        g0, a0 = self.cfg["plate_medium"]
        std = (np.abs(U[:, 2] - g0) < 0.02) & (np.abs(U[:, 3] - a0) < 0.01)
        pen = template_pen(U[:, 0] - bl["t_map"][C], U[:, 1] - bl["p_map"][C], w, 5.0)
        v = np.where(std, ly + np.minimum(pen, self.cfg.get("cap_pen", 1.0)), -np.inf)
        hcap = np.full(sp.nc, -np.inf)
        np.maximum.at(hcap, C, v)
        hcap = np.where(np.isfinite(hcap), hcap, bl["height"])
        self.peaks = {"height": hcap if self.cfg.get("rank_h") == "cap" else bl["height"], "hcap": hcap,
                      "t": bl["t"], "p": bl["p"], "seen": seen, "eu": bl["eu"]}
        self.peaks["n"] = len(plate)

    def clone_opt(self, c):
        if self.peaks is None or not self.peaks["seen"][c]:
            return 0.5, 0.5
        if self.n_plates_own < self.cfg["min_fit_plates"]:
            # too few plate conditions for the peak model: use the clone's best observed plate condition
            if self.best_cond_cache is None or self.best_cond_cache[0] != len(self.rows):
                best = {}
                for r, y in self.split()[0]:
                    k = self.sp.clones.index(r[1])
                    if k not in best or y > best[k][0]:
                        best[k] = (y, self.sp.unit(r)[:2])
                self.best_cond_cache = (len(self.rows), best)
            b = self.best_cond_cache[1].get(c)
            if b is not None:
                return float(b[1][0]), float(b[1][1])
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
    rank = plate_rank(st)
    # value of information: the plate condition whose (template-predicted) titers best sharpen the operating
    # T/pH decisions of the top-ranked clones; wells go to the clones in rank order
    U = np.array([sp.unit(r)[:2] for r, _ in plate])
    C = np.array([sp.clones.index(r[1]) for r, _ in plate])
    Y = np.array([y for _, y in plate])
    w = (TMPL_W[0] / sp.span[0], TMPL_W[1] / sp.span[0], TMPL_W[2] / sp.span[1], TMPL_W[3] / sp.span[1])
    k = cfg["voi_k"]
    wts = [1.0 / (1 + 0.3 * i) for i in range(k)]
    t, p, _ = voi_condition(U[:, 0], U[:, 1], C, Y, rank[:k], wts, w, cfg["bayes_sig"],
                            wscale=cfg["bayes_wscale"])
    kk = min(size, sp.nc)
    clones = list(rank[:kk])
    while len(clones) < size:
        clones.append(clones[len(clones) % kk])
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
    # screened = run at the prior recipe in a mini bioreactor (initial-design runs used random recipes, so they do not
    # count) or already run at Pilot scale
    tested = set(prior_titers(st, sp.mbr)) | {sp.clones.index(r[1]) for r in st.rows if r[0] == sp.pilot}
    if st.gp is None:
        refine = 0
    order = plate_rank(st)
    if cfg["screen_eu"] > 0 and st.peaks is not None:
        # expected screened titer: plate height plus log expected relative titer at the chosen T/pH (clones whose
        # optimum is still uncertain are likely to give a collapsed, uninformative screen)
        s = np.where(st.peaks["seen"], st.peaks["height"] + cfg["screen_eu"] * np.log(np.maximum(st.peaks["eu"], 1e-3)),
                     -np.inf)
        order = list(np.argsort(-s))
    screen = [c for c in order if c not in tested][:max(0, size - refine)]
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


# ------------------------------------------------------------------------------------------------ glutamine library
class GlnLib:
    """Library of simulated Pilot glutamine-response surfaces (log titer on an absolute log-grid of glutamine start,
    feed 1 and feed 2). Acts as a prior over response surfaces: observations of one clone re-weight library members
    (each with a free log offset), and the next recipe maximises the mixture's expected improvement."""

    def __init__(self, L, A0, A1, B2):
        self.L = np.asarray(L, float)
        self.ax = [np.log(np.asarray(A0, float)), np.log(np.asarray(A1, float)), np.asarray(B2, float)]
        self.shape = (len(A0), len(A1), len(B2))
        G = np.array(np.meshgrid(*[np.arange(n) for n in self.shape], indexing="ij")).reshape(3, -1).T
        self.pts = np.c_[np.asarray(A0)[G[:, 0]], np.asarray(A1)[G[:, 1]], np.asarray(B2)[G[:, 2]]]

    def interp(self, q):
        """Library log titers at absolute points q (n x 3) by trilinear interpolation -> (N, n)."""
        q = np.atleast_2d(np.asarray(q, float))
        coords = [np.log(np.maximum(q[:, 0], 1e-9)), np.log(np.maximum(q[:, 1], 1e-9)), q[:, 2]]
        idx, frac = [], []
        for ax, v in zip(self.ax, coords):
            v = np.clip(v, ax[0], ax[-1])
            i = np.clip(np.searchsorted(ax, v) - 1, 0, len(ax) - 2)
            idx.append(i); frac.append((v - ax[i]) / (ax[i + 1] - ax[i]))
        out = np.zeros((len(self.L), len(q)))
        n1, n2 = self.shape[1], self.shape[2]
        for d0 in (0, 1):
            for d1 in (0, 1):
                for d2 in (0, 1):
                    w = (frac[0] if d0 else 1 - frac[0]) * (frac[1] if d1 else 1 - frac[1]) * (frac[2] if d2 else 1 - frac[2])
                    flat = (idx[0] + d0) * n1 * n2 + (idx[1] + d1) * n2 + (idx[2] + d2)
                    out += self.L[:, flat] * w
        return out

    def weights(self, q, y, sig, prior=None):
        R = np.asarray(y, float)[None] - self.interp(q)
        c = R.mean(1)
        lw = -((R - c[:, None]) ** 2).sum(1) / (2 * sig ** 2)
        if prior is not None:
            lw = lw + np.log(np.maximum(prior, 1e-300))
        w = np.exp(lw - lw.max())
        return w / w.sum(), c

    def next_points(self, q, y, sig, cand, acq="ei", prior=None):
        """Mixture expected improvement (titer scale) at absolute candidate points."""
        w, c = self.weights(q, y, sig, prior)
        P = self.interp(cand) + c[:, None]
        if acq == "logei":
            return w @ np.maximum(P - np.max(y), 0.0)
        return w @ np.maximum(np.exp(P) - np.exp(np.max(y)), 0.0)


def load_library():
    """Rebuild the library (member x grid log efficiency) from the embedded compressed tables."""
    L = np.asarray(GLN_MEAN)[None] + np.asarray(GLN_COEF) @ np.asarray(GLN_PCS)
    return GlnLib(L, GLN_A0, GLN_A1, GLN_B2)


LIB = None


def lib_pilot(st, inc):
    """Next Pilot recipe for the incumbent clone from the glutamine library, or None."""
    global LIB
    sp, cfg = st.sp, st.cfg
    if LIB is None:
        LIB = load_library()
    _, bio = st.split()
    q, y = [], []
    for r, v in bio:
        if r[0] == sp.pilot and sp.clones.index(r[1]) == inc and v > 0:
            x = [float(u) for u in r[2:]]
            q.append((x[3], x[5], x[7])); y.append(np.log(v))
    if not q:
        return None
    lo, hi = sp.lo, sp.hi
    cand = LIB.pts
    ok = (cand[:, 0] >= lo[3]) & (cand[:, 0] <= hi[3]) & (cand[:, 1] >= lo[5]) & (cand[:, 1] <= hi[5]) \
        & (cand[:, 2] >= lo[7]) & (cand[:, 2] <= hi[7])
    cand = cand[ok]
    if len(cand) == 0:
        return None
    a = LIB.next_points(np.array(q), np.array(y), cfg["lib_sig"], cand, cfg["lib_acq"],
                        feat_prior(st, inc) if cfg.get("feat_lib") else None)
    _, t, p, g, _, _, _ = prior_recipe(st, inc)
    for i in np.argsort(-a):
        a0, a1, b2 = cand[i]
        rec = (inc, t, p, g, float(np.log10(max((a0 - lo[3]) / sp.span[3], 1e-12))),
               float(np.log10(max((a1 - lo[5]) / sp.span[5], 1e-12))), float((b2 - lo[7]) / sp.span[7]))
        if not too_close(st, rec, sp.pilot):
            return rec
    return None


def feat_prior(st, c):
    """Prior weights of the library members from the clone's plate peak height. The library members' plate titer
    at the standard plate medium (pre-computed with the simulator at each member's optimum) predicts which glutamine
    recipe suits the clone (high producers need more glutamine): offline first-Pilot relative titer 0.77 -> 0.85."""
    cfg = st.cfg
    s = cfg.get("feat_s", 0.0)
    if s <= 0 or st.peaks is None or not st.peaks["seen"][c] or st.n_plates_own < cfg["min_fit_plates"] \
            or tuple(cfg["plate_medium"]) != (1.0, 0.05):
        return None
    F = np.asarray(GLN_PLATE, float)
    lw = -0.5 * ((F - float(st.peaks["hcap" if cfg.get("feat_h") == "cap" else "height"][c])) / s) ** 2
    w = np.exp(lw - lw.max())
    return w / w.sum()


def feat_recipe(st, c):
    """First Pilot recipe of a clone: maximise the prior-weighted library mean relative titer, or None."""
    global LIB
    w = feat_prior(st, c)
    if w is None:
        return None
    sp = st.sp
    if LIB is None:
        LIB = load_library()
    lo, hi = sp.lo, sp.hi
    cand = LIB.pts
    ok = (cand[:, 0] >= lo[3]) & (cand[:, 0] <= hi[3]) & (cand[:, 1] >= lo[5]) & (cand[:, 1] <= hi[5]) \
        & (cand[:, 2] >= lo[7]) & (cand[:, 2] <= hi[7])
    cand = cand[ok]
    if len(cand) == 0:
        return None
    v = w @ np.exp(LIB.interp(cand))
    a0, a1, b2 = cand[int(np.argmax(v))]
    _, t, p, g, _, _, _ = prior_recipe(st, c)
    return (c, t, p, g, float(np.log10(max((a0 - lo[3]) / sp.span[3], 1e-12))),
            float(np.log10(max((a1 - lo[5]) / sp.span[5], 1e-12))), float((b2 - lo[7]) / sp.span[7]))


def first_pilot_recipe(st, c):
    return (feat_recipe(st, c) if st.cfg.get("feat_first") else None) or prior_recipe(st, c)


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
        # no prior-recipe screens: initial-design bioreactor runs used random recipes, so trust the plates instead
        order = (plate_rank(st) if cfg["fallback_plates"] else []) or rank_observed(st) or list(range(sp.nc))
    if not pp:
        if not mp and st.n_plates_own == 0 and cfg["blind_copy"]:
            # no screening information of our own yet: repeat the best initial-design bioreactor run at Pilot scale
            _, bio = st.split()
            if bio:
                r, _ = max(bio, key=lambda t: t[1])
                row = np.array([[sp.pilot, r[1]] + [float(v) for v in r[2:]]], dtype=object)
                return np.repeat(row, size, axis=0)
        return sp.bio_rows(sp.pilot, [first_pilot_recipe(st, order[0])] * size)
    inc = max(pp, key=pp.get)
    challengers = [c for c in order if c not in pp]
    if challengers and mp.get(challengers[0], -np.inf) > cfg["switch"] * mp.get(inc, -np.inf):
        return sp.bio_rows(sp.pilot, [first_pilot_recipe(st, challengers[0])] * size)
    n_inc = sum(1 for r in st.rows if r[0] == sp.pilot and sp.clones.index(r[1]) == inc)
    if cfg["use_lib"] and cfg["lib_from"] <= n_inc < cfg["lib_until"]:
        rec = lib_pilot(st, inc)
        if rec is not None:
            return sp.bio_rows(sp.pilot, [rec] * size)
    if n_inc < len(cfg["portfolio"]):
        # a fixed portfolio of complementary glutamine recipes beats model-based search for the first Pilot runs of a
        # clone (the optimum is a sharp, clone-specific cusp in log concentration)
        _, t, p, g, _, _, _ = prior_recipe(st, inc)
        la0, la1, b2 = cfg["portfolio"][n_inc]
        return sp.bio_rows(sp.pilot, [(inc, t, p, g, la0, la1, b2)] * size)
    cand = [r for r in candidates(st, [inc], cfg["pilot_tp_fixed"]) if not too_close(st, r, sp.pilot)]
    gp = st.gp
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
    if cfg["seg_advance"] and not last and (pil.get(seg, 0) > 0 or cfg["advance_unreachable"]) and money < pc:
        # this checkpoint already has a Pilot run and its leftover money cannot buy another: plan the next segment
        # (a Pilot fired now would spend the next segment's money before its screening)
        seg, seg_end = seg + 1, (min(upcoming[1], lab.budget) if len(upcoming) > 1 else lab.budget)
        money = seg_end - lab.spent
        last = len(upcoming) <= 2
    if seg not in plan:
        # allocations follow the stage (segments that already got a Pilot run), not the segment index: when a
        # checkpoint was unreachable, the next segment still needs the screening a first segment would have done
        stage = sum(1 for k, v in pil.items() if v > 0) if cfg["plan_by_stage"] else seg
        pick = lambda key: cfg[key][min(stage, len(cfg[key]) - 1)]  # noqa: E731
        max_p = int(money // pc)
        want = min(max(int(pick("pilot_frac") * money // pc), 1), max_p)
        plan[seg] = {"pilot": want, "plate": pick("plates_per_seg"), "mbr": pick("mbr_per_seg")}
        # carry screening that earlier segments planned but could not afford (their money went to the Pilot)
        for s0 in [k for k in plan if isinstance(k, int) and k < seg]:
            for key, cap in (("plate", cfg.get("carry_plates", 0)), ("mbr", cfg.get("carry_mbr", 0))):
                c = min(plan[s0][key], max(cap - plan.get("_carried_" + key, 0), 0))
                if c > 0:
                    plan[seg][key] += c; plan[s0][key] -= c
                    plan["_carried_" + key] = plan.get("_carried_" + key, 0) + c
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


# ------------------------------------------------------------------------------------------------ library tables
# Pilot glutamine-response library, pre-computed offline with the dbiolab simulator (the same model family the lab
# uses) for 1950 simulated clones drawn from the documented parameter ranges (no clone of any lab, no lab files):
# log(titer / clone's own best) on an absolute grid of glutamine start x feed 1 x feed 2 (mM), glucose at its upper
# bound, temperature/pH near the clone's optimum (with realistic location errors). Stored compressed: mean surface +
# 12 principal components; GLN_COEF holds the component scores of 300 library members.
GLN_A0 = [0.015, 0.0381, 0.0965, 0.2449, 0.6214, 1.5766, 4]
GLN_A1 = [0.06, 0.1521, 0.3856, 0.9774, 2.4777, 6.2808]
GLN_B2 = [6, 18, 32, 48]
GLN_MEAN = [-4.8958, -4.8523, -4.9001, -4.9569, -3.8001, -3.7582, -3.8067, -3.8644, -2.6084, -2.5704, -2.6149, -2.6677, -1.8144, -1.7689, -1.8036, -1.8491, -1.4428, -1.3999, -1.4281, -1.4669, -1.3023, -1.2882, -1.3168, -1.3527, -3.9565, -3.9137, -3.9615, -4.0185, -3.0392, -2.9997, -3.0457, -3.1007, -2.0711, -2.025, -2.0644, -2.1127, -1.4472, -1.3802, -1.4064, -1.4449, -1.1591, -1.0895, -1.1081, -1.1398, -1.0371, -1.0017, -1.0223, -1.0522, -2.4778, -2.4367, -2.4796, -2.5308, -1.9378, -1.8874, -1.9255, -1.9725, -1.3753, -1.2924, -1.3166, -1.3546, -1.0175, -0.9011, -0.9067, -0.934, -0.8368, -0.7296, -0.7288, -0.7497, -0.7349, -0.6769, -0.6806, -0.7034, -1.1433, -1.0383, -1.0605, -1.0966, -0.9999, -0.8677, -0.8797, -0.9117, -0.8295, -0.6705, -0.6636, -0.6857, -0.7008, -0.5299, -0.514, -0.524, -0.6178, -0.4751, -0.4599, -0.4685, -0.5446, -0.464, -0.4566, -0.4655, -0.6183, -0.3975, -0.3898, -0.4107, -0.6049, -0.3793, -0.3642, -0.3827, -0.5859, -0.3604, -0.3336, -0.3469, -0.5701, -0.3697, -0.3371, -0.3359, -0.555, -0.4019, -0.3745, -0.3724, -0.5214, -0.4334, -0.4186, -0.4198, -0.5362, -0.307, -0.2753, -0.2824, -0.5387, -0.3144, -0.2804, -0.2866, -0.5439, -0.332, -0.2943, -0.2978, -0.553, -0.3669, -0.3271, -0.3268, -0.5617, -0.4194, -0.3895, -0.3842, -0.5543, -0.4728, -0.4571, -0.4556, -0.5723, -0.4166, -0.3818, -0.3781, -0.573, -0.4202, -0.3863, -0.3818, -0.5756, -0.4286, -0.3956, -0.3913, -0.5801, -0.4474, -0.4176, -0.412, -0.5848, -0.4803, -0.4579, -0.4524, -0.5804, -0.5179, -0.505, -0.5033]
GLN_PCS = [
    [-0.0934, -0.0948, -0.0928, -0.0906, -0.1202, -0.1211, -0.1196, -0.1179, -0.1324, -0.1332, -0.1324, -0.1315, -0.1226, -0.1259, -0.1262, -0.1261, -0.1105, -0.1153, -0.1165, -0.1169, -0.107, -0.1104, -0.1116, -0.1122, -0.1143, -0.1154, -0.1137, -0.1118, -0.1289, -0.1295, -0.1284, -0.1271, -0.1268, -0.1291, -0.1288, -0.1284, -0.1078, -0.1141, -0.1154, -0.1159, -0.0921, -0.0999, -0.102, -0.1031, -0.0886, -0.0942, -0.0961, -0.0972, -0.1264, -0.1274, -0.1265, -0.1256, -0.1212, -0.1239, -0.1236, -0.1232, -0.1011, -0.1086, -0.11, -0.1105, -0.0736, -0.085, -0.0882, -0.0897, -0.0569, -0.0674, -0.0709, -0.0727, -0.0542, -0.0611, -0.0641, -0.0653, -0.0836, -0.0935, -0.0949, -0.0953, -0.0701, -0.0829, -0.0854, -0.0862, -0.0489, -0.0631, -0.0672, -0.0689, -0.0273, -0.0405, -0.0447, -0.0472, -0.0143, -0.0244, -0.0279, -0.03, -0.0116, -0.0181, -0.0209, -0.0228, -0.0152, -0.0316, -0.0351, -0.0362, -0.0106, -0.0265, -0.0305, -0.0318, -0.0027, -0.0167, -0.0214, -0.023, 0.0073, -0.0034, -0.0076, -0.0101, 0.016, 0.0082, 0.0053, 0.0033, 0.0199, 0.015, 0.0129, 0.0114, 0.019, 0.0079, 0.0041, 0.0028, 0.02, 0.0093, 0.0055, 0.0041, 0.0221, 0.0124, 0.0087, 0.0071, 0.026, 0.018, 0.0146, 0.0129, 0.0314, 0.0255, 0.023, 0.0214, 0.0358, 0.0321, 0.0305, 0.0293, 0.035, 0.0288, 0.0263, 0.0248, 0.0353, 0.0292, 0.0267, 0.0252, 0.036, 0.0301, 0.0277, 0.0263, 0.0375, 0.0322, 0.03, 0.0286, 0.04, 0.0358, 0.034, 0.0328, 0.0424, 0.0395, 0.0382, 0.0372],
    [-0.1618, -0.16, -0.1607, -0.1615, -0.123, -0.1208, -0.1221, -0.1237, -0.0553, -0.0571, -0.0593, -0.0618, 0.0183, 0.0034, -0.0031, -0.0062, 0.0573, 0.0377, 0.0289, 0.0242, 0.0627, 0.0473, 0.0391, 0.0342, -0.1334, -0.1312, -0.1323, -0.1338, -0.0879, -0.0859, -0.0878, -0.0901, -0.0102, -0.0203, -0.0242, -0.0271, 0.0612, 0.0383, 0.029, 0.0239, 0.0958, 0.0709, 0.0599, 0.0538, 0.098, 0.0785, 0.069, 0.0634, -0.0543, -0.0559, -0.0581, -0.0607, -0.0035, -0.0149, -0.0186, -0.0215, 0.0666, 0.0408, 0.031, 0.0262, 0.1186, 0.0905, 0.0766, 0.07, 0.1348, 0.1139, 0.1015, 0.0949, 0.133, 0.117, 0.1066, 0.1033, 0.0861, 0.0551, 0.0461, 0.0427, 0.1082, 0.0751, 0.0621, 0.0584, 0.1284, 0.1037, 0.0872, 0.0807, 0.1329, 0.1198, 0.1078, 0.1002, 0.1307, 0.1219, 0.1139, 0.1095, 0.1294, 0.1202, 0.1141, 0.111, 0.1116, 0.0948, 0.082, 0.0807, 0.1088, 0.0958, 0.0817, 0.0802, 0.1024, 0.0953, 0.0807, 0.078, 0.0927, 0.0886, 0.0779, 0.0737, 0.0858, 0.0807, 0.0748, 0.0713, 0.0872, 0.0801, 0.0762, 0.0749, 0.0589, 0.0546, 0.0429, 0.0425, 0.0575, 0.0534, 0.0416, 0.0409, 0.0548, 0.0508, 0.0398, 0.0381, 0.051, 0.0466, 0.0375, 0.0345, 0.0485, 0.0427, 0.0369, 0.0338, 0.0525, 0.0453, 0.0418, 0.0407, 0.0326, 0.0266, 0.0193, 0.0168, 0.0326, 0.0265, 0.0193, 0.0168, 0.0325, 0.0261, 0.0195, 0.0168, 0.0328, 0.0259, 0.0199, 0.0174, 0.0344, 0.0269, 0.0222, 0.0201, 0.0389, 0.031, 0.028, 0.0272],
    [0.149, 0.1459, 0.1479, 0.1511, 0.105, 0.1014, 0.1041, 0.1081, 0.0392, 0.0387, 0.0414, 0.0453, -0.0158, -0.009, -0.0049, -0.0013, -0.0362, -0.0298, -0.0259, -0.0225, -0.0388, -0.0348, -0.0316, -0.0285, 0.1167, 0.1131, 0.1157, 0.1197, 0.0706, 0.0674, 0.0702, 0.0742, 0.0082, 0.0129, 0.0161, 0.0197, -0.0322, -0.0248, -0.0204, -0.017, -0.0373, -0.0367, -0.0347, -0.0319, -0.0366, -0.0383, -0.0368, -0.0349, 0.0452, 0.0446, 0.0473, 0.0512, 0.0087, 0.0141, 0.0172, 0.0208, -0.0259, -0.0194, -0.0146, -0.0113, -0.0239, -0.0289, -0.0295, -0.0283, -0.0112, -0.0219, -0.0277, -0.0294, -0.0099, -0.0197, -0.0262, -0.0274, -0.0192, -0.0159, -0.0128, -0.0104, -0.0151, -0.0188, -0.017, -0.0151, 0.0042, -0.0102, -0.0154, -0.0149, 0.033, 0.0087, -0.0009, -0.0061, 0.0534, 0.0255, 0.0145, 0.0095, 0.051, 0.0282, 0.0184, 0.0144, 0.0551, 0.0196, 0.01, 0.0103, 0.0642, 0.0259, 0.0137, 0.014, 0.0807, 0.0388, 0.0222, 0.0209, 0.1027, 0.0671, 0.0481, 0.04, 0.1184, 0.0894, 0.0709, 0.0617, 0.1159, 0.0927, 0.0781, 0.072, 0.1266, 0.0841, 0.0637, 0.0614, 0.129, 0.0882, 0.0663, 0.0637, 0.1339, 0.0972, 0.0727, 0.0692, 0.1422, 0.1107, 0.0851, 0.0793, 0.1485, 0.12, 0.1005, 0.09, 0.1432, 0.1203, 0.106, 0.0988, 0.1567, 0.1264, 0.1038, 0.0957, 0.1571, 0.1269, 0.1047, 0.0964, 0.1575, 0.1278, 0.1066, 0.0977, 0.158, 0.1296, 0.11, 0.1004, 0.1562, 0.1301, 0.1142, 0.1042, 0.149, 0.1278, 0.1153, 0.1091],
    [0.0494, 0.0472, 0.0485, 0.05, 0.0197, 0.0175, 0.0187, 0.02, -0.0031, -0.0114, -0.0105, -0.0094, 0.01, -0.0242, -0.0311, -0.0304, 0.0266, -0.0256, -0.0378, -0.0405, 0.0104, -0.0309, -0.0416, -0.0443, 0.0284, 0.0262, 0.0274, 0.0287, 0.0059, 0.0029, 0.004, 0.0053, 0.004, -0.0197, -0.021, -0.0198, 0.042, -0.0219, -0.0357, -0.0382, 0.0652, -0.0213, -0.0431, -0.0484, 0.0412, -0.0287, -0.047, -0.0516, 0.0088, 0.0007, 0.0018, 0.0031, 0.0132, -0.0129, -0.0139, -0.0126, 0.0579, -0.0166, -0.0314, -0.0323, 0.108, -0.0067, -0.0443, -0.0508, 0.122, 0.001, -0.0444, -0.0554, 0.0848, -0.0078, -0.0445, -0.0553, 0.0885, -0.0083, -0.02, -0.0174, 0.1136, -0.0082, -0.0332, -0.0318, 0.1474, 0.003, -0.0502, -0.0567, 0.1694, 0.0156, -0.0435, -0.0653, 0.1663, 0.0214, -0.0312, -0.0477, 0.1247, 0.0122, -0.0259, -0.0381, 0.1747, -0.0089, -0.0503, -0.0461, 0.1768, -0.0085, -0.0601, -0.0566, 0.1773, -0.0069, -0.0759, -0.0773, 0.1708, -0.006, -0.0886, -0.1011, 0.153, -0.0042, -0.0657, -0.0882, 0.1129, -0.0046, -0.044, -0.0569, 0.1559, -0.0461, -0.1122, -0.1109, 0.1536, -0.0464, -0.1164, -0.1169, 0.1488, -0.0463, -0.1228, -0.1275, 0.1381, -0.0447, -0.1244, -0.1382, 0.1183, -0.0406, -0.1032, -0.1268, 0.0844, -0.0297, -0.0674, -0.0817, 0.107, -0.0648, -0.1352, -0.1531, 0.106, -0.0646, -0.1338, -0.1525, 0.1028, -0.0638, -0.1301, -0.1499, 0.096, -0.0607, -0.122, -0.1433, 0.0826, -0.0521, -0.1003, -0.1213, 0.0606, -0.0355, -0.0675, -0.0801],
    [-0.1323, -0.1241, -0.1255, -0.129, -0.0691, -0.0615, -0.0632, -0.0672, 0.0255, 0.0102, 0.0089, 0.0061, 0.1085, 0.0724, 0.0505, 0.0477, 0.1119, 0.1089, 0.0854, 0.0755, 0.1218, 0.1144, 0.096, 0.0865, -0.0858, -0.0779, -0.0796, -0.0834, -0.0314, -0.0272, -0.0287, -0.032, 0.0679, 0.0264, 0.0175, 0.0151, 0.0784, 0.0882, 0.063, 0.0514, 0.0406, 0.1012, 0.0883, 0.078, 0.0599, 0.0951, 0.0852, 0.0795, 0.0007, -0.0137, -0.0148, -0.0173, 0.0542, 0.0113, 0.0037, 0.0016, 0.0361, 0.064, 0.0361, 0.0261, -0.0655, 0.0481, 0.0478, 0.0432, -0.1, -0.0108, -0.0019, 0.0023, -0.0764, -0.0308, -0.0225, -0.0286, -0.0214, 0.0419, 0.018, 0.0171, -0.0774, 0.0408, 0.0188, 0.0164, -0.1401, -0.0217, -0.0177, -0.024, -0.1464, -0.1099, -0.1046, -0.1009, -0.1223, -0.1482, -0.1638, -0.1628, -0.1209, -0.1573, -0.1751, -0.1772, -0.0873, -0.0449, -0.0482, -0.046, -0.0679, -0.0539, -0.058, -0.0562, -0.0291, -0.0656, -0.0754, -0.0757, 0.0258, -0.0451, -0.0857, -0.0846, 0.0543, -0.0223, -0.0802, -0.0924, 0.03, -0.0483, -0.0933, -0.1095, 0.1276, 0.0727, 0.0336, 0.0258, 0.1326, 0.0771, 0.0353, 0.0267, 0.1393, 0.0845, 0.0361, 0.0256, 0.145, 0.0862, 0.0311, 0.0172, 0.1334, 0.0608, 0.0006, -0.0173, 0.0799, 0.0013, -0.0424, -0.0603, 0.1527, 0.0877, 0.0283, 0.0142, 0.1515, 0.0849, 0.0258, 0.0118, 0.1476, 0.0791, 0.0199, 0.0061, 0.1366, 0.0636, 0.0067, -0.0092, 0.11, 0.0309, -0.0182, -0.0363, 0.0542, -0.0212, -0.0565, -0.0727],
    [-0.0835, -0.0813, -0.0863, -0.0905, -0.0035, -0.0019, -0.0057, -0.0089, 0.01, 0.0339, 0.0321, 0.0307, -0.0792, 0.02, 0.0412, 0.0423, -0.1357, -0.0159, 0.0181, 0.0269, -0.1096, -0.0239, 0.0073, 0.0171, -0.0048, -0.0031, -0.0072, -0.0106, 0.0339, 0.0381, 0.0355, 0.0336, -0.022, 0.05, 0.0571, 0.0565, -0.1449, -0.0022, 0.0353, 0.0455, -0.1759, -0.0561, -0.009, 0.0083, -0.1474, -0.0654, -0.0286, -0.013, 0.0298, 0.0545, 0.0526, 0.0511, -0.0203, 0.0615, 0.0677, 0.0668, -0.1451, 0.0127, 0.0535, 0.0625, -0.1714, -0.0805, -0.0246, -0.0057, -0.1249, -0.109, -0.0855, -0.0674, -0.1178, -0.1086, -0.0942, -0.1058, -0.1463, 0.0268, 0.0637, 0.064, -0.1439, -0.0064, 0.0583, 0.0613, -0.0766, -0.0517, 0.0124, 0.0349, 0.0197, -0.0252, -0.0237, -0.0061, 0.0605, 0.0148, -0.0089, -0.013, 0.0402, 0.0015, -0.0173, -0.0226, 0.0905, 0.0904, 0.1407, 0.1348, 0.1048, 0.0959, 0.1485, 0.1449, 0.1222, 0.104, 0.143, 0.1529, 0.1256, 0.1029, 0.1118, 0.1229, 0.1006, 0.0653, 0.0396, 0.0448, 0.0458, 0.0111, -0.0026, -0.0072, 0.094, 0.079, 0.1219, 0.1351, 0.0892, 0.0701, 0.1102, 0.1262, 0.0779, 0.0509, 0.078, 0.1009, 0.0567, 0.0171, 0.0195, 0.0447, 0.0199, -0.029, -0.0484, -0.0334, -0.0451, -0.0857, -0.0961, -0.0972, -0.0028, -0.0573, -0.0631, -0.0445, -0.0055, -0.0593, -0.0667, -0.048, -0.0098, -0.0644, -0.0733, -0.0568, -0.0244, -0.0771, -0.0906, -0.0764, -0.0534, -0.1019, -0.1164, -0.109, -0.101, -0.1357, -0.1443, -0.1448],
    [0.0092, -0.0031, -0.0036, -0.0018, 0.0044, -0.0084, -0.0089, -0.0072, -0.0329, -0.0581, -0.0592, -0.0581, -0.0192, -0.1031, -0.126, -0.1262, -0.0077, -0.1112, -0.1535, -0.1674, -0.035, -0.1279, -0.1674, -0.183, 0.0553, 0.0425, 0.0421, 0.0439, 0.0426, 0.0284, 0.0279, 0.0296, 0.0403, -0.0229, -0.0282, -0.027, 0.0733, -0.0346, -0.0799, -0.0935, 0.0779, -0.019, -0.085, -0.1108, 0.0635, -0.0345, -0.092, -0.1156, 0.091, 0.0674, 0.0671, 0.069, 0.1084, 0.042, 0.0374, 0.0392, 0.1454, 0.0392, -0.0075, -0.015, 0.098, 0.0853, -0.0017, -0.0324, 0.0454, 0.1001, 0.013, -0.0228, 0.0807, 0.0755, 0.0021, -0.0389, 0.162, 0.0703, 0.0318, 0.032, 0.1257, 0.0984, 0.0284, 0.0233, 0.0238, 0.1444, 0.0403, 0.012, -0.0853, 0.1034, 0.0375, -0.0153, -0.1032, 0.0362, -0.0084, -0.0417, -0.0366, 0.0117, -0.0326, -0.0583, -0.1014, 0.1341, 0.0483, 0.0471, -0.116, 0.142, 0.047, 0.0445, -0.1298, 0.1318, 0.0388, 0.0294, -0.1229, 0.0548, 0.0031, -0.0366, -0.1011, -0.0183, -0.0668, -0.1104, -0.0586, -0.067, -0.1185, -0.1475, -0.0418, 0.1802, 0.087, 0.0764, -0.0379, 0.1752, 0.0831, 0.0719, -0.031, 0.1536, 0.0698, 0.0558, -0.0203, 0.104, 0.0357, 0.004, -0.0129, 0.0336, -0.0289, -0.0747, -0.0163, -0.0591, -0.1165, -0.1486, 0.0317, 0.1034, 0.0289, -0.0083, 0.0315, 0.0988, 0.0248, -0.0133, 0.0309, 0.0881, 0.0147, -0.0257, 0.0274, 0.058, -0.0101, -0.053, 0.0172, -0.0016, -0.0702, -0.113, 0.0035, -0.0652, -0.1219, -0.1526],
    [-0.152, -0.1654, -0.1677, -0.1688, -0.054, -0.0672, -0.0704, -0.0723, -0.0193, -0.0248, -0.0276, -0.0291, -0.0855, -0.0693, -0.0571, -0.0579, -0.1465, -0.1201, -0.1009, -0.0939, -0.1754, -0.1401, -0.1218, -0.1148, -0.0055, -0.0188, -0.0217, -0.0234, 0.0691, 0.0566, 0.0537, 0.0522, 0.0564, 0.0652, 0.0669, 0.066, -0.0298, -0.0066, 0.0146, 0.0227, -0.0871, -0.0655, -0.0312, -0.0185, -0.1185, -0.0819, -0.0511, -0.0396, 0.1536, 0.1448, 0.1425, 0.1417, 0.1464, 0.1551, 0.1569, 0.1567, 0.0754, 0.1017, 0.1256, 0.1317, 0.02, 0.007, 0.0576, 0.0768, -0.0214, -0.0524, 0.0003, 0.0246, -0.0703, -0.0672, -0.0199, 0.0058, 0.0977, 0.1223, 0.1461, 0.1472, 0.0832, 0.0818, 0.1262, 0.1314, 0.0621, 0.0043, 0.0727, 0.0952, 0.0366, -0.0525, -0.0036, 0.0387, 0.0019, -0.0717, -0.0358, -0.0065, -0.0528, -0.0751, -0.0373, -0.0152, 0.0409, -0.071, -0.0132, -0.0095, 0.0385, -0.092, -0.0268, -0.0219, 0.032, -0.1188, -0.0516, -0.0412, 0.0287, -0.107, -0.0678, -0.0291, 0.0154, -0.0625, -0.0329, 0.0017, -0.012, -0.0236, 0.006, 0.0219, 0.0243, -0.1308, -0.0751, -0.0713, 0.0235, -0.127, -0.0721, -0.0675, 0.0212, -0.114, -0.0603, -0.0562, 0.0181, -0.0762, -0.0309, -0.0139, 0.0189, -0.0188, 0.0195, 0.0502, 0.0243, 0.0459, 0.0783, 0.0952, 0.0113, -0.0295, 0.0209, 0.0427, 0.0122, -0.0261, 0.0247, 0.0462, 0.012, -0.0186, 0.0311, 0.0566, 0.0162, 0.0002, 0.047, 0.0748, 0.0272, 0.0391, 0.0794, 0.1062, 0.0394, 0.0816, 0.1139, 0.1288],
    [0.1095, 0.111, 0.1163, 0.1233, 0.0082, 0.0108, 0.0153, 0.0218, -0.0495, -0.0287, -0.0262, -0.0224, -0.0871, -0.0324, -0.0126, -0.0098, -0.0858, -0.0242, 0.0004, 0.0088, -0.0561, -0.0024, 0.0175, 0.0238, 0.0125, 0.0148, 0.0194, 0.026, -0.0441, -0.0382, -0.0352, -0.0305, -0.0942, -0.0473, -0.0389, -0.0364, -0.1084, -0.0432, -0.0162, -0.0087, -0.09, 0.0012, 0.0301, 0.0389, -0.0502, 0.0336, 0.0581, 0.0649, -0.0693, -0.0485, -0.0465, -0.0435, -0.1009, -0.052, -0.0446, -0.0427, -0.1118, -0.046, -0.0171, -0.0117, -0.0808, 0.0226, 0.0697, 0.0757, -0.0572, 0.0618, 0.1273, 0.1444, -0.0112, 0.0802, 0.1396, 0.1889, -0.0801, 0.0106, 0.0354, 0.0301, -0.0692, 0.0381, 0.0796, 0.0768, -0.0604, 0.0736, 0.1543, 0.1677, -0.0656, 0.0441, 0.1206, 0.1651, -0.0644, -0.0153, 0.041, 0.0726, -0.0688, -0.0373, 0.0111, 0.0384, -0.0539, 0.0755, 0.1672, 0.1703, -0.0503, 0.0438, 0.1531, 0.1616, -0.0413, -0.0272, 0.0952, 0.1184, -0.0271, -0.1416, -0.083, -0.0069, -0.023, -0.1618, -0.1488, -0.1098, -0.0734, -0.1547, -0.1438, -0.1289, 0.0279, -0.1285, -0.0359, -0.0136, 0.0338, -0.1334, -0.045, -0.0195, 0.0505, -0.1356, -0.0646, -0.0345, 0.078, -0.1164, -0.0826, -0.0489, 0.0977, -0.0632, -0.0685, -0.0465, 0.043, -0.0429, -0.0461, -0.0436, 0.1541, -0.0205, -0.0127, 0.0097, 0.1548, -0.0189, -0.012, 0.0102, 0.1551, -0.0127, -0.009, 0.0118, 0.1543, 0.0016, -0.0016, 0.0147, 0.1325, 0.0176, 0.0104, 0.0198, 0.0529, -0.0021, -0.0034, -0.0037],
    [-0.1098, -0.102, -0.1077, -0.1194, -0.0108, -0.0037, -0.0087, -0.0193, 0.0575, 0.073, 0.0706, 0.0633, -0.0223, 0.0802, 0.0891, 0.0857, -0.0773, 0.0442, 0.0732, 0.0763, -0.0556, 0.0333, 0.056, 0.0615, -0.0383, -0.031, -0.0363, -0.0472, 0.0277, 0.0349, 0.0313, 0.0225, 0.0059, 0.0704, 0.0703, 0.0647, -0.1115, 0.0204, 0.0537, 0.0581, -0.0925, -0.0316, 0, 0.0103, -0.061, -0.0365, -0.0228, -0.0158, 0.0225, 0.038, 0.0355, 0.0282, -0.026, 0.0476, 0.0473, 0.0419, -0.1386, -0.0083, 0.0263, 0.0289, -0.0593, -0.0544, -0.0489, -0.0429, 0.0083, 0.0059, -0.0688, -0.0886, 0.0426, 0.0212, -0.065, -0.1214, -0.1332, -0.0343, -0.0112, -0.0111, -0.085, -0.0627, -0.0268, -0.0264, 0.0005, -0.0072, -0.0509, -0.0468, 0.034, 0.1252, 0.0381, -0.0264, 0.0527, 0.1834, 0.1007, 0.0525, 0.1106, 0.1491, 0.0741, 0.0407, -0.0378, 0.0781, 0.0182, 0.0129, -0.046, 0.0996, 0.0177, 0.0173, -0.0669, 0.1126, -0.0041, -0.006, -0.0971, 0.0756, -0.072, -0.0885, -0.0746, 0.0494, -0.055, -0.1208, 0.003, 0.013, -0.0754, -0.1042, -0.1635, -0.0055, -0.1395, -0.1359, -0.159, 0.0003, -0.1446, -0.1444, -0.1418, 0.0231, -0.1473, -0.1555, -0.0979, 0.0739, -0.1069, -0.1484, -0.0012, 0.1199, -0.008, -0.0927, 0.0933, 0.0845, -0.0141, -0.059, 0.0246, 0.1669, 0.0059, -0.0579, 0.0293, 0.1697, 0.0124, -0.0544, 0.0432, 0.1731, 0.0248, -0.0468, 0.0729, 0.1747, 0.0412, -0.0342, 0.1187, 0.1538, 0.0482, -0.027, 0.1247, 0.0692, -0.0145, -0.0545],
    [0.0341, 0.0228, 0.0241, 0.0254, -0.0044, -0.0156, -0.0153, -0.0153, 0.0056, -0.0619, -0.0633, -0.0647, 0.1418, -0.0169, -0.0925, -0.0983, 0.1146, 0.0454, -0.0542, -0.0899, 0.1116, 0.0326, -0.0528, -0.0884, 0.0251, 0.0138, 0.0145, 0.015, 0.0018, -0.0153, -0.0154, -0.0158, 0.1332, -0.0259, -0.0534, -0.0548, 0.1086, 0.0852, -0.0221, -0.06, -0.0208, 0.1019, 0.0267, -0.025, 0.0047, 0.0765, 0.021, -0.018, 0.0538, -0.0094, -0.0099, -0.0104, 0.1621, -0.0019, -0.0264, -0.0273, 0.0815, 0.1087, -0.007, -0.0374, -0.1523, 0.0681, 0.0477, 0.0042, -0.1784, -0.0404, -0.0001, -0.0182, -0.1516, -0.0441, -0.0147, -0.0911, -0.0265, 0.0665, -0.0407, -0.0441, -0.1309, 0.0936, -0.0369, -0.0523, -0.1527, -0.0022, -0.0191, -0.082, -0.0183, -0.0342, 0.0251, 0.0058, 0.0413, 0.0061, 0.0525, 0.05, 0.0133, 0.0304, 0.0539, 0.0448, 0.1077, 0.0863, -0.007, -0.0194, 0.1369, 0.1002, 0.014, -0.0036, 0.1636, 0.112, 0.0701, 0.0272, 0.1277, 0.0943, 0.3363, 0.1154, 0.024, -0.0389, 0.0318, 0.0587, -0.1092, -0.1331, -0.1087, -0.0974, 0.0197, -0.0978, -0.1603, -0.188, 0.0121, -0.0944, -0.1469, -0.1894, 0.0016, -0.0702, -0.0857, -0.1543, -0.0074, -0.0129, 0.0351, -0.0267, -0.0206, -0.0092, 0.0518, 0.0524, -0.0935, -0.0793, -0.0658, -0.0637, 0.0043, 0.0457, 0.0997, 0.065, 0.0027, 0.0467, 0.1002, 0.0707, 0.0015, 0.0461, 0.1035, 0.0805, -0.0091, 0.0338, 0.0914, 0.0835, -0.037, -0.0091, 0.0232, 0.0322, -0.0828, -0.0561, -0.0481, -0.0491],
    [-0.1095, -0.1037, -0.1105, -0.1175, 0.0064, 0.0119, 0.0066, 0.0006, 0.0483, 0.0827, 0.0806, 0.0775, -0.0168, 0.0537, 0.0894, 0.0911, 0.0057, -0.0073, 0.0398, 0.0628, -0.0068, -0.0264, 0.0191, 0.0456, -0.0175, -0.0122, -0.0176, -0.0236, 0.0425, 0.0503, 0.0469, 0.0427, -0.0247, 0.0627, 0.0717, 0.0711, -0.0028, -0.0392, 0.0088, 0.0345, 0.0861, -0.0955, -0.0707, -0.0359, 0.0546, -0.1049, -0.0782, -0.0515, 0.0047, 0.0384, 0.0366, 0.0338, -0.0617, 0.029, 0.0368, 0.0359, -0.008, -0.0791, -0.0279, -0.0085, 0.1202, -0.0988, -0.1347, -0.1065, 0.1284, 0.0026, -0.0578, -0.0512, 0.1396, 0.022, -0.0142, 0.0252, 0.0275, -0.1278, -0.0819, -0.0749, 0.0771, -0.1394, -0.1094, -0.0998, 0.0441, -0.0296, -0.1036, -0.0798, -0.0464, 0.0943, 0.0657, 0.0764, -0.0577, 0.0621, 0.1317, 0.1569, -0.0619, 0.0356, 0.1184, 0.1466, -0.0505, -0.0105, -0.0426, -0.0455, -0.0457, 0.007, -0.0301, -0.032, -0.0285, 0.0023, -0.0146, -0.0129, -0.0082, -0.0732, -0.0207, 0.021, -0.0325, -0.1548, -0.0173, 0.0321, -0.1449, -0.1923, -0.0891, -0.0492, 0.1428, -0.0288, -0.0208, -0.0286, 0.1466, -0.0277, -0.0109, -0.019, 0.1501, -0.0256, 0.0138, 0.0063, 0.1402, -0.0279, 0.0628, 0.0711, 0.0588, -0.0808, 0.0546, 0.111, -0.1172, -0.1624, -0.0674, -0.0229, 0.1165, -0.0188, 0.1056, 0.1309, 0.1094, -0.0228, 0.1049, 0.132, 0.092, -0.0372, 0.0981, 0.1304, 0.0431, -0.0744, 0.0651, 0.1123, -0.0603, -0.1469, -0.037, 0.0229, -0.1593, -0.15, -0.0667, -0.0256],
]
GLN_COEF = [
    [-3.511, 0.835, -0.622, -2.248, 0.047, -0.778, -0.089, -0.217, 0.087, -0.166, 0.155, -0.035],
    [-3.983, 1.429, -0.268, 1.213, -0.753, -0.031, -0.057, 0.042, 0.448, 0.263, 0.323, 0.016],
    [5.664, -5.676, 3.728, 0.726, -1.03, -0.438, 0.509, -0.107, 0.31, -0.457, 0.11, -0.301],
    [-5.628, 0.988, 0.607, -2.315, 0.322, -0.846, -0.123, -0.104, -0.34, 0.371, 0.238, 0.043],
    [15.815, -1.233, -0.277, 0.183, 0.128, 0.203, 0.128, 0.03, 0.165, 0.011, 0.028, 0.011],
    [5.202, 1.602, 1.622, 0.027, 0.249, 0.051, 0.176, 0.698, -0.022, 0.032, 0.02, 0.028],
    [1.974, 2.447, 2.022, -0.075, 0.413, -0.066, 0.145, 0.958, -0.156, 0.129, -0.002, 0.085],
    [-6.454, -0.201, -0.932, -2.433, 0.098, -0.375, 1.364, -0.613, 0, 0.055, 0.462, 0.242],
    [-9.963, -1.792, 1.162, -2.724, 0.589, 0.146, 0.645, -0.135, -0.096, 0.346, 0.47, -0.265],
    [-10.012, -0.764, 0.939, 1.355, -0.293, -0.718, -0.033, 0.231, 0.013, -0.288, -0.129, -0.189],
    [17.979, -1.173, -2.493, -0.837, 0.021, 0.014, -0.467, -0.349, -0.735, 0.229, 0.023, -0.096],
    [-4.665, 1.99, 0.675, 0.921, -0.382, -0.095, 0, 0.363, 0.162, 0.144, 0.217, 0.028],
    [-5.569, 1.551, 0.631, 0.646, 0.03, -0.181, 0.158, 0.234, 0.204, -0.067, -0.132, 0.106],
    [14.216, -2.147, 0.087, 0.155, 0.364, -0.119, -0.217, -0.348, 0.112, 0.08, -0.024, 0.11],
    [-8.465, -0.104, 0.701, -0.557, 0.838, -0.451, -0.12, -0.194, 0.025, 0.21, 0.05, -0.187],
    [-3.963, 1.622, -0.47, 0.79, -0.511, -0.132, 0.072, 0.059, 0.476, 0.338, -0.046, 0.115],
    [14.016, -1.636, -0.184, 0.057, 0.571, -0.152, -0.293, -0.242, 0.083, 0.176, -0.06, 0.152],
    [-11.221, -1.405, 0.142, 1.48, 0.163, -0.62, -0.36, 0.41, -0.262, 0.097, 0.007, -0.065],
    [10.324, -0.922, 0.993, 0.25, 0.061, 0.052, 0.231, 0.301, 0.1, -0.023, 0.022, -0.013],
    [-8.434, 0.64, 0.481, 0.405, 0.412, 0.079, 0.616, 0.036, 0.079, -0.213, -0.392, 0.183],
    [-13.501, -7.465, 2.043, 1.665, -1.151, 1.448, 1.143, -1.19, 0.245, -0.457, 0.393, 0.935],
    [-6.73, 0.614, -2.205, -2.618, -0.584, -0.009, 2.344, -0.521, -0.871, 0.287, 0.323, 0.885],
    [14.432, -2.558, 0.36, 0.213, 0.049, 0.172, 0.2, 0.175, 0.071, 0.007, 0.001, 0.028],
    [-10.705, -1.916, 1.444, -1.386, 0.649, 0.5, 0.368, -0.603, -0.051, 0.233, 0.222, -0.312],
    [-3.277, 0.573, -1.13, -2.87, -0.007, -0.799, -0.203, 0.376, 0.038, -0.377, 0.285, 0.164],
    [-8.576, -2.02, -0.316, -2.824, 0.272, -0.198, 0.03, -0.425, 0.501, -0.441, 0.325, -0.384],
    [-10.296, -0.958, 1.994, -0.58, 0.337, -0.032, 0.4, -0.411, 0.03, -0.013, 0.216, -0.082],
    [13.085, -2.375, 0.587, 0.261, 0.076, -0.124, -0.024, -0.235, 0.158, -0.018, 0.015, 0.015],
    [6.439, 3.29, -0.087, 0.012, 0.51, 0.569, -0.045, 0.28, -0.189, -0.173, -0.127, 0.069],
    [9.334, 3.109, -1.284, 0.163, 0.266, 0.77, -0.274, -0.566, -0.108, -0.291, -0.165, -0.069],
    [-1.109, 2.787, 0.888, 0.495, -0.083, 0.041, -0.375, -0.195, 0.187, -0.119, 0.166, -0.094],
    [11.22, -5.702, 1.983, 0.615, -0.631, -0.256, 0.545, 0.181, 0.226, -0.26, 0.07, -0.201],
    [6.19, 3.007, -0.469, 0.289, 0.172, 0.729, -0.187, -0.036, -0.154, -0.221, -0.071, -0.03],
    [13.227, -2.829, 0.573, 0.357, 0.049, 0.102, 0.354, 0.419, -0.016, 0.061, 0.024, 0.042],
    [-13.288, -5.946, 2.713, 0.961, -0.794, 0.819, 0.469, -0.757, 0.063, 0.114, 0.077, 0.659],
    [-12.382, -2.058, 2.401, 0.389, 0.158, 0.215, 0.59, -0.462, 0.091, -0.385, -0.087, -0.083],
    [0.817, 0.715, -0.786, -1.268, -1.265, -0.078, 0.484, -0.25, -0.392, -0.005, -0.287, -0.229],
    [-2.307, 3.301, 1.132, 0.373, 0.246, 0.306, -0.012, 0.601, -0.301, -0.163, -0.074, 0.015],
    [-7.746, -0.059, -3.154, 1.806, 0.315, -0.252, 0.254, 0.711, 0.284, -0.192, -0.269, -0.137],
    [3.089, 1.629, 2.214, 0.046, 0.113, -0.164, 0.068, 0.458, 0.144, -0.083, 0.004, -0.056],
    [6.814, 1.111, -1.445, -1.493, -0.948, -0.155, 0.136, 0.253, -0.985, -0.029, 0.005, -0.093],
    [1.11, 2.486, -1.677, 1.06, -0.698, 0.634, 0.027, 0.239, 0.319, 0.172, 0.492, -0.033],
    [-7.03, -2.298, -0.702, -3.469, 0.122, 0.141, -1.343, 1.15, -0.08, 0.094, 0.003, 0.264],
    [11.71, 0.852, -0.298, -0.145, 0.7, 0.127, 0.029, 0.616, -0.223, 0.212, -0.018, 0.084],
    [-2.412, 2.857, -0.036, 0.716, -0.126, 0.178, 0.01, 0.514, 0.134, 0.195, 0.137, 0.1],
    [-10.533, -0.839, 0.814, 0.82, 0.476, -0.102, 0.129, -0.059, 0.004, -0.233, -0.093, 0.081],
    [-13.074, -8.024, -2.881, 3.028, -0.295, 0.239, -0.452, -0.226, -1.181, 0.183, 0.082, 0.237],
    [-7.734, 1.426, 1.289, 0.702, 0.316, 0.202, 0.243, 0.18, -0.089, -0.272, -0.105, 0.34],
    [2.766, 2.194, 0.603, -0.454, 0.036, -0.007, -0.449, -0.591, -0.325, 0.038, -0.214, -0.177],
    [-10.765, -1.438, 1.76, -0.436, 0.672, 0.114, -0.105, -0.353, 0.011, 0.307, 0.141, -0.047],
    [-5.101, 1.801, 0.615, 0.737, -0.422, 0.358, 0.148, 0.075, -0.046, -0.068, 0.025, 0.001],
    [-6.227, 1.751, 0.489, 0.401, 0.322, -0.21, -0.048, -0.04, 0.195, 0.251, -0.082, 0.326],
    [-5.344, 1.992, 0.694, 0.747, -0.003, 0.077, -0.011, 0.447, 0.119, 0.032, -0.038, 0.155],
    [11.713, -0.632, 0.089, 0.032, 0.703, -0.261, -0.321, -0.159, -0.005, 0.159, -0.041, 0.125],
    [-7.21, 0.892, -0.093, 1.291, -0.411, -0.394, -0.088, 0.424, 0.274, 0.188, 0.086, -0.001],
    [11.671, -4.982, 1.301, 0.436, -0.061, -0.256, 0.289, 0.288, 0.041, -0.035, 0.014, 0.004],
    [-13.336, -3.741, 1.507, 1.676, -0.281, -0.779, -0.453, 0.185, -0.429, 0.222, 0.056, 0.08],
    [-2.092, 1.798, 1.388, -0.729, -0.492, -0.413, 0.279, 0.198, -0.326, 0.125, -0.154, -0.166],
    [5.646, 2.325, 1.314, -0.138, 0.438, -0.053, -0.27, 0.122, 0.124, 0.024, -0.036, 0.063],
    [-0.116, 1.99, 1.972, 0.238, -0.153, 0.279, 0.05, 0.397, -0.196, -0.172, -0.087, -0.027],
    [-11.572, -1.772, -0.379, 1.832, -0.194, -0.222, -0.255, -0.014, -0.376, -0.255, -0.109, -0.178],
    [4.937, 1.453, -2.216, -1.943, -1.109, -0.518, 0.41, 0.659, -0.599, -0.105, 0.122, 0.292],
    [15.157, -5.66, 0.428, 0.389, 0.081, -0.401, 0.042, -0.18, 0.119, 0.015, -0.019, 0.001],
    [-12.032, -3.305, 1.942, 0.363, 0.155, 0.232, 0.022, -0.429, 0.086, 0.112, 0.278, -0.118],
    [12.699, -3.941, 0.858, 0.439, 0.111, -0.194, 0.126, -0.031, -0.002, 0.038, 0.025, 0.047],
    [7.438, 1.383, 1.059, -0.006, 0.337, 0.145, 0.16, 0.691, -0.081, 0.066, 0.007, 0.06],
    [-7.749, -3.956, -0.806, -3.447, -0.206, 0.735, -1.429, 0.865, 0.259, 0.008, -0.354, 0.307],
    [-11.111, -3.897, 0.459, -1.97, 0.721, 1.431, 0.662, -0.427, -0.167, 0.396, 0.282, -0.63],
    [-9.861, -0.78, 0.107, 1.284, 0.516, -0.27, -0.011, 0.334, -0.262, -0.31, -0.054, -0.015],
    [14.57, -5.253, 0.564, 0.401, 0.113, -0.476, -0.125, -0.488, 0.13, 0.011, -0.008, 0.017],
    [0.925, 2.538, 0.004, -1.824, -1.066, -0.538, 0.347, 0.463, -1.183, 0.444, -0.006, -0.611],
    [-2.395, 2.41, -1.051, 0.923, -0.002, -0.209, -0.122, 0.259, 0.322, -0.009, -0.074, -0.023],
    [10.554, 0.05, 0.381, 0.003, 0.699, -0.191, -0.395, -0.3, 0.013, 0.137, -0.036, 0.129],
    [-1.676, 1.81, 1.821, 0.519, -0.355, 0.131, -0.037, -0.007, -0.106, -0.207, 0.089, -0.052],
    [-9.695, -1.129, -1.274, 1.767, -0.032, -0.388, -0.243, 0.32, -0.33, -0.195, -0.056, -0.254],
    [-2.799, 2.061, 1.627, 0.278, -0.049, -0.09, -0.358, -0.232, 0.018, 0.088, 0.074, 0.081],
    [-8.101, 0.736, 1.342, 0.77, 0.164, -0.406, 0.065, 0.159, 0.09, -0.261, -0.155, 0.087],
    [-4.187, 1.865, 0.218, -0.108, 0.025, -0.163, 0.649, 0.255, 0.186, 0.083, -0.154, 0.112],
    [-0.483, 1.876, 0.772, -0.897, -0.279, -0.405, -0.032, -0.485, -0.31, 0.147, -0.148, -0.101],
    [12.548, -2.333, 0.965, 0.334, -0.182, 0.207, 0.388, 0.312, 0.07, -0.084, 0.048, -0.032],
    [-7.91, 0.731, 0.714, 0.84, 0.13, -0.133, 0.361, 0.375, 0.067, -0.198, -0.126, 0.135],
    [-1.702, 2.566, -2.306, 0.758, 0.073, -0.083, -0.028, 0.243, 0.579, 0.453, -0.138, 0.21],
    [-3.806, -0.233, -3.905, -3.726, -0.722, -0.64, 1.337, -0.37, 0.191, -0.73, 0.417, 1.166],
    [-11.771, -3.779, -2.064, 2.079, 0.488, -1.422, -0.395, 0.582, -0.09, 0.09, -0.209, -0.269],
    [-1.074, 1.615, 1.336, 0.64, -0.459, 0.112, -0.246, -0.167, 0.082, -0.042, 0.287, -0.025],
    [5.523, -5.862, 3.916, 0.757, -1.167, -0.366, 0.625, -0.025, 0.338, -0.513, 0.125, -0.344],
    [-3.062, 1.932, 0.225, -0.921, -0.207, -0.338, 0.484, -0.202, -0.088, 0.184, -0.151, 0.075],
    [-6.848, 1.6, -0.418, -1.331, 1.066, -0.216, 0.992, 0.002, -0.142, 0.345, 0.127, 0.651],
    [8.183, 3.406, -1.342, -0.853, 0.024, 0.451, -0.265, -0.071, -0.799, -0.047, -0.005, -0.141],
    [1.926, 2.451, 1.524, 0.212, 0.111, 0.143, -0.288, -0.265, 0.016, -0.297, -0.072, -0.031],
    [-4.015, 1.75, -1.32, 0.717, 0.133, -0.064, 0.238, 0.022, 0.47, 0.049, -0.143, 0.145],
    [-1.844, 2.533, 1.632, 0.373, -0.078, 0.079, -0.065, 0.417, -0.001, -0.057, 0.136, -0.015],
    [-10.602, -2.876, -4.032, 1.687, 1.3, 0.111, 0.758, -0.193, 0.433, -0.777, -0.584, -0.014],
    [-1.624, 1.327, -4.261, -2.357, 1.343, 0.049, 1.645, -0.294, 0.391, -0.353, 0.407, 0.106],
    [-4.97, 1.93, -0.086, -0.21, 0.479, -0.328, 0, -0.46, 0.35, 0.165, -0.239, 0.158],
    [14.606, -1.507, -0.529, 0.203, 0.711, -0.106, -0.094, 0.03, -0.005, 0.231, -0.022, 0.149],
    [-5.005, 0.8, -1.198, -2.775, 0.782, -0.466, 0.272, 0.241, -0.023, -0.065, 0.262, 0.032],
    [11.542, -0.199, -0.007, 0.007, 0.817, -0.187, -0.299, -0.045, -0.124, 0.197, -0.03, 0.148],
    [-8.299, -1.094, 0.542, -2.325, 0.281, 0.001, 0.299, 0.106, 0.032, -0.009, 0.209, -0.423],
    [-6.96, -0.696, -1.167, -3.256, 1.298, -0.127, 0.395, -0.376, 0.085, 0.031, 0.715, -0.222],
    [-6.846, 1.429, 1.648, 0.75, -0.275, -0.157, -0.058, 0.165, 0.073, 0.06, 0.058, 0.125],
    [-9.27, -2.158, -1.57, 0.825, 1.656, -0.292, -0.119, 0.523, -0.472, 0.149, 0.198, 0.006],
    [-7.081, 1.029, -1.549, 1.331, 0.396, -0.106, -0.093, 0.223, -0.069, -0.318, -0.19, -0.081],
    [-4.723, -0.021, -4.69, 2.427, -1.18, -0.121, 0.351, 0.409, 1.126, 0.21, 0.158, -0.257],
    [-13.315, -8.809, -3.224, 3.376, -0.968, 0.605, -0.026, 0.17, -0.975, 0.34, 0.061, 0.299],
    [7.693, 2.506, 0.131, -0.433, 0.84, -0.265, -0.652, -0.231, -0.118, 0.246, -0.048, 0.057],
    [-6.464, 0.751, 0.776, -0.671, 0.089, -0.368, 0.491, -0.143, 0.231, 0.076, -0.096, -0.073],
    [1.994, 2.067, 1.703, 0.125, 0.28, -0.014, -0.567, -0.7, 0.016, -0.249, -0.107, 0.026],
    [-7.569, 1.017, 1.295, 0.517, 0.163, -0.194, 0.324, 0.311, 0.059, -0.011, -0.108, 0.228],
    [14.749, -1.486, -0.135, 0.204, 0.23, 0.264, 0.277, 0.462, 0.061, 0.079, -0.008, 0.083],
    [-12.69, -2.668, 2.262, 0.426, 0.161, 0.88, 0.421, -0.505, -0.041, -0.037, 0.184, 0.352],
    [16.168, -2.851, -0.118, 0.251, 0.122, -0.031, 0.007, -0.251, 0.195, -0.001, 0.011, 0.003],
    [-3.484, 0.517, 0.108, -2.037, -0.661, -0.785, 0.56, -0.014, -0.18, -0.078, 0.146, 0.189],
    [-11.588, -0.752, 1.519, 1.275, -0.004, 0.127, 0.029, 0.11, -0.317, -0.295, -0.067, 0.092],
    [-5.788, 2.058, -0.593, 1.022, 0.139, -0.237, -0.359, 0.218, 0.145, 0.056, -0.022, -0.037],
    [11.123, -0.452, 0.732, 0.239, 0.145, 0.067, 0.074, 0.01, 0.102, -0.014, 0.043, 0.021],
    [2.779, 2.063, 2.181, 0.142, 0.159, -0.135, 0.027, 0.217, 0.037, -0.066, 0.057, -0.011],
    [-5.743, 0.386, -3.377, -0.319, -0.336, -0.115, 1.37, -0.293, 0.606, 0.405, -0.641, 0.236],
    [-7.304, -0.367, 0.648, -2.57, 0.363, -0.375, -0.436, 0.137, 0.014, 0.035, 0.242, -0.23],
    [9.474, 1.032, 0.202, 0.057, 0.614, 0.105, -0.516, -0.645, -0.127, -0.162, -0.12, 0.115],
    [-8.312, 0.921, 0.783, 0.085, 0.779, 0.028, 0.403, -0.12, -0.031, -0.11, -0.129, 0.23],
    [3.334, 2.608, 0.135, 0.541, -0.315, 0.601, -0.067, 0.027, 0.006, -0.282, 0.023, -0.194],
    [1.89, 1.066, -4.628, -2.742, -0.382, -0.555, 0.167, 0.1, 0.4, -0.607, -0.217, 0.55],
    [-10.539, -1.557, 3.22, -1.802, -0.017, -0.403, -0.174, -0.108, -0.259, 0.504, 0.29, -0.14],
    [14.28, -1.149, 0.004, 0.159, 0.212, 0.06, 0.006, -0.089, 0.189, 0.01, 0.006, 0.012],
    [-10.176, -4.027, 0.483, -1.684, 0.095, 0.391, -0.858, 0.044, 0.405, 0.739, -0.148, -0.708],
    [4.598, 1.797, 0.785, 0.247, 0.107, 0.582, -0.124, -0.185, -0.214, -0.311, -0.137, 0.027],
    [-7.974, -0.556, 0.23, -2.491, 0.564, -0.172, 0.171, -0.07, 0.015, 0.025, 0.19, -0.301],
    [-10.96, -3.936, -2.455, 2.214, 0.462, -0.177, -0.272, 0.072, -0.774, -0.304, 0.051, -0.211],
    [15.313, -4.044, 0.528, 0.363, -0.232, 0.084, 0.368, 0.224, 0.163, -0.123, 0.019, -0.052],
    [-12.798, -4.783, 1.638, 1.423, -0.321, 0.81, 0.362, -0.481, -0.332, -0.076, 0.177, 0.769],
    [-9.374, 0.369, 1.246, 1.079, -0.1, -0.27, -0.174, -0.041, -0.059, -0.259, -0.156, -0.008],
    [-2.37, 2.39, -1.282, -0.209, 0.249, -0.091, 0.82, 0.61, 0.232, 0.183, -0.21, 0.159],
    [-11.695, -6.202, -0.147, -0.178, 0.095, 1.455, 0.037, -0.254, 0.284, 0.919, -0.005, -0.331],
    [0.622, 3.036, -0.251, 0.667, -0.224, 0.284, -0.06, 0.243, 0.224, 0.103, 0.196, 0.001],
    [-0.651, 3.15, -0.229, 0.709, -0.228, 0.032, -0.226, 0.024, 0.383, 0.067, 0.182, -0.09],
    [-7.962, 1.144, 0.564, 0.59, 0.526, -0.12, -0.118, -0.24, 0.013, -0.203, -0.158, 0.221],
    [-4.769, 2.169, 1.062, 0.635, -0.036, -0.132, -0.356, -0.326, 0.098, 0.017, 0.069, 0.108],
    [-7.641, 0.807, -2.213, 1.15, 1.1, 0.371, -0.066, -0.21, -0.29, -0.36, -0.064, 0.302],
    [-11.088, -2.332, 0.617, 0.583, 0.505, 0.185, -0.093, -0.49, -0.017, -0.063, 0.226, -0.074],
    [1.451, 2.39, -1.49, 0.997, -0.587, 0.488, -0.438, -0.487, 0.417, 0.086, 0.466, -0.089],
    [11.482, -5.807, 1.655, 0.49, -0.369, -0.381, 0.338, 0.137, 0.201, -0.17, 0.022, -0.111],
    [-9.098, -0.2, -0.648, 1.535, -0.056, -0.301, -0.225, 0.05, -0.121, -0.318, -0.144, -0.234],
    [-11.967, -3.062, -1.088, 2.026, 0.007, -0.498, -0.144, 0.533, -0.515, 0.004, 0.058, -0.125],
    [-3.128, 2.47, 0.138, 0.436, -0.066, 0.095, 0.361, 0.586, 0.196, 0.084, -0.094, 0.118],
    [-6.539, -1.658, -0.618, -3.237, 0.137, -0.26, -0.668, 0.49, 0.205, -0.403, 0.464, -0.075],
    [-8.409, 0.812, 1.673, 0.91, -0.132, -0.405, -0.108, 0.06, 0.088, -0.256, -0.123, -0.054],
    [15.56, -3.877, -0.328, 0.205, 0.726, -0.402, -0.381, -0.424, -0.104, 0.284, -0.07, 0.239],
    [6.395, 0.789, 1.599, 0.079, 0.202, -0.247, -0.238, -0.265, 0.23, -0.069, 0.011, -0.014],
    [19.335, -1.751, -1.241, 0.131, 0.409, 0.32, 0.121, 0.133, 0.03, 0.153, 0.001, 0.116],
    [3.624, 2.273, 1.2, 0.31, -0.064, 0.32, 0.011, -0.068, 0.001, -0.403, -0.05, -0.104],
    [14.083, -1.181, -0.376, 0.086, 0.61, -0.024, -0.032, 0.204, 0.042, 0.199, -0.053, 0.127],
    [2.901, 2.446, -1.852, -1.119, -0.625, 0.204, 0.737, -0.181, -0.259, 0.197, -0.27, -0.133],
    [6.077, -6.02, 3.607, 0.725, -0.98, -0.466, 0.518, -0.078, 0.313, -0.438, 0.107, -0.298],
    [-9.095, 0.416, 0.878, 0.807, 0.318, -0.021, 0.268, 0.129, -0.021, -0.29, -0.136, 0.186],
    [-11.498, -2.841, 1.578, -0.174, 0.788, 0.466, 0.06, 0.065, -0.236, 0.463, 0.179, 0.039],
    [13.591, -1.189, 0.462, 0.306, -0.254, 0.383, 0.639, 0.71, 0.168, -0.1, 0.057, -0.097],
    [-1.818, 2.702, 0.581, 0.612, -0.111, 0.315, -0.152, 0.156, -0.066, -0.095, 0.125, 0.019],
    [-1.036, 3.328, -0.55, 0.629, 0.051, 0.115, -0.426, -0.137, 0.295, 0.103, 0.088, 0.045],
    [2.66, 2.463, 0.856, 0.247, 0.094, 0.238, -0.385, -0.279, -0.053, -0.128, 0.047, 0.017],
    [-10.755, -2.658, -1.41, 1.178, 1.064, 0.701, 0.146, -0.128, -0.335, -0.183, 0.298, 0.127],
    [16.139, -2.189, -0.509, 0.04, 0.532, -0.021, -0.272, -0.266, 0.109, 0.146, -0.071, 0.176],
    [-8.956, -1.873, -1.869, -4.172, 1.417, 0.002, 1.866, -0.593, -0.202, 0.241, 1.042, 0.057],
    [-6.636, -5.41, -4.765, -4.22, 1.283, 1.617, -0.996, 0.938, 0.556, -0.727, 0.045, -0.307],
    [11.548, -5.855, 2.109, 0.657, -0.82, -0.146, 0.687, 0.253, 0.268, -0.335, 0.088, -0.258],
    [-3.622, 1.139, 0.053, -1.746, -0.145, -0.562, 0.375, -0.154, 0.059, 0.016, -0.026, -0.105],
    [-2.648, 2.074, -1.447, 1.385, -0.756, 0.692, -0.424, -0.534, 0.156, -0.176, 0.363, -0.112],
    [-10.775, -2.084, -3.398, 2.247, 0.316, 0.165, -0.435, 0.118, -0.614, -0.003, -0.038, -0.053],
    [-1.632, 2.521, -1.439, 0.839, -0.28, 0.114, 0.378, 0.609, 0.323, 0.288, 0.025, 0.199],
    [11.852, -3.499, 1.186, 0.295, -0.089, -0.223, -0.01, -0.274, 0.186, -0.081, 0.001, -0.03],
    [3.35, 1.638, 1.739, -0.202, -0.022, -0.056, 0.167, 0.652, -0.401, 0.173, 0.044, -0.122],
    [-1.126, 0.078, -2.469, -2.797, -0.29, -0.876, -0.488, 0.241, 0.237, -0.631, 0.067, 0.245],
    [12.103, -0.433, 0.237, 0.179, 0.414, -0.051, -0.042, -0.019, 0.077, 0.056, 0.018, 0.057],
    [-1.416, 2.064, -0.887, 0.966, -0.494, 0.252, -0.356, -0.299, 0.397, 0.236, 0.446, 0.046],
    [-11.752, -1.463, 0.81, 1.388, -0.07, 0.146, -0.359, -0.448, -0.319, -0.333, -0.108, -0.074],
    [-9.461, -1.234, 0.685, -0.853, 0.594, -0.068, 0.481, -0.233, 0.234, 0.064, 0.043, -0.457],
    [-9.092, -0.175, 0.755, 1.202, -0.1, -0.91, -0.423, -0.036, 0.236, -0.058, -0.154, -0.079],
    [-4.412, 1.694, 1.712, 0.638, -0.355, 0.014, -0.246, 0.01, 0.124, 0.03, 0.211, 0.053],
    [-6.305, -0.781, -1.375, -2.366, 0.942, 0.093, -0.161, 0.076, 0.176, -0.061, 0.063, -0.583],
    [-3.688, 1.79, -1.244, 0.836, -0.347, -0.14, -0.185, -0.438, 0.486, 0.439, 0.156, 0.14],
    [-8.037, 0.01, 0.795, -0.384, 0.465, -0.394, 0.293, -0.286, 0.29, 0.007, 0.083, -0.165],
    [4.494, 2.707, 0.207, -0.472, 0.206, 0.162, -0.286, -0.076, -0.372, 0.112, -0.176, -0.124],
    [-5.252, 1.98, -0.886, 0.957, 0.321, -0.268, -0.355, -0.126, 0.267, -0.098, -0.155, -0.023],
    [-7.349, -0.928, 0.747, -2.308, 0.01, -0.606, -0.279, -0.453, 0.275, -0.197, 0.289, -0.309],
    [7.485, 1.853, 0.72, -0.043, 0.485, 0.038, -0.35, -0.243, 0.029, -0.099, -0.076, 0.051],
    [-4.718, 1.236, -0.367, 0.42, 0.356, -0.739, -0.124, -0.418, 0.42, 0.048, -0.237, 0.123],
    [4.562, 2.472, 0.661, 0.094, -0.041, 0.496, 0.158, 0.405, -0.26, -0.086, -0.067, -0.1],
    [13.708, -6.24, 1.111, 0.562, -0.229, -0.49, 0.194, -0.194, 0.162, -0.099, 0.025, -0.087],
    [-8.743, 0.764, 2.11, 0.855, -0.23, -0.075, -0.126, 0.074, 0.038, 0.002, 0.019, 0.225],
    [-6.727, -1.337, -0.966, -3.323, 0.508, -0.026, -0.046, 0.675, 0.06, -0.278, 0.635, -0.06],
    [14.395, -3.435, 0.268, 0.262, 0.126, -0.213, -0.005, -0.143, 0.18, 0.035, -0.012, 0.013],
    [12.865, -5.455, 0.645, 0.385, 0.391, -0.694, -0.23, -0.363, 0.016, 0.159, -0.041, 0.114],
    [2.277, 1.335, 0.502, -1.379, -0.566, -0.547, 0.126, 0.498, -0.718, 0.317, -0.111, -0.256],
    [10.208, -0.992, 0.728, 0.12, 0.488, -0.33, -0.335, -0.353, 0.08, 0.087, -0.024, 0.092],
    [-0.208, 1.729, -1.742, -0.687, -0.838, 0.098, 0.178, -1.154, -0.467, 0.149, -0.37, -0.297],
    [-5.916, 1.621, 0.879, 0.62, -0.122, -0.105, 0.122, 0.406, 0.172, 0.139, -0.085, 0.167],
    [-4.928, 1.619, 0.09, 0.947, -0.062, -0.409, -0.418, -0.225, 0.378, -0.031, -0.034, -0.027],
    [-2.385, 2.533, 0.141, 0.805, -0.446, 0.139, -0.095, 0.333, 0.359, 0.083, 0.134, -0.059],
    [-4.99, 0.684, -1.212, -2.303, 0.485, -0.382, 0.451, -0.194, 0.098, -0.114, -0.063, -0.112],
    [-9.784, -4.087, 0.39, -2.802, -0.166, 0.938, -0.761, 0.267, 0.308, 0.16, -0.205, -0.373],
    [-0.807, 2.479, -2.471, 1.345, -0.602, 0.248, 0.033, 0.239, 0.519, 0.454, 0.343, 0.088],
    [-10.515, -0.96, -0.481, 1.616, 0.007, 0.093, -0.05, 0.052, -0.382, -0.401, -0.122, -0.103],
    [16.166, -3.081, -0.925, -0.109, -0.073, -0.292, -0.212, -0.176, -0.279, 0.119, 0.012, -0.08],
    [4.489, 3.446, -0.451, 0.285, 0.424, 0.666, -0.428, -0.659, -0.179, -0.366, -0.167, 0.064],
    [0.026, 2.47, -1.042, -0.942, -0.15, -0.41, 0.017, -0.835, 0.171, 0.379, -0.118, 0.122],
    [-6.052, 0.234, 0.831, -2.129, -0.382, -0.652, 0.513, -0.14, -0.198, 0.109, 0.132, 0.14],
    [-9.056, -0.701, -0.258, 1.413, 0.132, -0.7, 0.09, 0.594, 0.037, -0.163, -0.216, -0.064],
    [-7.494, 1.015, 1.132, 0.901, -0.026, -0.313, -0.028, 0.3, 0.022, -0.197, -0.131, -0.005],
    [-8.661, 0.283, 1.417, 1.062, -0.199, -0.622, -0.184, 0.204, 0.021, -0.143, -0.103, -0.071],
    [-4.322, 2.579, -0.223, 0.774, 0.171, -0.036, -0.343, 0.149, 0.164, 0.147, 0.007, 0.107],
    [-4.287, 0.647, -0.471, -0.901, -0.558, -0.486, 0.732, -0.388, 0.311, 0.285, -0.286, 0.012],
    [-0.65, 2.525, -0.904, -0.291, -0.24, -0.04, -0.043, -0.412, 0.095, 0.422, -0.036, -0.002],
    [-7.974, -0.685, -2.131, -0.151, 0.928, -0.291, 0.873, -0.285, 0.565, -0.113, -0.481, -0.161],
    [6.165, 0.398, 0.318, -0.873, -0.471, -0.601, -0.438, -0.45, -0.462, 0.164, 0.027, -0.313],
    [14.53, -5.363, 0.403, 0.44, 0.317, -0.512, -0.117, -0.352, 0.003, 0.105, -0.01, 0.086],
    [-7.17, 0.116, -1.628, -0.146, 1.387, -0.282, 0.102, -0.099, 0.12, 0.074, 0.009, -0.039],
    [-3.094, 2.276, 0.626, 0.669, -0.11, -0.231, -0.534, -0.445, 0.402, 0.004, 0.047, -0.006],
    [-3.626, 1.699, 1.79, 0.578, -0.297, -0.061, -0.252, -0.08, 0.092, 0.043, 0.23, 0.052],
    [6.787, 1.089, 1.166, 0.065, 0.34, 0.045, 0.235, 0.745, -0.21, 0.113, 0.052, 0.05],
    [-8.906, -4.082, 0.14, -3.045, -0.443, 0.742, -1.241, 0.609, 0.298, 0.227, -0.537, 0.098],
    [4.006, 1.242, -0.985, -0.722, -0.992, 0.376, 0.37, -0.454, -0.77, -0.088, -0.305, -0.48],
    [17.012, -5.026, -0.683, 0.289, 0.71, -0.457, -0.202, -0.066, -0.154, 0.351, -0.065, 0.228],
    [15.944, -6.899, 0.252, 0.482, 0.172, -0.622, -0.005, -0.28, 0.11, 0.06, -0.017, 0.01],
    [5.884, 2.396, 0.376, -0.772, -0.266, -0.322, -0.311, -0.075, -0.224, 0.119, -0.03, -0.363],
    [-10.321, -6.959, -4.777, 0.322, 2.28, 0.777, 0.01, 0.278, -0.562, 0.593, 0.483, -0.663],
    [-1.041, 1.92, 1.269, 0.15, -0.467, 0.166, 0.266, 0.452, -0.242, 0.061, 0.011, -0.082],
    [14.155, -0.086, -0.496, 0.016, 0.73, -0.064, -0.312, -0.279, 0.057, 0.169, -0.032, 0.148],
    [5.996, 0.268, -3.327, -2.071, -1.53, -0.175, 0.323, 0.567, -0.659, -0.426, -0.034, 0.389],
    [0.489, 1.701, 0.735, -0.757, -0.544, -0.12, 0.197, -0.114, -0.49, 0.114, -0.221, -0.232],
    [15.662, -1.548, -0.312, 0.105, 0.359, -0.016, -0.213, -0.406, 0.154, 0.069, -0.015, 0.085],
    [6.69, 2.341, 0.691, 0.036, 0.359, 0.313, -0.223, -0.403, 0.039, -0.198, 0.014, 0.219],
    [4.34, -0.685, -1.135, -1.641, -1.436, -0.526, -0.223, -0.016, -0.673, -0.263, -0.028, 0.083],
    [5.143, 1.162, -0.45, -1.488, -0.964, -0.308, 0.312, 0.516, -1.006, 0.232, -0.009, -0.327],
    [11.311, 0.216, 0.149, -0.074, 0.613, -0.025, -0.212, 0.055, 0.044, 0.147, -0.06, 0.105],
    [7.228, 2.389, 0.232, 0.027, 0.473, 0.337, -0.417, -0.451, -0.025, -0.22, -0.122, 0.057],
    [-6.973, -1.635, -2.124, -1.908, -0.242, 0.081, 1.596, -0.59, 0.406, -0.38, -0.042, -0.362],
    [-13.189, -4.558, 1.9, 1.474, -0.041, 0.344, 0.008, -0.401, -0.506, -0.141, 0.211, 0.482],
    [-4.636, 2.245, -1.473, 0.811, 0.739, -0.249, -0.252, 0.05, 0.263, 0.064, -0.16, 0.202],
    [1.262, 2.258, -0.118, -1.592, -0.268, -0.535, 0.036, -0.179, -0.587, 0.302, -0.13, -0.068],
    [15.763, -2.725, -0.073, 0.25, 0.182, 0.206, 0.211, 0.241, 0.037, 0.055, 0.002, 0.08],
    [7.202, 3.317, -1.319, 0.234, 0.354, 0.848, -0.394, -0.717, -0.436, -0.367, -0.225, 0.017],
    [2.697, 1.691, -1.385, -2.04, -0.675, -0.624, 0.045, 0.154, -0.603, 0.032, -0.048, 0.177],
    [1.027, -0.335, -4.421, -2.897, -0.711, -0.533, 0.403, 0.664, 0.415, -1.048, 0.222, 0.503],
    [-2.774, 2.29, 1.023, 0.69, -0.357, -0.004, -0.138, 0.065, 0.217, 0.029, 0.164, -0.016],
    [5.907, 1.79, 1.366, -0.013, 0.299, 0.063, 0.147, 0.685, 0.024, 0.038, 0.005, 0.014],
    [-9.836, -2.669, -1.175, -1.067, 1.32, 1.206, 0.696, -0.885, 0.189, 0.024, 0.102, -0.528],
    [9.789, 1.777, 0.101, -0.121, 0.596, 0.294, -0.38, -0.303, 0.013, -0.163, -0.147, 0.078],
    [-6.556, 1.588, 0.595, 0.994, -0.262, -0.331, -0.439, -0.165, 0.301, 0.191, 0.129, 0.037],
    [-8.779, 0.016, 0.476, 0.786, 0.547, -0.538, 0.026, -0.019, 0.129, -0.159, -0.225, 0.228],
    [-7.306, 0.637, -0.348, -0.147, 0.629, -0.338, 0.257, -0.644, 0.428, -0.042, -0.331, -0.052],
    [-14.476, -7.053, 3.395, 2.062, -1.38, -0.6, 0.003, -0.068, -0.115, 0.082, 0.162, -0.023],
    [7.441, 1.852, 0.833, 0.033, 0.484, 0.165, -0.025, 0.301, -0.148, -0.019, 0.048, 0.15],
    [-10.813, -0.53, 3.114, 0.297, -0.049, -0.272, 0.249, -0.257, -0.046, -0.236, -0.019, 0.164],
    [-12.419, -2.301, -0.041, 1.724, 0.123, 0.857, 0.289, -0.19, -0.574, -0.521, 0.091, 0.307],
    [-3.414, 2.196, 1.84, 0.506, -0.145, -0.219, -0.382, -0.253, 0.142, 0.032, 0.141, 0.04],
    [-0.281, 1.935, -2.9, -1.305, -0.497, 0.051, 1.095, -0.773, 0.069, 0.158, -0.357, 0.105],
    [2.786, 2.405, 0.145, 0.408, 0.056, 0.295, -0.673, -0.634, -0.059, -0.032, 0.149, 0.07],
    [1.11, 2.21, -0.433, -1.517, -0.626, -0.019, 0.891, 0.794, -0.779, 0.255, -0.178, -0.057],
    [11.479, -4.418, 1.692, 0.537, -0.467, -0.098, 0.452, 0.134, 0.169, -0.207, 0.067, -0.143],
    [2.931, 1.979, -0.221, -0.49, -0.364, 0.287, -0.265, -0.663, -0.474, 0.048, -0.304, -0.305],
    [11.177, -0.604, 0.594, 0.065, 0.392, -0.124, -0.313, -0.366, 0.139, 0.039, -0.036, 0.109],
    [-4.811, -3.627, -5.831, -4.62, 0.774, 0.147, -0.772, 0.834, 0.887, -1.979, 0.483, 0.111],
    [16.04, -0.918, -0.934, -0.005, 0.726, -0.099, -0.383, -0.391, -0.095, 0.277, -0.028, 0.192],
    [-0.028, 3.217, 0.431, 0.407, 0.092, 0.058, -0.473, -0.326, 0.178, -0.029, 0.117, 0.006],
    [-6.756, 1.073, 0.079, -0.001, 0.498, -0.159, 0.545, -0.391, 0.23, -0.164, -0.397, 0.066],
    [11.362, 1.138, 0.162, -0.01, 0.451, 0.137, -0.052, 0.145, 0.105, 0.071, -0.006, 0.064],
    [-3.317, 1.314, -0.855, -0.539, 0.008, -0.677, 0.205, -0.449, 0.415, 0.299, -0.227, -0.028],
    [-0.765, 1.201, 1.223, -0.529, -0.892, 0.033, 0.426, 0.124, -0.427, 0.073, -0.312, -0.336],
    [12.545, -2.394, 0.443, 0.081, 0.466, -0.306, -0.454, -0.571, 0.126, 0.112, -0.069, 0.154],
    [-7.135, -0.611, -2.887, 1.727, 0.392, -1.257, -0.503, -0.117, 0.436, -0.15, -0.356, -0.268],
    [16.412, -5.001, 0.029, 0.39, 0.128, -0.385, -0.063, -0.418, 0.159, 0.034, -0.004, 0.012],
    [-10.857, -3.836, 1.874, -2.754, -0.291, 0.662, -0.599, 0.153, 0.192, 0.345, -0.035, -0.143],
    [-7.882, 0.364, -1.261, 1.344, 0.136, -0.243, 0.198, 0.499, 0.316, -0.126, -0.223, -0.05],
    [-8.922, -1.806, 1.219, -2.775, 0.073, -0.078, -0.247, 0.393, -0.038, 0.154, 0.433, -0.185],
    [-7.272, 0.827, 1.439, 0.599, 0.006, -0.486, -0.144, -0.256, 0.31, 0.033, -0.088, 0.134],
    [-6.179, 1.756, 0.794, 0.748, -0.041, -0.061, 0.154, 0.294, 0.065, 0.013, -0.012, 0.178],
    [14.453, -3.42, 0.599, 0.452, -0.252, 0.26, 0.666, 0.647, 0.086, -0.076, 0.06, -0.082],
    [-11.695, -1.987, -0.37, 1.825, -0.022, 0.284, 0.161, -0.003, -0.514, -0.507, -0.019, 0.002],
    [13.117, -5.22, 0.945, 0.348, 0.065, -0.531, -0.181, -0.456, 0.125, -0.017, -0.031, 0.036],
    [-1.183, 1.619, -1.025, -1.768, -0.493, -0.365, 0.547, -0.079, -0.31, 0.145, -0.256, 0.115],
    [-7.324, 1.045, 1.023, -1.239, 0.12, -0.399, 0.741, -0.211, -0.336, 0.347, -0.063, 0.428],
    [8.529, 1.114, 0.715, -0.085, -0.336, 0.179, 0.372, 0.579, -0.319, 0.026, 0.069, -0.229],
    [3.269, 2.164, 1.925, 0.021, 0.323, -0.043, 0.07, 0.584, -0.048, 0.054, 0.029, 0.049],
    [-1.361, 1.935, 1.816, 0.612, -0.393, 0.374, 0.09, 0.057, -0.125, -0.45, -0.064, -0.149],
    [-2.069, 2.686, 0.951, 0.531, -0.122, 0.216, -0.011, 0.648, -0.068, 0.04, 0.213, 0.014],
    [-6.757, 0.88, 1.35, -1.423, -0.054, -0.529, 0.703, 0.312, -0.321, 0.371, 0.179, 0.255],
    [9.708, 0.307, 0.075, -0.444, -0.325, -0.075, 0.158, 0.444, -0.382, 0.083, 0.032, -0.29],
    [-5.003, 0.991, -2.934, 1.406, -0.947, -0.098, 0.268, -0.016, 0.835, 0.65, 0.178, 0.025],
    [-2.671, 2.661, -0.868, 1.011, -0.294, 0.053, -0.215, 0.003, 0.327, 0.112, 0.171, -0.013],
    [-7.544, 1.107, 0.372, 0.785, 0.394, -0.201, 0.14, 0.382, 0.019, -0.26, -0.155, 0.058],
    [-4.108, 1.2, -0.402, -0.034, -0.164, -0.523, 0.229, -0.331, 0.485, 0.276, -0.198, 0.055],
    [-8.809, 0.068, 1.712, 1.041, -0.247, -0.695, -0.085, 0.092, 0.161, -0.258, -0.123, -0.102],
    [-10.744, -0.177, 2.323, 1.022, -0.28, -0.631, -0.397, 0.067, 0.017, 0.095, -0.019, 0.083],
    [2.572, 3.336, -0.709, 0.524, -0.084, 0.358, -0.469, -0.527, 0.298, 0.092, 0.176, 0.103],
    [-10.118, -0.619, 1.923, 0.916, 0.162, -0.736, -0.104, -0.088, 0.116, -0.296, -0.186, 0.067],
    [-8.265, -3.224, -4.117, -3.121, 1.443, 0.874, 2.091, -0.498, 0.355, -0.631, 0.259, -0.734],
    [16.571, -4.883, -0.524, 0.311, 0.734, -0.425, -0.218, -0.193, -0.132, 0.306, -0.054, 0.224],
    [-6.14, 1.132, 0.5, -1.038, 0.432, -0.371, 0.303, -0.451, 0.074, 0.076, 0.021, 0.001],
    [-4.587, 2.137, -0.61, -0.155, 0.672, -0.213, 0.283, -0.225, 0.274, 0.142, -0.17, 0.263],
    [-4.562, 1.85, 1.098, 0.653, -0.322, 0.013, 0.141, 0.334, 0.113, 0.118, 0.03, 0.187],
]


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
        if lab.time_remaining > cfg["time_guard"]:  # little time left: keep the last fits
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

# Plate titer (log, standard plate medium: glucose at its upper bound, glutamine at 5 % of its range) of each
# library member at its own temperature/pH optimum, pre-computed with the simulator (feature for feat_prior).
GLN_PLATE = [8.82, 9.74, 5.90, 9.26, 5.96, 8.86, 9.10, 7.60, 8.91, 9.35, 7.54, 9.52, 9.23, 5.58, 8.69, 8.62, 6.62, 9.10, 7.14, 8.29, 9.50, 9.02, 5.58, 8.53, 8.91, 8.57, 8.91, 6.45, 8.29, 8.67, 7.47, 3.88, 8.71, 4.88, 9.66, 9.51, 8.44, 8.59, 8.79, 7.30, 8.61, 8.63, 8.67, 7.58, 9.67, 9.08, 9.26, 8.96, 7.84, 9.25, 9.03, 9.14, 9.16, 8.43, 8.67, 2.27, 8.90, 7.18, 6.47, 8.09, 9.42, 8.82, 4.08, 9.32, 5.58, 7.85, 8.86, 9.00, 9.07, 5.64, 8.95, 9.36, 8.31, 7.31, 9.29, 8.41, 8.87, 8.13, 9.15, 4.89, 8.97, 9.42, 7.18, 8.48, 7.79, 4.45, 8.25, 9.22, 8.18, 7.89, 9.31, 9.05, 8.65, 7.36, 8.94, 8.27, 9.42, 8.40, 8.62, 7.55, 9.49, 9.01, 8.26, 6.42, 8.27, 8.47, 9.09, 8.09, 8.20, 7.63, 9.20, 5.49, 8.68, 8.67, 8.21, 6.49, 6.47, 8.99, 8.83, 8.50, 9.06, 8.35, 9.01, 8.27, 5.96, 8.69, 8.33, 9.03, 9.44, 5.72, 9.50, 9.45, 8.19, 9.00, 7.85, 8.82, 9.59, 9.39, 9.13, 9.64, 8.17, 6.69, 9.01, 9.12, 7.76, 8.63, 9.08, 5.36, 7.24, 5.27, 8.90, 7.85, 8.21, 5.79, 9.50, 9.01, 6.08, 8.23, 8.49, 8.35, 8.70, 5.27, 8.20, 8.59, 4.35, 8.53, 7.76, 9.11, 8.58, 7.47, 8.61, 8.58, 6.53, 7.65, 9.00, 8.89, 8.44, 9.70, 8.73, 8.35, 9.02, 8.46, 8.65, 8.92, 7.80, 9.25, 7.96, 5.03, 8.91, 8.52, 6.19, 3.98, 7.45, 7.23, 8.60, 8.27, 8.30, 8.28, 8.61, 8.82, 8.59, 9.15, 5.89, 7.94, 8.54, 8.90, 8.28, 8.88, 8.79, 8.43, 9.07, 8.08, 9.36, 7.20, 4.07, 9.09, 8.35, 8.87, 9.16, 9.04, 7.58, 3.95, 4.39, 6.40, 9.50, 8.46, 6.64, 7.64, 8.54, 5.58, 8.54, 7.94, 7.60, 6.88, 7.44, 8.33, 9.78, 8.87, 7.47, 4.62, 8.10, 8.32, 7.84, 8.07, 8.71, 7.85, 8.08, 7.88, 9.28, 8.62, 8.92, 8.55, 9.16, 8.94, 7.74, 7.93, 8.75, 8.87, 4.68, 7.36, 5.67, 8.78, 6.27, 8.45, 8.85, 7.43, 9.31, 8.85, 5.94, 8.55, 8.46, 9.26, 8.35, 9.27, 9.22, 8.18, 8.68, 9.38, 4.82, 9.53, 9.53, 6.71, 8.88, 7.49, 8.10, 9.34, 8.77, 8.79, 8.79, 9.03, 7.98, 8.15, 8.96, 7.42, 9.46, 8.98, 4.18, 8.75, 9.38, 9.29]

CFG.update({"carry_plates": 0, "n_space_plates": 1})
