"""Offline Q2 (generic track): sequential Pilot glutamine search on library members from the generic centre recipe.
(a) GP expected improvement in log-glutamine space vs (b) refit textbook kinetic model v2 (weak Gaussian prior on log
parameters around the box centre) and take its predicted optimum. Reports mean relative titer of the best of 1..4 runs."""
import numpy as np, sys, json
from multiprocessing import Pool
from scipy.optimize import least_squares
from scipy.stats import norm
sys.path.insert(0, "dbg")
from kin_model2 import simulate, LO, HI
from sim import load_scenario
D = {k: v for k, v in np.load("gln_lib.npz").items()}
A0, A1, B2 = D["A0"], D["A1"], D["B2"]
a0 = np.repeat(A0, 84); a1 = np.tile(np.repeat(A1, 7), 13); b2 = np.tile(B2, 156)
Z = np.c_[np.log10(a0), np.log10(a1), b2 / 48.0]
LS = np.array([0.6, 0.6, 0.3])
N_PILOT = 6


def gp_next(idx, y, cand):
    X = Z[idx]; yy = np.array(y); m = yy.mean(); s = max(yy.std(), 0.3)
    k = lambda A, B: s ** 2 * np.exp(-0.5 * (((A[:, None, :] - B[None, :, :]) / LS) ** 2).sum(2))
    K = k(X, X) + 1e-4 * np.eye(len(X)); Ks = k(Z[cand], X)
    L = np.linalg.cholesky(K); al = np.linalg.solve(L.T, np.linalg.solve(L, yy - m))
    mu = m + Ks @ al; v = np.linalg.solve(L, Ks.T); sd = np.sqrt(np.maximum(s ** 2 - (v ** 2).sum(0), 1e-12))
    z = (mu - yy.max()) / sd; ei = (mu - yy.max()) * norm.cdf(z) + sd * norm.pdf(z)
    ei[np.isin(cand, idx)] = -1
    return cand[np.argmax(ei)]


def kin_next(idx, y, cand, hi, rng):
    c0 = (LO + HI) / 2; sd0 = (HI - LO) / 4
    def res(th):
        p = np.log(np.maximum(simulate(th, hi[2], a0[idx], hi[4], a1[idx], hi[6], b2[idx]), 1e-12))
        r = p - np.array(y)
        return np.r_[(r - r.mean()) / 0.1, (th - c0) / sd0]
    best = None
    for s in range(4):
        th0 = np.clip(c0 + sd0 * rng.standard_normal(len(c0)) * 0.7, LO, HI)
        f = least_squares(res, th0, bounds=(LO, HI), max_nfev=150, diff_step=1e-3)
        if best is None or f.cost < best.cost: best = f
    p = simulate(best.x, hi[2], a0[cand], hi[4], a1[cand], hi[6], b2[cand])
    p[np.isin(cand, idx)] = -1
    return cand[np.argmax(p)]


def kin_fit(idx, y, hi, rng):
    c0 = (LO + HI) / 2; sd0 = (HI - LO) / 4
    def res(th):
        p = np.log(np.maximum(simulate(th, hi[2], a0[idx], hi[4], a1[idx], hi[6], b2[idx]), 1e-12))
        r = p - np.array(y)
        return np.r_[(r - r.mean()) / 0.1, (th - c0) / sd0]
    best = None
    for s in range(4):
        th0 = np.clip(c0 + sd0 * rng.standard_normal(len(c0)) * 0.7, LO, HI)
        f = least_squares(res, th0, bounds=(LO, HI), max_nfev=150, diff_step=1e-3)
        if best is None or f.cost < best.cost: best = f
    return best.x


def hyb_next(idx, y, cand, hi, rng):
    """GP on the residual of the fitted kinetic model (physics prior mean), EI on log titer."""
    th = kin_fit(idx, y, hi, rng)
    pm = np.log(np.maximum(simulate(th, hi[2], a0, hi[4], a1, hi[6], b2), 1e-12))
    yy = np.array(y); off = np.mean(yy - pm[idx]); base = pm + off
    r = yy - base[idx]; s = 0.3
    X = Z[idx]
    k = lambda A, B: s ** 2 * np.exp(-0.5 * (((A[:, None, :] - B[None, :, :]) / LS) ** 2).sum(2))
    K = k(X, X) + 1e-4 * np.eye(len(X)); Ks = k(Z[cand], X)
    L = np.linalg.cholesky(K); al = np.linalg.solve(L.T, np.linalg.solve(L, r))
    mu = base[cand] + Ks @ al; v = np.linalg.solve(L, Ks.T); sd = np.sqrt(np.maximum(s ** 2 - (v ** 2).sum(0), 1e-12))
    z = (mu - yy.max()) / sd; ei = (mu - yy.max()) * norm.cdf(z) + sd * norm.pdf(z)
    ei[np.isin(cand, idx)] = -1
    return cand[np.argmax(ei)]


def run(j):
    sc = load_scenario(int(D["sid"][j])); b = np.array(sc["bounds"]); lo, hi = b[:, 0], b[:, 1]
    ok = (a0 <= hi[3] * 1.0001) & (a1 <= hi[5] * 1.0001) & (b2 <= hi[7] * 1.0001)
    cand = np.where(ok)[0]
    R = D["Y"][j] / D["Y"][j][cand].max(); ly = np.log(np.maximum(D["Y"][j], 1e-9))
    q = (lo[3] + 10 ** -1.65 * (hi[3] - lo[3]), lo[5] + 10 ** -1.65 * (hi[5] - lo[5]), lo[7] + 0.5 * (hi[7] - lo[7]))
    start = cand[np.argmin(((Z[cand] - [np.log10(q[0]), np.log10(q[1]), q[2] / 48]) ** 2).sum(1))]
    out = {}
    pm0 = simulate((LO + HI) / 2, hi[2], a0[cand], hi[4], a1[cand], hi[6], b2[cand])
    phys = int(cand[np.argmax(pm0)])
    for name in ("gp", "kin2gp", "rnd2gp", "phys1gp", "phys1kin2gp"):
        st0 = phys if name.startswith("phys1") else start
        rng = np.random.default_rng(j); idx = [st0]; y = [ly[st0]]
        while len(idx) < N_PILOT:
            if name in ("kin2gp", "phys1kin2gp") and len(idx) == 1:
                nx = kin_next(idx, y, cand, hi, rng)
            elif name == "rnd2gp" and len(idx) == 1:
                nx = int(rng.choice(cand))
            elif name == "far2gp" and len(idx) == 1:
                nx = cand[np.argmax((((Z[cand] - Z[idx[0]]) / LS) ** 2).sum(1))]
            else:
                nx = gp_next(idx, y, cand)
            idx.append(int(nx)); y.append(ly[nx])
        out[name] = [float(max(R[idx[:k + 1]])) for k in range(N_PILOT)]
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    js = np.random.default_rng(2).choice(len(D["Y"]), n, replace=False)
    with Pool(20) as p:
        res = p.map(run, js)
    for name in ("gp", "kin2gp", "rnd2gp", "phys1gp", "phys1kin2gp"):
        M = np.array([r[name] for r in res])
        print("%-4s best of 1..%d: %s" % (name, N_PILOT, " ".join("%.3f" % v for v in M.mean(0))))
    for name in ("gp", "rnd2gp"):
        for k in (0, 1, 3, N_PILOT - 1):
            d = np.array([r["phys1gp"][k] - r[name][k] for r in res])
            print("phys1gp - %s best of %d: %+.3f +- %.3f (wins %.0f%%, ties %.0f%%)" % (name, k + 1, d.mean(), d.std() / np.sqrt(len(d)),
                  100 * np.mean(d > 1e-9), 100 * np.mean(np.abs(d) <= 1e-9)))
