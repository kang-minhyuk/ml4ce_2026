"""Gaussian-factor peak model for plate data (analysis prototype).
log y = h_c + k_c * (max(fT,.2) * max(fP,.2) - 1), fT/fP asymmetric Gaussians with shared widths (unit coords)."""
import numpy as np
from scipy.optimize import least_squares


def gauss(d, wl, wr):
    return np.exp(-np.where(d < 0, (d / wl) ** 2, (d / wr) ** 2))


def fit_g(T, P, C, Y, nc, rng, per_clone_k=True, w0=(0.33, 0.8, 0.84, 0.2), clip=0.2):
    ly = np.log(np.maximum(Y, 1e-4 * max(Y.max(), 1e-12)))
    seen = np.zeros(nc, bool); seen[np.unique(C)] = True
    nk = nc if per_clone_k else 1

    def unpack(th):
        h, tc, pc = th[:nc], th[nc:2 * nc], th[2 * nc:3 * nc]
        k = np.exp(th[3 * nc:3 * nc + nk]); w = np.exp(th[3 * nc + nk:3 * nc + nk + 4])
        return h, tc, pc, (k[C] if per_clone_k else k[0]), w, k

    def res(th):
        h, tc, pc, kC, w, k = unpack(th)
        F = np.maximum(gauss(T - tc[C], w[0], w[1]), clip) * np.maximum(gauss(P - pc[C], w[2], w[3]), clip)
        pred = h[C] + kC * (F - 1)
        extra = [0.1 * (tc - 0.5), 0.1 * (pc - 0.5), 0.3 * (np.log(k) - np.log(k).mean())]
        return np.concatenate([pred - ly] + extra)

    h0 = np.array([ly[C == c].max() if seen[c] else ly.mean() for c in range(nc)])
    t0 = np.array([T[C == c][np.argmax(ly[C == c])] if seen[c] else 0.5 for c in range(nc)])
    p0 = np.array([P[C == c][np.argmax(ly[C == c])] if seen[c] else 0.5 for c in range(nc)])
    lo = np.concatenate([np.full(nc, -60.0), np.full(nc, -0.3), np.full(nc, -0.3), np.full(nk, np.log(0.05)),
                         [np.log(0.03)] * 4])
    hi = np.concatenate([np.full(nc, 60.0), np.full(nc, 1.3), np.full(nc, 1.3), np.full(nk, np.log(20.0)),
                         [np.log(5.0)] * 4])
    best = None
    for s in range(2):
        st = np.concatenate([h0, t0, p0, np.full(nk, np.log(2.0)), np.log(w0) + (rng.normal(0, .3, 4) if s else 0)])
        r = least_squares(res, np.clip(st, lo + 1e-9, hi - 1e-9), bounds=(lo, hi), loss="soft_l1", f_scale=0.3,
                          max_nfev=300)
        if best is None or r.cost < best.cost:
            best = r
    h, tc, pc, _, w, k = unpack(best.x)
    return {"height": h, "t": np.clip(tc, 0, 1), "p": np.clip(pc, 0, 1), "k": k, "w": w, "seen": seen}
