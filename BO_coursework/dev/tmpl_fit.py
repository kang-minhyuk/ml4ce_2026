"""Per-clone template peak fit (fixed widths, fixed-set floor search). Draft for the algorithm."""
import numpy as np
from scipy.optimize import least_squares


def fit_tmpl_pc(T, P, C, Y, nc, w, floors=(3.5, 5.0, 6.5), prior=0.1):
    ly = np.log(np.maximum(Y, 1e-4 * max(Y.max(), 1e-12)))
    seen = np.zeros(nc, bool); seen[np.unique(C)] = True
    best_tot, best = np.inf, None
    for fl in floors:
        tot = 0.0; H = np.full(nc, ly.mean()); Tc = np.full(nc, 0.5); Pc = np.full(nc, 0.5)
        for c in np.flatnonzero(seen):
            m = C == c; t, p, y = T[m], P[m], ly[m]

            def res(th):
                dt, dp = t - th[1], p - th[2]
                d = (dt / np.where(dt < 0, w[0], w[1])) ** 2 + (dp / np.where(dp < 0, w[2], w[3])) ** 2
                return np.concatenate([th[0] - fl * (1 - np.exp(-d / fl)) - y, [prior * (th[1] - 0.5), prior * (th[2] - 0.5)]])

            def jac(th):
                dt, dp = t - th[1], p - th[2]
                wt, wp = np.where(dt < 0, w[0], w[1]), np.where(dp < 0, w[2], w[3])
                e = np.exp(-((dt / wt) ** 2 + (dp / wp) ** 2) / fl)
                J = np.zeros((len(y) + 2, 3)); J[:len(y), 0] = 1; J[:len(y), 1] = e * 2 * dt / wt ** 2
                J[:len(y), 2] = e * 2 * dp / wp ** 2; J[len(y), 1] = prior; J[len(y) + 1, 2] = prior
                return J
            k = np.argmax(y); rb = None
            for dt0, dp0 in ((0, 0), (0.1, 0), (-0.1, 0), (0, 0.1), (0, -0.1)):
                x0 = np.array([y[k], np.clip(t[k] + dt0, -0.15, 1.15), np.clip(p[k] + dp0, -0.15, 1.15)])
                r = least_squares(res, x0, jac=jac, bounds=([-60, -0.2, -0.2], [60, 1.2, 1.2]), loss="soft_l1",
                                  f_scale=0.3, max_nfev=100)
                if rb is None or r.cost < rb.cost: rb = r
            tot += rb.cost; H[c], Tc[c], Pc[c] = rb.x
        if tot < best_tot:
            best_tot, best = tot, (H.copy(), Tc.copy(), Pc.copy())
    H, Tc, Pc = best
    return {"height": H, "t": np.clip(Tc, 0, 1), "p": np.clip(Pc, 0, 1), "seen": seen}


def fit_tmpl_vec(T, P, C, Y, nc, w, floors=(3.5, 5.0, 6.5), prior=0.1, fscale=0.3, iters=60):
    """Vectorised robust Levenberg-Marquardt over (clone x start x floor) copies; per clone keep the best start for
    each floor, then pick the floor with the lowest total cost."""
    ly = np.log(np.maximum(Y, 1e-4 * max(Y.max(), 1e-12)))
    seen = np.zeros(nc, bool); seen[np.unique(C)] = True
    cl = np.flatnonzero(seen)
    offs = np.array([(0, 0), (0.1, 0), (-0.1, 0), (0, 0.1), (0, -0.1)], float)
    K, F = len(offs), len(floors)
    # best well per clone
    bt = np.zeros(nc); bp = np.zeros(nc); ba = np.zeros(nc)
    for c in cl:
        m = np.flatnonzero(C == c); k = m[np.argmax(ly[m])]
        bt[c], bp[c], ba[c] = T[k], P[k], ly[k]
    # virtual copies v = (clone index j in cl, start k, floor f)
    J_, K_, F_ = np.meshgrid(np.arange(len(cl)), np.arange(K), np.arange(F), indexing="ij")
    J_, K_, F_ = J_.ravel(), K_.ravel(), F_.ravel()
    V = len(J_)
    vc = cl[J_]
    th = np.stack([ba[vc], np.clip(bt[vc] + offs[K_, 0], -0.15, 1.15), np.clip(bp[vc] + offs[K_, 1], -0.15, 1.15)], 1)
    fl = np.asarray(floors, float)[F_]
    # observation index lists: for every virtual copy, all wells of its clone
    pos = {c: np.flatnonzero(C == c) for c in cl}
    obs_v = np.concatenate([np.full(len(pos[vc[v]]), v) for v in range(V)])
    obs_i = np.concatenate([pos[vc[v]] for v in range(V)])
    t, p, y, flo = T[obs_i], P[obs_i], ly[obs_i], fl[obs_v]

    def evaluate(th):
        a, tc, pc = th[obs_v, 0], th[obs_v, 1], th[obs_v, 2]
        dt, dp = t - tc, p - pc
        wt, wp = np.where(dt < 0, w[0], w[1]), np.where(dp < 0, w[2], w[3])
        d = (dt / wt) ** 2 + (dp / wp) ** 2
        e = np.exp(-d / flo)
        r = a - flo * (1 - e) - y
        rho = 2 * fscale ** 2 * (np.sqrt(1 + (r / fscale) ** 2) - 1)
        cost = np.bincount(obs_v, rho, V) + (prior * (th[:, 1] - 0.5)) ** 2 + (prior * (th[:, 2] - 0.5)) ** 2
        return r, e, dt, dp, wt, wp, cost

    lam = np.full(V, 1e-2)
    r, e, dt, dp, wt, wp, cost = evaluate(th)
    for _ in range(iters):
        wgt = 1 / np.sqrt(1 + (r / fscale) ** 2)
        Jr = np.stack([np.ones_like(r), e * 2 * dt / wt ** 2, e * 2 * dp / wp ** 2], 1)
        A = np.zeros((V, 3, 3)); g = np.zeros((V, 3))
        np.add.at(A, obs_v, wgt[:, None, None] * Jr[:, :, None] * Jr[:, None, :])
        np.add.at(g, obs_v, (wgt * r)[:, None] * Jr)
        A[:, 1, 1] += prior ** 2; A[:, 2, 2] += prior ** 2
        g[:, 1] += prior ** 2 * (th[:, 1] - 0.5); g[:, 2] += prior ** 2 * (th[:, 2] - 0.5)
        Ad = A + lam[:, None, None] * (np.eye(3)[None] * (np.einsum("vii->vi", A)[:, :, None] + 1e-9))
        step = -np.linalg.solve(Ad, g[:, :, None])[:, :, 0]
        new = th + step
        new[:, 1:] = np.clip(new[:, 1:], -0.2, 1.2)
        r2, e2, dt2, dp2, wt2, wp2, cost2 = evaluate(new)
        ok = cost2 < cost
        th = np.where(ok[:, None], new, th)
        lam = np.where(ok, lam * 0.3, lam * 5)
        r, e, dt, dp, wt, wp, cost = evaluate(th)
    # best start per (clone, floor), then best floor by total cost
    cost3 = cost.reshape(len(cl), K, F); th3 = th.reshape(len(cl), K, F, 3)
    kb = np.argmin(cost3, 1)                                 # (clones, F)
    cb = np.take_along_axis(cost3, kb[:, None, :], 1)[:, 0]  # (clones, F)
    fb = int(np.argmin(cb.sum(0)))
    sel = th3[np.arange(len(cl)), kb[:, fb], fb]
    H = np.full(nc, ly.mean()); Tc = np.full(nc, 0.5); Pc = np.full(nc, 0.5)
    H[cl], Tc[cl], Pc[cl] = sel[:, 0], sel[:, 1], sel[:, 2]
    return {"height": H, "t": np.clip(Tc, 0, 1), "p": np.clip(Pc, 0, 1), "seen": seen, "floor": floors[fb]}
