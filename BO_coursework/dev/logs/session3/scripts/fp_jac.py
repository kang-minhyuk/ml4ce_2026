import numpy as np, time
from scipy.optimize import least_squares
def drop(d, wl, wr):
    return np.where(d < 0, (d / wl) ** 2, (d / wr) ** 2)
def make(T, P, C, ly, nc):
    n = len(ly)
    def unpack(th):
        return th[:nc], th[nc:2 * nc], th[2 * nc:3 * nc], np.exp(th[3 * nc:3 * nc + 4]), th[3 * nc + 4]
    def res(th):
        a, tc, pc, w, fl = unpack(th)
        pred = a[C] - np.minimum(drop(T - tc[C], w[0], w[1]) + drop(P - pc[C], w[2], w[3]), np.exp(fl))
        return np.concatenate([pred - ly, 0.1 * (tc - 0.5), 0.1 * (pc - 0.5)])
    def jac(th):
        a, tc, pc, w, fl = unpack(th)
        dt, dp = T - tc[C], P - pc[C]
        lt = dt < 0; lp = dp < 0
        wt = np.where(lt, w[0], w[1]); wp = np.where(lp, w[2], w[3])
        D = (dt / wt) ** 2 + (dp / wp) ** 2
        act = D < np.exp(fl)
        J = np.zeros((n + 2 * nc, 3 * nc + 5))
        r = np.arange(n)
        J[r, C] = 1.0
        J[r, nc + C] = np.where(act, 2 * dt / wt ** 2, 0)
        J[r, 2 * nc + C] = np.where(act, 2 * dp / wp ** 2, 0)
        g_t = np.where(act, 2 * (dt / wt) ** 2, 0); g_p = np.where(act, 2 * (dp / wp) ** 2, 0)
        J[r, 3 * nc + 0] = np.where(lt, g_t, 0); J[r, 3 * nc + 1] = np.where(~lt, g_t, 0)
        J[r, 3 * nc + 2] = np.where(lp, g_p, 0); J[r, 3 * nc + 3] = np.where(~lp, g_p, 0)
        J[r, 3 * nc + 4] = np.where(act, 0, -np.exp(fl))
        J[n + np.arange(nc), nc + np.arange(nc)] = 0.1
        J[n + nc + np.arange(nc), 2 * nc + np.arange(nc)] = 0.1
        return J
    return res, jac
rng = np.random.default_rng(0); nc = 30; n = 200
T, P = rng.random(n), rng.random(n); C = rng.integers(0, nc, n)
tc0, pc0 = rng.random(nc), rng.random(nc)
ly = rng.normal(0, 1, nc)[C] - np.minimum(drop(T - tc0[C], .3, .6) + drop(P - pc0[C], .6, .2), 3) + rng.normal(0, .05, n)
res, jac = make(T, P, C, ly, nc)
th = np.concatenate([rng.normal(0, 1, nc), rng.random(nc), rng.random(nc), np.log([.3, .5, .5, .3]), [1.0]])
from scipy.optimize import approx_fprime
Jn = np.array([approx_fprime(th, lambda t: res(t)[i], 1e-7) for i in range(len(res(th)))])
print("jac err", np.abs(Jn - jac(th)).max())
lo = np.concatenate([np.full(nc, -60.0), np.full(nc, -0.2), np.full(nc, -0.2), [np.log(0.03)] * 4, [-3.0]])
hi = np.concatenate([np.full(nc, 60.0), np.full(nc, 1.2), np.full(nc, 1.2), [np.log(5.0)] * 4, [3.0]])
for j in ("2-point", jac):
    t0 = time.time(); r = least_squares(res, th, jac=j, bounds=(lo, hi), loss="soft_l1", f_scale=0.3, max_nfev=400)
    print(time.time() - t0, r.cost, r.nfev)
