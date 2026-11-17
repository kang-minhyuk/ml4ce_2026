import sys, numpy as np; sys.path.insert(0, "algos")
import i4_funnel as M
from scipy.optimize import least_squares
import scipy.optimize as so
rng = np.random.default_rng(0); nc = 25; n = 150
T, P = rng.random(n), rng.random(n); C = rng.integers(0, nc, n); ly = rng.normal(0, 1, n)
cap = {}
orig = so.least_squares
def spy(fun, x0, jac=None, **kw):
    J = jac(x0); Jn = np.zeros_like(J)
    for j in range(len(x0)):
        e = np.zeros_like(x0); e[j] = 1e-6; Jn[:, j] = (fun(x0 + e) - fun(x0 - e)) / 2e-6
    print("max jac err", np.abs(J - Jn).max())
    return orig(fun, x0, jac=jac, **kw)
M.least_squares = spy
for k in (True, False):
    M.fit_gauss_peaks(T, P, C, ly, nc, rng, k)
