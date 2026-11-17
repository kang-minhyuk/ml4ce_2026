import sys, numpy as np; sys.path.insert(0, "algos")
import i3_funnel as M
from scipy.optimize import approx_fprime
rng = np.random.default_rng(1)
gp = M.BioGP(M.CFG, rng); n = 30
gp.z = rng.random((n, 4)); gp.c = rng.integers(0, 5, n); gp.s = (rng.random(n) < .3).astype(int); gp.y = rng.normal(0, 1, n)
th = rng.normal(0, .3, 13)
v, g = gp.nll(th); gn = approx_fprime(th, lambda t: gp.nll(t)[0], 1e-6)
print(np.round(g, 4)); print(np.round(gn, 4))
