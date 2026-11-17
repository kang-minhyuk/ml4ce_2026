"""Offline: which plate-signature features best predict a clone's Pilot glutamine optimum?
Leave-one-scenario-out kernel-weighted library recipe; reports target relative titer (analysis only)."""
import numpy as np, sys
d = np.load("gln_lib.npz"); s = np.load("lib_signature.npz")
Y = d["Y"]; sid = d["sid"]; gmax = d["gmax"]; A0 = d["A0"]
S = np.log(np.maximum(s["S"], 1e-9))
n = len(Y); a0 = np.repeat(A0, 12 * 7)
R = Y / Y.max(1, keepdims=True)
LR = np.log(np.maximum(R, 1e-3))
rng = np.random.default_rng(0)
def evaluate(F, sc, noise=0.0):
    F = np.atleast_2d(F.T).T
    Fn = F + noise * rng.standard_normal(F.shape)
    out = []
    for j in range(n):
        m = sid != sid[j]
        dd = (((F[m] - Fn[j]) / sc) ** 2).sum(1)
        w = np.exp(-0.5 * (dd - dd.min())); w /= w.sum()
        sc_ = w @ R[m]
        sc_[a0 > gmax[j] * 1.0001] = -1
        out.append(R[j, sc_.argmax()])
    return np.mean(out)
base = S[:, 14]
print("none (portfolio-ish)", evaluate(np.zeros(n), 1.0))
for noise in (0.0, 0.5):
    print("noise", noise)
    print(" F14 s.5", evaluate(base, 0.5, noise))
    for c in range(18):
        if c == 14: continue
        print(" F14+F%02d s.5" % c, round(evaluate(np.c_[base, S[:, c]], 0.5, noise), 4),
              " ratio s.3", round(evaluate(np.c_[base, S[:, c] - base], np.array([0.5, 0.3]), noise), 4))
