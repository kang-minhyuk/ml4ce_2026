"""Offline: does the plate feature help rank clones from MBR screens at the prior recipe? (analysis only)
Within each library scenario (3 clones), pick by raw MBR titer at the prior recipe vs MBR titer x predicted
(Pilot potential / MBR prior-recipe titer) ratio from feature-kernel-weighted other scenarios."""
import numpy as np, json
from sim import load_scenario
P = np.load("gln_lib.npz"); M = np.load("gln_lib_mbr.npz"); S = np.load("lib_signature.npz")
assert (P["sid"] == M["sid"]).all() and (P["ci"] == M["ci"]).all()
A0, A1, B2 = P["A0"], P["A1"], P["B2"]; sid = P["sid"]; n = len(sid)
LA0, LA1, B0 = -1.7959, -2.699, 0.6  # replaced below from CFG if found
import re
src = open("final/informed_bayes2.py").read()
cfgv = {k: float(re.search(r"'%s': ([-0-9.]+)" % k, src).group(1)) for k in ("la0", "la1", "b0")}
a0g = np.repeat(A0, 84); a1g = np.tile(np.repeat(A1, 7), 13); b2g = np.tile(B2, 156)
pot = np.empty(n); mpr = np.empty(n)
cache = {}
for j in range(n):
    s = int(sid[j])
    if s not in cache:
        b = np.array(load_scenario(s)["bounds"]); cache[s] = b
    b = cache[s]; lo, hi = b[:, 0], b[:, 1]
    ok = (a0g <= hi[3]) & (a1g <= hi[5]) & (b2g <= hi[7]) & (b2g >= lo[7])
    pot[j] = P["Y"][j][ok].max()
    q = (lo[3] + 10 ** cfgv["la0"] * (hi[3] - lo[3]), lo[5] + 10 ** cfgv["la1"] * (hi[5] - lo[5]), lo[7] + cfgv["b0"] * (hi[7] - lo[7]))
    d = (np.log(a0g) - np.log(q[0])) ** 2 + (np.log(a1g) - np.log(q[1])) ** 2 + ((b2g - q[2]) / 10) ** 2
    mpr[j] = M["Y"][j][d.argmin()]
F = np.log(np.maximum(S["S"][:, 14], 1e-9)); R = np.log(pot) - np.log(np.maximum(mpr, 1e-9))
print("cfg", cfgv, "corr(F, log ratio pot/mbr)", np.corrcoef(F, R)[0, 1], "sd ratio", R.std())
rng = np.random.default_rng(0)
for s_, noise in ((0.5, 0), (0.5, 0.5), (0.3, 0.5), (0.8, 0.5)):
    Fn = F + noise * rng.standard_normal(n)
    pred = np.empty(n)
    for j in range(n):
        m = sid != sid[j]; w = np.exp(-0.5 * ((F[m] - Fn[j]) / s_) ** 2); pred[j] = w @ R[m] / w.sum()
    qr, qa, qf = [], [], []
    for s in np.unique(sid):
        i = np.where(sid == s)[0]
        if len(i) < 2: continue
        best = pot[i].max()
        qr.append(pot[i[np.argmax(np.log(mpr[i]))]] / best)
        qa.append(pot[i[np.argmax(np.log(mpr[i]) + pred[i])]] / best)
        qf.append(pot[i[np.argmax(Fn[i])]] / best)
    print("s %.1f noise %.1f: pick raw MBR %.4f  adjusted %.4f  plate feature only %.4f" % (s_, noise, np.mean(qr), np.mean(qa), np.mean(qf)))
