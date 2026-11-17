"""Offline: does the clone's MBR titer at the prior recipe, paired with its first Pilot titer, improve the 2nd Pilot
recipe through the library (members carry both MBR and Pilot absolute surfaces)? (analysis only)"""
import numpy as np, re, sys
from sim import load_scenario
P = np.load("gln_lib.npz"); M = np.load("gln_lib_mbr.npz"); S = np.load("lib_signature.npz")
sid = P["sid"]; n = len(sid); A0, A1, B2 = P["A0"], P["A1"], P["B2"]
a0g = np.repeat(A0, 84); a1g = np.tile(np.repeat(A1, 7), 13); b2g = np.tile(B2, 156)
LP = np.log(np.maximum(P["Y"], 1e-6)); LM = np.log(np.maximum(M["Y"], 1e-6))
F = np.log(np.maximum(S["S"][:, 14], 1e-9))
src = open("final/informed_bayes2.py").read()
cfgv = {k: float(re.search(r"'%s': ([-0-9.]+)" % k, src).group(1)) for k in ("la0", "la1", "b0")}
qm = np.empty(n, int); okm = []
cache = {}
for j in range(n):
    s = int(sid[j])
    if s not in cache: cache[s] = np.array(load_scenario(s)["bounds"])
    b = cache[s]; lo, hi = b[:, 0], b[:, 1]
    okm.append((a0g <= hi[3]) & (a1g <= hi[5]) & (b2g <= hi[7]) & (b2g >= lo[7]))
    q = (lo[3] + 10 ** cfgv["la0"] * (hi[3] - lo[3]), lo[5] + 10 ** cfgv["la1"] * (hi[5] - lo[5]), lo[7] + cfgv["b0"] * (hi[7] - lo[7]))
    qm[j] = ((np.log(a0g) - np.log(q[0])) ** 2 + (np.log(a1g) - np.log(q[1])) ** 2 + ((b2g - q[2]) / 10) ** 2).argmin()
okm = np.array(okm)
RP = P["Y"] / np.where(okm, P["Y"], 0).max(1, keepdims=True)
rng = np.random.default_rng(0)
Fn = F + 0.5 * rng.standard_normal(n)
for sigm in (0.1, 0.2, 0.4, None):
    r1, r2 = [], []
    for j in range(n):
        m = sid != sid[j]
        lw = -0.5 * ((F[m] - Fn[j]) / 0.5) ** 2
        w = np.exp(lw - lw.max())
        rel = LP[m] - LP[m].max(1, keepdims=True)
        sc = np.where(okm[j], (w / w.sum()) @ np.exp(rel), -1); q1 = sc.argmax()
        y1 = LP[j, q1]; ym = LM[j, qm[j]]
        c = y1 - LP[m][:, q1]                     # member offsets from the Pilot obs
        lw2 = lw.copy()
        if sigm is not None:
            pred_m = LM[m][:, qm[j]] + c          # same clone scale for MBR and Pilot
            lw2 = lw2 - 0.5 * ((ym - pred_m) / sigm) ** 2
        w2 = np.exp(lw2 - lw2.max()); w2 /= w2.sum()
        Pp = LP[m] + c[:, None]
        ei = np.where(okm[j], w2 @ np.maximum(np.exp(Pp) - np.exp(y1), 0), -1); ei[q1] = -1
        q2 = ei.argmax()
        r1.append(RP[j, q1]); r2.append(max(RP[j, q1], RP[j, q2]))
    print("sig_mbr", sigm, "first %.4f  best of 2 %.4f" % (np.mean(r1), np.mean(r2)))
