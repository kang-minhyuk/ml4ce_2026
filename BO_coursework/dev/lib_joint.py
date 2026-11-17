"""Offline: MBR glutamine observations -> Pilot recipe via the joint MBR/Pilot library (analysis only)."""
import numpy as np, itertools
from multiprocessing import Pool
P = np.load("gln_lib.npz"); M = np.load("gln_lib_mbr.npz")
assert (P["sid"] == M["sid"]).all() and (P["ci"] == M["ci"]).all()
def lg(Y): return np.log(np.maximum(Y, 1e-3 * Y.max(1, keepdims=True)))
LP, LM = lg(P["Y"]), lg(M["Y"]); grp = (P["sid"] >= 10380).astype(int)
A0, A1, B2 = P["A0"], P["A1"], P["B2"]
G = np.array(list(itertools.product(range(13), range(12), range(7))))
def gi(i, j, k): return i * 84 + j * 7 + k
def rec(LMtr, LPtr, tM, J, sig):
    if len(J) == 0: w = np.ones(len(LMtr)) / len(LMtr)
    else:
        R = tM[J][None] - LMtr[:, J]; c = R.mean(1); lw = -((R - c[:, None]) ** 2).sum(1) / (2 * sig ** 2)
        w = np.exp(lw - lw.max()); w /= w.sum()
    return int(np.argmax(w @ np.exp(LPtr - LPtr.max(1, keepdims=True))))
def evaluate(args):
    tr, te, J, sig = args
    LMtr, LPtr = LM[grp == tr], LP[grp == tr]
    out = []
    for tM, tP in zip(LM[grp == te], LP[grp == te]):
        x = rec(LMtr, LPtr, tM, J, sig); out.append(np.exp(tP[x] - tP.max()))
    return np.mean(out)
if __name__ == "__main__":
    # prior-like point ~ (0.8, 0.1, 30) abs -> nearest grid indices
    def near(a0, a1, b2): return gi(int(np.argmin(abs(np.log(A0 / a0)))), int(np.argmin(abs(np.log(A1 / a1)))), int(np.argmin(abs(B2 - b2))))
    pri = near(0.8, 0.1, 30)
    designs = {
        "none": [],
        "prior": [pri],
        "prior+lo/hi a0": [pri, near(0.2, 0.1, 30), near(3, 0.1, 30)],
        "prior+lo/hi a1": [pri, near(0.8, 0.03, 30), near(0.8, 1.0, 30)],
        "4: a0/a1 corners": [pri, near(0.2, 0.06, 30), near(3, 1.0, 30), near(0.8, 1.0, 45)],
        "4: mix": [pri, near(0.2, 0.3, 18), near(2.5, 0.1, 45), near(0.8, 2.0, 30)],
    }
    # greedy design search on train group 0 (prior forced first)
    rng = np.random.default_rng(0)
    with Pool(20) as pool:
        for name, J in designs.items():
            r = pool.map(evaluate, [(0, 1, J, s) for s in (0.1, 0.2, 0.4)] + [(1, 0, J, s) for s in (0.1, 0.2, 0.4)])
            print("%-18s 0->1 sig .1/.2/.4: %.3f %.3f %.3f | 1->0: %.3f %.3f %.3f" % (name, *r), flush=True)
        J = [pri]
        for k in range(3):
            cands = rng.choice(len(G), 120, replace=False)
            sc = pool.map(evaluate, [(0, 0, J + [int(c)], 0.2) for c in cands])  # in-sample on train group
            J.append(int(cands[int(np.argmax(sc))]))
            r = pool.map(evaluate, [(0, 1, J, 0.2), (1, 0, J, 0.2)])
            print("greedy k=%d" % len(J), [tuple(np.round([A0[G[j][0]], A1[G[j][1]], B2[G[j][2]]], 3)) for j in J], "in %.3f | 0->1 %.3f 1->0 %.3f" % (max(sc), *r), flush=True)
