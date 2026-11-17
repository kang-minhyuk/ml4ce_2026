"""Offline: does adding MBR probes to Pilot observations improve the later Pilot sequence? (analysis only)"""
import numpy as np
from multiprocessing import Pool
from lib_joint import LM, LP, grp, A0, A1, B2, gi
def near(a0, a1, b2): return gi(int(np.argmin(abs(np.log(A0 / a0)))), int(np.argmin(abs(np.log(A1 / a1)))), int(np.argmin(abs(B2 - b2))))
pri = near(0.8, 0.1, 30)
PORT2 = near(1.5, 1.0, 42)
def post(Ltr, JM, yM, JP, yP, sig):
    lw = np.zeros(len(Ltr[0])); cP = None
    for L, J, y, k in ((Ltr[0], JM, yM, 0), (Ltr[1], JP, yP, 1)):
        if len(J) == 0: continue
        R = y[None] - L[:, J]; c = R.mean(1); lw -= ((R - c[:, None]) ** 2).sum(1) / (2 * sig ** 2)
        if k: cP = c
    w = np.exp(lw - lw.max()); return w / w.sum(), cP
def run(args):
    tr, te, JM, n, sig = args
    Ltr = (LM[grp == tr], LP[grp == tr]); out = []
    for tM, tP in zip(LM[grp == te][::3], LP[grp == te][::3]):
        JP = [pri]; res = [np.exp(tP[pri] - tP.max())]
        while len(res) < n:
            w, cP = post(Ltr, JM, tM[JM], JP, tP[JP], sig)
            a = w @ np.maximum(np.exp(Ltr[1] + cP[:, None]) - np.exp(tP[JP].max()), 0); a[JP] = -np.inf
            JP.append(int(np.argmax(a))); res.append(np.exp(tP[JP].max() - tP.max()))
        out.append(res)
    return np.mean(out, 0)
if __name__ == "__main__":
    designs = {"pilot only": [], "MBR prior": [pri], "MBR prior+1": [pri, near(0.154, 0.096, 6)],
               "MBR prior+2": [pri, near(0.154, 0.096, 6), near(0.015, 0.096, 25)],
               "MBR prior+3": [pri, near(0.154, 0.096, 6), near(0.015, 0.096, 25), near(2.5, 0.98, 48)],
               "MBR prior+hi": [pri, near(2.5, 0.98, 48)]}
    with Pool(12) as p:
        res = p.map(run, [(tr, 1 - tr, J, 6, 0.1) for J in designs.values() for tr in (0, 1)])
    for i, name in enumerate(designs):
        print("%-14s" % name, " ".join("%.3f" % v for v in (res[2 * i] + res[2 * i + 1]) / 2))
