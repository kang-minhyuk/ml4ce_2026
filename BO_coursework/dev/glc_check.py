"""Offline: is glucose at the upper bound (start + both feeds) optimal at Pilot scale? (analysis only)"""
import itertools, json, numpy as np
from multiprocessing import Pool
from sim import Sim, load_scenario
A0 = [.001, .003, .008, .02, .05]; A1 = [.003, .008, .02, .05, .12]; B2 = [.3, .6]
GL = [(1, 1, 1), (0.6, 1, 1), (1, 0.6, 0.6), (0.6, 0.6, 0.6), (0.3, 1, 1), (1, 1, 0.3)]
def run(sid):
    sc = load_scenario(sid); f = Sim(sc); b = f.bounds; lo, sp = b[:, 0], b[:, 1] - b[:, 0]
    out = []
    for ci, c in enumerate(sc["clones"][:12]):
        T = float(np.clip(c["e_activation"], *b[0])); P = float(np.clip(c["ph_opt"], *b[1]))
        def val(g0, g1, g2, a0, a1, b2):
            u = np.array([0, 0, g0, a0, g1, a1, g2, b2, 0, 0, 0, 0.]); x = lo + u * sp
            x[[0, 8, 10]] = T; x[[1, 9, 11]] = P
            return f("Pilot", ci, x)
        best = max(itertools.product(A0, A1, B2), key=lambda q: val(1, 1, 1, *q))
        out.append([val(*g, *best) for g in GL])
    return out
if __name__ == "__main__":
    ids = json.load(open("splits.json"))["dev"][:25]
    with Pool(4) as p: R = np.array([r for rr in p.map(run, ids) for r in rr])
    rel = R / R[:, :1]
    for g, col in zip(GL, rel.T):
        print(g, "mean rel %.3f  frac clones better than (1,1,1) by >1%%: %.2f  max %.3f" % (col.mean(), (col > 1.01).mean(), col.max()))
