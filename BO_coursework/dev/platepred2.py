"""Plate ranking power vs plate medium (true T/pH optimum) against per-clone Pilot potential (analysis only)."""
import json, numpy as np
from multiprocessing import Pool
from scipy.stats import spearmanr
from sim import Sim, load_scenario
V = json.load(open("clone_value.json")); IDS = list(V)[:60]
MEDIA = [(1, .05), (1, .02), (1, .01), (1, .1), (.5, .05), (.3, .3), (.5, .01), (1, .003)]
def run(sid):
    sc = load_scenario(int(sid)); f = Sim(sc); b = np.array(sc["bounds"]); lo, hi = b[:, 0], b[:, 1]
    H = np.array(V[sid]); out = {}
    for g, q in MEDIA:
        a = []
        for ci, c in enumerate(sc["clones"]):
            x = np.zeros(12); x[2] = lo[2] + g * (hi[2] - lo[2]); x[3] = lo[3] + q * (hi[3] - lo[3])
            x[[0, 8, 10]] = np.clip(c["e_activation"], lo[0], hi[0]); x[[1, 9, 11]] = np.clip(c["ph_opt"], lo[1], hi[1])
            a.append(f("MTP", ci, x))
        a = np.array(a); o = np.argsort(-a)
        out[(g, q)] = (spearmanr(a, H)[0], *[H[o[:k]].max() / H.max() for k in (1, 2, 4, 8)])
    # MBR at prior recipe (a0 .01 a1 .005 b2 .6) at the true optimum
    m = []
    for ci, c in enumerate(sc["clones"]):
        u = np.array([0, 0, 1, .01, 1, .005, 1, .6, 0, 0, 0, 0.]); x = lo + u * (hi - lo)
        x[[0, 8, 10]] = np.clip(c["e_activation"], lo[0], hi[0]); x[[1, 9, 11]] = np.clip(c["ph_opt"], lo[1], hi[1])
        m.append(f("MBR", ci, x))
    m = np.array(m); o = np.argsort(-m)
    out["MBR prior"] = (spearmanr(m, H)[0], *[H[o[:k]].max() / H.max() for k in (1, 2, 4, 8)])
    return out
if __name__ == "__main__":
    with Pool(12) as p: res = p.map(run, IDS)
    for m in MEDIA + ["MBR prior"]:
        v = np.array([r[m] for r in res])
        print(f"{str(m):12s} spearman {v[:, 0].mean():.2f} capture top1 {v[:, 1].mean():.2f} top2 {v[:, 2].mean():.2f} top4 {v[:, 3].mean():.2f} top8 {v[:, 4].mean():.2f}")
