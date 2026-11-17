import json, numpy as np
from multiprocessing import Pool
from sim import Sim, load_scenario
from decomp import heur
O = json.load(open("oracle.json"))
dev = json.load(open("splits.json"))["dev"][:60]
def run(sid):
    sc = load_scenario(sid); f = Sim(sc); b = np.array(sc["bounds"]); lo, hi = b[:, 0], b[:, 1]
    n = len(sc["clones"])
    H = np.array([heur(f, sc, ci) for ci in range(n)])
    P, M = [], []
    for ci, c in enumerate(sc["clones"]):
        To = np.clip(c["e_activation"], lo[0], hi[0]); po = np.clip(c["ph_opt"], lo[1], hi[1])
        x = np.zeros(12); x[2] = lo[2] + 0.3 * (hi[2] - lo[2]); x[3] = lo[3] + 0.3 * (hi[3] - lo[3])
        x[[0, 8, 10]] = To; x[[1, 9, 11]] = po
        P.append(f("MTP", ci, x))
        u = np.array([0, 0, 1, 0.02, 1, 0.03, 1, 0.4, 0, 0, 0, 0.]); xm = lo + u * (hi - lo); xm[[0, 8, 10]] = To; xm[[1, 9, 11]] = po
        M.append(f("MBR", ci, xm))
    P, M = np.array(P), np.array(M)
    out = {"best_heur": H.max() / O[str(sid)]["oracle"]}
    for k in (4, 8, 12, n):
        top = np.argsort(-P)[:k]
        pick = top[np.argmax(M[top])]
        out[k if k < n else "all"] = H[pick] / O[str(sid)]["oracle"]
    return out
if __name__ == "__main__":
    with Pool(21) as p: res = p.map(run, dev)
    for k in ["best_heur", 4, 8, 12, "all"]:
        print(k, round(np.mean([r[k] for r in res]), 3))
