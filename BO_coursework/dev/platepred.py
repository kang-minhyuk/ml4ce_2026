import json, numpy as np
from multiprocessing import Pool
from scipy.stats import spearmanr
from sim import Sim, load_scenario
from decomp import heur
dev = json.load(open("splits.json"))["dev"][:50]
MEDIA = [(1, 0.02), (1, 0.1), (1, 0.3), (1, 0.6), (0.5, 0.1), (0.3, 0.3)]
def run(sid):
    sc = load_scenario(sid); f = Sim(sc); b = np.array(sc["bounds"]); lo, hi = b[:, 0], b[:, 1]
    H = np.array([heur(f, sc, ci) for ci in range(len(sc["clones"]))])
    out = {}
    for g, q in MEDIA:
        at_opt, at_mid, mbr = [], [], []
        for ci, c in enumerate(sc["clones"]):
            To = np.clip(c["e_activation"], lo[0], hi[0]); po = np.clip(c["ph_opt"], lo[1], hi[1])
            x = np.zeros(12); x[2] = lo[2] + g * (hi[2] - lo[2]); x[3] = lo[3] + q * (hi[3] - lo[3])
            x[[0, 8, 10]] = To; x[[1, 9, 11]] = po
            at_opt.append(f("MTP", ci, x))
        a = np.array(at_opt)
        out[(g, q)] = (spearmanr(a, H)[0], H[np.argmax(a)] / H.max(), np.mean(np.sort(H)[::-1][:1] <= H[np.argsort(-a)[:3]].max()))
    return out
if __name__ == "__main__":
    with Pool(21) as p: res = p.map(run, dev)
    for m in MEDIA:
        v = np.array([r[m] for r in res])
        print(f"plate medium glc={m[0]} gln={m[1]}: spearman {v[:,0].mean():.2f}  top1-by-plate/best {v[:,1].mean():.2f}  best-in-plate-top3 {v[:,2].mean():.2f}")
