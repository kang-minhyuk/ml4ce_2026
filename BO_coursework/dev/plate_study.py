"""How well do N rich-medium plates locate each clone's T/pH optimum and rank clones? (analysis only)
python plate_study.py"""
import json, sys
import numpy as np
import sobol_seq
from multiprocessing import Pool
from sim import Sim, load_scenario
from decomp import heur
from peakfit_g import fit_g
MODELS = ['quad', 'g1', 'gk']
sys.path.insert(0, "algos")

DEV = json.load(open("splits.json"))["dev"][:30]
NPL = [3, 4, 6, 8]
MED = (1.0, 0.05)


def run(sid):
    import i3_funnel as M
    sc = load_scenario(sid); f = Sim(sc); nc = len(sc["clones"])
    b = np.array(sc["bounds"], float); lo, span = b[:, 0], b[:, 1] - b[:, 0]
    H = np.array([heur(f, sc, ci) for ci in range(nc)])
    tt = np.array([(np.clip(c["e_activation"], *b[0]) - lo[0]) / span[0] for c in sc["clones"]])
    pt = np.array([(np.clip(c["ph_opt"], *b[1]) - lo[1]) / span[1] for c in sc["clones"]])
    rng = np.random.default_rng(sid)
    pts = (sobol_seq.i4_sobol_generate(2, 16, 1) + rng.random(2)) % 1.0
    T, P, C, Y = [], [], [], []
    out = {}
    order = list(rng.permutation(nc))
    k = 0
    for npl in range(1, max(NPL) + 1):
        t, p = pts[npl - 1]
        for w in range(24):
            c = order[k % nc]; k += 1
            x = np.zeros(12); x[0], x[1] = lo[0] + t * span[0], lo[1] + p * span[1]
            x[2], x[3] = lo[2] + MED[0] * span[2], lo[3] + MED[1] * span[3]
            x[[8, 10]], x[[9, 11]] = x[0], x[1]
            T.append(t); P.append(p); C.append(c); Y.append(f("MTP", c, x))
        if npl in NPL:
            for name in MODELS:
                a = (np.array(T), np.array(P), np.array(C), np.array(Y), nc, rng)
                pk = M.fit_peaks(*a) if name == "quad" else fit_g(*a, per_clone_k=(name == "gk"))
                te = np.abs(pk["t"] - tt) * span[0]; pe = np.abs(pk["p"] - pt) * span[1]
                rk = np.argsort(-pk["height"])
                cap = [H[rk[:kk]].max() / H.max() for kk in (1, 4, 8)]
                out[(name, npl)] = (np.median(te), np.median(pe), *cap)
    return out


if __name__ == "__main__":
    with Pool(12) as p:
        res = p.map(run, DEV)
    for m in MODELS:
      for n in NPL:
        v = np.array([r[(m, n)] for r in res])
        print(f"{m:5s} {n} plates: med|T err| {v[:, 0].mean():.2f} C  med|pH err| {v[:, 1].mean():.3f}  "
              f"capture top1 {v[:, 2].mean():.2f} top4 {v[:, 3].mean():.2f} top8 {v[:, 4].mean():.2f}")
