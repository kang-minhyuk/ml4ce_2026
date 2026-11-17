"""Plate designs: Sobol-only vs Sobol + targeted (cluster) plates. Ranking capture vs per-clone Pilot potential,
optimum location error. python plate_study2.py MODEL(quad|g1|gk|hyb) MEDIUM_G MEDIUM_A"""
import json, sys
import numpy as np
import sobol_seq
from multiprocessing import Pool
from sim import Sim, load_scenario
sys.path.insert(0, "algos")
from peakfit_g import fit_g
V = json.load(open("clone_value.json")); IDS = list(V)[:40]
MODEL = sys.argv[1]; MED = (float(sys.argv[2]), float(sys.argv[3]))
TOTAL = [3, 4, 5, 6, 8]


def fit(T, P, C, Y, nc, rng):
    import i3_funnel as M
    a = (np.array(T), np.array(P), np.array(C), np.array(Y), nc, rng)
    if MODEL == "quad":
        return M.fit_peaks(*a)
    if MODEL == "hyb":
        k = fit_g(*a, per_clone_k=True); s = fit_g(*a, per_clone_k=False)
        return {"t": k["t"], "p": k["p"], "height": s["height"]}
    return fit_g(*a, per_clone_k=(MODEL == "gk"))


def run(args):
    sid, design = args
    sc = load_scenario(int(sid)); f = Sim(sc); nc = len(sc["clones"])
    b = np.array(sc["bounds"], float); lo, span = b[:, 0], b[:, 1] - b[:, 0]
    H = np.array(V[sid])
    tt = np.array([(np.clip(c["e_activation"], *b[0]) - lo[0]) / span[0] for c in sc["clones"]])
    pt = np.array([(np.clip(c["ph_opt"], *b[1]) - lo[1]) / span[1] for c in sc["clones"]])
    rng = np.random.default_rng(int(sid))
    pts = (sobol_seq.i4_sobol_generate(2, 16, 1) + rng.random(2)) % 1.0
    T, P, C, Y = [], [], [], []
    order = list(rng.permutation(nc)); k = 0; out = {}; pk = None

    def plate(t, p, clones):
        for c in clones:
            x = np.zeros(12); x[0], x[1] = lo[0] + t * span[0], lo[1] + p * span[1]
            x[2], x[3] = lo[2] + MED[0] * span[2], lo[3] + MED[1] * span[3]
            T.append(t); P.append(p); C.append(c); Y.append(f("MTP", c, x))
    for npl in range(1, max(TOTAL) + 1):
        if design == "sobol" or npl <= 3:
            t, p = pts[npl - 1]; cl = [order[(k + i) % nc] for i in range(24)]; k += 24
        else:
            rank = np.argsort(-pk["height"]); U = np.c_[T, P]; Ca = np.array(C)
            target = rank[0]
            for c in rank[:max(6, nc // 3)]:
                d = np.sqrt(((U[Ca == c] - [pk["t"][c], pk["p"][c]]) ** 2).sum(1)).min()
                if d > 0.06:
                    target = c; break
            t, p = pk["t"][target], pk["p"][target]
            dist = np.hypot(pk["t"] - t, pk["p"] - p)
            score = dist - 0.02 * (nc - np.argsort(np.argsort(rank)))
            cl = list(np.argsort(score)[:min(24, nc)])
            while len(cl) < 24: cl.append(cl[len(cl) % nc])
        plate(t, p, cl)
        pk = fit(T, P, C, Y, nc, rng)
        if npl in TOTAL:
            te = np.abs(pk["t"] - tt) * span[0]; pe = np.abs(pk["p"] - pt) * span[1]
            rk = np.argsort(-pk["height"])
            out[npl] = (np.median(te), np.median(pe), *[H[rk[:kk]].max() / H.max() for kk in (1, 2, 4, 8)])
    return design, out


if __name__ == "__main__":
    with Pool(12) as p:
        res = p.map(run, [(s, d) for d in ("sobol", "target") for s in IDS])
    for d in ("sobol", "target"):
        for n in TOTAL:
            v = np.array([o[n] for dd, o in res if dd == d])
            print(f"{MODEL} {MED} {d:6s} {n} plates: T err {v[:, 0].mean():.2f} pH err {v[:, 1].mean():.3f} | capture top1 "
                  f"{v[:, 2].mean():.2f} top2 {v[:, 3].mean():.2f} top4 {v[:, 4].mean():.2f} top8 {v[:, 5].mean():.2f}")
