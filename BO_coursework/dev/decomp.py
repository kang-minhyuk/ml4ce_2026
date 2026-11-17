import json, sys, numpy as np
from multiprocessing import Pool
from sim import Sim, load_scenario
T = json.load(open(sys.argv[1])) if __name__ == "__main__" else None
def heur(f, sc, ci):
    b = np.array(sc["bounds"]); lo, hi = b[:, 0], b[:, 1]; c = sc["clones"][ci]
    best = 0
    for q2 in (0.25, 0.4, 0.6):
        u = np.array([0, 0, 1, 0.02, 1, 0.03, 1, q2, 0, 0, 0, 0.]); x = lo + u * (hi - lo)
        x[[0, 8, 10]] = np.clip(c["e_activation"], lo[0], hi[0]); x[[1, 9, 11]] = np.clip(c["ph_opt"], lo[1], hi[1])
        best = max(best, f("Pilot", ci, x))
    return best
def run(r):
    sc = load_scenario(r["sid"]); f = Sim(sc)
    hs = [heur(f, sc, ci) for ci in range(len(sc["clones"]))]
    out = []
    for p in r["pilots"]:
        out.append((p["true"] / r["oracle"], hs[p["clone"]] / r["oracle"], max(hs) / r["oracle"], sorted(hs, reverse=True).index(hs[p["clone"]])))
    return out
if __name__ == "__main__":
    with Pool(21) as pool: res = pool.map(run, T)
    first = np.array([o[0] for o in res if o]); allp = np.array([x for o in res for x in o])
    print("first pilot: actual/oracle %.3f | its clone's structured-heuristic/oracle %.3f | best clone heuristic %.3f | median clone rank %d"
          % (first[:, 0].mean(), first[:, 1].mean(), first[:, 2].mean(), np.median(first[:, 3])))
    print("all pilots: actual %.3f clone-heur %.3f ; frac clone in top3 by heuristic %.2f" % (allp[:, 0].mean(), allp[:, 1].mean(), np.mean(allp[:, 3] < 3)))
    print("recipe efficiency (actual/clone-heur) first: %.2f all: %.2f" % (np.mean(first[:, 0] / first[:, 1]), np.mean(allp[:, 0] / allp[:, 1])))
