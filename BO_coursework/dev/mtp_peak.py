import json, numpy as np
from multiprocessing import Pool
from sim import Sim, load_scenario
dev = json.load(open("splits.json"))["dev"][:12]
def run(sid):
    sc = load_scenario(sid); f = Sim(sc); b = np.array(sc["bounds"]); lo, hi = b[:, 0], b[:, 1]; sp = hi - lo
    out = []
    for ci, c in enumerate(sc["clones"][:10]):
        for med in [(0.3, 0.3), (1.0, 0.05)]:
            best = (-1, 0, 0)
            for t in np.linspace(0, 1, 21):
                for p in np.linspace(0, 1, 21):
                    x = np.zeros(12); x[[0, 8, 10]] = lo[0] + t * sp[0]; x[[1, 9, 11]] = lo[1] + p * sp[1]
                    x[2] = lo[2] + med[0] * sp[2]; x[3] = lo[3] + med[1] * sp[3]
                    y = f("MTP", ci, x)
                    if y > best[0]: best = (y, x[0], x[1])
            out.append((med, best[1] - c["e_activation"], best[2] - c["ph_opt"]))
    return out
if __name__ == "__main__":
    with Pool(12) as p: res = [r for rr in p.map(run, dev) for r in rr]
    for med in [(0.3, 0.3), (1.0, 0.05)]:
        d = np.array([(a, b) for m, a, b in res if m == med])
        print(med, "T peak - Topt pct10/50/90:", np.round(np.percentile(d[:, 0], [10, 50, 90]), 2), " pH:", np.round(np.percentile(d[:, 1], [10, 50, 90]), 2))
