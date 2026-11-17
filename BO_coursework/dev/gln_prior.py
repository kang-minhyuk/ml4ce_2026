"""Fixed glutamine prior recipes at the oracle clone (true T/pH): mean value / oracle (analysis only)."""
import json, itertools, numpy as np
from multiprocessing import Pool
from sim import Sim, load_scenario
O = json.load(open("oracle.json")); DEV = json.load(open("splits.json"))["dev"][:40]
A0 = [.0005, .001, .002, .004, .008, .016, .03]; A1 = [.005, .01, .02, .03, .05, .1]; B2 = [.25, .4, .6]
def run(sid):
    sc = load_scenario(sid); f = Sim(sc); b = f.bounds; o = O[str(sid)]; ci = o["clone"]; c = sc["clones"][ci]
    lo, sp = b[:, 0], b[:, 1] - b[:, 0]
    out = {}
    for r in ("Pilot", "MBR"):
        for a0, a1, b2 in itertools.product(A0, A1, B2):
            u = np.array([0, 0, 1, a0, 1, a1, 1, b2, 0, 0, 0, 0.]); x = lo + u * sp
            x[[0, 8, 10]] = np.clip(c["e_activation"], *b[0]); x[[1, 9, 11]] = np.clip(c["ph_opt"], *b[1])
            out[(r, a0, a1, b2)] = f(r, ci, x)
    pil = max(v for k, v in out.items() if k[0] == "Pilot"); mb = max(v for k, v in out.items() if k[0] == "MBR")
    best_mbr = max((k for k in out if k[0] == "MBR"), key=out.get)
    return {k: v / (o["oracle"] if k[0] == "Pilot" else mb) for k, v in out.items()}, out[("Pilot",) + best_mbr[1:]] / o["oracle"], pil / o["oracle"]
if __name__ == "__main__":
    with Pool(8) as p: res = p.map(run, DEV)
    keys = list(res[0][0])
    m = {k: np.mean([r[0][k] for r in res]) for k in keys}
    q = {k: np.percentile([r[0][k] for r in res], 10) for k in keys}
    for r in ("Pilot", "MBR"):
        top = sorted([k for k in keys if k[0] == r], key=lambda k: -m[k])[:8]
        for k in top: print(r, k[1:], f"mean {m[k]:.3f} p10 {q[k]:.3f}")
    print("grid-best Pilot / oracle %.3f ; MBR-grid-best recipe on Pilot %.3f" % (np.mean([r[2] for r in res]), np.mean([r[1] for r in res])))
