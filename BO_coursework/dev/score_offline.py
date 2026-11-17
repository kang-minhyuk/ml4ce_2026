"""Selection among MBR-screened clones in traced runs: MBR prior titer alone vs + beta * fitted plate height."""
import json, sys
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, "algos")
from sim import load_scenario
import tmp_base as M
from loc_offline import fit_fixed, SOFT
V = json.load(open("clone_value.json"))
CFG = M.CFG


def run(r):
    sc = load_scenario(r["sid"]); names = [c["name"] for c in sc["clones"]]; nc = len(names)
    b = np.array(sc["bounds"], float); lo, span = b[:, 0], b[:, 1] - b[:, 0]
    pot = np.array(V[str(r["sid"])])
    plates = []; mbr = {}
    for k, blk in enumerate(r["log"]):
        for x, y in zip(blk["x"], blk["y"]):
            if x[0] == "MTP": plates.append((names.index(x[1]), float(x[2]), float(x[3]), y))
            if x[0] == "MBR" and k > 0:
                u = (np.array(x[2:], float) - lo) / np.where(span > 0, span, 1)
                if abs(np.log10(max(u[3], 1e-6)) - CFG["la0"]) < 0.05 and abs(u[7] - CFG["b0"]) < 0.02:
                    c = names.index(x[1]); mbr[c] = max(mbr.get(c, 0), y)
    if len(mbr) < 2 or not plates: return None
    C = np.array([p[0] for p in plates]); T = (np.array([p[1] for p in plates]) - lo[0]) / span[0]
    P = (np.array([p[2] for p in plates]) - lo[1]) / span[1]; Y = np.array([p[3] for p in plates])
    hq = M.fit_quad_peaks(T, P, C, Y, nc, np.random.default_rng(0))["height"]
    w = (SOFT[0] / span[0], SOFT[1] / span[0], SOFT[2] / span[1], SOFT[3] / span[1])
    ht = fit_fixed(T, P, C, Y, nc, w, True, 5.0)["height"]
    cs = np.array(sorted(mbr)); m = np.log(np.maximum([mbr[c] for c in cs], 1.0))
    out = {}
    for beta in (0, 0.2, 0.35, 0.5, 0.7):
        for nm, h in (("quad", hq), ("tmpl", ht)):
            s = m + beta * h[cs]
            out[(nm, beta)] = pot[cs[np.argmax(s)]] / pot[cs].max()
    return out


if __name__ == "__main__":
    T = json.load(open(sys.argv[1]))
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 4) as p:
        res = [x for x in p.map(run, T) if x]
    for k in sorted(res[0]):
        print(k, "%.3f" % np.mean([x[k] for x in res]), "top1 %.2f" % np.mean([np.isclose(x[k], 1) for x in res]))
