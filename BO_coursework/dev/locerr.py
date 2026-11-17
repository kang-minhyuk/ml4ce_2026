"""Location error of plate models on real pipeline plate data, for the Pilot clones (analysis only)."""
import json, sys
import numpy as np
from multiprocessing import Pool
from sim import load_scenario
sys.path.insert(0, "algos")
T = json.load(open(sys.argv[1]))


def run(r):
    import i18_funnel as M
    sc = load_scenario(r["sid"]); names = [c["name"] for c in sc["clones"]]; nc = len(names)
    b = np.array(sc["bounds"], float); lo, sp = b[:, 0], b[:, 1] - b[:, 0]
    rows = [(row, y) for blk in r["log"] for row, y in zip(blk["x"], blk["y"])]
    first_p = next((k for k, (row, _) in enumerate(rows) if row[0] == "Pilot"), len(rows))
    plates = [(row, y) for row, y in rows[:first_p] if row[0] == "MTP"]
    if not plates or not r["pilots"]:
        return None
    U = np.array([(np.array(row[2:6], float) - lo[:4]) / sp[:4] for row, _ in plates])
    C = np.array([names.index(row[1]) for row, _ in plates]); Y = np.array([y for _, y in plates])
    rng = np.random.default_rng(0)
    q = M.fit_peaks(U[:, 0], U[:, 1], C, Y, nc, rng)
    own = np.all(np.abs(U[:, 2:4] - [1.0, 0.05]) < 0.02, axis=1)
    if own.sum() < 24:
        return None
    ly = np.log(np.maximum(Y[own], 1e-4 * Y[own].max()))
    _, tk, pk, _ = M.fit_gauss_peaks(U[own, 0], U[own, 1], C[own], ly, nc, rng, True)
    lya = np.log(np.maximum(Y, 1e-4 * Y.max()))
    _, ta, pa, _ = M.fit_gauss_peaks(U[:, 0], U[:, 1], C, lya, nc, rng, True)
    c = r["pilots"][0]["clone"]; cl = sc["clones"][c]
    tt, pt = np.clip(cl["e_activation"], *b[0]), np.clip(cl["ph_opt"], *b[1])
    nw = int((C == c).sum()); nwo = int((C[own] == c).sum())
    e = lambda t, p: (t * sp[0] + lo[0] - tt, p * sp[1] + lo[1] - pt)
    return (*e(q["t"][c], q["p"][c]), *e(tk[c], pk[c]), *e(ta[c], pa[c]), nw, nwo, len(plates) // 24, int((~own).sum()) // 24)


if __name__ == "__main__":
    with Pool(12) as p:
        R = np.array([x for x in p.map(run, T) if x is not None])
    for name, k in (("quad all", 0), ("gk own", 2), ("gk all", 4)):
        print(f"{name:9s} |dT| med {np.median(np.abs(R[:, k])):.2f} mean {np.mean(np.abs(R[:, k])):.2f} p90 {np.percentile(np.abs(R[:, k]), 90):.2f}"
              f" | |dpH| med {np.median(np.abs(R[:, k + 1])):.3f} mean {np.mean(np.abs(R[:, k + 1])):.3f} | mean dT {R[:, k].mean():+.2f} dpH {R[:, k + 1].mean():+.3f}")
    print("wells of first-pilot clone: all", R[:, 6].mean(), "own", R[:, 7].mean(), "| plates", R[:, 8].mean(), "random-medium plates", R[:, 9].mean())
