"""For first Pilots of a feature-variant trace: titer with the chosen recipe vs portfolio[0] vs feature recipe from the
TRUE plate feature, all at the pipeline's T/pH (analysis)."""
import json, sys, importlib.util, numpy as np
from multiprocessing import Pool
sys.path.insert(0, '../scenario_suite'); import run_suite
from sim import Sim, load_scenario
spec = importlib.util.spec_from_file_location("m", "algos/v2_base.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
LIB = m.load_library(); Fe = np.array(m.GLN_PLATE)
R = json.load(open(sys.argv[1]))
def frac(a, lo, sp): return (a - lo) / sp
def one(r):
    sc = load_scenario(r["sid"]); f = Sim(sc)
    b = f.bounds; lo, sp = b[:, 0], b[:, 1] - b[:, 0]; seen = set(); out = []
    cand = LIB.pts; ok = (cand[:, 0] <= b[3, 1]) & (cand[:, 1] <= b[5, 1]) & (cand[:, 2] <= b[7, 1]); cand = cand[ok]
    for blk in r["log"][1:]:
        for x in blk["x"]:
            if x[0] != "Pilot": continue
            c = int(x[1].split("_")[1]) - 1
            if c in seen: continue
            seen.add(c); u = np.array(x[2:], float); cl = sc["clones"][c]
            act = f("Pilot", c, u)
            p0 = u.copy(); p0[3] = lo[3] + 10 ** -1.7959 * sp[3]; p0[5] = lo[5] + 10 ** -2.699 * sp[5]; p0[7] = lo[7] + 0.6 * sp[7]
            T = float(np.clip(cl["e_activation"], *b[0])); P = float(np.clip(cl["ph_opt"], *b[1]))
            Ft = np.log(max(f("MTP", c, [T, P, lo[2] + sp[2], lo[3] + 0.05 * sp[3], 0, 0, 0, 0, T, P, T, P]), 1))
            w = np.exp(-0.5 * ((Fe - Ft) / 0.5) ** 2); w /= w.sum(); a0, a1, b2 = cand[np.argmax(w @ np.exp(LIB.interp(cand)))]
            pt = u.copy(); pt[3], pt[5], pt[7] = a0, a1, b2
            out.append((act, f("Pilot", c, p0), f("Pilot", c, pt)))
    return out
if __name__ == "__main__":
    with Pool(4) as p: A = np.array(sum(p.map(one, R), []))
    print("n", len(A), "mean ratio chosen/portfolio0 %.3f  truefeat/portfolio0 %.3f" % (np.mean(A[:, 0] / A[:, 1]), np.mean(A[:, 2] / A[:, 1])))
    print("median ratio chosen/p0 %.3f truefeat/p0 %.3f; frac chosen<p0 %.2f" % (np.median(A[:, 0] / A[:, 1]), np.median(A[:, 2] / A[:, 1]), np.mean(A[:, 0] < A[:, 1])))
