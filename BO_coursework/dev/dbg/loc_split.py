"""First Pilot per clone in DEV traces: split the efficiency loss into location vs recipe; phase sensitivity."""
import json, sys, numpy as np
from multiprocessing import Pool
sys.path.insert(0, '../scenario_suite'); import run_suite
from sim import Sim, load_scenario
CV = json.load(open("clone_value.json")); R = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "traces/informed_bayes_dev.json"))

def one(r):
    sc = load_scenario(r["sid"]); f = Sim(sc)
    v = np.array(CV[str(r["sid"])]); seen = set(); out = []
    for b in r["log"][1:]:
        for x in b["x"]:
            if x[0] != "Pilot": continue
            c = int(x[1].split("_")[1]) - 1
            if c in seen: continue
            seen.add(c); cl = sc["clones"][c]; u = np.array(x[2:], float)
            T0 = float(np.clip(cl["e_activation"], *f.bounds[0])); P0 = float(np.clip(cl["ph_opt"], *f.bounds[1]))
            act = f("Pilot", c, u)
            w = u.copy(); w[[0, 8, 10]] = T0; w[[1, 9, 11]] = P0; tru = f("Pilot", c, w)
            w3 = u.copy(); w3[10] = T0; w3[11] = P0; ph3 = f("Pilot", c, w3)          # fix only phase 3
            w12 = w.copy(); w12[10] = u[10]; w12[11] = u[11]; ph12 = f("Pilot", c, w12)  # true in 1-2, actual in 3
            out.append((v[c] / v.max(), act / v[c], tru / v[c], ph3 / v[c], ph12 / v[c]))
    return out

if __name__ == "__main__":
    with Pool(8) as p: res = sum(p.map(one, R), [])
    A = np.array(res)
    print("n", len(A)); print("median eff actual %.3f | true T/pH all phases %.3f | true only phase3 %.3f | true phases1-2 only %.3f" % tuple(np.median(A[:, 1:], 0)))
    print("mean   eff actual %.3f | true T/pH all phases %.3f | true only phase3 %.3f | true phases1-2 only %.3f" % tuple(A[:, 1:].mean(0)))
    bad = A[:, 1] < 0.6
    print("bad (eff<.6) n=%d: actual %.3f, true loc %.3f, fix ph3 %.3f, fix ph12 %.3f" % ((bad.sum(),) + tuple(A[bad, 1:].mean(0))))
    top = A[:, 0] > 0.9
    print("top clones (pot>.9) n=%d: actual %.3f, true loc %.3f" % (top.sum(), A[top, 1].mean(), A[top, 2].mean()))
