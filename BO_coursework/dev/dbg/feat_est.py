"""Fit/compare in-run estimators of a clone's plate peak titer from the plate wells seen before its first Pilot."""
import json, sys, importlib.util, numpy as np
from multiprocessing import Pool
sys.path.insert(0, '../scenario_suite'); import run_suite
from sim import Sim, load_scenario
spec = importlib.util.spec_from_file_location("m", "algos/v2_base.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
R = json.load(open(sys.argv[1]))
def one(r):
    sc = load_scenario(r["sid"]); f = Sim(sc)
    b = f.bounds; lo, sp = b[:, 0], b[:, 1] - b[:, 0]; names = [c["name"] for c in sc["clones"]]
    wells = []; seen = set(); out = []
    w = (m.TMPL_W[0] / sp[0], m.TMPL_W[1] / sp[0], m.TMPL_W[2] / sp[1], m.TMPL_W[3] / sp[1])
    for blk in r["log"]:
        for x, y in zip(blk["x"], blk["y"]):
            c = names.index(x[1]); u = (np.array(x[2:], float) - lo) / sp
            if x[0] == "MTP": wells.append((c, u, y)); continue
            if x[0] != "Pilot" or c in seen: continue
            seen.add(c)
            W = [(u2, y2) for c2, u2, y2 in wells if c2 == c]
            if len(W) < 2: continue
            U = np.array([a for a, _ in W]); ly = np.log(np.maximum([v for _, v in W], 1.0))
            std = (np.abs(U[:, 2] - 1.0) < .02) & (np.abs(U[:, 3] - .05) < .01)
            if not std.any(): continue
            bl = m.bayes_locations(U[:, 0], U[:, 1], np.zeros(len(U), int), np.exp(ly), 1, w)
            pen = m.template_pen(U[:, 0] - bl["t_map"][0], U[:, 1] - bl["p_map"][0], w, 5.0)
            cl = sc["clones"][c]; T = float(np.clip(cl["e_activation"], *b[0])); P = float(np.clip(cl["ph_opt"], *b[1]))
            tru = np.log(max(f("MTP", c, [T, P, lo[2] + sp[2], lo[3] + .05 * sp[3], 0, 0, 0, 0, T, P, T, P]), 1))
            ls, ps = ly[std], pen[std]
            feats = [ls.max()] + [np.max(ls + np.minimum(ps, k)) for k in (0.5, 1, 2, 5)] + \
                    [ls[np.argmin(ps)] + ps.min(), bl["height"][0], ps.min(), std.sum(), bl["spread"][0]]
            out.append([tru] + feats)
    return out
if __name__ == "__main__":
    with Pool(4) as p: A = np.array(sum(p.map(one, R), []))
    y = A[:, 0]; names = ["maxobs", "cap.5", "cap1", "cap2", "cap5", "nearest+pen", "postheight"]
    for k, n in enumerate(names, 1):
        d = A[:, k] - y; print("%-12s bias %+.3f sd %.3f corr %.3f" % (n, d.mean(), d.std(), np.corrcoef(A[:, k], y)[0, 1]))
    # linear regression (2-fold CV) on [maxobs, cap1, cap5, minpen, nwells, spread]
    X = np.c_[np.ones(len(A)), A[:, [1, 3, 5, 6, 7, 8, 9, 10]]]
    idx = np.random.default_rng(0).permutation(len(A)); h = len(A) // 2; pred = np.zeros(len(A))
    for tr, te in ((idx[:h], idx[h:]), (idx[h:], idx[:h])):
        beta = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]; pred[te] = X[te] @ beta
    d = pred - y; print("regression CV  sd %.3f corr %.3f" % (d.std(), np.corrcoef(pred, y)[0, 1]))
    beta = np.linalg.lstsq(X, y, rcond=None)[0]; print("beta", np.round(beta, 3))
