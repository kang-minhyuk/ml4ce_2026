"""Offline location study on traced plate data: free-width quad peak model vs fixed (template) widths.
python loc_offline.py traces/X_dev.json  (analysis only)"""
import json, sys
import numpy as np
from multiprocessing import Pool
from scipy.optimize import least_squares
sys.path.insert(0, "algos")
from sim import load_scenario
import tmp_base as M
from tmpl_fit import fit_tmpl_pc
import time

V = json.load(open("clone_value.json"))
TT = []
HARD = (1.72, 4.01, 0.836, 0.21)
SOFT = (1.25, 3.08, 0.65, 0.15)


def fit_fixed(T, P, C, Y, nc, w, soft, fl0, prior=0.1):
    ly = np.log(np.maximum(Y, 1e-4 * max(Y.max(), 1e-12)))
    seen = np.zeros(nc, bool); seen[np.unique(C)] = True
    def res(th):
        a, tc, pc, fl = th[:nc], th[nc:2 * nc], th[2 * nc:3 * nc], np.exp(th[3 * nc])
        d = M.drop(T - tc[C], w[0], w[1]) + M.drop(P - pc[C], w[2], w[3])
        pen = fl * (1 - np.exp(-d / fl)) if soft else np.minimum(d, fl)
        return np.concatenate([a[C] - pen - ly, prior * (tc - 0.5), prior * (pc - 0.5)])
    a0 = np.array([ly[C == c].max() if seen[c] else ly.mean() for c in range(nc)])
    t0 = np.array([T[C == c][np.argmax(ly[C == c])] if seen[c] else 0.5 for c in range(nc)])
    p0 = np.array([P[C == c][np.argmax(ly[C == c])] if seen[c] else 0.5 for c in range(nc)])
    lo = np.concatenate([np.full(nc, -60.0), np.full(nc, -0.2), np.full(nc, -0.2), [np.log(0.5)]])
    hi = np.concatenate([np.full(nc, 60.0), np.full(nc, 1.2), np.full(nc, 1.2), [np.log(12)]])
    best = None
    for k in range(3):
        rng = np.random.default_rng(k)
        st = np.concatenate([a0, t0 + (k > 0) * rng.normal(0, 0.08, nc), p0 + (k > 0) * rng.normal(0, 0.08, nc), [np.log(fl0)]])
        r = least_squares(res, np.clip(st, lo + 1e-9, hi - 1e-9), bounds=(lo, hi), loss="soft_l1", f_scale=0.3, max_nfev=400)
        if best is None or r.cost < best.cost:
            best = r
    x = best.x
    return {"height": x[:nc], "t": np.clip(x[nc:2 * nc], 0, 1), "p": np.clip(x[2 * nc:3 * nc], 0, 1), "seen": seen}


def run(r):
    sc = load_scenario(r["sid"]); names = [c["name"] for c in sc["clones"]]; nc = len(names)
    b = np.array(sc["bounds"], float); lo, span = b[:, 0], b[:, 1] - b[:, 0]
    tt = np.array([(np.clip(c["e_activation"], *b[0]) - lo[0]) / span[0] for c in sc["clones"]])
    tp = np.array([(np.clip(c["ph_opt"], *b[1]) - lo[1]) / span[1] for c in sc["clones"]])
    pot = np.array(V[str(r["sid"])])
    plates = []; screened = []
    stop = None
    for k, blk in enumerate(r["log"]):
        for x, y in zip(blk["x"], blk["y"]):
            if x[0] == "MTP": plates.append((names.index(x[1]), float(x[2]), float(x[3]), y))
            if x[0] == "MBR" and k > 0 and stop is None: stop = len(plates)
            if x[0] == "MBR" and k > 0: screened.append(names.index(x[1]))
    out = {}
    for tag, n in (("first_mbr", stop), ("all", len(plates))):
        if not n: continue
        P = plates[:n]
        C = np.array([p[0] for p in P]); T = (np.array([p[1] for p in P]) - lo[0]) / span[0]
        Pp = (np.array([p[2] for p in P]) - lo[1]) / span[1]; Y = np.array([p[3] for p in P])
        fits = {"quad": M.fit_quad_peaks(T, Pp, C, Y, nc, np.random.default_rng(0))}
        for nm, w, soft in (("hard", HARD, False), ("soft", SOFT, True)):
            wu = (w[0] / span[0], w[1] / span[0], w[2] / span[1], w[3] / span[1])
            fits[nm] = fit_fixed(T, Pp, C, Y, nc, wu, soft, 4.4 if not soft else 5.0)
            if soft:
                t0 = time.time(); fits["pc"] = fit_tmpl_pc(T, Pp, C, Y, nc, wu); TT.append(time.time() - t0)
        scr = sorted(set(screened)) or list(np.unique(C))
        for nm, f in fits.items():
            e = np.hypot((f["t"] - tt) * span[0] / 3.0, (f["p"] - tp) * span[1] / 0.5)  # scaled: 3 C ~ 0.5 pH
            et = np.abs(f["t"] - tt) * span[0]; ep = np.abs(f["p"] - tp) * span[1]
            h = np.where(f["seen"], f["height"], -np.inf); order = np.argsort(-h)
            out[(tag, nm)] = dict(et=et[scr].tolist(), ep=ep[scr].tolist(),
                                  top4=float(pot[order[:4]].max() / pot.max()), top8=float(pot[order[:8]].max() / pot.max()),
                                  top1=float(pot[order[0]] / pot.max()))
    return out


if __name__ == "__main__":
    T = json.load(open(sys.argv[1]))
    with Pool(20) as p:
        res = p.map(run, T)
    keys = sorted({k for o in res for k in o})
    for k in keys:
        d = [o[k] for o in res if k in o]
        et = np.concatenate([x["et"] for x in d]); ep = np.concatenate([x["ep"] for x in d])
        print(f"{k[0]:9s} {k[1]:5s} |T err| med {np.median(et):.2f} p75 {np.percentile(et, 75):.2f} p90 {np.percentile(et, 90):.2f} "
              f"|pH err| med {np.median(ep):.3f} p75 {np.percentile(ep, 75):.3f} p90 {np.percentile(ep, 90):.3f} | "
              f"rank top1 {np.mean([x['top1'] for x in d]):.3f} top4 {np.mean([x['top4'] for x in d]):.3f} top8 {np.mean([x['top8'] for x in d]):.3f}")
