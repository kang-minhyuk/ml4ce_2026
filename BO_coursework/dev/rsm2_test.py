import json, numpy as np, sobol_seq
from multiprocessing import Pool
from scipy.optimize import least_squares
from sim import Sim, load_scenario
from decomp import heur
dev = json.load(open("splits.json"))["dev"][:50]
MED = (1.0, 0.05)

def bump(d, wl, wr):
    return np.where(d < 0, (d / wl) ** 2, (d / wr) ** 2)

def fit(T, P, C, Y, nc, plate_id, n_plates):
    ly = np.log(np.maximum(Y, 1e-4 * Y.max()))
    def unpack(th):
        a, Tc, Pc = th[:nc], th[nc:2 * nc], th[2 * nc:3 * nc]
        w = np.exp(th[3 * nc:3 * nc + 4]); fl = th[3 * nc + 4]
        return a, Tc, Pc, w, fl
    def pred(th):
        a, Tc, Pc, w, fl = unpack(th)
        drop = bump(T - Tc[C], w[0], w[1]) + bump(P - Pc[C], w[2], w[3])
        return a[C] - np.minimum(drop, np.exp(fl))
    def res(th):
        a, Tc, Pc, w, fl = unpack(th)
        return np.concatenate([pred(th) - ly, 0.1 * (Tc - 0.5), 0.1 * (Pc - 0.5)])
    best = None
    rng = np.random.default_rng(0)
    for s in range(4):
        a0 = np.array([ly[C == c].max() if np.any(C == c) else ly.mean() for c in range(nc)])
        T0 = np.array([T[C == c][np.argmax(ly[C == c])] if np.any(C == c) else 0.5 for c in range(nc)]) + rng.normal(0, 0.05 * s, nc)
        P0 = np.array([P[C == c][np.argmax(ly[C == c])] if np.any(C == c) else 0.5 for c in range(nc)]) + rng.normal(0, 0.05 * s, nc)
        w0 = np.log([0.3, 0.6, 0.6, 0.2]) + rng.normal(0, 0.3 * (s > 0), 4)
        th0 = np.concatenate([a0, T0, P0, w0, [np.log(3.0)]])
        lo = np.concatenate([np.full(nc, -50), np.full(nc, -0.2), np.full(nc, -0.2), [np.log(0.03)] * 4, [-3]])
        hi = np.concatenate([np.full(nc, 50), np.full(nc, 1.2), np.full(nc, 1.2), [np.log(5)] * 4, [3]])
        r = least_squares(res, np.clip(th0, lo + 1e-9, hi - 1e-9), bounds=(lo, hi), loss="soft_l1", f_scale=0.3)
        if best is None or r.cost < best.cost: best = r
    return unpack(best.x)

def run(sid):
    sc = load_scenario(sid); f = Sim(sc); b = np.array(sc["bounds"]); lo, hi = b[:, 0], b[:, 1]; sp = hi - lo
    n = len(sc["clones"])
    H = np.array([heur(f, sc, ci) for ci in range(n)])
    out = {}
    for k in (3, 5, 8):
        des = sobol_seq.i4_sobol_generate(2, k, 3)
        T, P, C, Y = [], [], [], []
        for t, p in des:
            for ci in range(n):
                x = np.zeros(12); x[0] = x[8] = x[10] = lo[0] + t * sp[0]; x[1] = x[9] = x[11] = lo[1] + p * sp[1]
                x[2] = lo[2] + MED[0] * sp[2]; x[3] = lo[3] + MED[1] * sp[3]
                T.append(t); P.append(p); C.append(ci); Y.append(f("MTP", ci, x))
        a, Tc, Pc, w, fl = fit(np.array(T), np.array(P), np.array(C), np.array(Y), n, None, k)
        errT = np.abs(lo[0] + Tc * sp[0] - np.array([c["e_activation"] for c in sc["clones"]]))
        errP = np.abs(lo[1] + Pc * sp[1] - np.array([c["ph_opt"] for c in sc["clones"]]))
        best = int(np.argmax(H)); v = 0
        for q2 in (0.25, 0.4, 0.6):
            u = np.array([0, 0, 1, 0.02, 1, 0.03, 1, q2, 0, 0, 0, 0.]); x = lo + u * sp
            x[[0, 8, 10]] = np.clip(lo[0] + Tc[best] * sp[0], lo[0], hi[0]); x[[1, 9, 11]] = np.clip(lo[1] + Pc[best] * sp[1], lo[1], hi[1])
            v = max(v, f("Pilot", best, x))
        out[k] = (np.median(errT), np.median(errP), v / H[best], np.corrcoef(a, np.log(H))[0, 1])
    return out
if __name__ == "__main__":
    with Pool(21) as p: res = p.map(run, dev)
    for k in (3, 5, 8):
        v = np.array([r[k] for r in res])
        print(f"{k} plates: |T err| {np.mean(v[:,0]):.2f} C, |pH err| {np.mean(v[:,1]):.2f}; best-clone heuristic at est/true {np.mean(v[:,2]):.3f} (p10 {np.percentile(v[:,2],10):.2f}); corr(a_c, log Pilot-heur) {np.mean(v[:,3]):.2f}")
