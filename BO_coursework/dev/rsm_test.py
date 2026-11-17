import json, numpy as np, sobol_seq
from multiprocessing import Pool
from scipy.optimize import least_squares
from sim import Sim, load_scenario
from decomp import heur
O = json.load(open("oracle.json")); dev = json.load(open("splits.json"))["dev"][:50]
MED = (0.3, 0.3)

def fit_rsm(T, P, C, Y, nc):
    """log y = a_c - ((T-Tc)/wT)^2 - ((P-Pc)/wP)^2 ; shared wT, wP (asymmetric optional) ; robust."""
    ly = np.log(np.maximum(Y, 1e-6 * Y.max()) )
    def unpack(th):
        a = th[:nc]; Tc = th[nc:2*nc]; Pc = th[2*nc:3*nc]; lwT, lwP = th[3*nc:3*nc+2]
        return a, Tc, Pc, np.exp(lwT), np.exp(lwP)
    def res(th):
        a, Tc, Pc, wT, wP = unpack(th)
        r = a[C] - ((T - Tc[C]) / wT) ** 2 - ((P - Pc[C]) / wP) ** 2 - ly
        reg = np.concatenate([(Tc - 0.5) * 0.3, (Pc - 0.5) * 0.3])  # weak pull to centre for unidentified clones
        return np.concatenate([r, reg])
    a0 = np.array([ly[C == c].max() if np.any(C == c) else ly.mean() for c in range(nc)])
    T0 = np.array([T[C == c][np.argmax(ly[C == c])] if np.any(C == c) else 0.5 for c in range(nc)])
    P0 = np.array([P[C == c][np.argmax(ly[C == c])] if np.any(C == c) else 0.5 for c in range(nc)])
    th0 = np.concatenate([a0, T0, P0, [np.log(0.4), np.log(0.4)]])
    lo = np.concatenate([np.full(nc, -50), np.full(nc, -0.3), np.full(nc, -0.3), [np.log(0.05)] * 2])
    hi = np.concatenate([np.full(nc, 50), np.full(nc, 1.3), np.full(nc, 1.3), [np.log(3)] * 2])
    r = least_squares(res, np.clip(th0, lo, hi), bounds=(lo, hi), loss="soft_l1", f_scale=0.5)
    return unpack(r.x)

def run(sid):
    sc = load_scenario(sid); f = Sim(sc); b = np.array(sc["bounds"]); lo, hi = b[:, 0], b[:, 1]; sp = hi - lo
    n = len(sc["clones"])
    out = {}
    H = np.array([heur(f, sc, ci) for ci in range(n)])
    for k in (3, 5, 8):
        des = sobol_seq.i4_sobol_generate(2, k, 3)
        T, P, C, Y = [], [], [], []
        for t, p in des:
            for ci in range(n):
                x = np.zeros(12); x[0] = x[8] = x[10] = lo[0] + t * sp[0]; x[1] = x[9] = x[11] = lo[1] + p * sp[1]
                x[2] = lo[2] + MED[0] * sp[2]; x[3] = lo[3] + MED[1] * sp[3]
                T.append(t); P.append(p); C.append(ci); Y.append(f("MTP", ci, x))
        a, Tc, Pc, wT, wP = fit_rsm(np.array(T), np.array(P), np.array(C), np.array(Y), n)
        errT = np.abs(lo[0] + Tc * sp[0] - np.array([c["e_activation"] for c in sc["clones"]]))
        errP = np.abs(lo[1] + Pc * sp[1] - np.array([c["ph_opt"] for c in sc["clones"]]))
        # value of structured heuristic at estimated T/pH for the oracle-best heuristic clone
        best = int(np.argmax(H)); vals = []
        for ci in [best]:
            v = 0
            for q2 in (0.25, 0.4, 0.6):
                u = np.array([0, 0, 1, 0.02, 1, 0.03, 1, q2, 0, 0, 0, 0.]); x = lo + u * sp
                x[[0, 8, 10]] = np.clip(lo[0] + Tc[ci] * sp[0], lo[0], hi[0]); x[[1, 9, 11]] = np.clip(lo[1] + Pc[ci] * sp[1], lo[1], hi[1])
                v = max(v, f("Pilot", ci, x))
            vals.append(v / H[ci])
        out[k] = (np.median(errT), np.median(errP), vals[0], np.corrcoef(a, np.log(H))[0, 1])
    return out
if __name__ == "__main__":
    with Pool(21) as p: res = p.map(run, dev)
    for k in (3, 5, 8):
        v = np.array([r[k] for r in res])
        print(f"{k} plates: median |T err| {np.mean(v[:,0]):.2f} C, |pH err| {np.mean(v[:,1]):.2f}; best clone heuristic at est/true {np.mean(v[:,2]):.3f} (p10 {np.percentile(v[:,2],10):.2f}); corr(a_c, log Pilot-heur) {np.mean(v[:,3]):.2f}")
