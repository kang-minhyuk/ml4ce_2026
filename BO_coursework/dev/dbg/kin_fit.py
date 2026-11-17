"""Offline Q1: can the textbook kinetic model reproduce simulated Pilot glutamine surfaces (library members)?
Fit theta to a member's log titer surface (free scale), report fit RMSE and the true relative titer at the model's
optimum (in-bounds grid). Compares with the library baseline numbers (first Pilot 0.849, best of 2 0.887)."""
import numpy as np, sys, json
from multiprocessing import Pool
from scipy.optimize import least_squares
sys.path.insert(0, "dbg")
from kin_model import simulate, LO, HI
from sim import load_scenario
D = {k: v for k, v in np.load("gln_lib.npz").items()}
A0, A1, B2 = D["A0"], D["A1"], D["B2"]
a0 = np.repeat(A0, 84); a1 = np.tile(np.repeat(A1, 7), 13); b2 = np.tile(B2, 156)


def fit_member(j, idx=None, starts=6, seed=0):
    sc = load_scenario(int(D["sid"][j])); hi = np.array(sc["bounds"])[:, 1]
    y = np.log(np.maximum(D["Y"][j], 1e-3 * D["Y"][j].max()))
    ok = a0 <= D["gmax"][j] * 1.0001
    use = np.where(ok)[0] if idx is None else idx
    def res(th):
        p = np.log(np.maximum(simulate(th, hi[2], a0[use], hi[4], a1[use], hi[6], b2[use]), 1e-12))
        r = p - y[use]
        return r - r.mean()
    rng = np.random.default_rng(seed); best = None
    for s in range(starts):
        th0 = LO + rng.random(len(LO)) * (HI - LO)
        try:
            f = least_squares(res, th0, bounds=(LO, HI), max_nfev=300, diff_step=1e-3)
        except Exception:
            continue
        if best is None or f.cost < best.cost: best = f
    th = best.x
    pall = simulate(th, hi[2], a0, hi[4], a1, hi[6], b2)
    pall = np.where(ok, pall, -1)
    R = D["Y"][j] / np.where(ok, D["Y"][j], 0).max()
    return {"j": int(j), "rmse": float(np.sqrt(2 * best.cost / len(use))), "rel_at_model_opt": float(R[pall.argmax()]),
            "theta": th.tolist()}


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    js = np.random.default_rng(1).choice(len(D["Y"]), n, replace=False)
    with Pool(20) as p:
        out = p.map(fit_member, js)
    json.dump(out, open("dbg/kin_fit_q1.json", "w"))
    r = np.array([o["rel_at_model_opt"] for o in out]); e = np.array([o["rmse"] for o in out])
    print("n %d | log-titer RMSE median %.3f mean %.3f | true relative titer at model optimum: mean %.3f median %.3f p10 %.3f"
          % (n, np.median(e), e.mean(), r.mean(), np.median(r), np.quantile(r, .1)))
