import json, numpy as np
from multiprocessing import Pool
from sim import Sim, load_scenario
O = json.load(open("oracle.json")); DEV = json.load(open("splits.json"))["dev"][:6]
A0 = np.logspace(-4, -0.7, 12)
def run(sid):
    sc = load_scenario(sid); f = Sim(sc); b = f.bounds; o = O[str(sid)]; ci = o["clone"]; c = sc["clones"][ci]
    lo, sp = b[:, 0], b[:, 1] - b[:, 0]
    tc = np.clip(c["e_activation"], *b[0]); pc = np.clip(c["ph_opt"], *b[1])
    def val(r, a0, a1, b2, cc=ci):
        u = np.array([0, 0, 1, a0, 1, a1, 1, b2, 0, 0, 0, 0.]); x = lo + u * sp
        cl = sc["clones"][cc]; x[[0, 8, 10]] = np.clip(cl["e_activation"], *b[0]); x[[1, 9, 11]] = np.clip(cl["ph_opt"], *b[1]); return f(r, cc, x)
    out = {}
    for r in ("Pilot", "MBR"):
        out[r + " a0"] = [val(r, a, .027, .37) for a in A0]
        out[r + " a1"] = [val(r, .003, a, .37) for a in A0]
        out[r + " a0 other clone"] = [val(r, a, .027, .37, (ci + 1) % len(sc["clones"])) for a in A0]
    return sid, out
if __name__ == "__main__":
    with Pool(6) as p: res = p.map(run, DEV)
    print("a0 grid", np.round(A0, 4))
    for sid, out in res:
        print("sid", sid)
        for k, v in out.items():
            v = np.array(v); print(f"  {k:18s}", np.round(v / v.max(), 2), "argmax", round(A0[np.argmax(v)], 4))
