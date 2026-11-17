import json, itertools, numpy as np
from multiprocessing import Pool
from sim import Sim, load_scenario
O = json.load(open("oracle.json")); DEV = json.load(open("splits.json"))["dev"][:20]
DT = [-2, -1, -0.5, 0.5, 1, 2]; DP = [-0.3, -0.2, -0.1, 0.1, 0.2, 0.3]
def run(sid):
    sc = load_scenario(sid); f = Sim(sc); b = f.bounds; o = O[str(sid)]; ci = o["clone"]; c = sc["clones"][ci]
    lo, sp = b[:, 0], b[:, 1] - b[:, 0]
    def val(dt, dp, a0, a1, b2):
        u = np.array([0, 0, 1, a0, 1, a1, 1, b2, 0, 0, 0, 0.]); x = lo + u * sp
        x[[0, 8, 10]] = c["e_activation"] + dt; x[[1, 9, 11]] = c["ph_opt"] + dp
        return f("Pilot", ci, np.clip(x, b[:, 0], b[:, 1]))
    g = max(itertools.product([.001, .003, .008, .02, .05], [.003, .008, .02, .05, .12], [.3, .6]), key=lambda z: val(0, 0, *z))
    v0 = val(0, 0, *g)
    return [val(d, 0, *g) / v0 for d in DT] + [val(0, d, *g) / v0 for d in DP]
if __name__ == "__main__":
    with Pool(20) as p: r = np.array(p.map(run, DEV))
    print("dT", DT, r[:, :6].mean(0).round(2)); print("dpH", DP, r[:, 6:].mean(0).round(2))
