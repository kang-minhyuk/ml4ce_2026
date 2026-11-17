"""Greedy portfolio of fixed Pilot glutamine recipes (oracle clone, true T/pH): which k recipes maximise mean best?
Fit on DEV oracle clones, check on official-DEV oracle clones (analysis only)."""
import json, itertools, numpy as np
from multiprocessing import Pool
from sim import Sim, load_scenario
O = json.load(open("oracle.json")); SP = json.load(open("splits.json"))
A0 = [.0005, .001, .002, .004, .008, .016, .03, .06]; A1 = [.002, .005, .01, .02, .04, .08, .15]; B2 = [.25, .4, .6, .85]
GRID = list(itertools.product(A0, A1, B2))
def run(sid):
    sc = load_scenario(sid); f = Sim(sc); b = f.bounds; o = O[str(sid)]; ci = o["clone"]; c = sc["clones"][ci]
    lo, sp = b[:, 0], b[:, 1] - b[:, 0]; out = []
    for a0, a1, b2 in GRID:
        u = np.array([0, 0, 1, a0, 1, a1, 1, b2, 0, 0, 0, 0.]); x = lo + u * sp
        x[[0, 8, 10]] = np.clip(c["e_activation"], *b[0]); x[[1, 9, 11]] = np.clip(c["ph_opt"], *b[1])
        out.append(f("Pilot", ci, x) / o["oracle"])
    return out
if __name__ == "__main__":
    with Pool(20) as p:
        D = np.array(p.map(run, SP["dev"])); F = np.array(p.map(run, SP["off_dev"]))
    np.save("gln_grid_dev.npy", D); np.save("gln_grid_off.npy", F)
    chosen = []
    for k in range(5):
        best = max(range(len(GRID)), key=lambda j: np.max(D[:, chosen + [j]], 1).mean())
        chosen.append(best)
        print(k + 1, GRID[best], "dev mean best %.3f | off %.3f" % (np.max(D[:, chosen], 1).mean(), np.max(F[:, chosen], 1).mean()))
    pr = GRID.index((.008, .005, .6)); print("prior (.008,.005,.6) dev %.3f off %.3f" % (D[:, pr].mean(), F[:, pr].mean()))
    print("grid-best per scenario dev %.3f off %.3f" % (D.max(1).mean(), F.max(1).mean()))
