"""Proxy oracle for hold-out splits: max over clones of the grid-best glutamine recipe at the true T/pH optimum
(≈0.95 of the DE oracle). Writes oracle_proxy.json {sid: {"oracle": value}}. python proxy_oracle.py SPLIT..."""
import json, sys, itertools
import numpy as np
from multiprocessing import Pool
from pathlib import Path
from sim import Sim, load_scenario
A0 = [.0005, .001, .002, .004, .008, .016, .03, .06, .12]; A1 = [.002, .005, .01, .02, .04, .08, .15]; B2 = [.25, .4, .6, .85]


def run(sid):
    sc = load_scenario(sid); f = Sim(sc); b = np.array(sc["bounds"], float); lo, sp = b[:, 0], b[:, 1] - b[:, 0]
    best = 0
    for ci, c in enumerate(sc["clones"]):
        for a0, a1, b2 in itertools.product(A0, A1, B2):
            u = np.array([0, 0, 1, a0, 1, a1, 1, b2, 0, 0, 0, 0.]); x = lo + u * sp
            x[[0, 8, 10]] = np.clip(c["e_activation"], *b[0]); x[[1, 9, 11]] = np.clip(c["ph_opt"], *b[1])
            best = max(best, f("Pilot", ci, x))
    return sid, best


if __name__ == "__main__":
    SP = json.load(open("splits.json")); out = Path("oracle_proxy.json")
    d = json.loads(out.read_text()) if out.exists() else {}
    ids = [i for s in sys.argv[1:] for i in SP[s] if str(i) not in d]
    with Pool(int(__import__("os").environ.get("J", "8"))) as p:
        for sid, v in p.imap_unordered(run, ids):
            d[str(sid)] = {"oracle": v}
    out.write_text(json.dumps(d)); print("proxy oracles:", len(d))
