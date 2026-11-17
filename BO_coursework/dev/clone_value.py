"""Per-clone Pilot potential: grid-best glutamine recipe at the true T/pH optimum (analysis only).
python clone_value.py SPLIT N -> clone_value.json {sid: [value per clone]}"""
import json, sys, itertools
import numpy as np
from multiprocessing import Pool
from pathlib import Path
from sim import Sim, load_scenario
A0 = [.001, .003, .008, .02, .05]; A1 = [.003, .008, .02, .05, .12]; B2 = [.3, .6]


def potential(f, sc, ci, reactor="Pilot"):
    b = np.array(sc["bounds"], float); lo, sp = b[:, 0], b[:, 1] - b[:, 0]; c = sc["clones"][ci]
    best = 0
    for a0, a1, b2 in itertools.product(A0, A1, B2):
        u = np.array([0, 0, 1, a0, 1, a1, 1, b2, 0, 0, 0, 0.]); x = lo + u * sp
        x[[0, 8, 10]] = np.clip(c["e_activation"], *b[0]); x[[1, 9, 11]] = np.clip(c["ph_opt"], *b[1])
        best = max(best, f(reactor, ci, x))
    return best


def run(sid):
    sc = load_scenario(sid); f = Sim(sc)
    return sid, [potential(f, sc, ci) for ci in range(len(sc["clones"]))]


if __name__ == "__main__":
    ids = json.load(open("splits.json"))[sys.argv[1]][:int(sys.argv[2])]
    out = Path("clone_value.json"); d = json.loads(out.read_text()) if out.exists() else {}
    ids = [i for i in ids if str(i) not in d]
    with Pool(12) as p:
        for sid, v in p.imap_unordered(run, ids):
            d[str(sid)] = v
    out.write_text(json.dumps(d))
    print("done", len(d))
