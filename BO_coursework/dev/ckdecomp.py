"""Checkpoint loss decomposition of a traced run (analysis only).
python ckdecomp.py traces/ALGO_dev.json

For the best Pilot at each checkpoint (33/66/100 % of budget):
  clone  = potential of its clone (grid-best gln at true T/pH) / oracle
  loc    = same recipe with T/pH replaced by the clone's true optimum, relative to actual
  recipe = the rest (glutamine/glucose) = actual-at-true-T/pH / clone potential
"""
import json, sys
import numpy as np
from multiprocessing import Pool
from sim import Sim, load_scenario

V = json.load(open("clone_value.json"))
FR = (1 / 3, 2 / 3, 1.0)


def run(r):
    sc = load_scenario(r["sid"]); f = Sim(sc); names = [c["name"] for c in sc["clones"]]
    b = np.array(sc["bounds"], float)
    pot = np.array(V[str(r["sid"])])
    oracle = max(r["oracle"], pot.max())
    budget = sc["lab"]["budget"]
    rows = []
    for blk in r["log"][1:]:
        for x, y in zip(blk["x"], blk["y"]):
            if x[0] == "Pilot":
                rows.append((names.index(x[1]), np.array(x[2:], float), blk["spent"]))
    out = []
    for fr in FR:
        ck = fr * budget
        cand = [(f("Pilot", c, x), c, x) for c, x, s in rows if s <= ck + 1e-6]
        if not cand:
            out.append(dict(actual=0, clone=np.nan, attp=np.nan, n=0))
            continue
        a, c, x = max(cand, key=lambda t: t[0])
        cl = sc["clones"][c]
        xt = x.copy()
        xt[[0, 8, 10]] = np.clip(cl["e_activation"], *b[0]); xt[[1, 9, 11]] = np.clip(cl["ph_opt"], *b[1])
        at = f("Pilot", c, xt)
        terr = x[10] - np.clip(cl["e_activation"], *b[0]); perr = x[11] - np.clip(cl["ph_opt"], *b[1])
        out.append(dict(actual=a / oracle, clone=pot[c] / oracle, attp=at / oracle, n=len(cand),
                        rank=int((pot > pot[c]).sum()), terr=terr, perr=perr, best_any=pot.max() / oracle))
    return r["sid"], out


if __name__ == "__main__":
    T = json.load(open(sys.argv[1]))
    with Pool(20) as p:
        res = p.map(run, T)
    for k, fr in enumerate(FR):
        d = [o[k] for _, o in res]
        A = np.array([x["actual"] for x in d]); Cl = np.array([x["clone"] for x in d], float)
        At = np.array([x["attp"] for x in d], float); has = A > 0
        te = np.array([x.get("terr", np.nan) for x in d]); pe = np.array([x.get("perr", np.nan) for x in d])
        rk = np.array([x.get("rank", -1) for x in d])
        print(f"ck{k + 1}: actual {A.mean():.3f} | no pilot {np.mean(~has):.2f} | with pilot: clone {np.nanmean(Cl[has]):.3f} "
              f"at-true-T/pH {np.nanmean(At[has]):.3f} actual {A[has].mean():.3f} | clone top1 {np.mean(rk[has] == 0):.2f} "
              f"|T err| med {np.nanmedian(np.abs(te)):.2f} p90 {np.nanpercentile(np.abs(te), 90):.2f} "
              f"|pH err| med {np.nanmedian(np.abs(pe)):.3f} p90 {np.nanpercentile(np.abs(pe), 90):.3f} "
              f"T bias {np.nanmedian(te):+.2f} pH bias {np.nanmedian(pe):+.3f}")
        # loss shares (with pilot): clone 1-Cl, location (At-A), recipe (Cl-At)
        print(f"     loss: clone {np.nanmean(1 - Cl[has]):.3f}  recipe {np.nanmean(Cl[has] - At[has]):.3f}  "
              f"location {np.nanmean(At[has] - A[has]):.3f}  missing-pilot {np.mean(~has) * 1:.3f}")
