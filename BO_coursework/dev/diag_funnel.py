"""Diagnose a funnel algorithm's trace: python diag_funnel.py traces/ALGO_dev.json
Losses at the first Pilot: coverage (best MBR-tested clone vs best clone, both at the structured heuristic with the
true T/pH optimum), selection (chosen clone vs best tested), recipe (actual vs chosen clone's heuristic).
Also the plate peak-fit error of the T/pH optimum for the chosen clone."""
import json, sys
import numpy as np
from multiprocessing import Pool
from sim import Sim, load_scenario
from decomp import heur
sys.path.insert(0, "algos")


def run(r):
    import i5_funnel as M
    sc = load_scenario(r["sid"]); f = Sim(sc); names = [c["name"] for c in sc["clones"]]
    b = np.array(sc["bounds"], float); lo, span = b[:, 0], b[:, 1] - b[:, 0]
    V = json.load(open("clone_value.json"))
    hs = (np.array(V[str(r["sid"])]) if str(r["sid"]) in V else
          np.array([heur(f, sc, ci) for ci in range(len(names))])) / r["oracle"]
    rows = [(row, y, blk["spent"]) for blk in r["log"] for row, y in zip(blk["x"], blk["y"])]
    first_p = next((k for k, (row, _, _) in enumerate(rows) if row[0] == "Pilot"), None)
    if first_p is None:
        return None
    before = rows[:first_p]
    tested = sorted({names.index(row[1]) for row, _, _ in before if row[0] == "MBR"})
    pc = names.index(rows[first_p][0][1])
    plates = [(row, y) for row, y, _ in before if row[0] == "MTP"]
    terr = perr = np.nan
    if plates:
        U = np.array([(np.array(row[2:6], float) - lo[:4]) / span[:4] for row, _ in plates])
        C = np.array([names.index(row[1]) for row, _ in plates]); Y = np.array([y for _, y in plates])
        pk = M.fit_peaks(U[:, 0], U[:, 1], C, Y, len(names), np.random.default_rng(0))
        c = sc["clones"][pc]
        terr = (pk["t"][pc] * span[0] + lo[0]) - np.clip(c["e_activation"], *b[0])
        perr = (pk["p"][pc] * span[1] + lo[1]) - np.clip(c["ph_opt"], *b[1])
    actual = r["pilots"][0]["true"] / r["oracle"]
    best_all = max(r["pilots"], key=lambda p: p["true"])["true"] / r["oracle"]
    return dict(sid=r["sid"], best=hs.max(), cover=hs[tested].max() if tested else 0, chosen=hs[pc], actual=actual,
                best_all=best_all, n_tested=len(tested), n_plate=len(plates) // 24, terr=terr, perr=perr,
                rank=int((hs > hs[pc]).sum()), spent=rows[first_p][2])


if __name__ == "__main__":
    T = json.load(open(sys.argv[1]))
    with Pool(21) as p:
        res = [x for x in p.map(run, T) if x]
    k = lambda key: np.array([x[key] for x in res], float)
    print(f"n={len(res)}  heuristic(best clone) {k('best').mean():.3f} | best tested {k('cover').mean():.3f} | "
          f"chosen {k('chosen').mean():.3f} | first pilot actual {k('actual').mean():.3f} | best pilot {k('best_all').mean():.3f}")
    print(f"tested clones {k('n_tested').mean():.1f}, plates {k('n_plate').mean():.1f}, chosen-clone rank median "
          f"{np.median(k('rank')):.0f}; |T err| median {np.nanmedian(np.abs(k('terr'))):.2f} C p90 "
          f"{np.nanpercentile(np.abs(k('terr')), 90):.2f}; |pH err| median {np.nanmedian(np.abs(k('perr'))):.3f}")
    for x in sorted(res, key=lambda x: x["best_all"])[:12]:
        print({kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in x.items()})
