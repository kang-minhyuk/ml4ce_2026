import json, sys, numpy as np
sys.path.insert(0, '.')
from sim import load_scenario
V = json.load(open("clone_value.json"))
T = json.load(open(sys.argv[1]))
O = json.load(open("oracle.json"))
R = []
for r in T:
    sc = load_scenario(r["sid"]); names = [c["name"] for c in sc["clones"]]
    pot = np.array(V[str(r["sid"])]); orc = max(r["oracle"], pot.max())
    mbr = set(); pil = {}; npl = 0; nmbr = 0
    for blk in r["log"][1:]:
        for x, y in zip(blk["x"], blk["y"]):
            c = names.index(x[1])
            if x[0] == "MBR": mbr.add(c); nmbr += 1
            if x[0] == "Pilot": pil[c] = pil.get(c, 0) + 1
            if x[0] == "MTP": npl += 1
    init = set(names.index(x[1]) for x in r["log"][0]["x"] if x[0] != "MTP")
    scr = mbr | set(pil)
    best_scr = max(pot[list(scr)]) / orc if scr else 0
    inc = max(r["pilots"], key=lambda p: p["true"])["clone"] if r["pilots"] else None
    R.append(dict(nc=len(names), best=pot.max() / orc, scr=best_scr, inc=pot[inc] / orc if inc is not None else 0,
                  nscr=len(scr), npil=sum(pil.values()), nclp=len(pil), nmbr=nmbr, nplate_wells=npl,
                  second=np.sort(pot)[-2] / orc, budget=sc["lab"]["budget"], costs=sc["lab"]["reactor_costs"],
                  maxexp=sc["lab"]["max_experiments"]))
k = lambda key: np.array([x[key] for x in R], float)
print("n", len(R), "clones/scen", k("nc").mean())
print(f"best clone pot {k('best').mean():.3f} 2nd {k('second').mean():.3f} | best screened {k('scr').mean():.3f} | incumbent {k('inc').mean():.3f}")
print(f"screened {k('nscr').mean():.1f} (MBR rows {k('nmbr').mean():.1f}) pilots {k('npil').mean():.1f} on {k('nclp').mean():.1f} clones; plate wells {k('nplate_wells').mean():.0f}")
print("frac best screened", np.mean(np.isclose(k('scr'), k('best'))), "frac inc==best screened", np.mean(np.isclose(k('inc'), k('scr'))))
