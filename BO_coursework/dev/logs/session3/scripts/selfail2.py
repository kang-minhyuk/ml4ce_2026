# For the "mbr lower" failures: the best-screened clone's MBR row(s): T/pH used vs true, and gln recipe
import json, sys, numpy as np
sys.path.insert(0, '.')
from sim import load_scenario, Sim
V = json.load(open("clone_value.json")); T = json.load(open(sys.argv[1]))
for r in T:
    sc = load_scenario(r["sid"]); names = [c["name"] for c in sc["clones"]]; f = Sim(sc); b = np.array(sc["bounds"])
    pot = np.array(V[str(r["sid"])]); orc = max(r["oracle"], pot.max())
    mbr = {}; pil = set()
    for blk in r["log"][1:]:
        for x, y in zip(blk["x"], blk["y"]):
            c = names.index(x[1])
            if x[0] == "MBR" and y > mbr.get(c, (None, -1))[1]: mbr[c] = (x, y)
            if x[0] == "Pilot": pil.add(c)
    if not r["pilots"]: continue
    inc = max(r["pilots"], key=lambda p: p["true"])["clone"]
    scr = set(mbr) | pil
    bb = max(scr, key=lambda c: pot[c])
    if bb == inc or bb in pil or bb not in mbr: continue
    x, y = mbr[bb]; cl = sc["clones"][bb]
    xr = np.array(x[2:], float); xt = xr.copy()
    xt[[0, 8, 10]] = np.clip(cl["e_activation"], *b[0]); xt[[1, 9, 11]] = np.clip(cl["ph_opt"], *b[1])
    print(f"sid {r['sid']:3d} y {y:7.0f} sim {f('MBR', bb, xr):7.0f} at-true-T/pH {f('MBR', bb, xt):7.0f} | dT {xr[0]-xt[0]:+.2f} dpH {xr[1]-xt[1]:+.2f} | inc MBR {mbr[inc][1] if inc in mbr else -1:7.0f}")
