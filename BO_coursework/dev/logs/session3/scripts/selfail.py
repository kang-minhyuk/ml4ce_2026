import json, sys, numpy as np
sys.path.insert(0, '.')
from sim import load_scenario
V = json.load(open("clone_value.json")); T = json.load(open(sys.argv[1]))
cnt = {"b_piloted": 0, "b_mbr_lower": 0, "b_mbr_higher_not_piloted": 0}; n = 0; loss = []
for r in T:
    sc = load_scenario(r["sid"]); names = [c["name"] for c in sc["clones"]]
    pot = np.array(V[str(r["sid"])]); orc = max(r["oracle"], pot.max())
    mbr = {}; pil = {}
    for blk in r["log"][1:]:
        for x, y in zip(blk["x"], blk["y"]):
            c = names.index(x[1])
            if x[0] == "MBR": mbr[c] = max(mbr.get(c, 0), y)
            if x[0] == "Pilot": pil.setdefault(c, []).append(y)
    scr = set(mbr) | set(pil)
    if not r["pilots"]: continue
    inc = max(r["pilots"], key=lambda p: p["true"])["clone"]
    b = max(scr, key=lambda c: pot[c])
    if b == inc: continue
    n += 1; loss.append((pot[b] - pot[inc]) / orc)
    first_pilot = r["pilots"][0]["clone"]
    if b in pil: cnt["b_piloted"] += 1; tag = "piloted"
    elif mbr.get(b, 0) < mbr.get(inc, 0): cnt["b_mbr_lower"] += 1; tag = "mbr lower"
    else: cnt["b_mbr_higher_not_piloted"] += 1; tag = "mbr higher"
    print(f"sid {r['sid']:3d} loss {(pot[b]-pot[inc])/orc:.2f} {tag:10s} inc pot {pot[inc]/orc:.2f} mbr {mbr.get(inc,0):7.0f} pil {[round(v) for v in pil[inc]]} | best-scr pot {pot[b]/orc:.2f} mbr {mbr.get(b,0):7.0f} pil {[round(v) for v in pil.get(b,[])]}")
print(n, cnt, "mean loss when wrong", np.mean(loss), "total", np.sum(loss) / len(T))
