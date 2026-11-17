# Location error at MBR-screen time: algorithm estimate vs best plate well vs truth, for all screened clones
import json, sys, numpy as np
sys.path.insert(0, '.')
from sim import load_scenario
T = json.load(open(sys.argv[1]))
A = []; Bw = []; N = []; big = 0
for r in T:
    sc = load_scenario(r["sid"]); names = [c["name"] for c in sc["clones"]]; b = np.array(sc["bounds"])
    span = b[:, 1] - b[:, 0]
    plates = []; seen = set()
    for blk in r["log"]:
        for x, y in zip(blk["x"], blk["y"]):
            c = names.index(x[1])
            if x[0] == "MTP": plates.append((c, float(x[2]), float(x[3]), y))
            if x[0] == "MBR" and blk is not r["log"][0] and c not in seen:
                seen.add(c)
                cl = sc["clones"][c]; tt = np.clip(cl["e_activation"], *b[0]); tp = np.clip(cl["ph_opt"], *b[1])
                own = [(t, p, yy) for cc, t, p, yy in plates if cc == c]
                ea = np.hypot((float(x[2]) - tt) / span[0], (float(x[3]) - tp) / span[1])
                if own:
                    t, p, _ = max(own, key=lambda z: z[2])
                    eb = np.hypot((t - tt) / span[0], (p - tp) / span[1])
                else: eb = np.nan
                A.append(ea); Bw.append(eb); N.append(len(own))
A, Bw, N = map(np.array, (A, Bw, N))
print("n", len(A), "wells/clone med", np.median(N))
for q in (50, 75, 90, 95):
    print(q, "est %.3f  best-well %.3f" % (np.percentile(A, q), np.nanpercentile(Bw, q)))
print("frac est err>0.2 (unit)", np.mean(A > 0.2), "best-well>0.2", np.nanmean(Bw > 0.2), "est worse than well", np.nanmean(A > Bw))
print("among est>0.2: best-well err med", np.nanmedian(Bw[A > 0.2]), "wells", np.median(N[A > 0.2]))
