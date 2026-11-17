import json, numpy as np
D = json.load(open("plate_mod_study.json")); V = json.load(open("clone_value.json")); S = json.load(open("screen_study.json"))
OFF = [(0, 0), (-1, 0), (1, 0), (0, -0.15), (0, 0.15), (-2, 0), (2, 0), (0, -0.3), (0, 0.3), (1, -0.15), (-1, 0.15)]
MEDIA = ["mod(.3,.3)", "rich(1,.05)", "mid(.5,.2)"]
no = len(OFF)
for mi, mname in enumerate(MEDIA):
    print("==", mname)
    # mean log titer relative to offset 0 (shape)
    rel = []
    for s in D:
        a = np.log(np.maximum(np.array(D[s])[:, mi * no:(mi + 1) * no], 1e-3)); rel.append(a - a[:, :1])
    rel = np.vstack(rel)
    print("mean log rel by offset:", " ".join(f"{o}:{m:+.2f}" for o, m in zip(OFF, np.median(rel, 0))))
    for oi, o in enumerate(OFF):
        c1 = []; c4 = []
        for s in D:
            pot = np.array(V[s]); sc = np.array(D[s])[:, mi * no + oi]; od = np.argsort(-sc)
            c1.append(pot[od[0]] / pot.max()); c4.append(pot[od[:4]].max() / pot.max())
        print(f"  all clones at offset {o}: top1 {np.mean(c1):.3f} top4 {np.mean(c4):.3f}")
    # random per-clone offsets among the 9 small ones (realistic error)
    rng = np.random.default_rng(0); c1 = []; c4 = []
    for s in D:
        pot = np.array(V[s]); A = np.array(D[s])[:, mi * no:(mi + 1) * no]
        for _ in range(20):
            pick = rng.integers(0, 5, len(pot)); sc = A[np.arange(len(pot)), pick]; od = np.argsort(-sc)
            c1.append(pot[od[0]] / pot.max()); c4.append(pot[od[:4]].max() / pot.max())
    print(f"  random small offsets per clone: top1 {np.mean(c1):.3f} top4 {np.mean(c4):.3f}")
