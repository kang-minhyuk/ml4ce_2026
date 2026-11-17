import json, os, sys, importlib.util
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, "../scenario_suite"); sys.path.insert(0, ".")
import numpy as np, run_suite
import digital_lab.lab as L
from digital_lab import experiment
from multiprocessing import Pool
ALG = sys.argv[1]
def one(sid):
    sc = json.loads((run_suite.SUITE / "scenarios" / f"scenario_{sid:04d}.json").read_text())
    names = [c["name"] for c in sc["clones"]]
    L.CELLTYPES = names; L.LAB_CONFIG_FILENAMES = {**L.LAB_CONFIG_FILENAMES, **{n: f"{n}.json" for n in names}}
    L.CONTINUOUS_BOUNDS = [tuple(b) for b in sc["bounds"]]
    spec = importlib.util.spec_from_file_location("m", ALG); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    ev = []
    o1, o2 = m.first_pilot_recipe, m.feat_recipe
    def fpr(st, c):
        r = o2(st, c) if st.cfg.get("feat_first") else None
        ev.append((st.n_plates_own, r is not None, None if r is None else tuple(round(v, 2) for v in r[4:]), round(st.sp.lab.spent)))
        return o1(st, c)
    m.first_pilot_recipe = fpr
    experiment.run_algorithm(m.my_algorithm, "t", sc["seed"], {"config_dir": str(run_suite.materialise(sc)), **sc["lab"]})
    return sid, ev
if __name__ == "__main__":
    ids = json.load(open("splits.json"))["hold"][:16]
    with Pool(4) as p:
        for sid, ev in p.map(one, ids): print(sid, ev[:5])
