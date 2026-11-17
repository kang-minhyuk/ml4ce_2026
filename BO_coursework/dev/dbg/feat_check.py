"""Compare the pipeline's plate peak height (feature) with the simulator's plate titer at the true optimum."""
import json, os, sys, importlib.util
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, "../scenario_suite"); sys.path.insert(0, ".")
import numpy as np, run_suite
import digital_lab.lab as L
from digital_lab import experiment
from sim import Sim
from multiprocessing import Pool

def one(sid):
    sc = json.loads((run_suite.SUITE / "scenarios" / f"scenario_{sid:04d}.json").read_text())
    names = [c["name"] for c in sc["clones"]]
    L.CELLTYPES = names
    L.LAB_CONFIG_FILENAMES = {**L.LAB_CONFIG_FILENAMES, **{n: f"{n}.json" for n in names}}
    L.CONTINUOUS_BOUNDS = [tuple(b) for b in sc["bounds"]]
    spec = importlib.util.spec_from_file_location("m", "algos/v1_f03L.py"); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    rec = {}
    orig = m.feat_prior
    def fp(st, c):
        if st.peaks is not None and st.peaks["seen"][c]:
            sp = st.sp
            wl = [(sp.unit(r), y) for r, y in st.split()[0] if sp.clones.index(r[1]) == c]
            U = np.array([u for u, _ in wl]); ly = np.log(np.maximum([y for _, y in wl], 1.0))
            w = (m.TMPL_W[0] / sp.span[0], m.TMPL_W[1] / sp.span[0], m.TMPL_W[2] / sp.span[1], m.TMPL_W[3] / sp.span[1])
            bl = m.bayes_locations(U[:, 0], U[:, 1], np.zeros(len(U), int), np.exp(ly), 1, w, sig=st.cfg["bayes_sig"])
            pen = m.template_pen(U[:, 0] - bl["t_map"][0], U[:, 1] - bl["p_map"][0], w, 5.0)
            near = pen < 1.0
            hB = float(np.median((ly + pen)[near])) if near.any() else float(ly.max() + 1.0)
            hD = float(np.median(ly + pen))
            hE = float(np.max(ly + np.minimum(pen, 1.0)))
            rec[c] = (float(st.peaks["height"][c]), float(ly.max()), hB, hD, hE)
        return orig(st, c)
    m.feat_prior = fp
    experiment.run_algorithm(m.my_algorithm, "t", sc["seed"], {"config_dir": str(run_suite.materialise(sc)), **sc["lab"]})
    f = Sim(sc); b = f.bounds; lo, sp = b[:, 0], b[:, 1] - b[:, 0]; out = []
    for c, (h, mx, hB, hD, hE) in rec.items():
        cl = sc["clones"][c]; T = float(np.clip(cl["e_activation"], *b[0])); P = float(np.clip(cl["ph_opt"], *b[1]))
        tru = f("MTP", c, [T, P, lo[2] + 1.0 * sp[2], lo[3] + 0.05 * sp[3], 0, 0, 0, 0, T, P, T, P])
        out.append((np.log(max(tru, 1)), h, mx, hB, hD, hE))
    return out

if __name__ == "__main__":
    ids = json.load(open("splits.json"))["hold"][:24]
    with Pool(4) as p: R = sum(p.map(one, ids), [])
    A = np.array(R)
    for k, nm in enumerate(["height", "maxobs", "near-median", "map-median", "max capped"], 1):
        d = A[:, k] - A[:, 0]
        print("%-12s bias %.3f sd %.3f corr %.3f" % (nm, d.mean(), d.std(), np.corrcoef(A[:, k], A[:, 0])[0, 1]))
