"""Profile one algorithm run on a suite scenario: python profile_one.py ALGO.py SID"""
import json, os, sys, time, cProfile, pstats
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, "../scenario_suite"); sys.path.insert(0, ".")
import run_suite
import digital_lab.lab as L
from digital_lab import experiment
if __name__ == "__main__":
    path, sid = sys.argv[1], int(sys.argv[2])
    sc = json.loads((run_suite.SUITE / "scenarios" / f"scenario_{sid:04d}.json").read_text())
    names = [c["name"] for c in sc["clones"]]
    L.CELLTYPES = names
    L.LAB_CONFIG_FILENAMES = {**L.LAB_CONFIG_FILENAMES, **{n: f"{n}.json" for n in names}}
    L.CONTINUOUS_BOUNDS = [tuple(b) for b in sc["bounds"]]
    algo = run_suite.load_algorithm(path)
    pr = cProfile.Profile(); pr.enable()
    _, info = experiment.run_algorithm(algo, "t", sc["seed"], {"config_dir": str(run_suite.materialise(sc)), **sc["lab"]})
    pr.disable()
    print("score", round(info["score"]), info["checkpoint_bests"], "wall", round(info["wall_time"], 1), info["end_reason"])
    pstats.Stats(pr).sort_stats("cumulative").print_stats(25)
