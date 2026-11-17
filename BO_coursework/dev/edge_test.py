"""Edge-case robustness of final algorithms on the practice lab with extreme settings (analysis only)."""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, "../scenario_suite")
import run_suite
from digital_lab import experiment
CASES = {
    "tiny_budget": {"budget": 4000},
    "one_pilot": {"budget": 3500},
    "few_experiments": {"max_experiments": 45},
    "odd_batches": {"batch_sizes": {"MTP": 8, "MBR": 2, "Pilot": 2}},
    "big_init": {"initial_design": {"MTP": 3, "MBR": 3, "Pilot": 2}},
    "no_init": {"initial_design": {"MTP": 0, "MBR": 0, "Pilot": 0}},
    "cheap_mbr": {"reactor_costs": {"MTP": 5, "MBR": 150, "Pilot": 3000}},
    "pricey_mbr": {"reactor_costs": {"MTP": 20, "MBR": 1500, "Pilot": 2500}},
    "huge_budget": {"budget": 100000, "max_experiments": 600},
}
if __name__ == "__main__":
    for path in sys.argv[1:]:
        algo = run_suite.load_algorithm(path)
        for name, kw in CASES.items():
            try:
                _, info = experiment.run_algorithm(algo, "t", 42, kw)
                print(f"{os.path.basename(path):20s} {name:16s} score {info['score']:9.0f} ckpt {[round(b) if b else None for b in info['checkpoint_bests']]} "
                      f"end {info['end_reason']} err {bool(info['error'])} wall {info['wall_time']:.1f}")
                if info["error"]:
                    print(info["error"][-600:])
            except Exception as e:  # noqa
                print(name, "HARNESS EXC", repr(e)[:300])
