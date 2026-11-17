"""ANALYSIS ONLY (cheats): racing uses the TRUE Pilot potential of screened clones instead of their MBR titers."""
import importlib.util, json, glob
from pathlib import Path
D = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ibm", str(D / "final" / "informed_bayes.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
V = json.loads((D / "clone_value.json").read_text())
SEED2SID = {}
for f in glob.glob(str(D.parent / "scenario_suite" / "scenarios" / "scenario_*.json")):
    sc = json.loads(Path(f).read_text()); SEED2SID[sc["seed"]] = sc["id"]
orig_pt = m.prior_titers
CUR = {}
def prior_titers(st, reactor):
    out = orig_pt(st, reactor)
    if reactor == st.sp.mbr and CUR.get("pot") is not None:
        return {c: CUR["pot"][c] for c in out}
    return out
m.prior_titers = prior_titers
def my_algorithm(initial_x, initial_y, lab, seed):
    sid = SEED2SID.get(seed)
    CUR["pot"] = V.get(str(sid)) if sid is not None else None
    return m.my_algorithm(initial_x, initial_y, lab, seed)
