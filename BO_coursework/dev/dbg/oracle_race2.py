"""ANALYSIS ONLY (cheats): racing AND incumbent choice use the TRUE Pilot potential of the clones."""
import importlib.util, json, glob
from pathlib import Path
D = Path(__file__).resolve().parents[1]
src = (D / "final" / "informed_bayes.py").read_text()
src = src.replace("    inc = max(pp, key=pp.get)", "    inc = max(pp, key=lambda c: CUR['pot'][c]) if CUR.get('pot') else max(pp, key=pp.get)")
ns = {"__name__": "ibm2"}
exec(compile(src, "ibm2", "exec"), ns)
V = json.loads((D / "clone_value.json").read_text())
SEED2SID = {}
for f in glob.glob(str(D.parent / "scenario_suite" / "scenarios" / "scenario_*.json")):
    sc = json.loads(Path(f).read_text()); SEED2SID[sc["seed"]] = sc["id"]
CUR = ns["CUR"] = {}
orig_pt = ns["prior_titers"]
def prior_titers(st, reactor):
    out = orig_pt(st, reactor)
    if reactor == st.sp.mbr and CUR.get("pot") is not None:
        return {c: CUR["pot"][c] for c in out}
    return out
ns["prior_titers"] = prior_titers
def my_algorithm(initial_x, initial_y, lab, seed):
    sid = SEED2SID.get(seed)
    CUR["pot"] = V.get(str(sid)) if sid is not None else None
    return ns["my_algorithm"](initial_x, initial_y, lab, seed)
