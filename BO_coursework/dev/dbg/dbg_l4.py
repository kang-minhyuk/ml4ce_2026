import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("l4m", str(Path(__file__).resolve().parents[1] / "algos" / "l4_inc.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
orig = m.choose_pilot_race
def wrapped(st):
    sp = st.sp; pp = {}
    for r, y in zip(st.rows, st.ys):
        if r[0] == sp.pilot:
            c = sp.clones.index(r[1]); pp[c] = max(pp.get(c, -1), y)
    est = {c: round(m.lib_potential(st, c)) for c in pp}
    b = orig(st)
    print("SID", st.sp.lab.seed if hasattr(st.sp.lab, "seed") else "", "pp", {c: round(v) for c, v in pp.items()}, "est", est, "-> clone", b[0][1], flush=True)
    return b
m.choose_pilot_race = wrapped
my_algorithm = m.my_algorithm
