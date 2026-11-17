"""Fold a dev algorithm's CFG.update chain into one CFG block, set a docstring, add the time guard.
python consolidate.py SRC.py OUT.py DOCFILE"""
import importlib.util, sys
from pathlib import Path


def load(path):
    spec = importlib.util.spec_from_file_location("consolidate_src", path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


src, out, docfile = sys.argv[1:4]
cfg = dict(load(src).CFG)
cfg = {"time_guard": 20.0, **cfg}
lines = Path(src).read_text().splitlines()
lines = [l for l in lines if not l.startswith("CFG.update(")]
s = "\n".join(lines).rstrip() + "\n"
a = s.index("CFG = {"); b = s.index("\n}\n", a) + 3
block = ["CFG = {"] + [f"    {k!r}: {v!r}," for k, v in cfg.items()] + ["}\n"]
s = s[:a] + "\n".join(block) + s[b:]
d0 = s.index('"""'); d1 = s.index('"""', d0 + 3) + 3
s = s[:d0] + Path(docfile).read_text().strip() + s[d1:]
old = "        st.update_peaks()\n        st.fit_gp()\n"
new = ("        if lab.time_remaining > cfg[\"time_guard\"]:  # little time left: keep the last fits\n"
       "            st.update_peaks()\n            st.fit_gp()\n")
assert s.count(old) == 1
s = s.replace(old, new)
while "\n\n\n\n" in s:
    s = s.replace("\n\n\n\n", "\n\n\n")
Path(out).write_text(s)
assert dict(load(out).CFG) == cfg, "config mismatch"
print(out, len(s.splitlines()), "lines, config verified")
