"""Make a variant file: python mkvar.py BASE.py OUT.py 'key=value' ... (values are Python literals)."""
import sys
base, out, *kv = sys.argv[1:]
src = open(base).read().rstrip() + "\n"
upd = ", ".join(f'"{k}": {v}' for k, v in (s.split("=", 1) for s in kv))
open(out, "w").write(src + "\nCFG.update({" + upd + "})\n")
print(out, "->", upd)
