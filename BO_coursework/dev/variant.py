"""Make config variants of an algorithm file: python dev/variant.py BASE.py NAME '{"key": value, ...}'

Writes dev/algos/NAME.py = BASE.py + `CFG.update({...})` (JSON parsed, so use lists not tuples).
"""
import json
import sys
from pathlib import Path

base, name, upd = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
src = Path(base).read_text()
out = Path(__file__).resolve().parent / "algos" / f"{name}.py"
out.write_text(src + f"\n\nCFG.update({upd!r})\n")
print(out)
