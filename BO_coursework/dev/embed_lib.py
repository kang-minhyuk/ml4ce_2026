"""Embed the compressed Pilot glutamine library into an algorithm file (no file reading at run time).

python dev/embed_lib.py BASE.py OUT.py [r N seed]
Library: dev/gln_lib.npz (1950 simulated clone curves, analysis data) -> 7x6x4 absolute log-grid, PCA rank r,
N random curves (coefficients), rounded. Replaces load_dev_library() in BASE.py.
"""
import sys, re, numpy as np
from pathlib import Path
base, out = sys.argv[1], sys.argv[2]
r = int(sys.argv[3]) if len(sys.argv) > 3 else 12
N = int(sys.argv[4]) if len(sys.argv) > 4 else 300
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
D = Path(__file__).resolve().parent
Z = np.load(D / "gln_lib.npz"); Y = Z["Y"]
L = np.log(np.maximum(Y, 1e-3 * Y.max(1, keepdims=True))); L = L - L.max(1, keepdims=True)
i0, i1, i2 = np.arange(0, 13, 2), np.arange(0, 12, 2), np.arange(0, 7, 2)
A0, A1, B2 = Z["A0"][i0], Z["A1"][i1], Z["B2"][i2]
Ls = L.reshape(-1, 13, 12, 7)[:, i0][:, :, i1][:, :, :, i2].reshape(len(L), -1)
mu = Ls.mean(0); U, S, Vt = np.linalg.svd(Ls - mu, full_matrices=False)
idx = np.sort(np.random.default_rng(seed).choice(len(Ls), N, replace=False))
coef = np.round(U[idx, :r] * S[:r], 3); pcs = np.round(Vt[:r], 4); mu = np.round(mu, 4)
rec = mu + coef @ pcs
print("max abs rounding error vs exact PCA:", np.abs(rec - (Ls.mean(0) + (U[idx, :r] * S[:r]) @ Vt[:r])).max())
def fmt(a, nd):
    return "[" + ", ".join(f"{v:.{nd}f}".rstrip("0").rstrip(".") if v != 0 else "0" for v in a) + "]"
def rows(M, nd, per=None):
    return "[\n" + "".join(f"    {fmt(m, nd)},\n" for m in M) + "]"
tables = f'''

# ------------------------------------------------------------------------------------------------ library tables
# Pilot glutamine-response library, pre-computed offline with the dbiolab simulator (the same model family the lab
# uses) for 1950 simulated clones drawn from the documented parameter ranges (no clone of any lab, no lab files):
# log(titer / clone's own best) on an absolute grid of glutamine start x feed 1 x feed 2 (mM), glucose at its upper
# bound, temperature/pH near the clone's optimum (with realistic location errors). Stored compressed: mean surface +
# {r} principal components; GLN_COEF holds the component scores of {N} library members.
GLN_A0 = {fmt(A0, 4)}
GLN_A1 = {fmt(A1, 4)}
GLN_B2 = {fmt(B2, 2)}
GLN_MEAN = {fmt(mu, 4)}
GLN_PCS = {rows(pcs, 4)}
GLN_COEF = {rows(coef, 3)}
'''
s = Path(base).read_text()
m = re.search(r"\ndef load_dev_library\(\):\n(?:    .*\n|\n)*?(?=\n\nLIB = None)", s)
assert m, "load_dev_library not found"
new_loader = '''
def load_dev_library():
    """Rebuild the library (member x grid log efficiency) from the embedded compressed tables."""
    L = np.asarray(GLN_MEAN)[None] + np.asarray(GLN_COEF) @ np.asarray(GLN_PCS)
    return GlnLib(L, GLN_A0, GLN_A1, GLN_B2)
'''
s = s[:m.start()] + new_loader + s[m.end():]
anchor = "\n\n# ------------------------------------------------------------------------------------------------ main"
assert anchor in s
s = s.replace(anchor, tables + anchor, 1)
s = s.replace("def load_dev_library():", "def load_library():").replace("LIB = load_dev_library()", "LIB = load_library()")
Path(out).write_text(s)
print("wrote", out, len(s) // 1024, "KB")
