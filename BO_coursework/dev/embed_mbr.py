"""Append the compressed MBR glutamine library (same 300 members and grid as GLN_*) to an algorithm file.
python dev/embed_mbr.py BASE.py OUT.py [r]
MBR surface of each member as log(MBR titer / the member's best PILOT titer), so a clone's MBR and Pilot titers share
one offset (analysis data dev/gln_lib_mbr.npz; no file reading at run time)."""
import sys, numpy as np
from pathlib import Path
base, out = sys.argv[1], sys.argv[2]
r = int(sys.argv[3]) if len(sys.argv) > 3 else 12
D = Path(__file__).resolve().parent
P = np.load(D / "gln_lib.npz"); M = np.load(D / "gln_lib_mbr.npz")
assert (P["sid"] == M["sid"]).all() and (P["ci"] == M["ci"]).all()
pm = P["Y"].max(1, keepdims=True)
L = np.log(np.maximum(M["Y"], 1e-3 * pm) / pm)
i0, i1, i2 = np.arange(0, 13, 2), np.arange(0, 12, 2), np.arange(0, 7, 2)
Ls = L.reshape(-1, 13, 12, 7)[:, i0][:, :, i1][:, :, :, i2].reshape(len(L), -1)
mu = Ls.mean(0); U, S, Vt = np.linalg.svd(Ls - mu, full_matrices=False)
idx = np.sort(np.random.default_rng(0).choice(len(Ls), 300, replace=False))
coef = np.round(U[idx, :r] * S[:r], 3); pcs = np.round(Vt[:r], 4); mu = np.round(mu, 4)
rec = mu + coef @ pcs
print("PCA rank", r, "rms error vs exact members %.4f" % np.sqrt(((rec - Ls[idx]) ** 2).mean()))
def fmt(a, nd):
    return "[" + ", ".join(f"{v:.{nd}f}".rstrip("0").rstrip(".") if v != 0 else "0" for v in a) + "]"
tables = f'''

# MBR glutamine-response surfaces of the same library members (same grid), as log(MBR titer / member's best Pilot
# titer): a clone's MBR screen and its Pilot runs then share one offset. Compressed: mean + {r} components.
MBR_MEAN = {fmt(mu, 4)}
MBR_PCS = [
''' + "".join(f"    {fmt(m, 4)},\n" for m in pcs) + "]\nMBR_COEF = [\n" + "".join(f"    {fmt(m, 3)},\n" for m in coef) + "]\n"
s = Path(base).read_text()
anchor = "\nGLN_PLATE = "
assert anchor in s
s = s.replace(anchor, tables + anchor, 1)
Path(out).write_text(s)
print("wrote", out, len(s) // 1024, "KB")
