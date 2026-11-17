"""Compressed variants of the Pilot glutamine library for in-pipeline tests (analysis only)."""
import numpy as np, os
Z = np.load("gln_lib.npz"); Y = Z["Y"]
L = np.log(np.maximum(Y, 1e-3 * Y.max(1, keepdims=True)))
L = L - L.max(1, keepdims=True)  # each curve relative to its own optimum
def save(name, Lc, idx=None):
    d = f"libvar/{name}"; os.makedirs(d, exist_ok=True)
    np.savez_compressed(f"{d}/gln_lib.npz", Y=np.exp(Lc), sid=Z["sid"] if idx is None else Z["sid"][idx], A0=Z["A0"], A1=Z["A1"], B2=Z["B2"])
rng = np.random.default_rng(0)
for N in (100, 200, 400):
    idx = np.sort(rng.choice(len(L), N, replace=False)); save(f"n{N}", L[idx], idx)
mu = L.mean(0); U, S, Vt = np.linalg.svd(L - mu, full_matrices=False)
for r in (6, 10, 16):
    save(f"pca{r}", mu + (U[:, :r] * S[:r]) @ Vt[:r])
# PCA r on a random subset of curves (stored coefficients only for N curves)
for r, N in ((10, 300), (16, 300)):
    idx = np.sort(rng.choice(len(L), N, replace=False)); save(f"pca{r}_n{N}", (mu + (U[:, :r] * S[:r]) @ Vt[:r])[idx], idx)
print("ok", (S ** 2).cumsum()[:16] / (S ** 2).sum())
