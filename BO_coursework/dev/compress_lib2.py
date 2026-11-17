import numpy as np, os
Z = np.load("gln_lib.npz"); Y = Z["Y"]
L = np.log(np.maximum(Y, 1e-3 * Y.max(1, keepdims=True))); L = L - L.max(1, keepdims=True)
A0, A1, B2 = Z["A0"], Z["A1"], Z["B2"]
def save(name, Lc, a0, a1, b2):
    d = f"libvar/{name}"; os.makedirs(d, exist_ok=True)
    np.savez_compressed(f"{d}/gln_lib.npz", Y=np.exp(Lc), A0=a0, A1=a1, B2=b2)
def pca(Lx, r, N, seed=0):
    mu = Lx.mean(0); U, S, Vt = np.linalg.svd(Lx - mu, full_matrices=False)
    idx = np.sort(np.random.default_rng(seed).choice(len(Lx), N, replace=False))
    return (mu + (U[:, :r] * S[:r]) @ Vt[:r])[idx]
save("pca12_n300", pca(L, 12, 300), A0, A1, B2)
i0, i1, i2 = np.arange(0, 13, 2), np.arange(0, 12, 2), np.arange(0, 7, 2)
Ls = L.reshape(-1, 13, 12, 7)[:, i0][:, :, i1][:, :, :, i2].reshape(len(L), -1)
save("sub", Ls, A0[i0], A1[i1], B2[i2])
save("sub_pca12_n300", pca(Ls, 12, 300), A0[i0], A1[i1], B2[i2])
i0b, i1b = np.arange(0, 13), np.arange(0, 12)
Lb = L.reshape(-1, 13, 12, 7)[:, :, :, i2].reshape(len(L), -1)
save("b4_pca12_n300", pca(Lb, 12, 300), A0, A1, B2[i2])
print("sizes: full", L.shape[1], "sub", Ls.shape[1], "b4", Lb.shape[1])
