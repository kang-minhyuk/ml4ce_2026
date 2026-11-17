"""Mixture-library policy with a compressed library (PCA rank r, N curves), parallel offline test (analysis only)."""
import numpy as np, sys
from multiprocessing import Pool
Z = np.load("gln_lib.npz"); Y = Z["Y"]; sid = Z["sid"]
LY = np.log(np.maximum(Y, 1e-3 * Y.max(1, keepdims=True)))
grp = np.where(sid >= 10380, 1, 0)
def compress(L, r, N, seed=0):
    rng = np.random.default_rng(seed)
    if N < len(L): L = L[rng.choice(len(L), N, replace=False)]
    if r is None: return L
    off = L.mean(1, keepdims=True); C = L - off; mu = C.mean(0); U, S, Vt = np.linalg.svd(C - mu, full_matrices=False)
    return mu + (U[:, :r] * S[:r]) @ Vt[:r] + off
def policy(args):
    L, t, n, sig, first = args
    J = [] if first is None else [first]; out = []
    if J: out.append(np.exp(t[J].max() - t.max()))
    while len(out) < n:
        if not J: a = np.exp(L - L.max(1, keepdims=True)).mean(0)
        else:
            y = t[J]; R = y[None] - L[:, J]; c = R.mean(1); ssr = ((R - c[:, None]) ** 2).sum(1)
            lw = -ssr / (2 * sig ** 2); w = np.exp(lw - lw.max()); w /= w.sum()
            a = w @ np.maximum(np.exp(L + c[:, None]) - np.exp(y.max()), 0)
        a[J] = -np.inf; J.append(int(np.argmax(a))); out.append(np.exp(t[J].max() - t.max()))
    return out
def evaluate(L, T, sig=0.1, n=6, first=None, pool=None):
    return np.array(pool.map(policy, [(L, t, n, sig, first) for t in T], chunksize=8)).mean(0)
if __name__ == "__main__":
    rng = np.random.default_rng(5)
    with Pool(20) as pool:
        for tr, te in [(0, 1), (1, 0)]:
            Lf = LY[grp == tr]; T = LY[grp == te]; T = T[rng.choice(len(T), 300, replace=False)]
            print("train %d -> test %d" % (tr, te), flush=True)
            for r, N, sig in [(None, 9999, .1), (None, 300, .1), (None, 150, .1), (20, 9999, .1), (10, 9999, .1),
                              (5, 9999, .1), (10, 9999, .15), (10, 300, .1), (None, 9999, .05)]:
                L = compress(Lf, r, N)
                print("  r %-4s N %-5d sig %.2f" % (r, min(N, len(Lf)), sig), " ".join("%.3f" % v for v in evaluate(L, T, sig, pool=pool)), flush=True)
