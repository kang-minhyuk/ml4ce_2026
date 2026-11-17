"""Offline comparison of library-based Pilot glutamine policies on the absolute-grid library (analysis only).
Cross-validated by scenario group: train on suite+off_extra curves, test on new official curves, and reverse."""
import numpy as np, sys
from scipy.stats import norm
Z = np.load("gln_lib.npz"); Y = Z["Y"]; sid = Z["sid"]
LY = np.log(np.maximum(Y, 1e-3 * Y.max(1, keepdims=True)))
grp = np.where(sid >= 10380, 1, 0)
def lognorm_ei(m, s, best):
    # E[max(exp(Y) - exp(best), 0)], Y ~ N(m, s^2)
    s = np.maximum(s, 1e-9); b = best
    d = (m + s * s - b) / s
    return np.exp(m + s * s / 2) * norm.cdf(d) - np.exp(b) * norm.cdf(d - s)
class LibMix:
    def __init__(self, L, sig): self.L = L; self.sig = sig
    def acq(self, J, y):
        L = self.L
        if not J: return np.exp(L - L.max(1, keepdims=True)).mean(0)
        R = y[None] - L[:, J]; c = R.mean(1); ssr = ((R - c[:, None]) ** 2).sum(1)
        lw = -ssr / (2 * self.sig ** 2); w = np.exp(lw - lw.max()); w /= w.sum()
        return w @ np.maximum(np.exp(L + c[:, None]) - np.exp(y.max()), 0)
class LibGauss:
    def __init__(self, L, rank=30, nug=0.02, c_var=25.0):
        C = L - L.mean(1, keepdims=True); self.mu = C.mean(0); D = C - self.mu
        U, S, Vt = np.linalg.svd(D, full_matrices=False); k = rank
        self.V = Vt[:k].T * (S[:k] / np.sqrt(len(L)))  # Sigma ~ V V^T
        self.nug = nug; self.cv = c_var; self.mu0 = L.mean()
    def post(self, J, y):
        V = self.V; n = len(J)
        if n == 0: return self.mu + self.mu0, np.sqrt((V ** 2).sum(1) + self.cv)
        K = V[J] @ V[J].T + self.cv + self.nug * np.eye(n)
        Ks = V @ V[J].T + self.cv
        A = np.linalg.solve(K, np.c_[y - self.mu[J] - self.mu0, Ks.T])
        m = self.mu + self.mu0 + Ks @ A[:, 0]
        v = (V ** 2).sum(1) + self.cv - np.einsum("ij,ji->i", Ks, A[:, 1:])
        return m, np.sqrt(np.maximum(v, 1e-9))
    def acq(self, J, y):
        m, s = self.post(J, y)
        if not J: return m
        return lognorm_ei(m, s, y.max())
def run(model, T, n=8):
    out = np.zeros((len(T), n))
    for i, t in enumerate(T):
        J = []
        for k in range(n):
            a = model.acq(J, t[J]).copy(); a[J] = -np.inf; J.append(int(np.argmax(a)))
            out[i, k] = np.exp(t[J].max() - t.max())
    return out.mean(0)
def greedy(L, T, n=8):
    E = np.exp(L - L.max(1, keepdims=True)); ch = []
    for _ in range(n): ch.append(max(range(L.shape[1]), key=lambda j: E[:, ch + [j]].max(1).mean()))
    return np.maximum.accumulate(np.exp(T[:, ch] - T.max(1, keepdims=True)), 1).mean(0)
if __name__ == "__main__":
    for tr, te in [(0, 1), (1, 0)]:
        L, T = LY[grp == tr], LY[grp == te]
        print("train %d (%d) -> test %d (%d)" % (tr, len(L), te, len(T)))
        print("  greedy fixed     ", " ".join("%.3f" % v for v in greedy(L, T)))
        for sig in [0.1, 0.2, 0.3]:
            print("  mix sig %.1f      " % sig, " ".join("%.3f" % v for v in run(LibMix(L, sig), T)))
        for rank in [10, 30, 80]:
            for nug in [0.01, 0.05]:
                print("  gauss r%-3d nug %.2f" % (rank, nug), " ".join("%.3f" % v for v in run(LibGauss(L, rank, nug), T)))
