"""Offline test: library-posterior BO over the Pilot glutamine grid (oracle clone, true T/pH).
Library = response tables of other clones; test clones = the other split. Analysis only."""
import numpy as np, itertools, sys
D = np.load("gln_grid_dev.npy"); F = np.load("gln_grid_off.npy")
A0 = [.0005, .001, .002, .004, .008, .016, .03, .06]; A1 = [.002, .005, .01, .02, .04, .08, .15]; B2 = [.25, .4, .6, .85]
GRID = list(itertools.product(A0, A1, B2))
def lg(x): return np.log(np.maximum(x, 1e-3))
def greedy(L, k):
    ch = []
    for _ in range(k):
        ch.append(max(range(L.shape[1]), key=lambda j: np.max(L[:, ch + [j]], 1).mean()))
    return ch
def lib_policy(lib, test, n, sig=0.3, acq="ei", first=None, temp=1.0):
    LL = lg(lib); out = np.zeros((len(test), n))
    for i, t in enumerate(test):
        obs = []; yt = lg(t)
        for s in range(n):
            if s == 0 and first is not None:
                j = first
            else:
                if obs:
                    J = np.array([o for o in obs]); y = yt[J]
                    R = y[None, :] - LL[:, J]; c = R.mean(1)
                    ssr = ((R - c[:, None]) ** 2).sum(1)
                    lw = -ssr / (2 * sig ** 2) / temp; w = np.exp(lw - lw.max()); w /= w.sum()
                    pred = LL + c[:, None]; best = y.max()
                else:
                    w = np.ones(len(LL)) / len(LL); pred = LL; best = -np.inf
                if acq == "ei":
                    # improvement in titer space (exp), relative to best
                    imp = np.maximum(np.exp(pred) - (np.exp(best) if obs else 0), 0)
                    a = w @ imp
                elif acq == "pmax":
                    a = w @ (pred >= pred.max(1, keepdims=True) - 1e-9)
                elif acq == "mean":
                    a = w @ np.exp(pred)
                a[obs] = -np.inf; j = int(np.argmax(a))
            obs.append(j)
            out[i, s] = t[obs].max()
    return out
if __name__ == "__main__":
    n = 8
    for name, lib, test in [("dev->off", D, F), ("off->dev", F, D)]:
        g = greedy(lib, n); fixed = np.maximum.accumulate(test[:, g], 1).mean(0)
        print(name, "gridbest %.3f" % test.max(1).mean())
        print("  fixed greedy      ", " ".join("%.3f" % v for v in fixed))
        for sig in [0.1, 0.2, 0.4, 0.8]:
            for acq in ["ei", "pmax"]:
                r = lib_policy(lib, test, n, sig, acq).mean(0)
                print("  lib %-4s sig %.1f  " % (acq, sig), " ".join("%.3f" % v for v in r))
