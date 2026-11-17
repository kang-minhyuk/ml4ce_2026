"""Bayes-optimal operating T/pH per clone from plate data with the fixed T/pH template (draft for the algorithm)."""
import numpy as np


def template_pen(dt, dp, w, fl):
    d = (dt / np.where(dt < 0, w[0], w[1])) ** 2 + (dp / np.where(dp < 0, w[2], w[3])) ** 2
    return fl * (1 - np.exp(-d / fl))


def bayes_locations(T, P, C, Y, nc, w, sig=0.35, fl=5.0, ng=31, na=21, wscale=1.3, lo=-0.05, hi=1.05):
    """For every clone with plate data: grid posterior over its optimum (uniform prior on the box, height profiled
    out), then the operating point maximising the expected relative titer under a Pilot-scale template (widths x
    wscale). Returns dict with t, p (unit), map t/p and the posterior spread."""
    ly = np.log(np.maximum(Y, 1e-4 * max(Y.max(), 1e-12)))
    g = np.linspace(lo, hi, ng)
    GT, GP = np.meshgrid(g, g, indexing="ij"); GT, GP = GT.ravel(), GP.ravel()
    a = np.linspace(0.0, 1.0, na)
    AT, AP = np.meshgrid(a, a, indexing="ij"); AT, AP = AT.ravel(), AP.ravel()
    wp = tuple(x * wscale for x in w)
    # utility of action (rows) given optimum (cols): relative Pilot titer
    U = np.exp(-template_pen(AT[:, None] - GT[None, :], AP[:, None] - GP[None, :], wp, fl))
    t_out = np.full(nc, 0.5); p_out = np.full(nc, 0.5); spread = np.full(nc, 1.0)
    tm = np.full(nc, 0.5); pm = np.full(nc, 0.5)
    for c in np.unique(C):
        m = C == c
        pen = template_pen(T[m][None, :] - GT[:, None], P[m][None, :] - GP[:, None], w, fl)  # grid x wells
        h = np.median(ly[m][None, :] + pen, axis=1)                                         # profiled height
        r = ly[m][None, :] - (h[:, None] - pen)
        nll = np.sum(np.log1p((r / sig) ** 2), axis=1)                                      # Cauchy-robust
        post = np.exp(-(nll - nll.min())); post /= post.sum()
        eu = U @ post
        k = int(np.argmax(eu))
        t_out[c], p_out[c] = AT[k], AP[k]
        j = int(np.argmax(post)); tm[c], pm[c] = GT[j], GP[j]
        spread[c] = np.sqrt(post @ ((GT - post @ GT) ** 2 + (GP - post @ GP) ** 2))
    return {"t": t_out, "p": p_out, "t_map": tm, "p_map": pm, "spread": spread}
