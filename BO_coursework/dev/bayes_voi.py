"""Value-of-information plate condition (draft for the algorithm)."""
import numpy as np
from bayes_loc import template_pen


def clone_posterior(t, p, ly, GT, GP, w, sig=0.35, fl=5.0):
    pen = template_pen(t[None, :] - GT[:, None], p[None, :] - GP[:, None], w, fl)
    h = np.median(ly[None, :] + pen, axis=1)
    r = ly[None, :] - (h[:, None] - pen)
    nll = np.sum(np.log1p((r / sig) ** 2), axis=1)
    post = np.exp(-(nll - nll.min()))
    return post / post.sum(), h


def voi_condition(T, P, C, Y, clones, weights, w, sig=0.35, fl=5.0, wscale=1.3, ng=15, na=11, nx=11, support=40,
                  lo=-0.05, hi=1.05):
    """Plate condition (unit t, p) maximising the summed value of information about the operating points of
    `clones` (weighted)."""
    ly = np.log(np.maximum(Y, 1e-4 * max(Y.max(), 1e-12)))
    g = np.linspace(lo, hi, ng)
    GT, GP = [v.ravel() for v in np.meshgrid(g, g, indexing="ij")]
    a = np.linspace(0, 1, na)
    AT, AP = [v.ravel() for v in np.meshgrid(a, a, indexing="ij")]
    x = np.linspace(0, 1, nx)
    XT, XP = [v.ravel() for v in np.meshgrid(x, x, indexing="ij")]
    wp = tuple(v * wscale for v in w)
    U = np.exp(-template_pen(AT[:, None] - GT[None, :], AP[:, None] - GP[None, :], wp, fl))   # A x G
    total = np.zeros(len(XT))
    for c, wt in zip(clones, weights):
        m = C == c
        if not np.any(m):
            post, h = np.full(len(GT), 1.0 / len(GT)), np.zeros(len(GT))
        else:
            post, h = clone_posterior(T[m], P[m], ly[m], GT, GP, w, sig, fl)
        base = np.max(U @ post)
        sup = np.argsort(-post)[:support]
        ps = post[sup] / post[sup].sum()
        # predicted log titer at each candidate x under each hypothesis g: (X, G)
        mx = h[None, :] - template_pen(XT[:, None] - GT[None, :], XP[:, None] - GP[None, :], w, fl)
        for i in range(len(XT)):
            d = (mx[i][None, :] - mx[i][sup][:, None]) / sig                     # support x G
            L = post[None, :] / (1 + d ** 2)
            L /= L.sum(1, keepdims=True)
            V = np.max(L @ U.T, axis=1)                                         # support
            total[i] += wt * (ps @ V - base)
    k = int(np.argmax(total))
    return float(XT[k]), float(XP[k]), total[k]
