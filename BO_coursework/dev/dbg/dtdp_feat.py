"""Offline: does conditioning the library on the T/pH offset of the operating point (member dT, dp) improve the
first-Pilot recipe beyond the plate feature? Leave-one-scenario-out (analysis only)."""
import numpy as np
d = np.load("gln_lib.npz"); s = np.load("lib_signature.npz")
Y = d["Y"]; sid = d["sid"]; gmax = d["gmax"]; A0 = d["A0"]; dT = d["dT"]; dp = d["dp"]
F = np.log(np.maximum(s["S"][:, 14], 1e-9)); n = len(Y); a0 = np.repeat(A0, 84)
R = Y / Y.max(1, keepdims=True)
rng = np.random.default_rng(0); Fn = F + 0.5 * rng.standard_normal(n)
def ev(sT, sp, noise_off=0.0):
    out = []
    dTn = dT + noise_off * rng.standard_normal(n); dpn = dp + noise_off * 0.2 * rng.standard_normal(n)
    for j in range(n):
        m = sid != sid[j]
        lw = -0.5 * ((F[m] - Fn[j]) / 0.5) ** 2
        if sT: lw = lw - 0.5 * ((dT[m] - dTn[j]) / sT) ** 2 - 0.5 * ((dp[m] - dpn[j]) / sp) ** 2
        w = np.exp(lw - lw.max()); sc = (w / w.sum()) @ R[m]; sc[a0 > gmax[j] * 1.0001] = -1
        out.append(R[j, sc.argmax()])
    out = np.array(out); off = (dT != 0)
    return out.mean(), out[off].mean(), out[~off].mean()
print("F only            all %.4f  offset members %.4f  at-optimum members %.4f" % ev(0, 0))
for sT, sp in ((0.5, 0.1), (1.0, 0.2), (2.0, 0.4)):
    print("F + dT/dp s(%.1f,%.1f) all %.4f  offset %.4f  at-opt %.4f" % ((sT, sp) + ev(sT, sp)))
print("F + dT/dp s(1,.2), offset known +-0.7C: all %.4f offset %.4f at-opt %.4f" % ev(1.0, 0.2, 0.7))
