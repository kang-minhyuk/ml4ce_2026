"""Textbook fed-batch model v2 (generic): v1 + spontaneous glutamine degradation to ammonia, overflow glutamine uptake
(saturating in glutamine, not tied to growth) and ammonia-inhibited specific productivity. Analysis only."""
import numpy as np

NAMES = ["mu", "Kgln", "Yxgln", "kd0", "KIamm", "Yamm", "ba", "kdamm", "qov", "Yxglc", "kdeg", "KIp", "Kov"]
LO = np.log([0.005, 0.005, 0.02, 1e-4, 0.2, 0.05, 1e-3, 1e-4, 1e-4, 0.005, 1e-5, 0.2, 0.05])
HI = np.log([0.10, 5.0, 5.0, 0.05, 60.0, 1.5, 0.5, 0.02, 0.2, 1.0, 0.01, 60.0, 50.0])
X0 = 0.3


def simulate(th, glc0, gln0, glcf1, glnf1, glcf2, glnf2, dt=0.5, T_end=336.0):
    mu, Kg, Yxg, kd0, KIa, Ya, ba, kda, qov, Yxc, kdeg, KIp, Kov = np.exp(th)
    n = np.broadcast(glc0, gln0, glnf1, glnf2).shape
    X = np.full(n, X0); C = np.broadcast_to(np.asarray(glc0, float), n).copy()
    Q = np.broadcast_to(np.asarray(gln0, float), n).copy(); A = np.zeros(n); P = np.zeros(n)
    steps = int(round(T_end / dt)); f1, f2 = int(round(120 / dt)), int(round(240 / dt))
    for k in range(steps):
        if k == f1: C = C + glcf1; Q = Q + glnf1
        if k == f2: C = C + glcf2; Q = Q + glnf2
        Qp = np.maximum(Q, 0.0); Cp = np.maximum(C, 0.0)
        g = mu * Qp / (Kg + Qp) * Cp / (0.5 + Cp) * KIa / (KIa + A)
        d = kd0 + kda * A
        up_q = (g / Yxg + qov * Qp / (Kov + Qp)) * X
        deg = kdeg * Qp
        P = P + (g + ba * mu) * KIp / (KIp + A) * X * dt
        A = A + (Ya * up_q + deg) * dt
        Q = Q - (up_q + deg) * dt; C = C - g / Yxc * X * dt
        X = X * np.exp((g - d) * dt)
    return P
