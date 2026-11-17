"""Textbook mammalian fed-batch kinetic model (generic; written from standard bioprocess modelling, not from the lab
simulator): Monod growth on glucose and glutamine with lactate/ammonia inhibition, ammonia-dependent death,
Luedeking-Piret product formation, bolus feeds on day 5 and day 10, 14-day run. Vectorised over recipes.
Analysis only (offline feasibility test of a physics-based recipe model)."""
import numpy as np

# theta (all log-transformed): mu_max, K_gln, Y_x/gln, kd0, K_I,amm, Y_amm/gln, beta/alpha, kd_amm, m_gln, Y_x/glc
NAMES = ["mu", "Kgln", "Yxgln", "kd0", "KIamm", "Yamm", "ba", "kdamm", "mgln", "Yxglc"]
LO = np.log([0.005, 0.005, 0.02, 1e-4, 0.2, 0.05, 1e-3, 1e-4, 1e-5, 0.005])
HI = np.log([0.10, 5.0, 5.0, 0.05, 60.0, 1.5, 0.5, 0.02, 0.05, 1.0])
X0 = 0.3


def simulate(th, glc0, gln0, glcf1, glnf1, glcf2, glnf2, dt=1.0, T_end=336.0):
    """Final product (arbitrary units) for arrays of recipes (absolute concentrations)."""
    mu, Kg, Yxg, kd0, KIa, Ya, ba, kda, mg, Yxc = np.exp(th)
    n = np.broadcast(glc0, gln0, glnf1, glnf2).shape
    X = np.full(n, X0); C = np.broadcast_to(np.asarray(glc0, float), n).copy()
    Q = np.broadcast_to(np.asarray(gln0, float), n).copy(); A = np.zeros(n); P = np.zeros(n)
    t = 0.0
    while t < T_end - 1e-9:
        if abs(t - 120.0) < 1e-9: C = C + glcf1; Q = Q + glnf1
        if abs(t - 240.0) < 1e-9: C = C + glcf2; Q = Q + glnf2
        Qp = np.maximum(Q, 0.0); Cp = np.maximum(C, 0.0)
        g = mu * Qp / (Kg + Qp) * Cp / (0.5 + Cp) * KIa / (KIa + A)
        d = kd0 + kda * A
        up_q = (g / Yxg + mg * Qp / (Kg + Qp)) * X
        up_c = g / Yxc * X
        P = P + (g + ba * mu) * X * dt
        A = A + Ya * up_q * dt
        Q = Q - up_q * dt; C = C - up_c * dt
        X = X * np.exp((g - d) * dt)
        t += dt
    return P
