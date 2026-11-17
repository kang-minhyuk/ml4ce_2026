"""Offline: plate signatures (true T/pH, grid of plate media) for the clones of the glutamine library (gln_lib.npz),
to test whether cheap plates predict a clone's Pilot glutamine optimum. -> lib_signature.npz (analysis only)"""
import sys
import numpy as np
from multiprocessing import Pool
from sim import Sim, load_scenario

GLC = [0.1, 0.3, 1.0]
GLN = [0.005, 0.02, 0.05, 0.15, 0.4, 1.0]


def run(args):
    sid, cis = args
    sc = load_scenario(sid); f = Sim(sc); b = f.bounds; lo, sp = b[:, 0], b[:, 1] - b[:, 0]
    out = []
    for ci in cis:
        c = sc["clones"][ci]
        T = float(np.clip(c["e_activation"], *b[0])); P = float(np.clip(c["ph_opt"], *b[1]))
        out.append([f("MTP", ci, [T, P, lo[2] + g * sp[2], lo[3] + a * sp[3], 0, 0, 0, 0, T, P, T, P])
                    for g in GLC for a in GLN])
    return out


if __name__ == "__main__":
    d = np.load("gln_lib.npz"); sid = d["sid"]; ci = d["ci"]
    jobs, idx = [], []
    for s in np.unique(sid):
        m = np.where(sid == s)[0]
        jobs.append((int(s), [int(ci[i]) for i in m])); idx.append(m)
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as p:
        res = p.map(run, jobs, chunksize=2)
    S = np.zeros((len(sid), len(GLC) * len(GLN)))
    for m, r in zip(idx, res):
        S[m] = r
    np.savez_compressed("lib_signature.npz", S=S, GLC=GLC, GLN=GLN)
    print(S.shape)
