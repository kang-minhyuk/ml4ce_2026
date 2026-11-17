import json, sys, numpy as np
from multiprocessing import Pool
from scipy.stats import spearmanr
from sim import Sim, load_scenario
O=json.load(open("oracle.json")); dev=json.load(open("splits.json"))["dev"]
def run(sid):
    sc=load_scenario(sid); f=Sim(sc); b=np.array(sc["bounds"]); lo,hi=b[:,0],b[:,1]
    res=[]
    for ci,c in enumerate(sc["clones"]):
        To=np.clip(c["e_activation"],lo[0],hi[0]); po=np.clip(c["ph_opt"],lo[1],hi[1])
        u=np.array([0,0,1,0.03,1,0.03,1,0.4,0,0,0,0.]); x=lo+u*(hi-lo); x[[0,8,10]]=To; x[[1,9,11]]=po
        p=f("Pilot",ci,x); m=f("MBR",ci,x)
        xm=x.copy(); xm[4:8]=0; mt=f("MTP",ci,xm)
        res.append((p,m,mt))
    R=np.array(res); o=O[str(sid)]
    return sid, R[:,0].max()/o["oracle"], R[o["clone"],0]/o["oracle"], spearmanr(R[:,0],R[:,1])[0], spearmanr(R[:,0],R[:,2])[0], int(np.argmax(R[:,1])==o["clone"]), int(np.argmax(R[:,0])==o["clone"]), R[np.argmax(R[:,1]),0]/o["oracle"], R[np.argmax(R[:,2]),0]/o["oracle"]
if __name__ == "__main__":
  with Pool(21) as p: out=p.map(run, dev)
  A=np.array([o[1:] for o in out])
  print("median/mean: best-heuristic/oracle %.3f %.3f | oracle-clone heur/oracle %.3f %.3f"%(np.median(A[:,0]),A[:,0].mean(),np.median(A[:,1]),A[:,1].mean()))
  print("spearman pilot vs mbr (heur) %.2f, pilot vs mtp %.2f"%(np.median(A[:,2]),np.median(A[:,3])))
  print("argmax MBR-heur is oracle clone %.2f ; argmax pilot-heur is oracle clone %.2f"%(A[:,4].mean(),A[:,5].mean()))
  print("value if clone picked by MBR-heur: %.3f ; by MTP-heur: %.3f"%(A[:,6].mean(),A[:,7].mean()))
