import sys, numpy as np
sys.path.insert(0, "src")
from analyze_run2 import load
from analyze_addB import clusters_of
from conformal import split_conformal_quantile
ALPHA, DELTA = 0.1, 0.1

def wsr_trace(L, alpha):
    mu, var, n = alpha/2, alpha/4, 1.0
    k, kmax, kmin, neg = 1.0, 1.0, 1.0, False
    for x in L:
        edge = alpha - mu
        lam = 0.0 if edge <= 0 else min(edge/(var+edge*edge), 0.5/alpha)
        k *= 1.0 + lam*(alpha - x)
        if k < 0: neg = True
        kmax = max(kmax, k); kmin = min(kmin, k)
        n += 1; d = x-mu; mu += d/n; var += (d*(x-mu)-var)/n
    return min(1.0, 1.0/kmax), neg

for model, cal_name, tgt_name in [("av2_gpu_full","av2cal","wm_av2gpu_test"), ("av2_gpu_full","av2cal","ns")]:
    cal, tgt = load(model, cal_name), load(model, tgt_name)
    q_src = split_conformal_quantile(cal["scores"], ALPHA)
    cl = clusters_of(tgt); u, inv = np.unique(cl, return_inverse=True)
    groups = [np.sort(tgt["scores"][inv==c]) for c in range(len(u))]
    sizes = np.array([len(g) for g in groups])
    rng = np.random.default_rng(0)
    for m in (40, 60):
        neg_rej = neg_any = 0; D = 200
        for _ in range(D):
            lab = rng.permutation(len(u))[:m]
            grid = q_src*np.exp(np.linspace(np.log(4), np.log(.5), 300))
            for q in grid:
                L = np.array([1-np.searchsorted(groups[i], q, side="right")/len(groups[i]) for i in lab])
                p, neg = wsr_trace(L, ALPHA)
                neg_any += neg
                if p > DELTA/2: break
                if neg: neg_rej += 1
        print(f"{tgt_name:16s} m={m}: scenes={len(u)} singleton-agent scenes={np.mean(sizes==1):.0%}  "
              f"grid p-values computed with NEGATIVE capital (any)={neg_any}  rejections made while capital went negative={neg_rej}")
