"""Addendum J: does a source-trained per-scene loss predictor rank target scenes? See notes/falsification.md."""
import json, re, numpy as np
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
ROOT = Path(__file__).resolve().parents[1]; P = ROOT / "results" / "preds2"; ALPHA = 0.1
CELLS = {"av2_gpu_full": ("av2cal", ["ns", "wm_av2gpu_test"]), "av2_wayformer": ("av2cal", ["ns", "wmtest"]),
         "ns_gpu_full": ("nscal", ["av2test", "wm_nsgpu_test"])}

def load(m, s):
    d = np.load(P / m / f"{s}.npz", allow_pickle=True)
    pt, pr = d["pred_trajs"], d["pred_probs"]
    end = pt[:, :, -1, :]; start = pt[:, :, 0, :]
    dist = np.linalg.norm(end - start, axis=-1)
    ent = -(pr * np.log(np.clip(pr, 1e-9, 1))).sum(1)
    X = np.column_stack([d["feats_lf"], dist.mean(1), dist.max(1), end.std(1).mean(-1), ent, pr.max(1)])
    X = np.nan_to_num(X)
    sid = np.array([(re.search(r"scene-\d+", str(x)) or [str(x)])[0] for x in d["scenario_id"]])
    return X, d["scores"].astype(float), sid

def qhat(s):
    n = len(s); return float(np.quantile(s, min(np.ceil((n + 1) * (1 - ALPHA)) / n, 1.0), method="higher"))

def scene_stats(pred, miss, sid):
    u, inv = np.unique(sid, return_inverse=True)
    c = np.bincount(inv); return (np.bincount(inv, weights=pred) / c, np.bincount(inv, weights=miss) / c, c)

def boot_rho(a, b, reps=1000, seed=0):
    rng = np.random.default_rng(seed); n = len(a); v = []
    for _ in range(reps):
        i = rng.integers(0, n, n); v.append(spearmanr(a[i], b[i])[0])
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]

out = {}
for m, (cal, tg) in CELLS.items():
    X, s, sid = load(m, cal); q = qhat(s); y = (s > q).astype(int)
    perm = np.random.default_rng(0).permutation(len(y)); A, B = perm[: len(y) // 2], perm[len(y) // 2:]
    models = {"logit": make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)),
              "gbm": GradientBoostingClassifier(max_depth=3, n_estimators=200, random_state=0)}
    out[m] = {"q_src": q}
    for name, mod in models.items():
        mod.fit(X[A], y[A]); cell = {}
        for tname, (Xt, st, sidt, ref) in {"control_heldout_source": (X[B], s[B], sid[B], B), **{t: (*load(m, t), None) for t in tg}}.items():
            p = mod.predict_proba(Xt)[:, 1]; miss = (st > q).astype(float)
            ps, ms_, cnt = scene_stats(p, miss, sidt)
            ok = ms_.std() > 0 and ps.std() > 0
            rho = float(spearmanr(ps, ms_)[0]) if ok else float("nan"); r = float(pearsonr(ps, ms_)[0]) if ok else float("nan")
            cell[tname] = {"n_scenes": int(len(ps)), "mean_pred": float(ps.mean()), "mean_true": float(ms_.mean()),
                           "spearman": rho, "spearman_ci": boot_rho(ps, ms_) if ok else None, "r2": r * r if ok else None}
        out[m][name] = cell
(ROOT / "results" / "rev1" / "methodA_step1.json").write_text(json.dumps(out, indent=1))
for m, r in out.items():
    print(m, "q=%.2f" % r["q_src"])
    for name in ("logit", "gbm"):
        for t, c in r[name].items():
            print(f"  {name:5s} {t:24s} n={c['n_scenes']:5d} pred={c['mean_pred']:.2f} true={c['mean_true']:.2f} rho={c['spearman']:.2f} r2={c['r2']:.2f}")
