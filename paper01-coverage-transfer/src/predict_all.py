"""
Resumable prediction driver: run ONE model over a list of evaluation sets, writing
results/preds2/<model>/<set>.npz (schema v2: includes pred_scale + label-free features).

* idempotent -- an existing, loadable output is skipped, so re-running after a crash continues
* every step is recorded in journal/events.jsonl
* --ckpt auto  -> lowest-minADE checkpoint in results/ckpts/<model>/ (filename epochNN-minADEx.xxx.ckpt)

  python src/predict_all.py --model av2_cpu_v1 --ckpt results/ckpts/av2_cpu_v1/epoch08-minADE1.349.ckpt
  python src/predict_all.py --model av2_cpu_v2 --ckpt auto --sets av2cal av2test ns
"""
from __future__ import annotations

import argparse
import re
import sys
import traceback
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))
import journal  # noqa: E402

# evaluation sets: name -> (ScenarioNet DB dir, dataset key for native-dt lookup)
SETS = {
    "av2cal":  ("data/av2_splits/val/cal", "av2"),
    "av2test": ("data/av2_splits/val/test", "av2"),
    "ns":      ("data/nuscenes_scenarionet/val", "ns"),
    # 2026-09-23: (1) was nuscenes_splits/val/{cal,test}; UniTraj keys its cache on the last two path components,
    # so those names collided with av2_splits/val/{cal,test} and silently loaded AV2 data.  (2) now split by SCENE
    # (salt nsv2scene, 64/74 scenes, 0 overlap): nuScenes has ~60 correlated agent-scenarios per scene.
    "nscal":   ("data/nuscenes_splits/nsscene/nscal", "ns"),
    "nstest":  ("data/nuscenes_splits/nsscene/nstest", "ns"),
}


def best_ckpt(model: str) -> Path:
    d = ROOT / "results" / "ckpts" / model
    cands = list(d.glob("epoch*.ckpt"))
    if not cands:
        raise FileNotFoundError(f"no epoch*.ckpt in {d}")

    def score(p):
        m = re.search(r"minADE([\d.]+)\.ckpt$", p.name)
        return float(m.group(1)) if m else 1e9
    return min(cands, key=score)


def _ok(path: Path) -> bool:
    try:
        d = np.load(path, allow_pickle=True)
        return "pred_scale" in d.files and len(d["scores"]) > 0
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", required=True, help="path or 'auto'")
    ap.add_argument("--sets", nargs="+", default=["av2cal", "av2test", "ns"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=2)
    a = ap.parse_args()

    ckpt = best_ckpt(a.model) if a.ckpt == "auto" else Path(a.ckpt)
    if not ckpt.is_absolute():
        ckpt = (ROOT / ckpt).resolve()
    outdir = ROOT / "results" / "preds2" / a.model
    exp_id = f"predict_{a.model}"
    journal.event(exp_id, "attempt_start", f"ckpt={ckpt.name} sets={a.sets}")

    from predict import dump
    failed = []
    for name in a.sets:
        db, key = SETS[name]
        out = outdir / f"{name}.npz"
        if out.exists() and _ok(out):
            print(f"[skip] {out} exists")
            continue
        try:
            dump(str(ckpt), str(ROOT / db), key, f"{name}_from_{a.model}", str(outdir),
                 device=a.device, batch_size=a.batch_size, num_workers=a.num_workers, out_file=str(out))
            journal.event(exp_id, "artifact", f"{out.relative_to(ROOT)}")
        except Exception as e:  # keep going: other sets may still succeed; caller retries
            traceback.print_exc()
            failed.append(name)
            journal.event(exp_id, "attempt_end", f"set {name} FAILED: {type(e).__name__}: {str(e)[:120]}")
    if failed:
        journal.event(exp_id, "failed", f"sets failed: {failed}")
        sys.exit(1)
    journal.event(exp_id, "done", f"model={a.model} ckpt={ckpt.name} -> {outdir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
