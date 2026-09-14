"""
Single-shot status check + logger for the two background jobs, meant to be re-invoked
every few minutes by Task Scheduler (see run/watch.cmd). Each call:

  1. appends one line to logs/heartbeat.log (human) and logs/heartbeat.jsonl (machine)
  2. on job completion (detected via the DONE_EXIT_<code> marker each .cmd appends),
     takes the next safe automated step exactly once, tracked in logs/watch_state.json

Completion signals:
  results/train_av2_valsplit_v1.out  contains "DONE_EXIT_"   -> training finished
  data/convert_train.log             contains "DONE_EXIT_"   -> AV2 train conversion finished

On training completion (first time seen):
  run predict.py on av2_splits/val/{cal,test} with the best checkpoint, then
  coverage_matrix.py --mode coverage (in-domain only -- no cross-domain data yet).

On conversion completion (first time seen):
  just log it + drop data/av2_scenarionet/train/READY.txt as a marker. The follow-up
  (full-scale retrain) is a deliberate decision, not auto-launched, because of the
  P2000 epoch-time budget tradeoff.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
LOGS.mkdir(exist_ok=True)
STATE_FILE = LOGS / "watch_state.json"
HB_LOG = LOGS / "heartbeat.log"
HB_JSONL = LOGS / "heartbeat.jsonl"
PY = ROOT.parents[0] / "shared" / "envs" / "unitraj" / "Scripts" / "python.exe"

EXP = "av2_cpu_v1"  # bump this + reset watch_state.json's training_done_handled to
                     # track a new training run. History:
                     #  av2_valsplit_v1 (done 09-11): 13,770 scenes, minADE6=1.347, missed
                     #    the ~0.85 kill condition (too little data).
                     #  av2_full_v1 (killed 09-11): launched on the full 199,908-scene AV2
                     #    train set, but SupChain2's ablation sweep had the P2000 at ~4.9/5GB
                     #    and 91-100% util; my job crawled at ~6s/step (would've taken ~6
                     #    days). Killed after confirming it wasn't hung, just contended.
                     #  av2_cpu_v1 (started 09-14): user asked to also run on CPU rather
                     #    than fight for the GPU. Benchmarked 6.2 samp/s @ 16 threads --
                     #    comparable to the contended GPU rate, but doesn't contend.
                     #    60,000-scene subsample (of the cached 199,908, no reprocessing
                     #    needed), 10 epochs, ETA ~25h.
TRAIN_OUT = ROOT / "results" / f"train_{EXP}.out"
TRAIN_ERR = ROOT / "results" / f"train_{EXP}.err"
CKPT_DIR = ROOT / "results" / "ckpts" / EXP
METRICS_CSV = ROOT / "results" / "logs" / EXP / "version_0" / "metrics.csv"

CONV_TRAIN_DIR = ROOT / "data" / "av2_scenarionet" / "train"
CONV_TRAIN_LOG = ROOT / "data" / "convert_train.log"


def _load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"training_done_handled": False, "conversion_done_handled": False}


def _save_state(s: dict):
    STATE_FILE.write_text(json.dumps(s, indent=2))


def _done(marker_file: Path) -> bool:
    if not marker_file.exists():
        return False
    try:
        return "DONE_EXIT_" in marker_file.read_text(errors="ignore")[-500:]
    except Exception:
        return False


def _count_pkl(d: Path) -> int:
    if not d.exists():
        return 0
    return sum(1 for _ in d.rglob("*.pkl"))


def _last_metrics_row() -> dict | None:
    if not METRICS_CSV.exists():
        return None
    lines = METRICS_CSV.read_text().strip().splitlines()
    if len(lines) < 2:
        return None
    header = lines[0].split(",")
    # walk backwards for the last row that actually has values (rows can be sparse)
    for line in reversed(lines[1:]):
        vals = line.split(",")
        row = dict(zip(header, vals))
        if row.get("val/minADE6") or row.get("train/minADE6"):
            return row
    return None


def _best_ckpt() -> Path | None:
    if not CKPT_DIR.exists():
        return None
    cands = [p for p in CKPT_DIR.glob("epoch*.ckpt")]
    if not cands:
        return None
    # filename pattern epoch{NN}-minADE{X.XXX}.ckpt -- lower minADE is better
    def score(p):
        m = re.search(r"minADE([\d.]+)\.ckpt$", p.name)
        return float(m.group(1)) if m else float("inf")
    return min(cands, key=score)


def heartbeat(state: dict) -> dict:
    ts = datetime.now().isoformat(timespec="seconds")
    conv_n = _count_pkl(CONV_TRAIN_DIR)
    conv_done = _done(CONV_TRAIN_LOG)
    train_done = _done(TRAIN_OUT)
    row = _last_metrics_row() or {}
    best = _best_ckpt()

    rec = {
        "ts": ts,
        "conv_train_pkl": conv_n,
        "conv_train_pct": round(100 * conv_n / 199908, 1),
        "conv_train_done": conv_done,
        "train_epoch": row.get("epoch"),
        "train_minADE6": row.get("train/minADE6"),
        "val_minADE6": row.get("val/minADE6"),
        "train_done": train_done,
        "best_ckpt": best.name if best else None,
    }
    HB_JSONL.open("a").write(json.dumps(rec) + "\n")
    line = (f"{ts}  conv={conv_n}/199908({rec['conv_train_pct']}%){' DONE' if conv_done else ''}"
           f"  |  train epoch={rec['train_epoch']} trainADE={rec['train_minADE6']} "
           f"valADE={rec['val_minADE6']}{' DONE' if train_done else ''}"
           f"  best={rec['best_ckpt']}")
    with HB_LOG.open("a") as f:
        f.write(line + "\n")
    print(line)

    # ---- one-shot actions on first-seen completion ----
    if train_done and not state["training_done_handled"]:
        _log_milestone(f"TRAINING FINISHED. best_ckpt={best}")
        if best is not None:
            _run_post_training(best)
        state["training_done_handled"] = True

    if conv_done and not state["conversion_done_handled"]:
        (CONV_TRAIN_DIR / "READY.txt").write_text(f"conversion finished {ts}\n")
        _log_milestone("AV2 TRAIN CONVERSION FINISHED (199,908 scenarios ready). "
                       "Full-scale retrain is a deliberate next step, not auto-launched "
                       "(epoch-time budget tradeoff) -- see STATUS.md.")
        state["conversion_done_handled"] = True

    return state


def _log_milestone(msg: str):
    ts = datetime.now().isoformat(timespec="seconds")
    line = f"{ts}  *** MILESTONE *** {msg}"
    print(line)
    with HB_LOG.open("a") as f:
        f.write(line + "\n")


def _run_post_training(ckpt: Path):
    """
    Predict on cal+test of the val-split (untouched by av2_full_v1's training data --
    AV2's own train/val split is disjoint at the source), then in-domain coverage check.

    NOTE on what this in-domain check IS and ISN'T: av2cal/av2test/av2 below are all the
    SAME underlying distribution (AV2's val split, different halves) -- this confirms SCP
    holds coverage on REAL model predictions (not just synthetic data), which matters, but
    it is NOT the cross-dataset H1 result. That needs a genuinely different dataset
    (nuScenes) predicted with THIS SAME checkpoint, tagged "<other>_from_av2full", dropped
    into results/preds/, then rerun coverage_matrix.py.
    """
    src = Path(__file__).resolve().parent
    preds_dir = ROOT / "results" / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    for split, tag in [("cal", f"av2cal_from_{EXP}"), ("test", f"av2test_from_{EXP}")]:
        db = ROOT / "data" / "av2_splits" / "val" / split
        cmd = [str(PY), str(src / "predict.py"), "--ckpt", str(ckpt), "--db", str(db),
              "--dataset", "av2", "--tag", tag, "--out", str(preds_dir),
              "--device", "cpu", "--batch_size", "32", "--num_workers", "4"]
        _log_milestone(f"running predict.py for {tag} ...")
        r = subprocess.run(cmd, cwd=str(src), capture_output=True, text=True)
        (LOGS / f"predict_{tag}.log").write_text(r.stdout + "\n---STDERR---\n" + r.stderr)
        _log_milestone(f"predict.py {tag} exit={r.returncode}")

    # coverage_matrix.py needs a "<src>_from_<src>.npz" file for its internal 50/50
    # cal/test split; use the calibration-half predictions as that in-domain reference.
    import shutil
    src_file = preds_dir / f"av2cal_from_{EXP}.npz"
    if src_file.exists():
        shutil.copy(src_file, preds_dir / f"{EXP}_from_{EXP}.npz")

    cmd = [str(PY), str(src / "coverage_matrix.py"), "--preds", str(preds_dir),
          "--out", str(ROOT / "results"), "--mode", "coverage"]
    r = subprocess.run(cmd, cwd=str(src), capture_output=True, text=True)
    (LOGS / f"coverage_matrix_{EXP}.log").write_text(r.stdout + "\n---STDERR---\n" + r.stderr)
    _log_milestone(f"coverage_matrix.py exit={r.returncode} -- see logs/coverage_matrix_{EXP}.log "
                   f"and results/coverage_transfer.{{json,csv}} (OVERWRITES the previous run's file -- "
                   f"copy it out first if you need both)")


if __name__ == "__main__":
    state = _load_state()
    state = heartbeat(state)
    _save_state(state)
