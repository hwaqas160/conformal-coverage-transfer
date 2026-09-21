"""
Chunked, resumable nuScenes (prediction-challenge split) -> ScenarioNet conversion.

Why: the stock converter has no resume; the val split took ~36 h wall-clock with two native
GEOS 'bad allocation' crashes, each restarting from zero. Here each chunk of N scenarios is written to its
own database  data/nuscenes_scenarionet/<split>_c<NN>/ , and a chunk is SKIPPED if its dataset_summary.pkl already
exists.  A crash therefore loses at most one chunk. UniTraj accepts a LIST of data paths, so no merge step is needed.

  python src/convert_ns_chunked.py --split train --chunk 3000 --workers 3
  python src/convert_ns_chunked.py --split train --chunk 3000 --workers 3 --max_chunks 4     # only first 4 chunks

Records every chunk start/finish/failure in journal/events.jsonl.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))
import journal  # noqa: E402

DATAROOT = ROOT / "data" / "nuscenes" / "v1.0-trainval_meta"
OUTROOT = ROOT / "data" / "nuscenes_scenarionet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "train_val", "val"])
    ap.add_argument("--chunk", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=3, help="8 workers crashed with GEOS bad_alloc; 3 is stable")
    ap.add_argument("--max_chunks", type=int, default=None)
    a = ap.parse_args()

    from scenarionet.converter.nuscenes.utils import (convert_nuscenes_scenario,
                                                       get_nuscenes_prediction_split)
    from scenarionet.converter.utils import write_to_directory

    exp = f"ns_{a.split}_convert"
    scenarios, nuscs = get_nuscenes_prediction_split(str(DATAROOT), a.split, 2, 6, a.workers)
    n = len(scenarios)
    n_chunks = (n + a.chunk - 1) // a.chunk
    journal.event(exp, "attempt_start", f"{n} scenarios, {n_chunks} chunks of {a.chunk}, workers={a.workers}")

    done = 0
    for c in range(n_chunks):
        if a.max_chunks is not None and c >= a.max_chunks:
            break
        out = OUTROOT / f"{a.split}_c{c:02d}"
        if (out / "dataset_summary.pkl").exists():
            print(f"[skip] chunk {c} already converted -> {out.name}")
            done += 1
            continue
        if out.exists():                       # partial output from a crashed attempt
            shutil.rmtree(out, ignore_errors=True)
        chunk = scenarios[c * a.chunk:(c + 1) * a.chunk]
        t0 = time.time()
        journal.event(exp, "artifact", f"chunk {c}/{n_chunks} start ({len(chunk)} scenarios)")
        try:
            write_to_directory(
                convert_func=convert_nuscenes_scenario, scenarios=chunk, output_path=str(out),
                dataset_version=a.split, dataset_name=f"ns_{a.split}_c{c:02d}", overwrite=True,
                num_workers=a.workers, nuscenes=nuscs,
                past=[2] * a.workers, future=[6] * a.workers,
                prediction=[True] * a.workers, map_radius=[500] * a.workers,
            )
        except BaseException as e:            # noqa: BLE001  (native GEOS failures surface as many types)
            journal.event(exp, "attempt_end", f"chunk {c} FAILED after {time.time()-t0:.0f}s: {type(e).__name__}: {str(e)[:100]}")
            shutil.rmtree(out, ignore_errors=True)
            raise
        journal.event(exp, "artifact", f"chunk {c} done in {time.time()-t0:.0f}s -> {out.relative_to(ROOT)}")
        done += 1
    journal.event(exp, "done" if done == n_chunks else "attempt_end", f"{done}/{n_chunks} chunks present")


if __name__ == "__main__":
    main()
