"""
Copy UniTraj cache directories to another drive AND rewrite their index so the copy is actually used.

UniTraj's <cache>/<a>/<b>/file_list.pkl stores an ABSOLUTE h5_path per sample.  Copying the folder alone changes
nothing: training keeps reading the original disk (found 2026-09-24: an "SSD copy" was never read; HDD at 10 MB/s).
This tool (1) copies with robocopy (restartable), (2) rewrites every h5_path to the new location, asserting each target
file exists, (3) keeps the original index as file_list.pkl.orig, (4) copies/keeps the SOURCE_DB.txt provenance marker.

  python src/relocate_cache.py --src data/unitraj_cache --dst C:\\p01_cache --rel train_c00/nuscenes_scenarionet ...
  python src/relocate_cache.py --src data/unitraj_cache --dst C:\\p01_cache --all_nuscenes
"""
from __future__ import annotations

import argparse
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def relocate(src_root: Path, dst_root: Path, rel: str) -> int:
    s, d = src_root / rel, dst_root / rel
    d.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["robocopy", str(s), str(d), "/E", "/J", "/R:3", "/W:10", "/NP", "/NFL", "/NDL", "/NJH"],
                       capture_output=True, text=True)
    if r.returncode >= 8:                                   # robocopy: >= 8 means failure
        raise RuntimeError(f"robocopy failed ({r.returncode}) for {rel}:\n{r.stdout[-800:]}")
    idx = d / "file_list.pkl"
    orig = d / "file_list.pkl.orig"
    if not orig.exists():
        shutil.copy2(idx, orig)
    fl = pickle.load(open(orig, "rb"))
    for k, v in fl.items():
        new = d / Path(str(v["h5_path"]).replace("\\", "/")).name
        if not new.exists():
            raise FileNotFoundError(new)
        v["h5_path"] = str(new)
    tmp = d / "file_list.pkl.tmp"
    pickle.dump(fl, open(tmp, "wb")); os.replace(tmp, idx)
    return len(fl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data" / "unitraj_cache"))
    ap.add_argument("--dst", required=True)
    ap.add_argument("--rel", nargs="*", default=[])
    ap.add_argument("--all_nuscenes", action="store_true", help="train_c00..c10 + val/nuscenes_scenarionet + nscal/nsscene + nstest/nsscene")
    a = ap.parse_args()
    rels = list(a.rel)
    if a.all_nuscenes:
        rels += [f"train_c{i:02d}/nuscenes_scenarionet" for i in range(11)] + ["val/nuscenes_scenarionet", "nscal/nsscene", "nstest/nsscene"]
    for r in rels:
        n = relocate(Path(a.src), Path(a.dst), r)
        print(f"[ok] {r}: {n} entries -> {a.dst}", flush=True)


if __name__ == "__main__":
    main()
