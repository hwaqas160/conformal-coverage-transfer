"""
Deterministic scene-level split of a ScenarioNet DB into train / cal / test sub-DBs.

A ScenarioNet 'database' is just dataset_summary.pkl + dataset_mapping.pkl pointing at
existing per-worker .pkl files. Splitting = writing filtered copies of those two files
into new directories (no scenario data is copied).

Split is by a stable hash of the scenario_id, so:
  * it is reproducible
  * the same scenario always lands in the same split regardless of DB ordering
  * train/cal/test never overlap

Usage:
  python split_db.py --db data/av2_scenarionet/val --out data/av2_splits/val \
      --fractions 0.6 0.2 0.2 --names train cal test --salt av2v1
"""
from __future__ import annotations

import argparse
import hashlib
import pickle
from pathlib import Path


def _bucket(scenario_id: str, salt: str, nbuckets: int = 10000) -> int:
    h = hashlib.sha256(f"{salt}:{scenario_id}".encode()).hexdigest()
    return int(h[:8], 16) % nbuckets


def split_db(db: str, out: str, fractions, names, salt: str, group_regex: str | None = None):
    """group_regex: hash a GROUP key (first regex match in the scenario id) instead of the scenario id, so every
    member of a group lands in the same split.  Needed for nuScenes, whose ~60 agent-scenarios per 20 s scene are
    strongly correlated (use r'scene-\\d+'); AV2 scenarios are independent clips and need no grouping."""
    import re
    db = Path(db)
    out = Path(out)
    assert abs(sum(fractions) - 1.0) < 1e-6, "fractions must sum to 1"
    assert len(fractions) == len(names)

    summary = pickle.load(open(db / "dataset_summary.pkl", "rb"))
    mapping = pickle.load(open(db / "dataset_mapping.pkl", "rb"))

    # cumulative bucket edges
    edges, acc = [], 0.0
    for f in fractions:
        acc += f
        edges.append(int(round(acc * 10000)))

    parts = {n: ({}, {}) for n in names}
    for fname, meta in summary.items():
        sid = str(meta.get("scenario_id", fname))
        if group_regex:
            m = re.search(group_regex, sid) or re.search(group_regex, fname)
            if m is None:
                raise ValueError(f"group_regex {group_regex!r} does not match scenario {sid!r}")
            sid = m.group(0)
        b = _bucket(sid, salt)
        for name, edge in zip(names, edges):
            if b < edge:
                parts[name][0][fname] = meta
                parts[name][1][fname] = mapping[fname]
                break

    out.mkdir(parents=True, exist_ok=True)
    for name in names:
        sub = out / name
        sub.mkdir(parents=True, exist_ok=True)
        s_map, m_map = parts[name]
        # relativise the worker-dir paths against the NEW db location if needed:
        # ScenarioNet resolves mapping paths relative to the db dir, so we must also
        # copy/symlink the worker dirs OR keep the mapping pointing at the source.
        # Simplest + robust on Windows: make the mapping values ABSOLUTE paths into the
        # original db, and drop a marker file. UniTraj's read_scenario joins db_path with
        # the mapping value, so an absolute value wins the join.
        abs_map = {k: str((db / v).resolve()) for k, v in m_map.items()}
        pickle.dump(s_map, open(sub / "dataset_summary.pkl", "wb"))
        pickle.dump(abs_map, open(sub / "dataset_mapping.pkl", "wb"))
        (sub / "SOURCE.txt").write_text(f"split of {db.resolve()}\nsalt={salt}\ngroup_regex={group_regex}\nn={len(s_map)}\n")
        print(f"  {name:8s} {len(s_map):7d} scenarios -> {sub}")

    total = sum(len(parts[n][0]) for n in names)
    print(f"total {total} (source had {len(summary)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.6, 0.2, 0.2])
    ap.add_argument("--names", nargs="+", default=["train", "cal", "test"])
    ap.add_argument("--salt", default="av2v1")
    ap.add_argument("--group_regex", default=None, help=r"e.g. 'scene-\d+' for nuScenes (scene-level split)")
    a = ap.parse_args()
    split_db(a.db, a.out, a.fractions, a.names, a.salt, a.group_regex)
