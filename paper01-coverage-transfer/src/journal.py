"""
Crash-safe research journal. Two append-only artifacts (append = a crash can never corrupt
earlier records), both committed to git under journal/:

  journal/events.jsonl   machine log: one JSON object per line
                         {ts, id, event, detail, sha}. Written by run/*.cmd wrappers and by
                         analysis scripts, so a fresh session can reconstruct exactly what ran,
                         what finished, and what failed.
  journal/JOURNAL.md     human log: dated entries (decision / evidence / outcome).

CLI
  python src/journal.py event  <id> <event> [detail...]     e.g.  event av2_cpu_v2 attempt_start 1
  python src/journal.py note   "free text" [--tag TAG]       append a dated entry to JOURNAL.md
  python src/journal.py status                               last event per experiment id
  python src/journal.py tail   [N]                           last N events (default 15)

Conventions for <event>: created | attempt_start | attempt_end | done | failed | artifact | result
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JDIR = ROOT / "journal"
EVENTS = JDIR / "events.jsonl"
JOURNAL = JDIR / "JOURNAL.md"


def _sha() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def event(id_: str, ev: str, detail: str = "") -> dict:
    JDIR.mkdir(exist_ok=True)
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "id": id_, "event": ev,
           "detail": detail, "sha": _sha()}
    with EVENTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def note(text: str, tag: str = "") -> None:
    JDIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M")
    head = f"\n### {stamp}" + (f" — {tag}" if tag else "")
    with JOURNAL.open("a", encoding="utf-8") as f:
        f.write(f"{head}\n{text.strip()}\n")


def _read_events() -> list[dict]:
    if not EVENTS.exists():
        return []
    out = []
    for line in EVENTS.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # tolerate a torn last line after a hard crash
    return out


def status() -> None:
    last: dict[str, dict] = {}
    for r in _read_events():
        last[r["id"]] = r
    for id_, r in sorted(last.items()):
        print(f"{id_:34s} {r['event']:14s} {r['ts']}  {r['detail'][:70]}")


def tail(n: int = 15) -> None:
    for r in _read_events()[-n:]:
        print(f"{r['ts']}  {r['id']:30s} {r['event']:14s} {r['detail'][:80]}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        print(__doc__); sys.exit(0)
    cmd = a[0]
    if cmd == "event" and len(a) >= 3:
        print(json.dumps(event(a[1], a[2], " ".join(a[3:]))))
    elif cmd == "note" and len(a) >= 2:
        tag = ""
        if "--tag" in a:
            i = a.index("--tag"); tag = a[i + 1]; a = a[:i] + a[i + 2:]
        note(" ".join(a[1:]), tag)
    elif cmd == "status":
        status()
    elif cmd == "tail":
        tail(int(a[1]) if len(a) > 1 else 15)
    else:
        print(__doc__)
