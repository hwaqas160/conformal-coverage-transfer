"""Check every arXiv identifier and DOI in paper/refs.bib against the arXiv API / Crossref: the returned title must match the
bibliography title (normalised word overlap). Writes results/rev1/refs_check.json. Network access required."""
import json
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
bib = (ROOT / "paper" / "refs.bib").read_text(encoding="utf-8")
entries = re.findall(r"@(\w+)\{([^,]+),(.*?)\n\}", bib, flags=re.S)


def field(body, name):
    m = re.search(rf"\b{name}\s*=\s*[{{\"](.*?)[}}\"]\s*,?\s*(?:\n|$)", body, flags=re.S)
    return re.sub(r"[{}\s]+", " ", m.group(1)).strip() if m else None


def norm(t):
    return set(re.findall(r"[a-z0-9]{3,}", re.sub(r"[^a-z0-9 ]", " ", t.lower())))


def overlap(a, b):
    A, B = norm(a), norm(b)
    return len(A & B) / max(1, min(len(A), len(B)))


rows, arx = [], {}
for kind, key, body in entries:
    title = field(body, "title") or ""
    text = body
    m = re.search(r"arXiv:(\d{4}\.\d{4,5})", text) or re.search(r"eprint\s*=\s*[{\"](\d{4}\.\d{4,5})", text)
    doi = field(body, "doi")
    rows.append({"key": key, "kind": kind, "title": title, "arxiv": m.group(1) if m else None, "doi": doi})
    if m:
        arx[m.group(1)] = key

ids = sorted(arx)
titles = {}
for i in range(0, len(ids), 20):
    chunk = ids[i:i + 20]
    url = "http://export.arxiv.org/api/query?max_results=50&id_list=" + ",".join(chunk)
    root = ET.fromstring(urllib.request.urlopen(url, timeout=60).read())
    for e in root.findall("{http://www.w3.org/2005/Atom}entry"):
        eid = e.find("{http://www.w3.org/2005/Atom}id").text.rsplit("/", 1)[-1]
        eid = re.sub(r"v\d+$", "", eid)
        titles[eid] = re.sub(r"\s+", " ", e.find("{http://www.w3.org/2005/Atom}title").text).strip()
out = {"arxiv": [], "doi": [], "no_identifier": []}
for r in rows:
    if r["arxiv"]:
        got = titles.get(r["arxiv"])
        ov = overlap(r["title"], got) if got else 0.0
        out["arxiv"].append({"key": r["key"], "id": r["arxiv"], "bib_title": r["title"], "arxiv_title": got, "overlap": round(ov, 2),
                             "status": "OK" if ov >= 0.7 else ("NOT FOUND" if not got else "MISMATCH")})
    elif r["doi"]:
        try:
            j = json.loads(urllib.request.urlopen("https://api.crossref.org/works/" + urllib.parse.quote(r["doi"]), timeout=60).read())
            got = (j["message"].get("title") or [""])[0]
            ov = overlap(r["title"], got)
            out["doi"].append({"key": r["key"], "doi": r["doi"], "bib_title": r["title"], "crossref_title": got, "overlap": round(ov, 2),
                               "status": "OK" if ov >= 0.7 else "MISMATCH"})
        except Exception as e:  # noqa: BLE001
            out["doi"].append({"key": r["key"], "doi": r["doi"], "status": f"ERROR {e}"})
    else:
        out["no_identifier"].append(r["key"])
(ROOT / "results" / "rev1" / "refs_check.json").write_text(json.dumps(out, indent=1))
bad = [x for x in out["arxiv"] + out["doi"] if x["status"] != "OK"]
print(f"entries {len(rows)} | arXiv {len(out['arxiv'])} (ok {sum(x['status']=='OK' for x in out['arxiv'])}) | "
      f"DOI {len(out['doi'])} (ok {sum(x['status']=='OK' for x in out['doi'])}) | no identifier {len(out['no_identifier'])}")
for x in bad:
    print("PROBLEM:", x["key"], x["status"], "|", x.get("bib_title"), "|", x.get("arxiv_title") or x.get("crossref_title"))
print("no identifier (books/proceedings; check by hand):", out["no_identifier"])
