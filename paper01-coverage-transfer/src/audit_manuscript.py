"""Mechanical manuscript audit used for the initial-submission package: citations vs bibliography, repeated words, spelling-variant
consistency (British vs American), double spaces, space before punctuation, unmatched TeX braces/math. Prints findings; exit 0."""
import glob
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "paper"
files = sorted(glob.glob(str(ROOT / "sections" / "*.tex"))) + [str(ROOT / "main.tex")]
txt = {f: Path(f).read_text(encoding="utf-8") for f in files}
allt = "\n".join(txt.values())
bib = (ROOT / "refs.bib").read_text(encoding="utf-8")
keys = re.findall(r"@\w+\{([^,]+),", bib)
cited = set()
for m in re.findall(r"\\cite[a-z]*(?:\[[^\]]*\])?\{([^}]*)\}", allt):
    cited |= {k.strip() for k in m.split(",")}
print(f"bibliography entries {len(keys)} | cited {len(cited & set(keys))} | uncited {len(set(keys) - cited)} | "
      f"cited but missing {sorted(cited - set(keys))}")

plain = re.sub(r"%.*", "", allt)
plain = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", plain)
plain = re.sub(r"\$[^$]*\$", " ", plain)
words = re.findall(r"[A-Za-z']+", plain)
rep = [(a, b) for a, b in zip(words, words[1:]) if a.lower() == b.lower() and len(a) > 2 and a.lower() not in ("that", "had")]
print("repeated words:", sorted(set(f"{a} {b}" for a, b in rep)) or "none")

pairs = [("labelled", "labeled"), ("unlabelled", "unlabeled"), ("normalised", "normalized"), ("behaviour", "behavior"),
         ("modelling", "modeling"), ("colour", "color"), ("generalis", "generaliz"), ("recognis", "recogniz"), ("centre", "center"),
         ("analyse", "analyze"), ("summaris", "summariz"), ("organis", "organiz"), ("regularis", "regulariz"), ("optimis", "optimiz"),
         ("licence", "license"), ("favour", "favor")]
low = plain.lower()
for br, am in pairs:
    nb, na = low.count(br), low.count(am)
    if nb and na:
        print(f"MIXED spelling: {br} x{nb} vs {am} x{na}")
print("double spaces in prose lines:", sum(1 for l in plain.split("\n") if re.search(r"\S  +\S", l)))
print("space before , . ; :", len(re.findall(r"\w [,.;:](?!\w)", plain)))
for f, t in txt.items():
    t2 = re.sub(r"%.*", "", t)
    if t2.count("{") != t2.count("}"):
        print("UNBALANCED braces:", Path(f).name, t2.count("{"), t2.count("}"))
    if len(re.findall(r"(?<!\\)\$", t2)) % 2:
        print("ODD number of $:", Path(f).name)
