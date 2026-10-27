"""Build the IEEE T-IV initial-submission package from the committed manuscript.

Output folder: T-IV_initial_submission_package/ (manuscript PDF, LaTeX source zip, checklist copy placeholders are written separately).
The source zip contains only what the manuscript compiles from (no photos: T-IV asks for them with the final files) and is test-compiled
in a clean directory by test_package_build() to prove that the zip alone reproduces the PDF."""
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
PKG = ROOT / "T-IV_initial_submission_package"
NAME = "Saleem_ConformalCoverageTransfer_TIV"
CLS = Path(r"C:/Users/hassan.waqas/AppData/Local/Programs/MiKTeX/tex/latex/ieeetran/IEEEtran.cls")
BST = Path(r"C:/Users/hassan.waqas/AppData/Local/Programs/MiKTeX/bibtex/bst/ieeetran/IEEEtran.bst")

GEN = ["numbers.tex", "numbers_addB.tex", "numbers_addC.tex", "numbers_rev1.tex", "table_models.tex", "table_inject.tex",
       "fig_gap.pdf", "fig_scene_repair.pdf", "fig_traj_conf.pdf"]
SRC_FILES = ["main.tex", "defaults.tex", "refs.bib"] + [f"sections/{p.name}" for p in sorted((PAPER / "sections").glob("*.tex"))] \
    + [f"generated/{g}" for g in GEN]


def build():
    for d in ("01_manuscript", "02_source", "03_checklist"):
        (PKG / d).mkdir(parents=True, exist_ok=True)   # documents added by hand are kept; files below are overwritten
    shutil.copy2(PAPER / "main.pdf", PKG / "01_manuscript" / f"{NAME}_manuscript.pdf")
    zp = PKG / "02_source" / f"{NAME}_source.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in SRC_FILES:
            z.write(PAPER / f, f)
        z.write(CLS, "IEEEtran.cls"); z.write(BST, "IEEEtran.bst")
    return zp


def test_package_build(zp):
    tmp = Path(os.environ.get("TEMP", ".")) / "tiv_pkg_test"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    with zipfile.ZipFile(zp) as z:
        z.extractall(tmp)
    run = lambda *a: subprocess.run(a, cwd=tmp, capture_output=True, text=True)
    run("pdflatex", "-interaction=nonstopmode", "main.tex"); run("bibtex", "main")
    run("pdflatex", "-interaction=nonstopmode", "main.tex"); r = run("pdflatex", "-interaction=nonstopmode", "main.tex")
    pdf = tmp / "main.pdf"
    ok = pdf.exists()
    errs = [l for l in r.stdout.splitlines() if l.startswith("!") or "Undefined control" in l or "undefined" in l.lower() and "reference" in l.lower()]
    txt_a = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True, encoding="utf-8", errors="ignore").stdout
    txt_b = subprocess.run(["pdftotext", str(PAPER / "main.pdf"), "-"], capture_output=True, text=True, encoding="utf-8", errors="ignore").stdout
    pages = lambda p: next((l.split()[-1] for l in subprocess.run(["pdfinfo", str(p)], capture_output=True, text=True).stdout.splitlines() if l.startswith("Pages")), "?")
    print(f"zip -> PDF built: {ok}; LaTeX errors: {errs or 'none'}; pages zip/manuscript: {pages(pdf)}/{pages(PAPER / 'main.pdf')}; "
          f"text identical: {txt_a == txt_b}")
    return ok and not errs and txt_a == txt_b


if __name__ == "__main__":
    zp = build()
    print("package written:", PKG)
    sys.exit(0 if test_package_build(zp) else 1)
