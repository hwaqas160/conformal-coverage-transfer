"""Build the T-IV portal upload folder for Paper 1 (layout follows the real portal's 'Upload Manuscript' page).

Portal slots seen in the authors' screenshots (2026-10-29): Main Manuscript (required; MS Word or LaTeX, LaTeX bundled as one archive),
Conflict of Interest (required; tick 'None of the authors have a conflict' when none, no file), optional: Supplementary Material, Tracked
Changes, Image, Previously Published, Cover letter / Comments, LaTeX Supplementary File.
The LaTeX archive is test-compiled twice in clean folders: with BibTeX, and WITHOUT BibTeX using the shipped main.bbl."""
import os, re, shutil, subprocess, sys, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT = ROOT / "T-IV_UPLOAD_FOLDER"
NAME = "Saleem_ConformalCoverageTransfer"
CLS = Path(r"C:/Users/hassan.waqas/AppData/Local/Programs/MiKTeX/tex/latex/ieeetran/IEEEtran.cls")
BST = Path(r"C:/Users/hassan.waqas/AppData/Local/Programs/MiKTeX/bibtex/bst/ieeetran/IEEEtran.bst")
GEN = ["numbers.tex", "numbers_addB.tex", "numbers_addC.tex", "numbers_rev1.tex", "table_models.tex", "table_inject.tex",
       "fig_gap.pdf", "fig_scene_repair.pdf", "fig_traj_conf.pdf"]
SRC = ["main.tex", "defaults.tex", "refs.bib"] + [f"sections/{p.name}" for p in sorted((PAPER / "sections").glob("*.tex"))] + [f"generated/{g}" for g in GEN]


def run(cwd, *a):
    return subprocess.run(a, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="ignore")


def compile_tree(tmp, bibtex):
    run(tmp, "pdflatex", "-interaction=nonstopmode", "main.tex")
    if bibtex:
        run(tmp, "bibtex", "main")
    run(tmp, "pdflatex", "-interaction=nonstopmode", "main.tex")
    r = run(tmp, "pdflatex", "-interaction=nonstopmode", "main.tex")
    return r.stdout


def fresh(p):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True)
    return p


def text(pdf):
    return subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True, encoding="utf-8", errors="ignore").stdout


def main():
    tmp = Path(os.environ.get("TEMP", ".")) / "tiv_upload_test"
    # 1. stage the sources, compile once with BibTeX to obtain main.bbl
    stage = fresh(tmp / "stage")
    for f in SRC:
        (stage / f).parent.mkdir(parents=True, exist_ok=True); shutil.copy2(PAPER / f, stage / f)
    shutil.copy2(CLS, stage / "IEEEtran.cls"); shutil.copy2(BST, stage / "IEEEtran.bst")
    compile_tree(stage, bibtex=True)
    assert (stage / "main.bbl").exists(), "bbl not produced"
    # 2. zip: sources + class + bst + main.bbl (no PDF, no aux files)
    fresh(OUT)
    d1 = OUT / "1_MAIN_MANUSCRIPT_required"; d1.mkdir()
    zp = d1 / f"{NAME}_TIV_main_manuscript_LaTeX.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in SRC:
            z.write(PAPER / f, f)
        z.write(CLS, "IEEEtran.cls"); z.write(BST, "IEEEtran.bst"); z.write(stage / "main.bbl", "main.bbl")
    # 3. test A: with BibTeX; test B: without BibTeX (portal that only runs pdflatex)
    ref_txt = text(PAPER / "main.pdf"); ok = True
    for label, bib in (("with BibTeX", True), ("WITHOUT BibTeX (uses shipped main.bbl)", False)):
        t = fresh(tmp / ("A" if bib else "B"))
        with zipfile.ZipFile(zp) as z:
            z.extractall(t)
        log = compile_tree(t, bib)
        pdf = t / "main.pdf"
        errs = [l for l in log.splitlines() if l.startswith("!") or "Undefined control" in l or ("undefined" in l.lower() and "reference" in l.lower())]
        pages = next((l.split()[-1] for l in subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout.splitlines() if l.startswith("Pages")), "?")
        same = pdf.exists() and text(pdf) == ref_txt
        print(f"  zip compile {label}: pdf={pdf.exists()} errors={errs or 'none'} pages={pages} text identical to paper/main.pdf={same}")
        ok = ok and pdf.exists() and not errs and same
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
