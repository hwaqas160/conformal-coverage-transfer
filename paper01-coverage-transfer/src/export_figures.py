"""Export every manuscript figure as separate vector PDF and 600-dpi PNG files (for the portal's optional Image slot).
Fig. 2-4 are the generated PDFs; Fig. 1 (TikZ) and Fig. 5 (boxed protocol) are compiled standalone from the manuscript's own source,
so the images are the manuscript's figures, not redrawn."""
import os, re, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; PAPER = ROOT / "paper"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "T-IV_UPLOAD_FOLDER" / "5_IMAGES_optional_figures"
TMP = Path(os.environ.get("TEMP", ".")) / "tiv_fig_export"
shutil.rmtree(TMP, ignore_errors=True); TMP.mkdir(parents=True); OUT.mkdir(parents=True, exist_ok=True)

PRE = r"""\documentclass[border=3pt]{standalone}
\usepackage{amsmath,amssymb,enumitem,xcolor,tikz}
\usetikzlibrary{positioning,arrows.meta,calc}
\definecolor{okBlue}{HTML}{0072B2}\definecolor{okSky}{HTML}{56B4E9}\definecolor{okOrange}{HTML}{E69F00}
\definecolor{okVermillion}{HTML}{D55E00}\definecolor{okGreen}{HTML}{009E73}\definecolor{okPurple}{HTML}{CC79A7}
\InputIfFileExists{generated/numbers_rev1.tex}{}{}
\usepackage{times}
\begin{document}
"""

def block(path, start, end):
    s = (PAPER / path).read_text(encoding="utf-8")
    a = s.index(start); b = s.index(end, a) + len(end)
    return s[a:b]

fig1 = block("sections/intro.tex", r"\begin{tikzpicture}", r"\end{tikzpicture}").replace("text width=24mm", "text width=27mm")  # export only: avoids a font-dependent hyphenation
fig5 = block("sections/discussion.tex", r"\fbox{\begin{minipage}", r"\end{minipage}}")
fig5 = (fig5.replace(r"\begin{minipage}{0.95\columnwidth}", r"\begin{minipage}{3.4in}")
        .replace(r"Sec.~\ref{sec:audit}", "Sec.~III-B").replace(r"Sec.~\ref{sec:scene}", "Sec.~III-D"))
assert r"\ref" not in fig5, "unresolved reference in Fig. 5 source"

def compile_standalone(name, body):
    d = TMP / name; (d / "generated").mkdir(parents=True)
    shutil.copy2(PAPER / "generated/numbers_rev1.tex", d / "generated/numbers_rev1.tex")
    (d / "f.tex").write_text(PRE + body + "\n\\end{document}\n", encoding="utf-8")
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "f.tex"], cwd=d, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if not (d / "f.pdf").exists():
        print(r.stdout[-1500:]); raise SystemExit(f"{name} failed")
    return d / "f.pdf"

sources = {
    "Fig1_transfer_problem": compile_standalone("fig1", fig1),
    "Fig2_coverage_source_vs_target": PAPER / "generated/fig_gap.pdf",
    "Fig3_coverage_by_manoeuvre": PAPER / "generated/fig_traj_conf.pdf",
    "Fig4_scene_level_repair": PAPER / "generated/fig_scene_repair.pdf",
    "Fig5_audit_then_repair_protocol": compile_standalone("fig5", fig5),
}
for name, pdf in sources.items():
    shutil.copy2(pdf, OUT / f"{name}.pdf")
    subprocess.run(["pdftoppm", "-png", "-r", "600", "-singlefile", str(pdf), str(OUT / name)], check=True)
    w = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
    size = re.search(r"Page size:\s+([\d.]+) x ([\d.]+)", w)
    print(f"{name}: {float(size.group(1))/72:.2f} x {float(size.group(2))/72:.2f} in, PDF + 600-dpi PNG")
