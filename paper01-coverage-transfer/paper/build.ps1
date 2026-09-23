# One-command, idempotent rebuild: regenerate results, compile, and report what's still unresolved.
# Safe to rerun any time (e.g. right after a crash) -- never hand-edit generated/*.tex or the .pdf.
# Usage: cd paper; powershell -File build.ps1
$ErrorActionPreference = "Stop"
$PY = "F:\CLAUDE\AI1\shared\envs\unitraj\Scripts\python.exe"
$root = Split-Path $PSScriptRoot -Parent

Write-Output "== 1/3 regenerating generated/*.tex and figures from results/run2/*.json =="
& $PY "$PSScriptRoot\make_results.py"

Write-Output "== 2/3 compiling (pdflatex, bibtex, pdflatex x2) =="
Push-Location $PSScriptRoot
try {
    & pdflatex -interaction=nonstopmode main.tex | Out-Null
    & bibtex main | Out-Null
    & pdflatex -interaction=nonstopmode main.tex | Out-Null
    $log = & pdflatex -interaction=nonstopmode main.tex
    Remove-Item main.aux, main.bbl, main.blg, main.log, main.out -ErrorAction SilentlyContinue
} finally {
    Pop-Location
}

Write-Output "== 3/3 status =="
$undefined = $log | Select-String -Pattern "Undefined control|Citation.*undefined|Reference.*undefined"
if ($undefined) { Write-Output "COMPILE PROBLEMS:"; $undefined | ForEach-Object { Write-Output "  $_" } }
else { Write-Output "compiled clean, no undefined refs/citations" }

$todos = Select-String -Path "$PSScriptRoot\sections\*.tex","$PSScriptRoot\main.tex" -Pattern '\\todo\{' -AllMatches
Write-Output "$($todos.Count) \todo{} placeholder(s) remaining, by file:"
$todos | Group-Object Path | ForEach-Object { Write-Output ("  {0,3}  {1}" -f $_.Count, (Split-Path $_.Name -Leaf)) }

Write-Output "PDF: $PSScriptRoot\main.pdf"
