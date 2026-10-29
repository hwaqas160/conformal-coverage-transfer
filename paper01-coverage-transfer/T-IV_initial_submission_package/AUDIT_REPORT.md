# Final IEEE T-IV initial-submission audit (2026-10-27)

Manuscript: *Does the Conformal Guarantee Travel? ...* (Regular paper, 10 pages). Repository state: `paper01-coverage-transfer`, built from the
committed files; a fresh clone regenerates every generated number and table identically, and the source zip in this package rebuilds the PDF
(10 pages, no LaTeX errors, text identical to the manuscript PDF).

## A. Executive verdict: **READY AFTER MINOR FIXES**
The manuscript meets every T-IV initial-submission requirement that I could verify. Two items remain that only the authors can settle
(section E); none requires new experiments or rewriting. One item could not be verified from the portal (section B, row 24).

## Sources (fetched 2026-10-26/27)
- S1 T-IV journal page: https://ieee-itss.org/pub/t-iv/ (checklist, paper types, lengths, abstract rules, final-file items)
- S2 T-IV author information: https://ieee-itss.org/pub/t-iv/author/ (initial-submission formatting, biographies, overlength rule)
- S3 IEEE AI-content policy: https://open.ieee.org/author-guidelines-for-artificial-intelligence-ai-generated-text/
- S4 IEEE graphics pages: https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-graphics-for-your-article/ (+ /resolution-and-size/ and /file-formatting/)
- S5 TIV Author Portal https://ieee.atyponrex.com/journal/t-iv (redirects to ieee.submission.researchexchange.com): JavaScript only; its content could not be read here.

## B. Compliance table
| # | Requirement | Source | Status | Evidence / action |
|---|---|---|---|---|
| 1 | IEEE T-IV template, double column, single spaced | S1, S2 | PASS | `IEEEtran` journal class, no overrides; letter size |
| 2 | Regular paper length 10 pages; up to +6 allowed at submission | S1, S2 | PASS | 10 pages including references and biographies |
| 3 | Abstract 150-250 words, one paragraph, no abbreviations, footnotes, references, equations | S1, S2 | PASS | 248 words; none of the excluded items |
| 4 | Three or four keywords | S1, S2 | PASS | Four index terms |
| 5 | Author list under title without e-mail or affiliations | S1 | PASS | Names only under the title; affiliations and e-mails in the first-page footnote, as in the IEEE template |
| 6 | Institutional e-mail of all co-authors | S1 | PASS | Both addresses are NUST; the corresponding author confirmed both are working and in use |
| 7 | No non-English characters in author information | S1 | PASS | ASCII only |
| 8 | Conflict-of-interest disclosure | S1 | PASS (authors confirmed: no conflict) | Section "Conflict of Interest" |
| 9 | Short biographies for all authors, at the end (regular papers) | S1, S2 | PASS | Two text-only biographies (supplied by the authors) |
| 10 | Author photographs | S1 (final files only) | NOT APPLICABLE now | Requested with final files; removed from the initial manuscript |
| 11 | Dates on the first page | S1, S2 | PASS | T-IV states no requirement; none printed |
| 12 | Cover letter | S1 | NOT APPLICABLE | Only a response letter is required, and only for resubmissions |
| 13 | AI-use disclosure in the acknowledgments: system, sections, level of use | S3 | PASS (authors must confirm accuracy) | Acknowledgment names Anthropic's Claude, the sections (all text), code, experiments and figures |
| 14 | Review model | S1 | PASS | Single-blind; author names stay on the manuscript |
| 15 | Format self-check completed | S1 | NEEDS CONFIRMATION IN PORTAL | Evidence in `03_checklist/`; tick the boxes in the portal |
| 16 | Figures: vector preferred, PDF accepted, fonts embedded, single-column width 3.5 in | S4 | PASS | 3 matplotlib figures (vector PDF, 3.45 in wide, TrueType fonts) plus one native LaTeX/TikZ figure; no Type 3 fonts anywhere in the PDF |
| 17 | Figure text "about 9-10 point at full size" | S4 (guidance) | DEVIATION, accepted | Figure text is 6.3-7.5 pt. It was raised from 5-7 pt and checked at print size; reaching 9-10 pt would need larger or fewer panels and more pages. Guidance, not a stated rejection criterion |
| 18 | Figures readable in greyscale (IEEE accessibility advice) | S4 | PASS | Hatches, markers and line styles accompany every colour |
| 19 | Figures and tables embedded vs separate files | S1, S4 | PASS | Figures are embedded; S1 asks for separate figure files only with final files and only "if they aren't embedded" |
| 20 | References formatted by IEEEtran, all cited, none missing | S1 | PASS | 65 references; 73 bibliography entries, 8 uncited (not printed); 0 cited-but-missing |
| 21 | Reference identifiers correct | n/a (integrity) | PASS | 39/39 arXiv ids and 7/7 DOIs match; 22 more matched on Crossref; 5 conference papers checked on proceedings pages (`paper/refs_status.md`) |
| 22 | No placeholders, comments, tracked changes, broken links | n/a | PASS | 0 `\todo`, 0 "[pending]"; PDF title, author and keyword metadata set |
| 23 | Cross-references and numbers consistent | n/a | PASS | No undefined references; all numbers generated from result files; fresh-clone rebuild identical; abstract and conclusion ranges match the results |
| 24 | Portal-side fields (ORCID for the co-author, funding, contributions, cover-letter field) | S5 | NEEDS CONFIRMATION | Portal content not readable here; the T-IV pages do not mention ORCID, funding or contribution statements. Both authors' ORCIDs (0009-0003-4461-8005; 0009-0002-7006-386X, checksum-valid) are in the manuscript |
| 25 | Code archive consistent with the manuscript | n/a (integrity) | PASS | Zenodo v3 (10.5281/zenodo.23693027, commit 9a9cf500e7, two creators) is published; the cited concept DOI 10.5281/zenodo.23054412 resolves to it |

## C. Minimal-change log (this audit round)
**Mandatory for compliance**
1. Author photograph removed from the initial manuscript (both biographies text-only): T-IV asks for photos with final files only. `main.tex` biography block; original line kept commented.
2. AI-use disclosure rewritten to the three elements IEEE requires (system, sections, level). `main.tex`, Acknowledgment.
3. Figure fonts: Type 3 DejaVu replaced by embedded TrueType (Arial, DejaVu for math glyphs); text enlarged from 5-7 pt to 6.3-7.5 pt; "nominal" line moved into the Fig. 2 legend; two-line labels in Fig. 3. `paper/make_results_rev1.py` (data unchanged).
4. PDF metadata (title, author, keywords) set via hyperref. `main.tex`.

**Needed for internal consistency (no result changed)**
5. "silent / invisible" replaced by "cannot be measured without labels, although the drift of the predicted scale warns of it": it contradicted the label-free monitor result. Discussion; Fig. 1 caption and box.
6. Fig. 3 intervals changed from agent-level Wilson to scene-cluster bootstrap (`results/rev1/fig3_cluster_ci.json`); caption updated. Findings unchanged (right turns in Singapore 0.30 [0.25, 0.38] against 0.57 [0.49, 0.66] in Boston).
7. Setup: one sentence stating when the source split-conformal guarantee applies (AV2: one score per independent clip) and when it does not (nuScenes, Waymo), with the same-domain control.
8. Results (Audit): the 750 / 1000 budgets explained. Methods: Remark 2 (scene-averaged risk versus agent-weighted miss rate) and a pointer to the betting regression test.
9. "explain" changed to "reproduce" for the input-injection results (abstract, results, discussion): injection is an intervention, not a causal decomposition.
10. Related Work and the Wayformer subsection condensed to stay at 10 pages. No result, number or claim was removed.

**Author list (authors' instruction, 2026-10-27)**: third author (Hina Maryam) removed from the author line, footnotes and biographies; the two biographies were replaced verbatim by the text the authors supplied; running head changed to "Saleem and Saleem"; PDF author metadata updated. Note: the public Zenodo records (v1, v2) still list three creators; a new version is needed to correct them (D2).

**Optional (not done)**: American "organization" in the footnotes against British spelling elsewhere; journal-name abbreviations in references.

## D. Upload inventory and repository
**D1. Files** (portal category: to be confirmed in the portal)
| Filename | Purpose | Mandatory at initial submission |
|---|---|---|
| `01_manuscript/Saleem_ConformalCoverageTransfer_TIV_manuscript.pdf` | Manuscript | Yes |
| `02_source/Saleem_ConformalCoverageTransfer_TIV_source.zip` | LaTeX source, test-compiled | No (final files) |
| `03_checklist/TIV_format_self_check_completed.md` | Evidence for the portal checklist | Checklist yes (ticked in portal) |

**D2. Zenodo / GitHub (not required by T-IV; no DOI deposit is needed for submission). STATUS 2026-10-27: v3 published (10.5281/zenodo.23693027); text below is the plan that was followed.** The manuscript cites the code archive, so the cited record
should match the paper. Nothing has been published by this package. If you want that: (1) provide a Zenodo token; (2) I confirm the target is the
existing record 23054412 (new version, public, CC BY 4.0, no raw third-party data, single-blind compatible since authors are named); (3) I build
the archive from the final commit, publish the new version after your go-ahead, and update the DOI line only if the concept DOI changes.
Alternative without a token: remove the archive sentence and cite GitHub only. The token is never written to any file.

## E. Items only the authors can resolve
1. **COI: confirmed by the authors, no conflict.** Wording in the manuscript: "The authors declare no competing financial interests or personal relationships that could have influenced this work."
2. **AI-use disclosure (as written by the authors, with the drafting of all text added because it is a fact):** Claude assisted with code development, debugging, data analysis, experiment execution, and the drafting and editing of the manuscript text (all sections); the authors directed the research, reviewed the outputs and take full responsibility. Please confirm it is accurate.
3. **Affiliations:** first-page footnotes now give the NUST affiliations (SEECS; NUST Business School) and the no-funding sentence, as the authors supplied. The biographies (also author-supplied) say the authors are currently with a public-sector / private-sector organization; that wording is accepted as supplied, but the footnote affiliation and the biography employer differ, which an editor may query.
4. **Funding: confirmed by the authors, none.** The Acknowledgment now begins "This research received no external funding."
5. **Portal:** enter author details so they match page 1; ORCIDs where the portal asks; tick the self-check boxes.

## F. Scientific concerns (separate from formatting; none blocks submission)
Single seed for the reverse direction and the multi-dataset model; two architectures; 138 nuScenes validation scenes (repair uses training-split labels,
stated with its exchangeability assumption); no planner-level evaluation; the monitor result has 24 non-independent pairs and one blind spot (all stated).

## G. Stop / go
**GO after items E1-E3** (and E4 if applicable). No mandatory formatting item is unresolved. Not verified: portal-side fields (row 24) and
whether the portal requires a source file at initial submission.
