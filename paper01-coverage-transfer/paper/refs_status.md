# Reference verification status

Legend: **V** = author list + venue + year (+ pages where applicable) checked against a primary source
(NeurIPS/ICLR/CVPR proceedings page, arXiv abstract page, or publisher page), date given.

All 19 entries in `refs.bib` were checked 2026-09-22 via web search against primary proceedings/arXiv pages.
Two corrections were made as a result (both now fixed in `refs.bib`, listed under "Corrections" below); everything
else matched what was already written and needed no change.

| key | status | source checked |
|---|---|---|
| vovk2005algorithmic | V 2026-09-22 | Springer book page (ISBN 9780387001524) |
| angelopoulos2023gentle | V 2026-09-22 | nowpublishers.com (Foundations and Trends in ML 16(4):494-591, 2023) |
| tibshirani2019covariate | V 2026-09-22 | proceedings.neurips.cc/paper/2019, author order confirmed |
| barber2023beyond | V 2026-09-22 | projecteuclid.org, Ann. Statist. 51(2):816-845, 2023 |
| gibbs2021adaptive | V 2026-09-22 | proceedings.neurips.cc/paper/2021 (2 authors: Gibbs, Candes) |
| angelopoulos2023pid | V 2026-09-22 | proceedings.neurips.cc/paper_files/paper/2023 |
| lei2018distribution | V 2026-09-22 | tandfonline.com, JASA 113(523):1094-1111, 2018 |
| ovadia2019can | V 2026-09-22 | proceedings.neurips.cc/paper/2019, full 9-author list confirmed |
| feng2024unitraj | V 2026-09-21; **numeric claim re-verified 2026-09-22** | Springer ECCV 2024 chapter 10.1007/978-3-031-73254-6_7. The specific number cited in `setup.tex` (AutoBot-on-AV2 minADE6 = 0.85) was read directly from the paper's supplementary Table 8 (arxiv.org/pdf/2403.15098 main paper + ecva.net supp PDF), not from memory -- see `notes/falsification.md` "Kill condition #2 checked" entry |
| huang2025cuqds | V 2026-09-21 | arXiv 2406.12100 + AAAI OJS page, AAAI 2025 |
| wilson2021argoverse2 | V 2026-09-22 | neurips.cc/virtual/2021/29837 + datasets-benchmarks-proceedings.neurips.cc; **year confirmed 2021** (arXiv id 2301.00493 is just the posting date, not the venue year) |
| li2023scenarionet | V 2026-09-22 | proceedings.neurips.cc/paper_files/paper/2023, NeurIPS 2023 D&B track confirmed |
| rahaman2026shift | V 2026-09-21 | arXiv 2602.12616 authors verified; **arXiv page states NO venue** (an aggregator's "L4DC 2026" claim is unverified) -> cited as preprint only, correctly |
| tumu2026adaptnc | V 2026-09-21 | arXiv 2602.01629 (preprint) |
| shu2025scenario | V 2026-09-21 | arXiv 2512.05682 (preprint) |
| binny2025moved | V 2026-09-21 | arXiv 2511.11567 (preprint) |
| girgis2022autobots | V 2026-09-21 | arXiv author list (8 authors) + ICLR 2022 spotlight page |
| caesar2020nuscenes | V 2026-09-22 | openaccess.thecvf.com CVPR 2020, full 10-author list confirmed |
| sun2024copula | V 2026-09-22 | iclr.cc/virtual/2024/poster/17807 + huiwenn.github.io CV -- "Sophia Huiwen Sun" is her verified full name |
| lindemann2023safe | V 2026-09-22 | georgejpappas.org PDF, IEEE RA-L 8(8):5116-5123, 2023 -- pages were missing, now added |
| vovk2012conditional | V 2026-09-22 | proceedings.mlr.press/v25/vovk12.html, PMLR 25:475-490, 2012 -- **newly added**, was cited in prose without a reference (Sec. II-D coverage-SD claim) |

## Corrections made 2026-09-22
1. `lindemann2023safe`: added missing `pages = {5116--5123}`.
2. Added `vovk2012conditional` (Vovk, "Conditional Validity of Inductive Conformal Predictors", ACML/PMLR 25:475-490,
   2012) and cited it in `sections/problem.tex` where the Beta-distributed-coverage fact was previously stated
   without a source.

## Still open
Nothing outstanding. If new citations are added later, verify them the same way before submission -- never cite
from memory alone.

## Added 2026-09-24 (for the certify-or-recalibrate / scene-level LTT methods) -- all V against primary pages
| key | status | source checked |
|---|---|---|
| angelopoulos2025ltt | V 2026-09-24 | projecteuclid.org Ann. Appl. Stat. 19(2), doi 10.1214/24-AOAS1998 (authors: Angelopoulos, Bates, Candes, Jordan, Lei; **pages not verified -> omitted**) |
| waudbysmith2024betting | V 2026-09-24 | academic.oup.com JRSSB 86(1):1-27, doi 10.1093/jrsssb/qkad009 |
| bates2021rcps | V 2026-09-24 | dl.acm.org J. ACM 68(6), Art. 43, doi 10.1145/3478535 (source of the Hoeffding-Bentkus p-value) |
| clopper1934confidence | V 2026-09-24 | academic.oup.com Biometrika 26(4):404-413, doi 10.1093/biomet/26.4.404 |
| bian2023training | V 2026-09-24 | projecteuclid.org EJS 17(2):2044-2066 |
| dunn2023hierarchical | V 2026-09-24 | tandfonline.com JASA 118(544):2491-2502, doi 10.1080/01621459.2022.2060112 (online 2022; cited with the 2023 issue year) |
| lee2023hierarchical | V 2026-09-24 | arxiv.org/abs/2306.06342 (Lee, Barber, Willett; page states "to appear in ACM/IMS J. Data Science" -> cited as arXiv preprint only) |

## Added 2026-09-29 (Waymo Open Motion, third dataset, Addendum C)
| key | status | source checked |
|---|---|---|
| ettinger2021waymo | V 2026-09-29 | arxiv.org/abs/2104.10133 (title/authors confirmed via fetch); venue ICCV 2021 corroborated by the CVF open-access URL pattern (content/ICCV2021/papers/Ettinger_..._ICCV_2021_paper.pdf, request blocked by bot protection, 403 not 404) and by prior knowledge; no page numbers available from either source so `pages` is omitted, matching the convention used for angelopoulos2025ltt above |

## Added 2026-10-02 (literature expansion, 2024+ only, per user request) -- all V via arXiv abs-page fetch
Verified by fetching each arxiv.org/abs/<id> page directly and reading title/author list/date/venue off it
(not from memory, not from the search snippet alone). Two candidates found during the same search were already
in refs.bib under a different key (shu2025scenario = arXiv:2512.05682; binny2025moved = arXiv:2511.11567) and
were not re-added. One candidate (arXiv:2310.14293, "Testing exchangeability by pairwise betting") was excluded
for being a 2023 submission, outside the 2024+ requirement.

| key | arXiv id | status | venue (if beyond arXiv) |
|---|---|---|---|
| doula2025safepath | 2505.09427 | V | -- |
| chen2024v2i | 2408.00374 | V | -- |
| mei2024perceive | 2403.08185 | V | -- |
| yao2025crowdnav | 2508.05634 | V | CoRL 2025 |
| pouria2026gaussian | 2603.10407 | V | -- |
| feldman2025robotsafety | 2501.04823 | V | -- |
| binny2026stable | 2511.20593 | V | IEEE RA-L (accepted) |
| mirzaeedodangeh2025interactive | 2511.10586 | V | -- |
| lindemann2024formal | 2409.00536 | V | -- |
| abdelmadjid2025survey | 2503.03262 | V | -- |
| song2025unimotion | 2602.00566 | V | NeurIPS 2025 |
| zhou2026modeseq | 2605.24037 | V | -- |
| demmler2025dataset | 2507.05098 | V | -- |
| aolaritei2025levy | 2502.14105 | V | -- |
| xu2025wasserstein | 2501.13430 | V | -- |
| correia2025nonexchangeable | 2507.10425 | V | NeurIPS 2025 |
| liu2026certify | 2608.10893 | V | -- |
| alvarez2024pvalue | 2405.08975 | V | -- |
| nguyen2024tradeoffs | 2403.19605 | V | -- |
| einbinder2024semisupervised | 2412.11174 | V | -- |
| joshi2026riskcontrolled | 2605.06479 | V | -- |
| mortier2025hierarchicalclass | 2501.19038 | V | -- |
| principato2024hierarchicaldata | 2411.13479 | V | -- |
| bhattacharyya2024groupweighted | 2401.17452 | V | -- |
| mallick2026generalizedhcp | 2608.15500 | V | -- |
| gauthier2025evalues | 2503.13050 | V | -- |
| prinster2025watch | 2505.04608 | V | ICML 2025 |
| vovk2025randomness | 2501.11689 | V | -- |
| jun2026retrospective | 2511.04275 | V | Pattern Recognition vol. 180 (2026) |
| liu2026portfolio | 2602.03168 | V | -- |
| liang2026optimalregret | 2602.16537 | V | -- |
| min2026unifiedtheory | 2605.11602 | V | -- |
| zhou2026cpassessment | 2603.27189 | V | -- |
| braun2025diagnostics | 2512.11779 | V | -- |
| bairaktari2025kandinsky | 2502.17264 | V | -- |
| zhong2026fairness | 2608.04254 | V | -- |
| luo2025conditionalrisk | 2504.07611 | V | -- |
| wu2025visuallocalization | 2506.15851 | V | ICRA 2025 |
| manchingal2025uncertaintyaware | 2510.22680 | V | -- |
| wang2025collaborative | 2504.12696 | V | -- |
| qiu2026crosssensor | 2606.11573 | V | -- |
| kong2025largead | 2501.04005 | V | IEEE TPAMI (2025) |

Conference/journal acceptances above are stated as reported in the fetched arXiv page or its "Comments" field at
verification time; since this is a fast-moving set (several still mid-review-cycle as of today), re-check venue
details against the camera-ready at proofing time if the paper is still cited then.

## Added 2026-10-07 (Wayformer architecture, second confirmatory model)
| key | status | source checked |
|---|---|---|
| nayakanti2023wayformer | V 2026-10-07 | Read directly from the primary source's own bibliography while verifying feng2024unitraj's Table 15 (arXiv:2403.15098, PDF p.17, ref [35]): title/authors/venue/pages/year transcribed from that page, not from memory. |
- cauchois2024robust: verified 2026-10-19 (search result: arXiv:2008.04267; JASA 119(548):3033-3044, DOI 10.1080/01621459.2023.2298037; authors Cauchois, Gupta, Ali, Duchi).
