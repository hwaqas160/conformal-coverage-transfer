# TIV Author Portal: entry sheet (copy these fields exactly)

Portal: https://ieee.atyponrex.com/journal/t-iv (redirects to ieee.submission.researchexchange.com/journal/t-iv). Paper type: **Regular paper**.
Everything below is copied from the manuscript so the portal matches page 1.

**Title**
Does the Conformal Guarantee Travel? Measuring, Auditing and Repairing Coverage Loss in Cross-Dataset Trajectory Prediction

**Abstract** (248 words)
A conformal prediction region promises to contain a road user's true future path a fixed fraction of the time. We ask whether that promise survives when a trajectory forecaster is calibrated on one driving dataset and deployed, without new labels, on another. Using forecasters calibrated on Argoverse 2 or nuScenes and evaluated on each other and on Waymo Open Motion, we measure coverage losses of 6 to 25 percentage points at a 90 percent nominal level, on two architectures whose accuracy is within 15 percent of published values; the third dataset was registered in advance. Reweighting by an estimated covariate-shift likelihood ratio makes the loss worse, and we show that no label-free method can repair an unrestricted shift, although the drift of the forecaster's own predicted scale tracks the loss. The loss concentrates in turns, most of all in left-hand traffic. Measured input differences between the datasets reproduce about a quarter of it for one architecture and, with caveats, all of it for the other. Labelled target data can repair it, but labels arrive one scene at a time, and a guarantee that treats agents as independent can fail when scenes contain many agents. A scene-level Learn-then-Test rule with a betting test restores the guarantee at every labelling budget we tested, and returns a region within twice the oracle size from about 150 to 200 labelled scenes. Conformal guarantees for trajectory prediction should therefore not be assumed to transfer across datasets without an audit and a scene-level recalibration.

**Keywords / index terms** (4)
Conformal prediction; trajectory prediction; distribution shift; autonomous driving

**Authors (in this order)**
1. Hassan Waqas Saleem (corresponding author); e-mail hsaleem.msee20seecs@seecs.edu.pk; ORCID 0009-0003-4461-8005; affiliation as in the manuscript footnote: School of Electrical Engineering and Computer Science, National University of Sciences and Technology (NUST), Islamabad 44000, Pakistan
2. Rahat Zeeshan Saleem; e-mail rahat.mscm21nbs@student.nust.edu.pk; ORCID 0009-0002-7006-386X; affiliation as in the manuscript footnote: NUST Business School, National University of Sciences and Technology (NUST), Islamabad 44000, Pakistan

**Conflict of interest** (as in the manuscript)
The authors declare no competing financial interests or personal relationships that could have influenced this work.

**Funding** (as in the manuscript)
This research received no external funding.

**Generative-AI disclosure** (Acknowledgment in the manuscript)
The authors used Anthropic's Claude to assist with code development, debugging, data analysis, experiment execution, and the drafting and editing of the manuscript text. The authors directed the research, reviewed the outputs, and take full responsibility for the accuracy, integrity, and content of this article. This research received no external funding.

**Data and code availability** (as in the manuscript)
Code, configuration files, registered hypotheses, experiment log and aggregate result files: https://github.com/hwaqas160/conformal-coverage-transfer ; archived at Zenodo, concept DOI https://doi.org/10.5281/zenodo.23054412 (version 3: https://doi.org/10.5281/zenodo.23693027). Third-party datasets (Argoverse 2, nuScenes, Waymo Open Motion) are available from their owners under their own licences.

**Portal upload page (as shown in the authors' screenshots)**
- Main Document - LaTeX (required): `1_MAIN_DOCUMENT_LaTeX_required/Saleem_ConformalCoverageTransfer_TIV_main_manuscript_LaTeX.zip`
- Main Document - PDF (required): `2_MAIN_DOCUMENT_PDF_required/Saleem_ConformalCoverageTransfer_TIV_main_manuscript.pdf`
- Conflict of Interest (required): no file; tick "None of the authors have a conflict of interest to disclose"
- Cover letter / Comments (optional): `4_COVER_LETTER_optional/Saleem_cover_letter.pdf`
- Image (optional): the files in `5_IMAGES_optional_figures/` (Fig. 1-5, PDF and 600-dpi PNG)
- Leave empty: Supplementary Material, Tracked Changes, Previously Published (both), LaTeX Supplementary File
