# Paper 01 -- Revision plan (R1), written 2026-09-30

Source: the Q1-style review of the submission draft (same date). Goal: move from "Major revision" risk to
"Minor revision" quality before submitting to IEEE T-IV. Every step below has a **done-when** check and a
crash-resume marker, so any session can pick up from the last completed step.

Status markers live in `results/rev1/<step>.ok` (created only when the step's done-when check passes).
Resume rule: find the lowest-numbered step without a `.ok` marker and continue from there.

---

## 0. Root cause found before planning (2026-09-30)

The review's #1 risk (scene-level repair rate 0.80-0.86 < 1-delta = 0.90) is **a code defect, not noise**:

- `src/solutions.py::wsr_pvalue` caps the bet at `lam <= 0.5/alpha = 5`.
- It tests H0: E[L] >= alpha with payoff (alpha - L), L in [0,1], so capital stays non-negative only if
  `lam <= 1/(1-alpha) = 1.11`.
- Diagnostic (scratchpad `diag_bet.py`, av2_gpu_full): negative capital occurred in thousands of grid tests, and
  most of the resulting rejections were made while capital was negative:
  - Waymo m=40: 6346; m=60: 8842.
  - nuScenes m=40: 2685; m=60: 5647.
- Ville's inequality no longer applies, so the returned thresholds are not guaranteed.
- Why tests missed it: the synthetic test used scenes with many agents, so per-scene losses stayed small.

**Consequence:**
- Every betting-LTT number in the paper (exploratory, confirmatory, Waymo) must be recomputed.
- HB-LTT, agent-level C-or-R, direct SCP and all coverage-gap numbers are unaffected (different code paths).
- The published Zenodo record (10.5281/zenodo.23054413, v1) contains the defective function. It cannot be
  deleted, so v2 must be published with a changelog.

Second, smaller evaluation issue (nuScenes only): labels are drawn from the same 138 scenes whose *remainder* is
the evaluation set. At m=60 that is 43% of the pool, so the held-out mean is negatively dependent on the labelled
mean. Fix: disjoint label and evaluation pools (step 1.4).

---

## Phase 1 -- Correctness (P1). CPU mostly. ~1 working day. Blocks all writing.

| Step | Action | Done when | Effort |
|---|---|---|---|
| 1.1 | **Register Addendum D** in `notes/falsification.md` BEFORE any rerun: declare the defect, the fix (`lam <= c/(1-alpha)` with **c = 0.5**, fixed now, not tuned), the new evaluation design (1.4), the success criterion (below), and that the old H16/H16b/H19 outcome logs stay unedited and are superseded. | Addendum committed; commit hash noted here | S |
| 1.2 | Fix `wsr_pvalue`: cap `lam` at `0.5/(1-alpha)`; add `assert k >= 0`. Keep predictability (bet uses past losses only). | Code committed | S |
| 1.3 | Regression tests in `src/tests/test_solutions_synthetic.py`: (a) adversarial scenes with L in {0,1} (singleton scenes) must never give negative capital; (b) exact-superpopulation validity: draw m scenes i.i.d. from a known scene distribution, compute the true R(q) analytically, require P[R(q_hat) <= alpha] >= 1-delta - 2 SE over 2000 draws for m in {20,40,60,100}; (c) old tests still pass. | All tests pass; a deliberately re-broken cap makes test (a) fail (mutation check) | M |
| 1.4 | **Disjoint pools.** Forward AV2->nuScenes: label pool = nuScenes *train* scenes (~700 scenes, never seen by the AV2 model); evaluation pool = all 138 nuScenes val scenes. Waymo: label pool = `wmcal` (4331), evaluation = `wmtest` (4318). Reverse ns->AV2: AV2 clips are single-agent, so the scene rule reduces to agent level; label pool = AV2 cal half, evaluation = AV2 test half. | Pools defined in `src/analyze_rev1.py`; no scene appears in both pools (asserted) | S |
| 1.5 | GPU predictions: av2_gpu_full on nuScenes train (11 chunks, ~32k agents, ~10-15 min). Run **before** Phase 2 claims the GPU. | `results/preds2/av2_gpu_full/nstrain.npz` exists, row count matches | S |
| 1.6 | Rerun the scene-repair analysis with corrected code and disjoint pools for av2_gpu_full (nuScenes, Waymo) and ns_gpu_full (Waymo), plus the 3 exploratory models for completeness. 1000 draws (was 200), m in {20,30,40,60,80,100}. Metric: **R(q_hat) computed on the full evaluation pool**, reported as P[R <= alpha] with a Wilson 95% CI, and median area vs oracle. | `results/rev1/scene_*.json` for every model/target | M |
| 1.7 | Log outcomes in `notes/falsification.md` (append-only), including whichever way it lands. | Outcome entry committed | S |

**Pre-registered success criterion (fixed in 1.1):**
- Validity: Wilson lower bound of P[R <= alpha] >= 0.85, and point estimate >= 0.88, for every m tested.
- Recommended budget: the smallest m whose median area <= 2.0x oracle. This is reported, not chosen by eye.
- If validity fails after the fix, the paper drops the betting variant and reports HB-LTT, which is valid but
  wider.

**Rollback:** if 1.3(b) fails, stop. Do not write any repair claim until the p-value passes the exact test.

---

## Phase 2 -- Second architecture (P1). GPU, ~2-2.5 days unattended. Start right after 1.5.

| Step | Action | Done when | Effort |
|---|---|---|---|
| 2.1 | **Check first for official pretrained UniTraj checkpoints** (Wayformer or MTR, AV2-trained). If one exists and its reported accuracy is documented, use it: zero training cost and no resource objection possible. Verify licence and provenance before use. | Decision recorded here | S |
| 2.2 | If none: register **Addendum E** (Wayformer confirmatory: same H1/H17/H18 plus the corrected scene-repair tests; gate = val minADE6 <= 0.98, i.e. within 15% of UniTraj's 0.85). Also fix the decision rule if the gate is missed (2.6). | Committed before training starts | S |
| 2.3 | Benchmark Wayformer on the P2000 (`--profile_steps 50 --method wayformer`): s/step and peak memory at batch 16 and 32. If OOM, use batch 16 x accum 8 (effective 128, same as the AutoBot recipe). | Numbers recorded | S |
| 2.4 | Choose epochs E so the run finishes within ~60 h with a **complete** one-cycle schedule (set `max_epochs=E`; never stop a one-cycle run early). Write `run/train_av2_wayformer.cmd` (same watchdog/marker pattern as `train_av2_gpu_full.cmd`) and install the watchdog. | Watchdog running; first checkpoint saved | S |
| 2.5 | Post-processing watchdog (`run/post_av2_wayformer.cmd`): predict av2cal / av2test / ns / nstrain / wmcal / wmtest; run run2, addB, addC and the corrected scene repair. Injections need velocity/acceleration recomputed for Wayformer (it reads all object features); implement `recompute_kinematics` in `shift_inject.py` and test it on AutoBot, where results must match the existing ones exactly. **Channel layout confirmed 2026-09-30** from `code/UniTraj/unitraj/datasets/base_dataset.py:560-577`: final `obj_trajs` (39 ch for Wayformer) = [0:6] pos+dims, [6:11] type one-hot, [11:33] time embedding, **[33:35] heading (sin,cos)**, **[35:37] velocity (vx,vy)**, **[37:39] acceleration**, computed from the PRE-injection track. `recompute_kinematics(inp)`: after `pos[...,:2]` is modified by hz2/jitter, recompute heading via `common_utils.get_heading` (atan2 of consecutive valid positions), velocity via finite difference of position / 0.1s, acceleration via finite difference of the recomputed velocity / 0.1s (copy frame 0 from frame 1, matching `base_dataset.py`'s own convention) -- masked positions unchanged. No-op for AutoBot (it never reads channels 6+), so the "must match exactly" check is a strict regression test, not a tolerance one. | `results/post/av2_wayformer/POST_DONE` | M |
| 2.6 | Gate decision (registered in 2.2): if val minADE6 > 0.98 at the end, extend once by +50% epochs as a fresh complete schedule from the last weights. If still missed, report it as a second architecture with its accuracy stated, labelled exploratory, and never with resource wording. | Decision logged | S |

---

## Phase 3 -- New analyses requested by the review (P2). CPU, parallel with Phase 2. ~1 day.

Register all of these in Addendum D (step 1.1) before running. Mark each as confirmatory or exploratory there.

| Step | Analysis | Why | Done when |
|---|---|---|---|
| 3.1 | **[WITHDRAWN 2026-10-08, premise false -- see falsification.md Addendum H-0]** History-length confound: inject "AV2 history truncated to 1.1 s" (mask frames older than 1.1 s, exactly what UniTraj does to Waymo) into AV2 test inputs; measure the coverage lost as a share of the Waymo gap. Exact for AutoBot (position-only). | Waymo's 1.1 s history vs UniTraj's 2.1 s input: a reviewer will attribute the Waymo gap to padding | `inject_hist11_*.json` |
| 3.2 | **Hierarchical CP baseline** (Dunn et al. 2023): (a) subsample one agent per labelled scene and run split CP; (b) the scene-pooled quantile. Add it to the scene-repair comparison. | Cited as the natural competitor but never run | Column in the scene-repair results |
| 3.3 | **Mechanism: source overconfidence.** For all models, report q_src, q_oracle (metres) and their ratio against minADE6; plus score CDFs source vs target. | Explains "gap grows with accuracy" with evidence, not speculation | `results/rev1/mechanism.json` + Fig. D/E data |
| 3.4 | **T-IV relevance metric:** region radius in metres (source vs repaired); share of agents whose true path leaves the region, by manoeuvre; share of turning agents at intersections missed. | Reviewers at an application journal will ask what a 15-point gap means for a planner | Numbers in a macro file |
| 3.5 | Wilson 95% CIs on every rate in the paper; a multiplicity note (list of registered tests, Holm-adjusted p-values where p-values exist). | Missing statistics | Generators updated |
| 3.6 | Optional, if time allows: Barber et al. (2023) bound as a gap predictor (estimate the TV distance with the domain classifier). Skip without loss if Phase 2 runs late. | Strengthens the shift-theory link | -- |

---

## Phase 4 -- Manuscript rewrite (P1/P2). ~1.5 days. Starts when Phase 1 is done; Phase 2 numbers slot in last.

**4.1 Structure**
- Results headings become Q1-Q4:
  - Q1: How large is the gap? (3 datasets, both directions, 2 architectures)
  - Q2: Does reweighting fix it?
  - Q3: What does label-free repair achieve?
  - Q4: How many labels to audit and to repair?
- Confirmatory results first. The exploratory zoo moves to one short "Exploratory analyses" subsection.
- Detailed exploratory tables go to a **Supplementary Material** PDF (T-IV allows it), along with a
  hypothesis-code-to-section map.
- Add a Conclusion section.

**4.2 Wording (must all be gone; checked with grep in 5.2)**
- Remove "CPU", "shared CPU", "for time", "time available", "dropped".
- The exploratory zoo becomes "reduced-budget models (training-data subsets, shorter schedules)".
- "State-of-the-art" becomes "matches UniTraj's published AutoBot baseline".
- Hedge the accuracy-gap claim (n models, one or two families).
- Fix: the 98% sentence, the Waymo split count, and the four-vs-five count.
- Remove the defensive and self-congratulatory sentences (review section 2, items 7-9, 11-12).
- H-codes in the running text become descriptive names.

**4.3 Figures and tables**
- Add: Fig. A (pipeline block diagram, end of Intro), Fig. C (gap bar chart with scene-clustered CIs), Fig. D
  (score CDFs), Fig. E (accuracy vs gap scatter), Fig. G (rate/area vs m, replacing both scene tables), and
  Algorithm 1 (scene-level certify-or-recalibrate).
- Merge 8 tables into 3:
  - main transfer table: 2-3 confirmatory models x targets, with reweighting and clustering columns;
  - injection table;
  - supplementary tables.
- Redo the label-budget figure on the confirmatory model, with distinct markers and line styles (greyscale-safe).
- Qualitative BEV figure (F) only if pages allow.

**4.4 Methods**
- State the corrected betting cap and why (non-negative capital).
- Add a one-line proof note to Proposition 3: validity needs predictable bets in [0, 1/(1-alpha)).
- Disclose the defect in "Pre-registration and disclosures": it was found in review before submission, and every
  affected number was recomputed.
- Pre-registration wording becomes "registered in a version-controlled, timestamped log (repository and Zenodo)".
  It must not imply a public registry.

**4.5 Front matter**
- Remove the `\markboth` "[pending]" text.
- 4 index terms.
- Move the AI disclosure into an Acknowledgment.
- Affiliation wording: "was with NUST when this work was done; is now with ...". Needs the authors to confirm.
- Photos for R. Z. Saleem and H. Maryam.
- Confirm the student e-mails are active.

**4.6 Style pass**
- Apply `writting_guidelines.txt` to every rewritten paragraph (and only those; do not churn clear text).

---

## Phase 5 -- Verification and release. ~0.5 day.

| Step | Check | Pass condition |
|---|---|---|
| 5.1 | `powershell -File paper/build.ps1` | Compiles clean, 0 `\todo`, 0 undefined references |
| 5.2 | Grep audit of `paper/sections/*.tex` for banned words (CPU, "for time", "state-of-the-art", H[0-9]+ in running text, "---" density) | 0 hits for banned words |
| 5.3 | **Number audit:** a script lists every numeral in `sections/*.tex` that is not inside a macro, a citation, an equation constant or a registered threshold; each must be justified or turned into a macro | List empty or every item justified |
| 5.4 | Page count | <= 10 pages, or the authors approve 11 ($175) |
| 5.5 | Abstract | 150-250 words, no abbreviations, no references |
| 5.6 | All tests (`src/tests/*.py`) | Pass |
| 5.7 | Fresh-clone reproduction: clone the repo into a temp folder, run `make_results.py` from the committed result JSONs | Identical `generated/*.tex` |
| 5.8 | GitHub push (needs the user's go-ahead); **new Zenodo version (v2)** with corrected code and a changelog naming the defect; cite the **concept DOI** (always resolves to the latest version) | DOI resolves to v2 |
| 5.9 | Refresh `submission_IEEE-TIV/` (PDF, source, photos, supplementary) and `CHECKLIST.md` | Checklist all ticked |

---

## Schedule (critical path = Phase 2 GPU training)

| Day | GPU (unattended) | Active work |
|---|---|---|
| 1 | nuScenes-train predictions (1.5), then Wayformer benchmark and launch (2.1-2.4) | Phase 1 (1.1-1.7) |
| 2 | Wayformer training | Phase 3 (3.1-3.5); start Phase 4 structure and figures from Phase 1 and 3 numbers |
| 3 | Wayformer training | Phase 4 writing |
| 4 | Wayformer finishes; post-processing (2.5) | Slot in Wayformer numbers; Phase 5 |
| 5 | -- | Buffer (gate extension 2.6, or interruptions) |

**Total: about 4-5 calendar days**, with about 3 days of active work.

## Needs the authors (cannot be done by Claude)
1. Photos of R. Z. Saleem and H. Maryam.
2. Confirm the current-affiliation wording and that the student e-mails are active.
3. Approve: GitHub push, the Zenodo v2 publish, and 11 pages if needed.
4. Keep the PC on and awake for the Wayformer run.

## Contingencies
| Risk | Response |
|---|---|
| Corrected betting LTT is valid but wider, so the recommended budget moves up | Report the registered rule's answer. The paper's claim becomes whatever the data say. |
| Corrected betting LTT still fails the exact synthetic test | Stop, re-derive, and report HB-LTT only |
| Wayformer out of memory | Batch 16 x accum 8 |
| Wayformer misses the gate | Rule 2.6. Never with resource wording. |
| Machine interrupted | Watchdogs plus `.ok` markers; resume from the lowest missing marker |
| Added figures push the paper past 10 pages | Move the exploratory material and the qualitative figure to Supplementary first |

## Progress log
- 2026-10-02: Phase 1 done (1.1-1.7). Literature expanded to 71 verified refs (42 new, 2024+).
- 2026-10-02: Phase 4 structure done early to fit 10 pages: Results reorganised as Q1-Q4, confirmatory first;
  8 tables -> 2 (compact model table, injection table); new figures: Fig.1 TikZ pipeline, Fig.2 gap bars (4
  transfers, scene-clustered CIs), Fig.3 manoeuvre coverage by city (confirmatory model), Fig.4 corrected scene repair
  (rev1, disjoint pools); Conclusion added; resource wording removed; AI disclosure moved to Acknowledgment; 4
  keywords; all scene-repair claims corrected to the rev1 numbers (~150-200 scenes). 9 pages.
- Still open: Wayformer (Phase 2, training), Phase 3 analyses (history-length confound, mechanism), Zenodo v2 with
  corrected code (cite concept DOI), GitHub push of these commits (needs go-ahead).

## Review-item 3 (run the optimal-transport label-free baseline) -- NOT DONE, with reason -- 2026-10-08
Read Correia & Louizos (arXiv:2507.10425, ~96% of the text) before attempting a faithful implementation. Their label-free
step brackets the unlabelled test-score distribution with s#Q-down / s#Q-up built from the min and max score over ALL
candidate labels per instance (or model-predicted vs uniform label sampling). Their scores are bounded in [0,1]; the
only non-classification experiment is a 4-D synthetic regression handled by regression-as-classification with 50
bins of a 1-D output; the paper states that for regression "designing s#Q-down and s#Q-up is more challenging" and
gives no construction for high-dimensional or structured outputs. Our score is a metre-scale min-over-modes,
max-over-horizon distance over a 120-dim multi-modal output: min over candidate trajectories is 0 and max is
unbounded, so the published construction is undefined here. Any trajectory version would be my own new construction,
and presenting it as "the NeurIPS method" would be the simplified-reproduction mistake to avoid. Decision: not run;
state exactly this in Related Work; list a trajectory-specific bracketing as future work.
