
### 2026-09-21 16:36 — Run 1 outcome
Run 1 result (AV2 -> nuScenes, av2_cpu_v1 ep08, zero labels): coverage 86.6% vs 90% nominal (gap +3.4 pt, CI 2.7-4.1), same sign at alpha .05/.10/.20; sampling-rate control leaves it unchanged. H2 refuted (adj R2 .147, reweighting -57%), H3 not supported. Decision: paper cannot rest on 'covariate reweighting repairs it'. Need (i) diagnosis of why, (ii) a solution that actually works, (iii) stronger base model + more pairs.

### 2026-09-21 16:36 — flaw disclosed
Found while designing the solution: shift_factors 5-7 (ego_speed_mean, turning, curvature) are computed from the GROUND-TRUTH FUTURE, so Run 1's H3 was not label-free. Fixed by adding label-free history-based factors (shift_factors.compute_factors_lf; hist_speed correlates r=.81 with GT speed, confirming the tensor indices). Disclosed in falsification.md Addendum A0 BEFORE re-running.

### 2026-09-21 16:36 — bugs
Two silent bugs found by reading logs, not by tests: (1) train_autobot --lr/--lr_sched wrote only cfg.method.* while UniTraj's optimizer reads the top-level key (log kept printing lr 7.5e-4) -> earlier runs all used the default; fixed. (2) predict --n is ignored for validation datasets (UniTraj applies max_data_num only in training mode) -> added --limit_batches.

### 2026-09-21 16:36 — compute decision
Compute reality: P2000 shared; the user's AI2 project (Paper 02 detection cache) holds ~4 GB of the 5 GB card, likely for days-weeks. Plan therefore uses CPU (6 samples/s) for training and treats GPU as opportunistic. Strong-model path chosen: fine-tune av2_cpu_v1 with LR decay (v1 never decayed; train ADE still falling) rather than a from-scratch 100-epoch run that only fits a Kaggle/Colab GPU. Model-zoo robustness (H8) partly substitutes for one very strong model.

### 2026-09-21 16:36 — solution design
Solution design (post-hoc after Run 1, pre-registered before Run 2): identifiability proposition explains WHY label-free covariate methods cannot fix conditional shift; H4 = normalise scores by the model's own predicted Laplace scale (label-free; Spearman(pred scale, actual score)=0.54 on 192 samples); H5/H5b = label budget + binomial coverage audit; H6 = label-free shift monitor across AV2 cities; H7 city-level coverage. Synthetic tests with known truth: H4 fixes visible shift only (honest negative control), direct estimator needs ~1000 labels for +/-2 pt, pooled/shrink biased, audit matches exact binomial theory.
