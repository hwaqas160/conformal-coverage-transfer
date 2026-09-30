# RESUME — start here after a crash, a reboot, or a new session

_Keep this file accurate: update it whenever a job is launched/finished or a decision changes. Last full update:
2026-09-24 11:10._

## 00. CURRENT STATE (2026-09-24) — read this first; §0 below is the older (2026-09-23 morning) summary
**Plan changed 2026-09-23:** after a reviewer-objection review, the paper is being strengthened (Addendum B in
`notes/falsification.md`, registered before each run): (1) full-data GPU AutoBot, gate val minADE6 <= 0.98;
(2) Wayformer as 2nd architecture; (3) Waymo as 3rd dataset (**needs the user to accept the Waymo Open licence at
waymo.com/open** — ask again if not done); (4) certify-or-recalibrate repair (H14), scene-level version (H16/H16b), ACI
baseline; (5) scene-clustered statistics (B10), conditional coverage (B7), driving-side mechanism (B11: right-turn
loss concentrated in left-hand-traffic Singapore, supported on 3/3 exploratory models); (6) shift-injection (B4, TODO).
**Jobs:**
| Job | State | Resume |
|---|---|---|
| `P01_TrainAv2GpuFull` (`run\train_av2_gpu_full.cmd`, GPU, full AV2, 60 ep) | **WATCHDOG task: repeats every 10 min, IgnoreNew, idempotent, resumes `results/ckpts/av2_gpu_full/last.ckpt`, stops at DONE/FAILED marker.** Was at epoch 6 (val minADE6 1.241 @ep5, high LR; decays at ep 10/20/30/40/50). ~1.9 s/step observed (5729 steps/epoch) -> ~3 h/epoch unless data-bound issue is fixed. | `powershell -File run\install_watchdog.ps1 -Name P01_TrainAv2GpuFull -Cmd F:\...\run\train_av2_gpu_full.cmd` |
| `ns_cpu_v1` (CPU nuScenes model) | **PAUSED** at epoch 5.x (last.ckpt 09-23 13:46) to leave CPU to the GPU job. Its validation ran on AV2 data by a cache collision (checkpoint RANKING invalid) -> use `last.ckpt` for H9. Will be superseded by a GPU nuScenes model. | `schedule.ps1 -Name P01_TrainNsCpuV1 -Cmd ...\run\train_ns_cpu_v1.cmd` |
**Why watchdog:** the user session is killed from outside (see memory `reference-machine-gotchas`): logoff 09-23 11:09;
09-24 10:50 a FAILED shutdown attempt still terminated all processes incl. the wrapper.
**Data hygiene changes:** nuScenes cal/test are now scene-disjoint (`data/nuscenes_splits/nsscene/{nscal,nstest}`,
salt nsv2scene, 64/74 scenes); UniTraj cache provenance guard (`SOURCE_DB.txt`); training cache on SSD `C:\p01_cache`;
local UniTraj edits captured in `code/patches/`.
**Analyses done on the 3 forward models** (`results/addB/*.json`, outcome log in falsification.md): C-or-R guarantee holds
for i.i.d. labels (.91-.97) but FAILS for whole-scene labels (.59-.71) -> scene-level LTT valid but conservative (HB)
-> H16b betting version 1.39x vs 3.41x area (synthetic). Turn-specific loss (right turn -12..-32 pt).
**Next:** (a) confirm the watchdog restarted GPU training and measure its real step rate; find why 1.9 s/step vs 0.5-0.8
profile; (b) register H16b before running it on confirmatory models; (c) after the GPU AV2 model passes its gate:
predict on av2cal/av2test/ns (+ annotations), run `analyze_run2.py` + `analyze_addB.py`; (d) GPU nuScenes model (full
32,186 train chunks) for H9; (e) Wayformer; (f) shift injection B4; (g) Waymo when licensed; (h) fill paper
(results/discussion/abstract) from `results/`.

## 0. One-paragraph state of the project
Paper 01 asks: **does the conformal-prediction coverage guarantee survive a zero-label change of dataset in
trajectory prediction, and what repairs it?** **The forward-direction (AV2->nuScenes) H8 model zoo is now COMPLETE:
3/3 models analysed on real data.** Gaps at alpha=0.10: av2\_cpu\_v1 +3.3pt CI[2.1,4.5], av2\_valsplit\_v1 +8.0pt
CI[6.4,9.4], av2\_cpu\_v2 +10.5pt CI[9.4,12.2] — always under-coverage, magnitude ranges 3.2x and is NOT monotonic in
point-accuracy (the most accurate model has the worst gap). **The two rock-solid findings, with zero exceptions
across all 3 models:** covariate reweighting makes the gap worse, not better (H2/H3 refuted every time); the
labelled coverage audit (H5b) reliably detects the violation at k=1000 (power >=0.95, false-alarm <=6.2%) every
time. The normalised score (H4) and label-free monitor (H6) are genuinely mixed — supported-ish on one model,
refuted on another, inconclusive on the third — reported as such, not cherry-picked. A pre-registered kill condition
(base model within 15% of UniTraj's published AV2 minADE) was checked against the actual UniTraj paper and found
**triggered** (+28% to +59% worse) — disclosed as a quantified limitation, reasoned about, not hidden.
**nuScenes train-data conversion is DONE (all 11 chunks, ~40h wall time due to persistent CPU contention with AI2).
`ns_cpu_v1` training is now RUNNING (launched 2026-09-23 06:18).** Remaining: H9 (reverse direction) needs this
training to finish, then predict + analyze.
**Target: a paper that covers problem (measurement + diagnosis) and solution (audit + normalised score + label
budget), reporting all of this honestly including the mixed H4/H6 results and the kill-condition disclosure.**

## 1. Machine facts (do not re-derive)
* Windows 10, i9-10980XE (18C/36T), 64 GB RAM, **Quadro P2000 5 GB (Pascal, no tensor cores)**.
* The GPU is **shared with the user's other projects** (`F:\CLAUDE\AI2` Paper-02 detection cache holds ~4 GB of it).
  If `nvidia-smi` shows >2.5 GB free, GPU training is ~9x faster than CPU (53 vs 6 samples/s) — use `--device cuda`.
* Env: `F:\CLAUDE\AI1\shared\envs\unitraj\Scripts\python.exe` (Python 3.10, torch 2.0.1+cu118, numpy 1.24.2).
  Never `pip install` without `-c code/constraints.txt` (numpy must stay <2).
* `F:\CLAUDE\AI2` (Paper 02) and `AI3` (Paper 03) are the user's relocated projects. **Do not touch them or kill
  their processes.** Paper 02/03 files were removed from this repo on purpose (still in git history before e664282).
* Long jobs MUST run via Windows Task Scheduler (`run/schedule.ps1`); `Start-Process`/bash background tasks die when
  the host session ends.

## 2. How to see what is happening (30 seconds)
```powershell
cd F:\CLAUDE\AI1\paper01-coverage-transfer
F:\CLAUDE\AI1\shared\envs\unitraj\Scripts\python.exe src\journal.py status      # last event per experiment
F:\CLAUDE\AI1\shared\envs\unitraj\Scripts\python.exe src\journal.py tail 25     # recent events
Get-ScheduledTask -TaskName P01_* | Select TaskName,State                        # which tasks are Running
Get-Content journal\JOURNAL.md -Tail 60                                          # decisions/reasoning
```
Raw logs: `results/train_<exp>.{out,err}`, `results/predict_<model>.log`, `data/convert_ns_train.log`.

## 3. Jobs and how to relaunch each (all are idempotent/resumable)
| Job (task) | What | Status (2026-09-23 06:20) | Relaunch if dead |
|---|---|---|---|
| `av2_cpu_v2` (P01_TrainCpuV2) | fine-tune v1-ep08 with LR decay; 4 epochs x 60k scenes, CPU | **DONE** — `results/ckpts/av2_cpu_v2/epoch03-minADE1.092.ckpt` | n/a |
| `predict_chain1` (P01_PredictChain1) | schema-v2 predictions, models av2_cpu_v1 + av2_valsplit_v1 on av2cal/av2test/ns | **DONE** — all 6 npz exist | n/a |
| `predict_v2` (P01_PredictV2) | schema-v2 predictions for av2_cpu_v2 on av2cal/av2test/ns | **DONE** — all 3 npz exist | n/a |
| `ns_train_convert` (P01_ConvertNsTrain) | nuScenes prediction **train** split (32,186) in 11 chunks of 3,000 -> `data/nuscenes_scenarionet/train_cNN/` | **DONE, all 11/11 chunks.** Total wall time ~40h due to persistent AI2 CPU contention (per-chunk range: ~26min to 6.7h). Finished 2026-09-23 06:15:48. | n/a |
| `run2_real` (manual) | `analyze_run2.py --models av2_cpu_v1 av2_valsplit_v1 av2_cpu_v2` on **real** preds2 data | **DONE, all 3 models** — `results/run2/{av2_cpu_v1,av2_valsplit_v1,av2_cpu_v2}__forward.json`. **H8 model zoo (forward) is complete.** | rerun same command if a 4th model is added; ~5 min |
| `ns_cpu_v1` (P01_TrainNsCpuV1) | reverse-direction source model, `run\train_ns_cpu_v1.cmd` (all 11 chunk paths, 10 epochs like av2\_cpu\_v1, CPU) | **RUNNING**, launched 2026-09-23 06:18, confirmed producing output (data loading across 18 processes as of first check). **NOTE: an earlier auto-launch attempt (from inside a Monitor script) silently failed with a PowerShell execution-policy error; the script wrongly logged success. Relaunched manually and verified real output before trusting it — always verify with `Get-ScheduledTaskInfo` + log content, not just the launch command's own echo.** | `powershell -File run\schedule.ps1 -Name P01_TrainNsCpuV1 -Cmd F:\CLAUDE\AI1\paper01-coverage-transfer\run\train_ns_cpu_v1.cmd` (auto-resumes from `last.ckpt` if present) |

**Next launches, in order — H8 (forward model zoo) is DONE, `ns_cpu_v1` training is RUNNING:**
1. ~~Launch `run\train_ns_cpu_v1.cmd`~~ **DONE**, running since 2026-09-23 06:18. Check progress:
   `Get-Content results\train_ns_cpu_v1.log -Tail 20` or `python src\journal.py tail 10`.
2. When that finishes: `run\predict_model.cmd ns_cpu_v1 auto nscal nstest av2cal av2test` (the `nscal`/`nstest` sets
   already exist in `predict_all.py`, pointed at `data/nuscenes_splits/val/{cal,test}` which is **already split**).
3. Then `python src\analyze_run2.py --models ns_cpu_v1 --direction reverse` (H9), then `cd paper; powershell -File build.ps1`
   to regenerate the manuscript. `analyze_run2.py`'s reverse-direction code path is written but has not yet been
   run on real data — smoke-test it on a small `--limit` first if unsure, or just run it and check the printed gap
   is a sane number (roughly 0-15pt, not negative-huge or >50pt) before trusting the JSON.
4. `paper/make_results.py` already picks up `ns_cpu_v1__reverse.json` automatically (`REVERSE_MODELS` list) — no
   code change needed, just rerun `build.ps1`. Then fill the H9 `\todo` in `results.tex`, the model-zoo `\todo` in
   `setup.tex`, the contributions list in `intro.tex`, and write the abstract last, from the final numbers.
   Currently 4 `\todo`s left, all blocked on this chain.

## 4. Key files
* `notes/falsification.md` — **pre-registration**, Run-1 outcome log, erratum (unverified 0.73 yardstick),
  Addendum A (H4–H9, A2b audit). Never edit pre-registered text; append only.
* `src/conformal.py` core SCP; `src/solutions.py` H4/H5/H5b/H6 methods; `src/shift_factors.py` (label-free set = `*_LF`);
  `src/predict.py` / `predict_all.py`; `src/analyze_pair.py` (Run 1, legacy); `src/analyze_run2.py` (Run 2).
* `journal/events.jsonl` (append-only machine log) + `journal/JOURNAL.md` (human log) — **write to them**
  (`python src/journal.py note "..."`).
* `paper/` — manuscript (LaTeX, IEEEtran) + `make_results.py` (numbers/tables are regenerated from `results/*.json`;
  **never hand-type a result into the paper**).

## 5. Data / results map
* AV2: `data/av2_scenarionet/{train,val}`; splits `data/av2_splits/val/{train,cal,test}` (14,990/4,971/5,027; hash split).
  `cal`/`test` are used ONLY for calibration/evaluation; `val/train` is the clean model-selection set.
* nuScenes: `data/nuscenes_scenarionet/val` (9,041); train chunks `train_cNN` (in progress).
* Checkpoints: `results/ckpts/{av2_cpu_v1,av2_cpu_v2}`; archived 13.7k-scene model: `results/archive_valsplit_v1/ckpts`.
* Predictions: `results/preds/` = OLD schema (Run 1, do not use for new claims); `results/preds2/<model>/<set>.npz` = schema v2.
* Results JSON: `results/pair_*.json` (Run 1), `results/run2/*.json` (Run 2).

## 6. Known pitfalls (each cost real time once)
* UniTraj `--n`/`max_data_num` is **ignored for validation datasets** — use `--limit_batches` for quick tests.
* Relative paths break after UniTraj's `os.chdir`: always pass/resolve absolute paths; DB paths must be posix-style (`as_posix()`).
* `--lr`/`--lr_sched` must be set on the **top-level** cfg (fixed 2026-09-21; earlier runs used the default 7.5e-4).
* AV2 converter: `--num_files` ignored; absolute `--raw_data_path` required. nuScenes converter: don't pass
  `--past/--future` (untyped argparse -> str); >3 workers => GEOS `bad allocation`. AV2 `test` split has no GT.
* `obj_trajs` layout: xy 0:2 | type 6:11 | time 11:33 | heading 33:35 | velocity 35:37 | accel 37:39.
* DataLoader workers can die silently on CPU (happened once, ~3 h in) — `last.ckpt` every 200 steps + the retry wrapper cover it.
* Console encoding: set `PYTHONIOENCODING=utf-8` when printing non-ASCII.

## AUTONOMOUS PIPELINE (added 2026-09-25) -- runs with NO Claude session and NO user action
Every stage is a Task Scheduler watchdog (tick every 10 min, MultipleInstances=IgnoreNew, no time limit). Each is idempotent and
writes markers, so after a crash/logoff/reboot the next tick resumes where it stopped. Chain:
  P01_TrainAv2GpuFull -> results/ckpts/av2_gpu_full/DONE
  P01_PostAv2GpuFull  (waits for that DONE) -> predict, analyze_run2, analyze_addB, shift_inject+analyze_inject; markers results/post/av2_gpu_full/*.ok, POST_DONE; log results/post_av2_gpu_full.log
  P01_TrainNsGpuFull  (waits for av2_gpu_full DONE/FAILED) -> results/ckpts/ns_gpu_full/DONE
  P01_PostNsGpuFull   (waits for ns DONE) -> reverse predictions + analyses; results/post/ns_gpu_full/POST_DONE
Status any time:  python src/journal.py status ; python src/journal.py tail 20 ; ls results/post/*
Manual steps left AFTER POST_DONE (need a Claude session): add av2_gpu_full to make_results_addB CONFIRMATORY list, evaluate H16b/H10 in
notes/falsification.md, rewrite abstract/intro/discussion, fill setup.tex TODO, python paper/make_results.py, powershell -File paper/build.ps1.
Limits: PC must stay ON and not sleep (power plan: never sleep). A logoff can kill tasks; the watchdog restarts them at the next tick if the
task is set to run when user is logged on and the user logs back in.

## WAYMO (added 2026-09-28)
User accepted the Waymo licence (h.waqas160@gmail.com); gcloud SDK installed at shared/tools/google-cloud-sdk (auth done; CLOUDSDK_PYTHON=shared/envs/gcs).
Bucket gs://waymo_open_dataset_motion_v_1_2_1/uncompressed/scenario/validation (150 shards ~275 MB, ~294 scenarios each; we take the first 30).
Watchdog P01_DownloadWaymo -> data/waymo_raw/validation, marker results/ckpts/waymo_dl/DONE, log results/download_waymo.log.
NEXT: convert to ScenarioNet (needs tensorflow + waymo-open-dataset in a SEPARATE env, not the unitraj env), then scene-level cal/test split, predict with av2_gpu_full and ns_gpu_full, pre-register Waymo hypotheses (Addendum C) BEFORE looking at results.

## ZENODO ARCHIVE (added 2026-09-30)
Draft deposition created via the Zenodo API (user's personal access token, used once, not stored anywhere in the
repo). Deposition id 23053619, reserved DOI 10.5281/zenodo.23053619, state unsubmitted/draft, NO files uploaded
(by design, per user instruction). Cited in paper/main.tex's Data Availability section.
NEXT when the code repo is made public: upload the release archive to this deposition (bucket URL retrievable again
via the API with the token, or via zenodo.org web UI while logged into the account that created it) and click
Publish -- this locks the DOI permanently and it will resolve. Do this AFTER, not before, main.tex's numbers are
final, since re-uploading to a published record requires a new version.

## ZENODO: PDF uploaded (2026-09-30)
main.pdf (588733 bytes, md5 95cd6ccf7eeccd6427941977ad9aaaa6) uploaded to the draft deposition (id 23053619,
DOI 10.5281/zenodo.23053619) via the bucket API. Record is still UNSUBMITTED/draft: not public, not published,
DOI does not resolve yet. Only the PDF is on Zenodo, per instruction -- no .tex source, no code. Publish only when
told to.
