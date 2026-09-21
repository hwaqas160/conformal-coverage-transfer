# RESUME — start here after a crash, a reboot, or a new session

_Keep this file accurate: update it whenever a job is launched/finished or a decision changes. Last full update:
2026-09-21 17:00._

## 0. One-paragraph state of the project
Paper 01 asks: **does the conformal-prediction coverage guarantee survive a zero-label change of dataset in
trajectory prediction, and what repairs it?** Run 1 (AutoBot trained on Argoverse 2 -> nuScenes, zero labels) showed a
real under-coverage of **+3.4 pts** at alpha=0.10 (95% CI 2.7–4.1), robust to a sampling-rate control; covariate
reweighting did **not** repair it (H2/H3 refuted). Since then: a disclosed flaw (H3 used GT-derived features) was
fixed, hypotheses H4–H9 were pre-registered (`notes/falsification.md` Addendum A), solution code was written and
validated on synthetic data, and a stronger base model + more dataset pairs are being produced. **Target: a paper
that covers problem (measurement + diagnosis) and solution (audit + normalised score + label budget).**

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
| Job (task) | What | Done when | Relaunch if dead |
|---|---|---|---|
| `av2_cpu_v2` (P01_TrainCpuV2) | fine-tune v1-ep08 with LR decay; 4 epochs x 60k scenes, CPU; ~10 h | journal `av2_cpu_v2 done`; `results/ckpts/av2_cpu_v2/epoch*.ckpt` | `powershell -File run\schedule.ps1 -Name P01_TrainCpuV2 -Cmd F:\CLAUDE\AI1\paper01-coverage-transfer\run\train_cpu_v2.cmd` (auto-resumes from `last.ckpt`) |
| `predict_chain1` (P01_PredictChain1) | schema-v2 predictions, models av2_cpu_v1 + av2_valsplit_v1 on av2cal/av2test/ns | `results/preds2/<model>/{av2cal,av2test,ns}.npz` exist | rerun `run\predict_chain1.cmd` via schedule.ps1 (skips existing) |
| `ns_train_convert` (P01_ConvertNsTrain) | nuScenes prediction **train** split (32,186) in 11 chunks of 3,000 -> `data/nuscenes_scenarionet/train_cNN/` | all 11 `train_cNN/dataset_summary.pkl` exist | rerun `run\convert_ns_train.cmd` (skips finished chunks) |

**Next launches (not started yet), in order:**
1. After `av2_cpu_v2` finishes: `run\predict_model.cmd av2_cpu_v2 auto av2cal av2test ns` (model-zoo entry + main model).
2. After ns chunks finish: train ns-source model (`src/train_autobot.py --train_db data/nuscenes_scenarionet/train_c00 ...`;
   `train_data_path` accepts a list -> extend `train_autobot.py` for multiple DBs), then predict on ns (val halves)
   and av2 -> reverse-direction pair (H9). Split nuScenes val into cal/test first:
   `python src/split_db.py --db data/nuscenes_scenarionet/val --out data/nuscenes_splits/val --fractions 0.5 0.5 --names cal test --salt nsv1`.
3. Analyses: `python src/analyze_run2.py` (see §5), then `python paper/make_results.py`, then compile the paper.

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
