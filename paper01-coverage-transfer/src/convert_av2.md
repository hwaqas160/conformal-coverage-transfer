# Argoverse 2 -> ScenarioNet conversion

The `av2` devkit is installed. The converter is `scenarionet.convert_argoverse2`.
**Tested working 2026-09-10** (see gotchas below — two of them cost real time).

## GOTCHA 1 — `--raw_data_path` MUST be an absolute Windows path

`convert_argoverse2.py` does `os.path.join(SCENARIONET_DATASET_PATH, args.raw_data_path)`.
A relative path gets prepended with `.../code/scenarionet/dataset/` and silently finds
**0 scenarios** (converter "succeeds" with an empty database). An absolute path like
`F:\CLAUDE\AI1\...\val` survives the join unchanged. Always pass absolute paths for BOTH
`-d` and `--raw_data_path`.

## GOTCHA 2 — `--num_files` / `--start_file_index` are IGNORED for AV2

`get_av2_scenarios()` takes those args but its body ignores them — it always returns
every `*.parquet` under `--raw_data_path`. So you cannot do a "small test" by capping
files. To test on a subset, point `--raw_data_path` at a folder containing only a few
scenario subdirs (copy ~20 dirs elsewhere first). Otherwise the converter processes the
whole split. **To resume a died `train` run you must move already-done raw dirs aside**,
there is no built-in resume for AV2.

## GOTCHA 3 — run from a directory NOT named `scenarionet`

ScenarioNet docs: `python -m` fails if CWD contains a folder called `scenarionet`.
Run from `paper01-coverage-transfer/`.

## Raw data layout (verified)

```
data/argoverse2/
├── train/  199,908 scenarios   (each: scenario_<id>.parquet + log_map_archive_<id>.json)
├── val/     24,988 scenarios
└── test/    24,984 scenarios
```

## Converter arguments (from `-h`)

| arg | meaning |
|---|---|
| `-d, --database_path` | output dir for converted DB |
| `-n, --dataset_name` | name prefix for scenario files |
| `--raw_data_path` | input dir (one of train/ val/ test/) |
| `--num_workers` | parallel workers (use ~8; you have 18 cores but I/O-bound) |
| `--start_file_index` / `--num_files` | process files[start : start+num] — for testing / chunking |
| `--overwrite` | replace existing output dir |

## Step 1 — SMALL TEST FIRST (do this before the full run)

```bash
cd F:\CLAUDE\AI1\paper01-coverage-transfer
set PY=F:\CLAUDE\AI1\shared\envs\unitraj\Scripts\python.exe

%PY% -m scenarionet.convert_argoverse2 ^
  -d data/av2_scenarionet_TEST ^
  -n av2_test ^
  --raw_data_path data/argoverse2/val ^
  --num_workers 4 ^
  --num_files 20 ^
  --overwrite
```

Then check it:
```bash
%PY% -m scenarionet.num -d data/av2_scenarionet_TEST     # should report ~20
```

If that works, delete the test dir and do the full conversion.

## Step 2 — FULL conversion (one command per split)

Each split is independent. `val` first (smallest, fastest to get a usable pipeline):

```bash
%PY% -m scenarionet.convert_argoverse2 -d data/av2_scenarionet/val   -n av2_val   --raw_data_path data/argoverse2/val   --num_workers 8 --overwrite
%PY% -m scenarionet.convert_argoverse2 -d data/av2_scenarionet/test  -n av2_test  --raw_data_path data/argoverse2/test  --num_workers 8 --overwrite
%PY% -m scenarionet.convert_argoverse2 -d data/av2_scenarionet/train -n av2_train --raw_data_path data/argoverse2/train --num_workers 8 --overwrite
```

`train` will take hours and produce a large output. Run it overnight. If it dies partway,
use `--start_file_index` to resume from where it stopped.

## Step 3 — point UniTraj at the converted data

In `code/UniTraj/unitraj/configs/config.yaml`:

```yaml
train_data_path: [ "F:/CLAUDE/AI1/paper01-coverage-transfer/data/av2_scenarionet/train" ]
val_data_path:   [ "F:/CLAUDE/AI1/paper01-coverage-transfer/data/av2_scenarionet/val" ]
```

Note UniTraj parses `phase, dataset_name = data_path.split('/')[-2], [-1]` — so the path
must end in `.../<phase>/<dataset_name>` style OR just verify the cache path it prints
makes sense. The shipped example used `data_samples/nuscenes`. Mirror that: a parent dir
and a leaf. `av2_scenarionet/train` satisfies it (phase=`av2_scenarionet`, name=`train`).

## For Paper 01 specifically

You do NOT need to convert all 199,908 train scenarios to start. For the coverage-transfer
experiment a stratified subset (e.g. 20-40k train, full val) is enough to train AutoBot and
calibrate conformal intervals. Use `--num_files` to cap it. Document the subset size in
`notes/`.
