# Paper 02 — Dataset Acquisition

Boreas is large and Paper 02 stalls without it. **Start the download in month 4**, while
Paper 01 is under review. Download time is dead time you can overlap.

---

## 1. Boreas  ← PRIMARY, no account needed

Commands below are **verified from `code/pyboreas/README.md`**, not invented.

- Website / sequence browser: https://www.boreas.utias.utoronto.ca/#/download
- Devkit: `code/pyboreas/` (already cloned)

### Full dataset (very large — probably not what you want)

```bash
root=F:/CLAUDE/AI1/paper02-conflict-fusion/data/boreas/
aws s3 sync s3://boreas $root --no-sign-request
```

### List available sequences first

```bash
aws s3 ls s3://boreas --no-sign-request
```

### Selective download — DO THIS INSTEAD

Pull only the sensors you need, for the sequences you need:

```bash
root=F:/CLAUDE/AI1/paper02-conflict-fusion/data/boreas/
cd $root
aws s3 sync s3://boreas/boreas-2020-11-26-13-58 boreas-2020-11-26-13-58 --no-sign-request \
    --exclude "*" \
    --include "lidar/*" --include "radar/*" \
    --include "applanix/*" --include "calib/*"
```

Add `--include "camera/*"` when you need imagery.

> **No AWS account required** — `--no-sign-request` handles anonymous access. You only need
> the AWS CLI installed: https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html

### Sequence selection strategy

Boreas repeats **the same route** across a full year. That is the whole point — it gives you
matched clear / rain / snow / night traversals of identical geometry.

Pick, at minimum:
- 2–3 **clear daytime** sequences → calibration set
- 2–3 **snow** sequences → shift test
- 1–2 **rain** and **night** sequences → secondary conditions

Use the [sequence browser](https://www.boreas.utias.utoronto.ca/#/download) to identify
conditions per sequence; it generates the AWS CLI commands for you.

### Verify the loader works

```python
from pyboreas import BoreasDataset
bd = BoreasDataset('F:/CLAUDE/AI1/paper02-conflict-fusion/data/boreas/')
```

---

## 2. CADC — Canadian Adverse Driving Conditions

- ~7k annotated frames of snowy driving, from Waterloo
- Search "CADC dataset Waterloo" for the current host and registration form — I could not
  verify a stable URL at time of writing, so check rather than guess.

Role: **external validation.** Boreas is your development set; CADC is the independent snow
cohort you touch once, at the end.

---

## 3. nuScenes — clear-weather reference

Already downloading for Paper 01. No extra work.

---

## 4. SeeingThroughFog / DENSE  (optional)

Adds fog as a third degradation type. Registration required. Only pursue if Boreas + CADC
land early.

---

## Disk planning

Boreas is the big one. Use `--include` filters aggressively — you do **not** need every
sensor for every sequence. Record actual usage in `notes/disk.md` as you go.

---

## Checklist

- [ ] AWS CLI installed
- [ ] `aws s3 ls s3://boreas --no-sign-request` returns a sequence list
- [ ] Sequence conditions catalogued → `notes/boreas_sequences.md`
- [ ] Clear-weather sequences downloaded
- [ ] Snow sequences downloaded
- [ ] `BoreasDataset` loads a sequence successfully
- [ ] CADC registration submitted
- [ ] Pretrained detector checkpoints identified and recorded
