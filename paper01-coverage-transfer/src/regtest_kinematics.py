"""One-off regression check (Addendum E step 2.5): recompute_kinematics must be an exact no-op for AutoBot.
Re-runs 10 batches of the existing hz2 injection on av2_gpu_full and diffs against the already-published
inj_hz2.npz. Not a paper result; writes to a scratch path, never touches results/preds2."""
import json
import sys

import numpy as np

sys.path.insert(0, "src")
from predict import dump  # noqa: E402
from shift_inject import make_transform, PARAMS  # noqa: E402

if __name__ == "__main__":
    params = json.loads(PARAMS.read_text())
    dump("results/ckpts/av2_gpu_full/last.ckpt", "data/av2_splits/val/test", "av2", "regtest", "results",
         device="cuda", batch_size=32, num_workers=3, out_file="C:/CLAUDE_scratch_regtest.npz",
         limit_batches=10, cache_root="C:\\p01_cache", transform=make_transform("hz2", params))

    new = np.load("C:/CLAUDE_scratch_regtest.npz", allow_pickle=True)
    old = np.load("results/preds2/av2_gpu_full/inj_hz2.npz", allow_pickle=True)
    n = len(new["scores"])
    print("n rows compared:", n, "old file total rows:", len(old["scores"]))
    match_scores = np.allclose(new["scores"], old["scores"][:n], atol=1e-5)
    match_traj = np.allclose(new["pred_trajs"], old["pred_trajs"][:n], atol=1e-5)
    print("scores match:", match_scores, " pred_trajs match:", match_traj)
    if not match_scores:
        print("max abs diff scores:", np.abs(new["scores"] - old["scores"][:n]).max())
