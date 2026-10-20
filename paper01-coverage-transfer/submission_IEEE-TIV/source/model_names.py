"""One mapping from internal run identifiers (used in the released code and result files) to the names used in the paper.

Scheme: <architecture>-<training data>-<run>
  architecture: AB = AutoBot-Ego, WF = Wayformer
  training data: AV2 = Argoverse 2, NS = nuScenes
  run: s0, s1 = full-data confirmatory runs with training seeds 0 and 1; r1-r3 = reduced-budget exploratory runs.
"""

NAMES = {
    "av2_gpu_full": "AB-AV2-s0",
    "av2_gpu_full_seed1": "AB-AV2-s1",
    "av2_wayformer": "WF-AV2-s0",
    "av2_wayformer_seed1": "WF-AV2-s1",
    "ns_gpu_full": "AB-NS-s0",
    "av2_cpu_v1": "AB-AV2-r1",
    "av2_valsplit_v1": "AB-AV2-r2",
    "av2_cpu_v2": "AB-AV2-r3",
    "ns_cpu_v1": "AB-NS-r1",
    "multi_av2ns": "AB-AV2+NS-s0",
}


def paper_name(run_id: str) -> str:
    """LaTeX-safe paper name for an internal run identifier (falls back to the escaped identifier)."""
    return NAMES.get(run_id, run_id.replace("_", r"\_"))
