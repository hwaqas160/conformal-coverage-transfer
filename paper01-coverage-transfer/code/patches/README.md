# Local modifications to UniTraj

`code/UniTraj` is a clone of https://github.com/vita-epfl/UniTraj pinned at upstream commit **dea7c74**
("updated README (#66)"). It is git-ignored here (re-clone, don't vendor). All local changes are captured in
`unitraj_dea7c74_local.patch` and must be re-applied after cloning:

    git clone https://github.com/vita-epfl/UniTraj code/UniTraj
    cd code/UniTraj && git checkout dea7c74 && git apply ../patches/unitraj_dea7c74_local.patch

Regenerate the patch after any further change:  `cd code/UniTraj && git diff > ../patches/unitraj_dea7c74_local.patch`

Notable change (2026-09-23): `models/autobot/autobot.py` `get_road_pts_mask` -- `.type(torch.BoolTensor).to(dev)`
replaced by `.bool()`. `torch.BoolTensor` is a CPU type, so the original copied the map mask GPU->CPU->GPU and
synchronised every step. The mask is bit-identical (verified with torch.equal on a real batch), so no model output
changes; only speed.

Notable change (2026-09-24): `models/autobot/autobot.py` `Criterion.get_BVG_distributions` now returns a closed-form
bivariate-Gaussian entropy object instead of building a MultivariateNormal (batched Cholesky + CUDA sync every mode
every step; 25.5% of main-thread wall time by py-spy). Only `.entropy()` is ever used. Verified on real batches with
identical dropout seeds, train and eval mode: loss rel. diff <= 1.5e-7, gradient rel-L2 <= 5.9e-7, cosine 1.0000.
The original remains as `get_BVG_distributions_reference`. (An earlier comparison without fixed seeds showed 2-5%
loss differences and cosine 0.2-0.96: that was dropout noise between two forward passes, not the formula.)
