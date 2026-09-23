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
