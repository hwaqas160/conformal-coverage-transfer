@echo off
REM model-zoo predictions, sequential to limit CPU contention with training. Resumable (skips existing outputs).
cd /d F:\CLAUDE\AI1\paper01-coverage-transfer
call run\predict_model.cmd av2_cpu_v1 F:\CLAUDE\AI1\paper01-coverage-transfer\results\ckpts\av2_cpu_v1\epoch08-minADE1.349.ckpt av2cal av2test ns
call run\predict_model.cmd av2_valsplit_v1 F:\CLAUDE\AI1\paper01-coverage-transfer\results\archive_valsplit_v1\ckpts\epoch17-minADE1.347.ckpt av2cal av2test ns
