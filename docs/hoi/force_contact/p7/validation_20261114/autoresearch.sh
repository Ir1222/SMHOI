#!/usr/bin/env bash
# Exact historical commands, including expected failures; retained output directories intentionally refuse reuse.
# Historical source_v1/v2/v3/v4 archives preserve old identities; source_v5 matches current equal-LR interface code.
# This is a command transcript. Review and use NEW output directories for independent reproductions.
exit 1 # Prevent accidental reruns against preserved evidence.
# reset_source_v1: exit=0, charged=6.117s
env -u PYTHONPATH OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python scripts/hoi/validate_reset_source.py --output logs/hoi_validation_20261114/reset_source_v1
# train_small_v1: exit=0, charged=23.411s
env -u PYTHONPATH OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python -m supermimic.hoi.train --log-root logs/hoi_validation_20261114 --run-name train_small_v1 --env.num-envs 2 --env.replay-capacity 64 --schedule.total-iterations 64 --schedule.warmup-iterations 12 --schedule.updates-per-env-step 0.4 --schedule.log-interval 8 --schedule.checkpoint-interval 32 --startup.wm-only-updates 2 --startup.min-windows 2 --startup.min-policy-starts 2 --wm-trainer.batch-size 8 --wm-trainer.batches-per-update 1 --policy-trainer.batch-size 8 --policy-trainer.horizon 4
# resume_small_v1: exit=0, charged=17.305s
env -u PYTHONPATH OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python -m supermimic.hoi.train --log-root logs/hoi_validation_20261114 --run-name resume_small_v1 --env.num-envs 2 --env.replay-capacity 64 --schedule.total-iterations 64 --schedule.warmup-iterations 12 --schedule.updates-per-env-step 0.4 --schedule.log-interval 8 --schedule.checkpoint-interval 32 --startup.wm-only-updates 2 --startup.min-windows 2 --startup.min-policy-starts 2 --wm-trainer.batch-size 8 --wm-trainer.batches-per-update 1 --policy-trainer.batch-size 8 --policy-trainer.horizon 4 --resume logs/hoi_validation_20261114/train_small_v1/hoi_0000032.pt
# compare_small_v1: exit=1, charged=3.069s
env -u PYTHONPATH OMP_NUM_THREADS=1 .venv/bin/python scripts/hoi/compare_training_checkpoints.py logs/hoi_validation_20261114/train_small_v1/hoi_0000064.pt logs/hoi_validation_20261114/resume_small_v1/hoi_0000064.pt --output logs/hoi_validation_20261114/resume_comparison_v1.json
# resume_state_v1: exit=0, charged=8.153s
env -u PYTHONPATH OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python scripts/hoi/validate_resume_state.py --checkpoint logs/hoi_validation_20261114/train_small_v1/hoi_0000032.pt --output logs/hoi_validation_20261114/resume_state_v1
# train_medium_v2: exit=0, charged=57.986s
env -u PYTHONPATH OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python -m supermimic.hoi.train --log-root logs/hoi_validation_20261114 --run-name train_medium_v2 --env.num-envs 16 --env.replay-capacity 128 --schedule.total-iterations 96 --schedule.warmup-iterations 20 --schedule.updates-per-env-step 0.4 --schedule.log-interval 16 --schedule.checkpoint-interval 96 --startup.wm-only-updates 10 --startup.min-windows 32 --startup.min-policy-starts 32 --wm-trainer.batch-size 64 --wm-trainer.batches-per-update 2 --policy-trainer.batch-size 64 --policy-trainer.horizon 16
# train_scale64_v2: exit=1, charged=7.131s
env -u PYTHONPATH OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python -m supermimic.hoi.train --log-root logs/hoi_validation_20261114 --run-name train_scale64_v2 --schedule.max-hours 0.05 --schedule.checkpoint-interval 100
# train_scale64_solver50_v3: exit=0, charged=231.935s
env -u PYTHONPATH OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python -m supermimic.hoi.train --log-root logs/hoi_validation_20261114 --run-name train_scale64_solver50_v3 --solver-iterations 50 --schedule.max-hours 0.05 --schedule.checkpoint-interval 100
# regression_v4: exit=1, charged=21.387s
env -u PYTHONPATH OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/pytest tests/hoi/test_scratch_lifecycle.py tests/hoi/test_training_integration.py tests/hoi/test_training_robot_regression.py tests/hoi/test_preflight_provenance.py tests/hoi/test_readme_hoi.py tests/hoi/test_solver_precision.py tests/hoi/test_imagination_boundaries.py tests/hoi/test_force_replay.py --basetemp tests/.artifacts/real_training_v4 -q
# regression_fix_v4: exit=1, charged=10.203s
env -u PYTHONPATH OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/pytest tests/hoi/test_scratch_lifecycle.py::test_original_loop_stage_order_and_robot_defaults tests/hoi/test_scratch_lifecycle.py::test_zero_update_smoke_resume_and_policy_play --basetemp tests/.artifacts/real_training_fix_v4 -q
# regression_mock_final: exit=0, charged=5.117s
env -u PYTHONPATH OMP_NUM_THREADS=1 .venv/bin/pytest tests/hoi/test_scratch_lifecycle.py::test_original_loop_stage_order_and_robot_defaults -q
# solver10_failure_v4: exit=1, charged=7.138s
env -u PYTHONPATH OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python -m supermimic.hoi.train --log-root logs/hoi_validation_20261114 --run-name solver10_failure_v4 --solver-iterations 10 --schedule.total-iterations 16
# readme_final_parse: exit=0, charged=5.117s
env -u PYTHONPATH OMP_NUM_THREADS=1 .venv/bin/pytest tests/hoi/test_readme_hoi.py::test_readme_commands_parse_and_resume_contract -q
# train_final_v4: exit=0, charged=230.940s
env -u PYTHONPATH OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python -m supermimic.hoi.train --log-root logs/hoi_validation_20261114 --run-name train_final_v4 --reference outputs/hoi/asset_audit/part_b/reference_refined/reference_refined.npz --schedule.max-hours 0.05 --schedule.checkpoint-interval 250
# resume_state_final_v4: exit=0, charged=8.161s
env -u PYTHONPATH OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python scripts/hoi/validate_resume_state.py --checkpoint logs/hoi_validation_20261114/train_final_v4/hoi_0000250.pt --output logs/hoi_validation_20261114/resume_state_final_v4
# resume_final_v4: exit=0, charged=208.568s
env -u PYTHONPATH OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python -m supermimic.hoi.train --log-root logs/hoi_validation_20261114 --run-name resume_final_v4 --reference outputs/hoi/asset_audit/part_b/reference_refined/reference_refined.npz --schedule.max-hours 0.05 --schedule.checkpoint-interval 250 --resume logs/hoi_validation_20261114/train_final_v4/hoi_0000250.pt
# play_final_v4: exit=0, charged=9.183s
env -u PYTHONPATH OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python -m supermimic.hoi.play --ckpt logs/hoi_validation_20261114/resume_final_v4/hoi_0000350.pt --output-dir logs/hoi_validation_20261114/play_final_v4 --max-steps 839 --headless
# boundaries_final_v4: exit=0, charged=27.483s
env -u PYTHONPATH OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python -m supermimic.hoi.train --log-root logs/hoi_validation_20261114 --run-name boundaries_final_v4 --reference outputs/hoi/asset_audit/part_b/reference_refined/reference_refined.npz --env.num-envs 2 --env.sampling-mode start --reset-frames 0 830 --env.episode-length-s 0.24 --env.replay-capacity 64 --schedule.total-iterations 48 --schedule.warmup-iterations 12 --schedule.log-interval 12 --schedule.checkpoint-interval 48 --startup.wm-only-updates 2 --startup.min-windows 2 --startup.min-policy-starts 2 --wm-trainer.batch-size 8 --wm-trainer.batches-per-update 1 --policy-trainer.batch-size 8 --policy-trainer.horizon 16
# final_config_parity: exit=0, charged=5.114s
env -u PYTHONPATH OMP_NUM_THREADS=1 .venv/bin/pytest tests/hoi/test_readme_hoi.py::test_readme_commands_parse_and_resume_contract -q
# critic_diagnosis_v4: exit=0, charged=8.167s
env -u PYTHONPATH OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python scripts/hoi/diagnose_critic_checkpoint.py --checkpoint logs/hoi_validation_20261114/resume_final_v4/hoi_0000350.pt --output logs/hoi_validation_20261114/critic_diagnosis_v4
# lr_interface_regression: exit=0, charged=7.140s
env -u PYTHONPATH OMP_NUM_THREADS=1 .venv/bin/pytest tests/hoi/test_training_lr.py tests/hoi/test_scratch_lifecycle.py::test_original_loop_stage_order_and_robot_defaults tests/hoi/test_training_robot_regression.py tests/hoi/test_training_integration.py::test_inherited_update_interfaces_with_one_synthetic_batch tests/hoi/test_readme_hoi.py::test_readme_commands_parse_and_resume_contract -q
# interface_current_v5: exit=0, charged=17.317s
env -u PYTHONPATH OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python -m supermimic.hoi.train --log-root logs/hoi_validation_20261114 --run-name interface_current_v5 --reference outputs/hoi/asset_audit/part_b/reference_refined/reference_refined.npz --env.num-envs 2 --env.replay-capacity 64 --schedule.total-iterations 48 --schedule.warmup-iterations 12 --schedule.log-interval 12 --schedule.checkpoint-interval 24 --startup.wm-only-updates 2 --startup.min-windows 2 --startup.min-policy-starts 2 --wm-trainer.batch-size 8 --wm-trainer.batches-per-update 1 --policy-trainer.batch-size 8 --policy-trainer.horizon 4
# interface_resume_state_v5: exit=0, charged=7.136s
env -u PYTHONPATH OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python scripts/hoi/validate_resume_state.py --checkpoint logs/hoi_validation_20261114/interface_current_v5/hoi_0000024.pt --output logs/hoi_validation_20261114/interface_resume_state_v5
# interface_resume_v5: exit=0, charged=14.243s
env -u PYTHONPATH OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python -m supermimic.hoi.train --log-root logs/hoi_validation_20261114 --run-name interface_resume_v5 --resume logs/hoi_validation_20261114/interface_current_v5/hoi_0000024.pt --reference outputs/hoi/asset_audit/part_b/reference_refined/reference_refined.npz --env.num-envs 2 --env.replay-capacity 64 --schedule.total-iterations 48 --schedule.warmup-iterations 12 --schedule.log-interval 12 --schedule.checkpoint-interval 24 --startup.wm-only-updates 2 --startup.min-windows 2 --startup.min-policy-starts 2 --wm-trainer.batch-size 8 --wm-trainer.batches-per-update 1 --policy-trainer.batch-size 8 --policy-trainer.horizon 4
# interface_play_v5: exit=0, charged=9.170s
env -u PYTHONPATH OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python -m supermimic.hoi.play --ckpt logs/hoi_validation_20261114/interface_resume_v5/hoi_0000048.pt --output-dir logs/hoi_validation_20261114/interface_play_v5 --max-steps 839 --headless
# final_document_regression: exit=0, charged=8.151s
env -u PYTHONPATH OMP_NUM_THREADS=1 .venv/bin/pytest tests/hoi/test_training_lr.py tests/hoi/test_readme_hoi.py tests/hoi/test_preflight_provenance.py -q
