#!/usr/bin/env bash
# Manual full training. Bounded validation evidence and limitations: README_HOI.md.
set -euo pipefail
run_name=${1:?Usage: bash configs/hoi/refined_full.sh RUN_NAME [--resume EXACT_CHECKPOINT]}
shift
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd -- "$repo_dir"
exec env -u PYTHONPATH OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python -m supermimic.hoi.train \
  --run-name "$run_name" --log-root logs --device cuda:0 \
  --profile refined-single-body --solver-iterations 50 \
  --reference outputs/hoi/asset_audit/part_b/reference_refined/reference_refined.npz \
  --env.num-envs 64 --env.sampling-mode return --env.replay-capacity 256 \
  --schedule.total-iterations 320000 --schedule.updates-per-env-step 0.4 \
  --schedule.warmup-iterations 40 \
  --startup.wm-only-updates 50 --startup.min-windows 32 --startup.min-policy-starts 32 \
  --startup.max-collect-iterations 2000 \
  --wm-trainer.batch-size 256 --wm-trainer.batches-per-update 10 \
  --policy-trainer.batch-size 256 --policy-trainer.horizon 32 \
  --schedule.log-interval 50 --schedule.checkpoint-interval 4000 "$@"
