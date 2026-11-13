# SMHOI 使用说明

SMHOI 使用在线 Joint WM、Actor B 的 BPTT 和 Critic B+ 的 TD(λ) 学习 humanoid–Box motion imitation。数据由 Actor 与真实 MuJoCo 场景交互产生；复用原 `Trainer.run`、Replay 和三套 Adam，没有离线采集或 reference-PD 前置流程。

**真实训练闭环已跑通至 64 环境、batch 256、horizon 32；当前结论为 full-train ready = false。** 已验证内部 WM 启动、交替更新与完整保存恢复，但 reference 的 reset 穿透仍存在，续训中 Critic loss 最高约 448.7，新 Actor 轨迹上的 Box 预测也劣于零 residual 先验。本机磁盘空间也不足以保存正式长训练的全部快照。详细证据见 [本轮报告](docs/hoi/force_contact/p7/validation_20261114/REPORT.md)；历史物理 FAIL / `training_ready=false` 没有改成 PASS。

## 环境与本地数据

使用已有 `.venv`，不需要安装或升级依赖。物理训练支持单进程单 CUDA GPU；实测机器为 RTX 4090 D 24 GB。

```bash
cd /home/han/pyx/SMHOI
env -u PYTHONPATH .venv/bin/python -c 'import torch, mujoco, mjlab; print(torch.__version__, mujoco.__version__, torch.cuda.is_available())'
sha256sum outputs/hoi/asset_audit/part_b/reference_refined/reference_refined.npz
```

必要输入：

- `outputs/hoi/asset_audit/part_b/reference_refined/reference_refined.npz`
- 同目录 `reference_manifest.json`
- NPZ SHA256：`f6ba446c16e71ab3cffb09ec015ca9a224305f6ac82d01f3631d4d5990a5b5d7`
- Profile：`refined-single-body`；840 帧、50 Hz，839 个控制间隔（16.78 s），m/s/rad、wxyz。

**refined reference 是本地文件，Git 克隆不会自动提供。** 换机器时应从已批准的本地来源复制这两个文件并核对哈希；同时检查现有资产 manifest 引用的本地资产路径。缺失或身份不符会报错，不重新转换、不静默回退。Reference 用于 reset、目标和 reward；Replay 启动时为空；首次训练不需要 checkpoint。

## 手动 full train

[正式训练候选配置](configs/hoi/refined_full.sh) 固定下述参数。本轮没有启动正式长训练；以下命令供你审查，当前学习与几何阻塞解决前不作为已验收的长训建议。脚本用法为 `bash configs/hoi/refined_full.sh 新运行名`。

<!-- scratch-command: parsed by tests/hoi/test_readme_hoi.py -->
```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python -m supermimic.hoi.train \
  --run-name hoi_refined_scratch_v1 --log-root logs --device cuda:0 \
  --profile refined-single-body --solver-iterations 50 \
  --reference outputs/hoi/asset_audit/part_b/reference_refined/reference_refined.npz \
  --env.num-envs 64 --env.sampling-mode return --env.replay-capacity 256 \
  --schedule.total-iterations 320000 --schedule.updates-per-env-step 0.4 \
  --schedule.warmup-iterations 40 \
  --startup.wm-only-updates 50 --startup.min-windows 32 --startup.min-policy-starts 32 \
  --startup.max-collect-iterations 2000 \
  --wm-trainer.batch-size 256 --wm-trainer.batches-per-update 10 \
  --policy-trainer.batch-size 256 --policy-trainer.horizon 32 \
  --schedule.log-interval 50 --schedule.checkpoint-interval 4000
```

运行目录必须不存在。默认 scratch，不加载旧权重。流程为：

1. 新建 WM/Actor/Critic、Adam、sampler 和空 Replay。WM residual 层零初始化，Actor 原 action 输出 gain 0.01。
2. Actor 在线采集。前 40 步只采集；之后要求至少 32 个有效八步窗口和 32 个 policy start。
3. 内部 WM-only 启动：50 次 WM update，每次 10 个 minibatch；期间持续 Actor 采集。
4. 自动进入持续 WM → Actor BPTT → Critic TD(λ) 交替更新。无需分阶段批准或外部 WM checkpoint。

这些参数验证了短时工程链路，尚未通过学习稳定性验收。`solver_iterations=50` 修复实测 64 环境在第 10 个控制步触及旧上限 10 的问题；求解容差、line-search 上限 20、dt=0.005、四子步、碰撞和所有 overflow 检查保持不变。提高迭代预算不消除 reference 初态穿透。

| 参数 | 含义 |
|---|---|
| 320000 iterations | 向量控制步，总物理 transitions 约 2048 万 |
| UTD 0.4 | 每控制步累计 0.4 次更新；每次 WM update 有 10 个 minibatch |
| Replay capacity 256 | 每环境 ring 行数；terminal state 另占一行 |
| Actor batch 256 / H32 | 已实测短跑；Critic 保持 5 epochs × 4 minibatches |
| 学习率 | 三套初始 2e-4，原 cosine schedule；γ=0.99、λ=0.95，compile 关闭 |
| Return sampler | 原 1 秒 bins；50% uniform、40% softmax、10% memory；理论 Gmax=18 |
| 扰动 | 动作噪声 std=0.1；观测/reset 噪声和 domain randomization 关闭 |

Robot reward 不变。Box sigmas `(0.2 m, 0.4 rad, 1 m/s, 2 rad/s)`，weights `(1,1,0.5,0.5)`；proximity `sigma_pos=0.05 m, beta_c=1, dt=0.02`，按准确 reached-reference 帧的原二值 intent gating。这些仍是初始实验值。实际 force/contact 仅作独立物理诊断，不进入 WM target 或 Actor reward。

## 保存与同配置续训

`logs/运行名/` 保存：

- `config.json`、`identity.json`：完整配置、reference/资产/代码/依赖身份。
- `startup.json`、`updates.json`、`learning.jsonl`：阶段、Replay 条件、三套实际 Adam 次数、loss、梯度和显存。
- `prediction.jsonl`：在线 Replay 的 Robot/Box 一步、八步误差及零 residual 运动先验对照；属于拟合诊断，不是 held-out 验证。
- `reset_geometry.json`：原始 reference reset 的 signed distance；`reached_geometry.jsonl`：日志时刻真实 reached state 的 signed distance，未覆盖全部子步。
- `guard.json`、`physical_diagnostics.json`：solver 迭代分布、overflow 检查、独立 contact 统计。
- `checkpoints.jsonl`、`hoi_*.pt`：快照大小、保存耗时和完整状态。不会覆盖已有文件，也没有 `ckpt_latest.pt`。
- 异常时 `failure.txt`、`failure_context.json`、`failure_state.pt`：失败现场，不作为续训 checkpoint。

完整快照包含三个网络/Adam、启动阶段、iteration/UTD、Replay、sampler、仿真 workspace、history、episode identity 和 Python/NumPy/Torch/CUDA RNG。正式命令每 4000 步保存；最终**未来输出**为 `logs/hoi_refined_scratch_v1/hoi_0320000.pt`，本轮没有生成它。

以下 checkpoint 必须由你未来的正式运行产生：

```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python -m supermimic.hoi.train \
  --run-name hoi_refined_resume_v1 \
  --resume logs/hoi_refined_scratch_v1/hoi_0004000.pt \
  --reference outputs/hoi/asset_audit/part_b/reference_refined/reference_refined.npz
```

其余默认值与上述正式候选配置相同（包括 solver 50）。也可以 `bash configs/hoi/refined_full.sh 新运行名 --resume 完整快照路径`。改过参数必须原样重复；环境数、batch、horizon、总目标步数、时间上限、reward、solver、reference 路径表达等均属于严格 resume 契约。代码、Master Plan、物理资产和依赖身份也必须相同。仅输出目录/名称、resume 路径和 check-only 可不同；不兼容即报错，不自动迁移或回退 scratch。

已实测刚加载的完整状态逐位一致；**后续 GPU 物理轨迹不保证逐位复现**。同状态、同动作重复求解也会出现微小浮点差异，详见报告。已完成的 WM 启动不会重做。短验证配置带时间上限/不同快照频率，不能改成正式配置后直接 resume。

## 物理 Actor play

使用明确的、有 Actor 更新的完整 HOI checkpoint。下面的正式 checkpoint 路径是未来输出示例：

```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python -m supermimic.hoi.play \
  --ckpt logs/hoi_refined_scratch_v1/hoi_0320000.pt \
  --output-dir logs/hoi_refined_policy_eval_v1 --max-steps 839 --headless
```

需要桌面窗口时设置 `MUJOCO_GL=glfw` 并使用 `--no-headless`。从 frame 0 reset 后只执行 Actor 动作，关闭观测/动作噪声，遇到失败/timeout/clip-end 结束。`policy_play.json` 保存 reward、Box 误差/速度、左右实际 contact、终止原因和 reached-state 穿透，以及冻结训练 WM 在这条新轨迹上的预测误差；播放不做任何 optimizer 更新。

本轮当前代码可读取的真实小规模快照为 `logs/hoi_validation_20261114/interface_resume_v5/hoi_0000048.pt`（12 次 Actor 更新）；将上面 `--ckpt` 替换成该路径、使用新的输出目录即可手动播放。实测它在第 65 步 robot_failure；64 环境旧版本快照在第 56 步失败，均未完成搬运。64 环境证据对应实现提交 `6799072`；之后的 HOI 学习率接口修复改变了代码身份，旧快照不可绕过契约直接加载，见报告中的源码存档说明。

零 Actor 更新的快照默认拒绝。`--allow-untrained-test-artifact` 只允许 tests 目录内最多 16 步接口测试。策略播放不同于下述 reference viewer；能播放、数值有限或到达末帧均不等于搬运成功，`task_success` 保持未判定。

## Reference viewer

无需 checkpoint，显示现有 reference，不推进物理：

```bash
env -u PYTHONPATH MUJOCO_GL=egl .venv/bin/python scripts/hoi/replay_reference.py \
  --display-only --browser --port 8765 \
  --profile refined-single-body \
  --reference outputs/hoi/asset_audit/part_b/reference_refined/reference_refined.npz
```

使用终端打印的浏览器地址；远端可 SSH 转发 8765。少量 EGL 导出：

```bash
env -u PYTHONPATH MUJOCO_GL=egl .venv/bin/python scripts/hoi/replay_reference.py \
  --display-only --headless --frames 0 406 460 839 --output-dir /tmp/smhoi_reference_frames
```

输出目录必须为新目录。参考逐帧显示没有 Actor，也不证明物理任务完成。

## 资源与验收边界

64 环境、batch256、H32 的有界运行已实际完成三套更新；最终测量和命令见[验证报告](docs/hoi/force_contact/p7/validation_20261114/REPORT.md)。此前 2026-11-13 的 123 个测试只覆盖软件/零更新物理 smoke，本轮真实更新证据单独记录，不混用计数。

当前短跑测得完整 alternating 快照约 269 MB，80 个周期快照约 **21.5 GB**；建议为正式运行至少预留 **25–30 GB**。本机当前剩余空间低于此需求，尚未清理任何用户文件。实测 GPU 总显存约 2.4 GB、进程 RAM 峰值约 3.5 GB；建议可用显存至少 4 GB、系统 RAM 至少 8 GB，并为长时间运行重新监控。eager 全尺寸交替阶段约 0.5 向量步/s，320000 步粗估约 7 天；这是短跑外推，不是长期吞吐保证。

已发现的阻塞：reference/reset 几何穿透（最大 44.1 mm、frame0 足底 12.2 mm）；64 环境续训的 Critic loss 从小于 1 增长至 448.7；新物理播放轨迹的 Box 八步位置预测误差为 16.95 mm，零 residual 先验为 0.98 mm。Robot 八步归一化误差在这一个 episode 上较先验下降，但不能据此声称整体学习或策略表现改善。还缺长期稳定、多 episode/多 seed 验证，以及完整拾取、搬运、放置、释放评测。

已修复 HOI 调度器覆盖显式 `critic_lr` 的接口问题，当前三套初始学习率仍为 2e-4，默认行为不变。[Critic 单参数对照方案](docs/hoi/force_contact/p7/validation_20261114/CRITIC_LR_PROPOSAL.md) 尚待批准，未执行；[reference 几何处置方案](docs/hoi/force_contact/p7/validation_20261114/RESET_GEOMETRY_DISPOSITION.md) 也未执行。

零更新构造检查：

```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python -m supermimic.hoi.train \
  --check-only --env.num-envs 2 --log-root tests/.artifacts/manual_checks --run-name construct_v1
```

TensorBoard：

```bash
env -u PYTHONPATH .venv/bin/tensorboard --logdir logs/hoi_refined_scratch_v1/tb --port 6006
```

唯一 [Master Plan](docs/hoi/force_contact/SMHOI_FORCE_AWARE_IMPLEMENTATION_PLAN.md) §1.1 记录本轮 60 分钟 GPU 授权。主要入口为 `hoi/train.py`、`training_config.py`、`training.py`；实际调度仍在 `training/loop.py`，完整保存恢复在 `hoi/training_checkpoint.py`。本轮不 push，也不自动启动正式长训。
