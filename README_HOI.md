# SMHOI 使用说明

SMHOI 是在线 world-model-based humanoid-object motion imitation：Actor 在真实 MuJoCo Robot–Box 场景中交互，在线 Replay 保存物理转移，Joint WM 学习联合动力学，Actor 经可微 imagination/BPTT 更新，Critic 使用原 SuperMimic TD(λ)。使用已批准的 Actor B、Critic B+ 和 Robot/Box/proximity reward，没有 force/contact prediction head。

**默认从头训练，不需要任何已有 checkpoint。** 本次交付实现接口并做软件及零更新物理测试，没有启动真实训练，也没有产出已训练 policy。当前 refined reference 的 reset 几何仍有未解决的穿透，详见文末；软件测试通过不代表训练稳定或搬运成功。

三类数据用途不同：

| 数据 | 用途 | 首次训练是否需要 |
|---|---|---|
| Reference NPZ + manifest | 示范动作、Robot/Box 同帧 reset、未来目标和 reward/contact intent | 需要，现有文件直接使用 |
| Online Replay | Actor 与物理仿真产生的状态、实际动作、终止标志和诊断 | 启动时为空，由本次运行填充 |
| HOI policy/training checkpoint | 网络、optimizer 和完整训练生命周期 | 首次不需要；训练后用于 play 或可选 resume |

算法和使用方式参考 [SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced) 的 [README](https://github.com/CuboiLeo/SuperMimic-Reproduced/blob/main/README.md)。上游性能数字不是 SMHOI 实验结果。

## 环境

在仓库根目录使用当前已经配置的 `.venv`。以下命令不安装或升级任何依赖。

```bash
cd /home/han/pyx/SMHOI
env -u PYTHONPATH .venv/bin/python -c 'import torch, mujoco, mjlab; print(torch.__version__, mujoco.__version__, torch.cuda.is_available())'
```

物理训练/策略回放需要 CUDA；当前实现支持单进程单 GPU。`env -u PYTHONPATH` 避免系统 Python 包污染；`OMP_NUM_THREADS=1` 限制 CPU 线程；headless 渲染使用 `MUJOCO_GL=egl`。本机已有 `smHOI-track` 入口；文档统一用 `python -m`，可直接使用新增入口而无需重新安装包。

## Reference 与物理配置

- NPZ：`outputs/hoi/asset_audit/part_b/reference_refined/reference_refined.npz`
- Manifest：`outputs/hoi/asset_audit/part_b/reference_refined/reference_manifest.json`
- SHA256：`f6ba446c16e71ab3cffb09ec015ca9a224305f6ac82d01f3631d4d5990a5b5d7`
- 命名配置：`--profile refined-single-body`，默认启用。Box 使用既有 `outputs/hoi/contact_supervision/p6_closure/candidates/single_body.json` 的几何和惯性；Robot、support、actuator/action mapping 不变。
- 840 帧、50 Hz；frame 0 到 frame 839 有 839 个控制间隔（16.78 s）。单位 m/s/rad，四元数 wxyz。

无需重新转换。下面只检查哈希、数组 schema/有限值、manifest 和配置绑定：

```bash
sha256sum outputs/hoi/asset_audit/part_b/reference_refined/reference_refined.npz
env -u PYTHONPATH .venv/bin/python - <<'PY'
from supermimic.hoi.reference import ReferenceClip
from supermimic.hoi.training_config import HOITrainConfig
from supermimic.hoi.training_scene import profile_manifest, validate_reference_binding
c = HOITrainConfig()
c.validate()
r = ReferenceClip(c.reference)
validate_reference_binding(c.profile, r)
m = profile_manifest(c.profile)
print(r.n_frames, r.fps, r.manifest['sha256'], len(m['object']['geoms']))
PY
```

该配置拒绝回退到旧 `reference/reference.npz` 或 candidate reference。MotionLibrary、reset cache、registry、未来目标、reward、contact intent、checkpoint 和两个 viewer 均绑定同一 refined reference。Manifest 的 `training_ready=false` 和历史 FAIL 保持原样；命名配置记录本次实验选择，不声称全局物理 PASS。已退休 V3 不再是运行依赖。

## Reference motion play：无需 checkpoint

浏览器中查看参考动作：

```bash
env -u PYTHONPATH MUJOCO_GL=egl .venv/bin/python scripts/hoi/replay_reference.py \
  --display-only --browser --port 8765 \
  --profile refined-single-body \
  --reference outputs/hoi/asset_audit/part_b/reference_refined/reference_refined.npz
```

打开终端打印的本地地址，使用播放/暂停、进度条和视角控制。服务器上运行时可通过 SSH 转发 8765 端口。

无桌面环境导出少量参考帧（输出目录必须不存在）：

```bash
env -u PYTHONPATH MUJOCO_GL=egl .venv/bin/python scripts/hoi/replay_reference.py \
  --display-only --headless --frames 0 406 460 839 \
  --output-dir /tmp/smhoi_reference_frames
```

本地桌面可去掉 `--headless/--output-dir/--frames` 并指定 `MUJOCO_GL=glfw`；原生 viewer 支持 `--autoplay`、`--close-after-s 2`。此入口逐帧赋参考姿态后 `mj_forward`，**不推进物理，不执行 Actor，不表示训练后的策略效果**。

## Train from scratch

下面是首次训练命令。**本次没有执行它；由你之后手动启动。** 不含 `--resume` 或 `--initial-wm`：

<!-- scratch-command: parsed by tests/hoi/test_readme_hoi.py -->
```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python -m supermimic.hoi.train \
  --run-name hoi_refined_scratch_v1 --log-root logs --device cuda:0 \
  --profile refined-single-body \
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

运行目录必须是新的；重复实验请更换 `--run-name`。不自动查找、选择最新或加载旧模型。

启动流程在一次进程中完成：

1. Scratch 初始化：Joint WM 的最终 residual 层置零，保留 constant-velocity prior 和原 termination 初始化；Actor 使用现有 B 架构初始化、action 输出层 gain 0.01；Critic B+ 独立初始化。三个 Adam、sampler、Replay 和进度全新创建。
2. Actor 在线物理交互并填充 Replay；前 40 个控制步只采集。Replay 必须有至少 32 个有效连续八步窗口和 32 个可采样 policy start。
3. 内部 `wm_bootstrap` 阶段只执行 WM 更新，共 50 次 `WMTrainer.update`；每次包含 10 个 minibatch。Actor 继续采集数据。
4. 自动进入 `alternating`：每轮 WM 更新后执行原 Actor BPTT 和 Critic TD(λ)，WM 持续学习。没有外部 WM 预训练步骤、PD 采集前置流程或审批文件。

上述数量是**未经学习实验调优的初始配置**；它们只是可解释的调度条件，不是 WM 预测质量证书。若 Replay 长时间不满足条件，2000 步的等待上限会报错，打印实际窗口数与 policy start 数。检查 `startup.json` 中失败/timeout/clip-end 计数、`reset_geometry.json` 和失败日志；不要求策略先学会抓取、搬运或释放。

### 参数与来源

| 参数 | 含义与来源 |
|---|---|
| `env.num-envs=64` | 并行世界数，保守的工程初始值；本次物理 smoke 仅验证 2 环境 |
| `schedule.total-iterations=320000` | 向量控制步数，每次推进所有环境一次；不是 minibatch 数；正常情况下总 env transitions 约为它乘环境数 |
| `updates-per-env-step=0.4` | 原 SuperMimic UTD 累加器，每个控制步累计 0.4 次更新；每次 WM update 内再执行配置的 minibatch 数 |
| `replay-capacity=256` | 每环境 ring 行数；terminal state 额外占一行；不是离线数据集 |
| WM batch 256 / 10 batches | 原 WM 优化器机制；batch 为本项目资源初始值 |
| Actor batch 256 / H=32 | H 沿用原 imagination horizon；γ=0.99、λ=0.95、Critic 5 epochs × 4 minibatches 沿用原实现 |
| 学习率 | 三套初始 lr 均为 2e-4，沿用原配置和 cosine schedule；本配置关闭 compile，便于先验证接口 |
| `env.sampling-mode` | `return` / `uniform` / `start`。前两者从整段 reference 采样，同帧初始化 Robot 和 Box；`start` 使用 `--reset-frames`，默认 `(0,)` |
| Return sampler | 原 `MultiClipMotionCommand`：1 秒 bins，50% uniform + 40% softmax(−25 G/Gmax) + 10% memory，return EMA 0.05、memory EMA 0.01 |
| `sampler-g-max=None` | 从理论正 reward 上界自动计算 `max_step_reward/(1−γ)`；默认 0.18/0.01=18。可显式覆盖；不是测得最优 return 标度 |
| 噪声 | 执行动作噪声 std=0.1，记录 clean/executed action；观测噪声默认关闭，reset 扰动和 domain randomization 关闭以保持同步 reset |

Reward 公式不变。Robot 项来自 `envs/mdp/rewards.py::TrackingRewardCfg`，每步正奖励上界 0.1。Box 使用原 Scheme A 的 `dt × weight × exp(−error²/sigma²)`：位置/姿态/线速度/角速度的初始 sigma 为 `(0.2 m, 0.4 rad, 1 m/s, 2 rad/s)`，weight 为 `(1, 1, 0.5, 0.5)`。Proximity 参考 [P4 距离审计](docs/hoi/force_contact/p4/CONTACT_TARGET_AUDIT.md) 的 0.0252/0.0423/0.0629 m 候选范围，选择方便解释的初始值 `sigma_pos=0.05 m`、`beta_c=1`，按当前目标 reference 帧的二值 intent gating；dt=0.02。参数见 `training_config.py::initial_reward`，CLI 为 `--reward.box-sigmas`、`--reward.box-weights`、`--reward.sigma-pos`、`--reward.beta-c`。这些 Box/proximity 数值是初始实验配置，未经过学习实验标定，不能称为最优值。

### 日志与保存

输出在 `logs/hoi_refined_scratch_v1/`：

- `config.json`、`identity.json`：实际配置、模型/物理/reference/代码/依赖身份。
- `tb/`：TensorBoard；`startup.json`：当前阶段、Replay 条件和 optimizer 次数。
- `reset_geometry.json`：当前采样模式覆盖的 reset 帧的原生 signed contact distance；不会修改姿态。
- `guard.json`、`updates.json`、`physical_diagnostics.json`：solver 证据、真实更新次数、保留 Replay 的独立 contact 统计。
- `hoi_0004000.pt`、`hoi_0008000.pt`……：每 4000 iteration 一个完整快照。正常完成 320000 步时最终文件为 **`logs/hoi_refined_scratch_v1/hoi_0320000.pt`**；提前按 max-hours 正常退出时保存实际 iteration 对应文件。**这些是将来训练生成的路径，本次不存在 trained checkpoint。**

HOI 不创建或自动查找 `ckpt_latest.pt`，文件使用独占创建，已有 checkpoint 不覆盖。快照包含 WM/Actor/Critic、三个 Adam、启动阶段、iteration/UTD/update counts、在线 Replay、sampler、物理 workspace、history、episode/clip identity 和 Python/NumPy/Torch/CUDA RNG。完整物理快照比纯 Actor 权重大；应预留磁盘空间。异常时保存 `failure.txt`、失败现场和 solver 信息；失败现场不是可续训 checkpoint，恢复使用此前正常保存的完整快照。

## Policy play / evaluation

**等待你实际训练产生包含 Actor 更新的 HOI checkpoint 后再使用以下命令。** 路径是未来文件示例：

```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python -m supermimic.hoi.play \
  --ckpt logs/hoi_refined_scratch_v1/hoi_0320000.pt \
  --output-dir logs/hoi_refined_policy_eval_v1 --max-steps 839 --headless
```

交互窗口（需要可用桌面显示）：

```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 MUJOCO_GL=glfw .venv/bin/python scripts/hoi/play_policy.py \
  --ckpt logs/hoi_refined_scratch_v1/hoi_0320000.pt \
  --output-dir logs/hoi_refined_policy_view_v1 --max-steps 839 --no-headless
```

它从明确指定的 HOI checkpoint 加载 Actor，在真实 Robot–Box 场景从 frame 0 执行策略，沿用训练 history、Robot/Object goal 和原生 actuator/action mapping；评测关闭动作/观测噪声。只在 episode reset 时赋参考初态，之后动作均来自 Actor。显示窗口复制实际达到的状态，仅用于渲染。此入口与 reference viewer 是不同代码路径，也不使用 robot-only `scripts/play.py`。

`policy_play.json` 保存每步 reward、Box 高度/位置误差/垂直速度、reference intent、独立标定的左右实际 contact 和终止原因；遇到跌倒/timeout/clip-end 即记录并结束，NaN/Inf/求解错误单独报错。`contact_missing_diagnostic` 只表示 intent ON 时未测得双手接触。默认未增加新的 Box 失败 termination 或更改 imagination 边界；抓取失败、掉箱需要结合 Box 误差和独立 contact 诊断分析，不能将“有限数值”或“播到结尾”当作搬运成功。`task_success` 保持未判定。这是最小物理回放接口，完整多次评测/成功率尚未执行。

默认拒绝零 Actor 更新的 checkpoint。`--allow-untrained-test-artifact` 仅供接口测试，限制最多 16 步且输出必须在 `tests/` 下，不能把测试快照用作正式训练输入。

## Optional resume

只有中断后需要恢复时才使用。首次训练不需要本节。以下命令中的 `hoi_0004000.pt` 必须由你前次运行实际保存：

```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python -m supermimic.hoi.train \
  --run-name hoi_refined_resume_v1 \
  --resume logs/hoi_refined_scratch_v1/hoi_0004000.pt \
  --reference outputs/hoi/asset_audit/part_b/reference_refined/reference_refined.npz
```

这里其他参数的默认值与首次训练命令一致。若首次运行改过其他参数，恢复命令必须重复那些修改。`--run-name` 使用新目录，保留旧 run；总 iteration 表示原目标总量，不是额外步数。已完成的 WM-only 启动不会无条件重做。

精确恢复要求同配置（包括环境数、batch、总步数、lr schedule、reference 路径表达、reward/sampler 和阶段参数）、代码指纹、Master Plan、物理资产和依赖；允许改变输出目录/名称、resume 文件路径和 check-only。不同配置不做隐式转换。文件不存在、robot-only/旧 schema/迁移 checkpoint、输入身份变化均明确报错，不回退 scratch。`--initial-wm` 不属于默认或续训路线；现有权重迁移工具保留为独立工具。

## TensorBoard

```bash
env -u PYTHONPATH .venv/bin/tensorboard --logdir logs/hoi_refined_scratch_v1/tb --port 6006
```

观察 `startup/phase`（0=collect、1=wm_bootstrap、2=alternating）、`updates/wm|actor|critic`、`replay/windows8`、reward/WM/policy 指标和 episode return。启动阶段 Actor/Critic 更新数为 0 是正常调度；不应将此时的快照当作已训练 policy。

## Plan 与代码导航

唯一 Master Plan：[SMHOI_FORCE_AWARE_IMPLEMENTATION_PLAN.md](docs/hoi/force_contact/SMHOI_FORCE_AWARE_IMPLEMENTATION_PLAN.md)，§1.1 记录本次默认 scratch、可选 resume 和只实现/测试不训练的决定。历史报告继续保留。

| 接口 | 实现 |
|---|---|
| Train / config | `src/supermimic/hoi/train.py`、`training_config.py`、`training.py` |
| 原训练循环 / 启动状态 | `src/supermimic/training/loop.py`、`hoi/startup.py` |
| Policy play | `src/supermimic/hoi/play.py`、`scripts/hoi/play_policy.py` |
| Reference play | `scripts/hoi/replay_reference.py`、`hoi/reference_viewer.py`、`hoi/reference_browser.py` |
| Joint WM / 原 WM updater | `hoi/world_model.py`、`hoi/wm_trainer.py`、`world_model/wm_trainer.py` |
| Actor B / Critic B+ / BPTT / TD(λ) | `hoi/policy.py`、`hoi/policy_features.py`、`hoi/imagination.py`、`imagination/policy_trainer.py` |
| Reward / reference registry | `hoi/tracking_reward.py`、`hoi/rewards.py`、`hoi/reward_reference.py` |
| Reset / Replay / terminal capture | `hoi/state.py`、`hoi/training_env.py`、`hoi/recorder.py`、`hoi/replay.py`、`data/replay_buffer.py` |
| 完整保存恢复 / 数值和 solver 检查 | `hoi/training_checkpoint.py`、`hoi/training_safety.py` |
| Reset 几何诊断 | `hoi/reset_diagnostics.py` |

`hoi/...` 路径均相对于 `src/supermimic/`。

## 验证范围与已知限制

可进行零更新构造检查（用新测试目录，不执行 `Trainer.run`，不保存 checkpoint）：

```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python -m supermimic.hoi.train \
  --check-only --env.num-envs 2 --log-root tests/.artifacts/manual_checks --run-name construct_v1
```

本次实际验证：默认 CLI 配置解析/校验、指定 NPZ/manifest 哈希、实际两环境 scratch 构造（`torch.load` spy 确认没有旧权重读取）、空 Replay/新 optimizer/零计数、原 `Trainer.run` mock 调度、阶段序列化/恢复、两环境 16 步零更新采集、从第 8 步快照恢复再执行 8 步、单环境 8 步未训练 Actor 的独立 policy play、reference browser/EGL 有界加载与渲染、robot-only 回归。所有测试快照仅位于 `tests/.artifacts/`，标注 `untrained/test artifact`。物理 smoke 用 spy 禁止 Adam.step；不会生成正式训练输入。

README 首次训练命令有独立解析测试，并走相同构造路径做两环境零步检查；这不等于执行过 64 环境完整训练。不能仅凭 `--help` 成功声称训练通过。

2026-11-13 验证结果：**123 个独立测试分批通过**。其中最后一轮接口/恢复/安全/文档/reference 回归为 68 passed；其余 55 项覆盖 Replay、reward/reference、imagination 边界、Joint WM、Actor/Critic 和 CPU/CUDA eager/compiled 合成前向/梯度一致性。初次 browser 发现旧构建器只接受七块碰撞体，修复为复用既有 single-body 构件后复测通过。另实际完成 reference 帧 0/460 的 EGL 导出和 840 帧静态变换核对，physics steps=0。静态检查与 `git diff --check` 通过。合成 Adam 测试仅验证 optimizer 序列化/调度，不使用采集的物理数据。

本地证据保存在 `tests/.artifacts/VALIDATION.json`、`tests/.artifacts/validation_logs/` 和 `tests/.artifacts/final_interfaces/`，这些均为被 Git 忽略的测试产物。复查本次新增接口测试（输出目录每次换新）：

```bash
mkdir -p tests/.artifacts
env -u PYTHONPATH OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/pytest \
  tests/hoi/test_scratch_lifecycle.py tests/hoi/test_readme_hoi.py \
  --basetemp tests/.artifacts/manual_interfaces_v1 -q
```

**未解决的 reset 几何：** 原生模型 `mj_forward` 对完整 refined reference 的静态扫描发现最深穿透约 **44.1 mm**（frame 406，右手–single-body Box）；frame 0 足底–地面约 **12.2 mm**。这属于真实初始化几何问题，不能由 solver 数值有限证明通过。运行会打印 `UNRESOLVED reset overlap` 并保存具体 frame/geom/signed distance。当前保留原始姿态和碰撞，1 mm 为既有 P6 诊断报告带，并非自动修正/过滤阈值；未偷偷筛除难帧、抬高 Robot/Box 或改 reward。历史物理 FAIL/限制未被重新标成 PASS。手动训练即使能启动，仍可能受这些初态影响。

未执行：真实 WM warm-up、物理数据上的 WM/Actor/Critic optimizer 更新、短训练调参、长期训练、学习稳定性验证、已训练策略的完整任务评测及成功率统计。早期抓取失败、跌倒、掉箱是待学习表现；无效求解、NaN/Inf 是应中止的数值问题，二者分开记录。初始超参、默认 64 环境资源使用和真实收敛效果尚未由学习实验验证。
