# 2026-11-14 真实训练闭环验证

**结论：有界工程链路验证通过，full-train ready = false。** 已运行真实 Actor 在线采集、内部 WM 启动、三套更新、完整保存、严格同配置续训和物理 Actor 播放。当前仍有 reference 几何穿透、Critic loss 显著增长、Box 预测劣于运动先验以及长训磁盘不足。没有启动正式长训，没有 push，没有修改受保护 reference、碰撞资产、reward、Actor B、Critic B+、Joint WM 或其优化算法。

基线 `c6225c1c0a66591dbb42e52760cc1119aad3414d`；主要实现提交 `6799072`。之后仅增加 HOI 学习率配置接口修复及审计/文档。当前源码身份、每轮完整配置、checkpoint SHA256、失败记录摘要和命令见 [EVIDENCE.json](EVIDENCE.json)。本文件是实验报告；唯一执行授权仍为 [Master Plan §1.1](../../SMHOI_FORCE_AWARE_IMPLEMENTATION_PLAN.md)。原有 `.vscode/`、本地 refined reference、`.venv`、所有历史证据均保留。

## Reset 穿透来源

使用指定本地 `reference_refined.npz` 和对应 manifest、`refined-single-body` profile，未转换或回退：

- NPZ SHA256：`f6ba446c16e71ab3cffb09ec015ca9a224305f6ac82d01f3631d4d5990a5b5d7`。
- 现有 single-body Box 候选 SHA256：`6c3f1a9c601336cb264cd0f979ec4d039ed380e76df6ee1a0a33358d5a82e331`。
- 对全部 840 帧独立按原生关节名称赋值 NPZ pose，与训练 reset cache 的 qpos 和 signed contact distance **完全相同**。
- 原生 FK 与 reference 存储位置最大差异 `8.3563e-8 m`；两世界下 frame 0/406/460/839 的状态往返误差小于 `1e-6`。
- 最大穿透：frame 406 右手–Box **44.1007 mm**；足底–地面最大为 frame 155 的 **24.7182 mm**；frame 0 足底为 **12.2356 mm**。
- 训练可 reset 的 839 帧中，837 帧超过原有 1 mm 报告带。阈值未放宽。

这些结果支持穿透来自固定 reference pose 与原生碰撞几何的兼容问题，而非 reset cache、环境平移或关节映射错误。没有抬高 root、过滤 frame、移除碰撞或更改目标。实际训练 reached state 的稀疏 signed-geometry 采样仍有穿透；它只覆盖日志时刻，不能代替全部子步的几何有效性验证。有限 loss、梯度、状态或 solver 返回均不代表几何有效。

原始报告：`logs/hoi_validation_20261114/reset_source_v1/reset_source.json`。受保护修改的可审查但未执行方案见 [RESET_GEOMETRY_DISPOSITION.md](RESET_GEOMETRY_DISPOSITION.md)。

## 实际故障和实现修复

64 环境、原 solver iteration 上限 10 的两轮 scratch 均在第 10 个控制间隔前后触发 `ITERATIONS=512`，第二轮定位到 world 63、reference frame 454、episode age 9，尚未进行任何 Adam 更新。`failure_state.pt`、`failure_context.json`、guard 和完整日志均保留。

将 HOI 训练的显式 solver iteration 资源上限设为 50，并纳入配置和物理身份。50 上限的 64 环境运行实际使用最高 12 次迭代；原上限确实不足。求解容差、line-search 上限 20、dt=0.005、四子步、碰撞资产及 fail-on-any-overflow 检查均未改变。通用场景标定及 Robot-only 默认仍为 10。这是求解资源修复，不是穿透修复。

其余实现变化：记录各次真实更新、梯度、预测误差、稀疏 reached geometry、checkpoint 大小和耗时；修复共用控制台误取 Robot-only metric key、把未测量值显示为 n/a。预测报告保存/恢复 Torch 和 CUDA RNG，不影响后续采样。

最后发现共用 `PolicyTrainer.set_lr` 会用 Actor scheduled LR 覆盖显式 Critic LR。新增 HOI-only override 保持配置的 Critic/Actor 比例；当前两者同为 `2e-4`，继续走原 setter 的相同路径。没有改变当前超参数，Robot-only setter 不变。三种 schedule fraction 的合成测试及实际默认路径闭环通过。

整个实现复用原 `Trainer.run`、Replay/window/history、WMTrainer.update、Actor BPTT、Critic TD(λ)、Adam 和完整 checkpoint。线上数据均来自 scratch/current Actor 与原生 actuator；没有 reference-PD、离线采集或旧权重初始化。

## 逐级真实实验

下表中的 WM/Actor/Critic 为该 checkpoint 的累计 **Adam step**，不是所有实验加总；resume 行包括其起点已经完成的更新。每轮具体命令、参数、失败和耗时见 [autoresearch.jsonl](autoresearch.jsonl) 与 [命令转录](autoresearch.sh)。变更环境数、batch、horizon、solver 或其他配置均开新 scratch。

| 运行 | env / batch / H | 控制步 | WM / Actor / Critic | 结果 |
|---|---|---:|---:|---|
| train_small_v1 | 2 / 8 / 4 | 64 | 21 / 19 / 380 | scratch、内部启动与交替更新 |
| resume_small_v1 | 同配置 | 32 → 64 | 21 / 19 / 380 | 真实续训；终态逐位比较 FAIL，原因见下文 |
| train_medium_v2 | 16 / 64 / 16 | 96 | 60 / 20 / 400 | 正常保存 |
| train_scale64_v2 / solver10_failure_v4 | 64 / 256 / 32 | 最后完成 9 | 0 / 0 / 0 | solver iteration 上限 FAIL，两次复现 |
| train_scale64_solver50_v3 | 64 / 256 / 32 | 250 | 840 / 34 / 680 | 资源上限修复后通过 |
| train_final_v4 | 64 / 256 / 32 | 250 | 840 / 34 / 680 | 提交 6799072 的最终 scratch |
| resume_final_v4 | 同配置 | 250 → 350 | 1240 / 74 / 1480 | 保存恢复闭环通过；Critic 稳定性未通过 |
| boundaries_final_v4 | 2 / 8 / 16 | 48 | 14 / 12 / 240 | timeout/clip-end 实际触发与审计 |
| interface_current_v5 | 2 / 8 / 4 | 48 | 14 / 12 / 240 | 最终学习率接口修复后的默认路径 |
| interface_resume_v5 | 同配置 | 24 → 48 | 14 / 12 / 240 | 严格续训通过；额外 9 / 9 / 180 次更新 |

64 环境 scratch 的启动在 iteration 164 转入 alternating；之前有 50 次 WM update，每次 10 个 minibatch，持续由 Actor 采集。两个 `max_hours=0.05` 进程在日志边界正常结束于 250 和 350。初始化、日志边界和保存使完整进程耗时分别为 230.94 s、208.57 s，均按完整时间计入预算，没有把配置时间上限当作 GPU 消耗数值。

## Loss、梯度与预测：不能宣称学习稳定

64 环境 WM 聚合 loss 从首条 51.50 到 iteration 349 的 5.05，在线 Replay 拟合改善；数据分布不断变化，这不是独立测试集学习曲线。所有三套更新器实际产生有限梯度并被既有 clipping 处理；最后三套 norm 均约 1。续训的 Actor clipping 前梯度最大约 12.79。250 → 350 的 WM、Actor、Critic 参数 L2 变化分别约 9.477、5.229、11.278，确认模型不只是推进计数。

**Critic 出现明显漂移**：前 250 步 loss 在 0.128–0.828，续训到 iteration 349 达到 **448.6768**，value mean 为 **-64.6209**，target mean **-60.2372**，imagined mean reward 为 +0.0542。有限值和 clipping 通过不能作为学习稳定性 PASS。

冻结 350 checkpoint，以独立 CPU double 精度逐轨迹 recurrence 核对现有 TD(λ) 和 valid/bootstrap masks，最大绝对差 `6.8393e-5`，通过 `rtol=1e-5, atol=1e-5`，没有 optimizer 更新。该诊断同时发现 imagined reward 最小 -60.80、Critic feature 绝对值最大 162.99；这些是风险证据，尚未证明唯一因果。未改 reward、损失或 TD 算法。

在全新 frame-0 物理 Actor episode 上冻结 WM（无拟合、单 episode、32 个抽样窗口），比较零 residual 动力先验：

| 八步误差 | 训练 WM | 零 residual 先验 |
|---|---:|---:|
| Robot normalized MSE | 0.10143 | 0.69914 |
| Box position | 16.950 mm | 0.983 mm |
| Box orientation | 0.04410 rad | 约 0 |
| Box linear velocity | 0.18129 m/s | 0.00747 m/s |
| Box angular velocity | 0.44085 rad/s | 约 0 |

只支持这个 episode 的 Robot 预测优于先验；Box 预测更差，而且 Box 基本静止。没有证据支持整体 WM 达到可用精度、Actor 性能改善或搬运成功。Replay 预测文件明确标注为在线拟合诊断；上述播放样本未用于训练，但一个短 episode 也不足以代表泛化。

## Replay、终止边界与恢复

CPU 完整 checkpoint 审计复用实际 `_gather_windows` 和 WM physical-state contracts，遍历所有可采样八步窗口，检查 episode/frame 连续性、terminal 状态不作为有效 transition、有限 state/action、reason/terminal/bootstrap 一致性，并检查每个 Adam parameter state 的 step。

- 250：13855 个窗口、15988 个 retained transition；144 robot_failure、101 clip_end。
- 350：14036 个窗口、16135 个 retained transition；125 robot_failure、124 clip_end；Adam step 与 1240/74/1480 完全一致。
- 边界新实验：frame 0/830、episode length 0.24 s，实际得到 4 timeout 和 4 clip_end；28 个窗口全部通过，timeout 保留 bootstrap，physical failure/clip_end 不 bootstrap。
- 最终代码小实验：66 个窗口、96 个 transition、2 robot_failure 全部通过。

上述为 Replay 时序和数值完整性，不是几何有效性。计数描述各快照保留的数据，不是实验全过程终止总数。

64 环境 snapshot 刚恢复时，573 个 tensor / 268654120 tensor bytes **逐位一致**，还比较其余嵌套配置、网络、Adam、Replay、sampler、仿真 workspace、history、episode、UTD、启动状态及 RNG；仅排除重新运行时累积的 guard 审计事件。当前代码的 2 环境加载也通过 573 tensor 比较。

**后续轨迹不逐位确定**：首次小实验连续训练与 resume 的最终 qpos 最大差 5.13e-5，比较按原标准保留 FAIL。相同 snapshot 连续恢复两次、完全不做 optimizer 更新，64 环境第一步 clean/executed action 完全相同，qpos 仍相差 1.19e-7、qvel 相差 9.54e-6；8 步 qpos 最大约 2.21e-6。这表明精确状态恢复与原生 GPU 物理轨迹逐位确定性是不同要求，不能通过放宽比较把后者标为 PASS。

## 物理播放和可立即审查的文件

`play_final_v4`：使用真实 74 次 Actor 更新的 350 checkpoint，从 frame 0 运行 **56 步后 robot_failure**；reward sum 5.9384、Box 最大跟踪误差 12.40 mm、reached 穿透最大 12.34 mm；三套 optimizer 更新均为 0，`task_success=null`。

`interface_play_v5`：当前代码、12 次 Actor 更新的 checkpoint，**65 步后 robot_failure**，同样没有任务成功证据。它与前者训练配置不同，65 与 56 不能用来宣称策略改善。

当前代码可直接手动播放（输出目录须不存在）：

```bash
cd /home/han/pyx/SMHOI
env -u PYTHONPATH OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python -m supermimic.hoi.play   --ckpt logs/hoi_validation_20261114/interface_resume_v5/hoi_0000048.pt   --output-dir logs/hoi_manual_review_v5 --max-steps 839 --headless
```

当前可读快照 SHA256 为 `55d8e0e2a3ec4aeb92bcff7ed8cb1e7046c40941bfdd1fff593b01862d48f954`。64 环境快照对应旧实现提交 6799072，之后接口修复改变了代码身份；不可在当前源码下绕过 loader 检查。`source_v1`–`source_v5.tar.gz` 保留各版本源码，`source_v4` 对应 6799072，`source_v5` 与当前 checkpoint 的 source hash 已核对。文档和测试文件变更不改变源码指纹。

所有原始 checkpoint、逐步日志、config/identity、failure snapshots 均留在 `logs/hoi_validation_20261114/`。它们及 reference 是本地数据，Git 克隆不会提供。提交中的 EVIDENCE 包含关键数值、精确命令、快照和原始文本 SHA256；不要把代码提交当作大型权重/数据备份。

## 正式候选配置、资源与预算

[README_HOI.md](../../../../../README_HOI.md) 和 [configs/hoi/refined_full.sh](../../../../../configs/hoi/refined_full.sh) 提供完整 scratch、严格 resume 和 play 命令。手动候选入口：

```bash
bash configs/hoi/refined_full.sh hoi_refined_scratch_v1
```

该命令**未在本轮启动正式长训**，也未达到可建议直接长训的验收状态。64 env、Replay256、WM/Actor batch256、Actor H32、WM10 minibatches、startup50、UTD0.4、total320000、solver50，三套初始 LR 仍为 2e-4。与实测 v4 配置的数值差异仅为正式 checkpoint interval4000（验证250）和 max_hours=None（验证0.05）。配置比较文件及 README/脚本/defaults parity 测试通过。

不得把验证 checkpoint 改 total target、env、batch、H、max_hours、solver、reference 路径或其他配置后直接 resume。改变配置开新 scratch；旧状态不迁移。正式未来 checkpoint 路径在 README 明确标为未来输出。

RTX 4090 D 24 GB 实测：GPU 总占用峰值 2460 MiB（含进程外基线）、进程 RAM 峰值 3720900 KiB；完整 64 环境 alternating snapshot 268907909 bytes，保存约 0.195 s。建议至少可用 GPU 4 GiB、系统 RAM 8 GiB，并重新监控长时运行。eager alternating 约 0.5 向量步/s，320000 步粗估约 7 天；短跑外推不保证长期吞吐。80 个周期 checkpoint 约 21.5 GB，至少预留 25–30 GB；当前仅约 1.3 GB 可用，未清理用户文件。

本轮预算上限 3600 s；保守累计 **964.843 s（16.08 分钟）**。记录器对每个 GPU 相关子进程的完整寿命计费，包含导入、CPU 工作、测试和失败；全部串行受预算锁约束。另行 CPU snapshot 审计明确检查未初始化 CUDA。停止原因是未解决的几何/学习阻塞和待批准的受保护参数实验，**不是预算耗尽**。

## 回归、授权边界和未完成项

本轮 **50 个不同测试分批通过**：原本轮 47 个加新学习率比例 3 个。首次 suite 为 46 pass/1 mock failure；第一次修复又发现 mock 缺 Path import，最终单项重跑通过。全部失败日志保留。最后相关套件 7 passed、最后 README/provenance/LR 套件 9 passed；这些是重复验证，不能额外计入 50。历史 2026-11-13 的 123 项未作为本轮完整套件重跑，旧 VALIDATION 已备份；新 `tests/.artifacts/VALIDATION.json` 指向本轮证据。Ruff 与 diff whitespace 检查通过。

自动审批审查拒绝过“接口修复 + 把 Critic 初始 LR/default/full config 从 2e-4 改为 2e-5”的组合操作，理由为后半部分属于需明确批准的受保护训练参数修改。被拒命令没有执行。之后仅实施了默认路径不变的接口修复，并提交了 [具体单参数对照方案](CRITIC_LR_PROPOSAL.md)，目前仍待用户批准；未运行较低 Critic LR 实验，也未修改正式候选参数。降 LR 不保证解决问题，尤其不会解决几何穿透。

本轮可交付的是有真实证据的工程实现、可审查配置和准确失败边界。缺口是几何兼容问题、Critic/WM 稳定与泛化、多个独立物理 episode/seed 的策略收益、完整拾取搬运放置释放评测及长训资源准备。最终验收和 push 由用户决定。
