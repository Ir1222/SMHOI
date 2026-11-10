# SMHOI HDMI 物理资产审计（Part A M0–M6）

**结论：审计完成，物理接触和支撑门禁未通过，暂停在 M6。** 未启动 HOI 训练，未实现 M7–M9、训练入口、WM 或策略迁移；未生成任何 NPZ，也未重算机器人 reference。

用户明确的九项审计范围与计划 Part A 一致。计划 Part D 对 M0–M6 使用了另一套编号，将实现和训练混入其中；本次按用户限定和 Part A 执行，不据此扩展工作范围。

## 1. 位置、来源与 reference 完整性

目标为 `/home/han/pyx/SMHOI`，不是同级另一个 `SuperMimic-Reproduced` 目录。源仓库 `/home/han/pyx/humanoid-amp-ppo` 全程只读。初始目标 HEAD `ba61053`，源 HEAD `114f766`；完整 SHA、branch、status 和 53 个基线文件 SHA256 见 [baseline.json](baseline.json)。两个仓库及上级目录未发现适用的 AGENTS.md。

完整读取了用户提供的 `SMHOI_TRACK_IMPLEMENTATION_PLAN_V3.md`，该文件仍保持原始未跟踪状态。源仓库原有 `MUJOCO_LOG.TXT`、`docs/hoi/audits/hoi_research_freeze_20261107.md` 保留；没有 reset、clean、stash、checkout、checkpoint 修改或实验记录覆盖。目标仓库没有 Git 作者配置，提交使用源仓库已有的 `shuayyy <nshuaibmohamed@gmail.com>`，仅逐命令指定，没有改全局配置。

两份输入为 `/home/han/pyx/hdmi_bread_box_robot_only_50hz_supermimic_pyx.npz` 和 `/home/han/pyx/hdmi_bread_box_hoi_v1.npz`。840 帧、50 Hz；帧数/FPS 为 16.80 s，最后采样时刻 16.78 s。六个机器人数值数组 dtype、shape、值、数组字节哈希全部一致，最大数值差 0。joint/body 名称和顺序一致，字符串存储宽度分别 U26/U25 与 U64，所以不宣称字符串数组字节一致。全部 key、范围、有限性、四元数范数和名称顺序见 [npz_audit.json](npz_audit.json)。

`object_contact` 是期望接触意图 `[260,550)`，共 290 帧，不是双手实测接触。对象位置和 wxyz 姿态直接来自官方 motion 的 bread_box body，转 float32 后相等。现有 hoi_v1 对象速度是 `pose_fd_link_world_v1` 派生量；与官方原始速度的最大差分别为 2.71e-6 m/s、0.094204 rad/s，历史派生 manifest 的输出 SHA 与输入 NPZ 完全一致。原始录制速度的 COM/root 来源仍未确证；当前派生字段则有世界坐标根点差分来源。两套 phase metadata 都是既有诊断启发式，不是官方实测标签。源 `reference.py::derive_reference()` 会重写机器人速度，本次没有调用。

## 2. 真正的 bread box 几何和物理参数

权威 XML：`/tmp/supermimic_hoi_audit_20261104/HDMI/active_adaptation/assets_mjcf/objects/bread_box/bread_box.xml`。本地 HDMI checkout 的 HEAD、状态、文件哈希均已记录。raw USD 另行重新解析，并与旧 manifest 完全相符。XML 与 USD 碰撞顶点范围也有独立数值交叉验证。

所有尺寸为米，以下均为 box-root 局部位置及 **半尺寸**；七个 geom 姿态均为单位 wxyz 四元数，附着在同一个 free rigid body 上。

| Geom | 局部位置 | 半尺寸 |
|---|---|---|
| bottom | (0,0,0) | (.09,.17,.01) |
| side1 | (-.08,0,.13) | (.01,.17,.13) |
| side2 | (.08,0,.13) | (.01,.17,.13) |
| side3 | (0,.16,.13) | (.09,.01,.13) |
| side4 | (0,-.16,.13) | (.09,.01,.13) |
| handle_left | (0,.17,.21) | (.03,.02,.01) |
| handle_right | (0,-.17,.21) | (.03,.02,.01) |

联合 AABB 为 `[-.09,-.19,-.01]` 到 `[.09,.19,.26]`，完整尺寸 **.18 × .38 × .27 m**；不含把手的主体为 **.18 × .34 × .27 m**。几何中心 `[0,0,.125]`，不是 box root。源诊断 OBB 的下边界为 0，漏掉底板下方 1 cm；本次没有用它作碰撞体，也没有采用 Lift 的单 cuboid。

名义质量 **2 kg**，COM/root 偏移 **[0,0,-.2] m**，主惯性 **[.0334,.02167,.04167] kg·m²**，主轴方向单位四元数。这些与 `env.py::box_spec()` 完全一致。COM 比真实几何最低点低 **19 cm**，无法解释为仅由这七个实体内正质量组成的分布；它是官方资产的特殊建模选择。本次保留并显式警示，没有偷偷重定心或重算惯性。

XML 未显式指定摩擦/接触维数，编译后 MuJoCo 默认为 `friction=[1,.005,.0001]`、condim=3、contype/conaffinity=1/1、priority=0。源 box_spec 将 priority 改为 **1**，其他上述参数相同；推荐 manifest 保留这项源实现适配并列出差异。完整 solref、solimp、margin、gap 等见 [box_audit.json](box_audit.json)。MJCF 不包含直接 restitution 系数，不能把 solver 参数当作 PhysX restitution。

必须区分三层质量：XML/raw USD **2 kg**；HDMI `objects.py::BREAD_BOX_CFG` spawn override **3 kg**；任务 YAML randomization 要求 **1.2–1.8 kg**，另有 scale .9–1.1、dynamic friction .3–.8、static/dynamic ratio 1–2、restitution 0–.2。本次选择名义 XML 资产，并未复现随机化训练物理。raw USD 的受检材质/碰撞节点没有作者指定的物理摩擦或 restitution，运行时有效值不能凭视觉材质推断，见 [usd_materials.json](usd_materials.json)。

## 3. Reproduced 与 HDMI 手部碰撞差异

目标实际使用安装的 `mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml`，由 `tracking_env_cfg.py` 的 `get_g1_robot_cfg()` 选择；不是源仓库自定义 G1 XML。已在 CollisionCfg 生效后编译检查。

| 项目 | Reproduced 原生 | HDMI raw USD |
|---|---|---|
| 腕部 | wrist_pitch 上 capsule，r=.035，端点 X=-.01 到 .06 | 没有对应 pitch capsule |
| 手部 | wrist_yaw 上 capsule，r=.035，端点 (.07,0,0) 到 (.15,∓.02,0) | wrist_yaw 上三个 box |
| 三个 box | 不适用 | 中心 (.02,0,0)、(.09,0,0)、(.15,∓.01,0) |
| 三个 box 半尺寸 | 不适用 | (.05,.025,.025)、(.05,.01,.05)、(.025,.01,.05) |
| 末端 box 旋转 | 不适用 | 左 -0.4 rad，右 +0.4 rad，绕局部 Z |
| 每脚 | 7 个 r=.01 capsule | 4 个 r=.005 sphere |

native wrist/hand 本身约重叠 14 mm；HDMI 相邻 box 约重叠 30 mm 和 15.789 mm，中间 box 连接首尾，非相邻间隙不代表断开的手。视觉 mesh 均未误启用为碰撞体。几何、父 body、姿态、mask、group、condim、priority、摩擦逐项数据见 [robot_audit.json](robot_audit.json)；内部距离见 [hand_internal_pairs.csv](hand_internal_pairs.csv)。

原适配器的几何与 raw USD 相差仅浮点精度（位置最大 5.96e-9 m），但它还改脚部；新加 geom 的默认 condim 与 env.py 后续施加的 CollisionCfg 也不同。因此不直接搬运它。**默认保留原生机器人；另提供 `hdmi_hands_only_candidate`，只禁用原腕/手碰撞并增加六个 box，保留原脚和原惯性，且不批准用于训练。** 原生 wrist_yaw body 每侧质量 0.254576 kg，HDMI raw USD 为 0.0845765 kg；COM、主轴和主惯性也不同，见 [wrist_inertia_comparison.json](wrist_inertia_comparison.json)。两引擎的质量、碰撞材质、solver 差异没有因此消失。

## 4. Target、EEF 和真实碰撞面的关系

官方 YAML、NPZ、源 features 均定义 target 为 box-root `[0,±.17,.14]`，EEF 为 wrist_yaw 局部 `[.05,0,0]`。目标正落在 side3/side4 外表面，法向分别 `[0,±1,0]`，低于把手；signed distance 小于 1e-7 m 容差。

计算使用 `p_target = p_box + R_box target_root` 和 `p_eef = p_wrist + R_wrist offset`。实际目标 G1 根据原 joint/root 位姿 FK 后，与记录腕部误差最大 8.19e-8 m、8.30e-8 rad，**没有发现关节顺序、四元数顺序、腕部 FK 或 box-root/world 转换错误**。COM、AABB 中心和 root 没有混用。

但 EEF 是内部参考点，不是手碰撞表面。期望接触段结果如下，单位 mm：

| 指标 | 左手 | 右手 |
|---|---:|---:|
| EEF-target 中位误差 | 41.80 | 42.61 |
| P95 误差 | 85.73 | 86.37 |
| 最大误差 | 96.10 | 118.22 |
| 原生最大 geom-pair 穿透 | 37.22 | 44.99 |
| 原生穿透帧数（>0.1 mm / 290） | 192 | 231 |
| HDMI 最大 geom-pair 穿透 | 31.07 | 57.89 |
| HDMI 穿透帧数（>0.1 mm / 290） | 141 | 207 |
| 原生分离帧数（>1 mm / 290） | 92 | 57 |
| HDMI 分离帧数（>1 mm / 290） | 144 | 78 |

代表帧如下；每格左/右，距离为“最小手–箱 primitive-pair signed distance”，负号是重叠，不是接触力：

| 帧/场景 | EEF 误差 mm | 原生距离 mm | HDMI 距离 mm |
|---|---|---|---|
| 250 / 接触前 | 71.00 / 73.30 | +9.49 / +25.52 | +23.67 / +27.13 |
| 260 / 首个 intent | 46.94 / 40.61 | -9.62 / -6.14 | +3.16 / +3.97 |
| 280 / 持续段 | 40.93 / 37.91 | -11.42 / -8.68 | -.68 / -1.28 |
| 298 / 抬升 | 72.59 / 63.59 | +14.26 / +14.62 | +28.23 / +13.09 |
| 400 / 搬运 | 48.26 / 43.58 | +4.07 / -30.09 | +8.84 / -27.19 |
| 446 / 深穿透 | 20.44 / 31.75 | -35.35 / -42.85 | -19.99 / -57.89 |
| 530 / 放置 | 11.98 / 14.51 | -35.00 / -32.91 | -27.77 / -23.37 |
| 549 / intent 最后帧 | 11.47 / 18.21 | -33.15 / -30.48 | -23.76 / -22.21 |
| 550 / release intent | 10.97 / 19.09 | -33.12 / -30.66 | -23.28 / -22.49 |
| 839 / 末帧 | 1433.26 / 1425.79 | +1290.18 / +1277.82 | +1272.65 / +1259.88 |

保持每帧腕部朝向、平移手部使 EEF 精确对齐 target 的反事实检查，在两种 profile 的所有期望接触帧均导致几何重叠。这不是受关节约束的 IK 可达性证明；它证明不能将“内部点误差为零”当作无穿透接触条件。目标面可接近，但当前 reference 并不提供持续、无穿透的双手物理接触证明。

统计来自 MuJoCo primitive distance 的逐帧几何计算，没有推进动力学，没有假造实测力。最深 primitive-pair 穿透不是整个 compound 的最小分离平移量；point-inside-hand 的 min primitive SDF 在重叠内部也不等于精确 union 出界距离。目标恰在暴露箱壁上，其距离/法向没有该歧义。所有 840 帧的点、腕姿态、误差、法向见 [contact_points.csv](contact_points.csv)，逐碰撞对见 [hand_box_pairs.csv](hand_box_pairs.csv)。

## 5. 两个 support 和场景坐标

两者真实碰撞完整尺寸均为 **.2 × .2 × .04 m**，局部中心为零，局部姿态单位四元数。根姿态在全部帧严格恒定：

| support | world root (m) | top Z (m) |
|---|---|---:|
| support0 | (-1.466734171,-.159754217,.660549879) | .680549879 |
| support1 | (1.381318212,-.284394860,.625122786) | .645122786 |

两者 quaternion wxyz 均为 `(.029028187,0,0,.999578595)`。raw USD mass=1 kg，官方 spawn mass override=.5 kg 且 kinematic=True；本地 source support_spec 使用固定 geom，摩擦 `[1,.005,.0001]`、condim=3、priority=1。

初始箱底 Z=.690549901，最终箱底 Z=.655122807，均比对应顶面高 **10.000022 mm**，超过 1 mm 容差。实际碰撞对距离确认水平投影对齐时仍有间隙；整个 clip 都没有 box-support 几何交叠，最小距离分别 7.146/4.306 mm。其他 support 的无限平面距离可能为负，但不代表横向相隔数米的几何发生碰撞。

推荐 manifest 保留原 pose 并标记失败。简单把 support 抬高 10 mm 虽能消除首尾间隙，却可能在其他帧产生约 5.7 mm 重叠，不能作为未经复查的修复。修改物体轨迹、机器人 reference 都不在本次范围。完整数据见 [support_audit.json](support_audit.json)、[support_frames.csv](support_frames.csv)。

## 6. 推荐 manifest、门禁及交付

推荐文件：[asset_manifest.json](../../../data/assets/hdmi_bread_box/asset_manifest.json)。这是可独立读取的静态物理契约；包含源 SHA、单位/坐标、七几何、显式质量/COM/惯性、接触参数、正确诊断 AABB、两个 support、原生 robot hash、独立 hand-only candidate、contact/EEF 及 sensor filter 契约。没有嵌入逐帧轨迹，也没有复制机器人 reference。

其状态为 `AUDITED_BLOCKED_CONTACT_AND_SUPPORT`，`training_ready=false`。默认使用官方名义七几何、源 priority=1 适配、原始 static support pose、原生手脚。hand-only HDMI profile 只是比较候选，不能声称已解决 reference 几何问题。不要为消除报警而改 reference、把箱子实心化或增加接触奖励。

离线构造器位于 `src/supermimic/hoi/asset_manifest.py`，从 manifest 构造一个 free box 与两个 fixed supports，支持统一场景 origin；未接入训练环境。原 `sm-train`、Trainer、actor/critic、WM、配置和依赖文件均未修改。没有加入第二套训练循环。

| Gate | 结论 |
|---|---|
| Asset parity | 通过，priority=1 及名义物理与 randomized runtime 的差异明确记录 |
| Robot collider parity | 已测量并作候选选择；不宣称物理等价 |
| Contact target surface | 通过，目标在真正暴露箱壁 |
| EEF convention | 通过，官方 wrist+5 cm 定义一致 |
| Physical hand contact | **失败：穿透与分离并存** |
| Reference integrity | 通过，原 NPZ 与机器人数组未动 |
| Scene support validity | **失败：初末约 10 mm 间隙** |
| Pose coordinate consistency | 通过；raw velocity 记录点来源保留限制说明 |

后续应先决定接触参考点与碰撞表面如何约束、是否接受官方特殊 COM、以及 scene support 的显式适配方案，再用全帧几何与受控动力学验证。当前证据不足以批准训练；本次到此暂停。

可视化（点击打开独立 PNG/PDF）：[接触前/首接触/抬升/搬运](contact_snapshots_1.png)、[深穿透/放置/释放/末帧](contact_snapshots_2.png)、[全程误差与距离曲线](contact_timeseries.png)、[曲线 PDF](contact_timeseries.pdf)、[支撑 10 mm 间隙](support_gaps.png)、[支撑 PDF](support_gaps.pdf)。快照使用 box-root 坐标；RGB 为腕坐标轴、红点为 target、紫点为 EEF，末帧扩大视野显示已经离开的双手。

验证结果与复现命令见 [VALIDATION.md](VALIDATION.md)。分阶段提交和范围记录见 [WORK_LOG.md](WORK_LOG.md)。
