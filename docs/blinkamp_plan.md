# BlinkAMP：基于世界模型的 AMP 行走迁移方案

状态：规划完成，尚未实现或启动 BlinkAMP 训练。开发分支：`blinkamp-walking`。
参考：本仓库 BlinkMimic，以及 `../physhsi` 的行走环境、动作库和 AMP 实现。

## 1. 首版目标与固定约束

在现有 G1 / mjlab / MuJoCo Warp 上，实现不依赖参考动作相位的速度指令行走。
动作示范只用于 AMP 专家分布和训练期参考状态初始化，不作为策略未来目标。
策略通过可微世界模型的短视野想象轨迹优化任务奖励与 AMP 风格奖励。

用户明确要求：**使用 root 坐标**，root 固定指 G1 pelvis，而非 torso。
用户补充约束：不添加外部 push、力或速度扰动；观测噪声复用 BlinkMimic 的定义，真实与想象保持一致。
首版先做平地普通行走、起停，随后扩展转向；高抬腿、恐龙步态和人景交互不在首版范围内。
保留原 BlinkMimic 跟踪入口和已完成的复现实验，独立增加 `blinkamp` 包及配置。

## 2. 已检查的源码与数据事实

- `../physhsi/rsl_rl/rsl_rl/modules/amp.py`：最小二乘判别器，专家目标 +1、策略目标 -1，专家输入梯度惩罚；奖励为 `max(0, 1 - (D(x)-1)^2/4)`。
- `../physhsi/legged_gym/legged_gym/envs/g1/styleloco.py`：速度指令、多帧 AMP 窗口、本体历史、参考状态初始化。
- `styleloco_highknee_config.py`：10 帧 AMP 窗口，每帧实际为 57 维（实现验收纠正了规划中的加法错误），AMP 系数 0.35。其 runner 还将 AMP 奖励乘 0.5，不能只照搬系数就认为奖励尺度等价。
- 经后续逐字段审计，HighKnees 的 `control.upper_body_link` 实际为 pelvis；同名 asset 字段为 torso 但不选取这里的观测坐标。此前规划对上躯干坐标的概括不准确；BlinkAMP 运行时一直使用 pelvis，未因此引入坐标变换差异。
- `dataset_standup/loco_direction/` 有 11 段普通行走、起停和转向数据；不混用四个任务目录的重复数据。
- 已实际读取 `B3_-_walk1_stageii.pt`：456 帧、29 关节，含根位姿、关节角及速度；对应配置声明原始频率 60 Hz。
- 29 个关节的名称和排列已与 BlinkMimic G1 核对一致，正式加载仍按名称显式映射。
- 该文件 `link_position` 是 6 个点，不能按 styleloco 的 5 点语义直接截取。
- BlinkMimic 的 `MDP`、`TrackingSim`、策略 goal、参考动作 skip、critic 特征、采样和探测重置均与跟踪任务耦合，不能仅换奖励完成迁移。

## 3. 坐标与特征契约

内部物理状态继续使用现有 `S=(p_W, R_WR, v_W, w_W, q, qd)`，保证积分、仿真写入和世界模型兼容。
对外的本体和 AMP 特征统一由 root 位姿转换：

- 关键点：`p_key_R = R_WR^T (p_key_W - p_root_W)`。
- 线速度、角速度：`v_R = R_WR^T v_W`、`w_R = R_WR^T w_W`。
- 重力：`g_R = R_WR^T (0,0,-1)`。
- 指令：`c=(vx_R, vy_R, wz_R)`，线速度误差和角速度误差使用相同 root 定义。
- 根高度：`p_root_W.z - z_ground`，是地面相对标量，不能把 root 自身的局部 z=0 当高度。
- 根倾斜：对 `R_WR` 去除世界 z 轴航向，再取明确约定的 6D 表示；不把 root 相对自身的单位旋转作为姿态特征。
- 世界平面平移和共同绕重力轴旋转不应改变这些特征。任意俯仰旋转不是重力环境的对称性。

首版 AMP 每帧 57 维：高度 1 + 关节角 29 + 五个 root 相对关键点 15 + 根线速度 3 + 根角速度 3 + 去航向姿态 6。
关键点固定为左右 wrist_yaw、左右 ankle_roll、torso，均用当前 G1 的 FK 重建。
这不是照搬 PhysHSI 的 palm/head 定义；专家、在线仿真、想象状态必须使用同一映射。
AMP 历史为 10 帧、50 Hz、最旧到最新；不含指令、相位、clip ID 或世界 xy。

Actor 首版输入为 6 帧本体历史：root 角速度、重力、关节角偏移、关节速度、上一动作，以及当前 3 维 root 指令。
默认不向 actor 输入真实 root 线速度，critic 可以使用；关键点可由本体 FK 推导，首版不额外加入 actor。
Actor 输出 29 维静态默认姿态上的 PD 残差，移除参考动作 skip 和未来动作编码。

## 4. 数据与重置

新增显式数据 manifest，默认只读引用 `../physhsi/legged_gym/resources/dataset/dataset_standup/loco_direction/`。
路径可配置，不把参考项目作为 Python/Isaac Gym 运行依赖。

1. 明确读取 root 四元数 xyzw，转为当前工具要求的 wxyz；关节按名称映射。
2. 用文件的原始时间轴从 60 Hz 重采样到 50 Hz；四元数用最短弧插值。
3. 从重采样位姿统一重算速度，角速度用旋转差分，不用欧拉角直接差分。
4. 在当前 MJCF 上重建 FK、根相对关键点和地面高度修正，检查脚底悬空/穿地。
5. 保存片段边界、来源、频率、处理配置和哈希；专家窗口不得跨 clip。
6. 首阶段使用普通直行、起步、停止片段；转向片段在扩展指令课程时加入。
7. 训练初期混合参考状态和站立状态初始化，最终必须从站立状态通过评估；推理不读取示范。

reset 时 actor 和 AMP 历史均按当前初始帧填充，清空动作与 episode 计数。
判别器训练只使用已有完整 10 帧的真实历史，排除 reset 填充窗口；终止前的完整失败窗口仍可作为负样本。
真实 transition 必须在 auto-reset 前保存 terminal next state，禁止把 reset 跳变当动力学训练样本。
动作结束不是行走任务的终止条件；训练不再推进参考帧来决定任务长度。

## 5. BlinkAMP 算法数据流

每个外层迭代：

1. 在真实仿真中按速度指令执行策略并加探索噪声，记录状态、动作、上一动作、仿真上下文、指令、本体历史、终止/超时信息。
2. 用真实连续 transition 训练结构化世界模型，保留多步损失和左右镜像增强。
3. 用专家窗口及真实策略窗口训练 AMP 判别器；负样本混合近期数据与有上限的历史回放，避免陈旧分布主导。
4. 从真实回放采样初态，构造短视野想象轨迹，首版 horizon=16，再验证 32。
5. 冻结世界模型、判别器、critic 参数和归一化统计，但保留它们相对输入的梯度；通过任务奖励、AMP 奖励及价值 bootstrap 优化 actor。
6. 更新 critic、目标/EMA 参数、日志和检查点。

判别器不使用想象轨迹作为首版负样本：避免它学习世界模型伪影。
想象时前 9 帧真实 AMP 历史 detach，之后逐步追加可微预测特征，保留窗口内部对 actor 的梯度。
不能调用 PhysHSI 带 `no_grad()` 的 `predict_reward` 作为想象奖励，否则风格梯度会断开。
判别器更新时 actor 样本 detach；actor 更新时判别器参数冻结，二者优化器独立。
归一化若启用，只从未归一化专家/真实数据更新，并在整个 actor 更新期间固定。

初始奖励建议：

`r = dt * [(1-alpha) * r_task + alpha * r_amp - regularizers]`，`alpha=0.35` 为起始实验值。

`r_task` 将 root 平面速度跟踪与 root 偏航角速度跟踪的指数奖励归一到 [0,1]。
`r_amp=max(0,1-(D(x)-1)^2/4)`；这里不额外乘 PhysHSI runner 的 0.5，奖励尺度和区别写入配置。
正则包括动作变化、软关节限位、竖直速度及过度倾斜；只有可从状态/动作一致计算的项进入首版想象奖励。
真实扭矩指标用于诊断；没有可靠扭矩预测时不把它伪装成想象奖励。
最小二乘 + 专家梯度惩罚沿用方法思想，权重与输出尺度显式配置。

初期进行短暂真实采集和 WM 预热；不沿用跟踪任务“零预热”假设。
任务奖励为判别器尚未提供有效梯度时提供学习信号；监控 clamp 后 AMP 奖励为零的比例。
用 held-out 真实轨迹测量多步模型误差，缩短 horizon、调整更新比率以限制模型利用，不只观察想象回报。
终止采用无参考的根高度/倾斜与接触规则；想象使用对应平滑近似。
超时与物理跌倒分别标记，critic 在超时时从 terminal state bootstrap，不从 reset state bootstrap。
每个想象 rollout 内固定采样指令，真实仿真按配置重采样；回放中必须保存实际指令。

## 6. 组件与实施顺序

新增独立 `src/blinkamp/`，复用 `blinkmimic` 的机器人描述、旋转、FK、DR、世界模型、优化器和 actor objective。
优先显式组合/抽取低层仿真 stepping，不给 `TrackingSim` 塞虚假 motion 对象规避接口。

### M1：数据、特征与真实行走环境

- `motion.py`：PhysHSI 数据 adapter、manifest、时间轴与合法专家窗口。
- `features.py`：root 特征和完整 AMP 历史。
- `mdp.py`、`env.py`：指令课程、无参考 reward/done/reset、29 维 PD 动作。
- 从零动作、随机动作、参考状态播放验证 FK 和仿真一致性。
- 验收：root 特征不变性、四元数/关节映射、时间重采样、clip/reset 边界、终止状态保存通过检查。

### M2：AMP 与可微目标

- `amp.py`：判别器、真实负样本 replay、梯度惩罚、可微 reward。
- `networks.py`：指令策略与独立 critic，无 reference goal/skip。
- `buffer.py`：真实状态/指令/本体历史/AMP 历史及边界标记；为新增字段核算显存。
- 验收：D 能区分专家和扰动数据；真实/想象特征一致；AMP reward → imagined state → action → actor 梯度非零且有限，冻结模块参数不更新。

### M3：可运行训练、监控与续训

- `train.py`：WM / discriminator / actor / critic 的独立更新；先 eager，稳定后再 compile/graph。
- `scripts/train_amp.sh`、配置：普通前进行走，`vx` 先覆盖示范支持的速度区间，起始候选 [0.3,0.9] m/s；起停加入 0 指令。`vy=wz=0` 起步，之后加入转向。
- 先 64～256 env、短训练验证闭环，再上 1024～4096 env；在 RTX 5080 16GB 上实测显存决定批量，不承诺原 main preset 参数直接适用。
- 原生 TensorBoard + CSV：真实/想象任务回报、AMP 回报、速度误差、存活、D 真/假分数、各项损失、梯度惩罚、模型多步误差、非有限更新、吞吐和显存。
- checkpoint 保存 actor、WM、D、critic、全部优化器/归一化器、随机数与配置；区分 WM 权重初始化和可复现续训。
- 已有 BlinkMimic 权重仅作为后续 WM warm-start 消融；检查 architecture、dt、DR、输入/输出归一化兼容，不能直接加载跟踪 actor。

### M4：评估与迁移验收

- 独立 `evaluate.py`、`scripts/reproduce_amp.sh`，锁定模型/归一化、关闭探索噪声，在真实 MuJoCo Warp 中测试。
- 初版候选网格：前进 0、0.3、0.6、0.9 m/s；扩展阶段加入 0.6 m/s + ±0.3 rad/s 转向。
- 每条指令至少 32 次、每次 20 s；单独测起步、停止、速度切换与转向；多种子重复。
- 起始工程门槛（不是已取得结果）：非零前进指令存活率 ≥95%，稳定段线速度 RMSE ≤0.15 m/s；转向阶段角速度 RMSE ≤0.2 rad/s。0 指令单独报告静止漂移，不被高存活率掩盖。
- 报告脚滑、落脚接触、穿地、动作平滑、能耗代理及视频；D 分数只是诊断，不能跨不同训练的判别器当共同质量标尺。
- 先对比相同世界模型训练流程下 task-only 与 task+AMP，确认风格改善且速度控制不退化；随后做 root 特征、horizon、WM warm-start 消融。
- 如果要声称比 model-free AMP 快，另加同机器人/动作/奖励/数据的 PPO+AMP 对照，按墙钟和真实仿真 transition 两种预算报告；不能直接拿 PhysHSI 原始 Isaac Gym 耗时比较。
- 记录首次编译和稳态耗时，3 seeds 汇总；行走任务不套用 BlinkMimic 的 0.82 跟踪阈值。

## 7. 实现约束与可追溯性

- 默认独立实现，不整体复制 PhysHSI 环境或引入其 Isaac Gym/Python 3.8 依赖。
- PhysHSI 仓库声明 CC BY-NC-SA 4.0；保留来源说明，数据与任何实际移植代码记录单独来源和适用许可，不自动标成本仓库 MIT。
- 原始行走数据、训练输出和模型继续保存在被忽略的目录；manifest 和处理代码入版本控制。
- 当前只完成分支与方案。下一实施步骤是 M1 的数据转换和 root 特征验证，不直接启动大规模训练。
