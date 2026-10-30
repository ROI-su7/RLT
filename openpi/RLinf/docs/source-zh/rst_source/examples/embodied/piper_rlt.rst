Piper 双臂 RLT 真机适配
=======================

当前适配以 RLinf 官方 RLT 为算法基线，只增加 Cobot Magic 所需的 ROS1、三相机、
14 维绝对关节动作、限幅、evdev 键盘与双机 Ray 支持。

关键约定
--------

- Stage 1 冻结原生 JAX ``pi05_connect_strings`` checkpoint，使用 8 卡 FSDP 只优化
  RL-token encoder/decoder 的重建损失；checkpoint 记录精确 JAX config、image-only
  prefix 约束、fp32 主权重和 encoder fingerprint；
- 在全新 Stage 2 轮次中，先收集 1000 个 transition，此前 learner 不更新；
  完成 3000 次 learner update 前，``b`` 始终由 JAX
  reference 执行；
- warmup 后每新增一个 transition 只产生 5 次 learner 更新预算；没有新数据时不更新，
  critic 与 actor 的更新比例为 2:1；
- Stage 2 使用稀疏二值终局奖励：``c=+1``，``x/q=0``；
- ``b`` 标记下一完整 10-action critical chunk；warmup 时仍由 JAX reference 控制并
  写 replay，learner 就绪后才切入 actor；
- online actor 对齐 RL-Token-Pi05-open 的 full-output Gaussian 契约：
  ``mu=MLP(z,p,a_ref)``、随机初始化、固定 normalized ``sigma=0.002``；Actor 使用
  现有两层 ``Linear-ReLU-Linear`` MLP（hidden width 256），保持 checkpoint 兼容。
  平滑方案对齐 Yyshadow/openpi-RLT，目标为
  ``bc_weight * BC - 0.1 * Q1 + 10 * delta_penalty``；更新 3000 之前 BC 权重 10，
  之后权重 5。Actor LR 为 1e-4，恢复 checkpoint 时也生效。
  并保留 50% reference dropout。
  SAC 训练仍从该分布采样，真机执行其均值。Actor 保留完整动作输出。
  ``algorithm.rlt_schedule.actor_direct_execution: true`` 关闭 Actor 残差 EMA/taper、
  执行器 EMA、chunk 边界混合和 RTC；仍保留相对 reference 的逐关节 0.01 rad、
  夹爪 0.0015 m 限幅。Actor 窗口从 reference 第 0 步开始。BC 约束原始
  采样动作，Actor Q 与 target actor bootstrap 查询限幅后的动作。
  ``delta_penalty`` 在物理单位比较两个六关节手臂的相邻动作增量与 reference/人工
  目标增量，排除夹爪、padding 和来源切换对；它约束原始输出，超限时仍保留梯度。
  跨 chunk 纠正 EMA 已撤回，恢复命令跟踪器：接受目标相对上一接受目标每策略步
  最多改变 0.004 rad / 0.0002 m；200 Hz 跟踪器限制手臂速度/加速度为
  0.10 rad/s / 0.6 rad/s2，夹爪为 0.004 m/s / 0.02 m/s2，响应时间 0.12 s。
  命令状态跨 chunk 保留，HOLD 重置。它限制完整 Actor 目标，因此也可能减慢
  reference 分量。Q 的共享变换建模纠正限幅，不包含此有状态命令跟踪器。
  JAX 保持原有 EMA、RTC bridge 和对齐，包括 warmup。
  Actor 关闭 RTC 后，推理延迟可能造成 chunk 边界停顿，直接执行不保证运动平滑。
  不使用整段偏差回退。
  critic bootstrap 使用 C 现有 EMA target actor，再取 target twin-Q 的最小值；
  target model 保持每次 update 做 Polyak 的现有 C 频率；
- 本方案采用 openpi-RLT 的物理单位增量损失，未替换成其投影
  LayerNorm/GELU 网络。真机仍执行均值；加速度跟踪是本地扩展，非 B 默认实现。
  命令平滑不降低当前 15 Hz 的纠正更新频率，降低纠正调度频率需要单独实现。
- replay 使用 uniform 随机采样，不设来源配额。``[RLT correction]`` 日志区分
  ACTOR、JAX_REFERENCE、HUMAN 和混合控制；对比请求动作与 RTC 对齐后的
  同时刻 JAX 窗口，排除无效步和人工步。左右臂以 rad 统计，夹爪以 m 统计；
  数值包含后续混合/目标限幅的影响，不是实测电机位置，也不证明任务成功率提高；
- 每 2 次 critic update 严格执行 1 次 online-actor update；partial transition 屏蔽
  padding reward，并使用 ``gamma ** 实际执行步数``，完整 chunk 仍为 ``gamma ** 10``；
- policy target 以 15 Hz 生成，独立控制进程在 200 Hz 插值发布。四 CAN 配置在推理等待时
  JAX 路径执行有界 RTC 后缀；HOLD/T 丢弃旧后缀，I 从新观测重新规划；
- 相机帧使用与 ``examples/piper_real`` 完全相同的 OpenPI PIL
  ``resize_with_pad``；不要替换为 OpenCV 插值，像素差异曾在真机上造成可重复的
  第一槽 reference 横向偏移；
- 仍保留关节边界、单步关节和夹爪变化限制；``q`` 不是实体急停。

完整的 checkpoint 转换、Stage-1/Stage-2 配置、双机启动命令和安全流程见仓库根目录
``COBOT_MAGIC_RLT_DEPLOYMENT.md``。

Stage 2 四 CAN 人工接管
-----------------------

默认 Piper Stage-2 配置复用 OpenPI 已实现的监督推理 launch，不修改原驱动、
角色切换、两秒接管等待或 HOME：

.. code-block:: bash

   roslaunch piper start_inference_takeover.launch \
     master_left_can:=can_l_leader master_right_can:=can_r_leader \
     puppet_left_can:=can_l_follower puppet_right_can:=can_r_follower \
     publish_hz:=200

RLinf 统一处理按键，不要同时运行独立 OpenPI 客户端或 ``piper_mode_keyboard.py``。
策略只发布到 ``/teleop/policy/joint_left/right``；原 mux 是
``/teleop/command/joint_left/right`` 的唯一从臂命令源。独立执行进程清除待执行命令
之后才发布 ``/teleop/policy_hold_ack``。

Stage 2 启动脚本会自动选择唯一具备全部控制按键的 evdev 键盘。检测到多个候选时，
使用 ``RLINF_KEYBOARD_DEVICE=/dev/input/by-id/...`` 指定设备；不要依赖会随重启或
USB 插拔变化的 ``/dev/input/eventN`` 编号。

真机本地动作服务使用以下入口：

.. code-block:: bash

   cd /home/agilex/FluxVla/openpi/RLinf
   bash examples/embodiment/serve_piper_rlt_compact.sh

脚本使用 ``pi05_connect_strings`` 配置内的提示词，加载同一原生 checkpoint 和冻结 RL token，
预热两条模型路径后监听 6006；不打开 ROS/CAN 接口。动作请求的精确 prefix 在机器人本地编码，
跨机发送 8192 字节的 ``rlt_z_rl``，取代 3145728 字节的原始 prefix。stride 观测拆为单个低优先级
任务，已排队的动作请求优先执行。四 CAN preflight 要求 ``rlt_action_z_rl`` metadata。

RTC 仍受已确认策略控制权和观测代次门控，执行进程完成后缀后对齐下一 actor 窗口。
空格、T 和终止 HOLD 会清除待执行 RTC，即使模型请求仍在等待。

- ``I``：idle 时开始回合；HOLD 或人工接管后恢复模型推理。
- ``B``：下一完整 chunk 开始采集；learner 就绪决定 JAX reference 或 actor 控制。
  Actor 纠正逐维限幅，不使用整段偏差回退。
- ``T``：B 后请求人工控制。只有确认 ``TAKEOVER/ACTIVE`` 后才标记干预，并记录
  mux 仲裁的 ``/master/joint_left/right`` 人工目标，单位为弧度/米。
- ``C``：成功 reward +1；``X``：失败 reward 0；均终止回合并 HOLD。
- 空格：调用原软件 HOLD，保留回合和 B 状态；HOLD 帧与过期 chunk 后缀不写 replay，I 重新规划。
- ``H``：idle/HOLD 时调用 mux HOME，四臂返回本次 ROS launch 捕获的启动姿态并保持，
  不记录回位数据。``Q`` 保留为零奖励终止 HOLD；此后端的 ``A/R/G/F`` 不承担功能。

独立验收 ``I-B-C``、``I-B-X``、``I-B-T-C``、``I-B-T-X``。
人工与自主数据共用 ``logs/<run>/replay_buffer/rank_0``；人工步带 ``intervene_flags``，
warmup 数据保留 ``reference_warmup``，两类采样标签可以重叠。终止补齐位置带
``action_valid_mask``，不参与 BC；跨越已丢弃 HOLD 动作的窗口会被丢弃。
四臂与仅从臂模式都在 B 后首次 C/X 上报 ``rlt_keyboard_terminal``，使跨 chunk
到达的终止事件能补到最后一条真实 transition；重复终止帧不重复上报，未按 B 不写标签。

单机 Stage 2 续训
----------------

单机启动脚本默认重新初始化 Stage 2 权重和 replay。续训时同时设置
``RLINF_STAGE2_RESUME_DIR`` 与 ``RLINF_REPLAY_LOAD_PATH``，保留旧权重及全部数据。
旧运行停止后，可把它的完整 replay
与后续采集运行的 replay 合并到一个新目录；源数据保持不变：

.. code-block:: bash

   cd /home/agilex/FluxVla/openpi/RLinf
   source .venv-rlt/bin/activate
   python toolkits/merge_rlt_replay_buffers.py \
     --destination /absolute/path/to/merged/rank_0 \
     /absolute/path/to/first/replay_buffer/rank_0 \
     /absolute/path/to/second/replay_buffer/rank_0
   export RLINF_REPLAY_LOAD_PATH=/absolute/path/to/merged/rank_0
   export RLINF_STAGE2_RESUME_DIR=/absolute/path/to/checkpoints/global_step_50
   bash examples/embodiment/run_piper_rlt_stage2_single.sh

重启前核对 ``skipped_missing_index_entries=0`` 与预期 transition 总数。单独恢复
checkpoint 只会加载保存时较旧的 replay 快照。续训初期按 B 后，
``replay/actor_switch_rate=0`` 也可能表示 actor 已就绪，但动作偏差安全门选择了 JAX；
继续采集 reference 或人工数据，并检查 actor 与参考动作的误差后再考虑放宽门槛。

仅 Replay 的 Stage 2 重训
-------------------------

使用 replay-only 入口可从持久化 ``rank_0`` buffer 随机初始化并重训 Actor/Critic。
该入口只启动 ``ActorGroup``，不会创建 ``EnvGroup``、``RolloutGroup``、JAX client、
ROS 节点、相机或 CAN 接口。由于关闭 auto-save，源 replay 只读；checkpoint 仍包含
replay 快照，便于完整恢复。

.. code-block:: bash

   cd /home/agilex/FluxVla/openpi/RLinf
   source .venv-rlt/bin/activate

   export RLINF_REPLAY_LOAD_PATH=/absolute/path/to/replay_buffer/rank_0
   export RLINF_STAGE1_RLT_TOKEN=/absolute/path/to/rlt_token.pt

   bash examples/embodiment/run_piper_rlt_stage2_replay_only.sh

启动器使用全新 Stage 2 Actor/Critic 权重，默认运行 2000 个 global learner step，
每 50 step 保存一次，输出位于
``logs/<timestamp>-realworld_piper_rlt_stage2_replay_only/``。此从零重训流程不要设置
``runner.resume_dir``。设置 ``RLINF_REPLAY_ONLY_MAX_STEPS`` 可指定其他目标，
例如 ``10000``。本地 GPU 节点上必须已启动 Ray。

旧版可选 mode2 主从接管
----------------------

原有 mode0 和 mode1 ROS 路径保持不变。只有显式选择
``realworld_piper_rlt_stage2_ac_mlp_mode2.yaml``（单机）或
``realworld_piper_rlt_stage2_ac_mlp_dual_mode2.yaml``（双机）才会启用主从接管。
复合 ROS 驱动使用以下入口：

.. code-block:: bash

   roslaunch piper start_ms_piper.launch mode:=2 auto_enable:=true \
     mode2_leaders_preisolated:=true

该 launch 内部复用已经验证的 mode1 从臂位置控制节点，同时增加隔离的主臂 CAN
bridge；只有显式 mode2 分支启用它，原 mode0/mode1 条件和行为保持不变。控制权
顺序为：

``policy -> H/从臂 hold -> M/主臂低速接近 -> 人工微调 -> READY -> T/人工主从``。

当前 Piper API 没有公开示教输入主臂的任意位置控制，因此 ``M`` 启动的是受监控的
**人工慢速接近**：操作者缓慢移动持续上电的主臂，从臂保持锁存位置。状态机先检查
``approach_tolerance_rad``，再连续检查更严格的 ``position_tolerance_rad``，满足后
才提示 ``READY``。反馈缺失/过期、从臂偏离 hold 或对齐超时都会 fail closed。

按 ``T`` 时，系统先暂停 policy/RTC 发布器，再启用独立 bridge。此后 bridge 是
``/master/joint_left`` 和 ``/master/joint_right`` 的唯一发布源，以 200 Hz 转发主臂
偏移帧；环境只采观测，不再发第二路从臂命令，并把实测主臂动作写入
``intervene_action``。``C`` 标记成功、``X`` 标记失败，但两者都会进入不写 replay
的主从回程：人工将主臂移回记录起点，从臂同步跟随，再按 ``R``。``R`` 会验证主臂
和从臂都已回到起点、关闭 bridge、让从臂 hold，然后才允许下一次 ``A``；``R``
本身不会自动驱动主臂回位。

mode2 之外保持原生 ``0x15x``，无需 ROS 即可由主臂直接控制从臂。Piper
S-V1.8-8 会应答角色不变的 0x470 偏移指令，但不会实际应用；改变角色的 0x470
又会广播到共享 CAN 上的主臂和从臂。因此当前接线不支持纯软件在线切换
``0x15x``/``0x16x``。只有物理隔离、离线配置并验证两只主臂均为 ``0x16x`` 后，
mode2 才允许显式启动；恢复 ``0x15x`` 同样必须断开从臂。实测离线流程见
``COBOT_MAGIC_RLT_DEPLOYMENT.md``。

共享 CAN 离线切换流程
~~~~~~~~~~~~~~~~~~~~~~

从原生 ``0x15x`` 进入 mode2：

1. 在原生跟随状态把左右主从移动到彼此接近的安全姿态。停止 RLT、RTC、policy 和
   所有 Piper ROS 节点，然后关闭四臂电源；
2. 在四臂全部断电时，物理断开左右从臂航空/CAN 插头，并保持两只从臂断开、断电；
3. 只给两只主臂上电，先配置 CAN 接口，然后分别配置主臂：

   .. code-block:: bash

      cd /home/agilex/cobot_magic/Piper_ros_private-ros-noetic
      source /opt/ros/noetic/setup.bash
      bash can_config.sh

      cd /home/agilex/FluxVla/openpi/RLinf

      /usr/bin/python3 \
        toolkits/realworld_check/configure_piper_mode2_master_offset.py \
        --can can_left --apply --i-confirm-follower-disconnected

      /usr/bin/python3 \
        toolkits/realworld_check/configure_piper_mode2_master_offset.py \
        --can can_right --apply --i-confirm-follower-disconnected

4. 第一次实帧验证前不要断电：

   .. code-block:: bash

      timeout 3 candump can_left,160:7F0
      timeout 3 candump can_left,150:7F0
      timeout 3 candump can_right,160:7F0
      timeout 3 candump can_right,150:7F0

   两侧 ``0x16x`` 过滤器都必须看到主臂帧，两侧 ``0x15x`` 过滤器都必须无输出。
   ``timeout`` 返回 124 是正常现象，以实际 CAN ID 为判据；任一侧不符合就停止；
5. 关闭主臂电源。在四臂全部断电时重连两只从臂，全程不要移动机械臂，然后给四臂
   上电；
6. 只有前面的检查全部通过后才能启动 ROS：

   .. code-block:: bash

      cd /home/agilex/cobot_magic/Piper_ros_private-ros-noetic
      source /opt/ros/noetic/setup.bash
      bash can_config.sh
      source devel/setup.bash
      export ROS_MASTER_URI=http://192.168.209.107:11311
      export ROS_IP=192.168.209.107

      roslaunch piper start_ms_piper.launch \
        mode:=2 \
        auto_enable:=true \
        mode2_leaders_preisolated:=true

7. 启动 policy 前验证两路主臂 topic 和带保护的 mode2 services：

   .. code-block:: bash

      rostopic hz /rlinf/mode2/leader_left
      rostopic hz /rlinf/mode2/leader_right
      rosservice list | grep /rlinf/mode2

``mode2_leaders_preisolated`` 只是操作者确认，不会切换地址；主臂仍发送原生
``0x15x`` 时绝对不能设置它。

恢复原生跟随时，停止全部控制进程、关闭四臂电源并再次断开两只从臂。只给主臂
上电后执行：

.. code-block:: bash

   cd /home/agilex/cobot_magic/Piper_ros_private-ros-noetic
   source /opt/ros/noetic/setup.bash
   bash can_config.sh

   cd /home/agilex/FluxVla/openpi/RLinf

   /usr/bin/python3 \
     toolkits/realworld_check/configure_piper_mode2_master_offset.py \
     --can can_left --apply --restore-default \
     --i-confirm-follower-disconnected

   /usr/bin/python3 \
     toolkits/realworld_check/configure_piper_mode2_master_offset.py \
     --can can_right --apply --restore-default \
     --i-confirm-follower-disconnected

确认两侧都发送 ``0x15x`` 且不再发送 ``0x16x`` 后，才能断电重连从臂并给四臂
上电。最终以不启动 ROS 时左右主臂能分别带动对应从臂为恢复成功判据。从臂仍连接
共享 CAN 时，绝对不能发送角色配置命令。

异常隔离会话后的一次性恢复可使用带保护的
``/rlinf/mode2/align_followers_to_leaders`` 服务，让从臂通过限速插值靠近主臂，
再进行离线硬件恢复；该服务本身不修改 CAN 地址。RLT policy 正在发布时绝对不能
调用该服务。
