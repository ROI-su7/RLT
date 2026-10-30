# Pi05 四 CAN 主从接管实现说明

更新时间：2026-08-28  
适用范围：**监督推理（主臂跟随 + 按 T 人工接管）**。本文不描述数据采集和
仅从臂普通推理。

## 1. 目标

在四条独立 CAN 的 PiPER 系统中实现：

```text
推理：主臂和从臂均由 Pi05 policy 驱动并保持相同位姿
按 T：立即停止 policy，保持从臂；主臂切为重力补偿 MASTER
接管：人工移动主臂，通过 ROS 话题实时控制对应从臂
按 I：主臂回 OUTPUT，重新对齐后恢复 policy 推理
Space：软件 HOLD，保持当前位置
H：在 HOLD/接管失败后，受限速回 HOME
```

此方案采用实现手册中的“**主臂模式 + 话题切换**”：不使用出厂双 CAN 的物理
主从直连，不在不同 CAN 之间转发原始 CAN 控制帧。

## 2. 四 CAN 架构和原理

### 2.1 物理接口

```text
can_l_leader     左主臂
can_r_leader     右主臂
can_l_follower   左从臂
can_r_follower   右从臂
```

出厂双 CAN 时，MASTER 在同一物理总线上发送 `0x151/0x155/0x156/0x157/0x159`，
从臂可直接接收。四独立 CAN 后这些帧不会跨总线传播，必须由 ROS 做协议语义层的
桥接和仲裁：

```text
主臂 CAN -> leader driver -> /teleop/leader/joint_* ------+
                                                           |
Pi05 policy -> /teleop/policy/joint_* -> four-state mux --+->
              /teleop/command/joint_* -> follower driver -> 从臂 CAN
```

核心约束：任一时刻，`/teleop/command/joint_left` 与
`/teleop/command/joint_right` 只能由 `piper_four_state_mux.py` 发布；
OpenPI 只能发布 `/teleop/policy/joint_*`。

### 2.2 PiPER 角色

| 角色 | 协议 | 主臂行为 | ROS 命令源 |
| --- | --- | --- | --- |
| OUTPUT（运动输出臂） | `0x470 = 0xFC` | 接受 CAN MOVE J 位置命令，跟随从臂/policy | policy 对齐命令 |
| MASTER（示教输入臂） | `0x470 = 0xFA` | 重力补偿，人工可拖动；输出 `0x155–0x157`、可选 `0x159` | 主臂实测输入 |

`0xFA` 后控制器需要重启或进入 standby 才能激活 MASTER。监督接管优先使用
运行时无缝 `0xFA`；只有首次等待失败时才使用协议 standby 回退。接管成功以持续
MASTER 关节帧为准，不能只依据 SDK 函数调用成功。

## 3. 状态机

```text
                         T
INFERENCE (OUTPUT) ----------------> HOLD / 停止 policy
     ^                                      |
     |                                      | 1 秒扶住主臂、OUTPUT TX 排空
     | I                                    v
     |                                SWITCHING_MASTER (0xFA)
     |                                      |
     |                      持续 0x155–0x157 验证成功
     |                                      v
     +---- OUTPUT 对齐 <------------- TAKEOVER (MASTER)
              /
              |
     H: HOME 完成后 HOLD

失败路径：SWITCHING_MASTER -> TEACH_FAILED -> OUTPUT/HOLD
```

### 3.1 T 的原子执行顺序

1. mux 冻结两只从臂的实测当前位置；
2. 发布 `hold_active=true`，OpenPI 丢弃排队的 action chunk 并确认停止；
3. 关闭左右主臂 OUTPUT TX gate，等待最后一个 CAN 位置命令完全发送；
4. 等待 1 秒，供单人扶住两只主臂；
5. 两只 leader driver 均发送 `0xFA`；
6. 持续验证两只主臂均产生完整 `0x155/0x156/0x157`；
7. 双侧都成功后，mux 才把从臂命令源一次性改为主臂 topic。

任何一侧失败，所有从臂都保持 HOLD 并恢复到安全 OUTPUT；不允许单侧进入人工
接管。

## 4. 键盘行为

在监督推理键盘终端执行：

```bash
rosrun piper piper_mode_keyboard.py _profile:=inference
```

| 按键 | 行为 |
| --- | --- |
| `T` | 只在监督 `INFERENCE` 中有效；请求人工接管。 |
| `I` | 请求恢复 `INFERENCE`；先完成双主臂 OUTPUT 对齐，后释放 policy。 |
| `Space` | 软件 HOLD；不发送 `enable=false`，避免从臂下落。 |
| `H` | 在 HOLD/失败接管后执行受限速 HOME。必须等 `HOME complete` 后再按 `I`。 |
| `Q` | HOLD 后退出键盘。 |

不要在 HOME 运行中按 `I`。这会取消 HOME，并会增加 S-V1.8-8 的角色状态不确定性。

## 5. 涉及的代码

### 5.1 Piper ROS（角色、CAN 和仲裁）

| 文件 | 作用 |
| --- | --- |
| `Piper_ros_private-ros-noetic/src/piper/launch/start_inference_takeover.launch` | 监督推理入口，启动四 CAN mux。 |
| `Piper_ros_private-ros-noetic/src/piper/launch/start_four_can_mux.launch` | 传递左右 CAN、角色时序、安全限速与接管参数。 |
| `Piper_ros_private-ros-noetic/src/piper/scripts/piper_four_state_mux.py` | 四状态仲裁：INFERENCE、TAKEOVER、HOLD、HOME/IDLE；负责 policy 停止确认、双侧同步和 fail-closed。 |
| `Piper_ros_private-ros-noetic/src/piper/scripts/piper_leader_mode_driver.py` | 每侧主臂唯一 SDK owner；执行 `0xFC/0xFA`、OUTPUT/Master 帧解析、当前姿态预装、恢复和 SocketCAN 诊断。 |
| `Piper_ros_private-ros-noetic/src/piper/scripts/piper_start_slave_node.py` | 每侧从臂唯一 SDK owner；订阅 mux 的最终 command topic。 |
| `Piper_ros_private-ros-noetic/src/piper/scripts/piper_mode_keyboard.py` | 发布 I/T/H/Space/Q 对应的 ROS 请求。 |
| `Piper_ros_private-ros-noetic/src/piper/scripts/piper_log_snapshot.sh` | 只读保存故障时 CAN、ROS graph 和相关节点日志。 |

### 5.2 OpenPI（policy 停止和恢复）

| 文件 | 作用 |
| --- | --- |
| `openpi/examples/piper_real/main.py` | 将 Pi05 输出发布到 `/teleop/policy/joint_*`，不直接控制 follower。 |
| `openpi/examples/piper_real/env.py` | 提供从臂状态、三相机观测与 HOLD 状态。 |
| `openpi/examples/piper_real/hold_aware_policy.py` | 收到 HOLD 后停止发布；恢复后丢弃旧 action chunk，以当前观测重新推理。 |
| `openpi/scripts/serve_pi05_wire_all_data_full_30000.sh` | 当前 Pi05 policy server 启动脚本。 |

## 6. S-V1.8-8 固件事实与最后一次修改

已用只读 SDK 查询确认：

```text
can_l_leader: S-V1.8-8
can_r_leader: S-V1.8-8
```

此版本的 `MASTER -> OUTPUT` 已知可能残留扭矩控制状态。后果是：OUTPUT 反馈和
推理表面正常，但下一次 `OUTPUT -> MASTER` 的 `0xFA` 可能不产生 `0x155–0x157`。
官方说明该问题在 `S-V1.8-9` 或更高版本修复；升级固件是永久解决方案。

### 6.1 最后一次修改内容

最后一次只修改监督接管链路；没有修改固定 MASTER 数据采集和 follower-only
普通推理的控制逻辑。

| 修改位置 | 修改前 | 修改后 | 目的 |
| --- | --- | --- | --- |
| `piper_leader_mode_driver.py` | 第一次 `0xFA` 无关节帧仅等待 `1.0s` | 等待 `4.0s` | 历史成功里右主臂完成 MASTER 最长约 `3.38s`；避免夹爪刚松开就被 standby 回退打断。 |
| `piper_leader_mode_driver.py` | 默认发送 `0x191 mode=0` | 默认关闭，保留参数供诊断 | 实测 `0x191` 有发送回读但未产生 MASTER 流；默认保持官方最小 `0xFA + standby` 路径。 |
| `piper_leader_mode_driver.py` | MASTER -> OUTPUT 后 6 次当前姿态 MOVE J | 20 次；首次 OUTPUT enable 也执行 | 清除固定 MASTER 采集切到监督推理时 S-V1.8-8 的扭矩残留。 |
| `piper_leader_mode_driver.py` | 原始诊断只报告 `0x155–0x157` | 同时报告 `0x159` 频率 | 区分“仅夹爪帧”与“完全没有 MASTER 帧”。 |
| `start_four_can_mux.launch` | 参数为 `1.0 / true / 6` | 参数为 `4.0 / false / 20` | 将最终策略传给左右主臂节点。 |
| `test_leader_role_transition.py` | 无首次 OUTPUT enable 恢复测试 | 新增 mock 测试和 `0x159` 字段检查 | 防止后续改动回退修复。 |
| `test_four_can_profiles.py` | 断言旧 launch 默认值 | 断言最终参数 | 保证 launch 配置不回退。 |

最终生效参数：

```text
master_first_no_stream_retry_s       = 4.0
master_restore_linkage_activation    = false
follower_restore_movej_repeats       = 20
follower_restore_movej_period_s      = 0.02
takeover_release_delay_s             = 1.0
takeover_role_timeout_s              = 15.0
```

## 7. 验证和现场判断

已完成的离线验证：

```text
Python 语法编译通过
44 项无 CAN 单元测试通过
start_inference_takeover.launch 节点展开正常：5 个节点
```

实际按 T 后，应等待以下成功标志，不要仅凭主臂手感判断：

```text
MASTER handshake VERIFIED
leader control state: MASTER
TAKEOVER active
```

出现以下信息表示固件未完成 MASTER 切换，系统已安全回退，不应连续反复按 T：

```text
MASTER stream timeout snapshot
TEACH_FAILED
A leader role transition failed: TAKEOVER -> HOLD
```

此时使用：

```bash
rosrun piper piper_log_snapshot.sh
```

保存现场日志。若多次失败，需在机械结构支撑下断电重启**左右主臂控制器**；不要
对负载中的从臂直接发布 `enable=false`。

## 8. 参考资料

- 本机 CAN 协议/实现手册：
  `/home/agilex/cobot_magic/Cobot_Magic_人在环遥操作接管功能实现手册_截图归档及CAN协议.docx`
- Piper SDK V2：<https://github.com/agilexrobotics/piper_sdk/blob/master/piper_sdk/interface/piper_interface_v2.py>
- Piper SDK V1：<https://github.com/agilexrobotics/piper_sdk/blob/master/piper_sdk/interface/piper_interface.py>
- Piper 双臂主从说明：<https://github.com/agilexrobotics/piper_sdk/blob/master/asserts/double_piper.MD>
- AgileX S-V1.8-8 seamless-switching 说明：
  <https://github.com/agilexrobotics/pyAgxArm/discussions/53>
