# 从零学习：Pi05 四 CAN 主从接管的代码、协议与原理

更新时间：2026-08-28  
读者：第一次把 PiPER 出厂双 CAN 主从改成四独立 CAN，并希望理解“推理 → T 接管
→ I 恢复推理”全部代码链路的开发者。

本文只讨论**监督推理中的主从接管**。数据采集固定 MASTER 和只从臂普通推理是
不同会话，不在本教程的控制流程内。

---

## 0. 先建立正确的心智模型

不要把四 CAN 接管理解成“把左主臂 CAN 帧转发给左从臂”。四 CAN 下每只臂连接到
独立 SocketCAN 接口，原始 `0x155` 等帧不会自动到达另一条 CAN。

正确实现有两层：

```text
层 1：PiPER 控制器角色层
  OUTPUT (0xFC) <----> MASTER (0xFA)

层 2：工控机 ROS 所有权层
  policy topic <----> four-state mux <----> leader topic
                                  |
                                  v
                        follower command topic
```

`0xFA/0xFC` 决定主臂控制器本身的物理行为；ROS mux 决定从臂究竟听 policy，还是
听主臂的实测关节输入。两层都完成后，人工接管才真正成立。

---

## 1. 最小词汇表

| 名称 | 含义 | 本项目中的具体表现 |
| --- | --- | --- |
| leader / 主臂 | 人手拖动的机械臂 | `can_l_leader`、`can_r_leader` |
| follower / 从臂 | 执行 policy 或示教动作的机械臂 | `can_l_follower`、`can_r_follower` |
| OUTPUT | PiPER 的运动输出臂角色 | `0x470` 的 Byte0 为 `0xFC`；可接收 JointCtrl/MOVE J |
| MASTER | PiPER 的示教输入臂角色 | `0x470` 的 Byte0 为 `0xFA`；重力补偿并输出主臂控制帧 |
| INFERENCE | mux 从 policy 接收目标 | 两主臂 OUTPUT，四臂按 policy 运动 |
| TAKEOVER | mux 从主臂接收目标 | 两主臂 MASTER，主臂动作驱动两从臂 |
| HOLD | 软件保持当前位置 | 不等于失能；避免从臂失能下落 |
| HOME | 受限速回启动时记录的四臂姿态 | 完成后仍为 HOLD，不自动恢复 policy |

### 1.1 CAN ID 为什么重要

| CAN ID | 方向 | 用途 |
| --- | --- | --- |
| `0x470` | 工控机 → 主控 | 主从角色：`0xFA` MASTER，`0xFC` OUTPUT |
| `0x151` | 工控机 → 主控 | 运动控制 2：standby / CAN control / MOVE J |
| `0x155/0x156/0x157` | MASTER → 工控机 | MASTER 的 J1--J6 控制输入，每帧两关节 |
| `0x159` | MASTER → 工控机 | MASTER 的夹爪控制输入 |
| `0x2A1` | 主控 → 工控机 | 控制模式/状态反馈；`ctrl_mode=0x06` 代表联动示教输入 |
| `0x2A5--0x2A8` | OUTPUT → 工控机 | 常规输出臂关节与夹爪反馈 |

在四 CAN 系统中，左主臂的 `0x155` 只出现在 `can_l_leader`，不会直接控制
`can_l_follower`。`piper_leader_mode_driver.py` 把它解析成 ROS `JointState`，
再由 mux 变成左从臂的目标。

---

## 2. 相关代码地图：建议按此顺序阅读

### 2.1 Piper ROS 层

1. [start_inference_takeover.launch](/home/agilex/cobot_magic/Piper_ros_private-ros-noetic/src/piper/launch/start_inference_takeover.launch)  
   监督推理入口。它选择 supervised profile，并包含四 CAN mux launch。

2. [start_four_can_mux.launch](/home/agilex/cobot_magic/Piper_ros_private-ros-noetic/src/piper/launch/start_four_can_mux.launch)  
   参数总表。先看 CAN 名称、leader/follower node、速度限制、`takeover_*` 和
   `leader_*` 参数。

3. [piper_four_state_mux.py](/home/agilex/cobot_magic/Piper_ros_private-ros-noetic/src/piper/scripts/piper_four_state_mux.py)  
   系统大脑：谁拥有从臂命令、何时冻结 policy、T 何时完成、HOLD/HOME 如何安全。

4. [piper_leader_mode_driver.py](/home/agilex/cobot_magic/Piper_ros_private-ros-noetic/src/piper/scripts/piper_leader_mode_driver.py)  
   每只主臂的唯一 CAN/SDK owner：角色切换、CAN 帧读取、当前位置恢复、主臂状态机。

5. [piper_start_slave_node.py](/home/agilex/cobot_magic/Piper_ros_private-ros-noetic/src/piper/scripts/piper_start_slave_node.py)  
   每只从臂的唯一 CAN/SDK owner。它只执行 mux 发布到 `/teleop/command/joint_*`
   的最终目标。

6. [piper_mode_keyboard.py](/home/agilex/cobot_magic/Piper_ros_private-ros-noetic/src/piper/scripts/piper_mode_keyboard.py)  
   键盘不是直接控制 CAN；它只发布 ROS 模式请求。

### 2.2 OpenPI 层

1. [main.py](/home/agilex/FluxVla/openpi/examples/piper_real/main.py)：启动真机 policy
   客户端，把动作写到 `/teleop/policy/joint_*`。
2. [env.py](/home/agilex/FluxVla/openpi/examples/piper_real/env.py)：读取 follower 状态、
   相机和 HOLD 状态。
3. [hold_aware_policy.py](/home/agilex/FluxVla/openpi/examples/piper_real/hold_aware_policy.py)：
   T/Space 后停止 policy 发布；恢复时丢弃旧 action chunk，重新推理。

阅读时始终检查：一个物理 CAN 只有一个 SDK 实例，一个 follower command topic
只有一个 publisher。

---

## 3. launch 实际启动了什么

监督推理启动命令：

```bash
cd ~/cobot_magic/Piper_ros_private-ros-noetic
source /opt/ros/noetic/setup.bash
source devel/setup.bash

roslaunch piper start_inference_takeover.launch \
  master_left_can:=can_l_leader \
  master_right_can:=can_r_leader \
  puppet_left_can:=can_l_follower \
  puppet_right_can:=can_r_follower \
  auto_enable:=false \
  publish_hz:=200
```

该 launch 最终固定为 5 个节点：

```text
piper_master_left_driver    piper_leader_mode_driver.py
piper_master_right_driver   piper_leader_mode_driver.py
piper_puppet_left_driver    piper_start_slave_node.py
piper_puppet_right_driver   piper_start_slave_node.py
piper_four_state_mux        piper_four_state_mux.py
```

这五个节点不是五个不同控制源：两个 follower driver 都只消费 mux 的最终 topic。

---

## 4. ROS topic 图与数据单位

### 4.1 关键 topic

| Topic | 发布者 | 订阅者 | 意义 |
| --- | --- | --- | --- |
| `/teleop/policy/joint_left/right` | OpenPI | mux | policy 原始候选动作 |
| `/teleop/leader/joint_left/right` | leader driver | mux | MASTER 实测关节输入，或 OUTPUT 实测反馈 |
| `/teleop/command/joint_left/right` | mux | follower driver | 唯一允许控制从臂的最终目标 |
| `/teleop/leader_command/joint_left/right` | mux | leader driver | INFERENCE/HOME/HOLD 时主臂的 OUTPUT 跟随目标 |
| `/teleop/mode` | mux | 两 leader driver | `IDLE/HOLD/INFERENCE/TAKEOVER` |
| `/teleop/mode_request` | 键盘 | mux | I/T 等用户请求 |
| `/teleop/hold_active` | mux | OpenPI | 暂停/恢复 policy action chunk |
| `/teleop/leader_tx_enable` | mux | leader driver | OUTPUT CAN 命令发送门；T 前必须真正排空 |
| `/teleop/leader/tx_quiet_*` | leader driver | mux | OUTPUT 最后一帧已发完的 ACK |
| `/teleop/policy_hold_ack` | OpenPI | mux | OpenPI 已停止旧动作的 ACK |

### 4.2 单位转换

SDK 与 ROS 统一使用 7 维关节数组：

```text
ROS index 0..5 = J1..J6，单位 rad
ROS index 6    = gripper，单位 m

rad    = CAN milli_degree * pi / 180000
meter  = CAN gripper_um / 1000000
CAN 输出 J1..J6 = rad * 180000 / pi
CAN 输出 gripper = meter * 1000000
```

`JointState.position` 少于 7 维、含 NaN/Inf、过期或映射跳变时都会被拒绝；这是
为了避免索引错位让关节或夹爪突跳。

---

## 5. `piper_four_state_mux.py`：所有权和安全状态机

### 5.1 `TimedJointState`

文件开头的 `TimedJointState` 为每个输入保存：

```text
position      最近的 7 维位置
received_at   到达单调时间
```

`sample(now, timeout)` 用于判定数据是否 stale。这个设计比只看 ROS topic 是否存在
更安全：一个已经停止的 publisher 不会继续驱动从臂。

### 5.2 `FourStateMux.__init__`

构造函数完成四件事：

1. 读取限速、对齐容差、超时、HOLD/T 接管参数；
2. 建立最终 follower command publishers；
3. 建立 leader command publisher、模式/ACK publishers；
4. 订阅左右 leader、左右 follower、左右 policy、键盘、OpenPI HOLD ACK。

最重要的 publisher 是：

```python
self._left_command_pub
self._right_command_pub
```

它们是从臂的唯一最终控制出口。

### 5.3 `_set_mode(requested)`

这是 I/T 模式请求的入口。

`requested == "TAKEOVER"` 时，它并不立即把主臂设为 MASTER，而是：

```python
self._mode = "HOLD"
self._hold_active = True
self._set_inference_active(False)
self._set_leader_tx_enabled(True)  # 先保持当前 OUTPUT 位姿
```

含义是：T 的第一瞬间必须先停止 policy、冻结 follower，不能让 policy 与手动输入
同时控制从臂。

`requested == "INFERENCE"` 时，mux 先保持 follower，等待两 leader 都确认 OUTPUT
并对齐，再把 policy 放行。

### 5.4 `_process_pending_takeover(now)`

这是 T 的等待阶段。它依次检查：

```text
policy_hold_ack == true              OpenPI 已停止旧 chunk
takeover_release_delay_s 已到        当前为 1 秒扶臂窗口
left/right tx_quiet == true          所有 OUTPUT JointCtrl/GripperCtrl 已发送完成
leader/follower pose aligned         切换前没有大的姿态差
```

满足后调用 `_begin_takeover_role_switch(now)`。

### 5.5 `_begin_takeover_role_switch(now)` 与 `_process_takeover_role_switch(now)`

前者把 mux mode 改为 `TAKEOVER`，并关闭 leader OUTPUT TX gate。后者持续读取两侧
`/teleop/leader/control_state_*`：

```text
左、右均为 MASTER 或 MASTER_ARMED
    -> TAKEOVER active
    -> mux 改用 leader topic 作为 follower 目标

任一侧 TEACH_FAILED/ERROR/超时
    -> _abort_pending_takeover()
    -> policy 保持停止，四臂维持 powered HOLD
```

双侧同步是必须的：只允许左或右单臂接管会让双臂任务的 policy/人工所有权不一致。

### 5.6 `_target_for_side(side, now)`

这是 run loop 每一周期选择最终 follower 目标的函数：

```text
IDLE/HOLD                       -> 已冻结的 follower anchor
INFERENCE 且 ACTIVE             -> policy
INFERENCE 尚未 ACTIVE           -> follower anchor
TAKEOVER 且 ACTIVE              -> leader
TAKEOVER 尚未 ACTIVE            -> follower anchor
```

这是“话题切换”的实际实现；它不是删除/重建 subscriber，而是在唯一 mux 内部切换
读取哪个 `TimedJointState`。

### 5.7 Space 和 H

`_enter_hold()` 记录当前 follower anchor，保持 enable，不发布 `enable=false`。
因为机械结构下 follower 失能会下落，软件 HOLD 必须是“位置保持”，不能是“断电”。

`_process_home()` 先等待两 leader 回到 OUTPUT，再发布限速 HOME 目标。完成调用
`_finish_home()` 后 mode 是 HOLD，不会自动恢复 policy；这是避免 HOME 结束后自动
突然推理。

---

## 6. `piper_leader_mode_driver.py`：主臂角色与 CAN 实现

每个主臂各运行一个 `PiperLeaderModeDriver` 实例。它是该 `can_*_leader` 的唯一
长期 SDK owner，任何其他脚本都不应同时打开同一主臂 CAN。

### 6.1 `RawMasterObserver`：为什么还要直接读 SocketCAN

SDK 的 `GetArmJointCtrl()` 只给解析后的结果；角色切换失败时，需要确认：

```text
主机是否真的发送了 0x470 0xFA？
控制器是否发送 CAN error frame？
0x2A1 ctrl_mode 是否从 0x01/0x00 转到 0x06？
0x155/156/157/159 实际频率是多少？
```

`RawMasterObserver` 以只读 raw SocketCAN 捕获这些帧：

- `_process_frame()` 解码 J1--J6、夹爪和 `0x2A1`；
- `snapshot()` 返回完整关节组三帧和夹爪的时间戳/频率；
- `transition_report()` 输出一次 T 的 `0x470/0x151/0x191/0x2A1/CAN_ERROR` 时序；
- `rate_text()` 现在额外输出 `0x159`，用于区分夹爪帧与关节帧问题。

它只观察，不向 CAN 发送任何控制帧。

### 6.2 OUTPUT 的安全进入：`_arm_output_at_current_pose()`

OUTPUT 不能直接 enable，因为控制器可能保存着旧目标，第一次使能会跳到旧姿态。
该函数的实际顺序是：

```text
读当前 OUTPUT 实测 pose
-> MotionCtrl_2(STANDBY)
-> JointCtrl/GripperCtrl 预装当前 pose（未使能）
-> MotionCtrl_2(CAN control + MOVE J)
-> 再次预装当前 pose
-> EnableArm(7) 并确认六个 enable bit
-> 再次预装当前 pose
-> 多次重复当前 pose MOVE J 恢复
```

`_seed_output_target()` 负责把 ROS 7 维数组编码成 `JointCtrl`、`GripperCtrl`。
`_send_command()` 才是正常推理过程每个周期发送的限速命令。

### 6.3 MASTER 的进入：`_enter_master_role()`

T 路径调用：

```python
_enter_master_role(allow_disable_fallback=False)
```

`False` 很关键：接管失败时不能为“重试”而 DisableArm，否则操作员手中的主臂会
突然失去支撑。

代码流程：

```text
_prepare_enabled_output_for_master()
  先在当前实测 pose 做 powered OUTPUT hold

_activate_master_stream(reference, "measured-pose OUTPUT hold")
  1) RawMasterObserver 开始本次握手捕获
  2) MasterSlaveConfig(0xFA, 0, 0, 0)
  3) 等待新的 0x155/0x156/0x157 完整关节组
  4) 仅在首次等待失败时：再次 0xFA 后进入 standby
  5) 有完整连续流：验证与切换前 pose 的映射连续性，状态改为 MASTER
```

MASTER 成功的最低证据不是 `MasterSlaveConfig()` 返回，而是持续的完整三关节组。
某些固件只有在操作者开始移动后才发帧；若 `0x2A1` 已明确为 `0x06`，代码会进入
`MASTER_ARMED`，保持当前 follower pose，等待首个真实主臂输入。

### 6.4 OUTPUT 的恢复：`_enter_output_role()`

I、H 或接管失败后需要回 OUTPUT：

```text
MasterSlaveConfig(0xFC, 0, 0, 0)
-> 等待常规 0x2A5--0x2A8 反馈
-> 验证 MASTER/OUTPUT 同一物理姿态的映射连续性
-> _arm_output_at_current_pose()
```

回 OUTPUT 后 mux 仍保持 follower anchor，直到两主臂 OUTPUT 对齐，policy 才恢复。

### 6.5 映射连续性：`_validate_role_pose_continuity()`

切换前后同一物理姿态应有相近的 J1--J6、夹爪值。该函数检查：

```text
nominal tolerance：记录 warning，但可继续
hard tolerance：拒绝接管
```

它防止错误的关节顺序、正负方向、单位或旧数据被当成手动输入，导致从臂突然大幅
运动。

---

## 7. `piper_start_slave_node.py`：从臂为什么不会争夺控制权

每侧 follower driver 的职责很窄：

```text
订阅 /teleop/command/joint_left 或 right
-> 检查长度和 enable 状态
-> MotionCtrl_2(CAN control + MOVE J)
-> JointCtrl(J1..J6)
-> GripperCtrl
```

它不知道当前是 policy 还是人工接管；这个抽象很重要。所有权判断只放在 mux，
follower driver 只执行最终命令。这样可以避免 OpenPI、键盘、主臂 reader 同时向
同一从臂 CAN 发送冲突的 JointCtrl。

从臂启用时同样采用“先读实测反馈、预装当前姿态、再 enable”的顺序，避免旧目标
造成上抬或夹爪突闭。

---

## 8. OpenPI 如何真正停止推理

仅切换 mux 不足以保证安全：Pi05 通常一次推理会产生一个 action chunk，旧 chunk
可能在 T 后继续发布。因此接管使用双层停止：

```text
T -> mux 发布 /teleop/hold_active=true
  -> hold_aware_policy 停止发布并清掉当前 chunk
  -> OpenPI 发布 /teleop/policy_hold_ack=true
  -> mux 收到 ACK 后才关闭 OUTPUT TX 并切 MASTER
```

I 恢复时 `hold_aware_policy.py` 不复用旧 chunk，而是用新的 follower/相机观测重新
向 policy server 请求动作。这样避免“按 T 前算出的动作”在 I 后突然补发。

---

## 9. 最后一次 S-V1.8-8 修复：逐项解释

### 9.1 观察到的现象

左右主臂实际固件均为：

```text
S-V1.8-8
```

在一次固定 MASTER 会话或接管结束回 OUTPUT 后，主臂可能表面上恢复 `0x2Ax`
feedback，却仍残留 torque-control 状态。下一次 T 发送 `0xFA` 时可能没有
`0x155–0x157`。

### 9.2 为什么从 1 秒改为 4 秒

旧流程在第一次 `0xFA` 后只等待 1 秒，就判断“没有关节 MASTER 帧”并发送下一轮
standby 回退。历史成功日志中右主臂完成 MASTER 最长约 3.38 秒；而现场会先感觉
夹爪松开，说明控制器可能还在切换。

最终参数：

```text
master_first_no_stream_retry_s = 4.0
```

这不是把总超时无限延长：总角色超时仍由 `role_timeout_s=5.0` 和 mux 的
`takeover_role_timeout_s=15.0` 约束。它的目的只是避免在正常固件时序内打断第一次
`0xFA`。

### 9.3 为什么默认关闭 `0x191 mode=0`

`0x191 mode=0` 是“恢复主从模式”请求，不是官方 `set_leader_mode()` 的最小 `0xFA`
序列。旧实验把它作为 standby 后 fallback；总线确认它确实发送，但没有带来
MASTER 流，因此最终默认关闭：

```text
master_restore_linkage_activation = false
```

代码没有删除该路径，诊断时可显式设置为 true；默认不会引入额外状态变化。

### 9.4 为什么从 6 次变为 20 次 current-pose MOVE J

AgileX 对 S-V1.8-5--1.8-8 的建议是：从 MASTER 回 OUTPUT 后，重复发送数次“当前
实测关节位置”的 MOVE J，清除残留扭矩模式。旧代码只做 6 次、间隔 20 ms，总时长
约 0.12 秒；最终设置为：

```text
follower_restore_movej_repeats  = 20
follower_restore_movej_period_s = 0.02
```

总恢复时间约 0.4 秒。关键改进是：即便监督 launch 启动时已经看到了 OUTPUT 反馈，
第一次 OUTPUT enable 也会执行这套恢复，而不是只在本进程亲自观察到 `0xFC` 时执行。

### 9.5 修改文件清单

| 文件 | 修改内容 |
| --- | --- |
| [piper_leader_mode_driver.py](/home/agilex/cobot_magic/Piper_ros_private-ros-noetic/src/piper/scripts/piper_leader_mode_driver.py) | 4 秒首次 `0xFA` 等待、默认关闭 0x191、20 次恢复、首次 OUTPUT enable 恢复、0x159 诊断。 |
| [start_four_can_mux.launch](/home/agilex/cobot_magic/Piper_ros_private-ros-noetic/src/piper/launch/start_four_can_mux.launch) | 把最终参数传给左右主臂节点。 |
| [test_leader_role_transition.py](/home/agilex/cobot_magic/Piper_ros_private-ros-noetic/src/piper/test/test_leader_role_transition.py) | Fake SDK 单元测试：OUTPUT enable 恢复与 raw 0x159 报告。 |
| [test_four_can_profiles.py](/home/agilex/cobot_magic/Piper_ros_private-ros-noetic/src/piper/test/test_four_can_profiles.py) | launch 默认值回归断言。 |

固定 MASTER 数据采集、普通 follower-only 推理的控制代码没有因此修改。

---

## 10. 如何读日志和定位问题

### 10.1 成功的 T

```text
TAKEOVER accepted
OpenPI stopped and one-second powered HOLD completed
leader OUTPUT TX gate closed; in-flight command drained
MASTER handshake VERIFIED
leader control state: MASTER
TAKEOVER active
```

### 10.2 失败的 T

```text
MASTER handshake attempt ... bus timeline
MASTER stream timeout snapshot
TEACH_FAILED
A leader role transition failed: TAKEOVER -> HOLD
```

按以下顺序判断：

1. `0x470_TX_ECHO` 缺失：主机 SocketCAN/gs_usb 发送路径问题；
2. 有 `CAN_ERROR`：检查 CAN 物理连接、适配器、队列和错误计数；
3. `ctrl_mode=0x06` 但无 `0x155–0x157`：角色可能进入但 MASTER 输出流异常；
4. `ctrl_mode=0x01` 或 `0x00` 且无 MASTER 帧：控制器未完成/拒绝角色切换；
5. 仅 `0x159` 有频率：夹爪帧存在但关节 MASTER 帧缺失，不能放行接管。

故障现场不要立刻连续按 T、重启 launch 或重新配置 CAN。先执行：

```bash
source /opt/ros/noetic/setup.bash
source ~/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash
rosrun piper piper_log_snapshot.sh
```

该脚本只读归档当前 CAN/ROS 状态和日志。

---

## 11. 推荐学习和调试顺序

1. 先看 `start_four_can_mux.launch`，理解配置如何传到左右节点；
2. 看 `piper_four_state_mux.py` 的 `_set_mode()`、`_process_pending_takeover()`、
   `_process_takeover_role_switch()`、`_target_for_side()`；
3. 看 `piper_leader_mode_driver.py` 的 `_arm_output_at_current_pose()`、
   `_activate_master_stream()`、`_enter_master_role()`、`_enter_output_role()`；
4. 阅读 `RawMasterObserver`，用 CAN 证据而不是手感判断角色是否成功；
5. 阅读 `piper_start_slave_node.py`，理解它为什么只是执行器而不判断所有权；
6. 最后阅读 OpenPI 的 `hold_aware_policy.py`，理解 T 为什么必须停止 action chunk。

建议只做低幅、单关节、两侧分别验证；出现过流、过热、抖动、异常抬升或夹爪突闭
时立即 HOLD/急停并提供日志快照，不要让未支撑的从臂直接失能。

---

## 12. 参考资料

- 本机协议/实现手册：
  `/home/agilex/cobot_magic/Cobot_Magic_人在环遥操作接管功能实现手册_截图归档及CAN协议.docx`
- Piper SDK V2：<https://github.com/agilexrobotics/piper_sdk/blob/master/piper_sdk/interface/piper_interface_v2.py>
- Piper SDK V1：<https://github.com/agilexrobotics/piper_sdk/blob/master/piper_sdk/interface/piper_interface.py>
- 双 Piper 主从说明：<https://github.com/agilexrobotics/piper_sdk/blob/master/asserts/double_piper.MD>
- AgileX seamless switching 固件说明：
  <https://github.com/agilexrobotics/pyAgxArm/discussions/53>
