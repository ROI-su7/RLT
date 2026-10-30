# Pi05 四独立 CAN：采集、两种推理与人工接管完整流程

更新时间：2026-08-27

本文是当前最终操作说明。系统已从出厂的左右两条共享 CAN 改为四条机械臂
独立 CAN；数据采集和两种推理均由不同 launch 命令选择，不再用 `C` 在同一
进程里切采集。最终仍采用实现手册的方案一：**主臂模式 + ROS 话题切换**。

最终只有三种互斥会话：

| 会话 | launch | 主臂 | 从臂命令源 |
| --- | --- | --- | --- |
| 数据采集 | `start_four_can_collect.launch` | 按官方最小序列保存 `0xFA`、重启控制器并只读验证 MASTER；launch 只读 | 左右主臂 |
| 普通推理（默认） | `start_inference_followers.launch` | 不打开主臂 CAN、不跟随 | Pi05 policy |
| 监督推理/接管 | `start_inference_takeover.launch` | 推理时 `0xFC OUTPUT` 跟随；`T` 后 `0xFA MASTER` | policy 或左右主臂 |

三套 launch 绝不能同时运行，也不能与旧 `start_ms_piper.launch`、直接调用 SDK
的脚本或其他 Piper ROS launch 同时运行。

---

## 1. 从出厂双 CAN 到四独立 CAN

### 1.1 硬件接口

```text
can0             底盘，500000 bit/s
can_l_leader     左主臂，1000000 bit/s
can_r_leader     右主臂，1000000 bit/s
can_l_follower   左从臂，1000000 bit/s
can_r_follower   右从臂，1000000 bit/s
```

Linux 接口名最多 15 个字符，必须使用这些短名；`follower` 不能再拼成
`folower`。

### 1.2 为什么四 CAN 必须增加 ROS 仲裁

出厂双臂主从方案中，每一对主从臂共用一条 CAN，主臂在 `0xFA` 模式主动发送
`0x151/0x155/0x156/0x157/0x159`，从臂在同一物理总线上直接接收。改成四条
独立 CAN 后，这些帧不会跨 CAN 自动转发，因此需要：

```text
leader CAN -> leader driver -> ROS leader topic -> mux
policy -------------------------------------------> mux
follower feedback -------------------------------> mux
mux -> 唯一 follower command topic -> follower driver -> follower CAN
```

任何时刻，左右从臂的最终命令只能由当前 mux 发布：

```text
/teleop/command/joint_left
/teleop/command/joint_right
```

OpenPI 只能向仲裁输入发布：

```text
/teleop/policy/joint_left
/teleop/policy/joint_right
```

不能让 OpenPI 直接绕过 mux 控制从臂。

---

## 2. 最终状态和按键

### 2.1 命令行入口

- 数据采集：先在无 Piper 节点时按官方最小序列保存 `0xFA`，完整重启两只主臂
  控制器，再通过
  只读 MASTER 预检；之后启动采集 launch 并发布一次 `/enable_flag=true`，不按
  `C`。
- 普通推理：启动 follower-only launch；Pi05 真机客户端启动时自动发布
  `/enable_flag=true`，自动推理，不按 `I`。
- 监督推理：启动 takeover launch；Pi05 真机客户端启动时自动使能并自动进入
  `INFERENCE`，初次启动不按 `I`。

### 2.2 监督推理键盘

```text
T       冻结模型；两秒后进入人工接管
I       从 HOLD 恢复推理，或人工纠正结束后恢复模型推理
Space   软件 HOLD：冻结实测姿态，保持使能和位置闭环
H       从 HOLD 限速返回本次 launch 捕获的启动姿态，到位后继续 HOLD
Q       先 HOLD，再只退出键盘；机械臂 launch 不退出
```

协议 `0x150` 快速急停会进入阻尼；PiPER 没有关节抱闸，因此可能下落。推理键盘
空格不再发送该命令，而是屏蔽 policy 并以 200 Hz 持续保持位置。确认安全后可
按 `H` 回启动姿态或按 `I` 恢复推理。软件 HOLD 不是安全认证急停，电源、CAN
或驱动故障导致闭环不可用时必须使用物理支撑和现场硬件安全措施。

### 2.3 `T` 的准确时序

1. `T` 按下瞬间停止采用新 policy，两个从臂保持当时姿态。
2. 两只主臂仍以 OUTPUT 保持已与从臂对齐的姿态两秒，供单人操作者把手放好。
3. 两秒后，每个 leader driver 只发送官方
   `MasterSlaveConfig(0xFA, 0, 0, 0)`。
4. 左右主臂都持续产生完整、新鲜的 `0x155/0x156/0x157/0x159` 后，才进入
   `TAKEOVER active`。
5. 此时两个从臂同时改由对应主臂控制；不会出现一侧人工、一侧仍推理。

没有看到 `TAKEOVER active` 时不要强行摇动主臂。单侧失败会整体回到 IDLE，
不能靠连续按 `T` 或放宽验证掩盖故障。

---

## 3. 首次部署或代码更新后的离线构建

以下命令不会启动机械臂，也不会打开 CAN：

```bash
cd ~/cobot_magic/Piper_ros_private-ros-noetic

source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash

python3 -m unittest discover -s src/piper/test -p 'test_*.py' -v

roslaunch --nodes piper start_four_can_collect.launch
roslaunch --nodes piper start_inference_followers.launch
roslaunch --nodes piper start_inference_takeover.launch
```

预期节点数：

```text
数据采集：5
普通推理：3（必须没有 leader driver）
监督推理：5
```

---

## 4. 每次开机的公共准备

### 4.1 上电前安全条件

1. 四只机械臂周围无人、无障碍物，主臂已有物理支撑。
2. 电机过热时先断电并完全冷却，不能用反复使能绕过保护。
3. 确认没有旧 Piper launch、SDK 脚本或 OpenPI 真机客户端占用 CAN/话题。

```bash
ps -ef | grep -E "piper|roslaunch|examples.piper_real" | grep -v grep
```

### 4.2 终端 1：ROS Master

```bash
source /opt/ros/noetic/setup.bash
roscore
```

### 4.3 配置五条 CAN

只能在全部 Piper 节点都停止时执行：

```bash
cd ~/cobot_magic/Piper_ros_private-ros-noetic

sudo bash can_config.sh
ip -br link show type can
```

必须看到：

```text
can0             UP
can_l_leader     UP
can_l_follower   UP
can_r_leader     UP
can_r_follower   UP
```

任一接口 DOWN 时不要启动 Piper。launch 运行期间严禁重新执行 `can_config.sh`，
否则会出现 `SEND_MESSAGE_FAILED (100017)`、接口 not UP、发送队列或 bus-off
问题。

### 4.4 终端 3：启动相机

```bash
cd ~/cobot_magic/camera_ws

source /opt/ros/noetic/setup.bash
source devel/setup.bash

roslaunch astra_camera multi_camera.launch
```

可选预览：

```bash

cd /home/agilex/Desktop/pi05_cobot_work

source /opt/ros/noetic/setup.bash
source ros_ws/devel/setup.bash

python3 scripts/preview_cameras.py
```

`Can't open /dev/bus/usb/: Busy (-6)` 表示同一相机已被另一进程占用。不要同时
重复启动相机 launch 和预览程序；先检查：

```bash
ps -ef | grep -E "astra|camera|preview_cameras" | grep -v grep
```

---

## 5. 数据采集完整流程

数据采集是独立命令行会话，不使用 `C`，最终采用“预置固定 MASTER + ROS
话题转发”。采集 launch 不再把主臂从 OUTPUT 动态切到 MASTER，也不会向主臂
发送使能、失能、位置保持、`0xFA` 或 `0xFC`。

这个方向在 2026-08-25 曾尝试过，但旧 reader 把 `0x155-0x157` 和持续 `0x159`
捆绑为一个模糊条件，失败日志没有逐帧频率；当时角色配置程序还尝试运行时验证，
因此不能据此否定固定 MASTER。本版的区别是：

- 配置工具默认执行官方最小序列：SDK 正常初始化后只保存一次 `0xFA`，不使能
  主臂，也不再默认执行已经反复失败的运行时 OUTPUT→MASTER 切换；
- 保存后彻底断开两只主臂控制器电源 10 秒，再上电并只读复核；
- 启动 ROS 前用原始 SocketCAN 同时统计协议允许的 `0x15x/0x16x/0x17x` 三组；
- 固定 reader 直接读原始 CAN，自动识别三种偏移，以持续完整的三条关节帧作为
  MASTER 必要条件，对应夹爪帧单独告警；
- 预检通过后，采集 launch 全程只读主臂 CAN。

如果使用官方最小 `0xFA` 保存序列并彻底断电重启后，三组合法 ID 仍没有 MASTER 关节控制
帧，那么问题在主臂控制器/固件没有进入 MASTER。受“只能用主臂模式 + 话题
切换”约束，ROS 端没有第二种算法可以凭空生成主臂输入；此时必须升级/修复两只
主臂固件或由 AgileX 确认对应固件的固定 MASTER 配置方式，不能退回反复动态
切换或伪造 MASTER 成功。

### 5.1 无 Piper 节点时：配置并验证固定 MASTER

停止采集、两种推理、OpenPI 真机客户端和所有直连 Piper SDK 程序。`roscore`
和相机可以保留，但四个 Piper CAN 不能有其他 owner。然后执行：

```bash
cd ~/cobot_magic/Piper_ros_private-ros-noetic

source /opt/ros/noetic/setup.bash
source devel/setup.bash

rosrun piper piper_configure_leader_role.py \
  --role master \
  --master-activation restart \
  --left-can can_l_leader \
  --right-can can_r_leader \
  --verify-timeout-s 8
```

配置工具先确认两只主臂仍有 OUTPUT 反馈，再按官方 demo 初始化 SDK、打印固件
版本并只保存 `0xFA`。它不调用 `EnableArm/DisableArm`，也不发送关节/夹爪
目标。正确输出必须包含左右两侧 `saved 0xFA`。随后关闭的是左右主臂控制器
电源，不是只拔 CAN 航空插头；等待 10 秒后再上电。从臂无需断电。

命令成功后立即执行独立只读预检：

```bash
ip -br link show type can

rosrun piper piper_verify_fixed_master.py \
  --left-can can_l_leader \
  --right-can can_r_leader \
  --duration-s 5 \
  --min-joint-hz 10
```

工具会分别打印 offset `0x00/0x10/0x20`。每侧至少一组的三条关节帧必须明显
大于 `10 Hz`，最后显示：

```text
MASTER preflight PASSED for both leaders; fixed collection may be started.
```

对应的夹爪帧会单独显示。若其为 0，关节摇操仍可验证，但正式录制前必须小幅操作
夹爪并确认 `/teleop/leader/joint_*` 的第 7 维能够更新；否则该固件下的夹爪 action
不完整，不能开始采集。

如果三组都为 0 且工具显示 `observed no CAN frame at all`，但恢复 `0xFC` 又能
得到约 200 Hz，则 CAN 名称、波特率和接收线路已经排除，失败点是控制器没有从
`0xFA` 进入协议控制模式 `0x06`。不要启动采集 launch，也不要重复发送
`/enable_flag=true`；记录配置工具打印的固件版本并由 AgileX 确认/升级支持
0x470 联动示教输入的固件。ROS 不能在不伪造主臂输入的情况下补出这些帧。
保存一次诊断快照：

```bash
rosrun piper piper_log_snapshot.sh
```

### 5.2 终端 2：启动只读固定 MASTER 采集配置

```bash
cd ~/cobot_magic/Piper_ros_private-ros-noetic

source /opt/ros/noetic/setup.bash
source devel/setup.bash

roslaunch piper start_four_can_collect.launch \
  master_left_can:=can_l_leader \
  master_right_can:=can_r_leader \
  puppet_left_can:=can_l_follower \
  puppet_right_can:=can_r_follower \
  auto_enable:=false \
  publish_hz:=200 
  #max_joint_velocity_rad_s:=2.5
```

预期五个节点：

```text
/piper_fixed_master_left_reader
/piper_fixed_master_right_reader
/piper_collect_puppet_left_driver
/piper_collect_puppet_right_driver
/piper_fixed_collect_mux
```

其中两个 fixed reader 只使用原始 SocketCAN 接收，自动识别合法控制偏移。启动
后如果主臂静止不再发送新目标，reader 会保持最后姿态；首次启动仍建议在 30 秒
等待窗口内分别移动左右主臂一次，使 reader 完成角色确认。看到以下两行后，再
让操作者把主臂和
从臂放在相近、安全的姿态，执行一次显式使能：

```text
can_l_leader fixed MASTER stream verified at control offset 0x..
can_r_leader fixed MASTER stream verified at control offset 0x..
```

```bash
rostopic pub -1 /enable_flag std_msgs/Bool "data: true"
```

预期关键日志：

```text
Fixed collection enable received; seeding commands from follower feedback
Fixed collection active: verified MASTER inputs own both followers
```

只有最后一行出现后才做低幅映射验收：左只控制左、右只控制右；J1～J6 不交换
且符号一致；夹爪开合方向一致。`2.5 rad/s` 是 ROS 跟随斜率上限，不会修改
主臂重力补偿参数；200 Hz、`queue_size=1` 和 TCP_NODELAY 防止旧命令排队。

检查话题：

```bash
rostopic echo -n 1 /teleop/mode
rostopic hz /teleop/leader/joint_left
rostopic hz /teleop/leader/joint_right
rostopic hz /puppet/joint_left
rostopic hz /puppet/joint_right
```

`/teleop/mode` 必须为 `COLLECT`，四路数据必须持续更新。

### 5.3 可选采集急停键盘

```bash
source /opt/ros/noetic/setup.bash
source ~/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash

rosrun piper piper_mode_keyboard.py _profile:=collect
```

该配置只使用空格急停和 `Q` 退出；`C/I/T/H` 都禁用。

### 5.4 终端 6：连续录制 episode

必须显式使用独立主臂话题作为 action。当前默认值也已改正为这两个话题：

```bash
conda activate aloha

cd ~/cobot_magic/collect_data

source /opt/ros/noetic/setup.bash
source ~/cobot_magic/camera_ws/devel/setup.bash
source ~/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash

python3 new_collect_data.py \
  --dataset_dir ~/data/new_hdf5 \
  --task_name aloha_mobile_dummy \
  --max_timesteps 3900 \
  --episode_idx 0 \
  --frame_rate 30 \
  --next_episode_delay -1 \
  --master_arm_left_topic /teleop/leader/joint_left \
  --master_arm_right_topic /teleop/leader/joint_right \
  --puppet_arm_left_topic /puppet/joint_left \
  --puppet_arm_right_topic /puppet/joint_right
```

录制逻辑：

- 到 `max_timesteps` 后自动保存当前 episode，但程序不退出。
- 出现 `Ready for next episode` 后，按右方向键、`n` 或空格开始下一条。
- 录制中按右方向键、`n` 或空格会提前保存本条；再按一次开始下一条。
- `q` 保存已有帧并退出。
- episode 编号从 `--episode_idx` 自动递增。

每条 HDF5 至少应有：

```text
/observations/qpos    (N, 14)：两个从臂状态
/observations/qvel    (N, 14)
/observations/effort  (N, 14)
/action               (N, 14)：两个主臂示教目标
三路图像              N 帧
```

若 `/action` 与 `/observations/qpos` 总是完全相同，首先检查 recorder 是否错误
订阅了 `/puppet/joint_*` 作为 master action。

### 5.5 正常结束采集

1. 在 recorder 按 `q`。
2. 在采集键盘按 `Q`（若启动了键盘）。
3. 执行失能：

```bash
rostopic pub -1 /enable_flag std_msgs/Bool "data: false"
```

4. 确认从臂停止后，再在采集 launch 终端按 `Ctrl+C`。

固定 MASTER 会保留到主动恢复 OUTPUT。切换到普通 follower-only 推理时可直接
关闭两只主臂电源；切换到监督推理前，必须在所有 Piper 节点停止时恢复：

```bash
rosrun piper piper_configure_leader_role.py \
  --role output \
  --left-can can_l_leader \
  --right-can can_r_leader \
  --verify-timeout-s 8
```

该命令即使当前已经能看到 `0x2Ax` OUTPUT 反馈，也会再次写入 `0xFC` 并进入
待机，确保 Flash 中残留的 `0xFA` 被清除；否则当前运行态虽像 OUTPUT，下一次
主臂掉电重启仍可能重新进入重力补偿 MASTER 状态。

---

## 6. Pi05 公共服务命令

两种推理使用同一个模型服务和真机客户端，区别只在 Piper launch。推荐顺序：

```text
选择并启动一种 Piper launch -> 启动 policy server -> 启动真机客户端
```

### 6.1 Policy Server 终端

用户指定的 30000 checkpoint 脚本：

```bash
cd /home/agilex/FluxVla/openpi
./scripts/serve_pi05_wire_all_data_full_30000.sh

cd /home/agilex/FluxVla/openpi
./scripts/serve_pi05_connect_strings_30000.sh
```

保持该终端运行。脚本监听 `6006`，prompt 为 `insert the wire`。

### 6.2 真机客户端公共命令

```bash
cd /home/agilex/FluxVla/openpi

conda activate pi0
source /opt/ros/noetic/setup.bash
source ~/cobot_magic/camera_ws/devel/setup.bash
source ~/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash

export PYTHONPATH=/home/agilex/FluxVla/openpi:$PYTHONPATH

python -m examples.piper_real.main \
  --host 127.0.0.1 \
  --port 6006 \
  --max-hz 15 \
  --action-horizon 20 \
  --rtc-enabled \
  --rtc-prefix-len 15 \
  --command-publish-hz 200 \
  --command-time-scale 1.0 \
  --action-smoothing-alpha 0.4 \
  --max-arm-velocity-rad-s 0.4 \
  --num-episodes 1 \
  --max-episode-steps 0 \
  --left-command-topic /teleop/policy/joint_left \
  --right-command-topic /teleop/policy/joint_right
```

这两个 topic 参数不可省略：OpenPI 原默认值是 `/master/joint_left/right`，会
绕过当前设计的 policy 输入。客户端默认 `enable_on_reset=true`，启动时已经
向 `/enable_flag` 发送 5 次使能脉冲；不要再另外重复执行 `rostopic pub true`。

---

## 7. 推理方案 A：只有两个从臂推理（默认、最常用）

此配置不会打开两个主臂 CAN。可关闭左右主臂控制器电源以避免无意义发热；
**两个从臂必须保持通电**，因为它们需要执行模型动作。断主臂电时不需要断
从臂电。

### 7.1 终端 2：启动 follower-only 配置

```bash
cd ~/cobot_magic/Piper_ros_private-ros-noetic

source /opt/ros/noetic/setup.bash
source devel/setup.bash

roslaunch piper start_inference_followers.launch \
  puppet_left_can:=can_l_follower \
  puppet_right_can:=can_r_follower \
  publish_hz:=200
```

预期只有：

```text
/piper_inference_left_follower
/piper_inference_right_follower
/piper_policy_follower_mux
```

接着启动第 6.1 节服务，再启动第 6.2 节客户端。无需按 `I`。客户端发出的
`/enable_flag=true` 现在只武装 mux，不会直接使能从臂。mux 等待双 policy 和
双 follower feedback 后，先发布从臂实测姿态种子，再通过私有
`/teleop/follower_enable=true` 原子使能两个从臂：

```text
Follower inference armed; waiting for two policies and two feedback streams
Follower-only inference active; leader CANs are not opened
```

Pi05 第一帧推理超过 10 秒时，不再锁存急停。两个从臂保持失能，日志每 5 秒
明确显示缺少哪一条，例如：

```text
Follower inference still waiting; followers remain disabled;
missing streams: policy_left, policy_right
```

最新 14:30 那次“从臂完全不动”的直接原因就是旧 mux 在 OpenPI 注册发布者后
10 秒仍未收到完整四流，于是提前锁存 E-STOP；再次发布 true 只会被拒绝。本次
已经把输入等待与执行期 stale 分开：启动等待不急停；执行期 policy stale 自动
进入带使能的位置 HOLD，只有 follower 反馈丢失等无法闭环的故障才使用硬件急停。

验证：

```bash
rostopic echo -n 1 /teleop/mode
rostopic echo -n 1 /teleop/follower_enable
rostopic hz /teleop/policy/joint_left
rostopic hz /teleop/policy/joint_right
rostopic hz /puppet/joint_left
rostopic hz /puppet/joint_right
```

mode 应为 `INFERENCE_FOLLOWERS`。该配置没有 `T`，但支持 HOLD/HOME 键盘：

```bash
source /opt/ros/noetic/setup.bash
source ~/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash

rosrun piper piper_mode_keyboard.py _profile:=follower
```

空格进入软件 `HOLD`，不撤销从臂使能，同时令 OpenPI 停止执行并清空 RTC/动作
块和 200 Hz 插补目标；`H` 以默认 `0.25 rad/s` 限速回到本次 launch 捕获的启动
姿态并继续 HOLD；`I` 解除暂停后以当前相机/关节观测重新推理，mux 只接受晚于
本次 `I` 的左右新 policy 消息；`Q` 先 HOLD，再只退出键盘，机械臂 launch 保持
运行。因此 HOLD 期间积累的旧动作不会在恢复瞬间补执行。

---

### 7.2 RLinf Stage 2：两从臂推理、按键打分与采集

此模式使用第 7.1 节同一个 launch，增加 `rlt_stage2:=true`。先退出旧的
Piper launch 和训练进程，保留相机、ROS master 和已经配置好的 JAX 服务。
本模式不打开或控制两个主臂 CAN；不要同时启动独立 `piper_mode_keyboard.py`
或第 6.2 节 OpenPI 真机客户端，Stage 2 自己负责键盘和 policy 发布。

```bash
cd /home/agilex/cobot_magic/Piper_ros_private-ros-noetic
source /opt/ros/noetic/setup.bash
source devel/setup.bash

roslaunch piper start_inference_followers.launch \
  puppet_left_can:=can_l_follower \
  puppet_right_can:=can_r_follower \
  publish_hz:=200 \
  rlt_stage2:=true
```

训练终端（Ray 须在加载 ROS 与 Piper workspace 的环境中启动）：

```bash
source /opt/ros/noetic/setup.bash
source /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash
cd /home/agilex/FluxVla/openpi/RLinf
source .venv-rlt/bin/activate
export PATH="$PWD/.venv-rlt/bin:$PATH"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
export RLINF_NODE_RANK=0
export ROS_MASTER_URI=http://192.168.209.107:11311
export ROS_IP=192.168.209.107
export RLINF_STAGE1_RLT_TOKEN=/home/agilex/FluxVla/openpi/checkpoints/rlt_token_connect_strings_a100_20260927/step_10000/rlt_token.pt
export RLINF_JAX_REF_HOST=127.0.0.1
export RLINF_JAX_PREFIX_HOST=127.0.0.1
export RLINF_JAX_REF_PORT=6006

bash examples/embodiment/run_realworld_async.sh \
  realworld_piper_rlt_stage2_followers
```

已有 replay 使用 `RLINF_REPLAY_LOAD_PATH=/绝对路径/replay_buffer/rank_0`；已有
Stage 2 checkpoint 使用 `runner.resume_dir=/绝对路径/checkpoints/global_step_N`。
仅加载 replay 会重新初始化 Stage 2 权重及更新计数；恢复训练权重需同时指定
checkpoint。没有 checkpoint 时不需要设置 resume_dir。
默认新增数据写入本次新日志目录；下一次启动应加载这个新目录中的 replay。

| 按键 | 行为 |
| --- | --- |
| `I` | 开始/恢复推理，等待新鲜 policy；不自动开始采集 |
| `B` | 从下一完整 action chunk 开始记录 transition；Actor 仍遵守 warmup 门槛 |
| `C` | 成功，终止回合并 HOLD，终止奖励 `+1` |
| `X` | 失败，终止回合并 HOLD，终止奖励 `0` |
| 空格 | 软件 HOLD；回合及 B 采集开关保留，暂停中不写动作 transition；I 恢复 |
| `H` | idle/HOLD 时限速回到两从臂本次 ROS launch 的启动姿态，到位继续 HOLD；回程不写 replay |
| `T` | 无效，不打开主臂 CAN，不进入人工接管 |

验收路径是 `I → B → C`、`I → B → X`、`I → B → 空格 → I → C/X`，以及
`C/X → H` 或 `空格 → H → I`。奖励仍由现有 Stage 2 wrapper/transition/replay
通路保存：C/X 引起 HOLD 时即使当前动作被中断，也会保留独立的终止标签。
部分 chunk 的标签写到最后实际执行的动作；若 C/X 到达 chunk 间隙，则按
`rlt_episode_id` 写到同一回合最近的有效 transition。不会为 HOLD 伪造动作，
不会重复加分，也不会给其他回合贴标签。边界标签补写会打印 `[RLT terminal]`。
终止标记与正奖励应在实际 replay 文件中核对，键盘打印 C 仅代表已接受请求。
不由 ROS mux 存储。H 的“原点”是首次收到的有效从臂姿态，
不是硬编码的所有关节 0 rad；重新启动 ROS 会重新记录，训练回合 reset 不会重置它。

不传 `rlt_stage2:=true` 时保留第 7.1 节普通 OpenPI 自动推理行为。
在 Stage 2 模式下，INFERENCE、I 恢复等待、HOLD 或 HOME 中的从臂反馈超时
都冻结当前目标并保持使能，不自动发布 `/teleop/follower_enable=false` 或
协议 `0x150` QUICK EMERGENCY STOP，以避免无抱闸机械臂因短暂 RX 间隙失去支撑。
反馈恢复不会自动继续运动，须再次按 I；反馈仍超时时 I 会被拒绝。
日志包含左右反馈年龄和超时阈值，便于区分 ROS 调度间隙与持续断流。
这不保证真实电源/CAN 故障下仍可保持位置；显式硬件急停请求仍会急停和失能。
四臂接管 launch、主臂驱动及原 T 接管状态机保持原有实现。

### 7.3 Stage 2 Actor 纠正链路日志

两从臂与四臂 Stage 2 共用同一诊断链路。Actor 实际接管时，终端约每秒打印
一组 `[RLT actor chain]`，每组包含当前 chunk/update、RTC 偏移、手臂/夹爪
限幅、Actor EMA alpha、taper 步数，以及执行器的 EMA alpha、速度限制和插补时长。
随后逐左右 J1～J6、夹爪打印同一控制样本的以下字段：

| 字段 | 含义 |
| --- | --- |
| `step / plan_step` | 当前执行 chunk 下标 / RTC 对齐后的 JAX 计划下标，均从 0 开始 |
| `ref / target` | 原始对齐 JAX reference 的绝对关节目标 / 执行器滤波后接受的绝对目标 |
| `raw` | 原始 Actor 输出减去对应 reference |
| `clip` | 限幅后的残差 |
| `smooth` | Actor 残差 EMA 平滑后，尚未 taper |
| `taper / taper_weight` | taper 后的残差及该控制点的衰减权重 |
| `rtc` | RTC 对齐/边界融合后的环境请求动作减去对应 reference |
| `executor` | 执行器滤波后接受的目标减去同一个 reference |
| `retained` | 同一控制点的 `executor / raw`，原始残差接近零时为 n/a |

手臂单位为 rad，夹爪单位为 m。Actor EMA/taper 仅作用于手臂关节，夹爪仍
限幅但不做这两项滤波。每一行选当前 chunk 该关节 `abs(raw)` 最大的有效
Actor 控制点，各列均使用此同一点；不将不同时间点的最大值混在一起。
无效 padding、人工接管步骤和未采用 Actor 的计划前缀不会进入该表。

例如 `raw=+0.06000 ... executor=+0.01000 rad retained=16.7%` 表示 Actor 提出
0.06 rad 偏移，而执行器接受的目标相对该 reference 只增加 0.01 rad。
`executor` 是 ROS mux 之前的控制目标，不是 CAN 最终位置或机械臂实测运动；
ROS mux 仍可能进一步限速，电机也有跟踪误差。此差值包含 RTC 和执行器的
影响，不能把它全部归因为 Actor 的净效果，也不是运行一次纯 JAX 的反事实对照。

诊断张量在 EnvWorker 中取出，不进入 replay；不改变 Actor/Critic 的动作
变换、奖励或采样比例。代码更新后需重启 RLinf 训练进程；已有训练权重须
按 checkpoint 恢复，不要仅为看日志重新初始化 Stage 2。ROS/JAX 服务无需修改。

## 8. 推理方案 B：四臂跟随 + `T` 人工接管

此配置用于模型偏离目标时的小范围人工纠正和后续干预数据采集。四臂都必须
通电，四条 CAN 都必须 UP；主臂启动姿态应有支撑。

监督推理启动前必须让对应主从臂接近同一姿态。推荐先用已经验证的数据采集
profile 把从臂带到主臂姿态，在有支撑的位置停止采集，保持四臂不动，再启动
监督推理。mux 的启用前安全门默认要求每个关节差不超过 `0.08 rad`、夹爪差不
超过 `0.015 m`；超差时会打印左右 J1..J6/G 的实际 delta，主臂保持失能，不再
用电机扭矩强行追赶一个大初始差值。

### 8.1 终端 2：启动监督推理配置

```bash
cd ~/cobot_magic/Piper_ros_private-ros-noetic

source /opt/ros/noetic/setup.bash
source devel/setup.bash

roslaunch piper start_inference_takeover.launch \
  master_left_can:=can_l_leader \
  master_right_can:=can_r_leader \
  puppet_left_can:=can_l_follower \
  puppet_right_can:=can_r_follower \
  publish_hz:=200
```

预期五个节点：

```text
/piper_master_left_driver
/piper_master_right_driver
/piper_puppet_left_driver
/piper_puppet_right_driver
/piper_four_state_mux
```
stage 2:
cd /home/agilex/FluxVla/openpi/RLinf

source /opt/ros/noetic/setup.bash
source /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash
export RLINF_ROS_PYTHONPATH="$PYTHONPATH"

source .venv-rlt/bin/activate
export PATH="$PWD/.venv-rlt/bin:$PATH"
export PYTHONPATH="$PWD:$RLINF_ROS_PYTHONPATH"
hash -r

export CUDA_VISIBLE_DEVICES=0
export RLINF_NODE_RANK=0
export RLINF_COMM_NET_DEVICES=eno1
export ROS_MASTER_URI=http://192.168.209.107:11311
export ROS_IP=192.168.209.107

python -c "import sys, hydra, rospy, sensor_msgs; print(sys.executable); print('Environment OK')"

ray stop --force
ray start --head \
  --port=6379 \
  --node-ip-address=192.168.209.107 \
  --num-cpus=24 \
  --num-gpus=1 \
  --disable-usage-stats

export RLINF_STAGE1_RLT_TOKEN=/home/agilex/FluxVla/openpi/checkpoints/rlt_token_connect_strings_a100_20260927/step_10000/rlt_token.pt
export RLINF_JAX_REF_HOST=127.0.0.1
export RLINF_JAX_PREFIX_HOST=127.0.0.1
export RLINF_JAX_REF_PORT=6006

unset RLINF_REPLAY_LOAD_PATH
unset RLINF_STAGE2_RESUME_DIR

bash examples/embodiment/run_realworld_async.sh \
  realworld_piper_rlt_stage2_ac_mlp \
  runner.resume_dir=null \
  algorithm.replay_buffer.load_path=null \
  algorithm.replay_buffer.min_buffer_size=1000 \
  algorithm.rlt_schedule.warmup_min_size=1000 \
  algorithm.rlt_schedule.warmup_post_collect_updates=5000 \
  algorithm.rlt_schedule.max_actor_reference_arm_delta=0.03 \
  algorithm.replay_buffer.sample_strategy=uniform




export RLINF_STAGE1_RLT_TOKEN=/home/agilex/FluxVla/openpi/checkpoints/rlt_token_connect_strings_a100_20260927/step_10000/rlt_token.pt

export RLINF_JAX_REF_HOST=127.0.0.1
export RLINF_JAX_PREFIX_HOST=127.0.0.1
export RLINF_JAX_REF_PORT=6006

export RLINF_REPLAY_LOAD_PATH=/home/agilex/FluxVla/openpi/RLinf/logs/20261010-16:42:38-realworld_piper_rlt_stage2_ac_mlp/replay_buffer/rank_0

unset RLINF_STAGE2_RESUME_DIR

bash examples/embodiment/run_realworld_async.sh \
  realworld_piper_rlt_stage2_ac_mlp \
  runner.resume_dir=null \
  algorithm.replay_buffer.min_buffer_size=1000 \
  algorithm.rlt_schedule.warmup_min_size=1000 \
  algorithm.rlt_schedule.warmup_post_collect_updates=5000 \
  algorithm.replay_buffer.sample_strategy=uniform


### 8.1.1 终端 6：人工接管数据采集（LeRobot 格式）

按第 8.6 节的 schema 录制跨越 `T` 的整段 episode：推理段和接管段都保存，
每帧带 `intervention_mask` 区分。可以和第 8.4 节的接管操作同时进行。

解释器必须用 `aloha` conda 环境。它是 Python 3.8.19，和 ROS Noetic 同版本，
`source` 之后能拿到 `rospy`/`cv_bridge`，同时自带 `dm_env`/`h5py`；系统
`/usr/bin/python3` 没有 `dm_env`，直接跑会 `ModuleNotFoundError`。

```bash
source /opt/ros/noetic/setup.bash
source ~/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash
cd ~/cobot_magic/collect_data

~/miniconda3/envs/aloha/bin/python takeover_collect_data.py \
  --dataset_dir ~/data \
  --task_name piper_takeover \
  --instruction "insert the wire into the slot" \
  --require_takeover
```

录制生命周期由 mux 状态驱动，不需要在这个终端里手动开始或分段：

```text
/teleop/mode 进入 INFERENCE  -> 自动开始录制
按 T（第 8.2 节键盘）        -> 转入 TAKEOVER，录制不中断
按 H                         -> 结束本条 episode（H 同时让四臂回 HOME）
在本终端按 q                 -> 结束整个会话并自动转换为 LeRobot 格式
```

预期输出，`takeover_frames` 应随按 `T` 之后的操作递增：

```text
mode=INFERENCE; recording starts now
frame 412  mode=TAKEOVER  phase=ACTIVE  takeover_frames=87
H pressed (/teleop/home_request); ending episode.
Saved 411 frames (87 takeover) in 6.3s: /home/agilex/data/piper_takeover/episode_0
```

`--require_takeover` 会丢弃完全没有接管帧的 episode。加 `--no_convert`
可以只留 HDF5，之后再手动转换。

会话结束时自动调用 `hdf5_lerobot.py --includeIntervention` 完成转换。
之所以要落一层 HDF5：`rospy`/`cv_bridge` 只存在于 Python 3.8，LeRobot 只
存在于 `FluxVla/openpi/.venv` 的 Python 3.11，同一个进程装不下这两者。
转换默认输出到 `<dataset_dir>/<task_name>_lerobot`，`repo_id` 为
`local/<task_name>`，可用 `--lerobot_dir` / `--lerobot_name` 覆盖。

转换会先 `rmtree` 目标目录再从 task 目录下**全部** HDF5 episode 重建，
所以多次会话使用同一 `--task_name` 时数据是累积的，不会丢失既有 episode。

产出的 LeRobot 数据集除标准字段外还包含：

```text
action           mux 仲裁后实际执行的动作（推理段=policy，接管段=人）
policy_action    /teleop/policy/joint_*，模型当时想做什么
leader_action    /teleop/leader/joint_*，人手主臂位姿
intervention     0/1，等价于 /teleop/mode == TAKEOVER
```

注意 J4/J5/J6 在快速运动下从臂有 360~395 ms 跟随滞后、峰值误差可达 70°，
这是 MOVE J 规划器约 25 rad/s² 的加速度上限造成的，无法用 ROS 侧参数消除。
接管纠正时动作要慢，否则 `leader_action` 会大幅领先从臂实际到达的状态，
干预数据里会混入机器人执行不了的目标。

### 8.2 终端 5：监督键盘

```bash
source /opt/ros/noetic/setup.bash
source ~/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash

rosrun piper piper_mode_keyboard.py _profile:=inference
```

### 8.3 启动模型

启动第 6.1 节 Policy Server，再启动第 6.2 节真机客户端。首次启动不按 `I`。
客户端自动使能；mux 自动让两个主臂以 OUTPUT 低速追上对应从臂，只有四臂
对齐后才放行 policy：

```text
Command-line supervised profile has fresh inputs; requesting INFERENCE
INFERENCE requested: ... holding followers until both leaders confirm OUTPUT alignment
INFERENCE active: all four arms aligned; policy commands released
```

若主臂只抬一点就失能，应查看同一时刻是否报告过流、过温、堵转或 enable
timeout。当前 driver 在任何使能前先用实测关节/夹爪预装目标并限制对齐速度；
保护仍触发时必须排查负载、支撑、散热和机械阻力，不能提高速度或循环使能。

2026-08-27 最近一次监督推理的直接故障不是右从臂：右从臂日志已确认六电机
使能。真正的终止源来自 `/piper_master_left_driver`：左主臂固件同时报告
`J5 driver overcurrent`、`J5 driver error` 和 `J5 stall protection`，mux 因此
执行 `INFERENCE -> HOLD`，并按安全设计没有放行两个从臂。该固件保护不能由
ROS 自动清除；应停止监督 launch、支撑机械臂，检查 J5 负载/卡阻并待其冷却后
对左主臂控制器完整断电重启。若在主从 delta 已通过上述安全门后仍复现，按
电机/驱动硬件故障处理，不要循环发布 enable。

### 8.4 人工接管与恢复推理

1. 推理正常且只需小幅纠正时按 `T`。
2. 两秒内扶住左右主臂。
3. 看到 `TAKEOVER active` 后再摇动主臂纠正从臂。
4. 完成纠正后按 `I`。
5. 系统将主臂恢复 OUTPUT，重新对齐并在确认后恢复 policy。

如果出现：

```text
did not start 0x155-0x157 control feedback after 0xFA
A leader role transition failed
TAKEOVER master-role transition timed out
```

说明固件没有产生可持续验证的 MASTER 流；从臂不会接受半套接管。按空格进入
HOLD，不要重复按 `T`。

### 8.5 急停和 HOME

- 异常运动立即按空格；四臂冻结实测姿态、保持使能和位置闭环；OpenPI 暂停且
  丢弃已有动作块，policy 不只是被 mux 屏蔽。
- 确认空间安全后按 `H`，系统以限速返回本次 launch 启动时捕获的四臂姿态，
  到位后继续 HOLD；按 `I` 才恢复 policy。
- `Q` 先 HOLD 再退出键盘，不会关闭 launch 或撤销使能。
- HOME 不是硬编码全零，也不会使用未验证的控制器旧目标。
- 软件 HOLD 不是安全认证急停；电源/CAN/驱动故障使闭环不可用时仍需物理支撑。

### 8.6 记录人工干预

监督 mux 将实际执行的最终动作兼容发布到：

```text
/master/joint_left
/master/joint_right
```

现有 recorder 可记录该动作；正式用于 intervention retraining 时，推荐后续在
数据格式中分别保存：

```text
policy_action       /teleop/policy/joint_left/right
executed_action     /master/joint_left/right
intervention_mask   /teleop/mode == TAKEOVER
```

本轮没有擅自改变现有 HDF5 字段结构。

---

## 9. 两种推理的正常停机

1. 先停止 OpenPI 真机客户端（`Ctrl+C`；默认会发布当前反馈姿态作为 hold）。
2. 立即显式失能：

```bash
rostopic pub -1 /enable_flag std_msgs/Bool "data: false"
```

3. 监督推理键盘按 `Q`；普通推理键盘若启动也按 `Q`。
4. 再停止当前唯一 Piper launch、Policy Server、相机和 roscore。

不要先停止 CAN owner 后再发送失能，因为那时 SDK socket 已关闭，可能看到
shutdown 阶段的 `SEND_MESSAGE_FAILED (100017)`。

---

## 10. 三种会话之间怎样切换

这不是热切换。必须：

```text
停止动作源 -> /enable_flag=false -> 停止当前 Piper launch
-> 确认无 Piper/SDK owner -> 必要时重新执行 can_config.sh -> 启动另一套 launch
```

- 普通推理：可以只给从臂通电；主臂可断电。
- 数据采集：四臂都需通电，主臂必须预置、重启并预检为固定 MASTER。
- 监督推理：四臂都需通电，主臂必须先恢复为 OUTPUT。
- 从普通推理切到采集/监督推理前，应先关闭普通推理 launch，再给主臂上电并
  确认 leader CAN 为 UP。
- 采集与监督推理不能热切：固定 MASTER 采集结束后，必须停止所有 Piper 节点，
  用 `piper_configure_leader_role.py --role output` 恢复并验证 OUTPUT，才能启动
  监督推理。

---

## 11. 今日日志问题与对应修正

### 11.1 CAN 不存在、DOWN、`100017`

历史日志中的：

```text
CAN socket ... does not exist
RuntimeError: CAN port ... is not UP
SendCanMessage(SEND_MESSAGE_FAILED (100017))
```

属于接口命名/UP 状态、运行中重配 CAN 或发送队列问题。处理顺序：急停、失能、
停止 owner，再检查/重配；不能在 launch 存活时执行 `can_config.sh`。

```bash
ip -details -statistics link show can_l_follower
ip -details -statistics link show can_r_follower
tc -s qdisc show dev can_l_follower
tc -s qdisc show dev can_r_follower
```

### 11.2 主臂抬起、数秒后失能或过热

最新日志证明该现象不是关节映射超差或硬件过热保护，而是运行时 `0xFA` 没有
产生持续 `0x155-0x157`，随后 mux 主动 fail-close。最终修正：

- 数据采集彻底删除运行时 OUTPUT→MASTER；
- 采集 reader 只读已经配置并验证为固定 MASTER 的主臂；
- launch 前用原始 CAN 逐项预检 `0x15x/0x16x/0x17x` 三组合法控制 ID；
- follower 仍先读当前 J1～J6/夹爪并预装同一目标，再进入 MOVE J 和使能；
- 固件报告过流、过温、碰撞、driver error、stall 或 NTC 过温时拒绝使能/运动；
- 六个电机必须全部反馈使能成功。

### 11.3 `MASTER` 假就绪或 T 后仍不能控制

固定采集要求左右两侧至少一组合法的三条关节控制帧持续刷新，对应夹爪帧单独
报告；预检不通过时 launch 根本不启动。监督推理的 T 路径仍要求运行时角色
持续流和映射连续性通过，T 期间从臂先冻结，双侧都成功后才统一换 owner。

### 11.4 普通推理从臂使能但不动

14:30 日志中，OpenPI 在 14:31:55 注册 policy 发布者并发出 true，从臂成功
使能，但旧 mux 在 10 秒内未获得双 policy + 双 follower 的新鲜组合，于
14:32:05 锁存 quick stop。本次把 follower driver 的使能入口改为 mux 私有
`/teleop/follower_enable`：四流齐全前从臂保持失能并持续等待；激活后 policy
stale 进入带使能的 HOLD，只有 follower feedback stale 等无法闭环的故障才进入
硬件急停。日志同时逐项显示 `policy_left/right`、`follower_left/right` 哪条缺失。

### 11.5 采集 action 记录错误

`new_collect_data.py` 的 master 默认话题曾指向 `/puppet/joint_*`，会把 follower
qpos 重复写成 action。本次改为 `/teleop/leader/joint_*`，完整命令中仍显式写
出，避免以后改配置时误用。

---

## 12. 从 2 CAN 到 4 CAN：新增和修改代码

### 12.1 最终流程使用的 Piper ROS 文件

| 文件 | 类型 | 重点 |
| --- | --- | --- |
| `can_config.sh` | 修改 | 五条 CAN 的 USB 映射、短接口名、bitrate、TX queue 和失败重试。 |
| `src/piper/launch/start_four_can_mux.launch` | 新增 | 四臂公共底层，CAN owner、话题 remap、角色/速度/超时参数。 |
| `src/piper/launch/start_four_can_collect.launch` | 新增并定稿 | 固定 MASTER 只读采集；两个 reader、两个 follower driver、fixed mux；默认 `2.5 rad/s`。 |
| `src/piper/launch/start_inference_followers.launch` | 新增并优化 | 默认只从臂推理；完全不构造 leader driver；从臂改接私有 gated enable。 |
| `src/piper/launch/start_inference_takeover.launch` | 新增 | 监督推理 wrapper；自动 INFERENCE，保留 T/I/H/空格。 |
| `src/piper/scripts/piper_leader_mode_driver.py` | 新增并优化 | V2 SDK 主臂 `0xFA/0xFC`、OUTPUT 跟随、持续流验证、映射检查、实测姿态预装、保护检查。 |
| `src/piper/scripts/piper_four_state_mux.py` | 新增并优化 | IDLE/HOLD/COLLECT/INFERENCE/TAKEOVER，原子 owner 切换、T 两秒、带使能 HOLD/HOME、监督启用前姿态安全门和 HOLD→I 新 policy 握手。 |
| `src/piper/scripts/piper_policy_follower_mux.py` | 新增并优化 | follower-only policy 仲裁；四流就绪后才使能、policy stale 自动 HOLD、HOLD→I 必须收到本次恢复后的左右新 policy。 |
| `src/piper/scripts/piper_fixed_master_reader.py` | 新增并定稿 | 原始 CAN 只读固定 MASTER；自动识别 `0x15x/0x16x/0x17x`，夹爪帧单独报告。 |
| `src/piper/scripts/piper_fixed_collect_mux.py` | 新增并定稿 | 固定 MASTER 话题到双 follower 的唯一 owner；200 Hz、queue 1、默认 `2.5 rad/s`。 |
| `src/piper/scripts/piper_configure_leader_role.py` | 新增并修正 | 默认官方 `0xFA` 保存+控制器重启，打印固件；可选待机诊断；恢复并验证 `0xFC`。 |
| `src/piper/scripts/piper_verify_fixed_master.py` | 新增 | 原始 SocketCAN 只读预检，统计三组合法控制偏移和总线实际 ID。 |
| `src/piper/scripts/piper_log_snapshot.sh` | 新增 | 只读归档 CAN、ROS graph、话题关系与本次节点日志。 |
| `src/piper/scripts/piper_mode_keyboard.py` | 新增 | 推理 Space=HOLD、H=HOME、I=恢复、Q=HOLD 后退键盘；监督配置另有 T。 |
| `src/piper/scripts/piper_start_slave_node.py` | 修改 | 独立 CAN、跳过可选 SDK 查询、日志节流、使能确认、当前姿态预装、夹爪保持和急停互斥。 |
| `src/piper/test/test_leader_role_transition.py` | 新增 | 无 CAN mock 测试：拒绝单次时间戳、接受持续完整流、0xFA 不附加 standby。 |
| `src/piper/test/test_four_can_profiles.py` | 新增 | 无 CAN 检查三套 launch 隔离和采集 action 默认话题。 |
| `src/piper/FOUR_CAN_MUX.md` | 更新 | Piper ROS 内部架构、角色握手和 fail-closed 说明。 |

### 12.2 采集与 OpenPI 接口

| 文件 | 类型 | 重点 |
| --- | --- | --- |
| `collect_data/new_collect_data.py` | 修改 | episode 保存后继续等待右键/`n`/空格；master action 默认改为 `/teleop/leader/joint_*`。 |
| `examples/piper_real/main.py` | 四 CAN 接口使用并优化 | 必须用命令行把输出指向 `/teleop/policy/joint_*`；将 action broker 包装为 HOLD-aware policy。 |
| `examples/piper_real/env.py` | 四 CAN 接口使用并优化 | 订阅双 follower/三相机及 `/teleop/hold_active`；HOLD 时停止插补发布并拒绝旧 epoch 动作。 |
| `examples/piper_real/hold_aware_policy.py` | 新增 | HOLD 时暂停策略执行；恢复时 reset RTC/action chunk，并用当前观测重新推理。 |
| `scripts/serve_pi05_wire_all_data_full_30000.sh` | 当前服务脚本 | 启动指定 Pi05 30000 checkpoint 和 6006 服务。 |

### 12.3 历史文件，不在最终三套 launch 中

```text
start_ms_piper_mode2.launch
piper_mode2_ros_driver.py
```

这些是实体按钮或早期 mode2 实验遗留。没有删除它们以避免擅自破坏现有工作，
但最终命令不要调用。旧 `start_ms_piper.launch` 和
`piper_start_ms_node.py` 也不属于四 CAN 最终入口。

---

## 13. 如果自己重新实现，必须重点写哪些部分

### 13.1 CAN 和 publisher 唯一所有权

- 一个物理臂 CAN 只有一个长期 SDK owner。
- 一个从臂最终 command topic 只有一个 mux publisher。
- OpenPI 只写 raw policy input；recorder 只订阅，不控制。
- 三个 launch 必须互斥，并各自限制允许的状态。

### 13.2 主臂状态机

- `0xFC OUTPUT` 用于监督推理时让主臂跟随 policy。
- `0xFA MASTER` 用于采集和 T 接管的重力补偿示教输入。
- 采集的 `0xFA` 默认按官方 SDK demo 最小序列保存，随后完整断开两只主臂控制器
  电源 10 秒并重启，再通过原始 CAN 预检；采集节点始终只读。
- T 接管的运行时 `0xFA` 使用官方 `MasterSlaveConfig`，随后进入待机以激活联动
  示教输入模式；不要伪造流成功。
- 角色成功必须由持续、完整的 `0x15x/0x16x/0x17x` 任一合法三关节帧组证明，
  不能只看函数返回或单帧；对应夹爪帧必须独立检查数据完整性。
- 切换前后检查同一物理姿态的 J1～J6/夹爪映射；不自动换序或翻符号。

### 13.3 安全使能

- 使能前读当前实测姿态，依次执行 standby、预装当前目标、MOVE J、重复当前
  目标、使能确认；夹爪也必须保留当前开度。
- 六个 motor enable bit 全部成立后才能放行命令。
- 读取低速 FOC 和主控状态；低压、过流、过温、碰撞、stall 等拒绝运动。
- CAN 发送、enable/disable、command 和 shutdown 要互斥。

### 13.4 原子接管

- T 的第一步是冻结 policy 和 follower target。
- 两秒只是单人扶住主臂的准备窗口，不应继续模型推理。
- 双 MASTER 验证前两个 follower 都保持冻结。
- 双侧成功后一次性改 owner；任一侧失败整体 fail-closed。
- I 恢复时先双 OUTPUT 并重新对齐，再放 policy。

### 13.5 数据和单位

```text
SDK joint_1..joint_6 -> ROS joint0..joint5
SDK gripper          -> ROS joint6
rad = milli_degree * pi / 180000
meter = gripper_um / 1000000
```

JointState 必须恰有至少 7 个有限 position；少维、NaN/Inf、stale 均拒绝。
高频 ROS topic 使用 `queue_size=1`、`tcp_nodelay=True`，200 Hz 日志必须关闭或
节流。

### 13.6 数据集区分

- `qpos` 记录 follower 实测状态。
- 示范采集的 `action` 记录 leader 目标。
- 监督干预最好同时保存 raw policy、executed action 和 intervention mask。
- 每个 episode 验证 14 维关节/夹爪、时间长度和三路图像完整性。

---

## 14. 验证状态和实机验收顺序

截至本文更新时间，已经在**不启动机械臂、不打开 CAN**的条件下完成：

- Python 语法编译；
- 无 CAN 单元测试（含固定 MASTER 只读性、2.5 rad/s 默认值和 gated enable）；
- 三个 launch XML/节点展开，节点数 5/3/5；
- `catkin_make` 全部通过；
- 今日 ROS 日志中 CAN DOWN/100017、角色流丢失、stale、使能保护等时序审计。

由于明确要求不能启动机械臂，以下实机验收尚需现场逐项执行，不能把离线测试
表述为真实运动验证：

```text
1. 数据采集：固定 MASTER 原始帧预检 -> COLLECT -> 左右低幅单关节/夹爪映射 -> 连续 episode
2. 普通推理：仅三节点 -> INFERENCE_FOLLOWERS -> 主臂 CAN 未打开
3. 监督推理：四臂低速对齐 -> 短时 policy -> T -> TAKEOVER active
4. 接管后小幅纠正 -> I -> 双 OUTPUT 对齐 -> 恢复 policy
5. Space -> HOLD 且 enable 保持 true -> H 限速 HOME -> 到位后仍为 HOLD
```

任何一步出现抖动、异常抬升、夹爪突闭、过流、过热或单侧状态异常，立即急停并
使用物理支撑，不进入下一步；不要在机械臂未被支撑时直接发布 enable=false。

---

## 15. 独立日志归档与故障复现

以后出现“不动、失能、角色切换失败、单侧异常”时，先不要反复 true、重配 CAN
或重启节点。在故障现场仍保持原样时，新开终端执行：

```bash
source /opt/ros/noetic/setup.bash
source ~/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash

rosrun piper piper_log_snapshot.sh
```

默认生成：

```text
~/cobot_magic/piper_runtime_logs/YYYYMMDD_HHMMSS/
  summary.txt
  rosnodes.txt
  rostopics.txt
  topic_info.txt
  topic_values.txt
  important_ros_lines.txt
  ros_logs/
```

脚本只读：不会 publish ROS 话题、不会配置 CAN、不会启停进程。它会保存 CAN
错误计数/队列统计、相关进程、ROS graph、关键 topic 的 publisher/subscriber、
模式/使能值以及本次 `~/.ros/log/latest` 中的 Piper 节点日志。把该时间戳目录
提供给排障即可，不再需要逐个复制所有终端。

Policy server 和 OpenPI 客户端的普通 stdout 不一定进入 ROS 日志；需要保留时
可在各自命令末尾使用 `tee`，例如：

```bash
mkdir -p ~/cobot_magic/piper_runtime_logs/manual

./scripts/serve_pi05_wire_all_data_full_30000.sh 2>&1 \
  | tee ~/cobot_magic/piper_runtime_logs/manual/policy_server.log
```

真机客户端命令同样可在最后追加：

```bash
2>&1 | tee ~/cobot_magic/piper_runtime_logs/manual/openpi_client.log
```

## 16. 参考资料

- 本机实现手册：
  `/home/agilex/cobot_magic/Cobot_Magic_人在环遥操作接管功能实现手册_截图归档及CAN协议.docx`
- 官方 Piper SDK：<https://github.com/agilexrobotics/piper_sdk>
- 官方 V2 接口：<https://github.com/agilexrobotics/piper_sdk/blob/master/piper_sdk/interface/piper_interface_v2.py>
- 官方 V1 接口：<https://github.com/agilexrobotics/piper_sdk/blob/master/piper_sdk/interface/piper_interface.py>
- 官方双 Piper 说明：<https://github.com/agilexrobotics/piper_sdk/blob/master/asserts/double_piper.MD>
- AgileX seamless switching 固件说明：
  <https://github.com/agilexrobotics/pyAgxArm/discussions/53>

本次实现没有复制第三方控制代码。SDK 角色、CAN ID、单位和接口行为以 AgileX
官方仓库及本机安装的 `piper_sdk 0.6.1` 为准；四独立 CAN 的 ROS 转发、仲裁和
fail-closed 是本项目应用层实现。
