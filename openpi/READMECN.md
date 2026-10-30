# OpenPI 数据采集、训练与推理流程

本文档整理了从机械臂数据采集、数据重播、HDF5 数据转换、代码同步、服务器训练、权重拷贝到本地推理的完整流程。

## 目录

- [1. 数据集录制](#1-数据集录制)
- [2. 数据重播](#2-数据重播)
- [3. HDF5 转 LeRobot 数据格式](#3-hdf5-转-lerobot-数据格式)
- [4. 快速生成归一化统计](#4-快速生成归一化统计)
- [5. 服务器环境检查](#5-服务器环境检查)
- [6. 代码同步](#6-代码同步)
- [7. 模型训练](#7-模型训练)
- [8. 权重拷贝](#8-权重拷贝)
- [9. 模型推理](#9-模型推理)
- [10. 当前终端登录 Codex](#10-当前终端登录-codex)

## 1. 数据集录制

### 1.1 启动 roscore

新开一个终端：

```bash
roscore
```

### 1.2 启动相机

新开一个终端：

```bash
cd ~/cobot_magic/camera_ws
source devel/setup.bash
roslaunch astra_camera multi_camera.launch
```

### 1.3 预览相机画面

新开一个终端：

```bash
cd /home/agilex/Desktop/pi05_cobot_work
source /opt/ros/noetic/setup.bash
source ros_ws/devel/setup.bash
python3 scripts/preview_cameras.py
```

### 1.4 启动机械臂数采 launch

新开一个终端，进入 `Piper_ros` 目录，激活 CAN 模块并启动机械臂数采：

```bash
cd ~/cobot_magic/Piper_ros_private-ros-noetic/
bash can_config.sh
source devel/setup.bash
roslaunch piper start_ms_piper.launch mode:=0 auto_enable:=false
```

### 1.5 激活采集环境

新开一个终端：

```bash
conda activate aloha
```

### 1.6 启动数据采集

进入采集代码目录：

```bash
cd ~/cobot_magic/collect_data
```

查看 `collect_data.py` 参数：

```bash
python collect_data.py -h
```

采集数据示例：

```bash
python collect_data.py --dataset_dir ~/data --max_timesteps 500 --episode_idx 0
```

建议采用：
使用 `new_collect_data.py` 采集数据示例：

```bash
python new_collect_data.py --dataset_dir ~/data/new_hdf5 --max_timesteps 1500 --episode_idx 0
```

## 2. 数据重播

重要注意事项：

```text
需要拔掉主臂的航空插头。
需要拔掉主臂的航空插头。
需要拔掉主臂的航空插头。
```

### 2.1 启动 roscore 和从臂

新开一个终端：

```bash
roslaunch piper start_ms_piper.launch mode:=1 auto_enable:=true
```

### 2.2 激活虚拟环境

新开一个终端：

```bash
conda activate aloha
```

### 2.3 重播数据集

进入采集代码目录：

```bash
cd ~/cobot_magic/collect_data/
```

发布彩色图、主臂、从臂消息，并让从臂重复执行录制数据：

```bash
python replay_data.py --dataset_dir ~/data --task_name aloha_mobile_dummy --episode_idx 37
```

只发布彩色图和从臂消息：

```bash
python replay_data.py --dataset_dir ~/data --task_name aloha_mobile_dummy --only_pub_master --episode_idx 0
```

## 3. HDF5 转 LeRobot 数据格式

执行转换脚本：

```bash
python /home/agilex/cobot_magic/collect_data/hdf5_lerobot.py \
  --datasetDir /home/agilex/data/five_data/aloha_mobile_dummy \
  --datasetName local/cobot_magic \
  --targetDir /home/agilex/data/lerobot_data/five_data \
  --instruction "Pick up the wire ends one by one and press each wire into the grooves on the corresponding posts."
```

参数说明：

- `--datasetDir`：原始 HDF5 数据集目录。
- `--datasetName`：转换后的数据集名称。
- `--targetDir`：LeRobot 格式数据输出目录。
- `--instruction`：任务指令文本。

## 4. 快速生成归一化统计

`pi05_wire` 训练需要 `state` 和 `actions` 的归一化统计。不要对本数据集使用
`scripts/compute_norm_stats.py`：该脚本会逐帧解码三路视频，而统计并不依赖图像，完整计算可能需要数小时。

使用快速脚本。它直接扫描 LeRobot Parquet 中的 `observation.state` 和 `action` 两列，跳过视频解码，并复现
`pi05_wire` 使用的 Aloha 状态/夹爪转换、10-step action horizon 与 delta-action 变换。

```bash
cd /home/agilex/FluxVla/openpi

PYTHONPATH=/home/agilex/FluxVla/openpi/src \
python scripts/compute_lerobot_parquet_norm_stats.py
```

本机实测 600 个 episode 约 12 秒完成。输出文件为：

```text
assets/pi05_wire/new_total_data/norm_stats.json
```

默认读取的数据集为 `/home/agilex/data/lerobot_data/new_total_data`。若路径或输出目录不同，可显式指定：

```bash
PYTHONPATH=/home/agilex/FluxVla/openpi/src \
python scripts/compute_lerobot_parquet_norm_stats.py \
  --dataset-root /path/to/lerobot_dataset \
  --output-dir assets/pi05_wire/new_total_data
```

## 5. 服务器环境检查

登录服务器后，可以用下面的命令确认 `openpi` 是否能正常导入：

```bash
~/openpi/envs/pi0/bin/python -c "import openpi; print(openpi.__file__)"
```

## 6. 代码同步

### 6.1 从 cobot_magic 机器上传代码到服务器

在 `cobot_magic` 机器执行：

```bash
cd ~/openpi

rsync -av \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='envs' \
  --exclude='logs' \
  --exclude='outputs' \
  --exclude='checkpoints' \
  --exclude='wandb' \
  ./ train:~/openpi/
```

### 6.2 从个人电脑同步代码到 192.168.85.129

在个人电脑执行：

```bash
cd /home/xz/IDEA/openpi

rsync -av \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='envs' \
  --exclude='logs' \
  --exclude='outputs' \
  --exclude='checkpoints' \
  --exclude='wandb' \
  ./ wangyumu@192.168.85.129:~/openpi/
```

### 6.3 从个人电脑同步代码到 train

在个人电脑执行：

```bash
cd /home/xz/IDEA/openpi

rsync -av \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='envs' \
  --exclude='logs' \
  --exclude='outputs' \
  --exclude='checkpoints' \
  --exclude='wandb' \
  ./ train:~/openpi/
```

## 7. 模型训练

### 7.1 准备 Slurm 提交脚本

登录服务器后执行：

```bash
cd ~/openpi
mkdir -p slurm logs
nano slurm/train_pi05_wire.sbatch
```

写入以下内容：

```bash
#!/bin/bash
#SBATCH --job-name=pi05_wire
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

ssh myserver
cd ~/FluxVla/openpi
export PYTHONPATH="$HOME/FluxVla/openpi/src:$PYTHONPATH"
export WANDB_MODE=disabled                     # wandb 禁用（环境层）
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export NCCL_P2P_DISABLE=1

srun -p gpu --gres=gpu:8 --cpus-per-task=64 --mem=500G --time=72:00:00 \
  ./envs/pi0/bin/python scripts/train.py pi05_connect_strings \
    --exp-name=pi05_connect_strings_20260923 \
    --no-wandb-enabled \
    --overwrite
```

### 7.2 保存并退出 nano

在 `nano` 中：

1. 按 `Ctrl + O` 保存。
2. 底部显示 `File Name to Write: slurm/train_pi05_wire.sbatch` 时，直接按 `Enter`。
3. 按 `Ctrl + X` 退出。

### 7.3 提交训练任务

```bash
sbatch slurm/train_pi05_wire.sbatch
```

### 7.4 查看任务状态

```bash
squeue -u wangyumu
```

状态说明：

- `ST` 为 `R`：任务已经开始运行。
- `ST` 为 `PD`：任务仍在排队。

### 7.5 查看日志

提交后记下 job id，例如 `3415610`。

查看报错日志：

```bash
tail -f ~/openpi/logs/pi05_wire_3423691.err
```

查看输出和 loss：

```bash
tail -f ~/openpi/logs/pi05_wire_3417909.out
```

### 7.6 取消任务

```bash
scancel 3415633
```

其中 `3415633` 替换为实际 job id。

## 8. 权重拷贝

以下命令在 `cobot_magic` 机器上执行。

### 8.1 拷贝完整实验目录

```bash
rsync -avP \
  wangyumu@192.168.85.129:~/openpi/checkpoints/pi05_wire/7_6_pi05_wire \
  /home/agilex/openpi/checkpoints/pi05_wire/
```

### 8.2 只拷贝指定 checkpoint

例如只拷贝 `10000` 这个 checkpoint：

```bash
mkdir -p /home/agilex/openpi/checkpoints/pi05_wire/7_6_pi05_wire

rsync -avP \
  wangyumu@finaicpu01:~/openpi/checkpoints/pi05_wire/7_6_pi05_wire/10000 \
  /home/agilex/openpi/checkpoints/pi05_wire/7_6_pi05_wire/
```

## 9. 模型推理

### 8.1 启动 roscore

新开一个终端：

```bash
roscore
```

### 8.2 启动相机

新开一个终端：

```bash
cd ~/cobot_magic/camera_ws
source devel/setup.bash
roslaunch astra_camera multi_camera.launch
```

### 8.3 预览相机画面

新开一个终端：

```bash
cd /home/agilex/Desktop/pi05_cobot_work
source /opt/ros/noetic/setup.bash
source ros_ws/devel/setup.bash
python3 scripts/preview_cameras.py
```

### 8.4 启动从臂

新开一个终端：

```bash
cd ~/cobot_magic/Piper_ros_private-ros-noetic/
bash can_config.sh
source devel/setup.bash
roslaunch piper start_ms_piper.launch mode:=1 auto_enable:=true
```

### 8.5 激活虚拟环境

```bash
conda activate aloha
```

### 8.6 启动 policy server

终端一：

```bash
cd /home/agilex/openpi

UV_CACHE_DIR=/tmp/uv-cache uv run python -u scripts/serve_policy.py \
  --env ALOHA \
  --port 6006 \
  --default_prompt "insert the wire" \
  policy:checkpoint \
  --policy.config pi05_wire \
  --policy.dir /home/agilex/FluxVla/openpi/checkpoints/pi05_wire/7_16_pi05_wire/30000

 cd /home/agilex/FluxVla/openpi

  PYTHONPATH=/home/agilex/FluxVla/openpi/src \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /home/agilex/FluxVla/openpi/RLinf/.venv-rlt/bin/python -u scripts/
  serve_policy.py \
    --env ALOHA \
    --port 6006 \
    --default_prompt "insert the wire" \
    policy:checkpoint \
    --policy.config pi05_wire \
    --policy.dir  cd /home/agilex/FluxVla/openpi

   cd /home/agilex/FluxVla/openpi
  ./scripts/serve_pi05_wire_35000.sh




```

### 8.7 启动真实机械臂推理程序

终端二：

```bash
cd /home/agilex/openpi
conda activate pi0
source /opt/ros/noetic/setup.bash
source ~/interbotix_ws/devel/setup.bash
export PYTHONPATH=/home/agilex/openpi:$PYTHONPATH

python -m examples.piper_real.main \
  --host 127.0.0.1 \
  --port 6006 \
  --action-horizon 5 \
  --num-episodes 1 \
  --max-episode-steps 3000

python -m examples.piper_real.main     --host 127.0.0.1     --port 6006     --max-hz 15     --action-horizon 20     --rtc-enabled     --rtc-prefix-len 15     --command-publish-hz 200     --command-time-scale 1.0     --action-smoothing-alpha 0.4     --max-arm-velocity-rad-s 0.4     --num-episodes 1     --max-episode-steps 0

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
    --reset-on-exit \
    --reset-duration-s 3 \
    --reset-timeout-s 12 \
   # --so101-takeover

```

#### 推理退出后自动复位

在真实机械臂推理命令中增加 `--reset-on-exit`。程序会在 episode 开始时记录双臂的首个有效 14 维关节反馈。推理中第一次按 `Ctrl+C` 只停止策略并保持当前关节位置；程序会保留在等待状态，第二次按 `Ctrl+C` 才会通过现有 ROS 命令 topic 平滑回到记录的起始位。

```bash
python -m examples.piper_real.main \
  --host 127.0.0.1 --port 6006 \
  --max-hz 15 --action-horizon 10 \
  --num-episodes 1 --max-episode-steps 0 \
  --reset-on-exit \
  --reset-duration-s 5 \
  --reset-timeout-s 15
```

不要在 Piper ROS 驱动运行期间另开 `pyAgxArm` 直连 CAN 并调用 `move_p()`，否则会出现两个控制源争用 CAN。当前实现复用推理时已经验证过的 `/master/joint_left` 与 `/master/joint_right` topic；底层 Piper ROS 驱动再执行相应的关节运动。

若工装要求回到另一个已验证的安全位，可以覆盖自动记录的起始位。参数顺序为“左臂 6 关节、左夹爪、右臂 6 关节、右夹爪”：

```bash
--reset-home-action "l1,l2,l3,l4,l5,l6,l_gripper,r1,r2,r3,r4,r5,r6,r_gripper"
```

复位期间会暂停 200 Hz 策略插补线程，避免其短时目标覆盖 `--reset-duration-s` 的慢速复位轨迹。先在无工件、低速、急停可用的条件下验证该安全位；如复位超时，程序会停止发送新的复位轨迹并记录错误，需要人工确认机械臂状态后再继续。

### 8.8 OpenPI Piper 的 RTC 式动作流水线（可选）

原版 OpenPI 的 `ActionChunkBroker` 会在一个动作块用完后同步请求下一块动作；模型推理期间不会发布新的 `/master/joint_*` 命令。Piper 示例新增的 `--rtc-enabled` 会在当前动作块还剩 `--rtc-prefix-len` 个动作时，后台请求下一块，同时继续发布当前剩余动作；从而避免动作块切换时的控制空档。

```bash
python -m examples.piper_real.main \
  --host 127.0.0.1 --port 6006 \
  --max-hz 30 \
  --action-horizon 10 \
  --rtc-enabled \
  --rtc-prefix-len 5 \
  --num-episodes 1 \
  --max-episode-steps 3000
```

这属于**执行侧 RTC**，不是 FluxVLA 的模型前缀 RTC：当前 OpenPI 的 `pi05_wire` 模型没有 `prev_actions` / `prefix_len` 输入，也没有使用这种条件进行训练。因此它能减少推理等待造成的命令断档，但不会使模型按上一段动作前缀进行条件预测。若日志持续出现 `holding the final command`，说明推理时间超过剩余动作缓冲：增大 `--rtc-prefix-len`（但必须小于 `--action-horizon`），或降低 `--max-hz`。

后台结果接入前会按真实推理耗时执行时间对齐，日志会输出 `RTC action chunk aligned by ... policy steps`。这是必要的：若模型以较早观测预测出动作后，机器人已继续执行了一段旧动作，直接执行新结果的第 0 个动作会回到过去姿态，造成小幅“伸出—缩回”。

### 8.9 30 Hz 策略 + 200 Hz 线性插补与输出平滑（可选）

`examples/piper_real/main.py` 的 `max_hz` 默认值为 **15 Hz**。它是策略输出到 ROS 的默认上限；原版同步动作块推理时，实际频率还可能低于它。数据为 30 Hz 时可传 `--max-hz 30`。

当设置 `--command-publish-hz 200` 后，策略仍以 30 Hz 推理；Piper 输出端会在相邻策略关节目标之间以线性插值发送 200 Hz 的臂关节参考。`--action-smoothing-alpha` 是对每次新策略目标的指数平滑系数：`1.0` 表示不平滑，`0.6` 是推荐的初始测试值。默认不插补夹爪，避免改变开闭时机。

`--command-time-scale` 会放慢每个策略目标的线性过渡时间，默认 `1.0`；`1.5` 表示以原来 1.5 倍时间过渡。`--max-arm-velocity-rad-s` 是 12 个臂关节的硬速度上限，`0` 表示关闭；例如 `0.4` 表示每个臂关节最大 0.4 rad/s。夹爪不参与这两个限制。

```bash
python -m examples.piper_real.main \
  --host 127.0.0.1 --port 6006 \
  --max-hz 30 \
  --action-horizon 20 \
  --rtc-enabled --rtc-prefix-len 10 \
  --rtc-boundary-blend-steps 5 \
  --command-publish-hz 200 \
  --command-time-scale 1.5 \
  --action-smoothing-alpha 0.8 \
  --max-arm-velocity-rad-s 0.4 \
  --num-episodes 1 \
  --max-episode-steps 0
```

`--max-episode-steps 0` 表示不按步数自动结束，使用 `Ctrl+C` 人工停止。推理运行时检查：

```bash
rostopic hz /master/joint_right
```

插补开启后，连续执行段应接近 200 Hz。若动作反应过慢，先将 `--max-arm-velocity-rad-s` 调大到 `0.6`，再将 `--command-time-scale` 调回 `1.0`；若抖动仍明显，将 `--action-smoothing-alpha` 从 `0.8` 降到 `0.7`。每次只改一个参数。

`--rtc-boundary-blend-steps` 控制新旧动作块的边界融合长度，默认 `5`；在 30 Hz 下约为 0.17 秒。只融合左右臂的 12 个关节，夹爪仍保持离散开闭。若块切换处仍有轻微停顿可试 `7`；若动作变得迟缓则调回 `3`。

### 8.10 RTC、动作平滑与动作块融合：代码改动和原理

这三项改动均位于 `examples/piper_real/`，其完整执行链路为：

```text
策略服务器输出动作块
    -> RTC 后台预取下一动作块
    -> 新动作块按推理延迟进行时间对齐
    -> 新旧动作块边界融合
    -> 单步动作送入 PiperRealEnvironment
    -> EMA 平滑、速度限制、线性插补
    -> ROS /master/joint_left 与 /master/joint_right
```

#### 8.10.1 代码位置和调用关系

| 功能 | 代码位置 | 主要内容 |
| --- | --- | --- |
| RTC 调度器 | `examples/piper_real/rtc_action_chunk_broker.py` | `RealtimeActionChunkBroker`：后台请求、动作块切换、超时保持与 reset 同步。 |
| RTC 入口和参数 | `examples/piper_real/main.py` | `rtc_enabled`、`rtc_prefix_len`、`rtc_boundary_blend_steps`、`action_horizon`、`max_hz`。 |
| 动作平滑与插补 | `examples/piper_real/env.py` | `PiperRealEnvironment.apply_action`、`_interpolated_command_locked`、`_run_command_interpolator`。 |
| 平滑参数入口 | `examples/piper_real/main.py` | `command_publish_hz`、`command_time_scale`、`action_smoothing_alpha`、`max_arm_velocity_rad_s`、`interpolate_gripper`。 |

`main.py` 在 `--rtc-enabled` 启用时，以 `RealtimeActionChunkBroker` 替代 OpenPI 默认的 `ActionChunkBroker`；未启用时仍保留原始同步动作块行为。`env.py` 负责将 broker 每次提供的一条 14 维动作发布到 Piper ROS。

14 维动作的关节约定如下：

```text
左臂：action[0:7]，其中 action[6] 是左夹爪
右臂：action[7:14]，其中 action[13] 是右夹爪
双臂关节（不含夹爪）：[0:6] 和 [7:13]
```

#### 8.10.2 RTC：推理与旧动作块尾部并行

原始 `ActionChunkBroker` 在一个动作块耗尽后同步请求下一动作块；网络通信和模型推理期间，控制端没有新的命令可发。`RealtimeActionChunkBroker` 将该过程改为异步预取：当当前块只剩 `prefix_len` 步时，调用 `_start_prefetch` 创建后台线程请求下一块；控制主线程继续执行旧块的剩余动作。当前块耗尽后，若预取结果已返回便立即切换。

```text
第 0 到 H-P 步：执行当前动作块
剩余 P 步时：后台请求下一块，同时继续执行当前块尾部
当前块结束时：切换到已返回的新块；若仍未返回，则保持最后一个动作
```

其中 `H = action_horizon`，`P = rtc_prefix_len`，控制频率为 `f = max_hz`。理想情况下，预取缓冲时间 `P / f` 应不小于模型推理与网络传输耗时；若频繁打印 `holding the final command`，说明缓冲不足，应增大 `P`（仍须小于 `H`）、降低控制频率或缩短模型推理时间。

首次推理仍必须同步等待，因为尚不存在可执行的旧动作。后台推理失败会在控制线程中重新抛出；若推理尚未完成，则保持最后一条关节目标而不是停止发布命令，以避免控制器失去参考值。

此处实现的是**执行侧 RTC（execution-side RTC）**，并非模型前缀 RTC。它不向 PI0.5 传入 `prev_actions` 或 `prefix_len`，因此不改变模型条件输入或训练目标；它只通过客户端调度隐藏推理延迟。若要使用模型前缀 RTC，必须修改模型结构并用对应输入重新训练。

#### 8.10.3 动作块时间对齐和边界融合

新动作块以发起请求时刻的观测为条件生成，但返回时机器人已经继续执行了旧块的一段动作。若将新块的第 0 步直接发送给机器人，该动作对应的是过去的时刻，机器人可能发生“先伸出、再缩回”的回撤或振荡。

`_align_pending_actions` 根据实际推理耗时计算偏移量：

```text
offset = (当前时间 - 请求时间) * control_hz
```

对于机械臂关节，使用线性插值从新块的 `offset` 时刻开始重采样，舍弃已经过期的开头部分；对于夹爪，使用零阶保持（选择对应整数步的值），避免合成无意义的“半开/半闭”状态。若延迟已超过该动作块的有效 horizon，则丢弃该结果并保持最后一个动作。

时间对齐后，`_blend_chunk_boundary` 对新旧块交界的前 `K = rtc_boundary_blend_steps` 步进行融合。对第 `k` 步（从 1 开始）和每个非夹爪关节：

```text
w_k = k / K
q_blend[k] = (1 - w_k) * q_last + w_k * q_new[k]
```

其中 `q_last` 是上一动作块实际下发的最后一个关节目标，`q_new[k]` 是时间对齐后的新块动作。该方法降低位置目标的突变；夹爪不参与融合，保持离散开闭语义。`K` 过小可能仍有边界跳变，过大则会引入额外滞后；30 Hz 下，`K=5` 约为 0.17 秒，可作为初始值。

需要注意：线性边界融合只改善位置连续性，并不能严格保证速度或加速度连续；更严格的轨迹连续性可使用三次/五次多项式或 minimum-jerk 轨迹拼接。

#### 8.10.4 动作平滑、限速和高频插补

当 `command_publish_hz > 0` 时，`PiperRealEnvironment` 启动独立 ROS 发布线程。策略仍以 `max_hz` 产生目标，而发布线程以 `command_publish_hz` 连续输出命令。每次收到策略动作后，执行顺序如下：

1. **指数滑动平均（EMA）**：对 12 个机械臂关节应用

   ```text
   q_filtered = alpha * q_policy + (1 - alpha) * q_filtered_previous
   ```

   `alpha = action_smoothing_alpha`；`alpha=1.0` 表示关闭平滑，较小的值可抑制模型单点抖动，但会增加响应滞后。夹爪不参与 EMA。

2. **每关节硬速度限制**：以当前插补命令为参考，将本次目标变化裁剪为

   ```text
   delta_max = max_arm_velocity_rad_s * command_interpolation_duration_s
   q_target = q_current + clip(q_filtered - q_current, -delta_max, delta_max)
   ```

   这比仅依靠 EMA 更能限制单个异常动作点造成的大幅跳变；设为 `0` 时关闭限制。

3. **线性插补与高频发布**：在 `command_interpolation_duration_s` 内，从当前命令线性过渡到新目标，并按照 `command_publish_hz` 发布。`command_time_scale / max_hz` 决定插补持续时间。默认夹爪在插补结束前保持旧值、结束时切换；设置 `--interpolate-gripper` 后才对夹爪线性插补。

建议的真机调参顺序是：先低速、短 horizon 验证，再单独调整一个参数。若抖动明显，可逐步降低 `action_smoothing_alpha`；若过慢，先提高 `max_arm_velocity_rad_s` 或降低 `command_time_scale`；若动作块交界不平滑，再适度增加 `rtc_boundary_blend_steps`。

## 10. 当前终端登录 Codex

在当前终端中设置临时 `CODEX_HOME`，并在退出时自动登出和清理：

```bash
export CODEX_HOME="$(mktemp -d -t codex-session.XXXXXX)"
trap 'codex logout >/dev/null 2>&1; rm -rf "$CODEX_HOME"' EXIT
```
