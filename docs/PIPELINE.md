# 完整流程：从环境搭建到 RLT（Cobot Magic）

本文按执行顺序列出从零复现的每一步。所有命令都在仓库根目录 `$RLT_ROOT`（默认 `~/RLT`）下执行。
每一步的详细说明在各子目录的文档里，文中给出链接。

```text
0 环境 ─▶ 1 CAN/相机/ROS ─▶ 2 示教采集 ─▶ 3 LeRobot 转换 ─▶ 4 π0.5 微调 ─▶ 5 推理服务
      ─▶ 6 Stage 1 RL Token ─▶ 7 接头位姿检测器 ─▶ 8 Stage 2 预热采集 ─▶ 9 Stage 2 在线 RL
```

## 0. 环境

```bash
git clone --recursive https://github.com/ROI-su7/RLT.git ~/RLT && cd ~/RLT
bash env/install_all.sh      # 01 apt → 02 ROS+catkin → 03 openpi/.venv → 04 RLinf/.venv-rlt → 05 基座权重
bash env/verify.sh
```

- 已克隆但没带 `--recursive`：`git submodule update --init --recursive`（`install_all.sh` 会自动执行）。
- `env/steps/02_ros.sh` 会把 `patches/` 中的本地改动应用到 Piper ROS 与 piper_sdk_demo 子模块，然后 `catkin_make`。
- 只装某一步：`bash env/install_all.sh --only 04`。

## 1. CAN、相机与 ROS

Cobot Magic 使用四条独立 CAN（左右主臂、左右从臂）加一条底盘 CAN。改线、命名与 udev 规则见
`openpi/README_PIPER_FOUR_CAN.md` 第 1 节；`cobot_magic/tools/` 中有出厂的 CAN/相机辅助脚本。

```bash
bash cobot_magic/Piper_ros_private-ros-noetic/can_config.sh   # 激活并命名 CAN（需要 sudo）
bash cobot_magic/tools/camera_serial.sh                      # 查看相机序列号
```

三种互斥的 Piper 会话（同一时间只能运行一个）：

| 会话 | 命令 |
|---|---|
| 示教采集 | `roslaunch piper start_four_can_collect.launch` |
| 普通推理 | `roslaunch piper start_inference_followers.launch` |
| 监督推理 / 人工接管（RLT 使用） | `roslaunch piper start_inference_takeover.launch` |

（先 `source cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash`。）

## 2. 示教数据采集

```bash
source /opt/ros/noetic/setup.bash
source cobot_magic/camera_ws/devel/setup.bash
source cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash
/usr/bin/python3 cobot_magic/collect_data/collect_data.py --dataset_dir ~/rlt_data/hdf5 --task_name connect_strings --episode_idx 0
```

监督推理 + 接管段的采集（录制执行动作、策略动作、主臂动作与接管掩码）用
`cobot_magic/collect_data/takeover_collect_data.py`，它在结束时用 `openpi/.venv` 自动转换为 LeRobot。

## 3. 转换为 LeRobot 数据集

```bash
openpi/.venv/bin/python cobot_magic/collect_data/hdf5_lerobot.py \
  --datasetDir ~/rlt_data/hdf5/connect_strings --datasetName local/all_wire_data \
  --targetDir ~/rlt_data/lerobot_data \
  --instruction "connect the light strings and the battery box to the splitter cable so the lights turn on."
ln -s ~/rlt_data/lerobot_data openpi/lerobot_data     # 训练配置读取 ./lerobot_data/<name>
```

参考数据集规模：350 条示教，30 fps，三路相机（cam_high / cam_left_wrist / cam_right_wrist），14 维关节。

## 4. π0.5 全参微调（openpi，JAX）

配置在 `openpi/src/openpi/training/config.py`（`pi05_connect_strings`：基座 `./checkpoints/pi05_base`，batch 32，40k 步）。

```bash
cd openpi
# 归一化统计：直接读 parquet，不解码视频（与 compute_norm_stats.py 结果一致，最大差异约 5e-3）
.venv/bin/python scripts/compute_lerobot_parquet_norm_stats.py --dataset-root lerobot_data/all_wire_data \
  --output-dir assets/pi05_connect_strings/all_wire_data --action-horizon 50
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 .venv/bin/python scripts/train.py pi05_connect_strings \
  --exp-name=pi05_connect_strings_$(date +%Y%m%d) --no-wandb-enabled --overwrite
```

- 全参微调需要多卡（参考：8×A100 或 8×3090 的 Slurm 脚本 `openpi/slurm/*.sbatch`）；单张 24 GB 显卡只能跑 LoRA 配置 `pi05_wire`。
- 已知问题：在本环境中 `scripts/compute_norm_stats.py` 会在“先导入 openpi 配置、再导入 lerobot”时段错误（原机器上同样复现），
  因此用上面的 parquet 版本。
产物：`openpi/checkpoints/pi05_connect_strings/<exp>/<step>/`（我们部署的是 step 30000）。

## 5. 原生 JAX 推理服务（参考动作 + RL Token 前缀）

π0.5 始终以原生 JAX 运行、全程冻结。普通推理（不带 RL）：

```bash
bash openpi/scripts/serve_pi05_connect_strings_30000.sh      # 端口 6006
```

RLT 使用的“紧凑”服务一次前向同时返回参考动作块与 Stage 1 编码后的 z_rl（需要先完成第 6 步）：

```bash
RLINF_PI05_JAX_DIR=<π0.5 checkpoint> RLINF_ROBOT_RLT_TOKEN=<rlt_token.pt> \
  bash openpi/RLinf/examples/embodiment/serve_piper_rlt_compact.sh
```

纯 π0.5 真机客户端（不经 RLinf）：`openpi/examples/piper_real/main.py`。

## 6. Stage 1：RL Token

在微调数据上训练 RL Token 自编码器（π0.5 冻结，只训练编码器/解码器；JAX 前向 + PyTorch FSDP）。

```bash
cd openpi                      # 数据配置中的 ./lerobot_data 相对 openpi/ 解析
source RLinf/.venv-rlt/bin/activate
export PYTHONPATH="$PWD/RLinf:$PWD/src" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
torchrun --standalone --nproc-per-node=<GPU 数> RLinf/toolkits/realworld_check/train_piper_rlt_token_native_jax.py \
  --jax-checkpoint <π0.5 checkpoint> --dataset-dir ~/rlt_data/lerobot_data/all_wire_data \
  --output-dir ~/rlt_checkpoints/rlt_token --config-name pi05_connect_strings --steps 10000 --accumulate 8
```

显存：冻结的 π0.5（JAX）与约 7.5 亿参数的 RL Token（fp32 主权重 + AdamW）同卡常驻，**单张 24 GB 显卡放不下**，
至少需要 2 张 24 GB（FSDP 分片优化器状态）或单卡 ≥ 40 GB。参考：8×A100 训练 10000 步。Slurm 模板：`openpi/RLinf/slurm/train_piper_rlt_token_native_jax_8x*.sbatch`。
产物 `step_<N>/rlt_token.pt`（format_version 2）。

## 7. 接头位姿检测器（右腕相机）

Stage 2 的状态与奖励塑形使用右腕相机上的接头关键点 `[dx, dy, dangle, visible]`。

```bash
cd openpi/RLinf
.venv-rlt/bin/python toolkits/wire_head_pose/extract_contact_frames.py --dataset ~/rlt_data/lerobot_data/all_wire_data --out ~/rlt_data/wire_head_pose
.venv-rlt/bin/python toolkits/wire_head_pose/annotate.py --root ~/rlt_data/wire_head_pose          # 标注两端接头
.venv-rlt/bin/python toolkits/wire_head_pose/keypoint_model.py train --root ~/rlt_data/wire_head_pose \
  --out ~/rlt_data/wire_head_pose/head_pose_net_v2.pt
```

训练好的权重路径用 `RLT_HEAD_POSE_WEIGHTS` 指定（默认 `~/rlt_data/wire_head_pose/head_pose_net_v2.pt`）。

## 8–9. Stage 2：真机 RLT

```bash
# 终端 1：ROS（roscore + 相机 + Piper 监督推理/接管 launch，tmux 会话 rlt_stage2_ros）
bash openpi/RLinf/examples/embodiment/start_piper_ros_tmux.sh
# 终端 2：π0.5 紧凑推理服务（第 5 步）
# 终端 3：RLT
export RLINF_STAGE1_RLT_TOKEN=<rlt_token.pt>
unset RLINF_STAGE2_RESUME_DIR RLINF_REPLAY_LOAD_PATH          # 全新开始
bash openpi/RLinf/examples/embodiment/run_piper_rlt_stage2_single.sh
```

键盘：`I` 开始推理，`B` 进入关键窗口并记录，`T` 人工接管（主臂遥操作），`C` 成功，`X` 失败，`H` 回初始位，空格暂停。

调度（配置 `openpi/RLinf/examples/embodiment/config/realworld_piper_rlt_stage2_ac_mlp.yaml`）：

1. 预热：按 B 后 π0.5 继续控制并记录，直到回放达到 1000 条；
2. 学习器做 3000 次纯 BC 更新后，按 B 才由 Actor 控制；
3. 之后 Q 权重在 2000 次更新内渐进到 `online_q_weight`。

续接（每次运行都把完整回放写入自己的日志目录）：

```bash
export RLINF_STAGE2_RESUME_DIR=$PWD/openpi/RLinf/logs/<run>/piper_wire_rlt_stage2/checkpoints/global_step_<N>
export RLINF_REPLAY_LOAD_PATH=$PWD/openpi/RLinf/logs/<run>/replay_buffer/rank_0
```

监控与停止线：`q_pi − q_data > 0.05`、`q_pi > 1`、`action_ref_abs_mean > 0.010` 任一出现就停止并降低 Q 权重。
只用已有回放离线重训：`openpi/RLinf/examples/embodiment/run_piper_rlt_stage2_replay_only.sh`。

## 结论与已知问题

- 2026-10-23 的完整实验中，纯 π0.5 自动成功率约 52%（15/29），任何 Actor 配置都没有超过它；
  Q 权重 0.1 时 Critic 在两分钟内高估失控（q_pi > 1），Q 0.01 时训练稳定但 Actor 只学到与状态无关的固定偏移。
- 根本原因：① 最后 1–2 mm 的横向偏差在 RL Token 与右腕相机中不可靠（沿视线方向完全不可见）；
  ② 接触后横向动作几乎不改变相对位置；③ 稀疏奖励下 Critic 的动作梯度方向与随机无异。
- 2026-10-30 离线验证了 Evo-RLT 式的“BC 只锚定参考 + 离线分批训练”，没有明显进步，已撤回。
- 详细原理、公式与数据：`openpi/RLinf/COBOT_MAGIC_RLT_DEPLOYMENT.md`、
  `openpi/RLinf/docs/source-zh/rst_source/examples/embodied/piper_rlt.rst`。
