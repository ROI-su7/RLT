# RLT on Cobot Magic: π0.5 + RL Token + chunk actor-critic

真机强化学习微调 VLA 的完整工程：在冻结的 π0.5（openpi，原生 JAX）之上，用 **RL Token（Stage 1）**
压缩 VLA 表征，再在真机上训练 **chunk 级 Actor-Critic（Stage 2）** 对 π0.5 的动作块做精细修正。
目标平台是 AgileX **Cobot Magic（双 Piper）**；SO101 机械臂的适配见 [docs/SO101.md](docs/SO101.md)（尚未完成）。

> 本仓库只包含代码与配置。**数据集、微调权重和 Python 环境都不上传**；环境用 `env/install_all.sh` 一键安装，
> 公开的基座权重（π0.5 base、PaliGemma tokenizer）由脚本下载，任务数据需要按 [docs/PIPELINE.md](docs/PIPELINE.md) 自行采集。

## 目录

| 路径 | 内容 |
|---|---|
| `openpi/` | openpi fork：π0.5 训练/推理，增加了 Piper 数据配置、前缀特征导出（`Pi0._prefix_forward`）、双输出推理服务 |
| `openpi/RLinf/` | RLinf fork：RL Token（Stage 1）、RLT Actor-Critic（Stage 2）、Piper 真机环境、四 CAN 接管、接头位姿检测 |
| `openpi/third_party/lerobot/` | openpi 使用的 lerobot 0.1.0 |
| `cobot_magic/` | Cobot Magic 机器端代码：数据采集、CAN 工具、相机工作空间、ACT 等 |
| `cobot_magic/Piper_ros_private-ros-noetic/` | 子模块：Piper ROS 驱动（四 CAN 复用器、主从接管），本地改动在 `patches/` |
| `cobot_magic/collect_data/piper_sdk_demo/` | 子模块：AgileX piper_sdk 示例，本地改动在 `patches/` |
| `cobot_magic/aloha_des/src/mobile_aloha_sim/` | 子模块：AgileX mobile_aloha_sim 描述文件 |
| `env/` | 一键环境安装（`install_all.sh`）、固定版本依赖（`requirements-rlt.txt`）、自检（`verify.sh`） |
| `docs/` | 完整流程、原理与问题记录 |

## 快速开始（Cobot Magic）

参考环境：Ubuntu 20.04、ROS Noetic、NVIDIA 驱动 ≥ 535（CUDA 12）、单卡 RTX 4090（24 GB）。

```bash
git clone --recursive https://github.com/ROI-su7/RLT.git ~/RLT
cd ~/RLT
bash env/install_all.sh          # apt/ROS（需要 sudo）+ 两个 Python 环境 + 基座权重
bash env/verify.sh               # 环境、ROS 工作空间与单元测试自检
```

只在训练服务器上（不连机器人）：`bash env/install_all.sh --no-ros`，自检用 `bash env/verify.sh --no-ros`。

两个 Python 环境：

| 环境 | 位置 | 用途 |
|---|---|---|
| openpi | `openpi/.venv`（`uv sync`，`openpi/uv.lock` 固定） | π0.5 微调、归一化统计、数据转换 |
| RLT | `openpi/RLinf/.venv-rlt`（`env/requirements-rlt.txt` 固定） | 原生 JAX π0.5 推理服务、Stage 1、Stage 2 |

完整流程（数据采集 → π0.5 微调 → 推理服务 → Stage 1 → 接头检测器 → Stage 2 → 部署）见
**[docs/PIPELINE.md](docs/PIPELINE.md)**。原理与公式见 `openpi/RLinf/COBOT_MAGIC_RLT_DEPLOYMENT.md` 和
`openpi/RLinf/docs/source-zh/rst_source/examples/embodied/piper_rlt.rst`。

## 路径约定

脚本通过自身位置定位仓库，也可以用环境变量覆盖：

| 变量 | 默认 | 含义 |
|---|---|---|
| `RLT_ROOT` | 仓库根目录 | 本仓库位置 |
| `~/rlt_data/` | — | 数据根目录：`lerobot_data/`、`wire_head_pose/` 等 |
| `RLT_HEAD_POSE_WEIGHTS` | `~/rlt_data/wire_head_pose/head_pose_net_v2.pt` | 接头位姿检测器权重 |
| `RLINF_PI05_JAX_DIR` | `openpi/checkpoints/pi05_connect_strings/.../30000` | 微调后的 π0.5 checkpoint |
| `RLINF_STAGE1_RLT_TOKEN` | — | Stage 1 RL Token（`rlt_token.pt`） |

## 当前结论

RLT 在本任务（DC 插头对插）上没有超过纯 π0.5：最后 1–2 mm 的横向偏差在 Actor/Critic 的状态里不可靠、接触后横向动作几乎无效、
稀疏奖励下 Critic 学不到动作梯度。详细数据与分析见 [docs/PIPELINE.md#结论与已知问题](docs/PIPELINE.md#结论与已知问题)。

## 许可

openpi、RLinf、lerobot 均为 Apache-2.0；`cobot_magic/camera_ws/src` 中的 realsense-ros（Apache-2.0）与 ros_astra_camera
保留各自许可证；子模块遵循其上游许可证。见 [NOTICE](NOTICE)。
