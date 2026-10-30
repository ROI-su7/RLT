# Cobot Magic / Piper：独立 RECAP 适配与真机流程

本文件说明 RLinf 官方 RECAP 离线流程在 Piper 上的完整使用方法：

```text
JAX 基础策略 / 上一轮 RECAP 策略 + 原有 T 人工接管
       ↓ 采集图像、状态、实际仲裁动作，C/X 标记整条回合
独立 raw episodes → 新的 LeRobot 快照（rollout）
       ↓ + 原 350 条示教（sft）
计算 returns → 训练 Value Model → 计算 advantage → CFG 微调 π₀.₅
       ↓
独立 websocket 服务 → 原 OpenPI 真机客户端 → 原 ROS mux → 下一轮采集
```

本方案不训练 RLT 的 RL Token、MLP Actor 或 twin Q。RECAP 的 Value Model 只用于
给离线数据计算 advantage；最终更新的是 π₀.₅ 动作专家上的 LoRA 适配器，基础权重冻结。

参考：仓库内 [官方 RECAP 说明](docs/source-en/rst_source/examples/embodied/recap.rst)、
[中文说明](docs/source-zh/rst_source/examples/embodied/recap.rst)，以及本机
[/home/agilex/FluxVla/openpi/README_PIPER_FOUR_CAN.md](/home/agilex/FluxVla/openpi/README_PIPER_FOUR_CAN.md)
第 4、6、8 节。人工接管的 CAN 角色切换、延时、HOLD 和 HOME 都由该文档中的
原 ROS launch/mux 完成，新代码只发送相同请求。

## 0. 总流程一览

| 步骤 | 在哪做 | 本文章节 |
| --- | --- | --- |
| 开机准备：roscore、CAN、相机 | 本机 | 第 3 节 |
| 用原生 JAX π₀.₅ 采集成功/失败/接管回合 | 本机 | 第 4 节 |
| 导出 LeRobot 快照 | 本机 | 第 5 节 |
| 准备 350 条示教的隔离副本 | 本机或服务器 | 第 6 节 |
| JAX→PyTorch 转换、SigLIP2/Gemma3 权重 | 服务器（推荐）或本机 | 第 7 节 |
| returns → value → advantages → cfg | 服务器（推荐，第 9 节）或本机 4090（第 8 节） | 第 8、9 节 |
| 对照基线：SFT + 人工接管片段 | 本机 4090 或服务器 | 第 8.5 节 |
| 部署 CFG 策略、评估并采集下一轮 | 本机 | 第 11 节 |

第一轮不需要任何转换，直接用已训练好的 JAX checkpoint 采集。

## 1. 数据组成：RECAP 到底需要什么

### 1.1 三类回合及其作用

| 回合类型 | 怎么得到 | 标记 | 在 RECAP 中的作用 |
| --- | --- | --- | --- |
| 自主成功 | `I → C` | `is_success=true` | 告诉 Value 哪些状态离完成近 |
| 自主失败 | `I → X` | `is_success=false` | **RECAP 最依赖的数据**：Value 只有见过失败才能区分好坏动作 |
| 人工接管 | `I → T → 人工 → I → … → C/X` | 接管帧 `intervention=true` | 在策略容易出错的阶段提供纠正动作 |

接管回合也必须按最终结果打 C 或 X，人工救回来的回合标 C。接管帧中保存的是主臂经
mux 仲裁后的实际命令，即人工动作。

### 1.1.1 按键与 RECAP 打分机制的对应

RECAP（π\*0.6 式 5 与 IV-B/V-C/V-D 节）只有三个打分量：每回合一个**成功标签**、由它推出的
**稀疏奖励**，以及人工纠正的**改进指示 Iₜ**。每个按键对应其中一项：

| 按键 / 事件 | 记录的字段 | RECAP 中的量 | 训练时的取值（默认配置） |
| --- | --- | --- | --- |
| **I**（推理确认后自动开回合）之后的每个记录帧 | 一帧 | 普通步奖励 | rₜ = −1 |
| **C**（三根都插好、灯亮） | 终止帧 `is_success=1` | 成功标签 = 1 | r_T = 0 |
| **X** | 终止帧 `is_success=0`，`failed_insertion` | 成功标签 = 0 | r_T = −C_fail，`failure_reward: -3500` |
| `--time-limit-s` 到时 | 同 X | "在时限内完成"才算成功（论文 200/500/600 s） | 同 X |
| **T** … **I** | 人工帧 `intervention=1` | 人工纠正：**强制 Iₜ = True** | positive，与回合成败无关（`intervention_success_only: false`） |
| 其余帧 | `intervention=0` | Iₜ = 1[A > ε]，ε 取 30% 分位 | 由 Value 计算的 advantage 决定 |
| 示教（`type: sft`） | 无按键 | 全部成功 | r 同上；RLinf 约定全部 positive |
| N / U（可选，**不需要按**） | `insertions_done` | **不进入奖励**，只用于按根统计 | — |
| 空格 / H / Q | 不写帧 / 回合作废 | 不打分 | 不进入训练 |

γ = 1，回报 Rₜ = Σ r，即"−剩余步数"（失败再加 −C_fail）。C_fail 按论文要求"使失败状态的价值足够低"取
3500，大于最长成功回合（示教最长 3336 步）。上游默认值 −300 只适合约 300 步的任务。

下面两项**不属于 RECAP 原文**，单独列出，可分别关闭：

| 扩展 | 配置键（默认值） | 来源 | 关闭后 |
| --- | --- | --- | --- |
| 接管惩罚 | `data.takeover_penalty: 300.0` | Evo-RL：接管前一步 rₜ = −1 − C_int | `=0` 即严格的 RECAP 奖励 |
| 按剩余根数加重失败 | `data.stage_failure_cost: 0.0`（默认已关闭） | 本适配的可选项 | — |

`intervention_success_only=true` 是 Evo-RL 的变体，只对成功回合的人工帧强制为正。C_int=300 是经验起点
（Evo-RL 未公开取值），大约相当于"一次接管 = 多走 300 步"。

采集建议（经验值，不是官方要求）：

- 第一轮先采 30–50 条，失败回合不少于 1/3。不要只保留成功回合，也不要中途删掉失败回合。
- 不要刻意制造失败：按平常的初始摆放让策略自己执行，自然失败就按 X。
- 策略在精细阶段明显要失败时再按 T 接管，接管前的帧仍然是策略自己的动作。
- 每次只改一个条件（摆放、光照、线材位置），以便判断哪个阶段在改进。

### 1.2 要与原来的 350 条示教融合吗

**要，推荐融合。** 这也是官方 RECAP 的做法：示教作为 `type: sft`（全部视为成功），
真机采集作为 `type: rollout`（含成功、失败和接管）。各阶段的用法：

| 阶段 | 是否加入 350 条 | 原因 |
| --- | --- | --- |
| returns | 两份分别计算 | sft 每条都按成功计算回报；rollout 读取 C/X |
| Value 训练 | **加入** | 350 条提供大量"离完成多远"的信号，少量 rollout 提供失败信号 |
| advantages | **两份都算** | 全局统一回报范围和 30% 阈值；CFG 训练要求每份数据都有 advantage。RLinf 约定 `type: sft` 的**所有帧**强制为 positive，阈值只作用于 rollout |
| CFG 训练 | **加入** | 保持原有示教分布，避免只在几十条 rollout 上微调导致退化；`balance_dataset_weights: true` 按帧数比例采样（见第 8 节第四步） |

已核对的兼容性：

- 原示教 `all_wire_data`：350 条、562k 帧，三路相机名与 14 维 state/action 与导出快照一致，
  任务提示词与 `pi05_connect_strings` 相同。
- 示教数据为 30 fps，记录器默认 15 fps。原客户端以 `--max-hz 15` 执行，每个动作步
  1/15 s，因此**按"步"计算**，自主回合与示教一致。回报以步计，CFG 动作块按各数据集
  自身 fps 取连续 50 帧。**不要**把示教下采样到 15 fps，否则动作块每步位移翻倍、执行速度变快。
- 人工接管段是按真实时间以 15 fps 记录的，每步位移约为示教的 2 倍。为免把接管段的
  速度学进去，接管时主臂动作宜慢而平稳，评估时注意接近段速度是否变化。
- 示教回合长度为 1166–3336 步，平均 1606 步。上游默认 `failure_reward=-300` 是按较短回合
  设定的，用在这里会使"失败前的状态"看起来比"刚开始的成功状态"更好。Piper 配置因此默认
  `failure_reward: -3500`（大于最长成功回合的步数），保证任何失败状态的回报都低于任何
  成功状态。同一轮 returns、value、advantages 必须使用同一套 returns 结果。
- `compute_returns` 会**原地改写**数据集的 `meta/stats.json` 和 `meta/info.json`。因此不能
  直接指向 pi05 原训练数据目录，要先按第 6 节建立副本。
- 示教为 mp4 视频格式，导出快照为 PNG。视频解码由首次 2-step GPU 冒烟验证。

## 2. 隔离与当前验证边界

| 项目 | RECAP 使用位置 |
| --- | --- |
| 原始回合（可持续追加） | `recap_data/raw/episode_<uuid>/` |
| 每轮不可覆盖的 LeRobot 快照 | `recap_data/lerobot_v1/`、`lerobot_v2/` 等 |
| 350 条示教的隔离副本 | `recap_data/sft_all_wire_data/` |
| 转换后的基础策略、Value 预训练权重 | `recap_artifacts/` |
| HF 数据缓存 | `recap_artifacts/datasets_cache/` |
| 训练日志、checkpoint | `recap_runs/<时间>/value/`、`cfg/` |
| websocket | `127.0.0.1:6016` |
| 离线训练 Ray head | `127.0.0.1:6385`，dashboard 可用 8266 |

RLT 的入口、主配置、训练/执行代码、RL token、replay、checkpoint、6006 服务均不改。
脚本不执行 `ray stop`，不自动启动机械臂或修改 CAN。离线训练只连接显式指定的
RECAP Ray head，不自动接入 6379。**RLT 服务或训练运行时，不要在同一张 4090 上启动
RECAP 模型服务、转换或训练；端口分开并不隔离显存。** 脚本的 GPU 空闲检查会拒绝启动。

当前解释器复用 `.venv-rlt/bin/python` 中已经安装的包，只读使用，不执行 pip 升级。
可通过 `RECAP_PYTHON` 指定另一个兼容环境。没有更改原 OpenPI 或 ROS 源码。

已在 CPU 上验证：原始回合写入、成功/失败回报、LeRobot 导出、三相机 Value 输入、
advantage 标签保留、LoRA 参数选择、四步配置解析（含第 8 节的示教+rollout 融合覆盖）。
尚未验证：实际预训练权重加载、显存峰值、示教视频解码，以及 CFG 的真机效果。CPU
测试通过不代表真机成功率。推荐先用 `runner.max_steps=2` 做完整 GPU 冒烟。

## 3. 每个新终端与每次开机的准备

### 3.1 每个新终端先加载独立环境

```bash
cd /home/agilex/FluxVla/openpi/RLinf
source examples/recap_piper/env.sh
bash examples/recap_piper/run.sh check
```

默认使用本机已经存在的原生基础策略：

```text
config: pi05_connect_strings
JAX: /home/agilex/FluxVla/openpi/checkpoints/pi05_connect_strings/pi05_connect_strings_20260923/30000
stats: <上述路径>/assets/all_wire_data/norm_stats.json
```

`RECAP_TASK` 通过 AST 读取原 OpenPI config 的 `default_prompt`，不会导入并初始化
JAX。当前应输出：

```text
connect the light strings and the battery box to the splitter cable so the lights turn on.
```

不要将另一篇 README 的 `insert the wire` 提示词直接套到 connect-strings checkpoint。
若换基础策略，在 `source` **之前**设置匹配的配置名、JAX 路径、stats，并让 task 重新解析：

```bash
export RECAP_SOURCE_CONFIG=pi05_wire_all_data_full
export RECAP_PI05_JAX_DIR=/home/agilex/FluxVla/openpi/checkpoints/pi05_wire_all_data_full/pi05_wire_all_data_full_20260818/30000
# 从该 checkpoint 的 assets 中选择与它训练时一致的 norm_stats.json，不能沿用另一模型的统计。
export RECAP_NORM_STATS=/绝对路径/该模型的/assets/数据集名/norm_stats.json
unset RECAP_TASK
source examples/recap_piper/env.sh
```

同一轮采集、转换、CFG 训练和部署必须使用同一套 source/task/stats。

### 3.2 开机准备（四 CAN README 第 4 节）

先确认 GPU 上没有 RLT 服务，且没有旧的 Piper/客户端进程：

```bash
nvidia-smi
ps -ef | grep -E "piper|roslaunch|examples.piper_real|serve_piper_rlt" | grep -v grep
```

终端 1，ROS Master：

```bash
source /opt/ros/noetic/setup.bash
roscore
```

配置 CAN。只能在全部 Piper 节点都停止时执行；launch 运行期间严禁重新执行：

```bash
cd ~/cobot_magic/Piper_ros_private-ros-noetic
sudo bash can_config.sh
ip -br link show type can
# 必须看到 can_l_leader、can_l_follower、can_r_leader、can_r_follower 均为 UP
```

终端 2，相机：

```bash
cd ~/cobot_magic/camera_ws
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch astra_camera multi_camera.launch
```

## 4. 真机采集：复用现有主从接管

不运行 RLT trainer，不运行独立 `piper_mode_keyboard.py`，也不同时运行另一个 policy 客户端。

### 终端 A：原四臂 launch

```bash
cd /home/agilex/cobot_magic/Piper_ros_private-ros-noetic
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch piper start_inference_takeover.launch \
  master_left_can:=can_l_leader \
  master_right_can:=can_r_leader \
  puppet_left_can:=can_l_follower \
  puppet_right_can:=can_r_follower \
  publish_hz:=200
```

### 终端 B：初始基础策略服务

```bash
cd /home/agilex/FluxVla/openpi/RLinf
bash examples/recap_piper/run_native_server.sh
```

它只在 RECAP 的 6016 端口启动原生 JAX 基础策略，不加载 RL token。
后续已经训练好 RECAP 时，用第 11 节的 CFG 服务替代此服务，不能两者同占 6016。

### 终端 C：唯一键盘和记录器

```bash
cd /home/agilex/FluxVla/openpi/RLinf
bash examples/recap_piper/run.sh collect --profile takeover
```

### 终端 D：原 OpenPI 真机客户端

```bash
cd /home/agilex/FluxVla/openpi/RLinf
bash examples/recap_piper/run_client.sh takeover
```

这个脚本复用原 OpenPI 客户端，使用 6016 端口、关闭视频。
`takeover` 和 `followers` 都采用桌面《Piper双从臂推理问题与解决方案》的 Pi05
抓取时序参数：`--action-horizon 10 --rtc-prefix-len 4 --rtc-boundary-blend-steps 5
--action-smoothing-alpha 0.5`。`takeover` 按现场要求保留原来的手臂速度上限
`--max-arm-velocity-rad-s 0.4`；`followers` 保持文档参数 `0.6`。
旧 takeover 的 20/15 设置会让约 85 ms 的推理结果在约 16 步（1.07 秒）后才执行，
可能使夹爪在手臂到达前闭合。新设置缩短旧观测预测的执行时间，并减小手臂滞后。
运行 `bash examples/recap_piper/run_client.sh takeover --dry-run` 可查看命令而不启动机器人
（未设置 `RECAP_TASK` 时仍会读取任务配置）。启动日志也记录完整命令。
修改后需在 HOLD 状态停止旧客户端并重新启动终端 D 才生效；抓取效果仍需实机复测。
客户端使用本机 `pi0` conda Python，默认会使能并开始推理；不要另外发布 `/enable_flag=true`。
原客户端仍负责 200 Hz 发布、RTC 和 HOLD ACK，mux 负责主从控制权。
RECAP 入口 `rlinf.recap_piper.client` 只对交接位置偏差做 5 步平滑衰减，
保留新预测内部的运动增量，避免旧轨迹慢速时反复压低互插阶段的小幅推进。
每臂交接修正量不超过 `0.002 rad` 时直接采用新预测，由客户端原有 EMA 平滑；
更大偏差的过渡限制在上一目标与新预测窗口范围内，夹爪指令不参与融合。
启动后的 ROS 日志包含 `RTC offset-decay handover` 及最大修正量，便于核查。
抓取阶段执行长度保持 10 步。两个夹爪都经历过张开（>35 mm），随后连续 3 次观测
均闭合到 ≤30 mm，且新预测直到将执行窗口结束都保持 ≤30 mm 时，执行长度扩展为
30 步，以执行对齐计划的后续推进段；这只是持续夹持的调度判据，不保证已抓住物体。
预取仍为 4 步，不改变速度上限；新预测包含松爪、观测条件不满足或预测剩余长度不足时，
下次动作块交接恢复 10 步。只在交接处切换长度，保持预取触发点正确。
ROS 日志中的 `RTC execution horizon 10 -> 30` 表示进入持续夹持调度。
2026-10-29 最后一条 `episode_a19e24e0eeba48b9b07f21025dbac441` 的第 650/850 帧
离线重复推理显示，右臂预测第 5–15 步净位移约 0.002–0.007 rad，而第 30–45 步
约 0.017–0.033 rad；固定短动作块会反复重规划，截断后续推进。该对比是离线预测，
不能等同于实机提速倍数。
推理等待导致 HOLD 时不沿用旧速度，键盘 HOLD/reset 仍走原客户端的清队列流程。
HOLD/reset 同时清除持续夹持计数和张开历史，恢复后先采用短动作块。
该修正保留当前速度上限和抓取时序参数；需要重启终端 D，实机效果待复测。

### 按键（全部在终端 C）

| 按键 | 行为 |
| --- | --- |
| I | **没有打开的回合时**：请求 INFERENCE，mux 确认后**自动开始新回合**，从此每步 rₜ = −1。**回合进行中**：恢复推理，仍是同一回合。接管（T）之后直接按 I 即可，记录器会自动先请求 HOLD 再请求 INFERENCE（原 mux 只允许从 HOLD 恢复推理），**不需要再先按空格**。上一回合 C 成功后，必须先 H 回原点才接受 I |
| B | 可选：手动开始回合（例如回合从 TAKEOVER 开始）。一般不需要 |
| T | 原 TAKEOVER 请求；看到原 launch 的 ACTIVE 确认后再动主臂。**不结束回合**，按 I 回到推理仍是同一回合 |
| C | 成功：三根都插好、灯亮时按。保存 `is_success=true`，请求 HOLD |
| N / U | **可选，正常采集不需要按**。每插好一根按 N（U 撤销），只用于 `run.sh score` 的按根统计。加 `--require-marks` 启动时，C 必须先有 3 次 N |
| X | 失败：保存 `is_success=false`，并记下失败发生在第几根，请求 HOLD |
| 空格 | 原软件 HOLD，保持使能；**回合不结束**，暂停帧不写；I 恢复 |
| H | 已确认 HOLD 后请求 HOME：从臂夹爪先原地张开 1.5 s 放下物体，张着回原点，到位后合上（超时也会合上）。**回合进行中按 H 会作废整条回合**，只在 C/X 之后用 |
| Q / Ctrl-C | 请求 HOLD 并退出记录器；没有 C/X 的当前回合标记 aborted，不入训练 |

正常序列（三次插接）：

- 自主回合：`I →（策略插三根）→ 灯亮按 C → 等待 HOLD → H → 重置三根线 → I`
- 失败后两种做法，按现场决定（失败回合在按 X 时已经保存，后面按什么都不会丢）：
  - 现场还能继续：`X → I`，新回合从当前现场开始，插完剩下的根数后灯亮即可按 C。
  - 现场无法继续（线扯坏、插头掉落、姿态卡住等）：`X → 等待 HOLD → H → 重置三根线 → I`，
    新回合从头开始。

**推理中途机械臂自己停下：** 原 mux 在策略命令超过 `policy_input_timeout_s`（1 s）没有更新时会自动
HOLD。记录器会打印"`The mux entered HOLD by itself`"，回合不结束，直接按 I 恢复。客户端和服务端的
输出会自动保存到 `recap_runs/robot_logs/client_*.log`、`server_*.log`，出现停顿时用来排查原因。

**某一根策略插不上、人工接管插好之后怎么做：** 直接按 I 让策略继续插下一根。
**不要按空格 + H**：空格只是暂停，H 会作废整条回合。三根都适用同样的做法，一个回合里可以接管多次：

```text
I                      策略开始，回合开始（每步 −1）
  第 1 根策略插不上  →  T（等 ACTIVE）→ 主臂插好 → I（策略继续，同一回合）
  第 2 根策略插好    →  不用按键
  第 3 根又插不上    →  T → 主臂插好 →（灯亮可直接 C）
  灯亮               →  C   成功回合：is_success=1，所有人工帧强制 positive
  等待 HOLD → H → 重置三根线 → I 开始下一回合

  救不回来           →  X   失败回合：is_success=0
  现场能继续 → I；不能继续 → 等待 HOLD → H → 重置三根线 → I
```

只有判定"这一回合救不回来"（线扯坏、超时、不打算接管）时才按 X。接管把这一根救回来不算失败。

H 的释放动作由原 ROS mux（`piper_four_state_mux.py`）执行，参数在 `start_inference_takeover.launch` 中
默认开启：`home_release_gripper_m:=0.06`（张开宽度，米）、`home_release_s:=1.5`（原地张开时长）。
设 `home_release_gripper_m:=0` 恢复原 HOME。其他 launch（含 RLT 用的 `start_inference_followers.launch`）
不受影响。修改前的文件备份在 `backups/ros_home_release_20261028/`。

### 4.1 三次插接的打分规则（RECAP 专用）

打分实现在 `rlinf/recap_piper/scoring.py`，与 RLT 的键盘奖励
（`rlinf/envs/realworld/piper/tasks/__init__.py`，g/i/f/c/x）完全独立，RLT 代码未改。

| 规则 | 说明 |
| --- | --- |
| 回合 = 完整的三次插接 | 与 350 条示教一致；不要按单根分回合，否则同一状态的回报目标会与示教冲突 |
| 成功 | 三根都插好且灯亮时按 C |
| 失败 | 判定救不回来时立即按 X（越接近出错时刻越好）；记录 `failed_insertion` |
| 超时 | RECAP 的成功是"在时限内完成"。`--time-limit-s` 到时自动按 X 记录并 HOLD |
| 接管 | 是纠正而不是失败；救回来、最终三根都插好就按 C |
| 从中途开始的回合 | 有效数据（Value 估计"从这里到三根全部插好还要多远"），但不计入自主成功率 |

时间限制只计记录的帧（HOLD 暂停不计），默认关闭。示教平均 1606 步，在 15 Hz 执行约 107 s，
最长 3336 步约 222 s。建议先取 240 s，同一轮采集保持不变：

```bash
bash examples/recap_piper/run.sh collect --profile takeover --time-limit-s 240
```

每个回合的 `episode.json` 中保存 `score`。只按 C/X 时只有成功/失败；按了 N 才有每根的帧号和
"失败在第几根"。导出时生成逐帧的 `insertions_done` 和 `insertion_event` 列，**不改变 RECAP 奖励**。

可选扩展（不属于 RECAP，默认关闭）：按剩余根数加重失败惩罚。

```text
失败终止奖励 = failure_reward − stage_failure_cost × (3 − 已完成根数)
data.stage_failure_cost=535     # ≈ 示教平均 1606 步 / 3；默认 0
```

统计采集结果（只读 raw 回合，不导出）：

```bash
bash examples/recap_piper/run.sh score
```

输出：回合数、成功数、从头开始的回合数、**无接管的自主成功数**（只按 C/X 即可得到）。
每一根的尝试次数、完成率、接管率和平均用时只统计按了 N 的回合（`per_insertion_episodes`）；
需要这部分时用 `--require-marks` 启动记录器。
旧版记录器采的、没有 `score` 的回合不计入统计，导出时 `insertions_done` 为 0。

`T` 两侧角色切换、实际等待时间、HOME 的路径均由当前 ROS 代码决定，记录器不会复写。
H 不会猜测成功/失败；按 H 前忘了 C/X，该条回合保留为 aborted，不会自动当成功。
记录中缺图像、反馈/命令过期或磁盘来不及写入，会 HOLD 并排除这条不完整回合。

每帧保存三路 RGB PNG、从臂状态、`/master/joint_left/right` 的最终仲裁目标、policy
目标、leader 位姿及人工标记。`intervention=true` 只在 `TAKEOVER/ACTIVE` 确认后记录。
policy 在人工阶段可能暂停，其新鲜性由 `policy_valid` 表示；没有真实 leader 流时
`leader_valid=false`。动作是 rad/米的命令目标，不是电机已经达到的位置。

仅从臂采集时改用原 `start_inference_followers.launch`，记录器和客户端都选择
`followers` profile，T 被禁用。不要将 RLT 的 `rlt_stage2:=true` 键盘所有权配置混入
这个原 OpenPI 客户端流程。

停机：先在终端 C 按 Q，再停终端 D、B、A，最后停相机与 roscore（四 CAN README 第 9 节）。

## 5. 导出新的 LeRobot 数据快照

停止记录后执行；导出期间不要再往同一批待导出的 raw 回合写入。

```bash
cd /home/agilex/FluxVla/openpi/RLinf
source examples/recap_piper/env.sh
bash examples/recap_piper/run.sh export
bash examples/recap_piper/run.sh check --stage dataset
```

输出 `recap_data/lerobot_v1`，显示 episodes、successes、failures、frames。无 C/X 的
回合不会计入。导出保存本次的原始回合映射 `meta/piper_recap_manifest.json`。

以后继续采集时重复运行 collect 即可，raw 目录按 UUID 追加，不会覆盖旧回合。
下一轮导出选择新路径：

```bash
export RECAP_DATASET=$PWD/recap_data/lerobot_v2
bash examples/recap_piper/run.sh export
```

已有快照一律拒绝覆盖；不要删除正在训练的 dataset。四步使用同一个 `RECAP_DATASET`。
RLT 的 z_rl/replay 不能直接代替这些 RGB 回合。评估时按整条 episode 留出数据。

## 6. 准备 350 条示教的隔离副本

`compute_returns` 会改写数据集的 `meta/stats.json` 与 `meta/info.json`，因此建立副本：
`data/`、`videos/` 用硬链接（不占额外空间），`meta/` 完整复制（避免改写原文件）。

```bash
cd /home/agilex/FluxVla/openpi/RLinf
SRC=/home/agilex/FluxVla/openpi/lerobot_data/all_wire_data
export RECAP_SFT_DATASET=$PWD/recap_data/sft_all_wire_data
test ! -e "$RECAP_SFT_DATASET"            # 已存在就不要覆盖
mkdir -p "$RECAP_SFT_DATASET"
cp -al "$SRC/data" "$SRC/videos" "$RECAP_SFT_DATASET/"
cp -a  "$SRC/meta" "$RECAP_SFT_DATASET/"
```

硬链接要求在同一文件系统；服务器上用服务器的 `~/FluxVla/openpi/lerobot_data/all_wire_data`
重复同样步骤。每轮可以复用这份副本，但换 `failure_reward` 后要重新计算 returns。

## 7. 准备模型资产

### 7.1 基础策略转换

```bash
cd /home/agilex/FluxVla/openpi/RLinf
source examples/recap_piper/env.sh
bash examples/recap_piper/run.sh convert
```

转换以 CPU 运行，输出 `recap_artifacts/pi05_base`，需要较多 CPU RAM 和磁盘。本机只有
31 GB 内存，推荐在服务器的 CPU 节点上转换后再传回本机。源模型是完整参数时保留完整
参数；源模型含原生 JAX LoRA 时保留其独立 A/B 分支。不写入源 checkpoint，已存在的
转换输出拒绝覆盖。

CFG 默认对动作专家做 LoRA 微调：新加 LoRA 的 A 随机初始化、B 初始化为零，因此初始输出
等同基础模型。保留源模型 50-step、32-D padded attention 布局；真机仍输出绝对 14-D 动作。

### 7.2 Value Model 的预训练权重

需要本地完整的 SigLIP2-so400m 与 Gemma3-270M（含 tokenizer），不能用 Stage 1 RL token
替代。两者都是 Google gated 模型：先在 Hugging Face 网页接受许可，再在本机登录并下载
（服务器无外网，下载后用 rsync 传过去）：

```bash
source examples/recap_piper/env.sh
"$(dirname "$RECAP_PYTHON")/huggingface-cli" login
"$(dirname "$RECAP_PYTHON")/huggingface-cli" download \
  google/siglip2-so400m-patch14-224 --local-dir "$RECAP_SIGLIP_DIR"
"$(dirname "$RECAP_PYTHON")/huggingface-cli" download \
  google/gemma-3-270m --local-dir "$RECAP_GEMMA_DIR"
```

不要将 token 写进 README 或 shell 脚本。该步骤不升级当前 `.venv-rlt`。

## 8. 四步离线训练（示教 + rollout 融合）

先退出模型服务与真机客户端，使 4090 空闲。离线训练不连接 ROS/CAN，也不需要
机械臂通电。脚本会拒绝在已有大显存模型进程运行时再启动模型任务。

新终端中设置本轮固定目录与数据，并启动**独立的** RECAP Ray：

```bash
cd /home/agilex/FluxVla/openpi/RLinf
source examples/recap_piper/env.sh
export RECAP_RUN_DIR=$PWD/recap_runs/round_001
export RECAP_DATASET=$PWD/recap_data/lerobot_v1
export RECAP_SFT_DATASET=$PWD/recap_data/sft_all_wire_data
export CUDA_VISIBLE_DEVICES=0
export RLINF_NODE_RANK=0
export RLINF_COMM_NET_DEVICES=eno1

# 融合数据的 Hydra 覆盖；同一轮四步必须一致。
RET_DATA="data.train_data_paths=[{dataset_path:$RECAP_SFT_DATASET,type:sft},{dataset_path:$RECAP_DATASET,type:rollout}]"
VAL_DATA="data.train_data_paths=[{dataset_path:$RECAP_SFT_DATASET,type:sft,weight:1.0,robot_type:piper,model_type:pi05},{dataset_path:$RECAP_DATASET,type:rollout,weight:1.0,robot_type:piper,model_type:pi05}]"
ADV_DATA="data.train_data_paths=[{dataset_path:$RECAP_SFT_DATASET,type:sft,weight:1.0,robot_type:piper},{dataset_path:$RECAP_DATASET,type:rollout,weight:1.0,robot_type:piper}]"
CFG_DATA="data.train_data_paths=[{dataset_path:$RECAP_SFT_DATASET,type:sft,weight:1.0},{dataset_path:$RECAP_DATASET,type:rollout,weight:1.0}]"

# 不执行 ray stop，也不接入 RLT 的 6379。
env -u RAY_ADDRESS "$(dirname "$RECAP_PYTHON")/ray" start --head \
  --port=6385 --dashboard-port=8266 --node-ip-address=127.0.0.1 \
  --num-cpus=8 --num-gpus=1 \
  --temp-dir=/tmp/rlinf_piper_recap_ray --disable-usage-stats
```

只用 rollout、不融合时，去掉各步的 `*_DATA` 参数即可（使用 yaml 默认值）。打分参数（C_fail=3500、
接管惩罚、强制正样本）已是 Piper 配置的默认值，见第 1.1.1 节。
每一步都可以先加 `--dry-run` 只解析配置，不建立 worker、不占显存。

第一步，计算回报：

```bash
bash examples/recap_piper/run.sh returns "$RET_DATA"
```

在两个数据集的 `meta/` 下分别写出 `returns_piper_recap_v1.parquet`，并更新各自的 stats/info。
回报按 RECAP 构造：普通步 -1，成功终止步 0，失败终止步为 `failure_reward`（-3500），gamma=1；
扩展项：每次接管发生前的最后一个策略帧再扣 `takeover_penalty`（默认 300，第 1.1.1 节）。
sft 数据集每条都按成功计算，没有 `intervention` 列，不受接管惩罚影响。这个负回报不是 RLT 的 reward，也不回写 raw 数据。

第二步，训练 Value Model：

```bash
bash examples/recap_piper/run.sh check --stage value
# 首次先做 2-step GPU 冒烟（同时验证示教视频解码）；不要把它当可用的 value model。
bash examples/recap_piper/run.sh value "$VAL_DATA" runner.max_steps=2 runner.save_interval=1
# 正式训练（换新的运行目录，避免覆盖冒烟 checkpoint）：
export RECAP_RUN_DIR=$PWD/recap_runs/round_001_full
bash examples/recap_piper/run.sh value "$VAL_DATA"
```

4090 默认 micro batch=1、global batch=8，冻结 SigLIP 视觉编码器，训练 Gemma3、
新建的图像投影和轻量 Critic Expert。输出 201-bin 的归一化 V 值，回报范围取所有
数据集 stats 的全局最小/最大值。训练上限默认 2000 step、每 100 step 保存。正式步数应
按留出回合表现决定，固定步数不保证收敛。

第三步，给两份数据计算 advantage：

```bash
# 替换为第二步实际保存的 checkpoint，不要指向 RLT 的 global_step。
export RECAP_VALUE_CHECKPOINT=$RECAP_RUN_DIR/value/piper_recap_value/checkpoints/global_step_2000/actor/model_state_dict
bash examples/recap_piper/run.sh advantages "$ADV_DATA"
```

使用 10-step lookahead，在每个数据集的 `meta/` 下写 `advantages_piper_recap_adv_v1.parquet`。
阈值在合并后的全部帧上统一计算，rollout 中最高 30% 标为 positive；sft 示教的所有帧按 RLinf
约定强制为 positive；所有人工接管帧（不论回合成败，RECAP 原文做法）也强制为 positive（日志打印强制了多少帧、
最终正样本比例，sidecar 中有 `forced_positive` 列）。rollout 的 positive 不等同于"成功回合的
所有帧"：失败回合中也可能有相对更好的动作。新采集改变数据和回报范围后，应重新计算
returns、训练 Value、计算 advantage，不能复制旧 sidecar。

第四步，CFG 微调 π₀.₅：

```bash
bash examples/recap_piper/run.sh check --stage cfg
bash examples/recap_piper/run.sh cfg "$CFG_DATA"
```

正 advantage 帧使用带 `Advantage: positive` 的条件提示；负帧训练无条件分支，10% 正帧也
随机使用无条件分支。`balance_dataset_weights: true`（RLinf 原语义）把每个数据集的 `weight` 再乘以
它的帧数，即在全部帧上均匀采样：round_001 中示教约 87.7%、rollout 约 12.3%（人工接管帧约 0.9%）。
要提高 rollout 比例，调大 `CFG_DATA` 中 rollout 条目的 `weight`（如 3.0 → rollout 约 30%）。使用原始物理动作、Piper/Aloha 适配、源统计归一化和 50-step flow-matching
训练，不在动作上叠加 RLT Actor。

默认只训练动作专家 LoRA，冻结其余基础参数，micro batch=1、global batch=8，保存完整 FSDP
权重和 `piper_recap_policy.json`，用于严格恢复以及校验基础模型、任务和归一化统计。训练
上限默认 2000 step，每 100 step 保存；增大步数时同时调整优化器 total_training_steps。

### 8.5 对照基线：SFT + 人工接管（Evo-RL 的 SFT+Int）

只做普通 LoRA 微调：数据是 350 条示教，加上**成功回合中的人工接管片段**（每段连续的人工帧
导出为一个独立回合，自主帧和失败回合全部丢弃）。不需要 Value、advantage 和 CFG，是最便宜
的对照。Evo-RL 论文在 PiperX 插装上的结果：SFT 22.7% → SFT+Int 78.7%，而他们实现的 RECAP 为
57.8%。该结论单次实验、样本少，需要在本机任务上自行验证。

```bash
# 与 RECAP 使用同一批 raw 回合；默认输出 recap_data/interventions_v1，至少 10 帧的片段才保留
bash examples/recap_piper/run.sh export-interventions
bash examples/recap_piper/run.sh check --stage sftint     # 同时检查示教副本和接管片段
bash examples/recap_piper/run.sh sftint runner.max_steps=2 runner.save_interval=1   # GPU 冒烟
export RECAP_RUN_DIR=$PWD/recap_runs/round_001_sftint
bash examples/recap_piper/run.sh sftint
```

配置 `piper_recap_sft_int.yaml` 与 `piper_recap_cfg.yaml` 使用相同的基础模型、LoRA 参数、
优化器和步数（单元测试保证两者一致），区别只在：`data.plain_sft: true`（所有帧走原任务提示词，
不加 `Advantage: positive`），以及训练数据为 `RECAP_SFT_DATASET` + `RECAP_INT_DATASET`。
`export-interventions` 可加 `--min-segment-frames N` 调整最短片段。

部署时 `serve` 会读取 checkpoint 中 `piper_recap_policy.json` 的 `policy_mode` 自动选择配置；
基线只走无条件分支，不要传 `--guidance-scale`：

```bash
bash examples/recap_piper/run.sh serve \
  --checkpoint /绝对路径/sftint/piper_recap_sft_int/checkpoints/global_step_2000
```

建议对比方式：同一批初始摆放，交替部署 RECAP 和 SFT+Int（每个至少 20 次），统计无接管成功率、
完成用时和接近段速度。两者训练步数相同，才是公平对比。

训练结束后不要执行 `ray stop`：它会停止本机**所有** Ray 进程，包括 RLT 的 6379 集群。
只有确认本机没有其他 Ray 集群时才可以停。

## 9. 在服务器上训练（推荐）

服务器 `ssh myserver`（<user>@<server>）是 Slurm 集群：登录节点只有 CPU，可以提交作业、
看日志、开 TensorBoard，但不能跑模型。可用分区：`3090`（finai01-16，每节点 8×RTX 3090 24 GB）、
`gpu`（hgx，8×A100-40G，经常要排队）；本账号**不能用 `cpu` 分区**。服务器无外网。代码镜像在
`~/FluxVla/openpi/RLinf`，`.venv-rlt` 可用；服务器上的 openpi `config.py` 是训练版本，
不要用本机文件覆盖。

### 9.1 一轮的总流程

```text
本机：collect → export（lerobot_vN）→ check --stage dataset
  ↓ rsync 代码 + 数据（9.3）→ touch .inputs_ready
服务器 prep 作业（3090 单卡，CPU 活）：示教副本 → JAX→PyTorch 转换 → returns
  ↓ afterok 依赖自动开始
服务器 train 作业（8 卡一个节点）：Value（8000 步）→ advantages（8 卡推理）→ CFG（30000 步）
  ↓ rsync 回本机（9.6）
本机：run.sh serve → 真机评估 / 下一轮采集（第 11 节）
```

| 脚本 | 作用 |
| --- | --- |
| `slurm/recap_server_env.sh` | 服务器路径（`RECAP_RUN_DIR`、`RECAP_DATASET`、示教副本）和四步的 `*_DATA` 覆盖 |
| `slurm/recap_round_prep.sbatch` | 等 `.inputs_ready` 标记（最多 8 h）→ 示教副本 → 转换 → returns → preflight → `prep.done` |
| `slurm/recap_round_train_8xa100.sbatch` | Value → advantages → CFG；每步写 `<stage>.done`，重交跳过已完成的步，并从最新 checkpoint 续训 |
| `slurm/recap_status.sh` | 一屏查看作业、阶段、步数/剩余时间、最新 checkpoint 和报错（只读） |

训练超参（与原 `pi05_connect_strings` 一致）：CFG global batch 128、30000 步、cosine
（warmup 1000，峰值 2.5e-5 → 2.5e-6）、AdamW β2=0.95、clip 1.0；Value global batch 256、8000 步。
正式训练前，每一步先用 3 个 step 的冒烟测试从大到小试 micro batch，取第一个能跑通的；不足的部分
用梯度累积补齐，global batch 不变。冒烟测试单次超过 45 分钟（`SMOKE_TIMEOUT`）判为失败，防止卡死。

### 9.2 3090 与 A100 的区别

脚本默认值按 A100-40G 设置（`no_shard`，micro batch Value 32/16/8、CFG 16/8/4/2）。3090 只有 24 GB，
提交时要改成参数分片和更小的 micro batch（9.4 的命令已经包含）：

| 变量 | A100 默认 | 3090 |
| --- | --- | --- |
| `RECAP_FSDP_SHARDING`（只作用于 CFG） | `no_shard` | `full_shard`（参数/梯度/优化器状态分到 8 卡；尚未实测） |
| `VALUE_MICRO_CANDIDATES` | `32 16 8` | `16 8 4 2` |
| `CFG_MICRO_CANDIDATES` | `16 8 4 2` | `8 4 2 1` |

Value 模型在 FSDP 单元的 forward 之外直接调用 embedding，`full_shard` 下会报 `'weight' must be 2-D`，
所以 Value 固定用配置里的 `no_shard`（约 0.7B，24 GB 卡也放得下）。两种分片都会保存完整的 `full_weights.pt`，部署方式相同。3090 单步更慢，30000 步的总时长以
`recap_status.sh` 显示的剩余时间为准。

### 9.3 同步代码和数据

同步代码（不删服务器文件，被覆盖的旧文件备份到 `backups/sync_<日期>/`）：

```bash
cd /home/agilex/FluxVla/openpi/RLinf
rsync -a --backup --backup-dir=<SERVER_HOME>/FluxVla/openpi/RLinf/backups/sync_$(date +%Y%m%d) \
  --exclude .git/ --exclude '.venv*/' --exclude logs/ --exclude outputs/ --exclude backups/ \
  --exclude checkpoints/ --exclude recap_data/ --exclude recap_artifacts/ --exclude recap_runs/ \
  --exclude __pycache__/ --exclude HOLD \
  ./ myserver:FluxVla/openpi/RLinf/
```

需要传到服务器的数据（一次性的已传过，服务器上 sha256 已核对）：

| 内容 | 说明 |
| --- | --- |
| `30000/params` 与 `30000/assets`（约 12 GB） | 已传，放在 `~/FluxVla/openpi/checkpoints/pi05_connect_strings/pi05_connect_strings_20260923/30000` |
| `recap_artifacts/siglip2-so400m-patch14-224`、`gemma-3-270m` | 已传 |
| `recap_data/lerobot_v<N>` | **每轮导出后传**（AV1 视频，约 2 GB/62 回合） |

```bash
N=1   # 本轮快照编号
rsync -a recap_data/lerobot_v$N/ myserver:FluxVla/openpi/RLinf/recap_data/lerobot_v$N/
# 校验：输出 0 才算一致
rsync -an --checksum --out-format='%i %n' recap_data/lerobot_v$N/ \
  myserver:FluxVla/openpi/RLinf/recap_data/lerobot_v$N/ | grep -c '^<f'
```

### 9.4 提交作业

在登录节点执行（`ssh myserver` 后）。新一轮换 `RECAP_ROUND`，并用 `RECAP_DATASET` 指向本轮快照：

```bash
cd ~/FluxVla/openpi/RLinf
export RECAP_ROUND=round_001                                   # 新一轮改成 round_002 …
export RECAP_DATASET=$PWD/recap_data/lerobot_v1                # 默认就是 lerobot_v1
mkdir -p recap_runs/$RECAP_ROUND && touch recap_runs/$RECAP_ROUND/.inputs_ready   # 数据已同步并校验后再打

P=$(sbatch --parsable --export=ALL slurm/recap_round_prep.sbatch)

# 在 3090 上训练：
export RECAP_FSDP_SHARDING=full_shard VALUE_MICRO_CANDIDATES="16 8 4 2" CFG_MICRO_CANDIDATES="8 4 2 1"
T=$(sbatch --parsable -p 3090 --job-name=recap_train_3090 --export=ALL \
      --dependency=afterok:$P slurm/recap_round_train_8xa100.sbatch)
# 3090 与 A100 谁先空用谁（候选覆盖两种卡，冒烟测试自动选）：
#   export RECAP_FSDP_SHARDING=full_shard VALUE_MICRO_CANDIDATES="32 16 8 4 2" CFG_MICRO_CANDIDATES="16 8 4 2 1"
#   T=$(sbatch --parsable -p 3090,gpu --job-name=recap_train --export=ALL \
#         --dependency=afterok:$P slurm/recap_round_train_8xa100.sbatch)
# 或在 A100 上训练（用脚本默认值，不要导出上面三个变量）：
#   T=$(sbatch --parsable --export=ALL --dependency=afterok:$P slurm/recap_round_train_8xa100.sbatch)
echo "prep=$P train=$T"
```

- 3090 节点偶尔被别人的纯 CPU 作业占满 64 核（GPU 空着，`sinfo` 显示 allocated），此时作业显示
  `Pending (Resources)`，只能等或改用 `-p 3090,gpu`。
- prep 成功后 train 才开始；prep 失败时 train 会显示 `DependencyNeverSatisfied`，要 `scancel` 掉再重交。
- 停止训练：`scancel <作业号>`。已完成的阶段保留，重交同一条 train 命令（去掉 `--dependency`）会跳过
  已完成阶段并从最新 checkpoint 续训。
- 想重做某一步：删掉对应的 `recap_runs/<轮次>/<stage>.done`（以及它之后各步的 `.done`）再重交。

### 9.5 查看训练情况

**一屏总览**（本机直接运行，最常用）：

```bash
ssh myserver 'bash ~/FluxVla/openpi/RLinf/slurm/recap_status.sh'         # 加 --gpu 看显卡占用
ssh myserver 'RECAP_ROUND=round_002 bash ~/FluxVla/openpi/RLinf/slurm/recap_status.sh'   # 其他轮次
```

输出依次是：Slurm 作业状态、四个阶段是否完成、micro batch 冒烟结果、Value/CFG 当前步数
（`3600/30000 [已用<剩余, 每步耗时]`）、最新 checkpoint、prep/train 日志最后几行和报错。

**实时日志**：

```bash
ssh myserver 'tail -f ~/FluxVla/openpi/RLinf/logs/recap_train_<作业号>.out'      # 作业总日志
ssh myserver "tail -f ~/FluxVla/openpi/RLinf/recap_runs/round_001/cfg.log | tr '\r' '\n'"   # 某一阶段（value.log / advantages.log / cfg.log）
```

**TensorBoard 曲线**（loss、学习率、每步耗时）：在本机开一个终端，保持不关：

```bash
ssh -L 6106:127.0.0.1:6106 myserver \
  'cd ~/FluxVla/openpi/RLinf && .venv-rlt/bin/tensorboard --logdir recap_runs/round_001 --host 127.0.0.1 --port 6106'
```

然后在本机浏览器打开 <http://127.0.0.1:6106>。端口被占用就换一个（两处一起改）。不要用 6006/6016，
那是本机的策略服务端口。

**怎么判断是否正常**：

| 现象 | 含义 / 处理 |
| --- | --- |
| prep 日志停在 `Merged 0 OpenPI LoRA tensors` 一段时间 | 正在写转换后的权重，正常 |
| `smoke_cfg_mb8.log` FAILED（OOM），`mb4` ok | 正常，正式训练用 4 |
| 所有候选都 FAILED | 作业退出；看 `recap_runs/<轮次>/smoke_*.log` 末尾的报错。不是 OOM 的错误不会再试更小的 micro batch |
| `Could not load libtorchcodec` / `libavutil.so.5x` | 服务器没有 torchcodec 能用的 FFmpeg；`recap_server_env.sh` 已默认 `RECAP_VIDEO_BACKEND=pyav`（PyAV 自带解码器，单帧约 0.06–0.14 s） |
| `train/` loss 前 1000 步（warmup）明显下降后趋平 | 正常 |
| loss 变成 nan，或每步耗时突然变长很多 | `scancel`，看日志；从上一个 checkpoint 续训 |

Value 每 1000 步、CFG 每 2500 步保存一次 checkpoint，位置在
`recap_runs/<轮次>/{value,cfg}/<实验名>/checkpoints/global_step_<N>/`。

### 9.6 取回结果并部署

训练结束时 train 日志最后一行是 `RECAP ROUND DONE: <checkpoint 路径>`。只取部署需要的文件
（不取优化器状态）：

```bash
cd /home/agilex/FluxVla/openpi/RLinf
R=round_001; STEP=30000
SRC=FluxVla/openpi/RLinf/recap_runs/$R/cfg/piper_recap_cfg/checkpoints/global_step_$STEP
rsync -am --include='*/' --include='full_weights.pt' --include='piper_recap_policy.json' --exclude='*' \
  myserver:$SRC/ recap_runs/$R/cfg/piper_recap_cfg/checkpoints/global_step_$STEP/
# 基础模型转换结果只需取一次
rsync -a myserver:FluxVla/openpi/RLinf/recap_artifacts/pi05_base/ recap_artifacts/pi05_base/
```

然后按第 11 节 `run.sh serve --checkpoint $PWD/recap_runs/$R/cfg/piper_recap_cfg/checkpoints/global_step_$STEP`。
中间 checkpoint（如 15000、25000）也可以用同样方式取回做对比。

## 10. 断点恢复

raw 采集重启：重复 `run.sh collect`，已 C/X 提交的 episode 均保留；退出时尚未打分
的片段不会猜测标签。

Value 或 CFG 恢复使用**相应阶段**的完整 checkpoint，并带上与原运行相同的数据覆盖：

```bash
bash examples/recap_piper/run.sh value "$VAL_DATA" \
  runner.resume_dir=/绝对路径/value/piper_recap_value/checkpoints/global_step_100

bash examples/recap_piper/run.sh cfg "$CFG_DATA" \
  runner.resume_dir=/绝对路径/cfg/piper_recap_cfg/checkpoints/global_step_100
```

新一轮数据集属于另一轮离线训练，先重新计算标签；不要照搬 RLT 的
`RLINF_REPLAY_LOAD_PATH` 或 `RLINF_STAGE2_RESUME_DIR`。

## 11. 部署 RECAP 策略并迭代

需要本机已有 `recap_artifacts/pi05_base` 和 CFG checkpoint。停止 6016 上的基础服务（终端 B）后运行：

```bash
cd /home/agilex/FluxVla/openpi/RLinf
source examples/recap_piper/env.sh
bash examples/recap_piper/run.sh serve \
  --checkpoint /绝对路径/cfg/piper_recap_cfg/checkpoints/global_step_2000 \
  --guidance-scale 1.0
```

服务不打开 ROS/CAN，不会使能机器人。只接受完整 RECAP FSDP 权重，缺失键时直接报错，
不会把 RLT MLP checkpoint 当成 RECAP。端口 6006 被入口明确拒绝，默认 6016。

CFG 在每一步流积分中组合速度场：

```text
v = (1 - guidance_scale) * v_unconditional + guidance_scale * v_positive
```

`1.0` 使用 positive 条件速度场；更大的 guidance 不是更小的物理动作，也不是 Actor
纠正限幅。第一次保持 1.0，不要把它当速度旋钮。

终端 A、C、D 与第 4 节相同。训练后服务与初始服务都返回 `[50,14]` 的绝对关节/夹爪
计划，客户端负责 RTC、平滑和实际执行。CFG 双分支的推理延迟可能高于原 JAX，先核对
延迟及接近段速度，再统计无人工接管的成功率，不能只看训练 loss。

下一轮：继续 collect → 导出到 `lerobot_v2` → 第 8 节四步（示教副本可复用，但 returns、
Value、advantage 都要重新计算）→ 部署新 checkpoint。也可以把 `lerobot_v1` 作为第三个
rollout 条目一并加入 `*_DATA`。

## 12. 代码与检查入口

| 文件 | 作用 |
| --- | --- |
| `examples/recap_piper/env.sh`、`run.sh` | 本机独立环境与阶段入口 |
| `run_native_server.sh`、`run_client.sh` | 复用原基础服务和已验证真机客户端 |
| `examples/offline_rl/config/piper_recap_*.yaml` | 四步单机配置，以及 `piper_recap_sft_int.yaml` 对照基线 |
| `rlinf/recap_piper/collect.py` | 原 ROS 请求 + 图像/动作/标签记录 |
| `rlinf/recap_piper/scoring.py` | 三次插接打分（N/U/C 校验）、逐帧进度、按根统计 |
| `rlinf/recap_piper/data.py` | 原始回合事务、不可覆盖 LeRobot 导出 |
| `rlinf/recap_piper/source.py`、`cli.py` | 原配置 prompt/架构解析、转换、CFG 服务 |
| `rlinf/recap_piper/policy.py` | 原生 LoRA、源统计、14-D CFG 动作适配 |
| `rlinf/recap_piper/preflight.py` | 不启动 ROS/CUDA/Ray 的数据与资产检查 |
| `tests/unit_tests/test_piper_recap.py` | 无硬件数据、标签、变换和部署契约测试 |

对既有 RECAP 代码的补充包括：ValueDataset 的 Piper 三相机、Value 推理映射、
advantage 的 Piper key mapping、CFG 的 openpi_data 传递和 Piper 模型加载分支、
`[False]` 成功标记的解析修复，以及三个默认关闭、由 Piper 配置开启的选项：
`compute_returns` 的 `takeover_penalty`、`compute_advantages` 的 `force_intervention_positive`、
`FSDPCfgWorker` 的 `data.plain_sft`。未改 RLT 路由、训练或 Piper ROS 驱动。

```bash
.venv-rlt/bin/python -m pytest -q tests/unit_tests/test_piper_recap.py
```

本文件为独立中文 README，未新增 Sphinx 页面。
