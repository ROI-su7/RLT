# Cobot Magic（双 Piper）RLT 当前实施说明

> 更新日期：2026-07-25。本文只保留当前有效方案、已经验证的问题和实施命令。
> 旧的 PyTorch π0.5 转换训练、旧 Stage 1/Stage 2 checkpoint、episode-gated
> learner、residual actor和milestone reward均不再使用。
>
> 2026-09-11：第 9 节记录阶段 0/1 实施：全参 pi05 Stage 1、15 Hz/C=10、stride-2、
> 归一化 delta actor、RTC 时间对齐修复，以及“actor 接管后机械臂乱动”的诊断。前文与
> 第 9 节冲突时，以第 9 节为准。

## 1. 当前结论

任务使用的策略是经过 LoRA 微调的原生 JAX π0.5：

```text
/home/agilex/FluxVla/openpi/checkpoints/pi05_wire/7_16_pi05_wire/30000
```

JAX checkpoint 本身可以正常完成当前插线任务。问题出现在把它转换成 PyTorch 后的视觉
骨干路径，因此当前方案不再让转换后的 PyTorch π0.5 参与 Stage 1 特征提取或真机动作
生成：

```text
相机 + 14维关节状态
        │
        ▼
原生 JAX π0.5 + LoRA（冻结）
        ├── reference action / 真机基础动作
        └── 视觉 prefix feature
                     │ action+prefix WebSocket
                     ▼
              PyTorch RL Token
                     │
                     ▼
              Stage 2 actor / critic
```

Stage 1 只训练 RL Token 自编码器。π0.5、视觉骨干和 LoRA 均冻结。

## 2. 问题与解决方案

### 问题 1：转换后的 PyTorch 策略出现真机插入偏移

真机表现：

- 第一槽相对原生 JAX 偏右后才下插；
- 第三槽在距离目标还差一点时提前插入；
- 这不是随机抖动，而是可重复的动作位置和时机偏移。

验证方法：

1. 对同一 observation 使用相同输入预处理和固定去噪 noise；
2. 比较 JAX/PyTorch 输入、视觉 prefix、每一步去噪状态和最终 action；
3. 对多条数据检查动作 MAE、最大误差和方向一致率。

验证结果：

| 检查项 | 结果 |
| --- | --- |
| state、token、mask | 一致 |
| 图像（消除 NHWC/NCHW 表示差异后） | MAE 约 `7e-8` |
| 首个 suffix token | MAE 约 `3.86e-4` |
| 视觉 prefix token | MAE 约 `1.89e-2`，最大误差 `1.0` |
| prefix transformer 输出 | MAE 约 `8.26e-2`，最大误差 `62.375` |
| 最终 14 维机械臂 action | MAE `0.01919 rad`，最大误差 `0.05421 rad` |
| action 方向一致率 | `0.6911` |

结论：误差从 PyTorch 视觉 prefix 路径开始，在 transformer 和 10 步去噪过程中继续
放大。它足以改变插入位置和插入时机，不能通过降低 parity 门槛接受。

解决方案：

- 保留并直接使用原生 JAX LoRA checkpoint；
- JAX π0.5 全冻结，只输出 reference action 和视觉 prefix feature；
- 用 DLPack 将 feature 交给 PyTorch；
- PyTorch 只训练 RL Token，后续只承载轻量 actor/critic；
- RL 关闭时必须保持原生 JAX 动作，不再经过转换后的 PyTorch 视觉骨干。

该方案消除的是“模型转换造成的系统性偏移”。Stage 2 学习后产生的动作修正属于 RL
策略变化，仍需通过动作限幅、独立评估和真机安全测试约束。

### 问题 2：8 个 JAX 进程恢复到同一张 GPU

首次 8 卡测试中，Orbax 在 GPU 0 申请额外 `2.25 GiB` 时 OOM。原因是 torchrun 的
每个子进程都能看到全部 GPU，而单设备 JAX checkpoint 默认恢复到进程内 device 0。

解决方案：

- 在导入 JAX 和 PyTorch 前，按 `LOCAL_RANK` 缩小 `CUDA_VISIBLE_DEVICES`；
- 每个进程只看见一张物理 3090；
- JAX、PyTorch、NCCL 和 DLPack 在进程内统一使用逻辑 `cuda:0`。

### 问题 3：Torch 数据图像布局与 JAX 不一致

第二次测试到达 JAX SigLIP 前向后报错：

```text
expected (14, 14, 3, 1152), got (14, 14, 224, 1152)
```

原因是 OpenPI Torch loader 输出 `NCHW`，原生 JAX SigLIP 接收 `NHWC`，宽度 224
被误当成通道数。

解决方案：只在进入 JAX 前执行 `NCHW -> NHWC` 转置，不改变像素值、归一化、
observation、action 或训练标签。

### 问题 4：JAX 与 DDP RL Token 的常驻显存竞争

`accumulate=1` 的完整测试已经成功：

- 8 卡均完成 JAX checkpoint 恢复；
- 完成一次 JAX prefix 前向；
- 完成 DLPack 特征传递和 RL Token 反向；
- `rlt_loss=4.5399`；
- 成功保存并重新读取 `rlt_token.pt`，包含 62 个模型 tensor 和 62 个 optimizer
  state。

正式 `accumulate=8` 测试随后失败：第一次 PyTorch 反向后，PyTorch allocator 保留
activation cache，第二次 JAX 前向申请 `68.34 MiB` 时 OOM。先提取 8 个 JAX
feature、再集中做 8 次 backward 的两段式执行解决了同一 step 内的竞争，但
`optimizer.step()` 后完整 AdamW 状态仍常驻每张 GPU，step 2 的 JAX 前向申请
`255.34 MiB` 时再次 OOM。

最终解决方案同时采用两段式执行和 FSDP `FULL_SHARD`：

```text
先连续提取 8 个冻结 JAX feature并暂存 CPU
                         ↓
FSDP 分片执行 8 次 PyTorch RL Token backward
                         ↓
optimizer.step()
```

RL Token 参数、梯度和 AdamW 状态在 8 张 GPU 间切分，保存时再汇总为完整
checkpoint；step 之间释放 PyTorch activation cache。该修改不改变 loss、样本顺序、
有效 batch 或梯度累积。

最终 2-step、`accumulate=8` 测试 `3428196` 已通过：

- step 1：`rlt_loss=4.54675`；
- step 2：`rlt_loss=4.52119`；
- 没有 OOM 或 `ChildFailedError`；
- `step_2/rlt_token.pt` 大小约 4.46 GB；
- checkpoint 可重新读取，包含 62 个 RL Token tensor 和 62 个 optimizer state；
- 元数据记录 `sharding_strategy=FULL_SHARD`。

## 3. 当前 Stage 1 配置

| 项目 | 当前值 |
| --- | --- |
| JAX checkpoint | `checkpoints/pi05_wire/7_16_pi05_wire/30000` |
| 数据集 | `lerobot_data/total_data` |
| GPU | 8×RTX 3090 |
| 每卡 micro batch | 1 |
| 梯度累积 | 8 |
| 每个 optimizer step 的有效样本 | 64 |
| 首轮 optimizer steps | 2000 |
| 首轮累计样本数 | 约 128,000 |
| 保存间隔 | 250 steps |
| 学习率 | `2.5e-5` |
| warmup | 500 steps |
| 最低学习率 | `2.5e-6` |
| 训练参数 | 仅 RL Token encoder/decoder |
| prefix | 三相机共 768 个 image token，不使用固定指令 token |

2000 step 是第一轮验证规模，不是硬性上限。需要比较 250、500、…、2000 的 validation
重建误差；若 2000 step 时 loss 仍明显下降，再从 checkpoint 继续到 5000。不能只凭
训练步数判定 RL Token 已可用于 Stage 2。

实现文件：

- `toolkits/realworld_check/train_piper_rlt_token_native_jax.py`
- `slurm/train_piper_rlt_token_native_jax_8x3090.sbatch`
- `toolkits/realworld_check/generate_openpi_action_parity.py`
- `toolkits/realworld_check/compare_openpi_denoise_trace.py`

## 4. 服务器训练与查看

### 4.1 当前测试

已经通过的完整配置 2-step、`accumulate=8`、FSDP 测试：

```text
JOB_ID=3428196
```

复查日志：

```bash
ssh myserver
cd <SERVER_HOME>/openpi/RLinf

grep -E '"step"|RESOURCE_EXHAUSTED|out of memory|ChildFailedError' \
  logs/piper_rlt_token_jax_3428196.out \
  logs/piper_rlt_token_jax_3428196.err
```

另开终端查看错误：

```bash
ssh myserver
cd <SERVER_HOME>/openpi/RLinf
ls -lh results/piper_rlt_token_native_jax_fsdp_smoke/step_2/rlt_token.pt
```

本次已经满足：

- 日志同时出现 step 1 和 step 2；
- 两个 loss 均为有限值且第二步下降；
- 没有 `RESOURCE_EXHAUSTED`、`CUDA out of memory` 或 `ChildFailedError`；
- 生成 `results/piper_rlt_token_native_jax_fsdp_smoke/step_2/rlt_token.pt`；
- checkpoint 可以重新读取。

日志里的 ROCm/TPU backend 初始化失败可以忽略；实际 backend 是 CUDA。

`3428194` 和 `3428195` 是 FSDP 修复前失败的作业，没有可用正式权重，不得用于
Stage 2。

### 问题 5：Stage 2 动作抖动且实际控制频率只有约 5.7 Hz

`3428642` 的 reference-only 真机测试已经确认动作位置偏移消失，但出现两个新的执行
问题：

- 机械臂收到稀疏且不连续的目标点，抖动明显；
- 日志每 15 steps 约耗时 2.6 秒，实际只有约 `5.7 Hz`，远低于配置的 `15 Hz`。

分段测速结果：

| 阶段 | 平均耗时 |
| --- | ---: |
| 真机本地原生 JAX 模型 | 约 `71–75 ms` |
| 服务器到真机 action+3 MiB prefix 完整请求 | 约 `144 ms` |
| 3090 Stage-1 RLT encoder | 约 `11.4 ms` |

因此慢速不是 Stage-1 checkpoint 或 RL Token 网络造成的。根因是当前调度每执行 5 个
动作就同步等待下一次远程请求，同时 `PiperWireEnv.step()` 在一个低层 step 内重复
获取并解码了两组三相机 observation。

当前修复：

1. 安全限幅前和一个 C=20 chunk 的中间 19 步只读取 14 维关节反馈；仅在 chunk
   最后一步解码三路相机并产生下一次推理所需的完整 observation；
2. action chunk 从 5 调整到 20，与已验证的 `piper_real` 部署和论文中的
   20-step VLA 执行前缀一致；
3. 策略目标仍按 15 Hz 产生，但与已验证的 `examples/piper_real` 完全一致，按
   200 Hz 连续发布臂关节线性插补目标；
4. 12 个臂关节使用与 `piper_real` 相同的 EMA `alpha=0.4` 和 `0.4 rad/s` 硬速度
   限制；夹爪不参与插补、EMA 或限速，保持原始开合时机；
5. 新旧 chunk 的边界融合仅由 RTC 在按实际推理延迟对齐后执行，避免在环境层再次融合
   同一段动作；
6. 开启 execution-side RTC：当 C=20 执行完而下一次 JAX 请求尚未返回时，执行同一次
   原生 JAX H=50 计划的未使用 suffix；新 chunk 返回后按实际 bridge 步数对齐，避免
   20-step 边界的约 140 ms 指令空档。
7. 主 chunk 与 RTC bridge 共用唯一的 15 Hz 策略目标时钟；每 100 个目标记录策略
   时钟误差、原始动作步差、滤波后步差和原始二阶差分，以区分模型抖动与执行调度
   抖动；
8. 真机实测发现 Ray EnvWorker 在动作块数据处理期间会同时饿死进程内的 RTC 和
   200 Hz ROS Python 线程：空闲时 ROS 发布稳定为 200 Hz、最大间隔约 6 ms，运动时
   降到约 187--189 Hz，并周期性出现 111--115 ms 空洞。配置
   `isolated_control_process: true` 后，15 Hz RTC 尾动作消费、EMA/限速、线性插补和
   200 Hz ROS 发布全部由独立进程执行，不再受 Ray 主进程 GIL 阻塞；
9. 每 25 次请求记录 `jax_wall`、`jax_model`、`rpc_and_prefix` 和
   `rlt_encoder` 耗时，下一轮真机直接以日志验收。

由于 chunk 从 5 变为 20，旧的 Stage-2 actor/checkpoint/replay 不得直接恢复；新的
warmup 门槛为 600 个干净的 chunk transition。

### 4.2 测试通过后提交正式 2000-step

正式作业已经完成：

```text
JOB_ID=3428217
output=results/piper_rlt_stage1_native_jax_token_20260724_v2
```

直接查看当前作业：

```bash
ssh myserver
cd <SERVER_HOME>/openpi/RLinf

export JOB_ID=3428217
squeue -j "$JOB_ID" -o "%.18i %.24j %.9T %.10M %.30R"
tail -f "logs/piper_rlt_token_jax_${JOB_ID}.out"
```

下面是以后重新提交相同训练时使用的模板。

```bash
ssh myserver
cd <SERVER_HOME>/openpi/RLinf

export RLINF_OPENPI_SOURCE_DIR=<SERVER_HOME>/openpi
export RLINF_PI05_JAX_DIR=<SERVER_HOME>/openpi/checkpoints/pi05_wire/7_16_pi05_wire/30000
export RLINF_PIPER_DATASET_DIR=<SERVER_HOME>/openpi/lerobot_data/total_data
export RLINF_STAGE1_OUTPUT_DIR=<SERVER_HOME>/openpi/RLinf/results/piper_rlt_stage1_native_jax_token_20260724_v2
export RLINF_STAGE1_STEPS=2000
export RLINF_STAGE1_ACCUMULATE=8

JOB_ID=$(sbatch --parsable --export=ALL \
  slurm/train_piper_rlt_token_native_jax_8x3090.sbatch)
echo "JOB_ID=$JOB_ID"
```

监控正式训练：

```bash
squeue -j "$JOB_ID" -o "%.18i %.24j %.9T %.10M %.30R"
tail -f "logs/piper_rlt_token_jax_${JOB_ID}.out"
```

快速查看 loss：

```bash
grep -E '"step"|"rlt_loss"|RESOURCE_EXHAUSTED|out of memory|ChildFailedError' \
  "logs/piper_rlt_token_jax_${JOB_ID}.out" \
  "logs/piper_rlt_token_jax_${JOB_ID}.err"
```

查看已保存 checkpoint：

```bash
find "$RLINF_STAGE1_OUTPUT_DIR" -maxdepth 2 -name rlt_token.pt \
  -printf '%p %s bytes\n' | sort -V
```

停止作业：

```bash
scancel "$JOB_ID"
```

## 5. Stage 1 验收与 Stage 2 门禁

Stage 1 完成后必须满足：

1. checkpoint 来源指向原生 JAX LoRA `7_16_pi05_wire/30000`；
2. 只有 RL Token 权重发生更新，π0.5 和 LoRA 没有训练；
3. validation reconstruction loss 有下降且未出现 NaN/Inf；
4. RL 关闭时，reference action 与现有原生 JAX 真机策略一致；
5. RL 逐步开启时，动作变化来自 actor，而不是 JAX/PyTorch 转换；
6. 第一槽和第三槽先做低速、限幅、无障碍测试。

8 个保存点已使用相同的 32 个固定样本完成比较。由于 Stage 1 训练前没有划分独立
validation episode，这里只能判断训练分布内的重建稳定性，不能当作独立泛化指标。

| step | fixed-data MSE | cosine |
| ---: | ---: | ---: |
| 250 | 1.246477 | 0.730261 |
| 500 | 0.555311 | 0.873192 |
| 750 | 0.515904 | 0.882675 |
| 1000 | 0.491121 | 0.889334 |
| 1250 | 0.470783 | 0.894630 |
| 1500 | 0.460346 | 0.897937 |
| 1750 | 0.456184 | 0.899009 |
| 2000 | **0.455664** | **0.899181** |

当前选择：

```text
<SERVER_HOME>/openpi/RLinf/results/piper_rlt_stage1_native_jax_token_20260724_v2/step_2000/rlt_token.pt
```

`step_2000` 略优于 `step_1750`，且未出现后期退化。最终是否可用于真机仍由
reference-only 回归和小规模 RLT rollout 决定。

### 5.1 Stage 2 原生 JAX 双输出接口

Stage 2 已改为以下数据流：

```text
真机 observation
  └── 原生 JAX pi05_wire + LoRA（一次 prefix/KV-cache forward）
        ├── reference_actions: [50, 14]
        └── rlt_prefix: [768, 2048] + mask
              └── 服务器冻结 Stage-1 RLT encoder
                    └── z_rl: [2048]
                          └── Stage-2 actor / twin critic
```

关键约束：

1. `reference_actions` 和 `rlt_prefix` 必须来自同一次请求、同一个 observation；
2. prefix forward 同时用于填充动作去噪的 KV cache，不重复运行视觉骨干；
3. GPU rollout 节点只加载 Stage-1 RLT encoder，不加载转换后的 PyTorch π0.5；
4. prefix 以 float16 传输，三相机 768 token 约 3 MiB；启动前需用真实 observation
   检查 bfloat16 → float16 → bfloat16 的序列化误差；
5. Stage 2 replay 继续保存 `z_rl/proprio/ref_chunk` 对齐的 transition。

### 5.1.1 基础 VLA reference warmup

Piper 暂时没有可用的人工接管 action，第一轮 Stage 2 使用两类 replay 数据：

1. 已有一定成功率的原生 JAX VLA reference rollout；
2. learner-ready 后的 RL actor rollout。

键盘操作保持不变，`b` 始终表示“进入关键阶段并记录”，但实际执行策略由门禁决定：

```text
有效 replay < 600：
    b → 原生 JAX 控制 + 记录 reference transition

有效 replay ≥ 600，但 learner update_step < 1000：
    b → 原生 JAX 继续控制 + 继续记录
    learner 使用 replay 完成初始 critic/BC 更新

learner update_step ≥ 1000，并已同步到 rollout：
    b → RL actor 控制 + 记录 actor transition
```

按约 10 秒关键窗口和 `C=20` 估计，每条约产生 7–8 个 chunk transition，约 80 个
episode 才能产生 600 transitions。最终以 `rlt/global_replay_samples` 为准；不足 600
时继续补充。

在 1,000 次 learner update 完成前，actor 不得接管真机；持续以 JAX
reference 执行并记录 transition。

关键监控指标：

```text
replay/reference_warmup_rate = 1  # 当前记录的是JAX warmup
replay/actor_switch_rate = 0      # actor尚未控制
rlt/global_replay_samples         # replay有效transition数量
rlt/update_step                   # learner累计更新数
rlt/ready_for_online = 1          # actor接管门禁已开放
```

`x/q` 是失败/安全终止标签，不是人工示教 action；在真正接入主从或其他人工控制器前，
不要把它们统计成人工 intervention 数据。

相关实现：

- `src/openpi/models/pi0.py`：`sample_actions_with_prefix`；
- `src/openpi/policies/policy.py`：双输出策略；
- `scripts/serve_policy.py`：`--return-rlt-prefix`；
- `rlinf/models/embodiment/native_jax_rlt/__init__.py`：远程 JAX 客户端和冻结 RLT encoder；
- `examples/embodiment/config/realworld_piper_rlt_stage2_ac_mlp.yaml`：Stage 2 新数据流。

旧的 `rollout.rlt_feature_model.model_type=openpi` 路径不得再用于 Piper，它会重新加载
已确认存在视觉偏差的 PyTorch π0.5。

### 5.2 Stage 2 启动前剩余门禁

代码契约、Hydra 配置、transition/reward 单元测试已经通过。真机启动前仍必须完成：

1. 使用选定 checkpoint 启动原生 JAX 双输出 server；
2. preflight 必须显示 `native action+prefix, prefix_seq_len=768`；
3. 日志中的平均一次请求必须明显小于 20-action chunk 的约 `1333 ms` 执行窗口；
   `Episode progress` 的实际频率应接近 15 Hz；
4. 关闭 RL（不按 `b`）做第一槽、第三槽 reference-only 回归，动作必须恢复原生 JAX
   基线，不再出现 PyTorch 转换造成的系统性右偏/提前插入；
5. 通过以上门禁后，才开始 Stage 2 在线采集与更新。

### 5.3 新接口启动模板

> 2026-09-11 起使用全参 checkpoint、`--rlt-token-checkpoint` 和 C=10，命令见 9.9。
> 本节是 LoRA 时期的历史模板。

真机先启动双输出 JAX server：

```bash
cd /home/agilex/FluxVla/openpi

PYTHONPATH=/home/agilex/FluxVla/openpi/src \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
/home/agilex/FluxVla/openpi/RLinf/.venv-rlt/bin/python -u scripts/serve_policy.py \
  --env ALOHA \
  --port 6006 \
  --default-prompt "insert the wire" \
  --return-rlt-prefix \
  --rlt-prefix-seq-len 768 \
  policy:checkpoint \
  --policy.config pi05_wire \
  --policy.dir /home/agilex/FluxVla/openpi/checkpoints/pi05_wire/7_16_pi05_wire/30000
```

服务器提交 Stage 2 时必须传入选定的 `rlt_token.pt`，不再传
`RLINF_PI05_TORCH_DIR` 或旧 `RLINF_STAGE1_DIR`：

```bash
ssh myserver
cd <SERVER_HOME>/openpi/RLinf

export RLINF_STAGE1_RLT_TOKEN=<SERVER_HOME>/openpi/RLinf/results/piper_rlt_stage1_native_jax_token_20260724_v2/step_2000/rlt_token.pt
export RLINF_JAX_REF_HOST=192.168.209.107
export RLINF_JAX_REF_PORT=6006

JOB_ID=$(sbatch --parsable \
  --export=ALL,RLINF_STAGE1_RLT_TOKEN="$RLINF_STAGE1_RLT_TOKEN",RLINF_JAX_REF_HOST="$RLINF_JAX_REF_HOST",RLINF_JAX_REF_PORT="$RLINF_JAX_REF_PORT" \
  slurm/train_piper_rlt_stage2_dual.sbatch)
echo "JOB_ID=$JOB_ID"
```

真机加入 Ray 前，`join_piper_rlt_stage2_dual_robot.sh` 的 preflight 会拒绝旧的
action-only JAX server。

### 5.4 Stage 2 Warmup 作业模板（2026-07-25）

`3428580` 和 `3428642` 已停止，仅用于接口和动作位置验证。应用本节抖动/速度修复后
必须全新提交，不能恢复它们的 5-action actor 或 replay。

```text
Stage-1 RLT=<SERVER_HOME>/openpi/RLinf/results/piper_rlt_stage1_native_jax_token_20260724_v2/step_2000/rlt_token.pt
Warmup=600 reference transitions + 1000 learner updates
```

#### 服务器：监控已提交作业

```bash
ssh myserver
cd <SERVER_HOME>/openpi/RLinf

export JOB_ID=<新作业号>

squeue -j "$JOB_ID" -o "%.18i %.16j %.2t %.10M %R"
tail -f "logs/piper_rlt_s2_${JOB_ID}.out"
```

重新读取本次计算节点 IP：

```bash
export GPU_HEAD_IP=$(cat "logs/stage2_${JOB_ID}.head")
echo "GPU_HEAD_IP=$GPU_HEAD_IP"
```

#### 真机终端1：原生 JAX action+prefix server

如果已经运行，保持原终端，不要重复启动。需要重启时使用：

```bash
cd /home/agilex/FluxVla/openpi

PYTHONPATH=/home/agilex/FluxVla/openpi/src \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
/home/agilex/FluxVla/openpi/RLinf/.venv-rlt/bin/python -u scripts/serve_policy.py \
  --env ALOHA \
  --port 6006 \
  --default-prompt "insert the wire" \
  --return-rlt-prefix \
  --rlt-prefix-seq-len 768 \
  policy:checkpoint \
  --policy.config pi05_wire \
  --policy.dir /home/agilex/FluxVla/openpi/checkpoints/pi05_wire/7_16_pi05_wire/30000
```

### 5.5 Stage 2 执行侧 RTC

Stage 2 保持论文定义的 `H=50、C=20`。JAX 返回完整 50 步参考计划，actor 和
critic 仍只学习 20 步 chunk。执行完一个 chunk 后，Piper 在下一次
JAX/RL-Token 请求期间最多继续执行 5 个尚未使用的 VLA 尾部动作：

```text
执行 C=20
→ 用 H=50 中的后续最多 5 步覆盖推理空窗
→ 按实际执行的 bridge step 平移新计划
→ 双臂前 5 步边界融合
→ 夹爪保持离散
```

对应配置：

```yaml
rtc_enabled: true
rtc_max_bridge_steps: 5
rtc_boundary_blend_steps: 5
native_action_horizon: 50
```

RTC 只使用已经由冻结 VLA 预测的安全尾部，不做关节速度外推。对齐后的实际
`C=20` 动作会回写 replay transition。日志指标 `env/rtc_bridge_steps` 表示每次
交接实际使用了多少尾部动作；正常应在 `2–5` 之间。按 `q` 或 episode terminal
时不会启动新的 RTC 尾部，环境关闭和 hold 操作也会先停止 RTC 线程。

#### 真机终端2：ROS、相机和 Piper

若下列 ROS topics 已经持续发布，不要重复运行启动脚本。

```bash
cd /home/agilex/FluxVla/openpi/RLinf
bash examples/embodiment/start_piper_ros_tmux.sh

export ROS_MASTER_URI=http://192.168.209.107:11311
export ROS_IP=192.168.209.107

rostopic echo -n 1 /puppet/joint_left
rostopic echo -n 1 /puppet/joint_right
rostopic echo -n 1 /camera_f/color/image_raw
```

#### 真机终端3：加入本次 Ray 集群

```bash
cd /home/agilex/FluxVla/openpi/RLinf

export JOB_ID=3430510
export GPU_HEAD_IP=$(ssh myserver "cat <SERVER_HOME>/openpi/RLinf/logs/stage2_${JOB_ID}.head")
export ROS_MASTER_URI=http://192.168.209.107:11311
export ROS_IP=192.168.209.107
export RLINF_JAX_REF_HOST=192.168.209.107
export RLINF_JAX_REF_PORT=6006

RLINF_GPU_HEAD_IP="$GPU_HEAD_IP" \
RLINF_JAX_REF_HOST="$RLINF_JAX_REF_HOST" \
RLINF_JAX_REF_PORT="$RLINF_JAX_REF_PORT" \
bash examples/embodiment/join_piper_rlt_stage2_dual_robot.sh
```

启动脚本会自动选择唯一具备全部控制按键的可读 evdev 键盘。只有检测到多个候选
设备时，才设置 ``RLINF_KEYBOARD_DEVICE``；此时优先使用
``/dev/input/by-id/...`` 稳定路径，不要写死会随重启或 USB 插拔变化的
``/dev/input/eventN``。本机 Logitech K400 Plus 的稳定路径为
``/dev/input/by-id/usb-Logitech_USB_Receiver-if02-event-mouse``。

cd /home/agilex/FluxVla/openpi/RLinf

  RLINF_GPU_HEAD_IP=192.168.81.200 \
  RLINF_JAX_REF_HOST=192.168.209.107 \
  RLINF_JAX_REF_PORT=6006 \
  bash examples/embodiment/join_piper_rlt_stage2_dual_robot.sh


preflight 必须全部通过，并最终显示：

```text
Piper RLT Stage-2 preflight PASSED
Piper host joined <GPU_HEAD_IP>:6379 as RLINF node rank 1
```

#### 50条基础 VLA Warmup 操作

```text
i：开始 episode
JAX：完成抓取和运输
b：进入关键阶段，JAX继续控制并开始记录
c：成功结束，reward=1
x：普通失败结束，reward=0
空格：软件 HOLD，暂停但不终止；i 恢复
t：b 后可切换人工采集；等待 TAKEOVER active
h：结束回合后返回 launch 捕获的四臂原点
q：终止并 HOLD，reward=0
```

达到约50条后，以服务器指标而不是肉眼 episode 数为准：

```text
rlt/global_replay_samples >= 600
replay/reference_warmup_rate ≈ 1
replay/actor_switch_rate = 0
```

达到600 transitions 后暂停开始新 episode，等待：

```text
rlt/update_step >= 1000
rlt/ready_for_online = 1
```

门禁开放并同步权重后，下一条按 `b` 才会执行 RL actor。

#### 停止本次作业

服务器：

```bash
ssh myserver "scancel <JOB_ID>"
```

真机只停止 Ray，保留 ROS/JAX：

```bash
cd /home/agilex/FluxVla/openpi/RLinf
source .venv-rlt/bin/activate
ray stop --force
```

## 7. 真机固定约定

| 项目 | 当前约定 |
| --- | --- |
| 指令 | `insert the wire` |
| 状态/动作 | 左 6 关节 + 左夹爪 + 右 6 关节 + 右夹爪，共 14 维绝对位置 |
| 相机顺序 | `cam_high`、`cam_left_wrist`、`cam_right_wrist` |
| 策略频率 | 15 Hz |
| 执行 chunk | 20 actions |
| RTC连续缓冲 | 最多 5 actions，覆盖约 145 ms 请求并限制旧计划开环执行 |
| ROS命令发布 | 200 Hz线性插补，EMA 0.4，臂关节上限0.4 rad/s |
| 控制隔离 | `isolated_control_process: true`，15 Hz/200 Hz 时钟不与 Ray EnvWorker 共进程 |
| 模型 horizon | 50 |
| 去噪步数 | 10 |
| reward | 成功终止 `+1`，其他 transition 和失败终止为 `0` |

### 7.1 抖动根因与最终修复

本次问题不是单纯的模型动作噪声，也不是 C=20 的动作块长度不足。真机日志和 ROS
频率测量给出了两层证据：

| 测量项 | 修复前 |
| --- | --- |
| 正常策略目标周期 | `66.7 ms`（15 Hz） |
| chunk 边界 `rtc->rtc` 周期 | `120--162 ms` |
| 空闲 ROS 发布 | `200.00 Hz`，最大间隔约 `6 ms` |
| 运动中 ROS 发布 | 约 `187--189 Hz`，周期性最大间隔 `111--115 ms` |
| 三相机 observation | 均值约 `6--7 ms`，最大值约 `17 ms`，不是主要阻塞源 |
| native-JAX 请求 | 约 `140--146 ms`，由 RTC 尾部覆盖 |

每个 20-action chunk 结束时，EnvWorker 需要处理 observation、Torch tensor、Ray
序列化和跨节点通信。其中部分原生调用会长时间占用 EnvWorker 的 Python GIL。
旧实现虽然创建了 RTC 线程和 200 Hz ROS 发布线程，但两个线程仍与 Ray EnvWorker
处于同一个 Python 进程。主线程阻塞期间，两条控制线程会一起饿死，因此出现以下
重复模式：

1. 15 Hz 目标更新漏掉约一个周期；
2. 200 Hz ROS 发布同时出现约 110 ms 空洞；
3. 插值已经走到旧目标，只能短暂保持；
4. 新目标到达后重新追踪，末端先停再动；
5. 末端位移和力矩经机械臂几何结构放大，视觉上表现为末端带动整臂间歇抖动。

因此降低速度或增加加速度限制只能让动作变慢，不能消除控制时钟空洞。最终修复是
在 `realworld_piper_rlt_stage2_ac_mlp.yaml` 中启用：

```yaml
isolated_control_process: true
```

`PiperControlProcess` 使用 `spawn` 创建独立 Python 进程。该进程独占：

- 15 Hz 主动作和 RTC 尾动作的统一截止时间；
- RTC suffix 队列、已执行步数和最后动作；
- arm EMA `alpha=0.4` 与 `0.4 rad/s` 命令空间速度限制；
- 15 Hz 目标之间的线性插补；
- 左右臂 ROS 指令的 200 Hz 发布。

Ray EnvWorker 只负责 observation、推理数据和动作块交接。即使 EnvWorker 在 Ray
序列化期间阻塞，独立进程仍会按时消费已经入队的 RTC 动作并持续发布 ROS 指令。
独立进程上线后的首轮真机观察为“基本不抖”，说明进程内线程饥饿是主要根因。

独立进程必须跟随 EnvWorker 生命周期退出。仅设置 Python
`multiprocessing.Process(daemon=True)` 不能覆盖 Slurm/Ray 强制终止父进程的情况；
父进程被直接杀死时，子进程可能成为仍以 200 Hz 发布最后目标的孤儿。控制进程启动
时通过 Linux `prctl(PR_SET_PDEATHSIG, SIGTERM)` 注册父进程死亡信号，并检查
`getppid()` 消除启动竞态。停止作业后的验收必须确认
`rosnode list | grep rlinf_piper_control_executor` 无旧节点，且进程表中没有
`multiprocessing.spawn ... --multiprocessing-fork` 残留。

### 7.2 动作连续性如何保证

连续性由不同时间尺度上的四层机制共同保证：

1. **动作块内部连续**：每个 chunk 执行 C=20 个目标，策略目标频率固定为 15 Hz；
2. **推理期间连续**：C=20 执行完后，不等待下一次 Ray/JAX 请求，而是由独立控制
   进程继续消费同一 native-JAX H=50 计划的最多 15 个未使用 suffix 动作；
3. **时间对齐连续**：新 chunk 返回后，根据 RTC 实际已经执行的步数（首轮通常为
   6--7 步）丢弃已经过时的前缀，避免重新执行“过去的动作”造成先伸后缩；
4. **边界位置连续**：对齐后的新 chunk 只在最前 5 个臂关节目标上从 RTC 最后动作
   线性融合；夹爪保持离散时机，不参与融合。

200 Hz 层不是额外生成动作，而是在相邻两个 15 Hz 绝对关节目标之间做线性插补。
因此 15 Hz 决定策略轨迹，200 Hz 只负责让 Piper 底层控制器看到密集、等间隔的
参考点。RTC 解决“下一块还没返回”，时间对齐解决“新块从错误时间开始”，5 步融合
解决“两个有效计划在交界处的位置差”，独立进程解决“Ray 阻塞控制时钟”。

### 7.3 四 CAN `h` 回 launch 原点

默认 Stage 2 使用原 OpenPI `start_inference_takeover.launch`。RLinf 只向
`/teleop/policy/joint_left/right` 发布模型动作，由原 mux 仲裁从臂命令；不要同时
运行独立 OpenPI 真机客户端或 `piper_mode_keyboard.py`。`h` 在 idle/HOLD 状态调用
`/teleop/home_request`，四臂返回本次 launch 捕获的启动姿态，到位后继续 HOLD，
回位过程不写 replay。原 OpenPI 源码不修改。

空格调用 `/teleop/hold_request`，保留回合和 B 采集门控；I 恢复时清除旧 chunk，
从新观测重新规划。四 CAN 恢复有界 RTC，推理等待期间继续执行已预测的 VLA 后缀；
HOLD/T 清除后缀并由执行器 ACK，恢复时不会重放旧动作。

为消除原始 prefix 的跨机传输延迟，真机 JAX 服务改用 RLinf 入口：

```bash
cd /home/agilex/FluxVla/openpi/RLinf
bash examples/embodiment/serve_piper_rlt_compact.sh
```

使用 `pi05_connect_strings` 的配置提示词与原生权重，精确动作 prefix 在机器人本地通过
同一冻结 RL token 编码。跨机传 `rlt_z_rl`（8 KiB）而非原始 prefix（3 MiB）；stride
拆成单观测低优先级任务，动作请求优先。原 OpenPI 的模型和四臂接管源码不修改。
T 使用原两秒等待、HOLD ACK 与双主臂 MASTER 验证；确认 ACTIVE 后记录 mux 的
`/master/joint_left/right` 人工动作，带 `intervene_flags`，与自主数据共用
`logs/<run>/replay_buffer/rank_0`。warmup/human 标签可重叠，原采样比例不变。

以下 `r` 回位是仅用于旧 direct-control/mode2 配置的历史流程：

`r` 只在 idle 状态生效。首次按 `a` 开始回合时记录唯一的初始关节位置，后续回位
不会用失败回合的末端位置覆盖它。回位顺序保持为：

1. 双臂保持当前位置，先打开两个夹爪；
2. 以 15 Hz 和原有单步限幅返回首次记录的臂关节位置；
3. 两臂到位后在原位关闭夹爪；
4. `q` 可随时中断并 hold，回位动作不写入 replay。

引入独立控制进程后，回位目标也必须通过子进程的 `reset_target` 通道发送，并明确
关闭 EMA；否则目标只会写入 EnvWorker 内已经停用的旧插值状态，日志会显示开始回位
但机械臂不运动。该兼容路径保留回位单步限幅和 200 Hz 插补，不进入推理用的 15 Hz
RTC 队列。

ROS：

- 状态：`/puppet/joint_left`、`/puppet/joint_right`
- 指令：`/master/joint_left`、`/master/joint_right`
- 相机：`/camera_f/color/image_raw`、`/camera_l/color/image_raw`、
  `/camera_r/color/image_raw`
- 使能：`/enable_flag`

键盘：

| 按键 | 行为 |
| --- | --- |
| `i` | idle 时开始推理；人工接管/HOLD 后恢复推理，同一活动回合保留 B 状态 |
| `b` | 从下一完整 action chunk 开始采集；warmup/actor 就绪门控保留 |
| `t` | B 后请求原四臂人工接管，确认 ACTIVE 后记录人工动作 |
| `c` | 成功，reward `+1`，终止 |
| `x` | 普通失败/放弃，reward `0`，终止 |
| `q` | HOLD，reward `0`，终止；不是硬件急停 |
| 空格 | 原 OpenPI 软件 HOLD，暂停但不终止回合，I 从新观测恢复 |
| `h` | idle/HOLD 时返回本次 launch 捕获的四臂原点，继续 HOLD，不写 replay |

分别验收 `i,b,c`、`i,b,x`、`i,b,t,c`、`i,b,t,x`；人工接管是可选采集方式。

真机安全边界和键盘适配继续保留。当前不使用 residual action；任何
第一槽偏右、第三槽提前插入、双臂方向异常或碰撞风险都必须立即停止。

RTC suffix 必须在当前 chunk 的最后一个动作下发后、同步解码三路相机之前交给独立
控制进程。若等 `chunk_step` 返回后再启动，每 20 个动作会漏掉约一个 15 Hz 控制
周期。不要用降低速度或增加加速度限制掩盖该问题；验收时运动中的
`rostopic hz -w 2000 /master/joint_left` 不应再出现约 100 ms 的周期性发布空洞。

三路相机的 640×480 到 224×224 预处理必须与已验证的 `piper_real` 完全相同，
不得为了性能改成另一套 OpenCV resize。`Piper observation timing` 每 25 个完整观测
报告同步等待、解码均值和总耗时最大值。

### 7.4 2026-07-27：视觉偏差数据清空与干净重采集

第一槽曾出现稳定的向右偏差。已经确认该批 Stage 2 replay 和由其训练得到的
actor/critic checkpoint 不能作为后续训练或恢复的基础。清理范围仅包括服务器
`logs/` 下的以下 Stage 2 目录：

- `*-realworld_piper_rlt_stage2_ac_mlp_dual`；
- `piper_rlt_stage2_resume_replay_*`；
- `piper_rlt_stage2_warmup_*`。

不得删除或替换 Stage 1 RL Token、native-JAX pi05 checkpoint、`piper_real` 或 ROS
配置。新一轮启动时不要设置 `RLINF_REPLAY_LOAD_PATH` 或
`RLINF_STAGE2_RESUME_DIR`，确保 replay 和 Stage 2 actor/critic 均从空状态开始。

新的采集和接管门槛如下：

1. 先仅使用修复后的 native JAX reference 验证四个槽，尤其是第一槽；
2. 从该路径重新采集 action-aligned transition；
3. replay 未达到 `600` 条前，learner 完全不更新；按 `b` 仍记录 transition，
   但执行 JAX reference；
4. replay 达到 600 条后，按论文 UTD `G=5` 开始 learner 更新；
5. 完成 `1000` 次 learner updates 前，`b` 不切换到 actor；
6. 仅当 `rlt/update_step >= 1000` 且最新权重已同步给 rollout 时，下一完整 `b`
   chunk 才可由 full-output actor 接管。

对应配置为：

```yaml
rlt_schedule:
  warmup_min_size: 600
  warmup_post_collect_updates: 1000
replay_buffer:
  min_buffer_size: 600
```

### 7.5 第一槽稳定偏右：具体根因、证据与修复

**现象。** 在同一套 native-JAX pi05 权重、同一相机和同一初始姿态下，旧版
`examples/piper_real` 能正常插入第一槽，而 RLT 的 JAX reference 和按 `b` 接管路径
连续向右偏。该偏差不是随机探索噪声：同一第一槽连续多次复现，且在 actor 尚未参与
控制时已经存在，因此不能通过减速、加速度限制、chunk 融合或继续训练解决。

**根因。** 两条路径送给 JAX 的像素并不相同：

- 已验证的 `piper_real` 在
  `examples/piper_real/env.py::_decode_image` 中将 ROS 图像统一为 BGR，随后调用
  `openpi_client.image_tools.resize_with_pad(..., 224, 224)` 和
  `convert_to_uint8`；
- 旧 RLT `PiperRealWorldEnv._decode` 虽然也读取同一帧，却使用另一套手工解码和
  OpenCV resize 路径。其缩放/填充和边界像素与 `piper_real` 不同；实机同帧比对显示
  约 `29.7%` 的通道值发生变化，平均绝对误差约 `1.51`，最大像素误差为 `96`。

视觉上几乎看不出的像素差异会改变视觉动作模型在精细插入阶段的末端方向，因此形成
稳定的第一槽右偏。这里**不是** JAX 权重转 PyTorch 造成的：
`NativeJaxRLTFeatureModel` 直接请求 native JAX server 的 action 和 RLT prefix；
PyTorch 只运行 Stage 1 token encoder 与 Stage 2 actor/critic。

**代码修复。** `rlinf/envs/realworld/piper/piper_env.py::_decode` 现在先按 ROS
`rgb8`/`bgr8` encoding 统一为 BGR，再调用与 `piper_real` 完全相同的：

```python
image = image_tools.resize_with_pad(image, render_height, render_width)
image = image_tools.convert_to_uint8(image)
```

随后 `NativeJaxRLTFeatureModel._chw_uint8` 只做 HWC→CHW 的内存布局转换，不再缩放、
填充或改变像素数值。`tests/unit_tests/test_piper_direct_control.py` 中的
`test_camera_decode_is_byte_identical_to_validated_piper_real` 对 `rgb8` 和 `bgr8`
两种 ROS 输入均逐字节验证这一等价性。

**数据处置与验收。** 该修复之前采集的第一槽 transition 含有错误视觉条件，且其
Stage 2 actor/critic 已从这些 transition 学到错误状态分布，因此必须按 7.4 清空，
不能合并或恢复。新一轮应先用纯 JAX reference 连续验证四个槽；第一槽不再稳定右偏后
才记录 2000 条干净 replay。若纯 JAX reference 仍右偏，应停止采集并检查相机安装、
相机 topic/serial 映射和机械臂初始姿态，而不是启动 Stage 2 训练。

### 7.6 mode2 主从接管（2026-07-28）

为避免破坏已经验证的真机路径，ROS `mode:=0`、`mode:=1` 的条件和行为保持不变。
统一 launch/node 只新增显式 `mode:=2` 分支；新功能只由
`realworld_piper_rlt_stage2_ac_mlp_mode2.yaml`（单机）或
`realworld_piper_rlt_stage2_ac_mlp_dual_mode2.yaml`（双机）显式启用；普通
Stage 2 配置缺省仍按 mode1 工作。

mode2 使用单一控制权状态机：

1. `POLICY`：JAX/RLT 正常推理；
2. 按 `h`：立即停止 RTC suffix，锁存从臂实测 14 维关节/夹爪位置并持续 hold；
3. 按 `m`：进入受监控的人工慢速接近；从臂持续 hold，操作者移动上电且处于示教/
   重力补偿状态的主臂靠近从臂位姿。公开 SDK 没有主臂任意关节位置接口，因此代码
   不会猜测性地自动驱动主臂；
4. 主从最大关节误差连续 `approach_stable_samples` 帧小于
   `approach_tolerance_rad` 后进入精对齐；随后误差连续 `stable_samples` 帧小于
   `position_tolerance_rad` 才进入 `READY`；
5. 按 `t`：再次检查反馈新鲜度和精对齐误差；从下一控制步起，
   先暂停原 policy/RTC 发布进程，再由独立 bridge 以 200 Hz 把主臂 14 维动作发送
   给从臂。环境只采观测、不发布第二路命令，并通过
   `intervene_action`/`intervene_flag` 把实测主臂动作写入 replay；
6. 按 `c` 以 reward `+1` 成功结束，按 `x` 以 reward `0` 失败结束；reward/done
   立即写入 replay，但 bridge 保持接管，后续回程不写 replay；
7. 人工移动主臂回到首次按 `a` 时记录的起点，从臂同步跟随；到位后按 `r`。系统
   同时验证主臂和从臂（默认旋转关节误差不超过 `0.03 rad`、夹爪误差不超过
   `0.005 m`），验证通过才关闭 bridge、进入 hold 并允许下一次 `a`。`r` 是到位
   确认键，不会自动驱动主臂回原点。

任何反馈超时、NaN、从臂离开 hold、对齐超时或 bridge service 失败都会 fail
closed：不进入 `TELEOP`，从臂继续 hold。`q` 会立即关闭 bridge 并锁定 `FAULT`，
必须检查后重启；`b` 仍保留给原 RLT actor/reference 切换，不承担主从接管功能。

#### 7.6.1 驱动隔离原理

Piper 主臂默认持续广播 `0x151/155/156/157/159`，因此未启动 ROS 时可直接控制
从臂并进行原生主从数据采集。如果它与 policy 同时接在从臂 CAN 上，会和推理命令
冲突。mode2 bridge 只能读取已经离线配置好的
`0x161/165/166/167/169`，不能在四臂连接时动态完成地址切换：

- 原生采集状态：保持 `0x15x`，主臂直接控制从臂；
- 预隔离 mode2：主臂已离线配置为 `0x16x`，原 mode1 从臂驱动独占
  `/master/joint_left/right`，bridge 发布 `/rlinf/mode2/leader_left/right`；
- `T` 后：policy 发布进程先 suspend，bridge 才成为从臂命令的唯一发布者；
- 主臂反馈过期超过 `0.10 s`：bridge 自动关闭 teleop，停止发送新目标；
- mode2 退出不会改硬件地址；恢复 `0x15x` 必须物理断开对应从臂后离线执行。

对外统一使用 `start_ms_piper.launch mode:=2`。mode2 内部复用 mode1 已验证的
从臂位置控制代码，但 mode0/mode1 的判断和行为没有改变。旧
`start_ms_piper_mode2.launch` 仅作为兼容 wrapper，内部转调统一入口。

#### 7.6.2 `0x15x`/`0x16x` 真机结论

固件 `S-V1.8-8` 实测结果：

- `470#0000001000000000` 和 `470#0000000000000000` 都返回
  `0x476#70...`，但地址不发生变化；`linkage_config=0x00` 使配置无效；
- `MasterSlaveConfig(0xFA, 0, 0, 0x10)` 可把主臂持久配置为 `0x16x`；
- `MasterSlaveConfig(0xFA, 0, 0, 0x00)` 可持久恢复 `0x15x`；
- `0x470` 是共享 CAN 广播帧。四臂连接时发送 `0xFA` 可能把从臂也改成主臂，
  因此严禁在线切换。

所以“未启动 ROS 使用原生 `0x15x`，只靠 `roslaunch mode:=2` 自动切到 `0x16x`”
在当前两条主从共用 CAN 的拓扑上不可实现。要实现这一语义，必须让主臂和从臂使用
独立 CAN 接口，或增加可控的物理 CAN/航空链路隔离。

当前安全实现为 fail closed：`mode:=2` 缺省拒绝启动；只有已经按离线流程验证两只
主臂均为 `0x16x` 后，才能显式传入 `mode2_leaders_preisolated:=true`。

`configure_piper_mode2_master_offset.py` 是离线硬件配置/恢复工具。每次操作都必须：
关闭 ROS 和四臂电源、物理断开对应从臂、重新上电后配置主臂、用 `candump` 验证、
再次断电后才重连从臂。

若人工对齐因关节耦合无法稳定收敛，可在工作区无人员/障碍、急停可触达且没有 RLT
policy 发布时，显式调用一次性恢复服务：

```bash
rosservice call /rlinf/mode2/align_followers_to_leaders
rosservice call /rlinf/mode2/alignment_status
rosservice call /rlinf/mode2/cancel_alignment
```

该服务从从臂实测位置开始，以默认 `0.05 rad/s` 的关节速度和 `0.01 m/s` 的夹爪
速度逐步插值到主臂，而不是直接跳到主臂目标。反馈过期立即停止；连续满足恢复阈值
后 `alignment_status` 才返回成功。命令轨迹从启动时的从臂实测位置累积推进；
命令与实测关节差超过 `0.05 rad` 时暂停推进，30 秒未收敛自动停止，也可随时调用
`cancel_alignment`。它只负责退出前的姿态对齐，不会在线恢复 `0x15x`；硬件地址
恢复仍需断开从臂后离线执行。

#### 7.6.3 从原生 `0x15x` 进入 mode2：完整离线流程

当前共享 CAN 接线不能在线切换地址。进入 mode2 必须完成以下全部步骤；任何一步
不符合预期都应停止，不得通过增加速度、强制启动 bridge 或跳过验证来继续。

**步骤 1：原生状态下对齐并停止控制进程**

1. 在原生主从跟随状态，用主臂把两只从臂带到无自碰、远离桌面和障碍物的安全姿态；
2. 确认左右主从姿态接近，之后不要再人工搬动任何一只机械臂；
3. 按正常流程结束当前 episode，停止 RLT、RTC、policy 和所有 Piper ROS 节点；
4. 确认没有进程继续向 `can_left` 或 `can_right` 发送控制帧；
5. 关闭四只机械臂电源。急停必须可触达，工作区内不得有人。

**步骤 2：断电隔离两只从臂**

在四臂均已断电的情况下，物理断开左从臂和右从臂的航空/CAN 插头。两只主臂保持
连接。严禁带电插拔，也严禁在从臂仍连接共享 CAN 时运行下面的 `0x470` 配置命令。

**步骤 3：只给主臂上电并配置 CAN**

保持两只从臂断开、断电，只给左右主臂上电，然后执行：

```bash
cd /home/agilex/cobot_magic/Piper_ros_private-ros-noetic
source /opt/ros/noetic/setup.bash
bash can_config.sh
```

必须确认 `can_left` 和 `can_right` 均成功启用，且 Piper 使用的速率为 `1000000`。

**步骤 4：分别把两只主臂持久配置为 `0x16x`**

```bash
cd /home/agilex/FluxVla/openpi/RLinf

/usr/bin/python3 \
  toolkits/realworld_check/configure_piper_mode2_master_offset.py \
  --can can_left \
  --apply \
  --i-confirm-follower-disconnected

/usr/bin/python3 \
  toolkits/realworld_check/configure_piper_mode2_master_offset.py \
  --can can_right \
  --apply \
  --i-confirm-follower-disconnected
```

两个命令都必须报告主臂已应答 `0x16x (mode 2)` 配置。该应答本身不是最终成功
判据；在第一次 CAN 帧验证前不要断电。

**步骤 5：保持主臂上电，验证实际发送地址**

```bash
timeout 3 candump can_left,160:7F0
timeout 3 candump can_left,150:7F0

timeout 3 candump can_right,160:7F0
timeout 3 candump can_right,150:7F0
```

左右两侧都必须同时满足：

- `160:7F0` 能持续看到 `0x161/165/166/167/169` 一类主臂帧；
- `150:7F0` 不再出现 `0x151/155/156/157/159`。

`timeout` 正常结束时可能返回 `124`，是否成功以输出的 CAN ID 为准。如果任一侧
仍发送 `0x15x`、无 `0x16x`、CAN 接口报错或两侧结果不一致：立即停止，不得重连
从臂，也不得启动 mode2。先排查供电、接口映射和配置应答。

**步骤 6：再次断电并重连从臂**

只有步骤 5 完全通过后：

1. 关闭左右主臂电源，确认四臂全部断电；
2. 重新连接左右从臂航空/CAN 插头；
3. 确认四臂姿态没有被移动，主从仍保持在步骤 1 的近邻安全姿态；
4. 给四只机械臂重新上电。

此时不开 ROS 时主臂不再带动从臂是预期现象：主臂发送 `0x16x`，从臂仍由
`0x15x` 控制目标驱动。

**步骤 7：启动 mode2 ROS**

只有两只主臂已经按步骤 5 验证为 `0x16x` 时，才允许显式确认预隔离状态：

```bash
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
```

`mode2_leaders_preisolated:=true` 是操作者对已完成物理隔离和实帧验证的确认，不会
替操作者切换 CAN 地址。若当前仍是原生 `0x15x` 状态，不得填写该参数。缺少确认时
bridge 会直接报错退出，防止主臂原生帧和 policy 同时控制从臂。

**步骤 8：启动 RLT 前验证 mode2**

```bash
rostopic hz /rlinf/mode2/leader_left
rostopic hz /rlinf/mode2/leader_right
rosservice list | grep /rlinf/mode2
```

两路 leader topic 都必须持续收到新消息，并且能看到
`/rlinf/mode2/align_followers_to_leaders`、`alignment_status` 和
`cancel_alignment`。首次测试先不启动 RLT policy，只验证反馈和从臂 hold。

如果断电重连后主从姿态存在小幅差异，可在确认没有 policy/RTC 发布机械臂命令后
调用一次受保护的低速对齐：

```bash
rosservice call /rlinf/mode2/align_followers_to_leaders
rosservice call /rlinf/mode2/alignment_status
```

需要中止时立即执行：

```bash
rosservice call /rlinf/mode2/cancel_alignment
```

只有 `alignment_status` 明确成功、从臂稳定 hold、两路反馈无超时后，才能加入
Stage 2 Ray 集群。

#### 7.6.4 mode2 的 Stage 2 配置

双机训练服务器需导出：

```bash
export RLINF_PIPER_STAGE2_CONFIG=realworld_piper_rlt_stage2_ac_mlp_dual_mode2
```

真机加入 Ray 前需额外导出：

```bash
export RLINF_PIPER_CONTROL_MODE=2
```

preflight 会额外要求两路主臂 topic 有新消息及三个 `/rlinf/mode2/*` service 存在。
注意：bridge 能验证主臂帧完整、新鲜，并能保证从臂命令单一所有权；主臂松手是否
稳定保持/重力补偿仍由主臂固件示教模式负责。首次测试必须空载、低速、急停可触达，
先单侧小范围验证，不能仅凭 ROS service 成功就假设主臂具备刚性位置保持。

默认近邻阈值是 `approach_tolerance_rad: 0.08`，精对齐阈值是
`position_tolerance_rad: 0.02`。两个阈值都是 12 个旋转关节中最大绝对误差的弧度值，
不是末端笛卡尔距离。近邻阈值必须严格大于精对齐阈值，防止状态机跳过人工微调阶段。

#### 7.6.5 退出 mode2 并恢复原生 `0x15x`

停止 mode2 ROS 并不会恢复原生跟随。恢复时同样必须使用离线流程：

1. 停止 episode、RLT、RTC、policy 和全部 Piper ROS 节点；
2. 关闭四臂电源；
3. 在断电状态下物理断开左右从臂航空/CAN 插头；
4. 保持从臂断开，只给两只主臂上电并运行 `bash can_config.sh`；
5. 分别执行恢复命令：

```bash
cd /home/agilex/FluxVla/openpi/RLinf

/usr/bin/python3 \
  toolkits/realworld_check/configure_piper_mode2_master_offset.py \
  --can can_left \
  --apply \
  --restore-default \
  --i-confirm-follower-disconnected

/usr/bin/python3 \
  toolkits/realworld_check/configure_piper_mode2_master_offset.py \
  --can can_right \
  --apply \
  --restore-default \
  --i-confirm-follower-disconnected
```

保持主臂上电，验证恢复结果：

```bash
timeout 3 candump can_left,150:7F0
timeout 3 candump can_left,160:7F0

timeout 3 candump can_right,150:7F0
timeout 3 candump can_right,160:7F0
```

左右两侧必须都能看到 `0x151/155/156/157/159`，且不再出现
`0x161/165/166/167/169`。只有验证通过后，才能再次关闭主臂电源，在四臂全部断电
时重连两只从臂，然后给四臂上电。最终以“不启动 ROS 时左右主臂分别正常带动对应
从臂”为恢复成功判据。

若恢复验证失败，不得在从臂已连接的情况下重复发送配置帧；保持从臂断开继续排查。

修改前备份：

- `backups/piper_rlt_pre_leader_takeover_20260728.tar.gz`
  （SHA-256 `8139fd3075d17fbf290ee5c3619800de02ef65b4a22b30014b621404d2ecb350`）；
- `backups/piper_ros_pre_leader_takeover_20260728.tar.gz`
  （SHA-256 `4dac1da51f7c7a5779b7fed287a8bff004d72f55e048cdc4770d53d3738980af`）。
- `backups/piper_ros_before_mode2_driver_20260728.tar.gz`
  （最终同步 mode2 前再次备份原 mode0/1，SHA-256
  `967562dd960b57bd12fbe69bf0cc65f934dc2083be0b74df5c46199fc743cff6`）。

---

## 8. 论文对齐修订（2026-09-10）

本节记录一次针对论文（<https://www.pi.website/download/rlt.pdf>）的对齐修订：
改了哪里、为什么改、验证到什么程度。**触发原因是从 LoRA 微调切换到全参微调
pi05**，Stage 1 的 token encoder 必须重训，正好一并把 Stage 1/Stage 2 相对论文
的偏离修掉。

### 8.0 前置：Stage 1 必须重训

本节的所有改动都不能替代这一步。原 `rlt_token.pt` 是针对 **LoRA** checkpoint
`checkpoints/pi05_wire/7_16_pi05_wire/30000` 训练的（见本文档第 1 节）。换成
全参微调模型后 prefix embedding 分布完全改变，冻结的 encoder 失效，Stage 2
拿到的 `z_rl` 没有意义。

据此已删除 `checkpoints/pi05_wire_30000_torch_official_lora_merged_bf16`
（6.8 GB，LoRA 合并后的 torch 转换，专为旧 Stage 1/2 生成，已失效）。
`openpi/checkpoints/` 下的 `pi05_wire`、`pi05_wire_all_data_full` 和
`lerobot_data` 未做改动。

### 8.1 actor 改回论文的高斯分布（`rlt_mlp_policy.py`）

**改了什么。** 原实现是有界残差：

```python
residual_mean = tanh(logits) * reference_delta_scale   # arm ±0.02 rad
action       = reference + clamp(residual, ±scale)
```

当前改为与 B 一致的 full-output Gaussian：reference 只作为输入，Actor 直接输出完整
normalized action mean，**无 additive reference skip、无 zero-init**：

```python
mean = actor_mean(backbone(actor_state))
return Normal(mean, action_std)
```

Actor 子网络同 B：hidden width 256、两层 ``Linear-ReLU-Linear``，全部使用 PyTorch
默认随机初始化；Critic 仍保留 C 的 twin-Q 实现。

**为什么。** 论文用 BC 项的系数 `beta` 做**软锚**，明确要求「critic 有信号后
actor 自然偏离 reference」。硬残差箱把这条路结构性堵死：actor 永远不可能偏离
reference 超过 0.02 rad（1.15°），叠加当时旧配置的 `q_weight: 0.1` 后 RL 的实际权限接近零；
当前 Actor 目标已改为 B 式固定 `10 * BC - min(Q1,Q2)`，
方法退化成论文中作为 baseline 并被击败的 Policy Decorator。

**当前契约。** `reference` 仅参与 MLP 输入；随机初始化 Actor 通过固定
`10 * BC - min(Q1,Q2)` 目标学习 reference/human target。首次切换不再由结构性跳连保证
等于 reference。当前单机真机执行路径另有整段偏差回退、动作平滑与环境最终限位。

上游 RLinf 原版在这里是 `tanh(Normal(...))`，值域被压到 [-1, 1]。Stage 2 用的
是绝对关节角（J3 可到 3.14 rad，夹爪 0–0.08 m），直接用上游版本是错的——
原作者引入残差正是为了解决这一点，动机成立，代价是丢了论文语义。现在的写法
两者兼顾。

Gaussian sigma 对齐 B，所有 normalized action 维度固定为 `0.05`。

删除的配置键（`realworld_piper_rlt_stage2_ac_mlp.yaml`、
`mlp_policy/__init__.py`）：`reference_arm_delta_scale`、
`reference_gripper_delta_scale`。它们只是残差箱的边界，随箱子一起移除。

> **当前单机配置**：保留 B 的 full-output Actor 和归一化 `sigma=0.05`。
> 真机执行均值；动作相对 JAX reference 超过手臂 0.05 rad 或夹爪 0.005 m 时整段回退。
> 合格手臂动作采用 `alpha=0.4` 的平滑与 2 步尾部渐隐，环境限速仍生效。

### 8.2 reference dropout 语义修正（`rlt_mlp_policy.py`）

**当前行为。** full-output Actor 没有跳连；dropout 只需将 actor MLP 输入中的 reference
置零：

```python
if apply_reference_dropout:
    reference = self._maybe_drop_reference(reference, p)
mean = actor_mean(backbone(self._actor_state(reference, obs)))
```

50% reference dropout 继续强制 Actor 在没有 reference 输入时保持独立生成能力。

### 8.3 full-output Actor 的执行语义（`route.py`、`fsdp_rlt_ac_policy_worker.py`）

当前单机配置保留 full-output Actor，`max_actor_reference_*` 钳位关闭。
真机执行 Actor 均值；偏离 JAX reference 超过安全门的整段 chunk 回退为 JAX。
合格动作的手臂残差经 `alpha=0.4` 平滑及 2 步尾部渐隐；
`_apply_execution_transform` 与 route 使用相同变换。RTC 负责时间对齐。

### 8.4 补齐 stride-2 子采样（`piper_env.py`、`algorithms/rlt/subsample.py`）

**为什么。** 论文原文：

> While the RL policy uses an action chunk length of C, we obtain observations
> for each intermediate step... we pick a stride of 2 and save transitions
> corresponding to `<x0, a0:C>`, `<x2, a2:C+2>`, `<x4, a4:C+4>` to the replay
> buffer.

原实现每个 chunk 只产出 1 条 transition。15 Hz、C=20 下即
**0.75 transition/秒**，`warmup_min_size: 600` 光填满 warmup 就要约 13 分钟纯
critical-phase 机器人时间。RLT 的核心卖点就是样本效率，这一项缺失影响最大。
C=20、stride=2 时每 chunk 可产出 10 条，密度提升 10 倍。

**已完成的部分。**

`PiperWireEnvConfig.rlt_subsample_stride`（默认 `0` = 关闭，行为与改动前逐字节
一致）。非零时 `piper_env.step` 在 chunk 内每 N 个控制步解码一次完整观测，而不是
只在 chunk 最后一步解码：

```python
is_subsample_step = stride > 0 and step_index % stride == 0
if is_chunk_final_step or is_subsample_step or self._last_observation is None:
    observation = self._get_observation()
```

> **实时性代价**：每个额外偏移多一次三相机同步解码。原注释
> 「Intermediate camera decodes dominate the control period on the Piper host」
> 就是为此而写。C=20/stride=2 时每 chunk 从 1 次解码变成 11 次。**开启后必须
> 盯 `_log_observation_timing` 打印的 `decode_mean` 与 `total_max`**，若接近
> 15 Hz 的 66.7 ms 周期就要回退 stride 或降低分辨率。

`rlinf/algorithms/rlt/subsample.py` 提供纯函数形式的窗口索引运算：
`stride_offsets`、`slice_stride_actions`、`slice_stride_rewards`、
`subsample_obs_key`。索引算错会静默污染 replay buffer，因此与 worker 管路分开
并单独验证：C=20/stride=2 得到 10 个偏移；动作窗口逐元素精确且正确跨越 chunk
边界；无下一个 chunk 时 `offset>0` 的窗口被**丢弃**而不是静默截断（短 chunk
不是合法的 RLT 行）；稀疏终端奖励只落在真正包含成功步的窗口里。

**尚未接通的部分（重要）。** 以下三处管路尚未接线，因此
`rlt_subsample_stride` 目前保持默认 `0`：

1. `env_worker.env_interact_step` 已经把 `obs_list`（逐步观测）和
   `chunk_actions` 放进 `chunk_step_payload`，但需要把按 stride 取样的观测随
   下一次 rollout 请求送到 rollout worker。
2. `algorithms/rlt/rollout.py` 需要对这批观测调用
   `feature_model.extract_rlt_obs(...)`，并按 `subsample_obs_key(i, key)` 写入
   `forward_inputs`。这与 `_append_rlt_transition_obs` 处理 `final_obs` 是同一
   模式，无需新的消息类型。
3. `fsdp_rlt_ac_policy_worker._transition_replay_trajectories` 需要用
   `subsample.py` 的纯函数，为每个偏移额外产出 replay 行；`next_obs` 取下一个
   chunk 中同一偏移的观测。

这三处跨 env worker / rollout worker / learner 三个进程，本机没有同时具备
`ray` 与 `torch` 的环境（训练在 slurm 集群上），**无法端到端验证**。这里的
错误属于「训练照常跑、学到的是垃圾」的静默失败，因此没有在未经验证的情况下
接线。接通后应先跑一次短 smoke 作业，确认 replay 行数是 chunk 数的
`C/stride` 倍、且抽查行的动作窗口与执行日志一致，再上真机。

### 8.5 清理过期配置（`realworld_piper_rlt_stage1_sft_openpi_pi05.yaml`）

`rlt_prefix_seq_len: 1024` / `rlt_image_only: false` 与实际训练 token encoder
的 `train_piper_rlt_token_native_jax.py`（768 / `image_only=True`）不一致，也与
Stage 2 加载时的硬断言（`native_jax_rlt/__init__.py`）冲突。已改为
`768` / `true`。768 = 3 相机 × 256 patch，语言 token 丢弃（论文脚注 1）。

### 8.6 未做改动的部分

- **夹爪相关参数**：见 8.1。
- **奖励设计**：每个线槽是独立 episode，终端 +1 的信用分配没有问题。
- **critic/actor target（2026-09-29）**：critic bootstrap 使用 C 现有、每次 update
  Polyak 的 EMA target actor 生成完整 next action，再送入 target twin-Q 并取 min。
  online actor 使用固定 ``10 * BC - min(Q1, Q2)``；配置后的 execution transform 为恒等映射。
- **更新频率（2026-09-29）**：外层 learner 与底层 optimizer 都显式执行
  critic:actor=2:1，避免配置、日志和实际 actor step 产生歧义。
- **partial discount（2026-09-29）**：完整 chunk 仍使用 ``gamma ** C``；终止或中断
  的 partial transition 按左对齐 ``action_valid_mask`` 使用
  ``gamma ** valid_mask.sum()``，padding reward 不参与 return。
- **推理端 RTC**：不在本次范围内。
- **`C=20` 与 `gamma`**：C=20 @15 Hz = 1.33 s，是论文 C=10 @50 Hz = 0.2 s 的
  6.7 倍。这是真机控制频率决定的，未改，但重训 Stage 2 时值得连同 `gamma`
  一起复核。

### 8.7 验证状态

| 项 | 验证方式 | 结果 |
|---|---|---|
| full-output actor 无 additive reference skip | 数值 | 通过 |
| normalized sigma 固定为 0.05 | 数值 | 通过 |
| Actor 输出层使用随机初始化 | 数值 | 通过 |
| dropout p=1 时 reference 输入清零 | 数值 | 通过 |
| 重参数化梯度到达 `actor_mean` | 数值 | 通过 |
| 平滑函数改写前后逐元素等价（7 组参数） | 数值，max diff 0 | 通过 |
| 平滑函数可微、夹爪维度不受影响 | 数值 | 通过 |
| stride 窗口索引、跨 chunk 边界、奖励归属 | 数值 | 通过 |
| Stage 2 端到端训练 | — | **未验证**，本机无 ray+torch 环境 |
| 真机 rollout | — | **未验证** |

`tests/unit_tests/test_rlt_mlp_policy.py` 等单测在本机跑不起来（缺 `ray`），
上集群后应先跑一遍。

### 8.8 已被第 9 节修订的内容

- 8.1：actor 不再直接作用于绝对关节角，改为归一化 delta 空间（9.3）；
- 8.3：learner 的执行变换现在也包含钳位，并用于 critic target（9.4、9.6）；
- 8.4：解码步错一位、stride 解码会提前启动 RTC 尾部两个问题已修复；“rollout worker
  对中间观测跑完整 VLA”“无下一个 chunk 时丢弃窗口”“`stride_offsets` 含 0”已被 9.5
  的实现取代；
- 8.6：`C=20`、`gamma=0.96` 已改为 `C=10`、`0.9867`（9.2）；
- 8.7：单测已在机器人的 `.venv-rlt` 中运行（9.11）。

---

## 9. 阶段 0/1 实施（2026-09-11）

范围：按确认的方案 a 最终规划，实施阶段 0（Stage 1）和阶段 1（Stage 2 代码与配置）。
阶段 2 真机实验**未执行**，没有跑端到端 Ray 作业，也没有真机验证（见 9.11）。

修改前备份（`RLinf/backups/`）：

| 备份 | 说明 | SHA-256 |
| --- | --- | --- |
| `piper_rlt_pre_paper_alignment_20260911.tar.gz` | 阶段 0/1 开工时创建（RLinf） | `cc37a6d4fb3603e0096d2aefc53cced37607dc420efc2cca2ea0970364db1402` |
| `openpi_rlt_prefix_pre_paper_alignment_20260911.tar.gz` | 阶段 0/1 开工时创建（openpi） | `56cd9ce5d8e7e0ccf8307740602df15d4927ccec6aa1c96e037565e7d704370f` |
| `piper_rlt_pre_phase01_20260911.tar.gz` | 本轮覆盖前的 28 个既有文件（openpi + RLinf） | `be4877da7b2bf39e766c0efb512348a7e43ea88d66eff455f41d8d91ba8aae7d` |
| `piper_rlt_pre_phase01_preflight_20260911.tar.gz` | 指纹校验与 preflight 第二批覆盖前 | `7c770ed4235d531dcd0fab2814f01db04220d6c205566ac5b58bfea1465f1607` |

### 9.1 Stage 1：全参 pi05 的 RL token

| 改动 | 文件 | 原因 |
| --- | --- | --- |
| `--config-name`，默认 `pi05_wire_all_data_full`；sbatch 读 `RLINF_PI05_JAX_CONFIG` | `toolkits/realworld_check/train_piper_rlt_token_native_jax.py`、`slurm/train_piper_rlt_token_native_jax_8x3090.sbatch` | 原脚本写死 `get_config("pi05_wire")`（LoRA 配置）。pi05 的 prefix 含按配置 norm stats 离散化的 state，配置与 checkpoint 不一致时 token 学到的是错误特征 |
| fp32 主权重，FSDP `MixedPrecision(param=bf16, reduce=fp32)` | 同上 | 原实现把 RLT 模块整体放在 bf16 上跑 AdamW，更新被舍入（实测见下） |
| decoder 同步为方案 b 上游 main 的自回归版本（teacher forcing + 因果 mask，单个 RL token），删除 `rlt_num_rl_tokens` | `rlinf/models/embodiment/modules/rlt_token_transformer.py`、`openpi_action_model.py`、4 个 maniskill YAML | 上游 main 已改为自回归 decoder；本仓库停留在旧 HEAD 的非自回归版本 |
| `Pi0._prefix_forward` / `extract_rlt_prefix`：Stage 1 训练、推理返回 prefix、prefix-only z_rl 共用同一段 prefix 前向 | `src/openpi/models/pi0.py` | 保证训练特征与推理特征出自同一代码路径 |
| checkpoint 写入 `format_version: 2`、`decoder`、`master_dtype`、`config_name`；加载端拒绝 `format_version < 2` | `rlt_token_checkpoint.py`（新）、`native_jax_rlt/__init__.py` | 旧 token 来自 LoRA 模型和 bf16 训练，不能被新 Stage 2 误用 |

bf16 舍入实测（`.venv-rlt`，AdamW lr=2.5e-5、betas=(0.9, 0.95)，20 步）：

| 参数 dtype | 权重被改动的比例 | LayerNorm scale 被改动的比例 |
| --- | --- | --- |
| bfloat16 | 29.1% | 0% |
| float32 | 100% | 100% |

bf16 下 `1.0 + 2.5e-5`、`0.02 + 2.5e-5`、`0.01 + 2.5e-5` 都等于原值：bf16 只有 7 位尾数，
|w|≈0.01 处相邻可表示数的间隔约 6.1e-5，大于两倍学习率。

Stage 1 必须用新脚本重训，旧 token 会被拒绝：

```bash
export RLINF_PI05_JAX_DIR=<全参 checkpoint；机器人上是 openpi/checkpoints/pi05_wire_all_data_full/pi05_wire_all_data_full_20260818/30000>
export RLINF_PI05_JAX_CONFIG=pi05_wire_all_data_full
export RLINF_PIPER_DATASET_DIR=<训练该 pi05 的 LeRobot 数据；9.3 的分位数用的是 openpi/lerobot_data/all_data>
export RLINF_OPENPI_SOURCE_DIR=<openpi 仓库>
export RLINF_STAGE1_OUTPUT_DIR=<输出目录>
sbatch slurm/train_piper_rlt_token_native_jax_8x3090.sbatch
```

### 9.2 控制频率、动作块与折扣

| 项 | 旧值 | 新值 | 说明 |
| --- | --- | --- | --- |
| 控制频率 | 15 Hz | 15 Hz | 不变 |
| `num_action_chunks` / `action_chunk_size` | 20 | 10 | 与 openpi-RLT 一致，0.67 s/块 |
| `gamma` | 0.96 | 0.9867 | 完整块 0.9867^10 ≈ 0.875；partial 块按实际执行步数 |
| `max_steps_per_rollout_epoch` | 20 | 10 | 每次 env/rollout 交换一个完整 chunk |
| `actor_refinement_tail_taper_steps` | 5 | 3 | 随 C 缩放 |
| `rtc_boundary_blend_steps` | 5 | 3 | 随 C 缩放 |
| `rtc_max_bridge_steps` | 5 | 5 | 覆盖推理延迟，与 C 无关；同时是 actor 在 VLA 计划中的起点（9.4） |
| `warmup_min_size` | 600 | 600 | 按确认保持；stride 窗口也计数 |
| `warmup_post_collect_updates` | 20000 | 1000 | 对齐 B 的 actor-start update 门槛 |

C=20 采集的 replay 与新 schema 不兼容，不要通过 `RLINF_REPLAY_LOAD_PATH` 加载。

### 9.3 actor/critic 动作空间：归一化 delta

`rlinf/models/embodiment/mlp_policy/rlt_action_space.py`（新）中的 `RLTDeltaActionSpace`：
12 个手臂关节减去 chunk 起点的 proprio，夹爪保持绝对值，每一维按 `[q01, q99] → [-1, 1]`
线性映射。分位数由 `toolkits/realworld_check/compute_piper_rlt_delta_action_stats.py` 在
`lerobot_data/all_data` 上统计（C=10、每 2 帧一个窗口），完整结果在
`examples/embodiment/config/stats/piper_wire_all_data_rlt_delta_c10.json`，已内联到 YAML 的
`actor.model.action_norm_stats`。

- actor：`mean = MLP(z, p, ã_norm)`，分布为 `N(mean, 0.05²)`；输出层随机初始化，
  不再添加 reference skip；
- reference dropout 作用于 MLP 的 reference 输入（语义同 8.2）；
- critic：`Q(z, p, a_norm)`，next action 按 next_obs 的 proprio 归一化；
- BC 在归一化空间计算；Actor 目标使用固定 `q_weight=1.0`、`bc_weight=10.0`，
  Q 聚合为 `min(Q1,Q2)`；
- replay、route 和机器人仍使用绝对关节角，只有网络内部是归一化的；
- 新指标 `actor/action_ref_norm_abs_mean/max`：归一化单位，1.0 约等于该关节 delta 范围的一半。

**为什么。** 8.1 的高斯 actor 直接输出绝对弧度，各关节 delta 范围差异很大（L_j1 的 q01–q99
跨度 0.098 rad，R_j2 为 0.477 rad），同一个 σ 和 BC 权重对各关节的约束强弱不一，critic 的
输入也被绝对姿态主导。训练数据里 reference 目标超出 ±1 rad 的比例：L_j2 81.0%、R_j2 84.1%、
R_j3 81.3%、R_j5 50.8%。这也是方案 b 的 tanh actor 不能直接用于 Piper 的原因（9.10）。

### 9.4 RTC 时间对齐（本轮发现并修复）

**问题。** 执行侧 RTC 在两个 chunk 之间先执行 e 步旧计划尾部（bridge），新 chunk 返回后按 e
平移，实际执行 `source[e:e+C]`，其中 `source = chunk[:C] ++ plan[C:]`。旧实现中：

1. actor 以 `ref[0:C]` 为输入输出 `a[0:C]`，实际执行的是 `a[e:C] ++ ref[C:C+e]`；
2. replay 存平移后的执行动作，`ref_chunk` 却没有平移；
3. critic 在平移后的动作上训练，actor 目标却在未平移的 `π(x, ref[0:C])` 上查询 Q。

actor 提出的动作与 critic 的训练数据在时间上错开 e 步。用训练数据量化 e=5 时的
`|plan[i+5] − plan[i]|`（σ=0.002 rad；C=20 时数值几乎相同）：

| 关节 | 均值 rad | p95 rad | 均值/σ | p95/σ |
| --- | --- | --- | --- | --- |
| L_j2 | 0.028 | 0.125 | 14 | 62 |
| L_j6 | 0.021 | 0.115 | 10 | 58 |
| R_j2 | 0.039 | 0.140 | 19 | 70 |
| R_j3 | 0.029 | 0.129 | 14 | 65 |
| R_j5 | 0.023 | 0.097 | 11 | 49 |

critic 只见过 reference 附近约 σ 宽的动作，而 actor 的查询点平均偏离 10–20σ，p95 偏离
40–70σ。这些位置的 Q 是外推值，梯度方向没有意义。07-27 版本是 `rtc_max_bridge_steps: 15`，
YAML 注释写明 “RTC shifts the next C=20 actor chunk by about six already-executed VLA bridge
samples”，同样存在这一错位。

**修复。**

- `route.rlt_actor_plan_offset(cfg)`：RTC 开启时等于 `rtc_max_bridge_steps`（5）。控制进程会
  执行完整个 bridge（`drain_rtc`）才切换新 chunk，所以只要有 bridge，偏移就是 5；
- `rollout.predict_rlt_actions`：actor 的 reference 输入为 `ref[5:15]`（`shift_reference_plan`）；
- `RealworldRLTRoute`：把 actor 窗口拼进 VLA 计划，得到
  `rtc_execution_plan = ref[0:5] ++ actor ++ ref[15:]`。EnvWorker 用它做 RTC 对齐，执行的
  正好是 actor 窗口，后续 bridge 仍是未改动的 VLA 尾部。没有 bridge 时（episode 第一块、人工
  接管之后），每个动作也仍落在它被计划的时刻；
- replay：当前观测的 `ref_chunk` 按实际偏移平移，next 观测按预期偏移平移；
- learner：actor 目标和 critic target 的 next action 都经过与机器人相同的钳位、平滑和 taper
  （`bound_and_smooth_actor_refinement`）；
- 指标 `env/rlt_rtc_offset_mismatch`：实际偏移与预期偏移之差，正常应为 0。

### 9.5 stride-2 子采样

**窗口定义（按执行时间线）。** 一个 chunk 在时间轴上是 `[bridge e 步][C 个动作]`，stride
观测在动作 j−1 之后解码（j = 2, 4, 6, 8）。每个窗口使用与部署相同的“观测→动作”延迟
L = e = 5：

```text
观测 x_t → 动作 a[t+5 : t+15]（可跨越下一次 bridge 和下一个 chunk）→ 下一观测 x_{t+15}
reference：t 时刻及之前最后一次选择的 VLA 计划，从 t+5 开始截取，不含未来信息
```

每个被记录的 chunk 产生 1 条 chunk 行 + 最多 4 条 stride 窗口，critical 窗口内的数据密度约为
原来的 5 倍。

- 终止：`RealWorldEnv.chunk_step` 在 info 里给出逐步终止标志 `chunk_step_dones`，auto-reset
  不再抹掉 C/X/Q 发生的具体步。跨过终止步的窗口用最后一个动作补齐（奖励 0、done=True）；
  起点在终止之后的窗口被丢弃；
- 人工接管：窗口内人工驱动的步替换动作和 reference（9.7）；
- 解码：`piper_env.step` 在 `(step_index + 1) % stride == 0` 时解码。8.4 写的
  `step_index % stride` 错了一位，解码的是窗口起点的前一步。mode-2 人工接管的
  `step_external_control` 现在也会解码；
- **RTC 尾部提前启动（8.4 引入的 bug）**：8.4 让 stride 解码也会启动排队中的 RTC 尾部，VLA
  尾部会与当前 chunk 剩余的动作同时执行。现在只有 chunk 最后一步才启动尾部；
- z_rl：机器人本地 JAX server 带 `--rlt-token-checkpoint` 启动后，接受 prefix-only 请求
  （`rlt_prefix_batch`），只跑 prefix 前向和 Stage-1 encoder，只回传 2048 维 z_rl。EnvWorker
  等下一个 chunk 的 reference 请求完成后，才用后台线程提交上一个 chunk 的 4 个观测，不与
  reference 请求争抢 GPU。终止 chunk 之后机械臂保持不动，立即编码；
- 一致性：rollout worker 连接 JAX server 时比对 server 的 encoder 指纹与自己加载的 token，
  不一致直接报错；EnvWorker 能读到 token 文件时再比对一次；preflight 要求 server 已开启
  prefix-only；
- 传输：窗口写入该 chunk 的 `forward_inputs` 固定槽位（`rlt_stride_*`，12 槽 + valid 掩码），
  learner 在 `_transition_replay_trajectories` 中展开为独立 replay 行。replay 行的
  `forward_inputs` 只保留 `record_transition/actor_switch/reference_warmup/rlt_episode_id`，
  窗口行与 chunk 行的 schema 完全一致；
- 指标：`env/rlt_stride_windows`（本步写入的窗口数）、`env/rlt_stride_dropped_windows_total`
  （累计丢弃数）；日志 `RLT stride z_rl timing` 每 25 个 chunk 打印一次 prefix-only 请求耗时。

**实时性。** 每个 chunk 多 4 次三相机同步解码（7.1 实测均值约 6–7 ms、最大约 17 ms，15 Hz
周期为 66.7 ms），外加一次在 chunk 执行期间进行的 prefix-only 请求。上线前检查
`Piper observation timing` 和 `rostopic hz`，确认没有重新出现 7.1 描述的发布空洞。

### 9.6 执行端钳位

- `max_actor_reference_arm_delta: null`：按确认取消以 reference 为中心的手臂钳位；
- `max_actor_reference_gripper_delta: 0.002`、`actor_refinement_smoothing_alpha: 0.4` 不变；
- 兜底仍在 `PiperWireEnv`：关节限位、`max_joint_delta: 0.05` rad/步、
  `max_arm_velocity_rad_s: 0.4`、EMA 0.4；
- learner 的 `_apply_execution_transform` 现在包含夹爪钳位（8.3 只做了平滑和 taper）；
- 修复：两个钳位都为 `null` 时，旧 route 会把平滑和 taper 一起跳过。

### 9.7 人工接管替换 reference

论文 IV-B：人工接管时，replay 中的 VLA reference 被人工动作替换。chunk 行
（`update_rlt_transitions`）和 stride 窗口都只替换人工实际驱动的步，BC 目标随之变为人工动作。

### 9.8 replay 分层采样

`algorithm.replay_buffer.sample_strategy: rlt_stratified`，比例与 openpi-RLT 相同：

| 池 | 份额 | 定义 |
| --- | --- | --- |
| recent_online | 0.4 | actor 控制（`actor_switch`），且属于最近 20 个 actor episode |
| warmup | 0.3 | `reference_warmup` |
| human | 0.2 | 含人工接管步 |
| 其余 | ≥0.1 | 全部 transition 均匀采样；空池的份额并入这里 |

EnvWorker 为每个 episode 分配跨 rank、跨重启唯一的 `rlt_episode_id`。标签写入
`trajectory_index.json`，重载 replay 后仍然有效。

### 9.9 启动命令变化

真机 JAX server（全参 checkpoint + Stage-1 token）。token 需拷贝到机器人上，且必须与服务器
作业 `RLINF_STAGE1_RLT_TOKEN` 是同一个文件：

```bash
cd /home/agilex/FluxVla/openpi

PYTHONPATH=/home/agilex/FluxVla/openpi/src \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
/home/agilex/FluxVla/openpi/RLinf/.venv-rlt/bin/python -u scripts/serve_policy.py \
  --env ALOHA \
  --port 6006 \
  --default-prompt "insert the wire" \
  --return-rlt-prefix \
  --rlt-prefix-seq-len 768 \
  --rlt-token-checkpoint <机器人上的 rlt_token.pt> \
  policy:checkpoint \
  --policy.config pi05_wire_all_data_full \
  --policy.dir /home/agilex/FluxVla/openpi/checkpoints/pi05_wire_all_data_full/pi05_wire_all_data_full_20260818/30000
```

server 日志应出现 `Serving RLT z_rl for prefix-only requests (encoder <指纹>)`，rollout worker
日志中的 `Stage-1 RLT encoder fingerprint <指纹>` 必须与之相同。`policy.py` 默认从
`openpi/RLinf` 按路径加载 `rlt_token_checkpoint.py`（openpi 环境没有 Ray），RLinf 不在该位置时
设置 `RLINF_DIR`。

机器人 Ray 节点上的可选环境变量：

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `RLINF_JAX_PREFIX_HOST` | `127.0.0.1` | EnvWorker 请求 stride z_rl 的 JAX server 地址 |
| `RLINF_STAGE1_RLT_FINGERPRINT` | 空 | 机器人读不到 token 文件时，显式给出期望指纹 |

### 9.10 诊断：Stage 2 actor 接管后机械臂乱动

**记录情况。** 本文档没有这次乱动的直接记录，只有间接痕迹：文件头“residual actor … 不再
使用”、7.3“当前不使用 residual action”、7.4 的“residual actor 接管”门槛，以及 8.1 对有界
残差的描述。直接证据在 `backups/piper_stage2_b_pre_paper_revert_20260727.tar.gz`，即当时基于
方案 b 的实现。

**证据。**

1. actor 头沿用方案 b：备份中的 `rlt_mlp_policy.py` 对整个输出执行
   `action = torch.tanh(action)`，没有 reference 跳连，critic 直接使用这个动作
   （`q_head(critic_state, actions)`）。方案 b 的 tanh 面向 Franka 的归一化末端 delta 动作，
   而 Piper 的动作是绝对关节角；
2. 训练数据中 reference 目标超出 ±1 rad 的比例：L_j2 81.0%、R_j2 84.1%、R_j3 81.3%、
   R_j5 50.8%、L_j3 44.5%、L_j6 23.9%。tanh 输出表示不了这些姿态，BC 项会把 actor 推进
   ±1 的饱和区，梯度随之消失；
3. 方案 b 的真机 route 直接下发 actor 动作，没有钳位。这对应“在方案 b 推理端发现问题”：
   接管瞬间关节目标从 reference（例如 2.0 rad）跳到 tanh 能表示的范围（≤1 rad）；
4. 当时的缓解都在执行端，也就是“在 actor 端加入 JAX pi05 限制运动幅度”：以 JAX reference
   为中心钳位 ±0.02 rad，外加 EMA 和尾部 taper。备份 route 的注释原文是 “This avoids
   turning a saturated, alternating actor residual into joint-target chatter”，YAML 注释是
   “a partially trained actor cannot throw the arm at policy handover”。钳位只能把饱和、正负
   交替的输出压成 ±0.02 rad 的抖动，原因并没有消除；
5. RTC 时间错位（9.4）：当时 `rtc_max_bridge_steps: 15`，注释称每块平移约 6 步。critic 在
   平移后的动作上训练，actor 却在未平移的 reference 上优化 Q；
6. 当时的 warmup 门槛是 400 transition + 2000 次更新；当前为 600 transition + 1000 update。

乱动出现在 update_step 达到接管门槛之后，那正是 actor 第一次控制机器人的时刻。1–3 在接管
瞬间就会表现出来；5 会随着 Q 项起作用逐步加重。

**结论。** 主因是 1–3：tanh 值域与绝对关节角不匹配，且接管时没有连续性保证。RTC 错位（5）
会放大 Q 驱动的漂移。±0.02 rad 钳位和后来的有界残差 actor 都是症状层面的限幅。

**现在的实现如何对应。**

| 原因 | 现在的处理 |
| --- | --- |
| tanh 值域与绝对关节角不匹配 | 去掉 tanh，actor/critic 工作在归一化 delta 空间（9.3） |
| 接管瞬间跳变 | 单机真机执行 Actor 均值；整段偏差门槛回退 JAX，合格动作做平滑与尾部渐隐 |
| 饱和导致 BC 梯度消失 | 没有饱和非线性，BC 在归一化空间直接起作用 |
| RTC 时间错位 | actor 输入、执行、replay、critic 对齐到同一时间窗（9.4） |
| learner 与机器人执行不一致 | learner 和 route 对合格 Actor 动作使用相同平滑与尾部渐隐 |

**仍然存在的风险与监控。**

- 首次在线接管时先检查 `actor/action_ref_norm_abs_max`、
  `actor/action_ref_abs_max`（弧度）、`actor/q_pi` 与 `replay/actor_switch_rate`；
  动作偏差门槛可使 B 继续执行 JAX，而数据仍进入 replay；
- 控制进程额外的 200 Hz 插补和 0.4 rad/s 最终限速仍位于 critic 之外；
- Stage 1 必须先用新脚本重训，旧 token 会被 Stage 2 拒绝。

### 仅使用已采集 replay 从零重训 Stage 2

``examples/embodiment/run_piper_rlt_stage2_replay_only.sh`` 只创建 ActorGroup，不创建
EnvGroup、RolloutGroup、JAX client、ROS、相机或 CAN 资源。它从
``RLINF_REPLAY_LOAD_PATH`` 加载 warmup/online/human 三类标签共存的持久化 replay，使用
全新 Actor/Critic 与当前 B 式 ``10 * BC - min(Q1,Q2)`` 目标默认训练 2000 global step，
可用 ``RLINF_REPLAY_ONLY_MAX_STEPS`` 改为更大的目标（例如 10000），每 50 step 保存一次。源 replay 因 ``auto_save=false`` 保持只读；checkpoint 仍保存完整
replay 快照。该流程禁止设置 ``runner.resume_dir``。

### 9.11 验证状态

| 项 | 方式 | 结果 |
| --- | --- | --- |
| `_prefix_forward` 重构：`sample_actions_with_prefix` 与 `sample_actions` 的动作、与 `extract_rlt_prefix` 的 prefix | 机器人 openpi `.venv`，CPU，dummy Gemma，相同 rng/noise | max\|diff\| = 0.0，mask 一致 |
| prefix-only 批量请求得到的 z_rl 与直接计算一致 | 同上，替身 encoder | max\|diff\| = 0.0 |
| openpi 环境（无 Ray）按路径加载 token 工具 | `_rlt_token_checkpoint_module()` | 通过 |
| bf16 与 fp32 AdamW 更新 | `.venv-rlt` 实测 | 见 9.1 |
| 单元测试：新增 5 个文件 + 9 个相关既有文件 | `.venv-rlt` pytest | 105 passed |
| 5 个 Stage-2 Hydra 变体组合 | `hydra.compose` | 均为 C=10、stride=2、γ=0.9867、手臂钳位 null、plan_offset=5；warmup 600（`dual_warmup500` 为 500） |
| 新 Stage 1 训练 | — | 未执行（需集群） |
| Stage 2 端到端 Ray 作业 | — | 未执行 |
| 真机（阶段 2） | — | 未执行 |
