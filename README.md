# video-rl-pair

用 DINOv3 的逐帧 **CLS 特征**学习同一任务的两段视频之间的对应关系。
训练目标保留 smoothDTW 与 Global Cycle-Consistency（GCC）；下游仅保留
**余弦相似度 → 匹配代价 → 硬 DTW 全局路径 → 匹配帧图片序列**。

要理解实现和各文件的关系，先阅读 [代码链路阅读指南](docs/code_walkthrough.md)。指南包含完整流程图、张量形状、损失计算、阶段切换和逐文件阅读顺序。

## 网络与三种模式

初始化模型为
[`facebook/dinov3-vitb16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)。
每帧独立经过同一个 ViT，读取 768 维 CLS。不同视频共享参数；不融合不同时间的
patch，也不向网络输入时间戳。32 帧对应 32 个向量。

| `model.mode` | 匹配所用特征 | 更新的参数 |
| --- | --- | --- |
| `baseline` | 原始 CLS → L2 归一化，768 维 | 无，仅用于评估 |
| `stage1`（默认） | CLS → Linear(768,512) → GELU → Linear(512,128) → L2 归一化 | 只有投影头 |
| `stage2` | 与 stage1 相同 | 投影头、最后 N 个 ViT block 与最终 LayerNorm |

`model.unfreeze_last_n` 控制 stage2 的 N，默认 2；stage1 忽略此设置。
这里的阶段指参数训练方式，并不改变 CLS 的读取方式。建议先保留 baseline 结果，再训练
stage1；需要更新主干时，从 stage1 检查点开始一个 stage2 新训练。

训练输入为 `[P,2,T,3,H,W]`，其中 P 是视频对数，T 默认 32。编码器接收
`[B,T,3,H,W]`，返回 `[B,T,D]`，训练损失只比较每对内部的两个序列。

## 环境与预训练模型下载

需要 Python 3.11+、uv，以及与训练机器兼容的 CUDA 版 PyTorch。仓库固定了
Transformers 版本；完整依赖以 `pyproject.toml` 和 `uv.lock` 为准。

```bash
cd /home/mxy/video-rl-pair
uv sync --group dev
source .venv/bin/activate
```

该模型需要先在 Hugging Face 模型页面申请并获得访问权限。下载前登录账号：

```bash
hf auth login
bash scripts/download_dinov3.sh
```

默认完整下载到 `checkpoints/dinov3-vitb16-pretrain-lvd1689m`。也可以指定路径和
模型仓库的提交号：

```bash
MODEL_DIR=/data/models/dinov3-vitb16 HF_REVISION=<模型仓库的提交号> \
  bash scripts/download_dinov3.sh
```

`HF_REVISION` 默认 `main`；正式对比实验建议使用同一个提交号。可先用
`bash scripts/download_dinov3.sh --dry-run` 查看下载命令。认证信息由 Hugging Face
登录状态或 `HF_TOKEN` 环境变量提供，不写入配置文件。

**训练不会自动下载权重。** `model.source` 指向包含 `config.json`、权重等文件的本地
模型目录。检查点保存模型结构配置和完整参数，恢复训练或评估时无需重新获取初始化权重。
初始模型、数据集和已有训练结果不随代码删除。

## 数据与采样

数据按任务分目录，可以在同一任务内自动构建所有无序视频对：

```text
dataset/
  hang-cup/
    demo_01.mp4
    demo_02.mp4
  open-drawer/
    demo_01.mp4
    demo_02.mp4
  pick-apple/
    demo_01.mp4
    demo_02.mp4
```

也可以使用 `task/pair_001/{first.mp4,second.mp4}` 显式指定视频对。同一任务不能混用
这两种目录布局。训练与评估任务列表必须分离；配置还会检查解析后的文件路径是否重叠。

- **训练**：固定 32 个输入位置，包含首尾帧；内部按区间随机取不同帧，并保持时间顺序。
  两段视频独立采样。种子由训练种子、epoch 和视频对索引确定。
- **短视频**：使用全部有效帧，重复末帧补齐输入张量，同时记录有效长度。
  **填充位置在计算 softmax、DTW 和循环损失前被移除**，不会增加该阶段的训练权重。
- **评估**：均匀、确定性采样，包含首尾；最多 `evaluation.positions` 帧，不重复补齐。
- **图像**：RGB，缩放到 `input.image_size × input.image_size`，默认 224，使用
  ImageNet 均值 `[0.485,0.456,0.406]`、标准差 `[0.229,0.224,0.225]`。
  正方形缩放不保持宽高比；训练与评估采用相同处理。

每段视频至少两帧。首尾约束要求视频覆盖大致相同的完整动作过程；若含长时间等待或中途
截断，优先裁好动作边界。相同的采样序号不被当作对应标签。

## 默认第一阶段训练

Franka 配置在 `hang-cup` 和 `open-drawer` 上训练，保留 `pick-apple` 供训练结束后检查。
默认数据路径是 `/data/mxy/collection_datasets/franka_robotic_operation_video`。
先用少量步骤检查环境和数据，再调整训练步数：

```bash
GPUS=0 MAX_STEPS=100 OUTPUT_DIR=runs/franka-dinov3-check \
  bash scripts/train_franka_dinov3.sh

GPUS=0,1 MODE=stage1 MAX_STEPS=6000 \
  BATCH_SIZE=2 NUM_WORKERS=2 OUTPUT_DIR=runs/franka-dinov3-stage1 \
  bash scripts/train_franka_dinov3.sh
```

若数据和模型位于其他位置：

```bash
GPUS=0 DATA_ROOT=/data/my-videos MODEL_DIR=/data/models/dinov3-vitb16 \
  OUTPUT_DIR=runs/my-stage1 bash scripts/train_franka_dinov3.sh \
  'data.train_tasks=[task_a,task_b]' 'data.eval_tasks=[task_c]'
```

脚本从任意目录调用都可以；相对路径按仓库根目录解释。先运行
`GPUS=0 bash scripts/train_franka_dinov3.sh --dry-run` 可以查看最终命令。
`bash scripts/train_franka_dinov3.sh --help` 列出全部环境变量。

不使用启动脚本时，相同的配置可以直接执行：

```bash
python -m video_rl_pair.train --config-name franka \
  model.mode=stage1 train.output_dir=runs/franka-stage1-direct
```

主要配置如下。末尾的 Hydra 参数优先于脚本环境变量。

| 配置 | 默认值 | 含义 |
| --- | --- | --- |
| `model.source` | `checkpoints/dinov3-vitb16-pretrain-lvd1689m` | 本地初始化模型目录 |
| `model.mode` | `stage1` | baseline / stage1 / stage2 |
| `model.hidden_dim` / `embedding_dim` | 512 / 128 | 投影头中间维度与输出维度 |
| `model.unfreeze_last_n` | 2 | stage2 解冻的尾部 block 数 |
| `input.positions` / `image_size` | 32 / 224 | 采样位置数与输入边长 |
| `train.batch_size` | 2 | 每张 GPU 的视频数量，必须为偶数 |
| `train.accumulate` | 1 | 每次更新累积的批次数 |
| `train.learning_rate` | `1e-4` | 投影头学习率 |
| `train.backbone_learning_rate` | `1e-5` | stage2 主干学习率 |
| `train.max_steps` | 6000 | 优化器更新次数 |
| `train.l2_weight` | `1e-5` | 投影头线性层权重、偏置的平方 L2 系数 |
| `train.precision` | `fp32` | 可选 bf16 / fp16，需 CUDA 支持 |
| `train.grad_clip` | 1.0 | 梯度范数裁剪上限 |
| `train.log_every` / `save_every` | 100 / 1200 | 按优化器步骤计数的日志、保存间隔 |
| `train.init_checkpoint` | null | 新训练加载的模型参数，例如 stage1 → stage2 |
| `train.resume` | null | 同一阶段恢复模型、优化器、步数及随机状态 |
| `evaluation.enabled` | true | 训练结束后执行 DTW 图片检查 |
| `wandb.mode` | disabled | 可选 offline / online |

有效视频数/更新为 `batch_size × accumulate × GPU 数`，视频对数是其一半。
输出包含 `config.json`、`metrics.jsonl`、按步数保留的 `step_XXXXXX.pt` 和最新的 `last.pt`。
默认训练 6000 步，每 1200 步保存一次，保留 `step_001200.pt`、`step_002400.pt`、
`step_003600.pt`、`step_004800.pt`、`step_006000.pt`；`last.pt` 更新为最新检查点。
最后一步总会保存，即使不是保存间隔的整数倍。每个检查点都可用于恢复训练或评估。
新训练使用新目录，避免覆盖已有检查点。

使用 W&B 时，先 `uv sync --extra wandb`，再登录并设置 `WANDB_MODE=online`；
默认关闭，不要求账号。训练日志记录 total、alignment、cycle、regularization。

## 第二阶段与恢复训练

从第一阶段模型初始化第二阶段，使用新目录：

```bash
GPUS=0,1 MODE=stage2 UNFREEZE_LAST_N=2 \
  INIT_CHECKPOINT=runs/franka-dinov3-stage1/last.pt \
  OUTPUT_DIR=runs/franka-dinov3-stage2 BACKBONE_LR=1e-5 LEARNING_RATE=1e-4 \
  bash scripts/train_franka_dinov3.sh
```

`INIT_CHECKPOINT` 只加载模型参数，新建优化器并从第 0 步开始。
`RESUME` 恢复同一阶段的完整训练状态，不用于切换训练阶段：

```bash
GPUS=0,1 MODE=stage1 RESUME=runs/franka-dinov3-stage1/last.pt \
  OUTPUT_DIR=runs/franka-dinov3-stage1 MAX_STEPS=20000 \
  bash scripts/train_franka_dinov3.sh
```

恢复要求 GPU 数、训练数据、模型和优化器配置一致；输出路径、日志/保存间隔、最大步数
可以改变。数据文件路径、大小、修改时间和配对顺序用于检查数据是否发生变化。

通用单机多 GPU 启动入口：

```bash
CUDA_VISIBLE_DEVICES=0,1 GPUS_PER_NODE=2 \
  bash scripts/train_single_node.sh --config-name franka \
  train.output_dir=runs/franka-multi-gpu
```

多机时，在每台机器设置相同的 `NNODES`、`MASTER_ADDR`、`MASTER_PORT`、
`GPUS_PER_NODE`，并设置不同的 `NODE_RANK`（从 0 开始），然后执行
`bash scripts/train_multi_node.sh --config-name franka train.output_dir=/shared/runs/experiment`。
各节点必须能访问模型、数据与恢复检查点，并采用相同环境；脚本不负责申请机器或 GPU。

## 唯一的下游检查：DTW 匹配帧图片序列

对两段视频得到特征序列 X、Y，计算：

```text
similarity[i,j] = cosine(X[i], Y[j])
cost[i,j]       = 1 - similarity[i,j]
path           = argmin_path sum(cost[i,j])
```

路径从 `(0,0)` 到 `(M-1,N-1)`，每次允许向右、向下或对角前进。
**输出保留完整路径的每个位置**，包括一个源帧对应多个目标帧的情况，不压缩为每帧一次匹配。
每个路径位置输出一张左右并排 PNG，标注采样位置、原始时间戳与匹配分数，按路径顺序编号。
这里使用不可微的硬 DTW；训练使用 smoothDTW，两者不是相同的运算。

先对同一视频对运行未经训练的 CLS 基线：

```bash
python -m video_rl_pair.evaluate \
  model.mode=baseline \
  model.source=checkpoints/dinov3-vitb16-pretrain-lvd1689m \
  evaluation.source=/path/to/first.mp4 \
  evaluation.target=/path/to/second.mp4 \
  evaluation.output_dir=runs/compare-baseline
```

再评估训练后的模型：

```bash
python -m video_rl_pair.evaluate \
  evaluation.checkpoint=runs/franka-dinov3-stage1/last.pt \
  evaluation.source=/path/to/first.mp4 \
  evaluation.target=/path/to/second.mp4 \
  evaluation.output_dir=runs/compare-stage1
```

显式提供两个视频时无需配置数据目录或任务列表。评估会检查它们与检查点记录的训练文件
是否重叠，但任意文件路径本身无法证明它们属于未见过的任务。

也可以按 held-out 任务批量生成结果：

```bash
python -m video_rl_pair.evaluate --config-name franka \
  evaluation.checkpoint=runs/franka-dinov3-stage1/last.pt \
  evaluation.output_dir=runs/franka-stage1-dtw \
  evaluation.max_pairs_per_task=2
```

评估输出包含：

- `report.json`：视频路径、采样时间戳、完整相似度矩阵、代价矩阵、DTW 路径与路径代价。
- `index.html`：本地图片浏览入口。
- 每对视频的图片子目录：按 DTW 路径顺序排列的匹配帧 PNG 图片序列。

`evaluation.positions=32` 控制采样密度，可增大；`chunk_size=8` 控制每次编码多少帧；
`image_size=512` 控制报告中的图片面板大小，与网络输入分辨率无关。
输出目录必须是新的，避免不同实验结果混在一起。对比 baseline、stage1、stage2 时保持
视频对、采样数量和预处理相同。

**这是定性对齐检查，不是语义准确率。** DTW 自带时间顺序与首尾约束；特征坍缩也可能
生成看似平滑的路径。应逐对查看同一时刻的动作阶段是否对应，不用较低路径代价代替语义
正确性；不同编码器的相似度尺度也可能不同。需要准确率时，另行添加独立的事件/阶段标注。

## 损失定义与代码入口

对于 L2 归一化后的 X、Y，先构造行归一化的对比匹配代价：

```text
C[i,j] = -log softmax_j(cosine(X[i],Y[j]) / beta)
R[i,j] = C[i,j] + smoothMin(R[i-1,j-1], R[i-1,j], R[i,j-1])
smoothMin(r) = sum(softmax(-r/gamma) * r)
```

此处 smoothMin 是加权平均，不是 log-sum-exp 版本。
反向匹配需重新计算 `R_yx`，不能用 `R_xy.T` 替代。

```text
p_xy = softmax(-R_xy / alpha)
p_yx = softmax(-R_yx / alpha)
cycle     = -sum_i log(sum_j p_xy[i,j] * p_yx[j,i])
alignment = R_xy[-1,-1] + R_yx[-1,-1]
loss      = cycle + 0.1 * alignment + l2_weight * head_squared_L2
```

默认 `beta=gamma=0.1`、`alpha=1`。时间维求和、视频对之间平均；增加采样长度会改变损失
尺度，不能直接比较不同长度的原始 loss。数学定义保持独立于 encoder。

| 文件 | 内容 |
| --- | --- |
| `src/video_rl_pair/model.py` | CLS 编码、投影头与冻结策略 |
| `src/video_rl_pair/losses.py` | 可微匹配代价、smoothDTW、GCC |
| `src/video_rl_pair/video.py` / `data.py` | 解码、采样、填充长度与视频对加载 |
| `src/video_rl_pair/trainer.py` | 参数更新、阶段初始化与恢复 |
| `src/video_rl_pair/matching.py` | 余弦代价与完整 DTW 路径 |
| `src/video_rl_pair/evaluation.py` / `visual_report.py` | 图片序列与报告 |
| `src/video_rl_pair/conf/` | Hydra 配置 |

```bash
pytest -q
python -m video_rl_pair.train --config-name franka --cfg job
```

测试使用小型本地模型和合成视频检查梯度、采样、恢复与 DTW 输出，不下载正式预训练权重。
旧架构检查点不兼容本次网络；已有 `runs/` 与 `checkpoints/` 文件保留。
方法依据：[Hadji et al., CVPR 2021](https://arxiv.org/abs/2105.05217)。
预训练模型依据：[DINOv3 官方仓库](https://github.com/facebookresearch/dinov3)。
