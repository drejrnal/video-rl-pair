# video-rl-pair

A small PyTorch implementation for studying **Representation Learning via Global
Temporal Alignment and Cycle-Consistency** (Hadji, Derpanis, Jepson, CVPR 2021).
Hydra controls the architecture, objective, training, and evaluation. MP4 pairs can
now run through a task-folder dataset, deterministic sampling, PyTorch DataLoader,
optimization, checkpointing, and held-out matching evaluation. A prepared-tensor
iterable remains available for experiments and debugging.

The default encoder starts from scratch. Optional checkpoint initialization is
configured independently of training resume. No weights are bundled or downloaded
automatically, and this package does not import anything from `VideoAlignment`.

## Reading order

| File | What to understand |
| --- | --- |
| [losses.py](src/video_rl_pair/losses.py) | Matching cost, SmoothMin, dynamic programming, global cycle loss |
| [model.py](src/video_rl_pair/model.py) | How context frames become one normalized embedding per moment |
| [backbone.py](src/video_rl_pair/backbone.py) | Preactivation ResNet50-v2, ending at Conv4c |
| [trainer.py](src/video_rl_pair/trainer.py) | Forward pass, pair construction, backpropagation, optimizer update |
| [conf/](src/video_rl_pair/conf/) | Paper dimensions and coefficients; smaller synthetic demo |
| [weights.py](src/video_rl_pair/weights.py) | Optional initialization from compatible local checkpoints |
| [distributed.py](src/video_rl_pair/distributed.py) | Process setup and reduction across GPUs |
| [checkpoint.py](src/video_rl_pair/checkpoint.py) | Resume state, including per-rank RNG and normalization buffers |
| [data.py](src/video_rl_pair/data.py), [video.py](src/video_rl_pair/video.py) | Task splits, MP4 decoding, context sampling, paired distributed batches |
| [matching.py](src/video_rl_pair/matching.py) | Cosine nearest neighbors, hard DTW, timestamp baseline |
| [evaluation.py](src/video_rl_pair/evaluation.py) | Frozen inference, pair diagnostics, equal task averages |

## Run the computation

Python 3.11+ is required. Install a CUDA-compatible PyTorch/torchvision pair on GPU
machines. These commands create an isolated environment and install the package:

```bash
uv sync --group dev
source .venv/bin/activate
python -m video_rl_pair.train --help
python -m video_rl_pair.train --cfg job
python -m video_rl_pair.train --config-name demo
pytest -q
```

The demo runs two CPU optimizer steps using random tensors, with the actual ResNet-v2
backbone and a smaller head. It checks the computation; it is not meaningful video
representation training. It writes `config.json`, `metrics.jsonl`, and `last.pt` to
`runs/demo`. Use a different `train.output_dir` for each new run.

The default configuration deliberately refuses to train without a batch source.
Use `--config-name pairs` for the complete MP4 workflow described below.
The base configuration uses 20 moments, 2 context frames, 224-pixel images,
128-dimensional embeddings, and 4 same-action videos per local batch. The MP4
preset uses 2 videos (one explicit pair) per local batch.

## Follow one pair through the loss

For embeddings `X: [M,D]` and `Y: [N,D]`:

1. `matching_cost`: normalize each embedding, compute cosine similarities, and
   apply **row-wise** negative log-softmax with temperature `beta`. A row compares
   one source moment against all destination moments.
2. `smooth_dtw`: fill the accumulated-cost table. Each cell adds its matching cost
   to the SmoothMin of diagonal, upper, and left predecessors. The path begins at
   the first moments and ends at the last moments.
3. `smooth_min`: compute `sum(softmax(-r/gamma) * r)`. This is the paper's weighted
   average, **not** the log-sum-exp soft-DTW approximation. Autograd differentiates
   through both the values and the weights.
4. Repeat the cost and dynamic program in the reverse direction. Because the
   matching probabilities are conditional, the reverse cost is recomputed;
   transposing the forward accumulated table is not equivalent.
5. `global_cycle_loss`: form row-wise `log_softmax(-R/alpha)` in each direction.
   The diagonal round-trip probability is `sum_j p(Y_j|X_i) p(X_i|Y_j)`.
   Compute its log with `logsumexp` and sum the negative logs over source moments.
6. `AlignmentLoss`: combine the cycle loss with both final DTW costs:

```text
alignment = R_xy[-1,-1] + R_yx[-1,-1]
cycle     = -sum_i log(sum_j p(Y_j | X_i) p(X_i | Y_j))
total     = cycle_weight * cycle + alignment_weight * alignment
```

Defaults: `beta=0.1`, `gamma=0.1`, `alpha=1`, `cycle_weight=1`,
`alignment_weight=0.1`. Losses are **summed over time and averaged over pairs**.
Changing sequence length therefore changes the loss scale. The cycle is
`X -> Y -> X`, as in the supplied text; only the alignment term is bidirectional.

The prepared-tensor interface uses video 0 as the anchor against videos 1 through B-1,
with every video in that batch belonging to one action category. The MP4 interface
instead supplies explicit `[P,2,T,K,3,H,W]` pairs. `AlignmentLoss.from_pairs` compares
only the two demonstrations within each pair; pairs from different tasks may share
a minibatch without being compared. There is no pairing across distributed ranks.

The dynamic program uses readable nested loops and ordinary autograd. It supports
unequal pair lengths, but training expects the fixed number of moments in the
configuration. No fused CUDA extension or padding/masking convention is hidden here.

## Follow the encoder's tensor shapes

| Operation | Shape with the paper configuration |
| --- | --- |
| Prepared clips | `[B,20,2,3,224,224]` |
| Flatten frames into the 2D backbone | `[B*20*2,3,224,224]` |
| Conv4c features | `[B*20*2,1024,14,14]` |
| Regroup one context per sampled moment | `[B*20,1024,2,14,14]` |
| Two 3D convolutions, each BN + ReLU | `[B*20,512,2,14,14]` |
| Global max pool over context and space | `[B*20,512]` |
| Two 512-unit FC layers + ReLU, dropout before each | `[B*20,512]` |
| Projection and L2 normalization | `[B,20,128]` |

Pooling never merges the 20 represented moments. For a moment at frame `t`, the
two context frames are `[max(t-15,0), t]`. Use sorted moments and past context only.
The MP4 loader constructs these tensors using `context_stride`. Custom tensor
sources must follow the same contract.
For the paper backbone, RGB floats are scaled to `[-1,1]`. Apply the preprocessing
required by a different pretrained backbone when selecting one.

Conv4c means the third block of the fourth convolutional stage. The v2 backbone
uses preactivation bottlenecks, with downsampling at the ends of preceding stages.
The ResNet50-v1 alternative uses torchvision's stage-three output and is an
explicit architectural alternative, not a paper-equivalent backbone.

## Optional initialization

The checkpoint entry exists by default and is disabled until a path is supplied:

```yaml
model:
  ckpt:
    path: null
    scope: backbone
    key: null
    prefix: null
    exclude_prefixes: []
```

For a checkpoint compatible with the configured backbone, only change its path:

```bash
python -m video_rl_pair.train --config-name demo model.ckpt.path=/path/to/backbone.pt
```

Supported formats are PyTorch tensor state dictionaries (`.pt`, `.pth`, `.ckpt`)
and `.safetensors`. The loader recognizes bare dictionaries and the common
`state_dict`, `model_state_dict`, `model`, `encoder`, and `backbone` wrappers.
It removes a uniform DDP `module.` prefix, and extracts a `backbone.` prefix when
loading a backbone from a complete encoder. Other layouts can be selected explicitly:

```bash
python -m video_rl_pair.train --config-name demo \
  model.ckpt.path=/path/to/custom.pt \
  model.ckpt.key=network.ema model.ckpt.prefix=visual.
```

For an official torchvision ResNet50 checkpoint, select its matching preset:

```bash
python -m video_rl_pair.train --config-name demo model=resnet50 \
  model.ckpt.path=/path/to/resnet50.pth
```

That preset explicitly excludes the unused `layer4` and `fc` keys. For real inputs,
use the selected checkpoint's ImageNet preprocessing. For complete encoder weights,
set `model.ckpt.scope=encoder` and match the head dimensions too. The default trains
all parameters; `model.train_backbone=only_bn` or `frozen` requires initialization.

A path cannot reconcile different architectures, tensor shapes, or arbitrary
foreign formats. The loader checks every retained parameter and fails with the
original incompatibility details. Keras/TensorFlow HDF5 files require a separate
conversion before use. Adding another architecture means adding a matching
backbone preset; compatible weights then reuse this loading entry point.

## Training procedure and custom data interface

The loop uses Adam at `1e-4` with epsilon `1e-7`, constant learning rate, and an
optional `1e-5` squared-L2 penalty on the head's convolution/linear kernels and biases.
BN parameters and backbone weights are excluded from that penalty. The mathematical
loss is available separately without regularization. FP32 is the default;
`train.precision=bf16` or `fp16` enables CUDA autocast for the encoder while retaining
FP32 alignment costs and cycle probabilities. FP16 uses gradient scaling.

Supply a callable returning an iterable with this signature:

```python
def make_batches(settings, runtime, start_step):
    # Return an iterable of [B,T,K,3,H,W] floating tensors.
    # On resume start at microstep start_step * settings.train.accumulate.
    # Use runtime.rank/world_size to divide data across workers.
    ...
```

Connect it using Hydra's callable target:

```bash
python -m video_rl_pair.train \
  'batch_source={_target_:my_project.make_batches}' \
  train.output_dir=runs/experiment
```

Every rank must yield the same number of fixed-size batches. The factory owns
video decoding, sampling, augmentation, normalization, action grouping, and data
cursor state when using this custom interface. The built-in MP4 path handles
decoding, context sampling, normalization, pair grouping and deterministic resume.
It currently uses square resizing without random spatial augmentation. The trainer checks
tensor shape, finite loss/gradients, and synchronized source exhaustion.

## Multiple GPUs and nodes

With the environment activated, an eight-GPU synthetic computation check is:

```bash
GPUS_PER_NODE=8 bash scripts/train_single_node.sh --config-name demo \
  train.device=cuda train.output_dir=runs/gpu-demo
```

For actual MP4 training, use `--config-name pairs` and the data overrides below.
Custom tensor factories remain supported by the default configuration.
The launcher uses one DDP process per GPU. The effective global sequence batch is:

```text
train.batch_size * train.accumulate * total_number_of_GPUs
```

For the MP4 preset, `batch_size=2` means one pair per GPU: eight GPUs give global
batch 16 videos and eight pairs per microstep. `batch_size` must be even; for example,
4 videos means two explicit pairs per GPU. The anchor-batch interface instead gives
B-1 pairs per local batch. DDP averages gradients; accumulation divides each local loss by the
accumulation count. These larger-batch runs are not reproductions of batch-4 training.
CUDA DDP uses synchronized BN by default. Accumulation does not combine BN
statistics across microsteps; CPU DDP uses local BN.

On each node of a multi-node run, set `NNODES`, a unique zero-based `NODE_RANK`,
the rank-zero node's `MASTER_ADDR`, a reachable `MASTER_PORT`, and `GPUS_PER_NODE`:

```bash
bash scripts/train_multi_node.sh \
  'batch_source={_target_:my_project.make_batches}' \
  train.output_dir=/shared/runs/experiment
```

Use matching code, environments, and configurations on every node. Rank zero writes
metrics and checkpoints; the resume checkpoint must be readable on every node.
Launch scripts do not allocate machines or choose network interfaces for a cluster.

To resume this implementation's checkpoint:

```bash
python -m video_rl_pair.train --config-name demo \
  train.resume=runs/demo/last.pt train.max_steps=4
```

Resume restores model, optimizer, scaler, step, per-rank torch RNG, and BN buffers.
It requires the same world size and training configuration, except checkpoint
initialization paths, output path, logging/saving intervals, and maximum step.
The MP4 loader reconstructs its epoch and position from the saved optimizer step;
it seeds sampling by epoch and pair identity without consuming model RNG. Whole
pairs are sharded across ranks, and incomplete rank batches are dropped. Resume
rejects changed training pair order or file size/modification metadata. Worker count
may change. A custom batch source must restore its own cursor/RNG. Exact GPU reproducibility
also depends on deterministic kernels. A new run never overwrites an existing
`last.pt` unless explicitly resumed.

## MP4 pairs: train and evaluate

Place demonstrations in task folders. A task can use either layout:

```text
dataset/
  task_01/                 # all unordered pairs of these demonstrations
    demo_a.mp4
    demo_b.mp4
    demo_c.mp4
  task_02/                 # explicit pairs, exactly two MP4s per pair folder
    pair_001/
      first.mp4
      second.mp4
  task_03/
    demo_a.mp4
    demo_b.mp4
```

Mixed layouts within a task are rejected. Each pair must contain distinct files.
Declare your 43 training task names and seven held-out task names explicitly in
`conf/pairs.yaml` or on the command line. Counts are configurable; this smaller
example uses two training tasks and one held-out task:

```bash
uv sync --group dev
uv run python -m video_rl_pair.train --config-name pairs \
  data.root=/path/to/dataset \
  'data.train_tasks=[task_01,task_02]' \
  'data.eval_tasks=[task_03]' \
  train.output_dir=runs/mp4-run
```

This command trains, saves `runs/mp4-run/last.pt`, then evaluates that checkpoint
on pairs **within each held-out task**. Under torchrun, only global rank zero
evaluates after training. Use the existing node launch scripts with the same
`--config-name pairs` and data overrides. Set `evaluation.enabled=false` to defer
evaluation, or lower `train.max_steps` for a computation check. A real run requires
enough pairs for one complete batch on every rank.

MP4s are decoded with PyAV. Training samples sorted, stratified random moments;
evaluation samples uniformly and deterministically. Context uses past frame indices,
with the earliest frame repeated near the beginning. Short training videos may
repeat sampled moments to keep T fixed; evaluation keeps unique moments and uses
at most `evaluation.positions` frames. Each video must have at least two frames
and valid increasing timestamps. `context_stride` is measured in decoded frames,
so its duration changes with frame rate. Square resizing does not preserve aspect
ratio. Normalize RGB to `[-1,1]` for the paper backbone or ImageNet mean/std for the
torchvision preset; the loader selects this from the model configuration.

There are no phase labels or event annotations. Train/eval task lists must be
disjoint. Resolved file paths are checked for overlap, including symlinks; duplicated
video contents stored as different files cannot be detected by this check.

To evaluate a saved training checkpoint separately:

```bash
uv run python -m video_rl_pair.evaluate \
  data.root=/path/to/dataset \
  'data.train_tasks=[task_01,task_02]' 'data.eval_tasks=[task_03]' \
  evaluation.checkpoint=runs/mp4-run/last.pt \
  evaluation.output_dir=runs/mp4-run/recheck
```

Architecture, image size, and context are restored from the checkpoint. The encoder
runs in eval mode with gradients disabled; `evaluation.chunk_size` bounds inference
memory. `evaluation.positions` controls the number of sampled evaluation moments.
No annotation or test score is used to choose the training checkpoint: automatic
evaluation uses the final checkpoint, not the best score on the held-out tasks.

To compare against initialization, run the same evaluation command with
`evaluation.checkpoint=null evaluation.allow_initialization=true` and a fresh output
directory. Set the same model dimensions/initialization path used before training;
the timestamp-only baseline is always included automatically. Use the same task
lists, evaluation seed, positions, and pair cap for comparisons.

### Reading the evaluation output

- `report.json`: selected pairs, times, directional matches, per-task diagnostics,
  equal-weight task macro averages, and within-video embedding spread.
- `index.html`: self-contained cosine-similarity heatmaps and matched video frames
  with timestamps. Open locally in a browser. By default it visualizes two pairs
  per task and evaluates up to 20 pairs per task, selected with a fixed seed.
- `progression.html`: interactive round-trip and progression inspection for the
  same visualized pairs, with all available matching methods selectable.

Every selected pair is evaluated in both directions. Nearest-neighbor matching
tests the embeddings directly. Hard DTW minimizes the sum of `1 - cosine` along
a monotonic endpoint-constrained path; it is distinct from training SmoothMin.
For a query moment with multiple path matches, select the highest-cosine candidate,
breaking ties by earliest target. The time baseline matches normalized timestamps.

| Diagnostic | Meaning and limitation |
| --- | --- |
| `mean_cosine` | Similarity at selected matches; collapsed embeddings can score highly |
| `tau_b` | Temporal ordering of matches, with ties handled; null if all targets tie |
| `unique_target_fraction` | Fraction of target moments selected at least once |
| `cycle_time_error` | Mean return-to-source timestamp error divided by source duration; lower is more self-consistent |
| `embedding_spread` | Mean squared distance of normalized embeddings from their centroid; zero means complete collapse |

**These are diagnostics, not semantic matching accuracy.** A timestamp-only method
or DTW can have excellent ordering without matching the correct event. Inspect the
frame pairs, compare initialization and timestamp baselines, and add annotated
events or phases later if semantic accuracy is needed. Within each task, all directed
pair results have equal weight; macro averages give each task equal weight. Undefined
tau values are excluded with their coverage count reported. Do not select hyperparameters
or checkpoints using the seven held-out tasks.

The checkpoint records training task IDs, resolved pair paths, file sizes and
modification times. Evaluation checks held-out tasks against this provenance when
present. Older checkpoints lack it and report `training_provenance_checked=false`.
Use a fresh evaluation output directory to avoid overwriting previous comparisons.

### Visually inspect a round trip and upcoming progression

Run the usual evaluation command with a trained checkpoint and a fresh output
directory. The viewer is generated automatically; no web server is required.
For a denser sampling grid and a longer initial chunk, add these overrides:

```bash
uv run python -m video_rl_pair.evaluate \
  data.root=/path/to/dataset \
  'data.train_tasks=[task_01,task_02]' 'data.eval_tasks=[task_03]' \
  evaluation.checkpoint=runs/mp4-run/last.pt \
  evaluation.output_dir=runs/mp4-run/progression-check \
  evaluation.positions=80 evaluation.progression_horizon=8
```

Open `runs/mp4-run/progression-check/progression.html`. Choose a pair, the source
robot video, and a matching method. Select the query moment `t`, then step or play
the chunk. The three panels show the fixed query `R[t]`, the intermediate
demonstration `D[q+i]`, and the retrieved robot moment `R[reverse[q+i]]`, where
`q = forward[t]`. The offset is applied on the demonstration timeline, inside the
reverse lookup. Switching methods keeps the same source query. Swapping direction
lets either recording act as the source robot video.

The reverse indices come from matching in the opposite direction, not from
inverting the forward assignments. The starting return `t_0 = reverse[q]` and its
timestamp error remain visible while the chunk advances. Inspect the intermediate
demonstration as well as the returned robot: a perfect round trip can still connect
different task stages. For later steps, compare `D[q+i]` with `R[t_i]`; they should
show corresponding stages, not necessarily resemble the fixed starting frame.

All indices refer to the **sampled evaluation grid**, not consecutive original
MP4 frames. Original timestamps in seconds are displayed. `evaluation.positions`
controls the grid density; the sampling interval may differ between videos.
Playback advances sampled moments for inspection, not at the recording's native
frame rate. The chunk shortens at the demonstration's end instead of padding with
repeated final frames. Nearest-neighbor matches may jump backward or repeat robot
moments; the viewer preserves these results so failures stay visible.

Nearest neighbor exposes embedding similarity, DTW adds temporal ordering, and
normalized time supplies an encoder-free reference. To assess training's benefit,
compare against the initialization evaluation described above using the same pairs
and sampling settings. These are retrieved recorded images, not generated future
observations. The report embeds JPEGs up to 256 pixels per side, bounded by the
encoder's input resolution; it contains no annotations or semantic accuracy score.
`evaluation.visual_pairs_per_task` controls how many pairs are available in the
viewer; setting it to zero produces an empty viewer while numerical evaluation
continues.

## Relation to the paper and reference

This implementation follows the equations in the supplied text. In contrast, the
author's release normalizes along axis 0 and feeds cycle probabilities to a
logits-based cross-entropy with label smoothing and a temporal mean. Here source
rows normalize over destinations, the cycle is negative log return probability,
and time is summed. These choices are explicit; numerical identity with the released
training loss or its historical scores is not claimed.

The paper's pretrained only-BN experiments require appropriate pretrained weights.
Scratch initialization here is a separate training setting. PyTorch initialization,
BN running-variance updates, larger batches, and mixed precision can also affect
results. No dataset results or full GPU training results are bundled.

References: [author's repository](https://github.com/hadjisma/VideoAlignment),
[ResNet-v2 architecture source](https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet_common.py),
[Hydra configuration](https://hydra.cc/docs/intro/),
[PyTorch distributed launch](https://docs.pytorch.org/docs/stable/elastic/run.html).
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution and
[VALIDATION.md](VALIDATION.md) for observed checks and remaining verification.
