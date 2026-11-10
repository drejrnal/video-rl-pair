# Copyright (c) 2026 video-rl-pair contributors.
"""Optimize CLS sequence alignment, excluding padded frames from every loss term."""

import json
import logging
from collections.abc import Callable, Iterable, Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Final, final

import torch
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel

from video_rl_pair.checkpoint import Checkpoint, atomic_save, gather_rank_states, restore_rank_state
from video_rl_pair.config import ExperimentConfig
from video_rl_pair.data import PairBatch
from video_rl_pair.distributed import Runtime
from video_rl_pair.losses import AlignmentLoss
from video_rl_pair.model import VideoEncoder

LOGGER: Final = logging.getLogger(__name__)


@final
class Trainer:
    """Own mutable optimizer state; DDP averages independent local pair losses."""

    def __init__(
        self, config: ExperimentConfig, runtime: Runtime, data_provenance: str | None = None
    ) -> None:
        """Initialize the model on its rank, then restore full state when resuming."""
        if config.model.mode == "baseline":
            msg = "baseline has no trainable parameters; use video_rl_pair.evaluate."
            raise ValueError(msg)
        self.runtime = runtime
        self.data_provenance = data_provenance
        self.step = 0
        _ = torch.manual_seed(config.train.seed)
        checkpoint_path = config.train.resume or config.train.init_checkpoint
        checkpoint = None
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if checkpoint["version"] != 2:
                msg = "Expected a DINOv3 checkpoint (version 2)."
                raise ValueError(msg)
            old = ExperimentConfig.model_validate_json(checkpoint["config"])
            if config.train.init_checkpoint is not None and (
                config.model.mode != "stage2" or old.model.mode not in {"stage1", "stage2"}
            ):
                msg = "init_checkpoint is for stage2, from a stage1/stage2 checkpoint."
                raise ValueError(msg)
            if checkpoint.get("data_provenance") != data_provenance:
                msg = "Checkpoint training pairs and file metadata must match this run."
                raise ValueError(msg)
            config = config.model_copy(
                update={
                    "model": config.model.model_copy(
                        update={"backbone_config": old.model.backbone_config}
                    )
                }
            )
        self.encoder = VideoEncoder(config.model, initialize=checkpoint is None).to(runtime.device)
        self.config = config.model_copy(update={"model": self.encoder.resolved_config})
        if checkpoint is not None:
            _ = self.encoder.load_state_dict(checkpoint["model"], strict=True)
        self.model = (
            DistributedDataParallel(
                self.encoder,
                device_ids=[runtime.device.index] if runtime.device.type == "cuda" else None,
                broadcast_buffers=False,
            )
            if runtime.world_size > 1
            else self.encoder
        )
        self.loss = AlignmentLoss(config.loss)
        groups = [
            {
                "params": [p for p in self.encoder.head.parameters() if p.requires_grad],
                "lr": config.train.learning_rate,
            }
        ]
        backbone_parameters = [p for p in self.encoder.backbone.parameters() if p.requires_grad]
        if backbone_parameters:
            groups.append(
                {"params": backbone_parameters, "lr": config.train.backbone_learning_rate}
            )
        self.optimizer = torch.optim.Adam(groups, eps=1e-7)
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.train.precision == "fp16")
        self.dtype = torch.bfloat16 if config.train.precision == "bf16" else torch.float16
        _ = torch.manual_seed(config.train.seed + runtime.rank)
        if config.train.resume is not None:
            self.restore(config.train.resume)

    def fit(
        self,
        batches: Iterable[Tensor | PairBatch],
        *,
        log_metrics: Callable[[Mapping[str, float | int]], None] | None = None,
    ) -> None:
        """Consume accumulate batches per optimizer step until max_steps.

        On resume the iterable must begin at microstep `step * accumulate`.
        Exhaustion is checked collectively to prevent one rank hanging in DDP.
        """
        cfg, runtime = self.config.train, self.runtime
        output = cfg.output_dir
        runtime.require_all(
            cfg.resume is not None or not (output / "last.pt").exists(),
            "Output already contains a checkpoint; resume it or select a new output_dir.",
        )
        if runtime.rank == 0:
            output.mkdir(parents=True, exist_ok=True)
            _ = (output / "config.json").write_text(self.config.model_dump_json(indent=2) + "\n")
            LOGGER.info(
                "world_size=%s global_batch_sequences=%s",
                runtime.world_size,
                cfg.batch_size * cfg.accumulate * runtime.world_size,
            )
        iterator = iter(batches)
        _ = self.model.train()
        while self.step < cfg.max_steps:
            self.optimizer.zero_grad(set_to_none=True)
            metrics = torch.zeros(4, device=runtime.device)
            for microstep in range(cfg.accumulate):
                batch = next(iterator, None)
                runtime.require_all(batch is not None, "Batch source exhausted before max_steps.")
                if batch is None:
                    message = "Batch source is exhausted."
                    raise RuntimeError(message)
                batch, lengths = self.prepare_batch(batch)
                synchronize = microstep == cfg.accumulate - 1
                context = (
                    self.model.no_sync()
                    if isinstance(self.model, DistributedDataParallel) and not synchronize
                    else nullcontext()
                )
                with context:
                    with torch.autocast(
                        device_type=runtime.device.type,
                        dtype=self.dtype,
                        enabled=cfg.precision != "fp32",
                    ):
                        embeddings: Tensor = self.model(batch.to(runtime.device, non_blocking=True))
                    # Keep accumulated costs and cycle probabilities out of reduced precision.
                    terms = self.loss.from_pairs(embeddings.float().unflatten(0, (-1, 2)), lengths)
                    penalty = cfg.l2_weight * self.encoder.regularization()
                    total = terms.total + penalty
                    runtime.require_all(bool(torch.isfinite(total)), "Non-finite training loss.")
                    _ = self.scaler.scale(total / cfg.accumulate).backward()
                    metrics += (
                        torch.stack(
                            [
                                total.detach(),
                                terms.alignment.detach(),
                                terms.cycle.detach(),
                                penalty.detach(),
                            ]
                        )
                        / cfg.accumulate
                    )
            self.scaler.unscale_(self.optimizer)
            norm = nn.utils.clip_grad_norm_(
                self.encoder.parameters(), cfg.grad_clip or float("inf")
            )
            runtime.require_all(bool(torch.isfinite(norm)), "Non-finite training gradients.")
            _ = self.scaler.step(self.optimizer)
            self.scaler.update()
            self.step += 1
            if self.step % cfg.log_every == 0 or self.step == cfg.max_steps:
                values = [float(value) for value in runtime.mean(metrics)]
                if runtime.rank == 0:
                    record: dict[str, float | int] = dict(
                        zip(("total", "alignment", "cycle", "regularization"), values, strict=True)
                    )
                    record["step"] = self.step
                    LOGGER.info(json.dumps(record))
                    with (output / "metrics.jsonl").open("a") as stream:
                        _ = stream.write(json.dumps(record) + "\n")
                    if log_metrics is not None:
                        log_metrics(record)
            if self.step % cfg.save_every == 0 or self.step == cfg.max_steps:
                self.save()

    def prepare_batch(self, batch: Tensor | PairBatch) -> tuple[Tensor, Tensor]:
        """Validate fixed storage shapes and effective lengths before DDP forward."""
        cfg = self.config
        if isinstance(batch, PairBatch):
            clips, lengths = batch
        else:
            clips = batch
            lengths = torch.full((cfg.train.batch_size // 2, 2), cfg.input.positions)
        expected = (
            cfg.train.batch_size // 2,
            2,
            cfg.input.positions,
            3,
            cfg.input.image_size,
            cfg.input.image_size,
        )
        self.runtime.require_all(
            tuple(clips.shape) == expected
            and clips.is_floating_point()
            and cfg.train.batch_size % 2 == 0
            and tuple(lengths.shape) == expected[:2]
            and lengths.dtype == torch.long
            and bool(((lengths >= 2) & (lengths <= cfg.input.positions)).all()),
            f"Expected paired floating clips {expected} and valid lengths [P,2].",
        )
        return clips.flatten(0, 1), lengths

    def save(self) -> None:
        """Retain a step snapshot and refresh last.pt; all ranks must enter."""
        states = gather_rank_states(self.encoder, self.runtime)
        if self.runtime.rank == 0:
            payload: Checkpoint = {
                "version": 2,
                "step": self.step,
                "config": self.config.model_dump_json(),
                "model": self.encoder.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
                "ranks": states,
                "data_provenance": self.data_provenance,
            }
            output = self.config.train.output_dir
            atomic_save(payload, output / f"step_{self.step:06d}.pt")
            atomic_save(payload, output / "last.pt")

    def restore(self, path: Path) -> None:
        """Resume a trusted local checkpoint with the same distributed topology."""
        checkpoint: Checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["version"] != 2 or len(checkpoint["ranks"]) != self.runtime.world_size:
            message = "Checkpoint version and world size must match this run."
            raise ValueError(message)
        old = ExperimentConfig.model_validate_json(checkpoint["config"])
        if checkpoint.get("data_provenance") != self.data_provenance:
            message = (
                "Resume requires the same video pairs and file metadata; "
                "start a new run for a different dataset."
            )
            raise ValueError(message)
        excluded = {
            "max_steps",
            "log_every",
            "save_every",
            "output_dir",
            "resume",
            "init_checkpoint",
        }
        same = (
            old.model.model_dump(exclude={"source"})
            == self.config.model.model_dump(exclude={"source"})
            and old.input == self.config.input
            and old.loss == self.config.loss
            and old.train.model_dump(exclude=excluded)
            == self.config.train.model_dump(exclude=excluded)
        )
        if not same:
            message = "Resume requires the same model, loss, input, and optimizer configuration."
            raise ValueError(message)
        _ = self.encoder.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        self.step = checkpoint["step"]
        if self.step > self.config.train.max_steps:
            msg = "train.max_steps cannot precede the resumed optimizer step."
            raise ValueError(msg)
        state = checkpoint["ranks"][self.runtime.rank]
        if state is None:
            message = "Checkpoint is missing this rank's state."
            raise ValueError(message)
        restore_rank_state(self.encoder, state)
