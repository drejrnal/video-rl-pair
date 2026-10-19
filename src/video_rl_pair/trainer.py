# Copyright (c) 2026 video-rl-pair contributors.
"""Optimization for anchor batches [B,T,K,C,H,W] or explicit MP4 pairs.

Paired batches [P,2,T,K,C,H,W] compare only the two videos within each pair.
The unpaired interface retains the original same-task anchor pairing.
"""

import json
import logging
from collections.abc import Iterable
from contextlib import nullcontext
from pathlib import Path
from typing import Final, final

import torch
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel

from video_rl_pair.checkpoint import Checkpoint, atomic_save, gather_rank_states, restore_rank_state
from video_rl_pair.config import ExperimentConfig
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
        self.config = config
        self.runtime = runtime
        self.data_provenance = data_provenance
        self.step = 0
        _ = torch.manual_seed(config.train.seed)
        if (
            config.train.resume is None
            and config.model.train_backbone != "all"
            and config.model.ckpt.path is None
        ):
            message = "only_bn/frozen requires model.ckpt.path; use all for scratch training."
            raise ValueError(message)
        self.encoder = VideoEncoder(config.model, initialize=config.train.resume is None).to(
            runtime.device
        )
        if config.train.sync_batchnorm and runtime.world_size > 1 and runtime.device.type == "cuda":
            _ = nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)
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
        self.optimizer = torch.optim.Adam(
            [p for p in self.encoder.parameters() if p.requires_grad],
            lr=config.train.learning_rate,
            eps=1e-7,
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.train.precision == "fp16")
        self.dtype = torch.bfloat16 if config.train.precision == "bf16" else torch.float16
        _ = torch.manual_seed(config.train.seed + runtime.rank)
        if config.train.resume is not None:
            self.restore(config.train.resume)

    def fit(self, batches: Iterable[Tensor]) -> None:
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
                paired = batch.ndim == 7
                batch = self.prepare_batch(batch)
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
                    terms = (
                        self.loss.from_pairs(embeddings.float().unflatten(0, (-1, 2)))
                        if paired
                        else self.loss.from_batch(embeddings.float())
                    )
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
            if self.step % cfg.save_every == 0 or self.step == cfg.max_steps:
                self.save()

    def prepare_batch(self, batch: Tensor) -> Tensor:
        """Validate the pair contract collectively, then flatten videos for encoding."""
        cfg = self.config
        tail = (
            cfg.input.positions,
            cfg.input.context_frames,
            3,
            cfg.input.image_size,
            cfg.input.image_size,
        )
        paired = batch.ndim == 7
        expected = (
            (cfg.train.batch_size // 2, 2, *tail) if paired else (cfg.train.batch_size, *tail)
        )
        self.runtime.require_all(
            tuple(batch.shape) == expected
            and batch.is_floating_point()
            and (not paired or cfg.train.batch_size % 2 == 0),
            f"Expected floating clips {expected}; paired batches require even batch_size.",
        )
        return batch.flatten(0, 1) if paired else batch

    def save(self) -> None:
        """Save all training state; each rank must enter this method."""
        states = gather_rank_states(self.encoder, self.runtime)
        if self.runtime.rank == 0:
            atomic_save(
                {
                    "version": 1,
                    "step": self.step,
                    "config": self.config.model_dump_json(),
                    "model": self.encoder.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scaler": self.scaler.state_dict(),
                    "ranks": states,
                    "data_provenance": self.data_provenance,
                },
                self.config.train.output_dir / "last.pt",
            )

    def restore(self, path: Path) -> None:
        """Resume a trusted local checkpoint with the same distributed topology."""
        checkpoint: Checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["version"] != 1 or len(checkpoint["ranks"]) != self.runtime.world_size:
            message = "Checkpoint version and world size must match this run."
            raise ValueError(message)
        old = ExperimentConfig.model_validate_json(checkpoint["config"])
        if checkpoint.get("data_provenance") != self.data_provenance:
            message = (
                "Resume requires the same video pairs and file metadata; "
                "use ckpt for initialization."
            )
            raise ValueError(message)
        excluded = {"max_steps", "log_every", "save_every", "output_dir", "resume"}
        same = (
            old.model.model_dump(exclude={"ckpt"}) == self.config.model.model_dump(exclude={"ckpt"})
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
        state = checkpoint["ranks"][self.runtime.rank]
        if state is None:
            message = "Checkpoint is missing this rank's state."
            raise ValueError(message)
        restore_rank_state(self.encoder, state)
