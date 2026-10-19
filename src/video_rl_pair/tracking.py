# Copyright (c) 2026 video-rl-pair contributors.
"""Optional rank-zero W&B logging; local metrics remain the training record."""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING, Final

from omegaconf import DictConfig, OmegaConf
from pydantic import JsonValue, TypeAdapter

from video_rl_pair.config import WandbConfig

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping

    from wandb.sdk.wandb_run import Run

    from video_rl_pair.distributed import Runtime
    from video_rl_pair.evaluation_types import EvaluationReport

LOGGER: Final = logging.getLogger(__name__)


@contextmanager
def tracking_run(config: DictConfig, runtime: Runtime) -> Generator[Run | None, None, None]:
    """Open one optional run and finish it even when training or evaluation fails."""
    resolved = TypeAdapter(dict[str, JsonValue]).validate_python(
        OmegaConf.to_container(config, resolve=True)
    )
    options = WandbConfig.model_validate(resolved["wandb"])
    if options.mode == "disabled" or runtime.rank != 0:
        yield None
        return

    try:
        import wandb  # noqa: PLC0415
        from wandb.errors import Error  # noqa: PLC0415
    except ImportError:
        LOGGER.warning("W&B unavailable; install with uv sync --extra wandb. Continuing locally.")
        yield None
        return

    resolved["world_size"] = runtime.world_size
    try:
        options.dir.mkdir(parents=True, exist_ok=True)
        run = wandb.init(
            project=options.project,
            entity=options.entity,
            name=options.name,
            group=options.group,
            tags=list(options.tags),
            mode=options.mode,
            dir=str(options.dir),
            id=options.id,
            resume=options.resume,
            config=resolved,
            settings=wandb.Settings(init_timeout=60, save_code=False),
        )
    except (Error, OSError) as error:
        LOGGER.warning("W&B initialization failed; continuing locally: %s", error)
        yield None
        return

    try:
        try:
            _ = run.define_metric("train/step")
            _ = run.define_metric("train/*", step_metric="train/step")
        except (Error, OSError) as error:
            LOGGER.warning("W&B metric axis setup failed: %s", error)
        yield run
    finally:
        exit_code = int(sys.exc_info()[0] is not None)
        try:
            run.finish(exit_code=exit_code)
        except (Error, OSError) as error:
            LOGGER.warning("W&B finalization failed: %s", error)


def log_metrics(run: Run | None, metrics: Mapping[str, float | int]) -> None:
    """Send scalars without letting a W&B transport failure interrupt DDP."""
    if run is None:
        return
    from wandb.errors import Error  # noqa: PLC0415

    try:
        run.log(dict(metrics))
    except (Error, OSError) as error:
        LOGGER.warning("W&B metric logging failed; local metrics remain available: %s", error)


def log_training(run: Run | None, metrics: Mapping[str, float | int]) -> None:
    """Use the actual optimizer step as the training chart axis, including resume."""
    log_metrics(run, {f"train/{key}": value for key, value in metrics.items()})


def log_evaluation(run: Run | None, report: EvaluationReport) -> None:
    """Log macro and task-level diagnostics, not annotation-based accuracy."""
    if run is None:
        return
    metrics: dict[str, float | int] = {}
    groups = {"macro": report.macro, **{f"task/{k}": v for k, v in report.by_task.items()}}
    for group, methods in groups.items():
        for method, summary in methods.items():
            prefix = f"eval/{group}/{method}"
            metrics[f"{prefix}/count"] = summary.count
            metrics[f"{prefix}/tau_defined_count"] = summary.tau_defined_count
            diagnostics = summary.diagnostics
            metrics[f"{prefix}/mean_cosine"] = diagnostics.mean_cosine
            metrics[f"{prefix}/unique_target_fraction"] = diagnostics.unique_target_fraction
            metrics[f"{prefix}/cycle_time_error"] = diagnostics.cycle_time_error
            if diagnostics.tau_b is not None:
                metrics[f"{prefix}/tau_b"] = diagnostics.tau_b
    log_metrics(run, metrics)
