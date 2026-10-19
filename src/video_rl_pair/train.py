# Copyright (c) 2026 video-rl-pair contributors.
"""Hydra entry point for paired MP4 training and optional held-out evaluation."""

from contextlib import ExitStack
from functools import partial

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from video_rl_pair.config import ExperimentConfig
from video_rl_pair.data import PairDataConfig, make_pair_batches
from video_rl_pair.distributed import distributed_runtime
from video_rl_pair.evaluation import run_evaluation
from video_rl_pair.evaluation_types import EvaluationConfig
from video_rl_pair.provenance import snapshot_data
from video_rl_pair.tracking import log_evaluation, log_training, tracking_run
from video_rl_pair.trainer import Trainer


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    """Compose configuration, initialize DDP, construct an iterable, and train."""
    has_data = config.get("data") is not None
    if config.batch_source is None and not has_data:
        message = (
            "Use --config-name pairs for MP4 data, --config-name demo for random tensors, "
            "or set batch_source._target_ to your own iterable factory."
        )
        raise ValueError(message)
    settings = ExperimentConfig.model_validate(
        OmegaConf.to_container(
            OmegaConf.masked_copy(config, ["model", "loss", "train", "input"]), resolve=True
        )
    )
    if has_data and config.batch_source is not None:
        message = "Choose either data or batch_source, not both."
        raise ValueError(message)
    data = (
        PairDataConfig.model_validate(OmegaConf.to_container(config.data, resolve=True))
        if has_data
        else None
    )
    options = (
        EvaluationConfig.model_validate(OmegaConf.to_container(config.evaluation, resolve=True))
        if config.get("evaluation") is not None
        else None
    )
    provenance = snapshot_data(data).model_dump_json() if data is not None else None
    with ExitStack() as tracking:
        with distributed_runtime(settings.train) as runtime:
            trainer = Trainer(settings, runtime, provenance)
            initial_step = trainer.step
            run = tracking.enter_context(tracking_run(config, runtime))
            batches = (
                make_pair_batches(settings, runtime, trainer.step, data=data)
                if data is not None
                else instantiate(
                    config.batch_source, settings=settings, runtime=runtime, start_step=trainer.step
                )
            )
            trainer.fit(batches, log_metrics=partial(log_training, run))
        if runtime.rank == 0 and data is not None and options is not None and options.enabled:
            checkpoint = settings.train.output_dir / "last.pt"
            if trainer.step == initial_step and settings.train.resume is not None:
                checkpoint = settings.train.resume
            report = run_evaluation(
                settings, data, options.model_copy(update={"checkpoint": checkpoint})
            )
            log_evaluation(run, report)


if __name__ == "__main__":
    main()
