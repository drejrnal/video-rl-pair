# Copyright (c) 2026 video-rl-pair contributors.
"""Hydra CLI for frozen-encoder matching of held-out MP4 demonstrations."""

import logging
from typing import Final

import hydra
from omegaconf import DictConfig, OmegaConf

from video_rl_pair.config import ExperimentConfig
from video_rl_pair.data import PairDataConfig
from video_rl_pair.evaluation import run_evaluation
from video_rl_pair.evaluation_types import EvaluationConfig

LOGGER: Final = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="conf", config_name="pairs")
def main(config: DictConfig) -> None:
    """Load a checkpoint and write JSON diagnostics plus a local visual report."""
    settings = ExperimentConfig.model_validate(
        OmegaConf.to_container(
            OmegaConf.masked_copy(config, ["model", "loss", "train", "input"]), resolve=True
        )
    )
    data = PairDataConfig.model_validate(OmegaConf.to_container(config.data, resolve=True))
    options = EvaluationConfig.model_validate(
        OmegaConf.to_container(config.evaluation, resolve=True)
    )
    report = run_evaluation(settings, data, options)
    LOGGER.info("Evaluated %s held-out tasks; report=%s", len(report.by_task), options.output_dir)


if __name__ == "__main__":
    main()
