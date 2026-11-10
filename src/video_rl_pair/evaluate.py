# Copyright (c) 2026 video-rl-pair contributors.
"""Align two explicit videos or held-out dataset pairs, then export matching PNGs."""

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
    """Explicit source/target mode requires no data.root or task declarations."""
    settings = ExperimentConfig.model_validate(
        OmegaConf.to_container(
            OmegaConf.masked_copy(config, ["model", "loss", "train", "input"]), resolve=True
        )
    )
    options = EvaluationConfig.model_validate(
        OmegaConf.to_container(config.evaluation, resolve=True)
    )
    data = (
        None
        if options.source is not None
        else PairDataConfig.model_validate(OmegaConf.to_container(config.data, resolve=True))
    )
    report = run_evaluation(settings, data, options)
    LOGGER.info("Aligned %s video pairs; output=%s", len(report.pairs), options.output_dir)


if __name__ == "__main__":
    main()
