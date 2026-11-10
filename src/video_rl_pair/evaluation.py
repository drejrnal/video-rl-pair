# Copyright (c) 2026 video-rl-pair contributors.
"""One evaluation task: complete DTW alignment and a matching-frame PNG sequence."""

import torch

from video_rl_pair.config import ExperimentConfig
from video_rl_pair.data import PairDataConfig, VideoPair, discover_pairs
from video_rl_pair.evaluation_types import EvaluationConfig, EvaluationReport, PairResult, PathStep
from video_rl_pair.inference import FrozenEncoder
from video_rl_pair.matching import match_embeddings
from video_rl_pair.visual_report import write_frame_sequence, write_visual_report


def _select_pairs(data: PairDataConfig | None, options: EvaluationConfig) -> tuple[VideoPair, ...]:
    if options.source is not None and options.target is not None:
        return (VideoPair("explicit", "source--target", options.source, options.target),)
    if data is None:
        message = "Provide evaluation.source/target or held-out data task folders."
        raise ValueError(message)
    discovered = discover_pairs(data, "eval")
    selected: list[VideoPair] = []
    for task in data.eval_tasks:
        candidates = [pair for pair in discovered if pair.task_id == task]
        generator = torch.Generator().manual_seed(options.seed)
        order = torch.randperm(len(candidates), generator=generator)[: options.max_pairs_per_task]
        selected.extend(candidates[int(index)] for index in order)
    return tuple(selected)


def run_evaluation(
    settings: ExperimentConfig,
    data: PairDataConfig | None,
    options: EvaluationConfig,
) -> EvaluationReport:
    """Freeze the encoder, align sampled frames and render every cell on each DTW path."""
    output = options.output_dir
    if output.exists() and any(output.iterdir()):
        message = "Evaluation output is not empty; choose a fresh evaluation.output_dir."
        raise ValueError(message)
    pairs = _select_pairs(data, options)
    encoder = FrozenEncoder(settings, data, options)
    output.mkdir(parents=True, exist_ok=True)
    results: list[PairResult] = []
    selected: dict[str, int] = {}
    for number, pair in enumerate(pairs):
        first, second = encoder.encode(pair.first), encoder.encode(pair.second)
        match = match_embeddings(first.embeddings, second.embeddings)
        images = write_frame_sequence(
            first, second, match, output / f"pair_{number:04d}" / "frames", options.image_size
        )
        path = tuple(
            PathStep(
                source_index=row,
                target_index=column,
                source_time=float(first.times[row]),
                target_time=float(second.times[column]),
                cosine=float(match.similarity[row, column]),
                cost=float(match.cost[row, column]),
                image=str(image.relative_to(output)),
            )
            for (row, column), image in zip(match.path, images, strict=True)
        )
        results.append(
            PairResult(
                task_id=pair.task_id,
                pair_id=pair.pair_id,
                source=str(pair.first.resolve()),
                target=str(pair.second.resolve()),
                source_times=tuple(float(time) for time in first.times),
                target_times=tuple(float(time) for time in second.times),
                similarity=tuple(tuple(float(value) for value in row) for row in match.similarity),
                cost=tuple(tuple(float(value) for value in row) for row in match.cost),
                path=path,
                total_cost=match.total_cost,
            )
        )
        selected[pair.task_id] = selected.get(pair.task_id, 0) + 1
    report = EvaluationReport(
        checkpoint=str(options.checkpoint) if options.checkpoint is not None else None,
        mode=encoder.settings.model.mode,
        training_provenance_checked=encoder.provenance_checked,
        provenance_check=encoder.provenance_check,
        options=options,
        selected_pairs=selected,
        pairs=tuple(results),
    )
    _ = (output / "report.json").write_text(
        report.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    write_visual_report(results, output / "index.html")
    return report
