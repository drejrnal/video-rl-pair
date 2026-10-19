# Copyright (c) 2026 video-rl-pair contributors.
"""Match demonstrations only within held-out tasks and report annotation-free diagnostics."""

from collections.abc import Sequence
from typing import Literal

import torch

from video_rl_pair.config import ExperimentConfig
from video_rl_pair.data import PairDataConfig, VideoPair, discover_pairs
from video_rl_pair.evaluation_types import (
    Diagnostics,
    EvaluationConfig,
    EvaluationReport,
    PairResult,
    aggregate,
)
from video_rl_pair.inference import EncodedVideo, FrozenEncoder
from video_rl_pair.matching import (
    MatchingResult,
    embedding_spread,
    match_embeddings,
    normalized_time_matches,
    summarize_matches,
)
from video_rl_pair.progression import ProgressionPair, ProgressionReport, make_progression_pair
from video_rl_pair.progression_report import write_progression_report
from video_rl_pair.visual_report import VisualMatch, write_visual_report


def compare_pair(
    pair: VideoPair,
    videos: tuple[EncodedVideo, EncodedVideo],
    methods: Sequence[Literal["nearest", "dtw"]],
) -> tuple[list[PairResult], list[VisualMatch]]:
    """Evaluate both directions, including a timestamp-only reference baseline."""
    first, second = videos
    results: list[PairResult] = []
    visuals: list[VisualMatch] = []
    all_methods: tuple[Literal["nearest", "dtw", "time"], ...] = (*dict.fromkeys(methods), "time")
    similarity = match_embeddings(first.embeddings, second.embeddings).similarity
    for method in all_methods:
        if method == "time":
            forward = MatchingResult(normalized_time_matches(first.times, second.times), similarity)
            reverse = MatchingResult(
                normalized_time_matches(second.times, first.times), similarity.T
            )
        else:
            forward = match_embeddings(first.embeddings, second.embeddings, method)
            reverse = match_embeddings(second.embeddings, first.embeddings, method)
        for source, target, match, back in (
            (first, second, forward, reverse),
            (second, first, reverse, forward),
        ):
            stats = summarize_matches(source.embeddings, target.embeddings, match.indices)
            returned: torch.Tensor = back.indices[match.indices]
            error = (source.times[returned] - source.times).abs().mean()
            error /= source.times[-1] - source.times[0]
            results.append(
                PairResult(
                    task_id=pair.task_id,
                    pair_id=pair.pair_id,
                    source=str(source.path),
                    target=str(target.path),
                    method=method,
                    source_times=tuple(float(t) for t in source.times),
                    target_times=tuple(float(t) for t in target.times),
                    indices=tuple(int(i) for i in match.indices),
                    diagnostics=Diagnostics(
                        mean_cosine=stats.mean_cosine,
                        tau_b=stats.tau_b,
                        unique_target_fraction=stats.unique_target_fraction,
                        cycle_time_error=float(error),
                    ),
                )
            )
        visuals.append(
            VisualMatch(
                task_id=pair.task_id,
                pair_id=pair.pair_id,
                method=method,
                source_times=tuple(float(t) for t in first.times),
                target_times=tuple(float(t) for t in second.times),
                indices=tuple(int(i) for i in forward.indices),
                similarity=forward.similarity,
                source_images=first.images,
                target_images=second.images,
            )
        )
    return results, visuals


def run_evaluation(
    settings: ExperimentConfig, data: PairDataConfig, options: EvaluationConfig
) -> EvaluationReport:
    """Freeze the checkpoint, sample pairs reproducibly, and average each task equally."""
    output = options.output_dir
    if any((output / name).exists() for name in ("report.json", "index.html", "progression.html")):
        message = "Evaluation output exists; choose a fresh evaluation.output_dir."
        raise ValueError(message)
    encoder = FrozenEncoder(settings, data, options)
    pairs = discover_pairs(data, "eval")
    results: list[PairResult] = []
    visuals: list[VisualMatch] = []
    progression: list[ProgressionPair] = []
    spread: dict[str, float] = {}
    available: dict[str, int] = {}
    selected: dict[str, int] = {}
    for task in data.eval_tasks:
        candidates = [pair for pair in pairs if pair.task_id == task]
        available[task] = len(candidates)
        generator = torch.Generator().manual_seed(options.seed)
        order = torch.randperm(len(candidates), generator=generator)[: options.max_pairs_per_task]
        chosen = [candidates[int(index)] for index in order]
        selected[task] = len(chosen)
        paths = sorted({path for pair in chosen for path in (pair.first, pair.second)})
        encoded = {path: encoder.encode(path) for path in paths}
        spread.update(
            {str(path): embedding_spread(video.embeddings) for path, video in encoded.items()}
        )
        for number, pair in enumerate(chosen):
            scores, images = compare_pair(
                pair, (encoded[pair.first], encoded[pair.second]), options.methods
            )
            results.extend(scores)
            if number < options.visual_pairs_per_task:
                visuals.extend(images)
                progression.append(
                    make_progression_pair(pair, (encoded[pair.first], encoded[pair.second]), scores)
                )
    by_task, macro = aggregate(results)
    report = EvaluationReport(
        checkpoint=str(options.checkpoint) if options.checkpoint is not None else None,
        initialization_baseline=options.checkpoint is None,
        training_provenance_checked=encoder.provenance_checked,
        train_tasks=data.train_tasks,
        eval_tasks=data.eval_tasks,
        options=options,
        available_pairs=available,
        selected_pairs=selected,
        embedding_spread=spread,
        by_task=by_task,
        macro=macro,
        pairs=tuple(results),
    )
    output.mkdir(parents=True, exist_ok=True)
    write_visual_report(visuals, output / "index.html")
    write_progression_report(
        ProgressionReport(
            horizon=options.progression_horizon,
            pairs=tuple(progression),
            checkpoint=report.checkpoint,
            initialization_baseline=report.initialization_baseline,
        ),
        output / "progression.html",
    )
    _ = (output / "report.json").write_text(report.model_dump_json(indent=2) + "\n")
    return report
