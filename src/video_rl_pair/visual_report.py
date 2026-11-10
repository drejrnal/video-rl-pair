# Copyright (c) 2026 video-rl-pair contributors.
"""Render every DTW path cell as a numbered pair of original video frames."""

from collections.abc import Sequence
from html import escape
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from video_rl_pair.evaluation_types import PairResult
from video_rl_pair.inference import EncodedVideo
from video_rl_pair.matching import MatchingResult


def write_frame_sequence(
    source: EncodedVideo,
    target: EncodedVideo,
    match: MatchingResult,
    directory: Path,
    panel_size: int,
) -> tuple[Path, ...]:
    """Save one side-by-side PNG per path step, retaining one-to-many correspondences."""
    directory.mkdir(parents=True, exist_ok=False)
    font = ImageFont.load_default(size=16)
    output: list[Path] = []
    header_height, footer_height = 54, 36
    for number, (row, column) in enumerate(match.path):
        canvas = Image.new(
            "RGB", (panel_size * 2, panel_size + header_height + footer_height), "#f1f3f6"
        )
        draw = ImageDraw.Draw(canvas)
        for panel, video, index, label in (
            (0, source, row, "Source"),
            (1, target, column, "Target"),
        ):
            frame = video.images[index].convert("RGB")
            frame.thumbnail((panel_size, panel_size), Image.Resampling.LANCZOS)
            left = panel * panel_size
            canvas.paste(
                frame,
                (
                    left + (panel_size - frame.width) // 2,
                    header_height + (panel_size - frame.height) // 2,
                ),
            )
            draw.text((left + 12, 8), f"{label} sampled frame {index}", fill="#172034", font=font)
            draw.text(
                (left + 12, 29),
                f"Time {float(video.times[index]):.3f} s",
                fill="#172034",
                font=font,
            )
        draw.text(
            (12, header_height + panel_size + 9),
            f"Step {number:04d}   cosine={float(match.similarity[row, column]):.4f}"
            f"   cost={float(match.cost[row, column]):.4f}",
            fill="#172034",
            font=font,
        )
        path = directory / f"{number:06d}.png"
        canvas.save(path)
        canvas.close()
        output.append(path)
    return tuple(output)


def write_visual_report(results: Sequence[PairResult], output: Path) -> None:
    """Write an offline gallery linking the actual full-resolution PNG sequence."""
    sections: list[str] = []
    for result in results:
        figures = "".join(
            '<figure><a href="'
            + escape(step.image, quote=True)
            + '"><img loading="lazy" src="'
            + escape(step.image, quote=True)
            + f'" alt="DTW step {number}: source {step.source_index}, '
            + f'target {step.target_index}"></a>'
            + f"<figcaption>Step {number}: {step.source_time:.3f}s → {step.target_time:.3f}s"
            + f" · cosine {step.cosine:.4f} · cost {step.cost:.4f}</figcaption></figure>"
            for number, step in enumerate(result.path)
        )
        sections.append(
            f"<section><h2>{escape(result.task_id)} · {escape(result.pair_id)}</h2>"
            f"<p>Source: {escape(result.source)}<br>Target: {escape(result.target)}</p>"
            f"<p>{len(result.source_times)} x {len(result.target_times)} sampled frames; "
            f"{len(result.path)} path steps; total cost {result.total_cost:.6f}.</p>"
            f'<div class="sequence">{figures}</div></section>'
        )
    _ = output.write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>DTW video alignment</title><style>"
        "body{font-family:system-ui,sans-serif;margin:24px auto;padding:0 20px;max-width:1400px;"
        "background:#f1f3f6;color:#172034}p{line-height:1.6;overflow-wrap:anywhere}"
        "section{background:white;padding:20px;border-radius:12px;margin:24px 0}"
        ".sequence{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr));gap:20px}"
        "figure{margin:0}img{display:block;width:100%;height:auto}figcaption{font-size:13px;margin-top:6px}"
        "</style></head><body><h1>DTW video alignment</h1>"
        "<p>Every path step is shown in order. Repeated source or target frames are intentional: "
        "DTW permits one-to-many matches. Click an image to open its PNG.</p>"
        "<p>The path minimizes the sum of 1 - cosine similarity with fixed endpoints and monotonic "
        "steps. Inspect whether corresponding action stages actually match; neither path order "
        "nor cost alone measures semantic accuracy. Times are original video timestamps "
        "in seconds; "
        "indices refer to the sampled frame grid.</p>"
        '<p><a href="report.json">Matrices, full path and timestamps (JSON)</a></p>'
        + "".join(sections)
        + "</body></html>\n",
        encoding="utf-8",
    )
