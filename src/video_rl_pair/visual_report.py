# Copyright (c) 2026 video-rl-pair contributors.
"""Self-contained visual diagnostics for directed, same-task video matches."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from html import escape
from io import BytesIO
from typing import TYPE_CHECKING, Final, Literal, assert_never

import torch
from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from torch import Tensor


@dataclass(frozen=True, slots=True)
class VisualMatch:
    """Trusted directed alignment with an [M,N] cosine matrix and JPEG data URIs."""

    task_id: str
    pair_id: str
    method: str
    source_times: tuple[float, ...]
    target_times: tuple[float, ...]
    indices: tuple[int, ...]
    similarity: Tensor
    source_images: tuple[str, ...]
    target_images: tuple[str, ...]


def _data_uri(image: Image.Image, encoding: Literal["JPEG", "PNG"]) -> str:
    with BytesIO() as buffer:
        image.save(buffer, format=encoding)
        payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{encoding.lower()};base64,{payload}"


def thumbnails(
    clips: Tensor, backbone: Literal["resnet50_v2", "resnet50_v1"], max_side: int = 128
) -> tuple[str, ...]:
    """Recover last context frames from [T,K,3,H,W] and return bounded JPEGs."""
    frames = clips[:, -1].detach().to(device="cpu", dtype=torch.float32)
    match backbone:
        case "resnet50_v2":
            frames = (frames + 1) / 2
        case "resnet50_v1":
            mean = frames.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
            std = frames.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
            frames = frames * std + mean
        case _:
            assert_never(backbone)
    pixels = (frames.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1)
    images: list[str] = []
    for frame in pixels:
        with Image.fromarray(frame.numpy()) as image:
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            images.append(_data_uri(image, "JPEG"))
    return tuple(images)


def _heatmap(item: VisualMatch) -> str:
    values = item.similarity.detach().to(device="cpu", dtype=torch.float32).clamp(-1, 1)
    channels = (1 + values.clamp(max=0), 1 - values.abs(), 1 - values.clamp(min=0))
    pixels = (torch.stack(channels, dim=-1) * 255).round().to(torch.uint8)
    rows, columns = values.shape
    with Image.fromarray(pixels.numpy()) as image:
        uri = _data_uri(image, "PNG")
    points = "".join(
        f"""<circle cx="{70 + (column + 0.5) * 480 / columns:.3f}"
        cy="{35 + (row + 0.5) * 320 / rows:.3f}" r="3"
        fill="#ffe761" stroke="#172034" stroke-width="1.2"/>"""
        for row, column in enumerate(item.indices)
    )
    return (
        '<svg viewBox="0 0 600 410" role="img" aria-label="Cosine similarity heatmap: '
        'source rows, target columns; yellow circles mark selected matches">'
        f'<image href="{uri}" x="70" y="35" width="480" height="320" '
        'preserveAspectRatio="none" style="image-rendering:pixelated"/>'
        f'{points}<g fill="currentColor" font-size="13" text-anchor="middle">'
        '<text x="310" y="402">Target j →</text>'
        '<text transform="translate(18 195) rotate(-90)">Source i →</text>'
        f'<text x="70" y="377">0</text><text x="550" y="377">{columns - 1}</text>'
        f'<text x="50" y="43">0</text><text x="50" y="355">{rows - 1}</text>'
        '</g></svg><div class="legend"><span>-1</span><span class="gradient"></span>'
        '<span>+1 cosine</span></div><p class="muted">Yellow circles: selected matches.</p>'
    )


def _thumbnail_pairs(item: VisualMatch) -> str:
    count = min(6, len(item.indices))
    rows = tuple(
        round(index * (len(item.indices) - 1) / max(1, count - 1)) for index in range(count)
    )
    pairs: list[str] = []
    for row in rows:
        column = item.indices[row]
        pairs.append(
            f"""<figure class="match"><div class="frames"><div>
            <img src="{escape(item.source_images[row], quote=True)}" alt="Source frame {row}">
            <span>Source {row} · {item.source_times[row]:.3f}s</span></div><b>→</b><div>
            <img src="{escape(item.target_images[column], quote=True)}" alt="Target frame {column}">
            <span>Target {column} · {item.target_times[column]:.3f}s</span></div></div>
            <figcaption>Cosine {float(item.similarity[row, column]):.3f}</figcaption></figure>"""
        )
    return '<div class="matches">' + "".join(pairs) + "</div>"


_STYLE: Final = """
:root {color-scheme:light; font-family:system-ui,sans-serif; color:#172034; background:#f3f5f8}
* {box-sizing:border-box} body {margin:0; padding:36px 20px} main {max-width:1120px; margin:auto}
h1 {font-size:clamp(24px,4vw,36px); letter-spacing:-.03em; margin:0 0 12px}
h2 {font-size:21px; margin:0 0 8px; overflow-wrap:anywhere} p {line-height:1.6}
.intro {max-width:800px; color:#4c596e} .muted {color:#58667a; font-size:13px}
section {background:white; border:1px solid #dae0e9; border-radius:16px; margin-top:28px;
 padding:24px; box-shadow:0 4px 20px #17203406}
.method {display:inline-block; background:#edf0fc; color:#354980; padding:4px 10px;
 border-radius:6px; font-size:13px; overflow-wrap:anywhere}
.layout {display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:28px; margin-top:20px}
svg {display:block; width:100%; max-height:410px} .legend {display:flex; align-items:center;
 justify-content:center; gap:8px; font-size:12px} .gradient {width:150px; height:12px;
 background:linear-gradient(90deg,#00f,#fff,#f00); border:1px solid #b0b8c6}
.matches {display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px;
 align-content:start}
figure {margin:0; min-width:0} .frames {display:flex; gap:6px; align-items:center}
.frames div {min-width:0; flex:1} img {display:block; width:100%; height:auto; border-radius:6px}
.frames span {display:block; margin-top:5px; font-size:10px; line-height:1.5; color:#58667a}
figcaption {font-size:11px; margin-top:7px; color:#354980}
@media(max-width:760px) {body {padding:20px 12px} section {padding:16px}
 .layout {grid-template-columns:1fr}}
"""


def write_visual_report(items: Sequence[VisualMatch], path: Path) -> None:
    """Write an offline HTML diagnostic report from trusted same-task matches."""
    sections = "".join(
        "<section><h2>"
        + escape(item.task_id)
        + " · "
        + escape(item.pair_id)
        + '</h2><span class="method">'
        + escape(item.method)
        + '</span><div class="layout"><div>'
        + _heatmap(item)
        + "</div>"
        + _thumbnail_pairs(item)
        + "</div></section>"
        for item in items
    )
    _ = path.write_text(
        f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
        <meta name="viewport" content="width=device-width,initial-scale=1">
        <title>Video pair alignment diagnostics</title><style>{_STYLE}</style></head>
        <body><main><h1>Video pair alignment diagnostics</h1><p class="intro">
        Directed source → target matches within the same held-out task. These annotation-free
        diagnostics do not measure semantic accuracy. High cosine similarity or orderly matches
        alone do not establish correct correspondence.</p><p class="muted">
        Thumbnails show the last context frame at each sampled time; up to six evenly spaced
        source moments are shown per alignment. Heatmap axes use sampled-frame indices.</p>
        <p><a href="progression.html">Open round-trip and progression viewer</a></p>
        {sections}</main></body></html>""",
        encoding="utf-8",
    )
