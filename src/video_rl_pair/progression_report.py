# Copyright (c) 2026 video-rl-pair contributors.
"""Serialize progression diagnostics into one portable offline HTML document."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from .progression import ProgressionReport


def write_progression_report(report: ProgressionReport, path: Path) -> None:
    """Embed trusted typed report data and package assets without external requests."""
    assets = files("video_rl_pair").joinpath("assets")
    payload = json.dumps(report.model_dump(mode="json"), ensure_ascii=True, allow_nan=False)
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    template = assets.joinpath("progression.html").read_text(encoding="utf-8")
    replacements = {
        "STYLE": assets.joinpath("progression.css").read_text(encoding="utf-8"),
        "DATA": payload,
        "LOGIC": assets.joinpath("progression_logic.js").read_text(encoding="utf-8"),
        "APP": assets.joinpath("progression_app.js").read_text(encoding="utf-8"),
    }
    # Split the template first so user data containing a template token remains literal.
    parts = template.split("@@")
    document = "".join(replacements.get(part, part) for part in parts)
    _ = path.write_text(document, encoding="utf-8")
