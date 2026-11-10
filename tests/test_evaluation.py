# Copyright (c) 2026 video-rl-pair contributors.
"""Exercise full-path image exports and checkpoint-only frozen inference offline."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from transformers import DINOv3ViTConfig

from video_rl_pair.config import ExperimentConfig, InputConfig, LossConfig, ModelConfig, TrainConfig
from video_rl_pair.data import PairDataConfig
from video_rl_pair.evaluation import run_evaluation
from video_rl_pair.evaluation_types import EvaluationConfig
from video_rl_pair.inference import EncodedVideo, FrozenEncoder
from video_rl_pair.model import VideoEncoder


def _settings() -> ExperimentConfig:
    backbone = DINOv3ViTConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=32,
        patch_size=16,
        num_register_tokens=4,
        pos_embed_rescale=None,
    )
    return ExperimentConfig(
        model=ModelConfig(
            source=Path("/nonexistent/initial-weights"),
            mode="stage1",
            hidden_dim=16,
            embedding_dim=8,
            backbone_config=backbone.to_dict(),
        ),
        loss=LossConfig(),
        train=TrainConfig(device="cpu"),
        input=InputConfig(positions=3, image_size=32),
    )


def _pair(tmp_path: Path) -> tuple[Path, Path]:
    first, second = tmp_path / "a.mp4", tmp_path / "b.mp4"
    first.touch()
    second.touch()
    return first, second


def test_export_contains_every_path_cell_and_original_rgb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, target = _pair(tmp_path)
    first = EncodedVideo(
        source,
        torch.eye(2),
        torch.tensor([0.0, 3.0]),
        (Image.new("RGB", (20, 10), "red"), Image.new("RGB", (20, 10), "green")),
    )
    second = EncodedVideo(
        target,
        torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([0.0, 1.0, 4.0]),
        tuple(Image.new("RGB", (10, 20), "blue") for _ in range(3)),
    )

    class StubEncoder:
        def __init__(
            self,
            settings: ExperimentConfig,
            _data: PairDataConfig | None,
            _options: EvaluationConfig,
        ) -> None:
            self.settings = settings
            self.provenance_check = "not_available"
            self.provenance_checked = False

        def encode(self, path: Path) -> EncodedVideo:
            return first if path == source else second

    monkeypatch.setattr("video_rl_pair.evaluation.FrozenEncoder", StubEncoder)
    output = tmp_path / "evaluation"
    options = EvaluationConfig(source=source, target=target, output_dir=output, image_size=128)
    report = run_evaluation(_settings(), None, options)
    result = report.pairs[0]
    assert [(step.source_index, step.target_index) for step in result.path] == [
        (0, 0),
        (0, 1),
        (1, 2),
    ]
    assert [step.target_time for step in result.path] == [0.0, 1.0, 4.0]
    assert len(list(output.rglob("*.png"))) == 3
    with Image.open(output / result.path[0].image) as image:
        assert image.getpixel((64, 54 + 64)) == (255, 0, 0)
        assert image.getpixel((128 + 64, 54 + 64)) == (0, 0, 255)
    serialized = json.loads((output / "report.json").read_text())
    assert serialized["pairs"][0]["total_cost"] == 0.0
    assert len(serialized["pairs"][0]["similarity"]) == 2
    html = (output / "index.html").read_text()
    assert "progression" not in html
    assert all(step.image in html for step in result.path)
    with pytest.raises(ValueError, match="not empty"):
        run_evaluation(_settings(), None, options)


def test_checkpoint_evaluation_needs_no_original_pretrained_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings()
    model = VideoEncoder(settings.model, initialize=False).eval()
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {"version": 2, "config": settings.model_dump_json(), "model": model.state_dict()},
        checkpoint,
    )
    source, target = _pair(tmp_path)
    options = EvaluationConfig(checkpoint=checkpoint, source=source, target=target, chunk_size=1)
    frames = torch.randn(2, 3, 32, 32)
    sampled = SimpleNamespace(
        clips=frames,
        times=torch.tensor([0.0, 2.0]),
        images=(Image.new("RGB", (32, 32)), Image.new("RGB", (32, 32))),
        valid_length=2,
    )
    monkeypatch.setattr("video_rl_pair.inference.sample_video", lambda _path, _settings: sampled)
    frozen = FrozenEncoder(settings, None, options)
    encoded = frozen.encode(source)
    with torch.no_grad():
        expected = model(frames.unsqueeze(0))[0]
    torch.testing.assert_close(encoded.embeddings, expected, rtol=1e-5, atol=1e-6)
    assert not any(parameter.requires_grad for parameter in frozen.model.parameters())
    assert not frozen.provenance_checked


def test_baseline_is_explicit_without_checkpoint() -> None:
    with pytest.raises(ValueError, match=r"model\.mode=baseline"):
        FrozenEncoder(_settings(), None, EvaluationConfig())


def test_explicit_pair_rejects_training_video_before_model_load(tmp_path: Path) -> None:
    settings = _settings()
    source, target = _pair(tmp_path)
    provenance = {
        "train_tasks": ["training"],
        "pairs": [],
        "videos": [{"path": str(source.resolve()), "size": 0, "modified_ns": 0}],
    }
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "version": 2,
            "config": settings.model_dump_json(),
            "model": {},
            "data_provenance": json.dumps(provenance),
        },
        checkpoint,
    )
    options = EvaluationConfig(checkpoint=checkpoint, source=source, target=target)
    with pytest.raises(ValueError, match="recorded training videos"):
        FrozenEncoder(settings, None, options)
