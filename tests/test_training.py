# Copyright (c) 2026 video-rl-pair contributors.
"""Tiny-model training, stage transitions, resume, and padding integration checks."""

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from video_rl_pair.config import ExperimentConfig, InputConfig, LossConfig, ModelConfig, TrainConfig
from video_rl_pair.data import PairBatch
from video_rl_pair.distributed import Runtime
from video_rl_pair.losses import AlignmentLoss
from video_rl_pair.trainer import Trainer


@pytest.fixture(autouse=True)
def single_cpu_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def source(tmp_path: Path) -> Path:
    local = tmp_path / "pretrained"
    torch.manual_seed(21)
    backbone = DINOv3ViTModel(
        DINOv3ViTConfig(
            image_size=32,
            patch_size=16,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_register_tokens=4,
            pos_embed_rescale=None,
        )
    )
    backbone.save_pretrained(local)
    return local


def settings(source: Path, output: Path, **train_options: object) -> ExperimentConfig:
    options = {
        "max_steps": 1,
        "batch_size": 2,
        "device": "cpu",
        "log_every": 1,
        "save_every": 1,
        "output_dir": output,
        "seed": 13,
        **train_options,
    }
    return ExperimentConfig(
        model=ModelConfig(source=source, hidden_dim=24, embedding_dim=16, unfreeze_last_n=1),
        input=InputConfig(positions=4, image_size=32),
        loss=LossConfig(),
        train=TrainConfig(**options),
    )


def runtime() -> Runtime:
    return Runtime(device=torch.device("cpu"), rank=0, world_size=1)


def batch(seed: int) -> PairBatch:
    generator = torch.Generator().manual_seed(seed)
    return PairBatch(torch.randn(1, 2, 4, 3, 32, 32, generator=generator), torch.tensor([[3, 4]]))


def test_stage1_optimizer_and_exact_resume(source: Path, tmp_path: Path) -> None:
    continuous = Trainer(settings(source, tmp_path / "continuous", max_steps=2), runtime())
    original = {
        name: value.detach().clone() for name, value in continuous.encoder.state_dict().items()
    }
    continuous.fit([batch(7), batch(8)])
    assert continuous.step == 2
    for name, value in continuous.encoder.backbone.state_dict().items():
        torch.testing.assert_close(value, original[f"backbone.{name}"], atol=0, rtol=0)
    assert any(
        not torch.equal(value, original[f"head.{name}"])
        for name, value in continuous.encoder.head.state_dict().items()
    )

    interrupted = Trainer(settings(source, tmp_path / "interrupted"), runtime())
    interrupted.fit([batch(7)])
    checkpoint = tmp_path / "interrupted" / "last.pt"
    payload = torch.load(checkpoint, weights_only=True)
    assert payload["version"] == 2
    # Source need not exist after checkpoint creation.
    resumed_config = settings(
        tmp_path / "no-source-needed", tmp_path / "resumed", max_steps=2, resume=checkpoint
    )
    resumed = Trainer(resumed_config, runtime())
    assert resumed.step == 1
    resumed.fit([batch(8)])
    for name, value in continuous.encoder.state_dict().items():
        torch.testing.assert_close(value, resumed.encoder.state_dict()[name], atol=0, rtol=0)
    for parameter, value in continuous.optimizer.state.items():
        # Compare Adam moments by the corresponding named model parameter.
        name = next(
            name
            for name, candidate in continuous.encoder.named_parameters()
            if candidate is parameter
        )
        counterpart = dict(resumed.encoder.named_parameters())[name]
        for key, item in value.items():
            torch.testing.assert_close(
                item, resumed.optimizer.state[counterpart][key], atol=0, rtol=0
            )


def test_checkpoint_snapshots_preserve_intervals_and_final_step(
    source: Path, tmp_path: Path
) -> None:
    output = tmp_path / "snapshots"
    trainer = Trainer(settings(source, output, max_steps=5, save_every=2), runtime())
    trainer.fit([batch(seed) for seed in range(5)])
    assert {path.name for path in output.glob("*.pt")} == {
        "step_000002.pt",
        "step_000004.pt",
        "step_000005.pt",
        "last.pt",
    }
    for step in (2, 4, 5):
        snapshot = torch.load(output / f"step_{step:06d}.pt", weights_only=True)
        assert snapshot["step"] == step
    latest = torch.load(output / "last.pt", weights_only=True)
    assert latest["step"] == 5
    for name, value in trainer.encoder.state_dict().items():
        torch.testing.assert_close(value, latest["model"][name], atol=0, rtol=0)
    resumed = Trainer(
        settings(
            source, tmp_path / "resume-snapshot", max_steps=5, resume=output / "step_000002.pt"
        ),
        runtime(),
    )
    assert resumed.step == 2


def test_stage2_transition_keeps_weights_but_restarts_optimizer(
    source: Path, tmp_path: Path
) -> None:
    first = Trainer(settings(source, tmp_path / "stage1"), runtime())
    first.fit([batch(23)])
    next_config = settings(
        tmp_path / "missing", tmp_path / "stage2", init_checkpoint=tmp_path / "stage1" / "last.pt"
    )
    next_config = next_config.model_copy(
        update={"model": next_config.model.model_copy(update={"mode": "stage2"})}
    )
    second = Trainer(next_config, runtime())
    assert second.step == 0
    assert not second.optimizer.state
    assert [group["lr"] for group in second.optimizer.param_groups] == [1e-4, 1e-5]
    for name, value in first.encoder.state_dict().items():
        torch.testing.assert_close(value, second.encoder.state_dict()[name], atol=0, rtol=0)
    original = {name: value.detach().clone() for name, value in second.encoder.state_dict().items()}
    second.fit([batch(24)])
    assert second.step == 1
    for name, parameter in second.encoder.named_parameters():
        trained = name.startswith(("head.", "backbone.layer.1.", "backbone.norm."))
        if trained:
            assert parameter.grad is not None, name
        else:
            assert parameter.grad is None, name
            torch.testing.assert_close(parameter, original[name], atol=0, rtol=0)
    assert any(
        not torch.equal(value, original[f"backbone.layer.1.{name}"])
        for name, value in second.encoder.backbone.layer[1].state_dict().items()
    )


def test_baseline_cannot_be_used_for_training(source: Path, tmp_path: Path) -> None:
    config = settings(source, tmp_path / "baseline")
    config = config.model_copy(
        update={"model": config.model.model_copy(update={"mode": "baseline"})}
    )
    with pytest.raises(ValueError, match="baseline has no trainable parameters"):
        Trainer(config, runtime())


def test_padding_is_excluded_before_costs_and_has_zero_gradient() -> None:
    generator = torch.Generator().manual_seed(91)
    original = torch.randn(2, 2, 5, 8, generator=generator)
    lengths = torch.tensor([[2, 4], [5, 3]])
    modified = original.clone()
    for pair in range(2):
        for video in range(2):
            modified[pair, video, lengths[pair, video] :] = 100 * torch.randn(
                modified[pair, video, lengths[pair, video] :].shape, generator=generator
            )
    original.requires_grad_()
    modified.requires_grad_()
    objective = AlignmentLoss(LossConfig())
    expected = objective.from_pairs(original, lengths)
    actual = objective.from_pairs(modified, lengths)
    for field in ("total", "alignment", "cycle"):
        torch.testing.assert_close(getattr(expected, field), getattr(actual, field), atol=0, rtol=0)
    expected.total.backward()
    actual.total.backward()
    torch.testing.assert_close(original.grad, modified.grad, atol=0, rtol=0)
    for pair in range(2):
        for video in range(2):
            assert torch.count_nonzero(original.grad[pair, video, lengths[pair, video] :]) == 0
