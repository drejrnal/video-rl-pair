# Copyright (c) 2026 video-rl-pair contributors.
"""Exercise real tiny DINOv3 models entirely offline."""

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from video_rl_pair.config import LossConfig, ModelConfig
from video_rl_pair.losses import AlignmentLoss
from video_rl_pair.model import VideoEncoder


def tiny_config(mode: str = "stage1") -> ModelConfig:
    backbone = DINOv3ViTConfig(
        image_size=32,
        patch_size=16,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_register_tokens=4,
        pos_embed_rescale=None,
    )
    return ModelConfig(
        mode=mode,
        hidden_dim=24,
        embedding_dim=16,
        unfreeze_last_n=1,
        backbone_config=backbone.to_dict(),
    )


@pytest.fixture(autouse=True)
def limit_cpu_threads() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_baseline_is_normalized_raw_cls_without_trainable_parameters() -> None:
    model = VideoEncoder(tiny_config("baseline"), initialize=False).eval()
    clips = torch.randn(2, 3, 3, 32, 32)
    actual = model(clips)
    with torch.no_grad():
        raw = model.backbone(pixel_values=clips.flatten(0, 1)).last_hidden_state[:, 0]
    expected = torch.nn.functional.normalize(raw, dim=-1).reshape(2, 3, 32)
    torch.testing.assert_close(actual, expected)
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert list(model.head.parameters()) == []
    assert model.regularization().item() == 0


def test_stage1_updates_head_and_preserves_backbone() -> None:
    torch.manual_seed(7)
    model = VideoEncoder(tiny_config(), initialize=False).train()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-3)
    clips = torch.randn(2, 4, 3, 32, 32)
    embeddings = model(clips)
    assert embeddings.shape == (2, 4, 16)
    torch.testing.assert_close(embeddings.norm(dim=-1), torch.ones(2, 4))
    AlignmentLoss(LossConfig())(embeddings[0], embeddings[1]).total.backward()
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.head.parameters()
    )
    optimizer.step()
    for name, parameter in model.backbone.named_parameters():
        torch.testing.assert_close(parameter, before[f"backbone.{name}"], rtol=0, atol=0)
    assert any(
        not torch.equal(parameter, before[f"head.{name}"])
        for name, parameter in model.head.named_parameters()
    )
    assert not any(module.training for module in model.backbone.modules())


def test_stage2_only_selected_blocks_and_head_receive_gradients() -> None:
    torch.manual_seed(11)
    model = VideoEncoder(tiny_config("stage2"), initialize=False).train()
    embeddings = model(torch.randn(2, 4, 3, 32, 32))
    AlignmentLoss(LossConfig())(embeddings[0], embeddings[1]).total.backward()
    trained = []
    for name, parameter in model.named_parameters():
        expected = name.startswith(("head.", "backbone.layer.1.", "backbone.norm."))
        assert parameter.requires_grad == expected, name
        if expected:
            assert parameter.grad is not None, name
            trained.append(parameter.grad.abs().sum().item())
        else:
            assert parameter.grad is None, name
    assert sum(trained) > 0
    assert not model.backbone.layer[0].training
    assert model.backbone.layer[1].training
    assert not model.backbone.rope_embeddings.training
    model.eval()
    assert not any(module.training for module in model.modules())


def test_saved_architecture_reloads_without_source_directory(tmp_path: Path) -> None:
    model = VideoEncoder(tiny_config(), initialize=False).eval()
    source = tmp_path / "missing-model"
    restored = VideoEncoder(
        model.resolved_config.model_copy(update={"source": source}), initialize=False
    )
    restored.load_state_dict(model.state_dict(), strict=True)
    restored.eval()
    clips = torch.randn(2, 3, 3, 32, 32)
    torch.testing.assert_close(model(clips), restored(clips), rtol=0, atol=0)


def test_local_pretrained_loading_preserves_backbone_weights(tmp_path: Path) -> None:
    hf_config = DINOv3ViTConfig.from_dict(tiny_config().backbone_config)
    pretrained = DINOv3ViTModel(hf_config)
    pretrained.save_pretrained(tmp_path)
    model = VideoEncoder(ModelConfig(source=tmp_path))
    for name, value in pretrained.state_dict().items():
        torch.testing.assert_close(value, model.backbone.state_dict()[name], rtol=0, atol=0)
    assert model.resolved_config.backbone_config["hidden_size"] == 32


def test_reconstruction_requires_embedded_architecture() -> None:
    with pytest.raises(ValueError, match="backbone_config"):
        VideoEncoder(ModelConfig(), initialize=False)


def test_stage2_rejects_unavailable_blocks() -> None:
    config = tiny_config("stage2").model_copy(update={"unfreeze_last_n": 3})
    with pytest.raises(ValueError, match="block count"):
        VideoEncoder(config, initialize=False)
