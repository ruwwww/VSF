import copy

import pytest
import torch

from src.value_sign_flip.anima import (
    AnimaVSFAttnPatch,
    AnimaVSFOptimizedAttention,
    fuse_anima_conditioning,
    patch_anima_model,
)
from src.value_sign_flip.nodes import (
    AnimaVSFAdvanced,
    AnimaVSFApply,
    NODE_CLASS_MAPPINGS,
)


def _conditioning(values, metadata=None):
    return [[values, metadata or {}]]


def test_fuse_anima_conditioning_concatenates_context_and_tracks_lengths():
    positive = torch.tensor([[[1.0], [2.0]]])
    negative = torch.tensor([[[3.0], [4.0], [5.0]]])

    fused, pos_len, neg_len = fuse_anima_conditioning(
        _conditioning(positive, {"positive_only": True}),
        _conditioning(negative, {"negative_only": True}),
    )

    assert pos_len == 2
    assert neg_len == 3
    assert torch.equal(fused[0][0], torch.cat((positive, negative), dim=1))
    assert fused[0][1]["positive_only"] is True
    assert fused[0][1]["vsf_anima_pos_len"] == 2
    assert fused[0][1]["vsf_anima_neg_len"] == 3
    assert "negative_only" not in fused[0][1]
    assert torch.equal(positive, torch.tensor([[[1.0], [2.0]]]))


def test_fuse_anima_conditioning_rejects_unpaired_entries():
    context = torch.zeros(1, 2, 4)

    with pytest.raises(ValueError, match="same number of entries"):
        fuse_anima_conditioning(_conditioning(context) * 2, _conditioning(context))


def test_anima_attention_patch_scales_only_negative_values_without_mutation():
    patch = AnimaVSFAttnPatch(pos_len=2, neg_len=2, scale=1.5)
    values = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)
    original = values.clone()

    result = patch(
        torch.zeros(1, 1, 2),
        torch.zeros(1, 4, 2),
        values,
        pe="rope",
        attn_mask="mask",
        extra_options={},
    )

    assert torch.equal(result["q"], torch.zeros(1, 1, 2))
    assert torch.equal(result["k"], torch.zeros(1, 4, 2))
    assert torch.equal(result["v"][:, :2], original[:, :2])
    assert torch.equal(result["v"][:, 2:], original[:, 2:] * -1.5)
    assert result["pe"] == "rope"
    assert result["attn_mask"] == "mask"
    assert torch.equal(values, original)


def test_anima_optimized_attention_adds_offset_to_negative_keys_only():
    captured = {}

    def fake_attention(q, k, v, heads, mask=None, **kwargs):
        captured["mask"] = mask
        return "attention-output"

    attention = AnimaVSFOptimizedAttention(
        pos_len=2,
        neg_len=3,
        offset=-0.2,
        base_attention=fake_attention,
    )
    q = torch.zeros(1, 2, 4, 8)
    k = torch.zeros(1, 2, 5, 8)
    v = torch.zeros(1, 2, 5, 8)

    assert attention(q, k, v, heads=2, skip_reshape=True) == "attention-output"
    assert captured["mask"].shape == (1, 1, 1, 5)
    assert torch.equal(captured["mask"][..., :2], torch.zeros(1, 1, 1, 2))
    assert torch.allclose(captured["mask"][..., 2:], torch.full((1, 1, 1, 3), -0.2))


def test_patch_anima_model_clones_and_registers_comfy_hooks():
    class FakeModel:
        def __init__(self):
            self.attn_patch = None
            self.attention_override = None

        def clone(self):
            return copy.copy(self)

        def set_model_attn2_patch(self, patch):
            self.attn_patch = patch

        def set_model_optimized_attention(self, override):
            self.attention_override = override

    model = FakeModel()
    patched = patch_anima_model(model, pos_len=2, neg_len=3, scale=1.5, offset=-0.1)

    assert patched is not model
    assert isinstance(patched.attn_patch, AnimaVSFAttnPatch)
    assert isinstance(patched.attention_override, AnimaVSFOptimizedAttention)
    assert patched.attn_patch.scale == pytest.approx(1.5)
    assert patched.attention_override.offset == pytest.approx(-0.1)


def test_patch_anima_model_supports_model_options_wrapper_fallback():
    class Wrapper:
        def __init__(self):
            self.model_options = {"transformer_options": {}}

        def clone(self):
            clone = Wrapper()
            clone.model_options = copy.deepcopy(self.model_options)
            return clone

    model = Wrapper()
    patched = patch_anima_model(model, pos_len=2, neg_len=2, scale=1.0, offset=0.0)

    patches = patched.model_options["transformer_options"]["patches"]["attn2_patch"]
    assert len(patches) == 1
    assert isinstance(patches[0], AnimaVSFAttnPatch)
    assert "patches" not in model.model_options["transformer_options"]
    assert "optimized_attention_override" not in patched.model_options["transformer_options"]


def test_anima_nodes_expose_model_and_conditioning_contracts():
    assert AnimaVSFApply.RETURN_TYPES == ("MODEL", "CONDITIONING")
    assert AnimaVSFAdvanced.RETURN_TYPES == ("MODEL", "CONDITIONING")
    assert {"AnimaVSFApply", "AnimaVSFAdvanced"}.issubset(NODE_CLASS_MAPPINGS)

    apply_inputs = AnimaVSFApply.INPUT_TYPES()["required"]
    advanced_inputs = AnimaVSFAdvanced.INPUT_TYPES()["required"]
    assert set(apply_inputs) == {"model", "positive", "negative", "scale"}
    assert set(advanced_inputs) == {"model", "positive", "negative", "scale", "offset"}
    assert apply_inputs["model"] == ("MODEL",)
    assert apply_inputs["positive"] == ("CONDITIONING",)
    assert apply_inputs["negative"] == ("CONDITIONING",)
    assert apply_inputs["scale"][1]["default"] == pytest.approx(1.5)
    assert advanced_inputs["offset"][1]["default"] == pytest.approx(0.0)
