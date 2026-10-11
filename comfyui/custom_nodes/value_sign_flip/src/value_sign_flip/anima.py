"""Value Sign Flip support for ComfyUI's Anima DiT implementation.

ComfyUI's Anima attention exposes ``attn2_patch`` before the key/value
projections.  Its value projection is bias-free, so scaling the negative
context tokens before projection is equivalent to scaling their projected
values.  See the official implementations:

* https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/anima/model.py
* https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/cosmos/predict2.py
* https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/model_patcher.py
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch


Conditioning = List[List[Any]]


def _conditioning_entries(conditioning: Conditioning, name: str) -> Sequence[Sequence[Any]]:
    if not isinstance(conditioning, (list, tuple)) or not conditioning:
        raise ValueError(f"{name} conditioning must contain at least one entry")
    for entry in conditioning:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError(f"{name} conditioning entries must be [tensor, metadata] pairs")
    return conditioning


def _conditioning_tensor(entry: Sequence[Any], name: str) -> torch.Tensor:
    tensor = entry[0]
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} conditioning values must be torch.Tensor instances")
    if tensor.ndim != 3:
        raise ValueError(f"{name} conditioning tensors must have shape [batch, tokens, channels]")
    return tensor


def fuse_anima_conditioning(
    positive: Conditioning,
    negative: Conditioning,
) -> Tuple[Conditioning, int, int]:
    """Concatenate matching positive and negative conditioning entries.

    Anima receives one context sequence.  The positive sequence is kept
    first, followed by the negative sequence, so the attention patch can
    identify the value tokens to negate without changing the model input
    batch or running a second denoising pass.
    """

    positive_entries = _conditioning_entries(positive, "positive")
    negative_entries = _conditioning_entries(negative, "negative")
    if len(positive_entries) != len(negative_entries):
        raise ValueError("positive and negative conditioning must have the same number of entries")

    fused: Conditioning = []
    lengths: Optional[Tuple[int, int]] = None

    for positive_entry, negative_entry in zip(positive_entries, negative_entries):
        positive_tensor = _conditioning_tensor(positive_entry, "positive")
        negative_tensor = _conditioning_tensor(negative_entry, "negative")
        if positive_tensor.shape[-1] != negative_tensor.shape[-1]:
            raise ValueError("positive and negative conditioning channel dimensions must match")

        positive_batch, negative_batch = positive_tensor.shape[0], negative_tensor.shape[0]
        if positive_batch != negative_batch:
            if positive_batch == 1:
                positive_tensor = positive_tensor.expand(negative_batch, -1, -1)
            elif negative_batch == 1:
                negative_tensor = negative_tensor.expand(positive_batch, -1, -1)
            else:
                raise ValueError("positive and negative conditioning batch dimensions must match")

        pos_len = positive_tensor.shape[1]
        neg_len = negative_tensor.shape[1]
        if pos_len == 0 or neg_len == 0:
            raise ValueError("positive and negative conditioning must contain at least one token")
        if lengths is None:
            lengths = (pos_len, neg_len)
        elif lengths != (pos_len, neg_len):
            raise ValueError("all conditioning entries must use the same positive and negative token lengths")

        metadata = copy.copy(positive_entry[1]) if isinstance(positive_entry[1], dict) else {}
        metadata["vsf_anima"] = True
        metadata["vsf_anima_pos_len"] = pos_len
        metadata["vsf_anima_neg_len"] = neg_len
        fused.append([torch.cat((positive_tensor, negative_tensor), dim=1), metadata])

    assert lengths is not None
    return fused, lengths[0], lengths[1]


# Shorter alias for callers that do not need to mention the model family.
fuse_conditioning = fuse_anima_conditioning


class AnimaVSFAttnPatch:
    """ComfyUI ``attn2_patch`` that flips the negative value tokens."""

    def __init__(self, pos_len: int, neg_len: int, scale: float = 1.5):
        if pos_len <= 0 or neg_len <= 0:
            raise ValueError("pos_len and neg_len must be positive")
        if scale < 0:
            raise ValueError("scale must be non-negative")
        self.pos_len = int(pos_len)
        self.neg_len = int(neg_len)
        self.scale = float(scale)

    @property
    def context_len(self) -> int:
        return self.pos_len + self.neg_len

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: Optional[torch.Tensor],
        *,
        pe: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if v is None:
            return {"q": q, "k": k, "v": v, "pe": pe, "attn_mask": attn_mask}
        if v.ndim != 3:
            raise ValueError("Anima attention values must have shape [batch, tokens, channels]")
        if v.shape[1] != self.context_len:
            raise ValueError(
                f"Anima VSF expected {self.context_len} context tokens, got {v.shape[1]}"
            )

        positive_values = v[:, : self.pos_len]
        negative_values = v[:, self.pos_len :] * (-self.scale)
        return {
            "q": q,
            "k": k,
            "v": torch.cat((positive_values, negative_values), dim=1),
            "pe": pe,
            "attn_mask": attn_mask,
        }


# Name aligned with ComfyUI's attention patch terminology and the project plan.
AnimaVSFAttnProcessor = AnimaVSFAttnPatch


def _negative_attention_mask(
    *,
    key_len: int,
    pos_len: int,
    offset: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    mask = torch.zeros((1, 1, 1, key_len), device=device, dtype=dtype)
    mask[..., pos_len:] = offset
    return mask


def _merge_attention_masks(
    mask: Optional[torch.Tensor],
    negative_mask: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    if mask is None:
        return negative_mask
    if mask.dtype == torch.bool:
        min_value = torch.finfo(dtype).min
        mask = torch.zeros_like(mask, dtype=dtype).masked_fill(~mask, min_value)
    else:
        mask = mask.to(dtype=dtype)
    return mask + negative_mask


class AnimaVSFOptimizedAttention:
    """Add the optional negative-key bias while preserving ComfyUI's backend.

    ``ModelPatcher.set_model_optimized_attention`` invokes this callable for
    Anima's native attention operation.  The configured ComfyUI backend is
    loaded lazily and called without the override in its transformer options,
    preventing recursive dispatch.
    """

    def __init__(
        self,
        pos_len: int,
        neg_len: int,
        offset: float,
        base_attention: Optional[Callable[..., torch.Tensor]] = None,
    ):
        if pos_len <= 0 or neg_len <= 0:
            raise ValueError("pos_len and neg_len must be positive")
        self.pos_len = int(pos_len)
        self.neg_len = int(neg_len)
        self.offset = float(offset)
        self.base_attention = base_attention

    @property
    def context_len(self) -> int:
        return self.pos_len + self.neg_len

    def _get_base_attention(self) -> Callable[..., torch.Tensor]:
        if self.base_attention is not None:
            return self.base_attention
        try:
            from comfy.ldm.modules.attention import optimized_attention
        except ImportError as exc:  # pragma: no cover - only reached outside ComfyUI
            raise RuntimeError("Anima VSF attention requires ComfyUI's attention module") from exc
        return optimized_attention

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        # ComfyUI consumes AttentionTensorContainer inputs before invoking a
        # custom optimized-attention function.  This also keeps the callable
        # convenient to exercise directly in unit tests.
        if not isinstance(q, torch.Tensor) and hasattr(q, "take"):
            q = q.take()
        if not isinstance(k, torch.Tensor) and hasattr(k, "take"):
            k = k.take()
        if not isinstance(v, torch.Tensor) and hasattr(v, "take"):
            v = v.take()

        base_kwargs = dict(kwargs)
        base_kwargs.pop("preferred_attention", None)
        base_kwargs.pop("transformer_options", None)
        base_kwargs.pop("_inside_attn_wrapper", None)

        # Anima's torch_attention_op supplies [batch, heads, tokens, dim].
        # Self-attention has a different token count, so it passes through
        # unchanged; only the fused text context receives the negative bias.
        if self.offset != 0.0 and k.ndim == 4 and k.shape[-2] == self.context_len:
            value_dtype = q.dtype if q.is_floating_point() else torch.float32
            negative_mask = _negative_attention_mask(
                key_len=k.shape[-2],
                pos_len=self.pos_len,
                offset=self.offset,
                device=q.device,
                dtype=value_dtype,
            )
            mask = _merge_attention_masks(mask, negative_mask, value_dtype)

        return self._get_base_attention()(
            q,
            k,
            v,
            heads,
            mask=mask,
            **base_kwargs,
        )


def _append_transformer_patch(model: Any, patch_name: str, patch: Any) -> None:
    model_options = getattr(model, "model_options", None)
    if not isinstance(model_options, dict):
        raise TypeError("model does not expose a ComfyUI model_options dictionary")
    transformer_options = model_options.setdefault("transformer_options", {})
    patches = transformer_options.setdefault("patches", {})
    patches.setdefault(patch_name, []).append(patch)


def _register_attn2_patch(model: Any, patch: AnimaVSFAttnPatch) -> None:
    setter = getattr(model, "set_model_attn2_patch", None)
    if callable(setter):
        setter(patch)
    else:
        generic_setter = getattr(model, "set_model_patch", None)
        if callable(generic_setter):
            generic_setter(patch, "attn2_patch")
        else:
            _append_transformer_patch(model, "attn2_patch", patch)


def _register_attention_override(model: Any, override: AnimaVSFOptimizedAttention) -> None:
    setter = getattr(model, "set_model_optimized_attention", None)
    if callable(setter):
        setter(override)
    else:
        model_options = getattr(model, "model_options", None)
        if not isinstance(model_options, dict):
            raise TypeError("model does not support ComfyUI attention overrides")
        transformer_options = model_options.setdefault("transformer_options", {})
        transformer_options["optimized_attention_override"] = override


def patch_anima_model(
    model: Any,
    pos_len: int,
    neg_len: int,
    scale: float = 1.5,
    offset: float = 0.0,
) -> Any:
    """Clone a ComfyUI MODEL and install the Anima VSF hooks on the clone."""

    clone = getattr(model, "clone", None)
    if not callable(clone):
        raise TypeError("Anima VSF expects a ComfyUI MODEL with clone()")
    patched_model = clone()

    _register_attn2_patch(
        patched_model,
        AnimaVSFAttnPatch(pos_len=pos_len, neg_len=neg_len, scale=scale),
    )
    if offset != 0.0:
        _register_attention_override(
            patched_model,
            AnimaVSFOptimizedAttention(pos_len=pos_len, neg_len=neg_len, offset=offset),
        )
    return patched_model


__all__ = [
    "AnimaVSFAttnPatch",
    "AnimaVSFAttnProcessor",
    "AnimaVSFOptimizedAttention",
    "fuse_anima_conditioning",
    "fuse_conditioning",
    "patch_anima_model",
]
