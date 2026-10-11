"""ComfyUI nodes for Value Sign Flip.

The Anima nodes operate on ComfyUI's native ``MODEL`` and ``CONDITIONING``
types.  The older all-in-one Wan node is retained for backwards compatibility
and imports its optional diffusers dependencies only when it is executed.
"""

from __future__ import annotations

from typing import Any

try:
    import folder_paths
except ImportError:  # Unit tests can import the package outside ComfyUI.
    folder_paths = None

from .anima import fuse_anima_conditioning, patch_anima_model


class AnimaVSFAdvanced:
    """Apply Anima VSF with an optional negative-key attention offset."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "scale": (
                    "FLOAT",
                    {"default": 1.5, "min": 0.0, "max": 10.0, "step": 0.1},
                ),
                "offset": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    RETURN_NAMES = ("patched_model", "fused_conditioning")
    FUNCTION = "apply"
    CATEGORY = "model/conditioning/transform"

    def apply(self, model: Any, positive: Any, negative: Any, scale: float, offset: float):
        fused_conditioning, pos_len, neg_len = fuse_anima_conditioning(positive, negative)
        patched_model = patch_anima_model(
            model,
            pos_len=pos_len,
            neg_len=neg_len,
            scale=scale,
            offset=offset,
        )
        return patched_model, fused_conditioning


class AnimaVSFApply(AnimaVSFAdvanced):
    """Apply Anima VSF with the standard zero-offset configuration."""

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()["required"]
        return {"required": {key: value for key, value in inputs.items() if key != "offset"}}

    def apply(self, model: Any, positive: Any, negative: Any, scale: float):
        return super().apply(model, positive, negative, scale, offset=0.0)


class AnimaVSFPatcher(AnimaVSFAdvanced):
    """Explicit patcher name matching the Anima VSF architecture contract."""


class VSF:
    """Legacy Wan all-in-one node retained for existing workflows."""

    @classmethod
    def INPUT_TYPES(cls):
        if folder_paths is None:
            raise RuntimeError("The legacy Wan node requires ComfyUI's folder_paths module")
        return {
            "required": {
                "clip": (folder_paths.get_filename_list("text_encoders"),),
                "wan": (folder_paths.get_filename_list("diffusion_models"),),
                "vae": (folder_paths.get_filename_list("vae"),),
                "cfg_lora": (folder_paths.get_filename_list("loras"),),
                "model_id": ("STRING", {"default": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"}),
                "positive_prompt": (
                    "STRING",
                    {
                        "default": "A cessna flying over a snowy mountain landscape, with a clear blue sky and fluffy white clouds."
                    },
                ),
                "negative_prompt": ("STRING", {"default": "wings"}),
                "steps": ("INT", {"default": 12}),
                "height": ("INT", {"default": 480}),
                "width": ("INT", {"default": 832}),
                "frames": ("INT", {"default": 81}),
                "scale": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 10.0, "step": 0.1}),
                "offset": ("FLOAT", {"default": -0.1, "min": -1.0, "max": 0.0, "step": 0.01}),
                "shift": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "seed": ("INT", {"default": 42}),
                "cfg_lora_scale": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "main"
    CATEGORY = "sampling"

    def main(
        self,
        clip,
        wan,
        cfg_lora,
        vae,
        model_id,
        positive_prompt,
        shift,
        negative_prompt,
        steps,
        height,
        width,
        frames,
        scale,
        offset,
        seed,
        cfg_lora_scale,
    ):
        import torch
        from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanTransformer3DModel
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoTokenizer, T5EncoderModel

        from .pipeline import WanPipeline
        from .processor import WanAttnProcessor2_0

        model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        vae = AutoencoderKLWan.from_single_file(
            folder_paths.get_full_path("vae", vae),
            local_files_only=True,
            torch_dtype=torch.float32,
        )
        transformer = WanTransformer3DModel.from_single_file(
            folder_paths.get_full_path("diffusion_models", wan),
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        text_encoder_state = load_file(folder_paths.get_full_path("text_encoders", clip), device="cpu")
        text_encoder = T5EncoderModel.from_pretrained(
            pretrained_model_name_or_path=None,
            config=AutoConfig.from_pretrained(model_id, subfolder="text_encoder"),
            state_dict=text_encoder_state,
            torch_dtype=torch.bfloat16,
        )

        pipe = WanPipeline(
            vae=vae,
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=UniPCMultistepScheduler.from_pretrained(model_id, subfolder="scheduler", flow_shift=shift),
        )
        pipe.load_lora_weights(folder_paths.get_full_path("loras", cfg_lora), adapter_name="lora")
        pipe.to(device="cuda", dtype=torch.bfloat16)

        neg_prompt_embeds, _ = pipe.encode_prompt(
            prompt=negative_prompt,
            padding=False,
            do_classifier_free_guidance=False,
        )
        pos_prompt_embeds, _ = pipe.encode_prompt(
            prompt=positive_prompt,
            do_classifier_free_guidance=False,
            max_sequence_length=512 - neg_prompt_embeds.shape[1],
        )
        pipe.set_adapters("lora", cfg_lora_scale)

        neg_len = neg_prompt_embeds.shape[1]
        pos_len = pos_prompt_embeds.shape[1]
        img_len = (height // 8) * (width // 8) * 3 * (frames // 4 + 1) // 12
        mask = torch.zeros((1, img_len, pos_len + neg_len), device="cuda")
        mask[:, :, -neg_len:] = offset

        for block in pipe.transformer.blocks:
            block.attn2.processor = WanAttnProcessor2_0(scale=scale, neg_prompt_length=neg_len, attn_mask=mask)

        prompt_embeds = torch.cat([pos_prompt_embeds, neg_prompt_embeds], dim=1)
        output = pipe(
            prompt_embeds=prompt_embeds,
            height=height,
            width=width,
            num_frames=frames,
            num_inference_steps=steps,
            guidance_scale=0.0,
            generator=torch.Generator(device="cuda").manual_seed(seed),
            output_type="pt",
        )[0].float()[0].permute(0, 2, 3, 1).contiguous()
        return (output,)


class Example:
    """Compatibility metadata for the package's original scaffold test."""

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "test"
    CATEGORY = "Example"


NODE_CLASS_MAPPINGS = {
    "Example": VSF,
    "AnimaVSFPatcher": AnimaVSFPatcher,
    "AnimaVSFApply": AnimaVSFApply,
    "AnimaVSFAdvanced": AnimaVSFAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Example": "Value Sign Flip (Wan)",
    "AnimaVSFPatcher": "Anima VSF Patcher",
    "AnimaVSFApply": "Anima VSF",
    "AnimaVSFAdvanced": "Anima VSF (Advanced)",
}


__all__ = [
    "AnimaVSFAdvanced",
    "AnimaVSFApply",
    "AnimaVSFPatcher",
    "Example",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "VSF",
]
