# Architecture Specification: VSF Support for Anima DiT (Cosmos 2) in ComfyUI

## 1. Goal
Add official Anima DiT (Cosmos 2 architecture) support to `weathon/VSF`'s ComfyUI custom node package (`comfyui/custom_nodes/value_sign_flip`).

## 2. Background & Mechanism
Anima uses a 2D DiT architecture (`comfy.ldm.cosmos.predict2.Block` and `comfy.ldm.anima.model.TransformerBlock`) where cross-attention is decoupled from self-attention:
- Self-attention operates on spatial image tokens with 3D RoPE.
- Cross-attention operates between normalized image query tokens (`x`) and text context tokens (`context` from Qwen3).

In VSF (Value Sign Flip, ICLR 2026):
- Negative prompt tokens are appended to positive prompt tokens: `K_fused = [K_pos, K_neg]`
- The values corresponding to negative tokens are negated and scaled by `-scale`:
  `V_fused = [V_pos, -scale * V_neg]`
- Softmax cross-attention over `K_fused` and `V_fused` performs token-weighted subtraction ("Active Noise Cancelling"), preventing unwanted attributes at CFG=1.0 without running 2 forward passes.
- An optional negative attention bias `offset` (e.g. `-0.1` or `-0.2`) on the negative key tokens prevents leakage when scale=0.

## 3. ComfyUI Node Design
Implement modular ComfyUI nodes:
1. `AnimaVSFPatcher`: Takes `model` (MODEL), `positive` (CONDITIONING), `negative` (CONDITIONING), `scale` (FLOAT, default 1.5), `offset` (FLOAT, default 0.0).
   Returns `patched_model` (MODEL), `fused_conditioning` (CONDITIONING).
2. Works seamlessly with standard `KSampler` at `cfg=1.0`.
3. Does NOT duplicate image tokens or double NFE (pure B=1 execution).

## 4. Git Identity Requirements
- Commits must be authored by:
  Author: kobeniw <kobeniw@users.noreply.github.com>
- Git push uses SSH key: `~/.ssh/id_ed25519_agent` via `git@github.com-agent:ruwwww/VSF.git`.
