# Implementation Plan: Anima VSF ComfyUI Nodes

## Tasks
1. [ ] Implement `comfyui/custom_nodes/value_sign_flip/src/value_sign_flip/anima.py` containing:
   - Conditioning fusion helper: encodes or merges `positive` and `negative` conditioning tensors, tracks `pos_len` and `neg_len`.
   - `AnimaVSFAttnProcessor` or custom model patcher hook for Anima's `cross_attn`.
   - Modulates text Key/Value projections such that `V[:, pos_len:, :] *= -scale`, and applies attention mask offset if `offset != 0.0`.
2. [ ] Expose `AnimaVSFApply` and `AnimaVSFAdvanced` in `comfyui/custom_nodes/value_sign_flip/src/value_sign_flip/nodes.py`.
   - Update `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`.
3. [ ] Add unit and contract tests in `comfyui/custom_nodes/value_sign_flip/tests/test_anima_vsf.py`.
4. [ ] Verify test suite passing using pytest with `/home/kuroko/.conda/envs/ai/bin/pytest`.
5. [ ] Commit with author `kobeniw <kobeniw@users.noreply.github.com>`.
