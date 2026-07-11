#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Ascend implementation of upstream :class:`MMEncoderAttention`.

FIA remains the general path and uses ``graph_task_group_begin/end`` during
ACL-graph capture. The opt-in Qwen3.5-VL ``head_dim=72`` path uses Triton at
the native dimension and reads replay-time sequence metadata from device.
"""

from __future__ import annotations

import einops
import torch
import torch.nn.functional as F
import torch_npu
from vllm.logger import logger
from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention  # type: ignore
from vllm.triton_utils import HAS_TRITON

from vllm_ascend import envs
from vllm_ascend.utils import weak_ref_tensors
from vllm_ascend.worker.encoder_acl_graph import (
    get_encoder_forward_context,
    get_encoder_graph_params,
    maybe_compute_actual_seq_lengths,
    update_encoder_graph_workspace,
)

MIN_PAD_SIZE: int = 64
MAX_PAD_SIZE: int = 128
SWA_INT_MAX: int = 2147483647
FIA_BLOCK_SIZE: int = 128


class AscendMMEncoderAttention(MMEncoderAttention):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float | None = None,
        num_kv_heads: int | None = None,
        prefix: str = "",
    ) -> None:
        """
        Args:
            num_heads: number of attention heads per partition.
            head_size: hidden_size per attention head.
            scale: scale factor.
            num_kv_heads: number of kv heads.
            prefix: This has no effect, it is only here to make it easier to
                    swap between Attention and MMEncoderAttention.
            multimodal_config: configs for multi-modal.
        """
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            prefix=prefix,
        )

        self.enable_pad = self.head_size > MIN_PAD_SIZE and self.head_size < MAX_PAD_SIZE
        if self.head_size == 72:
            logger.info_once(
                "[ViT Triton FA] head_dim=72 configuration: "
                "VLLM_ASCEND_ENABLE_VIT_TRITON_FA=%s, HAS_TRITON=%s. "
                "The optimized path requires NPU BF16 inputs.",
                envs.VLLM_ASCEND_ENABLE_VIT_TRITON_FA,
                HAS_TRITON,
            )

    def _reshape_qkv_to_3d(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        bsz: int,
        q_len: int,
        kv_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reshape query, key, value to 3D tensors:
        (batch_size * seq_len, num_heads, head_size)
        """
        query = query.view(bsz * q_len, self.num_heads, self.head_size)
        key = key.view(bsz * kv_len, self.num_kv_heads, self.head_size)
        value = value.view(bsz * kv_len, self.num_kv_heads, self.head_size)
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        if (num_repeat := self.num_queries_per_kv) > 1:
            # Handle MQA and GQA
            key = torch.repeat_interleave(key, num_repeat, dim=1)
            value = torch.repeat_interleave(value, num_repeat, dim=1)

        return query, key, value

    def _maybe_pad_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int | None]:
        if not self.enable_pad:
            return q, k, v, None
        origin_head_dim = q.shape[-1]
        pad_len = MAX_PAD_SIZE - origin_head_dim
        q = F.pad(q, (0, pad_len), mode="constant", value=0)
        k = F.pad(k, (0, pad_len), mode="constant", value=0)
        v = F.pad(v, (0, pad_len), mode="constant", value=0)
        return q, k, v, origin_head_dim

    def _maybe_compute_cu_seqlens(
        self,
        bsz: int,
        q_len: int,
        cu_seqlens: torch.Tensor | None,
        *,
        is_capturing: bool = False,
    ) -> torch.Tensor:
        # In the eager path, if cu_seqlens is provided by the model we use it; if it is not provided, we create a
        # default one assuming all sequences have the same length. This is used by models such as Hunyuan-OCR, which
        # always pass None as cu_seqlens and rely on the operator to compute it internally.
        # In the capture path, we always create the default cu_seqlens on CPU instead of using the model tensor, to
        # avoid a device-to-host sync (.cpu()).
        if is_capturing or cu_seqlens is None:
            cu_seqlens = torch.arange(0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device="cpu")
            return cu_seqlens

        return cu_seqlens.cpu()

    @staticmethod
    def _maybe_unpad_output(
        context_layer: torch.Tensor,
        origin_head_dim: int | None,
    ) -> torch.Tensor:
        if origin_head_dim is not None:
            return context_layer[..., :origin_head_dim]
        return context_layer

    @staticmethod
    def _restore_batch_layout(
        context_layer: torch.Tensor,
        *,
        bsz: int,
        q_len: int,
        is_reshaped: bool,
    ) -> torch.Tensor:
        if is_reshaped:
            return einops.rearrange(context_layer, "(b s) h d -> b s h d", b=bsz, s=q_len).contiguous()
        return einops.rearrange(context_layer, "(b s) h d -> b s (h d)", b=bsz, s=q_len).contiguous()

    def _run_vit_fia(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        actual_seq_lengths_q: list[int],
        actual_seq_lengths_kv: list[int],
        *,
        out: torch.Tensor | None = None,
        softmax_lse: torch.Tensor | None = None,
        workspace: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fia_kwargs = dict(
            query=query,
            key=key,
            value=value,
            atten_mask=None,
            block_table=None,
            input_layout="TND",
            block_size=FIA_BLOCK_SIZE,
            actual_seq_lengths=actual_seq_lengths_q,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=0,
            pre_tokens=SWA_INT_MAX,
            next_tokens=SWA_INT_MAX,
        )
        if out is None:
            context_layer, _ = torch_npu.npu_fused_infer_attention_score(**fia_kwargs)
            return context_layer
        if workspace is None:
            workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(**fia_kwargs)
        if softmax_lse is None:
            softmax_lse = torch.empty(1, dtype=query.dtype, device=query.device)
        torch_npu.npu_fused_infer_attention_score.out(
            workspace=workspace,
            out=[out, softmax_lse],
            **fia_kwargs,
        )
        return out

    def _run_vit_triton(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor,
        out: torch.Tensor | None = None,
        *,
        full_aligned: bool = False,
    ) -> torch.Tensor:
        """Triton FA path for head_dim FIA does not support natively (e.g. 72).

        Runs at the native head_dim -- no 72 -> 128 pad/unpad. ``cu_seqlens`` is
        the full ``[0, L0, L0+L1, ...]`` device tensor from ``forward_oot``
        (with leading 0); it is *not* the cumulative list from
        ``maybe_compute_actual_seq_lengths``.
        """
        from vllm_ascend.ops.triton.vit_flash_attention import (
            VIT_FA_A5_FULL_ALIGNED_CONFIG,
            get_vit_flash_attention_config,
            vit_flash_attention_fwd,
        )

        config = get_vit_flash_attention_config(full_aligned=full_aligned)
        full_aligned = full_aligned and config == VIT_FA_A5_FULL_ALIGNED_CONFIG
        if cu_seqlens.device != query.device:
            cu_seqlens = cu_seqlens.to(query.device)
        return vit_flash_attention_fwd(
            query,
            key,
            value,
            cu_seqlens,
            scale=self.scale,
            out=out,
            block_m=config.block_m,
            block_n=config.block_n,
            qk_pad=config.qk_pad,
            v_pad=config.v_pad,
            full_aligned=full_aligned,
            clear_tail=out is not None,
        )

    def _can_use_vit_triton_fa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
    ) -> bool:
        """Return whether this invocation is supported by the specialized kernel."""
        if not envs.VLLM_ASCEND_ENABLE_VIT_TRITON_FA or not HAS_TRITON:
            return False
        if self.head_size != 72 or cu_seqlens is None:
            return False
        if query.dim() != 3 or key.dim() != 3 or value.dim() != 3:
            return False
        if query.shape != key.shape or query.shape != value.shape or query.shape[-1] != 72:
            return False
        if query.dtype != torch.bfloat16 or key.dtype != query.dtype or value.dtype != query.dtype:
            return False
        if query.device != key.device or query.device != value.device or query.device.type != "npu":
            return False
        return cu_seqlens.dim() == 1 and cu_seqlens.dtype in (torch.int32, torch.int64)

    def _log_vit_triton_fa_fallback(self) -> None:
        if envs.VLLM_ASCEND_ENABLE_VIT_TRITON_FA and self.head_size == 72:
            logger.warning_once(
                "[ViT Triton FA] Requested but not selected for this invocation; "
                "falling back to FIA pad-to-128. Required: Triton, NPU BF16 "
                "Q/K/V shaped [T, H, 72], and int32/int64 cu_seqlens."
            )

    def _forward_eager_fia(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
        is_reshaped: bool,
        bsz: int,
        q_len: int,
    ) -> torch.Tensor:
        # Triton FA path for non-FIA-aligned head_dim (e.g. Qwen3.5-VL 72):
        # skip the 72 -> 128 pad/unpad, run the Triton kernel at native head_dim.
        # Gated to head_size==72 so other head_dims keep the FIA path. The
        # capture path (_forward_capture_fia) has its own triton branch.
        if self._can_use_vit_triton_fa(query, key, value, cu_seqlens):
            assert cu_seqlens is not None
            from vllm_ascend.ops.triton.vit_flash_attention import (
                A5_FULL_ALIGNED_HEADS,
                A5_FULL_ALIGNED_TOKENS,
            )

            # Qwen3.5 constructs cu_seqlens from the same grid metadata as the
            # query. In eager mode, two endpoints therefore prove one full
            # sequence without reading device data. This requests the aligned
            # shape contract; the kernel wrapper currently keeps constexpr-T
            # disabled behind its precision gate and runs the verified
            # dynamic-T A5 tile. Videos and packed images use the varlen path.
            # Capture has a separate branch and never requests this contract.
            full_aligned = (
                cu_seqlens.numel() == 2
                and bsz == 1
                and q_len == A5_FULL_ALIGNED_TOKENS
                and query.shape[:2] == (A5_FULL_ALIGNED_TOKENS, A5_FULL_ALIGNED_HEADS)
            )
            context_layer = self._run_vit_triton(
                query,
                key,
                value,
                cu_seqlens,
                full_aligned=full_aligned,
            )
            return self._restore_batch_layout(
                context_layer,
                bsz=bsz,
                q_len=q_len,
                is_reshaped=is_reshaped,
            )
        self._log_vit_triton_fa_fallback()
        actual_seq_lengths_q, actual_seq_lengths_kv = maybe_compute_actual_seq_lengths(
            self._maybe_compute_cu_seqlens(bsz, q_len, cu_seqlens),
            query.shape[0],
            key.shape[0],
            cudagraph_mm_encoder=False,
        )
        q, k, v, origin_head_dim = self._maybe_pad_qkv(query, key, value)
        context_layer = self._run_vit_fia(q, k, v, actual_seq_lengths_q, actual_seq_lengths_kv)
        context_layer = self._maybe_unpad_output(context_layer, origin_head_dim)
        return self._restore_batch_layout(
            context_layer,
            bsz=bsz,
            q_len=q_len,
            is_reshaped=is_reshaped,
        )

    def _forward_capture_fia(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
        is_reshaped: bool,
        bsz: int,
        q_len: int,
    ) -> torch.Tensor:
        context = get_encoder_forward_context()
        token_budget = context.token_budget
        is_capturing = context.capturing
        params = get_encoder_graph_params()
        if token_budget is None or params is None:
            raise RuntimeError("Encoder graph capture state was not initialized (missing token_budget).")

        if (
            self._can_use_vit_triton_fa(query, key, value, cu_seqlens)
            and cu_seqlens is not None
            and cu_seqlens.device == query.device
        ):
            # Triton FA capture path: run the kernel at native head_dim (no
            # 72->128 pad). Triton reads device cu_seqlens (whose content is
            # refreshed by _run_budget_graph before graph.replay()), so it does
            # NOT use FIA's graph_task_group/graph_task_update rebind and does
            # NOT pack attn_params/handles/events. `out` is pre-allocated here
            # (its address is baked into the graph; replay writes it).
            # task_num is baked from the fixed query capacity, not the encoder
            # token budget (Qwen3.5-VL attention runs before patch merging).
            # The extra num_seqs-1 term covers one tail block per sequence.
            # The Triton launch clears only the fixed-capacity tail outside the
            # packed sequences, avoiding a full-output zero node per layer.
            out = torch.empty_like(query, memory_format=torch.contiguous_format)
            self._run_vit_triton(
                query,
                key,
                value,
                cu_seqlens,
                out=out,
                full_aligned=False,
            )
            return self._restore_batch_layout(
                out,
                bsz=bsz,
                q_len=q_len,
                is_reshaped=is_reshaped,
            )

        self._log_vit_triton_fa_fallback()
        actual_seq_lengths_q, actual_seq_lengths_kv = maybe_compute_actual_seq_lengths(
            self._maybe_compute_cu_seqlens(bsz, q_len, cu_seqlens, is_capturing=is_capturing),
            query.shape[0],
            key.shape[0],
            cudagraph_mm_encoder=True,
        )
        q, k, v, origin_head_dim = self._maybe_pad_qkv(query, key, value)

        out = torch.empty_like(q)
        softmax_lse = torch.empty(1, dtype=q.dtype, device=q.device)

        workspace = params.workspaces.get(token_budget)
        if workspace is None:
            workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                query=q,
                key=k,
                value=v,
                atten_mask=None,
                block_table=None,
                input_layout="TND",
                block_size=FIA_BLOCK_SIZE,
                actual_seq_lengths=actual_seq_lengths_q,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                sparse_mode=0,
                scale=self.scale,
                pre_tokens=SWA_INT_MAX,
                next_tokens=SWA_INT_MAX,
            )
            update_encoder_graph_workspace(token_budget, workspace)

        stream = torch_npu.npu.current_stream()
        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)

        torch.npu.graph_task_group_begin(stream)
        self._run_vit_fia(
            q,
            k,
            v,
            actual_seq_lengths_q,
            actual_seq_lengths_kv,
            out=out,
            softmax_lse=softmax_lse,
            workspace=workspace,
        )
        handle = torch.npu.graph_task_group_end(stream)

        packed = (
            weak_ref_tensors(q),
            weak_ref_tensors(k),
            weak_ref_tensors(v),
            None,
            None,
            FIA_BLOCK_SIZE,
            self.num_kv_heads,
            self.num_heads,
            self.scale,
            weak_ref_tensors(out),
            weak_ref_tensors(softmax_lse),
        )
        params.attn_params[token_budget].append(packed)
        params.events[token_budget].append(event)
        params.handles[token_budget].append(handle)

        context_layer = self._maybe_unpad_output(out, origin_head_dim)
        return self._restore_batch_layout(
            context_layer,
            bsz=bsz,
            q_len=q_len,
            is_reshaped=is_reshaped,
        )

    def forward_oot(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
        sequence_lengths: torch.Tensor | None = None,
    ):
        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)
        is_reshaped = query.dim() == 4

        q, k, v = self._reshape_qkv_to_3d(query, key, value, bsz, q_len, kv_len)

        if get_encoder_forward_context().capturing:
            return self._forward_capture_fia(
                q,
                k,
                v,
                cu_seqlens=cu_seqlens,
                is_reshaped=is_reshaped,
                bsz=bsz,
                q_len=q_len,
            )

        return self._forward_eager_fia(
            q,
            k,
            v,
            cu_seqlens=cu_seqlens,
            is_reshaped=is_reshaped,
            bsz=bsz,
            q_len=q_len,
        )
