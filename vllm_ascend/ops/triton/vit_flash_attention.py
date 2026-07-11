# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Triton flash-attention for the Qwen3.5-VL vision encoder (head_dim = 72).

Why this kernel
---------------
Qwen3.5-VL's vision encoder uses ``head_dim = hidden_size // num_heads =
1152 // 16 = 72`` (see ``Qwen3_5VisionConfig`` in vllm). 72 is not a multiple of
16, so CANN's ``npu_fused_infer_attention_score`` (FIA), which only accepts
head_dim aligned to 128, cannot run on it directly. The existing path in
``mm_encoder_attention.py`` physically zero-pads 72 -> 128, runs FIA, then
slices back to 72. This kernel keeps the GM input/output at the real D=72 and
performs any alignment padding only in its on-chip blocks, avoiding the three
pad kernels and the materialized output slice.

How 72 is handled
-----------------
triton-ascend's ``tl.dot`` does *not* require the contracting dim to be a power
of two or a multiple of 16 (verified: ``DotOp::verifyDims`` only checks K
matches; the arange power-of-two gate is SIMT-only and Ascend defaults to simd).
So 72 *could* be used directly, but 72 = 16x4 + 8 leaves the last 16-element
cube tile half-empty. We instead use compile-time logical widths **without
physically padding the tensor**. Q/K use ``D_QK_PAD=80``, the smallest
16-element-aligned width covering D=72. V/output use ``D_V_PAD``: 80 on generic
SoCs, 88 on the A5 BM64 varlen path, and 96 on the A5 BM128 aligned path. V128
spills A5 UB in the aligned path, while V80 regressed the measured packed path;
V88 reduced the measured packed latency without changing the GM layout. Every
``make_block_ptr`` keeps its parent
*shape* at the real D=72, so ``boundary_check`` zero-fills input columns beyond
72 and drops output columns beyond 72 without a tensor slice or physical pad.

Why dynamic enumeration (no host chunk_indices)
-----------------------------------------------
The varlen q-block -> (sequence, intra-seq-block) mapping is computed **inside
the kernel** by scanning the device ``cu_seqlens`` tensor, NOT pre-computed on
host via ``prepare_chunk_indices``. Reason: that helper does a ``.tolist()``
D2H sync, which is illegal under ``torch.npu.graph`` capture, and its output
shape (num_q_blocks) changes per request while graph launch args must stay
fixed. With dynamic enumeration, ``task_num`` uses the static, device-free
upper bound ``(ceil(T_capacity / BM) + num_seqs - 1) * H``. This covers the
per-sequence tail blocks without reading lengths back to the host, and the
kernel skips tasks beyond the real total q-blocks (padding sequences have
length 0 and contribute no blocks). This makes eager and capture share the same
kernel, and replay works because only the ``cu_seqlens`` **content** needs
refreshing (already done by ``_run_budget_graph``'s ``buf.copy_(src)`` before
``graph.replay()``).

The A5 ``T=1024, H=16, cu_seqlens=[0, T]`` eager case uses
BM128/BN128/QK80/V96. A former compile-time ``T=1024`` specialization changed
the backend's numerical lowering and failed the NPU precision gate, so it keeps
``T`` dynamic. A5 packed, tail, and graph-replay inputs use the measured
BM64/BN128/QK80/V88 tile; other SoCs retain BM64/BN64/QK80/V80 until
separately validated.

Scope
-----
* bf16 only (vision attention always sees bf16 Q/K/V; MxFP8 lives in the linear
  projections and is dequantized before attention).
* varlen: packed sequences via ``cu_seqlens`` (shape fixed per token_budget,
  padded to max_batch_size+1 with trailing 0-length sequences).
* full bidirectional attention (no causal, no SWA). RoPE is applied upstream.
* forward only (vision encoder is forward-only in vllm).
* both eager and ACL-graph capture paths (capture relies on the dynamic
  enumeration above + device cu_seqlens content refresh; triton does NOT use
  FIA's ``graph_task_group``/``graph_task_update`` rebind -- that is host-list
  specific).

The online-softmax structure follows triton-ascend's
``tutorials/04-fused-attention.py`` (``exp2`` with ``scale * 1/log(2)``
pre-multiplication). The core-loop scheduling follows
``fla/chunk_scaled_dot_kkt.py``.

BM/BN/D_QK_PAD/D_V_PAD use SoC-specific production defaults. See the tuning
note at the bottom for the hardware benchmark that should be used before
changing them.
"""

import math
from functools import lru_cache
from typing import NamedTuple

import torch
import torch_npu
from vllm.logger import logger
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.fla.utils import input_guard
from vllm_ascend.ops.triton.triton_utils import (
    get_aicore_num,
    init_device_properties_triton,
)
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

TAIL_CLEAR_BLOCK: int = 256


class VitFlashAttentionConfig(NamedTuple):
    """Compile-time tile configuration for the vision attention kernel.

    ``qk_pad`` is the QK reduction width; ``v_pad`` is the PV/output width.
    Both are logical on-chip widths and never change the physical tensors.
    """

    block_m: int
    block_n: int
    qk_pad: int
    v_pad: int


VIT_FA_GENERIC_CONFIG = VitFlashAttentionConfig(block_m=64, block_n=64, qk_pad=80, v_pad=80)
VIT_FA_A5_CONFIG = VitFlashAttentionConfig(
    # Packed [512,512] A5 V88 candidate: five-process epoch-speedup median
    # 1.2018x and absolute-p50-ratio median 1.1974x, versus 1.0964x/1.1033x
    # for the immediately preceding V96 baseline. V80 previously regressed;
    # V88 is the measured middle point. A subsequent BN256 single-variable
    # candidate did not improve A5 performance, so retain BM64/BN128/QK80.
    block_m=64,
    block_n=128,
    qk_pad=80,
    # V88 applies only to the BM64 varlen/graph config. The BM128 aligned path
    # retains V96 below. Graph capture/replay uses this config via
    # full_aligned=False and must pass the full replay precision suite.
    v_pad=88,
)
VIT_FA_A5_FULL_ALIGNED_CONFIG = VitFlashAttentionConfig(
    block_m=128,
    block_n=128,
    qk_pad=80,
    # V96 (not V128) for the exact BM128/BN128 path: D_V_PAD=128 spills A5 UB at
    # this tile (MLIRCompilationError), while 96 fits. 96 is also faster than
    # 128 would be: columns 72..D_V_PAD of V are zero (boundary_check), so a
    # wider V tile only adds wasted PV cube work. 96 = 16*6 keeps full cube
    # tiles and minimizes the wasted tail. Measured ~1.20x over FIA in isolated
    # steady-state timing (Triton p50 0.088ms vs FIA 0.106ms) on A5.
    v_pad=96,
)
A5_FULL_ALIGNED_TOKENS: int = 1024
A5_FULL_ALIGNED_HEADS: int = 16
# The exact-T lowering is intentionally gated off: it is the only remaining
# A5 NPU precision failure while the same BM128/BN128/QK80/V96 kernel with a
# dynamic T passes. Keep the implementation available for future compiler
# validation, but never select it in production or the benchmark by default.
A5_EXACT_T_SPECIALIZATION_ENABLED: bool = False


@lru_cache(maxsize=2)
def get_vit_flash_attention_config(*, full_aligned: bool = False) -> VitFlashAttentionConfig:
    """Return the production tile configuration for the installed NPU SoC.

    A5 packed/varlen/graph inputs use the measured BM64/BN128/QK80/V88 tile.
    The ``full_aligned`` flag selects BM128/BN128/QK80/V96 for the contract-
    checked T=1024/H=16 eager shape, while its exact-T lowering remains behind
    the precision gate. Other SoCs retain the conservative configuration until
    they have NPU measurements.
    """
    try:
        device_type = get_ascend_device_type()
    except ImportError:
        # ``_build_info.py`` is generated by packaging and intentionally not
        # tracked. Standalone source-tree benchmarks commonly run through
        # PYTHONPATH before an editable install, so fall back to the actual SoC.
        is_a5 = torch_npu.npu.get_soc_version() == 260
        if is_a5:
            return VIT_FA_A5_FULL_ALIGNED_CONFIG if full_aligned else VIT_FA_A5_CONFIG
        return VIT_FA_GENERIC_CONFIG
    if device_type == AscendDeviceType.A5:
        return VIT_FA_A5_FULL_ALIGNED_CONFIG if full_aligned else VIT_FA_A5_CONFIG
    return VIT_FA_GENERIC_CONFIG


def _max_varlen_task_num(
    total_tokens: int,
    num_seqs: int,
    num_heads: int,
    block_m: int,
) -> int:
    """Return a device-free upper bound for varlen q-block/head tasks.

    For non-negative sequence lengths ``L_i`` whose sum does not exceed
    ``total_tokens``::

        sum(ceil(L_i / block_m))
            <= ceil(total_tokens / block_m) + num_seqs - 1

    The extra ``num_seqs - 1`` term is required because each sequence can have
    its own partial tail block. Keeping this calculation shape-only avoids a
    D2H read of ``cu_seqlens`` and is therefore safe during graph capture.
    """
    # Keep this as pure shape arithmetic. In the encoder graph path these may
    # be SymInts, and Python comparisons would add unnecessary compile guards.
    max_q_blocks = (total_tokens + block_m - 1) // block_m + num_seqs - 1
    return max_q_blocks * num_heads


@triton.jit(do_not_specialize=["T", "task_num"])
def _vit_fa_fwd_kernel(
    q,
    k,
    v,
    out,
    cu_seqlens,
    T,
    num_heads: tl.constexpr,
    task_num,
    num_core,
    NUM_SEQS: tl.constexpr,
    scale,
    D: tl.constexpr,
    D_QK_PAD: tl.constexpr,
    D_V_PAD: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    CLEAR_BLOCK: tl.constexpr,
    SINGLE_SEQUENCE: tl.constexpr,
    TWO_SEQUENCES: tl.constexpr,
    CLEAR_TAIL: tl.constexpr,
    EXACT_T: tl.constexpr,
):
    """One program iteration computes one (head, global-q-block) tile.

    Grid is (num_core,); each core strides over tasks (same pattern as
    chunk_scaled_dot_kkt). A task encodes (global_q_block, head):
        gqb = task_id // num_heads   -> global q-block index across all seqs
        i_h = task_id %  num_heads   -> head
    The general packed path scans ``cu_seqlens`` to map ``gqb`` to (sequence,
    intra-seq q-block); one-sequence metadata is compile-time specialized.
    Tasks beyond the real total q-blocks (padding seqs have length 0) are
    skipped via a scalar ``if``.
    """
    core_id = tl.program_id(0)
    hd = num_heads * D                       # token stride in TND [T, H, D]
    qk_scale = scale * 1.4426950408889634    # scale / log(2), for exp2

    # Exact eager inputs have no capacity tail and do not need device sequence
    # metadata. General eager inputs also have T == valid_tokens, but retain the
    # endpoint load for packed-sequence mapping. Graph inputs keep CLEAR_TAIL so
    # replay can zero the fixed-capacity suffix after shorter requests.
    if EXACT_T:
        valid_tokens = EXACT_T
    else:
        valid_tokens = tl.load(cu_seqlens + NUM_SEQS).to(tl.int32)
    if CLEAR_TAIL:
        tail_numel = (T - valid_tokens) * hd
        clear_offsets = tl.arange(0, CLEAR_BLOCK)
        for clear_block_id in tl.range(core_id, tl.cdiv(tail_numel, CLEAR_BLOCK), num_core):
            offsets = clear_block_id * CLEAR_BLOCK + clear_offsets
            tl.store(out + valid_tokens * hd + offsets, 0.0, mask=offsets < tail_numel)

    if SINGLE_SEQUENCE:
        # EXACT_T makes this a fully compile-time task count. The general
        # single-sequence path still reads the replay-time endpoint.
        effective_task_num = tl.cdiv(valid_tokens, BM) * num_heads
    elif TWO_SEQUENCES:
        # The common eager packed case can map q-blocks directly from the one
        # interior endpoint. This removes both the conservative extra q-block
        # and the per-task linear metadata scan.
        first_seq_end = tl.load(cu_seqlens + 1).to(tl.int32)
        first_seq_blocks = tl.cdiv(first_seq_end, BM)
        second_seq_blocks = tl.cdiv(valid_tokens - first_seq_end, BM)
        effective_task_num = (first_seq_blocks + second_seq_blocks) * num_heads
    else:
        # A common encoder-graph layout has one real image followed by fixed-
        # shape zero-length slots: [0, valid, valid, ...]. Detect it on device
        # so replay can switch layouts without a D2H synchronization.
        first_seq_end = tl.load(cu_seqlens + 1).to(tl.int32)
        single_active_seq = first_seq_end == valid_tokens
        single_seq_task_num = tl.cdiv(valid_tokens, BM) * num_heads
        effective_task_num = tl.where(single_active_seq, single_seq_task_num, task_num).to(tl.int32)

    for task_id in tl.range(core_id, effective_task_num, num_core):
        i_h = task_id % num_heads
        gqb = task_id // num_heads

        # Dynamic enumeration: find which sequence owns global q-block `gqb`.
        # No break/continue here (control-flow portability); use scalar if.
        i_n = -1
        i_t = 0
        bos = 0
        eos = 0
        if SINGLE_SEQUENCE:
            i_n = 0
            i_t = gqb
            bos = 0
            eos = valid_tokens
        elif TWO_SEQUENCES:
            if gqb < first_seq_blocks:
                i_n = 0
                i_t = gqb
                bos = 0
                eos = first_seq_end
            else:
                i_n = 1
                i_t = gqb - first_seq_blocks
                bos = first_seq_end
                eos = valid_tokens
        else:
            if single_active_seq:
                i_n = 0
                i_t = gqb
                bos = 0
                eos = valid_tokens
            else:
                acc_blocks = 0
                # NUM_SEQS is a launch-time constexpr because the endpoint
                # buffer shape is fixed during eager execution and graph
                # capture/replay. A runtime Python ``range(num_seqs)`` caused
                # AICore timeout 507014 on Triton-Ascend 3.2.1 for the first
                # three-sequence input ([1, 65, 127]). Static expansion keeps
                # the same device-only mapping without an unbounded device
                # control-flow loop. Rolling endpoints reduce metadata loads
                # from 2*NUM_SEQS+2 per task to NUM_SEQS+1.
                bos_s = tl.load(cu_seqlens).to(tl.int32)
                for s in tl.static_range(0, NUM_SEQS):
                    eos_s = tl.load(cu_seqlens + s + 1).to(tl.int32)
                    blocks_s = (eos_s - bos_s + BM - 1) // BM
                    if i_n == -1:
                        if gqb < acc_blocks + blocks_s:
                            i_n = s
                            i_t = gqb - acc_blocks
                            bos = bos_s
                            eos = eos_s
                    acc_blocks = acc_blocks + blocks_s
                    bos_s = eos_s

        # Skip tasks beyond real total q-blocks (padding sequences / overflow).
        if i_n != -1:
            if EXACT_T:
                T_seq = EXACT_T
            else:
                T_seq = eos - bos
            q_start = i_t * BM

            # TND [T, H, D]: for fixed head i_h starting at token bos, base
            # pointer is ptr + (bos*num_heads + i_h)*D, then (t_in_seq, d)
            # strides (H*D, 1).
            base = (bos * num_heads + i_h) * D
            qh = q + base
            kh = k + base
            vh = v + base
            oh = out + base

            # Q/K use the narrow logical width. The parent shape stays at the
            # REAL D=72, so boundary_check zero-fills columns D..D_QK_PAD-1
            # instead of reading the next head's data. Q stays resident.
            p_q = tl.make_block_ptr(
                qh,
                (T_seq, D),
                (hd, 1),
                (q_start, 0),
                (BM, D_QK_PAD),
                (1, 0),
            )
            if EXACT_T:
                # T=1024 is divisible by BM=128. Only the logical D padding
                # needs a boundary check in this specialization.
                b_q = tl.load(p_q, boundary_check=(1,), padding_option="zero")
            else:
                b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")

            m_i = tl.full([BM], float("-inf"), dtype=tl.float32)
            l_i = tl.zeros([BM], dtype=tl.float32)
            acc = tl.zeros([BM, D_V_PAD], dtype=tl.float32)

            # Full bidirectional: iterate all KV blocks within this sequence.
            for start_n in range(0, T_seq, BN):
                p_k = tl.make_block_ptr(
                    kh,
                    (T_seq, D),
                    (hd, 1),
                    (start_n, 0),
                    (BN, D_QK_PAD),
                    (1, 0),
                )
                p_v = tl.make_block_ptr(
                    vh,
                    (T_seq, D),
                    (hd, 1),
                    (start_n, 0),
                    (BN, D_V_PAD),
                    (1, 0),
                )
                if EXACT_T:
                    b_k = tl.load(
                        p_k,
                        boundary_check=(1,),
                        padding_option="zero",
                    )
                else:
                    b_k = tl.load(
                        p_k,
                        boundary_check=(0, 1),
                        padding_option="zero",
                    )  # [BN, D_QK_PAD]

                qk = tl.dot(b_q, tl.trans(b_k))  # [BM, BN], K=D_QK_PAD

                # Only the final partial block needs an OOB mask. Applying it
                # to every full block is especially expensive on Ascend: an
                # integer compare inside tl.where can fall back to scalar
                # instructions. Cast the tail positions to fp32 so the vector
                # compare path remains available.
                if start_n + BN > T_seq:
                    n_pos = (start_n + tl.arange(0, BN)).to(tl.float32)
                    qk = tl.where(n_pos[None, :] < T_seq, qk, float("-inf"))

                # online softmax
                m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)  # [BM]
                alpha = tl.math.exp2(m_i - m_ij)           # [BM]
                p = tl.math.exp2(qk * qk_scale - m_ij[:, None])  # [BM, BN]
                l_i = l_i * alpha + tl.sum(p, 1)
                # Load V only after P is formed to shorten its live range.
                if EXACT_T:
                    b_v = tl.load(
                        p_v,
                        boundary_check=(1,),
                        padding_option="zero",
                    )
                else:
                    b_v = tl.load(
                        p_v,
                        boundary_check=(0, 1),
                        padding_option="zero",
                    )  # [BN, D_V_PAD]
                acc = acc * alpha[:, None]
                # Pass acc into tl.dot so the backend can retain the PV result
                # in L0C instead of materializing a second result for a vector
                # add. This matches Triton-Ascend's fused-attention tutorial.
                acc = tl.dot(p.to(b_v.dtype), b_v, acc)
                m_i = m_ij

            # Make the per-row reciprocal explicit. This prevents a backend
            # from lowering the broadcast division to BM*D_V_PAD fp32 divides.
            inv_l = 1.0 / l_i
            acc = acc * inv_l[:, None]

            # Keep the parent shape at the REAL D=72 while using a D_V_PAD
            # block. boundary_check drops columns D..D_V_PAD-1, so the store
            # cannot clobber the next head. Storing the full accumulator also
            # avoids partial tensor slicing, which Triton-Ascend 3.2.1 does
            # not support.
            p_o = tl.make_block_ptr(
                oh,
                (T_seq, D),
                (hd, 1),
                (q_start, 0),
                (BM, D_V_PAD),
                (1, 0),
            )
            if EXACT_T:
                tl.store(p_o, acc.to(p_o.dtype.element_ty), boundary_check=(1,))
            else:
                tl.store(p_o, acc.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@input_guard
def vit_flash_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    *,
    out: torch.Tensor | None = None,
    max_task_num: int | None = None,
    num_seqs: int | None = None,
    block_m: int = 64,
    block_n: int = 64,
    d_pad: int | None = None,
    qk_pad: int | None = None,
    v_pad: int | None = None,
    full_aligned: bool = False,
    clear_tail: bool = True,
    one_program_per_task: bool = False,
) -> torch.Tensor:
    """Flash-attention forward for the Qwen3.5-VL vision encoder (head_dim = 72).

    Args:
        q, k, v: ``[T, H, 72]`` TND, bf16. ``k``/``v`` must already be repeated
            to ``H`` heads (GQA broadcast done by the caller, matching FIA).
        cu_seqlens: ``[B + 1]`` cumulative sequence boundaries **with leading 0**
            (``[0, L0, L0+L1, ...]``), on device. Under capture this is the
            fixed-shape device buffer whose **content** is refreshed before
            replay.
        scale: ``head_dim ** -0.5``.
        out: optional pre-allocated contiguous output (capture path; address
            must stay stable across capture/replay). If None, allocated
            internally (eager).
        max_task_num: total tasks = global q-blocks * H. If None, computed from
            the static safe upper bound
            ``(ceil(T / block_m) + num_seqs - 1) * H``. An explicit override
            must not be smaller than this safe bound.
        num_seqs: number of sequences (incl. padding seqs). If None,
            ``cu_seqlens.shape[0] - 1``.
        block_m, block_n: q / kv block sizes. See tuning note below.
        d_pad: backward-compatible uniform logical width for Q/K/V/output.
            It cannot be combined with ``qk_pad`` or ``v_pad``.
        qk_pad: optional QK reduction width used for offline tuning.
        v_pad: optional V/PV/output width used for offline tuning. Independent
            widths default to the next multiple of 16 (80 for D=72); each must
            be at least D and a multiple of 8.
        full_aligned: identify the contract-checked A5 T=1024/H=16 eager shape.
            Its compile-time-T lowering is currently disabled by the precision
            gate, so it uses the verified dynamic-T kernel. Never request this
            shape contract for graph capture/replay.
        clear_tail: clear output rows after ``cu_seqlens[-1]``. Graph replay
            requires this for fixed-capacity outputs. Production eager inputs
            have no capacity suffix and pass False so the clear loop is
            removed at compile time. Defaults to True for standalone callers.
        one_program_per_task: experimental eager-only scheduler used by the
            offline benchmark. False keeps the production persistent-AICore
            loop. True launches one Triton program for each safe task slot.

    Returns:
        ``[T, H, 72]`` bf16 context tensor (same tensor as ``out`` if given).
    """
    T, H, D = q.shape
    assert q.shape == k.shape == v.shape, "q/k/v shape mismatch"
    assert q.device == k.device == v.device, "q/k/v device mismatch"
    assert D == 72, f"vit_flash_attention_fwd expects head_dim=72, got {D}"
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16, "bf16 only"
    assert scale > 0.0 and math.isfinite(scale), f"scale must be positive and finite, got {scale}"
    assert cu_seqlens.dim() == 1, "cu_seqlens must be 1D [B+1]"
    assert cu_seqlens.dtype in (torch.int32, torch.int64), "cu_seqlens must be int32 or int64"
    assert cu_seqlens.device == q.device, "cu_seqlens must be on the same device as q/k/v"

    default_pad = ((D + 15) // 16) * 16
    if d_pad is not None:
        assert qk_pad is None and v_pad is None, "d_pad cannot be combined with qk_pad or v_pad"
        D_QK_PAD = d_pad
        D_V_PAD = d_pad
        pad_widths = (("d_pad", d_pad),)
    else:
        D_QK_PAD = default_pad if qk_pad is None else qk_pad
        D_V_PAD = default_pad if v_pad is None else v_pad
        pad_widths = (("qk_pad", D_QK_PAD), ("v_pad", D_V_PAD))
    for name, width in pad_widths:
        assert isinstance(width, int) and width % 8 == 0, (
            f"{name} must be an integer multiple of 8, got {width}"
        )
        assert width >= D, f"{name} must be at least head_dim={D}, got {width}"
    assert block_m > 0 and block_m & (block_m - 1) == 0, f"block_m must be a positive power of 2, got {block_m}"
    assert block_n > 0 and block_n & (block_n - 1) == 0, f"block_n must be a positive power of 2, got {block_n}"
    if one_program_per_task and clear_tail:
        raise ValueError("one_program_per_task is an eager-only tuning option and requires clear_tail=False")
    if out is None:
        out = torch.empty_like(q, memory_format=torch.contiguous_format)
    else:
        assert out.shape == q.shape, "out shape must match q"
        assert out.dtype == q.dtype, "out dtype must match q"
        assert out.device == q.device, "out must be on the same device as q"
    if num_seqs is None:
        num_seqs = cu_seqlens.shape[0] - 1
    assert 1 <= num_seqs <= cu_seqlens.shape[0] - 1, (
        f"num_seqs must be in [1, {cu_seqlens.shape[0] - 1}], got {num_seqs}"
    )
    if T == 0 or H == 0:
        return out

    # `full_aligned` validates the exact eager shape contract. Its constexpr-T
    # lowering is selected only when the precision gate above is enabled;
    # capture/replay callers always keep this false because replay endpoints
    # and active token counts can change.
    if full_aligned:
        assert cu_seqlens.shape[0] == 2, (
            "full_aligned requires exactly two cu_seqlens endpoints; "
            f"got {cu_seqlens.shape[0]}"
        )
        assert (T, H, num_seqs) == (
            A5_FULL_ALIGNED_TOKENS,
            A5_FULL_ALIGNED_HEADS,
            1,
        ), (
            "full_aligned requires exactly "
            f"T={A5_FULL_ALIGNED_TOKENS}, H={A5_FULL_ALIGNED_HEADS}, num_seqs=1; "
            f"got T={T}, H={H}, num_seqs={num_seqs}"
        )
        assert T % block_m == 0 and T % block_n == 0, (
            "full_aligned tuning requires block_m and block_n to divide "
            f"T={T}, got block_m={block_m}, block_n={block_n}"
        )
        assert get_vit_flash_attention_config(full_aligned=True) == VIT_FA_A5_FULL_ALIGNED_CONFIG, (
            "full_aligned is available only on A5"
        )

    exact_t_enabled = full_aligned and A5_EXACT_T_SPECIALIZATION_ENABLED
    if full_aligned and not exact_t_enabled:
        logger.warning_once(
            "[ViT Triton FA] A5 exact-T specialization is disabled by the "
            "precision gate; using the verified dynamic-T BM%s/BN%s/QK%s/V%s "
            "kernel for T=%s, H=%s.",
            block_m,
            block_n,
            D_QK_PAD,
            D_V_PAD,
            T,
            H,
        )

    required_task_num = _max_varlen_task_num(T, num_seqs, H, block_m)
    if max_task_num is None:
        max_task_num = required_task_num
    elif max_task_num < required_task_num:
        raise ValueError(
            f"max_task_num={max_task_num} is smaller than the safe varlen bound "
            f"{required_task_num} for T={T}, num_seqs={num_seqs}, H={H}, BM={block_m}"
        )

    # NOTE: do NOT cast cu_seqlens dtype here. Under capture, a cast would
    # create a new tensor whose address gets baked into the graph, while
    # _run_budget_graph refreshes the ORIGINAL cu_seqlens buffer content on
    # replay -- the cast copy would go stale and the kernel would read the
    # capture-time cu_seqlens. The kernel loads via tl.load(...).to(tl.int32),
    # so int32/int64 both work; keep the original tensor (stable address).
    # Worker initialization normally does this once. Keeping the wrapper
    # self-contained also makes direct operator tests and standalone use safe.
    init_device_properties_triton()
    num_core = get_aicore_num()
    active_core = min(num_core, max_task_num)
    launch_programs = max_task_num if one_program_per_task else active_core

    grid = (launch_programs,)
    _vit_fa_fwd_kernel[grid](
        q,
        k,
        v,
        out,
        cu_seqlens,
        T,
        H,
        max_task_num,
        launch_programs,
        num_seqs,
        scale,
        D=D,
        D_QK_PAD=D_QK_PAD,
        D_V_PAD=D_V_PAD,
        BM=block_m,
        BN=block_n,
        CLEAR_BLOCK=TAIL_CLEAR_BLOCK,
        SINGLE_SEQUENCE=num_seqs == 1,
        TWO_SEQUENCES=num_seqs == 2,
        CLEAR_TAIL=clear_tail,
        EXACT_T=A5_FULL_ALIGNED_TOKENS if exact_t_enabled else 0,
        num_stages=2,
        num_warps=4,
        # Keep the verified pipelined schedule explicit. The A5 kernel-launch
        # suite failed during MLIR compilation with it disabled on 3.2.1.
        multibuffer=True,
    )
    path_kind = "A5-exact" if exact_t_enabled else "varlen"
    path = f"{path_kind}-BM{block_m}-BN{block_n}-QK{D_QK_PAD}-V{D_V_PAD}"
    logger.info_once(
        "[ViT Triton FA] ACTIVE: path=%s "
        "(D_QK_PAD=%s, D_V_PAD=%s, BM=%s, BN=%s, PV_LAYOUT=standard, "
        "single-seq-specialization=%s, two-seq-specialization=%s, "
        "exact-T=%s, tail-clear=%s, grid-mode=%s, programs=%s, "
        "hardware-cores=%s, multibuffer=on); "
        "FIA pad-to-128 and output slicing are bypassed.",
        path,
        D_QK_PAD,
        D_V_PAD,
        block_m,
        block_n,
        num_seqs == 1,
        num_seqs == 2,
        A5_FULL_ALIGNED_TOKENS if exact_t_enabled else 0,
        clear_tail,
        "per-task" if one_program_per_task else "persistent-core",
        launch_programs,
        num_core,
    )
    return out


# ---------------------------------------------------------------------------
# Tuning note (BM / BN / logical D widths) -- measure on the target NPU before changing
# production defaults. The generic configuration is (BM=64, BN=64, QK=80,
# V=80). A5 uses (BM=64, BN=128, QK=80, V=88) for varlen/graph and
# (BM=128, BN=128, QK=80, V=96) for the aligned T=1024/H=16 eager shape.
# Exact-T specialization remains disabled by its precision gate. BN256 was
# tried to halve the L=512 online-softmax merge count but did not improve A5
# performance. V88 is the measured packed middle point: V80 regressed, while
# V96 did extra zero-tail PV work. The aligned BM128 path retains V96 because
# V128 spills A5 UB and V88 was not benchmarked for that distinct tile.
# Input/output remain physically D=72. Multibuffering stays enabled to retain
# the previously benchmarked A5 launch schedule on Triton-Ascend 3.2.1.
# A5 packed [512,512] BN128/V88 measured a 1.2018x five-process epoch-speedup
# median and 1.1974x absolute-p50-ratio median. Other SoCs retain
# BM64/BN64/QK80/V80.
#
# The relevant Ascend tunables are block_m (BM), block_n (BN), D_QK_PAD, and
# D_V_PAD. They depend on the real Qwen3.5-VL image token-length distribution,
# which only shows up on NPU runs. Triton-Ascend currently does not support
# tuning num_warps or num_stages; the launch values above are retained only for
# API compatibility. Things to sweep once on hardware:
#
#   * Typical per-image L after patch-embed (before patch-merger). Qwen3.5-VL
#     images commonly land around L = 256..1296 tokens, so num_kv_blocks =
#     ceil(L / BN) is small; BN=64 -> 4..20 blocks.
#   * Candidates: (BM, BN) in {(64,64), (64,128), (128,64), (128,128)}.
#     Do not repeat (64,256); its A5 measurement did not beat (64,128).
#     The O accumulator is [BM, D_V_PAD] fp32. Larger tiles can increase the
#     multibuffer working set and may overflow UB/L1.
#   * D_V_PAD is retunable via the wrapper without changing the physical
#     input/output tensor layout.
#   * ``one_program_per_task=True`` is an eager-only benchmark candidate for
#     comparing hardware wave scheduling against the production persistent-
#     core loop. It is deliberately disabled for graph capture and serving.
#   * Dynamic enumeration cost: each task scans `num_seqs` entries (=
#     max_batch_size under capture). Small, but if max_batch_size is large
#     consider a prefix-sum buffer (capture: refresh as input buffer).
#   * What to watch: cube utilization and L1/L0 hit rate per tile; pick the
#     config that maximizes cube busy % at the modal L.
# ---------------------------------------------------------------------------
