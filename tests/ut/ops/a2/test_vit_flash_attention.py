# SPDX-License-Identifier: Apache-2.0
"""NPU correctness and graph-replay tests for vision Triton FA.

The file lives in the A2 CI suite, but uses the runtime SoC configuration so it
can also validate the A5 specialization when run explicitly on A5 hardware.
"""

import pytest
import torch
from vllm.platforms import current_platform

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.ops.triton.vit_flash_attention import (
    A5_FULL_ALIGNED_HEADS,
    A5_FULL_ALIGNED_TOKENS,
    VIT_FA_A5_CONFIG,
    VIT_FA_A5_FULL_ALIGNED_CONFIG,
    get_vit_flash_attention_config,
    vit_flash_attention_fwd,
)


@pytest.fixture(scope="module", autouse=True)
def _initialize_triton_device_properties():
    init_device_properties_triton()


def _manual_fa_ref(q, k, v, cu_seqlens, scale):
    """CPU fp32 reference for packed, full bidirectional attention."""
    q_cpu = q.float().cpu()
    k_cpu = k.float().cpu()
    v_cpu = v.float().cpu()
    cu = cu_seqlens.cpu().long().tolist()
    out = torch.zeros_like(q_cpu)

    for bos, eos in zip(cu[:-1], cu[1:]):
        if bos == eos:
            continue
        q_seq = q_cpu[bos:eos].transpose(0, 1)
        k_seq = k_cpu[bos:eos].transpose(0, 1)
        v_seq = v_cpu[bos:eos].transpose(0, 1)
        scores = torch.matmul(q_seq, k_seq.transpose(-1, -2)) * scale
        probs = torch.softmax(scores, dim=-1)
        out[bos:eos] = torch.matmul(probs, v_seq).transpose(0, 1)

    return out


def _make_inputs(total_tokens, num_heads=4, head_dim=72, seed=0):
    generator = torch.Generator().manual_seed(seed)
    shape = (total_tokens, num_heads, head_dim)
    q = torch.randn(shape, dtype=torch.bfloat16, generator=generator).npu()
    k = torch.randn(shape, dtype=torch.bfloat16, generator=generator).npu()
    v = torch.randn(shape, dtype=torch.bfloat16, generator=generator).npu()
    return q, k, v


def _run_production_attention(q, k, v, cu_seqlens, scale, *, out):
    """Use the SoC-specific tile configuration selected by the real caller."""
    config = get_vit_flash_attention_config()
    return vit_flash_attention_fwd(
        q,
        k,
        v,
        cu_seqlens,
        scale,
        out=out,
        block_m=config.block_m,
        block_n=config.block_n,
        qk_pad=config.qk_pad,
        v_pad=config.v_pad,
    )


@pytest.mark.parametrize(
    "seq_lens",
    [
        [64],
        [65],
        [100],
        [127],
        [128],
        [129],
        [255],
        [256],
        [257],
        [37, 91],
        [1, 65, 127],
    ],
)
@torch.inference_mode()
def test_precision_vs_cpu_reference(seq_lens):
    head_dim = 72
    total_tokens = sum(seq_lens)
    q, k, v = _make_inputs(total_tokens)
    cu = torch.tensor(
        [0, *torch.tensor(seq_lens).cumsum(0).tolist()],
        dtype=torch.int32,
        device="npu",
    )
    out = torch.full_like(q, float("nan"))

    result = _run_production_attention(q, k, v, cu, head_dim**-0.5, out=out)
    torch.npu.synchronize()
    ref = _manual_fa_ref(q, k, v, cu, head_dim**-0.5)

    assert result is out
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result.cpu().float(), ref, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("total_tokens", [63, 64, 65, 127, 128, 129, 255, 256, 257])
@torch.inference_mode()
def test_precision_with_split_qk_and_value_widths(total_tokens):
    """Compile and validate the A5 BM64/BN128/QK80/V88 varlen tile."""
    config = get_vit_flash_attention_config()
    if config != VIT_FA_A5_CONFIG:
        pytest.skip("A5-specific tile configuration")
    head_dim = 72
    q, k, v = _make_inputs(total_tokens, num_heads=16)
    cu = torch.tensor([0, total_tokens], dtype=torch.int32, device="npu")
    result = vit_flash_attention_fwd(
        q,
        k,
        v,
        cu,
        head_dim**-0.5,
        block_m=config.block_m,
        block_n=config.block_n,
        qk_pad=config.qk_pad,
        v_pad=config.v_pad,
        clear_tail=False,
    )
    torch.npu.synchronize()
    ref = _manual_fa_ref(q, k, v, cu, head_dim**-0.5)

    assert result.shape == q.shape
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result.cpu().float(), ref, atol=3e-2, rtol=3e-2)


@torch.inference_mode()
def test_a5_full_aligned_bm128_precision():
    """Validate T=1024/H=16 through the precision-gated dynamic-T fallback."""
    config = get_vit_flash_attention_config(full_aligned=True)
    if config != VIT_FA_A5_FULL_ALIGNED_CONFIG:
        pytest.skip("A5-specific full-aligned specialization")

    head_dim = 72
    q, k, v = _make_inputs(
        A5_FULL_ALIGNED_TOKENS,
        num_heads=A5_FULL_ALIGNED_HEADS,
        seed=17,
    )
    cu = torch.tensor(
        [0, A5_FULL_ALIGNED_TOKENS],
        dtype=torch.int32,
        device="npu",
    )
    result = vit_flash_attention_fwd(
        q,
        k,
        v,
        cu,
        head_dim**-0.5,
        block_m=config.block_m,
        block_n=config.block_n,
        qk_pad=config.qk_pad,
        v_pad=config.v_pad,
        full_aligned=True,
        clear_tail=False,
    )
    torch.npu.synchronize()
    ref = _manual_fa_ref(q, k, v, cu, head_dim**-0.5)

    assert result.shape == q.shape
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result.cpu().float(), ref, atol=3e-2, rtol=3e-2)


@torch.inference_mode()
def test_a5_packed_bm64_precision():
    """Validate the production dynamic-T BM64 path for packed [512,512]."""
    config = get_vit_flash_attention_config()
    if config != VIT_FA_A5_CONFIG:
        pytest.skip("A5-specific packed tile configuration")

    head_dim = 72
    seq_len = 512
    total_tokens = seq_len * 2
    q, k, v = _make_inputs(total_tokens, num_heads=16)
    cu = torch.tensor(
        [0, seq_len, total_tokens],
        dtype=torch.int32,
        device="npu",
    )
    result = vit_flash_attention_fwd(
        q,
        k,
        v,
        cu,
        head_dim**-0.5,
        block_m=config.block_m,
        block_n=config.block_n,
        qk_pad=config.qk_pad,
        v_pad=config.v_pad,
        clear_tail=False,
    )
    torch.npu.synchronize()
    ref = _manual_fa_ref(q, k, v, cu, head_dim**-0.5)

    assert result.shape == q.shape
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result.cpu().float(), ref, atol=3e-2, rtol=3e-2)


@torch.inference_mode()
def test_precision_and_tail_clear_with_int64_cu_seqlens():
    total_tokens, head_dim = 128, 72
    q, k, v = _make_inputs(total_tokens)
    cu = torch.tensor([0, 37, 100], dtype=torch.int64, device="npu")
    out = torch.full_like(q, float("nan"))

    result = _run_production_attention(q, k, v, cu, head_dim**-0.5, out=out)
    torch.npu.synchronize()
    ref = _manual_fa_ref(q, k, v, cu, head_dim**-0.5)

    assert torch.isfinite(result).all()
    torch.testing.assert_close(result.cpu().float(), ref, atol=3e-2, rtol=3e-2)


@torch.inference_mode()
def test_rejects_non_72_head_dim():
    q, k, v = _make_inputs(64, head_dim=80)
    cu = torch.tensor([0, 64], dtype=torch.int32, device="npu")
    with pytest.raises(AssertionError, match="head_dim=72"):
        vit_flash_attention_fwd(q, k, v, cu, 80**-0.5)


@torch.inference_mode()
def test_npu_graph_replay_reads_current_cu_seqlens():
    # Match Qwen3.5-VL's production head-count specialization during capture.
    total_tokens, num_heads, head_dim = 384, 16, 72
    q, k, v = _make_inputs(total_tokens, num_heads=num_heads, seed=11)
    cu = torch.tensor([0, 128, 256, 384, 384], dtype=torch.int32, device="npu")
    out = torch.full_like(q, float("nan"))

    # Warm up the exact specialization so JIT and device discovery happen
    # before graph capture.
    _run_production_attention(q, k, v, cu, head_dim**-0.5, out=out)
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    graph_pool = current_platform.get_global_graph_pool()
    # Match EncoderAclGraphManager's production capture context exactly.
    with torch.npu.graph(graph, graph_pool):
        _run_production_attention(q, k, v, cu, head_dim**-0.5, out=out)

    replay_layouts = (
        [0, 96, 192, 288, 384],
        # Non-aligned BOS offsets exercise full/tail loops on packed sequences.
        [0, 37, 165, 294, 384],
        # Exact BN tiles followed by full+tail sequences.
        [0, 128, 256, 384, 384],
        [0, 129, 258, 384, 384],
        # One active sequence followed by fixed-shape empty graph slots. These
        # switch the same graph between two full blocks, full+tail, tail-only,
        # and a capacity tail that must be cleared.
        [0, 256, 256, 256, 256],
        [0, 257, 257, 257, 257],
        [0, 127, 127, 127, 127],
        # Switch back to the generic path in the same captured kernel.
        [0, 13, 150, 279, 384],
        # No active tasks: the fixed-capacity output must be cleared fully.
        [0, 0, 0, 0, 0],
    )
    for replay_idx, endpoints in enumerate(replay_layouts):
        q_next, k_next, v_next = _make_inputs(total_tokens, num_heads=num_heads, seed=100 + replay_idx)
        q.copy_(q_next)
        k.copy_(k_next)
        v.copy_(v_next)
        cu.copy_(torch.tensor(endpoints, dtype=torch.int32, device="npu"))
        out.fill_(float("nan"))

        graph.replay()
        torch.npu.synchronize()

        ref = _manual_fa_ref(q, k, v, cu, head_dim**-0.5)
        assert torch.isfinite(out).all(), f"replay left unwritten output for endpoints={endpoints}"
        torch.testing.assert_close(out.cpu().float(), ref, atol=3e-2, rtol=3e-2)
