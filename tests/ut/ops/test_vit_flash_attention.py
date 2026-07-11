# SPDX-License-Identifier: Apache-2.0
"""CPU-side launch and routing tests for vision Triton flash-attention."""

import os
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.ops.triton.vit_flash_attention import (
    A5_EXACT_T_SPECIALIZATION_ENABLED,
    A5_FULL_ALIGNED_HEADS,
    A5_FULL_ALIGNED_TOKENS,
    VIT_FA_A5_CONFIG,
    VIT_FA_A5_FULL_ALIGNED_CONFIG,
    VIT_FA_GENERIC_CONFIG,
    _max_varlen_task_num,
    get_vit_flash_attention_config,
    vit_flash_attention_fwd,
)
from vllm_ascend.utils import AscendDeviceType


@pytest.fixture(scope="module")
def vit_fa_benchmark_module():
    benchmark_path = Path(__file__).parents[3] / "benchmarks" / "ops" / "benchmark_vit_flash_attention.py"
    return runpy.run_path(str(benchmark_path))


@pytest.mark.parametrize(
    ("epoch_ratios", "expected"),
    [
        ([1.200 + index * 0.001 for index in range(13)], True),
        ([1.25 if index % 2 == 0 else 0.95 for index in range(13)], False),
        ([1.20] * 6 + [1.24] * 7, False),
        ([1.0, 1.0], False),
        ([0.0, 0.0, 0.0], False),
    ],
)
def test_benchmark_epoch_stability(vit_fa_benchmark_module, epoch_ratios, expected):
    is_stable = vit_fa_benchmark_module["_epoch_ratios_are_stable"]
    assert is_stable(epoch_ratios) is expected


def test_benchmark_accepts_one_candidate_tile_override(vit_fa_benchmark_module):
    parse_args = vit_fa_benchmark_module["_parse_args"]
    with patch(
        "sys.argv",
        [
            "benchmark_vit_flash_attention.py",
            "--seq-lens",
            "1024",
            "--block-m",
            "64",
            "--block-n",
            "128",
            "--qk-pad",
            "80",
            "--v-pad",
            "80",
            "--per-task-grid",
        ],
    ):
        args = parse_args()

    assert args.seq_lens == [[1024]]
    assert (args.block_m, args.block_n, args.qk_pad, args.v_pad) == (64, 128, 80, 80)
    assert args.per_task_grid is True


def test_benchmark_accepts_compact_profile_mode(vit_fa_benchmark_module):
    parse_args = vit_fa_benchmark_module["_parse_args"]
    with patch(
        "sys.argv",
        [
            "benchmark_vit_flash_attention.py",
            "--profile",
        ],
    ):
        args = parse_args()

    assert args.profile is True
    assert args.profile_dir == Path("vit_fa_profile")
    assert args.profile_runs == 5


def test_benchmark_formats_profile_as_one_result_line(vit_fa_benchmark_module):
    format_result = vit_fa_benchmark_module["_format_profile_result"]
    result = {
        "seq_lens": [512, 512],
        "config": SimpleNamespace(block_m=64, block_n=128, qk_pad=80, v_pad=88),
        "profiles": {
            "FIA": {"device_us_per_call": 120.0},
            "Triton": {
                "device_us_per_call": 100.0,
                "pipe_ratios_pct": {
                    "aic_mac_ratio": 70.04,
                    "aic_mte2_ratio": 20.06,
                    "aiv_vec_ratio": 80.04,
                    "aiv_mte2_ratio": 30.06,
                },
            },
        },
    }

    line = format_result(result)

    assert "\n" not in line
    assert line == (
        "PROFILE_RESULT seq=512+512 cfg=BM64-BN128-QK80-V88 "
        "fia_us=120.000 tri_us=100.000 prof_ratio=1.200x "
        "mac=70.0% aic_mte2=20.1% vec=80.0% aiv_mte2=30.1% aiv_mte3=NA"
    )


def test_benchmark_summarizes_kernel_profile_without_pandas(
    vit_fa_benchmark_module,
    tmp_path,
):
    kernel_csv = tmp_path / "kernel_details.csv"
    kernel_csv.write_text(
        "Name,Duration(us),aic_mac_ratio,aic_mte2_ratio,aiv_vec_ratio\n"
        "kernel_a,10,0.5,0.2,0.3\n"
        "kernel_a,20,70,40,10\n"
        "kernel_b,5,0,80,90\n",
        encoding="utf-8",
    )

    summarize = vit_fa_benchmark_module["_summarize_kernel_details"]
    summary = summarize([kernel_csv], profile_runs=5)

    assert summary["row_count"] == 3
    assert summary["device_us_per_call"] == pytest.approx(7.0)
    assert summary["kernel_rows_per_call"] == pytest.approx(0.6)
    assert summary["top_kernels"][0] == {
        "name": "kernel_a",
        "per_call_us": pytest.approx(6.0),
        "count": 2,
    }
    assert summary["pipe_ratios_pct"] == pytest.approx(
        {
            "aic_mac_ratio": 54.285714,
            "aic_mte2_ratio": 40.0,
            "aiv_vec_ratio": 27.142857,
        }
    )


def test_benchmark_profiles_one_variant_in_an_isolated_context(
    vit_fa_benchmark_module,
    tmp_path,
):
    profile_variant = vit_fa_benchmark_module["_profile_variant"]
    benchmark_globals = profile_variant.__globals__
    torch_module = benchmark_globals["torch"]
    torch_npu_module = benchmark_globals["torch_npu"]
    fn = MagicMock(return_value=MagicMock(spec=torch.Tensor))
    profiler_context = MagicMock()
    profiler = profiler_context.__enter__.return_value
    expected_summary = {"device_us_per_call": 12.5}
    analyse = MagicMock(return_value=expected_summary.copy())

    with (
        patch.dict(benchmark_globals, {"_analyse_profile_output": analyse}),
        patch.object(torch_module.npu, "synchronize") as synchronize,
        patch.object(
            torch_npu_module.profiler,
            "_ExperimentalConfig",
            return_value=MagicMock(),
        ) as experimental_config,
        patch.object(
            torch_npu_module.profiler,
            "tensorboard_trace_handler",
            return_value=MagicMock(),
        ),
        patch.object(
            torch_npu_module.profiler,
            "profile",
            return_value=profiler_context,
        ) as profile,
    ):
        summary = profile_variant(
            fn,
            label="triton",
            profile_dir=tmp_path / "triton",
            profile_runs=3,
        )

    assert fn.call_count == 8  # five warmups plus three captured calls
    assert profiler.step.call_count == 3
    assert synchronize.call_count == 2
    assert experimental_config.call_args.kwargs["aic_metrics"] == (
        torch_npu_module.profiler.AiCMetrics.PipeUtilization
    )
    assert profile.call_args.kwargs["activities"] == [
        torch_npu_module.profiler.ProfilerActivity.NPU
    ]
    analyse.assert_called_once_with(tmp_path / "triton", 3)
    assert summary["device_us_per_call"] == 12.5
    assert summary["profile_dir"] == tmp_path / "triton"


@pytest.mark.parametrize(
    ("device_type", "full_aligned", "expected"),
    [
        (AscendDeviceType.A2, False, VIT_FA_GENERIC_CONFIG),
        (AscendDeviceType.A3, True, VIT_FA_GENERIC_CONFIG),
        (AscendDeviceType.A5, False, VIT_FA_A5_CONFIG),
        (AscendDeviceType.A5, True, VIT_FA_A5_FULL_ALIGNED_CONFIG),
    ],
)
def test_production_config_is_soc_specific(device_type, full_aligned, expected):
    get_vit_flash_attention_config.cache_clear()
    try:
        with patch(
            "vllm_ascend.ops.triton.vit_flash_attention.get_ascend_device_type",
            return_value=device_type,
        ):
            assert get_vit_flash_attention_config(full_aligned=full_aligned) == expected
    finally:
        get_vit_flash_attention_config.cache_clear()


def test_production_config_uses_runtime_soc_without_generated_build_info():
    get_vit_flash_attention_config.cache_clear()
    try:
        with (
            patch(
                "vllm_ascend.ops.triton.vit_flash_attention.get_ascend_device_type",
                side_effect=ImportError,
            ),
            patch(
                "vllm_ascend.ops.triton.vit_flash_attention.torch_npu.npu.get_soc_version",
                return_value=260,
            ),
        ):
            assert get_vit_flash_attention_config() == VIT_FA_A5_CONFIG
            assert (
                get_vit_flash_attention_config(full_aligned=True)
                == VIT_FA_A5_FULL_ALIGNED_CONFIG
            )
    finally:
        get_vit_flash_attention_config.cache_clear()


def test_a5_configs_use_measured_pv_widths():
    assert A5_EXACT_T_SPECIALIZATION_ENABLED is False
    assert VIT_FA_A5_CONFIG.block_m == 64
    assert VIT_FA_A5_CONFIG.block_n == 128
    assert VIT_FA_A5_CONFIG.qk_pad == 80
    assert VIT_FA_A5_CONFIG.v_pad == 88
    assert VIT_FA_A5_FULL_ALIGNED_CONFIG.block_m == 128
    assert VIT_FA_A5_FULL_ALIGNED_CONFIG.block_n == 128
    assert VIT_FA_A5_FULL_ALIGNED_CONFIG.qk_pad == 80
    assert VIT_FA_A5_FULL_ALIGNED_CONFIG.v_pad == 96


@pytest.mark.parametrize(
    ("total_tokens", "num_seqs", "num_heads", "expected"),
    [
        (130, 1, 16, 48),
        (128, 2, 16, 48),
        (512, 4, 16, 176),
        (2048, 4, 16, 560),
    ],
)
def test_max_varlen_task_num(total_tokens, num_seqs, num_heads, expected):
    assert _max_varlen_task_num(total_tokens, num_seqs, num_heads, 64) == expected


@pytest.mark.parametrize(
    ("total_tokens", "cu_seqlens", "expected_tasks", "expected_cores"),
    [
        (130, [0, 130], 6, 6),
        (128, [0, 37, 128], 6, 6),
        (512, [0, 200, 400, 512, 512], 22, 8),
    ],
)
def test_wrapper_launch_uses_safe_varlen_bound(total_tokens, cu_seqlens, expected_tasks, expected_cores):
    num_heads, head_dim = 2, 72
    q = torch.empty(total_tokens, num_heads, head_dim, dtype=torch.bfloat16)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    cu = torch.tensor(cu_seqlens, dtype=torch.int64)

    with (
        patch("vllm_ascend.ops.triton.vit_flash_attention._vit_fa_fwd_kernel") as mock_kernel,
        patch("vllm_ascend.ops.triton.vit_flash_attention.init_device_properties_triton") as mock_init,
        patch("vllm_ascend.ops.triton.vit_flash_attention.get_aicore_num", return_value=8),
        patch("vllm_ascend.ops.triton.vit_flash_attention.logger.info_once") as active_log,
    ):
        out = vit_flash_attention_fwd.__wrapped__(q, k, v, cu, head_dim**-0.5)

    mock_init.assert_called_once_with()
    active_log.assert_called_once()
    assert "ACTIVE" in active_log.call_args.args[0]
    assert "D_QK_PAD=%s, D_V_PAD=%s" in active_log.call_args.args[0]
    assert "PV_LAYOUT=standard" in active_log.call_args.args[0]
    assert active_log.call_args.args[1] == "varlen-BM64-BN64-QK80-V80"
    assert active_log.call_args.args[2:6] == (80, 80, 64, 64)
    call = mock_kernel.__getitem__.return_value.call_args
    args = call.args
    kwargs = call.kwargs
    assert args[3] is out
    assert args[4] is cu, "int64 cu_seqlens must keep its stable input address"
    assert args[7] == expected_tasks
    assert args[8] == expected_cores
    assert args[9] == len(cu_seqlens) - 1
    assert kwargs["D"] == 72
    assert kwargs["D_QK_PAD"] == 80
    assert kwargs["D_V_PAD"] == 80
    assert kwargs["BM"] == 64
    assert kwargs["BN"] == 64
    assert kwargs["CLEAR_BLOCK"] == 256
    assert kwargs["SINGLE_SEQUENCE"] is (len(cu_seqlens) == 2)
    assert kwargs["TWO_SEQUENCES"] is (len(cu_seqlens) == 3)
    assert kwargs["CLEAR_TAIL"] is True
    assert kwargs["EXACT_T"] == 0
    assert kwargs["multibuffer"] is True


def test_wrapper_exposes_split_logical_widths():
    q = torch.empty(128, 2, 72, dtype=torch.bfloat16)
    cu = torch.tensor([0, 128], dtype=torch.int32)

    with (
        patch("vllm_ascend.ops.triton.vit_flash_attention._vit_fa_fwd_kernel") as mock_kernel,
        patch("vllm_ascend.ops.triton.vit_flash_attention.init_device_properties_triton"),
        patch("vllm_ascend.ops.triton.vit_flash_attention.get_aicore_num", return_value=8),
        patch("vllm_ascend.ops.triton.vit_flash_attention.logger.info_once") as active_log,
    ):
        vit_flash_attention_fwd.__wrapped__(
            q,
            q,
            q,
            cu,
            72**-0.5,
            block_m=128,
            block_n=128,
            qk_pad=80,
            v_pad=128,
        )

    launch = mock_kernel.__getitem__.return_value.call_args
    assert launch.kwargs["BM"] == 64
    assert launch.kwargs["BN"] == 128
    assert launch.kwargs["D_QK_PAD"] == 80
    assert launch.kwargs["D_V_PAD"] == 128
    assert launch.kwargs["SINGLE_SEQUENCE"] is True
    assert launch.kwargs["TWO_SEQUENCES"] is False
    assert launch.kwargs["CLEAR_TAIL"] is True
    assert launch.kwargs["EXACT_T"] == 0
    assert launch.kwargs["multibuffer"] is True
    assert active_log.call_args.args[1] == "varlen-BM128-BN128-QK80-V128"
    assert active_log.call_args.args[2:6] == (80, 128, 128, 128)
    assert "PV_LAYOUT=standard" in active_log.call_args.args[0]


def test_wrapper_launches_a5_varlen_pv_specialization():
    q = torch.empty(128, 2, 72, dtype=torch.bfloat16)
    cu = torch.tensor([0, 128], dtype=torch.int32)

    with (
        patch("vllm_ascend.ops.triton.vit_flash_attention._vit_fa_fwd_kernel") as mock_kernel,
        patch("vllm_ascend.ops.triton.vit_flash_attention.init_device_properties_triton"),
        patch("vllm_ascend.ops.triton.vit_flash_attention.get_aicore_num", return_value=8),
        patch("vllm_ascend.ops.triton.vit_flash_attention.logger.info_once") as active_log,
    ):
        vit_flash_attention_fwd.__wrapped__(
            q,
            q,
            q,
            cu,
            72**-0.5,
            block_m=VIT_FA_A5_CONFIG.block_m,
            block_n=VIT_FA_A5_CONFIG.block_n,
            qk_pad=VIT_FA_A5_CONFIG.qk_pad,
            v_pad=VIT_FA_A5_CONFIG.v_pad,
        )

    launch = mock_kernel.__getitem__.return_value.call_args
    assert launch.kwargs["D_QK_PAD"] == 80
    assert launch.kwargs["D_V_PAD"] == 88
    assert launch.kwargs["BM"] == 64
    assert launch.kwargs["BN"] == 128
    assert launch.kwargs["SINGLE_SEQUENCE"] is True
    assert launch.kwargs["TWO_SEQUENCES"] is False
    assert launch.kwargs["CLEAR_TAIL"] is True
    assert launch.kwargs["EXACT_T"] == 0
    assert launch.kwargs["multibuffer"] is True
    assert active_log.call_args.args[1] == "varlen-BM64-BN128-QK80-V88"
    assert active_log.call_args.args[2:6] == (80, 88, 64, 128)
    assert "PV_LAYOUT=standard" in active_log.call_args.args[0]


def test_wrapper_precision_gate_uses_dynamic_t_for_a5_aligned_shape():
    q = torch.empty(
        A5_FULL_ALIGNED_TOKENS,
        A5_FULL_ALIGNED_HEADS,
        72,
        dtype=torch.bfloat16,
    )
    cu = torch.tensor([0, A5_FULL_ALIGNED_TOKENS], dtype=torch.int32)
    config = VIT_FA_A5_FULL_ALIGNED_CONFIG

    with (
        patch("vllm_ascend.ops.triton.vit_flash_attention._vit_fa_fwd_kernel") as kernel,
        patch(
            "vllm_ascend.ops.triton.vit_flash_attention.get_vit_flash_attention_config",
            return_value=config,
        ),
        patch("vllm_ascend.ops.triton.vit_flash_attention.init_device_properties_triton"),
        patch("vllm_ascend.ops.triton.vit_flash_attention.get_aicore_num", return_value=32),
        patch("vllm_ascend.ops.triton.vit_flash_attention.logger.info_once") as active_log,
    ):
        out = vit_flash_attention_fwd.__wrapped__(
            q,
            q,
            q,
            cu,
            72**-0.5,
            block_m=config.block_m,
            block_n=config.block_n,
            qk_pad=config.qk_pad,
            v_pad=config.v_pad,
            full_aligned=True,
            clear_tail=False,
        )

    kernel.__getitem__.assert_called_once_with((32,))
    launch = kernel.__getitem__.return_value.call_args
    assert launch.args[3] is out
    assert launch.args[4] is cu
    assert launch.args[5] == A5_FULL_ALIGNED_TOKENS
    assert launch.args[6] == A5_FULL_ALIGNED_HEADS
    assert launch.args[7] == 128
    assert launch.args[8] == 32
    assert launch.args[9] == 1
    assert launch.args[10] == 72**-0.5
    assert launch.kwargs["BM"] == 128
    assert launch.kwargs["BN"] == 128
    assert launch.kwargs["D_QK_PAD"] == 80
    assert launch.kwargs["D_V_PAD"] == 96
    assert launch.kwargs["SINGLE_SEQUENCE"] is True
    assert launch.kwargs["TWO_SEQUENCES"] is False
    assert launch.kwargs["CLEAR_TAIL"] is False
    assert launch.kwargs["EXACT_T"] == 0
    assert launch.kwargs["multibuffer"] is True
    assert active_log.call_args.args[1] == "varlen-BM128-BN128-QK80-V96"


def test_wrapper_precision_gate_disables_exact_t_for_tile_candidate():
    q = torch.empty(
        A5_FULL_ALIGNED_TOKENS,
        A5_FULL_ALIGNED_HEADS,
        72,
        dtype=torch.bfloat16,
    )
    cu = torch.tensor([0, A5_FULL_ALIGNED_TOKENS], dtype=torch.int32)

    with (
        patch("vllm_ascend.ops.triton.vit_flash_attention._vit_fa_fwd_kernel") as kernel,
        patch(
            "vllm_ascend.ops.triton.vit_flash_attention.get_vit_flash_attention_config",
            return_value=VIT_FA_A5_FULL_ALIGNED_CONFIG,
        ),
        patch("vllm_ascend.ops.triton.vit_flash_attention.init_device_properties_triton"),
        patch("vllm_ascend.ops.triton.vit_flash_attention.get_aicore_num", return_value=32),
        patch("vllm_ascend.ops.triton.vit_flash_attention.logger.info_once"),
    ):
        vit_flash_attention_fwd.__wrapped__(
            q,
            q,
            q,
            cu,
            72**-0.5,
            block_m=64,
            block_n=128,
            qk_pad=80,
            v_pad=80,
            full_aligned=True,
            clear_tail=False,
            one_program_per_task=True,
        )

    kernel.__getitem__.assert_called_once_with((256,))
    launch = kernel.__getitem__.return_value.call_args
    assert launch.args[8] == 256
    assert launch.kwargs["BM"] == 64
    assert launch.kwargs["BN"] == 128
    assert launch.kwargs["D_QK_PAD"] == 80
    assert launch.kwargs["D_V_PAD"] == 80
    assert launch.kwargs["EXACT_T"] == 0
    assert launch.kwargs["CLEAR_TAIL"] is False


def test_wrapper_rejects_per_task_grid_with_capacity_tail_clear():
    q = torch.empty(128, 2, 72, dtype=torch.bfloat16)
    cu = torch.tensor([0, 128], dtype=torch.int32)

    with pytest.raises(ValueError, match="eager-only tuning option"):
        vit_flash_attention_fwd.__wrapped__(
            q,
            q,
            q,
            cu,
            72**-0.5,
            one_program_per_task=True,
        )


def test_wrapper_marks_preallocated_two_sequence_output_for_tail_clear():
    q = torch.empty(256, 2, 72, dtype=torch.bfloat16)
    cu = torch.tensor([0, 100, 200], dtype=torch.int32)
    out = torch.empty_like(q)

    with (
        patch("vllm_ascend.ops.triton.vit_flash_attention._vit_fa_fwd_kernel") as kernel,
        patch("vllm_ascend.ops.triton.vit_flash_attention.init_device_properties_triton"),
        patch("vllm_ascend.ops.triton.vit_flash_attention.get_aicore_num", return_value=8),
        patch("vllm_ascend.ops.triton.vit_flash_attention.logger.info_once"),
    ):
        result = vit_flash_attention_fwd.__wrapped__(
            q,
            q,
            q,
            cu,
            72**-0.5,
            out=out,
        )

    assert result is out
    launch = kernel.__getitem__.return_value.call_args
    assert launch.kwargs["SINGLE_SEQUENCE"] is False
    assert launch.kwargs["TWO_SEQUENCES"] is True
    assert launch.kwargs["CLEAR_TAIL"] is True
    assert launch.kwargs["EXACT_T"] == 0


def test_wrapper_full_aligned_rejects_extra_metadata_slots():
    q = torch.empty(
        A5_FULL_ALIGNED_TOKENS,
        A5_FULL_ALIGNED_HEADS,
        72,
        dtype=torch.bfloat16,
    )
    cu = torch.tensor([0, A5_FULL_ALIGNED_TOKENS, A5_FULL_ALIGNED_TOKENS], dtype=torch.int32)

    with pytest.raises(AssertionError, match="exactly two cu_seqlens endpoints"):
        vit_flash_attention_fwd.__wrapped__(
            q,
            q,
            q,
            cu,
            72**-0.5,
            num_seqs=1,
            block_m=128,
            block_n=128,
            qk_pad=80,
            v_pad=96,
            full_aligned=True,
        )


def test_wrapper_preserves_uniform_d_pad_alias():
    q = torch.empty(128, 2, 72, dtype=torch.bfloat16)
    cu = torch.tensor([0, 128], dtype=torch.int32)

    with (
        patch("vllm_ascend.ops.triton.vit_flash_attention._vit_fa_fwd_kernel") as mock_kernel,
        patch("vllm_ascend.ops.triton.vit_flash_attention.init_device_properties_triton"),
        patch("vllm_ascend.ops.triton.vit_flash_attention.get_aicore_num", return_value=8),
        patch("vllm_ascend.ops.triton.vit_flash_attention.logger.info_once"),
    ):
        vit_flash_attention_fwd.__wrapped__(q, q, q, cu, 72**-0.5, d_pad=128)

    launch = mock_kernel.__getitem__.return_value.call_args
    assert launch.kwargs["D_QK_PAD"] == 128
    assert launch.kwargs["D_V_PAD"] == 128
    assert launch.kwargs["CLEAR_TAIL"] is True
    assert launch.kwargs["EXACT_T"] == 0
    assert launch.kwargs["multibuffer"] is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"qk_pad": 73}, "multiple of 8"),
        ({"v_pad": 64}, "at least head_dim=72"),
        ({"d_pad": 128, "qk_pad": 80}, "cannot be combined"),
        ({"block_m": 48}, "positive power of 2"),
        ({"block_n": 0}, "positive power of 2"),
        ({"full_aligned": True}, "requires exactly"),
    ],
)
def test_wrapper_rejects_invalid_tile_tuning(kwargs, message):
    q = torch.empty(128, 2, 72, dtype=torch.bfloat16)
    cu = torch.tensor([0, 128], dtype=torch.int32)

    with pytest.raises(AssertionError, match=message):
        vit_flash_attention_fwd.__wrapped__(q, q, q, cu, 72**-0.5, **kwargs)


@pytest.mark.parametrize("scale", [0.0, -1.0, float("inf"), float("nan")])
def test_wrapper_rejects_non_positive_or_non_finite_scale(scale):
    q = torch.empty(128, 2, 72, dtype=torch.bfloat16)
    cu = torch.tensor([0, 128], dtype=torch.int32)

    with pytest.raises(AssertionError, match="scale must be positive and finite"):
        vit_flash_attention_fwd.__wrapped__(q, q, q, cu, scale)


def test_wrapper_returns_empty_input_without_launching_kernel():
    q = torch.empty(0, 2, 72, dtype=torch.bfloat16)
    cu = torch.tensor([0, 0], dtype=torch.int32)

    with patch("vllm_ascend.ops.triton.vit_flash_attention._vit_fa_fwd_kernel") as mock_kernel:
        result = vit_flash_attention_fwd.__wrapped__(q, q, q, cu, 72**-0.5)

    mock_kernel.__getitem__.assert_not_called()
    assert result.shape == q.shape


@pytest.mark.parametrize("num_seqs", [0, 2])
def test_wrapper_rejects_num_seqs_outside_metadata(num_seqs):
    q = torch.empty(128, 2, 72, dtype=torch.bfloat16)
    cu = torch.tensor([0, 128], dtype=torch.int32)

    with pytest.raises(AssertionError, match="num_seqs must be in"):
        vit_flash_attention_fwd.__wrapped__(q, q, q, cu, 72**-0.5, num_seqs=num_seqs)


def test_wrapper_rejects_unsafe_task_override():
    q = torch.empty(128, 2, 72, dtype=torch.bfloat16)
    cu = torch.tensor([0, 37, 128], dtype=torch.int32)

    with pytest.raises(ValueError, match="smaller than the safe varlen bound"):
        vit_flash_attention_fwd.__wrapped__(
            q,
            q,
            q,
            cu,
            72**-0.5,
            max_task_num=4,
        )


def test_capture_launch_uses_query_capacity_not_encoder_budget():
    from vllm_ascend.ops.mm_encoder_attention import AscendMMEncoderAttention

    total_tokens, num_heads, head_dim = 2048, 16, 72
    layer = AscendMMEncoderAttention.__new__(AscendMMEncoderAttention)
    layer.head_size = head_dim
    layer.num_heads = num_heads
    layer.scale = head_dim**-0.5

    q = torch.empty(total_tokens, num_heads, head_dim, dtype=torch.bfloat16)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    # The fixed buffer has four sequence slots, but the final capacity tail is
    # outside all packed sequences on this replay.
    cu = torch.tensor([0, 512, 1024, 1536, 1536], dtype=torch.int32)
    context = SimpleNamespace(token_budget=512, capturing=True)

    with (
        patch.object(layer, "_can_use_vit_triton_fa", return_value=True),
        patch("vllm_ascend.ops.mm_encoder_attention.get_encoder_forward_context", return_value=context),
        patch("vllm_ascend.ops.mm_encoder_attention.get_encoder_graph_params", return_value=SimpleNamespace()),
        patch("vllm_ascend.ops.triton.vit_flash_attention._vit_fa_fwd_kernel") as mock_kernel,
        patch("vllm_ascend.ops.triton.vit_flash_attention.init_device_properties_triton"),
        patch("vllm_ascend.ops.triton.vit_flash_attention.get_aicore_num", return_value=8),
        patch(
            "vllm_ascend.ops.triton.vit_flash_attention.get_vit_flash_attention_config",
            return_value=VIT_FA_GENERIC_CONFIG,
        ) as get_config,
        patch("vllm_ascend.ops.triton.vit_flash_attention.logger.info_once"),
    ):
        result = layer._forward_capture_fia(
            q,
            k,
            v,
            cu_seqlens=cu,
            is_reshaped=True,
            bsz=1,
            q_len=total_tokens,
        )

    launch = mock_kernel.__getitem__.return_value.call_args
    launch_args = launch.args
    get_config.assert_called_once_with(full_aligned=False)
    assert launch_args[7] == 560
    assert launch_args[9] == 4
    assert launch.kwargs["SINGLE_SEQUENCE"] is False
    assert launch.kwargs["TWO_SEQUENCES"] is False
    assert launch.kwargs["CLEAR_TAIL"] is True
    assert launch.kwargs["EXACT_T"] == 0
    assert result.shape == (1, total_tokens, num_heads, head_dim)
    assert result.data_ptr() == launch_args[3].data_ptr()


def test_run_vit_triton_uses_soc_config():
    from vllm_ascend.ops.mm_encoder_attention import AscendMMEncoderAttention

    layer = AscendMMEncoderAttention.__new__(AscendMMEncoderAttention)
    layer.scale = 72**-0.5
    q = torch.empty(128, 2, 72, dtype=torch.bfloat16)
    cu = torch.tensor([0, 128], dtype=torch.int32)
    out = torch.empty_like(q)

    with (
        patch(
            "vllm_ascend.ops.triton.vit_flash_attention.get_vit_flash_attention_config",
            return_value=VIT_FA_A5_CONFIG,
        ) as get_config,
        patch(
            "vllm_ascend.ops.triton.vit_flash_attention.vit_flash_attention_fwd",
            return_value=out,
        ) as mock_attention,
    ):
        result = layer._run_vit_triton(q, q, q, cu, out=out)

    assert result is out
    get_config.assert_called_once_with(full_aligned=False)
    mock_attention.assert_called_once_with(
        q,
        q,
        q,
        cu,
        scale=layer.scale,
        out=out,
        block_m=64,
        block_n=128,
        qk_pad=80,
        v_pad=88,
        full_aligned=False,
        clear_tail=True,
    )


def test_run_vit_triton_selects_a5_full_aligned_config():
    from vllm_ascend.ops.mm_encoder_attention import AscendMMEncoderAttention

    layer = AscendMMEncoderAttention.__new__(AscendMMEncoderAttention)
    layer.scale = 72**-0.5
    q = torch.empty(A5_FULL_ALIGNED_TOKENS, A5_FULL_ALIGNED_HEADS, 72, dtype=torch.bfloat16)
    cu = torch.tensor([0, A5_FULL_ALIGNED_TOKENS], dtype=torch.int32)
    out = torch.empty_like(q)

    with (
        patch(
            "vllm_ascend.ops.triton.vit_flash_attention.get_vit_flash_attention_config",
            return_value=VIT_FA_A5_FULL_ALIGNED_CONFIG,
        ) as get_config,
        patch(
            "vllm_ascend.ops.triton.vit_flash_attention.vit_flash_attention_fwd",
            return_value=out,
        ) as mock_attention,
    ):
        result = layer._run_vit_triton(
            q,
            q,
            q,
            cu,
            full_aligned=True,
        )

    assert result is out
    get_config.assert_called_once_with(full_aligned=True)
    mock_attention.assert_called_once_with(
        q,
        q,
        q,
        cu,
        scale=layer.scale,
        out=None,
        block_m=128,
        block_n=128,
        qk_pad=80,
        v_pad=96,
        full_aligned=True,
        clear_tail=False,
    )


def test_run_vit_triton_full_aligned_moves_host_metadata_to_device():
    from vllm_ascend.ops.mm_encoder_attention import AscendMMEncoderAttention

    layer = AscendMMEncoderAttention.__new__(AscendMMEncoderAttention)
    layer.scale = 72**-0.5
    npu_device = SimpleNamespace(type="npu")
    q = SimpleNamespace(device=npu_device)
    cu = MagicMock()
    cu.device = SimpleNamespace(type="cpu")
    cu_device = object()
    cu.to.return_value = cu_device
    out = object()

    with (
        patch(
            "vllm_ascend.ops.triton.vit_flash_attention.get_vit_flash_attention_config",
            return_value=VIT_FA_A5_FULL_ALIGNED_CONFIG,
        ),
        patch(
            "vllm_ascend.ops.triton.vit_flash_attention.vit_flash_attention_fwd",
            return_value=out,
        ) as mock_attention,
    ):
        result = layer._run_vit_triton(q, q, q, cu, full_aligned=True)

    assert result is out
    cu.to.assert_called_once_with(npu_device)
    assert mock_attention.call_args.args[3] is cu_device
    assert mock_attention.call_args.kwargs["full_aligned"] is True


def test_run_vit_triton_non_a5_downgrade_moves_metadata_to_device():
    from vllm_ascend.ops.mm_encoder_attention import AscendMMEncoderAttention

    layer = AscendMMEncoderAttention.__new__(AscendMMEncoderAttention)
    layer.scale = 72**-0.5
    npu_device = SimpleNamespace(type="npu")
    q = SimpleNamespace(device=npu_device)
    cu = MagicMock()
    cu.device = SimpleNamespace(type="cpu")
    cu_device = object()
    cu.to.return_value = cu_device
    out = object()

    with (
        patch(
            "vllm_ascend.ops.triton.vit_flash_attention.get_vit_flash_attention_config",
            return_value=VIT_FA_GENERIC_CONFIG,
        ),
        patch(
            "vllm_ascend.ops.triton.vit_flash_attention.vit_flash_attention_fwd",
            return_value=out,
        ) as mock_attention,
    ):
        result = layer._run_vit_triton(q, q, q, cu, full_aligned=True)

    assert result is out
    cu.to.assert_called_once_with(npu_device)
    assert mock_attention.call_args.args[3] is cu_device
    assert mock_attention.call_args.kwargs["full_aligned"] is False


@pytest.mark.parametrize(
    ("endpoints", "expected_full_aligned"),
    [
        ([0, A5_FULL_ALIGNED_TOKENS], True),
        ([0, A5_FULL_ALIGNED_TOKENS // 2, A5_FULL_ALIGNED_TOKENS], False),
    ],
)
def test_eager_enables_full_aligned_only_for_single_sequence_shape(
    endpoints,
    expected_full_aligned,
):
    from vllm_ascend.ops.mm_encoder_attention import AscendMMEncoderAttention

    layer = AscendMMEncoderAttention.__new__(AscendMMEncoderAttention)
    q = torch.empty(A5_FULL_ALIGNED_TOKENS, A5_FULL_ALIGNED_HEADS, 72, dtype=torch.bfloat16)
    cu = torch.tensor(endpoints, dtype=torch.int32)

    with (
        patch.object(layer, "_can_use_vit_triton_fa", return_value=True),
        patch.object(layer, "_run_vit_triton", return_value=q) as triton_path,
    ):
        result = layer._forward_eager_fia(
            q,
            q,
            q,
            cu_seqlens=cu,
            is_reshaped=True,
            bsz=1,
            q_len=A5_FULL_ALIGNED_TOKENS,
        )

    assert result.shape == (1, A5_FULL_ALIGNED_TOKENS, A5_FULL_ALIGNED_HEADS, 72)
    triton_path.assert_called_once_with(
        q,
        q,
        q,
        cu,
        full_aligned=expected_full_aligned,
    )


def test_eager_never_reads_npu_cu_seqlens_on_host():
    from vllm_ascend.ops.mm_encoder_attention import AscendMMEncoderAttention

    class FakeNpuCuSeqlens:
        device = SimpleNamespace(type="npu")

        def numel(self):
            return 2

        def tolist(self):
            raise AssertionError("NPU cu_seqlens must not be read on host")

    layer = AscendMMEncoderAttention.__new__(AscendMMEncoderAttention)
    q = torch.empty(A5_FULL_ALIGNED_TOKENS, A5_FULL_ALIGNED_HEADS, 72, dtype=torch.bfloat16)
    cu = FakeNpuCuSeqlens()

    with (
        patch.object(layer, "_can_use_vit_triton_fa", return_value=True),
        patch.object(layer, "_run_vit_triton", return_value=q) as triton_path,
    ):
        layer._forward_eager_fia(
            q,
            q,
            q,
            cu_seqlens=cu,
            is_reshaped=True,
            bsz=1,
            q_len=A5_FULL_ALIGNED_TOKENS,
        )

    triton_path.assert_called_once_with(
        q,
        q,
        q,
        cu,
        full_aligned=True,
    )


class _FakeTensor:
    def __init__(self, shape, dtype, device):
        self.shape = shape
        self.dtype = dtype
        self.device = device

    def dim(self):
        return len(self.shape)


@pytest.mark.parametrize("enabled", ["0", "1"])
def test_init_logs_vit_triton_configuration(enabled):
    from vllm_ascend.ops import mm_encoder_attention as mm_attention

    def fake_base_init(self, num_heads, head_size, scale=None, num_kv_heads=None, prefix=""):
        torch.nn.Module.__init__(self)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_size = head_size
        self.scale = head_size**-0.5 if scale is None else scale

    with (
        patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_VIT_TRITON_FA": enabled}),
        patch.object(mm_attention, "HAS_TRITON", True),
        patch.object(mm_attention.MMEncoderAttention, "__init__", fake_base_init),
        patch.object(mm_attention.logger, "info_once") as config_log,
    ):
        mm_attention.AscendMMEncoderAttention(num_heads=2, head_size=72)

    config_log.assert_called_once()
    log_args = config_log.call_args.args
    assert "head_dim=72 configuration" in log_args[0]
    assert log_args[1] is bool(int(enabled))
    assert log_args[2] is True


def test_triton_route_is_strictly_capability_gated():
    from vllm_ascend.ops import mm_encoder_attention as mm_attention
    from vllm_ascend.ops.mm_encoder_attention import AscendMMEncoderAttention

    layer = AscendMMEncoderAttention.__new__(AscendMMEncoderAttention)
    layer.head_size = 72
    npu_device = SimpleNamespace(type="npu")
    q = _FakeTensor((128, 2, 72), torch.bfloat16, npu_device)
    k = _FakeTensor(q.shape, q.dtype, npu_device)
    v = _FakeTensor(q.shape, q.dtype, npu_device)
    cu = _FakeTensor((3,), torch.int32, npu_device)

    with (
        patch.object(mm_attention, "HAS_TRITON", True),
        patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_VIT_TRITON_FA": "1"}),
    ):
        assert layer._can_use_vit_triton_fa(q, k, v, cu)

        q.dtype = torch.float16
        assert not layer._can_use_vit_triton_fa(q, k, v, cu)
        q.dtype = torch.bfloat16

        k.shape = (127, 2, 72)
        assert not layer._can_use_vit_triton_fa(q, k, v, cu)
        k.shape = q.shape

        cu.dtype = torch.float32
        assert not layer._can_use_vit_triton_fa(q, k, v, cu)

    with (
        patch.object(mm_attention, "HAS_TRITON", False),
        patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_VIT_TRITON_FA": "1"}),
    ):
        cu.dtype = torch.int32
        assert not layer._can_use_vit_triton_fa(q, k, v, cu)


def test_unsupported_invocation_falls_back_to_padded_fia():
    from vllm_ascend.ops.mm_encoder_attention import AscendMMEncoderAttention

    layer = AscendMMEncoderAttention.__new__(AscendMMEncoderAttention)
    layer.head_size = 72
    layer.num_heads = 2
    layer.num_kv_heads = 2
    layer.enable_pad = True
    layer.scale = 72**-0.5

    q = torch.empty(8, 2, 72, dtype=torch.float16)
    cu = torch.tensor([0, 8], dtype=torch.int32)

    def fake_fia(q_padded, k_padded, v_padded, *_args):
        assert q_padded.shape == k_padded.shape == v_padded.shape == (8, 2, 128)
        return torch.zeros_like(q_padded)

    with (
        patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_VIT_TRITON_FA": "1"}),
        patch.object(layer, "_can_use_vit_triton_fa", return_value=False),
        patch.object(layer, "_run_vit_triton") as triton_path,
        patch.object(layer, "_run_vit_fia", side_effect=fake_fia) as fia_path,
        patch("vllm_ascend.ops.mm_encoder_attention.logger.warning_once") as fallback_log,
    ):
        result = layer._forward_eager_fia(
            q,
            q,
            q,
            cu_seqlens=cu,
            is_reshaped=True,
            bsz=1,
            q_len=8,
        )

    triton_path.assert_not_called()
    fia_path.assert_called_once()
    fallback_log.assert_called_once()
    assert "falling back" in fallback_log.call_args.args[0]
    assert result.shape == (1, 8, 2, 72)
