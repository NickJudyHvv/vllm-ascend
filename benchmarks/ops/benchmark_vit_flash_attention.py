# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark the Qwen3.5-VL head_dim=72 Triton attention path on NPU.

The FIA baseline matches the NPU work in the production eager path: physically
pad Q/K/V from 72 to 128, run FIA, then materialize the sliced 72-wide output.
Host metadata conversion and ACL graph capture are intentionally outside this
device-event microbenchmark. The Triton side uses the exact production
configuration and runs without physical padding. The script reports timing
distributions rather than enforcing a hardware-dependent performance gate.
By default FIA and Triton are timed in separate sustained bursts, with the
first variant alternated every epoch to reduce order, DVFS, and temperature
bias. ``--interleaved`` remains available for diagnostic ABBA/BAAB timing.

Example::

    python3 benchmarks/ops/benchmark_vit_flash_attention.py

The default case uses two packed 512-token sequences (T=1024 total) and prints
the result to the terminal. Command-line options remain available for
additional eager cases. Tile overrides evaluate exactly one Triton candidate
against FIA, so the normal output stays a two-way comparison. For example,
the A5 exact V80 candidate is:

    python3 benchmarks/ops/benchmark_vit_flash_attention.py \
      --seq-lens 1024 --block-m 128 --block-n 128 --qk-pad 80 --v-pad 80

To collect separate FIA and Triton device profiles for the default production
case, without running the long timing benchmark::

    python3 benchmarks/ops/benchmark_vit_flash_attention.py --profile
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.metadata
import math
import statistics
import subprocess
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
import torch_npu

from vllm_ascend.ops.triton.vit_flash_attention import (
    A5_EXACT_T_SPECIALIZATION_ENABLED,
    A5_FULL_ALIGNED_HEADS,
    A5_FULL_ALIGNED_TOKENS,
    VIT_FA_A5_FULL_ALIGNED_CONFIG,
    VitFlashAttentionConfig,
    get_vit_flash_attention_config,
    vit_flash_attention_fwd,
)

HEAD_DIM = 72
FIA_HEAD_DIM = 128
FIA_BLOCK_SIZE = 128
SWA_INT_MAX = 2_147_483_647
# Packed two-image case (num_seqs=2, total=1024) probes the VARLEN path
# (SINGLE_SEQUENCE=False, dynamic cu_seqlens enumeration) using the real
# SoC-selected production configuration.
DEFAULT_SEQ_LENS = ("512,512",)
DEFAULT_WARMUP_SECONDS = 2.0
DEFAULT_CYCLES_PER_EPOCH = 24
DEFAULT_REPEATS = 20
DEFAULT_MIN_EPOCHS = 13
DEFAULT_MAX_EPOCHS = 25
MAX_RELATIVE_EPOCH_MAD = 0.015
MAX_RELATIVE_EPOCH_P10_P90_SPAN = 0.05
MAX_RELATIVE_EPOCH_HALF_DRIFT = 0.02
DEFAULT_PROFILE_DIR = "vit_fa_profile"
DEFAULT_PROFILE_RUNS = 5
PROFILE_WARMUP_RUNS = 5
PROFILE_TOP_KERNELS = 3


def _parse_seq_lens(value: str) -> list[int]:
    seq_lens = [int(item) for item in value.split(",")]
    if not seq_lens or any(length <= 0 for length in seq_lens):
        raise argparse.ArgumentTypeError("sequence lengths must be positive comma-separated integers")
    return seq_lens


def _percentile(sorted_values: list[float], quantile: float) -> float:
    index = (len(sorted_values) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _summarize(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    median = statistics.median(ordered)
    deviations = [abs(value - median) for value in ordered]
    return {
        "min_ms": ordered[0],
        "p10_ms": _percentile(ordered, 0.10),
        "p50_ms": median,
        "p90_ms": _percentile(ordered, 0.90),
        "mad_ms": statistics.median(deviations),
    }


def _relative_mad(values: list[float]) -> float:
    median = statistics.median(values)
    if median == 0.0:
        return float("inf")
    return statistics.median(abs(value - median) for value in values) / abs(median)


def _epoch_ratios_are_stable(values: list[float]) -> bool:
    """Reject noisy, bimodal, and time-drifting within-run speedup samples."""
    if len(values) < 3:
        return False
    ordered = sorted(values)
    median = statistics.median(ordered)
    if median == 0.0:
        return False
    p10 = _percentile(ordered, 0.10)
    p90 = _percentile(ordered, 0.90)
    relative_span = (p90 - p10) / abs(median)
    midpoint = len(values) // 2
    first_half = statistics.median(values[:midpoint])
    second_half = statistics.median(values[midpoint:])
    relative_half_drift = abs(first_half - second_half) / abs(median)
    return (
        _relative_mad(values) <= MAX_RELATIVE_EPOCH_MAD
        and relative_span <= MAX_RELATIVE_EPOCH_P10_P90_SPAN
        and relative_half_drift <= MAX_RELATIVE_EPOCH_HALF_DRIFT
    )


def _enqueue_timed_segment(
    fn: Callable[[], torch.Tensor],
    repeats: int,
) -> tuple[torch.npu.Event, torch.npu.Event]:
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    return start, end


def _time_epoch(
    variants: tuple[Callable[[], torch.Tensor], Callable[[], torch.Tensor]],
    *,
    epoch_idx: int,
    cycles: int,
    repeats: int,
) -> tuple[list[float], list[float]]:
    """Enqueue a full epoch continuously, then synchronize only once."""
    orders = ((0, 1, 1, 0), (1, 0, 0, 1))
    cycle_events = []
    for cycle_idx in range(cycles):
        events_by_variant = ([], [])
        order = orders[(epoch_idx + cycle_idx) % len(orders)]
        for variant_idx in order:
            events_by_variant[variant_idx].append(
                _enqueue_timed_segment(variants[variant_idx], repeats)
            )
        cycle_events.append(events_by_variant)

    torch.npu.synchronize()
    cycle_timings = ([], [])
    for events_by_variant in cycle_events:
        for variant_idx in range(2):
            segment_timings = [
                start.elapsed_time(end) / repeats for start, end in events_by_variant[variant_idx]
            ]
            cycle_timings[variant_idx].append(statistics.mean(segment_timings))
    return cycle_timings


def _time_epoch_isolated(
    variants: tuple[Callable[[], torch.Tensor], Callable[[], torch.Tensor]],
    *,
    epoch_idx: int,
    cycles: int,
    repeats: int,
) -> tuple[list[float], list[float]]:
    """Time each variant in a sustained back-to-back burst (no interleaving).

    Interleaving FIA and Triton per cycle thrashes L2 between their working
    sets and produces bimodal FIA latency on shared NPU boxes. Isolating each
    variant removes that alternation artifact so the two absolute latencies can
    be compared at their true steady state. The first burst alternates by epoch
    so FIA is not always measured on a colder or hotter device than Triton.
    """
    events_by_variant: tuple[list[tuple[torch.npu.Event, torch.npu.Event]], ...] = ([], [])
    order = (0, 1) if epoch_idx % 2 == 0 else (1, 0)
    for variant_idx in order:
        events_by_variant[variant_idx].extend(
            _enqueue_timed_segment(variants[variant_idx], repeats) for _ in range(cycles)
        )
    torch.npu.synchronize()
    baseline_timings = [s.elapsed_time(e) / repeats for s, e in events_by_variant[0]]
    candidate_timings = [s.elapsed_time(e) / repeats for s, e in events_by_variant[1]]
    return baseline_timings, candidate_timings


def _warmup_pair(
    variants: tuple[Callable[[], torch.Tensor], Callable[[], torch.Tensor]],
    *,
    warmup_seconds: float,
    cycles: int,
    repeats: int,
) -> None:
    """Keep the NPU busy long enough for allocator and frequency stabilization."""
    if warmup_seconds == 0.0:
        return

    orders = ((0, 1, 1, 0), (1, 0, 0, 1))
    warmup_repeats = max(1, repeats // 2)
    started = time.monotonic()
    batch_idx = 0
    while True:
        for cycle_idx in range(cycles):
            order = orders[(batch_idx + cycle_idx) % len(orders)]
            for variant_idx in order:
                for _ in range(warmup_repeats):
                    variants[variant_idx]()
        torch.npu.synchronize()
        batch_idx += 1
        if time.monotonic() - started >= warmup_seconds:
            return


def _warmup_pair_isolated(
    variants: tuple[Callable[[], torch.Tensor], Callable[[], torch.Tensor]],
    *,
    warmup_seconds: float,
    repeats: int,
) -> None:
    """Sustained per-variant warmup so each settles its own cache/DVFS state."""
    if warmup_seconds == 0.0:
        return
    warmup_repeats = max(1, repeats // 2)
    for variant in variants:
        started = time.monotonic()
        while True:
            for _ in range(warmup_repeats):
                variant()
            torch.npu.synchronize()
            if time.monotonic() - started >= warmup_seconds:
                break


def _time_pair(
    baseline: Callable[[], torch.Tensor],
    candidate: Callable[[], torch.Tensor],
    *,
    warmup_seconds: float,
    cycles: int,
    repeats: int,
    min_epochs: int,
    max_epochs: int,
    isolated: bool = False,
) -> tuple[list[float], list[float], list[float], bool]:
    baseline_timings: list[float] = []
    candidate_timings: list[float] = []
    epoch_speedups: list[float] = []
    variants = (baseline, candidate)
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    try:
        if isolated:
            _warmup_pair_isolated(
                variants,
                warmup_seconds=warmup_seconds,
                repeats=repeats,
            )
        else:
            _warmup_pair(
                variants,
                warmup_seconds=warmup_seconds,
                cycles=cycles,
                repeats=repeats,
            )
        for epoch_idx in range(max_epochs):
            if isolated:
                epoch_baseline, epoch_candidate = _time_epoch_isolated(
                    variants,
                    epoch_idx=epoch_idx,
                    cycles=cycles,
                    repeats=repeats,
                )
            else:
                epoch_baseline, epoch_candidate = _time_epoch(
                    variants,
                    epoch_idx=epoch_idx,
                    cycles=cycles,
                    repeats=repeats,
                )
            baseline_timings.extend(epoch_baseline)
            candidate_timings.extend(epoch_candidate)
            epoch_speedups.append(sum(epoch_baseline) / sum(epoch_candidate))
            if len(epoch_speedups) >= min_epochs and _epoch_ratios_are_stable(epoch_speedups):
                break
    finally:
        if gc_was_enabled:
            gc.enable()

    stable = _epoch_ratios_are_stable(epoch_speedups)
    return baseline_timings, candidate_timings, epoch_speedups, stable


def _make_inputs(total_tokens: int, num_heads: int, device: str) -> tuple[torch.Tensor, ...]:
    shape = (total_tokens, num_heads, HEAD_DIM)
    return tuple(torch.randn(shape, dtype=torch.bfloat16, device=device) for _ in range(3))


def _make_cu_seqlens(seq_lens: list[int], device: str) -> torch.Tensor:
    endpoints = [0]
    for length in seq_lens:
        endpoints.append(endpoints[-1] + length)
    return torch.tensor(endpoints, dtype=torch.int32, device=device)


def _fia_production_path(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    actual_seq_lengths: list[int],
    num_heads: int,
) -> torch.Tensor:
    q_padded = F.pad(q, (0, FIA_HEAD_DIM - HEAD_DIM))
    k_padded = F.pad(k, (0, FIA_HEAD_DIM - HEAD_DIM))
    v_padded = F.pad(v, (0, FIA_HEAD_DIM - HEAD_DIM))
    output, _ = torch_npu.npu_fused_infer_attention_score(
        query=q_padded,
        key=k_padded,
        value=v_padded,
        atten_mask=None,
        block_table=None,
        input_layout="TND",
        block_size=FIA_BLOCK_SIZE,
        actual_seq_lengths=actual_seq_lengths,
        actual_seq_lengths_kv=actual_seq_lengths,
        num_key_value_heads=num_heads,
        num_heads=num_heads,
        scale=HEAD_DIM**-0.5,
        sparse_mode=0,
        pre_tokens=SWA_INT_MAX,
        next_tokens=SWA_INT_MAX,
    )
    return output[..., :HEAD_DIM].contiguous()


def _normalize_csv_header(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def _find_csv_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    normalized = {_normalize_csv_header(field): field for field in fieldnames}
    for candidate in candidates:
        field = normalized.get(_normalize_csv_header(candidate))
        if field is not None:
            return field
    return None


def _parse_csv_number(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = value.strip().rstrip("%").replace(",", "")
    if not stripped or stripped.lower() in {"nan", "n/a", "na", "--"}:
        return None
    try:
        number = float(stripped)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _ratio_percent(value: str | None) -> float | None:
    number = _parse_csv_number(value)
    if number is None:
        return None
    # torch_npu profiler versions have emitted both [0, 1] ratios and [0, 100]
    # percentages. Normalize both forms before displaying the summary.
    return number * 100.0 if abs(number) <= 1.0 else number


def _summarize_kernel_details(
    csv_paths: list[Path],
    profile_runs: int,
) -> dict[str, object]:
    """Aggregate profiler CSVs without adding pandas as a benchmark dependency."""
    total_duration_us = 0.0
    row_count = 0
    columns: set[str] = set()
    kernel_totals: dict[str, float] = {}
    kernel_counts: dict[str, int] = {}
    ratio_weighted_sums = {
        "aic_mac_ratio": 0.0,
        "aic_mte2_ratio": 0.0,
        "aiv_vec_ratio": 0.0,
        "aiv_mte2_ratio": 0.0,
        "aiv_mte3_ratio": 0.0,
    }
    ratio_duration_sums = {name: 0.0 for name in ratio_weighted_sums}

    for csv_path in csv_paths:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            fieldnames = reader.fieldnames or []
            columns.update(fieldnames)
            duration_column = _find_csv_column(
                fieldnames,
                ("Duration(us)", "Duration (us)", "Task Duration(us)", "Duration"),
            )
            name_column = _find_csv_column(
                fieldnames,
                ("Name", "Kernel Name", "Op Name", "Model Name"),
            )
            if duration_column is None:
                raise RuntimeError(f"Duration(us) column is missing from {csv_path}")

            for row in reader:
                duration_us = _parse_csv_number(row.get(duration_column))
                if duration_us is None or duration_us < 0.0:
                    continue
                name_value = row.get(name_column, "") if name_column is not None else ""
                name = (name_value or "").strip() or "<unnamed>"
                total_duration_us += duration_us
                row_count += 1
                kernel_totals[name] = kernel_totals.get(name, 0.0) + duration_us
                kernel_counts[name] = kernel_counts.get(name, 0) + 1
                for ratio_name in ratio_weighted_sums:
                    ratio = _ratio_percent(row.get(ratio_name))
                    if ratio is not None:
                        ratio_weighted_sums[ratio_name] += ratio * duration_us
                        ratio_duration_sums[ratio_name] += duration_us

    if row_count == 0:
        raise RuntimeError(f"no valid kernel rows found in: {', '.join(map(str, csv_paths))}")

    top_kernels = []
    for name, total_us in sorted(kernel_totals.items(), key=lambda item: item[1], reverse=True)[
        :PROFILE_TOP_KERNELS
    ]:
        top_kernels.append(
            {
                "name": name,
                "per_call_us": total_us / profile_runs,
                "count": kernel_counts[name],
            }
        )
    pipe_ratios_pct = {
        name: ratio_weighted_sums[name] / ratio_duration_sums[name]
        for name in ratio_weighted_sums
        if ratio_duration_sums[name] > 0.0
    }
    return {
        "csv_paths": csv_paths,
        "columns": sorted(columns),
        "row_count": row_count,
        "device_us_per_call": total_duration_us / profile_runs,
        "kernel_rows_per_call": row_count / profile_runs,
        "pipe_ratios_pct": pipe_ratios_pct,
        "top_kernels": top_kernels,
    }


def _analyse_profile_output(profile_dir: Path, profile_runs: int) -> dict[str, object]:
    csv_paths = sorted(profile_dir.rglob("kernel_details.csv"))
    if not csv_paths:
        raw_dirs = sorted(path for path in profile_dir.rglob("*_ascend_pt") if path.is_dir())
        if not raw_dirs:
            raise RuntimeError(f"torch_npu profiler did not create an *_ascend_pt directory under {profile_dir}")
        from torch_npu.profiler.profiler import analyse

        for raw_dir in raw_dirs:
            analyse(str(raw_dir))
        csv_paths = sorted(profile_dir.rglob("kernel_details.csv"))
    if not csv_paths:
        raise RuntimeError(
            "torch_npu profiler analysis completed without kernel_details.csv; "
            f"raw data is preserved under {profile_dir}"
        )
    return _summarize_kernel_details(csv_paths, profile_runs)


def _profile_variant(
    fn: Callable[[], torch.Tensor],
    *,
    label: str,
    profile_dir: Path,
    profile_runs: int,
) -> dict[str, object]:
    """Warm up and collect exactly one variant in its own profiler context."""
    profile_dir.mkdir(parents=True, exist_ok=True)
    for _ in range(PROFILE_WARMUP_RUNS):
        fn()
    torch.npu.synchronize()

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=torch_npu.profiler.ExportType.Text,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        msprof_tx=False,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        l2_cache=False,
        op_attr=False,
        data_simplification=True,
        record_op_args=False,
        gc_detect_threshold=None,
    )
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        with_stack=False,
        profile_memory=False,
        with_modules=False,
        record_shapes=False,
        experimental_config=experimental_config,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
            str(profile_dir),
            worker_name=label,
        ),
    ) as profiler:
        for _ in range(profile_runs):
            fn()
            profiler.step()
        torch.npu.synchronize()

    summary = _analyse_profile_output(profile_dir, profile_runs)
    summary["profile_dir"] = profile_dir
    return summary


def _new_profile_root(base_dir: Path) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    suffix = time.time_ns() % 1_000_000
    profile_root = base_dir.expanduser().resolve() / f"{timestamp}-{suffix:06d}"
    profile_root.mkdir(parents=True, exist_ok=False)
    return profile_root


@torch.inference_mode()
def _run_case(
    args: argparse.Namespace,
    seq_lens: list[int],
    *,
    profile_root: Path | None = None,
    case_index: int = 0,
) -> dict[str, object]:
    valid_tokens = sum(seq_lens)
    q, k, v = _make_inputs(valid_tokens, args.num_heads, args.device)
    cu_seqlens = _make_cu_seqlens(seq_lens, args.device)
    # Mirror AscendMMEncoderAttention's eager routing contract. The exact-T
    # compiler specialization remains behind a precision gate; while disabled,
    # T=1024/H=16 uses the same verified dynamic-T A5 production kernel.
    full_aligned_shape = (
        len(seq_lens) == 1
        and valid_tokens == A5_FULL_ALIGNED_TOKENS
        and args.num_heads == A5_FULL_ALIGNED_HEADS
    )
    production_config = get_vit_flash_attention_config(full_aligned=full_aligned_shape)
    full_aligned = (
        full_aligned_shape
        and production_config == VIT_FA_A5_FULL_ALIGNED_CONFIG
        and A5_EXACT_T_SPECIALIZATION_ENABLED
    )
    has_config_override = args.per_task_grid or any(
        value is not None for value in (args.block_m, args.block_n, args.qk_pad, args.v_pad)
    )
    config = VitFlashAttentionConfig(
        block_m=production_config.block_m if args.block_m is None else args.block_m,
        block_n=production_config.block_n if args.block_n is None else args.block_n,
        qk_pad=production_config.qk_pad if args.qk_pad is None else args.qk_pad,
        v_pad=production_config.v_pad if args.v_pad is None else args.v_pad,
    )
    path_kind = "A5-exact" if full_aligned else "varlen"
    path_label = (
        f"{path_kind}-BM{config.block_m}-BN{config.block_n}-"
        f"QK{config.qk_pad}-V{config.v_pad}"
    )
    actual_seq_lengths = []
    endpoint = 0
    for length in seq_lens:
        endpoint += length
        actual_seq_lengths.append(endpoint)

    def baseline() -> torch.Tensor:
        return _fia_production_path(q, k, v, actual_seq_lengths, args.num_heads)

    def candidate() -> torch.Tensor:
        return vit_flash_attention_fwd(
            q,
            k,
            v,
            cu_seqlens,
            HEAD_DIM**-0.5,
            block_m=config.block_m,
            block_n=config.block_n,
            qk_pad=config.qk_pad,
            v_pad=config.v_pad,
            full_aligned=full_aligned,
            clear_tail=False,
            one_program_per_task=args.per_task_grid,
        )

    baseline_output = baseline()
    candidate_output = candidate()
    torch.npu.synchronize()
    torch.testing.assert_close(candidate_output, baseline_output, atol=3e-2, rtol=3e-2)
    del baseline_output, candidate_output
    common_result = {
        "seq_lens": seq_lens,
        "valid_tokens": valid_tokens,
        "num_seqs": len(seq_lens),
        "config": config,
        "config_source": "candidate" if has_config_override else "production",
        "grid_mode": "per-task" if args.per_task_grid else "persistent-core",
        "path": path_label,
    }
    if args.profile:
        if profile_root is None:
            raise RuntimeError("profile_root is required in --profile mode")
        case_dir = profile_root / (
            f"case-{case_index + 1:02d}-T{valid_tokens}-B{len(seq_lens)}"
        )
        profiles = {
            "FIA": _profile_variant(
                baseline,
                label="fia",
                profile_dir=case_dir / "fia",
                profile_runs=args.profile_runs,
            ),
            "Triton": _profile_variant(
                candidate,
                label="triton",
                profile_dir=case_dir / "triton",
                profile_runs=args.profile_runs,
            ),
        }
        return {
            **common_result,
            "profile_dir": case_dir,
            "profiles": profiles,
        }

    baseline_samples, candidate_samples, epoch_speedups, stable = _time_pair(
        baseline,
        candidate,
        warmup_seconds=args.warmup,
        cycles=args.samples,
        repeats=args.repeats,
        min_epochs=args.epochs,
        max_epochs=args.max_epochs,
        isolated=args.isolated,
    )
    baseline_stats = _summarize(baseline_samples)
    candidate_stats = _summarize(candidate_samples)
    ordered_speedups = sorted(epoch_speedups)

    return {
        **common_result,
        "fia": baseline_stats,
        "triton": candidate_stats,
        "speedup": statistics.median(epoch_speedups),
        "speedup_p10": _percentile(ordered_speedups, 0.10),
        "speedup_p90": _percentile(ordered_speedups, 0.90),
        "speedup_mad_pct": _relative_mad(epoch_speedups) * 100.0,
        "epochs": len(epoch_speedups),
        "stable": stable,
    }


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _soc_version() -> str:
    try:
        return str(torch_npu.npu.get_soc_version())
    except (AttributeError, RuntimeError):
        return "unknown"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-lens", action="append", type=_parse_seq_lens, help="CSV lengths; repeat per case")
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--block-m", type=int, help="override production query tile for one candidate")
    parser.add_argument("--block-n", type=int, help="override production KV tile for one candidate")
    parser.add_argument("--qk-pad", type=int, help="override production logical Q/K width")
    parser.add_argument("--v-pad", type=int, help="override production logical V/output width")
    parser.add_argument(
        "--per-task-grid",
        action="store_true",
        help="experimental eager-only grid with one Triton program per safe task slot",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="profile FIA and Triton separately, print a compact summary, and skip timing",
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=Path(DEFAULT_PROFILE_DIR),
        help=f"raw profiler output root (default: ./{DEFAULT_PROFILE_DIR})",
    )
    parser.add_argument(
        "--profile-runs",
        type=int,
        default=DEFAULT_PROFILE_RUNS,
        help="calls captured separately for FIA and Triton",
    )
    parser.add_argument("--warmup", type=float, default=DEFAULT_WARMUP_SECONDS, help="sustained warmup seconds")
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_CYCLES_PER_EPOCH,
        help="timing cycles per epoch",
    )
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_MIN_EPOCHS, help="minimum timing epochs")
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    timing_group = parser.add_mutually_exclusive_group()
    timing_group.add_argument(
        "--isolated",
        dest="isolated",
        action="store_true",
        default=True,
        help="time FIA and Triton in separate sustained bursts (default)",
    )
    timing_group.add_argument(
        "--interleaved",
        dest="isolated",
        action="store_false",
        help="diagnostic alternating ABBA/BAAB timing",
    )
    return parser.parse_args()


def _format_pipe_ratio(pipe_ratios: dict[str, float], column: str) -> str:
    ratio = pipe_ratios.get(column)
    return "NA" if ratio is None else f"{ratio:.1f}%"


def _format_profile_result(result: dict[str, object]) -> str:
    config = result["config"]
    profiles = result["profiles"]
    fia_us = profiles["FIA"]["device_us_per_call"]
    triton_summary = profiles["Triton"]
    triton_us = triton_summary["device_us_per_call"]
    profile_ratio = fia_us / triton_us
    pipe_ratios = triton_summary["pipe_ratios_pct"]
    seq_lens = "+".join(str(length) for length in result["seq_lens"])
    return (
        f"PROFILE_RESULT seq={seq_lens} "
        f"cfg=BM{config.block_m}-BN{config.block_n}-QK{config.qk_pad}-V{config.v_pad} "
        f"fia_us={fia_us:.3f} tri_us={triton_us:.3f} prof_ratio={profile_ratio:.3f}x "
        f"mac={_format_pipe_ratio(pipe_ratios, 'aic_mac_ratio')} "
        f"aic_mte2={_format_pipe_ratio(pipe_ratios, 'aic_mte2_ratio')} "
        f"vec={_format_pipe_ratio(pipe_ratios, 'aiv_vec_ratio')} "
        f"aiv_mte2={_format_pipe_ratio(pipe_ratios, 'aiv_mte2_ratio')} "
        f"aiv_mte3={_format_pipe_ratio(pipe_ratios, 'aiv_mte3_ratio')}"
    )


def main() -> None:
    args = _parse_args()
    if args.profile_runs <= 0:
        raise ValueError("profile-runs must be positive")
    if not args.profile and (
        not math.isfinite(args.warmup)
        or args.warmup < 0
        or args.samples <= 0
        or args.repeats <= 0
    ):
        raise ValueError("warmup must be finite and non-negative; samples and repeats must be positive")
    if not args.profile and (args.epochs < 3 or args.max_epochs < args.epochs):
        raise ValueError("epochs must be at least 3 and max-epochs must be at least epochs")
    if args.num_heads <= 0:
        raise ValueError("num_heads must be positive")
    torch.npu.set_device(args.device)
    torch.manual_seed(0)
    torch.npu.manual_seed_all(0)

    seq_lens_cases = args.seq_lens or [_parse_seq_lens(value) for value in DEFAULT_SEQ_LENS]
    profile_root = _new_profile_root(args.profile_dir) if args.profile else None
    if not args.profile:
        print("Qwen3.5-VL ViT head_dim=72 production-path benchmark")
        print(
            f"git={_git_sha()}, device={args.device}, soc={_soc_version()}, "
            f"torch={torch.__version__}, torch-npu={_package_version('torch-npu')}, "
            f"triton-ascend={_package_version('triton-ascend')}"
        )
        timing_mode = "isolated alternating-first" if args.isolated else "interleaved ABBA/BAAB"
        print(f"timing_mode={timing_mode}; {args.repeats} calls per timed segment")
    for case_index, seq_lens in enumerate(seq_lens_cases):
        if args.profile:
            profile_log = profile_root / "profile.log"
            with (
                profile_log.open("a", encoding="utf-8") as log_file,
                redirect_stdout(log_file),
                redirect_stderr(log_file),
            ):
                result = _run_case(
                    args,
                    seq_lens,
                    profile_root=profile_root,
                    case_index=case_index,
                )
            print(_format_profile_result(result))
            continue

        result = _run_case(args, seq_lens)
        config = result["config"]
        print(f"\nseq_lens={result['seq_lens']}, T={result['valid_tokens']}, H={args.num_heads}, D={HEAD_DIM}")
        print(
            f"Triton {result['config_source']} path={result['path']}: "
            f"BM={config.block_m}, BN={config.block_n}, "
            f"D_QK_PAD={config.qk_pad}, D_V_PAD={config.v_pad}, "
            f"grid_mode={result['grid_mode']}, PV_LAYOUT=standard, multibuffer=True"
        )
        print("Correctness: PASS (atol=3e-2, rtol=3e-2)")
        fia = result["fia"]
        triton_result = result["triton"]
        speedup = result["speedup"]
        stability = "in-run stable" if result["stable"] else "UNSTABLE"
        print(
            f"FIA (pad-to-128 + slice): p50={fia['p50_ms']:.6f} ms, "
            f"p90={fia['p90_ms']:.6f} ms, MAD={fia['mad_ms']:.6f} ms"
        )
        print(
            "Triton: "
            f"p50={triton_result['p50_ms']:.6f} ms, "
            f"p90={triton_result['p90_ms']:.6f} ms, MAD={triton_result['mad_ms']:.6f} ms, "
            f"epoch speedup (FIA / Triton)={speedup:.4f}x "
            f"({stability}, epoch p10-p90={result['speedup_p10']:.4f}-{result['speedup_p90']:.4f}x, "
            f"MAD={result['speedup_mad_pct']:.2f}%, epochs={result['epochs']})"
        )


if __name__ == "__main__":
    main()
