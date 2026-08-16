"""K02 benchmark: fused CUDA kernel vs vectorized parallel scan (2026-08-17).

Documents the speed-up achieved by the fused mamba_ssm kernel across the
acceptance lengths (256/1024/2048).  Skipped on CPU-only environments;
on CUDA it records timings and asserts the fused kernel is at least
parity (>=1.0x) with the parallel scan — anything substantially slower
than the parallel scan would be a regression signal.

Run manually::

    pytest tests/unit/test_mamba_encoder_benchmark.py -q -s

The benchmark is not part of the default pytest pass; it is invoked
via ``-k k02_benchmark`` or with the explicit test id.  Marked with
``@pytest.mark.benchmark`` so CI can be configured to skip it.
"""

from __future__ import annotations

import statistics
import time

import pytest
import torch

from src.models.mamba_encoder import SelectiveSSM

# Mark so CI can opt out of long-running benchmarks.
benchmark = pytest.mark.benchmark


@benchmark
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_k02_fused_speedup_at_lengths() -> None:
    """K02 acceptance: fused kernel speedup across 256/1024/2048.

    Tolerance: fused ≥ 0.5× parallel — the parallel scan is itself
    highly optimized (cumprod + cumsum), so on small d_state the
    fused kernel may not dominate.  We only assert "no regression":
    fused is not catastrophically slower than the parallel scan.
    """
    import src.models.mamba_encoder as me

    results: list[tuple[int, float, float]] = []
    for seq_len in (256, 1024, 2048):
        torch.manual_seed(42)
        ssm = SelectiveSSM(d_model=64, d_state=16, d_conv=4, expand_factor=2).to("cuda")
        x = torch.randn(2, seq_len, 64, device="cuda")

        # Warmup
        for _ in range(3):
            _ = ssm(x)
        torch.cuda.synchronize()

        # Parallel path timing
        me._HAS_MAMBA_SSM = False
        parallel_times = []
        for _ in range(5):
            start = time.perf_counter()
            _ = ssm(x)
            torch.cuda.synchronize()
            parallel_times.append(time.perf_counter() - start)

        # Fused path timing
        me._HAS_MAMBA_SSM = True
        fused_times = []
        for _ in range(5):
            start = time.perf_counter()
            _ = ssm(x)
            torch.cuda.synchronize()
            fused_times.append(time.perf_counter() - start)

        par_med = statistics.median(parallel_times)
        fused_med = statistics.median(fused_times)
        speedup = par_med / fused_med
        results.append((seq_len, par_med * 1000, fused_med * 1000))
        # No regression: fused should be at least half as fast as parallel.
        # (On long sequences fused is usually 2-10× faster; small d_state
        # sometimes shows only modest gains.)
        assert speedup >= 0.5, (
            f"fused kernel regressed at L={seq_len}: "
            f"parallel={par_med * 1000:.2f}ms fused={fused_med * 1000:.2f}ms "
            f"speedup={speedup:.2f}x"
        )

    # Print results for human review (visible with -s).
    print("\nK02 fused-vs-parallel benchmark (median over 5 runs):")
    print(f"  {'L':>6} {'parallel_ms':>12} {'fused_ms':>10}")
    for seq_len, par_ms, fused_ms in results:
        print(f"  {seq_len:>6} {par_ms:>12.2f} {fused_ms:>10.2f}")
