# FastNPP vs NPP — Benchmark Results

Measured on RTX PRO 6000 Blackwell (sm_120), CUDA 13.3, 1920x1080, median of 200
iterations after warmup, CUDA-event timing on a dedicated stream. Every fused
result is cross-checked against the NPP reference, so timings refer to identical
work. Reproduce with:

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_BENCHMARK=ON
cmake --build build --target benchmark_fastnpp_vs_npp -j
./build/bin/Release/benchmark_fastnpp_vs_npp
```

## Results

| Operation | NPP | FastNPP | Speedup | Category |
|-----------|-----|---------|---------|----------|
| AddC_32f_C1R | 0.0058 ms | 0.0058 ms | 1.00x | element-wise parity |
| Sqrt_32f_C1R | 0.0058 ms | 0.0058 ms | 0.99x | element-wise parity |
| AddC→MulC→SubC→DivC (4 ops) | 0.0191 ms | 0.0058 ms | **3.29x** | fusion |
| DilateBorder_8u_C1R 3x3 | 0.0058 ms | 0.0180 ms | 0.32x | neighbourhood (small) |
| FilterBoxBorder_8u_C1R 3x3 | 0.0051 ms | 0.0133 ms | 0.38x | neighbourhood (small) |
| FilterBoxBorder_8u_C1R 9x9 | 0.0473 ms | 0.0400 ms | **1.18x** | neighbourhood (large) |

## Interpretation

**Element-wise fusion is the headline win.** A chain of N element-wise operations
collapses into a single FKL kernel — one global read, one write — instead of N
NPP launches with N intermediate round-trips. The 4-op arithmetic chain runs
**3.29x** faster than four separate NPP calls. Single element-wise ops are at
parity (both saturate bandwidth).

**Large neighbourhood filters are also ahead.** At 9x9 the box filter is **1.18x**
faster than NPP: with enough work per thread, FKL's single-kernel approach beats
NPP's launch. (An interior fast-path skips per-tap border clamping for the common
case where the whole window is in-bounds.)

**Small neighbourhood filters (3x3) trail NPP** (~0.35x). This was profiled with
Nsight Compute: at 3x3 the FKL kernel is **latency-bound** (SM ~49%, DRAM ~8% — it
saturates nothing), doing one output pixel per thread. NPP's small-filter kernels
process **4 pixels per thread** ("Quad" variants), keeping more work in flight to
hide latency. Closing this specific gap requires multiple-elements-per-thread
support in the FKL executor (not an op-level change) and is tracked as future work.

## Summary

- All operations are **bit-exact vs NPP** (correctness parity across every family).
- **Element-wise pipelines: materially faster** via fusion (3.3x on a 4-op chain).
- **Large filters: faster** (1.18x at 9x9).
- **Small filters: slower** (latency-bound, 1 px/thread) — a known, profiled
  optimization target (multi-pixel-per-thread in the executor).
