# ruff: noqa: E402
# benchmark/benchmark_histogram_speed.py

import os
import sys
import time
import numpy as np
import pandas as pd

# make package importable without install -e
THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import cupy as cp
from skimage import exposure as skexp

# Optional GPU baseline: cuCIM (keeps everything on-GPU, channel-last)
try:
    from cucim.skimage import exposure as cucexp

    _HAS_CUCIM = True
except Exception:
    _HAS_CUCIM = False

from huetuber.match_histogram import HistogramMatching


def _sync():
    cp.cuda.Stream.null.synchronize()


def _run_gpu_once(norm: HistogramMatching, src_bchw_cp: cp.ndarray) -> float:
    _sync()
    t0 = time.perf_counter()
    _ = norm.normalize(src_bchw_cp)
    _sync()
    return time.perf_counter() - t0


def _run_cucim_once(src_hwc_cp: cp.ndarray, target_hwc_cp: cp.ndarray) -> float:
    """
    src_hwc_cp: (B,H,W,C) cupy uint8
    target_hwc_cp: (H,W,C) cupy uint8
    Runs cuCIM per-image (cuCIM API is image-wise), all on-GPU.
    """
    _sync()
    t0 = time.perf_counter()
    B = int(src_hwc_cp.shape[0])
    for i in range(B):
        _ = cucexp.match_histograms(src_hwc_cp[i], target_hwc_cp, channel_axis=-1)
    _sync()
    return time.perf_counter() - t0


def run_hist_match_speed(
    sizes=(512, 1024, 2048),
    batches=(1, 4, 16),
    repeat=5,
    seed=0,
):
    rng = np.random.default_rng(seed)
    results = []

    if not _HAS_CUCIM:
        print("[WARN] cuCIM not importable; will skip cuCIM timings and write NaNs.")

    for H in sizes:
        W = H

        # Build target (CPU HWC) + GPU BCHW + GPU HWC
        target_np = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
        target_bchw_cp = cp.asarray(target_np).transpose(2, 0, 1)[None]  # (1,3,H,W)
        target_hwc_cp = cp.asarray(target_np)  # (H,W,C)

        # Initialize GPU normalizer (HueTuber)
        norm = HistogramMatching(channel_axis=1, dtype_out=cp.uint8)
        norm.fit(target_bchw_cp)

        # Warm-up (HueTuber)
        _ = norm.normalize(target_bchw_cp[:, :, :64, :64])
        _sync()

        for B in batches:
            # Build batch sources on CPU & GPU
            src_np = rng.integers(0, 256, size=(B, H, W, 3), dtype=np.uint8)
            src_bchw_cp = cp.asarray(src_np).transpose(0, 3, 1, 2)  # (B,3,H,W)
            src_hwc_cp = cp.asarray(src_np)  # (B,H,W,3)

            # GPU timings (HueTuber) — repeated whole-batch calls
            gpu_times = [_run_gpu_once(norm, src_bchw_cp) for _ in range(repeat)]

            # GPU timings (cuCIM) — per-image loop, on GPU
            cucim_times = []
            if _HAS_CUCIM:
                for _ in range(repeat):
                    cucim_times.append(_run_cucim_once(src_hwc_cp, target_hwc_cp))
            else:
                cucim_times = [np.nan] * repeat

            # CPU baseline (skimage) — per-image HWC loop
            cpu_times = []
            for _ in range(repeat):
                t0 = time.perf_counter()
                for i in range(B):
                    _ = skexp.match_histograms(src_np[i], target_np, channel_axis=-1)
                t1 = time.perf_counter()
                cpu_times.append(t1 - t0)

            gpu_mean = float(np.nanmean(gpu_times))
            gpu_std = float(np.nanstd(gpu_times))
            cpu_mean = float(np.nanmean(cpu_times))
            cpu_std = float(np.nanstd(cpu_times))
            cucim_mean = float(np.nanmean(cucim_times))
            cucim_std = float(np.nanstd(cucim_times))

            row = {
                "H": H,
                "W": W,
                "B": B,
                "gpu_mean_s": gpu_mean,
                "gpu_std_s": gpu_std,
                "cpu_mean_s": cpu_mean,
                "cpu_std_s": cpu_std,
                "cucim_mean_s": cucim_mean,
                "cucim_std_s": cucim_std,
                "speedup_cpu_over_gpu": float(cpu_mean / max(gpu_mean, 1e-12)),
                "speedup_cpu_over_cucim": float(cpu_mean / max(cucim_mean, 1e-12))
                if _HAS_CUCIM
                else np.nan,
            }
            results.append(row)
            msg = (
                f"[H={H} B={B}] "
                f"GPU(HueTuber) {gpu_mean:.6f}s | "
                f"GPU(cuCIM) {cucim_mean:.6f}s | "
                f"CPU {cpu_mean:.6f}s | "
                f"speedup CPU/GPU-H {row['speedup_cpu_over_gpu']:.2f}x"
            )
            if _HAS_CUCIM:
                msg += f" | CPU/GPU-C {row['speedup_cpu_over_cucim']:.2f}x"
            print(msg)

    df = pd.DataFrame(results)
    out_dir = os.path.join(THIS_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "histogram_match_speed.csv")
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
    return df


if __name__ == "__main__":
    run_hist_match_speed()
