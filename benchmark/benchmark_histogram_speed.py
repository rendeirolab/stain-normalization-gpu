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

from huetuber.match_histogram import HistogramMatching


def _sync():
    cp.cuda.Stream.null.synchronize()


def _run_gpu_once(norm: HistogramMatching, src_bchw_cp: cp.ndarray) -> float:
    _sync()
    t0 = time.perf_counter()
    _ = norm.normalize(src_bchw_cp)
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

    for H in sizes:
        W = H

        # Build target (CPU HWC) + GPU BCHW
        target_np = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
        target_bchw_cp = cp.asarray(target_np).transpose(2, 0, 1)[None]  # (1,3,H,W)

        # Initialize GPU normalizer
        norm = HistogramMatching(channel_axis=1, dtype_out=cp.uint8)
        norm.fit(target_bchw_cp)

        # Warm-up
        _ = norm.normalize(target_bchw_cp[:, :, :64, :64])
        _sync()

        for B in batches:
            # Build batch sources on CPU & GPU
            src_np = rng.integers(0, 256, size=(B, H, W, 3), dtype=np.uint8)
            src_cp = cp.asarray(src_np).transpose(0, 3, 1, 2)  # (B,3,H,W)

            # GPU timings (HueTuber)
            gpu_times = [_run_gpu_once(norm, src_cp) for _ in range(repeat)]

            # CPU baseline (skimage) — per-image HWC loop
            cpu_times = []
            for _ in range(repeat):
                t0 = time.perf_counter()
                for i in range(B):
                    _ = skexp.match_histograms(src_np[i], target_np, channel_axis=-1)
                t1 = time.perf_counter()
                cpu_times.append(t1 - t0)

            row = {
                "H": H,
                "W": W,
                "B": B,
                "gpu_mean_s": float(np.mean(gpu_times)),
                "gpu_std_s": float(np.std(gpu_times)),
                "cpu_mean_s": float(np.mean(cpu_times)),
                "cpu_std_s": float(np.std(cpu_times)),
                "speedup_cpu_over_gpu": float(
                    np.mean(cpu_times) / max(np.mean(gpu_times), 1e-12)
                ),
            }
            results.append(row)
            print(
                f"[H={H} B={B}] GPU {row['gpu_mean_s']:.6f}s | "
                f"CPU {row['cpu_mean_s']:.6f}s | "
                f"speedup {row['speedup_cpu_over_gpu']:.2f}x"
            )

    df = pd.DataFrame(results)
    out_dir = os.path.join(THIS_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "histogram_match_speed.csv")
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
    return df


if __name__ == "__main__":
    run_hist_match_speed()
