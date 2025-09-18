# ruff: noqa: E402
# benchmark/benchmark_hist_match.py

import argparse
import os
import glob
import time
import sys

# --- make the package importable without installing (-e) ---
THIS_DIR = os.path.dirname(__file__)
PROJECT_SRC = os.path.abspath(os.path.join(THIS_DIR, "..", "src"))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

import cupy as cp
import numpy as np
from skimage import exposure as skexp  # for CPU baseline
from PIL import Image, ImageFile  # <-- use Pillow for I/O

# --- allow very large images (e.g., 16k x 16k) ---
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True  # optional: tolerate slightly broken files

from huetuber.match_histogram import HistogramMatching  # GPU-only version


def load_image_cpu(path: str) -> np.ndarray:
    """Read image as HWC uint8 RGB using Pillow."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        arr = np.asarray(im, dtype=np.uint8)  # HWC, uint8
    return arr


def save_image_cpu(path: str, arr_hwc_uint8: np.ndarray) -> None:
    """Write HWC uint8 RGB using Pillow."""
    Image.fromarray(arr_hwc_uint8, mode="RGB").save(path)


def hwc_np_to_bchw_cp(img_np):
    # (H,W,C) numpy -> cupy (1,C,H,W)
    arr = cp.asarray(img_np, dtype=cp.uint8)
    return arr.transpose(2, 0, 1)[None, ...]


def bchw_cp_to_hwc_np(img_bchw_cp):
    # (1,C,H,W) cupy -> numpy (H,W,C)
    out = img_bchw_cp[0].transpose(1, 2, 0)
    return cp.asnumpy(out)


def time_gpu(fn, *args, **kwargs):
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    cp.cuda.Stream.null.synchronize()
    t1 = time.perf_counter()
    return out, (t1 - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="Reference image path")
    ap.add_argument("--src_dir", required=True, help="Folder with source images")
    ap.add_argument("--out_dir", required=True, help="Output dir (will be created)")
    ap.add_argument(
        "--compare_skimage", action="store_true", help="Also run CPU skimage baseline"
    )
    ap.add_argument(
        "--channel_axis", type=int, default=1, help="1 for BCHW, -1 for BHWC"
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- load & move reference to GPU ---
    ref_np = load_image_cpu(args.ref)  # HWC uint8 on CPU
    ref_bchw_cp = hwc_np_to_bchw_cp(ref_np)  # (1,C,H,W) on GPU

    # --- init & warm-up ---
    norm = HistogramMatching(channel_axis=args.channel_axis, dtype_out=cp.uint8)
    norm.fit(ref_bchw_cp)
    _ = norm.normalize(ref_bchw_cp[:, :, :32, :32])  # small slice warm-up
    cp.cuda.Stream.null.synchronize()

    # --- collect images ---
    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(args.src_dir, e)))
    paths.sort()

    gpu_times = []
    cpu_times = []

    for pth in paths:
        name = os.path.basename(pth)
        print(f"[GPU] Processing {name} ...")

        # load source on CPU then move to GPU
        src_np = load_image_cpu(pth)
        src_bchw_cp = hwc_np_to_bchw_cp(src_np)

        # --- GPU timing ---
        out_bchw_cp, t_gpu = time_gpu(norm.normalize, src_bchw_cp)
        gpu_times.append(t_gpu)

        # save GPU output
        out_np = bchw_cp_to_hwc_np(out_bchw_cp)  # back to CPU for saving
        save_image_cpu(os.path.join(args.out_dir, f"matched_gpu_{name}"), out_np)

        # --- optional CPU baseline (skimage) ---
        if args.compare_skimage:
            print(f"[CPU] skimage match_histograms {name} ...")
            t0 = time.perf_counter()
            out_cpu = skexp.match_histograms(src_np, ref_np, channel_axis=-1)
            t1 = time.perf_counter()
            cpu_times.append(t1 - t0)
            save_image_cpu(
                os.path.join(args.out_dir, f"matched_cpu_{name}"),
                out_cpu.astype(np.uint8),
            )

        print(f" -> GPU time: {t_gpu:.6f}s")

    # --- summary ---
    if gpu_times:
        print(
            f"\nGPU:  mean={np.mean(gpu_times):.6f}s  median={np.median(gpu_times):.6f}s  n={len(gpu_times)}"
        )
    if cpu_times:
        print(
            f"CPU:  mean={np.mean(cpu_times):.6f}s  median={np.median(cpu_times):.6f}s  n={len(cpu_times)}"
        )

    # write a CSV summary
    with open(os.path.join(args.out_dir, "benchmark_hist_match.csv"), "w") as f:
        f.write("backend,filename,seconds\n")
        for pth, t in zip(paths, gpu_times):
            f.write(f"gpu,{os.path.basename(pth)},{t:.6f}\n")
        for pth, t in zip(paths, cpu_times):
            f.write(f"cpu,{os.path.basename(pth)},{t:.6f}\n")


if __name__ == "__main__":
    main()
