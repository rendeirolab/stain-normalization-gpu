from pathlib import Path

from huetuber import HistogramMatching
from time import time

import pandas as pd

import cupy as cp
import numpy as np

# Import histogram matching implementations
from skimage.exposure import match_histograms as skimage_match_histograms

try:
    from cucim.skimage.exposure import match_histograms as cucim_match_histograms

    CUCIM_AVAILABLE = True
except ImportError:
    CUCIM_AVAILABLE = False
    print("Warning: cucim not available, will skip cucim benchmarks")


def benchmark_histogram_matching_speed(
    repeat=5, img_size=512, num_imgs_list=[5, 10, 50, 100]
):
    """Benchmark histogram matching across different numbers of images."""

    all_results = []

    for num_imgs in num_imgs_list:
        print(f"\n=== Benchmarking with {num_imgs} images ===")

        records = {
            "huetuber": [],
            "skimage": [],
        }

        if CUCIM_AVAILABLE:
            records["cucim"] = []

        # === Test huetuber implementation ===
        print("Running huetuber Histogram Matching...")
        target_img = cp.random.randint(0, 256, (1, 3, img_size, img_size)).astype(
            cp.uint8
        )
        source_img = cp.random.randint(
            0, 256, (num_imgs, 3, img_size, img_size)
        ).astype(cp.uint8)
        warmup_source = cp.random.randint(0, 256, (1, 3, img_size, img_size)).astype(
            cp.uint8
        )

        histogram_matcher_huetuber = HistogramMatching(channel_axis=1)
        histogram_matcher_huetuber.fit(target_img)
        histogram_matcher_huetuber.normalize(warmup_source)  # Warm up

        for _ in range(repeat):
            start = time()
            histogram_matcher_huetuber.normalize(source_img)
            end = time()
            records["huetuber"].append(end - start)

        # === Test skimage implementation ===
        print("Running scikit-image histogram matching...")
        target_img_np = np.random.randint(
            0, 256, (img_size, img_size, 3), dtype=np.uint8
        )
        source_img_np = np.random.randint(
            0, 256, (num_imgs, img_size, img_size, 3), dtype=np.uint8
        )
        warmup_source_np = np.random.randint(
            0, 256, (img_size, img_size, 3), dtype=np.uint8
        )

        # Warm up
        skimage_match_histograms(warmup_source_np, target_img_np, channel_axis=-1)

        for _ in range(repeat):
            start = time()
            for i in range(num_imgs):
                skimage_match_histograms(
                    source_img_np[i], target_img_np, channel_axis=-1
                )
            end = time()
            records["skimage"].append(end - start)

        # === Test cucim implementation (if available) ===
        if CUCIM_AVAILABLE:
            print("Running cucim histogram matching...")
            target_img_cp = cp.random.randint(
                0, 256, (img_size, img_size, 3), dtype=cp.uint8
            )
            source_img_cp = cp.random.randint(
                0, 256, (num_imgs, img_size, img_size, 3), dtype=cp.uint8
            )
            warmup_source_cp = cp.random.randint(
                0, 256, (img_size, img_size, 3), dtype=cp.uint8
            )

            # Warm up
            cucim_match_histograms(warmup_source_cp, target_img_cp, channel_axis=-1)

            for _ in range(repeat):
                start = time()
                for i in range(num_imgs):
                    cucim_match_histograms(
                        source_img_cp[i], target_img_cp, channel_axis=-1
                    )
                end = time()
                records["cucim"].append(end - start)

        # Calculate means for this number of images
        huetuber_mean = np.array(records["huetuber"]).mean()
        skimage_mean = np.array(records["skimage"]).mean()

        result_data = {
            "num_images": num_imgs,
            "huetuber_mean": huetuber_mean,
            "skimage_mean": skimage_mean,
            "huetuber_std": np.array(records["huetuber"]).std(),
            "skimage_std": np.array(records["skimage"]).std(),
        }

        print(
            f"  huetuber: {huetuber_mean:.4f}s ± {np.array(records['huetuber']).std():.4f}s"
        )
        print(
            f"  scikit-image: {skimage_mean:.4f}s ± {np.array(records['skimage']).std():.4f}s"
        )
        print(
            f"  huetuber is {skimage_mean / huetuber_mean:.2f}x faster than scikit-image"
        )

        if CUCIM_AVAILABLE:
            cucim_mean = np.array(records["cucim"]).mean()
            result_data.update(
                {
                    "cucim_mean": cucim_mean,
                    "cucim_std": np.array(records["cucim"]).std(),
                }
            )
            print(
                f"  cucim: {cucim_mean:.4f}s ± {np.array(records['cucim']).std():.4f}s"
            )
            print(f"  huetuber is {cucim_mean / huetuber_mean:.2f}x faster than cucim")
            print(
                f"  cucim is {skimage_mean / cucim_mean:.2f}x faster than scikit-image"
            )

        all_results.append(result_data)

    # Save results to CSV
    save_path = Path(__file__).parent / "results"
    save_path.mkdir(exist_ok=True)

    df = pd.DataFrame(all_results)
    df.to_csv(save_path / "histogram_matching_benchmark_results.csv", index=False)
    print(
        f"\nResults saved to: {save_path / 'histogram_matching_benchmark_results.csv'}"
    )


def benchmark_visualize():
    """Create visualization of benchmark results."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    try:
        import mpl_fontkit as fk

        fk.install("Lato")
    except ImportError:
        print("Warning: mpl_fontkit not available, using default fonts")

    save_path = Path(__file__).parent / "results"
    csv_path = save_path / "histogram_matching_benchmark_results.csv"

    if not csv_path.exists():
        print("Error: benchmark results file not found. Run the benchmark first.")
        return

    bench = pd.read_csv(csv_path)

    # Create speedup columns (relative to scikit-image)
    bench["scikit-image"] = 1  # baseline
    bench["huetuber"] = bench["skimage_mean"] / bench["huetuber_mean"]

    # Check if cucim results are available
    value_vars = ["scikit-image", "huetuber"]
    palette = ["#E9D5DA", "#76B900"]

    if "cucim_mean" in bench.columns:
        bench["cucim"] = bench["skimage_mean"] / bench["cucim_mean"]
        value_vars.append("cucim")
        palette.append("#827397")

    data = bench.melt(
        id_vars="num_images", value_vars=value_vars, var_name="Implementation"
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=data,
        x="num_images",
        ax=ax,
        y="value",
        hue="Implementation",
        palette=palette,
        zorder=10,
    )

    ax.tick_params(labelleft=True)
    ax.set(xlabel="Number of images", ylabel="Speed Up (relative to scikit-image)")
    ax.grid(axis="y", alpha=0.3)
    sns.despine()

    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(
            container,
            fontsize=8,
            fmt=lambda x: f"{x:.1f}x" if 1 < x < 2 else f"{int(x)}x",
        )

    plt.title(
        "Histogram Matching Performance Comparison", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    output_path = save_path / "histogram_matching_benchmark.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Visualization saved to: {output_path}")

    # Show the plot
    plt.show()


if __name__ == "__main__":
    print("Starting Histogram Matching Benchmark...")
    print(f"CuCIM available: {CUCIM_AVAILABLE}")

    benchmark_histogram_matching_speed(
        img_size=512, num_imgs_list=[5, 10, 50, 100], repeat=5
    )
    benchmark_visualize()
