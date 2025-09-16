# ruff: noqa: E402
# benchmark/visualize_histogram_speed.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "results")
CSV = os.path.join(RESULTS, "histogram_match_speed.csv")


def main():
    if not os.path.exists(CSV):
        raise FileNotFoundError(
            f"CSV not found: {CSV}\nRun benchmark_histogram_speed.py first."
        )

    df = pd.read_csv(CSV)

    # Basic sanity
    needed = {"H", "W", "B", "gpu_mean_s", "cpu_mean_s", "speedup_cpu_over_gpu"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    # Labels
    df["size"] = df["H"].astype(int).astype(str) + "×" + df["W"].astype(int).astype(str)
    df["B"] = df["B"].astype(int)

    sns.set_context("talk")
    sns.set_style("whitegrid")

    # 1) Speedup bars per size
    sizes = df["size"].unique()
    for s in sizes:
        d = df[df["size"] == s].copy().sort_values("B")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=d, x="B", y="speedup_cpu_over_gpu", ax=ax)
        ax.set_title(f"Histogram Matching Speedup (CPU/Skimage ÷ GPU/HueTuber) — {s}")
        ax.set_xlabel("Batch size (B)")
        ax.set_ylabel("Speedup: CPU time / GPU time  (×)")
        for c in ax.containers:
            ax.bar_label(c, fmt="%.2fx", fontsize=10)
        fig.tight_layout()
        out = os.path.join(RESULTS, f"histogram_speedup_{s.replace('×', 'x')}.png")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print("Saved:", out)

    # 2) GPU time heatmap
    pivot_gpu = df.pivot_table(
        index="size", columns="B", values="gpu_mean_s", aggfunc="mean"
    )
    fig2, ax2 = plt.subplots(figsize=(6, 4.8))
    sns.heatmap(pivot_gpu, annot=True, fmt=".4f", cmap="viridis", ax=ax2)
    ax2.set_title("GPU mean time (s) — Histogram Matching")
    ax2.set_xlabel("Batch size (B)")
    ax2.set_ylabel("Size")
    fig2.tight_layout()
    out2 = os.path.join(RESULTS, "histogram_gpu_time_heatmap.png")
    fig2.savefig(out2, dpi=200, bbox_inches="tight")
    print("Saved:", out2)

    # 3) Speedup heatmap
    pivot_speed = df.pivot_table(
        index="size", columns="B", values="speedup_cpu_over_gpu", aggfunc="mean"
    )
    fig3, ax3 = plt.subplots(figsize=(6, 4.8))
    sns.heatmap(pivot_speed, annot=True, fmt=".2f", cmap="mako", ax=ax3)
    ax3.set_title("Speedup (CPU / GPU) — higher is better")
    ax3.set_xlabel("Batch size (B)")
    ax3.set_ylabel("Size")
    fig3.tight_layout()
    out3 = os.path.join(RESULTS, "histogram_speedup_heatmap.png")
    fig3.savefig(out3, dpi=200, bbox_inches="tight")
    print("Saved:", out3)


if __name__ == "__main__":
    main()
