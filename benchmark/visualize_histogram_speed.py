# ruff: noqa: E402
# benchmark/visualize_histogram_speed.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator, FuncFormatter

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "results")
CSV = os.path.join(RESULTS, "histogram_match_speed.csv")


def _mul_fmt(v, _pos):
    """Format 57.86 -> '57.9×' for y-axes."""
    return f"{v:.1f}×"


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

    # Global style
    sns.set_theme(context="talk", style="whitegrid")

    # ---------- 1) Speedup bars per size ----------
    sizes = df["size"].unique()
    for s in sizes:
        d = df[df["size"] == s].copy().sort_values("B")

        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        sns.barplot(data=d, x="B", y="speedup_cpu_over_gpu", ax=ax)

        # Titles and labels
        ax.set_title(
            f"Histogram Matching Speedup (CPU/Skimage ÷ GPU/HueTuber) — {s}", pad=12
        )
        ax.set_xlabel("Batch size (B)")
        ax.set_ylabel("Speedup: CPU time / GPU time")

        # Axes polish
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # integer ticks for B
        ymax = max(1.0, d["speedup_cpu_over_gpu"].max())
        ax.set_ylim(0, ymax * 1.10)  # start at 0 with headroom
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_formatter(FuncFormatter(_mul_fmt))
        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
        ax.grid(False, axis="x")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="both", labelsize=11)

        # Value labels on bars
        for c in ax.containers:
            ax.bar_label(
                c, labels=[f"{v:.2f}×" for v in c.datavalues], fontsize=10, padding=2
            )

        fig.tight_layout()
        out = os.path.join(RESULTS, f"histogram_speedup_{s.replace('×', 'x')}.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print("Saved:", out)

    # ---------- 2) GPU time heatmap ----------
    pivot_gpu = (
        df.pivot_table(
            index="size", columns="B", values="gpu_mean_s", aggfunc="mean"
        ).sort_index(axis=1)  # ensure B increases left->right
    )

    fig2, ax2 = plt.subplots(figsize=(7.5, 5.2))
    sns.heatmap(
        pivot_gpu,
        annot=True,
        fmt=".4f",
        cmap="viridis",
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "GPU mean time (s)"},
        ax=ax2,
    )
    ax2.set_title("GPU mean time — Histogram Matching", pad=10)
    ax2.set_xlabel("Batch size (B)")
    ax2.set_ylabel("Size")
    ax2.tick_params(axis="x", rotation=0)
    ax2.tick_params(axis="y", rotation=0)
    fig2.tight_layout()
    out2 = os.path.join(RESULTS, "histogram_gpu_time_heatmap.png")
    fig2.savefig(out2, dpi=300, bbox_inches="tight")
    print("Saved:", out2)

    # ---------- 3) Speedup heatmap ----------
    pivot_speed = df.pivot_table(
        index="size", columns="B", values="speedup_cpu_over_gpu", aggfunc="mean"
    ).sort_index(axis=1)

    fig3, ax3 = plt.subplots(figsize=(7.5, 5.2))
    sns.heatmap(
        pivot_speed,
        annot=True,
        fmt=".2f",
        cmap="mako",
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "CPU/Skimage ÷ GPU/HueTuber (×)"},
        ax=ax3,
    )
    ax3.set_title("Speedup (higher is better)", pad=10)
    ax3.set_xlabel("Batch size (B)")
    ax3.set_ylabel("Size")
    ax3.tick_params(axis="x", rotation=0)
    ax3.tick_params(axis="y", rotation=0)
    fig3.tight_layout()
    out3 = os.path.join(RESULTS, "histogram_speedup_heatmap.png")
    fig3.savefig(out3, dpi=300, bbox_inches="tight")
    print("Saved:", out3)


if __name__ == "__main__":
    main()
