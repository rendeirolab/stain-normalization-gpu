# ruff: noqa: E402
# benchmark/visualize_histogram_speed.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import numpy as np

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "results")
CSV = os.path.join(RESULTS, "histogram_match_speed.csv")


def main():
    if not os.path.exists(CSV):
        raise FileNotFoundError(
            f"CSV not found: {CSV}\nRun benchmark_histogram_speed.py first."
        )

    df = pd.read_csv(CSV)

    needed = {"H", "W", "B", "gpu_mean_s", "cpu_mean_s"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    has_cucim = "cucim_mean_s" in df.columns

    df["size"] = df["H"].astype(int).astype(str) + "×" + df["W"].astype(int).astype(str)
    df["B"] = df["B"].astype(int)

    sns.set_theme(context="talk", style="whitegrid")

    # ---------- A) GROUPED TIME BARS with GPU-only "× faster" labels ----------
    sizes = df["size"].unique()
    hue_order = (
        ["CPU (skimage)", "GPU (HueTuber)", "GPU (cuCIM)"]
        if has_cucim
        else ["CPU (skimage)", "GPU (HueTuber)"]
    )

    for s in sizes:
        d = df[df["size"] == s].copy().sort_values("B")

        rows = []
        for _, r in d.iterrows():
            rows.append(
                {"B": r["B"], "Impl": "CPU (skimage)", "time_s": r["cpu_mean_s"]}
            )
            rows.append(
                {"B": r["B"], "Impl": "GPU (HueTuber)", "time_s": r["gpu_mean_s"]}
            )
            if has_cucim and np.isfinite(r.get("cucim_mean_s", np.nan)):
                rows.append(
                    {"B": r["B"], "Impl": "GPU (cuCIM)", "time_s": r["cucim_mean_s"]}
                )
        gd = pd.DataFrame(rows)

        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=gd, x="B", y="time_s", hue="Impl", hue_order=hue_order, ax=ax)

        ax.set_title(f"Histogram Matching Time by Batch — {s}", pad=12)
        ax.set_xlabel("Batch size (B)")
        ax.set_ylabel("Mean time (s)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
        ax.grid(False, axis="x")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(title=None, frameon=False)

        # annotate ONLY GPU bars with "× faster (vs CPU)"
        hue_to_container = {
            h: c for h, c in zip(hue_order, ax.containers[: len(hue_order)])
        }
        cpu_lookup = d.set_index("B")["cpu_mean_s"]

        def annotate_container(container):
            xcats = sorted(d["B"].unique())
            for patch, B in zip(container.patches, xcats):
                height = patch.get_height()
                if not (height > 0 and np.isfinite(height)):
                    continue
                cpu_t = float(cpu_lookup.loc[int(B)])
                speed = cpu_t / height
                ax.text(
                    patch.get_x() + patch.get_width() / 2,
                    height,
                    f"{speed:.1f}×",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

        if "GPU (HueTuber)" in hue_to_container:
            annotate_container(hue_to_container["GPU (HueTuber)"])
        if has_cucim and "GPU (cuCIM)" in hue_to_container:
            annotate_container(hue_to_container["GPU (cuCIM)"])

        fig.tight_layout()
        out = os.path.join(RESULTS, f"histogram_grouped_{s.replace('×', 'x')}.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print("Saved:", out)

    # ---------- B) SPEED-UP STYLE FIGURE ----------
    for s in sizes:
        d = df[df["size"] == s].copy().sort_values("B")
        data = {"B": d["B"], "HueTuber": d["cpu_mean_s"] / d["gpu_mean_s"]}
        if has_cucim:
            data["cuCIM"] = d["cpu_mean_s"] / d["cucim_mean_s"]

        spd = pd.DataFrame(data)
        melt = spd.melt(id_vars="B", var_name="Impl", value_name="speedup_x")

        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=melt, x="B", y="speedup_x", hue="Impl", ax=ax)
        ax.set_title(f"Histogram Matching — Speed Up vs CPU — {s}", pad=12)
        ax.set_xlabel("Batch size (B)")
        ax.set_ylabel("Speed Up (× vs CPU)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
        ax.grid(False, axis="x")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(title=None, frameon=False)

        for container in ax.containers[: len(melt["Impl"].unique())]:
            for patch in container.patches:
                h = patch.get_height()
                if h > 0 and np.isfinite(h):
                    ax.text(
                        patch.get_x() + patch.get_width() / 2,
                        h,
                        f"{h:.0f}×" if h >= 10 else f"{h:.1f}×",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                    )

        fig.tight_layout()
        out = os.path.join(RESULTS, f"histogram_speedup_bars_{s.replace('×', 'x')}.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print("Saved:", out)


if __name__ == "__main__":
    main()
