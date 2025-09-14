from pathlib import Path

from huetuber import ReinhardNormalizer
from torchstain.numpy.normalizers.reinhard import NumpyReinhardNormalizer
from torchstain.torch.normalizers.reinhard import TorchReinhardNormalizer
from time import time

import pandas as pd

import cupy as cp
import numpy as np
import torch

def benchmark_reinhard_speed(repeat=5, img_size=512, num_imgs_list=[5, 10, 50, 100]):
    """Benchmark Reinhard normalization across different numbers of images."""
    
    # Initialize normalizers
    reinhard_torchstain_numpy = NumpyReinhardNormalizer()
    reinhard_torchstain_torch = TorchReinhardNormalizer()
    
    all_results = []
    
    for num_imgs in num_imgs_list:
        print(f"\n=== Benchmarking with {num_imgs} images ===")
        
        records = {
            "huetuber": [],
            "torchstain_numpy": [],
            "torchstain_torch": [],
        }
        
        # === Test huetuber implementation ===
        print("Running huetuber Reinhard...")
        target_img = cp.random.randint(0, 256, (1, 3, img_size, img_size)).astype(cp.uint8)
        source_img = cp.random.randint(0, 256, (num_imgs, 3, img_size, img_size)).astype(cp.uint8)
        warmup_source = cp.random.randint(0, 256, (1, 3, img_size, img_size)).astype(cp.uint8)
        
        reinhard_huetuber = ReinhardNormalizer(channel_axis=1)
        reinhard_huetuber.fit(target_img)
        reinhard_huetuber(warmup_source)  # Warm up
        
        for _ in range(repeat):
            start = time()
            reinhard_huetuber(source_img)
            end = time()
            records["huetuber"].append(end - start)
        
        # === Test torchstain numpy implementation ===
        print("Running torchstain numpy Reinhard...")
        target_img_np = np.random.randint(0, 256, (img_size, img_size, 3), dtype=np.uint8)
        warmup_source_np = np.random.randint(0, 256, (img_size, img_size, 3), dtype=np.uint8)
        source_img_np = np.random.randint(0, 256, (num_imgs, img_size, img_size, 3), dtype=np.uint8)
        
        reinhard_torchstain_numpy.fit(target_img_np)
        reinhard_torchstain_numpy.normalize(warmup_source_np)  # Warm up
        
        for _ in range(repeat):
            start = time()
            reinhard_torchstain_numpy.normalize(source_img_np)
            end = time()
            records["torchstain_numpy"].append(end - start)
        
        # === Test torchstain torch implementation ===
        print("Running torchstain torch Reinhard...")
        target_img_torch = torch.randint(0, 256, (1, 3, img_size, img_size), dtype=torch.uint8)
        warmup_source_torch = torch.randint(0, 256, (1, 3, img_size, img_size), dtype=torch.uint8)
        source_img_torch = torch.randint(0, 256, (num_imgs, 3, img_size, img_size), dtype=torch.uint8)
        
        reinhard_torchstain_torch.fit(target_img_torch)
        reinhard_torchstain_torch.normalize(warmup_source_torch)  # Warm up
        
        for _ in range(repeat):
            start = time()
            for img in source_img_torch:
                reinhard_torchstain_torch.normalize(img)
            end = time()
            records["torchstain_torch"].append(end - start)
        
        # Calculate means for this number of images
        huetuber_mean = np.array(records["huetuber"]).mean()
        numpy_mean = np.array(records["torchstain_numpy"]).mean()
        torch_mean = np.array(records["torchstain_torch"]).mean()
        
        # Store results
        all_results.append({
            "num_images": num_imgs,
            "huetuber_mean": huetuber_mean,
            "torchstain_numpy_mean": numpy_mean,
            "torchstain_torch_mean": torch_mean,
            "huetuber_std": np.array(records["huetuber"]).std(),
            "torchstain_numpy_std": np.array(records["torchstain_numpy"]).std(),
            "torchstain_torch_std": np.array(records["torchstain_torch"]).std(),
        })
        
        print(f"  huetuber: {huetuber_mean:.4f}s ± {np.array(records['huetuber']).std():.4f}s")
        print(f"  torchstain numpy: {numpy_mean:.4f}s ± {np.array(records['torchstain_numpy']).std():.4f}s")
        print(f"  torchstain torch: {torch_mean:.4f}s ± {np.array(records['torchstain_torch']).std():.4f}s")
        print(f"  huetuber is {numpy_mean / huetuber_mean:.2f}x faster than numpy")
        print(f"  huetuber is {torch_mean / huetuber_mean:.2f}x faster than torch")
    
    # Save results to CSV
    save_path = Path(__file__).parent / "results"
    save_path.mkdir(exist_ok=True)
    
    df = pd.DataFrame(all_results)
    df.to_csv(save_path / "reinhard_benchmark_results.csv", index=False)


def benchmark_visualize():
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    import mpl_fontkit as fk
    fk.install("Lato")

    save_path = Path(__file__).parent / "results"
    bench = pd.read_csv(f"{save_path}/reinhard_benchmark_results.csv")
    bench['torchstain (numpy)'] = 1
    bench['torchstain (torch)'] = bench['torchstain_numpy_mean'] / bench['torchstain_torch_mean']
    bench['huetuber'] = bench['torchstain_numpy_mean'] / bench['huetuber_mean']

    data = bench.melt(id_vars='num_images', 
           value_vars=['torchstain (numpy)', 'torchstain (torch)', 'huetuber'], 
           var_name="Implementation")

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(data=data, x="num_images", ax=ax,
                y="value", hue="Implementation",
                palette=['#E9D5DA', '#827397', '#76B900'],
                zorder=10,
            )
    ax.tick_params(labelleft=True)
    ax.set(xlabel="Number of images", ylabel="Speed Up")
    ax.grid(axis='y')
    sns.despine()
    for container in ax.containers:
        ax.bar_label(container, 
                    fontsize=8,
                    fmt=lambda x: f"{x:.1f}x" if 1 < x < 2 else f"{int(x)}x")
    
    fig.savefig(f"{save_path}/reinhard_benchmark.png", dpi=200, bbox_inches='tight')


if __name__ == "__main__":
    benchmark_reinhard_speed(img_size=512, num_imgs_list=[5, 10, 50, 100], repeat=5)
    benchmark_visualize()
