# GPU-Accelerated Stain Normalization for Histology Image Processing

**[scverse x NVIDIA Hackathon](https://developer.nvidia.com/accelerate-omics-hackathon)**

## Team

- **Team Lead**: Yimin Zheng [@Mr-Milk](https://github.com/Mr-Milk)
- **Team Members**: Arkajyoti Sarkar [@arka2696](https://github.com/arka2696), Sunag Parasu [@sunagparasu](https://github.com/sunagparasu)

## Project Overview

Stain normalization is a critical preprocessing step in computational pathology that addresses color variations caused by differences in staining protocols, scanners, and tissue preparation techniques. These variations can significantly impact the performance of downstream analysis tasks.

Recent research has shown that many histology foundation models fail to adequately eliminate staining batch effects (see [this paper](https://arxiv.org/abs/2411.05489v1)), which negatively impacts multimodal integration and biomarker prediction accuracy.

This project delivers GPU-accelerated stain normalization algorithms using CuPy, providing substantial performance improvements on CUDA-enabled GPUs for large-scale histology image processing workflows.

## Implementation

During the hackathon, we successfully implemented and benchmarked three widely-used stain normalization methods:

- **Reinhard normalization**
- **Histogram matching**
- **Macenko normalization**

## Results

### Reinhard ✅

Our GPU-accelerated Reinhard implementation achieves **50-100x speedup** compared to the NumPy-based torchstain implementation while maintaining identical visual results.

**Visual Validation**

<img width=300 src="https://raw.githubusercontent.com/rendeirolab/stain-normalization-gpu/refs/heads/main/benchmark/results/reinhard_visualization.png">

**Performance Benchmark**

<img width=300 src="https://raw.githubusercontent.com/rendeirolab/stain-normalization-gpu/refs/heads/main/benchmark/results/reinhard_benchmark.png">

### Histogram matching ✅

Our histogram matching implementation not only matches scikit-image results but also outperforms existing GPU implementations, including RAPIDS cuCIM.

**Visual Validation**

<img width=300 src="https://raw.githubusercontent.com/rendeirolab/stain-normalization-gpu/refs/heads/main/benchmark/results/histogram_matching_visualization.png">

**Performance Benchmark**

<img width=300 src="https://raw.githubusercontent.com/rendeirolab/stain-normalization-gpu/refs/heads/main/benchmark/results/histogram_matching_benchmark.png">

### Macenko ⚠️

While our Macenko implementation demonstrates the feasibility of GPU-accelerated stain separation, the results do not yet fully replicate existing implementations. This remains an area for future development.

**Visual Comparison**

<img width=300 src="https://raw.githubusercontent.com/rendeirolab/stain-normalization-gpu/refs/heads/main/benchmark/results/macenko_visualization.png">

## Key Achievements

- **High-performance computing**: Achieved 50-100x speedup for Reinhard normalization compared to existing CPU implementations
- **Algorithm fidelity**: Successfully replicated visual results of established implementations for Reinhard and histogram matching methods
- **Competitive performance**: Outperformed existing GPU implementations, including RAPIDS cuCIM, for histogram matching

## Future Work

- Optimize Macenko stain separation algorithm to match reference implementation accuracy
- Extend support for additional stain normalization methods (e.g., Vahadane)
- Integrate with popular computational pathology frameworks

## Technical References

- **Reinhard Color Transfer**: Reinhard, E., et al. "Color transfer between images." *IEEE Computer Graphics and Applications* (2001). [PDF](https://home.cis.rit.edu/~cnspci/references/dip/color_transfer/reinhard2001.pdf)
- **Macenko Stain Separation**: Macenko, M., et al. "A method for normalizing histology slides for quantitative analysis." *ISBI* (2009). [PDF](http://wwwx.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf)
- **Vahadane Method**: Vahadane, A., et al. "Structure-preserving color normalization and sparse stain separation for histological images." *IEEE TMI* (2016). [IEEE](https://ieeexplore.ieee.org/document/7460968)
