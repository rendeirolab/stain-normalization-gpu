# GPU-accelerated stain normalization for histology image processing

Stain normalization is a critical preprocessing step in computational pathology. Variations in staining protocols, scanners, and tissue preparation can lead to large color differences in histology slides. Many histology foundation model doesn't really eliminate the staining batch effect, see https://arxiv.org/abs/2411.05489v1, which in turn affect downstream tasks like multimodality integration and biomarker prediction.

This project implements GPU-accelerated stain normalization using CuPy
for massive speedups on CUDA-enabled GPUs, while falling back to NumPy on CPUs for maximum compatibility.

Roadmap:

- [ ] Normalization methods:
    - [ ] Reinhard
    - [ ] Macenko
    - [ ] Vahadane
- [ ] Benchmark against existing implementation:
    - [ ] torchstain
    - [ ] torchvahadane
- [ ] Ship into PyPI Package