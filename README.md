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
    - [ ] slideflow
- [ ] Ship into PyPI Package

References implementation:
- torchstain, https://github.com/EIDOSLAB/torchstain
- torchvahadane, https://github.com/cwlkr/torchvahadane/tree/main
- slideflow, https://github.com/slideflow/slideflow/tree/master/slideflow/norm
- cuCIM (for macenko implementation), https://github.com/rapidsai/cucim/blob/branch-25.10/python/cucim/src/cucim/core/operations/color/stain_normalizer.py

Papers:
- Reinhard: https://home.cis.rit.edu/~cnspci/references/dip/color_transfer/reinhard2001.pdf
- Macenko: http://wwwx.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf
- Vahadane: https://ieeexplore.ieee.org/document/7460968


Random stuff.