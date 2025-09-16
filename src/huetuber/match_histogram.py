"""
GPU-only, CuPy-accelerated histogram matching normalizer.

- fit(target):  target is (B, C, H, W) or (B, H, W, C)
- normalize(source): same layout; returns normalized image(s)

Per-channel CDF mapping on the GPU. Assumes CuPy is available.
"""

import cupy as cp
from .base import BaseStainNormalizer


class HistogramMatching(BaseStainNormalizer):
    def __init__(self, channel_axis: int = 1, dtype_out=None):
        """
        Parameters
        ----------
        channel_axis : int
            1 for (B, C, H, W) like the rest of the repo, or -1 for (B, H, W, C).
        dtype_out : cupy dtype or None
            If None, preserve input dtype; else cast (e.g. cp.uint8).
        """
        self.channel_axis = channel_axis
        self.dtype_out = dtype_out

        # reference (template) per-channel stats (on GPU)
        self._ref_vals = None  # list[cp.ndarray] (float32, sorted unique intensities)
        self._ref_cdf = (
            None  # list[cp.ndarray] (float32, CDF in [0,1], same len as _ref_vals)
        )

        # channel helpers
        if channel_axis in (1, -3):
            # (B, C, H, W)
            self._split = lambda a: [a[:, c, ...] for c in range(a.shape[1])]
            self._stack = lambda chs: cp.stack(chs, axis=1)
        else:
            # (B, H, W, C)
            self._split = lambda a: [a[..., c] for c in range(a.shape[-1])]
            self._stack = lambda chs: cp.stack(chs, axis=-1)

    # -------- internals (GPU only) --------

    @staticmethod
    def _cdf_from_counts(counts: cp.ndarray) -> cp.ndarray:
        """Return CDF in [0,1] as float32 from a counts vector."""
        q = cp.cumsum(counts, dtype=cp.float32)
        total = q[-1]
        # avoid div-by-zero; if empty, keep zeros
        q = cp.where(total > 0, q / total, q)
        return q

    @staticmethod
    def _unique_counts(flat: cp.ndarray):
        """
        Return (values, counts) with values sorted ascending.
        Fast path for uint8 via bincount.
        """
        if flat.dtype == cp.uint8:
            counts = cp.bincount(flat, minlength=256)
            idx = counts.nonzero()[0]
            vals = idx.astype(cp.float32)  # interp expects float
            cnts = counts[idx]
            return vals, cnts
        # generic (float / wider ints)
        vals, cnts = cp.unique(flat, return_counts=True)
        return vals.astype(cp.float32), cnts

    # -------- BaseStainNormalizer API --------

    def fit(self, target: cp.ndarray) -> None:
        """
        Compute per-channel reference CDFs from the target batch.

        target: (B, C, H, W) or (B, H, W, C), kept on GPU.
        """
        t = cp.asarray(target)  # ensure on GPU

        C = t.shape[1] if self.channel_axis in (1, -3) else t.shape[-1]
        if C != 3:
            raise ValueError(f"Expected 3-channel RGB, got C={C}")

        self._ref_vals, self._ref_cdf = [], []
        for ch in self._split(t):
            # flatten across batch + spatial
            flat = ch.reshape(-1)
            vals, cnts = self._unique_counts(flat)
            cdf = self._cdf_from_counts(cnts)  # float32
            self._ref_vals.append(vals)  # float32
            self._ref_cdf.append(cdf)  # float32

    def normalize(self, source: cp.ndarray) -> cp.ndarray:
        """
        Match source batch to the fitted reference CDFs (GPU).

        source: (B, C, H, W) or (B, H, W, C), on GPU.
        """
        if self._ref_vals is None or self._ref_cdf is None:
            raise RuntimeError("Call fit(target) before normalize(source).")

        s = cp.asarray(source)  # ensure on GPU

        # parse shape
        if self.channel_axis in (1, -3):
            B, C, H, W = s.shape
        else:
            B, H, W, C = s.shape

        if C != 3:
            raise ValueError(f"Expected 3-channel RGB, got C={C}")

        out_chs = []
        for c_idx, ch in enumerate(self._split(s)):
            ref_vals = self._ref_vals[c_idx]
            ref_cdf = self._ref_cdf[c_idx]

            # process per image to limit peak mem (GPU kernels still vectorize inside)
            out_imgs = []
            for b in range(ch.shape[0]):
                img = ch[b]  # (H, W)
                flat = img.reshape(-1)

                src_vals, src_cnts = self._unique_counts(flat)
                src_cdf = self._cdf_from_counts(src_cnts)

                # quantile -> reference intensity
                mapped_vals = cp.interp(src_cdf, ref_cdf, ref_vals)  # float32

                # map pixels by inverse indices
                _, inv = cp.unique(flat, return_inverse=True)
                out_b = mapped_vals[inv].reshape(img.shape).astype(cp.float32)
                out_imgs.append(out_b)

            out_chs.append(cp.stack(out_imgs, axis=0))

        out = self._stack(out_chs)

        # cast / clamp once at the end
        if self.dtype_out is not None:
            if self.dtype_out == cp.uint8:
                out = cp.clip(out, 0, 255).astype(cp.uint8)
            else:
                out = out.astype(self.dtype_out)
        else:
            out = out.astype(s.dtype, copy=False)
        return out
