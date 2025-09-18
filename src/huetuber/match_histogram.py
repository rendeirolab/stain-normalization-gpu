import cupy as cp
from .base import BaseStainNormalizer


class HistogramMatching(BaseStainNormalizer):
    def __init__(self, channel_axis: int = 1, dtype_out=None):
        self.channel_axis = channel_axis
        self.dtype_out = dtype_out
        self._ref_vals = None  # list[cp.ndarray] float32
        self._ref_cdf = None  # list[cp.ndarray] float32

        if channel_axis in (1, -3):
            self._split = lambda a: [a[:, c, ...] for c in range(a.shape[1])]
            self._stack = lambda chs: cp.stack(chs, axis=1)
        else:
            self._split = lambda a: [a[..., c] for c in range(a.shape[-1])]
            self._stack = lambda chs: cp.stack(chs, axis=-1)

    # ---------- internals ----------

    @staticmethod
    def _cdf_from_counts(counts: cp.ndarray) -> cp.ndarray:
        q = cp.cumsum(counts, dtype=cp.float32)
        total = q[-1]
        return cp.where(total > 0, q / total, q)

    @staticmethod
    def _unique_counts(flat: cp.ndarray):
        if flat.dtype == cp.uint8:
            counts = cp.bincount(flat, minlength=256)
            idx = counts.nonzero()[0]
            vals = idx.astype(cp.float32)  # interp expects float
            cnts = counts[idx]
            return vals, cnts
        vals, cnts = cp.unique(flat, return_counts=True)
        return vals.astype(cp.float32), cnts

    @staticmethod
    def _map_uint8_with_lut(
        img_hw_uint8: cp.ndarray, ref_vals: cp.ndarray, ref_cdf: cp.ndarray, out_dtype
    ) -> cp.ndarray:
        """
        Build a 256-entry LUT from source CDF and apply in one gather.
        Returns array in out_dtype (uint8 fast-path when requested).
        """
        img_hw_uint8 = cp.ascontiguousarray(img_hw_uint8)
        flat = img_hw_uint8.reshape(-1)

        counts = cp.bincount(flat, minlength=256)  # (256,)
        cdf = HistogramMatching._cdf_from_counts(counts)  # (256,) float32

        if out_dtype == cp.uint8:
            # build LUT directly as uint8 to avoid float32 full-image temp
            lut = cp.interp(cdf, ref_cdf, ref_vals)
            # round->clip->cast all on device
            lut = cp.rint(lut).astype(cp.int32)
            lut = cp.clip(lut, 0, 255).astype(cp.uint8, copy=False)
            return lut[img_hw_uint8]  # (H,W) uint8
        else:
            lut = cp.interp(cdf, ref_cdf, ref_vals).astype(cp.float32)
            return lut[img_hw_uint8]  # (H,W) float32

    # ---------- API ----------

    def fit(self, target: cp.ndarray) -> None:
        t = cp.asarray(target)
        C = t.shape[1] if self.channel_axis in (1, -3) else t.shape[-1]
        if C != 3:
            raise ValueError(f"Expected 3-channel RGB, got C={C}")

        self._ref_vals, self._ref_cdf = [], []
        for ch in self._split(t):
            flat = ch.reshape(-1)
            vals, cnts = self._unique_counts(flat)
            cdf = self._cdf_from_counts(cnts)
            self._ref_vals.append(vals)
            self._ref_cdf.append(cdf)

    def normalize(self, source: cp.ndarray) -> cp.ndarray:
        if self._ref_vals is None or self._ref_cdf is None:
            raise RuntimeError("Call fit(target) before normalize(source).")

        s = cp.asarray(source)

        # determine final output dtype up front
        out_dtype = self.dtype_out if self.dtype_out is not None else s.dtype

        # preallocate output in FINAL dtype to avoid huge float32 temporaries
        if self.channel_axis in (1, -3):
            B, C, H, W = s.shape
            if C != 3:
                raise ValueError(f"Expected 3-channel RGB, got C={C}")
            out = cp.empty((B, C, H, W), dtype=out_dtype)
        else:
            B, H, W, C = s.shape
            if C != 3:
                raise ValueError(f"Expected 3-channel RGB, got C={C}")
            out = cp.empty((B, H, W, C), dtype=out_dtype)

        # per channel
        for c_idx, ch in enumerate(self._split(s)):
            ref_vals = self._ref_vals[c_idx]
            ref_cdf = self._ref_cdf[c_idx]

            out_ch = (
                out[:, c_idx, ...] if self.channel_axis in (1, -3) else out[..., c_idx]
            )

            for b in range(ch.shape[0]):
                img = cp.ascontiguousarray(ch[b])  # (H,W)

                if img.dtype == cp.uint8:
                    out_ch[b] = self._map_uint8_with_lut(
                        img, ref_vals, ref_cdf, out_dtype
                    )
                else:
                    # Non-uint8 exact path
                    flat = img.reshape(-1)
                    try:
                        src_vals, inv, src_cnts = cp.unique(
                            flat, return_inverse=True, return_counts=True
                        )
                        src_vals = src_vals.astype(cp.float32)
                        src_cdf = self._cdf_from_counts(src_cnts)
                        mapped_vals = cp.interp(src_cdf, ref_cdf, ref_vals)
                        mapped = mapped_vals[inv].reshape(img.shape)
                    except TypeError:
                        src_vals, src_cnts = cp.unique(flat, return_counts=True)
                        src_vals = src_vals.astype(cp.float32)
                        src_cdf = self._cdf_from_counts(src_cnts)
                        mapped_vals = cp.interp(src_cdf, ref_cdf, ref_vals)
                        inv = cp.searchsorted(src_vals, flat)
                        mapped = mapped_vals[inv].reshape(img.shape)

                    if out_dtype == cp.uint8:
                        out_ch[b] = cp.clip(cp.rint(mapped), 0, 255).astype(
                            cp.uint8, copy=False
                        )
                    else:
                        out_ch[b] = mapped.astype(out_dtype, copy=False)

        return out
