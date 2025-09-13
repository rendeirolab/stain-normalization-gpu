"""CuPy implementation of Reinhard stain normalization."""
from abc import ABC
import cupy as cp
from cucim.skimage.color import rgb2lab, lab2rgb

from .base import BaseStainNormalizer


class _ReinhardBase(BaseStainNormalizer, ABC):

    def __init__(self, channel_axis=1):
        self.channel_axis = channel_axis
        self.target_means = None
        self.target_stds = None
        if channel_axis == 1:
            self.axis = (2, 3)
            self.split_channels = lambda arr: (arr[:, 0, ...], arr[:, 1, ...], arr[:, 2, ...])
            self.stack_channels = lambda c: cp.stack(c, axis=1)
        else:
            self.axis = (1, 2)
            self.split_channels = lambda arr: (arr[..., 0], arr[..., 1], arr[..., 2])
            self.stack_channels = lambda c: cp.stack(c, axis=-1)

    def fit(self, target: cp.ndarray):
        target = target.astype(cp.float32) / 255.0
        lab = rgb2lab(target, channel_axis=self.channel_axis)
        self.target_means = cp.mean(lab, axis=(0, *self.axis), keepdims=True)
        self.target_stds = cp.std(lab, axis=(0, *self.axis), keepdims=True)


class ReinhardNormalizer(_ReinhardBase):

    def normalize(self, source: cp.ndarray):
        source = source.astype(cp.float32) / 255.0
        lab = rgb2lab(source, channel_axis=self.channel_axis)
        mus = cp.mean(lab, axis=self.axis, keepdims=True)
        stds = cp.std(lab, axis=self.axis, keepdims=True)
        
        lab_out = ((lab - mus) / (stds + 1e-8)) * self.target_stds + self.target_means
        
        rgb = lab2rgb(lab_out, channel_axis=self.channel_axis)  # cucim expects numpy
        rgb = cp.clip(rgb * 255.0, 0, 255).astype(cp.uint8)
        return rgb


class ReinhardNormalizer2(_ReinhardBase):

    def fit(self, target: cp.ndarray):
        super().fit(target)
        # Keep only channel dimension (broadcastable)
        self.target_means = self.target_means.squeeze()
        self.target_stds = self.target_stds.squeeze()

    def normalize(self, source: cp.ndarray) -> cp.ndarray:
        source = source.astype(cp.float32) / 255.0
        lab = rgb2lab(source, channel_axis=self.channel_axis)
        mus = cp.mean(lab, axis=self.axis, keepdims=True)
        stds = cp.std(lab, axis=self.axis, keepdims=True)

        lab_out = cp.empty_like(lab)
        L, A, B = self.split_channels(lab)
        mu_L, mu_A, mu_B = self.split_channels(mus)
        std_L, _, _ = self.split_channels(stds)
        
        # L channel
        q = (self.target_stds[0] - std_L) / (self.target_stds[0] + 1e-8)
        q = cp.where(q <= 0, 0.05, q)
        L_out = mu_L + (L - mu_L) * (1 + q)

        # A and B channels
        A_out = self.target_means[1] + (A - mu_A)
        B_out = self.target_means[2] + (B - mu_B)

        lab_out = self.stack_channels([L_out, A_out, B_out])
        
        rgb = lab2rgb(lab_out, channel_axis=self.channel_axis)  # cucim expects numpy
        rgb = cp.clip(rgb * 255.0, 0, 255).astype(cp.uint8)
        return rgb
