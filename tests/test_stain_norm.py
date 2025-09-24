import pytest
import cupy as cp

from huetuber import (
    ReinhardNormalizer,
    ReinhardNormalizer2,
    HistogramMatching,
    MacenkoNormalizer,
)


NORMALIZERS = [
    ReinhardNormalizer(channel_axis=1),
    ReinhardNormalizer2(channel_axis=1),
    HistogramMatching(channel_axis=1),
    MacenkoNormalizer(channel_axis=1),
]


target_images = cp.random.randint(0, 256, size=(2, 3, 256, 256), dtype=cp.uint8)
source_images = cp.random.randint(0, 256, size=(2, 3, 256, 256), dtype=cp.uint8)


@pytest.mark.parametrize("norm", NORMALIZERS)
def test_fit(norm):
    norm.fit(target_images)


@pytest.mark.parametrize("norm", NORMALIZERS)
def test_normalize(norm):
    norm.fit(target_images)
    out = norm(source_images)
    assert out.shape == source_images.shape
    assert out.dtype == source_images.dtype
    assert cp.all(out >= 0) and cp.all(out <= 255)
