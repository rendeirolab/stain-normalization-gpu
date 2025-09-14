from abc import ABC, abstractmethod

import cupy as cp

class BaseStainNormalizer(ABC):
    """
    Abstract base class for stain normalization algorithms.
    """

    def __call__(self, *args, **kwargs):
        return self.normalize(*args, **kwargs)

    @abstractmethod
    def fit(self, target: cp.ndarray) -> None:
        """
        Fit the normalizer to the target image.

        Parameters
        ----------
        target (ndarray) : Target image in RGB format. Could be any array type (e.g., NumPy, CuPy).

        """
        pass

    @abstractmethod
    def normalize(self, source: cp.ndarray) -> cp.ndarray:
        """
        Normalize the source image to match the target image.

        Parameters
        ----------
        source (ndarray) : Source image in RGB format. Could be any array type (e.g., NumPy, CuPy).

        Returns
        -------
        ndarray : Normalized image in RGB format. Same array type as input.
        """
        pass
