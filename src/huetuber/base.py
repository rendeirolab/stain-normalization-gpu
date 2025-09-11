from abc import ABC, abstractmethod


class BaseStainNormalizer(ABC):
    """
    Abstract base class for stain normalization algorithms.
    """

    def __call__(self, *args, **kwargs):
        return self.normalize(*args, **kwargs)

    @abstractmethod
    def fit(self, target):
        """
        Fit the normalizer to the target image.

        Parameters
        ----------
        target (ndarray) : Target image in RGB format. Could be any array type (e.g., NumPy, CuPy).

        """
        pass

    @abstractmethod
    def normalize(self, source):
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
