"""CuPy implementation of Macenko stain normalization."""

# from base import BaseStainNormalizer
import cupy as cp
import numpy as np
from .base import BaseStainNormalizer


class MacenkoNormalizer(BaseStainNormalizer):
    """
    Macenko stain normalization.

    Parameters
    ----------
    alpha : float, optional
        Algorithm parameter controlling the percentile range, by default 1
    beta : float, optional
        Percentile for maximum concentration estimation, by default 0.345
    channel_axis : int, optional
        Axis of color channels, by default: 1 (batch, channels, height, width)
    """

    def __init__(self, alpha=1, beta=0.345, channel_axis=1, *args, **kwargs):
        super().__init__()

        if not (0 < beta < 1):
            raise ValueError("beta must be in the range (0, 1)")
        self.beta = beta
        if not (0 < alpha < 100):
            raise ValueError("alpha must be in the range (0, 100)")
        self.alpha = alpha

        self.target_stain_matrix = cp.array(
            [[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]]
        )
        self.target_max_conc = cp.array([1.9705, 1.0308], dtype=np.float32)
        self._channel_axis = channel_axis

    def _image_to_absorbance_single(
        self, image, source_intensity=240, image_type="intensity", dtype=cp.float32
    ):
        """
        Converts a single image to absorbance and reshapes it into a (3, n_pixels) matrix.
        This method assumes channels are on axis 0 for the single image.

        Parameters
        ----------
        image : cp.ndarray
            Single image with shape (3, H, W) - channels always on axis 0.
        source_intensity : float, optional
            Reference intensity for the image. Defaults to 240.
        image_type : {'intensity', 'absorbance'}, optional
            The type of the input image. If 'intensity', it will be converted.
            If 'absorbance', it will only be reshaped. Defaults to 'intensity'.
        dtype : cp.dtype, optional
            The floating-point precision for the calculation. Defaults to cp.float32.

        Returns
        -------
        cp.ndarray
            A 2D array of absorbance values with shape (3, n_pixels).
        """

        # For single images extracted from batches, channels are always on axis 0
        c = image.shape[0]
        if c != 3:
            raise ValueError(f"Expected 3 channels on axis 0, got {c}")

        # Check for memory constraints - limit total pixels to prevent crashes
        total_pixels = image.size // c
        if total_pixels > 100_000_000:  # 100M pixels limit
            raise ValueError(
                f"Image too large ({total_pixels} pixels). Maximum supported: 100M pixels"
            )

        if image_type == "absorbance":
            absorbance = image.astype(dtype, copy=True)
        elif image_type == "intensity":
            dtype = cp.dtype(dtype)
            if dtype.kind != "f":
                raise ValueError("dtype must be a floating point type")

            input_dtype = image.dtype
            image = image.astype(dtype, copy=False)
            if source_intensity < 0:
                raise ValueError(
                    "Source transmitted light intensity must be a positive value."
                )
            source_intensity = float(source_intensity)
            if input_dtype == "f":
                min_val = source_intensity / 255.0
                max_val = source_intensity
            else:
                min_val = 1.0
                max_val = source_intensity

            # Core element-wise operations
            clipped_image = cp.maximum(cp.minimum(image, max_val), min_val)
            absorbance = -cp.log(clipped_image / max_val)
        else:
            raise ValueError("`image_type` must be either 'intensity' or 'absorbance'.")

        # Reshape to form a (n_channels=3, n_pixels) matrix
        # Since channels are already on axis 0, just reshape
        return absorbance.reshape((c, -1))

    def _image_to_absorbance(
        self, image, source_intensity=240, image_type="intensity", dtype=cp.float32
    ):
        """
        Converts an image to absorbance and reshapes it into a (3, n_pixels) matrix.

        This is a fully merged, high-performance function that combines data
        validation, optional absorbance conversion using a fused kernel, and
        final reshaping.

        Parameters
        ----------
        image : cp.ndarray
            The image to convert.
        source_intensity : float, optional
            Reference intensity for the image. Defaults to 240.
        image_type : {'intensity', 'absorbance'}, optional
            The type of the input image. If 'intensity', it will be converted.
            If 'absorbance', it will only be reshaped. Defaults to 'intensity'.
        dtype : cp.dtype, optional
            The floating-point precision for the calculation. Defaults to cp.float32.

        Returns
        -------
        cp.ndarray
            A 2D array of absorbance values with shape (3, n_pixels).
        """

        c = image.shape[self._channel_axis]
        if c != 3:
            raise ValueError("Expected an RGB image")

        # Check for memory constraints - limit total pixels to prevent crashes
        total_pixels = image.size // c
        if total_pixels > 100_000_000:  # 100M pixels limit
            raise ValueError(
                f"Image too large ({total_pixels} pixels). Maximum supported: 100M pixels"
            )

        if image_type == "absorbance":
            absorbance = image.astype(dtype, copy=True)
        elif image_type == "intensity":
            dtype = cp.dtype(dtype)
            if dtype.kind != "f":
                raise ValueError("dtype must be a floating point type")

            input_dtype = image.dtype
            image = image.astype(dtype, copy=False)
            if source_intensity < 0:
                raise ValueError(
                    "Source transmitted light intensity must be a positive value."
                )
            source_intensity = float(source_intensity)
            if input_dtype == "f":
                min_val = source_intensity / 255.0
                max_val = source_intensity
            else:
                min_val = 1.0
                max_val = source_intensity

            # These next three lines are the core, element-wise operations
            # that will be fused into a single GPU kernel for high performance.
            clipped_image = cp.maximum(cp.minimum(image, max_val), min_val)
            absorbance = -cp.log(clipped_image / max_val)
        else:
            raise ValueError("`image_type` must be either 'intensity' or 'absorbance'.")

        # reshape to form a (n_channels, n_pixels) matrix
        if self._channel_axis != 0:
            absorbance = cp.moveaxis(
                absorbance, source=self._channel_axis, destination=0
            )

        return absorbance.reshape((c, -1))

    def _absorbance_to_image(self, absorbance, source_intensity=255, dtype=cp.uint8):
        """Convert an absorbance (optical density) image back to a standard image.

        Parameters
        ----------
        absorbance : ndarray
            The absorbance image to convert back to a linear intensity range.
        source_intensity : float, optional
            Reference intensity for `image`. This should match what was used with
            ``rgb_to_absorbance`` when creating `absorbance`.
        dtype : numpy.dtype, optional
            The datatype to cast the output image to.

        Returns
        -------
        image : ndarray
            An image computed from the absorbance

        """
        # absorbance must be floating point
        absorbance_dtype = cp.promote_types(absorbance.dtype, cp.float16)
        absorbance = absorbance.astype(absorbance_dtype, copy=False)

        if source_intensity < 0:
            raise ValueError(
                "Source transmitted light intensity must be a positive value."
            )

        # specialized code paths depending on output dtype
        dtype = cp.dtype(dtype)
        if dtype == cp.uint8:
            rgb = cp.exp(-absorbance) * source_intensity
            rgb = cp.minimum(cp.maximum(rgb, 0), 255)
            return cp.around(rgb).astype(cp.uint8)
        if dtype.kind in "iu":
            # round to nearest integer and cast to desired integer dtype
            rgb = cp.exp(-absorbance) * source_intensity
            iinfo = cp.iinfo(dtype)
            rgb = cp.minimum(cp.maximum(rgb, iinfo.min), iinfo.max)
            return cp.around(rgb).astype(dtype, copy=False)

        return cp.exp(-absorbance) * source_intensity

    def _validate_image(self, image):
        if not isinstance(image, cp.ndarray):
            raise TypeError("Image must be of type cupy.ndarray.")
        if image.dtype.kind != "u" and image.min() < 0:
            raise ValueError("Image should not have negative values.")

    def _covariance(self, X):
        # For small matrices (3x3 covariance), use CuPy's optimized functions
        # This is much safer and more efficient than a custom CUDA kernel
        n_samples, n_features = X.shape

        # Limit memory usage by subsampling if too many pixels
        if n_samples > 50000:  # Limit to 50k pixels to prevent memory explosion
            indices = cp.random.choice(n_samples, size=50000, replace=False)
            X = X[indices]
            n_samples = 50000

        # Use CuPy's built-in covariance calculation which is optimized and safe
        # Transpose to get features as rows for cp.cov
        return cp.cov(X.T)

    def _get_stain_matrix(
        self,
        image,
        source_intensity=240,
        alpha=1,
        beta=0.345,
        *,
        image_type="intensity",
    ):
        """Extract the matrix of H & E stain coefficient from an image.

        Uses a method that selects stain vectors based on the angle distribution
        within a best-fit plane determined by principle component analysis (PCA)
        [1]_.

        Parameters
        ----------
        image : cp.ndarray
            RGB image to perform stain extraction on. Intensities should typically
            be within unsigned 8-bit integer intensity range ([0, 255]) when
            ``image_type == "intensity"``.
        source_intensity : float, optional
            Transmitted light intensity. The algorithm will clip image intensities
            above the specified `source_intensity` and then normalize by
            `source_intensity` so that `image` intensities are <= 1.0. Only used
            when `image_type=="intensity"`.
        alpha : float, optional
            Algorithm parameter controlling the ``[alpha, 100 - alpha]``
            percentile range used as a robust [min, max] estimate.
        beta : float, optional
            Absorbance (optical density) threshold below which to consider pixels
            as transparent. Transparent pixels are excluded from the estimation.

        Additional Parameters
        ---------------------
        channel_axis : int, optional
            The axis corresponding to color channels (default is the last axis).
        image_type : {"intensity", "absorbance"}, optional
            With the default `image_type` of `"intensity"`, the image will be
            transformed to `absorbance` units via ``image_to_absorbance``. If
            the input `image` is already an absorbance image, then `image_type`
            should be set to `"absorbance"` instead.

        Returns
        -------
        stain_coeff : cp.ndarray
            Stain attenuation coefficient matrix derived from the image, where
            the first column corresponds to H, the second column is E and the rows
            are RGB values.

        Notes
        -----
        The default `beta` of 0.345 is equivalent to the use of 0.15 in [1]_. The
        difference is due to our use of the natural log instead of a decadic log
        (log10) when computing the absorbance.

        References
        ----------
        .. [1] M. Macenko et al., "A method for normalizing histology slides for
            quantitative analysis," 2009 IEEE International Symposium on
            Biomedical Imaging: From Nano to Macro, 2009, pp. 1107-1110,
            doi: 10.1109/ISBI.2009.5193250.
            http://wwwx.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf
        """

        self._validate_image(image)

        if alpha < 0 or alpha > 100:
            raise ValueError("alpha must be a percentile in range [0, 100].")
        if beta < 0:
            raise ValueError("beta must be nonnegative.")

        # convert to absorbance (optical density) matrix of shape (3, n_pixels)
        absorbance = self._image_to_absorbance_single(
            image,
            source_intensity=source_intensity,
            image_type=image_type,
        )

        # remove transparent pixels
        absorbance = absorbance[:, cp.any(absorbance > beta, axis=0)]
        if absorbance.size == 0 or absorbance.shape[1] <= 1:
            raise ValueError(
                "Multiple pixels of the input must be above the `beta` threshold."
            )

        # Limit number of pixels to prevent memory issues
        n_pixels = absorbance.shape[1]
        if n_pixels > 100000:  # Limit to 100k pixels for PCA
            indices = cp.random.choice(n_pixels, size=100000, replace=False)
            absorbance = absorbance[:, indices]

        # compute eigenvectors (do small 3x3 matrix calculations)
        # Transpose for covariance calculation (pixels × features)
        absorbance_T = absorbance.T
        cov = self._covariance(absorbance_T)
        cov = cp.asnumpy(cov).astype(np.float32, copy=False)
        _, ev = np.linalg.eigh(cov)
        ev = ev[:, [2, 1]]
        # flip to ensure positive first coordinate so arctan2 angles are about 0
        if ev[0, 0] < 0:
            ev[:, 0] *= -1
        if ev[0, 1] < 0:
            ev[:, 1] *= -1

        # project on the plane spanned by the eigenvectors
        projection = cp.dot(cp.asarray(ev.T), absorbance)

        # find the vectors that span the whole data (min and max angles)
        phi = cp.arctan2(projection[1], projection[0])
        min_phi, max_phi = cp.percentile(phi, (alpha, 100 - alpha))
        # need these scalars on the host
        min_phi, max_phi = float(min_phi), float(max_phi)

        # project back to absorbance space
        v_min = np.array([np.cos(min_phi), np.sin(min_phi)], dtype=np.float32)
        v_max = np.array([np.cos(max_phi), np.sin(max_phi)], dtype=np.float32)
        v1 = np.dot(ev, v_min)
        v2 = np.dot(ev, v_max)

        # Make Hematoxylin (H) first and eosin (E) second by comparing the
        # R channel value
        if v1[0] < v2[0]:
            v1, v2 = v2, v1
        stain_coeff = np.stack((v1, v2), axis=-1)

        # renormalize columns to reduce numerical error
        norms = np.linalg.norm(stain_coeff, axis=0, keepdims=True)
        # Prevent division by zero
        norms = np.maximum(norms, 1e-12)
        stain_coeff /= norms
        return cp.asarray(stain_coeff)

    def _get_raw_concentrations(self, src_stain_coeff, absorbance):
        if absorbance.ndim != 2 or absorbance.shape[0] != 3:
            raise ValueError("`absorbance` must be shape (3, n_pixels)")

        # estimate the raw stain concentrations
        try:
            # pseudo-inverse with regularization to prevent numerical issues
            reg_factor = 1e-10
            AtA = cp.dot(src_stain_coeff.T, src_stain_coeff)
            AtA_reg = AtA + reg_factor * cp.eye(AtA.shape[0])
            coeff_pinv = cp.dot(
                cp.linalg.inv(AtA_reg),
                src_stain_coeff.T,
            )

            if cp.any(cp.isnan(coeff_pinv)) or cp.any(cp.isinf(coeff_pinv)):
                # fall back to cp.linalg.pinv if pseudo-inverse above failed
                conc_raw = cp.dot(cp.linalg.pinv(src_stain_coeff), absorbance)
            else:
                conc_raw = cp.dot(cp.asarray(coeff_pinv, order="F"), absorbance)
        except Exception as e:
            # Final fallback - use Moore-Penrose pseudoinverse
            print(f"Warning: Using fallback pseudoinverse due to: {e}")
            conc_raw = cp.dot(cp.linalg.pinv(src_stain_coeff), absorbance)

        return conc_raw

    def fit(self, target):
        """Fit the normalizer to a batch of target images.

        Parameters
        ----------
        target : cp.ndarray
            Batch of RGB images with shape (batch_size, channels, height, width)
            or (batch_size, height, width, channels) depending on channel_axis.
        """

        # Only accept batches (4D arrays)
        if len(target.shape) != 4:
            raise ValueError(
                "Expected a batch of images with shape (N, C, H, W) or (N, H, W, C)"
            )

        # Validate channel axis
        if self._channel_axis == 1:
            if target.shape[1] != 3:
                raise ValueError(
                    f"Expected 3 channels on axis 1, got {target.shape[1]}"
                )
        elif self._channel_axis == -1 or self._channel_axis == 3:
            if target.shape[-1] != 3:
                raise ValueError(
                    f"Expected 3 channels on last axis, got {target.shape[-1]}"
                )
        else:
            raise ValueError("channel_axis must be 1 or -1/3")

        # Use the first image in the batch for fitting
        target_single = target[0]

        # For single images extracted from batches, channels are always on axis 0
        # regardless of the original batch channel_axis
        absorbance = self._image_to_absorbance_single(target_single)
        self._batch_shape = target.shape  # Store batch shape for later use
        self._single_shape = target_single.shape  # Store single image shape
        self._source_stain_matrix = self._get_stain_matrix(
            target_single, alpha=self.alpha, beta=self.beta
        )
        self._source_max_conc = self._get_raw_concentrations(
            self._source_stain_matrix, absorbance
        )

    def normalize(self, source):
        """Normalize a batch of source images to match the target style.

        Parameters
        ----------
        source : cp.ndarray
            Batch of RGB images with shape (batch_size, channels, height, width)
            or (batch_size, height, width, channels) depending on channel_axis.

        Returns
        -------
        cp.ndarray
            Normalized batch of images with the same shape as input.
        """

        # Only accept batches (4D arrays)
        if len(source.shape) != 4:
            raise ValueError(
                "Expected a batch of images with shape (N, C, H, W) or (N, H, W, C)"
            )

        # Validate shape matches what was used for fitting
        if source.shape != self._batch_shape:
            # Allow different batch sizes but same spatial/channel dimensions
            if self._channel_axis == 1:
                if source.shape[1:] != self._batch_shape[1:]:
                    raise ValueError(
                        f"Source shape {source.shape[1:]} doesn't match target shape {self._batch_shape[1:]} (excluding batch dimension)"
                    )
            else:
                if source.shape[1:] != self._batch_shape[1:]:
                    raise ValueError(
                        f"Source shape {source.shape[1:]} doesn't match target shape {self._batch_shape[1:]} (excluding batch dimension)"
                    )

        # Process each image in the batch
        normalized_batch = []
        for i in range(source.shape[0]):
            single_image = source[i]
            normalized_single = self._normalize_single_image(single_image)
            normalized_batch.append(normalized_single)
        return cp.stack(normalized_batch, axis=0)

    def _normalize_single_image(self, source):
        """Normalize a single image."""
        # Convert source to absorbance - use single image method
        absorbance = self._image_to_absorbance_single(source)

        # Get concentrations for the source image
        source_conc = self._get_raw_concentrations(
            self._source_stain_matrix, absorbance
        )

        # verify conc_raw is shape (2, n_pixels)
        if source_conc.ndim != 2 or source_conc.shape[0] != 2:
            raise ValueError(
                "`conc_raw` must be a 2D array of concentrations with size 2 on axis 0."
            )
        if self.target_stain_matrix.ndim != 2 or self.target_stain_matrix.shape[0] != 3:
            raise ValueError(
                "`ref_stain_coeff` must be a shape (3, n) matrix, representing "
                "n stain vectors."
            )
        if len(self.target_max_conc) != self.target_stain_matrix.shape[1]:
            raise ValueError(
                "`ref_max_conc` must have length equal to the number of stain "
                "coefficient vectors."
            )

        # normalize stain concentrations
        # Note: calling percentile separately for each channel is faster than:
        #       max_conc = cp.percentile(conc_raw, 100 - alpha, axis=1)
        max_conc = cp.concatenate(
            [
                cp.percentile(ch_raw, 100 - self.alpha)[cp.newaxis]
                for ch_raw in source_conc
            ]
        )
        normalization_factors = self.target_max_conc / max_conc
        normalized_conc = source_conc * normalization_factors[:, cp.newaxis]

        # reconstruct the image based on the reference stain matrix
        absorbance_norm = self.target_stain_matrix.dot(normalized_conc)
        image_norm = self._absorbance_to_image(absorbance_norm, 255, cp.uint8)

        # restore original shape for each channel
        # For single images from batches, channels are always on axis 0
        # So spatial_shape is everything after axis 0
        spatial_shape = source.shape[1:]  # Skip the channel dimension (axis 0)
        image_norm = cp.reshape(image_norm, (3,) + spatial_shape)

        # For single images extracted from batches, channels are already on axis 0
        # No need to move axes since the output should also have channels on axis 0
        return image_norm
