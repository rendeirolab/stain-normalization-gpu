"""CuPy implementation of Macenko stain normalization."""

from huetuber.base import BaseStainNormalizer
import cupy as cp
import numpy as np
# from kernel import covariance


# class MacenkoNormalizer(BaseStainNormalizer):
#     def __init__(self, *args, **kwargs):
#         pass

#     def fit(self, target):
#         pass

#     def normalize(self, source):
#         # Implement normalization logic here
#         pass

# ------------UNCOMMENT OUT THE CODE BELOW TO USE THE FULL IMPLEMENTATION----------------

# class MacenkoNormalizer(BaseStainNormalizer):
#     """
#     Macenko stain normalization.

#     Parameters
#     ----------
#     beta : float, optional
#         Percentile for maximum concentration estimation, by default 0.15
#     channel_axis : int, optional
#         Axis of color channels, by default: 1
#     """

#     def __init__(self, alpha=1, beta=0.345, channel_axis=1, *args, **kwargs):
#         super().__init__()

#         if not (0 < beta < 1):
#             raise ValueError("beta must be in the range (0, 1)")
#         self.beta = beta
#         if not (0 < alpha < 100):
#             raise ValueError("alpha must be in the range (0, 100)")
#         self.alpha = alpha

#         self.target_stain_matrix = cp.array([
#             [0.5626, 0.2159],
#             [0.7201, 0.8012],
#             [0.4062, 0.5581]
#             ])
#         self.target_max_conc = cp.array([1.9705, 1.0308], dtype=np.float32)
#         self._channel_axis = channel_axis

#     @cp.fuse()
#     # This function has been written to completion - no further edits needed
#     def _image_to_absorbance(self,
#         image,
#         source_intensity=240,
#         image_type="intensity",
#         dtype=cp.float32
#     ):
#         """
#         Converts an image to absorbance and reshapes it into a (3, n_pixels) matrix.

#         This is a fully merged, high-performance function that combines data
#         validation, optional absorbance conversion using a fused kernel, and
#         final reshaping.

#         Parameters
#         ----------
#         image : cp.ndarray
#             The image to convert.
#         source_intensity : float, optional
#             Reference intensity for the image. Defaults to 240.
#         image_type : {'intensity', 'absorbance'}, optional
#             The type of the input image. If 'intensity', it will be converted.
#             If 'absorbance', it will only be reshaped. Defaults to 'intensity'.
#         dtype : cp.dtype, optional
#             The floating-point precision for the calculation. Defaults to cp.float32.

#         Returns
#         -------
#         cp.ndarray
#             A 2D array of absorbance values with shape (3, n_pixels).
#         """

#         c = image.shape[self._channel_axis]
#         if c != 3:
#             raise ValueError("Expected an RGB image")
        
#         if image_type == "absorbance":
#             absorbance = image.astype(dtype, copy=True)
#         elif image_type == "intensity":
#             dtype = cp.dtype(dtype)
#             if dtype.kind != "f":
#                 raise ValueError("dtype must be a floating point type")

#             input_dtype = image.dtype
#             image = image.astype(dtype, copy=False)
#             if source_intensity < 0:
#                 raise ValueError(
#                     "Source transmitted light intensity must be a positive value."
#                 )
#             source_intensity = float(source_intensity)
#             if input_dtype == "f":
#                 min_val = source_intensity / 255.0
#                 max_val = source_intensity
#             else:
#                 min_val = 1.0
#                 max_val = source_intensity

#             # These next three lines are the core, element-wise operations
#             # that will be fused into a single GPU kernel for high performance.
#             clipped_image = cp.maximum(cp.minimum(image, max_val), min_val)
#             absorbance = -cp.log(clipped_image / max_val)
#         else:
#             raise ValueError(
#                 "`image_type` must be either 'intensity' or 'absorbance'."
#             )
        
#         # reshape to form a (n_channels, n_pixels) matrix
#         if self._channel_axis != 0:
#             absorbance = cp.moveaxis(absorbance, source=self._channel_axis, destination=0)

#         return absorbance.reshape((c, -1))
    
#     # @cp.fuse()
#     # def _absorbance_to_image_float(self, absorbance, source_intensity):
#     #     return cp.exp(-absorbance) * source_intensity


#     # @cp.fuse()
#     # def _absorbance_to_image_int(self, absorbance, source_intensity, min_val, max_val):
#     #     rgb = cp.exp(-absorbance) * source_intensity
#     #     # prevent overflow/underflow
#     #     rgb = cp.minimum(cp.maximum(rgb, min_val), max_val)
#     #     return cp.around(rgb)


#     # @cp.fuse()
#     # def _absorbance_to_image_uint8(self, absorbance, source_intensity):
#     #     rgb = cp.exp(-absorbance) * source_intensity
#     #     # prevent overflow/underflow
#     #     rgb = cp.minimum(cp.maximum(rgb, 0), 255)
#     #     return cp.around(rgb).astype(cp.uint8)
    
#     @cp.fuse()
#     def _absorbance_to_image(self, absorbance, source_intensity=255, dtype=cp.uint8):
#         """Convert an absorbance (optical density) image back to a standard image.

#         Parameters
#         ----------
#         absorbance : ndarray
#             The absorbance image to convert back to a linear intensity range.
#         source_intensity : float, optional
#             Reference intensity for `image`. This should match what was used with
#             ``rgb_to_absorbance`` when creating `absorbance`.
#         dtype : numpy.dtype, optional
#             The datatype to cast the output image to.

#         Returns
#         -------
#         image : ndarray
#             An image computed from the absorbance

#         """
#         # absorbance must be floating point
#         absorbance_dtype = cp.promote_types(absorbance.dtype, cp.float16)
#         absorbance = absorbance.astype(absorbance_dtype, copy=False)

#         if source_intensity < 0:
#             raise ValueError(
#                 "Source transmitted light intensity must be a positive value."
#             )

#         # specialized code paths depending on output dtype
#         dtype = cp.dtype(dtype)
#         if dtype == cp.uint8:
#             rgb = cp.exp(-absorbance) * source_intensity
#             rgb = cp.minimum(cp.maximum(rgb, 0), 255)
#             return cp.around(rgb).astype(cp.uint8)
#         if dtype.kind in "iu":
#             # round to nearest integer and cast to desired integer dtype
#             rgb = cp.exp(-absorbance) * source_intensity
#             iinfo = cp.iinfo(dtype)
#             rgb = cp.minimum(cp.maximum(rgb, iinfo.min), iinfo.max)
#             return cp.around(rgb).astype(dtype, copy=False)
        
#         return cp.exp(-absorbance) * source_intensity
    
#     def _validate_image(self, image):
#         if not isinstance(image, cp.ndarray):
#             raise TypeError("Image must be of type cupy.ndarray.")
#         if image.dtype.kind != "u" and image.min() < 0:
#             raise ValueError("Image should not have negative values.")


#     # def _prep_channel_axis(self, channel_axis, ndim):
#     #     if (channel_axis < -ndim) or (channel_axis > ndim - 1):
#     #         raise ValueError(
#     #             f"`channel_axis={channel_axis}` exceeds image dimensions"
#     #         )
#     #     return channel_axis % ndim

#     def _covariance(self, X):
#         # CUDA C kernel (remains unchanged)
#         code = r'''
#         extern "C" __global__
#         void covariance_matrix(const float* X, float* Cov, int n_samples, int n_features) {
#             int i = blockIdx.x * blockDim.x + threadIdx.x;
#             int j = blockIdx.y * blockDim.y + threadIdx.y;

#             if (i < n_features && j < n_features) {
#                 float mean_i = 0.0f, mean_j = 0.0f;
#                 for (int k = 0; k < n_samples; ++k) {
#                     mean_i += X[k * n_features + i];
#                     mean_j += X[k * n_features + j];
#                 }
#                 mean_i /= n_samples;
#                 mean_j /= n_samples;

#                 float cov = 0.0f;
#                 for (int k = 0; k < n_samples; ++k) {
#                     float xi = X[k * n_features + i] - mean_i;
#                     float xj = X[k * n_features + j] - mean_j;
#                     cov += xi * xj;
#                 }
#                 Cov[i * n_features + j] = cov / (n_samples - 1);
#             }
#         }
#         '''

#         # Compile the kernel
#         module = cp.RawModule(code=code)
#         cov_kernel = module.get_function("covariance_matrix")

#         n_samples, n_features = X.shape
#         Cov = cp.zeros((n_features, n_features), dtype=cp.float32)

#         # Define the number of threads per block
#         # A 16x16 block is a common and efficient choice
#         block_size = (16, 16)

#         # Calculate the number of blocks needed in each dimension
#         # We use ceiling division to ensure we have enough blocks
#         grid_size = (
#             cp.ceil(n_features / block_size[0]),
#             cp.ceil(n_features / block_size[1])
#         )

#         # Call the kernel with the correct (grid, block, args) syntax
#         cov_kernel(
#             grid_size,
#             block_size,
#             (X, Cov, n_samples, n_features)
#         )

#         return Cov
    
#     def _get_stain_matrix(self,
#         image,
#         source_intensity=240,
#         alpha=1,
#         beta=0.345,
#         *,
#         image_type="intensity",
#     ):
#         """Extract the matrix of H & E stain coefficient from an image.

#         Uses a method that selects stain vectors based on the angle distribution
#         within a best-fit plane determined by principle component analysis (PCA)
#         [1]_.

#         Parameters
#         ----------
#         image : cp.ndarray
#             RGB image to perform stain extraction on. Intensities should typically
#             be within unsigned 8-bit integer intensity range ([0, 255]) when
#             ``image_type == "intensity"``.
#         source_intensity : float, optional
#             Transmitted light intensity. The algorithm will clip image intensities
#             above the specified `source_intensity` and then normalize by
#             `source_intensity` so that `image` intensities are <= 1.0. Only used
#             when `image_type=="intensity"`.
#         alpha : float, optional
#             Algorithm parameter controlling the ``[alpha, 100 - alpha]``
#             percentile range used as a robust [min, max] estimate.
#         beta : float, optional
#             Absorbance (optical density) threshold below which to consider pixels
#             as transparent. Transparent pixels are excluded from the estimation.

#         Additional Parameters
#         ---------------------
#         channel_axis : int, optional
#             The axis corresponding to color channels (default is the last axis).
#         image_type : {"intensity", "absorbance"}, optional
#             With the default `image_type` of `"intensity"`, the image will be
#             transformed to `absorbance` units via ``image_to_absorbance``. If
#             the input `image` is already an absorbance image, then `image_type`
#             should be set to `"absorbance"` instead.

#         Returns
#         -------
#         stain_coeff : cp.ndarray
#             Stain attenuation coefficient matrix derived from the image, where
#             the first column corresponds to H, the second column is E and the rows
#             are RGB values.

#         Notes
#         -----
#         The default `beta` of 0.345 is equivalent to the use of 0.15 in [1]_. The
#         difference is due to our use of the natural log instead of a decadic log
#         (log10) when computing the absorbance.

#         References
#         ----------
#         .. [1] M. Macenko et al., "A method for normalizing histology slides for
#             quantitative analysis," 2009 IEEE International Symposium on
#             Biomedical Imaging: From Nano to Macro, 2009, pp. 1107-1110,
#             doi: 10.1109/ISBI.2009.5193250.
#             http://wwwx.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf
#         """

#         self._validate_image(image)

#         if alpha < 0 or alpha > 100:
#             raise ValueError("alpha must be a percentile in range [0, 100].")
#         if beta < 0:
#             raise ValueError("beta must be nonnegative.")

#         # convert to absorbance (optical density) matrix of shape (3, n_pixels)
#         absorbance = self._image_to_absorbance(
#             image,
#             source_intensity=source_intensity,
#             image_type=image_type,
#         )

#         # remove transparent pixels
#         absorbance = absorbance[:, cp.any(absorbance > beta, axis=0)]
#         if absorbance.size == 0 or absorbance.shape[1] <= 1:
#             raise ValueError(
#                 "Multiple pixels of the input must be above the `beta` threshold."
#             )

#         # compute eigenvectors (do small 3x3 matrix calculations on the host)
#         cov = self._covariance(absorbance)
#         cov = cp.asnumpy(cov).astype(np.float32, copy=False)
#         _, ev = np.linalg.eigh(cov)
#         ev = ev[:, [2, 1]]
#         # flip to ensure positive first coordinate so arctan2 angles are about 0
#         if ev[0, 0] < 0:
#             ev[:, 0] *= -1
#         if ev[0, 1] < 0:
#             ev[:, 1] *= -1

#         # project on the plane spanned by the eigenvectors
#         projection = cp.dot(cp.asarray(ev.T), absorbance)

#         # find the vectors that span the whole data (min and max angles)
#         phi = cp.arctan2(projection[1], projection[0])
#         min_phi, max_phi = cp.percentile(phi, (alpha, 100 - alpha))
#         # need these scalars on the host
#         min_phi, max_phi = float(min_phi), float(max_phi)

#         # project back to absorbance space
#         v_min = np.array([cp.cos(min_phi), cp.sin(min_phi)], dtype=np.float32)
#         v_max = np.array([cp.cos(max_phi), cp.sin(max_phi)], dtype=np.float32)
#         v1 = np.dot(ev, v_min)
#         v2 = np.dot(ev, v_max)

#         # Make Hematoxylin (H) first and eosin (E) second by comparing the
#         # R channel value
#         if v1[0] < v2[0]:
#             v1, v2 = v2, v1
#         stain_coeff = np.stack((v1, v2), axis=-1)

#         # renormalize columns to reduce numerical error
#         stain_coeff /= np.linalg.norm(stain_coeff, axis=0, keepdims=True)
#         return cp.asarray(stain_coeff)


#     def _get_raw_concentrations(self, src_stain_coeff, absorbance):
#         if absorbance.ndim != 2 or absorbance.shape[0] != 3:
#             raise ValueError("`absorbance` must be shape (3, n_pixels)")

#         # estimate the raw stain concentrations

#         # pseudo-inverse
#         coeff_pinv = cp.dot(
#             cp.linalg.inv(cp.dot(src_stain_coeff.T, src_stain_coeff)),
#             src_stain_coeff.T,
#         )
#         if cp.any(cp.isnan(coeff_pinv)):
#             # fall back to cp.linalg.lstsq if pseudo-inverse above failed
#             conc_raw = cp.linalg.lstsq(src_stain_coeff, absorbance, rcond=None)[0]
#         else:
#             conc_raw = cp.dot(cp.asarray(coeff_pinv, order="F"), absorbance)

#         return conc_raw
    
#     def fit(self, image):

#         if 3 not in image.shape or len(image.shape) != 3:
#             raise ValueError("Expected an RGB image")
        
#         absorbance = self._image_to_absorbance(
#             image,
#             source_intensity=240,
#             image_type="intensity",
#         )
#         self._shape = image.shape
#         self._source_stain_matrix = self._get_stain_matrix(image, alpha=self.alpha, beta=self.beta)
#         self._source_max_conc = self._get_raw_concentrations(self._source_stain_matrix, absorbance)
    
#     def normalize(self):
#         """Determine normalized image from concentrations.

#         Note: This function will also modify conc_raw in-place.
#         """

#         # verify conc_raw is shape (2, n_pixels)
#         if self._source_max_conc.ndim != 2 or self._source_max_conc.shape[0] != 2:
#             raise ValueError(
#                 "`conc_raw` must be a 2D array of concentrations with size 2 on "
#                 "axis 0."
#             )
#         if self.target_stain_matrix.ndim != 2 or self.target_stain_matrix.shape[0] != 3:
#             raise ValueError(
#                 "`ref_stain_coeff` must be a shape (3, n) matrix, representing "
#                 "n stain vectors."
#             )
#         if len(self.target_max_conc) != self.target_stain_matrix.shape[1]:
#             raise ValueError(
#                 "`ref_max_conc` must have length equal to the number of stain "
#                 "coefficient vectors."
#             )

#         # normalize stain concentrations
#         # Note: calling percentile separately for each channel is faster than:
#         #       max_conc = cp.percentile(conc_raw, 100 - alpha, axis=1)
#         max_conc = cp.concatenate(
#             [
#                 cp.percentile(ch_raw, 100 - self.alpha)[np.newaxis]
#                 for ch_raw in self._source_max_conc
#             ]
#         )
#         normalization_factors = self.target_max_conc / max_conc
#         self.source_max_conc = self._source_max_conc * normalization_factors[:, cp.newaxis]

#         # reconstruct the image based on the reference stain matrix
#         absorbance_norm = self.target_stain_matrix.dot(self.source_max_conc)
#         image_norm = self._absorbance_to_image(absorbance_norm, dtype=np.uint8)

#         # restore original shape for each channel
#         spatial_shape = (
#             self._shape[:self._channel_axis] + self._shape[self._channel_axis + 1 :]
#         )
#         image_norm = cp.reshape(image_norm, (3,) + spatial_shape)

#         # move channels from axis 0 to channel_axis
#         if self._channel_axis != 0:
#             image_norm = cp.moveaxis(image_norm, source=0, destination=self._channel_axis)
#         # restore original shape
#         return image_norm
    
class MacenkoNormalizer(BaseStainNormalizer):
    """
    Macenko stain normalization.

    Parameters
    ----------
    beta : float, optional
        Percentile for maximum concentration estimation, by default 0.15
    channel_axis : int, optional
        Axis of color channels, by default: 0
    """

    def __init__(self, alpha=1, beta=0.345, channel_axis=0, *args, **kwargs):
        super().__init__()

        if not (0 < beta < 1):
            raise ValueError("beta must be in the range (0, 1)")
        self.beta = beta
        if not (0 < alpha < 100):
            raise ValueError("alpha must be in the range (0, 100)")
        self.alpha = alpha

        self.target_stain_matrix = cp.array([
            [0.5626, 0.2159],
            [0.7201, 0.8012],
            [0.4062, 0.5581]
            ])
        self.target_max_conc = cp.array([1.9705, 1.0308], dtype=np.float32)
        self._channel_axis = channel_axis

    #@cp.fuse()
    # This function has been written to completion - no further edits needed
    def _image_to_absorbance(self,
        image,
        source_intensity=240,
        image_type="intensity",
        dtype=cp.float32
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
            raise ValueError(
                "`image_type` must be either 'intensity' or 'absorbance'."
            )

        # reshape to form a (n_channels, n_pixels) matrix
        if self._channel_axis != 0:
            absorbance = cp.moveaxis(absorbance, source=self._channel_axis, destination=0)

        return absorbance.reshape((c, -1))

    # @cp.fuse()
    # def _absorbance_to_image_float(self, absorbance, source_intensity):
    #     return cp.exp(-absorbance) * source_intensity


    # @cp.fuse()
    # def _absorbance_to_image_int(self, absorbance, source_intensity, min_val, max_val):
    #     rgb = cp.exp(-absorbance) * source_intensity
    #     # prevent overflow/underflow
    #     rgb = cp.minimum(cp.maximum(rgb, min_val), max_val)
    #     return cp.around(rgb)


    # @cp.fuse()
    # def _absorbance_to_image_uint8(self, absorbance, source_intensity):
    #     rgb = cp.exp(-absorbance) * source_intensity
    #     # prevent overflow/underflow
    #     rgb = cp.minimum(cp.maximum(rgb, 0), 255)
    #     return cp.around(rgb).astype(cp.uint8)

    @cp.fuse()
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


    # def _prep_channel_axis(self, channel_axis, ndim):
    #     if (channel_axis < -ndim) or (channel_axis > ndim - 1):
    #         raise ValueError(
    #             f"`channel_axis={channel_axis}` exceeds image dimensions"
    #         )
    #     return channel_axis % ndim

    def _covariance(self, X):
        # CUDA C kernel (remains unchanged)
        code = r'''
        extern "C" __global__
        void covariance_matrix(const float* X, float* Cov, int n_samples, int n_features) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            int j = blockIdx.y * blockDim.y + threadIdx.y;

            if (i < n_features && j < n_features) {
                float mean_i = 0.0f, mean_j = 0.0f;
                for (int k = 0; k < n_samples; ++k) {
                    mean_i += X[k * n_features + i];
                    mean_j += X[k * n_features + j];
                }
                mean_i /= n_samples;
                mean_j /= n_samples;

                float cov = 0.0f;
                for (int k = 0; k < n_samples; ++k) {
                    float xi = X[k * n_features + i] - mean_i;
                    float xj = X[k * n_features + j] - mean_j;
                    cov += xi * xj;
                }
                Cov[i * n_features + j] = cov / (n_samples - 1);
            }
        }
        '''

        # Compile the kernel
        module = cp.RawModule(code=code)
        cov_kernel = module.get_function("covariance_matrix")

        n_samples, n_features = X.shape
        Cov = cp.zeros((n_features, n_features), dtype=cp.float32)

        # Define the number of threads per block
        # A 16x16 block is a common and efficient choice
        block_size = (16, 16)

        # Calculate the number of blocks needed in each dimension
        # We use ceiling division to ensure we have enough blocks
        grid_size = (
            int(cp.ceil(n_features / block_size[0])),
            int(cp.ceil(n_features / block_size[1]))
        )

        # Call the kernel with the correct (grid, block, args) syntax
        cov_kernel(
            grid_size,
            block_size,
            (X, Cov, n_samples, n_features)
        )

        return Cov

    def _get_stain_matrix(self,
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
        absorbance = self._image_to_absorbance(
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

        # compute eigenvectors (do small 3x3 matrix calculations on the host)
        cov = self._covariance(absorbance)
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
        v_min = np.array([cp.cos(min_phi), cp.sin(min_phi)], dtype=np.float32)
        v_max = np.array([cp.cos(max_phi), cp.sin(max_phi)], dtype=np.float32)
        v1 = np.dot(ev, v_min)
        v2 = np.dot(ev, v_max)

        # Make Hematoxylin (H) first and eosin (E) second by comparing the
        # R channel value
        if v1[0] < v2[0]:
            v1, v2 = v2, v1
        stain_coeff = np.stack((v1, v2), axis=-1)

        # renormalize columns to reduce numerical error
        stain_coeff /= np.linalg.norm(stain_coeff, axis=0, keepdims=True)
        return cp.asarray(stain_coeff)


    def _get_raw_concentrations(self, src_stain_coeff, absorbance):
        if absorbance.ndim != 2 or absorbance.shape[0] != 3:
            raise ValueError("`absorbance` must be shape (3, n_pixels)")

        # estimate the raw stain concentrations

        # pseudo-inverse
        coeff_pinv = cp.dot(
            cp.linalg.inv(cp.dot(src_stain_coeff.T, src_stain_coeff)),
            src_stain_coeff.T,
        )
        if cp.any(cp.isnan(coeff_pinv)):
            # fall back to cp.linalg.lstsq if pseudo-inverse above failed
            conc_raw = cp.linalg.lstsq(src_stain_coeff, absorbance, rcond=None)[0]
        else:
            conc_raw = cp.dot(cp.asarray(coeff_pinv, order="F"), absorbance)

        return conc_raw

    def fit(self, image):

        if 3 not in image.shape or len(image.shape) != 3:
            raise ValueError("Expected an RGB image")

        absorbance = self._image_to_absorbance(image)
        self._shape = image.shape
        self._source_stain_matrix = self._get_stain_matrix(image, alpha=self.alpha, beta=self.beta)
        self._source_max_conc = self._get_raw_concentrations(self._source_stain_matrix, absorbance)

    def normalize(self):
        """Determine normalized image from concentrations.

        Note: This function will also modify conc_raw in-place.
        """

        # verify conc_raw is shape (2, n_pixels)
        if self._source_max_conc.ndim != 2 or self._source_max_conc.shape[0] != 2:
            raise ValueError(
                "`conc_raw` must be a 2D array of concentrations with size 2 on "
                "axis 0."
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
                cp.percentile(ch_raw, 100 - self.alpha)[np.newaxis]
                for ch_raw in self._source_max_conc
            ]
        )
        normalization_factors = self.target_max_conc / max_conc
        self.source_max_conc = self._source_max_conc * normalization_factors[:, cp.newaxis]

        # reconstruct the image based on the reference stain matrix
        absorbance_norm = self.target_stain_matrix.dot(self.source_max_conc)
        image_norm = self._absorbance_to_image(absorbance_norm, dtype=np.uint8)

        # restore original shape for each channel
        spatial_shape = (
            self._shape[:self._channel_axis] + self._shape[self._channel_axis + 1 :]
        )
        image_norm = cp.reshape(image_norm, (3,) + spatial_shape)

        # move channels from axis 0 to channel_axis
        if self._channel_axis != 0:
            image_norm = cp.moveaxis(image_norm, source=0, destination=self._channel_axis)
        # restore original shape
        return image_norm
    
if __name__ == "__main__":
    norm = MacenkoNormalizer()
    print("class initialized without any errors")