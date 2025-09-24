import cupy as cp

def covariance(X):
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
        cp.ceil(n_features / block_size[0]),
        cp.ceil(n_features / block_size[1])
    )

    # Call the kernel with the correct (grid, block, args) syntax
    cov_kernel(
        grid_size,
        block_size,
        (X, Cov, n_samples, n_features)
    )

    return Cov