/* common.cuh - error handling and small multi-accumulator reductions. */
#ifndef PRIA_COMMON_CUH
#define PRIA_COMMON_CUH

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

namespace pria {

/* Error handling: remember the first CUDA error and bail out of the solve.
 * PRIA_CHECK is used inside functions returning cudaError_t. */
#define PRIA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err_ = (call);                                           \
        if (err_ != cudaSuccess) {                                           \
            fprintf(stderr, "pria: CUDA error %s at %s:%d: %s\n",            \
                    cudaGetErrorName(err_), __FILE__, __LINE__,              \
                    cudaGetErrorString(err_));                               \
            return err_;                                                     \
        }                                                                    \
    } while (0)

constexpr int kBlock = 256;

inline int launch_grid(int64_t n, int block = kBlock, int max_grid = 4096) {
    int64_t g = (n + block - 1) / block;
    if (g < 1) g = 1;
    if (g > max_grid) g = max_grid;
    return (int)g;
}

/* ---- deterministic two-stage reductions -------------------------------- *
 *
 * reduce_k<K>(n, f, out) accumulates K double sums over g in [0, n), where
 * the functor adds its contribution via f(g, acc). Stage one writes fixed
 * per-block partials, stage two folds them in a single block, so results are
 * bitwise reproducible run to run.
 */
constexpr int kReduceMaxGrid = 720;

template <int K, typename F>
__global__ void k_reduce_partial(int64_t n, F f, double* partials) {
    double acc[K];
#pragma unroll
    for (int k = 0; k < K; ++k) acc[k] = 0.0;

    for (int64_t g = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; g < n;
         g += (int64_t)gridDim.x * blockDim.x)
        f(g, acc);

    __shared__ double sh[K][kBlock];
#pragma unroll
    for (int k = 0; k < K; ++k) sh[k][threadIdx.x] = acc[k];
    __syncthreads();
    for (int stride = kBlock / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
#pragma unroll
            for (int k = 0; k < K; ++k)
                sh[k][threadIdx.x] += sh[k][threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
#pragma unroll
        for (int k = 0; k < K; ++k) partials[blockIdx.x * K + k] = sh[k][0];
}

template <int K>
__global__ void k_reduce_final(int nparts, const double* partials, double* out) {
    double acc[K];
#pragma unroll
    for (int k = 0; k < K; ++k) acc[k] = 0.0;
    for (int p = threadIdx.x; p < nparts; p += blockDim.x)
#pragma unroll
        for (int k = 0; k < K; ++k) acc[k] += partials[p * K + k];

    __shared__ double sh[K][kBlock];
#pragma unroll
    for (int k = 0; k < K; ++k) sh[k][threadIdx.x] = acc[k];
    __syncthreads();
    for (int stride = kBlock / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
#pragma unroll
            for (int k = 0; k < K; ++k)
                sh[k][threadIdx.x] += sh[k][threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
#pragma unroll
        for (int k = 0; k < K; ++k) out[k] = sh[k][0];
}

/* partials must hold kReduceMaxGrid*K doubles; out holds K doubles. */
template <int K, typename F>
void reduce_k(int64_t n, F f, double* partials, double* out,
              cudaStream_t stream) {
    int grid = launch_grid(n, kBlock, kReduceMaxGrid);
    k_reduce_partial<K, F><<<grid, kBlock, 0, stream>>>(n, f, partials);
    k_reduce_final<K><<<1, kBlock, 0, stream>>>(grid, partials, out);
}

} // namespace pria

#endif /* PRIA_COMMON_CUH */
