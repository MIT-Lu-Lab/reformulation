/* solver.cuh - the PRIA solver: restarted Halpern-accelerated reflected PDHG
 * on the saddle-point form of the JRP LP, mirroring the JAX implementation
 * (pria-jax/pria/solver.py) step for step.
 *
 * Saddle-point problem (after eliminating the LP's y/z primal variables):
 *
 *   min_{x in X} max_{y in Y, z in Z}  <h, x> + <y, x> + <z, x>
 *
 *   X: per (i, t), {x[i][.][t] >= 0, sum_{s<=t} x[i][s][t] = 1}
 *   Y: per (i, s), {y[i][s][.] >= 0, sum_{t>=s} y[i][s][t] = K_i}
 *   Z: per s,      {z[.][s][.] >= 0, sum_{i, t>=s} z[i][s][t] = K0}
 *
 * One PDHG operator application T(x, y, z) with step eta and primal weight
 * omega (tau = eta/omega, sigma = eta*omega):
 *
 *   x' = proj_X(x - tau * (h + y + z))
 *   y' = proj_Y(y + sigma * (2x' - x))
 *   z' = proj_Z(z + sigma * (2x' - x))
 *
 * The main loop runs the Halpern iteration
 *   v_{k+1} = (k+1)/(k+2) * (2 T(v_k) - v_k) + 1/(k+2) * v_anchor
 * checking convergence and the sufficient/necessary/artificial restart
 * conditions every op_steps applications, restarting at T(v_k) with a
 * primal-weight (omega) update smoothed in log space by theta.
 */
#ifndef PRIA_SOLVER_CUH
#define PRIA_SOLVER_CUH

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <vector>

#include <cub/cub.cuh>

#include "common.cuh"
#include "layouts.cuh"
#include "projection.cuh"
#include "pria/pria.h"

namespace pria {

/* One variable-triple slab: x, y, z stored contiguously (each ND long, x in
 * column layout, y in row layout, z in dim2 layout). */
template <typename real>
struct Slab {
    real* base;
    real* x;
    real* y;
    real* z;
    void assign(real* b, i64 nd) { base = b; x = b; y = b + nd; z = b + 2 * nd; }
};

/* ---- elementwise / gather kernels -------------------------------------- */

/* Halpern combine. The restart counter arrives in device memory (slot is the
 * position inside the current op_steps block) so the launch can be captured
 * in a CUDA graph and replayed for every block between restarts. */
template <typename real>
__global__ void k_combine(i64 n3, real* cur, const real* step,
                          const real* anchor, const i64* riter_base,
                          int slot) {
    const double r = (double)(*riter_base + slot);
    const real lw = (real)((r + 1.0) / (r + 2.0));
    const real rw = (real)(1.0 / (r + 2.0));
    i64 stride = (i64)gridDim.x * blockDim.x;
    for (i64 g = blockIdx.x * (i64)blockDim.x + threadIdx.x; g < n3; g += stride)
        cur[g] = lw * (real(2) * step[g] - cur[g]) + rw * anchor[g];
}

/* tau/sigma live in device memory (scal = {tau, sigma}) so a captured graph
 * picks up the omega updates made between replays. */
template <typename real>
__global__ void k_pre_x(Dims d, const real* cx, const real* cy, const real* cz,
                        const real* h, const real* scal, real* pre) {
    const real tau = scal[0];
    i64 stride = (i64)gridDim.x * blockDim.x;
    for (i64 g = blockIdx.x * (i64)blockDim.x + threadIdx.x; g < d.ND;
         g += stride) {
        i64 gr = colg_to_rowg(d, g), gz = colg_to_d2g(d, g);
        pre[g] = cx[g] - tau * (h[g] + cy[gr] + cz[gz]);
    }
}

/* Reflected dual update term, materialized once in the column layout so the
 * y and z pre-projection kernels do a single gathered read each:
 * r[g] = sigma * (2 x_new[g] - x_old[g]). */
template <typename real>
__global__ void k_reflect(Dims d, const real* xnew, const real* xold,
                          const real* scal, real* r) {
    const real sigma = scal[1];
    i64 stride = (i64)gridDim.x * blockDim.x;
    for (i64 g = blockIdx.x * (i64)blockDim.x + threadIdx.x; g < d.ND;
         g += stride)
        r[g] = sigma * (real(2) * xnew[g] - xold[g]);
}

/* pre_y[g] = y[g] + r gathered at the matching column position; k_pre_z is
 * identical with the dim2 index map. */
template <typename real>
__global__ void k_pre_y(Dims d, const real* cy, const real* r, real* pre) {
    i64 stride = (i64)gridDim.x * blockDim.x;
    for (i64 g = blockIdx.x * (i64)blockDim.x + threadIdx.x; g < d.ND;
         g += stride)
        pre[g] = cy[g] + r[rowg_to_colg(d, g)];
}

template <typename real>
__global__ void k_pre_z(Dims d, const real* cz, const real* r, real* pre) {
    i64 stride = (i64)gridDim.x * blockDim.x;
    for (i64 g = blockIdx.x * (i64)blockDim.x + threadIdx.x; g < d.ND;
         g += stride)
        pre[g] = cz[g] + r[d2g_to_colg(d, g)];
}

/* Reflect-inline variants: compute sigma*(2*xnew - xold) at the gathered
 * position instead of reading a materialized reflection (saves the
 * k_reflect round trip on the sort-free non-mirror path). */
template <typename real>
__global__ void k_pre_y_r(Dims d, const real* cy, const real* xnew,
                          const real* xold, const real* scal, real* pre) {
    const real sigma = scal[1];
    i64 stride = (i64)gridDim.x * blockDim.x;
    for (i64 g = blockIdx.x * (i64)blockDim.x + threadIdx.x; g < d.ND;
         g += stride) {
        i64 cg = rowg_to_colg(d, g);
        pre[g] = cy[g] + sigma * (real(2) * xnew[cg] - xold[cg]);
    }
}

template <typename real>
__global__ void k_pre_z_r(Dims d, const real* cz, const real* xnew,
                          const real* xold, const real* scal, real* pre) {
    const real sigma = scal[1];
    i64 stride = (i64)gridDim.x * blockDim.x;
    for (i64 g = blockIdx.x * (i64)blockDim.x + threadIdx.x; g < d.ND;
         g += stride) {
        i64 cg = d2g_to_colg(d, g);
        pre[g] = cz[g] + sigma * (real(2) * xnew[cg] - xold[cg]);
    }
}

/* ---- persistent megakernel (small instances) ---------------------------- *
 *
 * One Halpern iteration - combine, x projection, y and z projections with
 * the reflected term inline - as a single kernel with software grid-wide
 * barriers between the phases. A plain (non-cooperative) launch, so it
 * captures into the CUDA graphs like any other kernel; deadlock freedom
 * comes from sizing the grid to the kernel's measured residency
 * (setup_ctx, cudaOccupancyMaxActiveBlocksPerMultiprocessor).
 *
 * bar = {counter, generation} in device memory. The last block to arrive
 * resets the counter and bumps the generation; everyone else spins on the
 * generation. Fences order each phase's global writes before the bump and
 * after the observation, and generation increments survive graph replays
 * (the counter is self-cleaning). */
__device__ inline void dev_mega_sync(unsigned* bar, unsigned nblocks) {
    __syncthreads();
    if (threadIdx.x == 0) {
        volatile unsigned* gen = bar + 1;
        __threadfence(); /* release this block's phase writes */
        unsigned g = *gen;
        if (atomicAdd(bar, 1u) == nblocks - 1) {
            bar[0] = 0;
            __threadfence(); /* counter reset visible before the bump */
            atomicAdd((unsigned*)gen, 1u);
        } else {
            while (*gen == g) {}
            __threadfence(); /* acquire the other blocks' phase writes */
        }
    }
    __syncthreads();
}

/* One megakernel variant per precision, chosen empirically for the tiny
 * size class the auto gate admits; see the README for the measurements.
 *   f32 -> k_mega_block: the whole op_steps block in one launch, with z
 *          segments block-owned from global memory so shared holds just one
 *          row and occupancy is maxed.
 *   f64 -> k_mega_iter: one launch per iteration, z slices fused in shared.
 *          Its doubled global traffic makes shared-resident z rounds worth
 *          the occupancy they cost. */
template <typename real>
__global__ void k_mega_iter(Dims d, real* cur, real* step,
                            const real* anchor, const i64* riter_base,
                            int slot, const real* h, const real* scal,
                            const real* k, real k0, unsigned* bar) {
    extern __shared__ unsigned char smem[];
    real* srow = reinterpret_cast<real*>(smem); /* z-slice width reals */
    const unsigned nblocks = gridDim.x;
    const i64 nd = d.ND, n3 = 3 * nd;
    real* cx = cur;
    real* cy = cur + nd;
    real* cz = cur + 2 * nd;
    real* sx = step;
    real* sy = step + nd;
    real* sz = step + 2 * nd;

    /* phase A: Halpern combine (same math as k_combine) */
    {
        const double r = (double)(*riter_base + slot);
        const real lw = (real)((r + 1.0) / (r + 2.0));
        const real rw = (real)(1.0 / (r + 2.0));
        i64 stride = (i64)gridDim.x * blockDim.x;
        for (i64 g = blockIdx.x * (i64)blockDim.x + threadIdx.x; g < n3;
             g += stride)
            cur[g] = lw * (real(2) * step[g] - cur[g]) + rw * anchor[g];
    }
    dev_mega_sync(bar, nblocks);

    /* phase B: x pre+projection per dense row */
    const i64 rowsXY = d.N * d.P;
    for (i64 row = blockIdx.x; row < rowsXY; row += gridDim.x)
        dev_fused_pre_row(row, d.W, d, 0, 1, (const real*)cx, h,
                          (const real*)cy, (const real*)cz, scal, k, k0, sx,
                          srow);
    dev_mega_sync(bar, nblocks);

    /* phase C: y rows and z slices, reflected term inline from (sx, cx) */
    for (i64 w = blockIdx.x; w < rowsXY + d.P; w += gridDim.x) {
        if (w < rowsXY)
            dev_fused_pre_row(w, d.W, d, 1, 2, (const real*)cy,
                              (const real*)sx, (const real*)cx,
                              (const real*)nullptr, scal, k, k0, sy, srow);
        else
            dev_fused_pre_row(w - rowsXY, d.W * d.N, d, 2, 2,
                              (const real*)cz, (const real*)sx,
                              (const real*)cx, (const real*)nullptr, scal, k,
                              k0, sz, srow);
    }
}

template <typename real>
__global__ void k_mega_block(Dims d, real* cur, real* step,
                             const real* anchor, const i64* riter_base,
                             int op_steps, const real* h, const real* scal,
                             const real* k, real k0, const int* offZ,
                             real* warmZ, real* prez, unsigned* bar) {
    extern __shared__ unsigned char smem[];
    real* srow = reinterpret_cast<real*>(smem); /* one x/y row: W reals */
    const unsigned nblocks = gridDim.x;
    const i64 nd = d.ND, n3 = 3 * nd;
    const i64 rowsXY = d.N * d.P;
    const i64 nsegZ = 2 * d.P;
    real* cx = cur;
    real* cy = cur + nd;
    real* cz = cur + 2 * nd;
    real* sx = step;
    real* sy = step + nd;
    real* sz = step + 2 * nd;

    for (int s2 = 0; s2 < op_steps; ++s2) {
        /* phase A: Halpern combine (same math as k_combine) */
        {
            const double r = (double)(*riter_base + s2);
            const real lw = (real)((r + 1.0) / (r + 2.0));
            const real rw = (real)(1.0 / (r + 2.0));
            i64 stride = (i64)gridDim.x * blockDim.x;
            for (i64 g = blockIdx.x * (i64)blockDim.x + threadIdx.x; g < n3;
                 g += stride)
                cur[g] = lw * (real(2) * step[g] - cur[g]) + rw * anchor[g];
        }
        dev_mega_sync(bar, nblocks);

        /* phase B: x pre+projection per dense row */
        for (i64 row = blockIdx.x; row < rowsXY; row += gridDim.x)
            dev_fused_pre_row(row, d.W, d, 0, 1, (const real*)cx, h,
                              (const real*)cy, (const real*)cz, scal, k, k0,
                              sx, srow);
        dev_mega_sync(bar, nblocks);

        /* phase C: z segments first (the long pole), then y rows; the
         * reflected term is inline from (sx, cx) in both. Each z segment
         * is block-owned end to end: its pre values are materialized once
         * into prez (block-local, so no extra barrier), then Michelot -
         * warm-seeded like the launched seg kernel - re-reads them from
         * cache and the clamp writes sz. */
        for (i64 w = blockIdx.x; w < nsegZ + rowsXY; w += gridDim.x) {
            if (w < nsegZ) {
                const i64 s0 = offZ[w], s1 = offZ[w + 1];
                const real sigma = scal[1];
                for (i64 g = s0 + threadIdx.x; g < s1; g += kBlock) {
                    i64 cg = d2g_to_colg(d, g);
                    prez[g] = cz[g] + sigma * (real(2) * sx[cg] - cx[cg]);
                }
                __syncthreads();
                real theta0 = -std::numeric_limits<real>::infinity();
                if (warmZ) {
                    real ww = warmZ[w];
                    if (isfinite(ww)) theta0 = ww - fabs(ww) * real(0.01);
                }
                const real theta =
                    block_michelot((const real*)(prez + s0), s1 - s0, k0,
                                   theta0, nullptr, nullptr);
                if (warmZ && threadIdx.x == 0) warmZ[w] = theta;
                for (i64 g = s0 + threadIdx.x; g < s1; g += kBlock) {
                    real o = prez[g] - theta;
                    sz[g] = o > real(0) ? o : real(0);
                }
                __syncthreads(); /* prez region reuse next iteration */
            } else {
                dev_fused_pre_row(w - nsegZ, d.W, d, 1, 2, (const real*)cy,
                                  (const real*)sx, (const real*)cx,
                                  (const real*)nullptr, scal, k, k0, sy,
                                  srow);
            }
        }
        dev_mega_sync(bar, nblocks);
    }
}

/* ---- tiled layout mirrors ---------------------------------------------- *
 *
 * Materialize one dense layout gathered into another, staging 32x32 tiles
 * through shared memory so both sides stream coalesced. The direct gathers
 * are transpose-patterned (stride W or stride N), which overfetches badly
 * once the source arrays outgrow the L2; the mirrors trade one extra
 * coalesced pass for that. Values are copied bitwise, so solves with and
 * without mirrors are identical.
 */

/* Row layout <-> column layout, one (s, t) tile per block per item.
 * kToCol = true: src row layout -> dst column layout; false: reverse. */
template <typename real, bool kToCol>
__global__ void k_mirror_rowcol(Dims d, const real* __restrict__ src,
                                real* __restrict__ dst) {
    __shared__ real tile[32][33];
    const i64 t0 = (i64)blockIdx.x * 32;
    const i64 s0 = (i64)blockIdx.y * 32;
    if (s0 > t0 + 31) return; /* tile entirely below the diagonal */
    const real* sp = src + (i64)blockIdx.z * d.D;
    real* dp = dst + (i64)blockIdx.z * d.D;
    i64 j, cc;

    if (kToCol) {
        for (int r = 0; r < 32; r += 8) { /* x along t: row-coalesced */
            i64 s = s0 + threadIdx.y + r, t = t0 + threadIdx.x;
            if (t < d.T && s <= t) {
                st_to_row(d.T, s, t, j, cc);
                tile[threadIdx.y + r][threadIdx.x] = sp[j * d.W + cc];
            }
        }
        __syncthreads();
        for (int r = 0; r < 32; r += 8) { /* x along s: col-coalesced */
            i64 s = s0 + threadIdx.x, t = t0 + threadIdx.y + r;
            if (t < d.T && s <= t) {
                st_to_col(d.T, s, t, j, cc);
                dp[j * d.W + cc] = tile[threadIdx.x][threadIdx.y + r];
            }
        }
    } else {
        for (int r = 0; r < 32; r += 8) { /* x along s: col-coalesced */
            i64 s = s0 + threadIdx.x, t = t0 + threadIdx.y + r;
            if (t < d.T && s <= t) {
                st_to_col(d.T, s, t, j, cc);
                tile[threadIdx.x][threadIdx.y + r] = sp[j * d.W + cc];
            }
        }
        __syncthreads();
        for (int r = 0; r < 32; r += 8) { /* x along t: row-coalesced */
            i64 s = s0 + threadIdx.y + r, t = t0 + threadIdx.x;
            if (t < d.T && s <= t) {
                st_to_row(d.T, s, t, j, cc);
                dp[j * d.W + cc] = tile[threadIdx.y + r][threadIdx.x];
            }
        }
    }
}

/* Dim2 layout <-> column layout: the transpose plane is (item, s) at fixed
 * t (dim2 is item-innermost, the column layout item-outermost).
 * kToCol = true: src dim2 -> dst column; false: reverse. */
template <typename real, bool kToCol>
__global__ void k_mirror_d2col(Dims d, const real* __restrict__ src,
                               real* __restrict__ dst) {
    __shared__ real tile[32][33];
    const i64 t = blockIdx.z;
    const i64 s0 = (i64)blockIdx.y * 32;
    const i64 i0 = (i64)blockIdx.x * 32;
    if (s0 > t) return;
    i64 j, cc;

    if (kToCol) {
        for (int r = 0; r < 32; r += 8) { /* x along i: dim2-coalesced */
            i64 i = i0 + threadIdx.x, s = s0 + threadIdx.y + r;
            if (i < d.N && s <= t) {
                st_to_row(d.T, s, t, j, cc);
                tile[threadIdx.y + r][threadIdx.x] = src[(j * d.W + cc) * d.N + i];
            }
        }
        __syncthreads();
        for (int r = 0; r < 32; r += 8) { /* x along s: col-coalesced */
            i64 i = i0 + threadIdx.y + r, s = s0 + threadIdx.x;
            if (i < d.N && s <= t) {
                st_to_col(d.T, s, t, j, cc);
                dst[i * d.D + j * d.W + cc] = tile[threadIdx.x][threadIdx.y + r];
            }
        }
    } else {
        for (int r = 0; r < 32; r += 8) { /* x along s: col-coalesced */
            i64 i = i0 + threadIdx.y + r, s = s0 + threadIdx.x;
            if (i < d.N && s <= t) {
                st_to_col(d.T, s, t, j, cc);
                tile[threadIdx.y + r][threadIdx.x] = src[i * d.D + j * d.W + cc];
            }
        }
        __syncthreads();
        for (int r = 0; r < 32; r += 8) { /* x along i: dim2-coalesced */
            i64 i = i0 + threadIdx.x, s = s0 + threadIdx.y + r;
            if (i < d.N && s <= t) {
                st_to_row(d.T, s, t, j, cc);
                dst[(j * d.W + cc) * d.N + i] = tile[threadIdx.x][threadIdx.y + r];
            }
        }
    }
}

template <typename real, bool kToCol>
void launch_mirror_rowcol(const Dims& d, const real* src, real* dst,
                          cudaStream_t stream) {
    dim3 grid((unsigned)((d.T + 31) / 32), (unsigned)((d.T + 31) / 32),
              (unsigned)d.N);
    k_mirror_rowcol<real, kToCol><<<grid, dim3(32, 8), 0, stream>>>(d, src,
                                                                    dst);
}

template <typename real, bool kToCol>
void launch_mirror_d2col(const Dims& d, const real* src, real* dst,
                         cudaStream_t stream) {
    dim3 grid((unsigned)((d.N + 31) / 32), (unsigned)((d.T + 31) / 32),
              (unsigned)d.T);
    k_mirror_d2col<real, kToCol><<<grid, dim3(32, 8), 0, stream>>>(d, src,
                                                                   dst);
}

/* pre_x from mirrored (column-layout) duals: pure streaming. */
template <typename real>
__global__ void k_pre_x_m(Dims d, const real* cx, const real* my,
                          const real* mz, const real* h, const real* scal,
                          real* pre) {
    const real tau = scal[0];
    i64 stride = (i64)gridDim.x * blockDim.x;
    for (i64 g = blockIdx.x * (i64)blockDim.x + threadIdx.x; g < d.ND;
         g += stride)
        pre[g] = cx[g] - tau * (h[g] + my[g] + mz[g]);
}

/* pre_y / pre_z from a mirrored reflected term: pure streaming. */
template <typename real>
__global__ void k_add_pre(i64 nd, const real* a, const real* b, real* pre) {
    i64 stride = (i64)gridDim.x * blockDim.x;
    for (i64 g = blockIdx.x * (i64)blockDim.x + threadIdx.x; g < nd;
         g += stride)
        pre[g] = a[g] + b[g];
}

/* Gap gathers: x viewed through the y/z layouts (scaled by K), and the dual
 * objective's h + y + z viewed through the x layout. */
template <typename real>
__global__ void k_gather_row_kx(Dims d, const real* X, const real* k,
                                real* out) {
    i64 stride = (i64)gridDim.x * blockDim.x;
    for (i64 g = blockIdx.x * (i64)blockDim.x + threadIdx.x; g < d.ND;
         g += stride)
        out[g] = k[g / d.D] * X[rowg_to_colg(d, g)];
}

template <typename real>
__global__ void k_gather_d2_k0x(Dims d, const real* X, real k0, real* out) {
    i64 stride = (i64)gridDim.x * blockDim.x;
    for (i64 g = blockIdx.x * (i64)blockDim.x + threadIdx.x; g < d.ND;
         g += stride)
        out[g] = k0 * X[d2g_to_colg(d, g)];
}

template <typename real>
__global__ void k_gather_lhs(Dims d, const real* Y, const real* Z,
                             const real* h, real* out) {
    i64 stride = (i64)gridDim.x * blockDim.x;
    for (i64 g = blockIdx.x * (i64)blockDim.x + threadIdx.x; g < d.ND;
         g += stride)
        out[g] = h[g] + Y[colg_to_rowg(d, g)] + Z[colg_to_d2g(d, g)];
}

/* ---- init / finalize gathers ------------------------------------------- */

/* Sparsify to a full [N][Torig][Torig] upper-triangular array. mode: 0 = x
 * (column layout), 1 = y (row layout, scaled by invk[i]), 2 = z (dim2 layout,
 * scaled by invk0). */
template <typename real, int MODE>
__global__ void k_sparsify(Dims d, i64 torig, const real* dense,
                           const real* invk, real invk0, real* full) {
    i64 total = d.N * torig * torig;
    i64 stride = (i64)gridDim.x * blockDim.x;
    for (i64 f = blockIdx.x * (i64)blockDim.x + threadIdx.x; f < total;
         f += stride) {
        i64 i = f / (torig * torig), r = f - i * torig * torig;
        i64 s = r / torig, t = r - s * torig;
        real v = real(0);
        if (s <= t) {
            i64 j, c;
            if (MODE == 0) {
                st_to_col(d.T, s, t, j, c);
                v = dense[(i * d.P + j) * d.W + c];
            } else if (MODE == 1) {
                st_to_row(d.T, s, t, j, c);
                v = dense[(i * d.P + j) * d.W + c] * invk[i];
            } else {
                st_to_row(d.T, s, t, j, c);
                v = dense[(j * d.W + c) * d.N + i] * invk0;
            }
        }
        full[f] = v;
    }
}

/* ---- reduction functors ------------------------------------------------ */

/* Canonical-norm pieces of (cur - step): acc = {|dx|^2, |dy|^2 + |dz|^2,
 * <dx, dy + dz>} with the duals gathered into the column layout. */
template <typename real>
struct ErrFunctor {
    Dims d;
    const real *cx, *cy, *cz, *sx, *sy, *sz;
    __device__ void operator()(i64 g, double* acc) const {
        double dx = (double)cx[g] - (double)sx[g];
        i64 gr = colg_to_rowg(d, g);
        double dy = (double)cy[gr] - (double)sy[gr];
        i64 gz = colg_to_d2g(d, g);
        double dz = (double)cz[gz] - (double)sz[gz];
        acc[0] += dx * dx;
        acc[1] += dy * dy + dz * dz;
        acc[2] += dx * (dy + dz);
    }
};

template <typename real>
struct DotFunctor {
    const real *a, *b;
    __device__ void operator()(i64 g, double* acc) const {
        acc[0] += (double)a[g] * (double)b[g];
    }
};

template <typename real>
struct SumFunctor {
    const real* a;
    __device__ void operator()(i64 g, double* acc) const {
        acc[0] += (double)a[g];
    }
};

/* Squared distance to the anchor over the whole slab, split into the x part
 * (acc[0]) and the y+z part (acc[1]). */
template <typename real>
struct AnchorFunctor {
    const real *cur, *anchor;
    i64 nd;
    __device__ void operator()(i64 g, double* acc) const {
        double dv = (double)cur[g] - (double)anchor[g];
        acc[g < nd ? 0 : 1] += dv * dv;
    }
};

/* ---- CUDA-graph capture of the op_steps blocks ------------------------- *
 *
 * Everything inside an op_steps block (Halpern combines, operator steps and
 * the convergence reductions) is stream-capturable, and every scalar that
 * changes between blocks travels through a pinned host mirror re-read by a
 * memcpy node at each replay. The cur/step buffer roles swap at restarts, so
 * each block keeps one instantiated graph per parity, keyed by the cur base
 * pointer. */
struct BlockGraphs {
    cudaGraphExec_t exec[2] = {nullptr, nullptr};
    const void* key[2] = {nullptr, nullptr};
    void destroy() {
        for (int i = 0; i < 2; ++i)
            if (exec[i]) cudaGraphExecDestroy(exec[i]);
        *this = BlockGraphs{};
    }
};

/* Set by tests (or the PRIA_NO_GRAPHS environment variable) to run every
 * block as plain stream launches. */
inline bool& graphs_disabled() {
    static bool disabled = std::getenv("PRIA_NO_GRAPHS") != nullptr;
    return disabled;
}

/* Set by tests (or PRIA_NO_COND) to keep the loop host-driven with per-block
 * graphs instead of the whole-solve conditional graph. */
inline bool& cond_disabled() {
    static bool disabled = std::getenv("PRIA_NO_COND") != nullptr;
    return disabled;
}

/* y/z fork-join overlap. Auto (-1) turns it on only where the z projection
 * takes the device-wide path and there is real work to overlap with the
 * fused y pipeline; with everything fused the fork/join events cost more
 * than the kernels save. PRIA_OVERLAP=1 forces on within the safety gates,
 * =0 or PRIA_NO_OVERLAP forces off. */
inline int& overlap_mode() {
    static int mode = [] {
        if (std::getenv("PRIA_NO_OVERLAP")) return 0;
        const char* e = std::getenv("PRIA_OVERLAP");
        return e ? std::atoi(e) : -1;
    }();
    return mode;
}

/* PRIA_SF_RFUSE=0 materializes the reflected term with k_reflect instead
 * of computing it inline in the y/z consumers (sort-free non-mirror path
 * only; the mirror path always materializes for its transpose tiles). */
inline int& rfuse_mode() {
    static int mode = [] {
        const char* e = std::getenv("PRIA_SF_RFUSE");
        return e ? std::atoi(e) : 1;
    }();
    return mode;
}

/* Persistent megakernel for the Halpern inner iteration. PRIA_MEGA=1
 * forces it on (where structurally possible), =0 off; unset (-1) enables
 * it by the auto size gate in setup_ctx. */
inline int& mega_mode() {
    static int mode = [] {
        const char* e = std::getenv("PRIA_MEGA");
        return e ? std::atoi(e) : -1;
    }();
    return mode;
}

/* Set by solve_batch workers. Concurrent instances must not run persistent
 * full-GPU grids against each other, so mega is structurally off in batch
 * solves and batch results match solo solves made with PRIA_MEGA=0. */
inline bool& batch_worker() {
    static thread_local bool b = false;
    return b;
}

/* PRIA_ZSORT_PROBE=1 instruments the device-wide z projection with the
 * sort-order-stability probe (results unchanged; stats printed at solve
 * teardown). */
inline bool zsort_probe_enabled() {
    static bool on = std::getenv("PRIA_ZSORT_PROBE") != nullptr;
    return on;
}

/* PRIA_MIRRORS=1 forces the tiled-transpose mirrors on, =0 off; unset (-1)
 * enables them once the gather sources outgrow the L2 working set. */
inline int& mirrors_mode() {
    static int mode = [] {
        const char* e = std::getenv("PRIA_MIRRORS");
        return e ? std::atoi(e) : -1;
    }();
    return mode;
}

/* ---- solver context ---------------------------------------------------- */

template <typename real>
struct Ctx {
    Dims d;
    i64 torig = 0;
    double scale = 1.0;
    double k0s = 0.0; /* scaled joint cost */

    Slab<real> cur, step, anchor, pre;
    real* h = nullptr;
    real* kdev = nullptr;
    real* invk = nullptr;
    int *offX = nullptr, *offY = nullptr, *offZ = nullptr;
    /* per-segment warm thresholds for the sort-free projections */
    real *warmX = nullptr, *warmY = nullptr, *warmZ = nullptr;
    ProjWorkspace<real> ws;
    real* segred = nullptr;
    double* dred = nullptr;  /* device scalars, see slot map below */
    double* dpart = nullptr; /* reduction partials */
    double* hred = nullptr;  /* pinned host mirror of dred */
    real* outfull = nullptr; /* N*Torig*Torig staging for results */
    real* dscal = nullptr;   /* device {tau, sigma} */
    real* hscal = nullptr;   /* pinned mirror, rewritten when omega changes */
    i64* driter = nullptr;   /* device restart-counter base for k_combine */
    i64* hriter = nullptr;   /* pinned mirror, rewritten before each block */
    BlockGraphs g_halpern, g_vanilla;
    cudaGraphExec_t whole_exec = nullptr; /* whole-solve conditional graph */
    bool graphs_ok = true;
    cudaStream_t stream = nullptr;
    cudaStream_t stream2 = nullptr; /* forked z-projection branch */
    cudaEvent_t ev_fork = nullptr, ev_join = nullptr;
    bool overlap = false;      /* run y/z pipelines on forked streams */
    bool use_mirrors = false;  /* tiled-transpose mirrors for the gathers */
    bool rfuse = false;        /* reflect inline in the y/z consumers */
    bool mega = false;         /* persistent megakernel inner iteration */
    bool mega_block = false;   /* f32: k_mega_block; f64: k_mega_iter */
    int mega_grid = 0;         /* resident grid size for the mega kernel */
    unsigned* dbar = nullptr;  /* megakernel barrier {counter, generation} */
    real* mir1 = nullptr;      /* ND; y-in-col / refl-in-row mirror */
    real* mir2 = nullptr;      /* ND; z-in-col / refl-in-dim2 mirror */

    GeomX gx() const { return GeomX{d}; }
    GeomY gy() const { return GeomY{d}; }
    GeomZ gz() const { return GeomZ{d}; }
};

/* dred slot map: reduction results, then the device-resident solver state.
 * The state slots are written only by the k_*_state/logic kernels below so
 * that every execution mode (plain launches, block graphs, whole-solve
 * conditional graph) runs the identical arithmetic. */
enum {
    kErr0 = 0,   /* |dx|^2 */
    kErr1 = 1,   /* |dy|^2 + |dz|^2 */
    kErr2 = 2,   /* <dx, dy+dz> */
    kTerm1 = 3,  /* sum of K_i-weighted row maxes of x */
    kTerm2 = 4,  /* sum of K0-weighted period maxes of x */
    kTerm3 = 5,  /* <x, h> */
    kDualG = 6,  /* sum of column mins of h + y + z */
    kAnch0 = 7,  /* |x - x0|^2 */
    kAnch1 = 8,  /* |y - y0|^2 + |z - z0|^2 */
    kNumRed = 9,
    kStOmega = 9,     /* current primal weight */
    kStTheta = 10,    /* current omega-smoothing weight */
    kStFpErr = 11,    /* canonical error at the last restart */
    kStCheckErr = 12, /* canonical error at the last check */
    kStNewErr = 13,   /* canonical error at this check (for verbose) */
    kStPrimal = 14,   /* implied primal objective (scaled back) */
    kStDual = 15,     /* implied dual objective */
    kStGap = 16,      /* relative gap */
    kNumScalar = 17
};

/* driter slot map (i64 device state + decision flags) */
enum {
    kItRestart = 0,       /* iterations since the last restart */
    kItIter = 1,          /* total iterations */
    kItTerminated = 2,    /* decision flags... */
    kItShouldRestart = 3,
    kItSuff = 4,
    kItNec = 5,
    kItArt = 6,
    kNumIter = 7
};

/* ---- device-side solver logic ------------------------------------------ */

__device__ inline double dev_canonical_err(const double* sc, double eta,
                                           double omega) {
    /* negative cross term: the proximal-point metric (off-diagonal -I) in
     * which one PDHG step is firmly nonexpansive; matches the JAX
     * canonical_norm and the convergence analysis. */
    double v = (omega / eta) * sc[kErr0] + (1.0 / (eta * omega)) * sc[kErr1] -
               2.0 * sc[kErr2];
    return sqrt(v > 0.0 ? v : 0.0);
}

__device__ inline void dev_gap(const double* sc, double scale, double* primal,
                               double* dual, double* gap) {
    *primal = (sc[kTerm1] + sc[kTerm2] + sc[kTerm3]) * scale;
    *dual = sc[kDualG] * scale;
    *gap = (*primal - *dual) / (1.0 + fabs(*dual) + fabs(*primal));
}

template <typename real>
__global__ void k_set_omega(double* sc, real* scal, double omega, double eta) {
    sc[kStOmega] = omega;
    scal[0] = (real)(eta / omega);
    scal[1] = (real)(eta * omega);
}

/* omega re-estimate from the warmup displacement (dred holds the anchor
 * distances); falls back to the configured omega on 0/0. */
template <typename real>
__global__ void k_warm_omega(double* sc, real* scal, double fallback,
                             double eta) {
    double om = sqrt(sc[kAnch1]) / sqrt(sc[kAnch0]);
    if (isnan(om)) om = fallback;
    sc[kStOmega] = om;
    scal[0] = (real)(eta / om);
    scal[1] = (real)(eta * om);
}

/* Pre-loop state init: fp_err from the enqueued err reduction, gap from the
 * enqueued gap reductions, initial termination decision. */
static __global__ void k_init_state(double* sc, i64* it, double eta, double scale,
                             double tol, double theta_initial, i64 max_iter,
                             i64 iteration) {
    it[kItIter] = iteration;
    it[kItRestart] = 0;
    sc[kStTheta] = theta_initial;
    sc[kStFpErr] = dev_canonical_err(sc, eta, sc[kStOmega]);
    sc[kStNewErr] = sc[kStFpErr];
    sc[kStCheckErr] = std::numeric_limits<double>::infinity();
    double primal, dual, gap;
    dev_gap(sc, scale, &primal, &dual, &gap);
    sc[kStPrimal] = primal;
    sc[kStDual] = dual;
    sc[kStGap] = gap;
    it[kItTerminated] =
        (!(gap > tol) || (max_iter >= 0 && iteration >= max_iter)) ? 1 : 0;
    it[kItShouldRestart] = 0;
}

/* Per-block convergence/restart decisions, exactly the host logic it
 * replaces. When use_handles is set (whole-solve conditional graph), the
 * decisions also drive the WHILE (continue) and IF (restart) handles. */
static __global__ void k_halpern_logic(double* sc, i64* it, double eta, double scale,
                                double tol, double bsuf, double bnec,
                                double bart, i64 op_steps, i64 max_iter,
                                cudaGraphConditionalHandle h_while,
                                cudaGraphConditionalHandle h_restart,
                                int use_handles) {
    it[kItRestart] += op_steps;
    it[kItIter] += op_steps;
    double new_err = dev_canonical_err(sc, eta, sc[kStOmega]);
    double primal, dual, gap;
    dev_gap(sc, scale, &primal, &dual, &gap);
    sc[kStNewErr] = new_err;
    sc[kStPrimal] = primal;
    sc[kStDual] = dual;
    sc[kStGap] = gap;
    bool terminated =
        !(gap > tol) || (max_iter >= 0 && it[kItIter] >= max_iter);
    bool sufficient = new_err <= bsuf * sc[kStFpErr];
    bool necessary =
        (new_err <= bnec * sc[kStFpErr]) && (new_err > sc[kStCheckErr]);
    bool artificial = (double)it[kItRestart] >= bart * (double)it[kItIter];
    bool restart = (sufficient || necessary || artificial) && !terminated;
    it[kItTerminated] = terminated ? 1 : 0;
    it[kItShouldRestart] = restart ? 1 : 0;
    it[kItSuff] = sufficient ? 1 : 0;
    it[kItNec] = necessary ? 1 : 0;
    it[kItArt] = artificial ? 1 : 0;
    if (!terminated && !restart) sc[kStCheckErr] = new_err;
    if (use_handles) {
        cudaGraphSetConditional(h_while, terminated ? 0 : 1);
        cudaGraphSetConditional(h_restart, restart ? 1 : 0);
    }
}

/* Restart omega/theta update from the anchor distances. */
template <typename real>
__global__ void k_restart_update(double* sc, real* scal, i64* it, double eta,
                                 double theta_target, double theta_decay) {
    double pn = sqrt(sc[kAnch0]), dn = sqrt(sc[kAnch1]);
    double omega = sc[kStOmega], theta = sc[kStTheta];
    if (pn > 1e-16 && dn > 1e-16) {
        omega = exp(theta * (log(dn) - log(pn)) + (1.0 - theta) * log(omega));
        sc[kStOmega] = omega;
        sc[kStTheta] = theta_target + theta_decay * (theta - theta_target);
        scal[0] = (real)(eta / omega);
        scal[1] = (real)(eta * omega);
    }
    it[kItRestart] = 0;
    sc[kStCheckErr] = std::numeric_limits<double>::infinity();
}

static __global__ void k_post_restart(double* sc, double eta) {
    sc[kStFpErr] = dev_canonical_err(sc, eta, sc[kStOmega]);
}

/* Run one op_steps block through its per-parity instantiated graph,
 * capturing it on first use; falls back to plain launches if capture is
 * disabled or fails. enqueue must only enqueue stream work (no syncs, no
 * host-state mutation that a replay would need to repeat). */
template <typename real, typename F>
cudaError_t run_block_graph(Ctx<real>& c, BlockGraphs& bg, const void* key,
                            F&& enqueue) {
    if (!c.graphs_ok) return enqueue();
    int idx = key == bg.key[0] ? 0 : key == bg.key[1] ? 1 : -1;
    if (idx < 0) {
        idx = bg.key[0] ? 1 : 0;
        if (bg.key[idx] ||
            cudaStreamBeginCapture(c.stream, cudaStreamCaptureModeThreadLocal)
                != cudaSuccess) {
            cudaGetLastError();
            c.graphs_ok = false;
            return enqueue();
        }
        cudaError_t enq = enqueue();
        cudaGraph_t graph = nullptr;
        cudaError_t cap = cudaStreamEndCapture(c.stream, &graph);
        if (enq == cudaSuccess && cap == cudaSuccess)
            cap = cudaGraphInstantiate(&bg.exec[idx], graph, 0);
        if (graph) cudaGraphDestroy(graph);
        if (enq != cudaSuccess || cap != cudaSuccess || !bg.exec[idx]) {
            bg.exec[idx] = nullptr;
            cudaGetLastError();
            c.graphs_ok = false;
            return enqueue(); /* this time for real (capture ran nothing) */
        }
        bg.key[idx] = key;
    }
    return cudaGraphLaunch(bg.exec[idx], c.stream);
}

/* One PDHG operator application. tau/sigma live in device memory (dscal),
 * maintained by k_set_omega / k_warm_omega / k_restart_update, so the
 * launches are graph-replayable with no host involvement. */
template <typename real>
cudaError_t raw_step(Ctx<real>& c, const Slab<real>& in, Slab<real>& out) {
    const int grid = launch_grid(c.d.ND);

    if (c.use_mirrors) {
        launch_mirror_rowcol<real, true>(c.d, in.y, c.mir1, c.stream);
        launch_mirror_d2col<real, true>(c.d, in.z, c.mir2, c.stream);
    }
    const real* xs_y = c.use_mirrors ? (const real*)c.mir1 : (const real*)in.y;
    const real* xs_z = c.use_mirrors ? (const real*)c.mir2 : (const real*)in.z;
    if (!try_fused_pre(c.d, 0, c.use_mirrors ? 0 : 1, (const real*)in.x, c.h,
                       xs_y, xs_z, (const real*)c.dscal, c.kdev, (real)c.k0s,
                       out.x, c.stream)) {
        if (c.use_mirrors)
            k_pre_x_m<<<grid, kBlock, 0, c.stream>>>(
                c.d, in.x, (const real*)c.mir1, (const real*)c.mir2, c.h,
                (const real*)c.dscal, c.pre.x);
        else
            k_pre_x<<<grid, kBlock, 0, c.stream>>>(
                c.d, in.x, in.y, in.z, c.h, (const real*)c.dscal, c.pre.x);
        PRIA_CHECK(project_segments(c.ws, (const real*)c.pre.x, out.x, c.gx(),
                                    c.offX, c.warmX, c.kdev, (real)c.k0s,
                                    c.stream));
    }
    PRIA_CHECK(cudaPeekAtLastError());

    const bool rfuse = c.rfuse;

    /* pre.x has been consumed by the x projection; reuse it for the
     * reflected term */
    real* refl = c.pre.x;
    if (!rfuse)
        k_reflect<<<grid, kBlock, 0, c.stream>>>(
            c.d, (const real*)out.x, in.x, (const real*)c.dscal, refl);

    auto enq_y = [&](cudaStream_t st) -> cudaError_t {
        if (rfuse) {
            if (try_fused_pre(c.d, 1, 2, (const real*)in.y,
                              (const real*)out.x, (const real*)in.x,
                              (const real*)nullptr, (const real*)c.dscal,
                              c.kdev, (real)c.k0s, out.y, st))
                return cudaPeekAtLastError();
            k_pre_y_r<<<grid, kBlock, 0, st>>>(c.d, in.y,
                                               (const real*)out.x, in.x,
                                               (const real*)c.dscal, c.pre.y);
            return project_segments(c.ws, (const real*)c.pre.y, out.y,
                                    c.gy(), c.offY, c.warmY, c.kdev,
                                    (real)c.k0s, st);
        }
        const real* ysrc = (const real*)refl;
        int gath = 1;
        if (c.use_mirrors) {
            launch_mirror_rowcol<real, false>(c.d, (const real*)refl, c.mir1,
                                              st);
            ysrc = (const real*)c.mir1;
            gath = 0;
        }
        if (try_fused_pre(c.d, 1, gath, (const real*)in.y, ysrc,
                          (const real*)nullptr, (const real*)nullptr,
                          (const real*)c.dscal, c.kdev, (real)c.k0s, out.y,
                          st))
            return cudaPeekAtLastError();
        if (c.use_mirrors)
            k_add_pre<<<grid, kBlock, 0, st>>>(c.d.ND, in.y,
                                               (const real*)c.mir1, c.pre.y);
        else
            k_pre_y<<<grid, kBlock, 0, st>>>(c.d, in.y, (const real*)refl,
                                             c.pre.y);
        return project_segments(c.ws, (const real*)c.pre.y, out.y, c.gy(),
                                c.offY, c.warmY, c.kdev, (real)c.k0s, st);
    };
    auto enq_z = [&](cudaStream_t st) -> cudaError_t {
        if (rfuse) {
            if (try_fused_pre(c.d, 2, 2, (const real*)in.z,
                              (const real*)out.x, (const real*)in.x,
                              (const real*)nullptr, (const real*)c.dscal,
                              c.kdev, (real)c.k0s, out.z, st))
                return cudaPeekAtLastError();
            k_pre_z_r<<<grid, kBlock, 0, st>>>(c.d, in.z,
                                               (const real*)out.x, in.x,
                                               (const real*)c.dscal, c.pre.z);
            return project_segments(c.ws, (const real*)c.pre.z, out.z,
                                    c.gz(), c.offZ, c.warmZ, c.kdev,
                                    (real)c.k0s, st);
        }
        const real* zsrc = (const real*)refl;
        int gath = 1;
        if (c.use_mirrors) {
            launch_mirror_d2col<real, false>(c.d, (const real*)refl, c.mir2,
                                             st);
            zsrc = (const real*)c.mir2;
            gath = 0;
        }
        if (try_fused_pre(c.d, 2, gath, (const real*)in.z, zsrc,
                          (const real*)nullptr, (const real*)nullptr,
                          (const real*)c.dscal, c.kdev, (real)c.k0s, out.z,
                          st))
            return cudaPeekAtLastError();
        if (c.use_mirrors)
            k_add_pre<<<grid, kBlock, 0, st>>>(c.d.ND, in.z,
                                               (const real*)c.mir2, c.pre.z);
        else
            k_pre_z<<<grid, kBlock, 0, st>>>(c.d, in.z, (const real*)refl,
                                             c.pre.z);
        return project_segments(c.ws, (const real*)c.pre.z, out.z, c.gz(),
                                c.offZ, c.warmZ, c.kdev, (real)c.k0s, st);
    };

    if (c.overlap) {
        /* The y and z pipelines are independent given refl; run z on a
         * forked stream. Safe because the (guaranteed) fused y path touches
         * none of the projection scratch the z path may use, mir1/mir2 are
         * branch-private, and no result depends on execution order. Under
         * stream capture the fork becomes genuine graph parallelism. */
        PRIA_CHECK(cudaEventRecord(c.ev_fork, c.stream));
        PRIA_CHECK(cudaStreamWaitEvent(c.stream2, c.ev_fork, 0));
        PRIA_CHECK(enq_y(c.stream));
        PRIA_CHECK(enq_z(c.stream2));
        PRIA_CHECK(cudaEventRecord(c.ev_join, c.stream2));
        PRIA_CHECK(cudaStreamWaitEvent(c.stream, c.ev_join, 0));
    } else {
        PRIA_CHECK(enq_y(c.stream));
        PRIA_CHECK(enq_z(c.stream));
    }
    return cudaPeekAtLastError();
}

/* Test/setup convenience: set dscal from (omega, eta) through the pinned
 * mirror, then run the step. Callers sync between omega changes, so a
 * differently-valued copy is never still in flight when the mirror is
 * rewritten. */
template <typename real>
cudaError_t raw_step(Ctx<real>& c, const Slab<real>& in, Slab<real>& out,
                     double omega, double eta) {
    c.hscal[0] = (real)(eta / omega);
    c.hscal[1] = (real)(eta * omega);
    PRIA_CHECK(cudaMemcpyAsync(c.dscal, c.hscal, 2 * sizeof(real),
                               cudaMemcpyHostToDevice, c.stream));
    return raw_step(c, in, out);
}

/* Enqueue the canonical-norm reduction of (cur - step) into dred[0..2]. */
template <typename real>
void enqueue_err(Ctx<real>& c) {
    ErrFunctor<real> f{c.d, c.cur.x, c.cur.y, c.cur.z,
                       c.step.x, c.step.y, c.step.z};
    reduce_k<3>(c.d.ND, f, c.dpart, c.dred + kErr0, c.stream);
}

/* Enqueue the objective-gap reductions at the given slab into dred[3..6]. */
template <typename real>
cudaError_t enqueue_gap(Ctx<real>& c, const Slab<real>& S) {
    const i64 nsXY = 2 * c.d.N * c.d.P, nsZ = 2 * c.d.P;
    const int grid = launch_grid(c.d.ND);
    size_t bytes;

    /* term1: sum over (i, s) of max_{t>=s} K_i x  (row-layout segments) */
    k_gather_row_kx<<<grid, kBlock, 0, c.stream>>>(c.d, S.x, c.kdev,
                                                   c.ws.sorted);
    bytes = c.ws.cub_bytes;
    PRIA_CHECK(cub::DeviceSegmentedReduce::Max(c.ws.cub_tmp, bytes,
                                               (const real*)c.ws.sorted,
                                               c.segred, (int)nsXY, c.offY,
                                               c.offY + 1, c.stream));
    reduce_k<1>(nsXY, SumFunctor<real>{c.segred}, c.dpart, c.dred + kTerm1,
                c.stream);

    /* term2: sum over s of max_{i, t>=s} K0 x  (dim2-layout segments) */
    k_gather_d2_k0x<<<grid, kBlock, 0, c.stream>>>(c.d, S.x, (real)c.k0s,
                                                   c.ws.sorted);
    bytes = c.ws.cub_bytes;
    PRIA_CHECK(cub::DeviceSegmentedReduce::Max(c.ws.cub_tmp, bytes,
                                               (const real*)c.ws.sorted,
                                               c.segred, (int)nsZ, c.offZ,
                                               c.offZ + 1, c.stream));
    reduce_k<1>(nsZ, SumFunctor<real>{c.segred}, c.dpart, c.dred + kTerm2,
                c.stream);

    /* term3: <x, h> */
    reduce_k<1>(c.d.ND, DotFunctor<real>{S.x, c.h}, c.dpart, c.dred + kTerm3,
                c.stream);

    /* dual objective: sum over (i, t) of min_{s<=t} (h + y + z) */
    k_gather_lhs<<<grid, kBlock, 0, c.stream>>>(c.d, S.y, S.z, c.h,
                                                c.ws.sorted);
    bytes = c.ws.cub_bytes;
    PRIA_CHECK(cub::DeviceSegmentedReduce::Min(c.ws.cub_tmp, bytes,
                                               (const real*)c.ws.sorted,
                                               c.segred, (int)nsXY, c.offX,
                                               c.offX + 1, c.stream));
    reduce_k<1>(nsXY, SumFunctor<real>{c.segred}, c.dpart, c.dred + kDualG,
                c.stream);
    return cudaPeekAtLastError();
}

/* Enqueue the D2H copies of the device scalars/flags to the pinned mirrors
 * (also the tail of every op_steps block). */
template <typename real>
cudaError_t enqueue_readback(Ctx<real>& c) {
    PRIA_CHECK(cudaMemcpyAsync(c.hred, c.dred, kNumScalar * sizeof(double),
                               cudaMemcpyDeviceToHost, c.stream));
    PRIA_CHECK(cudaMemcpyAsync(c.hriter, c.driter, kNumIter * sizeof(i64),
                               cudaMemcpyDeviceToHost, c.stream));
    return cudaSuccess;
}

/* Copy the device scalars to the pinned host mirrors and wait. */
template <typename real>
cudaError_t sync_reductions(Ctx<real>& c) {
    PRIA_CHECK(enqueue_readback(c));
    PRIA_CHECK(cudaStreamSynchronize(c.stream));
    return cudaSuccess;
}

/* The restart sequence, shared by the host-driven paths and (later) the
 * conditional graph's IF body. Buffer roles are FIXED: instead of the
 * pointer swap, the vanilla-step point is copied into both anchor and cur
 * (bitwise the same states as the swap, so trajectories are unchanged).
 * Everything is stream work - no host decisions, no syncs. */
template <typename real>
cudaError_t enqueue_restart(Ctx<real>& c, const pria_params& p, double eta) {
    /* anchor distances of the pre-restart combined point, then the omega /
     * theta update from them */
    reduce_k<2>(3 * c.d.ND,
                AnchorFunctor<real>{c.cur.base, c.anchor.base, c.d.ND},
                c.dpart, c.dred + kAnch0, c.stream);
    k_restart_update<<<1, 1, 0, c.stream>>>(c.dred, c.dscal, c.driter, eta,
                                            p.theta_target, p.theta_decay);
    /* restart at the vanilla step point: anchor <- step, cur <- step,
     * step <- T(step) with the new omega */
    PRIA_CHECK(cudaMemcpyAsync(c.anchor.base, c.step.base,
                               3 * c.d.ND * sizeof(real),
                               cudaMemcpyDeviceToDevice, c.stream));
    PRIA_CHECK(cudaMemcpyAsync(c.cur.base, c.step.base,
                               3 * c.d.ND * sizeof(real),
                               cudaMemcpyDeviceToDevice, c.stream));
    PRIA_CHECK(raw_step(c, c.cur, c.step));
    enqueue_err(c);
    k_post_restart<<<1, 1, 0, c.stream>>>(c.dred, eta);
    return cudaPeekAtLastError();
}

/* ---- whole-solve conditional graph ------------------------------------- *
 *
 * The entire Halpern loop as one graph: a device-evaluated WHILE node whose
 * body is [op_steps block -> reductions -> k_halpern_logic -> IF(restart)
 * {restart sequence}], with the state readback after the loop. One launch and
 * one sync per solve; k_halpern_logic drives both conditional handles. The
 * kernels are the same ones the host-driven paths run, in the same order, so
 * the modes stay bitwise identical.
 *
 * On any construction failure the graph is discarded and the caller falls
 * back to the host-driven loop. */
template <typename real, typename FBlock>
cudaError_t build_whole_solve(Ctx<real>& c, const pria_params& p, double eta,
                              FBlock&& block_core) {
    cudaGraph_t root = nullptr;
    auto fail = [&](cudaError_t e) {
        cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(c.stream, &st);
        if (st != cudaStreamCaptureStatusNone) {
            cudaGraph_t g2 = nullptr;
            cudaStreamEndCapture(c.stream, &g2);
        }
        if (root) cudaGraphDestroy(root);
        cudaGetLastError();
        return e != cudaSuccess ? e : cudaErrorUnknown;
    };

    PRIA_CHECK(cudaGraphCreate(&root, 0));

    /* WHILE node; the handle resets to 1 at every launch, so the body always
     * runs at least once (the caller only launches when not terminated). */
    cudaGraphConditionalHandle h_while;
    cudaError_t st = cudaGraphConditionalHandleCreate(
        &h_while, root, 1, cudaGraphCondAssignDefault);
    if (st != cudaSuccess) return fail(st);
    cudaGraphNodeParams wp = {};
    wp.type = cudaGraphNodeTypeConditional;
    wp.conditional.handle = h_while;
    wp.conditional.type = cudaGraphCondTypeWhile;
    wp.conditional.size = 1;
    cudaGraphNode_t while_node = nullptr;
    st = cudaGraphAddNode(&while_node, root, nullptr, nullptr, 0, &wp);
    if (st != cudaSuccess) return fail(st);
    cudaGraph_t body = wp.conditional.phGraph_out[0];

    /* restart IF handle lives on the body graph (which contains the node);
     * k_halpern_logic assigns it every block, so no default is needed */
    cudaGraphConditionalHandle h_restart;
    st = cudaGraphConditionalHandleCreate(&h_restart, body, 0, 0);
    if (st != cudaSuccess) return fail(st);

    /* capture the block into the WHILE body, then splice the IF node in at
     * the capture tail */
    st = cudaStreamBeginCaptureToGraph(c.stream, body, nullptr, nullptr, 0,
                                       cudaStreamCaptureModeThreadLocal);
    if (st != cudaSuccess) return fail(st);
    st = block_core(h_while, h_restart, 1);
    if (st != cudaSuccess) return fail(st);

    cudaStreamCaptureStatus cst = cudaStreamCaptureStatusNone;
    cudaGraph_t cap_graph = nullptr;
    const cudaGraphNode_t* deps = nullptr;
    size_t ndeps = 0;
    st = cudaStreamGetCaptureInfo(c.stream, &cst, nullptr, &cap_graph, &deps,
                                  nullptr, &ndeps);
    if (st != cudaSuccess || cst != cudaStreamCaptureStatusActive)
        return fail(st);

    cudaGraphNodeParams ip = {};
    ip.type = cudaGraphNodeTypeConditional;
    ip.conditional.handle = h_restart;
    ip.conditional.type = cudaGraphCondTypeIf;
    ip.conditional.size = 1;
    cudaGraphNode_t if_node = nullptr;
    st = cudaGraphAddNode(&if_node, cap_graph, deps, nullptr, ndeps, &ip);
    if (st != cudaSuccess) return fail(st);
    cudaGraph_t restart_body = ip.conditional.phGraph_out[0];
    st = cudaStreamUpdateCaptureDependencies(c.stream, &if_node, nullptr, 1,
                                             cudaStreamSetCaptureDependencies);
    if (st != cudaSuccess) return fail(st);
    cudaGraph_t ret = nullptr;
    st = cudaStreamEndCapture(c.stream, &ret);
    if (st != cudaSuccess) return fail(st);

    /* capture the restart sequence into the IF body */
    st = cudaStreamBeginCaptureToGraph(c.stream, restart_body, nullptr,
                                       nullptr, 0,
                                       cudaStreamCaptureModeThreadLocal);
    if (st != cudaSuccess) return fail(st);
    st = enqueue_restart(c, p, eta);
    if (st != cudaSuccess) return fail(st);
    st = cudaStreamEndCapture(c.stream, &ret);
    if (st != cudaSuccess) return fail(st);

    /* state readback after the loop completes */
    st = cudaStreamBeginCaptureToGraph(c.stream, root, &while_node, nullptr,
                                       1, cudaStreamCaptureModeThreadLocal);
    if (st != cudaSuccess) return fail(st);
    st = enqueue_readback(c);
    if (st != cudaSuccess) return fail(st);
    st = cudaStreamEndCapture(c.stream, &ret);
    if (st != cudaSuccess) return fail(st);

    st = cudaGraphInstantiate(&c.whole_exec, root, 0);
    if (st != cudaSuccess || !c.whole_exec) {
        c.whole_exec = nullptr;
        return fail(st);
    }
    cudaGraphDestroy(root);
    return cudaSuccess;
}

inline double canonical_err(const double* hred, double eta, double omega) {
    /* keep in sync with dev_canonical_err: negative cross term (PPA metric) */
    double v = (omega / eta) * hred[kErr0] +
               (1.0 / (eta * omega)) * hred[kErr1] - 2.0 * hred[kErr2];
    return std::sqrt(v > 0.0 ? v : 0.0);
}

struct GapInfo {
    double primal; /* implied primal objective (upper bound) */
    double dual;   /* implied dual objective (lower bound) */
    double gap;    /* relative gap */
};

inline GapInfo gap_from(const double* hred, double scale) {
    GapInfo g;
    g.primal = (hred[kTerm1] + hred[kTerm2] + hred[kTerm3]) * scale;
    g.dual = hred[kDualG] * scale;
    g.gap = (g.primal - g.dual) /
            (1.0 + std::fabs(g.dual) + std::fabs(g.primal));
    return g;
}

/* ---- setup / teardown -------------------------------------------------- */

template <typename real>
void free_ctx(Ctx<real>& c) {
    /* stream-ordered frees return the blocks to the shared pool */
    auto ffree = [&](void* p) {
        if (!p) return;
        if (c.stream) cudaFreeAsync(p, c.stream);
        else cudaFree(p);
    };
    if (c.ws.zprobe && c.ws.zstats) {
        unsigned long long zs[2] = {0, 0};
        cudaMemcpy(zs, c.ws.zstats, sizeof(zs), cudaMemcpyDeviceToHost);
        if (zs[0] + zs[1])
            fprintf(stderr,
                    "pria: z-sort probe: %llu of %llu projections still "
                    "sorted (%.1f%%)\n",
                    zs[0], zs[0] + zs[1],
                    100.0 * (double)zs[0] / (double)(zs[0] + zs[1]));
    }
    if (c.ws.sfstats) {
        unsigned long long s[3] = {0, 0, 0};
        cudaStreamSynchronize(c.stream);
        cudaMemcpy(s, c.ws.sfstats, sizeof(s), cudaMemcpyDeviceToHost);
        if (s[0])
            fprintf(stderr,
                    "pria: sf michelot: %llu calls, %.2f rounds/call, "
                    "%llu warm fallbacks (%.1f%%)\n",
                    s[0], (double)s[1] / (double)s[0], s[2],
                    100.0 * (double)s[2] / (double)s[0]);
    }
    ffree(c.ws.sfstats);
    ffree(c.dbar);
    ffree(c.warmX);
    ffree(c.warmY);
    ffree(c.warmZ);
    ffree(c.ws.perm);
    ffree(c.ws.iota);
    ffree(c.ws.sortflag);
    ffree(c.ws.zstats);
    c.g_halpern.destroy();
    c.g_vanilla.destroy();
    if (c.whole_exec) cudaGraphExecDestroy(c.whole_exec);
    ffree(c.dscal);
    ffree(c.driter);
    if (c.hscal) cudaFreeHost(c.hscal);
    if (c.hriter) cudaFreeHost(c.hriter);
    ffree(c.cur.base);
    ffree(c.step.base);
    ffree(c.anchor.base);
    ffree(c.pre.base);
    ffree(c.h);
    ffree(c.kdev);
    ffree(c.invk);
    ffree(c.offX);
    ffree(c.offY);
    ffree(c.offZ);
    ffree(c.ws.sorted);
    ffree(c.ws.pairs);
    ffree(c.ws.rho);
    ffree(c.ws.theta);
    ffree(c.ws.cub_tmp);
    ffree(c.segred);
    ffree(c.dred);
    ffree(c.dpart);
    ffree(c.outfull);
    ffree(c.mir1);
    ffree(c.mir2);
    if (c.stream) cudaStreamSynchronize(c.stream);
    if (c.hred) cudaFreeHost(c.hred);
    if (c.ev_fork) cudaEventDestroy(c.ev_fork);
    if (c.ev_join) cudaEventDestroy(c.ev_join);
    if (c.stream2) cudaStreamDestroy(c.stream2);
    if (c.stream) cudaStreamDestroy(c.stream);
    c = Ctx<real>{};
}

template <typename real, typename cost_t>
cudaError_t setup_ctx(Ctx<real>& c, i64 n, i64 t, const cost_t* k_host,
                      double k0, const cost_t* h_host) {
    c.torig = t;
    const i64 T = t + (t & 1);
    c.d = Dims::make(n, T);
    const Dims d = c.d;
    if (d.ND > (i64)INT32_MAX) return cudaErrorInvalidValue;

    /* objective scaling: 2-norm of (triu(h), k, k0), as in the JAX solver */
    double ss = k0 * k0;
    for (i64 i = 0; i < n; ++i) ss += (double)k_host[i] * (double)k_host[i];
    for (i64 i = 0; i < n; ++i)
        for (i64 s = 0; s < t; ++s)
            for (i64 tt = s; tt < t; ++tt) {
                double v = (double)h_host[(i * t + s) * t + tt];
                ss += v * v;
            }
    c.scale = std::sqrt(ss);
    if (c.scale == 0.0) c.scale = 1.0;
    c.k0s = k0 / c.scale;

    /* Non-blocking: batch solves capture graphs concurrently on several
     * threads, and a legacy-stream operation from one thread must not
     * implicitly depend on another thread's capturing stream. */
    PRIA_CHECK(cudaStreamCreateWithFlags(&c.stream, cudaStreamNonBlocking));

    /* Stream-ordered allocations below; keep freed blocks in the default
     * pool so repeated and batched solves reuse them instead of serializing
     * on the driver allocator. */
    {
        static std::once_flag pool_once;
        std::call_once(pool_once, [] {
            int dev = 0;
            cudaMemPool_t pool = nullptr;
            if (cudaGetDevice(&dev) == cudaSuccess &&
                cudaDeviceGetDefaultMemPool(&pool, dev) == cudaSuccess) {
                unsigned long long thr = ~0ull;
                cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold,
                                        &thr);
            }
        });
    }

    real* b;
    PRIA_CHECK(cudaMallocAsync(&b, 3 * d.ND * sizeof(real), c.stream));
    c.cur.assign(b, d.ND);
    PRIA_CHECK(cudaMallocAsync(&b, 3 * d.ND * sizeof(real), c.stream));
    c.step.assign(b, d.ND);
    PRIA_CHECK(cudaMallocAsync(&b, 3 * d.ND * sizeof(real), c.stream));
    c.anchor.assign(b, d.ND);
    PRIA_CHECK(cudaMallocAsync(&b, 3 * d.ND * sizeof(real), c.stream));
    c.pre.assign(b, d.ND);
    PRIA_CHECK(cudaMallocAsync(&c.h, d.ND * sizeof(real), c.stream));
    PRIA_CHECK(cudaMallocAsync(&c.kdev, n * sizeof(real), c.stream));
    PRIA_CHECK(cudaMallocAsync(&c.invk, n * sizeof(real), c.stream));

    const i64 nsXY = 2 * d.N * d.P, nsZ = 2 * d.P;
    PRIA_CHECK(cudaMallocAsync(&c.offX, (nsXY + 1) * sizeof(int), c.stream));
    PRIA_CHECK(cudaMallocAsync(&c.offY, (nsXY + 1) * sizeof(int), c.stream));
    PRIA_CHECK(cudaMallocAsync(&c.offZ, (nsZ + 1) * sizeof(int), c.stream));
    PRIA_CHECK(cudaMallocAsync(&c.warmX, nsXY * sizeof(real), c.stream));
    PRIA_CHECK(cudaMallocAsync(&c.warmY, nsXY * sizeof(real), c.stream));
    PRIA_CHECK(cudaMallocAsync(&c.warmZ, nsZ * sizeof(real), c.stream));
    {
        const real NEG = -std::numeric_limits<real>::infinity();
        k_fill<<<launch_grid(nsXY), kBlock, 0, c.stream>>>(nsXY, NEG, c.warmX);
        k_fill<<<launch_grid(nsXY), kBlock, 0, c.stream>>>(nsXY, NEG, c.warmY);
        k_fill<<<launch_grid(nsZ), kBlock, 0, c.stream>>>(nsZ, NEG, c.warmZ);
    }
    if (sf_stats_enabled()) {
        PRIA_CHECK(cudaMallocAsync(&c.ws.sfstats,
                                   3 * sizeof(unsigned long long), c.stream));
        PRIA_CHECK(cudaMemsetAsync(c.ws.sfstats, 0,
                                   3 * sizeof(unsigned long long), c.stream));
    }
    /* ws.sorted doubles as the sort output and the sort-free compaction
     * scratch; the sort-only scratch (pairs/rho/theta and the big CUB
     * temps) is skipped entirely under the sort-free backend. The backend
     * is latched here: tests that flip sortfree_mode() must do so before
     * setup_ctx. */
    PRIA_CHECK(cudaMallocAsync(&c.ws.sorted, d.ND * sizeof(real), c.stream));
    if (!sortfree_mode()) {
        PRIA_CHECK(cudaMallocAsync(&c.ws.pairs, d.ND * sizeof(SegPair<real>),
                                   c.stream));
        PRIA_CHECK(cudaMallocAsync(&c.ws.rho, nsXY * sizeof(int), c.stream));
        PRIA_CHECK(cudaMallocAsync(&c.ws.theta, nsXY * sizeof(real),
                                   c.stream));
    }
    PRIA_CHECK(cudaMallocAsync(&c.segred, nsXY * sizeof(real), c.stream));
    PRIA_CHECK(cudaMallocAsync(&c.dred, kNumScalar * sizeof(double), c.stream));
    PRIA_CHECK(cudaMallocAsync(&c.dpart, kReduceMaxGrid * 4 * sizeof(double), c.stream));
    PRIA_CHECK(cudaMallocAsync(&c.outfull, n * t * t * sizeof(real), c.stream));
    PRIA_CHECK(cudaMallocHost(&c.hred, kNumScalar * sizeof(double)));
    PRIA_CHECK(cudaMallocAsync(&c.dscal, 2 * sizeof(real), c.stream));
    PRIA_CHECK(cudaMallocAsync(&c.driter, kNumIter * sizeof(i64), c.stream));
    PRIA_CHECK(cudaMemsetAsync(c.driter, 0, kNumIter * sizeof(i64), c.stream));
    PRIA_CHECK(cudaMallocHost(&c.hscal, 2 * sizeof(real)));
    PRIA_CHECK(cudaMallocHost(&c.hriter, kNumIter * sizeof(i64)));
    c.hscal[0] = c.hscal[1] = real(0);
    for (int i = 0; i < kNumIter; ++i) c.hriter[i] = 0;
    c.graphs_ok = !graphs_disabled();

    /* y/z overlap requires the y projection to take the fused path (the
     * device-wide scratch may then be used exclusively by z) */
    PRIA_CHECK(cudaStreamCreateWithFlags(&c.stream2, cudaStreamNonBlocking));
    PRIA_CHECK(cudaEventCreateWithFlags(&c.ev_fork, cudaEventDisableTiming));
    PRIA_CHECK(cudaEventCreateWithFlags(&c.ev_join, cudaEventDisableTiming));
    /* Auto: only when z is device-wide (see overlap_mode); the W gate
     * guarantees the y projection takes the fused path (sole-user scratch
     * safety), the ND gate keeps the fork off cache-thrashing sizes. */
    const bool z_devwide =
        d.W * d.N * (i64)sizeof(real) > (i64)fused_sf_shared_cap<real>();
    c.overlap = !fused_projection_disabled() && overlap_mode() != 0 &&
                (overlap_mode() == 1 || z_devwide) &&
                d.W <= 256 * (sizeof(real) == 4 ? 16 : 8) &&
                d.ND * (i64)sizeof(real) <= (i64)16 * 1024 * 1024;

    c.ws.zprobe = zsort_probe_enabled();
    if (c.ws.zprobe) {
        PRIA_CHECK(cudaMallocAsync(&c.ws.perm, d.ND * sizeof(int), c.stream));
        PRIA_CHECK(cudaMallocAsync(&c.ws.iota, d.ND * sizeof(int), c.stream));
        PRIA_CHECK(cudaMallocAsync(&c.ws.sortflag, sizeof(int), c.stream));
        PRIA_CHECK(cudaMallocAsync(&c.ws.zstats, 2 * sizeof(unsigned long long), c.stream));
        PRIA_CHECK(cudaMemsetAsync(c.ws.zstats, 0,
                                   2 * sizeof(unsigned long long), c.stream));
        k_iota<<<launch_grid(d.ND), kBlock, 0, c.stream>>>(d.ND, c.ws.iota);
        k_iota<<<launch_grid(d.ND), kBlock, 0, c.stream>>>(d.ND, c.ws.perm);
    }

    /* Tiled-transpose mirrors, worthwhile once a variable exceeds all of
     * L2. Sort-free reads each gather source once per iteration, so below
     * that the staging does not amortize. */
    c.use_mirrors =
        mirrors_mode() == 1 ||
        (mirrors_mode() < 0 &&
         d.ND * (i64)sizeof(real) > (i64)96 * 1024 * 1024);
    if (c.use_mirrors) {
        PRIA_CHECK(cudaMallocAsync(&c.mir1, d.ND * sizeof(real), c.stream));
        PRIA_CHECK(cudaMallocAsync(&c.mir2, d.ND * sizeof(real), c.stream));
    }

    /* reflect-inline in the y/z consumers: it doubles the scattered
     * gathers, which loses badly once the working set leaves cache comfort
     * (+6% f32 / +23% f64 at 200x300) but saves a kernel and an ND round
     * trip where everything is tiny (-1% at 30x75). Sort-free non-mirror
     * only (the mirror kernels re-read the reflection tile-wise). */
    c.rfuse = sortfree_mode() && rfuse_mode() != 0 &&
              d.ND * (i64)sizeof(real) <= (i64)2 * 1024 * 1024;

    /* persistent megakernel: variant by precision (see the kernel comment
     * for the measurement matrix), grid sized to residency for
     * deadlock-free software barriers. Auto only for tiny instances; mid
     * sizes lose with every variant (+9..+23%). */
    {
        c.mega_block = sizeof(real) == 4;
        const i64 shbytes = c.mega_block
                                ? d.W * (i64)sizeof(real)
                                : d.W * d.N * (i64)sizeof(real);
        const bool mega_capable = sortfree_mode() && !c.use_mirrors &&
                                  !batch_worker() &&
                                  shbytes <= (i64)fused_sf_shared_cap<real>();
        const bool mega_want =
            mega_mode() == 1 ||
            (mega_mode() < 0 && shbytes <= (i64)24 * 1024 &&
             d.ND * (i64)sizeof(real) <= (i64)2 * 1024 * 1024);
        if (mega_capable && mega_want) {
            int nb = 0, sms = 0, dev = 0;
            cudaGetDevice(&dev);
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                                   dev);
            cudaError_t occ =
                c.mega_block
                    ? cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                          &nb, k_mega_block<real>, kBlock, (size_t)shbytes)
                    : cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                          &nb, k_mega_iter<real>, kBlock, (size_t)shbytes);
            if (occ == cudaSuccess && nb > 0 && sms > 0) {
                const i64 work =
                    std::max((3 * d.ND + kBlock - 1) / kBlock,
                             d.N * d.P + 2 * d.P);
                c.mega_grid = (int)std::min((i64)nb * sms, work);
                if (c.mega_grid > 0) {
                    c.mega = true;
                    PRIA_CHECK(cudaMallocAsync(&c.dbar, 2 * sizeof(unsigned),
                                               c.stream));
                    PRIA_CHECK(cudaMemsetAsync(c.dbar, 0,
                                               2 * sizeof(unsigned),
                                               c.stream));
                }
            }
            cudaGetLastError(); /* clear any occupancy-query error */
        }
    }

    /* segment offsets. All setup uploads run async on c.stream with
     * stream-local syncs: the solver must never touch the legacy stream or
     * synchronize the device, since batch workers on other threads may be
     * mid-capture (device-wide sync would have to synchronize their
     * capturing streams, which is illegal). */
    {
        std::vector<int> tmp(nsXY + 1);
        build_offsets_xy(d, true, tmp.data());
        PRIA_CHECK(cudaMemcpyAsync(c.offX, tmp.data(),
                                   (nsXY + 1) * sizeof(int),
                                   cudaMemcpyHostToDevice, c.stream));
        PRIA_CHECK(cudaStreamSynchronize(c.stream)); /* tmp is reused */
        build_offsets_xy(d, false, tmp.data());
        PRIA_CHECK(cudaMemcpyAsync(c.offY, tmp.data(),
                                   (nsXY + 1) * sizeof(int),
                                   cudaMemcpyHostToDevice, c.stream));
        std::vector<int> tmpz(nsZ + 1);
        build_offsets_z(d, tmpz.data());
        PRIA_CHECK(cudaMemcpyAsync(c.offZ, tmpz.data(),
                                   (nsZ + 1) * sizeof(int),
                                   cudaMemcpyHostToDevice, c.stream));
        PRIA_CHECK(cudaStreamSynchronize(c.stream));
    }

    /* scaled costs */
    {
        std::vector<real> kk(n), ik(n);
        for (i64 i = 0; i < n; ++i) {
            double ks = (double)k_host[i] / c.scale;
            kk[i] = (real)ks;
            ik[i] = ks > 0.0 ? (real)(1.0 / ks) : real(0);
        }
        PRIA_CHECK(cudaMemcpyAsync(c.kdev, kk.data(), n * sizeof(real),
                                   cudaMemcpyHostToDevice, c.stream));
        PRIA_CHECK(cudaMemcpyAsync(c.invk, ik.data(), n * sizeof(real),
                                   cudaMemcpyHostToDevice, c.stream));

        std::vector<real> hd(d.ND);
        for (i64 i = 0; i < n; ++i)
            for (i64 j = 0; j < d.P; ++j)
                for (i64 cc = 0; cc < d.W; ++cc) {
                    i64 s, tt;
                    col_to_st(d.T, j, cc, s, tt);
                    double v = (tt < t) ? (double)h_host[(i * t + s) * t + tt]
                                        : 0.0;
                    hd[(i * d.P + j) * d.W + cc] = (real)(v / c.scale);
                }
        PRIA_CHECK(cudaMemcpyAsync(c.h, hd.data(), d.ND * sizeof(real),
                                   cudaMemcpyHostToDevice, c.stream));
        PRIA_CHECK(cudaStreamSynchronize(c.stream));
    }

    /* CUB temp storage: max requirement over every call we make */
    {
        size_t need = 0, b2 = 0;
        auto upd = [&](size_t nb) { if (nb > need) need = nb; };

        if (!sortfree_mode()) {
            b2 = 0;
            PRIA_CHECK(cub::DeviceSegmentedSort::SortKeysDescending(
                nullptr, b2, (const real*)c.pre.x, c.ws.sorted, d.ND, nsXY,
                c.offX, c.offX + 1, c.stream));
            upd(b2);
            b2 = 0;
            PRIA_CHECK(cub::DeviceSegmentedSort::SortKeysDescending(
                nullptr, b2, (const real*)c.pre.z, c.ws.sorted, d.ND, nsZ,
                c.offZ, c.offZ + 1, c.stream));
            upd(b2);
            if (zsort_probe_enabled()) {
                b2 = 0;
                PRIA_CHECK(cub::DeviceSegmentedSort::SortPairsDescending(
                    nullptr, b2, (const real*)c.pre.z, c.ws.sorted,
                    (const int*)c.ws.iota, c.ws.perm, d.ND, nsZ, c.offZ,
                    c.offZ + 1, c.stream));
                upd(b2);
            }

            b2 = 0;
            PRIA_CHECK(cub::DeviceScan::InclusiveScan(
                nullptr, b2, PairLoadIter<real, GeomX>{c.ws.sorted, c.gx(), 0},
                c.ws.pairs, SegScanOp<real>{}, d.ND, c.stream));
            upd(b2);
            b2 = 0;
            PRIA_CHECK(cub::DeviceScan::InclusiveScan(
                nullptr, b2, PairLoadIter<real, GeomY>{c.ws.sorted, c.gy(), 0},
                c.ws.pairs, SegScanOp<real>{}, d.ND, c.stream));
            upd(b2);
            b2 = 0;
            PRIA_CHECK(cub::DeviceScan::InclusiveScan(
                nullptr, b2, PairLoadIter<real, GeomZ>{c.ws.sorted, c.gz(), 0},
                c.ws.pairs, SegScanOp<real>{}, d.ND, c.stream));
            upd(b2);
        }

        b2 = 0;
        PRIA_CHECK(cub::DeviceSegmentedReduce::Max(
            nullptr, b2, (const real*)c.ws.sorted, c.segred, (int)nsXY, c.offY,
            c.offY + 1, c.stream));
        upd(b2);
        b2 = 0;
        PRIA_CHECK(cub::DeviceSegmentedReduce::Max(
            nullptr, b2, (const real*)c.ws.sorted, c.segred, (int)nsZ, c.offZ,
            c.offZ + 1, c.stream));
        upd(b2);
        b2 = 0;
        PRIA_CHECK(cub::DeviceSegmentedReduce::Min(
            nullptr, b2, (const real*)c.ws.sorted, c.segred, (int)nsXY, c.offX,
            c.offX + 1, c.stream));
        upd(b2);

        c.ws.cub_bytes = need;
        PRIA_CHECK(cudaMallocAsync(&c.ws.cub_tmp, need, c.stream));
    }
    return cudaSuccess;
}

/* Densify a full [N][Torig][Torig] host array into a dense-layout host
 * staging vector (mode 0/1/2 = col/row/dim2), padding with zeros. */
template <typename real, typename src_t>
void densify_host(const Dims& d, i64 torig, const src_t* full, int mode,
                  real* out) {
    for (i64 i = 0; i < d.N; ++i)
        for (i64 j = 0; j < d.P; ++j)
            for (i64 cc = 0; cc < d.W; ++cc) {
                i64 s, t;
                if (mode == 0) col_to_st(d.T, j, cc, s, t);
                else row_to_st(d.T, j, cc, s, t);
                real v = real(0);
                if (t < torig)
                    v = (real)full[(i * torig + s) * torig + t];
                i64 g = (mode == 2) ? (j * d.W + cc) * d.N + i
                                    : (i * d.P + j) * d.W + cc;
                out[g] = v;
            }
}

/* ---- the solver -------------------------------------------------------- */

template <typename real>
cudaError_t solve_impl(i64 n, i64 t, const real* k_host, real k0,
                       const real* h_host, const pria_params* params_in,
                       const real* warm_x, const real* warm_y,
                       const real* warm_z, real* out_x, real* out_y,
                       real* out_z, pria_result_info* info) {
    if (n < 1 || t < 1) return cudaErrorInvalidValue;

    pria_params p = *params_in;
    if (p.op_steps < 2) p.op_steps = 2;
    if (p.warm_steps < 0) p.warm_steps = 0;

    Ctx<real> c;
    cudaError_t st = setup_ctx(c, n, t, k_host, (double)k0, h_host);
    if (st != cudaSuccess) { free_ctx(c); return st; }
    const Dims d = c.d;

    struct Guard {
        Ctx<real>& c;
        ~Guard() { free_ctx(c); }
    } guard{c};

    /* --- initial point: project the (densified) warm start ------------- *
     * Warm duals arrive normalized (like the solver's outputs); scale them
     * back to the internal K_i/K0 simplices so a returned solution is a
     * fixed point of the warm start. */
    if (warm_x && warm_y && warm_z) {
        /* async on c.stream + stream-local syncs (stage is reused, and the
         * solver must not touch the legacy stream or sync the device - see
         * setup_ctx) */
        std::vector<real> stage(d.ND);
        densify_host(d, t, warm_x, 0, stage.data());
        PRIA_CHECK(cudaMemcpyAsync(c.pre.x, stage.data(),
                                   d.ND * sizeof(real),
                                   cudaMemcpyHostToDevice, c.stream));
        PRIA_CHECK(cudaStreamSynchronize(c.stream));
        densify_host(d, t, warm_y, 1, stage.data());
        for (i64 g = 0; g < d.ND; ++g)
            stage[g] *= (real)((double)k_host[g / d.D] / c.scale);
        PRIA_CHECK(cudaMemcpyAsync(c.pre.y, stage.data(),
                                   d.ND * sizeof(real),
                                   cudaMemcpyHostToDevice, c.stream));
        PRIA_CHECK(cudaStreamSynchronize(c.stream));
        densify_host(d, t, warm_z, 2, stage.data());
        for (i64 g = 0; g < d.ND; ++g) stage[g] *= (real)c.k0s;
        PRIA_CHECK(cudaMemcpyAsync(c.pre.z, stage.data(),
                                   d.ND * sizeof(real),
                                   cudaMemcpyHostToDevice, c.stream));
        PRIA_CHECK(cudaStreamSynchronize(c.stream));
    } else {
        PRIA_CHECK(cudaMemsetAsync(c.pre.base, 0, 3 * d.ND * sizeof(real),
                                   c.stream));
    }
    PRIA_CHECK(project_segments(c.ws, (const real*)c.pre.x, c.cur.x, c.gx(),
                                c.offX, c.warmX, c.kdev, (real)c.k0s,
                                c.stream));
    PRIA_CHECK(project_segments(c.ws, (const real*)c.pre.y, c.cur.y, c.gy(),
                                c.offY, c.warmY, c.kdev, (real)c.k0s,
                                c.stream));
    PRIA_CHECK(project_segments(c.ws, (const real*)c.pre.z, c.cur.z, c.gz(),
                                c.offZ, c.warmZ, c.kdev, (real)c.k0s,
                                c.stream));

    /* keep the initial point for the post-warmup omega estimate */
    PRIA_CHECK(cudaMemcpyAsync(c.anchor.base, c.cur.base,
                               3 * d.ND * sizeof(real),
                               cudaMemcpyDeviceToDevice, c.stream));

    const double eta = p.eta;
    i64 iteration = 0;
    k_set_omega<<<1, 1, 0, c.stream>>>(c.dred, c.dscal, p.omega, eta);
    PRIA_CHECK(raw_step(c, c.cur, c.step));

    GapInfo gi{0.0, 0.0, std::numeric_limits<double>::infinity()};

    /* One op_steps block of plain PDHG steps. The cur/step buffer roles
     * alternate through local slab copies so the enqueue is pure stream work
     * and can be captured in a graph; the host applies the net parity change
     * after the block. */
    auto vanilla_block = [&]() -> cudaError_t {
        Slab<real> a = c.step, b = c.cur;
        for (int s2 = 0; s2 < p.op_steps; ++s2) {
            PRIA_CHECK(raw_step(c, a, b));
            std::swap(a, b);
        }
        return cudaSuccess;
    };

    /* Vanilla loop: check the gap at cur, then run op_steps plain PDHG
     * steps, until the gap or the local iteration budget is hit. */
    auto run_vanilla = [&](i64 local_max) -> cudaError_t {
        for (;;) {
            PRIA_CHECK(enqueue_gap(c, c.cur));
            PRIA_CHECK(sync_reductions(c));
            gi = gap_from(c.hred, c.scale);
            if (p.verbose)
                fprintf(stderr,
                        "pria: iter %8lld  primal %.9e  dual %.9e  gap %.3e\n",
                        (long long)iteration, gi.primal, gi.dual, gi.gap);
            bool within = (iteration < local_max) || (local_max < 0);
            if (!within || !(gi.gap > p.tolerance)) return cudaSuccess;
            PRIA_CHECK(run_block_graph(c, c.g_vanilla, c.cur.base,
                                       vanilla_block));
            if (p.op_steps & 1) std::swap(c.cur, c.step);
            iteration += p.op_steps;
        }
    };

    /* --- phase 1: vanilla warmup (or the whole solve without Halpern) -- */
    i64 warm_budget = p.use_halpern ? (i64)p.op_steps * p.warm_steps
                                    : p.max_iter;
    if (!p.use_halpern || warm_budget != 0)
        PRIA_CHECK(run_vanilla(warm_budget));

    /* --- omega re-estimate from the warmup displacement (device-side) -- */
    if (iteration > 0) {
        reduce_k<2>(3 * d.ND,
                    AnchorFunctor<real>{c.cur.base, c.anchor.base, d.ND},
                    c.dpart, c.dred + kAnch0, c.stream);
        k_warm_omega<<<1, 1, 0, c.stream>>>(c.dred, c.dscal, p.omega, eta);
    }

    /* Which slab holds the reported iterate: cur unless the Halpern loop ran
     * and terminated, in which case it is the vanilla-step point (step). */
    bool report_step = false;

    if (!p.use_halpern) {
        /* vanilla solve is complete (matching the JAX flow, where the second
         * multi-step never runs another iteration) */
    } else {
        PRIA_CHECK(cudaMemcpyAsync(c.anchor.base, c.cur.base,
                                   3 * d.ND * sizeof(real),
                                   cudaMemcpyDeviceToDevice, c.stream));
        enqueue_err(c);
        PRIA_CHECK(enqueue_gap(c, c.cur));
        k_init_state<<<1, 1, 0, c.stream>>>(c.dred, c.driter, eta, c.scale,
                                            p.tolerance, p.theta_initial,
                                            (i64)p.max_iter, iteration);
        PRIA_CHECK(sync_reductions(c));
        gi = GapInfo{c.hred[kStPrimal], c.hred[kStDual], c.hred[kStGap]};
        bool terminated = c.hriter[kItTerminated] != 0;

        /* One Halpern op_steps block: op_steps combine + operator
         * applications, the convergence reductions, and the on-device
         * decision kernel. All scalars it consumes live on the device, so
         * the block replays as a graph with no host-side inputs; with
         * handles wired it forms the WHILE body of the whole-solve graph. */
        auto block_core = [&](cudaGraphConditionalHandle hw,
                              cudaGraphConditionalHandle hr,
                              int use_handles) -> cudaError_t {
            if (c.mega && c.mega_block) {
                /* f32: the whole op_steps block in one launch; c.pre.z
                 * doubles as the z pre scratch (unused on this path) */
                k_mega_block<<<c.mega_grid, kBlock,
                               (size_t)(c.d.W * sizeof(real)), c.stream>>>(
                    c.d, c.cur.base, c.step.base,
                    (const real*)c.anchor.base,
                    (const i64*)(c.driter + kItRestart), p.op_steps, c.h,
                    (const real*)c.dscal, c.kdev, (real)c.k0s, c.offZ,
                    sf_warm_mode() ? c.warmZ : (real*)nullptr, c.pre.z,
                    c.dbar);
                PRIA_CHECK(cudaPeekAtLastError());
            } else if (c.mega) {
                /* f64: one launch per iteration, z fused in shared */
                for (int s2 = 0; s2 < p.op_steps; ++s2)
                    k_mega_iter<<<c.mega_grid, kBlock,
                                  (size_t)(c.d.W * c.d.N * sizeof(real)),
                                  c.stream>>>(
                        c.d, c.cur.base, c.step.base,
                        (const real*)c.anchor.base,
                        (const i64*)(c.driter + kItRestart), s2, c.h,
                        (const real*)c.dscal, c.kdev, (real)c.k0s, c.dbar);
                PRIA_CHECK(cudaPeekAtLastError());
            } else {
                for (int s2 = 0; s2 < p.op_steps; ++s2) {
                    k_combine<<<launch_grid(3 * d.ND), kBlock, 0,
                                c.stream>>>(
                        3 * d.ND, c.cur.base, (const real*)c.step.base,
                        (const real*)c.anchor.base,
                        (const i64*)(c.driter + kItRestart), s2);
                    PRIA_CHECK(raw_step(c, c.cur, c.step));
                }
            }
            enqueue_err(c);
            PRIA_CHECK(enqueue_gap(c, c.step));
            k_halpern_logic<<<1, 1, 0, c.stream>>>(
                c.dred, c.driter, eta, c.scale, p.tolerance,
                p.beta_sufficient, p.beta_necessary, p.beta_artificial,
                (i64)p.op_steps, (i64)p.max_iter, hw, hr, use_handles);
            return cudaPeekAtLastError();
        };

        auto halpern_block = [&]() -> cudaError_t {
            PRIA_CHECK(block_core(0, 0, 0));
            PRIA_CHECK(enqueue_readback(c));
            return cudaSuccess;
        };

        /* Preferred: run the whole loop as one conditional graph. Verbose
         * needs per-block host visibility, so it stays on the driven loop. */
        if (!terminated && c.graphs_ok && !cond_disabled() && !p.verbose) {
            if (build_whole_solve(c, p, eta, block_core) == cudaSuccess) {
                report_step = true;
                PRIA_CHECK(cudaGraphLaunch(c.whole_exec, c.stream));
                PRIA_CHECK(cudaStreamSynchronize(c.stream));
                iteration = c.hriter[kItIter];
                terminated = c.hriter[kItTerminated] != 0;
                gi = GapInfo{c.hred[kStPrimal], c.hred[kStDual],
                             c.hred[kStGap]};
            }
            /* on build failure fall through to the host-driven loop */
        }

        while (!terminated) {
            report_step = true;
            PRIA_CHECK(run_block_graph(c, c.g_halpern, c.cur.base,
                                       halpern_block));
            PRIA_CHECK(cudaStreamSynchronize(c.stream));

            iteration = c.hriter[kItIter];
            terminated = c.hriter[kItTerminated] != 0;
            gi = GapInfo{c.hred[kStPrimal], c.hred[kStDual], c.hred[kStGap]};

            if (p.verbose)
                fprintf(stderr,
                        "pria: iter %8lld  primal %.9e  dual %.9e  gap %.3e  "
                        "err %.3e%s%s%s\n",
                        (long long)iteration, gi.primal, gi.dual, gi.gap,
                        c.hred[kStNewErr],
                        c.hriter[kItSuff] ? "  restart:sufficient" : "",
                        (!c.hriter[kItSuff] && c.hriter[kItNec])
                            ? "  restart:necessary" : "",
                        (!c.hriter[kItSuff] && !c.hriter[kItNec] &&
                         c.hriter[kItArt]) ? "  restart:artificial" : "");

            if (terminated) break;
            if (c.hriter[kItShouldRestart])
                PRIA_CHECK(enqueue_restart(c, p, eta));
        }
    }

    /* --- finalize: sparsify, unscale the duals, copy out --------------- */
    const Slab<real> fin = report_step ? c.step : c.cur;
    const i64 full = n * t * t;
    const int fgrid = launch_grid(full);
    const real invk0 = c.k0s > 0.0 ? (real)(1.0 / c.k0s) : real(0);

    if (out_x) {
        k_sparsify<real, 0><<<fgrid, kBlock, 0, c.stream>>>(
            d, t, (const real*)fin.x, (const real*)c.invk, invk0, c.outfull);
        PRIA_CHECK(cudaMemcpyAsync(out_x, c.outfull, full * sizeof(real),
                                   cudaMemcpyDeviceToHost, c.stream));
    }
    if (out_y) {
        k_sparsify<real, 1><<<fgrid, kBlock, 0, c.stream>>>(
            d, t, (const real*)fin.y, (const real*)c.invk, invk0, c.outfull);
        PRIA_CHECK(cudaMemcpyAsync(out_y, c.outfull, full * sizeof(real),
                                   cudaMemcpyDeviceToHost, c.stream));
    }
    if (out_z) {
        k_sparsify<real, 2><<<fgrid, kBlock, 0, c.stream>>>(
            d, t, (const real*)fin.z, (const real*)c.invk, invk0, c.outfull);
        PRIA_CHECK(cudaMemcpyAsync(out_z, c.outfull, full * sizeof(real),
                                   cudaMemcpyDeviceToHost, c.stream));
    }
    PRIA_CHECK(cudaStreamSynchronize(c.stream));

    if (info) {
        info->iterations = iteration;
        info->primal_objective = gi.primal;
        info->dual_objective = gi.dual;
        info->relative_gap = gi.gap;
    }
    return cudaSuccess;
}

} // namespace pria

#endif /* PRIA_SOLVER_CUH */
