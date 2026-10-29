/* projection.cuh - batched sort-based simplex projections.
 *
 * Projects every segment of a dense buffer onto its simplex {v >= 0,
 * sum v = target} with the classic sort-and-threshold algorithm (Blondel et
 * al.), exactly as the JAX implementation's double_simplex_projection:
 *
 *   u = sort_descending(segment)
 *   S_k = u_1 + ... + u_k
 *   rho = #{k : u_k * k > S_k - target}     (the condition is prefix-true)
 *   theta = (S_rho - target) / rho
 *   out = max(v - theta, 0)
 *
 * Pipeline per projection:
 *   1. cub::DeviceSegmentedSort::SortKeysDescending over all segments
 *   2. segmented inclusive scan of the sorted keys: cub::DeviceScan with
 *      head-flag (flag, sum) pairs; flags come from the analytic geometry
 *   3. count rho per segment (warp-aggregated integer atomics, deterministic)
 *   4. theta per segment (+inf when rho = 0, i.e. target <= 0, so the
 *      projection onto the degenerate simplex {0} comes out as exact zeros)
 *   5. clamp the ORIGINAL values: out = max(pre - theta[seg], 0)
 */
#ifndef PRIA_PROJECTION_CUH
#define PRIA_PROJECTION_CUH

#include <cub/cub.cuh>
#include <cstdlib>
#include <iterator>
#include <limits>

#include "common.cuh"
#include "layouts.cuh"

namespace pria {

template <typename real>
struct SegPair {
    int flag;
    real sum;
};

template <typename real>
struct SegScanOp {
    __host__ __device__ SegPair<real> operator()(const SegPair<real>& a,
                                                 const SegPair<real>& b) const {
        SegPair<real> r;
        r.flag = a.flag | b.flag;
        r.sum = b.flag ? b.sum : a.sum + b.sum;
        return r;
    }
};

/* Random-access input iterator feeding (segment-start flag, sorted value)
 * pairs to cub::DeviceScan. */
template <typename real, typename Geom>
struct PairLoadIter {
    using iterator_category = std::random_access_iterator_tag;
    using value_type = SegPair<real>;
    using difference_type = i64;
    using pointer = void;
    using reference = SegPair<real>;

    const real* v;
    Geom geom;
    i64 base;

    __host__ __device__ SegPair<real> operator[](i64 idx) const {
        i64 g = base + idx;
        SegPos p = geom.pos(g);
        SegPair<real> r;
        r.flag = (p.klocal == 1) ? 1 : 0;
        r.sum = v[g];
        return r;
    }
    __host__ __device__ SegPair<real> operator*() const { return (*this)[0]; }
    __host__ __device__ PairLoadIter operator+(i64 o) const {
        return PairLoadIter{v, geom, base + o};
    }
    __host__ __device__ PairLoadIter& operator+=(i64 o) {
        base += o;
        return *this;
    }
    __host__ __device__ PairLoadIter& operator++() {
        ++base;
        return *this;
    }
};

/* rho[seg] += #{g : u_g * klocal > S_g - target}; integer atomics are exact
 * and order-independent, so the count is deterministic. Lanes that share a
 * segment combine their votes before touching global memory. */
template <typename real, typename Geom>
__global__ void k_count_rho(i64 n, const real* sorted,
                            const SegPair<real>* pairs, Geom geom,
                            const real* k, real k0, int* rho) {
    i64 stride = (i64)gridDim.x * blockDim.x;
    i64 start = blockIdx.x * (i64)blockDim.x + threadIdx.x;
    for (i64 g = start; g - threadIdx.x % 32 < n; g += stride) {
        bool active = g < n;
        i64 seg = -1;
        bool cond = false;
        if (active) {
            SegPos p = geom.pos(g);
            real tgt = geom.target(p.seg, k, k0);
            cond = sorted[g] * (real)p.klocal > pairs[g].sum - tgt;
            seg = p.seg;
        }
        unsigned peers = __match_any_sync(0xffffffffu, seg);
        int votes = __popc(__ballot_sync(0xffffffffu, cond) & peers);
        int lane = threadIdx.x % 32;
        if (active && votes > 0 && lane == (__ffs(peers) - 1))
            atomicAdd(&rho[seg], votes);
    }
}

template <typename real, typename Geom>
__global__ void k_theta(i64 nseg, const SegPair<real>* pairs, const int* offs,
                        const int* rho, Geom geom, const real* k, real k0,
                        real* theta) {
    i64 stride = (i64)gridDim.x * blockDim.x;
    for (i64 seg = blockIdx.x * (i64)blockDim.x + threadIdx.x; seg < nseg;
         seg += stride) {
        int r = rho[seg];
        real th;
        if (r > 0) {
            real s_rho = pairs[(i64)offs[seg] + r - 1].sum;
            th = (s_rho - geom.target(seg, k, k0)) / (real)r;
        } else {
            th = std::numeric_limits<real>::infinity();
        }
        theta[seg] = th;
    }
}

template <typename real, typename Geom>
__global__ void k_clamp(i64 n, const real* pre, const real* theta, Geom geom,
                        real* out) {
    i64 stride = (i64)gridDim.x * blockDim.x;
    for (i64 g = blockIdx.x * (i64)blockDim.x + threadIdx.x; g < n;
         g += stride) {
        real v = pre[g] - theta[geom.pos(g).seg];
        out[g] = v > real(0) ? v : real(0);
    }
}

/* ---- fused per-row projection ------------------------------------------ *
 *
 * The column/row layouts put exactly two segments, back to back, into every
 * dense row of W = T+1 elements. When a row fits into one thread block's
 * registers, the whole projection of that row - both segment sorts
 * (cub::BlockRadixSort), the prefix sums, the rho counts, the thresholds and
 * the clamp - runs on chip in a single kernel. This replaces the device-wide
 * sort + scan + count + threshold + clamp pipeline for the x and y
 * projections; the z projection (whose segments couple all items) always
 * uses the device-wide path.
 */
/* mode: 0 = x (group = dense row, split j+1, target 1), 1 = y (group =
 * dense row, split T-j, target k[i]), 2 = z (group = dense slice of width
 * W*N, split (T-j)*N, target k0). */
template <typename real, int ITEMS>
__global__ void k_fused_row_project(i64 n_groups, i64 width, i64 P, i64 T,
                                    i64 N, int mode, const real* pre,
                                    const real* k, real k0, real* out) {
    constexpr int BLOCK = 256;
    using Sort = cub::BlockRadixSort<real, BLOCK, ITEMS>;
    using Scan = cub::BlockScan<real, BLOCK>;
    using Reduce = cub::BlockReduce<int, BLOCK>;
    __shared__ union {
        typename Sort::TempStorage sort;
        typename Scan::TempStorage scan;
        typename Reduce::TempStorage red;
    } tmp;
    __shared__ real sh_theta;
    __shared__ int sh_rho;

    const i64 row = blockIdx.x;
    if (row >= n_groups) return;
    const i64 j = mode == 2 ? row : row % P;
    const i64 base = row * width;
    const i64 split = mode == 0 ? (j + 1)
                    : mode == 1 ? (T - j)
                                : (T - j) * N;
    const real target = mode == 0 ? real(1) : (mode == 1 ? k[row / P] : k0);
    const real NEG = -std::numeric_limits<real>::infinity();

    for (int seg = 0; seg < 2; ++seg) {
        const i64 s0 = base + (seg ? split : 0);
        const i64 len = seg ? (width - split) : split;

        /* striped (coalesced) load; the input arrangement is irrelevant to
         * the sorted result */
        real v[ITEMS];
#pragma unroll
        for (int i = 0; i < ITEMS; ++i) {
            i64 r = threadIdx.x + (i64)i * BLOCK;
            v[i] = r < len ? pre[s0 + r] : NEG;
        }
        __syncthreads();
        Sort(tmp.sort).SortDescending(v);
        /* v is now BLOCKED: rank = threadIdx.x*ITEMS + i, descending, with
         * the -inf padding in ranks >= len */

        real masked[ITEMS], local = real(0);
#pragma unroll
        for (int i = 0; i < ITEMS; ++i) {
            i64 r = (i64)threadIdx.x * ITEMS + i;
            masked[i] = r < len ? v[i] : real(0);
            local += masked[i];
        }
        real prefix;
        __syncthreads();
        Scan(tmp.scan).ExclusiveSum(local, prefix);

        int cnt = 0;
        real S[ITEMS], run = prefix;
#pragma unroll
        for (int i = 0; i < ITEMS; ++i) {
            run += masked[i];
            S[i] = run;
            i64 r = (i64)threadIdx.x * ITEMS + i;
            if (r < len && v[i] * (real)(r + 1) > S[i] - target) ++cnt;
        }
        __syncthreads();
        int rho = Reduce(tmp.red).Sum(cnt);
        if (threadIdx.x == 0) {
            sh_rho = rho;
            sh_theta = std::numeric_limits<real>::infinity();
        }
        __syncthreads();
        rho = sh_rho;
#pragma unroll
        for (int i = 0; i < ITEMS; ++i) {
            i64 r = (i64)threadIdx.x * ITEMS + i;
            if (rho > 0 && r + 1 == rho)
                sh_theta = (S[i] - target) / (real)rho; /* single writer */
        }
        __syncthreads();
        const real theta = sh_theta;
#pragma unroll
        for (int i = 0; i < ITEMS; ++i) {
            i64 r = threadIdx.x + (i64)i * BLOCK;
            if (r < len) {
                real o = pre[s0 + r] - theta;
                out[s0 + r] = o > real(0) ? o : real(0);
            }
        }
        __syncthreads();
    }
}

/* Set by tests to force the device-wide pipeline for rowwise geometries. */
inline bool& fused_projection_disabled() {
    static bool disabled = false;
    return disabled;
}

/* Sort-free Michelot projections are the default; PRIA_SORTFREE=0 restores
 * the sort-based pipelines, which are kept as a reference backend. */
inline int& sortfree_mode() {
    static int mode = [] {
        const char* e = std::getenv("PRIA_SORTFREE");
        return e ? std::atoi(e) : 1;
    }();
    return mode;
}

/* PRIA_SF_WARM=0 disables warm-started thresholds in the device-wide
 * sort-free projection (each call then solves cold from the full active
 * set). */
inline int& sf_warm_mode() {
    static int mode = [] {
        const char* e = std::getenv("PRIA_SF_WARM");
        return e ? std::atoi(e) : 1;
    }();
    return mode;
}

/* Active-set compaction in the device-wide sort-free projection.
 * PRIA_SF_COMPACT=1 forces it on, =0 off; unset (-1) enables it when the
 * projected variable outgrows half of L2, so its Michelot rounds stream
 * against the rest of the iteration's working set, or when too few segments
 * fill the GPU, so the rounds are latency and compaction cuts their count.
 * L2-resident with a full grid, the extra write pass is a net loss. */
inline int& sf_compact_mode() {
    static int mode = [] {
        const char* e = std::getenv("PRIA_SF_COMPACT");
        return e ? std::atoi(e) : -1;
    }();
    return mode;
}

/* PRIA_SF_STATS=1 counts device-wide Michelot calls / rounds / warm-start
 * fallbacks (printed at solve teardown). */
inline bool sf_stats_enabled() {
    static bool on = std::getenv("PRIA_SF_STATS") != nullptr;
    return on;
}

/* ---- sort-free (Michelot) projections ----------------------------------- *
 *
 * The simplex threshold theta solves sum(max(v - theta, 0)) = target, and
 * Michelot's finite algorithm finds it without sorting: starting from the
 * full segment, repeatedly average the active set and drop everything at or
 * below the average,
 *
 *   theta <- (sum_{v > theta} v - target) / #{v > theta}
 *
 * until the active count stops shrinking. theta is monotone non-decreasing
 * and the count strictly decreases every continued round, so it terminates
 * in at most len iterations (typically a handful). Each round is one strided
 * pass plus a block reduction in a fixed order, so the result is
 * deterministic - but it is NOT bitwise equal to the sort pipeline's theta
 * (different summation order), so PRIA_SORTFREE changes iterate trajectories
 * at the last-ulp level while converging to the same solution. */
template <typename real>
struct SumCnt {
    real sum;
    int cnt;
};

template <typename real>
struct SumCntOp {
    __device__ SumCnt<real> operator()(const SumCnt<real>& a,
                                       const SumCnt<real>& b) const {
        return SumCnt<real>{a.sum + b.sum, a.cnt + b.cnt};
    }
};

/* Threshold for one segment v[0..len); v may live in shared or global
 * memory. Must be called by the whole block; the returned theta is valid in
 * every thread. target <= 0 (the degenerate simplex {0}) yields +inf, so the
 * caller's clamp produces exact zeros, matching the sort pipeline.
 *
 * theta0 warm-starts the iteration: any finite theta0 <= theta* keeps the
 * monotone convergence (its active set is a superset of the true support),
 * and the first round doubles as validation - f(theta0) = sum_{v>theta0} v
 * - count*theta0 - target < 0 means theta0 overshot, so the round is
 * discarded and the iteration restarts cold. Pass -inf for a plain cold
 * start. rounds/fellback (optional, thread 0 only) report effort. */
template <typename real>
__device__ real block_michelot(const real* v, i64 len, real target,
                               real theta0, int* rounds, int* fellback) {
    using Reduce = cub::BlockReduce<SumCnt<real>, kBlock>;
    __shared__ typename Reduce::TempStorage tmp;
    __shared__ real sh_theta;
    __shared__ int sh_stop;
    const real NEG = -std::numeric_limits<real>::infinity();

    if (target <= real(0)) return std::numeric_limits<real>::infinity();

    real theta = theta0;
    int prev = std::numeric_limits<int>::max();
    bool validate = theta0 > NEG;
    for (;;) {
        SumCnt<real> part{real(0), 0};
        for (i64 r = threadIdx.x; r < len; r += kBlock) {
            real x = v[r];
            if (x > theta) {
                part.sum += x;
                part.cnt += 1;
            }
        }
        __syncthreads(); /* tmp/sh_* reuse across rounds and segments */
        SumCnt<real> tot = Reduce(tmp).Reduce(part, SumCntOp<real>{});
        if (threadIdx.x == 0) {
            if (rounds) ++*rounds;
            int stop = 0;
            real th;
            if (validate && (tot.cnt == 0 ||
                             tot.sum - (real)tot.cnt * theta < target)) {
                /* f(theta0) < 0: warm start above the answer; restart cold
                 * (prev is still INT_MAX, so the next round is a fresh
                 * first round) */
                th = NEG;
                if (fellback) *fellback = 1;
            } else if (tot.cnt == 0) { /* len 0 (target <= 0 handled) */
                th = std::numeric_limits<real>::infinity();
                stop = 1;
            } else {
                th = (tot.sum - target) / (real)tot.cnt;
                /* fixed point: the active set (nested, comparison-stable)
                 * kept its size, so this theta reproduces itself */
                stop = (tot.cnt >= prev);
                prev = tot.cnt;
            }
            sh_theta = th;
            sh_stop = stop;
        }
        __syncthreads();
        theta = sh_theta;
        if (sh_stop) return theta;
        validate = false;
    }
}

/* Sort-free device-wide projection: one block per segment. Short row
 * segments stay L1-resident across Michelot rounds; the wide z segments
 * stream from L2/DRAM. Uses no CUB scratch.
 *
 * warm (nullable) holds each segment's previously converged theta; the seed
 * backs off by 1% relative so the usual small drift of theta* between PDHG
 * iterations stays above the seed (a seed above theta* costs a wasted
 * validation round plus a cold solve). The converged theta is written
 * back.
 *
 * scratch (nullable, ND reals, per-segment region [s0,s1)) enables
 * active-set compaction: the first round with a finite, validated theta
 * also writes its actives - in segment order, via a tile-wise block scan,
 * so sums stay deterministic - into scratch, and later rounds iterate over
 * that (shrinking) prefix instead of re-reading the full segment. theta
 * only rises, so actives only shrink and the compacted set stays a
 * superset of every later active set. Never during a warm validation
 * round: a failed validation would leave the wrong actives behind.
 *
 * stats (nullable): {calls, rounds, fallbacks}. */
template <typename real, typename Geom>
__global__ void k_seg_project_sf(i64 nseg, const real* pre, const int* offs,
                                 Geom geom, const real* k, real k0,
                                 real* warm, real* scratch,
                                 unsigned long long* stats, real* out) {
    using Reduce = cub::BlockReduce<SumCnt<real>, kBlock>;
    using Scan = cub::BlockScan<int, kBlock>;
    __shared__ union {
        typename Reduce::TempStorage red;
        typename Scan::TempStorage scan;
    } tmp;
    __shared__ real sh_theta;
    __shared__ int sh_stop;
    __shared__ int sh_cnt;
    __shared__ int sh_base;

    const i64 seg = blockIdx.x;
    if (seg >= nseg) return;
    const i64 s0 = offs[seg], s1 = offs[seg + 1];
    const i64 len = s1 - s0;
    const real target = geom.target(seg, k, k0);
    const real INF = std::numeric_limits<real>::infinity();
    const real NEG = -INF;
    const real* v = pre + s0;

    real theta;
    int rounds = 0, fellback = 0;
    if (target <= real(0))
        theta = INF;
    else {
        real theta0 = NEG;
        if (warm) {
            real w = warm[seg];
            if (isfinite(w)) theta0 = w - fabs(w) * real(0.01);
        }
        theta = theta0;
        int prev = std::numeric_limits<int>::max();
        bool validate = theta0 > NEG;
        bool compacted = false;
        const real* src = v;
        i64 slen = len;
        for (;;) {
            const bool do_compact =
                scratch && !compacted && !validate && isfinite(theta);
            SumCnt<real> part{real(0), 0};
            if (do_compact) {
                if (threadIdx.x == 0) sh_base = 0;
                __syncthreads();
                for (i64 t0 = 0; t0 < slen; t0 += kBlock) {
                    i64 r = t0 + threadIdx.x;
                    real x = r < slen ? src[r] : real(0);
                    int active = (r < slen && x > theta) ? 1 : 0;
                    int pos, tile_total;
                    Scan(tmp.scan).ExclusiveSum(active, pos, tile_total);
                    if (active) {
                        scratch[s0 + sh_base + pos] = x;
                        part.sum += x;
                        part.cnt += 1;
                    }
                    __syncthreads();
                    if (threadIdx.x == 0) sh_base += tile_total;
                    __syncthreads();
                }
            } else {
                for (i64 r = threadIdx.x; r < slen; r += kBlock) {
                    real x = src[r];
                    if (x > theta) {
                        part.sum += x;
                        part.cnt += 1;
                    }
                }
            }
            __syncthreads(); /* tmp/sh_* reuse across rounds */
            SumCnt<real> tot = Reduce(tmp.red).Reduce(part, SumCntOp<real>{});
            if (threadIdx.x == 0) {
                ++rounds;
                int stop = 0;
                real th;
                if (validate && (tot.cnt == 0 ||
                                 tot.sum - (real)tot.cnt * theta < target)) {
                    /* f(theta0) < 0: warm start above the answer; restart
                     * cold (prev is still INT_MAX, so the next round is a
                     * fresh first round) */
                    th = NEG;
                    fellback = 1;
                } else if (tot.cnt == 0) { /* len 0 (target <= 0 handled) */
                    th = INF;
                    stop = 1;
                } else {
                    th = (tot.sum - target) / (real)tot.cnt;
                    /* fixed point: the active set (nested, comparison-
                     * stable) kept its size, so this theta reproduces
                     * itself */
                    stop = (tot.cnt >= prev);
                    prev = tot.cnt;
                }
                sh_theta = th;
                sh_stop = stop;
                sh_cnt = tot.cnt;
            }
            __syncthreads();
            theta = sh_theta;
            if (sh_stop) break;
            if (do_compact) {
                compacted = true;
                src = scratch + s0;
                slen = sh_cnt;
            }
            validate = false;
        }
    }
    if (threadIdx.x == 0) {
        if (warm) warm[seg] = theta;
        if (stats) {
            atomicAdd(&stats[0], 1ull);
            atomicAdd(&stats[1], (unsigned long long)rounds);
            if (fellback) atomicAdd(&stats[2], 1ull);
        }
    }
    for (i64 r = s0 + threadIdx.x; r < s1; r += kBlock) {
        real o = pre[r] - theta;
        out[r] = o > real(0) ? o : real(0);
    }
}

/* Sort-free twin of k_fused_pre_project: the pre row is computed once into
 * dynamic shared memory (no register tiling, so one instantiation covers
 * every width up to the 48KB shared cap), then each segment gets its
 * Michelot threshold and clamp from shared. */
/* One group's pre+projection, callable per row from any kernel (the plain
 * wrapper below and the persistent megakernel); srow is caller-provided
 * shared memory of width reals. Must be called by the whole block. */
template <typename real>
__device__ void dev_fused_pre_row(i64 row, i64 width, const Dims& d, int mode,
                                  int gathered, const real* v, const real* a1,
                                  const real* a2, const real* a3,
                                  const real* scal, const real* k, real k0,
                                  real* out, real* srow) {
    const i64 j = mode == 2 ? row : row % d.P;
    const i64 base = row * width;
    const i64 split = mode == 0 ? (j + 1)
                    : mode == 1 ? (d.T - j)
                                : (d.T - j) * d.N;
    const real target = mode == 0 ? real(1) : (mode == 1 ? k[row / d.P] : k0);
    const real tau = mode == 0 ? scal[0] : real(0);
    const real sigma = gathered == 2 ? scal[1] : real(0);

    for (i64 r = threadIdx.x; r < width; r += kBlock) {
        i64 g = base + r;
        real val;
        if (mode == 0) {
            real ys = gathered ? a2[colg_to_rowg(d, g)] : a2[g];
            real zs = gathered ? a3[colg_to_d2g(d, g)] : a3[g];
            val = v[g] - tau * (a1[g] + ys + zs);
        } else if (mode == 1) {
            i64 cg = gathered ? rowg_to_colg(d, g) : g;
            val = v[g] + (gathered == 2
                              ? sigma * (real(2) * a1[cg] - a2[cg])
                              : a1[cg]);
        } else {
            i64 cg = gathered ? d2g_to_colg(d, g) : g;
            val = v[g] + (gathered == 2
                              ? sigma * (real(2) * a1[cg] - a2[cg])
                              : a1[cg]);
        }
        srow[r] = val;
    }
    __syncthreads();

    for (int seg = 0; seg < 2; ++seg) {
        const i64 s0 = seg ? split : 0;
        const i64 len = seg ? (width - split) : split;
        const real theta =
            block_michelot(srow + s0, len, target,
                           -std::numeric_limits<real>::infinity(), nullptr,
                           nullptr);
        for (i64 r = threadIdx.x; r < len; r += kBlock) {
            real o = srow[s0 + r] - theta;
            out[base + s0 + r] = o > real(0) ? o : real(0);
        }
        __syncthreads(); /* srow reads done before the next segment's rounds */
    }
}

template <typename real>
__global__ void k_fused_pre_project_sf(i64 n_groups, i64 width, Dims d,
                                       int mode, int gathered, const real* v,
                                       const real* a1, const real* a2,
                                       const real* a3, const real* scal,
                                       const real* k, real k0, real* out) {
    extern __shared__ unsigned char smem[];
    real* srow = reinterpret_cast<real*>(smem);
    const i64 row = blockIdx.x;
    if (row >= n_groups) return;
    dev_fused_pre_row(row, width, d, mode, gathered, v, a1, a2, a3, scal, k,
                      k0, out, srow);
}

/* ---- fused pre + projection -------------------------------------------- *
 *
 * Computes the pre-projection values inline (striped, once, into a shared
 * row) instead of reading a materialized pre array twice, eliminating the
 * intermediate array's global round trip. mode as in k_fused_row_project;
 * gathered selects whether the addend sources are read through the layout
 * converters (plain duals / reflected term) or directly (mirrored buffers);
 * gathered == 2 (modes 1/2, sort-free path) additionally computes the
 * reflected term inline from a1 = xnew, a2 = xold instead of reading a
 * materialized reflection.
 *   mode 0: pre = v - tau*(a1 + ys + zs)   (v = x, a1 = h, ys/zs from a2/a3)
 *   mode 1: pre = v + rs                   (v = y, rs from a1)
 *   mode 2: pre = v + rs                   (v = z, rs from a1)
 */
template <typename real, int ITEMS>
__global__ void k_fused_pre_project(i64 n_groups, i64 width, Dims d, int mode,
                                    int gathered, const real* v,
                                    const real* a1, const real* a2,
                                    const real* a3, const real* scal,
                                    const real* k, real k0, real* out) {
    constexpr int BLOCK = 256;
    using Sort = cub::BlockRadixSort<real, BLOCK, ITEMS>;
    using Scan = cub::BlockScan<real, BLOCK>;
    using Reduce = cub::BlockReduce<int, BLOCK>;
    __shared__ union {
        typename Sort::TempStorage sort;
        typename Scan::TempStorage scan;
        typename Reduce::TempStorage red;
    } tmp;
    __shared__ real srow[BLOCK * ITEMS];
    __shared__ real sh_theta;
    __shared__ int sh_rho;

    const i64 row = blockIdx.x;
    if (row >= n_groups) return;
    const i64 j = mode == 2 ? row : row % d.P;
    const i64 base = row * width;
    const i64 split = mode == 0 ? (j + 1)
                    : mode == 1 ? (d.T - j)
                                : (d.T - j) * d.N;
    const real target = mode == 0 ? real(1) : (mode == 1 ? k[row / d.P] : k0);
    const real NEG = -std::numeric_limits<real>::infinity();
    const real tau = mode == 0 ? scal[0] : real(0);

    /* compute the pre-projection row once, striped, into shared */
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        i64 r = threadIdx.x + (i64)i * BLOCK;
        if (r < width) {
            i64 g = base + r;
            real val;
            if (mode == 0) {
                real ys = gathered ? a2[colg_to_rowg(d, g)] : a2[g];
                real zs = gathered ? a3[colg_to_d2g(d, g)] : a3[g];
                val = v[g] - tau * (a1[g] + ys + zs);
            } else if (mode == 1) {
                val = v[g] + (gathered ? a1[rowg_to_colg(d, g)] : a1[g]);
            } else {
                val = v[g] + (gathered ? a1[d2g_to_colg(d, g)] : a1[g]);
            }
            srow[r] = val;
        }
    }
    __syncthreads();

    for (int seg = 0; seg < 2; ++seg) {
        const i64 s0 = seg ? split : 0;
        const i64 len = seg ? (width - split) : split;

        real vv[ITEMS];
#pragma unroll
        for (int i = 0; i < ITEMS; ++i) {
            i64 r = threadIdx.x + (i64)i * BLOCK;
            vv[i] = r < len ? srow[s0 + r] : NEG;
        }
        __syncthreads();
        Sort(tmp.sort).SortDescending(vv);
        /* vv is now BLOCKED: rank = threadIdx.x*ITEMS + i, descending */

        real masked[ITEMS], local = real(0);
#pragma unroll
        for (int i = 0; i < ITEMS; ++i) {
            i64 r = (i64)threadIdx.x * ITEMS + i;
            masked[i] = r < len ? vv[i] : real(0);
            local += masked[i];
        }
        real prefix;
        __syncthreads();
        Scan(tmp.scan).ExclusiveSum(local, prefix);

        int cnt = 0;
        real S[ITEMS], run = prefix;
#pragma unroll
        for (int i = 0; i < ITEMS; ++i) {
            run += masked[i];
            S[i] = run;
            i64 r = (i64)threadIdx.x * ITEMS + i;
            if (r < len && vv[i] * (real)(r + 1) > S[i] - target) ++cnt;
        }
        __syncthreads();
        int rho = Reduce(tmp.red).Sum(cnt);
        if (threadIdx.x == 0) {
            sh_rho = rho;
            sh_theta = std::numeric_limits<real>::infinity();
        }
        __syncthreads();
        rho = sh_rho;
#pragma unroll
        for (int i = 0; i < ITEMS; ++i) {
            i64 r = (i64)threadIdx.x * ITEMS + i;
            if (rho > 0 && r + 1 == rho)
                sh_theta = (S[i] - target) / (real)rho; /* single writer */
        }
        __syncthreads();
        const real theta = sh_theta;
#pragma unroll
        for (int i = 0; i < ITEMS; ++i) {
            i64 r = threadIdx.x + (i64)i * BLOCK;
            if (r < len) {
                real o = srow[s0 + r] - theta;
                out[base + s0 + r] = o > real(0) ? o : real(0);
            }
        }
        __syncthreads();
    }
}

/* Dynamic-shared budget for the sort-free fused kernel, at the cap that
 * needs no opt-in. Opting in to a larger one would fuse wider rows, but at
 * one block per SM and without the warm-threshold, compaction and overlap
 * paths of the segment kernel it displaces. */
template <typename real>
inline size_t fused_sf_shared_cap() {
    return 48 * 1024 - 1024;
}

/* Dispatch the fused pre+projection when every group fits one block tile;
 * returns false if the caller must materialize pre and use the pipelines. */
template <typename real>
bool try_fused_pre(const Dims& d, int mode, int gathered, const real* v,
                   const real* a1, const real* a2, const real* a3,
                   const real* scal, const real* d_k, real k0, real* out,
                   cudaStream_t stream) {
    if (fused_projection_disabled()) return false;
    const i64 groups = mode == 2 ? d.P : d.N * d.P;
    const i64 width = mode == 2 ? d.W * d.N : d.W;

    if (sortfree_mode()) {
        const size_t shbytes = (size_t)width * sizeof(real);
        if (shbytes > fused_sf_shared_cap<real>()) return false;
        k_fused_pre_project_sf<real>
            <<<(unsigned)groups, kBlock, shbytes, stream>>>(
                groups, width, d, mode, gathered, v, a1, a2, a3, scal, d_k,
                k0, out);
        return true;
    }

    if (width <= 256 * 2)
        k_fused_pre_project<real, 2><<<(unsigned)groups, 256, 0, stream>>>(
            groups, width, d, mode, gathered, v, a1, a2, a3, scal, d_k, k0,
            out);
    else if (width <= 256 * 4)
        k_fused_pre_project<real, 4><<<(unsigned)groups, 256, 0, stream>>>(
            groups, width, d, mode, gathered, v, a1, a2, a3, scal, d_k, k0,
            out);
    else if (width <= 256 * 8)
        k_fused_pre_project<real, 8><<<(unsigned)groups, 256, 0, stream>>>(
            groups, width, d, mode, gathered, v, a1, a2, a3, scal, d_k, k0,
            out);
    else if constexpr (sizeof(real) == 4) {
        if (width <= 256 * 16)
            k_fused_pre_project<real, 16><<<(unsigned)groups, 256, 0,
                                            stream>>>(
                groups, width, d, mode, gathered, v, a1, a2, a3, scal, d_k,
                k0, out);
        else
            return false;
    } else {
        return false;
    }
    return true;
}

/* Scratch shared by all three projections of a solver (they run one after
 * another on the same stream). */
template <typename real>
struct ProjWorkspace {
    real* sorted = nullptr;         /* ND */
    SegPair<real>* pairs = nullptr; /* ND */
    int* rho = nullptr;             /* max segments */
    real* theta = nullptr;          /* max segments */
    void* cub_tmp = nullptr;
    size_t cub_bytes = 0;
    /* z sort-order probe (PRIA_ZSORT_PROBE=1): measures how often the
     * previous sort's permutation still sorts the current values */
    bool zprobe = false;
    int* perm = nullptr;               /* ND */
    int* iota = nullptr;               /* ND */
    int* sortflag = nullptr;           /* 1 */
    unsigned long long* zstats = nullptr; /* {hits, misses} */
    /* device-wide Michelot counters (PRIA_SF_STATS=1):
     * {calls, rounds, warm fallbacks} */
    unsigned long long* sfstats = nullptr;
};

template <typename real>
__global__ void k_fill(i64 n, real val, real* out) {
    i64 stride = (i64)gridDim.x * blockDim.x;
    for (i64 g = blockIdx.x * (i64)blockDim.x + threadIdx.x; g < n;
         g += stride)
        out[g] = val;
}

/* out[g] = pre[perm[g]] - the previous sort order applied to fresh values */
template <typename real>
__global__ void k_perm_gather(i64 n, const real* pre, const int* perm,
                              real* out) {
    i64 stride = (i64)gridDim.x * blockDim.x;
    for (i64 g = blockIdx.x * (i64)blockDim.x + threadIdx.x; g < n;
         g += stride)
        out[g] = pre[perm[g]];
}

/* flag = 1 if any within-segment neighbor pair is ascending */
template <typename real, typename Geom>
__global__ void k_check_desc(i64 n, const real* v, Geom geom, int* flag) {
    i64 stride = (i64)gridDim.x * blockDim.x;
    for (i64 g = blockIdx.x * (i64)blockDim.x + threadIdx.x; g + 1 < n;
         g += stride)
        if (v[g] < v[g + 1] && geom.pos(g).seg == geom.pos(g + 1).seg)
            *flag = 1;
}

static __global__ void k_probe_tally(const int* flag,
                                     unsigned long long* stats) {
    ++stats[*flag ? 1 : 0];
}

static __global__ void k_iota(i64 n, int* out) {
    i64 stride = (i64)gridDim.x * blockDim.x;
    for (i64 g = blockIdx.x * (i64)blockDim.x + threadIdx.x; g < n;
         g += stride)
        out[g] = (int)g;
}

/* Run one projection: pre (dense buffer) -> out = proj(pre). pre and out may
 * not alias; pre is left untouched. */
/* Launch the fused per-group kernel when every group (dense row for x/y,
 * dense slice for z) fits in one block's tile; returns false if the caller
 * must use the device-wide pipeline. */
template <typename real, typename Geom>
bool try_fused_projection(const real* pre, real* out, const Geom& geom,
                          const real* d_k, real k0, cudaStream_t stream) {
    if (fused_projection_disabled()) return false;
    const Dims& d = geom.d;
    const int mode = Geom::kRowwise ? (Geom::kIsX ? 0 : 1) : 2;
    const i64 groups = Geom::kRowwise ? d.N * d.P : d.P;
    const i64 width = Geom::kRowwise ? d.W : d.W * d.N;

    if (width <= 256 * 2)
        k_fused_row_project<real, 2><<<(unsigned)groups, 256, 0, stream>>>(
            groups, width, d.P, d.T, d.N, mode, pre, d_k, k0, out);
    else if (width <= 256 * 4)
        k_fused_row_project<real, 4><<<(unsigned)groups, 256, 0, stream>>>(
            groups, width, d.P, d.T, d.N, mode, pre, d_k, k0, out);
    else if (width <= 256 * 8)
        k_fused_row_project<real, 8><<<(unsigned)groups, 256, 0, stream>>>(
            groups, width, d.P, d.T, d.N, mode, pre, d_k, k0, out);
    else if constexpr (sizeof(real) == 4) {
        if (width <= 256 * 16)
            k_fused_row_project<real, 16><<<(unsigned)groups, 256, 0,
                                            stream>>>(
                groups, width, d.P, d.T, d.N, mode, pre, d_k, k0, out);
        else
            return false;
    } else {
        return false;
    }
    return true;
}

/* warm (nullable): per-segment threshold store for the sort-free path,
 * owned by the caller and matched to this geometry's segments; ignored by
 * the sort backend. */
template <typename real, typename Geom>
cudaError_t project_segments(ProjWorkspace<real>& ws, const real* pre,
                             real* out, const Geom& geom, const int* d_offs,
                             real* warm, const real* d_k, real k0,
                             cudaStream_t stream) {
    const i64 n = geom.d.ND;
    const i64 nseg = geom.num_segments();

    if (sortfree_mode()) {
        const int cm = sf_compact_mode();
        const bool compact =
            cm == 1 ||
            (cm < 0 && (n * (i64)sizeof(real) > (i64)48 * 1024 * 1024 ||
                        nseg < 256));
        k_seg_project_sf<<<(unsigned)nseg, kBlock, 0, stream>>>(
            nseg, pre, d_offs, geom, d_k, k0,
            sf_warm_mode() ? warm : (real*)nullptr,
            compact ? ws.sorted : (real*)nullptr, ws.sfstats, out);
        return cudaPeekAtLastError();
    }

    if (try_fused_projection(pre, out, geom, d_k, k0, stream))
        return cudaPeekAtLastError();

    size_t bytes = ws.cub_bytes;
    if (!Geom::kRowwise && ws.zprobe) {
        /* probe: does the previous sort's permutation still sort pre?
         * (Counts feed the sort-reuse decision; the sort still runs, as
         * SortPairs to keep the permutation fresh. Keys out are identical
         * to SortKeys, so results are unchanged.) */
        const int g2 = launch_grid(n);
        k_perm_gather<<<g2, kBlock, 0, stream>>>(n, pre, (const int*)ws.perm,
                                                 ws.sorted);
        PRIA_CHECK(cudaMemsetAsync(ws.sortflag, 0, sizeof(int), stream));
        k_check_desc<<<g2, kBlock, 0, stream>>>(n, (const real*)ws.sorted,
                                                geom, ws.sortflag);
        k_probe_tally<<<1, 1, 0, stream>>>(ws.sortflag, ws.zstats);
        PRIA_CHECK(cub::DeviceSegmentedSort::SortPairsDescending(
            ws.cub_tmp, bytes, pre, ws.sorted, (const int*)ws.iota, ws.perm,
            n, nseg, d_offs, d_offs + 1, stream));
    } else {
        PRIA_CHECK(cub::DeviceSegmentedSort::SortKeysDescending(
            ws.cub_tmp, bytes, pre, ws.sorted, n, nseg, d_offs, d_offs + 1,
            stream));
    }

    PairLoadIter<real, Geom> it{ws.sorted, geom, 0};
    bytes = ws.cub_bytes;
    PRIA_CHECK(cub::DeviceScan::InclusiveScan(ws.cub_tmp, bytes, it, ws.pairs,
                                              SegScanOp<real>{}, n, stream));

    PRIA_CHECK(cudaMemsetAsync(ws.rho, 0, nseg * sizeof(int), stream));
    int grid = launch_grid(n);
    k_count_rho<<<grid, kBlock, 0, stream>>>(n, ws.sorted, ws.pairs, geom, d_k,
                                             k0, ws.rho);
    k_theta<<<launch_grid(nseg), kBlock, 0, stream>>>(
        nseg, ws.pairs, d_offs, ws.rho, geom, d_k, k0, ws.theta);
    k_clamp<<<grid, kBlock, 0, stream>>>(n, pre, ws.theta, geom, out);
    return cudaPeekAtLastError();
}

} // namespace pria

#endif /* PRIA_PROJECTION_CUH */
