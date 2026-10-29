/* PRAQ solver internals: the port of ../praq-jax/praq/ to CUDA.
 *
 * What is ported, and from where:
 *
 *   half_edge.py      the group-indexed lift.  Every edge e = {i, j} owns two
 *                     half-edges, slot 0 at i and slot 1 at j; the dual y lives
 *                     on them as one float2/double2 per edge.  gather is x[end]
 *                     and scatter is the sum of a node's incident half-edges,
 *                     which here walks a CSR incidence list (node_ptr, node_half)
 *                     instead of atomics, so it is deterministic and run to run
 *                     the iterates are bitwise identical.  A node's list is
 *                     split over a group of lanes, sized from the average
 *                     degree (Solver::lanes_for), and summed in a fixed
 *                     butterfly, so the order of additions depends on the
 *                     instance alone, never on timing.  The scatter is the
 *                     one random access per iteration, so what it reads is one
 *                     Real per half-edge, `gather`: the Halpern-combined dual
 *                     the next primal step needs, which the dual kernel
 *                     computes as it finishes the step before.
 *   projection.py     the two closed-form per-edge projections, inline in the
 *                     dual kernel.  No sort.
 *   halpern_operator.py  the canonical norm, assembled on the host from three
 *                     sums the reduction kernels produce.
 *   solver.py         reflected Halpern PDHG, the restart tests and the omega
 *                     update, the bounds and the termination test.  The
 *                     per-iteration work is two kernels; every op_steps
 *                     iterations one reduction pass and one device-to-host copy
 *                     of nine doubles feed the scalar logic, which runs on the
 *                     host exactly as solver.py writes it.
 *
 * The build is SPARSE where praq-jax's is dense.  praq-jax gives every pair
 * i < j an edge so that jit sees static shapes, and a pair with W_ij = 0 is
 * inert -- its dual collapses to (0, 0) under either projection -- except that
 * it counts toward the degree.  Here the instance's own edge list is the
 * half-edge set, so an instance costs O(n + m) and the degree is the true
 * incident count (floored at 1, as in HalfEdge.build).  Both satisfy
 * D >= S^T S, which is all the convergence proof needs, and ||A||_2 = 1 holds
 * for both since D^{-1/2} S^T S D^{-1/2} is the identity on every node that has
 * an edge.  The two therefore converge to the same bounds, and on a dense
 * instance take the same path; on a sparse one this is the better-conditioned
 * of the two and takes fewer iterations.
 *
 * Termination happens on the relative gap AT THE ORIGINAL SCALE: both bounds
 * are multiplied back by the objective scale before forming
 * (upper - lower) / (1 + |lower| + |upper|).  The relative gap is not
 * scale-invariant -- the "1 +" fixes a unit -- so testing it on rescaled
 * bounds would be weaker by a factor approaching the scale itself.
 *
 * Two invariants the tests assert: ||A||_2 = 1 exactly (so eta = 0.999 needs no
 * line search and no power iteration), and both iterates stay feasible, so the
 * bounds bracket the optimum from the first step and not only at convergence.
 *
 * Everything in this header is templated on the real type of the iterates.
 * The reductions accumulate in double regardless, and so does the scalar logic
 * on the host; use_float32 therefore changes the arrays, not the bookkeeping.
 */
#ifndef PRAQ_SOLVER_CUH
#define PRAQ_SOLVER_CUH

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "praq/praq.h"

namespace praq {

/* --- defaults, mirroring praq-jax's Parameters.with_auto_step ------------
 * These are what praq_default_parameters() hands out; the solver itself reads
 * the PraqParameters it is given, which is how scripts/tune_praq.py varies
 * them without a rebuild. */

/* ||A||_2 for the PRAQ coupling, exactly 1 for every graph (see the header
 * comment), so this is the step size for every primal weight. */
constexpr double kEta = 0.999;
/* omega = 1 minimises lambda_max of the canonical metric; the restart logic
 * re-estimates it from the first restart onward. */
constexpr double kOmegaInitial = 1.0;
/* exp(-1): minimises the worst-case complexity of sufficient-decay restarts. */
constexpr double kBetaSufficient = 0.36787944117144233;
/* The rest are PRIA's JRP-tuned values, taken as starting points. */
constexpr double kBetaNecessary = 0.5706;
constexpr double kBetaArtificial = 0.2468;
constexpr double kThetaInitial = 0.81930;
constexpr double kThetaTarget = 0.42818;
constexpr double kThetaDecay = 0.90474;
/* Iterations between two restart-and-termination checks. */
constexpr int kOpSteps = 16;
/* The check-interval rule's defaults (see PraqParameters in praq.h): the
 * logarithm 3 log2(n / 50k) = (3 / ln 2) ln(n / 50k).
 * It only applies when a caller sets op_steps <= 0; the default op_steps
 * above is what praq_default_parameters hands out. */
constexpr double kIntervalNodes = 50000.0;
constexpr double kIntervalOffset = 0.0;
constexpr double kIntervalScale = 4.328085122666891;   /* 3 / ln 2 */
constexpr double kIntervalPower = 0.0;

/* interval(n) of the rule, on the host; praq-jax's check_interval mirrors it
 * expression for expression. */
inline int64_t check_interval(int64_t n, double offset, double scale, double power) {
    double u = std::log(double(n) / kIntervalNodes);
    double b = power == 0.0 ? u : std::expm1(power * u) / power;
    double v = std::floor(offset + scale * b + 0.5);
    if (!(v >= 2.0)) return 2;       /* also catches NaN */
    if (v > 1e9) return 1000000000;
    return int64_t(v);
}

/* --- launch geometry ------------------------------------------------------ */

/* How many half-edges one lane of primal_kernel's gather should walk before
 * the node gets another lane; see Solver::lanes_for. */
constexpr double kHalfEdgesPerLane = 8.0;
/* with_auto_step's warm_steps is 0 and PraqParameters exposes no way to raise
 * it, so the vanilla warm phase of solve_qubo_problem_warm_start is not
 * ported: with no warm steps it is a no-op and leaves omega at kOmegaInitial. */

/* Error codes `praq_solve` returns, all negative. */
constexpr int kErrorArgument = -1;   /* a null pointer, or a bad parameter     */
constexpr int kErrorInstance = -2;   /* an index out of range, a loop, n = 0   */
constexpr int kErrorCuda = -3;       /* any CUDA runtime failure               */

constexpr int kBlock = 256;
/* The reduction kernels are launched with this many blocks and stride over
 * their input; the finalize kernel is one block of the same size, one thread
 * per partial row.  That ties the two, hence the static_assert. */
constexpr int kReduceBlocks = 256;
static_assert(kReduceBlocks == kBlock, "finalize_kernel reads one partial per thread");

/* What one reduction pass produces, in this order. */
enum Sum {
    kPrimalNorm = 0,     /* sum deg (x - sx)^2            canonical norm    */
    kPrimalDrift = 1,    /* sum deg (x - x0)^2            omega update      */
    kUpperNode = 2,      /* sum diag bx                   upper bound       */
    kLowerNode = 3,      /* sum min(diag - S^T by, 0)     lower bound       */
    kDualNorm = 4,       /* sum |y - sy|^2                canonical norm    */
    kCross = 5,          /* sum <S(x - sx), y - sy>       canonical norm    */
    kDualDrift = 6,      /* sum |y - y0|^2                omega update      */
    kUpperEdge = 7,      /* sum w [...] at bx             upper bound       */
    kLowerEdge = 8,      /* sum over covering edges of by  lower bound      */
    kSums = 9
};
constexpr int kNodeSums = 4;
constexpr int kEdgeSums = kSums - kNodeSums;

template <class Real> struct Pair;
template <> struct Pair<double> { using type = double2; };
template <> struct Pair<float> { using type = float2; };

/* --- device kernels ------------------------------------------------------- */

template <class Real>
__device__ inline Real clamp01(Real v) {
    return v < Real(0) ? Real(0) : (v > Real(1) ? Real(1) : v);
}

template <class Real>
__device__ inline Real clamp(Real v, Real lo, Real hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/* project_dual for one edge, from projection.py.  A covering edge (W_ij > 0)
 * lands on the segment {(mu, mu) : -w <= mu <= 0}; a packing edge on the
 * weighted 2-simplex {Y >= 0, Y_1 + Y_2 = w}.  A weight of 0 lands on (0, 0)
 * under either branch. */
template <class Real, class P>
__device__ inline P project_dual(P v, Real w, bool positive) {
    P out;
    if (positive) {
        Real mu = clamp(Real(0.5) * (v.x + v.y), -w, Real(0));
        out.x = mu;
        out.y = mu;
    } else {
        Real line = v.x + Real(0.5) * (w - v.x - v.y);
        Real first = clamp(line, Real(0), w);
        out.x = first;
        out.y = w - first;
    }
    return out;
}

/* The Halpern combination of _reflect_step: (k/(k+1)) (2 T z - z) + (1/(k+1)) z0
 * with k = restart_iter + 1, or T z itself for a vanilla step.  Written out as
 * praq-jax writes it so that the rounding matches. */
template <class Real, bool Reflect>
__device__ inline Real combine(Real z, Real tz, Real z0, Real lw, Real rw) {
    if (Reflect) return lw * (Real(2) * tz - z) + rw * z0;
    return tz;
}

/* combine<Real, true> for the `gather` array, with the fused multiply-add
 * spelled out.  nvcc contracts `a * b + c` on its own and chooses the product
 * to fuse per call site; this is the form it chose when the primal kernel
 * evaluated this expression itself, so the iterates are bitwise what they
 * were.  (The same pinning inside combine() would move the primal's own x
 * combination by an ulp, which it contracted the other way.) */
template <class Real>
__device__ inline Real combine_next(Real z, Real tz, Real z0, Real lw, Real rw) {
    return fma(lw, Real(2) * tz - z, rw * z0);
}

/* Half of one PDHG step (_raw_step): the primal update, with the Halpern
 * combination of the previous iterate folded in.
 *
 *   x1 = combine(x, sx, x0)                  the point the step is taken from
 *   g  = diag - S^T y1                       y1 = combine(y, sy, y0), read from
 *                                            `gather` (see dual_kernel)
 *   x2 = clip(x1 - (eta/omega) g / deg)
 *
 * and afterwards x holds x1 and sx holds x2, written by the group's first
 * lane, which is the only one that touches node i's arrays, so in place.
 * It also leaves 2 x2 - x1 in `reflected`, the one value per node the dual
 * kernel gathers at each end of an edge.
 *
 * Node i's incidence list is strided over `Lanes` consecutive lanes of one
 * warp, each summing every Lanes-th half-edge, and the partial sums meet in
 * an xor butterfly.  One thread per node (Lanes = 1) walks the list serially,
 * a chain of dependent loads as long as the degree, which on a small dense
 * instance leaves the GPU almost idle: n = 100 is one block, each thread
 * fetching 99 half-edges one after another.  The butterfly leaves every lane
 * with the same bits, so the sum is a function of the instance and Lanes
 * only.  Every lane reaches the shuffles: a group past n has an empty list
 * and drops out only after them. */
template <class Real, bool Reflect, int Lanes>
__global__ void primal_kernel(int n, int n_edges, const int* __restrict__ node_ptr,
                              const int* __restrict__ node_half,
                              const Real* __restrict__ diag,
                              const Real* __restrict__ degree,
                              Real* __restrict__ x, Real* __restrict__ sx,
                              const Real* __restrict__ x0, Real* __restrict__ reflected,
                              const typename Pair<Real>::type* __restrict__ sy,
                              const Real* __restrict__ gather,
                              Real lw, Real rw, Real primal_step) {
    static_assert(Lanes >= 1 && Lanes <= 32 && (Lanes & (Lanes - 1)) == 0,
                  "a power of two, at most a warp");
    int lane = threadIdx.x % Lanes;
    int i = (blockIdx.x * blockDim.x + threadIdx.x) / Lanes;
    int begin = 0, end = 0;
    if (i < n) {
        begin = node_ptr[i];
        end = node_ptr[i + 1];
    }

    /* y1 on each incident half-edge: the dual kernel left it in `gather` for
     * a reflected step; a vanilla step's y1 is sy itself. */
    Real scattered = Real(0);
    for (int k = begin + lane; k < end; k += Lanes) {
        int h = node_half[k];
        if (Reflect) {
            scattered += gather[(h & 1) * n_edges + (h >> 1)];
        } else {
            scattered += (h & 1) ? sy[h >> 1].y : sy[h >> 1].x;
        }
    }
    for (int offset = Lanes / 2; offset > 0; offset >>= 1) {
        scattered += __shfl_xor_sync(0xffffffffu, scattered, offset);
    }
    if (i >= n || lane != 0) return;

    Real x1 = combine<Real, Reflect>(x[i], sx[i], x0[i], lw, rw);
    Real gradient = diag[i] - scattered;
    Real x2 = clamp01(x1 - primal_step * (gradient / degree[i]));

    x[i] = x1;
    sx[i] = x2;
    reflected[i] = Real(2) * x2 - x1;
}

/* The other half: the dual update, which needs x1 and x2 at both ends, so it
 * runs after primal_kernel.
 *
 *   y1 = combine(y, sy, y0)
 *   y2 = project_dual(y1 + eta omega (q - S (2 x2 - x1)))
 *
 * q is 1 in slot 0 of a covering edge and 0 elsewhere.  Afterwards y holds y1
 * and sy holds y2; thread e owns both, so in place again.
 *
 * It also writes `gather`: the y1 the NEXT step's primal will need on each
 * half-edge, combine(y1, y2, anchor, lw_next, rw_next), where the next step
 * is reflected with the Halpern weights of restart_iter + 1 -- or, after a
 * vanilla step, of restart_iter = 0 with y1 itself as the anchor, which is
 * what the host makes y0 before that step.  The primal's scatter is the
 * solver's one random access, and this makes it one Real per half-edge
 * instead of a read of y, sy and y0 each.  It is the same expression the
 * primal used to evaluate, so the iterates do not change.  When the host
 * restarts or stops instead, the prediction is simply unused: the restart's
 * vanilla step reads sy directly.  The array is slot-major, slot 0 of every
 * edge and then slot 1: an instance whose edges are sorted by their lower
 * end, as the generated ones are, then has each node's slot-0 half-edges at
 * consecutive Reals, and the primal's gather reads whole sectors there
 * instead of every other Real (2.8 -> 2.3 ms per launch at 2M nodes).
 *
 * `gather` is also where this kernel's own y1 comes from on a reflected
 * step: it is the value the step before predicted, bit for bit.  So within
 * a block of op_steps the dual recurrence runs on gather and y0 alone, and
 * y and sy, which only the check and the host's restart logic read, are
 * stored when `store` says the check comes next (and always on a vanilla
 * step, whose y1 is sy).  Per edge and reflected step that is 48 bytes read
 * and 16 written instead of 64 and 48.  This kernel is the solver's
 * streaming pass and runs at the memory roofline, so bytes are time. */
template <class Real, bool Reflect>
__global__ void dual_kernel(int m, const int* __restrict__ end0,
                            const int* __restrict__ end1,
                            const Real* __restrict__ weight,
                            const Real* __restrict__ reflected,
                            typename Pair<Real>::type* __restrict__ y,
                            typename Pair<Real>::type* __restrict__ sy,
                            const typename Pair<Real>::type* __restrict__ y0,
                            Real* __restrict__ gather,
                            Real lw_next, Real rw_next, Real dual_step, bool store) {
    using P = typename Pair<Real>::type;
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= m) return;

    int i = end0[e], j = end1[e];
    Real signed_w = weight[e];
    Real w = fabs(signed_w);
    bool positive = signed_w > Real(0);

    P y1, y0e;
    if (Reflect) {
        y1.x = gather[e];
        y1.y = gather[m + e];
        y0e = y0[e];
    } else {
        y1 = sy[e];
        y0e = y1;
    }

    P inner;
    inner.x = y1.x + dual_step * ((positive ? Real(1) : Real(0)) - reflected[i]);
    inner.y = y1.y + dual_step * (Real(0) - reflected[j]);

    P y2 = project_dual<Real, P>(inner, w, positive);
    if (store || !Reflect) {
        y[e] = y1;
        sy[e] = y2;
    }

    gather[e] = combine_next<Real>(y1.x, y2.x, y0e.x, lw_next, rw_next);
    gather[m + e] = combine_next<Real>(y1.y, y2.y, y0e.y, lw_next, rw_next);
}

/* The cold start is the centroid of the feasible region: x = 1/2, and on each
 * edge the centre of its dual set, (w/2, w/2) on a packing edge and
 * (-w/2, -w/2) on a covering one -- which is (-W_ij, -W_ij) either way, or
 * -signed_w / 2 here.  It is also the anchor of the first Halpern phase, which
 * is why it matters more than a start does for plain PDHG.  On an instance
 * whose diagonal is minus its row sums -- max-cut form -- this pair is
 * already an optimal certificate: every reduced cost is exactly 0 and the
 * bounds coincide, so the solve ends at iteration 0.
 *
 * Both the anchor and the "step" copies are set to the same point, so the
 * first vanilla step reads the right x1, y1 and leaves the state as
 * solve_qubo_problem_warm_start's StepArguments(x0, y0, x1, y1, x0, y0, ...). */
template <class Real>
__global__ void init_kernel(int n, int m, const Real* __restrict__ weight,
                            Real* x, Real* sx, Real* x0,
                            typename Pair<Real>::type* y,
                            typename Pair<Real>::type* sy,
                            typename Pair<Real>::type* y0, Real* gather) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) {
        x[k] = Real(0.5);
        sx[k] = Real(0.5);
        x0[k] = Real(0.5);
    }
    if (k < m) {
        typename Pair<Real>::type start;
        start.x = Real(-0.5) * weight[k];
        start.y = start.x;
        y[k] = start;
        sy[k] = start;
        y0[k] = start;
        gather[k] = start.x;
        gather[m + k] = start.y;
    }
}

/* --- reductions ----------------------------------------------------------- */

__device__ inline double warp_sum(double v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

/* Sum K values over the block; thread 0 writes them to out[0..K).  The order
 * of additions depends only on the block shape, so the result is deterministic. */
template <int K>
__device__ inline void block_sums(double (&v)[K], double* out) {
    __shared__ double shared[K][32];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int warps = (blockDim.x + 31) >> 5;

    for (int k = 0; k < K; ++k) {
        v[k] = warp_sum(v[k]);
        if (lane == 0) shared[k][warp] = v[k];
    }
    __syncthreads();
    if (warp == 0) {
        for (int k = 0; k < K; ++k) {
            double t = lane < warps ? shared[k][lane] : 0.0;
            t = warp_sum(t);
            if (lane == 0) out[k] = t;
        }
    }
}

/* The node sums of one check.  (x, sx) and x0 feed the canonical norm and the
 * omega update; the bounds are taken at (bx, by), which is (sx, sy) at a
 * regular check and (x, y) at the one before the loop starts. */
template <class Real>
__global__ void node_sums_kernel(int n, const int* __restrict__ node_ptr,
                                 const int* __restrict__ node_half,
                                 const Real* __restrict__ diag,
                                 const Real* __restrict__ degree,
                                 const Real* __restrict__ x, const Real* __restrict__ sx,
                                 const Real* __restrict__ x0,
                                 const Real* __restrict__ bx,
                                 const typename Pair<Real>::type* __restrict__ by,
                                 double* __restrict__ partial) {
    double acc[kNodeSums] = {0.0, 0.0, 0.0, 0.0};
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        double xi = x[i], sxi = sx[i], deg = degree[i], di = diag[i];
        double dx = xi - sxi, drift = xi - x0[i];
        acc[kPrimalNorm] += deg * dx * dx;
        acc[kPrimalDrift] += deg * drift * drift;
        acc[kUpperNode] += di * double(bx[i]);

        double scattered = 0.0;
        for (int k = node_ptr[i]; k < node_ptr[i + 1]; ++k) {
            int h = node_half[k];
            scattered += (h & 1) ? double(by[h >> 1].y) : double(by[h >> 1].x);
        }
        double reduced_cost = di - scattered;
        acc[kLowerNode] += reduced_cost < 0.0 ? reduced_cost : 0.0;
    }
    block_sums<kNodeSums>(acc, partial + blockIdx.x * kSums);
}

/* The edge sums of one check; see node_sums_kernel. */
template <class Real>
__global__ void edge_sums_kernel(int m, const int* __restrict__ end0,
                                 const int* __restrict__ end1,
                                 const Real* __restrict__ weight,
                                 const Real* __restrict__ x, const Real* __restrict__ sx,
                                 const typename Pair<Real>::type* __restrict__ y,
                                 const typename Pair<Real>::type* __restrict__ sy,
                                 const typename Pair<Real>::type* __restrict__ y0,
                                 const Real* __restrict__ bx,
                                 const typename Pair<Real>::type* __restrict__ by,
                                 double* __restrict__ partial) {
    double acc[kEdgeSums] = {0.0, 0.0, 0.0, 0.0, 0.0};
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < m;
         e += gridDim.x * blockDim.x) {
        int i = end0[e], j = end1[e];
        double signed_w = weight[e];
        double w = fabs(signed_w);
        bool positive = signed_w > 0.0;

        typename Pair<Real>::type ye = y[e], sye = sy[e], y0e = y0[e], bye = by[e];
        double dyx = double(ye.x) - double(sye.x), dyy = double(ye.y) - double(sye.y);
        acc[kDualNorm - kNodeSums] += dyx * dyx + dyy * dyy;
        acc[kCross - kNodeSums] += (double(x[i]) - double(sx[i])) * dyx
                                 + (double(x[j]) - double(sx[j])) * dyy;
        double drx = double(ye.x) - double(y0e.x), dry = double(ye.y) - double(y0e.y);
        acc[kDualDrift - kNodeSums] += drx * drx + dry * dry;

        double bi = bx[i], bj = bx[j];
        acc[kUpperEdge - kNodeSums] += positive ? w * fmax(bi + bj - 1.0, 0.0)
                                               : -w * fmin(bi, bj);
        acc[kLowerEdge - kNodeSums] += positive ? double(bye.x) : 0.0;
    }
    block_sums<kEdgeSums>(acc, partial + blockIdx.x * kSums + kNodeSums);
}

/* One block, one thread per partial row: adds the kReduceBlocks rows. */
__global__ void finalize_kernel(const double* __restrict__ partial, double* __restrict__ sums) {
    double v[kSums];
    for (int k = 0; k < kSums; ++k) v[k] = partial[threadIdx.x * kSums + k];
    block_sums<kSums>(v, sums);
}

/* --- host side ------------------------------------------------------------ */

#define PRAQ_CUDA_CHECK(call)                                   \
    do {                                                        \
        cudaError_t praq_error_ = (call);                       \
        if (praq_error_ != cudaSuccess) return kErrorCuda;      \
    } while (0)

template <class T>
struct DeviceArray {
    T* ptr = nullptr;
    ~DeviceArray() { if (ptr) cudaFree(ptr); }
    int allocate(size_t count) {
        PRAQ_CUDA_CHECK(cudaMalloc(&ptr, sizeof(T) * (count ? count : 1)));
        return 0;
    }
    int upload(const std::vector<T>& host) {
        if (allocate(host.size())) return kErrorCuda;
        if (!host.empty()) {
            PRAQ_CUDA_CHECK(cudaMemcpy(ptr, host.data(), sizeof(T) * host.size(),
                                       cudaMemcpyHostToDevice));
        }
        return 0;
    }
};

/* The instance in its scaled units plus the CSR incidence list, on the host. */
struct Structure {
    int n = 0, m = 0;
    double scale = 1.0;
    std::vector<int> node_ptr, node_half, end0, end1;
    std::vector<double> weight, diag, degree;   /* weight is signed: 2 W_ij / scale */

    /* Returns 0, or kErrorInstance for an instance the solver cannot take. */
    int build(const PraqInstance& in) {
        const int64_t limit = std::numeric_limits<int>::max();
        if (in.n <= 0 || in.n > limit || in.m < 0 || in.m > limit / 2) return kErrorInstance;
        if (!in.diag || (in.m > 0 && (!in.rows || !in.cols || !in.vals))) return kErrorInstance;
        n = int(in.n);
        m = int(in.m);

        end0.resize(m); end1.resize(m); weight.resize(m);
        for (int e = 0; e < m; ++e) {
            int64_t i = in.rows[e], j = in.cols[e];
            if (i < 0 || i >= in.n || j < 0 || j >= in.n || i == j) return kErrorInstance;
            end0[e] = int(i);
            end1[e] = int(j);
            weight[e] = 2.0 * in.vals[e];
        }

        /* objective_scale: ||(diag; w_e)||_2, or 1 for an all-zero instance. */
        double squares = 0.0;
        for (int i = 0; i < n; ++i) squares += in.diag[i] * in.diag[i];
        for (int e = 0; e < m; ++e) squares += weight[e] * weight[e];
        scale = squares > 0.0 ? std::sqrt(squares) : 1.0;

        diag.resize(n);
        for (int i = 0; i < n; ++i) diag[i] = in.diag[i] / scale;
        for (int e = 0; e < m; ++e) weight[e] /= scale;

        /* CSR over half-edges: node i's incident half-edges are
         * node_half[node_ptr[i] .. node_ptr[i+1]), each 2 e + slot. */
        node_ptr.assign(n + 1, 0);
        for (int e = 0; e < m; ++e) { ++node_ptr[end0[e] + 1]; ++node_ptr[end1[e] + 1]; }
        for (int i = 0; i < n; ++i) node_ptr[i + 1] += node_ptr[i];
        node_half.resize(2 * size_t(m));
        std::vector<int> fill(node_ptr.begin(), node_ptr.end() - 1);
        for (int e = 0; e < m; ++e) {
            node_half[fill[end0[e]]++] = 2 * e;
            node_half[fill[end1[e]]++] = 2 * e + 1;
        }

        degree.resize(n);
        for (int i = 0; i < n; ++i) {
            int incident = node_ptr[i + 1] - node_ptr[i];
            degree[i] = incident > 0 ? double(incident) : 1.0;
        }
        return 0;
    }
};

template <class Real>
struct Solver {
    using P = typename Pair<Real>::type;

    const Structure& s;
    PraqParameters p;       /* validated by praq_solve; op_steps >= 2         */
    double tolerance;
    int64_t max_iter;

    DeviceArray<int> node_ptr, node_half, end0, end1;
    DeviceArray<Real> weight, diag, degree, x, sx, x0, reflected;
    DeviceArray<P> y, sy, y0;
    DeviceArray<Real> gather;   /* 2 m, slot-major: see dual_kernel */
    DeviceArray<double> partial, sums;

    /* The scalar state of StepArguments, plus the parameters that change. */
    double omega, theta;
    double fp_error = 0.0, check_error = std::numeric_limits<double>::infinity();
    int64_t restart_iter = 0, iteration = 0;
    double lower = 0.0, upper = 0.0;
    bool terminated = false;
    int lanes;

    Solver(const Structure& structure, const PraqParameters& parameters)
        : s(structure), p(parameters), tolerance(parameters.tolerance),
          max_iter(parameters.max_iter), omega(parameters.omega),
          theta(parameters.theta_initial), lanes(lanes_for(structure)) {}

    static int blocks(int count) { return (count + kBlock - 1) / kBlock; }
    static int blocks_for(size_t threads) { return int((threads + kBlock - 1) / kBlock); }

    /* Lanes per node in primal_kernel, from the average degree 2 m / n: the
     * smallest power of two that leaves each lane no more than
     * kHalfEdgesPerLane half-edges, capped at a warp.  A function of the
     * instance, so the summation order is too. */
    static int lanes_for(const Structure& s) {
#ifdef PRAQ_FORCE_LANES
        return PRAQ_FORCE_LANES;
#endif
        double average = 2.0 * double(s.m) / double(s.n);
        int lanes = 1;
        while (lanes < 32 && double(lanes) * kHalfEdgesPerLane < average) lanes *= 2;
        return lanes;
    }

    template <class T>
    static std::vector<T> cast(const std::vector<double>& v) {
        return std::vector<T>(v.begin(), v.end());
    }

    int upload() {
        if (node_ptr.upload(s.node_ptr) || node_half.upload(s.node_half)
            || end0.upload(s.end0) || end1.upload(s.end1)
            || weight.upload(cast<Real>(s.weight)) || diag.upload(cast<Real>(s.diag))
            || degree.upload(cast<Real>(s.degree))) return kErrorCuda;
        if (x.allocate(s.n) || sx.allocate(s.n) || x0.allocate(s.n) || reflected.allocate(s.n)
            || y.allocate(s.m) || sy.allocate(s.m) || y0.allocate(s.m)
            || gather.allocate(2 * size_t(s.m))
            || partial.allocate(size_t(kReduceBlocks) * kSums) || sums.allocate(kSums)) {
            return kErrorCuda;
        }
        int count = s.n > s.m ? s.n : s.m;
        init_kernel<Real><<<blocks(count), kBlock>>>(s.n, s.m, weight.ptr, x.ptr, sx.ptr,
                                                     x0.ptr, y.ptr, sy.ptr, y0.ptr, gather.ptr);
        PRAQ_CUDA_CHECK(cudaGetLastError());
        return 0;
    }

    template <bool Reflect, int Lanes>
    void launch_primal(Real lw, Real rw, Real primal_step) {
        primal_kernel<Real, Reflect, Lanes><<<blocks_for(size_t(s.n) * Lanes), kBlock>>>(
            s.n, s.m, node_ptr.ptr, node_half.ptr, diag.ptr, degree.ptr, x.ptr, sx.ptr, x0.ptr,
            reflected.ptr, sy.ptr, gather.ptr, lw, rw, primal_step);
    }

    /* One PDHG step from (sx, sy), Halpern-combined with (x, y) and (x0, y0)
     * when Reflect; afterwards (x, y) is the point stepped from and (sx, sy)
     * the step -- on the dual side only when `store`, which must be set on
     * the step before a reduce (see dual_kernel).  Counts toward restart_iter
     * and iteration only if `count`. */
    template <bool Reflect>
    int step(double step_omega, bool count, bool store = true) {
        double k = double(restart_iter + 1);
        Real lw = Real(k / (k + 1.0)), rw = Real(1.0 / (k + 1.0));
        /* The step after this one is reflected, with restart_iter one higher
         * -- or, after a vanilla step, reset to 0.  See dual_kernel. */
        double kn = Reflect ? k + 1.0 : 1.0;
        Real lwn = Real(kn / (kn + 1.0)), rwn = Real(1.0 / (kn + 1.0));
        Real primal_step = Real(kEta / step_omega);
        switch (lanes) {
            case 1: launch_primal<Reflect, 1>(lw, rw, primal_step); break;
            case 2: launch_primal<Reflect, 2>(lw, rw, primal_step); break;
            case 4: launch_primal<Reflect, 4>(lw, rw, primal_step); break;
            case 8: launch_primal<Reflect, 8>(lw, rw, primal_step); break;
            case 16: launch_primal<Reflect, 16>(lw, rw, primal_step); break;
            default: launch_primal<Reflect, 32>(lw, rw, primal_step); break;
        }
        if (s.m > 0) {
            dual_kernel<Real, Reflect><<<blocks(s.m), kBlock>>>(
                s.m, end0.ptr, end1.ptr, weight.ptr, reflected.ptr, y.ptr, sy.ptr, y0.ptr,
                gather.ptr, lwn, rwn, Real(kEta * step_omega), store);
        }
        PRAQ_CUDA_CHECK(cudaGetLastError());
        if (count) { ++restart_iter; ++iteration; }
        return 0;
    }

    /* The nine sums, with the bounds taken at (bx, by).  Synchronises. */
    int reduce(const Real* bx, const P* by, double* out) {
        node_sums_kernel<Real><<<kReduceBlocks, kBlock>>>(
            s.n, node_ptr.ptr, node_half.ptr, diag.ptr, degree.ptr, x.ptr, sx.ptr, x0.ptr,
            bx, by, partial.ptr);
        edge_sums_kernel<Real><<<kReduceBlocks, kBlock>>>(
            s.m, end0.ptr, end1.ptr, weight.ptr, x.ptr, sx.ptr, y.ptr, sy.ptr, y0.ptr,
            bx, by, partial.ptr);
        finalize_kernel<<<1, kReduceBlocks>>>(partial.ptr, sums.ptr);
        PRAQ_CUDA_CHECK(cudaGetLastError());
        PRAQ_CUDA_CHECK(cudaMemcpy(out, sums.ptr, sizeof(double) * kSums, cudaMemcpyDeviceToHost));
        return 0;
    }

    /* canonical_norm of (x - sx, y - sy) in the metric P(eta, omega). */
    static double canonical_norm(const double* sums, double norm_omega) {
        double value = (norm_omega / kEta) * sums[kPrimalNorm]
                     + (1.0 / (kEta * norm_omega)) * sums[kDualNorm]
                     + 2.0 * sums[kCross];
        return std::sqrt(value > 0.0 ? value : 0.0);
    }

    /* The bounds at the original scale and the relative gap between them. */
    double gap(const double* sums) {
        upper = (sums[kUpperNode] + sums[kUpperEdge]) * s.scale;
        lower = (sums[kLowerNode] + sums[kLowerEdge]) * s.scale;
        return (upper - lower) / (1.0 + std::fabs(upper) + std::fabs(lower));
    }

    /* _check_termination_with_gap, negated: true when the loop should stop.
     * `!(gap <= tol)` rather than `gap > tol` so that a NaN gap keeps
     * iterating up to max_iter, which is the honest outcome. */
    bool should_stop(double relative_gap) const {
        bool within_max_iter = iteration < max_iter || max_iter < 0;
        bool outside_tolerance = !(relative_gap <= tolerance);
        return !(within_max_iter && outside_tolerance);
    }

    int start() {
        if (upload()) return kErrorCuda;
        /* x1, y1 = _raw_step(x0, y0) with the initial omega. */
        if (step<false>(omega, false)) return kErrorCuda;

        double sums[kSums];
        if (reduce(x.ptr, y.ptr, sums)) return kErrorCuda;
        fp_error = canonical_norm(sums, omega);
        check_error = std::numeric_limits<double>::infinity();
        terminated = should_stop(gap(sums));
        return 0;
    }

    /* _halpern_step: op_steps reflected steps, then the check. */
    int halpern_step() {
        for (int64_t k = 0; k < p.op_steps; ++k) {
            if (step<true>(omega, true, k + 1 == p.op_steps)) return kErrorCuda;
        }
        double sums[kSums];
        if (reduce(sx.ptr, sy.ptr, sums)) return kErrorCuda;

        double new_error = canonical_norm(sums, omega);
        terminated = should_stop(gap(sums));

        bool sufficient = new_error <= p.beta_sufficient * fp_error;
        bool necessary = new_error <= p.beta_necessary * fp_error && new_error > check_error;
        bool artificial = double(restart_iter) >= p.beta_artificial * double(iteration);
        bool should_restart = sufficient || necessary || artificial;

        double new_omega = omega, new_theta = theta;
        if (should_restart) {
            double primal_norm = std::sqrt(sums[kPrimalDrift]);
            double dual_norm = std::sqrt(sums[kDualDrift]);
            if (primal_norm > 1e-16 && dual_norm > 1e-16) {
                /* exponential smoothing in log space */
                new_omega = std::exp(theta * (std::log(dual_norm) - std::log(primal_norm))
                                     + (1.0 - theta) * std::log(omega));
                new_theta = p.theta_target + p.theta_decay * (theta - p.theta_target);
            }
        }
        omega = new_omega;
        theta = new_theta;

        if (terminated) {
            /* The returned point is the one the gap was measured at. */
            PRAQ_CUDA_CHECK(cudaMemcpy(x.ptr, sx.ptr, sizeof(Real) * s.n, cudaMemcpyDeviceToDevice));
            if (s.m > 0) {
                PRAQ_CUDA_CHECK(cudaMemcpy(y.ptr, sy.ptr, sizeof(P) * s.m, cudaMemcpyDeviceToDevice));
            }
            return 0;
        }
        if (should_restart) {
            /* Restart from the vanilla point: one uncounted step from (sx, sy)
             * with the new omega, which becomes the new anchor. */
            if (step<false>(omega, false)) return kErrorCuda;
            PRAQ_CUDA_CHECK(cudaMemcpy(x0.ptr, x.ptr, sizeof(Real) * s.n, cudaMemcpyDeviceToDevice));
            if (s.m > 0) {
                PRAQ_CUDA_CHECK(cudaMemcpy(y0.ptr, y.ptr, sizeof(P) * s.m, cudaMemcpyDeviceToDevice));
            }
            if (reduce(sx.ptr, sy.ptr, sums)) return kErrorCuda;
            fp_error = canonical_norm(sums, omega);
            check_error = std::numeric_limits<double>::infinity();
            restart_iter = 0;
        } else {
            check_error = new_error;
        }
        return 0;
    }

    int run(double* x_out, PraqResult* result) {
        if (start()) return kErrorCuda;
        while (!terminated) {
            if (halpern_step()) return kErrorCuda;
        }
        std::vector<Real> host(s.n);
        PRAQ_CUDA_CHECK(cudaMemcpy(host.data(), x.ptr, sizeof(Real) * s.n, cudaMemcpyDeviceToHost));
        for (int i = 0; i < s.n; ++i) x_out[i] = double(host[i]);

        result->lower = lower;
        result->upper = upper;
        result->iterations = iteration;
        double relative_gap = (upper - lower) / (1.0 + std::fabs(upper) + std::fabs(lower));
        result->status = relative_gap <= tolerance ? 0 : 1;
        return result->status;
    }
};

}  /* namespace praq */

#endif /* PRAQ_SOLVER_CUH */
