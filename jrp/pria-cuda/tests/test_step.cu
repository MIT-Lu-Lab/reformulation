/* test_step.cu - the GPU PDHG operator (dense layouts + fused gathers + CUB
 * projections) against the CPU full-matrix reference, elementwise:
 *   - a single raw_step from a random state
 *   - a Halpern sequence (combine + raw_step) including a forced restart
 */
#include <vector>

#include "ref_impl.hpp"
#include "solver.cuh"
#include "test_common.hpp"

using namespace pria;

/* Inverse of densify_host on the padded size. */
template <typename real>
static void undensify_host(const Dims& d, const real* dense, int mode,
                           std::vector<double>& full) {
    full.assign(d.N * d.T * d.T, 0.0);
    for (i64 i = 0; i < d.N; ++i)
        for (i64 j = 0; j < d.P; ++j)
            for (i64 c = 0; c < d.W; ++c) {
                i64 s, t;
                if (mode == 0) col_to_st(d.T, j, c, s, t);
                else row_to_st(d.T, j, c, s, t);
                i64 g = (mode == 2) ? (j * d.W + c) * d.N + i
                                    : (i * d.P + j) * d.W + c;
                full[(i * d.T + s) * d.T + t] = (double)dense[g];
            }
}

static void upload_state(Ctx<double>& c, const ref::State& s, Slab<double>& to) {
    std::vector<double> stage(c.d.ND);
    densify_host(c.d, c.d.T, s.x.data(), 0, stage.data());
    CHECK(cudaMemcpy(to.x, stage.data(), c.d.ND * 8,
                     cudaMemcpyHostToDevice) == cudaSuccess);
    densify_host(c.d, c.d.T, s.y.data(), 1, stage.data());
    CHECK(cudaMemcpy(to.y, stage.data(), c.d.ND * 8,
                     cudaMemcpyHostToDevice) == cudaSuccess);
    densify_host(c.d, c.d.T, s.z.data(), 2, stage.data());
    CHECK(cudaMemcpy(to.z, stage.data(), c.d.ND * 8,
                     cudaMemcpyHostToDevice) == cudaSuccess);
    /* c.stream is non-blocking: order the legacy-stream uploads explicitly */
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
}

static void download_state(Ctx<double>& c, const Slab<double>& from,
                           ref::State& s) {
    std::vector<double> stage(c.d.ND);
    CHECK(cudaMemcpy(stage.data(), from.x, c.d.ND * 8,
                     cudaMemcpyDeviceToHost) == cudaSuccess);
    undensify_host(c.d, stage.data(), 0, s.x);
    CHECK(cudaMemcpy(stage.data(), from.y, c.d.ND * 8,
                     cudaMemcpyDeviceToHost) == cudaSuccess);
    undensify_host(c.d, stage.data(), 1, s.y);
    CHECK(cudaMemcpy(stage.data(), from.z, c.d.ND * 8,
                     cudaMemcpyDeviceToHost) == cudaSuccess);
    undensify_host(c.d, stage.data(), 2, s.z);
}

static void check_state(const ref::State& a, const ref::State& b, double tol) {
    for (size_t g = 0; g < a.x.size(); ++g) {
        CHECK_NEAR(a.x[g], b.x[g], tol);
        CHECK_NEAR(a.y[g], b.y[g], tol);
        CHECK_NEAR(a.z[g], b.z[g], tol);
    }
}

static void run_case(uint64_t seed, i64 n, i64 t) {
    ref::Instance inst = ref::random_instance(seed, n, t);
    Ctx<double> c;
    CHECK(setup_ctx(c, n, t, inst.k.data(), inst.k0, inst.h.data()) ==
          cudaSuccess);
    const Dims d = c.d;
    const i64 T = d.T, sz = n * T * T;

    /* scaled padded costs for the CPU side, exactly as the ctx holds them */
    std::vector<double> hs(sz, 0.0), ks(n);
    for (i64 i = 0; i < n; ++i)
        for (i64 s = 0; s < t; ++s)
            for (i64 tt = s; tt < t; ++tt)
                hs[(i * T + s) * T + tt] =
                    inst.h[(i * t + s) * t + tt] / c.scale;
    for (i64 i = 0; i < n; ++i) ks[i] = inst.k[i] / c.scale;
    const double k0s = c.k0s;

    const double omega = 0.2347, eta = 0.999 / std::sqrt(2.0);

    /* --- single raw_step from a random (upper-triangular) state -------- */
    ref::Pcg32 rng(seed * 977 + 13);
    ref::State s0;
    s0.x.assign(sz, 0.0);
    s0.y.assign(sz, 0.0);
    s0.z.assign(sz, 0.0);
    for (i64 i = 0; i < n; ++i)
        for (i64 s = 0; s < T; ++s)
            for (i64 tt = s; tt < T; ++tt) {
                i64 g = (i * T + s) * T + tt;
                s0.x[g] = rng.uniform(-1.0, 1.0);
                s0.y[g] = rng.uniform(-1.0, 1.0);
                s0.z[g] = rng.uniform(-1.0, 1.0);
            }

    ref::State want;
    ref::raw_step(n, T, hs, ks, k0s, omega, eta, s0, want);

    upload_state(c, s0, c.cur);
    CHECK(raw_step(c, c.cur, c.step, omega, eta) == cudaSuccess);
    CHECK(cudaStreamSynchronize(c.stream) == cudaSuccess);
    ref::State got;
    download_state(c, c.step, got);
    check_state(want, got, 1e-10);

    /* --- Halpern sequence from the projected zero start ---------------- */
    ref::State cur, step, anchor;
    cur.x.assign(sz, 0.0);
    cur.y.assign(sz, 0.0);
    cur.z.assign(sz, 0.0);
    ref::proj_x(n, T, cur.x);
    ref::proj_y(n, T, cur.y, ks);
    ref::proj_z(n, T, cur.z, k0s);
    anchor = cur;
    ref::raw_step(n, T, hs, ks, k0s, omega, eta, cur, step);

    CHECK(cudaMemset(c.pre.base, 0, 3 * d.ND * 8) == cudaSuccess);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(project_segments(c.ws, (const double*)c.pre.x, c.cur.x, c.gx(),
                           c.offX, c.warmX, c.kdev, c.k0s,
                           c.stream) == cudaSuccess);
    CHECK(project_segments(c.ws, (const double*)c.pre.y, c.cur.y, c.gy(),
                           c.offY, c.warmY, c.kdev, c.k0s,
                           c.stream) == cudaSuccess);
    CHECK(project_segments(c.ws, (const double*)c.pre.z, c.cur.z, c.gz(),
                           c.offZ, c.warmZ, c.kdev, c.k0s,
                           c.stream) == cudaSuccess);
    CHECK(cudaMemcpyAsync(c.anchor.base, c.cur.base, 3 * d.ND * 8,
                          cudaMemcpyDeviceToDevice, c.stream) == cudaSuccess);
    CHECK(raw_step(c, c.cur, c.step, omega, eta) == cudaSuccess);

    i64 restart_iter = 0;
    auto h_step_both = [&]() {
        double lw = (double)(restart_iter + 1) / (restart_iter + 2);
        double rw = 1.0 / (restart_iter + 2);
        for (i64 g = 0; g < sz; ++g) {
            cur.x[g] = lw * (2.0 * step.x[g] - cur.x[g]) + rw * anchor.x[g];
            cur.y[g] = lw * (2.0 * step.y[g] - cur.y[g]) + rw * anchor.y[g];
            cur.z[g] = lw * (2.0 * step.z[g] - cur.z[g]) + rw * anchor.z[g];
        }
        ref::State next;
        ref::raw_step(n, T, hs, ks, k0s, omega, eta, cur, next);
        step = next;

        /* driter is zeroed in setup_ctx, so the counter goes in as the slot */
        k_combine<<<launch_grid(3 * d.ND), kBlock, 0, c.stream>>>(
            3 * d.ND, c.cur.base, (const double*)c.step.base,
            (const double*)c.anchor.base, (const i64*)c.driter,
            (int)restart_iter);
        CHECK(raw_step(c, c.cur, c.step, omega, eta) == cudaSuccess);
        ++restart_iter;
    };
    for (int it = 0; it < 6; ++it) h_step_both();

    CHECK(cudaStreamSynchronize(c.stream) == cudaSuccess);
    download_state(c, c.cur, got);
    check_state(cur, got, 1e-9);
    download_state(c, c.step, got);
    check_state(step, got, 1e-9);

    /* --- forced restart mechanics -------------------------------------- */
    anchor = step;
    {
        ref::State next;
        ref::raw_step(n, T, hs, ks, k0s, 2.0 * omega, eta, step, next);
        cur = step;
        step = next;
    }
    CHECK(cudaMemcpyAsync(c.anchor.base, c.step.base, 3 * d.ND * 8,
                          cudaMemcpyDeviceToDevice, c.stream) == cudaSuccess);
    CHECK(raw_step(c, c.step, c.cur, 2.0 * omega, eta) == cudaSuccess);
    std::swap(c.cur, c.step);
    CHECK(cudaStreamSynchronize(c.stream) == cudaSuccess);

    download_state(c, c.cur, got);
    check_state(cur, got, 1e-9);
    download_state(c, c.step, got);
    check_state(step, got, 1e-9);
    download_state(c, c.anchor, got);
    check_state(anchor, got, 1e-9);

    /* --- reductions: canonical error and objective gap ----------------- */
    enqueue_err(c);
    CHECK(enqueue_gap(c, c.step) == cudaSuccess);
    CHECK(sync_reductions(c) == cudaSuccess);
    double err_gpu = canonical_err(c.hred, eta, 2.0 * omega);
    GapInfo gi_gpu = gap_from(c.hred, c.scale);
    double err_ref = ref::canonical_err(n, T, cur, step, eta, 2.0 * omega);
    ref::Gap gi_ref = ref::objective_gap(n, T, hs, ks, k0s, c.scale, step);
    CHECK_NEAR(err_gpu, err_ref, 1e-9 * (1.0 + err_ref));
    CHECK_NEAR(gi_gpu.primal, gi_ref.primal, 1e-9 * (1.0 + std::fabs(gi_ref.primal)));
    CHECK_NEAR(gi_gpu.dual, gi_ref.dual, 1e-9 * (1.0 + std::fabs(gi_ref.dual)));
    CHECK_NEAR(gi_gpu.gap, gi_ref.gap, 1e-9);

    free_ctx(c);
}

/* Tiled layout mirrors: every element must land exactly where the closed
 * form converters say, in all four directions. */
static void run_mirrors(uint64_t seed, i64 n, i64 t) {
    ref::Instance inst = ref::random_instance(seed, n, t);
    Ctx<double> c;
    CHECK(setup_ctx(c, n, t, inst.k.data(), inst.k0, inst.h.data()) ==
          cudaSuccess);
    const Dims d = c.d;

    std::vector<double> src(d.ND), dst(d.ND);
    ref::Pcg32 rng(seed * 31 + 7);
    for (auto& v : src) v = rng.uniform(-1.0, 1.0);
    double *dsrc, *ddst;
    CHECK(cudaMalloc(&dsrc, d.ND * 8) == cudaSuccess);
    CHECK(cudaMalloc(&ddst, d.ND * 8) == cudaSuccess);
    CHECK(cudaMemcpy(dsrc, src.data(), d.ND * 8, cudaMemcpyHostToDevice) ==
          cudaSuccess);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);

    auto pull = [&]() {
        CHECK(cudaStreamSynchronize(c.stream) == cudaSuccess);
        CHECK(cudaMemcpy(dst.data(), ddst, d.ND * 8,
                         cudaMemcpyDeviceToHost) == cudaSuccess);
    };

    launch_mirror_rowcol<double, true>(d, dsrc, ddst, c.stream);
    pull();
    for (i64 g = 0; g < d.ND; ++g) CHECK(dst[g] == src[colg_to_rowg(d, g)]);

    launch_mirror_rowcol<double, false>(d, dsrc, ddst, c.stream);
    pull();
    for (i64 g = 0; g < d.ND; ++g) CHECK(dst[g] == src[rowg_to_colg(d, g)]);

    launch_mirror_d2col<double, true>(d, dsrc, ddst, c.stream);
    pull();
    for (i64 g = 0; g < d.ND; ++g) CHECK(dst[g] == src[colg_to_d2g(d, g)]);

    launch_mirror_d2col<double, false>(d, dsrc, ddst, c.stream);
    pull();
    for (i64 g = 0; g < d.ND; ++g) CHECK(dst[g] == src[d2g_to_colg(d, g)]);

    cudaFree(dsrc);
    cudaFree(ddst);
    free_ctx(c);
}

int main() {
    run_mirrors(21, 1, 1);
    run_mirrors(22, 3, 8);
    run_mirrors(23, 40, 70);  /* item tiles beyond 32 */
    run_mirrors(24, 5, 33);   /* odd t, partial (s,t) tiles */
    run_mirrors(25, 33, 3);
    run_case(11, 1, 1);
    run_case(12, 2, 3);
    run_case(13, 3, 8);
    run_case(14, 5, 7);
    run_case(15, 4, 16);
    run_case(16, 9, 21);
    test_done("test_step");
    return 0;
}
