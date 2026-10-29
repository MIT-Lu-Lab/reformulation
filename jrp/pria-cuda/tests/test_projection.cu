/* test_projection.cu - GPU segmented double-simplex projections against the
 * CPU reference, over the real X/Y/Z geometries with adversarial values
 * (negatives, exact ties, degenerate targets). */
#include <vector>

#include "ref_impl.hpp"
#include "solver.cuh"
#include "test_common.hpp"

using namespace pria;

template <typename real>
static void run_case(uint64_t seed, i64 n, i64 t, bool zero_joint,
                     double tol) {
    ref::Instance inst = ref::random_instance(seed, n, t);
    if (zero_joint) inst.k0 = 0.0;

    std::vector<real> k(n), h(inst.h.size());
    for (i64 i = 0; i < n; ++i) k[i] = (real)inst.k[i];
    for (size_t i = 0; i < h.size(); ++i) h[i] = (real)inst.h[i];

    const int sf_default = sortfree_mode();
    const int warm_default = sf_warm_mode();
    const int comp_default = sf_compact_mode();
    /* same adversarial inputs for both backends; filled on first setup */
    std::vector<real> pre;
    std::vector<double> pre_d;

    /* the backend is latched at setup_ctx (it decides which projection
     * scratch exists), so each backend gets its own ctx */
    for (int sf = 0; sf < 2; ++sf) {
        sortfree_mode() = sf;
        Ctx<real> c;
        CHECK(setup_ctx(c, n, t, k.data(), inst.k0, h.data()) == cudaSuccess);
        const Dims d = c.d;

        if (pre.empty()) {
            /* mixture of signs, scales and exact duplicates */
            ref::Pcg32 rng(seed ^ 0xabcdef);
            pre.resize(d.ND);
            pre_d.resize(d.ND);
            for (i64 g = 0; g < d.ND; ++g) {
                double v;
                switch (rng.next() % 4) {
                    case 0: v = rng.uniform(-1.0, 1.0); break;
                    case 1: v = std::nearbyint(rng.uniform(-3.0, 3.0)); break;
                    case 2: v = rng.uniform(-1e-3, 1e-3); break;
                    default: v = rng.uniform(-20.0, 20.0); break;
                }
                pre[g] = (real)v;
                pre_d[g] = (double)pre[g];
            }
        }

        const i64 nsXY = 2 * d.N * d.P, nsZ = 2 * d.P;
        std::vector<int> offX(nsXY + 1), offY(nsXY + 1), offZ(nsZ + 1);
        build_offsets_xy(d, true, offX.data());
        build_offsets_xy(d, false, offY.data());
        build_offsets_z(d, offZ.data());

        /* scaled targets as the solver uses them */
        std::vector<double> ks(n);
        for (i64 i = 0; i < n; ++i)
            ks[i] = (double)(real)(inst.k[i] / c.scale);
        double k0s = c.k0s;

        auto check_geom = [&](int which, real* warm, double scale) {
            std::vector<real> pre_s(d.ND);
            for (i64 g = 0; g < d.ND; ++g)
                pre_s[g] = (real)(pre_d[g] * scale);
            CHECK(cudaMemcpy(c.pre.x, pre_s.data(), d.ND * sizeof(real),
                             cudaMemcpyHostToDevice) == cudaSuccess);
            /* the projection runs on the non-blocking c.stream, which does
             * not order against the legacy-stream upload's DMA */
            CHECK(cudaDeviceSynchronize() == cudaSuccess);
            cudaError_t st;
            if (which == 0)
                st = project_segments(c.ws, (const real*)c.pre.x, c.step.x,
                                      c.gx(), c.offX, warm, c.kdev,
                                      (real)c.k0s, c.stream);
            else if (which == 1)
                st = project_segments(c.ws, (const real*)c.pre.x, c.step.x,
                                      c.gy(), c.offY, warm, c.kdev,
                                      (real)c.k0s, c.stream);
            else
                st = project_segments(c.ws, (const real*)c.pre.x, c.step.x,
                                      c.gz(), c.offZ, warm, c.kdev,
                                      (real)c.k0s, c.stream);
            CHECK(st == cudaSuccess);
            /* c.stream is non-blocking: the legacy-stream D2H below does
             * not order against it implicitly */
            CHECK(cudaStreamSynchronize(c.stream) == cudaSuccess);
            std::vector<real> out(d.ND);
            CHECK(cudaMemcpy(out.data(), c.step.x, d.ND * sizeof(real),
                             cudaMemcpyDeviceToHost) == cudaSuccess);

            const double etol = tol * (scale > 1.0 ? scale : 1.0);
            const std::vector<int>& offs =
                which == 0 ? offX : (which == 1 ? offY : offZ);
            i64 nseg = which == 2 ? nsZ : nsXY;
            for (i64 seg = 0; seg < nseg; ++seg) {
                std::vector<double> v(offs[seg + 1] - offs[seg]);
                for (i64 e = 0; e < (i64)v.size(); ++e)
                    v[e] = (double)pre_s[offs[seg] + e];
                double target = which == 0 ? 1.0
                              : which == 1 ? ks[(seg >> 1) / d.P]
                                           : k0s;
                ref::simplex_project(v, target);
                double sum_gpu = 0.0;
                for (i64 e = 0; e < (i64)v.size(); ++e) {
                    CHECK_NEAR((double)out[offs[seg] + e], v[e], etol);
                    sum_gpu += (double)out[offs[seg] + e];
                }
                if (target > 0.0)
                    CHECK_NEAR(sum_gpu, target, etol * (double)v.size());
            }
        };

        /* fused per-row path and device-wide pipeline of this backend */
        for (int disable_fused = 0; disable_fused < 2; ++disable_fused) {
            fused_projection_disabled() = (disable_fused != 0);
            check_geom(0, nullptr, 1.0);
            check_geom(1, nullptr, 1.0);
            check_geom(2, nullptr, 1.0);
        }
        fused_projection_disabled() = false;

        if (sf == 1) {
            /* warm-started thresholds under both compaction modes: cold
             * populate, warm re-hit, shrink the values (stale theta
             * overshoots -> fallback), grow them (stale theta undershoots
             * -> monotone climb); every call checked elementwise */
            sf_warm_mode() = 1;
            for (int comp = 0; comp < 2; ++comp) {
                sf_compact_mode() = comp;
                for (int which = 0; which < 3; ++which) {
                    i64 ns = which == 2 ? nsZ : nsXY;
                    real* warm = nullptr;
                    CHECK(cudaMalloc(&warm, ns * sizeof(real)) ==
                          cudaSuccess);
                    std::vector<real> neg(
                        ns, -std::numeric_limits<real>::infinity());
                    CHECK(cudaMemcpy(warm, neg.data(), ns * sizeof(real),
                                     cudaMemcpyHostToDevice) == cudaSuccess);
                    CHECK(cudaDeviceSynchronize() == cudaSuccess);
                    check_geom(which, nullptr, 1.0); /* cold + compaction */
                    check_geom(which, warm, 1.0);
                    check_geom(which, warm, 1.0);
                    check_geom(which, warm, 0.25);
                    check_geom(which, warm, 4.0);
                    CHECK(cudaFree(warm) == cudaSuccess);
                }
            }
            sf_compact_mode() = comp_default;
            sf_warm_mode() = warm_default;
        }
        free_ctx(c);
    }
    sortfree_mode() = sf_default;
}

int main() {
    /* f64: tight tolerances */
    run_case<double>(1, 1, 1, false, 1e-11);
    run_case<double>(2, 2, 3, false, 1e-11);
    run_case<double>(3, 3, 8, false, 1e-11);
    run_case<double>(4, 5, 7, false, 1e-11);
    run_case<double>(5, 4, 16, false, 1e-10);
    run_case<double>(6, 1, 12, true, 1e-11);  /* degenerate K0 = 0 */
    run_case<double>(7, 12, 33, false, 1e-10);

    /* f32 */
    run_case<float>(8, 3, 8, false, 2e-4);
    run_case<float>(9, 6, 21, false, 2e-3);
    run_case<float>(10, 2, 5, true, 2e-4);

    /* wide rows: exercise every fused tile size and the f64 fallback */
    run_case<double>(20, 1, 600, false, 1e-9);   /* ITEMS=4 */
    run_case<double>(21, 1, 1400, false, 1e-9);  /* ITEMS=8 */
    run_case<double>(22, 1, 2500, false, 1e-9);  /* f64 -> device-wide */
    run_case<float>(23, 1, 2500, false, 2e-2);   /* ITEMS=16 */

    test_done("test_projection");
    return 0;
}
