/* pria_bench.cu - standalone timing of the PRIA CUDA solver on random
 * instances (structure matching the JAX package's generator).
 *
 * Usage: pria_bench [--f64] [--tol T]... [--sizes n:t:count,...] [--batch B]
 *                   [--instances DIR]
 * Timings are end-to-end pria_solve_* calls (allocation, transfers, solve).
 * With --batch B, each size entry solves B instances through
 * pria_solve_batch_* and reports total and per-instance time.
 *
 * With --instances DIR the instances are read from DIR/n<n>_t<t>/inst_<i>.bin
 * (the format of tools/pria_solve_file.cu) instead of being generated, so the
 * run covers exactly the problems the JAX-side benchmarks solve; without it
 * they come from ref::random_instance, which matches the distribution but not
 * the draws.
 */
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "pria/pria.h"
#include "ref_impl.hpp"

using ref::i64;

struct SizeSpec {
    i64 n, t;
    int count;
};

static const i64 kMagic = 0x41495250; /* "PRIA" */

/* Read one serialized instance; aborts on a missing or malformed file, since a
 * silently skipped instance would make the benchmark incomparable. */
static ref::Instance load_instance(const std::string& dir, i64 n, i64 t,
                                   int index) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/n%lld_t%lld/inst_%04d.bin", dir.c_str(),
             (long long)n, (long long)t, index);
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(2);
    }
    i64 magic = 0, fn = 0, ft = 0;
    double k0 = 0;
    if (fread(&magic, 8, 1, f) != 1 || fread(&fn, 8, 1, f) != 1 ||
        fread(&ft, 8, 1, f) != 1 || fread(&k0, 8, 1, f) != 1 ||
        magic != kMagic || fn != n || ft != t) {
        fprintf(stderr, "%s: bad header (magic/%lld/%lld)\n", path,
                (long long)fn, (long long)ft);
        exit(2);
    }
    ref::Instance in;
    in.n = n;
    in.t = t;
    in.k0 = k0;
    in.k.resize(n);
    in.h.resize((size_t)n * t * t);
    if (fread(in.k.data(), 8, in.k.size(), f) != in.k.size() ||
        fread(in.h.data(), 8, in.h.size(), f) != in.h.size()) {
        fprintf(stderr, "%s: truncated\n", path);
        exit(2);
    }
    fclose(f);
    return in;
}

static ref::Instance bench_instance(const std::string& dir, i64 n, i64 t,
                                    int index) {
    if (!dir.empty()) return load_instance(dir, n, t, index);
    return ref::random_instance(1000 + 77 * index, n, t);
}

/* Same summary statistic the JAX-side harness prints (bench_instances.py). */
static double shifted_geometric_mean(const std::vector<double>& v,
                                     double shift) {
    if (v.empty()) return 0.0;
    double logsum = 0.0;
    for (double x : v) logsum += std::log(x + shift);
    return std::exp(logsum / (double)v.size()) - shift;
}

template <typename real>
static int run(const std::vector<SizeSpec>& sizes,
               const std::vector<double>& tols, const std::string& dir) {
    printf("%6s %6s %10s | %10s %12s %10s %10s\n", "items", "T", "tol",
           "solve ms", "iters", "primal", "gap");
    for (const SizeSpec& sz : sizes) {
        for (double tol : tols) {
            std::vector<double> secs;
            int solved = 0;
            for (int rep = 0; rep < sz.count; ++rep) {
                ref::Instance in = bench_instance(dir, sz.n, sz.t, rep);
                std::vector<real> k(in.k.begin(), in.k.end());
                std::vector<real> h(in.h.begin(), in.h.end());

                pria_params p = pria_params_auto(-1, tol);
                pria_result_info info;
                auto t0 = std::chrono::steady_clock::now();
                int st;
                if (sizeof(real) == 8)
                    st = pria_solve_f64(sz.n, sz.t, (const double*)k.data(),
                                        in.k0, (const double*)h.data(), &p,
                                        nullptr, nullptr, nullptr, nullptr,
                                        nullptr, nullptr, &info);
                else
                    st = pria_solve_f32(sz.n, sz.t, (const float*)k.data(),
                                        (float)in.k0, (const float*)h.data(),
                                        &p, nullptr, nullptr, nullptr, nullptr,
                                        nullptr, nullptr, &info);
                auto t1 = std::chrono::steady_clock::now();
                if (st != 0) {
                    fprintf(stderr, "solve failed: %d\n", st);
                    return 1;
                }
                double ms =
                    std::chrono::duration<double, std::milli>(t1 - t0).count();
                printf("%6lld %6lld %10.1e | %10.2f %12lld %10.1f %10.2e\n",
                       (long long)sz.n, (long long)sz.t, tol, ms,
                       (long long)info.iterations, info.primal_objective,
                       info.relative_gap);
                fflush(stdout);
                secs.push_back(ms / 1000.0);
                if (info.relative_gap <= tol) ++solved;
            }
            printf("solved %d/%d, 1-shifted geometric mean: %.6f\n\n", solved,
                   sz.count, shifted_geometric_mean(secs, 1.0));
            fflush(stdout);
        }
    }
    return 0;
}

template <typename real>
static int run_batch(const std::vector<SizeSpec>& sizes,
                     const std::vector<double>& tols, i64 batch,
                     const std::string& dir) {
    printf("%6s %6s %10s %6s | %12s %14s\n", "items", "T", "tol", "batch",
           "total ms", "ms/instance");
    for (const SizeSpec& sz : sizes) {
        const i64 e = sz.n * sz.t * sz.t;
        std::vector<real> k(batch * sz.n), k0(batch), h(batch * e);
        for (i64 b = 0; b < batch; ++b) {
            ref::Instance in = bench_instance(dir, sz.n, sz.t, (int)b);
            std::copy(in.k.begin(), in.k.end(), k.begin() + b * sz.n);
            k0[b] = (real)in.k0;
            std::copy(in.h.begin(), in.h.end(), h.begin() + b * e);
        }
        for (double tol : tols) {
            pria_params p = pria_params_auto(-1, tol);
            std::vector<pria_result_info> infos(batch);
            auto t0 = std::chrono::steady_clock::now();
            int st;
            if (sizeof(real) == 8)
                st = pria_solve_batch_f64(batch, sz.n, sz.t,
                                          (const double*)k.data(),
                                          (const double*)k0.data(),
                                          (const double*)h.data(), &p,
                                          nullptr, nullptr, nullptr,
                                          infos.data());
            else
                st = pria_solve_batch_f32(batch, sz.n, sz.t,
                                          (const float*)k.data(),
                                          (const float*)k0.data(),
                                          (const float*)h.data(), &p,
                                          nullptr, nullptr, nullptr,
                                          infos.data());
            auto t1 = std::chrono::steady_clock::now();
            if (st != 0) {
                fprintf(stderr, "batch solve failed: %d\n", st);
                return 1;
            }
            double ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
            printf("%6lld %6lld %10.1e %6lld | %12.2f %14.2f\n",
                   (long long)sz.n, (long long)sz.t, tol, (long long)batch,
                   ms, ms / (double)batch);
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    bool f64 = false;
    i64 batch = 0;
    std::string dir;
    std::vector<double> tols;
    std::vector<SizeSpec> sizes;
    for (int a = 1; a < argc; ++a) {
        if (!strcmp(argv[a], "--f64")) f64 = true;
        else if (!strcmp(argv[a], "--batch") && a + 1 < argc)
            batch = atoll(argv[++a]);
        else if (!strcmp(argv[a], "--instances") && a + 1 < argc)
            dir = argv[++a];
        else if (!strcmp(argv[a], "--tol") && a + 1 < argc)
            tols.push_back(atof(argv[++a]));
        else if (!strcmp(argv[a], "--sizes") && a + 1 < argc) {
            std::string s = argv[++a];
            size_t pos = 0;
            while (pos < s.size()) {
                size_t end = s.find(',', pos);
                if (end == std::string::npos) end = s.size();
                SizeSpec spec{0, 0, 1};
                sscanf(s.c_str() + pos, "%lld:%lld:%d", (long long*)&spec.n,
                       (long long*)&spec.t, &spec.count);
                sizes.push_back(spec);
                pos = end + 1;
            }
        }
    }
    if (tols.empty()) tols = {1e-4, 1e-6};
    if (sizes.empty())
        sizes = {{30, 75, 3}, {75, 150, 3}, {200, 300, 2}, {350, 450, 1}};

    /* warm up the CUDA context so the first row is not charged for it */
    {
        ref::Instance in = ref::random_instance(1, 2, 4);
        pria_params p = pria_params_auto(32, 1e-2);
        pria_solve_f64(2, 4, in.k.data(), in.k0, in.h.data(), &p, nullptr,
                       nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    }

    printf("PRIA CUDA benchmark (%s), instances: %s\n",
           f64 ? "float64" : "float32",
           dir.empty() ? "generated (ref::random_instance)" : dir.c_str());
    if (batch > 0)
        return f64 ? run_batch<double>(sizes, tols, batch, dir)
                   : run_batch<float>(sizes, tols, batch, dir);
    return f64 ? run<double>(sizes, tols, dir) : run<float>(sizes, tols, dir);
}
