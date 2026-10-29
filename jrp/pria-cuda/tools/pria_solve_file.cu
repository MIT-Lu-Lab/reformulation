/* pria_solve_file.cu - file-based CLI around the PRIA solver.
 *
 * The harness (harness/backends/pria_cuda.py) drives this binary rather than
 * linking against the library, so the wire format below is the contract
 * between the two.  Keep the two definitions in step.
 *
 * Instance file (little-endian binary):
 *   int64 magic = 0x41495250 ("PRIA")
 *   int64 n, int64 t
 *   double k0
 *   double k[n]
 *   double h[n*t*t]          (row-major, upper-triangular part used)
 *
 * Result file:
 *   int64 n, int64 t, int64 iterations, int64 status
 *   double lower, double upper, double gap
 *   double x[n*t*t], y[n*t*t], z[n*t*t]    (only with the "solution" flag)
 *
 * where status is 0 converged and 1 iteration limit.  The library reports no
 * status of its own -- pria_result_info carries only the bounds and the gap --
 * so it is derived here, from whether the gap it reached is the gap it was
 * asked for.  That keeps the derivation in one place instead of in every
 * caller.
 *
 * The solution arrays are three n*t*t blocks, over half a gigabyte each at the
 * top of the benchmark's ladder, and the harness records bounds and iterations
 * and nothing else -- so they are written only when asked for.  The file is
 * written only when the solver ran; on any other failure nothing is, and the
 * exit status says why.  Stdout gets the same numbers as "key: value" lines
 * plus two timings: setup_seconds covers reading the file and creating the
 * CUDA context, solve_seconds the pria_solve call.
 *
 * Usage: pria_solve_file <instance> <result> [tol] [max_iter] [flags...]
 *
 *   tol       relative duality gap to stop on; default pria_params_auto's
 *   max_iter  iteration cap; -1 or omitted for unlimited
 *   flags     f32       iterate in single precision
 *             vanilla   plain PDHG instead of restarted Halpern
 *             verbose   per-check progress on stderr
 *             solution  write x, y and z into the result file
 *
 * Exit status: 0 converged, 1 iteration limit, 2 usage or unreadable input,
 * 3 the solver reported an error.
 */
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "pria/pria.h"

static const int64_t kMagic = 0x41495250;

static double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

template <class T>
static bool read_array(FILE* file, std::vector<T>& out, int64_t count) {
    out.resize(size_t(count));
    return count == 0 || fread(out.data(), sizeof(T), size_t(count), file) == size_t(count);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
                "usage: %s <instance> <result> [tol] [max_iter] "
                "[f32|vanilla|verbose|solution]...\n",
                argv[0]);
        return 2;
    }
    double tol = 1e-4;
    int64_t max_iter = -1;
    bool f32 = false, vanilla = false, verbose = false, solution = false;
    if (argc > 3) tol = atof(argv[3]);
    if (argc > 4) max_iter = atoll(argv[4]);
    for (int a = 5; a < argc; ++a) {
        if (!strcmp(argv[a], "f32")) f32 = true;
        else if (!strcmp(argv[a], "vanilla")) vanilla = true;
        else if (!strcmp(argv[a], "verbose")) verbose = true;
        else if (!strcmp(argv[a], "solution")) solution = true;
        else { fprintf(stderr, "pria_solve_file: unknown flag %s\n", argv[a]); return 2; }
    }

    auto started = std::chrono::steady_clock::now();

    FILE* file = fopen(argv[1], "rb");
    if (!file) {
        fprintf(stderr, "pria_solve_file: cannot open %s\n", argv[1]);
        return 2;
    }
    int64_t header[3];
    double k0 = 0.0;
    std::vector<double> k, h;
    bool ok = fread(header, sizeof(int64_t), 3, file) == 3 && header[0] == kMagic
              && header[1] > 0 && header[2] > 0
              && fread(&k0, sizeof(double), 1, file) == 1
              && read_array(file, k, header[1])
              && read_array(file, h, header[1] * header[2] * header[2]);
    fclose(file);
    if (!ok) {
        fprintf(stderr, "pria_solve_file: %s is not a PRIA instance file\n", argv[1]);
        return 2;
    }
    const int64_t n = header[1], t = header[2];
    const size_t size = size_t(n) * size_t(t) * size_t(t);

    pria_params parameters = pria_params_auto(max_iter, tol);
    parameters.use_halpern = vanilla ? 0 : 1;
    parameters.verbose = verbose ? 1 : 0;

    /* Create the CUDA context here, so its one-off cost is charged to
     * setup_seconds rather than to the solve. */
    cudaFree(nullptr);
    double setup_seconds = seconds_since(started);

    std::vector<double> x, y, z;
    if (solution) { x.resize(size); y.resize(size); z.resize(size); }
    double* xp = solution ? x.data() : nullptr;
    double* yp = solution ? y.data() : nullptr;
    double* zp = solution ? z.data() : nullptr;

    pria_result_info info;
    int status;
    started = std::chrono::steady_clock::now();
    if (f32) {
        std::vector<float> kf(k.begin(), k.end()), hf(h.begin(), h.end());
        std::vector<float> xf, yf, zf;
        if (solution) { xf.resize(size); yf.resize(size); zf.resize(size); }
        status = pria_solve_f32(n, t, kf.data(), float(k0), hf.data(), &parameters,
                                nullptr, nullptr, nullptr,
                                solution ? xf.data() : nullptr,
                                solution ? yf.data() : nullptr,
                                solution ? zf.data() : nullptr, &info);
        for (size_t g = 0; solution && g < size; ++g) {
            x[g] = xf[g]; y[g] = yf[g]; z[g] = zf[g];
        }
    } else {
        status = pria_solve_f64(n, t, k.data(), k0, h.data(), &parameters,
                                nullptr, nullptr, nullptr, xp, yp, zp, &info);
    }
    double solve_seconds = seconds_since(started);

    if (status != 0) {
        fprintf(stderr, "pria_solve_file: pria_solve failed with status %d\n", status);
        return 3;
    }

    /* primal_objective is the objective at the point the solver holds, so it
     * is the upper bound; dual_objective is the lower one.  The library's
     * names say which side of the saddle each came from, these say which side
     * of the optimum they sit on, which is what a caller needs. */
    const double lower = info.dual_objective, upper = info.primal_objective;
    const int64_t converged = std::isfinite(info.relative_gap)
                              && info.relative_gap <= tol ? 0 : 1;

    FILE* out = fopen(argv[2], "wb");
    if (!out) {
        fprintf(stderr, "pria_solve_file: cannot write %s\n", argv[2]);
        return 2;
    }
    int64_t words[4] = {n, t, info.iterations, converged};
    double bounds[3] = {lower, upper, info.relative_gap};
    ok = fwrite(words, sizeof(int64_t), 4, out) == 4
         && fwrite(bounds, sizeof(double), 3, out) == 3;
    if (ok && solution) {
        ok = fwrite(x.data(), sizeof(double), size, out) == size
             && fwrite(y.data(), sizeof(double), size, out) == size
             && fwrite(z.data(), sizeof(double), size, out) == size;
    }
    ok = fclose(out) == 0 && ok;
    if (!ok) {
        fprintf(stderr, "pria_solve_file: short write to %s\n", argv[2]);
        return 2;
    }

    printf("items: %lld\n", (long long)n);
    printf("periods: %lld\n", (long long)t);
    printf("precision: %s\n", f32 ? "float32" : "float64");
    printf("status: %s\n", converged == 0 ? "converged" : "iteration_limit");
    printf("iterations: %lld\n", (long long)info.iterations);
    printf("lower: %.17g\n", lower);
    printf("upper: %.17g\n", upper);
    printf("relative_gap: %.6e\n", info.relative_gap);
    printf("setup_seconds: %.6f\n", setup_seconds);
    printf("solve_seconds: %.6f\n", solve_seconds);
    return int(converged);
}
