/* praq_solve_file.cu - file-based CLI around the PRAQ solver.
 *
 * The harness (harness/backends/praq_cuda.py) drives this binary rather than
 * linking against the library, so the wire format below is the contract between
 * the two.  Keep the two definitions in step.
 *
 * Instance file (little-endian binary):
 *   int64 magic = 0x51415250 ("PRAQ")
 *   int64 n, int64 m
 *   double diag[n]
 *   int64 rows[m], int64 cols[m], double vals[m]
 *
 * Result file:
 *   int64 n, int64 iterations, int64 status
 *   double lower, double upper
 *   double x[n]
 *
 * where status is PraqResult.status: 0 converged, 1 iteration limit.  The file
 * is written only when the solver ran; on any other failure nothing is, and the
 * exit status says why.  Stdout gets the same numbers as "key: value" lines plus
 * two timings: setup_seconds covers reading the file and creating the CUDA
 * context, solve_seconds the praq_solve call.
 *
 * Usage: praq_solve_file <instance> <result> [tol] [max_iter] [f32]
 *
 *   tol       relative duality gap to stop on; default praq_default_parameters
 *   max_iter  iteration cap; 0 or omitted for the solver's own
 *   f32       the literal "f32" to iterate in single precision
 *
 * Exit status: 0 converged, 1 iteration limit, 2 usage or unreadable input,
 * 3 the solver reported an error.
 */
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "praq/praq.h"

static const int64_t kMagic = 0x51415250;

static double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

template <class T>
static bool read_array(FILE* file, std::vector<T>& out, int64_t count) {
    out.resize(size_t(count));
    return count == 0 || fread(out.data(), sizeof(T), size_t(count), file) == size_t(count);
}

int main(int argc, char** argv) {
    if (argc < 3 || argc > 6) {
        fprintf(stderr, "usage: %s <instance> <result> [tol] [max_iter] [f32]\n", argv[0]);
        return 2;
    }
    PraqParameters parameters = praq_default_parameters();
    if (argc > 3) parameters.tolerance = atof(argv[3]);
    if (argc > 4) parameters.max_iter = atoll(argv[4]);
    if (argc > 5) parameters.use_float32 = strcmp(argv[5], "f32") == 0;

    auto started = std::chrono::steady_clock::now();

    FILE* file = fopen(argv[1], "rb");
    if (!file) {
        fprintf(stderr, "praq_solve_file: cannot open %s\n", argv[1]);
        return 2;
    }
    int64_t header[3];
    std::vector<double> diag, vals;
    std::vector<int64_t> rows, cols;
    bool ok = fread(header, sizeof(int64_t), 3, file) == 3 && header[0] == kMagic
              && header[1] >= 0 && header[2] >= 0
              && read_array(file, diag, header[1]) && read_array(file, rows, header[2])
              && read_array(file, cols, header[2]) && read_array(file, vals, header[2]);
    fclose(file);
    if (!ok) {
        fprintf(stderr, "praq_solve_file: %s is not a PRAQ instance file\n", argv[1]);
        return 2;
    }

    PraqInstance instance;
    instance.n = header[1];
    instance.m = header[2];
    instance.diag = diag.data();
    instance.rows = rows.data();
    instance.cols = cols.data();
    instance.vals = vals.data();

    /* Context creation is a per-process cost of a few hundred milliseconds
     * that would otherwise land in solve_seconds. */
    cudaFree(nullptr);
    double setup_seconds = seconds_since(started);

    std::vector<double> x(size_t(instance.n));
    PraqResult result;
    started = std::chrono::steady_clock::now();
    int status = praq_solve(&instance, &parameters, x.data(), &result);
    double solve_seconds = seconds_since(started);

    if (status < 0) {
        fprintf(stderr, "praq_solve_file: praq_solve failed with status %d\n", status);
        return 3;
    }

    FILE* out = fopen(argv[2], "wb");
    if (!out) {
        fprintf(stderr, "praq_solve_file: cannot write %s\n", argv[2]);
        return 2;
    }
    int64_t words[3] = {instance.n, result.iterations, int64_t(result.status)};
    double bounds[2] = {result.lower, result.upper};
    ok = fwrite(words, sizeof(int64_t), 3, out) == 3 && fwrite(bounds, sizeof(double), 2, out) == 2
         && fwrite(x.data(), sizeof(double), x.size(), out) == x.size();
    ok = fclose(out) == 0 && ok;
    if (!ok) {
        fprintf(stderr, "praq_solve_file: short write to %s\n", argv[2]);
        return 2;
    }

    printf("n: %lld\n", (long long)instance.n);
    printf("edges: %lld\n", (long long)instance.m);
    printf("precision: %s\n", parameters.use_float32 ? "float32" : "float64");
    printf("status: %s\n", result.status == 0 ? "converged" : "iteration_limit");
    printf("iterations: %lld\n", (long long)result.iterations);
    printf("lower: %.17g\n", result.lower);
    printf("upper: %.17g\n", result.upper);
    printf("setup_seconds: %.6f\n", setup_seconds);
    printf("solve_seconds: %.6f\n", solve_seconds);
    return result.status;
}
