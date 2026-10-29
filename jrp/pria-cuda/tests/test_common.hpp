/* test_common.hpp - tiny assertion helpers for the PRIA test binaries. */
#ifndef PRIA_TESTS_TEST_COMMON_HPP
#define PRIA_TESTS_TEST_COMMON_HPP

#include <cmath>
#include <cstdio>
#include <cstdlib>

static int g_checks = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        ++g_checks;                                                          \
        if (!(cond)) {                                                       \
            fprintf(stderr, "CHECK FAILED at %s:%d: %s\n", __FILE__,         \
                    __LINE__, #cond);                                        \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                \
    do {                                                                     \
        ++g_checks;                                                          \
        double a_ = (a), b_ = (b), tol_ = (tol);                             \
        if (!(std::fabs(a_ - b_) <= tol_)) {                                 \
            fprintf(stderr,                                                  \
                    "CHECK_NEAR FAILED at %s:%d: |%.17g - %.17g| = %.3e > "  \
                    "%.3e\n",                                                \
                    __FILE__, __LINE__, a_, b_, std::fabs(a_ - b_), tol_);   \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

static inline void test_done(const char* name) {
    printf("%s: PASS (%d checks)\n", name, g_checks);
}

#endif
