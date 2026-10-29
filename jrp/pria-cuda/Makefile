# PRIA CUDA solver.
#
#   make            build the static and the shared library
#   make tools      build the file-based CLI the harness drives
#   make lib        build the shared library scripts/tune_pria.py loads
#   make bench      build the standalone benchmark driver
#   make test       build + run the test suite (needs a GPU)
#   make clean
#
# CUDA_ARCH and NVCC have NO default here on purpose.  ../env.sh is their single
# home; `source env.sh` before building, or pass them on the command line.
#
# Everything lands under build/sm$(CUDA_ARCH): the binaries are GPU-specific,
# and this is what lets a batch job build for an H200 (`make CUDA_ARCH=90`)
# beside an interactive session's L40S build without either clobbering the other.

ifndef CUDA_ARCH
$(error CUDA_ARCH is not set -- run `source env.sh` from the repository root, \
        or `make CUDA_ARCH=<sm>` (89 = L40S/Ada, 80 = A100, 90 = H100))
endif

NVCC ?= nvcc

NVCCFLAGS ?= -O3 -std=c++17 --extended-lambda --expt-relaxed-constexpr -lineinfo \
             -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) \
             -Iinclude -Isrc -Xcompiler -Wall -Xcompiler -fPIC

BUILD := build/sm$(CUDA_ARCH)
LIB   := $(BUILD)/libpria.a
SO    := $(BUILD)/libpria.so

TESTS := $(BUILD)/test_layouts $(BUILD)/test_projection $(BUILD)/test_step \
         $(BUILD)/test_solver

.PHONY: all tools lib bench test clean

all: $(LIB) $(SO)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/pria.o: src/pria.cu src/solver.cuh src/projection.cuh \
                 src/layouts.cuh src/common.cuh include/pria/pria.h | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(LIB): $(BUILD)/pria.o
	ar rcs $@ $^

$(SO): $(BUILD)/pria.o
	$(NVCC) $(NVCCFLAGS) -shared $^ -o $@

lib: $(SO)

$(BUILD)/pria_solve_file: tools/pria_solve_file.cu $(LIB) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $< $(LIB) -o $@

tools: $(BUILD)/pria_solve_file

$(BUILD)/pria_bench: bench/pria_bench.cu tests/ref_impl.hpp $(LIB) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -Itests $< $(LIB) -o $@

bench: $(BUILD)/pria_bench

$(BUILD)/test_%: tests/test_%.cu tests/ref_impl.hpp $(LIB) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -Itests $< $(LIB) -o $@

test: $(TESTS)
	@set -e; for t in $(TESTS); do echo "== $$t"; $$t; done; echo "ALL TESTS PASSED"

clean:
	rm -rf build
