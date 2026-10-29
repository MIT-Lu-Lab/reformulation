# PRAQ CUDA solver.
#
#   make            build the static and the shared library
#   make tools      build the file-based CLI the harness drives
#   make lib        build the shared library scripts/tune_praq.py loads
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
LIB   := $(BUILD)/libpraq.a
SO    := $(BUILD)/libpraq.so

.PHONY: all tools lib test clean

all: $(LIB) $(SO)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/praq.o: src/praq.cu src/solver.cuh include/praq/praq.h | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(LIB): $(BUILD)/praq.o
	ar rcs $@ $^

$(SO): $(BUILD)/praq.o
	$(NVCC) $(NVCCFLAGS) -shared $^ -o $@

lib: $(SO)

$(BUILD)/praq_solve_file: tools/praq_solve_file.cu $(LIB) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $< $(LIB) -o $@

tools: $(BUILD)/praq_solve_file

$(BUILD)/test_solver: tests/test_solver.cu $(LIB) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $< $(LIB) -o $@

test: $(BUILD)/test_solver
	$(BUILD)/test_solver

clean:
	rm -rf build
