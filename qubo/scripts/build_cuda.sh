#!/usr/bin/env bash
# Build the CUDA binaries and install them into $PRAQ_BIN_DIR.
#
#     source env.sh && scripts/build_cuda.sh [praq-cuda|cupdlpx]
#
# With no arguments both are attempted.  Each target is skipped with a message
# rather than failing the script when its prerequisites are missing, so this is
# runnable on a laptop to check that the sources at least parse.
set -uo pipefail

: "${PRAQ_ROOT:?run 'source env.sh' first}"
: "${PRAQ_BIN_DIR:?run 'source env.sh' first}"
: "${CUDA_ARCH:?run 'source env.sh' first}"

status=0

have_nvcc() {
    if command -v "$NVCC" >/dev/null 2>&1 || [ -x "$NVCC" ]; then return 0; fi
    echo "  no nvcc at '$NVCC' -- set CUDA_HOME/NVCC in env.local.sh" >&2
    return 1
}

have_cmake() {
    if command -v cmake >/dev/null 2>&1; then return 0; fi
    echo "  no cmake on PATH -- cuPDLPx needs 3.20 or newer" >&2
    return 1
}

build_praq_cuda() {
    echo "praq-cuda (CUDA_ARCH=$CUDA_ARCH)"
    have_nvcc || return 1
    make -C "$PRAQ_ROOT/praq-cuda" tools lib || return 1
    local built="$PRAQ_ROOT/praq-cuda/build/sm$CUDA_ARCH"
    install -m 755 "$built/praq_solve_file" "$built/libpraq.so" "$PRAQ_BIN_DIR/" || return 1
    echo "  installed $PRAQ_BIN_DIR/praq_solve_file and libpraq.so"
}

build_cupdlpx() {
    echo "cupdlpx"
    if [ ! -d "${CUPDLPX_DIR:-}" ]; then
        echo "  no checkout at '${CUPDLPX_DIR:-unset}'" >&2
        echo "  git clone https://github.com/MIT-Lu-Lab/cuPDLPx \"\$CUPDLPX_DIR\"" >&2
        return 1
    fi
    have_nvcc || return 1
    have_cmake || return 1
    # Upstream's README builds with "cmake -B build && cmake --build build" and
    # leaves the binary at build/cupdlpx; this is that, plus Release and a single
    # architecture.  Its CMakeLists otherwise fans out over every SM from 75 up,
    # which is minutes of nvcc for code nothing here runs -- and it does mean the
    # binary only runs on a $CUDA_ARCH card, so a move to a different GPU is a
    # rebuild, not just a resubmit.
    cmake -S "$CUPDLPX_DIR" -B "$CUPDLPX_DIR/build" \
          -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" || return 1
    cmake --build "$CUPDLPX_DIR/build" -j || return 1
    install -m 755 "$CUPDLPX_DIR/build/cupdlpx" "$CUPDLPX_BIN" || return 1
    echo "  installed $CUPDLPX_BIN"
}

targets=("$@")
[ ${#targets[@]} -eq 0 ] && targets=(praq-cuda cupdlpx)

for target in "${targets[@]}"; do
    case "$target" in
        praq-cuda) build_praq_cuda || status=1 ;;
        cupdlpx)   build_cupdlpx   || status=1 ;;
        *) echo "unknown target: $target" >&2; exit 2 ;;
    esac
done

[ $status -eq 0 ] || echo "one or more targets did not build (see above)" >&2
exit $status
