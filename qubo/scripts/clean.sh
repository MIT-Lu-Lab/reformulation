#!/usr/bin/env bash
# Remove generated files.
#
#     source env.sh && scripts/clean.sh [--results] [--figs] [--all] [-n]
#
# By default this removes only what is cheap to recreate: build products, Python
# caches, compiled binaries and exported MPS files.  Benchmark results are
# experiment output that cannot be recreated at all, and figures are usually
# the thing you are looking at even though redrawing them is one
# plot_results.py away -- so each needs its own flag.  Pass -n to list what
# would go without removing anything.
set -euo pipefail

: "${PRAQ_ROOT:?run 'source env.sh' first}"
: "${PRAQ_RESULTS_DIR:?run 'source env.sh' first}"
: "${PRAQ_BIN_DIR:?run 'source env.sh' first}"

results=0 figs=0 dry=0
for arg in "$@"; do
    case "$arg" in
        --results) results=1 ;;
        --figs)    figs=1 ;;
        --all)     results=1; figs=1 ;;
        -n|--dry-run) dry=1 ;;
        -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

removed=0

drop() {   # drop <path>
    [ -e "$1" ] || return 0
    echo "  ${1#"$PRAQ_ROOT"/}"
    removed=$((removed + 1))
    [ $dry -eq 1 ] || rm -rf -- "$1"
}

# `find -print0` + read keeps paths with spaces intact, and avoids mapfile,
# which macOS's bash 3.2 does not have.
drop_found() {   # drop_found <description> <root> <find args...>
    local description="$1" root="$2"; shift 2
    [ -d "$root" ] || return 0
    local listed=0 path
    while IFS= read -r -d '' path; do
        [ $listed -eq 0 ] && { echo "$description"; listed=1; }
        drop "$path"
    done < <(find "$root" "$@" -print0 2>/dev/null)
}

echo "build products"
drop "$PRAQ_ROOT/praq-cuda/build"
drop "$PRAQ_ROOT/docs/_build"
drop "$PRAQ_BIN_DIR/praq_solve_file"
drop "$PRAQ_BIN_DIR/libpraq.so"

drop_found "python caches" "$PRAQ_ROOT" \
    \( -name __pycache__ -o -name '*.egg-info' -o -name .pytest_cache \
       -o -name .mypy_cache \) -prune
# .pyc files inside __pycache__ went with the directory above; only loose ones
# and .DS_Store remain to list here.
drop_found "stray files" "$PRAQ_ROOT" \
    -not -path '*/__pycache__/*' \
    \( -name '*.pyc' -o -name .DS_Store \) -type f

# Exported LP files: regenerated on demand by the cuPDLPx backend.
echo "exported MPS"
drop "$PRAQ_RESULTS_DIR/mps"

# Instances staged for the praq-cuda CLI: each call removes its own, so only a
# killed run leaves anything here.
echo "praq-cuda staging"
drop "$PRAQ_RESULTS_DIR/praq-cuda"

[ $results -eq 1 ] && drop_found "benchmark results (--results)" \
    "$PRAQ_RESULTS_DIR" -mindepth 1 -maxdepth 1
# Figures and tables are one scripts/plot_results.py away from being back, but
# they are usually what you are in the middle of reading, and redrawing them
# needs the CSVs to still be there.  Hence a flag of their own rather than the
# default sweep.
[ $figs -eq 1 ] && drop_found "figures (--figs)" \
    "$PRAQ_RESULTS_DIR/figs" -mindepth 1 -maxdepth 1

if [ $dry -eq 1 ]; then
    echo "(dry run: $removed item(s) would be removed)"
else
    echo "removed $removed item(s)"
fi
exit 0
