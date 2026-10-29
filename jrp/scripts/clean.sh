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

: "${PRIA_ROOT:?run 'source env.sh' first}"
: "${PRIA_RESULTS_DIR:?run 'source env.sh' first}"
: "${PRIA_BIN_DIR:?run 'source env.sh' first}"

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
    echo "  ${1#"$PRIA_ROOT"/}"
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
drop "$PRIA_ROOT/pria-cuda/build"
drop "$PRIA_ROOT/docs/_build"
drop "$PRIA_BIN_DIR/pria_solve_file"
drop "$PRIA_BIN_DIR/libpria.so"

# $PRUNED is pruned first, and silently.  A virtualenv at the repository root
# holds thousands of __pycache__ directories belonging to installed packages,
# and while removing them is harmless it is not this project's business, takes
# a while, and buries the handful of lines a person is actually reading.
PRUNED='( -name .venv -o -name .git ) -prune -o'

# shellcheck disable=SC2086  # PRUNED is a find expression, not one word
drop_found "python caches" "$PRIA_ROOT" $PRUNED \
    \( -name __pycache__ -o -name '*.egg-info' -o -name .pytest_cache \
       -o -name .mypy_cache \) -prune
# .pyc files inside __pycache__ went with the directory above; only loose ones
# and .DS_Store remain to list here.
# shellcheck disable=SC2086
drop_found "stray files" "$PRIA_ROOT" $PRUNED \
    -not -path '*/__pycache__/*' \
    \( -name '*.pyc' -o -name .DS_Store \) -type f

# Exported LP files: regenerated on demand, by baselines/model.py's write_mps.
echo "exported MPS"
drop "$PRIA_RESULTS_DIR/mps"

# Instances staged for the pria-cuda CLI: each call removes its own, so only a
# killed run leaves anything here.  They are 8 N T^2 bytes apiece, so a killed
# run at the top of the ladder leaves half a gigabyte behind.
echo "pria-cuda staging"
drop "$PRIA_RESULTS_DIR/pria-cuda"

[ $results -eq 1 ] && drop_found "benchmark results (--results)" \
    "$PRIA_RESULTS_DIR" -mindepth 1 -maxdepth 1
# Redrawing figures needs the CSVs, which --results may just have taken.
[ $figs -eq 1 ] && drop_found "figures (--figs)" \
    "$PRIA_RESULTS_DIR/figs" -mindepth 1 -maxdepth 1

if [ $dry -eq 1 ]; then
    echo "(dry run: $removed item(s) would be removed)"
else
    echo "removed $removed item(s)"
fi
exit 0
