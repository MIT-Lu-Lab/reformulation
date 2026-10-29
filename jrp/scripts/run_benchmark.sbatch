#!/usr/bin/env bash
# Run a benchmark pass on one GPU, as a batch job.
#
#     source env.sh && sbatch scripts/run_benchmark.sbatch --pria-cuda
#     source env.sh && sbatch scripts/run_benchmark.sbatch --cupdlpx
#
# A pass that needs no GPU should not hold one, and the vertex baselines are
# the reason that matters: HiGHS and Gurobi are CPU computations, and a CPU
# partition has idle nodes when every GPU is spoken for.  Override the
# directives rather than keeping a second copy of this file:
#
#     sbatch --partition=<cpu-partition> --gres=none \
#            scripts/run_benchmark.sbatch --highs
#
# Raise --time as well for a baseline that is not first-order.  The GPU
# default below is sized for the solvers this is usually pointed at; a vertex
# solve of the largest rungs is in another regime entirely, so ask for a day
# or more on a partition that allows it:
#
#     sbatch --partition=<cpu-partition> --gres=none --time=2-00:00:00 \
#            scripts/run_benchmark.sbatch --gurobi
#
# Pick a CPU partition whose OS matches the one the venv was built on: numpy
# built against a newer glibc dies on an older node with `GLIBC_2.27 not
# found`.  Nodes of one cluster can differ in this.
#
# Submit from the repository root.  Everything after the script name reaches
# run_benchmark.py, including which implementations to run: this script names
# none of them, because which ones a pass is for is the one thing that changes
# between submissions.  The log lands in logs/benchmark-<jobid>.out, unbuffered
# so a long pass is watchable; that directory is created by env.sh and ignored
# by git.  The CSV goes under $PRIA_RESULTS_DIR and is rewritten after every
# solve, so a job that runs out of time still leaves everything it measured.
#
# One backend per job, rather than several in one, because they are not rerun
# at the same rate: pria-cuda is the thing under development and gets measured
# again after every change to it, while cuPDLPx is a fixed baseline that only
# has to be measured once.  Putting them in one job would pay for the baseline
# on every pria-cuda rerun.  The cost is that each job generates the instances
# itself, which for this problem is a real share of the pass -- a JRP instance
# is a dense N x T x T array -- and the risk is that two jobs are not
# comparable unless they run on the same hardware, which is what the next
# paragraph is for.
#
# --gres=gpu:h100:1 names the card rather than taking what is free, and that is
# deliberate: a pass is compared against passes submitted days apart, so the
# card cannot be whatever the scheduler had spare.
#
# Submitting two of these at once needs a dependency, not a second partition:
#
#     a=$(sbatch --parsable scripts/run_benchmark.sbatch --pria-cuda)
#     sbatch --dependency=afterany:$a scripts/run_benchmark.sbatch --cupdlpx
#
# A per-user GPU cap counts an interactive session holding a GPU against it,
# so an unchained second pass can stop on QOSMaxGRESPerUser behind your own
# shell.
#
#SBATCH --job-name=pria-benchmark
# Your cluster's GPU partition.  This line is site-local: set it once.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
# Sized for cuPDLPx, which builds the LP in this process and holds it twice
# over the conversion to CSR; pria-cuda is a child reading its instance from a
# file and needs far less.  Derived from the structure rather than measured,
# with room to be wrong -- `run_benchmark.py --memory` replaces the estimate.
#SBATCH --mem=96G
# Under the six-hour cap that short-queue partitions commonly impose.
# See above for a CPU baseline, which needs much more.
#SBATCH --time=06:00:00
#SBATCH --output=logs/benchmark-%j.out
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
# shellcheck source=/dev/null
source env.sh

# Build pria-cuda only where there is a card to build for, so the same script
# serves a CPU pass on a CPU node.  A CPU backend asked for on a GPU node still
# works; a GPU backend asked for without one reports itself unavailable, which
# is the backends' own contract and not this script's to second-guess.
if nvidia-smi -L >/dev/null 2>&1; then
    gpu=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1)
    CUDA_ARCH=$(echo "$gpu" | sed 's/.*, *//; s/\.//')   # "9.0" -> 90
    export CUDA_ARCH
    make -C pria-cuda tools
    # The backend looks for pria_solve_file in $PRIA_BIN_DIR, and the binaries
    # are built per architecture.  Point that at this job's own build rather
    # than installing over what an interactive session left there for its card.
    PRIA_BIN_DIR="$PRIA_ROOT/pria-cuda/build/sm$CUDA_ARCH"
    export PRIA_BIN_DIR
    echo "== $(hostname): $gpu (sm_$CUDA_ARCH)"
else
    echo "== $(hostname): no GPU, CPU backends only"
fi
exec "$PRIA_PYTHON" -u scripts/run_benchmark.py "$@"
