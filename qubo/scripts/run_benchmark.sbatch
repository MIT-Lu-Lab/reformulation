#!/usr/bin/env bash
# Run a benchmark pass on one GPU, as a batch job.
#
#     source env.sh && sbatch scripts/run_benchmark.sbatch --praq-cuda
#     source env.sh && sbatch scripts/run_benchmark.sbatch --cupdlpx
#
# A pass that needs no GPU should not hold one, and --maxflow is the whole
# reason that matters: the roof dual as a maximum flow is a CPU computation,
# and a CPU partition has idle nodes when every GPU is spoken for.  Override
# the directives rather than keeping a second copy of this file:
#
#     sbatch --partition=<cpu-partition> --gres=none \
#            scripts/run_benchmark.sbatch --maxflow
#
# Raise --time as well for a CPU baseline that is not first-order.  The GPU
# default below is sized for the solvers this is usually pointed at, and a
# branch-and-bound baseline is in another regime entirely, and spends most of
# its time on instances it then abandons at the per-instance limit.  Ask for a
# day or more, on a partition that allows it:
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
# by git.  The CSV goes under $PRAQ_RESULTS_DIR and is rewritten after every
# solve, so a job that runs out of time still leaves everything it measured.
#
# One backend per job, rather than both in one, because the two are not rerun
# at the same rate: praq-cuda is the thing under development and gets measured
# again after every change to it, while cuPDLPx is a fixed baseline that only
# has to be measured once.  Putting them in one job would pay for the baseline
# on every praq-cuda rerun.  The cost is that each job generates the instances
# itself -- about as long as solving them -- and the risk is that the two jobs
# are not comparable unless they run on the same hardware, which is what the
# next paragraph is for.
#
# --gres=gpu:h100:1 names the card rather than taking what is free, and that is
# deliberate: a pass is compared against passes submitted days apart, so the
# card cannot be whatever the scheduler had spare.  The L40S is excluded as a
# different class of card, and the H100 is named over the H200 because the
# pool has more H100 nodes to draw from.
#
# Submitting two of these at once needs a dependency, not a second partition:
#
#     a=$(sbatch --parsable scripts/run_benchmark.sbatch --praq-cuda)
#     sbatch --dependency=afterany:$a scripts/run_benchmark.sbatch --cupdlpx
#
# A per-user GPU cap counts an interactive session holding a GPU against it,
# so an unchained second pass can stop on QOSMaxGRESPerUser behind your own
# shell, which reads as a busy cluster and is not one.  Running the passes
# back to back is the reliable order.
#
#SBATCH --job-name=praq-benchmark
# Your cluster's GPU partition.  This line is site-local: set it once.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
# Sized for cuPDLPx, which is the hungry one: it builds the LP in this
# process, where praq-cuda is a child reading its instance from a file.  A
# quarter of an H100 node is proportionate for a single-GPU job and has not
# cost queue time; `run_benchmark.py --memory` is how to check it against a
# measurement.
#SBATCH --mem=250G
# Under the six-hour cap that short-queue partitions commonly impose, and
# comfortably above a GPU pass.  See above for a CPU baseline.
#SBATCH --time=06:00:00
#SBATCH --output=logs/benchmark-%j.out
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
# shellcheck source=/dev/null
source env.sh

# Build praq-cuda only where there is a card to build for, so the same script
# serves a CPU pass on a CPU node.  A CPU backend asked for on a GPU node still
# works; a GPU backend asked for without one reports itself unavailable, which
# is the backends' own contract and not this script's to second-guess.
if nvidia-smi -L >/dev/null 2>&1; then
    gpu=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1)
    CUDA_ARCH=$(echo "$gpu" | sed 's/.*, *//; s/\.//')   # "9.0" -> 90
    export CUDA_ARCH
    make -C praq-cuda tools
    # The backend looks for praq_solve_file in $PRAQ_BIN_DIR, and the binaries
    # are built per architecture.  Point that at this job's own build rather
    # than installing over what an interactive session left there for its card.
    PRAQ_BIN_DIR="$PRAQ_ROOT/praq-cuda/build/sm$CUDA_ARCH"
    export PRAQ_BIN_DIR
    echo "== $(hostname): $gpu (sm_$CUDA_ARCH)"
else
    echo "== $(hostname): no GPU, CPU backends only"
fi
exec "$PRAQ_PYTHON" -u scripts/run_benchmark.py "$@"
