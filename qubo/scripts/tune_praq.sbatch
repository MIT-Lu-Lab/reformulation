#!/usr/bin/env bash
# Tune PRAQ's restart schedule on one GPU, as a batch job.
#
#     source env.sh && sbatch scripts/tune_praq.sbatch [tune_praq.py options]
#
# Submit from the repository root.  The log lands in logs/tune-praq-<jobid>.out,
# unbuffered so it shows progress while the job runs; that directory is created
# by env.sh and ignored by git, because an #SBATCH directive cannot expand a
# variable and a log written to the root gets swept up by `git add -A`
# (commit 95306b7 did exactly that with three of them);
# the CSV of every configuration goes under
# $PRAQ_RESULTS_DIR/tuning as the script's own output says.  Options after the
# script name reach tune_praq.py, so a different `--time-budget` or a subset of
# `--buckets` needs no edit here.  It trains on every bucket and confirms the
# leaders on the benchmark itself, which is tune_praq.py's default.
#
# Any GPU will do, and the job asks for whichever frees first on either of the
# partitions below: the schedule is scored in iterations, which do not depend
# on the GPU, so the GPU only sets how many configurations the budget buys.
# The H200s are usually the scarcer resource.  The
# binaries are built per architecture and not installed: the build lands in
# praq-cuda/build/sm<cc> for the GPU the job lands on, beside whatever the
# interactive session built, so $PRAQ_BIN_DIR is untouched.  The partitions
# are the ones this account can submit to; change them if yours differ.
#
#SBATCH --job-name=tune-praq
# Your cluster's GPU partition.  This line is site-local: set it once.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=logs/tune-praq-%j.out
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
# shellcheck source=/dev/null
source env.sh

gpu=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1)
CUDA_ARCH=$(echo "$gpu" | sed 's/.*, *//; s/\.//')   # "9.0" -> 90
export CUDA_ARCH
make -C praq-cuda lib

echo "== $(hostname): $gpu (sm_$CUDA_ARCH)"
exec "$PRAQ_PYTHON" -u scripts/tune_praq.py \
    --library "praq-cuda/build/sm$CUDA_ARCH/libpraq.so" \
    --configs 1000000 --time-budget $((3 * 3600)) \
    "$@"
