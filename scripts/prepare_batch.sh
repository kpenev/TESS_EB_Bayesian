#!/bin/bash
# Prepare the starting walkers of many targets in parallel on one machine,
# without SLURM: runs prepare_sampling.sh for each TIC of a list, several at a
# time.
#
# Usage:
#   prepare_batch.sh TIC_LIST OUTPUT_DIR [TARGETS_AT_ONCE [WORKERS_PER_TARGET]]
#
#   TIC_LIST            one TIC per line; blank lines and lines starting with
#                       # are skipped
#   OUTPUT_DIR          each target's results go to OUTPUT_DIR/<TIC>/ (its log
#                       is OUTPUT_DIR/<TIC>/prepare_sampling.out), its samples
#                       file to OUTPUT_DIR/samples_seeded/tess<TIC>_samples.h5
#   TARGETS_AT_ONCE     default 8
#   WORKERS_PER_TARGET  default 16 (the NUM_PARALLEL of prepare_sampling.sh)
#
# Environment: PYTHON, as for prepare_sampling.sh (e.g. the full path of the
# environment's python3).
#
# A target whose run fails does not stop the others. Rerunning resumes:
# prepare_sampling.sh skips the steps whose output exists. To keep it running
# after logging out, e.g.:
#   setsid nohup ./prepare_batch.sh LIST OUT > OUT.log 2>&1 < /dev/null &
set -eu

tic_list=$(realpath "$1")
output_dir=$(realpath -m "$2")
targets_at_once=${3:-8}
workers_per_target=${4:-16}
script_dir=$(cd "$(dirname "$0")" && pwd)

mkdir -p "${output_dir}/samples_seeded"

# Strip Windows line endings, comments and blank lines; one TIC per line.
tr -d '\r' < "${tic_list}" \
    | grep -v -e '^[[:space:]]*#' -e '^[[:space:]]*$' \
    | xargs -P "${targets_at_once}" -I{} \
        bash -c 'mkdir -p "$2/$1" \
            && "$0" "$1" "$2/$1" "$2/samples_seeded/tess$1_samples.h5" "$3" \
            > "$2/$1/prepare_sampling.out" 2>&1 \
            || echo "$(date) TIC $1 failed, see $2/$1/"' \
        "${script_dir}/prepare_sampling.sh" {} "${output_dir}" \
        "${workers_per_target}"
echo "$(date) All targets of ${tic_list} done."
