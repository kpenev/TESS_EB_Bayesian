#!/bin/bash
# Prepare the starting walkers of many targets in parallel on one machine,
# without SLURM: runs prepare_sampling.sh for each target to prepare, several
# at a time.
#
# Usage:
#   prepare_batch.sh OUTPUT_DIR [--priority TIC_LIST] [TARGETS_AT_ONCE
#                    [WORKERS_PER_TARGET]]
#
#   OUTPUT_DIR          each target's results go to OUTPUT_DIR/<TIC>/ (its log
#                       is OUTPUT_DIR/<TIC>/prepare_sampling.out), its samples
#                       file to OUTPUT_DIR/samples_seeded/tess<TIC>_samples.h5
#   --priority TIC_LIST targets to prepare first, one TIC per line (blank lines
#                       and lines starting with # are skipped)
#   TARGETS_AT_ONCE     default 8
#   WORKERS_PER_TARGET  default 16 (the NUM_PARALLEL of prepare_sampling.sh)
#
# What to prepare comes from the review machine's database: first,
# prepare_pending.py applies the re-preparation requests not applied yet and
# lists the targets with status prepare that have no samples file yet, those
# of TIC_LIST first (see prepare_pending.py). The database snapshot is read
# from EXCHANGE_DIR/to_prep/; without it, exactly the targets of TIC_LIST are
# prepared, and without either nothing is.
#
# Environment:
#   PYTHON        as for prepare_sampling.sh (e.g. the full path of the
#                 environment's python3)
#   EXCHANGE_DIR  the folder shared with the review machine: the database
#                 snapshots are read from it, and each finished target is
#                 exported to it (see prepare_sampling.sh)
#
# A target whose run fails does not stop the others. Rerunning resumes:
# prepare_sampling.sh skips the steps whose output exists. The list of targets
# is made when the batch starts; rerun it to pick up targets or requests added
# since. To keep it running after logging out, e.g.:
#   setsid nohup ./prepare_batch.sh OUT > OUT.log 2>&1 < /dev/null &
set -eu

output_dir=$(realpath -m "$1")
shift
priority=
if [ "${1:-}" = "--priority" ]; then
    priority=$(realpath "$2")
    shift 2
fi
targets_at_once=${1:-8}
workers_per_target=${2:-16}
python=${PYTHON:-python3}
script_dir=$(cd "$(dirname "$0")" && pwd)

mkdir -p "${output_dir}/samples_seeded"

# Apply new re-preparation requests and list the targets to prepare; stop if
# that fails.
pending=$(mktemp)
trap 'rm -f "${pending}"' EXIT
${python} "${script_dir}/prepare_pending.py" "${output_dir}" \
    ${priority:+--priority "${priority}"} > "${pending}"
echo "$(date) Preparing $(wc -l < "${pending}") targets."

xargs -P "${targets_at_once}" -I{} \
    bash -c 'mkdir -p "$2/$1" \
        && "$0" "$1" "$2/$1" "$2/samples_seeded/tess$1_samples.h5" "$3" \
        > "$2/$1/prepare_sampling.out" 2>&1 \
        || echo "$(date) TIC $1 failed, see $2/$1/"' \
    "${script_dir}/prepare_sampling.sh" {} "${output_dir}" \
    "${workers_per_target}" < "${pending}"
echo "$(date) All targets done."
