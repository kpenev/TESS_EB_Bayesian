#!/bin/bash

# Re-initialize the walkers of targets prepared before the family proposals
# were saved and before the 0.03 mag sed_sys cap (the batch-1 preparations,
# TEC-124, TEC-128), keeping their SED search, seeded fits and families.
#
# For each target in OUTPUT_DIR (the OUTPUT_DIR of prepare_batch.sh) whose
# preparation finished and whose samples file has no family proposals:
#   1. convert_families.py converts OUTPUT_DIR/<TIC>/families.npy to the
#      current sed_sys range and refits it, writing families.npz (it keeps the
#      families' isochrone interpolation: linear for batch-1),
#   2. the old samples file and the outputs made from it (initial_walkers.log,
#      init_models*, plot_init_models.log) are moved to
#      OUTPUT_DIR/<TIC>/superseded-<time>/,
#   3. prepare_sampling.sh is run again: the steps whose outputs exist are
#      skipped, so it runs initial_walkers.py (saving the family proposals),
#      plot_init_models.py and the export to EXCHANGE_DIR.
# Its log is OUTPUT_DIR/<TIC>/reinitialize.out (prepare_sampling.out keeps the
# original preparation's).
#
# A target counts as finished when its prepare_sampling.out (written by
# prepare_batch.sh) ends with prepare_sampling.sh's "Done:" line, so targets a
# batch is still preparing (or plotting and exporting) are left alone; run
# this again to pick up targets finished since. A target is done when its
# samples file holds the family proposals; rerunning after an interruption
# resumes the targets left halfway (families.npz without a samples file).
#
# Usage:
#   reinitialize_prepared.sh OUTPUT_DIR [--tics TIC_LIST] [TARGETS_AT_ONCE
#                            [WORKERS_PER_TARGET]]
#
#   OUTPUT_DIR          as for prepare_batch.sh: OUTPUT_DIR/<TIC>/ and
#                       OUTPUT_DIR/samples_seeded/tess<TIC>_samples.h5
#   --tics TIC_LIST     only these targets, in this order (one TIC per line;
#                       blank lines and lines starting with # are skipped)
#   TARGETS_AT_ONCE     default 2
#   WORKERS_PER_TARGET  default 12
#
# Environment:
#   PYTHON        as for prepare_sampling.sh; for a worktree of the new code
#                 next to a running batch, e.g. "conda run -n tess-ebs env
#                 PYTHONPATH=<worktree>/scripts:<gpm worktree parent> python3"
#   EXCHANGE_DIR  required: where prepare_sampling.sh exports each target
#   WALKER_SEED   as for prepare_sampling.sh
#
# Runs the scripts next to this one. Never run it while a prepare_batch.sh of
# the new code works on the same OUTPUT_DIR.

set -eu

output_dir=$(realpath -m "$1")
shift
tic_list=
if [ "${1:-}" = "--tics" ]; then
    tic_list=$(realpath "$2")
    shift 2
fi
targets_at_once=${1:-2}
workers_per_target=${2:-12}
python=${PYTHON:-python3}
script_dir=$(cd "$(dirname "$0")" && pwd)

if [ -z "${EXCHANGE_DIR:-}" ]; then
    echo "EXCHANGE_DIR must be set: the re-initialized targets are exported." >&2
    exit 1
fi

# List the targets needing work, one per line: those finished by the batch
# whose samples file has no family proposals, and those left halfway by an
# earlier run of this script (families.npz but no samples file).
pending=$(mktemp)
trap 'rm -f "${pending}"' EXIT
${python} -c '
import sys
from glob import glob
from os import path
import h5py

output_dir, tic_list = sys.argv[1], sys.argv[2]
if tic_list:
    with open(tic_list, encoding="ascii") as tic_file:
        tics = [
            line.split()[0]
            for line in tic_file
            if line.strip() and not line.lstrip().startswith("#")
        ]
else:
    tics = sorted(
        path.basename(path.dirname(fname))
        for fname in glob(path.join(output_dir, "*", "families.npy"))
    )
for tic in tics:
    target_dir = path.join(output_dir, tic)
    samples = path.join(output_dir, "samples_seeded", f"tess{tic}_samples.h5")
    if not path.exists(path.join(target_dir, "families.npy")):
        print(f"TIC {tic}: no families.npy, skipped.", file=sys.stderr)
        continue
    if not path.exists(samples):
        if path.exists(path.join(target_dir, "families.npz")):
            print(tic)
        continue
    with h5py.File(samples, "r") as samples_file:
        if "family_proposals" in samples_file:
            continue
    log_fname = path.join(target_dir, "prepare_sampling.out")
    if not path.exists(log_fname):
        continue
    with open(log_fname, encoding="utf-8") as log_file:
        lines = log_file.read().splitlines()
    if lines and " Done: " in lines[-1]:
        print(tic)
' "${output_dir}" "${tic_list}" > "${pending}"
echo "$(date) Re-initializing $(wc -l < "${pending}") targets."

reinitialize_one() {
    local tic=$1 output_dir=$2 num_parallel=$3
    local target_dir=${output_dir}/${tic}
    local samples=${output_dir}/samples_seeded/tess${tic}_samples.h5
    local families=${target_dir}/families.npz

    if [ ! -e "${families}" ]; then
        # Written under another name and renamed when complete.
        local converting=${target_dir}/families.converting.npz
        rm -f "${converting}"
        echo "$(date) Running: convert_families.py ${tic}"
        ${python} -u convert_families.py "${tic}" \
            "${target_dir}/families.npy" "${converting}" \
            --num-parallel "${num_parallel}" \
            > "${target_dir}/convert_families.log" 2>&1 \
            || { echo "$(date) Failed, see convert_families.log."; return 1; }
        mv "${converting}" "${families}"
    fi

    if [ -e "${samples}" ]; then
        local superseded
        superseded=${target_dir}/superseded-$(date +%Y%m%d%H%M%S)
        mkdir "${superseded}"
        mv "${samples}" "${superseded}/"
        for fname in "${samples}.unsaved_steps" \
                "${target_dir}/initial_walkers.log" \
                "${target_dir}/plot_init_models.log" \
                "${target_dir}"/init_models*; do
            if [ -e "${fname}" ]; then
                mv "${fname}" "${superseded}/"
            fi
        done
        echo "$(date) Moved the old walkers and their plots to ${superseded}."
    fi

    "${script_dir}/prepare_sampling.sh" "${tic}" "${target_dir}" \
        "${samples}" "${num_parallel}"
}
export -f reinitialize_one
export python script_dir

xargs -P "${targets_at_once}" -I{} \
    bash -c 'cd "$3" && reinitialize_one "$0" "$1" "$2" \
        > "$1/$0/reinitialize.out" 2>&1 \
        || echo "$(date) TIC $0 failed, see $1/$0/reinitialize.out"' \
    {} "${output_dir}" "${workers_per_target}" "${script_dir}" \
    < "${pending}"
echo "$(date) All targets done."
