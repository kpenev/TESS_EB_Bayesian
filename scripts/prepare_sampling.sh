#!/bin/bash

# Find the starting walkers for sampling one TIC from scratch:
#   1. sed_grid_search.py   SED-only fits from a grid of starting points
#   2. sed_seeded_fits.py   light curve + SED fits from the SED seeds, at a
#                           grid of w covering both eclipse geometries
#   3. select_families.py   group the good fit end points into families
#   4. initial_walkers.py   walkers around the best point of each family
# Each step is skipped if its output exists, so an interrupted run resumes
# (sed_seeded_fits.py only writes its output at the end, so it restarts from
# scratch if interrupted). Plots of the SED search and, per family, of its
# best fit with its initial walkers (plot_init_models.py) are made along the
# way, as PNG; a failed plot does not stop the pipeline. The review images
# are sed_grid_corner_TIC.png and init_models_k.png (one per family, best
# first, listed in init_models.txt). If
# a step fails, OUTPUT_DIR/prepare_failed.txt records it (TIC, step, end of
# its log).
#
# Usage:
#   prepare_sampling.sh TIC OUTPUT_DIR SAMPLES_FNAME [NUM_PARALLEL [NUM_WALKERS]]
#
# Environment:
#   PYTHON       the Python command (default python3), e.g.
#                "conda run -n py311 python3"
#   WALKER_SEED  random seed for initial_walkers.py (default: random)
#
# Runs from the directory of this script, like mcmc_sampling.py. The
# intermediate results and plots go to OUTPUT_DIR. The walkers are saved as
# the starting positions of a new samples file SAMPLES_FNAME, marked as a
# final run; mcmc_sampling.py with that samples file samples from them.
# NUM_WALKERS (default 256) must match its scenario grid (see
# initial_walkers.py), and both must run the same git commit.

set -eu

# The steps parallelize over processes; threaded BLAS/LAPACK (one thread per
# core in every process by default) would oversubscribe the CPUs. Values
# already set in the environment are kept.
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

tic=$1
output_dir=$2
samples=$3
num_parallel=${4:-16}
num_walkers=${5:-256}
python=${PYTHON:-python3}

cd "$(dirname "$0")"
mkdir -p "${output_dir}" "$(dirname "${samples}")"

sed_results=${output_dir}/sed_grid_search_${tic}.ecsv
seeded_results=${output_dir}/sed_seeded_fits_${tic}.ecsv
families=${output_dir}/families.npy

# If the pipeline stops with an error, the record of the failure (for the
# review: the system becomes prepare_failed) is written here; a new run
# starts by removing the record of an earlier one.
failure_record=${output_dir}/prepare_failed.txt
rm -f "${failure_record}"
# The log of the step running (set by run_step), named in the record.
current_log=

write_failure_record() {
    {
        echo "TIC: ${tic}"
        echo "Step: $(basename "${current_log:-setup}" .log)"
        echo "Date: $(date --iso-8601=seconds)"
        echo "Commit: $(git rev-parse --short HEAD 2>/dev/null)"
        if [ -n "${current_log}" ]; then
            echo "Last 50 lines of ${current_log}:"
            tail -n 50 "${current_log}"
        fi
    } > "${failure_record}"
}
trap '[ $? -eq 0 ] || write_failure_record' EXIT

# run_step OUTPUT LOG COMMAND...: run COMMAND (output to LOG) unless OUTPUT
# exists; stop the pipeline if it fails.
run_step() {
    local output=$1 log=$2
    shift 2
    if [ -e "${output}" ]; then
        echo "$(date) Skipping, ${output} exists."
        return 0
    fi
    echo "$(date) Running: $*"
    current_log=${log}
    if ! "$@" > "${log}" 2>&1; then
        echo "$(date) Failed, see ${log}."
        return 1
    fi
}

# plot_step OUTPUT LOG COMMAND...: like run_step, but a failure only warns.
plot_step() {
    run_step "$@" || echo "$(date) Plot failed, continuing."
}

run_step "${sed_results}" "${output_dir}/sed_grid_search.log" \
    ${python} -u sed_grid_search.py "${tic}" \
    --num-parallel "${num_parallel}" \
    --output "${sed_results}"

plot_step "${output_dir}/sed_grid_corner_${tic}.pdf" \
    "${output_dir}/plot_sed_grid_search.log" \
    ${python} -u plot_sed_grid_search.py "${sed_results}" \
    "${output_dir}/sed_grid_corner_${tic}" \
    --title "TIC ${tic}: "

run_step "${seeded_results}" "${output_dir}/sed_seeded_fits.log" \
    ${python} -u sed_seeded_fits.py "${tic}" \
    --sed-results "${sed_results}" \
    --max-nfev 2000 \
    --time-limit 21600 \
    --num-parallel "${num_parallel}" \
    --output "${seeded_results}"

run_step "${families}" "${output_dir}/select_families.log" \
    ${python} -u select_families.py "${tic}" "${seeded_results}" \
    "${families}" \
    --num-parallel "${num_parallel}"

run_step "${samples}" "${output_dir}/initial_walkers.log" \
    ${python} -u initial_walkers.py "${tic}" "${families}" "${samples}" \
    --num-walkers "${num_walkers}" \
    --num-parallel "${num_parallel}" \
    ${WALKER_SEED:+--seed "${WALKER_SEED}"}

plot_step "${output_dir}/init_models.txt" \
    "${output_dir}/plot_init_models.log" \
    ${python} -u plot_init_models.py "${tic}" "${samples}" \
    "${output_dir}/init_models"

echo "$(date) Done: ${samples}"
