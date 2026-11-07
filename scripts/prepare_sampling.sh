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
# way; a failed plot does not stop the pipeline.
#
# Usage: prepare_sampling.sh TIC OUTPUT_DIR [NUM_PARALLEL [NUM_WALKERS]]
#
# Environment:
#   PYTHON       the Python command (default python3), e.g.
#                "conda run -n py311 python3"
#   WALKER_SEED  random seed for initial_walkers.py (default: random)
#
# Runs from the directory of this script, like mcmc_sampling.py. The walkers
# end up in OUTPUT_DIR/initial_walkers.npy, for
#   mcmc_sampling.py TIC --initial-points OUTPUT_DIR/initial_walkers.npy
# NUM_WALKERS (default 256) must match the walkers of the sampling run.

set -eu

tic=$1
output_dir=$2
num_parallel=${3:-16}
num_walkers=${4:-256}
python=${PYTHON:-python3}

cd "$(dirname "$0")"
mkdir -p "${output_dir}"

sed_results=${output_dir}/sed_grid_search_${tic}.ecsv
seeded_results=${output_dir}/sed_seeded_fits_${tic}.ecsv
families=${output_dir}/families.npy
walkers=${output_dir}/initial_walkers.npy

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
    --stall-evaluations 2000 \
    --stall-tolerance 1e-6 \
    --num-parallel "${num_parallel}" \
    --output "${seeded_results}"

run_step "${families}" "${output_dir}/select_families.log" \
    ${python} -u select_families.py "${tic}" "${seeded_results}" \
    "${families}" \
    --num-parallel "${num_parallel}"

run_step "${walkers}" "${output_dir}/initial_walkers.log" \
    ${python} -u initial_walkers.py "${tic}" "${families}" "${walkers}" \
    --num-walkers "${num_walkers}" \
    --num-parallel "${num_parallel}" \
    ${WALKER_SEED:+--seed "${WALKER_SEED}"}

plot_step "${output_dir}/init_models_0.pdf" \
    "${output_dir}/plot_init_models.log" \
    ${python} -u plot_init_models.py "${tic}" "${families}" "${walkers}" \
    "${output_dir}/init_models" \
    --num-parallel "${num_parallel}"

echo "$(date) Done: ${walkers}"
