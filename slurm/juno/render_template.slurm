#!/bin/bash

#SBATCH --nodes=1 	    # Number of nodes on which to distribute the tasks evenly
#SBATCH --ntasks-per-node=1 # Total number of tasks to run at one time
#SBATCH --cpus-per-task=@@NUM_PARALLEL@@  # One core per plotting process
#SBATCH --mem=375G          # Full node memory
#SBATCH --time=@@TIME_LIMIT@@
#SBATCH --partition=@@PARTITION@@
# Same name as the sampling job, so singleton keeps the two from running at the
# same time: rendering never reads samples files while they are being written.
#SBATCH --job-name=TESS_EBs_@@JOB_GROUP@@_@@JOB_NAME@@
#SBATCH --output=/scratch/juno/kxp174430/output/TESS_EBs/render_output/render_@@JOB_NAME@@_%j.out
#SBATCH --mail-user=kxp174430@utdallas.edu
#SBATCH --mail-type=all
#SBATCH --dependency=singleton

# The jobs parallelize over processes; threaded BLAS/LAPACK (one thread per
# core in every process by default) would oversubscribe the CPUs.
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

render() {

    plot_mode=$1

    source_dir=~/projects/git/TESS_EB_Bayesian/scripts
    export PYTHONPATH=$PYTHONPATH:${source_dir}:~/projects/git

    output_root=/scratch/juno/kxp174430/TESS_EBs/juno/output/render_juno/
    log_dir=${output_root}/logs/@@TABLE_NAME@@/@@JOB_NAME@@/${plot_mode}
    plot_dir=${output_root}/plots/@@TABLE_NAME@@/${plot_mode}
    mkdir -p ${log_dir}
    mkdir -p ${plot_dir}

    cd ${source_dir}

    conda run --no-capture-output -n py314 python3 -u \
        ${source_dir}/bui/select_ticids/plots.py "${plot_mode}" \
        --limit-to-jobs @@JOB_GROUP@@:@@JOB_ID@@ \
        --num-parallel @@NUM_PARALLEL@@ \
        --table-name @@TABLE_NAME@@ \
        --plot-dir ${plot_dir} \
        --std-out-err-fname \
        "${log_dir}/render_{table_name}_{start}_{count}_{now!s}_{pid:d}.outerr"\
        --logging-fname \
        "${log_dir}/render_{table_name}_{start}_{count}_{now!s}_{pid:d}.log" \
        --verbosity debug \
        >\
        ${log_dir}/plots.outerr \
        2>&1
}

source ~/module_load_order.sh
source ~/set_environment.sh

for plot_mode in @@PLOT_TYPES@@; do
    render ${plot_mode}
done
