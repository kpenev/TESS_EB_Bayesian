#!/bin/bash

#SBATCH --nodes=1  #Number of nodes on which to distribute the tasks evenly
#SBATCH --ntasks-per-node=64 #Total number of tasks to run at one time
#SBATCH --mem=384000M #Maximum amount of memory likely to be needed
#SBATCH --time=48:00:00
#SBATCH --partition=normal
#SBATCH --job-name=render_prsa
#SBATCH --output=/scratch/juno/kxp174430/output/TESS_EBs/render_output/slurm.out
#SBATCH --mail-user=kxp174430@utdallas.edu
#SBATCH --mail-type=all

render() {

    table=$1
    plot_mode=$2

    source_dir=~/projects/git/TESS_EB_Bayesian/scripts
    export PYTHONPATH=$PYTHONPATH:${source_dir}:~/projects/git

    output_root=/scratch/juno/kxp174430/output/TESS_EBs/render_output/
    log_dir=${output_root}/logs/${table}/${plot_mode}
    plot_dir=${output_root}/plots/${table}/${plot_mode}

    mkdir -p ${log_dir}
    mkdir -p ${plot_dir}

    cd ${source_dir}

    conda run -n py311 python3 \
        ${source_dir}/bui/select_ticids/plots.py "${plot_mode}"\
        --num-parallel 64 \
        --table-name "${table}" \
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

render attempted_sampling best

wait
