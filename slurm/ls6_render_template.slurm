#!/bin/bash

#SBATCH -J render_prsa_@@JOBIND@@
#SBATCH -N 1  #Number of nodes on which to distribute the tasks evenly
#SBATCH -n 1 #Total number of tasks to run at one time
#SBATCH -o /scratch/05392/kpenev/output/TESS_EBs/render_output/@@JOBIND@@_slurm.out
#SBATCH --mail-user=kxp174430@utdallas.edu
#SBATCH --mail-type=all
#SBATCH -t 24:00:00
#SBATCH -p normal

render() {
    source_dir=~/projects/git/TESS_EB_Bayesian/scripts
    export PYTHONPATH=$PYTHONPATH:${source_dir}:~/projects/git

    output_root=/scratch/05392/kpenev/output/TESS_EBs/render_output
    log_dir=${output_root}/logs
    plot_dir=${output_root}/plots

    start=$1
    count=$2

    mkdir -p ${log_dir}
    mkdir -p ${plot_dir}

    cd ${source_dir}

    python3 \
        ${source_dir}/bui/select_ticids/plots.py \
        --start ${start} \
        --count ${count} \
        --num-parallel 128 \
        --plot-dir ${plot_dir} \
        --std-out-err-fname \
        "${log_dir}/render_{table_name}_{start}_{count}_{now!s}_{pid:d}.outerr"\
        --logging-fname \
        "${log_dir}/render_{table_name}_{start}_{count}_{now!s}_{pid:d}.log" \
        --verbosity debug \
        >\
        ${log_dir}/plots_${start}_${count}.outerr \
        2>&1
}

source /home1/08402/vortebo/module_load_order.txt


export PYTHONUNBUFFERED=1

render $((@@JOBIND@@ * 256)) 256

wait
