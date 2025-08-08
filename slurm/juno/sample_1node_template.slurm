#!/bin/bash

#SBATCH --nodes=1  #Number of nodes on which to distribute the tasks evenly
#SBATCH --ntasks-per-node=64 #Total number of tasks to run at one time
#SBATCH --mem=384000M #Maximum amount of memory likely to be needed
#SBATCH --time=@@TIME_LIMIT@@
#SBATCH --partition=@@PARTITION@@
#SBATCH --job-name=likelihood20250724_@@JOBID@@
#SBATCH --output=/scratch/juno/kxp174430/output/TESS_EBs/sample_output/likelihood20250724_@@JOBID@@_slurm.out
#SBATCH --mail-user=kxp174430@utdallas.edu
#SBATCH --mail-type=all

SOURCE_DIR=~/projects/git/TESS_EB_Bayesian/scripts
export PYTHONPATH=$PYTHONPATH:${SOURCE_DIR}:~/projects/git

sample() {
    tic=$1 

    output_root=/scratch/juno/kxp174430/output/TESS_EBs/sample_output/
    log_dir=${output_root}/logs/${tic}/

    mkdir -p ${log_dir}

    cd ${SOURCE_DIR}

    while true; do
        conda run -n py311 python3 -u \
            ${SOURCE_DIR}/mcmc_sampling.py $tic \
	    --samples-fname-pattern \
	    /scratch/juno/kxp174430/sampling/tess{tic_id:d}_samples.h5 \
            --num-parallel 16 \
            --std-out-err-fname \
            "${log_dir}/tess{tic_id:d}_{task}_{now!s}_{pid:d}.outerr"\
            --logging-fname \
            "${log_dir}/tess{tic_id:d}_{task}_{now!s}_{pid:d}.log"\
            >\
            ${log_dir}/mcmc_sampling_${tic}.outerr \
            2>&1
    done
}

source ~/module_load_order.sh

for TIC in @@TIC_LIST@@; do 
    sample $TIC &
done

wait
