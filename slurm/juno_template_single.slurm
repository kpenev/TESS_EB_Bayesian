#!/bin/bash

#SBATCH --nodes=1  #Number of nodes on which to distribute the tasks evenly
#SBATCH --ntasks-per-node=64 #Total number of tasks to run at one time
#SBATCH --mem=256G #Maximum amount of memory likely to be needed
#SBATCH --time=48:00:00
#SBATCH --partition=@@PARTITION@@
#SBATCH --job-name=TESS_EBs_@@JOBID@@
#SBATCH --output=/scratch/juno/kxp174430/output/TESS_EBs/worker@@JOBID@@_slurm.out
#SBATCH --mail-user=kxp174430@utdallas.edu
#SBATCH --mail-type=all

run_system() {

    ticid=$1

    scratch=/scratch/juno/kxp174430/
    work=/work/kxp174430/

    source_dir=~/projects/git/TESS_EB_Bayesian/scripts
    log_dir=output/TESS_EBs/${ticid}
    samples=TESS_EBs/samples

    mkdir -p ${work}/${log_dir}
    mkdir -p ${scratch}/${log_dir}
    mkdir -p ${work}/${samples}
    mkdir -p ${scratch}/${samples}

    samples=${samples}/tess${ticid}_samples.h5

    mv ${scratch}/${log_dir}/* ${work}/${log_dir}/
    log_dir=${scratch}/${log_dir}/run$(date +%Y%m%d%H%M%S)
    mkdir -p ${log_dir}

    cp ${scratch}/${samples} ${work}/${samples}
    cp ${work}/${samples} ${scratch}/${samples} 

    samples=${scratch}/${samples}

    cd ${source_dir}

    while true; do
        echo "Starting $(date +%Y%m%d%H%M%S)" \
            >> ${log_dir}/run_system_${ticid}.out

        conda run -n py311 python3 mcmc_sampling.py ${ticid} \
            --verbosity debug \
            --samples-fname-pattern ${samples} \
            --std-out-err-fname \
            ${log_dir}/tess{tic_id:d}_{task}_{now!s}_{pid:d}.outerr \
            --logging-fname \
            ${log_dir}/tess{tic_id:d}_{task}_{now!s}_{pid:d}.log \
            --logging-verbosity debug \
            >> ${log_dir}/run_system_${ticid}.out \
            2>&1
    done
}

source ~/module_load_order.sh
source ~/set_environment.sh

for TICID in @@TIC_LIST@@; do 
    run_system $TICID &
done

wait
