#!/bin/bash

#SBATCH -J TESS_EB_mcmc@@JOBID@@
#SBATCH -N @@NODES_PER_JOB@@  #Number of nodes on which to distribute the tasks evenly
#SBATCH -o /scratch/05392/kpenev/output/TESS_EBs/mcmc_output/@@JOBID@@_slurm.out
#SBATCH --mail-user=kxp174430@utdallas.edu
#SBATCH --mail-type=all
#SBATCH -t @@TIME_LIMIT@@
#SBATCH -p @@PARTITION@@

source ~/module_load_order.txt

export PYTHONUNBUFFERED=1
export PYTHONPATH=$PYTHONPATH:${source_dir}:~/projects/git

SOURCE_DIR=~/projects/git/TESS_EB_Bayesian/scripts

OUTPUT_ROOT=/scratch/05392/kpenev/output/TESS_EBs/render_output
LOG_DIR=${OUTPUT_ROOT}/logs
PLOT_DIR=${OUTPUT_ROOT}/plots

mkdir -p ${LOG_DIR}
mkdir -p ${PLOT_DIR}

cd ${SOURCE_DIR}
python3 -c \
    'import pylauncher pylauncher.ClassicLauncher("@@LAUNCHER_CMD@@", cores=@@NUM_PARALLEL@@, debug="host+job")'

