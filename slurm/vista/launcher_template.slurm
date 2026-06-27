#!/bin/bash

#SBATCH --job-name=TESS_EBs_@@JOB_NAME@@
#SBATCH -N @@NODES_PER_JOB@@  #Number of nodes on which to distribute the tasks evenly
#SBATCH -o /scratch/05392/kpenev/output/TESS_EBs/mcmc_output/@@JOB_NAME@@_slurm.out
#SBATCH -e /scratch/05392/kpenev/output/TESS_EBs/mcmc_output/@@JOB_NAME@@_slurm.out
#SBATCH --mail-user=kxp174430@utdallas.edu
#SBATCH --mail-type=all
#SBATCH -t @@TIME_LIMIT@@
#SBATCH -p @@PARTITION@@
#SBATCH --dependency singleton

source ~/module_load_order.txt
source ~/set_environment.sh

SOURCE_DIR=~/projects/git/TESS_EB_Bayesian/scripts

cd $SOURCE_DIR
python3 -c \
    'import pylauncher; pylauncher.ClassicLauncher("@@LAUNCHER_CMD@@", cores=@@NUM_PARALLEL@@, debug="host+job")'
