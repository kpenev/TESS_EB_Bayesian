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

# The jobs parallelize over processes; threaded BLAS/LAPACK (one thread per
# core in every process by default) would oversubscribe the CPUs.
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

SOURCE_DIR=~/projects/git/TESS_EB_Bayesian/scripts

# Memory mapped copies of the isochrone grids, shared by all processes on a
# node (see CMDInterpolator). Not removed at the end of the job: a later job on
# the node reuses them. Listed (first node only) to see what earlier jobs left.
export ISOCHRONE_BINARY_DIR=/dev/shm/$USER/isochrones
echo "Isochrone copies on $(hostname) at job start:"
ls -la --time-style=full-iso $ISOCHRONE_BINARY_DIR 2>&1

cd $SOURCE_DIR
python3 -c \
    'import pylauncher; pylauncher.ClassicLauncher("@@LAUNCHER_CMD@@", cores=@@NUM_PARALLEL@@, debug="host+job")'
