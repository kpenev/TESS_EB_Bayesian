#!/bin/bash

#SBATCH -N @@NODES_PER_JOB@@ # Number of nodes to request
#SBATCH --ntasks=@@NODES_PER_JOB@@ # 1 launcher task per node
#SBATCH --cpus-per-task=64       # Reserve all 64 cores on each node
#SBATCH --time=48:00:00
#SBATCH --mem=375G               # Full node memory
#SBATCH --partition=@@PARTITION@@
#SBATCH --job-name=TESS_EBs_@@JOB_GROUP@@_@@JOB_NAME@@
#SBATCH --output=/scratch/juno/kxp174430/output/TESS_EBs/render_output/slurm.out
#SBATCH --mail-user=kxp174430@utdallas.edu
#SBATCH --mail-type=all
#------------------------------------------------------

source ~/module_load_order.sh

SOURCE_DIR=~/projects/git/TESS_EB_Bayesian/scripts

# Launcher: 4 tasks per node, each task runs one python job that spawns 16 workers
export LAUNCHER_PPN=4
export LAUNCHER_WORKDIR=$SOURCE_DIR
export LAUNCHER_JOB_FILE='@@LAUNCHER_CMD@@'

export PYTHONPATH=$PYTHONPATH:~/projects/git

$LAUNCHER_DIR/paramrun

