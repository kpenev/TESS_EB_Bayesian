#!/bin/bash

PYTHONPATH=${PYTHONPATH}:~/projects/git/TESS_EB_Bayesian/scripts
systemd-run --scope -p MemoryMax=24G --user \
    python3 select_ticids/plots.py best \
    --num-parallel 8 \
    --table-name "attempted_sampling" \
    --plot-dir "/mnt/md2/TESS_EBs/attempted_sampling/best" \
    > prsa_plots.out 2>&1 &
