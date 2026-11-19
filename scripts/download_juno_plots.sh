#!/bin/bash

for plt in best convergence starting; do
    rsync \
        kxp174430@juno:/scratch/juno/kxp174430/TESS_EBs/juno/output/render_juno/plots/sample_prsa/${plt}/*.png \
        ~/projects/git/TESS_EB_Bayesian/results/render/sample_prsa/${plt}/ \
        -avz
done

# The parameter summaries saved with the convergence plots.
rsync \
    kxp174430@juno:/scratch/juno/kxp174430/TESS_EBs/juno/output/render_juno/plots/sample_prsa/convergence/*_parameters.json \
    ~/projects/git/TESS_EB_Bayesian/results/render/sample_prsa/convergence/ \
    -avz
