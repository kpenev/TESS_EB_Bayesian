#!/bin/bash

#TIC=33419790
TIC=5205367
#TIC=189639080

#BURNIN=4000
BURNIN=8000
#BURNIN=6000

#PERIOD_MATCH='(2.00 < per) & (per < 2.03)'
PERIOD_MATCH='(3.98 < per) & (per < 4)'
#PERIOD_MATCH='(3.12 < per) & (per < 3.14)'

./visualize.py ${TIC} \
    --corner-plot-fname tess${TIC}_corner_eproj.pdf \
    --corner-plot-expression '$M_1 + M_2$=mtotal' \
    --corner-plot-expression '$M_1 / M_2$=mratio' \
    --corner-plot-expression 'Age [Gyr]=age_gyr' \
    --corner-plot-expression '$[M/H]$=meh' \
    --corner-plot-expression '$e \cos \omega$=ecc * cos(w)' \
    --corner-plot-expression '$e \sin \omega$=ecc * sin(w)' \
    --corner-plot-expression '$b$=primary_impact_param' \
    --burn-in ${BURNIN} \
    && \
./visualize.py ${TIC} \
    --corner-plot-fname tess${TIC}_corner_ew.pdf \
    --corner-plot-expression '$M_1 + M_2$=mtotal' \
    --corner-plot-expression '$M_1 / M_2$=mratio' \
    --corner-plot-expression 'Age [Gyr]=age_gyr' \
    --corner-plot-expression '$[M/H]$=meh' \
    --corner-plot-expression '$e$=ecc' \
    --corner-plot-expression '$b$=primary_impact_param' \
    --burn-in ${BURNIN} \
    && \
./visualize.py ${TIC} \
    --plot-lightcurve \
    tess${TIC}_lc_top1.pdf \
    '[[full, full], [folded, folded], [folded_diff, folded_diff], [zoom_ooe_folded, zoom_ooe_folded], [zoom_primary, zoom_secondary], [sed, empty]]' \
    --show-model-with-lc top1 \
    && \
./visualize.py ${TIC} \
    --plot-lightcurve \
    tess${TIC}_lc_sampled.pdf \
    '[[full, full], [folded, folded], [folded_diff, folded_diff], [zoom_ooe_folded, zoom_ooe_folded], [zoom_primary, zoom_secondary], [sed, empty]]' \
    --show-model-with-lc random100 \
    --sample-condition "${PERIOD_MATCH}"\
    && \
./visualize.py 5205367 \
    --chain-name prelim_mcmc_0 \
    --plot-lightcurve \
    tess${TIC}_starting.pdf \
    '[[full, full], [folded, sed], [zoom_primary, zoom_secondary]]' \
    --show-model-with-lc -1 \
    --sample-condition '(3.98 < per) & (per < 4)'\
    --data-on-top
