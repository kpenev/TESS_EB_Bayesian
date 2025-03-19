#!/bin/bash

TIC=$1
BURNIN=$2

if [ "$BURNIN" == "" ]; then
    BURNIN=5000
fi

./visualize.py ${TIC} \
    --corner-plot-fname tess${TIC}_corner_eproj.pdf \
    --corner-plot-expression '$M_1 + M_2$=mtotal' \
    --corner-plot-expression '$M_1 / M_2$=mratio' \
    --corner-plot-expression 'Age [Gyr]=age_gyr' \
    --corner-plot-expression '$[M/H]$=meh' \
    --corner-plot-expression '$P_{orb}$ [d]=per' \
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
    --corner-plot-expression '$P_{orb}$ [d]=per' \
    --corner-plot-expression '$e$=ecc' \
    --corner-plot-expression '$\omega$=w' \
    --corner-plot-expression '$b$=primary_impact_param' \
    --burn-in ${BURNIN} \
    && \
./visualize.py ${TIC} \
    --plot-lightcurve \
    tess${TIC}_lc_best.pdf \
    '[[full, full], [folded, folded], [zoom_primary, zoom_secondary]]' \
    --show-model-with-lc top1 \
    && \
./visualize.py ${TIC} \
    --plot-lightcurve \
    tess${TIC}_lc_sampled.pdf \
    '[[full, full], [folded, folded], [zoom_primary, zoom_secondary]]' \
    --show-model-with-lc random100 \
    && \
./visualize.py ${TIC} \
    --chain-name prelim_mcmc \
    --plot-lightcurve \
    tess${TIC}_starting.pdf \
    '[[full, full], [folded, folded], [zoom_primary, zoom_secondary]]' \
    --show-model-with-lc -1 \
    --sample-condition '(3.98 < per) & (per < 4)'
