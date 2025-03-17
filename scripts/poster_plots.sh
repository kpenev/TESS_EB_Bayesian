./visualize.py 33419790 --corner-plot-fname tess33419790_corner.pdf --corner-plot-expression '$M_1 + M_2$=mtotal' --corner-plot-expression '$M_1 / M_2$=mratio' --corner-plot-expression 'Age [Gyr]=age_gyr' --corner-plot-expression '$[M/H]$=meh' --corner-plot-expression '$e \cos \omega$=ecc * cos(w)' --corner-plot-expression '$e \sin \omega$=ecc * sin(w)' --corner-plot-expression '$b$=primary_impact_param' --burn-in 5000

./visualize.py 33419790 --plot-lightcurve tess33419790_lc.pdf  '[[full, full], [folded, folded], [zoom_primary, zoom_secondary]]' --show-model-with-lc top1
