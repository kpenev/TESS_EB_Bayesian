#!/bin/bash

SAMPLES_DIR=$1

for f in ${SAMPLES_DIR}/*.h5; do
	iter=$(h5dump -a '/mcmc/iteration' $f 2>/dev/null|grep '(0)' |awk '{print $2;}' || echo "0")
	starting=$(h5dump -a '/mcmc/starting_positions/num_positions_found' $f 2>/dev/null|grep '(0)' |awk '{print $2;}' || echo "0")
	echo $f ${starting}:${iter}
done
