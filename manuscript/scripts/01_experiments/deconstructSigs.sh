#!/bin/bash

BSUB_PARAMS="bsub -W 1:00 -R rusage[mem=4]"
SIGS_FILE="data/COSMIC_v3.2_SBS_GRCh37.txt"
COUNTS_FILE='data/pgh_counts.csv'
OUT_FILE="results/figure_data/phg_all_deconstructsigs_activities.csv"

#head -1 $SIGS_FILE > $OUT_FILE

(tail -n +2 $COUNTS_FILE) | while read p; do
  uid=$(echo $p | cut -f1 -d",")
  cmd="$BSUB_PARAMS Rscript scripts/estimate_deconstructSigs.R $SIGS_FILE $COUNTS_FILE $uid $OUT_FILE"
  bsub $cmd
done 
