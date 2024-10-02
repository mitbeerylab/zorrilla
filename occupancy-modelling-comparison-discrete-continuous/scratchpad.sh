#!/bin/bash

Rscript run_comparisons.R 1 5 ./output/ $(date '+%Y-%m-%d-%H-%M') Nelder-Mead

echo 'knitr::purl(input = "Ain_lynx_occupancy.Rmd", output = "Ain_lynx_occupancy.R")' | R --no-save