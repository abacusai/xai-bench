#!/bin/bash

cat script.sh

output_path="results/regression/wed/exp-gaussian-new-3/"

mkdir -p "${output_path}/csv"

for dataset in "gaussianLinear" "gaussianNonLinearAdditive" "gaussianPiecewiseConstant";
do
    echo "Done with ${dataset}"
    for rho in 0.0 0.25 0.5 0.75 0.99;
    do
        python main_driver.py --mode regression --seed 7 --experiment --experiment-json configs/experiment_config.json --rho $rho --dataset $dataset --results-dir $output_path &
    done
    wait
done
wait