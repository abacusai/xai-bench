#!/bin/bash

output_path="results/classification/exp2/"

mkdir -p "${output_path}/csv"

for dataset in "gaussianLinear" "gaussianNonLinearAdditive" "gaussianPiecewiseConstant";
do
    for rho in 0.0 0.25 0.5 0.75 0.99;
    do
        python main_driver.py --mode classification --seed 5 --experiment --experiment-json configs/experiment_config.json \
            --rho $rho --dataset $dataset --results-dir $output_path
    done
done
