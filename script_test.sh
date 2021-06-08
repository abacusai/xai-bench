#!/bin/bash

cat script_test.sh

output_path="results/regression/exp-gaussian-dim=20/"

mkdir -p "${output_path}/csv"

for dataset in "gaussianLinear" "gaussianNonLinearAdditive" "gaussianPiecewiseConstant";
do
    for rho in 0.0 0.25 0.5 0.75 0.99;
    do
        python main_driver.py --mode regression --seed 3 --experiment --experiment-json configs/experiment_config_dim=20.json --rho $rho --dataset $dataset --results-dir $output_path 
    done
done