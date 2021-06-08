#!/bin/bash

output_path="results/regression/exp-mixture-3/"

mkdir -p "${output_path}/csv"

for dataset in "mixtureLinear" "mixtureNonLinearAdditive" "mixturePiecewiseConstant";
do
    for rho in 0.0 0.25 0.5 0.75 0.99;
    do
        python main_driver.py --mode regression --seed 7 --experiment --experiment-json configs/experiment_mixture_config.json --rho $rho --dataset $dataset --results-dir $output_path &
    done
    wait
done
wait
