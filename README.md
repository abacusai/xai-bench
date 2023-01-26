# trustyai_xai_bench
TrustyAI's (heavily modified) fork of [abacusai's xai-bench repo](https://github.com/abacusai/xai-bench).

# Available Benchmark Configs
* `0` : A very quick benchmark to test if explainers and xai-bench is working. Runs in a few seconds.
* `1` : A moderate benchmark to evaluate performance of TrustyAI LIME + SHAP versus official versions over a variety of models and datasets. Runs in many minutes.
* `2` : A very thorough benchmark to evaluate performance of TrustyAI LIME + SHAP versus official versions over a huge variety of models and datasets. Runs in many hours.
* `lime` : Config 1, but just benchmarking LIME.
* `shap` : Config 1, but just benchmarking SHAP.

# Command line arguments
`python3 main.py --config $CONFIG --label $LABEL`
or
`python3 main.py --c $CONFIG --l $LABEL`

* `--config` : set the config to benchmark, one of `0`, `1`, `2`, `lime`, or `shap`
* `--label` : an optional suffix to add to saved files and produced plots, e.g., the branch name being tested

# Usage (in a TrustyAI dev context)
1) Build the version of TrustyAI you'd like to benchmark
2) Run `python3 main.py --config $CONFIG --label $LABEL` to run a benchmark config

