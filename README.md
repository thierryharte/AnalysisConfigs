# AnalysisConfigs
Repository containing analysis configurations for PocketCoffea


## Setup

The first step is installing the main `PocketCoffea` package in your python environment.

Please have a look at the [Installation guide](https://pocketcoffea.readthedocs.io/en/latest/installation.html).

The `configs` package has been created to separate the core of the framework from all the necessary configuration files
and customization code needed the different analyses. The configuration is structured as a python package to make easier
the import of customization code into the framework configuration and also to make the sharing of analysis code easier.

Once you have a `PocketCoffea` local installation, you can install the `configs` and `utils` package with:

```python
pip install -e .
```

This will install the `configs` package in editable mode.

## Analysis examples

- Simple **Z-> mumu** invariant mass analysis [here](./configs/zmumu)

## Run ggF HH4b analysis

```python
cd configs/HH4b
pocket-coffea run --cfg HH4b_parton_matching_config.py -e dask@T3_CH_PSI --custom-run-options params/t3_run_options_spanet_predict.yaml -o /work/mmalucch/out_test --executor-custom-setup onnx_executor.py
```

## Run VBF HH4b analysis

```python
cd configs/VBF_HH4b
pocket-coffea run --cfg VBF_HH4b_test_config.py -e dask@T3_CH_PSI --custom-run-options params/t3_run_options_spanet_predict.yaml -o /work/mmalucch/out_hh4b/out_vbf_jets_candidates/  --executor-custom-setup onnx_executor.py
```