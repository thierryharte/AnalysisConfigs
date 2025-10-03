# HH4b analysis
This folder contains the configuration files and customization code for the HH4b analysis.

## Full analysis workflow
The full analysis workflow is composed by multiple steps, which are spread in different repositories:
- https://github.com/matteomalucchi/AnalysisConfigs
- https://github.com/matteomalucchi/HH4b_SPANet
- https://github.com/matteomalucchi/SPANet
- https://github.com/matteomalucchi/ML_pytorch

### Produce SPANet input files

### Train SPANet model

#### Plot pairing efficiency and mass sculpting

### Apply SPANet model to data for background morphing

### Train DNN model for background morphing

### Apply background morphing DNN to data and produce MC signal files

#### Plot morphed 2b vs 4b

### Train DNN for signal / background classification

### Apply DNN to data and MC signal files

#### Plot DNN score



## Example commands

### Run analysis

```python
run_pocket_coffea <config_name> <config_file> <run_options> <output_dir> <--test>
```
E.g.
```python
run_pocket_coffea spanet_ptflat_rerun_matteo_transform VBF_HH4b_config.py params/t3_run_options_spanet_predict_10Gb.yaml /work/mmalucch/out_hh4b/out_transformed_DNN_score
```

### Run ggF HH4b analysis

```python
cd configs/HH4b
pocket-coffea run --cfg HH4b_parton_matching_config.py -e dask@T3_CH_PSI --custom-run-options params/t3_run_options_spanet_predict.yaml -o /work/mmalucch/out_test --executor-custom-setup onnx_executor.py
```

### Run VBF HH4b analysis

```python
cd configs/VBF_HH4b
pocket-coffea run --cfg VBF_HH4b_test_config.py -e dask@T3_CH_PSI --custom-run-options params/t3_run_options_spanet_predict.yaml -o /work/mmalucch/out_hh4b/out_vbf_jets_candidates/  --executor-custom-setup onnx_executor.py
```

### Run plot 2bvs4b

```python
sbatch -p short --account=t3 --time=00:05:00 --mem 25gb --cpus-per-task=8 --wrap="python plot_2bMorphedvs4b.py -i <input_directory> -o <output_directory>"
```


### Run plot DNN_score

```python
sbatch -p short --account=t3 --time=00:10:00 --mem 40gb --cpus-per-task=1 --wrap="python ~/AnalysisConfigs/scripts/plot_DNN_score.py -id ./  -im output_GluGlutoHHto4B_spanet_kl-1p00_kt-1p00_c2-0p00_2022_postEE.coffea -r2 -om /work/mmalucch/out_ML_pytorch/DNN_DHH_method_class_weights_e5drop75_postEE_allklambda_matteo/state_dict/model_best_epoch_19.onnx"
```