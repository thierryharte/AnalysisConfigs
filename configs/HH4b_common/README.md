# HH4b analysis
This folder contains the configuration files and customization code for the HH4b analysis.

## Full analysis workflow
The full analysis workflow is composed by multiple steps, which are spread in different repositories:
- Pocket Coffea
  - https://github.com/matteomalucchi/AnalysisConfigs
  - https://github.com/matteomalucchi/PocketCoffea
- SPANet
  - https://github.com/matteomalucchi/HH4b_SPANet
  - https://github.com/matteomalucchi/SPANet
- DNN training
  - https://github.com/matteomalucchi/ML_pytorch

### Produce SPANet input files
On `tier-3`, run the following commands to produce the input files for SPANet training.
#### Run pocket-coffea to produce coffea files
> [!TIP] @Tier-3/AnalysisConfigs



```bash
cd AnalysisConfigs/configs/HH4b_common
# SPANet training with normal pT spectrum
run_pocket_coffea no_model <config> <t3_run_options> <output_dir>
# SPANet training with flat pT spectrum
run_pocket_coffea pt_vary <config> <t3_run_options> <output_dir>

# e.g.
run_pocket_coffea no_model HH4b_parton_matching_config.py params/t3_run_options.yaml ../../../sample_spanet/loose_MC_postEE_btagWP
```

#### Convert coffea files to h5 files
> [!TIP] @Tier-3/HH4b_SPANet
```bash
cd HH4b_SPANet/utils/dataset
python3 coffea_to_parquet.py -i <input_coffea_file> -o <output_dir> -c 4b_region
python3 parquet_to_h5.py -i <input_parquet_files> -o <output_dir> -f 0.8

# e.g. 
python3 /work/tharte/HH4b_SPANet/utils/dataset/coffea_to_parquet.py -i .//output_all.coffea -o . -c 4b_region
python3 /work/tharte/HH4b_SPANet/utils/dataset/parquet_to_h5.py -i ./*.parquet -o /scratch/tharte/166814/ -f 0.8
```

#### Copy h5 files to `lxplus`
> [!TIP] @Tier-3
```bash
scp -r <dir> <user>> [!TIP] @lxplus.cern.ch:<dir>

# e.g.
scp -r loose_MC_postEE_btagWP tharte> [!TIP] @lxplus.cern.ch:/eos/user/t/tharte/Analysis_data/spanet_samples
```

### Train and evaluate SPANet model
> [!TIP] @lxplus/HH4b_SPANet

Edit the option_file accordingly to the training you want to perform. 

- Event info file (input parameters to SPANet):
  - Inside the folder `HH4b_SPANet/event_files/HH4b/`
- Option file:
  - Inside the folder `HH4b_SPANet/Option_files/HH4b`
  - Lines to edit:
  - ```json
    "event_info_file": /afs/cern.ch/user/t/tharte/public/Software/HH4b_SPANet/event_files/HH4b/hh4b_5jet_btag_wp.yaml,                
    "training_file": "/eos/user/t/tharte/Analysis_data/spanet_samples/loose_MC_postEE_btagWP/output_JetGood_train.h5",  
    ```

Then run the following command on `lxplus`:

```bash
cd HH4b_SPANet/
python jobs/submit_jobs_seed.py -o <options_files/option_file.json> -c <jobs/config/config.yaml> -s <start_seed>:<end_seed> -a <"additional arguments to pass to spanet.train"> --suffix <directory_suffix> -out <output_dir>

# e.g. 
python3 ~/public/Software/HH4b_SPANet/jobs/submit_jobs_seed.py -c ~/public/Software/HH4b_SPANet/jobs/config/jet_assignment_deep_network_3d.yaml -s 100:101 -o options_files/HH4b/hh4b_5jets_ptreg_loose_300_btag_wp.json -out /eos/user/t/tharte/Analysis_data/spanet_output
```

#### Compute SPANet predictions
> [!TIP] @lxplus/HH4b_SPANet

Once the model is trained, compute the predictions on the input h5 files using the following command on `lxplus`:

```bash
#TODO

app_spanet
. ~/.bashrc
env_spanet
python -m spanet.predict ./spanet_output/out_spanet_outputs/out_hh4b_5jets_ptreg_loose_300_btag_wp/out_seed_trainings_100/version_0/ predictions/spanet_hh4b_5jets_300_ptreg_loose_s100_btag_wp.h5 -tf spanet_samples/loose_MC_postEE_btagWP/output_JetGood_test.h5 --gpu 
# In case of checking the mass sculpting with data, choose a data file as -tf argument
```

#### Plot pairing efficiency and mass sculpting 
> [!TIP] @lxplus/HH4b_SPANet

Next step is to fill in an entry in the efficiency script [efficiency_configurations](.py./utils/performance/efficiency_configurations.py):


```python
#TODO


#e.g.
f'{spanet_dir}spanet_hh4b_5jets_300_ptvary_loose_s100_btag_wp.h5': {
    'true': '5_jets_pt_true_wp_allklambda',
    'label': 'SPANet btag WP - Flattened pt [0.3,1.7]',
    'color': 'firebrick'
    },
```

And then run the efficiency script in an empty folder (A different environment is needed without the apptainer image):

```bash
#TODO

# e.g.
env_utils 
python3 ~/public/Software/HH4b_SPANet/utils/performance/efficiency_studies.py -pd . -k # To run the mass shapes using data inputs, replace -k with -d
```

#### Convert SPANet model to ONNX
> [!TIP] @lxplus/HH4b_SPANet

Converting the file to `onnx` to use it in PocketCoffea (You need again the SPANet environment from before for the prediction):

``` bash
#TODO

#e.g.
python -m spanet.export out_spanet_outputs/out_hh4b_5jets_ptvary_loose_300_btag_wp/out_seed_trainings_100/version_0/ spanet_hh4b_5jets_ptvary_loose_300_btag_5wp_s100.onnx --gpu
```

#### Copy ONNX model to `tier-3`
> [!TIP] @lxplus

Finally, copy the model to `tier-3` to use it in PocketCoffea:
```bash
scp <model.onnx> <user>> [!TIP] @<tier-3-address>:<dir>
```

### Apply SPANet model to data for background morphing
> [!TIP] @Tier-3/AnalysisConfigs


### Train DNN model for background morphing
> [!TIP] @Tier-3/ML_pytorch

### Apply background morphing DNN to data and produce MC signal files
> [!TIP] @Tier-3/AnalysisConfigs

#### Plot morphed 2b vs 4b
> [!TIP] @Tier-3/AnalysisConfigs

### Train DNN for signal / background classification
> [!TIP] @Tier-3/ML_pytorch

### Apply DNN to data and MC signal files
> [!TIP] @Tier-3/AnalysisConfigs

#### Plot DNN score
> [!TIP] @Tier-3/AnalysisConfigs



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
pocket-coffea run --cfg HH4b_parton_matching_config.py -e dask> [!TIP] @T3_CH_PSI --custom-run-options params/t3_run_options_spanet_predict.yaml -o /work/mmalucch/out_test --executor-custom-setup onnx_executor.py
```

### Run VBF HH4b analysis

```python
cd configs/VBF_HH4b
pocket-coffea run --cfg VBF_HH4b_test_config.py -e dask> [!TIP] @T3_CH_PSI --custom-run-options params/t3_run_options_spanet_predict.yaml -o /work/mmalucch/out_hh4b/out_vbf_jets_candidates/  --executor-custom-setup onnx_executor.py
```

### Run plot 2bvs4b

```python
sbatch -p short --account=t3 --time=00:05:00 --mem 25gb --cpus-per-task=8 --wrap="python plot_2bMorphedvs4b.py -i <input_directory> -o <output_directory>"
```


### Run plot DNN_score

```python
sbatch -p short --account=t3 --time=00:10:00 --mem 40gb --cpus-per-task=1 --wrap="python ~/AnalysisConfigs/scripts/plot_DNN_score.py -id ./  -im output_GluGlutoHHto4B_spanet_kl-1p00_kt-1p00_c2-0p00_2022_postEE.coffea -r2 -om /work/mmalucch/out_ML_pytorch/DNN_DHH_method_class_weights_e5drop75_postEE_allklambda_matteo/state_dict/model_best_epoch_19.onnx"
```