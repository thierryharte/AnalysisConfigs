# Running the boosted VBF HH→4b analysis on lxplus

This guide covers installing PocketCoffea + AnalysisConfigs on lxplus inside the
PocketCoffea apptainer image, and running the `VBF_HH4b_boosted` workflow.

> Testing so far has been limited. If something breaks, let us know.

All commands below use two shell variables. Set them once per shell:

```bash
export EOS_HOME=/eos/user/${USER:0:1}/${USER}
export PCW=${EOS_HOME}/PocketCoffeaWorld      # working area holding both repositories
```

## 1. Clone the repositories

Fork either repository first if you plan to push changes.

```bash
mkdir -p ${PCW} && cd ${PCW}
git clone git@github.com:PocketCoffea/PocketCoffea.git
git clone git@github.com:thierryharte/AnalysisConfigs.git

# The lxplus setup lives on the `swan` branch
cd AnalysisConfigs
git checkout swan
cd ..
```

## 2. Start the apptainer image

All software dependencies come from the PocketCoffea lxplus image on CVMFS. Start a
shell in it, binding AFS, CVMFS, your EOS area and the Kerberos/HTCondor credentials
(needed to submit jobs from inside the container):

```bash
apptainer shell \
    -B /afs \
    -B /cvmfs/cms.cern.ch \
    -B /cvmfs/cms-griddata.cern.ch \
    -B /tmp \
    -B ${EOS_HOME} \
    -B /etc/sysconfig/ngbauth-submit \
    -B ${XDG_RUNTIME_DIR} \
    --env KRB5CCNAME="FILE:${XDG_RUNTIME_DIR}/krb5cc" \
    /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/pocketcoffea:lxplus-el9-stable
```

Every step below runs **inside** this container. Apptainer passes on your host
environment, so `EOS_HOME` and `PCW` are still set inside it.

## 3. Create the virtual environment (first time only)

The venv reuses the packages from the image (`--system-site-packages`), so only the
two repositories need to be installed:

```bash
cd ${PCW}
python -m venv --system-site-packages pypocket
source pypocket/bin/activate

python -m pip install -e ./PocketCoffea
python -m pip install -e ./AnalysisConfigs
```

### Make sure your PocketCoffea checkout is used

The image already ships its own PocketCoffea. Because it is a regular install, Python
finds it before the editable install, so your checkout would be ignored. To fix this,
put the checkouts on `PYTHONPATH` whenever the venv is activated:

```bash
cat >> pypocket/bin/activate << 'EOF'

# Use the EOS checkouts instead of the PocketCoffea built into the image
_PCW="$(dirname "$VIRTUAL_ENV")"
export PYTHONPATH=$_PCW/PocketCoffea:$_PCW/AnalysisConfigs${PYTHONPATH:+:$PYTHONPATH}
unset _PCW
EOF

deactivate
source pypocket/bin/activate
```

Check it. The path must point into `${PCW}/PocketCoffea/pocket_coffea/`:

```bash
python -c "import pocket_coffea; print(pocket_coffea.__file__)"
```

## 4. Daily usage

Each new session needs only two steps: start the container (step 2) and activate the venv:

```bash
source /eos/user/${USER:0:1}/${USER}/PocketCoffeaWorld/pypocket/bin/activate
```

## 5. Configure the run

The lxplus configuration is
[`configs/HH4b_common/config_files/lxplus_boosted_other_group_50GeV_lower_sideband.py`](configs/HH4b_common/config_files/lxplus_boosted_other_group_50GeV_lower_sideband.py).
Two paths in it must be adapted to your setup:

```python
# Path to the trained BDT model
"bdt_model": "/eos/user/t/tharte/PocketCoffeaWorld/resources/bdt_models/boosted/bdt_trainings_run3/25Feb5_v13_glopartv2_rawmass/trained_bdt.model",

# Directory where the per-chunk parquet files (event-level arrays) are written
"save_chunk": "/eos/user/t/tharte/PocketCoffeaWorld/outputs_datasets/parquet/spanet_1_16_9_boosted_other_group_50GeV_lower_sideband_leading_higgs_region_fixed_nJets",
```

- **`bdt_model`**: the BDT model file to evaluate.
- **`save_chunk`**: the event arrays (columns) are written as parquet files to this
  directory, separately from the `.coffea` output, so the `.coffea` files stay small
  and the large arrays go straight to storage. If you leave it empty, the columns are
  stored inside the `.coffea` output instead.

The ONNX models in `onnx_model_dict` (e.g. `bkg_morphing_dnn`) already point to
publicly readable AFS paths, so they work as they are.

## 6. Run PocketCoffea

Run from the `VBF_HH4b_boosted` config folder:

```bash
cd ${PCW}/AnalysisConfigs/configs/VBF_HH4b_boosted/

run_pocket_coffea \
    lxplus_boosted_other_group_50GeV_lower_sideband.py \
    VBF_HH4b_boosted_config.py \
    params/swan_run_options.yaml \
    ${PCW}/outputs_datasets/lxplus_boosted_other_group_50GeV_lower_sideband
```

The arguments are:

| # | Argument | Meaning |
|---|----------|---------|
| 1 | `lxplus_boosted_..._lower_sideband.py` | Name of the config in `configs/HH4b_common/config_files/` (with or without `.py`) |
| 2 | `VBF_HH4b_boosted_config.py` | PocketCoffea configuration template |
| 3 | `params/swan_run_options.yaml` | Executor run options (workers, memory, HTCondor queue, chunk size) |
| 4 | output path | Where the `.coffea` output is written |

`run_pocket_coffea` is a wrapper around `pocket-coffea run`. It detects lxplus from the
hostname and selects the `dask@lxplus` executor. It already passes `--process-separately`.

Useful extra flags:

- `--test`: runs locally without an executor, for a quick check.
- `--copy-to <dir>`: rsyncs the output to `<dir>` when the job finishes, verifies the
  copy, then deletes the original.
- Any other flags are passed on to `pocket-coffea run`.
