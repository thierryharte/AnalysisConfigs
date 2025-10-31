import argparse
import json
import logging
import os
import numpy as np
import re

import hist
from coffea.util import load
from pocket_coffea.utils.stat import (
    Datacard,
    DataProcess,
    DataProcesses,
    MCProcess,
    MCProcesses,
    Systematics,
    SystematicUncertainty,
)


def add_variation_axis(histogram):
    """Return a histogram with an extra variation axis ('nominal'), preserving values+variances."""
    import hist

    # build new axes
    axes = list(histogram.axes)
    var_axis = hist.axis.StrCategory(["nominal"], name="variation", label="Variation")

    cat_idx = next((i for i, ax in enumerate(axes) if getattr(ax, "name", None) == "cat"), None)
    if cat_idx is None:
        new_axes = axes + [var_axis]
    else:
        new_axes = axes[:cat_idx+1] + [var_axis] + axes[cat_idx+1:]

    # create new histogram
    new_hist = hist.Hist(*new_axes, storage=hist.storage.Weight())

    # --- copy content ---
    for cat in histogram.axes[cat_idx]:
        idx_old = histogram.axes[cat_idx].index(cat)
        idx_new = new_hist.axes[cat_idx].index(cat)

        sl_old = [slice(None)] * histogram.ndim
        sl_new = [slice(None)] * new_hist.ndim
        sl_old[cat_idx] = idx_old
        sl_new[cat_idx] = idx_new
        sl_new[1] = 0  # idx 1="variations", 0="nominal"

        vals = histogram.view(flow=True)[tuple(sl_old)]
        new_hist.view(flow=True)[tuple(sl_new)] = vals

    return new_hist





def duplicate_category(histo, src_label, new_label, axis=0, rescale_label=False):
    """Copy the histogram for a category from one region into a new one.

    Essentially all the required datasets/regions for the analysis are to be copied into the newly created region
    """
    cat_axis = histo.axes[axis]
    if not isinstance(cat_axis, hist.axis.StrCategory):
        raise TypeError(f"Axis {axis} is not a StrCategory axis")

    if src_label not in cat_axis:
        raise ValueError(f"Source label {src_label!r} not found in category axis")

    # build a new histoogram with the extended categories
    new_cats = list(cat_axis) + [new_label]
    new_axes = list(histo.axes)
    new_axes[axis] = hist.axis.StrCategory(new_cats, name=cat_axis.name, label=cat_axis.label)

    new_histo = hist.Hist(*new_axes, storage=hist.storage.Weight())

    # ---- copy over each old category one by one ----
    for cat in cat_axis:
        idx_old = histo.axes[axis].index(cat)
        idx_new = new_histo.axes[axis].index(cat)

        sl_old = [slice(None)] * histo.ndim
        sl_new = [slice(None)] * new_histo.ndim
        sl_old[axis] = idx_old
        sl_new[axis] = idx_new

        vals = histo.view(flow=True)[tuple(sl_old)]
        new_histo.view(flow=True)[tuple(sl_new)] = vals

    # ---- duplicate requested category ----
    idx_src = histo.axes[axis].index(src_label)
    idx_dst = new_histo.axes[axis].index(new_label)

    sl_src = [slice(None)] * histo.ndim
    sl_dst = [slice(None)] * histo.ndim
    sl_src[axis] = idx_src
    sl_dst[axis] = idx_dst

    vals = histo.view(flow=True)[tuple(sl_src)]
    if not rescale_label:
        new_histo.view(flow=True)[tuple(sl_dst)] = vals
    else:
        idx_ref = histo.axes[axis].index(rescale_label)
        sl_ref = [slice(None)] * histo.ndim
        sl_ref[axis] = idx_ref

        ref_vals = histo.view(flow=True)[tuple(sl_ref)]
        ref_sum = ref_vals["value"].sum()
        src_sum = vals["value"].sum()

        factor = ref_sum / src_sum

        vals["value"] *= factor
        vals["variance"] *= factor**2

        new_histo.view(flow=True)[tuple(sl_dst)] = vals
        new_histo = add_variation_axis(new_histo)

    return new_histo


def create_new_region(coffea_file, cat_name, run2):
    """Create a new region in the coffea file, that contains all the regions we need for the analysis."""
    suffix = "Run2" if run2 else ""
    for key in ["sumw", "sumw2"]:
        coffea_file[key][cat_name] = coffea_file[key][f"4b_signal_region{suffix}"]
    # == adding cutflow ==
    presel_samples = list(coffea_file["cutflow"]["presel"].keys())
    for key in presel_samples:
        coffea_file["cutflow"]["presel"][f"{key}_background"] = coffea_file["cutflow"]["presel"][key]
    coffea_file["cutflow"][cat_name] = coffea_file["cutflow"][f"4b_signal_region{suffix}"]
    for key in coffea_file["cutflow"][f"2b_signal_region_postW{suffix}"].keys():
        if "DATA" in key:
            coffea_file["cutflow"][cat_name][f"{key}_background"] = {}
            for subkey, subvalue in coffea_file["cutflow"][f"2b_signal_region_postW{suffix}"][key].items():
                coffea_file["cutflow"][cat_name][f"{key}_background"][f"{subkey}_background"] = subvalue
    # == variables ==
    for column in coffea_file["variables"].keys():
        sample_list = list(coffea_file["variables"][column])  # I will change the samples and I am not sure, if it would otherwise also loop over changes.
        for sample in sample_list:
            dset_list = list(coffea_file["variables"][column][sample])
            for dataset in dset_list:
                histogram = coffea_file["variables"][column][sample][dataset]
                if "DATA" not in dataset:
                    histogram_with_new = duplicate_category(histogram, f"4b_signal_region{suffix}", cat_name, axis=0, rescale_label=False)
                    histogram_with_new = duplicate_category(histogram, f"4b_signal_region{suffix}", cat_name, axis=0, rescale_label=False)
                else:
                    histogram_with_new = duplicate_category(histogram, f"4b_signal_region{suffix}", cat_name, axis=0, rescale_label=False)
                coffea_file["variables"][column][sample][dataset] = histogram_with_new
                if "DATA" in dataset:
                    histogram_with_new = duplicate_category(histogram, f"2b_signal_region_postW{suffix}", cat_name, axis=0, rescale_label=f"4b_signal_region{suffix}")
                    coffea_file["variables"][column][f"{sample}_background"][f"{dataset}_background"] = histogram_with_new
    # == datasets_metadata : by_datataking_period ==
    for period in coffea_file["datasets_metadata"]["by_datataking_period"].keys():
        samples = list(coffea_file["datasets_metadata"]["by_datataking_period"][period].keys())
        datasets = list(coffea_file["datasets_metadata"]["by_datataking_period"][period].values())
        for sample, dataset in zip(samples, datasets):
            if "DATA" in sample:
                coffea_file["datasets_metadata"]["by_datataking_period"][period][f"{sample}_background"] = {f"{d}_background" for d in dataset}
    # == datasets_metadata : by_dataset ==
    orig_dset_list = list(coffea_file["datasets_metadata"]["by_dataset"].keys())
    for dataset in orig_dset_list:
        if "DATA" in dataset:
            temp_dset = coffea_file["datasets_metadata"]["by_dataset"][dataset].copy()
            temp_dset["sample"] = f"{temp_dset['sample']}_background"
            temp_dset["primaryDataset"] = f"{temp_dset['primaryDataset']}_background"
            coffea_file["datasets_metadata"]["by_dataset"][f"{dataset}_background"] = temp_dset
    return coffea_file


logging.basicConfig(format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()


parser = argparse.ArgumentParser(description="Build datacards from pocket-coffea outputs")
parser.add_argument(
    "-i",
    "--input-data",
    type=str,
    nargs="+",
    required=True,
    help="Input directory for data with coffea files or coffea files themselves",
)
parser.add_argument(
    "-r2",
    "--run2",
    action="store_true",
    help="If running with Run2 method",
    default=False,
)
parser.add_argument(
    "-o", "--output", type=str, help="Output directory", default="./datacards"
)
args = parser.parse_args()

input_dir = os.path.dirname(args.input_data[0])

if not os.path.exists(args.output):
    os.makedirs(args.output)


# I want to use the coffea file with all outputs. Therefore, it should be merged beforehand.
coffea_list = [file for file in os.listdir(input_dir) if file.endswith(".coffea")]
if "output_all.coffea" in coffea_list:
    coffea_file = os.path.join(input_dir, "output_all.coffea")
else:
    raise NameError(f"No combined coffea file found in {coffea_list}")

# At some point, this should be the defining dictionary telling us, what sample/dataset belongs in which bin:
sig_bkg_dict = {
        "signal": {
            "ggHH_kl_1_kt_1_13p6TeV_hbbhbb": ["GluGlutoHHto4B_spanet_kl-1p00_kt-1p00_c2-0p00_2022_postEE"],
            "ggHH_kl_2p45_kt_1_13p6TeV_hbbhbb": ["GluGlutoHHto4B_spanet_kl-2p45_kt-1p00_c2-0p00_2022_postEE"],
            "ggHH_kl_5_kt_1_13p6TeV_hbbhbb": ["GluGlutoHHto4B_spanet_kl-5p00_kt-1p00_c2-0p00_2022_postEE"],
            },
        "data": {
            # This needs to have this name, and only be one category
            "data_obs": ["DATA_JetMET_JMENano_E_2022_postEE_EraE", "DATA_JetMET_JMENano_F_2022_postEE_EraF", "DATA_JetMET_JMENano_G_2022_postEE_EraG"]
            },
        "background": {
            "bbbb_background": ["DATA_JetMET_JMENano_E_2022_postEE_EraE_background", "DATA_JetMET_JMENano_F_2022_postEE_EraF_background", "DATA_JetMET_JMENano_G_2022_postEE_EraG_background"]
            }
        }

# -- Load Coffea file and config.json --
coffea_file = load(coffea_file)

region_name = "bbbb_signal_analysis_region"
coffea_file = create_new_region(coffea_file, cat_name=region_name, run2=args.run2)

# -- Histograms --
histograms_dict = {}
for key, sob_hist in coffea_file["variables"].items():
    if "sig_bkg" in key:
        histograms_dict[key] = sob_hist
        # "SoB": coffea_file["variables"]["sig_bkg_dnn_score"],
# -- Create Processes
meta_dict = coffea_file['datasets_metadata']['by_dataset']

for dataset_list in sig_bkg_dict["signal"].values():
    for dataset in dataset_list:
        if dataset not in meta_dict.keys():
            raise Exception(f"Signal dataset {dataset} not found in file")
for dataset in sig_bkg_dict["data"].values():
    for dataset in dataset_list:
        if dataset not in meta_dict.keys():
            raise Exception(f"Data dataset {dataset} not found in file")
for dataset in sig_bkg_dict["background"].values():
    for dataset in dataset_list:
        if dataset not in meta_dict.keys():
            raise Exception(f"Background dataset {dataset} not found in file")

logger.info(f"These are the MC samples: {sig_bkg_dict['signal']}")
logger.info(f"These are the Data samples: {sig_bkg_dict['data']}")
logger.info(f"These are the Data background samples: {sig_bkg_dict['background']}")

# -- Filling metadata into the respective objects --
mc_process = []
for name, datasets in sig_bkg_dict["signal"].items():
    mc_process.append(MCProcess(
            name=name,
            # All these ugly catings are to get a list with unique values.
            samples=set([meta_dict[dataset]["sample"] for dataset in datasets]),
            years=set([meta_dict[dataset]["year"] for dataset in datasets]),
            is_signal=True,
            ))
data_bg_process = []
for name, datasets in sig_bkg_dict["background"].items():
    data_bg_process.append(MCProcess(
            name=name,
            samples=set([meta_dict[dataset]["sample"] for dataset in datasets]),
            years=set([meta_dict[dataset]["year"] for dataset in datasets]),
            is_signal=False,
            ))
mc_processes = MCProcesses(mc_process + data_bg_process)

if len(sig_bkg_dict["data"].keys()) > 1:
    raise Exception("Only one single data process is allowed with fixed name 'data_obs'")
for name, datasets in sig_bkg_dict["data"].items():
    data_process = DataProcess(
            name=name,
            samples=set([meta_dict[dataset]["sample"] for dataset in datasets]),
            years=set([meta_dict[dataset]["year"] for dataset in datasets]),
            )
data_processes = DataProcesses([data_process])

# -- Systematics --
# common_systematics = [
#     "JES_Total_AK4PFPuppi", "JER_AK4PFPuppi"
# ]

# Trying to make this generic. Ideally, we want exactly one single set of variations at the moment because we are using MC only for signal. This has to be improved im some shape or form.
# Essentially, right now, all MC sets belong to "GluGluHHto4b"
for hist_cat, sob_hist in histograms_dict.items():
    # Get to the variation infos in the histograms for MC signal:
    systematics_list = []
    # Iterate through different signal types
    for sig_type, datasets in sig_bkg_dict["signal"].items():
        # Iterate through the datasets in a particular signal type (often a signle one)
        variations_updown = list(sob_hist[meta_dict[datasets[0]]["sample"]][datasets[0]].axes['variation'])
        for var in variations_updown:
            sliced = sob_hist[meta_dict[datasets[0]]["sample"]][datasets[0]][{"variation": var, "cat": region_name}]
            print(f"Variation: {var}")
            print(sliced.values())
        variations = set([re.sub(r'(Up|Down)$', '', var) for var in variations_updown])
        try:
            variations.remove("nominal")
        except:
            raise ValueError(f"Variations list {variations} does not contain 'nominal'.")
        logger.info(f"Found variations: {variations}")
        for syst in variations:
            systematics_list.append(SystematicUncertainty(name=syst, datacard_name=f"{syst}_{meta_dict[datasets[0]]['year']}", typ="shape", processes=list(sig_bkg_dict["signal"].keys()), years=[meta_dict[datasets[0]]["year"]], value=1.0))
        systematics = Systematics(systematics_list)

    _label = "run3"
    _datacard_name = f"datacard_combined_{_label}"
    _workspace_name = f"workspace_{_label}.root"

    datacard = Datacard(
            histograms=sob_hist,
            datasets_metadata=coffea_file["datasets_metadata"],
            cutflow=coffea_file["cutflow"],
            systematics=systematics,
            # This might have to change. Right now I am binding the year to the data year...
            years=set([meta_dict[dataset]["year"] for dataset in sig_bkg_dict["data"]["data_obs"]]),
            mc_processes=mc_processes,
            data_processes=data_processes,
            category=region_name,
            process_suffix="",
            )
    datacard.dump(directory=f"{args.output}/{hist_cat}", card_name=f"{region_name}_{_label}.txt", shapes_name=f"shapes_{region_name}_{_label}.root")

# _datacards = {
#         "4b_signal_region": datacard
#         }

# combine_datacards(
#         datacards=_datacards,
#         directory=args.output,
#         path=f"combine_datacards_{_label}.sh",
#         card_name=_datacard_name,
#         workspace_name=_workspace_name,
#         channel_masks=False
#         )
# create_scripts(
#         datacards=_datacards,
#         directory=args.output,
#         card_name=_datacard_name,
#         categories_masked=None,
#         suffix=_label
#         )
#
# def create_scripts(
#     datacards: dict[Datacard],
#     directory: str,
#     card_name: str = "datacard_combined.txt",
#     workspace_name : str = "workspace.root",
#     categories_masked : list[str] = None,
#     suffix: str = None,
#     ) -> None:
#     """
#     Write the bash scripts to run the fit with CMS Combine Tool."""
#
#     # Save fit scripts
#     freezeParameters = ["r"]
#
#     args = {
#         "run_MultiDimFit.sh": [[
#             "combine -M MultiDimFit",
#             f"-d {workspace_name}",
#             "-n .snapshot_all_channels",
#             f"--freezeParameters {','.join(freezeParameters)}",
#             "--cminDefaultMinimizerStrategy 2 --robustFit=1",
#             "--saveWorkspace",
#         ]],
#         "run_FitDiagnostics.sh": [[
#             "combine -M FitDiagnostics",
#             f"-d {workspace_name}",
#             "-n .snapshot_all_channels",
#             f"--freezeParameters {','.join(freezeParameters)}",
#             "--cminDefaultMinimizerStrategy 2 --robustFit=1",
#             "--saveWorkspace",
#             "--saveShapes",
#             "--saveWithUncertainties"
#         ]],
#         "run_MultiDimFit_scan1d.sh": [[
#             "combine -M MultiDimFit",
#             f"-d {workspace_name}",
#             "-n .scan1d",
#             f"--freezeParameters {','.join(freezeParameters)}",
#             "-t -1 --toysFrequentist --expectSignal=1",
#             "--cminDefaultMinimizerStrategy 2 --robustFit=1",
#             "--saveWorkspace",
#             "-v 2 --algo grid --points=30 --rMin 0 --rMax 2",
#         ]],
#         "run_MultiDimFit_toysFrequentist.sh": [[
#             "combine -M MultiDimFit",
#             f"-d {workspace_name}",
#             "-n .asimov_fit",
#             "-t -1 --toysFrequentist --expectSignal=1",
#             "--cminDefaultMinimizerStrategy 2 --robustFit=1",
#             "--saveWorkspace -v 2",
#         ]],
#         "run_FitDiagnostics_toysFrequentist.sh": [[
#             "combine -M FitDiagnostics",
#             f"-d {workspace_name}",
#             "-n .asimov_fit",
#             "-t -1 --toysFrequentist --expectSignal=1",
#             "--cminDefaultMinimizerStrategy 2 --robustFit=1",
#             "--saveWorkspace -v 2",
#         ]],
#         "run_MultiDimFit_toysFrequentist_scan1d.sh": [[
#             "combine -M MultiDimFit",
#             f"-d {workspace_name}",
#             "-n .asimov_scan1d",
#             "-t -1 --toysFrequentist --expectSignal=1",
#             "--cminDefaultMinimizerStrategy 2 --robustFit=1",
#             "--saveWorkspace",
#             "-v 2 --algo grid --points=30 --rMin 0 --rMax 2",
#         ]],
#         "run_impacts.sh": [
#             [f"combineTool.py -M Impacts -d {workspace_name}",
#             f"--freezeParameters {','.join(freezeParameters)}",
#             "-t -1 --toysFrequentist --expectSignal=1 --cminDefaultMinimizerStrategy 2 --robustFit=1",
#             "-v 2 --rMin 0 --rMax 2 -m 125 --doInitialFit"],
#             [f"combineTool.py -M Impacts -d {workspace_name}",
#             f"--freezeParameters {','.join(freezeParameters)}",
#             "-t -1 --toysFrequentist --expectSignal=1 --cminDefaultMinimizerStrategy 2 --robustFit=1",
#             "-v 2 --rMin 0 --rMax 2 -m 125 --doFits --job-mode slurm --job-dir jobs --parallel 100"]
#         ],
#         "plot_impacts.sh": [
#             [f"combineTool.py -M Impacts -d {workspace_name}",
#             f"--freezeParameters {','.join(freezeParameters)}",
#             "-t -1 --toysFrequentist --expectSignal=1 --cminDefaultMinimizerStrategy 2 --robustFit=1",
#             "-v 2 --rMin 0 --rMax 2 -m 125 -o impacts.json"],
#             ["plotImpacts.py -i impacts.json -o impacts"]
#         ],
#         # -v 2 --rMin -5 --rMax 5 --robustHesse=1 --robustHesseSave 1 --saveFitResult
#         "run_correlation_matrix.sh": [
#             ["combine -M MultiDimFit",
#             f"-d {workspace_name}",
#             "-n .covariance_matrix",
#             "-t -1 --toysFrequentist --expectSignal=1",
#             "--cminDefaultMinimizerStrategy 2 --robustFit=1",
#             "-v 2 --rMin -5 --rMax 5 --robustHesse=1 --robustHesseSave 1 --saveFitResult"]
#         ]
#     }
#     if categories_masked:
#         args.update({
#             f"run_MultiDimFit_mask_{'_'.join(categories_masked)}.sh" : [[
#                 "combine -M MultiDimFit",
#                 f"-d {workspace_name}",
#                 f"-n .snapshot_{'_'.join(categories_masked)}",
#                 f"--freezeParameters {','.join(freezeParameters)}",
#                 "--cminDefaultMinimizerStrategy 2 --robustFit=1",
#                 "--saveWorkspace",
#                 f"--setParameters {','.join([f'mask_{cat}=1' for cat in categories_masked ])}"
#             ]],
#             f"run_FitDiagnostics_mask_{'_'.join(categories_masked)}.sh" : [[
#                 "combine -M FitDiagnostics",
#                 f"-d {workspace_name}",
#                 f"-n .snapshot_{'_'.join(categories_masked)}",
#                 f"--freezeParameters {','.join(freezeParameters)}",
#                 "--cminDefaultMinimizerStrategy 2 --robustFit=1",
#                 "--saveWorkspace",
#                 "--saveShapes",
#                 "--saveWithUncertainties",
#                 f"--setParameters {','.join([f'mask_{cat}=1' for cat in categories_masked ])}"
#             ]],
#         })
#
#     scripts = {}
#     for path, lines in args.items():
#         scripts[path] = [f"{' '.join(l)}\n" for l in lines]
#
#     for script_name, commands in scripts.items():
#         script_name = script_name.replace(".sh", f"_{suffix}.sh") if suffix else script_name
#         output_file = os.path.join(directory, script_name)
#         print(f"Writing fit script to {output_file}")
#         with open(output_file, "w") as file:
#             file.write("#!/bin/bash\n")
#             file.writelines(commands)
