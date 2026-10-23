import argparse
import copy
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
    help="Input directory for data with coffea files or coffea files themselves (needs a '/' in the end, if not a single file)",
)
parser.add_argument(
    "-r2",
    "--run2",
    action="store_true",
    help="If running with Run2 method",
    default=False,
)
parser.add_argument(
    "-sn",
    "--separate-normalisation",
    action="store_true",
    help="If true, normalise signal region and control region separately. Otherwise, normalise everything to control region.",
    default=False,
)
parser.add_argument(
    "-o", "--output", type=str, help="Output directory", default="./datacards"
)
parser.add_argument(
    "-b",
    "--boosted",
    action="store_true",
    help="If a boosted analysis is being performed",
    default=False,
)
parser.add_argument(
    "-sb",
    "--single-bin",
    action="store_true",
    help="rebins fit histogram to only have a single bin",
    default=False,
)
parser.add_argument(
    "-nb",
    "--nbins",
    type=int,
    help="rebins fit histogram to this number of bins (must divide the original number of bins). Default: keep original binning",
    default=None,
)
parser.add_argument(
    "-mu",
    "--max-bkg-unc",
    type=float,
    help="Minimum-statistics rebinning: merge adjacent bins (per category, left to right) until the relative statistical "
         "uncertainty of the total background in every bin is at most this value (e.g. 0.3). Applied after --nbins.",
    default=None,
)
parser.add_argument(
    "-my",
    "--min-bkg-yield",
    type=float,
    help="Minimum-statistics rebinning: additionally require at least this total background yield per bin (e.g. 1.0)",
    default=None,
)

args = parser.parse_args()
if args.single_bin:
    args.nbins = 1


input_dir = os.path.dirname(f"{args.input_data[0]}/")

if not os.path.exists(args.output):
    os.makedirs(args.output)


# Define signal, background, and data datasets
# == Layout: {"signal"/"data"/"background": {process_name: [dataset names as in the coffea file]}}
# Data datasets listed under "background" are taken from bkg_region_name and reweighted (data-driven background);
# they are listed with their plain names, the "_background" suffix is added internally.
if not args.boosted:
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
                "bbbb_background": ["DATA_JetMET_JMENano_E_2022_postEE_EraE", "DATA_JetMET_JMENano_F_2022_postEE_EraF", "DATA_JetMET_JMENano_G_2022_postEE_EraG"]
                }
            }
else:
    sig_bkg_dict = {
            "signal": {
                "qqHH_CV_1_C2V_1_kl_1_13p6TeV_hbbhbb":              ["VBFHHto4B_Par-CV-1-C2V-1-C3-1_TuneCP5_13p6TeV_madgraph-pythia8_2024"],
                "qqHH_CV_1_C2V_0_kl_1_13p6TeV_hbbhbb":              ["VBFHHto4B_Par-CV-1-C2V-0-C3-1_TuneCP5_13p6TeV_madgraph-pythia8_2024"],
                "qqHH_CV_m0p758_C2V_1p44_kl_m19p3_13p6TeV_hbbhbb":  ["VBFHHto4B_Par-CV-m0p758-C2V-1p44-C3-m19p3_TuneCP5_13p6TeV_madgraph-pythia8_2024"],
                "qqHH_CV_1p74_C2V_1p37_kl_14p4_13p6TeV_hbbhbb":     ["VBFHHto4B_Par-CV-1p74-C2V-1p37-C3-14p4_TuneCP5_13p6TeV_madgraph-pythia8_2024"],
                "qqHH_CV_m0p962_C2V_0p959_kl_m1p43_13p6TeV_hbbhbb": ["VBFHHto4B_Par-CV-m0p962-C2V-0p959-C3-m1p43_TuneCP5_13p6TeV_madgraph-pythia8_2024"],
                "qqHH_CV_m1p60_C2V_2p72_kl_m1p36_13p6TeV_hbbhbb":   ["VBFHHto4B_Par-CV-m1p60-C2V-2p72-C3-m1p36_TuneCP5_13p6TeV_madgraph-pythia8_2024"],
            },
            "data": {
                # This needs toTTto2L2Nu_2024 have this name, and only be one category
                "data_obs": ["DATA_JetMET0_HH4bBoosted_2024_EraC", "DATA_JetMET0_HH4bBoosted_2024_EraD", "DATA_JetMET0_HH4bBoosted_2024_EraE", "DATA_JetMET0_HH4bBoosted_2024_EraF", "DATA_JetMET0_HH4bBoosted_2024_EraG", "DATA_JetMET0_HH4bBoosted_2024_EraH", "DATA_JetMET0_HH4bBoosted_2024_EraI", "DATA_JetMET1_HH4bBoosted_2024_EraC", "DATA_JetMET1_HH4bBoosted_2024_EraD", "DATA_JetMET1_HH4bBoosted_2024_EraE", "DATA_JetMET1_HH4bBoosted_2024_EraF", "DATA_JetMET1_HH4bBoosted_2024_EraG", "DATA_JetMET1_HH4bBoosted_2024_EraH", "DATA_JetMET1_HH4bBoosted_2024_EraI"]
            },
            "background": {
                "QCD_background": ["DATA_JetMET0_HH4bBoosted_2024_EraC", "DATA_JetMET0_HH4bBoosted_2024_EraD", "DATA_JetMET0_HH4bBoosted_2024_EraE", "DATA_JetMET0_HH4bBoosted_2024_EraF", "DATA_JetMET0_HH4bBoosted_2024_EraG", "DATA_JetMET0_HH4bBoosted_2024_EraH", "DATA_JetMET0_HH4bBoosted_2024_EraI", "DATA_JetMET1_HH4bBoosted_2024_EraC", "DATA_JetMET1_HH4bBoosted_2024_EraD", "DATA_JetMET1_HH4bBoosted_2024_EraE", "DATA_JetMET1_HH4bBoosted_2024_EraF", "DATA_JetMET1_HH4bBoosted_2024_EraG", "DATA_JetMET1_HH4bBoosted_2024_EraH", "DATA_JetMET1_HH4bBoosted_2024_EraI"],
                "TTto2L2Nu": ["TTto2L2Nu_2024"],
                "TTto4Q": ["TTto4Q_2024"],
                "TTtoLNu2Q": ["TTtoLNu2Q_2024"],
                }
            }

# Define the new region name
if args.boosted:
    region_name_list = ["catvbf", "cat1", "cat2", "cat3"]
    sig_region_name = ["boosted_signal_region_D_catvbf", "boosted_signal_region_D_cat1", "boosted_signal_region_D_cat2", "boosted_signal_region_D_cat3"]
    bkg_region_name = ["boosted_control_region_B_catvbf_postW", "boosted_control_region_B_cat1_postW", "boosted_control_region_B_cat2_postW", "boosted_control_region_B_cat3_postW"]
    region_definitions = {}
    for name, sig, bkg in zip(region_name_list, sig_region_name, bkg_region_name):
        region_definitions[f"HH4b_merged_{name}"] = {
                "fit_variable": "SubLeadingHiggsMass",
                "sig_region_name": sig,
                "bkg_region_name": bkg,
                "sig_control_region_name": "boosted_signal_sideband_region_C",
                "bkg_control_region_name": "boosted_control_sideband_region_A_postW",
                }

else:
    region_definitions = {
            "ggHH": {
                "fit_variable": "sig_bkg_dnn_score",
                "sig_region_name": "4b_signal_region",
                "bkg_region_name": "2b_signal_region_postW",
                "sig_control_region_name": "4b_control_region",
                "bkg_control_region_name": "2b_control_region_postW",
                }
            }


def get_year(pocketcoffea_year, year_type=0):
    """Get year string for datacard from PocketCoffea version.

    :param year_type: There are three different ways, how year is used.
                      - separated by pre/post eras (year_type=0)
                      - actual year (year_type=1)
                      - separated by parking/no-parking (year_type=2)
    """
    # Different categories for how to separated years/datataking periods
    dict_year_type = [
            {"2022_preEE": "2022", "2022_postEE": "2022EE", "2023_preBPix": "2023", "2023_postBPix": "2023BPIX", "2024": "2024"},
            {"2022_preEE": "2022", "2022_postEE": "2022", "2023_preBPix": "2023", "2023_postBPix": "2023", "2024": "2024"},
            {"2022_preEE": "2022", "2022_postEE": "2022", "2023_preBPix": "2023", "2023_postBPix": "2023ParkingHH", "2024": "2024"}  # <-- Probably wrong. But used for background datadriven
            ]
    try:
        return dict_year_type[year_type][pocketcoffea_year]
    except:
        raise ValueError(f"Wrong values for either input year: {pocketcoffea_year} or year_type: {year_type}")


def get_uncertainty_name(pocketcoffea_name, year):
    """Get conventional datacard name for systematic uncertainties"""
    # Try to remove unnecessary things in front (e.g. "AK4PFPuppi", "AK8PFPuppi")
    if "Puppi" in pocketcoffea_name or "AK" in pocketcoffea_name:
        pocketcoffea_name = pocketcoffea_name.split("_", 1)[-1]
    unc_map = {
        "JES_Total": f"CMS_scale_j_{get_year(year, year_type=0)}",
        "JER": f"CMS_res_j_{get_year(year, year_type=0)}",
        "pileup": f"CMS_pileup_{get_year(year, year_type=0)}",
        "luminosity": f"lumi_13TeV_{get_year(year, year_type=1)}",
        }
    if pocketcoffea_name in unc_map.keys():
        return unc_map[pocketcoffea_name]
    # Variations connected to b-tag SF
    elif "btag" in pocketcoffea_name:
        variation = pocketcoffea_name.split("_")[-1]
        if "heavy" in pocketcoffea_name:
            return f"CMS_btag_fixedWP_bc_{variation}_{year}"
        elif "light" in pocketcoffea_name:
            return f"CMS_btag_fixedWP_light_{variation}_{year}"
        else:
            raise ValueError(f"Unclear flavour type for b-tag correction {pocketcoffea_name}. Should contain 'light' or 'heavy' in name")
    else:
        raise ValueError(f"Did not found name mapping for {pocketcoffea_name}")

def add_variation_axis(histogram):
    """Return a histogram with an extra variation axis ('nominal'), preserving values+variances."""
    # build new axes
    axes = list(histogram.axes)
    var_axis = hist.axis.StrCategory(["nominal"], name="variation", label="Variation")

    cat_idx = next((i for i, ax in enumerate(axes) if getattr(ax, "name", None) == "cat"), None)
    if cat_idx is None:
        raise ValueError(f"Histogram has no 'cat' axis: {histogram.axes}")
    var_idx = cat_idx + 1
    new_axes = axes[:var_idx] + [var_axis] + axes[var_idx:]

    # create new histogram
    try:
        new_hist = hist.Hist(*new_axes, storage=hist.storage.Weight())
    except:
        breakpoint()

    # --- copy content (the variation axis only has 'nominal' at index 0) ---
    sl_new = [slice(None)] * new_hist.ndim
    sl_new[var_idx] = 0
    new_hist.view(flow=True)[tuple(sl_new)] = histogram.view(flow=True)

    return new_hist


def fold_flow(histogram, axis=-1):
    """Return a copy of the histogram with under-/overflow of `axis` added to the first/last bin.

    The Datacard only reads the in-range bins, so without folding events outside the axis range
    (e.g. a score of exactly 1.0 for Regular(0, 1)) would silently be dropped.
    """
    new_hist = histogram.copy()
    traits = new_hist.axes[axis].traits
    view = new_hist.view(flow=True)
    view_ax = np.moveaxis(view, axis, -1)  # still a view into new_hist
    for field in ("value", "variance"):
        if traits.underflow:
            view_ax[field][..., 1] += view_ax[field][..., 0]
            view_ax[field][..., 0] = 0
        if traits.overflow:
            view_ax[field][..., -2] += view_ax[field][..., -1]
            view_ax[field][..., -1] = 0
    return new_hist


def compute_min_stat_edges(histograms, category, bkg_datasets, meta_dict, max_rel_unc=None, min_yield=None):
    """Bin edges for which every bin of the total background fulfills the minimum statistics requirement.

    Adjacent bins of the variable axis (last axis) are merged from left to right until the summed nominal
    background has a yield > 0, >= `min_yield` and a relative stat. uncertainty <= `max_rel_unc`.
    A remaining tail not fulfilling the requirement is merged into the last bin.

    :param histograms: histograms of a single variable, layout {sample: {dataset: hist}}
    :param bkg_datasets: background datasets (as in the histograms) entering the total background
    """
    values, variances, edges = 0.0, 0.0, None
    for dataset in bkg_datasets:
        sample = meta_dict[dataset]["sample"]
        if dataset not in histograms.get(sample, {}):
            continue
        histogram = histograms[sample][dataset]
        selection = {"cat": category}
        if "variation" in histogram.axes.name:
            selection["variation"] = "nominal"
        # Flow bins were already folded in create_new_region
        view = histogram[selection].view(flow=False)
        values = values + view["value"]
        variances = variances + view["variance"]
        edges = histogram.axes[-1].edges
    if edges is None:
        raise ValueError(f"No background histograms found for category {category}")

    def passes(val, var):
        if val <= 0:
            return False
        if min_yield is not None and val < min_yield:
            return False
        if max_rel_unc is not None and np.sqrt(var) / val > max_rel_unc:
            return False
        return True

    new_edges = [edges[0]]
    acc_val, acc_var = 0.0, 0.0
    for i in range(len(values)):
        acc_val += values[i]
        acc_var += variances[i]
        if passes(acc_val, acc_var):
            new_edges.append(edges[i + 1])
            acc_val, acc_var = 0.0, 0.0
    if not np.isclose(new_edges[-1], edges[-1]):
        if len(new_edges) > 1:
            new_edges[-1] = edges[-1]
        else:
            new_edges.append(edges[-1])
    new_edges = np.array(new_edges)

    # Report the resulting background statistics per bin
    idx = np.searchsorted(edges, new_edges[:-1] + 0.5 * np.min(np.diff(edges))) - 1
    bin_val = np.add.reduceat(np.asarray(values), idx)
    bin_unc = np.sqrt(np.add.reduceat(np.asarray(variances), idx))
    logger.info(f"Min-stat rebinning for {category}: {len(edges) - 1} -> {len(new_edges) - 1} bins, edges {new_edges.tolist()}")
    logger.info(f"  bkg yield per bin: {np.round(bin_val, 2).tolist()}")
    logger.info(f"  bkg rel. unc. per bin: {np.round(bin_unc / np.where(bin_val > 0, bin_val, np.nan), 3).tolist()}")
    return new_edges


def compute_norm_factor(histograms, num_label, denom_label):
    """Global normalisation factor sum(num_label) / sum(denom_label), summed over all data datasets.

    :param histograms: histograms of a single variable, layout {sample: {dataset: hist}}
    """
    sum_num = 0.0
    sum_denom = 0.0
    for datasets in histograms.values():
        for dataset, histogram in datasets.items():
            if "DATA" not in dataset or dataset.endswith("_background"):
                continue
            sum_num += histogram[{"cat": num_label}].sum(flow=True).value
            sum_denom += histogram[{"cat": denom_label}].sum(flow=True).value
    if sum_denom == 0:
        raise ValueError(f"No data events in normalisation region {denom_label!r}")
    factor = sum_num / sum_denom
    logger.info(f"Normalisation factor {num_label} / {denom_label} = {sum_num} / {sum_denom} = {factor:.5f}")
    return factor


def duplicate_category(histo, src_label, new_label, axis=0, scale=None):
    """Copy the histogram for a category from one region into a new one.

    Essentially all the required datasets/regions for the analysis are to be copied into the newly created region.
    If `scale` is given, the copied category is scaled by it and a 'nominal' variation axis is added
    (needed, because the scaled data is treated as an MC process in the datacard).
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

    # Copy, so that scaling does not modify the input histogram in place
    vals = histo.view(flow=True)[tuple(sl_src)].copy()
    if scale is None:
        new_histo.view(flow=True)[tuple(sl_dst)] = vals
    else:
        vals["value"] *= scale
        vals["variance"] *= scale**2

        new_histo.view(flow=True)[tuple(sl_dst)] = vals
        if "variation" not in new_histo.axes.name:
            new_histo = add_variation_axis(new_histo)

    return fold_flow(new_histo)


def create_new_region(coffea_file, cat_name, sig_region_name, bkg_region_name, fit_variable, sig_control_region_name, bkg_control_region_name):
    """Create a new region in the coffea file, that contains all the regions we need for the analysis."""

    for key in ["sumw", "sumw2"]:
        # Only works, if all MC are in the signal region (will hopefully stay this way...)
        coffea_file[key][cat_name] = copy.deepcopy(coffea_file[key][sig_region_name])
    # == adding cutflow ==
    presel_samples = [k for k in coffea_file["cutflow"]["presel"].keys() if not k.endswith("_background")]
    for key in presel_samples:
        coffea_file["cutflow"]["presel"][f"{key}_background"] = copy.deepcopy(coffea_file["cutflow"]["presel"][key])
    # deepcopy, as otherwise the "_background" entries below would also be added to the signal region cutflow
    coffea_file["cutflow"][cat_name] = copy.deepcopy(coffea_file["cutflow"][sig_region_name])
    for key in coffea_file["cutflow"][bkg_region_name].keys():
        if "DATA" in key:
            coffea_file["cutflow"][cat_name][f"{key}_background"] = {}
            for subkey, subvalue in coffea_file["cutflow"][bkg_region_name][key].items():
                coffea_file["cutflow"][cat_name][f"{key}_background"][f"{subkey}_background"] = subvalue
    # == variables ==
    # One global factor (from the fit variable, flow included) for all eras and variables
    if args.separate_normalisation:
        norm_factor = compute_norm_factor(coffea_file["variables"][fit_variable], sig_region_name, bkg_region_name)
    else:
        norm_factor = compute_norm_factor(coffea_file["variables"][fit_variable], sig_control_region_name, bkg_control_region_name)
    for column in coffea_file["variables"].keys():
        sample_list = [s for s in coffea_file["variables"][column] if not s.endswith("_background")]
        for sample in sample_list:
            for dataset in list(coffea_file["variables"][column][sample]):
                histogram = coffea_file["variables"][column][sample][dataset]
                # Copy the histgam of Data and MC from the signal region.
                coffea_file["variables"][column][sample][dataset] = duplicate_category(histogram, sig_region_name, cat_name, axis=0)  # If Data, also copy from the background region
                if "DATA" in dataset:
                    bkg_sample, bkg_dataset = f"{sample}_background", f"{dataset}_background"
                    # Extend the background histogram from earlier passes if it exists, else start from the data histogram
                    base = coffea_file["variables"][column].get(bkg_sample, {}).get(bkg_dataset, histogram)
                    coffea_file["variables"][column].setdefault(bkg_sample, {})[bkg_dataset] = duplicate_category(
                        base, bkg_region_name, cat_name, axis=0, scale=norm_factor)
    # == datasets_metadata : by_datataking_period ==
    for period, entries in coffea_file["datasets_metadata"]["by_datataking_period"].items():
        for sample, dataset in list(entries.items()):
            if "DATA" in sample and not sample.endswith("_background"):
                entries[f"{sample}_background"] = {f"{d}_background" for d in dataset}
    # == datasets_metadata : by_dataset ==
    orig_dset_list = list(coffea_file["datasets_metadata"]["by_dataset"].keys())
    for dataset in orig_dset_list:
        if "DATA" in dataset and not dataset.endswith("_background"):
            temp_dset = coffea_file["datasets_metadata"]["by_dataset"][dataset].copy()
            temp_dset["sample"] = f"{temp_dset['sample']}_background"
            temp_dset["primaryDataset"] = f"{temp_dset['primaryDataset']}_background"
            coffea_file["datasets_metadata"]["by_dataset"][f"{dataset}_background"] = temp_dset
    return coffea_file




if __name__ == "__main__":
    # I want to use the coffea file with all outputs. Therefore, it should be merged beforehand.
    if "output_all.coffea" in input_dir:
        coffea_file = input_dir
    else:
        coffea_list = [file for file in os.listdir(input_dir) if file.endswith(".coffea")]
        if "output_all.coffea" in coffea_list:
            coffea_file = os.path.join(input_dir, "output_all.coffea")
        else:
            raise NameError(f"No combined coffea file found in {coffea_list}")

    # -- Load Coffea file and config.json --
    coffea_file = load(coffea_file)

    # Data-driven backgrounds are listed with the plain data dataset names. Their reweighted copies
    # (created in create_new_region) carry the "_background" suffix, so point to those instead.
    for name, datasets in sig_bkg_dict["background"].items():
        sig_bkg_dict["background"][name] = [f"{d}_background" if "DATA" in d else d for d in datasets]

    for region_name, value in region_definitions.items():
        coffea_file = create_new_region(coffea_file,
                                        cat_name=region_name,
                                        **value
                                        # sig_region_name=value["sig_region_name"],
                                        # bkg_region_name=value["bkg_region_name"],
                                        # fit_variable=value["fit_variable"],
                                        # sig_control_region_name=value["sig_control_region_name"],
                                        # bkg_control_region_name=value["sig_control_region_name"],
                                        )

        # -- Histograms --
        histograms_dict = {}
        for key, sob_hist in coffea_file["variables"].items():
            if value["fit_variable"] in key:
                if args.nbins is None:
                    histograms_dict[key] = sob_hist
                else:
                    # Rebin the variable axis (last axis) of every histogram into args.nbins bins.
                    # Flow bins were already folded in create_new_region, so no events are lost.
                    def rebin_factor(h):
                        size = h.axes[-1].size
                        if size % args.nbins != 0:
                            raise ValueError(f"Cannot rebin {size} bins of {key} into {args.nbins} bins (not a divisor)")
                        return size // args.nbins
                    histograms_dict[key] = {
                        sample: {
                            dataset: h[{h.axes[-1].name: hist.rebin(rebin_factor(h))}]
                            for dataset, h in datasets.items()
                        }
                        for sample, datasets in sob_hist.items()
                    }
            # TODO: Make sure, that the correct signal histograms are chosen
            # if "sig_bkg" in key:
            #     histograms_dict[key] = sob_hist
            #     # "SoB": coffea_file["variables"]["sig_bkg_dnn_score"],
        # -- Create Processes
        meta_dict = coffea_file['datasets_metadata']['by_dataset']

        for datasets in sig_bkg_dict["signal"].values():
            for dataset in datasets:
                if dataset not in meta_dict.keys():
                    raise Exception(f"Signal dataset {dataset} not found in file")
        for datasets in sig_bkg_dict["data"].values():
            for dataset in datasets:
                if dataset not in meta_dict.keys():
                    raise Exception(f"Data dataset {dataset} not found in file")
        for datasets in sig_bkg_dict["background"].values():
            for dataset in datasets:
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
                    samples={meta_dict[dataset]["sample"] for dataset in datasets},
                    years={meta_dict[dataset]["year"] for dataset in datasets},
                    is_signal=True,
                    ))
        bg_process = []
        for name, datasets in sig_bkg_dict["background"].items():
            bg_process.append(MCProcess(
                    name=name,
                    samples={meta_dict[dataset]["sample"] for dataset in datasets},
                    years={meta_dict[dataset]["year"] for dataset in datasets},
                    is_signal=False,
                    ))
        mc_processes = MCProcesses(mc_process + bg_process)

        if len(sig_bkg_dict["data"].keys()) > 1 or "data_obs" not in sig_bkg_dict["data"].keys():
            raise Exception("Only one single data process is allowed with fixed name 'data_obs'")
        for name, datasets in sig_bkg_dict["data"].items():
            data_process = DataProcess(
                    name=name,
                    samples={meta_dict[dataset]["sample"] for dataset in datasets},
                    years={meta_dict[dataset]["year"] for dataset in datasets},
                    )
        data_processes = DataProcesses([data_process])

        # -- Systematics --
        # common_systematics = [
        #     "JES_Total_AK4PFPuppi", "JER_AK4PFPuppi"
        # ]

        # Trying to make this generic. Ideally, we want exactly one single set of variations at the moment because we are using MC only for signal. This has to be improved in some shape or form.
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
                variations = {re.sub(r'(Up|Down)$', '', var) for var in variations_updown}
                try:
                    variations.remove("nominal")
                except:
                    raise ValueError(f"Variations list {variations} does not contain 'nominal'.")
                logger.info(f"Found variations: {variations}")
                for syst in variations:
                    if args.boosted and "AK4" in syst:
                        continue
                    systematics_list.append(SystematicUncertainty(name=syst, datacard_name=get_uncertainty_name(syst, meta_dict[datasets[0]]['year']), typ="shape", processes=list(sig_bkg_dict["signal"].keys()), years=[meta_dict[datasets[0]]["year"]], value=1.0))
            # At some point we will have a variation for the reweighting uncertainty, but right now, we only want variations on the MC processes
            bg_mc = {k: v for k, v in sig_bkg_dict["background"].items() if "DATA" not in v[0]}
            for sig_type, datasets in bg_mc.items():
                # Iterate through the datasets in a particular background type (often a signle one)
                variations_updown = list(sob_hist[meta_dict[datasets[0]]["sample"]][datasets[0]].axes['variation'])
                for var in variations_updown:
                    sliced = sob_hist[meta_dict[datasets[0]]["sample"]][datasets[0]][{"variation": var, "cat": region_name}]
                    print(f"Variation: {var}")
                    print(sliced.values())
                variations = {re.sub(r'(Up|Down)$', '', var) for var in variations_updown}
                try:
                    variations.remove("nominal")
                except:
                    raise ValueError(f"Variations list {variations} does not contain 'nominal'.")
                logger.info(f"Found variations: {variations}")
                for syst in variations:
                    systematics_list.append(SystematicUncertainty(name=syst, datacard_name=get_uncertainty_name(syst, meta_dict[datasets[0]]['year']), typ="shape", processes=list(bg_mc.keys()), years=[meta_dict[datasets[0]]["year"]], value=1.0))
            systematics = Systematics(systematics_list)

            _label = "run3"
            _datacard_name = f"datacard_combined_{_label}"
            _workspace_name = f"workspace_{_label}.root"

            auto_mc_stats = {
                "threshold": 10,
                "include_signal": 0,
                "hist_mode": 1,
            }

            bins_edges = None
            if args.max_bkg_unc is not None or args.min_bkg_yield is not None:
                bkg_datasets = [d for datasets in sig_bkg_dict["background"].values() for d in datasets]
                bins_edges = compute_min_stat_edges(sob_hist, region_name, bkg_datasets, meta_dict,
                                                    max_rel_unc=args.max_bkg_unc, min_yield=args.min_bkg_yield)
                os.makedirs(f"{args.output}/{hist_cat}", exist_ok=True)
                with open(f"{args.output}/{hist_cat}/bin_edges_{region_name}.json", "w") as f:
                    json.dump({"edges": bins_edges.tolist(), "max_bkg_unc": args.max_bkg_unc,
                               "min_bkg_yield": args.min_bkg_yield}, f, indent=2)

            datacard = Datacard(
                    histograms=sob_hist,
                    bins_edges=bins_edges,
                    datasets_metadata=coffea_file["datasets_metadata"],
                    cutflow=coffea_file["cutflow"],
                    systematics=systematics,
                    # This might have to change. Right now I am binding the year to the data year...
                    years={meta_dict[dataset]["year"] for dataset in sig_bkg_dict["data"]["data_obs"]},
                    mc_processes=mc_processes,
                    mcstat=auto_mc_stats,
                    data_processes=data_processes,
                    category=region_name,
                    single_year=True,
                    )
            datacard.dump(directory=f"{args.output}/{hist_cat}", card_name=f"{region_name}_{_label}.txt", shapes_name=f"shapes_{region_name}_{_label}.root")
