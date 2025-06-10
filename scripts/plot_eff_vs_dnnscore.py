from coffea.util import load
import os
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import matplotlib
matplotlib.rcParams["figure.dpi"] = 300
import matplotlib as mpl

import vector
vector.register_awkward

from utils.plot.get_era_lumi import get_era_lumi
from utils.plot.weighted_quantile import weighted_quantile
from utils.plot.plot_names import plot_regions_names
import argparse

hep.style.use("CMS")


import matplotlib

matplotlib.rcParams["agg.path.chunksize"] = 10000  # or try 5000, depending on size
NUMBER_OF_BINS = 20

parser = argparse.ArgumentParser(description="Plot 2b morphed vs 4b data")
parser.add_argument(
    "-is",
    "--input-spanet",
    type=str,
    default="/work/tharte/datasets/data_samples/spanet_ptflat_true_idx/output_GluGlutoHHto4B_spanet_kl-1p00_kt-1p00_c2-0p00_2022_postEE.coffea")
parser.add_argument(
    "-ir",
    "--input-run2",
    type=str,
    default = "/work/tharte/datasets/data_samples/DHH_method_true_idx/output_GluGlutoHHto4B_spanet_kl-1p00_kt-1p00_c2-0p00_2022_postEE.coffea")
parser.add_argument(
    "-o", "--output", type=str, help="Output directory", default="./plots_eff_vs_dnnscore"
)
args = parser.parse_args()

o = load(f"{args.input_spanet}")
orun2 = load(f"{args.input_run2}")
#if "sum_genweights" in o:
#    print(f'Genweight : {o["sum_genweights"]}')

sample = list(o["columns"].keys())[0]
dataset = list(o["columns"][sample].keys())[0]
category = "4b_signal_region"
category_Run2 = "4b_signal_regionRun2"
print(sample)
print(dataset)
cols_spanet = o["columns"][sample][dataset][category]
cols_run2 = orun2["columns"][sample][dataset][category_Run2]

# bin the dataset into bins of same amount of data:
dnn_score = cols_spanet["events_sig_bkg_dnn_score"].value
print(cols_run2.keys())
dnn_scoreR2 = cols_run2["events_sig_bkg_dnn_scoreRun2"].value
# dnn_score = cols_spanet["events_dR_min"].value
print(min(dnn_score))
print(type(dnn_score))
bin_edges = weighted_quantile(
    dnn_score,
    np.linspace(0, 1, NUMBER_OF_BINS + 1))
# Overwriting the bins due to combination of SPANet and DHH
bin_edges = np.linspace(0,1, NUMBER_OF_BINS +1)
print(bin_edges)

eff_list = { 
    "SPANet efficiency": [],
    "SPANet total efficiency": [],
    "DHH method efficiency": [],
    "DHH method total efficiency": [],
}

for low, high in zip(bin_edges[:-1],bin_edges[1:]):
    print(f"Range bin: {low} - {high}")

    # Spanet
    mask_low = ak.Array(dnn_score) > low  
    mask_high = ak.Array(dnn_score) < high
    mask = mask_low & mask_high
    pairings_bin = cols_spanet["events_correct_prediction"].value[mask]
    pairings_bin_matched = cols_spanet["events_correct_prediction"].value[mask][cols_spanet["events_mask_fully_matched"].value[mask]]
    print(f"Pairing bin: {sum(pairings_bin)}, {len(pairings_bin)}")
    print(f"Pairing bin matched: {sum(pairings_bin_matched)}, {len(pairings_bin_matched)}")

    eff = sum(pairings_bin_matched)/len(pairings_bin_matched)
    tot_eff = sum(pairings_bin)/len(pairings_bin)
    print(f"Efficiency: {eff}")
    print(f"Total Efficiency: {tot_eff}")

    # Run2
    mask_low = ak.Array(dnn_scoreR2) > low  
    mask_high = ak.Array(dnn_scoreR2) < high
    mask = mask_low & mask_high
    pairings_binR2 = cols_run2["events_correct_predictionRun2"].value[mask]
    pairings_bin_matchedR2 = cols_run2["events_correct_predictionRun2"].value[mask][cols_run2["events_mask_fully_matched"].value[mask]]
    print(f"Pairing bin: {sum(pairings_binR2)}, {len(pairings_binR2)}")
    print(f"Pairing bin matched: {sum(pairings_bin_matchedR2)}, {len(pairings_bin_matchedR2)}")


    effR2 = sum(pairings_bin_matchedR2)/len(pairings_bin_matchedR2)
    tot_effR2 = sum(pairings_binR2)/len(pairings_binR2)

    print(f"Efficiency DHH: {effR2}")
    print(f"Total Efficiency DHH: {tot_effR2}")

    eff_list["SPANet efficiency"].append(eff)
    eff_list["SPANet total efficiency"].append(tot_eff)
    eff_list["DHH method efficiency"].append(effR2)
    eff_list["DHH method total efficiency"].append(tot_effR2)


labels = ["Pairing Efficiency", "Total Pairing Efficiency"]
arrays = [["SPANet efficiency","DHH method efficiency"],["SPANet total efficiency","DHH method total efficiency"]]
col_list = [["blue","red"],["blue","red"]]
plotnames = ["pairing_efficiency", "total_pairing_efficiency"]

#bins = (bin_edges[:-1]+bin_edges[1:])/2
bins = np.linspace(0, 1, NUMBER_OF_BINS)

for label, arrays, colors, plotname in zip(labels, arrays,col_list,plotnames):
    fig_events, ax_events = plt.subplots(figsize=[13, 13])
    for idx, array_name in enumerate(arrays):
        ax_events.errorbar(
            bins,
            eff_list[array_name],
            yerr=np.zeros_like(eff_list),     
            fmt=".",
            label=array_name,   
            color=colors[idx],  
        )
    ax_events.legend(loc="upper left")
    ax_events.set_yscale("linear")
    #hep.cms.lumitext(
    #    f"{era_string}, {lumi}" + r" $fb^{-1}$, (13.6 TeV)",  
    #    ax=ax_events,  
    #)
    ax_events.set_xlabel("DNN Score")
    ax_events.set_ylabel(label)    
    ax_events.grid()
    fig_events.show()
    fig_events.savefig(
            os.path.join(args.output, f"{plotname}.png"),
        bbox_inches="tight",   
        dpi=300,
    )
    plt.close(fig_events)
