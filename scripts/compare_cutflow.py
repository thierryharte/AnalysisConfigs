# load a coffea file
from coffea.util import load
import os
import awkward as ak
import numpy as np
import logging
import matplotlib.pyplot as plt
import mplhep as hep
import matplotlib
import argparse

matplotlib.rcParams["figure.dpi"] = 300
from collections import OrderedDict

hep.style.use("CMS")




parser = argparse.ArgumentParser(description="Plot 2b morphed vs 4b data")
parser.add_argument(
    "-i",
    "--input-file",
    type=str,
    required=True,
    help="Input coffea file for data",
)
parser.add_argument(
    "-o", "--output", type=str, help="Output directory", default="plots_cutflow"
)
parser.add_argument(
    "-p",
    "--four-jet-presel",
    action="store_true",
    help="If true, the effiiencies are computed w.r.t. the 4 jets, pT>25 GeV, |eta|<2.5 preselection",
    default=False,
)
args = parser.parse_args()

EFF_WRT_4JETS_PRESEL = args.four_jet_presel

# make output directory if it does not exist
if not os.path.exists(args.output):
    os.makedirs(args.output)

# Logger
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()
formatter = logging.Formatter("%(asctime)s  - %(levelname)s - %(message)s")
file_handler = logging.FileHandler(f"{args.output}/compare_cutflow.log")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)

cumulative_eff_AN = [0, 100, 94, 86, 29, 23, 22, 18, 7, 4.7] if EFF_WRT_4JETS_PRESEL  else [0, 100, 94, 86, 29, 23, 22, 18, 7]
cumulative_eff_AN = [x / 100 for x in cumulative_eff_AN]


def autolabel(ax, bars):
    for bar in bars:
        height = bar.get_height()
        # transform to percentage
        percentage = height * 100

        ax.annotate(
            f"{percentage:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
        )


def plot_efficiencies(list_cuts, efficiencies, efficiencies_names, title, output_name):
    # plot the efficiencies
    cut_names = list_cuts
    x = np.arange(len(cut_names))
    eff_vals = [e[0] for e in efficiencies]
    total_eff_vals = [e[1] for e in efficiencies]
    fig, ax = plt.subplots(figsize=(19, 13))
    # plot the 3 bars
    width = 0.25
    bars1 = ax.bar(x - width, eff_vals, width, label=efficiencies_names[0])
    bars2 = ax.bar(x, total_eff_vals, width, label=efficiencies_names[1])
    # logger.info also the values on top of the bars
    autolabel(ax, bars1)
    autolabel(ax, bars2)

    # check if need to plot also efficiency wrt to initial events
    if len(efficiencies[0]) == 3:
        total_eff_from_initial_vals = [e[2] for e in efficiencies]
        bars3 = ax.bar(
            x + width, total_eff_from_initial_vals, width, label=efficiencies_names[2]
        )
        autolabel(ax, bars3)

    ax.set_ylabel("Efficiency %")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(cut_names, rotation=45, ha="right")
    ax.legend(fontsize="small")
    ax.set_ylim(0, 1.2)
    hep.style.use(hep.style.CMS)
    hep.cms.lumitext("(13.6 TeV)", ax=ax)
    hep.cms.text("Preliminary", ax=ax)
    plt.tight_layout()
    plt.savefig(f"{args.output}/{output_name}.png")


def compute_cutflow():
    # compute the efficiencies of each cut respect to the previous one and also to the initial number of events
    efficiencies = []
    list_cuts = list(num_events_dict.keys())
    num_events_presel = num_events_dict[list_cuts[0]]
    relative_efficiencies = []
    cumulative_efficiencies = []
    for name, num_events in num_events_dict.items():
        prev_cut_name = list_cuts[list_cuts.index(name) - 1]
        prev_cut_events = num_events_dict[prev_cut_name]
        if name == list_cuts[0]:
            total_eff = 1
            total_eff_from_initial = num_events / num_events_initial
            eff = 1
        else:
            total_eff = num_events / num_events_presel if num_events_presel > 0 else 0
            total_eff_from_initial = num_events / num_events_initial
            eff = num_events / prev_cut_events if prev_cut_events > 0 else 0

        relative_efficiencies.append(eff)
        cumulative_efficiencies.append(total_eff)
        efficiencies.append(
            (eff, total_eff, total_eff_from_initial)
            if EFF_WRT_4JETS_PRESEL
            else (eff, total_eff)
        )

    efficiencies_names = (
        [
            "Efficiency w.r.t. previous cut",
            r"Efficiency w.r.t. 4 jets, $p_{\mathrm{T}}$>25 GeV, |$\eta$|<2.5",
            "Efficiency w.r.t. initial events",
        ]
        if EFF_WRT_4JETS_PRESEL
        else [
            "Efficiency w.r.t. previous cut",
            "Efficiency w.r.t. initial events",
        ]
    )

    plot_efficiencies(
        list_cuts,
        efficiencies,
        efficiencies_names,
        "Cutflow with PocketCoffea",
        "Cutflow_PocketCoffea",
    )

    # Efficiency from the AN
    list_cuts_AN = list(num_events_dict.keys())
    # replace element in list
    list_cuts_AN[list_cuts_AN.index("2b_selection")] = "HLT_jet_matching+2b_selection"
    # list_cuts_AN[list_cuts_AN.index("HLT_selection")]="L1+HLT_selection"

    # compute the relative efficiency wrt to last cut
    relative_eff_AN = []
    for i in range(len(cumulative_eff_AN)):
        if i == 0:
            relative_eff_AN.append(0)
            continue
        if cumulative_eff_AN[i - 1] != 0:
            rel_eff = cumulative_eff_AN[i] / cumulative_eff_AN[i - 1]
        else:
            rel_eff = cumulative_eff_AN[i] / 1
        relative_eff_AN.append(rel_eff)

    efficiencies_AN = [
        (relative_eff_AN[i], cumulative_eff_AN[i])
        for i in range(len(cumulative_eff_AN))
    ]
    efficiencies_names_AN = [
        "Efficiency w.r.t. previous cut",
        (
            r"Efficiency w.r.t. 4 jets, $p_{\mathrm{T}}$>25 GeV, |$\eta$|<2.5"
            if EFF_WRT_4JETS_PRESEL
            else "Efficiency w.r.t. initial events"
        ),
    ]

    plot_efficiencies(
        list_cuts_AN,
        efficiencies_AN,
        efficiencies_names_AN,
        "Cutflow from AN-23-184",
        "Cutflow_AN_23_184",
    )

    efficiencies_rel = [
        (relative_eff_AN[i], relative_efficiencies[i])
        for i in range(len(relative_efficiencies))
    ]
    efficiencies_names_comparison = ["AN-23-184", "PocketCoffea"]
    list_cuts_comp = list_cuts.copy()
    list_cuts_comp[list_cuts_comp.index("2b_selection")] = (
        "(HLT_jet_matching+)2b_selection"
    )
    # list_cuts_comp[list_cuts_comp.index("HLT_selection")]="(L1+)HLT_selection"

    plot_efficiencies(
        list_cuts_comp,
        efficiencies_rel,
        efficiencies_names_comparison,
        "Efficiency w.r.t. previous cut",
        "Efficiency_wrt_previous_cut_comparison",
    )

    efficiencies_cum = [
        (cumulative_eff_AN[i], cumulative_efficiencies[i])
        for i in range(len(cumulative_efficiencies))
    ]
    plot_efficiencies(
        list_cuts_comp,
        efficiencies_cum,
        efficiencies_names_comparison,
        (
            r"Efficiency w.r.t. 4 jets, $p_{\mathrm{T}}$>25 GeV, |$\eta$|<2.5"
            if EFF_WRT_4JETS_PRESEL
            else "Efficiency w.r.t. initial events"
        ),
        (
            "Efficiency_wrt_4jets_presel_comparison"
            if EFF_WRT_4JETS_PRESEL
            else "Efficiency_wrt_initial_events_comparison"
        ),
    )


if __name__ == "__main__":
    # Load coffea file
    input_file = args.input_file
    logger.info(f"Loading coffea file: {input_file}")
    o = load(input_file)

    logger.info(o["cutflow"])
    for k in o["cutflow"].keys():
        logger.info(k)
        for kk in o["cutflow"][k].keys():
            logger.info(f"\t{kk} {o['cutflow'][k][kk]}")
    dataset = list(o["cutflow"]["initial"].keys())[0]
    logger.info(f"Dataset: {dataset}")
    # sample=list(o["columns"].keys())[0]
    sample = "GluGlutoHHto4B"
    logger.info(f"Sample: {sample}")

    num_events_dict = OrderedDict()
    for k in o["cutflow"].keys():
        for kk in o["cutflow"][k].keys():
            if k == "initial":
                num_events_initial = o["cutflow"][k][kk]
            if type(o["cutflow"][k][kk]) == int or "no_selections" in k:
                continue
            new_value = o["cutflow"][k][kk][sample]
            logger.info(k)
            logger.info(f"\t{new_value}")
            num_events_dict[k] = new_value

    # Compute the total yield for MC
    if o["columns"] and o["sum_genweights"]:
        for category in o["columns"][sample][dataset].keys():
            logger.info(f"Category: {category}")
            total_yield = (
                sum(o["columns"][sample][dataset][category]["weight"].value)
                / o["sum_genweights"][dataset]
            )
            logger.info(
                f"Total yield for MC sample {sample} in category {category}: {total_yield}"
            )

    compute_cutflow()
