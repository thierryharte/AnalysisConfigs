import os
import sys
import re
from matplotlib import pyplot as plt
from coffea.util import load
from omegaconf import OmegaConf
import numpy as np
from pocket_coffea.utils.plot_utils import PlotManager
import argparse
import mplhep as hep
from multiprocessing import Pool

hep.style.use("CMS")


parser = argparse.ArgumentParser(description="Plot 2b morphed vs 4b data")
parser.add_argument("-i", "--input", type=str, required=True, help="Input directory")
parser.add_argument(
    "-o", "--output", type=str, help="Output directory", default="plots_2bVS4b"
)
parser.add_argument(
    "-n",
    "--normalisation",
    type=str,
    help="Type of normalisation (num_events, sum_weights, const_frac)",
    default="num_events",
)
parser.add_argument("-w", "--workers", type=int, default=8, help="Number of workers")
parser.add_argument(
    "-l", "--linear", action="store_true", help="Linear scale", default=False
)
args = parser.parse_args()


NORMALIZE_WEIGHTS = True

# Using the `input_dir` argument, read the default config and coffea files (if not set with argparse):
input_dir = args.input
cfg = os.path.join(input_dir, "parameters_dump.yaml")
inputfile = os.path.join(input_dir, "output_all.coffea")
log_scale = not args.linear
outputdir = os.path.join(input_dir, args.output) + f"_{args.normalisation}"

cat_dict = {
    "CR": ["4b_control_region", "2b_control_region_preW", "2b_control_region_postW"],
    "CRRun2": [
        "4b_control_regionRun2",
        "2b_control_region_preWRun2",
        "2b_control_region_postWRun2",
    ],
    "SR": ["4b_signal_region", "2b_signal_region_preW", "2b_signal_region_postW"],
    "SRRun2": [
        "4b_signal_regionRun2",
        "2b_signal_region_preWRun2",
        "2b_signal_region_postWRun2",
    ],
}


color_list = ["black", "red", "blue"]


def plot_weights(weights_list, suffix):
    fig, ax = plt.subplots()
    for i, weights in enumerate(weights_list):
        ax.hist(
            weights,
            bins=np.logspace(-3, 2, 100),
            histtype="step",
            label="weights_"
            + str(i)
            + "\nmean: {:.2f}\nstd: {:.2f}".format(np.mean(weights), np.std(weights)),
        )
        # ax.text(
        #     0.75,
        #     0.95 - 0.2 * i,
        #     "mean: {:.2f}\nstd: {:.2f}".format(
        #         np.mean(weights), np.std(weights)
        #     ),
        #     horizontalalignment="left",
        #     verticalalignment="top",
        #     transform=ax.transAxes,
        #     # ha="center",
        # )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.set_xlabel("weights")
    ax.set_ylabel("Events")
    fig.savefig(os.path.join(outputdir, f"weights_{suffix}.png"))
    plt.close(fig)


def plot_single_var_from_hist(
    var, plotter, cat_list, year, dir_cat, norm_factor_dict=None
):
    # if "Jet" in var: continue
    fig, (ax, ax_ratio) = plt.subplots(
        2,
        1,
        figsize=[13, 13],
        sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1]},
    )
    for i, cat in enumerate(cat_list):
        shape = plotter.shape_objects[f"{var}_{year}"]

        sample = list(shape.h_dict.keys())[0]

        h = shape.h_dict[sample][{"cat": cat}]
        h_num = shape.h_dict[sample][{"cat": cat_list[0]}]

        h_den = h
        if norm_factor_dict:
            norm_factor = norm_factor_dict[cat]
        else:
            norm_factor = h_num.values().sum() / h_den.values().sum()
        h_den = h_den * norm_factor
        h_ratio = (
            h_num.values() / h_den.values()
        )  # *(h_den.values().sum()/h.values().sum())

        err_num = np.sqrt(h_num.values())
        err_den = np.sqrt(h_den.values())
        ratio_err = np.sqrt(
            (err_num / h_den.values()) ** 2
            + (h_num.values() * err_den / h_den.values() ** 2) ** 2
        )

        print(f"Plotting from histograms {var} for {cat} with norm {norm_factor}")

        if "4b" in cat:
            ax.errorbar(
                h.axes[0].centers,
                h.values(),
                yerr=np.sqrt(h.values()),
                label=cat,
                color=color_list[i],
                fmt=".",
            )
        else:
            ax.step(
                h.axes[0].edges,
                np.append(h_den.values(), h_den.values()[-1]),
                where="post",
                label=cat,
                color=color_list[i],
            )

        if "4b" not in cat:
            ax_ratio.errorbar(
                h.axes[0].centers,
                h_ratio,
                yerr=ratio_err,
                fmt=".",
                label=cat,
                color=color_list[i],
            )
        else:
            ax_ratio.axhline(y=1, color=color_list[i], linestyle="--")
            ax_ratio.fill_between(
                h.axes[0].centers,
                1 - ratio_err,
                1 + ratio_err,
                color="grey",
                alpha=0.5,
            )

    ax.legend(loc="upper right")
    ax.set_yscale("log" if log_scale else "linear")
    ax.set_ylim(
        top=1.5 * ax.get_ylim()[1] if not log_scale else ax.get_ylim()[1] ** 1.5
    )
    ax_ratio.set_ylim(0.5, 1.5)

    hep.cms.lumitext(r"22EE Era E, 6 $fb^{-1}$, (13.6 TeV)", ax=ax)
    hep.cms.text(text="Preliminary", ax=ax)
    ax.grid()
    ax_ratio.grid()

    ax_ratio.set_xlabel(var)
    ax.set_ylabel("Events")
    ax_ratio.set_ylabel("Data/Pred.")

    # save figure
    fig.savefig(
        os.path.join(dir_cat, f"{var}.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


def plot_from_hist(accumulator, norm_factor_dict=None):
    variables = accumulator["variables"].keys()
    only_cat = None
    log = False
    density = True
    verbose = 1
    index_file = None
    year = "2022_postEE"
    style_cfg = parameters["plotting_style"]
    hist_objs = {v: accumulator["variables"][v] for v in variables}

    plotter = PlotManager(
        variables=variables,
        hist_objs=hist_objs,
        datasets_metadata=accumulator["datasets_metadata"],
        plot_dir=outputdir,
        style_cfg=style_cfg,
        only_cat=only_cat,
        only_year=year,
        workers=args.workers,
        log=log,
        density=density,
        verbose=verbose,
        save=False,
        index_file=index_file,
    )

    for cats_name, cat_list in cat_dict.items():
        dir_cat = f"{outputdir}/{cats_name}_histograms"
        if not os.path.exists(dir_cat):
            os.makedirs(dir_cat)
        with Pool(args.workers) as p:
            p.starmap(
                plot_single_var_from_hist,
                [
                    (var, plotter, cat_list, year, dir_cat, norm_factor_dict)
                    for var in variables
                ],
            )


def plot_single_var_from_columns(
    var, col_cat, cat_list, dir_cat, norm_factor_dict=None
):
    fig, (ax, ax_ratio) = plt.subplots(
        2,
        1,
        figsize=[13, 13],
        sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1]},
    )
    weights_plotted = False
    for i, cat in enumerate(cat_list):

        weights = col_cat[cat]["weight"]
        weights_num = col_cat[cat_list[0]]["weight"]

        # renormalize the weights
        if NORMALIZE_WEIGHTS and "Run2" not in cat and "2b" in cat and "postW" in cat:
            mean_weight = np.mean(weights)
            # mean_weight = np.std(weights)
            # std_weight = np.std(weights)
            std_weight = np.mean(weights)
            # std_weight = 1.
            
            print(
                f"Normalizing weights for {cat} with mean {mean_weight} and std {std_weight}"
            )
            original_weights = weights
            weights = (weights - mean_weight) / std_weight + 1.0
            print(f"New mean: {np.mean(weights)} and std: {np.std(weights)}")
            if not weights_plotted:
                plot_weights([original_weights, weights], f"{cat}_normalized")
                weights_plotted = True

        col_num = col_cat[cat_list[0]][var]

        mask_w = weights > -1
        weights = weights[mask_w]
        col = col_cat[cat][var][mask_w]

        if norm_factor_dict:
            norm_factor = norm_factor_dict[cat]
        else:
            norm_factor = weights_num.sum() / weights.sum()

        print(
            f"Plotting from columns {var} for {cat} with norm {norm_factor} and weights sum {weights.sum()}"
        )

        range_4b = (np.min(col), np.max(col)) if "4b" in cat else range_4b
        weights = weights[(col > range_4b[0]) & (col < range_4b[1])]
        col = col[(col > range_4b[0]) & (col < range_4b[1])]

        h, bins = np.histogram(
            col, bins=30, weights=weights * norm_factor, range=range_4b
        )
        bins_center = (bins[1:] + bins[:-1]) / 2
        if "4b" in cat:
            ax.errorbar(
                bins_center,
                h,
                yerr=np.sqrt(h),
                label=cat,
                color=color_list[i],
                fmt=".",
            )
        else:
            ax.hist(
                col,
                bins=30,
                histtype="step",
                label=cat,
                weights=weights * norm_factor,
                color=color_list[i],
            )

        # draw the ratio
        h_den = h
        h_num, bins = np.histogram(
            col_num, bins=30, weights=weights_num, range=range_4b
        )
        ratio = h_num / h_den
        err_num = np.sqrt(h_num)
        err_den = np.sqrt(h_den)
        ratio_err = np.sqrt((err_num / h_den) ** 2 + (h_num * err_den / h_den**2) ** 2)
        if "4b" not in cat:
            ax_ratio.errorbar(
                bins_center,
                ratio,
                yerr=ratio_err,
                fmt=".",
                label=cat,
                color=color_list[i],
            )
        else:
            ax_ratio.axhline(y=1, color=color_list[i], linestyle="--")
            ax_ratio.fill_between(
                bins_center,
                1 - ratio_err,
                1 + ratio_err,
                color="grey",
                alpha=0.5,
            )
    ax.legend(loc="upper right")
    ax.set_yscale("log" if log_scale else "linear")
    hep.cms.lumitext(r"22EE Era E, 6 $fb^{-1}$, (13.6 TeV)", ax=ax)
    hep.cms.text(text="Preliminary", ax=ax)

    ax_ratio.set_xlabel(var)
    ax.set_ylabel("Events")
    ax_ratio.set_ylabel("Data/Pred.")

    ax.grid()
    ax_ratio.grid()
    ax_ratio.set_ylim(0.5, 1.5)
    ax.set_ylim(
        top=1.5 * ax.get_ylim()[1] if not log_scale else ax.get_ylim()[1] ** 1.5
    )
    fig.savefig(
        os.path.join(dir_cat, f"{var}.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


def plot_from_columns(accumulator, norm_factor_dict=None):
    col_cat = accumulator["columns"][sample][dataset]

    for cats_name, cat_list in cat_dict.items():
        dir_cat = f"{outputdir}/{cats_name}_columns"
        if not os.path.exists(dir_cat):
            os.makedirs(dir_cat)
        vars = col_cat[cat_list[0]].keys()
        with Pool(args.workers) as p:
            p.starmap(
                plot_single_var_from_columns,
                [(var, col_cat, cat_list, dir_cat, norm_factor_dict) for var in vars],
            )


if __name__ == "__main__":

    # Load yaml file with OmegaConf
    if cfg[-5:] == ".yaml":
        parameters_dump = OmegaConf.load(cfg)
    else:
        raise Exception(
            "The input file format is not valid. The config file should be a in .yaml format."
        )

    parameters = parameters_dump

    # Resolving the OmegaConf
    try:
        OmegaConf.resolve(parameters)
    except Exception as e:
        print(
            "Error during resolution of OmegaConf parameters magic, please check your parameters files."
        )
        raise (e)

    if os.path.isfile(inputfile):
        accumulator = load(inputfile)
    else:
        sys.exit(f"Input file '{inputfile}' does not exist")

    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    for sample in accumulator["columns"].keys():
        for dataset in accumulator["columns"][sample].keys():
            for category in accumulator["columns"][sample][dataset].keys():
                col = accumulator["columns"][sample][dataset][category]
                for k in col.keys():
                    col[k] = col[k].value

    # plot the weights
    sample = list(accumulator["columns"].keys())[0]
    dataset = list(accumulator["columns"][sample].keys())[0]
    for category in accumulator["columns"][sample][dataset].keys():
        col = accumulator["columns"][sample][dataset][category]
        weights = col["weight"]

        plot_weights([weights], category)

    # Get the normalization factors
    num_4b_CR = accumulator["cutflow"]["4b_control_region"][
        "DATA_JetMET_JMENano_2022_postEE_EraE"
    ]["DATA_JetMET_JMENano_skimmed"]
    num_2b_CR = accumulator["cutflow"]["2b_control_region_preW"][
        "DATA_JetMET_JMENano_2022_postEE_EraE"
    ]["DATA_JetMET_JMENano_skimmed"]
    num_4b_SR = accumulator["cutflow"]["4b_signal_region"][
        "DATA_JetMET_JMENano_2022_postEE_EraE"
    ]["DATA_JetMET_JMENano_skimmed"]
    num_2b_SR = accumulator["cutflow"]["2b_signal_region_preW"][
        "DATA_JetMET_JMENano_2022_postEE_EraE"
    ]["DATA_JetMET_JMENano_skimmed"]
    num_4b_CRRun2 = accumulator["cutflow"]["4b_control_regionRun2"][
        "DATA_JetMET_JMENano_2022_postEE_EraE"
    ]["DATA_JetMET_JMENano_skimmed"]
    num_2b_CRRun2 = accumulator["cutflow"]["2b_control_region_preWRun2"][
        "DATA_JetMET_JMENano_2022_postEE_EraE"
    ]["DATA_JetMET_JMENano_skimmed"]
    num_4b_SRRun2 = accumulator["cutflow"]["4b_signal_regionRun2"][
        "DATA_JetMET_JMENano_2022_postEE_EraE"
    ]["DATA_JetMET_JMENano_skimmed"]
    num_2b_SRRun2 = accumulator["cutflow"]["2b_signal_region_preWRun2"][
        "DATA_JetMET_JMENano_2022_postEE_EraE"
    ]["DATA_JetMET_JMENano_skimmed"]

    norm_factor_dict = {
        "4b_control_region": 1,
        "2b_control_region_preW": num_4b_CR / num_2b_CR,
        "2b_control_region_postW": num_4b_CR / num_2b_CR,
        "4b_signal_region": 1,
        "2b_signal_region_preW": num_4b_CR / (num_2b_CR),
        "2b_signal_region_postW": num_4b_CR / (num_2b_CR),
        "4b_control_regionRun2": 1,
        "2b_control_region_preWRun2": num_4b_CRRun2 / num_2b_CRRun2,
        "2b_control_region_postWRun2": num_4b_CRRun2 / num_2b_CRRun2,
        "4b_signal_regionRun2": 1,
        "2b_signal_region_preWRun2": num_4b_CRRun2 / (num_2b_CRRun2),
        "2b_signal_region_postWRun2": num_4b_CRRun2 / (num_2b_CRRun2),
    }
    if args.normalisation == "sum_weights":
        norm_factor_dict = None
    elif args.normalisation == "const_frac":
        norm_factor_dict = {
            k: (0.018824706 if "2b" in k else 1.0) for k in norm_factor_dict.keys()
        }
    elif args.normalisation == "num_events":
        pass
    else:
        raise ValueError(f"Normalisation type {args.normalisation} not recognised")

    print("norm_factor_dict", norm_factor_dict)

    plot_from_hist(accumulator, norm_factor_dict)
    plot_from_columns(accumulator, norm_factor_dict)

    print(f"\nPlots saved in {outputdir}")
