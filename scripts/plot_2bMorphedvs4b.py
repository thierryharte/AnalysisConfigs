import os
import sys
from matplotlib import pyplot as plt
from coffea.util import load
from omegaconf import OmegaConf
import numpy as np
from scipy.stats.distributions import chi2
from pocket_coffea.utils.plot_utils import PlotManager
import argparse
import mplhep as hep
from multiprocessing import Pool

from utils.plot.get_era_lumi import get_era_lumi
from utils.plot.get_columns_from_files import get_columns_from_files
from utils.plot.weighted_quantile import weighted_quantile
from utils.plot.plot_names import plot_regions_names

hep.style.use("CMS")


parser = argparse.ArgumentParser(description="Plot 2b morphed vs 4b data")
parser.add_argument(
    "-i",
    "--input-data",
    type=str,
    required=True,
    help="Input directory for data with coffea files or coffea file itself",
)
parser.add_argument(
    "-im", "--input-mc", type=str, help="Input coffea file monte carlo", default=None
)
parser.add_argument(
    "-o", "--output", type=str, help="Output directory", default="plots_2bVS4b"
)
parser.add_argument(
    "-n",
    "--normalisation",
    type=str,
    help="Type of normalisation (num_events, sum_weights)",
    default="sum_weights",
)
parser.add_argument(
    "-r",
    "--region-suffix",
    type=str,
    help="Suffix for the region",
    default="",
)
parser.add_argument("-w", "--workers", type=int, default=8, help="Number of workers")
parser.add_argument(
    "-l", "--linear", action="store_true", help="Linear scale", default=False
)
parser.add_argument(
    "-t", "--test", action="store_true", help="Test on one variable", default=False
)
parser.add_argument(
    "-s",
    "--spread",
    action="store_true",
    help="Perform the spread morphing plot of the DNN score",
    default=False,
)

args = parser.parse_args()

if args.test:
    args.workers = 1
    args.output = "test"

NUMBER_OF_BINS = 20
PAD_VALUE = -999
BLIND_VALUE=0.9

input_dir = os.path.dirname(args.input_data)
log_scale = not args.linear
outputdir = os.path.join(input_dir, args.output) + f"_{args.normalisation}"


# To mix categories with Run2 and SPANet, put first the Run2 category
# because first the name of the variables is try with the Run2 string
# and after without it
cat_dict = {}
cat_dict |= {
    f"CR{args.region_suffix}": [
        f"4b{args.region_suffix}_control_region",
        f"2b{args.region_suffix}_control_region_postW",
        f"2b{args.region_suffix}_control_region_preW",
    ],
    f"CR{args.region_suffix}Run2": [
        f"4b{args.region_suffix}_control_regionRun2",
        f"2b{args.region_suffix}_control_region_postWRun2",
        f"2b{args.region_suffix}_control_region_preWRun2",
    ],
    f"SR{args.region_suffix}": [
        f"4b{args.region_suffix}_signal_region",
        f"2b{args.region_suffix}_signal_region_postW",
        f"2b{args.region_suffix}_signal_region_preW",
    ],
    # f"SR{args.region_suffix}_blind": [
    #     f"4b{args.region_suffix}_signal_region_blind",
    #     f"2b{args.region_suffix}_signal_region_postW_blind",
    #     f"2b{args.region_suffix}_signal_region_preW_blind",
    # ],
    f"SR{args.region_suffix}Run2": [
        f"4b{args.region_suffix}_signal_regionRun2",
        f"2b{args.region_suffix}_signal_region_postWRun2",
        f"2b{args.region_suffix}_signal_region_preWRun2",
    ],
    # f"SR{args.region_suffix}_blindRun2": [
    #     f"4b{args.region_suffix}_signal_region_blindRun2",
    #     f"2b{args.region_suffix}_signal_region_postW_blindRun2",
    #     f"2b{args.region_suffix}_signal_region_preW_blindRun2",
    # ],
    #
    # Special case for the 2b morphed with the spread of the morphing weights
    # Keyword is "SPREAD"
    #
    f"SR{args.region_suffix}_SPREAD": [
        f"2b{args.region_suffix}_signal_region_postW",
        f"2b{args.region_suffix}_signal_region_postW_SPREAD",
    ],
    # f"CR{args.region_suffix}_2b_Run2SPANet": [f"2b{args.region_suffix}_control_region_preWRun2", f"2b{args.region_suffix}_control_region_preW"],
    # f"CR{args.region_suffix}_4b_Run2SPANet": [f"4b{args.region_suffix}_control_regionRun2", f"4b{args.region_suffix}_control_region"],
}

if args.test:
    cat_dict = {
        f"CR": [
            f"4b_control_region",
            f"2b_control_region_postW",
            f"2b_control_region_preW",
        ],
    }


color_list_orig = [("black",), ("blue", "dodgerblue"), ("red",)]
color_list_spread = [("green", "red")] + [("green",)] * 20 + [("orange",)] + [("blue",)]
color_list_alt = [("purple",), ("darkorange", "orange"), ("green",)]


if not os.path.exists(outputdir):
    os.makedirs(outputdir)

# load the data
if args.input_data.endswith(".coffea"):
    inputfiles = [args.input_data]
else:
    # get list of coffea files
    inputfiles = [
        os.path.join(input_dir, file)
        for file in os.listdir(input_dir)
        if file.endswith(".coffea") and "DATA" in file
    ]

filter_lambda = (lambda x: ("weight" in x or "score" in x)) if args.spread else None
cat_col_data, total_datasets_list = get_columns_from_files(inputfiles, filter_lambda)

if args.input_mc:
    inputfiles_mc = [args.input_mc]
    cat_col_mc, _ = get_columns_from_files(inputfiles_mc, filter_lambda)
    cols_sig_mc = cat_col_mc[f"4b{args.region_suffix}_signal_region"]
    for col in cols_sig_mc:
        print(col)
        if "score" in col:
            
            print("\n WEIGHTED")
            CONSTANT_SIGNAL_BINS =weighted_quantile(
                cols_sig_mc[col], np.linspace(0, 1, NUMBER_OF_BINS + 1), weights=cols_sig_mc["weight"]
            )
            CONSTANT_SIGNAL_BLIND_BINS = CONSTANT_SIGNAL_BINS[CONSTANT_SIGNAL_BINS< BLIND_VALUE]
            print(f"Constant signal bins: {CONSTANT_SIGNAL_BINS}")
            score_hist= np.histogram(
                cols_sig_mc[col], bins=CONSTANT_SIGNAL_BINS, weights=cols_sig_mc["weight"]
            )
            print(f"Constant signal bins histogram: {score_hist}")

    CONST_SIG_BINNING = True
else:
    CONST_SIG_BINNING = False

def plot_weights(weights_list, suffix, lumi, era_string):
    fig, ax = plt.subplots(figsize=[13, 13])
    for i, weights in enumerate(weights_list):
        ax.hist(
            weights,
            bins=np.logspace(-3, 2, 100),
            histtype="step",
            label="Morphing weights "
            + (f"{i}" if len(weights_list) > 1 else "")
            + "\nmean: {:.2f}\nstd: {:.2f}".format(np.mean(weights), np.std(weights)),
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.set_xlabel("Morphing weights")
    ax.set_ylabel("Events")

    hep.cms.lumitext(f"{era_string}, {lumi}" + r" $fb^{-1}$, (13.6 TeV)", ax=ax)
    hep.cms.text(text="Preliminary", ax=ax)

    fig.savefig(os.path.join(outputdir, f"weights_{suffix}.png"))
    plt.close(fig)


def plot_single_var_from_columns(
    var,
    col_dict,
    weight_dict,
    cats_name,
    cat_list,
    dir_cat,
    chi_squared,
    color_list,
    lumi,
    era_string,
):
    fig, (ax, ax_ratio) = plt.subplots(
        2,
        1,
        figsize=[13, 13],
        sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1]},
    )

    print(var)
    ratios_spread = []
    histos_spread = []

    for i, cat in enumerate(cat_list):

        if "SPREAD" in cats_name:
            if "SPREAD" in cat:
                cat_plot_name=plot_regions_names(cat, " (k-folds)")
            else:
                cat_plot_name=plot_regions_names(cat, " (mean weight per event)")
            cat_plot_name_alt=plot_regions_names(cat," (median per bin)")
        else:
            cat_plot_name=plot_regions_names(cat)

        weights_den = weight_dict[cat]
        weights_num = weight_dict[cat_list[0]]

        col_den = col_dict[cat]
        col_num = col_dict[cat_list[0]]

        # remove padded values
        weights_den = weights_den[col_den != PAD_VALUE]
        weights_num = weights_num[col_num != PAD_VALUE]
        col_den = col_den[col_den != PAD_VALUE]
        col_num = col_num[col_num != PAD_VALUE]

        # WARNING: the weights are different for the additional jets because
        # the number of events is different since it's computed after the masking of the PAD_VALUE
        if args.normalisation == "num_events":
            norm_factor_den = len(weights_den) / len(weights_num)
            norm_factor_num = 1.0
        else:
            norm_factor_den = weights_num.sum() / weights_den.sum()
            norm_factor_num = 1.0
        print(
            f"Plotting from columns {var} for {cat} with norm {norm_factor_den} and weights sum {weights_den.sum()}"
        )

        # normalize the weights
        weights_den = weights_den * norm_factor_den
        weights_num = weights_num * norm_factor_num

        # fix the bins and range
        if "score" in var and CONST_SIG_BINNING:
            bins = (
                CONSTANT_SIGNAL_BINS
                if "blind" not in cat
                else CONSTANT_SIGNAL_BLIND_BINS
            )
        else:
            # compute the range of the 4b category considering the 0.1% and 99.9% quantile
            range_4b = (
                tuple(np.quantile(col_den, [0.001, 0.999])) if i == 0 else range_4b
            )

            print(f"range_4b {range_4b}")

            mask_num_range4b = (col_num > range_4b[0]) & (col_num < range_4b[1])
            weights_num = weights_num[mask_num_range4b]
            col_num = col_num[mask_num_range4b]

            mask_den_range4b = (col_den > range_4b[0]) & (col_den < range_4b[1])
            weights_den = weights_den[mask_den_range4b]
            col_den = col_den[mask_den_range4b]

            bins = np.linspace(range_4b[0], range_4b[1], NUMBER_OF_BINS + 1)
            
            # print(f"weights_den {weights_den}", type(weights_den))
            # print(f"weights_num {weights_num}")
            # print(f"col_num {col_num}", type(col_num))
            # print(f"col_den {col_den}")


        idx_den = np.digitize(col_den, bins)
        idx_num = np.digitize(col_num, bins)
        
        if "TRANSFORM" in var:
            bins = np.linspace(
                bins[0], bins[-1], NUMBER_OF_BINS + 1
            )
        bins_center = (bins[1:] + bins[:-1]) / 2

        h_den = []
        h_num = []
        err_den = []
        err_num = []

        for j in range(1, len(bins)):
            h_den.append(np.sum(weights_den[idx_den == j]))
            h_num.append(np.sum(weights_num[idx_num == j]))
            err_den.append(np.sqrt(np.sum(weights_den[idx_den == j] ** 2)))
            err_num.append(np.sqrt(np.sum(weights_num[idx_num == j] ** 2)))
            # print('weights_den[idx_den == j]', weights_den[idx_den == j])

        h_den = np.array(h_den)
        h_num = np.array(h_num)
        err_den = np.array(err_den)
        err_num = np.array(err_num)

        # print("h_den", h_den, len(h_den))
        # print("h_num", h_num, len(h_num))
        # print("err_den", err_den)
        # print("err_num", err_num)

        chi2_norm = None
        if i > 0 and chi_squared:
            # compute the chi square between the two histograms (divide by the error on data)
            chi2s = ((h_den - h_num) / np.where(err_num == 0, 1, err_num)) ** 2
            chi2_value = np.sum(chi2s)
            ndof = len(h_den) - 1
            chi2_norm = chi2_value / ndof
            pvalue = chi2.sf(chi2_value, ndof)
            print("chi2", chi2s, chi2_value, ndof, chi2_norm)

        ratio = h_num / h_den

        print("ratio", ratio)
        if i == 0:
            ratio_err = err_num / h_num
            ax.errorbar(
                bins_center,
                h_den,
                yerr=err_den,
                label=cat_plot_name,
                color=color_list[i][0],
                fmt=".",
            )

            ax_ratio.axhline(y=1, color=color_list[i][0], linestyle="--")
            if "SPREAD" in cats_name:
                ax_ratio.step(
                    bins,
                    1 - np.append(ratio_err, ratio_err[-1]),
                    where="post",
                    color=color_list[i][1],
                    label=r"$\pm \sigma_{stat}$",
                    linewidth=1,
                )
                ax_ratio.step(
                    bins,
                    1 + np.append(ratio_err, ratio_err[-1]),
                    where="post",
                    color=color_list[i][1],
                    linewidth=1,
                )
            else:
                ax_ratio.fill_between(
                    bins_center,
                    1 - ratio_err,
                    1 + ratio_err,
                    color="grey",
                    alpha=0.5,
                )

        else:
            ratio_err = np.sqrt(
                (err_num / h_den) ** 2 + (h_num * err_den / h_den**2) ** 2
            )

            # ax.hist(
            #     col_den,
            #     bins=bins,
            #     histtype="step",
            #     label=cat_plot_name if "SPREAD" not in cat or i == 1 else None,
            #     weights=weights_den,
            #     edgecolor=color_list[i][0],
            #     facecolor=color_list[i][1] if len(color_list[i]) > 1 else None,
            #     fill=True if len(color_list[i]) > 1 else False,
            #     # linestyle="--" if "SPREAD" in cat else "-",
            #     alpha=0.5,
            # )
            
            ## plot the histogram
            ax.step(
                bins,
                np.append(h_den, h_den[-1]),
                where="post",
                label=cat_plot_name if "SPREAD" not in cat or i == 1 else None,
                color=color_list[i][0],
                linewidth=1,
            )

            x0 = bins[0]
            x1 = bins[-1]
            y0 = h_den[0]
            y1 = h_den[-1]
            ax.plot(
                [x0, x0], [ax.get_ylim()[0], y0], color=color_list[i][0]
            )  # first bin edge
            ax.plot(
                [x1, x1], [ax.get_ylim()[0], y1], color=color_list[i][0]
            )  # last bin edge

            if len(color_list[i]) > 1:
                ax.fill_between(
                    bins,
                    np.append(h_den, h_den[-1]),
                    step="post",
                    alpha=0.5,
                    color=color_list[i][1],
                )

            if "SPREAD" in cat:
                histos_spread.append(h_den)
                # plot the spread of the DNN score as histogram in the ratio
                ax_ratio.step(
                    bins,
                    np.append(ratio, ratio[-1]),
                    where="post",
                    color=color_list[i][0],
                    # linestyle="--",
                    linewidth=1,
                    zorder=0,
                )
                ratios_spread.append(ratio)
            else:
                ax_ratio.errorbar(
                    bins_center,
                    ratio,
                    yerr=ratio_err,
                    fmt=".",
                    color=color_list[i][0],
                )

        if chi2_norm:
            ax.text(
                0.05,
                0.95 - 0.05 * i,
                r"$\chi^2$/ndof= {:.1f},".format(chi2_norm)
                + f"  p-value= {pvalue:.2f}",
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax.transAxes,
                color=color_list[i][0],
                fontsize=20,
            )

        del col_den, col_num

    if "SPREAD" in cats_name:
        # plot the median of the spread
        h_median_spread = np.median(histos_spread, axis=0)
        ax.step(
            bins,
            np.append(h_median_spread, h_median_spread[-1]),
            where="post",
            color=color_list[-2][0],
            linewidth=1,
            label=cat_plot_name_alt,
            zorder=10,
        )
        x0 = bins[0]
        x1 = bins[-1]
        y0 = h_median_spread[0]
        y1 = h_median_spread[-1]
        ax.plot(
            [x0, x0], [ax.get_ylim()[0], y0], color=color_list[-2][0]
        )  # first bin edge
        ax.plot(
            [x1, x1], [ax.get_ylim()[0], y1], color=color_list[-2][0]
        )  # last bin edge

        if len(color_list[-2]) > 1:
            ax.fill_between(
                bins,
                np.append(h_median_spread, h_median_spread[-1]),
                step="post",
                alpha=0.5,
                color=color_list[-2][1],
            )
            
            
        ratio_median = h_num / h_median_spread
        ax_ratio.step(
            bins,
            np.append(ratio_median, ratio_median[-1]),
            where="post",
            color=color_list[-2][0],
            linewidth=1,
        )

        # plot the 16th and 84th percentiles of the spread
        ratio_16 = np.percentile(ratios_spread, 16, axis=0)
        ratio_84 = np.percentile(ratios_spread, 84, axis=0)
        ax_ratio.step(
            bins,
            np.append(ratio_16, ratio_16[-1]),
            where="post",
            color=color_list[-1][0],
            linewidth=1,
            label=r"$\pm \sigma_{k-folds}$",
        )
        ax_ratio.step(
            bins,
            np.append(ratio_84, ratio_84[-1]),
            where="post",
            color=color_list[-1][0],
            linewidth=1,
        )

    ax.legend(loc="upper right")
    ax_ratio.legend(loc="upper left")
    ax.set_yscale("log" if log_scale else "linear")

    hep.cms.lumitext(f"{era_string}, {lumi}" + r" $fb^{-1}$, (13.6 TeV)", ax=ax)
    hep.cms.text(text="Preliminary", ax=ax)

    var_plot_name = var.replace("Run2", "")
    ax_ratio.set_xlabel(var_plot_name)
    ax.set_ylabel("Events")
    ax_ratio.set_ylabel("Data/Pred.")

    ax.grid()
    ax_ratio.grid()
    if "SPREAD" in cat:
        ax_ratio.set_ylim(0.75, 1.25)
    else:
        ax_ratio.set_ylim(0.5, 1.5)
    ax.set_ylim(
        top=(1.3 * ax.get_ylim()[1] if not log_scale else ax.get_ylim()[1] ** 1.3)
    )
    fig.savefig(
        os.path.join(dir_cat, f"{var}.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


def plot_from_columns(cat_col, lumi, era_string):

    print(f"CATEGORIES ARE:")
    print(cat_dict)
    for cats_name, cat_list in cat_dict.items():
        if args.spread:
            if "SPREAD" not in cats_name:
                continue

        if "Run2SPANet" in cats_name:
            chi_squared = False
            color_list = color_list_alt
        elif "SPREAD" in cats_name:
            chi_squared = False
            color_list = color_list_spread
        else:
            chi_squared = True
            color_list = color_list_orig

        # check if the categories are in the accumulator
        try:
            for cat in cat_list:
                if "SPREAD" not in cat:
                    cat_col[cat]
        except KeyError:
            print(f"KeyError: {cat} not in {cat_col.keys()}, skipping {cats_name}")
            continue

        vars_tot = list(cat_col[cat_list[0]].keys())
        if "SPREAD" in cats_name:
            vars_tot = [v for v in vars_tot if "weight" in v or "score" in v]

        dir_cat = f"{outputdir}/{cats_name}_columns"
        if not os.path.exists(dir_cat):
            os.makedirs(dir_cat)

        if args.test:
            vars_tot = vars_tot[:3]
        print("vars_tot", vars_tot)

        vars_to_plot = []

        col_dict = {}
        for v in vars_tot:
            if "_N" in v:
                continue
            v_pref = v.split("_")[0]
            if v_pref + "_N" in vars_tot:
                N = cat_col[cat_list[0]][v_pref + "_N"][0]
                try:
                    assert (cat_col[cat_list[0]][v_pref + "_N"] == N).all()
                except AssertionError:
                    print(
                        f"Variables {v_pref} have different N values: {cat_col[cat_list[0]][v_pref + '_N']}"
                    )
                    sys.exit(1)

                for idx in range(N):
                    col_dict[f"{v}_{idx}"] = {}
                    vars_to_plot.append(f"{v}_{idx}")
                    for cat in cat_list:
                        if "SPREAD" in cat:
                            continue
                        print(v, cat)
                        try:
                            col_dict[f"{v}_{idx}"][cat] = cat_col[cat][v][
                                np.arange(len(cat_col[cat][v])) % N == idx
                            ]
                        except KeyError:
                            col_dict[f"{v}_{idx}"][cat] = cat_col[cat][
                                v.replace("Run2", "")
                            ][
                                np.arange(len(cat_col[cat][v.replace("Run2", "")])) % N
                                == idx
                            ]
            else:
                col_dict[v] = {}
                if "weight" not in v:
                    vars_to_plot.append(v)
                for cat in cat_list:
                    if "SPREAD" in cat:
                        continue
                    # swap the dict keys
                    print(v, cat)
                    try:
                        col_dict[v][cat] = cat_col[cat][v]
                    except KeyError:
                        col_dict[v][cat] = cat_col[cat][v.replace("Run2", "")]

        cat_list_final = cat_list.copy()
        for cat in cat_list:
            if "SPREAD" in cat:
                for i in range(
                    len(
                        col_dict["events_bkg_morphing_spread_dnn_weights"][
                            cat_list_final[0]
                        ][0]
                    )
                ):
                    for v in vars_tot:
                        if "score" in v:
                            col_dict[v][f"{cat}_{i}"] = col_dict[v][cat_list_final[0]]

                    col_dict["weight"][f"{cat}_{i}"] = col_dict[
                        "events_bkg_morphing_spread_dnn_weights"
                    ][cat_list_final[0]][:, i]

                    cat_list_final.append(f"{cat}_{i}")

                    if cat in cat_list_final:
                        # remove the original cat
                        cat_list_final.remove(cat)

        vars_to_plot+=[f"{v}_TRANSFORM" for v in vars_to_plot if "score" in v]

        print("col_dict", col_dict)
        print("cat_list_final", cat_list_final)

        with Pool(args.workers) as p:
            p.starmap(
                plot_single_var_from_columns,
                [
                    (
                        var,
                        col_dict[var.replace("_TRANSFORM", "")],
                        col_dict["weight"],
                        cats_name,
                        cat_list_final,
                        dir_cat,
                        chi_squared,
                        color_list,
                        lumi,
                        era_string,
                    )
                    for var in vars_to_plot
                ],
            )
        del col_dict


if __name__ == "__main__":

    print(cat_col_data)
    lumi, era_string = get_era_lumi(total_datasets_list)

    # plot the weights
    for category in cat_col_data.keys():
        weights = cat_col_data[category]["weight"]
        plot_weights([weights], category, lumi, era_string)

    plot_from_columns(cat_col_data, lumi, era_string)

    print(f"\nPlots saved in {outputdir}")
