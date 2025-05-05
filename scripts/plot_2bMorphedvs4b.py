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

from utils.get_era_lumi import get_era_lumi

hep.style.use("CMS")

NUMBER_OF_BINS = 20
CONSTANT_SIGNAL_BLIND_BINS = np.array(
    [
        1.34472444e-04,
        2.39650306e-01,
        4.05424041e-01,
        5.30443925e-01,
        6.24511921e-01,
        6.93721980e-01,
        7.47704566e-01,
        7.91207588e-01,
        8.27359927e-01,
        8.57444859e-01,
        8.82451415e-01,
        9.03700429e-01,
    ]
)

CONSTANT_SIGNAL_BINS = np.array(
    [
        1.34472444e-04,
        2.39650306e-01,
        4.05424041e-01,
        5.30443925e-01,
        6.24511921e-01,
        6.93721980e-01,
        7.47704566e-01,
        7.91207588e-01,
        8.27359927e-01,
        8.57444859e-01,
        8.82451415e-01,
        9.03700429e-01,
        9.21699631e-01,
        9.36757421e-01,
        9.49532866e-01,
        9.60485041e-01,
        9.69833755e-01,
        9.77849567e-01,
        9.84888554e-01,
        9.91273439e-01,
        9.99923229e-01,
    ]
)


parser = argparse.ArgumentParser(description="Plot 2b morphed vs 4b data")
parser.add_argument(
    "-i",
    "--input",
    type=str,
    required=True,
    help="Input directory with coffea files or coffea file itself",
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
parser.add_argument("-w", "--workers", type=int, default=8, help="Number of workers")
parser.add_argument(
    "-l", "--linear", action="store_true", help="Linear scale", default=False
)
parser.add_argument(
    "-t", "--test", action="store_true", help="Test on one variable", default=False
)
parser.add_argument(
    "-c",
    "--const-bin",
    action="store_true",
    help="Use constant signal binning",
    default=False,
)
args = parser.parse_args()

if args.test:
    args.workers = 1
    args.output = "test"

PAD_VALUE = -999


input_dir = os.path.dirname(args.input)
log_scale = not args.linear
outputdir = (
    os.path.join(input_dir, args.output)
    + f"_{args.normalisation}{'_ConstSigBin' if args.const_bin else ''}"
)
if args.input.endswith(".coffea"):
    inputfiles = [args.input]
else:
    # get list of coffea files
    inputfiles = [
        os.path.join(input_dir, file)
        for file in os.listdir(input_dir)
        if file.endswith(".coffea") and "DATA" in file
    ]


# To mix categories with Run2 and SPANet, put first the Run2 category
# because first the name of the variables is try with the Run2 string
# and after without it
cat_dict = {}
for region_suffix in ["", "_VR1"]:
    cat_dict |= {
        f"CR{region_suffix}": [
            f"4b{region_suffix}_control_region",
            f"2b{region_suffix}_control_region_postW",
            f"2b{region_suffix}_control_region_preW",
        ],
        f"CR{region_suffix}Run2": [
            f"4b{region_suffix}_control_regionRun2",
            f"2b{region_suffix}_control_region_postWRun2",
            f"2b{region_suffix}_control_region_preWRun2",
        ],
        f"SR{region_suffix}": [
            f"4b{region_suffix}_signal_region",
            f"2b{region_suffix}_signal_region_postW",
            f"2b{region_suffix}_signal_region_preW",
        ],
        f"SR{region_suffix}_blind": [
            f"4b{region_suffix}_signal_region_blind",
            f"2b{region_suffix}_signal_region_postW_blind",
            f"2b{region_suffix}_signal_region_preW_blind",
        ],
        f"SR{region_suffix}_blindRun2": [
            f"4b{region_suffix}_signal_region_blindRun2",
            f"2b{region_suffix}_signal_region_postW_blindRun2",
            f"2b{region_suffix}_signal_region_preW_blindRun2",
        ],
        f"SR{region_suffix}Run2": [
            f"4b{region_suffix}_signal_regionRun2",
            f"2b{region_suffix}_signal_region_postWRun2",
            f"2b{region_suffix}_signal_region_preWRun2",
        ],
        # f"CR{region_suffix}_2b_Run2SPANet": [f"2b{region_suffix}_control_region_preWRun2", f"2b{region_suffix}_control_region_preW"],
        # f"CR{region_suffix}_4b_Run2SPANet": [f"4b{region_suffix}_control_regionRun2", f"4b{region_suffix}_control_region"],
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
color_list_alt = [("purple",), ("darkorange", "orange"), ("green",)]


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

    # hep.cms.lumitext(r"22EE Era E, 6 $fb^{-1}$, (13.6 TeV)", ax=ax)
    # hep.cms.lumitext(r"22EE, (13.6 TeV)", ax=ax)
    hep.cms.lumitext(f"{era_string}, {lumi}" + r" $fb^{-1}$, (13.6 TeV)", ax=ax)
    hep.cms.text(text="Preliminary", ax=ax)

    fig.savefig(os.path.join(outputdir, f"weights_{suffix}.png"))
    plt.close(fig)


def plot_single_var_from_columns(
    var,
    col_dict,
    weight_dict,
    cat_list,
    dir_cat,
    chi_squared=True,
    color_list=color_list_orig,
    lumi=5.79,
    era_string="22 E",
):
    fig, (ax, ax_ratio) = plt.subplots(
        2,
        1,
        figsize=[13, 13],
        sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1]},
    )

    print(var)
    # range_4b = (0, 0)

    for i, cat in enumerate(cat_list):

        cat_plot_name = cat.replace("Run2", "_DHH")

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
        if "sig_bkg_dnn" in var or args.const_bin:
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

            # print(f"weights_den {weights_den}", type(weights_den))
            # print(f"weights_num {weights_num}")
            # print(f"col_num {col_num}", type(col_num))
            # print(f"col_den {col_den}")

            bins = np.linspace(range_4b[0], range_4b[1], NUMBER_OF_BINS + 1)

        # print("bins", bins, len(bins))
        bins_center = (bins[1:] + bins[:-1]) / 2
        # print("bins_center", bins_center, len(bins_center))
        idx_den = np.digitize(col_den, bins)
        idx_num = np.digitize(col_num, bins)
        # print("idx_den", idx_den, len(idx_den))
        # print("idx_num", idx_num, len(idx_num))

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
        else:
            ratio_err = np.sqrt(
                (err_num / h_den) ** 2 + (h_num * err_den / h_den**2) ** 2
            )
        # print("ratio_err", ratio_err)

        if i == 0:
            ax.errorbar(
                bins_center,
                h_den,
                yerr=err_den,
                label=cat_plot_name,
                color=color_list[i][0],
                fmt=".",
            )
            ax_ratio.axhline(y=1, color=color_list[i][0], linestyle="--")
            ax_ratio.fill_between(
                bins_center,
                1 - ratio_err,
                1 + ratio_err,
                color="grey",
                alpha=0.5,
            )
        else:

            ax.hist(
                col_den,
                bins=bins,
                histtype="step",
                label=cat_plot_name,
                weights=weights_den,
                edgecolor=color_list[i][0],
                facecolor=color_list[i][1] if len(color_list[i]) > 1 else None,
                fill=True if len(color_list[i]) > 1 else False,
                alpha=0.5,
                # range=range_4b,
            )
            ax_ratio.errorbar(
                bins_center,
                ratio,
                yerr=ratio_err,
                fmt=".",
                label=cat_plot_name,
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

    ax.legend(loc="upper right")
    ax.set_yscale("log" if log_scale else "linear")

    # hep.cms.lumitext(r"22EE Era E, 6 $fb^{-1}$, (13.6 TeV)", ax=ax)
    # hep.cms.lumitext(r"22EE, (13.6 TeV)", ax=ax)
    hep.cms.lumitext(f"{era_string}, {lumi}" + r" $fb^{-1}$, (13.6 TeV)", ax=ax)
    hep.cms.text(text="Preliminary", ax=ax)

    var_plot_name = var.replace("Run2", "")
    ax_ratio.set_xlabel(var_plot_name)
    ax.set_ylabel("Events")
    ax_ratio.set_ylabel("Data/Pred.")

    ax.grid()
    ax_ratio.grid()
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
    print(f"{cat_dict.keys()}")
    for cats_name, cat_list in cat_dict.items():
        if "Run2SPANet" in cats_name:
            chi_squared = False
            color_list = color_list_alt
        else:
            chi_squared = True
            color_list = color_list_orig
        dir_cat = f"{outputdir}/{cats_name}_columns"

        # check if the categories are in the accumulator
        try:
            for cat in cat_list:
                cat_col[cat]
        except KeyError:
            print(f"KeyError: {cat} not in {cat_col.keys()}, skipping {cats_name}")
            continue

        vars_tot = list(cat_col[cat_list[0]].keys())
        if not os.path.exists(dir_cat):
            os.makedirs(dir_cat)

        if args.const_bin:
            vars_tot = [v for v in vars_tot if "sig_bkg_dnn" in v or "weight" in v]
        if args.test:
            vars_tot = vars_tot[:3]
        print("vars_tot", vars_tot)

        vars = []

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
                    vars.append(f"{v}_{idx}")
                    for cat in cat_list:
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
                if v != "weight":
                    vars.append(v)
                for cat in cat_list:
                    # swap the dict keys
                    print(v, cat)
                    try:
                        col_dict[v][cat] = cat_col[cat][v]
                    except KeyError:
                        col_dict[v][cat] = cat_col[cat][v.replace("Run2", "")]
        print(col_dict)

        with Pool(args.workers) as p:
            p.starmap(
                plot_single_var_from_columns,
                [
                    (
                        var,
                        col_dict[var],
                        col_dict["weight"],
                        cat_list,
                        dir_cat,
                        chi_squared,
                        color_list,
                        lumi,
                        era_string,
                    )
                    for var in vars
                ],
            )
        del col_dict


if __name__ == "__main__":

    cat_col = {}
    # for cat_dict_value in cat_dict.values():
    #     cat_col |= {cat: {} for cat in cat_dict_value}

    # print("cat_col", cat_col)

    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    total_datasets_list = []

    # get the columns
    for inputfile in inputfiles:
        accumulator = load(inputfile)
        samples = list(accumulator["columns"].keys())
        print(f"inputfile {inputfile}")
        for sample in samples:
            print(f"sample {sample}")
            datasets = list(accumulator["columns"][sample].keys())
            for dataset in datasets:
                if dataset not in total_datasets_list:
                    total_datasets_list.append(dataset)
                print(f"dataset {dataset}")
                categories = list(accumulator["columns"][sample][dataset].keys())
                for category in categories:
                    print(f"category {category}")
                    if category not in cat_col:
                        cat_col[category] = {}
                    columns = list(
                        accumulator["columns"][sample][dataset][category].keys()
                    )
                    for i, column in enumerate(columns):
                        column_array = accumulator["columns"][sample][dataset][
                            category
                        ][column].value
                        if column not in cat_col[category]:
                            cat_col[category][column] = column_array
                        else:
                            cat_col[category][column] = np.concatenate(
                                (cat_col[category][column], column_array)
                            )

                        if i == 0:
                            print(
                                f"column {column}",
                                column_array.shape,
                                cat_col[category][column].shape,
                            )

    print(cat_col)
    lumi, era_string = get_era_lumi(total_datasets_list)

    # plot the weights
    for category in cat_col.keys():
        weights = cat_col[category]["weight"]
        plot_weights([weights], category, lumi, era_string)

    plot_from_columns(cat_col, lumi, era_string)

    print(f"\nPlots saved in {outputdir}")
