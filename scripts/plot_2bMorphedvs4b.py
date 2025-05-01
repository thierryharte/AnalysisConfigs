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

hep.style.use("CMS")


parser = argparse.ArgumentParser(description="Plot 2b morphed vs 4b data")
parser.add_argument("-i", "--input", type=str, required=True, help="Input coffea file")
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
args = parser.parse_args()

if args.test:
    args.workers = 1
    args.output = "test"

PAD_VALUE = -999


inputfile = args.input
input_dir = os.path.dirname(args.input)
log_scale = not args.linear
outputdir = os.path.join(input_dir, args.output) + f"_{args.normalisation}"

# To mix categories with Run2 and SPANet, put first the Run2 category
# because first the name of the variables is try with the Run2 string
# and after without it
cat_dict={}
for region_suffix in ["", "_VR1"]:
    cat_dict |= {
        f"CR{region_suffix}": [f"4b{region_suffix}_control_region", f"2b{region_suffix}_control_region_postW", f"2b{region_suffix}_control_region_preW"],
        f"CR{region_suffix}Run2": [
            f"4b{region_suffix}_control_regionRun2",
            f"2b{region_suffix}_control_region_postWRun2",
            f"2b{region_suffix}_control_region_preWRun2",
        ],
        f"SR{region_suffix}": [f"4b{region_suffix}_signal_region", f"2b{region_suffix}_signal_region_postW", f"2b{region_suffix}_signal_region_preW"],
        f"SR{region_suffix}_blind": [f"4b{region_suffix}_signal_region_blind", f"2b{region_suffix}_signal_region_postW_blind", f"2b{region_suffix}_signal_region_preW_blind"],
        f"SR{region_suffix}_blindRun2": [f"4b{region_suffix}_signal_region_blindRun2", f"2b{region_suffix}_signal_region_postW_blindRun2", f"2b{region_suffix}_signal_region_preW_blindRun2"],
        f"SR{region_suffix}Run2": [
            f"4b{region_suffix}_signal_regionRun2",
            f"2b{region_suffix}_signal_region_postWRun2",
            f"2b{region_suffix}_signal_region_preWRun2",
        ],
        #f"CR{region_suffix}_2b_Run2SPANet": [f"2b{region_suffix}_control_region_preWRun2", f"2b{region_suffix}_control_region_preW"],
        #f"CR{region_suffix}_4b_Run2SPANet": [f"4b{region_suffix}_control_regionRun2", f"4b{region_suffix}_control_region"],
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


def plot_weights(weights_list, suffix):
    fig, ax = plt.subplots(figsize=[13, 13])
    for i, weights in enumerate(weights_list):
        ax.hist(
            weights,
            bins=np.logspace(-3, 2, 100),
            histtype="step",
            label="Morphing weights "
            + (f"{i}" if len(weights_list)>1 else "")
            + "\nmean: {:.2f}\nstd: {:.2f}".format(np.mean(weights), np.std(weights)),
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.set_xlabel("Morphing weights")
    ax.set_ylabel("Events")
    
    hep.cms.lumitext(r"22EE Era E, 6 $fb^{-1}$, (13.6 TeV)", ax=ax)
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
):
    fig, (ax, ax_ratio) = plt.subplots(
        2,
        1,
        figsize=[13, 13],
        sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1]},
    )
    weights_plotted = False
    print(var)
    range_4b = (0, 0)

    for i, cat in enumerate(cat_list):
        
        cat_plot_name=cat.replace("Run2", "_DHH")

        weights_den = weight_dict[cat]
        weights_num = weight_dict[cat_list[0]]

        col_den = col_dict[cat]
        col_num = col_dict[cat_list[0]]


        # remove padded values
        weights_den = weights_den[col_den != PAD_VALUE]
        weights_num = weights_num[col_num != PAD_VALUE]
        col_den = col_den[col_den != PAD_VALUE]
        col_num = col_num[col_num != PAD_VALUE]

        if args.normalisation == "num_events":
            norm_factor_den = len(weights_den) / len(weights_num)
            norm_factor_num = 1.0
        else:
            norm_factor_den = weights_num.sum() / weights_den.sum()
            norm_factor_num = 1.0
        print(
            f"Plotting from columns {var} for {cat} with norm {norm_factor_den} and weights sum {weights_den.sum()}"
        )

        # compute the range of the 4b category considering the 0.1% and 99.9% quantile
        range_4b = tuple(np.quantile(col_den, [0.001, 0.999])) if i == 0 else range_4b

        print(f"range_4b {range_4b}")

        mask_num_range4b = (col_num > range_4b[0]) & (col_num < range_4b[1])
        weights_num = weights_num[mask_num_range4b]
        col_num = col_num[mask_num_range4b]

        mask_den_range4b = (col_den > range_4b[0]) & (col_den < range_4b[1])
        weights_den = weights_den[mask_den_range4b]
        col_den = col_den[mask_den_range4b]

        # normalize the weights
        weights_den = weights_den * norm_factor_den
        weights_num = weights_num * norm_factor_num

        # print(f"weights_den {weights_den}", type(weights_den))
        # print(f"weights_num {weights_num}")
        # print(f"col_num {col_num}", type(col_num))
        # print(f"col_den {col_den}")


        bins = np.linspace(range_4b[0], range_4b[1], 31)
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
            chi2_value = np.sum(
                ((h_den - h_num) / np.where(err_num == 0, 1, err_num)) ** 2
            )
            ndof = len(h_den) - 1
            chi2_norm = chi2_value / ndof
            pvalue = chi2.sf(chi2_value, ndof)

        ratio = h_num / h_den

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
                bins=30,
                histtype="step",
                label=cat_plot_name,
                weights=weights_den,
                edgecolor=color_list[i][0],
                facecolor=color_list[i][1] if len(color_list[i]) > 1 else None,
                fill=True if len(color_list[i]) > 1 else False,
                alpha=0.5,
                range=range_4b,
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
    
    # hep.cms.lumitext(r"2022 (13.6 TeV)", ax=ax)
    hep.cms.lumitext(r"22EE Era E, 6 $fb^{-1}$, (13.6 TeV)", ax=ax)
    hep.cms.text(text="Preliminary", ax=ax)

    var_plot_name = var.replace("Run2", "")
    ax_ratio.set_xlabel(var_plot_name)
    ax.set_ylabel("Events")
    ax_ratio.set_ylabel("Data/Pred.")

    ax.grid()
    ax_ratio.grid()
    ax_ratio.set_ylim(0.5, 1.5)
    ax.set_ylim(
        top=(
            1.3 * ax.get_ylim()[1]
            if not log_scale
            else ax.get_ylim()[1] ** 1.3
        )
    )
    fig.savefig(
        os.path.join(dir_cat, f"{var}.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


def plot_from_columns(accumulator):
    col_cat = accumulator["columns"][sample][dataset]

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
                col_cat[cat]
        except KeyError:
            print(f"KeyError: {cat} not in {col_cat.keys()}, skipping {cats_name}")
            continue
        
        vars_tot = list(col_cat[cat_list[0]].keys())
        if not os.path.exists(dir_cat):
            os.makedirs(dir_cat)
        if args.test:
            vars_tot = vars_tot[:3]
        print("vars_tot", vars_tot)
        vars = []
        # vars_tot = [v for v in vars_tot if "add" in v or "weight"  in v]
        col_dict = {}
        for v in vars_tot:
            if "_N" in v:
                continue
            v_pref = v.split("_")[0]
            if v_pref + "_N" in vars_tot:
                N = col_cat[cat_list[0]][v_pref + "_N"].value[0]
                try:
                    assert (col_cat[cat_list[0]][v_pref + "_N"].value == N).all()
                except AssertionError:
                    print(
                        f"Variables {v_pref} have different N values: {col_cat[cat_list[0]][v_pref + '_N'].value}"
                    )
                    sys.exit(1)

                for idx in range(N):
                    col_dict[f"{v}_{idx}"] = {}
                    vars.append(f"{v}_{idx}")
                    for cat in cat_list:
                        print(v, cat)
                        try:
                            col_dict[f"{v}_{idx}"][cat] = col_cat[cat][v].value[
                                np.arange(len(col_cat[cat][v].value)) % N == idx
                            ]
                        except KeyError:
                            col_dict[f"{v}_{idx}"][cat] = col_cat[cat][
                                v.replace("Run2", "")
                            ].value[
                                np.arange(
                                    len(col_cat[cat][v.replace("Run2", "")].value)
                                )
                                % N
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
                        col_dict[v][cat] = col_cat[cat][v].value
                    except KeyError:
                        col_dict[v][cat] = col_cat[cat][v.replace("Run2", "")].value
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
                    )
                    for var in vars
                ],
            )
        del col_dict


if __name__ == "__main__":


    if os.path.isfile(inputfile):
        accumulator = load(inputfile)
    else:
        sys.exit(f"Input file '{inputfile}' does not exist")

    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    # plot the weights
    sample = list(accumulator["columns"].keys())[0]
    dataset = list(accumulator["columns"][sample].keys())[0]
    for category in accumulator["columns"][sample][dataset].keys():
        weights = accumulator["columns"][sample][dataset][category]["weight"].value
        plot_weights([weights], category)

    plot_from_columns(accumulator)

    print(f"\nPlots saved in {outputdir}")
