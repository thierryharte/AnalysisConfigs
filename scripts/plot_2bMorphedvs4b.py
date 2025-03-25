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
    "-d", "--density", action="store_true", help="Normalize plots to 1", default=False
)
parser.add_argument(
    "-t", "--test", action="store_true", help="Test on one variable", default=False
)
args = parser.parse_args()

if args.test:
    args.workers = 1
    args.output = "test"

NORMALIZE_WEIGHTS = False
PAD_VALUE = -999


inputfile = args.input
input_dir = os.path.dirname(args.input)
cfg = os.path.join(input_dir, "parameters_dump.yaml")
log_scale = not args.linear
outputdir = os.path.join(input_dir, args.output) + f"_{args.normalisation}"


# To mix categories with Run2 and SPANet, put first the Run2 category
# because first the name of the variables is try with the Run2 string
# and after without it
cat_dict = {
    "CR": ["4b_control_region", "2b_control_region_postW", "2b_control_region_preW"],
    "CRRun2": [
        "4b_control_regionRun2",
        "2b_control_region_postWRun2",
        "2b_control_region_preWRun2",
    ],
    "SR": ["4b_signal_region", "2b_signal_region_postW", "2b_signal_region_preW"],
    "SRRun2": [
        "4b_signal_regionRun2",
        "2b_signal_region_postWRun2",
        "2b_signal_region_preWRun2",
    ],
    "CR_2b_Run2SPANet": ["2b_control_region_preWRun2", "2b_control_region_preW"],
    "CR_4b_Run2SPANet": ["4b_control_regionRun2", "4b_control_region"],
}

if args.test:
    cat_dict = {
        "CRRun2": [
            "4b_control_regionRun2",
            "2b_control_region_postWRun2",
            "2b_control_region_preWRun2",
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
                color=color_list_orig[i][0],
                fmt=".",
            )
        else:
            ax.step(
                h.axes[0].edges,
                np.append(h_den.values(), h_den.values()[-1]),
                where="post",
                label=cat,
                color=color_list_orig[i][0],
            )

        if "4b" not in cat:
            ax_ratio.errorbar(
                h.axes[0].centers,
                h_ratio,
                yerr=ratio_err,
                fmt=".",
                label=cat,
                color=color_list_orig[i][0],
            )
        else:
            ax_ratio.axhline(y=1, color=color_list_orig[i][0], linestyle="--")
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

    # hep.cms.lumitext("(13.6 TeV)", ax=ax)
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
    var,
    col_dict,
    weight_dict,
    cat_list,
    dir_cat,
    norm_factor_dict=None,
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

        # renormalize the weights
        if NORMALIZE_WEIGHTS and "Run2" not in cat and i != 0 and "postW" in cat:
            mean_weight = np.mean(weights_den)
            # mean_weight = np.std(weights)
            # std_weight = np.std(weights)
            std_weight = np.mean(weights_den)
            # std_weight = 1.

            print(
                f"Normalizing weights for {cat} with mean {mean_weight} and std {std_weight}"
            )
            original_weights = weights_den
            weights_den = (weights_den - mean_weight) / std_weight + 1.0
            print(f"New mean: {np.mean(weights_den)} and std: {np.std(weights_den)}")
            if not weights_plotted:
                plot_weights([original_weights, weights_den], f"{cat}_normalized")
                weights_plotted = True

        # mask_w = weights > -1
        # weights = weights[mask_w]

        # remove padded values
        weights_den = weights_den[col_den != PAD_VALUE]
        weights_num = weights_num[col_num != PAD_VALUE]
        col_den = col_den[col_den != PAD_VALUE]
        col_num = col_num[col_num != PAD_VALUE]

        if norm_factor_dict:
            norm_factor_den = norm_factor_dict[cat]
            norm_factor_num = norm_factor_dict[cat_list[0]]
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

        print(f"weights_den {weights_den}", type(weights_den))
        print(f"weights_num {weights_num}")
        print(f"col_num {col_num}", type(col_num))
        print(f"col_den {col_den}")

        # h_den, bins = np.histogram(
        #     col_den, bins=30, range=range_4b
        # )
        # # draw the ratio
        # h_num, _ = np.histogram(
        #     col_num, bins=bins, range=range_4b
        # )

        bins = np.linspace(range_4b[0], range_4b[1], 31)
        print("bins", bins, len(bins))
        bins_center = (bins[1:] + bins[:-1]) / 2
        print("bins_center", bins_center, len(bins_center))
        idx_den = np.digitize(col_den, bins)
        idx_num = np.digitize(col_num, bins)
        print("idx_den", idx_den, len(idx_den))
        print("idx_num", idx_num, len(idx_num))

        h_den = []
        h_num = []
        err_den = []
        err_num = []

        for j in range(1, len(bins)):
            h_den.append(np.sum(weights_den[idx_den == j]))
            h_num.append(np.sum(weights_num[idx_num == j]))
            err_den.append(np.sqrt(np.sum(weights_den[idx_den == j] ** 2)))
            err_num.append(np.sqrt(np.sum(weights_num[idx_num == j] ** 2)))
            print('weights_den[idx_den == j]', weights_den[idx_den == j])

        h_den = np.array(h_den)
        h_num = np.array(h_num)
        err_den = np.array(err_den)
        err_num = np.array(err_num)

        print("h_den", h_den, len(h_den))
        print("h_num", h_num, len(h_num))
        print("err_den", err_den)
        print("err_num", err_num)

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
        print("ratio_err", ratio_err)

        if args.density:
            h_den, bins = np.histogram(
                col_den,
                bins=30,
                weights=weights_den,
                range=range_4b,
                density=True,
            )
            h_num, bins = np.histogram(
                col_num,
                bins=30,
                weights=weights_num,
                range=range_4b,
                density=True,
            )

        if i == 0:
            ax.errorbar(
                bins_center,
                h_den,
                yerr=err_den if not args.density else 0,
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
                density=args.density,
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
            else ax.get_ylim()[1] ** (1.3 if not args.density else -1.3)
        )
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
        if "Run2SPANet" in cats_name:
            chi_squared = False
            color_list = color_list_alt
        else:
            chi_squared = True
            color_list = color_list_orig
        dir_cat = f"{outputdir}/{cats_name}_columns"
        if not os.path.exists(dir_cat):
            os.makedirs(dir_cat)
        vars_tot = list(col_cat[cat_list[0]].keys())
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
                        norm_factor_dict,
                        chi_squared,
                        color_list,
                    )
                    for var in vars
                ],
            )
        del col_dict


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

    # plot the weights
    sample = list(accumulator["columns"].keys())[0]
    dataset = list(accumulator["columns"][sample].keys())[0]
    for category in accumulator["columns"][sample][dataset].keys():
        weights = accumulator["columns"][sample][dataset][category]["weight"].value
        plot_weights([weights], category)

    if args.normalisation == "sum_weights":
        norm_factor_dict = None
    elif args.normalisation == "num_events":
        num_ev_dict = {}
        # Get the normalization factors
        num_ev_dict["num_4b_CR"] = accumulator["cutflow"]["4b_control_region"][
            "DATA_JetMET_JMENano_2022_postEE_EraE"
        ]["DATA_JetMET_JMENano_skimmed"]
        num_ev_dict["num_2b_CR"] = accumulator["cutflow"]["2b_control_region_preW"][
            "DATA_JetMET_JMENano_2022_postEE_EraE"
        ]["DATA_JetMET_JMENano_skimmed"]
        num_ev_dict["num_4b_SR"] = accumulator["cutflow"]["4b_signal_region"][
            "DATA_JetMET_JMENano_2022_postEE_EraE"
        ]["DATA_JetMET_JMENano_skimmed"]
        num_ev_dict["num_2b_SR"] = accumulator["cutflow"]["2b_signal_region_preW"][
            "DATA_JetMET_JMENano_2022_postEE_EraE"
        ]["DATA_JetMET_JMENano_skimmed"]
        num_ev_dict["num_4b_CRRun2"] = accumulator["cutflow"]["4b_control_regionRun2"][
            "DATA_JetMET_JMENano_2022_postEE_EraE"
        ]["DATA_JetMET_JMENano_skimmed"]
        num_ev_dict["num_2b_CRRun2"] = accumulator["cutflow"][
            "2b_control_region_preWRun2"
        ]["DATA_JetMET_JMENano_2022_postEE_EraE"]["DATA_JetMET_JMENano_skimmed"]
        num_ev_dict["num_4b_SRRun2"] = accumulator["cutflow"]["4b_signal_regionRun2"][
            "DATA_JetMET_JMENano_2022_postEE_EraE"
        ]["DATA_JetMET_JMENano_skimmed"]
        num_ev_dict["num_2b_SRRun2"] = accumulator["cutflow"][
            "2b_signal_region_preWRun2"
        ]["DATA_JetMET_JMENano_2022_postEE_EraE"]["DATA_JetMET_JMENano_skimmed"]

        print("num_ev_dict", num_ev_dict)

        norm_factor_dict = {
            "4b_control_region": 1,
            "2b_control_region_preW": num_ev_dict["num_4b_CR"]
            / num_ev_dict["num_2b_CR"],
            "2b_control_region_postW": num_ev_dict["num_4b_CR"]
            / num_ev_dict["num_2b_CR"],
            "4b_signal_region": 1,
            "2b_signal_region_preW": num_ev_dict["num_4b_CR"]
            / (num_ev_dict["num_2b_CR"]),
            "2b_signal_region_postW": num_ev_dict["num_4b_CR"]
            / (num_ev_dict["num_2b_CR"]),
            "4b_control_regionRun2": 1,
            "2b_control_region_preWRun2": num_ev_dict["num_4b_CRRun2"]
            / num_ev_dict["num_2b_CRRun2"],
            "2b_control_region_postWRun2": num_ev_dict["num_4b_CRRun2"]
            / num_ev_dict["num_2b_CRRun2"],
            "4b_signal_regionRun2": 1,
            "2b_signal_region_preWRun2": num_ev_dict["num_4b_CRRun2"]
            / (num_ev_dict["num_2b_CRRun2"]),
            "2b_signal_region_postWRun2": num_ev_dict["num_4b_CRRun2"]
            / (num_ev_dict["num_2b_CRRun2"]),
        }
    else:
        raise ValueError(f"Normalisation type {args.normalisation} not recognised")

    print("norm_factor_dict", norm_factor_dict)

    # plot_from_hist(accumulator, norm_factor_dict)
    plot_from_columns(accumulator, norm_factor_dict)

    print(f"\nPlots saved in {outputdir}")
