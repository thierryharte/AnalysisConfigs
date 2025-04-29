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

import matplotlib
matplotlib.rcParams['agg.path.chunksize'] = 10000  # or try 5000, depending on size

parser = argparse.ArgumentParser(description="Plot 2b morphed vs 4b data")
parser.add_argument("-id", "--input-data", type=str, required=True, help="Input coffea file data")
parser.add_argument("-im", "--input-mc", type=str, required=True, help="Input coffea file monte carlo")
parser.add_argument(
    "-o", "--output", type=str, help="Output directory", default="plots_DNN_data_and_mc"
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
    "-d", "--density", action="store_true", help="Normalize plots to 1", default=False
)
parser.add_argument(
    "-t", "--test", action="store_true", help="Test on one variable", default=False
)
parser.add_argument(
    "-r2", "--run2", action="store_true", help="Also run run2 regions", default=False
)
args = parser.parse_args()

if args.test:
    args.workers = 1
    args.output = "test"

NORMALIZE_WEIGHTS = False
PAD_VALUE = -999


inputfile_data = args.input_data
input_dir_data = os.path.dirname(args.input_data)

inputfile_mc = args.input_mc
input_dir_mc = os.path.dirname(args.input_mc)


log_scale = not args.linear
outputdir = os.path.join(input_dir_data, args.output) + f"_{args.normalisation}"


# To mix categories with Run2 and SPANet, put first the Run2 category
# because first the name of the variables is try with the Run2 string
# and after without it
cat_dict = {
    f"CR{args.region_suffix}": [f"4b{args.region_suffix}_control_region", f"2b{args.region_suffix}_control_region_postW", 
        #f"2b{args.region_suffix}_control_region_preW"
        ],
    f"SR{args.region_suffix}_blinded": [f"4b{args.region_suffix}_signal_region_blinded", f"2b{args.region_suffix}_signal_region_postW_blinded", 
        #f"2b{args.region_suffix}_signal_region_preW"
        ],
    #    f"CR{args.region_suffix}_2b_Run2SPANet": [f"2b{args.region_suffix}_control_region_preWRun2", f"2b{args.region_suffix}_control_region_preW"],
    #    f"CR{args.region_suffix}_4b_Run2SPANet": [f"4b{args.region_suffix}_control_regionRun2", f"4b{args.region_suffix}_control_region"],
}
if args.run2:
    cat_dict[f"CR{args.region_suffix}Run2"] = [f"4b{args.region_suffix}_control_regionRun2", f"2b{args.region_suffix}_control_region_postWRun2"]
    cat_dict[f"SR{args.region_suffix}_blindedRun2"] = [f"4b{args.region_suffix}_signal_region_blindedRun2", f"2b{args.region_suffix}_signal_region_postW_blindedRun2"]

if args.test:
    cat_dict = {
        f"CR{args.region_suffix}Run2": [
            f"4b{args.region_suffix}_control_regionRun2",
            f"2b{args.region_suffix}_control_region_postWRun2",
            #  f"2b{args.region_suffix}_control_region_preWRun2",
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
    

    plotdict = {}
    for i, cat in enumerate(cat_list):
        # I only want the following columns:
        # postW data
        # signal data
        # signal MC
        cat_plot_name=cat.replace("Run2", "_DHH")
        print(cat_plot_name)
        for data_mc in ["mc", "data"]:
            # we dont need the reweighted MC region
            if data_mc=="mc" and "postW" in cat_plot_name:
                continue

            weights_den = weight_dict[cat][data_mc]
            weights_num = weight_dict[cat_list[0]][data_mc]

            col_den = col_dict[cat][data_mc]
            col_num = col_dict[cat_list[0]][data_mc]

            # remove padded values
            weights_den = weights_den[col_den != PAD_VALUE]
            weights_num = weights_num[col_num != PAD_VALUE]
            col_den = col_den[col_den != PAD_VALUE]
            col_num = col_num[col_num != PAD_VALUE]

            # the range for the score can be inclusively all the range.
            # I still have to define it this way to make up for different ranges of blinded histograms.
            # range_4b is basically a global variable. It is calculated only once.
            # Same as bin_edges. Both depend on the MC signal
            if i==0 and data_mc=="mc":
                range_4b = tuple(np.quantile(col_den, [0, 1]))
                nbins = 30
                bin_edges = np.quantile(col_den,np.linspace(min(col_den),max(col_den), nbins+1))
                print(f"range_4b {range_4b}")

            #mask_num_range4b = (col_num > range_4b[0]) & (col_num < range_4b[1])
            #weights_num = weights_num[mask_num_range4b]
            #col_num = col_num[mask_num_range4b]

            #mask_den_range4b = (col_den > range_4b[0]) & (col_den < range_4b[1])
            #weights_den = weights_den[mask_den_range4b]
            #col_den = col_den[mask_den_range4b]

            norm_factor_den = weights_num.sum() / weights_den.sum()
            norm_factor_num = 1.0
            print(
                f"Plotting from columns {var} for {cat} with norm {norm_factor_den} and weights sum {weights_den.sum()}"
            )

            # normalize the weights
            weights_den = weights_den * norm_factor_den
            weights_num = weights_num * norm_factor_num

            print(f"weights_den {weights_den}", type(weights_den))
            print(f"weights_num {weights_num}")
            print(f"col_num {col_num}", type(col_num))
            print(f"col_den {col_den}")

            print("bin_edges", bin_edges, len(bin_edges))
            bins_center = (bin_edges[1:] + bin_edges[:-1]) / 2
            print("bins_center", bins_center, len(bins_center))

            print(f"{cat_plot_name}_{data_mc}")
            if i==0 and data_mc == "mc":
                print("Found MC signal")
                #this is the mc data that I want to add to the histogram of the reweighted data
                mc_signal = f"{cat_plot_name}_{data_mc}"

            print("Found something to plot")
            # Here we save the MC and the bg reweighted
            # The signal was already plotted and we don't need it anymore
            plotdict[f"{cat_plot_name}_{data_mc}"] = {
            "bin_edges": bin_edges,
            "bins_center": bins_center,
            "color": color_list[i],
            "col_den": col_den,
            "col_num": col_num,
            "weights_den": weights_den,
            "weights_num": weights_num,
            }
            del col_den, col_num

    print(plotdict)
    for region, values in plotdict.items():
        print(f"Plotting region {region}")
        #Trying to add up the reweighted data from 2b with the MC signal
        #MC is still plotted independently.

        # Essentially the idea is:
        # -> events are appended with np.concatenate()
        # -> histograms are added binwise with +
        if "postW" in region:
            values["col_den"] = np.concatenate((values["col_den"], plotdict[mc_signal]["col_den"]))
            values["weights_den"] = np.concatenate((values["weights_den"], plotdict[mc_signal]["weights_den"]))

        # Filling the histograms
        idx_den = np.digitize(values["col_den"], values["bin_edges"])
        idx_num = np.digitize(values["col_num"], values["bin_edges"])
        print(f"bin_edges {values['bin_edges']}")
        print(len(values["col_den"]))
        print(len(values["weights_den"]))
        print("idx_den", idx_den, len(idx_den))
        print("idx_num", idx_num, len(idx_num))

        values["h_den"] = []
        values["h_num"] = []
        values["err_den"] = []
        values["err_num"] = []

        for j in range(1, len(values["bin_edges"])):
            values["h_den"].append(np.sum(values["weights_den"][idx_den == j]))
            values["h_num"].append(np.sum(values["weights_num"][idx_num == j]))
            values["err_den"].append(np.sqrt(np.sum(values["weights_den"][idx_den == j] ** 2)))
            values["err_num"].append(np.sqrt(np.sum(values["weights_num"][idx_num == j] ** 2)))
            print('values["weights_den"][idx_den == j]', values["weights_den"][idx_den == j])

        values["h_den"] = np.array(values["h_den"])
        values["h_num"] = np.array(values["h_num"])
        values["err_den"] = np.array(values["err_den"]) if not args.density else 0
        values["err_num"] = np.array(values["err_num"])

        print("h_den", values["h_den"], len(values["h_den"]))
        print("h_num", values["h_num"], len(values["h_num"]))
        print("err_den", values["err_den"])
        print("err_num", values["err_num"])

        # I don't fully get this part actually, but copied as it is from the other plotting script
        if args.density:
            values["h_den"], values["bin_edges"] = np.histogram(
                col_den,
                bins=nbins,
                weights=weights_den,
                range=range_4b,
                density=True,
            )
        
        ratio = values["h_num"]/values["h_den"]
        ratio_err =  np.sqrt((values["err_num"] / values["h_den"]) ** 2 + (values["h_num"] * values["err_den"] / values["h_den"]**2) ** 2)
        print("These are ratio and ratio error")
        print(ratio[0])
        print(ratio_err)
        print(values["bin_edges"])
        
        # Reference dataset (data in 4b)
        if not "postW" in region and "data" in region:
            print("Found signal region data")
            ratio = values["h_num"] / values["h_den"]
            ratio_err = values["err_num"] / values["h_num"]
            print("ratio_err", ratio_err)

            ax.errorbar(
                values["bins_center"],
                values["h_den"],
                yerr=values["err_den"] if not args.density else 0,
                label=region,
                color=values["color"][0],
                fmt=".",
            )
            ax_ratio.axhline(y=1, color=values["color"][0], linestyle="--")
            ax_ratio.fill_between(
                values["bins_center"],
                1 - ratio_err,
                1 + ratio_err,
                color="grey",
                alpha=0.5,
            )
        else:
            chi2_norm = None
            if "postW" in region and chi_squared:
                # compute the chi square between the two histograms (divide by the error on data)
                chi2_value = np.sum(
                    ((values["h_den"] - values["h_num"]) / np.where(values["err_num"] == 0, 1, values["err_num"])) ** 2
                )
                ndof = len(values["h_den"]) - 1
                chi2_norm = chi2_value / ndof
                pvalue = chi2.sf(chi2_value, ndof)

                ax.text(
                    0.05,
                    0.95 - 0.05,
                    r"$\chi^2$/ndof= {:.1f},".format(chi2_norm)
                    + f"  p-value= {pvalue:.2f}",
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=ax.transAxes,
                    color=values["color"][0],
                    fontsize=20,
                )


            ax.hist(
                values["col_den"],
                bins=bin_edges,
                histtype="step",
                label=region,
                weights=values["weights_den"],
                edgecolor=values["color"][0],
                facecolor=values["color"][1] if len(values["color"]) > 1 else None,
                fill=True if len(values["color"]) > 1 else False,
                alpha=0.5,
                range=range_4b,
                density=args.density,
            )
            ax_ratio.errorbar(
                values["bins_center"],
                ratio,
                yerr=ratio_err,
                fmt=".",
                label=region,
                color=values["color"][0],
            )
    del plotdict

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
    print(f"Plotname: {os.path.join(dir_cat, f'{var}.png')}")
    fig.savefig(
        os.path.join(dir_cat, f"{var}.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


def plot_from_columns(col_cats, genweight):
    
    # cat_dict defined on top (global variable)
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
        # From here on we have to extract for signal an MC:
        # :param: data_mc means that we make the dictionary one longer such that for each category we save the data and the MC values.
        col_dict = {}
        for data_mc, col_cat in zip(["data", "mc"], col_cats):
            print(data_mc)
            print(col_cat.keys())
            vars_tot = list(col_cat[cat_list[0]].keys())
            if args.test:
                vars_tot = vars_tot[:3]
            #print("vars_tot", vars_tot)
            vars = []
            # vars_tot = [v for v in vars_tot if "add" in v or "weight"  in v]
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
                        if not f"{v}_{idx}" in col_dict.keys():
                            col_dict[f"{v}_{idx}"] = {}
                        vars.append(f"{v}_{idx}")
                        for cat in cat_list:
                            if not cat in col_dict[f"{v}_{idx}"].keys():
                                col_dict[f"{v}_{idx}"][cat] = {}
                            try:
                                col_dict[f"{v}_{idx}"][cat][data_mc] = col_cat[cat][v].value[
                                    np.arange(len(col_cat[cat][v].value)) % N == idx
                                ]
                            except KeyError:
                                col_dict[f"{v}_{idx}"][cat][data_mc] = col_cat[cat][
                                    v.replace("Run2", "")
                                ].value[
                                    np.arange(
                                        len(col_cat[cat][v.replace("Run2", "")].value)
                                    )
                                    % N
                                    == idx
                                ]
                else:
                    if not v in col_dict.keys():
                        col_dict[v] = {}
                    if v != "weight":
                        vars.append(v)
                    for cat in cat_list:
                        if not cat in col_dict[v].keys():
                            col_dict[v][cat]={}
                        try:
                            col_dict[v][cat][data_mc] = col_cat[cat][v].value 
                        except KeyError:
                            col_dict[v][cat][data_mc] = col_cat[cat][v.replace("Run2", "")].value
                        if v == "weight":
                            col_dict[v][cat][data_mc] = col_dict[v][cat][data_mc] / (genweight if data_mc == "mc" else 1)
        print(col_dict)
        print(vars)

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
                    for var in vars if "score" in var
                ],
            )
        del col_dict


if __name__ == "__main__":
    
    ## For data:
    
    print(f"InputFile: {inputfile_data}")
    if os.path.isfile(inputfile_data):
        accumulator_data = load(inputfile_data)
    else:
        sys.exit(f"Input file '{inputfile_data}' does not exist")
    
    ## For MC:
    
    print(f"InputFile: {inputfile_mc}")
    if os.path.isfile(inputfile_mc):
        accumulator_mc = load(inputfile_mc)
    else:
        sys.exit(f"Input file '{inputfile_mc}' does not exist")

    ## Finished loading files

    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    sample_data = "DATA_JetMET_JMENano_E_skimmed"
    dataset_data = "DATA_JetMET_JMENano_E_2022_postEE_EraE"
    sample_mc = "GluGlutoHHto4B_spanet"
    dataset_mc = "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_spanet__2022_postEE"
    
    print(accumulator_data["columns"].keys())
    print(accumulator_mc["columns"].keys())
    assert sample_data in list(accumulator_data["columns"].keys())
    assert sample_mc in list(accumulator_mc["columns"].keys())
    
    print(accumulator_data["columns"][sample_data].keys())
    print(accumulator_mc["columns"][sample_mc].keys())
    assert dataset_data in list(accumulator_data["columns"][sample_data].keys())
    assert dataset_mc in list(accumulator_mc["columns"][sample_mc].keys())
   
    col_cat_data = accumulator_data["columns"][sample_data][dataset_data]
    col_cat_mc = accumulator_mc["columns"][sample_mc][dataset_mc]
    
    print(accumulator_mc["sum_genweights"][dataset_mc])
    ############# Actual plotting command. Now a list with [datastuff, mcstuff] ######################33
    plot_from_columns([col_cat_data,col_cat_mc],accumulator_mc["sum_genweights"][dataset_mc])

    print(f"\nPlots saved in {outputdir}")
