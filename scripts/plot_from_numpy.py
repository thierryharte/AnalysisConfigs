'''
Very small script to test loading the saved numpy arrays for plotting of different datasets (DHH and SPANet)
'''
import numpy as np
import os
import sys
from matplotlib import pyplot as plt
import mplhep as hep
import argparse
from pathlib import Path
import glob
import logging

hep.style.use("CMS")

import matplotlib

matplotlib.rcParams["agg.path.chunksize"] = 10000  # or try 5000, depending on size



parser = argparse.ArgumentParser(description="Plot 2b morphed vs 4b data")
parser.add_argument(
    "-i",
    "--input-data",
    type=str,
    required=True,
    nargs="+",
    help="Input directories or files to load (format npz)",
)
parser.add_argument(
    "-o", "--output", type=str, help="Output directory", default="plots_from_onnx"
)
parser.add_argument(
    "-l", "--linear", action="store_true", help="Linear scale", default=False
)
args = parser.parse_args()


log_scale = not args.linear

outputdir = args.output
if not os.path.exists(outputdir):
    os.makedirs(outputdir)

logging.basicConfig(filename=f"{outputdir}/logger.log",
                    filemode='a',
                    format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logger = logging.getLogger()


filedict = {}
for input in args.input_data:
    input = os.path.abspath(input)
    if os.path.exists(input) and "npz" in input:
        logger.info(f"Found file {input}")
        filedict[Path(input).stem] = np.load(input)
    elif os.path.exists(input):
        logger.info(f"Found folder {input}")
        for file in glob.glob(f"{input}/*.npz"):
            logger.info(f"Found file {file}")
            filedict[Path(file).stem] = np.load(file)
    else:
        raise ValueError(f"The path {input} does not exist!")


# Get list of plot names:
plots = []
for file, data in filedict.items():
    logger.info(file)
    logger.info(data.files)
    logger.info(data["counts"])
    logger.info(data["bin_edges"])
    plots.append(data["plot"].item())
plots = set(plots)

colour_mc = ["mediumblue", "darkred", "darkgreen", "goldenrod"]
colour_both = ["royalblue", "coral", "palegreen", "palegoldenrod"]
colour_data = ["deepskyblue", "orangered", "darkseagreen", "gold"]

for plot in plots:
    ref = {"mc": None, "data": None, "both": None}
    idx = {"mc": -1, "data": -1, "both": -1}
    hist_fig, (ax, ax_ratio) = plt.subplots(
        2,
        1,
        figsize=[13, 13],
        sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1]},
    )
    fig_sob, ax_sob = plt.subplots(
        figsize=[13, 13],
    )
    # Load histogram data
    idx_data = -1
    idx_mc = -1
    for file, data in filedict.items():
        if not data["plot"] == plot:
            continue

        name = file.replace("_UNIFORM", "").replace("_TRANSFORM", "").replace("hist_columns_","").replace("_events_sig_bkg_dnn_score", "")

        logger.info(f"== Amount of weighted events in {name} ==")
        logger.info(sum(data["counts"]))
        logger.info(f"== Amount of events in {name} ==")
        logger.info(data["num_events"])

        if "MC" in file:
            cmap = colour_mc
            ref_key = "mc"
        elif "DATA" in file and "kl" in file:
            cmap = colour_both
            ref_key = "both"
        else:
            cmap = colour_data
            ref_key = "data"
        print(f"The file {file} is part of {ref_key}")

        idx[ref_key] += 1
        if ref[ref_key] is None:
            ref[ref_key] = data
            ax_ratio.axhline(y=1, color="k", linestyle="--")
        ratio = data["counts"] / ref[ref_key]["counts"]
        logger.info(ratio)
        ratio_err = np.sqrt(
                (ref[ref_key]["count_err"] / data["counts"][:-1]) ** 2
                + (ref[ref_key]["counts"][:-1] * data["count_err"] / data["counts"][:-1] ** 2) ** 2
        )
        ax_ratio.errorbar(
            data["bin_edges"],
            ratio,
            yerr=np.zeros(len(ratio)),
            fmt=".",
            label=name,
            color=cmap[idx[ref_key]],
        )

        ax.step(
            data["bin_edges"],
            data["counts"],
            where="post",
            label=name,
            color=cmap[idx[ref_key]],
        )
        ax.fill_between(
            data["bin_edges"],
            data["counts"],
            step="post",
            alpha=0.2,
            color=cmap[idx[ref_key]],
        )
    
        if "sob" in data.files:
            # plot the sob for each bin
            ax_sob.errorbar(
                (data["bin_edges"][:-1]+data["bin_edges"][1:])/2,
                data["sob"],
                yerr=data["sob_err"],
                fmt=".",
                label=name,
                color=cmap[idx[ref_key]],
            )
            ax_sob.fill_between(
                (data["bin_edges"][:-1]+data["bin_edges"][1:])/2,
                data["sob"] - data["sob_err"],
                data["sob"] + data["sob_err"],
                color="grey",
                alpha=0.2,
            )
            ax_sob.legend(loc="upper left")
            ax_sob.set_yscale("linear")
            #hep.cms.lumitext(
            #    f"{era_string}, {lumi}" + r" $fb^{-1}$, (13.6 TeV)",
            #    ax=ax_sob,
            #)
            ax_sob.set_xlabel(plot)
            ax_sob.set_ylabel(r"$s/\sqrt{b}$")
            ax_sob.grid()
            fig_sob.savefig(
                os.path.join(outputdir, f"{plot}_sob.png"),
                bbox_inches="tight",
                dpi=300,
            )
            plt.close(fig_sob)




    ax.legend(loc="upper right")
    ax.set_yscale("log" if log_scale else "linear")

    #hep.cms.lumitext(f"{era_string}, {lumi}" + r" $fb^{-1}$, (13.6 TeV)", ax=ax)
    hep.cms.text(text="Preliminary", ax=ax)

    ax_ratio.set_xlabel(plot)
    ax.set_ylabel("Events")
    ax_ratio.set_ylabel("ratio")

    ax.grid()
    ax_ratio.grid()
    #ax_ratio.set_ylim(0.5, 1.5)
    ax.set_ylim(
        top=(1.3 * ax.get_ylim()[1] if not log_scale else ax.get_ylim()[1] ** 1.3)
    )

    hist_fig.savefig(
        os.path.join(outputdir, f"{plot}.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(hist_fig)
