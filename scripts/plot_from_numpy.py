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


filedict = {}
for input in args.input_data:
    input = os.path.abspath(input)
    if os.path.exists(input) and "npz" in input:
        print(f"Found file {input}")
        filedict[Path(input).stem] = np.load(input)
    elif os.path.exists(input):
        print(f"Found folder {input}")
        for file in glob.glob(f"{input}/*.npz"):
            print(f"Found file {file}")
            filedict[Path(file).stem] = np.load(file)
    else:
        raise ValueError(f"The path {input} does not exist!")


# Get list of plot names:
plots = []
for file, data in filedict.items():
    print(file)
    print(data.files)
    print(data["counts"])
    print(data["bin_edges"])
    plots.append(data["plot"].item())
plots = set(plots)

colours = ["b", "r", "g", "y"]

for plot in plots:
    ref_data = None
    fig, (ax, ax_ratio) = plt.subplots(
        2,
        1,
        figsize=[13, 13],
        sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1]},
    )
    # Load histogram data
    idx_data = -1
    idx_mc = -1
    for file, data in filedict.items():
        if not data["plot"] == plot:
            continue

        name = file.replace("_UNIFORM", "").replace("_TRANSFORM", "").replace("hist_columns_","").replace("_events_sig_bkg_dnn_score", "")

        if "DATA" in file:
            idx_data += 1
            if ref_data is None:
                ref_data = data
                ax_ratio.axhline(y=1, color="k", linestyle="--")
            ratio = data["counts"] / ref_data["counts"]
            print(ratio)
            # To be added when we also save the errors
            #  ratio_err = np.sqrt(
            #      (values["err_num"] / values["h_den"]) ** 2
            #      + (values["h_num"] * values["err_den"] / values["h_den"] ** 2) ** 2
            #  )
            ax_ratio.errorbar(
                data["bin_edges"],
                ratio,
                yerr=np.zeros(len(ratio)),
                fmt=".",
                label=name,
                color=colours[idx_data],
            )
        else:
            idx_mc +=1

        ax.step(
            data["bin_edges"],
            data["counts"],
            where="post",
            label=name,
            color=colours[idx_data if "DATA" in file else idx_mc],
        )
        ax.fill_between(
            data["bin_edges"],
            data["counts"],
            step="post",
            alpha=0.5,
            color=colours[idx_data if "DATA" in file else idx_mc],
        )


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

    fig.savefig(
        os.path.join(outputdir, f"{plot}.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)
