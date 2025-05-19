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
    if os.path.exists(input) and "npz" in input:
        filedict[Path(input).stem] = np.load(input)
    elif os.path.isdir(input):
        input_dir = os.path.dirname(input)
        for file in glob.glob(f"{input_dir}/*.npz"):
            filedict[Path(file).stem] = np.load(file)
    else:
        raise ValueError(f"The path {input} does not exist!")

fig, (ax) = plt.subplots(
    1,
    1,
    figsize=[13, 13],
    sharex=True,
    #gridspec_kw={"height_ratios": [2.5, 1]},
)

# Load histogram data
for file, data in filedict.items():
    print(file)
    print(data.files)
    print(data["counts"])
    print(data["bin_edges"])


    ax.step(
        data["bin_edges"],
        data["counts"],
        where="post",
        label=file,
    )
    ax.fill_between(
        data["bin_edges"],
        data["counts"],
        step="post",
        alpha=0.5,
        #color=values["color"][1],
    )


ax.legend(loc="upper right")
ax.set_yscale("log" if log_scale else "linear")

#hep.cms.lumitext(f"{era_string}, {lumi}" + r" $fb^{-1}$, (13.6 TeV)", ax=ax)
hep.cms.text(text="Preliminary", ax=ax)

#ax_ratio.set_xlabel(var_plot_name)
ax.set_ylabel("Events")
#ax_ratio.set_ylabel("Data/Pred.")

ax.grid()
#ax_ratio.grid()
#ax_ratio.set_ylim(0.5, 1.5)
ax.set_ylim(
    top=(1.3 * ax.get_ylim()[1] if not log_scale else ax.get_ylim()[1] ** 1.3)
)

fig.savefig(
    os.path.join(outputdir, f"plots_from_onnx.png"),
    bbox_inches="tight",
    dpi=300,
)
plt.close(fig)
