import matplotlib.pyplot as plt
import mplhep as hep
import hist
hep.style.use("CMS")

def plot_1d_histograms(hists_dict, output_name, var_label, log_scale, ratio_label=None):
    
    # hists_dict ={hist_name: (hist_1d, ratio, color)}
    ratio_plot= False
    for hist_name, (hist_1d, ratio_hist_den, color) in hists_dict.items():
        if not isinstance(hist_1d, hist.Hist):
            raise ValueError(f"Expected hist_1d to be of type hist.Hist, got {type(hist_1d)}")
        if not isinstance(ratio_hist_den, bool):
            raise ValueError(f"Expected ratio to be of type bool, got {type(ratio_hist_den)}")
        # if multiple histograms have the ratio_hist_den set to True, raise an error
        if ratio_hist_den and ratio_plot:
            raise ValueError("Multiple histograms with ratio_hist_den set to True found.")
        
        if ratio_hist_den:
            ratio_plot = True
            hist_name_ratio = hist_name
            # print(f"Found histogram {hist_name} with ratio_hist_den set to True, will plot ratio.")
    
    if ratio_plot:
        fig, (ax, ax_ratio) = plt.subplots(
                    2,
                    1,
                    figsize=[13, 13],
                    sharex=True,
                    gridspec_kw={"height_ratios": [2.5, 1]},
                )
                
    else:
        fig, ax = plt.subplots(
            figsize=[13, 13],
        )
        
        
    for hist_name, (hist_1d, ratio_hist_den, color) in hists_dict.items():
        print(f"Plotting 1d histogram for {hist_name}, var  {var_label} in {output_name}")
        
        bins=hist_1d.axes[0].edges
        bins_center = (hist_1d.axes[0].edges[1:] + hist_1d.axes[0].edges[:-1]) / 2
        
        # hep.histplot(
        #     hist_1d,
        #     ax=ax,
        #     label=hist_name,
        #     # color=hep.style.CMS.colors[key],
        #     histtype="step",
        # )
        hep.histplot(
                    hist_1d,
                    # bins=bins,
                    # yerr=True,
                    # w2=hist_num.variances(),
                    w2method="sqrt",
                    histtype="step",
                    label=hist_name,
                    ax=ax,
                    color=color
                )
        ax.set_xlabel("")
        
        if ratio_plot:
            ratio, err_ratio_up, err_ratio_down = hep.get_comparison(
                hist_1d, hists_dict[hist_name_ratio][0], comparison="split_ratio" if ratio_hist_den else "ratio"
            )
            if ratio_hist_den:
                ax_ratio.axhline(y=1, linestyle="--",color=color)
                ax_ratio.fill_between(
                            bins_center,
                            1 - err_ratio_up, # ensure that the errors are symmetric
                            1 + err_ratio_up, # ensure that the errors are symmetric
                            alpha=0.5,
                            color=color,
                        )
            else:
                hep.histplot(
                        ratio,
                        bins=bins,
                        yerr=err_ratio_up,
                        histtype="errorbar",
                        label=hist_name,
                        ax=ax_ratio,
                        color=color,
                    )
        
    ax.legend(loc="best")
    ax.set_yscale("log" if log_scale else "linear")
    ax.set_ylabel("Events")
    ax.set_ylim(
                top=(2 * ax.get_ylim()[1] if not log_scale else ax.get_ylim()[1] ** (2))
            )
    ax.grid()
    
    if ratio_plot:
        ax_ratio.set_xlabel(var_label)
        ax_ratio.set_ylabel(ratio_label if ratio_label else "Ratio")
        ax_ratio.grid()
        ax_ratio.set_yscale("log" if log_scale else "linear")
        if not log_scale:
            ax_ratio.set_ylim(0.5, 1.5)
        # ax_ratio.axhline(y=1, linestyle="--", color="black")
        # ax_ratio.legend(loc="best")
    else:
        ax.set_xlabel(var_label)
    
    hep.cms.lumitext(r"(13.6 TeV)", ax=ax)
    hep.cms.text(text="Preliminary", ax=ax)
    
    fig.savefig(f"{output_name}.png", bbox_inches="tight", dpi=300)
    fig.savefig(f"{output_name}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{output_name}.svg", bbox_inches="tight", dpi=300)
    plt.close(fig)
        