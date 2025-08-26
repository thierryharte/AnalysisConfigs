import matplotlib.pyplot as plt
import matplotlib
import mplhep as hep
import hist
import os

hep.style.use("CMS")


class HEPPlotter:
    def __init__(self, style: str = "CMS", figsize=None):
        if style:
            hep.style.use(style)
        self.figsize = figsize

    # =============================
    # COMMON HELPERS
    # =============================

    def _save_figure(self, fig, output_base: str):
        for ext in ["png", "pdf", "svg"]:
            fig.savefig(f"{output_base}.{ext}", bbox_inches="tight", dpi=300)
        plt.close(fig)

    def _add_cms_labels(self, ax, lumi="(13.6 TeV)", text="Preliminary"):
        hep.cms.lumitext(lumi, ax=ax)
        hep.cms.text(text, ax=ax)

    def _finalize_plot(
        self,
        fig,
        ax,
        output_base: str,
        xlabel: str,
        ylabel: str,
        ax_ratio=None,
        ratio_label: str = None,
        y_log_scale: bool = False,
        legend: bool = True,
        grid: bool = True,
        set_ylim: bool = True,
    ):
        if legend:
            ax.legend(loc="upper right")

        if y_log_scale:
            ax.set_yscale("log")
        if grid:
            ax.grid()

        if ax_ratio:
            ax_ratio.set_xlabel(xlabel)
            ax_ratio.set_ylabel(ratio_label if ratio_label else "Ratio")
            ax_ratio.grid()
            ax_ratio.set_yscale("log" if y_log_scale else "linear")
            if not y_log_scale:
                ax_ratio.set_ylim(0.5, 1.5)
        else:
            ax.set_xlabel(xlabel)

        ax.set_ylabel(ylabel)
        if set_ylim:
            ax.set_ylim(
                top=(
                    1.7 * ax.get_ylim()[1]
                    if not y_log_scale
                    else ax.get_ylim()[1] ** (1.7)
                )
            )

        self._add_cms_labels(ax)
        self._save_figure(fig, output_base)

    # =============================
    # 1D PLOTTING
    # =============================
    
    # top-level helper (picklable) for multiprocessing
    def plot_1d_histograms_pool_map(self, kwargs):
        # Directly call the plot_1d_histograms method
        return self.plot_1d_histograms(**kwargs)

    def _validate_inputs(self, series_dict):
        ratio_plot = False
        ref_name = None
        for name, props in series_dict.items():
            hist_1d = props["data"]
            is_reference = props.get("style", {}).get("is_reference", False)
            if not isinstance(hist_1d, hist.Hist):
                raise ValueError(f"Expected hist.Hist for {name}, got {type(hist_1d)}")
            if is_reference and ratio_plot:
                raise ValueError("Multiple reference histograms found.")
            if is_reference:
                ratio_plot = True
                ref_name = name
        return ratio_plot, ref_name

    def _create_figure(self, ratio_plot=False):
        if ratio_plot:
            fig, (ax, ax_ratio) = plt.subplots(
                2,
                1,
                figsize=self.figsize,
                sharex=True,
                gridspec_kw={"height_ratios": [2.5, 1]},
            )
            return fig, ax, ax_ratio
        else:
            fig, ax = plt.subplots(figsize=self.figsize)
            return fig, ax, None

    def _plot_histogram(self, ax, name, hist_1d, style,**kwargs):
        histtype = style.get("histtype", "step")
        if histtype == "fill":
            hep.histplot(
                hist_1d,
                histtype="fill",
                label=name,
                ax=ax,
                facecolor=style.get("facecolor", style.get("color")),
                edgecolor=style.get("edgecolor", style.get("color")),
                alpha=style.get("alpha", 0.5),
                **kwargs
            )
        else:
            hep.histplot(
                hist_1d,
                w2method="sqrt",
                histtype=histtype,
                label=name,
                ax=ax,
                color=style.get("color"),
                **kwargs
            )
        ax.set_xlabel("")

    def _plot_ratio(self, ax_ratio, hist_1d, ref_hist, name, is_reference, color):
        bins = hist_1d.axes[0].edges
        bins_center = (bins[1:] + bins[:-1]) / 2

        ratio, err_up, err_down = hep.get_comparison(
            hist_1d,
            ref_hist,
            comparison="split_ratio" if is_reference else "ratio",
        )

        if is_reference:
            ax_ratio.axhline(y=1, linestyle="--", color=color)
            ax_ratio.fill_between(
                bins_center,
                1 - err_up,
                1 + err_up,
                alpha=0.5,
                color=color,
            )
        else:
            hep.histplot(
                ratio,
                bins=bins,
                yerr=err_up,
                histtype="errorbar",
                label=name,
                ax=ax_ratio,
                color=color,
            )

    def plot_1d_histograms(
        self,
        series_dict,
        output_base,
        xlabel,
        ylabel="Events",
        log_scale=False,
        ratio_label=None,
        **kwargs
    ):
        print(f"Plotting 1D histograms to {output_base}")
        ratio_plot, ref_name = self._validate_inputs(series_dict)
        fig, ax, ax_ratio = self._create_figure(ratio_plot)
        for name, props in series_dict.items():
            hist_1d = props["data"]
            style = props.get("style")
            is_reference = style.get("is_reference", False) if style else False

            self._plot_histogram(ax, name, hist_1d, style,**kwargs)

            if ratio_plot and ax_ratio is not None:
                self._plot_ratio(
                    ax_ratio,
                    hist_1d,
                    series_dict[ref_name]["data"],
                    name,
                    is_reference,
                    style["color"],
                )

        self._finalize_plot(
            fig,
            ax,
            output_base,
            xlabel=xlabel,
            ylabel=ylabel,
            ax_ratio=ax_ratio,
            ratio_label=ratio_label,
            y_log_scale=log_scale,
        )

    # =============================
    # 2D PLOTTING
    # =============================
    
    # Top-level helper (picklable) for multiprocessing
    def plot_2d_histograms_pool_map(self, kwargs):
        # Directly call the plot_2d_histogram method
        return self.plot_2d_histogram(**kwargs)

    def _plot_2d_histogram(self, ax, hist2d, log_scale, label, **kwargs):
        hep.hist2dplot(
            hist2d,
            ax=ax,
            label=label,
            norm=matplotlib.colors.LogNorm() if log_scale else None,
            cmap="viridis" if "cmap" not in kwargs else kwargs.pop("cmap"),
            **kwargs,
        )

    # def plot_2d_histograms(
    #     self, series_dict, output_base, xlabel, ylabel, log_scale=False
    # ):
    #     for name, props in series_dict.items():
    #         hist2d = props["data"]
    #         fig, ax, _ = self._create_figure()
    #         self._plot_2d_histogram(ax, hist2d, name)

    #         self._finalize_plot(
    #             fig,
    #             ax,
    #             output_base + f"_{name}",
    #             xlabel=xlabel,
    #             ylabel=ylabel,
    #             log_scale=log_scale,
    #         )

    def plot_2d_histogram(
        self, hist2d, output_base, xlabel, ylabel, log_scale=False, label="", **kwargs
    ):

        fig, ax, _ = self._create_figure()
        self._plot_2d_histogram(ax, hist2d, log_scale, label, **kwargs)

        self._finalize_plot(
            fig,
            ax,
            output_base,
            xlabel=xlabel,
            ylabel=ylabel,
            y_log_scale=False,
            legend=False,
            set_ylim=False,
        )

    # =============================
    # CURVE PLOTTING
    # =============================
    
     # Top-level helper (picklable) for multiprocessing
    def plot_curves_pool_map(self, kwargs):
        # Directly call the plot_curves method
        return self.plot_curves(**kwargs)
    
    def _plot_curve(self, ax, x_bins, values, errors, label, style,**kwargs):
        hep.histplot(
            values,
            bins=x_bins,
            yerr=errors,
            xerr=True,
            histtype="errorbar",
            label=label,
            ax=ax,
            color=style.get("color"),
            marker=style.get("marker"),
            markersize=style.get("markersize", 6),
            **kwargs
        )

    def plot_curves(
        self,
        series_dict,
        output_base,
        x_bins,
        xlabel,
        ylabel,
        log_scale=False,
        **kwargs
    ):
        fig, ax, _ = self._create_figure()

        for name, props in series_dict.items():
            values, errors = props["values"], props["errors"]
            style = props.get("style")
            self._plot_curve(ax, x_bins, values, errors, name, style,**kwargs)

        self._finalize_plot(
            fig,
            ax,
            output_base,
            xlabel=xlabel,
            ylabel=ylabel,
            y_log_scale=log_scale,
        )
