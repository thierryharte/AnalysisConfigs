import matplotlib.pyplot as plt
import matplotlib
import mplhep as hep
import hist
import os

hep.style.use("CMS")


class CMSPlotter:
    def __init__(self, style: str = "CMS", figsize=(13, 13)):
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
        log_scale: bool = False,
        legend: bool = True,
        grid: bool = True,
    ):
        if legend:
            ax.legend(loc="best")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if log_scale:
            ax.set_yscale("log")
        if grid:
            ax.grid()
        self._add_cms_labels(ax)
        self._save_figure(fig, output_base)

    # =============================
    # 1D PLOTTING
    # =============================

    def _validate_inputs(self, series_dict):
        ratio_plot = False
        ref_name = None
        for name, props in series_dict.items():
            hist_1d = props["data"]
            is_reference = props.get("is_reference", False)
            if not isinstance(hist_1d, hist.Hist):
                raise ValueError(
                    f"Expected hist.Hist for {name}, got {type(hist_1d)}"
                )
            if is_reference and ratio_plot:
                raise ValueError("Multiple reference histograms found.")
            if is_reference:
                ratio_plot = True
                ref_name = name
        return ratio_plot, ref_name

    def _create_figure(self, ratio_plot):
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

    def _plot_histogram(self, ax, name, hist_1d, color):
        hep.histplot(
            hist_1d,
            w2method="sqrt",
            histtype="step",
            label=name,
            ax=ax,
            color=color,
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
        self, plots_dict, output_dir, xlabel, ylabel,
        log_scale=False, ratio_label=None
    ):
        os.makedirs(output_dir, exist_ok=True)

        for obs_name, series_dict in plots_dict.items():
            ratio_plot, ref_name = self._validate_inputs(series_dict)
            fig, ax, ax_ratio = self._create_figure(ratio_plot)

            for name, props in series_dict.items():
                hist_1d = props["data"]
                color = props.get("color")
                is_reference = props.get("is_reference", False)

                self._plot_histogram(ax, name, hist_1d, color)

                if ratio_plot and ax_ratio is not None:
                    self._plot_ratio(
                        ax_ratio,
                        hist_1d,
                        series_dict[ref_name]["data"],
                        name,
                        is_reference,
                        color,
                    )

            out_base = os.path.join(output_dir, f"1d_{obs_name}")
            self._finalize_plot(
                fig, ax, out_base,
                xlabel=("" if ratio_plot else xlabel),
                ylabel=ylabel,
                log_scale=log_scale,
            )

            if ax_ratio:
                ax_ratio.set_xlabel(xlabel)
                ax_ratio.set_ylabel(ratio_label if ratio_label else "Ratio")
                ax_ratio.grid()
                ax_ratio.set_yscale("log" if log_scale else "linear")
                if not log_scale:
                    ax_ratio.set_ylim(0.5, 1.5)

    # =============================
    # 2D PLOTTING
    # =============================

    def _plot_2d_histogram(self, ax, hist2d, label):
        hep.hist2dplot(
            hist2d,
            ax=ax,
            label=label,
            cmap="viridis",
            norm=matplotlib.colors.LogNorm(),
        )

    def plot_2d_histograms(
        self, plots_dict, output_dir,
        xlabel, ylabel, log_scale=False
    ):
        os.makedirs(output_dir, exist_ok=True)

        for obs_name, series_dict in plots_dict.items():
            for name, props in series_dict.items():
                hist2d = props["data"]
                fig, ax = plt.subplots(figsize=self.figsize)
                self._plot_2d_histogram(ax, hist2d, name)

                out_base = os.path.join(output_dir, f"2d_{obs_name}_{name}")
                self._finalize_plot(
                    fig, ax, out_base,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    log_scale=log_scale,
                )

    # =============================
    # CURVE PLOTTING
    # =============================

    def _plot_curve(self, ax, x_bins, values, errors, label, color):
        x_centers = (x_bins[1:] + x_bins[:-1]) / 2.0
        x_errors = (x_bins[1:] - x_bins[:-1]) / 2.0
        ax.errorbar(
            x_centers,
            values,
            xerr=x_errors,
            yerr=errors,
            label=label,
            color=color,
            fmt=".",
        )

    def plot_curves(
        self,
        plots_dict,
        output_dir,
        x_bins,
        xlabel,
        ylabel,
        log_scale=False,
    ):
        os.makedirs(output_dir, exist_ok=True)

        for obs_name, series_dict in plots_dict.items():
            fig, ax = plt.subplots(figsize=self.figsize)

            for name, props in series_dict.items():
                values, errors = props["data"]
                color = props.get("color")
                self._plot_curve(ax, x_bins, values, errors, name, color)

            out_base = os.path.join(output_dir, f"curve_{obs_name}")
            self._finalize_plot(
                fig, ax, out_base,
                xlabel=xlabel,
                ylabel=ylabel,
                log_scale=log_scale,
            )
