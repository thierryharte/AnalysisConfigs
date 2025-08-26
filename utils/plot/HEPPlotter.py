import matplotlib.pyplot as plt
import matplotlib
import mplhep as hep
import hist
import os

hep.style.use("CMS")

class HEPPlotter:
    """
    A helper class for standardized plotting in HEP analyses using `mplhep`.

    Supports:
    - 1D histograms with optional ratio plots
    - 2D histograms
    - Curve plotting (values with errors vs binning)

    Parameters
    ----------
    style : str, optional
        Plotting style to use (default: "CMS").
    figsize : tuple, optional
        Figure size to pass to matplotlib (default: None).
    """

    def __init__(self, style: str = "CMS", figsize=None):
        if style:
            hep.style.use(style)
        self.figsize = figsize

    # =============================
    # COMMON HELPERS
    # =============================

    def _save_figure(self, fig, output_base: str):
        """
        Save a matplotlib figure to multiple formats (png, pdf, svg).

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure object to save.
        output_base : str
            Base name for output files (extensions are added automatically).
        """
        for ext in ["png", "pdf", "svg"]:
            fig.savefig(f"{output_base}.{ext}", bbox_inches="tight", dpi=300)
        plt.close(fig)

    def _add_cms_labels(self, ax, lumi="(13.6 TeV)", text="Preliminary"):
        """
        Add CMS-style labels to a plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to which the labels are added.
        lumi : str, optional
            Luminosity text (default: "(13.6 TeV)").
        text : str, optional
            Additional label text (default: "Preliminary").
        """
        hep.cms.lumitext(lumi, ax=ax)
        hep.cms.text(text, ax=ax)
        
    def _add_annotation(
        self,
        ax,
        text: str,
        x: float = 0.05,
        y: float = 0.95,
        ha: str = "left",
        va: str = "center",
        color: str = "black",
        fontsize: int = 14,
        **kwargs,
    ):
        """
        Add a text annotation to the given axis.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis to annotate.
        text : str
            The text to display (supports LaTeX formatting).
        x, y : float, optional
            Position in axis coordinates (default: (0.05, 0.95)).
        ha, va : str, optional
            Horizontal and vertical alignment.
        color : str, optional
            Text color.
        fontsize : int, optional
            Font size.
        kwargs : dict, optional
            Extra arguments forwarded to `ax.text`.
        """
        ax.text(
            x,
            y,
            text,
            transform=ax.transAxes,
            horizontalalignment=ha,
            verticalalignment=va,
            color=color,
            fontsize=fontsize,
            **kwargs,
        )

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
        annotations: list = None,  
    ):
        """
        Finalize a plot by setting labels, scales, legend, CMS text, and saving.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure object.
        ax : matplotlib.axes.Axes
            Main axis.
        output_base : str
            Output base filename for saving.
        xlabel : str
            Label for x-axis.
        ylabel : str
            Label for y-axis.
        ax_ratio : matplotlib.axes.Axes, optional
            Axis for ratio plot if enabled.
        ratio_label : str, optional
            Label for ratio axis.
        y_log_scale : bool, optional
            If True, set y-axis to log scale.
        legend : bool, optional
            If True, draw legend.
        grid : bool, optional
            If True, show grid lines.
        set_ylim : bool, optional
            If True, extend y-axis upper limit.
        annotations : list of dict, optional
            List of annotation dictionaries with keys matching `ax.text` parameters.
        """
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
        
        if annotations:
            for ann in annotations:
                self._add_annotation(
                    ax,
                    text=ann.get("text", ""),
                    x=ann.get("x", 0.05),
                    y=ann.get("y", 0.95),
                    ha=ann.get("ha", "left"),
                    va=ann.get("va", "center"),
                    color=ann.get("color", "black"),
                    fontsize=ann.get("fontsize", 14),
                    **{k: v for k, v in ann.items() if k not in ["x", "y", "text", "ha", "va", "color", "fontsize"]}
                )

        self._add_cms_labels(ax)
        self._save_figure(fig, output_base)

    # =============================
    # 1D PLOTTING
    # =============================

    def plot_1d_histograms_pool_map(self, kwargs):
        """
        Wrapper for multiprocessing (picklable) version of `plot_1d_histograms`.
        """
        return self.plot_1d_histograms(**kwargs)

    def _validate_inputs(self, series_dict):
        """
        Validate inputs for 1D histograms and determine if a ratio plot is needed.

        Parameters
        ----------
        series_dict : dict
            Dictionary of histograms with styles.

        Returns
        -------
        ratio_plot : bool
            Whether a ratio plot is required.
        ref_name : str
            Name of the reference histogram for ratio plots.
        """
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
        """
        Create a figure and axes, with optional ratio subplot.

        Parameters
        ----------
        ratio_plot : bool, optional
            If True, create a ratio subplot.

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        ax_ratio : matplotlib.axes.Axes or None
        """
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

    def _plot_histogram(self, ax, name, hist_1d, style, **kwargs):
        """
        Plot a 1D histogram onto the given axis.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to draw on.
        name : str
            Label for legend.
        hist_1d : hist.Hist
            Histogram to plot.
        style : dict
            Style dictionary (histtype, color, etc.).
        """
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
        """
        Plot ratio of histogram to a reference histogram.

        Parameters
        ----------
        ax_ratio : matplotlib.axes.Axes
            Ratio subplot axis.
        hist_1d : hist.Hist
            Histogram to compare.
        ref_hist : hist.Hist
            Reference histogram.
        name : str
            Label for legend.
        is_reference : bool
            If True, draw uncertainty band instead of points.
        color : str
            Color of markers or band.
        """
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
        annotations=None,
        **kwargs
    ):
        """
        Plot one or more 1D histograms, optionally with ratio subplot.

        Parameters
        ----------
        series_dict : dict
            Dictionary mapping names to {"data": hist.Hist, "style": dict}.
            The reference histogram must include {"is_reference": True}.
        output_base : str
            Base name for saved figures.
        xlabel : str
            Label for x-axis.
        ylabel : str, optional
            Label for y-axis (default: "Events").
        log_scale : bool, optional
            If True, use logarithmic y-axis.
        ratio_label : str, optional
            Label for ratio subplot (default: "Ratio").
        annotations : list of dict, optional
            List of annotation dictionaries with keys matching `ax.text` parameters.
        """
        print(f"Plotting 1D histograms to {output_base}")
        ratio_plot, ref_name = self._validate_inputs(series_dict)
        fig, ax, ax_ratio = self._create_figure(ratio_plot)
        for name, props in series_dict.items():
            hist_1d = props["data"]
            style = props.get("style")
            is_reference = style.get("is_reference", False) if style else False

            self._plot_histogram(ax, name, hist_1d, style, **kwargs)

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
            annotations=annotations,
        )

    # =============================
    # 2D PLOTTING
    # =============================

    def plot_2d_histograms_pool_map(self, kwargs):
        """Wrapper for multiprocessing version of `plot_2d_histogram`."""
        return self.plot_2d_histogram(**kwargs)

    def _plot_2d_histogram(self, ax, hist2d, log_scale, label, **kwargs):
        """
        Plot a 2D histogram.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to draw on.
        hist2d : hist.Hist
            2D histogram to plot.
        log_scale : bool
            If True, use log color scale.
        label : str
            Label for legend.
        """
        hep.hist2dplot(
            hist2d,
            ax=ax,
            label=label,
            norm=matplotlib.colors.LogNorm() if log_scale else None,
            cmap="viridis" if "cmap" not in kwargs else kwargs.pop("cmap"),
            **kwargs,
        )

    def plot_2d_histogram(
        self, hist2d, output_base, xlabel, ylabel, log_scale=False, label="", annotations=None, **kwargs
    ):
        """
        Plot and save a 2D histogram.

        Parameters
        ----------
        hist2d : hist.Hist
            2D histogram to plot.
        output_base : str
            Base name for saved figures.
        xlabel : str
            Label for x-axis.
        ylabel : str
            Label for y-axis.
        log_scale : bool, optional
            If True, use log color scale (default: False).
        label : str, optional
            Legend label (default: "").
        annotations : list of dict, optional
            List of annotation dictionaries with keys matching `ax.text` parameters.
        """
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
            annotations=annotations
        )

    # =============================
    # CURVE PLOTTING
    # =============================

    def plot_curves_pool_map(self, kwargs):
        """Wrapper for multiprocessing version of `plot_curves`."""
        return self.plot_curves(**kwargs)

    def _plot_curve(self, ax, x_bins, values, errors, label, style, **kwargs):
        """
        Plot values with errors vs binning (curve).

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to draw on.
        x_bins : array-like
            Bin edges for x-axis.
        values : array-like
            Y values to plot.
        errors : array-like
            Errors associated with y values.
        label : str
            Legend label.
        style : dict
            Style dictionary (color, marker, etc.).
        """
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
        annotations=None,
        **kwargs
    ):
        """
        Plot multiple curves (values with errors) and save.

        Parameters
        ----------
        series_dict : dict
            Mapping of names to {"values": arr, "errors": arr, "style": dict}.
        output_base : str
            Base name for saved figures.
        x_bins : array-like
            Bin edges for x-axis.
        xlabel : str
            Label for x-axis.
        ylabel : str
            Label for y-axis.
        log_scale : bool, optional
            If True, set y-axis to log scale.
        annotations : list of dict, optional
            List of annotation dictionaries with keys matching `ax.text` parameters.
        """
        fig, ax, _ = self._create_figure()

        for name, props in series_dict.items():
            values, errors = props["values"], props["errors"]
            style = props.get("style")
            self._plot_curve(ax, x_bins, values, errors, name, style, **kwargs)

        self._finalize_plot(
            fig,
            ax,
            output_base,
            xlabel=xlabel,
            ylabel=ylabel,
            y_log_scale=log_scale,
            annotations=annotations,
        )
