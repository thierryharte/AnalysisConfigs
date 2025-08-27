import matplotlib.pyplot as plt
import matplotlib
import hist
import matplotlib.ticker as mtick

import mplhep as hep

class HEPPlotter:
    """
    A helper class for standardized plotting in HEP analyses using `mplhep`.

    Supports:
    - 1D histograms with optional ratio plots
    - 2D histograms
    - Graph plotting (values with error bars)

    Usage
    -----
    job = (HEPPlotter("CMS"
            .set_output("out/plot")
            .set_labels(xlabel="pT [GeV]", ylabel="Events")
            .set_data(series_dict)
            .set_options(log_scale=True, ratio_label="Data/MC"))
    job.run()  # produces the plot
    """

    def __init__(self, style="CMS"):
        # core settings
        self.style = style
        hep.style.use(style)
        
        self.figsize = None
        self.lumitext = "(13.6 TeV)"
        self.data_formats = ["png", "pdf", "svg"]

        # user-configurable
        self.output_base = None
        self.xlabel = None
        self.ylabel = "Events"
        self.cbar_label = "Events"
        self.series_dict = None
        self.plot_type = "1d"  # "1d", "2d", "graph"

        # options
        self.log_scale = False
        self.ratio_label = None
        self.grid = True
        self.legend = True
        self.legend_loc = "best"
        self.set_ylim = True
        self.extra_kwargs = {}

        # internal
        self._annotations = []
        self._lines = []

    # ----------------------------
    # CONFIGURATION METHODS
    # ----------------------------

    def set_plot_config(self, figsize=None, lumitext="(13.6 TeV)", formats=None):
        """Set the plotting style and related options."""
        self.figsize = figsize
        self.lumitext = lumitext
        if formats:
            self.data_formats = formats
        return self

    def set_output(self, output_base):
        """Set the base name for output files (without extension)."""
        self.output_base = output_base
        return self

    def set_labels(self, xlabel, ylabel="Events", cbar_label="Events"):
        """Set the x and y axis labels."""
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.cbar_label = cbar_label
        return self

    def set_data(self, series_dict, plot_type="1d"):
        """Provide the plotting data (dict structure varies by plot_type).
        series_dict structure:
        - For 1D histograms:
            series_dict : dict
                Dictionary mapping names to {name:{"data": hist.Hist, "style": dict}}.
                The reference histogram must include {"is_reference": True}.
        - For 2D histograms:
            series_dict : dict
                {name:{"data": hist.Hist, "style": dict}}
        - For graphs:
            series_dict : dict
                Dictionary mapping names to {name:{
                                                "data": {
                                                    "x":[list, list],
                                                    "y":[list, list],
                                                },
                                                "style": dict
                                                }
                                            }.
        """
        self.series_dict = series_dict
        self.plot_type = plot_type
        return self

    def set_extra_kwargs(self, **kwargs):
        """Extra kwargs passed to the plotting functions."""
        self.extra_kwargs.update(kwargs)
        return self

    def set_options(self, **kwargs):
        """Generic options setter (log_scale, ratio_label, legend, grid...)."""
        self.log_scale = kwargs.get("log_scale", self.log_scale)
        self.ratio_label = kwargs.get("ratio_label", self.ratio_label)
        self.legend = kwargs.get("legend", self.legend)
        self.legend_loc = kwargs.get("legend_loc", self.legend_loc)
        self.grid = kwargs.get("grid", self.grid)
        self.set_ylim = kwargs.get("set_ylim", self.set_ylim)
        return self

    def add_annotation(self, **kwargs):
        """Annotation kwargs go directly to ax.text() with transform=ax.transAxes"""
        self._annotations.append(kwargs)
        return self

    def add_line(self, orientation="h", **kwargs):
        """Add a horizontal or vertical line to the plot.
        orientation: 'h' for horizontal, 'v' for vertical
        kwargs: passed directly to ax.axhline or ax.axvline
        """
        self._lines.append((orientation, kwargs))
        return self

    # ----------------------------
    # INTERNAL HELPERS
    # ----------------------------
    
    def _close_fig(self, fig):
        """Close the figure to free memory."""
        plt.close(fig)

    def _save(self, fig):
        """Save the figure in all specified formats."""
        for ext in self.data_formats:
            fig.savefig(f"{self.output_base}.{ext}", bbox_inches="tight", dpi=300)


    def _add_cms_labels(self, ax):
        """Add CMS style labels to the plot."""
        hep.cms.lumitext(self.lumitext, ax=ax)
        hep.cms.text("Preliminary", ax=ax)

    def _apply_annotations(self, ax):
        """Apply all stored annotations to the axes."""
        for ann in self._annotations:
            ax.text(
                **ann,
                transform=ax.transAxes,
            )

    def _add_lines(self, ax):
        """Add all stored horizontal/vertical lines to the axes."""
        for orient, kwargs in self._lines:
            if orient == "h":
                ax.axhline(**kwargs)
            else:
                ax.axvline(**kwargs)

    # ----------------------------
    # IMPLEMENTATIONS
    # ----------------------------

    def _plot_1d(self):
        """Plot 1D histograms with optional ratio plot."""
        ratio_plot, ref_name = self._validate_inputs(self.series_dict)
        fig, ax, ax_ratio = self._create_figure(ratio_plot)

        for name, props in self.series_dict.items():
            hist_1d = props["data"]
            style = props.get("style", {})
            is_ref = style.get("is_reference", False)

            # draw histogram
            self._plot_histogram(ax, name, hist_1d, style, **self.extra_kwargs)

            # ratio
            if ratio_plot and ax_ratio is not None:
                self._plot_ratio(
                    ax_ratio,
                    hist_1d,
                    self.series_dict[ref_name]["data"],
                    name,
                    is_ref,
                    style.get("color"),
                )

        self._finalize(fig, ax, ax_ratio)

    def _plot_2d(self):
        """Plot 2D histograms."""
        fig, ax, _ = self._create_figure()
        for name, props in self.series_dict.items():
            if not isinstance(props["data"], hist.Hist) or props["data"].ndim != 2:
                raise ValueError(
                    f"Expected 2D hist.Hist for {name}, got {type(props['data'])} with ndim={props['data'].ndim}"
                )
            hist2d = props["data"]
            label = props["style"].get("label", "")
            hep.hist2dplot(
                hist2d,
                ax=ax,
                label=label,
                norm=matplotlib.colors.LogNorm() if self.log_scale else None,
                cmap=props["style"].get("cmap", "viridis"),
                **self.extra_kwargs,
            )
        self._finalize(fig, ax)

    def _plot_graph(self):
        """Plot graphs with error bars."""
        fig, ax, _ = self._create_figure()
        for name, props in self.series_dict.items():
            y_values, y_errors = (
                props["data"]["y"][0],
                props["data"]["y"][1],
            )
            x_values, x_errors = (
                props["data"]["x"][0],
                props["data"]["x"][1],
            )
            style = props.get("style", {})
            ax.errorbar(
                y=y_values,
                x=x_values,
                yerr=y_errors,
                xerr=x_errors,
                fmt=style.get("fmt", "o"),
                label=name,
                color=style.get("color"),
                markersize=style.get("markersize"),
                **self.extra_kwargs,
            )
        self._finalize(fig, ax)

    # ----------------------------
    # UTILITIES
    # ----------------------------

    def _validate_inputs(self, series_dict):
        """Validate the input series_dict for 1D plotting."""
        ratio_plot = False
        ref_name = None
        for name, props in series_dict.items():
            hist_1d = props["data"]
            if not isinstance(hist_1d, hist.Hist):
                raise ValueError(f"Expected hist.Hist for {name}, got {type(hist_1d)}")
            if props.get("style", {}).get("is_reference", False):
                if ratio_plot:
                    raise ValueError("Multiple reference histograms found.")
                ratio_plot = True
                ref_name = name
        return ratio_plot, ref_name

    def _create_figure(self, ratio_plot=False):
        """Create figure and axes, with optional ratio subplot."""
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
        """Plot a single 1D histogram on the given axes."""
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
                **kwargs,
            )
        else:
            hep.histplot(
                hist_1d,
                w2method="sqrt",
                histtype=histtype,
                label=name,
                ax=ax,
                color=style.get("color"),
                **kwargs,
            )
        ax.set_xlabel("")

    def _plot_ratio(self, ax_ratio, hist_1d, ref_hist, name, is_reference, color):
        """Plot the ratio of hist_1d to ref_hist on the ratio axes."""
        bins = hist_1d.axes[0].edges
        centers = (bins[1:] + bins[:-1]) / 2
        ratio, err_up, err_down = hep.get_comparison(
            hist_1d,
            ref_hist,
            comparison="split_ratio" if is_reference else "ratio",
        )
        if is_reference:
            ax_ratio.axhline(y=1, linestyle="--", color=color)
            ax_ratio.fill_between(
                centers, 1 - err_up, 1 + err_up, alpha=0.5, color=color
            )
        else:
            hep.histplot(
                ratio,
                bins=bins,
                yerr=err_up,
                xerr=True,
                histtype="errorbar",
                label=name,
                ax=ax_ratio,
                color=color,
            )

    def _finalize(self, fig, ax, ax_ratio=None):
        """Final adjustments and saving the figure."""
        if self.legend:
            handles, labels = ax.get_legend_handles_labels()
            if len(handles) > 5:
                ax.legend(loc=self.legend_loc, ncol=2, fontsize="small")
            else:
                ax.legend(loc=self.legend_loc)

        if self.log_scale and self.plot_type != "2d":
            ax.set_yscale("log")
        if self.grid:
            ax.grid()

        if ax_ratio:
            ax_ratio.set_xlabel(self.xlabel)
            ax_ratio.set_ylabel(self.ratio_label or "Ratio")
            if self.grid:
                ax_ratio.grid()
            if self.log_scale:
                ax_ratio.set_yscale("log")
            if not self.log_scale and self.set_ylim:
                ax_ratio.set_ylim(0.5, 1.5)
        else:
            ax.set_xlabel(self.xlabel)

        ax.set_ylabel(self.ylabel)
        
        # define the scalar format for y-axis
        # ax.yaxis.set_major_formatter(mtick.ScalarFormatter())
        # ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))


        if self.set_ylim and self.plot_type != "2d":
            ax.set_ylim(
                top=(
                    1.7 * ax.get_ylim()[1]
                    if not self.log_scale
                    else ax.get_ylim()[1] ** (1.7)
                )
            )
        
        if self.plot_type == "2d":
            # label colorbar
            cbar = ax.collections[0].colorbar
            cbar.set_label(self.cbar_label)
        
        self._apply_annotations(ax)
        self._add_lines(ax)
        
        self._add_cms_labels(ax)
        self._save(fig)
        self._close_fig(fig)

    # ----------------------------
    # PLOTTING DISPATCH
    # ----------------------------

    def run(self):
        """Execute the plotting based on the configured plot_type."""
        if self.plot_type == "1d":
            self._plot_1d()
        elif self.plot_type == "2d":
            self._plot_2d()
        elif self.plot_type == "graph":
            self._plot_graph()
        else:
            raise ValueError(f"Unknown plot_type={self.plot_type}")
