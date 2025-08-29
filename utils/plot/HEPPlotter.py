import matplotlib.pyplot as plt
import matplotlib
import hist
import matplotlib.ticker as mtick
from scipy.stats.distributions import chi2

from scipy.stats import chisquare
import numpy as np
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
    job = (HEPPlotter("CMS")
            .set_output("out/plot")
            .set_labels(xlabel="pT [GeV]", ylabel="Events")
            .set_data(series_dict)
            .set_options(y_log=True, ratio_label="Data/MC"))
    job.run()  # produces the plot

    Parameters
    ----------
    style : str
        The mplhep style to use (default "CMS").
    debug : bool
        If True, print debug information during plotting (default False).
    -----------

    Methods
    -------
    - set_plot_config(figsize=None, lumitext="(13.6 TeV)", formats=None)
    - set_output(output_base)
    - set_labels(xlabel, ylabel="Events", cbar_label="Events", ratio_label="Ratio")
    - set_data(series_dict, plot_type="1d")
    - set_extra_kwargs(**kwargs)
    - set_options(**kwargs)
    - add_ratio_hists(ratio_hists)
    - add_annotation(**kwargs)
    - add_chi_square(**kwargs)
    - add_line(orientation="h", **kwargs)
    - run()
    """

    def __init__(self, style="CMS", debug=False):
        # core settings
        self.style = style
        hep.style.use(style)
        self.debug = debug

        self.figsize = None
        self.lumitext = "(13.6 TeV)"
        self.data_formats = ["png", "pdf", "svg"]

        # inputs
        self.output_base = None
        self.series_dict = None
        self.plot_type = "1d"  # "1d", "2d", "graph"

        # labels
        self.xlabel = None
        self.ylabel = "Events"
        self.cbar_label = "Events"
        self.ratio_label = "Ratio"

        # log scales
        self.y_log = False
        self.x_log = False
        self.y_log_ratio = False
        self.cbar_log = False

        self.reference_to_den = True
        self.grid = True

        # legend
        self.legend = True
        self.legend_loc = "best"
        self.legend_ratio = False
        self.legend_ratio_loc = "best"

        self.set_ylim = True
        self.extra_kwargs = {}

        self.plot_chi_square = None

        # internal
        self._ratio_hists = {}
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

    def set_labels(
        self, xlabel, ylabel="Events", cbar_label="Events", ratio_label="Ratio"
    ):
        """Set the x and y axis labels."""
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.cbar_label = cbar_label
        self.ratio_label = ratio_label
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
        """Generic options setter (y_log, legend, grid...)."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        return self

    def add_ratio_hists(self, ratio_hists):
        """Add precomputed ratio histograms to be plotted on the ratio subplot.
        ratio_hists: dict of histograms with the same keys as series_dict
        """
        self._ratio_hists = ratio_hists
        return self

    def add_annotation(self, **kwargs):
        """Annotation kwargs go directly to ax.text() with transform=ax.transAxes"""
        self._annotations.append(kwargs)
        return self

    def add_chi_square(self, **kwargs):
        """Add chi-square text to the plot (only for 1D with ratio).
        kwargs: passed directly to ax.text()
        """
        self.plot_chi_square = True
        self._chi_square_style = kwargs
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

    def _apply_cms_labels(self, ax):
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

    def _apply_lines(self, ax):
        """Add all stored horizontal/vertical lines to the axes."""
        for orient, kwargs in self._lines:
            if orient == "h":
                ax.axhline(**kwargs)
            else:
                ax.axvline(**kwargs)

    def _apply_chi_square(self, ax, hist_1d, ref_hist, index, style):
        """Compute and add chi-square text to the plot."""

        # if bin is empty in one of the two histograms, set it to nan
        # hist_1d = hist_1d.copy()
        # ref_hist = ref_hist.copy()
        # for i in range(len(hist_1d.values())):
        #     if hist_1d.values()[i] == 0 or ref_hist.values()[i] == 0:
        #         hist_1d.values()[i] = np.nan
        #         ref_hist.values()[i] = np.nan

        # # compute the chi square between the two histograms (divide by the error on data)
        # chi2_value, pvalue = chisquare(
        #     f_obs=hist_1d.values(),
        #     f_exp=ref_hist.values(),
        #     sum_check=False,
        #     nan_policy="omit",
        # )

        chi2_value = np.sum(
            (hist_1d.values() - ref_hist.values()) ** 2 / (hist_1d.variance())
        )
        ndof = len(hist_1d.values()) - 1

        chi2_norm = chi2_value / (  ndof if ndof > 0 else 1)
        
        pvalue=chi2.sf(chi2_value, ndof)

        self.chi_square_text = (
            r"$\chi^2$/ndof= {:.3f},".format(chi2_norm) + f"  p-value= {pvalue:.3f}"
        )

        color_chi2 = self._chi_square_style.get(
            "color",
            style.get("color", style.get("edgecolor", style.get("facecolor"))),
        )

        # plot the chi2 text
        ax.text(
            self._chi_square_style.get("x", 0.05),
            self._chi_square_style.get("y", 0.95) - index * 0.05,
            self.chi_square_text,
            transform=ax.transAxes,
            fontsize=self._chi_square_style.get("fontsize", 20),
            color=color_chi2,
        )

    def _color_handler(self, histtype, style, kwargs, use_lists=False):
        if use_lists:
            if histtype == "fill":
                if "edgecolor" not in kwargs:
                    kwargs.update(
                        {
                            "edgecolor": [style.get("edgecolor", style.get("color"))],
                            "facecolor": [style.get("facecolor", style.get("color"))],
                            "alpha": [style.get("alpha", 0.5)],
                        }
                    )
                else:
                    kwargs["edgecolor"].append(
                        style.get("edgecolor", style.get("color"))
                    )
                    kwargs["facecolor"].append(
                        style.get("facecolor", style.get("color"))
                    )
                    kwargs["alpha"].append(style.get("alpha", 0.5))
            else:
                if "color" not in kwargs:
                    kwargs.update(
                        {
                            "color": [style.get("color")],
                        }
                    )
                else:
                    kwargs["color"].append(style.get("color"))
        else:
            if histtype == "fill":
                kwargs.update(
                    {
                        "edgecolor": style.get("edgecolor", style.get("color")),
                        "facecolor": style.get("facecolor", style.get("color")),
                        "alpha": style.get("alpha", 0.5),
                    }
                )
            else:
                kwargs.update(
                    {
                        "color": style.get("color"),
                    }
                )

    def _stack_plot_order(self, hist_1d, style):
        # reorder the histograms to be plotted in stack order (first is bottom)
        if isinstance(hist_1d, list):
            # get the sorting indexes
            idxes = sorted(
                range(len(hist_1d)),
                key=lambda i: hist_1d[i].integrate(name=hist_1d[i].axes[0].name).value,
            )
            hist_1d = [hist_1d[i] for i in idxes]
            # orer the elements in the style dict if they are lists
            for key in style:
                if isinstance(style[key], list):
                    style[key] = [style[key][i] for i in idxes]

        return hist_1d, style

    # ----------------------------
    # IMPLEMENTATIONS
    # ----------------------------

    def _plot_1d(self):
        """Plot 1D histograms with optional ratio plot."""
        ratio_plot, ref_name = self._validate_inputs(self.series_dict)
        fig, ax, ax_ratio = self._create_figure(ratio_plot)

        hist_1d_stack = []
        kwargs_stack = {}
        legend_name_stack = []

        ref_hist = self.series_dict[ref_name]["data"] if ref_name else None

        for index, (name, props) in enumerate(self.series_dict.items()):

            hist_1d = props["data"]
            style = props.get("style", {})
            hist_1d, style = self._stack_plot_order(hist_1d.copy(), style.copy())

            is_ref = style.get("is_reference", False)

            legend_name = (
                style.get("legend_name", name)
                if style.get("appear_in_legend", True)
                else None
            )
            legend_name_ratio = (
                style.get("legend_name_ratio", name)
                if style.get("appear_in_legend_ratio", True)
                else None
            )

            kwargs = self.extra_kwargs.copy()
            histtype = style.get("histtype", "step")
            stack = style.get("stack", False)
            kwargs.update(
                {
                    "histtype": histtype,
                    "linewidth": style.get("linewidth", 2),
                    "stack": stack,
                }
            )

            self._color_handler(histtype, style, kwargs)

            # draw histogram
            self._plot_histogram(
                ax, legend_name, hist_1d, style.get("plot_errors", True), **kwargs
            )

            if isinstance(hist_1d, list):
                hist_1d = sum(hist_1d)
            # keep the style of the last histogram in the list
            for key in style:
                if isinstance(style[key], list):
                    style[key] = style[key][-1]

            if self.plot_chi_square and ratio_plot and not is_ref:
                self._apply_chi_square(ax, hist_1d, ref_hist, index, style)

            # ratio
            if ratio_plot and ax_ratio is not None:
                if self.reference_to_den:
                    self._plot_ratio(
                        ax_ratio,
                        hist_1d,
                        ref_hist,
                        legend_name_ratio,
                        is_ref,
                        style,
                    )
                else:
                    self._plot_ratio(
                        ax_ratio,
                        ref_hist,
                        hist_1d,
                        legend_name_ratio,
                        is_ref,
                        style,
                    )

        # plot precomputed ratio hists
        if self._ratio_hists and ratio_plot and ax_ratio is not None:
            for name, props in self._ratio_hists.items():
                style = self._ratio_hists[name].get("style", {})
                is_ref = self._ratio_hists[name]["style"].get("is_reference", False)
                ratio_hist = props["data"]
                legend_name_ratio = (
                    style.get("legend_name_ratio", name)
                    if style.get("appear_in_legend_ratio", True)
                    else None
                )
                self._plot_ratio(
                    ax_ratio,
                    None,
                    None,
                    legend_name_ratio,
                    is_ref,
                    style,
                    ratio_hist=ratio_hist,
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
                norm=matplotlib.colors.LogNorm() if self.cbar_log else None,
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
            # extra_kwargs = self.extra_kwargs.copy()
            # extra_kwargs.update(
            #     {
            #         "color": style.get("color"),
            #         "markersize": style.get("markersize", 6),
            #     }
            # )
            # hep.histplot(
            #     (y_values, x_values),
            #     yerr=y_errors,
            #     xerr=x_errors,
            #     histtype="errorbar",
            #     # fmt=style.get("fmt", "o"),
            #     label=name,
            #     color=style.get("color"),
            #     markersize=style.get("markersize"),
            #     **self.extra_kwargs,
            # )
            legend_name = (
                style.get("legend_name", name)
                if style.get("appear_in_legend", True)
                else None
            )
            ax.errorbar(
                y=y_values,
                x=x_values,
                yerr=y_errors,
                xerr=x_errors,
                fmt=style.get("fmt", "o"),
                label=legend_name,
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
            if not isinstance(hist_1d, hist.Hist) and not isinstance(
                hist_1d[0], hist.Hist
            ):
                raise ValueError(f"Expected hist.Hist for {name}, got {type(hist_1d)}")
            if props.get("style", {}).get("is_reference", False):
                if ratio_plot:
                    raise ValueError("Multiple reference histograms found.")
                ratio_plot = True
                ref_name = name
            if isinstance(hist_1d[0], hist.Hist):
                style = props.get("style", {})
                # check that the lists of histograms have the same dimension
                lenght_hists = len(hist_1d)
                for key in style:
                    if isinstance(style[key], list):
                        if len(style[key]) != lenght_hists:
                            raise ValueError(
                                f"Length mismatch in style lists for {name}: expected {lenght_hists}, got {len(style[key])} for key {key}"
                            )

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
                constrained_layout=True,
            )
            return fig, ax, ax_ratio
        else:
            fig, ax = plt.subplots(figsize=self.figsize)
            return fig, ax, None

    def _plot_histogram(self, ax, name, hist_1d, plot_errors, **kwargs):
        """Plot a single 1D histogram on the given axes."""
        hep.histplot(
            hist_1d,
            w2method="sqrt" if plot_errors else None,
            label=name,
            yerr=False if not plot_errors else None,
            xerr=True,
            ax=ax,
            **kwargs,
        )
        ax.set_xlabel("")

    def _plot_ratio(
        self, ax_ratio, hist_num, hist_den, name, is_reference, style, ratio_hist=None
    ):
        """Plot the ratio of hist_1d to ref_hist on the ratio axes."""

        if not ratio_hist:
            bins = hist_num.axes[0].edges
            ratio, err_up, err_down = hep.get_comparison(
                hist_num,
                hist_den,
                comparison="split_ratio" if is_reference else "ratio",
                h1_w2method="sqrt",
            )
        else:
            bins = None
            ratio = ratio_hist
            err_up, err_down = None, None

        color = style.get("color", style.get("edgecolor", style.get("facecolor")))
        histtype_ratio = style.get("histtype_ratio", "errorbar")

        if is_reference:
            ax_ratio.axhline(y=1, linestyle="--", color=color, zorder=0)

            # centers = (bins[1:] + bins[:-1]) / 2
            # ax_ratio.fill_between(
            #     centers, 1 - err_up, 1 + err_up, alpha=0.2, color=color, label=name,zorder=0
            # )
            hep.histplot(
                ratio,
                bins=bins,
                yerr=err_up,
                xerr=True,
                histtype="band",
                label=name,
                ax=ax_ratio,
                facecolor=color,
                zorder=0,
                alpha=0.2,
            )
        else:
            hep.histplot(
                ratio,
                bins=bins,
                yerr=err_up if style.get("plot_errors", True) else None,
                xerr=True,
                histtype=histtype_ratio,
                label=name,
                ax=ax_ratio,
                color=color,
                edges=style.get("edges_ratio", True),
                linewidth=style.get("linewidth", 2),
            )

    def _set_legend(self, ax, pos):
        """Set the legend on the axes."""
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 5:
            ax.legend(loc=pos, ncol=2, fontsize="small")
        else:
            ax.legend(loc=pos)

    def _finalize(self, fig, ax, ax_ratio=None):
        """Final adjustments and saving the figure."""
        if self.legend:
            self._set_legend(ax, self.legend_loc)

        if self.y_log:
            ax.set_yscale("log")
        if self.x_log:
            ax.set_xscale("log")

        if self.grid:
            ax.grid()

        if ax_ratio:
            ax_ratio.set_xlabel(self.xlabel)
            ax_ratio.set_ylabel(self.ratio_label)
            if self.grid:
                ax_ratio.grid()
            if self.y_log_ratio:
                ax_ratio.set_yscale("log")
            if not self.y_log_ratio and self.set_ylim:
                ax_ratio.set_ylim(0.5, 1.5)
            if self.legend_ratio:
                self._set_legend(ax_ratio, self.legend_ratio_loc)
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
                    if not self.y_log
                    else ax.get_ylim()[1] ** (1.7)
                ),
                bottom=(0 if not self.y_log else 1e-1),
            )

        if self.plot_type == "2d":
            # label colorbar
            cbar = ax.collections[0].colorbar
            cbar.set_label(self.cbar_label)

        self._apply_annotations(ax)
        self._apply_lines(ax)

        self._apply_cms_labels(ax)
        self._save(fig)
        self._close_fig(fig)

    # ----------------------------
    # PLOTTING DISPATCH
    # ----------------------------

    def run(self):
        """Execute the plotting based on the configured plot_type."""
        if self.debug:
            print(
                f"Running HEPPlotter with plot_type={self.plot_type}, output_base={self.output_base}"
            )
        if self.plot_type == "1d":
            self._plot_1d()
        elif self.plot_type == "2d":
            self._plot_2d()
        elif self.plot_type == "graph":
            self._plot_graph()
        else:
            raise ValueError(f"Unknown plot_type={self.plot_type}")
