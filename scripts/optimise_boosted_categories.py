"""Optimise the boosted category cuts (cat1, catvbf, cat2, cat3) on a ggF and a VBF discriminant
plus the subleading TXbb score.

The category logic mirrors configs/HH4b_common/custom_cut_functions_common.py (hh4b_boosted_category_*),
with the BDT scores replaced by a ggF and a VBF discriminant:
    cat1   : txbb > txbb_c1 & D_ggF > bdt_ggf_c1
    catvbf : !cat1 & txbb > txbb_vbf & D_VBF > bdt_vbf
    cat2   : !cat1 & !catvbf & ( (txbb_c2a < txbb < txbb_c1 & D_ggF > bdt_ggf_c1) | (txbb > txbb_c1 & bdt_ggf_c2b < D_ggF < bdt_ggf_c1) )
    cat3   : !cat1 & !catvbf & !cat2 & txbb > txbb_c2a & D_ggF > bdt_ggf_bkg

Processes:
    - ggF / VBF signal and ttbar: MC in boosted_signal_region_D (TXbb not morphed)
    - QCD: data in boosted_control_region_B_postW (morphed TXbb, morphing weights), normalised with
      (sum(data in boosted_signal_sideband_region_C) - sum(ttbar MC in boosted_signal_sideband_region_C))
      / sum(data in boosted_control_sideband_region_A_postW), i.e. the same estimate as in build_datacards.py

Figure of merit (lower is better), evaluated per category (and optionally per mass bin), combined over all categories:
    - limit       : expected 95% CL upper limit on the signal strength from the Asimov exclusion significance
                    (generalises 2*sqrt(B)/S to small B and includes the background uncertainty)
    - significance: 1 / combined Asimov discovery significance
    - simple      : 2 / sqrt(sum_i S_i^2 / B_i), i.e. the combination of 2*sqrt(B)/S over the categories
The ggF and VBF figures of merit are combined as F_ggF^(1-w) * F_VBF^w (w = --vbf-weight).

The cuts are optimised by coordinate descent over parameter pairs on a grid. The events are binned once
on the grid edges, so each evaluation is only a sum over bins.
"""
import argparse
import json
import logging
import os

import numpy as np

from utils_configs.plot.get_columns_from_files import get_columns_from_files
from utils_configs.plot.weighted_quantile import weighted_quantile


logging.basicConfig(format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()


parser = argparse.ArgumentParser(description="Optimise the boosted category cuts on the sig/bkg DNN scores and the subleading TXbb", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--input-data", type=str, required=True, help="Directory with the coffea output files (one per dataset)")
parser.add_argument("-o", "--output", type=str, default="./category_optimisation", help="Output directory")
parser.add_argument(
    "-d",
    "--discriminant",
    choices=["ratio", "raw", "bdt"],
    default="ratio",
    help="ratio: X/(X+QCD+TTbar) for X=ggF,VBF; raw: the softmax outputs themselves (also separate ggF from VBF); "
         "bdt: the current boosted BDT scores (validation / baseline)",
)
parser.add_argument("-m", "--metric", choices=["limit", "significance", "simple"], default="limit", help="Figure of merit")
parser.add_argument("-w", "--vbf-weight", type=float, default=0.5, help="Weight w of the VBF figure of merit: F = F_ggF^(1-w) * F_VBF^w")
parser.add_argument("--ggf-pattern", type=str, default="c2-0p00-kl-1p00-kt-1p00", help="Substring selecting the ggF signal dataset(s)")
parser.add_argument("--vbf-pattern", type=str, default="CV-1-C2V-1-C3-1_", help="Substring selecting the VBF signal dataset(s), e.g. 'CV-1-C2V-0-C3-1_' for kappa_2V=0")
parser.add_argument("--other-signal-as-bkg", action="store_true", default=False, help="Count the other HH signal as background when computing the figure of merit of one signal")
parser.add_argument("--mass-column", type=str, default="HiggsSubLeading_mass", help="Fit variable, used for the mass bins")
parser.add_argument("--mass-bins", type=int, default=1, help="Number of equal-width bins in the fit variable (1: counting experiment)")
parser.add_argument("--bkg-syst", type=float, default=0.0, help="Relative systematic uncertainty on the background per bin")
parser.add_argument("--no-bkg-stat", action="store_true", default=False, help="Do not include the background MC/data statistical uncertainty in the figure of merit")
parser.add_argument("--min-qcd-events", type=int, default=10, help="Minimum number of raw QCD (B_postW data) events per category")
parser.add_argument("--min-bkg", type=float, default=0.0, help="Minimum weighted total background per category")
parser.add_argument("--n-score-points", type=int, default=40, help="Number of grid points for the score cuts (quantiles of the signal)")
parser.add_argument("--txbb-grid", type=float, nargs=3, default=[0.65, 0.995, 0.005], help="TXbb grid: start stop step")
parser.add_argument("--scan-cat3", action="store_true", default=False, help="Also optimise the loose ggF cut of cat3 (otherwise fixed to the loosest grid point)")
parser.add_argument("--max-iter", type=int, default=10, help="Maximum number of coordinate-descent iterations")
parser.add_argument("--holdout", action="store_true", default=False, help="Optimise on even event numbers, report on odd ones (weights x2) to check for overtuning")
parser.add_argument("--cache", type=str, default=None, help="npz file with the loaded events: read if it exists, else written after loading (delete it when the inputs change)")
parser.add_argument("--region-setting", choices=["40GeV", "50GeV"], default="50GeV", help="Region setting for the BDT baseline cuts")
parser.add_argument("--signal-region", type=str, default="boosted_signal_region_D")
parser.add_argument("--qcd-region", type=str, default="boosted_control_region_B_postW")
parser.add_argument("--norm-num-region", type=str, default="boosted_signal_sideband_region_C")
parser.add_argument("--norm-denom-region", type=str, default="boosted_control_sideband_region_A_postW")
args = parser.parse_args()

CATEGORIES = ["none", "cat1", "catvbf", "cat2", "cat3"]
# Parameter names follow _get_params in configs/HH4b_common/custom_cuts_common.py, so that the optimised cuts
# can be copied directly into the `optimised_signal_cuts` config option
PARAMS = ["txbb_c1", "bdt_ggf_c1", "txbb_vbf", "bdt_vbf", "txbb_c2a", "bdt_ggf_c2b", "bdt_ggf_bkg"]
# Which grid axis each parameter lives on
PARAM_AXIS = {"txbb_c1": "t", "txbb_vbf": "t", "txbb_c2a": "t", "bdt_ggf_c1": "g", "bdt_ggf_c2b": "g", "bdt_ggf_bkg": "g", "bdt_vbf": "v"}
Z_CL = 1.96  # Z of the expected 95% CL limit (2*sqrt(B)/S convention uses ~2)

SCORE_COLUMNS = {
    "ratio": ["events_sig_bkg_dnn_score_ggF", "events_sig_bkg_dnn_score_VBF", "events_sig_bkg_dnn_score_QCD", "events_sig_bkg_dnn_score_TTbar"],
    "raw": ["events_sig_bkg_dnn_score_ggF", "events_sig_bkg_dnn_score_VBF"],
    "bdt": ["events_boosted_bdt_score", "events_boosted_bdt_vbf_score"],
}


# ======================= Loading =======================
def load_group(files, categories, columns):
    """Load the columns of a group of datasets, merged over the datasets: {category: {column: array}}."""
    if not files:
        return {}
    cat_col, _ = get_columns_from_files(
        files, "nominal", lambda c: c in columns, category_filter=lambda c: c in categories,
    )
    for category in categories:
        if category not in cat_col:
            raise ValueError(f"Category {category} not found for {files[0]} (and others)")
        missing = [c for c in columns if c not in cat_col[category]]
        if missing:
            raise ValueError(f"Columns {missing} not found in category {category}. Available: {sorted(cat_col[category])}")
    return cat_col


def get_discriminants(cols):
    if args.discriminant == "bdt":
        return cols["events_boosted_bdt_score"], cols["events_boosted_bdt_vbf_score"]
    ggf, vbf = cols["events_sig_bkg_dnn_score_ggF"], cols["events_sig_bkg_dnn_score_VBF"]
    if args.discriminant == "raw":
        return ggf, vbf
    bkg = cols["events_sig_bkg_dnn_score_QCD"] + cols["events_sig_bkg_dnn_score_TTbar"]
    return ggf / (ggf + bkg), vbf / (vbf + bkg)


def build_process(cols, txbb_column, scale=1.0):
    d_ggf, d_vbf = get_discriminants(cols)
    return {
        "t": np.asarray(cols[txbb_column], dtype=np.float64),
        "g": np.asarray(d_ggf, dtype=np.float64),
        "v": np.asarray(d_vbf, dtype=np.float64),
        "m": np.asarray(cols[args.mass_column], dtype=np.float64),
        "w": np.asarray(cols["weight"], dtype=np.float64) * scale,
        "event": np.asarray(cols["events_event"]),
    }


def load_processes():
    files = sorted(os.path.join(args.input_data, f) for f in os.listdir(args.input_data) if f.endswith(".coffea") and f != "output_all.coffea")
    groups = {
        "ggF": [f for f in files if "GluGlu" in f and args.ggf_pattern in f],
        "VBF": [f for f in files if "VBFHH" in f and args.vbf_pattern in f],
        "TT": [f for f in files if "TTto" in f],
        "DATA": [f for f in files if "DATA" in f],
    }
    for name, group in groups.items():
        logger.info(f"{name}: {[os.path.basename(f) for f in group]}")
        if not group and name != "TT":
            raise ValueError(f"No files found for {name}")

    columns = ["weight", "events_event", args.mass_column, "HiggsSubLeading_btagBBTXbb", "HiggsSubLeading_btagBBTXbb_morph"] + SCORE_COLUMNS[args.discriminant]
    processes = {}
    for name in ["ggF", "VBF", "TT"]:
        cat_col = load_group(groups[name], [args.signal_region], columns)
        if cat_col:
            processes[name] = build_process(cat_col[args.signal_region], "HiggsSubLeading_btagBBTXbb")

    # QCD: data-driven from the reweighted B region, normalised with (C - ttbar MC in C) / A_postW
    norm = load_group(groups["DATA"], [args.norm_num_region, args.norm_denom_region], ["weight"])
    sum_num, sum_denom = norm[args.norm_num_region]["weight"].sum(), norm[args.norm_denom_region]["weight"].sum()
    norm_tt = load_group(groups["TT"], [args.norm_num_region], ["weight"])
    sum_tt = norm_tt[args.norm_num_region]["weight"].sum() if norm_tt else 0.0
    norm_factor = (sum_num - sum_tt) / sum_denom
    logger.info(f"QCD normalisation ({args.norm_num_region} - TT) / {args.norm_denom_region} = "
                f"({sum_num:.1f} - {sum_tt:.1f}) / {sum_denom:.1f} = {norm_factor:.5f}")
    cat_col = load_group(groups["DATA"], [args.qcd_region], columns)
    # In the control region the TXbb is morphed into the signal region range (as in the category cuts)
    processes["QCD"] = build_process(cat_col[args.qcd_region], "HiggsSubLeading_btagBBTXbb_morph", scale=norm_factor)
    for name, proc in processes.items():
        logger.info(f"{name}: {len(proc['w'])} events, yield {proc['w'].sum():.3f}")
    return processes


def select_events(proc, parity):
    """Select even (parity=0) or odd (parity=1) event numbers, doubling the weights. parity=None: all events."""
    if parity is None:
        return proc
    mask = (proc["event"] % 2) == parity
    out = {k: v[mask] for k, v in proc.items()}
    out["w"] = out["w"] * 2
    return out


# ======================= Categories =======================
def assign_categories(t, g, v, cuts):
    """Category index (see CATEGORIES) for each entry. Works on values with value cuts, or on grid indices
    with cuts = index - 0.5 (then x > cut <=> index >= k)."""
    cat1 = (t > cuts["txbb_c1"]) & (g > cuts["bdt_ggf_c1"])
    catvbf = ~cat1 & (t > cuts["txbb_vbf"]) & (v > cuts["bdt_vbf"])
    cat2 = ~cat1 & ~catvbf & (
        ((t > cuts["txbb_c2a"]) & (t < cuts["txbb_c1"]) & (g > cuts["bdt_ggf_c1"]))
        | ((t > cuts["txbb_c1"]) & (g > cuts["bdt_ggf_c2b"]) & (g < cuts["bdt_ggf_c1"]))
    )
    cat3 = ~cat1 & ~catvbf & ~cat2 & (t > cuts["txbb_c2a"]) & (g > cuts["bdt_ggf_bkg"])
    category = np.zeros(np.shape(t), dtype=np.int64)
    for i, mask in enumerate([cat1, catvbf, cat2, cat3], start=1):
        category[mask] = i
    return category


def event_yields(processes, cuts, mass_edges):
    """Yields with exact (unbinned) cuts: {proc: (sumw, sumw2, n)} each of shape (n_categories, n_mass_bins)."""
    n_mass = len(mass_edges) - 1
    yields = {}
    for name, proc in processes.items():
        category = assign_categories(proc["t"], proc["g"], proc["v"], cuts)
        idx = category * n_mass + mass_index(proc["m"], mass_edges)
        size = len(CATEGORIES) * n_mass
        yields[name] = tuple(
            np.bincount(idx, weights=w, minlength=size).reshape(len(CATEGORIES), n_mass)
            for w in (proc["w"], proc["w"] ** 2, np.ones_like(proc["w"]))
        )
    return yields


def mass_index(m, mass_edges):
    return np.clip(np.searchsorted(mass_edges, m, side="right") - 1, 0, len(mass_edges) - 2)


class BinnedYields:
    """Events binned on the cut grid, so that any cut on the grid is a sum over bins."""

    def __init__(self, processes, grids, mass_edges):
        self.names = list(processes)
        self.n_mass = len(mass_edges) - 1
        # Bin k of an axis covers [grid[k-1], grid[k]), bin 0 is everything below grid[0]
        shape = tuple(len(grids[a]) + 1 for a in "tgv")
        it, ig, iv = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
        self.index = {"t": it.ravel(), "g": ig.ravel(), "v": iv.ravel()}
        n_bins = int(np.prod(shape))
        columns = []
        for name in self.names:
            proc = processes[name]
            flat = np.ravel_multi_index(
                tuple(np.searchsorted(grids[a], proc[a], side="right") for a in "tgv"), shape
            ) * self.n_mass + mass_index(proc["m"], mass_edges)
            for w in (proc["w"], proc["w"] ** 2, np.ones_like(proc["w"])):
                columns.append(np.bincount(flat, weights=w, minlength=n_bins * self.n_mass).reshape(n_bins, self.n_mass))
        # (n_bins, n_proc * 3 * n_mass)
        self.histogram = np.concatenate(columns, axis=1)
        self.n_bins = n_bins

    def yields(self, cut_indices):
        cuts = {p: cut_indices[p] - 0.5 for p in PARAMS}
        category = assign_categories(self.index["t"], self.index["g"], self.index["v"], cuts)
        one_hot = np.zeros((len(CATEGORIES), self.n_bins))
        one_hot[category, np.arange(self.n_bins)] = 1.0
        summed = (one_hot @ self.histogram).reshape(len(CATEGORIES), len(self.names), 3, self.n_mass)
        return {name: tuple(summed[:, i, j, :] for j in range(3)) for i, name in enumerate(self.names)}


# ======================= Figure of merit =======================
def z_discovery(s, b, var):
    """Squared Asimov discovery significance with background uncertainty (Cowan)."""
    s, b, var = np.broadcast_arrays(s, b, var)
    z2 = np.zeros_like(s)
    ok = (b > 0) & (s > 0)
    no_unc = ok & (var <= 1e-12 * b**2)
    z2[no_unc] = 2 * ((s + b) * np.log1p(s / b) - s)[no_unc]
    unc = ok & ~no_unc
    s_, b_, v_ = s[unc], b[unc], var[unc]
    # Profile likelihood ratio test & Asimov
    z2[unc] = 2 * ((s_ + b_) * np.log((s_ + b_) * (b_ + v_) / (b_**2 + (s_ + b_) * v_))
                   - b_**2 / v_ * np.log1p(v_ * s_ / (b_ * (b_ + v_))))
    return np.clip(z2, 0, None)


def z_exclusion(s, b, var):
    """Squared Asimov exclusion significance (of s+b, assuming b only) with background uncertainty."""
    s, b, var = np.broadcast_arrays(s, b, var)
    z2 = np.zeros_like(s)
    ok = (b > 0) & (s > 0)
    no_unc = ok & (var <= 1e-12 * b**2)
    z2[no_unc] = 2 * (s - b * np.log1p(s / b))[no_unc]
    unc = ok & ~no_unc
    s_, b_, v_ = s[unc], b[unc], var[unc]
    x = np.sqrt((s_ + b_) ** 2 - 4 * s_ * b_ * v_ / (b_ + v_))
    z2[unc] = (2 * (s_ - b_ * np.log((b_ + s_ + x) / (2 * b_)) - b_**2 / v_ * np.log((b_ - s_ + x) / (2 * b_)))
               - (b_ + s_ - x) * (1 + b_ / v_))
    return np.clip(z2, 0, None)


def expected_limit(s, b, var):
    """Signal strength mu for which the combined Asimov exclusion significance is Z_CL."""
    if np.sum(s) <= 0:
        return np.inf
    lo, hi = 0.0, 1.0
    while np.sum(z_exclusion(hi * s, b, var)) < Z_CL**2:
        hi *= 2
        if hi > 1e9:
            return np.inf
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if np.sum(z_exclusion(mid * s, b, var)) < Z_CL**2:
            lo = mid
        else:
            hi = mid
    return hi


def figure_of_merit(s, b, var):
    """Lower is better. s, b, var have shape (n_categories, n_mass_bins) without the 'none' category."""
    if args.metric == "limit":
        return expected_limit(s, b, var)
    if args.metric == "significance":
        z2 = np.sum(z_discovery(s, b, var))
        return 1 / np.sqrt(z2) if z2 > 0 else np.inf
    ok = b > 0
    sum_s2_b = np.sum(s[ok] ** 2 / b[ok])
    return 2 / np.sqrt(sum_s2_b) if sum_s2_b > 0 else np.inf


def passes_requirements(yields):
    bkg = yields["QCD"][0][1:] + (yields["TT"][0][1:] if "TT" in yields else 0)
    return not (np.any(yields["QCD"][2][1:].sum(axis=1) < args.min_qcd_events) or np.any(bkg.sum(axis=1) < args.min_bkg))


def evaluate(yields, enforce=True):
    """Combined figure of merit and the per-signal ones. Infinite if a category fails the statistics requirements (and enforce)."""
    qcd_w, qcd_w2 = yields["QCD"][0][1:], yields["QCD"][1][1:]
    bkg, bkg_w2 = qcd_w.copy(), qcd_w2.copy()
    if "TT" in yields:
        bkg += yields["TT"][0][1:]
        bkg_w2 += yields["TT"][1][1:]
    if enforce and not passes_requirements(yields):
        return np.inf, np.inf, np.inf
    foms = {}
    for signal, other in [("ggF", "VBF"), ("VBF", "ggF")]:
        b, var = bkg, (0 if args.no_bkg_stat else bkg_w2) + (args.bkg_syst * bkg) ** 2
        if args.other_signal_as_bkg:
            b = b + yields[other][0][1:]
        foms[signal] = figure_of_merit(yields[signal][0][1:], b, var)
    combined = foms["ggF"] ** (1 - args.vbf_weight) * foms["VBF"] ** args.vbf_weight
    return combined, foms["ggF"], foms["VBF"]


# ======================= Optimisation =======================
def make_grids(processes):
    start, stop, step = args.txbb_grid
    efficiencies = np.linspace(0.99, 0.01, args.n_score_points)
    grids = {"t": np.round(np.arange(start, stop + 0.5 * step, step), 6)}
    for axis, signal in [("g", "ggF"), ("v", "VBF")]:
        # Cut values at fixed signal efficiencies: the scores pile up close to 1 for signal
        values, weights = processes[signal][axis], processes[signal]["w"]
        grids[axis] = np.unique(weighted_quantile(values, 1 - efficiencies, weights=weights))
    return grids


def initial_cuts(grids, processes):
    if args.discriminant == "bdt":
        # Baseline: the BDT cuts the input was produced with, else the current ones in custom_cuts_common.py
        try:
            with open(os.path.join(args.input_data, "config.json")) as f:
                category_cuts = json.load(f)["categories"]["categories"][f"{args.signal_region}_cat1"]
            params = next(c["params"] for c in category_cuts if c["name"] == "hh4b_boosted_category_1")
            logger.info("Starting from the BDT cuts in config.json of the input")
        except (OSError, KeyError, StopIteration) as e:
            from configs.HH4b_common.custom_cuts_common import _get_params
            logger.info(f"No BDT cuts found in config.json of the input ({e!r}), starting from _get_params({args.region_setting})")
            params = _get_params(args.region_setting)
        return {p: params[p] for p in PARAMS}

    def at_eff(axis, signal, eff):
        return weighted_quantile(processes[signal][axis], [1 - eff], weights=processes[signal]["w"])[0]
    return {
        "txbb_c1": 0.95, "bdt_ggf_c1": at_eff("g", "ggF", 0.3), "txbb_vbf": 0.85, "bdt_vbf": at_eff("v", "VBF", 0.3),
        "txbb_c2a": 0.85, "bdt_ggf_c2b": at_eff("g", "ggF", 0.6), "bdt_ggf_bkg": grids["g"][0],
    }


def to_indices(cuts, grids):
    """Snap value cuts to the nearest grid point: x > cut <=> bin index >= k + 1, cut = grid[k]."""
    return {p: int(np.argmin(np.abs(grids[PARAM_AXIS[p]] - cuts[p]))) + 1 for p in PARAMS}


def to_values(indices, grids):
    return {p: float(grids[PARAM_AXIS[p]][indices[p] - 1]) for p in PARAMS}


def valid(indices):
    return indices["txbb_c2a"] <= indices["txbb_c1"] and indices["bdt_ggf_bkg"] <= indices["bdt_ggf_c2b"] <= indices["bdt_ggf_c1"]


def optimise(binned, grids, start):
    pairs = [("txbb_c1", "bdt_ggf_c1"), ("txbb_vbf", "bdt_vbf"), ("txbb_c2a", "bdt_ggf_c2b")]
    if args.scan_cat3:
        pairs.append(("bdt_ggf_bkg",))
    best = dict(start)
    best_fom = evaluate(binned.yields(best))[0]
    logger.info(f"Start: F = {best_fom:.4g}")
    for iteration in range(args.max_iter):
        improved = False
        for pair in pairs:
            ranges = [range(1, len(grids[PARAM_AXIS[p]]) + 1) for p in pair]
            for point in np.array(np.meshgrid(*ranges, indexing="ij")).reshape(len(pair), -1).T:
                trial = dict(best, **{p: int(k) for p, k in zip(pair, point)})
                if not valid(trial):
                    continue
                fom = evaluate(binned.yields(trial))[0]
                if fom < best_fom * (1 - 1e-6):
                    best, best_fom, improved = trial, fom, True
            logger.info(f"Iteration {iteration}, scanned {pair}: F = {best_fom:.4g}, cuts {to_values(best, grids)}")
        if not improved:
            break
    return best


def report(title, processes, cuts, mass_edges):
    yields = event_yields(processes, cuts, mass_edges)
    combined, fom_ggf, fom_vbf = evaluate(yields, enforce=False)
    passes = passes_requirements(yields)
    lines = [
        f"===== {title} =====",
        f"F = {combined:.4g} (ggF: {fom_ggf:.4g}, VBF: {fom_vbf:.4g}, metric: {args.metric})"
        + ("" if passes else "  WARNING: fails the minimum background statistics requirement"),
        "cuts: " + ", ".join(f"{p}={cuts[p]:.5g}" for p in PARAMS),
    ]
    header = f"{'category':>8} {'ggF':>9} {'VBF':>9} {'QCD':>10} {'+-':>8} {'n_QCD':>7} {'TT':>9} {'S_ggF/sqrtB':>11} {'S_VBF/sqrtB':>11}"
    lines.append(header)
    table = {}
    for i, category in enumerate(CATEGORIES[1:], start=1):
        row = {name: float(yields[name][0][i].sum()) for name in yields}
        qcd_unc = float(np.sqrt(yields["QCD"][1][i].sum()))
        n_qcd = int(yields["QCD"][2][i].sum())
        b = row["QCD"] + row.get("TT", 0.0)
        lines.append(
            f"{category:>8} {row['ggF']:9.4f} {row['VBF']:9.4f} {row['QCD']:10.3f} {qcd_unc:8.3f} {n_qcd:7d} {row.get('TT', 0.0):9.4f} "
            f"{row['ggF'] / np.sqrt(b) if b > 0 else np.nan:11.4f} {row['VBF'] / np.sqrt(b) if b > 0 else np.nan:11.4f}"
        )
        table[category] = row | {"QCD_unc": qcd_unc, "n_QCD": n_qcd}
    logger.info("\n" + "\n".join(lines))
    return {"F": combined, "F_ggF": fom_ggf, "F_VBF": fom_vbf, "passes_requirements": passes, "cuts": cuts, "yields": table}


if __name__ == "__main__":
    CACHE_KEYS = ["discriminant", "mass_column", "ggf_pattern", "vbf_pattern", "signal_region",
              "qcd_region", "norm_num_region", "norm_denom_region"]
    cache_meta = json.dumps({k: getattr(args, k) for k in CACHE_KEYS}, sort_keys=True)

    if args.cache and os.path.exists(args.cache):
        logger.info(f"Reading the events from the cache {args.cache}")
        with np.load(args.cache) as cache:
            stored = str(cache["__meta__"]) if "__meta__" in cache.files else None
            if stored != cache_meta:
                raise ValueError(f"Cache {args.cache} was made with different settings:\n"
                                 f"  cache: {stored}\n  now:   {cache_meta}\n"
                                 f"Delete it or use another --cache file.")
            processes_all = {}
            for key in cache.files:
                if key == "__meta__":
                    continue
                name, field = key.split("/")
                processes_all.setdefault(name, {})[field] = cache[key]
    else:
        processes_all = load_processes()
        if args.cache:
            np.savez(args.cache, __meta__=cache_meta,
                     **{f"{name}/{field}": arr for name, proc in processes_all.items() for field, arr in proc.items()})
            logger.info(f"Saved the events to the cache {args.cache}")

    all_mass = np.concatenate([p["m"] for p in processes_all.values()])
    mass_edges = np.linspace(all_mass.min(), all_mass.max() + 1e-6, args.mass_bins + 1)

    if args.holdout:
        processes_opt = {n: select_events(p, 0) for n, p in processes_all.items()}
        processes_test = {n: select_events(p, 1) for n, p in processes_all.items()}
    else:
        processes_opt = processes_all

    grids = make_grids(processes_opt)
    logger.info(f"Grid sizes: TXbb {len(grids['t'])}, ggF {len(grids['g'])}, VBF {len(grids['v'])}")
    binned = BinnedYields(processes_opt, grids, mass_edges)

    start_cuts = initial_cuts(grids, processes_opt)
    results = {"args": vars(args), "start": report("Start (exact cuts)", processes_all, start_cuts, mass_edges)}

    best = optimise(binned, grids, to_indices(start_cuts, grids))
    best_cuts = to_values(best, grids)
    results["optimised"] = report("Optimised", processes_all, best_cuts, mass_edges)
    if args.holdout:
        results["optimised_train"] = report("Optimised, optimisation half", processes_opt, best_cuts, mass_edges)
        results["optimised_test"] = report("Optimised, held-out half", processes_test, best_cuts, mass_edges)
        results["start_test"] = report("Start, held-out half", processes_test, start_cuts, mass_edges)

    output_file = os.path.join(args.output, f"optimised_cuts_{args.discriminant}_{args.metric}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=float)
    logger.info(f"Saved results to {output_file}")
