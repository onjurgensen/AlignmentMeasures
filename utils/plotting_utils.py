"""Shared code for the BrainAlign paper: configuration, loaders, analyses and plot functions.

Imported by the analysis notebook (and later the figures notebook):

    %load_ext autoreload
    %autoreload 2
    from plotting_utils import *
    set_paper_style()

Sections
    1. Paths and analysis switches
    2. Names, orders, palettes
    3. Style and saving
    4. Loaders
    5. Correlation maps
    6. PCA comparisons and sweep
    7. ROI profiles (level agreement, relative profiles)
    8. Hierarchical correspondence (depth profiles, best layers)
    9. Layer structure (blocks, layer types, variance decomposition)
   10. PCA curves (supplement)
"""

import os
import pickle
import sys
import warnings
from contextlib import contextmanager
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import seaborn as sns
from IPython.display import display
from scipy.stats import false_discovery_control, spearmanr, ttest_1samp
from scipy.stats import t as t_dist


# =============================================================================
# 1. Paths and analysis switches
# =============================================================================

SCRIPTS_DIR = Path("~/BrainAlign_Scripts.").expanduser()
RESULTS_DIR = Path("/home/c13745859/Documents/BrainAlign_results")

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR   = REPO_ROOT / "figures"
CACHE_DIR = REPO_ROOT / "results_cache"


@contextmanager
def _in_dir(path):
    """Temporarily change the working directory (for code that expects to run from SCRIPTS_DIR)."""
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
with _in_dir(SCRIPTS_DIR):
    from Alignment_Project import util


# Single switch for the abs() decision -- used by all loaders.
TAKE_ABS = True

PCA_COMPONENTS = [2, 5, 10, 25, 50, 100, 500]
PCA_MAIN = 100
PCA_EXCLUDE = ["CKA_debiased"]   # designed for feature-count mismatch; unreliable after PCA
LSM_MEASURES = ["LinearShape_alpha1", "LinearShape_alpha0"]   # always taken from PCA_MAIN

# PCA sensitivity analyses
PCA_MEASURES = ["RSA_euc", "RSA_cor", "CKA"]
N_ORDER = [str(n) for n in PCA_COMPONENTS] + ["none"]
SWEEP_LABELS = {n: ("no PCA" if n == "none" else n) for n in N_ORDER}
# reference condition: standard features ("none"), or PCA-100 for LSM (its main-analysis version)
STAB_REF = {m: ("100" if m == "LinearShape_alpha1" else "none") for m in PCA_MEASURES}

WITHIN = ["model", "brain_depth"]   # main: correlate across layers within each model x depth
DEPTH_BIN_DECIMALS = 1              # normalised layer depth rounded to 0.0, 0.1, ..., 1.0
REL_HOW = "max"                     # relative profiles: best-layer score per region; "mean" as a check
N_BOOT = 1000
BOOT_SEED = 0


# =============================================================================
# 2. Names, orders, palettes
# =============================================================================

MODEL_ORDER = ["alexnet", "resnet50", "convnext_tiny"]
MODEL_NAMES = {
    "alexnet": "AlexNet",
    "resnet50": "ResNet-50",
    "convnext_tiny": "ConvNeXt Tiny",
}

MEASURE_ORDER = [
    "RSA_euc",
    "RSA_cor",
    "CKA",
    "CKA_debiased",
    "LinearShape_alpha1",
    "VERSA_cor",
    "VERSA_euc",
    "Ridge",
    "PLS",
]
MEASURE_NAMES = {
    "RSA_cor": "RSA (cor)",
    "RSA_euc": "RSA (euc)",
    "CKA": "CKA",
    "CKA_debiased": "CKA debiased",
    "PLS": "PLSR",
    "Ridge": "Ridge",
    "VERSA_cor": "VERSA (cor)",
    "VERSA_euc": "VERSA (euc)",
    "LinearShape_alpha1": "LSM",
    "LinearShape_alpha0": "LSM α = 0",
}
MEASURE_GROUPS = {
    "Geometry level": ["RSA_euc", "RSA_cor", "CKA", "CKA_debiased",
                       "LinearShape_alpha1", "VERSA_cor", "VERSA_euc"],
    "Fitted mapping": ["LinearShape_alpha1", "VERSA_cor", "VERSA_euc",
                       "Ridge", "PLS"],
}
MEASURE_MARKERS = dict(zip(MEASURE_ORDER, ["o", "s", "D", "^", "v", "<", ">", "P", "X"]))
RSA_MEASURES = ["RSA_euc", "RSA_cor"]

MODALITY_ORDER = ["ephys", "fMRI", "EEG"]
MODALITY_NAMES = {"ephys": "Ephys", "fMRI": "fMRI", "EEG": "EEG"}
MODALITY_STYLE = {"ephys": ("0.15", "o"), "fMRI": ("0.45", "s"), "EEG": ("0.7", "D")}

DEPTH_ORDER = ["early", "mid", "late"]
ROI_TO_DEPTH = {
    "ephys": {"V1": "early", "V4": "mid", "IT": "late"},
    "fMRI":  {"V1": "early", "V4": "mid", "FFA": "late"},
}
EEG_BINS = {                       # seconds; labels used in the paper
    "50-150 ms":  (0.05, 0.15),
    "150-300 ms": (0.15, 0.30),
    "300-450 ms": (0.30, 0.45),
}
EEG_BIN_TO_DEPTH = dict(zip(EEG_BINS, DEPTH_ORDER))

# one palette keyed by internal names; labels kept separate
DEPTH_PALETTE = dict(zip(DEPTH_ORDER, sns.color_palette("deep", 3)))
DEPTH_LABELS = {
    "early": "V1, 50-150 ms",
    "mid":   "V4, 150-300 ms",
    "late":  "IT, FFA, 300-450 ms",
}
DEPTH_LABELS_MULTILINE = {
    "early": "Early\n- V1\n- 50-150 ms",
    "mid":   "Mid\n- V4\n- 150-300 ms",
    "late":  "Late\n- IT\n- FFA\n- 300-450 ms",
}

# models in the layer-type analysis, with layer types in within-block order
LAYER_TYPE_ORDER = {
    "resnet50":      ["Conv", "BN", "ReLU", "Conv1", "BN1", "ReLU-1", "Conv2", "BN2", "Add", "ReLU-2"],
    "convnext_tiny": ["Conv", "LN", "Linear1", "GeLU", "Linear2", "Scale", "Add"],
}
LAYER_TYPE_MODELS = list(LAYER_TYPE_ORDER)


# =============================================================================
# 3. Style and saving
# =============================================================================

PAPER_STYLE = {
    # output
    "figure.dpi": 75, "savefig.dpi": 600, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    # fonts
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "axes.titleweight": "normal",
    # axes
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "axes.grid": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 3, "ytick.major.size": 3,
    # marks
    "lines.linewidth": 1.0, "errorbar.capsize": 0,
    "legend.frameon": False,
}


def set_paper_style(**overrides):
    """Reset matplotlib and apply PAPER_STYLE (plus any overrides)."""
    mpl.rcdefaults()
    mpl.rcParams.update({**PAPER_STYLE, **overrides})


def save_fig(fig, name, formats=("png",)):
    """Save a figure to FIG_DIR in every requested format."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(FIG_DIR / f"{name}.{ext}")


def save_tables(tables, name="analysis_results"):
    """Pickle a dict of derived results (DataFrames, dicts of maps, ...) to CACHE_DIR."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_DIR / f"{name}.pkl", "wb") as f:
        pickle.dump(tables, f)
    print(f"saved {len(tables)} objects to {CACHE_DIR / f'{name}.pkl'}")


def load_tables(name="analysis_results"):
    with open(CACHE_DIR / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


# small shared helpers -------------------------------------------------------

def present_measures(df):
    """Measures present in df, in MEASURE_ORDER."""
    present = set(df["measure"].astype(str).unique())
    return [m for m in MEASURE_ORDER if m in present]


def _display(measures):
    return [MEASURE_NAMES.get(m, m) for m in measures]


def _fmt_short(val):
    """0.12 -> .12, -0.12 -> -.12"""
    s = f"{val:.2f}"
    return s.replace("0.", ".", 1) if s.startswith(("0.", "-0.")) else s


def _blend(color, amount, target=(1, 1, 1)):
    base = mpl.colors.to_rgb(color)
    return tuple(c + (t - c) * amount for c, t in zip(base, target))


def _mean_ci(v, ci=0.95, min_n_ci=3):
    """Mean with t-based CI (n >= min_n_ci) or the range across subjects."""
    n, mean = len(v), v.mean()
    if n >= min_n_ci:
        h = t_dist.ppf(0.5 + ci / 2, n - 1) * v.sem()
        return mean, mean - h, mean + h, n
    return mean, v.min(), v.max(), n


def _summarise_ci(df, keys, value, ci=0.95):
    """One row per group of `keys`: mean, low, high, n of `value` (see _mean_ci)."""
    rows = []
    for k, v in df.groupby(keys, observed=True)[value]:
        mean, low, high, n = _mean_ci(v, ci)
        rows.append({**dict(zip(keys, map(str, k))), "mean": mean, "low": low, "high": high, "n": n})
    return pd.DataFrame(rows)


def depth_ticklabels(mod):
    """Brain-depth tick labels for one modality, in DEPTH_ORDER."""
    mapping = EEG_BIN_TO_DEPTH if mod == "EEG" else ROI_TO_DEPTH[mod]
    inv = {v: k for k, v in mapping.items()}
    return [inv[d] for d in DEPTH_ORDER]


def _row_label(ax, text, x=-0.42):
    """Rotated row header to the left of an axis."""
    ax.annotate(text, xy=(x, 0.5), xycoords="axes fraction", rotation=90,
                ha="center", va="center", fontsize=mpl.rcParams["axes.titlesize"],
                annotation_clip=False)


# =============================================================================
# 4. Loaders
# =============================================================================

KEY = ["modality", "model", "subject", "layer", "brain_depth", "measure"]
COLUMNS = KEY + ["score"]
ROW_KEY = ["modality", "model", "subject", "layer", "brain_depth"]
_RESULT_FILES = [("ephys", "ephys_main"), ("fMRI", "fmri_main"), ("EEG", "eeg_main")]


def _read_results(name, n_components=None):
    """Main results, or PCA results at a given number of components."""
    if n_components is None:
        df = pd.read_parquet(RESULTS_DIR / f"{name}.parquet")
        df = df[~df["measure"].isin(LSM_MEASURES)]
        lsm = pd.read_parquet(
            RESULTS_DIR / f"{name}_pca.parquet",
            filters=[("n_pca_components", "==", PCA_MAIN), ("measure", "in", LSM_MEASURES)],
        )
        assert len(lsm), f"{name}_pca: no LSM rows at {PCA_MAIN} components"
        df = pd.concat([df, lsm], ignore_index=True)
    else:
        assert n_components in PCA_COMPONENTS, f"unknown n_components: {n_components}"
        df = pd.read_parquet(
            RESULTS_DIR / f"{name}_pca.parquet",
            filters=[("n_pca_components", "==", n_components)],
        )
        assert len(df), f"{name}_pca: no rows for n_components={n_components}"

    df = df[df["score_type"] != "Mean R2 score"].copy()
    seeds = df["seed"].unique()
    assert len(seeds) == 1, f"{name}: expected one seed, found {seeds}"
    return df


def _drop_duplicates(df, key, name):
    dup_any = df.duplicated(subset=key, keep=False)
    if dup_any.any():
        groups = df[dup_any].groupby(key, observed=True)["score"]
        n_conflict = (groups.nunique() > 1).sum()
        layers = sorted(df.loc[dup_any, "layer"].unique())
        warnings.warn(
            f"{name}: dropping {df.duplicated(subset=key).sum()} duplicate rows "
            f"(keeping first); {n_conflict} groups have DIFFERING scores; layers: {layers}"
        )
    return df[~df.duplicated(subset=key, keep="first")]


def measures_for(n_components):
    """Measures expected in the main results (None) or at a PCA component count."""
    if n_components is None:
        return MEASURE_ORDER
    return [m for m in MEASURE_ORDER if m not in PCA_EXCLUDE]


def _as_ordered(s, order, name):
    unknown = set(s.unique()) - set(order)
    assert not unknown, f"{name}: values not in order list: {unknown}"
    return pd.Categorical(s, categories=order, ordered=True)


def _finalise(df, modality, take_abs, measures=MEASURE_ORDER):
    """Common tail: optional abs, filter measures, enforce schema and category orders."""
    if take_abs:
        df = df.assign(score=df["score"].abs())
    df = df.assign(modality=modality)

    excluded = sorted(set(df["measure"]) - set(measures))
    if excluded:
        print(f"{modality}: excluding measures {excluded}")
    df = df[df["measure"].isin(measures)]

    assert df["brain_depth"].notna().all(), f"{modality}: unmapped ROI/bin"

    df = df[COLUMNS].reset_index(drop=True)
    df["modality"]    = _as_ordered(df["modality"], MODALITY_ORDER, "modality")
    df["model"]       = _as_ordered(df["model"], MODEL_ORDER, "model")
    df["measure"]     = _as_ordered(df["measure"], MEASURE_ORDER, "measure")
    df["brain_depth"] = _as_ordered(df["brain_depth"], DEPTH_ORDER, "brain_depth")
    n_nan = df["score"].isna().sum()
    if n_nan:
        warnings.warn(f"{modality}: {n_nan} NaN scores; by measure: "
                      f"{df[df['score'].isna()].groupby('measure', observed=True).size().to_dict()}")
    return df


def load_ephys(take_abs=TAKE_ABS, n_components=None):
    df = _read_results("ephys_main", n_components)
    df = _drop_duplicates(df, ["model", "subject", "layer", "roi", "measure"], "ephys")
    df["brain_depth"] = df["roi"].map(ROI_TO_DEPTH["ephys"])
    return _finalise(df, "ephys", take_abs, measures_for(n_components))


def load_fmri(take_abs=TAKE_ABS, n_components=None):
    df = _read_results("fmri_main", n_components)
    df = _drop_duplicates(df, ["model", "subject", "layer", "roi", "measure"], "fMRI")

    # average hemispheres
    df["roi"] = df["roi"].str.replace(r"_(lh|rh)$", "", regex=True)
    df = (
        df.groupby(["model", "subject", "layer", "roi", "measure"], observed=True)["score"]
          .agg(score="mean", n_hemi="size")
          .reset_index()
    )
    bad = df[df["n_hemi"] != 2]
    if len(bad):
        warnings.warn(f"fMRI: {len(bad)} rows not averaged over exactly 2 hemispheres")

    df["brain_depth"] = df["roi"].map(ROI_TO_DEPTH["fMRI"])
    return _finalise(df, "fMRI", take_abs, measures_for(n_components))


def load_eeg(take_abs=TAKE_ABS, n_components=None):
    df = _read_results("eeg_main", n_components)
    df = _drop_duplicates(df, ["model", "subject", "layer", "measure", "time_point"], "EEG")

    # bins: [0.05, 0.15], (0.15, 0.30], (0.30, 0.45]
    bounds = list(EEG_BINS.values())
    assert all(a[1] == b[0] for a, b in zip(bounds, bounds[1:])), "EEG bins must be contiguous"
    edges = [bounds[0][0]] + [end for _, end in bounds]

    t = df["time_point"].round(6)                 # guard against float noise at edges
    df["bin"] = pd.cut(t, bins=edges, labels=list(EEG_BINS), right=True, include_lowest=True)
    df = df.dropna(subset=["bin"])

    print("EEG time points per bin:",
          df.groupby("bin", observed=True)["time_point"].nunique().to_dict())

    df = (
        df.groupby(["model", "subject", "layer", "measure", "bin"], observed=True)["score"]
          .mean()
          .reset_index()
    )
    df["brain_depth"] = df["bin"].map(EEG_BIN_TO_DEPTH).astype(str)
    return _finalise(df, "EEG", take_abs, measures_for(n_components))


def load_all(take_abs=TAKE_ABS, n_components=None):
    return pd.concat(
        [load_ephys(take_abs, n_components),
         load_fmri(take_abs, n_components),
         load_eeg(take_abs, n_components)],
        ignore_index=True,
    )


def load_sweep(df_all, measures=PCA_MEASURES, components=PCA_COMPONENTS):
    """`measures` at every PCA component count, plus the standard features as n_components='none'."""
    sweep = []
    for n in components:
        d = load_all(n_components=n)
        sweep.append(d[d["measure"].isin(measures)].assign(n_components=str(n)))
    std = df_all[df_all["measure"].isin([m for m in measures if m != "LinearShape_alpha1"])]
    return pd.concat(sweep + [std.assign(n_components="none")], ignore_index=True)


def check_completeness(df, measures=MEASURE_ORDER):
    """Rows missing from the full model-layer x subject x depth x measure grid."""
    missing = []
    for (mod, model), sub in df.groupby(["modality", "model"], observed=True):
        grid = pd.MultiIndex.from_product(
            [sub["layer"].unique(), sub["subject"].unique(),
             sub["brain_depth"].unique(), measures],
            names=["layer", "subject", "brain_depth", "measure"],
        ).to_frame(index=False)
        obs = sub[["layer", "subject", "brain_depth", "measure"]].astype(str).drop_duplicates()
        m = grid.astype(str).merge(obs, how="left", indicator=True)
        m = m[m["_merge"] == "left_only"].drop(columns="_merge")
        missing.append(m.assign(modality=mod, model=model))
    return pd.concat(missing, ignore_index=True)


def load_variance_explained():
    out = []
    for modality, name in _RESULT_FILES:
        df = pd.read_parquet(
            RESULTS_DIR / f"{name}_pca.parquet",
            columns=["model", "layer", "n_pca_components", "variance_explained"],
        )
        key = ["model", "layer", "n_pca_components"]
        out.append(df.drop_duplicates(key).assign(modality=modality))
    return pd.concat(out, ignore_index=True)


def variance_explained_lookup(df_var_explained):
    """(modality, model, layer, n_components as str) -> fraction of variance explained."""
    ve = df_var_explained.dropna(subset=["n_pca_components"])
    return {(str(r.modality), str(r.model), r.layer, str(int(r.n_pca_components))): r.variance_explained
            for r in ve.itertuples()}


def check_cka_debiased():
    rows = []
    for modality, name in _RESULT_FILES:
        d = pd.read_parquet(
            RESULTS_DIR / f"{name}_pca.parquet",
            filters=[("measure", "==", "CKA_debiased")],
        )
        d = d[d["score_type"] != "Mean R2 score"]
        s = d.groupby("n_pca_components")["score"].agg(
            n="size",
            frac_nan=lambda x: x.isna().mean(),
            frac_out_of_range=lambda x: (x.abs() > 1).mean(),
            median="median",
        )
        rows.append(s.assign(modality=modality))
    return pd.concat(rows).reset_index()


# =============================================================================
# 5. Correlation maps
# =============================================================================

POOL_KEY = ["model", "layer", "brain_depth"]   # rows that get correlated


def subject_cormaps(df, method="spearman", group_by=None):
    """Measure x measure map per subject (averaged over `group_by` groups within subject)."""
    measures = present_measures(df)
    index = [c for c in POOL_KEY if c not in (group_by or [])]

    out = {}
    for subj, sub_s in df.groupby("subject", observed=True):
        parts = sub_s.groupby(group_by, observed=True) if group_by else [(None, sub_s)]
        mats = []
        for _, sub in parts:
            wide = sub.pivot(index=index, columns="measure", values="score")
            wide.columns = wide.columns.astype(str)
            mats.append(wide.reindex(columns=measures).corr(method=method))
        out[subj] = pd.concat(mats).groupby(level=0).mean().loc[measures, measures]
    return out


def measure_cormap(df, method="spearman", group_by=None):
    maps = subject_cormaps(df, method, group_by)
    measures = present_measures(df)
    return pd.concat(maps.values()).groupby(level=0).mean().loc[measures, measures]


def cormaps(df, depth=None, modalities=None, **kwargs):
    """One map per modality; optionally restricted to one brain depth."""
    out = {}
    for mod in modalities or MODALITY_ORDER:
        sub = df[df["modality"] == mod]
        if depth is not None:
            sub = sub[sub["brain_depth"] == depth]
        out[mod] = measure_cormap(sub, **kwargs)
    return out


def average_maps(maps):
    """Label-aligned mean over modalities (requires identical measure sets)."""
    sets = {mod: tuple(m.index) for mod, m in maps.items()}
    assert len(set(sets.values())) == 1, f"measure sets differ: {sets}"
    measures = list(next(iter(maps.values())).index)
    return pd.concat(maps.values()).groupby(level=0).mean().loc[measures, measures]


def _blank_diagonal(cormap):
    """Copy of a square map with NaN on the diagonal (works with pandas copy-on-write)."""
    vals = cormap.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(vals, np.nan)
    return pd.DataFrame(vals, index=cormap.index, columns=cormap.columns)


def mean_agreement(cormap):
    """Average correlation of each measure with all others (diagonal excluded)."""
    return _blank_diagonal(cormap).mean(axis=1)


def map_similarity(maps, n_perm=10_000, seed=0, exclude=None):
    """Spearman rho between the upper triangles of each pair of modality maps.

    p-values from a one-sided Mantel-style permutation test (shuffling the measure labels
    of one map), FDR-corrected across pairs. The entries of a correlation matrix are not
    independent, so parametric p-values would be too small.
    """
    rng = np.random.default_rng(seed)
    maps = {k: (v.drop(index=exclude, columns=exclude) if exclude else v) for k, v in maps.items()}
    measures = list(next(iter(maps.values())).index)
    iu = np.triu_indices(len(measures), k=1)

    rows = []
    for a, b in combinations(maps, 2):
        A = maps[a].loc[measures, measures].values
        B = maps[b].loc[measures, measures].values
        rho = spearmanr(A[iu], B[iu]).statistic
        null = np.empty(n_perm)
        for i in range(n_perm):
            p = rng.permutation(len(measures))
            null[i] = spearmanr(A[iu], B[np.ix_(p, p)][iu]).statistic
        rows.append({
            "pair": f"{MODALITY_NAMES[a]} - {MODALITY_NAMES[b]}",
            "rho": rho,
            "p_perm": (1 + np.sum(null >= rho)) / (1 + n_perm),
        })

    out = pd.DataFrame(rows).sort_values("rho", ascending=False, ignore_index=True)
    out["p_fdr"] = false_discovery_control(out["p_perm"])
    return out


def cormap_stats(df, method="spearman", group_by=None, min_subjects=3):
    """Per measure pair: mean r across subjects, 95% CI and t-test on Fisher-z."""
    maps = subject_cormaps(df, method, group_by)
    n = len(maps)
    if n < min_subjects:
        print(f"only {n} subjects: no subject-level inference")
        return None

    measures = list(next(iter(maps.values())).index)
    iu = np.triu_indices(len(measures), k=1)
    r = np.stack([m.values[iu] for m in maps.values()])
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))

    res = ttest_1samp(z, 0, axis=0)
    ci = res.confidence_interval()
    return pd.DataFrame({
        "measure_a": [measures[i] for i in iu[0]],
        "measure_b": [measures[j] for j in iu[1]],
        "mean_r":    r.mean(axis=0),          # matches the heatmap
        "ci_low":    np.tanh(ci.low),
        "ci_high":   np.tanh(ci.high),
        "t":         res.statistic,
        "p":         res.pvalue,
        "n_subjects": n,
    })


def cormap_stats_all(df, modalities=None, **kwargs):
    """Stats per modality, FDR-corrected jointly over all tests reported."""
    out = []
    for mod in modalities or MODALITY_ORDER:
        s = cormap_stats(df[df["modality"] == mod], **kwargs)
        if s is not None:
            out.append(s.assign(modality=mod))
    out = pd.concat(out, ignore_index=True)
    out["p_fdr"] = false_discovery_control(out["p"])
    return out


def cormap_stats_by_depth(df, **kwargs):
    """cormap_stats_all per brain depth, FDR-corrected jointly across depths."""
    out = pd.concat(
        [cormap_stats_all(df[df["brain_depth"] == d], **kwargs).assign(brain_depth=d)
         for d in DEPTH_ORDER],
        ignore_index=True,
    )
    out["p_fdr"] = false_discovery_control(out["p"])
    return out


SIG_LEVELS = [(0.001, "***"), (0.01, "**"), (0.05, "*")]


def sig_code(p):
    """'***', '**', '*' or 'ns' for one p-value (thresholds from SIG_LEVELS)."""
    return next((c for thr, c in SIG_LEVELS if p < thr), "ns")


def sig_labels(stats, modality, measures, col="p_fdr"):
    """Measure x measure matrix of significance codes ('***', '**', '*', 'ns');
    empty string where no test exists."""
    lab = pd.DataFrame("", index=measures, columns=measures)
    s = stats[stats["modality"] == modality]
    for a, b, p in zip(s["measure_a"], s["measure_b"], s[col]):
        lab.loc[a, b] = lab.loc[b, a] = sig_code(p)
    return lab


def plot_cormap(cormap, ax=None, annot=True, labelsize=14, xlabels=True, ylabels=True,
                cbar=False, cbar_ax=None, mark=None, triangle=False, mark_fontsize=7):
    """Heatmap of a measure x measure map, diagonal blanked.
    `mark`: optional DataFrame of strings (e.g. significance codes) drawn in each cell."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    m = _blank_diagonal(cormap)

    mask = None
    if triangle:
        m = m.iloc[1:, :-1]                                   # drop empty first row / last column
        mask = np.triu(np.ones(m.shape, dtype=bool), k=1)     # keep lower triangle incl. new diagonal

    sns.heatmap(m, mask=mask, annot=annot, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1,
                square=True, cbar=cbar, cbar_ax=cbar_ax, ax=ax)

    ax.set_xticklabels(_display(m.columns), rotation=45, ha="right", fontsize=labelsize)
    ax.set_yticklabels(_display(m.index), rotation=0, fontsize=labelsize)
    ax.tick_params(labelbottom=xlabels, labelleft=ylabels)
    ax.set_xlabel(None)
    ax.set_ylabel(None)

    if mark is not None:
        labels = mark.loc[m.index, m.columns].values
        vals = m.values
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                if not labels[i, j] or np.isnan(vals[i, j]) or (mask is not None and mask[i, j]):
                    continue
                ax.text(j + 0.5, i + 0.5, labels[i, j], ha="center", va="center",
                        fontsize=mark_fontsize,
                        color="white" if abs(vals[i, j]) > 0.6 else "black")
    return ax


def plot_mean_agreement(values, figsize=(2.5, 7)):
    """Horizontal bars of each measure's mean correlation with the others.
    Bars run top-to-bottom in the same order as the heatmap rows."""
    fig, ax = plt.subplots(figsize=figsize)
    vals = values[::-1]
    bars = ax.barh(range(len(vals)), vals.values, height=0.5,
                   color=plt.get_cmap("coolwarm")(0.85))

    ax.spines[:].set_visible(False)
    ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

    offset = 0.02 * (vals.abs().max() or 1)
    for bar, v in zip(bars, vals):
        y = bar.get_y() + bar.get_height() / 2
        ax.text(v + offset if v >= 0 else v - offset, y, _fmt_short(v),
                va="center", ha="left" if v >= 0 else "right", fontsize=16)

    ax.set_xlabel("Average \n correlation", fontsize=18)
    fig.tight_layout()
    return fig


def plot_cormaps_stacked(maps, figsize=(3.5, 9), stats=None, triangle=True, mark_fontsize=10):
    """One map per modality, stacked, shared colorbar;
    × marks pairs that are not significant (FDR) where tests exist."""
    fig, axes = plt.subplots(len(maps), 1, figsize=figsize)
    cbar_ax = fig.add_axes([0.91, 0.20, 0.04, 0.60])

    for i, (ax, (mod, cm)) in enumerate(zip(axes, maps.items())):
        last = i == len(maps) - 1
        mark = None
        if stats is not None and mod in set(stats["modality"]):
            codes = sig_labels(stats, mod, list(cm.index))
            mark = codes.apply(lambda col: col.map(lambda c: "×" if c == "ns" else ""))
        plot_cormap(cm, ax=ax, annot=False, labelsize=None, xlabels=last,
                    cbar=last, cbar_ax=cbar_ax if last else None,
                    mark=mark, triangle=triangle, mark_fontsize=mark_fontsize)
        ax.set_title(MODALITY_NAMES[mod], fontsize=14)

    cbar_ax.set_ylabel("Spearman ρ", rotation=270, labelpad=12)
    cbar_ax.tick_params(labelsize=10)
    fig.subplots_adjust(right=0.88, hspace=0.2)
    return fig


def plot_map_similarity(sim, figsize=(2.4, 4.8), show_sig=True, sig_col="p_fdr"):
    """Bar per modality pair: rho between their maps, with significance codes."""
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(sim["pair"], sim["rho"], color=plt.get_cmap("coolwarm")(0.9), width=0.6)

    if show_sig and sig_col in sim.columns:
        for bar, rho, p in zip(bars, sim["rho"], sim[sig_col]):
            y = rho + 0.03 if rho >= 0 else rho - 0.03
            ax.text(bar.get_x() + bar.get_width() / 2, y, sig_code(p),
                    ha="center", va="bottom" if rho >= 0 else "top", fontsize=10)

    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(min(0, sim["rho"].min() - 0.1), 1.08)
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    ax.set_ylabel("Spearman ρ between datasets", fontsize=12)
    fig.tight_layout()
    return fig


def cormap_figure_set(maps, tag, sim=None, stats=None, save=True,
                      panels=("overall", "mean", "by_modality", "similarity"),
                      overall_figsize=(6, 6)):
    """Correlation-map figures for one set of modality maps."""
    overall = average_maps(maps)
    figs = {}

    if "overall" in panels:
        fig, ax = plt.subplots(figsize=overall_figsize)
        plot_cormap(overall, ax=ax)
        fig.tight_layout()
        figs["overall"] = fig
    if "mean" in panels:
        figs["mean"] = plot_mean_agreement(mean_agreement(overall))
    if "by_modality" in panels:
        figs["by_modality"] = plot_cormaps_stacked(maps, stats=stats)
    if "similarity" in panels:
        if sim is None:
            sim = map_similarity(maps)
        figs["similarity"] = plot_map_similarity(sim)

    if save:
        for name, fig in figs.items():
            save_fig(fig, f"cormap_{tag}_{name}")
    plt.show()
    return figs


def _depth_agreement_handles(with_overall):
    handles = [Patch(facecolor=DEPTH_PALETTE[d], label=DEPTH_LABELS[d]) for d in DEPTH_ORDER]
    if with_overall:
        handles.append(Line2D([], [], color="black", linestyle="--", linewidth=1.5,
                              label="All depths"))
    return handles


def plot_mean_agreement_by_depth(maps_by_depth, overall_map=None, modality=None, ax=None,
                                 figsize=(9, 4.5), ylim=None, legend=True, xlabels=True):
    """Average correlation of each measure with all others, per brain depth (bars, ROI palette).
    modality=None: average over modalities (as in the overall map); otherwise that modality only.
    overall_map: optional map whose mean agreement is drawn as a dashed line per measure."""
    get = (lambda maps: average_maps(maps)) if modality is None else (lambda maps: maps[modality])
    depth_vals = {d: mean_agreement(get(maps_by_depth[d])) for d in DEPTH_ORDER}
    measures = [m for m in MEASURE_ORDER if m in depth_vals[DEPTH_ORDER[0]].index]

    n_d = len(DEPTH_ORDER)
    bw = 0.8 / n_d
    xb = np.arange(len(measures))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    fig = ax.figure

    for i, d in enumerate(DEPTH_ORDER):
        vals = depth_vals[d].reindex(measures).values
        ax.bar(xb + (i - (n_d - 1) / 2) * bw, vals, width=bw * 0.9,
               color=DEPTH_PALETTE[d], alpha=0.9, edgecolor="white", linewidth=0.6)

    all_vals = np.concatenate([depth_vals[d].values for d in DEPTH_ORDER])
    if overall_map is not None:
        overall = mean_agreement(overall_map).reindex(measures)
        half = bw * n_d / 2
        for x, v in zip(xb, overall.values):
            ax.plot([x - half, x + half], [v, v], color="black", linestyle="--",
                    linewidth=1.5, zorder=5)
        all_vals = np.concatenate([all_vals, overall.values])

    if ylim is None:
        ylim = (min(0, np.nanmin(all_vals)) - 0.05, np.nanmax(all_vals) + 0.1)
    ax.set_ylim(*ylim)
    ax.axhline(0, color="0.6", linewidth=0.8, zorder=1)

    ax.set_xticks(xb)
    ax.set_xticklabels(_display(measures), rotation=45, ha="right", fontsize=11)
    ax.tick_params(axis="x", labelbottom=xlabels)
    ax.set_ylabel("Average correlation\nwith other measures", fontsize=12)
    ax.tick_params(axis="y", labelsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="-", alpha=0.2)
    ax.set_axisbelow(True)

    if legend:
        ax.legend(handles=_depth_agreement_handles(overall_map is not None),
                  loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=10)
    return fig


def plot_mean_agreement_by_depth_stacked(maps_by_depth, overall_maps=None, figsize=(9, 11)):
    """One panel per modality (same layout as the stacked correlation maps), shared y-axis."""
    vals = [mean_agreement(maps_by_depth[d][mod]).values
            for d in DEPTH_ORDER for mod in MODALITY_ORDER]
    if overall_maps is not None:
        vals += [mean_agreement(overall_maps[mod]).values for mod in MODALITY_ORDER]
    vals = np.concatenate(vals)
    ylim = (min(0, np.nanmin(vals)) - 0.05, np.nanmax(vals) + 0.1)

    fig, axes = plt.subplots(len(MODALITY_ORDER), 1, figsize=figsize, sharex=True)
    for k, (ax, mod) in enumerate(zip(axes, MODALITY_ORDER)):
        plot_mean_agreement_by_depth(
            maps_by_depth,
            overall_map=None if overall_maps is None else overall_maps[mod],
            modality=mod, ax=ax, ylim=ylim, legend=False,
            xlabels=(k == len(MODALITY_ORDER) - 1),
        )
        ax.set_title(MODALITY_NAMES[mod], fontsize=14)

    fig.legend(handles=_depth_agreement_handles(overall_maps is not None),
               loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=10)
    fig.tight_layout()
    return fig


# =============================================================================
# 6. PCA comparisons and sweep
# =============================================================================

def match_pca_rows(df_std, df_pca, swap, modalities=None):
    """Standard and PCA versions on the same rows.

    Keeps the rows (modality x model x subject x layer x depth) where every measure in `swap`
    exists in df_pca. Returns (std, mixed): df_std restricted to those rows, and the same rows
    with the `swap` measures replaced by their PCA versions."""
    if modalities is not None:
        df_std = df_std[df_std["modality"].isin(modalities)]
        df_pca = df_pca[df_pca["modality"].isin(modalities)]
    pca = df_pca[df_pca["measure"].isin(swap)]
    complete = pca.groupby(ROW_KEY, observed=True)["measure"].nunique()
    complete = complete[complete == len(swap)].reset_index()[ROW_KEY]

    std = df_std.merge(complete, on=ROW_KEY)
    n_drop = len(df_std) - len(std)
    if n_drop:
        warnings.warn(f"{n_drop} standard rows without a complete PCA counterpart; "
                      "restricting both versions to rows where all PCA measures exist")
    mixed = pd.concat([std[~std["measure"].isin(swap)], pca.merge(complete, on=ROW_KEY)],
                      ignore_index=True)
    return std, mixed


def highlight_cormap(ax, labels, highlight, color="0.15"):
    """Bold tick labels and a thin frame around the rows/columns of `highlight` (contiguous)."""
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        if lbl.get_text() in highlight:
            lbl.set_fontweight("bold")
    idx = [labels.index(h) for h in highlight]
    i0, i1, n = min(idx), max(idx) + 1, len(labels)
    kw = dict(fill=False, edgecolor=color, linewidth=0.8, alpha=0.6, zorder=5, clip_on=False)
    ax.add_patch(Rectangle((0, i0), n, i1 - i0, **kw))      # row band
    ax.add_patch(Rectangle((i0, 0), i1 - i0, n, **kw))      # column band


def plot_agreement_gain(vals, vals_std, highlight, figsize=(2.8, 6)):
    """Mean agreement of one version (filled) vs a reference (dashed outline);
    change printed for the highlighted measures. Bars align with the heatmap rows."""
    order = list(vals.index)[::-1]
    v, vs = vals.reindex(order), vals_std.reindex(order)
    y = np.arange(len(order))

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(y, v.values, height=0.5, color=plt.get_cmap("coolwarm")(0.85))
    ax.barh(y, vs.values, height=0.5, fill=False, edgecolor="0.2",
            linewidth=0.8, linestyle="--")

    span = np.nanmax(np.abs(np.r_[v.values, vs.values])) or 1
    for yi, m in zip(y, order):
        right = max(v[m], vs[m], 0)
        txt, weight = _fmt_short(v[m]), "normal"
        if m in highlight:
            d = v[m] - vs[m]
            txt += f"  ({'+' if d >= 0 else '−'}{_fmt_short(abs(d))})"
            weight = "bold"
        ax.text(right + 0.03 * span, yi, txt, va="center", ha="left", fontweight=weight)

    ax.spines[:].set_visible(False)
    ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
    ax.set_xlabel("average\ncorrelation")
    fig.tight_layout()
    return fig


def pca_comparison_figures(df_std, df_mixed, labels, highlight, tag, frame=False,
                           group_by=WITHIN, modalities=None):
    """Correlation-map figures for a PCA version (df_mixed) against the standard version on
    the same rows (df_std): overall map, mean-agreement change, per-modality maps, similarity.
    `labels` renames the PCA measures for display; `frame` outlines the highlighted rows."""
    maps     = cormaps(df_mixed, modalities=modalities, group_by=group_by)
    maps_std = cormaps(df_std, modalities=modalities, group_by=group_by)
    stats    = cormap_stats_all(df_mixed, modalities=modalities, group_by=group_by)

    overall, overall_std = average_maps(maps), average_maps(maps_std)
    print("change in overall map (PCA - standard, same rows):")
    print((overall - overall_std).loc[highlight].round(3))

    disp = overall.rename(index=labels, columns=labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_cormap(disp, ax=ax)
    if frame:
        highlight_cormap(ax, list(disp.index), [labels[m] for m in highlight])
    fig.tight_layout()
    save_fig(fig, f"cormap_{tag}_overall")

    fig = plot_agreement_gain(mean_agreement(overall), mean_agreement(overall_std), highlight)
    save_fig(fig, f"cormap_{tag}_mean")

    maps_disp  = {k: v.rename(index=labels, columns=labels) for k, v in maps.items()}
    stats_disp = stats.replace({"measure_a": labels, "measure_b": labels})
    cormap_figure_set(maps_disp, tag=tag, stats=stats_disp, panels=("by_modality", "similarity"))
    return maps, maps_std, stats


def pca_rsa_pair_changes(df_std, df_mixed, rsa=RSA_MEASURES, group_by=WITHIN, modalities=None):
    """Change in RSA's correlations with every other measure when RSA is computed on PCA features.
    Per pair: mean r in both versions, mean difference, paired t-test on Fisher-z across subjects
    (FDR over all pairs). The RSA-RSA pair is excluded since both sides change."""
    others = [m for m in MEASURE_ORDER if m not in rsa]
    z = lambda r: np.arctanh(np.clip(r, -0.999999, 0.999999))

    rows = []
    for mod in modalities or MODALITY_ORDER:
        std = subject_cormaps(df_std[df_std["modality"] == mod], group_by=group_by)
        pca = subject_cormaps(df_mixed[df_mixed["modality"] == mod], group_by=group_by)
        assert std.keys() == pca.keys(), f"{mod}: subject sets differ"
        subjects = list(std)

        for r in rsa:
            for o in others:
                if o not in std[subjects[0]].columns:
                    continue
                r_std = np.array([std[s].loc[r, o] for s in subjects])
                r_pca = np.array([pca[s].loc[r, o] for s in subjects])
                row = {"modality": mod, "rsa": r, "other": o,
                       "r_std": r_std.mean(), "r_pca": r_pca.mean(),
                       "diff": (r_pca - r_std).mean(), "n_subjects": len(subjects),
                       "t": np.nan, "p": np.nan}
                if len(subjects) >= 3:
                    res = ttest_1samp(z(r_pca) - z(r_std), 0)
                    row.update(t=res.statistic, p=res.pvalue)
                rows.append(row)

    diff = pd.DataFrame(rows)
    diff["p_fdr"] = np.nan
    sel = diff["p"].notna()
    diff.loc[sel, "p_fdr"] = false_discovery_control(diff.loc[sel, "p"])

    summary = diff.groupby(["modality", "rsa"], observed=True).apply(
        lambda d: pd.Series({
            "mean_abs_diff": d["diff"].abs().mean(),
            "max_abs_diff":  d["diff"].abs().max(),
            "profile_rho":   spearmanr(d["r_std"], d["r_pca"]).statistic,   # descriptive, 7 points
            "n_sig_fdr":     (d["p_fdr"] < 0.05).sum() if d["p"].notna().any() else np.nan,
        })
    )
    return diff, summary


def stability_maps(df_sweep, measure, group_by=WITHIN):
    """Within-profile Spearman correlation between the layer profiles obtained with different
    component counts: per subject across layers within each model x depth group, averaged
    over groups, then over subjects. Returns one map per modality."""
    d = df_sweep[df_sweep["measure"] == measure]
    order = [n for n in N_ORDER if n in set(d["n_components"])]

    maps = {}
    for mod in MODALITY_ORDER:
        subj_maps = []
        for _, s in d[d["modality"] == mod].groupby("subject", observed=True):
            mats = []
            for _, g in s.groupby(group_by, observed=True):
                wide = (g.pivot(index="layer", columns="n_components", values="score")
                          .reindex(columns=order))
                mats.append(wide.corr(method="spearman"))
            subj_maps.append(pd.concat(mats).groupby(level=0).mean().loc[order, order])
        maps[mod] = pd.concat(subj_maps).groupby(level=0).mean().loc[order, order]
    return maps


def stability_to_ref(stab_maps):
    """Per measure x component count: profile correlation with the reference condition
    (STAB_REF), one column per modality."""
    return pd.concat(
        [pd.DataFrame({mod: maps[mod][STAB_REF[m]] for mod in MODALITY_ORDER}).assign(measure=m)
         for m, maps in stab_maps.items()]
    ).rename_axis("n_components").reset_index()


def plot_stability_maps(stab_overall, measures=PCA_MEASURES, figsize=None):
    """One triangular map per measure: within-profile correlation between component counts."""
    figsize = figsize or (4 * len(measures), 4)
    fig, axes = plt.subplots(1, len(measures), figsize=figsize, squeeze=False)
    for ax, m in zip(axes[0], measures):
        plot_cormap(stab_overall[m], ax=ax, triangle=True, labelsize=None)
        ax.set_title(MEASURE_NAMES[m])
    fig.tight_layout()
    return fig


def plot_stability_to_ref(stab_to_ref, measures=PCA_MEASURES, figsize=None):
    """Per measure: correlation of each component count's layer profile with the reference
    profile (standard features; PCA-100 for LSM), one line per modality."""
    figsize = figsize or (3.2 * len(measures), 3.2)
    fig, axes = plt.subplots(1, len(measures), figsize=figsize, sharey=True, squeeze=False)
    for ax, m in zip(axes[0], measures):
        sub = stab_to_ref[(stab_to_ref["measure"] == m)
                          & (stab_to_ref["n_components"] != STAB_REF[m])
                          & (stab_to_ref["n_components"] != "none")]
        x = sub["n_components"].astype(int)
        for mod in MODALITY_ORDER:
            color, marker = MODALITY_STYLE[mod]
            ax.plot(x, sub[mod], color=color, marker=marker, markersize=4, linewidth=1.2,
                    label=MODALITY_NAMES[mod])
        ax.set_xscale("log")
        ax.set_xticks(PCA_COMPONENTS)
        ax.set_xticklabels(PCA_COMPONENTS)
        ax.minorticks_off()
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks(np.linspace(0, 1, 5))
        ax.set_title(f"{MEASURE_NAMES[m]}\n(vs {'PCA-100' if STAB_REF[m] == '100' else 'no PCA'})")
        ax.set_xlabel("number of components")
    axes[0, 0].set_ylabel("profile correlation\nwith reference")
    axes[0, -1].legend(loc="lower right")
    fig.tight_layout()
    return fig


def best_layers_sweep(df_sweep):
    """Best layer per modality x model x subject x depth x measure, for every component count."""
    out = []
    for n in N_ORDER:
        d = df_sweep[df_sweep["n_components"] == n]
        if not d.empty:
            out.append(best_layers(d.drop(columns="n_components")).assign(n_components=n))
    return pd.concat(out, ignore_index=True)


def sweep_best_summary(bl):
    """Mean best-layer depth across subjects (each subject first averaged over models)."""
    per_subj = (bl.groupby(["modality", "subject", "measure", "brain_depth", "n_components"],
                           observed=True)["norm_layer_depth"].mean().reset_index())
    return _summarise_ci(per_subj, ["modality", "measure", "brain_depth", "n_components"],
                         "norm_layer_depth")


def hierarchy_summary(bl):
    """Per subject (averaged over models): gap = late - early best-layer depth, and whether
    early <= mid <= late holds. Mean + CI of the gap, mean share of ordered cases."""
    w = bl.pivot_table(index=["modality", "subject", "model", "measure", "n_components"],
                       columns="brain_depth", values="norm_layer_depth", observed=True)
    w.columns = w.columns.astype(str)
    w = w.reset_index()
    w["gap"] = w["late"] - w["early"]
    w["ordered"] = ((w["early"] <= w["mid"]) & (w["mid"] <= w["late"])).astype(float)
    per_subj = (w.groupby(["modality", "subject", "measure", "n_components"], observed=True)
                  [["gap", "ordered"]].mean().reset_index())

    keys = ["modality", "measure", "n_components"]
    rows = []
    for k, g in per_subj.groupby(keys, observed=True):
        mean, low, high, n = _mean_ci(g["gap"])
        rows.append({**dict(zip(keys, map(str, k))), "gap": mean, "low": low, "high": high,
                     "ordered": g["ordered"].mean(), "n": n})
    return pd.DataFrame(rows)


def _sweep_axis(ax, ref):
    """Shared x-axis for component-count curves: evenly spaced counts, 'no PCA' last,
    a light band at the measure's reference condition."""
    ax.axvspan(N_ORDER.index(ref) - 0.35, N_ORDER.index(ref) + 0.35,
               color="0.92", zorder=0)
    ax.set_xticks(range(len(N_ORDER)))
    ax.set_xticklabels([SWEEP_LABELS[n] for n in N_ORDER], rotation=45, ha="right")
    ax.set_xlim(-0.5, len(N_ORDER) - 0.5)


def plot_sweep_best_layers(summ, measures=PCA_MEASURES, figsize=None):
    """Rows = measure, columns = modality: mean best-layer depth per brain region
    as a function of the number of components."""
    figsize = figsize or (3.3 * len(MODALITY_ORDER) + 1.5, 2.6 * len(measures))
    fig, axes = plt.subplots(len(measures), len(MODALITY_ORDER), figsize=figsize,
                             sharex=True, sharey=True, squeeze=False)
    for r, m in enumerate(measures):
        for c, mod in enumerate(MODALITY_ORDER):
            ax = axes[r, c]
            sub = summ[(summ["measure"] == m) & (summ["modality"] == mod)]
            for depth in DEPTH_ORDER:
                s = (sub[sub["brain_depth"] == depth].set_index("n_components")
                        .reindex(N_ORDER).dropna(subset=["mean"]))
                x = [N_ORDER.index(n) for n in s.index]
                ax.errorbar(x, s["mean"], yerr=[s["mean"] - s["low"], s["high"] - s["mean"]],
                            color=DEPTH_PALETTE[depth], marker="o", markersize=4,
                            linewidth=1.2, elinewidth=0.8)
            _sweep_axis(ax, STAB_REF[m])
            ax.set_ylim(-0.05, 1.05)
            ax.set_yticks(np.linspace(0, 1, 5))
            if r == 0:
                ax.set_title(MODALITY_NAMES[mod], pad=8)
            if c == 0:
                ax.set_ylabel("best-layer depth")
                _row_label(ax, MEASURE_NAMES[m])
            if r == len(measures) - 1:
                ax.set_xlabel("number of components")

    handles = [Line2D([], [], color=DEPTH_PALETTE[d], marker="o", markersize=4,
                      label=DEPTH_LABELS[d]) for d in DEPTH_ORDER]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    return fig


def plot_hierarchy_gap(hs, measures=PCA_MEASURES, figsize=None):
    """Per measure: late - early best-layer depth as a function of the number of components,
    one line per modality (> 0 = later regions matched by deeper layers)."""
    figsize = figsize or (3.2 * len(measures), 3.2)
    fig, axes = plt.subplots(1, len(measures), figsize=figsize, sharey=True, squeeze=False)
    for ax, m in zip(axes[0], measures):
        for mod in MODALITY_ORDER:
            s = (hs[(hs["measure"] == m) & (hs["modality"] == mod)].set_index("n_components")
                   .reindex(N_ORDER).dropna(subset=["gap"]))
            color, marker = MODALITY_STYLE[mod]
            x = [N_ORDER.index(n) for n in s.index]
            ax.errorbar(x, s["gap"], yerr=[s["gap"] - s["low"], s["high"] - s["gap"]],
                        color=color, marker=marker, markersize=4, linewidth=1.2,
                        elinewidth=0.8, label=MODALITY_NAMES[mod])
        ax.axhline(0, color="0.6", linestyle="--", linewidth=0.8, zorder=1)
        _sweep_axis(ax, STAB_REF[m])
        ax.set_title(MEASURE_NAMES[m])
        ax.set_xlabel("number of components")
    axes[0, 0].set_ylabel("hierarchy gap\n(late − early best-layer depth)")
    axes[0, -1].legend(loc="lower right")
    fig.tight_layout()
    return fig


# =============================================================================
# 7. ROI profiles (level agreement, relative profiles)
# =============================================================================

def level_table(df, how="mean"):
    """One score per modality x model x subject x depth x measure: the mean over layers
    (how="mean") or the best-layer score (how="max"). The layer column is set to a constant,
    so the correlation code treats the 9 model x depth groups as the rows."""
    agg = (df.groupby(["modality", "model", "subject", "brain_depth", "measure"], observed=True)
             ["score"].agg(how).reset_index())
    return agg.assign(layer=how)


def relative_region_profiles(df, how=REL_HOW):
    """Score per modality x model x subject x depth x measure, divided by that measure's
    mean over the three depths (1 = the measure's own average level)."""
    lv = level_table(df, how)
    g = lv.groupby(["modality", "model", "subject", "measure"], observed=True)["score"]
    lv["rel"] = lv["score"] / g.transform("mean")
    return lv


def summarise_relative(rel, ci=0.95):
    """Mean relative level per modality x model x measure x depth, plus 'all' (averaged over
    models per subject). t-based CI if n >= 3, otherwise the range across subjects."""
    all_models = (rel.groupby(["modality", "subject", "measure", "brain_depth"], observed=True)
                     ["rel"].mean().reset_index().assign(model="all"))
    both = pd.concat([rel[all_models.columns].astype({"model": str}), all_models], ignore_index=True)
    return _summarise_ci(both, ["modality", "model", "measure", "brain_depth"], "rel", ci)


def relative_contrast_tests(rel, tested=("RSA_euc", "RSA_cor")):
    """Per subject: (late - early) relative level of each tested measure minus the mean of the
    same contrast over the non-tested measures, averaged over models; t-test across subjects."""
    others = [m for m in present_measures(rel) if m not in tested]
    w = rel.pivot_table(index=["modality", "subject", "model", "measure"], columns="brain_depth",
                        values="rel", observed=True)
    c = ((w["late"] - w["early"])
           .groupby(["modality", "subject", "measure"], observed=True).mean()
           .unstack("measure"))
    c.columns = c.columns.astype(str)

    rows = []
    for mod, cm in c.groupby(level="modality", observed=True):
        ref = cm[others].mean(axis=1)
        for m in tested:
            d = (cm[m] - ref).dropna()
            row = {"modality": str(mod), "measure": m, "late_minus_early": cm[m].mean(),
                   "others": ref.mean(), "diff": d.mean(), "n": len(d), "t": np.nan, "p": np.nan}
            if len(d) >= 3:
                res = ttest_1samp(d, 0)
                row.update(t=res.statistic, p=res.pvalue)
            rows.append(row)
    out = pd.DataFrame(rows)
    out["p_fdr"] = np.nan
    sel = out["p"].notna()
    out.loc[sel, "p_fdr"] = false_discovery_control(out.loc[sel, "p"])
    return out


def plot_relative_profiles(rs, highlight=("RSA_euc", "RSA_cor"), models=("all",),
                           figsize=None):
    """Rows = modality, columns = model (or 'all' = averaged over models). Grey: other
    measures, dark line: their mean, colour: highlighted measures with CI / range."""
    hl_colors = dict(zip(highlight, ["#c1121f", "#1d3557", "#2a9d8f"]))
    figsize = figsize or (3.2 * len(models) + 1.5, 7.5)
    x = np.arange(len(DEPTH_ORDER))

    fig, axes = plt.subplots(len(MODALITY_ORDER), len(models), figsize=figsize,
                             sharey=True, squeeze=False)
    for r, mod in enumerate(MODALITY_ORDER):
        for c, model in enumerate(models):
            ax = axes[r, c]
            sub = rs[(rs["modality"] == mod) & (rs["model"] == model)]
            present = present_measures(sub)
            others = [m for m in present if m not in highlight]

            ys = []
            for m in others:
                y = sub[sub["measure"] == m].set_index("brain_depth").reindex(DEPTH_ORDER)["mean"]
                ys.append(y.values)
                ax.plot(x, y.values, color="0.8", linewidth=0.8,
                        marker=MEASURE_MARKERS[m], markersize=3, zorder=1)
            ax.plot(x, np.nanmean(ys, axis=0), color="0.3", linewidth=1.6, zorder=2)

            for m in highlight:
                s = sub[sub["measure"] == m].set_index("brain_depth").reindex(DEPTH_ORDER)
                ax.errorbar(x, s["mean"], yerr=[s["mean"] - s["low"], s["high"] - s["mean"]],
                            color=hl_colors[m], marker=MEASURE_MARKERS[m], markersize=5,
                            linewidth=1.5, elinewidth=0.8, zorder=3)

            ax.axhline(1, color="0.85", linestyle=":", linewidth=0.8, zorder=0)
            ax.set_xticks(x)
            ax.set_xticklabels(depth_ticklabels(mod))
            ax.set_xlim(-0.3, len(DEPTH_ORDER) - 0.7)
            if r == 0:
                ax.set_title(MODEL_NAMES.get(model, "all models"), pad=8)
            if c == 0:
                ax.set_ylabel("relative alignment\n(1 = measure mean)")
                _row_label(ax, MODALITY_NAMES[mod])

    handles = [Line2D([], [], color=hl_colors[m], marker=MEASURE_MARKERS[m], markersize=5,
                      linewidth=1.5, label=MEASURE_NAMES[m]) for m in highlight]
    handles += [Line2D([], [], color="0.3", linewidth=1.6, label="other measures (mean)"),
                Line2D([], [], color="0.8", linewidth=0.8, label="other measures")]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    return fig


# =============================================================================
# 8. Hierarchical correspondence (depth profiles, best layers)
# =============================================================================

def add_layer_depth(df):
    """Add normalised layer depth (computed once per model x layer)."""
    pairs = df[["model", "layer"]].drop_duplicates().copy()
    with _in_dir(SCRIPTS_DIR):
        pairs["norm_layer_depth"] = [
            util.get_layer_depth(m, l, normalize=True)
            for m, l in zip(pairs["model"].astype(str), pairs["layer"])
        ]
    return df.merge(pairs, on=["model", "layer"], how="left")


def depth_profiles(df, decimals=DEPTH_BIN_DECIMALS):
    """Mean score per modality x model x depth x measure x depth bin
    (averaged over subjects and over the layers falling into each bin)."""
    d = add_layer_depth(df)
    assert d["norm_layer_depth"].notna().all(), "layers without a depth"
    d["depth_bin"] = d["norm_layer_depth"].round(decimals)
    return (d.groupby(["modality", "model", "brain_depth", "measure", "depth_bin"], observed=True)
              ["score"].mean().reset_index())


def _minmax(d, by):
    g = d.groupby(by, observed=True)["score"]
    lo, hi = g.transform("min"), g.transform("max")
    return d.assign(score=(d["score"] - lo) / (hi - lo).replace(0, 1))


def scaled_profiles(prof, scale_first=False):
    """Average over modalities, min-max scaled per measure x model x depth.

    scale_first=False reproduces the old figure: average raw scores across modalities, then scale.
    scale_first=True scales each modality's profile first, then averages.
    """
    keys = ["measure", "model", "brain_depth"]
    if scale_first:
        prof = _minmax(prof, keys + ["modality"])
    avg = prof.groupby(keys + ["depth_bin"], observed=True)["score"].mean().reset_index()
    return avg if scale_first else _minmax(avg, keys)


def profile_consistency(prof):
    """Per measure x depth: mean pairwise Pearson r between the layer profiles of all
    modality x model combinations (raw scores, aligned by depth bin)."""
    rows = []
    for (meas, depth), sub in prof.groupby(["measure", "brain_depth"], observed=True):
        wide = sub.pivot(index="depth_bin", columns=["modality", "model"], values="score")
        c = wide.corr().values
        iu = np.triu_indices_from(c, k=1)
        rows.append({"measure": str(meas), "brain_depth": str(depth),
                     "cor": np.nanmean(c[iu]), "n_profiles": wide.shape[1]})
    cons = pd.DataFrame(rows)
    cons_mean = cons.groupby("measure")["cor"].mean().sort_values(ascending=False)
    return cons, cons_mean


def plot_depth_profiles(scaled, figsize=(4, 9)):
    """One panel per brain depth; one line per measure (mean over models),
    coloured by depth, marker per measure."""
    plot_df = (scaled.groupby(["brain_depth", "measure", "depth_bin"], observed=True)["score"]
                     .mean().reset_index())                     # explicit mean over models
    measures = present_measures(plot_df)

    fig, axes = plt.subplots(len(DEPTH_ORDER), 1, figsize=figsize, sharex=True, sharey=True)
    for ax, depth in zip(axes, DEPTH_ORDER):
        color = DEPTH_PALETTE[depth]
        for m in measures:
            sub = plot_df[(plot_df["brain_depth"] == depth) & (plot_df["measure"] == m)]
            sub = sub.sort_values("depth_bin")
            ax.plot(sub["depth_bin"], sub["score"], color=color, alpha=0.9,
                    marker=MEASURE_MARKERS[m], markersize=4, markeredgewidth=0.6, linewidth=1.2)

        ax.text(0.5, 0.2, DEPTH_LABELS[depth], transform=ax.transAxes, color=color,
                fontsize=mpl.rcParams["axes.titlesize"], ha="center", va="top", fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks(np.linspace(0, 1, 5))
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_ylabel("scaled alignment score")

    axes[-1].set_xlabel("normalized layer depth")
    fig.tight_layout()
    return fig


def plot_profile_legend(measures, figsize=(3.5, 2.6)):
    """Separate legend: marker per measure, colour per depth."""
    measure_handles = [Line2D([0], [0], marker=MEASURE_MARKERS[m], linestyle="None",
                              color="grey", markersize=8, label=MEASURE_NAMES[m])
                       for m in measures]
    depth_handles = [Patch(facecolor=DEPTH_PALETTE[d], edgecolor="none",
                           label=DEPTH_LABELS_MULTILINE[d]) for d in DEPTH_ORDER]

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    leg1 = ax.legend(handles=measure_handles, title="Measures", loc="upper left",
                     bbox_to_anchor=(0.0, 1.0), frameon=False, labelspacing=0.6)
    ax.add_artist(leg1)
    ax.legend(handles=depth_handles, title="Neural Depth", loc="upper left",
              bbox_to_anchor=(0.55, 1.0), frameon=False, labelspacing=1.2)
    fig.tight_layout()
    return fig


def plot_consistency(cons, cons_mean, figsize=(6, 3.5), legend=False):
    """Within-measure consistency: one point per brain depth,
    black line = mean across depths."""
    order = list(cons_mean.index)

    fig, ax = plt.subplots(figsize=figsize)
    for i, m in enumerate(order):
        sub = cons[cons["measure"] == m]
        ax.plot([i - 0.3, i + 0.3], [cons_mean[m]] * 2, color="black", linewidth=1.2, zorder=2)
        ax.scatter(np.full(len(sub), i), sub["cor"],
                   c=[DEPTH_PALETTE[d] for d in sub["brain_depth"]], s=22, zorder=3)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(_display(order), rotation=45, ha="right")
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(min(0, cons["cor"].min() - 0.05), 1)
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.set_ylabel("within-measure consistency")

    if legend:
        handles = [Line2D([], [], marker="o", linestyle="None", color=DEPTH_PALETTE[d],
                          markersize=5, label=DEPTH_LABELS[d]) for d in DEPTH_ORDER]
        handles.append(Line2D([], [], color="black", linewidth=1.2, label="mean"))
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    return fig


BEST_KEY = ["modality", "model", "subject", "brain_depth", "measure"]


def best_layers(df):
    """Per modality x model x subject x depth x measure: the layer with the highest score."""
    d = add_layer_depth(df)
    idx = d.groupby(BEST_KEY, observed=True)["score"].idxmax()
    return d.loc[idx.values, BEST_KEY + ["layer", "norm_layer_depth", "score"]].reset_index(drop=True)


def best_layer_summary(best, ci=0.95, min_n_ci=3):
    """Mean normalised depth of the best layer across subjects, with an interval:
    t-based CI if n >= min_n_ci, otherwise the range across subjects (ephys, n = 2)."""
    keys = ["modality", "model", "brain_depth", "measure"]
    rows = []
    for k, v in best.groupby(keys, observed=True)["norm_layer_depth"]:
        mean, low, high, n = _mean_ci(v, ci, min_n_ci)
        interval = f"{int(ci * 100)}% CI" if n >= min_n_ci else "range"
        rows.append({**dict(zip(keys, map(str, k))),
                     "mean": mean, "low": low, "high": high, "n": n, "interval": interval})
    return pd.DataFrame(rows)


def plot_hierarchical_correspondence(summary, figsize=(12, 7.5), total_width=0.6,
                                     legend=True, separators=False):
    """Rows = modality, columns = model; per brain depth, the mean normalised depth of the
    best layer for each measure (marker per measure, colour per depth)."""
    measures = present_measures(summary)
    offsets = np.linspace(-total_width / 2, total_width / 2, len(measures))

    fig, axes = plt.subplots(len(MODALITY_ORDER), len(MODEL_ORDER), figsize=figsize,
                             sharey=True, squeeze=False)
    for r, mod in enumerate(MODALITY_ORDER):
        for c, model in enumerate(MODEL_ORDER):
            ax = axes[r, c]
            sub = summary[(summary["modality"] == mod) & (summary["model"] == model)]

            for k, m in enumerate(measures):
                xs, ys = [], []
                for i, depth in enumerate(DEPTH_ORDER):
                    row = sub[(sub["measure"] == m) & (sub["brain_depth"] == depth)]
                    if row.empty:
                        continue
                    row = row.iloc[0]
                    x = i + offsets[k]
                    ax.errorbar(
                        x, row["mean"],
                        yerr=[[row["mean"] - row["low"]], [row["high"] - row["mean"]]],
                        fmt=MEASURE_MARKERS[m], color=DEPTH_PALETTE[depth],
                        markersize=5, elinewidth=0.8, markeredgewidth=0.6, zorder=3,
                    )
                    xs.append(x)
                    ys.append(row["mean"])
                if len(xs) > 1:
                    ax.plot(xs, ys, color="0.75", linewidth=0.6, linestyle="--", zorder=2)

            ax.set_xticks(range(len(DEPTH_ORDER)))
            ax.set_xticklabels(depth_ticklabels(mod))
            ax.set_xlim(-0.5, len(DEPTH_ORDER) - 0.5)
            ax.set_ylim(-0.05, 1.05)
            ax.set_yticks(np.linspace(0, 1, 5))
            if separators:
                for x in range(len(DEPTH_ORDER) - 1):
                    ax.axvline(x + 0.5, color="0.9", linewidth=0.6, zorder=0)

            if r == 0:                                    # column header: model
                ax.set_title(MODEL_NAMES[model], pad=8)
            if c == 0:                                    # row header + y label
                ax.set_ylabel("normalized layer depth")
                _row_label(ax, MODALITY_NAMES[mod], x=-0.38)

    if legend:
        handles = [Line2D([], [], marker=MEASURE_MARKERS[m], linestyle="None", color="0.35",
                          markersize=5, label=MEASURE_NAMES[m]) for m in measures]
        fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
                   title="measure", alignment="left")
    fig.tight_layout()
    return fig


# =============================================================================
# 9. Layer structure (blocks, layer types, variance decomposition)
# =============================================================================

def get_layer_type(model_name, layer_name):
    """Extract layer type from layer name."""
    if model_name == "alexnet":
        mapping = {
            'Conv1': 'Conv', 'Relu1': 'ReLU', 'MaxPool1': 'MaxPool',
            'Conv2': 'Conv', 'Relu2': 'ReLU', 'MaxPool2': 'MaxPool',
            'Conv3': 'Conv', 'Relu3': 'ReLU',
            'Conv4': 'Conv', 'Relu4': 'ReLU',
            'Conv5': 'Conv', 'Relu5': 'ReLU', 'MaxPool5': 'MaxPool',
        }
        return mapping.get(layer_name, np.nan)

    if model_name == "resnet50":
        mapping = {
            "conv1": "Conv", "bn1": "BN", "relu": "ReLU",
            "conv2": "Conv1", "bn2": "BN1", "relu_1": "ReLU-1",
            "conv3": "Conv2", "bn3": "BN2", "add": "Add", "relu_2": "ReLU-2",
        }
        if "." in layer_name:
            return mapping.get(layer_name.split(".", 2)[-1], np.nan)
        return np.nan

    if model_name == "convnext_tiny":
        if layer_name.startswith("features.4.") or layer_name.startswith("features.6."):
            return np.nan
        mapping = {
            "block.0": "Conv", "block.2": "LN", "block.3": "Linear1",
            "block.4": "GeLU", "block.5": "Linear2", "mul": "Scale", "add": "Add",
        }
        parts = layer_name.split(".")
        if len(parts) >= 4:
            suffix = ".".join(parts[3:])
        elif len(parts) >= 3:
            suffix = parts[2]
        else:
            return np.nan
        if suffix in ["1", "layer_norm"]:
            return np.nan
        return mapping.get(suffix, np.nan)

    return np.nan


def get_block_number(model_name, layer_name):
    """Extract block id from layer name."""
    if model_name == "alexnet":
        mapping = {
            'Conv1': 'block.1', 'Relu1': 'block.1', 'MaxPool1': 'block.1',
            'Conv2': 'block.2', 'Relu2': 'block.2', 'MaxPool2': 'block.2',
            'Conv3': 'block.3', 'Relu3': 'block.3',
            'Conv4': 'block.4', 'Relu4': 'block.4',
            'Conv5': 'block.5', 'Relu5': 'block.5', 'MaxPool5': 'block.5',
            'AvgPool': 'block.7', 'Dropout': 'block.7',
            'FullyConnected1': 'block.7', 'Relu6': 'block.7', 'Dropout2': 'block.7',
            'FullyConnected2': 'block.7', 'Relu7': 'block.7', 'FullyConnected3': 'block.7',
        }
        return mapping.get(layer_name, np.nan)
    if model_name == "resnet50":
        return ".".join(layer_name.split(".")[:2]) if "." in layer_name else np.nan
    if model_name == "convnext_tiny":
        if layer_name.startswith("features.4.") or layer_name.startswith("features.6."):
            return np.nan
        return ".".join(layer_name.split(".")[:3]) if "." in layer_name else np.nan
    return np.nan


def get_stage(model, block_idx):
    """Network stage (1-4) of a block."""
    if model == "resnet50":                                   # 'layer3.2' -> 3
        return int(block_idx.split(".")[0].replace("layer", ""))
    if model == "convnext_tiny":                              # 'features.5.2' -> 3
        return {1: 1, 3: 2, 5: 3, 7: 4}[int(block_idx.split(".")[1])]
    raise ValueError(model)


def add_layer_structure(df):
    """Add layer_type and block_idx (computed once per model x layer)."""
    pairs = df[["model", "layer"]].drop_duplicates().copy()
    pairs["layer_type"] = [get_layer_type(str(m), l) for m, l in zip(pairs["model"], pairs["layer"])]
    pairs["block_idx"]  = [get_block_number(str(m), l) for m, l in zip(pairs["model"], pairs["layer"])]
    return df.merge(pairs, on=["model", "layer"], how="left")


def layer_table(df, models=LAYER_TYPE_MODELS):
    """Subject-averaged score per modality x model x measure x layer x depth,
    in-block layers only, with layer type, block and normalised depth."""
    d = df[df["model"].isin(models)]
    d = (d.groupby(["modality", "model", "measure", "layer", "brain_depth"], observed=True)
           ["score"].mean().reset_index())
    d = add_layer_depth(add_layer_structure(d))
    return d.dropna(subset=["layer_type", "block_idx"]).reset_index(drop=True)


def check_balance(lt):
    """Every block should contain each layer type exactly once."""
    for model, sub in lt.groupby("model", observed=True):
        layers = sub[["block_idx", "layer_type", "layer"]].drop_duplicates()
        ct = pd.crosstab(layers["block_idx"], layers["layer_type"])
        ct = ct.reindex(columns=LAYER_TYPE_ORDER[str(model)], fill_value=0)
        bad = ct[(ct != 1).any(axis=1)]
        print(f"{model}: {ct.shape[0]} blocks x {ct.shape[1]} types, "
              f"{'balanced' if bad.empty else f'{len(bad)} unbalanced blocks'}")
        if not bad.empty:
            display(bad)


# --- variance decomposition (subject-averaged profiles) ---

def variance_decomposition(group):
    """Share of a layer profile's variance explained by block, by layer type, and the rest.
    In a balanced design (one layer per block x type) the residual is the block x type interaction."""
    y = group["score"]
    gm = y.mean()
    ss_tot = ((y - gm) ** 2).sum()

    def ss_between(col):
        g = group.groupby(col, observed=True)["score"]
        return (g.count() * (g.mean() - gm) ** 2).sum()

    ss_block, ss_type = ss_between("block_idx"), ss_between("layer_type")
    n = len(y)
    k_block, k_type = group["block_idx"].nunique(), group["layer_type"].nunique()
    return pd.Series({
        "eta_sq_block":     ss_block / ss_tot,
        "eta_sq_layertype": ss_type / ss_tot,
        "eta_sq_residual":  (ss_tot - ss_block - ss_type) / ss_tot,
        "chance_block":     (k_block - 1) / (n - 1),      # expected eta^2 under no effect
        "chance_layertype": (k_type - 1) / (n - 1),
        "n_layers": n, "n_blocks": k_block, "n_types": k_type,
    })


def variance_decomposition_table(lt):
    """variance_decomposition per modality x model x measure x depth."""
    return (lt.groupby(["modality", "model", "measure", "brain_depth"], observed=True)
              [["score", "block_idx", "layer_type"]]
              .apply(variance_decomposition)
              .reset_index())


def block_pattern_table(df):
    """Subject average per modality, average across modalities,
    in-block layers only, then min-max scaled per measure x depth x model."""
    lt_ = layer_table(df)
    avg = (lt_.groupby(["model", "measure", "layer", "brain_depth",
                        "norm_layer_depth", "layer_type", "block_idx"], observed=True)
              ["score"].mean().reset_index())
    return _minmax(avg, ["measure", "brain_depth", "model"])


def variance_legend_handles():
    depth = [Patch(facecolor=DEPTH_PALETTE[d], label=DEPTH_LABELS[d]) for d in DEPTH_ORDER]
    comp = [
        Patch(facecolor="0.3", label="between-block"),
        Patch(facecolor="0.75", label="layer type"),
        Line2D([], [], marker="D", color="0.25", linestyle="None", markersize=4,
               markeredgecolor="white", label="block + layer type (mean)"),
        Line2D([], [], color="0.4", linestyle="--", linewidth=0.8,
               label="chance (block + layer type)"),
    ]
    return depth + comp


def plot_variance_decomposition(vd, modality=None, ax=None, show_chance=True, figsize=(7, 3.5)):
    """Stacked bars per measure x depth: between-block (dark) + layer type (light).
    Averaged over models, and over modalities unless `modality` is given."""
    d = vd if modality is None else vd[vd["modality"] == modality]
    cols = ["eta_sq_block", "eta_sq_layertype", "chance_block", "chance_layertype"]
    summ = d.groupby(["measure", "brain_depth"], observed=True)[cols].mean().reset_index()
    measures = present_measures(summ)

    n_d = len(DEPTH_ORDER)
    bw = 0.8 / n_d
    xb = np.arange(len(measures))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    fig = ax.figure

    for i, depth in enumerate(DEPTH_ORDER):
        dd = summ[summ["brain_depth"] == depth].set_index("measure").reindex(measures)
        xo = xb + (i - (n_d - 1) / 2) * bw
        ax.bar(xo, dd["eta_sq_block"], width=bw * 0.9, color=DEPTH_PALETTE[depth],
               edgecolor="white", linewidth=0.4)
        ax.bar(xo, dd["eta_sq_layertype"], bottom=dd["eta_sq_block"], width=bw * 0.9,
               color=_blend(DEPTH_PALETTE[depth], 0.6), edgecolor="white", linewidth=0.4)

    total = (summ.assign(t=summ["eta_sq_block"] + summ["eta_sq_layertype"])
                 .groupby("measure", observed=True)["t"].mean().reindex(measures))
    ax.scatter(xb, total.values, marker="D", s=18, color="0.25", zorder=5,
               edgecolor="white", linewidth=0.6)

    if show_chance:
        ax.axhline((summ["chance_block"] + summ["chance_layertype"]).mean(),
                   color="0.4", linestyle="--", linewidth=0.8, zorder=1)

    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.set_xticks(xb)
    ax.set_xticklabels(_display(measures), rotation=45, ha="right")
    ax.set_xlim(-0.6, len(measures) - 0.4)
    ax.set_ylabel("fraction of variance (η²)")
    return fig


def plot_variance_by_modality(vd, figsize=(7, 8)):
    """One panel per modality, averaged over models."""
    fig, axes = plt.subplots(len(MODALITY_ORDER), 1, figsize=figsize, sharex=True)
    for ax, mod in zip(axes, MODALITY_ORDER):
        plot_variance_decomposition(vd, modality=mod, ax=ax)
        ax.set_title(MODALITY_NAMES[mod])
    fig.legend(handles=variance_legend_handles(), loc="center left",
               bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    return fig


PATTERN_LINESTYLES = [(0, ()), (0, (5, 2)), (0, (1, 1)), (0, (3, 1, 1, 1))]


def plot_block_pattern(pat, measures=("Ridge",), models=LAYER_TYPE_MODELS, figsize=None):
    """Rows = brain depth, columns = model: scaled score along layer depth (thin line),
    block means (thick segments), residual additions (+)."""
    measures = [m for m in MEASURE_ORDER if m in measures and m in set(pat["measure"].astype(str))]
    lighten = dict(zip(measures, np.linspace(0, 0.5, len(measures)) if len(measures) > 1 else [0]))
    styles = dict(zip(measures, PATTERN_LINESTYLES))
    figsize = figsize or (3.5 * len(models), 3 * len(DEPTH_ORDER))

    fig, axes = plt.subplots(len(DEPTH_ORDER), len(models), figsize=figsize,
                             sharey=True, squeeze=False)
    for r, depth in enumerate(DEPTH_ORDER):
        for c, model in enumerate(models):
            ax = axes[r, c]
            for m in measures:
                d = pat[(pat["model"] == model) & (pat["brain_depth"] == depth)
                        & (pat["measure"] == m)].sort_values("norm_layer_depth")
                if d.empty:
                    continue
                color = _blend(DEPTH_PALETTE[depth], lighten[m])

                blocks = (d.groupby("block_idx")
                            .agg(x_min=("norm_layer_depth", "min"),
                                 x_max=("norm_layer_depth", "max"),
                                 y_mean=("score", "mean"))
                            .sort_values("x_min").reset_index(drop=True))
                x_min, x_max = blocks["x_min"].values, blocks["x_max"].values
                if len(blocks) > 1:
                    mids = (x_max[:-1] + x_min[1:]) / 2
                    lefts  = np.r_[x_min[0] - (x_min[1] - x_max[0]) / 2, mids]
                    rights = np.r_[mids, x_max[-1] + (x_min[-1] - x_max[-2]) / 2]
                else:
                    lefts, rights = x_min - 0.02, x_max + 0.02
                for left, right, ym in zip(lefts, rights, blocks["y_mean"]):
                    ax.plot([left, right], [ym, ym], color=color, linewidth=2.5,
                            solid_capstyle="butt", zorder=9)

                ax.plot(d["norm_layer_depth"], d["score"], color=color, linewidth=0.65,
                        alpha=0.5, zorder=4, linestyle=styles[m])

                add = d[d["layer_type"] == "Add"]
                ax.scatter(add["norm_layer_depth"], add["score"], color=color, s=22, marker="P",
                           edgecolors="black", linewidths=0.4, zorder=10)

            ax.spines[["top", "right"]].set_visible(False)
            if r == 0:
                ax.set_title(MODEL_NAMES[model], fontsize=13, fontweight="bold")

    fig.supylabel("Scaled alignment score", fontsize=13)
    fig.supxlabel("Normalized layer depth", fontsize=13)

    base = (0.3, 0.3, 0.3)
    handles = [Line2D([], [], color=_blend(base, lighten[m]), linewidth=2,
                      linestyle=styles[m], label=MEASURE_NAMES[m]) for m in measures]
    handles += [
        Line2D([], [], color="0.3", linewidth=5, solid_capstyle="butt", label="Block mean"),
        Line2D([], [], color="0.3", marker="P", linestyle="None", markersize=6,
               markeredgecolor="black", markeredgewidth=0.7, label="Residual addition"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.04),
               ncol=len(handles), frameon=False, fontsize=10)
    fig.tight_layout()
    return fig


# --- layer structure test: global vs local pattern, noise estimated from subjects ---

STATS = ["frac_block", "frac_layertype", "frac_interaction", "frac_noise",
         "dominance", "peak_z", "slope_z", "slope_prof"]
COMPONENTS = ["block", "layertype", "interaction"]


def subject_layer_table(df, models=LAYER_TYPE_MODELS):
    """Subject-level scores for in-block layers, with layer type, block, stage and the
    block's normalised position in the network (0 = first block, 1 = last)."""
    d = df[df["model"].isin(models)]
    d = add_layer_depth(add_layer_structure(d)).dropna(subset=["layer_type", "block_idx"])

    blocks = (d.groupby(["model", "block_idx"], observed=True)["norm_layer_depth"]
                .min().reset_index().sort_values(["model", "norm_layer_depth"]))
    blocks["block_order"] = blocks.groupby("model", observed=True).cumcount()
    blocks["block_pos"] = (blocks["block_order"]
                           / blocks.groupby("model", observed=True)["block_order"].transform("max"))
    blocks["stage"] = [get_stage(str(m), b) for m, b in zip(blocks["model"], blocks["block_idx"])]
    return d.merge(blocks[["model", "block_idx", "block_order", "block_pos", "stage"]],
                   on=["model", "block_idx"])


def _ss(V, P):
    """Sum of squares of each profile in V (..., layers) within the subspace of projector P."""
    return np.einsum("...i,ij,...j->...", V, P, V)


def _design(layers):
    """Projection matrices onto block, layer-type and interaction variance (balanced design)."""
    L = len(layers)
    J = np.full((L, L), 1 / L)

    def averaging(col):
        lab = layers[col].to_numpy()
        same = (lab[:, None] == lab[None, :]).astype(float)
        return same / same.sum(axis=1, keepdims=True)

    P = {"total": np.eye(L) - J,
         "block": averaging("block_idx") - J,
         "layertype": averaging("layer_type") - J}
    P["interaction"] = P["total"] - P["block"] - P["layertype"]
    if not np.allclose(P["interaction"] @ P["interaction"], P["interaction"], atol=1e-8):
        warnings.warn("design not balanced: interaction is not an exact projection")
    return P


def _block_contrasts(layers):
    """Add-peak weights per block (+1 at Add, -1/(k-1) at the block's other layers),
    block masks and block positions, in network order."""
    order = layers.drop_duplicates("block_idx").sort_values("block_order")
    W, masks, pos = [], [], []
    for blk, p in zip(order["block_idx"], order["block_pos"]):
        m = (layers["block_idx"] == blk).to_numpy()
        w = np.where(m, -1 / (m.sum() - 1), 0.0)
        w[m & (layers["layer_type"] == "Add").to_numpy()] = 1.0
        W.append(w)
        masks.append(m)
        pos.append(p)
    return np.array(W), masks, np.array(pos)


def _stats(Y, P, W, masks, pos):
    """Y: (..., subjects, layers) -> (..., len(STATS)).
    Components are estimated on the subject-averaged profile, minus their expected noise."""
    n = Y.shape[-2]
    ybar = Y.mean(axis=-2)
    E = Y - ybar[..., None, :]
    total = _ss(ybar, P["total"])

    comp, noise = {}, 0.0
    for c in COMPONENTS:
        nz = _ss(E, P[c]).sum(axis=-1) / (n * (n - 1))     # expected noise SS in this component
        comp[c] = _ss(ybar, P[c]) - nz
        noise = noise + nz

    c_raw = ybar @ W.T                                       # Add contrast per block
    sd_block = np.stack([ybar[..., m].std(axis=-1, ddof=1) for m in masks], axis=-1)
    sd_prof = ybar.std(axis=-1, ddof=1)[..., None]
    xc = pos - pos.mean()
    slope = lambda c: ((c - c.mean(axis=-1, keepdims=True)) @ xc) / (xc ** 2).sum()
    cz = c_raw / sd_block

    return np.stack([
        comp["block"] / total, comp["layertype"] / total,
        comp["interaction"] / total, noise / total,
        (comp["block"] - comp["layertype"]) / total,
        cz.mean(axis=-1), slope(cz), slope(c_raw / sd_prof),
    ], axis=-1)


def layer_structure_analysis(sl, n_boot=N_BOOT, seed=BOOT_SEED, min_n_boot=3):
    """Point estimates per modality x model x measure x depth, plus bootstrap replicates.
    Subjects are resampled with the same indices for every group of a modality."""
    rng = np.random.default_rng(seed)
    subjects = {m: sorted(sl.loc[sl["modality"] == m, "subject"].unique()) for m in MODALITY_ORDER}
    boot_idx = {m: (rng.integers(0, len(s), size=(n_boot, len(s))) if len(s) >= min_n_boot else None)
                for m, s in subjects.items()}

    designs, points, boots = {}, [], {}
    for key, g in sl.groupby(["modality", "model", "measure", "brain_depth"], observed=True):
        key = tuple(map(str, key))
        mod, model = key[0], key[1]
        Yd = g.pivot(index="subject", columns="layer", values="score")
        assert list(Yd.index) == subjects[mod], f"{key}: subject set differs"
        assert not Yd.isna().any().any(), f"{key}: missing layers"

        if model not in designs:
            layers = (g.drop_duplicates("layer").set_index("layer")
                       .loc[Yd.columns, ["block_idx", "layer_type", "block_order", "block_pos"]]
                       .reset_index())
            designs[model] = (list(Yd.columns), _design(layers), *_block_contrasts(layers))
        cols, P, W, masks, pos = designs[model]
        assert list(Yd.columns) == cols, f"{key}: layer set differs within {model}"

        Y = Yd.to_numpy()
        points.append({**dict(zip(["modality", "model", "measure", "brain_depth"], key)),
                       **dict(zip(STATS, _stats(Y, P, W, masks, pos))),
                       "n_subjects": len(Y), "group_key": key})
        boots[key] = None if boot_idx[mod] is None else _stats(Y[boot_idx[mod]], P, W, masks, pos)
    return pd.DataFrame(points), boots


def summarise_layer_structure(points, boots, by=("measure",), modalities=("fMRI", "EEG"), ci=0.95):
    """Mean of each statistic over groups sharing `by` (equal weight per modality),
    with percentile bootstrap CIs. Modalities without a bootstrap (ephys) get no CI."""
    q = [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100]
    pts = points[points["modality"].isin(modalities)]
    rows = []
    for key, g in pts.groupby(list(by), observed=True):
        key = key if isinstance(key, tuple) else (key,)
        point = g.groupby("modality")[STATS].mean().mean()
        per_mod = []
        for _, gm in g.groupby("modality"):
            reps = [boots[k] for k in gm["group_key"]]
            per_mod.append(None if any(r is None for r in reps) else np.mean(reps, axis=0))
        if any(r is None for r in per_mod):
            lo = hi = np.full(len(STATS), np.nan)
        else:
            lo, hi = np.percentile(np.mean(per_mod, axis=0), q, axis=0)
        for s, p_, l_, h_ in zip(STATS, point, lo, hi):
            rows.append({**dict(zip(by, key)), "stat": s, "point": p_, "low": l_, "high": h_})
    return pd.DataFrame(rows)


def stage_patterns(sl, measures=None):
    """Within-block z-scored pattern per model x stage x layer type
    (subject-averaged; equal weight per modality)."""
    d = sl if measures is None else sl[sl["measure"].isin(measures)]
    keys = ["modality", "model", "measure", "brain_depth", "block_idx"]
    avg = (d.groupby(keys + ["stage", "layer", "layer_type"], observed=True)["score"]
             .mean().reset_index())
    g = avg.groupby(keys, observed=True)["score"]
    avg["z"] = (avg["score"] - g.transform("mean")) / g.transform("std")
    per_mod = avg.groupby(["modality", "model", "stage", "layer_type"], observed=True)["z"].mean().reset_index()
    return per_mod.groupby(["model", "stage", "layer_type"], observed=True)["z"].mean().reset_index()


def plot_layer_structure(summ, figsize=(7, 3.5)):
    """Stacked bars per measure: noise-corrected between-block, layer type and interaction
    shares, plus the noise share."""
    measures = [m for m in MEASURE_ORDER if m in set(summ["measure"])]
    s = summ.pivot(index="measure", columns="stat", values="point").reindex(measures)
    parts = [("frac_block", "between-block", "0.25", None),
             ("frac_layertype", "layer type", "0.55", None),
             ("frac_interaction", "block × layer type", "0.8", None),
             ("frac_noise", "noise", "white", "////")]

    fig, ax = plt.subplots(figsize=figsize)
    x, bottom = np.arange(len(measures)), np.zeros(len(measures))
    for col, label, color, hatch in parts:
        v = s[col].clip(lower=0).to_numpy()
        ax.bar(x, v, bottom=bottom, width=0.7, color=color, edgecolor="0.3",
               linewidth=0.5, hatch=hatch, label=label)
        bottom += v

    ax.set_xticks(x)
    ax.set_xticklabels(_display(measures), rotation=45, ha="right")
    ax.set_xlim(-0.6, len(measures) - 0.4)
    ax.set_ylim(0, max(1, bottom.max()))
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.set_ylabel("fraction of variance")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    return fig


def plot_stat_by_measure(summ, stat, ylabel, ax=None, figsize=(6, 3)):
    """Point + bootstrap CI per measure; filled if the CI excludes 0, open otherwise."""
    measures = [m for m in MEASURE_ORDER if m in set(summ["measure"])]
    s = summ[summ["stat"] == stat].set_index("measure").reindex(measures)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    for i, (_, r) in enumerate(s.iterrows()):
        excludes_0 = np.isfinite(r["low"]) and (r["low"] > 0 or r["high"] < 0)
        ax.errorbar(i, r["point"], yerr=[[r["point"] - r["low"]], [r["high"] - r["point"]]],
                    fmt="o", color="0.2", mfc="0.2" if excludes_0 else "white",
                    markersize=5, elinewidth=0.8)
    ax.axhline(0, color="0.6", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_xticks(range(len(measures)))
    ax.set_xticklabels(_display(measures), rotation=45, ha="right")
    ax.set_xlim(-0.6, len(measures) - 0.4)
    ax.set_ylabel(ylabel)
    return ax.figure


def _shade_add(ax, cats):
    """Light band behind the residual-addition ('Add') positions."""
    for pos in [i for i, c in enumerate(cats) if c == "Add"]:
        ax.axvspan(pos - 0.4, pos + 0.4, color="#f4a259", alpha=0.15, zorder=0)


def plot_stage_patterns(sp, models=LAYER_TYPE_MODELS, figsize=(8, 3.5)):
    """Within-block pattern per network stage (light = early, dark = late)."""
    stages = sorted(sp["stage"].unique())
    colors = plt.cm.Greys(np.linspace(0.35, 0.95, len(stages)))
    fig, axes = plt.subplots(1, len(models), figsize=figsize, sharey=True, squeeze=False)
    for ax, model in zip(axes[0], models):
        cats = LAYER_TYPE_ORDER[model]
        x = np.arange(len(cats))
        _shade_add(ax, cats)
        for s, col in zip(stages, colors):
            y = (sp[(sp["model"] == model) & (sp["stage"] == s)]
                   .set_index("layer_type")["z"].reindex(cats))
            ax.plot(x, y.values, color=col, marker="o", markersize=3.5, linewidth=1.2,
                    label=f"stage {s}")
        ax.axhline(0, color="0.7", linewidth=0.6, linestyle="--", zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=45, ha="right")
        ax.set_title(MODEL_NAMES[model])
    axes[0, 0].set_ylabel("score (z-scored within block)")
    axes[0, -1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    return fig


# --- residual blocks: within-block z-scores per layer type ---

def block_zscores(df, models=LAYER_TYPE_MODELS, zscore_over="typed"):
    """Subject-averaged scores, z-scored within each modality x model x measure x block x depth.

    zscore_over="typed":        mean/std over the layers that have a layer type (default)
    zscore_over="all_in_block": also include untyped layers inside a block (e.g. ResNet
                                downsample), which is what the old notebook did
    """
    d = df[df["model"].isin(models)]
    d = (d.groupby(["modality", "model", "measure", "layer", "brain_depth"], observed=True)
           ["score"].mean().reset_index())
    d = add_layer_structure(d).dropna(subset=["block_idx"])
    if zscore_over == "typed":
        d = d.dropna(subset=["layer_type"])

    g = d.groupby(["modality", "model", "measure", "block_idx", "brain_depth"], observed=True)["score"]
    d["z"] = (d["score"] - g.transform("mean")) / g.transform("std")
    return d.dropna(subset=["layer_type"]).reset_index(drop=True)


def layer_type_profiles(z):
    """Mean z per layer type: over blocks and depths within each modality, then over modalities."""
    per_mod = (z.groupby(["modality", "model", "measure", "layer_type"], observed=True)
                ["z"].mean().reset_index())
    return (per_mod.groupby(["model", "measure", "layer_type"], observed=True)
                   ["z"].mean().reset_index())


def plot_layer_type_pattern(prof, models=LAYER_TYPE_MODELS, figsize=None, ylim=(-1.7, 1.7)):
    """Per model: within-block z-score per layer type, one grey line per measure,
    red bar = mean across measures, residual additions highlighted."""
    figsize = figsize or (4 * len(models), 5)
    fig, axes = plt.subplots(1, len(models), figsize=figsize, sharey=True, squeeze=False)
    axes = axes[0]

    for ax, model in zip(axes, models):
        cats = LAYER_TYPE_ORDER[model]
        x = np.arange(len(cats))
        sub = prof[prof["model"] == model]
        measures = present_measures(sub)
        _shade_add(ax, cats)

        series = []
        for m in measures:
            y = sub[sub["measure"] == m].set_index("layer_type")["z"].reindex(cats).values
            series.append(y)
            ax.plot(x, y, marker=MEASURE_MARKERS[m], markersize=4, linewidth=1,
                    color="0.6", alpha=0.5, zorder=2)

        trend = np.nanmean(np.vstack(series), axis=0)
        ax.bar(x, trend, width=0.6, color="#c1121f", alpha=0.8, zorder=1)

        lo, hi = np.nanmin(np.vstack(series)), np.nanmax(np.vstack(series))
        if lo < ylim[0] or hi > ylim[1]:
            warnings.warn(f"{model}: values [{lo:.2f}, {hi:.2f}] exceed ylim {ylim}")

        ax.axhline(0, color="0.7", linewidth=0.8, linestyle="--", zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels(cats)
        ax.set_title(MODEL_NAMES[model], fontsize=14, fontweight="bold")
        ax.tick_params(axis="x", rotation=45, labelsize=10)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="y", linestyle="-", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Score (z-scored within block)", fontsize=13)
    axes[0].set_ylim(*ylim)

    present = present_measures(prof)
    handles = [Line2D([], [], color="0.6", marker=MEASURE_MARKERS[m], linestyle="None",
                      markersize=7, label=MEASURE_NAMES[m]) for m in present]
    handles += [
        Patch(color="#c1121f", alpha=0.8, label="Mean across measures"),
        Line2D([], [], color="#f4a259", marker="s", linestyle="None", markersize=10,
               alpha=0.4, label="Residual addition"),
    ]
    fig.legend(handles=handles, title="Measure", bbox_to_anchor=(1.01, 0.5),
               loc="center left", frameon=False)
    fig.tight_layout()
    return fig


# =============================================================================
# 10. PCA curves (supplement)
# =============================================================================

SPLIT_STYLE = {
    "brain_depth": {d: (DEPTH_PALETTE[d], "o", DEPTH_LABELS[d]) for d in DEPTH_ORDER},
    "model": {m: (c, mk, MODEL_NAMES[m])
              for m, c, mk in zip(MODEL_ORDER, ["0.15", "0.45", "0.7"], ["o", "s", "D"])},
}


def pca_score_curves(df_sweep, how="max", split=None):
    """Score per subject: best-layer score (how="max") or mean over layers (how="mean") within each
    model x depth group, averaged over the groups not in `split` ('brain_depth', 'model' or None),
    then mean + CI across subjects."""
    lv = (df_sweep.groupby(["modality", "model", "subject", "brain_depth", "measure", "n_components"],
                           observed=True)["score"].agg(how).reset_index())
    keys = ["modality", "measure", "n_components"] + ([split] if split else [])
    per_subj = lv.groupby(keys + ["subject"], observed=True)["score"].mean().reset_index()
    return _summarise_ci(per_subj, keys, "score")


def sample_layers(sampled_layers, df_sweep):
    """{model: {position: layer}} -> table with normalised depth; checks the layers exist."""
    sampled = pd.DataFrame([{"model": model, "position": pos, "layer": layer}
                            for model, layers in sampled_layers.items()
                            for pos, layer in layers.items()])
    known = set(zip(df_sweep["model"].astype(str), df_sweep["layer"]))
    missing = [(m, l) for m, l in zip(sampled["model"], sampled["layer"]) if (m, l) not in known]
    assert not missing, f"layers not in the data: {missing}"
    return add_layer_depth(sampled)


def layer_pca_curves(df_sweep, sampled):
    """Mean score + CI across subjects per modality x model x sampled layer x measure x depth x n."""
    d = (df_sweep.assign(model=df_sweep["model"].astype(str))
                 .merge(sampled[["model", "layer", "position"]], on=["model", "layer"]))
    keys = ["modality", "model", "position", "layer", "measure", "brain_depth", "n_components"]
    return _summarise_ci(d, keys, "score")


def _curve(ax, s, color, marker, label=None):
    s = s.set_index("n_components").reindex(N_ORDER).dropna(subset=["mean"])
    x = [N_ORDER.index(n) for n in s.index]
    ax.errorbar(x, s["mean"], yerr=[s["mean"] - s["low"], s["high"] - s["mean"]],
                color=color, marker=marker, markersize=4, linewidth=1.2, elinewidth=0.8,
                label=label)


def plot_pca_curves_pooled(curves, measures=PCA_MEASURES, figsize=None, ylabel="best-layer score"):
    """One panel per measure, one line per modality (pooled over regions and models)."""
    figsize = figsize or (3.2 * len(measures), 3.2)
    fig, axes = plt.subplots(1, len(measures), figsize=figsize, squeeze=False)
    for ax, m in zip(axes[0], measures):
        for mod in MODALITY_ORDER:
            color, marker = MODALITY_STYLE[mod]
            _curve(ax, curves[(curves["measure"] == m) & (curves["modality"] == mod)],
                   color, marker, MODALITY_NAMES[mod])
        _sweep_axis(ax, STAB_REF[m])
        ax.set_title(MEASURE_NAMES[m])
        ax.set_xlabel("number of components")
    axes[0, 0].set_ylabel(ylabel)
    axes[0, -1].legend(loc="best")
    fig.tight_layout()
    return fig


def plot_pca_curves_split(curves, split, measures=PCA_MEASURES, figsize=None,
                          ylabel="best-layer score"):
    """Rows = measure, columns = modality, one line per level of `split`."""
    style = SPLIT_STYLE[split]
    figsize = figsize or (3.3 * len(MODALITY_ORDER) + 1.5, 2.6 * len(measures))
    fig, axes = plt.subplots(len(measures), len(MODALITY_ORDER), figsize=figsize,
                             sharex=True, squeeze=False)
    for r, m in enumerate(measures):
        for c, mod in enumerate(MODALITY_ORDER):
            ax = axes[r, c]
            sub = curves[(curves["measure"] == m) & (curves["modality"] == mod)]
            for level, (color, marker, _) in style.items():
                _curve(ax, sub[sub[split] == level], color, marker)
            _sweep_axis(ax, STAB_REF[m])
            if r == 0:
                ax.set_title(MODALITY_NAMES[mod], pad=8)
            if c == 0:
                ax.set_ylabel(ylabel)
                _row_label(ax, MEASURE_NAMES[m])
            if r == len(measures) - 1:
                ax.set_xlabel("number of components")

    handles = [Line2D([], [], color=color, marker=marker, markersize=4, label=label)
               for color, marker, label in style.values()]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    return fig


def plot_layer_pca_curves(curves, sampled, modality, model, var_explained=None,
                          measures=PCA_MEASURES, figsize=None):
    """Rows = measure, columns = sampled layer (early / mid / late); one line per brain region.
    Top axis: variance explained by each component count in that layer
    (`var_explained` from variance_explained_lookup)."""
    var_explained = var_explained or {}
    layers = sampled[sampled["model"] == model].set_index("position")["layer"]
    positions = list(layers.index)
    figsize = figsize or (3.4 * len(positions) + 1.5, 2.6 * len(measures) + 0.6)

    fig, axes = plt.subplots(len(measures), len(positions), figsize=figsize,
                             sharex=True, squeeze=False)
    for r, m in enumerate(measures):
        for c, pos in enumerate(positions):
            ax, layer = axes[r, c], layers[pos]
            sub = curves[(curves["modality"] == modality) & (curves["model"] == model)
                         & (curves["layer"] == layer) & (curves["measure"] == m)]
            for depth in DEPTH_ORDER:
                _curve(ax, sub[sub["brain_depth"] == depth], DEPTH_PALETTE[depth], "o")
            _sweep_axis(ax, STAB_REF[m])

            if r == 0:                                   # variance explained, once per column
                labels = []
                for n in N_ORDER:
                    if n == "none":
                        labels.append("100")
                    else:
                        v = var_explained.get((modality, model, layer, n))
                        labels.append("" if v is None or pd.isna(v) else f"{100 * v:.0f}")
                sec = ax.secondary_xaxis("top")
                sec.set_xticks(range(len(N_ORDER)))
                sec.set_xticklabels(labels, fontsize=mpl.rcParams["xtick.labelsize"] - 1)
                sec.set_xlabel("variance explained (%)", fontsize=mpl.rcParams["xtick.labelsize"])
                ax.set_title(f"{pos}: {layer}", pad=36)
            if c == 0:
                ax.set_ylabel("score")
                _row_label(ax, MEASURE_NAMES[m])
            if r == len(measures) - 1:
                ax.set_xlabel("number of components")

    handles = [Line2D([], [], color=DEPTH_PALETTE[d], marker="o", markersize=4,
                      label=DEPTH_LABELS[d]) for d in DEPTH_ORDER]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle(f"{MODALITY_NAMES[modality]} | {MODEL_NAMES[model]}")
    fig.tight_layout()
    return fig
