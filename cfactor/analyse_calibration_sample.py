"""In-sample analysis of the calibration set.

Scope
-----
This script analyses **only the parcels that were sampled to calibrate β**
(i.e. the rows in `calibration_results_per_pixel.csv`). It does NOT apply
the calibrated method to fresh parcels — that is the job of
`apply_and_compare.py`.

The goal here is to answer the calibration-quality questions you can answer
without geometry, an AOI, or the R pipeline:

  - Is the global β fit good for every crop, or are there crops where the
    per-pixel mean drifts away from the crop-level fit?
  - Where does the disagreement (C_new − C_ref) live?  Among small-n crops?
    Among low-quality FC pixels?  In particular years?  At low or high
    reference C-magnitude?
  - Is the per-pixel spread of C_new informative — i.e. does the satellite
    actually see within-crop variation, or are pixels of the same crop
    clustered together?
  - Does β land in the bowl, or at the search bounds?

Modes
-----
The script auto-detects which calibration output it is reading:

- *Unstratified* (legacy): per-pixel CSV holds one C_ref per crop (the
  `Total` column of C_Faktoren.csv). The calibration target is per-crop.

- *Stratified*: per-pixel CSV also carries `region` (Tal/Berg) and
  `tillage` (Pflug/Mulch/Direkt) columns and a per-stratum C_ref already
  joined in by `calibrate_cfactor.run_calibration_stratified`. The
  calibration target is per-stratum (up to 6 cells per crop). All
  per-crop diagnostics are replaced by per-stratum equivalents.

Detection is by column presence (`region` + `tillage`). Override with
`cfg['stratified_mode']` if you want to force it.

Inputs
------
- per-pixel CSV (auto-named):
    unstratified: `calibration_results_per_pixel.csv`
    stratified:   `calibration_results_stratified_per_pixel.csv`
- per-crop / per-stratum CSV (companion diag from the calibrator)
- `C_Faktoren.csv`                      project file (Total only; for
                                                       attaching C_ref in
                                                       unstratified mode)
- `LNF_classification.xlsx`             project file (lnf_code ↔ Crop_DE)
- (optional) gapfilled FC parquet       from sample_FC.py — only used if you
                                         want by-quality-flag and by-FC stats
                                         (`fc_quality_flag`, fc_total mean…).

What it produces
----------------
Unstratified mode:
- `cal_summary.txt`                     human-readable headline numbers
- `cal_per_crop.csv`                    per-crop bias/MAE/RMSE/spread/n + C_ref
- `cal_per_year.csv`                    interannual stability of C_new per crop
- `cal_by_quality.csv` (if FC parquet)  by `fc_quality_flag` (sampling QC)
- `cal_by_c_magnitude.csv`              by reference-C magnitude class
- plots/
    cal_scatter_per_crop.png            crop-mean predicted vs reference
                                         (Matthews-style, with point size ~ n)
    cal_strengths_weaknesses.png        bias × within-crop spread per crop
    cal_box_per_crop.png                C_new boxplot per crop with C_ref dot
    cal_interannual.png                 C_new vs year per crop
    cal_residual_vs_n.png               |bias| vs sample size per crop
    cal_quality_bars.png (if FC parquet) bias/MAE per FC quality flag

Stratified mode (same skeleton, granularity raised to stratum):
- `cal_summary.txt`                     adds per-region and per-tillage rollups
- `cal_per_stratum.csv`                 per (crop, region, tillage) cell
- `cal_per_crop.csv`                    also kept — mean across that crop's strata
- `cal_per_year.csv`                    per (crop, region, tillage, yr)
- `cal_by_quality.csv` / `cal_by_c_magnitude.csv` — unchanged
- `cal_by_region.csv`, `cal_by_tillage.csv`        — new rollups
- plots/  same skeleton, suffixed `_per_stratum`. Strata identified by
  colour=region + marker=tillage on every plot where it fits.

The script does NOT re-fit β. β has already been chosen; we only diagnose
how good that choice was on the calibration set.
"""

from __future__ import annotations

import os
import unicodedata
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ===========================================================================
# Default config — edit paths to your environment.
# ===========================================================================
CONFIG = {
    'results_dir':              'calibration_analysis_default',
    'per_pixel_path':           'calibration_results_per_pixel.csv',
    'per_crop_path':            'calibration_results.csv',
    'c_factor_table_path':      'C_Faktoren.csv',
    'lnf_classification_path':  '~/mnt/eo-nas1/data/landuse/documentation/'
                                'LNF_code_classification_20260217.xlsx',

    # Optional — gives you per-quality-flag diagnostics. Same parquet the
    # calibration consumed.
    'gapfilled_fc_path':        'samples_data_gpr.parquet',
    'min_n_per_crop':           10,   # crops with fewer pixels are downweighted
                                       # in the strengths/weaknesses scatter

    # Stratified mode: None → auto-detect from per-pixel CSV columns
    # (`region` + `tillage` present → stratified). Set to True/False to force.
    'stratified_mode':          None,
}


# Plot styling for strata — kept in module scope so all plots match.
REGION_COLORS  = {'Tal': '#2E86AB', 'Berg': '#A23B72'}
TILLAGE_MARKERS = {'Pflug': 'o', 'Mulch': 's', 'Direkt': '^'}


# ===========================================================================
# Helpers
# ===========================================================================
def _norm_name(s: str) -> str:
    if not isinstance(s, str):
        return ''
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r'[^\w\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip().lower()


def load_c_table_total(c_factor_table_path: str) -> pd.DataFrame:
    """Return one row per crop with C_ref (the `Total` column) + normalised key."""
    df = pd.read_csv(c_factor_table_path, sep=';', encoding='cp1252')
    df = df.rename(columns={'Kultur Kategorien 2020': 'crop_de'})
    df['C_ref'] = pd.to_numeric(df['Total'], errors='coerce')
    df['_norm'] = df['crop_de'].apply(_norm_name)
    return df[['crop_de', 'C_ref', '_norm']].dropna(subset=['C_ref']).drop_duplicates()


def load_lnf_bridge(lnf_path: str) -> pd.DataFrame:
    """Return lnf_code ↔ Crop_DE bridge (matches calibrate_cfactor.py)."""
    df = pd.read_excel(os.path.expanduser(lnf_path), sheet_name='label_sheet')
    bridge = (df[['LNF_code', 'Crop_DE']]
              .dropna().drop_duplicates()
              .rename(columns={'LNF_code': 'lnf_code',
                               'Crop_DE': 'crop_de'}))
    bridge['_norm'] = bridge['crop_de'].apply(_norm_name)
    return bridge


def attach_ref(df_pixel: pd.DataFrame, c_table: pd.DataFrame,
               lnf_bridge: pd.DataFrame, stratified: bool) -> pd.DataFrame:
    """Attach crop_de and (in unstratified mode) C_ref to per-pixel data.

    In stratified mode the per-pixel CSV already carries a per-stratum
    C_ref written by `calibrate_cfactor.run_calibration_stratified`, so
    we only attach crop_de and leave C_ref untouched.
    """
    df = df_pixel.merge(lnf_bridge[['lnf_code', 'crop_de', '_norm']],
                        on='lnf_code', how='left')
    if not stratified:
        df = df.merge(c_table[['_norm', 'C_ref']], on='_norm', how='left')
    elif 'C_ref' not in df.columns:
        # Defensive: stratified per-pixel files written by an older
        # calibrate_cfactor may not have C_ref. Fall back to Total.
        print("  WARNING: stratified file has no C_ref column — falling back "
              "to Total. Update calibrate_cfactor.py for proper stratified C_ref.")
        df = df.merge(c_table[['_norm', 'C_ref']], on='_norm', how='left')
    return df.drop(columns='_norm')


def stratum_metrics(g: pd.DataFrame, ref_col: str = 'C_ref',
                    new_col: str = 'C_predicted') -> dict:
    sub = g.dropna(subset=[ref_col, new_col])
    n = len(sub)
    if n == 0:
        return dict(n=0, mean_ref=np.nan, mean_new=np.nan,
                    bias=np.nan, MAE=np.nan, RMSE=np.nan, spread=np.nan)
    diff = sub[new_col].values - sub[ref_col].values
    return dict(
        n=int(n),
        mean_ref=float(sub[ref_col].mean()),
        mean_new=float(sub[new_col].mean()),
        bias=float(diff.mean()),
        MAE=float(np.abs(diff).mean()),
        RMSE=float(np.sqrt((diff ** 2).mean())),
        spread=float(sub[new_col].std()) if n > 1 else 0.0,
    )


def _detect_stratified(per_pixel: pd.DataFrame, override: bool | None) -> bool:
    if override is not None:
        return bool(override)
    return ('region' in per_pixel.columns) and ('tillage' in per_pixel.columns)


def _stratum_label(row, max_crop_chars: int = 18) -> str:
    """Compact label for a (crop, region, tillage) row in stratified plots."""
    crop = str(row.get('crop_de', row.get('lnf_code', '?')))[:max_crop_chars]
    return f"{crop} | {row['region'][0]}/{row['tillage'][:3]}"


# ===========================================================================
# Plots — unstratified
# ===========================================================================
def plot_scatter_per_crop(per_crop: pd.DataFrame, out: str) -> None:
    """Crop-mean predicted vs reference. Point size ~ √n."""
    df = per_crop.dropna(subset=['mean_ref', 'mean_new']).copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 7))
    n = df['n'].clip(lower=1).values
    sizes = 30 + 250 * np.sqrt(n / n.max())
    ax.scatter(df['mean_ref'], df['mean_new'], s=sizes, alpha=0.7,
               edgecolor='k', linewidth=0.4)
    for _, r in df.iterrows():
        ax.annotate(str(r['crop_de'])[:22],
                    (r['mean_ref'], r['mean_new']),
                    fontsize=7, alpha=0.8,
                    xytext=(4, 3), textcoords='offset points')
    m = max(df['mean_ref'].max(), df['mean_new'].max()) * 1.1
    ax.plot([0, m], [0, m], 'k--', lw=1, alpha=0.5, label='1:1')
    bias = (df['mean_new'] - df['mean_ref']).mean()
    mae  = (df['mean_new'] - df['mean_ref']).abs().mean()
    ax.set_xlabel('C_ref  (per-crop Total in C_Faktoren.csv)')
    ax.set_ylabel('C_new  (mean of sampled-pixel SLR with calibrated β)')
    ax.set_title(f'Calibration sample — crop-mean predicted vs reference\n'
                 f'n_crops={len(df)}  bias={bias:+.4f}  MAE={mae:.4f}')
    ax.set_xlim(0, m); ax.set_ylim(0, m)
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"  saved {out}")


def plot_strengths_weaknesses(per_crop: pd.DataFrame, out: str,
                              min_n: int = 10) -> None:
    """bias × spread per crop. Origin = agrees with table & no within-crop
    variation. Top-left/right = high within-crop variation. Far from x=0 =
    systematic mismatch."""
    df = per_crop[per_crop['n'] >= min_n].dropna(subset=['bias', 'spread']).copy()
    if df.empty:
        print('  (not enough data per crop for strengths/weaknesses)')
        return
    fig, ax = plt.subplots(figsize=(11, 7))
    sizes = 30 + 6 * np.sqrt(df['n'].clip(upper=10000))
    sc = ax.scatter(df['bias'], df['spread'], s=sizes,
                    c=df['bias'].abs(), cmap='Reds',
                    alpha=0.8, edgecolor='k', linewidth=0.5)
    ax.axvline(0, color='k', lw=0.7, alpha=0.6)
    for _, r in df.iterrows():
        ax.annotate(str(r['crop_de'])[:22], (r['bias'], r['spread']),
                    fontsize=7, alpha=0.85,
                    xytext=(4, 3), textcoords='offset points')
    ax.set_xlabel('bias  (C_new − C_ref)  →  S2 systematically higher')
    ax.set_ylabel('within-crop spread of C_new (σ across pixels)')
    ax.set_title('Per-crop strengths & weaknesses on calibration sample\n'
                 'origin = agrees with table & no within-crop variation')
    plt.colorbar(sc, ax=ax, label='|bias|')
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"  saved {out}")


def plot_box_per_crop(per_pixel: pd.DataFrame, out: str,
                      min_n: int = 10) -> None:
    counts = per_pixel.groupby('crop_de').size()
    keep = counts[counts >= min_n].index
    sub = per_pixel[per_pixel['crop_de'].isin(keep)].copy()
    if sub.empty:
        return
    order = (sub.groupby('crop_de')['C_predicted'].median()
                .sort_values().index.tolist())
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(order)), 6))
    data = [sub.loc[sub['crop_de'] == c, 'C_predicted'].values for c in order]
    ax.boxplot(data, labels=[c[:25] for c in order], showfliers=False,
               medianprops=dict(color='steelblue', lw=1.5))
    for i, c in enumerate(order, start=1):
        ref = sub.loc[sub['crop_de'] == c, 'C_ref'].iloc[0]
        ax.scatter(i, ref, color='red', s=40, zorder=5,
                   label='C_ref (Total)' if i == 1 else None)
    ax.set_ylabel('C-factor')
    ax.set_title('Calibration sample — C_new distribution per crop, with C_ref')
    plt.xticks(rotation=70, ha='right', fontsize=8)
    ax.legend(loc='upper left')
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"  saved {out}")


def plot_interannual(per_pixel: pd.DataFrame, out: str,
                     min_n: int = 30) -> None:
    """Mean C_new per (crop, year). Tells you if the satellite product is
    interannually stable or sees real year-to-year variation."""
    if 'yr' not in per_pixel.columns or per_pixel['yr'].nunique() < 2:
        return
    counts = per_pixel.groupby('crop_de').size()
    keep = counts[counts >= min_n].index
    sub = per_pixel[per_pixel['crop_de'].isin(keep)].copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for crop, g in sub.groupby('crop_de'):
        yearly = g.groupby('yr')['C_predicted'].mean()
        ax.plot(yearly.index, yearly.values, marker='o', label=crop[:25])
    ax.set_xlabel('Year')
    ax.set_ylabel('Mean C_new on calibration sample')
    ax.set_title('Interannual variation in S2-derived C-factor — '
                 'invisible to the static table')
    ax.legend(fontsize=7, ncol=2, loc='upper right')
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"  saved {out}")


def plot_residual_vs_n(per_crop: pd.DataFrame, out: str) -> None:
    """|bias| vs n. Crops with high |bias| AND high n are the meaningful
    weaknesses; high |bias| at low n is just sampling noise."""
    df = per_crop.dropna(subset=['bias', 'n']).copy()
    if df.empty:
        return
    df['abs_bias'] = df['bias'].abs()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(df['n'], df['abs_bias'], s=40, alpha=0.7, edgecolor='k', lw=0.4)
    for _, r in df.iterrows():
        if r['abs_bias'] > df['abs_bias'].median():
            ax.annotate(str(r['crop_de'])[:18], (r['n'], r['abs_bias']),
                        fontsize=7, alpha=0.7,
                        xytext=(4, 3), textcoords='offset points')
    ax.set_xscale('log')
    ax.set_xlabel('Sample size n (parcel-pixels)')
    ax.set_ylabel('|bias|  =  |C_new_mean − C_ref|')
    ax.set_title('Where do residuals come from? '
                 '(meaningful weaknesses are high-n + high |bias|)')
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"  saved {out}")


def plot_quality_bars(by_q: pd.DataFrame, out: str) -> None:
    if by_q is None or by_q.empty:
        return
    by_q = by_q.dropna(subset=['fc_quality_flag']).copy()
    if by_q.empty:
        return
    by_q['fc_quality_flag'] = by_q['fc_quality_flag'].astype(int)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = by_q['fc_quality_flag'].astype(str).values
    ax.bar(x, by_q['MAE'].values, color='steelblue', alpha=0.7, label='MAE')
    ax.plot(x, by_q['bias'].values, 'o-', color='firebrick', lw=1.5,
            label='bias')
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel('FC quality flag (0 = best, 2 = worst)')
    ax.set_ylabel('C-factor units')
    ax.set_title('Calibration disagreement by S2 FC quality\n'
                 '(if MAE rises with flag, residuals are sampling noise)')
    ax.legend()
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"  saved {out}")


# ===========================================================================
# Plots — stratified (per-stratum equivalents)
# ===========================================================================
def plot_scatter_per_stratum(per_stratum: pd.DataFrame, out: str) -> None:
    """One point per (crop, region, tillage) stratum.

    Colour by region, marker by tillage. Point size ~ √n. Each crop
    appears up to 6 times.
    """
    df = per_stratum.dropna(subset=['mean_ref', 'mean_new']).copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 8))
    n = df['n'].clip(lower=1).values
    sizes = 30 + 250 * np.sqrt(n / max(n.max(), 1))
    df['_size'] = sizes
    for (reg, til), sub in df.groupby(['region', 'tillage']):
        ax.scatter(sub['mean_ref'], sub['mean_new'],
                   s=sub['_size'].values,
                   c=REGION_COLORS.get(reg, 'grey'),
                   marker=TILLAGE_MARKERS.get(til, 'x'),
                   alpha=0.75, edgecolor='k', linewidth=0.4,
                   label=f'{reg}/{til}')
    # Annotate only the visually-interesting ones (largest residuals or
    # largest n) to avoid clutter.
    df['_abs_resid'] = (df['mean_new'] - df['mean_ref']).abs()
    interesting = pd.concat([
        df.nlargest(8, '_abs_resid'),
        df.nlargest(8, 'n'),
    ]).drop_duplicates()
    for _, r in interesting.iterrows():
        ax.annotate(_stratum_label(r),
                    (r['mean_ref'], r['mean_new']),
                    fontsize=6.5, alpha=0.8,
                    xytext=(4, 3), textcoords='offset points')
    m = max(df['mean_ref'].max(), df['mean_new'].max()) * 1.1
    ax.plot([0, m], [0, m], 'k--', lw=1, alpha=0.5, label='1:1')
    bias = (df['mean_new'] - df['mean_ref']).mean()
    mae  = (df['mean_new'] - df['mean_ref']).abs().mean()
    ax.set_xlabel('C_ref  (stratified C_Faktoren entry)')
    ax.set_ylabel('C_new  (mean of sampled-pixel SLR with calibrated β)')
    ax.set_title(f'Calibration sample — stratum-mean predicted vs reference\n'
                 f'n_strata={len(df)}  bias={bias:+.4f}  MAE={mae:.4f}')
    ax.set_xlim(0, m); ax.set_ylim(0, m)
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"  saved {out}")


def plot_strengths_weaknesses_stratum(per_stratum: pd.DataFrame, out: str,
                                       min_n: int = 10) -> None:
    """bias × spread per stratum, coloured by region, markered by tillage."""
    df = per_stratum[per_stratum['n'] >= min_n].dropna(subset=['bias', 'spread']).copy()
    if df.empty:
        print('  (not enough data per stratum for strengths/weaknesses)')
        return
    fig, ax = plt.subplots(figsize=(11, 7))
    sizes = 30 + 6 * np.sqrt(df['n'].clip(upper=10000))
    df['_size'] = sizes
    for (reg, til), sub in df.groupby(['region', 'tillage']):
        ax.scatter(sub['bias'], sub['spread'], s=sub['_size'].values,
                   c=REGION_COLORS.get(reg, 'grey'),
                   marker=TILLAGE_MARKERS.get(til, 'x'),
                   alpha=0.8, edgecolor='k', linewidth=0.4,
                   label=f'{reg}/{til}')
    ax.axvline(0, color='k', lw=0.7, alpha=0.6)
    df['_abs_bias'] = df['bias'].abs()
    for _, r in df.nlargest(15, '_abs_bias').iterrows():
        ax.annotate(_stratum_label(r), (r['bias'], r['spread']),
                    fontsize=6.5, alpha=0.85,
                    xytext=(4, 3), textcoords='offset points')
    ax.set_xlabel('bias  (C_new − C_ref)  →  S2 systematically higher')
    ax.set_ylabel('within-stratum spread of C_new (σ across pixels)')
    ax.set_title('Per-stratum strengths & weaknesses on calibration sample')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"  saved {out}")


def plot_box_per_stratum(per_pixel: pd.DataFrame, out: str,
                          min_n: int = 10) -> None:
    """One box per (crop, region, tillage) with the matching C_ref marked.

    Crops are kept together: boxes are sorted by crop_de first, then by
    stratum label, so the reader can scan a crop's 6 strata side-by-side.
    """
    grp_cols = ['crop_de', 'region', 'tillage']
    counts = per_pixel.groupby(grp_cols).size()
    keep_keys = counts[counts >= min_n].index
    sub = per_pixel.set_index(grp_cols).loc[keep_keys].reset_index()
    if sub.empty:
        return
    # Order by crop_de's overall median, then by region+tillage within crop
    crop_med = (sub.groupby('crop_de')['C_predicted'].median()
                   .sort_values().index.tolist())
    sub['crop_de'] = pd.Categorical(sub['crop_de'], categories=crop_med, ordered=True)
    sub = sub.sort_values(['crop_de', 'region', 'tillage'])
    order = list(sub.groupby(grp_cols, observed=True).groups.keys())
    data   = [sub.loc[(sub['crop_de'] == c)
                      & (sub['region'] == r)
                      & (sub['tillage'] == t), 'C_predicted'].values
              for (c, r, t) in order]
    labels = [f"{str(c)[:18]} | {r[0]}/{t[:3]}" for (c, r, t) in order]
    fig, ax = plt.subplots(figsize=(max(10, 0.30 * len(order)), 6))
    ax.boxplot(data, labels=labels, showfliers=False,
               medianprops=dict(color='steelblue', lw=1.5))
    for i, (c, r, t) in enumerate(order, start=1):
        ref_vals = sub.loc[(sub['crop_de'] == c)
                            & (sub['region'] == r)
                            & (sub['tillage'] == t), 'C_ref']
        if len(ref_vals):
            ax.scatter(i, ref_vals.iloc[0], color='red', s=30, zorder=5,
                       label='C_ref (stratum)' if i == 1 else None)
    ax.set_ylabel('C-factor')
    ax.set_title('Calibration sample — C_new distribution per stratum, with C_ref')
    plt.xticks(rotation=80, ha='right', fontsize=7)
    ax.legend(loc='upper left')
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"  saved {out}")


def plot_interannual_stratum(per_pixel: pd.DataFrame, out: str,
                              min_n: int = 30) -> None:
    """Mean C_new per (crop, region, tillage, year). One line per stratum."""
    if 'yr' not in per_pixel.columns or per_pixel['yr'].nunique() < 2:
        return
    grp_cols = ['crop_de', 'region', 'tillage']
    counts = per_pixel.groupby(grp_cols).size()
    keep_keys = counts[counts >= min_n].index
    sub = per_pixel.set_index(grp_cols).loc[keep_keys].reset_index()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    for (c, r, t), g in sub.groupby(grp_cols):
        yearly = g.groupby('yr')['C_predicted'].mean()
        ax.plot(yearly.index, yearly.values,
                color=REGION_COLORS.get(r, 'grey'),
                marker=TILLAGE_MARKERS.get(t, 'x'),
                alpha=0.7, lw=1.0,
                label=f"{str(c)[:18]}|{r[0]}/{t[:3]}")
    ax.set_xlabel('Year')
    ax.set_ylabel('Mean C_new on calibration sample')
    ax.set_title('Interannual variation per stratum — '
                 'invisible to the static table')
    ax.legend(fontsize=6, ncol=3, loc='upper right')
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"  saved {out}")


def plot_residual_vs_n_stratum(per_stratum: pd.DataFrame, out: str) -> None:
    """|bias| vs n at the stratum level, coloured by region."""
    df = per_stratum.dropna(subset=['bias', 'n']).copy()
    if df.empty:
        return
    df['abs_bias'] = df['bias'].abs()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for (reg, til), sub in df.groupby(['region', 'tillage']):
        ax.scatter(sub['n'], sub['abs_bias'], s=40, alpha=0.7,
                   c=REGION_COLORS.get(reg, 'grey'),
                   marker=TILLAGE_MARKERS.get(til, 'x'),
                   edgecolor='k', lw=0.4,
                   label=f'{reg}/{til}')
    median = df['abs_bias'].median()
    for _, r in df.iterrows():
        if r['abs_bias'] > median:
            ax.annotate(_stratum_label(r), (r['n'], r['abs_bias']),
                        fontsize=6, alpha=0.7,
                        xytext=(4, 3), textcoords='offset points')
    ax.set_xscale('log')
    ax.set_xlabel('Sample size n (parcel-pixels per stratum)')
    ax.set_ylabel('|bias|  =  |C_new_mean − C_ref|')
    ax.set_title('Stratum residuals vs sample size')
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"  saved {out}")


# ===========================================================================
# Driver
# ===========================================================================
def run(cfg: dict) -> None:
    out_dir  = Path(cfg['results_dir']);     out_dir.mkdir(exist_ok=True, parents=True)
    plot_dir = out_dir / 'plots';        plot_dir.mkdir(exist_ok=True)

    print("Loading per-pixel calibration output...")
    per_pixel = pd.read_csv(os.path.join(out_dir, cfg['per_pixel_path']))

    stratified = _detect_stratified(per_pixel, cfg.get('stratified_mode'))
    mode_label = 'STRATIFIED' if stratified else 'UNSTRATIFIED'
    print(f"  mode: {mode_label}")
    print(f"  {len(per_pixel):,} per-pixel rows  "
          f"({per_pixel['lnf_code'].nunique()} crops, "
          f"{per_pixel['yr'].nunique()} years"
          + (f", {per_pixel.groupby(['region', 'tillage']).ngroups} populated strata"
             if stratified else "")
          + ")")

    c_table   = load_c_table_total(cfg['c_factor_table_path'])
    lnf_bridge = load_lnf_bridge(cfg['lnf_classification_path'])
    per_pixel = attach_ref(per_pixel, c_table, lnf_bridge, stratified=stratified)

    miss = per_pixel['C_ref'].isna().sum()
    if miss:
        print(f"  WARNING: {miss} per-pixel rows have no C_ref "
              + ("(should not happen in stratified mode — check the "
                 "per-pixel file)" if stratified
                 else "(crop name not bridged) — they are dropped from "
                      "per-crop diagnostics."))

    # Optional: pull in fc_quality_flag and fc_total from the gapfilled parquet
    fc_path = cfg.get('gapfilled_fc_path')
    if fc_path and os.path.exists(fc_path):
        print(f"Loading FC stats from {fc_path}...")
        df_fc = pd.read_parquet(fc_path,
                                columns=['lnf_code', 'yr', 'poly_id',
                                         'fc_total', 'fc_quality_flag'])
        # Aggregate per (lnf_code, yr, poly_id) — same key the per-pixel
        # output uses. fc_quality_flag = max over the time series so any
        # bad obs in the year flags the pixel.
        fc_extra = (df_fc.groupby(['lnf_code', 'yr', 'poly_id'], as_index=False)
                         .agg(fc_total_mean=('fc_total', 'mean'),
                              fc_quality_flag=('fc_quality_flag', 'max')))
        per_pixel['yr'] = per_pixel['yr'].astype(int)
        fc_extra['yr'] = fc_extra['yr'].astype(int)
        per_pixel = per_pixel.merge(fc_extra,
                                    on=['lnf_code', 'yr', 'poly_id'],
                                    how='left')

    # ---------------------------------------------------------------------
    # Per-crop / per-stratum summaries
    # ---------------------------------------------------------------------
    per_stratum = None
    if stratified:
        print("\nPer-stratum diagnostics...")
        rows = []
        for (crop, reg, til), g in per_pixel.groupby(
                ['crop_de', 'region', 'tillage'], dropna=False):
            m = stratum_metrics(g)
            m['crop_de'] = crop
            m['lnf_code'] = g['lnf_code'].iloc[0]
            m['region'] = reg
            m['tillage'] = til
            m['C_ref'] = g['C_ref'].iloc[0] if g['C_ref'].notna().any() else np.nan
            rows.append(m)
        per_stratum = (pd.DataFrame(rows)
                         .sort_values('MAE', ascending=False)
                         .reset_index(drop=True))
        per_stratum.to_csv(out_dir / 'cal_per_stratum.csv', index=False)
        print(f"  wrote cal_per_stratum.csv  ({len(per_stratum)} strata)")

        # Rollup to crop level too (mean across that crop's strata) — keeps
        # backward-comparable per-crop file, but flagged as a rollup.
        rollup = (per_stratum.dropna(subset=['mean_ref', 'mean_new'])
                             .groupby(['crop_de', 'lnf_code'], as_index=False)
                             .agg(n=('n', 'sum'),
                                  mean_ref=('mean_ref', 'mean'),
                                  mean_new=('mean_new', 'mean'),
                                  bias=('bias', 'mean'),
                                  MAE=('MAE', 'mean'),
                                  RMSE=('RMSE', 'mean'),
                                  spread=('spread', 'mean'),
                                  n_strata=('n', 'size'),
                                  C_ref=('C_ref', 'mean')))
        per_crop = rollup.sort_values('MAE', ascending=False).reset_index(drop=True)
    else:
        print("\nPer-crop diagnostics...")
        rows = []
        for crop, g in per_pixel.groupby('crop_de', dropna=False):
            m = stratum_metrics(g)
            m['crop_de'] = crop
            m['lnf_code'] = g['lnf_code'].iloc[0]
            m['C_ref'] = g['C_ref'].iloc[0] if g['C_ref'].notna().any() else np.nan
            rows.append(m)
        per_crop = (pd.DataFrame(rows)
                      .sort_values('MAE', ascending=False)
                      .reset_index(drop=True))
    per_crop.to_csv(out_dir / 'cal_per_crop.csv', index=False)
    print(f"  wrote cal_per_crop.csv  ({len(per_crop)} crops"
          + (', rolled up from strata)' if stratified else ')'))

    # ---------------------------------------------------------------------
    # Per-year summary
    # ---------------------------------------------------------------------
    if per_pixel['yr'].nunique() > 1:
        rows = []
        if stratified:
            for (crop, reg, til, yr), g in per_pixel.groupby(
                    ['crop_de', 'region', 'tillage', 'yr'], dropna=False):
                m = stratum_metrics(g)
                m['crop_de'] = crop; m['region'] = reg
                m['tillage'] = til;  m['yr'] = yr
                rows.append(m)
        else:
            for (crop, yr), g in per_pixel.groupby(['crop_de', 'yr'], dropna=False):
                m = stratum_metrics(g)
                m['crop_de'] = crop; m['yr'] = yr
                rows.append(m)
        per_year = pd.DataFrame(rows)
        per_year.to_csv(out_dir / 'cal_per_year.csv', index=False)
        print(f"  wrote cal_per_year.csv  ({len(per_year)} rows)")

    # ---------------------------------------------------------------------
    # By C-magnitude (low vs high reference C — different relative metrics)
    # ---------------------------------------------------------------------
    bins = [-0.001, 0.01, 0.05, 0.10, 0.20, 1.0]
    labels = ['very_low (≤0.01)', 'low (0.01–0.05)', 'medium (0.05–0.10)',
              'high (0.10–0.20)', 'very_high (>0.20)']
    per_pixel['C_ref_class'] = pd.cut(per_pixel['C_ref'], bins=bins,
                                       labels=labels)
    rows = []
    for cls, g in per_pixel.groupby('C_ref_class', dropna=False, observed=True):
        m = stratum_metrics(g); m['C_ref_class'] = cls
        rows.append(m)
    by_mag = pd.DataFrame(rows)
    by_mag.to_csv(out_dir / 'cal_by_c_magnitude.csv', index=False)

    # ---------------------------------------------------------------------
    # By FC quality flag (only if FC parquet was available)
    # ---------------------------------------------------------------------
    by_q = None
    if 'fc_quality_flag' in per_pixel.columns:
        rows = []
        for q, g in per_pixel.groupby('fc_quality_flag', dropna=False):
            m = stratum_metrics(g); m['fc_quality_flag'] = q
            rows.append(m)
        by_q = (pd.DataFrame(rows)
                  .sort_values('fc_quality_flag', na_position='last')
                  .reset_index(drop=True))
        by_q.to_csv(out_dir / 'cal_by_quality.csv', index=False)
        print(f"  wrote cal_by_quality.csv")

    # ---------------------------------------------------------------------
    # Stratified-only rollups: by region and by tillage
    # ---------------------------------------------------------------------
    by_region = by_tillage = None
    if stratified:
        rows = []
        for reg, g in per_pixel.groupby('region', dropna=False):
            m = stratum_metrics(g); m['region'] = reg
            rows.append(m)
        by_region = pd.DataFrame(rows)
        by_region.to_csv(out_dir / 'cal_by_region.csv', index=False)
        print(f"  wrote cal_by_region.csv")

        rows = []
        for til, g in per_pixel.groupby('tillage', dropna=False):
            m = stratum_metrics(g); m['tillage'] = til
            rows.append(m)
        by_tillage = pd.DataFrame(rows)
        by_tillage.to_csv(out_dir / 'cal_by_tillage.csv', index=False)
        print(f"  wrote cal_by_tillage.csv")

    # ---------------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------------
    print("\nPlots...")
    if stratified:
        plot_scatter_per_stratum     (per_stratum, plot_dir / 'cal_scatter_per_stratum.png')
        plot_strengths_weaknesses_stratum(
            per_stratum, plot_dir / 'cal_strengths_weaknesses_per_stratum.png',
            min_n=cfg['min_n_per_crop'])
        plot_box_per_stratum         (per_pixel, plot_dir / 'cal_box_per_stratum.png',
                                       min_n=cfg['min_n_per_crop'])
        plot_interannual_stratum     (per_pixel, plot_dir / 'cal_interannual_per_stratum.png')
        plot_residual_vs_n_stratum   (per_stratum, plot_dir / 'cal_residual_vs_n_per_stratum.png')
        # Crop-rollup scatter is still nice to have side-by-side with
        # stratum-level scatter so you can see how much the stratification
        # actually changed the per-crop picture.
        plot_scatter_per_crop        (per_crop, plot_dir / 'cal_scatter_per_crop_rollup.png')
    else:
        plot_scatter_per_crop        (per_crop, plot_dir / 'cal_scatter_per_crop.png')
        plot_strengths_weaknesses    (per_crop, plot_dir / 'cal_strengths_weaknesses.png',
                                       min_n=cfg['min_n_per_crop'])
        plot_box_per_crop            (per_pixel, plot_dir / 'cal_box_per_crop.png',
                                       min_n=cfg['min_n_per_crop'])
        plot_interannual             (per_pixel, plot_dir / 'cal_interannual.png')
        plot_residual_vs_n           (per_crop, plot_dir / 'cal_residual_vs_n.png')

    plot_quality_bars(by_q, plot_dir / 'cal_quality_bars.png')

    # ---------------------------------------------------------------------
    # Headline summary
    # ---------------------------------------------------------------------
    valid = per_pixel.dropna(subset=['C_ref', 'C_predicted'])
    overall_bias = float((valid['C_predicted'] - valid['C_ref']).mean())
    overall_mae  = float((valid['C_predicted'] - valid['C_ref']).abs().mean())

    target_df = per_stratum if stratified else per_crop
    target_means = target_df.dropna(subset=['mean_ref', 'mean_new'])
    target_bias = float((target_means['mean_new'] - target_means['mean_ref']).mean())
    target_mae  = float((target_means['mean_new'] - target_means['mean_ref']).abs().mean())

    target_unit_label = 'Stratum' if stratified else 'Crop'

    summary_lines = [
        '=' * 70,
        f'In-sample calibration analysis  ({mode_label})',
        '=' * 70,
        f'Per-pixel rows                : {len(per_pixel):>10,}',
        f'Crops covered                 : {per_pixel["lnf_code"].nunique():>10}',
    ]
    if stratified:
        summary_lines.append(
            f'Strata covered                : {len(target_means):>10}')
    summary_lines.append(
        f'Years covered                 : {sorted(per_pixel["yr"].dropna().unique().tolist())}')
    summary_lines += [
        '',
        '--- Pixel-level (every row vs its '
        + ('stratum' if stratified else 'crop')
        + "'s C_ref) ---",
        f'  bias (C_new − C_ref)        : {overall_bias:+.4f}',
        f'  MAE                          : {overall_mae:.4f}',
        '',
        f"--- {target_unit_label}-level "
        + ('(the calibration target — stratified)'
           if stratified
           else '(the calibration target — Matthews-style)')
        + ' ---',
        f'  n {target_unit_label.lower()}s'.ljust(31) + f': {len(target_means)}',
        f'  bias of {target_unit_label.lower()} means'.ljust(31) + f': {target_bias:+.4f}',
        f'  MAE of {target_unit_label.lower()} means'.ljust(31) + f': {target_mae:.4f}',
        '',
        f'--- Top-5 worst-fitting {target_unit_label.lower()}s (by MAE, n ≥ 10) ---',
    ]
    worst_cols = (['crop_de', 'region', 'tillage', 'n', 'C_ref',
                   'mean_new', 'bias', 'MAE', 'spread']
                  if stratified
                  else ['crop_de', 'n', 'C_ref',
                        'mean_new', 'bias', 'MAE', 'spread'])
    worst = target_df[target_df['n'] >= 10].nlargest(5, 'MAE')[worst_cols]
    summary_lines.append(worst.to_string(index=False, float_format='%.4f'))
    summary_lines += ['', f'--- Top-5 best-fitting {target_unit_label.lower()}s '
                      '(by MAE, n ≥ 10) ---']
    best = target_df[target_df['n'] >= 10].nsmallest(5, 'MAE')[worst_cols]
    summary_lines.append(best.to_string(index=False, float_format='%.4f'))

    if stratified and by_region is not None:
        summary_lines += ['', '--- By region ---',
                          by_region[['region', 'n', 'mean_ref',
                                     'mean_new', 'bias', 'MAE']]
                            .to_string(index=False, float_format='%.4f')]
    if stratified and by_tillage is not None:
        summary_lines += ['', '--- By tillage ---',
                          by_tillage[['tillage', 'n', 'mean_ref',
                                      'mean_new', 'bias', 'MAE']]
                            .to_string(index=False, float_format='%.4f')]

    if by_q is not None:
        summary_lines += ['', '--- By FC quality flag ---',
                          by_q[['fc_quality_flag', 'n', 'mean_ref',
                                'mean_new', 'bias', 'MAE']]
                            .to_string(index=False, float_format='%.4f')]

    summary_lines += ['', '--- By reference-C magnitude ---',
                      by_mag[['C_ref_class', 'n', 'mean_ref',
                              'mean_new', 'bias', 'MAE']]
                        .to_string(index=False, float_format='%.4f')]

    summary_lines += ['', '=' * 70,
                      'How to read this',
                      '=' * 70, """
- Target-level MAE is the calibration loss.  This is what β was fit to;
  small values here only mean the optimiser worked, not that the product
  is good.  In unstratified mode the target is per-crop; in stratified
  mode the target is per (crop, region, tillage) cell.

- Pixel-level MAE > target-level MAE is normal and expected: the loss
  averages pixels first, so per-pixel residuals can be much larger
  while the per-target mean still hits the reference.

- Big spread + small bias on the strengths/weaknesses plot is the
  satellite product's selling point: it sees within-crop variation
  the table cannot encode while still landing on the right central
  value.

- High |bias| at low n in the residual_vs_n plot is sampling noise.
  High |bias| at high n is the genuine weakness — investigate the
  crop's FC dynamics or whether C_ref is the table-default 0.1/0.004.

- If MAE rises with fc_quality_flag, residuals are at least partly
  driven by S2 noise, so fresh-area performance will improve where
  cloud cover is lower.

- In stratified mode, compare the by-region and by-tillage rollups:
  a large bias in only one region or one tillage class usually points
  at a coverage issue (e.g. very few Berg/Direkt pixels) rather than
  a structural problem with β.  Cross-check against n in the
  per-stratum table.
"""]
    (out_dir / 'cal_summary.txt').write_text('\n'.join(summary_lines),
                                              encoding='utf-8')
    print(f"  wrote {out_dir / 'cal_summary.txt'}")
    print()
    print('\n'.join(summary_lines[:28]))


if __name__ == '__main__':
    run(CONFIG)