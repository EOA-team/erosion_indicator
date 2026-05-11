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

Inputs
------
- `calibration_results_per_pixel.csv`   from calibrate_cfactor.py
- `calibration_results.csv`             from calibrate_cfactor.py (per-crop diag)
- `C_Faktoren.csv`                      project file (for C_ref + C_old_total
                                                       per crop)
- `LNF_classification.xlsx`             project file (lnf_code ↔ Crop_DE)
- (optional) gapfilled FC parquet       from sample_FC.py — only used if you
                                         want by-quality-flag and by-FC stats
                                         (`fc_quality_flag`, fc_total mean…).

What it produces
----------------
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
    'per_pixel_path':           'calibration_results_per_pixel.csv',
    'per_crop_path':            'calibration_results.csv',
    'c_factor_table_path':      'C_Faktoren.csv',
    'lnf_classification_path':  '~/mnt/eo-nas1/data/landuse/documentation/'
                                'LNF_code_classification_20260217.xlsx',

    # Optional — gives you per-quality-flag diagnostics. Same parquet the
    # calibration consumed.
    'gapfilled_fc_path':        'samples_data_gpr.parquet',

    'out_dir':                  'calibration_analysis',
    'min_n_per_crop':           10,   # crops with fewer pixels are downweighted
                                       # in the strengths/weaknesses scatter
}


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
               lnf_bridge: pd.DataFrame) -> pd.DataFrame:
    """Attach crop_de and C_ref to per-pixel data."""
    df = df_pixel.merge(lnf_bridge[['lnf_code', 'crop_de', '_norm']],
                        on='lnf_code', how='left')
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


# ===========================================================================
# Plots
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
# Driver
# ===========================================================================
def run(cfg: dict) -> None:
    out_dir  = Path(cfg['out_dir']);     out_dir.mkdir(exist_ok=True, parents=True)
    plot_dir = out_dir / 'plots';        plot_dir.mkdir(exist_ok=True)

    print("Loading per-pixel calibration output...")
    per_pixel = pd.read_csv(cfg['per_pixel_path'])
    print(f"  {len(per_pixel):,} per-pixel rows  "
          f"({per_pixel['lnf_code'].nunique()} crops, "
          f"{per_pixel['yr'].nunique()} years)")

    c_table   = load_c_table_total(cfg['c_factor_table_path'])
    lnf_bridge = load_lnf_bridge(cfg['lnf_classification_path'])
    per_pixel = attach_ref(per_pixel, c_table, lnf_bridge)

    miss = per_pixel['C_ref'].isna().sum()
    if miss:
        print(f"  WARNING: {miss} per-pixel rows have no C_ref (crop name "
              f"not bridged) — they are dropped from per-crop diagnostics.")

    # Optional: pull in fc_quality_flag and fc_total from the gapfilled parquet
    fc_extra = None
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
    # Per-crop summary
    # ---------------------------------------------------------------------
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
    print(f"  wrote cal_per_crop.csv  ({len(per_crop)} crops)")

    # ---------------------------------------------------------------------
    # Per-year summary (per crop and overall)
    # ---------------------------------------------------------------------
    if per_pixel['yr'].nunique() > 1:
        rows = []
        for (crop, yr), g in per_pixel.groupby(['crop_de', 'yr'], dropna=False):
            m = stratum_metrics(g)
            m['crop_de'] = crop; m['yr'] = yr
            rows.append(m)
        per_year = pd.DataFrame(rows)
        per_year.to_csv(out_dir / 'cal_per_year.csv', index=False)
        print(f"  wrote cal_per_year.csv  ({len(per_year)} (crop, yr) combos)")

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
    # Plots
    # ---------------------------------------------------------------------
    print("\nPlots...")
    plot_scatter_per_crop      (per_crop, plot_dir / 'cal_scatter_per_crop.png')
    plot_strengths_weaknesses  (per_crop, plot_dir / 'cal_strengths_weaknesses.png',
                                min_n=cfg['min_n_per_crop'])
    plot_box_per_crop          (per_pixel, plot_dir / 'cal_box_per_crop.png',
                                min_n=cfg['min_n_per_crop'])
    plot_interannual           (per_pixel, plot_dir / 'cal_interannual.png')
    plot_residual_vs_n         (per_crop, plot_dir / 'cal_residual_vs_n.png')
    plot_quality_bars          (by_q,     plot_dir / 'cal_quality_bars.png')

    # ---------------------------------------------------------------------
    # Headline summary
    # ---------------------------------------------------------------------
    valid = per_pixel.dropna(subset=['C_ref', 'C_predicted'])
    overall_bias = float((valid['C_predicted'] - valid['C_ref']).mean())
    overall_mae  = float((valid['C_predicted'] - valid['C_ref']).abs().mean())

    crop_means = per_crop.dropna(subset=['mean_ref', 'mean_new'])
    crop_bias  = float((crop_means['mean_new'] - crop_means['mean_ref']).mean())
    crop_mae   = float((crop_means['mean_new'] - crop_means['mean_ref']).abs().mean())

    summary_lines = [
        '=' * 70,
        'In-sample calibration analysis',
        '=' * 70,
        f'Per-pixel rows                : {len(per_pixel):>10,}',
        f'Crops covered                 : {per_pixel["lnf_code"].nunique():>10}',
        f'Years covered                 : {sorted(per_pixel["yr"].dropna().unique().tolist())}',
        '',
        '--- Pixel-level (every row vs its crop\'s C_ref) ---',
        f'  bias (C_new − C_ref)        : {overall_bias:+.4f}',
        f'  MAE                          : {overall_mae:.4f}',
        '',
        '--- Crop-level (the calibration target — Matthews-style) ---',
        f'  n crops                      : {len(crop_means)}',
        f'  bias of crop means           : {crop_bias:+.4f}',
        f'  MAE of crop means            : {crop_mae:.4f}',
        '',
        '--- Top-5 worst-fitting crops (by MAE, n ≥ 10) ---',
    ]
    worst = (per_crop[per_crop['n'] >= 10]
                .nlargest(5, 'MAE')[['crop_de', 'n', 'C_ref',
                                      'mean_new', 'bias', 'MAE', 'spread']])
    summary_lines.append(worst.to_string(index=False, float_format='%.4f'))
    summary_lines += ['', '--- Top-5 best-fitting crops (by MAE, n ≥ 10) ---']
    best = (per_crop[per_crop['n'] >= 10]
              .nsmallest(5, 'MAE')[['crop_de', 'n', 'C_ref',
                                     'mean_new', 'bias', 'MAE', 'spread']])
    summary_lines.append(best.to_string(index=False, float_format='%.4f'))

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
- Crop-level MAE is the calibration loss.  This is what β was fit to;
  small values here only mean the optimiser worked, not that the product
  is good.

- Pixel-level MAE > crop-level MAE is normal and expected: the loss
  averages pixels first, so per-pixel residuals can be much larger
  while the per-crop mean still hits the target.

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
"""]
    (out_dir / 'cal_summary.txt').write_text('\n'.join(summary_lines),
                                              encoding='utf-8')
    print(f"  wrote {out_dir / 'cal_summary.txt'}")
    print()
    print('\n'.join(summary_lines[:25]))


if __name__ == '__main__':
    run(CONFIG)
