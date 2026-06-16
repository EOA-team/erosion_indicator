"""Compare empirical β vs ML stratum-level predictions, area-weighted by crop.

What this does
--------------
Inner-joins the two pipelines' per-pixel predictions on (lnf_code, yr, poly_id),
aggregates to the stratum level (lnf_code, region, tillage), and produces a
side-by-side comparison of:
  - mean predicted C per stratum (β vs ML vs reference)
  - per-stratum residuals
  - area-weighted stratum MAE (the headline number)
  - per-crop and per-year breakdowns

The "fair" comparison assumed here
----------------------------------
  - empirical: one β fit to all data. In-sample but cannot over-fit (1-2 DoF),
    so in-sample MAE ≈ generalisation MAE.
  - ML: the per-pixel CSV's `split` column determines what each row is. For the
    'stratified' / 'parcel' holdouts, `ml_predictions_source='test'` (default)
    uses only the held-out test parcels — no parcel/pixel leakage, an honest
    generalisation estimate that is slightly pessimistic vs the deployed model
    (which trains on all parcels). For a 'loyo' run, set source='loyo' to use
    the out-of-fold rows. Set source='final' to use the deployed in-sample MLP
    predictions as an optimistic ceiling sanity check.

Because the comparison inner-joins on (lnf_code, yr, poly_id), the common set
reduces to the ML rows selected by `source` (e.g. the held-out test pixels), so
β and ML are scored on exactly the same pixels and strata. The residual
asymmetry runs against the ML: if ML wins here, deployed ML is at least as good
as β; if they tie, ML is probably marginally better at deployment.

Area weighting
--------------
Stratum weight = A_c × n_pixels_in_stratum / n_pixels_in_crop, normalised over
the COMMON stratum set (after the inner-join). This matches the empirical
calibration's stratum weighting and the ML pipeline's sample-weighting scheme
(per-pixel weight A_c / n_pixels_in_crop, which sums to the same stratum total).

Inputs (paths in CONFIG)
------------------------
  - β  per-pixel CSV       calibration_results_stratified_per_pixel.csv
  - β  per-stratum CSV     calibration_results_stratified.csv          (for area_ha)
  - ML per-pixel CSV       nn_predictions_per_pixel.csv
  - LNF classification XLSX (optional, for crop_de names)

Outputs (CONFIG['out_dir'])
---------------------------
  - compare_summary.txt              headline area-weighted stratum MAE for both
  - compare_per_stratum.csv          full stratum-level join with both predictions
  - compare_per_crop.csv             rolled up to crop (mean over strata)
  - compare_per_year.csv             year-wise area-weighted MAE (if multi-year)
  - plots/compare_scatter.png        side-by-side scatter, β and ML vs C_ref
  - plots/compare_paired.png         β residual vs ML residual per stratum
  - plots/compare_per_crop_bars.png  top crops by area, paired |bias| bars
  - plots/compare_per_year.png       year-wise MAE (if multi-year)

Run:
    python compare_beta_vs_ml.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


CONFIG = {
    # Inputs
    'beta_per_pixel_path':      'calibration_analysis_twobeta_noley/calibration_results_stratified_per_pixel.csv',
    'beta_per_stratum_path':    'calibration_analysis_twobeta_noley/calibration_results_stratified.csv',
    'ml_per_pixel_path':        'calibration_analysis_mlp_strat_noley/nn_predictions_per_pixel.csv',
    'lnf_classification_path':  '~/mnt/eo-nas1/data/landuse/documentation/'
                                'LNF_code_classification_20260217.xlsx',

    # Which ML predictions to use:
    #   'auto'   -> 'loyo' if split looks like years, else 'test' (holdout modes)
    #   'test'   -> held-out test rows only (stratified/parcel holdout — default,
    #               recommended; auto-drops 'excluded' config crops)
    #   'all'    -> all rows (train+test pooled — optimistic, not honest)
    #   'loyo'   -> all rows (LOYO mode; every row is OOF)
    #   'final'  -> use C_pred_final column (deployed model in-sample —
    #               optimistic, sanity-check only; written in LOYO mode)
    'ml_predictions_source':    'test',

    # Output
    'out_dir':                  'compare',

    # Reporting knobs
    'top_n_crops':              20,
    'min_n_pixels_per_stratum': 1,    # drop very small strata; 1 = no filter
}


# ============================================================================
# Loading
# ============================================================================
def load_beta_per_pixel(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = ['lnf_code', 'yr', 'poly_id', 'region', 'tillage',
              'C_predicted', 'C_ref']
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"β per-pixel CSV missing columns: {miss}")
    return df.rename(columns={'C_predicted': 'C_pred_beta'})


def _read_split_strategy(per_pixel_path: str) -> str:
    """Read the precise split strategy from the sibling nn_metrics.json.

    The per-pixel `split` column is {train,test,excluded} for BOTH the
    'parcel' and 'stratified' holdouts, so the column alone can't tell them
    apart. nn_metrics.json carries the exact 'split_strategy' string written
    by the trainer; we map its leading word to a short mode tag for labels.
    Falls back to a generic 'holdout' if the JSON is missing/unreadable.
    """
    metrics_path = os.path.join(os.path.dirname(per_pixel_path), 'nn_metrics.json')
    try:
        import json
        with open(metrics_path) as f:
            strat = str(json.load(f).get('split_strategy', '')).lower()
    except Exception:
        return 'holdout'
    if strat.startswith('stratified'):
        return 'stratified'
    if strat.startswith('parcel'):
        return 'parcel'
    if strat.startswith('loyo'):
        return 'loyo'
    return 'holdout'


def load_ml_per_pixel(path: str, source: str
                       ) -> tuple[pd.DataFrame, str, str]:
    """Return (df, split_mode, effective_source)."""
    df = pd.read_csv(path)
    pred_col = 'C_pred' if 'C_pred' in df.columns else (
        'C_predicted' if 'C_predicted' in df.columns else None)
    if pred_col is None:
        raise ValueError(f"ML CSV {path} has no C_pred / C_predicted column")
    if 'split' not in df.columns:
        df['split'] = 'all'

    # 'excluded' (config-excluded crops / missing C_ref) is an inference-only
    # tag present in every holdout mode; ignore it when classifying the split.
    vals = set(df['split'].dropna().astype(str).unique())
    non_excluded = vals - {'excluded'}
    if non_excluded and all(v.isdigit() and len(v) == 4 for v in non_excluded):
        split_mode = 'loyo'
        df['split'] = df['split'].astype(str)
    elif non_excluded <= {'train', 'test', 'all'}:
        # parcel vs stratified holdout — disambiguate via nn_metrics.json.
        split_mode = _read_split_strategy(path)
    else:
        split_mode = 'unknown'

    if source == 'auto':
        source = 'loyo' if split_mode == 'loyo' else 'test'

    if source == 'final':
        if 'C_pred_final' not in df.columns:
            raise ValueError("source='final' requires C_pred_final column "
                             "(only written in LOYO mode)")
        df_use = df.rename(columns={'C_pred_final': 'C_pred_ml'}).copy()
    elif source == 'test':
        df_use = df[df['split'] == 'test'].rename(
            columns={pred_col: 'C_pred_ml'}).copy()
        if len(df_use) == 0:
            raise ValueError(f"source='test' but no rows with split='test' "
                             f"in {path}")
    elif source in ('all', 'loyo'):
        df_use = df.rename(columns={pred_col: 'C_pred_ml'}).copy()
    else:
        raise ValueError(f"Unknown ml_predictions_source={source!r}")

    return df_use, split_mode, source


def ml_source_phrase(source: str, split_mode: str) -> str:
    """Short phrase describing the ML predictions, for plot titles/legends."""
    if source == 'final':
        return 'final, in-sample'
    if split_mode == 'loyo':
        return 'LOYO OOF'
    if split_mode in ('stratified', 'parcel', 'holdout'):
        return 'held-out test'
    return source


def ml_eval_description(source: str, split_mode: str) -> str:
    """One-line description of what the ML predictions represent."""
    if source == 'final':
        return 'final model, in-sample (optimistic ceiling)'
    if split_mode == 'loyo':
        return 'leave-one-year-out, out-of-fold (honest generalisation)'
    if split_mode == 'stratified':
        return 'held-out test parcels, per-stratum holdout (no leakage)'
    if split_mode in ('parcel', 'holdout'):
        return 'held-out test parcels (no leakage)'
    return source


def load_area_ha(beta_per_stratum_path: str) -> pd.DataFrame:
    """Pull area_ha per lnf_code from the empirical per-stratum CSV."""
    if not os.path.exists(beta_per_stratum_path):
        return pd.DataFrame(columns=['lnf_code', 'area_ha'])
    df = pd.read_csv(beta_per_stratum_path)
    if 'area_ha' not in df.columns:
        return pd.DataFrame(columns=['lnf_code', 'area_ha'])
    return (df[['lnf_code', 'area_ha']]
              .drop_duplicates(subset='lnf_code')
              .reset_index(drop=True))


def load_crop_names(lnf_classification_path: str) -> pd.DataFrame:
    path = os.path.expanduser(lnf_classification_path)
    if not os.path.exists(path):
        return pd.DataFrame(columns=['lnf_code', 'crop_de'])
    df = pd.read_excel(path, sheet_name='label_sheet')
    return (df[['LNF_code', 'Crop_DE']]
              .rename(columns={'LNF_code': 'lnf_code', 'Crop_DE': 'crop_de'})
              .dropna()
              .drop_duplicates(subset='lnf_code')
              .reset_index(drop=True))


# ============================================================================
# Merging
# ============================================================================
def inner_join_pixels(df_beta: pd.DataFrame,
                       df_ml: pd.DataFrame) -> pd.DataFrame:
    """Inner-join at pixel level (lnf_code, yr, poly_id)."""
    join_cols = ['lnf_code', 'yr', 'poly_id']
    b_cols = join_cols + ['region', 'tillage', 'C_pred_beta', 'C_ref']
    m_cols = join_cols + ['region', 'tillage', 'C_pred_ml', 'C_ref']
    if 'split' in df_ml.columns:
        m_cols.append('split')
    df_b = df_beta[b_cols].copy()
    df_m = df_ml[m_cols].rename(columns={'region':  'region_ml',
                                          'tillage': 'tillage_ml',
                                          'C_ref':   'C_ref_ml'}).copy()
    merged = df_b.merge(df_m, on=join_cols, how='inner')

    bad_reg = (merged['region']  != merged['region_ml']).sum()
    bad_til = (merged['tillage'] != merged['tillage_ml']).sum()
    bad_ref = (~np.isclose(merged['C_ref'].values, merged['C_ref_ml'].values,
                            equal_nan=True)).sum()
    if bad_reg or bad_til or bad_ref:
        print(f"  WARNING: pixel-level disagreement (region={bad_reg}, "
              f"tillage={bad_til}, C_ref={bad_ref}) — keeping β's values. "
              "Pipelines may have been configured differently "
              "(e.g. different tillage_assignment seed or area_years).")
    return merged.drop(columns=['region_ml', 'tillage_ml', 'C_ref_ml'])


def aggregate_to_stratum(df_pix: pd.DataFrame,
                          area_ha: pd.DataFrame) -> pd.DataFrame:
    agg = (df_pix.groupby(['lnf_code', 'region', 'tillage'], as_index=False)
                  .agg(mean_pred_beta=('C_pred_beta', 'mean'),
                       mean_pred_ml=('C_pred_ml', 'mean'),
                       C_ref=('C_ref', 'first'),
                       n_pixels=('C_pred_beta', 'size')))
    agg['bias_beta']     = agg['mean_pred_beta'] - agg['C_ref']
    agg['bias_ml']       = agg['mean_pred_ml']   - agg['C_ref']
    agg['abs_bias_beta'] = agg['bias_beta'].abs()
    agg['abs_bias_ml']   = agg['bias_ml'].abs()
    if len(area_ha):
        agg = agg.merge(area_ha, on='lnf_code', how='left')
    else:
        agg['area_ha'] = np.nan
    return agg


def compute_weights(agg: pd.DataFrame) -> pd.DataFrame:
    """Stratum weight = A_c × n_pixels_in_stratum / n_pixels_in_crop, normalised."""
    out = agg.copy()
    if 'area_ha' in out.columns and out['area_ha'].notna().any():
        area = out['area_ha'].fillna(0.0).clip(lower=0.0).values
        n_per_crop = out.groupby('lnf_code')['n_pixels'].transform('sum').values
        w_raw = area * out['n_pixels'].values / np.maximum(n_per_crop, 1)
        total = float(w_raw.sum())
        out['weight'] = w_raw / total if total > 0 else 1.0 / len(out)
        out['weighting'] = 'area_x_pixels'
        n_zero = int((out['weight'] == 0).sum())
        if n_zero:
            print(f"  ({n_zero}/{len(out)} strata have area_ha=NaN/0 and "
                  "contribute zero weight)")
    else:
        out['weight'] = 1.0 / len(out)
        out['weighting'] = 'equal'
        print("  WARNING: area_ha unavailable — falling back to equal-weighted "
              "stratum MAE.")
    return out


# ============================================================================
# Reporting helpers
# ============================================================================
def winner_col(df: pd.DataFrame,
                a_col: str = 'abs_bias_beta',
                b_col: str = 'abs_bias_ml',
                tol: float = 1e-4) -> pd.Series:
    diff = df[a_col] - df[b_col]   # positive ⇒ ML's |bias| smaller ⇒ ML wins
    out = np.where(diff >  tol, 'ml',
          np.where(diff < -tol, 'beta', 'tie'))
    return pd.Series(out, index=df.index)


def rollup_to_crop(strata: pd.DataFrame) -> pd.DataFrame:
    """Mean over a crop's populated strata (equal across strata within crop)."""
    return (strata.groupby('lnf_code', as_index=False)
                  .agg(n_strata=('lnf_code', 'size'),
                       n_pixels=('n_pixels', 'sum'),
                       area_ha=('area_ha', 'first'),
                       C_ref_mean=('C_ref', 'mean'),
                       mean_pred_beta=('mean_pred_beta', 'mean'),
                       mean_pred_ml=('mean_pred_ml', 'mean'),
                       abs_bias_beta=('abs_bias_beta', 'mean'),
                       abs_bias_ml=('abs_bias_ml', 'mean')))


def headline_metrics(strata: pd.DataFrame) -> dict:
    w = strata['weight'].values
    return {
        'n_strata':              int(len(strata)),
        'n_crops':               int(strata['lnf_code'].nunique()),
        'n_pixels':              int(strata['n_pixels'].sum()),
        'weighting':             strata['weighting'].iloc[0],
        'beta_stratum_bias':     float((strata['bias_beta'].values * w).sum()),
        'ml_stratum_bias':       float((strata['bias_ml'].values   * w).sum()),
        'beta_stratum_mae':      float((strata['abs_bias_beta'].values * w).sum()),
        'ml_stratum_mae':        float((strata['abs_bias_ml'].values   * w).sum()),
    }


def per_year_metrics(df_pix: pd.DataFrame,
                      area_ha: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for yr, g in df_pix.groupby('yr'):
        agg = aggregate_to_stratum(g, area_ha)
        if agg.empty:
            continue
        agg = compute_weights(agg)
        h = headline_metrics(agg)
        rows.append({'yr': int(yr),
                     'n_strata': h['n_strata'],
                     'n_pixels': h['n_pixels'],
                     'beta_mae':  h['beta_stratum_mae'],
                     'ml_mae':    h['ml_stratum_mae'],
                     'beta_bias': h['beta_stratum_bias'],
                     'ml_bias':   h['ml_stratum_bias']})
    return pd.DataFrame(rows).sort_values('yr').reset_index(drop=True)


# ============================================================================
# Plots
# ============================================================================
def _short_name(row, max_len: int = 22) -> str:
    """Crop label for plot annotations."""
    name = row.get('crop_de') if isinstance(row.get('crop_de'), str) else None
    base = name if name else str(row['lnf_code'])
    if len(base) > max_len:
        base = base[:max_len - 1] + '…'
    return base


def plot_scatter_side_by_side(strata: pd.DataFrame, out: str, h: dict,
                               ml_label: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharex=True, sharey=True)
    mx = max(strata['C_ref'].max(),
             strata['mean_pred_beta'].max(),
             strata['mean_pred_ml'].max()) * 1.05
    s = 30 + 220 * (strata['weight'].values
                    / max(strata['weight'].max(), 1e-12))

    panels = [
        (axes[0], 'mean_pred_beta', 'abs_bias_beta',
         'Empirical (in-sample)',
         h['beta_stratum_mae'], h['beta_stratum_bias'], '#1f77b4'),
        (axes[1], 'mean_pred_ml',   'abs_bias_ml',
         f'ML ({ml_label})',
         h['ml_stratum_mae'], h['ml_stratum_bias'], '#d62728'),
    ]
    for ax, pred_col, abs_col, title, mae, bias, color in panels:
        ax.scatter(strata['C_ref'], strata[pred_col],
                   s=s, alpha=0.6, c=color,
                   edgecolor='k', linewidth=0.3)
        ax.plot([0, mx], [0, mx], 'k--', lw=1, alpha=0.5, label='1:1')

        # Label the 5 worst-fit strata on this panel (largest |residual|).
        worst = strata.nlargest(5, abs_col)
        for _, r in worst.iterrows():
            ax.annotate(_short_name(r, 18),
                        (r['C_ref'], r[pred_col]),
                        xytext=(4, 4), textcoords='offset points',
                        fontsize=7, alpha=0.85,
                        bbox=dict(boxstyle='round,pad=0.15',
                                  facecolor='white', edgecolor='none',
                                  alpha=0.7))

        ax.set_xlim(0, mx); ax.set_ylim(0, mx)
        ax.set_xlabel('C_ref (tabulated, per stratum)')
        ax.set_ylabel('mean C_pred (per stratum)')
        ax.set_title(f'{title}\n'
                     f'Area-weighted MAE = {mae:.4f}   '
                     f'signed bias = {bias:+.4f}')

        sign_text = ('mean over-prediction' if bias > 1e-5
                     else 'mean under-prediction' if bias < -1e-5
                     else 'unbiased on average')
        ax.text(0.98, 0.02,
                f'Above 1:1  → over-predict\n'
                f'Below 1:1  → under-predict\n'
                f'Net: {sign_text}\n'
                f'Labelled = 5 worst-fit strata',
                transform=ax.transAxes, fontsize=7.5,
                ha='right', va='bottom', alpha=0.75,
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='#f5f5f5', edgecolor='#cccccc'))
        ax.legend(loc='upper left', fontsize=9)

    fig.suptitle(f'β vs ML — stratum-mean predicted vs reference  '
                 f"({h['n_strata']} strata, {h['n_crops']} crops, "
                 f"{h['n_pixels']:,} pixels)   "
                 f"·  marker size ∝ stratum weight",
                 fontsize=10.5)
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"  saved {out}")


def plot_paired_residuals(strata: pd.DataFrame, out: str,
                           top_n_labels: int = 20) -> None:
    """Per-stratum β-residual vs ML-residual.

    Encodings
    ---------
      colour : region          (one colour per unique region value)
      marker : tillage         (one shape per unique tillage value)
      size   : stratum weight  (area share)
      labels : top-N strata by weight (the operationally important ones)

    Quadrant meaning (where x = β residual, y = ML residual):
      upper-right : both over-predict
      lower-left  : both under-predict
      upper-left  : β under, ML over
      lower-right : β over,  ML under
      near y = x  : same residual sign & magnitude → data issue, not model issue
    """
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_axes([0.09, 0.09, 0.66, 0.82])

    sizes = 30 + 260 * (strata['weight'].values
                        / max(strata['weight'].max(), 1e-12))

    regions  = sorted(strata['region'].dropna().unique().tolist())
    tillages = sorted(strata['tillage'].dropna().unique().tolist())
    region_palette = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd',
                      '#ff7f0e', '#17becf']
    marker_cycle   = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
    region_colour = {r: region_palette[i % len(region_palette)]
                     for i, r in enumerate(regions)}
    tillage_marker = {t: marker_cycle[i % len(marker_cycle)]
                      for i, t in enumerate(tillages)}

    for (reg, til), g in strata.groupby(['region', 'tillage']):
        idx = g.index
        ax.scatter(g['bias_beta'], g['bias_ml'],
                   s=sizes[strata.index.get_indexer(idx)],
                   c=region_colour.get(reg, '#888888'),
                   marker=tillage_marker.get(til, 'o'),
                   alpha=0.65, edgecolor='k', linewidth=0.3)

    lim = max(strata[['bias_beta', 'bias_ml']].abs().max().max() * 1.15, 0.01)
    ax.axhline(0, color='k', lw=0.6, alpha=0.4)
    ax.axvline(0, color='k', lw=0.6, alpha=0.4)
    ax.plot([-lim, lim], [-lim, lim], 'k--', lw=1, alpha=0.4,
            label='y = x   (both models agree)')
    # ML-wins region: |y| < |x|, i.e. inside the |y|=|x| wedge around the x-axis.
    ax.plot([-lim, lim], [lim, -lim], color='grey', ls=':', lw=0.8, alpha=0.5,
            label='y = −x')
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)

    # Quadrant labels (corners).
    corner_kw = dict(fontsize=8, alpha=0.55, style='italic')
    ax.text( 0.97 * lim,  0.95 * lim, 'both over-predict',
            ha='right', va='top',    **corner_kw)
    ax.text(-0.97 * lim, -0.95 * lim, 'both under-predict',
            ha='left',  va='bottom', **corner_kw)
    ax.text(-0.97 * lim,  0.95 * lim, 'β under, ML over',
            ha='left',  va='top',    **corner_kw)
    ax.text( 0.97 * lim, -0.95 * lim, 'β over, ML under',
            ha='right', va='bottom', **corner_kw)
    # Wedge labels: shade nothing, just annotate which side is whose win.
    ax.text(0.97 * lim, 0.02 * lim,
            'ML wins  (|ML res| < |β res|)',
            ha='right', va='bottom', fontsize=8.5, color='#444444',
            fontweight='bold')
    ax.text(0.02 * lim, 0.97 * lim,
            'β wins  (|β res| < |ML res|)',
            ha='left', va='top',    fontsize=8.5, color='#444444',
            fontweight='bold', rotation=90)

    # Label the top-N strata by weight (the ones that move the headline MAE).
    top = strata.nlargest(top_n_labels, 'weight')
    for _, r in top.iterrows():
        lbl = f"{_short_name(r, 16)} · {r['region'][:1]}/{str(r['tillage'])[:3]}"
        ax.annotate(lbl, (r['bias_beta'], r['bias_ml']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=7, alpha=0.9,
                    bbox=dict(boxstyle='round,pad=0.15',
                              facecolor='white', edgecolor='none',
                              alpha=0.75))

    ax.set_xlabel('β residual  (mean C_pred − C_ref, per stratum)')
    ax.set_ylabel('ML residual (mean C_pred − C_ref, per stratum)')
    win_counts = strata['winner'].value_counts().to_dict()
    n_total = len(strata)
    ax.set_title('Per-stratum residual comparison  ·  '
                 f'β wins {win_counts.get("beta", 0)} '
                 f'/ ML wins {win_counts.get("ml", 0)} '
                 f'/ tie {win_counts.get("tie", 0)}   '
                 f'(of {n_total} strata)\n'
                 f'colour = region · marker = tillage · '
                 f'size ∝ weight · labelled = top {top_n_labels} by weight',
                 fontsize=10)

    # Legend axis on the right.
    leg_ax = fig.add_axes([0.78, 0.09, 0.20, 0.82])
    leg_ax.axis('off')

    # Region legend.
    leg_ax.text(0, 1.0, 'Region (colour)', fontsize=10, fontweight='bold',
                transform=leg_ax.transAxes, va='top')
    y = 0.95
    for r in regions:
        leg_ax.scatter([0.05], [y], s=80, c=region_colour[r],
                       marker='o', edgecolor='k', linewidth=0.3,
                       transform=leg_ax.transAxes, clip_on=False)
        leg_ax.text(0.18, y, str(r), fontsize=9,
                    transform=leg_ax.transAxes, va='center')
        y -= 0.045

    # Tillage legend.
    y -= 0.04
    leg_ax.text(0, y, 'Tillage (marker)', fontsize=10, fontweight='bold',
                transform=leg_ax.transAxes, va='top')
    y -= 0.05
    for t in tillages:
        leg_ax.scatter([0.05], [y], s=80, c='#666666',
                       marker=tillage_marker[t], edgecolor='k', linewidth=0.3,
                       transform=leg_ax.transAxes, clip_on=False)
        leg_ax.text(0.18, y, str(t), fontsize=9,
                    transform=leg_ax.transAxes, va='center')
        y -= 0.045

    # Reference lines + how-to-read.
    y -= 0.04
    leg_ax.text(0, y, 'Reading the plot', fontsize=10, fontweight='bold',
                transform=leg_ax.transAxes, va='top')
    y -= 0.055
    leg_ax.text(0, y,
                'Near origin: both fine.\n'
                'Near y = x: same error,\n'
                '  data issue not model.\n'
                'Near x-axis: ML beats β.\n'
                'Near y-axis: β beats ML.\n'
                'Off-diagonal: models\n'
                '  disagree in direction.',
                fontsize=8, transform=leg_ax.transAxes, va='top',
                linespacing=1.3)

    plt.savefig(out, dpi=150); plt.close()
    print(f"  saved {out}")


def plot_per_crop_bars(crops: pd.DataFrame, out: str, top_n: int) -> None:
    d = crops.dropna(subset=['area_ha']).nlargest(top_n, 'area_ha').copy()
    if d.empty:
        return
    d['delta'] = d['abs_bias_beta'] - d['abs_bias_ml']   # >0 → ML wins
    d['winner_crop'] = np.where(d['delta'] >  1e-4, 'ML',
                        np.where(d['delta'] < -1e-4, 'β', 'tie'))

    def _label(r):
        name = r.get('crop_de') if isinstance(r.get('crop_de'), str) else str(r['lnf_code'])
        marker = {'ML': '▶ ML', 'β': '◀ β', 'tie': '= tie'}[r['winner_crop']]
        return f"{marker:6s}  {name[:30]} ({r['area_ha']/1000:.1f} kha)"
    d['label'] = d.apply(_label, axis=1)

    y = np.arange(len(d))
    width = 0.38
    fig, (ax, ax_d) = plt.subplots(
        1, 2, figsize=(13, 0.45 * len(d) + 1.8),
        gridspec_kw={'width_ratios': [2.4, 1.0], 'wspace': 0.04},
        sharey=True,
    )

    # Left panel: paired bars.
    ax.barh(y - width/2, d['abs_bias_beta'], width,
            color='#1f77b4', label='β  |bias|', alpha=0.85)
    ax.barh(y + width/2, d['abs_bias_ml'], width,
            color='#d62728', label='ML |bias|', alpha=0.85)
    ax.set_yticks(y); ax.set_yticklabels(d['label'], fontsize=8,
                                         family='monospace')
    ax.invert_yaxis()
    ax.set_xlabel('Mean |stratum residual| within crop  (smaller = better)')
    ax.set_title(f'Top {top_n} crops by Swiss arable area — |bias| comparison')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='x', alpha=0.25, linestyle=':')

    # Right panel: Δ = β − ML  (positive → ML wins).
    colors = ['#d62728' if v > 1e-4 else '#1f77b4' if v < -1e-4 else '#888888'
              for v in d['delta']]
    ax_d.barh(y, d['delta'], color=colors, alpha=0.85, edgecolor='k',
              linewidth=0.3)
    ax_d.axvline(0, color='k', lw=0.8)
    ax_d.set_xlabel('Δ = |β| − |ML|\n← β better   |   ML better →', fontsize=9)
    ax_d.set_title('Per-crop winner', fontsize=10)
    ax_d.grid(axis='x', alpha=0.25, linestyle=':')
    # Symmetric x-limits so the zero line is centred.
    xmax = max(d['delta'].abs().max() * 1.15, 1e-3)
    ax_d.set_xlim(-xmax, xmax)

    # Footer summary.
    n_ml = (d['winner_crop'] == 'ML').sum()
    n_b  = (d['winner_crop'] == 'β').sum()
    n_t  = (d['winner_crop'] == 'tie').sum()
    fig.suptitle(f'Among the top {top_n} crops:  ML wins {n_ml}, '
                 f'β wins {n_b}, tie {n_t}',
                 y=0.995, fontsize=10, alpha=0.7)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f"  saved {out}")


def plot_per_year(per_yr: pd.DataFrame, out: str, ml_label: str = 'ML') -> None:
    if len(per_yr) < 2:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(per_yr['yr'], per_yr['beta_mae'], 'o-',
            color='#1f77b4', label='β (in-sample)', lw=2)
    ax.plot(per_yr['yr'], per_yr['ml_mae'], 's-',
            color='#d62728', label=ml_label, lw=2)
    ax.set_xlabel('Year'); ax.set_ylabel('Area-weighted stratum MAE')
    ax.set_title('Per-year stratum MAE — β vs ML')
    ax.legend(); plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"  saved {out}")


# ============================================================================
# Summary
# ============================================================================
def write_summary(h: dict, strata: pd.DataFrame, crops: pd.DataFrame,
                  per_yr: pd.DataFrame, effective_source: str,
                  split_mode: str, path: str) -> None:
    delta = h['beta_stratum_mae'] - h['ml_stratum_mae']
    winner = ('ML wins'  if delta >  1e-4 else
              'β wins'   if delta < -1e-4 else
              'tie')
    win_counts = strata['winner'].value_counts().to_dict()

    lines = [
        '=' * 72,
        'β vs ML comparison — stratum-level, area-weighted by crop',
        '=' * 72,
        f'ML source            : {effective_source}  (split_mode={split_mode})',
        f'ML evaluation        : {ml_eval_description(effective_source, split_mode)}',
        f'n_pixels (common)    : {h["n_pixels"]:>10,}',
        f'n_strata (common)    : {h["n_strata"]:>10}',
        f'n_crops              : {h["n_crops"]:>10}',
        f'Weighting            : {h["weighting"]}',
        '',
        '--- Headline area-weighted stratum MAE ---',
        f'  β    : {h["beta_stratum_mae"]:.5f}   '
        f'(signed bias = {h["beta_stratum_bias"]:+.5f})',
        f'  ML   : {h["ml_stratum_mae"]:.5f}   '
        f'(signed bias = {h["ml_stratum_bias"]:+.5f})',
        f'  Δ    : {delta:+.5f}   ({winner})',
        '',
        '--- Stratum-level winner counts (smaller |residual|) ---',
        f'  β wins  : {win_counts.get("beta", 0)}',
        f'  ML wins : {win_counts.get("ml",   0)}',
        f'  tie     : {win_counts.get("tie",  0)}',
    ]

    if len(per_yr):
        lines += [
            '',
            '--- Per-year area-weighted stratum MAE ---',
            per_yr[['yr', 'n_strata', 'n_pixels',
                    'beta_mae', 'ml_mae',
                    'beta_bias', 'ml_bias']]
              .to_string(index=False, float_format='%.4f'),
        ]

    if 'crop_de' in crops.columns:
        bigs = crops.dropna(subset=['area_ha']).nlargest(10, 'area_ha').copy()
        if len(bigs):
            bigs['delta_abs'] = bigs['abs_bias_beta'] - bigs['abs_bias_ml']
            cols = ['lnf_code', 'crop_de', 'area_ha', 'n_strata',
                    'abs_bias_beta', 'abs_bias_ml', 'delta_abs']
            lines += [
                '',
                '--- Top 10 crops by area: |bias| comparison ---',
                '    delta_abs > 0  =>  ML beats β on that crop',
                bigs[cols].to_string(index=False, float_format='%.4f'),
            ]

    if split_mode == 'loyo':
        eval_note = (
"- In LOYO mode every ML prediction is out-of-fold: the model never saw its\n"
"  year. β predictions are in-sample but β has only 1-2 DoF, so its in-sample\n"
"  MAE is approximately its generalisation MAE. The residual asymmetry: LOYO\n"
"  ML slightly understates the deployed all-years model. Direction: against ML.")
    elif effective_source == 'final':
        eval_note = (
"- ML predictions here are the FINAL model in-sample (source='final'): trained\n"
"  on all trainable rows, so this flatters ML (an MLP can over-fit; β with 1-2\n"
"  DoF cannot). Treat it as an optimistic ceiling, not the headline number.")
    else:
        eval_note = (
"- ML predictions are on HELD-OUT test parcels (per-stratum holdout, no parcel\n"
"  or pixel leakage). β predictions are in-sample but β has only 1-2 DoF, so its\n"
"  in-sample MAE ≈ its generalisation MAE. The inner-join scores BOTH models on\n"
"  the same held-out pixels per stratum. Mild asymmetry: the deployed ML is\n"
"  trained on all parcels, so test-set ML slightly understates it (direction:\n"
"  against ML). Single-parcel strata are train-only and absent from this set.")

    lines += [
        '',
        '=' * 72,
        'How to read this',
        '=' * 72,
        f"""
- The headline area-weighted stratum MAE is the operational number: for each
  stratum (crop x region x tillage), how far is the mean predicted C-factor
  from the tabulated C_ref, averaged with weights proportional to
  A_c × n_pixels_in_stratum / n_pixels_in_crop. Crops contribute proportional
  to their Swiss arable area.

{eval_note}

- If ML wins this comparison, deployed ML is at least as good as β. If they
  tie, ML is probably marginally better at deployment. If β wins, ML might
  still tie at deployment but is unlikely to clearly beat β.

- Look at compare_paired.png: red points (ML wins) clustered on a specific
  region/tillage or crop type tell you where the nonlinear FC->C mapping adds
  value over the exp(-β·FC) form. Diagonal cluster (both residuals correlated)
  means the two methods make the same kinds of mistakes — likely a data issue
  (FC bias, EI mismatch) rather than a model-choice issue.

- compare_per_year.png shows temporal stability. If one year's β and ML MAE
  both jump together, that year's FC distribution is unusual (drought, wet
  year). If only ML jumps, the MLP didn't see enough of that regime.
"""
    ]
    Path(path).write_text('\n'.join(lines), encoding='utf-8')
    print(f"  wrote {path}")


# ============================================================================
# Runner
# ============================================================================
def run(cfg: dict) -> None:
    out_dir  = Path(cfg['out_dir']); out_dir.mkdir(exist_ok=True, parents=True)
    plot_dir = out_dir / 'plots';    plot_dir.mkdir(exist_ok=True)

    print("Loading β per-pixel predictions ...")
    df_b = load_beta_per_pixel(cfg['beta_per_pixel_path'])
    print(f"  {len(df_b):,} rows, {df_b['lnf_code'].nunique()} crops")

    print("Loading ML per-pixel predictions ...")
    df_m, split_mode, effective_source = load_ml_per_pixel(
        cfg['ml_per_pixel_path'], cfg.get('ml_predictions_source', 'auto'))
    print(f"  {len(df_m):,} rows (source={effective_source!r}, "
          f"split_mode={split_mode}), {df_m['lnf_code'].nunique()} crops")

    print("Loading area_ha from β per-stratum CSV ...")
    area = load_area_ha(cfg['beta_per_stratum_path'])
    print(f"  area_ha available for {len(area)} crops"
          if len(area) else "  area_ha unavailable")

    print("Loading crop_de names ...")
    names = load_crop_names(cfg['lnf_classification_path'])
    print(f"  bridged {len(names)} crops")

    print("Inner-joining per-pixel predictions ...")
    pix = inner_join_pixels(df_b, df_m)
    dropped_b = len(df_b) - len(pix)
    dropped_m = len(df_m) - len(pix)
    print(f"  {len(pix):,} pixels common to both pipelines  "
          f"(β dropped {dropped_b:,}, ML dropped {dropped_m:,})")
    if len(pix) == 0:
        raise RuntimeError("No common pixels — check inputs.")

    min_n = cfg.get('min_n_pixels_per_stratum', 1)
    if min_n > 1:
        n_per = pix.groupby(['lnf_code', 'region', 'tillage']).size()
        keep_keys = n_per[n_per >= min_n].index
        pix = (pix.set_index(['lnf_code', 'region', 'tillage'])
                  .loc[keep_keys].reset_index())
        print(f"  {len(pix):,} pixels after min_n_pixels_per_stratum={min_n}")

    print("Aggregating to stratum level + computing weights ...")
    strata = aggregate_to_stratum(pix, area)
    strata = compute_weights(strata)
    strata['winner'] = winner_col(strata)
    if len(names):
        strata = strata.merge(names, on='lnf_code', how='left')

    h = headline_metrics(strata)

    crops = rollup_to_crop(strata)
    if len(names):
        crops = crops.merge(names, on='lnf_code', how='left')
    crops['winner'] = winner_col(crops)

    per_yr = pd.DataFrame()
    if pix['yr'].nunique() > 1:
        print("Per-year breakdown ...")
        per_yr = per_year_metrics(pix, area)

    # ---- CSVs ----
    strat_cols = ['lnf_code']
    if 'crop_de' in strata.columns:
        strat_cols.append('crop_de')
    strat_cols += ['region', 'tillage', 'n_pixels', 'C_ref',
                   'mean_pred_beta', 'mean_pred_ml',
                   'bias_beta', 'bias_ml',
                   'abs_bias_beta', 'abs_bias_ml',
                   'area_ha', 'weight', 'winner']
    (strata[strat_cols]
      .sort_values('weight', ascending=False)
      .to_csv(out_dir / 'compare_per_stratum.csv', index=False))
    print(f"  wrote compare_per_stratum.csv")

    crop_cols = ['lnf_code']
    if 'crop_de' in crops.columns:
        crop_cols.append('crop_de')
    crop_cols += ['n_strata', 'n_pixels', 'area_ha',
                  'C_ref_mean', 'mean_pred_beta', 'mean_pred_ml',
                  'abs_bias_beta', 'abs_bias_ml', 'winner']
    (crops.sort_values('area_ha', ascending=False, na_position='last')
          [crop_cols]
          .to_csv(out_dir / 'compare_per_crop.csv', index=False))
    print(f"  wrote compare_per_crop.csv")

    if len(per_yr):
        per_yr.to_csv(out_dir / 'compare_per_year.csv', index=False)
        print(f"  wrote compare_per_year.csv")

    # ---- Plots ----
    print("Plots ...")
    ml_phrase = ml_source_phrase(effective_source, split_mode)
    plot_scatter_side_by_side(strata, str(plot_dir / 'compare_scatter.png'),
                               h, ml_label=ml_phrase)
    plot_paired_residuals(strata, str(plot_dir / 'compare_paired.png'))
    plot_per_crop_bars(crops, str(plot_dir / 'compare_per_crop_bars.png'),
                       cfg.get('top_n_crops', 15))
    if len(per_yr):
        plot_per_year(per_yr, str(plot_dir / 'compare_per_year.png'),
                      ml_label=f'ML ({ml_phrase})')

    # ---- Summary ----
    write_summary(h, strata, crops, per_yr,
                  effective_source=effective_source,
                  split_mode=split_mode,
                  path=str(out_dir / 'compare_summary.txt'))

    # ---- Console headline ----
    print()
    print('=' * 60)
    print(f"β  area-weighted stratum MAE : {h['beta_stratum_mae']:.5f}")
    print(f"ML area-weighted stratum MAE : {h['ml_stratum_mae']:.5f}")
    delta = h['beta_stratum_mae'] - h['ml_stratum_mae']
    print(f"Δ (β − ML)                   : {delta:+.5f}  "
          f"({'ML wins' if delta > 1e-4 else 'β wins' if delta < -1e-4 else 'tie'})")
    print('=' * 60)


if __name__ == '__main__':
    run(CONFIG)