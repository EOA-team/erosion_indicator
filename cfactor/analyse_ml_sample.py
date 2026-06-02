"""Analysis of the ML C-factor model predictions.

Counterpart to ``analyse_calibration_sample.py``, adapted for the outputs of
``train_cfactor_ml.py``. Same questions, same plots, same summary skeleton —
but two things differ from the β-calibration case and are handled explicitly:

1. **There is an evaluation split.** ``train_cfactor_ml.py`` writes a ``split``
   column whose values depend on ``split_strategy``:
     - *parcel*: ``'train'`` / ``'test'`` (grouped by poly_id). Plots default
       to the TEST rows.
     - *loyo*:   year strings (``'2019'``, ``'2020'``, …). Every row is an
       honest out-of-fold prediction. Plots default to ALL rows (all OOF).
   This script auto-detects the mode and adapts metrics, plots, and summary.

2. **Column names.** The NN per-pixel CSV uses ``C_pred`` (not
   ``C_predicted``). It is renamed on load so all the shared plotting/metric
   helpers from ``analyse_calibration_sample`` work unchanged.

What this reuses vs. what it adds
---------------------------------
Reused verbatim from ``analyse_calibration_sample`` (imported, not copied):
  - ``stratum_metrics``                  per-group bias/MAE/RMSE/spread/n
  - ``load_lnf_bridge`` / ``_norm_name`` crop_de bridge
  - all ``plot_*`` functions             scatter / strengths-weaknesses /
                                          box / interannual / residual-vs-n,
                                          both unstratified and stratified
  - ``_detect_stratified``, ``_stratum_label``

Added here (NN-specific):
  - split-aware metric tables (train/test) per crop or per stratum
  - a train-vs-test scatter so over-fit is visible at a glance
  - ``nn_summary.txt`` with the same "How to read this" guidance, retuned for
    a learned model (over-fit, not optimiser convergence, is the risk)

Inputs (defaults read from calibration_analysis_{model}/, where train_cfactor_ml.py writes)
-------------------------------------------------------------------------------------
- ``nn_predictions_per_pixel.csv``   per-pixel: ts_cols [+region,tillage] +
                                       C_ref, C_pred, split
- ``nn_predictions_per_crop.csv``    crop (or crop×stratum) means + C_ref
                                       (used only for cross-checking)
- ``nn_metrics.json``                the model's own train/test metrics
- ``LNF_classification.xlsx``        lnf_code ↔ Crop_DE
- ``C_Faktoren.csv``                 only needed to label C_ref provenance

Outputs (written to ``cfg['results_dir']``)
-------------------------------------------
- ``nn_summary.txt``
- ``nn_per_crop.csv`` / ``nn_per_stratum.csv``   (one block per split)
- ``nn_by_region.csv`` / ``nn_by_tillage.csv``   (stratified, test split)
- ``nn_by_c_magnitude.csv``
- plots/  (drawn on cfg['analyse_split'], default 'test')
    nn_scatter_per_{crop,stratum}.png
    nn_strengths_weaknesses[_per_stratum].png
    nn_box_per_{crop,stratum}.png
    nn_interannual[_per_stratum].png
    nn_residual_vs_n[_per_stratum].png
    nn_train_vs_test_scatter.png        (NN-specific: over-fit check)
"""

from __future__ import annotations

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Reuse all the shared machinery from the calibration analyser.
from analyse_calibration_sample import (
    _norm_name,
    load_lnf_bridge,
    stratum_metrics,
    _detect_stratified,
    _stratum_label,
    plot_scatter_per_crop,
    plot_strengths_weaknesses,
    plot_box_per_crop,
    plot_interannual,
    plot_residual_vs_n,
    plot_scatter_per_stratum,
    plot_strengths_weaknesses_stratum,
    plot_box_per_stratum,
    plot_interannual_stratum,
    plot_residual_vs_n_stratum,
    REGION_COLORS,
    TILLAGE_MARKERS,
)


# ===========================================================================
# Config
# ===========================================================================
CONFIG = {
    'results_dir':              'calibration_analysis_mlp/',
    'per_pixel_path':           'calibration_analysis_mlp/nn_predictions_per_pixel.csv',
    'per_crop_path':            'calibration_analysis_mlp/nn_predictions_per_crop.csv',
    'metrics_json_path':        'calibration_analysis_mlp/nn_metrics.json',
    'lnf_classification_path':  '~/mnt/eo-nas1/data/landuse/documentation/'
                                'LNF_code_classification_20260217.xlsx',

    # Which split the plots are drawn on. 'all' = whole dataset (train+test
    # pooled), 'test' = honest generalisation only, 'train' = fit only.
    'analyse_split':            'all',

    'min_n_per_crop':           10,
    # None → auto-detect (region + tillage columns present).
    'stratified_mode':          None,
}


# ===========================================================================
# Loading
# ===========================================================================
def _detect_split_mode(df: pd.DataFrame) -> str:
    """Return 'parcel', 'loyo', or 'none' based on the split column values."""
    if 'split' not in df.columns:
        return 'none'
    vals = set(df['split'].dropna().unique())
    if vals <= {'train', 'test'}:
        return 'parcel'
    # Year values (int or string like '2019') → LOYO
    if all(str(v).isdigit() and len(str(v)) == 4 for v in vals):
        # Normalise to string so downstream groupby/filtering is consistent
        df['split'] = df['split'].astype(str)
        return 'loyo'
    return 'parcel'  # fallback


def load_nn_pixels(cfg: dict, stratified_override) -> tuple[pd.DataFrame, bool, str]:
    """Load NN per-pixel predictions, rename C_pred→C_predicted, attach crop_de.

    Returns (df, stratified, split_mode) where split_mode is 'parcel', 'loyo',
    or 'none'.
    """
    df = pd.read_csv(cfg['per_pixel_path'])
    if 'C_pred' in df.columns and 'C_predicted' not in df.columns:
        df = df.rename(columns={'C_pred': 'C_predicted'})
    if 'C_pred_final' in df.columns:
        df = df.rename(columns={'C_pred_final': 'C_predicted_final'})
    if 'split' not in df.columns:
        print("  WARNING: no 'split' column — treating all rows as a single "
              "split. Re-run train_cfactor_ml.py to get split breakdown.")
        df['split'] = 'all'

    split_mode = _detect_split_mode(df)

    stratified = _detect_stratified(df, stratified_override)
    bridge = load_lnf_bridge(cfg['lnf_classification_path'])
    df = df.merge(bridge[['lnf_code', 'crop_de']], on='lnf_code', how='left')
    return df, stratified, split_mode


def metrics_table(per_pixel: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Per-group stratum_metrics + carry crop_de / region / tillage / C_ref."""
    rows = []
    for key, g in per_pixel.groupby(group_cols, dropna=False):
        m = stratum_metrics(g)
        if not isinstance(key, tuple):
            key = (key,)
        for col, val in zip(group_cols, key):
            m[col] = val
        # carry descriptive columns (constant within group)
        for c in ('crop_de', 'region', 'tillage', 'C_ref'):
            if c in g.columns and c not in group_cols:
                m[c] = g[c].iloc[0]
        rows.append(m)
    return pd.DataFrame(rows)


# ===========================================================================
# Split-aware plots
# ===========================================================================
def plot_train_vs_test(per_pixel: pd.DataFrame, group_cols: list[str],
                       out: str, stratified: bool) -> None:
    """Crop/stratum-mean predicted vs reference, train and test overlaid.

    Over-fit shows up as test points scattering off the 1:1 line while train
    points hug it.
    """
    splits = [s for s in ('train', 'test') if s in per_pixel['split'].unique()]
    if not splits:
        return
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    colors = {'train': '#999999', 'test': '#D7263D'}
    mx = 0.0
    for sp in splits:
        t = metrics_table(per_pixel[per_pixel['split'] == sp], group_cols)
        t = t.dropna(subset=['mean_ref', 'mean_new'])
        if t.empty:
            continue
        n = t['n'].clip(lower=1).values
        sizes = 25 + 200 * np.sqrt(n / max(n.max(), 1))
        bias = (t['mean_new'] - t['mean_ref']).mean()
        mae = (t['mean_new'] - t['mean_ref']).abs().mean()
        ax.scatter(t['mean_ref'], t['mean_new'], s=sizes, alpha=0.65,
                   c=colors.get(sp, 'grey'), edgecolor='k', linewidth=0.4,
                   label=f'{sp}  (bias={bias:+.4f}, MAE={mae:.4f})')
        mx = max(mx, t['mean_ref'].max(), t['mean_new'].max())
    m = mx * 1.1 if mx > 0 else 1.0
    ax.plot([0, m], [0, m], 'k--', lw=1, alpha=0.5, label='1:1')
    ax.set_xlim(0, m); ax.set_ylim(0, m)
    unit = 'stratum' if stratified else 'crop'
    ax.set_xlabel('C_ref (tabulated)')
    ax.set_ylabel(f'C_pred ({unit}-mean of pixels)')
    ax.set_title(f'{unit}-mean predicted vs reference — train vs test\n'
                 'test scattering off 1:1 while train hugs it = over-fit')
    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"  saved {out}")


def plot_loyo_per_year(per_pixel: pd.DataFrame, group_cols: list[str],
                       out: str, stratified: bool) -> None:
    """Crop/stratum-mean predicted vs reference, one colour per held-out year.

    All predictions are out-of-fold; points far from 1:1 for a specific year
    indicate that year's FC/EI distribution is poorly predicted by the
    remaining years.
    """
    years = sorted(per_pixel['split'].unique())
    if len(years) < 2:
        return
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    cmap = plt.cm.tab10
    mx = 0.0
    for i, yr in enumerate(years):
        t = metrics_table(per_pixel[per_pixel['split'] == yr], group_cols)
        t = t.dropna(subset=['mean_ref', 'mean_new'])
        if t.empty:
            continue
        n = t['n'].clip(lower=1).values
        sizes = 25 + 200 * np.sqrt(n / max(n.max(), 1))
        mae = (t['mean_new'] - t['mean_ref']).abs().mean()
        ax.scatter(t['mean_ref'], t['mean_new'], s=sizes, alpha=0.6,
                   c=[cmap(i % 10)], edgecolor='k', linewidth=0.3,
                   label=f'{yr}  (MAE={mae:.4f}, n_groups={len(t)})')
        mx = max(mx, t['mean_ref'].max(), t['mean_new'].max())
    m = mx * 1.1 if mx > 0 else 1.0
    ax.plot([0, m], [0, m], 'k--', lw=1, alpha=0.5, label='1:1')
    ax.set_xlim(0, m); ax.set_ylim(0, m)
    unit = 'stratum' if stratified else 'crop'
    ax.set_xlabel('C_ref (tabulated)')
    ax.set_ylabel(f'C_pred (LOYO OOF, {unit}-mean)')
    ax.set_title(f'{unit}-mean predicted vs reference — per held-out year\n'
                 'all predictions are out-of-fold (model never saw this year)')
    ax.legend(loc='upper left', fontsize=7)
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"  saved {out}")


def plot_loyo_vs_final(per_pixel: pd.DataFrame, group_cols: list[str],
                       out: str, stratified: bool) -> None:
    """Compare LOYO OOF crop-means vs final-model (all-years) crop-means,
    both plotted against C_ref on the same axes.

    The gap between the two shows how much temporal leave-out penalises
    the predictions versus the model you'd actually deploy.
    """
    if 'C_predicted_final' not in per_pixel.columns:
        return

    unit = 'stratum' if stratified else 'crop'
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    mx = 0.0

    # OOF predictions (C_predicted)
    t_oof = metrics_table(per_pixel, group_cols).dropna(subset=['mean_ref', 'mean_new'])
    if not t_oof.empty:
        mae_oof = (t_oof['mean_new'] - t_oof['mean_ref']).abs().mean()
        ax.scatter(t_oof['mean_ref'], t_oof['mean_new'], alpha=0.5,
                   c='#D7263D', edgecolor='k', linewidth=0.3, s=60,
                   label=f'LOYO OOF  (MAE={mae_oof:.4f})')
        mx = max(mx, t_oof['mean_ref'].max(), t_oof['mean_new'].max())

    # Final model predictions — temporarily swap columns to reuse metrics_table
    tmp = per_pixel.copy()
    tmp['C_predicted'] = tmp['C_predicted_final']
    t_fin = metrics_table(tmp, group_cols).dropna(subset=['mean_ref', 'mean_new'])
    if not t_fin.empty:
        mae_fin = (t_fin['mean_new'] - t_fin['mean_ref']).abs().mean()
        ax.scatter(t_fin['mean_ref'], t_fin['mean_new'], alpha=0.5,
                   c='#2E8B57', edgecolor='k', linewidth=0.3, s=60,
                   marker='D',
                   label=f'Final model  (MAE={mae_fin:.4f})')
        mx = max(mx, t_fin['mean_ref'].max(), t_fin['mean_new'].max())

    m = mx * 1.1 if mx > 0 else 1.0
    ax.plot([0, m], [0, m], 'k--', lw=1, alpha=0.5, label='1:1')
    ax.set_xlim(0, m); ax.set_ylim(0, m)
    ax.set_xlabel('C_ref (tabulated)')
    ax.set_ylabel(f'C_pred ({unit}-mean)')
    ax.set_title(f'LOYO OOF vs final model (all years) — {unit}-level\n'
                 'gap = temporal leave-out penalty')
    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"  saved {out}")


# ===========================================================================
# Runner
# ===========================================================================
def run(cfg: dict) -> None:
    out_dir = Path(cfg['results_dir']); out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / 'plots'; plot_dir.mkdir(exist_ok=True)

    per_pixel, stratified, split_mode = load_nn_pixels(cfg, cfg.get('stratified_mode'))
    mode_label = 'STRATIFIED' if stratified else 'unstratified'
    group_cols = (['lnf_code', 'region', 'tillage'] if stratified
                  else ['lnf_code'])
    unit = 'stratum' if stratified else 'crop'
    print(f"ML analysis — {mode_label} mode, split_mode={split_mode}, "
          f"{len(per_pixel):,} pixel rows, splits="
          f"{sorted(per_pixel['split'].unique())}")

    # The model's own reported metrics, if present.
    model_metrics = None
    mpath = cfg.get('metrics_json_path')
    if mpath and os.path.exists(mpath):
        with open(mpath) as f:
            model_metrics = json.load(f)

    # -------------------------------------------------------------------
    # Per-group metric tables, one block per split
    # -------------------------------------------------------------------
    per_unit_blocks = []
    for sp, g in per_pixel.groupby('split'):
        t = metrics_table(g, group_cols)
        t.insert(0, 'split', sp)
        per_unit_blocks.append(t)
    per_unit = pd.concat(per_unit_blocks, ignore_index=True)
    fname = f"nn_per_{unit}.csv"
    per_unit.to_csv(out_dir / fname, index=False)
    print(f"  wrote {fname}")

    # -------------------------------------------------------------------
    # Select rows for plots
    # -------------------------------------------------------------------
    # Default analyse_split: 'test' for parcel (honest OOS), 'all' for LOYO
    # (every row is OOF).
    asplit = cfg.get('analyse_split')
    if asplit is None:
        asplit = 'test' if split_mode == 'parcel' else 'all'

    if asplit == 'all':
        plot_pixels = per_pixel.copy()
    else:
        plot_pixels = per_pixel[per_pixel['split'] == asplit].copy()
        if plot_pixels.empty:
            print(f"  WARNING: split '{asplit}' empty; falling back to all rows.")
            plot_pixels = per_pixel.copy()
            asplit = 'all'
    print(f"  plots drawn on split='{asplit}' ({len(plot_pixels):,} rows)")

    # Reference-C magnitude classes
    by_mag = None
    if plot_pixels['C_ref'].notna().any():
        pp = plot_pixels.dropna(subset=['C_ref']).copy()
        pp['C_ref_class'] = pd.cut(
            pp['C_ref'],
            bins=[-np.inf, 0.01, 0.05, 0.1, 0.2, np.inf],
            labels=['≤0.01', '0.01–0.05', '0.05–0.10', '0.10–0.20', '>0.20'])
        rows = []
        for cls, g in pp.groupby('C_ref_class', observed=True):
            m = stratum_metrics(g); m['C_ref_class'] = cls
            rows.append(m)
        by_mag = pd.DataFrame(rows)
        by_mag.to_csv(out_dir / 'nn_by_c_magnitude.csv', index=False)
        print("  wrote nn_by_c_magnitude.csv")

    # Stratified rollups (on analysed split)
    by_region = by_tillage = None
    if stratified:
        by_region = metrics_table(plot_pixels, ['region'])
        by_region.to_csv(out_dir / 'nn_by_region.csv', index=False)
        by_tillage = metrics_table(plot_pixels, ['tillage'])
        by_tillage.to_csv(out_dir / 'nn_by_tillage.csv', index=False)
        print("  wrote nn_by_region.csv, nn_by_tillage.csv")

    # -------------------------------------------------------------------
    # Plots — reuse the calibration plotters on the analysed split.
    # -------------------------------------------------------------------
    print("\nPlots...")
    plot_unit = metrics_table(plot_pixels, group_cols)
    if 'crop_de' not in plot_pixels.columns:
        plot_pixels['crop_de'] = plot_pixels['lnf_code'].astype(str)

    if stratified:
        plot_scatter_per_stratum(plot_unit, plot_dir / 'nn_scatter_per_stratum.png')
        plot_strengths_weaknesses_stratum(
            plot_unit, plot_dir / 'nn_strengths_weaknesses_per_stratum.png',
            min_n=cfg['min_n_per_crop'])
        plot_box_per_stratum(plot_pixels, plot_dir / 'nn_box_per_stratum.png',
                             min_n=cfg['min_n_per_crop'])
        plot_interannual_stratum(plot_pixels,
                                 plot_dir / 'nn_interannual_per_stratum.png')
        plot_residual_vs_n_stratum(
            plot_unit, plot_dir / 'nn_residual_vs_n_per_stratum.png')

        rollup = (plot_unit.dropna(subset=['mean_ref', 'mean_new'])
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
        rollup = rollup.sort_values('MAE', ascending=False).reset_index(drop=True)
        rollup.to_csv(out_dir / 'nn_per_crop_rollup.csv', index=False)
        plot_scatter_per_crop(rollup, plot_dir / 'nn_scatter_per_crop_rollup.png')
        print("  wrote nn_per_crop_rollup.csv + nn_scatter_per_crop_rollup.png")
    else:
        plot_scatter_per_crop(plot_unit, plot_dir / 'nn_scatter_per_crop.png')
        plot_strengths_weaknesses(plot_unit,
                                  plot_dir / 'nn_strengths_weaknesses.png',
                                  min_n=cfg['min_n_per_crop'])
        plot_box_per_crop(plot_pixels, plot_dir / 'nn_box_per_crop.png',
                          min_n=cfg['min_n_per_crop'])
        plot_interannual(plot_pixels, plot_dir / 'nn_interannual.png')
        plot_residual_vs_n(plot_unit, plot_dir / 'nn_residual_vs_n.png')

    # Mode-specific diagnostic plot
    if split_mode == 'loyo':
        plot_loyo_per_year(per_pixel, group_cols,
                           plot_dir / 'nn_loyo_per_year_scatter.png', stratified)
        plot_loyo_vs_final(per_pixel, group_cols,
                           plot_dir / 'nn_loyo_vs_final_scatter.png', stratified)

        # Final model plots — same plotters, using C_predicted_final
        if 'C_predicted_final' in plot_pixels.columns:
            print("\n  Final model (all years) plots...")
            pp_final = plot_pixels.copy()
            pp_final['C_predicted'] = pp_final['C_predicted_final']
            pu_final = metrics_table(pp_final, group_cols)

            if stratified:
                plot_scatter_per_stratum(pu_final,
                    plot_dir / 'nn_scatter_per_stratum_final.png')
                plot_box_per_stratum(pp_final,
                    plot_dir / 'nn_box_per_stratum_final.png',
                    min_n=cfg['min_n_per_crop'])
            else:
                plot_scatter_per_crop(pu_final,
                    plot_dir / 'nn_scatter_per_crop_final.png')
                plot_box_per_crop(pp_final,
                    plot_dir / 'nn_box_per_crop_final.png',
                    min_n=cfg['min_n_per_crop'])
    else:
        plot_train_vs_test(per_pixel, group_cols,
                           plot_dir / 'nn_train_vs_test_scatter.png', stratified)

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    def _split_headline(sp_df: pd.DataFrame, label: str) -> list[str]:
        g = sp_df.dropna(subset=['C_ref', 'C_predicted'])
        if g.empty:
            return [f'--- {label} ---', '  (no data)']
        d = g['C_predicted'] - g['C_ref']
        t = metrics_table(g, group_cols).dropna(subset=['mean_ref', 'mean_new'])
        td = t['mean_new'] - t['mean_ref']
        return [
            '',
            f'--- {label} ({len(g):,} pixels, {len(t)} {unit}s) ---',
            f'  pixel-level bias            : {float(d.mean()):+.4f}',
            f'  pixel-level MAE             : {float(d.abs().mean()):.4f}',
            f'  {unit}-level bias'.ljust(31) + f': {float(td.mean()):+.4f}',
            f'  {unit}-level MAE'.ljust(31) + f': {float(td.abs().mean()):.4f}',
        ]

    lines = [
        '=' * 70,
        f'ML analysis  ({mode_label}, split_mode={split_mode})',
        '=' * 70,
        f'Per-pixel rows                : {len(per_pixel):>10,}',
        f'Crops covered                 : {per_pixel["lnf_code"].nunique():>10}',
    ]
    if stratified:
        lines.append(f'Strata covered (analysed)     : '
                     f'{plot_unit.dropna(subset=["mean_new"]).shape[0]:>10}')
    lines.append(f'Years covered                 : '
                 f'{sorted(per_pixel["yr"].dropna().unique().tolist())}'
                 if 'yr' in per_pixel.columns else '')

    if split_mode == 'loyo':
        # Aggregate LOYO (all rows are OOF)
        lines += _split_headline(per_pixel, 'LOYO aggregate (all rows are OOF)')
        # Per-year breakdown
        for yr in sorted(per_pixel['split'].unique()):
            yr_df = per_pixel[per_pixel['split'] == yr]
            lines += _split_headline(yr_df, f'Year {yr} (OOF)')
        # Final model (trained on all years, in-sample)
        if 'C_predicted_final' in per_pixel.columns:
            tmp = per_pixel.copy()
            tmp['C_predicted'] = tmp['C_predicted_final']
            lines += _split_headline(tmp, 'Final model (all years, in-sample)')
    else:
        for sp in ('train', 'test', 'all'):
            if sp not in per_pixel['split'].unique():
                continue
            sp_df = per_pixel[per_pixel['split'] == sp]
            lines += _split_headline(sp_df, sp.upper())

    # Model-reported metrics from JSON
    if model_metrics is not None:
        if 'loyo_aggregate' in model_metrics:
            cl = model_metrics['loyo_aggregate'].get('crop_level', {})
            lines += ['',
                      '--- model-reported LOYO aggregate crop-level (nn_metrics.json) ---',
                      f"  R²  : {cl.get('r2', float('nan')):.4f}",
                      f"  MAE : {cl.get('mae', float('nan')):.4f}",
                      f"  RMSE: {cl.get('rmse', float('nan')):.4f}"]
        if 'final_model_insample' in model_metrics:
            fl = model_metrics['final_model_insample'].get('crop_level', {})
            lines += ['',
                      '--- model-reported FINAL MODEL crop-level (nn_metrics.json) ---',
                      f"  R²  : {fl.get('r2', float('nan')):.4f}",
                      f"  MAE : {fl.get('mae', float('nan')):.4f}",
                      f"  RMSE: {fl.get('rmse', float('nan')):.4f}"]
        if 'crop_level' in model_metrics and 'loyo_aggregate' not in model_metrics:
            cl = model_metrics['crop_level']
            lines += ['',
                      '--- model-reported crop-level (nn_metrics.json) ---',
                      f"  R²  : {cl.get('r2', float('nan')):.4f}",
                      f"  MAE : {cl.get('mae', float('nan')):.4f}",
                      f"  RMSE: {cl.get('rmse', float('nan')):.4f}"]

    # worst/best on the analysed split
    cols = (['crop_de', 'region', 'tillage', 'n', 'C_ref',
             'mean_new', 'bias', 'MAE', 'spread'] if stratified
            else ['crop_de', 'n', 'C_ref', 'mean_new', 'bias', 'MAE', 'spread'])
    cols = [c for c in cols if c in plot_unit.columns]
    elig = plot_unit[plot_unit['n'] >= cfg['min_n_per_crop']].dropna(subset=['MAE'])
    lines += ['', f'--- Worst-fitting {unit}s on {asplit} (MAE, n≥{cfg["min_n_per_crop"]}) ---',
              elig.nlargest(5, 'MAE')[cols].to_string(index=False, float_format='%.4f'),
              '', f'--- Best-fitting {unit}s on {asplit} ---',
              elig.nsmallest(5, 'MAE')[cols].to_string(index=False, float_format='%.4f')]

    if stratified and by_region is not None:
        lines += ['', '--- By region (analysed split) ---',
                  by_region[['region', 'n', 'mean_ref', 'mean_new', 'bias', 'MAE']]
                    .to_string(index=False, float_format='%.4f')]
        lines += ['', '--- By tillage (analysed split) ---',
                  by_tillage[['tillage', 'n', 'mean_ref', 'mean_new', 'bias', 'MAE']]
                    .to_string(index=False, float_format='%.4f')]
    if by_mag is not None:
        lines += ['', '--- By reference-C magnitude (analysed split) ---',
                  by_mag[['C_ref_class', 'n', 'mean_ref', 'mean_new', 'bias', 'MAE']]
                    .to_string(index=False, float_format='%.4f')]

    lines += ['', '=' * 70, 'How to read this', '=' * 70]
    if split_mode == 'loyo':
        lines.append(f"""
- All OOF predictions are OUT-OF-FOLD: each row was predicted by a model that
  never saw its year. The aggregate metrics are the honest estimate of
  operational performance on a new year.

- The FINAL MODEL (all years) predictions are in-sample. They show what the
  deployed model produces but are NOT an honest evaluation — use them to
  inspect model behaviour, not to claim performance. The nn_loyo_vs_final
  scatter shows both side by side: the gap is the temporal leave-out penalty.

- Per-year breakdown shows which years are hardest. A year with high MAE
  likely had unusual weather (drought, wet year) whose FC/EI distribution
  differs from the training years. Check whether specific crops drive
  the error (nn_per_{unit}.csv filtered by split=year).

- The saved model (nn_model.joblib) IS the final all-years model. Expect
  its true performance on a genuinely new year to lie between the LOYO
  aggregate (pessimistic — each fold had less data) and the final model
  in-sample (optimistic — the model saw everything).

- Because every pixel of a crop was given that crop's C_ref as its label,
  the meaningful metric is {unit}-level MAE; pixel-level MAE is inflated
  by the broadcast labels.
""")
    else:
        lines.append(f"""
- TRAIN vs TEST is the headline for a learned model. If train MAE is much
  lower than test MAE, the model memorised the training pixels — tighten
  regularisation (config['model_params']['alpha']), shrink hidden layers
  (MLP), or get more crops. With β there was no such gap to worry about.

- Because every pixel of a crop was given that crop's C_ref as its label,
  the meaningful metric is {unit}-level MAE on the TEST split; pixel-level
  MAE is inflated by the broadcast labels.

- Within-{unit} spread of C_pred shows whether the model learned to move
  pixels around within a crop from their FC/EI features, or just predicts
  one value per crop.

- A test {unit} far off the 1:1 line in the train-vs-test scatter, with
  large n, is a genuine generalisation failure for that crop/stratum.
""")

    (out_dir / 'nn_summary.txt').write_text('\n'.join(lines), encoding='utf-8')
    print(f"  wrote {out_dir / 'nn_summary.txt'}")
    print()
    print('\n'.join(lines[:30]))


if __name__ == '__main__':
    run(CONFIG)