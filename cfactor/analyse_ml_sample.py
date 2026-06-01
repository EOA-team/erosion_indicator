"""In-sample analysis of the ML C-factor model.

Counterpart to ``analyse_calibration_sample.py``, adapted for the outputs of
``train_cfactor_ml.py``. Same questions, same plots, same summary skeleton —
but two things differ from the β-calibration case and are handled explicitly:

1. **There is a held-out test set.** ``train_cfactor_ml.py`` writes a ``split``
   column ('train'/'test') with a *grouped* split (by ``poly_id``). The β
   calibration had no such split. So every headline metric here is reported
   **train vs test**, and the scatter/strengths plots are drawn on the TEST
   rows by default (override with ``cfg['analyse_split']``). Test-set
   crop/stratum performance is the honest read on generalisation; train-set
   numbers only tell you the model converged.

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
def load_nn_pixels(cfg: dict, stratified_override) -> tuple[pd.DataFrame, bool]:
    """Load NN per-pixel predictions, rename C_pred→C_predicted, attach crop_de."""
    df = pd.read_csv(cfg['per_pixel_path'])
    if 'C_pred' in df.columns and 'C_predicted' not in df.columns:
        df = df.rename(columns={'C_pred': 'C_predicted'})
    if 'split' not in df.columns:
        print("  WARNING: no 'split' column — treating all rows as a single "
              "split. Re-run train_cfactor_nn.py to get train/test breakdown.")
        df['split'] = 'all'

    stratified = _detect_stratified(df, stratified_override)

    bridge = load_lnf_bridge(cfg['lnf_classification_path'])
    df = df.merge(bridge[['lnf_code', 'crop_de']], on='lnf_code', how='left')
    return df, stratified


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
# NN-specific plot: train vs test, side by side
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
    ax.set_ylabel(f'C_pred (NN, {unit}-mean of pixels)')
    ax.set_title(f'NN {unit}-mean predicted vs reference — train vs test\n'
                 'test scattering off 1:1 while train hugs it = over-fit')
    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"  saved {out}")


# ===========================================================================
# Runner
# ===========================================================================
def run(cfg: dict) -> None:
    out_dir = Path(cfg['results_dir']); out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / 'plots'; plot_dir.mkdir(exist_ok=True)

    per_pixel, stratified = load_nn_pixels(cfg, cfg.get('stratified_mode'))
    mode_label = 'STRATIFIED' if stratified else 'unstratified'
    group_cols = (['lnf_code', 'region', 'tillage'] if stratified
                  else ['lnf_code'])
    unit = 'stratum' if stratified else 'crop'
    print(f"NN analysis — {mode_label} mode, "
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

    # Reference-C magnitude classes (computed on the analysed split only)
    asplit = cfg.get('analyse_split', 'test')
    if asplit == 'all':
        plot_pixels = per_pixel.copy()
    else:
        plot_pixels = per_pixel[per_pixel['split'] == asplit].copy()
        if plot_pixels.empty:
            print(f"  WARNING: split '{asplit}' empty; falling back to all rows.")
            plot_pixels = per_pixel.copy()
    print(f"  plots drawn on split='{asplit}' ({len(plot_pixels):,} rows)")

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
    # The shared plotters expect a `per_crop`/`per_stratum` table with
    # mean_ref/mean_new/bias/spread/n + crop_de [+region/tillage].
    # -------------------------------------------------------------------
    print("\nPlots...")
    plot_unit = metrics_table(plot_pixels, group_cols)
    # shared plotters key boxplots/interannual on crop_de; ensure present
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

        # Crop-level rollup: collapse each crop's strata back to one row and
        # reuse the per-crop scatter, so it sits side-by-side with the
        # per-stratum view (mirrors cal_scatter_per_crop_rollup in the
        # calibration analyser).
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

    # NN-specific: over-fit check
    plot_train_vs_test(per_pixel, group_cols,
                       plot_dir / 'nn_train_vs_test_scatter.png', stratified)

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    def _split_headline(sp: str) -> tuple[float, float, float, float]:
        g = per_pixel[per_pixel['split'] == sp].dropna(
            subset=['C_ref', 'C_predicted'])
        if g.empty:
            return (np.nan,) * 4
        d = g['C_predicted'] - g['C_ref']
        t = metrics_table(g, group_cols).dropna(subset=['mean_ref', 'mean_new'])
        td = t['mean_new'] - t['mean_ref']
        return (float(d.mean()), float(d.abs().mean()),
                float(td.mean()), float(td.abs().mean()))

    lines = [
        '=' * 70,
        f'NN in-sample analysis  ({mode_label})',
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

    for sp in ('train', 'test', 'all'):
        if sp not in per_pixel['split'].unique():
            continue
        pb, pm, tb, tm = _split_headline(sp)
        lines += [
            '',
            f'--- {sp.upper()} split ---',
            f'  pixel-level bias            : {pb:+.4f}',
            f'  pixel-level MAE             : {pm:.4f}',
            f'  {unit}-level bias'.ljust(31) + f': {tb:+.4f}',
            f'  {unit}-level MAE'.ljust(31) + f': {tm:.4f}',
        ]

    if model_metrics is not None:
        cl = model_metrics.get('crop_level', {})
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

    lines += ['', '=' * 70, 'How to read this', '=' * 70, """
- TRAIN vs TEST is the headline for a learned model. If train MAE is much
  lower than test MAE, the model memorised the training pixels — tighten
  regularisation (config['model_params']['alpha']), shrink hidden layers
  (MLP), or get more crops. With β there was no such gap to worry about.

- Because every pixel of a crop was given that crop's C_ref as its label,
  the NN can at best recover the crop/stratum MEAN. So the meaningful
  metric is {unit}-level MAE on the TEST split; pixel-level MAE is inflated
  by the broadcast labels and is not a fault of the model.

- Within-{unit} spread of C_pred shows whether the NN learned to move
  pixels around within a crop from their FC/EI features, or just predicts
  one value per crop. Large spread + small {unit} bias = it is using the
  satellite signal; near-zero spread = it collapsed to the crop mean
  (which is all the labels actually told it to do).

- A test {unit} far off the 1:1 line in the train-vs-test scatter, with
  large n, is a genuine generalisation failure for that crop/stratum —
  check its feature distribution against neighbours.

- In stratified mode, a region or tillage class with large bias AND few
  pixels (low n in nn_by_region/tillage) is a coverage problem, not a
  model problem: the NN never saw enough of that cell.
""".replace('{unit}', unit)]

    (out_dir / 'nn_summary.txt').write_text('\n'.join(lines), encoding='utf-8')
    print(f"  wrote {out_dir / 'nn_summary.txt'}")
    print()
    print('\n'.join(lines[:30]))


if __name__ == '__main__':
    run(CONFIG)