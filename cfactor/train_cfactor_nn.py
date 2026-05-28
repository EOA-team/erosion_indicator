"""C-factor estimation via a neural network on Sentinel-2 FC + rainfall erosivity.

This is a **drop-in alternative to ``calibrate_cfactor.py``**. Instead of fitting
the single physical parameter β in the soil-loss-ratio model
``SLR(t) = exp(-β·FC(t))`` and aggregating with EI weights, this module trains a
``sklearn.neural_network.MLPRegressor`` to map *aggregated per-pixel features*
directly to the tabulated C-factor.

Design decisions (fixed by the project owner)
---------------------------------------------
1. **What the NN learns.** It maps one feature vector *per sampled pixel time
   series* (one row per ``(lnf_code, yr, poly_id)``) directly to a C-factor.
   The EI-weighted ``exp(-β·FC)`` kernel is *not* used — the NN replaces the
   whole FC/EI → C mapping.

2. **Supervision target.** There is no per-pixel ground-truth C-factor; the
   reference table only gives one C per crop (unstratified) or per
   (crop, region, tillage) stratum (stratified). We therefore **broadcast the
   crop's (or stratum's) ``C_ref`` onto every pixel of that crop/stratum** and
   train at the pixel level. Many rows, but every pixel of a crop shares the
   same label, so the NN effectively learns to predict the crop/stratum mean
   from pixel features; within-crop FC/EI variation acts as label noise. This
   mirrors the calibration's "truth lives at the crop level" assumption while
   giving the NN enough rows to fit.

3. **Reference granularity** follows ``config['stratified_calibration']`` —
   exactly like ``run_calibration`` dispatches to the stratified path. When
   True, strata (region × tillage) are assigned with the same
   ``assign_strata`` machinery and the target is the stratified ``C_ref``;
   when False, the per-crop ``Total`` column is used.

4. **Features (EI-weighted summary per pixel).** For each pixel time series we
   compute EI-weighted and plain summaries of the fractional-cover signal:
     - ``fc_total`` (= (PV+NPV)·100): EI-weighted mean, plain mean, min, max, std
     - ``pv``:  EI-weighted mean, plain mean
     - ``npv``: EI-weighted mean, plain mean
     - EI:      sum, mean, max
     - ``n_obs``: number of (gap-filled) timesteps in the series
   The EI-weighted FC mean is the most direct analogue of the calibration's
   ``Σ exp(-β·FC)·EI / Σ EI`` numerator, so the NN has access to essentially
   the same information the β model used, plus shape statistics.

5. **Framework.** ``sklearn.neural_network.MLPRegressor`` (CPU, no GPU needed).

Reuse
-----
All data loading (FC parquet, EI snap/join, reference-table resolution, stratum
assignment) is imported from ``calibrate_cfactor`` so the inputs are *identical*
to the calibration path. Only the final β-fit step is replaced.

Outputs (written to ``config['results_folder']``)
-------------------------------------------------
- ``nn_model.joblib``                   — fitted Pipeline (scaler + MLP) + metadata
- ``nn_training_features.csv``          — per-pixel features + target + split flag
- ``nn_predictions_per_pixel.csv``      — per-pixel NN C prediction (all rows)
- ``nn_predictions_per_crop.csv``       — pixel predictions averaged to crop
                                          (or crop×stratum) vs C_ref
- ``nn_metrics.json``                   — train/test R², MAE, RMSE (pixel & crop)
- ``nn_scatter.png``                    — predicted vs reference at crop level
- ``nn_loss_curve.png``                 — MLP training loss curve

Entry point: ``run_nn_training(config)`` — uses the same CONFIG dict as
``main.py`` plus the optional ``nn_*`` keys documented in ``DEFAULT_NN_PARAMS``.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# Reuse the calibration module's data plumbing so inputs are identical.
from calibrate_cfactor import (
    get_ei_grid_offset,
    snap_to_ei_grid,
    load_ei_for_pixels,
    join_ei_to_fc,
    load_reference_cfactors,
    load_reference_cfactors_stratified,
    assign_strata,
)


# ---------------------------------------------------------------------------
# Default NN hyper-parameters (override via config['nn_params'])
# ---------------------------------------------------------------------------
DEFAULT_NN_PARAMS = {
    'hidden_layer_sizes': (64, 32),
    'activation':         'relu',
    'solver':             'adam',
    'alpha':              1e-3,        # L2 regularisation
    'learning_rate_init': 1e-3,
    'max_iter':           500,
    'early_stopping':     True,
    'validation_fraction': 0.1,
    'n_iter_no_change':   20,
    'random_state':       42,
}


# ---------------------------------------------------------------------------
# Feature engineering: one row per pixel time series
# ---------------------------------------------------------------------------
def build_pixel_features(df: pd.DataFrame, ts_cols: list[str],
                         fc_col: str = 'fc_total', ei_col: str = 'ei',
                         pv_col: str = 'pv', npv_col: str = 'npv',
                         extra_cols: list[str] | None = None) -> pd.DataFrame:
    """Collapse each pixel time series to a single EI-weighted feature row.

    Each ``ts_cols`` group is one sampled-pixel time series (the upstream
    sampling step keeps one pixel per parcel per year), so this aggregates
    over **time** within a pixel. Rows missing FC or EI are dropped before
    aggregation, consistent with ``compute_cfactors_per_pixel``.

    ``extra_cols`` (e.g. ['region', 'tillage', 'x', 'y']) are carried through
    unchanged — they must be constant within a ts_cols group.
    """
    extra_cols = extra_cols or []
    needed = ts_cols + [fc_col, ei_col, pv_col, npv_col] + extra_cols
    valid = df[needed].dropna(subset=[fc_col, ei_col, pv_col, npv_col])

    ei = valid[ei_col].values
    # EI-weighted contributions; weighted means computed via groupby sums.
    work = valid.assign(
        _ei=ei,
        _ei_fc=ei * valid[fc_col].values,
        _ei_pv=ei * valid[pv_col].values,
        _ei_npv=ei * valid[npv_col].values,
    )

    g = work.groupby(ts_cols, sort=False)
    feats = g.agg(
        fc_mean=(fc_col, 'mean'),
        fc_min=(fc_col, 'min'),
        fc_max=(fc_col, 'max'),
        fc_std=(fc_col, 'std'),
        pv_mean=(pv_col, 'mean'),
        npv_mean=(npv_col, 'mean'),
        ei_sum=('_ei', 'sum'),
        ei_mean=('_ei', 'mean'),
        ei_max=('_ei', 'max'),
        n_obs=(fc_col, 'size'),
        _ei_tot=('_ei', 'sum'),
        _ei_fc=('_ei_fc', 'sum'),
        _ei_pv=('_ei_pv', 'sum'),
        _ei_npv=('_ei_npv', 'sum'),
    ).reset_index()

    # EI-weighted means (guard against zero EI sum)
    den = feats['_ei_tot'].replace(0.0, np.nan)
    feats['fc_ei_wmean'] = feats['_ei_fc'] / den
    feats['pv_ei_wmean'] = feats['_ei_pv'] / den
    feats['npv_ei_wmean'] = feats['_ei_npv'] / den
    feats['fc_std'] = feats['fc_std'].fillna(0.0)  # std is NaN for single-obs series
    feats = feats.drop(columns=['_ei_tot', '_ei_fc', '_ei_pv', '_ei_npv'])

    # Attach extra (constant-per-group) columns
    if extra_cols:
        extras = valid[ts_cols + extra_cols].drop_duplicates(subset=ts_cols)
        feats = feats.merge(extras, on=ts_cols, how='left')

    return feats


FEATURE_COLUMNS = [
    'fc_ei_wmean', 'fc_mean', 'fc_min', 'fc_max', 'fc_std',
    'pv_ei_wmean', 'pv_mean', 'npv_ei_wmean', 'npv_mean',
    'ei_sum', 'ei_mean', 'ei_max', 'n_obs',
]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_crop_scatter(df_crop: pd.DataFrame, path: str, stratified: bool) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df_crop['C_ref'], df_crop['C_pred_crop'], alpha=0.7)
    lims = [0, max(df_crop['C_ref'].max(), df_crop['C_pred_crop'].max()) * 1.05]
    ax.plot(lims, lims, 'k--', lw=1, label='1:1')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Reference C-factor')
    ax.set_ylabel('NN predicted C-factor (crop mean)')
    title = 'NN C-factor: predicted vs reference'
    ax.set_title(title + (' (per crop×stratum)' if stratified else ' (per crop)'))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_loss_curve(mlp: MLPRegressor, path: str) -> None:
    if not hasattr(mlp, 'loss_curve_'):
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(mlp.loss_curve_)
    ax.set_xlabel('Iteration'); ax.set_ylabel('Training loss')
    ax.set_title('MLP training loss curve')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_nn_training(config: dict) -> None:
    """Train an MLP mapping per-pixel FC/EI features to tabulated C-factors.

    Follows ``config['stratified_calibration']`` for the target granularity,
    exactly as ``run_calibration`` does.
    """
    stratified = bool(config.get('stratified_calibration', False))

    fc_path                 = 'samples_data_gpr_regular_strat.parquet' #config['gapfilled_fc_path']
    ei_path                 = os.path.expanduser(config['ei_path'])
    c_factor_table_path     = os.path.expanduser(config['c_factor_table_path'])
    lnf_classification_path = os.path.expanduser(config['lnf_classification_path'])
    manual_overrides_path   = config.get('manual_overrides_path')
    if manual_overrides_path:
        manual_overrides_path = os.path.expanduser(manual_overrides_path)
    # NN artefacts go to their own folder, independent of the main pipeline's
    # results_folder, so they don't mix with the β-calibration outputs.
    results_dir             = config.get('nn_results_folder', 'calibration_analysis_nn')
    ts_cols                 = config.get('ts_cols', ['lnf_code', 'yr', 'poly_id'])
    crop_col                = config.get('crop_col', 'lnf_code')
    exclude_lnf_codes       = config.get('exclude_calibration_lnf_codes', []) or []
    area_years              = config.get('area_years', []) or None

    nn_params = {**DEFAULT_NN_PARAMS, **config.get('nn_params', {})}
    test_fraction = float(config.get('nn_test_fraction', 0.2))
    seed          = int(config.get('nn_random_seed', nn_params.get('random_state', 42)))

    os.makedirs(results_dir, exist_ok=True)

    # ---- Load FC ----
    print(f"[nn] Loading gapfilled FC timeseries (stratified={stratified}) ...")
    df_fc = pd.read_parquet(fc_path)

    # ---- Reference table ----
    sampled_lnf_codes = sorted(df_fc[crop_col].unique().tolist())
    if stratified:
        for need in ('uuid', 'betr_ID'):
            if need not in df_fc.columns:
                raise KeyError(
                    f"Gapfilled parquet at {fc_path} has no '{need}' column, "
                    "required for stratified targets. Re-run sample_FC.py."
                )
        df_ref = load_reference_cfactors_stratified(
            c_factor_table_path, lnf_classification_path, sampled_lnf_codes,
            manual_overrides_path=manual_overrides_path, area_years=area_years)
        join_cols = [crop_col, 'region', 'tillage']
    else:
        df_ref = load_reference_cfactors(
            c_factor_table_path, lnf_classification_path, sampled_lnf_codes,
            manual_overrides_path=manual_overrides_path, area_years=area_years)
        join_cols = [crop_col]

    if exclude_lnf_codes:
        before = df_ref[crop_col].nunique()
        df_ref = df_ref[~df_ref[crop_col].isin(exclude_lnf_codes)].reset_index(drop=True)
        print(f"[nn] Excluded {before - df_ref[crop_col].nunique()} crops.")
    df_ref[crop_col] = df_ref[crop_col].astype(df_fc[crop_col].dtype)

    # ---- Strata (stratified only) ----
    if stratified:
        df_fc = assign_strata(
            df_fc,
            nutzung_csv=os.path.expanduser(config['nutzung_csv']),
            reb_csv=os.path.expanduser(config['ressourceneffizienz_csv']),
            cutoff_m=float(config.get('grenze_tal_berg', 600)),
            default_tillage=config.get('standardansaatverfahren', 'Pflug'),
            tillage_method=config.get('tillage_assignment', 'stochastic'),
            seed=int(config.get('tillage_random_seed', 42)),
        )

    # ---- EI snap + join (identical to calibration) ----
    print("[nn] Snapping FC to EI grid and loading EI ...")
    x_off, y_off = get_ei_grid_offset(ei_path)
    x_snap, y_snap = snap_to_ei_grid(df_fc['x'].values, df_fc['y'].values, x_off, y_off)
    df_ei = load_ei_for_pixels(ei_path, np.unique(x_snap), np.unique(y_snap))
    df = join_ei_to_fc(df_fc, df_ei, x_off, y_off)

    # ---- Per-pixel features ----
    print("[nn] Building per-pixel EI-weighted features ...")
    extra = ['region', 'tillage'] if stratified else []
    feats = build_pixel_features(df, ts_cols, extra_cols=extra)
    print(f"[nn] {len(feats)} pixel feature rows from {feats[crop_col].nunique()} crops.")

    # ---- Attach target by broadcasting C_ref onto pixels ----
    feats = feats.merge(df_ref[join_cols + ['C_ref']], on=join_cols, how='inner')
    n_drop = len(build_pixel_features(df, ts_cols, extra_cols=extra)) - len(feats)
    if n_drop:
        print(f"[nn] {n_drop} pixel rows dropped (crop/stratum not in reference "
              "table or excluded).")
    feats = feats.dropna(subset=FEATURE_COLUMNS + ['C_ref']).reset_index(drop=True)
    print(f"[nn] {len(feats)} training rows after target join + NA drop.")

    if len(feats) < 10:
        raise RuntimeError("Too few training rows — check reference matching / inputs.")

    # ---- Train/test split, grouped by pixel-id so the same parcel can't leak ----
    # Group on poly_id (parcel) so a parcel's multi-year pixels stay on one side.
    groups = feats['poly_id'].astype(str).values if 'poly_id' in feats.columns \
        else np.arange(len(feats))
    gss = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    train_idx, test_idx = next(gss.split(feats, groups=groups))
    feats['split'] = 'train'
    feats.loc[feats.index[test_idx], 'split'] = 'test'

    X = feats[FEATURE_COLUMNS].values
    y = feats['C_ref'].values
    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    # ---- Fit pipeline (scaler + MLP) ----
    print(f"[nn] Training MLPRegressor {nn_params['hidden_layer_sizes']} on "
          f"{len(train_idx)} rows ...")
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(**nn_params)),
    ])
    pipe.fit(X_train, y_train)

    # ---- Pixel-level predictions + metrics ----
    feats['C_pred'] = pipe.predict(X)
    pred_train = feats.loc[feats['split'] == 'train', 'C_pred'].values
    pred_test  = feats.loc[feats['split'] == 'test',  'C_pred'].values

    def _metrics(y_true, y_hat):
        return {
            'r2':   float(r2_score(y_true, y_hat)) if len(y_true) > 1 else float('nan'),
            'mae':  float(mean_absolute_error(y_true, y_hat)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_hat))),
            'n':    int(len(y_true)),
        }

    metrics = {
        'stratified': stratified,
        'pixel_level': {
            'train': _metrics(y_train, pred_train),
            'test':  _metrics(y_test,  pred_test),
        },
    }

    # ---- Aggregate pixel predictions to crop (or crop×stratum) ----
    agg = (feats.groupby(join_cols, as_index=False)
                .agg(C_pred_crop=('C_pred', 'mean'),
                     n_pixels=('C_pred', 'size'))
                .merge(df_ref[join_cols + ['C_ref']], on=join_cols, how='left'))
    metrics['crop_level'] = _metrics(agg['C_ref'].values, agg['C_pred_crop'].values)
    print(f"[nn] Pixel test  R²={metrics['pixel_level']['test']['r2']:.3f} "
          f"MAE={metrics['pixel_level']['test']['mae']:.4f}")
    print(f"[nn] Crop  level R²={metrics['crop_level']['r2']:.3f} "
          f"MAE={metrics['crop_level']['mae']:.4f}")

    # ---- Save artefacts ----
    model_path = os.path.join(results_dir, 'nn_model.joblib')
    joblib.dump({'pipeline': pipe,
                 'feature_columns': FEATURE_COLUMNS,
                 'join_cols': join_cols,
                 'stratified': stratified,
                 'nn_params': nn_params}, model_path)
    print(f"[nn] Model → {model_path}")

    feats.to_csv(os.path.join(results_dir, 'nn_training_features.csv'), index=False)
    feats[ts_cols + extra + ['C_ref', 'C_pred', 'split']].to_csv(
        os.path.join(results_dir, 'nn_predictions_per_pixel.csv'), index=False)
    agg.to_csv(os.path.join(results_dir, 'nn_predictions_per_crop.csv'), index=False)

    with open(os.path.join(results_dir, 'nn_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    plot_crop_scatter(agg.dropna(subset=['C_ref', 'C_pred_crop']),
                      os.path.join(results_dir, 'nn_scatter.png'), stratified)
    plot_loss_curve(pipe.named_steps['mlp'],
                    os.path.join(results_dir, 'nn_loss_curve.png'))

    print("[nn] Done.")


if __name__ == '__main__':
    # Standalone use: reuse the same CONFIG as main.py.
    from main import CONFIG
    run_nn_training(CONFIG)