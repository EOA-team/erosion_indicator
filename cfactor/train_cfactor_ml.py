"""Train an ML C-factor model (MLP or Ridge), standalone.

Self-contained alternative to ``train_cfactor_nn.py`` that:
  - holds its own CONFIG at the top of the file (no import from ``main.py``);
  - dispatches on ``CONFIG['model_kind']``: ``'mlp'`` or ``'ridge'``;
  - reuses the existing data plumbing (``assemble_training_table``,
    ``get_feature_columns``) from ``train_cfactor_nn.py`` so inputs are
    identical to what the NN trainer sees;
  - writes outputs to ``calibration_analysis_{model_kind}/`` by default
    (configurable), with the **same filenames** the analyser already reads —
    so ``analyse_nn_sample.py`` works unchanged after one path change.

Why ridge is worth running alongside the MLP
--------------------------------------------
With broadcast labels (every pixel of a crop carries that crop's tabulated C),
the number of independent learnable things is roughly the number of groups,
not the number of pixels. ``tune_cfactor_nn.py`` often returns a verdict that
ridge ties or beats the best MLP under grouped CV — meaning the data has
little non-linear signal at the group level. When that happens ridge is the
honest deployment choice: smaller, faster, fully interpretable (one
coefficient per timestep-channel), and not subject to MLP-style over-fit.

Tuned-alpha lookup
------------------
If ``CONFIG['tune_best_config_path']`` (default
``calibration_analysis_nn/tune_best_config.json``) exists, ``alpha`` is read
from its ``selected_nn_params`` block — but **only blindly for MLP**. For
ridge the same value is used as a sensible starting point with a printed
warning that it was selected for an MLP, not for ridge: their optimal
regularisation strengths can differ by orders of magnitude. If the JSON is
missing or the user sets ``model_params['alpha']`` explicitly in CONFIG, the
JSON is ignored.

Outputs (to ``CONFIG['results_folder']``)
-----------------------------------------
Same filenames as ``train_cfactor_nn.py`` so the analyser is plug-compatible:
- ``nn_model.joblib``                fitted pipeline + metadata (model_kind, ...)
- ``nn_training_features.csv``       per-pixel features + target + split flag
- ``nn_predictions_per_pixel.csv``   per-pixel prediction (all rows)
- ``nn_predictions_per_crop.csv``    pixel predictions averaged to group
- ``nn_metrics.json``                train/test MAE/RMSE/R² (pixel + group)
- ``nn_scatter.png``                 predicted vs reference at group level
- ``nn_loss_curve.png``              MLP only
- ``nn_ridge_coefficients.png``      ridge only — coefficient magnitudes per
                                     timestep, faceted by channel (PV/NPV/EI)

To analyse: copy ``analyse_nn_sample.py``'s CONFIG and change
``per_pixel_path`` / ``per_crop_path`` / ``metrics_json_path`` /
``results_dir`` to point at ``calibration_analysis_ridge/`` (or whatever
model folder you used). Filenames are identical.

Entry point: ``run_ml_training(CONFIG)``.
"""

from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
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
# Default MLP hyper-parameters (override via config['nn_params'] / model_params)
# ---------------------------------------------------------------------------
DEFAULT_NN_PARAMS = {
    'hidden_layer_sizes': (32,),
    'activation':         'relu',
    'solver':             'adam',
    'alpha':              1,
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
                         time_col: str = 'time',
                         ei_col: str = 'ei',
                         pv_col: str = 'pv', npv_col: str = 'npv',
                         extra_cols: list[str] | None = None) -> pd.DataFrame:
    """Pivot the gap-filled time series to one wide feature row per pixel.

    Each timestep contributes three columns (pv_t000, npv_t000, ei_t000, ...).
    Requires a regular-grid gap-fill (all series same length); fails loudly
    if lengths differ.
    """
    extra_cols = extra_cols or []
    needed = ts_cols + [time_col, ei_col, pv_col, npv_col] + extra_cols
    valid = df[needed].dropna(subset=[ei_col, pv_col, npv_col])
    valid = valid.sort_values(ts_cols + [time_col]).reset_index(drop=True)
    valid['_t_idx'] = valid.groupby(ts_cols, sort=False).cumcount()

    lengths = valid.groupby(ts_cols, sort=False)['_t_idx'].size()
    n_steps_per_group = lengths.unique()
    if len(n_steps_per_group) != 1:
        lo, hi = int(lengths.min()), int(lengths.max())
        n_bad = int((lengths != lengths.mode().iloc[0]).sum())
        raise ValueError(
            f"Pixel time series have mismatched lengths after NA drop: "
            f"min={lo}, max={hi}, {n_bad}/{len(lengths)} groups differ from "
            "the modal length. Raw-grid features require the regular-grid "
            "gap-fill (sample_FC.py with gapfill_method='regular')."
        )
    n_steps = int(n_steps_per_group[0])

    def _pivot(col, prefix):
        w = (valid.pivot_table(index=ts_cols, columns='_t_idx',
                               values=col, sort=False))
        w.columns = [f'{prefix}_t{int(c):03d}' for c in w.columns]
        return w.reset_index()

    feats = _pivot(pv_col, 'pv')
    feats = feats.merge(_pivot(npv_col, 'npv'), on=ts_cols)
    feats = feats.merge(_pivot(ei_col,  'ei'),  on=ts_cols)

    if extra_cols:
        extras = valid[ts_cols + extra_cols].drop_duplicates(subset=ts_cols)
        feats = feats.merge(extras, on=ts_cols, how='left')

    feats.attrs['n_steps']  = n_steps
    feats.attrs['pv_cols']  = [f'pv_t{i:03d}'  for i in range(n_steps)]
    feats.attrs['npv_cols'] = [f'npv_t{i:03d}' for i in range(n_steps)]
    feats.attrs['ei_cols']  = [f'ei_t{i:03d}'  for i in range(n_steps)]
    return feats


def get_feature_columns(feats: pd.DataFrame) -> list[str]:
    """Return ordered per-timestep feature column names from feats.attrs."""
    return feats.attrs['pv_cols'] + feats.attrs['npv_cols'] + feats.attrs['ei_cols']


# ---------------------------------------------------------------------------
# Data assembly: shared between training and (if imported) tuning
# ---------------------------------------------------------------------------
def assemble_training_table(config: dict) -> tuple[pd.DataFrame, list[str], bool]:
    """Build the per-pixel feature+target table.

    Returns (feats, join_cols, stratified). No train/test split; callers apply
    their own scheme (holdout for training, GroupKFold for tuning).
    """
    stratified = bool(config.get('stratified_calibration', False))

    fc_path                 = config['gapfilled_fc_path']
    ei_path                 = os.path.expanduser(config['ei_path'])
    c_factor_table_path     = os.path.expanduser(config['c_factor_table_path'])
    lnf_classification_path = os.path.expanduser(config['lnf_classification_path'])
    manual_overrides_path   = config.get('manual_overrides_path')
    if manual_overrides_path:
        manual_overrides_path = os.path.expanduser(manual_overrides_path)
    ts_cols           = config.get('ts_cols', ['lnf_code', 'yr', 'poly_id'])
    crop_col          = config.get('crop_col', 'lnf_code')
    exclude_lnf_codes = config.get('exclude_calibration_lnf_codes', []) or []
    area_years        = config.get('area_years', []) or None

    print(f"[ml] Loading gapfilled FC timeseries (stratified={stratified}) ...")
    df_fc = pd.read_parquet(fc_path)

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
        print(f"[ml] Excluded {before - df_ref[crop_col].nunique()} crops.")
    df_ref[crop_col] = df_ref[crop_col].astype(df_fc[crop_col].dtype)

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

    print("[ml] Snapping FC to EI grid and loading EI ...")
    x_off, y_off = get_ei_grid_offset(ei_path)
    x_snap, y_snap = snap_to_ei_grid(df_fc['x'].values, df_fc['y'].values, x_off, y_off)
    df_ei = load_ei_for_pixels(ei_path, np.unique(x_snap), np.unique(y_snap))
    df = join_ei_to_fc(df_fc, df_ei, x_off, y_off)

    print("[ml] Building per-pixel features ...")
    extra = ['region', 'tillage'] if stratified else []
    feats = build_pixel_features(df, ts_cols, extra_cols=extra)
    n_feat_rows = len(feats)
    print(f"[ml] {n_feat_rows} pixel feature rows from {feats[crop_col].nunique()} crops.")

    feats_attrs = dict(feats.attrs)
    feats = feats.merge(df_ref[join_cols + ['C_ref']], on=join_cols, how='inner')
    feats.attrs.update(feats_attrs)
    n_drop = n_feat_rows - len(feats)
    if n_drop:
        print(f"[ml] {n_drop} pixel rows dropped (crop/stratum not in reference "
              "table or excluded).")
    feature_columns = get_feature_columns(feats)
    attrs = dict(feats.attrs)
    feats = feats.dropna(subset=feature_columns + ['C_ref']).reset_index(drop=True)
    feats.attrs.update(attrs)
    print(f"[ml] {len(feats)} training rows after target join + NA drop.")

    if len(feats) < 10:
        raise RuntimeError("Too few training rows — check reference matching / inputs.")

    return feats, join_cols, stratified


# ---------------------------------------------------------------------------
# Plots (shared between training and ridge diagnostics)
# ---------------------------------------------------------------------------
def plot_crop_scatter(df_crop: pd.DataFrame, path: str, stratified: bool) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df_crop['C_ref'], df_crop['C_pred_crop'], alpha=0.7)
    lims = [0, max(df_crop['C_ref'].max(), df_crop['C_pred_crop'].max()) * 1.05]
    ax.plot(lims, lims, 'k--', lw=1, label='1:1')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Reference C-factor')
    ax.set_ylabel('Predicted C-factor (crop mean)')
    title = 'C-factor: predicted vs reference'
    ax.set_title(title + (' (per crop×stratum)' if stratified else ' (per crop)'))
    ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_loss_curve(mlp, path: str) -> None:
    if not hasattr(mlp, 'loss_curve_'):
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(mlp.loss_curve_)
    ax.set_xlabel('Iteration'); ax.set_ylabel('Training loss')
    ax.set_title('MLP training loss curve')
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ===========================================================================
# CONFIG — self-contained. Edit data paths to match your setup.
# (The data-path keys mirror what main.py's CONFIG uses; copy from there once.)
# ===========================================================================
CONFIG: dict = {
    # ---- Model selection ----
    'model_kind':            'mlp',     # 'mlp' or 'ridge'

    # Where the tuner's JSON lives. If present, its alpha is used.
    'tune_best_config_path': 'calibration_analysis_tune/tune_best_config.json',

    # Per-model overrides; if alpha is set here, the JSON is ignored.
    'model_params': {
        # 'alpha': 1.0,                   # uncomment to override the JSON
    },

    # ---- Output ----
    # If the path contains '{model}', it is filled with model_kind.
    'results_folder':        'calibration_analysis_{model}',

    # ---- Train/test split ----
    'test_fraction':         0.2,
    'random_seed':           42,

    # ====================================================================
    # Data paths and pipeline knobs (copy values from your main.py CONFIG)
    # ====================================================================
    'stratified_calibration': True,
    'gapfilled_fc_path':      'samples_data_gpr.parquet',
    'ei_path':                '../erosivity_index/predictions/grid_EI_daily_avg_pred_20260424_nn3.parquet',
    'c_factor_table_path':    '~/mnt/Data-Labo-RE/27_Natural_Resources-RE/321.4_WAUM_protected/Daten/Erosionsrisiko/C_Faktoren.csv',
    'lnf_classification_path':'~/mnt/eo-nas1/data/landuse/documentation/LNF_code_classification_20260217.xlsx',
    'manual_overrides_path':  None,
    'ts_cols':                ['lnf_code', 'yr', 'poly_id'],
    'crop_col':               'lnf_code',
    'exclude_calibration_lnf_codes': [601, 602],
    'area_years':             None,

    # Stratified-only knobs
    'nutzung_csv':            '~/mnt/Data-Labo-RE/27_Natural_Resources-RE/321.4_WAUM_protected/Daten/Core_Snapshot/Agrarbericht_2025/tbl_nutzungsdaten.csv',
    'ressourceneffizienz_csv':'~/mnt/Data-Labo-RE/27_Natural_Resources-RE/321.4_WAUM_protected/Daten/Core_Snapshot/Agrarbericht_2025/tbl_ressourceneffizienzbeitrag.csv',
    'grenze_tal_berg':        600,
    'standardansaatverfahren':'Pflug',
    'tillage_assignment':     'stochastic',
    'tillage_random_seed':    42,
}


# ===========================================================================
# Model factory
# ===========================================================================
def _resolve_alpha(model_kind: str, model_params: dict,
                   json_path: str) -> tuple[float | None, str]:
    """Return (alpha, source_msg). User override > tuner JSON > model default.

    For ridge, the JSON's alpha is used but with a printed warning since the
    tuner selected it against an MLP loss surface — different scales of L2
    penalty are typical between the two.
    """
    if 'alpha' in model_params:
        return float(model_params['alpha']), 'user override (model_params)'
    if json_path and os.path.exists(json_path):
        try:
            with open(json_path) as f:
                j = json.load(f)
            a = j.get('selected_nn_params', {}).get('alpha')
            if a is not None:
                msg = f'tune_best_config.json ({json_path})'
                if model_kind == 'ridge':
                    print(f"[ml] NOTE: applying tuned alpha={a} from {json_path}, "
                          "but it was selected for the MLP. Ridge typically wants "
                          "a different scale of regularisation — treat this as a "
                          "starting point and re-tune if predictions look poor.")
                return float(a), msg
        except (json.JSONDecodeError, KeyError, OSError) as e:
            print(f"[ml] WARN: could not read alpha from {json_path}: {e}")
    return None, 'model default'


def build_estimator(model_kind: str, model_params: dict, seed: int,
                    json_path: str) -> tuple[Pipeline, dict, str]:
    """Build the (scaler, model) pipeline. Returns (pipe, effective_params,
    alpha_source) so downstream logging/joblib can record what was used."""
    alpha, alpha_src = _resolve_alpha(model_kind, model_params, json_path)

    if model_kind == 'mlp':
        params = {**DEFAULT_NN_PARAMS, **model_params}
        if alpha is not None:
            params['alpha'] = alpha
        params.setdefault('random_state', seed)
        pipe = Pipeline([('scaler', StandardScaler()),
                         ('mlp', MLPRegressor(**params))])
        return pipe, params, alpha_src

    if model_kind == 'ridge':
        # Ridge default — explicit so the joblib record is complete.
        defaults = {'alpha': 1.0, 'random_state': seed, 'solver': 'auto'}
        params = {**defaults, **model_params}
        if alpha is not None:
            params['alpha'] = alpha
        pipe = Pipeline([('scaler', StandardScaler()),
                         ('ridge', Ridge(**params))])
        return pipe, params, alpha_src

    raise ValueError(f"Unknown model_kind={model_kind!r}; expected 'mlp' or 'ridge'.")


# ===========================================================================
# Ridge-specific diagnostic plot
# ===========================================================================
def plot_ridge_coefficients(pipe: Pipeline, feature_columns: list[str],
                            n_steps: int, path: str) -> None:
    """Bar plots of ridge coefficient magnitudes by timestep, one panel per
    channel (PV/NPV/EI). Direct readout of which timesteps the model relies
    on, in the same scaled units the model was fit in (so magnitudes are
    comparable across channels).

    A flat coefficient profile suggests the model couldn't isolate
    erosion-relevant timesteps; a profile peaked in spring/summer suggests
    it locked onto the agronomically meaningful window.
    """
    coef = pipe.named_steps['ridge'].coef_
    # Map back to (channel, timestep) layout — assumes the column order from
    # build_pixel_features: pv_t000..pv_tN, npv_t000..., ei_t000...
    by_channel = {'pv': [], 'npv': [], 'ei': []}
    for col, c in zip(feature_columns, coef):
        for ch in by_channel:
            if col.startswith(f'{ch}_t'):
                by_channel[ch].append(c)
                break

    fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
    colors = {'pv': '#2E8B57', 'npv': '#B8860B', 'ei': '#4169E1'}
    x = np.arange(n_steps)
    for ax, (ch, vals) in zip(axes, by_channel.items()):
        if not vals:    # channel absent
            ax.set_visible(False); continue
        ax.bar(x, vals, color=colors[ch], alpha=0.85)
        ax.axhline(0, color='k', lw=0.5)
        ax.set_ylabel(f'{ch.upper()} coef')
    axes[-1].set_xlabel('Timestep index (chronological, Jul-1 yr-1 → Jun-30 yr)')
    fig.suptitle('Ridge coefficients per timestep (scaled features)')
    fig.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)


# ===========================================================================
# Runner
# ===========================================================================
def run_ml_training(config: dict) -> None:
    model_kind = config['model_kind'].lower()
    if model_kind not in ('mlp', 'ridge'):
        raise ValueError(f"model_kind must be 'mlp' or 'ridge', got {model_kind!r}.")

    results_dir = config.get('results_folder', 'calibration_analysis_{model}')
    results_dir = results_dir.replace('{model}', model_kind)
    os.makedirs(results_dir, exist_ok=True)

    test_fraction = float(config.get('test_fraction', 0.2))
    seed          = int(config.get('random_seed', 42))
    json_path     = config.get('tune_best_config_path',
                               'calibration_analysis_nn/tune_best_config.json')
    model_params  = config.get('model_params', {})

    # ---- Data ---------------------------------------------------------
    feats, join_cols, stratified = assemble_training_table(config)
    feature_columns = get_feature_columns(feats)
    n_steps = int(feats.attrs.get('n_steps', 0))
    print(f"[ml] model_kind={model_kind}  using {len(feature_columns)} features "
          f"({n_steps} timesteps × channels).")

    # ---- Train/test split, grouped by poly_id -------------------------
    groups = feats['poly_id'].astype(str).values if 'poly_id' in feats.columns \
        else np.arange(len(feats))
    gss = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    train_idx, test_idx = next(gss.split(feats, groups=groups))
    feats['split'] = 'train'
    feats.loc[feats.index[test_idx], 'split'] = 'test'

    X = feats[feature_columns].values
    y = feats['C_ref'].values
    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    # ---- Build + fit --------------------------------------------------
    pipe, effective_params, alpha_src = build_estimator(
        model_kind, model_params, seed, json_path)
    print(f"[ml] alpha={effective_params.get('alpha', 'n/a')}  "
          f"(source: {alpha_src})")
    print(f"[ml] Training on {len(train_idx)} rows ...")
    pipe.fit(X_train, y_train)

    # ---- Predictions + metrics ----------------------------------------
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
        'model_kind': model_kind,
        'alpha_source': alpha_src,
        'effective_params': {k: (list(v) if isinstance(v, tuple) else v)
                             for k, v in effective_params.items()},
        'stratified': stratified,
        'pixel_level': {
            'train': _metrics(y_train, pred_train),
            'test':  _metrics(y_test,  pred_test),
        },
    }

    # ---- Aggregate pixel predictions to group --------------------------
    agg = (feats.groupby(join_cols, as_index=False)
                .agg(C_pred_crop=('C_pred', 'mean'),
                     n_pixels=('C_pred', 'size'),
                     C_ref=('C_ref', 'first')))
    metrics['crop_level'] = _metrics(agg['C_ref'].values, agg['C_pred_crop'].values)
    print(f"[ml] Pixel test  R²={metrics['pixel_level']['test']['r2']:.3f} "
          f"MAE={metrics['pixel_level']['test']['mae']:.4f}")
    print(f"[ml] Group level R²={metrics['crop_level']['r2']:.3f} "
          f"MAE={metrics['crop_level']['mae']:.4f}")

    # ---- Save artefacts -----------------------------------------------
    model_path = os.path.join(results_dir, 'nn_model.joblib')
    joblib.dump({'pipeline': pipe,
                 'model_kind': model_kind,
                 'feature_columns': feature_columns,
                 'join_cols': join_cols,
                 'stratified': stratified,
                 'effective_params': effective_params,
                 'alpha_source': alpha_src,
                 'n_steps': n_steps}, model_path)
    print(f"[ml] Model → {model_path}")

    feats.to_csv(os.path.join(results_dir, 'nn_training_features.csv'), index=False)
    extra = ['region', 'tillage'] if stratified else []
    ts_cols = config.get('ts_cols', ['lnf_code', 'yr', 'poly_id'])
    feats[ts_cols + extra + ['C_ref', 'C_pred', 'split']].to_csv(
        os.path.join(results_dir, 'nn_predictions_per_pixel.csv'), index=False)
    agg.to_csv(os.path.join(results_dir, 'nn_predictions_per_crop.csv'), index=False)

    with open(os.path.join(results_dir, 'nn_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    plot_crop_scatter(agg.dropna(subset=['C_ref', 'C_pred_crop']),
                      os.path.join(results_dir, 'nn_scatter.png'), stratified)

    # Model-specific diagnostic
    if model_kind == 'mlp':
        plot_loss_curve(pipe.named_steps['mlp'],
                        os.path.join(results_dir, 'nn_loss_curve.png'))
    elif model_kind == 'ridge':
        plot_ridge_coefficients(
            pipe, feature_columns, n_steps,
            os.path.join(results_dir, 'nn_ridge_coefficients.png'))

    print(f"[ml] Done. To analyse, point analyse_ml_sample.py's CONFIG "
          f"paths at '{results_dir}/'.")


if __name__ == '__main__':
    run_ml_training(CONFIG)
