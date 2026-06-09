"""Right-size and model-select the C-factor estimator under grouped CV.

Companion to ``train_cfactor_ml.py``. Where that script *fits* one MLP with a
fixed architecture, this script answers **how big the model should be** and
**whether an MLP is even the right choice**, by measuring generalisation under
cross-validation that is grouped so the model is tested on crops/strata whose
label it has not seen during fitting.

Why grouped CV is the whole point
---------------------------------
Every pixel of a crop carries that crop's tabulated C as its (broadcast)
label. A *random* train/test split therefore puts pixels sharing a label on
both sides, so the model can score well by memorising per-crop means — which
hides over-fitting and makes an over-sized network look fine. Splitting by
group (``cv_group``: 'crop' = lnf_code, or 'stratum' = crop×region×tillage)
holds out *entire* crops/strata, so the CV score measures the thing that
actually matters: does the FC/EI → C mapping generalise to unseen crops?

Consequence: the number of independent units is the number of GROUPS (tens to
low hundreds), not the number of pixel rows. Capacity should be sized against
that. This script makes the train-vs-CV gap visible so you can see where extra
capacity stops buying generalisation and starts buying memorisation.

Mirroring the train split
-------------------------
If ``config['split_strategy'] == 'stratified'`` (the per-stratum parcel split
used by ``train_cfactor_ml``), this script evaluates under the SAME geometry
instead of GroupKFold: parcels (poly_id) are split within each stratum so every
multi-parcel stratum is in both train and test, and single-parcel strata are
pinned to train. ``cv_group`` is ignored in that mode. Note this no longer
holds out whole strata, so the CV score then measures per-stratum *coverage*,
not unseen-stratum generalisation — the train-vs-CV gap reading below does not
apply in that mode.

What it does
------------
1. Assembles the per-pixel feature+target table via
   ``train_cfactor_nn.assemble_training_table`` (identical inputs to training).
2. Baselines under GroupKFold (``baselines=True``):
     - DummyRegressor (predict global mean)   — the floor any model must beat
     - Ridge                                   — linear, the "is the MLP
                                                  buying anything?" reference
     - HistGradientBoostingRegressor           — strong tabular baseline +
                                                  feature importances
3. MLP grid search under GroupKFold over ``hidden_layer_sizes × alpha``
   (``grid`` overridable in config), scaler always prepended.
4. Validation curve: CV vs train score across ``alpha`` for the best
   architecture, so the capacity/regularisation sweet spot is visible.
5. Refits the selected config on all data and reports its grouped-CV score
   next to the baselines.

Scoring
-------
Primary metric is **negative MAE** (higher = better), evaluated at the PIXEL
level by default. Because labels are broadcast, pixel-level CV MAE has an
irreducible floor (within-crop label noise); the script therefore ALSO reports
a **group-mean MAE** per fold — predictions averaged to the group, compared to
the group's C_ref — which is the scientifically meaningful number and directly
comparable to the calibration's per-crop MAE.

Outputs (to ``config['tune_results_folder']``, default ``calibration_analysis_tune/``)
-----------------------------------------------------------------------
- ``tune_cv_results.csv``         full grid-search table (mean/std CV per config)
- ``tune_baselines.csv``          baseline + selected-model CV scores
- ``tune_feature_importance.csv`` permutation/gain importances (tree)
- ``tune_best_config.json``       chosen hyper-parameters + rationale fields
- ``plots/tune_validation_curve_alpha.png``
- ``plots/tune_model_comparison.png``
- ``tune_summary.txt``

This script does NOT overwrite ``nn_model.joblib``. To adopt the chosen config,
copy ``tune_best_config.json``'s ``nn_params`` into your CONFIG and re-run
``train_cfactor_ml.py``.

Entry point: ``run_tuning(config)`` — same CONFIG dict as ``main.py`` plus the
optional ``tune_*`` keys in ``DEFAULT_TUNE_PARAMS``.
"""

from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, cross_validate
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error

from weighted_mlp import WeightedMLPRegressor

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
# Default MLP hyper-parameters — kept here so tune_cfactor_ml.py needs no
# external script imports. Must stay in sync with train_cfactor_ml.DEFAULT_NN_PARAMS.
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


def build_pixel_features(df: pd.DataFrame, ts_cols: list[str],
                         time_col: str = 'time',
                         ei_col: str = 'ei',
                         pv_col: str = 'pv', npv_col: str = 'npv',
                         extra_cols: list[str] | None = None) -> pd.DataFrame:
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
        w = valid.pivot_table(index=ts_cols, columns='_t_idx', values=col, sort=False)
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
    return feats.attrs['pv_cols'] + feats.attrs['npv_cols'] + feats.attrs['ei_cols']


def assemble_training_table(config: dict) -> tuple[pd.DataFrame, list[str], bool]:
    """Build per-pixel feature+target table (no split — callers apply their own)."""
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

    print(f"[tune] Loading gapfilled FC timeseries (stratified={stratified}) ...")
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
        print(f"[tune] Excluded {before - df_ref[crop_col].nunique()} crops.")
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

    print("[tune] Snapping FC to EI grid and loading EI ...")
    x_off, y_off = get_ei_grid_offset(ei_path)
    x_snap, y_snap = snap_to_ei_grid(df_fc['x'].values, df_fc['y'].values, x_off, y_off)
    df_ei = load_ei_for_pixels(ei_path, np.unique(x_snap), np.unique(y_snap))
    df = join_ei_to_fc(df_fc, df_ei, x_off, y_off)

    print("[tune] Building per-pixel features ...")
    extra = ['region', 'tillage'] if stratified else []
    feats = build_pixel_features(df, ts_cols, extra_cols=extra)
    n_feat_rows = len(feats)
    print(f"[tune] {n_feat_rows} pixel feature rows from {feats[crop_col].nunique()} crops.")

    feats_attrs = dict(feats.attrs)
    ref_merge_cols = join_cols + ['C_ref']
    if 'area_ha' in df_ref.columns:
        ref_merge_cols.append('area_ha')
    feats = feats.merge(df_ref[ref_merge_cols], on=join_cols, how='inner')
    feats.attrs.update(feats_attrs)
    n_drop = n_feat_rows - len(feats)
    if n_drop:
        print(f"[tune] {n_drop} pixel rows dropped (crop/stratum not in reference "
              "table or excluded).")
    feature_columns = get_feature_columns(feats)
    attrs = dict(feats.attrs)
    feats = feats.dropna(subset=feature_columns + ['C_ref']).reset_index(drop=True)
    feats.attrs.update(attrs)
    print(f"[tune] {len(feats)} training rows after target join + NA drop.")

    if len(feats) < 10:
        raise RuntimeError("Too few training rows — check reference matching / inputs.")

    return feats, join_cols, stratified


DEFAULT_TUNE_PARAMS = {
    'cv_group':   'stratum',     # 'crop' (lnf_code) or 'stratum' (crop×reg×till)
    'n_splits':   2,
    'random_state': 42,
    'baselines':  True,
    # MLP architectures to compare — kept small on purpose; the effective
    # number of labels is the group count, so big nets only memorise.
    'grid': {
        'mlp__hidden_layer_sizes': [(8,), (16,), (32,), (16, 8), (32, 16)],
        'mlp__alpha':              [1e-3, 1e-2, 1e-1, 1.0],
    },
    # alpha values for the validation curve (best architecture held fixed)
    'alpha_curve': [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _group_vector(feats: pd.DataFrame, cv_group: str, join_cols: list[str],
                  stratified: bool) -> np.ndarray:
    """Build the CV grouping key.

    'crop'    → group by lnf_code (hold out whole crops).
    'stratum' → group by the full join key (hold out whole crop×strata);
                falls back to crop if not stratified.
    """
    if cv_group == 'stratum' and stratified:
        return (feats[join_cols].astype(str)
                     .agg('|'.join, axis=1).values)
    return feats['lnf_code'].astype(str).values


class StratifiedParcelKFold:
    """CV splitter that mirrors ``train_cfactor_ml``'s 'stratified' split.

    Parcels (``poly_id``) are the atomic unit, so no parcel — and therefore no
    pixel — is split across the train/test boundary of a fold. Within each
    stratum (the full ``join_cols`` key) parcels are distributed across folds
    via :class:`sklearn.model_selection.StratifiedGroupKFold`, so every
    multi-parcel stratum sits in BOTH the train and test side of the folds it
    appears in (its label is always seen during fitting). Strata with fewer
    than ``min_parcels`` parcels cannot be held out without vanishing from
    training, so — exactly like the train script — they are pinned to TRAIN in
    every fold and never appear in a test set.

    Consequence: unlike ``GroupKFold`` grouped by crop/stratum, this does NOT
    measure generalisation to UNSEEN strata. It measures the same thing the
    train 'stratified' split measures (per-stratum coverage), so the
    train-vs-CV gap reading elsewhere in this script does not apply here.
    """

    def __init__(self, n_splits, parcels, strata, seed=42, min_parcels=2):
        self.n_splits = int(n_splits)
        self.parcels = np.asarray(parcels)
        self.strata = np.asarray(strata)
        self.seed = int(seed)
        self.min_parcels = int(min_parcels)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X=None, y=None, groups=None):
        n = len(self.parcels)
        all_idx = np.arange(n)
        n_parcels_per_stratum = (pd.DataFrame({'p': self.parcels, 's': self.strata})
                                 .groupby('s')['p'].transform('nunique').values)
        pinned_mask = n_parcels_per_stratum < self.min_parcels   # single-parcel → train-only
        split_idx = all_idx[~pinned_mask]
        pinned_idx = all_idx[pinned_mask]

        sgkf = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True,
                                    random_state=self.seed)
        X_dummy = np.zeros((len(split_idx), 1))
        for tr_rel, te_rel in sgkf.split(X_dummy, self.strata[~pinned_mask],
                                         self.parcels[~pinned_mask]):
            yield (np.concatenate([split_idx[tr_rel], pinned_idx]),
                   split_idx[te_rel])


def _group_mean_mae(estimator, X, y, groups) -> float:
    """MAE after averaging predictions to the group level (the meaningful one)."""
    pred = estimator.predict(X)
    df = pd.DataFrame({'g': groups, 'yhat': pred, 'y': y})
    agg = df.groupby('g').agg(yhat=('yhat', 'mean'), y=('y', 'first'))
    return mean_absolute_error(agg['y'], agg['yhat'])


def compute_sample_weights(feats: pd.DataFrame, join_cols: list[str],
                           crop_col: str = 'lnf_code') -> np.ndarray:
    """Per-pixel sample weights from stratum area (mirrors β calibration).

    See ``train_cfactor_ml.compute_sample_weights`` for full docstring.
    """
    has_area = 'area_ha' in feats.columns and feats['area_ha'].notna().any()
    if has_area:
        crop_area = (feats[[crop_col, 'area_ha']]
                     .drop_duplicates(subset=[crop_col])
                     .set_index(crop_col)['area_ha']
                     .fillna(0.0).clip(lower=0.0))
        n_per_crop = feats.groupby(crop_col)[crop_col].transform('count')
        w = feats[crop_col].map(crop_area).fillna(0.0).values / np.maximum(n_per_crop.values, 1)
        total = w.sum()
        w = w / total if total > 0 else np.ones(len(feats)) / len(feats)
        print(f"[tune] Sample weights: area-weighted, "
              f"{(w > 0).sum()}/{len(w)} pixels with non-zero weight.")
    else:
        group_key = feats[join_cols].astype(str).agg('|'.join, axis=1)
        n_per_group = group_key.groupby(group_key).transform('count')
        n_groups = group_key.nunique()
        w = (1.0 / n_groups) / n_per_group.values
        w = w / w.sum()
        print(f"[tune] Sample weights: equal-group (no area_ha), {n_groups} groups.")
    return w


def _weighted_fit(estimator, X: np.ndarray, y: np.ndarray,
                  sample_weights: np.ndarray | None,
                  idx: np.ndarray | None = None,
                  seed: int = 42) -> None:
    """Fit any estimator with optional sample weighting.

    All estimators in the tuning pipeline (WeightedMLPRegressor, Ridge,
    DummyRegressor, HistGBR) accept ``sample_weight`` natively.
    For Pipelines the weight is routed via ``{last_step}__sample_weight``.
    """
    X_fit = X[idx] if idx is not None else X
    y_fit = y[idx] if idx is not None else y

    if sample_weights is None:
        estimator.fit(X_fit, y_fit)
        return

    w = sample_weights[idx] if idx is not None else sample_weights
    total = w.sum()
    if total > 0:
        w = w / total

    if isinstance(estimator, Pipeline):
        step_name = estimator.steps[-1][0]
        estimator.fit(X_fit, y_fit, **{f'{step_name}__sample_weight': w})
    else:
        estimator.fit(X_fit, y_fit, sample_weight=w)


def _cv_scores(estimator, X, y, groups, cv,
               sample_weights=None, seed=42) -> dict:
    """Pixel-level and group-mean MAE across folds (manual loop, so we can
    compute the group-mean metric the built-in scorers can't express)."""
    pix, grp = [], []
    for tr, te in cv.split(X, y, groups):
        est = estimator
        _weighted_fit(est, X, y, sample_weights, idx=tr, seed=seed)
        pix.append(mean_absolute_error(y[te], est.predict(X[te])))
        grp.append(_group_mean_mae(est, X[te], y[te], groups[te]))
    return {
        'pixel_mae_mean': float(np.mean(pix)), 'pixel_mae_std': float(np.std(pix)),
        'group_mae_mean': float(np.mean(grp)), 'group_mae_std': float(np.std(grp)),
    }


def _make_mlp(hidden, alpha, seed) -> Pipeline:
    params = {**DEFAULT_NN_PARAMS,
              'hidden_layer_sizes': hidden, 'alpha': alpha,
              'random_state': seed}
    return Pipeline([('scaler', StandardScaler()),
                     ('mlp', WeightedMLPRegressor(**params))])


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_validation_curve(alphas, train_mae, cv_mae, cv_std, hidden, out):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(alphas, train_mae, 'o-', label='train MAE', color='#999999')
    ax.plot(alphas, cv_mae, 'o-', label='CV MAE (grouped)', color='#D7263D')
    ax.fill_between(alphas, np.array(cv_mae) - np.array(cv_std),
                    np.array(cv_mae) + np.array(cv_std), color='#D7263D', alpha=0.15)
    ax.set_xscale('log'); ax.set_xlabel('alpha (L2 regularisation)')
    ax.set_ylabel('MAE (pixel-level)')
    ax.set_title(f'Validation curve — MLP {hidden}\n'
                 'gap between train and CV = over-fit; pick alpha near CV min')
    ax.legend(); plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"  saved {out}")


def plot_model_comparison(baseline_df, out):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    d = baseline_df.sort_values('group_mae_mean')
    ax.barh(d['model'], d['group_mae_mean'],
            xerr=d['group_mae_std'], color='#3A86FF', alpha=0.8)
    ax.set_xlabel('Group-mean MAE (grouped CV, lower = better)')
    ax.set_title('Model comparison under grouped CV')
    ax.invert_yaxis(); plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_tuning(config: dict) -> None:
    tp = {**DEFAULT_TUNE_PARAMS, **config.get('tune_params', {})}
    results_dir = config.get('tune_results_folder', 'calibration_analysis_tune')
    plot_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    seed = int(tp['random_state'])

    feats, join_cols, stratified = assemble_training_table(config)
    feature_columns = get_feature_columns(feats)
    X = feats[feature_columns].values
    y = feats['C_ref'].values

    # When train is configured for the per-stratum parcel split, mirror it here
    # so tuning evaluates the same split geometry (rather than GroupKFold, which
    # holds out whole strata). cv_group is ignored in this mode.
    split_strategy = config.get('split_strategy', 'parcel')
    if split_strategy == 'stratified':
        groups = feats[join_cols].astype(str).agg('|'.join, axis=1).values
        n_groups = len(np.unique(groups))
        if 'poly_id' in feats.columns:
            parcels = feats['poly_id'].astype(str).values
        else:
            print("[tune] WARNING: no 'poly_id' column — stratified CV falls "
                  "back to pixel-level splitting (possible label leakage).")
            parcels = np.arange(len(feats)).astype(str)
        # only parcels in multi-parcel strata are test-eligible
        npp = (pd.DataFrame({'p': parcels, 's': groups})
               .groupby('s')['p'].transform('nunique').values)
        n_test_parcels = int(pd.unique(parcels[npp >= 2]).size)
        n_splits = int(min(tp['n_splits'], n_test_parcels))
        if n_splits < 2:
            raise RuntimeError(
                f"Only {n_test_parcels} parcel(s) in multi-parcel strata — "
                "cannot build a stratified parcel CV. Need >=2 strata with "
                ">=2 parcels each, or set split_strategy != 'stratified'.")
        cv = StratifiedParcelKFold(n_splits, parcels, groups, seed=seed)
        print(f"[tune] {len(feats):,} rows, {n_groups} strata; stratified "
              f"parcel {n_splits}-fold CV (mirrors train split_strategy="
              "'stratified').")
        print("[tune] NOTE: strata are NOT held out whole — this CV measures "
              "per-stratum coverage, not unseen-stratum generalisation.")
    else:
        groups = _group_vector(feats, tp['cv_group'], join_cols, stratified)
        n_groups = len(np.unique(groups))
        n_splits = int(min(tp['n_splits'], n_groups))
        if n_splits < 2:
            raise RuntimeError(f"Only {n_groups} CV group(s) for cv_group="
                               f"'{tp['cv_group']}' — cannot cross-validate. "
                               "Use more crops or cv_group='crop'.")
        cv = GroupKFold(n_splits=n_splits)
        print(f"[tune] {len(feats):,} rows, {n_groups} groups "
              f"({tp['cv_group']}), {n_splits}-fold GroupKFold.")
    print(f"[tune] Effective independent labels ≈ {n_groups}; sizing against that.")

    # ---- Sample weights (optional, mirrors β calibration) ----------------
    area_weight = bool(config.get('area_weight_loss', False))
    crop_col = config.get('crop_col', 'lnf_code')
    if area_weight:
        sw = compute_sample_weights(feats, join_cols, crop_col)
    else:
        sw = None
        print("[tune] Sample weights: disabled (area_weight_loss=False).")

    # ---- Baselines -----------------------------------------------------
    rows = []
    if tp['baselines']:
        models = {
            'dummy_mean': DummyRegressor(strategy='mean'),
            'ridge':      Pipeline([('scaler', StandardScaler()),
                                    ('ridge', Ridge(alpha=1.0, random_state=seed))]),
            'hist_gbr':   HistGradientBoostingRegressor(random_state=seed,
                                                        max_iter=300),
        }
        for name, est in models.items():
            s = _cv_scores(est, X, y, groups, cv, sample_weights=sw)
            s['model'] = name
            rows.append(s)
            print(f"[tune] {name:12s} group-MAE={s['group_mae_mean']:.4f}"
                  f" ±{s['group_mae_std']:.4f}  pixel-MAE={s['pixel_mae_mean']:.4f}")

    # ---- MLP grid search ----------------------------------------------
    grid = tp['grid']
    grid_rows = []
    best = None
    for hidden in grid['mlp__hidden_layer_sizes']:
        for alpha in grid['mlp__alpha']:
            est = _make_mlp(hidden, alpha, seed)
            s = _cv_scores(est, X, y, groups, cv, sample_weights=sw)
            rec = {'hidden_layer_sizes': str(hidden), 'alpha': alpha, **s}
            grid_rows.append(rec)
            key = s['group_mae_mean']
            if best is None or key < best['group_mae_mean']:
                best = {'hidden': hidden, 'alpha': alpha, **s}
            print(f"[tune] MLP {str(hidden):8s} alpha={alpha:<6g} "
                  f"group-MAE={s['group_mae_mean']:.4f} ±{s['group_mae_std']:.4f}")
    grid_df = pd.DataFrame(grid_rows).sort_values('group_mae_mean')
    grid_df.to_csv(os.path.join(results_dir, 'tune_cv_results.csv'), index=False)

    # add selected MLP to comparison
    rows.append({'model': f"mlp_{best['hidden']}_a{best['alpha']:g}",
                 **{k: best[k] for k in
                    ('pixel_mae_mean', 'pixel_mae_std',
                     'group_mae_mean', 'group_mae_std')}})
    baseline_df = pd.DataFrame(rows)
    baseline_df.to_csv(os.path.join(results_dir, 'tune_baselines.csv'), index=False)

    # ---- Validation curve over alpha (best architecture) --------------
    train_mae, cv_mae, cv_std = [], [], []
    for alpha in tp['alpha_curve']:
        est = _make_mlp(best['hidden'], alpha, seed)
        # train MAE: fit on all, predict on all (optimistic by design)
        _weighted_fit(est, X, y, sw, seed=seed)
        train_mae.append(mean_absolute_error(y, est.predict(X)))
        s = _cv_scores(_make_mlp(best['hidden'], alpha, seed), X, y, groups, cv,
                        sample_weights=sw, seed=seed)
        cv_mae.append(s['pixel_mae_mean']); cv_std.append(s['pixel_mae_std'])
    plot_validation_curve(tp['alpha_curve'], train_mae, cv_mae, cv_std,
                          best['hidden'],
                          os.path.join(plot_dir, 'tune_validation_curve_alpha.png'))

    # ---- Feature importance (tree, permutation on a grouped holdout) --
    try:
        first_tr, first_te = next(cv.split(X, y, groups))
        gbr = HistGradientBoostingRegressor(random_state=seed, max_iter=300)
        _weighted_fit(gbr, X, y, sw, idx=first_tr, seed=seed)
        pi = permutation_importance(gbr, X[first_te], y[first_te],
                                    n_repeats=10, random_state=seed)
        imp = (pd.DataFrame({'feature': feature_columns,
                             'importance_mean': pi.importances_mean,
                             'importance_std': pi.importances_std})
                 .sort_values('importance_mean', ascending=False))
        imp.to_csv(os.path.join(results_dir, 'tune_feature_importance.csv'),
                   index=False)
        print("[tune] wrote tune_feature_importance.csv")
    except Exception as e:    # importance is diagnostic, never fatal
        print(f"[tune] feature importance skipped: {e}")
        imp = None

    if tp['baselines']:
        plot_model_comparison(baseline_df,
                              os.path.join(plot_dir, 'tune_model_comparison.png'))

    # ---- Best-config JSON + summary -----------------------------------
    if split_strategy == 'stratified':
        cv_label = f"stratified-parcel, {n_splits}-fold (mirrors train split)"
        cv_group_field = 'stratified-parcel'
    else:
        cv_label = f"cv_group={tp['cv_group']}, {n_splits}-fold GroupKFold"
        cv_group_field = tp['cv_group']

    best_params = {**DEFAULT_NN_PARAMS,
                   'hidden_layer_sizes': best['hidden'], 'alpha': best['alpha']}
    ridge_mae = baseline_df.loc[baseline_df['model'] == 'ridge',
                                'group_mae_mean']
    ridge_mae = float(ridge_mae.iloc[0]) if len(ridge_mae) else None
    verdict = None
    if ridge_mae is not None:
        if best['group_mae_mean'] >= ridge_mae - 1e-4:
            verdict = ("Ridge matches or beats the best MLP under grouped CV: "
                       "the MLP's extra capacity buys nothing. Prefer ridge, or "
                       "treat the aggregate-level reformulation as the real fix.")
        else:
            verdict = ("Best MLP beats ridge under grouped CV, so non-linear "
                       "capacity helps — adopt the selected config.")

    json.dump({
        'split_strategy': split_strategy,
        'cv_group': cv_group_field, 'n_groups': int(n_groups),
        'n_splits': n_splits,
        'area_weight_loss': sw is not None,
        'selected_nn_params': {k: (list(v) if isinstance(v, tuple) else v)
                               for k, v in best_params.items()},
        'selected_group_mae': best['group_mae_mean'],
        'selected_pixel_mae': best['pixel_mae_mean'],
        'ridge_group_mae': ridge_mae,
        'verdict': verdict,
    }, open(os.path.join(results_dir, 'tune_best_config.json'), 'w'), indent=2)

    lines = [
        '=' * 70,
        f'NN right-sizing / model selection  ({cv_label})',
        '=' * 70,
        f'Rows                          : {len(feats):>10,}',
        f'Independent groups (≈ labels) : {n_groups:>10}',
        f'Stratified                    : {stratified}',
        '',
        '--- Grouped-CV group-mean MAE (lower = better) ---',
        baseline_df.sort_values('group_mae_mean')[
            ['model', 'group_mae_mean', 'group_mae_std',
             'pixel_mae_mean']].to_string(index=False, float_format='%.4f'),
        '',
        '--- Best MLP architecture ---',
        f"  hidden_layer_sizes : {best['hidden']}",
        f"  alpha              : {best['alpha']}",
        f"  group-mean CV MAE  : {best['group_mae_mean']:.4f} "
        f"± {best['group_mae_std']:.4f}",
    ]
    if imp is not None:
        lines += ['',
                  f'--- Feature importance (permutation, tree) — top 15 of '
                  f'{len(imp)} features (full table in tune_feature_importance.csv) ---',
                  imp.head(15).to_string(index=False, float_format='%.4f')]
    if verdict:
        lines += ['', '--- Verdict ---', verdict]
    lines += ['', '=' * 70, 'How to read this', '=' * 70, """
- The honest number is GROUP-MEAN MAE under grouped CV. It holds out whole
  crops/strata, so it measures generalisation to unseen crops — unlike the
  random holdout in train_cfactor_nn.py, which lets the model see every
  crop's label.

- Compare the best MLP to RIDGE. If they tie, the data has little non-linear
  signal at the group level; a big MLP is wasted capacity. Shrink the model
  (smaller hidden layer / larger alpha) or move to the aggregate-level
  reformulation, which removes the broadcast-label noise entirely.

- The validation curve shows train MAE far below CV MAE at small alpha
  (memorising) converging as alpha grows. Pick alpha near the CV minimum;
  if CV keeps improving up to the largest alpha, the model wants to be even
  simpler than the grid allows — add larger alphas / drop to linear.

- 'Effective independent labels ≈ n_groups' is the real sample size for
  capacity decisions. With tens of groups, single-hidden-layer nets of
  8–32 units are typically the ceiling worth considering.

- To adopt a config: copy selected_nn_params from tune_best_config.json into
  CONFIG['nn_params'] and re-run train_cfactor_ml.py. This script never
  overwrites the fitted model.
"""]
    open(os.path.join(results_dir, 'tune_summary.txt'), 'w').write('\n'.join(lines))
    print(f"[tune] wrote {os.path.join(results_dir, 'tune_summary.txt')}")
    print('\n'.join(lines[:24]))


if __name__ == '__main__':
    # Copy CONFIG from train_cfactor_ml.py (or main.py) and add tune_params if needed.
    from train_cfactor_ml import CONFIG
    run_tuning(CONFIG)