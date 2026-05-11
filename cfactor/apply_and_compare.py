"""End-to-end AOI pipeline: extract FC for every pixel → apply β → compare to R.

Scope
-----
This is the operational sibling of `analyse_calibration_sample.py`. The
calibration is treated as frozen — β is a constant, not refit. For any
chosen AOI (Gemeinden, kantons, a custom polygon, or a CSV of parcels),
the script runs three stages, each cached so re-runs only redo what's
missing.

Difference from `sample_FC.py`
------------------------------
`sample_FC.py` samples ONE pixel per parcel for calibration — that's the
right granularity when fitting β. For comparing the new product to the R
pipeline at parcel / farm / Gemeinde level, sampling is the wrong choice:
the R pipeline produces one C per parcel from a lookup table, so any
within-parcel sampling noise just degrades the comparison.

This script processes EVERY S2 pixel inside every AOI parcel, applies
the GPR gapfilling at the field level (same model as sample_FC, just
predicting for all pixels rather than the single sampled one), then:
  - applies β to every pixel-time series → C per pixel
  - aggregates within parcel → C per parcel (mean)
  - aggregates across parcels → C per farm / Gemeinde (area-weighted)
The within-parcel spread of C is also kept as a diagnostic — that's the
signal the lookup table cannot encode.

STAGE A — FC pipeline (calls into sample_FC.py)
-----------------------------------------------
  A1. Build the AOI samples pickle. Mirrors sample_FC.sample_locations_with_field
      EXCEPT no point sampling: every parcel polygon goes in directly.
      extract_s2_data_field then extracts every S2 pixel in every polygon.
  A2. Extract S2 + soil per parcel polygon  (sample_FC.extract_s2_data_field).
      Note: `is_sample_pixel` is set on the sampled-point pixel, but here
      ALL pixels are processed downstream — the flag is just inherited and
      ignored.
  A3. Predict FC from S2 reflectances        (sample_FC.predict_FC).
  A4. Clean cloud/shadow/snow/cirrus per
      field-date                              (sample_FC.clean_timeseries_field).
  A5. GPR gapfill for ALL pixels per field,
      using the same Matern+White kernel and
      ALR-transformed targets as sample_FC,
      but predicting at fill-dates for every
      pixel rather than the single sampled
      one                                    (this script — gapfill_field_all_pixels).
  Output: a per-pixel parquet with schema —
  [lnf_code, yr, poly_id, x, y, time, pv, npv, soil, fc_total,
   is_gapfilled, fc_quality_flag].

  Stage A can be skipped (CONFIG['skip_sampling'] = True) if you already
  have an AOI per-pixel FC parquet.

STAGE B — Apply β
-----------------
  B1. Compute C per pixel: C_pix = Σ exp(-β·fc_total) · EI / Σ EI
      grouped by (lnf_code, yr, poly_id, x, y).

STAGE C — Aggregate and compare
-------------------------------
  C1. Aggregate C_pix → C_new per parcel (mean across pixels in the
      parcel). Also save C_new_std as a within-parcel-spread diagnostic.
  C2. Attach parcel attributes (Region from elevation, Bodenbearbeitung,
      Gemeinde, betr_ID, Flaeche).
  C3. Look up legacy product on the same parcels:
        C_old_total = per-crop `Total` value (calibration target)
        C_old_mgmt  = per-(crop, Region, Bodenbearbeitung) value
                      from C_Faktoren.csv
  C4. (optional) Compute erosion risk = C × pot_er × P from a potential
      erosion raster.
  C5. (optional) Join an actual R-pipeline export to see the real R
      outputs alongside the lookup-reproduced ones.
  C6. Stratified comparison: per crop, Region (Tal/Berg), tillage, year,
      parcel-size class, FC quality flag, reference-C magnitude. Plus
      farm/Gemeinde area-weighted aggregations and a risk-class confusion
      matrix.

Frozen β — how to obtain it
---------------------------
β is not currently saved by `calibrate_cfactor.py`; it's only printed and
embedded in plot titles. Either:
  - run calibration once and copy the printed β into CONFIG['beta'],  or
  - patch calibrate_cfactor.py to dump β to a small JSON, then load it
    here.

Inputs
------
Stage A (only if not skipped):
  CONFIG['parcels_path']           AOI parcels GeoPackage with at least:
                                     poly_id, lnf_code, betr_ID, geometry,
                                     elevation_m
                                   optional: Bodenbearbeitung, Jahr
  CONFIG['s2_grid_path']           S2 tile grid (same as sample_FC.py)
  CONFIG['s2_dir']                 raw S2 reflectance dir
  CONFIG['soil_dir']               soil-suite predictions dir
  CONFIG['years']                  list of years to sample, e.g. [2022,2023]
                                   (sample_FC.py samples each year separately)
  CONFIG['samples_path']           cache: AOI samples pickle  (Stage A1 out)
  CONFIG['samples_s2_path']        cache: S2-extracted parcel data  (A2 out)
  CONFIG['fc_preds_path']          cache: FC-predicted parcel data  (A3 out)
  CONFIG['fc_parquet_aoi']         cache: gapfilled FC parquet      (A5 out)

Stage B:
  CONFIG['fc_parquet_aoi']         input from A5 (or pre-existing if you
                                   set skip_sampling=True)
  CONFIG['ei_path']                EI parquet (same as calibration)
  CONFIG['beta']                   scalar β from the calibration step

Stage C:
  CONFIG['parcels_path']           AOI parcels GeoPackage (also used here
                                   for attributes; same file as Stage A1)
  CONFIG['gemeinden_path']         Gemeinde polygons with `Gemeinde` column
  CONFIG['c_factor_table_path']    C_Faktoren.csv
  CONFIG['lnf_classification_path']  LNF spreadsheet
  CONFIG['p_factor']               erosion_config.yaml -> P_Faktor (0.8)
  CONFIG['grenze_tal_berg']        erosion_config.yaml -> grenze_tal_berg
  CONFIG['default_tillage']        erosion_config.yaml -> standardansaatverfahren
  CONFIG['pot_erosion_raster']     optional .tif for ER comparison
  CONFIG['r_pipeline_csv']         optional CSV export of alle_schlaege_gemeinden_P

Outputs (in CONFIG['out_dir'])
------------------------------
- comparison_parcel.csv          one row per parcel-year, all C and ER vars
- comparison_crop.csv            crop-mean (C_new vs C_old_total vs C_old_mgmt)
- comparison_farm.csv            area-weighted ER per farm
- comparison_gemeinde.csv        area-weighted ER per Gemeinde
- agreement_summary.csv          bias/MAE/RMSE/Pearson/Spearman at every level
- assessment_by_crop.csv         per-crop strengths/weaknesses metrics
- assessment_by_region.csv       Tal vs Berg
- assessment_by_tillage.csv      per Bodenbearbeitung
- assessment_by_crop_region.csv  per (crop, Region)
- assessment_by_year.csv         interannual stability of the disagreement
- assessment_by_size.csv         per parcel-size class
- assessment_by_quality.csv      per FC quality flag (if available)
- assessment_by_c_magnitude.csv  per reference-C magnitude
- risk_class_confusion.csv       confusion matrix on legacy risk classes
- assessment_report.txt          plain-text strengths/weaknesses summary
- plots/                         all diagnostic plots
"""

from __future__ import annotations

import argparse
import os
import sys
import unicodedata
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.dataset as pads
import pyarrow.compute as pc

try:
    import geopandas as gpd
    HAS_GPD = True
except ImportError:
    HAS_GPD = False

try:
    from rasterstats import zonal_stats
    HAS_RASTERSTATS = True
except ImportError:
    HAS_RASTERSTATS = False

from scipy import stats as scistats

# Stage A reuses the user's own sampling/gapfilling functions verbatim.
# Imported lazily inside _ensure_sample_fc(); we don't want this script to
# fail at import time when the user is in skip_sampling mode and doesn't
# have xarray/sklearn/rioxarray installed locally.
_SAMPLE_FC = None


def _ensure_sample_fc():
    """Import sample_FC on demand. Stage A needs it; Stages B+C don't."""
    global _SAMPLE_FC
    if _SAMPLE_FC is None:
        # Allow CONFIG['sample_fc_dir'] to point to the directory holding
        # sample_FC.py (typically the same directory as calibrate_cfactor.py).
        # Falls back to the directory of this script.
        for d in (os.environ.get('SAMPLE_FC_DIR'),
                  os.path.dirname(os.path.abspath(__file__))):
            if d and os.path.isdir(d) and os.path.exists(os.path.join(d, 'sample_FC.py')):
                if d not in sys.path:
                    sys.path.insert(0, d)
                break
        import sample_FC  # noqa: E402  (lazy import is intentional)
        _SAMPLE_FC = sample_FC
    return _SAMPLE_FC


# ===========================================================================
# Default config — edit paths to your environment.
# ===========================================================================
CONFIG = {
    # ===== Stage A — FC sampling pipeline (skip with skip_sampling=True) =====
    'skip_sampling':            False,
    # AOI parcels GeoPackage (input)
    'parcels_path':             '~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/'
                                'LAI_retrieval_model/data/valsites_CH/Strickhof.gpkg',
    # Years to sample. The AOI sample step calls extract_s2_data_field which
    # internally uses each parcel's `yr` column to select S2 dates. Each
    # parcel ends up with one sample point per year listed here.
    'years':                    [2022, 2023],
    # Same paths sample_FC.py uses
    's2_grid_path':             '~/mnt/eo-nas1/eoa-share/projects/'
                                '012_EO_dataInfrastructure/Project layers/'
                                'gridface_s2tiles_CH.shp',
    's2_dir':                   '~/mnt/eo-nas1/data/satellite/sentinel2/raw/CH',
    'soil_dir':                 '~/mnt/eo-nas1/data/satellite/sentinel2/'
                                'DLR_soilsuite_preds/',
    # Stage A intermediate caches (skipped if file exists)
    'samples_path':             'aoi_samples.pkl',
    'samples_s2_path':          'aoi_samples_data.pkl',
    'fc_preds_path':            'aoi_samples_data_pred.pkl',
    'fc_parquet_aoi':           'aoi_samples_data_gpr.parquet',
    'max_gap_days':             15,
    'drop_fraction_threshold':  0.7,
    'n_jobs':                   1,

    # ===== Stage B — apply β =====
    'ei_path':                  '../erosivity_index/predictions/'
                                'grid_EI_daily_avg_pred_20260424_nn3.parquet',
    'beta':                     0.033,    # MUST be set — copy from calibration

    # ===== Stage C — comparison =====
    'gemeinden_path':           'gemeinden_aoi.gpkg',
    'aoi_filter':               None,    # e.g. ['Regensdorf', 'Eschenbach']
    'tillage_col':              None,    # column in parcels_path holding
                                          # Pflug / Mulch / Direkt, if any
    'c_factor_table_path':      'C_Faktoren.csv',
    'lnf_classification_path':  '~/mnt/eo-nas1/data/landuse/documentation/'
                                'LNF_code_classification_20260217.xlsx',
    # Optional add-ons
    'pot_erosion_raster':       None,
    'r_pipeline_csv':           None,
    'r_pipeline_columns': {
        'parcel_id':     'Flaechen_ID',
        'farm_id':       'betr_ID',
        'year':          'Jahr',
        'area_m2':       'Flaeche',
        'pot_er':        'Pot_Erosionsrisiko_t_ha_y',
        'c_old':         'C_fact_detail',
        'er_old':        'ers_risk_P',
    },
    # Erosion model knobs (mirror erosion_config.yaml)
    'p_factor':            0.8,
    'grenze_tal_berg':     600.0,
    'default_tillage':     'Pflug',

    'out_dir':             'aoi_comparison',
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


# ===========================================================================
# STAGE A — FC sampling pipeline (calls into sample_FC.py)
# ===========================================================================
def build_aoi_samples(parcels_path: str, years: list, save_path: str,
                       target_crs: int = 32632) -> None:
    """Build the samples pickle for an AOI — every parcel, every year, no sampling.

    extract_s2_data_field expects a pickle with one row per (parcel, year)
    holding `polygon_geom` plus a `point_geom`/`sampled_x`/`sampled_y` (used
    only to set the `is_sample_pixel` flag, which we ignore downstream).

    For an all-pixel AOI run we set the "sample point" to each parcel's
    centroid: a real coordinate inside the polygon so the function doesn't
    error, but every pixel returned by extract_s2_data_field gets used by
    later stages regardless of the flag.
    """
    if not HAS_GPD:
        raise ImportError("geopandas required for AOI sample building")

    parcels = gpd.read_file(os.path.expanduser(parcels_path))
    if 'lnf_code' not in parcels.columns:
        raise ValueError("AOI parcels GeoPackage must have an 'lnf_code' column")
    if 'poly_id' not in parcels.columns:
        parcels = parcels.copy()
        parcels['poly_id'] = parcels.index
    parcels = parcels.to_crs(target_crs)

    rows = []
    for yr in years:
        for _, p in parcels.iterrows():
            geom = p.geometry
            if geom is None or geom.is_empty:
                continue
            cx, cy = geom.centroid.x, geom.centroid.y
            rows.append(dict(
                poly_id     = p['poly_id'],
                lnf_code    = p['lnf_code'],
                polygon_geom= geom,
                point_geom  = geom.centroid,
                x           = cx,
                y           = cy,
                yr          = str(int(yr)),
            ))
    samples = gpd.GeoDataFrame(rows, geometry='point_geom', crs=target_crs)
    samples.to_pickle(save_path)
    print(f"  wrote {save_path}  ({len(samples):,} parcel-year polygons; "
          f"all S2 pixels in each will be processed)")


def _gapfill_field_all_pixels(keys, group_clean, ts_cols, value_cols,
                                pixel_cols, max_gap_days=15,
                                max_train_points=1000):
    """All-pixel field-level GPR gapfill.

    Same model as sample_FC._gapfill_one_field_alr — Matern + White on
    [t_norm, sin(doy), cos(doy), x_scaled, y_scaled], ALR-transformed
    (pv, npv, soil) target, kernel fitted once on j=0 and reused for j=1.
    Difference: predicts at fill-dates for EVERY pixel of the field, not
    just a single sampled one.

    Returns clean observations + GPR predictions for every field pixel.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        ConstantKernel, Matern, WhiteKernel)
    from sklearn.preprocessing import StandardScaler

    sf = _ensure_sample_fc()  # for alr_transform / alr_inverse / generate_fill_dates

    if len(group_clean) < 3:
        return None

    # Per-pixel clean observations — these are kept as-is (not gapfilled).
    keep_cols = ts_cols + pixel_cols + ["time"] + value_cols
    result_clean = group_clean[keep_cols].copy()
    result_clean["is_gapfilled"]    = False
    result_clean["fc_quality_flag"] = 0

    # Compute fill dates per pixel (each pixel has its own clean schedule).
    pixel_fill_specs = []
    for px_keys, gpx in group_clean.groupby(pixel_cols):
        px_keys_t = px_keys if isinstance(px_keys, tuple) else (px_keys,)
        clean_t   = pd.DatetimeIndex(pd.to_datetime(gpx["time"])
                                       .sort_values().unique())
        fill_t    = sf.generate_fill_dates(clean_t, max_gap_days)
        if len(fill_t) > 0:
            pixel_fill_specs.append((px_keys_t, fill_t))

    if not pixel_fill_specs:
        return result_clean.sort_values("time").reset_index(drop=True)

    # ---- Feature engineering on all clean field pixels for training ----
    dates_train  = pd.to_datetime(group_clean["time"])
    t0           = dates_train.min()
    t_train      = (dates_train - t0).dt.days.values.reshape(-1, 1).astype(float)

    sin_train    = np.sin(2 * np.pi * (t_train % 365) / 365)
    cos_train    = np.cos(2 * np.pi * (t_train % 365) / 365)
    t_train_norm = t_train / 365.0

    p_scaler     = StandardScaler()
    pixel_train  = p_scaler.fit_transform(group_clean[pixel_cols].values)

    n_pixel_dims = pixel_train.shape[1]
    n_features   = 3 + n_pixel_dims      # t, sin, cos, spatial dims

    X_train_full = np.hstack([t_train_norm, sin_train, cos_train,
                               pixel_train]).astype(np.float32)
    fc_train_full  = group_clean[value_cols].values.astype(np.float32)
    alr_train_full = sf.alr_transform(fc_train_full)

    # Stratified subsample over time if the field has too many clean points.
    # No "always keep target" carve-out — every pixel is equally a target now.
    if len(group_clean) > max_train_points:
        rng     = np.random.default_rng(42)
        n_bins  = min(10, max_train_points)
        bins    = np.linspace(t_train_norm.min(), t_train_norm.max(), n_bins + 1)
        bin_id  = np.digitize(t_train_norm[:, 0], bins) - 1
        per_bin = max(1, max_train_points // n_bins)
        keep    = []
        for b in range(n_bins):
            in_bin = np.where(bin_id == b)[0]
            if len(in_bin) > 0:
                keep.append(rng.choice(in_bin,
                                        size=min(per_bin, len(in_bin)),
                                        replace=False))
        keep      = np.concatenate(keep) if keep else np.arange(len(group_clean))
        X_train   = X_train_full[keep]
        alr_train = alr_train_full[keep]
    else:
        X_train   = X_train_full
        alr_train = alr_train_full

    estimated_alpha = float(np.var(alr_train, axis=0).mean()) * 0.05
    estimated_alpha = float(np.clip(estimated_alpha, 1e-4, 0.5))

    ls_init   = [1.0] * n_features
    ls_bounds = (
        [(0.1,  5.0)] * 1            +   # t
        [(0.5, 10.0)] * 2            +   # sin, cos
        [(0.05, 10.0)] * n_pixel_dims    # spatial
    )
    base_kernel = ConstantKernel(1.0, (0.1, 5.0)) * Matern(
        length_scale=ls_init,
        length_scale_bounds=ls_bounds,
        nu=1.5
    ) + WhiteKernel(noise_level=estimated_alpha,
                    noise_level_bounds=(1e-5, 0.5))

    # ---- Build a single concatenated prediction matrix across all pixels ----
    pred_t_lists, pred_pix_lists, pred_index = [], [], []
    for (px_keys_t, fill_t) in pixel_fill_specs:
        n_fill = len(fill_t)
        # one row per (pixel, fill-date)
        t_fill = np.array([(d - t0).days for d in fill_t],
                          dtype=float).reshape(-1, 1)
        sin_f  = np.sin(2 * np.pi * (t_fill % 365) / 365)
        cos_f  = np.cos(2 * np.pi * (t_fill % 365) / 365)
        t_norm = t_fill / 365.0

        pix_vec = p_scaler.transform(np.array([px_keys_t], dtype=float))
        pix_block = np.repeat(pix_vec, n_fill, axis=0)
        block = np.hstack([t_norm, sin_f, cos_f, pix_block]).astype(np.float32)

        pred_t_lists.append(block)
        pred_index.append((px_keys_t, fill_t))

    X_pred = np.vstack(pred_t_lists)

    alr_preds = np.zeros((len(X_pred), 2), dtype=np.float32)
    alr_unc   = np.zeros((len(X_pred), 2), dtype=np.float32)
    for j in range(2):
        y_train = alr_train[:, j]
        if j == 0:
            gp = GaussianProcessRegressor(kernel=base_kernel,
                                          alpha=estimated_alpha,
                                          normalize_y=True,
                                          n_restarts_optimizer=5)
            gp.fit(X_train, y_train)
            fitted_kernel = gp.kernel_
        else:
            gp = GaussianProcessRegressor(kernel=fitted_kernel,
                                          alpha=estimated_alpha,
                                          normalize_y=True,
                                          n_restarts_optimizer=2)
            gp.fit(X_train, y_train)
        mean, std = gp.predict(X_pred, return_std=True)
        alr_preds[:, j] = mean
        alr_unc[:, j]   = std

    fc_pred = sf.alr_inverse(alr_preds)

    # Quality flag thresholds use the global ALR uncertainty distribution
    total_unc = np.linalg.norm(alr_unc, axis=1)
    p80, p95  = np.percentile(total_unc, 80), np.percentile(total_unc, 95)
    qflag     = np.zeros(len(total_unc), dtype=int)
    qflag[total_unc > p80] = 1
    qflag[total_unc > p95] = 2

    # Reassemble per-pixel fill-point dataframe
    ts_dict   = dict(zip(ts_cols, keys if isinstance(keys, tuple) else (keys,)))
    fill_rows = []
    cursor    = 0
    for (px_keys_t, fill_t) in pred_index:
        n_fill   = len(fill_t)
        sub_pred = fc_pred[cursor:cursor + n_fill]
        sub_qf   = qflag[cursor:cursor + n_fill]
        sub      = pd.DataFrame({
            "time": fill_t.astype("datetime64[ns]"),
            **ts_dict,
            **dict(zip(pixel_cols, px_keys_t)),
        })
        for i, col in enumerate(value_cols):
            sub[col] = sub_pred[:, i]
        sub["is_gapfilled"]    = True
        sub["fc_quality_flag"] = sub_qf
        fill_rows.append(sub)
        cursor += n_fill

    fill_result = pd.concat(fill_rows, ignore_index=True)
    return (pd.concat([result_clean, fill_result], ignore_index=True)
              .sort_values(["time"] + pixel_cols)
              .reset_index(drop=True))


def gapfill_aoi_all_pixels(df_clean: pd.DataFrame, ts_cols: list[str],
                            value_cols: list[str], pixel_cols: list[str],
                            max_gap_days: int = 15, n_jobs: int = 1
                            ) -> pd.DataFrame:
    """Parallel orchestrator over fields. Same shape as
    sample_FC.gapfill_dataframe_gpr_per_field_parallel, but uses
    _gapfill_field_all_pixels which keeps every pixel rather than just the
    sampled one."""
    from joblib import Parallel, delayed

    tasks = list(df_clean.groupby(ts_cols))
    print(f"  GPR gapfilling {len(tasks)} fields (all pixels) "
          f"with {n_jobs} workers")
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_gapfill_field_all_pixels)(
            keys, g, ts_cols, value_cols, pixel_cols, max_gap_days)
        for keys, g in tasks)
    results = [r for r in results if r is not None]
    if not results:
        raise RuntimeError("All fields returned None from GPR gapfill — "
                           "check that clean observations are present.")
    return pd.concat(results, ignore_index=True)


def run_aoi_sampling_pipeline(cfg: dict) -> None:
    """Stage A: build AOI samples → S2 extract → predict FC → clean → GPR fill.

    All pixels (not a single sampled point) are processed end to end. Each
    step is cached on disk; reruns only redo what's missing.
    """
    sf = _ensure_sample_fc()

    # A1 — build AOI samples
    if not os.path.exists(cfg['samples_path']):
        print("[A1] Building AOI samples (one row per parcel-year, "
              "no point sampling)...")
        build_aoi_samples(cfg['parcels_path'], cfg['years'],
                           cfg['samples_path'])
    else:
        print(f"[A1] Skipping — {cfg['samples_path']} exists")

    # A2 — extract S2 + soil per parcel polygon
    if not os.path.exists(cfg['samples_s2_path']):
        print("[A2] Extracting S2 + soil per parcel polygon "
              "(every pixel inside)...")
        sf.extract_s2_data_field(
            cfg['samples_path'],
            os.path.expanduser(cfg['s2_grid_path']),
            os.path.expanduser(cfg['s2_dir']),
            os.path.expanduser(cfg['soil_dir']),
            cfg['samples_s2_path'])
    else:
        print(f"[A2] Skipping — {cfg['samples_s2_path']} exists")

    # A3 — predict FC from S2 reflectances
    if not os.path.exists(cfg['fc_preds_path']):
        print("[A3] Predicting FC...")
        sf.predict_FC(cfg['samples_s2_path'], cfg['fc_preds_path'])
    else:
        print(f"[A3] Skipping — {cfg['fc_preds_path']} exists")

    # A4 + A5 — clean and GPR-gapfill (all pixels, not just sampled one)
    if not os.path.exists(cfg['fc_parquet_aoi']):
        print("[A4] Cleaning per-field timeseries (cloud/shadow/snow/cirrus)...")
        df_samples = pd.read_pickle(cfg['fc_preds_path'])
        ts_cols = ['lnf_code', 'yr', 'poly_id']

        # Apply the field-level cleaner per (lnf_code, yr, poly_id) group.
        # Iterate explicitly so behaviour is invariant to pandas version.
        pieces = []
        for keys, g in df_samples.groupby(ts_cols):
            cleaned = sf.clean_timeseries_field(g)
            if len(cleaned) == 0:
                continue
            cleaned = cleaned.copy()
            keys_t  = keys if isinstance(keys, tuple) else (keys,)
            for col, val in zip(ts_cols, keys_t):
                if col not in cleaned.columns:
                    cleaned[col] = val
            pieces.append(cleaned)
        df_clean = (pd.concat(pieces, ignore_index=True)
                    if pieces else df_samples.iloc[0:0].copy())

        # Drop fields where the date-level mask removed too much
        n_before = df_samples.groupby(ts_cols).size().rename('n_total')
        n_after  = df_clean.groupby(ts_cols).size().rename('n_kept')
        stats_df = pd.concat([n_before, n_after], axis=1).fillna(0)
        stats_df['drop_fraction'] = ((stats_df['n_total'] - stats_df['n_kept'])
                                      / stats_df['n_total'])
        stats_df = stats_df.reset_index()
        thresh = cfg.get('drop_fraction_threshold', 0.7)
        keys_keep = stats_df.loc[stats_df['drop_fraction'] <= thresh, ts_cols]
        df_clean_filtered = df_clean.merge(keys_keep, on=ts_cols, how='inner')
        print(f"  fields retained: {len(keys_keep)}/{len(stats_df)}")

        # Dedupe overlapping satellite pixels at the same (poly_id, x, y, time).
        # Drop is_sample_pixel from the dedupe key — for all-pixel runs the
        # centroid pixel and the regular pixel at the same coords would
        # otherwise be split into two rows.
        dedupe_cols = ['lnf_code', 'yr', 'poly_id', 'x', 'y', 'time']
        df_clean_filtered = (df_clean_filtered
                              .groupby(dedupe_cols, as_index=False)
                              .mean(numeric_only=True))

        print("[A5] GPR gapfilling — all pixels per field...")
        df_gpr = gapfill_aoi_all_pixels(
            df_clean_filtered,
            ts_cols=ts_cols,
            value_cols=['pv', 'npv', 'soil'],
            pixel_cols=['x', 'y'],
            max_gap_days=cfg.get('max_gap_days', 15),
            n_jobs=cfg.get('n_jobs', 1))

        # Restrict to agronomic year window and compute fc_total — same as
        # sample_FC.run_sampling_pipeline. fc_total is on a 0–100 scale to
        # match how β was calibrated.
        df_gpr['time'] = pd.to_datetime(df_gpr['time'])
        s = pd.to_datetime((df_gpr['yr'].astype(int) - 1).astype(str) + '-07-01')
        e = pd.to_datetime(df_gpr['yr'].astype(str) + '-06-30')
        df_gpr = df_gpr[(df_gpr['time'] >= s) & (df_gpr['time'] <= e)]
        df_gpr['fc_total'] = (df_gpr['pv'] + df_gpr['npv']) * 100
        df_gpr.to_parquet(cfg['fc_parquet_aoi'], index=False)
        n_pixyrs = df_gpr.groupby(ts_cols + ['x', 'y']).ngroups
        print(f"  wrote {cfg['fc_parquet_aoi']}  "
              f"({len(df_gpr):,} pixel-time rows; "
              f"{n_pixyrs:,} unique pixel-years across "
              f"{df_gpr.groupby(ts_cols).ngroups} parcel-years)")
    else:
        print(f"[A4-A5] Skipping — {cfg['fc_parquet_aoi']} exists")


# ===========================================================================
# STAGE B — apply β to FC × EI
# ===========================================================================
def get_ei_grid_offset(ei_path: str, resolution: int = 100) -> tuple[float, float]:
    sample = (pads.dataset(ei_path, format='parquet')
              .to_table(columns=['x', 'y']).slice(0, 1).to_pandas())
    return (float(sample['x'].iloc[0]) % resolution,
            float(sample['y'].iloc[0]) % resolution)


def snap_to_ei_grid(x, y, x_off, y_off, res=100):
    return (np.round((x - x_off) / res) * res + x_off,
            np.round((y - y_off) / res) * res + y_off)


def load_ei_for_pixels(ei_path: str, x_snapped, y_snapped) -> pd.DataFrame:
    filt = (pc.field('x').isin(x_snapped.tolist())
            & pc.field('y').isin(y_snapped.tolist()))
    table = (pads.dataset(ei_path, format='parquet')
             .to_table(columns=['x', 'y', 'doy', 'predicted_EI_daily_avg'],
                       filter=filt))
    return table.to_pandas().rename(columns={'predicted_EI_daily_avg': 'ei'})


def apply_beta(fc_path: str, ei_path: str, beta: float
               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the calibrated β to all pixels in the AOI FC parquet.

    Returns
    -------
    c_pixel : DataFrame with one row per (lnf_code, yr, poly_id, x, y)
        Columns: lnf_code, yr, poly_id, x, y, C_pixel, fc_quality_flag
        (fc_quality_flag is the per-pixel max over the time series).
    c_parcel : DataFrame with one row per (lnf_code, yr, poly_id)
        Columns: lnf_code, yr, poly_id, C_new, C_new_std, n_pixels,
                 fc_quality_flag
        C_new is the within-parcel mean of C_pixel; C_new_std is the
        within-parcel spread (sd) — that's the within-class signal the
        legacy lookup table cannot encode.
    """
    print(f"Loading FC for AOI from {fc_path}...")
    df_fc = pd.read_parquet(fc_path)

    pix_keys = ['lnf_code', 'yr', 'poly_id', 'x', 'y']

    # Snap FC pixels to the EI grid; load only the EI cells we need.
    x_off, y_off = get_ei_grid_offset(ei_path)
    print(f"EI grid offset: x={x_off}, y={y_off}")
    x_snap, y_snap = snap_to_ei_grid(df_fc['x'].values, df_fc['y'].values,
                                     x_off, y_off)
    df_ei = load_ei_for_pixels(ei_path, np.unique(x_snap), np.unique(y_snap))
    print(f"  loaded EI: {len(df_ei):,} rows for "
          f"{df_ei[['x','y']].drop_duplicates().shape[0]:,} cells")

    df = df_fc.copy()
    df['x_snap'], df['y_snap'] = x_snap, y_snap
    df['doy'] = pd.to_datetime(df['time']).dt.dayofyear
    df = df.merge(df_ei.rename(columns={'x': 'x_snap', 'y': 'y_snap'}),
                  on=['x_snap', 'y_snap', 'doy'], how='left')

    # ---- C_pixel: one row per pixel-year ----
    valid = df.dropna(subset=['fc_total', 'ei'])
    slr = np.exp(-beta * valid['fc_total'].values)
    grouped = (valid.assign(_num=slr * valid['ei'].values,
                             _den=valid['ei'].values)
               .groupby(pix_keys, as_index=False)
               [['_num', '_den']].sum())
    grouped['C_pixel'] = np.where(grouped['_den'] > 0,
                                   grouped['_num'] / grouped['_den'], np.nan)
    c_pixel = grouped[pix_keys + ['C_pixel']]

    if 'fc_quality_flag' in df_fc.columns:
        qf = (df_fc.groupby(pix_keys, as_index=False)
                   ['fc_quality_flag'].max())
        c_pixel = c_pixel.merge(qf, on=pix_keys, how='left')

    # ---- C_parcel: aggregate pixels within parcel-year ----
    parcel_keys = ['lnf_code', 'yr', 'poly_id']
    agg = (c_pixel.groupby(parcel_keys, as_index=False)
                  .agg(C_new      = ('C_pixel', 'mean'),
                       C_new_std  = ('C_pixel', 'std'),
                       n_pixels   = ('C_pixel', 'size')))
    if 'fc_quality_flag' in c_pixel.columns:
        # Worst quality across pixels (max) is the conservative summary
        agg = agg.merge(c_pixel.groupby(parcel_keys, as_index=False)
                                ['fc_quality_flag'].max(),
                         on=parcel_keys, how='left')
    return c_pixel, agg


# ===========================================================================
# Reference C-table (Tal/Berg × tillage + Total)
# ===========================================================================
def load_c_table(path: str) -> pd.DataFrame:
    """Long form keyed by (Crop_DE, Region, Bodenbearbeitung) with C_old_mgmt
    plus a per-crop C_old_total."""
    df = pd.read_csv(path, sep=';', encoding='cp1252')
    df = df.rename(columns={'Kultur Kategorien 2020': 'Nutzung_DE_KatNutz'})
    long = df.melt(id_vars=['Nutzung_DE_KatNutz', 'Total'],
                   value_vars=['Tal_Pflug', 'Tal_Mulch', 'Tal_Direkt',
                               'Berg_Pflug', 'Berg_Mulch', 'Berg_Direkt'],
                   var_name='region_tillage', value_name='C_old_mgmt')
    long[['Region', 'Bodenbearbeitung']] = (
        long['region_tillage'].str.split('_', expand=True))
    long = long.drop(columns='region_tillage').rename(
        columns={'Total': 'C_old_total'})
    for c in ('C_old_mgmt', 'C_old_total'):
        long[c] = pd.to_numeric(long[c], errors='coerce')
    long['_norm'] = long['Nutzung_DE_KatNutz'].apply(_norm_name)
    return long


def load_lnf_bridge(path: str) -> pd.DataFrame:
    df = pd.read_excel(os.path.expanduser(path), sheet_name='label_sheet')
    bridge = (df[['LNF_code', 'Crop_DE']].dropna().drop_duplicates()
              .rename(columns={'LNF_code': 'lnf_code',
                               'Crop_DE': 'Nutzung_DE_KatNutz'}))
    bridge['_norm'] = bridge['Nutzung_DE_KatNutz'].apply(_norm_name)
    return bridge


# ===========================================================================
# Build the parcel-level comparison table
# ===========================================================================
def assemble_parcel_table(c_new: pd.DataFrame, parcels, gemeinden,
                          c_table: pd.DataFrame, lnf_bridge: pd.DataFrame,
                          tillage_col: Optional[str],
                          grenze_tal_berg: float,
                          default_tillage: str) -> pd.DataFrame:
    """Join C_new with parcel attributes and the C-table. Returns a regular
    DataFrame (no geometry) ready for diagnostics."""
    p = parcels.copy()
    if HAS_GPD and isinstance(p, gpd.GeoDataFrame):
        if 'Gemeinde' not in p.columns and gemeinden is not None:
            p = gpd.sjoin(p, gemeinden[['Gemeinde', 'geometry']],
                          how='left', predicate='intersects')
            p = p.drop(columns=[c for c in ('index_right',) if c in p.columns])
        if 'Flaeche' not in p.columns:
            p['Flaeche'] = p.geometry.area
        p = pd.DataFrame(p.drop(columns='geometry'))

    if 'elevation_m' not in p.columns:
        raise ValueError("AOI parcels must have an 'elevation_m' column "
                         "(mean DEM per parcel).")
    p['Region'] = np.where(p['elevation_m'] > grenze_tal_berg, 'Berg', 'Tal')

    if tillage_col and tillage_col in p.columns:
        p['Bodenbearbeitung'] = p[tillage_col].fillna(default_tillage)
    else:
        p['Bodenbearbeitung'] = default_tillage

    # Bridge lnf_code → Crop_DE/Nutzung_DE_KatNutz
    p = p.merge(lnf_bridge[['lnf_code', 'Nutzung_DE_KatNutz', '_norm']],
                on='lnf_code', how='left')

    # Lookup C_old_mgmt and C_old_total via normalised crop name + Region/tillage
    p = p.merge(c_table[['_norm', 'Region', 'Bodenbearbeitung',
                          'C_old_mgmt', 'C_old_total']],
                on=['_norm', 'Region', 'Bodenbearbeitung'], how='left')
    p = p.drop(columns='_norm')

    # Join calibrated C_new
    out = p.merge(c_new, on=['lnf_code', 'poly_id'], how='inner')

    # If AOI parcels carried Jahr, only keep matching years
    if 'Jahr' in out.columns and 'yr' in out.columns:
        out = out[out['Jahr'] == out['yr']]
    out = out.rename(columns={'yr': 'Jahr'}) if 'yr' in out.columns else out

    n_no_ref = out['C_old_total'].isna().sum()
    if n_no_ref:
        print(f"  WARNING: {n_no_ref} parcel-years have no C_old_total "
              f"(crop not in C_Faktoren.csv) — they keep C_new but C_old "
              f"diagnostics will skip them.")
    return out


# ===========================================================================
# Optional: potential erosion + ER columns
# ===========================================================================
def attach_potential_erosion(df, raster_path):
    if not raster_path:
        return df
    if not HAS_RASTERSTATS:
        print("  rasterstats not installed — skipping ER step "
              "(`pip install rasterstats`).")
        return df
    if 'geometry' not in df.columns:
        print("  Parcels DataFrame lost geometry before raster step — "
              "skipping ER. Pass parcels through with geometry intact.")
        return df
    print(f"Extracting mean potential erosion from {raster_path}...")
    stats = zonal_stats(df['geometry'], raster_path, stats=['mean'])
    df = df.copy()
    df['erskacker22fl1'] = [s['mean'] if s else np.nan for s in stats]
    return df


def compute_er(df: pd.DataFrame, p_factor: float) -> pd.DataFrame:
    if 'erskacker22fl1' not in df.columns:
        return df
    df = df.copy()
    df['ers_risk_new']       = df['C_new']        * df['erskacker22fl1'] * p_factor
    df['ers_risk_old_mgmt']  = df['C_old_mgmt']   * df['erskacker22fl1'] * p_factor
    df['ers_risk_old_total'] = df['C_old_total']  * df['erskacker22fl1'] * p_factor
    return df


def attach_r_export(df: pd.DataFrame, csv_path: str, cols: dict) -> pd.DataFrame:
    """Optional: attach actual R-pipeline outputs for a direct R-vs-S2 view.

    The lookup-driven C_old_total/C_old_mgmt should match these closely; any
    drift exposes implementation differences (e.g. R area-weighting nuances,
    parcel polygon edits)."""
    print(f"Joining R-pipeline export {csv_path}...")
    r = pd.read_csv(csv_path)
    inv = {v: k for k, v in cols.items()}
    miss = set(cols.values()) - set(r.columns)
    if miss:
        print(f"  WARNING: R export missing columns {miss} — skipping join.")
        return df
    r = r.rename(columns=inv).rename(columns={'parcel_id':  'poly_id',
                                              'year':       'Jahr',
                                              'area_m2':    'Flaeche_R',
                                              'pot_er':     'pot_er_R',
                                              'c_old':      'C_old_R',
                                              'er_old':     'ers_risk_old_R'})
    keep = ['poly_id', 'Jahr', 'C_old_R', 'ers_risk_old_R',
            'pot_er_R', 'Flaeche_R']
    r['poly_id'] = r['poly_id'].astype(str)
    df['poly_id'] = df['poly_id'].astype(str)
    out = df.merge(r[keep], on=['poly_id', 'Jahr'], how='left')
    return out


# ===========================================================================
# Stratified diagnostics
# ===========================================================================
def stratum_metrics(g: pd.DataFrame, ref: str, new: str,
                    weight: str = 'Flaeche') -> dict:
    sub = g.dropna(subset=[ref, new])
    n = len(sub)
    if n == 0:
        return dict(n=0, area_ha=0.0, mean_old=np.nan, mean_new=np.nan,
                    bias=np.nan, rel_bias_pct=np.nan, MAE=np.nan,
                    RMSE=np.nan, spread_C_new=np.nan, spearman_rho=np.nan)
    diff = sub[new].values - sub[ref].values
    if sub[new].nunique() > 1 and sub[ref].nunique() > 1:
        rho = float(scistats.spearmanr(sub[ref], sub[new]).statistic)
    else:
        rho = np.nan
    mo = float(sub[ref].mean())
    return dict(
        n=int(n),
        area_ha=float(sub[weight].sum() / 1e4) if weight in sub.columns else np.nan,
        mean_old=mo,
        mean_new=float(sub[new].mean()),
        bias=float(diff.mean()),
        rel_bias_pct=float(100 * diff.mean() / mo) if mo > 0 else np.nan,
        MAE=float(np.abs(diff).mean()),
        RMSE=float(np.sqrt((diff ** 2).mean())),
        spread_C_new=float(sub[new].std()) if n > 1 else 0.0,
        spearman_rho=rho,
    )


def stratify(parcel: pd.DataFrame, by_cols: list[str],
             ref: str = 'C_old_total', new: str = 'C_new'
             ) -> pd.DataFrame:
    rows = []
    for keys, g in parcel.groupby(by_cols, dropna=False):
        keys = (keys,) if not isinstance(keys, tuple) else keys
        row = dict(zip(by_cols, keys))
        row.update(stratum_metrics(g, ref, new))
        rows.append(row)
    return (pd.DataFrame(rows)
              .sort_values('MAE', ascending=False, na_position='last')
              .reset_index(drop=True))


def build_assessments(parcel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """All stratified diagnostics in one pass."""
    out = {'by_crop':         stratify(parcel, ['Nutzung_DE_KatNutz']),
           'by_region':       stratify(parcel, ['Region']),
           'by_tillage':      stratify(parcel, ['Bodenbearbeitung']),
           'by_crop_region':  stratify(parcel, ['Nutzung_DE_KatNutz', 'Region'])}
    if 'Jahr' in parcel.columns:
        out['by_year']        = stratify(parcel, ['Jahr'])
        out['by_crop_year']   = stratify(parcel, ['Nutzung_DE_KatNutz', 'Jahr'])
    if 'Flaeche' in parcel.columns:
        size_bins = [0, 5_000, 10_000, 30_000, 100_000, np.inf]
        size_labels = ['<0.5ha', '0.5–1ha', '1–3ha', '3–10ha', '>10ha']
        parcel = parcel.copy()
        parcel['size_class'] = pd.cut(parcel['Flaeche'], bins=size_bins,
                                       labels=size_labels, include_lowest=True)
        out['by_size']        = stratify(parcel, ['size_class'])
    if 'fc_quality_flag' in parcel.columns:
        out['by_quality']     = stratify(parcel, ['fc_quality_flag'])
    bins = [-0.001, 0.01, 0.05, 0.10, 0.20, 1.0]
    labels = ['very_low (≤0.01)', 'low (0.01–0.05)', 'medium (0.05–0.10)',
              'high (0.10–0.20)', 'very_high (>0.20)']
    parcel = parcel.copy()
    parcel['C_old_class'] = pd.cut(parcel['C_old_total'], bins=bins,
                                    labels=labels)
    out['by_c_magnitude']     = stratify(parcel, ['C_old_class'])
    return out


def risk_class_confusion(parcel: pd.DataFrame,
                          new_col: str = 'ers_risk_new',
                          old_col: str = 'ers_risk_old_total'
                          ) -> Optional[pd.DataFrame]:
    """Confusion matrix on legacy risk classes (breaks from f_create_ers_plot.R:
    0.25, 2, 8, 20 t/ha/yr). Tells you whether the new product reclassifies
    parcels — the operationally relevant question."""
    if new_col not in parcel or old_col not in parcel:
        return None
    sub = parcel.dropna(subset=[new_col, old_col])
    if sub.empty:
        return None
    bins   = [-0.001, 0.25, 2, 8, 20, np.inf]
    labels = ['low', 'medium', 'high', 'very_high', 'extreme']
    return pd.crosstab(pd.cut(sub[old_col], bins=bins, labels=labels),
                       pd.cut(sub[new_col], bins=bins, labels=labels),
                       rownames=['old (legacy)'],
                       colnames=['new (S2)'], dropna=False)


# ===========================================================================
# Aggregations
# ===========================================================================
def _wmean(values, weights):
    v, w = np.asarray(values, float), np.asarray(weights, float)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    return float(np.average(v[m], weights=w[m])) if m.any() else np.nan


def aggregate(df: pd.DataFrame, by: list[str], cols: list[str],
              weight: str = 'Flaeche') -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(by, dropna=False):
        keys = (keys,) if not isinstance(keys, tuple) else keys
        row = dict(zip(by, keys))
        for c in cols:
            if c in g.columns:
                row[c] = _wmean(g[c], g[weight])
        row['Flaeche_ha']     = g[weight].sum() / 1e4 if weight in g else np.nan
        row['n_parcel_years'] = len(g)
        rows.append(row)
    return pd.DataFrame(rows)


# ===========================================================================
# Plots
# ===========================================================================
def _scatter(ax, x, y, hue=None, hue_label='', max_legend=20):
    if hue is None:
        ax.scatter(x, y, s=10, alpha=0.35, edgecolor='none', color='steelblue')
    else:
        cmap = plt.get_cmap('tab20')
        cats = pd.Series(hue).astype(str).fillna('NA')
        for i, c in enumerate(cats.unique()[:max_legend]):
            m = (cats == c).values
            ax.scatter(x[m], y[m], s=10, alpha=0.45, edgecolor='none',
                       color=cmap(i % 20), label=c[:20])
        ax.legend(fontsize=6, loc='upper left', ncol=2, framealpha=0.7)


def plot_parcel_scatter(parcel, save):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, ref, label in zip(axes,
                               ['C_old_total', 'C_old_mgmt'],
                               ['Total (per crop)',
                                'Management-resolved (Region × Tillage)']):
        sub = parcel.dropna(subset=[ref, 'C_new'])
        if sub.empty:
            continue
        _scatter(ax, sub[ref].values, sub['C_new'].values,
                 hue=sub['Nutzung_DE_KatNutz'].values)
        m = max(sub[ref].max(), sub['C_new'].max()) * 1.05
        ax.plot([0, m], [0, m], 'k--', lw=1, alpha=0.6, label='1:1')
        b = (sub['C_new'] - sub[ref]).mean()
        rmse = float(np.sqrt(((sub['C_new'] - sub[ref])**2).mean()))
        ax.set_xlabel(f'C_old ({ref})')
        ax.set_ylabel('C_new (S2)')
        ax.set_title(f'{label}\nn={len(sub):,}  bias={b:+.4f}  RMSE={rmse:.4f}')
        ax.set_xlim(0, m); ax.set_ylim(0, m)
    plt.tight_layout(); plt.savefig(save, dpi=150); plt.close()
    print(f"  saved {save}")


def plot_box_per_crop(parcel, save, min_n: int = 10):
    counts = parcel.groupby('Nutzung_DE_KatNutz').size()
    keep = counts[counts >= min_n].index
    sub = parcel[parcel['Nutzung_DE_KatNutz'].isin(keep)].copy()
    if sub.empty:
        return
    order = (sub.groupby('Nutzung_DE_KatNutz')['C_new']
                .median().sort_values().index.tolist())
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(order)), 6))
    data = [sub.loc[sub['Nutzung_DE_KatNutz'] == c, 'C_new'].values
            for c in order]
    ax.boxplot(data, labels=[c[:25] for c in order], showfliers=False,
               medianprops=dict(color='steelblue', lw=1.5))
    for i, c in enumerate(order, start=1):
        c_old = sub.loc[sub['Nutzung_DE_KatNutz'] == c, 'C_old_total'].iloc[0]
        ax.scatter(i, c_old, color='red', s=40, zorder=5,
                   label='C_old_total' if i == 1 else None)
    ax.set_ylabel('C-factor')
    ax.set_title('Per-crop C-factor on AOI: S2 distribution vs legacy table')
    plt.xticks(rotation=70, ha='right', fontsize=8)
    ax.legend(loc='upper left')
    plt.tight_layout(); plt.savefig(save, dpi=150); plt.close()
    print(f"  saved {save}")


def plot_strengths_weaknesses(by_crop, save, min_n=10):
    df = by_crop[by_crop['n'] >= min_n].dropna(subset=['bias',
                                                       'spread_C_new']).copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 7))
    sizes = 30 + 6 * np.sqrt(df['n'].clip(upper=10000))
    sc = ax.scatter(df['bias'], df['spread_C_new'], s=sizes,
                    c=df['bias'].abs(), cmap='Reds',
                    alpha=0.8, edgecolor='k', linewidth=0.5)
    ax.axvline(0, color='k', lw=0.7, alpha=0.6)
    for _, r in df.iterrows():
        ax.annotate(str(r['Nutzung_DE_KatNutz'])[:22],
                    (r['bias'], r['spread_C_new']),
                    fontsize=7, alpha=0.85,
                    xytext=(4, 3), textcoords='offset points')
    ax.set_xlabel('bias (C_new − C_old_total)  →  S2 systematically higher')
    ax.set_ylabel('within-crop spread of C_new (σ)  →  more parcel-level signal')
    ax.set_title('Per-crop strengths & weaknesses on AOI')
    plt.colorbar(sc, ax=ax, label='|bias|')
    plt.tight_layout(); plt.savefig(save, dpi=150); plt.close()
    print(f"  saved {save}")


def plot_heatmap(by_crop_region, save, metric='bias', min_n=5):
    df = by_crop_region[by_crop_region['n'] >= min_n]
    if df.empty:
        return
    pivot = df.pivot_table(index='Nutzung_DE_KatNutz', columns='Region',
                           values=metric, aggfunc='first')
    if pivot.empty:
        return
    pivot = pivot.reindex(pivot.abs().mean(axis=1).sort_values(ascending=False).index)
    fig, ax = plt.subplots(figsize=(6, max(4, 0.3 * len(pivot))))
    vmax = float(np.nanmax(np.abs(pivot.values)))
    im = ax.imshow(pivot.values, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   aspect='auto')
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([s[:30] for s in pivot.index], fontsize=8)
    ax.set_title(f'{metric} of (C_new − C_old_total)  by crop × Region')
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f'{v:+.3f}', ha='center', va='center',
                        fontsize=7,
                        color='white' if abs(v) > 0.6 * vmax else 'black')
    plt.colorbar(im, ax=ax, label=metric)
    plt.tight_layout(); plt.savefig(save, dpi=150); plt.close()
    print(f"  saved {save}")


def plot_size_quality(strata, save):
    panels = [('by_size',         'size_class',      'Parcel size'),
              ('by_quality',      'fc_quality_flag', 'S2 quality flag'),
              ('by_c_magnitude',  'C_old_class',     'Reference C magnitude')]
    panels = [p for p in panels if p[0] in strata and not strata[p[0]].empty]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.5))
    if len(panels) == 1:
        axes = [axes]
    for ax, (k, x_col, title) in zip(axes, panels):
        df = strata[k].copy().sort_values(x_col)
        x = df[x_col].astype(str).values
        ax.bar(x, df['MAE'].values, color='steelblue', alpha=0.7, label='MAE')
        ax.plot(x, df['bias'].values, 'o-', color='firebrick', lw=1.5,
                label='bias')
        ax.axhline(0, color='k', lw=0.6)
        ax.set_title(title); ax.set_ylabel('C-factor units')
        ax.tick_params(axis='x', rotation=30)
        ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(save, dpi=150); plt.close()
    print(f"  saved {save}")


def plot_aggregation_scatter(farm_df, gem_df, save,
                             new='C_new', old='C_old_total'):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, df, title in zip(axes, (farm_df, gem_df), ('Farm', 'Gemeinde')):
        sub = df.dropna(subset=[new, old])
        if sub.empty:
            ax.set_visible(False); continue
        rho = scistats.spearmanr(sub[old], sub[new]).statistic
        ax.scatter(sub[old], sub[new], s=22, alpha=0.6, edgecolor='none')
        m = max(sub[old].max(), sub[new].max()) * 1.05
        ax.plot([0, m], [0, m], 'k--', lw=1, alpha=0.5, label='1:1')
        ax.set_xlabel(f'{old} (legacy)'); ax.set_ylabel(f'{new} (S2)')
        ax.set_title(f'{title} aggregation — n={len(sub)}, '
                     f'Spearman ρ={rho:.3f}')
        ax.legend()
    plt.tight_layout(); plt.savefig(save, dpi=150); plt.close()
    print(f"  saved {save}")


def plot_risk_confusion(cm, save):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(cm.values, cmap='Blues')
    ax.set_xticks(range(len(cm.columns))); ax.set_xticklabels(cm.columns)
    ax.set_yticks(range(len(cm.index)));   ax.set_yticklabels(cm.index)
    ax.set_xlabel(cm.columns.name); ax.set_ylabel(cm.index.name)
    ax.set_title('Risk-class agreement\n(diagonal = same class)')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm.values[i, j]), ha='center', va='center',
                    color='white' if cm.values[i, j] > cm.values.max() / 2
                    else 'black')
    plt.colorbar(im, ax=ax, label='parcel-years')
    plt.tight_layout(); plt.savefig(save, dpi=150); plt.close()
    print(f"  saved {save}")


# ===========================================================================
# Report
# ===========================================================================
def write_report(strata: dict, cm, parcel: pd.DataFrame, save: str,
                 min_n_crop: int = 20) -> None:
    lines = [
        '=' * 75,
        'AOI strengths & weaknesses assessment of the new C-factor product',
        '=' * 75, '',
        f'Total parcel-years   : {len(parcel):,}',
        f'Total area           : {parcel["Flaeche"].sum() / 1e4:,.0f} ha'
            if 'Flaeche' in parcel else '',
        f'Years covered        : {sorted(parcel["Jahr"].dropna().unique().tolist())}'
            if 'Jahr' in parcel else '',
        f'Crops covered        : {parcel["Nutzung_DE_KatNutz"].nunique()}',
        f'Gemeinden covered    : {parcel["Gemeinde"].nunique()}'
            if 'Gemeinde' in parcel else '',
        '']
    bc = strata.get('by_crop')
    if bc is not None and not bc.empty:
        bc = bc[bc['n'] >= min_n_crop]
        if not bc.empty:
            cols = ['Nutzung_DE_KatNutz', 'n', 'mean_old', 'mean_new',
                    'bias', 'MAE', 'spread_C_new']
            lines += [f'--- Best-agreeing crops (lowest MAE, n≥{min_n_crop}) ---',
                      bc.nsmallest(5, 'MAE')[cols].to_string(index=False,
                                                              float_format='%.4f'),
                      '',
                      f'--- Worst-agreeing crops (highest MAE, n≥{min_n_crop}) ---',
                      bc.nlargest(5, 'MAE')[cols].to_string(index=False,
                                                             float_format='%.4f'), '']
    for key, label in [('by_region',        'Tal vs Berg'),
                       ('by_tillage',       'Tillage'),
                       ('by_year',          'Per year'),
                       ('by_size',          'Parcel size'),
                       ('by_quality',       'S2 quality flag'),
                       ('by_c_magnitude',   'Reference C magnitude')]:
        d = strata.get(key)
        if d is not None and not d.empty:
            lines += [f'--- {label} ---',
                      d.to_string(index=False, float_format='%.4f'), '']
    if cm is not None:
        diag = np.trace(cm.values); tot = cm.values.sum()
        lines += ['--- Risk-class confusion matrix (parcel-years) ---',
                  cm.to_string(),
                  f'\nAgreement on risk class: {diag}/{tot} = {100 * diag / tot:.1f}%',
                  '']
    lines += ['=' * 75, 'How to read this', '=' * 75, """
- Compare against C_old_total to gauge calibration generalisation; against
  C_old_mgmt to see if S2 picks up the management/elevation effect the
  table encodes through tillage class.

- Look at by_crop_region heatmaps: if Berg cells consistently have a
  different sign of bias than Tal cells, β alone can't capture the
  elevation effect — you'd need a regional β.

- Risk-class confusion matrix: even if the absolute ER differs, the
  operational question is whether parcels move risk classes. Most should
  stay on the diagonal; off-diagonal moves are what matters for the
  published indicator.

- High MAE concentrated in small parcels or high quality flag = sampling
  noise, not a product weakness — it should disappear if the AOI runs use
  multiple pixels per parcel rather than one.
"""]
    Path(save).write_text('\n'.join(lines), encoding='utf-8')
    print(f"  wrote {save}")


# ===========================================================================
# Driver
# ===========================================================================
def run(cfg: dict) -> None:
    if cfg.get('beta') is None:
        raise ValueError("CONFIG['beta'] must be set to the calibrated β. "
                         "Run calibrate_cfactor.py and copy the printed value.")

    out_dir  = Path(cfg['out_dir']);    out_dir.mkdir(exist_ok=True, parents=True)
    plot_dir = out_dir / 'plots';       plot_dir.mkdir(exist_ok=True)

    # ---- Stage A: full FC sampling pipeline (skippable) ----
    if cfg.get('skip_sampling'):
        print("=" * 60)
        print(f"STAGE A: skipped (skip_sampling=True). "
              f"Using existing FC parquet: {cfg['fc_parquet_aoi']}")
        print("=" * 60)
        if not os.path.exists(cfg['fc_parquet_aoi']):
            raise FileNotFoundError(
                f"skip_sampling=True but {cfg['fc_parquet_aoi']} doesn't "
                f"exist. Either set skip_sampling=False to run the sampling "
                f"pipeline, or supply an existing FC parquet at this path.")
    else:
        print("=" * 60)
        print("STAGE A: Sample FC for AOI parcels")
        print("=" * 60)
        run_aoi_sampling_pipeline(cfg)

    # ---- Stage B: apply β ----
    print("\n" + "=" * 60)
    print(f"STAGE B: Apply β = {cfg['beta']} to AOI FC parquet")
    print("=" * 60)
    c_pixel, c_new = apply_beta(cfg['fc_parquet_aoi'],
                                 os.path.expanduser(cfg['ei_path']),
                                 beta=float(cfg['beta']))
    c_pixel.to_parquet(out_dir / 'C_per_pixel.parquet', index=False)
    c_new.to_csv(out_dir / 'C_per_parcel.csv', index=False)
    print(f"  computed C for {len(c_pixel):,} pixel-years; "
          f"aggregated to {len(c_new):,} parcel-years")
    print(f"  wrote C_per_pixel.parquet and C_per_parcel.csv")

    # ---- Stage C: comparison ----
    print("\n" + "=" * 60)
    print("STAGE C.1: Load AOI parcel geometries")
    print("=" * 60)
    if not HAS_GPD:
        raise ImportError("geopandas required to load parcel/Gemeinde geometries")
    parcels   = gpd.read_file(os.path.expanduser(cfg['parcels_path']))
    gemeinden = gpd.read_file(os.path.expanduser(cfg['gemeinden_path']))
    if cfg.get('aoi_filter'):
        gemeinden = gemeinden[gemeinden['Gemeinde'].isin(cfg['aoi_filter'])]
        parcels   = gpd.overlay(parcels, gemeinden[['geometry']],
                                how='intersection')
    print(f"  parcels: {len(parcels):,}  Gemeinden: {len(gemeinden)}")

    print("\n" + "=" * 60)
    print("STAGE C.2: Assemble parcel-level table with C_new and C_old lookups")
    print("=" * 60)
    c_table    = load_c_table(os.path.expanduser(cfg['c_factor_table_path']))
    lnf_bridge = load_lnf_bridge(cfg['lnf_classification_path'])
    parcel = assemble_parcel_table(c_new, parcels, gemeinden, c_table,
                                   lnf_bridge,
                                   tillage_col=cfg.get('tillage_col'),
                                   grenze_tal_berg=cfg['grenze_tal_berg'],
                                   default_tillage=cfg['default_tillage'])
    print(f"  parcel-level table: {len(parcel):,} parcel-years")

    # Optional: ER + R-export joins
    parcel = compute_er(attach_potential_erosion(parcel,
                                                 cfg.get('pot_erosion_raster')),
                        cfg['p_factor'])
    if cfg.get('r_pipeline_csv'):
        parcel = attach_r_export(parcel, cfg['r_pipeline_csv'],
                                  cfg['r_pipeline_columns'])

    # Save parcel table
    parcel.to_csv(out_dir / 'comparison_parcel.csv', index=False)
    print(f"  wrote comparison_parcel.csv")

    # Stage C.3 — aggregations
    agg_vals = [c for c in ['C_new', 'C_old_total', 'C_old_mgmt', 'C_old_R',
                             'ers_risk_new', 'ers_risk_old_mgmt',
                             'ers_risk_old_total', 'ers_risk_old_R']
                if c in parcel.columns]
    crop_df = aggregate(parcel, ['Nutzung_DE_KatNutz'], agg_vals)
    farm_df = aggregate(parcel, ['betr_ID'],            agg_vals)
    gem_df  = aggregate(parcel, ['Gemeinde'],           agg_vals)
    crop_df.to_csv(out_dir / 'comparison_crop.csv',     index=False)
    farm_df.to_csv(out_dir / 'comparison_farm.csv',     index=False)
    gem_df.to_csv (out_dir / 'comparison_gemeinde.csv', index=False)
    print(f"  wrote crop ({len(crop_df)}), farm ({len(farm_df)}), "
          f"Gemeinde ({len(gem_df)}) tables")

    # Stage C.4 — stratified diagnostics
    print("\n" + "=" * 60)
    print("STAGE C.4: Stratified diagnostics")
    print("=" * 60)
    strata = build_assessments(parcel)
    for name, df in strata.items():
        df.to_csv(out_dir / f'assessment_{name}.csv', index=False)
        print(f"  wrote assessment_{name}.csv  ({len(df)} rows)")
    cm = risk_class_confusion(parcel)
    if cm is not None:
        cm.to_csv(out_dir / 'risk_class_confusion.csv')

    # Agreement summary at the four levels
    rows = [dict(level='parcel_C',
                 **stratum_metrics(parcel, 'C_old_total', 'C_new'))]
    if 'ers_risk_new' in parcel:
        rows.append(dict(level='parcel_ER',
                         **stratum_metrics(parcel, 'ers_risk_old_total',
                                            'ers_risk_new')))
    rows.append(dict(level='crop_C',
                     **stratum_metrics(crop_df, 'C_old_total', 'C_new')))
    if 'ers_risk_new' in farm_df:
        rows.append(dict(level='farm_ER',
                         **stratum_metrics(farm_df, 'ers_risk_old_total',
                                            'ers_risk_new', 'Flaeche_ha')))
        rows.append(dict(level='gemeinde_ER',
                         **stratum_metrics(gem_df, 'ers_risk_old_total',
                                            'ers_risk_new', 'Flaeche_ha')))
    pd.DataFrame(rows).to_csv(out_dir / 'agreement_summary.csv', index=False)

    # Stage C.5 — plots
    print("\n" + "=" * 60)
    print("STAGE C.5: Plots")
    print("=" * 60)
    plot_parcel_scatter        (parcel, plot_dir / 'parcel_scatter.png')
    plot_box_per_crop          (parcel, plot_dir / 'box_per_crop.png')
    plot_strengths_weaknesses  (strata['by_crop'],
                                plot_dir / 'strengths_weaknesses.png')
    if 'by_crop_region' in strata:
        plot_heatmap(strata['by_crop_region'],
                     plot_dir / 'bias_heatmap_crop_region.png', metric='bias')
        plot_heatmap(strata['by_crop_region'],
                     plot_dir / 'mae_heatmap_crop_region.png',  metric='MAE')
    plot_size_quality          (strata, plot_dir / 'size_quality_magnitude.png')
    if 'C_new' in farm_df.columns and 'C_old_total' in farm_df.columns:
        plot_aggregation_scatter(farm_df, gem_df,
                                  plot_dir / 'aggregation_C.png')
    if 'ers_risk_new' in farm_df.columns:
        plot_aggregation_scatter(farm_df, gem_df,
                                  plot_dir / 'aggregation_ER.png',
                                  new='ers_risk_new',
                                  old='ers_risk_old_total')
    if cm is not None:
        plot_risk_confusion(cm, plot_dir / 'risk_class_confusion.png')

    # Stage C.6 — report
    write_report(strata, cm, parcel, out_dir / 'assessment_report.txt')

    # Headline
    print("\n=== Headline ===")
    df = parcel.dropna(subset=['C_new', 'C_old_total'])
    print(f"Parcel-level C  bias vs C_old_total: "
          f"{(df['C_new'] - df['C_old_total']).mean():+.4f}  "
          f"(MAE {(df['C_new'] - df['C_old_total']).abs().mean():.4f}, "
          f"n={len(df):,})")
    if cm is not None:
        diag = np.trace(cm.values); tot = cm.values.sum()
        print(f"Risk-class agreement: {diag}/{tot} = {100 * diag / tot:.1f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Apply the C-factor pipeline to an AOI and compare to '
                    'the legacy R-pipeline product.')
    parser.add_argument('--skip-sampling', action='store_true',
                        help='Skip Stage A. Requires CONFIG[fc_parquet_aoi] '
                             'to already exist.')
    parser.add_argument('--beta', type=float, default=None,
                        help='Override CONFIG[beta] from the command line.')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Override CONFIG[out_dir].')
    args = parser.parse_args()

    cfg = dict(CONFIG)
    if args.skip_sampling:
        cfg['skip_sampling'] = True
    if args.beta is not None:
        cfg['beta'] = args.beta
    if args.out_dir is not None:
        cfg['out_dir'] = args.out_dir
    run(cfg)
