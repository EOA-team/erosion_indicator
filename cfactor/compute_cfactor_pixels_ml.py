"""
Per-pixel annual C-factor inference for a region — ML-model variant.

Companion to ``compute_cfactor_pixels.py`` (the empirical β path). It produces
the *same outputs* (hive-partitioned Parquet, optional per-field summary,
optional GeoTIFF) but replaces the physical SLR reduction
(``C = Σ exp(-β·FC)·EI / Σ EI``) with the trained ML pipeline from
``train_cfactor_ml.py`` (``nn_model.joblib``: MLP or Ridge).

Why the pipeline is almost identical
-------------------------------------
Stages A (clean observations) and A½ (FC prediction with the SALI ensemble) are
*unchanged* — they are imported directly from ``compute_cfactor_pixels`` so the
two products share exactly the same input path. Only Stage B differs:

  empirical  Stage B:  GP gap-fill → SLR = exp(-β·FC) → C = Σ SLR·EI / Σ EI
  ML         Stage B:  GP gap-fill → build the per-pixel wide feature vector
                       [pv_t000.., npv_t000.., ei_t000..] (== build_pixel_features
                       in train_cfactor_ml) → C = pipeline.predict(features)

The ML model consumes *only* the regular-grid PV/NPV/EI time series. The
stratification (region, tillage) used during training defines the *target*, not
the input features, so prediction needs no strata, no soil_group for the C step
(soil_group is still used by Stage A½ for FC), and no β.

EI join — IMPORTANT
-------------------
``train_cfactor_ml`` joins EI by snapping the *raw* pixel x/y to the EI grid
(``join_ei_to_fc``, NO half-pixel shift). ``compute_cfactor_pixels`` applies a
``x-5, y+5`` shift. To feed the model the same EI values it learned from, this
script defaults to NO shift (``ei_half_pixel_shift=False``). Set it to True only
if your gap-filled training parquet was built with shifted coordinates.

Output: hive-partitioned Parquet  <output_dir>/year=YYYY/lnf_code=NNN/part-*.parquet
        columns: poly_id, lnf_code, x, y, year, c_factor,
                 n_clean_obs_field, n_grid_pts
        + optional per-field summary parquet and per-tile GeoTIFF (same as the
          empirical script).

Usage:
    python compute_cfactor_pixels_ml.py --region region.shp --years 2022
    python compute_cfactor_pixels_ml.py --region region.gpkg --years 2021 2022 \
        --model calibration_analysis_mlp/nn_model.joblib --gpu --geotiff
"""

import argparse
import os
import sys
import tempfile
import time as _time
import warnings

# Limit BLAS threads BEFORE importing numpy/scipy/sklearn (process-pool friendly).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

# Reuse the empirical script's machinery so the input path is identical.
import compute_cfactor_pixels as ccp
from compute_cfactor_pixels import (
    TILE_SIZE_M,
    PIX_RES,
    VALUE_COLS,
    list_agyear_tiles,
    process_tile,
    predict_fc,
    write_geotiff,
)
from calibrate_cfactor import (
    get_ei_grid_offset,
    load_ei_for_pixels,
    snap_to_ei_grid,
)

_T0 = _time.time()


def _log(msg):
    el = _time.time() - _T0
    print(f"[{int(el // 60):02d}:{el % 60:05.2f}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Configuration  (data paths mirror compute_cfactor_pixels.CONFIG)
# ---------------------------------------------------------------------------

CONFIG = {
    # ---- ML model (replaces beta_path) ----
    "model_path":   "calibration_analysis_mlp_strat_noley/nn_model.joblib",
    "clip_cfactor": True,            # clip predictions to [0, 1]

    "region_path":  None,            # set via --region; any OGR vector
    "s2_dir":       "~/mnt/eo-nas1/data/satellite/sentinel2/raw/CH",
    "soil_dir":     "~/mnt/eo-nas1/data/satellite/sentinel2/DLR_soilsuite_preds/",
    "lnf_dir":      "~/mnt/eo-nas1/data/landuse/raw",
    "ei_path":      "../erosivity_index/predictions/grid_EI_daily_avg_pred_20260424_nn3.parquet",
    "output_dir":   os.path.join(os.path.dirname(__file__), "output", "cfactor_pixels_ml"),
    "years":        [2022],
    "lnf_codes":    None,
    "lnf_ignore":   [],
    "n_workers_io": 6,
    "n_workers_gp": 6,

    # ---- FC prediction (SALI NN) — same as the empirical script ----
    "sali_src_path": "~/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure/SALI_models",
    "fc_model_dir":  "~/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure/SALI_models/FC/models/",
    "s2_bands":      ["s2_B02", "s2_B03", "s2_B04", "s2_B05", "s2_B06",
                      "s2_B07", "s2_B08", "s2_B8A", "s2_B11", "s2_B12"],
    "use_gpu":       False,
    "fc_chunk":      50000,

    "soil_name_template": "SRC_{minx}_{maxy}.zarr",

    # ---- Cleaning ----
    "clean_mode":      "per_pixel",
    "cirrus_thresh":   500,
    "max_missing_frac": 0.05,

    # ---- GP gap-fill (must match training: grid_step_days defines n_steps) ----
    "grid_step_days":   10,
    "max_train_points": 1000,
    "min_clean_obs":    3,
    "pred_chunk":       20000,

    # ---- EI join (training-faithful default) ----
    # False: snap raw x/y to EI grid (== train_cfactor_ml.join_ei_to_fc).
    # True : apply x-5, y+5 before snapping (== compute_cfactor_pixels).
    "ei_half_pixel_shift": True,

    # ---- Output ----
    "write_geotiff":       False,
    "write_field_summary": True,
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: str):
    """Load the joblib bundle written by train_cfactor_ml._save_outputs.

    Returns (pipeline, feature_columns, n_steps, model_kind).
    """
    # Ensure the MLP wrapper class is importable before unpickling.
    try:
        import weighted_mlp  # noqa: F401
    except Exception:
        pass
    bundle = joblib.load(os.path.expanduser(model_path))
    pipe = bundle["pipeline"]
    feature_columns = bundle["feature_columns"]
    n_steps = int(bundle["n_steps"])
    model_kind = bundle.get("model_kind", "?")
    return pipe, feature_columns, n_steps, model_kind


# ---------------------------------------------------------------------------
# Stage B (ML): per-field GP gap-fill -> wide features -> pipeline.predict
# ---------------------------------------------------------------------------

_ML = {}   # worker globals (model + EI + config)


def _b_init(model_path, grid_start_iso, grid_step, max_train, pred_chunk,
            min_clean, ei_parquet, x_off, y_off, ei_shift, clip_cfactor,
            feature_columns, n_steps):
    # _gp_predict_grid (imported from ccp) reads ccp._B for grid params.
    ccp._B["grid_start"] = pd.Timestamp(grid_start_iso)
    ccp._B["grid_dates"] = pd.date_range(
        ccp._B["grid_start"], ccp._B["grid_start"] + pd.Timedelta(days=364),
        freq=f"{grid_step}D")
    ccp._B["max_train"] = max_train
    ccp._B["pred_chunk"] = pred_chunk

    _ML["min_clean"] = min_clean
    _ML["x_off"], _ML["y_off"] = x_off, y_off
    # Dedupe so the per-pixel/per-doy left merge stays one-to-one (and the
    # reshape to (n_pix, n_grid) is valid).
    _ML["ei"] = (pd.read_parquet(ei_parquet)
                 .drop_duplicates(["x_snap", "y_snap", "doy"]))
    _ML["ei_shift"] = ei_shift
    _ML["clip"] = clip_cfactor
    _ML["feature_columns"] = feature_columns
    _ML["n_steps"] = n_steps
    _ML["pipe"], *_ = load_model(model_path)


def _ei_snap(x, y):
    """Snap pixel coords to EI grid, applying the half-pixel shift only if asked."""
    if _ML["ei_shift"]:
        return snap_to_ei_grid(x + ccp.X_EI_SHIFT, y + ccp.Y_EI_SHIFT,
                               _ML["x_off"], _ML["y_off"])
    return snap_to_ei_grid(x, y, _ML["x_off"], _ML["y_off"])


def _b_process_field(args):
    poly_id, lnf_code, obs, pixels = args
    if len(obs) < _ML["min_clean"]:
        return None

    n_steps = _ML["n_steps"]
    # GP gap-fill on the regular grid -> (pixel × grid-date) rows of pv/npv.
    # Row layout (from ccp._gp_predict_grid): pixel index varies fastest,
    # grid-date index varies slowest -> reshape(n_grid, n_pix).
    grid = ccp._gp_predict_grid(obs[["x", "y", "time"] + VALUE_COLS], pixels)
    n_pix = len(pixels)
    n_grid = len(grid) // n_pix
    if n_grid != n_steps:
        # Should not happen if grid_step_days matches training; guard loudly.
        raise ValueError(
            f"field {poly_id}: gap-fill produced {n_grid} grid steps but the "
            f"model expects {n_steps}. Set grid_step_days to the value used to "
            "build the training parquet (samples_data_gpr.parquet).")

    # Per-timestep EI for each pixel (snap once per pixel, look up by doy).
    # Build the (pixel × grid-date) key table explicitly in pixel-major order
    # so the reshape below is order-safe regardless of merge internals.
    xs, ys = _ei_snap(pixels["x"].values, pixels["y"].values)
    grid_doy = grid["doy"].values[:n_grid * n_pix:n_pix]   # one doy per grid step
    ei_keys = pd.DataFrame({
        "x_snap": np.repeat(xs, n_grid),
        "y_snap": np.repeat(ys, n_grid),
        "doy":    np.tile(grid_doy, n_pix),
    })
    # how='left' on a key unique in _ML['ei'] preserves left order one-to-one.
    ei_keys = ei_keys.merge(_ML["ei"], on=["x_snap", "y_snap", "doy"], how="left")
    ei_mat = ei_keys["ei"].values.reshape(n_pix, n_grid)

    # pv/npv reshaped to (n_pix, n_grid): grid rows are grid-major, pixel-minor.
    pv_mat = grid["pv"].values.reshape(n_grid, n_pix).T
    npv_mat = grid["npv"].values.reshape(n_grid, n_pix).T

    # Assemble the wide feature frame; select in the model's column order so we
    # are robust to the saved ordering.
    cols = {}
    for i in range(n_steps):
        cols[f"pv_t{i:03d}"] = pv_mat[:, i]
        cols[f"npv_t{i:03d}"] = npv_mat[:, i]
        cols[f"ei_t{i:03d}"] = ei_mat[:, i]
    feat = pd.DataFrame(cols)

    X = feat[_ML["feature_columns"]].to_numpy(dtype=np.float32)
    valid = np.isfinite(X).all(axis=1)        # drop pixels missing any EI step
    if not valid.any():
        return None

    c_pred = _ML["pipe"].predict(X[valid])
    if _ML["clip"]:
        c_pred = np.clip(c_pred, 0.0, 1.0)

    out = pixels.iloc[np.where(valid)[0]][["x", "y"]].copy()
    out["c_factor"] = c_pred.astype(np.float32)
    out["poly_id"] = int(poly_id)
    out["lnf_code"] = int(lnf_code)
    out["n_clean_obs_field"] = int(len(obs))
    out["n_grid_pts"] = int(n_steps)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Orchestrator (Stages A + A½ identical to the empirical script)
# ---------------------------------------------------------------------------

def run(cfg):
    pipe, feature_columns, n_steps, model_kind = load_model(cfg["model_path"])
    print(f"Loaded ML model: kind={model_kind}, n_steps={n_steps}, "
          f"{len(feature_columns)} features from {cfg['model_path']}")

    lnf_dir = os.path.expanduser(cfg["lnf_dir"])
    ei_path = os.path.expanduser(cfg["ei_path"])
    output_dir = os.path.expanduser(cfg["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    region_2056 = None
    region_bounds_32632 = None
    if cfg.get("region_path"):
        region = gpd.read_file(os.path.expanduser(cfg["region_path"]))
        region_2056 = region.to_crs("EPSG:2056").union_all()
        region_bounds_32632 = tuple(region.to_crs("EPSG:32632").total_bounds)

    x_off, y_off = get_ei_grid_offset(ei_path)

    for year in cfg["years"]:
        print(f"\n{'='*60}\nYear {year}\n{'='*60}")
        lnf_path = os.path.join(lnf_dir, f"lnf{year}.gpkg")
        if not os.path.exists(lnf_path):
            print(f"  [SKIP] {lnf_path} not found")
            continue

        tiles = list_agyear_tiles(cfg["s2_dir"], year)
        if region_bounds_32632 is not None:
            rminx, rminy, rmaxx, rmaxy = region_bounds_32632
            tiles = {k: v for k, v in tiles.items()
                     if not (k[0] > rmaxx or k[0] + TILE_SIZE_M < rminx
                             or k[1] < rminy or k[1] - TILE_SIZE_M > rmaxy)}
        if not tiles:
            print("  [SKIP] No S2 tiles intersect the region")
            continue
        _log(f"  {len(tiles)} spatial tiles")

        # ---- Stage A: parallel over tiles (identical to empirical) ----
        obs_parts, pix_parts = [], []
        with ProcessPoolExecutor(max_workers=cfg["n_workers_io"]) as ex:
            futs = {ex.submit(process_tile, k, v, lnf_path, year, region_2056,
                              cfg["lnf_codes"], cfg["lnf_ignore"], cfg): k
                    for k, v in tiles.items()}
            for i, fut in enumerate(as_completed(futs), 1):
                o, p = fut.result()
                if o is not None:
                    obs_parts.append(o)
                    pix_parts.append(p)
                if i <= 5 or i % 25 == 0 or i == len(futs):
                    _log(f"    Stage A {i}/{len(futs)} tiles")
        if not obs_parts:
            print(f"  [WARN] No clean observations for {year}")
            continue
        obs = pd.concat(obs_parts, ignore_index=True)
        pix = pd.concat(pix_parts, ignore_index=True).drop_duplicates(["poly_id", "x", "y"])
        _log(f"  Stage A done: {len(obs):,} clean obs, {pix['poly_id'].nunique():,} fields, "
             f"{len(pix):,} pixels")

        # ---- Stage A½: FC prediction (identical to empirical) ----
        obs = predict_fc(obs, cfg)

        # ---- EI subset for the region -> temp parquet ----
        # Use the SAME snap convention as Stage B (no shift by default).
        _log("  Building EI subset for region ...")
        if cfg["ei_half_pixel_shift"]:
            xs, ys = snap_to_ei_grid(pix["x"].values + ccp.X_EI_SHIFT,
                                     pix["y"].values + ccp.Y_EI_SHIFT, x_off, y_off)
        else:
            xs, ys = snap_to_ei_grid(pix["x"].values, pix["y"].values, x_off, y_off)
        df_ei = (load_ei_for_pixels(ei_path, np.unique(xs), np.unique(ys))
                 .rename(columns={"x": "x_snap", "y": "y_snap"}))
        ei_tmp = os.path.join(tempfile.gettempdir(), f"ei_subset_ml_{year}.parquet")
        df_ei.to_parquet(ei_tmp, index=False)
        _log(f"  EI subset: {len(df_ei):,} rows → {ei_tmp}")

        # ---- Stage B: parallel over fields (ML prediction) ----
        _log("  Grouping observations by field ...")
        obs_by = dict(tuple(obs.groupby("poly_id")))
        pix_by = dict(tuple(pix.groupby("poly_id")))
        lnf_by = pix.groupby("poly_id")["lnf_code"].first().to_dict()
        tasks = [(pid, lnf_by[pid], obs_by[pid], pix_by[pid][["x", "y"]].drop_duplicates())
                 for pid in obs_by]
        _log(f"  Stage B: {len(tasks):,} fields to predict "
             f"(warming up {cfg['n_workers_gp']} spawn workers) ...")

        results = []
        with ProcessPoolExecutor(
            max_workers=cfg["n_workers_gp"], mp_context=mp.get_context("spawn"),
            initializer=_b_init,
            initargs=(cfg["model_path"], f"{year-1}-07-01", cfg["grid_step_days"],
                      cfg["max_train_points"], cfg["pred_chunk"], cfg["min_clean_obs"],
                      ei_tmp, x_off, y_off, cfg["ei_half_pixel_shift"],
                      cfg["clip_cfactor"], feature_columns, n_steps),
        ) as ex:
            futs = [ex.submit(_b_process_field, t) for t in tasks]
            n_total = len(futs)
            for i, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                if r is not None and len(r):
                    results.append(r)
                if i <= 5 or i % 25 == 0 or i == n_total:
                    _log(f"    Stage B {i}/{n_total} fields "
                         f"({100*i/n_total:.0f}%), {sum(len(d) for d in results):,} pixels")

        if not results:
            print(f"  [WARN] No pixels produced for {year}")
            continue
        out = pd.concat(results, ignore_index=True)
        out["year"] = np.int16(year)
        _log(f"  {len(out):,} pixels, {out['poly_id'].nunique():,} fields")

        out.to_parquet(output_dir, partition_cols=["year", "lnf_code"],
                       engine="pyarrow", index=False,
                       existing_data_behavior="delete_matching")
        if cfg["write_geotiff"]:
            write_geotiff(out, year, output_dir)

        # ---- Per-field roll-up (matches the R product's granularity) ----
        if cfg["write_field_summary"]:
            fld = (out.groupby(["poly_id", "lnf_code", "year"], as_index=False)
                       .agg(c_factor_mean=("c_factor", "mean"),
                            c_factor_std=("c_factor", "std"),
                            c_factor_median=("c_factor", "median"),
                            n_pixels=("c_factor", "size")))
            fld["area_m2"] = fld["n_pixels"] * (PIX_RES ** 2)
            fld_path = os.path.join(output_dir, f"cfactor_fields_{year}.parquet")
            fld.to_parquet(fld_path, index=False)
            _log(f"  Field summary: {len(fld):,} fields → {fld_path}")

        _log(f"  Written → {output_dir}/year={year}/...")


def main():
    p = argparse.ArgumentParser(description="Per-pixel C-factor inference (ML model)")
    p.add_argument("--region", help="Region vector (any CRS)")
    p.add_argument("--years", nargs="+", type=int)
    p.add_argument("--model", help="Path to nn_model.joblib")
    p.add_argument("--gpu", action="store_true", help="Use GPU for FC prediction if available")
    p.add_argument("--geotiff", action="store_true")
    p.add_argument("--ei-shift", action="store_true",
                   help="Apply x-5,y+5 EI shift (only if training parquet used it)")
    a = p.parse_args()

    cfg = dict(CONFIG)
    if a.region:
        cfg["region_path"] = a.region
    if a.years:
        cfg["years"] = a.years
    if a.model:
        cfg["model_path"] = a.model
    if a.gpu:
        cfg["use_gpu"] = True
    if a.geotiff:
        cfg["write_geotiff"] = True
    if a.ei_shift:
        cfg["ei_half_pixel_shift"] = True
    if not cfg.get("region_path"):
        print("[WARN] No --region: processing ALL tiles (whole country).")
    run(cfg)
    print("\nDone.")


if __name__ == "__main__":
    main()
