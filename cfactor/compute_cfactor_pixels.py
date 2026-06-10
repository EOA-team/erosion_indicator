"""
Per-pixel annual C-factor inference for a region (raw-S2, calibration-faithful).

Applies the calibrated soil-loss-ratio model (β from `beta.json`) to every
Sentinel-2 pixel inside agricultural fields within a region, for one
agricultural year. This mirrors the *active* calibration data path
(`fc_precompute=False`): FC is predicted from raw S2 with the SALI NN models,
and cloud cleaning uses the SCL / mask bands that live in the same raw S2 zarr.

Pipeline (two stages, so fields spanning tiles are never split):

  Stage A  — parallel over TILES (CPU, I/O bound). For each spatial tile:
             rasterise fields onto the zarr grid, load raw S2 (bands +
             SCL/mask) for the agricultural year, clean per the chosen mode,
             attach the per-field dominant soil_group, and emit
               * clean observations  [poly_id, lnf_code, x, y, time, <bands>, soil_group]
               * pixel inventory     [poly_id, lnf_code, x, y, soil_group]
             Each tile is read exactly once.

  Stage A½ — single process. Predict FC (pv, npv, soil) for ALL clean
             observations with the SALI per-soil-group ensemble (== predict_FC
             in sample_FC: bands/10000, 5-iteration mean, clip[0,1], renorm).
             Runs on GPU when `use_gpu` and CUDA are available, else CPU.

  Stage B  — parallel over FIELDS (CPU). For each field: one GP in ALR space
             (Matern over [t, sin, cos, x, y], identical to
             `_gapfill_one_field_alr_regular`) trained on the field's clean
             pixel observations and predicted at EVERY field pixel on the
             regular `grid_step_days` grid over [yr-1-07-01 .. yr-06-30]; then
             SLR = exp(-β·FC), join climatological EI by (snapped cell, doy),
             and reduce to one C per pixel = Σ SLR·EI / Σ EI.

Output: hive-partitioned Parquet  <output_dir>/year=YYYY/lnf_code=NNN/part-*.parquet
        columns: poly_id, lnf_code, x, y, year, c_factor,
                 n_clean_obs_field, n_grid_pts
Optionally one 10 m C-factor GeoTIFF per tile-block (--geotiff).

Usage:
    python compute_cfactor_pixels.py --region region.shp --years 2022
    python compute_cfactor_pixels.py --region region.gpkg --years 2021 2022 --gpu --geotiff

Cost: one GP fit per field (O(n_train^3), n_train ≤ max_train_points) plus a
prediction at n_pixels × n_grid points. Start on a small region.
"""

import argparse
import json
import os
import sys
import tempfile
import warnings

# Limit BLAS threads BEFORE importing numpy/scipy/sklearn: with a process pool,
# multi-threaded BLAS oversubscribes the CPUs and makes every GP fit crawl.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from rasterio.features import rasterize as rio_rasterize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from calibrate_cfactor import (          # light imports (no torch)
    get_ei_grid_offset,
    load_ei_for_pixels,
    snap_to_ei_grid,
)

import time as _time
_T0 = _time.time()


def _log(msg):
    """Timestamped, flushed print so progress shows even when output is piped."""
    el = _time.time() - _T0
    print(f"[{int(el // 60):02d}:{el % 60:05.2f}] {msg}", flush=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    "beta_path":    "calibration_analysis_twobeta_noley/beta_stratified.json",
    "region_path":  None,            # set via --region; any OGR vector
    "s2_dir":       "~/mnt/eo-nas1/data/satellite/sentinel2/raw/CH",
    "soil_dir":     "~/mnt/eo-nas1/data/satellite/sentinel2/DLR_soilsuite_preds/",
    "lnf_dir":      "~/mnt/eo-nas1/data/landuse/raw",
    "ei_path":      "../erosivity_index/predictions/grid_EI_daily_avg_pred_20260424_nn3.parquet",
    "output_dir":   os.path.join(os.path.dirname(__file__), "output", "cfactor_pixels"),
    "years":        [2022],
    "lnf_codes":    None,            # None = all crop types; or list of int LNF codes
    "lnf_ignore":   [],              # LNF codes to always exclude
    "n_workers_io": 6,               # Stage A (tile) workers
    "n_workers_gp": 6,               # Stage B (field) workers

    # ---- FC prediction (SALI NN) ----
    "sali_src_path": "~/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure/SALI_models",
    "fc_model_dir":  "~/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure/SALI_models/FC/models/",
    "s2_bands":      ["s2_B02", "s2_B03", "s2_B04", "s2_B05", "s2_B06",
                      "s2_B07", "s2_B08", "s2_B8A", "s2_B11", "s2_B12"],
    "use_gpu":       False,          # --gpu; falls back to CPU if CUDA absent
    "fc_chunk":      50000,          # rows per NN inference batch

    # ---- Soil-group lookup (per field, dominant) ----
    # Soil-suite zarr filename for a tile, formatted with minx/maxy (EPSG:32632).
    # Adjust if your soil tiles use a different grid/naming than the S2 tiles.
    "soil_name_template": "SRC_{minx}_{maxy}.zarr",

    # ---- Cleaning ----
    # 'per_pixel'  : drop each masked pixel-date (recommended for a per-pixel product)
    # 'field_date' : sampling-faithful — drop a date for the field when > max_missing_frac
    #                of its pixels are masked; keep ALL pixels on surviving dates.
    "clean_mode":      "per_pixel",
    "cirrus_thresh":   500,
    "max_missing_frac": 0.05,        # only used by clean_mode == 'field_date'

    # ---- GP gap-fill (== _gapfill_one_field_alr_regular) ----
    "grid_step_days":   10,
    "max_train_points": 1000,
    "min_clean_obs":    3,
    "pred_chunk":       20000,

    # ---- Output ----
    "write_geotiff":    False,
    "write_field_summary": True,     # also write per-field mean C (R-product granularity)
}

TILE_SIZE_M = 1280
TILE_PIXELS = 128
PIX_RES     = 10.0
VALUE_COLS  = ["pv", "npv", "soil"]          # ALR order; soil is the reference
# Half-pixel shift applied before EI snapping, matching sample_FC (`x-=5, y+=5`).
X_EI_SHIFT  = -5.0
Y_EI_SHIFT  = +5.0


# ---------------------------------------------------------------------------
# Beta + SLR (robust to single / two_beta / stratified schemas)
# ---------------------------------------------------------------------------

def load_beta(beta_path: str):
    with open(beta_path) as f:
        rec = json.load(f)
    if "beta_pv" in rec and "beta_npv" in rec:
        return (float(rec["beta_pv"]), float(rec["beta_npv"])), "two_beta"
    if "beta" in rec:
        return float(rec["beta"]), "single"
    raise KeyError(f"{beta_path}: no 'beta' or 'beta_pv'/'beta_npv' ({list(rec)})")


def slr_from_composition(pv, npv, beta):
    """SLR = exp(-exponent) — mirrors calibrate_cfactor._slr_exponent."""
    if np.isscalar(beta):
        return np.exp(-float(beta) * (pv + npv) * 100.0)
    b_pv, b_npv = beta
    return np.exp(-(float(b_pv) * pv * 100.0 + float(b_npv) * npv * 100.0))


# ---------------------------------------------------------------------------
# ALR (copied from sample_FC to avoid importing torch in workers)
# ---------------------------------------------------------------------------

def alr_transform(fc):
    fc = np.clip(fc, 1e-6, 1 - 1e-6)
    return np.log(fc[:, :-1] / fc[:, -1:])


def alr_inverse(log_ratios):
    e = np.exp(log_ratios)
    denom = 1 + e.sum(axis=1, keepdims=True)
    return np.hstack([e / denom, 1.0 / denom])


# ---------------------------------------------------------------------------
# Tile discovery (raw S2 zarrs)
# ---------------------------------------------------------------------------

def _spatial_key(fname):
    parts = os.path.basename(fname).split("_")
    return int(parts[1]), int(parts[2])          # (minx, maxy) in EPSG:32632


def _date_year(fname):
    return os.path.basename(fname).split("_")[3][:4]


def list_agyear_tiles(s2_dir, yr):
    """Group raw S2 zarr tiles by spatial key, keeping calendar years yr-1 and yr."""
    s2_dir = os.path.expanduser(s2_dir)
    want = {str(yr - 1), str(yr)}
    tiles = {}
    for f in os.listdir(s2_dir):
        if f.endswith(".zarr") and _date_year(f) in want:
            tiles.setdefault(_spatial_key(f), []).append(os.path.join(s2_dir, f))
    return tiles


def _tile_bounds(minx, maxy):
    return minx, maxy - TILE_SIZE_M, minx + TILE_SIZE_M, maxy


def _box(minx, miny, maxx, maxy):
    from shapely.geometry import box
    return box(minx, miny, maxx, maxy)


# ---------------------------------------------------------------------------
# Soil-group per field (dominant), from the soil-suite zarr for the tile
# ---------------------------------------------------------------------------

def _field_soil_groups(ds_grid, poly_raster, soil_dir, template, key):
    """Return {poly_id: dominant soil_group}. Defaults to 0 where unavailable."""
    minx, maxy = key
    path = os.path.join(os.path.expanduser(soil_dir),
                        template.format(minx=minx, maxy=maxy))
    if not os.path.exists(path):
        print(f"    [soil] tile zarr not found: {path}", flush=True)
        return {}
    try:
        soil = xr.open_zarr(path)[["soil_group"]]
        # The soil zarr already uses x/y dims (unlike the S2 zarr, which is
        # lat/lon). Only rename if a legacy lat/lon-named tile turns up;
        # renaming unconditionally raises and silently zeroes every field.
        if "lat" in soil.dims or "lon" in soil.dims:
            soil = soil.rename({"lat": "y", "lon": "x"})
        soil = soil.rio.write_crs(32632).rio.reproject_match(ds_grid)  # align to S2 grid
        sg = soil["soil_group"].values
        if sg.ndim == 3:
            sg = sg[0]
    except Exception as e:
        # Don't fail the tile, but make the cause visible — a silent {} here
        # collapses the whole run to soil_group 0.
        print(f"    [soil] read failed for {path}: {e!r}", flush=True)
        return {}

    out = {}
    for pid in np.unique(poly_raster[poly_raster > 0]):
        vals = sg[poly_raster == pid]
        vals = vals[(vals != -10000) & ~np.isnan(vals)]
        if len(vals):
            v, c = np.unique(vals.astype(int), return_counts=True)
            out[int(pid)] = int(v[c.argmax()])
        else:
            out[int(pid)] = 0
    return out


# ---------------------------------------------------------------------------
# Stage A: one tile -> clean observations + pixel inventory
# ---------------------------------------------------------------------------

def process_tile(key, s2_paths, lnf_path, year, region_2056,
                 lnf_codes, lnf_ignore, cfg):
    """Return (obs_df, pix_df) for one tile, or (None, None)."""
    try:
        minx, maxy = key
        minx, miny, maxx, maxy = _tile_bounds(minx, maxy)

        # --- Open raw S2 for the agricultural year (concat yr-1 + yr) ---
        want = cfg["s2_bands"] + ["s2_mask", "s2_SCL"]
        parts = []
        for p in s2_paths:
            ds = xr.open_zarr(p)
            if not all(b in ds.data_vars for b in want):
                continue
            parts.append(ds[want])
        if not parts:
            return None, None
        ds = (xr.concat(parts, dim="time") if len(parts) > 1 else parts[0])
        ds = ds.rename({"lat": "y", "lon": "x"}).rio.write_crs(32632).sortby("time")

        gstart = pd.Timestamp(f"{year - 1}-07-01")
        gend = pd.Timestamp(f"{year}-06-30")
        ds = ds.sel(time=slice(gstart, gend)).load()
        if ds.sizes["time"] == 0:
            return None, None

        # --- Fields in tile ∩ region (EPSG:2056), reproject to 32632 ---
        bbox_2056 = tuple(gpd.GeoSeries([_box(minx, miny, maxx, maxy)],
                                        crs=32632).to_crs(2056).total_bounds)
        fields = gpd.read_file(lnf_path, bbox=bbox_2056)
        if fields.empty:
            return None, None
        fields = fields[fields.geometry.is_valid & ~fields.geometry.is_empty]
        if region_2056 is not None:
            fields = fields[fields.intersects(region_2056)]
        if lnf_codes is not None:
            fields = fields[fields["lnf_code"].isin(lnf_codes)]
        if lnf_ignore:
            fields = fields[~fields["lnf_code"].isin(lnf_ignore)]
        if fields.empty:
            return None, None
        fields = fields.to_crs("EPSG:32632")
        fields["poly_id"] = fields["id"].astype("int64")

        # --- Rasterise poly_id / lnf_code onto the zarr grid (rio transform) ---
        transform = ds.rio.transform()
        shape = (ds.sizes["y"], ds.sizes["x"])
        poly_raster = rio_rasterize(
            [(g, int(p)) for g, p in zip(fields.geometry, fields["poly_id"])
             if g is not None and not g.is_empty],
            out_shape=shape, transform=transform, fill=0, dtype="int64")
        lnf_raster = rio_rasterize(
            [(g, int(c)) for g, c in zip(fields.geometry, fields["lnf_code"])
             if g is not None and not g.is_empty],
            out_shape=shape, transform=transform, fill=0, dtype="int32")
        field_mask = poly_raster > 0
        if field_mask.sum() == 0:
            return None, None

        # --- Per-field dominant soil_group ---
        soil_map = _field_soil_groups(ds, poly_raster, cfg["soil_dir"],
                                      cfg["soil_name_template"], key)

        # --- Pixel-centre coordinates from the zarr (EPSG:32632) ---
        xx, yy = np.meshgrid(ds.x.values, ds.y.values)        # (y, x)

        li, ci = np.where(field_mask)
        pid_pix = poly_raster[li, ci]
        lnf_pix = lnf_raster[li, ci]
        x_pix = xx[li, ci]
        y_pix = yy[li, ci]
        sg_pix = np.array([soil_map.get(int(p), 0) for p in pid_pix], dtype=np.int32)

        # --- Cleaning masks (per pixel-date) ---
        mask = ds["s2_mask"].values                        # (T, y, x)
        scl = ds["s2_SCL"].values
        b02 = ds["s2_B02"].values
        masked = ((mask == 4)
                  | (mask == 1) | np.isin(scl, [8, 9, 10])    # cloud
                  | (mask == 2) | (scl == 3)                  # shadow
                  | (mask == 3) | (scl == 11)                 # snow
                  | ((scl == 10) & (b02 > cfg["cirrus_thresh"])))  # cirrus

        times = ds.time.values
        band_arr = np.stack([ds[b].values for b in cfg["s2_bands"]], axis=-1)  # (T,y,x,B)
        band_f = band_arr[:, li, ci, :]                    # (T, n_fp, B)
        masked_f = masked[:, li, ci]                       # (T, n_fp)

        T, n_fp = masked_f.shape
        ti = np.repeat(np.arange(T), n_fp)
        pxi = np.tile(np.arange(n_fp), T)
        df = pd.DataFrame({
            "poly_id": pid_pix[pxi],
            "lnf_code": lnf_pix[pxi],
            "x": x_pix[pxi],
            "y": y_pix[pxi],
            "soil_group": sg_pix[pxi],
            "time": times[ti],
            "masked": masked_f.reshape(-1),
        })
        for bi, b in enumerate(cfg["s2_bands"]):
            df[b] = band_f[:, :, bi].reshape(-1)

        # drop rows with no usable bands
        df = df[~df[cfg["s2_bands"]].isna().all(axis=1)]
        df = df[(df[cfg["s2_bands"]] != 65535).any(axis=1)]

        # --- Cleaning mode ---
        if cfg["clean_mode"] == "per_pixel":
            obs = df[~df["masked"]].drop(columns="masked")
        else:  # field_date
            frac = df.groupby(["poly_id", "time"])["masked"].mean().rename("frac")
            df = df.merge(frac, on=["poly_id", "time"])
            obs = df[df["frac"] <= cfg["max_missing_frac"]].drop(columns=["masked", "frac"])
        if obs.empty:
            return None, None

        pix = pd.DataFrame({
            "poly_id": pid_pix, "lnf_code": lnf_pix,
            "x": x_pix, "y": y_pix, "soil_group": sg_pix,
        }).drop_duplicates(["poly_id", "x", "y"])

        return obs.reset_index(drop=True), pix.reset_index(drop=True)

    except Exception as exc:                               # noqa: BLE001
        print(f"  [WARN] tile {key}: {exc}")
        return None, None


# ---------------------------------------------------------------------------
# Stage A½: batched FC prediction (== sample_FC.predict_FC), GPU optional
# ---------------------------------------------------------------------------

def predict_fc(obs, cfg):
    """Add pv/npv/soil columns to `obs` using the SALI per-soil-group ensemble."""
    import torch
    sys.path.insert(0, os.path.expanduser(cfg["sali_src_path"]))
    from src.model_utils import load_all_models

    bands = cfg["s2_bands"]
    device = torch.device("cuda" if (cfg["use_gpu"] and torch.cuda.is_available()) else "cpu")
    _log(f"  Stage A½: predicting FC on {len(obs):,} clean observations (device={device})")

    X = obs[bands].to_numpy(dtype=np.float32) / 10000.0
    sg = obs["soil_group"].to_numpy().astype(int)
    groups = sorted(np.unique(sg).tolist())
    _log(f"    loading FC models for soil groups {groups}")
    models = load_all_models(os.path.expanduser(cfg["fc_model_dir"]), groups)

    out = np.zeros((len(obs), 3), dtype=np.float32)
    chunk = cfg["fc_chunk"]
    for gi, g in enumerate(groups, 1):
        idx = np.where(sg == g)[0]
        _log(f"    soil group {g} ({gi}/{len(groups)}): {len(idx):,} obs")
        acc = np.zeros((len(idx), 3), dtype=np.float32)
        for it in range(1, 6):
            # load_all_models may place weights on CUDA; force them onto `device`
            m_pv = models[g][it]["PV"].to(device).eval()
            m_npv = models[g][it]["NPV"].to(device).eval()
            m_s = models[g][it]["Soil"].to(device).eval()
            for s in range(0, len(idx), chunk):
                blk = idx[s:s + chunk]
                xb = torch.from_numpy(X[blk]).float().to(device)
                with torch.no_grad():
                    acc[s:s + chunk, 0] += m_pv(xb).cpu().numpy().squeeze()
                    acc[s:s + chunk, 1] += m_npv(xb).cpu().numpy().squeeze()
                    acc[s:s + chunk, 2] += m_s(xb).cpu().numpy().squeeze()
        acc /= 5.0
        acc = acc.clip(0, 1)
        rs = acc.sum(axis=1, keepdims=True)
        acc /= np.where(rs > 0, rs, 1.0)
        out[idx] = acc
    _log("    FC prediction done")

    obs = obs.copy()
    obs[["pv", "npv", "soil"]] = out
    return obs.drop(columns=bands)


# ---------------------------------------------------------------------------
# Stage B: per-field GP gap-fill at all pixels + EI + C  (parallel over fields)
# ---------------------------------------------------------------------------

_B = {}   # worker globals


def _b_init(beta, grid_start_iso, grid_step, max_train, pred_chunk, min_clean,
            ei_parquet, x_off, y_off):
    _B["beta"] = beta
    _B["grid_start"] = pd.Timestamp(grid_start_iso)
    _B["grid_dates"] = pd.date_range(_B["grid_start"],
                                     _B["grid_start"] + pd.Timedelta(days=364),
                                     freq=f"{grid_step}D")
    _B["max_train"] = max_train
    _B["pred_chunk"] = pred_chunk
    _B["min_clean"] = min_clean
    _B["x_off"], _B["y_off"] = x_off, y_off
    _B["ei"] = pd.read_parquet(ei_parquet)


def _gp_predict_grid(obs, pixels):
    """GP in ALR space -> FC composition for every pixel × grid date."""
    grid_dates = _B["grid_dates"]
    t0 = _B["grid_start"]

    tt = (pd.to_datetime(obs["time"]) - t0).dt.days.values.reshape(-1, 1).astype(float)
    t_tr = tt / 365.0
    sin_tr = np.sin(2 * np.pi * (tt % 365) / 365)
    cos_tr = np.cos(2 * np.pi * (tt % 365) / 365)
    scaler = StandardScaler()
    xy_tr = scaler.fit_transform(obs[["x", "y"]].values)
    X_full = np.hstack([t_tr, sin_tr, cos_tr, xy_tr]).astype(np.float32)
    alr_full = alr_transform(obs[VALUE_COLS].values.astype(np.float32))

    if len(obs) > _B["max_train"]:
        times = t_tr[:, 0]
        nb = min(10, _B["max_train"])
        edges = np.linspace(times.min(), times.max(), nb + 1)
        bins = np.clip(np.digitize(times, edges) - 1, 0, nb - 1)
        per = max(1, _B["max_train"] // nb)
        rng = np.random.default_rng(42)
        keep = np.concatenate([rng.choice(np.where(bins == b)[0],
                               size=min(per, int((bins == b).sum())), replace=False)
                               for b in range(nb) if (bins == b).any()])
        X_train, alr_train = X_full[keep], alr_full[keep]
    else:
        X_train, alr_train = X_full, alr_full

    nf = X_train.shape[1]
    est_alpha = float(np.clip(np.var(alr_train, axis=0).mean() * 0.05, 1e-4, 0.5))
    ls_bounds = [(0.1, 5.0)] + [(0.5, 10.0)] * 2 + [(0.05, 10.0)] * 2
    kernel = (ConstantKernel(1.0, (0.1, 5.0))
              * Matern(length_scale=[1.0] * nf, length_scale_bounds=ls_bounds, nu=1.5)
              + WhiteKernel(noise_level=est_alpha, noise_level_bounds=(1e-5, 0.5)))

    n_pix, n_grid = len(pixels), len(grid_dates)
    tg = (grid_dates - t0).days.values.astype(float)
    tblock = np.repeat(np.column_stack([tg / 365.0,
                                        np.sin(2 * np.pi * (tg % 365) / 365),
                                        np.cos(2 * np.pi * (tg % 365) / 365)]).astype(np.float32),
                       n_pix, axis=0)
    xyblock = np.tile(scaler.transform(pixels[["x", "y"]].values).astype(np.float32), (n_grid, 1))
    X_pred = np.hstack([tblock, xyblock])

    alr_pred = np.zeros((len(X_pred), 2), dtype=np.float32)
    fitted = None
    for j in range(2):
        gp = GaussianProcessRegressor(kernel=kernel if j == 0 else fitted,
                                      alpha=est_alpha, normalize_y=True,
                                      n_restarts_optimizer=5 if j == 0 else 2)
        gp.fit(X_train, alr_train[:, j])
        if j == 0:
            fitted = gp.kernel_
        for s in range(0, len(X_pred), _B["pred_chunk"]):
            alr_pred[s:s + _B["pred_chunk"], j] = gp.predict(X_pred[s:s + _B["pred_chunk"]])
    fc = alr_inverse(alr_pred)

    return pd.DataFrame({
        "x": np.tile(pixels["x"].values, n_grid),
        "y": np.tile(pixels["y"].values, n_grid),
        "doy": np.repeat(pd.DatetimeIndex(grid_dates).dayofyear.values.astype(np.int16), n_pix),
        "pv": fc[:, 0], "npv": fc[:, 1],
    })


def _b_process_field(args):
    poly_id, lnf_code, obs, pixels = args
    if len(obs) < _B["min_clean"]:
        return None
    grid = _gp_predict_grid(obs[["x", "y", "time"] + VALUE_COLS], pixels)

    grid["slr"] = slr_from_composition(grid["pv"].values, grid["npv"].values,
                                       _B["beta"]).astype(np.float32)
    xs, ys = snap_to_ei_grid(grid["x"].values + X_EI_SHIFT,
                             grid["y"].values + Y_EI_SHIFT, _B["x_off"], _B["y_off"])
    grid["x_snap"], grid["y_snap"] = xs, ys
    grid = grid.merge(_B["ei"], on=["x_snap", "y_snap", "doy"], how="left").dropna(subset=["ei"])
    if grid.empty:
        return None
    grid["_num"] = grid["slr"] * grid["ei"]
    out = (grid.groupby(["x", "y"], as_index=False)
               .agg(_num=("_num", "sum"), _den=("ei", "sum"), n_grid_pts=("slr", "size")))
    out["c_factor"] = np.where(out["_den"] > 0, out["_num"] / out["_den"], np.nan)
    out["poly_id"] = int(poly_id)
    out["lnf_code"] = int(lnf_code)
    out["n_clean_obs_field"] = int(len(obs))
    return out.drop(columns=["_num", "_den"])


# ---------------------------------------------------------------------------
# GeoTIFF writer
# ---------------------------------------------------------------------------

def write_geotiff(df, year, output_dir):
    import rasterio
    from rasterio.transform import from_origin
    if df.empty:
        return
    xs, ys = np.sort(df["x"].unique()), np.sort(df["y"].unique())
    ncol = int(round((xs.max() - xs.min()) / PIX_RES)) + 1
    nrow = int(round((ys.max() - ys.min()) / PIX_RES)) + 1
    arr = np.full((nrow, ncol), np.nan, dtype=np.float32)
    col = np.round((df["x"].values - xs.min()) / PIX_RES).astype(int)
    row = np.round((ys.max() - df["y"].values) / PIX_RES).astype(int)
    arr[row, col] = df["c_factor"].values
    tdir = os.path.join(output_dir, "geotiff")
    os.makedirs(tdir, exist_ok=True)
    path = os.path.join(tdir, f"cfactor_{year}_{int(xs.min())}_{int(ys.max())}.tif")
    with rasterio.open(path, "w", driver="GTiff", height=nrow, width=ncol, count=1,
                       dtype="float32", crs="EPSG:32632",
                       transform=from_origin(xs.min() - PIX_RES / 2, ys.max() + PIX_RES / 2,
                                              PIX_RES, PIX_RES),
                       nodata=np.nan, compress="deflate") as dst:
        dst.write(arr, 1)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run(cfg):
    beta, mode = load_beta(os.path.expanduser(cfg["beta_path"]))
    print(f"Loaded β ({mode}) = {beta}")

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

        # ---- Stage A: parallel over tiles ----
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

        # ---- Stage A½: FC prediction (GPU optional) ----
        obs = predict_fc(obs, cfg)

        # ---- EI subset for the region (single scan) -> temp parquet ----
        _log("  Building EI subset for region ...")
        xs, ys = snap_to_ei_grid(pix["x"].values + X_EI_SHIFT,
                                 pix["y"].values + Y_EI_SHIFT, x_off, y_off)
        df_ei = (load_ei_for_pixels(ei_path, np.unique(xs), np.unique(ys))
                 .rename(columns={"x": "x_snap", "y": "y_snap"}))
        ei_tmp = os.path.join(tempfile.gettempdir(), f"ei_subset_{year}.parquet")
        df_ei.to_parquet(ei_tmp, index=False)
        _log(f"  EI subset: {len(df_ei):,} rows → {ei_tmp}")

        # ---- Stage B: parallel over fields ----
        _log("  Grouping observations by field ...")
        obs_by = dict(tuple(obs.groupby("poly_id")))
        pix_by = dict(tuple(pix.groupby("poly_id")))
        lnf_by = pix.groupby("poly_id")["lnf_code"].first().to_dict()
        tasks = [(pid, lnf_by[pid], obs_by[pid], pix_by[pid][["x", "y"]].drop_duplicates())
                 for pid in obs_by]
        _log(f"  Stage B: {len(tasks):,} fields to gap-fill "
             f"(warming up {cfg['n_workers_gp']} spawn workers — first results take a moment) ...")

        results = []
        # 'spawn' (not fork): predict_fc initialised a CUDA context in this
        # process, and forking after CUDA init deadlocks the child workers.
        with ProcessPoolExecutor(
            max_workers=cfg["n_workers_gp"], mp_context=mp.get_context("spawn"),
            initializer=_b_init,
            initargs=(beta, f"{year-1}-07-01", cfg["grid_step_days"],
                      cfg["max_train_points"], cfg["pred_chunk"], cfg["min_clean_obs"],
                      ei_tmp, x_off, y_off),
        ) as ex:
            futs = [ex.submit(_b_process_field, t) for t in tasks]
            n_total = len(futs)
            for i, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                if r is not None and len(r):
                    results.append(r)
                # verbose at the start (so you see workers are alive), then every 25
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
        # The R erosion product assigns ONE crop C-factor per Nutzungsfläche and
        # averages potential erosion risk over the field. The comparable
        # quantity is the area-mean of the per-pixel C-factors per field.
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
    p = argparse.ArgumentParser(description="Per-pixel C-factor inference (raw-S2, two-stage)")
    p.add_argument("--region", help="Region vector (any CRS)")
    p.add_argument("--years", nargs="+", type=int)
    p.add_argument("--gpu", action="store_true", help="Use GPU for FC prediction if available")
    p.add_argument("--geotiff", action="store_true")
    a = p.parse_args()

    cfg = dict(CONFIG)
    if a.region:
        cfg["region_path"] = a.region
    if a.years:
        cfg["years"] = a.years
    if a.gpu:
        cfg["use_gpu"] = True
    if a.geotiff:
        cfg["write_geotiff"] = True
    if not cfg.get("region_path"):
        print("[WARN] No --region: processing ALL tiles (whole country).")
    run(cfg)
    print("\nDone.")


if __name__ == "__main__":
    main()