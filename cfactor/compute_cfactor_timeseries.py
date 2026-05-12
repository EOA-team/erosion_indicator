"""
Compute per-field SLR(t)·EI(t) time series for all agricultural fields,
for any year with available Sentinel-2 FC data.

Outputs hive-partitioned Parquet:
    <output_dir>/year=YYYY/lnf_code=NNN/part-000.parquet

Each row is one field × one (valid, cloud-free) timestep:
    poly_id, lnf_code, year, time, doy,
    fc_mean, fc_std, n_valid_pixels, slr, ei, slr_ei

Annual C factor per field = slr_ei.sum() / ei.sum()  (group by poly_id, year)

Usage:
    python compute_cfactor_timeseries.py
    python compute_cfactor_timeseries.py --years 2022 2023
"""

import argparse
import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import xarray as xr
from rasterio.features import rasterize as rio_rasterize
from rasterio.transform import from_bounds
from shapely.geometry import box
from shapely.ops import transform as shapely_transform

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from calibrate_cfactor import (
    get_ei_grid_offset,
    load_ei_for_pixels,
    snap_to_ei_grid,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    "beta_path":    os.path.join(os.path.dirname(__file__), "beta.json"),
    "fc_dir":       "~/mnt/eo-nas1/data/satellite/sentinel2/FC",
    "s2_dir":       "~/mnt/eo-nas1/data/satellite/sentinel2/raw/CH",
    "lnf_dir":      "~/mnt/eo-nas1/data/landuse/raw",
    "ei_path":      "../erosivity_index/predictions/grid_EI_daily_avg_pred_20260424_nn3.parquet",
    "output_dir":   os.path.join(os.path.dirname(__file__), "output", "cfactor_timeseries"),
    "years":        [2021, 2022, 2023, 2024],
    "lnf_codes":    None,       # None = all crop types; or list of int LNF codes
    "lnf_ignore":   [],         # LNF codes to always exclude
    "n_workers":    6,
    # Cloud/snow/shadow thresholds (tile-level, same as create_datalayers.py)
    "cloud_thresh":  0.05,
    "snow_thresh":   0.10,
    "shadow_thresh": 0.10,
    "cirrus_thresh": 800,
}

TILE_SIZE_M = 1280   # tile side length in metres (UTM zone 32)
TILE_PIXELS  = 128   # pixels per side (10 m resolution)


# ---------------------------------------------------------------------------
# Tile helpers (mirrors create_datalayers.py)
# ---------------------------------------------------------------------------

def _parse_tile_name(fname: str) -> tuple[int, int, int, int]:
    """Return (minx, miny, maxx, maxy) from S2_minx_maxy_*.zarr filename."""
    parts = os.path.basename(fname).split("_")
    minx = int(parts[1])
    maxy = int(parts[2])
    return minx, maxy - TILE_SIZE_M, minx + TILE_SIZE_M, maxy


def _list_fc_tiles(fc_dir: str, year: int) -> list[str]:
    """Return absolute paths to all FC zarr tiles for *year*."""
    fc_dir = os.path.expanduser(fc_dir)
    return [
        os.path.join(fc_dir, f)
        for f in os.listdir(fc_dir)
        if f.endswith(".zarr") and f.split("_")[3].startswith(str(year))
    ]


def _tile_bbox_2056(minx_32632, miny_32632, maxx_32632, maxy_32632):
    """Reproject tile bbox from EPSG:32632 to EPSG:2056 for LNF spatial query."""
    project = pyproj.Transformer.from_crs(
        "EPSG:32632", "EPSG:2056", always_xy=True
    ).transform
    bbox_poly = box(minx_32632, miny_32632, maxx_32632, maxy_32632)
    return shapely_transform(project, bbox_poly).bounds  # (minx, miny, maxx, maxy) in 2056


# ---------------------------------------------------------------------------
# Cloud/snow masking (reused from create_datalayers.py logic)
# ---------------------------------------------------------------------------

def _valid_time_mask(fc_ds: xr.Dataset, s2_path: str, cfg: dict) -> np.ndarray:
    """Return boolean mask over fc_ds.time for timestamps that pass cloud filter.

    Filters tile-level cloud/snow/shadow using the corresponding raw S2 zarr.
    Uses time-coordinate matching (both zarrs share the same time axis).
    Falls back to all-True if the raw S2 file is missing.
    """
    if not os.path.exists(s2_path):
        return np.ones(fc_ds.sizes["time"], dtype=bool)

    s2 = xr.open_zarr(s2_path)[
        ["s2_mask", "s2_SCL", "s2_B02", "s2_B03", "s2_B11"]
    ].load()

    valid_pix = (s2.s2_mask != 4).sum(dim=["lat", "lon"]).clip(min=1)

    mask_cloud = (
        ((s2.s2_mask == 1) | s2.s2_SCL.isin([8, 9, 10])).sum(dim=["lat", "lon"])
        / valid_pix
    ) > cfg["cloud_thresh"]

    mask_shadow = (
        ((s2.s2_mask == 2) | (s2.s2_SCL == 3)).sum(dim=["lat", "lon"])
        / valid_pix
    ) > cfg["shadow_thresh"]

    ndsi_denom = s2.s2_B03 / 10000 + s2.s2_B11 / 10000 + 1e-9
    ndsi = (s2.s2_B03 / 10000 - s2.s2_B11 / 10000) / ndsi_denom
    n_valid = (s2.s2_mask != 4).sum(dim=["lat", "lon"]).clip(min=1)
    mask_snow = (
        ((ndsi > 0.4) & (s2.s2_mask != 4)).sum(dim=["lat", "lon"]) / n_valid
    ) > cfg["snow_thresh"]

    cirrus_b02 = s2.s2_B02.where(s2.s2_SCL == 10).mean(dim=["lat", "lon"])
    mask_cirrus = cirrus_b02 > cfg["cirrus_thresh"]

    drop_s2 = (mask_cloud | mask_shadow | mask_snow | mask_cirrus).values
    valid_s2_times = set(s2.time.values[~drop_s2])

    fc_times = fc_ds.time.values
    return np.array([t in valid_s2_times for t in fc_times])


# ---------------------------------------------------------------------------
# Stage 1: per-tile partial aggregation
# ---------------------------------------------------------------------------

def process_tile(
    fc_path: str,
    s2_dir: str,
    lnf_path: str,
    year: int,
    lnf_codes,
    lnf_ignore: list,
    cfg: dict,
) -> pd.DataFrame | None:
    """
    Aggregate FC over pixels within each field for every valid timestep.

    Returns a DataFrame with columns:
        poly_id, lnf_code, time, fc_sum, fc_sq_sum, n_pixels
    or None if the tile contains no matching fields.
    """
    try:
        minx, miny, maxx, maxy = _parse_tile_name(fc_path)

        # --- Load fields intersecting this tile ---
        bbox_2056 = _tile_bbox_2056(minx, miny, maxx, maxy)
        fields = gpd.read_file(lnf_path, bbox=bbox_2056)
        if fields.empty:
            return None
        fields = fields[fields.geometry.is_valid & ~fields.geometry.is_empty]
        if lnf_codes is not None:
            fields = fields[fields["lnf_code"].isin(lnf_codes)]
        if lnf_ignore:
            fields = fields[~fields["lnf_code"].isin(lnf_ignore)]
        if fields.empty:
            return None

        fields = fields.to_crs("EPSG:32632")
        fields["poly_id"] = fields["id"]  # stable field identifier

        # --- Rasterize field polygons onto tile grid ---
        # Rasterio: origin top-left, y decreases downward
        transform = from_bounds(minx, miny, maxx, maxy, TILE_PIXELS, TILE_PIXELS)
        pid_shapes = [
            (geom, int(pid))
            for geom, pid in zip(fields.geometry, fields["poly_id"])
            if geom is not None and not geom.is_empty
        ]
        lnf_shapes = [
            (geom, int(code))
            for geom, code in zip(fields.geometry, fields["lnf_code"])
            if geom is not None and not geom.is_empty
        ]
        poly_raster = rio_rasterize(
            pid_shapes, out_shape=(TILE_PIXELS, TILE_PIXELS),
            transform=transform, fill=0, dtype="int64",
        )
        lnf_raster = rio_rasterize(
            lnf_shapes, out_shape=(TILE_PIXELS, TILE_PIXELS),
            transform=transform, fill=0, dtype="int32",
        )
        # Flip vertically: rasterio row 0 = north; zarr lat[0] = south
        poly_raster = np.flipud(poly_raster)
        lnf_raster = np.flipud(lnf_raster)

        # --- Load FC zarr ---
        fc_ds = xr.open_zarr(fc_path)
        if "PV_norm_soil" not in fc_ds.data_vars or "NPV_norm_soil" not in fc_ds.data_vars:
            return None

        # --- Cloud/snow filter ---
        s2_path = os.path.join(os.path.expanduser(s2_dir), os.path.basename(fc_path))
        time_mask = _valid_time_mask(fc_ds, s2_path, cfg)
        if time_mask.sum() == 0:
            return None

        # Load only valid timestamps
        fc_ds = fc_ds.isel(time=time_mask).compute()
        times = fc_ds.time.values  # (T,)
        T = len(times)

        # fc_total = (PV_norm_soil + NPV_norm_soil) * 100 — matches calibration
        # which used soil-group-specific NN predictions via predict_FC()
        fc_arr = (fc_ds["PV_norm_soil"].values + fc_ds["NPV_norm_soil"].values) * 100.0

        # --- Scatter-sum FC per field per timestep ---
        # fc_arr layout: (lat=128, lon=128, time=T)
        # poly_raster layout: (lat=128, lon=128) — same orientation
        pid_flat = poly_raster.flatten()   # (N=128*128,)
        lnf_flat = lnf_raster.flatten()

        field_mask = pid_flat > 0
        if field_mask.sum() == 0:
            return None

        fc_flat = fc_arr.reshape(-1, T)[field_mask]  # (N_field_pixels, T)
        pid_valid = pid_flat[field_mask]
        lnf_valid = lnf_flat[field_mask]

        unique_pids, inv_idx = np.unique(pid_valid, return_inverse=True)
        n_fields = len(unique_pids)

        nan_mask = np.isnan(fc_flat)
        fc_filled = np.where(nan_mask, 0.0, fc_flat)

        fc_sum = np.zeros((n_fields, T))
        fc_sq_sum = np.zeros((n_fields, T))
        n_pix = np.zeros((n_fields, T), dtype=np.int32)

        np.add.at(fc_sum, inv_idx, fc_filled)
        np.add.at(fc_sq_sum, inv_idx, fc_filled ** 2)
        np.add.at(n_pix, inv_idx, (~nan_mask).astype(np.int32))

        # Build lnf_code lookup from raster (majority vote per poly_id)
        pid_to_lnf = {}
        for pid, lnf in zip(pid_valid, lnf_valid):
            pid_to_lnf[pid] = lnf  # one lnf_code per poly_id (consistent by definition)

        # Assemble long-form DataFrame — only rows where n_pixels > 0
        parts = []
        for i, pid in enumerate(unique_pids):
            valid_t = n_pix[i] > 0
            if not valid_t.any():
                continue
            parts.append(pd.DataFrame({
                "poly_id":    int(pid),
                "lnf_code":   int(pid_to_lnf[pid]),
                "time":       times[valid_t],
                "fc_sum":     fc_sum[i, valid_t],
                "fc_sq_sum":  fc_sq_sum[i, valid_t],
                "n_pixels":   n_pix[i, valid_t].astype(np.int32),
            }))

        if not parts:
            return None
        return pd.concat(parts, ignore_index=True)

    except Exception as exc:
        print(f"  [WARN] {os.path.basename(fc_path)}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Stage 2: merge partials across tiles
# ---------------------------------------------------------------------------

def _merge_tile_partials(partials: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Combine partial tile results (same field can appear in multiple tiles).
    Returns fc_mean, fc_std, n_valid_pixels per (poly_id, time).
    """
    df = pd.concat(partials, ignore_index=True)
    agg = (
        df.groupby(["poly_id", "lnf_code", "time"], as_index=False)
        .agg(fc_sum=("fc_sum", "sum"), fc_sq_sum=("fc_sq_sum", "sum"),
             n_valid_pixels=("n_pixels", "sum"))
    )
    agg["fc_mean"] = agg["fc_sum"] / agg["n_valid_pixels"]
    # std = sqrt(E[x²] - E[x]²), clipped to 0 for float precision
    agg["fc_std"] = np.sqrt(
        np.clip(agg["fc_sq_sum"] / agg["n_valid_pixels"] - agg["fc_mean"] ** 2, 0, None)
    )
    agg["doy"] = pd.to_datetime(agg["time"]).dt.dayofyear.astype(np.int16)
    return agg.drop(columns=["fc_sum", "fc_sq_sum"])


# ---------------------------------------------------------------------------
# Stage 3: EI join and SLR computation
# ---------------------------------------------------------------------------

def _compute_slr_ei(
    fc_df: pd.DataFrame,
    beta: float,
    ei_path: str,
    fields_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Join EI climatology and compute SLR = exp(-β·fc_mean), slr_ei = slr·ei.
    Uses field centroid (in EPSG:32632) snapped to the 100 m EI grid.
    """
    ei_path = os.path.expanduser(ei_path)
    x_off, y_off = get_ei_grid_offset(ei_path)

    # Build centroid lookup (poly_id → x_snap, y_snap)
    fields_32632 = fields_gdf.to_crs("EPSG:32632") if fields_gdf.crs.to_epsg() != 32632 else fields_gdf
    centroids = fields_32632.copy()
    centroids["centroid"] = centroids.geometry.centroid
    centroids["cx"] = centroids["centroid"].x
    centroids["cy"] = centroids["centroid"].y
    centroids = centroids[["id", "cx", "cy"]].rename(columns={"id": "poly_id"})

    # Keep only fields present in fc_df
    present_pids = fc_df["poly_id"].unique()
    centroids = centroids[centroids["poly_id"].isin(present_pids)]

    cx = centroids["cx"].values
    cy = centroids["cy"].values
    x_snap, y_snap = snap_to_ei_grid(cx, cy, x_off, y_off)
    centroids = centroids.copy()
    centroids["x_snap"] = x_snap
    centroids["y_snap"] = y_snap

    # Load EI only for the unique (x_snap, y_snap) cells needed
    x_unique = np.unique(x_snap)
    y_unique = np.unique(y_snap)
    df_ei = load_ei_for_pixels(ei_path, x_unique, y_unique)

    # Merge centroid snap coords onto fc_df
    df = fc_df.merge(centroids[["poly_id", "x_snap", "y_snap"]], on="poly_id", how="left")

    # Merge EI by (x_snap, y_snap, doy)
    df_ei = df_ei.rename(columns={"x": "x_snap", "y": "y_snap"})
    df = df.merge(df_ei, on=["x_snap", "y_snap", "doy"], how="left")

    n_miss = df["ei"].isna().sum()
    if n_miss:
        print(f"  [WARN] {n_miss} rows ({n_miss/len(df):.1%}) have no EI match — excluded from slr_ei")

    df["slr"] = np.exp(-beta * df["fc_mean"]).astype(np.float32)
    df["slr_ei"] = (df["slr"] * df["ei"]).astype(np.float32)
    df["ei"] = df["ei"].astype(np.float32)
    df["fc_mean"] = df["fc_mean"].astype(np.float32)
    df["fc_std"] = df["fc_std"].astype(np.float32)

    return df.drop(columns=["x_snap", "y_snap"])


# ---------------------------------------------------------------------------
# Field metadata geometry file
# ---------------------------------------------------------------------------

def _write_field_metadata(fields_gdf: gpd.GeoDataFrame, year: int, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    out = fields_gdf[["id", "lnf_code", "geometry"]].copy()
    out = out.rename(columns={"id": "poly_id"})
    out["area_m2"] = out.geometry.area.astype(np.float32)
    centroids = out.to_crs("EPSG:32632").geometry.centroid
    out["centroid_x"] = centroids.x.astype(np.float32)
    out["centroid_y"] = centroids.y.astype(np.float32)
    path = os.path.join(output_dir, f"field_metadata_{year}.gpkg")
    out.to_file(path, driver="GPKG", layer="fields")
    print(f"  Field metadata → {path}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_cfactor_timeseries(config: dict) -> None:
    beta_path  = os.path.expanduser(config["beta_path"])
    fc_dir     = os.path.expanduser(config["fc_dir"])
    s2_dir     = config["s2_dir"]
    lnf_dir    = os.path.expanduser(config["lnf_dir"])
    ei_path    = os.path.expanduser(config["ei_path"])
    output_dir = os.path.expanduser(config["output_dir"])
    years      = config["years"]
    lnf_codes  = config.get("lnf_codes")
    lnf_ignore = config.get("lnf_ignore", [])
    n_workers  = config.get("n_workers", 6)

    with open(beta_path) as f:
        beta = json.load(f)["beta"]
    print(f"Loaded β = {beta:.5f} from {beta_path}")

    os.makedirs(output_dir, exist_ok=True)

    for year in years:
        print(f"\n{'='*60}\nYear {year}\n{'='*60}")
        lnf_path = os.path.join(lnf_dir, f"lnf{year}.gpkg")
        if not os.path.exists(lnf_path):
            print(f"  [SKIP] {lnf_path} not found")
            continue

        fc_tiles = _list_fc_tiles(fc_dir, year)
        if not fc_tiles:
            print(f"  [SKIP] No FC tiles found for {year} in {fc_dir}")
            continue
        print(f"  {len(fc_tiles)} FC tiles to process")

        # Stage 1: parallel tile processing
        partials = []
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {
                ex.submit(process_tile, p, s2_dir, lnf_path, year,
                          lnf_codes, lnf_ignore, config): p
                for p in fc_tiles
            }
            for i, fut in enumerate(as_completed(futures), 1):
                result = fut.result()
                if result is not None and len(result):
                    partials.append(result)
                if i % 50 == 0 or i == len(futures):
                    print(f"  {i}/{len(futures)} tiles done, {len(partials)} with data")

        if not partials:
            print(f"  [WARN] No field data extracted for {year}")
            continue

        # Stage 2: merge across tiles
        print(f"  Merging {len(partials)} partial results …")
        fc_df = _merge_tile_partials(partials)
        fc_df["year"] = np.int16(year)
        print(f"  {len(fc_df):,} field×timestep rows, "
              f"{fc_df['poly_id'].nunique():,} unique fields")

        # Load all fields once for centroid lookup and metadata
        print("  Loading full LNF fields for centroid lookup …")
        all_fields = gpd.read_file(lnf_path)
        all_fields = all_fields[all_fields["id"].isin(fc_df["poly_id"].unique())]
        if lnf_codes is not None:
            all_fields = all_fields[all_fields["lnf_code"].isin(lnf_codes)]
        if lnf_ignore:
            all_fields = all_fields[~all_fields["lnf_code"].isin(lnf_ignore)]

        # Stage 3: EI join + SLR
        print("  Joining EI and computing SLR …")
        final_df = _compute_slr_ei(fc_df, beta, ei_path, all_fields)

        # Write partitioned Parquet
        print("  Writing partitioned Parquet …")
        final_df.to_parquet(
            output_dir,
            partition_cols=["year", "lnf_code"],
            engine="pyarrow",
            index=False,
            existing_data_behavior="delete_matching",
        )

        # Write companion geometry file
        _write_field_metadata(all_fields, year, output_dir)
        print(f"  Done: {year}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute C-factor time series for all fields")
    parser.add_argument("--years", nargs="+", type=int,
                        help="Years to process (default: all in CONFIG['years'])")
    args = parser.parse_args()

    cfg = dict(CONFIG)
    if args.years:
        cfg["years"] = args.years

    run_cfactor_timeseries(cfg)
    print("\nAll years complete.")


if __name__ == "__main__":
    main()
