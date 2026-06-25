"""
Per-pixel gap-filled FC time series for a region (Stage A + A½ + GP, no C).

This is a standalone extractor that reuses Stage A, Stage A½, and the per-field
GP gap-fill from `compute_cfactor_pixels.py`, but persists the FC grid instead
of collapsing it to a C-factor. Use it to inspect HOW the input to the C-factor
model (the fractional cover composition) evolves through the year and changes
across years -- e.g. did the residue cover (NPV) drop between 2021 and 2022?

Pipeline (reuses compute_cfactor_pixels.py end-to-end except the EI/SLR step):

  Stage A   — parallel over TILES. Rasterise fields onto the zarr grid, load
              raw S2 (bands + SCL/mask), clean per the chosen mode, attach
              per-field dominant soil_group, emit clean observations.
  Stage A½  — predict pv/npv/soil for ALL clean observations with the SALI
              per-soil-group ensemble (== `predict_FC`).
  Stage B'  — parallel over FIELDS. One GP in ALR space (identical kernel to
              `_gp_predict_grid`) trained on the field's clean pixel obs, then
              predicted at every field pixel on the regular `grid_step_days`
              cadence over the agricultural year [yr-1-07-01 .. yr-06-30].
              EMITS the FC grid instead of computing SLR.EI/EI.

Output (two parquets per year, both under <output_dir>):
  - year=YYYY/lnf_code=NNN/part-*.parquet
        per-pixel x per-DOAY grid; columns:
        poly_id, lnf_code, x, y, time, doay, pv, npv, fc_total,
        n_clean_obs_field, year
  - fc_fields_{year}.parquet
        per-field per-DOAY mean (collapsed across pixels); columns:
        poly_id, lnf_code, doay, time, pv_mean, npv_mean, fc_total_mean,
        pv_std, npv_std, fc_total_std, n_pixels, n_clean_obs_field, year
        Compact (~MB) and is the form `compare_products.py` reads by default.

DOAY ("day of agricultural year") is days-since-1-July-of-yr-1, with DOAY = 1
on 1 July. Same anchor every year, so 2021 and 2022 line up on one x-axis.

Usage:
    python extract_fc_pixels.py --region seeland.gpkg --years 2021 2022
    python extract_fc_pixels.py --region seeland.gpkg --years 2022 --gpu
    python extract_fc_pixels.py --region seeland.gpkg --years 2022 \\
        --no-write-per-pixel               # just the small per-field summary

Cost is dominated by the per-field GP fits (one fit per field, plus prediction
at n_pixels x ~37 grid dates). Memory and disk: the per-pixel grid is large --
roughly n_pixels * (365 / grid_step_days) rows per year; the per-field summary
is ~3-4 orders of magnitude smaller.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

# Limit BLAS threads BEFORE numpy/scipy/sklearn -- with a process pool,
# multi-threaded BLAS oversubscribes the CPUs and makes every GP fit crawl.
# `compute_cfactor_pixels` sets the same env at import time, but be explicit
# here so the script is robust when run as __main__.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import geopandas as gpd
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Reuse Stages A + A½ + the per-field GP from compute_cfactor_pixels so the
# cleaning / gap-fill behaviour stays bit-identical to the C-factor product.
sys.path.insert(0, os.path.dirname(__file__))
import compute_cfactor_pixels as _ccp
from compute_cfactor_pixels import (
    _log,
    list_agyear_tiles,
    process_tile,
    predict_fc,
    _gp_predict_grid,
    PIX_RES,            # noqa: F401  (kept for downstream callers)
    TILE_SIZE_M,
    VALUE_COLS,
    CONFIG as _CCP_CONFIG,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Inherit cleaning / GP defaults from compute_cfactor_pixels so the FC grid
# we extract is the SAME grid that goes into the C-factor model. The only
# overrides are the output directory and a couple of unused flags.
CONFIG = dict(_CCP_CONFIG)
CONFIG.update({
    "output_dir":            os.path.join(os.path.dirname(__file__),
                                          "output", "fc_pixels"),
    "years":                 [2022],
    "write_geotiff":         False,
    "write_field_summary":   False,
    # New knobs (no equivalent in compute_cfactor_pixels.CONFIG)
    "write_per_pixel":       True,    # the big hive-partitioned parquet
    "write_fc_field_summary": True,   # the small per-field summary parquet
})


# ---------------------------------------------------------------------------
# Worker: per-field GP that emits the FC grid (no SLR / no EI / no C)
# ---------------------------------------------------------------------------

def _fc_worker_init(grid_start_iso, grid_step, max_train, pred_chunk, min_clean):
    """Populate `compute_cfactor_pixels._B` in the worker process so we can
    call `_gp_predict_grid` directly (it reads its grid / training settings
    from that module-level dict). Avoids duplicating ~50 lines of GP code.
    """
    _ccp._B["grid_start"] = pd.Timestamp(grid_start_iso)
    _ccp._B["grid_dates"] = pd.date_range(
        _ccp._B["grid_start"],
        _ccp._B["grid_start"] + pd.Timedelta(days=364),
        freq=f"{grid_step}D",
    )
    _ccp._B["max_train"]  = max_train
    _ccp._B["pred_chunk"] = pred_chunk
    _ccp._B["min_clean"]  = min_clean


def _fc_worker_field(args):
    """One field -> per-pixel x per-DOAY FC grid, ready to write.

    `_gp_predict_grid` returns (x, y, doy, pv, npv) where `doy` is calendar
    day-of-year. We attach the actual `time` and an agricultural-year DOAY
    (1 July of yr-1 == DOAY 1) so cross-year overlays line up on one x-axis.
    """
    poly_id, lnf_code, obs, pixels = args
    if len(obs) < _ccp._B["min_clean"]:
        return None
    grid = _gp_predict_grid(obs[["x", "y", "time"] + VALUE_COLS], pixels)
    grid_dates = _ccp._B["grid_dates"]
    t0         = _ccp._B["grid_start"]
    n_pix      = len(pixels)

    # `grid` is laid out as np.tile(pixels, n_grid) for (x, y) and
    # np.repeat(grid_dates, n_pix) for the date axis -- same convention as
    # _gp_predict_grid -- so we repeat the per-date arrays here.
    grid["time"] = np.repeat(grid_dates.values, n_pix)
    doay = (grid_dates - t0).days.astype(np.int16) + 1     # 1 == 1 July of yr-1
    grid["doay"] = np.repeat(doay.values, n_pix)
    grid["fc_total"] = ((grid["pv"] + grid["npv"]) * 100).astype(np.float32)
    grid["poly_id"]  = int(poly_id)
    grid["lnf_code"] = int(lnf_code)
    grid["n_clean_obs_field"] = int(len(obs))
    # Drop calendar DOY (we keep DOAY which is the year-anchored one)
    grid = grid.drop(columns=["doy"])

    # Cast to compact dtypes (the per-pixel grid is large)
    for c in ("pv", "npv", "fc_total"):
        grid[c] = grid[c].astype(np.float32)
    grid["x"] = grid["x"].astype(np.float64)
    grid["y"] = grid["y"].astype(np.float64)
    return grid[["poly_id", "lnf_code", "x", "y", "time", "doay",
                 "pv", "npv", "fc_total", "n_clean_obs_field"]]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _field_summary(out: pd.DataFrame) -> pd.DataFrame:
    """Per (poly_id, lnf_code, doay) mean / std / n over pixels.

    The companion of `cfactor_fields_{year}.parquet`: one row per field x DOAY
    instead of one row per pixel x DOAY. Tiny next to the per-pixel grid.
    """
    return (out.groupby(["poly_id", "lnf_code", "doay"], as_index=False)
               .agg(time=("time", "first"),
                    pv_mean=("pv", "mean"),
                    npv_mean=("npv", "mean"),
                    fc_total_mean=("fc_total", "mean"),
                    pv_std=("pv", "std"),
                    npv_std=("npv", "std"),
                    fc_total_std=("fc_total", "std"),
                    n_pixels=("fc_total", "size"),
                    n_clean_obs_field=("n_clean_obs_field", "first")))


def run(cfg):
    s2_dir     = os.path.expanduser(cfg["s2_dir"])
    lnf_dir    = os.path.expanduser(cfg["lnf_dir"])
    output_dir = os.path.expanduser(cfg["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    region_2056         = None
    region_bounds_32632 = None
    if cfg.get("region_path"):
        region              = gpd.read_file(os.path.expanduser(cfg["region_path"]))
        region_2056         = region.to_crs("EPSG:2056").union_all()
        region_bounds_32632 = tuple(region.to_crs("EPSG:32632").total_bounds)
    else:
        print("  [WARN] --region not set; running over the full LNF gpkg "
              "for each year. This is expensive at country scale.")

    for year in cfg["years"]:
        print(f"\n{'='*60}\nYear {year}\n{'='*60}")
        lnf_path = os.path.join(lnf_dir, f"lnf{year}.gpkg")
        if not os.path.exists(lnf_path):
            print(f"  [SKIP] {lnf_path} not found")
            continue

        # ---- Tile shortlist (same bbox cull as compute_cfactor_pixels) ----
        tiles = list_agyear_tiles(s2_dir, year)
        if region_bounds_32632 is not None:
            rminx, rminy, rmaxx, rmaxy = region_bounds_32632
            tiles = {k: v for k, v in tiles.items()
                     if not (k[0] > rmaxx or k[0] + TILE_SIZE_M < rminx
                             or k[1] < rminy or k[1] - TILE_SIZE_M > rmaxy)}
        if not tiles:
            print("  [SKIP] No S2 tiles intersect the region")
            continue
        _log(f"  {len(tiles)} spatial tiles")

        # ---- Stage A: tile-parallel S2 cleaning (identical to ccp) ----
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
        _log(f"  Stage A done: {len(obs):,} clean obs, "
             f"{pix['poly_id'].nunique():,} fields, {len(pix):,} pixels")

        # ---- Stage A½: FC prediction (GPU optional) ----
        obs = predict_fc(obs, cfg)

        # ---- Stage B': per-field GP -> FC grid (no SLR / no EI / no C) ----
        _log("  Grouping observations by field ...")
        obs_by = dict(tuple(obs.groupby("poly_id")))
        pix_by = dict(tuple(pix.groupby("poly_id")))
        lnf_by = pix.groupby("poly_id")["lnf_code"].first().to_dict()
        tasks  = [(pid, lnf_by[pid], obs_by[pid],
                   pix_by[pid][["x", "y"]].drop_duplicates())
                  for pid in obs_by]
        _log(f"  Stage B' (FC-only): {len(tasks):,} fields to gap-fill "
             f"(warming up {cfg['n_workers_gp']} spawn workers -- first "
             f"results take a moment) ...")

        # 'spawn' (not fork): predict_fc may have initialised a CUDA context
        # in this process; forking after CUDA init deadlocks the workers.
        grid_start_iso = f"{year-1}-07-01"
        results, n_total = [], len(tasks)
        with ProcessPoolExecutor(
            max_workers=cfg["n_workers_gp"],
            mp_context=mp.get_context("spawn"),
            initializer=_fc_worker_init,
            initargs=(grid_start_iso, cfg["grid_step_days"],
                      cfg["max_train_points"], cfg["pred_chunk"],
                      cfg["min_clean_obs"]),
        ) as ex:
            futs = [ex.submit(_fc_worker_field, t) for t in tasks]
            for i, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                if r is not None and len(r):
                    results.append(r)
                if i <= 5 or i % 25 == 0 or i == n_total:
                    _log(f"    Stage B' {i}/{n_total} fields "
                         f"({100*i/n_total:.0f}%), "
                         f"{sum(len(d) for d in results):,} rows")

        if not results:
            print(f"  [WARN] No FC rows produced for {year}")
            continue
        out = pd.concat(results, ignore_index=True)
        out["year"] = np.int16(year)
        _log(f"  {len(out):,} per-pixel rows, "
             f"{out['poly_id'].nunique():,} fields, "
             f"{out['doay'].nunique()} DOAYs")

        # ---- Persist ----
        if cfg.get("write_per_pixel", True):
            out.to_parquet(output_dir, partition_cols=["year", "lnf_code"],
                           engine="pyarrow", index=False,
                           existing_data_behavior="delete_matching")
            _log(f"  per-pixel grid -> {output_dir}/year={year}/...")

        if cfg.get("write_fc_field_summary", True):
            fld = _field_summary(out)
            fld["year"] = np.int16(year)
            fld_path = os.path.join(output_dir, f"fc_fields_{year}.parquet")
            fld.to_parquet(fld_path, index=False)
            _log(f"  per-field summary ({len(fld):,} rows) -> {fld_path}")


def main():
    p = argparse.ArgumentParser(
        description="Per-pixel gap-filled FC time series "
                    "(Stage A + A½ + GP, no C-factor integration)")
    p.add_argument("--region", help="Region vector (any CRS)")
    p.add_argument("--years", nargs="+", type=int, default=CONFIG["years"])
    p.add_argument("--lnf-codes", nargs="*", type=int, default=None,
                   help="Restrict to these LNF codes (default: all crops)")
    p.add_argument("--output-dir", default=CONFIG["output_dir"])
    p.add_argument("--gpu", action="store_true",
                   help="Use GPU for FC prediction if CUDA is available")
    p.add_argument("--n-workers-io", type=int, default=CONFIG["n_workers_io"])
    p.add_argument("--n-workers-gp", type=int, default=CONFIG["n_workers_gp"])
    p.add_argument("--no-write-per-pixel", action="store_true",
                   help="Skip the big hive-partitioned per-pixel parquet "
                        "(keep only the small per-field summary)")
    p.add_argument("--no-write-field-summary", action="store_true",
                   help="Skip the per-field summary parquet")
    args = p.parse_args()

    cfg = dict(CONFIG)
    if args.region:
        cfg["region_path"] = args.region
    cfg["years"]        = args.years
    if args.lnf_codes is not None:
        cfg["lnf_codes"] = args.lnf_codes
    cfg["output_dir"]   = args.output_dir
    cfg["use_gpu"]      = args.gpu
    cfg["n_workers_io"] = args.n_workers_io
    cfg["n_workers_gp"] = args.n_workers_gp
    cfg["write_per_pixel"]       = not args.no_write_per_pixel
    cfg["write_fc_field_summary"] = not args.no_write_field_summary

    print(f"  region:          {cfg.get('region_path')}")
    print(f"  years:           {cfg['years']}")
    print(f"  output_dir:      {cfg['output_dir']}")
    print(f"  per-pixel grid:  {cfg['write_per_pixel']}")
    print(f"  per-field summ.: {cfg['write_fc_field_summary']}")
    print(f"  workers io/gp:   {cfg['n_workers_io']}/{cfg['n_workers_gp']}")
    print(f"  GPU FC pred.:    {cfg['use_gpu']}")

    run(cfg)


if __name__ == "__main__":
    main()
