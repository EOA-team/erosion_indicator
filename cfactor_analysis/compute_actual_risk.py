"""
Actual erosion risk (pot_risk x C x P), per polygon, for the 3 C-factor products.

Standalone follow-up to ``compare_products.py``: the C-factor comparison there
asks "where do the three products disagree on C?". This script asks the next
question: "given each product, what does *actual* erosion risk look like, and
how do the resulting risk maps differ?".

The three products
------------------
  1. EMPIRICAL  per-pixel parquet  (``output/cfactor_pixels``)
  2. ML         per-pixel parquet  (``output/cfactor_pixels_ml``)
  3. PREVIOUS   per-(betr_ID, crop) C from the R erosion pipeline.

Design (matches the user's choices)
-----------------------------------
* Granularity        : field-level for all 3 products (per LNF polygon).
* Combine rule       : empirical/ML use mean(C * pot_risk) * P over each
                       polygon's 10 m pixels (uses within-polygon coupling).
                       R is pot_risk_poly * C_R * P (C constant inside polygon).
* P-factor           : 0.8 (matches R's erosion_config.yaml; overridable).
* R's actual risk    : recomputed with the SAME pot_risk_poly used for emp/ML,
                       so all cross-product differences come from C.

Pot_risk source
---------------
The 2 m *Erosionsrisikokarte des Ackerlandes* (R's ``erskacker22fl1.tif``,
referenced in ``02-Init.R``). Units: t ha^-1 y^-1. It's read, windowed to the
region, and reprojected with mean resampling to a 10 m EPSG:32632 grid that
snaps to the empirical pixel-product grid. The reprojected grid is cached as a
GeoTIFF so subsequent runs skip the slow source read.

Outputs (all under cfg['out_dir'])
----------------------------------
  actualrisk_{year}_{product}_fieldmap.png            per-product actual risk
  actualrisk_{year}_{a}_vs_{b}_diff_fieldmap.png      signed pairwise diff
  actualrisk_{year}_{a}_vs_{b}_absdiff_fieldmap.png   hotspot magnitude
  actualrisk_field_table_{year}.csv                   one row per polygon
  actualrisk_{year}_hotspots.csv                      top-20 divergent fields/pair
  actualrisk_{year}_by_crop.{png,csv}                 mean risk per crop, 3 products
  actualrisk_{year}_amplification_by_crop.{png,csv}   mean C*P per crop, 3 products
  actualrisk_{year}_crop_diff_vs_previous.{png,csv}   per-crop dRisk: emp/ML - R
  actualrisk_{year}_by_tillage.{png,csv}              actual risk + C*P by tillage
  actualrisk_{year}_drivers.{png,csv}                 variance decomposition:
                                                       topography vs management
  actualrisk_summary.txt                              headline numbers

Usage
-----
    python compute_actual_risk.py
    python compute_actual_risk.py --years 2022
    python compute_actual_risk.py --no-cache              # ignore cached raster
    python compute_actual_risk.py --skip-maps             # CSVs only
    python compute_actual_risk.py --skip-r                # emp/ML only

Shared paths (region, LNF dir, erosion-results dir, etc.) come from
``compare_products.CONFIG`` by import; override here only the actualrisk-
specific keys.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the data-access and plotting plumbing from compare_products. The script
# is expected to sit next to compare_products.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_products as cp
from compare_products import (
    # loaders + bridges
    load_pixels, load_r_results,
    _build_bridges,
    build_tillage_map,
    # plotting helpers
    _field_value_map, _field_diff_map, _field_absdiff_map, _joint_vmax,
    # IO + summary
    _expand, _ensure_dir, _savecsv, savefig, Summary,
    # constants
    PIX_RES, PROD_COLORS, PROD_LABELS, TILLAGE_ORDER,
)

warnings.filterwarnings("ignore")

import geopandas as gpd


# ---------------------------------------------------------------------------
# Configuration -- inherits paths from compare_products.CONFIG, overrides the
# actualrisk-specific knobs below.
# ---------------------------------------------------------------------------

CONFIG = {
    **cp.CONFIG,

    # 2 m Erosionsrisikokarte des Ackerlandes (referenced by the R pipeline in
    # `02-Init.R` as `erosion <- paste0(waum_daten, "ersk_acker22fl/erskacker22fl1.tif")`).
    # Units: t ha^-1 y^-1. CRS: EPSG:2056 (LV95).
    "pot_risk_raster":   "~/mnt/Data-Labo-RE/27_Natural_Resources-RE/"
                         "321.4_WAUM_protected/Daten/ersk_acker22fl/erskacker22fl1.tif",

    # P-factor used to combine pot_risk x C x P. Matches R's erosion_config.yaml
    # (`P_Faktor: 0.8`). Output units: t ha^-1 y^-1.
    "p_factor":          0.8,

    # GeoTIFF cache for the 10 m reprojected pot_risk grid. If it's a bare
    # filename, it lives under cfg['out_dir']; absolute paths or paths
    # containing a directory are used as-is. Set to None / use --no-cache to
    # disable.
    "pot_risk_cache":    "pot_risk_10m_utm32n.tif",

    # Top-N crops shown in per-crop bars (defaults to cp.CONFIG['top_n_crops']).
    "actualrisk_top_n_crops": None,

    # Separate output folder so this doesn't pollute figures_compare/.
    "out_dir":           "figures_actualrisk",
}


# ===========================================================================
# Raster: read, reproject (mean), cache
# ===========================================================================

def _resolve_cache_path(cfg: dict) -> str | None:
    """Resolve the raster cache path; bare filenames are placed under out_dir."""
    p = cfg.get("pot_risk_cache")
    if not p:
        return None
    p_exp = _expand(p)
    if os.path.isabs(p_exp) or os.path.dirname(p_exp):
        return p_exp
    return os.path.join(_expand(cfg["out_dir"]), p_exp)


def _build_pot_risk_grid(cfg: dict, years: list,
                         use_cache: bool = True) -> tuple:
    """Return ``(arr, transform)`` for the 10 m EPSG:32632 pot_risk grid.

    Read order:
      1. cache GeoTIFF, if present and ``use_cache`` is True
      2. otherwise read + reproject the source 2 m raster, then save the cache
    """
    try:
        import rasterio
        from rasterio.warp import (
            reproject, Resampling, transform_bounds,
        )
        from rasterio.transform import from_origin
        from rasterio.windows import from_bounds
    except Exception as e:
        print(f"  [WARN] rasterio not available -- cannot load pot_risk: {e}")
        return None, None

    cache = _resolve_cache_path(cfg) if use_cache else None
    if cache and os.path.exists(cache):
        try:
            with rasterio.open(cache) as src:
                arr = src.read(1).astype(np.float32)
                if src.nodata is not None and not np.isnan(src.nodata):
                    arr = np.where(arr == src.nodata, np.nan, arr)
                tr = src.transform
            print(f"  pot_risk cache hit: {cache}  ({arr.shape[0]} x {arr.shape[1]})")
            return arr, tr
        except Exception as e:
            print(f"  [WARN] could not read cache {cache}: {e} -- rebuilding")

    src_path = _expand(cfg["pot_risk_raster"])
    if not os.path.exists(src_path):
        print(f"  [WARN] pot_risk raster not found: {src_path}")
        return None, None

    # 1) Destination bbox from empirical pixels (union across years).
    xs_min, xs_max, ys_min, ys_max = np.inf, -np.inf, np.inf, -np.inf
    found_any = False
    for y in years:
        pix = load_pixels("empirical", cfg, y)
        if pix is None or pix.empty:
            continue
        found_any = True
        xs_min = min(xs_min, float(pix["x"].min()))
        xs_max = max(xs_max, float(pix["x"].max()))
        ys_min = min(ys_min, float(pix["y"].min()))
        ys_max = max(ys_max, float(pix["y"].max()))
    if not found_any:
        print("  [WARN] no empirical pixels found -- cannot derive grid bbox")
        return None, None

    pad = PIX_RES * 4
    x_min = float(np.floor((xs_min - pad) / PIX_RES) * PIX_RES)
    x_max = float(np.ceil((xs_max + pad) / PIX_RES) * PIX_RES)
    y_min = float(np.floor((ys_min - pad) / PIX_RES) * PIX_RES)
    y_max = float(np.ceil((ys_max + pad) / PIX_RES) * PIX_RES)
    ncol = int(round((x_max - x_min) / PIX_RES))
    nrow = int(round((y_max - y_min) / PIX_RES))
    dst_transform = from_origin(x_min, y_max, PIX_RES, PIX_RES)
    print(f"  pot_risk dest grid: {nrow} x {ncol} cells "
          f"({nrow * ncol / 1e6:.1f} M) at 10 m UTM 32N")

    # 2) Window-read the source raster (in source CRS).
    with rasterio.open(src_path) as src:
        try:
            src_bbox = transform_bounds("EPSG:32632", src.crs,
                                        x_min, y_min, x_max, y_max,
                                        densify_pts=21)
            window = from_bounds(*src_bbox, transform=src.transform)
            window = window.round_offsets().round_lengths()
            src_arr = src.read(1, window=window, masked=False)
            src_transform = src.window_transform(window)
        except Exception as e:
            print(f"  [WARN] could not window-read pot_risk ({e}); reading full")
            src_arr = src.read(1, masked=False)
            src_transform = src.transform
        src_crs = src.crs
        src_nodata = src.nodata

    src_arr_f = src_arr.astype("float32", copy=False)
    if src_nodata is not None and not (isinstance(src_nodata, float) and np.isnan(src_nodata)):
        src_arr_f = np.where(src_arr_f == src_nodata, np.nan, src_arr_f)

    # 3) Reproject with mean resampling (each 10 m cell becomes the mean of
    #    its ~25 underlying 2 m cells, same operation R's `terra::extract`
    #    does at polygon level).
    dst_arr = np.full((nrow, ncol), np.nan, dtype=np.float32)
    reproject(
        source=src_arr_f,
        destination=dst_arr,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs="EPSG:32632",
        resampling=Resampling.average,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    nvalid = int(np.isfinite(dst_arr).sum())
    if nvalid == 0:
        print("  [WARN] no valid pot_risk cells in destination grid")
        return None, None
    print(f"  pot_risk valid cells: {nvalid:,} "
          f"({100 * nvalid / (nrow * ncol):.1f}%); "
          f"mean={np.nanmean(dst_arr):.3f} t/ha/y  "
          f"p98={np.nanpercentile(dst_arr, 98):.2f} t/ha/y")

    # 4) Optionally cache.
    if cache:
        try:
            os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
            with rasterio.open(cache, "w", driver="GTiff",
                               height=nrow, width=ncol, count=1,
                               dtype="float32", crs="EPSG:32632",
                               transform=dst_transform, nodata=np.nan,
                               compress="deflate") as dst:
                dst.write(dst_arr, 1)
            print(f"  pot_risk cache written: {cache}")
        except Exception as e:
            print(f"  [WARN] could not write cache {cache}: {e}")

    return dst_arr, dst_transform


def _sample_pot_risk(x, y, arr, transform) -> np.ndarray:
    """Vectorised lookup of pot_risk at (x, y) in EPSG:32632, nearest cell."""
    x = np.asarray(x); y = np.asarray(y)
    cols = np.floor((x - transform.c) / transform.a).astype(np.int64)
    rows = np.floor((transform.f - y) / (-transform.e)).astype(np.int64)
    nrow, ncol = arr.shape
    mask = (rows >= 0) & (rows < nrow) & (cols >= 0) & (cols < ncol)
    out = np.full(len(x), np.nan, dtype=np.float32)
    out[mask] = arr[rows[mask], cols[mask]]
    return out


def _per_polygon_pot_risk(cfg: dict, year: int, arr, transform) -> dict:
    """{poly_id: mean_pot_risk} from sampling at empirical pixel centers."""
    pix = load_pixels("empirical", cfg, year)
    if pix is None or pix.empty:
        return {}
    pot = _sample_pot_risk(pix["x"].values, pix["y"].values, arr, transform)
    df = pd.DataFrame({"poly_id": pix["poly_id"].values, "pot_risk": pot})
    g = df.dropna().groupby("poly_id")["pot_risk"].mean()
    return g.to_dict()


# ===========================================================================
# Crop labelling: produce {poly_id -> crop_label} respecting both modes
# ===========================================================================

def _crop_by_poly_for_year(year: int, cfg: dict,
                            lnf_bridge_y: pd.DataFrame | None,
                            crop_by_lnf: dict,
                            crop_by_poly: dict) -> dict:
    """Single ``{poly_id: crop_label}`` dict for one year, honouring
    ``cfg['crop_label_method']`` and falling back through the kulturmapping
    bridge for unlabelled polygons. Mirrors ``_label_field_crops`` from
    compare_products but flattened to a dict for direct .map() use.
    """
    method = cfg.get("crop_label_method", "bridge")
    out: dict = {}
    if method == "lnf":
        primary = (crop_by_poly or {}).get(year, {})
        for pid, lbl in primary.items():
            if lbl and str(lbl).strip().lower() not in ("", "nan"):
                out[int(pid)] = str(lbl).strip()
    if lnf_bridge_y is None or lnf_bridge_y.empty:
        return out
    # Fill remaining polygons via lnf_code -> crop_kat
    for r in lnf_bridge_y.itertuples():
        pid = int(r.poly_id)
        if pid in out:
            continue
        code = getattr(r, "lnf_code", None)
        if code is not None and pd.notna(code):
            lbl = crop_by_lnf.get(int(code))
            if lbl and str(lbl).strip().lower() not in ("", "nan"):
                out[pid] = str(lbl).strip()
    return out


# ===========================================================================
# Per-product actual risk (per polygon)
# ===========================================================================

def _actual_risk_pixel_product(product_key: str, cfg: dict, year: int,
                                arr, transform, p_factor: float,
                                lnf_bridge_y: pd.DataFrame | None
                                ) -> pd.DataFrame | None:
    """Per-polygon actual risk for emp/ML:
        actual_risk = mean(C * pot_risk) * P    over the polygon's 10 m pixels.
    """
    pix = load_pixels(product_key, cfg, year)
    if pix is None or pix.empty:
        return None
    # pyarrow returns hive-partition columns (lnf_code, sometimes poly_id) as
    # Dictionary/Categorical. That breaks groupby(..., as_index=False).agg via
    # `_insert_inaxis_grouper`. Coerce to plain numeric and use as_index=True +
    # reset_index() to avoid that code path entirely.
    for col in ("poly_id", "lnf_code"):
        if col in pix.columns:
            pix[col] = pd.to_numeric(pix[col], errors="coerce").astype("Int64")
    pot = _sample_pot_risk(pix["x"].values, pix["y"].values, arr, transform)
    pix = pix.assign(pot_risk=pot)
    pix = pix.dropna(subset=["pot_risk", "c_factor", "poly_id", "lnf_code"])
    if pix.empty:
        return None
    pix["cpot"] = pix["c_factor"].astype(np.float32) * pix["pot_risk"]
    g = (pix.groupby(["poly_id", "lnf_code"])
            .agg(n_pixels=("c_factor", "size"),
                 c_mean=("c_factor", "mean"),
                 pot_mean=("pot_risk", "mean"),
                 cpot_mean=("cpot", "mean"))
            .reset_index())
    g["area_m2"] = g["n_pixels"] * PIX_RES ** 2
    g["actual_risk"] = g["cpot_mean"] * float(p_factor)
    # Effective amplification per polygon: actual_risk / pot_mean = (mean(C*pot)
    # / mean(pot)) * P. For emp/ML it differs from mean(C)*P when C and pot
    # co-vary inside the polygon; for R it equals C * P (C constant inside).
    g["amplification"] = np.where(
        g["pot_mean"] > 0, g["actual_risk"] / g["pot_mean"], np.nan)
    g["year"] = year
    g["product"] = product_key
    if lnf_bridge_y is not None and not lnf_bridge_y.empty:
        meta = lnf_bridge_y[["poly_id", "betr_ID"]].drop_duplicates("poly_id")
        g = g.merge(meta, on="poly_id", how="left")
    else:
        g["betr_ID"] = pd.NA
    return g


def _actual_risk_r(cfg: dict, year: int,
                   lnf_bridge_y: pd.DataFrame | None,
                   pot_risk_by_poly: dict,
                   crop_for_y: dict,
                   p_factor: float) -> pd.DataFrame | None:
    """Per-polygon actual risk for R:
        actual_risk = pot_risk_poly * C_R * P
    where pot_risk_poly is the same per-polygon mean used for emp/ML (so any
    cross-product diff is only from C). C_R is the area-weighted (betr_ID, crop)
    mean of C_fact_detail from R's CSV, attached via the LNF bridge.
    """
    r = load_r_results(cfg, year)
    if r is None or r.empty or lnf_bridge_y is None:
        return None

    r2 = r.copy()
    r2["_wC"] = r2["C_fact_detail"] * r2["Flaeche"]
    cR = (r2.groupby(["betr_ID", "crop"], as_index=False)
             .agg(_wC=("_wC", "sum"), Flaeche=("Flaeche", "sum")))
    cR["C_R"] = cR["_wC"] / cR["Flaeche"]

    br = lnf_bridge_y[["poly_id", "betr_ID", "lnf_code"]].copy()
    br["crop"] = br["poly_id"].map(crop_for_y)
    br = br.dropna(subset=["crop", "betr_ID"])

    br = br.merge(cR[["betr_ID", "crop", "C_R"]],
                  on=["betr_ID", "crop"], how="inner")
    br["pot_mean"] = br["poly_id"].map(pot_risk_by_poly)
    br = br.dropna(subset=["pot_mean"])

    br["c_mean"] = br["C_R"]
    br["cpot_mean"] = br["C_R"] * br["pot_mean"]
    br["actual_risk"] = br["cpot_mean"] * float(p_factor)
    br["amplification"] = br["c_mean"] * float(p_factor)
    br["year"] = year
    br["product"] = "previous"
    br["n_pixels"] = pd.NA
    if "lnf_area_m2" in lnf_bridge_y.columns:
        a_map = lnf_bridge_y.set_index("poly_id")["lnf_area_m2"].to_dict()
        br["area_m2"] = br["poly_id"].map(a_map)
    else:
        br["area_m2"] = pd.NA
    return br[["poly_id", "lnf_code", "betr_ID", "n_pixels", "area_m2",
               "c_mean", "pot_mean", "cpot_mean",
               "actual_risk", "amplification", "year", "product"]]


# ===========================================================================
# Per-crop and per-tillage bar charts
# ===========================================================================

def _actualrisk_per_crop(frames: dict, crop_for_y: dict, cfg: dict,
                          year: int, summ: Summary) -> None:
    """Two per-crop bar charts (mean actual risk, mean C*P amplification),
    3 products side-by-side. Crops ranked by total area across all products."""
    rows = []
    for k, d in frames.items():
        d2 = d.copy()
        d2["crop"] = d2["poly_id"].map(crop_for_y)
        d2 = d2.dropna(subset=["crop"])
        if d2.empty:
            continue
        w = pd.to_numeric(d2["area_m2"], errors="coerce")
        w = w.where(w > 0, 1.0).fillna(1.0)
        d2["_w"] = w
        g = (d2.groupby("crop", as_index=False)
                .apply(lambda dd: pd.Series({
                    "mean_actual_risk":
                        float(np.average(dd["actual_risk"], weights=dd["_w"])),
                    "mean_amplification":
                        float(np.average(dd["amplification"].fillna(0),
                                         weights=dd["_w"])),
                    "mean_pot_risk":
                        float(np.average(dd["pot_mean"], weights=dd["_w"])),
                    "area_ha": float(dd["_w"].sum() / 1e4),
                    "n_poly":  int(len(dd)),
                }))
                .reset_index(drop=True))
        g["product"] = k
        rows.append(g)
    if not rows:
        summ.add("  [skip] no per-crop rows")
        return
    df = pd.concat(rows, ignore_index=True)
    _savecsv(df, f"actualrisk_{year}_by_crop.csv", cfg)

    area_per_crop = df.groupby("crop")["area_ha"].max()
    top_n = cfg.get("actualrisk_top_n_crops") or cfg["top_n_crops"]
    top_crops = (area_per_crop.sort_values(ascending=False).head(top_n).index
                 if not cfg.get("show_all_crops", False)
                 else area_per_crop.sort_values(ascending=False).index)
    prods_present = [p for p in ("empirical", "ml", "previous")
                     if p in df["product"].unique()]
    if not prods_present:
        return

    def _bar(metric, xlabel, fname, title):
        wide = df.pivot_table(index="crop", columns="product", values=metric)
        wide = wide.loc[[c for c in top_crops if c in wide.index],
                        prods_present]
        n_crops, n_prods = len(wide), len(prods_present)
        fig, ax = plt.subplots(figsize=(8, max(3.2, 0.36 * n_crops)))
        ypos = np.arange(n_crops); bw = 0.8 / max(1, n_prods)
        for i, p in enumerate(prods_present):
            off = (i - (n_prods - 1) / 2) * bw
            ax.barh(ypos + off, wide[p].to_numpy(), height=bw,
                    color=PROD_COLORS[p], label=PROD_LABELS[p])
        ax.set_yticks(ypos); ax.set_yticklabels(wide.index, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=8)
        ax.axvline(0, color="k", lw=0.6)
        fig.tight_layout()
        savefig(fig, fname, cfg)

    _bar("mean_actual_risk",
         f"mean actual risk (t ha$^{{-1}}$ y$^{{-1}}$), area-weighted",
         f"actualrisk_{year}_by_crop.png",
         f"Mean actual erosion risk per crop, {year}\n"
         f"(pot_risk x C x P={cfg['p_factor']}; top {len(top_crops)} crops by area)")
    _bar("mean_amplification",
         f"mean C x P (= actual / potential), area-weighted",
         f"actualrisk_{year}_amplification_by_crop.png",
         f"Mean C-factor amplification per crop, {year}\n"
         f"(higher = C-factor exacerbates pot_risk more; P={cfg['p_factor']})")

    # Where do products disagree most per crop?
    if {"empirical", "ml", "previous"}.issubset(df["product"].unique()):
        wide_amp = df.pivot_table(index="crop", columns="product",
                                  values="mean_amplification")
        wide_amp = wide_amp.loc[[c for c in top_crops if c in wide_amp.index]]
        wide_amp["spread"] = wide_amp.max(axis=1) - wide_amp.min(axis=1)
        worst = wide_amp.sort_values("spread", ascending=False).head(5)
        summ.add(f"  crops with biggest C*P spread across products ({year}):")
        for crop, r in worst.iterrows():
            summ.add(f"    {str(crop)[:40]:<40} "
                     f"emp={r.get('empirical', np.nan):.3f}  "
                     f"ml={r.get('ml', np.nan):.3f}  "
                     f"prev={r.get('previous', np.nan):.3f}  "
                     f"(spread={r['spread']:.3f})")

    # Per-crop dRisk vs the previous (R) product -- which crops do emp/ML
    # disagree with R on most in actual t ha^-1 y^-1?
    _actualrisk_crop_diff_vs_previous(df, cfg, year, summ)


def _actualrisk_crop_diff_vs_previous(df: pd.DataFrame, cfg: dict,
                                       year: int, summ: Summary,
                                       top_n_show: int = 15,
                                       min_area_ha: float = 5.0) -> None:
    """Per-crop signed actual-risk difference vs the PREVIOUS (R) product.

    Distinct from the C*P-spread block above: that one summarises divergence
    in the amplification factor (unit-free). This one summarises divergence
    in actual erosion (t ha^-1 y^-1), which couples C with the typical
    pot_risk of polygons in that crop -- a 0.05 C*P difference on a 10 t/ha
    pot_risk crop matters far more than the same C*P difference on a 1 t/ha
    crop, and only this metric makes that visible.

    Diff convention: ``product - previous``
        + : product predicts more erosion than R for this crop.
        - : product predicts less.

    Crops with total area < ``min_area_ha`` are dropped (noisy means).
    Ranking is by max(|d_empirical|, |d_ml|) so a crop ranks if either pixel
    product disagrees with R strongly. CSV is full table (all qualifying
    crops); plot shows top ``top_n_show``.
    """
    prods = set(df["product"].unique())
    if "previous" not in prods:
        summ.add("  [skip] crop dRisk vs previous: R product unavailable")
        return
    pixel_prods = [p for p in ("empirical", "ml") if p in prods]
    if not pixel_prods:
        return

    wide = df.pivot_table(index="crop", columns="product",
                          values="mean_actual_risk")
    area_per_crop = df.groupby("crop")["area_ha"].max()
    n_poly = df.groupby("crop")["n_poly"].max()
    wide["area_ha"] = area_per_crop
    wide["n_poly"] = n_poly
    wide = wide[wide["area_ha"] >= float(min_area_ha)]
    if wide.empty:
        summ.add(f"  [skip] crop dRisk: no crops with area >= {min_area_ha} ha")
        return

    diff_cols = []
    for p in pixel_prods:
        if p in wide.columns:
            col = f"dRisk_{p}_minus_prev"
            wide[col] = wide[p] - wide["previous"]
            # Relative: only where prev > a tiny epsilon, else NaN.
            prev = wide["previous"].astype(float)
            wide[f"{col}_pct"] = np.where(
                prev > 0.05, 100.0 * wide[col] / prev, np.nan)
            diff_cols.append((p, col))
    if not diff_cols:
        return

    wide["max_abs_dRisk"] = wide[[c for _, c in diff_cols]].abs().max(axis=1)
    out = wide.sort_values("max_abs_dRisk", ascending=False).reset_index()
    _savecsv(out, f"actualrisk_{year}_crop_diff_vs_previous.csv", cfg)

    # Summary: top crops per pair (ranked by |d| for that pair specifically).
    for p, col in diff_cols:
        s = wide[col].dropna()
        if s.empty:
            continue
        ranked = s.reindex(s.abs().sort_values(ascending=False).index)
        summ.add(f"  crops with biggest actual-risk diff "
                 f"{PROD_LABELS[p]} - {PROD_LABELS['previous']} ({year}):")
        for crop in ranked.head(top_n_show).index:
            r = wide.loc[crop]
            pct = r.get(f"{col}_pct", np.nan)
            pct_str = f"{pct:+.0f}%" if pd.notna(pct) else "  n/a"
            summ.add(f"    {str(crop)[:38]:<38} "
                     f"prev={r['previous']:.3f}  "
                     f"{p[:3]}={r[p]:.3f}  "
                     f"d={r[col]:+.3f} t/ha/y ({pct_str})  "
                     f"area={r['area_ha']:,.0f} ha (n={int(r['n_poly'])})")

    # Area-weighted regional mean dRisk per pair (across ALL qualifying crops,
    # not just the top-N shown). Equivalent to the polygon-level area-weighted
    # mean dRisk over the region, restricted to crops above the area floor.
    overall_mean = {}
    for p, col in diff_cols:
        s = wide[col].dropna()
        if s.empty:
            continue
        a = wide.loc[s.index, "area_ha"].astype(float)
        if a.sum() > 0:
            overall_mean[(p, col)] = float(np.average(s.to_numpy(),
                                                       weights=a.to_numpy()))
        else:
            overall_mean[(p, col)] = float(s.mean())
        summ.add(f"  regional area-weighted mean dRisk "
                 f"{PROD_LABELS[p]} - {PROD_LABELS['previous']} ({year}): "
                 f"{overall_mean[(p, col)]:+.3f} t/ha/y "
                 f"(over {int(s.shape[0])} crops, {a.sum():,.0f} ha)")

    # Plot: top crops by max|d| across both pixel products.
    top_crops_plot = out["crop"].head(top_n_show).tolist()
    sub = wide.reindex(top_crops_plot)
    fig, ax = plt.subplots(figsize=(9, max(3.2, 0.4 * len(sub))))
    ypos = np.arange(len(sub))
    bw = 0.8 / max(1, len(diff_cols))
    for i, (p, col) in enumerate(diff_cols):
        off = (i - (len(diff_cols) - 1) / 2) * bw
        mv = overall_mean.get((p, col))
        label = f"{PROD_LABELS[p]} - {PROD_LABELS['previous']}"
        if mv is not None:
            label += f"  (region mean = {mv:+.2f})"
        ax.barh(ypos + off, sub[col].to_numpy(), height=bw,
                color=PROD_COLORS[p], label=label)
    ax.set_yticks(ypos)
    ax.set_yticklabels(sub.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(f"actual risk difference (t ha$^{{-1}}$ y$^{{-1}}$)")
    ax.set_title(f"Crops with largest actual-risk disagreement vs "
                 f"{PROD_LABELS['previous']}, {year}\n"
                 f"(+ = product predicts more erosion than R; top "
                 f"{len(sub)} crops by max|diff|, area >= {min_area_ha:g} ha)")
    ax.axvline(0, color="k", lw=0.6)

    # Overlay regional area-weighted mean dRisk per pair as a dashed vertical
    # line in the matching colour.
    drew_mean_line = False
    for p, col in diff_cols:
        if (p, col) not in overall_mean:
            continue
        ax.axvline(overall_mean[(p, col)], color=PROD_COLORS[p],
                   linestyle="--", lw=1.2, alpha=0.85)
        drew_mean_line = True

    # Legend: bar handles + a single neutral proxy explaining the dashed lines.
    handles, labels = ax.get_legend_handles_labels()
    if drew_mean_line:
        from matplotlib.lines import Line2D
        handles.append(Line2D([0], [0], color="0.4", linestyle="--", lw=1.2,
                              label="region area-wtd mean"))
        labels.append("region area-wtd mean")
    ax.legend(handles=handles, labels=labels, loc="lower right", fontsize=8)
    fig.tight_layout()
    savefig(fig, f"actualrisk_{year}_crop_diff_vs_previous.png", cfg)


def _actualrisk_per_tillage(frames: dict, crop_for_y: dict,
                             tillage_map_y: dict, cfg: dict,
                             year: int, summ: Summary) -> None:
    """Mean actual risk + mean amplification by tillage class (Pflug/Mulch),
    3 products. Uses R's (betr_ID, crop) -> tillage map for grouping all three.
    """
    if not tillage_map_y:
        return
    rows = []
    for k, d in frames.items():
        d2 = d.copy()
        d2["crop"] = d2["poly_id"].map(crop_for_y)
        d2 = d2.dropna(subset=["crop", "betr_ID"])
        d2["tillage"] = [tillage_map_y.get((b, c))
                         for b, c in zip(d2["betr_ID"], d2["crop"])]
        d2 = d2[d2["tillage"].isin(TILLAGE_ORDER)]
        if d2.empty:
            continue
        w = pd.to_numeric(d2["area_m2"], errors="coerce")
        w = w.where(w > 0, 1.0).fillna(1.0)
        d2["_w"] = w
        g = (d2.groupby("tillage", as_index=False)
                .apply(lambda dd: pd.Series({
                    "mean_actual_risk":
                        float(np.average(dd["actual_risk"], weights=dd["_w"])),
                    "mean_amplification":
                        float(np.average(dd["amplification"].fillna(0),
                                         weights=dd["_w"])),
                    "area_ha": float(dd["_w"].sum() / 1e4),
                    "n_poly":  int(len(dd)),
                }))
                .reset_index(drop=True))
        g["product"] = k
        rows.append(g)
    if not rows:
        return
    df = pd.concat(rows, ignore_index=True)
    _savecsv(df, f"actualrisk_{year}_by_tillage.csv", cfg)

    prods_present = [p for p in ("empirical", "ml", "previous")
                     if p in df["product"].unique()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, metric, ylabel in (
        (axes[0], "mean_actual_risk",
         f"actual risk (t ha$^{{-1}}$ y$^{{-1}}$)"),
        (axes[1], "mean_amplification",
         f"C x P (= actual / potential)"),
    ):
        x = np.arange(len(TILLAGE_ORDER))
        bw = 0.8 / max(1, len(prods_present))
        for i, p in enumerate(prods_present):
            vals, ns = [], []
            for t in TILLAGE_ORDER:
                row = df[(df["product"] == p) & (df["tillage"] == t)]
                vals.append(float(row[metric].iloc[0]) if not row.empty else np.nan)
                ns.append(int(row["n_poly"].iloc[0]) if not row.empty else 0)
            off = (i - (len(prods_present) - 1) / 2) * bw
            bars = ax.bar(x + off, vals, bw,
                          color=PROD_COLORS[p], label=PROD_LABELS[p])
            for k, b in enumerate(bars):
                h = b.get_height()
                if not np.isnan(h):
                    ax.text(b.get_x() + b.get_width() / 2,
                            h, f"n={ns[k]}", ha="center",
                            va="bottom", fontsize=6, color="#444444")
        ax.set_xticks(x); ax.set_xticklabels(TILLAGE_ORDER)
        ax.set_ylabel(ylabel)
        ax.axhline(0, color="k", lw=0.6)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"Actual risk by tillage class, {year}  "
                 f"(R's tillage attribution used to group all 3 products)")
    fig.tight_layout()
    savefig(fig, f"actualrisk_{year}_by_tillage.png", cfg)

    for p in prods_present:
        sub = df[df["product"] == p].set_index("tillage").reindex(TILLAGE_ORDER)
        if sub["mean_amplification"].notna().sum() == 2:
            d_amp = (sub.loc["Pflug", "mean_amplification"]
                     - sub.loc["Mulch", "mean_amplification"])
            d_risk = (sub.loc["Pflug", "mean_actual_risk"]
                      - sub.loc["Mulch", "mean_actual_risk"])
            summ.add(f"  {PROD_LABELS[p]:<18} Pflug - Mulch: "
                     f"d(C*P)={d_amp:+.4f}  d(actual_risk)={d_risk:+.3f} t/ha/y")


# ===========================================================================
# Drivers of actual risk: topography vs management (variance decomposition)
# ===========================================================================

def _actualrisk_drivers(frames: dict, cfg: dict, year: int,
                        summ: Summary) -> None:
    """Decompose what drives the per-polygon spread in actual erosion risk.

    Identity (per polygon, per product):
        actual_risk = pot_mean * amplification        [exact, by construction]
    Taking logs:
        log(AR) = log(pot_mean) + log(amplification)
        var(log AR) = var(log pot) + var(log amp) + 2*cov(log pot, log amp)
    So three shares (summing to 1) describe what drives the spread of actual
    risk *between fields* under each product:

        topography  : var(log pot_mean)  / var(log AR)   -- same baseline for
                                                            all 3 products by
                                                            construction, so
                                                            the absolute number
                                                            here is identical
                                                            across products.
        management  : var(log amp)       / var(log AR)
        covariation : 2*cov(log pot, log amp) / var(log AR)
                       > 0 : product places HIGHER C on already-high-pot
                             fields (compounds topographic risk)
                       < 0 : product places LOWER  C there (compensates)

    Between-product differences therefore live in the management and
    covariation shares -- the topography share moves only because var(log AR)
    differs (denominator changes).

    Also reports a univariate linear R^2 of AR ~ pot_risk and AR ~ amp per
    product as an interpretable readout ("topography alone explains X%").
    """
    if not frames:
        return

    rows = []
    # Iterate in the conventional order for stable plot/CSV ordering.
    for k in ("empirical", "ml", "previous"):
        if k not in frames:
            continue
        d = frames[k]
        m = pd.DataFrame({
            "pot_mean":     pd.to_numeric(d["pot_mean"], errors="coerce"),
            "amplification": pd.to_numeric(d["amplification"], errors="coerce"),
            "actual_risk":  pd.to_numeric(d["actual_risk"], errors="coerce"),
        }).dropna()
        # log requires strictly positive; drop the (rare) non-positive rows.
        m = m[(m["pot_mean"] > 0) & (m["amplification"] > 0)
              & (m["actual_risk"] > 0)]
        if len(m) < 10:
            continue

        log_pot = np.log(m["pot_mean"].to_numpy())
        log_amp = np.log(m["amplification"].to_numpy())
        log_ar  = np.log(m["actual_risk"].to_numpy())
        v_pot = float(np.var(log_pot, ddof=0))
        v_amp = float(np.var(log_amp, ddof=0))
        v_ar  = float(np.var(log_ar,  ddof=0))
        cv    = float(np.cov(log_pot, log_amp, ddof=0)[0, 1])
        share_topo = v_pot / v_ar if v_ar > 0 else np.nan
        share_mgmt = v_amp / v_ar if v_ar > 0 else np.nan
        share_cov  = 2 * cv / v_ar if v_ar > 0 else np.nan

        # Linear univariate R^2 ("if you knew only X, how well could you
        # predict AR?").
        pot = m["pot_mean"].to_numpy()
        amp = m["amplification"].to_numpy()
        ar  = m["actual_risk"].to_numpy()
        r2_pot_lin = (float(np.corrcoef(pot, ar)[0, 1]) ** 2
                       if np.var(pot) > 0 else np.nan)
        r2_amp_lin = (float(np.corrcoef(amp, ar)[0, 1]) ** 2
                       if np.var(amp) > 0 else np.nan)
        corr_pot_amp = (float(np.corrcoef(pot, amp)[0, 1])
                        if np.var(pot) > 0 and np.var(amp) > 0 else np.nan)

        rows.append({
            "product": k,
            "n_poly": int(len(m)),
            "var_log_pot": v_pot,
            "var_log_amp": v_amp,
            "var_log_ar":  v_ar,
            "cov_log_pot_amp": cv,
            "share_topography": share_topo,
            "share_management": share_mgmt,
            "share_covariation": share_cov,
            "r2_pot_linear": r2_pot_lin,
            "r2_amp_linear": r2_amp_lin,
            "corr_pot_amp_linear": corr_pot_amp,
        })
    if not rows:
        summ.add("  [skip] drivers: no usable polygons")
        return

    out = pd.DataFrame(rows)
    _savecsv(out, f"actualrisk_{year}_drivers.csv", cfg)

    # Summary text.
    summ.add(f"\n  drivers of actual risk ({year}) -- variance decomposition "
             f"on log scale")
    summ.add(f"    identity: var(log AR) = var(log pot) + var(log amp) "
             f"+ 2*cov(log pot, log amp)   (shares sum to 1)")
    for r in rows:
        summ.add(f"    {PROD_LABELS[r['product']]:<18} "
                 f"share: topo={r['share_topography']:+.1%}  "
                 f"mgmt={r['share_management']:+.1%}  "
                 f"cov={r['share_covariation']:+.1%}  | "
                 f"corr(pot,amp)={r['corr_pot_amp_linear']:+.3f}  "
                 f"R^2 lin: AR~pot={r['r2_pot_linear']:.3f}  "
                 f"AR~amp={r['r2_amp_linear']:.3f}")

    # ---------- plot ----------
    prods_ordered = [r["product"] for r in rows]
    x = np.arange(len(prods_ordered))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2),
                                    gridspec_kw={"width_ratios": [1.4, 1]})

    # Left: clustered bars showing the three variance shares per product.
    # Three categories: topo / mgmt / cov; one bar per product.
    cats = ["topography", "management", "covariation"]
    cat_keys = ["share_topography", "share_management", "share_covariation"]
    bw = 0.8 / len(prods_ordered)
    xc = np.arange(len(cats))
    for i, p in enumerate(prods_ordered):
        r = next(rr for rr in rows if rr["product"] == p)
        vals = [r[k] for k in cat_keys]
        off = (i - (len(prods_ordered) - 1) / 2) * bw
        ax1.bar(xc + off, vals, bw, color=PROD_COLORS[p],
                label=PROD_LABELS[p], edgecolor="white", linewidth=0.5)
    ax1.set_xticks(xc)
    ax1.set_xticklabels([
        "topography\nvar(log pot)",
        "management\nvar(log C*P)",
        "covariation\n2*cov(log pot, log C*P)",
    ], fontsize=8)
    ax1.set_ylabel("share of var(log actual_risk)")
    ax1.axhline(0, color="k", lw=0.6)
    ax1.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax1.set_title(f"What drives between-field spread in actual risk? {year}\n"
                  f"(topography baseline shared across products; differences "
                  f"live in mgmt + cov)")
    ax1.legend(loc="best", fontsize=8, frameon=False)

    # Right: linear R^2 bars (univariate fit "AR ~ pot" and "AR ~ amp"
    # per product).
    bw2 = 0.4
    r2_pot = [next(rr for rr in rows if rr["product"] == p)["r2_pot_linear"]
              for p in prods_ordered]
    r2_amp = [next(rr for rr in rows if rr["product"] == p)["r2_amp_linear"]
              for p in prods_ordered]
    ax2.bar(x - bw2/2, r2_pot, bw2, color="#888888",
            label=r"$R^2$: AR ~ pot_risk", edgecolor="white")
    ax2.bar(x + bw2/2, r2_amp, bw2,
            color=[PROD_COLORS[p] for p in prods_ordered],
            label=r"$R^2$: AR ~ C*P", edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels([PROD_LABELS[p] for p in prods_ordered], fontsize=8)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel(r"$R^2$ (linear, univariate)")
    ax2.set_title(f"How well does each driver alone predict AR? {year}")
    ax2.legend(loc="best", fontsize=8, frameon=False)

    fig.tight_layout()
    savefig(fig, f"actualrisk_{year}_drivers.png", cfg)


# ===========================================================================
# Per-year orchestration
# ===========================================================================

def compare_actual_risk(cfg: dict, year: int,
                         lnf_bridge: dict,
                         crop_by_lnf: dict,
                         crop_by_poly: dict,
                         centroid_by_poly: dict,
                         tillage_map: dict | None,
                         pot_risk_arr,
                         pot_risk_transform,
                         summ: Summary,
                         make_maps: bool) -> pd.DataFrame | None:
    """Three-way per-polygon comparison of actual erosion risk for one year."""
    summ.add(f"\n-- Actual erosion risk, {year}  "
             f"(pot_risk x C x P={cfg['p_factor']}, units: t ha^-1 y^-1) --")
    if pot_risk_arr is None:
        summ.add("  [skip] pot_risk raster not available")
        return None
    lnf_y = lnf_bridge.get(year)

    # 1) Per-polygon pot_risk (shared baseline for all 3 products).
    pot_by_poly = _per_polygon_pot_risk(cfg, year, pot_risk_arr,
                                         pot_risk_transform)
    if not pot_by_poly:
        summ.add("  [skip] no per-polygon pot_risk (raster empty in region?)")
        return None
    summ.add(f"  per-polygon pot_risk: n={len(pot_by_poly):,}  "
             f"mean={np.mean(list(pot_by_poly.values())):.3f}  "
             f"median={np.median(list(pot_by_poly.values())):.3f}  "
             f"p98={np.percentile(list(pot_by_poly.values()), 98):.2f} t/ha/y")

    # Baseline (pot_risk) gets its own colour scale calibrated to pot_risk's
    # own p98 -- pot_risk is the unmodified soil-loss signal and dominates the
    # value range.
    pot_vals = np.asarray(list(pot_by_poly.values()))
    baseline_vmax = float(np.nanpercentile(pot_vals, 98))

    # Baseline map: potential erosion risk per field (no C, no P).
    if make_maps and centroid_by_poly:
        _field_value_map(pot_by_poly, centroid_by_poly, cfg,
                          title=f"Potential erosion risk",
                          label="potential risk (t ha$^{-1}$ y$^{-1}$)\n"
                                "Erosionsrisikokarte des Ackerlandes (2 m)",
                          fname=f"potential_{year}_fieldmap.png",
                          vmin=0, vmax=baseline_vmax, cmap="OrRd")

    # 2) Unified crop labels for this year (honours crop_label_method, with
    #    fallback through the kulturmapping bridge).
    crop_for_y = _crop_by_poly_for_year(year, cfg, lnf_y, crop_by_lnf,
                                          crop_by_poly)

    # 3) Per-product per-polygon actual risk.
    frames: dict[str, pd.DataFrame] = {}
    for k in ("empirical", "ml"):
        df = _actual_risk_pixel_product(k, cfg, year, pot_risk_arr,
                                         pot_risk_transform,
                                         cfg["p_factor"], lnf_y)
        if df is not None and not df.empty:
            frames[k] = df
    if cfg.get("_use_r", True):
        rfr = _actual_risk_r(cfg, year, lnf_y, pot_by_poly, crop_for_y,
                              cfg["p_factor"])
        if rfr is not None and not rfr.empty:
            frames["previous"] = rfr

    if not frames:
        summ.add("  [skip] no products usable")
        return None

    for k, d in frames.items():
        v = pd.to_numeric(d["actual_risk"], errors="coerce").dropna()
        a = pd.to_numeric(d["amplification"], errors="coerce").dropna()
        summ.add(f"  {PROD_LABELS.get(k, k):<18} "
                 f"actual_risk: mean={v.mean():.3f}  median={v.median():.3f}  "
                 f"p98={v.quantile(.98):.2f} t/ha/y  "
                 f"(C*P mean={a.mean():.4f}; n_poly={len(d):,})")

    # 4) Per-field map per product (sequential colour, shared scale across the
    #    three products so they're directly comparable to each other). The scale
    #    is calibrated to actual_risk's joint p98, NOT pot_risk's -- pot_risk is
    #    typically 5-10x larger than actual_risk (because C*P << 1), so using
    #    the pot_risk scale would compress every product map into the bottom of
    #    the colour ramp and hide all between-product variation. Read the
    #    baseline and the actual-risk maps as two different things: baseline =
    #    "where is the topographic risk?", actual = "after C and P, who agrees
    #    with whom and where?".
    if make_maps and centroid_by_poly:
        all_actual = pd.concat(
            [pd.to_numeric(d["actual_risk"], errors="coerce")
             for d in frames.values()]).dropna()
        actual_vmax = (float(np.nanpercentile(all_actual, 98))
                       if len(all_actual) else 1.0)
        summ.add(f"  map scales: baseline p98={baseline_vmax:.2f}  "
                 f"actual (shared) p98={actual_vmax:.3f} t/ha/y")
        for k, d in frames.items():
            series_by_poly = dict(zip(d["poly_id"].astype("int64"),
                                       pd.to_numeric(d["actual_risk"],
                                                     errors="coerce")))
            _field_value_map(series_by_poly, centroid_by_poly, cfg,
                              title=f"Actual erosion risk - "
                                    f"{PROD_LABELS.get(k, k)}, {year}",
                              label="actual risk (t ha$^{-1}$ y$^{-1}$)\n"
                                    "= pot_risk x C x P",
                              fname=f"actualrisk_{year}_{k}_fieldmap.png",
                              vmin=0, vmax=actual_vmax, cmap="OrRd")

    # 5) Wide table: one row per polygon, columns per product.
    wide = None
    for k, d in frames.items():
        keep = d[["poly_id", "actual_risk", "amplification",
                   "pot_mean", "c_mean", "area_m2"]].rename(columns={
                       "actual_risk":   f"actual_{k}",
                       "amplification": f"amp_{k}",
                       "pot_mean":      f"pot_{k}",
                       "c_mean":        f"C_{k}",
                       "area_m2":       f"area_m2_{k}",
                   })
        wide = keep if wide is None else wide.merge(keep, on="poly_id",
                                                     how="outer")
    pot_cols = [c for c in wide.columns if c.startswith("pot_")]
    if pot_cols:
        wide["pot_risk"] = wide[pot_cols].mean(axis=1)
    if lnf_y is not None:
        meta = lnf_y[["poly_id", "betr_ID"]].drop_duplicates("poly_id")
        wide = wide.merge(meta, on="poly_id", how="left")
    wide["crop"] = wide["poly_id"].map(crop_for_y)
    if tillage_map and year in tillage_map:
        tm = tillage_map[year]
        wide["tillage"] = [tm.get((b, c)) if pd.notna(b) and pd.notna(c) else None
                            for b, c in zip(wide["betr_ID"], wide["crop"])]
    _savecsv(wide, f"actualrisk_field_table_{year}.csv", cfg)

    # 6) Pairwise differences + hotspots.
    pairs = []
    if "ml" in frames and "empirical" in frames:
        pairs.append(("ml", "empirical"))
    if "ml" in frames and "previous" in frames:
        pairs.append(("ml", "previous"))
    if "empirical" in frames and "previous" in frames:
        pairs.append(("empirical", "previous"))

    hotspots = []
    if pairs:
        pair_merges = {}
        dvals_all = []
        for (a, b) in pairs:
            mm = (frames[a][["poly_id", "actual_risk"]].rename(
                       columns={"actual_risk": f"r_{a}"})
                  .merge(
                       frames[b][["poly_id", "actual_risk"]].rename(
                           columns={"actual_risk": f"r_{b}"}),
                       on="poly_id", how="inner"))
            mm["dRisk"] = mm[f"r_{a}"] - mm[f"r_{b}"]
            pair_merges[(a, b)] = mm
            dvals_all.append(mm["dRisk"].to_numpy())
        vmax_shared = _joint_vmax(dvals_all, p=98, floor=0.5)

        for (a, b), mm in pair_merges.items():
            if mm.empty:
                continue
            bias = float(mm["dRisk"].mean())
            mabs = float(mm["dRisk"].abs().mean())
            corr = (float(np.corrcoef(mm[f"r_{a}"], mm[f"r_{b}"])[0, 1])
                    if len(mm) > 2 else np.nan)
            summ.add(f"  dRisk {PROD_LABELS[a]} - {PROD_LABELS[b]}:  "
                     f"bias={bias:+.3f}  MAE={mabs:.3f} t/ha/y  "
                     f"corr={corr:.3f}  n={len(mm):,}")
            if make_maps and centroid_by_poly:
                _field_diff_map(mm["poly_id"].to_numpy(),
                                 mm["dRisk"].to_numpy(),
                                 centroid_by_poly, cfg,
                                 title=f"Actual risk: {PROD_LABELS[a]} - "
                                       f"{PROD_LABELS[b]}, {year}",
                                 fname=f"actualrisk_{year}_{a}_vs_{b}_diff_fieldmap.png",
                                 direction=f"{PROD_LABELS[a]} - {PROD_LABELS[b]} "
                                           f"(t ha$^{{-1}}$ y$^{{-1}}$)",
                                 vmax_shared=vmax_shared)
                _field_absdiff_map(mm["poly_id"].to_numpy(),
                                    mm["dRisk"].to_numpy(),
                                    centroid_by_poly, cfg,
                                    title=f"|Actual risk diff|: "
                                          f"{PROD_LABELS[a]} vs "
                                          f"{PROD_LABELS[b]}, {year}",
                                    fname=f"actualrisk_{year}_{a}_vs_{b}_absdiff_fieldmap.png",
                                    direction=f"{PROD_LABELS[a]} - {PROD_LABELS[b]}",
                                    cmap="OrRd")
            top = (mm.assign(abs_dR=mm["dRisk"].abs())
                       .sort_values("abs_dR", ascending=False)
                       .head(20)
                       .copy())
            top["pair"] = f"{a}-{b}"
            top["crop"] = top["poly_id"].map(crop_for_y)
            if lnf_y is not None:
                bmap = lnf_y.set_index("poly_id")["betr_ID"].to_dict()
                top["betr_ID"] = top["poly_id"].map(bmap)
            hotspots.append(top)

    if hotspots:
        hot = pd.concat(hotspots, ignore_index=True)
        keep_cols = ["pair", "poly_id", "betr_ID", "crop", "dRisk", "abs_dR"]
        keep_cols += [c for c in hot.columns if c.startswith("r_")]
        _savecsv(hot[[c for c in keep_cols if c in hot.columns]],
                  f"actualrisk_{year}_hotspots.csv", cfg)

    # 7) Per-crop bars.
    _actualrisk_per_crop(frames, crop_for_y, cfg, year, summ)

    # 8) Per-tillage bars.
    if tillage_map and year in tillage_map and tillage_map[year]:
        _actualrisk_per_tillage(frames, crop_for_y, tillage_map[year],
                                 cfg, year, summ)

    # 9) Drivers: variance decomposition (topography vs management vs cov).
    _actualrisk_drivers(frames, cfg, year, summ)

    return wide


# ===========================================================================
# Main
# ===========================================================================

def _merge_centroids_local(lnf_bridge: dict, years) -> dict:
    """Union of per-year poly_id -> (cx, cy) EPSG:2056 centroids."""
    out = {}
    for y in years:
        lb = lnf_bridge.get(y)
        if lb is None:
            continue
        for r in lb.itertuples():
            out[int(r.poly_id)] = (r.cx, r.cy)
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Actual erosion risk per polygon (pot_risk x C x P), "
                    "three-way comparison.")
    ap.add_argument("--years", nargs="+", type=int, default=None)
    ap.add_argument("--region", type=str, default=None,
                    help="override region path")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="override output directory")
    ap.add_argument("--pot-risk-cache", type=str, default=None,
                    help="path to 10 m pot_risk GeoTIFF cache (read or write); "
                         "default = pot_risk_10m_utm32n.tif under out-dir")
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore the pot_risk cache (always re-read source raster)")
    ap.add_argument("--skip-maps", action="store_true",
                    help="skip the per-field maps (CSVs + bar charts only)")
    ap.add_argument("--skip-r", action="store_true",
                    help="skip the previous (R) product")
    ap.add_argument("--p-factor", type=float, default=None,
                    help=f"override P-factor (default {CONFIG['p_factor']})")
    a = ap.parse_args()

    cfg = dict(CONFIG)
    if a.years:
        cfg["years"] = a.years
    if a.region:
        cfg["region_path"] = a.region
    if a.out_dir:
        cfg["out_dir"] = a.out_dir
    if a.pot_risk_cache is not None:
        cfg["pot_risk_cache"] = a.pot_risk_cache
    if a.p_factor is not None:
        cfg["p_factor"] = float(a.p_factor)
    cfg["_use_r"] = not a.skip_r
    _ensure_dir(cfg["out_dir"])

    summ = Summary(cfg)
    summ.add(f"Actual erosion risk computation for years {cfg['years']}")
    summ.add(f"  empirical: {cfg['empirical_dir']}")
    summ.add(f"  ml:        {cfg['ml_dir']}")
    if cfg["_use_r"]:
        summ.add(f"  previous:  {cfg['erosion_results_dir']}")
    summ.add(f"  pot_risk:  {cfg['pot_risk_raster']}")
    summ.add(f"  P-factor:  {cfg['p_factor']}")

    # 1) Region + bridges (lifted from compare_products._build_bridges).
    region = gpd.read_file(_expand(cfg["region_path"]))
    region_2056 = region.to_crs("EPSG:2056").union_all()
    summ.add(f"  region area: {region_2056.area / 1e6:.1f} km^2")
    elev_by_poly, crop_by_lnf, crop_bridge, lnf_bridge, crop_by_poly = \
        _build_bridges(cfg, region_2056, cfg["years"])
    centroid_by_poly = _merge_centroids_local(lnf_bridge, cfg["years"])

    # 2) Tillage map (needs R) -- skipped if --skip-r.
    tillage_map = None
    if cfg["_use_r"]:
        summ.add("\nBuilding tillage map from R results ...")
        tillage_map = build_tillage_map(cfg, cfg["years"])

    # 3) Build (or read) the 10 m pot_risk grid once, reuse across years.
    summ.section("POT_RISK GRID")
    pot_arr, pot_tr = _build_pot_risk_grid(cfg, cfg["years"],
                                            use_cache=not a.no_cache)
    if pot_arr is None:
        summ.add("  [FATAL] no pot_risk grid -- aborting")
        summ.flush()
        return

    # 4) Per-year comparison.
    summ.section("ACTUAL EROSION RISK -- per year, three products")
    make_maps = not a.skip_maps
    for year in cfg["years"]:
        compare_actual_risk(cfg, year, lnf_bridge, crop_by_lnf, crop_by_poly,
                             centroid_by_poly, tillage_map,
                             pot_arr, pot_tr, summ, make_maps)

    # 5) Summary file -- write under a distinctive name so it doesn't collide
    #    with compare_products' summary in a shared out_dir.
    out_path = os.path.join(cfg["out_dir"], "actualrisk_summary.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(summ.lines) + "\n")
    print(f"\n  Summary written to {out_path}")
    print(f"\nDone. See {cfg['out_dir']}/")


if __name__ == "__main__":
    main()