"""
Compare the three C-factor products over a region (e.g. Seeland), 2021 vs 2022.

The products
------------
  1. EMPIRICAL  per-pixel parquet from ``compute_cfactor_pixels.py``
                (``output/cfactor_pixels``): C = Sum SLR.EI / Sum EI, SLR=exp(-b.FC)
  2. ML         per-pixel parquet from ``compute_cfactor_pixels_ml.py``
                (``output/cfactor_pixels_ml``): C = trained pipeline.predict(...)
  3. PREVIOUS   the R erosion product, per farm x crop
                (``*_Kultur_Betrieb_Erosionsrisiko_2x2_{year}.csv``):
                reported ``C_fact_detail`` (Prasuhn lookup x Region x tillage).

EMPIRICAL and ML share *exactly* the same 10 m pixel grid (the ML script reuses
the empirical Stage A), so they join pixel-for-pixel on (x, y). The R product has
no pixel geometry; it lives at (betr_ID, crop) granularity, so the pixel products
are rolled up to (betr_ID, crop) to meet it there.

What it answers
---------------
A. WITHIN a product, ACROSS the years  -> how stable is each product 2021->2022?
   A'. ALSO compares stability of all 3 products on common (betr_ID, crop).
B. WITHIN a year, ACROSS the products  -> where/why do they disagree
   (by crop, by elevation / Tal-Berg, by municipality), and which runs higher.
D. TILLAGE-STRATIFIED breakdown (Pflug vs Mulch).
E. FC TIME SERIES from extract_fc_pixels.py (the actual per-pixel gap-filled
   FC in the region), year-to-year overlay -- overall and small-multiples
   per crop. Context for why C may differ across years.
C. SYNTHESIS -> headline strengths / weaknesses / patterns + a decision cheat-sheet.

Sign conventions on every plot
------------------------------
* Across-year   :  dC = year_max - year_min  (e.g. 2022 - 2021)
* Empirical/ML  :  dC = ML - Empirical
* R alone       :  dC = year_max - year_min  (same as the pixel products)
Difference maps share a single ±colour bound across products (joint p98 of |dC|,
overridable via ``cfg['dC_vmax_fixed']``) so eyeballing is honest.

Outputs (all under CONFIG['out_dir'])
-------------------------------------
  figures (*.png), breakdown tables (*.csv), and ``compare_summary.txt``.

Usage
-----
    python compare_products.py
    python compare_products.py --years 2021 2022
    python compare_products.py --skip-maps          # skip the heavy pixel maps
    python compare_products.py --skip-r             # empirical vs ML only
    python compare_products.py --skip-fc            # skip the FC time-series section

NB: paths below mirror ``analyse_area.py``. Edit CONFIG to match your machine.
This script only *reads* the products; it never recomputes them.
"""
from __future__ import annotations

import argparse
import glob
import os
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Optional dep for basemap tiles under the field maps. Falls back gracefully.
try:
    import contextily as cx                # noqa: F401
    _HAS_CTX = True
except Exception:
    _HAS_CTX = False


# ---------------------------------------------------------------------------
# Configuration -- EDIT PATHS HERE
# ---------------------------------------------------------------------------

CONFIG = {
    # --- The three products ---
    "empirical_dir":  "../cfactor/output/cfactor_pixels",      # compute_cfactor_pixels.py
    "ml_dir":         "../cfactor/output/cfactor_pixels_ml",   # compute_cfactor_pixels_ml.py
    "erosion_results_dir": "~/mnt/Data-Labo-RE/27_Natural_Resources-RE/"
                           "321.4_WAUM_protected/Resultate/Erosionsrisiko",

    # --- Region + land use (for the betr_ID / elevation bridges) ---
    "region_path":    "../cfactor/seeland.gpkg",
    "lnf_dir":        "~/mnt/eo-nas1/data/landuse/raw",        # lnf{year}.gpkg
    "nutzung_csv":    "~/mnt/Data-Labo-RE/27_Natural_Resources-RE/"
                      "321.4_WAUM_protected/Daten/Core_Snapshot/"
                      "Agrarbericht_2025/tbl_nutzungsdaten.csv",

    # --- Crop-label bridge: lnf_code -> Kultur_AUI -> "Kultur Kategorien 2020"
    #     (== R's Nutzung_DE_KatNutz). Both files ship with the project. ---
    "kulturmapping_csv": "~/mnt/Data-Labo-RE/27_Natural_Resources-RE/"
                         "321.4_WAUM_protected/Daten/Core_Snapshot/"
                         "Agrarbericht_2025/tbl_kulturmapping.csv",  # kulturcode -> Kultur_AUI / _nutzung / -gruppe
    "c_factor_csv":     "~/mnt/Data-Labo-RE/27_Natural_Resources-RE/"
                        "321.4_WAUM_protected/Daten/Erosionsrisiko/C_Faktoren.csv",    # Kultur AUI -> Kultur Kategorien 2020

    # --- Municipality boundaries for the choropleth maps (set None to skip) ---
    "gemeinde_boundaries_path": "~/mnt/eo-nas1/eoa-share/projects/028_Erosion/"
                                "Erosion/FC_mapping/swissBOUNDARIES3D_1_5_LV95_LN02.gpkg",
    "gemeinde_name_field":      "name",

    # --- Knobs ---
    "years":            [2021, 2022],
    "grenze_tal_berg":  600,                 # m a.s.l. Tal/Berg cutoff
    "elev_bins":        [0, 450, 550, 650, 800, 3000],   # elevation bins for breakdowns
    "agree_tol":        0.02,                # |dC| <= tol counts as "agreement"
    "top_n_crops":      15,                  # crops shown in per-crop breakdowns
    "map_max_pixels":   8_000_000,           # cap for the *inline* pixel PNG (imshow)
    "map_write_geotiff": True,               # write a full-res dC GeoTIFF (EPSG:32632)
    "map_geotiff_max_pixels": 400_000_000,   # memory guard for the GeoTIFF array
    "map_field_aggregate": True,             # also plot dC averaged per field (light)

    # --- Field-map cosmetics ---
    "basemap":          "osm",    # "swisstopo_grey", "swisstopo_color",
                                             #   "osm", or None to disable
    "basemap_zoom":     None,                # contextily zoom (None = auto)
    "dC_vmax_fixed":    None,                # if set (e.g. 0.10), forces |colour|
                                             #   bound everywhere; else joint p98
    "absdC_vmax_fixed": None,                # cap for magnitude-only (|dC|) maps
    "show_all_crops":   False,               # if True, by-crop bars show all crops
                                             #   (else top_n_crops by pixel count)

    # --- Crop labelling for ML/empirical pixel products ---
    #   "bridge" (default): lnf_code (from parquet) -> kulturmapping ->
    #                       C_Faktoren -> "Kultur Kategorien 2020"  (== crop_kat).
    #   "lnf":              each pixel's poly_id -> the LNF gpkg's nutzung_DE
    #                       for THAT analysis year. nutzung_DE shares its
    #                       vocabulary with the R output's Nutzung_DE_KatNutz,
    #                       so all three products end up labelled in the same
    #                       string space and the downstream (betr_ID, crop)
    #                       joins line up by construction.
    # The R product's crop label is ALWAYS Nutzung_DE_KatNutz, independent of
    # this setting.
    "crop_label_method": "lnf",

    # --- FC time-series analysis (section E) -----------------------------
    # Output of extract_fc_pixels.py (Stage A + A½ + GP, no C). This is the
    # ACTUAL per-pixel gap-filled FC of pixels in the region -- NOT the
    # calibration sample. The directory holds:
    #   - year=YYYY/lnf_code=NNN/part-*.parquet  (per-pixel x per-DOAY grid)
    #   - fc_fields_{year}.parquet               (per-field per-DOAY summary)
    # Section E reads the small per-field summary by default; flip
    # `fc_use_per_pixel` to True to load the full per-pixel grid instead.
    "fc_pixels_dir":     "../cfactor/output/fc_pixels",
    "fc_use_per_pixel":  False,
    # How many crops to show in the per-crop small-multiples grid.
    # Ranked by number of unique (poly_id, yr) fields (i.e. sample size).
    "fc_top_n_crops":    12,
    # Minimum per-(crop, year) field count for a crop panel to be plotted.
    "fc_min_samples_per_crop": 30,

    "out_dir":          "figures_compare",
}

PIX_RES = 10.0   # m, the pixel products' resolution (matches compute_cfactor_pixels)

# Consistent colours
PROD_COLORS = {"empirical": "#4477AA", "ml": "#CC6677", "previous": "#228833"}
PROD_LABELS = {"empirical": "Empirical (beta)", "ml": "ML", "previous": "Previous (R)"}
YEAR_COLORS = {2021: "#4477AA", 2022: "#EE9944"}

# Tillage analysis (Section D) -- binary scheme mirrors R's 05-Dataprep.R
TILLAGE_ORDER    = ["Pflug", "Mulch"]
TRANSITION_ORDER = ["Pflug>Pflug", "Pflug>Mulch", "Mulch>Pflug", "Mulch>Mulch"]


# ===========================================================================
# Small helpers
# ===========================================================================

def _expand(p: str) -> str:
    return os.path.expanduser(p)


def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def savefig(fig, name: str, cfg: dict, dpi: int = 170):
    _ensure_dir(cfg["out_dir"])
    path = os.path.join(cfg["out_dir"], name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}")


def _savecsv(df: pd.DataFrame, name: str, cfg: dict, index=False):
    _ensure_dir(cfg["out_dir"])
    path = os.path.join(cfg["out_dir"], name)
    df.to_csv(path, index=index)
    print(f"  Saved {path}")


def _norm_farm_id(s: pd.Series) -> pd.Series:
    """Normalise farm id to a clean integer-string (robust to 1000.0 vs 1000)."""
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().all():
        return num.astype("Int64").astype(str)
    return s.astype(str)


def _cell_key(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Integer (1 m) key for joining identical 10 m pixel grids without float noise."""
    return (np.round(x).astype(np.int64), np.round(y).astype(np.int64))


def _wavg(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    m = v.notna() & w.notna() & (w > 0)
    if not m.any():
        return np.nan
    return float(np.average(v[m], weights=w[m]))


def _add_basemap(ax, cfg: dict, crs: str = "EPSG:2056"):
    """Add a basemap tile under `ax`. Silent no-op if contextily missing or fails.

    `cfg["basemap"]` selects the provider:
      - "swisstopo_grey"  : Swisstopo NationalMapGrey  (default; CH only)
      - "swisstopo_color" : Swisstopo NationalMapColor (CH only)
      - "osm"             : OpenStreetMap.Mapnik       (worldwide)
      - None / "none"     : skip
    """
    name = cfg.get("basemap")
    if not name or name == "none" or not _HAS_CTX:
        return
    try:
        import contextily as cx
        providers = {
            "swisstopo_grey":  cx.providers.SwissFederalGeoportal.NationalMapGrey,
            "swisstopo_color": cx.providers.SwissFederalGeoportal.NationalMapColor,
            "osm":             cx.providers.OpenStreetMap.Mapnik,
        }
        src = providers.get(name, providers["swisstopo_grey"])
        kwargs = dict(crs=crs, source=src, attribution_size=5)
        if cfg.get("basemap_zoom") is not None:
            kwargs["zoom"] = cfg["basemap_zoom"]
        cx.add_basemap(ax, **kwargs)
    except Exception as e:
        # last-ditch fall-back to OSM, then give up quietly
        try:
            import contextily as cx
            cx.add_basemap(ax, crs=crs, source=cx.providers.OpenStreetMap.Mapnik,
                           attribution_size=5)
        except Exception:
            print(f"  [INFO] basemap unavailable ({e}); plotting without tiles")


def _joint_vmax(arrs, p: float = 98.0, floor: float = 0.05) -> float:
    """98th percentile of |x| over the *concatenation* of `arrs`. Used for the
    shared dC colour scale across the empirical/ML products."""
    arrs = [np.abs(np.asarray(a)) for a in arrs if a is not None and len(a) > 0]
    if not arrs:
        return floor
    v = float(np.nanpercentile(np.concatenate(arrs), p))
    return v if v > 0 else floor


class Summary:
    """Collects headline lines, printed live and flushed to compare_summary.txt."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.lines: list[str] = []

    def add(self, line: str = ""):
        print(line)
        self.lines.append(line)

    def section(self, title: str):
        bar = "=" * 70
        self.add("")
        self.add(bar)
        self.add(title)
        self.add(bar)

    def flush(self):
        _ensure_dir(self.cfg["out_dir"])
        path = os.path.join(self.cfg["out_dir"], "compare_summary.txt")
        with open(path, "w") as f:
            f.write("\n".join(self.lines) + "\n")
        print(f"\n  Summary written to {path}")


# ===========================================================================
# Loaders -- products
# ===========================================================================

def _read_hive_year(product_dir: str, year: int) -> pd.DataFrame | None:
    """Read one year of a hive-partitioned pixel parquet (year=YYYY/lnf_code=NNN)."""
    year_dir = os.path.join(_expand(product_dir), f"year={year}")
    if not os.path.isdir(year_dir):
        print(f"  [WARN] {year_dir} not found")
        return None
    df = pd.read_parquet(year_dir)            # pyarrow recovers nested lnf_code partition
    if "lnf_code" not in df.columns:
        print(f"  [WARN] lnf_code missing after reading {year_dir}")
    df["year"] = year
    return df


def load_pixels(product_key: str, cfg: dict, year: int) -> pd.DataFrame | None:
    """Per-pixel product -> [poly_id, lnf_code, x, y, c_factor, n_clean_obs_field, year]."""
    d = cfg["empirical_dir"] if product_key == "empirical" else cfg["ml_dir"]
    df = _read_hive_year(d, year)
    if df is None:
        return None
    keep = [c for c in ["poly_id", "lnf_code", "x", "y", "c_factor",
                        "n_clean_obs_field", "year"] if c in df.columns]
    return df[keep]


def load_field_summary(product_key: str, cfg: dict, year: int) -> pd.DataFrame | None:
    """Per-field roll-up. Prefer the written cfactor_fields_{year}.parquet; else
    aggregate the pixel parquet. Returns
    [poly_id, lnf_code, year, c_factor_mean, n_pixels, area_m2]."""
    d = _expand(cfg["empirical_dir"] if product_key == "empirical" else cfg["ml_dir"])
    fld_path = os.path.join(d, f"cfactor_fields_{year}.parquet")
    if os.path.exists(fld_path):
        fld = pd.read_parquet(fld_path)
        if "year" not in fld.columns:
            fld["year"] = year
        if "area_m2" not in fld.columns and "n_pixels" in fld.columns:
            fld["area_m2"] = fld["n_pixels"] * PIX_RES ** 2
        return fld
    pix = load_pixels(product_key, cfg, year)
    if pix is None:
        return None
    fld = (pix.groupby(["poly_id", "lnf_code"], as_index=False)
              .agg(c_factor_mean=("c_factor", "mean"),
                   n_pixels=("c_factor", "size")))
    fld["year"] = year
    fld["area_m2"] = fld["n_pixels"] * PIX_RES ** 2
    return fld


def load_r_results(cfg: dict, year: int) -> pd.DataFrame | None:
    """Previous R product for ``year`` -> tidy [betr_ID, crop, Region, Gemeinde,
    Flaeche, C_fact_detail]. Picks the most recent dated file in {dir}/{year}/."""
    base = _expand(cfg["erosion_results_dir"])
    pattern = os.path.join(base, str(year),
                           f"*_Kultur_Betrieb_Erosionsrisiko_2x2_{year}.csv")
    matches = sorted(glob.glob(pattern))
    if not matches:
        # also try a flat layout (files directly in erosion_results_dir)
        matches = sorted(glob.glob(os.path.join(
            base, f"*_Kultur_Betrieb_Erosionsrisiko_2x2_{year}.csv")))
    if not matches:
        print(f"  [WARN] No R result file for {year} ({pattern})")
        return None
    path = matches[-1]
    print(f"  {year}: reading {os.path.basename(path)}")
    df = pd.read_csv(path, sep=";", encoding="latin1")

    # The R output's crop column is Nutzung_DE_KatNutz. The string values in
    # that column live in the same vocabulary as the LNF gpkg's `nutzung_DE`,
    # which is what makes the "lnf" labelling method round-trip cleanly.
    if "Nutzung_DE_KatNutz" not in df.columns:
        print(f"  [WARN] {os.path.basename(path)} has no Nutzung_DE_KatNutz "
              f"(have {df.columns.tolist()[:12]}...)")
        return None
    required = {"betr_ID", "Flaeche", "C_fact_detail"}
    if required - set(df.columns):
        print(f"  [WARN] {os.path.basename(path)} missing required cols "
              f"(have {df.columns.tolist()[:12]}...)")
        return None

    out = pd.DataFrame({
        "betr_ID":       _norm_farm_id(df["betr_ID"]),
        "crop":          df["Nutzung_DE_KatNutz"].astype(str).str.strip(),
        "tillage":       (df["Bodenbearbeitung"].astype(str).str.strip()
                          if "Bodenbearbeitung" in df.columns else "Unknown"),
        "Region":        df["Region"].astype(str) if "Region" in df else pd.NA,
        "Gemeinde":      df["Gemeinde"].astype(str) if "Gemeinde" in df else pd.NA,
        "Flaeche":       pd.to_numeric(df["Flaeche"], errors="coerce"),
        "C_fact_detail": pd.to_numeric(df["C_fact_detail"], errors="coerce"),
        "year":          year,
    })
    # Normalise C_Faktoren typography to match LNF gpkg's nutzung_DE (mirrors
    # sample_FC.py:73-76). Required for (betr_ID, crop) joins to line up.
    out["crop"] = out["crop"].replace({
        "Einjährige Freilandgemüse (ohne Konservengemüse)":
            "Einjährige Freilandgemüse, ohne Konservengemüse",
    })
    out = out.dropna(subset=["Flaeche", "C_fact_detail"])
    return out[out["Flaeche"] > 0]


def load_fc_timeseries(cfg: dict, lnf_bridge: dict, years) -> pd.DataFrame | None:
    """Per-field per-DOAY gap-filled FC time series, region-restricted.

    Data source is the output of ``extract_fc_pixels.py`` (the standalone
    extractor that runs Stage A + A½ + GP from ``compute_cfactor_pixels.py``
    but persists the FC grid instead of integrating to C). This is the
    ACTUAL FC of pixels in the region -- NOT the calibration sample.

    By default we read the per-field summary parquet
    (``fc_fields_{year}.parquet``); set ``cfg['fc_use_per_pixel'] = True`` to
    read the full hive-partitioned per-pixel grid instead (much heavier but
    lets you compute pixel-level IQRs).

    Returns
    -------
    DataFrame [yr, poly_id, lnf_code, time, doay, pv, npv, fc_total]
        Per-field-per-DOAY when ``fc_use_per_pixel`` is False; per-pixel-per-
        DOAY otherwise. Returns None if the directory or per-year files are
        missing.

    Notes
    -----
    The extractor was run with the same region.gpkg as this script, so every
    pixel is already inside the region. ``lnf_bridge`` is used only as a
    sanity filter -- rows whose ``poly_id`` falls outside the bridge are
    dropped with an informational message.
    """
    fc_dir = cfg.get("fc_pixels_dir")
    if not fc_dir:
        print("  [INFO] fc_pixels_dir not configured -- skipping FC section")
        return None
    fc_dir = _expand(fc_dir)
    if not os.path.isdir(fc_dir):
        print(f"  [WARN] FC pixels dir not found at {fc_dir} -- skipping FC section")
        print(f"         Generate it with `python extract_fc_pixels.py "
              f"--region <region.gpkg> --years {' '.join(map(str, years))}`")
        return None

    use_pixel = bool(cfg.get("fc_use_per_pixel", False))
    parts = []
    for year in years:
        if use_pixel:
            ydir = os.path.join(fc_dir, f"year={year}")
            if not os.path.isdir(ydir):
                print(f"  [WARN] {ydir} not found -- year {year} missing "
                      "from per-pixel grid")
                continue
            cols = ["poly_id", "lnf_code", "x", "y", "time", "doay",
                    "pv", "npv", "fc_total"]
            df = pd.read_parquet(ydir, columns=cols)
            print(f"  {year}: {len(df):,} per-pixel rows, "
                  f"{df['poly_id'].nunique():,} fields, "
                  f"{df['doay'].nunique()} DOAYs")
        else:
            fld_path = os.path.join(fc_dir, f"fc_fields_{year}.parquet")
            if not os.path.exists(fld_path):
                print(f"  [WARN] {fld_path} not found -- "
                      f"year {year} missing from per-field summary")
                continue
            # Per-field schema -> standardise to the same column names as
            # the per-pixel schema for downstream code (no per-field "x, y").
            fld = pd.read_parquet(fld_path)
            df = pd.DataFrame({
                "poly_id":  fld["poly_id"],
                "lnf_code": fld["lnf_code"],
                "time":     fld["time"],
                "doay":     fld["doay"],
                "pv":       fld["pv_mean"],
                "npv":      fld["npv_mean"],
                "fc_total": fld["fc_total_mean"],
                "n_pixels": fld.get("n_pixels", 1),
            })
            print(f"  {year}: {len(df):,} per-field rows, "
                  f"{df['poly_id'].nunique():,} fields, "
                  f"{df['doay'].nunique()} DOAYs")
        df["yr"] = int(year)
        parts.append(df)
    if not parts:
        return None
    df_fc = pd.concat(parts, ignore_index=True)

    # Sanity filter against the bridge. Normally a no-op (extractor ran on
    # the same region) -- loud if it isn't.
    region_polys = set()
    for y in years:
        lb = lnf_bridge.get(y)
        if lb is not None and "poly_id" in lb.columns:
            region_polys |= set(pd.to_numeric(lb["poly_id"], errors="coerce")
                                  .dropna().astype(int).tolist())
    if region_polys:
        df_fc["poly_id"] = pd.to_numeric(df_fc["poly_id"], errors="coerce").astype("Int64")
        keep = df_fc["poly_id"].astype("Int64").isin(region_polys)
        if not keep.all():
            print(f"  [INFO] {(~keep).sum():,}/{len(df_fc):,} rows have a "
                  f"poly_id outside the bridge region -- dropped (likely "
                  f"a region mismatch between the extractor and this script)")
            df_fc = df_fc[keep]

    if df_fc.empty:
        print("  [WARN] No FC rows after region filter")
        return None

    df_fc["time"]     = pd.to_datetime(df_fc["time"])
    df_fc["lnf_code"] = pd.to_numeric(df_fc["lnf_code"], errors="coerce").astype("Int64")
    df_fc["doay"]     = pd.to_numeric(df_fc["doay"], errors="coerce").astype("Int64")
    return df_fc.reset_index(drop=True)


# ===========================================================================
# Loaders -- bridges (LNF, elevation, crop labels)
# ===========================================================================

def load_lnf_bridge(cfg: dict, year: int, region_2056) -> pd.DataFrame | None:
    """Per-field bridge for ``year`` restricted to the region.

    Returns [poly_id, uuid, betr_ID, lnf_code, lnf_area_m2, cx, cy] where
    ``poly_id`` == LNF ``id`` (the pixel products' key), ``uuid`` joins the
    nutzungsdaten elevation, ``betr_ID`` joins the R product, and (cx, cy) are
    EPSG:2056 centroids for the aggregated field maps.
    """
    lnf_path = os.path.join(_expand(cfg["lnf_dir"]), f"lnf{year}.gpkg")
    if not os.path.exists(lnf_path):
        print(f"  [WARN] {lnf_path} not found -- no betr_ID/elevation bridge for {year}")
        return None
    bbox = tuple(gpd.GeoSeries([region_2056], crs="EPSG:2056").total_bounds)
    lnf = gpd.read_file(lnf_path, bbox=bbox)
    lnf = lnf[lnf.intersects(region_2056)]
    if lnf.empty:
        print(f"  [WARN] No LNF fields intersect the region for {year}")
        return None

    cols = lnf.columns
    if "id" not in cols:
        print(f"  [WARN] LNF {year} has no 'id' column (have {list(cols)}) -- "
              f"cannot bridge poly_id")
        return None
    id_uuid = lnf["uuid"].astype(str) if "uuid" in cols else lnf["id"].astype(str)
    betr = (_norm_farm_id(lnf["betriebsnummer"]) if "betriebsnummer" in cols
            else pd.Series(pd.NA, index=lnf.index))
    cent = lnf.to_crs("EPSG:2056").geometry.centroid

    # nutzung_DE is the LNF gpkg's German crop label and is the field the R
    # pipeline groups by (`Nutzung_DE` in the R output). Pulling it here makes
    # the `lnf`-style crop labelling possible. Column casing varies, so match
    # case-insensitively.
    nut_col = next((c for c in cols if str(c).lower() == "nutzung_de"), None)
    nutzung_de = (lnf[nut_col].astype(str).str.strip()
                  if nut_col is not None
                  else pd.Series(pd.NA, index=lnf.index, dtype=object))

    out = pd.DataFrame({
        "poly_id":     lnf["id"].astype("int64"),
        "uuid":        id_uuid,
        "betr_ID":     betr,
        "lnf_code":    pd.to_numeric(lnf["lnf_code"], errors="coerce").astype("Int64")
                       if "lnf_code" in cols else pd.NA,
        "nutzung_DE":  nutzung_de,
        "lnf_area_m2": lnf.to_crs("EPSG:2056").geometry.area,
        "cx":          cent.x.values,
        "cy":          cent.y.values,
    })
    return out.drop_duplicates("poly_id")


def load_elevation(cfg: dict, year: int) -> pd.DataFrame | None:
    """uuid -> swissALTI3D + Region(Tal/Berg) for ``year`` (from nutzungsdaten)."""
    path = _expand(cfg.get("nutzung_csv", ""))
    if not os.path.exists(path):
        print(f"  [WARN] {path} not found -- no elevation breakdown")
        return None
    rows = []
    for chunk in pd.read_csv(path, encoding="latin1", sep=";",
                             usecols=["Jahr", "Flaechen_ID", "swissALTI3D"],
                             chunksize=300_000):
        chunk = chunk[chunk["Jahr"] == year]
        if not chunk.empty:
            rows.append(chunk)
    if not rows:
        print(f"  [WARN] No nutzungsdaten rows for {year}")
        return None
    df = (pd.concat(rows)
            .dropna(subset=["swissALTI3D"])
            .drop_duplicates("Flaechen_ID"))
    cut = cfg["grenze_tal_berg"]
    return pd.DataFrame({
        "uuid":        df["Flaechen_ID"].astype(str),
        "swissALTI3D": pd.to_numeric(df["swissALTI3D"], errors="coerce"),
        "Region":      np.where(df["swissALTI3D"] <= cut, "Tal", "Berg"),
    })


def build_crop_bridge(cfg: dict) -> pd.DataFrame | None:
    """lnf_code -> common crop label.

    Two hops, both from project CSVs:
      kulturmapping:  kulturcode (== lnf_code) -> Kultur_AUI, Kultur_nutzung, Kulturgruppe
      C_Faktoren:     "Kultur AUI" -> "Kultur Kategorien 2020"  (== R's crop label)

    Returns [lnf_code, crop_kat, crop_group, crop_name] where ``crop_kat`` is the
    label that matches the R product. ``crop_name`` / ``crop_group`` are fallbacks
    / coarser groupings for plots.
    """
    km_path = _expand(cfg["kulturmapping_csv"])
    cf_path = _expand(cfg["c_factor_csv"])
    if not os.path.exists(km_path):
        print(f"  [WARN] {km_path} not found -- no crop-label bridge")
        return None
    km = pd.read_csv(km_path, sep=";", encoding="cp1252")
    km.columns = [c.strip() for c in km.columns]
    need = {"kulturcode", "Kultur_AUI", "Kultur_nutzung"}
    if need - set(km.columns):
        print(f"  [WARN] kulturmapping missing {need - set(km.columns)}")
        return None
    bridge = pd.DataFrame({
        "lnf_code":   pd.to_numeric(km["kulturcode"], errors="coerce").astype("Int64"),
        "Kultur_AUI": km["Kultur_AUI"].astype(str).str.strip(),
        "crop_name":  km["Kultur_nutzung"].astype(str).str.strip(),
        "crop_group": (km["Kulturgruppe"].astype(str).str.strip()
                       if "Kulturgruppe" in km.columns else km["Kultur_nutzung"]),
    }).dropna(subset=["lnf_code"])

    # AUI -> Kategorien via C_Faktoren
    if os.path.exists(cf_path):
        cf = pd.read_csv(cf_path, sep=";", encoding="cp1252")
        cf.columns = [c.strip() for c in cf.columns]
        kat_col = next((c for c in cf.columns if c.lower().startswith("kultur kateg")), None)
        aui_col = next((c for c in cf.columns if c.lower().replace(" ", "") == "kulturaui"), None)
        if kat_col and aui_col:
            aui_to_kat = (cf[[aui_col, kat_col]].dropna()
                          .assign(**{aui_col: lambda d: d[aui_col].astype(str).str.strip()})
                          .drop_duplicates(aui_col)
                          .set_index(aui_col)[kat_col].to_dict())
            bridge["crop_kat"] = bridge["Kultur_AUI"].map(aui_to_kat)
        else:
            print("  [WARN] C_Faktoren: could not find AUI / Kategorien columns")
            bridge["crop_kat"] = pd.NA
    else:
        print(f"  [WARN] {cf_path} not found -- crop_kat will fall back to crop_name")
        bridge["crop_kat"] = pd.NA

    bridge["crop_kat"] = bridge["crop_kat"].fillna(bridge["crop_name"])
    return bridge[["lnf_code", "crop_kat", "crop_group", "crop_name"]].drop_duplicates("lnf_code")


def _label_field_crops(df: pd.DataFrame, crop_by_lnf: dict,
                       crop_by_poly: dict | None, cfg: dict,
                       poly_col: str | None = None,
                       lnf_col: str = "lnf_code",
                       year: int | None = None) -> pd.Series:
    """Return a per-row crop-label Series for ML/empirical rows.

    Source priority depends on ``cfg['crop_label_method']``:

    * ``"lnf"``    : ``poly_id`` -> ``Nutzung_DE_KatNutz`` via
                     ``crop_by_poly[year]``, where each year's map was built
                     from THAT year's LNF gpkg joined to THAT year's R output
                     (so crop rotations are respected).
    * ``"bridge"`` (default): ``lnf_code`` -> ``crop_kat`` via
                     ``crop_by_lnf`` (kulturmapping + C_Faktoren).

    Unmapped rows in the chosen mode fall back through the bridge, and finally
    to the raw ``lnf_code`` rendered as a string. So switching modes never
    *introduces* unknowns -- it only relabels rows the other map covers.
    """
    method = cfg.get("crop_label_method", "bridge")

    # Pick the right per-year poly map. crop_by_poly is {year: {poly: label}}.
    # If a flat dict snuck in (legacy / single-year run), accept it too.
    year_map = None
    if method == "lnf" and crop_by_poly and poly_col is not None and poly_col in df.columns:
        is_per_year = isinstance(next(iter(crop_by_poly.values())), dict)
        if is_per_year:
            if year is not None:
                # A specific year was asked for. If it's not in the map, fall
                # through to the bridge -- do NOT silently mix years.
                year_map = crop_by_poly.get(year)
            else:
                # No year hint at all -> union (last-write-wins).
                year_map = {}
                for ym in crop_by_poly.values():
                    year_map.update(ym)
        else:
            # flat dict (poly -> label) -- treat as a single map
            year_map = crop_by_poly

    if year_map:
        primary = df[poly_col].map(year_map)
        # empty / 'nan' strings count as missing
        primary = primary.where(primary.astype(str).str.strip().ne(""), pd.NA)
    else:
        primary = pd.Series(pd.NA, index=df.index, dtype=object)

    if lnf_col in df.columns:
        primary = primary.where(primary.notna(),
                                df[lnf_col].map(crop_by_lnf))
        primary = primary.where(primary.notna(),
                                df[lnf_col].astype(str))
    return primary


# ===========================================================================
# Rasterise a value column for the full per-pixel maps
# ===========================================================================

def _rasterise(df: pd.DataFrame, value_col: str, cfg: dict):
    """(x, y, value) on the 10 m grid -> (2D array, extent) for imshow (EPSG:32632)."""
    x = df["x"].values
    y = df["y"].values
    xs = np.round((x - x.min()) / PIX_RES).astype(int)
    ys = np.round((y.max() - y) / PIX_RES).astype(int)
    ncol, nrow = xs.max() + 1, ys.max() + 1
    if nrow * ncol > cfg["map_max_pixels"]:
        print(f"  [INFO] map grid {nrow}x{ncol} exceeds cap "
              f"({cfg['map_max_pixels']:,}) -- skipping pixel map")
        return None, None
    arr = np.full((nrow, ncol), np.nan, dtype=np.float32)
    arr[ys, xs] = df[value_col].values.astype(np.float32)
    extent = [x.min() - PIX_RES / 2, x.min() - PIX_RES / 2 + ncol * PIX_RES,
              y.max() + PIX_RES / 2 - nrow * PIX_RES, y.max() + PIX_RES / 2]
    return arr, extent


def _write_diff_geotiff(df: pd.DataFrame, value_col: str, cfg: dict, fname: str):
    """Write (x, y, value) on the 10 m grid to a full-res GeoTIFF (EPSG:32632).

    Unlike the inline PNG this has no plotting cap -- it is the right artifact for
    a region too large to render, and opens directly in QGIS. Only the array
    allocation is guarded (``map_geotiff_max_pixels``).
    """
    try:
        import rasterio
        from rasterio.transform import from_origin
    except Exception:
        print("  [WARN] rasterio not available -- cannot write GeoTIFF")
        return
    x, y = df["x"].values, df["y"].values
    cols = np.round((x - x.min()) / PIX_RES).astype(int)
    rows = np.round((y.max() - y) / PIX_RES).astype(int)
    ncol, nrow = int(cols.max()) + 1, int(rows.max()) + 1
    if nrow * ncol > cfg.get("map_geotiff_max_pixels", 400_000_000):
        print(f"  [INFO] GeoTIFF grid {nrow}x{ncol} exceeds "
              f"map_geotiff_max_pixels -- skipping")
        return
    arr = np.full((nrow, ncol), np.nan, dtype=np.float32)
    arr[rows, cols] = df[value_col].values.astype(np.float32)
    _ensure_dir(cfg["out_dir"])
    path = os.path.join(cfg["out_dir"], fname)
    transform = from_origin(x.min() - PIX_RES / 2, y.max() + PIX_RES / 2,
                            PIX_RES, PIX_RES)
    with rasterio.open(path, "w", driver="GTiff", height=nrow, width=ncol,
                       count=1, dtype="float32", crs="EPSG:32632",
                       transform=transform, nodata=np.nan,
                       compress="deflate") as dst:
        dst.write(arr, 1)
    print(f"  Saved {path}  ({nrow}x{ncol})")


def _field_diff_map(poly_ids: np.ndarray, dvals: np.ndarray,
                    centroid_by_poly: dict, cfg: dict, title: str, fname: str,
                    direction: str = "", vmax_shared: float | None = None):
    """Plot mean dC per field at its LNF centroid (EPSG:2056).

    ``direction`` (e.g. '2022 - 2021' or 'ML - Empirical') is shown on title
    AND colourbar so the sign of dC is unambiguous (red = positive).
    ``vmax_shared`` lets the caller force the same colour scale across products.
    """
    g = pd.DataFrame({"poly": poly_ids, "dC": dvals}).groupby("poly")["dC"].mean()
    xs, ys, vs = [], [], []
    for poly, v in g.items():
        c = centroid_by_poly.get(int(poly))
        if c is not None:
            xs.append(c[0]); ys.append(c[1]); vs.append(v)
    if not vs:
        print("  [INFO] no field centroids available -- skipping field map")
        return
    xs, ys, vs = np.asarray(xs), np.asarray(ys), np.asarray(vs)

    if vmax_shared is not None:
        vmax = float(vmax_shared)
    elif cfg.get("dC_vmax_fixed") is not None:
        vmax = float(cfg["dC_vmax_fixed"])
    else:
        vmax = float(np.nanpercentile(np.abs(vs), 98) or 0.05)

    # Marker size shrinks as the field count grows, to limit overlap; the most
    # divergent fields are drawn last so they are never hidden underneath.
    n = len(vs)
    size = float(np.clip(900.0 / np.sqrt(max(n, 1)), 2.5, 9.0))
    order = np.argsort(np.abs(vs))

    fig, ax = plt.subplots(figsize=(8, 8))
    # Plot the data FIRST so the axes have proper LV95 limits; only then add the
    # basemap. Calling add_basemap on an empty axes leaves xlim/ylim at (0,1),
    # which pins the view at the EPSG:2056 origin and pushes the real points
    # out of frame.
    sc = ax.scatter(xs[order], ys[order], c=vs[order], cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax, s=size, marker="s",
                    alpha=0.85, linewidths=0.12, edgecolors="white", zorder=3)
    ax.set_aspect("equal")
    ax.set_xlabel("E (m, LV95)"); ax.set_ylabel("N (m, LV95)")
    ax.margins(0.01)
    _add_basemap(ax, cfg, crs="EPSG:2056")
    head = f"{title} (mean per field)"
    if direction:
        head += f"\ndC = {direction}"
    ax.set_title(head)
    cb_label = (f"mean dC = {direction}\n(red > 0)"
                if direction else "mean dC (red > 0)")
    fig.colorbar(sc, ax=ax, label=cb_label, shrink=0.7)
    savefig(fig, fname, cfg, dpi=200)


def _field_absdiff_map(poly_ids: np.ndarray, dvals: np.ndarray,
                       centroid_by_poly: dict, cfg: dict, title: str,
                       fname: str, direction: str = "",
                       vmax_shared: float | None = None,
                       cmap: str = "OrRd"):
    """Map of |dC| per field (magnitude only). Sequential cmap, basemap added.

    Useful for `where do products disagree the MOST?`. Sign-agnostic; pair with
    the signed map (`_field_diff_map`) to see which way each field leans.
    """
    g = pd.DataFrame({"poly": poly_ids, "abs_dC": np.abs(dvals)}).groupby("poly")["abs_dC"].mean()
    xs, ys, vs = [], [], []
    for poly, v in g.items():
        c = centroid_by_poly.get(int(poly))
        if c is not None:
            xs.append(c[0]); ys.append(c[1]); vs.append(v)
    if not vs:
        print("  [INFO] no field centroids available -- skipping abs-diff map")
        return
    xs, ys, vs = np.asarray(xs), np.asarray(ys), np.asarray(vs)

    if vmax_shared is not None:
        vmax = float(vmax_shared)
    elif cfg.get("absdC_vmax_fixed") is not None:
        vmax = float(cfg["absdC_vmax_fixed"])
    else:
        vmax = float(np.nanpercentile(vs, 98) or 0.05)

    n = len(vs)
    size = float(np.clip(900.0 / np.sqrt(max(n, 1)), 2.5, 9.0))
    order = np.argsort(vs)              # high values drawn on top

    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(xs[order], ys[order], c=vs[order], cmap=cmap,
                    vmin=0, vmax=vmax, s=size, marker="s", alpha=0.85,
                    linewidths=0.12, edgecolors="white", zorder=3)
    ax.set_aspect("equal")
    ax.set_xlabel("E (m, LV95)"); ax.set_ylabel("N (m, LV95)")
    ax.margins(0.01)
    _add_basemap(ax, cfg, crs="EPSG:2056")
    head = f"{title} (mean |dC| per field)"
    if direction:
        head += f"\ndC = {direction}"
    ax.set_title(head)
    fig.colorbar(sc, ax=ax, label="mean |dC| per field", shrink=0.7)
    savefig(fig, fname, cfg, dpi=200)


def _emit_diff_maps(x, y, dvals, poly_ids, centroid_by_poly, cfg,
                    tag: str, title: str, make_maps: bool, direction: str = "",
                    vmax_shared: float | None = None):
    """One entry point for all difference-map variants.

      1. full-res GeoTIFF (always, if enabled)              -> {tag}_map.tif
      2. inline pixel PNG  (only if grid <= map_max_pixels)  -> {tag}_map.png
      3. per-field PNG     (always, if enabled)              -> {tag}_fieldmap.png

    ``direction`` (e.g. '2022 - 2021', 'ML - Empirical') labels the sign of dC.
    ``vmax_shared`` forces a common ±colour bound on PNG + fieldmap, so multiple
    products can be eyeballed side-by-side.
    """
    if not make_maps:
        return
    df = pd.DataFrame({"x": x, "y": y, "dC": dvals})
    dlabel = f"dC = {direction}" if direction else "dC"

    if cfg.get("map_write_geotiff", True):
        _write_diff_geotiff(df, "dC", cfg, f"{tag}_map.tif")

    arr, extent = _rasterise(df, "dC", cfg)
    if arr is not None:
        if vmax_shared is not None:
            vmax = float(vmax_shared)
        elif cfg.get("dC_vmax_fixed") is not None:
            vmax = float(cfg["dC_vmax_fixed"])
        else:
            vmax = float(np.nanpercentile(np.abs(arr), 98) or 0.05)
        fig, ax = plt.subplots(figsize=(7, 7))
        im = ax.imshow(arr, extent=extent, origin="upper", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, interpolation="nearest")
        head = f"{title} (per pixel)"
        if direction:
            head += f"\ndC = {direction}"
        ax.set_title(head)
        ax.set_aspect("equal"); ax.set_xlabel("E (m)"); ax.set_ylabel("N (m)")
        fig.colorbar(im, ax=ax, label=dlabel, shrink=0.7)
        savefig(fig, f"{tag}_map.png", cfg)

    if cfg.get("map_field_aggregate", True) and centroid_by_poly:
        _field_diff_map(np.asarray(poly_ids), np.asarray(dvals),
                        centroid_by_poly, cfg, title, f"{tag}_fieldmap.png",
                        direction=direction, vmax_shared=vmax_shared)


def _merge_centroids(lnf_bridge: dict, years) -> dict:
    """Union of per-year poly_id -> (cx, cy) EPSG:2056 centroids."""
    out = {}
    for y in years:
        lb = lnf_bridge.get(y)
        if lb is not None:
            for r in lb.itertuples():
                out[int(r.poly_id)] = (r.cx, r.cy)
    return out


# ===========================================================================
# Plausibility (a cheap strengths/weaknesses signal per product)
# ===========================================================================

def plausibility(name: str, c: pd.Series, summ: Summary):
    c = pd.to_numeric(c, errors="coerce").dropna()
    if c.empty:
        return
    n = len(c)
    summ.add(f"  {name:<16} n={n:,}  mean={c.mean():.4f}  median={c.median():.4f}  "
             f"p5={c.quantile(.05):.4f}  p95={c.quantile(.95):.4f}")
    summ.add(f"  {'':<16} <0: {100*(c < 0).mean():.2f}%  =0: {100*(c == 0).mean():.2f}%  "
             f">1: {100*(c > 1).mean():.2f}%  >=0.999: {100*(c >= 0.999).mean():.2f}%")


# ===========================================================================
# A. WITHIN a product, ACROSS the years
# ===========================================================================

def _compute_across_years_pixel(product_key: str, cfg: dict):
    """Load both years for a per-pixel product, join on the 10 m grid, compute dC.

    Returns ``(m, y0, y1)`` where ``m`` carries ``dC = c_factor_{y1} - c_factor_{y0}``
    plus the per-year columns, or ``None`` if a year is missing / no overlap.
    Split out so the caller can compute a joint vmax across products before
    plotting (avoids re-reading the parquets twice).
    """
    years = cfg["years"]
    if len(years) < 2:
        return None
    y0, y1 = years[0], years[-1]
    p0 = load_pixels(product_key, cfg, y0)
    p1 = load_pixels(product_key, cfg, y1)
    if p0 is None or p1 is None:
        return None
    for p in (p0, p1):
        kx, ky = _cell_key(p["x"].values, p["y"].values)
        p["k"] = kx.astype(np.int64) * 1_000_000_007 + ky
    m = p0.merge(p1, on="k", suffixes=(f"_{y0}", f"_{y1}"))
    if m.empty:
        return None
    m["dC"] = m[f"c_factor_{y1}"] - m[f"c_factor_{y0}"]
    return m, y0, y1


def across_years_pixel(product_key: str, cfg: dict, elev_by_poly: dict,
                       crop_by_lnf: dict, centroid_by_poly: dict,
                       summ: Summary, make_maps: bool,
                       precomputed=None, vmax_shared: float | None = None,
                       crop_by_poly: dict | None = None):
    """2021 vs 2022 for a per-pixel product, joined on the shared (x, y) grid.

    ``precomputed`` (output of ``_compute_across_years_pixel``) lets the caller
    reuse the merge across calls; ``vmax_shared`` forces a common colour scale.
    """
    if precomputed is None:
        precomputed = _compute_across_years_pixel(product_key, cfg)
    name = PROD_LABELS[product_key]
    if precomputed is None:
        years = cfg["years"]
        summ.add(f"\n-- {name}: {years[0]} vs {years[-1]} (per-pixel) --")
        summ.add("  [skip] missing a year")
        return
    m, y0, y1 = precomputed
    summ.add(f"\n-- {name}: {y0} vs {y1} (per-pixel) --")

    c0 = m[f"c_factor_{y0}"].to_numpy()
    c1 = m[f"c_factor_{y1}"].to_numpy()
    d = m["dC"].to_numpy()
    summ.add(f"  shared px = {len(m):,}")
    summ.add(f"  dC ({y1}-{y0}): mean={d.mean():+.4f}  median={np.median(d):+.4f}  "
             f"MAD={np.median(np.abs(d - np.median(d))):.4f}  "
             f"mean|dC|={np.abs(d).mean():.4f}  corr={np.corrcoef(c0, c1)[0, 1]:.3f}")
    stable = 100 * (np.abs(d) <= cfg["agree_tol"]).mean()
    summ.add(f"  {stable:.0f}% of pixels stable within +/-{cfg['agree_tol']} C")

    # --- figure: hexbin + dC histogram ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    hb = axes[0].hexbin(c0, c1, gridsize=60, mincnt=1, cmap="viridis", bins="log")
    lim = [min(c0.min(), c1.min()), max(c0.max(), c1.max())]
    axes[0].plot(lim, lim, "r--", lw=1)
    axes[0].set_xlabel(f"C {y0}"); axes[0].set_ylabel(f"C {y1}")
    axes[0].set_title(f"{name}: pixel C, {y0} vs {y1}")
    fig.colorbar(hb, ax=axes[0], label="log10(pixels)")
    axes[1].hist(d, bins=80, color=PROD_COLORS[product_key], alpha=0.85)
    axes[1].axvline(0, color="k", lw=0.8)
    axes[1].set_xlabel(f"dC = {y1} - {y0}"); axes[1].set_ylabel("pixels")
    axes[1].set_title(f"Year-to-year change (dC = {y1} - {y0})")
    fig.tight_layout()
    savefig(fig, f"acrossyear_{product_key}_pixel.png", cfg)

    # --- breakdown by crop and elevation band ---
    m_view = m.copy()
    m_view["lnf_code"] = m_view[f"lnf_code_{y0}"]
    _breakdown_change(m_view, "dC", crop_by_lnf, elev_by_poly,
                      poly_col=f"poly_id_{y0}", cfg=cfg,
                      tag=f"acrossyear_{product_key}", summ=summ,
                      direction=f"{y1} - {y0}", also_all_crops=True,
                      crop_by_poly=crop_by_poly, year=y0)

    # --- difference maps: GeoTIFF (full res) + inline PNG (if small) + field map ---
    _emit_diff_maps(m[f"x_{y0}"].to_numpy(), m[f"y_{y0}"].to_numpy(), d,
                    m[f"poly_id_{y0}"].to_numpy(), centroid_by_poly, cfg,
                    tag=f"acrossyear_{product_key}",
                    title=f"{name}: dC {y1}-{y0}", make_maps=make_maps,
                    direction=f"{y1} - {y0}", vmax_shared=vmax_shared)


def across_years_r(cfg: dict, summ: Summary):
    """2021 vs 2022 for the R product, joined on (betr_ID, crop).

    Mirrors the per-pixel `across_years_pixel` plots (hexbin + dC hist + by-crop
    bar), at (betr_ID, crop) granularity. No field map (R has no per-field
    geometry). Returns the merged dataframe (or None) so downstream code can
    re-use it without reloading.
    """
    years = cfg["years"]
    if len(years) < 2:
        return None
    y0, y1 = years[0], years[-1]
    summ.add(f"\n-- {PROD_LABELS['previous']}: {y0} vs {y1} (farm x crop) --")
    r0, r1 = load_r_results(cfg, y0), load_r_results(cfg, y1)
    if r0 is None or r1 is None:
        summ.add("  [skip] missing a year")
        return None
    a0 = _agg_farm_crop(r0, "C_fact_detail", "Flaeche")
    a1 = _agg_farm_crop(r1, "C_fact_detail", "Flaeche")
    m = a0.merge(a1, on=["betr_ID", "crop"], suffixes=(f"_{y0}", f"_{y1}"))
    if m.empty:
        summ.add("  [skip] no common (betr_ID, crop)")
        return None
    c0 = m[f"C_{y0}"].to_numpy()
    c1 = m[f"C_{y1}"].to_numpy()
    d = c1 - c0
    m["dC"] = d
    # area weight = max area across years (used for area-weighted by-crop mean)
    m["area_ha"] = np.fmax(m.get(f"area_ha_{y0}", np.nan),
                           m.get(f"area_ha_{y1}", np.nan))
    summ.add(f"  common farm x crop = {len(m):,};  "
             f"dC mean={d.mean():+.4f}  median={np.median(d):+.4f}  "
             f"mean|dC|={np.abs(d).mean():.4f}  "
             f"corr={np.corrcoef(c0, c1)[0, 1]:.3f}")

    # --- figure: hexbin + dC histogram (mirror of pixel products) ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    hb = axes[0].hexbin(c0, c1, gridsize=40, mincnt=1, cmap="viridis", bins="log")
    lim = [float(min(c0.min(), c1.min())), float(max(c0.max(), c1.max()))]
    axes[0].plot(lim, lim, "r--", lw=1)
    axes[0].set_xlabel(f"C {y0}"); axes[0].set_ylabel(f"C {y1}")
    axes[0].set_title(f"{PROD_LABELS['previous']}: C per (farm,crop), {y0} vs {y1}")
    fig.colorbar(hb, ax=axes[0], label="log10(units)")
    axes[1].hist(d, bins=60, color=PROD_COLORS["previous"], alpha=0.85)
    axes[1].axvline(0, color="k", lw=0.8)
    axes[1].set_xlabel(f"dC = {y1} - {y0}"); axes[1].set_ylabel("(farm,crop) units")
    axes[1].set_title(f"Year-to-year change (dC = {y1} - {y0})")
    fig.tight_layout()
    savefig(fig, "acrossyear_previous_scatterhist.png", cfg)

    # --- by-crop bar (area-weighted mean dC), mirroring _breakdown_change ---
    g = (m.assign(_w=m["area_ha"].fillna(0.0),
                  _wd=m["dC"] * m["area_ha"].fillna(0.0))
            .groupby("crop")
            .agg(mean_d=("dC", "mean"),
                 area_weighted_d=("_wd", "sum"),
                 area_ha=("_w", "sum"),
                 n=("dC", "size")))
    g["area_weighted_d"] = np.where(g["area_ha"] > 0,
                                    g["area_weighted_d"] / g["area_ha"], np.nan)
    g_all = g.sort_values("area_ha", ascending=False)
    _savecsv(g_all.reset_index(), "acrossyear_previous_by_crop.csv", cfg)

    # Use the area-weighted dC for plotting (consistent with the rest of the
    # R-product analysis) and fall back to the unweighted mean where area is 0.
    # We rebuild a thin frame the shared helpers can plot.
    g_plot_all = pd.DataFrame({
        "mean_d": g_all["area_weighted_d"].fillna(g_all["mean_d"]),
        "n":      g_all["n"],
    }, index=g_all.index)

    g_top = g_plot_all.head(cfg["top_n_crops"])
    _plot_by_crop_horizontal(
        g_top, "acrossyear_previous", cfg, direction=f"{y1} - {y0}",
        caption=(f"top {len(g_top)} of {len(g_plot_all)} crops by area "
                 f"-- bars are area-weighted mean dC"),
        fname="acrossyear_previous_by_crop.png")
    if len(g_plot_all) > len(g_top):
        _plot_by_crop_landscape(
            g_plot_all, "acrossyear_previous", cfg, direction=f"{y1} - {y0}",
            caption=(f"all {len(g_plot_all)} crops "
                     f"-- bars are area-weighted mean dC"),
            fname="acrossyear_previous_by_crop_all.png")

    # --- by-Region quick text (kept) + plot if useful ---
    if f"Region_{y0}" in m.columns:
        reg_rows = []
        for reg, gg in m.groupby(f"Region_{y0}"):
            dd = gg[f"C_{y1}"] - gg[f"C_{y0}"]
            summ.add(f"    {reg}: dC mean={dd.mean():+.4f} (n={len(gg):,})")
            reg_rows.append({"Region": reg, "mean_dC": dd.mean(), "n": len(gg)})
        if reg_rows:
            _savecsv(pd.DataFrame(reg_rows), "acrossyear_previous_by_region.csv", cfg)
    return m


def _agg_farm_crop(df: pd.DataFrame, val: str, wt: str) -> pd.DataFrame:
    """Area-weighted mean of ``val`` per (betr_ID, crop), keep first Region/Gemeinde."""
    df = df.copy()
    df["_w"] = pd.to_numeric(df[wt], errors="coerce")
    df["_wv"] = pd.to_numeric(df[val], errors="coerce") * df["_w"]
    agg = (df.groupby(["betr_ID", "crop"], as_index=False)
             .agg(_wv=("_wv", "sum"), _w=("_w", "sum"),
                  Region=("Region", "first") if "Region" in df else ("crop", "first"),
                  Gemeinde=("Gemeinde", "first") if "Gemeinde" in df else ("crop", "first")))
    agg["C"] = agg["_wv"] / agg["_w"]
    agg["area_ha"] = agg["_w"] / 1e4
    return agg.drop(columns=["_wv", "_w"])


def _pixel_product_to_farm_crop(product_key: str, cfg: dict, year: int,
                                bridge: pd.DataFrame,
                                lnf_bridge: pd.DataFrame,
                                crop_by_lnf: dict,
                                crop_by_poly: dict | None = None
                                ) -> pd.DataFrame | None:
    """Roll a per-pixel product up to (betr_ID, crop) area-weighted C for one year.

    Crop labelling follows ``cfg['crop_label_method']``: either the kulturmapping/
    C_Faktoren bridge ('bridge') or the LNF gpkg's nutzung_DE ('lnf').
    Returned columns: betr_ID, crop, C, area_ha.
    """
    fld = load_field_summary(product_key, cfg, year)
    if fld is None or lnf_bridge is None:
        return None
    j = fld.merge(lnf_bridge[["poly_id", "betr_ID"]], on="poly_id", how="left")
    j["crop"] = _label_field_crops(j, crop_by_lnf, crop_by_poly, cfg,
                                   poly_col="poly_id", lnf_col="lnf_code",
                                   year=year)
    return _agg_farm_crop(
        j.rename(columns={"c_factor_mean": "C_fact_detail", "area_m2": "Flaeche"}),
        "C_fact_detail", "Flaeche")


def stability_three_way(cfg: dict, crop_bridge: pd.DataFrame,
                        lnf_bridge: dict, crop_by_lnf: dict,
                        r_merged: pd.DataFrame | None, summ: Summary,
                        crop_by_poly: dict | None = None):
    """Year-to-year stability of all 3 products, per crop, on (betr_ID, crop).

    Why: a single chart that answers "which product is most stable for *which*
    crops?". All three are rolled up to (betr_ID, crop) so the comparison is
    apples-to-apples (the R product has no pixels). Per-crop bars show
    ``mean|dC|`` (lower = more stable). The matching CSV also carries signed
    ``mean dC`` and n-pairs so you can see direction + sample size.
    Crop labelling source is governed by ``cfg['crop_label_method']``.
    """
    years = cfg["years"]
    if len(years) < 2:
        return
    y0, y1 = years[0], years[-1]
    summ.section(f"A'. Stability of 3 products on (betr_ID, crop), {y0} vs {y1}")

    # Build per-product (betr_ID, crop) merged frames carrying dC = y1 - y0.
    per_prod = {}
    for key in ("empirical", "ml"):
        if crop_bridge is None:
            continue
        a0 = _pixel_product_to_farm_crop(key, cfg, y0, crop_bridge,
                                         lnf_bridge.get(y0), crop_by_lnf,
                                         crop_by_poly=crop_by_poly)
        a1 = _pixel_product_to_farm_crop(key, cfg, y1, crop_bridge,
                                         lnf_bridge.get(y1), crop_by_lnf,
                                         crop_by_poly=crop_by_poly)
        if a0 is None or a1 is None:
            continue
        mm = a0.merge(a1, on=["betr_ID", "crop"], suffixes=(f"_{y0}", f"_{y1}"))
        if mm.empty:
            continue
        mm["dC"] = mm[f"C_{y1}"] - mm[f"C_{y0}"]
        mm["area_ha"] = np.fmax(mm[f"area_ha_{y0}"], mm[f"area_ha_{y1}"])
        per_prod[key] = mm

    if r_merged is not None and not r_merged.empty:
        per_prod["previous"] = r_merged[["betr_ID", "crop", "dC", "area_ha"]].copy()

    if not per_prod:
        summ.add("  [skip] no products usable for 3-way stability comparison")
        return

    # Per-crop stats, per product
    rows = []
    for key, mm in per_prod.items():
        g = (mm.groupby("crop")
               .agg(n_pairs=("dC", "size"),
                    mean_dC=("dC", "mean"),
                    mean_abs_dC=("dC", lambda s: float(np.abs(s).mean())),
                    median_abs_dC=("dC",
                                    lambda s: float(np.median(np.abs(s)))),
                    area_ha=("area_ha", "sum")))
        for crop, r in g.iterrows():
            rows.append({"product": key, "crop": crop, **r.to_dict()})
    df = pd.DataFrame(rows)
    _savecsv(df, f"stability_threeway_{y0}_{y1}_by_crop.csv", cfg)

    # Wide for plotting: mean|dC| per crop x product, rank crops by area
    wide_abs = df.pivot_table(index="crop", columns="product",
                              values="mean_abs_dC")
    area_per_crop = (df.groupby("crop")["area_ha"].max())
    wide_abs["area_ha"] = area_per_crop
    if cfg.get("show_all_crops", False):
        plot_df = wide_abs.sort_values("area_ha", ascending=False)
        cap = f"all {len(plot_df)} crops"
    else:
        plot_df = (wide_abs.sort_values("area_ha", ascending=False)
                           .head(cfg["top_n_crops"]))
        cap = (f"top {len(plot_df)} of {len(wide_abs)} crops by area "
               f"(across all 3 products)")
    plot_df = plot_df.drop(columns=["area_ha"], errors="ignore")
    plot_df = plot_df.sort_values(
        [c for c in ("previous", "empirical", "ml") if c in plot_df.columns][:1],
        ascending=False)

    prods_present = [p for p in ("empirical", "ml", "previous") if p in plot_df.columns]
    if not prods_present:
        summ.add("  [skip] no product columns to plot")
        return

    # Grouped horizontal bars: easier to read crop labels.
    n_crops = len(plot_df); n_prods = len(prods_present)
    fig, ax = plt.subplots(figsize=(8, max(3.2, 0.36 * n_crops)))
    ypos = np.arange(n_crops); bw = 0.8 / max(1, n_prods)
    for i, p in enumerate(prods_present):
        off = (i - (n_prods - 1) / 2) * bw
        ax.barh(ypos + off, plot_df[p].to_numpy(), height=bw,
                color=PROD_COLORS[p], label=PROD_LABELS[p])
    ax.set_yticks(ypos); ax.set_yticklabels(plot_df.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(f"mean |dC| year-to-year  (dC = {y1} - {y0}; lower = more stable)")
    ax.set_title(f"Year-to-year stability per crop, by product  "
                 f"({y0} -> {y1})\n{cap}")
    ax.legend(loc="lower right", fontsize=8)
    ax.axvline(0, color="k", lw=0.6)
    fig.tight_layout()
    savefig(fig, f"stability_threeway_{y0}_{y1}_meanAbsDC_by_crop.png", cfg)

    # And a paired signed mean dC bar (so direction is visible too) -- often the
    # interesting question is `which product drifts up / which down by crop`.
    wide_signed = df.pivot_table(index="crop", columns="product", values="mean_dC")
    common = [c for c in plot_df.index if c in wide_signed.index]
    wide_signed = wide_signed.loc[common, prods_present]
    fig, ax = plt.subplots(figsize=(8, max(3.2, 0.36 * len(wide_signed))))
    ypos = np.arange(len(wide_signed))
    for i, p in enumerate(prods_present):
        off = (i - (n_prods - 1) / 2) * bw
        ax.barh(ypos + off, wide_signed[p].to_numpy(), height=bw,
                color=PROD_COLORS[p], label=PROD_LABELS[p])
    ax.set_yticks(ypos); ax.set_yticklabels(wide_signed.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(f"mean signed dC = {y1} - {y0}  (red side = product drifted UP)")
    ax.set_title(f"Year-to-year drift per crop, by product  ({y0} -> {y1})\n{cap}")
    ax.axvline(0, color="k", lw=0.8)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    savefig(fig, f"stability_threeway_{y0}_{y1}_signedDC_by_crop.png", cfg)

    # Summary lines
    for key in prods_present:
        s = df[df["product"] == key]
        summ.add(f"  {PROD_LABELS[key]:<18} "
                 f"overall mean|dC|={s['mean_abs_dC'].mean():.4f}  "
                 f"crops with n_pairs>=10: {(s['n_pairs']>=10).sum()}")


# ===========================================================================
# B. WITHIN a year, ACROSS the products
# ===========================================================================

def empirical_vs_ml_pixel(cfg: dict, year: int, elev_by_poly: dict,
                          crop_by_lnf: dict, centroid_by_poly: dict,
                          summ: Summary, make_maps: bool,
                          crop_by_poly: dict | None = None):
    """Pixel-for-pixel empirical vs ML for ``year`` (the headline comparison)."""
    summ.add(f"\n-- Empirical vs ML, {year} (per pixel) --")
    e = load_pixels("empirical", cfg, year)
    ml = load_pixels("ml", cfg, year)
    if e is None or ml is None:
        summ.add("  [skip] a product is missing for this year")
        return None
    for p in (e, ml):
        kx, ky = _cell_key(p["x"].values, p["y"].values)
        p["k"] = kx.astype(np.int64) * 1_000_000_007 + ky
    m = e.merge(ml, on="k", suffixes=("_emp", "_ml"))
    if m.empty:
        summ.add("  [skip] no shared pixels (unexpected -- same grid)")
        return None

    ce, cm = m["c_factor_emp"].to_numpy(), m["c_factor_ml"].to_numpy()
    d = cm - ce
    mae = np.abs(d).mean()
    summ.add(f"  shared px = {len(m):,};  emp mean={ce.mean():.4f}  ml mean={cm.mean():.4f}")
    summ.add(f"  ML-emp: bias={d.mean():+.4f}  MAE={mae:.4f}  RMSE={np.sqrt((d**2).mean()):.4f}  "
             f"corr={np.corrcoef(ce, cm)[0, 1]:.3f}")
    summ.add(f"  agreement within +/-{cfg['agree_tol']}: {100*(np.abs(d) <= cfg['agree_tol']).mean():.0f}%")

    # figure: hexbin + signed-diff histogram
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    hb = axes[0].hexbin(ce, cm, gridsize=60, mincnt=1, cmap="viridis", bins="log")
    lim = [min(ce.min(), cm.min()), max(ce.max(), cm.max())]
    axes[0].plot(lim, lim, "r--", lw=1)
    axes[0].set_xlabel("Empirical C"); axes[0].set_ylabel("ML C")
    axes[0].set_title(f"Empirical vs ML, {year}")
    fig.colorbar(hb, ax=axes[0], label="log10(pixels)")
    axes[1].hist(d, bins=80, color="#AA3377", alpha=0.85)
    axes[1].axvline(0, color="k", lw=0.8)
    axes[1].set_xlabel("ML - Empirical"); axes[1].set_ylabel("pixels")
    axes[1].set_title("Per-pixel difference")
    fig.tight_layout()
    savefig(fig, f"withinyear_{year}_emp_vs_ml_pixel.png", cfg)

    # breakdowns
    m["lnf_code"] = m["lnf_code_emp"]
    m["dC"] = d
    _breakdown_change(m, "dC", crop_by_lnf, elev_by_poly,
                      poly_col="poly_id_emp", cfg=cfg,
                      tag=f"withinyear_{year}_emp_vs_ml", summ=summ,
                      direction="ML - Empirical",
                      crop_by_poly=crop_by_poly, year=year)

    # maps: difference GeoTIFF (full res) + inline PNG (if small) + field map
    _emit_diff_maps(m["x_emp"].to_numpy(), m["y_emp"].to_numpy(), d,
                    m["poly_id_emp"].to_numpy(), centroid_by_poly, cfg,
                    tag=f"withinyear_{year}_emp_vs_ml",
                    title=f"ML - Empirical, {year}", make_maps=make_maps,
                    direction="ML - Empirical")
    return m


def _categorical_agree_map(x, y, code, cfg, year):
    """Discrete match / ML>emp / emp>ML map (GeoTIFF codes + inline PNG)."""
    from matplotlib.colors import ListedColormap, BoundaryNorm
    df = pd.DataFrame({"x": x, "y": y, "val": code})
    if cfg.get("map_write_geotiff", True):
        # codes 0/1/2 (nodata = NaN); style in QGIS as needed
        _write_diff_geotiff(df, "val", cfg, f"disagree_{year}_classmap.tif")
    arr, ext = _rasterise(df, "val", cfg)
    if arr is None:
        return
    cmap = ListedColormap(["#DDDDDD", "#CC6677", "#4477AA"])  # match, ML>emp, emp>ML
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(arr, extent=ext, origin="upper", cmap=cmap, norm=norm,
                   interpolation="nearest")
    ax.set_aspect("equal"); ax.set_xlabel("E (m)"); ax.set_ylabel("N (m)")
    ax.set_title(f"Empirical vs ML agreement, {year}")
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2], shrink=0.7)
    cbar.ax.set_yticklabels(["match", "ML > emp", "emp > ML"])
    savefig(fig, f"disagree_{year}_classmap.png", cfg)


def _field_value_map(series_by_poly, centroid_by_poly, cfg, title, label, fname,
                     vmin=None, vmax=None, cmap="OrRd"):
    """Sequential per-field value map at LNF centroids (e.g. disagreement rate)."""
    xs, ys, vs = [], [], []
    for poly, v in series_by_poly.items():
        c = centroid_by_poly.get(int(poly))
        if c is not None and pd.notna(v):
            xs.append(c[0]); ys.append(c[1]); vs.append(v)
    if not vs:
        print(f"  [INFO] no field centroids -- skipping {fname}")
        return
    xs, ys, vs = np.asarray(xs), np.asarray(ys), np.asarray(vs)
    size = float(np.clip(900.0 / np.sqrt(max(len(vs), 1)), 2.5, 9.0))
    order = np.argsort(vs)                              # high values drawn on top
    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(xs[order], ys[order], c=vs[order], cmap=cmap, vmin=vmin, vmax=vmax,
                    s=size, marker="s", alpha=0.85, linewidths=0.12,
                    edgecolors="white", zorder=3)
    ax.set_aspect("equal")
    ax.set_xlabel("E (m, LV95)"); ax.set_ylabel("N (m, LV95)"); ax.margins(0.01)
    _add_basemap(ax, cfg, crs="EPSG:2056")
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, label=label, shrink=0.7)
    savefig(fig, fname, cfg, dpi=200)


def disagreement_analysis(m, cfg, year, crop_by_lnf, centroid_by_poly,
                          summ: Summary, make_maps: bool,
                          crop_by_poly: dict | None = None):
    """WHERE do empirical and ML disagree, on WHICH crops, and is the mismatch
    spatially clustered? Uses the merged per-pixel frame from
    ``empirical_vs_ml_pixel`` (cols: x_emp, y_emp, poly_id_emp, lnf_code, dC).

    Disagreement = |dC| > agree_tol. Outputs:
      disagree_{year}_by_crop.{csv,png}     crops driving the disagreeing area
      disagree_{year}_crop_lift.csv          crops over/under-represented in mismatch
      disagree_{year}_classmap.{tif,png}     match / ML>emp / emp>ML spatial pattern
      disagree_{year}_fieldrate.png          per-field disagreement rate (clustering)
      disagree_{year}_fieldmagnitude.png     per-field mean |dC| (where products differ
                                             MOST, sign-agnostic)
    Crop labelling source is governed by ``cfg['crop_label_method']``.
    """
    if m is None or len(m) == 0:
        return
    tol = cfg["agree_tol"]
    summ.add(f"\n-- Disagreement (|dC| > {tol}), {year} --")
    d = m["dC"].to_numpy()
    absd = np.abs(d)
    disagree = absd > tol
    n, ndis = len(m), int(disagree.sum())
    if ndis == 0:
        summ.add("  products agree everywhere within tolerance")
        return
    summ.add(f"  {ndis:,}/{n:,} px disagree ({100*ndis/n:.1f}%);  of those: "
             f"ML>emp {100*(d[disagree] > 0).mean():.0f}%, "
             f"emp>ML {100*(d[disagree] < 0).mean():.0f}%")

    crop = _label_field_crops(m, crop_by_lnf, crop_by_poly, cfg,
                              poly_col="poly_id_emp", lnf_col="lnf_code",
                              year=year)
    tab = pd.DataFrame({"crop": crop.values, "absd": absd, "d": d, "dis": disagree})

    # ---- crops driving the disagreeing area ----
    by = (tab.groupby("crop")
             .agg(n_px=("dis", "size"), n_disagree=("dis", "sum"),
                  mean_absdC=("absd", "mean"), mean_signed_dC=("d", "mean")))
    by["disagree_rate"] = by["n_disagree"] / by["n_px"]
    by["share_of_disagree"] = by["n_disagree"] / ndis
    by = by.sort_values("n_disagree", ascending=False)
    _savecsv(by.reset_index(), f"disagree_{year}_by_crop.csv", cfg)

    summ.add("  crops contributing most disagreeing pixels (share | within-crop rate | mean|dC|):")
    for c, row in by.head(8).iterrows():
        summ.add(f"    {str(c):<28} {100*row.share_of_disagree:4.1f}%  "
                 f"rate={100*row.disagree_rate:4.0f}%  |dC|={row.mean_absdC:.3f}")

    show = by.head(cfg["top_n_crops"]).sort_values("share_of_disagree")
    fig, ax = plt.subplots(figsize=(7.5, max(3, 0.34 * len(show))))
    ax.barh(range(len(show)), 100 * show["share_of_disagree"],
            color=plt.cm.OrRd(np.clip(show["disagree_rate"].to_numpy(), 0, 1)))
    ax.set_yticks(range(len(show))); ax.set_yticklabels(show.index, fontsize=8)
    ax.set_xlabel("% of all disagreeing pixels")
    ax.set_title(f"Disagreeing area by crop, {year}\n"
                 f"(bar colour = within-crop disagreement rate)")
    sm = plt.cm.ScalarMappable(cmap="OrRd", norm=plt.Normalize(0, 1)); sm.set_array([])
    fig.colorbar(sm, ax=ax, label="within-crop disagree rate")
    fig.tight_layout()
    savefig(fig, f"disagree_{year}_by_crop.png", cfg)

    # ---- crops over/under-represented in disagreement (lift) ----
    overall_share = tab.groupby("crop").size() / n
    dis_share = tab[tab["dis"]].groupby("crop").size() / ndis
    lift_df = pd.DataFrame({"overall_share": overall_share, "disagree_share": dis_share})
    lift_df["lift"] = lift_df["disagree_share"] / lift_df["overall_share"]
    lift_df = lift_df.sort_values("lift", ascending=False)
    _savecsv(lift_df.reset_index(), f"disagree_{year}_crop_lift.csv", cfg)
    over = lift_df[(lift_df["overall_share"] >= 0.01) & lift_df["lift"].notna()].head(5)
    if not over.empty:
        summ.add("  crops over-represented in disagreement (lift>1 = more mismatch than its area share):")
        for c, row in over.iterrows():
            summ.add(f"    {str(c):<28} lift={row.lift:.2f}  (area {100*row.overall_share:.1f}%)")

    # ---- spatial structure: per-field disagreement rate ----
    fld = (pd.DataFrame({"poly": m["poly_id_emp"].values, "dis": disagree})
             .groupby("poly")["dis"].mean())
    lo = float((fld <= 0.05).mean()); hi = float((fld >= 0.50).mean())
    summ.add(f"  per-field rate: {100*lo:.0f}% of fields ~agree (<=5% px), "
             f"{100*hi:.0f}% mostly disagree (>=50% px) -> "
             f"{'mismatch is field-structured' if hi > 0.10 else 'mismatch is mostly scattered'}")

    if make_maps:
        code = np.where(~disagree, 0.0, np.where(d > 0, 1.0, 2.0))
        _categorical_agree_map(m["x_emp"].to_numpy(), m["y_emp"].to_numpy(),
                               code, cfg, year)
        _field_value_map(fld, centroid_by_poly, cfg,
                         title=f"Field disagreement RATE, {year} (Empirical vs ML)",
                         label="fraction of field pixels with |dC| > tol",
                         fname=f"disagree_{year}_fieldrate.png",
                         vmin=0, vmax=1, cmap="OrRd")
        # NEW: magnitude map -- mean |dC| per field. Tells you WHERE differences
        # between the two products are big vs small (independent of sign).
        _field_absdiff_map(m["poly_id_emp"].to_numpy(), d, centroid_by_poly, cfg,
                           title=f"Where products differ MOST, {year}",
                           fname=f"disagree_{year}_fieldmagnitude.png",
                           direction="ML - Empirical")


def _breakdown_change(m: pd.DataFrame, dcol: str, crop_by_lnf: dict,
                      elev_by_poly: dict, poly_col: str, cfg: dict,
                      tag: str, summ: Summary, direction: str = "",
                      also_all_crops: bool = False,
                      crop_by_poly: dict | None = None,
                      year: int | None = None):
    """Mean signed difference by crop and by elevation band; writes CSV + bar chart.

    ``direction`` (e.g. '2022 - 2021', 'ML - Empirical') is shown on the bar-chart
    axis and title so the sign of dC cannot be misread.

    Two by-crop plots are written when ``also_all_crops=True``:
      * ``{tag}_by_crop.png``     -- top ``cfg['top_n_crops']`` crops by pixel
                                     count, horizontal-bar layout (compact).
      * ``{tag}_by_crop_all.png`` -- ALL crops, landscape vertical-bar layout
                                     (so a long crop list fits across the page).
    The exported CSV always contains every crop.

    Labelling source is governed by ``cfg['crop_label_method']`` -- see
    ``_label_field_crops``. ``year`` selects the per-year crop_by_poly map.
    """
    df = m[[poly_col, "lnf_code", dcol]].copy()
    df["crop"] = _label_field_crops(df, crop_by_lnf, crop_by_poly, cfg,
                                    poly_col=poly_col, lnf_col="lnf_code",
                                    year=year)

    by_crop_all = (df.groupby("crop")
                     .agg(mean_d=(dcol, "mean"), median_d=(dcol, "median"),
                          n=(dcol, "size"))
                     .sort_values("n", ascending=False))
    _savecsv(by_crop_all.reset_index(), f"{tag}_by_crop.csv", cfg)   # all crops

    # --- top-N horizontal-bar plot (compact, the default reading) ---
    by_crop_top = by_crop_all.head(cfg["top_n_crops"])
    crop_caption_top = (f"top {len(by_crop_top)} of {len(by_crop_all)} crops "
                        f"by pixel count")
    _plot_by_crop_horizontal(by_crop_top, tag, cfg, direction, crop_caption_top,
                             fname=f"{tag}_by_crop.png")

    # --- optional all-crops landscape plot ---
    if also_all_crops and len(by_crop_all) > len(by_crop_top):
        _plot_by_crop_landscape(by_crop_all, tag, cfg, direction,
                                f"all {len(by_crop_all)} crops",
                                fname=f"{tag}_by_crop_all.png")

    # elevation band
    if elev_by_poly:
        df["elev"] = df[poly_col].map(elev_by_poly)
        df = df.dropna(subset=["elev"])
        if not df.empty:
            df["band"] = pd.cut(df["elev"], bins=cfg["elev_bins"])
            by_el = df.groupby("band").agg(mean_d=(dcol, "mean"), n=(dcol, "size"))
            _savecsv(by_el.reset_index().astype({"band": str}),
                     f"{tag}_by_elev.csv", cfg)
            dir_txt = f"  (dC = {direction})" if direction else ""
            summ.add(f"  by elevation band (mean dC{dir_txt}):")
            for band, row in by_el.iterrows():
                summ.add(f"    {str(band):<18} {row['mean_d']:+.4f}  (n={int(row['n']):,})")


def _plot_by_crop_horizontal(by_crop: pd.DataFrame, tag: str, cfg: dict,
                             direction: str, caption: str, fname: str):
    """Compact horizontal-bar by-crop chart used for the top-N selection."""
    fig, ax = plt.subplots(figsize=(7, max(3, 0.32 * len(by_crop))))
    order = by_crop.sort_values("mean_d")
    ax.barh(range(len(order)), order["mean_d"],
            color=np.where(order["mean_d"] >= 0, "#CC6677", "#4477AA"))
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order.index, fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    xlab = f"mean dC = {direction}" if direction else "mean dC"
    ax.set_xlabel(xlab + "  (red = positive, blue = negative)")
    title = f"{tag}: mean dC by crop"
    if direction:
        title += f"  (dC = {direction})"
    title += f"\n{caption}"
    ax.set_title(title)
    fig.tight_layout()
    savefig(fig, fname, cfg)


def _plot_by_crop_landscape(by_crop: pd.DataFrame, tag: str, cfg: dict,
                            direction: str, caption: str, fname: str):
    """Wide vertical-bar by-crop chart used when ALL crops are shown.

    Width scales with the number of crops so labels don't collide. Crops are
    sorted by mean_d (most negative -> most positive), matching the top-N
    chart's reading order and producing a clean monotonic visual.
    """
    order = by_crop.sort_values("mean_d")
    n = len(order)
    # ~0.22 in per crop, with sensible bounds; height tuned for rotated labels
    width = float(np.clip(0.22 * n + 4.0, 8.0, 48.0))
    fig, ax = plt.subplots(figsize=(width, 5.0))
    xpos = np.arange(n)
    ax.bar(xpos, order["mean_d"],
           color=np.where(order["mean_d"] >= 0, "#CC6677", "#4477AA"))
    ax.set_xticks(xpos)
    ax.set_xticklabels(order.index, rotation=75, ha="right",
                       fontsize=7 if n > 40 else 8)
    ax.axhline(0, color="k", lw=0.8)
    ylab = f"mean dC = {direction}" if direction else "mean dC"
    ax.set_ylabel(ylab + "  (red>0, blue<0)")
    title = f"{tag}: mean dC by crop"
    if direction:
        title += f"  (dC = {direction})"
    title += f"  --  {caption}"
    ax.set_title(title)
    ax.margins(x=0.005)
    fig.tight_layout()
    savefig(fig, fname, cfg)


def three_way_per_crop(cfg: dict, year: int, bridge: pd.DataFrame,
                       lnf_bridge: pd.DataFrame, crop_by_lnf: dict,
                       summ: Summary, crop_by_poly: dict | None = None):
    """Area-weighted mean C per crop for all three products (distribution-level),
    plus paired (betr_ID, crop) residuals emp-R, ml-R, emp-ml.
    Crop labelling source is governed by ``cfg['crop_label_method']``.
    """
    summ.add(f"\n-- Three-way comparison, {year} (farm x crop) --")

    # New products -> (betr_ID, crop) area-weighted means
    fc = {}
    for key in ("empirical", "ml"):
        fld = load_field_summary(key, cfg, year)
        if fld is None or lnf_bridge is None:
            continue
        j = fld.merge(lnf_bridge[["poly_id", "betr_ID"]], on="poly_id", how="left")
        j["crop"] = _label_field_crops(j, crop_by_lnf, crop_by_poly, cfg,
                                       poly_col="poly_id", lnf_col="lnf_code",
                                       year=year)
        fc[key] = _agg_farm_crop(
            j.rename(columns={"c_factor_mean": "C_fact_detail", "area_m2": "Flaeche"}),
            "C_fact_detail", "Flaeche")

    r = load_r_results(cfg, year)
    rr = _agg_farm_crop(r, "C_fact_detail", "Flaeche") if r is not None else None

    # --- per-crop area-weighted mean for each product (its own coverage) ---
    rows = []
    sources = {"empirical": fc.get("empirical"), "ml": fc.get("ml"), "previous": rr}
    for key, a in sources.items():
        if a is None:
            continue
        g = (a.assign(_wv=a["C"] * a["area_ha"])
               .groupby("crop")
               .agg(area_ha=("area_ha", "sum"), _wv=("_wv", "sum")))
        g["C"] = g["_wv"] / g["area_ha"]
        for crop, row in g.iterrows():
            rows.append({"crop": crop, "product": key, "C": row["C"],
                         "area_ha": row["area_ha"]})
    if not rows:
        summ.add("  [skip] nothing to compare")
        return
    per_crop = pd.DataFrame(rows)
    wide = per_crop.pivot_table(index="crop", columns="product", values="C")
    area = per_crop.pivot_table(index="crop", columns="product", values="area_ha")
    wide["area_ha"] = area.max(axis=1)
    wide = wide.sort_values("area_ha", ascending=False).head(cfg["top_n_crops"])
    _savecsv(wide.reset_index(), f"threeway_{year}_per_crop_meanC.csv", cfg, index=False)

    # grouped bar chart
    prod_present = [p for p in ("empirical", "ml", "previous") if p in wide.columns]
    fig, ax = plt.subplots(figsize=(max(7, 0.6 * len(wide)), 4.5))
    xpos = np.arange(len(wide)); bw = 0.8 / max(1, len(prod_present))
    for i, p in enumerate(prod_present):
        ax.bar(xpos + (i - (len(prod_present) - 1) / 2) * bw, wide[p], width=bw,
               color=PROD_COLORS[p], label=PROD_LABELS[p])
    ax.set_xticks(xpos); ax.set_xticklabels(wide.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("area-weighted mean C"); ax.legend()
    ax.set_title(f"Mean C per crop by product, {year}")
    fig.tight_layout()
    savefig(fig, f"threeway_{year}_per_crop_meanC.png", cfg)

    # --- paired residuals on the common (betr_ID, crop) set ---
    pairs = [("empirical", "previous", fc.get("empirical"), rr),
             ("ml", "previous", fc.get("ml"), rr),
             ("empirical", "ml", fc.get("empirical"), fc.get("ml"))]
    for a_name, b_name, A, B in pairs:
        if A is None or B is None:
            continue
        j = A.merge(B, on=["betr_ID", "crop"], suffixes=("_a", "_b"))
        if j.empty:
            continue
        d = j["C_a"] - j["C_b"]
        summ.add(f"  {PROD_LABELS[a_name]} - {PROD_LABELS[b_name]}: "
                 f"n={len(j):,}  bias={d.mean():+.4f}  MAE={d.abs().mean():.4f}  "
                 f"corr={np.corrcoef(j['C_a'], j['C_b'])[0, 1]:.3f}")
        # by Region from whichever side carries it
        reg_col = "Region_a" if "Region_a" in j and j["Region_a"].notna().any() else (
                  "Region_b" if "Region_b" in j else None)
        if reg_col:
            for reg, g in j.groupby(reg_col):
                if str(reg) in ("nan", "<NA>"):
                    continue
                dd = g["C_a"] - g["C_b"]
                summ.add(f"      {reg}: bias={dd.mean():+.4f} (n={len(g):,})")


# ===========================================================================
# Synthesis
# ===========================================================================

def synthesis(cfg: dict, summ: Summary):
    """Plausibility per product/year + a short decision cheat-sheet."""
    summ.section("C. SYNTHESIS -- strengths / weaknesses / decision hints")

    summ.add("\nValue plausibility (per-pixel products) and reported range (R):")
    for year in cfg["years"]:
        summ.add(f"\n  Year {year}")
        for key in ("empirical", "ml"):
            pix = load_pixels(key, cfg, year)
            if pix is not None:
                plausibility(PROD_LABELS[key], pix["c_factor"], summ)
        r = load_r_results(cfg, year)
        if r is not None:
            plausibility(PROD_LABELS["previous"], r["C_fact_detail"], summ)

    summ.add("\nHow to read the outputs:")
    summ.add("  * acrossyear_*  -> stability of each product 2021->2022 "
             "(smaller mean|dC| = more temporally stable). Sign convention: "
             "dC = later year - earlier year.")
    summ.add("  * acrossyear_previous_*  -> same idea for the R product, on "
             "(betr_ID, crop). scatterhist + by_crop bar.")
    summ.add("  * stability_threeway_* -> side-by-side mean|dC| by crop for "
             "all 3 products on common (betr_ID, crop). Use this chart to spot "
             "which crops are unstable in which product.")
    summ.add("  * withinyear_*_emp_vs_ml_*  -> where the two satellite products "
             "diverge; sign convention: dC = ML - Empirical. Check the by_crop "
             "/ by_elev tables for the drivers.")
    summ.add("  * disagree_{year}_*  -> where |dC|>tol concentrates: by_crop / "
             "crop_lift (which crops), classmap (match vs ML>emp vs emp>ML spatial "
             "pattern), fieldrate (how field-structured the mismatch is), "
             "fieldmagnitude (mean |dC| per field -- where differences are big).")
    summ.add("  * threeway_*_per_crop_meanC -> how each product positions each crop "
             "vs the established R lookup (systematic high/low by crop).")
    summ.add("  * plausibility above -> ML is clipped to [0,1]; a high share at the "
             "0 or ~1 bounds flags saturation. Empirical values <0 or >1 flag "
             "extrapolation of exp(-b.FC) beyond the calibrated range.")
    summ.add("\nChoosing a product (rules of thumb):")
    summ.add("  - Need agreement with the official indicator -> the product closest "
             "to 'previous' per crop (see threeway bias).")
    summ.add("  - Need spatial/temporal detail within fields -> a per-pixel product "
             "(empirical or ML); pick the more temporally stable one unless a year "
             "had a real land-use change.")
    summ.add("  - Distrust crops where emp vs ML disagree most and where either "
             "deviates strongly from R -> candidates to improve / recalibrate.")


# ===========================================================================
# Main
# ===========================================================================

def _build_bridges(cfg: dict, region_2056, years):
    """elev_by_poly[year], crop_by_lnf, crop_bridge, lnf_bridge[year], crop_by_poly.

    ``crop_by_poly`` is ``{year: {poly_id: nutzung_DE}}``, populated only when
    ``cfg['crop_label_method'] == 'lnf'``. For each analysis year we use THAT
    year's LNF gpkg. The polygon's ``nutzung_DE`` value is used directly as
    the crop label -- this is the same vocabulary as the R output's
    ``Nutzung_DE_KatNutz`` column, so the downstream (betr_ID, crop) join
    across products lines up by construction. No round-trip through R is
    needed: the labels already live in the same string space.

    Per-year coverage is reported (polygons with a usable nutzung_DE vs total).
    Unmatched polygons fall back through ``_label_field_crops`` to the bridge
    method (and finally to the raw lnf_code string).
    """
    crop_bridge = build_crop_bridge(cfg)
    crop_by_lnf = ({int(r.lnf_code): r.crop_kat for r in crop_bridge.itertuples()
                    if pd.notna(r.lnf_code)} if crop_bridge is not None else {})
    lnf_bridge, elev_by_poly = {}, {}
    for y in years:
        lb = load_lnf_bridge(cfg, y, region_2056)
        lnf_bridge[y] = lb
        el = load_elevation(cfg, y)
        if lb is not None and el is not None:
            j = lb.merge(el, on="uuid", how="left")
            elev_by_poly[y] = dict(zip(j["poly_id"], j["swissALTI3D"]))
        else:
            elev_by_poly[y] = {}

    crop_by_poly: dict[int, dict] = {}        # {year: {poly_id: label}}
    if cfg.get("crop_label_method", "bridge") == "lnf":
        for y in years:
            crop_by_poly[y] = {}
            lb = lnf_bridge.get(y)
            if lb is None or "nutzung_DE" not in lb.columns:
                print(f"  [INFO] {y}: LNF has no nutzung_DE -- 'lnf' method "
                      f"falls back to bridge for this year")
                continue
            nut = lb["nutzung_DE"].astype(str).str.strip()
            ok = nut.notna() & nut.ne("") & (nut.str.lower() != "nan")
            year_map = dict(zip(lb.loc[ok, "poly_id"].astype(int), nut[ok]))
            crop_by_poly[y] = year_map
            n_ok, n_total = int(ok.sum()), int(len(lb))
            pct = 100 * n_ok / max(n_total, 1)
            print(f"  [INFO] {y}: {n_ok:,}/{n_total:,} polygons ({pct:.0f}%) "
                  f"have a usable nutzung_DE; the rest fall back to bridge")
    return elev_by_poly, crop_by_lnf, crop_bridge, lnf_bridge, crop_by_poly


# ===========================================================================
# D. Tillage-stratified breakdowns
#
# Two analyses:
#   D.1  within-year, across products  -- mean C per (product, crop, tillage)
#        for the top-N crops. Tells you whether emp/ML mirror R's tillage
#        stratification at all, or are flat across Pflug vs Mulch.
#   D.2  across-year, per pixel product -- mean dC per tillage transition
#        class (Pflug>Pflug / Pflug>Mulch / Mulch>Pflug / Mulch>Mulch).
#        R is excluded -- its C is *defined* by tillage so the relationship
#        is tautological.
#
# Tillage class is taken straight from the R output's `Bodenbearbeitung`
# column (populated in load_r_results above). Binary scheme `Pflug` /
# `Mulch` mirrors R's 05-Dataprep.R logic exactly.
# Join level is (betr_ID, crop) per year, matching what R works on.
# ===========================================================================

def build_tillage_map(cfg: dict, years: list) -> dict:
    """{year: {(betr_ID, crop): tillage}} from R output.

    If a (betr_ID, crop) row appears more than once for a given year
    (unusual at this granularity), the tillage from the largest-area row
    is kept.
    """
    out: dict = {}
    for y in years:
        r = load_r_results(cfg, y)
        if r is None or "tillage" not in r.columns:
            out[y] = {}
            continue
        idx = r.groupby(["betr_ID", "crop"])["Flaeche"].idxmax()
        keep = r.loc[idx, ["betr_ID", "crop", "tillage"]]
        m = dict(zip(zip(keep["betr_ID"], keep["crop"]), keep["tillage"]))
        out[y] = m
        n_pflug = sum(1 for v in m.values() if v == "Pflug")
        n_mulch = sum(1 for v in m.values() if v == "Mulch")
        n_other = len(m) - n_pflug - n_mulch
        print(f"  [tillage] {y}: {len(m):,} (betr_ID, crop) entries  "
              f"Pflug={n_pflug:,}  Mulch={n_mulch:,}  Other={n_other}")
    return out


def _aggregate_pixel_product_betr_crop(prod_key: str, cfg: dict, year: int,
                                       lnf_bridge_y: pd.DataFrame,
                                       crop_by_lnf: dict,
                                       crop_by_poly: dict):
    """Per (betr_ID, crop) area-weighted mean C for one pixel product."""
    fld = load_field_summary(prod_key, cfg, year)
    if fld is None or fld.empty:
        return None
    bridge = lnf_bridge_y[["poly_id", "betr_ID"]].drop_duplicates("poly_id")
    fld = fld.merge(bridge, on="poly_id", how="left")
    fld["crop"] = _label_field_crops(fld, crop_by_lnf, crop_by_poly, cfg,
                                     poly_col="poly_id", year=year)
    fld = fld.dropna(subset=["betr_ID", "crop"])
    if fld.empty:
        return None
    g = (fld.groupby(["betr_ID", "crop"], as_index=False)
            .apply(lambda d: pd.Series({
                "mean_C":   float(np.average(d["c_factor_mean"],
                                             weights=d["area_m2"])),
                "area_m2": float(d["area_m2"].sum()),
                "n_fields": int(len(d)),
            }))
            .reset_index(drop=True))
    g["product"] = prod_key
    return g


def _aggregate_r_betr_crop(prev: pd.DataFrame) -> pd.DataFrame:
    """Same shape as _aggregate_pixel_product_betr_crop, for R."""
    g = (prev.groupby(["betr_ID", "crop"], as_index=False)
             .apply(lambda d: pd.Series({
                 "mean_C":   float(np.average(d["C_fact_detail"],
                                              weights=d["Flaeche"])),
                 "area_m2": float(d["Flaeche"].sum()),
                 "n_fields": int(len(d)),
             }))
             .reset_index(drop=True))
    g["product"] = "previous"
    return g


def within_year_by_tillage(cfg: dict, year: int, tillage_map: dict,
                           lnf_bridge_y: pd.DataFrame,
                           crop_by_lnf: dict, crop_by_poly: dict,
                           summ: "Summary") -> None:
    """D.1 -- mean C per (product, crop, tillage) for top-N crops by area."""
    summ.add(f"\n--- D.1 Within-year tillage breakdown, {year} ---")
    if lnf_bridge_y is None:
        summ.add(f"  [skip] no LNF bridge for {year}")
        return

    parts = []
    for p in ("empirical", "ml"):
        a = _aggregate_pixel_product_betr_crop(p, cfg, year, lnf_bridge_y,
                                               crop_by_lnf, crop_by_poly)
        if a is not None:
            parts.append(a)
    prev = load_r_results(cfg, year)
    if prev is not None:
        parts.append(_aggregate_r_betr_crop(prev))
    if not parts:
        summ.add("  [skip] no product data"); return
    df = pd.concat(parts, ignore_index=True)

    tm = tillage_map.get(year, {})
    df["tillage"] = [tm.get((b, c), pd.NA)
                     for b, c in zip(df["betr_ID"], df["crop"])]
    miss = int(df["tillage"].isna().sum())
    summ.add(f"  rows without a tillage class: {miss:,} / {len(df):,}  "
             f"(dropped from figure; kept in CSV as 'Unknown')")
    df["tillage"] = df["tillage"].fillna("Unknown")

    area_per_crop = (df[df["tillage"].isin(TILLAGE_ORDER)]
                       .groupby("crop")["area_m2"].sum()
                       .sort_values(ascending=False))
    top_crops = area_per_crop.head(cfg.get("top_n_crops", 15)).index.tolist()

    tab = (df[df["tillage"].isin(TILLAGE_ORDER + ["Unknown"])]
             .groupby(["crop", "tillage", "product"], as_index=False)
             .apply(lambda d: pd.Series({
                 "mean_C":   float(np.average(d["mean_C"], weights=d["area_m2"]))
                              if d["area_m2"].sum() > 0 else np.nan,
                 "area_ha":  float(d["area_m2"].sum() / 1e4),
                 "n_fields": int(d["n_fields"].sum()),
             }))
             .reset_index(drop=True))
    _savecsv(tab, f"withinyear_{year}_meanC_by_tillage.csv", cfg)

    crops = [c for c in top_crops if c in tab["crop"].unique()]
    if not crops:
        summ.add("  [skip] no crops with Pflug/Mulch coverage"); return
    n = len(crops); ncols = 3; nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.6 * ncols, 2.3 * nrows),
                             sharey=True, squeeze=False)
    axes = axes.flatten()
    prods_present = [p for p in ("empirical", "ml", "previous")
                     if p in tab["product"].unique()]
    bw = 0.8 / max(1, len(prods_present))
    for i, c in enumerate(crops):
        ax = axes[i]
        sub = tab[(tab["crop"] == c) & (tab["tillage"].isin(TILLAGE_ORDER))]
        x = np.arange(len(TILLAGE_ORDER))
        for j, p in enumerate(prods_present):
            vals = []
            for t in TILLAGE_ORDER:
                s = sub[(sub["tillage"] == t) & (sub["product"] == p)]["mean_C"]
                vals.append(float(s.iloc[0]) if not s.empty else np.nan)
            off = (j - (len(prods_present) - 1) / 2) * bw
            ax.bar(x + off, vals, bw,
                   color=PROD_COLORS.get(p, "#888888"),
                   label=PROD_LABELS.get(p, p) if i == 0 else None)
        ax.set_xticks(x); ax.set_xticklabels(TILLAGE_ORDER, fontsize=8)
        ax.set_title(c[:34], fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_ylim(bottom=0)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    axes[0].set_ylabel("area-weighted mean C", fontsize=8)
    if prods_present:
        axes[0].legend(loc="upper right", fontsize=7, frameon=False)
    fig.suptitle(f"Mean C by tillage class and product, {year}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    savefig(fig, f"withinyear_{year}_meanC_by_tillage.png", cfg)

    # Paired residuals (emp - R, ml - R, emp - ml) per (crop, tillage)
    wide = tab.pivot_table(index=["crop", "tillage"], columns="product",
                           values="mean_C")
    if {"empirical", "ml", "previous"}.issubset(wide.columns):
        wide = wide.assign(
            d_emp_R   =lambda d: d["empirical"] - d["previous"],
            d_ml_R    =lambda d: d["ml"]        - d["previous"],
            d_emp_ml  =lambda d: d["empirical"] - d["ml"],
        ).reset_index()
        _savecsv(wide, f"withinyear_{year}_residuals_by_tillage.csv", cfg)


def across_year_by_tillage_transition(cfg: dict, y0: int, y1: int,
                                      tillage_map: dict,
                                      lnf_bridge: dict,
                                      crop_by_lnf: dict,
                                      crop_by_poly: dict,
                                      summ: "Summary") -> None:
    """D.2 -- mean dC per (tillage_y0 -> tillage_y1) per pixel product."""
    summ.add(f"\n--- D.2 Across-year tillage transitions, {y0} -> {y1} ---")
    lb0, lb1 = lnf_bridge.get(y0), lnf_bridge.get(y1)
    if lb0 is None or lb1 is None:
        summ.add("  [skip] missing LNF bridge"); return
    tm0, tm1 = tillage_map.get(y0, {}), tillage_map.get(y1, {})

    all_rows = []
    for p in ("empirical", "ml"):
        a0 = _aggregate_pixel_product_betr_crop(p, cfg, y0, lb0,
                                                crop_by_lnf, crop_by_poly)
        a1 = _aggregate_pixel_product_betr_crop(p, cfg, y1, lb1,
                                                crop_by_lnf, crop_by_poly)
        if a0 is None or a1 is None:
            summ.add(f"  [warn] missing {p} data for {y0} or {y1}"); continue
        m = a0.merge(a1, on=["betr_ID", "crop"], suffixes=(f"_{y0}", f"_{y1}"))
        m["dC"] = m[f"mean_C_{y1}"] - m[f"mean_C_{y0}"]
        m["till_y0"] = [tm0.get((b, c), pd.NA)
                        for b, c in zip(m["betr_ID"], m["crop"])]
        m["till_y1"] = [tm1.get((b, c), pd.NA)
                        for b, c in zip(m["betr_ID"], m["crop"])]
        m = m.dropna(subset=["till_y0", "till_y1"])
        m = m[m["till_y0"].isin(TILLAGE_ORDER) & m["till_y1"].isin(TILLAGE_ORDER)]
        if m.empty:
            summ.add(f"  [warn] {p}: no (betr_ID, crop) pairs with known "
                     f"tillage in both years")
            continue
        m["transition"] = m["till_y0"] + ">" + m["till_y1"]
        m["product"] = p
        m["area_m2_avg"] = 0.5 * (m[f"area_m2_{y0}"] + m[f"area_m2_{y1}"])
        all_rows.append(m[["product", "betr_ID", "crop", "transition",
                           "dC", "area_m2_avg"]])
    if not all_rows:
        summ.add("  [skip] no usable rows"); return
    df = pd.concat(all_rows, ignore_index=True)

    # Sample-size diagnostic (shared across products -- same join key)
    one = df[df["product"] == df["product"].iloc[0]]
    diag = (one.groupby("transition").size()
               .reindex(TRANSITION_ORDER, fill_value=0))
    total = int(diag.sum())
    summ.add("  sample size per transition class:")
    for t in TRANSITION_ORDER:
        pct = 100 * diag[t] / max(total, 1)
        summ.add(f"    {t:<15} {int(diag[t]):>6,}  ({pct:4.1f}%)")
    n_changed = int(diag.drop(["Pflug>Pflug", "Mulch>Mulch"]).sum())
    if n_changed < 50:
        summ.add(f"  [note] only {n_changed} cells changed tillage class -- "
                 "interpret the change-vs-stayed comparison with caution.")

    agg = (df.groupby(["product", "transition"], as_index=False)
             .apply(lambda d: pd.Series({
                 "mean_dC":   float(np.average(d["dC"],
                                               weights=d["area_m2_avg"])),
                 "median_dC": float(np.median(d["dC"])),
                 "n_fields":  int(len(d)),
                 "area_ha":   float(d["area_m2_avg"].sum() / 1e4),
             }))
             .reset_index(drop=True))
    _savecsv(agg, f"acrossyear_{y0}_{y1}_meanDC_by_tillage_transition.csv", cfg)

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    x = np.arange(len(TRANSITION_ORDER))
    prods_present = [p for p in ("empirical", "ml") if p in agg["product"].unique()]
    bw = 0.8 / max(1, len(prods_present))
    for j, p in enumerate(prods_present):
        vals, ns = [], []
        for t in TRANSITION_ORDER:
            row = agg[(agg["product"] == p) & (agg["transition"] == t)]
            vals.append(float(row["mean_dC"].iloc[0]) if not row.empty else np.nan)
            ns.append(int(row["n_fields"].iloc[0]) if not row.empty else 0)
        off = (j - (len(prods_present) - 1) / 2) * bw
        bars = ax.bar(x + off, vals, bw,
                      color=PROD_COLORS.get(p, "#888888"),
                      label=PROD_LABELS.get(p, p))
        for k, b in enumerate(bars):
            h = b.get_height()
            if not np.isnan(h):
                ax.text(b.get_x() + b.get_width() / 2,
                        h + (0.001 if h >= 0 else -0.003),
                        f"n={ns[k]}",
                        ha="center",
                        va="bottom" if h >= 0 else "top",
                        fontsize=6, color="#444444")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace(">", " \u2192 ") for t in TRANSITION_ORDER],
                       fontsize=9)
    ax.set_ylabel(f"mean dC ({y1} - {y0}), area-weighted")
    ax.set_title(f"Year-to-year dC by tillage transition, {y0} \u2192 {y1}\n"
                 f"(pixel products only; R excluded -- C is defined by tillage)")
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    savefig(fig, f"acrossyear_{y0}_{y1}_meanDC_by_tillage_transition.png", cfg)


# ===========================================================================
# E. FC TIME SERIES (calibration sample, 2021 vs 2022, overall + per-crop)
# ===========================================================================

# Day-of-agricultural-year (DOAY) reference: 1 July of yr-1 == DOAY 1.
# Month tick positions are the DOAY of the 1st of each month, computed once.
_AGRI_MONTHS = [("Jul", 1), ("Aug", 32), ("Sep", 63), ("Oct", 93),
                ("Nov", 124), ("Dec", 154), ("Jan", 185), ("Feb", 216),
                ("Mar", 244), ("Apr", 275), ("May", 305), ("Jun", 336)]


def _agri_month_ticks():
    """(positions, labels) for a DOAY x-axis."""
    labs, pos = zip(*[(m, d) for m, d in _AGRI_MONTHS])
    return list(pos), list(labs)


def _fc_summary_by_doay(df: pd.DataFrame,
                        group_cols: list[str],
                        value_cols: list[str]) -> pd.DataFrame:
    """Per (group, doay): mean, p25, p75, n_pixels for each value_col.

    n_pixels = unique (poly_id, x, y) count, but the parquet has been
    deduplicated upstream so just len(group) is fine here.
    """
    rows = []
    for keys, g in df.groupby(group_cols + ["doay"], dropna=False):
        keys = (keys,) if not isinstance(keys, tuple) else keys
        rec = dict(zip(group_cols + ["doay"], keys))
        rec["n"] = int(len(g))
        for v in value_cols:
            s = pd.to_numeric(g[v], errors="coerce").dropna()
            if s.empty:
                rec[f"{v}_mean"] = np.nan
                rec[f"{v}_p25"]  = np.nan
                rec[f"{v}_p75"]  = np.nan
            else:
                rec[f"{v}_mean"] = float(s.mean())
                rec[f"{v}_p25"]  = float(s.quantile(0.25))
                rec[f"{v}_p75"]  = float(s.quantile(0.75))
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(group_cols + ["doay"]).reset_index(drop=True)


def _plot_fc_panel(ax, summ_df: pd.DataFrame, value: str,
                   years: list[int], ylabel: str, title: str | None = None,
                   show_xticks: bool = True, show_legend: bool = True,
                   show_n: bool = False):
    """One panel: mean+IQR vs DOAY, one curve per year."""
    for y in years:
        d = summ_df[summ_df["yr"] == y].sort_values("doay")
        if d.empty:
            continue
        c = YEAR_COLORS.get(y, "#888888")
        ax.fill_between(d["doay"], d[f"{value}_p25"], d[f"{value}_p75"],
                        color=c, alpha=0.18, linewidth=0)
        lbl = f"{y}"
        if show_n:
            lbl = f"{y} (n={int(d['n'].median()):,}/doay)"
        ax.plot(d["doay"], d[f"{value}_mean"], color=c, lw=1.6, label=lbl)
    pos, labs = _agri_month_ticks()
    ax.set_xlim(1, 365)
    if show_xticks:
        ax.set_xticks(pos)
        ax.set_xticklabels(labs, fontsize=8)
    else:
        ax.set_xticks(pos)
        ax.set_xticklabels([])
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    if title:
        ax.set_title(title, fontsize=9)
    if show_legend:
        ax.legend(loc="best", fontsize=8, frameon=False)


def plot_fc_overall(df_fc: pd.DataFrame, cfg: dict, summ: "Summary") -> None:
    """Overall mean+IQR FC curves by DOAY, both years overlaid.

    3 stacked panels: pv (live veg.), npv (residue), fc_total (= (pv+npv)*100).
    """
    years = sorted(df_fc["yr"].unique().tolist())
    value_cols = ["pv", "npv", "fc_total"]
    summ_df = _fc_summary_by_doay(df_fc, ["yr"], value_cols)
    _savecsv(summ_df, f"fc_timeseries_overall_{'_'.join(map(str, years))}.csv", cfg)

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 7.0), sharex=True)
    _plot_fc_panel(axes[0], summ_df, "pv", years, "PV (live veg.)",
                   show_xticks=False, show_legend=True, show_n=True)
    _plot_fc_panel(axes[1], summ_df, "npv", years, "NPV (residue)",
                   show_xticks=False, show_legend=False)
    _plot_fc_panel(axes[2], summ_df, "fc_total", years, "FC total (%)",
                   show_xticks=True, show_legend=False)
    axes[0].set_title(f"FC time series, gap-filled per-field summary over the region "
                      f"({', '.join(map(str, years))}) -- mean +/- IQR by DOAY",
                      fontsize=10)
    axes[2].set_xlabel("Day of agricultural year (1 Jul = DOAY 1)", fontsize=9)
    fig.tight_layout()
    savefig(fig, f"fc_timeseries_overall_{'_'.join(map(str, years))}.png", cfg)

    # Headline numbers in the summary
    summ.add("")
    summ.add("FC time series, REGION (extract_fc_pixels output, overall):")
    for y in years:
        d = summ_df[summ_df["yr"] == y]
        if d.empty:
            continue
        peak_doay = int(d.loc[d["fc_total_mean"].idxmax(), "doay"])
        peak_val  = float(d["fc_total_mean"].max())
        ann_mean  = float(d["fc_total_mean"].mean())
        n_med     = int(d["n"].median())
        summ.add(f"  {y}: mean FC_total over year = {ann_mean:5.1f}%,  "
                 f"peak = {peak_val:5.1f}% at DOAY {peak_doay:3d},  "
                 f"~{n_med:,} pixels/DOAY")
    if len(years) == 2:
        d0 = summ_df[summ_df["yr"] == years[0]].set_index("doay")["fc_total_mean"]
        d1 = summ_df[summ_df["yr"] == years[1]].set_index("doay")["fc_total_mean"]
        common = d0.index.intersection(d1.index)
        if len(common) > 0:
            delta = (d1.loc[common] - d0.loc[common]).mean()
            summ.add(f"  mean(FC_total {years[1]}) - mean(FC_total {years[0]}) "
                     f"= {delta:+.2f} pp (averaged over common DOAYs)")


def plot_fc_per_crop(df_fc: pd.DataFrame, cfg: dict,
                     crop_by_lnf: dict, crop_by_poly: dict | None,
                     summ: "Summary") -> None:
    """Small-multiples grid of FC_total curves, one panel per top-N crop.

    Top-N selected by total number of unique (poly_id, yr) samples;
    crops with < ``cfg['fc_min_samples_per_crop']`` samples in EITHER year
    are dropped.
    """
    years = sorted(df_fc["yr"].unique().tolist())
    top_n = int(cfg.get("fc_top_n_crops", 12))
    min_n = int(cfg.get("fc_min_samples_per_crop", 30))

    # Label rows with a crop string. crop_by_poly is a per-year dict; iterate
    # per year, label, and concat -- consistent with how _label_field_crops
    # treats the per-year poly map.
    parts = []
    for y in years:
        sub = df_fc[df_fc["yr"] == y].copy()
        if sub.empty:
            continue
        sub["poly_id"] = sub["poly_id"].astype("Int64")
        sub["crop"] = _label_field_crops(
            sub, crop_by_lnf,
            (crop_by_poly if crop_by_poly else None),
            cfg, poly_col="poly_id", lnf_col="lnf_code", year=y,
        )
        parts.append(sub)
    if not parts:
        print("  [WARN] No FC rows to label per crop")
        return
    df_fc_lab = pd.concat(parts, ignore_index=True)
    df_fc_lab["crop"] = df_fc_lab["crop"].astype(str).str.strip().replace(
        {"nan": "", "None": "", "<NA>": ""})
    df_fc_lab = df_fc_lab[df_fc_lab["crop"].ne("")]

    # Rank by sample size = unique (poly_id, yr) count across years.
    counts = (df_fc_lab[["crop", "poly_id", "yr"]]
              .drop_duplicates()
              .groupby("crop").size().rename("n_samples")
              .sort_values(ascending=False))
    # Require enough samples per (crop, year) -- otherwise IQR is noise.
    per_yr = (df_fc_lab[["crop", "poly_id", "yr"]]
              .drop_duplicates()
              .groupby(["crop", "yr"]).size().unstack(fill_value=0))
    keep = per_yr[(per_yr >= min_n).all(axis=1)].index
    counts = counts.loc[counts.index.isin(keep)]
    crops = counts.head(top_n).index.tolist()
    if not crops:
        summ.add(f"  [WARN] No crop has >= {min_n} samples in every year "
                 f"-- skipping per-crop FC panels")
        return
    summ.add(f"  FC per-crop: showing top {len(crops)} of {len(counts)} crops "
             f"with >= {min_n} samples per year")

    summ_df = _fc_summary_by_doay(df_fc_lab[df_fc_lab["crop"].isin(crops)],
                                  ["crop", "yr"], ["fc_total"])
    _savecsv(summ_df, f"fc_timeseries_by_crop_{'_'.join(map(str, years))}.csv", cfg)

    # Layout: 3 columns, ceil(N/3) rows
    ncols = 3
    nrows = int(np.ceil(len(crops) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.2 * ncols, 2.6 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    pos, labs = _agri_month_ticks()
    for i, crop in enumerate(crops):
        ax = axes[i // ncols, i % ncols]
        d_crop = summ_df[summ_df["crop"] == crop]
        for y in years:
            d = d_crop[d_crop["yr"] == y].sort_values("doay")
            if d.empty:
                continue
            c = YEAR_COLORS.get(y, "#888888")
            ax.fill_between(d["doay"], d["fc_total_p25"], d["fc_total_p75"],
                            color=c, alpha=0.18, linewidth=0)
            n_y = int(per_yr.loc[crop, y]) if crop in per_yr.index else 0
            ax.plot(d["doay"], d["fc_total_mean"], color=c, lw=1.3,
                    label=f"{y} (n={n_y})")
        # Truncate long crop labels so they don't overflow the panel
        title = crop if len(crop) <= 32 else crop[:30] + ".."
        ax.set_title(title, fontsize=9)
        ax.set_xlim(1, 365)
        ax.set_xticks(pos)
        ax.set_xticklabels(labs, fontsize=7, rotation=0)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
        ax.legend(loc="upper right", fontsize=7, frameon=False)
    # Hide any unused axes
    for j in range(len(crops), nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    # One y-label per row, one x-label on the bottom row
    for r in range(nrows):
        axes[r, 0].set_ylabel("FC total (%)", fontsize=9)
    for c in range(ncols):
        bot = min(nrows - 1, (len(crops) - 1) // ncols)
        axes[bot, c].set_xlabel("DOAY", fontsize=8)
    fig.suptitle(f"FC_total time series by crop, region "
                 f"({', '.join(map(str, years))}), mean +/- IQR",
                 fontsize=11, y=1.005)
    fig.tight_layout()
    savefig(fig, f"fc_timeseries_by_crop_{'_'.join(map(str, years))}.png", cfg)


def fc_timeseries_section(cfg: dict, lnf_bridge: dict,
                          crop_by_lnf: dict, crop_by_poly: dict | None,
                          summ: "Summary") -> None:
    """Section E orchestrator: overall + per-crop FC curves, both years overlaid."""
    df_fc = load_fc_timeseries(cfg, lnf_bridge, cfg["years"])
    if df_fc is None or df_fc.empty:
        return
    plot_fc_overall(df_fc, cfg, summ)
    plot_fc_per_crop(df_fc, cfg, crop_by_lnf, crop_by_poly, summ)


def main():
    ap = argparse.ArgumentParser(description="Compare empirical / ML / previous C-factor products")
    ap.add_argument("--years", nargs="+", type=int, default=None)
    ap.add_argument("--region", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--skip-across", action="store_true", help="skip across-year analysis")
    ap.add_argument("--skip-within", action="store_true", help="skip within-year analysis")
    ap.add_argument("--skip-r", action="store_true", help="skip the R product entirely")
    ap.add_argument("--skip-maps", action="store_true", help="skip the heavy pixel maps")
    ap.add_argument("--skip-tillage", action="store_true", help="skip the tillage-stratified breakdown")
    ap.add_argument("--skip-fc", action="store_true", help="skip the FC time-series breakdown")
    a = ap.parse_args()

    cfg = dict(CONFIG)
    if a.years:
        cfg["years"] = a.years
    if a.region:
        cfg["region_path"] = a.region
    if a.out_dir:
        cfg["out_dir"] = a.out_dir
    _ensure_dir(cfg["out_dir"])
    make_maps = not a.skip_maps

    summ = Summary(cfg)
    summ.add(f"Comparing C-factor products for years {cfg['years']}")
    summ.add(f"  empirical: {cfg['empirical_dir']}")
    summ.add(f"  ml:        {cfg['ml_dir']}")
    summ.add(f"  previous:  {cfg['erosion_results_dir']}")

    # region + bridges
    region = gpd.read_file(_expand(cfg["region_path"]))
    region_2056 = region.to_crs("EPSG:2056").union_all()
    print(f"  Region area: {region_2056.area / 1e6:.1f} km2")
    elev_by_poly, crop_by_lnf, crop_bridge, lnf_bridge, crop_by_poly = \
        _build_bridges(cfg, region_2056, cfg["years"])
    centroid_by_poly = _merge_centroids(lnf_bridge, cfg["years"])
    summ.add(f"  crop_label_method = '{cfg.get('crop_label_method', 'bridge')}'  "
             f"(crop_by_lnf entries: {len(crop_by_lnf):,}; "
             f"crop_by_poly entries: {len(crop_by_poly):,})")

    # A. within product, across years
    if not a.skip_across:
        summ.section("A. WITHIN a product, ACROSS the years (stability)")
        # Pre-compute the per-product (y0,y1) merge so we (a) share IO with the
        # plotting call and (b) can build a JOINT |dC| p98 -> common colour
        # bound on both products' difference maps.
        elev_union = _merge_elev(elev_by_poly, cfg["years"])
        precomp = {}
        for key in ("empirical", "ml"):
            pc = _compute_across_years_pixel(key, cfg)
            if pc is not None:
                precomp[key] = pc
        if cfg.get("dC_vmax_fixed") is not None:
            vmax_shared = float(cfg["dC_vmax_fixed"])
            summ.add(f"  shared dC colour bound: +/-{vmax_shared:.3f} (fixed)")
        else:
            vmax_shared = _joint_vmax([pc[0]["dC"].to_numpy() for pc in precomp.values()])
            summ.add(f"  shared dC colour bound: +/-{vmax_shared:.3f} "
                     f"(joint p98 across products)")
        for key in ("empirical", "ml"):
            across_years_pixel(key, cfg, elev_union, crop_by_lnf,
                               centroid_by_poly, summ, make_maps,
                               precomputed=precomp.get(key),
                               vmax_shared=vmax_shared,
                               crop_by_poly=crop_by_poly)
        r_merged = None
        if not a.skip_r:
            r_merged = across_years_r(cfg, summ)

        # A'. stability of all 3 products on common (betr_ID, crop)
        if not a.skip_r and crop_bridge is not None:
            stability_three_way(cfg, crop_bridge, lnf_bridge, crop_by_lnf,
                                r_merged, summ, crop_by_poly=crop_by_poly)

    # B. within year, across products
    if not a.skip_within:
        summ.section("B. WITHIN a year, ACROSS the products (divergence)")
        for year in cfg["years"]:
            m_eml = empirical_vs_ml_pixel(cfg, year, elev_by_poly.get(year, {}),
                                          crop_by_lnf, centroid_by_poly, summ, make_maps,
                                          crop_by_poly=crop_by_poly)
            disagreement_analysis(m_eml, cfg, year, crop_by_lnf,
                                  centroid_by_poly, summ, make_maps,
                                  crop_by_poly=crop_by_poly)
            if not a.skip_r and crop_bridge is not None:
                three_way_per_crop(cfg, year, crop_bridge, lnf_bridge.get(year),
                                   crop_by_lnf, summ, crop_by_poly=crop_by_poly)

    # D. tillage-stratified breakdown (needs R for the tillage map)
    if not a.skip_tillage and not a.skip_r:
        summ.section("D. TILLAGE-STRATIFIED breakdown (Pflug vs Mulch)")
        tillage_map = build_tillage_map(cfg, cfg["years"])
        if not a.skip_within:
            for year in cfg["years"]:
                within_year_by_tillage(cfg, year, tillage_map,
                                       lnf_bridge.get(year),
                                       crop_by_lnf, crop_by_poly, summ)
        if not a.skip_across and len(cfg["years"]) >= 2:
            ys = sorted(cfg["years"])
            across_year_by_tillage_transition(cfg, ys[0], ys[-1], tillage_map,
                                              lnf_bridge, crop_by_lnf,
                                              crop_by_poly, summ)

    # E. FC time-series breakdown (calibration sample, region-restricted)
    if not a.skip_fc:
        summ.section("E. FC TIME SERIES (calibration sample, year-to-year)")
        fc_timeseries_section(cfg, lnf_bridge, crop_by_lnf, crop_by_poly, summ)

    # C. synthesis
    synthesis(cfg, summ)
    summ.flush()
    print(f"\nDone. See {cfg['out_dir']}/")


def _merge_elev(elev_by_poly: dict, years) -> dict:
    """Union of per-year poly_id->elevation (poly_id can carry over years)."""
    out = {}
    for y in years:
        out.update(elev_by_poly.get(y, {}))
    return out


if __name__ == "__main__":
    main()