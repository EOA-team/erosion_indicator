"""
Area analysis for the Seeland district — 2021 vs 2022.

Produces figures and summary statistics characterising the study area:
  1. Climate (MeteoSwiss: rainfall, temperature, sunshine)
  2. Crop distribution (from LNF gpkg)
  3. Tillage distribution (from REB / Schonende Bodenbearbeitung)
  4. Reference C-factor landscape (C_Faktoren.csv × crop/tillage/region mix)
  5. Rainfall erosivity (EI)
  6. Sentinel-2 data quality (from pixel-level outputs)

Usage:
    python area_analysis.py
    python area_analysis.py --skip-meteo   # skip slow MeteoSwiss loading
    python area_analysis.py --skip-s2      # skip pixel-output quality section
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rio_mask
import xarray as xr

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration — EDIT PATHS HERE
# ---------------------------------------------------------------------------

CONFIG = {
    # Region boundary (passed to compute_cfactor_pixels.py --region)
    "region_path":      "../cfactor/seeland.gpkg",  # adjust if different

    # LNF land-use polygons (one gpkg per year — used for elevation + ref C)
    "lnf_dir":          "~/mnt/eo-nas1/data/landuse/raw",

    # Crop1990 raster predictions (wall-to-wall crop map per year)
    "crop_raster_pattern": "~/mnt/eo-nas1/eoa-share/projects/020_crop1990/"
                           "Crop1990/storage/CDL_Sentinel/predictions/"
                           "{year}_predictions.tif",
    "crop_colors_path":    "~/mnt/eo-nas1/eoa-share/projects/020_crop1990/"
                           "Crop1990/storage/CDL_Sentinel/predictions/"
                           "crop1990_colors.txt.txt",

    # LNF classification spreadsheet (Crop_Label bridges crop1990 ↔ lnf_code)
    "lnf_labels_path":  "~/mnt/eo-nas1/eoa-share/projects/020_crop1990/"
                        "data/LNF_code_classification_20260217.xlsx",

    # Crop name mapping (kulturcode <-> Kultur_nutzung)
    "kulturmapping_csv": "~/mnt/Data-Labo-RE/27_Natural_Resources-RE/"
                         "321.4_WAUM_protected/Daten/Core_Snapshot/"
                         "Agrarbericht_2025/tbl_kulturmapping.csv",

    # C-factor reference table
    "c_factor_csv":     "~/mnt/Data-Labo-RE/27_Natural_Resources-RE/"
                        "321.4_WAUM_protected/Daten/Erosionsrisiko/C_Faktoren.csv",

    # Conservation tillage (canonical CSV with real betr_ID matching LNF
    # betriebsnummer). Keyed on betr_ID + JAHR + CODE_KULTUR.
    "tillage_csv":      "~/mnt/Data-Labo-RE/27_Natural_Resources-RE/"
                        "321.4_WAUM_protected/Daten/Erosionsrisiko/"
                        "schonende_bodenbearbeitung.csv",

    # Nutzungsdaten (field-level: Flaechen_ID, betr_ID, swissALTI3D, Kultur_nutzung)
    "nutzung_csv":      "~/mnt/Data-Labo-RE/27_Natural_Resources-RE/"
                        "321.4_WAUM_protected/Daten/Core_Snapshot/"
                        "Agrarbericht_2025/tbl_nutzungsdaten.csv",

    # Swiss canton boundaries (swissBOUNDARIES3D or similar — set None to use naturalearth)
    "canton_boundaries_path": None,

    # REB table (farm-level tillage for all years)
    "reb_csv":          "~/mnt/Data-Labo-RE/27_Natural_Resources-RE/"
                        "321.4_WAUM_protected/Daten/Core_Snapshot/"
                        "Agrarbericht_2025/tbl_ressourceneffizienzbeitrag.csv",

    # MeteoSwiss gridded data (zarr, same tiling as S2)
    # Pattern: {meteo_base}/{var}/MeteoSwiss_{var}D_{minx}_{maxy}_{year}0101_{year}1231.zarr
    "meteo_base":       "~/mnt/eo-nas1/data/meteo",
    "meteo_vars":       ["RhiresD", "TabsD", "SrelD"],  # rainfall, T_mean, sunshine

    # Climatological EI (erosivity index)
    "ei_path":          "../erosivity_index/predictions/"
                        "grid_EI_daily_avg_pred_20260424_nn3.parquet",

    # S2 tile grid (to discover which tiles overlap the region)
    "s2_grid_path":     "~/mnt/eo-nas1/eoa-share/projects/"
                        "012_EO_dataInfrastructure/Project layers/"
                        "gridface_s2tiles_CH.shp",
    "s2_dir":           "~/mnt/eo-nas1/data/satellite/sentinel2/raw/CH",

    # Pixel-level C-factor outputs from compute_cfactor_pixels.py
    "pixel_output_dir": "../cfactor/output/cfactor_pixels",

    # Altitude threshold for Tal/Berg classification
    "grenze_tal_berg":  600,

    # Years to analyse
    "years":            [2021, 2022],

    # Output
    "out_dir":          "figures",
}

TILE_SIZE_M = 1280  # same as in compute_cfactor_pixels.py


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _expand(p: str) -> str:
    return os.path.expanduser(p)


def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def load_region(cfg: dict) -> gpd.GeoDataFrame:
    """Load the region boundary and return in EPSG:2056 + EPSG:32632."""
    region = gpd.read_file(_expand(cfg["region_path"]))
    return region


def region_bounds_32632(region: gpd.GeoDataFrame) -> tuple:
    return tuple(region.to_crs("EPSG:32632").total_bounds)


def discover_meteo_tiles(cfg: dict, region: gpd.GeoDataFrame) -> list[tuple[int, int]]:
    """Find (minx, maxy) tile keys overlapping the region, using the S2 grid."""
    grid = gpd.read_file(_expand(cfg["s2_grid_path"]))
    region_crs = region.to_crs(grid.crs)
    hits = grid[grid.intersects(region_crs.union_all())]
    # The grid has 'left' and 'top' columns matching the tile naming
    return list(zip(hits["left"].astype(int), hits["top"].astype(int)))


def savefig(fig, name: str, cfg: dict, dpi: int = 180):
    _ensure_dir(cfg["out_dir"])
    path = os.path.join(cfg["out_dir"], name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Crop1990 raster helpers
# ---------------------------------------------------------------------------

def load_crop1990_labels(cfg: dict) -> dict[int, dict]:
    """Parse crop1990_colors.txt.txt → {code: {name, rgb}}.

    Format: ``code R G B label_with_spaces``  (space-separated, 1-indexed)
    """
    path = _expand(cfg["crop_colors_path"])
    labels = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                # code R G B (no label)
                code, r, g, b = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                name = f"Class {code}"
            else:
                code = int(parts[0])
                r, g, b = int(parts[1]), int(parts[2]), int(parts[3])
                name = " ".join(parts[4:])
            labels[code] = {"name": name, "rgb": (r, g, b)}
    return labels


def load_crop_raster(year: int, region: gpd.GeoDataFrame,
                     cfg: dict) -> tuple[np.ndarray, dict, float]:
    """Load the crop1990 prediction raster clipped to the region.

    Returns (data_2d, transform_meta, pixel_area_m2).
    ``data_2d`` contains crop1990 class codes; 0 / nodata = outside region.
    """
    path = _expand(cfg["crop_raster_pattern"].format(year=year))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Crop raster not found: {path}")

    with rasterio.open(path) as src:
        region_reproj = region.to_crs(src.crs)
        geoms = region_reproj.geometry.values
        out_image, out_transform = rio_mask(src, geoms, crop=True,
                                            nodata=0, indexes=1)
        pixel_area = abs(src.res[0] * src.res[1])  # m²
        meta = {
            "crs": src.crs,
            "transform": out_transform,
            "height": out_image.shape[0],
            "width": out_image.shape[1],
        }
    return out_image, meta, pixel_area


def crop1990_to_lnf_bridge(cfg: dict) -> pd.DataFrame:
    """Build a bridge: crop1990 label name → list of lnf_codes.

    Uses `Crop_Label` in the LNF classification Excel to link the two systems.
    Returns DataFrame with columns [crop_label, lnf_code, Crop_Label_lv3].
    """
    path = _expand(cfg["lnf_labels_path"])
    df = pd.read_excel(path, sheet_name="label_sheet")
    df = df[["LNF_code", "Crop_Label", "Crop_Label_lv3"]].rename(
        columns={"LNF_code": "lnf_code", "Crop_Label": "crop_label"})
    df = df.dropna(subset=["crop_label"])
    return df


# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------

YEAR_COLORS = {2021: "#4477AA", 2022: "#CC6677"}
YEAR_LABELS = {2021: "2021", 2022: "2022"}


def _year_color(yr: int) -> str:
    return YEAR_COLORS.get(yr, "#999999")


# ===========================================================================
# Section 0 — Study area overview
# ===========================================================================

def _load_swiss_outline() -> gpd.GeoDataFrame | None:
    """Try to load a Switzerland country outline for the context map."""
    # Option 1: naturalearth via geopandas (bundled low-res)
    try:
        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        ch = world[world["iso_a3"] == "CHE"]
        if not ch.empty:
            return ch
    except Exception:
        pass
    # Option 2: naturalearth_cities or similar
    try:
        import cartopy.feature as cfeature
        import cartopy.io.shapereader as shpreader
        shp = shpreader.natural_earth(resolution="10m", category="cultural",
                                      name="admin_0_countries")
        reader = shpreader.Reader(shp)
        for rec in reader.records():
            if rec.attributes.get("ISO_A3") == "CHE":
                return gpd.GeoDataFrame(geometry=[rec.geometry], crs="EPSG:4326")
    except Exception:
        pass
    return None


def _load_canton_boundaries(cfg: dict) -> gpd.GeoDataFrame | None:
    """Load canton boundaries if a path is configured."""
    path = cfg.get("canton_boundaries_path")
    if path and os.path.exists(_expand(path)):
        return gpd.read_file(_expand(path))
    return None


def _plot_switzerland_context(ax, region: gpd.GeoDataFrame, cfg: dict):
    """Plot the region within Switzerland, with a web-tile basemap if available."""
    region_3857 = region.to_crs("EPSG:3857")
    region_2056 = region.to_crs("EPSG:2056")
    centroid = region_2056.union_all().centroid

    # Try a contextily basemap (needs internet + the package)
    basemap_ok = False
    try:
        import contextily as cx
        # Frame Switzerland: pad generously around the region centroid
        # CH extent in EPSG:3857 (approx)
        ch_xmin, ch_ymin, ch_xmax, ch_ymax = 660_000, 5_730_000, 1_180_000, 6_080_000
        ax.set_xlim(ch_xmin, ch_xmax)
        ax.set_ylim(ch_ymin, ch_ymax)
        region_3857.plot(ax=ax, color="#CC6677", alpha=0.7,
                         edgecolor="#8B0000", linewidth=1.5, zorder=5)
        cx.add_basemap(ax, crs="EPSG:3857",
                       source=cx.providers.CartoDB.Positron,
                       attribution_size=5)
        basemap_ok = True
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
    except Exception as e:
        print(f"  [INFO] contextily basemap unavailable ({type(e).__name__}); "
              f"using vector outline")

    if not basemap_ok:
        # Fallback: plain Swiss outline in EPSG:2056
        ch = _load_swiss_outline()
        cantons = _load_canton_boundaries(cfg)
        if ch is not None:
            ch_2056 = ch.to_crs("EPSG:2056")
            ch_2056.plot(ax=ax, color="#EEEEEE", edgecolor="#555555",
                         linewidth=0.8, zorder=1)
            if cantons is not None:
                cantons.to_crs("EPSG:2056").boundary.plot(
                    ax=ax, color="#AAAAAA", linewidth=0.3, zorder=2)
            region_2056.plot(ax=ax, color="#CC6677", alpha=0.7,
                             edgecolor="#8B0000", linewidth=1.5, zorder=5)
            b = ch_2056.total_bounds
            pad = 20_000
            ax.set_xlim(b[0] - pad, b[2] + pad)
            ax.set_ylim(b[1] - pad, b[3] + pad)
        else:
            region_2056.plot(ax=ax, color="#CC6677", alpha=0.7,
                             edgecolor="#8B0000")
            ax.set_xlim(2_480_000, 2_840_000)
            ax.set_ylim(1_070_000, 1_300_000)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("E (m)", fontsize=8)
        ax.set_ylabel("N (m)", fontsize=8)

    ax.set_aspect("equal")
    ax.set_title("Seeland district in Switzerland", fontsize=11)


def analyse_study_area(cfg: dict, region: gpd.GeoDataFrame):
    """Produce the study-area overview: context map, crop raster, elevation."""
    print("\n=== Section 0: Study area overview ===")
    region_2056 = region.to_crs("EPSG:2056")
    area_km2 = region_2056.union_all().area / 1e6
    print(f"  Region area: {area_km2:.1f} km²")

    # --- Load crop raster for the first year ---
    yr0 = cfg["years"][0]
    crop_labels = load_crop1990_labels(cfg)
    try:
        crop_data, crop_meta, pix_area = load_crop_raster(yr0, region, cfg)
        print(f"  Crop raster ({yr0}): {crop_data.shape}, "
              f"pixel = {pix_area:.0f} m², "
              f"{np.count_nonzero(crop_data)} classified pixels")
    except FileNotFoundError as e:
        print(f"  [WARN] {e}")
        crop_data = None

    # --- Area stats from the raster ---
    total_ag_ha = 0
    lv3_summary = {}
    if crop_data is not None:
        bridge = crop1990_to_lnf_bridge(cfg)
        # crop1990 code → Crop_Label_lv3 (take most common lv3 per crop label)
        label_to_lv3 = (bridge.drop_duplicates("crop_label")
                        .set_index("crop_label")["Crop_Label_lv3"].to_dict())
        code_to_name = {c: v["name"] for c, v in crop_labels.items()}
        code_to_lv3 = {c: label_to_lv3.get(v["name"], "Other")
                       for c, v in crop_labels.items()}

        codes, counts = np.unique(crop_data[crop_data > 0], return_counts=True)
        for code, cnt in zip(codes, counts):
            ha = cnt * pix_area / 1e4
            lv3 = code_to_lv3.get(int(code), "Other")
            lv3_summary[lv3] = lv3_summary.get(lv3, 0) + ha
        total_ag_ha = sum(lv3_summary.values())
        n_crop_types = len(codes)
        arable_ha = lv3_summary.get("Arable Land", 0)
        grassland_ha = lv3_summary.get("Grassland", 0)
        print(f"  Classified area: {total_ag_ha:.0f} ha, "
              f"{n_crop_types} crop types")
        for cat in sorted(lv3_summary, key=lv3_summary.get, reverse=True):
            print(f"    {cat}: {lv3_summary[cat]:.0f} ha "
                  f"({100 * lv3_summary[cat] / total_ag_ha:.1f}%)")

    # --- Elevation stats from tbl_nutzungsdaten + LNF gpkg ---
    elev_stats = _elevation_stats(cfg, region)

    # --- Figure: 3-panel overview ---
    #   (1) Switzerland context with basemap
    #   (2) per-crop map
    #   (3) lv3 (arable / grassland / permanent / ...) classification
    fig = plt.figure(figsize=(16, 5.5))
    fig = plt.figure(figsize=(16, 6.5))
    ax_ch = fig.add_axes([0.02, 0.20, 0.27, 0.72])
    ax_crop = fig.add_axes([0.34, 0.20, 0.30, 0.72])
    ax_lv3 = fig.add_axes([0.68, 0.20, 0.30, 0.72])

    lv3_rgb = {
        "Arable Land":                (232, 197, 71),
        "Grassland":                  (102, 187, 106),
        "Permanent Crops":            (171, 71, 188),
        "Ecological infrastructure":  (38, 166, 154),
        "Other":                      (189, 189, 189),
    }

    # ---- Panel 1: Switzerland context map with basemap ----
    _plot_switzerland_context(ax_ch, region, cfg)

    # ---- Panel 2: per-crop map ----
    region_2056.boundary.plot(ax=ax_crop, color="#333333", linewidth=1.2,
                              linestyle="--", zorder=5)
    if crop_data is not None:
        tf = crop_meta["transform"]
        h, w = crop_data.shape
        extent = [tf.c, tf.c + tf.a * w, tf.f + tf.e * h, tf.f]

        # per-crop RGB
        rgb_crop = np.ones((h, w, 3), dtype=np.uint8) * 255
        for code, info in crop_labels.items():
            mask = crop_data == code
            if mask.any():
                rgb_crop[mask] = info["rgb"]
        ax_crop.imshow(rgb_crop, extent=extent, origin="upper",
                       interpolation="nearest")

        # Compact legend: top crops by area, restricted to Arable Land /
        # Grassland classes (skip forest, built-up, water, etc.)
        from matplotlib.patches import Patch
        crop_lv3 = {"Arable Land", "Grassland"}
        codes, counts = np.unique(crop_data[crop_data > 0], return_counts=True)
        order = np.argsort(counts)[::-1]
        top_handles = []
        for i in order:
            code = int(codes[i])
            if code not in crop_labels:
                continue
            if code_to_lv3.get(code) not in crop_lv3:
                continue
            top_handles.append(
                Patch(facecolor=np.array(crop_labels[code]["rgb"]) / 255,
                      label=code_to_name.get(code, str(code))))
            if len(top_handles) >= 12:
                break
        ax_crop.legend(handles=top_handles, loc="upper center",
                       bbox_to_anchor=(0.5, -0.12), fontsize=6,
                       framealpha=0.9, title="Top crops", title_fontsize=7,
                       ncol=3, columnspacing=1.0, handletextpad=0.4)
    ax_crop.set_aspect("equal")
    ax_crop.set_title(f"Crop map ({yr0})", fontsize=11)

    # ---- Panel 3: lv3 classification ----
    region_2056.boundary.plot(ax=ax_lv3, color="#333333", linewidth=1.2,
                              linestyle="--", zorder=5)
    if crop_data is not None:
        rgb_lv3 = np.ones((h, w, 3), dtype=np.uint8) * 255
        for code in crop_labels:
            mask = crop_data == code
            if not mask.any():
                continue
            lv3 = code_to_lv3.get(code, "Other")
            rgb_lv3[mask] = lv3_rgb.get(lv3, lv3_rgb["Other"])
        ax_lv3.imshow(rgb_lv3, extent=extent, origin="upper",
                      interpolation="nearest")

        from matplotlib.patches import Patch
        lv3_handles = [
            Patch(facecolor=np.array(c) / 255,
                  label=f"{cat} ({lv3_summary.get(cat, 0):.0f} ha)")
            for cat, c in lv3_rgb.items()
            if lv3_summary.get(cat, 0) > 0
        ]
        ax_lv3.legend(handles=lv3_handles, loc="lower right", fontsize=7,
                      framealpha=0.9)
    ax_lv3.set_aspect("equal")
    ax_lv3.set_title(f"Land-use classification ({yr0})", fontsize=11)

    # Shared axis formatting for the two zoom panels
    for ax in (ax_crop, ax_lv3):
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{x / 1e3:.0f}"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{x / 1e3:.0f}"))
    # crop panel x-label omitted (legend sits below it); lv3 keeps it
    ax_lv3.set_xlabel("E (km)", fontsize=8)
    ax_crop.set_ylabel("N (km)", fontsize=8)

    # --- Text box with key stats (on the lv3 panel) ---
    stats_text = f"District area: {area_km2:.0f} km²"
    if total_ag_ha > 0:
        stats_text += f"\nClassified agric. land: {total_ag_ha:.0f} ha"
        stats_text += f"\n  Arable: {arable_ha:.0f} ha"
        stats_text += f"\n  Grassland: {grassland_ha:.0f} ha"
        stats_text += f"\n  Crop types: {n_crop_types}"
    if elev_stats is not None:
        stats_text += (f"\nElevation: {elev_stats['min']:.0f}–"
                       f"{elev_stats['max']:.0f} m a.s.l."
                       f" (median {elev_stats['median']:.0f} m)")
        stats_text += (f"\nTal / Berg: {elev_stats['frac_tal']:.0f}% / "
                       f"{elev_stats['frac_berg']:.0f}%")
    ax_lv3.text(0.02, 0.98, stats_text, transform=ax_lv3.transAxes,
                fontsize=8, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          alpha=0.85, edgecolor="#CCCCCC"))

    savefig(fig, "study_area_overview.png", cfg)

    # --- Elevation histogram ---
    if elev_stats is not None and "values" in elev_stats:
        fig_e, ax_e = plt.subplots(figsize=(7, 3.5))
        ax_e.hist(elev_stats["values"], bins=40, color="#7986CB",
                  edgecolor="white", alpha=0.85)
        ax_e.axvline(cfg["grenze_tal_berg"], color="red", ls="--", lw=1.2,
                     label=f"Tal/Berg threshold ({cfg['grenze_tal_berg']} m)")
        ax_e.set_xlabel("Elevation (m a.s.l.)")
        ax_e.set_ylabel("Number of fields")
        ax_e.set_title("Elevation distribution of agricultural fields — Seeland")
        ax_e.legend(fontsize=9)
        ax_e.grid(axis="y", alpha=0.3)
        savefig(fig_e, "elevation_distribution.png", cfg)

    return {"area_km2": area_km2, "elev_stats": elev_stats}


def _elevation_stats(cfg: dict, region: gpd.GeoDataFrame) -> dict | None:
    """Get elevation stats for fields in the region.

    Loads the LNF gpkg to get Flaechen_IDs, then joins with tbl_nutzungsdaten.
    """
    nutzung_path = _expand(cfg.get("nutzung_csv", ""))
    if not os.path.exists(nutzung_path):
        print("  [WARN] tbl_nutzungsdaten not found — skipping elevation stats")
        return None

    yr = cfg["years"][0]

    # Load LNF gpkg to get field IDs within the region
    lnf_path = os.path.join(_expand(cfg["lnf_dir"]), f"lnf{yr}.gpkg")
    if not os.path.exists(lnf_path):
        print(f"  [WARN] {lnf_path} not found — skipping elevation stats")
        return None

    print(f"  Loading LNF gpkg for elevation join ...")
    lnf = gpd.read_file(lnf_path, bbox=tuple(region.to_crs("EPSG:2056").total_bounds))
    region_2056 = region.to_crs(lnf.crs).union_all()
    lnf = lnf[lnf.intersects(region_2056)]

    # Identify the field ID column (uuid for ≤2022, identifikator_be for ≥2023)
    if "uuid" in lnf.columns:
        field_ids = set(lnf["uuid"].astype(str).unique())
    elif "id" in lnf.columns:
        field_ids = set(lnf["id"].astype(str).unique())
    else:
        print("  [WARN] No uuid/id column in LNF — skipping elevation stats")
        return None

    print(f"  Loading elevation for {len(field_ids):,} fields from nutzungsdaten ...")
    chunks = pd.read_csv(nutzung_path, encoding="latin1", sep=";",
                         usecols=["Jahr", "Flaechen_ID", "swissALTI3D"],
                         chunksize=200_000)
    rows = []
    for chunk in chunks:
        chunk = chunk[chunk["Jahr"] == yr]
        chunk["Flaechen_ID"] = chunk["Flaechen_ID"].astype(str)
        chunk = chunk[chunk["Flaechen_ID"].isin(field_ids)]
        if not chunk.empty:
            rows.append(chunk)
    if not rows:
        print("  [WARN] No elevation data matched — "
              "Flaechen_IDs may differ between LNF gpkg and nutzungsdaten")
        return None

    df = pd.concat(rows).drop_duplicates("Flaechen_ID").dropna(subset=["swissALTI3D"])
    vals = df["swissALTI3D"].values
    cutoff = cfg["grenze_tal_berg"]
    stats = {
        "min": vals.min(),
        "max": vals.max(),
        "mean": vals.mean(),
        "median": np.median(vals),
        "std": vals.std(),
        "frac_tal": 100 * (vals <= cutoff).sum() / len(vals),
        "frac_berg": 100 * (vals > cutoff).sum() / len(vals),
        "n_matched": len(vals),
        "values": vals,
    }
    print(f"  Elevation: {stats['min']:.0f}–{stats['max']:.0f} m, "
          f"median {stats['median']:.0f} m, "
          f"{stats['frac_tal']:.0f}% Tal / {stats['frac_berg']:.0f}% Berg "
          f"({stats['n_matched']} fields matched)")
    return stats

def _load_meteo_var(var: str, year: int, tiles: list[tuple[int, int]],
                    cfg: dict) -> xr.Dataset:
    """Load and merge all tiles for one MeteoSwiss variable + year."""
    base = _expand(cfg["meteo_base"])
    # var names in config are like "RhiresD" — the folder name drops the D
    # Pattern: MeteoSwiss_{var}_{minx}_{maxy}_{year}0101_{year}1231.zarr
    # Folder:  {meteo_base}/{var_folder}/
    # The var in filename includes the D suffix: e.g. RhiresD
    var_folder = var.replace("D", "")  # Rhires, Tabs, Srel
    datasets = []
    for minx, maxy in tiles:
        fname = f"MeteoSwiss_{var}_{minx}_{maxy}_{year}0101_{year}1231.zarr"
        path = os.path.join(base, var_folder, fname)
        if not os.path.exists(path):
            # try with the folder name = var (with D)
            path = os.path.join(base, var, fname)
        if not os.path.exists(path):
            continue
        try:
            ds = xr.open_zarr(path)
            datasets.append(ds)
        except Exception as e:
            print(f"    [WARN] Cannot open {path}: {e}")
    if not datasets:
        return None
    return xr.concat(datasets, dim="time") if len(datasets) > 1 else datasets[0]


def _meteo_monthly_agg(var: str, year: int, tiles: list, cfg: dict,
                       agg: str = "sum") -> pd.Series | None:
    """Return monthly aggregated values (mean over spatial tiles) for one var+year."""
    ds = _load_meteo_var(var, year, tiles, cfg)
    if ds is None:
        return None
    # Take spatial mean first, then monthly aggregate
    data_vars = list(ds.data_vars)
    if len(data_vars) == 1:
        da = ds[data_vars[0]]
    else:
        # pick the one that looks like the main variable
        candidates = [v for v in data_vars if var.replace("D", "").lower() in v.lower()]
        da = ds[candidates[0]] if candidates else ds[data_vars[0]]

    spatial_mean = da.mean(dim=[d for d in da.dims if d != "time"])
    df = spatial_mean.to_dataframe().reset_index()
    df["month"] = pd.to_datetime(df["time"]).dt.month
    col = da.name
    if agg == "sum":
        monthly = df.groupby("month")[col].sum()
    else:
        monthly = df.groupby("month")[col].mean()
    return monthly


def analyse_climate(cfg: dict, region: gpd.GeoDataFrame):
    """Load MeteoSwiss grids, compute monthly stats, produce comparison plots."""
    print("\n=== Section 1: Climate analysis ===")
    tiles = discover_meteo_tiles(cfg, region)
    print(f"  {len(tiles)} tiles overlap the region")

    meteo_specs = {
        "RhiresD": {"agg": "sum",  "label": "Precipitation (mm)", "title": "Monthly precipitation"},
        "TabsD":   {"agg": "mean", "label": "Temperature (°C)",   "title": "Monthly mean temperature"},
        "SrelD":   {"agg": "mean", "label": "Sunshine rel. (%)",  "title": "Monthly relative sunshine duration"},
    }

    results = {}
    for var in cfg["meteo_vars"]:
        spec = meteo_specs.get(var, {"agg": "mean", "label": var, "title": var})
        fig, ax = plt.subplots(figsize=(8, 4))
        for yr in cfg["years"]:
            print(f"  Loading {var} {yr} ...")
            monthly = _meteo_monthly_agg(var, yr, tiles, cfg, agg=spec["agg"])
            if monthly is None:
                print(f"    [WARN] No data for {var} {yr}")
                continue
            results[(var, yr)] = monthly
            if spec["agg"] == "sum":
                ax.bar(monthly.index + (cfg["years"].index(yr) - 0.5) * 0.35,
                       monthly.values, width=0.35,
                       color=_year_color(yr), label=YEAR_LABELS[yr])
            else:
                ax.plot(monthly.index, monthly.values, "o-",
                        color=_year_color(yr), label=YEAR_LABELS[yr])

        ax.set_xlabel("Month")
        ax.set_ylabel(spec["label"])
        ax.set_title(spec["title"])
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(["J", "F", "M", "A", "M", "J",
                            "J", "A", "S", "O", "N", "D"])
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        savefig(fig, f"climate_{var}.png", cfg)

    # Annual totals / means summary
    summary_lines = []
    for var in cfg["meteo_vars"]:
        spec = meteo_specs.get(var, {"agg": "mean", "label": var})
        for yr in cfg["years"]:
            key = (var, yr)
            if key in results:
                val = results[key].sum() if spec["agg"] == "sum" else results[key].mean()
                summary_lines.append(f"  {var} {yr}: {'total' if spec['agg'] == 'sum' else 'mean'} = {val:.1f}")
    if summary_lines:
        print("  Annual summary:")
        for line in summary_lines:
            print(line)
    return results


# ===========================================================================
# Section 2 — Crop distribution (crop1990 raster)
# ===========================================================================

def _crop_area_from_raster(year: int, region: gpd.GeoDataFrame,
                           cfg: dict) -> pd.DataFrame | None:
    """Count pixels per crop class → area (ha). Returns DataFrame."""
    try:
        data, meta, pix_area = load_crop_raster(year, region, cfg)
    except FileNotFoundError as e:
        print(f"  [WARN] {e}")
        return None
    codes, counts = np.unique(data[data > 0], return_counts=True)
    df = pd.DataFrame({"crop_code": codes.astype(int),
                        "n_pixels": counts.astype(int)})
    df["area_ha"] = df["n_pixels"] * pix_area / 1e4
    return df


def analyse_crops(cfg: dict, region: gpd.GeoDataFrame):
    """Crop distribution comparison for 2021 vs 2022 using crop1990 rasters."""
    print("\n=== Section 2: Crop distribution ===")
    crop_labels = load_crop1990_labels(cfg)
    code_to_name = {c: v["name"] for c, v in crop_labels.items()}

    # Build lv3 lookup via the bridge
    bridge = crop1990_to_lnf_bridge(cfg)
    label_to_lv3 = (bridge.drop_duplicates("crop_label")
                    .set_index("crop_label")["Crop_Label_lv3"].to_dict())

    crop_stats = {}
    for yr in cfg["years"]:
        print(f"  Loading crop raster {yr} ...")
        df = _crop_area_from_raster(yr, region, cfg)
        if df is None:
            continue
        df["crop_name"] = df["crop_code"].map(code_to_name).fillna("Unknown")
        df["lv3"] = df["crop_name"].map(label_to_lv3).fillna("Other")
        df = df.sort_values("area_ha", ascending=False)
        crop_stats[yr] = df
        print(f"    {df['area_ha'].sum():.0f} ha classified, "
              f"{len(df)} crop types")

    if len(crop_stats) < 2:
        print("  [WARN] Need both years for comparison")
        return crop_stats

    # --- Top-N crops bar chart (Arable Land / Grassland only) ---
    n_top = 15
    crop_lv3 = {"Arable Land", "Grassland"}
    top_codes = set()
    for yr in cfg["years"]:
        cs_crop = crop_stats[yr][crop_stats[yr]["lv3"].isin(crop_lv3)]
        top_codes.update(cs_crop.head(n_top)["crop_code"].tolist())

    rows = []
    for code in top_codes:
        for yr in cfg["years"]:
            df_yr = crop_stats[yr]
            match = df_yr[df_yr["crop_code"] == code]
            area = match["area_ha"].values[0] if len(match) else 0
            name = code_to_name.get(code, str(code))
            rows.append({"crop_code": code, "year": yr, "area_ha": area,
                         "crop": name})
    merged = pd.DataFrame(rows)
    order = (merged.groupby("crop")["area_ha"].sum()
             .sort_values(ascending=True).index.tolist())

    fig, ax = plt.subplots(figsize=(8, max(5, len(order) * 0.35)))
    y_pos = np.arange(len(order))
    bar_h = 0.35
    for i, yr in enumerate(cfg["years"]):
        sub = merged[merged["year"] == yr].set_index("crop").reindex(order)
        ax.barh(y_pos + (i - 0.5) * bar_h, sub["area_ha"].fillna(0),
                height=bar_h, color=_year_color(yr), label=YEAR_LABELS[yr])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("Area (ha)")
    ax.set_title(f"Top {n_top} crops by area — Seeland district")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    savefig(fig, "crop_distribution.png", cfg)

    # --- Land-use category (lv3) summary ---
    for yr in cfg["years"]:
        lv3_area = (crop_stats[yr].groupby("lv3")["area_ha"].sum()
                    .sort_values(ascending=False))
        total = lv3_area.sum()
        print(f"  {yr} land-use categories (ha):")
        for cat, area in lv3_area.items():
            print(f"    {cat}: {area:.0f} ha ({100 * area / total:.1f}%)")

    # --- Year-over-year change ---
    df_2021 = (crop_stats[2021].set_index("crop_code")
               [["area_ha", "crop_name"]].rename(columns={"area_ha": "ha_2021"}))
    df_2022 = (crop_stats[2022].set_index("crop_code")
               [["area_ha"]].rename(columns={"area_ha": "ha_2022"}))
    change = df_2021.join(df_2022, how="outer").fillna(0)
    change["delta_ha"] = change["ha_2022"] - change["ha_2021"]
    change["delta_pct"] = np.where(
        change["ha_2021"] > 0,
        100 * change["delta_ha"] / change["ha_2021"], np.nan)
    change = change.sort_values("delta_ha")
    change_path = os.path.join(cfg["out_dir"], "crop_change_2021_2022.csv")
    _ensure_dir(cfg["out_dir"])
    change.to_csv(change_path)
    print(f"  Saved {change_path}")

    big = change[(change["ha_2021"] + change["ha_2022"]) > 10]
    big_sorted = big.reindex(big["delta_ha"].abs().sort_values(ascending=False).index)
    print("  Largest area changes (>10 ha in either year):")
    for code, row in big_sorted.head(10).iterrows():
        name = row["crop_name"] if pd.notna(row["crop_name"]) else str(code)
        print(f"    {name}: {row['ha_2021']:.0f} → {row['ha_2022']:.0f} ha "
              f"({row['delta_ha']:+.0f} ha)")

    return crop_stats


# ===========================================================================
# Section 3 — Tillage distribution
# ===========================================================================

# Conservation-tillage VERFAHREN → standard class. Anything not in this CSV
# is inferred to be conventional plough (Pflug).
VERFAHREN_TO_TILLAGE = {
    "Mulchsaat":     "Mulch",
    "Direktsaat":    "Direkt",
    "Streifensaat":  "Direkt",   # note: value is "Streifensaat" (not -fräs-)
}


def _load_tillage_lookup(cfg: dict) -> pd.DataFrame | None:
    """Load the conservation-tillage CSV → one row per (betr_ID, year, crop).

    Returns DataFrame [betr_ID, year, lnf_code, tillage] where tillage is
    'Mulch' or 'Direkt'. Farm-crops listing more than one distinct tillage
    class are dropped (ambiguous).
    """
    path = _expand(cfg["tillage_csv"])
    if not os.path.exists(path):
        print(f"  [WARN] {path} not found")
        return None

    df = pd.read_csv(path, encoding="latin-1", delimiter=";")
    required = {"betr_ID", "JAHR", "VERFAHREN", "CODE_KULTUR"}
    missing = required - set(df.columns)
    if missing:
        print(f"  [WARN] tillage CSV missing columns {missing}; "
              f"found {df.columns.tolist()}")
        return None

    df = df.rename(columns={"JAHR": "year", "CODE_KULTUR": "lnf_code"})
    df["tillage"] = df["VERFAHREN"].map(VERFAHREN_TO_TILLAGE)
    unmapped = df.loc[df["tillage"].isna(), "VERFAHREN"].unique().tolist()
    if unmapped:
        print(f"  [WARN] unmapped VERFAHREN values treated as Other: {unmapped}")
    df = df.dropna(subset=["tillage"])

    # Normalise key dtypes
    df["betr_ID"] = _norm_farm_id(df["betr_ID"])
    df["year"] = df["year"].astype(int)
    df["lnf_code"] = pd.to_numeric(df["lnf_code"], errors="coerce").astype("Int64")

    # Collapse to one tillage class per (farm, year, crop); drop ambiguous
    grp = (df.groupby(["betr_ID", "year", "lnf_code"])["tillage"]
           .agg(lambda s: sorted(set(s))))
    clean = grp[grp.apply(len) == 1].apply(lambda s: s[0]).reset_index()
    n_drop = (grp.apply(len) > 1).sum()
    if n_drop:
        print(f"  Dropped {n_drop} farm-crops with ambiguous tillage")
    return clean


def _norm_farm_id(s: pd.Series) -> pd.Series:
    """Normalise farm id to a clean integer-string (robust to 1000.0 vs 1000)."""
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().all():
        return num.astype("Int64").astype(str)
    return s.astype(str)


def analyse_tillage(cfg: dict, region: gpd.GeoDataFrame):
    """Field-level tillage distribution over the whole arable area of the region.

    Conservation tillage (Mulch/Direkt) comes from the schonende_bodenbearbeitung
    CSV, joined to LNF polygons by (betr_ID, year, lnf_code). Arable fields with
    no conservation-tillage record are inferred to be Pflug. Areas are therefore
    the *real arable area* of the district, not just the conservation subset.
    """
    print("\n=== Section 3: Tillage distribution ===")

    tillage = _load_tillage_lookup(cfg)
    if tillage is None:
        print("  [SKIP] No tillage data")
        return None

    # Restrict to arable crops via the lv3 bridge (Pflug only makes sense on arable)
    bridge = crop1990_to_lnf_bridge(cfg)
    arable_lnf = set(bridge.loc[
        bridge["Crop_Label_lv3"].isin(["Arable Land"]), "lnf_code"].tolist())

    results = {}      # year -> Series(area_ha by tillage)
    crosstabs = {}    # year -> DataFrame(crop × tillage area_ha)

    for yr in cfg["years"]:
        lnf_path = os.path.join(_expand(cfg["lnf_dir"]), f"lnf{yr}.gpkg")
        if not os.path.exists(lnf_path):
            print(f"  [WARN] {lnf_path} not found — skipping {yr}")
            continue

        lnf = gpd.read_file(
            lnf_path, bbox=tuple(region.to_crs("EPSG:2056").total_bounds))
        region_2056 = region.to_crs(lnf.crs).union_all()
        lnf = lnf[lnf.intersects(region_2056)].copy()
        lnf = lnf[lnf.geometry.is_valid & ~lnf.geometry.is_empty]

        if "betriebsnummer" not in lnf.columns:
            print(f"  [WARN] 'betriebsnummer' not in LNF {yr} "
                  f"(have {lnf.columns.tolist()}) — skipping")
            continue

        # Keep only arable fields
        lnf = lnf[lnf["lnf_code"].isin(arable_lnf)].copy()
        if lnf.empty:
            print(f"  [WARN] No arable fields in region for {yr}")
            continue

        lnf["area_ha"] = lnf.geometry.area / 1e4
        lnf["betr_ID"] = _norm_farm_id(lnf["betriebsnummer"])
        lnf["year"] = yr
        lnf["lnf_code"] = pd.to_numeric(lnf["lnf_code"],
                                        errors="coerce").astype("Int64")

        # Join conservation tillage; non-matches → Pflug
        merged = lnf.merge(tillage, on=["betr_ID", "year", "lnf_code"],
                           how="left")
        merged["tillage"] = merged["tillage"].fillna("Pflug")

        by_till = merged.groupby("tillage")["area_ha"].sum()
        results[yr] = by_till
        total = by_till.sum()
        n_cons = (merged["tillage"] != "Pflug").sum()
        print(f"  {yr}: {total:.0f} ha arable, "
              f"{n_cons}/{len(merged)} fields with conservation tillage")
        for t in ["Pflug", "Mulch", "Direkt"]:
            a = by_till.get(t, 0)
            print(f"    {t}: {a:.0f} ha ({100 * a / total:.1f}%)")

        # crop × tillage cross-tab (area ha), top crops
        ct = (merged.groupby(["lnf_code", "tillage"])["area_ha"].sum()
              .unstack(fill_value=0))
        crosstabs[yr] = ct

    if not results:
        print("  [WARN] No tillage results produced")
        return None

    # --- Bar chart: tillage area by year ---
    till_order = ["Pflug", "Mulch", "Direkt"]
    till_colors = {"Pflug": "#B07A56", "Mulch": "#E8C547", "Direkt": "#66BB6A"}
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bar_w = 0.38
    x = np.arange(len(till_order))
    for i, yr in enumerate(cfg["years"]):
        if yr not in results:
            continue
        vals = [results[yr].get(t, 0) for t in till_order]
        ax.bar(x + (i - 0.5) * bar_w, vals, width=bar_w,
               color=_year_color(yr), label=YEAR_LABELS[yr])
    ax.set_xticks(x)
    ax.set_xticklabels(till_order)
    ax.set_ylabel("Arable area (ha)")
    ax.set_title("Tillage distribution — Seeland district\n"
                 "(Pflug inferred where no conservation record)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    savefig(fig, "tillage_distribution.png", cfg)

    # --- Stacked share chart (proportions) ---
    fig2, ax2 = plt.subplots(figsize=(6.5, 4))
    years_present = [y for y in cfg["years"] if y in results]
    bottoms = np.zeros(len(years_present))
    for t in till_order:
        shares = [100 * results[y].get(t, 0) / results[y].sum()
                  for y in years_present]
        ax2.bar([str(y) for y in years_present], shares, bottom=bottoms,
                color=till_colors[t], label=t, width=0.5)
        bottoms += np.array(shares)
    ax2.set_ylabel("Share of arable area (%)")
    ax2.set_title("Tillage share — Seeland district")
    ax2.legend()
    ax2.set_ylim(0, 100)
    savefig(fig2, "tillage_share.png", cfg)

    # --- Crop × tillage cross-tab CSV (first year) ---
    code_to_name = {row.lnf_code: row.crop_label
                    for row in bridge.drop_duplicates("lnf_code").itertuples()}
    for yr in years_present:
        ct = crosstabs[yr].copy()
        ct["total"] = ct.sum(axis=1)
        ct = ct.sort_values("total", ascending=False).head(15).drop(columns="total")
        ct.index = [code_to_name.get(c, str(c)) for c in ct.index]
        out_path = os.path.join(cfg["out_dir"], f"tillage_by_crop_{yr}.csv")
        ct.to_csv(out_path)
        print(f"  Saved {out_path}")

    return results


# ===========================================================================
# Section 4 — Reference C-factor landscape
# ===========================================================================

def load_c_factor_table(cfg: dict) -> pd.DataFrame:
    """Load the C-factor reference table."""
    path = _expand(cfg["c_factor_csv"])
    df = pd.read_csv(path, sep=";", encoding="latin1")
    df = df.rename(columns={"Kultur Kategorien 2020": "Kultur_nutzung"})
    return df


def load_kulturmapping(cfg: dict) -> pd.DataFrame:
    path = _expand(cfg["kulturmapping_csv"])
    df = pd.read_csv(path, sep=";", encoding="latin1")
    return df


def _build_crop1990_cfactor_bridge(cfg: dict) -> pd.DataFrame:
    """Build crop1990 label → mean reference C-factor.

    Chain: crop_label (= Crop_Label in Excel) → lnf_code → kulturcode
           → Kultur_nutzung → C_Faktoren columns.
    Multiple lnf_codes may map to one crop_label; we average the C-factors.
    """
    c_table = load_c_factor_table(cfg)
    kmapping = load_kulturmapping(cfg)
    bridge_lnf = crop1990_to_lnf_bridge(cfg)  # crop_label, lnf_code, lv3

    # lnf_code → Kultur_nutzung
    lnf_to_kultur = (kmapping[["kulturcode", "Kultur_nutzung"]]
                     .drop_duplicates("kulturcode")
                     .rename(columns={"kulturcode": "lnf_code"}))

    c_cols = ["Total", "Tal_Pflug", "Tal_Mulch", "Tal_Direkt",
              "Berg_Pflug", "Berg_Mulch", "Berg_Direkt"]
    c_sub = c_table[["Kultur_nutzung"] + c_cols].copy()
    for col in c_cols:
        c_sub[col] = pd.to_numeric(c_sub[col], errors="coerce")

    # Join: crop_label → lnf_code → Kultur_nutzung → C-factor
    merged = (bridge_lnf
              .merge(lnf_to_kultur, on="lnf_code", how="inner")
              .merge(c_sub, on="Kultur_nutzung", how="inner"))

    # Average C across lnf_codes within each crop_label
    result = (merged.groupby("crop_label")[c_cols]
              .mean().reset_index())
    return result


def analyse_reference_cfactors(cfg: dict, region: gpd.GeoDataFrame,
                               crop_stats: dict | None = None):
    """Compute the area-weighted reference C-factor distribution for each year.

    Uses crop1990 raster areas bridged to C_Faktoren via the LNF classification.
    """
    print("\n=== Section 4: Reference C-factor landscape ===")
    crop_labels = load_crop1990_labels(cfg)
    code_to_name = {c: v["name"] for c, v in crop_labels.items()}
    c_bridge = _build_crop1990_cfactor_bridge(cfg)

    # If we have crop_stats from Section 2, use those
    if crop_stats is None:
        crop_stats = {}
        for yr in cfg["years"]:
            df = _crop_area_from_raster(yr, region, cfg)
            if df is not None:
                df["crop_name"] = df["crop_code"].map(code_to_name)
                crop_stats[yr] = df

    for yr in cfg["years"]:
        if yr not in crop_stats:
            continue
        cs = crop_stats[yr].copy()
        if "crop_name" not in cs.columns:
            cs["crop_name"] = cs["crop_code"].map(code_to_name)
        merged = cs.merge(c_bridge, left_on="crop_name", right_on="crop_label",
                          how="inner")
        total_area = merged["area_ha"].sum()
        if total_area == 0:
            print(f"  {yr}: no crops matched the C-factor table")
            continue

        c_mean = np.average(merged["Total"], weights=merged["area_ha"])
        print(f"  {yr}: area-weighted mean C_ref (Total) = {c_mean:.4f} "
              f"({total_area:.0f} ha matched / "
              f"{cs['area_ha'].sum():.0f} ha total)")

        c_tal_pflug = np.average(merged["Tal_Pflug"], weights=merged["area_ha"])
        print(f"       Tal_Pflug weighted mean = {c_tal_pflug:.4f}")

    # Histogram of crop-level reference C (using Total) for both years
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for i, yr in enumerate(cfg["years"]):
        if yr not in crop_stats:
            continue
        cs = crop_stats[yr].copy()
        if "crop_name" not in cs.columns:
            cs["crop_name"] = cs["crop_code"].map(code_to_name)
        merged = cs.merge(c_bridge, left_on="crop_name", right_on="crop_label",
                          how="inner")
        axes[i].hist(merged["Total"], weights=merged["area_ha"],
                     bins=30, color=_year_color(yr), alpha=0.8, edgecolor="white")
        axes[i].set_xlabel("Reference C-factor (Total)")
        axes[i].set_title(f"{yr}")
        axes[i].axvline(0.15, color="red", ls="--", lw=0.8, label="Risk threshold")
        axes[i].legend(fontsize=8)
    axes[0].set_ylabel("Area (ha)")
    fig.suptitle("Distribution of reference C-factors — Seeland district", y=1.02)
    fig.tight_layout()
    savefig(fig, "cfactor_reference_distribution.png", cfg)

    return c_bridge


# ===========================================================================
# Section 5 — Rainfall erosivity (EI)
# ===========================================================================

def analyse_ei(cfg: dict, region: gpd.GeoDataFrame):
    """Summarise climatological EI distribution across the region."""
    print("\n=== Section 5: Rainfall erosivity (EI) ===")
    ei_path = _expand(cfg["ei_path"])
    if not os.path.exists(ei_path):
        print(f"  [WARN] {ei_path} not found — skipping EI analysis")
        return None

    # Load EI for the region's bounding box
    import pyarrow.dataset as ds
    bounds = region_bounds_32632(region)
    minx, miny, maxx, maxy = bounds
    # Filter spatially
    filt = ((ds.field("x") >= minx) & (ds.field("x") <= maxx) &
            (ds.field("y") >= miny) & (ds.field("y") <= maxy))
    table = (ds.dataset(ei_path, format="parquet")
             .to_table(columns=["x", "y", "doy", "predicted_EI_daily_avg"],
                       filter=filt))
    df_ei = table.to_pandas().rename(columns={"predicted_EI_daily_avg": "ei"})
    print(f"  Loaded {len(df_ei):,} EI rows "
          f"({df_ei[['x','y']].drop_duplicates().shape[0]} grid cells)")

    if df_ei.empty:
        return None

    # Annual total EI per grid cell
    annual_ei = df_ei.groupby(["x", "y"])["ei"].sum()
    print(f"  Annual EI: mean = {annual_ei.mean():.1f}, "
          f"median = {annual_ei.median():.1f}, "
          f"range = [{annual_ei.min():.1f}, {annual_ei.max():.1f}]")

    # Seasonal distribution (mean across grid cells)
    monthly_ei = df_ei.copy()
    # Convert DOY to month (approximate)
    monthly_ei["month"] = pd.to_datetime(
        monthly_ei["doy"], format="%j").dt.month
    monthly_mean = monthly_ei.groupby("month")["ei"].sum() / df_ei[["x", "y"]].drop_duplicates().shape[0]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(monthly_mean.index, monthly_mean.values, color="#228833",
           edgecolor="white")
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean EI (MJ mm ha⁻¹ h⁻¹)")
    ax.set_title("Seasonal distribution of climatological rainfall erosivity — Seeland")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["J", "F", "M", "A", "M", "J",
                        "J", "A", "S", "O", "N", "D"])
    ax.grid(axis="y", alpha=0.3)
    savefig(fig, "ei_seasonal.png", cfg)

    return df_ei


# ===========================================================================
# Section 6 — Sentinel-2 data quality (from pixel outputs)
# ===========================================================================

def analyse_s2_quality(cfg: dict):
    """Load pixel-level outputs and summarise clean-obs counts."""
    print("\n=== Section 6: S2 data quality ===")
    pixel_dir = _expand(cfg["pixel_output_dir"])
    if not os.path.exists(pixel_dir):
        print(f"  [WARN] {pixel_dir} not found — skipping S2 quality analysis")
        return None

    all_fields = []
    for yr in cfg["years"]:
        # Try field summary first (lighter)
        fld_path = os.path.join(pixel_dir, f"cfactor_fields_{yr}.parquet")
        if os.path.exists(fld_path):
            fld = pd.read_parquet(fld_path)
            fld["year"] = yr
            all_fields.append(fld)
            print(f"  {yr}: {len(fld)} fields from field summary")
        else:
            # Load from hive-partitioned pixel parquet
            yr_dir = os.path.join(pixel_dir, f"year={yr}")
            if not os.path.exists(yr_dir):
                print(f"  [WARN] No pixel data for {yr}")
                continue
            pix = pd.read_parquet(yr_dir)
            fld = (pix.groupby(["poly_id", "lnf_code"], as_index=False)
                   .agg(n_pixels=("c_factor", "size"),
                        n_clean_obs_field=("n_clean_obs_field", "first")))
            fld["year"] = yr
            all_fields.append(fld)
            print(f"  {yr}: {len(fld)} fields from pixel parquet")

    if not all_fields:
        return None
    fields = pd.concat(all_fields, ignore_index=True)

    if "n_clean_obs_field" not in fields.columns:
        print("  [WARN] n_clean_obs_field column not found — skipping quality plot")
        return fields

    # Clean observations histogram (2021 vs 2022)
    fig, ax = plt.subplots(figsize=(7, 4))
    for yr in cfg["years"]:
        sub = fields[fields["year"] == yr]
        ax.hist(sub["n_clean_obs_field"], bins=40, alpha=0.6,
                color=_year_color(yr), label=f"{yr} (n={len(sub)})",
                edgecolor="white")
        med = sub["n_clean_obs_field"].median()
        ax.axvline(med, color=_year_color(yr), ls="--", lw=1.2)
    ax.set_xlabel("Clean S2 observations per field")
    ax.set_ylabel("Number of fields")
    ax.set_title("Sentinel-2 data quality — clean observations per field")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    savefig(fig, "s2_clean_obs.png", cfg)

    for yr in cfg["years"]:
        sub = fields[fields["year"] == yr]
        print(f"  {yr}: median clean obs = {sub['n_clean_obs_field'].median():.0f}, "
              f"mean = {sub['n_clean_obs_field'].mean():.1f}, "
              f"5th pct = {sub['n_clean_obs_field'].quantile(0.05):.0f}")

    return fields


# ===========================================================================
# Summary text
# ===========================================================================

def write_summary(cfg: dict):
    """Write a short summary text file collecting all printed stats."""
    # This is a placeholder — the script prints stats as it goes.
    # A more elaborate version could capture stdout or collect stats in a dict.
    print(f"\n{'='*60}")
    print(f"All figures saved to {cfg['out_dir']}/")
    print(f"{'='*60}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Area analysis: Seeland 2021 vs 2022")
    parser.add_argument("--skip-overview", action="store_true",
                        help="Skip Section 0 (study area overview)")
    parser.add_argument("--skip-meteo", action="store_true",
                        help="Skip MeteoSwiss loading (slow)")
    parser.add_argument("--skip-s2", action="store_true",
                        help="Skip pixel-output quality section")
    parser.add_argument("--skip-ei", action="store_true",
                        help="Skip EI analysis")
    parser.add_argument("--region", type=str, default=None,
                        help="Override region file path")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    cfg = dict(CONFIG)
    if args.region:
        cfg["region_path"] = args.region
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    _ensure_dir(cfg["out_dir"])

    # Load region
    print("Loading region boundary ...")
    region = load_region(cfg)
    region_2056 = region.to_crs("EPSG:2056")
    print(f"  Region area: {region_2056.union_all().area / 1e6:.1f} km²")

    # Section 0: Study area overview
    if not args.skip_overview:
        overview = analyse_study_area(cfg, region)
    else:
        print("\n[SKIP] Study area overview (--skip-overview)")

    # Section 1: Climate
    meteo_results = None
    if not args.skip_meteo:
        meteo_results = analyse_climate(cfg, region)
    else:
        print("\n[SKIP] Climate analysis (--skip-meteo)")

    # Section 2: Crops
    crop_stats = analyse_crops(cfg, region)

    # Section 3: Tillage
    tillage_data = analyse_tillage(cfg, region)

    # Section 4: Reference C-factors
    bridge = analyse_reference_cfactors(cfg, region, crop_stats)

    # Section 5: EI
    if not args.skip_ei:
        ei_data = analyse_ei(cfg, region)
    else:
        print("\n[SKIP] EI analysis (--skip-ei)")

    # Section 6: S2 quality
    if not args.skip_s2:
        s2_fields = analyse_s2_quality(cfg)
    else:
        print("\n[SKIP] S2 quality analysis (--skip-s2)")

    write_summary(cfg)


if __name__ == "__main__":
    main()