"""
Area analysis for the Seeland district — 2021 vs 2022.
 
Produces figures and summary statistics characterising the study area:
  1. Climate (MeteoSwiss: rainfall, temperature, sunshine)
  2. Crop distribution (from LNF gpkg)
  3. Tillage distribution — two variants: real arable area (LNF + conservation
     CSV) and Bodenbearbeitung as reported in the erosion-risk results
  4. Reported C-factor landscape (from per-farm × crop erosion-risk results)
  5. Rainfall erosivity (EI)
  6. Sentinel-2 data quality (from pixel-level outputs)
  7. Erosion risk — potential & P-combined (stats, plots, municipality maps)
 
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
 
    # C-factor reference table (no longer used for Section 4; kept for reference)
    "c_factor_csv":     "~/mnt/Data-Labo-RE/27_Natural_Resources-RE/"
                        "321.4_WAUM_protected/Daten/Erosionsrisiko/C_Faktoren.csv",
 
    # Previously reported erosion-risk results (per farm × crop), one folder per
    # year. Section 4 auto-picks the most recent
    #   *_Kultur_Betrieb_Erosionsrisiko_2x2_{year}.csv  in each {year}/ folder.
    # NB: the O:\ drive maps to ~/mnt/Data-Labo-Cert here — adjust if your mount
    # differs.
    "erosion_results_dir": "~/mnt/Data-Labo-RE/27_Natural_Resources-RE/"
                           "321.4_WAUM_protected/Resultate/Erosionsrisiko",
 
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
 
    # Municipality (Gemeinde) boundaries for the erosion-risk choropleth maps.
    # Must contain a name field matching the 'Gemeinde' column of the result
    # files. Set to None to skip the maps (stats/plots are still produced).
    "gemeinde_boundaries_path": "~/mnt/eo-nas1/eoa-share/projects/028_Erosion/"
                                "Erosion/FC_mapping/swissBOUNDARIES3D_1_5_LV95_LN02.gpkg",
    "gemeinde_name_field":      "name",
 
    # Erosion-risk threshold (t ha⁻¹ y⁻¹) used to report the "share of area at
    # risk" in the potential / P-risk section. Adjust to your tolerated soil loss.
    "erosion_risk_threshold_t_ha_y": 2.0,
 
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
 
 
def discover_meteo_tiles(cfg: dict, region: gpd.GeoDataFrame
                         ) -> list[tuple[int, int, float]]:
    """Find tiles overlapping the region with intersection-area weights.

    Returns a list of ``(minx, maxy, weight_m2)`` tuples, where ``weight_m2``
    is the area of intersection between the tile polygon and the region
    (computed in the grid's CRS, typically metres). Tiles that only touch
    the boundary (zero overlap area) are dropped.
    """
    grid = gpd.read_file(_expand(cfg["s2_grid_path"]))
    region_geom = region.to_crs(grid.crs).union_all()
    hits = grid[grid.intersects(region_geom)].copy()
    # Intersection area as the per-tile weight (in CRS units, here m²)
    hits["weight"] = hits.geometry.intersection(region_geom).area
    hits = hits[hits["weight"] > 0]
    return list(zip(
        hits["left"].astype(int),
        hits["top"].astype(int),
        hits["weight"].astype(float),
    ))
 
 
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
d
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


# --- Agricultural year helpers --------------------------------------------
# Agricultural year `year` runs 1 July (year-1) to 30 June (year).
# These are used by the climate (meteo) section so figures span the
# growing-season cycle rather than the calendar year.

def _agri_year_range(year: int) -> tuple[str, str]:
    """ISO start/end of agricultural year `year` (Jul (year-1) → Jun year)."""
    return f"{year - 1}-07-01", f"{year}-06-30"


def _agri_year_label(year: int) -> str:
    """Compact label, e.g. 2021 → '2020/21'."""
    return f"{year - 1}/{year % 100:02d}"


# 1=July, 12=June — position within the agricultural year
_AGRI_MONTH_ORDER = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]
_AGRI_MONTH_LABELS = ["J", "A", "S", "O", "N", "D", "J", "F", "M", "A", "M", "J"]
 
 
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
 
def _meteo_monthly_agg(var: str, year: int, tiles: list, cfg: dict,
                       agg: str = "sum") -> pd.Series | None:
    """Return monthly aggregated values for AGRICULTURAL year ``year``.

    The agricultural year covers 1 Jul (year-1) → 30 Jun (year). Per tile,
    the previous and current calendar-year zarr files are opened, the
    series is sliced to Jul–Jun, the spatial mean is taken (one tile in
    memory at a time), and per-tile series are combined as an
    area-weighted mean across tiles (weights from :func:`discover_meteo_tiles`).

    Returned Series is indexed 1..12 = Jul..Jun, so callers can plot
    against ``range(1, 13)`` together with ``_AGRI_MONTH_LABELS``.

    The previous implementation concatenated all tiles' lazy zarr arrays
    along ``time`` and used calendar years; the spatial-tile concat
    triggered an outer-join on mismatched ``x,y`` coords and materialised
    an enormous, mostly NaN array on ``.to_dataframe()`` — the cause of
    the 9-hour hang and OOM kill.
    """
    base = _expand(cfg["meteo_base"])
    # var names in config are like "RhiresD" — the folder name drops the D.
    # Filename pattern: MeteoSwiss_{var}_{minx}_{maxy}_{year}0101_{year}1231.zarr
    var_folder = var.replace("D", "")  # Rhires, Tabs, Srel
    start, end = _agri_year_range(year)

    def _open_tile_year(minx: int, maxy: int, yr: int) -> xr.DataArray | None:
        fname = f"MeteoSwiss_{var}_{minx}_{maxy}_{yr}0101_{yr}1231.zarr"
        path = os.path.join(base, var_folder, fname)
        if not os.path.exists(path):
            path = os.path.join(base, var, fname)
        if not os.path.exists(path):
            return None
        try:
            ds = xr.open_zarr(path)
        except Exception as e:
            print(f"    [WARN] Cannot open {path}: {e}")
            return None
        data_vars = list(ds.data_vars)
        if len(data_vars) == 1:
            return ds[data_vars[0]]
        cands = [v for v in data_vars
                 if var.replace("D", "").lower() in v.lower()]
        return ds[cands[0]] if cands else ds[data_vars[0]]

    series: list[xr.DataArray] = []
    weights: list[float] = []
    missing_prev = 0
    for tile in tiles:
        # Back-compat unpack: old (minx, maxy) tuples still work
        if len(tile) == 3:
            minx, maxy, w = tile
        else:
            minx, maxy = tile
            w = 1.0

        parts = []
        prev_da = _open_tile_year(minx, maxy, year - 1)
        if prev_da is not None:
            parts.append(prev_da)
        else:
            missing_prev += 1
        cur_da = _open_tile_year(minx, maxy, year)
        if cur_da is not None:
            parts.append(cur_da)
        if not parts:
            continue
        # Same tile across years → identical x,y coords → safe to concat on time
        da = xr.concat(parts, dim="time") if len(parts) > 1 else parts[0]
        da = da.sel(time=slice(start, end))
        if da.sizes.get("time", 0) == 0:
            continue

        # Spatial mean for this tile only — small, loads just this tile.
        try:
            tile_mean = (da.mean(dim=[d for d in da.dims if d != "time"])
                           .compute())
        except Exception as e:
            print(f"    [WARN] Cannot reduce tile ({minx},{maxy}): {e}")
            continue
        series.append(tile_mean)
        weights.append(float(w))

    if missing_prev:
        print(f"    [WARN] Previous-year zarr missing for {missing_prev} "
              f"tile(s); their Jul–Dec {year - 1} contribution is omitted.")
    if not series:
        return None

    stacked = xr.concat(series, dim="tile")
    w_da = xr.DataArray(np.asarray(weights, dtype=float), dims="tile")
    combined = stacked.weighted(w_da).mean(dim="tile")

    df = combined.to_dataframe(name="val").reset_index()
    df["month"] = pd.to_datetime(df["time"]).dt.month
    monthly = (df.groupby("month")["val"].sum() if agg == "sum"
               else df.groupby("month")["val"].mean())
    # Reorder to agricultural-year position: 1=Jul ... 12=Jun
    monthly = monthly.reindex(_AGRI_MONTH_ORDER)
    monthly.index = pd.RangeIndex(1, 13, name="agri_month")
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
            agri = _agri_year_label(yr)
            print(f"  Loading {var} {agri} (agri-year, Jul {yr-1}–Jun {yr}) ...")
            monthly = _meteo_monthly_agg(var, yr, tiles, cfg, agg=spec["agg"])
            if monthly is None:
                print(f"    [WARN] No data for {var} {agri}")
                continue
            results[(var, yr)] = monthly
            if spec["agg"] == "sum":
                ax.bar(monthly.index + (cfg["years"].index(yr) - 0.5) * 0.35,
                       monthly.values, width=0.35,
                       color=_year_color(yr), label=agri)
            else:
                ax.plot(monthly.index, monthly.values, "o-",
                        color=_year_color(yr), label=agri)
 
        ax.set_xlabel("Month (agricultural year, Jul → Jun)")
        ax.set_ylabel(spec["label"])
        ax.set_title(spec["title"])
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(_AGRI_MONTH_LABELS)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        savefig(fig, f"climate_{var}.png", cfg)
 
    # Annual totals / means summary (agricultural year)
    summary_lines = []
    for var in cfg["meteo_vars"]:
        spec = meteo_specs.get(var, {"agg": "mean", "label": var})
        for yr in cfg["years"]:
            key = (var, yr)
            if key in results:
                val = (results[key].sum() if spec["agg"] == "sum"
                       else results[key].mean())
                kind = "total" if spec["agg"] == "sum" else "mean"
                summary_lines.append(
                    f"  {var} {_agri_year_label(yr)}: {kind} = {val:.1f}")
    if summary_lines:
        print("  Annual summary (agricultural year):")
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
 
def _norm_farm_id(s: pd.Series) -> pd.Series:
    """Normalise farm id to a clean integer-string (robust to 1000.0 vs 1000)."""
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().all():
        return num.astype("Int64").astype(str)
    return s.astype(str)
 
 
# Conservation-tillage VERFAHREN → standard class. Anything not in this CSV
# is inferred to be conventional plough (Pflug). Used by Variant A only.
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
 
 
def analyse_tillage(cfg: dict, region: gpd.GeoDataFrame):
    """Tillage distribution — produced two ways so they can be cross-checked.
 
    Variant A ('realarea'): the real arable area of the region from LNF
    polygons, with conservation tillage (Mulch/Direkt) joined from the
    schonende_bodenbearbeitung CSV and Pflug inferred for the rest.
 
    Variant B ('results'): the Bodenbearbeitung as actually used in the erosion
    run, read from the result files and weighted by reported Flaeche.
 
    Output files are suffixed _realarea / _results so both are kept.
    """
    print("\n=== Section 3: Tillage distribution ===")
    realarea = _tillage_from_realarea(cfg, region)
    results = _tillage_from_results(cfg, region)
    return {"realarea": realarea, "results": results}
 
 
def _tillage_from_realarea(cfg: dict, region: gpd.GeoDataFrame):
    """Variant A — real arable area from LNF + conservation-tillage CSV."""
    print("\n  -- Variant A: real arable area (LNF + conservation CSV) --")
 
    tillage = _load_tillage_lookup(cfg)
    if tillage is None:
        print("  [SKIP] No conservation-tillage data")
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
        print("  [WARN] No tillage results produced (Variant A)")
        return None
 
    _plot_tillage(cfg, results, crosstabs, suffix="realarea",
                  area_label="Arable area (ha)",
                  subtitle="(real arable area; Pflug inferred)",
                  crop_index_map={
                      row.lnf_code: row.crop_label
                      for row in bridge.drop_duplicates("lnf_code").itertuples()})
    return results
 
 
def _tillage_from_results(cfg: dict, region: gpd.GeoDataFrame):
    """Variant B — Bodenbearbeitung as reported in the erosion results."""
    print("\n  -- Variant B: from erosion-result Bodenbearbeitung --")
 
    results = {}      # year -> Series(area_ha by Bodenbearbeitung)
    crosstabs = {}    # year -> DataFrame(crop × tillage area_ha)
 
    for yr in cfg["years"]:
        res = _load_erosion_results(cfg, yr)
        if res is None or res.empty:
            continue
        if res["Bodenbearbeitung"].isna().all():
            print(f"  [WARN] {yr}: no Bodenbearbeitung in result file — skipping")
            continue
 
        # Restrict to farms present in the region (same mode as Section 4).
        farm_ids = _region_farm_ids(cfg, region, yr)
        if farm_ids is not None:
            before = res["betr_ID"].nunique()
            res = res[res["betr_ID"].isin(farm_ids)]
            print(f"  {yr}: {res['betr_ID'].nunique()}/{before} farms within region")
            if res.empty:
                print(f"  [WARN] {yr}: no result rows fall within the region")
                continue
 
        res = res.assign(area_ha=res["Flaeche"] / 1e4)
        by_till = res.groupby("Bodenbearbeitung")["area_ha"].sum()
        results[yr] = by_till
        total = by_till.sum()
        print(f"  {yr}: {total:.0f} ha over {len(res)} polygons")
        for t in ["Pflug", "Mulch", "Direkt"]:
            a = by_till.get(t, 0.0)
            print(f"    {t}: {a:.0f} ha ({100 * a / total:.1f}%)")
        extra = sorted(set(by_till.index) - {"Pflug", "Mulch", "Direkt"})
        if extra:
            ex_area = by_till.reindex(extra).sum()
            print(f"    Other ({', '.join(map(str, extra))}): {ex_area:.0f} ha "
                  f"({100 * ex_area / total:.1f}%)")
 
        # crop × tillage cross-tab (area ha)
        ct = (res.groupby(["crop", "Bodenbearbeitung"])["area_ha"].sum()
              .unstack(fill_value=0))
        crosstabs[yr] = ct
 
    if not results:
        print("  [WARN] No tillage results produced (Variant B)")
        return None
 
    _plot_tillage(cfg, results, crosstabs, suffix="results",
                  area_label="Area (ha)",
                  subtitle="(Bodenbearbeitung as reported in the erosion results)",
                  crop_index_map=None)
    return results
 
 
def _plot_tillage(cfg, results, crosstabs, suffix, area_label, subtitle,
                  crop_index_map):
    """Shared plotting for both tillage variants. Files suffixed by ``suffix``."""
    till_order = ["Pflug", "Mulch", "Direkt"]
    till_colors = {"Pflug": "#B07A56", "Mulch": "#E8C547", "Direkt": "#66BB6A"}
    years_present = [y for y in cfg["years"] if y in results]
 
    # --- Bar chart: tillage area by year ---
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
    ax.set_ylabel(area_label)
    ax.set_title(f"Tillage distribution — Seeland district\n{subtitle}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    savefig(fig, f"tillage_distribution_{suffix}.png", cfg)
 
    # --- Stacked share chart (proportions of Pflug/Mulch/Direkt) ---
    fig2, ax2 = plt.subplots(figsize=(6.5, 4))
    bottoms = np.zeros(len(years_present))
    denom = {y: sum(results[y].get(t, 0) for t in till_order)
             for y in years_present}
    for t in till_order:
        shares = [100 * results[y].get(t, 0) / denom[y] if denom[y] else 0
                  for y in years_present]
        ax2.bar([str(y) for y in years_present], shares, bottom=bottoms,
                color=till_colors[t], label=t, width=0.5)
        bottoms += np.array(shares)
    ax2.set_ylabel("Share of tilled area (%)")
    ax2.set_title(f"Tillage share — Seeland district {subtitle}")
    ax2.legend()
    ax2.set_ylim(0, 100)
    savefig(fig2, f"tillage_share_{suffix}.png", cfg)
 
    # --- Crop × tillage cross-tab CSV (top crops by area) ---
    for yr in years_present:
        ct = crosstabs[yr].copy()
        ct["total"] = ct.sum(axis=1)
        ct = ct.sort_values("total", ascending=False).head(15).drop(columns="total")
        if crop_index_map is not None:
            ct.index = [crop_index_map.get(c, str(c)) for c in ct.index]
        out_path = os.path.join(cfg["out_dir"],
                                f"tillage_by_crop_{suffix}_{yr}.csv")
        ct.to_csv(out_path)
        print(f"  Saved {out_path}")
 
 
# ===========================================================================
# Section 4 — Reported C-factor landscape (from erosion-risk results)
# ===========================================================================
 
# Cache the parsed result frame per file so Sections 3 and 4 don't both re-read it.
_RESULTS_CACHE: dict[str, pd.DataFrame] = {}
 
 
def _load_erosion_results(cfg: dict, year: int) -> pd.DataFrame | None:
    """Load the previously reported per-farm × crop erosion results for ``year``.
 
    Picks the most recent ``*_Kultur_Betrieb_Erosionsrisiko_2x2_{year}.csv`` in
    ``{erosion_results_dir}/{year}/`` (the date prefix sorts chronologically).
    Returns a tidy frame: betr_ID, crop, Bodenbearbeitung, Region, Gemeinde,
    Flaeche (m²), C_fact_detail, pot_risk (Pot_Erosionsrisiko_t_ha_y),
    ers_risk_P — one row per polygon group.
    """
    import glob
    base = _expand(cfg["erosion_results_dir"])
    year_dir = os.path.join(base, str(year))
    pattern = os.path.join(
        year_dir, f"*_Kultur_Betrieb_Erosionsrisiko_2x2_{year}.csv")
    matches = sorted(glob.glob(pattern))
    if not matches:
        print(f"  [WARN] No result file matching {pattern}")
        return None
    path = matches[-1]  # latest date prefix
    if path in _RESULTS_CACHE:
        return _RESULTS_CACHE[path].copy()
    print(f"  {year}: reading {os.path.basename(path)}")
 
    df = pd.read_csv(path, sep=";", encoding="latin1")
 
    # Crop column is 'Nutzung_DE_KatNutz' in the export; accept 'Nutzung_DE' too.
    crop_col = next((c for c in ("Nutzung_DE_KatNutz", "Nutzung_DE")
                     if c in df.columns), None)
    required = {"betr_ID", "Flaeche", "C_fact_detail"}
    missing = required - set(df.columns)
    if crop_col is None:
        missing.add("Nutzung_DE_KatNutz")
    if missing:
        print(f"  [WARN] result file missing columns {missing}; "
              f"found {df.columns.tolist()}")
        return None
 
    out = pd.DataFrame({
        "betr_ID":          _norm_farm_id(df["betr_ID"]),
        "crop":             df[crop_col].astype(str),
        "Bodenbearbeitung": (df["Bodenbearbeitung"].astype(str)
                             if "Bodenbearbeitung" in df.columns else pd.NA),
        "Region":           df["Region"] if "Region" in df.columns else pd.NA,
        "Gemeinde":         (df["Gemeinde"].astype(str)
                             if "Gemeinde" in df.columns else pd.NA),
        "Flaeche":          pd.to_numeric(df["Flaeche"], errors="coerce"),
        "C_fact_detail":    pd.to_numeric(df["C_fact_detail"], errors="coerce"),
        "pot_risk":         (pd.to_numeric(df["Pot_Erosionsrisiko_t_ha_y"],
                                           errors="coerce")
                             if "Pot_Erosionsrisiko_t_ha_y" in df.columns
                             else np.nan),
        "ers_risk_P":       (pd.to_numeric(df["ers_risk_P"], errors="coerce")
                             if "ers_risk_P" in df.columns else np.nan),
    })
    out = out.dropna(subset=["Flaeche", "C_fact_detail"])
    out = out[out["Flaeche"] > 0]
    _RESULTS_CACHE[path] = out
    return out.copy()
 
 
def _region_farm_ids(cfg: dict, region: gpd.GeoDataFrame,
                     year: int) -> set | None:
    """Return betr_IDs whose LNF fields intersect the region in ``year``.
 
    Returns None if the LNF layer is unavailable (caller then keeps all farms).
    Note: this is a farm-level filter — a farm with any field in the region is
    kept in full, so a few of its out-of-region polygons may be included.
    """
    lnf_path = os.path.join(_expand(cfg["lnf_dir"]), f"lnf{year}.gpkg")
    if not os.path.exists(lnf_path):
        print(f"  [WARN] {lnf_path} not found — cannot restrict {year} to region")
        return None
    lnf = gpd.read_file(
        lnf_path, bbox=tuple(region.to_crs("EPSG:2056").total_bounds))
    if "betriebsnummer" not in lnf.columns:
        print(f"  [WARN] 'betriebsnummer' not in LNF {year} "
              f"(have {lnf.columns.tolist()}) — cannot restrict to region")
        return None
    region_geom = region.to_crs(lnf.crs).union_all()
    lnf = lnf[lnf.intersects(region_geom)]
    return set(_norm_farm_id(lnf["betriebsnummer"]).tolist())
 
 
def _aggregate_farm_crop_c(res: pd.DataFrame) -> pd.DataFrame:
    """Area-weighted mean C_fact_detail per (betr_ID, crop) across polygons."""
    res = res.copy()
    res["_wC"] = res["C_fact_detail"] * res["Flaeche"]
    agg = (res.groupby(["betr_ID", "crop"], as_index=False)
           .agg(_wC=("_wC", "sum"),
                Flaeche=("Flaeche", "sum"),
                n_poly=("C_fact_detail", "size")))
    agg["C_fact_detail"] = agg["_wC"] / agg["Flaeche"]
    agg["area_ha"] = agg["Flaeche"] / 1e4
    return agg.drop(columns=["_wC", "Flaeche"])
 
 
def analyse_reference_cfactors(cfg: dict, region: gpd.GeoDataFrame):
    """Describe the previously reported C-factors over the area of interest.
 
    Reads the per-farm × crop erosion-risk result files (one per year) and uses
    the reported ``C_fact_detail`` directly — no longer derived from the
    C_Faktoren reference table. Results are restricted to farms whose LNF fields
    intersect the region, then aggregated to one value per (betr_ID, crop) by
    area-weighting C_fact_detail with Flaeche (a farm-crop may span several
    polygons).
    """
    print("\n=== Section 4: Reported C-factor landscape (from erosion results) ===")
 
    per_year = {}   # year -> farm×crop frame [betr_ID, crop, n_poly, C_fact_detail, area_ha]
    for yr in cfg["years"]:
        res = _load_erosion_results(cfg, yr)
        if res is None or res.empty:
            continue
 
        # Restrict to farms present in the region (chosen restriction mode).
        farm_ids = _region_farm_ids(cfg, region, yr)
        if farm_ids is not None:
            before = res["betr_ID"].nunique()
            res = res[res["betr_ID"].isin(farm_ids)]
            print(f"  {yr}: {res['betr_ID'].nunique()}/{before} farms within region")
            if res.empty:
                print(f"  [WARN] {yr}: no result rows fall within the region")
                continue
 
        agg = _aggregate_farm_crop_c(res)
        per_year[yr] = agg
 
        cmean = np.average(agg["C_fact_detail"], weights=agg["area_ha"])
        print(f"  {yr}: {len(agg)} farm×crop units, "
              f"{agg['area_ha'].sum():.0f} ha")
        print(f"       area-weighted mean C_fact_detail = {cmean:.4f}, "
              f"median = {agg['C_fact_detail'].median():.4f}, "
              f"range = [{agg['C_fact_detail'].min():.4f}, "
              f"{agg['C_fact_detail'].max():.4f}]")
 
    if not per_year:
        print("  [WARN] No reported C-factor data produced")
        return None
 
    # --- Per-crop area-weighted summary (top crops by area) ---
    for yr in cfg["years"]:
        if yr not in per_year:
            continue
        agg = per_year[yr].assign(
            _wC=lambda d: d["C_fact_detail"] * d["area_ha"])
        by_crop = (agg.groupby("crop")
                   .agg(area_ha=("area_ha", "sum"),
                        _wC=("_wC", "sum"),
                        n_units=("crop", "size")))
        by_crop["C_fact_detail"] = by_crop["_wC"] / by_crop["area_ha"]
        by_crop = (by_crop.drop(columns="_wC")
                   .sort_values("area_ha", ascending=False).head(20))
        out_path = os.path.join(cfg["out_dir"],
                                f"cfactor_reported_by_crop_{yr}.csv")
        by_crop.to_csv(out_path)
        print(f"  Saved {out_path}")
 
    # --- Histogram of farm×crop C (area-weighted) for both years ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for i, yr in enumerate(cfg["years"]):
        if yr not in per_year:
            axes[i].set_visible(False)
            continue
        agg = per_year[yr]
        axes[i].hist(agg["C_fact_detail"], weights=agg["area_ha"],
                     bins=30, color=_year_color(yr), alpha=0.8,
                     edgecolor="white")
        axes[i].set_xlabel("Reported C-factor (C_fact_detail)")
        axes[i].set_title(f"{yr}")
        axes[i].axvline(0.15, color="red", ls="--", lw=0.8,
                        label="Risk threshold")
        axes[i].legend(fontsize=8)
    axes[0].set_ylabel("Area (ha)")
    fig.suptitle("Distribution of reported C-factors — area of interest", y=1.02)
    fig.tight_layout()
    savefig(fig, "cfactor_reported_distribution.png", cfg)
 
    return per_year
 
 
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
# Section 7 — Erosion risk (potential & P-combined) from the result files
# ===========================================================================
 
def _weighted_quantile(values, weights, q: float) -> float:
    """Area-weighted quantile (q in [0, 1])."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    order = np.argsort(v)
    v, w = v[order], w[order]
    cw = np.cumsum(w)
    if cw[-1] == 0:
        return float("nan")
    return float(v[np.searchsorted(cw, q * cw[-1])])
 
 
def _wmean_by(df: pd.DataFrame, value_col: str, by: str) -> pd.DataFrame:
    """Area-weighted mean of ``value_col`` grouped by ``by`` (drops NaN)."""
    d = df.dropna(subset=[value_col]).copy()
    d["_w"] = d[value_col] * d["area_ha"]
    g = d.groupby(by).agg(_w=("_w", "sum"), area_ha=("area_ha", "sum"))
    g[value_col] = g["_w"] / g["area_ha"]
    return g.drop(columns="_w").reset_index()
 
 
def _plot_risk_histograms(cfg: dict, per_year: dict, thr: float):
    """Area-weighted histograms of potential & P-combined risk, per year."""
    years = [y for y in cfg["years"] if y in per_year]
    metrics = [("pot_risk", "Potential"), ("ers_risk_P", "P-combined")]
    ncol = max(len(years), 1)
    fig, axes = plt.subplots(len(metrics), ncol,
                             figsize=(5 * ncol, 7), squeeze=False)
    for r, (col, name) in enumerate(metrics):
        for c, yr in enumerate(years):
            ax = axes[r][c]
            sub = per_year[yr].dropna(subset=[col])
            if sub.empty:
                ax.set_visible(False)
                continue
            hi = _weighted_quantile(sub[col], sub["area_ha"], 0.99)
            hi = hi if hi and hi > 0 else float(sub[col].max() or 1)
            ax.hist(np.clip(sub[col], 0, hi), weights=sub["area_ha"],
                    bins=40, color=_year_color(yr), alpha=0.85,
                    edgecolor="white")
            ax.axvline(thr, color="red", ls="--", lw=0.9,
                       label=f"{thr:g} t ha⁻¹ y⁻¹")
            ax.set_title(f"{name} — {yr}")
            ax.set_xlabel("t ha⁻¹ y⁻¹")
            if c == 0:
                ax.set_ylabel("Area (ha)")
            ax.legend(fontsize=8)
    fig.suptitle("Erosion-risk distribution (area-weighted) — area of interest",
                 y=1.01)
    fig.tight_layout()
    savefig(fig, "erosion_risk_histograms.png", cfg)
 
 
def _plot_risk_by_tillage(cfg: dict, per_year: dict):
    """Bar chart: area-weighted mean P-combined risk by tillage class."""
    till_order = ["Pflug", "Mulch", "Direkt"]
    years = [y for y in cfg["years"] if y in per_year]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bar_w = 0.38
    x = np.arange(len(till_order))
    any_data = False
    for i, yr in enumerate(years):
        res = per_year[yr].dropna(subset=["ers_risk_P"])
        if res.empty or res["Bodenbearbeitung"].isna().all():
            continue
        g = (_wmean_by(res, "ers_risk_P", "Bodenbearbeitung")
             .set_index("Bodenbearbeitung"))
        vals = [g["ers_risk_P"].get(t, np.nan) for t in till_order]
        ax.bar(x + (i - 0.5) * bar_w, vals, width=bar_w,
               color=_year_color(yr), label=YEAR_LABELS[yr])
        any_data = True
    if not any_data:
        plt.close(fig)
        return
    ax.set_xticks(x)
    ax.set_xticklabels(till_order)
    ax.set_ylabel("Mean P-combined risk (t ha⁻¹ y⁻¹)")
    ax.set_title("P-combined erosion risk by tillage — Seeland district")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    savefig(fig, "erosion_risk_by_tillage.png", cfg)
 
 
def _gemeinde_choropleth(cfg: dict, region: gpd.GeoDataFrame, gem_stats: dict,
                         value_col: str, label: str, fname: str):
    """Choropleth of ``value_col`` per municipality, one panel per year.
 
    Needs ``gemeinde_boundaries_path`` with a name field matching the result
    files' ``Gemeinde`` column. Skips gracefully (stats/plots already produced)
    if the file or a name match is unavailable.
    """
    bnd_path = cfg.get("gemeinde_boundaries_path")
    if not bnd_path:
        print(f"  [INFO] gemeinde_boundaries_path not set — skipping {fname} map")
        return
    bnd_path = _expand(bnd_path)
    if not os.path.exists(bnd_path):
        print(f"  [WARN] {bnd_path} not found — skipping {fname} map")
        return
    name_field = cfg.get("gemeinde_name_field", "name")
    try:
        gem = gpd.read_file(bnd_path, layer='tlm_hoheitsgebiet')
    except Exception as e:
        print(f"  [WARN] could not read {bnd_path} ({e}) — skipping {fname} map")
        return
    if name_field not in gem.columns:
        print(f"  [WARN] name field '{name_field}' not in boundaries "
              f"(have {gem.columns.tolist()[:12]}) — skipping {fname} map")
        return
 
    region_geom = region.to_crs(gem.crs).union_all()
    gem = gem[gem.intersects(region_geom)].copy()
    if gem.empty:
        print(f"  [WARN] no municipalities intersect region — skipping {fname} map")
        return
    gem["_key"] = gem[name_field].astype(str).str.strip()
 
    years = [y for y in cfg["years"] if y in gem_stats]
    if not years:
        return
    allvals = pd.concat([gem_stats[y][value_col] for y in years])
    vmin = float(np.nanmin(allvals))
    vmax = float(np.nanpercentile(allvals, 98))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.nanmax(allvals))
 
    ncol = max(len(years), 1)
    fig, axes = plt.subplots(1, ncol, figsize=(6 * ncol, 6), squeeze=False)
    axes = axes[0]
    region_gem_crs = region.to_crs(gem.crs)
    for ax, yr in zip(axes, years):
        st = gem_stats[yr].copy()
        st["_key"] = st["Gemeinde"].astype(str).str.strip()
        merged = gem.merge(st[["_key", value_col]], on="_key", how="left")
        n_match = int(merged[value_col].notna().sum())
        print(f"  {fname} {yr}: {n_match}/{len(gem)} municipalities matched")
        merged.plot(column=value_col, ax=ax, cmap="YlOrRd",
                    vmin=vmin, vmax=vmax, edgecolor="white", linewidth=0.3,
                    legend=True, missing_kwds={"color": "lightgrey"})
        region_gem_crs.boundary.plot(ax=ax, color="black", linewidth=0.8)
        ax.set_title(f"{yr}")
        ax.set_axis_off()
    fig.suptitle(f"{label} by municipality — area of interest", y=1.02)
    fig.tight_layout()
    savefig(fig, f"{fname}.png", cfg)
 
 
def analyse_erosion_risk(cfg: dict, region: gpd.GeoDataFrame):
    """Describe potential and P-combined erosion risk over the area of interest.
 
    Both come straight from the per-farm × crop result files:
      - ``pot_risk``   = Pot_Erosionsrisiko_t_ha_y (potential, no C/P)
      - ``ers_risk_P`` = actual risk with the P-factor applied
    Restricted to farms within the region. Produces stats, area-weighted
    histograms, a by-tillage bar chart, per-crop CSVs, and (if municipality
    boundaries are configured) choropleth maps.
    """
    print("\n=== Section 7: Erosion risk — potential & P-combined ===")
    thr = cfg.get("erosion_risk_threshold_t_ha_y", 2.0)
 
    per_year = {}    # year -> tidy frame [..., area_ha, pot_risk, ers_risk_P]
    gem_pot = {}     # year -> per-municipality area-weighted pot_risk
    gem_p = {}       # year -> per-municipality area-weighted ers_risk_P
 
    for yr in cfg["years"]:
        res = _load_erosion_results(cfg, yr)
        if res is None or res.empty:
            continue
        has_pot = res["pot_risk"].notna().any()
        has_p = res["ers_risk_P"].notna().any()
        if not (has_pot or has_p):
            print(f"  [WARN] {yr}: no Pot_Erosionsrisiko_t_ha_y / ers_risk_P columns")
            continue
 
        farm_ids = _region_farm_ids(cfg, region, yr)
        if farm_ids is not None:
            before = res["betr_ID"].nunique()
            res = res[res["betr_ID"].isin(farm_ids)]
            print(f"  {yr}: {res['betr_ID'].nunique()}/{before} farms within region")
            if res.empty:
                print(f"  [WARN] {yr}: no result rows fall within the region")
                continue
 
        res = res.assign(area_ha=res["Flaeche"] / 1e4)
        per_year[yr] = res
 
        total_area = res["area_ha"].sum()
        print(f"  {yr}: {len(res)} polygons, {total_area:.0f} ha")
        for col, name in [("pot_risk", "potential"), ("ers_risk_P", "P-combined")]:
            sub = res.dropna(subset=[col])
            if sub.empty:
                continue
            wm = np.average(sub[col], weights=sub["area_ha"])
            med = _weighted_quantile(sub[col], sub["area_ha"], 0.5)
            at_risk = sub.loc[sub[col] > thr, "area_ha"].sum()
            print(f"    {name}: area-wt mean = {wm:.2f}, median = {med:.2f}, "
                  f"max = {sub[col].max():.1f} t/ha/y; "
                  f">{thr:g} on {at_risk:.0f} ha "
                  f"({100 * at_risk / sub['area_ha'].sum():.1f}%)")
 
        # Breakdowns of P-combined risk by tillage and by Tal/Berg
        if has_p:
            for grp_col in ["Bodenbearbeitung", "Region"]:
                if res[grp_col].isna().all():
                    continue
                g = _wmean_by(res.dropna(subset=["ers_risk_P"]),
                              "ers_risk_P", grp_col)
                print(f"    P-combined by {grp_col}:")
                for _, row in g.iterrows():
                    print(f"      {row[grp_col]}: {row['ers_risk_P']:.2f} t/ha/y "
                          f"({row['area_ha']:.0f} ha)")
 
        # Per-municipality aggregation for maps
        if res["Gemeinde"].notna().any():
            gg = res.dropna(subset=["Gemeinde"])
            if has_pot:
                gem_pot[yr] = _wmean_by(gg, "pot_risk", "Gemeinde")
            if has_p:
                gem_p[yr] = _wmean_by(gg, "ers_risk_P", "Gemeinde")
 
        # Per-crop summary CSV
        rows = []
        for crop, d in res.groupby("crop"):
            row = {"crop": crop, "area_ha": d["area_ha"].sum(), "n_poly": len(d)}
            for col in ["pot_risk", "ers_risk_P"]:
                dd = d.dropna(subset=[col])
                row[f"mean_{col}"] = (np.average(dd[col], weights=dd["area_ha"])
                                      if not dd.empty else np.nan)
            rows.append(row)
        by_crop = (pd.DataFrame(rows)
                   .sort_values("area_ha", ascending=False).head(20))
        out_path = os.path.join(cfg["out_dir"], f"erosion_risk_by_crop_{yr}.csv")
        by_crop.to_csv(out_path, index=False)
        print(f"  Saved {out_path}")
 
    if not per_year:
        print("  [WARN] No erosion-risk data produced")
        return None
 
    _plot_risk_histograms(cfg, per_year, thr)
    _plot_risk_by_tillage(cfg, per_year)
 
    if gem_pot:
        _gemeinde_choropleth(cfg, region, gem_pot, "pot_risk",
                             "Potential erosion risk (t ha⁻¹ y⁻¹)",
                             "map_pot_risk")
    if gem_p:
        _gemeinde_choropleth(cfg, region, gem_p, "ers_risk_P",
                             "P-combined erosion risk (t ha⁻¹ y⁻¹)",
                             "map_ers_risk_P")
 
    return per_year
 
 
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
    parser.add_argument("--skip-risk", action="store_true",
                        help="Skip Section 7 (potential & P-combined erosion risk)")
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
 
    # Section 4: Reported C-factors (from erosion-risk result files)
    reported_c = analyse_reference_cfactors(cfg, region)
 
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
 
    # Section 7: Erosion risk (potential & P-combined)
    if not args.skip_risk:
        risk_data = analyse_erosion_risk(cfg, region)
    else:
        print("\n[SKIP] Erosion-risk analysis (--skip-risk)")
 
    write_summary(cfg)
 
 
if __name__ == "__main__":
    main()