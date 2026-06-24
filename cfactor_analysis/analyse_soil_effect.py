"""
Investigate whether soil_group (DLR soil-suite) affects the C-factor.

Reads existing pixel parquets and the R per-farm CSV; never recomputes C.
Soil_group is sampled PER PIXEL by nearest-neighbour lookup against the
SRC_{minx}_{maxy}.zarr tiles (same files used by Stage A½ of
compute_cfactor_pixels.py). For each LNF field we keep the modal soil_group
across its pixels -- this matches the field-level granularity at which the
FC predictor was actually conditioned on soil.

Why the question is interesting
-------------------------------
Soil_group can influence the C-factor through three pathways, and the three
products see them differently:
  * Empirical (beta):  via SALI's per-soil-group FC predictor -> SLR -> C.
  * ML:                via the trained ML pipeline (FC predictor + features).
  * Previous (R):      none directly. Prasuhn's C lookup is
                       crop x region x tillage; any soil signal there is the
                       crop <-> soil correlation. R is the NULL baseline.

What it produces (in cfg['out_dir'])
------------------------------------
  soil_summary_by_product_year.csv     per (product, year, soil_group)
  soil_field_table_all.csv             tidy per-field table behind the plots
  soil_by_crop_{product}_{year}.csv    per-crop heatmap data
  soil_dist_by_product_year.png        boxplots of field-mean C by soil_group
  soil_by_crop_heatmap_{p}_{y}.png     mean C per (crop, soil_group)
  soil_kruskal.csv                     Kruskal-Wallis H per (product, year)
  soil_summary.txt                     headline summary

Usage
-----
    python analyse_soil_effect.py
    python analyse_soil_effect.py --years 2021 2022
    python analyse_soil_effect.py --skip-r       # empirical & ML only

NB: shares CONFIG with compare_products.py (region, lnf paths, bridges) and
adds two soil-specific keys (soil_dir, soil_name_template). Edit CONFIG
below or pass --region / --out-dir on the CLI.
"""
from __future__ import annotations

import argparse
import os
import warnings

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reuse compare_products' bridges/helpers so we stay consistent.
from compare_products import (
    _build_bridges,
    _ensure_dir,
    _expand,
    _savecsv,
    _wavg,
    CONFIG as BASE_CONFIG,
    PIX_RES,
    PROD_COLORS,
    PROD_LABELS,
    Summary,
    load_pixels,
    load_r_results,
    savefig,
)

warnings.filterwarnings("ignore")

# DLR_soilsuite tiles are 1280 m x 1280 m in EPSG:32632, named
# SRC_{minx}_{maxy}.zarr (same convention as the S2 raw tiles used by
# compute_cfactor_pixels.py -- see TILE_SIZE_M there).
TILE_SIZE_M = 1280


# ---------------------------------------------------------------------------
# Configuration (inherits from compare_products.CONFIG; override soil bits)
# ---------------------------------------------------------------------------

CONFIG = dict(BASE_CONFIG)
CONFIG.update({
    # Per-tile zarr with a `soil_group` variable. Mirrors
    # compute_cfactor_pixels.CONFIG['soil_dir'] / 'soil_name_template'.
    "soil_dir":            "~/mnt/eo-nas1/data/satellite/sentinel2/"
                           "DLR_soilsuite_preds/",
    "soil_name_template":  "SRC_{minx}_{maxy}.zarr",

    # --- Analysis knobs ---
    "top_n_crops_heatmap": 12,   # crops shown in per-crop heatmaps
    "min_fields_per_cell": 5,    # mask (crop, soil) cells with fewer fields
    "drop_unknown_soil":   True, # drop soil_group=0 (unavailable) from analyses

    "out_dir":             "figures_soil",
})


# ===========================================================================
# Soil-group sampling -- per pixel, nearest-neighbour against SRC_*.zarr
# ===========================================================================

def _list_soil_tiles(soil_dir: str, template: str
                     ) -> list[tuple[int, int]]:
    """Discover available SRC_*.zarr tiles by directory listing.

    Returns the list of (minx, maxy) tile keys parsed straight from the
    filenames. We deliberately do NOT assume the tile grid is anchored at
    multiples of TILE_SIZE_M -- the actual S2-MGRS-derived origins can carry
    an offset, so computing keys arithmetically silently misses everything.
    """
    import re
    d = _expand(soil_dir)
    if not os.path.isdir(d):
        return []
    # Build a regex from the user-supplied template by escaping it and then
    # un-escaping the {minx}/{maxy} placeholders into capture groups.
    pat = (re.escape(template)
             .replace(re.escape("{minx}"), r"(?P<minx>-?\d+)")
             .replace(re.escape("{maxy}"), r"(?P<maxy>-?\d+)"))
    rx = re.compile(f"^{pat}$")
    keys = []
    for f in os.listdir(d):
        m = rx.match(f)
        if m:
            keys.append((int(m.group("minx")), int(m.group("maxy"))))
    return sorted(keys)


def _sample_soil_for_tile(soil_dir: str, template: str, key,
                          sub_xy: pd.DataFrame) -> np.ndarray:
    """Open ONE SRC_*.zarr and sample its `soil_group` variable at the given
    EPSG:32632 (x, y) points. Returns int array (0 = unavailable)."""
    import xarray as xr
    import rioxarray   # noqa: F401  registers the .rio accessor

    minx, maxy = key
    path = os.path.join(_expand(soil_dir),
                        template.format(minx=minx, maxy=maxy))
    if not os.path.exists(path):
        return np.zeros(len(sub_xy), dtype=np.int32)
    try:
        ds = xr.open_zarr(path)
        # The zarrs use lat/lon names for what are actually projected y/x
        # (matches _field_soil_groups in compute_cfactor_pixels.py).
        rename = {}
        if "lat" in ds.dims or "lat" in ds.coords:
            rename["lat"] = "y"
        if "lon" in ds.dims or "lon" in ds.coords:
            rename["lon"] = "x"
        if rename:
            ds = ds.rename(rename)
        sg = ds["soil_group"]
        if sg.ndim == 3:                        # drop time/band if present
            sg = sg.isel({sg.dims[0]: 0})
        # Vectorised nearest-neighbour sample (no full-array load).
        xs = xr.DataArray(sub_xy["x"].to_numpy(), dims="pt")
        ys = xr.DataArray(sub_xy["y"].to_numpy(), dims="pt")
        vals = sg.sel(x=xs, y=ys, method="nearest").values
        vals = np.where(np.isnan(vals) | (vals == -10000), 0, vals)
        return vals.astype(np.int32)
    except Exception as exc:                    # noqa: BLE001
        print(f"  [WARN] soil tile {key}: {exc}")
        return np.zeros(len(sub_xy), dtype=np.int32)


def attach_soil_group(pixels: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add a per-pixel `soil_group` column to `pixels` (must have x, y in
    EPSG:32632). Discovers available soil tiles from the directory, then
    for each tile masks the pixels falling inside its bbox and samples
    soil_group for those pixels. Robust to tile-grid offsets.
    """
    x = pixels["x"].to_numpy()
    y = pixels["y"].to_numpy()
    out = np.zeros(len(pixels), dtype=np.int32)

    tiles = _list_soil_tiles(cfg["soil_dir"], cfg["soil_name_template"])
    if not tiles:
        print(f"  [WARN] No tiles matching '{cfg['soil_name_template']}' "
              f"found in {_expand(cfg['soil_dir'])} -- "
              f"soil_group will be 0 everywhere")
        return pixels.assign(soil_group=out)

    # Show one example so the user can sanity-check the tile-grid origin.
    ex = tiles[0]
    print(f"  Found {len(tiles)} soil tiles in {_expand(cfg['soil_dir'])} "
          f"(e.g. minx={ex[0]}, maxy={ex[1]}); "
          f"pixel x range [{x.min():.0f}, {x.max():.0f}], "
          f"y range [{y.min():.0f}, {y.max():.0f}]")

    # Restrict to tiles whose bbox intersects the pixel bbox.
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    tiles_xy = [(tx, ty) for tx, ty in tiles
                if (tx + TILE_SIZE_M) > xmin and tx < xmax
                and ty > ymin and (ty - TILE_SIZE_M) < ymax]

    n_hit = 0
    for tx, ty in tiles_xy:
        # tile bbox = [tx, tx + TILE_SIZE_M) x [ty - TILE_SIZE_M, ty)
        mask = ((x >= tx) & (x < tx + TILE_SIZE_M)
                & (y >= ty - TILE_SIZE_M) & (y < ty))
        if not mask.any():
            continue
        sub = pixels.loc[mask, ["x", "y"]].reset_index(drop=True)
        vals = _sample_soil_for_tile(cfg["soil_dir"], cfg["soil_name_template"],
                                     (tx, ty), sub)
        out[mask] = vals
        if (vals > 0).any():
            n_hit += 1

    pct = (out > 0).sum() / max(1, len(pixels)) * 100
    print(f"  Sampled soil_group: {len(tiles_xy)} tiles intersect the pixel "
          f"bbox ({n_hit} with data), {len(pixels):,} pixels, "
          f"{pct:.0f}% labelled (0 = unavailable)")
    return pixels.assign(soil_group=out)


# ===========================================================================
# Field-level table builders
# ===========================================================================

def _crop_label(poly_id, lnf_code, year, crop_by_poly, crop_by_lnf):
    """Mirror compare_products' crop labelling: prefer the per-year LNF
    nutzung_DE (lnf method), fall back to the lnf_code -> kategorie bridge."""
    if crop_by_poly:
        pp = crop_by_poly.get(year, {}).get(int(poly_id))
        if pp and str(pp).strip() and str(pp).lower() != "nan":
            return str(pp).strip()
    if pd.notna(lnf_code):
        cc = crop_by_lnf.get(int(lnf_code))
        if cc:
            return str(cc).strip()
    return None


def _rollup_to_field(pix: pd.DataFrame, product_key: str, year: int,
                     crop_by_poly: dict, crop_by_lnf: dict) -> pd.DataFrame:
    """Pixel df with c_factor + soil_group -> per-(poly_id, lnf_code) row.

    The modal soil_group is computed via a value-counts pivot rather than
    `groupby().agg(lambda)`: lambdas inside `agg` mix poorly with the
    `as_index=False` codepath when group keys carry nullable dtypes
    (lnf_code is Int64), which raises a misleading
    "Length of values does not match length of index" error.
    """
    # Be defensive about lnf_code's nullable Int64 (hive-partition artefact):
    # cast to a plain int64 so all downstream groupby paths are uniform.
    pix = pix.copy()
    if "lnf_code" in pix.columns:
        pix["lnf_code"] = pd.to_numeric(pix["lnf_code"],
                                        errors="coerce").astype("int64")

    # Simple numeric aggregations -- no lambdas, no edge cases.
    fld = (pix.groupby(["poly_id", "lnf_code"], as_index=False)
              .agg(c_factor_mean=("c_factor", "mean"),
                   n_pixels=("c_factor", "size")))

    # Modal soil_group via per-(poly_id, lnf_code, soil_group) counts, then
    # keep the most-frequent row per (poly_id, lnf_code). Ties broken by
    # the lower soil_group label (deterministic).
    counts = (pix.groupby(["poly_id", "lnf_code", "soil_group"], as_index=False)
                 .size())
    sg = (counts.sort_values(["poly_id", "lnf_code", "size", "soil_group"],
                             ascending=[True, True, False, True])
                 .drop_duplicates(["poly_id", "lnf_code"])
                 [["poly_id", "lnf_code", "soil_group"]])
    fld = fld.merge(sg, on=["poly_id", "lnf_code"], how="left")
    fld["soil_group"] = fld["soil_group"].fillna(0).astype(int)

    fld["area_m2"] = fld["n_pixels"] * PIX_RES ** 2
    fld["year"] = year
    fld["product"] = product_key
    fld["crop"] = [_crop_label(p, l, year, crop_by_poly, crop_by_lnf)
                   for p, l in zip(fld["poly_id"], fld["lnf_code"])]
    return fld


def build_pixel_product_table(product_key: str, year: int, cfg: dict,
                              crop_by_poly: dict, crop_by_lnf: dict,
                              soil_xy_cache: dict | None = None
                              ) -> pd.DataFrame | None:
    """Empirical / ML: load per-pixel parquet, attach per-pixel soil_group
    (reusing the cache if the empirical pass already paid the IO), roll up
    to field level. Returns None if the parquet is missing.
    """
    pix = load_pixels(product_key, cfg, year)
    if pix is None or pix.empty:
        return None
    # Empirical and ML share the SAME 10 m pixel grid by construction
    # (the ML script reuses Stage A's pixel inventory). So the soil sampling
    # is identical -- do it once per year, join the rest.
    if soil_xy_cache is not None and year in soil_xy_cache:
        key = soil_xy_cache[year]
        pix = pix.merge(key, on=["x", "y"], how="left")
        pix["soil_group"] = pix["soil_group"].fillna(0).astype(np.int32)
        n_miss = int(pix["soil_group"].eq(0).sum() - key["soil_group"].eq(0).sum())
        if n_miss > 0:
            print(f"  [INFO] {product_key} {year}: {n_miss:,} extra pixels "
                  f"not in soil cache -- resampling those")
            mask = pix["soil_group"].eq(0)
            extra = attach_soil_group(pix.loc[mask].copy(), cfg)
            pix.loc[mask, "soil_group"] = extra["soil_group"].values
    else:
        pix = attach_soil_group(pix, cfg)
        if soil_xy_cache is not None:
            soil_xy_cache[year] = pix[["x", "y", "soil_group"]].drop_duplicates(
                ["x", "y"]).reset_index(drop=True)
    return _rollup_to_field(pix, product_key, year, crop_by_poly, crop_by_lnf)


def build_r_product_table(year: int, cfg: dict, lnf_bridge_y: pd.DataFrame,
                          soil_by_poly_year: dict) -> pd.DataFrame | None:
    """R: per (betr_ID, crop) -> propagate to LNF polys via the bridge.
    Soil_group comes from the per-pixel sampling cached at the LNF poly_id
    (modal across the field's pixels, computed once on the empirical pass).
    """
    if lnf_bridge_y is None or not soil_by_poly_year:
        return None
    r = load_r_results(cfg, year)
    if r is None:
        return None

    # If R has duplicates at (betr_ID, crop) (rare at this granularity),
    # take an area-weighted mean of C_fact_detail.
    r_agg = (r.groupby(["betr_ID", "crop"], as_index=False)
              .apply(lambda d: pd.Series({
                  "C_fact_detail": _wavg(d["C_fact_detail"], d["Flaeche"]),
              }))
              .reset_index(drop=True))

    lb = lnf_bridge_y.copy()
    lb = lb[lb["betr_ID"].notna() & lb["nutzung_DE"].notna()]
    lb["crop"] = lb["nutzung_DE"].astype(str).str.strip()
    # Match betr_ID typing (both should already be the normalised string form
    # from _norm_farm_id, but be defensive).
    lb["betr_ID"] = lb["betr_ID"].astype(str)
    r_agg["betr_ID"] = r_agg["betr_ID"].astype(str)

    m = lb.merge(r_agg, on=["betr_ID", "crop"], how="inner")
    if m.empty:
        return None

    out = pd.DataFrame({
        "poly_id":       m["poly_id"].astype(int),
        "lnf_code":      m["lnf_code"] if "lnf_code" in m.columns else pd.NA,
        "c_factor_mean": m["C_fact_detail"],
        "n_pixels":      pd.NA,
        "soil_group":    m["poly_id"].map(soil_by_poly_year)
                                     .fillna(0).astype(int),
        "area_m2":       m["lnf_area_m2"].astype(float),
        "year":          year,
        "product":       "previous",
        "crop":          m["crop"],
    })
    return out


# ===========================================================================
# Analyses
# ===========================================================================

def soil_summary_table(df: pd.DataFrame, cfg: dict, summ: Summary
                       ) -> pd.DataFrame:
    """Per (product, year, soil_group): n, area, area-weighted and unweighted
    central tendency. The area-weighted mean (`mean_C_w`) is the one to compare
    across products / years -- it's the analogue of the headline numbers used
    by the rest of compare_products."""
    g = (df.groupby(["product", "year", "soil_group"], as_index=False)
           .apply(lambda d: pd.Series({
               "n_fields":  int(len(d)),
               "area_ha":   float(d["area_m2"].sum() / 1e4),
               "mean_C_w":  _wavg(d["c_factor_mean"], d["area_m2"]),
               "mean_C":    float(d["c_factor_mean"].mean()),
               "median_C":  float(d["c_factor_mean"].median()),
               "std_C":     float(d["c_factor_mean"].std()),
           }))
           .reset_index(drop=True))
    _savecsv(g, "soil_summary_by_product_year.csv", cfg)
    return g


def plot_distribution(df: pd.DataFrame, cfg: dict, summ: Summary):
    """Boxplots of field-mean C by soil_group, faceted by product x year."""
    order = ["empirical", "ml", "previous"]
    products = [p for p in order if p in df["product"].unique()]
    years = sorted(df["year"].unique())
    groups = sorted(df["soil_group"].unique())

    fig, axes = plt.subplots(len(products), len(years),
                             figsize=(4.2 * max(1, len(years)),
                                      3.0 * max(1, len(products))),
                             sharey=True, squeeze=False)
    for i, p in enumerate(products):
        for j, y in enumerate(years):
            ax = axes[i, j]
            sub = df[(df["product"] == p) & (df["year"] == y)]
            data = [sub.loc[sub["soil_group"] == g, "c_factor_mean"]
                          .dropna().values for g in groups]
            ns = [len(d) for d in data]
            bp = ax.boxplot(data, positions=range(len(groups)),
                            widths=0.6, showfliers=False, patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_facecolor(PROD_COLORS.get(p, "#888888"))
                patch.set_alpha(0.55)
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels([str(g) for g in groups], fontsize=8)
            # n labels
            ymax = ax.get_ylim()[1]
            for k, n in enumerate(ns):
                if n > 0:
                    ax.text(k, ymax * 0.97, f"n={n}", ha="center", va="top",
                            fontsize=6, color="#444444")
            if i == len(products) - 1:
                ax.set_xlabel("soil_group")
            if j == 0:
                ax.set_ylabel(f"{PROD_LABELS.get(p, p)}\nfield-mean C")
            if i == 0:
                ax.set_title(f"{y}")
    fig.suptitle("Field-level mean C by soil_group "
                 "(outliers hidden; n above each box)")
    fig.tight_layout()
    savefig(fig, "soil_dist_by_product_year.png", cfg)


def plot_per_crop_heatmap(df: pd.DataFrame, cfg: dict, summ: Summary):
    """One heatmap per (product, year): rows = top crops, cols = soil_group,
    cell = area-weighted mean C. Cells with fewer than `min_fields_per_cell`
    fields are masked (NaN, shown blank) -- they're too thin to interpret.

    Crop order is computed ONCE across all (product, year) -- ranked by total
    area pooled over every panel -- so the y-axis is identical on every
    heatmap. A crop with no data in some (product, year) shows up as a blank
    row in that panel, which is itself informative (absence stands out).
    """
    top_n = cfg.get("top_n_crops_heatmap", 12)
    min_n = cfg.get("min_fields_per_cell", 5)
    order = ["empirical", "ml", "previous"]
    products = [p for p in order if p in df["product"].unique()]

    # Single global crop ordering for the whole figure set.
    global_top = (df.dropna(subset=["crop"])
                    .groupby("crop")["area_m2"].sum()
                    .sort_values(ascending=False)
                    .head(top_n).index.tolist())
    if not global_top:
        return

    for p in products:
        for y in sorted(df["year"].unique()):
            sub = df[(df["product"] == p) & (df["year"] == y)].dropna(
                subset=["crop"])
            if sub.empty:
                continue
            sub = sub[sub["crop"].isin(global_top)]
            if sub.empty:
                continue
            cells = (sub.groupby(["crop", "soil_group"], as_index=False)
                        .apply(lambda d: pd.Series({
                            "mean_C": _wavg(d["c_factor_mean"], d["area_m2"]),
                            "n":      int(len(d)),
                            "area_ha": float(d["area_m2"].sum() / 1e4),
                        }))
                        .reset_index(drop=True))
            _savecsv(cells, f"soil_by_crop_{p}_{y}.csv", cfg)

            mat = cells.pivot(index="crop", columns="soil_group",
                              values="mean_C")
            cnts = cells.pivot(index="crop", columns="soil_group",
                               values="n").fillna(0)
            mat = mat.where(cnts >= min_n)
            # Enforce the global row order; missing crops become NaN rows.
            mat = mat.reindex(global_top)
            cnts = cnts.reindex(global_top).fillna(0)

            if mat.dropna(how="all").empty:
                continue

            fig, ax = plt.subplots(
                figsize=(1.0 * len(mat.columns) + 3.5,
                         0.45 * len(mat.index) + 2.5))
            im = ax.imshow(mat.values, aspect="auto", cmap="viridis")
            ax.set_xticks(range(len(mat.columns)))
            ax.set_xticklabels(mat.columns)
            ax.set_yticks(range(len(mat.index)))
            ax.set_yticklabels(mat.index, fontsize=8)
            ax.set_xlabel("soil_group")
            ax.set_title(f"{PROD_LABELS.get(p, p)}, {y}: "
                         f"mean C per (crop, soil_group)\n"
                         f"area-weighted; cells with <{min_n} fields masked")
            mean_v = np.nanmean(mat.values)
            for ri in range(mat.shape[0]):
                for ci in range(mat.shape[1]):
                    v = mat.values[ri, ci]
                    n = int(cnts.values[ri, ci])
                    if not np.isnan(v):
                        ax.text(ci, ri, f"{v:.3f}\nn={n}",
                                ha="center", va="center", fontsize=6,
                                color="white" if v > mean_v else "black")
            fig.colorbar(im, ax=ax, label="mean C (area-weighted)")
            fig.tight_layout()
            savefig(fig, f"soil_by_crop_heatmap_{p}_{y}.png", cfg)


def kruskal_per_product_year(df: pd.DataFrame, summ: Summary
                             ) -> pd.DataFrame | None:
    """Kruskal-Wallis H on field-mean C ~ soil_group, per (product, year).
    Reports unweighted (we'd need a custom permutation test to do this right
    weighted; the unweighted version is a fine first pass and matches what
    most stats packages give you out of the box).
    """
    try:
        from scipy.stats import kruskal
    except ImportError:
        print("  [WARN] scipy not available -- skipping Kruskal-Wallis")
        return None
    rows = []
    for (p, y), sub in df.groupby(["product", "year"]):
        groups = [g["c_factor_mean"].dropna().values
                  for _, g in sub.groupby("soil_group") if len(g) >= 5]
        if len(groups) < 2:
            continue
        try:
            h, pval = kruskal(*groups)
            # eta-squared analogue for KW: (H - k + 1) / (N - k)
            N = sum(len(g) for g in groups)
            k = len(groups)
            eta2 = max(0.0, (h - k + 1) / max(1, N - k))
            rows.append({"product": p, "year": int(y), "n_groups": k,
                         "n_total": N, "H": float(h), "p_value": float(pval),
                         "eta2": float(eta2)})
        except Exception:
            continue
    return pd.DataFrame(rows) if rows else None


# ===========================================================================
# Main
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Analyse soil_group effect on the C-factor")
    ap.add_argument("--years", nargs="+", type=int, default=None)
    ap.add_argument("--region", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--skip-r", action="store_true",
                    help="skip the R product (analyse empirical/ML only)")
    ap.add_argument("--keep-unknown-soil", action="store_true",
                    help="keep soil_group=0 (unavailable) in plots/tests")
    a = ap.parse_args()

    cfg = dict(CONFIG)
    if a.years:
        cfg["years"] = a.years
    if a.region:
        cfg["region_path"] = a.region
    if a.out_dir:
        cfg["out_dir"] = a.out_dir
    if a.keep_unknown_soil:
        cfg["drop_unknown_soil"] = False
    _ensure_dir(cfg["out_dir"])

    summ = Summary(cfg)
    # Rename the summary file (Summary writes compare_summary.txt by default;
    # we want soil_summary.txt to avoid stepping on compare_products' output).
    summ.add(f"Soil-group effect analysis for years {cfg['years']}")
    summ.add(f"  region:     {cfg['region_path']}")
    summ.add(f"  soil_dir:   {cfg['soil_dir']}")
    summ.add(f"  out_dir:    {cfg['out_dir']}")

    # Bridges (region polygon, LNF per year, crop labels).
    region = gpd.read_file(_expand(cfg["region_path"]))
    region_2056 = region.to_crs("EPSG:2056").union_all()
    _, crop_by_lnf, _, lnf_bridge, crop_by_poly = \
        _build_bridges(cfg, region_2056, cfg["years"])

    parts: list[pd.DataFrame] = []
    soil_xy_cache: dict = {}        # year -> [x, y, soil_group] (shared emp/ML)
    soil_by_poly: dict = {}         # year -> {poly_id: modal soil_group}

    for year in cfg["years"]:
        summ.section(f"Year {year}")
        for pk in ("empirical", "ml"):
            summ.add(f"  Loading {pk} pixels for {year} ...")
            tab = build_pixel_product_table(pk, year, cfg,
                                            crop_by_poly, crop_by_lnf,
                                            soil_xy_cache=soil_xy_cache)
            if tab is None:
                summ.add(f"    [skipped] no parquet for {pk} {year}")
                continue
            summ.add(f"    {len(tab):,} fields, "
                     f"{tab['area_m2'].sum() / 1e4:,.0f} ha")
            parts.append(tab)
            # Cache poly_id -> modal soil_group ONCE per year (from empirical;
            # ml shares the grid and would yield the same modes).
            if year not in soil_by_poly:
                soil_by_poly[year] = dict(zip(tab["poly_id"], tab["soil_group"]))

        if not a.skip_r:
            summ.add(f"  Loading R results for {year} ...")
            r_tab = build_r_product_table(year, cfg, lnf_bridge.get(year),
                                          soil_by_poly.get(year, {}))
            if r_tab is None:
                summ.add("    [skipped] no R results or no LNF bridge")
            else:
                summ.add(f"    {len(r_tab):,} poly_id-level rows propagated "
                         f"from {r_tab['betr_ID'].nunique() if 'betr_ID' in r_tab.columns else r_tab[['poly_id']].drop_duplicates().shape[0]:,} fields")
                parts.append(r_tab)

    if not parts:
        summ.add("  [ERROR] no tables built -- nothing to analyse.")
        _flush_named(summ, "soil_summary.txt")
        return

    df_all = pd.concat(parts, ignore_index=True)
    # Field-level rows: drop any with bad C.
    df_all = df_all.dropna(subset=["c_factor_mean", "soil_group", "area_m2"])
    df_all["soil_group"] = df_all["soil_group"].astype(int)

    summ.section("Combined table")
    counts = df_all.groupby("product").size().to_dict()
    summ.add(f"  rows by product: {counts}")
    summ.add(f"  soil groups present: {sorted(df_all['soil_group'].unique())}")
    if cfg.get("drop_unknown_soil", True):
        n_before = len(df_all)
        df_all = df_all[df_all["soil_group"] > 0].reset_index(drop=True)
        summ.add(f"  dropping soil_group=0 (unavailable): "
                 f"{n_before - len(df_all):,} rows removed; "
                 f"{len(df_all):,} kept")
    _savecsv(df_all, "soil_field_table_all.csv", cfg)

    # E2: summary table
    summ.section("E2. Summary table (per product, year, soil_group)")
    summary = soil_summary_table(df_all, cfg, summ)
    # Inline the headline rows (area-weighted means per product/year)
    by_py = (df_all.groupby(["product", "year"])
                   .apply(lambda d: _wavg(d["c_factor_mean"], d["area_m2"]))
                   .rename("mean_C_w").reset_index())
    for _, r in by_py.iterrows():
        summ.add(f"    {r['product']:>10s} {int(r['year'])}: "
                 f"overall mean_C_w = {r['mean_C_w']:.4f}")

    # E1: distributions
    summ.section("E1. Distribution: field-mean C by soil_group (boxplots)")
    plot_distribution(df_all, cfg, summ)
    summ.add("  -> soil_dist_by_product_year.png")

    # E4: per-crop heatmap (the "is it real conditional on crop?" diagnostic)
    summ.section("E4. Per-crop stratification (heatmap)")
    plot_per_crop_heatmap(df_all, cfg, summ)
    summ.add("  -> soil_by_crop_heatmap_{product}_{year}.png")

    # E3: Kruskal-Wallis (unconditional; pair with the heatmap above)
    summ.section("E3. Effect-size test: Kruskal-Wallis on C ~ soil_group")
    kw = kruskal_per_product_year(df_all, summ)
    if kw is not None and not kw.empty:
        _savecsv(kw, "soil_kruskal.csv", cfg)
        for _, r in kw.iterrows():
            stars = ("***" if r["p_value"] < 0.001
                     else "**" if r["p_value"] < 0.01
                     else "*" if r["p_value"] < 0.05 else "ns")
            summ.add(f"    {r['product']:>10s} {int(r['year'])}: "
                     f"H={r['H']:7.1f}, p={r['p_value']:.2e} ({stars}), "
                     f"k={int(r['n_groups'])}, "
                     f"eta2={r['eta2']:.3f}")
        summ.add("")
        summ.add("  Interpretation:")
        summ.add("    * The R (previous) product is the NULL baseline -- its")
        summ.add("      Prasuhn lookup is crop x region x tillage, so any soil")
        summ.add("      signal there is the crop <-> soil correlation only.")
        summ.add("    * If empirical/ML eta2 noticeably exceeds R's, that's")
        summ.add("      the direct FC-pathway contribution (different soil")
        summ.add("      groups -> different per-soil FC predictor -> SLR -> C).")
        summ.add("    * The per-crop heatmap (E4) lets you eyeball whether")
        summ.add("      the effect persists WITHIN crops; flat rows there mean")
        summ.add("      the marginal effect was mostly soil-crop confounding.")

    _flush_named(summ, "soil_summary.txt")
    print(f"\nDone. See {cfg['out_dir']}/")


def _flush_named(summ: Summary, name: str):
    """Write the Summary to a custom filename (Summary.flush() hardcodes
    compare_summary.txt, and we don't want to collide with that here)."""
    _ensure_dir(summ.cfg["out_dir"])
    path = os.path.join(summ.cfg["out_dir"], name)
    with open(path, "w") as f:
        f.write("\n".join(summ.lines) + "\n")
    print(f"\n  Summary written to {path}")


if __name__ == "__main__":
    main()