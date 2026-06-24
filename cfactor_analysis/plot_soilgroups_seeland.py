"""
Plot DLR soil-group predictions for the Seeland region over a Swisstopo
national colour map.

Loads every soil-group zarr (SRC_{left}_{top}.zarr) whose S2 tile footprint
intersects seeland.gpkg, mosaics them, clips to the region, masks
soil_group == 0 (and the standard -10000 / NaN sentinels), reprojects to
EPSG:2056 (Swiss LV95) and overlays it on the Swisstopo pixelkarte-farbe
basemap.

Run from the cfactor working directory (so seeland.gpkg resolves).
"""
from __future__ import annotations

import os

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rioxarray  # noqa: F401  -- registers the .rio accessor
import xarray as xr
from rasterio.enums import Resampling

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REGION_PATH = "../cfactor/seeland.gpkg"

SOIL_DIR = os.path.expanduser(
    "~/mnt/eo-nas1/data/satellite/sentinel2/DLR_soilsuite_preds"
)
S2_GRID_PATH = os.path.expanduser(
    "~/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure/"
    "Project layers/gridface_s2tiles_CH.shp"
)

# Data are stored in UTM 32N; plot in Swiss LV95 (native CRS of swisstopo).
SOIL_CRS = "EPSG:32632"
PLOT_CRS = "EPSG:2056"
SOIL_NODATA = (-10000,)        # sentinel(s) beyond NaN and the masked-out 0

# Background map. PositronNoLabels is light, subtle, no city/road labels —
# keeps the soil-group overlay readable. To change basemap, just uncomment
# a different return below.
def _basemap_source():
    import contextily as cx
    return cx.providers.CartoDB.PositronNoLabels
    # return cx.providers.CartoDB.Positron            # same, with faint labels
    # return cx.providers.CartoDB.Voyager             # pale green forests/parks
    # return cx.providers.Esri.WorldShadedRelief      # hillshade only, no labels
    # return ("https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.pixelkarte-grau/"
    #         "default/current/3857/{z}/{x}/{y}.jpeg")  # swisstopo grayscale

# Explicit categorical colours per soil-group integer.
SOIL_COLORS = {
    1: "teal",
    2: "darkorange",
    3: "purple",
    4: "deeppink",
    5: "limegreen",
}

# How opaque the soil-group overlay is (0 = invisible, 1 = hides basemap).
OVERLAY_ALPHA = 0.65

OUT_PNG = "seeland_soilgroups.png"


# ---------------------------------------------------------------------------
# Tile discovery
# ---------------------------------------------------------------------------

def discover_tiles(region: gpd.GeoDataFrame,
                   grid_path: str) -> list[tuple[int, int]]:
    """Return (left, top) tile IDs whose footprint intersects the region."""
    grid = gpd.read_file(grid_path)
    region_g = region.to_crs(grid.crs)
    hits = grid[grid.intersects(region_g.union_all())]
    return list(zip(hits["left"].astype(int), hits["top"].astype(int)))


# ---------------------------------------------------------------------------
# Mosaic + clip
# ---------------------------------------------------------------------------

def load_soilgroup_mosaic(tiles: list[tuple[int, int]],
                          soil_dir: str,
                          region_geom_32632) -> xr.DataArray:
    """Open every tile zarr, mosaic, clip to region (still in EPSG:32632)."""
    paths = []
    for left, top in tiles:
        p = os.path.join(soil_dir, f"SRC_{left}_{top}.zarr")
        if os.path.exists(p):
            paths.append(p)
        else:
            print(f"  [skip] missing tile zarr: {p}")
    if not paths:
        raise RuntimeError("No soil-group zarrs found for the region.")

    print(f"  opening {len(paths)} tile zarrs ...")
    ds = xr.open_mfdataset(paths, engine="zarr", combine="by_coords")[
        ["soil_group"]
    ]
    if "lat" in ds.dims or "lon" in ds.dims:
        ds = ds.rename({"lat": "y", "lon": "x"})

    ds = ds.rio.write_crs(SOIL_CRS)
    ds = ds.rio.clip([region_geom_32632], ds.rio.crs, drop=True)
    return ds["soil_group"]


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_soilgroups(sg: xr.DataArray,
                    region: gpd.GeoDataFrame,
                    out_path: str) -> None:
    """Reproject to LV95, overlay on the swisstopo basemap, save a PNG."""
    # Reproject categorical data with nearest-neighbour resampling.
    sg = sg.rio.write_crs(SOIL_CRS).rio.reproject(
        PLOT_CRS, resampling=Resampling.nearest
    )

    arr = sg.values.astype("float32")
    if arr.ndim == 3:
        arr = arr[0]

    # Mask: 0 (per request), sentinel(s), NaN.
    mask = np.isnan(arr) | (arr == 0)
    for nd in SOIL_NODATA:
        mask |= (arr == nd)
    arr = np.where(mask, np.nan, arr)

    if np.all(np.isnan(arr)):
        raise RuntimeError("No valid soil-group pixels after masking 0.")

    classes = sorted(int(v) for v in np.unique(arr[~np.isnan(arr)]))
    print(f"  soil_group classes present (0 excluded): {classes}")

    # Build a ListedColormap from the user-supplied colour table; fall back
    # to lightgray for any class that isn't covered.
    color_list, missing = [], []
    for c in classes:
        if c in SOIL_COLORS:
            color_list.append(SOIL_COLORS[c])
        else:
            color_list.append("lightgray")
            missing.append(c)
    if missing:
        print(f"  [warn] no colour defined for soil_group(s) {missing}; "
              f"using lightgray")

    cmap = mcolors.ListedColormap(color_list)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, len(classes) + 0.5),
                                ncolors=cmap.N)

    # Reindex the array to 0..N-1 so it lines up with the listed colormap.
    indexed = np.full_like(arr, np.nan)
    for i, c in enumerate(classes):
        indexed[arr == c] = i

    # imshow row 0 must land at the correct edge in geographic space.
    y_vals = sg.y.values
    origin = "upper" if y_vals[0] > y_vals[-1] else "lower"

    fig, ax = plt.subplots(figsize=(11, 9))

    # 1) Basemap first so the overlay sits on top.
    bounds = region.to_crs(PLOT_CRS).total_bounds  # (minx, miny, maxx, maxy)
    pad = 1000  # 1 km padding around the region
    ax.set_xlim(bounds[0] - pad, bounds[2] + pad)
    ax.set_ylim(bounds[1] - pad, bounds[3] + pad)
    ax.set_aspect("equal")

    try:
        import contextily as cx
        cx.add_basemap(ax, crs=PLOT_CRS, source=_basemap_source(),
                       attribution_size=6)
    except Exception as e:
        print(f"  [warn] basemap unavailable "
              f"({type(e).__name__}: {e}); plotting without it")

    # 2) Soil-group overlay (with alpha so the basemap shows through).
    extent = (float(sg.x.min()), float(sg.x.max()),
              float(sg.y.min()), float(sg.y.max()))
    ax.imshow(indexed, extent=extent, origin=origin,
              cmap=cmap, norm=norm, interpolation="nearest",
              alpha=OVERLAY_ALPHA, zorder=2)

    # 3) Region outline on top.
    region.to_crs(PLOT_CRS).boundary.plot(
        ax=ax, color="black", linewidth=1.2, zorder=3
    )

    # Legend (clearer than a colorbar for a handful of discrete classes).
    handles = [mpatches.Patch(facecolor=SOIL_COLORS.get(c, "lightgray"),
                              edgecolor="black", linewidth=0.4,
                              label=str(c), alpha=OVERLAY_ALPHA)
               for c in classes]
    ax.legend(handles=handles, loc="lower right", frameon=True,
              framealpha=0.9, title="Soil group")

    ax.set_xlabel("Easting (m, EPSG:2056)")
    ax.set_ylabel("Northing (m, EPSG:2056)")
    ax.set_title("Predicted soil groups — Seeland (0 masked)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    region = gpd.read_file(REGION_PATH)
    print(f"Region: {len(region)} feature(s), CRS={region.crs}")

    tiles = discover_tiles(region, S2_GRID_PATH)
    print(f"  {len(tiles)} S2 tiles overlap the region")
    if not tiles:
        raise RuntimeError("No overlapping S2 tiles for the region.")

    region_geom = region.to_crs(SOIL_CRS).union_all()
    sg = load_soilgroup_mosaic(tiles, SOIL_DIR, region_geom)
    plot_soilgroups(sg, region, OUT_PNG)


if __name__ == "__main__":
    main()