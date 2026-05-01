import argparse
import os
import warnings
warnings.filterwarnings(
    "ignore",
    message="angle from rectified to skew grid parameter lost in conversion to CF",
    category=UserWarning,
    module="pyproj.crs._cf1x8"
)

import numpy as np
import xarray as xr
from numcodecs import Blosc
import geopandas as gpd
import rioxarray
from rasterio.enums import Resampling
import rasterio
import time
import zarr as _zarr


HSCLQMD_NC_PATH = "/home/f80873755@agsad.admin.ch/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure/DataInfrastructure/Meteo/HSCLQMD_ch01h.swiss.lv95_WY_1962_2023.nc"


def prepare_snow_depth(args, template):
    """
    Read HSCLQMD (daily snow height) from the raw NetCDF, compute long-term
    monthly climatology (12 months), reproject to the target grid, and write as
    zarr with 12 timestamps (2001-01-01 … 2001-12-01) representing Jan–Dec means.
    """
    print(f"Opening {HSCLQMD_NC_PATH} ...")
    ds = xr.open_dataset(HSCLQMD_NC_PATH)

    # Extract CRS from the grid_mapping variable before dropping it
    crs_wkt = ds["swiss_lv95_coordinates"].attrs["crs_wkt"]

    # Rename LV95 axes to standard x/y and drop auxiliary 2-D lon/lat coords
    ds = (
        ds.rename({"E": "x", "N": "y"})
          .drop_vars(["swiss_lv95_coordinates", "lon", "lat"], errors="ignore")
          .rio.write_crs(crs_wkt)
    )

    # Filter to 2004 onwards
    ds = ds.sel(time=slice("2004-01-01", None))
    print(f"Filtered to {len(ds.time)} time steps: "
          f"{str(ds.time.values[0])[:10]} to {str(ds.time.values[-1])[:10]}")

    # Long-term monthly climatology: group by calendar month, mean over all years
    print("Computing long-term monthly climatology (groupby month)...")
    ds_clim = ds.groupby("time.month").mean("time").load()
    ds_clim = ds_clim.rename({"HSCLQMD": "snow_depth"})
    print(f"Climatology computed: {dict(ds_clim.dims)}")

    out_path = os.path.join(args.out_dir, "snow_depth_monthly_avg.zarr")
    compressor = Blosc(cname="zstd", clevel=3, shuffle=2)

    print(f"Reprojecting and writing to {out_path} ...")
    t0 = time.time()
    for i, m in enumerate(ds_clim.month.values):
        print(f"  [month {m:02d}/12] ...", end=" ", flush=True)
        ds_m = ds_clim.sel(month=m).drop_vars("month").rio.write_crs(crs_wkt)
        ds_m = reproject_single_time(ds_m, template)
        ts = np.datetime64(f"2001-{m:02d}-01")
        ds_m = ds_m.expand_dims(time=[ts])
        for var in ds_m.data_vars:
            ds_m[var].attrs.pop("_FillValue", None)
        if i == 0:
            ds_m.to_zarr(
                out_path,
                mode="w",
                zarr_format=2,
                encoding={v: {"compressor": compressor} for v in ds_m.data_vars},
            )
        else:
            ds_m.to_zarr(out_path, append_dim="time", mode="a")
        elapsed = time.time() - t0
        print(f"done ({elapsed:.0f}s elapsed)")

    _zarr.consolidate_metadata(out_path)
    print(f"Saved: {out_path} ({time.time() - t0:.0f}s total)")


def load_data(args, year):
    """
    Load the data from NetCDF.
    """
    if args.var in ["TabsD", "TminD", "TmaxD"]:
        gridtype = "ch01r"
    else:
        gridtype = "ch01h"

    if year >= 2025:
        var_lower = args.var.lower()
        path = os.path.join(args.meteo_dir, f"ogd-surface-derived-grid-archive.{var_lower}_{gridtype}.swiss.lv95_{year}0101000000_{year}1231000000.nc")
    else:
        path = os.path.join(args.meteo_dir, f"{args.var}_{gridtype}.swiss.lv95_{year}01010000_{year}12310000.nc")

    return xr.open_dataset(path).rename({"E": "x", "N": "y"})

def reproject_single_time(da, template):
        return da.rio.reproject_match(template, resampling=rasterio.enums.Resampling.bilinear)

def main(args):

    # Load the satellite grid of interest
    sentinel_grid = "/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure/Project layers/gridface_s2tiles_CH.shp"
    landsat_grid = "/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure/Project layers/grid_landsat_CH.shp"
    if args.satellite == "sentinel":
        grid = gpd.read_file(sentinel_grid)
    elif args.satellite == "landsat":
        grid = gpd.read_file(landsat_grid)
    else:
        raise ValueError("Satellite must be one of 'sentinel' or 'landsat'")

    # Define final grid
    res = args.out_res  # desired resolution in meters
    minx, miny, maxx, maxy = grid.total_bounds
    x = np.arange(minx, maxx + res, res)
    y = np.arange(miny, maxy + res, res)

    template = xr.DataArray(
        np.zeros((len(y), len(x))),
        dims=("y", "x"),
        coords={"y": y, "x": x}
    ).rio.write_crs("EPSG:32632")  # target CRS

    if args.var == "snow_depth":
        prepare_snow_depth(args, template)
        return

    if args.year_start is None or args.year_end is None:
        raise ValueError("--year_start and --year_end are required for MeteoSwiss variables")

    for yr in range(args.year_start, args.year_end, 1):

        # Load data
        print(f"LOADING DATA", yr)

        ds = load_data(args, yr).load()

        # Get attributes of the data
        args.original_attributes = ds.attrs
        args.crs = ds[args.var].attrs["esri_pe_string"]
        ds = ds.rio.write_crs(args.crs)

        # Get the original pixel resolution and define the padding (default=3)
        x_res, y_res = (ds[args.var].x[1] - ds[args.var].x[0]).values, (ds[args.var].y[1] - ds[args.var].y[0]).values
        assert abs(x_res - y_res) < 1e-6, f"Pixel size mismatch: x={x_res}, y={y_res}"
        args.pad = 3

        # Reproject each timestamp
        ds_reprojected = xr.concat([reproject_single_time(ds.sel(time=t), template) for t in ds.time], dim='time')
        path = f"MeteoSwiss_{args.var}_{yr}0101_{yr}1231.zarr"
        compressor = Blosc(cname='zstd', clevel=3, shuffle=2)
        ds_reprojected.to_zarr(os.path.join(args.out_dir, path), consolidated=True, mode='w', zarr_format=2, encoding={var: {'compressor': compressor} for var in ds_reprojected.data_vars})
        print('Processed data for', yr)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
      Reproject and resample MeteoSwiss data. \n
      The --resampling flag allows you to define the resampling method. Be aware that, in order to generate data across the whole input raster, the outer most pixels are repeated across the raster boundaries. Otherwise, pixels would be lost due to
      the resampling mathematics. \n
      Be aware that small reprojection errors are introduced for the two most western pixel columns due to big differences between LV95 and EPSG:32632.
      Each year is processed and saved seperately.
      """, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--year_start", type=int, required=False, default=None, help="Start year (required for MeteoSwiss variables; not used for snow_depth).")
    parser.add_argument("--year_end", type=int, required=False, default=None, help="End year exclusive (required for MeteoSwiss variables; not used for snow_depth).")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for Zarr tiles.")
    parser.add_argument("--out_res", type=int, required=True, default=100, help="Output resolution in meters.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--var", type=str, help="The variable to process.", choices=["TabsD", "TminD", "TmaxD", "RhiresD", "SrelD", "snow_depth"])
    parser.add_argument("--meteo_dir", type=str, default="/mnt/Data-Raw-RE/27_Natural_Resources-RE/99_Meteo_Public/MeteoSwiss_netCDF/__griddedData/lv95updated/", help="The directory where the SwissMeteo data is stored in NetCDF format.")
    parser.add_argument("--satellite", type=str, default="sentinel", choices=["sentinel", "landsat"], help="The satellite to whose pixel grid the output should be aligned to.")
    parser.add_argument("--resampling", type=str, default="nearest", choices=[r.name for r in Resampling], help="The resampling method.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing Zarr stores if they already exist.")
    args = parser.parse_args()

    if args.resampling not in [r.name for r in Resampling]:
        parser.error("--resampling method is not supported")

    os.makedirs(args.out_dir, exist_ok=True)
    main(args)


# python meteo_daily_grids.py --year_start 2004 --year_end 2026 --out_dir covariates/temp --var RhiresD --out_res 100
# python meteo_daily_grids.py --var snow_depth --out_dir covariates/snow_depth --out_res 100
