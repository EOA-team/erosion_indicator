import argparse
import os
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from tqdm import tqdm
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds
import pyproj
from shapely.geometry import box
import rasterio
import time
import glob


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
        grid_lv95 = grid.to_crs(2056)
    elif args.satellite == "landsat":
        grid = gpd.read_file(landsat_grid)
        grid_lv95 = grid.to_crs(2056)
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
    parser.add_argument("--year_start", type=int, required=True, help="The year of the input data.")
    parser.add_argument("--year_end", type=int, required=True, help="The year of the input data.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for Zarr tiles.")
    parser.add_argument("--out_res", type=int, required=True, default=100, help="Output resolution in meters.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--var", type=str, help="The variable to process.", choices=["TabsD", "TminD", "TmaxD", "RhiresD", "SrelD"])
    parser.add_argument("--meteo_dir", type=str, default="/mnt/Data-Raw-RE/27_Natural_Resources-RE/99_Meteo_Public/MeteoSwiss_netCDF/__griddedData/lv95updated/", help="The directory where the SwissMeteo data is stored in NetCDF format.")
    parser.add_argument("--satellite", type=str, default="sentinel", choices=["sentinel", "landsat"], help="The satellite to whose pixel grid the output should be aligned to.")
    parser.add_argument("--resampling", type=str, default="nearest", choices=[r.name for r in Resampling], help="The resampling method.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing Zarr stores if they already exist.")
    args = parser.parse_args()
    
    if args.resampling not in [r.name for r in Resampling]:
        parser.error("--resampling method is not supported")

    os.makedirs(args.out_dir, exist_ok=True)
    main(args)


# taskset -c 40-45 python meteo_daily_grids.py --year_start 2004 --year_end 2026 --out_dir covariates/temp --var RhiresD --out_res 100



