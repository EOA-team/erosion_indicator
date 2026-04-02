import os
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray
import xdem

# Compute DEM on 100m grid with nearest interpolation
out_dir = 'covariates'
os.makedirs(out_dir, exist_ok=True)
out_res = 100

dem_grid_path = os.path.join(out_dir, f'dem_{out_res}m.zarr')
if not os.path.exists(dem_grid_path):

    dem_dir = os.path.expanduser('~/mnt/eo-nas1/data/swisstopo/DEM') # files sa3D_MINX_MAXY.zarr

    sentinel_grid = "/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure/Project layers/gridface_s2tiles_CH.shp"
    grid = gpd.read_file(sentinel_grid)
    grid_lv95 = grid.to_crs(2056)
        
    # Define final grid
    minx, miny, maxx, maxy = grid.total_bounds
    x = np.arange(minx, maxx + out_res, out_res)
    y = np.arange(miny, maxy + out_res, out_res)

    zarr_files = [os.path.join(dem_dir, f) for f in os.listdir(dem_dir) if f.endswith('.zarr')]
    valid_files = []
    for f in zarr_files:
        try:
            tmp = xr.open_zarr(f, chunks="auto")
            if tmp.data_vars:  # only keep if it has variables
                valid_files.append(f)
        except Exception as e:
            print(f"Skipping {f}: {e}")
    ds = xr.open_mfdataset(valid_files, engine="zarr", chunks="auto")
    
    # There are some artificially high values near border due to nan
    thresh = 5000
    ds["height"] = ds["height"].where(ds["height"] < thresh)
    ds_100m = ds.interp(x=x, y=y, method="linear")
    highest_point_CH = 4634
    ds_100m["height"] = ds_100m["height"].where(
        (ds_100m["height"] < highest_point_CH) & (ds_100m["height"] != 0) # Ensure 0 is written as nan
    )
    ds_100m.to_zarr(dem_grid_path)
    print('Saved to', dem_grid_path)


# Compute slope and aspect
ds_dem = xr.open_zarr(dem_grid_path)

dem = ds_dem['height']
slope = xdem.terrain.slope(dem.values, resolution=out_res) #degrees
aspect = xdem.terrain.aspect(dem) #degrees
ds_dem['slope'] = (('y','x'), slope)
ds_dem['aspect'] = (('y','x'), aspect)

ds_dem['slope'].rio.to_raster('slope_100m.tif')
ds_dem['aspect'].rio.to_raster('aspect_100m.tif')
ds_dem['height'].rio.to_raster('dem_100m.tif')
ds_dem.to_zarr(dem_grid_path, mode="a")
print('Added slope/aspect to', dem_grid_path)

