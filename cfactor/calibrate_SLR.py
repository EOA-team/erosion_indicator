import os
import numpy as np
import pandas as pd
import geopandas as gpd
from collections import Counter
import xarray as xr
import rioxarray
from joblib import dump, load, Parallel, delayed
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter("ignore")
import sys
sys.path.insert(0, os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure/SALI_models'))
from src.model_utils import compute_FC, compute_FC_grassland


def sample_locations(crop_labels, lnf_dir, tot_samples, save_path, seed=42):
    """Sample locations with a certain crop type, uniformly across crop types and available years (yearly crop maps) (EPSG:2056)"""

    lnf_files = sorted([f for f in os.listdir(lnf_dir) if f.endswith('.gpkg')])
    n_years = len(lnf_files)
    n_crops = len(top_lnfs_arable)
    samples_per_year = int(np.floor(tot_samples/n_years))
    samples_per_crop = int(np.floor(samples_per_year/n_crops))

    print(f'Sampling {tot_samples}: {samples_per_year} per year, {samples_per_crop} per crop in a year')
    all_samples = []
    for lnf_yr in lnf_files:
        yr = lnf_yr.split('lnf')[-1].split('.gpkg')[0]
        lnf = gpd.read_file(os.path.join(lnf_dir, lnf_yr))
        # Set all grasslands to same class
        lnf.loc[lnf['lnf_code'].isin(lnfs_grassland), 'lnf_code'] = lnfs_grassland[0]
        # Filter to keep only relevant polys
        lnf = lnf[lnf.lnf_code.isin(top_lnfs_arable)]
        lnf["poly_id"] = lnf.index
        if not len(lnf):
            continue
        seed += 1 # so that among differetn years it varies
        for crop in top_lnfs_arable:
            crop_polys = lnf[lnf.lnf_code == crop]

            sampled_polys = crop_polys.sample(
                n=samples_per_crop,
                weights=crop_polys.area,
                replace=True,
                random_state=seed
            )

            buffered = sampled_polys.copy()
            buffered["geometry"] = buffered.geometry.buffer(-10)
            pts = buffered.sample_points(1, random_state=seed)

            pts = pts.explode(index_parts=False).reset_index()
            pts = gpd.GeoDataFrame(pts, geometry='sampled_points', crs=crop_polys.crs)\
                .rename(columns={'sampled_points': "point_geom"})
            pts = pts.merge(
                crop_polys[["poly_id", "geometry", "lnf_code"]],
                left_on="index",
                right_index=True
            )
            pts["x"] = pts.point_geom.x
            pts["y"] = pts.point_geom.y
            pts["yr"] = yr

            all_samples.append(pts)

    samples = gpd.GeoDataFrame(pd.concat(all_samples, ignore_index=True), geometry="point_geom", crs=lnf.crs)
    samples.to_pickle(save_path)

    return


def sample_locations_with_field(crop_labels, lnf_dir, tot_samples, save_path, target_crs=32632, seed=42):
    """Sample locations with a certain crop type, uniformly across crop types and available years (yearly crop maps)"""

    lnf_files = sorted([f for f in os.listdir(lnf_dir) if f.endswith('.gpkg')])
    n_years = len(lnf_files)
    n_crops = len(top_lnfs_arable)
    samples_per_year = int(np.floor(tot_samples/n_years))
    samples_per_crop = int(np.floor(samples_per_year/n_crops))

    print(f'Sampling {tot_samples}: {samples_per_year} per year, {samples_per_crop} per crop in a year')
    all_samples = []
    for lnf_yr in lnf_files:
        yr = lnf_yr.split('lnf')[-1].split('.gpkg')[0]
        lnf = gpd.read_file(os.path.join(lnf_dir, lnf_yr))
        # Set all grasslands to same class
        lnf.loc[lnf['lnf_code'].isin(lnfs_grassland), 'lnf_code'] = lnfs_grassland[0]
        # Filter to keep only relevant polys
        lnf = lnf[lnf.lnf_code.isin(top_lnfs_arable)]
        lnf["poly_id"] = lnf.index
        if not len(lnf):
            continue
        seed += 1 # so that among differetn years it varies
        for crop in top_lnfs_arable:
            crop_polys = lnf[lnf.lnf_code == crop].to_crs(target_crs)

            sampled_polys = crop_polys.sample(
                n=samples_per_crop,
                weights=crop_polys.area,
                replace=False, 
                random_state=seed
            )

            buffered = sampled_polys.copy()
            buffered["geometry"] = buffered.geometry.buffer(-10)
            # Remove empty or invalid geometries
            buffered = buffered[~buffered.geometry.is_empty]
            buffered = buffered[buffered.geometry.notnull()]
            # Preserve index for joining
            buffered["orig_idx"] = buffered.index
            
            pts = buffered.sample_points(1, random_state=seed)
            pts = pts.explode(index_parts=False) # explode multipoint to point
            pts = gpd.GeoDataFrame(pts, geometry='sampled_points', crs=target_crs)\
                .rename(columns={'sampled_points': "point_geom"})
            # Attach attributes safely using index
            pts["orig_idx"] = pts.index
            pts = pts.merge(
                buffered[["orig_idx", "poly_id", "lnf_code", "geometry"]],
                on="orig_idx",
                how="left"
            )

            # Reproject to target CRS
            #pts = pts.set_geometry("geometry").to_crs(epsg=target_crs)
            pts = pts.rename(columns={"geometry": "polygon_geom"}).drop(columns="orig_idx")
    
            pts["x"] = pts.point_geom.x
            pts["y"] = pts.point_geom.y
            pts["yr"] = yr

            all_samples.append(pts)

    samples = gpd.GeoDataFrame(pd.concat(all_samples, ignore_index=True), geometry="point_geom", crs=target_crs)
    samples.to_pickle(save_path)

    return


def snap_to_grid(x, y, ds):

    res = abs(ds.rename({'lat':'y', 'lon':'x'}).rio.resolution()[0])

    x0 = ds.lon.values[0]
    y0 = ds.lat.values[0]

    snapped_x = x0 + np.round((x - x0) / res) * res
    snapped_y = y0 + np.round((y - y0) / res) * res

    return snapped_x, snapped_y


def extract_s2_data(save_path_sampledloc, s2_grid_path, s2_dir, soil_dir, save_path_sampledloc_S2):
    print('Extract S2 (30 Jun previous yr to 1 Jul current yr) and Soil data')
    samples = pd.read_pickle(save_path_sampledloc).to_crs(32632)

    # Intersect S2 grid with locations of samples to get list of files and coords concerned
    grid = gpd.read_file(s2_grid_path)
    grid_samples = gpd.overlay(samples, grid, how='intersection') # keeps sample point geometry and S2 name
    s2_files = [f for f in os.listdir(s2_dir)]
    
    soil_cache = {}
    df_samples = []
    for (left, top, yr), pts_df in grid_samples.groupby(['left', 'top', 'yr']):
        #print(f'Extracting for {yr}: {left}-{top}')
        # Extract S2 data
        file_str1 = f"S2_{int(left)}_{int(top)}_{yr}"
        file_str2 = f"S2_{int(left)}_{int(top)}_{int(yr)-1}"
        f = [os.path.join(s2_dir, f) for f in s2_files if f.startswith((file_str1, file_str2))]
        ds_s2 = xr.open_mfdataset(f).sel(time=slice(f'{int(yr)-1}-06-30', f'{yr}-07-01'))
        ds_s2_sampled = ds_s2.sel(
            lon=xr.DataArray(pts_df["x"].values, dims="points"),
            lat=xr.DataArray(pts_df["y"].values, dims="points"),
            method="nearest"
        )
        ds_s2_sampled = ds_s2_sampled.assign_coords(
            sampled_x=("points", pts_df["x"].values),
            sampled_y=("points", pts_df["y"].values),
            lnf_code=("points", pts_df["lnf_code"].values)
        )
        df_s2_sampled = ds_s2_sampled.to_dataframe().reset_index()
        df_s2_sampled["yr"] = yr

        # Extract soil data: if sampled coord is not labeled, look at mode in 10x10 pixels around
        key = (left, top)
        if key not in soil_cache:
            soil_file = os.path.join(soil_dir, f'SRC_{int(left)}_{int(top)}.zarr')
            if not os.path.exists(soil_file):
                soil_group = 0
            else:
                ds_soil = xr.open_zarr(os.path.join(soil_dir, f'SRC_{int(left)}_{int(top)}.zarr'))
                ds_soil_sampled = ds_soil.sel(
                    x=xr.DataArray(pts_df["x"].values, dims="points"),
                    y=xr.DataArray(pts_df["y"].values, dims="points"),
                    method="nearest"
                )
                soil_group = ds_soil_sampled.soil_group.values[0]
                
                if int(soil_group) == -10000:
                    offsets = np.arange(-5, 5) * 10
                    dx, dy = np.meshgrid(offsets, offsets)
                    dx = dx.ravel()
                    dy = dy.ravel()
                    expanded = pts_df.loc[pts_df.index.repeat(len(dx))].copy()
                    expanded["x"] = expanded["x"].values + np.tile(dx, len(pts_df))
                    expanded["y"] = expanded["y"].values + np.tile(dy, len(pts_df))
                    
                    ds_soil_around = ds_soil.sel(
                        x=xr.DataArray(expanded["x"].values, dims="points"),
                        y=xr.DataArray(expanded["y"].values, dims="points"),
                        method="nearest"
                    )
                    soil_around = ds_soil_around.soil_group.values
                    soil_around = [s for s in soil_around if int(s) != -10000]
                    if len(soil_around):
                        soil_group = Counter(soil_around).most_common(1)[0][0]
                    else:
                        soil_group = 0
                
                # Save for later
                soil_cache[key] = soil_group 
        else:
            soil_group = soil_cache[key]

        df_s2_sampled['soil_group'] = soil_group
        df_samples.append(df_s2_sampled)

    df_samples = pd.concat(df_samples, ignore_index=True)
    df_samples.to_pickle(save_path_sampledloc_S2)

    return


def extract_s2_data_field(save_path_sampledloc, s2_grid_path, s2_dir, soil_dir, save_path_sampledloc_S2):
    print('Extract S2 (1 Jun previous yr to 30 Jul current yr) and Soil data')
    samples = pd.read_pickle(save_path_sampledloc)

    # Intersect S2 grid with locations of samples to get list of files and coords concerned
    grid = gpd.read_file(s2_grid_path)
    grid_samples = gpd.overlay(samples, grid, how='intersection') # keeps sample point geometry and S2 name
    s2_files = [f for f in os.listdir(s2_dir)]
    # Prepare for searching S2 files
    file_lookup = {}
    for fname in s2_files:
        key = "_".join(fname.split("_")[:4])[:-4]  # S2_left_top_year
        file_lookup.setdefault(key, []).append(os.path.join(s2_dir, fname))
   
    df_samples = []
    # Iterate through fields and find for each the data
    for (poly_id, yr), poly_df in grid_samples.groupby(['poly_id', 'yr']):
        polygon = poly_df.iloc[0]['polygon_geom']
        tiles = poly_df[['left', 'top']].drop_duplicates()
        tiles = [(row['left'], row['top']) for _, row in tiles.iterrows()]
        years = [int(yr), int(yr) - 1]
        matched_files = []
        for (left, top) in tiles:
            for y in years:
                key = f"S2_{int(left)}_{int(top)}_{y}"
                matched_files.extend(file_lookup.get(key, []))
        if not len(matched_files):
            print('No files found for polygon')
            continue

        # Extract S2 data
        ds_s2 = xr.open_mfdataset(matched_files).sel(time=slice(f'{int(yr)-1}-06-01', f'{yr}-07-30'))
        try:
            mask = ds_s2.rename({'lat':'y', 'lon':'x'}).rio.write_crs(32632).rio.clip([polygon], ds_s2.rio.crs, drop=True)  # Clip S2 data to the polygon
        except:
            continue
        df_s2_sampled = mask.to_dataframe().reset_index()
        # Drop masked pixels (all s2 bands 0)
        s2_cols = [c for c in df_s2_sampled.columns if c.startswith("s2_")]
        df_s2_sampled = df_s2_sampled.loc[(df_s2_sampled[s2_cols] != 0).any(axis=1)]
        df_s2_sampled["lnf_code"] = poly_df.iloc[0]["lnf_code"]
        df_s2_sampled["yr"] = yr
        df_s2_sampled["poly_id"] = poly_df.iloc[0]["poly_id"]

        # Extract soil data for the polygon
        soil_files = [os.path.join(soil_dir, f'SRC_{int(left)}_{int(top)}.zarr') for left, right in tiles]
        if not len(soil_files):
            soil_group = 0
        else:
            ds_soil = xr.open_mfdataset(soil_files).astype("float32")
            try:
                ds_soil_clipped = ds_soil.rio.write_crs(32632).rio.clip([polygon], ds_soil.rio.crs, drop=True)
            except:
                continue 
            soil_values = ds_soil_clipped["soil_group"].values.flatten()
            soil_values = soil_values[(soil_values != -10000) & (~np.isnan(soil_values))] # Remove invalid values
            if len(soil_values):
                soil_group = Counter(soil_values).most_common(1)[0][0]
            else:
                soil_group = 0

        # Find the row corresponding to sampled point in the polygon and mark it
        sampled_x = poly_df.iloc[0]["x"]
        sampled_y = poly_df.iloc[0]["y"]
        snap_x, snap_y = snap_to_grid(sampled_x, sampled_y, ds_s2)
        df_s2_sampled["is_sample_pixel"] = (
            (df_s2_sampled["x"] == snap_x) &
            (df_s2_sampled["y"] == snap_y)
        )
        df_s2_sampled["sampled_x"] = sampled_x
        df_s2_sampled["sampled_y"] = sampled_y
        df_s2_sampled['x'] = df_s2_sampled['x'].apply(lambda x: x-5)
        df_s2_sampled['y'] = df_s2_sampled['y'].apply(lambda y: y+5)
            
        df_s2_sampled['soil_group'] = soil_group
        df_samples.append(df_s2_sampled)

    df_samples = pd.concat(df_samples, ignore_index=True)
    df_samples.to_pickle(save_path_sampledloc_S2)

    return


def predict_FC(save_path_sampledloc_S2, save_path_FCpreds):
    print('Predict FC')

    bands = ['s2_B02','s2_B03','s2_B04','s2_B05','s2_B06','s2_B07','s2_B08','s2_B8A','s2_B11','s2_B12']
    df_samples = pd.read_pickle(save_path_sampledloc_S2)

    # Prepare data
    df_samples[df_samples==65535] = np.nan
    df_samples = df_samples.dropna()
    df_samples[bands] /= 10000

    # Predict using correct model
    df_global = df_samples[df_samples.soil_group==0].copy()
    df_soil   = df_samples[df_samples.soil_group!=0].copy()

    data_global = df_samples[df_samples.soil_group==0][bands].to_numpy()
    data_soil = df_samples[df_samples.soil_group!=0][bands].to_numpy()

    all_preds_global = []
    all_preds_soil =  []
    chunk_size = 10000
    for i in range(0, len(data_global), chunk_size):
        chunk = data_global[i:i+chunk_size]
        preds = compute_FC_grassland(chunk)   # pv, npv, soil
        all_preds_global.append(preds)
    for i in range(0, len(data_soil), chunk_size):
        chunk = data_soil[i:i+chunk_size]
        preds = compute_FC(chunk)   # pv, npv, soil
        all_preds_soil.append(preds)

    # Combine results
    preds_global = np.vstack(all_preds_global)
    preds_soil = np.vstack(all_preds_soil)

    # Assing predictions
    df_global[['pv', 'npv', 'soil']] = preds_global

    preds_soil = pd.DataFrame(preds_soil, columns=['soil_group', 'pv', 'npv', 'soil'])
    df_soil = df_soil.copy()
    df_soil['row_idx'] = np.arange(len(df_soil))   
    preds_soil['row_idx'] = np.tile(np.arange(len(df_soil)), 5)
    df_soil = pd.merge(df_soil, preds_soil, on=['row_idx', 'soil_group'])

    df_samples = pd.concat([df_global, df_soil]).sort_index().drop(columns=['row_idx', 'soil_group'])
    df_samples.to_pickle(save_path_FCpreds)

    return


def clean_timeseries_df(group, cirrus_thresh=1000):

    band_cols = [c for c in group.columns if c.startswith('s2_B')]

    # Missing data
    missing_mask = (group[band_cols].isna()).all(axis=1)

    # Clouds
    cloud_mask = (
        (group['s2_mask'] == 1) |
        (group['s2_SCL'].isin([8, 9, 10]))
    )

    # Shadows
    shadow_mask = (
        (group['s2_mask'] == 2) |
        (group['s2_SCL'] == 3)
    )

    # Snow
    snow_mask = (
        (group['s2_mask'] == 3) |
        (group['s2_SCL'] == 11)
    )

    # Cirrus
    cirrus_mask = (
        (group['s2_SCL'] == 10) &
        (group['s2_B02'] > cirrus_thresh)
    )

    drop_mask = (
        missing_mask |
        cloud_mask |
        shadow_mask |
        snow_mask |
        cirrus_mask
    )

    return group.loc[~drop_mask]


def clean_timeseries_field(field_df, cirrus_thresh=500, max_missing_frac=0.05):
    """
    field_df: all pixels in one field for a given year (or grouping)
    max_missing_frac: drop dates where more than this fraction of pixels are masked
    """
    band_cols = [c for c in field_df.columns if c.startswith('s2_B')]

    # Create masks for all pixels
    missing_mask = (field_df[band_cols].isna()).all(axis=1) 
    missing_mask2 = (field_df[band_cols] == 65535).all(axis=1) 
    cloud_mask = (field_df['s2_mask'] == 1) | (field_df['s2_SCL'].isin([8, 9, 10]))
    shadow_mask = (field_df['s2_mask'] == 2) | (field_df['s2_SCL'] == 3)
    snow_mask = (field_df['s2_mask'] == 3) | (field_df['s2_SCL'] == 11)
    cirrus_mask = (field_df['s2_SCL'] == 10) & (field_df['s2_B02'] > cirrus_thresh)

    # Combine masks
    drop_mask = missing_mask | missing_mask2 | cloud_mask | shadow_mask | snow_mask | cirrus_mask
    field_df['masked'] = drop_mask.astype(int)

    # Compute fraction of masked pixels per date
    masked_frac_per_date = field_df.groupby('time')['masked'].mean()

    # Keep only dates where fraction of masked pixels <= threshold
    valid_dates = masked_frac_per_date[masked_frac_per_date <= max_missing_frac].index

    return field_df[field_df['time'].isin(valid_dates)].drop(columns='masked')


def gpr_fit_predict(doy_train, y_train, doy_pred, kernel, alpha):
    
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        optimizer=None,
        normalize_y=True
    )

    gp.fit(doy_train, y_train)

    mean, std = gp.predict(doy_pred, return_std=True)

    return mean, std


def train_gpr_kernel(df, ts_cols, value_col, n_samples=200, random_state=42,
                     kernel=None, alpha=1e-4):

    if kernel is None:
        kernel = RBF(length_scale=10) + WhiteKernel(noise_level=alpha)

    # select subset of timeseries
    unique_ts = df[ts_cols].drop_duplicates()
    train_ts = unique_ts.sample(min(n_samples, len(unique_ts)),
                                random_state=random_state)

    train_df = df.merge(train_ts, on=ts_cols)

    dates = pd.to_datetime(train_df['time'])
    doy = dates.dt.dayofyear.values.reshape(-1,1)

    y = train_df[value_col].values

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        normalize_y=True,
        n_restarts_optimizer=10
    )

    gp.fit(doy, y)

    print("Optimized kernel:", gp.kernel_)

    return gp.kernel_


def apply_gpr_timeseries(group_clean, group_raw, kernel, ts_cols, alpha=1e-6):

    dates_train = pd.to_datetime(group_clean["time"])
    doy_train = dates_train.dt.dayofyear.values.reshape(-1,1)

    dates_pred = pd.to_datetime(group_raw["time"])
    doy_pred = dates_pred.dt.dayofyear.values.reshape(-1,1)

    pv_mean, pv_std = gpr_fit_predict(
        doy_train, group_clean["pv"].values, doy_pred, kernel, alpha
    )
    npv_mean, npv_std = gpr_fit_predict(
        doy_train, group_clean["npv"].values, doy_pred, kernel, alpha
    )
    soil_mean, soil_std = gpr_fit_predict(
        doy_train, group_clean["soil"].values, doy_pred, kernel, alpha
    )

    total = pv_mean + npv_mean + soil_mean
    pv = pv_mean / total
    npv = npv_mean / total
    soil = soil_mean / total

    result = group_raw[ts_cols + ["time"]].copy()
    result["pv"] = pv
    result["npv"] = npv
    result["soil"] = soil

    return result


def gapfill_dataframe_gpr(df_raw, df_clean, kernel, ts_cols):

    results = []
    for keys, group_raw in df_raw.groupby(ts_cols):

        group_clean = df_clean[
            (df_clean[ts_cols] == pd.Series(keys, index=ts_cols)).all(axis=1)
        ]

        if len(group_clean) < 3:
            continue

        res = apply_gpr_timeseries(
            group_clean,
            group_raw,
            kernel,
            ts_cols
        )

        results.append(res)

    return pd.concat(results, ignore_index=True)


def alr_transform(fc, ref_idx=-1):
    """Additive log-ratio transform. fc: (n, 3) array of [pv, npv, soil]."""
    fc = np.clip(fc, 1e-6, 1 - 1e-6)  # avoid log(0)
    ref = fc[:, ref_idx:ref_idx+1] if ref_idx != -1 else fc[:, -1:]
    log_ratios = np.log(np.delete(fc, ref_idx if ref_idx != -1 else fc.shape[1]-1, axis=1) / ref)
    return log_ratios  # shape: (n, 2)


def alr_inverse(log_ratios, ref_idx=-1):
    """Inverse ALR: back to simplex."""
    exp_ratios = np.exp(log_ratios)  # shape: (n, 2)
    denom = 1 + exp_ratios.sum(axis=1, keepdims=True)
    parts = exp_ratios / denom
    ref = 1.0 / denom
    if ref_idx == -1:
        return np.hstack([parts, ref])  # [pv, npv, soil]
    else:
        result = np.insert(parts, ref_idx, ref.ravel(), axis=1)
        return result


def generate_fill_dates(clean_dates: pd.DatetimeIndex, max_gap_days: int = 15) -> pd.DatetimeIndex:
    """Return synthetic dates to insert wherever consecutive clean observations are more than max_gap_days apart."""
    dates = sorted(pd.to_datetime(clean_dates).unique())
    fill = []
    for d0, d1 in zip(dates[:-1], dates[1:]):
        gap = (d1 - d0).days
        if gap > max_gap_days:
            current = d0 + pd.Timedelta(days=max_gap_days)
            while current < d1:
                fill.append(current)
                current += pd.Timedelta(days=max_gap_days)
    return pd.DatetimeIndex(fill)


def plot_field_timeseries(keys, group_raw, group_clean, df_gapfilled, pixel_cols, value_col="pv", n_ts=5):

    # Select unique pixels
    unique_pixels = (
        df_gapfilled[pixel_cols]
        .drop_duplicates()
    )

    sampled_pixels = unique_pixels.sample(
        n=min(n_ts, len(unique_pixels)),
        random_state=42
    )

    ncols = n_ts
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(4 * ncols, 4),
        sharey=True
    )

    if n_ts == 1:
        axes = [axes]

    for i, (_, pixel_row) in enumerate(sampled_pixels.iterrows()):

        ax = axes[i]

        # Masks
        mask_raw = (
            group_raw[pixel_cols]
            == pixel_row[pixel_cols]
        ).all(axis=1)

        mask_clean = (
            group_clean[pixel_cols]
            == pixel_row[pixel_cols]
        ).all(axis=1)

        mask_gap = (
            df_gapfilled[pixel_cols]
            == pixel_row[pixel_cols]
        ).all(axis=1)

        df_raw_ts = group_raw[mask_raw].sort_values("time")
        df_clean_ts = group_clean[mask_clean].sort_values("time")
        df_gap_ts = df_gapfilled[mask_gap].sort_values("time")

        # Plot
        ax.plot(
            df_raw_ts["time"],
            df_raw_ts[value_col],
            "o-",
            alpha=0.4,
            label="Raw"
        )

        ax.plot(
            df_clean_ts["time"],
            df_clean_ts[value_col],
            "s-",
            alpha=0.6,
            label="Cleaned"
        )

        ax.plot(
            df_gap_ts["time"],
            df_gap_ts[value_col],
            "x--",
            alpha=0.9,
            label="Gapfilled"
        )

        ax.set_title(
            f"{pixel_row[pixel_cols].to_dict()}",
            fontsize=8
        )

        ax.tick_params(axis="x", rotation=45)

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper right"
    )

    fig.suptitle(
        f"Field {keys} — Sampled Timeseries"
    )

    plt.tight_layout()

    plt.savefig(
        f"field_{keys}_timeseries.png",
        dpi=150
    )

    return


def _gapfill_one_field_alr(keys, group_raw, group_clean, ts_cols, value_cols, pixel_cols, alpha, max_train_points=1000):
    """Process a single field — designed to be called in parallel."""
    if len(group_clean) < 3:
        return None

    # Prepare output dataframe
    res = group_raw[ts_cols + pixel_cols + ["time", "is_sample_pixel"]].copy()

    # Replace doy with continuous (cyclical) time
    dates_pred = pd.to_datetime(group_raw["time"])
    dates_train = pd.to_datetime(group_clean["time"])
    t0 = dates_train.min()
    t_pred = (dates_pred - t0).dt.days.values.reshape(-1, 1)
    t_train = (dates_train - t0).dt.days.values.reshape(-1, 1)

    # Add seasonal phase
    t_year_train = t_train % 365
    t_year_pred  = t_pred % 365
    sin_train = np.sin(2 * np.pi * t_year_train / 365)
    cos_train = np.cos(2 * np.pi * t_year_train / 365)
    sin_pred = np.sin(2 * np.pi * t_year_pred / 365)
    cos_pred = np.cos(2 * np.pi * t_year_pred / 365)

    # Normalise time
    t_train = t_train / 365.0
    t_pred  = t_pred  / 365.0
    
    # Spatial features
    p_scaler    = StandardScaler()
    pixel_train = p_scaler.fit_transform(group_clean[pixel_cols].values)
    pixel_pred  = p_scaler.transform(group_raw[pixel_cols].values)
    
    # --- final features ---
    n_time_dims  = 3                    # t, sin, cos
    n_pixel_dims = pixel_train.shape[1]
    n_features   = n_time_dims + n_pixel_dims
 
    X_train_full = np.hstack([t_train, sin_train, cos_train, pixel_train]).astype(np.float32)
    X_pred = np.hstack([t_pred,  sin_pred,  cos_pred,  pixel_pred ]).astype(np.float32)

    # --- ALR transform: fit GPR in unconstrained space ---
    fc_train_full = group_clean[value_cols].values.astype(np.float32)
    alr_train_full = alr_transform(fc_train_full)   # (n, 2)
 
    """
    # Subsample if needed
    if len(group_clean) > max_train_points:
        sample_pixel = (
            group_clean[group_clean["is_sample_pixel"]][pixel_cols].drop_duplicates()
        )
        other_pixels = (
            group_clean[~group_clean["is_sample_pixel"]][pixel_cols].drop_duplicates()
        )

        n_sample_pixels = len(sample_pixel)
        n_other_pixels = len(other_pixels)
        print("Total pixels in field:", n_sample_pixels+n_other_pixels)

        # Average points per pixel
        total_unique_pixels = (
            group_clean[pixel_cols]
            .drop_duplicates()
            .shape[0]
        )
        total_unique_pixels = len(group_clean[pixel_cols].drop_duplicates())
        avg_points_per_pixel = len(group_clean) / total_unique_pixels
        max_pixels_allowed = max(1, int(max_train_points / avg_points_per_pixel))
        remaining_pixels = max(0, max_pixels_allowed - n_sample_pixels)

        # --- Step 3: sample additional non-sample pixels ---
        if remaining_pixels > 0 and n_other_pixels > 0:
            selected_other_pixels = other_pixels.sample(
                n=min(remaining_pixels, n_other_pixels),
                random_state=42
            )
            selected_pixels = pd.concat(
                [sample_pixel, selected_other_pixels],
                ignore_index=True
            )
        else:
            selected_pixels = sample_pixel.copy()
        print("Total pixels after subsampling:", len(selected_pixels))

        # --- Step 4: build subsample mask ---
        merged = group_clean[pixel_cols].merge(
            selected_pixels,
            on=pixel_cols,
            how='left',
            indicator=True
        )
        subsample_idx = (merged["_merge"] == "both").values
        X_train = X_train_full[subsample_idx]
        y_train_mask = subsample_idx
    else:
        X_train = X_train_full
        y_train_mask = np.ones(len(group_clean), dtype=bool)
    """

    # -------------------------------------------------------------------------
    # Subsmaple data to fit in GPR
    # stratify in time, so that new data contributes new information (underepresented timestamps)
    # ensuring the target pixel's full time series is always present
    # then fills the budget with other-pixel observations spread evenly across time.
    # -------------------------------------------------------------------------
    target_mask = group_clean["is_sample_pixel"].values.astype(bool)
 
    if len(group_clean) > max_train_points:
        n_target = target_mask.sum()
        budget   = max(0, max_train_points - n_target)
 
        other_idx = np.where(~target_mask)[0]
 
        if budget > 0 and len(other_idx) > 0:
            # Stratified: pick other pixels so that their time distribution
            # mirrors the full dataset (preserves temporal coverage).
            other_times   = t_train[other_idx, 0]
            n_bins        = min(10, budget)
            bin_edges     = np.linspace(other_times.min(), other_times.max(), n_bins + 1)
            bin_ids       = np.digitize(other_times, bin_edges) - 1
            chosen_other  = []
            per_bin       = max(1, budget // n_bins)
            rng           = np.random.default_rng(42)
            for b in range(n_bins):
                in_bin = other_idx[bin_ids == b]
                if len(in_bin) > 0:
                    chosen_other.append(
                        rng.choice(in_bin, size=min(per_bin, len(in_bin)), replace=False)
                    )
            chosen_other = np.concatenate(chosen_other) if chosen_other else np.array([], dtype=int)
 
            keep = np.concatenate([np.where(target_mask)[0], chosen_other])
        else:
            keep = np.where(target_mask)[0]
 
        X_train    = X_train_full[keep]
        alr_train  = alr_train_full[keep]
        print(f"Subsampled: {len(keep)} pts ({target_mask.sum()} target, {len(keep)-target_mask.sum()} other)")
    else:
        X_train   = X_train_full
        alr_train = alr_train_full
 
    # -------------------------------------------------------------------------
    # Per-field noise estimation for alpha
    # Use ~5% of the ALR signal variance as nugget — adapts to field variability
    # instead of a fixed global constant.
    # -------------------------------------------------------------------------
    """
    target_alr = alr_train_full[target_mask]   # target pixel rows only
    target_t   = X_train_full[target_mask, 0]  # time column

    # Linear detrend per ALR component, take residual std
    residual_vars = []
    for j in range(2):
        coeffs   = np.polyfit(target_t, target_alr[:, j], deg=2)
        trend    = np.polyval(coeffs, target_t)
        residual_vars.append(np.var(target_alr[:, j] - trend))

    estimated_alpha = float(np.mean(residual_vars)) * 0.5  # 50% of residual variance
    estimated_alpha = float(np.clip(estimated_alpha, 1e-4, 0.5))
    """
    estimated_alpha = float(np.var(alr_train, axis=0).mean()) * 0.05
    estimated_alpha = float(np.clip(estimated_alpha, 1e-4, 0.5))

    ls_init   = [1.0] * n_features
    ls_bounds = (
        [(0.1,  5.0)]  * 1              +   # t: absolute time
        [(0.5, 10.0)]  * 2              +   # sin, cos: seasonal period
        [(0.05, 10.0)] * n_pixel_dims       # pixel / spatial dims
    )
    base_kernel = ConstantKernel(1.0, (0.1, 5.0)) * Matern(
        length_scale=ls_init,
        length_scale_bounds=ls_bounds,
        nu=1.5
    ) + WhiteKernel(
        noise_level=estimated_alpha,
        noise_level_bounds=(1e-5, 0.5)
    )
 
    alr_preds = np.zeros((len(X_pred), 2))
    alr_unc = np.zeros((len(X_pred), 2))
    for j in range(2):
        y_train = alr_train[:, j]
 
        if j == 0:
            # Full hyperparameter optimisation for first component
            gp = GaussianProcessRegressor(
                kernel=base_kernel,
                alpha=estimated_alpha,
                normalize_y=True,
                n_restarts_optimizer=5      # increased from 3
            )
            gp.fit(X_train, y_train)
            fitted_kernel = gp.kernel_      # save for reuse
        else:
            # Reuse optimized kernel from component 0 — fewer restarts needed
            gp = GaussianProcessRegressor(
                kernel=fitted_kernel,
                alpha=estimated_alpha,
                normalize_y=True,
                n_restarts_optimizer=2
            )
            gp.fit(X_train, y_train)
 
        mean, std = gp.predict(X_pred, return_std=True)
        alr_preds[:, j] = mean
        alr_unc[:, j]   = std
 

    # Back-transform to simplex — guaranteed to sum to 1 and be positive
    fc_pred = alr_inverse(alr_preds)  # (n, 3)
    for i, col in enumerate(value_cols):
        res[col] = fc_pred[:, i]

    # Plot some pixels to check
    df_gapfilled = res.copy()
    """
    plot_field_timeseries(
        keys, group_raw, group_clean, df_gapfilled,
        pixel_cols, value_col="pv", n_ts=5
    )
    """

    # Add uncertainty as quality indicator (still in ALR space)
    total_unc = np.linalg.norm(alr_unc, axis=1) # normalise per field
    p80 = np.percentile(total_unc, 80)
    p95 = np.percentile(total_unc, 95)
    qflag = np.zeros(len(total_unc), dtype=int)
    qflag[total_unc > p80] = 1   # medium
    qflag[total_unc > p95] = 2   # low confidence
    res["fc_quality_flag"] = qflag  # 0: high conf, 1: medium conf, 2: low conf

    # Return only the sampled pixel (not the whole field)
    res = res[res["is_sample_pixel"]].copy()
    res = res.drop(columns="is_sample_pixel")
 
    return res


def _gapfill_one_field_alr_median(keys, group_raw, group_clean, ts_cols, value_cols, pixel_cols, alpha, max_train_points=1000):
    """Process a single field — designed to be called in parallel."""
    if len(group_clean) < 3:
        return None

    # Prepare output dataframe
    res = group_raw[ts_cols + pixel_cols + ["time", "is_sample_pixel"]].copy()

    # ---- Step 1: Build field-median training signal per date ----
    field_median = (
        group_clean
        .groupby("time")[value_cols]
        .median()
        .reset_index()
        .sort_values("time")
    )

    if len(field_median) < 3:
        return None

    # ---- Step 2: Time features only for the GP ----
    dates_train = pd.to_datetime(field_median["time"])
    dates_pred = pd.to_datetime(group_raw["time"])
    t0 = dates_train.min()

    t_train = (dates_train - t0).dt.days.values.reshape(-1, 1).astype(np.float64)
    t_pred = (dates_pred - t0).dt.days.values.reshape(-1, 1).astype(np.float64)

    # Normalize time
    t_scaler = StandardScaler()
    t_train_scaled = t_scaler.fit_transform(t_train)
    t_pred_scaled = t_scaler.transform(t_pred)

    # ---- Step 3: ALR transform on field medians ----
    fc_train = field_median[value_cols].values.astype(np.float64)
    alr_train = alr_transform(fc_train)  # (n_dates, 2)

    # ---- Step 4: Fit one GP per ALR component (time-only) ----
    alr_preds_by_date = np.zeros((len(t_pred_scaled), 2))
    alr_unc = np.zeros((len(t_pred_scaled), 2))

    for j in range(2):
        y_train = alr_train[:, j]

        kernel = ConstantKernel(1.0, (0.1, 5.0)) * RBF(
            length_scale=1.0,
            length_scale_bounds=(0.3, 5.0)
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            normalize_y=True,
            n_restarts_optimizer=5
        )
        gp.fit(t_train_scaled, y_train)
        mean, std = gp.predict(t_pred_scaled, return_std=True)

        alr_preds_by_date[:, j] = mean
        alr_unc[:, j] = std

    # ---- Step 5: Per-pixel residual correction ----
    # Where we have clean observations for a pixel, compute offset from field median
    # and apply it to the GP prediction. Where we don't, use GP prediction as-is.

    # Map each prediction row to its date-level GP prediction
    # (multiple pixels on same date share the same GP output)
    pred_date_to_idx = {d: i for i, d in enumerate(dates_pred)}

    # Compute ALR of clean per-pixel observations
    clean_fc = group_clean[value_cols].values.astype(np.float64)
    clean_alr = alr_transform(clean_fc)  # (n_clean, 2)

    # Compute ALR of field median at each clean date
    clean_dates = pd.to_datetime(group_clean["time"])
    median_lookup = field_median.set_index("time")[value_cols]

    # For each clean observation, get the median ALR at that date
    clean_median_fc = median_lookup.loc[clean_dates.values].values.astype(np.float64)
    clean_median_alr = alr_transform(clean_median_fc)

    # Residual = pixel ALR - median ALR
    residuals = clean_alr - clean_median_alr  # (n_clean, 2)

    # Compute per-pixel mean residual (bias correction)
    group_clean_copy = group_clean.copy()
    for j in range(2):
        group_clean_copy[f"_resid_{j}"] = residuals[:, j]

    pixel_offsets = (
        group_clean_copy
        .groupby(pixel_cols)[[f"_resid_{j}" for j in range(2)]]
        .mean()
    )

    # ---- Step 6: Apply GP prediction + pixel offset to all prediction rows ----
    alr_preds = np.zeros((len(res), 2))
    for i in range(len(res)):
        # Base GP prediction for this date
        alr_preds[i] = alr_preds_by_date[i]

        # Look up pixel offset
        pixel_key = tuple(res.iloc[i][pixel_cols].values)
        if pixel_key in pixel_offsets.index:
            offset = pixel_offsets.loc[pixel_key].values
            alr_preds[i] += offset

    # ---- Step 7: Clamp ALR to prevent extreme predictions ----
    for j in range(2):
        alr_min = alr_train[:, j].min()
        alr_max = alr_train[:, j].max()
        margin = 0.5 * (alr_max - alr_min)
        alr_preds[:, j] = np.clip(alr_preds[:, j], alr_min - margin, alr_max + margin)

    # ---- Step 8: Back-transform to simplex ----
    fc_pred = alr_inverse(alr_preds)
    for i, col in enumerate(value_cols):
        res[col] = fc_pred[:, i]

    # Plot some pixels to check
    df_gapfilled = res.copy()
    plot_field_timeseries(
        keys, group_raw, group_clean, df_gapfilled,
        pixel_cols, value_col="pv", n_ts=5
    )

    # Quality flag
    total_unc = np.linalg.norm(alr_unc, axis=1)
    # Map date-level uncertainty to pixel-level rows
    pred_dates_list = dates_pred.values
    date_unc_map = dict(zip(pred_dates_list, total_unc))
    pixel_unc = np.array([
        date_unc_map.get(pd.to_datetime(res.iloc[i]["time"]), 0.0)
        for i in range(len(res))
    ])
    p80 = np.percentile(pixel_unc, 80)
    p95 = np.percentile(pixel_unc, 95)
    qflag = np.zeros(len(pixel_unc), dtype=int)
    qflag[pixel_unc > p80] = 1
    qflag[pixel_unc > p95] = 2
    res["fc_quality_flag"] = qflag

    # Return only the sampled pixel
    res = res[res["is_sample_pixel"]].copy()
    res = res.drop(columns="is_sample_pixel")

    return res


def _gapfill_one_field(keys, group_raw, group_clean, ts_cols, value_cols, pixel_cols, alpha, max_train_points=3000):
    """Process a single field — designed to be called in parallel."""
    if len(group_clean) < 3:
        return None

    res = group_raw[ts_cols + pixel_cols + ["time"]].copy()

    dates_pred = pd.to_datetime(group_raw["time"])
    doy_pred = dates_pred.dt.dayofyear.values.reshape(-1, 1)
    pixel_pred = group_raw[pixel_cols].values
    X_pred = np.hstack([doy_pred, pixel_pred])

    dates_train = pd.to_datetime(group_clean["time"])
    doy_train = dates_train.dt.dayofyear.values.reshape(-1, 1)
    pixel_train = group_clean[pixel_cols].values
    X_train_full = np.hstack([doy_train, pixel_train])

    # Normalize features so DOY and coordinates are on the same scale
    scaler = StandardScaler()
    X_train_full = scaler.fit_transform(X_train_full)
    X_pred = scaler.transform(X_pred)
    
    # Subsample complete pixel timeseries if too many training points
    if len(group_clean) > max_train_points:
        # Get unique pixels
        unique_pixels = group_clean[pixel_cols].drop_duplicates()
        n_pixels = len(unique_pixels)
        avg_points_per_pixel = len(group_clean) / n_pixels
        n_pixels_to_keep = max(1, int(max_train_points / avg_points_per_pixel))

        # Randomly select complete pixel timeseries
        selected_pixels = unique_pixels.sample(
            n=min(n_pixels_to_keep, n_pixels),
            random_state=42
        )
        # Use merge with indicator to build a positional boolean mask
        merged = group_clean[pixel_cols].reset_index(drop=True).merge(
            selected_pixels, on=pixel_cols, how='left', indicator=True
        )
        subsample_idx = (merged['_merge'] == 'both').values  # numpy bool array, positional
        X_train = X_train_full[subsample_idx]
        y_train_mask = subsample_idx
    else:
        X_train = X_train_full
        y_train_mask = np.ones(len(group_clean), dtype=bool)
        
    for value_col in value_cols:
        y_train = group_clean[value_col].values[y_train_mask]
        
        # Estimate variability for kernel
        variability = np.std(y_train)
        length_scale = 20 if variability > 0.2 else 80

        kernel = RBF(length_scale=length_scale) + WhiteKernel(noise_level=alpha)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True, optimizer=None)
        gp.fit(X_train, y_train)
        

        mean, _ = gp.predict(X_pred, return_std=True)
        res[value_col] = mean
    
    return res


def gapfill_dataframe_gpr_per_field_parallel(df_raw, df_clean, ts_cols, value_cols, pixel_cols, alpha=1e-2, n_jobs=1):
    
    # Pre-group both DataFrames (avoids repeated full-DataFrame scans)
    raw_groups = {keys: group for keys, group in df_raw.groupby(ts_cols)}
    clean_groups = {keys: group for keys, group in df_clean.groupby(ts_cols)}
    
    # Build task list: only fields that exist in both raw and clean
    tasks = []
    for keys, group_raw in raw_groups.items():
        group_clean = clean_groups.get(keys, pd.DataFrame())
        tasks.append((keys, group_raw, group_clean))

    print(f"To process: {len(tasks)} fields using {n_jobs} workers")

    # Run in parallel
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_gapfill_one_field_alr)(
            keys, group_raw, group_clean,
            ts_cols, value_cols, pixel_cols, alpha
        )
        for keys, group_raw, group_clean in tasks
    )

    # Filter None results and concatenate
    results = [r for r in results if r is not None]

    return pd.concat(results, ignore_index=True)


def gapfill_dataframe_gpr_per_field(df_raw, df_clean, ts_cols, value_cols, pixel_cols, alpha=1e-2):
    results = []
    print('To process:', len(df_raw.drop_duplicates(ts_cols)))
    for keys, group_raw in df_raw.groupby(ts_cols):
        print(keys)
        group_clean = df_clean[
            (df_clean[ts_cols] == pd.Series(keys, index=ts_cols)).all(axis=1)
        ]

        if len(group_clean) < 3:
            continue

        res = group_raw[ts_cols + pixel_cols + ["time"]].copy()

        for value_col in value_cols:
            # Train GPR for this value_col and timeseries
            dates_train = pd.to_datetime(group_clean["time"])
            doy_train = dates_train.dt.dayofyear.values.reshape(-1, 1)
            pixel_train = group_clean[pixel_cols].values
            X_train = np.hstack([doy_train, pixel_train])
            y_train = group_clean[value_col].values
            if len(y_train) > 5000:
                idx = np.random.choice(len(X_train), size=5000, replace=False)
                X_train = X_train[idx]
                y_train = y_train[idx]

            # Estimate variability for kernel
            variability = group_clean[value_col].std()
            length_scale = 20 if variability > 0.2 else 80 if variability <= 0.2 else 50
            alpha = max(1e-2, 1 / len(group_clean))

            kernel = RBF(length_scale=length_scale) + WhiteKernel(noise_level=alpha)
            gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True)
            gp.fit(X_train, y_train)
        
            # Predict for the raw timeseries
            dates_pred = pd.to_datetime(group_raw["time"])
            doy_pred = dates_pred.dt.dayofyear.values.reshape(-1, 1)
            pixel_pred = group_raw[pixel_cols].values # shape: (n_pred, 2)
            X_pred = np.hstack([doy_pred, pixel_pred])
            mean, _ = gp.predict(X_pred, return_std=True)
        
            res[value_col] = mean

        results.append(res)

    return pd.concat(results, ignore_index=True)



# =====================================
# Find top crops in Switzerland
#top_crops = None
top_crops = ['Winter wheat (excluding Swiss Granum fodder wheat)', 'Silage and green maize', 'Winter rapeseed for edible oil', 'Winter barley', 'Sugar beets',\
    'Grain maize', 'Annual open-field vegetables, excluding preserved vegetables', 'Potatoes', 'Fodder wheat according to Swiss Granum list',\
    'Sunflower for edible oil', 'Spelt', 'Triticale', 'Spring wheat (excluding Swiss Granum fodder wheat)', 'Soybean', 'Peas for grain production (e.g., protein peas)',\
    'Oats', 'Rye', 'Open-field vegetables for preservation', 'Seed potatoes (contract farming)', 'Spring wheat (excluding Swiss Granum fodder wheat)',\
    'Beans and vetches for grain production (e.g., field beans)', 'Mixtures of beans, vetch, peas, chickpeas, and lupins with cereals or camelina, minimum 30% legumes at harvest (for grain)',\
    'Spring barley', 'Annual berries (e.g., strawberries)']

if top_crops is None:
    top_crops = []
    lnf_labels = os.path.expanduser('~/mnt/eo-nas1/data/landuse/documentation/LNF_code_classification_20260217.xlsx')
    df_labels = pd.read_excel(lnf_labels, sheet_name='label_sheet')
    df_labels = df_labels[df_labels['Crop_Label_lv3'].isin(['Arable Land'])]
    for c in df_labels.columns:
        print(c)
        if 'Area' not in c:
            continue
        top_yr = df_labels.sort_values(by=c, ascending=False)[:20]
        top_crops.extend(top_yr['Crop_EN'].tolist())
    top_crops = set(top_crops)
    top_lnfs_arable = df_labels[df_labels['Crop_EN'].isin(top_crops)]['LNF_code'].unique().tolist()
else:
    lnf_labels = os.path.expanduser('~/mnt/eo-nas1/data/landuse/documentation/LNF_code_classification_20260217.xlsx')
    df_labels = pd.read_excel(lnf_labels, sheet_name='label_sheet')
    top_lnfs_arable = df_labels[df_labels['Crop_EN'].isin(top_crops)]['LNF_code'].unique().tolist()


lnfs_grassland = df_labels[df_labels['Crop_Label_lv3'].isin(['Grassland'])]['LNF_code'].tolist()
top_lnfs_arable += [lnfs_grassland[0]]
print("Will use folowwing LNF codes (and mixed grassland):\n", top_lnfs_arable)
print(top_crops)

# =====================================
# Sample main crops locations (per year, across space) and save
crop_labels = os.path.expanduser('~/mnt/eo-nas1/data/landuse/documentation/LNF_code_classification_20250217.xlsx')
lnf_dir = os.path.expanduser('~/mnt/eo-nas1/data/landuse/raw')
tot_samples = 1000
save_path_sampledloc = 'samples.pkl'
if not os.path.exists(save_path_sampledloc):
    sample_locations_with_field(crop_labels, lnf_dir, tot_samples, save_path_sampledloc)

# =====================================
# Extract S2 and Soil data
save_path_sampledloc_S2 = 'samples_data.pkl'
if not os.path.exists(save_path_sampledloc_S2):

    s2_grid_path = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure/Project layers/gridface_s2tiles_CH.shp')
    s2_dir = os.path.expanduser('~/mnt/eo-nas1/data/satellite/sentinel2/raw/CH')
    soil_dir = os.path.expanduser('~/mnt/eo-nas1/data/satellite/sentinel2/DLR_soilsuite_preds/')

    extract_s2_data_field(save_path_sampledloc, s2_grid_path, s2_dir, soil_dir, save_path_sampledloc_S2)

# =====================================
# Predict FC using right soil model

save_path_FCpreds = 'samples_data_pred.pkl'
if not os.path.exists(save_path_FCpreds):
    predict_FC(save_path_sampledloc_S2, save_path_FCpreds)

# =====================================
# Count how many available timeseries per crop and year

df_samples = pd.read_pickle(save_path_FCpreds)

counts = (
    df_samples
    .drop_duplicates(['lnf_code', 'yr', 'sampled_x', 'sampled_y'])
    .groupby(['lnf_code', 'yr'])
    .size()
    .rename('n_samples')
)
#print("Number of samples per year and crop:\n", counts)

# TO DO: if needed, cap the number per crop/year -> perhaps wait unitl cleaning to do that, to see how many avlid tiemseries there will be


# =====================================
# Cleaning per field
df_clean = (
    df_samples.groupby(['lnf_code', 'yr', 'poly_id'], group_keys=False)
            .apply(clean_timeseries_field)
            .reset_index(drop=True)
)

# Get stats on cleaning
ts_cols = ['lnf_code', 'yr', 'poly_id'] # to do: check per field or just per "is_sampled"
n_before = (
    df_samples
    .groupby(ts_cols)
    .size()
    .rename("n_total")
)
n_after = (
    df_clean
    .groupby(ts_cols)
    .size()
    .rename("n_kept")
)
df_drop_stats = (
    pd.concat([n_before, n_after], axis=1)
    .fillna(0)
)
df_drop_stats["n_kept"] = df_drop_stats["n_kept"].astype(int)
df_drop_stats["n_dropped"] = df_drop_stats["n_total"] - df_drop_stats["n_kept"]
df_drop_stats["drop_fraction"] = df_drop_stats["n_dropped"] / df_drop_stats["n_total"]
df_drop_stats = df_drop_stats.reset_index()

# =====================================
# Filter timeseries: only keep if enough data points

#print(df_drop_stats.drop_fraction.describe())

drop_fraction_threshold = 0.7 # max fraction dropped
filtered_stats = df_drop_stats[df_drop_stats["drop_fraction"] <= drop_fraction_threshold]
keys_to_keep = filtered_stats[ts_cols]
df_clean_filtered = df_clean.merge(keys_to_keep, on=ts_cols, how="inner")
df_samples_filtered = df_samples.merge(keys_to_keep, on=ts_cols, how="inner")
print(f"Number of fields retained: {len(filtered_stats)}/{len(df_drop_stats)}")

# TO DO: could cap the number of samples per crop/year

# ====================
# Deal with duplicated satellite pixels
group_cols = ['poly_id', 'x', 'y', 'time', 'yr', 'sampled_x', 'sampled_y', 'lnf_code', 'is_sample_pixel'] # same as ['poly_id', 'x', 'y', 'time']

df_clean_filtered = (
    df_clean_filtered
    .groupby(group_cols, as_index=False)
    .mean(numeric_only=True)
)
df_samples_filtered = (
    df_samples_filtered
    .groupby(group_cols, as_index=False)
    .mean(numeric_only=True)
)


# ====================
# Plot cleaned timeseries (before gapfilling)

ts_cols = ['lnf_code', 'poly_id', 'x', 'y', 'yr', 'sampled_x', 'sampled_y'] 
for yr, df_year in df_clean_filtered.groupby('yr'):

    # --- Select 10 random time series ---
    unique_ts = df_year[ts_cols].drop_duplicates()

    n_select = min(5, len(unique_ts))  # safe if <10 exist

    selected_ts = unique_ts.sample(
        n=n_select,
        random_state=42
    )

    # Keep only selected time series
    df_subset = df_year.merge(
        selected_ts,
        on=ts_cols,
        how='inner'
    )

    unique_codes = df_subset['lnf_code'].unique()
    cmap = plt.get_cmap('tab20')
    color_dict = {
        code: cmap(i % 10)
        for i, code in enumerate(unique_codes)
    }

    fig, ax = plt.subplots(figsize=(10,6))

    for keys, group in df_subset.groupby(ts_cols):
        lnf_code = keys[0]
        group = group.sort_values('time')

        ax.plot(
            group['time'],
            group['pv'],   # change if needed
            alpha=0.3,
            color=color_dict[lnf_code]
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("PV")
    ax.set_title(f"Example of cleaned pixel time series — Year {yr}")

    # --- Add legend (one per lnf_code) ---
    handles = [
        plt.Line2D([0], [0], color=color_dict[c], lw=2)
        for c in unique_codes
    ]
    ax.legend(handles, unique_codes, title="lnf_code")

    plt.tight_layout()
    plt.savefig(f'cleaned_timeseries_{yr}.png')
    plt.close()


# =====================================
# GPR gapfilling 

gapfilled_path = 'samples_data_gpr.pkl'

if 1: #not os.path.exists(gapfilled_path):
    
    # Train per field and apply GPR to sampled pixel timeseries
    ts_cols = ['lnf_code', 'yr', 'poly_id']
    value_cols = ["pv", "npv", "soil"]
    pixel_cols = ["x","y"] 
    
    df_gpr = gapfill_dataframe_gpr_per_field_parallel(
        df_samples_filtered, df_clean_filtered, 
        ts_cols, value_cols, pixel_cols, 
        alpha=1e-4, n_jobs=1  # control parallelisation
    )
    df_gpr.to_pickle('samples_data_gpr.pkl')


# =====================================
# Check some gapfilled timeseries

df_gpr = pd.read_pickle(gapfilled_path)

# Filter for time (1st Jul prev yr - 30 Jun current yr)
df_gpr['time'] = pd.to_datetime(df_gpr['time'])
start_dates = pd.to_datetime((df_gpr['yr'].astype(int) - 1).astype(str) + '-07-01')
end_dates   = pd.to_datetime(df_gpr['yr'].astype(str) + '-06-30')
df_gpr = df_gpr[
    (df_gpr['time'] >= start_dates) &
    (df_gpr['time'] <= end_dates)
]


ts_cols = ['lnf_code', 'yr', 'x', 'y']

#  Sample 5 random timeseries
unique_ts = df_gpr[ts_cols].drop_duplicates()
sampled_ts = unique_ts.sample(10)

nrows = len(sampled_ts)
fig, axes = plt.subplots(nrows, 1, figsize=(10, 4*nrows))
axes = axes.flatten()

for idx, (_, row) in enumerate(sampled_ts.iterrows()):
    ax = axes[idx]

    # Masks
    mask_raw = (df_samples_filtered[ts_cols] == row[ts_cols]).all(axis=1)
    mask_clean = (df_clean_filtered[ts_cols] == row[ts_cols]).all(axis=1)
    mask_gpr = (df_gpr[ts_cols] == row[ts_cols]).all(axis=1)
    
    # Select and sort
    df_raw_ts = df_samples_filtered[mask_raw].sort_values('time')
    df_clean_ts = df_clean_filtered[mask_clean].sort_values('time')
    df_gpr_ts = df_gpr[mask_gpr].sort_values('time')

    variability = df_clean_ts["pv"].std()
    print(f"Timeseries {idx}: Variability = {variability:.4f}")

    # Plot
    ax.plot(df_raw_ts['time'], df_raw_ts['pv'], 'o-', alpha=0.4, label='Raw PV')
    ax.plot(df_clean_ts['time'], df_clean_ts['pv'], 's-', alpha=0.6, label='Cleaned PV')
    ax.plot(df_gpr_ts['time'], df_gpr_ts['pv'], 'x--', alpha=0.8, label='GPR PV')

    ax.set_title(
        f"{row['lnf_code']} - {row['yr']}\n({row['x']},{row['y']})",
        fontsize=8
    )

# Remove unused subplots (if grid > number of TS)
for j in range(nrows, len(axes)):
    fig.delaxes(axes[j])

# Common labels
fig.supxlabel("Time")
fig.supylabel("PV fraction")

# Single legend for whole figure
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')

plt.tight_layout()
plt.savefig("gpr_all_timeseries_subplots_claude.png")

