import os
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import dask
from dask.diagnostics import ProgressBar
from shapely.geometry import Point, box
import joblib
import calendar
import matplotlib.pyplot as plt
import contextily as ctx
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")
import optuna
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from functools import reduce
import operator
import seaborn as sns


def load_stations_metadata(metadata_path):
    """Load station metadata."""
    vars_to_keep = [
        'station_abbr', 'station_name', 'station_height_masl',
        'station_coordinates_lv95_east', 'station_coordinates_lv95_north',
        'station_coordinates_wgs84_lat', 'station_coordinates_wgs84_lon',
        'station_exposition_en'
    ]
    stations = pd.read_csv(metadata_path, delimiter=';', encoding='latin1')
    return stations[vars_to_keep]


def load_ei_data(ei_dir):
    """Load EI data for all stations."""
    ei_station_dict = {}
    for station_file in os.listdir(ei_dir):
        df_ei = pd.read_csv(os.path.join(ei_dir, station_file))
        station_abbr = station_file.split('_')[-1].split('.csv')[0]
        ei_station_dict[station_abbr] = df_ei
    #print(f'Loaded EI data for {len(ei_station_dict.keys())} stations')
    return ei_station_dict


def prepare_covariates(data_vars, covariates_dir):
    """Prepare covariates based on the required variables."""
    cov_files = []
    if 'elev' in data_vars or 'slope' in data_vars or 'aspect' in data_vars:
        #df = get_topo(data_vars, covariates_dir)
        topo_path = os.path.join(covariates_dir, 'dem_100m.zarr')
        ds = xr.open_zarr(topo_path)
        # Check that variables exists, else return an error
        required_vars = []
        if 'elev' in data_vars:
            required_vars.append('height')  # Assuming 'height' corresponds to 'elev'
        if 'slope' in data_vars:
            required_vars.append('slope')
        if 'aspect' in data_vars:
            required_vars.append('aspect')
        missing_vars = [var for var in required_vars if var not in ds.data_vars]
        if missing_vars:
            raise ValueError(f"The following required variables are missing in the dataset: {missing_vars}")
        cov_files.append(topo_path)
    if 'prec_daily_avg' in data_vars:
        prec_daily_path = prep_daily_prec(data_vars, covariates_dir)
        cov_files.append(prec_daily_path)
    if 'prec_monthly_avg' in data_vars:
        prec_monthy_path = prep_monthly_prec(data_vars, covariates_dir)
        cov_files.append(prec_monthy_path)
    
    return cov_files


def get_topo(data_vars, covariates_dir):
    topo_path = os.path.join(covariates_dir, 'dem_100m.zarr')
    ds = xr.open_zarr(topo_path)
    df = ds.to_dataframe().reset_index()
    ds_vars = []
    if 'elev' in data_vars:
      ds_vars.append('height')
    if 'slope' in data_vars:
      ds_vars.append('slope')
    if 'aspect' in data_vars:
      ds_vars.append('aspect')
    ds_vars += ['x', 'y']
    df = df[ds_vars]

    if 'height' in df.columns:
      df = df.rename(columns={'height':'elev'})

    return df


def prep_daily_prec(data_vars, covariates_dir):
    """
    Process daily precipitation data from yearly .zarr files and compute daily averages.
    """
    prec_path = os.path.join(covariates_dir, 'precip')
    daily_avg_path = os.path.join(covariates_dir, 'precip', 'daily_avg.zarr')

    if not os.path.exists(daily_avg_path):
        zarr_files = [os.path.join(prec_path, f) for f in os.listdir(prec_path) if f.endswith('.zarr') and 'MeteoSwiss' in f]

        # Initialize an empty list to store daily averages
        daily_avg_list = []

        # Loop over each day of the year (DOY)
        for doy in range(1, 366):  # Assuming non-leap years; adjust for leap years if needed
            daily_data = []

            # Load data for the current DOY from all files
            for zarr_file in zarr_files:
                ds = xr.open_zarr(zarr_file)
                # Select data for the current DOY
                daily_ds = ds.sel(time=ds['time.dayofyear'] == doy)
                daily_data.append(daily_ds)

            # Combine data for the current DOY
            if daily_data:
                combined_daily_ds = xr.concat(daily_data, dim="time")
                # Compute the daily average
                daily_avg = combined_daily_ds.mean(dim="time")
                daily_avg_list.append(daily_avg)

        # Combine all daily averages into a single dataset
        combined_daily_avg = xr.concat(daily_avg_list, dim="time")
        combined_daily_avg.to_zarr(os.path.join(covariates_dir, 'precip', 'daily_avg.zarr'))
        print('Saved daily mean precip')
    """
    else:
        combined_daily_avg = xr.open_zarr(os.path.join(covariates_dir, 'precip', 'daily_avg.zarr'))

    print(combined_daily_avg)

    # Convert to DataFrame
    with ProgressBar():
        df_prec = combined_daily_avg.to_dataframe().reset_index()

    # Keep only relevant columns
    df_prec = df_prec[['time', 'x', 'y', 'RhiresD']]

    # Rename columns for consistency
    df_prec = df_prec.rename(columns={'RhiresD': 'prec_daily_avg'})
    """
    return daily_avg_path


def prep_monthly_prec(data_vars, covariates_dir):
    """
    Process daily precipitation data from yearly .zarr files and compute avg mothly total precip.
    """
    prec_path = os.path.join(covariates_dir, 'precip')
    monthly_avg_path = os.path.join(covariates_dir, 'precip', 'monthly_avg.zarr')
    if not os.path.exists(monthly_avg_path):
        zarr_files = [os.path.join(prec_path, f) for f in os.listdir(prec_path) if f.endswith('.zarr') and 'MeteoSwiss' in f]

        # Initialize an empty list to store datasets
        monthly_avg_list = []

        # Process each Zarr file
        for zarr_file in zarr_files:
            ds = xr.open_zarr(zarr_file)

            # Resample to compute monthly averages for each year
            monthly_avg = ds.resample(time='ME').sum()
            monthly_avg_list.append(monthly_avg)

        # Combine all monthly averages into a single dataset
        combined_monthly_avg = xr.concat(monthly_avg_list, dim="time")

        # Group by month (ignoring year) and compute the mean for each calendar month
        monthly_avg_over_years = combined_monthly_avg.groupby('time.month').mean()
        monthly_avg_over_years = monthly_avg_over_years.rename({'month': 'time'})

        # Save the combined dataset to a Zarr file
        monthly_avg_over_years.to_zarr(monthly_avg_path, mode='w')
        print('Saved monthly mean precip')
    return monthly_avg_path


def extract_features(data_path, gdf):
    ds = xr.open_zarr(data_path)
    if 'daily' in data_path:
        var_name_final = 'daily_avg_precip'
    if 'monthly' in data_path:
        var_name_final = 'monthly_avg_precip'

    coords = dict(
        x=('points', gdf['x'].values),
        y=('points', gdf['y'].values)
    )
    var_name = list(ds.data_vars)[0] 
    da = ds[var_name].interp(coords) 
    df = da.to_dataframe().reset_index()
    df = df.pivot(index='points', columns='time', values=var_name)
    df['x'] = gdf['x'].values
    df['y'] = gdf['y'].values
    time_cols = [col for col in df.columns if col not in ['x', 'y']]
    df = df[['x', 'y'] + time_cols]
    if 'daily' in data_path: 
        rename_dict = {col: f"{var_name_final}_{col+1}" for col in time_cols}
    else: 
        rename_dict = {col: f"{var_name_final}_{col}" for col in time_cols}
    df = df.rename(columns=rename_dict)
    df = df.reset_index(drop=True)

    return df
    

def make_model(model_type="rf"):
    """
    Create ML pipeline for EI modeling.

    Parameters
    ----------
    model_type : str
        One of: "gpr", "rf", "nn"

    Returns
    -------
    sklearn Pipeline
    """

    if model_type == "gpr":

        kernel = C(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3)

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                n_restarts_optimizer=2
            ))
        ])

    elif model_type == "rf":

        pipeline = Pipeline([
            ('model', RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                n_jobs=1,
                random_state=42
            ))
        ])

    elif model_type == "nn":

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', MLPRegressor(
                hidden_layer_sizes=(64, 64, 32),
                activation='relu',
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                random_state=42
            ))
        ])

    else:
        raise ValueError(
            "model_type must be 'gpr', 'rf', or 'nn'"
        )

    return pipeline


def prepare_grid_labeled(stations, cov_files, data_vars, dem_file_pattern):
    """Extract cells of output grid containing stations and add covariate values for those cells"""

    # Create GeoDataFrame for stations
    geometry = [Point(xy) for xy in zip(stations['station_coordinates_lv95_east'], stations['station_coordinates_lv95_north'])]
    stations_gdf = gpd.GeoDataFrame(stations, geometry=geometry, crs=2056)

    # Open DEM file and create GeoDataFrame -> serves as output grid reference
    dem_file = [f for f in cov_files if dem_file_pattern in f][0]
    dem_grid = xr.open_zarr(dem_file)
    dem_df = dem_grid.to_dataframe().reset_index().dropna()
    geometry = [box(x, y - 100, x + 100, y) for x, y in zip(dem_df['x'], dem_df['y'])]
    dem_gdf = gpd.GeoDataFrame(dem_df, geometry=geometry, crs=32632)
    if 'elev' in data_vars:
        dem_gdf = dem_gdf.rename(columns={'height':'elev'})

    # Spatial join: grid cells where station is present
    gdf_labeled = dem_gdf.sjoin(stations_gdf.to_crs(dem_gdf.crs))

    # Extract features for grid cells 
    for file in cov_files:
        if 'dem' in file:
            continue # any ['elev', 'slope', 'aspect'] should already be added from previous join
        df_file = extract_features(file, gdf_labeled)
        gdf_labeled = gdf_labeled.merge(df_file, on=['x', 'y'], how='left')
    
    return gdf_labeled


def prepare_training_data(gdf_labeled, data_vars, ei_station_dict, pred_col):
    """Prepare data for training: one row per grid cell and per doy, with EI daily as target variable"""
    
    # Prepare data for training: add feature
    static_cols = [c for c in data_vars if c in {'elev', 'aspect', 'slope'}]

    # --- DAILY features ---
    daily_cols = [c for c in gdf_labeled.columns if c.startswith("daily_avg_precip_")]
    if len(daily_cols) > 0:
        df_daily = gdf_labeled.melt(
            id_vars=['station_abbr', 'x', 'y'] + static_cols,  # keep spatial + station info
            value_vars=daily_cols,
            var_name='doy',
            value_name='prec_daily_avg'
        )
        df_daily['doy'] = df_daily['doy'].str.extract(r'(\d+)').astype(int)
        df_daily['monthnbr'] = pd.to_datetime(df_daily['doy'], format='%j').dt.month
    else:
        df_daily = pd.DataFrame()

    # --- MONTHLY features ---
    monthly_cols = [c for c in gdf_labeled.columns if c.startswith("monthly_avg_precip_")]
    if len(monthly_cols) > 0:
        df_monthly = gdf_labeled.melt(
            id_vars=['station_abbr', 'x', 'y'] + static_cols,
            value_vars=monthly_cols,
            var_name='monthnbr',
            value_name='prec_monthly_avg'
        )
        df_monthly['monthnbr'] = df_monthly['monthnbr'].str.extract(r'(\d+)').astype(int)
    else:
        df_monthly = pd.DataFrame()
    
    if not df_daily.empty and not df_monthly.empty:
        df_features = df_daily.merge(
            df_monthly,
            on=['station_abbr', 'x', 'y', 'monthnbr'] + static_cols,
            how='left'
        )
    elif not df_daily.empty:
        df_features = df_daily.copy()
    elif not df_monthly.empty:
        df_features = df_monthly.copy()
    else:
        raise ValueError("No daily or monthly precipitation columns found.")

    ei_all = []
    for station, df in ei_station_dict.items():
        tmp = df[['doy', pred_col]].copy() 
        tmp['station_abbr'] = station
        ei_all.append(tmp)
    ei_all = pd.concat(ei_all, ignore_index=True)

    if 'doy' not in df_features.columns: # Only monthly features were called
        # Expand months to DOY
        doy_rows = []
        for _, row in df_features.iterrows():
            month = int(row['monthnbr'])
            ndays = calendar.monthrange(2021, month)[1]
            # Get first DOY of month
            first_doy = (
                pd.Timestamp(year=2021, month=month, day=1) # random year
                .dayofyear
            )
            for d in range(ndays):
                new_row = row.copy()
                new_row['doy'] = first_doy + d
                doy_rows.append(new_row)
        df_features = pd.DataFrame(doy_rows)

    df_data = df_features.merge(
        ei_all,
        on=['station_abbr', 'doy'],
        how='inner'
    )

    if 'aspect' in data_vars:
        df_data["aspect_sin"] = np.sin(np.deg2rad(df_data["aspect"]))
        df_data["aspect_cos"] = np.cos(np.deg2rad(df_data["aspect"]))
    df_data['sin_doy'] = np.sin(2 * np.pi * df_data['doy'] / 365)
    df_data['cos_doy'] = np.cos(2 * np.pi * df_data['doy'] / 365)

    if 'prec_daily_fraction' in data_vars:
        if 'prec_daily_avg' not in data_vars:
            raise Exception("Called prec_daily_fraction but not prec_daily_avg") 
        df_data["prec_daily_fraction"] = (
            df_data["prec_daily_avg"] /
            df_data.groupby("station_abbr")["prec_daily_avg"].transform("sum")
        )

    return df_data


def split_stations(gdf, station_col, x_col, y_col, frac_train, frac_val, grid_size=25000, seed=42):
    """
    Spatially split stations on a grid so that entire stations stay in one split.
    Will create a gird based on grid_size and randomly assign cells of that grid to train/val/test.
    To get a spatial splitting, grid_size must be big enough.
    For Switzerland: it is 220km x 348 km. With a grid_size of 30km -> 60 grid cells
    """

    gdf = gdf.copy()

    # Unique stations with coordinates
    stations = gdf[[station_col, x_col, y_col]].drop_duplicates()

    # Assign stations to grid cells
    stations["grid_x"] = (stations[x_col] // grid_size).astype(int)
    stations["grid_y"] = (stations[y_col] // grid_size).astype(int)

    grid_cells = (
        stations[["grid_x", "grid_y"]]
        .drop_duplicates()
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    n = len(grid_cells)
    n_train = int(frac_train * n)
    n_val = int(frac_val * n)
    print(f'Splitting data using a regular grid with {n} cells: {n_train} train, {n_val} val, {n-n_train-n_val} test')
    
    # Assign splits to cells
    grid_cells["split"] = "test"
    grid_cells.loc[:n_train-1, "split"] = "train"
    grid_cells.loc[n_train:n_train+n_val-1, "split"] = "val"

    # Merge cell splits back to stations
    stations = stations.merge(grid_cells, on=["grid_x", "grid_y"], how="left")

    # Merge station splits back to full dataset
    gdf = gdf.merge(stations[[station_col, "split"]], on=station_col, how="left")

    return gdf


def objective(trial):
    """
    Objective function for Optuna hyperparameter tuning.

    Parameters:
    ----------
    trial : optuna.trial.Trial
        A single trial of hyperparameter combinations.

    Returns:
    -------
    float
        The evaluation metric (e.g., RMSE) to minimize.
    """
    # Define the hyperparameter search space
    if MODEL_TYPE == "rf":
        n_estimators = trial.suggest_int("n_estimators", 100, 500)
        max_depth = trial.suggest_int("max_depth", 10, 50)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
        pipeline = make_model("rf")
        pipeline.set_params(model__n_estimators=n_estimators,
                            model__max_depth=max_depth,
                            model__min_samples_leaf=min_samples_leaf)
    elif MODEL_TYPE == "nn":
        n_layers = trial.suggest_int("n_layers", 1, 3)
        layer_sizes = []
        for i in range(n_layers):
            layer_size = trial.suggest_int(f"n_units_l{i}", 16, 256, log=True)
            layer_sizes.append(layer_size)
        hidden_layer_sizes = tuple(layer_sizes)
        learning_rate_init = trial.suggest_float("learning_rate_init", 0.0001, 0.01, log=True)
        max_iter = trial.suggest_int("max_iter", 500, 2000)
        pipeline = make_model("nn")
        pipeline.set_params(model__hidden_layer_sizes=hidden_layer_sizes,
                            model__learning_rate_init=learning_rate_init,
                            model__max_iter=max_iter)
    else:
        raise ValueError("Unsupported model type")

    # Perform GroupKFold cross-validation
    gkf = GroupKFold(n_splits=5)
    rmse_scores = []
    for train_idx, val_idx in gkf.split(X_train, y_train, groups_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        pipeline.fit(X_tr, y_tr)
        y_pred = pipeline.predict(X_val)

        # Compute RMSE
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        rmse_scores.append(rmse)

    # Return the mean RMSE across folds
    return np.mean(rmse_scores)


def train_model(df_data, gdf_labeled, data_vars, pred_col, model_type, model_file, tune=False):

    # Split stations for train/val/test
    gdf_labeled = split_stations(gdf_labeled, 'station_abbr', 'station_coordinates_lv95_east', 'station_coordinates_lv95_north', 0.7, 0.15)
    train_stations = gdf_labeled[gdf_labeled["split"]=="train"]['station_abbr'].unique()
    val_stations = gdf_labeled[gdf_labeled["split"]=="val"]['station_abbr'].unique()
    
    # Plot split
    if not os.path.exists('results/station_train_split.png'):
        pt_stations = [Point(xy) for xy in zip(gdf_labeled['station_coordinates_lv95_east'], gdf_labeled['station_coordinates_lv95_north'])]
        pt_stations = gpd.GeoDataFrame(geometry=pt_stations, crs=2056)
        pt_stations['split'] = gdf_labeled['split']
        fig, ax = plt.subplots(figsize=(10,10))
        pt_stations.plot(ax=ax, column='split', markersize=50, edgecolor='black', legend=True, cmap='Accent')
        ctx.add_basemap(ax, source=ctx.providers.SwissFederalGeoportal.NationalMapColor, crs=pt_stations.crs)
        ax.set_axis_off()
        plt.savefig('results/station_train_split.png')

    # Filter data for train and validation sets
    train_df = df_data[df_data['station_abbr'].isin(train_stations)]
    val_df = df_data[df_data['station_abbr'].isin(val_stations)]

    # Prepare data for training
    global X_train, y_train, groups_train
    X_train = train_df[data_vars].values
    y_train = train_df[pred_col].values
    groups_train = train_df['station_abbr'].values

    if tune:
        # Run Optuna study, wit Kfold CV that return mean fold RMSE 
        print("\nRunning Optuna hyperparameter tuning...")
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=20)

        # Print the best hyperparameters
        best_params = study.best_params
        # Format to work with model
        if "n_layers" in best_params:
            n_layers = best_params.pop("n_layers")
            layer_sizes = []
            for i in range(n_layers):
                layer_size = best_params.pop(f"n_units_l{i}")
                layer_sizes.append(layer_size)
            best_params["hidden_layer_sizes"] = tuple(layer_sizes)
        print("\nBest hyperparameters:")
        print(best_params)

        # Train the final model on the full training set
        print("\nTraining final model on the full training set...")
        pipeline = make_model(model_type)
        pipeline.set_params(**{f"model__{key}": value for key, value in best_params.items()})
        
    else:
        # Train with default hyperparameters
        print("\nTraining with default hyperparameters...")
        pipeline = make_model(model_type)

    # Train final model
    pipeline.fit(X_train, y_train)

    # Evaluate on val stations
    X_val = val_df[data_vars].values
    y_val = val_df[pred_col].values
    y_val_pred = pipeline.predict(X_val)
    if pred_col == 'EI_daily_cumsum':
        y_val_pred = np.clip(y_val_pred, 0, 100)

    rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
    r2_val = r2_score(y_val, y_val_pred)
    print(f"\nValidation set RMSE: {rmse_val:.3f}, R²: {r2_val:.3f}")

    # Save the trained model
    joblib.dump(pipeline, model_file)
    print(f"Final model saved as {model_file}")

    return pipeline


def EI_to_cumul(df):

    df = df.copy()

    # Ensure correct ordering
    if 'doy' in df.columns:
        df = df.sort_values(['station_abbr', 'doy'])

    for prefix in ['predicted_', '']:

        avg_col = f'{prefix}EI_daily_avg'
        pct_col = f'{prefix}EI_daily_percent'
        csum_col = f'{prefix}EI_daily_cumsum'

        # --- Case 1: have daily average ---
        if avg_col in df.columns:

            total_EI = (
                df.groupby("station_abbr")[avg_col]
                .transform("sum")
            )
            df.loc[:, pct_col] = np.where(
                total_EI == 0,
                0,
                df[avg_col] / total_EI * 100
            )
            df.loc[:, csum_col] = (
                df.groupby("station_abbr")[pct_col]
                .cumsum()
            )

        # --- Case 2: have percent already ---
        elif pct_col in df.columns:

            # Force positivity
            df.loc[:, pct_col] = np.maximum(
                df[pct_col],
                0
            )
            # Normalize per station
            group_sums = (
                df.groupby("station_abbr")[pct_col]
                .transform("sum")
            )

            group_sums = group_sums.replace(0, np.nan)

            df.loc[:, pct_col] = (
                df[pct_col] / group_sums
            ) * 100

            df.loc[:, pct_col] = df[pct_col].fillna(0)

            # Cumulative per station
            df.loc[:, csum_col] = (
                df.groupby("station_abbr")[pct_col]
                .cumsum()
            )

        else:

            print(
                f"Skipping {prefix}: "
                "no avg or percent column found"
            )

    return df


def compute_kge_nse(obs, pred):
    """
    Compute KGE and NSE for one observed/predicted curve.

    Parameters
    ----------
    obs : array-like
        Observed values
    pred : array-like
        Predicted values

    Returns
    -------
    kge : float
    nse : float
    """

    obs = np.asarray(obs)
    pred = np.asarray(pred)

    # --- Correlation ---
    r = np.corrcoef(obs, pred)[0, 1]

    # --- Variability ratio ---
    alpha = np.std(pred) / np.std(obs)

    # --- Bias ratio ---
    beta = np.mean(pred) / np.mean(obs)

    # --- KGE ---
    kge = 1 - np.sqrt(
        (r - 1)**2 +
        (alpha - 1)**2 +
        (beta - 1)**2
    )

    # --- NSE ---
    nse = 1 - (
        np.sum((obs - pred)**2) /
        np.sum((obs - np.mean(obs))**2)
    )

    return kge, nse


def evaluate_curves_per_station(
    df,
    obs_col="EI_daily_cumsum",
    pred_col="predicted_EI_daily_cumsum",
    station_col="station_abbr"
    ):
    """
    Evaluate predicted vs observed curves per station.

    Returns
    -------
    metrics_df : DataFrame
        Metrics per station
    summary : dict
        Mean NSE and KGE
    """

    results = []

    stations = df[station_col].unique()

    for station in stations:

        sub = df[
            df[station_col] == station
        ].sort_values("doy")

        obs = sub[obs_col].values
        pred = sub[pred_col].values

        kge, nse = compute_kge_nse(obs, pred)

        results.append({
            "station": station,
            "KGE": kge,
            "NSE": nse
        })

    metrics_df = pd.DataFrame(results)

    summary = {
        "mean_KGE": metrics_df["KGE"].mean(),
        "std_KGE": metrics_df["KGE"].std(),
        "mean_NSE": metrics_df["NSE"].mean(),
        "std_NSE": metrics_df["NSE"].std()
    }

    return metrics_df, summary


def predict_test_set(df_data, gdf_labeled, data_vars, pred_col, model_file):
    """
    Evaluate the model on the test set.

    Parameters:
    ----------
    df_data : pd.DataFrame
        The full dataset containing features and station information.
    gdf_labeled : gpd.GeoDataFrame
        The labeled grid cells with station information.
    data_vars : list
        List of feature column names.
    pred_col : str
        The target column name.
    model_file : str
        Path to the saved model file.
    output_file : str
        Path to save the test predictions CSV file.
    """
    print("\nPreparing test set...")

    # Split stations for train/val/test
    gdf_labeled = split_stations(gdf_labeled, 'station_abbr', 'station_coordinates_lv95_east', 'station_coordinates_lv95_north', 0.7, 0.15)
    test_stations = gdf_labeled[gdf_labeled["split"]=="test"]['station_abbr'].unique()

    # Filter data for the test set
    test_df = df_data[df_data['station_abbr'].isin(test_stations)].copy()
    X_test = test_df[data_vars].values
    y_test = test_df[pred_col].values

    # Load the trained model
    print(f"Loading model from {model_file}...")
    pipeline = joblib.load(model_file)

    # Make predictions
    print("\nPredicting on the test set...")
    y_test_pred = pipeline.predict(X_test)
    if pred_col == 'EI_daily_cumsum':
        y_test_pred = np.clip(y_test_pred, 0, 100)

    # Add predictions to the test DataFrame
    test_df['predicted_' + pred_col] = y_test_pred

    # Evaluate the model: on the cumsum (easier to compare models)
    test_rmse = np.sqrt(mean_squared_error(test_df[pred_col], test_df['predicted_' + pred_col]))
    test_mae = mean_absolute_error(test_df[pred_col],test_df['predicted_' + pred_col])
    test_r2 = r2_score(test_df[pred_col], test_df['predicted_' + pred_col])
    print("\n===== Test Set Results =====")
    print(f"Test RMSE: {test_rmse:.3f}")
    print(f"Test MAE: {test_mae:.3f}")
    print(f"Test NRMSE: {test_rmse / (test_df[pred_col].max() - test_df[pred_col].min()):.3f}")
    print(f"Test R²:   {test_r2:.3f}")
    
    if pred_col != 'EI_daily_cumsum': # Cumulate to daily cumulative percent
        test_df = EI_to_cumul(test_df)

    # Evaluate the model: on the cumsum (easier to compare models)
    test_rmse = np.sqrt(mean_squared_error(test_df['EI_daily_cumsum'], test_df['predicted_EI_daily_cumsum']))
    test_mae = mean_absolute_error(test_df['EI_daily_cumsum'],test_df['predicted_EI_daily_cumsum'])
    test_r2 = r2_score(test_df['EI_daily_cumsum'], test_df['predicted_EI_daily_cumsum'])
    print("\n===== Test Set Results =====")
    print(f"Test RMSE: {test_rmse:.3f}")
    print(f"Test MAE: {test_mae:.3f}")
    print(f"Test NRMSE: {test_rmse / (test_df['EI_daily_cumsum'].max() - test_df['EI_daily_cumsum'].min()):.3f}")
    print(f"Test R²:   {test_r2:.3f}")

    # Evaluate with hydrology metriscs (NSE, KGE)
    metrics_df, summary = evaluate_curves_per_station(test_df, 'EI_daily_cumsum', 'predicted_EI_daily_cumsum', 'station_abbr')
    print("\n===== KGE/NSE Results =====")
    print("\nMean KGE:", summary["mean_KGE"])
    print("Std  KGE:", summary["std_KGE"])
    print("Mean NSE:", summary["mean_NSE"])
    print("Std  NSE:", summary["std_NSE"])

    # Plot curves
    plot_test_predictions(test_df, 'EI_daily_cumsum', f'results/predicted_test_{os.path.basename(model_file).split(".joblib")[0]}.png', num_stations=len(test_stations), random_seed=42)

    return test_df


def plot_test_predictions(df, pred_col, save_path, num_stations=10, random_seed=42):
    """
    Plot predictions for a random sample of stations on the same plot.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing actual and predicted values.
    save_path : str
        Path to save the plot.
    num_stations : int, optional
        Number of stations to randomly sample for plotting. Default is 10.
    random_seed : int, optional
        Random seed for reproducibility. Default is 42.
    """
    print("\nPlotting predictions...")

    # Get unique stations and randomly sample
    stations = df['station_abbr'].unique()
    np.random.seed(random_seed)
    sampled_stations = np.random.choice(stations, size=min(num_stations, len(stations)), replace=False)

    # Filter the DataFrame for the sampled stations
    df_sampled = df[df['station_abbr'].isin(sampled_stations)]

    # Assign a unique color to each station
    cmap = plt.get_cmap('tab20', len(sampled_stations))  # Use 'tab20' or another colormap
    station_colors = {station: cmap(i) for i, station in enumerate(sampled_stations)}

    # Plot predictions vs actual values for the sampled stations
    plt.figure(figsize=(12, 8))
    for station in sampled_stations:
        station_df = df_sampled[df_sampled['station_abbr'] == station]
        color = station_colors[station]  # Get the color for this station
        plt.plot(station_df['doy'], station_df[pred_col], label=f"Actual ({station})", color=color, alpha=0.7)
        plt.plot(station_df['doy'], station_df[f'predicted_{pred_col}'], label=f"Predicted ({station})", color=color, linestyle='--', alpha=0.7)

    # Customize the plot
    plt.title("Predictions vs Actuals for Randomly Sampled Stations", fontsize=16)
    plt.xlabel("Day of Year (DOY)", fontsize=14)
    plt.ylabel("EI Daily Cumulative Sum", fontsize=14)
    plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(alpha=0.5)
    plt.tight_layout()

    # Save and show the plot
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close()

    return


def prepare_full_grid(cov_files, data_vars, dem_file_pattern):
    """
    Prepare the full dataset for inference, including all grid cells from the DEM grid.

    Parameters:
    ----------
    cov_files : list
        List of covariate file paths.
    data_vars : list
        List of feature column names.
    dem_file_pattern : str
        Pattern to identify the DEM file.

    Returns:
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing all grid cells with extracted features.
    """
    print("\nGetting full grid for inference...")

    # Open DEM file and create GeoDataFrame -> serves as output grid reference
    dem_file = [f for f in cov_files if dem_file_pattern in f][0]
    dem_grid = xr.open_zarr(dem_file)
    dem_df = dem_grid.to_dataframe().reset_index().dropna()
    geometry = [box(x, y - 100, x + 100, y) for x, y in zip(dem_df['x'], dem_df['y'])]
    dem_gdf = gpd.GeoDataFrame(dem_df, geometry=geometry, crs=32632)
    if 'elev' in data_vars:
        dem_gdf = dem_gdf.rename(columns={'height': 'elev'})

    # Extract features for all grid cells
    for file in cov_files:
        if 'dem' in file:
            continue  # Skip DEM as it's already added
        df_file = extract_features(file, dem_gdf)
        dem_gdf = dem_gdf.merge(df_file, on=['x', 'y'], how='left')

    return dem_gdf


def predict_per_grid_cell(dem_gdf, pipeline, data_vars, pred_col, output_file, chunk_size=5000):

    # For a single grid cell: extract df with rows containing features for daily EI prediction
    static_cols = [c for c in data_vars if c in {'elev', 'aspect', 'slope'}]
    daily_cols = [c for c in dem_gdf.columns if c.startswith("daily_avg_precip_")]
    monthly_cols = [c for c in dem_gdf.columns if c.startswith("monthly_avg_precip_")]


    total_cells = len(dem_gdf)
    n_chunks = (total_cells + chunk_size - 1) // chunk_size
    print(f"\nPredicting {total_cells} grid cells in {n_chunks} chunks of {chunk_size}...")

    #all_predictions = []
    writer = None  # parquet writer
    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, total_cells)
        print(f"Processing chunk {chunk_idx + 1}/{n_chunks}...")

        chunk_gdf = dem_gdf.iloc[start:end]
        
        # --- DAILY features ---
        if len(daily_cols) > 0:
            df_daily = chunk_gdf.melt(
                id_vars=['x', 'y'] + static_cols,
                value_vars=daily_cols,
                var_name='doy',
                value_name='prec_daily_avg'
            )
            df_daily['doy'] = df_daily['doy'].str.extract(r'(\d+)').astype(int)
            df_daily['monthnbr'] = pd.to_datetime(df_daily['doy'], format='%j').dt.month
        else:
            df_daily = pd.DataFrame()

        # --- MONTHLY features ---
        if len(monthly_cols) > 0:
            df_monthly = chunk_gdf.melt(
                id_vars=['x', 'y'] + static_cols,
                value_vars=monthly_cols,
                var_name='monthnbr',
                value_name='prec_monthly_avg'
            )
            df_monthly['monthnbr'] = df_monthly['monthnbr'].str.extract(r'(\d+)').astype(int)
        else:
            df_monthly = pd.DataFrame()

        # Combine daily and monthly features
        if not df_daily.empty and not df_monthly.empty:
            df_features = df_daily.merge(
                df_monthly,
                on=['x', 'y', 'monthnbr'] + static_cols,
                how='left'
            )
        elif not df_daily.empty:
            df_features = df_daily.copy()
        elif not df_monthly.empty:
            df_features = df_monthly.copy()
        else:
            raise ValueError("No daily or monthly precipitation columns found.")

        # Add derived features
        if 'aspect' in df_features.columns:
            df_features["aspect_sin"] = np.sin(np.deg2rad(df_features["aspect"]))
            df_features["aspect_cos"] = np.cos(np.deg2rad(df_features["aspect"]))
        if 'doy' in df_features.columns:
            df_features['sin_doy'] = np.sin(2 * np.pi * df_features['doy'] / 365)
            df_features['cos_doy'] = np.cos(2 * np.pi * df_features['doy'] / 365)
        if 'prec_daily_fraction' in data_vars and 'prec_daily_avg' in df_features.columns:
            # Group by grid cell (x, y) for fraction computation
            df_features["prec_daily_fraction"] = (
                df_features["prec_daily_avg"] /
                df_features.groupby(["x", "y"])["prec_daily_avg"].transform("sum")
            )
        
        # Update feature list safely
        vars_local = data_vars.copy()
        if 'aspect' in vars_local:
            vars_local.remove('aspect')
            vars_local.extend(['aspect_sin', 'aspect_cos'])
        if 'doy' in vars_local:
            vars_local.remove('doy')
            vars_local.extend(['sin_doy', 'cos_doy'])
        
        X = df_features[vars_local].values
        chunk_predictions = pipeline.predict(X)
        if pred_col == 'EI_daily_cumsum': # Clip predictions if necessary
            chunk_predictions = np.clip(chunk_predictions, 0, 100)
        df_features['predicted_' + pred_col] = chunk_predictions

        # --- Write chunk to parquet ---
        table = pa.Table.from_pandas(df_features)
        if writer is None:
            writer = pq.ParquetWriter(
                output_file,
                table.schema
            )
        writer.write_table(table)

        # free memory
        del df_features, table
        
        """
        all_predictions.append(df_features)
    
    
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    print(len(predictions_df))
    #predictions_df.to_csv(output_file, index=False)
    predictions_df.to_parquet(output_file, index=False, engine='pyarrow')
    """
    if writer:
        writer.close()
    print(f"Predictions saved to {output_file}")

    return 
 

def main():
    
    # Load station metadata
    stations = load_stations_metadata(STATIONS_METADATA_PATH)

    # Load EI data
    ei_station_dict = load_ei_data(EI_DIR)

    # Prepare covariates
    cov_files = prepare_covariates(DATA_VARS, COVARIATES_DIR)

    # Extract features for grid cells containing stations
    gdf_labeled = prepare_grid_labeled(stations, cov_files, DATA_VARS, DEM_FILE_PATTERN)
   
    # Train model
    model_file = f"models/{MODEL_TYPE}_{MODEL_SUFFIX}_model.joblib"
    if not os.path.exists(model_file):
        # Prepare data for training: per row there is 1 EI daily for 1 grid cell + features
        df_data = prepare_training_data(gdf_labeled, DATA_VARS, ei_station_dict, PRED_COL)
        if 'aspect' in DATA_VARS:
            DATA_VARS.remove('aspect')
            DATA_VARS.extend(['aspect_sin', 'aspect_cos'])
        if 'doy' in DATA_VARS:
            DATA_VARS.remove('doy')
            DATA_VARS.extend(['sin_doy', 'cos_doy'])

        print("\nTraining model...")
        pipeline = train_model(df_data, gdf_labeled, DATA_VARS, PRED_COL, MODEL_TYPE, model_file, TUNE)
        # Evaluate on the test set
        predict_test_set(df_data, gdf_labeled, DATA_VARS, PRED_COL, model_file)


    # Inference on whole grid
    if not os.path.exists(SAVE_PATH):
        print("\nPerforming inference for all grid cells...")

        # Prepare or load data for the whole grid
        if not os.path.exists('covariates/data_grid.pkl'):
            gdf_grid = prepare_full_grid(cov_files, DATA_VARS, DEM_FILE_PATTERN)
            gdf_grid.to_pickle('covariates/data_grid.pkl')
        else:
            gdf_grid = pd.read_pickle('covariates/data_grid.pkl')
        
        # Run model
        pipeline = joblib.load(model_file)
        predict_per_grid_cell(gdf_grid, pipeline, DATA_VARS, PRED_COL, SAVE_PATH)



if __name__ == "__main__":


    # -------------------------
    # Constants and Configurations
    # -------------------------
    STATIONS_METADATA_PATH = 'stations_data/metadata/ogd-smn_meta_stations.csv'
    EI_DIR = 'stations_data/stations_EIdaily_percent'
    COVARIATES_DIR = 'covariates'
    DEM_FILE_PATTERN = 'dem'
    DATA_VARS = ['prec_daily_avg', 'prec_monthly_avg', 'elev', 'slope', 'aspect', 'doy', 'prec_daily_fraction'] 
    PRED_COL = 'EI_daily_avg' #'EI_daily_percent' #'EI_daily_cumsum' # 
    MODEL_TYPE = "rf"  # Options: "rf", "nn", "gpr
    if PRED_COL == 'EI_daily_avg':
        MODEL_SUFFIX = "absEI_tuned" # model will be named f"{MODEL_TYPE}_{MODEL_SUFFIX}_model.joblib"
    elif PRED_COL == 'EI_daily_cumsum':
        MODEL_SUFFIX = "cumEI"
    else:
        MODEL_SUFFIX = "percent_tuned"
    
    TUNE = True
    SAVE_PATH = f'grid_{PRED_COL}_pred.parquet'

    main()



# TO DO: if predict avg -> tranform to cumsum
# gpr subsampling
"""
# Subsample per station
df_sample = df_data.groupby('station_abbr', group_keys=False).apply(
    lambda x: x.sample(n=min(len(x), 150), random_state=42)
).reset_index(drop=True)
"""
# compare predicted cells to stations and use stations as instead of preds there