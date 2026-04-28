"""
Upscale long-term average daily rainfall erosivity (EI) from Swiss weather
stations to a 100 m grid over Switzerland using environmental/climate
covariates and a regression model (RF / NN / GPR).

Pipeline
--------
1. Load station metadata and per-station EI daily climatology.
2. Prepare covariate datasets (DEM, daily precip climatology, monthly precip
   min/avg/max, monthly avg temperature) from zarr sources.
3. Build a labeled grid (grid cells containing stations) and assemble a
   training table with one row per (station, DOY).
4. Train a model with an optional Optuna hyperparameter search on a spatial
   GroupKFold split; evaluate on a held-out test split.
5. Run inference over the full 100 m grid, in chunks, writing a Parquet
   dataset of predictions.

Usage
-----
    python upscale_EI.py --config config.yaml
    python upscale_EI.py --config config.yaml --date 20260419 --no-tune

A minimal YAML config looks like:

    stations_metadata_path: stations_data/metadata/ogd-smn_meta_stations.csv
    ei_dir: stations_data/stations_EIdaily_percent
    covariates_dir: covariates
    dem_file_pattern: dem
    data_vars:
      - prec_daily_avg
      - prec_monthly_avg
      - elev
      - slope
      - aspect
      - doy
      - prec_daily_fraction
    pred_col: EI_daily_avg     # EI_daily_avg | EI_daily_percent | EI_daily_cumsum
    model_type: rf              # rf | nn | gpr
    tune: true
"""

from __future__ import annotations

import argparse
import calendar
import glob
import json
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from functools import reduce
from typing import Any

import contextily as ctx
import geopandas as gpd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr
import yaml
from shapely.geometry import Point, box
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LV95_CRS = 2056                   # Swiss LV95 (EPSG:2056) — true CRS of stations and DEM
STATION_COL = "station_abbr"
STATIC_COV_SET = {"elev", "slope", "aspect"}
ID_LIKE_COLS = {STATION_COL, "x", "y", "doy", "monthnbr"}

# Maps a covariate variable name to the column prefix used after pivot.
MONTHLY_PREFIX: dict[str, str] = {
    "prec_monthly_avg": "monthly_avg_precip_",
    "prec_monthly_min": "monthly_min_precip_",
    "prec_monthly_max": "monthly_max_precip_",
    "monthly_avg_temp": "monthly_avg_temp_",
}
DAILY_PREFIX = {"prec_daily_avg": "daily_avg_precip_"}

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Runtime configuration loaded from YAML + CLI overrides."""

    stations_metadata_path: str
    ei_dir: str
    covariates_dir: str
    dem_file_pattern: str
    data_vars: list[str]
    pred_col: str = "EI_daily_avg"
    model_type: str = "rf"
    tune: bool = True
    models_dir: str = "models"
    results_dir: str = "results"
    predictions_dir: str = "."
    chunk_size: int = 5000
    optuna_trials: int = 20
    train_frac: float = 0.70
    val_frac: float = 0.15
    split_grid_size: int = 25_000
    random_seed: int = 42
    date_str: str | None = None    # None -> pick latest existing model or today
    version_tag: str | None = None  # e.g. "v2", "exp1" — disambiguates runs on the same date

    # Derived
    model_suffix: str = field(init=False)

    def __post_init__(self) -> None:
        if self.pred_col == "EI_daily_avg":
            self.model_suffix = "absEI_tuned" if self.tune else "absEI"
        elif self.pred_col == "EI_daily_cumsum":
            self.model_suffix = "cumEI"
        else:
            self.model_suffix = "percent_tuned" if self.tune else "percent"
        if self.version_tag:
            self.model_suffix = f"{self.model_suffix}_{self.version_tag}"

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw)

    def model_file(self) -> str:
        return os.path.join(
            self.models_dir,
            f"{self.model_type}_{self.model_suffix}_model_{self.date_str}.joblib",
        )

    def vars_file(self) -> str:
        return self.model_file().replace(".joblib", "_data_vars.json")

    def data_grid_pkl(self) -> str:
        tag = f"_{self.version_tag}" if self.version_tag else ""
        return os.path.join(self.covariates_dir, f"data_grid_{self.date_str}{tag}.pkl")

    def predictions_path(self) -> str:
        tag = f"_{self.version_tag}" if self.version_tag else ""
        os.makedirs('predictions', exist_ok=True)
        return os.path.join(
            self.predictions_dir,
            f"predictions/grid_{self.pred_col}_pred_{self.date_str}{tag}.parquet",
        )


# ---------------------------------------------------------------------------
# Feature name helpers (single source of truth for derived features)
# ---------------------------------------------------------------------------
def derive_feature_names(data_vars: list[str]) -> list[str]:
    """
    Translate a raw list of requested data_vars into the exact list of model
    feature columns that training and inference should both use.

    - ``aspect``  -> ``aspect_sin``, ``aspect_cos``
    - ``doy``     -> ``sin_doy``, ``cos_doy``
    - Everything else passes through unchanged.

    This function is the single authority on the model's feature schema: use
    it at both train time (to select columns out of ``df_data`` and save them
    alongside the model) and at inference time (to verify what was saved and
    decide what to select out of the grid features table).
    """
    features: list[str] = []
    for var in data_vars:
        if var == "aspect":
            features.extend(["aspect_sin", "aspect_cos"])
        elif var == "doy":
            features.extend(["sin_doy", "cos_doy"])
        else:
            features.append(var)
    # preserve order, drop accidental duplicates
    seen: set[str] = set()
    unique_features: list[str] = []
    for var in features:
        if var not in seen:
            unique_features.append(var)
            seen.add(var)
    return unique_features


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_stations_metadata(metadata_path: str) -> pd.DataFrame:
    """Load the subset of station metadata columns used downstream."""
    keep = [
        "station_abbr",
        "station_name",
        "station_height_masl",
        "station_coordinates_lv95_east",
        "station_coordinates_lv95_north",
        "station_coordinates_wgs84_lat",
        "station_coordinates_wgs84_lon",
        "station_exposition_en",
    ]
    stations = pd.read_csv(metadata_path, delimiter=";", encoding="latin1")
    return stations[keep]


def load_ei_data(ei_dir: str) -> dict[str, pd.DataFrame]:
    """Load per-station EI climatology CSVs. Filename suffix carries the station code."""
    ei_data = {}
    for fname in os.listdir(ei_dir):
        if not fname.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(ei_dir, fname))
        station = fname.split("_")[-1].split(".csv")[0]
        ei_data[station] = df
    return ei_data


# ---------------------------------------------------------------------------
# Covariate preparation (zarr -> climatology zarr)
# ---------------------------------------------------------------------------
def _write_zarr_atomic(ds: xr.Dataset, path: str) -> None:
    """
    Write a zarr store safely: write to ``path + '.tmp'`` first, then rename.

    Avoids leaving behind a half-written store if the process crashes mid-write.
    """
    tmp = path + ".tmp"
    if os.path.exists(tmp):
        import shutil
        shutil.rmtree(tmp)
    ds.to_zarr(tmp, mode="w")
    if os.path.exists(path):
        import shutil
        shutil.rmtree(path)
    os.rename(tmp, path)


def _list_precip_zarrs(prec_path: str) -> list[str]:
    return [
        os.path.join(prec_path, f)
        for f in os.listdir(prec_path)
        if f.endswith(".zarr") and "MeteoSwiss" in f
    ]


def prep_daily_prec(covariates_dir: str) -> str:
    """
    Build climatological daily mean precipitation on DOY 1–365.

    Feb 29 is dropped and post-Feb days in leap years are shifted back by one
    so the resulting DOY axis is dense 1..365.
    """
    prec_path = os.path.join(covariates_dir, "precip")
    out_path = os.path.join(prec_path, "daily_avg.zarr")
    if os.path.exists(out_path):
        return out_path

    yearly_datasets = []
    for zarr_path in _list_precip_zarrs(prec_path):
        ds = xr.open_zarr(zarr_path)
        time = ds.time
        is_feb29 = (time.dt.month == 2) & (time.dt.day == 29)
        ds = ds.sel(time=~is_feb29)

        time = ds.time
        shift_mask = time.dt.is_leap_year & (time.dt.dayofyear > 59)
        doy = xr.where(shift_mask, time.dt.dayofyear - 1, time.dt.dayofyear)
        ds = ds.assign_coords(doy=("time", doy.data))
        yearly_datasets.append(ds)

    all_years = xr.concat(yearly_datasets, dim="time")
    daily_avg = all_years.groupby("doy").mean("time").rename({"doy": "time"})
    _write_zarr_atomic(daily_avg, out_path)
    print(f"Saved daily mean precip climatology -> {out_path}")
    return out_path


def _prep_monthly(
    covariates_dir: str,
    subdir: str,
    out_name: str,
    reducer: str,
    require_meteoswiss: bool,
) -> str:
    """
    Shared helper for monthly precip (avg/min/max) and monthly avg temperature.

    ``reducer`` is the string passed to ``Dataset.resample(time='ME').{reducer}()``.
    """
    data_path = os.path.join(covariates_dir, subdir)
    out_path = os.path.join(data_path, out_name)
    if os.path.exists(out_path):
        return out_path

    if require_meteoswiss:
        zarr_files = _list_precip_zarrs(data_path)
    else:
        zarr_files = [
            os.path.join(data_path, f)
            for f in os.listdir(data_path)
            if f.endswith(".zarr")
        ]

    yearly_datasets = []
    for zarr_path in zarr_files:
        ds = xr.open_zarr(zarr_path)
        resampled = getattr(ds.resample(time="ME"), reducer)()
        yearly_datasets.append(resampled)

    all_years = xr.concat(yearly_datasets, dim="time")
    climatology = all_years.groupby("time.month").mean().rename({"month": "time"})
    _write_zarr_atomic(climatology, out_path)
    print(f"Saved {out_name} -> {out_path}")
    return out_path


def prep_monthly_prec(covariates_dir: str) -> str:
    return _prep_monthly(covariates_dir, "precip", "monthly_avg.zarr", "sum", True)


def prep_monthly_min_prec(covariates_dir: str) -> str:
    return _prep_monthly(covariates_dir, "precip", "monthly_min.zarr", "min", True)


def prep_monthly_max_prec(covariates_dir: str) -> str:
    return _prep_monthly(covariates_dir, "precip", "monthly_max.zarr", "max", True)


def prep_monthly_temp(covariates_dir: str) -> str:
    return _prep_monthly(covariates_dir, "temp", "monthly_avg_temp.zarr", "mean", False)


def prepare_covariates(data_vars: list[str], covariates_dir: str) -> list[str]:
    """
    Build/return paths to every covariate zarr needed to service ``data_vars``.

    The DEM (which carries elev/slope/aspect) is included exactly once if any
    topographic feature is requested.
    """
    files: list[str] = []

    if any(v in data_vars for v in STATIC_COV_SET):
        topo_path = os.path.join(covariates_dir, "dem_100m.zarr")
        ds = xr.open_zarr(topo_path)
        want_to_zarr = {"elev": "height", "slope": "slope", "aspect": "aspect"}
        required = [want_to_zarr[v] for v in STATIC_COV_SET if v in data_vars]
        missing = [v for v in required if v not in ds.data_vars]
        if missing:
            raise ValueError(f"DEM zarr is missing required variables: {missing}")
        files.append(topo_path)

    if "prec_daily_avg" in data_vars:
        files.append(prep_daily_prec(covariates_dir))
    if "prec_monthly_avg" in data_vars:
        files.append(prep_monthly_prec(covariates_dir))
    if "prec_monthly_min" in data_vars:
        files.append(prep_monthly_min_prec(covariates_dir))
    if "prec_monthly_max" in data_vars:
        files.append(prep_monthly_max_prec(covariates_dir))
    if "monthly_avg_temp" in data_vars:
        files.append(prep_monthly_temp(covariates_dir))

    return files


# ---------------------------------------------------------------------------
# Feature extraction from gridded covariates
# ---------------------------------------------------------------------------
# Maps zarr path tokens -> output column prefix in the wide features table.
_EXTRACT_VAR_PREFIX: list[tuple[str, str]] = [
    ("precip/daily_avg",   "daily_avg_precip"),
    ("precip/monthly_avg", "monthly_avg_precip"),
    ("precip/monthly_min", "monthly_min_precip"),
    ("precip/monthly_max", "monthly_max_precip"),
    ("temp/monthly_avg_temp", "monthly_avg_temp"),
]


def _var_prefix_for_path(data_path: str) -> str:
    """Return the column prefix to use for a given covariate zarr path."""
    for token, prefix in _EXTRACT_VAR_PREFIX:
        # Normalise separators so this works on Windows-style paths too.
        if token.replace("/", os.sep) in data_path or token in data_path:
            return prefix
    raise ValueError(
        f"Cannot determine feature name for covariate path: {data_path!r}. "
        "Update _EXTRACT_VAR_PREFIX if adding a new covariate."
    )


def extract_features(data_path: str, gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Interpolate a covariate zarr onto the points in ``gdf`` and pivot the time
    dimension into one column per timestep.

    The time coordinate value (DOY for daily climatologies, month number for
    monthly climatologies) is used directly as the column suffix, so that
    training and inference always produce the exact same column names.
    """
    var_prefix = _var_prefix_for_path(data_path)
    ds = xr.open_zarr(data_path)

    coords = dict(x=("points", gdf["x"].values), y=("points", gdf["y"].values))
    var_name = list(ds.data_vars)[0]
    data_array = ds[var_name].interp(coords)
    df = data_array.to_dataframe().reset_index()
    df = df.pivot(index="points", columns="time", values=var_name)

    df["x"] = gdf["x"].values
    df["y"] = gdf["y"].values
    time_cols = [c for c in df.columns if c not in ("x", "y")]
    df = df[["x", "y"] + time_cols]
    df = df.rename(columns={c: f"{var_prefix}_{c}" for c in time_cols})
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def make_model(model_type: str = "rf") -> Pipeline:
    """Build a scikit-learn pipeline for the requested model family."""
    if model_type == "gpr":
        kernel = C(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3)
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", GaussianProcessRegressor(
                kernel=kernel, normalize_y=True, n_restarts_optimizer=2,
            )),
        ])

    if model_type == "rf":
        return Pipeline([
            ("model", RandomForestRegressor(
                n_estimators=300, max_depth=None, min_samples_leaf=2,
                n_jobs=1, random_state=42,
            )),
        ])

    if model_type == "nn":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(64, 64, 32), activation="relu",
                learning_rate_init=0.001, max_iter=500, early_stopping=True,
                random_state=42,
            )),
        ])

    raise ValueError("model_type must be one of 'gpr', 'rf', 'nn'")


# ---------------------------------------------------------------------------
# Labeled grid (station cells) and full inference grid
# ---------------------------------------------------------------------------
def _open_dem_as_gdf(cov_files: list[str], dem_file_pattern: str, data_vars: list[str]) -> gpd.GeoDataFrame:
    """Build the 100 m DEM GeoDataFrame in LV95, with elev/slope/aspect columns as requested."""
    dem_file = next(f for f in cov_files if dem_file_pattern in f)
    dem_grid = xr.open_zarr(dem_file)
    dem_df = dem_grid.to_dataframe().reset_index().dropna()
    geometry = [box(x, y - 100, x + 100, y) for x, y in zip(dem_df["x"], dem_df["y"])]
    dem_gdf = gpd.GeoDataFrame(dem_df, geometry=geometry, crs=32632)
    if "elev" in data_vars and "height" in dem_gdf.columns:
        dem_gdf = dem_gdf.rename(columns={"height": "elev"})
    return dem_gdf


def _attach_non_dem_features(
    gdf: gpd.GeoDataFrame, cov_files: list[str], drop_na: bool
) -> gpd.GeoDataFrame:
    """Merge every non-DEM covariate onto the grid, optionally dropping NAs."""
    for f in cov_files:
        if "dem" in f:
            continue
        df_file = extract_features(f, gdf)
        if drop_na:
            df_file = df_file.dropna()
        gdf = gdf.merge(df_file, on=["x", "y"], how="left")
        if drop_na:
            # The precip grid is larger than the DEM grid, so a left merge can
            # still land NaNs in edge cells — drop those too.
            gdf = gdf.dropna()
    return gdf


def prepare_grid_labeled(
    stations: pd.DataFrame,
    cov_files: list[str],
    data_vars: list[str],
    dem_file_pattern: str,
) -> gpd.GeoDataFrame:
    """Return the DEM grid cells that contain a station, with all covariates attached."""
    geometry = [
        Point(xy) for xy in zip(
            stations["station_coordinates_lv95_east"],
            stations["station_coordinates_lv95_north"],
        )
    ]
    stations_gdf = gpd.GeoDataFrame(stations, geometry=geometry, crs=LV95_CRS)

    dem_gdf = _open_dem_as_gdf(cov_files, dem_file_pattern, data_vars)
    gdf_labeled = dem_gdf.sjoin(stations_gdf.to_crs(dem_gdf.crs))
    gdf_labeled = _attach_non_dem_features(gdf_labeled, cov_files, drop_na=False)

    na_counts = gdf_labeled.isna().sum()
    na_counts = na_counts[na_counts > 0]
    if na_counts.empty:
        print("No missing values in the labeled grid.")
    else:
        print(f"Missing values in labeled grid:\n{na_counts}")

    return gdf_labeled


def prepare_full_grid(
    cov_files: list[str], data_vars: list[str], dem_file_pattern: str
) -> gpd.GeoDataFrame:
    """Return the full DEM grid with all covariates attached, NA cells dropped."""
    print("\nBuilding full inference grid...")
    dem_gdf = _open_dem_as_gdf(cov_files, dem_file_pattern, data_vars)
    dem_gdf = _attach_non_dem_features(dem_gdf, cov_files, drop_na=True)

    na_counts = dem_gdf.isna().sum()
    na_counts = na_counts[na_counts > 0]
    if na_counts.empty:
        print("No missing values in the full inference grid.")
    else:
        print(f"Missing values in full grid:\n{na_counts}")

    return dem_gdf


# ---------------------------------------------------------------------------
# Core feature assembly (shared by training and inference)
# ---------------------------------------------------------------------------
def _melt_daily(
    gdf: pd.DataFrame, id_vars: list[str], data_vars: list[str]
) -> pd.DataFrame:
    """Melt the wide ``daily_avg_precip_<doy>`` columns into long form on ``doy``."""
    if "prec_daily_avg" not in data_vars:
        return pd.DataFrame()
    prefix = DAILY_PREFIX["prec_daily_avg"]
    cols = [c for c in gdf.columns if c.startswith(prefix)]
    if not cols:
        return pd.DataFrame()

    df = gdf.melt(
        id_vars=id_vars, value_vars=cols,
        var_name="doy", value_name="prec_daily_avg",
    )
    df["doy"] = df["doy"].str.extract(r"(\d+)").astype("Int64").astype(int)
    df["monthnbr"] = pd.to_datetime(
        df["doy"].astype(str).str.zfill(3), format="%j", errors="coerce",
    ).dt.month
    return df


def _melt_monthly(
    gdf: pd.DataFrame, id_vars: list[str], data_vars: list[str]
) -> pd.DataFrame:
    """
    Melt each requested monthly covariate independently, then merge on
    ``id_vars + ['monthnbr']`` so each covariate gets its own column.
    """
    monthly_frames: list[pd.DataFrame] = []
    for var, prefix in MONTHLY_PREFIX.items():
        if var not in data_vars:
            continue
        cols = [c for c in gdf.columns if c.startswith(prefix)]
        if not cols:
            continue
        melted = gdf.melt(
            id_vars=id_vars, value_vars=cols,
            var_name="monthnbr", value_name=var,
        )
        melted["monthnbr"] = melted["monthnbr"].str.extract(r"(\d+)").astype(int)
        monthly_frames.append(melted)

    if not monthly_frames:
        return pd.DataFrame()
    # Merge each monthly covariate on (id_vars + month) so each gets its own column.
    return reduce(
        lambda left, right: left.merge(right, on=id_vars + ["monthnbr"], how="outer"),
        monthly_frames,
    )


def _add_derived_features(
    df: pd.DataFrame, data_vars: list[str], fraction_group_cols: list[str]
) -> pd.DataFrame:
    """
    Add the cyclic / ratio features that both training and inference need.

    ``fraction_group_cols`` is the grouping used for ``prec_daily_fraction``:
    ``['station_abbr']`` at train time (one station = one grid cell) and
    ``['x', 'y']`` at inference time (grouping by grid cell).
    """
    if "aspect" in data_vars and "aspect" in df.columns:
        df["aspect_sin"] = np.sin(np.deg2rad(df["aspect"]))
        df["aspect_cos"] = np.cos(np.deg2rad(df["aspect"]))

    if "doy" in data_vars and "doy" in df.columns:
        df["sin_doy"] = np.sin(2 * np.pi * df["doy"] / 365)
        df["cos_doy"] = np.cos(2 * np.pi * df["doy"] / 365)

    if "prec_daily_fraction" in data_vars:
        if "prec_daily_avg" not in df.columns:
            raise ValueError(
                "prec_daily_fraction requires prec_daily_avg to be present."
            )
        totals = df.groupby(fraction_group_cols)["prec_daily_avg"].transform("sum")
        df["prec_daily_fraction"] = df["prec_daily_avg"] / totals.replace(0, np.nan)

    return df


def prepare_training_data(
    gdf_labeled: gpd.GeoDataFrame,
    data_vars: list[str],
    ei_station_dict: dict[str, pd.DataFrame],
    pred_col: str,
) -> pd.DataFrame:
    """
    Assemble the training table: one row per (station, grid cell, DOY).

    Produces only the columns implied by ``data_vars`` (plus ``station_abbr``,
    ``x``, ``y``, ``doy``, ``monthnbr``, and the target ``pred_col``). In
    particular, monthly covariates the caller did not request are **not**
    created — this is what the old ``.where(...)`` path got wrong.
    """
    static_cols = [c for c in data_vars if c in STATIC_COV_SET]
    id_vars = [STATION_COL, "x", "y"] + static_cols

    df_daily = _melt_daily(gdf_labeled, id_vars, data_vars)
    df_monthly = _melt_monthly(gdf_labeled, id_vars, data_vars)

    if not df_daily.empty and not df_monthly.empty:
        df_features = df_daily.merge(
            df_monthly, on=id_vars + ["monthnbr"], how="left",
        )
    elif not df_daily.empty:
        df_features = df_daily
    elif not df_monthly.empty:
        # Monthly-only: expand each month row to one row per DOY of that month.
        df_features = _expand_months_to_doys(df_monthly)
    else:
        raise ValueError("No daily or monthly precipitation columns found.")

    # Attach EI target per (station, DOY).
    ei_rows = []
    for station, df in ei_station_dict.items():
        tmp = df[["doy", pred_col]].copy()
        tmp[STATION_COL] = station
        ei_rows.append(tmp)
    ei_all = pd.concat(ei_rows, ignore_index=True)

    df_data = df_features.merge(ei_all, on=[STATION_COL, "doy"], how="inner")
    df_data = _add_derived_features(
        df_data, data_vars, fraction_group_cols=[STATION_COL]
    )
    return df_data


def _expand_months_to_doys(df_monthly: pd.DataFrame) -> pd.DataFrame:
    """Expand a per-month row into one row per DOY within that month (non-leap year)."""
    # Vectorised version of the old per-row loop.
    ref_year = 2021  # any non-leap year works
    days_in_month = {m: calendar.monthrange(ref_year, m)[1] for m in range(1, 13)}
    first_doy = {
        m: pd.Timestamp(year=ref_year, month=m, day=1).dayofyear for m in range(1, 13)
    }

    df_monthly = df_monthly.copy()
    df_monthly["ndays"] = df_monthly["monthnbr"].map(days_in_month)
    df_monthly["first_doy"] = df_monthly["monthnbr"].map(first_doy)

    expanded = df_monthly.loc[df_monthly.index.repeat(df_monthly["ndays"])].copy()
    expanded["offset"] = expanded.groupby(level=0).cumcount()
    expanded["doy"] = expanded["first_doy"] + expanded["offset"]
    return expanded.drop(columns=["ndays", "first_doy", "offset"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Spatial splitting
# ---------------------------------------------------------------------------
def split_stations(
    gdf: pd.DataFrame,
    station_col: str,
    x_col: str,
    y_col: str,
    frac_train: float,
    frac_val: float,
    grid_size: int = 25_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Split stations by an overlaid grid so that whole stations stay in one split.

    For Switzerland (~220 km × ~348 km) a ``grid_size`` of 25–30 km gives ~50–60
    cells, which is enough to approximate a spatial hold-out.
    """
    gdf = gdf.copy()
    stations = gdf[[station_col, x_col, y_col]].drop_duplicates()
    stations["grid_x"] = (stations[x_col] // grid_size).astype(int)
    stations["grid_y"] = (stations[y_col] // grid_size).astype(int)

    grid_cells = (
        stations[["grid_x", "grid_y"]]
        .drop_duplicates()
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
    n_cells = len(grid_cells)
    n_train = int(frac_train * n_cells)
    n_val = int(frac_val * n_cells)
    print(
        f"Splitting on {n_cells} grid cells: {n_train} train, {n_val} val, "
        f"{n_cells - n_train - n_val} test"
    )

    grid_cells["split"] = "test"
    grid_cells.loc[:n_train - 1, "split"] = "train"
    grid_cells.loc[n_train:n_train + n_val - 1, "split"] = "val"

    stations = stations.merge(grid_cells, on=["grid_x", "grid_y"], how="left")
    return gdf.merge(stations[[station_col, "split"]], on=station_col, how="left")


# ---------------------------------------------------------------------------
# Optuna + training
# ---------------------------------------------------------------------------
def objective(
    trial: optuna.trial.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    model_type: str,
) -> float:
    """Cross-validated RMSE objective for Optuna (GroupKFold by station)."""
    if model_type == "rf":
        pipeline = make_model("rf")
        pipeline.set_params(
            model__n_estimators=trial.suggest_int("n_estimators", 100, 500),
            model__max_depth=trial.suggest_int("max_depth", 10, 50),
            model__min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        )
    elif model_type == "nn":
        n_layers = trial.suggest_int("n_layers", 1, 3)
        layer_sizes = tuple(
            trial.suggest_int(f"n_units_l{i}", 16, 256, log=True) for i in range(n_layers)
        )
        pipeline = make_model("nn")
        pipeline.set_params(
            model__hidden_layer_sizes=layer_sizes,
            model__learning_rate_init=trial.suggest_float(
                "learning_rate_init", 1e-4, 1e-2, log=True
            ),
            model__max_iter=trial.suggest_int("max_iter", 500, 2000),
        )
    else:
        raise ValueError(f"Tuning not supported for model_type={model_type!r}")

    spatial_cv = GroupKFold(n_splits=5)
    rmses = []
    for train_idx, val_idx in spatial_cv.split(X_train, y_train, groups_train):
        pipeline.fit(X_train[train_idx], y_train[train_idx])
        pred = pipeline.predict(X_train[val_idx])
        rmses.append(np.sqrt(mean_squared_error(y_train[val_idx], pred)))
    return float(np.mean(rmses))


def _plot_station_splits(gdf_labeled: gpd.GeoDataFrame, out_path: str) -> None:
    """Save a quick map of which stations fell into train/val/test."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    stations_pts = gpd.GeoDataFrame(
        {"split": gdf_labeled["split"].values},
        geometry=[
            Point(xy) for xy in zip(
                gdf_labeled["station_coordinates_lv95_east"],
                gdf_labeled["station_coordinates_lv95_north"],
            )
        ],
        crs=LV95_CRS,
    )
    fig, ax = plt.subplots(figsize=(10, 10))
    stations_pts.plot(ax=ax, column="split", markersize=50, edgecolor="black", legend=True, cmap="Accent")
    ctx.add_basemap(ax, source=ctx.providers.SwissFederalGeoportal.NationalMapColor, crs=stations_pts.crs)
    ax.set_axis_off()
    plt.savefig(out_path)
    plt.close(fig)


def train_model(
    df_data: pd.DataFrame,
    gdf_labeled: gpd.GeoDataFrame,
    feature_names: list[str],
    cfg: Config,
) -> tuple[Pipeline, pd.DataFrame]:
    """
    Fit the final model.

    Returns the trained pipeline and ``gdf_labeled`` with a ``split`` column
    attached, so downstream test-set evaluation can reuse the exact same split
    without re-shuffling.
    """
    gdf_labeled = split_stations(
        gdf_labeled,
        STATION_COL,
        "station_coordinates_lv95_east",
        "station_coordinates_lv95_north",
        cfg.train_frac, cfg.val_frac,
        grid_size=cfg.split_grid_size, seed=cfg.random_seed,
    )
    splits = gdf_labeled[[STATION_COL, "split"]].drop_duplicates()
    df_data = df_data.merge(splits, on=STATION_COL, how="left")

    split_plot = os.path.join(cfg.results_dir, "station_train_split.png")
    if not os.path.exists(split_plot):
        _plot_station_splits(gdf_labeled, split_plot)

    train_df = df_data[df_data["split"] == "train"]
    val_df = df_data[df_data["split"] == "val"]

    X_train = train_df[feature_names].values
    y_train = train_df[cfg.pred_col].values
    groups_train = train_df[STATION_COL].values

    if cfg.tune and cfg.model_type in {"rf", "nn"}:
        print("\nRunning Optuna hyperparameter tuning...")
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda t: objective(t, X_train, y_train, groups_train, cfg.model_type),
            n_trials=cfg.optuna_trials,
        )
        best = study.best_params
        if "n_layers" in best:
            n_layers = best.pop("n_layers")
            layers = tuple(best.pop(f"n_units_l{i}") for i in range(n_layers))
            best["hidden_layer_sizes"] = layers
        print(f"\nBest hyperparameters: {best}")
        pipeline = make_model(cfg.model_type)
        pipeline.set_params(**{f"model__{k}": v for k, v in best.items()})
    else:
        print("\nTraining with default hyperparameters...")
        pipeline = make_model(cfg.model_type)

    pipeline.fit(X_train, y_train)

    # Quick validation sanity check
    X_val = val_df[feature_names].values
    y_val = val_df[cfg.pred_col].values
    y_val_pred = pipeline.predict(X_val)
    if cfg.pred_col == "EI_daily_cumsum":
        y_val_pred = np.clip(y_val_pred, 0, 100)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
    r2_val = r2_score(y_val, y_val_pred)
    print(f"\nValidation RMSE: {rmse_val:.3f}, R²: {r2_val:.3f}")

    os.makedirs(cfg.models_dir, exist_ok=True)
    joblib.dump(pipeline, cfg.model_file())
    with open(cfg.vars_file(), "w") as f:
        json.dump(feature_names, f)
    print(f"Saved model -> {cfg.model_file()}")
    print(f"Saved feature list -> {cfg.vars_file()}")

    return pipeline, gdf_labeled


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def EI_to_cumul(df: pd.DataFrame) -> pd.DataFrame:
    """Derive daily percent + cumulative percent curves per station, for both obs and pred."""
    df = df.copy()
    if "doy" in df.columns:
        df = df.sort_values([STATION_COL, "doy"])

    for prefix in ("predicted_", ""):
        avg_col = f"{prefix}EI_daily_avg"
        pct_col = f"{prefix}EI_daily_percent"
        csum_col = f"{prefix}EI_daily_cumsum"

        if avg_col in df.columns:
            total = df.groupby(STATION_COL)[avg_col].transform("sum")
            df.loc[:, pct_col] = np.where(total == 0, 0, df[avg_col] / total * 100)
            df.loc[:, csum_col] = df.groupby(STATION_COL)[pct_col].cumsum()

        elif pct_col in df.columns:
            df.loc[:, pct_col] = np.maximum(df[pct_col], 0)
            group_sums = df.groupby(STATION_COL)[pct_col].transform("sum").replace(0, np.nan)
            df.loc[:, pct_col] = (df[pct_col] / group_sums) * 100
            df.loc[:, pct_col] = df[pct_col].fillna(0)
            df.loc[:, csum_col] = df.groupby(STATION_COL)[pct_col].cumsum()

        else:
            print(f"Skipping {prefix}: no avg or percent column found")
    return df


def compute_kge_nse(obs: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """Return (KGE, NSE) for a single obs/pred curve."""
    obs = np.asarray(obs)
    pred = np.asarray(pred)
    correlation = np.corrcoef(obs, pred)[0, 1]
    alpha = np.std(pred) / np.std(obs)
    beta = np.mean(pred) / np.mean(obs)
    kge = 1 - np.sqrt((correlation - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    nse = 1 - np.sum((obs - pred) ** 2) / np.sum((obs - np.mean(obs)) ** 2)
    return float(kge), float(nse)


def evaluate_curves_per_station(
    df: pd.DataFrame,
    obs_col: str = "EI_daily_cumsum",
    pred_col: str = "predicted_EI_daily_cumsum",
    station_col: str = STATION_COL,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compute KGE/NSE per station, plus a summary dict."""
    results = []
    for station in df[station_col].unique():
        station_df = df[df[station_col] == station].sort_values("doy")
        kge, nse = compute_kge_nse(station_df[obs_col].values, station_df[pred_col].values)
        results.append({"station": station, "KGE": kge, "NSE": nse})
    metrics_df = pd.DataFrame(results)
    summary = {
        "mean_KGE": metrics_df["KGE"].mean(),
        "std_KGE": metrics_df["KGE"].std(),
        "mean_NSE": metrics_df["NSE"].mean(),
        "std_NSE": metrics_df["NSE"].std(),
    }
    return metrics_df, summary


def plot_test_predictions(
    df: pd.DataFrame,
    pred_col: str,
    save_path: str,
    num_stations: int = 10,
    random_seed: int = 42,
) -> None:
    """Overlay predicted vs actual curves for a random sample of stations."""
    stations = df[STATION_COL].unique()
    rng = np.random.default_rng(random_seed)
    sampled = rng.choice(stations, size=min(num_stations, len(stations)), replace=False)

    cmap = plt.get_cmap("tab20", len(sampled))
    colors = {station: cmap(i) for i, station in enumerate(sampled)}

    df_sampled = df[df[STATION_COL].isin(sampled)]
    plt.figure(figsize=(12, 8))
    for station in sampled:
        station_df = df_sampled[df_sampled[STATION_COL] == station]
        color = colors[station]
        plt.plot(station_df["doy"], station_df[pred_col], label=f"Actual ({station})", color=color, alpha=0.7)
        plt.plot(
            station_df["doy"], station_df[f"predicted_{pred_col}"],
            label=f"Predicted ({station})", color=color, linestyle="--", alpha=0.7,
        )
    plt.title("Predictions vs actuals (random station sample)", fontsize=16)
    plt.xlabel("Day of Year")
    plt.ylabel("EI daily cumulative sum")
    plt.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot -> {save_path}")


def predict_test_set(
    df_data: pd.DataFrame,
    gdf_labeled: gpd.GeoDataFrame,
    feature_names: list[str],
    cfg: Config,
) -> pd.DataFrame:
    """
    Evaluate the saved model on the held-out test split.

    Assumes ``gdf_labeled`` already carries the ``split`` column from training
    (so the same stations end up in the test set both times).
    """
    print("\nEvaluating on the test set...")

    if "split" not in gdf_labeled.columns:
        raise ValueError("gdf_labeled must carry a 'split' column from training.")
    if "split" not in df_data.columns:
        splits = gdf_labeled[[STATION_COL, "split"]].drop_duplicates()
        df_data = df_data.merge(splits, on=STATION_COL, how="left")

    test_df = df_data[df_data["split"] == "test"].copy()
    X_test = test_df[feature_names].values
    y_test = test_df[cfg.pred_col].values

    pipeline = joblib.load(cfg.model_file())
    y_pred = pipeline.predict(X_test)
    if cfg.pred_col == "EI_daily_cumsum":
        y_pred = np.clip(y_pred, 0, 100)
    test_df["predicted_" + cfg.pred_col] = y_pred

    def _report(label: str, y_true: np.ndarray, y_hat: np.ndarray) -> None:
        rmse = np.sqrt(mean_squared_error(y_true, y_hat))
        mae = mean_absolute_error(y_true, y_hat)
        r2 = r2_score(y_true, y_hat)
        nrmse = rmse / (y_true.max() - y_true.min())
        print(f"\n===== {label} =====")
        print(f"RMSE:  {rmse:.3f}")
        print(f"MAE:   {mae:.3f}")
        print(f"NRMSE: {nrmse:.3f}")
        print(f"R²:    {r2:.3f}")

    _report("Test set (raw target)", test_df[cfg.pred_col].values, y_pred)

    if cfg.pred_col != "EI_daily_cumsum":
        test_df = EI_to_cumul(test_df)

    _report(
        "Test set (cumulative EI)",
        test_df["EI_daily_cumsum"].values,
        test_df["predicted_EI_daily_cumsum"].values,
    )

    _, summary = evaluate_curves_per_station(test_df)
    print("\n===== KGE / NSE (per station, cumulative curve) =====")
    for k, v in summary.items():
        print(f"{k}: {v:.3f}")

    plot_path = os.path.join(
        cfg.results_dir,
        f"predicted_test_{os.path.basename(cfg.model_file()).split('.joblib')[0]}.png",
    )
    plot_test_predictions(
        test_df, "EI_daily_cumsum", plot_path,
        num_stations=test_df[STATION_COL].nunique(),
        random_seed=cfg.random_seed,
    )
    return test_df


# ---------------------------------------------------------------------------
# Full-grid inference
# ---------------------------------------------------------------------------
def predict_per_grid_cell(
    dem_gdf: gpd.GeoDataFrame,
    pipeline: Pipeline,
    data_vars: list[str],
    feature_names: list[str],
    cfg: Config,
) -> None:
    """
    Run inference over the full grid, writing a chunked Parquet of predictions.

    ``feature_names`` is the exact column list the model was trained on (the
    output of :func:`derive_feature_names`). The chunk DataFrame is required
    to contain every one of these after feature derivation; a mismatch is a
    hard error rather than a silent reindex-with-NaN.
    """
    output_file = cfg.predictions_path()
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    # Static columns to carry through the per-chunk melt.
    static_cols = [c for c in data_vars if c in STATIC_COV_SET]
    id_vars = ["x", "y"] + static_cols

    total = len(dem_gdf)
    n_chunks = (total + cfg.chunk_size - 1) // cfg.chunk_size
    print(f"\nPredicting {total} grid cells in {n_chunks} chunks of {cfg.chunk_size}...")

    writer: pq.ParquetWriter | None = None
    try:
        for i in range(n_chunks):
            start = i * cfg.chunk_size
            end = min(start + cfg.chunk_size, total)
            print(f"  chunk {i + 1}/{n_chunks}")

            chunk = dem_gdf.iloc[start:end]

            df_daily = _melt_daily(chunk, id_vars, data_vars)
            df_monthly = _melt_monthly(chunk, id_vars, data_vars)

            if not df_daily.empty and not df_monthly.empty:
                df_features = df_daily.merge(
                    df_monthly, on=id_vars + ["monthnbr"], how="left",
                )
            elif not df_daily.empty:
                df_features = df_daily
            elif not df_monthly.empty:
                df_features = _expand_months_to_doys(df_monthly)
            else:
                raise ValueError("No daily or monthly precipitation columns found.")

            df_features = _add_derived_features(
                df_features, data_vars, fraction_group_cols=["x", "y"]
            )

            missing = [c for c in feature_names if c not in df_features.columns]
            if missing:
                raise KeyError(
                    f"Feature columns expected by the model are missing from the "
                    f"inference table: {missing}. "
                    f"data_vars={data_vars}, feature_names={feature_names}."
                )

            na_counts = df_features[feature_names].isna().sum()
            if na_counts.sum() > 0:
                print(f"    missing values per feature:\n{na_counts[na_counts > 0]}")

            X = df_features[feature_names].values
            preds = pipeline.predict(X)
            if cfg.pred_col == "EI_daily_cumsum":
                preds = np.clip(preds, 0, 100)
            df_features[f"predicted_{cfg.pred_col}"] = preds

            table = pa.Table.from_pandas(df_features)
            if writer is None:
                writer = pq.ParquetWriter(output_file, table.schema)
            writer.write_table(table)

            del df_features, table
    finally:
        if writer is not None:
            writer.close()

    print(f"Predictions saved -> {output_file}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def resolve_date_str(cfg: Config) -> str:
    """Pick an explicit date, or reuse the latest model on disk, or use today."""
    if cfg.date_str is not None:
        return cfg.date_str
    today = datetime.now().strftime("%Y%m%d")
    pattern = os.path.join(
        cfg.models_dir,
        f"{cfg.model_type}_{cfg.model_suffix}_model_*.joblib",
    )
    existing = sorted(glob.glob(pattern))
    if not existing:
        return today
    return os.path.basename(existing[-1]).split("_model_")[1].replace(".joblib", "")


def load_trained_feature_names(cfg: Config) -> list[str]:
    """Load the feature list saved next to the model, with a sanity check."""
    pipeline = joblib.load(cfg.model_file())
    n_expected = pipeline.named_steps["model"].n_features_in_

    if not os.path.exists(cfg.vars_file()):
        raise FileNotFoundError(
            f"Feature list {cfg.vars_file()} not found. "
            f"Delete {cfg.model_file()} and retrain."
        )
    with open(cfg.vars_file()) as f:
        feature_names = json.load(f)
    if len(feature_names) != n_expected:
        raise ValueError(
            f"Feature list in {cfg.vars_file()} has {len(feature_names)} features but "
            f"the model expects {n_expected}. Delete {cfg.model_file()} and retrain."
        )
    return feature_names


def run(cfg: Config) -> None:
    """End-to-end pipeline for one configuration."""
    os.makedirs(cfg.models_dir, exist_ok=True)
    os.makedirs(cfg.results_dir, exist_ok=True)

    cfg.date_str = resolve_date_str(cfg)
    print(f"Run date: {cfg.date_str}")
    print(f"Model file: {cfg.model_file()}")

    stations = load_stations_metadata(cfg.stations_metadata_path)
    ei_station_dict = load_ei_data(cfg.ei_dir)
    cov_files = prepare_covariates(cfg.data_vars, cfg.covariates_dir)

    gdf_labeled = prepare_grid_labeled(
        stations, cov_files, cfg.data_vars, cfg.dem_file_pattern
    )

    # The model's feature schema is fully determined by cfg.data_vars.
    feature_names = derive_feature_names(cfg.data_vars)
    print(f"Feature names ({len(feature_names)}): {feature_names}")

    # ---- Train (if needed) + evaluate on test set ----
    if not os.path.exists(cfg.model_file()):
        print("\nPreparing training data...")
        df_data = prepare_training_data(
            gdf_labeled, cfg.data_vars, ei_station_dict, cfg.pred_col
        )
        missing = [c for c in feature_names if c not in df_data.columns]
        if missing:
            raise KeyError(
                f"Training table is missing expected feature columns: {missing}. "
                "This usually means a covariate zarr didn't produce the expected "
                "columns after pivot."
            )

        print("\nTraining model...")
        _, gdf_labeled = train_model(df_data, gdf_labeled, feature_names, cfg)
        predict_test_set(df_data, gdf_labeled, feature_names, cfg)

    # ---- Full-grid inference ----
    if os.path.exists(cfg.predictions_path()):
        print(f"\nPredictions already exist -> {cfg.predictions_path()}. Skipping.")
        return

    print("\nFull-grid inference...")
    if os.path.exists(cfg.data_grid_pkl()):
        gdf_grid = pd.read_pickle(cfg.data_grid_pkl())
    else:
        gdf_grid = prepare_full_grid(cov_files, cfg.data_vars, cfg.dem_file_pattern)
        gdf_grid.to_pickle(cfg.data_grid_pkl())

    pipeline = joblib.load(cfg.model_file())
    feature_names = load_trained_feature_names(cfg)
    predict_per_grid_cell(gdf_grid, pipeline, cfg.data_vars, feature_names, cfg)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--date", dest="date_str", default=None,
                        help="Run date as YYYYMMDD. Overrides config.date_str.")
    parser.add_argument("--version-tag", dest="version_tag", default=None,
                        help="Optional version label (e.g. v2, exp1) appended to model and prediction filenames.")
    parser.add_argument("--model-type", dest="model_type", default=None,
                        choices=["rf", "nn", "gpr"])
    parser.add_argument("--pred-col", dest="pred_col", default=None,
                        choices=["EI_daily_avg", "EI_daily_percent", "EI_daily_cumsum"])
    tune = parser.add_mutually_exclusive_group()
    tune.add_argument("--tune", dest="tune", action="store_true", default=None)
    tune.add_argument("--no-tune", dest="tune", action="store_false", default=None)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    """Load YAML config and apply CLI overrides."""
    with open(args.config) as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    for key in ("date_str", "model_type", "pred_col", "tune", "version_tag"):
        val = getattr(args, key)
        if val is not None:
            raw[key] = val
    return Config(**raw)


if __name__ == "__main__":
    run(build_config(parse_args()))
