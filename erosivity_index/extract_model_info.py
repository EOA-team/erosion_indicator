"""
extract_model_info.py
---------------------
Extract hyperparameters and evaluation scores from a saved EI upscaling model
WITHOUT retraining.

Requirements
------------
- A trained .joblib pipeline (produced by upscale_EI_edit.py)
- The matching _data_vars.json feature list
- The original EI station CSVs and station metadata (to recompute test scores)

Usage
-----
    # Auto-detect the latest model for a given config:
    python extract_model_info.py --config config.yaml

    # Pin a specific run date / variant:
    python extract_model_info.py --config config.yaml --date 20260419 --version-tag v2

    # Skip score recomputation (hyperparameters only):
    python extract_model_info.py --config config.yaml --no-scores
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import warnings
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Minimal re-use of constants from the main script
# ---------------------------------------------------------------------------
STATION_COL = "station_abbr"
STATIC_COV_SET = {"elev", "slope", "aspect"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_model_path(models_dir: str, model_type: str, model_suffix: str,
                       date_str: str | None) -> str:
    """Return the .joblib path, auto-detecting the latest file if date_str is None."""
    if date_str:
        path = os.path.join(
            models_dir,
            f"{model_type}_{model_suffix}_model_{date_str}.joblib",
        )
        if not os.path.exists(path):
            sys.exit(f"[ERROR] Model file not found: {path}")
        return path

    pattern = os.path.join(models_dir, f"{model_type}_{model_suffix}_model_*.joblib")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        sys.exit(f"[ERROR] No model files matched: {pattern}")
    path = candidates[-1]
    print(f"[INFO] Auto-selected latest model: {path}")
    return path


def build_model_suffix(pred_col: str, tune: bool, version_tag: str | None) -> str:
    if pred_col == "EI_daily_avg":
        suffix = "absEI_tuned" if tune else "absEI"
    elif pred_col == "EI_daily_cumsum":
        suffix = "cumEI"
    else:
        suffix = "percent_tuned" if tune else "percent"
    if version_tag:
        suffix = f"{suffix}_{version_tag}"
    return suffix


# ---------------------------------------------------------------------------
# 1. Hyperparameters
# ---------------------------------------------------------------------------

def extract_hyperparameters(pipeline) -> dict[str, Any]:
    """
    Pull all hyperparameters from the fitted sklearn Pipeline.

    Returns a flat dict with keys like 'model__n_estimators' (full pipeline
    params) plus a nested 'model_params' dict of just the estimator's params.
    """
    model_step = pipeline.named_steps["model"]
    model_params: dict[str, Any] = model_step.get_params()

    # Scaler params if present (NN / GPR pipelines include a StandardScaler)
    scaler_params: dict[str, Any] = {}
    if "scaler" in pipeline.named_steps:
        scaler_params = pipeline.named_steps["scaler"].get_params()

    return {
        "model_class": type(model_step).__name__,
        "model_params": model_params,
        "scaler_params": scaler_params if scaler_params else None,
    }


def print_hyperparameters(info: dict[str, Any]) -> None:
    print("\n" + "=" * 55)
    print(f"  MODEL CLASS : {info['model_class']}")
    print("=" * 55)
    print("  Hyperparameters:")
    for k, v in info["model_params"].items():
        print(f"    {k:<30s} = {v}")
    if info["scaler_params"]:
        print("\n  Scaler (StandardScaler) params:")
        for k, v in info["scaler_params"].items():
            print(f"    {k:<30s} = {v}")
    print("=" * 55)


# ---------------------------------------------------------------------------
# 2. Scores  (recomputed from saved model + station data — no retraining)
# ---------------------------------------------------------------------------

def load_ei_data(ei_dir: str) -> dict[str, pd.DataFrame]:
    ei_data = {}
    for fname in os.listdir(ei_dir):
        if not fname.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(ei_dir, fname))
        station = fname.split("_")[-1].replace(".csv", "")
        ei_data[station] = df
    return ei_data


def compute_kge_nse(obs: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    corr  = np.corrcoef(obs, pred)[0, 1]
    alpha = np.std(pred) / np.std(obs)
    beta  = np.mean(pred) / np.mean(obs)
    kge   = 1 - np.sqrt((corr - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    nse   = 1 - np.sum((obs - pred)**2) / np.sum((obs - np.mean(obs))**2)
    return float(kge), float(nse)


def EI_to_cumul(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    """Derive cumulative % curves for both observed and predicted columns."""
    df = df.copy().sort_values([STATION_COL, "doy"])
    for prefix in ("predicted_", ""):
        avg_col  = f"{prefix}EI_daily_avg"
        pct_col  = f"{prefix}EI_daily_percent"
        csum_col = f"{prefix}EI_daily_cumsum"
        if avg_col in df.columns:
            total = df.groupby(STATION_COL)[avg_col].transform("sum")
            df[pct_col]  = np.where(total == 0, 0, df[avg_col] / total * 100)
            df[csum_col] = df.groupby(STATION_COL)[pct_col].cumsum()
    return df


def _station_split(
    stations: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    grid_size: int,
    seed: int,
) -> pd.DataFrame:
    """Mirror split_stations() from upscale_EI_edit.py exactly."""
    stations = stations.copy()
    stations["grid_x"] = (stations["station_coordinates_lv95_east"]  // grid_size).astype(int)
    stations["grid_y"] = (stations["station_coordinates_lv95_north"] // grid_size).astype(int)
    grid_cells = (
        stations[["grid_x", "grid_y"]]
        .drop_duplicates()
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
    n_cells = len(grid_cells)
    n_train = int(train_frac * n_cells)
    n_val   = int(val_frac   * n_cells)
    grid_cells["split"] = "test"
    grid_cells.loc[:n_train - 1, "split"] = "train"
    grid_cells.loc[n_train : n_train + n_val - 1, "split"] = "val"
    return stations.merge(grid_cells, on=["grid_x", "grid_y"], how="left")


def assemble_test_data_from_grid_pkl(
    grid_pkl_path: str,
    ei_dir: str,
    stations_metadata_path: str,
    feature_names: list[str],
    data_vars: list[str],
    pred_col: str,
    pipeline,
    train_frac: float = 0.70,
    val_frac: float   = 0.15,
    grid_size: int    = 25_000,
    seed: int         = 42,
) -> pd.DataFrame:
    """
    Rebuild the (station, DOY) test table using the cached full-grid pickle.

    The pickle contains every covariate for every 100 m grid cell. For each
    test station we look up the nearest grid cell, melt its daily/monthly
    columns into the same long form the training code used, add derived
    features, then call pipeline.predict().
    """
    import calendar
    from functools import reduce

    # Constants copied from upscale_EI_edit.py (keep in sync)
    MONTHLY_PREFIX = {
        "prec_monthly_avg": "monthly_avg_precip_",
        "prec_monthly_min": "monthly_min_precip_",
        "prec_monthly_max": "monthly_max_precip_",
        "monthly_avg_temp": "monthly_avg_temp_",
    }
    DAILY_PREFIX = {"prec_daily_avg": "daily_avg_precip_"}

    # 1. Spatial split to identify test stations
    keep = [
        "station_abbr", "station_name",
        "station_coordinates_lv95_east",
        "station_coordinates_lv95_north",
    ]
    stations = pd.read_csv(stations_metadata_path, delimiter=";", encoding="latin1")[keep]
    stations = _station_split(stations, train_frac, val_frac, grid_size, seed)
    test_stations_df = stations[stations["split"] == "test"].copy()
    test_station_ids = set(test_stations_df[STATION_COL])
    print(f"[INFO] Test stations ({len(test_station_ids)}): {sorted(test_station_ids)}")

    # 2. Load cached grid of wide features
    print(f"[INFO] Loading grid pickle: {grid_pkl_path}")
    gdf_grid = pd.read_pickle(grid_pkl_path)
    print(f"[INFO] Grid shape: {gdf_grid.shape}")

    # 3. Nearest-cell lookup for each test station (grid is 100 m in LV95)
    grid_xy = gdf_grid[["x", "y"]].to_numpy()
    picks = []
    for _, row in test_stations_df.iterrows():
        sx, sy = row["station_coordinates_lv95_east"], row["station_coordinates_lv95_north"]
        d2 = (grid_xy[:, 0] - sx) ** 2 + (grid_xy[:, 1] - sy) ** 2
        idx = int(np.argmin(d2))
        cell = gdf_grid.iloc[idx].copy()
        cell[STATION_COL] = row[STATION_COL]
        picks.append(cell)
    station_cells = pd.DataFrame(picks).reset_index(drop=True)

    # 4. Melt daily / monthly columns into long form (one row per (station, DOY))
    static_cols = [c for c in data_vars if c in STATIC_COV_SET]
    id_vars = [STATION_COL, "x", "y"] + static_cols

    # Daily melt
    df_daily = pd.DataFrame()
    if "prec_daily_avg" in data_vars:
        prefix = DAILY_PREFIX["prec_daily_avg"]
        daily_cols = [c for c in station_cells.columns if c.startswith(prefix)]
        if daily_cols:
            df_daily = station_cells.melt(
                id_vars=id_vars, value_vars=daily_cols,
                var_name="doy", value_name="prec_daily_avg",
            )
            df_daily["doy"] = df_daily["doy"].str.extract(r"(\d+)").astype(int)
            df_daily["monthnbr"] = pd.to_datetime(
                df_daily["doy"].astype(str).str.zfill(3), format="%j", errors="coerce",
            ).dt.month

    # Monthly melts
    monthly_frames = []
    for var, prefix in MONTHLY_PREFIX.items():
        if var not in data_vars:
            continue
        cols = [c for c in station_cells.columns if c.startswith(prefix)]
        if not cols:
            continue
        melted = station_cells.melt(
            id_vars=id_vars, value_vars=cols,
            var_name="monthnbr", value_name=var,
        )
        melted["monthnbr"] = melted["monthnbr"].str.extract(r"(\d+)").astype(int)
        monthly_frames.append(melted)
    df_monthly = (
        reduce(lambda l, r: l.merge(r, on=id_vars + ["monthnbr"], how="outer"), monthly_frames)
        if monthly_frames else pd.DataFrame()
    )

    # Merge daily + monthly on (id_vars, monthnbr)
    if not df_daily.empty and not df_monthly.empty:
        df_features = df_daily.merge(df_monthly, on=id_vars + ["monthnbr"], how="left")
    elif not df_daily.empty:
        df_features = df_daily
    elif not df_monthly.empty:
        # Monthly-only: expand to DOY
        ref = 2021
        days_in = {m: calendar.monthrange(ref, m)[1] for m in range(1, 13)}
        first  = {m: pd.Timestamp(year=ref, month=m, day=1).dayofyear for m in range(1, 13)}
        df_monthly["ndays"]     = df_monthly["monthnbr"].map(days_in)
        df_monthly["first_doy"] = df_monthly["monthnbr"].map(first)
        expanded = df_monthly.loc[df_monthly.index.repeat(df_monthly["ndays"])].copy()
        expanded["offset"] = expanded.groupby(level=0).cumcount()
        expanded["doy"] = expanded["first_doy"] + expanded["offset"]
        df_features = expanded.drop(columns=["ndays", "first_doy", "offset"]).reset_index(drop=True)
    else:
        sys.exit("[ERROR] Grid has neither daily nor monthly precip columns.")

    # 5. Derived features (aspect sin/cos, doy sin/cos, daily fraction)
    if "aspect" in data_vars and "aspect" in df_features.columns:
        df_features["aspect_sin"] = np.sin(np.deg2rad(df_features["aspect"]))
        df_features["aspect_cos"] = np.cos(np.deg2rad(df_features["aspect"]))
    if "doy" in data_vars and "doy" in df_features.columns:
        df_features["sin_doy"] = np.sin(2 * np.pi * df_features["doy"] / 365)
        df_features["cos_doy"] = np.cos(2 * np.pi * df_features["doy"] / 365)
    if "prec_daily_fraction" in data_vars:
        totals = df_features.groupby(STATION_COL)["prec_daily_avg"].transform("sum")
        df_features["prec_daily_fraction"] = df_features["prec_daily_avg"] / totals.replace(0, np.nan)

    # 6. Attach EI observations
    ei_data = load_ei_data(ei_dir)
    obs_rows = []
    for station, df in ei_data.items():
        if station not in test_station_ids:
            continue
        tmp = df[["doy", pred_col]].copy()
        tmp[STATION_COL] = station
        obs_rows.append(tmp)
    if not obs_rows:
        sys.exit("[ERROR] No EI data found for test stations.")
    ei_all = pd.concat(obs_rows, ignore_index=True)
    df_test = df_features.merge(ei_all, on=[STATION_COL, "doy"], how="inner")

    # 7. Validate feature columns
    missing = [f for f in feature_names if f not in df_test.columns]
    if missing:
        sys.exit(
            f"[ERROR] Grid pickle is missing features the model needs: {missing}\n"
            f"  data_vars from config: {data_vars}\n"
            f"  Check that config matches the one used at training time."
        )

    # 8. Predict
    X = df_test[feature_names].values
    df_test[f"predicted_{pred_col}"] = pipeline.predict(X)
    print(f"[INFO] Predicted on {len(df_test)} (station, DOY) test rows.")

    return df_test


def compute_scores(df: pd.DataFrame, pred_col: str) -> dict[str, Any]:
    """Compute RMSE / MAE / NRMSE / R² on the raw target + KGE / NSE on cumulative curves."""
    y_true = df[pred_col].values
    y_pred = df[f"predicted_{pred_col}"].values

    if pred_col == "EI_daily_cumsum":
        y_pred = np.clip(y_pred, 0, 100)

    rmse  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae   = float(mean_absolute_error(y_true, y_pred))
    r2    = float(r2_score(y_true, y_pred))
    nrmse = rmse / float(y_true.max() - y_true.min()) if y_true.max() != y_true.min() else float("nan")

    scores: dict[str, Any] = {
        "raw_target": {"RMSE": rmse, "MAE": mae, "NRMSE": nrmse, "R2": r2}
    }

    # Cumulative KGE/NSE per station
    if pred_col != "EI_daily_cumsum":
        df = EI_to_cumul(df, pred_col)

    if "EI_daily_cumsum" in df.columns and "predicted_EI_daily_cumsum" in df.columns:
        kge_list, nse_list = [], []
        for station in df[STATION_COL].unique():
            s = df[df[STATION_COL] == station].sort_values("doy")
            kge, nse = compute_kge_nse(s["EI_daily_cumsum"].values,
                                       s["predicted_EI_daily_cumsum"].values)
            kge_list.append(kge)
            nse_list.append(nse)
        scores["cumulative_curve"] = {
            "mean_KGE": float(np.mean(kge_list)),
            "std_KGE":  float(np.std(kge_list)),
            "mean_NSE": float(np.mean(nse_list)),
            "std_NSE":  float(np.std(nse_list)),
        }

    return scores


def print_scores(scores: dict[str, Any]) -> None:
    print("\n" + "=" * 55)
    print("  SCORES — raw target")
    print("=" * 55)
    for k, v in scores["raw_target"].items():
        print(f"    {k:<10s} = {v:.4f}")
    if "cumulative_curve" in scores:
        print("\n  SCORES — cumulative EI curve (per-station KGE / NSE)")
        print("-" * 55)
        for k, v in scores["cumulative_curve"].items():
            print(f"    {k:<12s} = {v:.4f}")
    print("=" * 55)


# ---------------------------------------------------------------------------
# Scores from pre-existing prediction Parquet (alternative path)
# ---------------------------------------------------------------------------

def scores_from_parquet(parquet_path: str, pred_col: str,
                        ei_dir: str, station_col: str = STATION_COL) -> dict[str, Any]:
    """
    Compute scores by joining saved grid predictions with station observations.

    The Parquet must contain x/y/doy + predicted_{pred_col}.
    Station EI CSVs supply the ground truth.
    """
    print(f"[INFO] Loading predictions from {parquet_path}")
    preds = pd.read_parquet(parquet_path)

    ei_data = load_ei_data(ei_dir)
    rows = []
    for station, df in ei_data.items():
        tmp = df[["doy", pred_col]].copy()
        tmp[station_col] = station
        rows.append(tmp)
    obs_df = pd.concat(rows, ignore_index=True)

    merged = preds.merge(obs_df, on="doy", how="inner")
    if merged.empty:
        sys.exit("[ERROR] Parquet join with EI observations produced no rows.")

    return compute_scores(merged, pred_col)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="Path to the same YAML config used for training.")
    p.add_argument("--date",        dest="date_str",     default=None, help="Run date YYYYMMDD.")
    p.add_argument("--version-tag", dest="version_tag",  default=None)
    p.add_argument("--model-type",  dest="model_type",   default=None, choices=["rf", "nn", "gpr"])
    p.add_argument("--pred-col",    dest="pred_col",     default=None,
                   choices=["EI_daily_avg", "EI_daily_percent", "EI_daily_cumsum"])
    p.add_argument("--no-scores",   dest="compute_scores", action="store_false", default=True,
                   help="Only print hyperparameters, skip score computation.")
    p.add_argument("--parquet-predictions", dest="parquet_path", default=None,
                   help="Optional: path to an existing predictions .parquet to compute scores from "
                        "(full-grid predictions vs obs, not test-set only).")
    p.add_argument("--grid-pkl", dest="grid_pkl", default=None,
                   help="Optional: path to the cached covariates/data_grid_*.pkl. "
                        "Auto-inferred from config + date + version-tag if omitted.")
    p.add_argument("--output-json", dest="output_json", default=None,
                   help="Optional: save results as JSON to this path.")
    tune_grp = p.add_mutually_exclusive_group()
    tune_grp.add_argument("--tune",    dest="tune", action="store_true",  default=None)
    tune_grp.add_argument("--no-tune", dest="tune", action="store_false", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load config
    with open(args.config) as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    # CLI overrides
    for key in ("date_str", "model_type", "pred_col", "tune", "version_tag"):
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val

    model_type  = cfg.get("model_type",  "rf")
    pred_col    = cfg.get("pred_col",    "EI_daily_avg")
    tune        = cfg.get("tune",        True)
    version_tag = cfg.get("version_tag", None)
    date_str    = cfg.get("date_str",    None)
    models_dir  = cfg.get("models_dir",  "models")

    model_suffix = build_model_suffix(pred_col, tune, version_tag)
    model_path   = resolve_model_path(models_dir, model_type, model_suffix, date_str)

    # Derive vars_file path (same naming as Config.vars_file())
    vars_file = model_path.replace(".joblib", "_data_vars.json")

    print(f"\n[INFO] Loading model  : {model_path}")
    pipeline = joblib.load(model_path)

    print(f"[INFO] Loading features: {vars_file}")
    if not os.path.exists(vars_file):
        sys.exit(f"[ERROR] Feature list not found: {vars_file}")
    with open(vars_file) as f:
        feature_names: list[str] = json.load(f)
    print(f"[INFO] Features ({len(feature_names)}): {feature_names}")

    # ── 1. Hyperparameters ──────────────────────────────────────────────────
    hyper_info = extract_hyperparameters(pipeline)
    print_hyperparameters(hyper_info)

    result: dict[str, Any] = {
        "model_file":    model_path,
        "feature_names": feature_names,
        "hyperparameters": hyper_info,
        "scores": None,
    }

    # ── 2. Scores ────────────────────────────────────────────────────────────
    if args.compute_scores:
        if args.parquet_path:
            # Scores from existing prediction Parquet (joined with EI obs)
            scores = scores_from_parquet(
                args.parquet_path,
                pred_col,
                ei_dir=cfg["ei_dir"],
            )
        else:
            # Test-set scores: re-use the cached full-grid pickle
            covariates_dir = cfg.get("covariates_dir", "covariates")
            tag = f"_{version_tag}" if version_tag else ""
            grid_pkl = os.path.join(
                covariates_dir, f"data_grid_{date_str}{tag}.pkl"
            )
            if args.grid_pkl:
                grid_pkl = args.grid_pkl
            if not os.path.exists(grid_pkl):
                sys.exit(
                    f"[ERROR] Grid pickle not found: {grid_pkl}\n"
                    f"  Pass --grid-pkl <path> or --parquet-predictions <path> instead."
                )

            print(f"\n[INFO] Re-running inference on test stations using grid pickle...")
            df_test = assemble_test_data_from_grid_pkl(
                grid_pkl_path          = grid_pkl,
                ei_dir                 = cfg["ei_dir"],
                stations_metadata_path = cfg["stations_metadata_path"],
                feature_names          = feature_names,
                data_vars              = cfg["data_vars"],
                pred_col               = pred_col,
                pipeline               = pipeline,
                train_frac             = cfg.get("train_frac",     0.70),
                val_frac               = cfg.get("val_frac",       0.15),
                grid_size              = cfg.get("split_grid_size", 25_000),
                seed                   = cfg.get("random_seed",     42),
            )
            scores = compute_scores(df_test, pred_col)

        print_scores(scores)
        result["scores"] = scores

    # ── 3. Optional JSON dump ─────────────────────────────────────────────────
    if args.output_json:
        import json as _json
        with open(args.output_json, "w") as f:
            _json.dump(result, f, indent=2, default=str)
        print(f"\n[INFO] Results saved -> {args.output_json}")


if __name__ == "__main__":
    main()
