# Soil Erosion Monitoring for Switzerland

A remote sensing and machine learning pipeline for estimating soil erosion risk across Switzerland using Sentinel-2 imagery. The project derives the **C-factor** (cover management) of the **RUSLE** (Revised Universal Soil Loss Equation) via satellite-based fractional cover and rainfall erosivity.

For more detail, refer to the following documentation:

> Ledain S., Gilgen A., Aasen H. (2026) "Prediction and monitoring of fractional cover of arable land from Sentinel-2." *Under review.*

Or contact Anina Gilgen (anina.gilgen@agroscope.admin.ch) and Helge Aasen (helge.aasen@agroscope.admin.ch)


---

## Overview

```
Sentinel-2 imagery              MeteoSwiss rain-gauge stations
        │                                       │
        ▼                                       ▼
  Synthetic mixtures                  Erosivity Index (EI daily)
  (soil+PV+NPV endmembers)                      │
        │                                       ▼
        ▼                          Daily rainfall erosivity grids
  Spectral Unmixing                       (100 m, Switzerland)
       (NN)                                     │
        │                                       │
        ▼                                       │
  Cover Fractions (PV/NPV/Soil) ───────────────►┘
        │                                       │
        ▼                                       ▼
    FC maps                                  C-factor
  (weekly, municipal)
        │
        ▼
   Driver Analysis
   (climate, crop
    type, topography)
```

---

## Repository Structure

```
.
├── baresoil/               # Soil group classification from bare soil composites
├── pv_npv_members/         # PV and NPV spectral endmember sampling
├── spectral_mixing/        # Synthetic spectral mixture generation
├── spectral_unmixing/      # Model training, tuning, validation, inference
├── FC_mapping/             # Nationwide FC prediction and aggregation
├── erosivity_index/        # Rainfall erosivity (R-factor) from station data
├── cfactor/                # C-factor / Soil Loss Ratio computation
├── driver_analysis/        # Statistical and ML analysis of FC drivers (incomplete)
├── timeseries_cleaning/    # GPR-based timeseries gapfilling
└── requirements.txt
```

---

## Pipeline

Sections 1–4 develop the Fractional Cover (FC) models. Section 5 applies them across Switzerland to create FC products. Sections 6 derives the rainfall erosivity and Section 7 combines it with FC to produce the C-factor. Section 8 examines what drives FC spatially and temporally (incomplete). Section 9 describes the timeseries cleaning utilities.

---

### 1. Soil Group Classification (`baresoil/`)

Bare soil spectra across Switzerland are clustered into 5 groups to capture variation in soil optical properties. These groups drive the soil-specific spectral unmixing models in step 4.

**Script**: `baresoil/soilsuite.py`

Steps:
1. Download the [DLR SoilSuite](https://geoservice.dlr.de/web/datasets/soilsuite_eur_5y) bare soil composite for Switzerland
2. Resample to 10 m resolution (from 20 m); sample 250k points within Swiss borders where `MASK == 1` (bare soil present) AND falling in agricultural areas
3. Cluster using K-means; optimal cluster count determined via silhouette score, elbow method, and input from soil scientists — **5 groups** were selected
4. Fit final K-means; the 25/50/75th percentile spectra per group serve as soil endmembers

The cluster labels were remapped for interpretability:
```python
label_map = {0: 5, 1: 2, 2: 4, 3: 1, 4: 3}
```

| Cluster | Description |
|---------|-------------|
| 1 | Darkest soils — plateau/alpine valleys, high SOC |
| 2 | Medium-dark — plateau, Jura, alpine valleys, moderate SOC |
| 3 | Medium brightness — low-moderate SOC, moderate-high silt |
| 4 | Bright — low SOC, mainly plateau/Jura, moderate clay |
| 5 | Brightest — low SOC, low-moderate clay, plateau |

**Outputs**:
- `baresoil/models/kmeans_5_clusters_agri_v2.pkl` — fitted K-means model
- `baresoil/plots/soil_endmembers_5_clusters_agri_v2_renamed.png` — endmember spectra per group
- `baresoil/plots/sampled_pts_5_clusters_agri_v2_renamed.png` — spatial distribution
- `baresoil/summarised_soil_samples_renamed.pkl` — percentile spectra per group

**Additional scripts**:
- `baresoil/create_qgis_layers.py` — shapefiles of predicted soil groups for QGIS visualisation
- `baresoil/predict_SRC_soilgroup.py` — apply K-means to all SRC data

```bash
python baresoil/soilsuite.py
```

---

### 2. PV and NPV Endmember Sampling (`pv_npv_members/`)

Photosynthetic (PV) and non-photosynthetic (NPV) vegetation spectra are sampled from Sentinel-2 time series, stratified by crop type and year (2021–2023).

**Script**: `pv_npv_members/sample_endmembers.py`

Steps:
- For each year 2021–2023, identify the top 15 crop types by pixel count using Swiss crop maps (`pv_npv_members/crop_maps/`) and the LNF classification (`LNF_code_classification_20251031.xlsx`)
- Sample 10 000 locations per crop type and year (coordinates saved in `pv_npv_members/sampled_coords/`)
- For each location, extract the full-year Sentinel-2 time series (after cloud/snow/shadow cleaning)
- **PV sample**: spectrum at NDVI maximum and minimum SWIR ratio
- **NPV sample**: spectrum at NDVI and SWIR ratio minimum, constrained to June 1 – November 15
- For each crop type and year, summarise spectra as 25/50/75th percentiles

**Outputs**:
- `pv_npv_members/pv_spectra_v2.pkl`, `pv_npv_members/npv_spectra_v2.pkl` — raw sampled spectra
- `pv_npv_members/summarised_pv_samples_pername.pkl`, `summarised_npv_samples_pername.pkl` — percentile endmembers per crop type
- `pv_npv_members/plots/pv_npv_summary_per_cropname_*.png` — endmember visualisations
- `pv_npv_members/plots/pv_npv_samples_locations_per_crop.png` — sampling locations

```bash
python pv_npv_members/sample_endmembers.py
```

---

### 3. Synthetic Spectral Mixture Generation (`spectral_mixing/`)

Synthetic training datasets are generated by linearly mixing endmembers from the spectral library, following the methodology of [Locher et al. (2025)](https://www.sciencedirect.com/science/article/pii/S0034425724006205).

**Script**: `spectral_mixing/mix_spectra_composition.py`

- 10 000 synthetic two- or three-component mixtures per cover fraction, per soil group, per iteration (×5 iterations)
- Endmembers randomly sampled from the library; fractions assigned randomly, always summing to 1
- A shade endmember (near-zero reflectance = 0.01 across all bands) is included as a mixing component but not as a prediction target
- Pure endmembers are also added to each dataset; soil-specific datasets include only the relevant soil group
- **Features**: 10 Sentinel-2 spectral bands; **Labels**: PV / NPV / Soil fractions

Datasets are named:
```
SYNTHMIX_SOIL-00{soil_group}_SHADOW-TRUE_FEATURES_CLASS-00{class}_ITERATION-00{i}.txt
SYNTHMIX_SOIL-00{soil_group}_SHADOW-TRUE_RESPONSE_CLASS-00{class}_ITERATION-00{i}.txt
```
where soil group `0` = global model, `1–5` = soil-specific; class `1`=NPV, `2`=PV, `3`=Soil.

Saved to `spectral_mixing/synthetic_samples_composition_10k/`.

```bash
python spectral_mixing/mix_spectra_composition.py
```

---

### 4. Spectral Unmixing — Model Training & Validation (`spectral_unmixing/`)

One model is trained per cover class (NPV / PV / Soil) × soil group (0=global, 1–5) × iteration (×5). Predictions are always reported as the mean over the 5 iterations.

**Models tested**: SVR, Random Forest, Neural Network -> NN were found to be best
**Hyperparameter tuning**: Optuna

```bash
# Tune hyperparameters (results saved per model/class/soil group)
python spectral_unmixing/code/tune.py

# Train all tuned models from saved hyperparameters
python spectral_unmixing/code/train_tuned_from_csv.py
```

- Tuning scores (averaged over iterations) saved to `spectral_unmixing/results/{MODELTYPE}_tuning_{CLASSNAME}_soilgroup{j}.xlsx`
- Best hyperparameters manually curated in `spectral_unmixing/code/tuned_hyperparameters.csv`
- Trained models saved as `spectral_unmixing/models/{MODELTYPE}_CLASS{i}_SOIL{j}_ITER{N}.pkl`
- Per-model results (RMSE, R², MAE) saved to `spectral_unmixing/results/{MODELTYPE}_CLASS{i}_SOIL{j}.csv`
- Test-set prediction plots saved to `spectral_unmixing/results/test_preds_{MODELTYPE}.png`

#### Validation on drone data

PV fractions derived from high-resolution UAV imagery are compared to model predictions after upscaling to Sentinel-2 resolution.

```bash
python spectral_unmixing/code/UAV_validation.py
```

> [!WARNING]
> This validation step had issues do to field boundaries in the drone data not being explicit and therefore making FC comparisons harder.
> This would need to be improved, currently results from this validation were not taken into account.


#### Validation on farm data (ZA-AUI)

The models are evaluated against georeferenced farms with known field management calendars (ZA-AUI dataset).

- `spectral_unmixing/code/validation/za-aui_data.py` — produces FC time series and plots per farm/year/crop; basic cloud cleaning only. Results saved to `spectral_unmixing/results/ZA-AUI/<year>/<crop_name>/<farm_id>/`
- `spectral_unmixing/code/validation/za-aui_data_management.py` — plots for farms with a specific management event; results saved to `spectral_unmixing/results/ZA-AUI/NPV_check/`
- `spectral_unmixing/code/validation/za-aui_data_fcprecompute.py` — same as above but uses precomputed FC instead of on-the-fly predictions
- `spectral_unmixing/code/validation/za-aui_data_animation.py` — creates plots/animations for a single farm using precomputed FC
- Qualitative evaluation summarised in `spectral_unmixing/code/validation/ZA-AUI_evaluation.xlsx` and `spectral_unmixing/code/za-aui.ipynb`

To create an MP4 animation of FC predictions for any shapefile:
```bash
python animation_FC_from_shapefile.py
```

---

### 5. Nationwide FC Mapping (`FC_mapping/`)

Applies trained models to all available Sentinel-2 scenes over Switzerland. Requires multi-CPU and GPU.

```bash
# Step 1 – predict FC for every S2 pixel; saved as zarr files
# (same naming convention as S2 input files)
python FC_mapping/predict_FC_CH.py

# Step 2 – extract valid in-field, cloud-free data per municipality
# and save weekly .gpkg files in CH_fraction_weekly_{yr}/
# File naming: CH_fraction_{weeknbr}_soil_arable_raw.gpkg
#   'soil'   = predictions from soil-specific models (vs 'global')
#   'arable' = grassland LNF areas excluded
python FC_mapping/create_datalayers.py

# Step 3 – aggregate to municipality level and produce plots
# Optional: pass specific LNF codes to filter for particular crop(s)
python FC_mapping/aggregate_FC.py
```

Key functions in `aggregate_FC.py`:

| Function | Description | Output |
|----------|-------------|--------|
| `plot_median_of_pixelmean` | Weekly pixel mean → municipal median; requires >50% arable land covered | `CH_fraction_{weeknbr}_soil_arable_COMMUNE.gpkg/.png` |
| `count_fraction_soil_above_50percent` | Fraction of arable pixels per municipality with FC soil > 0.5 | `CH_fraction_soilthresh_{weeknbr}_COMMUNE.gpkg` |
| `avg_baresoil_days` | Average annual bare-soil days per municipality (pixel bare if FC soil > 0.5) | `CH_baredays_{level}.gpkg` |

Additional plot functions (weekly bare soil fraction per municipality, etc.) are provided in `FC_mapping/report_plots.py`.

FC predictions are stored as zarr at `~/mnt/eo-nas1/data/satellite/sentinel2/FC/`.  
Weekly aggregated products are stored in `FC_mapping/CH_fraction_weekly_{yr}/`.

---

### 6. Rainfall Erosivity Index (`erosivity_index/`)

Derives long-term climatological daily rainfall erosivity (EI30) from MeteoSwiss station records and spatially upscales it to a 100 m grid across Switzerland for every day of year (DOY 1–365). The EI30 computation follows the methodology from [Schmidt et al. (2016)](https://hess.copernicus.org/articles/20/4359/2016/hess-20-4359-2016.pdf).

#### Step 1 — Download precipitation data from weather stations `download_meteo_stations.py`
Through the MeteoSwiss API, download the precipitation data:
```bash
python erosivity_index/download_meteo_stations.py
```
Each station's data is saved to `stations_data/stations_csv/`

#### Step 2 — Prepare covariate grids:
The DEM and weather covariates are prepared. The DEM data is rescaled to a 100m grid, slope and aspect are computed, and the produt is saved to `covariates/dem_100m.zarr`. The DEM grid serves as reference for the upscaling.
```bash
python erosivity_index/dem_grid.py
```

The MeteoSwiss gridded climate data (NetCDF, LV95) is reprojected and resampled  to 100 m resolution aligned to the Sentinel-2 grid (EPSG:32632), saving one zarr per variable per year to `erosivity_index/covariates/`. Supported variables: daily precipitation (`RhiresD`), daily mean/min/max temperature (`TabsD`, `TminD`, `TmaxD`), relative sunshine duration (`SrelD`).

```bash
python erosivity_index/meteo_daily_grids.py \
    --year_start 2004 --year_end 2026 \
    --out_dir covariates/precip --var RhiresD --out_res 100
```

#### Step 3 — Compute station-level EI30: `compute_EIdaily.py`

Processes 10-minute station CSVs from `stations_data/stations_csv/` through three steps:

1. **`compute_EI30_fast`** — detects rainfall events (separated by ≥ 6 dry hours, minimum event precipitation 12.7 mm), computes unit rainfall energy per 10-minute interval (Brown & Foster 1987) and I30 (max rolling 30-minute intensity, computed per event to avoid boundary bleed), then saves EI30 = E × I30 per event to `stations_data/stations_EI30/`
2. **`compute_EIdaily_avg`** — distributes each event's EI30 proportionally across the days it spans, then averages over all years to produce a long-term mean EI per DOY (1–365) per station; DOY 366 is merged into DOY 365 to avoid underrepresentation in non-leap years; saved to `stations_data/stations_EIdaily/`
3. **`compute_EI_daily_percent`** — converts daily averages to fractional and cumulative percentage of annual EI per station, saved to `stations_data/stations_EIdaily_percent/`

```bash
python erosivity_index/compute_EIdaily.py
```

#### Step 3 — Spatial upscaling to 100 m grid: `upscale_EI.py`
Train a model to predict the long-term daily average EI for every DOY at every 100 m grid cell across Switzerland.

**Target**: `EI_daily_avg` (long-term mean EI per DOY per station, from step 2)

**Features** (all at 100 m, EPSG:32632):
- Terrain: elevation, slope, aspect (encoded as sin/cos)
- Daily precipitation climatology (DOY 1–365)
- Monthly precipitation climatology (avg, min, max)
- Monthly average temperature
- DOY (encoded as sin/cos)
- position (x,y)
- monthly average snow depth 

See the config files for the actually used features in the final model (`config_nn3.yaml`).

**Spatial train/val/test split**: stations assigned to 25 km grid cells; entire cells assigned to one split to prevent spatial autocorrelation leakage (≈70/15/15%)

**Tuning**: Optuna (20 trials), minimising spatially cross-validated RMSE

**Model selection**: Multiple model types and configurations were evaluated, including XGBoost (`config_xgb1.yaml`) and several neural network variants (`config_nn.yaml`, `config_nn2.yaml` through `config_nn5.yaml`), as well as earlier feature/preprocessing variants (`config_v1.yaml`–`config_v3.yaml`). Selection was based on test set scores extracted with `extract_model_info.py` and visual inspection of spatial and temporal prediction patterns produced by `check_predictions.py`. The model trained with **`config_nn3.yaml`** was chosen as the final model.

**Inference**: predictions made for all Swiss 100 m grid cells, one row per (grid cell × DOY); saved in chunks to parquet.
All model parameters can be set up in a config file; see `configs/config_example.yaml` for all options.
```bash
python erosivity_index/upscale_EI.py --config configs/config_nn3.yaml
```

**Outputs**:
- `erosivity_index/models/{model}_absEI_tuned_model_{suffix}.joblib` — trained model
- `erosivity_index/predictions/grid_EI_daily_avg_pred_{suffix}.parquet` — predicted long-term daily average EI for every 100 m grid cell × DOY (1–365) across Switzerland

The final predictions are saved as `grid_EI_daily_avg_pred_20260424_nn3.parquet`.

**Analyse models and predictions**:
- `erosivity_index/check_predictions.py` — spatial and temporal patterns in predictions, plotting. Saves in `erosivity_index/results`
- `erosivity_index/extract_model_info.py` — extract test scores from a trained model.

---

### 7. C-factor / Soil Loss Ratio (`cfactor/`)

Converts the FC predictions to the Soil Loss Ratio (SLR), which is the core of the RUSLE C-factor.

The SLR formula applied at each pixel and timestamp:

```
slr = exp(fc_total × β)    β = to calibrate
fc_total = PV + NPV × 100
```

- `cfactor/SLR.py` — 
- `cfactor/calibrate_SLR.py` — 

```bash
python cfactor/SLR.py
python cfactor/calibrate_SLR.py
```

---

### 8. Driver Analysis (`driver_analysis/`) *(incomplete)*

Investigates which factors explain spatial and temporal variation in fractional cover at the municipal scale.

**Script**: `driver_analysis/prepare_data.py` (data preparation) and `driver_analysis/model_fc_drivers.py` (modelling)

**Features used**:
- Lagged and rolling climate variables up to 12 weeks (temperature `Tabs`, precipitation `Rhires`, sunshine duration `Srel`)
- Topography: elevation, slope, aspect
- Current and previous crop type (from LNF classification)

**Models evaluated**: GLM, LASSO, Random Forest, XGBoost, MLP, GWR (Geographically Weighted Regression)  
**Interpretability**: SHAP values (bar, beeswarm, waterfall plots saved to `driver_analysis/shap_*.png`)

```bash
python driver_analysis/prepare_data.py
python driver_analysis/model_fc_drivers.py
python driver_analysis/spatial_analysis.py   # Moran's I spatial autocorrelation, GWR
```

---

### 9. Timeseries Cleaning (`timeseries_cleaning/`)

Utilities for cleaning and gapfilling FC timeseries, developed using the ZA-AUI farm data. This code was used for testing different smoothing and gapfilling options.

**Best pipeline**:
1. Strict cloud/snow/shadow cleaning: `clean_dataset(ds, cloud_thresh=0.05, snow_thresh=0.1, shadow_thresh=0.1, cirrus_thresh=800)`
2. Gaussian Process Regression (GPR) for gapfilling and uncertainty estimation on a single-year window

**Scripts**:
- `timeseries_cleaning/clean_zaaui.py` — timeseries + field calendar for farm data
- `timeseries_cleaning/clean_fulltime.py` — timeseries for a single year
- `timeseries_cleaning/generate_plot.py` — produce a single-farm plot (cloud cleaning only, no gapfilling)

---

## Installation

```bash
pip install -r requirements.txt
```

Core dependencies: `numpy`, `pandas`, `xarray`, `geopandas`, `rasterio`, `scikit-learn`, `torch`, `tensorflow`, `optuna`, `shap`, `dask`, `zarr`, `stackstac`

A GPU is required for `FC_mapping/predict_FC_CH.py`.

---

## Data

| Data | Source | Location |
|------|--------|----------|
| Sentinel-2 (L2A) | Planetary Computer / ODC | `~/mnt/eo-nas1/data/satellite/sentinel2/` |
| FC predictions (zarr) | This pipeline | `~/mnt/eo-nas1/data/satellite/sentinel2/FC/` |
| DLR SoilSuite bare soil composite | [DLR Geoservice](https://geoservice.dlr.de/web/datasets/soilsuite_eur_5y) | Downloaded by `baresoil/soilsuite.py` |
| Swiss crop maps (LNF) | swisstopo / cantonal | `pv_npv_members/crop_maps/`; classification in `LNF_code_classification_20251031.xlsx` |
| MeteoSwiss rain gauges (10 min) | MeteoSwiss | `erosivity_index/stations_data/` |
| MeteoSwiss station metadata | MeteoSwiss | `erosivity_index/stations_data/metadata/ogd-smn_meta_stations.csv` |
| SwissBOUNDARIES3D | swisstopo | `FC_mapping/swissBOUNDARIES3D_1_5_LV95_LN02.gpkg` |
| ZA-AUI farm/field calendar data | ZA-AUI | used in `spectral_unmixing/code/validation/` |

---

## Reference

The spectral unmixing methodology follows:

> Locher et al. (2025). *Remote sensing of fractional cover in heterogeneous agricultural landscapes.* Remote Sensing of Environment. [DOI](https://www.sciencedirect.com/science/article/pii/S0034425724006205)
