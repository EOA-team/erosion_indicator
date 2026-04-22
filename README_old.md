# Spectral unmixing


## 1. Baresoil composite classification

Different soil groups (in terms of spectral properties) are determine using K-means clustering. The bare soil composite from the [DLR soilsuite](https://geoservice.dlr.de/web/datasets/soilsuite_eur_5y) is used. 

In `baresoil/soilsuite.py` the following steps are performed:
1. Download data
    - The bare soil composite is downloaded for Switzerland
2. Sample points
    - The data is resampled to 10m resolution from 20m
    - 250k points that fall within Swiss borders and where MASK is 1 (presence of baresoil) are sampled
3. Cluster the samples
    - The samples are then clustered using their bands
    - Different numbers of clusters are tested, using silhouette score and elbow method to look at optimal cluster size
4. The final Kmeans model is fitted and the soil groups are plotted
    - We determined 5 groups to be enough to seperate the reflectectances and also discussed with soil scientists
    - Plot of the clustered data: `baresoil/plots/sampled_pts_5_clusters_agri_v2.png`
5. For each group, the 25/50/75th percentiles are determined
    - Plot of the summarised spectra per soil group: `baresoil/plots/soil_endmembers_5_clusters_agri_v2.png`

The soil clusters were then renamed:
```
label_map = {0: 5, 1: 2, 2: 4, 3: 1, 4: 3}
```
and the updated plots can be found in the following plots: `baresoil/plots/sampled_pts_5_clusters_agri_v2_renamed.png`, `baresoil/plots/soil_endmembers_5_clusters_agri_v2_renamed.png`

### Description of soil groups
- Soil clusters 4 and 5 are bright soils, 1 and 2 are darker soils and cluster 3 is a medium brightness soil
- Cluster 1 are the darkest soils, primarily found in the plateau where SOC is high and otherwise found in alpine valleys
- Cluster 2 are medium dark soils found in the plateau, the jura and alpine valleys. They have moderate SOC in the plateau and low-moderate SOC in the jura and alpine valleys. 
- Cluster 3 are medium bright soils with low-moderate SOC and moderate-high silt contents
- Cluster 4 are bright soils with low SOC, mainly in the plateau and in the jura range. They have moderate clay contents
- Cluster 5 represents the brightest soils, mainly in the plateau and characterized by low SOC and low-moderate clay contents


All soils will be classified to these groups. 

### Additional scripts
- `create_qgis_layers.py`: create shapefiles containing predicted soil groups for each sampled point
- `predict_SRC_soilgroup.py`: apply K-means mode to all SRC data and save



## 2. PV and NPV endmembers

Different PV (photosynthetic vegetation) and NPV (non photosynthetic vegetation) are also sampled.

In `pv_npv_members/sample_endmembers.py`
- For each year between 2021 and 2023, we identified the top 15 crops types according to pixel count
- We know the classification of each pixel thanks to crop maps (`pv_npv_members/crop_maps`) and we use the classification determined in `~/mnt/eo-nas1/data/landuse/documentation/LNF_code_classification_20250217.xlsx` (column "categories2024"). The crop type is thus a crop name or crop group.
- 10k locations are sampled per crop type and year (saved in `pv_npv_members/sampled_coords`)
- For each location, the yearly S2 data is extracted
  - After data cleaning (missing data, clouds, snow...), we sample PV and NPV spectra if possible
  - For PV it's when NDVI is max and SWIR ratio is min
  - For NPV it's when NDVI and SWIR ratio are min. Additionally this must be between Jun. 1st and Nov. 15th
- The spectra at these two moments are saved as examples of PV and NPV spectra (`pv_npv_members/pv_spectra_v2.pkl`, `pv_npv_members/npv_spectra_v2.pkl`)
- For each crop type and year, the PV and NPV spectra are summarised according to the 25/50/75th percentiles (`pv_npv_members/summarised_npv_samples_pername.pkl`, `pv_npv_members/summarised_pv_samples_pername.pkl`)

The summarised spectra will serve as endmembers for generating training data for a spectral unmixing model.
These can be visualised in the plots `pv_npv_members/plots/pv_npv_summary_per_cropname_*.png` and `pv_npv_members/plots/pv_npv_samples_locations_per_crop.png`


## 3. Spectral mixing

Following the methodology in [Locher et al. (2025)](https://www.sciencedirect.com/science/article/pii/S0034425724006205?via%3Dihub#s0040):

We created 1000 synthetic mixtures for each cover fraction based on the spectral library. Each synthetic mixture comprised two to three endmembers from the three cover fractions randomly sampled from the library. These were then linearly combined with random fractions assigned to each endmember class, ensuring the fractions always sum up to 1. The resulting synthetic spectrum was then added to the training dataset, with the six spectral bands as input variables and the share of the target cover fraction as the label. Additionally, we included a shade spectrum in the synthetic mixtures to represent direct and structural shade components (Shimabukuro and Smith, 1991). The shade endmember, with a near-zero reflectance of 0.01 across all bands, was treated like any other endmember during the mixing step but was not considered as target fraction.
The pure endmemberes were also added to each dataset, and for the soil-specific datasets only the relevant soils were included.

The mixing is done by providing paths to the soil, PV and NPV endmembers to the following script
```
python spectral_mixing/mix_spectra_composition.py
```

The synthetic mixtures (pairs of features/labels) are stored in the following way:
```
feat_filename = f'synthetic_samples_composition_10k/SYNTHMIX_SOIL-00{soil_group_nbr}_SHADOW-TRUE_FEATURES_CLASS-00{feature_class}_ITERATION-00{i}.txt'
resp_filename = f'synthetic_samples_composition_10k/SYNTHMIX_SOIL-00{soil_group_nbr}_SHADOW-TRUE_RESPONSE_CLASS-00{feature_class}_ITERATION-00{i}.txt'
```

where 0 means the global model, and the soil groups determined from the K-means clustering  are named 1 to 5. There are 3 feature classes (1=NPV, 2=PV, 3=Soil). Each dataset contains 10'000 mixtures, and these datasets are iterated 5 times.

The resonse contains the fractions of NPV, PV, Soil and shadow. 


## 4. Spectral unmixing

For each class (1=NPV, 2=PV, 3=Soil), each soil group (0=global, soil specific named 1-5) and iteration, and individual model is trained on its respective dataset generated above. Predictions and results are always presented as the mean over the 5 iterations.

Three models are tested: SVR, RF, NN. Hyperparameter tuning is performed with Optuna (see script for precise hyperparameter space):
```
python spectral_unmixing/code/tune.py
```
Tuning scores for each model-class-soilgroup (averaged already over the iterations) are saved in `spectral_unmixing/results/{MODELTYPE}_tuning_{CLASSNAME}_soilgroup{j}.xlsx`.

The best hyperparameters are manually saved in `code/tuned_hyperparameters.csv` and all these models are trained with
```
python spectral_unmixing/code/train_tuned_from_csv.py
```
which will save each model as `spectral_unmixing/models/{MODELTYPE}_CLASS{i}_SOIL{j}_ITER{N}.pkl` and the results (RMSE, R2, MAE) in `spectral_unmixing/results/{MODELTYPE}_CLASS{i}_SOIL{j}.csv`

Plots on the test set (leave-out from train set) are save in `spectral_unmixing/results/test_preds_{MODELTYPE}.png`

### Validation on drone data
The FC models were run on high-resolution drone data, where PV fractions were availbale for the drone data and compared to the predictions after upsacling to the same resolution as Sentinel-2:
```
python spectral_unmixing/code/UAV_validation.py
```


### Validation on farm data
The models are also evaulated on some georeferenced farm data with field calendars (ZA-AUI data).
FC timeseries (only basic cloud cleaning done) with images were produced for farms over multiple years.

- `spectral_unmixing/code/validation/za-aui_data.py`:  produces the timeseries and plots for the farms. Saves the results year by year, crop by crop in `spectral_unmixing/results/ZA-AUI/<year>/<crop_name>/<farm_id>/...png`. A visual/qualitative evaluation of the performance of the models based on these plots was done in `spectral_unmixing/code/validation/ZA-AUI_evaluation.xlsx` and these results were synthesised in `spectral_unmixing/code/za-aui.ipynb`
- `spectral_unmixing/code/validation/za-aui_data_management.py`: produces the plots for a farms where theres was a specific management activity. Results saved to `spectral_unmixing/results/ZA-AUI/NPV_check`
- `spectral_unmixing/code/validation/za-aui_data_fcprecompute.py`: Same as  `spectral_unmixing/code/za-aui_data.py` except that it uses precomputed FC data instead of running predictions on the fly.
- `spectral_unmixing/code/validation/za-aui_data_animation.py`: Uses precomputed FC data instead of running predictions on the fly, create specific plot or animation for a single farm


To create an mp4 of FC given a shapefile, use the script `animation_FC_from_shapefile.py`.


## 5. Timeseries cleaning

Different timeseries cleaning and gapfilling algorithms were tested, using the ZA-AUI data.
- `timeseries_cleaning/clean_zaaui.py`: timeseries for the farm data, and its field calendar
- `timeseries_cleaning/clean_fulltime.py`: timeseries for the farm data, on a single year

The best pipeline seems to be:
1. Apply strict cloud cleaning `clean_dataset(ds, cloud_thresh=0.05, snow_thresh=0.1, shadow_thresh=0.1, cirrus_thresh=800)`
2. Use GPR to gapfill and get uncertainty on a single year of data

To generate a single farm plot (no gapfilling, just timeseries cleaning) use `timeseries_cleaning/generate_plot.py`


## 6. FC product

The following steps were followed to obtain weekly municipal mean FC values:

First, predict FC for every available S2 pixel: 
```
python FC_mapping/predict_FC_CH.py #runs on multi-CPU and needs GPU
``` 
Prediction with global model and determined soil group will be saved in zarr files (same naming as Sentinel-2 files) top `~/mnt/eo-nas1/data/satellite/sentinel2/FC'

Then extract valid data points per municipality (in fields, cloud free...), and save to .gpkg for each week. These are saved in yearly folders as `CH_fraction_weekly_{yr}/CH_fraction_{weeknbr}_soil_arable_raw.gpkg`.  The `soil` (vs `global`) in the name means predictions from soil specific models are taken, and `arable` (vs `grassland`) means grassland LNF areas are left out. 
```
python FC_mapping/create_datalayers.py
```

Finally, to produce specific aggregated plots, use the script. At this step, speicifc LNF codes can be passed to the main fucntion to filter for specific crop(s).
```
python FC_mapping/aggregate_FC.py
```
This contains several functions:
- `plot_median_of_pixelmean`: get the weekly mean of each pixel, and aggregate with median over municipality. Consider only if >50% of the arable land of that muncipialtiy is covered by valid data. Results are saved to `CH_fraction_weekly_{yr}/CH_fraction_{weeknbr}_soil_arable_COMMUNE.gpkg` and `CH_fraction_weekly_{yr}/CH_fraction_{weeknbr}_soil_arable_COMMUNE.png`
- `count_fraction_soil_above_50percent`: get the weekly mean of each pixel. Per municipality count the fraction of pixels in arable land that have FC soil >0.5. Consider only if >50% of the arable land of that muncipialtiy is covered by valid data. Results are saved to CH_fraction_soilthresh_{weeknbr}_COMMUNE.gpkg
- `avg_baresoil_days`: get the weekly mean of each pixel. Over a year, compute how many days the pixel is bare (ie. FC soil > 0.5). Per municipality compute the average bare soil days. Results are saved to `CH_fraction_weekly_{yr}/CH_baredays_{level.upper()}.gpkg`


Additional functions to create specific plots (including weekly fraction of bare soil per municipality) are provided in the script `report_plots.py`.