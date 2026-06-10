"""
C-factor calibration pipeline — main entry point.

Usage:
    taskset -c 15-20 python main.py                   # full pipeline
    taskset -c 15-20 python main.py --skip-sampling   # calibration only
"""
import argparse
import os

from sample_FC import run_sampling_pipeline
from calibrate_cfactor import run_calibration


CONFIG = {
    # ---- Sampling / gapfilling (sample_FC.py) ----
    'lnf_labels_path':     '~/mnt/eo-nas1/data/landuse/documentation/LNF_code_classification_20260217.xlsx', 
    'lnf_dir':             '~/mnt/eo-nas1/data/landuse/raw', 
    'top_crops':            None,
    'lnf_ignore_codes':    [553, 554, 555, 556, 559, 572, 594, 595, 598, 618, 625], # arable or grassland classes to ignore
    'grassland_codes':     [601],  # LNF codes for grassland to include in sampling (e.g. [601, 611]) -> 601 is actually labeled as arable
    'tot_samples':         10000,
    'samples_path':        'samplesv2.pkl',
    'samples_s2_path':     'samples_datav2.pkl', # not generated if FC precomputed
    'fc_precompute':        False,  # whether to use precomputed FC. PROBLEM: need to attach cloud cleaning bands from S2 data
    'fc_dir':              '~/mnt/eo-nas1/data/satellite/sentinel2/FC',
    's2_grid_path':        '~/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure/Project layers/gridface_s2tiles_CH.shp',
    's2_dir':              '~/mnt/eo-nas1/data/satellite/sentinel2/raw/CH',
    'soil_dir':            '~/mnt/eo-nas1/data/satellite/sentinel2/DLR_soilsuite_preds/',
    'fc_preds_path':       'samples_data_predv2.pkl',
    'cirrus_thresh':        500,  # Max value of blue band when SCL=10 (cirrus cloud masking)
    'max_missing_frac':     0.05,  # Max amount of missing data per date in a field timeseries
    'drop_fraction_threshold': 0.7,  # drop fields where >drop_fraction_threshold [0-1] of observations are masked
    'gapfilled_fc_path':    'samples_data_gprv2.parquet',
    'max_gap_days':         15,
    # Gapfilling method:
    #   'irregular' -> keep cleaned obs, GP-fill only gaps > max_gap_days
    #   'regular'   -> drop cleaned obs, GP-predict on a regular grid_step_days
    #                  cadence anchored at July 1 of yr-1 (agricultural year)
    'gapfill_method':      'regular',
    'grid_step_days':      10,
    'n_jobs':              2,
    'years': [2019, 2020, 2021, 2022, 2023, 2024], # LNF years to use for sampling AGIS fields
    # Years used to pick the "top arable crops by area" from the LNF spreadsheet.
    # The spreadsheet's per-year area columns are only well-defined for 2021+
    # ('2021_Area_m22', '2022_Area_m23', ...); 2019/2020 use the generic
    # '*_Area_m2' columns and can't be addressed by the templated name. So the
    # crop *selection* is driven by these years only, even when we *sample*
    # fields from a wider range of years.
    'top_crops_area_years': [2022, 2023, 2024],

    # ---- Sampling strategy ----
    # 'random' -> sample_locations_with_field (weighted random across LNF polygons)
    # 'agis'   -> sample_locations_with_field_agis (AGIS-clean fields via check_agis.py rules)
    'sampling_strategy': 'agis',
    # `lnf_mapping_csv` (kulturcode <-> Kultur_nutzung) is used both by the AGIS
    # shortlist AND by the C-factor crop grouping below, so it is read for both
    # sampling strategies.
    'lnf_mapping_csv':   '~/mnt/Data-Labo-RE/27_Natural_Resources-RE/321.4_WAUM_protected/Daten/Core_Snapshot/Agrarbericht_2025/tbl_kulturmapping.csv',
    # Used only when sampling_strategy == 'agis':
    'nutzung_csv':       '~/mnt/Data-Labo-RE/27_Natural_Resources-RE/321.4_WAUM_protected/Daten/Core_Snapshot/Agrarbericht_2025/tbl_nutzungsdaten.csv',
    'agis_rules':              ('single_crop_farm', 'grassland_plus_one', 'dominant_crop'),
    'agis_grass_min_share':    0.50,
    'agis_arable_min_share':   0.02,
    'agis_other_max_share':    0.20,
    'agis_dominant_share':     0.70,

    # ---- VERFAHREN (conservation-tillage) augmentation ----
    # The VERFAHREN sampler runs AUTOMATICALLY whenever sampling_strategy ==
    # 'agis': its known-tillage fields are ADDED to the AGIS sample (union,
    # de-duped on (uuid, yr); VERFAHREN rows win so the known tillage is kept).
    # The resulting `tillage_class` flows to the gapfilled parquet and is used by
    # stratified calibration to match the correct C-factor stratum (then dropped).
    # Same crop grouping (analogy_to_main folding) as the AGIS sampler is applied.
    # Canonical source: CSV with a real `betr_ID` matching AGIS. (The *_Kultur.xlsx
    # export uses BBS_ID/KT_ID_P, which do NOT match AGIS betr_ID.)
    'verfahren_path':          '~/mnt/Data-Labo-RE/27_Natural_Resources-RE/321.4_WAUM_protected/Daten/Erosionsrisiko/schonende_bodenbearbeitung.csv',
    # Budget for the VERFAHREN draw (separate from the AGIS `tot_samples`).
    'verfahren_tot_samples':   5000,
    # Years present in the conservation-tillage file (subset of `years`).
    'verfahren_years':         [2021, 2022],
    'verfahren_betr_id_col':   'betr_ID',
    'verfahren_uniqueness_on': 'verfahren',          # 'verfahren' | 'tillage_class'
    'verfahren_stratify_cols': ('lnf_code', 'tillage_class'),

    # ---- Crop grouping by identical C-factors (applies to BOTH strategies) ----
    # Crops whose stratified C-factor vector (Tal/Berg x Pflug/Mulch/Direkt) is
    # identical to a top crop are pooled into it: their fields enter the main
    # crop's draw and are relabelled to the main code (the true code is kept in
    # `orig_lnf_code`). Set False to disable pooling entirely.
    'group_crops_by_cfactor':   True,
    # The C-factor table path is shared with calibration — see
    # `c_factor_table_path` in the calibration section below (single source of
    # truth; the grouping step reads the same key).
    # Provenance file used to drop fully-estimated (n_estimate == 6) codes before
    # grouping. Set an explicit path — the default is a bare relative filename.
    'c_factor_provenance_path': 'c_factor_provenance.xlsx',

    # ---- Calibration (calibrate_cfactor.py) ----
    'ei_path':                  '../erosivity_index/predictions/grid_EI_daily_avg_pred_20260424_nn3.parquet',
    'c_factor_table_path':      '~/mnt/Data-Labo-RE/27_Natural_Resources-RE/321.4_WAUM_protected/Daten/Erosionsrisiko/C_Faktoren.csv',
    'lnf_classification_path':  '~/mnt/eo-nas1/data/landuse/documentation/LNF_code_classification_20260217.xlsx',
    'manual_overrides_path':    None, # only need it if any of sampled crops fail to auto-match LNF codes
    'results_folder':           'calibration_analysis_twobeta_noley',
    'calibration_results_path': 'calibration_results.csv',
    'ts_cols':                  ['lnf_code', 'yr', 'poly_id'],
    'crop_col':                 'lnf_code',
    'beta_bounds':              (1e-4, 0.1),
    # ---- Calibration mode: single vs two betas ----
    # 'single'   -> one global β; SLR(t) = exp(-β · fc_total), fc_total = (PV+NPV)·100
    #               (Matthews et al. 2023). β_opt is a scalar.
    # 'two_beta' -> separate β for PV and NPV;
    #               SLR(t) = exp(-(β_pv·PV·100 + β_npv·NPV·100)).
    #               PV and NPV are each scaled ×100 so the βs are directly
    #               comparable to the single-β value (two_beta reduces to
    #               single when β_pv == β_npv). 2-D L-BFGS-B fit. β_opt is a
    #               (β_pv, β_npv) tuple; beta.json records both.
    'calibration_mode':         'two_beta',
    # Optional per-component bounds for 'two_beta' (each falls back to
    # `beta_bounds` if None). Set explicit tuples to constrain PV/NPV separately.
    'beta_bounds_pv':           None,
    'beta_bounds_npv':          None,
    'exclude_calibration_lnf_codes': [601, 602], #[601, 611, 545, 546],  # Kunstwiesen, Extensiv genutzte Wiesen
    # FC is on a 0–100 scale here (PV+NPV scaled by 100). Matthews et al. (2023) found β ≈ 0.04 with FC on 0–100
    'area_weight_loss':         True, # If True, weight the loss by Swiss arable area per crop (areas from the LNF spreadsheet)
    # Years to average for the area weights — should match (or be a subset of)
    # the years used in the FC sampling step so weights describe the same
    # crop landscape as the calibration data.
    'area_years':               [2022, 2023, 2024],

    # ---- Stratified calibration (optional) ----
    # When `stratified_calibration` is True, calibration matches predicted
    # C-factors against the stratified C_ref table (`Tal_Pflug, Tal_Mulch,
    # Tal_Direkt, Berg_Pflug, Berg_Mulch, Berg_Direkt` columns of
    # C_Faktoren.csv) instead of the single `Total` column. Each sampled
    # pixel is assigned one stratum based on field altitude (Tal/Berg from
    # tbl_nutzungsdaten.swissALTI3D) and farm-year soil preparation
    # (Pflug/Mulch/Direkt from tbl_ressourceneffizienzbeitrag.reb_sb).
    #
    # Requires `sampling_strategy == 'agis'` so `uuid` (Flaechen_ID) and
    # `betr_ID` are present in the gapfilled parquet.
    'stratified_calibration':    True,
    'grenze_tal_berg':           600,           # m, swissALTI3D cutoff matching erosion_config.yaml
    'standardansaatverfahren':   'Pflug',       # tillage class when reb_sb is NA
    # REB table is keyed by (betr_ID, Jahr) — does not contain Flaechen_ID,
    # so within a farm-year with mixed reb_sb the tillage of any one pixel
    # is unknown. Assignment policy:
    #   'stochastic' — draw per-pixel from empirical reb_sb frequencies of
    #                  the farm-year (zero bias in expectation, default)
    #   'first_row'  — first row's reb_sb (matches the R-side `slice(1)`
    #                  in 05-Dataprep.R; deterministic but arbitrary)
    #   'mode'       — most frequent reb_sb in the farm-year
    'ressourceneffizienz_csv':   '~/mnt/Data-Labo-RE/27_Natural_Resources-RE/321.4_WAUM_protected/Daten/Core_Snapshot/Agrarbericht_2025/tbl_ressourceneffizienzbeitrag.csv',
    'tillage_assignment':        'stochastic',
    'tillage_random_seed':       42,
}

def main() -> None:
    parser = argparse.ArgumentParser(description='C-factor calibration pipeline')
    parser.add_argument(
        '--skip-sampling', action='store_true',
        help='Skip sampling + gapfilling (requires samples_data_gpr.parquet to exist)'
    )
    args = parser.parse_args()

    if not args.skip_sampling:
        print("=" * 60)
        print("STEP 1: Sampling + gapfilling")
        print("=" * 60)
        run_sampling_pipeline(CONFIG)
    else:
        if not os.path.exists(CONFIG['gapfilled_fc_path']):
            raise FileNotFoundError(
                f"--skip-sampling set but {CONFIG['gapfilled_fc_path']} not found. "
                "Run without --skip-sampling first."
            )
        print(f"Skipping sampling — using existing {CONFIG['gapfilled_fc_path']}")

    print("=" * 60)
    print("STEP 2: C-factor calibration")
    print("=" * 60)
    run_calibration(CONFIG)

    print("Done.")


if __name__ == '__main__':
    main()