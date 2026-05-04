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
    'lnf_labels_path': '~/mnt/eo-nas1/data/landuse/documentation/LNF_code_classification_20260217.xlsx', # will use stats of 2021-2024 
    'lnf_dir':         '~/mnt/eo-nas1/data/landuse/raw', # will use 2021-2024 
    'top_crops': None,
    'lnf_ignore_codes': [553, 554, 555, 556, 559, 572, 594, 595, 598, 618, 625], # arable or grassland classes to ignore
    'tot_samples':         10000, 
    'samples_path':        'samples.pkl',
    'samples_s2_path':     'samples_data.pkl',
    's2_grid_path':        '~/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure/Project layers/gridface_s2tiles_CH.shp',
    's2_dir':              '~/mnt/eo-nas1/data/satellite/sentinel2/raw/CH',
    'soil_dir':            '~/mnt/eo-nas1/data/satellite/sentinel2/DLR_soilsuite_preds/',
    'fc_preds_path':       'samples_data_pred.pkl',
    'gapfilled_fc_path':   'samples_data_gpr.parquet',
    'max_gap_days':        15,
    'n_jobs':              1,

    # ---- Calibration (calibrate_cfactor.py) ----
    'ei_path':                  '../erosivity_index/predictions/grid_EI_daily_avg_pred_20260424_nn3.parquet',
    'c_factor_table_path':      'C_Faktoren.csv',
    'lnf_classification_path':  '~/mnt/eo-nas1/data/landuse/documentation/LNF_code_classification_20260217.xlsx',
    'manual_overrides_path':    None, # only need it if any of sampled crops fail to auto-match LNF codes
    'calibration_results_path': 'calibration_results.csv',
    'ts_cols':                  ['lnf_code', 'yr', 'poly_id'],
    'crop_col':                 'lnf_code',
    'beta_bounds':              (1e-4, 0.1),
    # FC is on a 0–100 scale here (PV+NPV scaled by 100). Matthews et al. (2023)
    # found β ≈ 0.04 with FC on 0–1, so the equivalent here is ≈ 0.0004
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
