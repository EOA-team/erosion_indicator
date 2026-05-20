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
import torch
import warnings
warnings.simplefilter("ignore")
import sys
sys.path.insert(0, os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure/SALI_models'))
from src.model_utils import load_all_models


def sample_locations(crop_labels, lnf_dir, tot_samples, save_path, lnf_codes, seed=42):
    """Sample locations with a certain crop type, uniformly across crop types and available years (yearly crop maps) (EPSG:2056)"""

    lnf_files = sorted([f for f in os.listdir(lnf_dir) if f.endswith('.gpkg')])
    n_years = len(lnf_files)
    n_crops = len(lnf_codes)
    samples_per_year = int(np.floor(tot_samples/n_years))
    samples_per_crop = int(np.floor(samples_per_year/n_crops))

    print(f'Sampling {tot_samples}: {samples_per_year} per year, {samples_per_crop} per crop in a year')
    all_samples = []
    for lnf_yr in lnf_files:
        yr = lnf_yr.split('lnf')[-1].split('.gpkg')[0]
        lnf = gpd.read_file(os.path.join(lnf_dir, lnf_yr))
        # Filter to keep only relevant polys
        lnf = lnf[lnf.lnf_code.isin(lnf_codes)]
        lnf["poly_id"] = lnf.index
        if not len(lnf):
            continue
        seed += 1 # so that among differetn years it varies
        for crop in lnf_codes:
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


def sample_locations_with_field(crop_labels, lnf_dir, tot_samples, save_path, lnf_codes, yrs, target_crs=32632, seed=42):
    """Sample locations with a certain crop type, uniformly across crop types and available years (yearly crop maps)"""

    # Consider only 2021-2025 (to match LNF code)
    lnf_files = sorted([f for f in os.listdir(lnf_dir) if f.endswith('.gpkg')])
    lnf_files = [f for f in lnf_files if yrs[0] <= int(f.split('lnf')[-1].split('.gpkg')[0]) <= yrs[-1]]

    n_years = len(lnf_files)
    n_crops = len(lnf_codes)
    samples_per_year = int(np.floor(tot_samples/n_years))
    samples_per_crop = int(np.floor(samples_per_year/n_crops))

    print(f'Sampling {tot_samples}: {samples_per_year} per year, {samples_per_crop} per crop in a year')
    all_samples = []
    for lnf_yr in lnf_files:
        yr = lnf_yr.split('lnf')[-1].split('.gpkg')[0]
        lnf = gpd.read_file(os.path.join(lnf_dir, lnf_yr))
        # Some years (2021) have duplicated geometries
        lnf["geom_wkb"] = lnf.geometry.to_wkb()
        lnf = lnf.drop_duplicates(subset=["lnf_code", "geom_wkb"])
        lnf = lnf.drop(columns="geom_wkb")
        # Filter to keep only relevant polys
        lnf = lnf[lnf.lnf_code.isin(lnf_codes)]
        lnf["poly_id"] = lnf.index
        if not len(lnf):
            continue
        seed += 1 # so that among different years it varies
        for crop in lnf_codes:
            crop_polys = lnf[lnf.lnf_code == crop].to_crs(target_crs)
            if len(crop_polys) == 0:
                continue
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


def _agis_shortlist(
    nutzung_csv,
    lnf_mapping_csv,
    lnf_labels_path,
    yrs,
    top_arable_codes,
    grassland_codes,
    rules=('single_crop_farm', 'grassland_plus_one', 'dominant_crop'),
    grass_min_share=0.50,
    arable_min_share=0.02,
    other_max_share=0.10,
    dominant_share=0.80,
):
    """Build a (Flaechen_ID, Jahr, kulturcode) shortlist of "clean" arable fields
    from AGIS Nutzungsdaten, applying the selection rules from check_agis.py.

    `top_arable_codes` and `grassland_codes` are LNF codes (matching `kulturcode`
    in tbl_kulturmapping). `rules` is a subset of:
        - 'single_crop_farm'  : farm grows literally one crop, top-arable
        - 'grassland_plus_one': >=grass_min_share grassland + exactly 1 top-arable
        - 'dominant_crop'     : one top-arable crop covers >=dominant_share of farm

    Returns a DataFrame with columns:
        Flaechen_ID, betr_ID, Jahr, kulturcode, Kultur_nutzung, land_type,
        flaeche_bewirt, selection_reason
    """
    rules = tuple(rules)
    valid = {'single_crop_farm', 'grassland_plus_one', 'dominant_crop'}
    bad = set(rules) - valid
    if bad:
        raise ValueError(f"Unknown AGIS selection rule(s): {bad}. Valid: {valid}")
    if not rules:
        raise ValueError("At least one AGIS selection rule must be provided.")

    print(f"Building AGIS shortlist with rules: {rules}")

    # Load AGIS field-year table and crop-name lookup
    df_nutzung = pd.read_csv(nutzung_csv, encoding="latin1", sep=";")
    df_mapping = (
        pd.read_csv(lnf_mapping_csv, encoding="latin1", sep=";")
          [['kulturcode', 'Kultur_nutzung']]
          .drop_duplicates()
    )

    # Restrict to relevant years up-front
    df_nutzung = df_nutzung[df_nutzung['Jahr'].isin(yrs)].copy()

    # Attach kulturcode to every field row, classify by land_type
    df_nutzung = df_nutzung.merge(df_mapping, on='Kultur_nutzung', how='left')
    df_nutzung['land_type'] = np.select(
        [df_nutzung['kulturcode'].isin(grassland_codes),
         df_nutzung['kulturcode'].isin(top_arable_codes)],
        ['grassland', 'arable_top'],
        default='other'
    )

    # Aggregate to (farm, year, kulturcode) and compute shares
    df_crop_area = (
        df_nutzung.groupby(
            ['betr_ID', 'Jahr', 'kulturcode', 'Kultur_nutzung', 'land_type'],
            as_index=False
        )['flaeche_bewirt'].sum()
    )
    df_crop_area['farm_area'] = (
        df_crop_area.groupby(['betr_ID', 'Jahr'])['flaeche_bewirt'].transform('sum')
    )
    df_crop_area['share'] = df_crop_area['flaeche_bewirt'] / df_crop_area['farm_area']

    # Per-(farm, year) shares by land_type, and count of top-arable crops
    shares_by_type = (
        df_crop_area.groupby(['betr_ID', 'Jahr', 'land_type'], as_index=False)['share']
                    .sum()
                    .pivot(index=['betr_ID', 'Jahr'], columns='land_type', values='share')
                    .fillna(0.0)
                    .reset_index()
    )
    for col in ['grassland', 'arable_top', 'other']:
        if col not in shares_by_type.columns:
            shares_by_type[col] = 0.0

    n_arable_top = (
        df_crop_area[df_crop_area['land_type'] == 'arable_top']
            .groupby(['betr_ID', 'Jahr'])['kulturcode'].nunique()
            .rename('n_arable_top_crops')
            .reset_index()
    )
    farm_summary = shares_by_type.merge(n_arable_top, on=['betr_ID', 'Jahr'], how='left')
    farm_summary['n_arable_top_crops'] = farm_summary['n_arable_top_crops'].fillna(0).astype(int)

    out_cols = ['Flaechen_ID', 'betr_ID', 'Jahr', 'kulturcode', 'Kultur_nutzung',
                'land_type', 'flaeche_bewirt']
    selections = []

    # --- Rule 1: single-crop farms ---
    if 'single_crop_farm' in rules:
        n_crops_per_farm = (
            df_crop_area.groupby(['betr_ID', 'Jahr'])['kulturcode']
                        .nunique()
                        .rename('n_crops')
                        .reset_index()
        )
        single = n_crops_per_farm[n_crops_per_farm['n_crops'] == 1][['betr_ID', 'Jahr']]
        f1 = (
            df_nutzung.merge(single, on=['betr_ID', 'Jahr'], how='inner')
                      .query('kulturcode in @top_arable_codes')
                      .copy()
        )
        f1 = f1[[c for c in out_cols if c in f1.columns]]
        f1['selection_reason'] = 'single_crop_farm'
        print(f"  single_crop_farm   : {len(f1):>7d} fields")
        selections.append(f1)

    # --- Rule 2: grassland-dominated + exactly one top-arable crop ---
    if 'grassland_plus_one' in rules:
        mask = (
            (farm_summary['grassland'] >= grass_min_share) &
            (farm_summary['n_arable_top_crops'] == 1) &
            (farm_summary['arable_top'] >= arable_min_share) &
            (farm_summary['other'] <= other_max_share)
        )
        qualifying = farm_summary.loc[mask, ['betr_ID', 'Jahr']]
        f2 = (
            df_nutzung.merge(qualifying, on=['betr_ID', 'Jahr'], how='inner')
                      .query('kulturcode in @top_arable_codes')
                      .copy()
        )
        f2 = f2[[c for c in out_cols if c in f2.columns]]
        f2['selection_reason'] = 'grassland_plus_one'
        print(f"  grassland_plus_one : {len(f2):>7d} fields")
        selections.append(f2)

    # --- Rule 3: a single top-arable crop dominates the farm ---
    if 'dominant_crop' in rules:
        dominant = (
            df_crop_area[df_crop_area['kulturcode'].isin(top_arable_codes) &
                         (df_crop_area['share'] >= dominant_share)]
                [['betr_ID', 'Jahr', 'kulturcode']]
                .rename(columns={'kulturcode': 'dominant_kulturcode'})
        )
        f3 = (
            df_nutzung.merge(dominant, on=['betr_ID', 'Jahr'], how='inner')
                      .query('kulturcode == dominant_kulturcode')
                      .copy()
        )
        f3 = f3[[c for c in out_cols if c in f3.columns]]
        f3['selection_reason'] = 'dominant_crop'
        print(f"  dominant_crop      : {len(f3):>7d} fields")
        selections.append(f3)

    fields_all = pd.concat(selections, ignore_index=True)

    # Collapse duplicates across rules: same field can match several rules → join reasons
    key_cols   = ['Flaechen_ID', 'betr_ID', 'Jahr']
    other_cols = [c for c in fields_all.columns if c not in key_cols + ['selection_reason']]
    fields_all = (
        fields_all.groupby(key_cols, as_index=False)
                  .agg({**{c: 'first' for c in other_cols},
                        'selection_reason': lambda x: ','.join(sorted(set(x)))})
    )

    print(f"  → unique clean fields: {len(fields_all)} "
          f"across {fields_all['kulturcode'].nunique()} crops, "
          f"{fields_all['Jahr'].nunique()} years")

    return fields_all


def sample_locations_with_field_agis(
    lnf_dir,
    nutzung_csv,
    lnf_mapping_csv,
    lnf_labels_path,
    tot_samples,
    save_path,
    arable_codes,
    grass_codes,
    yrs,
    grass_codes_for_shares=None,
    rules=('single_crop_farm', 'grassland_plus_one', 'dominant_crop'),
    grass_min_share=0.50,
    arable_min_share=0.02,
    other_max_share=0.10,
    dominant_share=0.80,
    target_crs=32632,
    seed=42,
    lnf_id_col_by_year=None,
):
    """AGIS-informed analogue of sample_locations_with_field.

    Builds an AGIS shortlist of "clean" arable fields (using one or more
    selection rules from check_agis.py), joins it onto yearly LNF geometries
    via the LNF field-id column matching Flaechen_ID, and samples one point
    per selected field — same downstream contract as sample_locations_with_field
    (poly_id, lnf_code, point_geom, polygon_geom, x, y, yr).

    Selection rules (`rules`, applied as a union, dedup by (Flaechen_ID, Jahr))
    --------------------------------------------------------------------------
    A *top-arable crop* means an LNF code in `arable_codes` (the top-20 arable
    crops by area across `yrs`, minus `exclude_codes` and any codes overridden
    to grassland via `config['grassland_codes']`).

    - 'single_crop_farm'   — the farm grows exactly one crop, AND that crop is
                             a top-arable crop. The single field of that crop
                             is kept.
    - 'grassland_plus_one' — top-arable fields on farms that satisfy ALL of:
                                grassland share         >= grass_min_share   (50%)
                                # of top-arable crops   == 1
                                that arable crop's share>= arable_min_share  (2%)
                                "other" land use share  <= other_max_share   (10%)
                             Only the arable field is kept (grassland fields on
                             the same farm are excluded).
    - 'dominant_crop'      — fields growing crop X, where X is a top-arable
                             crop covering >= dominant_share (80%) of farm area.
                             Only fields of crop X are kept; companion fields
                             (the remaining <=20%) are excluded.

    The three rule outputs are then unioned and deduplicated on
    (Flaechen_ID, Jahr); a single field can satisfy multiple rules and its
    `selection_reason` is the comma-joined list. Land-use shares are computed
    over `grass_codes_for_shares` (typically the wide Grassland-lv3 set ∪
    config overrides) and `top_arable_codes`; everything else is 'other'.

    On top of the union, a per-(year, crop) sampling cap is applied:
    `tot_samples / n_years / n_crops` fields are drawn area-weighted, WITHOUT
    replacement, and capped at the number of available clean fields. Crops
    that don't reach the cap log a warning and use everything they have.

    Grassland sampling
    ------------------
    AGIS "cleanness" rules target arable crops; this mirrors check_agis.py.
    Grassland codes in `grass_codes` (typically narrow, e.g. [601]) are sampled
    the standard random way directly from the LNF polygons — they are NOT
    subject to the AGIS filtering above. So the strict cleanness guarantee
    covers arable C-factor calibration but not the grassland baseline.

    Parameters
    ----------
    grass_codes : list[int]
        Grassland LNF codes from which to *draw FC samples* (typically narrow).
    grass_codes_for_shares : list[int] or None
        LNF codes that count as "grassland" when computing farm-level land
        shares inside the AGIS rules (typically wide — every Grassland-lv3
        code plus overrides). Falls back to `grass_codes` if None.
    lnf_id_col_by_year : dict[int, str] or None
        Map each year to the LNF column holding the AGIS Flaechen_ID. The
        schema changed in 2023: <=2022 uses 'uuid', >=2023 uses
        'identifikator_be'. Pass a dict to override per year.
    rules, grass_min_share, arable_min_share, other_max_share, dominant_share
        See the rule definitions above. Defaults match check_agis.py.
    """
    if grass_codes_for_shares is None:
        grass_codes_for_shares = grass_codes

    # Default LNF id-column lookup: schema changed in 2023.
    default_id_col_by_year = {
        2019: 'uuid', 2020: 'uuid', 2021: 'uuid', 2022: 'uuid',
        2023: 'identifikator_be', 2024: 'identifikator_be', 2025: 'identifikator_be',
    }
    if lnf_id_col_by_year:
        default_id_col_by_year.update(lnf_id_col_by_year)
    lnf_id_col_by_year = default_id_col_by_year

    # AGIS shortlist (arable only — grassland is sampled separately below if requested)
    shortlist = _agis_shortlist(
        nutzung_csv=nutzung_csv,
        lnf_mapping_csv=lnf_mapping_csv,
        lnf_labels_path=lnf_labels_path,
        yrs=yrs,
        top_arable_codes=arable_codes,
        grassland_codes=grass_codes_for_shares,  # wide set, for share computation
        rules=rules,
        grass_min_share=grass_min_share,
        arable_min_share=arable_min_share,
        other_max_share=other_max_share,
        dominant_share=dominant_share,
    )
    # Normalize id types for the eventual uuid join (LNF uuid is str-like)
    shortlist['Flaechen_ID'] = shortlist['Flaechen_ID'].astype(str)

    # Find matching LNF files for the requested years
    lnf_files = sorted([f for f in os.listdir(lnf_dir) if f.endswith('.gpkg')])
    lnf_files = [f for f in lnf_files if yrs[0] <= int(f.split('lnf')[-1].split('.gpkg')[0]) <= yrs[-1]]

    n_years = len(lnf_files)
    n_crops_total = len(arable_codes) + len(grass_codes)
    samples_per_year = int(np.floor(tot_samples / max(n_years, 1)))
    samples_per_crop = int(np.floor(samples_per_year / max(n_crops_total, 1)))
    print(f"Target sampling: {tot_samples} total → {samples_per_year}/year, "
          f"{samples_per_crop}/crop/year (area-weighted, no replacement, "
          f"capped at availability)")

    all_samples = []
    for lnf_yr in lnf_files:
        yr = int(lnf_yr.split('lnf')[-1].split('.gpkg')[0])
        lnf = gpd.read_file(os.path.join(lnf_dir, lnf_yr))
        # Some LNF years have duplicated geometries — de-dup as in the original
        lnf["geom_wkb"] = lnf.geometry.to_wkb()
        lnf = lnf.drop_duplicates(subset=["lnf_code", "geom_wkb"])
        lnf = lnf.drop(columns="geom_wkb")
        # Filter to relevant codes
        lnf = lnf[lnf.lnf_code.isin(arable_codes + grass_codes)]
        lnf["poly_id"] = lnf.index
        if not len(lnf):
            continue

        # AGIS shortlist for this year (arable codes only)
        clean_yr = shortlist[shortlist['Jahr'] == yr][['Flaechen_ID', 'kulturcode']]
        # Resolve the LNF column that holds the AGIS Flaechen_ID for this year,
        # tolerating case / whitespace variation. The schema changed in 2023.
        expected = lnf_id_col_by_year.get(yr)
        if expected is None:
            raise KeyError(
                f"[{yr}] No LNF id column configured. Extend lnf_id_col_by_year "
                f"with {{{yr}: '<colname>'}}. Columns present: {lnf.columns.tolist()}"
            )
        cand = {c.strip().lower(): c for c in lnf.columns}
        if expected.lower() not in cand:
            raise KeyError(
                f"[{yr}] Expected LNF id column '{expected}' not found in {lnf_yr}. "
                f"Columns present: {lnf.columns.tolist()}. "
                "Pass lnf_id_col_by_year={{{yr}: '<actual colname>'}} to override."
            )
        id_col = cand[expected.lower()]
        if id_col != 'uuid':
            print(f"  [{yr}] LNF id column is '{id_col}' → renaming to 'uuid' for join")
            lnf = lnf.rename(columns={id_col: 'uuid'})
        lnf['uuid'] = lnf['uuid'].astype(str)
        is_grass  = lnf['lnf_code'].isin(grass_codes)
        is_arable = lnf['lnf_code'].isin(arable_codes)
        lnf_arable_clean = lnf[is_arable].merge(
            clean_yr.rename(columns={'Flaechen_ID': 'uuid'}),
            on='uuid', how='inner'
        )
        # Sanity check: AGIS kulturcode and LNF lnf_code should agree for the same field
        mism = lnf_arable_clean[lnf_arable_clean['lnf_code'] != lnf_arable_clean['kulturcode']]
        if len(mism):
            print(f"  [{yr}] {len(mism)} arable fields with AGIS kulturcode != LNF lnf_code — keeping LNF code")
        lnf_arable_clean = lnf_arable_clean.drop(columns='kulturcode')
        lnf_grass = lnf[is_grass]
        lnf_keep = pd.concat([lnf_arable_clean, lnf_grass], ignore_index=True)
        lnf_keep = gpd.GeoDataFrame(lnf_keep, geometry='geometry', crs=lnf.crs)
        print(f"  [{yr}] LNF universe: {len(lnf)} polys → after AGIS filter "
              f"(+grassland): {len(lnf_keep)} polys "
              f"({len(lnf_arable_clean)} arable + {len(lnf_grass)} grassland)")

        seed += 1
        for crop in arable_codes + grass_codes:
            crop_polys = lnf_keep[lnf_keep.lnf_code == crop].to_crs(target_crs)
            if len(crop_polys) == 0:
                continue

            n_target = min(samples_per_crop, len(crop_polys))
            if n_target == 0:
                continue
            if n_target < samples_per_crop:
                print(f"  [{yr}] crop {crop}: requested {samples_per_crop}, "
                      f"only {len(crop_polys)} clean fields available — using all of them")

            sampled_polys = crop_polys.sample(
                n=n_target,
                weights=crop_polys.area,
                replace=False,
                random_state=seed,
            )

            buffered = sampled_polys.copy()
            buffered["geometry"] = buffered.geometry.buffer(-10)
            buffered = buffered[~buffered.geometry.is_empty]
            buffered = buffered[buffered.geometry.notnull()]
            if not len(buffered):
                continue
            buffered["orig_idx"] = buffered.index

            pts = buffered.sample_points(1, random_state=seed)
            pts = pts.explode(index_parts=False)
            pts = gpd.GeoDataFrame(pts, geometry='sampled_points', crs=target_crs)\
                .rename(columns={'sampled_points': "point_geom"})
            pts["orig_idx"] = pts.index
            pts = pts.merge(
                buffered[["orig_idx", "poly_id", "lnf_code", "geometry"]],
                on="orig_idx", how="left",
            )
            pts = pts.rename(columns={"geometry": "polygon_geom"}).drop(columns="orig_idx")

            pts["x"] = pts.point_geom.x
            pts["y"] = pts.point_geom.y
            pts["yr"] = str(yr)  # keep string-typed yr to match the rest of the pipeline

            all_samples.append(pts)

    if not all_samples:
        raise RuntimeError("AGIS sampling produced no samples — check inputs and rules.")

    samples = gpd.GeoDataFrame(pd.concat(all_samples, ignore_index=True),
                               geometry="point_geom", crs=target_crs)
    print(f"Total sampled fields: {len(samples)} "
          f"across {samples['lnf_code'].nunique()} crops, "
          f"{samples['yr'].nunique()} years")
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
    print('Extract S2 (01 Dec previous yr to 31 Jan next yr) and Soil data')
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

        # Extract S2 data: add a month before and after
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
        soil_files = [f for f in soil_files if os.path.exists(f)]
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


def extract_precomputed_fc_field(save_path_sampledloc, s2_grid_path, fc_dir, save_path_FCpreds):
    """Extract pre-computed FC (pv, npv, soil) from fc_dir.

    Files must follow the same naming convention as raw S2 files (S2_left_top_year.*).
    Returns True on success, False if any required files are missing (triggers S2+prediction fallback).
    """
    print(f'Extracting pre-computed FC from {fc_dir}')
    samples = pd.read_pickle(save_path_sampledloc)

    grid = gpd.read_file(s2_grid_path)
    grid_samples = gpd.overlay(samples, grid, how='intersection')

    file_lookup = {}
    for fname in os.listdir(fc_dir):
        key = "_".join(fname.split("_")[:4])[:-4]
        file_lookup.setdefault(key, []).append(os.path.join(fc_dir, fname))

    df_out = []
    missing = []

    for (poly_id, yr), poly_df in grid_samples.groupby(['poly_id', 'yr']):
        polygon = poly_df.iloc[0]['polygon_geom']
        tiles = poly_df[['left', 'top']].drop_duplicates()
        tiles = [(row['left'], row['top']) for _, row in tiles.iterrows()]
        matched_files = []
        for (left, top) in tiles:
            for y in [int(yr), int(yr) - 1]:
                key = f"S2_{int(left)}_{int(top)}_{y}"
                matched_files.extend(file_lookup.get(key, []))

        if not matched_files:
            missing.append((poly_id, yr))
            continue
        matched_files = [f for f in matched_files if f.endswith('zarr')]
        ds_fc = xr.open_mfdataset(matched_files).sel(time=slice(f'{int(yr)-1}-06-01', f'{yr}-07-30'))
        # PROBLEM: has no SCL/cloud bands
        try:
            clipped = ds_fc.rename({'lat': 'y', 'lon': 'x'}).rio.write_crs(32632).rio.clip(
                [polygon], ds_fc.rio.crs, drop=True
            )
        except Exception:
            missing.append((poly_id, yr))
            continue

        df_fc = clipped.to_dataframe().reset_index()
        # Normalise capitalised variable names to lowercase
        df_fc = df_fc.rename(columns={'PV': 'pv', 'NPV': 'npv', 'Soil': 'soil'})
        fc_cols = [c for c in ['pv', 'npv', 'soil'] if c in df_fc.columns]
        if fc_cols:
            df_fc = df_fc.loc[(df_fc[fc_cols] != 0).any(axis=1)]

        df_fc["lnf_code"] = poly_df.iloc[0]["lnf_code"]
        df_fc["yr"] = yr
        df_fc["poly_id"] = poly_id

        sampled_x = poly_df.iloc[0]["x"]
        sampled_y = poly_df.iloc[0]["y"]
        snap_x, snap_y = snap_to_grid(sampled_x, sampled_y, ds_fc)
        df_fc["is_sample_pixel"] = (df_fc["x"] == snap_x) & (df_fc["y"] == snap_y)
        df_fc["sampled_x"] = sampled_x
        df_fc["sampled_y"] = sampled_y
        df_fc['x'] = df_fc['x'].apply(lambda v: v - 5)
        df_fc['y'] = df_fc['y'].apply(lambda v: v + 5)

        df_out.append(df_fc)

    if missing:
        print(f'  Pre-computed FC missing for {len(missing)} poly/yr combinations — falling back to S2 extraction and prediction')
        return False

    pd.concat(df_out, ignore_index=True).to_pickle(save_path_FCpreds)
    return True


def predict_FC(save_path_sampledloc_S2: str, save_path_FCpreds: str) -> None:
    """Predict fractional cover (PV, NPV, Soil) for all sampled pixels.

    Models are loaded once and each soil group is processed by its own model only,
    avoiding repeated disk I/O and redundant cross-group predictions.
    """
    bands = ['s2_B02','s2_B03','s2_B04','s2_B05','s2_B06','s2_B07','s2_B08','s2_B8A','s2_B11','s2_B12']
    df_samples = pd.read_pickle(save_path_sampledloc_S2)

    df_samples[df_samples == 65535] = np.nan
    df_samples = df_samples.dropna()
    df_samples[bands] /= 10000
    df_samples['soil_group'] = df_samples['soil_group'].astype(int)

    soil_groups = sorted(df_samples['soil_group'].unique().tolist())
    n_groups = len(soil_groups)
    print(f'Predict FC — {len(df_samples):,} rows, {n_groups} soil groups — loading models...', end=' ', flush=True)

    model_dir = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure/SALI_models/FC/models/')
    models_dict = load_all_models(model_dir, soil_groups)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('done', flush=True)

    def _run_ensemble(data: np.ndarray, models: dict, chunk_size: int = 50000) -> np.ndarray:
        """Run 5-iteration ensemble for one soil group; returns (n, 3) normalised array."""
        n = len(data)
        acc = np.zeros((n, 3), dtype=np.float32)
        for iteration in range(1, 6):
            m_pv  = models[iteration]['PV']
            m_npv = models[iteration]['NPV']
            m_s   = models[iteration]['Soil']
            m_pv.eval(); m_npv.eval(); m_s.eval()
            for start in range(0, n, chunk_size):
                X = torch.from_numpy(data[start:start + chunk_size]).float().to(device)
                with torch.no_grad():
                    acc[start:start + chunk_size, 0] += m_pv(X).cpu().numpy().squeeze()
                    acc[start:start + chunk_size, 1] += m_npv(X).cpu().numpy().squeeze()
                    acc[start:start + chunk_size, 2] += m_s(X).cpu().numpy().squeeze()
        acc /= 5.0
        acc = acc.clip(0, 1)
        row_sums = acc.sum(axis=1, keepdims=True)
        acc /= np.where(row_sums > 0, row_sums, 1.0)
        return acc

    result_parts = []
    for idx, sg in enumerate(soil_groups):
        subset = df_samples[df_samples['soil_group'] == sg].copy()
        preds = _run_ensemble(subset[bands].to_numpy(), models_dict[sg])
        subset[['pv', 'npv', 'soil']] = preds
        result_parts.append(subset)
        print(f'\r  [{idx + 1}/{n_groups}] soil_group={sg}: {len(subset):,} rows done  ', end='', flush=True)

    print(flush=True)
    df_out = pd.concat(result_parts).sort_index().drop(columns=['soil_group'])
    df_out.to_pickle(save_path_FCpreds)


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

    When S2 bands are present, applies full cloud/shadow/snow masking.
    When only pre-computed FC (pv/npv/soil) is present, drops NaN-only rows.
    """
    band_cols = [c for c in field_df.columns if c.startswith('s2_B')]

    if band_cols:
        missing_mask = (field_df[band_cols].isna()).all(axis=1)
        missing_mask2 = (field_df[band_cols] == 65535).all(axis=1)
        cloud_mask = (field_df['s2_mask'] == 1) | (field_df['s2_SCL'].isin([8, 9, 10]))
        shadow_mask = (field_df['s2_mask'] == 2) | (field_df['s2_SCL'] == 3)
        snow_mask = (field_df['s2_mask'] == 3) | (field_df['s2_SCL'] == 11)
        cirrus_mask = (field_df['s2_SCL'] == 10) & (field_df['s2_B02'] > cirrus_thresh)
        drop_mask = missing_mask | missing_mask2 | cloud_mask | shadow_mask | snow_mask | cirrus_mask
    else:
        fc_cols = [c for c in ['pv', 'npv', 'soil'] if c in field_df.columns]
        drop_mask = field_df[fc_cols].isna().all(axis=1) if fc_cols else pd.Series(False, index=field_df.index)

    field_df = field_df.copy()
    field_df['masked'] = drop_mask.astype(int)
    masked_frac_per_date = field_df.groupby('time')['masked'].mean()
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


def _gapfill_one_field_alr(keys, group_clean, ts_cols, value_cols, pixel_cols, alpha, max_gap_days=15, max_train_points=1000):
    """Return cleaned observations + GPR fill-points (only where gap > max_gap_days) for the sampled pixel.

    Training uses all clean field pixels; prediction is performed only at synthetic dates
    inserted in gaps longer than max_gap_days in the target pixel's clean timeseries.
    """
    if len(group_clean) < 3:
        return None

    target_clean = group_clean[group_clean["is_sample_pixel"]].copy()
    if len(target_clean) == 0:
        return None

    # Cleaned observations for the target pixel — always kept as-is
    keep_cols = ts_cols + pixel_cols + ["time"] + value_cols
    result_clean = target_clean[keep_cols].copy()
    result_clean["is_gapfilled"] = False
    result_clean["fc_quality_flag"] = 0

    # Dates to insert within gaps > max_gap_days
    clean_times = pd.DatetimeIndex(pd.to_datetime(target_clean["time"]).sort_values().unique())
    fill_dates = generate_fill_dates(clean_times, max_gap_days)

    if len(fill_dates) == 0:
        return result_clean.sort_values("time").reset_index(drop=True)

    # --- Feature engineering (train on all field pixels) ---
    dates_train = pd.to_datetime(group_clean["time"])
    t0 = dates_train.min()
    t_train = (dates_train - t0).dt.days.values.reshape(-1, 1)

    t_fill = np.array([(d - t0).days for d in fill_dates], dtype=float).reshape(-1, 1)

    sin_train = np.sin(2 * np.pi * (t_train % 365) / 365)
    cos_train = np.cos(2 * np.pi * (t_train % 365) / 365)
    sin_fill  = np.sin(2 * np.pi * (t_fill  % 365) / 365)
    cos_fill  = np.cos(2 * np.pi * (t_fill  % 365) / 365)

    t_train_norm = t_train / 365.0
    t_fill_norm  = t_fill  / 365.0

    p_scaler    = StandardScaler()
    pixel_train = p_scaler.fit_transform(group_clean[pixel_cols].values)
    # Fill dates all belong to the single target pixel
    target_pixel = target_clean[pixel_cols].values[:1]
    pixel_fill   = p_scaler.transform(np.repeat(target_pixel, len(fill_dates), axis=0))

    n_pixel_dims = pixel_train.shape[1]
    n_features   = 3 + n_pixel_dims   # t, sin, cos, spatial

    X_train_full = np.hstack([t_train_norm, sin_train, cos_train, pixel_train]).astype(np.float32)
    X_pred       = np.hstack([t_fill_norm,  sin_fill,  cos_fill,  pixel_fill ]).astype(np.float32)

    fc_train_full  = group_clean[value_cols].values.astype(np.float32)
    alr_train_full = alr_transform(fc_train_full)

    # --- Stratified subsample: always keep target pixel, fill budget from others ---
    target_mask = group_clean["is_sample_pixel"].values.astype(bool)

    if len(group_clean) > max_train_points:
        n_target  = target_mask.sum()
        budget    = max(0, max_train_points - n_target)
        other_idx = np.where(~target_mask)[0]

        if budget > 0 and len(other_idx) > 0:
            other_times = t_train_norm[other_idx, 0]
            n_bins      = min(10, budget)
            bin_edges   = np.linspace(other_times.min(), other_times.max(), n_bins + 1)
            bin_ids     = np.digitize(other_times, bin_edges) - 1
            per_bin     = max(1, budget // n_bins)
            rng         = np.random.default_rng(42)
            chosen_other = []
            for b in range(n_bins):
                in_bin = other_idx[bin_ids == b]
                if len(in_bin) > 0:
                    chosen_other.append(rng.choice(in_bin, size=min(per_bin, len(in_bin)), replace=False))
            chosen_other = np.concatenate(chosen_other) if chosen_other else np.array([], dtype=int)
            keep = np.concatenate([np.where(target_mask)[0], chosen_other])
        else:
            keep = np.where(target_mask)[0]

        X_train   = X_train_full[keep]
        alr_train = alr_train_full[keep]
    else:
        X_train   = X_train_full
        alr_train = alr_train_full

    estimated_alpha = float(np.var(alr_train, axis=0).mean()) * 0.05
    estimated_alpha = float(np.clip(estimated_alpha, 1e-4, 0.5))

    ls_init   = [1.0] * n_features
    ls_bounds = (
        [(0.1,  5.0)]  * 1           +   # t
        [(0.5, 10.0)]  * 2           +   # sin, cos
        [(0.05, 10.0)] * n_pixel_dims    # spatial
    )
    base_kernel = ConstantKernel(1.0, (0.1, 5.0)) * Matern(
        length_scale=ls_init,
        length_scale_bounds=ls_bounds,
        nu=1.5
    ) + WhiteKernel(noise_level=estimated_alpha, noise_level_bounds=(1e-5, 0.5))

    alr_preds = np.zeros((len(X_pred), 2))
    alr_unc   = np.zeros((len(X_pred), 2))
    for j in range(2):
        y_train = alr_train[:, j]
        if j == 0:
            gp = GaussianProcessRegressor(
                kernel=base_kernel, alpha=estimated_alpha,
                normalize_y=True, n_restarts_optimizer=5
            )
            gp.fit(X_train, y_train)
            fitted_kernel = gp.kernel_
        else:
            gp = GaussianProcessRegressor(
                kernel=fitted_kernel, alpha=estimated_alpha,
                normalize_y=True, n_restarts_optimizer=2
            )
            gp.fit(X_train, y_train)

        mean, std = gp.predict(X_pred, return_std=True)
        alr_preds[:, j] = mean
        alr_unc[:, j]   = std

    fc_pred = alr_inverse(alr_preds)

    # Quality flag on fill points (based on ALR uncertainty)
    total_unc = np.linalg.norm(alr_unc, axis=1)
    p80 = np.percentile(total_unc, 80)
    p95 = np.percentile(total_unc, 95)
    qflag = np.zeros(len(total_unc), dtype=int)
    qflag[total_unc > p80] = 1
    qflag[total_unc > p95] = 2

    # Build fill-points dataframe
    ts_dict    = dict(zip(ts_cols, keys if isinstance(keys, tuple) else (keys,)))
    pixel_dict = dict(zip(pixel_cols, target_clean[pixel_cols].iloc[0]))
    fill_result = pd.DataFrame({"time": fill_dates.astype("datetime64[ns]"), **ts_dict, **pixel_dict})
    for i, col in enumerate(value_cols):
        fill_result[col] = fc_pred[:, i]
    fill_result["is_gapfilled"]    = True
    fill_result["fc_quality_flag"] = qflag

    return (
        pd.concat([result_clean, fill_result], ignore_index=True)
        .sort_values("time")
        .reset_index(drop=True)
    )


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


def gapfill_dataframe_gpr_per_field_parallel(df_clean, ts_cols, value_cols, pixel_cols, alpha=1e-2, max_gap_days=15, n_jobs=1):

    tasks = [(keys, group) for keys, group in df_clean.groupby(ts_cols)]
    print(f"To process: {len(tasks)} fields using {n_jobs} workers")

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_gapfill_one_field_alr)(
            keys, group_clean,
            ts_cols, value_cols, pixel_cols, alpha, max_gap_days
        )
        for keys, group_clean in tasks
    )

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


def run_sampling_pipeline(config: dict) -> None:
    """Run the full sampling + gapfilling pipeline. Skips steps whose output already exists."""

    # =====================================
    # Identify main crops to consider: 
    lnf_labels_path = os.path.expanduser(config['lnf_labels_path'])
    lnf_dir         = os.path.expanduser(config['lnf_dir'])
    exclude_codes = config['lnf_ignore_codes']
    yrs = config['years']

    # Resolve LNF codes for top arable crops
    top_crops = config.get('top_crops')
    df_labels = pd.read_excel(lnf_labels_path, sheet_name='label_sheet')
    if top_crops is None:
        df_arable = df_labels[df_labels['Crop_Label_lv3'].isin(['Arable Land'])]
        # Years used to pick top crops by area. Defaults to `yrs` for backward
        # compat, but the LNF spreadsheet's per-year area columns follow the
        # pattern `<yr>_Area_m<(yr+1) % 100>` only from 2021 onward (e.g.
        # '2021_Area_m22', '2022_Area_m23', ...); 2019/2020 use a different
        # naming ('2019_Area_m2', '2020_Area_m2') that the templated lookup
        # can't address. Pass a separate `top_crops_area_years` (e.g.
        # [2022, 2023, 2024]) when sampling a wider range of years.
        area_yrs = config.get('top_crops_area_years', yrs)
        area_cols = [f'{yr}_Area_m{str(yr+1)[-2:]}' for yr in area_yrs]
        missing = [c for c in area_cols if c not in df_arable.columns]
        if missing:
            print(f"[top_crops] Missing area columns in LNF labels file: {missing} "
                  f"— skipping. Available area columns: "
                  f"{[c for c in df_arable.columns if 'Area_m' in c]}")
            area_cols = [c for c in area_cols if c in df_arable.columns]
        if not area_cols:
            raise ValueError(
                "Cannot auto-detect top crops: none of the requested area "
                "columns exist in the LNF labels file. Either fix "
                "`top_crops_area_years` or pass an explicit `top_crops` list."
            )
        print(f"[top_crops] Picking top-20 arable crops by area from columns: {area_cols}")
        top_crops = set()
        for c in area_cols:
            top_crops.update(df_arable.sort_values(by=c, ascending=False)[:20]['Crop_EN'].tolist())

    arable_codes      = df_labels[df_labels['Crop_EN'].isin(top_crops)]['LNF_code'].unique().tolist()
    arable_codes      = [c for c in arable_codes if c not in exclude_codes]
    # Grassland overrides from config take precedence over the lv3 label —
    # mirrors GRASSLAND_OVERRIDE in check_agis.py. E.g. code 601 is labeled
    # 'Arable Land' in lv3 but is grassland in practice; if it's listed in
    # config['grassland_codes'] it must be stripped out of arable_codes here.
    grass_override = [c for c in config.get('grassland_codes', []) if c not in exclude_codes]
    arable_codes   = [c for c in arable_codes if c not in grass_override]
    grass_codes    = grass_override
    lnf_codes = arable_codes + grass_codes
    print(f"LNF codes (crops: {len(arable_codes)}, grass: {len(grass_codes)}): {lnf_codes}")

    # Wide grassland set for AGIS share computation (every Grassland-lv3 code
    # plus any narrow overrides from config['grassland_codes']). This is what
    # check_agis.py uses to decide whether a farm is "mostly grassland", and it
    # is distinct from grass_codes above, which is the (typically narrow) set
    # we actually draw FC samples from.
    grass_codes_lv3 = (
        df_labels[df_labels['Crop_Label_lv3'] == 'Grassland']['LNF_code']
        .unique().tolist()
    )
    grass_codes_for_shares = sorted(
        set(grass_codes_lv3) | set(config.get('grassland_codes', []))
    )
    grass_codes_for_shares = [c for c in grass_codes_for_shares if c not in exclude_codes]
    # Make sure none of these accidentally also live in the arable set
    grass_codes_for_shares = [c for c in grass_codes_for_shares if c not in arable_codes]

    code_name_map = dict(
        df_labels[df_labels['LNF_code'].isin(lnf_codes)]
        [['LNF_code', 'Crop_EN']]
        .drop_duplicates()
        .values
    )
    print("Code → Crop name mapping:")
    for code, name in code_name_map.items():
        print(f"{code}: {name}")

    # =====================================
    # Sample main crops locations (per year, across space) and save
    # Sampling is independent of how FC is obtained downstream — both the
    # precomputed-FC and S2+predict branches consume the same samples.pkl.
    samples_path = config['samples_path']
    if not os.path.exists(samples_path):
        strategy = config.get('sampling_strategy', 'random')
        if strategy == 'random':
            sample_locations_with_field(
                lnf_labels_path, lnf_dir,
                config['tot_samples'], samples_path,
                lnf_codes, yrs
            )
        elif strategy == 'agis':
            sample_locations_with_field_agis(
                lnf_dir=lnf_dir,
                nutzung_csv=os.path.expanduser(config['nutzung_csv']),
                lnf_mapping_csv=os.path.expanduser(config['lnf_mapping_csv']),
                lnf_labels_path=lnf_labels_path,
                tot_samples=config['tot_samples'],
                save_path=samples_path,
                arable_codes=arable_codes,
                grass_codes=grass_codes,
                grass_codes_for_shares=grass_codes_for_shares,
                yrs=yrs,
                rules=tuple(config.get('agis_rules',
                                       ('single_crop_farm',
                                        'grassland_plus_one',
                                        'dominant_crop'))),
                grass_min_share=config.get('agis_grass_min_share',  0.50),
                arable_min_share=config.get('agis_arable_min_share', 0.02),
                other_max_share=config.get('agis_other_max_share',   0.10),
                dominant_share=config.get('agis_dominant_share',     0.80),
                lnf_id_col_by_year=config.get('lnf_id_col_by_year'),
            )
        else:
            raise ValueError(
                f"Unknown sampling_strategy={strategy!r}. Use 'random' or 'agis'."
            )

    # =====================================
    # Extract FC: use pre-computed files if available, otherwise extract S2 + predict
    use_precomputed = config['fc_precompute']
    fc_preds_path = config['fc_preds_path']
    if not os.path.exists(fc_preds_path):
        if use_precomputed:
            fc_dir = os.path.expanduser(config.get('fc_dir', '~/mnt/eo-nas1/data/satellite/sentinel2/FC'))
            extract_precomputed_fc_field(
                samples_path,
                os.path.expanduser(config['s2_grid_path']),
                fc_dir,
                fc_preds_path
            )
        else:
            samples_s2_path = config['samples_s2_path']
            if not os.path.exists(samples_s2_path):
                extract_s2_data_field(
                    samples_path,
                    os.path.expanduser(config['s2_grid_path']),
                    os.path.expanduser(config['s2_dir']),
                    os.path.expanduser(config['soil_dir']),
                    samples_s2_path
                )
            predict_FC(samples_s2_path, fc_preds_path)

    # =====================================
    # Clean and filter timeseries
    df_samples = pd.read_pickle(fc_preds_path)

    # Remove bad dates within each field
    ts_cols = ['lnf_code', 'yr', 'poly_id']
    cirrus_thresh = config.get('cirrus_thresh')
    max_missing_frac = config.get('max_missing_frac')
    df_clean = (
        df_samples.groupby(ts_cols, group_keys=False)
                  .apply(clean_timeseries_field, max_missing_frac=max_missing_frac, cirrus_thresh=cirrus_thresh)
                  .reset_index(drop=True)
    )

    # Remove entire field at date if too much data was dropped per date
    n_before = df_samples.groupby(ts_cols).size().rename("n_total")
    n_after  = df_clean.groupby(ts_cols).size().rename("n_kept")
    df_drop_stats = pd.concat([n_before, n_after], axis=1).fillna(0)
    df_drop_stats["n_kept"]        = df_drop_stats["n_kept"].astype(int)
    df_drop_stats["n_dropped"]     = df_drop_stats["n_total"] - df_drop_stats["n_kept"]
    df_drop_stats["drop_fraction"] = df_drop_stats["n_dropped"] / df_drop_stats["n_total"]
    df_drop_stats = df_drop_stats.reset_index()

    drop_fraction_threshold = config.get('drop_fraction_threshold', 0.7)
    filtered_stats    = df_drop_stats[df_drop_stats["drop_fraction"] <= drop_fraction_threshold]
    keys_to_keep      = filtered_stats[ts_cols]
    df_clean_filtered = df_clean.merge(keys_to_keep, on=ts_cols, how="inner")
    df_samples_filtered = df_samples.merge(keys_to_keep, on=ts_cols, how="inner")
    print(f"Fields retained: {len(filtered_stats)}/{len(df_drop_stats)}")

    # De-duplicate overlapping satellite pixels
    group_cols = ['poly_id', 'x', 'y', 'time', 'yr', 'sampled_x', 'sampled_y', 'lnf_code', 'is_sample_pixel']
    df_clean_filtered   = df_clean_filtered.groupby(group_cols, as_index=False).mean(numeric_only=True)
    df_samples_filtered = df_samples_filtered.groupby(group_cols, as_index=False).mean(numeric_only=True)

    # ====================
    # Plot cleaned timeseries (before gapfilling)
    plot_ts_cols = ['lnf_code', 'poly_id', 'x', 'y', 'yr', 'sampled_x', 'sampled_y']
    for yr, df_year in df_clean_filtered.groupby('yr'):
        unique_ts   = df_year[plot_ts_cols].drop_duplicates()
        selected_ts = unique_ts.sample(n=min(5, len(unique_ts)), random_state=42)
        df_subset   = df_year.merge(selected_ts, on=plot_ts_cols, how='inner')

        unique_codes = df_subset['lnf_code'].unique()
        cmap         = plt.get_cmap('tab20')
        color_dict   = {code: cmap(i % 10) for i, code in enumerate(unique_codes)}

        fig, ax = plt.subplots(figsize=(10, 6))
        for keys, group in df_subset.groupby(plot_ts_cols):
            ax.plot(group.sort_values('time')['time'], group.sort_values('time')['pv'],
                    alpha=0.3, color=color_dict[keys[0]])

        ax.set_xlabel("Time")
        ax.set_ylabel("PV")
        ax.set_title(f"Cleaned pixel time series — Year {yr}")
        handles = [plt.Line2D([0], [0], color=color_dict[c], lw=2) for c in unique_codes]
        ax.legend(handles, unique_codes, title="lnf_code")
        plt.tight_layout()
        plt.savefig(f'plots/cleaned_timeseries_{yr}.png')
        plt.close()

    # =====================================
    # GPR gapfilling
    gapfilled_path = config['gapfilled_fc_path']
    if not os.path.exists(gapfilled_path):
        value_cols = ["pv", "npv", "soil"]
        pixel_cols = ["x", "y"]

        df_gpr = gapfill_dataframe_gpr_per_field_parallel(
            df_clean_filtered,
            ts_cols, value_cols, pixel_cols,
            alpha=1e-4,
            max_gap_days=config.get('max_gap_days', 15),
            n_jobs=config.get('n_jobs', 1)
        )

        df_gpr['time'] = pd.to_datetime(df_gpr['time'])
        start_dates = pd.to_datetime((df_gpr['yr'].astype(int) - 1).astype(str) + '-07-01')
        end_dates   = pd.to_datetime(df_gpr['yr'].astype(str) + '-06-30')
        df_gpr = df_gpr[(df_gpr['time'] >= start_dates) & (df_gpr['time'] <= end_dates)]
        df_gpr['fc_total'] = (df_gpr['pv'] + df_gpr['npv']) * 100
        df_gpr.to_parquet(gapfilled_path, index=False)

    # =====================================
    # Diagnostic plot: sample of gapfilled timeseries
    df_gpr = pd.read_parquet(gapfilled_path)
    plot_cols = ['lnf_code', 'yr', 'poly_id', 'x', 'y']

    unique_ts  = df_gpr[plot_cols].drop_duplicates()
    sampled_ts = unique_ts.sample(min(10, len(unique_ts)))

    nrows = len(sampled_ts)
    fig, axes = plt.subplots(nrows, 1, figsize=(10, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for idx, (_, row) in enumerate(sampled_ts.iterrows()):
        ax         = axes[idx]
        mask_raw   = (df_samples_filtered[plot_cols] == row[plot_cols]).all(axis=1)
        mask_clean = (df_clean_filtered[plot_cols]   == row[plot_cols]).all(axis=1)
        mask_gpr   = (df_gpr[plot_cols]              == row[plot_cols]).all(axis=1)
        df_raw_ts   = df_samples_filtered[mask_raw].sort_values('time')
        df_clean_ts = df_clean_filtered[mask_clean].sort_values('time')
        df_gpr_ts   = df_gpr[mask_gpr].sort_values('time')
        df_filled   = df_gpr_ts[df_gpr_ts['is_gapfilled']]
        print(f"TS {idx}: clean pts={len(df_clean_ts)}, fill pts={len(df_filled)}")

        ax.plot(df_raw_ts['time'],   df_raw_ts['pv'],   'o',  alpha=0.25, color='gray',      label='Raw PV')
        ax.scatter(df_clean_ts['time'], df_clean_ts['pv'],  marker='s', alpha=0.8,  color='steelblue', label='Cleaned PV')
        ax.scatter(df_filled['time'], df_filled['pv'], marker='x', s=80, color='red', zorder=5, label='GPR fill pts')
        ax.plot(df_gpr_ts['time'], df_gpr_ts['pv'], linestyle='--', alpha=0.8,  color='orange', label='Final')
        ax.set_title(f"{row['lnf_code']} - {row['yr']}  ({row['x']}, {row['y']})", fontsize=8)

    for j in range(nrows, len(axes)):
        fig.delaxes(axes[j])
    fig.supxlabel("Time")
    fig.supylabel("PV fraction")
    # Collect legend handles across all active subplots so entries are not missed
    # when the first subplot happens to have no GPR fill points
    legend_entries = {}
    for ax_i in axes[:nrows]:
        for handle, label in zip(*ax_i.get_legend_handles_labels()):
            if label not in legend_entries:
                legend_entries[label] = handle
    fig.legend(list(legend_entries.values()), list(legend_entries.keys()), loc='upper right')
    plt.tight_layout()
    plt.savefig("plots/gpr_all_timeseries_subplots.png")
    plt.close()


DEFAULT_CONFIG = {
    # ---- Sampling / gapfilling ----
    'lnf_labels_path': '~/mnt/eo-nas1/data/landuse/documentation/LNF_code_classification_20260217.xlsx',
    'lnf_dir':         '~/mnt/eo-nas1/data/landuse/raw',
    'top_crops': None,  # None = auto-detect top crops from LNF area stats; or pass an explicit list of crop names
    'lnf_ignore_codes': [553, 554, 555, 556, 559, 572, 594, 595, 598, 618, 625],  # arable or grassland classes to ignore
    'grassland_codes': [],  # LNF codes for grassland to include in sampling (e.g. [601, 611])
    'tot_samples':         10000,
    'samples_path':        'samples.pkl',
    'samples_s2_path':     'samples_data.pkl',
    'fc_dir':              '~/mnt/eo-nas1/data/satellite/sentinel2/FC',  # pre-computed FC; falls back to S2+prediction if files are missing
    's2_grid_path':        '~/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure/Project layers/gridface_s2tiles_CH.shp',
    's2_dir':              '~/mnt/eo-nas1/data/satellite/sentinel2/raw/CH',
    'soil_dir':            '~/mnt/eo-nas1/data/satellite/sentinel2/DLR_soilsuite_preds/',
    'fc_preds_path':       'samples_data_pred.pkl',
    'gapfilled_fc_path':   'samples_data_gpr.parquet',
    'max_gap_days':        15,
    'drop_fraction_threshold': 0.7,  # drop fields where >70% of observations are masked
    'n_jobs':              1,

    # ---- Sampling strategy ----
    # 'random' -> sample_locations_with_field (original: weighted random per crop/year)
    # 'agis'   -> sample_locations_with_field_agis (AGIS-clean fields only, per check_agis.py)
    'sampling_strategy': 'agis',
    # Used only when sampling_strategy == 'agis':
    'nutzung_csv':       '~/mnt/Data-Labo-RE/27_Natural_Resources-RE/321.4_WAUM_protected/Daten/Core_Snapshot/Agrarbericht_2025/tbl_nutzungsdaten.csv',
    'lnf_mapping_csv':   '~/mnt/Data-Labo-RE/27_Natural_Resources-RE/321.4_WAUM_protected/Daten/Core_Snapshot/Agrarbericht_2025/tbl_kulturmapping.csv',
    'agis_rules':              ('single_crop_farm', 'grassland_plus_one', 'dominant_crop'),
    'agis_grass_min_share':    0.50,
    'agis_arable_min_share':   0.02,
    'agis_other_max_share':    0.10,
    'agis_dominant_share':     0.80,
}


if __name__ == '__main__':
    run_sampling_pipeline(DEFAULT_CONFIG)