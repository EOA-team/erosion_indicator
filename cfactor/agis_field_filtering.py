import pandas as pd
import os
import numpy as np


# =============================================================================
# Identify farm-years where we can be confident a top arable field is "clean"
# (i.e. the field really is the AGIS-reported crop, with low contamination risk
# from other crops on the same farm). Three selection rules — outputs are
# Flaechen_ID / betr_ID lists, one per rule, plus a combined union.
#
#   Option 1: farm grows literally ONE crop, and it's a top arable crop.
#   Option 2: farm is mostly grassland (>=50%) plus exactly ONE top arable crop.
#   Option 3: a single top arable crop covers >=80% of the farm area.
#
# Output is at the FIELD level (one row per Flaechen_ID × Jahr).
# =============================================================================

# ---- Config ----
NUTZUNG_CSV     = os.path.expanduser('~/mnt/Data-Labo-RE/27_Natural_Resources-RE/321.4_WAUM_protected/Daten/Core_Snapshot/Agrarbericht_2025/tbl_nutzungsdaten.csv')
LNF_MAPPING     = os.path.expanduser('~/mnt/Data-Labo-RE/27_Natural_Resources-RE/321.4_WAUM_protected/Daten/Core_Snapshot/Agrarbericht_2025/tbl_kulturmapping.csv')
LNF_LABELS_PATH = os.path.expanduser('~/mnt/eo-nas1/data/landuse/documentation/LNF_code_classification_20260217.xlsx')

YRS                = [2022, 2023, 2024]
TOP_N_ARABLE       = 20
EXCLUDE_CODES      = [553, 554, 555, 556, 559, 572, 594, 595, 598, 618, 625]  # arable/grassland classes to ignore
GRASSLAND_OVERRIDE = [601]   # extra codes to treat as grassland (601 is labeled arable in lv3 but is grassland in practice)
GRASS_MIN_SHARE    = 0.50    # Option 2: farm is "mostly grassland" if grassland share >= this
ARABLE_MIN_SHARE   = 0.02    # Option 2: ignore farms with only token amounts of arable (avoids tiny fields)
OTHER_MAX_SHARE    = 0.10    # Option 2: cap on "other" land use (specialty crops, permanent, etc.)
DOMINANT_SHARE     = 0.80    # Option 3: a single top crop covers >= 80% of farm area

OUT_DIR = '.'  # where to write the CSV shortlists

# =============================================================================
# Load and prepare data
# =============================================================================

# tbl_nutzungsdaten has one row per FIELD per year (keyed by Flaechen_ID).
# A farm (betr_ID) can have many fields, possibly several growing the same crop.
df_nutzung = pd.read_csv(NUTZUNG_CSV, encoding="latin1", sep=";")
df_mapping = pd.read_csv(LNF_MAPPING, encoding="latin1", sep=";")[['kulturcode', 'Kultur_nutzung']].drop_duplicates()
df_labels  = pd.read_excel(LNF_LABELS_PATH, sheet_name='label_sheet')

# Identify top arable crops (over the relevant years) + grassland codes
df_arable_labels = df_labels[df_labels['Crop_Label_lv3'] == 'Arable Land']
top_crops_en = set()
for yr in YRS:
    col = f'{yr}_Area_m{str(yr + 1)[-2:]}'
    top_crops_en.update(
        df_arable_labels.sort_values(by=col, ascending=False)
                        .head(TOP_N_ARABLE)['Crop_EN']
                        .tolist()
    )
top_arable_codes = df_labels[df_labels['Crop_EN'].isin(top_crops_en)]['LNF_code'].unique().tolist()
top_arable_codes = [c for c in top_arable_codes if c not in EXCLUDE_CODES]

grassland_codes_lv3 = df_labels[df_labels['Crop_Label_lv3'] == 'Grassland']['LNF_code'].unique().tolist()
grassland_codes     = sorted(set(grassland_codes_lv3) | set(GRASSLAND_OVERRIDE))
grassland_codes     = [c for c in grassland_codes if c not in EXCLUDE_CODES]
# Make sure grassland codes don't end up in the "arable top" set
top_arable_codes    = [c for c in top_arable_codes if c not in grassland_codes]

code_name_map = dict(
    df_labels[df_labels['LNF_code'].isin(top_arable_codes)]
            [['LNF_code', 'Crop_EN']].drop_duplicates().values
)

# Attach kulturcode + land_type to every field row in df_nutzung
df_nutzung = df_nutzung.merge(df_mapping, on='Kultur_nutzung', how='left')
df_nutzung['land_type'] = np.select(
    [df_nutzung['kulturcode'].isin(grassland_codes),
     df_nutzung['kulturcode'].isin(top_arable_codes)],
    ['grassland', 'arable_top'],
    default='other'
)

# Compute per-(farm, year) shares by aggregating field areas
df_crop_area = (
    df_nutzung.groupby(['betr_ID', 'Jahr', 'kulturcode', 'Kultur_nutzung', 'land_type'],
                       as_index=False)['flaeche_bewirt']
              .sum()
)
df_crop_area['farm_area'] = df_crop_area.groupby(['betr_ID', 'Jahr'])['flaeche_bewirt'].transform('sum')
df_crop_area['share']     = df_crop_area['flaeche_bewirt'] / df_crop_area['farm_area']

# Shares aggregated by land_type, pivoted: one row per (farm, year)
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

# Count distinct top-arable crops per (farm, year)
n_arable_top = (
    df_crop_area[df_crop_area['land_type'] == 'arable_top']
        .groupby(['betr_ID', 'Jahr'])['kulturcode'].nunique()
        .rename('n_arable_top_crops')
        .reset_index()
)
farm_summary = shares_by_type.merge(n_arable_top, on=['betr_ID', 'Jahr'], how='left')
farm_summary['n_arable_top_crops'] = farm_summary['n_arable_top_crops'].fillna(0).astype(int)

# Output columns reused across options
out_cols_template = ['Flaechen_ID', 'betr_ID', 'Jahr', 'kulturcode', 'Kultur_nutzung',
                     'Crop_EN', 'land_type', 'flaeche_bewirt']

# =============================================================================
# Option 1: farm-years with literally ONE crop, and it's a top arable crop
# (strictest case — farm has no other land use at all)
# =============================================================================
n_crops_per_farm = (
    df_crop_area.groupby(['betr_ID', 'Jahr'])['kulturcode']
                .nunique()
                .rename('n_crops')
                .reset_index()
)
single_crop_farms = n_crops_per_farm[n_crops_per_farm['n_crops'] == 1][['betr_ID', 'Jahr']]

fields_op1 = (
    df_nutzung.merge(single_crop_farms, on=['betr_ID', 'Jahr'], how='inner')
              .query('kulturcode in @top_arable_codes')
              .copy()
)
fields_op1['Crop_EN'] = fields_op1['kulturcode'].map(code_name_map)
fields_op1 = fields_op1[[c for c in out_cols_template if c in fields_op1.columns]]
fields_op1 = fields_op1.sort_values(['betr_ID', 'Jahr', 'Flaechen_ID'])

print(f"Option 1 — Top-arable fields on single-crop farms")
print(f"  Qualifying farm-years: {len(single_crop_farms)}")
print(f"  Arable fields retained: {len(fields_op1)}")
print("  Top crops on single-crop farms:")
print(fields_op1['Crop_EN'].value_counts().head(20))

# =============================================================================
# Option 2: farm-years where >=50% grassland + exactly 1 top-arable crop
# =============================================================================
mask_grass_plus_one = (
    (farm_summary['grassland'] >= GRASS_MIN_SHARE) &
    (farm_summary['n_arable_top_crops'] == 1) &
    (farm_summary['arable_top'] >= ARABLE_MIN_SHARE) &
    (farm_summary['other'] <= OTHER_MAX_SHARE)
)
qualifying_farms_op2 = farm_summary.loc[mask_grass_plus_one, ['betr_ID', 'Jahr']]

# Select fields on these farms whose crop is in the TOP ARABLE set
# (grassland fields on the same farm are excluded — we only want the single arable crop)
fields_op2 = (
    df_nutzung.merge(qualifying_farms_op2, on=['betr_ID', 'Jahr'], how='inner')
              .query('kulturcode in @top_arable_codes')
              .copy()
)
fields_op2['Crop_EN'] = fields_op2['kulturcode'].map(code_name_map)
fields_op2 = fields_op2[[c for c in out_cols_template if c in fields_op2.columns]]
fields_op2 = fields_op2.sort_values(['betr_ID', 'Jahr', 'Flaechen_ID'])

print(f"\nOption 2 — Top-arable fields on farms with >={int(GRASS_MIN_SHARE*100)}% grassland + 1 top-arable crop")
print(f"  Qualifying farm-years: {len(qualifying_farms_op2)}")
print(f"  Arable fields retained: {len(fields_op2)}")
print("  Top arable crops on these farms:")
print(fields_op2['Crop_EN'].value_counts().head(20))

# =============================================================================
# Option 3: farm-years where one TOP ARABLE crop covers >= DOMINANT_SHARE of the farm
# =============================================================================
dominant_crop = (
    df_crop_area[df_crop_area['kulturcode'].isin(top_arable_codes) &
                 (df_crop_area['share'] >= DOMINANT_SHARE)]
        [['betr_ID', 'Jahr', 'kulturcode']]
        .rename(columns={'kulturcode': 'dominant_kulturcode'})
)

fields_op3 = (
    df_nutzung.merge(dominant_crop, on=['betr_ID', 'Jahr'], how='inner')
              .query('kulturcode == dominant_kulturcode')
              .copy()
)
fields_op3['Crop_EN'] = fields_op3['kulturcode'].map(code_name_map)
fields_op3 = fields_op3[[c for c in out_cols_template if c in fields_op3.columns]]
fields_op3 = fields_op3.sort_values(['betr_ID', 'Jahr', 'Flaechen_ID'])

print(f"\nOption 3 — Top-arable fields on farms dominated (>= {int(DOMINANT_SHARE*100)}%) by a single top arable crop")
print(f"  Qualifying farm-years: {len(dominant_crop)}")
print(f"  Arable fields retained: {len(fields_op3)}")
print("  Top dominant crops:")
print(fields_op3['Crop_EN'].value_counts().head(20))

# =============================================================================
# Option 3 diagnostic: what are the OTHER crops on dominant-crop farms?
# For each dominant-crop family, summarise the companion (non-dominant) crops.
# =============================================================================
companion = (
    df_crop_area.merge(dominant_crop, on=['betr_ID', 'Jahr'], how='inner')
                .query('kulturcode != dominant_kulturcode')   # only the non-dominant crops
                .copy()
)
companion['dominant_Crop_EN'] = companion['dominant_kulturcode'].map(code_name_map)
companion['companion_type'] = np.select(
    [companion['kulturcode'].isin(grassland_codes),
     companion['kulturcode'].isin(top_arable_codes)],
    ['grassland', 'other_top_arable'],
    default='other'
)

print(f"\nOption 3 diagnostic — companion-crop breakdown per dominant crop:")
for dom_code, dom_name in code_name_map.items():
    sub = companion[companion['dominant_kulturcode'] == dom_code]
    n_farms = dominant_crop[dominant_crop['dominant_kulturcode'] == dom_code].shape[0]
    if n_farms == 0:
        continue

    # Mean dominant share on these qualifying farms (the rest is "other stuff")
    farms_this_crop = dominant_crop[dominant_crop['dominant_kulturcode'] == dom_code]
    avg_dom_share = (
        df_crop_area.merge(farms_this_crop, on=['betr_ID', 'Jahr'], how='inner')
                    .loc[lambda d: d['kulturcode'] == dom_code, 'share']
                    .mean()
    )

    # Companion area, summed across all qualifying farm-years of this dominant crop
    type_breakdown = (
        sub.groupby('companion_type')['flaeche_bewirt'].sum()
           .reindex(['grassland', 'other_top_arable', 'other'], fill_value=0.0)
    )
    type_total = type_breakdown.sum()

    print(f"\n  Dominant crop: {dom_name} (n={n_farms} farm-years, mean dominant share={avg_dom_share:.2%})")
    if type_total == 0:
        print("    (no companion crops — these farms are pure single-crop)")
        continue
    for t, area in type_breakdown.items():
        print(f"    {t:18s}: {area/type_total:.1%} of non-dominant area")
    top_comp = (
        sub.groupby('Kultur_nutzung')['flaeche_bewirt']
           .sum().sort_values(ascending=False).head(5)
    )
    print("    Top 5 specific companion crops by area:")
    for crop, area in top_comp.items():
        print(f"      {crop:30s} {area/type_total:.1%}")

# Long-format table of (dominant_crop, companion_crop, share) for offline inspection
companion_summary = (
    companion.groupby(['dominant_Crop_EN', 'Kultur_nutzung', 'companion_type'], as_index=False)
             ['flaeche_bewirt'].sum()
             .rename(columns={'flaeche_bewirt': 'total_companion_area_ha'})
             .sort_values(['dominant_Crop_EN', 'total_companion_area_ha'], ascending=[True, False])
)
companion_summary['share_of_companion_area'] = (
    companion_summary['total_companion_area_ha'] /
    companion_summary.groupby('dominant_Crop_EN')['total_companion_area_ha'].transform('sum')
)

# =============================================================================
# Combined output: union of all three selection rules
# A field can satisfy multiple rules; we keep all reasons in a comma-separated string.
# =============================================================================
fields_op1['selection_reason'] = 'single_crop_farm'
fields_op2['selection_reason'] = 'grassland_plus_one'
fields_op3['selection_reason'] = 'dominant_crop'

fields_all = pd.concat([fields_op1, fields_op2, fields_op3], ignore_index=True)

# Collapse duplicates: same field can appear from multiple rules → join reasons
key_cols   = ['Flaechen_ID', 'betr_ID', 'Jahr']
other_cols = [c for c in fields_all.columns if c not in key_cols + ['selection_reason']]
fields_all = (
    fields_all.groupby(key_cols, as_index=False)
              .agg({**{c: 'first' for c in other_cols},
                    'selection_reason': lambda x: ','.join(sorted(set(x)))})
              .sort_values(['betr_ID', 'Jahr', 'Flaechen_ID'])
)

print(f"\nCombined — unique top-arable fields satisfying ANY rule: {len(fields_all)}")
print("Selection reasons:")
print(fields_all['selection_reason'].value_counts())

# =============================================================================
# Save field-level shortlists
# =============================================================================
"""
fields_op1.to_csv(os.path.join(OUT_DIR, 'fields_single_crop_farm.csv'), index=False)
fields_op2.to_csv(os.path.join(OUT_DIR, 'fields_grass_plus_one_crop.csv'), index=False)
fields_op3.to_csv(os.path.join(OUT_DIR, 'fields_single_dominant_crop.csv'), index=False)
fields_all.to_csv(os.path.join(OUT_DIR, 'fields_clean_crop_all.csv'), index=False)
companion_summary.to_csv(os.path.join(OUT_DIR, 'companion_crops_per_dominant.csv'), index=False)
print(f"\nSaved CSVs to {OUT_DIR}/")
"""