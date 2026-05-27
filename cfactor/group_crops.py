import os
import numpy as np
import pandas as pd
from collections import defaultdict


# =====================================
# Identify main crops (by area) to consider

LNF_LABELS_PATH = os.path.expanduser('~/mnt/eo-nas1/data/landuse/documentation/LNF_code_classification_20260217.xlsx')
LNF_DIR = os.path.expanduser('~/mnt/eo-nas1/data/landuse/raw')
EXCLUDE_CODES = [553, 554, 555, 556, 559, 572, 594, 595, 598, 618, 625]
GRASSLAND_CODES = [601]
yrs = [2019, 2020, 2021, 2022, 2023, 2024]
area_yrs = [2022, 2023, 2024]
top_crops = None

# Resolve LNF codes for top arable crops
df_labels = pd.read_excel(LNF_LABELS_PATH, sheet_name='label_sheet')
if top_crops is None:
    df_arable = df_labels[df_labels['Crop_Label_lv3'].isin(['Arable Land'])]
    # Years used to pick top crops by area. Defaults to `yrs` for backward
    # compat, but the LNF spreadsheet's per-year area columns follow the
    # pattern `<yr>_Area_m<(yr+1) % 100>` only from 2021 onward (e.g.
    # '2021_Area_m22', '2022_Area_m23', ...); 2019/2020 use a different
    # naming ('2019_Area_m2', '2020_Area_m2') that the templated lookup
    # can't address. Pass a separate `top_crops_area_years` (e.g.
    # [2022, 2023, 2024]) when sampling a wider range of years.
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
arable_codes      = [c for c in arable_codes if c not in EXCLUDE_CODES]
# Grassland overrides from config take precedence over the lv3 label —
# mirrors GRASSLAND_OVERRIDE in check_agis.py. E.g. code 601 is labeled
# 'Arable Land' in lv3 but is grassland in practice; if it's listed in
# config['grassland_codes'] it must be stripped out of arable_codes here.
grass_override = [c for c in GRASSLAND_CODES if c not in EXCLUDE_CODES]
arable_codes   = [c for c in arable_codes if c not in grass_override]
grass_codes    = grass_override
lnf_codes = arable_codes + grass_codes
print(f"LNF codes (crops: {len(arable_codes)}, grass: {len(grass_codes)}): {lnf_codes}")


# =====================================
# Check these codes in the C factor table: add analgoy crops, flag fully estimated crops

C_FACTOR_TABLE_PATH = os.path.expanduser('~/mnt/Data-Labo-RE/27_Natural_Resources-RE/321.4_WAUM_protected/Daten/Erosionsrisiko/C_Faktoren.csv')
LNF_CLASFFICIATION_PATH = os.path.expanduser('~/mnt/eo-nas1/data/landuse/documentation/LNF_code_classification_20260217.xlsx')
LNF_MAPPING_CSV = os.path.expanduser('~/mnt/Data-Labo-RE/27_Natural_Resources-RE/321.4_WAUM_protected/Daten/Core_Snapshot/Agrarbericht_2025/tbl_kulturmapping.csv')
C_FACTOR_PROVENANCE_PATH = 'c_factor_provenance.xlsx'
value_cols = ['Tal_Pflug', 'Tal_Mulch', 'Tal_Direkt', 'Berg_Pflug', 'Berg_Mulch', 'Berg_Direkt']
c_ref_col = 'Total'

# --- C-factor table: crop name -> values ---
df_c = pd.read_csv(C_FACTOR_TABLE_PATH, sep=';', encoding='latin-1')
df_c = df_c.rename(columns={'Kultur Kategorien 2020': 'Kultur_nutzung'})
df_c = df_c.dropna(subset=['Kultur_nutzung']).copy()
# Fix a naming
df_c.loc[
    df_c['Kultur_nutzung'] == 'Einjährige Freilandgemüse (ohne Konservengemüse)',
    'Kultur_nutzung'
] = 'Einjährige Freilandgemüse, ohne Konservengemüse'

for col in value_cols + [c_ref_col]:
    df_c[col] = pd.to_numeric(df_c[col], errors='coerce')
df_mapping = (
        pd.read_csv(LNF_MAPPING_CSV, encoding="latin1", sep=";")
          [['kulturcode', 'Kultur_nutzung']]
          .drop_duplicates()
    )   
df_c = df_c.merge(df_mapping, on='Kultur_nutzung', how='left')
df_c = df_c.rename(columns={'Kultur_nutzung': 'crop_name', 'kulturcode':'lnf_code'})

# --- LNF bridge: crop name -> lnf_code ---
df_lnf = pd.read_excel(LNF_CLASFFICIATION_PATH, sheet_name='label_sheet')
df_lnf = df_lnf[['LNF_code', 'Crop_DE']].rename(
    columns={'LNF_code': 'lnf_code', 'Crop_DE': 'crop_name'}).dropna()
df = df_lnf.merge(df_c[['lnf_code', c_ref_col] + value_cols],
                  on='lnf_code', how='inner').dropna(subset=value_cols)

# --- C factor provenance ---
df_prov = pd.read_excel(os.path.expanduser(C_FACTOR_PROVENANCE_PATH), sheet_name='provenance', header=4)

# Drop rows + lnf_codes where all where estimated
estim_codes = df_prov[df_prov['n_estimate']==6].BFS_code.tolist()
df_prov = df_prov[~df_prov['BFS_code'].isin(estim_codes)]
df = df[~df['lnf_code'].isin(estim_codes)]
lnf_codes = [c for c in lnf_codes if c not in estim_codes]

# --- step 1: build mapping groups from identical value_cols ---
grouped = df.groupby(value_cols)['lnf_code'].apply(list)

# --- step 2: build undirected graph ---
graph = defaultdict(set)

for codes in grouped:
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            a, b = codes[i], codes[j]
            graph[a].add(b)
            graph[b].add(a)

# --- step 3: find connected components ---
visited = set()
groups = []

all_nodes = set(lnf_codes) | set(graph.keys())

for node in all_nodes:
    if node not in visited:
        stack = [node]
        component = set()

        while stack:
            cur = stack.pop()

            if cur not in visited:
                visited.add(cur)
                component.add(cur)
                stack.extend(graph[cur] - visited)

        groups.append(component)

# --- output ---
groups = [g for g in groups if g & set(lnf_codes)]
print(groups)


# map lnf_code -> crop_name
lnf_to_crop = dict(zip(df['lnf_code'], df['crop_name']))

lnf_set = set(lnf_codes)

for g in groups:
    g = set(g)

    # keep only groups that contain at least one lnf_code (safety)
    main_candidates = g & lnf_set
    if not main_candidates:
        continue

    # choose the main lnf_code (if multiple, pick sorted first)
    main_code = sorted(main_candidates)[0]
    main_crop = lnf_to_crop.get(main_code, "UNKNOWN")

    # analogy codes = rest of group
    analogy_codes = g - {main_code}

    print(f"\nMAIN: {main_code} -> {main_crop}")

    print("ANALOGIES:")
    for code in sorted(analogy_codes):
        crop = lnf_to_crop.get(code, "UNKNOWN")
        print(f"  {code} -> {crop}")
