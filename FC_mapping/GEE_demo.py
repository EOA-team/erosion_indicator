"""
Export aggregate FC data into single year shapefiles to be used on GEE
"""

from pathlib import Path
import geopandas as gpd
import re


# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = Path(".")
OUT_DIR = Path("yearly_shapefiles")
OUT_DIR.mkdir(exist_ok=True)


# -----------------------------
# FIND FILES
# -----------------------------
FILES = sorted(
    BASE_DIR.glob(
        "CH_fraction_weekly_*/CH_fraction_*_arable_soil_COMMUNE.gpkg"
    )
)

if not FILES:
    raise RuntimeError("No arable_soil_COMMUNE files found")


# -----------------------------
# EXTRACT YEAR + WEEK
# -----------------------------
def extract_year_week(path):

    # year from folder
    year_match = re.search(r"(\d{4})", path.parent.name)
    if not year_match:
        raise ValueError(
            f"Cannot extract year from {path.parent.name}"
        )

    year = int(year_match.group(1))

    # week from filename
    week_match = re.search(
        r"CH_fraction_(\d+)_arable_soil_COMMUNE",
        path.name
    )

    if not week_match:
        raise ValueError(
            f"Cannot extract week from {path.name}"
        )

    week = int(week_match.group(1))

    return year, week


# -----------------------------
# GROUP FILES BY YEAR
# -----------------------------
files_by_year = {}

for f in FILES:
    year, week = extract_year_week(f)

    if year not in files_by_year:
        files_by_year[year] = []

    files_by_year[year].append((week, f))


# -----------------------------
# PROCESS EACH YEAR
# -----------------------------
for year in sorted(files_by_year):

    if year < 2022:
      continue

    print(f"\nProcessing year {year}")

    yearly_layers = []

    # sort weeks
    year_files = sorted(files_by_year[year], key=lambda x: x[0])

    for i, (week, f) in enumerate(year_files):

        print(
            f"[{i+1}/{len(year_files)}] "
            f"Week {week}: {f.name}"
        )

        wk = f"{week:02d}"

        gdf = gpd.read_file(f)

        required_cols = [
            "geometry",

            "PV_norm_soil_mean",
            "PV_norm_soil_std",
            "PV_norm_soil_count",

            "NPV_norm_soil_mean",
            "NPV_norm_soil_std",
            "NPV_norm_soil_count",

            "Soil_norm_soil_mean",
            "Soil_norm_soil_std",
            "Soil_norm_soil_count",
        ]

        missing = [
            c for c in required_cols
            if c not in gdf.columns
        ]

        if missing:
            print(f"Missing columns: {missing}")
            continue

        gdf = gdf[required_cols]

        # -----------------------------
        # short unique names
        # -----------------------------
        rename_dict = {
            "PV_norm_soil_mean": f"PVM{wk}",
            "PV_norm_soil_std": f"PVS{wk}",
            "PV_norm_soil_count": f"PVC{wk}",

            "NPV_norm_soil_mean": f"NPM{wk}",
            "NPV_norm_soil_std": f"NPS{wk}",
            "NPV_norm_soil_count": f"NPC{wk}",

            "Soil_norm_soil_mean": f"SM{wk}",
            "Soil_norm_soil_std": f"SS{wk}",
            "Soil_norm_soil_count": f"SC{wk}",
        }

        gdf = gdf.rename(columns=rename_dict)

        yearly_layers.append(gdf)

    # -----------------------------
    # MERGE WEEKS
    # -----------------------------
    print(f"Merging {year}...")

    result = yearly_layers[0]

    for gdf in yearly_layers[1:]:

        result = result.merge(
            gdf.drop(columns="geometry"),
            left_index=True,
            right_index=True,
            how="left"
        )

    # -----------------------------
    # CHECK DUPLICATES
    # -----------------------------
    dupes = result.columns[result.columns.duplicated()]

    if len(dupes) > 0:
        print("Duplicate columns:")
        print(dupes.tolist())
        continue

    # -----------------------------
    # SAVE
    # -----------------------------
    outfile = OUT_DIR / f"CH_fraction_{year}.shp"

    print(f"Writing {outfile}")

    result.to_file(outfile)

    print(
        f"Done {year} "
        f"({len(result.columns)} columns)"
    )

print("\nFinished all years.")