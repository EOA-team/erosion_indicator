"""
Download 10min precipitation data from MeteoSwiss stations: https://www.meteoswiss.admin.ch/services-and-publications/service/open-data.html
Includes data from automatic, precipitation and tower stations from 2000 onwards
"""


import os
import pandas as pd
import geopandas as gpd
from pystac_client import Client
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import contextily as ctx
from shapely.geometry import Point


def sanitize_column_names(df):
    """
    Removes single quotes from column names in a given Pandas DataFrame.
    Also rename "REFERENCE_TS" to "reference_timestamp".

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A new DataFrame with cleaned column names.
    """
    new_column_names = {col: col.replace("'", "") for col in df.columns}
    df_cleaned = df.rename(columns=new_column_names)
   
    if "reference_timestamp" in df_cleaned.columns:
        df_cleaned = df_cleaned.rename(columns={"reference_timestamp": "time"})

    if not 'rre150z0' in df_cleaned.columns:
        print("Error: No valid precipitation column found!")
        return None


    return df_cleaned


station_metadata_file = 'stations_data/metadata/ogd-smn_meta_stations.csv'
stations = pd.read_csv(station_metadata_file, delimiter=';', encoding='latin1')
stations_list = stations['station_abbr'].str.lower().tolist()

########
# STATION INFORMATION

geometry = [Point(xy) for xy in zip(stations['station_coordinates_wgs84_lon'],
                                    stations['station_coordinates_wgs84_lat'])]
gdf = gpd.GeoDataFrame(stations, geometry=geometry, crs="EPSG:4326")
gdf_web = gdf.to_crs(epsg=3857)

# Plot
fig, ax = plt.subplots(figsize=(10,10))
gdf_web.plot(ax=ax, color='red', markersize=50, alpha=0.7)
ctx.add_basemap(ax, source=ctx.providers.SwissFederalGeoportal.NationalMapColor)
red_patch = mpatches.Patch(color='red', label='Stations')
ax.legend(handles=[red_patch])
ax.set_axis_off()
plt.savefig('station_locations.png')

# Plot with categorisation
fig, ax = plt.subplots(figsize=(10,10))
gdf_web.plot(ax=ax, column='station_exposition_en', markersize=50, cmap='tab20', categorical=True, legend=True)
ctx.add_basemap(ax, source=ctx.providers.SwissFederalGeoportal.NationalMapColor)
ax.set_axis_off()
ax.get_legend().set_bbox_to_anchor((1, 1))  # top-right outside the map
plt.tight_layout()
plt.savefig('station_locations_types.png')


########
# DOWNLOAD DATA

output_dir = "stations_data/stations_csv"
os.makedirs(output_dir, exist_ok=True)

# STAC parameters
catalog = Client.open("https://data.geo.admin.ch/api/stac/v1")
collections = {
'automatic': 'ch.meteoschweiz.ogd-smn', #https://data.geo.admin.ch/browser/index.html#/collections/ch.meteoschweiz.ogd-smn?.language=en
'precip': 'ch.meteoschweiz.ogd-smn-precip', #https://data.geo.admin.ch/browser/index.html#/collections/ch.meteoschweiz.ogd-smn-precip?.language=en
#'tower': 'ch.meteoschweiz.ogd-smn-tower', #https://data.geo.admin.ch/browser/index.html#/collections/ch.meteoschweiz.ogd-smn-tower?.language=en
}
"""
# Download data
for station in stations_list:
    output_file =  os.path.join(output_dir, f"{station.upper()}.csv")

    # Skip if already downloaded
    if os.path.exists(output_file):
        print(f"{station.upper()} already downloaded, skipping.")
        continue

    t_assets = []

    for c in collections:
        collection_id = collections[c]

        # Search for the station item
        search = catalog.search(collections=[collection_id], ids=[station])
        items = list(search.items())
        if not items:
            print(f"No item found for station {station.upper()} in {c}")
            continue

        item = items[0]

        # Filter 10-min resolution assets
        for key, asset in item.assets.items():
            if "_t_" in key:  # 10-minute data
                t_assets.append(asset.href)

    if not t_assets:
        print(f"No 10-min assets found for station {station.upper()}")
        continue
    
    dfs = []
    for url in t_assets:  
        df = pd.read_csv(url, delimiter=';')
        dfs.append(df)
    df_station = pd.concat(dfs, ignore_index=True)
    df_station = sanitize_column_names(df_station) 

    # Check that the station actually collects variable rre150z0
    if  df_station is None:
        print('No 10min precip data for station:', station.upper())
        continue
    if not len(df_station["rre150z0"].dropna()):
        print('No 10min precip data for station:', station.upper())
        continue
    
    # Filter time < 2026    
    df_station["time"] = pd.to_datetime(
        df_station["time"],
        format="%d.%m.%Y %H:%M"
    )
    df_station = df_station[df_station["time"] < "2026-01-01"]
    df_station = df_station[["time", "rre150z0"]]
    df_station["station"] = station
    # Check that there is data
    if len(df_station.dropna()):
        save_path = os.path.join(output_dir, f"{station.upper()}.csv")
        df_station.to_csv(save_path, index=False)
        print(f'Saved for {station.upper()}:', save_path)
    else:
        print('No 10min precip data in time frame for station:', station.upper())
"""



#########
# DATA CHECKS
"""
# Some stats on the raw data:
# find min-max date for each site
# number of data points for each site

# Load all data
dfs = [pd.read_csv(os.path.join(output_dir, f)) for f in os.listdir(output_dir)]
data = pd.concat(dfs, ignore_index=True)

site_stats = data.groupby("station")["time"].agg(
    start_date="min",
    end_date="max",
    n_points="count"
).reset_index()
site_stats = site_stats.sort_values("start_date").reset_index(drop=True)

plt.figure(figsize=(8,5))
plt.hist(site_stats["n_points"], bins=40)
plt.xlabel("Number of observations per site")
plt.ylabel("Number of sites")
plt.title("Distribution of data points per site")
plt.savefig('sites_distribution.png')

plt.figure(figsize=(10,8))
for i, row in site_stats.iterrows():
    plt.plot([row["start_date"], row["end_date"]], [i, i])

plt.xlabel("Time")
plt.ylabel("Site index")
plt.title("Temporal coverage of each site")
plt.savefig('site_temporal.png')
"""