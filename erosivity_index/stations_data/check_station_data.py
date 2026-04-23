"""
Check the compute rainfall erosivity index values computed at the stations
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import zarr
import rasterio
from rasterio.features import geometry_mask
from rasterio.plot import show
import matplotlib.pyplot as plt
from pyproj import Transformer
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error


def sample_bafu_point(src, station_coords, nodata):
    """Sample the single pixel at each station coordinate."""
    raw = [
        next(src.sample([coord]))[0] if src.bounds.left <= coord[0] <= src.bounds.right
        and src.bounds.bottom <= coord[1] <= src.bounds.top else np.nan
        for coord in station_coords
    ]
    return [np.nan if v == nodata else float(v) for v in raw]


def sample_bafu_neighborhood(src, station_coords, nodata, radius_pixels=2):
    """
    Sample a square neighborhood around each station and return the mean of
    valid pixels. radius_pixels=2 gives a 5×5 window (±2 pixels in each direction).
    """
    data = src.read(1)
    transform = src.transform
    results = []
    for coord in station_coords:
        if not (src.bounds.left <= coord[0] <= src.bounds.right
                and src.bounds.bottom <= coord[1] <= src.bounds.top):
            results.append(np.nan)
            continue
        col, row = ~transform * coord  # map coords → fractional pixel indices
        col, row = int(col), int(row)
        r0 = max(row - radius_pixels, 0)
        r1 = min(row + radius_pixels + 1, data.shape[0])
        c0 = max(col - radius_pixels, 0)
        c1 = min(col + radius_pixels + 1, data.shape[1])
        window = data[r0:r1, c0:c1].astype(float)
        if nodata is not None:
            window[window == nodata] = np.nan
        valid = window[~np.isnan(window)]
        results.append(float(np.mean(valid)) if len(valid) > 0 else np.nan)
    return results


def build_comparison(station_monthly_data, stations_gdf, sample_fn):
    """
    Run the month-by-month BAFU extraction using the provided sample_fn and
    return a dict of {month_name: DataFrame(station_abbr, EI_monthly, bafu_value)}.
    """
    results = {}
    for month_idx, month_name in enumerate(german_months, start=1):
        print(f"Processing BAFU map for month {month_name}")
        bafu_map_path = os.path.join(bafu_map_dir, f"niederschlagserosivitaet-{month_name}_2056.tif")
        if not os.path.exists(bafu_map_path):
            print(f"BAFU map for month {month_name} not found, skipping.")
            continue

        with rasterio.open(bafu_map_path) as src:
            stations_proj = stations_gdf.to_crs(src.crs)
            station_coords = [(geom.x, geom.y) for geom in stations_proj.geometry]
            bafu_vals = sample_fn(src, station_coords, src.nodata)

        bafu_by_station = dict(zip(stations_proj['station_abbr'], bafu_vals))

        joined_data = []
        for station_abbr, monthly_df in station_monthly_data.items():
            station_monthly = monthly_df[monthly_df['month'] == month_idx]
            if station_monthly.empty:
                continue
            joined_data.append({
                'station_abbr': station_abbr,
                'EI_monthly': station_monthly['EI_monthly'].values[0],
                'bafu_value': bafu_by_station.get(station_abbr, np.nan)
            })

        if joined_data:
            results[month_name] = pd.DataFrame(joined_data)
    return results


def plot_comparison(comparison_results, title, save_path):
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    r2_all = []

    for month_idx, (month_name, ax) in enumerate(zip(german_months, axes), start=1):
        if month_name in comparison_results:
            data = comparison_results[month_name].dropna(subset=['bafu_value'])
            x = data['EI_monthly'].values
            y = data['bafu_value'].values
            ax.scatter(x, y, alpha=0.7, edgecolor='k')
            lim = max(x.max(), y.max()) * 1.05
            ax.plot([0, lim], [0, lim], 'r--', label='1:1')
            if len(x) > 1:
                r, _ = pearsonr(x, y)
                rmse = np.sqrt(mean_squared_error(x, y))
                r2_all.append(r**2)
                ax.text(0.05, 0.88, f'R²={r**2:.2f}\nRMSE={rmse:.1f}',
                        transform=ax.transAxes, fontsize=9)
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_aspect('equal')
            ax.set_title(month_name)
            ax.set_xlabel('Station EI (measured)')
            ax.set_ylabel('BAFU EI (mapped)')
            ax.legend(fontsize=8)
        else:
            ax.set_visible(False)

    fig.suptitle(f'{title}  |  Mean R²={np.mean(r2_all):.2f}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def classify_terrain_erosivity(elev: float, slope: float) -> str:
    if elev < 600 and slope < 5:
        return "valley"
    elif elev < 600:
        return "plateau"
    elif elev < 1500:
        return "hill" if slope < 15 else "steep_hill"
    elif elev < 2200:
        return "mountain"
    else:
        return "alpine"


def get_station_terrain_info(stations_gdf: gpd.GeoDataFrame, dem_zarr_path: str) -> pd.DataFrame:
    """Return a DataFrame with station_abbr, elev, slope sampled from the DEM zarr."""
    z = zarr.open(dem_zarr_path, mode='r')
    xs = z['x'][:]
    ys = z['y'][:]
    height_arr = z['height'][:]
    slope_arr = z['slope'][:]

    # Station coords are LV95 (EPSG:2056); zarr is EPSG:32632
    transformer = Transformer.from_crs("EPSG:2056", "EPSG:32632", always_xy=True)
    sx_lv95 = stations_gdf.geometry.x.values
    sy_lv95 = stations_gdf.geometry.y.values
    sx_utm, sy_utm = transformer.transform(sx_lv95, sy_lv95)

    elevs, slopes = [], []
    for sx, sy in zip(sx_utm, sy_utm):
        col = int(np.argmin(np.abs(xs - sx)))
        row = int(np.argmin(np.abs(ys - sy)))
        elevs.append(float(height_arr[row, col]))
        slopes.append(float(slope_arr[row, col]))

    result = stations_gdf[['station_abbr']].copy().reset_index(drop=True)
    result['elev'] = elevs
    result['slope'] = slopes
    return result


def plot_regional_cumulative_curves(
    station_dir: str,
    region_stations: dict[str, list[str]],
    save_path: str = 'regional_cumulative_curves.png',
    agg: str = 'median',
) -> None:
    """Plot one aggregated cumulative EI curve per biogeographic region plus Overall."""
    region_colors = {
        'Jura':           'tab:green',
        'Alpine Midland': 'tab:blue',
        'Northern Alps':  'tab:orange',
        'Eastern Alps':   'tab:red',
        'Southern Alps':  'tab:purple',
        'Western Alps':   'tab:brown',
        'Overall':        'black',
    }
    agg_str = 'median' if agg == 'median' else 'mean'

    all_files = {f.split('_')[-1].replace('.csv', ''): f for f in os.listdir(station_dir) if f.endswith('.csv')}

    def load_cumul(abbrevs: list[str]) -> pd.DataFrame:
        frames = []
        for abbr in abbrevs:
            fname = all_files.get(abbr)
            if fname is None:
                continue
            df = pd.read_csv(os.path.join(station_dir, fname))
            if 'doy' not in df.columns or 'EI_daily_avg' not in df.columns:
                continue
            df = df[['doy', 'EI_daily_avg']].sort_values('doy').copy()
            total = df['EI_daily_avg'].sum()
            if total == 0:
                continue
            df['EI_cumul_pct'] = df['EI_daily_avg'].cumsum() / total * 100
            df['station_abbr'] = abbr
            frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    fig, ax = plt.subplots(figsize=(12, 10))

    all_abbrevs = [abbr for abbrevs in region_stations.values() for abbr in abbrevs]
    regions_to_plot = {**region_stations, 'Overall': all_abbrevs}

    for region, abbrevs in regions_to_plot.items():
        df = load_cumul(abbrevs)
        if df.empty:
            continue
        curve = df.groupby('doy')['EI_cumul_pct'].agg(agg_str)
        col = region_colors.get(region, 'tab:gray')
        lw = 2.8 if region == 'Overall' else 2.0
        ls = '--' if region == 'Overall' else '-'
        n = df['station_abbr'].nunique()
        ax.plot(curve.index, curve.values, color=col, linewidth=lw, linestyle=ls,
                label=f'{region} (n={n})')

    ax.set_xlabel("Day of Year")
    ax.set_ylabel(f"Cumulative EI — {agg_str} (%)")
    ax.set_title("Daily cumulative EI by biogeographic region")
    ax.set_xlim(1, 365)
    ax.set_ylim(0, 100)
    ax.legend(title="Region", loc='upper left')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved → {save_path}")
    plt.close()


def plot_station_cumulative_curves(
    station_dir: str,
    terrain_info: pd.DataFrame,
    n_stations: int,
    random_seed: int,
    save_path: str = 'station_cumulative_curves.png',
) -> None:
    """Sample stations and plot daily cumulative EI curves hued by terrain type."""
    all_files = [f for f in os.listdir(station_dir) if f.endswith('.csv')]
    rng = np.random.default_rng(random_seed)
    sampled = rng.choice(all_files, size=min(n_stations, len(all_files)), replace=False)

    records = []
    for fname in sampled:
        station_abbr = fname.split('_')[-1].replace('.csv', '')
        df = pd.read_csv(os.path.join(station_dir, fname))
        if 'doy' not in df.columns or 'EI_daily_avg' not in df.columns:
            continue
        df = df[['doy', 'EI_daily_avg']].sort_values('doy').copy()
        annual_total = df['EI_daily_avg'].sum()
        if annual_total == 0:
            continue
        df['EI_cumul_pct'] = df['EI_daily_avg'].cumsum() / annual_total * 100
        df['station_abbr'] = station_abbr
        records.append(df)

    if not records:
        print("No station data loaded for cumulative curve plot.")
        return

    all_df = pd.concat(records, ignore_index=True)
    all_df = all_df.merge(terrain_info, on='station_abbr', how='left')
    all_df['terrain_class'] = all_df.apply(
        lambda r: classify_terrain_erosivity(r['elev'], r['slope']), axis=1
    )

    fig, ax = plt.subplots(figsize=(12, 10))
    for terrain in terrain_order:
        sub = all_df[all_df['terrain_class'] == terrain]
        if sub.empty:
            continue
        col = terrain_colors[terrain]
        first = True
        for _, grp in sub.groupby('station_abbr'):
            ax.plot(grp['doy'], grp['EI_cumul_pct'], color=col, alpha=0.6, linewidth=1.2,
                    label=terrain if first else None)
            first = False

    ax.set_xlabel("Day of Year")
    ax.set_ylabel("Cumulative EI (%)")
    ax.set_title(f"Daily cumulative EI — {len(records)} sampled stations")
    ax.set_xlim(1, 365)
    ax.set_ylim(0, 100)
    ax.legend(title="Terrain type", loc='upper left')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved → {save_path}")
    plt.close()




station_dir = 'stations_EIdaily_percent'
station_metadata = 'metadata/ogd-smn_meta_stations.csv'
bafu_map_dir = '../results/maps/bafu'  # Directory containing BAFU maps

# German short names for months
german_months = ['jan', 'feb', 'mrz', 'apr', 'mai', 'jun', 'jul', 'aug', 'sep', 'okt', 'nov', 'dez']

# Load station metadata
stations_gdf = pd.read_csv(station_metadata, delimiter=';', encoding='latin1')[['station_abbr', 'station_coordinates_lv95_east', 'station_coordinates_lv95_north']]
stations_gdf = gpd.GeoDataFrame(
    stations_gdf,
    geometry=gpd.points_from_xy(stations_gdf['station_coordinates_lv95_east'], stations_gdf['station_coordinates_lv95_north']),
    crs="EPSG:2056"  # Assuming LV95 Swiss coordinate system
)

# Step 1: Load all station data and compute monthly EI
station_monthly_data = {}  # Dictionary to store monthly EI per station
for file in os.listdir(station_dir):
    if not file.endswith('.csv'):
        continue

    file_path = os.path.join(station_dir, file)
    print(f"Processing {file} for monthly EI")

    df = pd.read_csv(file_path)
    if 'doy' not in df.columns or 'EI_daily_avg' not in df.columns:
        print(f"Skipping {file}: missing required columns")
        continue

    # Create a synthetic date using DOY (assuming a non-leap year)
    df['date'] = pd.to_datetime(df['doy'], format='%j', errors='coerce')

    # Group by month and compute the sum of EI_daily_avg
    df['month'] = df['date'].dt.month  # Extract month
    monthly_df = df.groupby('month', as_index=False)['EI_daily_avg'].sum()
    monthly_df.rename(columns={'EI_daily_avg': 'EI_monthly'}, inplace=True)

    # Store the monthly data for this station using the station abbreviation
    station_abbr = file.split('_')[-1].split('.csv')[0]
    station_monthly_data[station_abbr] = monthly_df





# --- Plot mean monthly erosivity for all stations ---

# Combine all stations
all_monthly = []
for station, df in station_monthly_data.items():
    df_copy = df.copy()
    df_copy['station'] = station
    all_monthly.append(df_copy)
all_monthly_df = pd.concat(all_monthly, ignore_index=True)

# Compute monthly mean and std across all stations
monthly_stats = (
    all_monthly_df
    .groupby('month')['EI_monthly']
    .agg(['mean', 'std'])
    .reset_index()
)

# Add German month labels
monthly_stats['month_name'] = [
    german_months[m-1] for m in monthly_stats['month']
]

# Plot barplot
plt.figure(figsize=(10,5))
plt.bar(
    monthly_stats['month_name'],
    monthly_stats['mean'],
    yerr=monthly_stats['std'],
    capsize=4
)
plt.xlabel('Month')
plt.ylabel('MMean monthly rainfall erosivity')
plt.tight_layout()
plt.savefig('monthly_barplot.png')



# --- Run point comparison (single pixel) ---
comparison_point = build_comparison(
    station_monthly_data, stations_gdf,
    sample_fn=lambda src, coords, nd: sample_bafu_point(src, coords, nd)
)
plot_comparison(comparison_point,
                title='Monthly EI: Measured vs BAFU mapped (point)',
                save_path='stations_EI_comparison_point.png')

# --- Run neighborhood comparison (mean of surrounding pixels, ±2 pixel radius) ---
NEIGHBORHOOD_RADIUS = 2  # pixels; adjust as needed
comparison_neighborhood = build_comparison(
    station_monthly_data, stations_gdf,
    sample_fn=lambda src, coords, nd: sample_bafu_neighborhood(src, coords, nd, radius_pixels=NEIGHBORHOOD_RADIUS)
)
plot_comparison(comparison_neighborhood,
                title=f'Monthly EI: Measured vs BAFU mapped (neighborhood ±{NEIGHBORHOOD_RADIUS}px)',
                save_path=f'stations_EI_comparison_neighborhood_{NEIGHBORHOOD_RADIUS}px.png')


# ---------------------------------------------------------------------------
# Daily cumulative EI curves for sampled stations, hued by terrain type
# ---------------------------------------------------------------------------

DEM_ZARR_PATH = '../covariates/dem_100m.zarr'
N_SAMPLE_STATIONS = 20
RANDOM_SEED = 42

terrain_colors = {
    'valley':     'tab:green',
    'plateau':    'tab:olive',
    'hill':       'tab:cyan',
    'steep_hill': 'tab:purple',
    'mountain':   'tab:red',
    'alpine':     'tab:gray',
}
terrain_order = ['valley', 'plateau', 'hill', 'steep_hill', 'mountain', 'alpine']

# Sample DEM values for every station and then plot
terrain_info = get_station_terrain_info(stations_gdf, DEM_ZARR_PATH)
plot_station_cumulative_curves(
    station_dir=station_dir,
    terrain_info=terrain_info,
    n_stations=N_SAMPLE_STATIONS,
    random_seed=RANDOM_SEED,
    save_path='station_cumulative_curves.png',
)


# ---------------------------------------------------------------------------
# Daily cumulative EI curves by biogeographic region
# ---------------------------------------------------------------------------

REGION_STATIONS: dict[str, list[str]] = {
    'Jura': [
        'BRL', 'CHA', 'CHB', 'COY', 'CRM', 'DEM', 'DOL', 'FAH', 'FRE', 'HLL', 'RUE',
    ],
    'Alpine Midland': [
        'ARH', 'BEZ', 'BIE', 'BIZ', 'CGI', 'CHZ', 'EGO', 'GOE', 'GRA', 'GRE', 'GUT',
        'HAI', 'KOP', 'LEI', 'MAH', 'MAS', 'MOA', 'MOE', 'MUB', 'ORO', 'PLF', 'PUY',
        'REH', 'SHA', 'TAE', 'VEV', 'VIT', 'WAE', 'WYN',
    ],
    'Northern Alps': [
        'ABO', 'BOL', 'BRZ', 'CDM', 'EBK', 'EIN', 'GES', 'GIH', 'GLA', 'GUE', 'HOE',
        'INT', 'MLS', 'NAP', 'OBR', 'PIL', 'SAG', 'SPF',
    ],
    'Eastern Alps': [
        'AND', 'ARO', 'BEH', 'BUF', 'COV', 'DIS', 'LAT', 'SCU', 'SMM', 'SRS', 'VAB',
        'VIO', 'VLS', 'WFJ',
    ],
    'Southern Alps': [
        'CEV', 'CIM', 'COM', 'GEN', 'GRO', 'MAG', 'PIO', 'ROE', 'SBO',
    ],
    'Western Alps': [
        'BLA', 'BOU', 'EVI', 'EVO', 'MAR', 'MTE', 'MVE', 'SIM', 'VIS', 'ZER',
    ],
}

plot_regional_cumulative_curves(
    station_dir=station_dir,
    region_stations=REGION_STATIONS,
    save_path='regional_cumulative_curves.png',
    agg='median',
)