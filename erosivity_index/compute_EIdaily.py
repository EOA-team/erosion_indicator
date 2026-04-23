import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compute_EI30_fast(input_dir, output_dir, timestep_min=10, min_event_precip=12.7, dry_hours=6):
    """
    Compute the erosive rainfall event erosivity:
    - product of unit rainfall energy and max rainfall amount within a 30min interval
    - unit rainfall energy: formula based on rainfall intensity in the time interval (10min)
    - the max rainfall is taken over the whole rainfall event

    - Rolling I30 is computed per event group to avoid bleeding across event boundaries
    - DOY 366 is merged into DOY 365 in downstream functions (see compute_EIdaily_avg)

    - A rainfall event is retained only if the total precipitation is
    ≥ 12.7 mm (0.5 inches), following the classical definition of
    erosive rainfall events introduced by Wischmeier and Smith (1978)
    in the Universal Soil Loss Equation (USLE).
    """

    os.makedirs(output_dir, exist_ok=True)

    # Define when a rainfall event should be considered "split"
    small_gap = 30     # minutes
    large_gap = 180    # minutes
    dry_steps = int(dry_hours * 60 / timestep_min)
    intervals_30min = int(30 / timestep_min)

    for file in os.listdir(input_dir):

        if not file.endswith('.csv'):
            continue

        file_path = os.path.join(input_dir, file)
        output_file = os.path.join(output_dir, f"EI30_{file}")

        if os.path.exists(output_file):
            continue

        print(f"Processing {file}")
        df = pd.read_csv(file_path)

        if 'time' not in df.columns or 'rre150z0' not in df.columns:
            print(f"Skipping {file}: missing required columns")
            continue

        # --- Preprocess ---
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').drop_duplicates().set_index('time')

        # Regular timestep
        expected_index = pd.date_range(df.index.min(), df.index.max(),
                                       freq=f'{timestep_min}min')
        df = df.reindex(expected_index)
        df.index.name = 'time'

        # --- Handle gaps (fast run-length encoding) ---
        df['missing'] = df['rre150z0'].isna()

        gap_group = (df['missing'] != df['missing'].shift()).cumsum()
        gap_sizes = df.groupby(gap_group).size()
        gap_minutes = gap_group.map(gap_sizes) * timestep_min

        # Fill small gaps with 0
        df.loc[df['missing'] & (gap_minutes <= small_gap), 'rre150z0'] = 0

        # Flags: will be considered as breaks in the rainfall event
        df['large_gap'] = df['missing'] & (gap_minutes >= large_gap)
        df['medium_gap'] = df['missing'] & (gap_minutes > small_gap) & (gap_minutes < large_gap)

        df['rre150z0'] = df['rre150z0'].fillna(0).clip(lower=0)

        # --- Rainfall physics ---
        df['ir'] = df['rre150z0'] * (60 / timestep_min)  # rainfall intensity [mm/h]

        df['er'] = np.where(
            df['rre150z0'] >= 0.1,
            0.29 * (1 - 0.72 * np.exp(-0.05 * df['ir'])),
            0
        )  # unit rainfall energy [MJ/ha/mm] Brown and Foster (1987)

        df['E'] = df['er'] * df['rre150z0']

        # --- Event detection (vectorized) ---
        df['dry'] = df['rre150z0'] == 0

        dry_group = (df['dry'] != df['dry'].shift()).cumsum()
        dry_sizes = df.groupby(dry_group).size()
        dry_length = dry_group.map(dry_sizes)

        df['event_break'] = (
            (df['dry'] & (dry_length >= dry_steps)) |
            df['large_gap'] |
            df['medium_gap']
        )

        df['event_id'] = df['event_break'].cumsum()

        # --- FIX #1: Rolling I30 computed per event to avoid boundary bleed ---
        # Rolling window is reset at each event boundary so that rainfall from
        # a previous event cannot inflate I30 of the following event.
        df['rain_30min'] = df.groupby('event_id')['rre150z0'].transform(
            lambda x: x.rolling(intervals_30min, min_periods=1).sum()
        )
        df['I30_local'] = df['rain_30min'] * 2  # (60/30)

        # --- FIX #2: start_time/end_time reflect actual rainy timesteps only ---
        # Using the first/last index of the group included dry tail timesteps;
        # now we track the first and last timestep where rain > 0 per event.
        def first_rainy(x):
            rainy = x[x > 0]
            return rainy.index[0] if len(rainy) else x.index[0]

        def last_rainy(x):
            rainy = x[x > 0]
            return rainy.index[-1] if len(rainy) else x.index[-1]

        # --- Aggregate per event ---
        grouped = df.groupby('event_id')

        summary = grouped.agg(
            total_precipitation=('rre150z0', 'sum'),
            E_total=('E', 'sum'),
            I30=('I30_local', 'max'),
            start_time=('rre150z0', first_rainy),
            end_time=('rre150z0', last_rainy)
        )

        # Remove non-rain events
        summary = summary[summary['total_precipitation'] >= min_event_precip]

        # Compute EI30
        summary['EI30'] = summary['E_total'] * summary['I30']  # Wischmeier and Smith (1978)

        # Clean output
        results_df = summary.reset_index()[[
            'event_id', 'start_time', 'end_time',
            'total_precipitation', 'I30', 'EI30'
        ]]

        # Save
        if len(results_df):
            results_df.to_csv(output_file, index=False)
            print(f"Saved EI30 results to {output_file}")

    return


def compute_EIdaily_avg(input_dir, output_dir):
    """
    Compute average daily EI (climatology) from per-event EI30 CSVs,
    splitting multi-day events proportionally by duration overlap.

    - DOY 366 is merged into DOY 365 to avoid underrepresentation in non-leap years.
      The assumption of uniform intensity for multi-day splitting is documented explicitly.

    Parameters:
        input_dir (str): Folder containing EI30 CSVs.
        output_dir (str): Folder where average daily EI CSVs will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if not file.endswith('.csv'):
            continue

        file_path = os.path.join(input_dir, file)
        print(f"Processing {file} for multi-day average daily EI")

        df = pd.read_csv(file_path)
        if 'start_time' not in df.columns or 'end_time' not in df.columns or 'EI30' not in df.columns:
            print(f"Skipping {file}: missing required columns")
            continue

        df['start_time'] = pd.to_datetime(df['start_time'])
        df['end_time'] = pd.to_datetime(df['end_time'])

        # Split multi-day events proportionally
        # NOTE: splitting assumes uniform rainfall intensity within each event.
        # This is a known simplification when only event-level EI30 is available.
        daily_records = []

        for _, row in df.iterrows():
            start = row['start_time']
            end = row['end_time']
            total_ei = row['EI30']

            # Edge case: zero-duration event (start == end after fix #2)
            if start == end:
                daily_records.append({'date': start.date(), 'EI_daily': total_ei})
                continue

            # Generate all days the event touches
            days = pd.date_range(start=start.date(), end=end.date(), freq='D')

            if len(days) == 1:
                # Single day event
                daily_records.append({'date': start.date(), 'EI_daily': total_ei})
            else:
                # Multi-day event: split EI30 proportionally by duration overlap
                total_seconds = (end - start).total_seconds()
                for day in days:
                    day_start = pd.Timestamp(day)
                    day_end = pd.Timestamp(day + pd.Timedelta(days=1))

                    # Calculate overlap in seconds
                    overlap_start = max(start, day_start)
                    overlap_end = min(end, day_end)
                    overlap_seconds = (overlap_end - overlap_start).total_seconds()

                    ei_fraction = total_ei * (overlap_seconds / total_seconds)
                    daily_records.append({'date': day, 'EI_daily': ei_fraction})

        # Aggregate EI_daily per day
        daily_df = pd.DataFrame(daily_records)
        daily_df['date'] = pd.to_datetime(daily_df['date'])
        daily_df = daily_df.groupby('date', as_index=False)['EI_daily'].sum()
        daily_df['date'] = pd.to_datetime(daily_df['date'])
        daily_df['doy'] = daily_df['date'].dt.dayofyear

        # FIX #3: Merge DOY 366 into DOY 365 to avoid underrepresentation.
        # DOY 366 only occurs in leap years; averaging it separately over a much
        # smaller sample of years would distort the climatology.
        daily_df['doy'] = daily_df['doy'].replace(366, 365)

        # Fraction of total EI contributed by top events
        N_TOP = 3
        MAX_TOP_FRACTION = 0.2  # sum of N_TOP events dominate >20%
        MIN_YEARS_TOP_DOY = 3
        daily_df_sorted = daily_df.sort_values('EI_daily', ascending=False)
        top_events = daily_df_sorted.head(N_TOP)
        top_fraction = top_events['EI_daily'].sum() / daily_df['EI_daily'].sum()
        if top_fraction > MAX_TOP_FRACTION:
            # Check that there are enough years covering the doys of extreme events
            daily_df['year'] = daily_df['date'].dt.year
            top_doys = top_events['doy'].unique()
            top_doy_years = daily_df[daily_df['doy'].isin(top_doys)].groupby('doy')['year'].nunique()
            under_years = top_doy_years[top_doy_years < MIN_YEARS_TOP_DOY]
            if len(under_years) > 0:
                print(f"Top DOYs underrepresented: {under_years.to_dict()} -> dropping station")
                continue

        # Average over DOY to get climatology.
        # Divide by total calendar years in the record, not just years with events.
        # Using .mean() would divide by event-years only, inflating the climatology
        # by a factor of (total_years / event_years) — typically 4–20x.
        total_years = daily_df['date'].dt.year.max() - daily_df['date'].dt.year.min() + 1
        avg_doy_df = daily_df.groupby('doy', as_index=False)['EI_daily'].sum()
        avg_doy_df['EI_daily'] = avg_doy_df['EI_daily'] / total_years
        avg_doy_df.rename(columns={'EI_daily': 'EI_daily_avg'}, inplace=True)

        # Fill missing DOY with 0 — DOY 1–365 only (366 merged above)
        all_doy = pd.DataFrame({'doy': range(1, 366)})
        avg_doy_complete = all_doy.merge(avg_doy_df, on='doy', how='left')
        avg_doy_complete['EI_daily_avg'] = avg_doy_complete['EI_daily_avg'].fillna(0)

        # Save result
        output_file = os.path.join(output_dir, f"EIdaily_avg_{file}")
        avg_doy_complete.to_csv(output_file, index=False)
        print(f"Saved multi-day average daily EI to {output_file}")

    return


def compute_EI_daily_percent(input_dir, output_dir):
    """
    For each average daily EI file, compute daily percentage and cumulative percentage of annual EI.

    Parameters:
        input_dir (str): Folder containing EIdaily_avg CSVs (doy + EI_daily_avg).
        output_dir (str): Folder to save updated CSVs with percentages.
    """
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if not file.endswith('.csv'):
            continue

        file_path = os.path.join(input_dir, file)
        print(f"Processing {file} for daily percentages")

        df = pd.read_csv(file_path)
        if 'doy' not in df.columns or 'EI_daily_avg' not in df.columns:
            print(f"Skipping {file}: missing required columns")
            continue

        # Total annual erosivity
        total_EI = df['EI_daily_avg'].sum()
        if total_EI == 0:
            df['EI_daily_percent'] = 0
            df['EI_daily_cumsum'] = 0
        else:
            # Compute daily percent contribution
            df['EI_daily_percent'] = df['EI_daily_avg'] / total_EI * 100
            # Cumulative percent
            df['EI_daily_cumsum'] = df['EI_daily_percent'].cumsum()

        # Save updated CSV
        output_file = os.path.join(output_dir, f"percent_{file}")
        df.to_csv(output_file, index=False)
        print(f"Saved daily percent file to {output_file}")

    return


def plot_EI_curves(input_dir, save_path, cat=False):
    """
    Plot EI curves. If cat, categorise the plots using the station metadata.
    """

    stations = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]
    """
    plot_only = ['ORO', 'TIT', 'PMA', 'FRU', 'ABO']
    stations = [f for f in stations if any(k in f for k in plot_only)]
    """

    if cat:
        station_metadata_file = 'stations_data/metadata/ogd-smn_meta_stations.csv'
        stations_metadata = pd.read_csv(station_metadata_file, delimiter=';', encoding='latin1')
        # Create color map
        unique_expo = stations_metadata['station_exposition_en'].dropna().unique()
        colors = plt.cm.tab20(range(len(unique_expo)))
        color_map = dict(zip(unique_expo, colors))

    fig, ax = plt.subplots(figsize=(12, 8))

    for s in stations:
        station_id = os.path.basename(s).replace(".csv", "").split("_")[-1]

        # Open file
        df = pd.read_csv(s)

        if cat:
            match = stations_metadata.loc[
                stations_metadata['station_abbr'] == station_id,
                'station_exposition_en'
            ]
            if match.empty:
                continue
            exposition = match.values[0]
            color = color_map.get(exposition, 'gray')
        else:
            color = None

        ax.plot(df['doy'], df['EI_daily_cumsum'], alpha=0.5, linewidth=2, color=color)

    # Add legend
    if cat:
        for expo, color in color_map.items():
            ax.plot([], [], color=color, label=expo)
        ax.legend(title="Exposition", fontsize=14, title_fontsize=16)

    # Format plot
    ax.set_xlabel("Day of Year", fontsize=18)
    ax.set_ylabel("Percent avg daily EI", fontsize=18)
    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_labels = [
        "01.01", "01.02", "01.03", "01.04", "01.05", "01.06",
        "01.07", "01.08", "01.09", "01.10", "01.11", "01.12"
    ]
    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_labels, rotation=45)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path)

    return


# Compute EI30 based on 10min data
input_dir = 'stations_data/stations_csv'
output_dir = 'stations_data/stations_EI30'
compute_EI30_fast(input_dir, output_dir)

# Compute EIdaily: accumulate values daily and compute long term daily mean
input_dir = 'stations_data/stations_EI30'
output_dir = 'stations_data/stations_EIdaily'
compute_EIdaily_avg(input_dir, output_dir)

# Convert to daily percentage sum
input_dir = 'stations_data/stations_EIdaily'           # folder with EIdaily_avg CSVs
output_dir = 'stations_data/stations_EIdaily_percent'  # folder to save daily % and cumulative %
compute_EI_daily_percent(input_dir, output_dir)

# Plot curves of EIdaily
input_dir = 'stations_data/stations_EIdaily_percent'
save_path = 'EIdaily_station_curves.png'
plot_EI_curves(input_dir, save_path, cat=True)

"""
# Check data that looks wrong
stations = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]
for s in stations:
    station_id = os.path.basename(s).replace(".csv", "").split("_")[-1]

    # Open file
    df = pd.read_csv(s)

    # Cumsum EI in december < 60%
    doy_check = 330
    if len(df[(df.doy==330) & (df.EI_daily_cumsum<60)]):
        print(station_id)
        break
"""
"""
# Check missing data per station
input_dir = 'stations_data/stations_csv'
for file in os.listdir(input_dir):
    if not file.endswith('.csv'):
        continue
    file_path = os.path.join(input_dir, file)
    df = pd.read_csv(file_path)

    # Check for missing data
    total_steps = len(df)
    missing_steps = df['rre150z0'].isna().sum()
    missing_percentage = (missing_steps / total_steps) * 100
    print(f"{file}: {missing_percentage:.2f}% data is missing")
"""
