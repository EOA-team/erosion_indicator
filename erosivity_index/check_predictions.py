import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow.dataset as ds
from functools import reduce
import operator
import rasterio
from rasterio.transform import from_origin
import glob
import os


def EI_to_cumul_gridcell(df):

    df = df.copy()

    # Ensure correct ordering
    if 'doy' in df.columns:
        df = df.sort_values(['x', 'y', 'doy'])

    avg_col = 'predicted_EI_daily_avg'
    pct_col = 'predicted_EI_daily_percent'
    csum_col = 'predicted_EI_daily_cumsum'

    # --- Case 1: have daily average ---
    if avg_col in df.columns:

        total_EI = (
            df.groupby(['x', 'y'])[avg_col]
            .transform("sum")
        )
        df.loc[:, pct_col] = np.where(
            total_EI == 0,
            0,
            df[avg_col] / total_EI * 100
        )
        df.loc[:, csum_col] = (
            df.groupby(['x', 'y'])[pct_col]
            .cumsum()
        )

    # --- Case 2: have percent already ---
    elif pct_col in df.columns:

        # Force positivity
        df.loc[:, pct_col] = np.maximum(
            df[pct_col],
            0
        )
        # Normalize per station
        group_sums = (
            df.groupby(['x', 'y'])[pct_col]
            .transform("sum")
        )

        group_sums = group_sums.replace(0, np.nan)

        df.loc[:, pct_col] = (
            df[pct_col] / group_sums
        ) * 100

        df.loc[:, pct_col] = df[pct_col].fillna(0)

        # Cumulative per station
        df.loc[:, csum_col] = (
            df.groupby(['x', 'y'])[pct_col]
            .cumsum()
        )

    else:
        print("Skipping: no avg or percent column found")

    return df


def load_bafu_monthly_means(bafu_dir):
    """
    Load all 12 BAFU monthly GeoTIFFs and return a dict:
        {month_int: mean_EI_value}
    using the same flexible file-matching logic as compare_bafu.
    """
    month_names = {
        1: 'jan', 2: 'feb', 3: 'mrz', 4: 'apr',
        5: 'mai', 6: 'jun', 7: 'jul', 8: 'aug',
        9: 'sep', 10: 'okt', 11: 'nov', 12: 'dez'
    }
    bafu_monthly_mean = {}

    for month_num, month_name in month_names.items():
        for pattern in [
            os.path.join(bafu_dir, f'*{month_name}*.*tif*'),
            os.path.join(bafu_dir, f'*{month_name.capitalize()}*.*tif*'),
            os.path.join(bafu_dir, f'*{month_num:02d}*.*tif*'),
        ]:
            matches = glob.glob(pattern, recursive=False)
            if matches:
                break

        if not matches:
            print(f"  Warning: No BAFU tif for month {month_num} ({month_name}), skipping.")
            continue

        with rasterio.open(matches[0]) as src:
            data = src.read(1).astype(float)
            nodata = src.nodata
            valid = data[data != nodata] if nodata is not None else data[~np.isnan(data)]

        if len(valid) == 0:
            print(f"  Warning: No valid pixels in {matches[0]}, skipping.")
            continue

        bafu_monthly_mean[month_num] = float(valid.mean())
        print(f"  BAFU month {month_num:02d} ({month_name}): mean={valid.mean():.2f}  [{matches[0]}]")

    return bafu_monthly_mean


def get_bafu_tif(bafu_dir, month_num):
    """
    Return the path to the BAFU GeoTIFF for a given month number (1–12),
    or None if no file is found. Uses the same flexible matching logic as
    the rest of the script (German month name → capitalised → numeric).
    """
    month_names = {
        1: 'jan', 2: 'feb', 3: 'mrz', 4: 'apr',
        5: 'mai', 6: 'jun', 7: 'jul', 8: 'aug',
        9: 'sep', 10: 'okt', 11: 'nov', 12: 'dez'
    }
    month_name = month_names[month_num]
    for pattern in [
        os.path.join(bafu_dir, f'*{month_name}*.*tif*'),
        os.path.join(bafu_dir, f'*{month_name.capitalize()}*.*tif*'),
        os.path.join(bafu_dir, f'*{month_num:02d}*.*tif*'),
    ]:
        matches = glob.glob(pattern, recursive=False)
        if matches:
            return matches[0]
    return None


def sample_bafu_at_coords(tif_path, xs, ys, src_crs="EPSG:32632"):
    """
    Look up the BAFU raster value at each (x, y) coordinate.
    Coordinates are assumed to be in src_crs (default EPSG:32632) and are
    reprojected to the BAFU raster CRS (LV95 / EPSG:2056) before sampling.
    Returns a NumPy array of the same length; NaN where coordinates fall
    outside the raster extent or on nodata pixels.
    """
    from pyproj import Transformer

    with rasterio.open(tif_path) as src:
        data = src.read(1).astype(float)
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
        transform = src.transform
        height, width = data.shape
        dst_crs = src.crs.to_epsg()

    # Reproject prediction coords → BAFU CRS
    transformer = Transformer.from_crs(src_crs, f"EPSG:{dst_crs}", always_xy=True)
    xs_reproj, ys_reproj = transformer.transform(xs, ys)

    values = []
    for x, y in zip(xs_reproj, ys_reproj):
        col, row = ~transform * (x, y)   # geographic → pixel coords
        col, row = int(col), int(row)
        if 0 <= row < height and 0 <= col < width:
            values.append(data[row, col])
        else:
            values.append(np.nan)

    return np.array(values)


def check_cumulative_pct_vs_bafu(dataset, columns, bafu_dir, out_dir='results', filter_expr=None):
    """
    Check #1 — Cumulative daily contribution (%) vs. BAFU monthly maps.

    Strategy
    --------
    For each sampled predicted pixel, the co-located BAFU value is looked
    up by converting the pixel's LV95 (x, y) coordinate to a raster index
    in the BAFU GeoTIFF.  Both sources are then normalised independently to
    % of their own annual total *per cell*, so the comparison is purely
    about seasonal shape, not absolute magnitude.

    Three plots are produced:

    1. cumul_pct_curves.png
       Median cumulative-% curve for predictions (orange) and matched BAFU
       pixels (blue), each with an IQR band.  Right panel shows the monthly
       (non-cumulative) % as side-by-side bars.

    2. cumul_pct_bias_per_month.png
       Bar chart of median(pred_cumul% − bafu_cumul%) across sampled cells
       at each month end.  Positive = predictions accumulate faster.

    3. cumul_pct_by_terrain.png
       Median pred cumulative-% split by terrain class, overlaid on the
       matched BAFU median curve.
    """

    os.makedirs(out_dir, exist_ok=True)

    month_doy = {
        1: (1, 31),   2: (32, 59),   3: (60, 90),   4: (91, 120),
        5: (121, 151), 6: (152, 181), 7: (182, 212), 8: (213, 243),
        9: (244, 273), 10: (274, 304), 11: (305, 334), 12: (335, 365)
    }
    month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

    # ------------------------------------------------------------------
    # 1. Aggregate predictions to monthly EI per sampled grid cell
    #    and simultaneously look up the co-located BAFU value
    # ------------------------------------------------------------------
    print("\n[cumul_pct] Aggregating predictions by month and sampling BAFU at pixel locations …")
    monthly_records = []

    for month, (doy_start, doy_end) in month_doy.items():

        # --- predictions ---
        filt = (ds.field("doy") >= doy_start) & (ds.field("doy") <= doy_end)
        combined_filter = filt & filter_expr if filter_expr is not None else filt
        df_m = dataset.to_table(columns=columns, filter=combined_filter).to_pandas()
        if df_m.empty:
            continue

        agg = df_m.groupby(['x', 'y']).agg(
            predicted_EI_monthly=('predicted_EI_daily_avg', 'sum'),
            elev=('elev', 'first'),
            slope=('slope', 'first'),
        ).reset_index()

        # --- co-located BAFU values ---
        tif_path = get_bafu_tif(bafu_dir, month)
        if tif_path:
            agg['bafu_EI_monthly'] = sample_bafu_at_coords(
                tif_path, agg['x'].values, agg['y'].values
            )
        else:
            print(f"  Warning: No BAFU tif for month {month}, BAFU set to NaN.")
            agg['bafu_EI_monthly'] = np.nan

        agg['month'] = month
        monthly_records.append(agg)

    if not monthly_records:
        print("  ERROR: No prediction data loaded.")
        return

    monthly_df = pd.concat(monthly_records, ignore_index=True)
    monthly_df = monthly_df.sort_values(['x', 'y', 'month'])

    # Report how many sampled cells had a valid BAFU match
    n_cells = monthly_df[monthly_df['month'] == monthly_df['month'].min()].shape[0]
    n_matched = monthly_df[
        (monthly_df['month'] == monthly_df['month'].min()) &
        monthly_df['bafu_EI_monthly'].notna()
    ].shape[0]
    print(f"  Sampled cells: {n_cells}  |  cells with valid BAFU overlap: {n_matched}")

    # ------------------------------------------------------------------
    # 2. Compute per-cell monthly % and cumulative % — independently
    #    for predictions and BAFU so both sum to 100 % per cell
    # ------------------------------------------------------------------

    # Predicted %
    pred_annual = monthly_df.groupby(['x', 'y'])['predicted_EI_monthly'].transform('sum')
    monthly_df['pred_monthly_pct'] = np.where(
        pred_annual == 0, 0,
        monthly_df['predicted_EI_monthly'] / pred_annual * 100
    )
    monthly_df['pred_cumul_pct'] = (
        monthly_df.groupby(['x', 'y'])['pred_monthly_pct'].cumsum()
    )

    # BAFU % — only for cells where all 12 months have a valid BAFU value
    bafu_annual = monthly_df.groupby(['x', 'y'])['bafu_EI_monthly'].transform('sum')
    monthly_df['bafu_monthly_pct'] = np.where(
        (bafu_annual == 0) | bafu_annual.isna(), np.nan,
        monthly_df['bafu_EI_monthly'] / bafu_annual * 100
    )
    monthly_df['bafu_cumul_pct'] = (
        monthly_df.groupby(['x', 'y'])['bafu_monthly_pct']
        .cumsum()
    )

    # Drop cells where BAFU is entirely missing (outside raster extent)
    valid_cells = (
        monthly_df.groupby(['x', 'y'])['bafu_EI_monthly']
        .apply(lambda s: s.notna().all())
    )
    valid_xy = valid_cells[valid_cells].index
    matched_df = monthly_df[
        monthly_df.set_index(['x', 'y']).index.isin(valid_xy)
    ].copy()

    n_valid = len(valid_xy)
    print(f"  Cells with complete BAFU coverage (all 12 months): {n_valid}")
    if n_valid == 0:
        print("  ERROR: No cells overlap with BAFU rasters. Check CRS/extent alignment.")
        return

    # Terrain classification
    matched_df['terrain_class'] = matched_df.apply(classify_terrain_erosivity, axis=1)

    months_avail = sorted(matched_df['month'].unique())

    # ------------------------------------------------------------------
    # 3. Plot 1 — cumulative % curves with IQR band (both sources)
    # ------------------------------------------------------------------
    print("\n[cumul_pct] Plot 1: Cumulative % curves …")

    def quantiles(df, col, months):
        q25, q50, q75 = [], [], []
        for m in months:
            v = df.loc[df['month'] == m, col].dropna()
            q25.append(v.quantile(0.25))
            q50.append(v.quantile(0.50))
            q75.append(v.quantile(0.75))
        return q25, q50, q75

    pred_q25, pred_q50, pred_q75 = quantiles(matched_df, 'pred_cumul_pct', months_avail)
    bafu_q25, bafu_q50, bafu_q75 = quantiles(matched_df, 'bafu_cumul_pct', months_avail)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: cumulative %
    ax = axes[0]
    ax.fill_between(months_avail, pred_q25, pred_q75, color='tab:orange', alpha=0.25, label='Pred IQR')
    ax.plot(months_avail, pred_q50, 'o-', color='tab:orange', linewidth=2, label='Pred median')
    ax.fill_between(months_avail, bafu_q25, bafu_q75, color='tab:blue', alpha=0.20, label='BAFU IQR')
    ax.plot(months_avail, bafu_q50, 's--', color='tab:blue', linewidth=2, label='BAFU median')
    ax.axhline(50, color='grey', linestyle=':', linewidth=1, label='50 % line')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels)
    ax.set_xlabel("Month")
    ax.set_ylabel("Cumulative EI (%)")
    ax.set_title(f"Cumulative EI contribution (%)\n{n_valid} matched pixels — Pred vs. BAFU")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(alpha=0.3)

    # Right: monthly % bars
    ax2 = axes[1]
    pred_med_monthly = [matched_df.loc[matched_df['month'] == m, 'pred_monthly_pct'].median() for m in months_avail]
    bafu_med_monthly = [matched_df.loc[matched_df['month'] == m, 'bafu_monthly_pct'].median() for m in months_avail]

    x = np.arange(len(months_avail))
    width = 0.35
    ax2.bar(x - width/2, pred_med_monthly, width, color='tab:orange', alpha=0.8, label='Pred median')
    ax2.bar(x + width/2, bafu_med_monthly, width, color='tab:blue',   alpha=0.8, label='BAFU median')
    ax2.set_xticks(x)
    ax2.set_xticklabels([month_labels[m-1] for m in months_avail])
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Monthly EI contribution (%)")
    ax2.set_title("Monthly EI contribution (%)\nPred vs. co-located BAFU")
    ax2.legend()
    ax2.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    out1 = os.path.join(out_dir, f'cumul_pct_curves_{SUFFIX}.png')
    plt.savefig(out1, dpi=150)
    plt.close()
    print(f"  Saved → {out1}")

    # ------------------------------------------------------------------
    # 4. Plot 2 — per-cell bias distribution as boxplots per month
    # ------------------------------------------------------------------
    print("\n[cumul_pct] Plot 2: Per-cell cumulative % bias …")

    matched_df['cumul_pct_bias'] = matched_df['pred_cumul_pct'] - matched_df['bafu_cumul_pct']

    bias_data = [
        matched_df.loc[matched_df['month'] == m, 'cumul_pct_bias'].dropna().values
        for m in months_avail
    ]
    median_bias = [np.nanmedian(b) for b in bias_data]

    fig, ax = plt.subplots(figsize=(12, 5))
    bp = ax.boxplot(
        bias_data, positions=months_avail, widths=0.6,
        patch_artist=True, showfliers=False,
        medianprops=dict(color='black', linewidth=2)
    )
    for patch, med in zip(bp['boxes'], median_bias):
        patch.set_facecolor('tab:orange' if med >= 0 else 'tab:blue')
        patch.set_alpha(0.6)

    ax.axhline(0, color='black', linewidth=1)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels)
    ax.set_xlabel("Month")
    ax.set_ylabel("Bias in cumulative % (pred − BAFU)")
    ax.set_title(
        f"Per-cell seasonal timing bias across {n_valid} matched pixels\n"
        "Positive = predictions accumulate faster; Negative = slower"
    )
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    out2 = os.path.join(out_dir, f'cumul_pct_bias_per_month_{SUFFIX}.png')
    plt.savefig(out2, dpi=150)
    plt.close()
    print(f"  Saved → {out2}")

    # ------------------------------------------------------------------
    # 5. Plot 3 — cumulative % by terrain class (pred vs BAFU per class)
    # ------------------------------------------------------------------
    print("\n[cumul_pct] Plot 3: Cumulative % by terrain class …")
    terrain_order = ['valley', 'plateau', 'hill', 'steep_hill', 'mountain', 'alpine']
    terrain_colors = {
        'valley':     'tab:green',
        'plateau':    'tab:olive',
        'hill':       'tab:cyan',
        'steep_hill': 'tab:purple',
        'mountain':   'tab:red',
        'alpine':     'tab:gray',
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    for terrain in terrain_order:
        sub = matched_df[matched_df['terrain_class'] == terrain]
        if sub.empty:
            continue
        col = terrain_colors[terrain]
        # Predictions — solid line with IQR
        t_pred_med = [sub.loc[sub['month'] == m, 'pred_cumul_pct'].median() for m in months_avail]
        t_pred_q25 = [sub.loc[sub['month'] == m, 'pred_cumul_pct'].quantile(0.25) for m in months_avail]
        t_pred_q75 = [sub.loc[sub['month'] == m, 'pred_cumul_pct'].quantile(0.75) for m in months_avail]
        ax.fill_between(months_avail, t_pred_q25, t_pred_q75, color=col, alpha=0.10)
        ax.plot(months_avail, t_pred_med, 'o-', color=col, linewidth=1.8, label=f'{terrain} pred')
        # BAFU — dashed line, same colour
        t_bafu_med = [sub.loc[sub['month'] == m, 'bafu_cumul_pct'].median() for m in months_avail]
        ax.plot(months_avail, t_bafu_med, '--', color=col, linewidth=1.2, alpha=0.7, label=f'{terrain} BAFU')

    ax.axhline(50, color='grey', linestyle=':', linewidth=1)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels)
    ax.set_xlabel("Month")
    ax.set_ylabel("Cumulative EI (%)")
    ax.set_title("Cumulative EI (%) by terrain class\nSolid = Pred, Dashed = co-located BAFU")
    ax.set_ylim(0, 100)
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out3 = os.path.join(out_dir, f'cumul_pct_by_terrain_{SUFFIX}.png')
    plt.savefig(out3, dpi=150)
    plt.close()
    print(f"  Saved → {out3}")

    # ------------------------------------------------------------------
    # 6. Console summary table
    # ------------------------------------------------------------------
    print(f"\n[cumul_pct] Summary table — median cumulative % across {n_valid} matched pixels:")
    print(f"  {'Month':<8} {'Pred':>8} {'BAFU':>8} {'Bias':>8}")
    print("  " + "-" * 36)
    for m in months_avail:
        p = matched_df.loc[matched_df['month'] == m, 'pred_cumul_pct'].median()
        b = matched_df.loc[matched_df['month'] == m, 'bafu_cumul_pct'].median()
        print(f"  {month_labels[m-1]:<8} {p:>8.1f} {b:>8.1f} {p-b:>+8.1f}")


def check_spatial_correlation(dataset, columns, bafu_dir, out_dir='results/spatial_corr',
                              filter_expr=None, src_crs="EPSG:32632"):
    """
    Check spatial consistency between predictions and BAFU monthly maps.

    For each month, matched pixel pairs (pred_monthly_EI, bafu_monthly_EI)
    are scatter-plotted with a 1:1 line and an OLS regression line.
    Pearson r and Spearman rho are annotated on each panel.

    Output
    ------
    spatial_corr_scatter.png  — 3×4 grid of scatter plots, one per month
    spatial_corr_summary.png  — line plot of Pearson r and Spearman rho
                                across months
    """
    from scipy import stats

    os.makedirs(out_dir, exist_ok=True)

    month_doy = {
        1: (1, 31),   2: (32, 59),   3: (60, 90),   4: (91, 120),
        5: (121, 151), 6: (152, 181), 7: (182, 212), 8: (213, 243),
        9: (244, 273), 10: (274, 304), 11: (305, 334), 12: (335, 365)
    }
    month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

    # ------------------------------------------------------------------
    # 1. Load matched pixel values for every month
    # ------------------------------------------------------------------
    print("\n[spatial_corr] Loading predictions and sampling BAFU at pixel locations …")
    monthly_records = []

    for month, (doy_start, doy_end) in month_doy.items():

        filt = (ds.field("doy") >= doy_start) & (ds.field("doy") <= doy_end)
        combined_filter = filt & filter_expr if filter_expr is not None else filt
        df_m = dataset.to_table(columns=columns, filter=combined_filter).to_pandas()
        if df_m.empty:
            continue

        agg = df_m.groupby(['x', 'y']).agg(
            pred_EI=('predicted_EI_daily_avg', 'sum'),
            elev=('elev', 'first'),
            slope=('slope', 'first'),
        ).reset_index()

        tif_path = get_bafu_tif(bafu_dir, month)
        if tif_path:
            agg['bafu_EI'] = sample_bafu_at_coords(
                tif_path, agg['x'].values, agg['y'].values, src_crs=src_crs
            )
        else:
            print(f"  Warning: No BAFU tif for month {month}.")
            agg['bafu_EI'] = np.nan

        agg['month'] = month
        monthly_records.append(agg)

    if not monthly_records:
        print("  ERROR: No data loaded.")
        return

    monthly_df = pd.concat(monthly_records, ignore_index=True)

    # Keep only rows where both values are valid and positive
    monthly_df = monthly_df.dropna(subset=['pred_EI', 'bafu_EI'])
    monthly_df = monthly_df[(monthly_df['pred_EI'] >= 0) & (monthly_df['bafu_EI'] >= 0)]

    months_avail = sorted(monthly_df['month'].unique())
    n_pixels = monthly_df[monthly_df['month'] == months_avail[0]].shape[0]
    print(f"  Matched pixels per month: ~{n_pixels}")

    # ------------------------------------------------------------------
    # 2. Plot 1 — 3×4 scatter grid, one panel per month
    # ------------------------------------------------------------------
    print("\n[spatial_corr] Plot 1: Monthly scatter panels …")

    pearson_r, spearman_rho = [], []

    fig, axes = plt.subplots(3, 4, figsize=(18, 13))
    axes_flat = axes.flatten()

    for i, month in enumerate(range(1, 13)):
        ax = axes_flat[i]

        if month not in months_avail:
            ax.set_visible(False)
            pearson_r.append(np.nan)
            spearman_rho.append(np.nan)
            continue

        sub = monthly_df[monthly_df['month'] == month]
        pred_vals = sub['pred_EI'].values
        bafu_vals = sub['bafu_EI'].values

        # Correlation stats
        r, p_r   = stats.pearsonr(pred_vals, bafu_vals)
        rho, p_s = stats.spearmanr(pred_vals, bafu_vals)
        pearson_r.append(r)
        spearman_rho.append(rho)

        # Scatter
        ax.scatter(bafu_vals, pred_vals, alpha=0.4, s=12, color='steelblue', linewidths=0)

        # 1:1 line across the full range
        all_vals = np.concatenate([pred_vals, bafu_vals])
        vmin, vmax = np.nanpercentile(all_vals, 1), np.nanpercentile(all_vals, 99)
        ax.plot([vmin, vmax], [vmin, vmax], 'k--', linewidth=1, label='1:1')

        # OLS regression line
        slope_ols, intercept, *_ = stats.linregress(bafu_vals, pred_vals)
        x_line = np.array([vmin, vmax])
        ax.plot(x_line, intercept + slope_ols * x_line, 'r-', linewidth=1.2, label=f'OLS (slope={slope_ols:.2f})')

        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        ax.set_title(month_labels[month - 1], fontsize=11)
        ax.set_xlabel("BAFU EI", fontsize=8)
        ax.set_ylabel("Pred EI", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.annotate(
            f"r={r:.2f}  ρ={rho:.2f}\nn={len(pred_vals)}",
            xy=(0.05, 0.88), xycoords='axes fraction',
            fontsize=8, color='black',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7)
        )
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Spatial correlation: Predicted EI vs. co-located BAFU EI\n"
        "(one point per sampled pixel)",
        fontsize=14
    )
    plt.tight_layout()
    out1 = os.path.join(out_dir, f'spatial_corr_scatter_{SUFFIX}.png')
    plt.savefig(out1, dpi=150)
    plt.close()
    print(f"  Saved → {out1}")

    # ------------------------------------------------------------------
    # 3. Plot 2 — r and rho across months
    # ------------------------------------------------------------------
    print("\n[spatial_corr] Plot 2: Correlation summary across months …")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, 13), pearson_r,   'o-', color='tab:blue',   linewidth=2, label="Pearson r")
    ax.plot(range(1, 13), spearman_rho,'s--', color='tab:orange', linewidth=2, label="Spearman ρ")
    ax.axhline(0, color='grey', linewidth=0.8, linestyle=':')
    ax.axhline(1, color='grey', linewidth=0.8, linestyle=':')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels)
    ax.set_xlabel("Month")
    ax.set_ylabel("Correlation coefficient")
    ax.set_ylim(-0.1, 1.05)
    ax.set_title("Spatial correlation (Pred vs. BAFU) by month")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out2 = os.path.join(out_dir, f'spatial_corr_summary_{SUFFIX}.png')
    plt.savefig(out2, dpi=150)
    plt.close()
    print(f"  Saved → {out2}")

    # ------------------------------------------------------------------
    # 4. Console summary
    # ------------------------------------------------------------------
    print(f"\n[spatial_corr] Summary (n ≈ {n_pixels} matched pixels per month):")
    print(f"  {'Month':<8} {'Pearson r':>10} {'Spearman ρ':>11}")
    print("  " + "-" * 32)
    for month, r, rho in zip(range(1, 13), pearson_r, spearman_rho):
        flag = "  ✓" if (not np.isnan(r) and r >= 0.5) else ""
        print(f"  {month_labels[month-1]:<8} {r:>10.3f} {rho:>11.3f}{flag}")


def classify_terrain_erosivity(row):

    elev = row["elev"]
    slope = row["slope"]

    # Valley floors
    if elev < 600 and slope < 5: # Volker used 600m as seperation valley/hill
        return "valley"

    # Swiss Plateau / wide valleys
    elif elev < 600:
        return "plateau"

    # Hills / pre-alpine
    elif elev < 1500:

        if slope < 15:
            return "hill"
        else:
            return "steep_hill"

    # Alpine
    elif elev < 2200:
        return "mountain"

    # High alpine
    else:
        return "alpine"


def plot_predictions(df, pred_col, save_path, color_col=None, random_seed=42):
    """
    Plot predictions for a random sample of stations on the same plot.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing actual and predicted values.
    save_path : str
        Path to save the plot.
    num_stations : int, optional
        Number of stations to randomly sample for plotting. Default is 10.
    random_seed : int, optional
        Random seed for reproducibility. Default is 42.
    """
    print("\nPlotting predictions...")

    df = df.copy()
    df['cell'] = df['x'].astype(int).astype(str) + ', ' + df['y'].astype(int).astype(str)

    show_legend = color_col is not None
    if color_col is None:
        color_col = 'cell' # show each cell in a different color

    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x='doy', y=pred_col, hue=color_col, alpha=0.4, palette='Dark2', units='cell', estimator=None, legend=show_legend)  # one line per grid cell
    
    # Customize the plot
    plt.title("Predictions for grid cells", fontsize=16)
    plt.xlabel("Day of Year (DOY)", fontsize=14)
    plt.ylabel("EI Daily Cumulative Sum", fontsize=14)
    if show_legend:
        plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(alpha=0.5)
    plt.tight_layout()

    # Save and show the plot
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close()

    return


def doy_to_month(df):
    df['month'] = pd.to_datetime(df['doy'], format='%j').dt.month
    return df


def aggregate_monthly(df):
    monthly_df = df.groupby(['x', 'y', 'month']).agg(
        predicted_EI_monthly=('predicted_EI_daily_avg', 'sum'),
        elev=('elev', 'first'),
        slope=('slope', 'first'),
        aspect=('aspect', 'first'),
        prec_daily_avg=('prec_daily_avg', 'mean')
    ).reset_index()
    return monthly_df


def main():

    # Open parquet dataset
    dataset = ds.dataset(SAVE_PATH, format="parquet")

    # Sample 100 random grid cells
    table = dataset.to_table(columns=["x", "y"])
    df_xy = table.to_pandas()
    unique_coords = df_xy.drop_duplicates()
    sample_coords = unique_coords.sample(100, random_state=42)

    # Build filter for sampled coordinates
    filters = [
        (ds.field("x") == row.x) & (ds.field("y") == row.y)
        for _, row in sample_coords.iterrows()
    ]
    filter_expr = reduce(operator.or_, filters)

    # Read filtered data
    columns = ['x', 'y', 'doy', 'elev', 'slope', 'aspect', 'prec_daily_avg', 'prec_monthly_avg', 'predicted_EI_daily_avg']
    sample_table = dataset.to_table(columns=columns, filter=filter_expr)
    sample_df = sample_table.to_pandas()

    # Convert to cumsum
    sample_df = EI_to_cumul_gridcell(sample_df)

    if plot_curves:
        # Plot all sampled grid cells
        plot_predictions(sample_df, 'predicted_EI_daily_cumsum', f'results/inference_EIdailyavg_{SUFFIX}.png')

        # Categorise terrain
        sample_df["terrain_class"] = sample_df.apply(classify_terrain_erosivity, axis=1)
        plot_predictions(sample_df, 'predicted_EI_daily_cumsum', f'results/inference_EIdailyavg_categorised_{SUFFIX}.png', color_col="terrain_class")

        # Plot only agriculturally relevant categories
        to_plot = ['valley', 'plateau', 'hill', 'steep_hill']
        plot_predictions(sample_df[sample_df["terrain_class"].isin(to_plot)], 'predicted_EI_daily_cumsum', f'results/inference_EIdailyavg_categorised_{SUFFIX}.png', color_col="terrain_class")

    if check_corr:

        os.makedirs('results/grid_sample', exist_ok=True)
        
        # Correlation with daily EI values
        cols_of_interest = ['predicted_EI_daily_cumsum', 'predicted_EI_daily_avg', 'elev', 'slope', 'aspect', 'prec_daily_avg', 'prec_monthly_avg']
        corr_matrix = sample_df[cols_of_interest].corr()
      
        plt.figure(figsize=(6,5))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
        plt.title("Correlation Heatmap: Predicted EI vs Covariates")
        plt.tight_layout()
        plt.savefig(f'results/grid_sample/heatmap_daily_{SUFFIX}.png')


        # Aggregate to monthly level for meaningful correlations
        sample_df['monthnbr'] = pd.to_datetime(sample_df['doy'], format='%j').dt.month
        monthly_df = sample_df.groupby(['x', 'y', 'monthnbr']).agg(
            predicted_EI_monthly=('predicted_EI_daily_avg', 'sum'),
            elev=('elev', 'first'),
            slope=('slope', 'first'),
            aspect=('aspect', 'first'),
            prec_daily_avg=('prec_daily_avg', 'mean'),
            prec_monthly_avg=('prec_monthly_avg', 'first'),
        ).reset_index()

        # Correlation with monthly EI values (sum of daily)
        cols_of_interest = ['predicted_EI_monthly', 'elev', 'slope', 'aspect', 'prec_daily_avg', 'prec_monthly_avg']
        corr_matrix = monthly_df[cols_of_interest].corr()

        plt.figure(figsize=(6, 5))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
        plt.title("Monthly Correlation: Predicted EI vs Covariates")
        plt.tight_layout()
        plt.savefig(f'results/grid_sample/heatmap_monthly_{SUFFIX}.png')


        # Scatter plots between monthly EI and covariates
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        sns.scatterplot(data=monthly_df, x='prec_monthly_avg', y='predicted_EI_monthly', alpha=0.3, ax=axes[0])
        axes[0].set_title("Monthly EI vs Precipitation")
        sns.scatterplot(data=monthly_df, x='elev', y='predicted_EI_monthly', alpha=0.3, ax=axes[1])
        axes[1].set_title("Monthly EI vs Elevation")
        sns.scatterplot(data=monthly_df, x='slope', y='predicted_EI_monthly', alpha=0.3, ax=axes[2])
        axes[2].set_title("Monthly EI vs Slope")
        plt.tight_layout()
        plt.savefig(f'results/grid_sample/scatter_monthly_{SUFFIX}.png')
    
    if check_distr:
        fig, axs = plt.subplots(figsize=(8,8))
        sample_df['predicted_EI_daily_avg'].hist(bins=30)
        plt.xlabel('Predicted EI')
        plt.ylabel('Count')
        plt.title('Distribution of predicted EI (sample)')
        plt.tight_layout()
        plt.savefig(f'results/grid_sample/hist_daily_{SUFFIX}.png')

        fig, axs = plt.subplots(figsize=(8,8))
        sample_df['month'] = pd.to_datetime(sample_df['doy'], format='%j').dt.month
        sample_df.boxplot(column='predicted_EI_daily_avg', by='month', figsize=(12,6))
        plt.ylabel('Predicted EI')
        plt.title('Monthly distribution of predicted EI (sample)')
        plt.tight_layout()
        plt.savefig(f'results/grid_sample/boxplot_monthly_{SUFFIX}.png')

    if check_spatial:

        month_doy = {
            1: (1, 31),
            2: (32, 59),
            3: (60, 90),
            4: (91, 120),
            5: (121, 151),
            6: (152, 181),
            7: (182, 212),
            8: (213, 243),
            9: (244, 273),
            10: (274, 304),
            11: (305, 334),
            12: (335, 365)
        }

        for month, (doy_start, doy_end) in month_doy.items():

            # Filter dataset for current month
            filter_expr = (ds.field("doy") >= doy_start) & (ds.field("doy") <= doy_end)
            table = dataset.to_table(columns=columns, filter=filter_expr)
            df = table.to_pandas()
            
            if df.empty:
                continue
            
            # Aggregate to monthly EI per grid cell
            monthly_df = df.groupby(['x', 'y']).agg(
                predicted_EI_monthly=('predicted_EI_daily_avg', 'mean'),
                elev=('elev', 'first'),
                slope=('slope', 'first'),
                aspect=('aspect', 'first'),
                prec_daily_avg=('prec_daily_avg', 'mean')
            ).reset_index()

            # Map-style plot
          
            plt.figure(figsize=(14,8))
            scatter = plt.scatter(
                monthly_df['x'], monthly_df['y'],
                c=monthly_df['predicted_EI_monthly'],
                cmap='inferno',
                s=5,
                alpha=0.7
            )
            plt.colorbar(scatter, label='Predicted monthly EI (mean)')
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title(f"Map of Monthly EI (Month {month})")
            plt.tight_layout()
            plt.savefig(f"results/maps/map_monthly_avg_EI_month{month}_{SUFFIX}.png")
            print("Saved map for month", month)
         
            # Create a TIF file
            x_vals = np.sort(monthly_df['x'].unique())
            y_vals = np.sort(monthly_df['y'].unique())
            grid = np.full((len(y_vals), len(x_vals)), np.nan)

            # Fill raster
            x_index = {x: i for i, x in enumerate(x_vals)}
            y_index = {y: i for i, y in enumerate(y_vals)}

            for _, row in monthly_df.iterrows():
                xi = x_index[row['x']]
                yi = y_index[row['y']]
                grid[yi, xi] = row['predicted_EI_monthly']

            # Flip vertically (important for geospatial orientation)
            grid = np.flipud(grid)

            # Resolution
            x_res = x_vals[1] - x_vals[0]
            y_res = y_vals[1] - y_vals[0]

            # Define transform
            transform = from_origin(
                x_vals.min(),
                y_vals.max(),
                x_res,
                y_res
            )

            # Save GeoTIFF
            output_path = f"results/maps/monthly_EI_month{month}_{SUFFIX}.tif"

            with rasterio.open(
                output_path,
                "w",
                driver="GTiff",
                height=grid.shape[0],
                width=grid.shape[1],
                count=1,
                dtype=grid.dtype,
                crs="EPSG:32632",  # change CRS if needed
                transform=transform,
            ) as dst:
                dst.write(grid, 1)

            print("Saved GeoTIFF for month", month)

    if compare_bafu:

        # Compare monthly mean to BAFU product
        # https://map.geo.admin.ch/#/map?lang=en&center=2610168.57,1202255.3&z=0.821&topic=bafu&layers=ch.bafu.erosion-gruenland_bodenabtrag_mai,f;ch.bafu.niederschlagserosivitaet-mai;ch.bafu.geochemischer-bodenatlas_schweiz_arsen,f;ch.bafu.niederschlagserosivitaet-jan,f&bgLayer=ch.swisstopo.pixelkarte-grau&catalogNodes=bafu,768,781,1361,767,784,798,804,806,826,843,849,851,1505,15157,2801,2828,2833

        month_doy = {
            1: (1, 31),
            2: (32, 59),
            3: (60, 90),
            4: (91, 120),
            5: (121, 151),
            6: (152, 181),
            7: (182, 212),
            8: (213, 243),
            9: (244, 273),
            10: (274, 304),
            11: (305, 334),
            12: (335, 365)
        }

        bafu_values = {
            1: (0, 71),
            2: (0, 247),
            3: (0, 179),
            4: (0,1014),
            5: (0, 1718),
            6: (4, 1262),
            7: (13, 1481),
            8: (8, 1995),
            9: (7, 6108),
            10: (6, 977),
            11: (5, 357),
            12: (1, 234)
        }

        # Read min/max BAFU values from GeoTIFF files
        bafu_dir = 'results/maps/bafu'
        bafu_values = {}
        month_names = {
            1: 'jan', 2: 'feb', 3: 'mrz', 4: 'apr',
            5: 'mai', 6: 'jun', 7: 'jul', 8: 'aug',
            9: 'sep', 10: 'okt', 11: 'nov', 12: 'dez'
        }

        for month_num, month_name in month_names.items():
            # Find the tif file for this month (flexible pattern matching)
            pattern = os.path.join(bafu_dir, f'*{month_name}*.*tif*')
            matches = glob.glob(pattern, recursive=False)
            if not matches:
                # Try with capitalized month name
                pattern = os.path.join(bafu_dir, f'*{month_name.capitalize()}*.*tif*')
                matches = glob.glob(pattern, recursive=False)
            if not matches:
                # Try numeric pattern
                pattern = os.path.join(bafu_dir, f'*{month_num:02d}*.*tif*')
                matches = glob.glob(pattern, recursive=False)
            if matches:
                tif_path = matches[0]
                with rasterio.open(tif_path) as src:
                    data = src.read(1)
                    nodata = src.nodata
                    if nodata is not None:
                        valid = data[data != nodata]
                    else:
                        valid = data[~np.isnan(data)]
                    if len(valid) > 0:
                        bafu_values[month_num] = (
                            float(valid.min()),
                            float(valid.max()),
                            float(valid.mean()),
                            float(np.median(valid))
                        )
                        print(f"BAFU month {month_num} ({month_name}): min={valid.min():.1f}, max={valid.max():.1f}, mean={valid.mean():.1f}, median={np.median(valid):.1f} from {tif_path}")
                    else:
                        print(f"Warning: No valid data in {tif_path}")
            else:
                print(f"Warning: No BAFU tif found for month {month_num} ({month_name}) in {bafu_dir}")
        print(bafu_values)

        pred_values = {}
        for month, (doy_start, doy_end) in month_doy.items():

            # Filter dataset for current month
            filter_expr = (ds.field("doy") >= doy_start) & (ds.field("doy") <= doy_end)
            table = dataset.to_table(columns=columns, filter=filter_expr)
            df = table.to_pandas()
            
            if df.empty:
                continue
            
            # Aggregate to monthly EI per grid cell
            monthly_df = df.groupby(['x', 'y']).agg(
                predicted_EI_monthly=('predicted_EI_daily_avg', 'sum'),
                elev=('elev', 'first'),
                slope=('slope', 'first'),
                aspect=('aspect', 'first'),
                prec_daily_avg=('prec_daily_avg', 'mean')
            ).reset_index()
            ei = monthly_df['predicted_EI_monthly']
            pred_values[month] = (ei.min(), ei.max(), ei.mean(), ei.median())


        
        # Plot min/max for both products across the months
        months = sorted(bafu_values.keys())

        bafu_min = [bafu_values[m][0] for m in months]
        bafu_max = [bafu_values[m][1] for m in months]
        bafu_mean = [bafu_values[m][2] for m in months]
        bafu_median = [bafu_values[m][3] for m in months]
        pred_min = [pred_values[m][0] for m in months if m in pred_values]
        pred_max = [pred_values[m][1] for m in months if m in pred_values]
        pred_mean = [pred_values[m][2] for m in months if m in pred_values]
        pred_median = [pred_values[m][3] for m in months if m in pred_values]
        pred_months = [m for m in months if m in pred_values]

        fig, ax = plt.subplots(figsize=(12, 6))

       # BAFU
        ax.plot(months, bafu_min, 'o--', color='tab:blue', label='BAFU min')
        ax.plot(months, bafu_max, 's--', color='tab:blue', label='BAFU max')
        ax.plot(months, bafu_mean, '^-', color='tab:blue', label='BAFU mean')
        ax.plot(months, bafu_median, 'D-', color='tab:blue', label='BAFU median')
        ax.fill_between(months, bafu_min, bafu_max, color='tab:blue', alpha=0.15)

        # Predictions
        ax.plot(pred_months, pred_min, 'o--', color='tab:orange', label='Predicted min')
        ax.plot(pred_months, pred_max, 's--', color='tab:orange', label='Predicted max')
        ax.plot(pred_months, pred_mean, '^-', color='tab:orange', label='Predicted mean')
        ax.plot(pred_months, pred_median, 'D-', color='tab:orange', label='Predicted median')
        ax.fill_between(pred_months, pred_min, pred_max, color='tab:orange', alpha=0.15)

        ax.set_xticks(months)
        ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
        ax.set_xlabel("Month")
        ax.set_ylabel("EI (monthly)")
        ax.set_title("Monthly EI sum: BAFU vs Predictions")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"results/compare_bafu_minmax_{SUFFIX}.png")
        print("Saved comparison plot.")
        plt.close()

    if compare_bafu_maps:

        month_doy = {
            1: (1, 31), 2: (32, 59), 3: (60, 90), 4: (91, 120),
            5: (121, 151), 6: (152, 181), 7: (182, 212), 8: (213, 243),
            9: (244, 273), 10: (274, 304), 11: (305, 334), 12: (335, 365)
        }

        bafu_dir = 'results/maps/bafu'
        month_names = {
            1: 'jan', 2: 'feb', 3: 'mrz', 4: 'apr',
            5: 'mai', 6: 'jun', 7: 'jul', 8: 'aug',
            9: 'sep', 10: 'okt', 11: 'nov', 12: 'dez'
        }
        month_labels = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr',
            5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug',
            9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        }

        for month, (doy_start, doy_end) in month_doy.items():

            # --- Load BAFU raster ---
            month_name = month_names[month]
            pattern = os.path.join(bafu_dir, f'*{month_name}*.*tif*')
            matches = glob.glob(pattern, recursive=False)
            if not matches:
                pattern = os.path.join(bafu_dir, f'*{month_name.capitalize()}*.*tif*')
                matches = glob.glob(pattern, recursive=False)
            if not matches:
                pattern = os.path.join(bafu_dir, f'*{month:02d}*.*tif*')
                matches = glob.glob(pattern, recursive=False)
            if not matches:
                print(f"Warning: No BAFU tif for month {month}, skipping.")
                continue

            tif_path = matches[0]
            with rasterio.open(tif_path) as src:
                bafu_data = src.read(1).astype(float)
                bafu_transform = src.transform
                nodata = src.nodata
                if nodata is not None:
                    bafu_data[bafu_data == nodata] = np.nan

            # Convert BAFU raster to x, y, value arrays
            rows, cols = np.where(~np.isnan(bafu_data))
            bafu_xs, bafu_ys = rasterio.transform.xy(bafu_transform, rows, cols)
            bafu_xs = np.array(bafu_xs)
            bafu_ys = np.array(bafu_ys)
            bafu_vals = bafu_data[rows, cols]

            # --- Load predictions ---
            filter_expr = (ds.field("doy") >= doy_start) & (ds.field("doy") <= doy_end)
            table = dataset.to_table(columns=columns, filter=filter_expr)
            df = table.to_pandas()

            if df.empty:
                print(f"Warning: No prediction data for month {month}, skipping.")
                continue

            monthly_df = df.groupby(['x', 'y']).agg(
                predicted_EI_monthly=('predicted_EI_daily_avg', 'sum'),
            ).reset_index()

            # --- Shared color scale ---
            vmin = 0
            vmax = max(
                np.nanpercentile(bafu_vals, 99) if len(bafu_vals) > 0 else 0,
                monthly_df['predicted_EI_monthly'].quantile(0.99)
            )

            # --- Plot side by side ---
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

            # Left: Predictions
            sc1 = ax1.scatter(
                monthly_df['x'], monthly_df['y'],
                c=monthly_df['predicted_EI_monthly'],
                cmap='inferno', s=2, alpha=0.7,
                vmin=vmin, vmax=vmax
            )
            ax1.set_title(f"Predictions — {month_labels[month]}", fontsize=14)
            ax1.set_xlabel("X")
            ax1.set_ylabel("Y")
            ax1.set_aspect('equal')

            # Right: BAFU
            sc2 = ax2.scatter(
                bafu_xs, bafu_ys,
                c=bafu_vals,
                cmap='inferno', s=2, alpha=0.7,
                vmin=vmin, vmax=vmax
            )
            ax2.set_title(f"BAFU — {month_labels[month]}", fontsize=14)
            ax2.set_xlabel("X")
            ax2.set_ylabel("Y")
            ax2.set_aspect('equal')

            # Shared colorbar
            #fig.colorbar(sc2, ax=[ax1, ax2], label='Monthly EI sum', shrink=0.8)
            fig.colorbar(sc1, ax=ax1, label='Predicted EI', shrink=0.8)
            fig.colorbar(sc2, ax=ax2, label='BAFU EI', shrink=0.8)

            plt.suptitle(f"Monthly EI Comparison — {month_labels[month]}", fontsize=16)
            plt.tight_layout()
            plt.savefig(f"results/maps/compare_bafu_pred_month{month:02d}_{SUFFIX}.png", dpi=150)
            print(f"Saved comparison map for month {month}")
            plt.close()

    if compare_cumul_pct or check_spatial_corr:
        pixel_filters = [
            (ds.field("x") == row.x) & (ds.field("y") == row.y)
            for _, row in sample_coords.iterrows()
        ]
        pixel_filter = reduce(operator.or_, pixel_filters)

    if compare_cumul_pct:
        check_cumulative_pct_vs_bafu(
            dataset=dataset,
            columns=columns,
            bafu_dir='results/maps/bafu',
            out_dir='results/cumul_pct',
            filter_expr=pixel_filter
        )

    if check_spatial_corr:
        check_spatial_correlation(
            dataset=dataset,
            columns=columns,
            bafu_dir='results/maps/bafu',
            out_dir='results/spatial_corr',
            filter_expr=pixel_filter,
            src_crs="EPSG:32632"
        )

    if check_station_stats:

        month_doy = {
            1: (1, 31),
            2: (32, 59),
            3: (60, 90),
            4: (91, 120),
            5: (121, 151),
            6: (152, 181),
            7: (182, 212),
            8: (213, 243),
            9: (244, 273),
            10: (274, 304),
            11: (305, 334),
            12: (335, 365)
        }

        # Load station data and compute monthly total avg
        all_station_data = []

        station_dir = 'stations_data/stations_EIdaily_percent'

        for f in os.listdir(station_dir):

            df = pd.read_csv(os.path.join(station_dir, f))

            station_name = f.replace(".csv", "")

            for month, (doy_start, doy_end) in month_doy.items():
                monthly_df = df[
                    (df["doy"] >= doy_start) &
                    (df["doy"] <= doy_end)
                ]
                total_monthly_ei = monthly_df["EI_daily_avg"].sum()
                all_station_data.append({
                    "station": station_name,
                    "month": month,
                    "EI_monthly_total": total_monthly_ei
                })

        all_monthly_df = pd.DataFrame(all_station_data)
        """
        # Convert to matrix
        heatmap_df = all_monthly_df.pivot(
            index="station",
            columns="month",
            values="EI_monthly_total"
        )

        plt.figure(figsize=(10, 16))
        plt.imshow(
            heatmap_df,
            aspect="auto"
        )
        plt.colorbar(label="Monthly EI")
        plt.xlabel("Month")
        plt.ylabel("Station")
        plt.title("Monthly EI totals for all stations")
        plt.tight_layout()
        plt.savefig('station_monthly.png')
        """

        """
        plt.figure(figsize=(12,6))
        for station in all_monthly_df["station"].unique():
            sub = all_monthly_df[
                all_monthly_df["station"] == station
            ]
            plt.plot(
                sub["month"],
                sub["EI_monthly_total"],
                linewidth=0.5,
                alpha=0.3
            )
        monthly_median = (
        all_monthly_df
        .groupby("month")["EI_monthly_total"]
        .median()
        )
        plt.plot(
            monthly_median.index,
            monthly_median.values,
            linewidth=3
        )
        plt.xlabel("Month")
        plt.ylabel("Monthly EI total")
        plt.title("Monthly EI for all stations")
        plt.xticks(range(1,13))

        plt.tight_layout()
        plt.savefig('stations_monthly_line.png')
        """
        
        #---------------
        # For each month, create a tif with the values of the stations

        # add coords of stations
        metadata_df = pd.read_csv('stations_data/metadata/ogd-smn_meta_stations.csv', delimiter=';', encoding='latin1')[['station_abbr', 'station_coordinates_lv95_east', 'station_coordinates_lv95_north']]
        all_monthly_df['station_abbr'] = all_monthly_df['station'].apply(lambda x: x.split('_')[-1])
        all_monthly_df = all_monthly_df.merge(metadata_df, on='station_abbr')
        
        # -----------------------------------------
        # Snap coordinates to 100 m grid
        # -----------------------------------------

        grid_size = 100  # meters

        all_monthly_df["x_snap"] = (
            np.floor(
                all_monthly_df["station_coordinates_lv95_east"]
                / grid_size
            ) * grid_size
        )

        all_monthly_df["y_snap"] = (
            np.floor(
                all_monthly_df["station_coordinates_lv95_north"]
                / grid_size
            ) * grid_size
        )

        # -----------------------------------------
        # Aggregate stations inside same grid cell
        # -----------------------------------------

        agg_df = (
            all_monthly_df
            .groupby(["month", "x_snap", "y_snap"])
            ["EI_monthly_total"]
            .mean()   # mean across stations in same cell
            .reset_index()
        )

        # -----------------------------------------
        # Create raster per month
        # -----------------------------------------

        grid_size = 1000  # meters

        # Swiss LV95 bounding box (safe coverage)
        xmin = 2480000
        xmax = 2840000
        ymin = 1070000
        ymax = 1300000

        # Raster dimensions
        width = int((xmax - xmin) / grid_size)
        height = int((ymax - ymin) / grid_size)

        transform = from_origin(
            xmin,
            ymax,
            grid_size,
            grid_size
        )

        # -----------------------------------------
        # Loop over months
        # -----------------------------------------

        for month in range(1, 13):

            month_df = all_monthly_df[
                all_monthly_df["month"] == month
            ].copy()

            if month_df.empty:
                continue

            # Create empty raster
            grid = np.full(
                (height, width),
                np.nan,
                dtype=np.float32
            )

            xcol = "station_coordinates_lv95_east"
            ycol = "station_coordinates_lv95_north"

            # Convert coordinates → raster indices
            col_index = (
                (month_df[xcol] - xmin) / grid_size
            ).astype(int)

            row_index = (
                (ymax - month_df[ycol]) / grid_size
            ).astype(int)

            # Assign values
            for r, c, val in zip(
                row_index,
                col_index,
                month_df["EI_monthly_total"]
            ):

                if (
                    0 <= r < height and
                    0 <= c < width
                ):
                    grid[r, c] = val

            # Save raster
            output_file = (
                f"results/maps/station_EI_month_{month:02d}_{SUFFIX}.tif"
            )

            with rasterio.open(
                output_file,
                "w",
                driver="GTiff",
                height=height,
                width=width,
                count=1,
                dtype="float32",
                crs="EPSG:2056",
                transform=transform,
                nodata=np.nan
            ) as dst:

                dst.write(grid, 1)

            print("Saved:", output_file)

if __name__ == "__main__":
    
    SAVE_PATH = 'predictions/grid_EI_daily_avg_pred_20260424_nn3.parquet'
    SUFFIX = "20260424_nn3"
    
    plot_curves = False
    check_corr = False
    check_distr = False
    check_spatial = False
    compare_bafu = False
    compare_bafu_maps = True
    compare_cumul_pct = False    # Check #1: cumulative % curves vs BAFU
    check_spatial_corr = False   # Check spatial correlation (scatter) vs BAFU
    check_station_stats = False

    main()
