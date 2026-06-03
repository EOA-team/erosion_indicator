"""
Diagnostic script for the FC sampling + cleaning + gapfilling pipeline in
sample_FC.py.

Inputs (produced by main.py):
  - samples.pkl              (optional, for sample-stage overview)
  - samples_data_pred.pkl    (raw FC predictions)
  - samples_data_gpr.parquet (optional, for gapfilling diagnostics)

Outputs (all in OUT_DIR):
  Sampling overview
    diag_samples_map.png            — sampled pixel locations on basemap
    diag_samples_per_crop.png       — stacked counts per crop × year
  Cleaning diagnostics
    diag_mask_breakdown_by_month.png— per-row mask attribution by month
    diag_drop_fraction_hist.png     — per-field drop fraction
    diag_pv_monthly_pre_post.png    — PV seasonal cycle pre vs post
    diag_pipeline_funnel.png        — counts at each stage
    diag_obs_coverage.png           — obs-per-field-year (hist + CDF)
    diag_coverage_by_crop.png       — median obs per LNF code
    diag_sweep_*.png + .csv         — sensitivity to (max_missing_frac,
                                      drop_fraction_threshold)
  Gapfilling diagnostics
    gpr_fill_accounting.png         — fraction of output that is gapfilled
    gpr_gap_lengths.png             — gap lengths before vs after gapfilling
    gpr_monthly_obs_vs_fill.png     — distribution check, per component
    gpr_composition_sum.png         — PV+NPV+Soil sum-to-1 check
    gpr_quality_flags.png           — quality flag breakdown
    gpr_cv_scatter.png + records.csv— held-out cross-validation

Run after the main pipeline has produced the required inputs. Sections gracefully
skip if their inputs are missing.
"""
import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sample_FC import clean_timeseries_field

# =============================================================================
# Config (matches main.py CONFIG; edit here to test alternatives)
# =============================================================================
TS_COLS = ['lnf_code', 'yr', 'poly_id']
GROUP_COLS = ['poly_id', 'x', 'y', 'time', 'yr',
              'sampled_x', 'sampled_y', 'lnf_code', 'is_sample_pixel']
PIXEL_COLS = ['x', 'y']
VALUE_COLS = ['pv', 'npv', 'soil']

CIRRUS_THRESH = 500
MAX_MISSING_FRAC = 0.05
DROP_FRACTION_THRESHOLD = 0.7
MAX_GAP_DAYS = 15

# Input files
SAMPLES_PATH = 'samples.pkl'
FC_RAW_PATH = 'samples_data_pred.pkl'
GAPFILLED_PATH = 'samples_data_gpr.parquet'
LNF_LABELS_PATH = os.path.expanduser(
    '~/mnt/eo-nas1/data/landuse/documentation/LNF_code_classification_20260217.xlsx'
)

# Cross-validation
N_CV_FIELDS = 100

OUT_DIR = 'sample_analysis'
os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
# Helpers
# =============================================================================
def dedupe(df):
    """Average duplicate (pixel, date) rows arising from overlapping S2 granules."""
    return df.groupby(GROUP_COLS, as_index=False).mean(numeric_only=True)


def plot_monthly_mean_compare(df_pre, df_post, value_col, group_col,
                              ylabel, title, save_path):
    """Side-by-side monthly mean of value_col per group_col, pre and post cleaning."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax, df, sub in zip(axes, (df_pre, df_post),
                           ('Pre-cleaning', 'Post-cleaning')):
        df = df.copy()
        df['time'] = pd.to_datetime(df['time'])
        df['ym'] = df['time'].dt.to_period('M')
        m = df.groupby([group_col, 'ym'])[value_col].mean().reset_index()
        m['ym'] = m['ym'].dt.to_timestamp()
        for code, g in m.groupby(group_col):
            ax.plot(g['ym'], g[value_col], label=code, lw=1)
        ax.set_xlabel('Time')
        ax.set_title(sub)
        ax.tick_params(axis='x', rotation=45)
    axes[0].set_ylabel(ylabel)
    axes[1].legend(title=group_col, bbox_to_anchor=(1.01, 1),
                   loc='upper left', fontsize=7)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


def per_row_mask_breakdown(df, cirrus_thresh):
    """Label each row with its dominant mask cause (precedence: missing > cloud >
    shadow > snow > cirrus). Used for the stacked-bar mask plot."""
    df = df.copy()
    band_cols = [c for c in df.columns if c.startswith('s2_B')]

    if band_cols:
        missing = (df[band_cols].isna().all(axis=1)
                   | (df[band_cols] == 65535).all(axis=1))
        cloud = (df['s2_mask'] == 1) | (df['s2_SCL'].isin([8, 9, 10]))
        shadow = (df['s2_mask'] == 2) | (df['s2_SCL'] == 3)
        snow = (df['s2_mask'] == 3) | (df['s2_SCL'] == 11)
        cirrus = (df['s2_SCL'] == 10) & (df['s2_B02'] > cirrus_thresh)
    else:
        fc_cols = [c for c in VALUE_COLS if c in df.columns]
        missing = (df[fc_cols].isna().all(axis=1)
                   if fc_cols else pd.Series(False, index=df.index))
        cloud = shadow = snow = cirrus = pd.Series(False, index=df.index)

    cat = pd.Series('kept', index=df.index)
    cat = cat.mask(cirrus, 'cirrus')
    cat = cat.mask(snow, 'snow')
    cat = cat.mask(shadow, 'shadow')
    cat = cat.mask(cloud, 'cloud')
    cat = cat.mask(missing, 'missing')
    df['mask_cat'] = cat
    return df


def threshold_sweep(df_samples,
                    max_missing_frac_grid,
                    drop_fraction_threshold_grid,
                    cirrus_thresh):
    """For each (mmf, dft) combination, report retention + coverage stats."""
    rows = []
    for mmf in max_missing_frac_grid:
        df_clean = (df_samples
                    .groupby(TS_COLS, group_keys=False)
                    .apply(clean_timeseries_field,
                           max_missing_frac=mmf,
                           cirrus_thresh=cirrus_thresh)
                    .reset_index(drop=True))

        n_before = df_samples.groupby(TS_COLS).size().rename('n_total')
        n_after = df_clean.groupby(TS_COLS).size().rename('n_kept')
        stats = (pd.concat([n_before, n_after], axis=1)
                 .fillna(0).astype({'n_kept': int}))
        stats['drop_fraction'] = (stats['n_total'] - stats['n_kept']) / stats['n_total']

        for dft in drop_fraction_threshold_grid:
            kept = stats[stats['drop_fraction'] <= dft]
            keys_kept = kept.reset_index()[TS_COLS]
            df_filt = df_clean.merge(keys_kept, on=TS_COLS, how='inner')
            df_filt = dedupe(df_filt)

            obs = df_filt.groupby(TS_COLS)['time'].nunique()
            rows.append({
                'max_missing_frac': mmf,
                'drop_fraction_thresh': dft,
                'pct_fields_retained': 100 * len(kept) / max(len(stats), 1),
                'median_obs': obs.median() if len(obs) else 0,
                'p10_obs': obs.quantile(0.1) if len(obs) else 0,
                'pct_ge_10obs': 100 * (obs >= 10).mean() if len(obs) else 0,
            })
    return pd.DataFrame(rows)


def gap_lengths(df, ts_cols=TS_COLS):
    """Return a Series of consecutive-observation gap lengths (in days)."""
    gaps = []
    for _, g in df.sort_values('time').groupby(ts_cols):
        times = pd.to_datetime(g['time']).drop_duplicates().sort_values()
        if len(times) < 2:
            continue
        gaps.extend(times.diff().dropna().dt.days.tolist())
    return pd.Series(gaps, dtype=float)


# =============================================================================
# Load FC data
# =============================================================================
df_samples = pd.read_pickle(FC_RAW_PATH)
df_samples['time'] = pd.to_datetime(df_samples['time'])
print(f'Loaded {len(df_samples):,} raw rows, '
      f'{df_samples.groupby(TS_COLS).ngroups:,} field-years.')


# =============================================================================
# PART A — Sampling overview (optional, runs if samples.pkl is present)
# =============================================================================
if os.path.exists(SAMPLES_PATH):
    print('\n=== A. Sampling overview ===')
    df_loc = pd.read_pickle(SAMPLES_PATH)
    gdf = (gpd.GeoDataFrame(df_loc, geometry=df_loc['point_geom'], crs=32632)
           .drop(columns=['point_geom', 'polygon_geom']))
    gdf_web = gdf.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(14, 10))
    gdf_web.plot(
        ax=ax, column='lnf_code', categorical=True, legend=True,
        alpha=0.7, markersize=4, cmap='tab20',
        legend_kwds={'title': 'lnf_code',
                     'bbox_to_anchor': (1.05, 1), 'loc': 'upper left'},
    )
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom='auto')
    ax.set_title('Sampled pixel locations (based on LNF 2021-2024)')
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/diag_samples_map.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {OUT_DIR}/diag_samples_map.png')

    counts = (df_loc.groupby(['lnf_code', 'yr']).size()
              .unstack('yr', fill_value=0)
              .sort_index(axis=1))

    if os.path.exists(LNF_LABELS_PATH):
        labels = pd.read_excel(LNF_LABELS_PATH,
                               sheet_name='label_sheet')[['LNF_code', 'Crop_EN']]
        name_map = dict(labels.drop_duplicates('LNF_code').values)
        counts.index = [f"{c} — {name_map.get(c, '?')}" for c in counts.index]
    else:
        print(f'(LNF labels not found at {LNF_LABELS_PATH}; using codes only)')

    counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).index]
    print(f'Total samples: {len(df_loc)}  |  crops: {counts.shape[0]}  |  '
          f'years: {list(counts.columns)}')

    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(counts)), 6))
    counts.plot(kind='bar', stacked=True, ax=ax, colormap='viridis',
                width=0.8, edgecolor='white')
    totals = counts.sum(axis=1)
    for i, t in enumerate(totals):
        ax.text(i, t, f' {int(t)}', ha='center', va='bottom', fontsize=8)
    ax.set_xlabel('LNF code — crop')
    ax.set_ylabel('Number of sampled fields')
    ax.set_title(f'Sampled fields per crop, by year (n={len(df_loc)})')
    ax.legend(title='Year', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.margins(y=0.08)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/diag_samples_per_crop.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {OUT_DIR}/diag_samples_per_crop.png')

    del df_loc, gdf, gdf_web
else:
    print(f'(Skipping sampling overview: {SAMPLES_PATH} not found)')


# =============================================================================
# PART B — Cleaning diagnostics
# =============================================================================
print('\n=== B. Cleaning diagnostics ===')

# -----------------------------------------------------------------------------
# B.1 Mask-category breakdown (BEFORE the per-date filter)
# -----------------------------------------------------------------------------
df_tagged = per_row_mask_breakdown(df_samples, CIRRUS_THRESH)
df_tagged['month'] = df_tagged['time'].dt.month

cat_order = ['missing', 'cloud', 'shadow', 'snow', 'cirrus', 'kept']
breakdown = (df_tagged
             .groupby(['month', 'mask_cat']).size()
             .unstack('mask_cat', fill_value=0)
             .reindex(columns=cat_order, fill_value=0))
breakdown_pct = breakdown.div(breakdown.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(11, 5))
colors = {'missing': '#888', 'cloud': '#bbb', 'shadow': '#5a78a8',
          'snow': '#c0d6e8', 'cirrus': '#e8a87c', 'kept': '#69a96b'}
breakdown_pct.plot(kind='bar', stacked=True, ax=ax,
                   color=[colors[c] for c in cat_order],
                   width=0.85, edgecolor='white')
ax.set_xlabel('Month'); ax.set_ylabel('% of rows')
ax.set_title(f'Per-row mask attribution by month (cirrus_thresh={CIRRUS_THRESH})')
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/diag_mask_breakdown_by_month.png',
            dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {OUT_DIR}/diag_mask_breakdown_by_month.png')
del df_tagged

# -----------------------------------------------------------------------------
# B.2 Run the cleaning pipeline once at the configured thresholds
# -----------------------------------------------------------------------------
df_clean = (df_samples
            .groupby(TS_COLS, group_keys=False)
            .apply(clean_timeseries_field,
                   max_missing_frac=MAX_MISSING_FRAC,
                   cirrus_thresh=CIRRUS_THRESH)
            .reset_index(drop=True))

n_before = df_samples.groupby(TS_COLS).size().rename('n_total')
n_after = df_clean.groupby(TS_COLS).size().rename('n_kept')
drop_stats = (pd.concat([n_before, n_after], axis=1)
              .fillna(0).astype({'n_kept': int}))
drop_stats['drop_fraction'] = (
    (drop_stats['n_total'] - drop_stats['n_kept']) / drop_stats['n_total']
)
drop_stats = drop_stats.reset_index()

keys_kept = drop_stats[drop_stats['drop_fraction']
                       <= DROP_FRACTION_THRESHOLD][TS_COLS]
df_filtered = df_clean.merge(keys_kept, on=TS_COLS, how='inner')
df_raw_matched = df_samples.merge(keys_kept, on=TS_COLS, how='inner')

df_samples_d = dedupe(df_samples)
df_clean_d = dedupe(df_clean)
df_filtered_d = dedupe(df_filtered)
df_raw_matched_d = dedupe(df_raw_matched)

print(f'Fields retained by drop_fraction_threshold={DROP_FRACTION_THRESHOLD}: '
      f'{len(keys_kept)}/{len(drop_stats)}')

# -----------------------------------------------------------------------------
# B.3 Drop-fraction histogram
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(drop_stats['drop_fraction'], bins=40,
        edgecolor='white', color='steelblue')
ax.axvline(DROP_FRACTION_THRESHOLD, color='crimson', lw=1.5,
           label=f'threshold = {DROP_FRACTION_THRESHOLD}')
pct_dropped = (drop_stats['drop_fraction'] > DROP_FRACTION_THRESHOLD).mean() * 100
ax.set_xlabel('Fraction of rows dropped per field-year')
ax.set_ylabel('Number of field-years')
ax.set_title(f'Per-field drop fraction (after per-row + per-date masking)\n'
             f'{pct_dropped:.1f}% of fields exceed threshold and are removed')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/diag_drop_fraction_hist.png',
            dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {OUT_DIR}/diag_drop_fraction_hist.png')

# -----------------------------------------------------------------------------
# B.4 Monthly mean PV — pre vs post cleaning
# -----------------------------------------------------------------------------
plot_monthly_mean_compare(
    df_raw_matched_d, df_filtered_d,
    value_col='pv', group_col='lnf_code',
    ylabel='Monthly mean PV',
    title='PV seasonal pattern before vs after cleaning (same field set)',
    save_path=f'{OUT_DIR}/diag_pv_monthly_pre_post.png',
)

# -----------------------------------------------------------------------------
# B.5 Pipeline funnel
# -----------------------------------------------------------------------------
def stage_metrics(df, name):
    obs = df.groupby(TS_COLS)['time'].nunique()
    return {
        'stage': name,
        'n_fields': len(obs),
        'mean_obs': obs.mean() if len(obs) else 0,
        'median_obs': obs.median() if len(obs) else 0,
        'pct_ge_10': 100 * (obs >= 10).mean() if len(obs) else 0,
    }

summary = pd.DataFrame([
    stage_metrics(df_samples_d, 'raw'),
    stage_metrics(df_clean_d, 'per-row+per-date'),
    stage_metrics(df_filtered_d, 'per-field filtered'),
])
print('\nPipeline funnel:')
print(summary.to_string(index=False))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.bar(summary['stage'], summary['n_fields'], color='steelblue')
ax1.set_ylabel('Number of field-years')
ax1.set_title('Field-years at each stage')
ax1.tick_params(axis='x', rotation=15)
for i, v in enumerate(summary['n_fields']):
    ax1.text(i, v, f' {int(v)}', ha='center', va='bottom', fontsize=9)

ax2b = ax2.twinx()
l1 = ax2.plot(summary['stage'], summary['median_obs'],
              marker='o', color='steelblue', label='median obs/field')
l2 = ax2b.plot(summary['stage'], summary['pct_ge_10'],
               marker='s', color='crimson', label='% ≥10 obs')
ax2.set_ylabel('Median obs/field', color='steelblue')
ax2b.set_ylabel('% ≥10 obs', color='crimson')
ax2.set_title('Temporal coverage at each stage')
ax2.tick_params(axis='x', rotation=15)
lns = l1 + l2
ax2.legend(lns, [l.get_label() for l in lns], loc='lower left')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/diag_pipeline_funnel.png',
            dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {OUT_DIR}/diag_pipeline_funnel.png')

# -----------------------------------------------------------------------------
# B.6 Obs-per-field-year distribution
# -----------------------------------------------------------------------------
clean_obs = df_filtered_d.groupby(TS_COLS)['time'].nunique().rename('n_obs')

print('\nFraction of field-years below thresholds (final stage):')
for thresh in [5, 10, 15, 20]:
    frac = (clean_obs < thresh).mean()
    print(f'  < {thresh:2d} obs: {100 * frac:.1f}%')

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(clean_obs, bins=40, edgecolor='white', color='steelblue')
for thresh, ls in [(5, '--'), (10, '-'), (20, ':')]:
    axes[0].axvline(thresh, color='crimson', linestyle=ls, lw=1.2,
                    label=f'{thresh} obs')
axes[0].set_xlabel('Clean observations per field-year')
axes[0].set_ylabel('Number of field-years')
axes[0].set_title('Distribution of clean observations')
axes[0].legend(fontsize=8)

sorted_obs = np.sort(clean_obs.values)
cdf = np.arange(1, len(sorted_obs) + 1) / len(sorted_obs)
axes[1].plot(sorted_obs, cdf, color='steelblue')
for thresh, ls in [(5, '--'), (10, '-'), (20, ':')]:
    axes[1].axvline(thresh, color='crimson', linestyle=ls, lw=1.2)
axes[1].set_xlabel('Clean observations per field-year')
axes[1].set_ylabel('Cumulative fraction')
axes[1].set_title('CDF of clean observations')

plt.suptitle(
    f'Field-years = {len(clean_obs)} | median = {clean_obs.median():.0f} | '
    f'p10 = {clean_obs.quantile(0.1):.0f}',
    fontsize=9,
)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/diag_obs_coverage.png',
            dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {OUT_DIR}/diag_obs_coverage.png')

# -----------------------------------------------------------------------------
# B.7 Per-crop median coverage
# -----------------------------------------------------------------------------
per_crop = (clean_obs.reset_index()
            .groupby('lnf_code')['n_obs']
            .agg(['median', 'count'])
            .sort_values('median'))

fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * len(per_crop))))
colors_bar = ['crimson' if m < 10 else 'steelblue' for m in per_crop['median']]
ax.barh([str(c) for c in per_crop.index], per_crop['median'], color=colors_bar)
ax.axvline(10, color='black', lw=0.8, linestyle='--', label='10 obs')
for i, (m, n) in enumerate(zip(per_crop['median'], per_crop['count'])):
    ax.text(m, i, f' {m:.0f} (n={n})', va='center', fontsize=7)
ax.set_xlabel('Median clean observations per field-year')
ax.set_ylabel('LNF code')
ax.set_title('Temporal coverage by crop (red = below 10 obs)')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/diag_coverage_by_crop.png',
            dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {OUT_DIR}/diag_coverage_by_crop.png')

# -----------------------------------------------------------------------------
# B.8 Threshold sweep
# -----------------------------------------------------------------------------
print('\nRunning threshold sweep...')
sweep = threshold_sweep(
    df_samples,
    max_missing_frac_grid=[0.05, 0.10, 0.20, 0.30],
    drop_fraction_threshold_grid=[0.5, 0.7, 0.9],
    cirrus_thresh=CIRRUS_THRESH,
)
print('\nThreshold sweep results:')
print(sweep.to_string(index=False))
sweep.to_csv(f'{OUT_DIR}/diag_threshold_sweep.csv', index=False)
print(f'Saved: {OUT_DIR}/diag_threshold_sweep.csv')

for metric in ['pct_fields_retained', 'median_obs', 'pct_ge_10obs']:
    piv = sweep.pivot(index='max_missing_frac',
                      columns='drop_fraction_thresh',
                      values=metric)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(piv.values, aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    ax.set_xlabel('drop_fraction_threshold')
    ax.set_ylabel('max_missing_frac')
    ax.set_title(metric)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f'{piv.values[i, j]:.1f}',
                    ha='center', va='center',
                    color='white' if piv.values[i, j] < piv.values.mean() else 'black',
                    fontsize=9)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/diag_sweep_{metric}.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {OUT_DIR}/diag_sweep_{metric}.png')


# =============================================================================
# PART B' — Sampling overview AFTER cleaning
#     Shows what's actually left after all three filters. Mirrors Part A but
#     restricted to field-years that survived per-row + per-date + per-field
#     filtering. Requires samples.pkl (for the geometry).
# =============================================================================
if os.path.exists(SAMPLES_PATH):
    print('\n=== B\'. Sampling overview after cleaning ===')
    df_loc = pd.read_pickle(SAMPLES_PATH)

    # Restrict samples.pkl to the (lnf_code, yr, poly_id) keys that survived
    surviving_keys = df_filtered_d[TS_COLS].drop_duplicates()
    df_loc_cleaned = df_loc.merge(surviving_keys, on=TS_COLS, how='inner')

    n_in = len(df_loc)
    n_out = len(df_loc_cleaned)
    print(f'Samples retained after cleaning: {n_out:,} / {n_in:,} '
          f'({100 * n_out / max(n_in, 1):.1f}%)')

    if n_out == 0:
        print('(No samples left after cleaning — skipping post-cleaning plots)')
    else:
        # ---- Map of cleaned samples ----
        gdf = (gpd.GeoDataFrame(df_loc_cleaned,
                                geometry=df_loc_cleaned['point_geom'], crs=32632)
               .drop(columns=['point_geom', 'polygon_geom']))
        gdf_web = gdf.to_crs(epsg=3857)

        fig, ax = plt.subplots(figsize=(14, 10))
        gdf_web.plot(
            ax=ax, column='lnf_code', categorical=True, legend=True,
            alpha=0.7, markersize=4, cmap='tab20',
            legend_kwds={'title': 'lnf_code',
                         'bbox_to_anchor': (1.05, 1), 'loc': 'upper left'},
        )
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom='auto')
        ax.set_title(f'Sampled pixel locations after cleaning '
                     f'(n={n_out:,}, {100 * n_out / max(n_in, 1):.0f}% of raw)')
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(f'{OUT_DIR}/diag_samples_map_cleaned.png',
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved: {OUT_DIR}/diag_samples_map_cleaned.png')

        # ---- Counts per crop × year, cleaned ----
        counts_c = (df_loc_cleaned.groupby(['lnf_code', 'yr']).size()
                    .unstack('yr', fill_value=0)
                    .sort_index(axis=1))

        if os.path.exists(LNF_LABELS_PATH):
            labels = pd.read_excel(LNF_LABELS_PATH,
                                   sheet_name='label_sheet')[['LNF_code', 'Crop_EN']]
            name_map = dict(labels.drop_duplicates('LNF_code').values)
            counts_c.index = [f"{c} — {name_map.get(c, '?')}" for c in counts_c.index]

        counts_c = counts_c.loc[counts_c.sum(axis=1).sort_values(ascending=False).index]

        fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(counts_c)), 6))
        counts_c.plot(kind='bar', stacked=True, ax=ax, colormap='viridis',
                      width=0.8, edgecolor='white')
        totals = counts_c.sum(axis=1)
        for i, t in enumerate(totals):
            ax.text(i, t, f' {int(t)}', ha='center', va='bottom', fontsize=8)
        ax.set_xlabel('LNF code — crop')
        ax.set_ylabel('Number of sampled fields')
        ax.set_title(f'Sampled fields per crop after cleaning, by year (n={n_out})')
        ax.legend(title='Year', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.margins(y=0.08)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{OUT_DIR}/diag_samples_per_crop_cleaned.png',
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved: {OUT_DIR}/diag_samples_per_crop_cleaned.png')

        # ---- Retention per crop: side-by-side raw vs cleaned ----
        counts_raw_by_crop = df_loc.groupby('lnf_code').size().rename('raw')
        counts_cln_by_crop = df_loc_cleaned.groupby('lnf_code').size().rename('cleaned')
        retention = (pd.concat([counts_raw_by_crop, counts_cln_by_crop], axis=1)
                     .fillna(0).astype(int))
        retention['pct_retained'] = (
            100 * retention['cleaned'] / retention['raw'].clip(lower=1)
        )
        retention = retention.sort_values('raw', ascending=False)

        if os.path.exists(LNF_LABELS_PATH):
            retention.index = [f"{c} — {name_map.get(c, '?')}"
                               for c in retention.index]

        fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, max(4, 0.3 * len(retention))),
                                       sharey=True,
                                       gridspec_kw={'width_ratios': [3, 1]})
        ypos = np.arange(len(retention))
        axL.barh(ypos - 0.2, retention['raw'], height=0.4,
                 color='lightgray', label='Raw')
        axL.barh(ypos + 0.2, retention['cleaned'], height=0.4,
                 color='steelblue', label='Cleaned')
        axL.set_yticks(ypos)
        axL.set_yticklabels(retention.index, fontsize=8)
        axL.invert_yaxis()
        axL.set_xlabel('Number of sampled fields')
        axL.set_title('Raw vs cleaned counts per crop')
        axL.legend(loc='lower right')

        colors_r = ['crimson' if p < 50 else 'steelblue'
                    for p in retention['pct_retained']]
        axR.barh(ypos, retention['pct_retained'], color=colors_r)
        axR.axvline(50, color='black', linestyle='--', lw=0.8, label='50%')
        axR.set_xlabel('% retained')
        axR.set_title('Retention rate (red = <50%)')
        axR.set_xlim(0, 105)
        axR.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(f'{OUT_DIR}/diag_retention_by_crop.png',
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved: {OUT_DIR}/diag_retention_by_crop.png')

        # ---- Crop-grouping view: orig_lnf_code within each lnf_code group ----
        # `lnf_code` is the pooled/main calibration code; `orig_lnf_code` is the
        # true crop before analogy pooling. These plots show the composition of
        # each pooled group. Skipped if the (older) samples.pkl lacks orig codes.
        if 'orig_lnf_code' not in df_loc_cleaned.columns:
            print("(Skipping crop-grouping plots: 'orig_lnf_code' not in "
                  "samples.pkl — re-run sampling to record it)")
        else:
            n_groups = df_loc_cleaned['lnf_code'].nunique()
            n_orig = df_loc_cleaned['orig_lnf_code'].nunique()
            n_pooled = (df_loc_cleaned.groupby('lnf_code')['orig_lnf_code']
                        .nunique() > 1).sum()
            print(f'Crop groups: {n_groups} lnf_code group(s) from {n_orig} '
                  f'orig_lnf_code(s); {n_pooled} group(s) pool >1 orig code')

            # German labels for the grouped bar (rest of script stays English).
            if os.path.exists(LNF_LABELS_PATH):
                labels_de = pd.read_excel(
                    LNF_LABELS_PATH, sheet_name='label_sheet')[['LNF_code', 'Crop_DE']]
                name_map_de = dict(labels_de.drop_duplicates('LNF_code').values)
            else:
                name_map_de = {}

            # other orig_lnf_code(s) pooled into each lnf_code group (excl. main)
            others = (df_loc_cleaned.groupby('lnf_code')['orig_lnf_code']
                      .apply(lambda s: sorted(set(s) - {s.name})))

            def _group_label(c):
                de = name_map_de.get(c, '?') if name_map_de else None
                base = f"{c} — {de}" if de else str(c)
                extra = others.get(c, [])
                if extra:
                    base += f" (+ {', '.join(str(o) for o in extra)})"
                return base

            # ---- Map of cleaned samples, colored by orig_lnf_code ----
            gdf_o = (gpd.GeoDataFrame(df_loc_cleaned,
                                      geometry=df_loc_cleaned['point_geom'], crs=32632)
                     .drop(columns=['point_geom', 'polygon_geom']))
            gdf_o_web = gdf_o.to_crs(epsg=3857)

            fig, ax = plt.subplots(figsize=(14, 10))
            gdf_o_web.plot(
                ax=ax, column='orig_lnf_code', categorical=True, legend=True,
                alpha=0.7, markersize=4, cmap='tab20',
                legend_kwds={'title': 'orig_lnf_code',
                             'bbox_to_anchor': (1.05, 1), 'loc': 'upper left'},
            )
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom='auto')
            ax.set_title(f'Sampled pixel locations after cleaning, by original crop '
                         f'(orig_lnf_code; n={n_out:,})')
            ax.set_axis_off()
            plt.tight_layout()
            plt.savefig(f'{OUT_DIR}/diag_samples_map_cleaned_by_origcrop.png',
                        dpi=150, bbox_inches='tight')
            plt.close()
            print(f'Saved: {OUT_DIR}/diag_samples_map_cleaned_by_origcrop.png')

            # ---- Stacked bar: x = lnf_code group, stack = orig_lnf_code ----
            grp = (df_loc_cleaned.groupby(['lnf_code', 'orig_lnf_code']).size()
                   .unstack('orig_lnf_code', fill_value=0))
            # order groups by total size (desc), orig codes by total (desc)
            grp = grp.loc[grp.sum(axis=1).sort_values(ascending=False).index]
            grp = grp[grp.sum(axis=0).sort_values(ascending=False).index]
            grp.index = [_group_label(c) for c in grp.index]

            fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(grp)), 6))
            grp.plot(kind='bar', stacked=True, ax=ax, colormap='tab20',
                     width=0.8, edgecolor='white')
            totals = grp.sum(axis=1)
            for i, t in enumerate(totals):
                ax.text(i, t, f' {int(t)}', ha='center', va='bottom', fontsize=8)
            ax.set_xlabel('LNF code group (pooled / main code)')
            ax.set_ylabel('Number of sampled fields')
            ax.set_title(f'Sampled fields per crop group after cleaning, '
                         f'by original crop (n={n_out})')
            ax.legend(title='orig_lnf_code', bbox_to_anchor=(1.02, 1),
                      loc='upper left', fontsize=7, ncol=1)
            ax.margins(y=0.08)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f'{OUT_DIR}/diag_samples_per_group_cleaned.png',
                        dpi=150, bbox_inches='tight')
            plt.close()
            print(f'Saved: {OUT_DIR}/diag_samples_per_group_cleaned.png')

            del gdf_o, gdf_o_web

    del df_loc
    if n_out > 0:
        del df_loc_cleaned, gdf, gdf_web
else:
    print(f'\n(Skipping post-cleaning sampling overview: {SAMPLES_PATH} not found)')


# =============================================================================
# PART C — Gapfilling diagnostics (optional, runs if parquet is present)
# =============================================================================

if not os.path.exists(GAPFILLED_PATH):
    print(f'\n(Skipping gapfilling diagnostics: {GAPFILLED_PATH} not found)')
    print('\nDone.')
    sys.exit(0)

print('\n=== C. Gapfilling diagnostics ===')

df_gpr = pd.read_parquet(GAPFILLED_PATH)
df_gpr['time'] = pd.to_datetime(df_gpr['time'])

# df_gpr keeps only the sampled pixel; restrict df_filtered_d the same way
df_clean_sp = df_filtered_d[df_filtered_d['is_sample_pixel']].copy()

print(f'  gapfilled    : {len(df_gpr):>10,} rows, '
      f'{df_gpr.groupby(TS_COLS).ngroups:>6,} field-years')
print(f'  cleaned (sp) : {len(df_clean_sp):>10,} rows, '
      f'{df_clean_sp.groupby(TS_COLS).ngroups:>6,} field-years')

# -----------------------------------------------------------------------------
# C.1 Fill-point accounting
# -----------------------------------------------------------------------------
n_fill = int(df_gpr['is_gapfilled'].sum())
n_obs = int((~df_gpr['is_gapfilled']).sum())
print(f'\nFill points: {n_fill:,}  ({100 * n_fill / len(df_gpr):.1f}% of output)')
print(f'Observed   : {n_obs:,}')

per_field = (df_gpr
             .groupby(TS_COLS)['is_gapfilled']
             .agg(['sum', 'count']))
per_field['fill_ratio'] = per_field['sum'] / per_field['count']

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(per_field['fill_ratio'], bins=30,
             edgecolor='white', color='steelblue')
axes[0].set_xlabel('Fill ratio per field-year')
axes[0].set_ylabel('Number of field-years')
axes[0].set_title(f'Fraction of output that is gapfilled\n'
                  f'(median = {per_field["fill_ratio"].median():.2f})')

df_gpr['month'] = df_gpr['time'].dt.month
month_breakdown = (df_gpr.groupby(['month', 'is_gapfilled']).size()
                   .unstack(fill_value=0))
month_pct = month_breakdown.div(month_breakdown.sum(axis=1), axis=0) * 100
month_pct.plot(kind='bar', stacked=True, ax=axes[1],
               color=['steelblue', 'crimson'], width=0.85, edgecolor='white')
axes[1].set_xlabel('Month'); axes[1].set_ylabel('% of output')
axes[1].set_title('Observed vs gapfilled, by month')
axes[1].legend(['Observed', 'Gapfilled'], fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/gpr_fill_accounting.png',
            dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {OUT_DIR}/gpr_fill_accounting.png')

# -----------------------------------------------------------------------------
# C.2 Gap-length distribution: before vs after
# -----------------------------------------------------------------------------
gaps_clean = gap_lengths(df_clean_sp)
gaps_gpr = gap_lengths(df_gpr)

fig, ax = plt.subplots(figsize=(9, 5))
bins = np.arange(0, 60, 2)
ax.hist(gaps_clean, bins=bins, alpha=0.6,
        label=f'Cleaned (median={gaps_clean.median():.0f}d)',
        color='steelblue', edgecolor='white')
ax.hist(gaps_gpr, bins=bins, alpha=0.6,
        label=f'Gapfilled (median={gaps_gpr.median():.0f}d)',
        color='crimson', edgecolor='white')
ax.axvline(MAX_GAP_DAYS, color='black', linestyle='--', lw=1,
           label=f'max_gap_days = {MAX_GAP_DAYS}')
ax.set_xlabel('Days between consecutive observations')
ax.set_ylabel('Count')
ax.set_title('Gap length distribution: cleaned vs gapfilled')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/gpr_gap_lengths.png',
            dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {OUT_DIR}/gpr_gap_lengths.png')

# -----------------------------------------------------------------------------
# C.3 Distribution check: do gapfilled values match the clean distribution?
# -----------------------------------------------------------------------------
df_obs_only = df_gpr[~df_gpr['is_gapfilled']].copy()
df_fill_only = df_gpr[df_gpr['is_gapfilled']].copy()

fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
for ax, col in zip(axes, VALUE_COLS):
    obs_m = df_obs_only.groupby('month')[col].agg(['mean', 'std'])
    fil_m = df_fill_only.groupby('month')[col].agg(['mean', 'std'])
    ax.errorbar(obs_m.index, obs_m['mean'], yerr=obs_m['std'],
                fmt='o-', color='steelblue', capsize=3, label='Observed', lw=1.5)
    ax.errorbar(fil_m.index + 0.1, fil_m['mean'], yerr=fil_m['std'],
                fmt='s-', color='crimson', capsize=3, label='Gapfilled', lw=1.5)
    ax.set_ylabel(f'{col} fraction')
    ax.set_title(f'{col}: monthly mean ± std')
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)
axes[-1].set_xlabel('Month')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/gpr_monthly_obs_vs_fill.png',
            dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {OUT_DIR}/gpr_monthly_obs_vs_fill.png')

# -----------------------------------------------------------------------------
# C.4 Composition validity: PV + NPV + Soil = 1, all in [0,1]
# -----------------------------------------------------------------------------
df_gpr['total'] = df_gpr[VALUE_COLS].sum(axis=1)
sum_dev = (df_gpr['total'] - 1.0).abs()
range_violations = (
    (df_gpr[VALUE_COLS] < 0).any(axis=1) | (df_gpr[VALUE_COLS] > 1).any(axis=1)
)
print(f'\nComposition validity:')
print(f'  Mean |sum-1|     : {sum_dev.mean():.4f}')
print(f'  Max  |sum-1|     : {sum_dev.max():.4f}')
print(f'  Rows outside [0,1]: {int(range_violations.sum())} '
      f'({100 * range_violations.mean():.2f}%)')

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, kind, mask, color in [
    (axes[0], 'Observed', ~df_gpr['is_gapfilled'], 'steelblue'),
    (axes[1], 'Gapfilled', df_gpr['is_gapfilled'], 'crimson'),
]:
    d = sum_dev[mask]
    ax.hist(d, bins=40, color=color, edgecolor='white')
    ax.axvline(1e-3, color='black', linestyle='--', lw=1, label='1e-3')
    ax.set_xlabel('|PV+NPV+Soil - 1|')
    ax.set_ylabel('Count')
    ax.set_title(f'{kind}: mean={d.mean():.4f}, max={d.max():.4f}')
    ax.set_yscale('log')
    ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/gpr_composition_sum.png',
            dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {OUT_DIR}/gpr_composition_sum.png')

# -----------------------------------------------------------------------------
# C.5 Quality-flag breakdown
# -----------------------------------------------------------------------------
if 'fc_quality_flag' in df_gpr.columns:
    qf = df_gpr.groupby(['is_gapfilled', 'fc_quality_flag']).size().unstack(fill_value=0)
    qf_pct = qf.div(qf.sum(axis=1), axis=0) * 100
    print(f'\nQuality flag distribution (% of rows):')
    print(qf_pct.round(1).to_string())

    fig, ax = plt.subplots(figsize=(8, 4))
    qf_pct.plot(kind='barh', stacked=True, ax=ax,
                colormap='RdYlGn_r', edgecolor='white')
    ax.set_xlabel('% of rows')
    ax.set_ylabel('is_gapfilled')
    ax.set_title('Quality flag breakdown\n(0=ok, 1=>p80 unc, 2=>p95 unc)')
    ax.legend(title='quality_flag', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/gpr_quality_flags.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {OUT_DIR}/gpr_quality_flags.png')

# -----------------------------------------------------------------------------
# C.6 Held-out cross-validation
#     Drop a random clean observation per field, re-fit gapfiller, predict it,
#     compare to truth. The only diagnostic that tests *accuracy*.
# -----------------------------------------------------------------------------
RNG = np.random.default_rng(42)

try:
    from sample_FC import _gapfill_one_field_alr
except ImportError:
    print('\nCould not import _gapfill_one_field_alr — skipping CV.')
    _gapfill_one_field_alr = None

if _gapfill_one_field_alr is not None:
    print(f'\nRunning held-out CV on {N_CV_FIELDS} random field-years...')

    field_keys = list(df_filtered_d.groupby(TS_COLS).groups.keys())
    sampled = RNG.choice(len(field_keys),
                         size=min(N_CV_FIELDS, len(field_keys)),
                         replace=False)
    sampled_keys = [field_keys[i] for i in sampled]

    records = []
    for n, keys in enumerate(sampled_keys):
        mask = (df_filtered_d[TS_COLS]
                == pd.Series(keys, index=TS_COLS)).all(axis=1)
        group_clean = df_filtered_d[mask].copy()
        if len(group_clean) < 5:
            continue

        sp_rows = group_clean[group_clean['is_sample_pixel']]
        if len(sp_rows) < 4:
            continue
        held_idx = RNG.choice(sp_rows.index)
        held_row = group_clean.loc[held_idx]
        held_time = held_row['time']

        gc_minus = group_clean.drop(index=held_idx)

        try:
            out = _gapfill_one_field_alr(
                keys, gc_minus, TS_COLS, VALUE_COLS, PIXEL_COLS,
                alpha=1e-4, max_gap_days=MAX_GAP_DAYS,
            )
        except Exception:
            continue

        if out is None or len(out) == 0:
            continue

        out['time'] = pd.to_datetime(out['time'])
        nearest = (out['time'] - held_time).abs().idxmin()
        delta_days = abs((out.loc[nearest, 'time'] - held_time).days)
        if delta_days > 8:
            continue

        rec = {
            'lnf_code': keys[0],
            'yr': keys[1],
            'delta_days': delta_days,
            'is_gapfill_at_held_date': bool(out.loc[nearest, 'is_gapfilled']),
        }
        for col in VALUE_COLS:
            rec[f'{col}_true'] = held_row[col]
            rec[f'{col}_pred'] = out.loc[nearest, col]
        records.append(rec)

        if (n + 1) % 20 == 0:
            print(f'  {n + 1}/{len(sampled_keys)} done', flush=True)

    cv = pd.DataFrame(records)
    print(f'\nCV records collected: {len(cv)}')

    if len(cv) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, col in zip(axes, VALUE_COLS):
            true = cv[f'{col}_true']
            pred = cv[f'{col}_pred']
            err = pred - true
            rmse = np.sqrt((err ** 2).mean())
            bias = err.mean()
            r = np.corrcoef(true, pred)[0, 1] if len(cv) > 1 else np.nan

            ax.scatter(true, pred, alpha=0.5, s=15, color='steelblue')
            ax.plot([0, 1], [0, 1], '--', color='black', lw=1)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xlabel(f'{col} true')
            ax.set_ylabel(f'{col} predicted')
            ax.set_title(f'{col}: RMSE={rmse:.3f}, bias={bias:+.3f}, r={r:.2f}')
        plt.suptitle(f'Held-out cross-validation (n={len(cv)} field-years)')
        plt.tight_layout()
        plt.savefig(f'{OUT_DIR}/gpr_cv_scatter.png',
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved: {OUT_DIR}/gpr_cv_scatter.png')

        cv.to_csv(f'{OUT_DIR}/gpr_cv_records.csv', index=False)
        print(f'Saved: {OUT_DIR}/gpr_cv_records.csv')

print('\nDone.')
