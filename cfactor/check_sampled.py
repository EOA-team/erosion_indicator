"""
Diagnostic script to evaluate the cleaning thresholds applied in sample_FC.py.

Three thresholds drive the cleaning pipeline:
  1. cirrus_thresh           — per-row mask inside clean_timeseries_field
  2. max_missing_frac        — per-date drop inside clean_timeseries_field
  3. drop_fraction_threshold — per-field drop in run_sampling_pipeline

This script produces:
  - Monthly mean PV (raw vs cleaned)  -> shape sanity check
  - Drop-fraction histogram per field -> judges drop_fraction_threshold
  - Mask-category breakdown by month  -> shows what each mask removes
  - Pipeline funnel (counts at each stage)
  - Obs-per-field-year distribution   -> usability for downstream gapfilling
  - Per-crop median coverage          -> spots under-sampled crops
  - Threshold sweep table             -> sensitivity to (max_missing_frac,
                                        drop_fraction_threshold)
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

# -----------------------------------------------------------------------------
# Config (matches main.py CONFIG; edit here to test alternatives)
# -----------------------------------------------------------------------------
TS_COLS = ['lnf_code', 'yr', 'poly_id']
GROUP_COLS = ['poly_id', 'x', 'y', 'time', 'yr',
              'sampled_x', 'sampled_y', 'lnf_code', 'is_sample_pixel']

CIRRUS_THRESH = 500
MAX_MISSING_FRAC = 0.05
DROP_FRACTION_THRESHOLD = 0.7

SAMPLES_PATH = 'samples.pkl'
LNF_LABELS_PATH = os.path.expanduser(
    '~/mnt/eo-nas1/data/landuse/documentation/LNF_code_classification_20260217.xlsx'
)

OUT_DIR = 'calibration_analysis'
os.makedirs(OUT_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def dedupe(df):
    """Average duplicate (pixel, date) rows arising from overlapping S2 granules."""
    return df.groupby(GROUP_COLS, as_index=False).mean(numeric_only=True)


def plot_monthly_mean_compare(df_pre, df_post, value_col, group_col,
                              ylabel, title, save_path):
    """Side-by-side monthly mean of value_col per group_col, pre and post cleaning.

    Both inputs should be on the SAME field set so the comparison is fair.
    """
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
    """Recompute the per-row mask categories from clean_timeseries_field, but
    label each masked row with its dominant cause. Used for stacked-bar plot.

    Categories follow the OR-precedence of the original function:
      missing -> cloud -> shadow -> snow -> cirrus.
    A row may match multiple masks, but we attribute it to the first hit so the
    bars sum to the true dropped count.
    """
    df = df.copy()
    band_cols = [c for c in df.columns if c.startswith('s2_B')]

    if band_cols:
        missing = df[band_cols].isna().all(axis=1) | \
                  (df[band_cols] == 65535).all(axis=1)
        cloud = (df['s2_mask'] == 1) | (df['s2_SCL'].isin([8, 9, 10]))
        shadow = (df['s2_mask'] == 2) | (df['s2_SCL'] == 3)
        snow = (df['s2_mask'] == 3) | (df['s2_SCL'] == 11)
        cirrus = (df['s2_SCL'] == 10) & (df['s2_B02'] > cirrus_thresh)
    else:
        # Pre-computed FC case — only "missing" is detectable
        fc_cols = [c for c in ['pv', 'npv', 'soil'] if c in df.columns]
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
        # Per-row + per-date cleaning at this mmf
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


# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
df_samples = pd.read_pickle('samples_data_pred.pkl')
print(f'Loaded {len(df_samples):,} raw rows, '
      f'{df_samples.groupby(TS_COLS).ngroups:,} field-years.')

# -----------------------------------------------------------------------------
# 0a. Sample overview — spatial distribution
#     Where in CH did sampling place the points? Run once; skip if file missing.
# -----------------------------------------------------------------------------
if os.path.exists(SAMPLES_PATH):
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
    print('Saved: diag_samples_map.png')

    # -------------------------------------------------------------------------
    # 0b. Sample overview — counts per crop, stacked by year
    # -------------------------------------------------------------------------
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
    print('Saved: diag_samples_per_crop.png')

    del df_loc, gdf, gdf_web
else:
    print(f'(Skipping sample overview: {SAMPLES_PATH} not found)')

# -----------------------------------------------------------------------------
# 1. Mask-category breakdown (BEFORE applying the per-date filter)
#    Tells us what each mask is doing month-by-month.
# -----------------------------------------------------------------------------
df_tagged = per_row_mask_breakdown(df_samples, CIRRUS_THRESH)
df_tagged['month'] = pd.to_datetime(df_tagged['time']).dt.month

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
print('Saved: diag_mask_breakdown_by_month.png')

# -----------------------------------------------------------------------------
# 2. Run the actual cleaning pipeline at the configured thresholds
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

keys_kept = drop_stats[drop_stats['drop_fraction'] <= DROP_FRACTION_THRESHOLD][TS_COLS]
df_filtered = df_clean.merge(keys_kept, on=TS_COLS, how='inner')
df_raw_matched = df_samples.merge(keys_kept, on=TS_COLS, how='inner')

# Dedupe ALL THREE consistently so downstream counts are comparable.
df_samples_d = dedupe(df_samples)
df_clean_d = dedupe(df_clean)
df_filtered_d = dedupe(df_filtered)
df_raw_matched_d = dedupe(df_raw_matched)

print(f'Fields retained by drop_fraction_threshold={DROP_FRACTION_THRESHOLD}: '
      f'{len(keys_kept)}/{len(drop_stats)}')

# -----------------------------------------------------------------------------
# 3. Drop-fraction histogram — judges drop_fraction_threshold directly
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
print('Saved: diag_drop_fraction_hist.png')

# -----------------------------------------------------------------------------
# 4. Monthly mean PV — pre vs post cleaning (matched field set, deduped)
# -----------------------------------------------------------------------------
plot_monthly_mean_compare(
    df_raw_matched_d, df_filtered_d,
    value_col='pv', group_col='lnf_code',
    ylabel='Monthly mean PV',
    title='PV seasonal pattern before vs after cleaning (same field set)',
    save_path=f'{OUT_DIR}/diag_pv_monthly_pre_post.png',
)

# -----------------------------------------------------------------------------
# 5. Pipeline funnel — fields and median obs at each stage
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
print('Saved: diag_pipeline_funnel.png')

# -----------------------------------------------------------------------------
# 6. Obs-per-field-year distribution (final stage)
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
print('Saved: diag_obs_coverage.png')

# -----------------------------------------------------------------------------
# 7. Per-crop median coverage — surfaces under-sampled crops
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
print('Saved: diag_coverage_by_crop.png')

# -----------------------------------------------------------------------------
# 8. Threshold sweep
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

# Heatmap-style visualization of the sweep
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
    print(f'Saved: diag_sweep_{metric}.png')

print('\nDone.')
