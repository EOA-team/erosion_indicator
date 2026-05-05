"""C-factor calibration via Sentinel-2 fractional cover and rainfall erosivity.

Calibrates a single global parameter β in the soil loss ratio model
    SLR(t) = exp(-β · FC(t))
so that EI-weighted annual C-factors derived from Sentinel-2 fractional cover
time series reproduce the per-crop tabulated C-factors used in the existing
Swiss erosion risk pipeline (Prasuhn, Hutchings, Gilgen 2023; Agroscope
Science 158).

Granularity of the input data
-----------------------------
The upstream sampling step (``sample_FC.py``) retains exactly **one S2 pixel
per parcel per year**: a single representative pixel inside each sampled LNF
parcel, gap-filled in time using the Gaussian-process model trained on all
clean pixels of that parcel. The (lnf_code, yr, poly_id) triple therefore
identifies one *pixel time series*, not a within-parcel pixel collection.
Throughout this module "pixel" means that single sampled pixel, and
``poly_id`` acts as its identifier (it is the parcel from which the pixel
was drawn).

The calibration follows the strategy of Matthews et al. (2023, ISWCR):
fit one β globally across all sampled crops by minimising the mean absolute
difference between predicted and reference C-factors at the *crop* level,
not at the pixel level. This keeps the calibration target consistent with
the granularity of the reference table (one tabulated C per crop) while
letting the operational product retain pixel-level spatial and temporal
variation through FC and EI.

Pipeline
--------
1. **Load FC time series.** Gapfilled Sentinel-2 fractional cover (PV+NPV
   scaled to 0–1) per (lnf_code, year, sampled-pixel, time) from the sampling
   step.

2. **Build the reference table.** Read `C_Faktoren.csv` and use the per-crop
   `Total` column as `C_ref`. Bridge LNF codes to crop names via the
   `Crop_DE` column of the LNF classification spreadsheet, with a normalised-
   form fallback for punctuation differences and an optional manual-overrides
   CSV for residual mismatches. Filter to crops actually sampled.

3. **Join climatological EI.** Snap each FC pixel to the 100 m EI grid and
   merge EI onto each FC observation by (x_snap, y_snap, day-of-year). EI
   is keyed by DOY so the same long-term-average EI is reused across years —
   appropriate for matching a reference table built against climatological EI.

4. **Calibrate β.** For each candidate β:
     a. compute the EI-weighted SLR per (crop, year, pixel):
            C_pixel(β) = Σₜ exp(-β · fc_total) · EI / Σₜ EI
     b. average per-pixel C-factors up to one value per crop;
     c. compute |C_predicted_crop − C_ref_crop| and take the mean across
        crops.
   `scipy.optimize.minimize_scalar` finds the β minimising that scalar.

5. **Save outputs.**
     - `calibration_results.csv`         — per-crop diagnostics at β_opt
     - `calibration_results_per_pixel.csv` — per-pixel C-factors at β_opt
       (one row per sampled pixel, i.e. per (lnf_code, yr, poly_id))
     - `calibration_scatter.png`         — predicted vs reference per crop
     - `beta_sensitivity.png`            — per-crop and global MAE vs β
       (Matthews et al. Fig. 2 style)

Notes
-----
- Matthews et al. reported β of ~0.04. Expect β_opt ≈ 0.03–0.1;
  If it lands at the search bounds, FC scaling or reference loading needs
  checking.

- Tillage and Tal/Berg are deliberately not split: per-pixel tillage isn't
  known operationally, so the reference is collapsed to the per-crop `Total`
  value and the calibrated method is applied uniformly to every arable
  pixel regardless of management or elevation.

- Crops whose tabulated C-factor is the table-wide default (0.1 or 0.004
  fallback values used where no measurement exists) will fit either
  trivially or arbitrarily and do not provide informative residuals — see
  the per-crop diagnostic plot to identify them.

Entry point: `run_calibration(config)` — see DEFAULT_CONFIG at the bottom
for the expected keys.
"""

import os
import numpy as np
import pandas as pd
import re as _re
import matplotlib.pyplot as plt
import pyarrow.dataset as ds
import pyarrow.compute as pc
from scipy.optimize import minimize_scalar
 
 
# ---------------------------------------------------------------------------
# EI loading and joining
# ---------------------------------------------------------------------------
 
def get_ei_grid_offset(ei_path: str, resolution: int = 100) -> tuple[float, float]:
    """Read one row from the EI parquet to determine the grid origin offset.

    The EI grid may not be anchored at multiples of `resolution` — e.g. cells
    could be at 250, 350, 450 … rather than 200, 300, 400 …  Reading the actual
    coordinates from the file avoids hard-coding an assumed origin.
    """
    sample = (
        ds.dataset(ei_path, format='parquet')
        .to_table(columns=['x', 'y'])
        .slice(0, 1)
        .to_pandas()
    )
    x_off = float(sample['x'].iloc[0]) % resolution
    y_off = float(sample['y'].iloc[0]) % resolution
    return x_off, y_off


def snap_to_ei_grid(x: np.ndarray, y: np.ndarray,
                    x_off: float, y_off: float,
                    resolution: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Snap FC pixel coordinates to the nearest EI grid cell centre.

    Uses the grid offset derived from the file so snapping is consistent with
    the actual EI grid regardless of its origin.
    """
    x_snap = np.round((x - x_off) / resolution) * resolution + x_off
    y_snap = np.round((y - y_off) / resolution) * resolution + y_off
    return x_snap, y_snap
 
 
def load_ei_for_pixels(ei_path: str, x_snapped: np.ndarray, y_snapped: np.ndarray) -> pd.DataFrame:
    """Load EI rows only for the requested (x, y) cells using pyarrow column filters.
 
    Returns DataFrame with columns [x, y, doy, ei].
    The full EI parquet is 13–33 GB; this loads only the ~n_pixels × 365 rows needed.
    """
    x_vals = x_snapped.tolist()
    y_vals = y_snapped.tolist()
    filt = pc.field('x').isin(x_vals) & pc.field('y').isin(y_vals)
    table = (
        ds.dataset(ei_path, format='parquet')
        .to_table(columns=['x', 'y', 'doy', 'predicted_EI_daily_avg'], filter=filt)
    )
    df = table.to_pandas().rename(columns={'predicted_EI_daily_avg': 'ei'})
    return df
 
 
def join_ei_to_fc(df_fc: pd.DataFrame, df_ei: pd.DataFrame,
                  x_off: float, y_off: float, resolution: int = 100) -> pd.DataFrame:
    """Add per-pixel EI value to each FC observation by matching on (x_snap, y_snap, doy).

    Modifies a copy of df_fc, adding columns: x_snap, y_snap, doy, ei.
    """
    df = df_fc.copy()
    df['x_snap'], df['y_snap'] = snap_to_ei_grid(df['x'].values, df['y'].values, x_off, y_off, resolution)
    df['doy'] = pd.to_datetime(df['time']).dt.dayofyear
 
    df_ei_renamed = df_ei.rename(columns={'x': 'x_snap', 'y': 'y_snap'})
    df = df.merge(df_ei_renamed, on=['x_snap', 'y_snap', 'doy'], how='left')
 
    n_missing = df['ei'].isna().sum()
    if n_missing > 0:
        print(f"Warning: {n_missing} FC rows ({n_missing/len(df):.1%}) had no EI match — they will be excluded from C-factor computation.")
    return df
 
 
# ---------------------------------------------------------------------------
# Reference C-factor loading
# ---------------------------------------------------------------------------
 
def _normalise_crop_name(s: str) -> str:
    """Normalise a German crop name for fuzzy matching.
 
    Strips brackets/punctuation (keeping their content), lowercases, and
    collapses whitespace. Used as a fallback when ``Crop_DE`` and
    ``Kultur Kategorien 2020`` differ only in punctuation (e.g. comma vs
    parenthesis around a sub-clause).
    """
    import re
    s = str(s).lower()
    s = re.sub(r'[(){}\[\],;:.\-/]', ' ', s)  # strip brackets/punct, keep content
    s = re.sub(r'\s+', ' ', s).strip()
    return s
 
 
def load_reference_cfactors(c_factor_table_path: str,
                            lnf_classification_path: str,
                            lnf_codes: list[int],
                            manual_overrides_path: str | None = None,
                            area_years: list[int] | None = None) -> pd.DataFrame:
    """Load the per-crop reference C-factor (`Total` column) for the sampled LNF codes.
 
    The bridge between LNF codes and crop names comes directly from the
    ``label_sheet`` of the LNF classification spreadsheet (``Crop_DE`` column).
    For most arable crops this matches the ``Kultur Kategorien 2020`` column of
    ``C_Faktoren.csv`` exactly; a normalised fallback handles minor punctuation
    differences (e.g. comma vs parenthesis around a sub-clause).
 
    Residual mismatches (e.g. when one source uses a more qualified name than
    the other) can be resolved with an optional ``manual_overrides`` CSV with
    columns ``lnf_code,crop_name`` — these override the auto-resolution and
    point a specific LNF code at a specific row of ``C_Faktoren.csv``.
 
    Parameters
    ----------
    c_factor_table_path
        Path to ``C_Faktoren.csv`` (semicolon-separated, latin-1 encoded).
    lnf_classification_path
        Path to ``LNF_code_classification_*.xlsx``. Must contain a sheet named
        ``label_sheet`` with columns ``LNF_code`` and ``Crop_DE``.
    lnf_codes
        LNF codes that were sampled — only these crops are kept.
    manual_overrides_path
        Optional CSV with columns ``lnf_code,crop_name`` to manually map any
        residual mismatches. ``crop_name`` must exactly match an entry in
        ``Kultur Kategorien 2020`` of the C-factor table.
 
    Returns
    -------
    DataFrame with columns ``lnf_code``, ``crop_name``, ``C_ref``.
    """
    if area_years is None:
        area_years = []

    # --- C-factor table ---
    df_c = pd.read_csv(c_factor_table_path, sep=';', encoding='latin-1')
    df_c = df_c.rename(columns={'Kultur Kategorien 2020': 'crop_name', 'Total': 'C_ref'})
    df_c = df_c[['crop_name', 'C_ref']].dropna(subset=['crop_name'])
    df_c['C_ref'] = pd.to_numeric(df_c['C_ref'], errors='coerce')
    df_c = df_c.dropna(subset=['C_ref'])
    df_c['crop_name_norm'] = df_c['crop_name'].map(_normalise_crop_name)
    # Drop duplicate normalised forms — they make the fuzzy merge ambiguous.
    # Keeping the first occurrence is conservative; ambiguous cases should be
    # resolved via manual_overrides_path anyway.
    df_c_unique_norm = df_c.drop_duplicates(subset='crop_name_norm', keep='first')
 
    # --- LNF classification ---
    df_lnf = pd.read_excel(lnf_classification_path, sheet_name='label_sheet')
    # Per-year area columns are named like '2021_Area_m22', '2022_Area_m23'
    # (units m², the trailing digits are spreadsheet quirks). Pick the columns
    # whose year prefix is in `area_years` and average them, converting to ha.
    year_cols = {}
    for col in df_lnf.columns:
        m = _re.match(r'^(\d{4})_Area_m', str(col))
        if m:
            year_cols[int(m.group(1))] = col
    selected = [year_cols[y] for y in area_years if y in year_cols]
    has_area = bool(selected)
    if has_area:
        # Mean across the selected years (m²), then m² → ha
        df_lnf['area_ha'] = df_lnf[selected].mean(axis=1) / 10_000.0
        missing_years = [y for y in area_years if y not in year_cols]
        if missing_years:
            print(f"Warning: requested area_years {missing_years} not present in "
                  f"LNF spreadsheet; using {sorted(year_cols.keys() & set(area_years))} instead.")
        print(f"Area weights computed as mean over years: "
              f"{sorted(set(area_years) & set(year_cols.keys()))}")
    else:
        print(f"Warning: no per-year area columns found for years {area_years}; "
              "area weighting will be unavailable.")

    keep = ['LNF_code', 'Crop_DE']
    if has_area:
        keep.append('area_ha')
    df_lnf = df_lnf[keep].rename(columns={
        'LNF_code': 'lnf_code', 'Crop_DE': 'crop_name'
    })
    df_lnf = df_lnf[df_lnf['lnf_code'].isin(lnf_codes)].dropna(subset=['crop_name'])
    df_lnf['crop_name_norm'] = df_lnf['crop_name'].map(_normalise_crop_name)
 
    # --- Step 1: exact match on crop_name ---
    exact = df_lnf.merge(df_c[['crop_name', 'C_ref']], on='crop_name', how='left')
 
    # --- Step 2: normalised-form fallback for unresolved rows ---
    needs_fallback_mask = exact['C_ref'].isna()
    if needs_fallback_mask.any():
        fb_rows = exact.loc[needs_fallback_mask, ['lnf_code', 'crop_name', 'crop_name_norm']]
        fb = fb_rows.merge(df_c_unique_norm[['crop_name_norm', 'C_ref']],
                           on='crop_name_norm', how='left')
        resolved = fb[fb['C_ref'].notna()]
        if len(resolved):
            print(f"Resolved {len(resolved)} LNF codes via normalised crop-name match:")
            print(resolved[['lnf_code', 'crop_name']].to_string(index=False))
        exact.loc[needs_fallback_mask, 'C_ref'] = fb['C_ref'].values
 
    # --- Step 3: manual overrides for any residual mismatches ---
    if manual_overrides_path and os.path.exists(manual_overrides_path):
        df_ov = pd.read_csv(manual_overrides_path)
        df_ov = df_ov.merge(df_c[['crop_name', 'C_ref']], on='crop_name', how='left')
        bad = df_ov[df_ov['C_ref'].isna()]
        if len(bad):
            print(f"Warning: {len(bad)} manual override rows do not match any "
                  "entry in C_Faktoren.csv — check spelling exactly:")
            print(bad[['lnf_code', 'crop_name']].to_string(index=False))
        df_ov = df_ov.dropna(subset=['C_ref'])
        # Apply overrides (replace existing C_ref values for these LNF codes)
        ov_map_c = dict(zip(df_ov['lnf_code'], df_ov['C_ref']))
        ov_map_name = dict(zip(df_ov['lnf_code'], df_ov['crop_name']))
        n_applied = exact['lnf_code'].isin(ov_map_c).sum()
        if n_applied:
            exact.loc[exact['lnf_code'].isin(ov_map_c), 'C_ref'] = (
                exact.loc[exact['lnf_code'].isin(ov_map_c), 'lnf_code'].map(ov_map_c)
            )
            exact.loc[exact['lnf_code'].isin(ov_map_name), 'crop_name'] = (
                exact.loc[exact['lnf_code'].isin(ov_map_name), 'lnf_code'].map(ov_map_name)
            )
            print(f"Applied {n_applied} manual override(s) from {manual_overrides_path}")
 
    # --- Final: drop unresolved with a clear warning ---
    missing = exact[exact['C_ref'].isna()]
    if len(missing):
        print(f"Warning: {len(missing)} sampled LNF codes have no C-factor entry "
              "and will be dropped from calibration. Add them to a manual "
              "overrides CSV (columns: lnf_code,crop_name) if needed:")
        print(missing[['lnf_code', 'crop_name']].to_string(index=False))
        exact = exact.dropna(subset=['C_ref'])
 
    print(f"Reference C-factors loaded for {len(exact)} crops.")
    out_cols = ['lnf_code', 'crop_name', 'C_ref']
    if has_area:
        out_cols.append('area_ha')
    return exact[out_cols].reset_index(drop=True)
 
 
# ---------------------------------------------------------------------------
# C-factor computation (vectorised)
# ---------------------------------------------------------------------------
 
def compute_cfactors_per_pixel(df: pd.DataFrame, beta: float, ts_cols: list[str],
                               fc_col: str = 'fc_total', ei_col: str = 'ei') -> pd.DataFrame:
    """Compute the EI-weighted SLR per (crop, year, pixel) group, vectorised.

    C(group) = sum(exp(-beta * fc_total) * EI) / sum(EI)

    Each ``ts_cols`` group identifies one *sampled-pixel time series* (the
    upstream sampling step retains exactly one pixel per parcel per year, so
    ``poly_id`` here is effectively a pixel identifier). The groupby therefore
    aggregates over **time** within a single pixel, not across pixels within
    a parcel.

    Vectorised over the entire dataframe in two groupby aggregations rather than
    iterating through groups in Python — substantially faster when the optimiser
    re-evaluates this for every β candidate.
    """
    valid = df[ts_cols + [fc_col, ei_col]].dropna(subset=[fc_col, ei_col])
    slr = np.exp(-beta * valid[fc_col].values)
    weighted = slr * valid[ei_col].values

    grouped = (
        valid.assign(_num=weighted, _den=valid[ei_col].values)
             .groupby(ts_cols, as_index=False)[['_num', '_den']]
             .sum()
    )
    grouped['C_predicted'] = np.where(grouped['_den'] > 0,
                                      grouped['_num'] / grouped['_den'],
                                      np.nan)
    return grouped[ts_cols + ['C_predicted']]
 
 
def aggregate_to_crop(df_pred: pd.DataFrame, crop_col: str = 'lnf_code') -> pd.DataFrame:
    """Average per-pixel C-factors up to one value per crop (Matthews et al. style)."""
    return (df_pred.dropna(subset=['C_predicted'])
                   .groupby(crop_col, as_index=False)['C_predicted']
                   .mean())
 
 
# ---------------------------------------------------------------------------
# Beta calibration  (Matthews et al. 2023 strategy)
# ---------------------------------------------------------------------------
 
def calibrate_beta(df: pd.DataFrame, df_ref: pd.DataFrame, ts_cols: list[str],
                   crop_col: str = 'lnf_code',
                   fc_col: str = 'fc_total', ei_col: str = 'ei',
                   beta_bounds: tuple[float, float] = (1e-4, 0.1),
                   area_weight: bool = True,
                   ) -> tuple[float, pd.DataFrame]:
    """Find β minimising the (area-weighted) mean absolute difference of
    crop-level C-factors.

    Following Matthews et al. (2023) but with optional area weighting:
      1) compute C per (crop, year, pixel) for the candidate β,
      2) average per-pixel C up to one value per crop,
      3) compute |C_predicted_crop − C_ref_crop| per crop,
      4) take the mean across crops — equal-weighted (Matthews\' default) or
         weighted by Swiss arable area (`Avg_Area_ha`) so cereals dominate
         the fit and minor crops don\'t skew β,
      5) minimise that scalar w.r.t. β.

    Area weighting is only applied if ``area_weight=True`` AND ``df_ref``
    contains an ``area_ha`` column. Otherwise the loss is the unweighted
    mean (original Matthews behaviour).

    Returns
    -------
    beta_opt : float
    df_crop  : DataFrame with one row per crop containing C_ref, C_predicted,
               residual, abs_residual, and (if used) area_ha and weight.
    """
    use_area = area_weight and 'area_ha' in df_ref.columns
    if use_area:
        # Normalise weights so they sum to 1 across the calibration set —
        # keeps the loss value comparable across runs with different crop
        # sets and makes "MAE" interpretable as a weighted mean of |residual|.
        w = df_ref['area_ha'].fillna(0.0).clip(lower=0.0).values
        if w.sum() <= 0:
            print("Warning: all area weights are zero/missing — falling back to "
                  "unweighted MAE.")
            use_area = False
        else:
            df_ref = df_ref.copy()
            df_ref['weight'] = w / w.sum()
            top = (df_ref.nlargest(3, 'weight')[['crop_name', 'weight']]
                          .to_dict('records'))
            top_str = ', '.join(f"{r['crop_name'][:25]} ({r['weight']:.0%})"
                                for r in top)
            print(f"Area-weighted MAE active — top contributors: {top_str}")
    else:
        reason = ("area_weight=False" if not area_weight
                  else "area_ha missing from df_ref")
        print(f"Equal-weighted MAE active ({reason}; Matthews et al. 2023 default)")

    def objective(beta: float) -> float:
        df_pred = compute_cfactors_per_pixel(df, beta, ts_cols, fc_col, ei_col)
        df_crop_pred = aggregate_to_crop(df_pred, crop_col)
        merged = df_ref.merge(df_crop_pred, on=crop_col, how='inner')
        if len(merged) == 0:
            return 1e6
        abs_res = np.abs(merged['C_predicted'] - merged['C_ref']).values
        if use_area:
            # Re-normalise weights of crops that actually merged — keeps the
            # loss a proper weighted mean if a crop has no FC samples for
            # this β-eval (shouldn\'t normally happen but defensive).
            w = merged['weight'].values
            w = w / w.sum() if w.sum() > 0 else w
            return float((abs_res * w).sum())
        return float(abs_res.mean())

    result = minimize_scalar(objective, bounds=beta_bounds, method='bounded')
    beta_opt = float(result.x)
    mae_opt = float(result.fun)
    label = "area-weighted MAE" if use_area else "MAE"
    print(f"Optimal β: {beta_opt:.5f}   {label} across crops: {mae_opt:.4f}")

    # Build the diagnostic crop-level table at β_opt
    df_pred = compute_cfactors_per_pixel(df, beta_opt, ts_cols, fc_col, ei_col)
    df_crop_pred = aggregate_to_crop(df_pred, crop_col)
    df_crop = df_ref.merge(df_crop_pred, on=crop_col, how='left')
    df_crop['residual'] = df_crop['C_predicted'] - df_crop['C_ref']
    df_crop['abs_residual'] = df_crop['residual'].abs()
    return beta_opt, df_crop
 
 
# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------
 
def plot_calibration_per_crop(df_crop: pd.DataFrame, beta_opt: float,
                              save_path: str, area_weight: bool = True) -> None:
    """One point per crop: predicted (mean of sampled pixels) vs reference (`Total`).

    Point sizes are proportional to Swiss arable area when ``area_ha`` is
    present and ``area_weight=True`` — so a glance at the plot tells you
    which crops dominate the (area-weighted) loss. The title reports both
    unweighted and area-weighted MAE so you can see the effect of weighting.
    """
    valid = df_crop.dropna(subset=['C_ref', 'C_predicted']).copy()
    use_area = area_weight and 'area_ha' in valid.columns and valid['area_ha'].notna().any()

    # Stats: always report unweighted MAE and bias; add area-weighted MAE if available.
    mae_unw = valid['abs_residual'].mean()
    bias = valid['residual'].mean()
    if use_area:
        w = valid['area_ha'].fillna(0.0).clip(lower=0.0).values
        w = w / w.sum() if w.sum() > 0 else None
        mae_aw = float((valid['abs_residual'].values * w).sum()) if w is not None else float('nan')
    else:
        mae_aw = float('nan')

    c_max = max(valid['C_ref'].max(), valid['C_predicted'].max()) * 1.1
    c_range = [0, c_max]

    # Point size: 30..400 scaled by sqrt(area) so a 100x area difference is
    # visible but doesn\'t make the largest crop dwarf everything.
    if use_area:
        a = valid['area_ha'].fillna(0.0).clip(lower=0.0).values
        if a.max() > 0:
            sizes = 30 + 370 * np.sqrt(a / a.max())
        else:
            sizes = np.full(len(valid), 80.0)
    else:
        sizes = np.full(len(valid), 80.0)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(valid['C_ref'], valid['C_predicted'],
               s=sizes, alpha=0.7, edgecolors='k', linewidths=0.5)
    for _, row in valid.iterrows():
        label = row['crop_name'] if 'crop_name' in row else str(row['lnf_code'])
        ax.annotate(str(label)[:20], (row['C_ref'], row['C_predicted']),
                    fontsize=7, alpha=0.7,
                    xytext=(4, 4), textcoords='offset points')
    ax.plot(c_range, c_range, 'k--', label='1:1', alpha=0.6)
    ax.set_xlim(c_range)
    ax.set_ylim(c_range)
    ax.set_xlabel('Reference C-factor (Total per crop)')
    ax.set_ylabel('Predicted C-factor (mean of sampled pixels)')
    title = (f'C-factor calibration per crop  '
             f'(β = {beta_opt:.5f}, MAE = {mae_unw:.4f}, bias = {bias:+.4f}')
    if use_area:
        title += f', area-w. MAE = {mae_aw:.4f})'
        ax.set_xlabel('Reference C-factor (Total per crop)  '
                      '— marker size ∝ √(Swiss arable area)')
    else:
        title += ')'
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Per-crop calibration scatter saved to {save_path}")
 
 
def plot_beta_sensitivity(df: pd.DataFrame, df_ref: pd.DataFrame, ts_cols: list[str],
                          beta_range: np.ndarray, beta_opt: float, save_path: str,
                          crop_col: str = 'lnf_code',
                          area_weight: bool = True) -> None:
    """Plot mean absolute crop-level error vs β, plus per-crop curves.

    Mirrors Fig. 2 of Matthews et al. (2023): per-crop curves expose the
    spread of optimal β values across crops, and the global "all crops"
    curve is the loss being minimised. If ``area_weight=True`` and
    ``df_ref`` has an ``area_ha`` column, the global curve is the
    area-weighted MAE — matching the calibration objective.
    """
    use_area = area_weight and 'area_ha' in df_ref.columns
    if use_area:
        w = df_ref['area_ha'].fillna(0.0).clip(lower=0.0).values
        if w.sum() > 0:
            df_ref = df_ref.copy()
            df_ref['weight'] = w / w.sum()
        else:
            use_area = False

    per_crop_records = []
    overall_mae = []

    for beta in beta_range:
        df_pred = compute_cfactors_per_pixel(df, float(beta), ts_cols)
        df_crop_pred = aggregate_to_crop(df_pred, crop_col)
        merged = df_ref.merge(df_crop_pred, on=crop_col, how='left')
        merged['abs_diff'] = (merged['C_predicted'] - merged['C_ref']).abs()
        for _, row in merged.iterrows():
            per_crop_records.append({
                'beta': float(beta),
                crop_col: row[crop_col],
                'crop_name': row.get('crop_name', str(row[crop_col])),
                'abs_diff': row['abs_diff'],
            })
        if use_area:
            valid = merged.dropna(subset=['abs_diff'])
            w_v = valid['weight'].values
            w_v = w_v / w_v.sum() if w_v.sum() > 0 else w_v
            overall_mae.append((valid['abs_diff'].values * w_v).sum())
        else:
            overall_mae.append(merged['abs_diff'].mean())

    df_curves = pd.DataFrame(per_crop_records)

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap('tab20')
    for i, (crop, sub) in enumerate(df_curves.groupby('crop_name')):
        ax.plot(sub['beta'], sub['abs_diff'], color=cmap(i % 20), alpha=0.6,
                lw=1, label=str(crop)[:25])
    global_label = ('All crops (area-weighted mean)'
                    if use_area else 'All crops (mean)')
    ax.plot(beta_range, overall_mae, color='black', lw=2.5,
            label=global_label, linestyle='--')
    ax.axvline(beta_opt, color='red', linestyle=':', label=f'β_opt = {beta_opt:.5f}')
    ax.set_xlabel('β')
    ax.set_ylabel('Mean absolute C-factor difference')
    title = 'β sensitivity per crop (Matthews et al. Fig. 2 style)'
    if use_area:
        title += ' — global curve area-weighted'
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"β sensitivity plot saved to {save_path}")
 
 
# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
 
def run_calibration(config: dict) -> None:
    """Run the full C-factor calibration pipeline."""
 
    fc_path                 = config['gapfilled_fc_path']
    ei_path                 = os.path.expanduser(config['ei_path'])
    c_factor_table_path     = os.path.expanduser(config['c_factor_table_path'])
    lnf_classification_path = os.path.expanduser(config['lnf_classification_path'])
    manual_overrides_path   = config.get('manual_overrides_path')
    if manual_overrides_path:
        manual_overrides_path = os.path.expanduser(manual_overrides_path)
    results_path            = config['calibration_results_path']
    ts_cols                 = config.get('ts_cols', ['lnf_code', 'yr', 'poly_id'])
    crop_col                = config.get('crop_col', 'lnf_code')
    beta_bounds             = config.get('beta_bounds', (1e-4, 0.1))
    exclude_lnf_codes       = config.get('exclude_calibration_lnf_codes', []) or []
    area_weight_loss        = config.get('area_weight_loss', True)
    area_years              = config.get('area_years', []) or None
 
    # Load gapfilled FC timeseries
    print("Loading gapfilled FC timeseries...")
    df_fc = pd.read_parquet(fc_path)
 
    # Load per-crop reference C-factors restricted to sampled LNF codes
    print("Loading reference C-factors...")
    sampled_lnf_codes = sorted(df_fc[crop_col].unique().tolist())
    df_ref = load_reference_cfactors(c_factor_table_path,
                                     lnf_classification_path,
                                     sampled_lnf_codes,
                                     manual_overrides_path=manual_overrides_path,
                                     area_years=area_years)
    print('Reference C-factors:\n', df_ref)
 
    # Drop crops excluded from calibration (e.g. permanent grasslands like
    # Kunstwiesen / Extensiv genutzte Wiesen, where exp(-β·FC) is the wrong
    # functional form). These crops still appear in the per-pixel operational
    # output below — they're only removed from the loss function.
    if exclude_lnf_codes:
        before = len(df_ref)
        excluded = df_ref[df_ref[crop_col].isin(exclude_lnf_codes)]
        df_ref = df_ref[~df_ref[crop_col].isin(exclude_lnf_codes)].reset_index(drop=True)
        print(f"Excluded {before - len(df_ref)} crops from calibration "
              f"({len(df_ref)} remaining):")
        if len(excluded):
            print(excluded[[crop_col, 'crop_name']].to_string(index=False))
 
    # Make crop column dtypes match between df_fc and df_ref so the merge works
    df_ref[crop_col] = df_ref[crop_col].astype(df_fc[crop_col].dtype)
 
    # Load EI for sampled pixels only
    print("Snapping FC coordinates to EI grid and loading EI data...")
    x_off, y_off = get_ei_grid_offset(ei_path)
    print(f"EI grid offset: x={x_off}, y={y_off}")
    x_snap, y_snap = snap_to_ei_grid(df_fc['x'].values, df_fc['y'].values, x_off, y_off)
    x_unique = np.unique(x_snap)
    y_unique = np.unique(y_snap)
    df_ei = load_ei_for_pixels(ei_path, x_unique, y_unique)
    print(f"EI loaded: {len(df_ei)} rows for {df_ei[['x','y']].drop_duplicates().shape[0]} unique cells")

    # Join EI to FC timeseries
    print("Joining EI to FC timeseries...")
    df = join_ei_to_fc(df_fc, df_ei, x_off, y_off)
 
    # Calibrate beta (Matthews et al. strategy: crop-level mean absolute error)
    print("Calibrating β (crop-level MAE objective)...")
    beta_opt, df_crop = calibrate_beta(df, df_ref, ts_cols,
                                       crop_col=crop_col,
                                       beta_bounds=beta_bounds,
                                       area_weight=area_weight_loss)
 
    # Save crop-level calibration table
    df_crop.to_csv(results_path, index=False)
    print(f"Per-crop calibration results saved to {results_path}")
 
    # Save full per-pixel C-factors at β_opt (operational output).
    # One row per sampled pixel, i.e. per (lnf_code, yr, poly_id) — see the
    # module docstring for why "pixel" is the right granularity here.
    df_pixel_c = compute_cfactors_per_pixel(df, beta_opt, ts_cols)
    pixel_c_path = results_path.replace('.csv', '_per_pixel.csv')
    df_pixel_c.to_csv(pixel_c_path, index=False)
    print(f"Per-pixel C-factors at β_opt saved to {pixel_c_path}")
 
    # Diagnostic plots
    plot_calibration_per_crop(df_crop, beta_opt, 'calibration_scatter.png',
                              area_weight=area_weight_loss)
    beta_range = np.linspace(beta_bounds[0], beta_bounds[1], 60)
    plot_beta_sensitivity(df, df_ref, ts_cols, beta_range, beta_opt,
                          'beta_sensitivity.png', crop_col=crop_col,
                          area_weight=area_weight_loss)
 
 
# Expected config keys (defined and owned by main.py):
#   gapfilled_fc_path        — parquet of gapfilled FC time series (from sampling step)
#   ei_path                  — parquet of climatological daily EI on 100 m grid
#   c_factor_table_path      — C_Faktoren.csv (semicolon-separated, latin-1)
#   lnf_classification_path  — LNF_code_classification_*.xlsx
#   manual_overrides_path    — optional CSV (lnf_code, crop_name) for residual mismatches; None if unused
#   calibration_results_path — output CSV path; per-pixel results saved alongside with `_per_pixel` suffix
#   ts_cols                  — group-by columns for one C per group; default ['lnf_code', 'yr', 'poly_id'].
#                              poly_id is the parcel identifier; since the upstream sampling
#                              step retains exactly one pixel per parcel per year, each group
#                              corresponds to one sampled-pixel time series.
#   crop_col                 — crop identifier column; default 'lnf_code'
#   beta_bounds              — (min, max) for the 1-D β search; default (1e-4, 0.1)
#   exclude_calibration_lnf_codes — optional list of LNF codes to drop from the loss
#                              function (kept in per-pixel output); default []
#   area_weight_loss         — if True, weight the loss by Swiss arable area per crop
#                              so cereals dominate; if False, equal-weight crops as in
#                              Matthews et al. 2023; default True
#   area_years               — list of years averaged into the area weights (e.g.
#                              [2021, 2022, 2023, 2024] to match the FC sampling
#                              window); None or [] disables area weighting