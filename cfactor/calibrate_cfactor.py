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

  Optionally, when ``stratified_calibration=True`` is set in the config,
  the calibration matches against the stratified C-factor columns
  (``Tal_Pflug, Tal_Mulch, Tal_Direkt, Berg_Pflug, Berg_Mulch,
  Berg_Direkt``) instead of ``Total``. Each sampled pixel is assigned one
  stratum based on field altitude (Tal/Berg via ``swissALTI3D`` in
  ``tbl_nutzungsdaten``) and farm-year soil preparation (Pflug/Mulch/Direkt
  via ``reb_sb`` in ``tbl_ressourceneffizienzbeitrag``). Outputs are
  written with a ``_stratified`` suffix so the two modes don't clobber
  each other. See ``run_calibration_stratified`` for details.

- Crops whose tabulated C-factor is the table-wide default (0.1 or 0.004
  fallback values used where no measurement exists) will fit either
  trivially or arbitrarily and do not provide informative residuals — see
  the per-crop diagnostic plot to identify them.

Entry point: `run_calibration(config)` — see the CONFIG dict in main.py
for the expected keys.
"""

import os
import numpy as np
import pandas as pd
import re as _re
import json
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
# Stratified calibration
# ---------------------------------------------------------------------------
#
# These helpers add an alternative target for the same SLR = exp(-β·FC)
# model: instead of one C_ref per crop (`Total` column), each pixel is
# matched to a C_ref drawn from the stratified C-factor columns
# (`Tal_Pflug, Tal_Mulch, ..., Berg_Direkt`).
#
# Pixel-to-stratum assignment:
#   • Region (Tal/Berg) from field altitude `swissALTI3D` in
#     `tbl_nutzungsdaten`, joined by (Flaechen_ID, Jahr). Cutoff
#     `grenze_tal_berg` (default 600 m, matching the R pipeline).
#   • Tillage (Pflug/Mulch/Direkt) from `reb_sb` in
#     `tbl_ressourceneffizienzbeitrag`, joined by (betr_ID, Jahr). The
#     REB table is farm-year-keyed and does not carry a field id, so
#     within a farm-year with mixed `reb_sb` the per-field assignment
#     is genuinely unknown. The `tillage_assignment` config knob
#     controls how to resolve this:
#       'stochastic' — draw per pixel from empirical reb_sb frequencies
#                      of the farm-year (zero bias in expectation;
#                      noisy per-pixel)
#       'first_row'  — first row's reb_sb wins (matches the R-side
#                      `slice(1)` in 05-Dataprep.R)
#       'mode'       — most frequent reb_sb in the farm-year
#     For farm-years not present in REB, tillage defaults to
#     `standardansaatverfahren` ('Pflug').
#
# Loss form: one residual per non-empty (crop, region, tillage) stratum,
# averaged with weights = crop area × pixel_count_in_stratum /
# pixel_count_in_crop. This preserves each crop's contribution to the
# loss equal to its unstratified contribution, just split across strata.
 
 
_TILLAGE_MAP = {
    'Mulchsaat':    'Mulch',
    'Direktsaat':   'Direkt',
    'Streifensaat': 'Direkt',
}
 
 
def _normalise_tillage(s: pd.Series, default: str) -> pd.Series:
    """Map raw `reb_sb` values to {Pflug, Mulch, Direkt}.
 
    NaN / unknown values become `default` (the `standardansaatverfahren`,
    typically 'Pflug').
    """
    out = s.map(_TILLAGE_MAP)
    return out.fillna(default)
 
 
def load_altitude(nutzung_csv: str, cutoff_m: float = 600.0) -> pd.DataFrame:
    """Load (Flaechen_ID, Jahr, swissALTI3D) and classify into Tal/Berg.
 
    Reads `tbl_nutzungsdaten`, keeps the three relevant columns, drops
    rows missing altitude. The same `(Flaechen_ID, Jahr)` may appear
    multiple times in the source; we keep the first occurrence with a
    defined altitude.
 
    Returns
    -------
    DataFrame with columns [`uuid`, `yr`, `swissALTI3D`, `region`] where
    `uuid` matches the AGIS `Flaechen_ID` in the gapfilled parquet, and
    `region ∈ {Tal, Berg}` (`Tal` if `swissALTI3D <= cutoff_m`).
    """
    df = pd.read_csv(nutzung_csv, encoding='latin1', sep=';',
                     usecols=['Jahr', 'Flaechen_ID', 'swissALTI3D'])
    df = df.dropna(subset=['swissALTI3D'])
    df = df.drop_duplicates(subset=['Jahr', 'Flaechen_ID'], keep='first')
    df = df.rename(columns={'Flaechen_ID': 'uuid', 'Jahr': 'yr'})
    # Match the gapfilled-parquet dtypes: uuid as str, yr as str
    df['uuid'] = df['uuid'].astype(str)
    df['yr']   = df['yr'].astype(str)
    df['region'] = np.where(df['swissALTI3D'] <= cutoff_m, 'Tal', 'Berg')
    return df[['uuid', 'yr', 'swissALTI3D', 'region']]
 
 
def load_tillage_table(reb_csv: str, default_tillage: str = 'Pflug') -> pd.DataFrame:
    """Load REB table and return per-(betr_ID, yr) row-count by tillage class.
 
    Returns a long-format DataFrame with columns
    [`betr_ID`, `yr`, `tillage`, `n_rows`] holding the empirical
    frequencies of each `reb_sb` class within every farm-year. NaN
    `reb_sb` rows are mapped to `default_tillage`.
 
    The downstream `assign_tillage` function consumes this to apply the
    chosen assignment policy. Keeping the long table (rather than
    collapsing here) means the same input drives all three policies
    consistently — and is what `'stochastic'` needs to sample from.
    """
    df = pd.read_csv(reb_csv, encoding='utf-8', sep=';', na_values=['NA'])
    if 'betr_ID' not in df.columns or 'Jahr' not in df.columns or 'reb_sb' not in df.columns:
        raise KeyError(
            f"REB table at {reb_csv} missing expected columns "
            f"(betr_ID, Jahr, reb_sb). Got: {df.columns.tolist()}"
        )
    df['tillage'] = _normalise_tillage(df['reb_sb'], default=default_tillage)
    df = df.rename(columns={'Jahr': 'yr'})
    df['yr'] = df['yr'].astype(str)
    counts = (df.groupby(['betr_ID', 'yr', 'tillage'])
                .size()
                .rename('n_rows')
                .reset_index())
    return counts
 
 
def assign_tillage(df_fc: pd.DataFrame, df_till_long: pd.DataFrame,
                   method: str = 'stochastic',
                   default_tillage: str = 'Pflug',
                   seed: int = 42) -> pd.DataFrame:
    """Assign one tillage class per (poly_id, yr) sampled pixel.
 
    Operates at the (poly_id, yr) granularity, since each sampled-pixel
    time series corresponds to one parcel-year (see the module docstring
    on what "pixel" means here). All FC observations of the same time
    series get the same tillage.
 
    Pixels with no `betr_ID` (e.g. grass polys, random-sampling mode) or
    whose `(betr_ID, yr)` is not in the REB table get `default_tillage`.
 
    Parameters
    ----------
    df_fc          gapfilled FC frame, must contain columns
                   `poly_id`, `yr`, `betr_ID`.
    df_till_long   output of `load_tillage_table`.
    method         'stochastic' | 'first_row' | 'mode'
    default_tillage default class when no REB match.
    seed           RNG seed for `'stochastic'`.
 
    Returns
    -------
    DataFrame with columns [`poly_id`, `yr`, `tillage`,
    `tillage_n_choices`] — one row per pixel time series.
    `tillage_n_choices` is the number of distinct REB tillage classes
    seen for the matched (betr_ID, yr); a diagnostic for how arbitrary
    the choice was.
    """
    if method not in ('stochastic', 'first_row', 'mode'):
        raise ValueError(
            f"Unknown tillage_assignment='{method}'. "
            "Expected 'stochastic', 'first_row', or 'mode'."
        )
 
    # One row per sampled-pixel time series
    pixels = (df_fc[['poly_id', 'yr', 'betr_ID']]
              .drop_duplicates()
              .reset_index(drop=True))
 
    # Empirical tillage distribution per (betr_ID, yr)
    grp = df_till_long.groupby(['betr_ID', 'yr'])
    farm_choices = grp.agg(
        n_choices=('tillage', 'nunique'),
    ).reset_index()
 
    rng = np.random.default_rng(seed)
 
    def _resolve_one(betr_id, yr):
        """Return (tillage, n_choices) for one farm-year."""
        if pd.isna(betr_id):
            return default_tillage, 0
        sub = df_till_long[(df_till_long['betr_ID'] == betr_id)
                           & (df_till_long['yr'] == yr)]
        if len(sub) == 0:
            return default_tillage, 0
        if len(sub) == 1:
            return sub['tillage'].iloc[0], 1
        # Multiple distinct tillage classes
        if method == 'first_row':
            return sub['tillage'].iloc[0], len(sub)
        if method == 'mode':
            return (sub.sort_values('n_rows', ascending=False)['tillage'].iloc[0],
                    len(sub))
        # stochastic
        probs = sub['n_rows'].to_numpy(dtype=float)
        probs = probs / probs.sum()
        choice = rng.choice(sub['tillage'].to_numpy(), p=probs)
        return choice, len(sub)
 
    # Iterate over unique (betr_ID, yr) tuples to keep this cheap; resolved
    # values are then merged back onto pixels. With 'stochastic' we still
    # draw once per pixel, not once per farm-year — preserving the
    # within-farm-year diversity at the population level. So we iterate
    # over pixel rows directly when method=='stochastic'.
    if method == 'stochastic':
        rows = pixels.apply(
            lambda r: _resolve_one(r['betr_ID'], r['yr']), axis=1
        )
    else:
        # Resolve once per (betr_ID, yr) then merge — much faster
        uniq = pixels[['betr_ID', 'yr']].drop_duplicates().reset_index(drop=True)
        uniq_res = uniq.apply(
            lambda r: _resolve_one(r['betr_ID'], r['yr']), axis=1
        )
        uniq['tillage'] = [t[0] for t in uniq_res]
        uniq['tillage_n_choices'] = [t[1] for t in uniq_res]
        pixels = pixels.merge(uniq, on=['betr_ID', 'yr'], how='left')
        return pixels[['poly_id', 'yr', 'tillage', 'tillage_n_choices']]
 
    pixels['tillage']           = [t[0] for t in rows]
    pixels['tillage_n_choices'] = [t[1] for t in rows]
    return pixels[['poly_id', 'yr', 'tillage', 'tillage_n_choices']]
 
 
def assign_strata(df_fc: pd.DataFrame,
                  nutzung_csv: str, reb_csv: str,
                  cutoff_m: float = 600.0,
                  default_tillage: str = 'Pflug',
                  tillage_method: str = 'stochastic',
                  seed: int = 42) -> pd.DataFrame:
    """Join (region, tillage) onto every FC observation.
 
    Drops pixels missing altitude (no Tal/Berg assignment possible) with
    a warning. Pixels whose farm-year is not in REB get the default
    tillage.
 
    Parameters
    ----------
    df_fc          gapfilled FC frame with `uuid`, `betr_ID`, `yr`,
                   `poly_id`.
    nutzung_csv    path to tbl_nutzungsdaten.
    reb_csv        path to tbl_ressourceneffizienzbeitrag.
    cutoff_m       Tal/Berg threshold in metres.
    default_tillage default tillage class.
    tillage_method 'stochastic' | 'first_row' | 'mode'.
    seed           RNG seed for stochastic mode.
 
    Returns
    -------
    DataFrame: copy of df_fc augmented with `region`, `tillage`, and the
    diagnostic `tillage_n_choices`.
    """
    n0 = len(df_fc)
 
    # Altitude
    print("Loading altitude from tbl_nutzungsdaten ...")
    df_alt = load_altitude(nutzung_csv, cutoff_m=cutoff_m)
    # Align dtypes for the join
    df_fc = df_fc.copy()
    df_fc['uuid'] = df_fc['uuid'].astype(str)
    df_fc['yr']   = df_fc['yr'].astype(str)
    df_fc = df_fc.merge(df_alt[['uuid', 'yr', 'swissALTI3D', 'region']],
                        on=['uuid', 'yr'], how='left')
 
    # Drop pixels with no altitude — at the pixel-time-series granularity
    no_alt = df_fc['region'].isna()
    n_pix_no_alt = (df_fc.loc[no_alt, ['poly_id', 'yr']]
                         .drop_duplicates().shape[0])
    if no_alt.any():
        print(f"Warning: {n_pix_no_alt} pixel time series have no altitude "
              f"({no_alt.sum()}/{n0} FC rows, {no_alt.mean():.1%}) — dropped.")
        df_fc = df_fc[~no_alt].reset_index(drop=True)
 
    # Tillage
    print(f"Loading tillage from REB (assignment='{tillage_method}', "
          f"default='{default_tillage}') ...")
    df_till_long = load_tillage_table(reb_csv, default_tillage=default_tillage)
    df_till_pixel = assign_tillage(df_fc, df_till_long,
                                   method=tillage_method,
                                   default_tillage=default_tillage,
                                   seed=seed)
    df_fc = df_fc.merge(df_till_pixel, on=['poly_id', 'yr'], how='left')
 
    # Diagnostic: how many pixel-time-series have ambiguous tillage?
    pix = df_fc[['poly_id', 'yr', 'tillage', 'tillage_n_choices']].drop_duplicates()
    n_pix       = len(pix)
    n_no_reb    = (pix['tillage_n_choices'] == 0).sum()
    n_unambig   = (pix['tillage_n_choices'] == 1).sum()
    n_ambig     = (pix['tillage_n_choices'] >  1).sum()
    print(f"Tillage assignment: {n_pix} pixel time series — "
          f"{n_unambig} unambiguous, {n_ambig} ambiguous "
          f"(multi-class farm-year), {n_no_reb} no REB entry → defaulted "
          f"to '{default_tillage}'.")
 
    # Stratum diagnostic
    print("Pixel-time-series distribution across strata:")
    pix_strata = (df_fc[['poly_id', 'yr', 'region', 'tillage']]
                  .drop_duplicates()
                  .groupby(['region', 'tillage']).size().rename('n_pixels'))
    print(pix_strata.to_string())
    return df_fc
 
 
def load_reference_cfactors_stratified(c_factor_table_path: str,
                                       lnf_classification_path: str,
                                       lnf_codes: list[int],
                                       manual_overrides_path: str | None = None,
                                       area_years: list[int] | None = None) -> pd.DataFrame:
    """Long-format stratified C-factor reference.
 
    Reuses the same crop-name resolution as `load_reference_cfactors` but
    melts the six stratified columns
    (`Tal_Pflug, Tal_Mulch, Tal_Direkt, Berg_Pflug, Berg_Mulch,
    Berg_Direkt`) into one row per (lnf_code, region, tillage).
 
    Returns
    -------
    DataFrame with columns [`lnf_code`, `crop_name`, `region`,
    `tillage`, `C_ref`] plus `area_ha` if `area_years` is provided.
    """
    if area_years is None:
        area_years = []
 
    # --- C-factor table (semicolon, latin-1) ---
    df_c = pd.read_csv(c_factor_table_path, sep=';', encoding='latin-1')
    strat_cols = ['Tal_Pflug', 'Tal_Mulch', 'Tal_Direkt',
                  'Berg_Pflug', 'Berg_Mulch', 'Berg_Direkt']
    missing = [c for c in strat_cols if c not in df_c.columns]
    if missing:
        raise KeyError(
            f"C-factor table at {c_factor_table_path} is missing stratified "
            f"columns: {missing}. Got: {df_c.columns.tolist()}"
        )
    df_c = df_c.rename(columns={'Kultur Kategorien 2020': 'crop_name'})
    df_c = df_c[['crop_name'] + strat_cols].dropna(subset=['crop_name'])
    for c in strat_cols:
        df_c[c] = pd.to_numeric(df_c[c], errors='coerce')
    # Melt to long: one row per (crop_name, region, tillage)
    df_c_long = df_c.melt(id_vars='crop_name', value_vars=strat_cols,
                          var_name='stratum', value_name='C_ref')
    df_c_long[['region', 'tillage']] = df_c_long['stratum'].str.split('_', expand=True)
    df_c_long = df_c_long.drop(columns='stratum').dropna(subset=['C_ref'])
    df_c_long['crop_name_norm'] = df_c_long['crop_name'].map(_normalise_crop_name)
 
    # Build a wide "is-resolvable-via-crop-name" lookup using the same path
    # as the unstratified loader. We resolve `lnf_code → crop_name` once,
    # then explode by stratum.
    df_c_unique_names = df_c[['crop_name']].drop_duplicates()
    df_c_unique_names['crop_name_norm'] = df_c_unique_names['crop_name'].map(_normalise_crop_name)
    df_c_unique_names_norm = df_c_unique_names.drop_duplicates(subset='crop_name_norm', keep='first')
 
    # --- LNF classification ---
    df_lnf = pd.read_excel(lnf_classification_path, sheet_name='label_sheet')
    year_cols = {}
    for col in df_lnf.columns:
        m = _re.match(r'^(\d{4})_Area_m', str(col))
        if m:
            year_cols[int(m.group(1))] = col
    selected = [year_cols[y] for y in area_years if y in year_cols]
    has_area = bool(selected)
    if has_area:
        df_lnf['area_ha'] = df_lnf[selected].mean(axis=1) / 10_000.0
        print(f"[stratified] Area weights computed as mean over years: "
              f"{sorted(set(area_years) & set(year_cols.keys()))}")
 
    keep = ['LNF_code', 'Crop_DE']
    if has_area:
        keep.append('area_ha')
    df_lnf = df_lnf[keep].rename(columns={'LNF_code': 'lnf_code', 'Crop_DE': 'crop_name'})
    df_lnf = df_lnf[df_lnf['lnf_code'].isin(lnf_codes)].dropna(subset=['crop_name'])
    df_lnf['crop_name_norm'] = df_lnf['crop_name'].map(_normalise_crop_name)
 
    # Resolve lnf_code → exact crop_name in the C-factor table.
    # Step 1: exact match
    bridge = df_lnf.merge(df_c_unique_names[['crop_name']].assign(_match=1),
                          on='crop_name', how='left')
    # Step 2: normalised fallback
    needs = bridge['_match'].isna()
    if needs.any():
        fb = (bridge.loc[needs, ['lnf_code', 'crop_name', 'crop_name_norm']]
                    .merge(df_c_unique_names_norm[['crop_name_norm', 'crop_name']]
                           .rename(columns={'crop_name': 'crop_name_c'}),
                           on='crop_name_norm', how='left'))
        resolved = fb[fb['crop_name_c'].notna()]
        if len(resolved):
            print(f"[stratified] Resolved {len(resolved)} LNF codes via normalised name match")
        # Replace bridge.crop_name with the canonical c-table name for matches
        c_name_map = dict(zip(fb['lnf_code'], fb['crop_name_c']))
        bridge.loc[needs, 'crop_name'] = (
            bridge.loc[needs, 'lnf_code'].map(c_name_map).fillna(bridge.loc[needs, 'crop_name'])
        )
        bridge.loc[needs, '_match'] = (
            bridge.loc[needs, 'lnf_code'].map(c_name_map).notna().astype(int)
        )
 
    # Step 3: manual overrides
    if manual_overrides_path and os.path.exists(manual_overrides_path):
        df_ov = pd.read_csv(manual_overrides_path)
        # Overrides give (lnf_code → crop_name in C_Faktoren); apply as
        # replacement of bridge.crop_name and mark as resolved.
        valid_names = set(df_c_unique_names['crop_name'])
        bad = df_ov[~df_ov['crop_name'].isin(valid_names)]
        if len(bad):
            print(f"[stratified] Warning: {len(bad)} manual override rows do not "
                  "match any entry in C_Faktoren.csv — check spelling.")
        df_ov = df_ov[df_ov['crop_name'].isin(valid_names)]
        ov_map = dict(zip(df_ov['lnf_code'], df_ov['crop_name']))
        m = bridge['lnf_code'].isin(ov_map)
        if m.any():
            bridge.loc[m, 'crop_name'] = bridge.loc[m, 'lnf_code'].map(ov_map)
            bridge.loc[m, '_match']    = 1
            print(f"[stratified] Applied {m.sum()} manual override(s)")
 
    # Drop unresolved
    unresolved = bridge[bridge['_match'].isna()]
    if len(unresolved):
        print(f"[stratified] Warning: {len(unresolved)} sampled LNF codes have "
              "no C-factor entry and will be dropped from calibration:")
        print(unresolved[['lnf_code', 'crop_name']].to_string(index=False))
        bridge = bridge.dropna(subset=['_match'])
 
    bridge = bridge.drop(columns=['_match', 'crop_name_norm'])
 
    # Now explode bridge by joining onto df_c_long
    out = bridge.merge(df_c_long[['crop_name', 'region', 'tillage', 'C_ref']],
                       on='crop_name', how='left')
 
    out_cols = ['lnf_code', 'crop_name', 'region', 'tillage', 'C_ref']
    if has_area:
        out_cols.append('area_ha')
    print(f"[stratified] Reference table: {bridge['lnf_code'].nunique()} crops × "
          f"up to 6 strata = {len(out)} (crop, region, tillage) cells.")
    return out[out_cols].reset_index(drop=True)
 
 
def compute_cfactors_stratified(df: pd.DataFrame, beta: float, ts_cols: list[str],
                                fc_col: str = 'fc_total', ei_col: str = 'ei') -> pd.DataFrame:
    """Per-pixel-time-series C-factor with `region` and `tillage` carried along.
 
    Mirrors `compute_cfactors_per_pixel` but adds the stratum identifiers
    (constant within a ts_cols group) so the downstream aggregation can
    group by (crop, region, tillage).
    """
    extra = [c for c in ('region', 'tillage') if c in df.columns]
    valid = df[ts_cols + extra + [fc_col, ei_col]].dropna(subset=[fc_col, ei_col])
    slr = np.exp(-beta * valid[fc_col].values)
    weighted = slr * valid[ei_col].values
    grouped = (valid.assign(_num=weighted, _den=valid[ei_col].values)
                    .groupby(ts_cols + extra, as_index=False)[['_num', '_den']]
                    .sum())
    grouped['C_predicted'] = np.where(grouped['_den'] > 0,
                                      grouped['_num'] / grouped['_den'],
                                      np.nan)
    return grouped[ts_cols + extra + ['C_predicted']]
 
 
def aggregate_to_stratum(df_pred: pd.DataFrame, crop_col: str = 'lnf_code') -> pd.DataFrame:
    """Mean of pixel-level C up to one value per (crop, region, tillage)."""
    return (df_pred.dropna(subset=['C_predicted'])
                   .groupby([crop_col, 'region', 'tillage'], as_index=False)
                   .agg(C_predicted=('C_predicted', 'mean'),
                        n_pixels=('C_predicted', 'size')))
 
 
def _compute_stratum_weights(df_ref: pd.DataFrame,
                             df_strata: pd.DataFrame,
                             crop_col: str = 'lnf_code') -> pd.DataFrame:
    """Distribute per-crop `area_ha` across that crop's populated strata
    in proportion to `n_pixels`, then normalise to sum to 1.
 
    Keeps each crop's contribution to the loss equal to its unstratified
    contribution, just split across the strata that actually have pixels.
    """
    if 'area_ha' not in df_ref.columns:
        return None
    # Crop-level area (one value per crop in df_ref; strata duplicate it)
    crop_area = (df_ref[[crop_col, 'area_ha']].drop_duplicates(subset=[crop_col])
                                              .set_index(crop_col)['area_ha'])
    # Total pixel count per crop in the populated strata
    n_per_crop = df_strata.groupby(crop_col)['n_pixels'].sum()
    df_strata = df_strata.copy()
    df_strata['weight'] = df_strata.apply(
        lambda r: (crop_area.get(r[crop_col], 0.0)
                   * r['n_pixels'] / n_per_crop[r[crop_col]])
                  if n_per_crop.get(r[crop_col], 0) > 0 else 0.0,
        axis=1,
    )
    total = df_strata['weight'].sum()
    if total > 0:
        df_strata['weight'] = df_strata['weight'] / total
    return df_strata
 
 
def calibrate_beta_stratified(df: pd.DataFrame, df_ref: pd.DataFrame,
                              ts_cols: list[str],
                              crop_col: str = 'lnf_code',
                              fc_col: str = 'fc_total', ei_col: str = 'ei',
                              beta_bounds: tuple[float, float] = (1e-4, 0.1),
                              area_weight: bool = True,
                              ) -> tuple[float, pd.DataFrame]:
    """Find β minimising MAE over (crop, region, tillage) strata.
 
    Loss form (Form 1): one residual per non-empty stratum, weighted by
    (crop area × pixel count in stratum / total pixel count in crop)
    normalised across all populated strata. Set `area_weight=False` for
    equal-weighted stratum MAE.
    """
    join_cols = [crop_col, 'region', 'tillage']
    use_area = area_weight and 'area_ha' in df_ref.columns
 
    def _build_loss_inputs(beta):
        df_pred   = compute_cfactors_stratified(df, beta, ts_cols, fc_col, ei_col)
        df_strata = aggregate_to_stratum(df_pred, crop_col)
        merged    = df_ref.merge(df_strata, on=join_cols, how='inner')
        return merged
 
    def objective(beta: float) -> float:
        merged = _build_loss_inputs(beta)
        if len(merged) == 0:
            return 1e6
        abs_res = np.abs(merged['C_predicted'] - merged['C_ref']).values
        if use_area:
            w = _compute_stratum_weights(df_ref, merged, crop_col)['weight'].values
            w = w / w.sum() if w.sum() > 0 else w
            return float((abs_res * w).sum())
        return float(abs_res.mean())
 
    result = minimize_scalar(objective, bounds=beta_bounds, method='bounded')
    beta_opt = float(result.x)
    mae_opt = float(result.fun)
    label = "area-weighted stratum MAE" if use_area else "equal-weighted stratum MAE"
    print(f"[stratified] Optimal β: {beta_opt:.5f}   {label}: {mae_opt:.4f}")
 
    # Diagnostic stratum-level table at β_opt
    merged = _build_loss_inputs(beta_opt)
    if use_area:
        merged = _compute_stratum_weights(df_ref, merged, crop_col)
    merged['residual']     = merged['C_predicted'] - merged['C_ref']
    merged['abs_residual'] = merged['residual'].abs()
    return beta_opt, merged
 
 
def plot_calibration_stratified(df_strata: pd.DataFrame, beta_opt: float,
                                save_path: str, area_weight: bool = True) -> None:
    """Predicted vs reference C per (crop, region, tillage) stratum.
 
    Colour by region (Tal/Berg), marker by tillage (Pflug/Mulch/Direkt).
    """
    valid = df_strata.dropna(subset=['C_ref', 'C_predicted']).copy()
    if len(valid) == 0:
        print(f"[stratified] No valid strata to plot; skipping {save_path}")
        return
    use_area = area_weight and 'weight' in valid.columns
 
    mae_unw = valid['abs_residual'].mean()
    bias    = valid['residual'].mean()
    mae_aw  = float((valid['abs_residual'].values *
                     (valid['weight'].values / valid['weight'].sum())).sum()) \
              if use_area and valid['weight'].sum() > 0 else float('nan')
 
    c_max = max(valid['C_ref'].max(), valid['C_predicted'].max()) * 1.1
    c_range = [0, c_max]
 
    region_colors = {'Tal': '#2E86AB', 'Berg': '#A23B72'}
    tillage_markers = {'Pflug': 'o', 'Mulch': 's', 'Direkt': '^'}
 
    if use_area and valid['weight'].sum() > 0:
        w = valid['weight'].values
        sizes = 30 + 370 * np.sqrt(w / w.max())
    else:
        sizes = np.full(len(valid), 80.0)
 
    fig, ax = plt.subplots(figsize=(8, 8))
    for (reg, til), sub in valid.groupby(['region', 'tillage']):
        idx = sub.index
        ax.scatter(sub['C_ref'], sub['C_predicted'],
                   s=sizes[valid.index.get_indexer(idx)],
                   c=region_colors.get(reg, 'grey'),
                   marker=tillage_markers.get(til, 'x'),
                   alpha=0.7, edgecolors='k', linewidths=0.4,
                   label=f"{reg}/{til}")
    ax.plot(c_range, c_range, 'k--', alpha=0.5, label='1:1')
    ax.set_xlim(c_range)
    ax.set_ylim(c_range)
    ax.set_xlabel('Reference C-factor (stratified)')
    ax.set_ylabel('Predicted C-factor (mean of sampled pixels)')
    title = (f'Stratified C-factor calibration  '
             f'(β = {beta_opt:.5f}, MAE = {mae_unw:.4f}, bias = {bias:+.4f}')
    title += f', area-w. MAE = {mae_aw:.4f})' if use_area else ')'
    ax.set_title(title)
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[stratified] Scatter saved to {save_path}")
 
 
def plot_beta_sensitivity_stratified(df: pd.DataFrame, df_ref: pd.DataFrame,
                                     ts_cols: list[str],
                                     beta_range: np.ndarray, beta_opt: float,
                                     save_path: str,
                                     crop_col: str = 'lnf_code',
                                     area_weight: bool = True) -> None:
    """Plot MAE vs β for stratified loss, plus per-stratum curves."""
    join_cols = [crop_col, 'region', 'tillage']
    use_area = area_weight and 'area_ha' in df_ref.columns
 
    per_stratum_records = []
    overall_mae = []
 
    for beta in beta_range:
        df_pred   = compute_cfactors_stratified(df, float(beta), ts_cols)
        df_strata = aggregate_to_stratum(df_pred, crop_col)
        merged    = df_ref.merge(df_strata, on=join_cols, how='inner')
        if len(merged) == 0:
            overall_mae.append(np.nan)
            continue
        merged['abs_diff'] = (merged['C_predicted'] - merged['C_ref']).abs()
        if use_area:
            merged = _compute_stratum_weights(df_ref, merged, crop_col)
            w = merged['weight'].values
            w = w / w.sum() if w.sum() > 0 else w
            overall_mae.append((merged['abs_diff'].values * w).sum())
        else:
            overall_mae.append(merged['abs_diff'].mean())
        for _, row in merged.iterrows():
            per_stratum_records.append({
                'beta': float(beta),
                'stratum': f"{row[crop_col]} | {row['region']}/{row['tillage']}",
                'abs_diff': row['abs_diff'],
            })
 
    df_curves = pd.DataFrame(per_stratum_records)
    fig, ax = plt.subplots(figsize=(10, 6))
    if len(df_curves):
        cmap = plt.get_cmap('tab20')
        # Don't legend per-stratum (too many) — just draw them faintly
        for i, (s, sub) in enumerate(df_curves.groupby('stratum')):
            ax.plot(sub['beta'], sub['abs_diff'], color=cmap(i % 20),
                    alpha=0.25, lw=0.7)
    label = ('All strata (area-weighted mean)' if use_area
             else 'All strata (mean)')
    ax.plot(beta_range, overall_mae, color='black', lw=2.5, linestyle='--',
            label=label)
    ax.axvline(beta_opt, color='red', linestyle=':',
               label=f'β_opt = {beta_opt:.5f}')
    ax.set_xlabel('β')
    ax.set_ylabel('Mean absolute C-factor difference (stratified)')
    ax.set_title('Stratified β sensitivity')
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[stratified] β sensitivity plot saved to {save_path}")
 
 
def run_calibration_stratified(config: dict) -> None:
    """Stratified-calibration entry point.
 
    Same SLR = exp(-β·FC) model as `run_calibration`, but targets
    stratified C_refs (Tal/Berg × Pflug/Mulch/Direkt). Outputs are
    written with a `_stratified` suffix.
    """
    if config.get('sampling_strategy', 'agis') != 'agis':
        raise ValueError(
            "stratified_calibration=True requires sampling_strategy='agis' "
            "so the gapfilled parquet carries `uuid` (Flaechen_ID) and "
            "`betr_ID` for joining altitude and tillage. Got "
            f"sampling_strategy='{config.get('sampling_strategy')}'."
        )
 
    fc_path                 = config['gapfilled_fc_path']
    ei_path                 = os.path.expanduser(config['ei_path'])
    c_factor_table_path     = os.path.expanduser(config['c_factor_table_path'])
    lnf_classification_path = os.path.expanduser(config['lnf_classification_path'])
    manual_overrides_path   = config.get('manual_overrides_path')
    if manual_overrides_path:
        manual_overrides_path = os.path.expanduser(manual_overrides_path)
    results_dir             = config['results_folder']
    nutzung_csv             = os.path.expanduser(config['nutzung_csv'])
    reb_csv                 = os.path.expanduser(config['ressourceneffizienz_csv'])
    results_path            = config['calibration_results_path']
    ts_cols                 = config.get('ts_cols', ['lnf_code', 'yr', 'poly_id'])
    crop_col                = config.get('crop_col', 'lnf_code')
    beta_bounds             = config.get('beta_bounds', (1e-4, 0.1))
    exclude_lnf_codes       = config.get('exclude_calibration_lnf_codes', []) or []
    area_weight_loss        = config.get('area_weight_loss', True)
    area_years              = config.get('area_years', []) or None
    cutoff_m                = float(config.get('grenze_tal_berg', 600))
    default_tillage         = config.get('standardansaatverfahren', 'Pflug')
    tillage_method          = config.get('tillage_assignment', 'stochastic')
    seed                    = int(config.get('tillage_random_seed', 42))
    
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, results_path)

    # Build stratified output paths from the unstratified ones
    base, ext = os.path.splitext(results_path)
    results_path_strat = base + '_stratified' + ext
    pixel_path_strat   = base + '_stratified_per_pixel' + ext
    scatter_path       = os.path.join(results_dir, 'calibration_scatter_stratified.png')
    sensitivity_path   = os.path.join(results_dir, 'beta_sensitivity_stratified.png')
    beta_json_path     = os.path.join(results_dir, 'beta_stratified.json')
 
    print("Loading gapfilled FC timeseries (stratified mode)...")
    df_fc = pd.read_parquet(fc_path)
    for need in ('uuid', 'betr_ID'):
        if need not in df_fc.columns:
            raise KeyError(
                f"Gapfilled parquet at {fc_path} has no '{need}' column. "
                "Stratified calibration needs AGIS identifiers carried "
                "through sample_FC.py — re-run sampling with the current "
                "version of sample_FC.py to regenerate it."
            )
 
    # Reference (stratified)
    sampled_lnf_codes = sorted(df_fc[crop_col].unique().tolist())
    df_ref = load_reference_cfactors_stratified(c_factor_table_path,
                                                lnf_classification_path,
                                                sampled_lnf_codes,
                                                manual_overrides_path=manual_overrides_path,
                                                area_years=area_years)
    if exclude_lnf_codes:
        before = df_ref[crop_col].nunique()
        df_ref = df_ref[~df_ref[crop_col].isin(exclude_lnf_codes)].reset_index(drop=True)
        print(f"[stratified] Excluded {before - df_ref[crop_col].nunique()} crops "
              f"from calibration.")
    df_ref[crop_col] = df_ref[crop_col].astype(df_fc[crop_col].dtype)
 
    # Assign strata onto FC
    df_fc = assign_strata(df_fc,
                          nutzung_csv=nutzung_csv,
                          reb_csv=reb_csv,
                          cutoff_m=cutoff_m,
                          default_tillage=default_tillage,
                          tillage_method=tillage_method,
                          seed=seed)
 
    # EI loading and join (same as unstratified)
    print("[stratified] Snapping FC coordinates to EI grid and loading EI ...")
    x_off, y_off = get_ei_grid_offset(ei_path)
    x_snap, y_snap = snap_to_ei_grid(df_fc['x'].values, df_fc['y'].values, x_off, y_off)
    df_ei = load_ei_for_pixels(ei_path, np.unique(x_snap), np.unique(y_snap))
    df = join_ei_to_fc(df_fc, df_ei, x_off, y_off)
 
    # Calibrate
    print("[stratified] Calibrating β (stratum-level MAE objective)...")
    beta_opt, df_strata = calibrate_beta_stratified(df, df_ref, ts_cols,
                                                    crop_col=crop_col,
                                                    beta_bounds=beta_bounds,
                                                    area_weight=area_weight_loss)
    with open(beta_json_path, 'w') as f:
        json.dump({'beta': beta_opt,
                   'area_weight_loss': area_weight_loss,
                   'mode': 'stratified',
                   'cutoff_m': cutoff_m,
                   'tillage_assignment': tillage_method,
                   'tillage_random_seed': seed,
                   'default_tillage': default_tillage}, f, indent=2)
 
    df_strata.to_csv(results_path_strat, index=False)
    print(f"[stratified] Per-stratum results → {results_path_strat}")
 
    # Per-pixel output at β_opt with stratum labels and matched C_ref
    df_pix = compute_cfactors_stratified(df, beta_opt, ts_cols)
    df_pix = df_pix.merge(
        df_ref[[crop_col, 'region', 'tillage', 'C_ref']],
        on=[crop_col, 'region', 'tillage'], how='left',
    )
    df_pix.to_csv(pixel_path_strat, index=False)
    print(f"[stratified] Per-pixel C-factors → {pixel_path_strat}")
 
    # Plots
    plot_calibration_stratified(df_strata, beta_opt, scatter_path,
                                area_weight=area_weight_loss)
    beta_range = np.linspace(beta_bounds[0], beta_bounds[1], 60)
    plot_beta_sensitivity_stratified(df, df_ref, ts_cols, beta_range, beta_opt,
                                     sensitivity_path, crop_col=crop_col,
                                     area_weight=area_weight_loss)
 
 
# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
 
def run_calibration(config: dict) -> None:
    """Run the full C-factor calibration pipeline.
 
    If ``config['stratified_calibration']`` is True, dispatches to
    ``run_calibration_stratified`` (separate outputs with `_stratified`
    suffix). Otherwise runs the original unstratified pipeline against
    the per-crop `Total` column of the C-factor table.
    """
    if config.get('stratified_calibration', False):
        return run_calibration_stratified(config)
 
    fc_path                 = config['gapfilled_fc_path']
    ei_path                 = os.path.expanduser(config['ei_path'])
    c_factor_table_path     = os.path.expanduser(config['c_factor_table_path'])
    lnf_classification_path = os.path.expanduser(config['lnf_classification_path'])
    manual_overrides_path   = config.get('manual_overrides_path')
    if manual_overrides_path:
        manual_overrides_path = os.path.expanduser(manual_overrides_path)
    results_dir             = config['results_folder']
    results_path            = config['calibration_results_path']
    ts_cols                 = config.get('ts_cols', ['lnf_code', 'yr', 'poly_id'])
    crop_col                = config.get('crop_col', 'lnf_code')
    beta_bounds             = config.get('beta_bounds', (1e-4, 0.1))
    exclude_lnf_codes       = config.get('exclude_calibration_lnf_codes', []) or []
    area_weight_loss        = config.get('area_weight_loss', True)
    area_years              = config.get('area_years', []) or None
    
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, results_path)

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
    with open(os.path.join(results_dir,'beta.json'), 'w') as f:
        json.dump({'beta': beta_opt, 'area_weight_loss': area_weight_loss}, f)
        
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
    plot_calibration_per_crop(df_crop, beta_opt, os.path.join(results_dir,'calibration_scatter.png'),
                              area_weight=area_weight_loss)
    beta_range = np.linspace(beta_bounds[0], beta_bounds[1], 60)
    plot_beta_sensitivity(df, df_ref, ts_cols, beta_range, beta_opt,
                          os.path.join(results_dir,'beta_sensitivity.png'), crop_col=crop_col,
                          area_weight=area_weight_loss)
 
 
    print("Done.")