import os
import io
import base64
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds
from rasterio.crs import CRS
from PIL import Image
import streamlit as st
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR, YEARS, FORECAST_YEARS, PIXEL_AREA_KM2, MAP_OVERLAY_MAX_PX


@st.cache_data
def load_summary():
    """Load precomputed timeseries summary CSV."""
    csv_path = os.path.join(RESULTS_DIR, 'timeseries_summary.csv')
    df = pd.read_csv(csv_path, index_col='year')
    # Backfill is_forecast column for CSVs written before the forecast feature
    if 'is_forecast' not in df.columns:
        df['is_forecast'] = False
    else:
        df['is_forecast'] = df['is_forecast'].fillna(False).astype(bool)
    return df


def is_forecast_year(year):
    """Return True if year is a model-predicted forecast, not an observed year."""
    return year in FORECAST_YEARS


@st.cache_data
def get_available_mask_years(years):
    """Return only years that have a water mask file on disk."""
    masks_dir = os.path.join(RESULTS_DIR, 'water_masks')
    return [y for y in years if os.path.exists(os.path.join(masks_dir, f'water_mask_{y}.tif'))]


@st.cache_data
def load_water_mask(year):
    """
    Load binary water mask for a given year.
    Returns mask (H, W uint8), rasterio bounds, crs, y_res_sign.
    y_res_sign: +1 means south-up (image needs vertical flip), -1 means north-up (standard).
    """
    path = os.path.join(RESULTS_DIR, 'water_masks', f'water_mask_{year}.tif')
    with rasterio.open(path) as src:
        mask        = src.read(1)
        bounds      = src.bounds
        crs         = src.crs
        y_res_sign  = 1 if src.transform.e > 0 else -1
    return mask, bounds, crs, y_res_sign


@st.cache_data
def get_map_bounds():
    """
    Return bounding box in WGS84 as [[south, west], [north, east]]
    for Folium map centering. Uses the first available year.
    """
    for year in YEARS:
        try:
            _, bounds, crs, _ = load_water_mask(year)
            west, south, east, north = transform_bounds(
                crs, CRS.from_epsg(4326),
                bounds.left, bounds.bottom, bounds.right, bounds.top
            )
            return [[south, west], [north, east]]
        except Exception:
            continue
    return [[-90, -180], [90, 180]]


@st.cache_data
def get_overlay_bounds(year):
    """
    Return WGS84 bounds [[south, west], [north, east]] for a specific year's raster.
    Used to position the ImageOverlay precisely.
    """
    _, bounds, crs, _ = load_water_mask(year)
    west, south, east, north = transform_bounds(
        crs, CRS.from_epsg(4326),
        bounds.left, bounds.bottom, bounds.right, bounds.top
    )
    return [[south, west], [north, east]]


@st.cache_data
def mask_to_overlay_png(year, color=(0, 116, 217), alpha=170):
    """
    Convert a water mask GeoTIFF to a base64 PNG for Folium ImageOverlay.
    Water pixels → semi-transparent blue; non-water → fully transparent.
    Returns (png_b64, overlay_bounds) where overlay_bounds = [[south, west], [north, east]].
    """
    mask, bounds, crs, y_res_sign = load_water_mask(year)

    # Flip vertically if raster is south-up (positive y resolution)
    if y_res_sign > 0:
        mask = np.flipud(mask)

    h, w = mask.shape
    rgba        = np.zeros((h, w, 4), dtype=np.uint8)
    water       = mask == 1
    rgba[water] = [color[0], color[1], color[2], alpha]

    img = Image.fromarray(rgba, 'RGBA')

    # Downscale for browser performance
    if max(h, w) > MAP_OVERLAY_MAX_PX:
        ratio    = MAP_OVERLAY_MAX_PX / max(h, w)
        new_size = (int(w * ratio), int(h * ratio))
        img      = img.resize(new_size, Image.NEAREST)

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    encoded = base64.b64encode(buf.getvalue()).decode()

    west, south, east, north = transform_bounds(
        crs, CRS.from_epsg(4326),
        bounds.left, bounds.bottom, bounds.right, bounds.top
    )
    overlay_bounds = [[south, west], [north, east]]
    return f'data:image/png;base64,{encoded}', overlay_bounds


@st.cache_data
def change_map_to_overlay_png(year_a, year_b, alpha=190):
    """
    Build a change-detection overlay PNG between two years.
    Blue  = stable water | Green = water gain | Red = water loss
    Returns (png_b64, overlay_bounds) where overlay_bounds = [[south, west], [north, east]].
    Uses year_b's extent and orientation as the reference.
    """
    mask_a, _, _, y_res_sign_a = load_water_mask(year_a)
    mask_b, bounds_b, crs_b, y_res_sign_b = load_water_mask(year_b)

    if y_res_sign_a > 0:
        mask_a = np.flipud(mask_a)
    if y_res_sign_b > 0:
        mask_b = np.flipud(mask_b)

    h, w  = mask_a.shape
    rgba  = np.zeros((h, w, 4), dtype=np.uint8)

    stable = (mask_a == 1) & (mask_b == 1)
    gain   = (mask_a == 0) & (mask_b == 1)
    loss   = (mask_a == 1) & (mask_b == 0)

    rgba[stable] = [0,   180, 216, alpha]
    rgba[gain]   = [45,  198, 83,  alpha]
    rgba[loss]   = [239, 35,  60,  alpha]

    img = Image.fromarray(rgba, 'RGBA')

    if max(h, w) > MAP_OVERLAY_MAX_PX:
        ratio    = MAP_OVERLAY_MAX_PX / max(h, w)
        new_size = (int(w * ratio), int(h * ratio))
        img      = img.resize(new_size, Image.NEAREST)

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    encoded = base64.b64encode(buf.getvalue()).decode()

    west, south, east, north = transform_bounds(
        crs_b, CRS.from_epsg(4326),
        bounds_b.left, bounds_b.bottom, bounds_b.right, bounds_b.top
    )
    overlay_bounds = [[south, west], [north, east]]
    return f'data:image/png;base64,{encoded}', overlay_bounds


def compute_stats_for_year(df, year):
    """Return a dict of display stats for a single year."""
    if year not in df.index:
        return {}

    row = df.loc[year]

    stats = {
        'water_area_km2'   : row.get('water_area_km2', None),
        'area_change_km2'  : row.get('area_change_km2', None),
        'area_change_pct'  : row.get('area_change_pct', None),
        'ndti'             : row.get('ndti', None),
        'ndci'             : row.get('ndci', None),
        'clarity'          : row.get('clarity', None),
        'sediment'         : row.get('sediment', None),
        'algae'            : row.get('algae', None),
        'is_forecast'      : bool(row.get('is_forecast', False)),
    }
    return stats
