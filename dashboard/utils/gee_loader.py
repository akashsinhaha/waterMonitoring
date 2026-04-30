import os
import math
import ee
import geemap


def init_gee(gee_project):
    """
    Initialize Google Earth Engine with the given project ID.
    Raises RuntimeError with user-friendly instructions if authentication fails.
    """
    if not gee_project:
        raise RuntimeError(
            "No GEE project ID provided.\n"
            "Enter your Google Earth Engine project ID in the 'GEE Project ID' field."
        )
    try:
        ee.Initialize(project=gee_project)
    except Exception as original_error:
        raise RuntimeError(
            f"GEE connection failed: {original_error}\n\n"
            "To fix this:\n"
            "  1. Open a terminal and run:  earthengine authenticate\n"
            "  2. Follow the browser link to log in\n"
            "  3. Make sure your GEE project ID is correct"
        ) from original_error


def check_aoi_size_km2(aoi_geojson):
    """Return approximate bounding-box area of a drawn GeoJSON shape in km²."""
    geometry = aoi_geojson.get('geometry', aoi_geojson)
    geom_type = geometry.get('type', '')

    if geom_type == 'Polygon':
        coords = geometry['coordinates'][0]
    elif geom_type == 'Rectangle' or (geom_type == 'Feature'):
        coords = geometry.get('coordinates', [[]])[0]
    else:
        # Fallback: try to parse directly
        coords = geometry.get('coordinates', [[]])[0]

    if not coords:
        return 0.0

    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    center_lat_rad = math.radians(sum(lats) / len(lats))
    lat_span_km = abs(max(lats) - min(lats)) * 111.0
    lon_span_km = abs(max(lons) - min(lons)) * 111.0 * math.cos(center_lat_rad)
    return lat_span_km * lon_span_km


def download_sentinel_composite(aoi_geojson, year, output_dir):
    """
    Download Sentinel-2 and Sentinel-1 March–June composites for a drawn AOI.

    S2: 10 bands (B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12) at 10 m, DN scale (0–10000)
    S1 ASC: VV band, ascending orbit, mean composite, 10 m
    S1 DESC: VV band, descending orbit, mean composite, 10 m

    Returns (s2_path, s1_asc_path, s1_desc_path) as absolute paths.
    """
    geometry = aoi_geojson.get('geometry', aoi_geojson)
    aoi_ee   = ee.Geometry(geometry)

    march_start = f'{year}-03-01'
    june_end    = f'{year}-06-30'

    s2_image = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(aoi_ee)
        .filterDate(march_start, june_end)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'])
        .median()
        .toFloat()
    )

    s1_asc_image = (
        ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterBounds(aoi_ee)
        .filterDate(march_start, june_end)
        .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .select('VV')
        .mean()
    )

    s1_desc_image = (
        ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterBounds(aoi_ee)
        .filterDate(march_start, june_end)
        .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .select('VV')
        .mean()
    )

    os.makedirs(output_dir, exist_ok=True)
    s2_path      = os.path.join(output_dir, f'S2_{year}.tif')
    s1_asc_path  = os.path.join(output_dir, f'S1_ASC_{year}.tif')
    s1_desc_path = os.path.join(output_dir, f'S1_DESC_{year}.tif')

    scale = 10
    geemap.ee_export_image(s2_image,      filename=s2_path,      scale=scale, region=aoi_ee, file_per_band=False)
    geemap.ee_export_image(s1_asc_image,  filename=s1_asc_path,  scale=scale, region=aoi_ee, file_per_band=False)
    geemap.ee_export_image(s1_desc_image, filename=s1_desc_path, scale=scale, region=aoi_ee, file_per_band=False)

    return s2_path, s1_asc_path, s1_desc_path
