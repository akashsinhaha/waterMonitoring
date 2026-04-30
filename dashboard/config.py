import os

# Points to results_timeseries/ relative to dashboard folder
# Override with environment variable for deployment:
#   export RESULTS_DIR=/path/to/results_timeseries
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.getenv('RESULTS_DIR', os.path.join(BASE_DIR, '..', 'results_timeseries'))

YEARS           = list(range(2017, 2026))
PIXEL_AREA_KM2  = (10 * 10) / 1e6   # 10m resolution → 100m² → 0.0001 km²
MAP_OVERLAY_MAX_PX = 1200            # max pixels on longest side for map overlay

GEE_PROJECT     = os.getenv('GEE_PROJECT', '')
UNET_MODEL_PATH = os.path.join(BASE_DIR, '..', 'results_unet', 'unet_best.pth')
AOI_MAX_KM2     = 500                # warn user if drawn AOI exceeds this area
