"""
U-Net inference utilities for the Custom AOI tab.
Extracted and adapted from water_timeseries.py.

Architecture must exactly match the saved weights in results_unet/unet_best.pth.
"""
import io
import base64

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.crs import CRS
import torch
from PIL import Image


# ── U-NET ARCHITECTURE (must match saved weights) ─────────────────────────────

class _DoubleConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class _EncoderBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = _DoubleConv(in_channels, out_channels)
        self.pool = torch.nn.MaxPool2d(2, 2)
    def forward(self, x):
        skip = self.conv(x)
        return skip, self.pool(skip)


class _DecoderBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = torch.nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv     = _DoubleConv(in_channels, out_channels)
    def forward(self, x, skip):
        x = self.upsample(x)
        return self.conv(torch.cat([skip, x], dim=1))


class _UNet(torch.nn.Module):
    def __init__(self, in_channels=7, features=None):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]
        self.encoder1   = _EncoderBlock(in_channels, features[0])
        self.encoder2   = _EncoderBlock(features[0], features[1])
        self.encoder3   = _EncoderBlock(features[1], features[2])
        self.encoder4   = _EncoderBlock(features[2], features[3])
        self.bottleneck = _DoubleConv(features[3], features[3] * 2)
        self.decoder4   = _DecoderBlock(features[3] * 2, features[3])
        self.decoder3   = _DecoderBlock(features[3],     features[2])
        self.decoder2   = _DecoderBlock(features[2],     features[1])
        self.decoder1   = _DecoderBlock(features[1],     features[0])
        self.final_conv = torch.nn.Conv2d(features[0], 1, 1)

    def forward(self, x):
        s1, x = self.encoder1(x)
        s2, x = self.encoder2(x)
        s3, x = self.encoder3(x)
        s4, x = self.encoder4(x)
        x = self.bottleneck(x)
        x = self.decoder4(x, s4)
        x = self.decoder3(x, s3)
        x = self.decoder2(x, s2)
        x = self.decoder1(x, s1)
        return self.final_conv(x)


# ── CONSTANTS ─────────────────────────────────────────────────────────────────

# Indices in the 10-band S2 file that map to B2, B3, B4, B8, B11
_S2_UNET_IDX     = [0, 1, 2, 6, 8]
_S2_SCALE_FACTOR = 10000.0
_PATCH_SIZE      = 256
_STRIDE          = 128
_PIXEL_AREA_KM2  = (10 * 10) / 1e6   # 10 m resolution


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def load_unet_model(model_path):
    """Load U-Net from a .pth checkpoint. Returns model in eval mode."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = _UNet(in_channels=7).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def load_rasters_from_paths(s2_path, s1_asc_path, s1_desc_path):
    """
    Load S2 (10-band) + S1 ASC/DESC and reproject S1 onto the S2 grid.

    GEE downloads S2 in DN (0–10000). Existing training TIFs are in 0–1
    reflectance and get multiplied by 10000 before U-Net input. This function
    auto-detects the scale and normalises so the U-Net always receives DN values.

    Returns
    -------
    s2_full   : (10, H, W) float32  0–1 reflectance
    s2_unet   : (5,  H, W) float32  DN scale for U-Net input
    s1_asc    : (H, W)     float32
    s1_desc   : (H, W)     float32
    crs       : rasterio CRS
    transform : rasterio Affine
    """
    with rasterio.open(s2_path) as s2_src:
        s2_raw    = s2_src.read().astype(np.float32)  # (bands, H, W)
        transform = s2_src.transform
        crs       = s2_src.crs
        height    = s2_src.height
        width     = s2_src.width

    s2_raw = np.nan_to_num(s2_raw, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalise to 0–1 reflectance regardless of source scale
    if s2_raw.max() > 1.0:
        s2_full = s2_raw / _S2_SCALE_FACTOR
    else:
        s2_full = s2_raw

    # Scale back to DN for U-Net (trained on DN-scale inputs)
    s2_unet = s2_full[_S2_UNET_IDX] * _S2_SCALE_FACTOR  # (5, H, W)

    def _reproject_s1(path):
        with rasterio.open(path) as src:
            dst_array = np.zeros((height, width), dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=dst_array,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=crs,
                resampling=Resampling.bilinear,
            )
        return np.nan_to_num(dst_array, nan=0.0, posinf=0.0, neginf=0.0)

    s1_asc  = _reproject_s1(s1_asc_path)
    s1_desc = _reproject_s1(s1_desc_path)

    return s2_full, s2_unet, s1_asc, s1_desc, crs, transform


def run_unet_inference(model, s2_unet, s1_asc, s1_desc):
    """
    Patch-based U-Net inference with overlap averaging.

    Input channels: B2, B3, B4, B8, B11 (from s2_unet) + S1_ASC + S1_DESC — all DN scale.

    Returns
    -------
    water_mask : (H, W) uint8  binary water map (1 = water)
    prob_map   : (H, W) float32  per-pixel water probability
    """
    device = next(model.parameters()).device

    image_stack = np.vstack([
        s2_unet,
        s1_asc[np.newaxis],
        s1_desc[np.newaxis],
    ])  # (7, H, W)

    _, orig_height, orig_width = image_stack.shape

    # Pad to at least _PATCH_SIZE so the loop always executes at least once
    pad_h = max(0, _PATCH_SIZE - orig_height)
    pad_w = max(0, _PATCH_SIZE - orig_width)
    if pad_h > 0 or pad_w > 0:
        image_stack = np.pad(
            image_stack,
            ((0, 0), (0, pad_h), (0, pad_w)),
            mode='reflect',
        )

    _, height, width = image_stack.shape
    prediction_sum   = np.zeros((height, width), dtype=np.float32)
    prediction_count = np.zeros((height, width), dtype=np.float32)

    with torch.no_grad():
        for row in range(0, height - _PATCH_SIZE + 1, _STRIDE):
            for col in range(0, width - _PATCH_SIZE + 1, _STRIDE):
                patch  = image_stack[:, row:row+_PATCH_SIZE, col:col+_PATCH_SIZE]
                tensor = torch.from_numpy(patch).unsqueeze(0).to(device)
                logits = model(tensor).squeeze().cpu().numpy()
                prob   = 1.0 / (1.0 + np.exp(-logits))   # sigmoid

                prediction_sum[row:row+_PATCH_SIZE, col:col+_PATCH_SIZE]   += prob
                prediction_count[row:row+_PATCH_SIZE, col:col+_PATCH_SIZE] += 1

    valid_pixels = prediction_count > 0
    prob_map     = np.zeros((height, width), dtype=np.float32)
    prob_map[valid_pixels] = prediction_sum[valid_pixels] / prediction_count[valid_pixels]

    # Crop back to original dimensions before padding
    prob_map   = prob_map[:orig_height, :orig_width]
    water_mask = (prob_map > 0.5).astype(np.uint8)
    return water_mask, prob_map


def compute_quality_stats(s2_full, water_mask):
    """
    Compute water area + quality indices within water pixels.

    s2_full: (10, H, W) float32 in 0–1 reflectance.
    Band layout: B2(0) B3(1) B4(2) B5(3) B6(4) B7(5) B8(6) B8A(7) B11(8) B12(9)

    Returns dict: water_area_km2, ndti, ndci, clarity, algae, sediment.
    """
    B2, B3, B4 = s2_full[0], s2_full[1], s2_full[2]
    B5         = s2_full[3]
    B8         = s2_full[6]

    water_pixels = water_mask.astype(bool)

    def _safe_norm_diff(band_a, band_b):
        with np.errstate(divide='ignore', invalid='ignore'):
            result = (band_a - band_b) / (band_a + band_b)
            result[~np.isfinite(result)] = 0.0
        return result

    def _masked_mean(index_map):
        values = index_map[water_pixels]
        values = values[np.isfinite(values)]
        return float(np.mean(values)) if len(values) > 0 else float('nan')

    return {
        'water_area_km2': float(water_pixels.sum()) * _PIXEL_AREA_KM2,
        'ndti'          : _masked_mean(_safe_norm_diff(B4, B3)),
        'ndci'          : _masked_mean(_safe_norm_diff(B5, B4)),
        'clarity'       : _masked_mean(np.where(B4 > 0, B2 / (B4 + 1e-8), 0.0)),
        'algae'         : _masked_mean(B8 - B4),
        'sediment'      : _masked_mean(B4),
    }


def build_change_overlay_png(mask_start, mask_end, crs, transform,
                             max_px=1200, alpha=190):
    """
    Build a change detection overlay PNG between two water masks.
    Teal = stable water | Green = new water gain | Red = water loss.
    Returns (png_data_url, [[south, west], [north, east]]).
    """
    h, w = mask_start.shape
    rgba_image = np.zeros((h, w, 4), dtype=np.uint8)

    stable_water = (mask_start == 1) & (mask_end == 1)
    water_gain   = (mask_start == 0) & (mask_end == 1)
    water_loss   = (mask_start == 1) & (mask_end == 0)

    rgba_image[stable_water] = [0,   180, 216, alpha]
    rgba_image[water_gain]   = [45,  198, 83,  alpha]
    rgba_image[water_loss]   = [239, 35,  60,  alpha]

    pil_image = Image.fromarray(rgba_image, 'RGBA')
    if max(h, w) > max_px:
        scale_ratio = max_px / max(h, w)
        new_size    = (int(w * scale_ratio), int(h * scale_ratio))
        pil_image   = pil_image.resize(new_size, Image.NEAREST)

    buffer  = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    encoded = base64.b64encode(buffer.getvalue()).decode()

    raster_left   = transform.c
    raster_top    = transform.f
    raster_right  = raster_left + transform.a * w
    raster_bottom = raster_top  + transform.e * h

    west, south, east, north = transform_bounds(
        crs, CRS.from_epsg(4326),
        min(raster_left, raster_right), min(raster_top,  raster_bottom),
        max(raster_left, raster_right), max(raster_top,  raster_bottom),
    )
    return f'data:image/png;base64,{encoded}', [[south, west], [north, east]]


def mask_array_to_overlay_png(water_mask, crs, transform,
                               max_px=1200, color=(0, 116, 217), alpha=170):
    """
    Convert an in-memory water mask array to a base64 PNG data URL + WGS84 bounds
    suitable for folium.ImageOverlay.

    Returns (png_data_url, [[south, west], [north, east]]).
    """
    h, w = water_mask.shape

    rgba_image          = np.zeros((h, w, 4), dtype=np.uint8)
    water_pixels        = water_mask == 1
    rgba_image[water_pixels] = [color[0], color[1], color[2], alpha]

    pil_image = Image.fromarray(rgba_image, 'RGBA')
    if max(h, w) > max_px:
        scale_ratio = max_px / max(h, w)
        new_size    = (int(w * scale_ratio), int(h * scale_ratio))
        pil_image   = pil_image.resize(new_size, Image.NEAREST)

    buffer  = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    encoded = base64.b64encode(buffer.getvalue()).decode()

    # Compute WGS84 bounding box from raster metadata
    raster_left   = transform.c
    raster_top    = transform.f
    raster_right  = raster_left + transform.a * w
    raster_bottom = raster_top  + transform.e * h

    west, south, east, north = transform_bounds(
        crs, CRS.from_epsg(4326),
        min(raster_left, raster_right),
        min(raster_top,  raster_bottom),
        max(raster_left, raster_right),
        max(raster_top,  raster_bottom),
    )
    overlay_bounds = [[south, west], [north, east]]
    return f'data:image/png;base64,{encoded}', overlay_bounds
