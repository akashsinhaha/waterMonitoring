"""
U-Net and ConvLSTM inference utilities for the Custom AOI tab.

U-Net architecture must exactly match the saved weights in results_unet/unet_best.pth.
ConvLSTM architecture must exactly match the saved weights in
results_timeseries/forecast/convlstm_best.pth (trained by water_spatial_forecast.py).
"""
import io
import os
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



# ── CONVLSTM ARCHITECTURE (must match water_spatial_forecast.py weights) ──────

_CONVLSTM_HIDDEN_CHANNELS = 16
_CONVLSTM_KERNEL_SIZE     = 3
_CONVLSTM_PATCH_SIZE      = 256
_CONVLSTM_STRIDE          = 128


class _ConvLSTMCell(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.fused_gates_conv = torch.nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(self, input_frame, hidden_state):
        h_previous, c_previous = hidden_state
        concatenated_input = torch.cat([input_frame, h_previous], dim=1)
        all_gate_outputs   = self.fused_gates_conv(concatenated_input)

        input_gate, forget_gate, cell_gate, output_gate = torch.chunk(all_gate_outputs, 4, dim=1)
        input_gate  = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        cell_gate   = torch.tanh(cell_gate)
        output_gate = torch.sigmoid(output_gate)

        c_next = forget_gate * c_previous + input_gate * cell_gate
        h_next = output_gate * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, spatial_height, spatial_width, device):
        zeros = torch.zeros(
            batch_size, self.hidden_channels, spatial_height, spatial_width, device=device
        )
        return zeros, zeros.clone()


class _ConvLSTMForecaster(torch.nn.Module):
    def __init__(self, hidden_channels=_CONVLSTM_HIDDEN_CHANNELS,
                 kernel_size=_CONVLSTM_KERNEL_SIZE):
        super().__init__()
        self.convlstm_cell = _ConvLSTMCell(
            input_channels=1,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
        )
        self.prediction_head = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(hidden_channels // 2, 1, kernel_size=1),
        )

    def forward(self, sequence_tensor):
        batch_size, seq_len, _, spatial_height, spatial_width = sequence_tensor.shape
        device = sequence_tensor.device
        h_state, c_state = self.convlstm_cell.init_hidden(
            batch_size, spatial_height, spatial_width, device
        )
        for timestep_idx in range(seq_len):
            current_frame = sequence_tensor[:, timestep_idx]   # (B, 1, H, W)
            h_state, c_state = self.convlstm_cell(current_frame, (h_state, c_state))
        return self.prediction_head(h_state)   # (B, 1, H, W) logits


def load_convlstm_model(model_path):
    """
    Load the pre-trained ConvLSTMForecaster checkpoint.
    Raises FileNotFoundError with instructions if the checkpoint is missing.
    Returns model in eval mode on the best available device.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f'ConvLSTM checkpoint not found: {model_path}\n'
            'Run  python water_spatial_forecast.py  first to train and save the model.'
        )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = _ConvLSTMForecaster(
        hidden_channels=_CONVLSTM_HIDDEN_CHANNELS,
        kernel_size=_CONVLSTM_KERNEL_SIZE,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def run_convlstm_forecast(model, annual_masks_by_year, input_years):
    """
    Run the pre-trained ConvLSTM on in-memory water masks from a custom AOI.

    Uses the same patch-based overlap-averaging strategy as the standalone
    water_spatial_forecast.py script, so the inference is spatially consistent
    regardless of AOI size.  Small AOIs (< 256 px) are padded via reflection
    before inference and cropped back afterwards.

    Parameters
    ----------
    model                : _ConvLSTMForecaster loaded via load_convlstm_model()
    annual_masks_by_year : dict of year (int) → (H, W) uint8 binary water mask
    input_years          : list of exactly CONVLSTM_SEQUENCE_LEN years to feed as input

    Returns
    -------
    probability_map : (H, W) float32  per-pixel water probability [0, 1]
    predicted_mask  : (H, W) uint8    binary mask thresholded at 0.5
    """
    device = next(model.parameters()).device

    reference_mask   = annual_masks_by_year[input_years[0]].astype(np.float32)
    orig_height, orig_width = reference_mask.shape

    # Stack input years → (T, H, W) float32
    spatial_input = np.stack(
        [annual_masks_by_year[y].astype(np.float32) for y in input_years], axis=0
    )

    # Pad so the sliding window executes at least once
    pad_h = max(0, _CONVLSTM_PATCH_SIZE - orig_height)
    pad_w = max(0, _CONVLSTM_PATCH_SIZE - orig_width)
    if pad_h > 0 or pad_w > 0:
        spatial_input = np.pad(
            spatial_input, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect'
        )

    _, full_height, full_width = spatial_input.shape

    prediction_sum_map   = np.zeros((full_height, full_width), dtype=np.float32)
    prediction_count_map = np.zeros((full_height, full_width), dtype=np.float32)

    with torch.no_grad():
        for row_start in range(0, full_height - _CONVLSTM_PATCH_SIZE + 1, _CONVLSTM_STRIDE):
            for col_start in range(0, full_width - _CONVLSTM_PATCH_SIZE + 1, _CONVLSTM_STRIDE):
                row_end = row_start + _CONVLSTM_PATCH_SIZE
                col_end = col_start + _CONVLSTM_PATCH_SIZE

                patch_array = spatial_input[:, row_start:row_end, col_start:col_end]  # (T,256,256)
                # ConvLSTM expects (B, T, C, H, W) → (1, T, 1, 256, 256)
                patch_tensor = (
                    torch.from_numpy(patch_array)
                    .unsqueeze(0).unsqueeze(2)
                    .to(device).float()
                )

                predicted_logits  = model(patch_tensor).squeeze().cpu().numpy()   # (256, 256)
                patch_probability = 1.0 / (1.0 + np.exp(-predicted_logits))       # sigmoid

                prediction_sum_map[row_start:row_end, col_start:col_end]   += patch_probability
                prediction_count_map[row_start:row_end, col_start:col_end] += 1

    valid_pixels    = prediction_count_map > 0
    probability_map = np.zeros((full_height, full_width), dtype=np.float32)
    probability_map[valid_pixels] = (
        prediction_sum_map[valid_pixels] / prediction_count_map[valid_pixels]
    )

    # Crop back to original dimensions before padding
    probability_map = probability_map[:orig_height, :orig_width]
    predicted_mask  = (probability_map > 0.5).astype(np.uint8)
    return probability_map, predicted_mask


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
