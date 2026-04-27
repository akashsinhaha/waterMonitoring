"""
Time Series Analysis — Water Body Area & Quality Change (2017–2025)
Same location, annual March–June composites.

Outputs (saved to RESULTS_DIR):
  water_masks/water_mask_{year}.tif     — binary water mask per year
  timeseries_summary.csv                — area + quality stats per year
  01_water_masks_grid.png               — all 9 water masks in a grid
  02_area_timeseries.png                — water area (km²) over time
  03_change_maps.png                    — year-over-year gain/loss maps
  04_quality_timeseries.png             — water quality indices over time
  05_summary_dashboard.png              — combined overview

Run:
    python water_timeseries.py
"""

import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR_TS  = r'D:\rfWater\DATA_DIR_TS'
MODEL_PATH   = r'D:\rfWater\results_unet\unet_best.pth'
RESULTS_DIR  = r'D:\rfWater\results_timeseries'

YEARS        = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
PIXEL_AREA_M2 = 10 * 10          # 10m resolution → 100 m² per pixel
PIXEL_AREA_KM2 = PIXEL_AREA_M2 / 1e6

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# New S2 has 10 bands: B2(0) B3(1) B4(2) B5(3) B6(4) B7(5) B8(6) B8A(7) B11(8) B12(9)
# U-Net was trained on 7 channels: B2, B3, B4, B8, B11, S1_ASC, S1_DESC
# Extract these S2 indices from the 10-band file:
S2_UNET_IDX = [0, 1, 2, 6, 8]   # → B2, B3, B4, B8, B11

# Old training data was raw DN (0–10000); new data is 0–1 reflectance.
# Multiply S2 by this factor before feeding to U-Net:
S2_SCALE_FACTOR = 10000.0

PATCH_SIZE = 256
STRIDE     = 128


# ── U-NET ARCHITECTURE (must match saved model) ───────────────────────────────

class DoubleConv(torch.nn.Module):
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

class EncoderBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = torch.nn.MaxPool2d(2, 2)
    def forward(self, x):
        skip = self.conv(x)
        return skip, self.pool(skip)

class DecoderBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = torch.nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv     = DoubleConv(in_channels, out_channels)
    def forward(self, x, skip):
        x = self.upsample(x)
        return self.conv(torch.cat([skip, x], dim=1))

class UNet(torch.nn.Module):
    def __init__(self, in_channels=7, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoder1   = EncoderBlock(in_channels, features[0])
        self.encoder2   = EncoderBlock(features[0], features[1])
        self.encoder3   = EncoderBlock(features[1], features[2])
        self.encoder4   = EncoderBlock(features[2], features[3])
        self.bottleneck = DoubleConv(features[3], features[3] * 2)
        self.decoder4   = DecoderBlock(features[3] * 2, features[3])
        self.decoder3   = DecoderBlock(features[3],     features[2])
        self.decoder2   = DecoderBlock(features[2],     features[1])
        self.decoder1   = DecoderBlock(features[1],     features[0])
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
        return self.final_conv(x)   # raw logits


# ── DATA LOADING ──────────────────────────────────────────────────────────────

def load_year_ts(year):
    """
    Load new 10-band S2 + S1 ASC/DESC for a given year.

    Returns
    -------
    s2_full  : np.ndarray (10, H, W) float32  — 0-1 reflectance, all 10 S2 bands
    s2_unet  : np.ndarray (5,  H, W) float32  — 5 S2 bands scaled to DN for U-Net
    s1_asc   : np.ndarray (H, W)     float32
    s1_desc  : np.ndarray (H, W)     float32
    transform: rasterio Affine
    crs      : rasterio CRS
    """
    s2_path      = os.path.join(DATA_DIR_TS, f'S2_March_June_{year}.tif')
    s1_asc_path  = os.path.join(DATA_DIR_TS, f'S1_ASC_March_June_{year}.tif')
    s1_desc_path = os.path.join(DATA_DIR_TS, f'S1_DESC_March_June_{year}.tif')

    with rasterio.open(s2_path) as src:
        s2_full   = src.read().astype(np.float32)   # (10, H, W)  0–1
        transform = src.transform
        crs       = src.crs
        height    = src.height
        width     = src.width

    # Clean NoData
    s2_full = np.nan_to_num(s2_full, nan=0.0, posinf=0.0, neginf=0.0)

    # Extract 5 bands for U-Net and scale back to DN range
    s2_unet = s2_full[S2_UNET_IDX] * S2_SCALE_FACTOR   # (5, H, W)

    # Reproject S1 onto S2 grid
    def reproject_s1(path):
        with rasterio.open(path) as src:
            dst = np.zeros((height, width), dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=crs,
                resampling=Resampling.bilinear,
            )
        return np.nan_to_num(dst, nan=0.0, posinf=0.0, neginf=0.0)

    s1_asc  = reproject_s1(s1_asc_path)
    s1_desc = reproject_s1(s1_desc_path)

    return s2_full, s2_unet, s1_asc, s1_desc, transform, crs


# ── WATER MASK GENERATION ─────────────────────────────────────────────────────

def load_model():
    model = UNet(in_channels=7).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f'Model loaded from {MODEL_PATH}')
    return model


def predict_water_mask(model, s2_unet, s1_asc, s1_desc):
    """
    Run U-Net on a full scene using patch-based inference with overlap averaging.

    Input stack: (7, H, W) — [B2, B3, B4, B8, B11, S1_ASC, S1_DESC] scaled to DN
    """
    image_stack = np.vstack([
        s2_unet,
        s1_asc[np.newaxis],
        s1_desc[np.newaxis],
    ])   # (7, H, W)

    _, height, width = image_stack.shape
    prediction_sum   = np.zeros((height, width), dtype=np.float32)
    prediction_count = np.zeros((height, width), dtype=np.float32)

    with torch.no_grad():
        for row in range(0, height - PATCH_SIZE + 1, STRIDE):
            for col in range(0, width - PATCH_SIZE + 1, STRIDE):
                patch = image_stack[:, row:row+PATCH_SIZE, col:col+PATCH_SIZE]
                tensor = torch.from_numpy(patch).unsqueeze(0).to(DEVICE)
                logits = model(tensor).squeeze().cpu().numpy()
                prob   = 1 / (1 + np.exp(-logits))   # sigmoid

                prediction_sum[row:row+PATCH_SIZE, col:col+PATCH_SIZE]   += prob
                prediction_count[row:row+PATCH_SIZE, col:col+PATCH_SIZE] += 1

    valid = prediction_count > 0
    prob_map = np.zeros((height, width), dtype=np.float32)
    prob_map[valid] = prediction_sum[valid] / prediction_count[valid]
    water_mask = (prob_map > 0.5).astype(np.uint8)
    return water_mask, prob_map


def generate_all_masks(model):
    """Generate and save water masks for all years. Skip if already exists."""
    masks_dir = os.path.join(RESULTS_DIR, 'water_masks')
    os.makedirs(masks_dir, exist_ok=True)

    all_masks = {}

    for year in YEARS:
        mask_path = os.path.join(masks_dir, f'water_mask_{year}.tif')

        if os.path.exists(mask_path):
            print(f'  {year}: mask already exists, loading...')
            with rasterio.open(mask_path) as src:
                all_masks[year] = src.read(1)
            continue

        print(f'  {year}: generating water mask...')
        s2_full, s2_unet, s1_asc, s1_desc, transform, crs = load_year_ts(year)
        water_mask, _ = predict_water_mask(model, s2_unet, s1_asc, s1_desc)
        all_masks[year] = water_mask

        with rasterio.open(
            mask_path, 'w', driver='GTiff',
            height=water_mask.shape[0], width=water_mask.shape[1],
            count=1, dtype='uint8', crs=crs, transform=transform, compress='lzw',
        ) as dst:
            dst.write(water_mask[np.newaxis])

        water_pct = water_mask.mean() * 100
        print(f'       water coverage: {water_pct:.2f}%')

    return all_masks


# ── QUALITY INDICES ───────────────────────────────────────────────────────────

def safe_index(a, b):
    """Normalised difference (a-b)/(a+b) with division-by-zero protection."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = (a - b) / (a + b)
        result[~np.isfinite(result)] = 0
    return result


def compute_quality_indices(s2_full, water_mask):
    """
    Compute water quality indices within water pixels only.
    s2_full: (10, H, W) float32 in 0-1 reflectance scale.

    New S2 band layout:
      0=B2  1=B3  2=B4  3=B5  4=B6  5=B7  6=B8  7=B8A  8=B11  9=B12
    """
    B2, B3, B4  = s2_full[0], s2_full[1], s2_full[2]
    B5, B6      = s2_full[3], s2_full[4]
    B8, B11, B12 = s2_full[6], s2_full[8], s2_full[9]

    water = water_mask.astype(bool)

    def masked_mean(index_map):
        values = index_map[water]
        values = values[np.isfinite(values)]
        return float(np.mean(values)) if len(values) > 0 else np.nan

    indices = {
        'mndwi'   : safe_index(B3, B11),           # water extent quality
        'ndti'    : safe_index(B4, B3),             # turbidity (higher=more turbid)
        'ndci'    : safe_index(B5, B4),             # chlorophyll-a
        'clarity' : np.where(B4 > 0, B2 / (B4 + 1e-8), 0),  # water transparency
        'algae'   : B8 - B4,                        # floating algae
        'sediment': B4,                             # suspended sediment proxy
    }

    return {name: masked_mean(idx_map) for name, idx_map in indices.items()}


# ── ANALYSIS ──────────────────────────────────────────────────────────────────

def build_summary(all_masks):
    """Build a DataFrame with area + quality stats per year."""
    print('\nComputing area and quality statistics...')
    rows = []

    for year in YEARS:
        water_mask = all_masks[year]
        water_pixels = int(water_mask.sum())
        water_area_km2 = water_pixels * PIXEL_AREA_KM2

        s2_full, _, _, _, _, _ = load_year_ts(year)
        quality = compute_quality_indices(s2_full, water_mask)

        row = {'year': year, 'water_pixels': water_pixels,
               'water_area_km2': water_area_km2}
        row.update(quality)
        rows.append(row)
        print(f'  {year}: {water_area_km2:.2f} km²  '
              f'NDTI={quality["ndti"]:.3f}  NDCI={quality["ndci"]:.3f}')

    df = pd.DataFrame(rows).set_index('year')

    # Year-over-year area change
    df['area_change_km2'] = df['water_area_km2'].diff()
    df['area_change_pct'] = df['water_area_km2'].pct_change() * 100

    csv_path = os.path.join(RESULTS_DIR, 'timeseries_summary.csv')
    df.to_csv(csv_path)
    print(f'\nSummary saved: {csv_path}')
    return df


# ── VISUALISATIONS ────────────────────────────────────────────────────────────

def make_rgb(year):
    """Return a display RGB from S2 B4/B3/B2 for the given year."""
    s2_full, _, _, _, _, _ = load_year_ts(year)
    rgb = np.stack([s2_full[2], s2_full[1], s2_full[0]], axis=-1)
    for ch in range(3):
        low, high = np.percentile(rgb[:, :, ch][rgb[:, :, ch] > 0], [2, 98])
        rgb[:, :, ch] = np.clip((rgb[:, :, ch] - low) / (high - low + 1e-8), 0, 1)
    return rgb


def plot_water_masks_grid(all_masks):
    """3×3 grid of water masks for all 9 years."""
    print('Plotting water mask grid...')
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('Water Body Masks — Annual (2017–2025)', fontsize=16, fontweight='bold')

    for ax, year in zip(axes.flat, YEARS):
        ax.imshow(all_masks[year], cmap='Blues', vmin=0, vmax=1, interpolation='nearest')
        area = all_masks[year].sum() * PIXEL_AREA_KM2
        ax.set_title(f'{year}  ({area:.2f} km²)', fontsize=11)
        ax.axis('off')

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, '01_water_masks_grid.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


def plot_area_timeseries(df):
    """Water area time series with change annotations."""
    print('Plotting area time series...')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    fig.suptitle('Water Body Area Change Over Time (2017–2025)',
                 fontsize=14, fontweight='bold')

    years = df.index.tolist()
    areas = df['water_area_km2'].tolist()

    # Top: absolute area
    ax1.plot(years, areas, 'o-', color='#0077b6', linewidth=2.5,
             markersize=8, markerfacecolor='white', markeredgewidth=2)
    ax1.fill_between(years, areas, alpha=0.15, color='#0077b6')
    for y, a in zip(years, areas):
        ax1.annotate(f'{a:.2f}', (y, a), textcoords='offset points',
                     xytext=(0, 10), ha='center', fontsize=8)
    ax1.set_ylabel('Water Area (km²)', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Absolute Water Area', fontsize=11)

    # Bottom: year-over-year change
    changes = df['area_change_km2'].fillna(0).tolist()
    colors  = ['#ef233c' if c < 0 else '#2dc653' for c in changes]
    ax2.bar(years, changes, color=colors, width=0.6, edgecolor='white')
    ax2.axhline(0, color='black', linewidth=0.8)
    for y, c in zip(years, changes):
        if c != 0:
            ax2.annotate(f'{c:+.2f}', (y, c),
                         textcoords='offset points',
                         xytext=(0, 5 if c >= 0 else -15),
                         ha='center', fontsize=8)
    ax2.set_ylabel('Change vs Previous Year (km²)', fontsize=11)
    ax2.set_xlabel('Year', fontsize=11)
    ax2.set_xticks(years)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_title('Year-over-Year Change  (green=gain, red=loss)', fontsize=11)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, '02_area_timeseries.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


def plot_change_maps(all_masks):
    """Year-over-year change detection maps (8 pairs for 9 years)."""
    print('Plotting change detection maps...')
    pairs    = [(YEARS[i], YEARS[i+1]) for i in range(len(YEARS)-1)]
    diff_cmap  = mcolors.ListedColormap(['#1a1a2e', '#00b4d8', '#2dc653', '#ef233c'])
    # 0=TN(stable non-water), 1=TP(stable water), 2=gain, 3=loss

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle('Year-over-Year Water Body Change', fontsize=14, fontweight='bold')

    legend_elements = [
        Patch(facecolor='#1a1a2e', label='Stable non-water'),
        Patch(facecolor='#00b4d8', label='Stable water'),
        Patch(facecolor='#2dc653', label='Water gain'),
        Patch(facecolor='#ef233c', label='Water loss'),
    ]

    for ax, (yr_a, yr_b) in zip(axes.flat, pairs):
        mask_a = all_masks[yr_a]
        mask_b = all_masks[yr_b]

        diff = np.zeros_like(mask_a, dtype=np.uint8)
        diff[(mask_a == 1) & (mask_b == 1)] = 1   # stable water
        diff[(mask_a == 0) & (mask_b == 1)] = 2   # gain
        diff[(mask_a == 1) & (mask_b == 0)] = 3   # loss

        gain_km2 = float((diff == 2).sum() * PIXEL_AREA_KM2)
        loss_km2 = float((diff == 3).sum() * PIXEL_AREA_KM2)

        ax.imshow(diff, cmap=diff_cmap, vmin=0, vmax=3, interpolation='nearest')
        ax.set_title(f'{yr_a} → {yr_b}\n'
                     f'Gain: +{gain_km2:.2f} km²  Loss: -{loss_km2:.2f} km²',
                     fontsize=9)
        ax.axis('off')

    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, '03_change_maps.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


def plot_quality_timeseries(df):
    """Multi-panel water quality indices over time."""
    print('Plotting quality time series...')
    years = df.index.tolist()

    metrics = [
        ('ndti',     'NDTI (Turbidity)',       '#e07b39', 'Higher = more turbid'),
        ('ndci',     'NDCI (Chlorophyll-a)',   '#2dc653', 'Higher = more algae'),
        ('clarity',  'Water Clarity (B2/B4)',  '#00b4d8', 'Higher = clearer'),
        ('algae',    'Algae Index (B8-B4)',    '#9b5de5', 'Higher = more algae/vegetation'),
        ('sediment', 'Sediment (B4 mean)',     '#f77f00', 'Higher = more suspended sediment'),
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(13, 16), sharex=True)
    fig.suptitle('Water Quality Indices Over Time (2017–2025)',
                 fontsize=14, fontweight='bold')

    for ax, (col, label, color, note) in zip(axes, metrics):
        values = df[col].tolist()
        ax.plot(years, values, 'o-', color=color, linewidth=2.2,
                markersize=7, markerfacecolor='white', markeredgewidth=2)
        ax.fill_between(years, values, alpha=0.12, color=color)

        # Trend line
        valid_idx = [i for i, v in enumerate(values) if not np.isnan(v)]
        if len(valid_idx) > 2:
            valid_x = [years[i] for i in valid_idx]
            valid_y = [values[i] for i in valid_idx]
            z = np.polyfit(valid_x, valid_y, 1)
            trend = np.poly1d(z)
            ax.plot(valid_x, trend(valid_x), '--', color=color,
                    alpha=0.5, linewidth=1.5, label=f'Trend ({z[0]:+.4f}/yr)')

        ax.set_ylabel(label, fontsize=10)
        ax.set_title(f'{label}  —  {note}', fontsize=10)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(years)

    axes[-1].set_xlabel('Year', fontsize=11)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, '04_quality_timeseries.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


def plot_summary_dashboard(df, all_masks):
    """Combined summary: RGB + water mask for first and last year, + key charts."""
    print('Plotting summary dashboard...')
    first_year, last_year = YEARS[0], YEARS[-1]

    fig = plt.figure(figsize=(22, 16))
    fig.suptitle(f'Water Body Monitor — Summary Dashboard\n'
                 f'Location: Same AOI | Period: {first_year}–{last_year} (March–June)',
                 fontsize=14, fontweight='bold')

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    # Row 0: RGB and water mask for first and last year
    for col_idx, year in enumerate([first_year, last_year]):
        rgb = make_rgb(year)
        ax_rgb  = fig.add_subplot(gs[0, col_idx * 2])
        ax_mask = fig.add_subplot(gs[0, col_idx * 2 + 1])
        ax_rgb.imshow(rgb)
        ax_rgb.set_title(f'RGB {year}', fontsize=10)
        ax_rgb.axis('off')
        ax_mask.imshow(all_masks[year], cmap='Blues', vmin=0, vmax=1)
        area = all_masks[year].sum() * PIXEL_AREA_KM2
        ax_mask.set_title(f'Water Mask {year} ({area:.2f} km²)', fontsize=10)
        ax_mask.axis('off')

    # Row 1: area chart
    ax_area = fig.add_subplot(gs[1, :])
    years = df.index.tolist()
    areas = df['water_area_km2'].tolist()
    ax_area.plot(years, areas, 'o-', color='#0077b6', linewidth=2.5, markersize=8,
                 markerfacecolor='white', markeredgewidth=2)
    ax_area.fill_between(years, areas, alpha=0.15, color='#0077b6')
    for y, a in zip(years, areas):
        ax_area.annotate(f'{a:.2f}', (y, a), textcoords='offset points',
                         xytext=(0, 8), ha='center', fontsize=8)
    total_change = areas[-1] - areas[0]
    pct_change   = (total_change / areas[0]) * 100 if areas[0] > 0 else 0
    ax_area.set_title(f'Water Area (km²) | Total change: {total_change:+.2f} km² '
                      f'({pct_change:+.1f}%) from {first_year} to {last_year}',
                      fontsize=11)
    ax_area.set_ylabel('km²')
    ax_area.set_xticks(years)
    ax_area.grid(True, alpha=0.3)

    # Row 2: NDTI and NDCI
    for col_idx, (col, label, color) in enumerate([
        ('ndti', 'Turbidity (NDTI)', '#e07b39'),
        ('ndci', 'Chlorophyll (NDCI)', '#2dc653'),
        ('clarity', 'Clarity (B2/B4)', '#00b4d8'),
        ('sediment', 'Sediment (B4)', '#f77f00'),
    ]):
        ax = fig.add_subplot(gs[2, col_idx])
        values = df[col].tolist()
        ax.plot(years, values, 'o-', color=color, linewidth=2,
                markersize=5, markerfacecolor='white', markeredgewidth=1.5)
        ax.set_title(label, fontsize=10)
        ax.set_xticks(years)
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.grid(True, alpha=0.3)

    path = os.path.join(RESULTS_DIR, '05_summary_dashboard.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print('=' * 60)
    print('WATER BODY TIME SERIES ANALYSIS')
    print(f'Years  : {YEARS[0]}–{YEARS[-1]}')
    print(f'Device : {DEVICE}')
    print('=' * 60)

    # Step 1: Generate water masks for all years
    print('\n── Step 1: Water Mask Generation ──')
    model = load_model()
    all_masks = generate_all_masks(model)

    # Step 2: Build area + quality summary
    print('\n── Step 2: Area & Quality Analysis ──')
    df = build_summary(all_masks)

    print('\n── Summary Table ──')
    print(df[['water_area_km2', 'area_change_km2', 'area_change_pct',
              'ndti', 'ndci', 'clarity']].round(4).to_string())

    # Step 3: Visualisations
    print('\n── Step 3: Generating Visualisations ──')
    plot_water_masks_grid(all_masks)
    plot_area_timeseries(df)
    plot_change_maps(all_masks)
    plot_quality_timeseries(df)
    plot_summary_dashboard(df, all_masks)

    print(f'\nAll outputs saved to: {RESULTS_DIR}')
    print('Done.')


if __name__ == '__main__':
    main()
