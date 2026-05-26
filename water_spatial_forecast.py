"""
Spatial Temporal Forecasting — Predict next year's water mask using ConvLSTM.

Given a sequence of annual water masks (e.g., 2023, 2024, 2025), a ConvLSTM
learns spatial-temporal patterns and predicts the water mask for 2026.

Prerequisites:
    Run water_timeseries.py first — it generates the per-year masks in:
        results_timeseries/water_masks/water_mask_{year}.tif

Outputs (saved to results_timeseries/forecast/):
    convlstm_best.pth                  — trained ConvLSTM weights
    predicted_mask_2026.tif            — predicted binary water mask
    07_spatial_forecast_2026.png       — input masks + prediction side-by-side
    08_change_forecast_2026.png        — predicted change map vs last observed year

Run:
    python water_spatial_forecast.py
"""

import os
import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# ── CONFIG ────────────────────────────────────────────────────────────────────
MASKS_DIR       = r'D:\rfWater\results_timeseries\water_masks'
FORECAST_DIR    = r'D:\rfWater\results_timeseries\forecast'

ALL_YEARS       = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
SEQUENCE_LENGTH = 3       # number of consecutive input years per training sample
FORECAST_YEAR   = 2026    # year to predict

PATCH_SIZE      = 256
STRIDE          = 128

BATCH_SIZE      = 4
LEARNING_RATE   = 1e-3
NUM_EPOCHS      = 30
HIDDEN_CHANNELS = 16      # kept small given limited training sequences
KERNEL_SIZE     = 3

PIXEL_AREA_KM2  = 10 * 10 / 1e6
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── CONVLSTM ─────────────────────────────────────────────────────────────────

class ConvLSTMCell(nn.Module):
    """
    Single ConvLSTM cell operating in 2D spatial space.

    Processes one timestep and updates the spatial hidden and cell states.

    Input  : (B, input_channels,  H, W)
    Hidden : (B, hidden_channels, H, W)  — h and c separately
    Output : updated (h_next, c_next)
    """

    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        # Single fused convolution for all four gates (i, f, g, o)
        self.fused_gates_conv = nn.Conv2d(
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

    def init_hidden(self, batch_size, spatial_height, spatial_width):
        zeros = torch.zeros(batch_size, self.hidden_channels, spatial_height, spatial_width,
                            device=DEVICE)
        return zeros, zeros.clone()


class ConvLSTMForecaster(nn.Module):
    """
    Sequence-to-one ConvLSTM: takes T consecutive water masks → predicts next mask.

    Input  : (B, T, 1, H, W)  — T binary water masks as float32
    Output : (B, 1, H, W)     — predicted next mask (raw logits, apply sigmoid for prob)
    """

    def __init__(self, hidden_channels=HIDDEN_CHANNELS, kernel_size=KERNEL_SIZE):
        super().__init__()
        self.convlstm_cell = ConvLSTMCell(
            input_channels=1,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
        )
        self.prediction_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, 1, kernel_size=1),
        )

    def forward(self, sequence_tensor):
        batch_size, seq_len, _, spatial_height, spatial_width = sequence_tensor.shape
        h_state, c_state = self.convlstm_cell.init_hidden(batch_size, spatial_height, spatial_width)

        for timestep_idx in range(seq_len):
            current_frame = sequence_tensor[:, timestep_idx]   # (B, 1, H, W)
            h_state, c_state = self.convlstm_cell(current_frame, (h_state, c_state))

        return self.prediction_head(h_state)   # (B, 1, H, W) logits


# ── DATASET ───────────────────────────────────────────────────────────────────

def load_all_annual_masks():
    """
    Load all annual binary water masks from disk.
    Returns dict: year (int) → ndarray (H, W) float32 with values 0.0 or 1.0.
    """
    annual_masks = {}
    for year in ALL_YEARS:
        mask_path = os.path.join(MASKS_DIR, f'water_mask_{year}.tif')
        if not os.path.exists(mask_path):
            raise FileNotFoundError(
                f'Mask not found: {mask_path}\n'
                'Run water_timeseries.py first to generate annual water masks.'
            )
        with rasterio.open(mask_path) as src:
            annual_masks[year] = src.read(1).astype(np.float32)   # 0.0 or 1.0

    reference_shape = list(annual_masks.values())[0].shape
    print(f'Loaded {len(annual_masks)} annual masks  |  spatial shape: {reference_shape}')
    return annual_masks


class TemporalMaskDataset(Dataset):
    """
    Sliding-window patch dataset over annual water masks.

    For SEQUENCE_LENGTH=3 and years 2017–2025, windows are:
        (2017,2018,2019)→2020, (2018,2019,2020)→2021, ..., (2022,2023,2024)→2025
    That's 6 sequence windows. Each window is further split into 256×256 patches
    with STRIDE overlap, yielding many spatial training samples despite the small
    temporal dataset.

    Returns per item:
        input_sequence_tensor : (T, 1, H, W) float32
        target_mask_tensor    : (1,    H, W) float32
    """

    def __init__(self, annual_masks, training_years):
        self.patch_pairs = []   # list of (input_array (T,H,W), target_array (H,W))

        reference_mask          = annual_masks[training_years[0]]
        full_height, full_width = reference_mask.shape

        # Build sliding windows over the year sequence
        for window_start in range(len(training_years) - SEQUENCE_LENGTH):
            input_years = training_years[window_start : window_start + SEQUENCE_LENGTH]
            target_year = training_years[window_start + SEQUENCE_LENGTH]

            input_volume  = np.stack([annual_masks[y] for y in input_years], axis=0)  # (T, H, W)
            target_volume = annual_masks[target_year]                                  # (H, W)

            for row_start in range(0, full_height - PATCH_SIZE + 1, STRIDE):
                for col_start in range(0, full_width - PATCH_SIZE + 1, STRIDE):
                    row_end = row_start + PATCH_SIZE
                    col_end = col_start + PATCH_SIZE

                    input_patch  = input_volume[:, row_start:row_end, col_start:col_end]  # (T,256,256)
                    target_patch = target_volume[row_start:row_end, col_start:col_end]     # (256,256)

                    self.patch_pairs.append((input_patch, target_patch))

        num_windows = len(training_years) - SEQUENCE_LENGTH
        print(f'Dataset: {num_windows} temporal windows → {len(self.patch_pairs)} patch pairs')

    def __len__(self):
        return len(self.patch_pairs)

    def __getitem__(self, idx):
        input_array, target_array = self.patch_pairs[idx]

        # (T, H, W) → (T, 1, H, W) to match ConvLSTM's channel dimension
        input_sequence_tensor = torch.from_numpy(input_array).unsqueeze(1).float()   # (T, 1, 256, 256)
        target_mask_tensor    = torch.from_numpy(target_array).unsqueeze(0).float()  # (1,    256, 256)

        return input_sequence_tensor, target_mask_tensor


# ── TRAINING ─────────────────────────────────────────────────────────────────

def _compute_batch_metrics(predicted_logits, target_mask_batch):
    """
    Compute IoU, precision, recall, and F1 for a single batch.
    All metrics are for the water (positive) class.
    """
    predicted_binary = (torch.sigmoid(predicted_logits) > 0.5).float()
    target_flat      = target_mask_batch.view(-1)
    predicted_flat   = predicted_binary.view(-1)

    true_positives  = (predicted_flat * target_flat).sum().item()
    false_positives = (predicted_flat * (1 - target_flat)).sum().item()
    false_negatives = ((1 - predicted_flat) * target_flat).sum().item()
    true_negatives  = ((1 - predicted_flat) * (1 - target_flat)).sum().item()

    total_pixels = true_positives + true_negatives + false_positives + false_negatives
    accuracy  = (true_positives + true_negatives) / (total_pixels + 1e-6)
    iou       = true_positives / (true_positives + false_positives + false_negatives + 1e-6)
    precision = true_positives / (true_positives + false_positives + 1e-6)
    recall    = true_positives / (true_positives + false_negatives + 1e-6)
    f1_score  = 2 * precision * recall / (precision + recall + 1e-6)

    return accuracy, iou, precision, recall, f1_score


def train_convlstm(model, train_loader):
    """Train ConvLSTM with BCE loss, logging per-epoch loss and accuracy metrics."""
    optimizer       = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    lr_scheduler    = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    bce_loss_fn     = nn.BCEWithLogitsLoss()
    best_epoch_loss = float('inf')
    best_model_path = os.path.join(FORECAST_DIR, 'convlstm_best.pth')

    print(f'\nTraining ConvLSTM on {DEVICE}')
    print(f'{"Epoch":>7} | {"Loss":>8} | {"Acc":>7} | {"IoU":>7} | {"Precision":>10} | {"Recall":>8} | {"F1":>7} | {"LR":>10} |')
    print('-' * 85)

    for epoch_num in range(1, NUM_EPOCHS + 1):
        model.train()
        cumulative_loss      = 0.0
        cumulative_accuracy  = 0.0
        cumulative_iou       = 0.0
        cumulative_precision = 0.0
        cumulative_recall    = 0.0
        cumulative_f1        = 0.0
        num_batches          = 0

        for input_sequence_batch, target_mask_batch in train_loader:
            input_sequence_batch = input_sequence_batch.to(DEVICE)
            target_mask_batch    = target_mask_batch.to(DEVICE)

            optimizer.zero_grad()
            predicted_logits = model(input_sequence_batch)
            batch_loss       = bce_loss_fn(predicted_logits, target_mask_batch)
            batch_loss.backward()
            optimizer.step()

            cumulative_loss += batch_loss.item()

            batch_accuracy, batch_iou, batch_precision, batch_recall, batch_f1 = \
                _compute_batch_metrics(predicted_logits.detach(), target_mask_batch)
            cumulative_accuracy  += batch_accuracy
            cumulative_iou       += batch_iou
            cumulative_precision += batch_precision
            cumulative_recall    += batch_recall
            cumulative_f1        += batch_f1
            num_batches          += 1

        epoch_avg_loss      = cumulative_loss      / num_batches
        epoch_avg_accuracy  = cumulative_accuracy  / num_batches
        epoch_avg_iou       = cumulative_iou       / num_batches
        epoch_avg_precision = cumulative_precision / num_batches
        epoch_avg_recall    = cumulative_recall    / num_batches
        epoch_avg_f1        = cumulative_f1        / num_batches
        current_lr          = optimizer.param_groups[0]['lr']

        lr_scheduler.step(epoch_avg_loss)

        is_best     = epoch_avg_loss < best_epoch_loss
        best_marker = ' ★' if is_best else ''
        print(
            f'{epoch_num:>3}/{NUM_EPOCHS} | '
            f'{epoch_avg_loss:>8.4f} | '
            f'{epoch_avg_accuracy:>7.4f} | '
            f'{epoch_avg_iou:>7.4f} | '
            f'{epoch_avg_precision:>10.4f} | '
            f'{epoch_avg_recall:>8.4f} | '
            f'{epoch_avg_f1:>7.4f} | '
            f'{current_lr:>10.2e} |'
            f'{best_marker}'
        )

        if is_best:
            best_epoch_loss = epoch_avg_loss
            torch.save(model.state_dict(), best_model_path)

    print('-' * 85)
    print(f'Training complete. Best loss: {best_epoch_loss:.4f} | Model: {best_model_path}')
    return model


# ── INFERENCE ─────────────────────────────────────────────────────────────────

def predict_next_annual_mask(model, annual_masks, input_sequence_years):
    """
    Run the trained ConvLSTM on the full image using patch-based overlap-averaging.

    Parameters
    ----------
    model                : trained ConvLSTMForecaster
    annual_masks         : dict year → (H, W) float32
    input_sequence_years : list of SEQUENCE_LENGTH years to use as input

    Returns
    -------
    probability_map  : (H, W) float32  — water probability [0, 1]
    predicted_mask   : (H, W) uint8    — thresholded at 0.5
    """
    model.eval()

    reference_mask     = annual_masks[input_sequence_years[0]]
    full_height, full_width = reference_mask.shape

    spatial_input = np.stack([annual_masks[y] for y in input_sequence_years], axis=0)  # (T, H, W)

    prediction_sum_map   = np.zeros((full_height, full_width), dtype=np.float32)
    prediction_count_map = np.zeros((full_height, full_width), dtype=np.float32)

    with torch.no_grad():
        for row_start in range(0, full_height - PATCH_SIZE + 1, STRIDE):
            for col_start in range(0, full_width - PATCH_SIZE + 1, STRIDE):
                row_end = row_start + PATCH_SIZE
                col_end = col_start + PATCH_SIZE

                patch_array = spatial_input[:, row_start:row_end, col_start:col_end]  # (T, 256, 256)
                # Shape needed: (1, T, 1, 256, 256)
                patch_tensor = torch.from_numpy(patch_array).unsqueeze(0).unsqueeze(2).to(DEVICE).float()

                predicted_logits = model(patch_tensor).squeeze().cpu().numpy()          # (256, 256)
                patch_probability = 1.0 / (1.0 + np.exp(-predicted_logits))            # sigmoid

                prediction_sum_map[row_start:row_end, col_start:col_end]   += patch_probability
                prediction_count_map[row_start:row_end, col_start:col_end] += 1

    valid_pixels      = prediction_count_map > 0
    probability_map   = np.zeros((full_height, full_width), dtype=np.float32)
    probability_map[valid_pixels] = (
        prediction_sum_map[valid_pixels] / prediction_count_map[valid_pixels]
    )

    predicted_mask = (probability_map > 0.5).astype(np.uint8)
    return probability_map, predicted_mask


# ── VISUALISATION ─────────────────────────────────────────────────────────────

def plot_spatial_forecast(annual_masks, predicted_mask, input_sequence_years, forecast_year):
    """
    Side-by-side: last SEQUENCE_LENGTH observed masks + predicted mask.
    Also generates a change map comparing last observed vs predicted.
    """
    print('Plotting spatial forecast visualizations...')

    # Panel 1: observed sequence + prediction
    total_panels = SEQUENCE_LENGTH + 1
    fig, panel_axes = plt.subplots(1, total_panels, figsize=(5 * total_panels, 6))
    fig.suptitle(f'Spatial Temporal Forecast — {forecast_year} Prediction (ConvLSTM)',
                 fontsize=14, fontweight='bold')

    for ax, year in zip(panel_axes[:-1], input_sequence_years):
        observed_mask = annual_masks[year]
        observed_area = observed_mask.sum() * PIXEL_AREA_KM2
        ax.imshow(observed_mask, cmap='Blues', vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(f'{year}  (Observed)\n{observed_area:.2f} km²', fontsize=11)
        ax.axis('off')

    predicted_area = predicted_mask.sum() * PIXEL_AREA_KM2
    panel_axes[-1].imshow(predicted_mask, cmap='Reds', vmin=0, vmax=1, interpolation='nearest')
    panel_axes[-1].set_title(f'{forecast_year}  (Predicted)\n{predicted_area:.2f} km²',
                              fontsize=11, color='#c1121f', fontweight='bold')
    panel_axes[-1].axis('off')

    plt.tight_layout()
    forecast_panel_path = os.path.join(FORECAST_DIR, f'07_spatial_forecast_{forecast_year}.png')
    plt.savefig(forecast_panel_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {forecast_panel_path}')

    # Panel 2: change map (last observed → predicted)
    last_observed_year = input_sequence_years[-1]
    last_observed_mask = annual_masks[last_observed_year]

    change_category_map = np.zeros_like(predicted_mask, dtype=np.uint8)
    change_category_map[(last_observed_mask == 1) & (predicted_mask == 1)] = 1  # stable water
    change_category_map[(last_observed_mask == 0) & (predicted_mask == 1)] = 2  # predicted gain
    change_category_map[(last_observed_mask == 1) & (predicted_mask == 0)] = 3  # predicted loss

    predicted_gain_km2 = float((change_category_map == 2).sum() * PIXEL_AREA_KM2)
    predicted_loss_km2 = float((change_category_map == 3).sum() * PIXEL_AREA_KM2)

    change_colormap = ListedColormap(['#1a1a2e', '#00b4d8', '#2dc653', '#ef233c'])

    fig2, change_ax = plt.subplots(figsize=(9, 7))
    change_ax.imshow(change_category_map, cmap=change_colormap, vmin=0, vmax=3,
                     interpolation='nearest')
    change_ax.set_title(
        f'Predicted Change: {last_observed_year} → {forecast_year}\n'
        f'Gain: +{predicted_gain_km2:.2f} km²   Loss: −{predicted_loss_km2:.2f} km²',
        fontsize=12, fontweight='bold',
    )
    change_ax.axis('off')

    legend_elements = [
        Patch(facecolor='#1a1a2e', label='Stable non-water'),
        Patch(facecolor='#00b4d8', label='Stable water'),
        Patch(facecolor='#2dc653', label='Predicted water gain'),
        Patch(facecolor='#ef233c', label='Predicted water loss'),
    ]
    change_ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.9)

    plt.tight_layout()
    change_map_path = os.path.join(FORECAST_DIR, f'08_change_forecast_{forecast_year}.png')
    plt.savefig(change_map_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {change_map_path}')


def save_predicted_mask_as_geotiff(predicted_mask, reference_year):
    """
    Save predicted mask as a GeoTIFF in two locations:
      1. forecast/predicted_mask_2026.tif   — archive copy
      2. water_masks/water_mask_2026.tif    — picked up by the dashboard
    """
    reference_mask_path = os.path.join(MASKS_DIR, f'water_mask_{reference_year}.tif')
    with rasterio.open(reference_mask_path) as reference_src:
        spatial_transform = reference_src.transform
        coordinate_crs    = reference_src.crs

    raster_meta = dict(
        driver='GTiff',
        height=predicted_mask.shape[0],
        width=predicted_mask.shape[1],
        count=1, dtype='uint8',
        crs=coordinate_crs,
        transform=spatial_transform,
        compress='lzw',
    )

    # Archive copy in forecast/
    archive_path = os.path.join(FORECAST_DIR, f'predicted_mask_{FORECAST_YEAR}.tif')
    with rasterio.open(archive_path, 'w', **raster_meta) as dst:
        dst.write(predicted_mask[np.newaxis])
    print(f'  Archive GeoTIFF : {archive_path}')

    # Dashboard copy in water_masks/
    dashboard_path = os.path.join(MASKS_DIR, f'water_mask_{FORECAST_YEAR}.tif')
    with rasterio.open(dashboard_path, 'w', **raster_meta) as dst:
        dst.write(predicted_mask[np.newaxis])
    print(f'  Dashboard mask  : {dashboard_path}')

    return archive_path


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(FORECAST_DIR, exist_ok=True)

    print('=' * 60)
    print('SPATIAL TEMPORAL FORECASTING — ConvLSTM')
    print(f'Historical years : {ALL_YEARS[0]}–{ALL_YEARS[-1]}')
    print(f'Sequence length  : {SEQUENCE_LENGTH}')
    print(f'Forecast year    : {FORECAST_YEAR}')
    print(f'Device           : {DEVICE}')
    print('=' * 60)

    # Step 1: Load all annual water masks
    print('\n── Step 1: Loading annual masks ──')
    annual_masks = load_all_annual_masks()

    # Step 2: Build patch dataset from sliding temporal windows
    print('\n── Step 2: Building temporal patch dataset ──')
    temporal_dataset = TemporalMaskDataset(annual_masks, training_years=ALL_YEARS)

    if len(temporal_dataset) == 0:
        print('ERROR: Dataset is empty. Check MASKS_DIR and ALL_YEARS.')
        return

    train_data_loader = DataLoader(
        temporal_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == 'cuda'),
    )

    # Step 3: Build and train ConvLSTM (or load existing)
    best_model_checkpoint = os.path.join(FORECAST_DIR, 'convlstm_best.pth')
    forecaster_model      = ConvLSTMForecaster(
        hidden_channels=HIDDEN_CHANNELS,
        kernel_size=KERNEL_SIZE,
    ).to(DEVICE)

    total_trainable_params = sum(p.numel() for p in forecaster_model.parameters() if p.requires_grad)
    print(f'\n── Step 3: ConvLSTM — {total_trainable_params:,} trainable parameters ──')

    if os.path.exists(best_model_checkpoint):
        print(f'Loading existing checkpoint: {best_model_checkpoint}')
        forecaster_model.load_state_dict(torch.load(best_model_checkpoint, map_location=DEVICE))
    else:
        forecaster_model = train_convlstm(forecaster_model, train_data_loader)
        forecaster_model.load_state_dict(torch.load(best_model_checkpoint, map_location=DEVICE))

    forecaster_model.eval()

    # Step 4: Predict FORECAST_YEAR using last SEQUENCE_LENGTH observed years
    input_years_for_inference = ALL_YEARS[-SEQUENCE_LENGTH:]   # e.g. [2023, 2024, 2025]

    print(f'\n── Step 4: Predicting {FORECAST_YEAR} ──')
    print(f'  Input sequence: {input_years_for_inference}')

    forecast_probability_map, forecast_predicted_mask = predict_next_annual_mask(
        forecaster_model, annual_masks, input_years_for_inference
    )

    predicted_water_area_km2 = forecast_predicted_mask.sum() * PIXEL_AREA_KM2
    last_observed_area_km2   = annual_masks[ALL_YEARS[-1]].sum() * PIXEL_AREA_KM2

    print(f'\n  Predicted water area ({FORECAST_YEAR}): {predicted_water_area_km2:.2f} km²')
    print(f'  Last observed area  ({ALL_YEARS[-1]}): {last_observed_area_km2:.2f} km²')
    area_delta_km2 = predicted_water_area_km2 - last_observed_area_km2
    print(f'  Predicted change                  : {area_delta_km2:+.2f} km²')

    # Step 5: Save GeoTIFF + visualizations
    print(f'\n── Step 5: Saving outputs ──')
    save_predicted_mask_as_geotiff(forecast_predicted_mask, reference_year=ALL_YEARS[-1])
    plot_spatial_forecast(annual_masks, forecast_predicted_mask,
                          input_years_for_inference, FORECAST_YEAR)

    print(f'\nAll forecast outputs saved to: {FORECAST_DIR}')
    print('Done.')


if __name__ == '__main__':
    main()
