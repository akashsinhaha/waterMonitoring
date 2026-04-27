import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import contextlib
import matplotlib.pyplot as plt
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_DIR        = r'D:\rfWater\DATA_DIR'
PATCHES_DIR     = r'D:\rfWater\patches'
RESULTS_DIR     = r'D:\rfWater\results_unet'

PATCH_SIZE      = 256
STRIDE          = 128
MNDWI_THRESHOLD = 0.0          # pixels with MNDWI > this → water

TRAIN_YEARS     = [2017, 2018, 2019, 2020, 2021, 2022]
TEST_YEARS      = [2023, 2024, 2025]

BATCH_SIZE          = 8
LEARNING_RATE       = 1e-4
NUM_EPOCHS          = 10
BATCHES_PER_EPOCH   = 80   # set to None to use all batches
BATCHES_PER_EVAL    = 40   # set to None to evaluate on full test set
DEVICE              = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Band indices in the 7-band stack:
# 0=B2, 1=B3(Green), 2=B4, 3=B8, 4=B11(SWIR), 5=S1_VV_ASC, 6=S1_VV_DESC
NUM_INPUT_BANDS = 7             # MNDWI excluded from X to avoid leakage


# ─── DATA LOADING ─────────────────────────────────────────────────────────────

def calculate_mndwi(green, swir):
    """Compute MNDWI = (Green - SWIR) / (Green + SWIR)"""
    with np.errstate(divide='ignore', invalid='ignore'):
        mndwi = (green - swir) / (green + swir)
        mndwi[np.isnan(mndwi)] = 0
        mndwi[np.isinf(mndwi)] = 0
    return mndwi


def load_year(year):
    """
    Load and stack all bands for a given year.
    S1 is reprojected to match S2 grid.

    Returns
    -------
    image_stack : np.ndarray  shape (7, H, W)  float32
        Bands: B2, B3, B4, B8, B11, S1_VV_ASC, S1_VV_DESC
    water_mask  : np.ndarray  shape (H, W)      uint8
        Binary mask derived from MNDWI > MNDWI_THRESHOLD
    transform   : rasterio.Affine
    crs         : rasterio CRS
    """
    s2_path      = os.path.join(DATA_DIR, f'S2_March_June_{year}.tif')
    s1_asc_path  = os.path.join(DATA_DIR, f'S1_ASC_March_June_{year}.tif')
    s1_desc_path = os.path.join(DATA_DIR, f'S1_DESC_March_June_{year}.tif')

    # Load Sentinel-2 (reference grid)
    with rasterio.open(s2_path) as src:
        s2_bands  = src.read().astype(np.float32)   # (5, H, W)
        transform = src.transform
        crs       = src.crs
        height    = src.height
        width     = src.width

    # Reproject Sentinel-1 bands onto S2 grid
    def reproject_s1(s1_path):
        with rasterio.open(s1_path) as src:
            destination = np.zeros((height, width), dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=crs,
                resampling=Resampling.bilinear,
            )
        return destination

    s1_asc  = reproject_s1(s1_asc_path)
    s1_desc = reproject_s1(s1_desc_path)

    # Stack: 7 input bands (no MNDWI to avoid label leakage)
    image_stack = np.vstack([
        s2_bands,                       # indices 0-4
        s1_asc[np.newaxis, :, :],       # index 5
        s1_desc[np.newaxis, :, :],      # index 6
    ])  # shape: (7, H, W)

    # Derive water mask from MNDWI
    green      = s2_bands[1].astype(np.float32)   # B3
    swir       = s2_bands[4].astype(np.float32)   # B11
    mndwi      = calculate_mndwi(green, swir)
    water_mask = (mndwi > MNDWI_THRESHOLD).astype(np.uint8)  # (H, W)

    return image_stack, water_mask, transform, crs


# ─── PATCH GENERATION ─────────────────────────────────────────────────────────

def compute_patch_transform(parent_transform, row_start, col_start):
    """Compute the affine transform for a patch extracted at (row_start, col_start)."""
    x_origin = parent_transform.c + col_start * parent_transform.a
    y_origin = parent_transform.f + row_start * parent_transform.e
    return Affine(
        parent_transform.a, parent_transform.b, x_origin,
        parent_transform.d, parent_transform.e, y_origin,
    )


def save_patch_as_tif(array, patch_transform, crs, output_path):
    """
    Save a numpy array as a GeoTIFF.

    Parameters
    ----------
    array        : np.ndarray  shape (C, H, W) or (H, W)
    patch_transform : rasterio.Affine
    crs          : rasterio CRS
    output_path  : str
    """
    if array.ndim == 2:
        array = array[np.newaxis, :, :]   # (1, H, W)

    num_bands, height, width = array.shape
    dtype = 'float32' if array.dtype == np.float32 else 'uint8'

    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=num_bands,
        dtype=dtype,
        crs=crs,
        transform=patch_transform,
        compress='lzw',
    ) as dst:
        dst.write(array)


def generate_patches(years, split_name):
    """
    Extract image and mask patches for the given years and save as TIF files.

    Saved under:
        PATCHES_DIR / split_name / images / patch_{year}_r{row}_c{col}.tif
        PATCHES_DIR / split_name / masks  / mask_{year}_r{row}_c{col}.tif

    Parameters
    ----------
    years      : list of int
    split_name : str  ('train' or 'test')
    """
    images_dir = os.path.join(PATCHES_DIR, split_name, 'images')
    masks_dir  = os.path.join(PATCHES_DIR, split_name, 'masks')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir,  exist_ok=True)

    total_patches = 0

    for year in years:
        print(f'\n[{split_name}] Loading year {year}...')
        image_stack, water_mask, transform, crs = load_year(year)
        _, height, width = image_stack.shape

        year_patches = 0

        for row_start in range(0, height - PATCH_SIZE + 1, STRIDE):
            for col_start in range(0, width - PATCH_SIZE + 1, STRIDE):
                row_end = row_start + PATCH_SIZE
                col_end = col_start + PATCH_SIZE

                image_patch = image_stack[:, row_start:row_end, col_start:col_end]  # (7, 256, 256)
                mask_patch  = water_mask[row_start:row_end, col_start:col_end]      # (256, 256)

                patch_transform = compute_patch_transform(transform, row_start, col_start)

                patch_name = f'patch_{year}_r{row_start}_c{col_start}.tif'
                mask_name  = f'mask_{year}_r{row_start}_c{col_start}.tif'

                save_patch_as_tif(image_patch, patch_transform, crs,
                                  os.path.join(images_dir, patch_name))
                save_patch_as_tif(mask_patch,  patch_transform, crs,
                                  os.path.join(masks_dir,  mask_name))

                year_patches += 1

        print(f'  → {year}: saved {year_patches} patches')
        total_patches += year_patches

    print(f'\n[{split_name}] Total patches saved: {total_patches}')
    return total_patches


# ─── DATASET ──────────────────────────────────────────────────────────────────

class WaterPatchDataset(Dataset):
    """
    Loads paired (image, mask) TIF patches from disk.

    Each image patch has shape (7, 256, 256) float32.
    Each mask patch has shape (1, 256, 256) float32 with values 0 or 1.
    """

    def __init__(self, split_name):
        self.images_dir = os.path.join(PATCHES_DIR, split_name, 'images')
        self.masks_dir  = os.path.join(PATCHES_DIR, split_name, 'masks')

        # Collect all image patch filenames
        self.image_files = sorted([
            f for f in os.listdir(self.images_dir) if f.endswith('.tif')
        ])

        if len(self.image_files) == 0:
            raise FileNotFoundError(
                f'No patches found in {self.images_dir}. Run generate_patches() first.'
            )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_filename = self.image_files[idx]
        # Derive corresponding mask filename by replacing 'patch_' with 'mask_'
        mask_filename  = image_filename.replace('patch_', 'mask_')

        with rasterio.open(os.path.join(self.images_dir, image_filename)) as src:
            image = src.read().astype(np.float32)   # (7, H, W)

        with rasterio.open(os.path.join(self.masks_dir, mask_filename)) as src:
            mask = src.read().astype(np.float32)    # (1, H, W)

        # Replace NaN / Inf from NoData pixels (reprojection borders, cloud masks)
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        mask  = np.nan_to_num(mask,  nan=0.0, posinf=0.0, neginf=0.0)
        mask  = np.clip(mask, 0.0, 1.0)   # ensure binary

        image_tensor = torch.from_numpy(image)      # (7, 256, 256)
        mask_tensor  = torch.from_numpy(mask)       # (1, 256, 256)

        return image_tensor, mask_tensor


# ─── U-NET ARCHITECTURE ───────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    """Two consecutive Conv → BatchNorm → ReLU blocks."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    """DoubleConv followed by MaxPool2d. Returns both the pre-pool feature map (skip) and pooled output."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip   = self.conv(x)
        pooled = self.pool(skip)
        return skip, pooled


class DecoderBlock(nn.Module):
    """Upsample → concatenate skip connection → DoubleConv."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv     = DoubleConv(in_channels, out_channels)   # in_channels because of concat

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net for binary water segmentation.

    Input  : (B, 7, 256, 256)  — 7-band satellite patch
    Output : (B, 1, 256, 256)  — water probability map (sigmoid applied)
    """

    def __init__(self, in_channels=NUM_INPUT_BANDS, features=[64, 128, 256, 512]):
        super().__init__()

        # Encoder
        self.encoder1 = EncoderBlock(in_channels, features[0])
        self.encoder2 = EncoderBlock(features[0], features[1])
        self.encoder3 = EncoderBlock(features[1], features[2])
        self.encoder4 = EncoderBlock(features[2], features[3])

        # Bottleneck
        self.bottleneck = DoubleConv(features[3], features[3] * 2)

        # Decoder
        self.decoder4 = DecoderBlock(features[3] * 2, features[3])
        self.decoder3 = DecoderBlock(features[3],     features[2])
        self.decoder2 = DecoderBlock(features[2],     features[1])
        self.decoder1 = DecoderBlock(features[1],     features[0])

        # Final classification head
        self.final_conv = nn.Conv2d(features[0], 1, kernel_size=1)

    def forward(self, x):
        skip1, x = self.encoder1(x)
        skip2, x = self.encoder2(x)
        skip3, x = self.encoder3(x)
        skip4, x = self.encoder4(x)

        x = self.bottleneck(x)

        x = self.decoder4(x, skip4)
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)

        return self.final_conv(x)   # raw logits — sigmoid applied in loss and inference


# ─── TRAINING ─────────────────────────────────────────────────────────────────

class DiceBCELoss(nn.Module):
    """
    Combined Dice + Binary Cross-Entropy loss.
    Works well for imbalanced binary segmentation (water vs non-water).
    """

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
        self.bce    = nn.BCEWithLogitsLoss()   # numerically stable: sigmoid + BCE fused

    def forward(self, logits, targets):
        # BCE component (operates on raw logits)
        bce_loss = self.bce(logits, targets)

        # Dice component (needs probabilities)
        probabilities    = torch.sigmoid(logits)
        predictions_flat = probabilities.view(-1)
        targets_flat     = targets.view(-1)
        intersection     = (predictions_flat * targets_flat).sum()
        dice_loss        = 1 - (2 * intersection + self.smooth) / (
            predictions_flat.sum() + targets_flat.sum() + self.smooth
        )

        return bce_loss + dice_loss


def train_epoch(model, loader, optimizer, criterion, scaler, epoch):
    model.train()
    total_loss  = 0.0
    limit       = BATCHES_PER_EPOCH if BATCHES_PER_EPOCH is not None else len(loader)
    print_every = max(1, limit // 10)

    for batch_idx, (image_batch, mask_batch) in enumerate(loader, 1):
        if batch_idx > limit:
            break
        image_batch = image_batch.to(DEVICE, non_blocking=True)
        mask_batch  = mask_batch.to(DEVICE,  non_blocking=True)

        optimizer.zero_grad()

        amp_context = torch.amp.autocast('cuda') if DEVICE.type == 'cuda' else contextlib.nullcontext()
        with amp_context:
            predictions = model(image_batch)
            loss        = criterion(predictions, mask_batch)

        if DEVICE.type == 'cuda':
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        if batch_idx % print_every == 0 or batch_idx == limit:
            print(f'  Epoch {epoch} | batch {batch_idx}/{limit} | loss {total_loss / batch_idx:.4f}',
                  end='\r', flush=True)

    print()   # newline after \r progress
    return total_loss / min(batch_idx, limit)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_iou  = 0.0

    limit = BATCHES_PER_EVAL if BATCHES_PER_EVAL is not None else len(loader)

    with torch.no_grad():
        for batch_idx, (image_batch, mask_batch) in enumerate(loader, 1):
            if batch_idx > limit:
                break

            image_batch = image_batch.to(DEVICE, non_blocking=True)
            mask_batch  = mask_batch.to(DEVICE,  non_blocking=True)

            amp_context = torch.amp.autocast('cuda') if DEVICE.type == 'cuda' else contextlib.nullcontext()
            with amp_context:
                predictions = model(image_batch)
                loss        = criterion(predictions, mask_batch)

            total_loss += loss.item()

            # IoU (Intersection over Union) for water class
            binary_predictions = (torch.sigmoid(predictions) > 0.5).float()
            intersection       = (binary_predictions * mask_batch).sum()
            union              = (binary_predictions + mask_batch).clamp(0, 1).sum()
            iou                = (intersection + 1e-6) / (union + 1e-6)
            total_iou         += iou.item()

    return total_loss / limit, total_iou / limit


def train(model, train_loader, test_loader):
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = DiceBCELoss()
    scaler    = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None

    os.makedirs(RESULTS_DIR, exist_ok=True)

    train_losses, test_losses, test_ious = [], [], []
    best_iou = 0.0

    print(f'\nTraining on {DEVICE}')
    if DEVICE.type == 'cuda':
        print(f'GPU : {torch.cuda.get_device_name(0)}')
        print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
        print(f'AMP : enabled (mixed precision)')
    print('=' * 60)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, epoch)
        test_loss, test_iou = evaluate(model, test_loader, criterion)

        scheduler.step(test_loss)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_ious.append(test_iou)

        print(f'Epoch {epoch:03d}/{NUM_EPOCHS} | '
              f'Train Loss: {train_loss:.4f} | '
              f'Test Loss: {test_loss:.4f} | '
              f'Test IoU: {test_iou:.4f}')

        # Save best model
        if test_iou > best_iou:
            best_iou = test_iou
            torch.save(model.state_dict(),
                       os.path.join(RESULTS_DIR, 'unet_best.pth'))
            print(f'  → Best model saved (IoU={best_iou:.4f})')

    # Save training curves
    plot_training_curves(train_losses, test_losses, test_ious)

    return model


def plot_training_curves(train_losses, test_losses, test_ious):
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_losses, label='Train Loss')
    ax1.plot(epochs, test_losses,  label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, test_ious, color='green', label='Test IoU')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('IoU')
    ax2.set_title('Test IoU (Water Class)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'training_curves.png'), dpi=150, bbox_inches='tight')
    print('Training curves saved.')


# ─── FULL IMAGE PREDICTION ────────────────────────────────────────────────────

def predict_full_image(model, year):
    """
    Run the trained U-Net on the full image for a given year.
    Uses the same PATCH_SIZE / STRIDE grid used during patch generation.
    Overlapping patches are averaged.

    Saves the predicted water mask as a GeoTIFF.
    """
    print(f'\nPredicting full image for year {year}...')
    model.eval()

    image_stack, _, transform, crs = load_year(year)
    _, height, width = image_stack.shape

    prediction_sum   = np.zeros((height, width), dtype=np.float32)
    prediction_count = np.zeros((height, width), dtype=np.float32)

    with torch.no_grad():
        for row_start in range(0, height - PATCH_SIZE + 1, STRIDE):
            for col_start in range(0, width - PATCH_SIZE + 1, STRIDE):
                row_end = row_start + PATCH_SIZE
                col_end = col_start + PATCH_SIZE

                image_patch = image_stack[:, row_start:row_end, col_start:col_end]
                image_tensor = torch.from_numpy(image_patch).unsqueeze(0).to(DEVICE)  # (1, 7, 256, 256)

                predicted_patch = torch.sigmoid(model(image_tensor)).squeeze().cpu().numpy()  # (256, 256)

                prediction_sum[row_start:row_end, col_start:col_end]   += predicted_patch
                prediction_count[row_start:row_end, col_start:col_end] += 1

    # Average overlapping predictions
    valid_mask        = prediction_count > 0
    probability_map   = np.zeros((height, width), dtype=np.float32)
    probability_map[valid_mask] = (
        prediction_sum[valid_mask] / prediction_count[valid_mask]
    )
    water_mask = (probability_map > 0.5).astype(np.uint8)

    # Save outputs
    os.makedirs(RESULTS_DIR, exist_ok=True)

    prob_output_path = os.path.join(RESULTS_DIR, f'probability_map_unet_{year}.tif')
    mask_output_path = os.path.join(RESULTS_DIR, f'water_mask_unet_{year}.tif')

    for output_path, array in [(prob_output_path, probability_map), (mask_output_path, water_mask)]:
        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=height, width=width,
            count=1,
            dtype=array.dtype,
            crs=crs,
            transform=transform,
            compress='lzw',
        ) as dst:
            dst.write(array[np.newaxis, :, :])

    print(f'  Probability map : {prob_output_path}')
    print(f'  Water mask      : {mask_output_path}')
    print(f'  Water coverage  : {water_mask.mean() * 100:.2f}%')

    return probability_map, water_mask


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def patches_exist():
    """Check if patches have already been generated for both splits."""
    for split in ('train', 'test'):
        images_dir = os.path.join(PATCHES_DIR, split, 'images')
        if not os.path.isdir(images_dir) or len(os.listdir(images_dir)) == 0:
            return False
    return True


def main():
    # ── Step 1: Generate and save patches (only if not already done) ───────────
    print('=' * 60)
    print('STEP 1: PATCH GENERATION')
    print('=' * 60)

    if patches_exist():
        train_count = len(os.listdir(os.path.join(PATCHES_DIR, 'train', 'images')))
        test_count  = len(os.listdir(os.path.join(PATCHES_DIR, 'test',  'images')))
        print(f'Patches already exist — skipping generation.')
        print(f'  Train: {train_count} patches | Test: {test_count} patches')
    else:
        train_count = generate_patches(TRAIN_YEARS, split_name='train')
        test_count  = generate_patches(TEST_YEARS,  split_name='test')
        print(f'\nTotal patches — Train: {train_count} | Test: {test_count}')

    # ── Step 2: Build datasets and loaders ────────────────────────────────────
    print('\n' + '=' * 60)
    print('STEP 2: BUILDING DATASETS')
    print('=' * 60)

    train_dataset = WaterPatchDataset('train')
    test_dataset  = WaterPatchDataset('test')

    pin_memory = DEVICE.type == 'cuda'
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=pin_memory)
    test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin_memory)

    print(f'Train patches: {len(train_dataset)} | Test patches: {len(test_dataset)}')

    # ── Step 3: Build and train U-Net ─────────────────────────────────────────
    print('\n' + '=' * 60)
    print('STEP 3: TRAINING U-NET')
    print('=' * 60)

    if DEVICE.type == 'cuda':
        torch.backends.cudnn.benchmark = True   # auto-tune kernels for fixed input size

    model = UNet(in_channels=NUM_INPUT_BANDS).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model parameters: {total_params:,}')

    model = train(model, train_loader, test_loader)

    # ── Step 4: Predict full images for test years ────────────────────────────
    print('\n' + '=' * 60)
    print('STEP 4: FULL IMAGE PREDICTION')
    print('=' * 60)

    # Load best model weights
    best_model_path = os.path.join(RESULTS_DIR, 'unet_best.pth')
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

    for year in TEST_YEARS:
        predict_full_image(model, year)


if __name__ == '__main__':
    main()
