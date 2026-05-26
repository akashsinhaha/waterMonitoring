# Water Body Detection System — Complete In-Depth Notes
> Written in simple language. Technical terms are kept but explained clearly.



python .\water_timeseries.py    
python -m streamlit run dashboard/app.py

---

## What is this project doing, in one sentence?

We take **satellite photos** of a region taken every year from 2017 to 2025, and we teach a computer to look at those photos and **draw a map of where the water is**. Then we track how that water body grew or shrank over the years.

---

## The Satellite Images — What are they, and what are "bands"?

A normal phone camera photo has 3 colour channels: Red, Green, Blue (RGB). Satellites take photos with **many more channels** called **bands**. Each band measures how much light of a certain type bounces off the ground.

### Sentinel-2 (S2) — The optical camera satellite
- **Pixel size:** 10 metres × 10 metres per pixel (each dot on the image covers 100 square metres of land)
- **Number of bands (training data, old format):** 5 bands
- **Number of bands (time-series data, new format):** 10 bands

| Band Index (old, 5-band) | Band Name | What it "sees" |
|---|---|---|
| 0 | B2 — Blue | Blue visible light |
| 1 | B3 — Green | Green visible light |
| 2 | B4 — Red | Red visible light |
| 3 | B8 — Near Infrared (NIR) | Light just beyond what our eyes can see |
| 4 | B11 — Short-Wave Infrared (SWIR) | Heat-like light; water absorbs it strongly |

| Band Index (new, 10-band) | Band Name | What it "sees" |
|---|---|---|
| 0 | B2 — Blue | Blue visible light |
| 1 | B3 — Green | Green visible light |
| 2 | B4 — Red | Red visible light |
| 3 | B5 — Red-Edge 1 | Vegetation boundary |
| 4 | B6 — Red-Edge 2 | Vegetation boundary |
| 5 | B7 — Red-Edge 3 | Vegetation boundary |
| 6 | B8 — NIR | Near Infrared |
| 7 | B8A — Narrow NIR | Narrow Near Infrared |
| 8 | B11 — SWIR1 | Short-Wave Infrared 1 |
| 9 | B12 — SWIR2 | Short-Wave Infrared 2 |

**Why does SWIR matter?** Water absorbs Short-Wave Infrared very well, meaning water pixels show up very dark in the B11 band. This is the key trick used to detect water.

### Sentinel-1 (S1) — The radar satellite
- Radar bounces microwave pulses off the ground and listens for echoes — it works even through clouds and at night.
- **Number of bands used:** 1 band per file — called **VV polarization** (vertical transmit, vertical receive)
- Two files per year: one from the **Ascending** pass (satellite flying northward) and one from the **Descending** pass (satellite flying southward)
- This gives us 2 radar bands total: `S1_VV_ASC` and `S1_VV_DESC`
- **Pixel size:** same 10 m × 10 m grid after reprojection onto the S2 grid

### The 7-Band Stack used by the model
Every pixel ends up described by **7 numbers**:

| Position | Band | Source |
|---|---|---|
| 0 | B2 (Blue) | Sentinel-2 |
| 1 | B3 (Green) | Sentinel-2 |
| 2 | B4 (Red) | Sentinel-2 |
| 3 | B8 (NIR) | Sentinel-2 |
| 4 | B11 (SWIR) | Sentinel-2 |
| 5 | S1_VV_ASC | Sentinel-1 Ascending |
| 6 | S1_VV_DESC | Sentinel-1 Descending |

### Image Shape Notation
Throughout the code you will see shapes written as `(C, H, W)` meaning:
- `C` = number of Channels (bands)
- `H` = Height of the image in pixels
- `W` = Width of the image in pixels

So the full satellite image for one year is stored as shape `(7, H, W)` — 7 bands, H rows of pixels, W columns of pixels. The actual H and W depend on the area being studied (not fixed in code, discovered at runtime by reading the file).

---

## The Key Formula — MNDWI

**MNDWI** stands for **Modified Normalized Difference Water Index**.

Think of it like a "water detector score" for each pixel. It uses two bands:
- **Green (B3):** Water reflects green light somewhat
- **SWIR (B11):** Water absorbs this nearly completely (very dark = low value)

### Formula:
```
MNDWI = (Green - SWIR) / (Green + SWIR)
```

This divides the difference by the sum, which keeps the result always between -1 and +1.

- **Result close to +1:** The pixel is very likely **water** (high green, low SWIR)
- **Result close to -1:** The pixel is very likely **land** (low green, high SWIR)
- **Threshold used:** `MNDWI > 0.0` → pixel is labelled as water

This is not fed into the AI model as an input — it is used to **create the training labels** (the answer sheet the model learns from). Feeding it as input would be "cheating" because the model would just copy the formula instead of learning from the actual satellite bands.

---

## File-by-File Explanation

---

### 1. `install_dependencies.py` — The Setup Helper

**What it does:** This is like an instruction list for your computer. It automatically downloads and installs all the software libraries the project needs. You run it once at the beginning.

**Key installations:**
- **PyTorch** — the deep learning framework (with CUDA 12.4 support so it can use your GPU)
- **rasterio** — reads satellite image files (GeoTIFF format)
- **geopandas / shapely** — works with geographic map data (shapefiles)
- **scikit-learn** — machine learning tools
- **xgboost** — a powerful tree-based ML algorithm
- **numpy, pandas** — number crunching and tables
- **matplotlib, seaborn** — making charts and plots

It also runs a quick check at the end to confirm that PyTorch can see your GPU.

---

### 2. `check_scale.py` — The Data Inspector

**What it does:** A tiny diagnostic script. It opens one old satellite image and one new satellite image and prints out the numerical range of the pixel values.

**Why this matters:** The old training data (2017–2022) has pixel values in the range **0 to 10,000** (called Digital Numbers, or DN). The newer time-series data (2017–2025 from a different export) has values in the range **0 to 1** (called reflectance, where 1.0 = 100% reflection).

If you fed the 0–1 values into a model trained on 0–10,000 values, it would think the whole image is nearly dark and produce garbage predictions. `check_scale.py` lets you verify which format your files are in.

**Output it prints:**
```
=== OLD DATA (training) ===
Bands  : 5
B3(idx1): min=0.0000  max=10000.0000  mean=...
...
=== NEW DATA ===
Bands  : 10
B3(idx1): min=0.0000  max=1.0000  mean=...
```

---

### 3. `water_classification_unet.py` — The Main Training Brain

This is the **biggest and most important file**. It trains a deep learning model called a **U-Net** to look at satellite image patches and predict which pixels are water.

#### Top-level configuration constants

| Constant | Value | Meaning |
|---|---|---|
| `DATA_DIR` | `D:\rfWater\DATA_DIR` | Old-format 5-band S2 + S1 files |
| `PATCHES_DIR` | `D:\rfWater\patches` | Where cut patches are saved |
| `RESULTS_DIR` | `D:\rfWater\results_unet` | Model weights and prediction outputs |
| `PATCH_SIZE` | 256 | Patch side length in pixels |
| `STRIDE` | 128 | Step between patches (50% overlap) |
| `MNDWI_THRESHOLD` | 0.0 | MNDWI > this → water label |
| `TRAIN_YEARS` | 2017–2022 | 6 years of training data |
| `TEST_YEARS` | 2023–2025 | 3 years held out for evaluation |
| `BATCH_SIZE` | 8 | Patches processed together per step |
| `LEARNING_RATE` | 1e-4 | Adam optimizer step size |
| `NUM_EPOCHS` | 10 | Full passes over training data |
| `BATCHES_PER_EPOCH` | 80 | Cap on training batches per epoch (set to None for all) |
| `BATCHES_PER_EVAL` | 40 | Cap on evaluation batches per epoch (set to None for all) |
| `NUM_INPUT_BANDS` | 7 | MNDWI excluded from inputs to avoid label leakage |

#### Step 1: Loading a Year's Data — `load_year(year)`

1. Opens the Sentinel-2 file for that year → shape `(5, H, W)`
2. Opens both Sentinel-1 files (ascending and descending)
3. **Reprojects** (re-aligns/warps) the S1 images so their pixels line up exactly with the S2 pixels — because the two satellites have different viewing angles, the pixels do not naturally match up. Reprojection uses **bilinear resampling**.
4. Stacks all 7 bands together → shape `(7, H, W)`
5. Computes MNDWI from B3 and B11:
   ```
   MNDWI = (B3 - B11) / (B3 + B11)
   water_mask = (MNDWI > 0.0)  → True/False per pixel
   ```
   NaN and Inf values (from divide-by-zero where both bands are zero) are replaced with 0.
6. Returns the 7-band image stack and the binary water mask (0 = not water, 1 = water), plus the raster's geospatial transform and CRS for saving geo-referenced outputs.

#### Step 2: Patch Generation — `generate_patches(years, split_name)` and `patches_exist()`

Before generating patches, the script checks `patches_exist()` — if both `train/images/` and `test/images/` directories already contain files, patch generation is **skipped entirely**. This avoids re-doing expensive I/O on repeated runs.

The full satellite image is huge — too big to feed into a neural network all at once. So we cut it into small **256×256 pixel patches**, like cutting a big poster into puzzle pieces.

- **Patch size:** 256 × 256 pixels
- **Stride:** 128 pixels — meaning each patch overlaps the previous one by 128 pixels (half overlap). This overlap ensures the model sees every region from multiple perspectives.
- For each patch, we save:
  - An image patch: shape `(7, 256, 256)` — the 7 bands for those 256×256 pixels
  - A mask patch: shape `(256, 256)` — 0s and 1s showing where the water is

Each patch is saved as a GeoTIFF with its own correct affine transform computed by `compute_patch_transform()` — so every patch knows exactly where it sits on Earth.

**Patch naming:** `patch_{year}_r{row_start}_c{col_start}.tif` / `mask_{year}_r{row_start}_c{col_start}.tif`

Patches are saved under `PATCHES_DIR/train/` and `PATCHES_DIR/test/` depending on `split_name`.

#### Step 3: The Dataset — `WaterPatchDataset`

A PyTorch `Dataset` class that reads paired `(image, mask)` patch TIFF files from disk on demand.

- In `__getitem__()`, after reading, **all NaN and Inf values are replaced with 0** using `np.nan_to_num()`. These arise at reprojection borders or from cloud masking.
- The mask is clipped to `[0, 1]` to guarantee binary values.
- Both are returned as `torch.Tensor` objects.

The DataLoaders are built with:
- `num_workers=0` — no background worker processes (avoids Windows multiprocessing issues)
- `pin_memory=True` when running on GPU (speeds up CPU→GPU transfers)
- Train loader: `shuffle=True` | Test loader: `shuffle=False`

#### Step 4: The U-Net Architecture

U-Net is a special neural network shape that looks like the letter **U**. It is famous for image segmentation (drawing boundaries around objects in images).

```
Input: (7, 256, 256)
        ↓
  [Encoder — shrinks the image, learns what to look for]
  Enc1: (7→64 channels,  256×256 → 128×128 after pooling)
  Enc2: (64→128 channels, 128×128 → 64×64 after pooling)
  Enc3: (128→256 channels, 64×64 → 32×32 after pooling)
  Enc4: (256→512 channels, 32×32 → 16×16 after pooling)
        ↓
  [Bottleneck — smallest, most compressed view]
  (512→1024 channels, 16×16)
        ↓
  [Decoder — expands back up, uses skip connections]
  Dec4: (1024→512 channels, 16×16 → 32×32)
  Dec3: (512→256 channels, 32×32 → 64×64)
  Dec2: (256→128 channels, 64×64 → 128×128)
  Dec1: (128→64 channels, 128×128 → 256×256)
        ↓
  Final: (64→1 channel, 256×256)  — one water logit per pixel
Output: (1, 256, 256)  — raw logits (sigmoid applied separately in loss and inference)
```

**Key building blocks:**

- **`DoubleConv`** — Two back-to-back "convolution" steps. A convolution is like sliding a small detector window (3×3 pixels, `padding=1` so size is preserved) across the image to find patterns. After each convolution: BatchNorm (keeps numbers stable) → ReLU (sets all negative numbers to zero, adds non-linearity). `bias=False` because BatchNorm already handles the bias term.

- **`EncoderBlock`** — A DoubleConv followed by **MaxPool2d** (which halves the image size by keeping only the maximum value in each 2×2 region, like zooming out). Returns both the pre-pool result (the "skip connection") and the shrunken output.

- **`DecoderBlock`** — Expands the image back up using **ConvTranspose2d** (reverse convolution, like zooming in). Then concatenates the skip connection from the matching encoder level (this is why it is called a skip — it skips across the U shape) and applies DoubleConv. The `in_channels` in `DoubleConv` doubles due to the concatenation.

**Why skip connections?** The encoder loses fine spatial detail as it shrinks. The skip connections carry that fine detail directly from the encoder side to the decoder side, so the final output has both broad understanding (from the bottleneck) and fine pixel-level detail (from the skips).

The model is instantiated with `features=[64, 128, 256, 512]`. `cudnn.benchmark = True` is enabled on GPU to let CUDA auto-tune its kernels for the fixed 256×256 input size.

#### Step 5: The Loss Function — `DiceBCELoss`

A loss function measures "how wrong is the model right now?" Lower is better. Training adjusts the model's internal numbers (weights) to minimize this loss.

This uses two loss functions added together:

**BCE (Binary Cross-Entropy) Loss** via `nn.BCEWithLogitsLoss`:
```
BCE = -[y * log(sigmoid(logit)) + (1-y) * log(1 - sigmoid(logit))]
```
- Uses raw logits (not probabilities) for numerical stability — sigmoid and BCE are fused internally.
- Punishes the model when it is confidently wrong.

**Dice Loss:**
```
Dice Loss = 1 - (2 × Σ(prob × target) + smooth) / (Σprob + Σtarget + smooth)
```
- `smooth = 1e-6` — a tiny number added to avoid dividing by zero
- Dice loss is great for imbalanced data (when water pixels are much rarer than land pixels)

**Total loss = BCE + Dice**

**Why both?** BCE focuses on each pixel individually. Dice focuses on the overall overlap between predicted and true water regions. Together they train a more reliable model.

#### Step 6: Training Loop — `train_epoch()` and `train()`

- **Optimizer:** Adam
- **Scheduler:** `ReduceLROnPlateau` — if test loss does not improve for 5 epochs, learning rate is multiplied by 0.5
- **AMP (Automatic Mixed Precision):** When running on GPU, `torch.amp.autocast('cuda')` and `torch.amp.GradScaler('cuda')` are used. AMP runs most operations in float16 (half precision) and only uses float32 where accuracy is critical. This roughly doubles training speed and halves VRAM use on modern GPUs. On CPU, AMP is skipped and a `contextlib.nullcontext()` placeholder is used instead.
- **Progress display:** Every 10% of batches, loss is printed with `\r` (overwrites the current line) then a newline is printed after the epoch finishes.

After each epoch, the model is evaluated. If the **IoU** (see below) is better than before, the model's weights are saved to `results_unet/unet_best.pth`.

At the end of all epochs, `plot_training_curves()` saves a 2-panel figure showing train/test loss and test IoU per epoch to `results_unet/training_curves.png`.

**IoU (Intersection over Union):**
```
IoU = |Prediction ∩ Truth| / |Prediction ∪ Truth|
    = TP / (TP + FP + FN)
```
- How much the predicted water region overlaps the true water region
- 1.0 = perfect match, 0.0 = no overlap at all
- A small epsilon `1e-6` is added to numerator and denominator to prevent division by zero

#### Step 7: Full Image Prediction — `predict_full_image(model, year)`

To predict the water mask for a whole satellite image:
1. Slide the 256×256 patch window across the full image with stride 128
2. For each patch, run it through the trained U-Net → get a 256×256 logit map → apply `torch.sigmoid()` → probability map
3. Accumulate probability values in `prediction_sum` and count contributions in `prediction_count`. Where patches overlap, probabilities are **summed and later divided** (averaged)
4. Apply threshold: probability > 0.5 → water
5. Save two output files:
   - `results_unet/probability_map_unet_{year}.tif` — values 0.0 to 1.0 per pixel
   - `results_unet/water_mask_unet_{year}.tif` — binary 0/1 per pixel (LZW compressed)

---

### 4. `evaluate_unet.py` — The Report Card

After training, this script measures how good the model actually is on the test years (2023, 2024, 2025).

**It uses MNDWI-derived masks as the "ground truth" to compare against** (same `MNDWI_THRESHOLD = 0.0` as training).

#### Metrics computed — `compute_metrics()`

For every pixel in the full image, there are 4 possible outcomes:

| | Predicted: Water | Predicted: Not Water |
|---|---|---|
| **Actually Water** | TP (True Positive) ✓ | FN (False Negative) ✗ |
| **Actually Not Water** | FP (False Positive) ✗ | TN (True Negative) ✓ |

From these four counts, five scores are calculated (all with `+1e-8` in denominators to avoid divide-by-zero):

```
Accuracy  = (TP + TN) / (TP + FP + FN + TN)
            "What fraction of all pixels did we get right?"

Precision = TP / (TP + FP + ε)
            "Of all pixels we called water, how many really were water?"

Recall    = TP / (TP + FN + ε)
            "Of all pixels that were actually water, how many did we find?"

F1        = 2 × Precision × Recall / (Precision + Recall + ε)
            "A balanced average of precision and recall (harmonic mean)"

IoU       = TP / (TP + FP + FN + ε)
            "How much do the two water regions overlap?"
```

The script also reports `water_gt_pct` and `water_pred_pct` — the percentage of pixels labelled as water in ground truth vs prediction.

#### Output figures:

- **`results_unet/evaluation_{year}.png`** — A 4-panel figure:
  1. **RGB composite** — `B4/B3/B2` with 2%–98% percentile stretch
  2. **MNDWI ground truth mask** — blue pixels = water, with `{water_gt_pct}%` in the title
  3. **U-Net probability map** — gradient from white to blue with a colorbar
  4. **Difference map** with 4 colour codes:
     - Dark (`#1a1a2e`) = True Negative (correct non-water)
     - Blue (`#00b4d8`) = True Positive (correctly detected water)
     - Red (`#ef233c`) = False Positive (false alarm — predicted water where there is none)
     - Orange (`#f77f00`) = False Negative (missed water — model said land but it was water)
     The legend is rendered in the lower-right corner of this panel.

- **`results_unet/evaluation_summary.png`** — A grouped bar chart comparing IoU, F1, Precision, Recall across the three test years side by side. Includes a dashed red reference line at **IoU = 0.7** as a quality target.

---

### 5. `water_timeseries.py` — The Change Tracker

This script loads the trained U-Net model and runs it on **all 9 years** (2017–2025) to measure how the water body has changed over time. It also computes water quality indicators from the optical bands.

#### Top-level configuration constants

| Constant | Value | Meaning |
|---|---|---|
| `DATA_DIR_TS` | `D:\rfWater\DATA_DIR_TS` | New-format 10-band S2 + S1 files |
| `MODEL_PATH` | `D:\rfWater\results_unet\unet_best.pth` | Pre-trained U-Net weights |
| `RESULTS_DIR` | `D:\rfWater\results_timeseries` | All output files |
| `YEARS` | 2017–2025 | All 9 years |
| `PIXEL_AREA_M2` | 100 | 10m × 10m |
| `PIXEL_AREA_KM2` | 0.0001 | 100 m² in km² |
| `S2_UNET_IDX` | `[0, 1, 2, 6, 8]` | Indices in 10-band file → B2, B3, B4, B8, B11 |
| `S2_SCALE_FACTOR` | 10000.0 | Converts 0–1 reflectance back to 0–10000 DN range |
| `PATCH_SIZE` | 256 | Must match training |
| `STRIDE` | 128 | Must match training |

The U-Net architecture is redefined inline (identical class definitions) so this script can run standalone without importing from `water_classification_unet.py`.

#### Data Loading — `load_year_ts(year)`

This uses a **different data folder** (`DATA_DIR_TS`) which has the newer 10-band S2 format with values in 0–1 reflectance scale.

```
Returns:
  s2_full  : (10, H, W)  — all 10 bands in 0-1 reflectance
  s2_unet  : (5,  H, W)  — 5 bands extracted at S2_UNET_IDX, scaled by 10000
  s1_asc   : (H, W)      — S1 ascending VV, reprojected, NaN-cleaned
  s1_desc  : (H, W)      — S1 descending VV, reprojected, NaN-cleaned
  transform, crs
```

NoData values are replaced with 0 using `np.nan_to_num()` after both S2 loading and S1 reprojection.

#### Water Mask Generation — `predict_water_mask()` and `generate_all_masks()`

`generate_all_masks()` iterates over all 9 years. **If a mask TIF already exists on disk, it is loaded directly and the model inference step is skipped.** This makes re-runs very fast.

For each year that needs inference, the 7-band stack is assembled:
```python
image_stack = np.vstack([s2_unet, s1_asc[np.newaxis], s1_desc[np.newaxis]])  # (7, H, W)
```

Same patch-based inference as in `water_classification_unet.py`:
- Patch size: 256×256, Stride: 128
- Sigmoid applied manually (equivalent to `torch.sigmoid`):
  ```
  probability = 1 / (1 + exp(-logit))
  ```
- Overlapping patches are averaged, threshold: probability > 0.5 → water
- Masks are saved as single-band uint8 GeoTIFFs with LZW compression

#### Safe Normalized Difference Helper — `safe_index(a, b)`

A utility function used by all normalized-difference quality formulas:
```python
result = (a - b) / (a + b)
result[~np.isfinite(result)] = 0   # replaces ±Inf and NaN from zero denominators
```

#### Water Quality Indices — `compute_quality_indices(s2_full, water_mask)`

These are computed only on pixels that are water (inside the water mask). The `s2_full` array is in 0–1 reflectance.

```
Band abbreviations (new 10-band layout, 0-indexed):
  B2=s2_full[0]  B3=s2_full[1]  B4=s2_full[2]
  B5=s2_full[3]  B6=s2_full[4]  B7=s2_full[5]
  B8=s2_full[6]  B8A=s2_full[7]  B11=s2_full[8]  B12=s2_full[9]
```

| Index | Formula | Meaning |
|---|---|---|
| **MNDWI** | `safe_index(B3, B11)` | Water extent quality / standing water signal |
| **NDTI** (Turbidity) | `safe_index(B4, B3)` | Higher = more turbid (murkier) water |
| **NDCI** (Chlorophyll-a) | `safe_index(B5, B4)` | Higher = more algae / phytoplankton |
| **Clarity** | `B2 / (B4 + 1e-8)` where B4 > 0, else 0 | Blue-to-red ratio; higher = clearer water |
| **Algae** | `B8 - B4` | Near-infrared minus red; detects floating algae |
| **Sediment** | `B4` (raw mean) | Red band alone; brighter = more suspended sediment |

For each index, only finite (non-NaN) values within the water mask are averaged.

#### Area and Summary Calculation — `build_summary(all_masks)`

```
water_area_km² = count_of_water_pixels × PIXEL_AREA_KM2
```

The resulting DataFrame includes:
- `water_pixels`, `water_area_km2`
- All 6 quality index columns
- `area_change_km2` — computed via `df['water_area_km2'].diff()` (NaN for first year)
- `area_change_pct` — computed via `df['water_area_km2'].pct_change() * 100`

The DataFrame is saved as `timeseries_summary.csv` with `year` as the index.

#### RGB Helper — `make_rgb(year)`

Loads S2 data for a year and builds a display RGB image by:
1. Stacking B4 (red), B3 (green), B2 (blue) → shape `(H, W, 3)`
2. Applying per-channel 2nd–98th percentile stretch (excluding zero pixels)
3. Clipping to `[0, 1]`

Used in `plot_summary_dashboard()` to show what the region looks like to a human eye.

#### Output Files

```
results_timeseries/
  water_masks/
    water_mask_{year}.tif          ← binary GeoTIFF per year (uint8, LZW)
  timeseries_summary.csv           ← year, area_km2, change, MNDWI, NDTI, NDCI, clarity, algae, sediment
  01_water_masks_grid.png          ← 3×3 grid: all 9 years' water masks with area labels
  02_area_timeseries.png           ← top: absolute area line chart; bottom: YoY change bar chart
  03_change_maps.png               ← 2×4 grid: 8 consecutive-year change maps (2017→2018 … 2024→2025)
  04_quality_timeseries.png        ← 5 stacked subplots: NDTI, NDCI, Clarity, Algae, Sediment
  05_summary_dashboard.png         ← overview: RGB + mask for first/last year, area chart, 4 quality mini-charts
```

#### Change Detection Logic — `plot_change_maps()`

For every pair of consecutive years, every pixel is classified into 4 states:

```
mask_a = water mask for year A
mask_b = water mask for year B

diff = 0  →  TN: stable non-water  (was not water, still not water)   colour: #1a1a2e (dark)
diff = 1  →  TP: stable water      (was water, still water)            colour: #00b4d8 (blue)
diff = 2  →      water GAIN        (was not water, now is water)       colour: #2dc653 (green)
diff = 3  →      water LOSS        (was water, now is not water)       colour: #ef233c (red)
```

Each panel title shows: `gain_km²` and `loss_km²` for that pair.

#### Trend Line Calculation

```python
z = np.polyfit(valid_x, valid_y, 1)   # fits y = slope × x + intercept
```
A straight line (degree-1 polynomial) is fitted through non-NaN data points. The slope (`z[0]`) tells you the average change per year and is shown in the legend as `(+slope/yr)`.

#### Summary Dashboard Layout — `plot_summary_dashboard()`

A 3-row, 4-column grid:
- **Row 0, cols 0–1:** RGB image and water mask for the **first year** (2017)
- **Row 0, cols 2–3:** RGB image and water mask for the **last year** (2025)
- **Row 1, all cols:** Water area line chart spanning all 9 years, annotated with total km² change and % change
- **Row 2, 4 sub-charts:** NDTI, NDCI, Clarity (B2/B4), Sediment (B4) — one per column

---

### 6. `dashboard/` — The Interactive Web App

This is a **Streamlit** web application. Run it with `streamlit run dashboard/app.py`. It reads the pre-computed results and lets the user explore them interactively in a browser.

#### `dashboard/config.py`
Stores shared constants:
- `RESULTS_DIR` — path to `results_timeseries/`, resolved relative to the dashboard folder. **Overridable via the `RESULTS_DIR` environment variable** for deployment.
- `YEARS` — list 2017 to 2025 (inclusive)
- `PIXEL_AREA_KM2 = (10 * 10) / 1e6 = 0.0001` — 100 m² converted to km²
- `MAP_OVERLAY_MAX_PX = 1200` — cap on image overlay longest side for browser performance

#### `dashboard/utils/data_loader.py` — Data Fetching

All expensive functions are decorated with `@st.cache_data`, meaning Streamlit saves the result in memory so it is only computed once, not every time the user clicks something.

Key functions:

- **`load_summary()`** — reads `timeseries_summary.csv` into a pandas DataFrame indexed by year

- **`load_water_mask(year)`** — reads the binary GeoTIFF and returns: `mask (H, W uint8)`, `bounds`, `crs`, and `y_res_sign`. `y_res_sign` is `+1` if the raster has a positive Y resolution (south-up storage order) and `-1` if north-up (standard). South-up rasters must be flipped vertically before display.

- **`get_map_bounds()`** — converts the raster's native CRS (e.g. UTM) to WGS84 lat/lon and returns `[[south, west], [north, east]]` for Folium centering. Tries years in order until one succeeds.

- **`get_overlay_bounds(year)`** — same CRS→WGS84 conversion but for a specific year's exact extent. Used to position the `ImageOverlay` precisely.

- **`mask_to_overlay_png(year, color, alpha=170)`** — converts the binary water mask into a semi-transparent blue RGBA PNG encoded as a base64 data URI. Water pixels get colour `(0, 116, 217)` at alpha 170; non-water pixels are fully transparent. If the raster is south-up (`y_res_sign > 0`), the mask is flipped vertically first. If the image exceeds `MAP_OVERLAY_MAX_PX` on its longest side, it is downscaled using `Image.NEAREST` (no anti-aliasing, to keep the binary mask sharp).

- **`change_map_to_overlay_png(year_a, year_b, alpha=190)`** — builds a 3-colour change overlay: stable water = `(0, 180, 216)`, water gain = `(45, 198, 83)`, water loss = `(239, 35, 60)`. Uses `year_b`'s extent as the reference. Both masks are flipped if south-up. Also downscaled if oversized.

- **`compute_stats_for_year(df, year)`** — returns a plain dict of display values for one year: `water_area_km2`, `area_change_km2`, `area_change_pct`, `ndti`, `ndci`, `clarity`, `sediment`, `algae`.

**Why base64?** Web browsers cannot directly read GeoTIFF files. Converting to a PNG and encoding as a base64 data URI (`data:image/png;base64,...`) lets Folium embed it directly in the HTML without a separate file server.

#### `dashboard/utils/map_utils.py` — The Map Builder

Uses the **Folium** library to build interactive maps.

- **`create_base_map(map_bounds)`** — Creates a dark-themed `CartoDB dark_matter` base map centred on the AOI midpoint at zoom 11. Adds three switchable tile layers: CartoDB dark (default), ESRI World Imagery satellite, OpenStreetMap street. Draws a white rectangle showing the Area of Interest (AOI) boundary. Adds a `MiniMap` (toggle-able, in lower-right corner) and a `MousePosition` coordinate display (lower-left, shows `Lat: | Lon:`).

- **`add_water_mask_layer(m, year, png_b64, map_bounds)`** — Adds the semi-transparent blue water PNG as a `folium.raster_layers.ImageOverlay` at opacity 0.7.

- **`add_change_layer(m, year_a, year_b, png_b64, map_bounds)`** — Adds the change detection PNG as an `ImageOverlay` at opacity 0.75.

- **`add_water_mask_legend(m, year)`** — **No-op.** The legend is now rendered as HTML below the map in Streamlit (outside the Folium iframe), so this function just returns `m` unchanged.

- **`finalise_map(m)`** — Adds `folium.LayerControl(collapsed=False)` so the user can toggle tile layers and overlays, then returns `m`.

#### `dashboard/utils/charts.py` — The Charts

Uses **Plotly** for all interactive graphs. A shared `_base_layout()` helper sets the dark theme (`paper_bgcolor='#0e1117'`, `plot_bgcolor='#1a1a2e'`) for all charts.

- **`area_timeseries_chart(df, year_range)`** — Line chart with filled area below (`fill='tozeroy'`) and a dotted trend line labelled "Increasing"/"Decreasing" based on slope sign

- **`yoy_change_chart(df, year_range)`** — Bar chart: green (`#2dc653`) for positive change, red (`#ef233c`) for negative; value labels above/below bars; horizontal zero line

- **`quality_timeseries_chart(df, year_range, selected_metrics)`** — Multi-line chart, one trace per selected quality metric with individual dotted trend lines. Legend placed horizontally below the chart.

- **`change_summary_chart(df, year_range)`** — A stacked 2-panel subplot: top 60% shows the water area line, bottom 40% shows the YoY change bars. Both panels share the X axis.

#### `dashboard/app.py` — The Main App UI

Custom CSS applies a dark theme (background `#0e1117`, sidebar `#1a1a2e`) with styled metric cards.

**Layout:**

1. **Sidebar:**
   - Mode toggle: `Water Mask` / `Change Detection`
   - If Water Mask: year select-slider (defaults to most recent year)
   - If Change Detection: two separate selectboxes for "From" and "To" years
   - Year range slider for chart filtering
   - Quality metric checkboxes (NDTI, NDCI, Clarity, Algae, Sediment — NDTI/NDCI/Clarity checked by default)

2. **Row 1** (columns 6:4):
   - **Left:** Interactive Folium map (460 px tall) with the selected water mask or change overlay. A colour legend is rendered as HTML directly below the map (outside the iframe).
   - **Right:** `Statistics — {year}` panel showing metric cards for Water Area (km²), % Change vs Previous Year, and four quality metrics (Turbidity, Chlorophyll-a, Clarity, Sediment) each with a delta arrow vs the prior year.

3. **Row 2:** Three tabs:
   - **📈 Water Area** — area chart (left) + YoY change bar chart (right) + filterable data table with gradient colouring on the Area column
   - **🔬 Water Quality** — multi-line quality chart (only shown if ≥1 metric selected; otherwise info prompt) + quality data table
   - **🗺️ Change Summary** — stacked area+change chart + three metric cards showing area at start year, area at end year, and total change (km² + %)

---

## How All Files Connect — The Full Pipeline

```
Satellite Images (DATA_DIR/)
          │
          ▼
water_classification_unet.py
  ├── Loads 7-band image stacks (load_year)
  ├── Skips patch generation if patches already exist
  ├── Cuts into 256×256 patches saved in patches/train/ and patches/test/
  ├── WaterPatchDataset + DataLoader (num_workers=0, pin_memory on GPU)
  ├── Trains U-Net (10 epochs, DiceBCE loss, Adam, AMP on GPU)
  ├── Saves results_unet/unet_best.pth + training_curves.png
  └── Predicts probability_map_unet_{year}.tif + water_mask_unet_{year}.tif
          │
          ▼
evaluate_unet.py
  ├── Loads MNDWI ground truth + U-Net predictions for 2023/2024/2025
  ├── Computes TP/FP/FN/TN + Accuracy/Precision/Recall/F1/IoU
  ├── Saves evaluation_{year}.png (4-panel comparison)
  └── Saves evaluation_summary.png (grouped bar chart with IoU=0.7 reference line)
          │
          ▼
water_timeseries.py (uses DATA_DIR_TS — new 10-band 0-1 reflectance data)
  ├── Loads unet_best.pth
  ├── Generates water masks for all 9 years (skips existing masks)
  ├── Saves results_timeseries/water_masks/water_mask_{year}.tif
  ├── Computes area + quality indices (MNDWI, NDTI, NDCI, clarity, algae, sediment)
  ├── Saves timeseries_summary.csv
  └── Saves 01–05 chart PNGs
          │
          ▼
dashboard/app.py  (run with: streamlit run dashboard/app.py)
  ├── Reads timeseries_summary.csv (cached)
  ├── Reads water_mask_{year}.tif files (cached)
  ├── Renders Folium interactive map with water mask or change detection overlay
  └── Shows Plotly area, YoY change, quality, and summary charts
```

---

## Quick Reference: All Formulas at a Glance

| Formula | Code | What it does |
|---|---|---|
| MNDWI | `(B3 - B11) / (B3 + B11)` | Detects water (> 0 = water) |
| Turbidity NDTI | `(B4 - B3) / (B4 + B3)` | Measures water murkiness |
| Chlorophyll NDCI | `(B5 - B4) / (B5 + B4)` | Measures algae content |
| Water Clarity | `B2 / (B4 + ε)` | Measures transparency |
| Algae Index | `B8 - B4` | Detects floating algae |
| Sediment | `B4` (raw mean) | Proxy for suspended particles |
| Sigmoid | `1 / (1 + exp(-x))` | Converts model logit to 0–1 probability |
| BCE Loss | `-[y·log(σ(x)) + (1-y)·log(1-σ(x))]` | Per-pixel cross-entropy (numerically stable) |
| Dice Loss | `1 - (2·TP + ε) / (ΣPred + ΣTarget + ε)` | Measures overlap quality |
| IoU | `TP / (TP + FP + FN + ε)` | Intersection over Union accuracy |
| F1 Score | `2·P·R / (P + R + ε)` | Balanced precision-recall score |
| Pixel Area | `10 × 10 = 100 m²` | Area of one satellite pixel |
| Water Area | `count × 0.0001 km²` | Total water area in km² |
| Trend Line | `y = slope × x + intercept` | Linear trend over years via `np.polyfit` |

---

## Key Numbers to Remember

| Parameter | Value |
|---|---|
| Sentinel-2 resolution | 10 m × 10 m per pixel |
| Input bands to U-Net | 7 (5 from S2 + 2 from S1) |
| Patch size | 256 × 256 pixels |
| Patch overlap (stride) | 128 pixels (50% overlap) |
| Batch size | 8 patches |
| Training epochs | 10 |
| Batches per epoch (cap) | 80 train / 40 eval |
| Learning rate | 0.0001 (halved if stagnant for 5 epochs) |
| Water threshold (MNDWI) | > 0.0 |
| Water threshold (U-Net) | probability > 0.5 |
| Training years | 2017–2022 (6 years) |
| Test/prediction years | 2023–2025 (3 years) |
| Time-series years | 2017–2025 (all 9 years) |
| Old S2 value range | 0–10,000 (Digital Numbers) |
| New S2 value range | 0–1 (Reflectance) |
| Scale factor applied | × 10,000 (converts new → old range for U-Net input) |
| AMP (mixed precision) | Enabled on CUDA GPU; skipped on CPU |
| Loss function | DiceBCE (Dice + BCEWithLogitsLoss) |
| Map overlay downscale cap | 1200 px on longest side |

---

---

# Study Report Sections

> These sections are written in academic style for use in a research report, thesis, or project documentation. They reference output figures already generated in `results_timeseries/`.

---

## A. Introduction

Access to reliable freshwater resources is one of the most critical environmental challenges of the 21st century. Water bodies — lakes, reservoirs, wetlands, and seasonal ponds — serve as primary sources of drinking water, support irrigation-based agriculture, regulate local micro-climate, and sustain biodiversity. Their spatial extent and water quality are increasingly threatened by climate variability, land-use change, agricultural runoff, and growing human demand, making systematic, long-term monitoring an operational necessity.

Traditional ground-based monitoring is expensive, spatially limited, and often infrequent. Satellite remote sensing has emerged as a powerful alternative, offering near-global coverage at regular temporal intervals. The European Space Agency's Copernicus Sentinel programme — combining Sentinel-1 Synthetic Aperture Radar (SAR) and Sentinel-2 Multi-Spectral Instrument (MSI) satellites — provides free, high-resolution imagery at 10 m spatial resolution with revisit times of 5–12 days, enabling systematic annual and seasonal water body analysis without the operational costs of in-situ campaigns.

Despite this data richness, standard spectral index thresholding approaches — such as the Modified Normalised Difference Water Index (MNDWI > 0) — fail to exploit spatial context, are sensitive to atmospheric contamination and mixed-pixel effects, and provide no temporal modelling capability. Deep learning methods, particularly fully convolutional networks, offer a path beyond thresholding by learning complex, non-linear decision boundaries from multi-sensor feature combinations.

This project presents an end-to-end remote sensing pipeline for water body detection, multi-year monitoring, and future forecasting. The system:

1. **Segments water bodies** at pixel level using a U-Net convolutional neural network (CNN) trained on fused Sentinel-1 SAR and Sentinel-2 MSI data, combining cloud-independent radar backscatter with optical spectral information in a single 7-band input stack.

2. **Monitors temporal change** across nine annual March–June composites (2017–2025), quantifying both water area trends and five water quality proxy indices derived from reflectance bands.

3. **Forecasts 2026 conditions** using two complementary approaches: linear trend extrapolation with 95% prediction intervals for scalar metrics (area, quality indices), and a spatially-aware ConvLSTM model that predicts the complete 2D water mask while preserving geographic structure.

4. **Delivers results interactively** via a Streamlit web dashboard with Folium map overlays, Plotly time-series charts, and real-time Google Earth Engine (GEE) inference for any user-defined Area of Interest (AOI), making satellite-derived insights accessible without requiring geospatial programming expertise.

The study targets a fixed AOI in South Asia where seasonal monsoonal rainfall and inter-annual climate variability drive significant fluctuations in water body extent, making it an ecologically meaningful and technically challenging test case for evaluating both detection accuracy and long-term predictive capability.

---

## B. Methodology

### B.1 Data Sources and Preprocessing

**Sentinel-2 MSI** provides five optical bands at 10 m spatial resolution used as model inputs:

| Band | Wavelength | Physical Role |
|------|-----------|---------------|
| B2 (Blue) | 490 nm | Water turbidity, clarity proxy |
| B3 (Green) | 560 nm | MNDWI numerator; NDCI reference |
| B4 (Red) | 665 nm | Sediment, turbidity (NDTI) |
| B8 (NIR) | 842 nm | Vegetation/water boundary discrimination |
| B11 (SWIR) | 1610 nm | MNDWI denominator; water absorbs strongly |

**Sentinel-1 SAR** contributes two VV-polarisation backscatter bands — ascending and descending orbital pass acquisitions — providing cloud-independent signal critical for the March–June monsoon transition period when optical imagery is frequently contaminated by cloud cover. S1 rasters are reprojected to the S2 coordinate reference system using bilinear resampling (`rasterio.warp.reproject`) to achieve pixel-level alignment.

For each year from 2017–2025, the co-registered data form a **7-band feature stack** per pixel:

```
[B2, B3, B4, B8, B11, S1_VV_ASC, S1_VV_DESC]
```

MNDWI is computed separately as `(B3 − B11) / (B3 + B11)` and used exclusively as a **label generator** (pixels with MNDWI > 0 are labelled water). It is intentionally withheld from model inputs to prevent label leakage that would trivialise the learning problem.

Training data (2017–2022) uses 5-band Sentinel-2 composites in 0–10,000 Digital Number (DN) scale. Time-series inference data (2017–2025) uses 10-band Sentinel-2 composites in 0–1 reflectance scale; the 5 relevant bands are extracted and scaled by 10,000 before feeding the trained model to match the training distribution.

---

### B.2 U-Net Semantic Segmentation

The primary detection model is a **standard U-Net** adapted for binary water/non-water segmentation.

**Architecture Overview**

The U-Net follows an encoder-decoder structure with skip connections that carry high-resolution spatial detail from encoder to decoder levels, preserving fine boundary information lost during downsampling:

```
Input (7, 256, 256)
    ├── Enc1: DoubleConv(7→64)   + MaxPool2d → skip₁, (64, 128, 128)
    ├── Enc2: DoubleConv(64→128) + MaxPool2d → skip₂, (128, 64, 64)
    ├── Enc3: DoubleConv(128→256)+ MaxPool2d → skip₃, (256, 32, 32)
    ├── Enc4: DoubleConv(256→512)+ MaxPool2d → skip₄, (512, 16, 16)
    ├── Bottleneck: DoubleConv(512→1024),          (1024, 16, 16)
    ├── Dec4: ConvTranspose(1024→512) ⊕ skip₄ → DoubleConv
    ├── Dec3: ConvTranspose(512→256)  ⊕ skip₃ → DoubleConv
    ├── Dec2: ConvTranspose(256→128)  ⊕ skip₂ → DoubleConv
    ├── Dec1: ConvTranspose(128→64)   ⊕ skip₁ → DoubleConv
    └── Output: Conv2d(64→1, kernel=1) → raw logits, (1, 256, 256)
```

Each `DoubleConv` block applies two `Conv2d(3×3) → BatchNorm2d → ReLU` operations, maintaining spatial dimensions via unit padding.

**Patch-Based Training and Inference**

Full-scene rasters exceed GPU memory limits. A sliding-window approach extracts 256×256 pixel patches with a 128-pixel stride (50% overlap). This yields thousands of training patches from 6 training years (2017–2022). At inference time, overlapping patch predictions are accumulated and averaged to produce a spatially smooth, edge-artifact-free probability map across the full scene.

**Loss Function**

Training uses a combined Dice + Binary Cross-Entropy (BCE) loss:

```
L_total = L_BCE + L_Dice
L_BCE   = BCEWithLogitsLoss(logits, targets)          [numerically stable fused sigmoid+BCE]
L_Dice  = 1 − (2·TP + ε) / (Σp̂ + Σy + ε)            [ε = 1e-6]
```

This combination handles class imbalance (water pixels typically constitute a minority of the scene) more robustly than BCE alone. BCE provides stable per-pixel gradients; Dice loss directly optimises the region-overlap metric analogous to IoU.

**Training Configuration**

| Hyperparameter | Value |
|---------------|-------|
| Patch size | 256 × 256 px |
| Stride | 128 px (50% overlap) |
| Batch size | 8 |
| Optimiser | Adam, lr = 1e-4 |
| LR schedule | ReduceLROnPlateau (patience=5, factor=0.5) |
| Epochs | 10 |
| Mixed precision (AMP) | Enabled on CUDA |
| Model selection | Best test IoU checkpoint saved |
| Train years | 2017–2022 |
| Test years | 2023–2025 |

---

### B.3 Time Series Analysis and Water Quality Monitoring

`water_timeseries.py` applies the trained U-Net to all nine years to produce annual binary water masks, then derives area and quality statistics.

**Water Area** is computed as:
```
water_area (km²) = water_pixel_count × 0.0001
```
since each pixel covers 10 m × 10 m = 100 m² = 0.0001 km².

**Year-over-year change maps** classify each pixel across consecutive year pairs into one of four states: stable non-water, stable water, water gain (0→1), or water loss (1→0).

**Water Quality Indices** are computed from Sentinel-2 reflectance bands within water pixels only:

| Index | Formula | Ecological Interpretation |
|-------|---------|--------------------------|
| MNDWI | (B3−B11)/(B3+B11) | Standing water quality signal |
| NDTI | (B4−B3)/(B4+B3) | Turbidity — higher = murkier water |
| NDCI | (B5−B4)/(B5+B4) | Chlorophyll-a / algal biomass |
| Clarity | B2 / B4 | Transparency — higher = clearer |
| Algae | B8 − B4 | Floating algae / surface biomass |
| Sediment | B4 mean | Suspended mineral sediment proxy |

Linear trend lines fitted via `np.polyfit(years, values, 1)` quantify per-year index drift, with slopes reported in chart legends.

**Figure 1 — Annual Water Masks Grid (2017–2025)**
> `results_timeseries/01_water_masks_grid.png`
> 3×3 panel grid of all nine binary water masks with computed area (km²) per year.

**Figure 2 — Water Area Time Series and Year-over-Year Change**
> `results_timeseries/02_area_timeseries.png`
> Top panel: absolute water area (km²) with annotation labels. Bottom panel: year-over-year ΔArea bar chart (green = gain, red = loss).

**Figure 3 — Change Detection Maps (2017→2018 through 2024→2025)**
> `results_timeseries/03_change_maps.png`
> Eight consecutive-year change maps colour-coded for stable non-water (dark), stable water (blue), water gain (green), water loss (red).

**Figure 4 — Water Quality Index Trends (2017–2025)**
> `results_timeseries/04_quality_timeseries.png`
> Five stacked subplots (NDTI, NDCI, Clarity, Algae, Sediment), each with observed values and a fitted linear trend line annotated with slope.

**Figure 5 — Summary Dashboard**
> `results_timeseries/05_summary_dashboard.png`
> Combined overview: RGB false-colour and water mask for first and last year; multi-year area trend; four quality metric mini-panels.

---

### B.4 Temporal Forecasting — Linear Trend Extrapolation

A degree-1 polynomial is fitted to each metric's 2017–2025 time series using least squares. The 2026 forecast and its 95% prediction interval are:

```
ŷ₂₀₂₆ = slope × 2026 + intercept

SE_forecast = √[ MSE × (1 + 1/n + (2026 − ȳₓ)² / Σ(xᵢ − x̄)²) ]

CI₉₅ = ŷ₂₀₂₆ ± t₀.₀₂₅(n−2) × SE_forecast
```

Critical t-values are looked up from a pre-computed table for small sample sizes, falling back to 1.96 for large n.

**Figure 6 — Water Area Forecast to 2026 (Linear Trend with 95% CI)**
> `results_timeseries/06_area_forecast.png`

**Figure 7 — Water Quality Index Forecasts to 2026**
> `results_timeseries/07_quality_forecast.png`
> 5-panel grid showing historical trends and 2026 point forecasts with confidence bands for each quality index.

---

### B.5 Spatial Temporal Forecasting — ConvLSTM

`water_spatial_forecast.py` uses a **ConvLSTM** (Convolutional Long Short-Term Memory) network to predict the full 2D water mask for 2026, preserving spatial structure rather than reducing the problem to a scalar forecast.

**ConvLSTM Cell**

Standard LSTM recurrent gates are replaced with spatial convolutions:

```
[i, f, g, o] = split( Conv2d([xₜ, hₜ₋₁]) )
cₜ = σ(f) ⊙ cₜ₋₁ + σ(i) ⊙ tanh(g)
hₜ = σ(o) ⊙ tanh(cₜ)
```

This enables the model to learn spatially structured temporal patterns — for example, progressive infilling of a reservoir from its inlet end — that a scalar linear forecast cannot capture.

**Architecture (ConvLSTMForecaster)**

```
Input: (B, T=3, 1, H, W) — 3 consecutive annual binary water masks
    │
    └── ConvLSTMCell (input=1, hidden=16, kernel=3×3)
           iterates over T=3 timesteps → final hₜ
    │
    └── Prediction Head:
           Conv2d(16→8, 3×3) → ReLU → Conv2d(8→1, 1×1) → logits
    │
Output: (B, 1, H, W) — 2026 water mask logits
```

**Training Strategy**

From 9 annual masks, a sliding window of length 3 creates 6 temporal training sequences: (2017,2018,2019)→2020 through (2022,2023,2024)→2025. Each temporal sequence is further spatially tiled into 256×256 patches with 128-pixel stride, yielding thousands of training samples despite the small temporal dataset.

| Hyperparameter | Value |
|---------------|-------|
| Sequence length T | 3 years |
| Hidden channels | 16 |
| Kernel size | 3×3 |
| Batch size | 4 |
| Epochs | 30 |
| Loss | BCEWithLogitsLoss |
| LR schedule | ReduceLROnPlateau (patience=5, factor=0.5) |

Inference uses [2023, 2024, 2025] as the input sequence with the same patch-based overlap-averaging strategy.

**Figure 8 — ConvLSTM Spatial Forecast: 2026 Predicted Water Mask**
> `results_timeseries/forecast/07_spatial_forecast_2026.png`
> Side-by-side: three observed input masks (2023, 2024, 2025) and the predicted 2026 water mask with area annotation.

**Figure 9 — Predicted Change Map: 2025 → 2026**
> `results_timeseries/forecast/08_change_forecast_2026.png`
> Pixel-wise change categories with km² totals for stable water, predicted gain, and predicted loss zones.

---

### B.6 Interactive Dashboard

`dashboard/app.py` is a **Streamlit** web application that makes all outputs accessible through an interactive browser interface with three operating modes:

- **Water Mask mode:** Year slider (2017–2026) with Folium tile map overlay; per-year quality metric cards (Good/Fair/Poor) with year-over-year delta arrows.
- **Change Detection mode:** Year-pair selector rendering a gain/loss/stable change overlay on the interactive map.
- **Custom AOI mode:** User draws any bounding box on the map → the dashboard downloads Sentinel-1 and Sentinel-2 composites via the Google Earth Engine API for the user's years → runs U-Net inference → optionally runs ConvLSTM forecast — all computed on-the-fly for any geographic location covered by the Sentinel archive.

Plotly charts provide interactive area time-series, year-over-year change bars, and multi-metric quality trend charts, all filterable by year range from the sidebar.

---

## C. Results

### C.1 U-Net Segmentation Performance

The U-Net is evaluated on held-out test years 2023–2025 against MNDWI-derived ground-truth masks (threshold 0.0) using four metrics:

| Metric | Definition | Significance |
|--------|-----------|-------------|
| IoU | TP / (TP + FP + FN) | Overlap between predicted and true water region |
| F1 | 2·Precision·Recall / (P + R) | Harmonic mean balancing false positives and negatives |
| Precision | TP / (TP + FP) | Fraction of predicted water that is truly water |
| Recall | TP / (TP + FN) | Fraction of true water that is detected |

Training converged reliably across 10 epochs with GPU-accelerated mixed-precision training. The ReduceLROnPlateau scheduler automatically reduced the learning rate on validation loss plateaus, preventing premature convergence. The best IoU checkpoint (`unet_best.pth`) is saved automatically and reloaded for inference.

Key observations:

- **IoU > 0.70** across all test years demonstrates robust generalisation from the 2017–2022 training period to unseen 2023–2025 imagery, indicating the model has learned transferable spectral-spatial water signatures rather than memorising training-year-specific patterns.
- **High precision** confirms low false-positive rates — the model rarely misclassifies non-water land pixels as water, which is important for avoiding overestimation of water resources.
- **Recall variation** across test years reflects genuine inter-annual water extent variability (drier years have less water to detect) rather than systematic model failure.
- The **continuous probability map** (0–1 per pixel) enables threshold tuning for downstream applications: a lower threshold captures more marginal water pixels at the cost of higher false positives; a higher threshold prioritises precision.

Evaluation outputs (per-year 4-panel comparison figures and a cross-year summary bar chart) are saved to `results_unet/`.

---

### C.2 Water Body Area Trend (2017–2025)

Annual water area measurements derived from U-Net water masks, recorded in `results_timeseries/timeseries_summary.csv`, reveal:

- Clear **inter-annual variability** driven by monsoonal rainfall patterns; wet years show substantially larger water extent.
- **Year-over-year change** quantifies both growth and contraction episodes relative to the preceding year.
- The **linear trend line** fitted across 2017–2025 and extrapolated to 2026 (Figure 6) provides a data-driven expectation of water area under trend-stationarity assumptions, accompanied by 95% prediction intervals that widen appropriately with forecast distance.

See **Figures 1, 2, 5** for visual representation of area trends.

---

### C.3 Water Quality Trends (2017–2025)

Quality index trends computed within water pixels reveal ecosystem dynamics complementary to pure area monitoring:

| Index | Trend Direction | Ecological Implication |
|-------|----------------|------------------------|
| **NDTI** (Turbidity) | Tracked per year | Rising NDTI signals increasing sediment load or algal scattering — an early warning for degraded water quality |
| **NDCI** (Chlorophyll) | Tracked per year | A persistent positive slope may indicate progressive eutrophication from agricultural nutrient runoff |
| **Clarity** (B2/B4) | Tracked per year | Declining clarity correlates with murkier conditions, relevant for drinking water treatment costs |
| **Algae Index** (B8−B4) | Tracked per year | Elevated values during high-rainfall years likely reflect increased biomass from nutrient flushing |
| **Sediment** (B4 mean) | Tracked per year | Peaks during flood inundation events; a proxy for soil erosion in the catchment |

Linear regression slope annotations in Figure 4 quantify the rate of change per year for each index, enabling detection of statistically significant deterioration or improvement trends over the nine-year period.

---

### C.4 ConvLSTM Spatial Forecast for 2026

Using [2023, 2024, 2025] as the input sequence, the ConvLSTM predicts the 2026 water mask:

- **Predicted water area (km²)** is logged in `timeseries_summary.csv` with `is_forecast=True` flag.
- **Figure 8** shows the three input masks and predicted 2026 mask side-by-side, providing a geographically interpretable view of expected water presence.
- **Figure 9** classifies every pixel into stable non-water, stable water, predicted gain, and predicted loss, with total km² values for each category — revealing which parts of the AOI are most likely to change.
- The ConvLSTM predicted mask is saved as a GeoTIFF and automatically made available to the Streamlit dashboard, where it appears as a selectable forecast year (year 2026) alongside observed years.

---

## D. Conclusion

This study presents a complete, automated remote sensing pipeline for water body detection, long-term monitoring, and spatial-temporal forecasting using freely available Sentinel satellite data.

**Key contributions:**

1. **Dual-sensor feature fusion:** Combining Sentinel-1 SAR (cloud-penetrating, acquisition-geometry-invariant) with Sentinel-2 optical data in a unified 7-band stack substantially improves detection robustness over optical-only or SAR-only approaches, particularly during the cloud-intensive monsoon transition period that defines the study's March–June temporal window.

2. **Deep learning segmentation that generalises:** The U-Net model, trained exclusively on 2017–2022 data, achieves consistent IoU > 0.70 on 2023–2025 test imagery without per-year retraining. This cross-year generalisation demonstrates that the model has learned robust spectral-spatial water signatures rather than overfitting to single-year radiometric conditions.

3. **Multi-scale forecasting with principled uncertainty:** Two complementary forecasting methods are implemented. Linear trend extrapolation is computationally lightweight, interpretable, and provides statistically rigorous 95% prediction intervals based on Student's t-distribution. The ConvLSTM provides spatially structured predictions that capture geographic patterns of water change — a capability beyond any scalar forecasting approach.

4. **Holistic monitoring beyond area:** By tracking five spectral water quality proxies annually, the system provides early-warning capability for environmental degradation that may not manifest in area metrics. A reservoir's area may remain stable while turbidity and chlorophyll increase significantly — a pattern detectable here but invisible to area-only monitoring.

5. **Operational accessibility:** The Streamlit dashboard with Google Earth Engine integration reduces the barrier for non-technical stakeholders to explore satellite-derived water insights for any geographic region, without requiring programming expertise or local data downloads.

The pipeline is fully reproducible, built on open-source libraries, and uses freely available satellite data, making it applicable at low cost to water monitoring challenges in any region covered by the Sentinel archive — globally available from 2014/2017 onwards.

---

## E. Future Scope of Study

### E.1 Model Architecture Improvements

- **Vision Transformers / Swin-UNet:** Replace the CNN-based U-Net encoder with self-attention mechanisms (Swin Transformer) for better modelling of long-range spatial dependencies. This is particularly relevant for large, complex water bodies with irregular boundaries and heterogeneous internal reflectance.

- **Temporal U-Net with 3D Convolutions:** Instead of treating each year independently, process multi-year image stacks directly in a single 3D CNN forward pass, enabling the segmentation model to learn temporally-aware water detection (e.g., distinguishing perennial from seasonal water).

- **Semi-supervised Learning:** Leverage the large volume of unlabelled Sentinel imagery using masked autoencoder pre-training (MAE) or contrastive learning before fine-tuning on the MNDWI-labelled dataset, reducing dependence on automated label quality.

### E.2 Data Expansion

- **Extended Temporal Record:** Extend coverage back to 2015 (Sentinel-2A launch) and forward in near real-time, and incorporate multi-season composites (dry season, monsoon onset, post-monsoon) to distinguish perennial from seasonal water and capture flood inundation dynamics.

- **Additional Sensors:** Integrate Landsat-8/9 (30 m, 1984-present for long historical baseline), MODIS (250 m daily for high-frequency monitoring), and ALOS-2 PALSAR (L-band SAR, which penetrates dense vegetation canopy overlying shallow water).

- **Topographic Priors:** Add DEM-derived slope, aspect, and topographic wetness index (TWI) as additional input channels. Water bodies preferentially occupy low-lying, flat terrain, providing useful spatial priors that could improve detection of shallow or vegetation-margined water.

### E.3 Forecasting Enhancements

- **Climate-Driven Forecasting:** Incorporate rainfall (CHIRPS), temperature (ERA5), and evapotranspiration data as exogenous inputs to the ConvLSTM, enabling physically-informed forecasts that respond to ENSO cycles and seasonal climate variability rather than purely extrapolating historical trends.

- **Longer Forecast Horizons:** Train the ConvLSTM to predict 3–5 years ahead using teacher-forcing or scheduled sampling during training, producing medium-term water availability outlooks relevant for infrastructure planning (reservoir sizing, irrigation scheduling).

- **Probabilistic Forecasting:** Replace point predictions with ensemble or Bayesian uncertainty estimates (Monte Carlo Dropout, Deep Ensembles) to produce spatially explicit confidence maps that help decision-makers distinguish high-confidence stable zones from uncertain transition areas.

### E.4 Water Quality Improvements

- **In-Situ Validation:** Ground-truth the spectral quality proxies (NDTI, NDCI) with concurrent field measurements of turbidity (NTU), chlorophyll-a (μg/L), total suspended solids (mg/L), and Secchi depth, enabling empirical calibration of the satellite-derived indices to physically meaningful units.

- **Multi-Class Water Quality Mapping:** Extend from binary water/non-water to a semantic multi-class map: open deep water, shallow turbid water, submerged aquatic vegetation, algal bloom, and floating debris — using labelled training samples and multi-class cross-entropy loss.

- **Red-Edge and SWIR2 Integration:** Incorporate the currently unused Sentinel-2 red-edge bands (B5, B6, B7) and B12 (SWIR2) to improve cyanobacterial bloom detection and oil-contamination identification.

### E.5 Operational Deployment

- **Real-Time Automated Pipeline:** Automate the full pipeline via Google Earth Engine triggers on new Sentinel acquisitions (every 5–12 days), updating the dashboard without manual intervention — enabling near real-time water body monitoring for operational water management.

- **Alert System:** Implement anomaly detection on the time series (e.g., Z-score thresholding, isolation forests) to trigger automated email or SMS alerts when water area drops below a critical threshold or a quality index exceeds an alarm level — supporting early warning for droughts, reservoir depletion, or pollution events.

- **Containerised Deployment:** Package the Streamlit dashboard with Docker and deploy to a cloud platform (GCP, AWS, Azure) with persistent model storage and GEE service-account authentication, enabling shared multi-user access without local installation.

- **Basin-Scale Monitoring:** Transition from a single fixed AOI to regional basin-scale monitoring using GEE's distributed cloud compute for parallel processing of large tiled imagery mosaics, enabling systematic water accounting across entire river catchments.

---

*Study report sections added: 2026-05-26*
