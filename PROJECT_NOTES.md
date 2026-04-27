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

#### Step 1: Loading a Year's Data — `load_year(year)`

1. Opens the Sentinel-2 file for that year → shape `(5, H, W)`
2. Opens both Sentinel-1 files (ascending and descending)
3. **Reprojects** (re-aligns/warps) the S1 images so their pixels line up exactly with the S2 pixels — because the two satellites have different viewing angles, the pixels do not naturally match up
4. Stacks all 7 bands together → shape `(7, H, W)`
5. Computes MNDWI from B3 and B11:
   ```
   MNDWI = (B3 - B11) / (B3 + B11)
   water_mask = (MNDWI > 0.0)  → True/False per pixel
   ```
6. Returns the 7-band image stack and the binary water mask (0 = not water, 1 = water)

#### Step 2: Patch Generation — `generate_patches(years, split_name)`

The full satellite image is huge — too big to feed into a neural network all at once. So we cut it into small **256×256 pixel patches**, like cutting a big poster into puzzle pieces.

- **Patch size:** 256 × 256 pixels
- **Stride:** 128 pixels — meaning each patch overlaps the previous one by 128 pixels (half overlap). This overlap ensures the model sees every region from multiple perspectives.
- For each patch, we save:
  - An image patch: shape `(7, 256, 256)` — the 7 bands for those 256×256 pixels
  - A mask patch: shape `(256, 256)` — 0s and 1s showing where the water is

Patches are saved as GeoTIFF files (satellite image files that know their real-world GPS location).

**Patch naming:** `patch_2017_r0_c0.tif` means the patch from year 2017, starting at row 0, column 0.

#### Step 3: The U-Net Architecture

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
  Final: (64→1 channel, 256×256)  — one water probability per pixel
Output: (1, 256, 256)  — water probability map
```

**Key building blocks:**

- **`DoubleConv`** — Two back-to-back "convolution" steps. A convolution is like sliding a small detector window (3×3 pixels) across the image to find patterns. After each convolution: BatchNorm (keeps numbers stable) → ReLU (sets all negative numbers to zero, adds non-linearity).

- **`EncoderBlock`** — A DoubleConv followed by **MaxPool2d** (which halves the image size by keeping only the maximum value in each 2×2 region, like zooming out). Returns both the pre-pool result (the "skip connection") and the shrunken output.

- **`DecoderBlock`** — Expands the image back up using **ConvTranspose2d** (reverse convolution, like zooming in). Then concatenates the skip connection from the matching encoder level (this is why it is called a skip — it skips across the U shape) and applies DoubleConv.

**Why skip connections?** The encoder loses fine spatial detail as it shrinks. The skip connections carry that fine detail directly from the encoder side to the decoder side, so the final output has both broad understanding (from the bottleneck) and fine pixel-level detail (from the skips).

#### Step 4: The Loss Function — `DiceBCELoss`

A loss function measures "how wrong is the model right now?" Lower is better. Training adjusts the model's internal numbers (weights) to minimize this loss.

This uses two loss functions added together:

**BCE (Binary Cross-Entropy) Loss:**
```
BCE = -[y * log(p) + (1-y) * log(1-p)]
```
- `y` = true label (0 or 1)
- `p` = predicted probability (0 to 1 after sigmoid)
- Punishes the model when it is confidently wrong

**Dice Loss:**
```
Dice Loss = 1 - (2 × |Prediction ∩ Truth| + smooth) / (|Prediction| + |Truth| + smooth)
```
- `|Prediction ∩ Truth|` = number of pixels correctly predicted as water (intersection)
- `smooth = 1e-6` = a tiny number added to avoid dividing by zero
- Dice loss is great for imbalanced data (when water pixels are much rarer than land pixels)

**Total loss = BCE + Dice**

**Why both?** BCE focuses on each pixel individually. Dice focuses on the overall overlap between predicted and true water regions. Together they train a more reliable model.

#### Step 5: Training Loop — `train_epoch()` and `train()`

- **Optimizer:** Adam (an algorithm that adjusts model weights step by step in the direction that reduces loss)
- **Learning Rate:** `0.0001` — how big each adjustment step is
- **Epochs:** 10 — the whole training dataset is seen 10 times
- **Batch size:** 8 — 8 patches are processed together at once (averaging their gradients makes training more stable)
- **Batches per epoch:** 80 (a cap to speed things up)
- **ReduceLROnPlateau scheduler:** If the test loss stops improving for 5 epochs, the learning rate is cut in half

After each epoch, the model is evaluated. If the **IoU** (see below) is better than before, the model's weights are saved to `unet_best.pth`.

**IoU (Intersection over Union):**
```
IoU = |Prediction ∩ Truth| / |Prediction ∪ Truth|
```
- How much the predicted water region overlaps the true water region
- 1.0 = perfect match, 0.0 = no overlap at all

#### Step 6: Full Image Prediction — `predict_full_image(model, year)`

To predict the water mask for a whole satellite image:
1. Slide the 256×256 patch window across the full image with stride 128
2. For each patch, run it through the trained U-Net → get a 256×256 probability map
3. Where patches overlap, **average** the probabilities (so overlapping predictions vote together)
4. Apply threshold: probability > 0.5 → water
5. Save two output files:
   - `probability_map_unet_{year}.tif` — values 0.0 to 1.0 per pixel
   - `water_mask_unet_{year}.tif` — binary 0/1 per pixel

---

### 4. `evaluate_unet.py` — The Report Card

After training, this script measures how good the model actually is on the test years (2023, 2024, 2025).

**It uses MNDWI-derived masks as the "ground truth" to compare against.**

#### Metrics computed — `compute_metrics()`

For every pixel in the full image, there are 4 possible outcomes:

| | Predicted: Water | Predicted: Not Water |
|---|---|---|
| **Actually Water** | TP (True Positive) ✓ | FN (False Negative) ✗ |
| **Actually Not Water** | FP (False Positive) ✗ | TN (True Negative) ✓ |

From these four counts, five scores are calculated:

```
Accuracy  = (TP + TN) / (TP + FP + FN + TN)
            "What fraction of all pixels did we get right?"

Precision = TP / (TP + FP)
            "Of all pixels we called water, how many really were water?"

Recall    = TP / (TP + FN)
            "Of all pixels that were actually water, how many did we find?"

F1        = 2 × Precision × Recall / (Precision + Recall)
            "A balanced average of precision and recall (harmonic mean)"

IoU       = TP / (TP + FP + FN)
            "How much do the two water regions overlap?"
```

#### Output figures:

- **`evaluation_{year}.png`** — A 4-panel figure showing:
  1. RGB composite of the satellite image (what it looks like to a human eye)
  2. MNDWI ground truth water mask (blue = water)
  3. U-Net probability map (gradient from white to blue)
  4. Difference map with colour coding: dark = correct non-water, blue = correct water (TP), red = false alarm (FP), orange = missed water (FN)

- **`evaluation_summary.png`** — A bar chart comparing IoU, F1, Precision, Recall across the three test years side by side

---

### 5. `water_timeseries.py` — The Change Tracker

This script loads the trained U-Net model and runs it on **all 9 years** (2017–2025) to measure how the water body has changed over time. It also computes water quality indicators from the optical bands.

#### Data Loading — `load_year_ts(year)`

This uses a **different data folder** (`DATA_DIR_TS`) which has the newer 10-band S2 format with values in 0–1 reflectance scale.

The 5 S2 bands needed by the U-Net are extracted from these positions:
```python
S2_UNET_IDX = [0, 1, 2, 6, 8]  # → B2, B3, B4, B8, B11
```
Then multiplied by 10,000 to convert from reflectance back to DN:
```
s2_unet = s2_full[S2_UNET_IDX] × 10000
```
This rescaling is essential because the model was trained on DN-range data.

#### Water Mask Generation — `predict_water_mask()`

Same patch-based inference as in `water_classification_unet.py`:
- Patch size: 256×256, Stride: 128
- Sigmoid formula applied manually:
  ```
  probability = 1 / (1 + exp(-logit))
  ```
  (This is mathematically identical to `torch.sigmoid()`)
- Threshold: probability > 0.5 → water

#### Water Quality Indices — `compute_quality_indices()`

These are computed only on pixels that are water (inside the water mask). They are simple formulas using the 10 S2 bands (in 0–1 reflectance):

```
Band abbreviations (new 10-band layout):
B2=blue, B3=green, B4=red, B5=RE1, B6=RE2, B8=NIR, B11=SWIR1, B12=SWIR2
```

| Index | Formula | Meaning |
|---|---|---|
| **MNDWI** | `(B3 - B11) / (B3 + B11)` | Water extent quality / standing water signal |
| **NDTI** (Turbidity) | `(B4 - B3) / (B4 + B3)` | Higher = more turbid (murkier) water |
| **NDCI** (Chlorophyll-a) | `(B5 - B4) / (B5 + B4)` | Higher = more algae / phytoplankton |
| **Clarity** | `B2 / (B4 + 1e-8)` | Blue-to-red ratio; higher = clearer water |
| **Algae** | `B8 - B4` | Near-infrared minus red; detects floating algae |
| **Sediment** | `B4` (raw) | Red band alone; brighter = more suspended sediment |

All normalized-difference formulas follow the same pattern: `(a - b) / (a + b)`, which always produces a value between -1 and +1.

#### Area Calculation

```
pixel_area = 10m × 10m = 100 m²
water_area_km² = count_of_water_pixels × 100 / 1,000,000
```

#### Output Files

```
results_timeseries/
  water_masks/
    water_mask_{year}.tif        ← binary GeoTIFF for each year
  timeseries_summary.csv         ← table: year, area_km2, NDTI, NDCI, clarity, etc.
  01_water_masks_grid.png        ← 3×3 grid showing all 9 years' water maps
  02_area_timeseries.png         ← line chart (absolute area) + bar chart (change)
  03_change_maps.png             ← 8 panels showing year-to-year gain/loss
  04_quality_timeseries.png      ← 5 quality index time series with trend lines
  05_summary_dashboard.png       ← overview combining RGB, masks, area chart, quality
```

#### Change Detection Logic (in `plot_change_maps`)

For every pair of consecutive years (e.g. 2017 and 2018), every pixel is classified into 4 states:

```
mask_a = water mask for year A
mask_b = water mask for year B

diff = 0  →  stable non-water  (was not water, still not water)
diff = 1  →  stable water      (was water, still water)
diff = 2  →  water GAIN        (was not water, now is water)
diff = 3  →  water LOSS        (was water, now is not water)
```

#### Trend Line Calculation (in charts)

```python
z = np.polyfit(years, values, 1)  # fits y = slope × x + intercept
```
A straight line (degree 1 polynomial) is fitted through the data points. The slope (`z[0]`) tells you the average change per year.

---

### 6. `dashboard/` — The Interactive Web App

This is a **Streamlit** web application (a Python library for making web dashboards). Run it with `streamlit run dashboard/app.py`. It reads the pre-computed results and lets the user explore them interactively in a browser.

#### `dashboard/config.py`
Stores shared constants:
- `RESULTS_DIR` — path to `results_timeseries/`
- `YEARS` — list 2017 to 2025
- `PIXEL_AREA_KM2 = 0.0001` — 100 m² converted to km²
- `MAP_OVERLAY_MAX_PX = 1200` — cap on image overlay resolution for browser performance

#### `dashboard/utils/data_loader.py` — Data Fetching
All functions are decorated with `@st.cache_data`, meaning Streamlit saves the result in memory so it is only computed once, not every time the user clicks something.

Key functions:
- **`load_summary()`** — reads `timeseries_summary.csv` into a pandas DataFrame (a table)
- **`load_water_mask(year)`** — reads the binary GeoTIFF for that year
- **`get_map_bounds()`** — converts the raster's CRS (Coordinate Reference System, e.g. UTM) to WGS84 latitude/longitude so Folium (the map library) knows where to centre the map
- **`mask_to_overlay_png(year)`** — turns the binary water mask into a semi-transparent blue PNG image encoded as base64 (a text string) for embedding in the map
- **`change_map_to_overlay_png(year_a, year_b)`** — builds a coloured difference image (blue=stable, green=gain, red=loss) for the change detection view

**Why base64?** Web browsers cannot directly read GeoTIFF files. Converting to a PNG and then encoding it as a base64 text string lets Folium embed it directly in the HTML without needing a separate file server.

#### `dashboard/utils/map_utils.py` — The Map Builder
Uses the **Folium** library to build interactive maps (the same engine behind many online map tools).

- **`create_base_map()`** — Creates a dark-themed base map, adds three tile layer options (dark, satellite, street), draws a white rectangle showing the Area of Interest (AOI) boundary, and adds a mini-map in the corner and a coordinate display at the bottom
- **`add_water_mask_layer()`** — Overlays the blue water PNG on top of the map
- **`add_change_layer()`** — Overlays the change detection PNG with a legend
- **`finalise_map()`** — Adds the layer switcher control (so the user can toggle layers)

#### `dashboard/utils/charts.py` — The Charts
Uses **Plotly** (an interactive charting library) to make all the graphs.

- **`area_timeseries_chart()`** — Line chart of water area over time with a shaded fill below and a dotted trend line
- **`yoy_change_chart()`** — Bar chart of year-over-year change: green bars = water gain, red bars = water loss
- **`quality_timeseries_chart()`** — Multi-line chart showing whichever quality metrics the user has ticked in the sidebar, with a dotted trend line for each
- **`change_summary_chart()`** — A stacked 2-panel chart: top panel is the area line, bottom panel is the change bars

#### `dashboard/app.py` — The Main App UI
Lays out everything the user sees:

1. **Sidebar** — Mode toggle (Water Mask / Change Detection), year slider, year range slider, quality metric checkboxes
2. **Row 1:** Left side = interactive map; Right side = metric cards showing water area, % change, and quality indices with arrows showing if they went up or down versus the previous year
3. **Row 2:** Three tabs:
   - **Water Area** — area chart + change bar chart + data table
   - **Water Quality** — multi-line quality index chart + data table
   - **Change Summary** — stacked area+change chart + overall start/end summary

---

## How All Files Connect — The Full Pipeline

```
Satellite Images (DATA_DIR/)
          │
          ▼
water_classification_unet.py
  ├── Loads 7-band image stacks
  ├── Cuts into 256×256 patches (saved in patches/)
  ├── Trains U-Net model (10 epochs)
  └── Saves unet_best.pth + predictions for 2023/2024/2025
          │
          ▼
evaluate_unet.py
  ├── Loads predictions and computes TP/FP/FN/TN metrics
  └── Saves evaluation_*.png figures
          │
          ▼
water_timeseries.py
  ├── Loads unet_best.pth
  ├── Runs inference on all 9 years (DATA_DIR_TS/)
  ├── Saves water_mask_{year}.tif files
  ├── Computes quality indices per year
  └── Saves timeseries_summary.csv + all chart PNGs
          │
          ▼
dashboard/app.py  (run with: streamlit run dashboard/app.py)
  ├── Reads timeseries_summary.csv
  ├── Reads water_mask_{year}.tif files
  └── Shows interactive maps, charts, and statistics in browser
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
| Sigmoid | `1 / (1 + exp(-x))` | Converts model output to 0–1 probability |
| Dice Loss | `1 - (2·TP + ε) / (Pred + Truth + ε)` | Measures overlap quality |
| IoU | `TP / (TP + FP + FN)` | Intersection over Union accuracy |
| F1 Score | `2·P·R / (P + R)` | Balanced precision-recall score |
| Pixel Area | `10 × 10 = 100 m²` | Area of one satellite pixel |
| Water Area | `count × 100 / 1,000,000 km²` | Total water area in km² |
| Trend Line | `y = slope × x + intercept` | Linear trend over years |

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
| Learning rate | 0.0001 |
| Water threshold (MNDWI) | > 0.0 |
| Water threshold (U-Net) | probability > 0.5 |
| Training years | 2017–2022 (6 years) |
| Test/prediction years | 2023–2025 (3 years) |
| Old S2 value range | 0–10,000 (Digital Numbers) |
| New S2 value range | 0–1 (Reflectance) |
| Scale factor applied | × 10,000 (to convert new → old range) |
