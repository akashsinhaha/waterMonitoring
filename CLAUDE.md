# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **water body detection system** using satellite imagery (Sentinel-1 and Sentinel-2). It trains ML ensemble models on multi-temporal data (2017-2022) and predicts water masks for test years (2023-2025).

## Running the Scripts

```bash
# Full pipeline: train ensemble + evaluate + predict
python water_classification_ensemble.py

# Individual model variants
python water_classification_rf.py
python water_classification_svm.py
python water_classification_xgboost.py

# Run inference with pre-trained models
python predict_new_ensemble.py
```

No test suite or linter configuration exists.

## Dependencies

No `requirements.txt` exists. Install manually:

```bash
pip install numpy rasterio geopandas scikit-learn xgboost matplotlib seaborn pandas shapely joblib
```

## Architecture

### Data Flow

```
DATA_DIR/
  S2_March_June_{year}.tif        # Sentinel-2: 5 bands (B2, B3, B4, B8, B11)
  S1_ASC_March_June_{year}.tif    # Sentinel-1 Ascending: VV polarization
  S1_DESC_March_June_{year}.tif   # Sentinel-1 Descending: VV polarization
  WaterPoints_Clipped_March_June_{year}.shp  # Ground truth water points
        ↓
  Feature extraction (7 bands per pixel + MNDWI index)
        ↓
  Train on 2017–2022, test on 2023–2025
        ↓
  results_ensemble/  (GeoTIFFs, PNGs, saved models)
```

### Feature Engineering

7 input features per pixel: 5 Sentinel-2 bands + 2 Sentinel-1 VV bands (ascending + descending). MNDWI `(B3 - B11) / (B3 + B11)` is computed separately to guide non-water sample generation, not as a model feature.

### Training Sample Generation

- **Water samples:** extracted from annual ground-truth shapefiles (~50/year)
- **Non-water samples:** pixels where MNDWI < threshold with a buffer excluding known water

### Ensemble Model (`WaterMaskEnsemble`)

Combines three models via soft voting (probability averaging):

| Component | Key Hyperparameters |
|-----------|---------------------|
| Random Forest | `n_estimators=200, max_depth=20, class_weight='balanced'` |
| SVM | `kernel='rbf', C=10, probability=True` + `StandardScaler` |
| XGBoost | `max_depth=7, lr=0.1, n_estimators=200, scale_pos_weight` tuned |

### Key Methods (shared across all classifier classes)

| Method | Purpose |
|--------|---------|
| `load_sentinel_data(year)` | Load and reproject S1/S2 rasters to common CRS |
| `generate_non_water_points()` | MNDWI-based non-water sample selection |
| `prepare_training_data()` | Aggregate features across 2017–2022 |
| `train_model(X, y)` | Fit ensemble (or single model) |
| `evaluate_model(X_test, y_test)` | Metrics + confusion matrix + feature importance plots |
| `predict_full_image(year)` | Batch-predict full raster (10,000 px/batch) |
| `save_models()` / `load_models()` | Persist to `results_ensemble/ensemble_models/` |

### Output Structure

```
results_ensemble/
  ensemble_models/         # rf_model.pkl, svm_model.pkl, scaler.pkl, xgb_model.json
  water_mask_ensemble_{year}.tif
  confusion_matrix_ensemble.png
  feature_importance_ensemble.png
  comprehensive_visualization_ensemble_{year}.png
  multi_year_comparison_ensemble.png
```

## Configuration

All configuration (data directory, train/test year splits, hyperparameters) is **hardcoded inside each script's `main()` and class `__init__`**. There is no config file. Change values directly in the source.

- Data root: `D:\rfWater\DATA_DIR`
- Training years: 2017–2022
- Test/prediction years: 2023–2025
