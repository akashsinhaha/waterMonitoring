"""
Evaluation script for the trained U-Net water segmentation model.

For each test year produces:
  - Numerical metrics vs MNDWI ground-truth mask
  - Side-by-side visual comparison (RGB | MNDWI mask | Predicted mask | Difference)
  - Summary table across all test years

Run:
    python evaluate_unet.py
"""

import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

# ── Config (must match water_classification_unet.py) ─────────────────────────
DATA_DIR        = r'D:\rfWater\DATA_DIR'
RESULTS_DIR     = r'D:\rfWater\results_unet'
TEST_YEARS      = [2023, 2024, 2025]
MNDWI_THRESHOLD = 0.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def calculate_mndwi(green, swir):
    with np.errstate(divide='ignore', invalid='ignore'):
        mndwi = (green - swir) / (green + swir)
        mndwi[np.isnan(mndwi)] = 0
        mndwi[np.isinf(mndwi)] = 0
    return mndwi


def load_ground_truth(year):
    """
    Derive MNDWI water mask directly from source imagery.

    Returns
    -------
    water_mask : np.ndarray  (H, W)  uint8
    s2_bands   : np.ndarray  (5, H, W)  float32
    """
    s2_path = os.path.join(DATA_DIR, f'S2_March_June_{year}.tif')
    with rasterio.open(s2_path) as src:
        s2_bands = src.read().astype(np.float32)

    green      = s2_bands[1]
    swir       = s2_bands[4]
    mndwi      = calculate_mndwi(green, swir)
    water_mask = (mndwi > MNDWI_THRESHOLD).astype(np.uint8)
    return water_mask, s2_bands


def load_predictions(year):
    """Load U-Net binary mask and probability map saved by the training script."""
    mask_path = os.path.join(RESULTS_DIR, f'water_mask_unet_{year}.tif')
    prob_path = os.path.join(RESULTS_DIR, f'probability_map_unet_{year}.tif')

    with rasterio.open(mask_path) as src:
        predicted_mask = src.read(1).astype(np.uint8)

    with rasterio.open(prob_path) as src:
        probability_map = src.read(1).astype(np.float32)

    return predicted_mask, probability_map


def compute_metrics(ground_truth, prediction):
    """
    Compute binary segmentation metrics.

    Returns dict: accuracy, precision, recall, f1, iou,
                  water_gt_pct, water_pred_pct, tp, fp, fn, tn
    """
    gt   = ground_truth.flatten().astype(bool)
    pred = prediction.flatten().astype(bool)

    tp = np.sum( gt &  pred)
    fp = np.sum(~gt &  pred)
    fn = np.sum( gt & ~pred)
    tn = np.sum(~gt & ~pred)

    accuracy  = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    iou       = tp / (tp + fp + fn + 1e-8)

    return {
        'accuracy'      : accuracy,
        'precision'     : precision,
        'recall'        : recall,
        'f1'            : f1,
        'iou'           : iou,
        'water_gt_pct'  : gt.mean()   * 100,
        'water_pred_pct': pred.mean() * 100,
        'tp': int(tp), 'fp': int(fp),
        'fn': int(fn), 'tn': int(tn),
    }


def make_rgb(s2_bands, percentile=2):
    """Create display RGB from S2 bands (R=B4, G=B3, B=B2) with percentile stretch."""
    rgb = np.stack([s2_bands[2], s2_bands[1], s2_bands[0]], axis=-1)
    for channel in range(3):
        low, high = np.percentile(rgb[:, :, channel], [percentile, 100 - percentile])
        rgb[:, :, channel] = np.clip(
            (rgb[:, :, channel] - low) / (high - low + 1e-8), 0, 1
        )
    return rgb.astype(np.float32)


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_comparison(year, rgb, gt_mask, pred_mask, prob_map, metrics):
    """
    4-panel figure:
      Panel 1 — RGB composite
      Panel 2 — MNDWI ground-truth mask
      Panel 3 — U-Net probability map
      Panel 4 — Difference (TP / FP / FN / TN)
    """
    # Encode difference into 4 classes
    diff = np.zeros_like(gt_mask, dtype=np.uint8)
    diff[(gt_mask == 1) & (pred_mask == 1)] = 1   # TP — correct water
    diff[(gt_mask == 0) & (pred_mask == 1)] = 2   # FP — false alarm
    diff[(gt_mask == 1) & (pred_mask == 0)] = 3   # FN — missed water

    diff_cmap   = mcolors.ListedColormap(['#1a1a2e', '#00b4d8', '#ef233c', '#f77f00'])
    diff_labels = ['True Negative', 'True Positive', 'False Positive', 'False Negative']
    diff_colors = ['#1a1a2e', '#00b4d8', '#ef233c', '#f77f00']

    fig, axes = plt.subplots(1, 4, figsize=(24, 7))
    fig.suptitle(
        f'U-Net Evaluation — {year}\n'
        f'IoU={metrics["iou"]:.3f}  |  F1={metrics["f1"]:.3f}  |  '
        f'Precision={metrics["precision"]:.3f}  |  Recall={metrics["recall"]:.3f}  |  '
        f'Accuracy={metrics["accuracy"]:.3f}',
        fontsize=13, fontweight='bold', y=1.01
    )

    # Panel 1: RGB
    axes[0].imshow(rgb)
    axes[0].set_title('RGB Composite (S2 B4/B3/B2)', fontsize=11)
    axes[0].axis('off')

    # Panel 2: MNDWI ground truth
    axes[1].imshow(gt_mask, cmap='Blues', vmin=0, vmax=1, interpolation='nearest')
    axes[1].set_title(f'MNDWI Ground Truth\n{metrics["water_gt_pct"]:.2f}% water', fontsize=11)
    axes[1].axis('off')

    # Panel 3: Probability map
    im = axes[2].imshow(prob_map, cmap='Blues', vmin=0, vmax=1)
    axes[2].set_title(f'U-Net Probability Map\n{metrics["water_pred_pct"]:.2f}% predicted water', fontsize=11)
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.03, pad=0.04, label='Water probability')

    # Panel 4: Difference map
    axes[3].imshow(diff, cmap=diff_cmap, vmin=0, vmax=3, interpolation='nearest')
    axes[3].set_title(
        f'Difference Map\nTP={metrics["tp"]:,}  FP={metrics["fp"]:,}  FN={metrics["fn"]:,}',
        fontsize=11
    )
    axes[3].axis('off')
    legend_elements = [
        Patch(facecolor=diff_colors[i], label=diff_labels[i]) for i in range(4)
    ]
    axes[3].legend(handles=legend_elements, loc='lower right', fontsize=8, framealpha=0.85)

    plt.tight_layout()
    output_path = os.path.join(RESULTS_DIR, f'evaluation_{year}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {output_path}')


def plot_summary(all_metrics):
    """Bar chart comparing IoU, F1, Precision, Recall across test years."""
    years   = list(all_metrics.keys())
    metrics = ['iou', 'f1', 'precision', 'recall']
    labels  = ['IoU', 'F1', 'Precision', 'Recall']
    colors  = ['#00b4d8', '#0077b6', '#48cae4', '#90e0ef']

    x      = np.arange(len(years))
    width  = 0.2
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        values = [all_metrics[y][metric] for y in years]
        bars   = ax.bar(x + i * width, values, width, label=label, color=color)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Year')
    ax.set_ylabel('Score')
    ax.set_title('U-Net Performance by Year')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(years)
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='IoU=0.7 target')

    plt.tight_layout()
    output_path = os.path.join(RESULTS_DIR, 'evaluation_summary.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {output_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('=' * 65)
    print('U-NET EVALUATION')
    print('=' * 65)

    all_metrics = {}

    for year in TEST_YEARS:
        print(f'\n── Year {year} ──────────────────────────────────────')

        gt_mask, s2_bands       = load_ground_truth(year)
        pred_mask, prob_map     = load_predictions(year)
        metrics                 = compute_metrics(gt_mask, pred_mask)
        all_metrics[year]       = metrics

        print(f'  GT water coverage   : {metrics["water_gt_pct"]:.2f}%')
        print(f'  Pred water coverage : {metrics["water_pred_pct"]:.2f}%')
        print(f'  Accuracy            : {metrics["accuracy"]:.4f}')
        print(f'  Precision           : {metrics["precision"]:.4f}')
        print(f'  Recall              : {metrics["recall"]:.4f}')
        print(f'  F1 Score            : {metrics["f1"]:.4f}')
        print(f'  IoU                 : {metrics["iou"]:.4f}')
        print(f'  TP={metrics["tp"]:,}  FP={metrics["fp"]:,}  FN={metrics["fn"]:,}  TN={metrics["tn"]:,}')

        rgb = make_rgb(s2_bands)
        plot_comparison(year, rgb, gt_mask, pred_mask, prob_map, metrics)

    # Summary table
    print('\n' + '=' * 65)
    print(f'{"Year":<6} {"IoU":>6} {"F1":>6} {"Prec":>7} {"Recall":>8} {"GT%":>6} {"Pred%":>7}')
    print('-' * 65)
    for year, m in all_metrics.items():
        print(f'{year:<6} {m["iou"]:>6.3f} {m["f1"]:>6.3f} '
              f'{m["precision"]:>7.3f} {m["recall"]:>8.3f} '
              f'{m["water_gt_pct"]:>6.2f} {m["water_pred_pct"]:>7.2f}')

    plot_summary(all_metrics)

    print(f'\nAll outputs saved to: {RESULTS_DIR}')


if __name__ == '__main__':
    main()
