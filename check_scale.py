"""Quick check: compare band value ranges between old and new data."""
import rasterio, numpy as np, os

old_path = r'D:\rfWater\DATA_DIR\S2_March_June_2017.tif'
new_dir  = r'D:\rfWater\DATA_DIR_TS'

# Find any new file
new_file = None
for f in os.listdir(new_dir):
    if f.startswith('S2_') and f.endswith('.tif'):
        new_file = os.path.join(new_dir, f)
        break

def stats(arr):
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        return 'no valid pixels'
    return (f'min={valid.min():.4f}  max={valid.max():.4f}  '
            f'mean={valid.mean():.4f}  valid_pct={100*len(valid)/arr.size:.1f}%')

print('=== OLD DATA (training) ===')
with rasterio.open(old_path) as src:
    old = src.read().astype(np.float32)
    print(f'Bands  : {src.count}')
    print(f'All    : {stats(old)}')
    print(f'B3(idx1): {stats(old[1])}')
    print(f'nodata : {src.nodata}')

print()
if new_file:
    print(f'=== NEW DATA ({os.path.basename(new_file)}) ===')
    with rasterio.open(new_file) as src:
        new = src.read().astype(np.float32)
        print(f'Bands  : {src.count}')
        print(f'All    : {stats(new)}')
        print(f'B3(idx1): {stats(new[1])}')
        print(f'B8(idx6): {stats(new[6])}')
        print(f'nodata : {src.nodata}')
else:
    print('No new S2 file found in DATA_DIR_TS')
