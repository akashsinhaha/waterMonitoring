"""
Dependency installer for rfWater project.

Run with:
    python install_dependencies.py

Installs PyTorch with CUDA 12.4 support (compatible with CUDA 13.x drivers)
and all other required packages.
"""

import subprocess
import sys

def run(command, description):
    print(f'\n>>> {description}')
    print(f'    {command}\n')
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f'[ERROR] Failed: {description}')
        sys.exit(1)

pip = f'"{sys.executable}" -m pip'

# ── PyTorch with CUDA 12.4 (works on CUDA 13.x drivers) ──────────────────────
run(
    f'{pip} install torch torchvision '
    f'--index-url https://download.pytorch.org/whl/cu124',
    'Installing PyTorch with CUDA 12.4 support'
)

# ── Geospatial packages ───────────────────────────────────────────────────────
run(f'{pip} install rasterio',    'Installing rasterio')
run(f'{pip} install geopandas',   'Installing geopandas')
run(f'{pip} install shapely',     'Installing shapely')

# ── ML packages ──────────────────────────────────────────────────────────────
run(f'{pip} install scikit-learn', 'Installing scikit-learn')
run(f'{pip} install xgboost',      'Installing xgboost')

# ── Data & visualisation ──────────────────────────────────────────────────────
run(f'{pip} install numpy',       'Installing numpy')
run(f'{pip} install pandas',      'Installing pandas')
run(f'{pip} install matplotlib',  'Installing matplotlib')
run(f'{pip} install seaborn',     'Installing seaborn')
run(f'{pip} install joblib',      'Installing joblib')

# ── Verify GPU is visible to PyTorch ─────────────────────────────────────────
print('\n' + '=' * 50)
print('VERIFICATION')
print('=' * 50)

verify = subprocess.run(
    [sys.executable, '-c',
     'import torch\n'
     'print("PyTorch :", torch.__version__)\n'
     'print("CUDA available :", torch.cuda.is_available())\n'
     'if torch.cuda.is_available():\n'
     '    print("GPU :", torch.cuda.get_device_name(0))\n'
     '    vram = torch.cuda.get_device_properties(0).total_memory / 1e9\n'
     '    print(f"VRAM : {vram:.1f} GB")\n'
    ]
)
if verify.returncode != 0:
    print('[ERROR] Verification failed — PyTorch may not have installed correctly.')
