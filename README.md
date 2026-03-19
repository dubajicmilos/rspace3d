
<img width="2007" height="512" alt="logo" src="https://github.com/user-attachments/assets/c2070640-e0fe-4d57-8f6e-a71db7082ba8" />


**Reciprocal Space 3D Processing and visualization Toolkit** for single-crystal X-ray diffuse scattering.

Process CrysAlisPro reciprocal space cuts that are stored as .img files into symmetrized 3D volume with GPU acceleration. 

<img width="1297" height="777" alt="image" src="https://github.com/user-attachments/assets/d99736f6-08b5-4f09-9e1d-eff895192ee0" />


## What it does

CrysAlisPro does not offer a reliable way to extract full 3D reciprocal space data from the raw measured frames. Using dc unwarp function it is possible to extract slices of reciprocal space. However, analyzing this data becomes difficult as exported files do not contain information about hkl grid. This software is a workaround to extract full 3D reciprocal space that is recontructed in CrysAlisPro into a single .h5 file with HKL grid calcuated from the UB matrix stored in CrysAlisPro .par file. This software offer also offers notebooks, CLI commands and guis for both processing and visualization. 

<img width="1408" height="768" alt="rspace3d" src="https://github.com/user-attachments/assets/ceb25190-5b8a-4a25-bb88-c54f1f2eb413" />

## Installation

### From source

```bash
git clone https://github.com/yourusername/rspace3d.git
cd rspace3d
pip install -e .
```

### Optional dependencies

```bash
pip install cupy-cuda12x   # GPU acceleration (6-10x faster)
pip install pyvista         # 3D isosurface visualization
pip install plotly           # Alternative 3D (browser-based)
```

### Standalone executables

Download from [Releases](https://github.com/yourusername/rspace3d/releases) — no Python installation needed:

- **rspace3d-viewer.exe** — View .img files and .h5 volumes
- **rspace3d-builder.exe** — One-button volume processing
- **rspace3d-process.exe** — Command-line processing

To build executables yourself:
```bash
pip install pyinstaller
python build_exe.py
```

## Quick Start

### Step 1: Prepare unwarp layer list

Before processing, you need CrysAlisPro to reconstruct HK planes at each l value. Generate the layer list file:

**GUI:**
```bash
python -m rspace3d.volume_builder_gui
```
Use the "Generate dcunwarp file" section. Set l range (e.g., -6 to 6), step (0.02), and resolution (0.8 A). Click Generate.

**CLI:**
```bash
python -m rspace3d.make_dcunwarp -6 6 0.02 0.8
```

If more than 500 layers are needed, multiple files are created automatically (CrysAlisPro limit).

### Step 2: Unwarp in CrysAlisPro

In the CrysAlisPro command line:
```
dc unwarp
```
Load the dcunwrap file we created previosly. 

<img width="589" height="558" alt="image" src="https://github.com/user-attachments/assets/394f02f9-40e8-4a8d-a874-c0f3347c9b5c" />


This creates numbered .img files in the `unwarp/` folder (e.g., `MAPbBr3_300K_1.img` through `MAPbBr3_300K_601.img`).

If multiple `.dcunwarp` files were generated, run `dc unwarp` once for each file.

### Step 3: Process the volume

**One-button GUI:**
```bash
python -m rspace3d.volume_builder_gui
```
1. Browse to the `unwarp/` folder. The GUI will also load cracker.par file which is usually in parent folder. This is used to calcuate HKL grid as .par file contains UB matrix and other info. 
2. Set Laue group (e.g., m-3m for cubic). This will be used both for outlier rejection and symetrization. 
3. Set binning (2x2 recommended), sigma (e.g. 3.0, the higher the less harsh the outlier rejection is), iterations (1)
4. Click **Process All**

Output: `sample_raw.h5` (unbinned) and `sample_sym_mbar3m.h5` (processed)

**Command line:**
```bash
python scripts/volume_process.py F:\path\to\unwarp --laue m-3m --sigma 3 --bin 2
```

### Step 4: View the results

**Unified viewer** (handles .img, .cbf, and .h5):
```bash
python -m rspace3d.rsp_viewer path/to/volume.h5
```

Features:
- Plane selection: HK (fix L), HL (fix K), KL (fix H)
- Slider to scrub through slices
- Integration over a range of slices
- Tilted Miller index grid overlay (correct for all crystal systems)
- Interactive line profile tool
- 3D isosurface visualization (PyVista)
- Export to PNG/PDF/SVG

## Tools

| Tool | What it does | How to run |
|------|-------------|------------|
| **Unified Viewer** | View .img/.cbf/.h5 files | `python -m rspace3d.rsp_viewer` |
| **Volume Processor GUI** | One-button processing pipeline | `python -m rspace3d.volume_builder_gui` |
| **Volume Processor CLI** | Scriptable processing | `python scripts/volume_process.py folder [options]` |
| **dcunwarp Generator** | Create CrysAlisPro layer lists | `python -m rspace3d.make_dcunwarp lmin lmax step [res]` |

## Python API

```python
from rspace3d import (
    load_unwarp_folder, bin_volume, reject_outliers,
    symmetrize_volume, save_volume_h5, load_volume_h5,
    extract_volume_slice, read_rsp_layer
)

# Load and process
vol = load_unwarp_folder('path/to/unwarp', bin_xy=2)
vol = reject_outliers(vol, 'm-3m', sigma=3.0, n_iter=1)
vol = symmetrize_volume(vol, 'm-3m')
save_volume_h5('output.h5', vol)

# Extract a slice
sl, H, K, xl, yl, fl, fv, n = extract_volume_slice(vol, 0, target_val=0.0)

# Read a single .img file
layer = read_rsp_layer('path/to/file.img')
print(f'{layer.plane_type}, {layer.fixed_label}={layer.fixed_value}')
```

## Laue Groups

All 11 crystallographic Laue groups are supported:

| Laue Group | Crystal System | Ops |
|------------|---------------|-----|
| -1         | Triclinic     | 2   |
| 2/m        | Monoclinic    | 4   |
| mmm        | Orthorhombic  | 8   |
| 4/m        | Tetragonal    | 8   |
| 4/mmm      | Tetragonal    | 16  |
| -3         | Trigonal      | 6   |
| -3m        | Trigonal      | 12  |
| 6/m        | Hexagonal     | 12  |
| 6/mmm      | Hexagonal     | 24  |
| m-3        | Cubic         | 24  |
| m-3m       | Cubic         | 48  |

## HDF5 Output Format

MATLAB-compatible:
```matlab
data = h5read('sample_sym_mbar3m.h5', '/data');   % 3D intensity
H = h5read('sample_sym_mbar3m.h5', '/H');          % h axis
K = h5read('sample_sym_mbar3m.h5', '/K');          % k axis
L = h5read('sample_sym_mbar3m.h5', '/L');          % l axis
```

Additional datasets: `/M_inv` (2x2), `/UB` (3x3). Attributes: `cell_a/b/c/alpha/beta/gamma`, `wavelength`, `laue_group`, `s`, `cx`, `cy`.

## GPU Acceleration

GPU support via CuPy is auto-detected. Benchmarks (MAPbI2Br, 839x737x601, m-3m):

| Operation | CPU (24 cores) | GPU (GTX 1070) | Speedup |
|-----------|---------------|----------------|---------|
| Symmetrize | 198s | 30s | **6.6x** |
| Reject + Sym | 218s | 36s | **6.0x** |

Force CPU: `--no-gpu` (CLI) or `use_gpu=False` (API).

## How the hkl grid calcuation works

CrysAlisPro stores reciprocal space maps as .img files. Each file contains:

- **Pixel data**: intensity on a regular Cartesian grid (1/A)
- **UB matrix**: orientation matrix (includes wavelength factor)
- **Metadata**: d_min, wavelength, plane type, fixed value

The Miller index grid is computed from the UB matrix:

```
s = 2 / (d_min * NX)                    # Cartesian step per pixel
cx = (NX + 1) // 2 + 0.5                # pixel center

# 2D basis from raw reciprocal vectors (NOT projected)
e_x = v1 / |v1|
e_y = (v2 - (v2 . e_x) * e_x) / |...|

# 2x2 transformation: (x, y) = M @ (idx1, idx2)
M = [[v1 . e_x, v2 . e_x], [v1 . e_y, v2 . e_y]]
M_inv = inv(M)

# Miller indices at pixel (i, j):
h = M_inv[0,0] * s * (i - cx) + M_inv[0,1] * s * (j - cy)
k = M_inv[1,0] * s * (i - cx) + M_inv[1,1] * s * (j - cy)
```


For monoclinic cells (beta != 90), the cross-term `M_inv[0,1]` is significant. This means:
- Grid lines are **tilted** (not vertical/horizontal)
- Volume slicing requires **cross-term interpolation** for HL/KL planes
- The h value at each pixel depends on **both** column and row

All of this is handled automatically.

## File Reference

```
rspace3d/
    __init__.py              # Package init, public API
    rsp_reader.py            # Read single .img files, compute Miller grids
    volume_builder.py        # Core: load, bin, reject, symmetrize, save
    rsp_viewer.py    # GUI: unified viewer (.img/.cbf/.h5)
    volume_builder_gui.py     # GUI: one-button volume processor
    volume_isosurface.py     # 3D isosurface (PyVista/Plotly)
    make_dcunwarp.py         # Generate CrysAlisPro layer lists
scripts/
    volume_process.py        # CLI volume processor
notebooks/
    volume_analysis.ipynb    # Jupyter notebook for analysis
docs/
    CrysAlisPro_hkl_layer_format.md  # .img header format documentation
```

## Requirements

- Python 3.10+
- NumPy, SciPy, fabio, h5py, matplotlib, PyQt6
- Optional: CuPy (GPU), PyVista (3D), Plotly (3D browser)

## License

MIT
