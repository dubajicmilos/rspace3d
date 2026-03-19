# CrysAlisPro Reconstructed hk-Layer .img Format

## Overview

CrysAlisPro exports reciprocal space reconstructions (unwarp layers) as `.img` files
in the Oxford Diffraction OXD format. Each file contains a 2D intensity map at a fixed
Miller index l, with axes corresponding to h and k.

The `.img` header (5120 bytes) stores all parameters needed to compute h, k Miller
indices at every pixel, including for non-orthogonal (monoclinic, triclinic) unit cells.

## Header Layout

The OXD header is 5120 bytes total, divided into sections:

| Section | Byte Range | Size |
|---------|-----------|------|
| General 1 | 0-255 | 256 B |
| General 2 | 256-511 | 256 B |
| Special | 512-1279 | 768 B |
| KM4 | 1280-2303 | 1024 B |
| Statistic | 2304-2815 | 512 B |
| History | 2816-4863 | 2048 B |
| ASCII | 4864-5119 | 256 B |

## Key Header Offsets

| Offset | Type | Field | Example Value |
|--------|------|-------|---------------|
| 772 | int16 | NX (image width) | 1679 |
| 774 | int16 | NY (image height) | 1475 |
| 880 | float64 | l value (layer coordinate) | 5.96 |
| 976 | float64 | l value (repeated) | 5.96 |
| 1008 | float64 | l value (repeated) | 5.96 |
| 1024 | float64 | d_min (resolution, Angstroms) | 0.8 |
| 2104 | float64 | wavelength (Angstroms) | 0.1722 |
| 2256-2320 | 9x float64 | UB matrix (row-major, includes lambda factor) | 3x3 matrix |

The UB matrix is stored row-major as 9 consecutive float64 values at offsets 2256
through 2320 (each 8 bytes). CrysAlisPro convention: `|UB column_i| = lambda / a_i`,
so the standard reciprocal lattice vectors are `a* = UB[:,0] / lambda`.

## Filename Convention

Files are named `{experiment}_{index}.img` where the index encodes the l value:

```
l = l_min + (index - 1) * l_step
```

The l_step and l_min can be determined from any two files with known indices.
For example, with `_599.img` (l=5.96) and `_238.img` (l=-1.26):

```
l_step = (5.96 - (-1.26)) / (599 - 238) = 0.02 r.l.u.
l_min = 5.96 - (599 - 1) * 0.02 = -6.00
```

## Computing h, k at Each Pixel

### Step 1: Reciprocal Lattice Vectors

Extract from the UB matrix, removing the wavelength factor:

```
a* = UB[:,0] / lambda    (3D vector, units 1/Angstrom)
b* = UB[:,1] / lambda
c* = UB[:,2] / lambda
```

### Step 2: Project onto the hk Plane

The hk plane is perpendicular to c*:

```
c_hat = c* / |c*|
a*_proj = a* - (a* . c_hat) * c_hat
b*_proj = b* - (b* . c_hat) * c_hat
```

For orthorhombic cells, the projections equal the original vectors.
For monoclinic (beta != 90), a* gains a c* component, so |a*_proj| < |a*|.
For triclinic, both projections differ from the originals.

### Step 3: Orthogonal Pixel Basis

```
e_x = a*_proj / |a*_proj|           (along projected a*)
e_y = c_hat x e_x / |c_hat x e_x|  (perpendicular, right-hand rule)
```

### Step 4: Build the 2x2 Transformation Matrix

```
M = [[a*_proj . e_x,  b*_proj . e_x],    =  [[|a*_proj|,             |b*_proj| cos(gamma*_proj)],
     [a*_proj . e_y,  b*_proj . e_y]]         [0,                     |b*_proj| sin(gamma*_proj)]]
```

This maps Miller indices to Cartesian coordinates: `M * (h, k)^T = (x, y)^T`.
The off-diagonal element encodes the shear from non-orthogonal cells.

### Step 5: Cartesian Step Size

```
s = 2 / (d_min * NX)    (Angstrom^-1 per pixel)
```

The reconstruction grid spans [-1/d_min, +1/d_min] across NX pixels.

### Step 6: Pixel to Miller Index

For pixel (i, j) in 1-based indexing:

```
cx = (NX + 2) / 2       (grid center x, 1-based)
cy = (NY + 2) / 2       (grid center y, 1-based)

(h)         (i - cx)
( ) = M^-1 * s * (      )
(k)         (j - cy)
```

Expanded:

```
h(i,j) = M_inv[0,0] * (i - cx) * s  +  M_inv[0,1] * (j - cy) * s
k(i,j) = M_inv[1,0] * (i - cx) * s  +  M_inv[1,1] * (j - cy) * s
```

## Crystal System Behavior

### Orthorhombic (alpha = beta = gamma = 90)

M is diagonal. h and k are independent (no shear):

```
h = (i - cx) * 2a / (d_min * NX)
k = (j - cy) * 2b / (d_min * NX)
```

### Monoclinic (beta != 90)

M is still diagonal (no shear), but |a*_proj| < |a*|, so the effective cell
parameter in the hk plane is a_eff = 1/|a*_proj| > a:

```
h = (i - cx) * 2 * a_eff / (d_min * NX)
k = (j - cy) * 2b / (d_min * NX)
```

### Triclinic (all angles != 90)

M has off-diagonal terms. h depends on both i AND j (shear):

```
h(i,j) = (i - cx) * A  +  (j - cy) * B
k(i,j) = (j - cy) * D
```

where B/A = -cos(gamma*_proj) / sin(gamma*_proj).

## Intensity Data

After the 5120-byte header, intensity data is stored as TY1-compressed int16 values.
Use `fabio.OXDimage.OxdImage` to decompress:

```python
import io
from fabio.OXDimage import OxdImage

with open(filename, 'rb') as f:
    buf = io.BytesIO(f.read())
img = OxdImage()
img.read(buf)
data = img.data  # shape (NY, NX), dtype int16
```

## Comparison with Other Formats

The same data may also be exported as .cbf, .tiff, or .png:

| Format | Data type | Contains h,k info | Fidelity |
|--------|-----------|-------------------|----------|
| .img (OXD) | int16, TY1 compressed | Full header with UB, d_min, l | Raw intensities |
| .cbf | int32, byte-offset compressed | Detector geometry only | Processed/binned |
| .tiff | uint8, palette indexed, LZW | No scientific metadata | Display image only |
| .png | uint8, palette indexed | No scientific metadata | Identical to .tiff |

Only the .img format preserves the full intensity range and contains all parameters
needed to reconstruct the h, k coordinate grid.

## Validated Accuracy

Tested against CrysAlisPro interface values at 5 pixel positions (4 corners + near
origin) for a nearly-cubic perovskite cell (a ~ b ~ c ~ 6.1 A, angles ~ 90):

| Pixel | h calculated | h interface | error |
|-------|-------------|-------------|-------|
| (1, 1) | -7.6168 | -7.617 | +0.0002 |
| (1679, 1) | 7.6152 | 7.615 | +0.0002 |
| (1, 1475) | -7.6243 | -7.624 | -0.0003 |
| (1679, 1475) | 7.6077 | 7.608 | -0.0003 |
| (841, 738) | 0.0045 | 0.005 | -0.0005 |

All errors are within the 3-decimal rounding precision of the CrysAlisPro display.
