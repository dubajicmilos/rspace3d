"""
volume_builder.py — Build, bin, outlier-reject, and symmetrize
3D reciprocal space volumes from CrysAlisPro unwarp .img files.

Reads .img headers directly for grid computation (no .par file needed).

Volume geometry
---------------
Every volume is stored in physical (h, k, l) axis order, `intensity[ih, ik, il]`,
and its grid is an affine map from array index to Miller index:

    hkl = origin + A @ [ih, ik, il]          (see `volume_affine`)

* CrysAlisPro unwarp rasters (`grid_kind='unwarp_raster'`): the two in-plane
  rows of `A` carry the raster's shear (`M_inv[0,1]/M_inv[1,1]`, non-zero for
  monoclinic/triclinic cells), the third axis is the layer stack.
* rawrecon volumes (`grid_kind='hkl_regular'`): `A` is diagonal.

Symmetry operations and non-native cuts go through `A`, so a sheared raster
and a regular hkl grid are handled by the same code.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import os
import struct
import io
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable
from scipy import ndimage
from scipy.ndimage import map_coordinates

from .rsp_reader import _PLANE_CONFIG, compute_plane_M_inv, read_rsp_layer


# ──────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────

@dataclass
class VolumeData:
    """3D reciprocal space volume on a regular Miller index grid.

    Axis order is always physical (h, k, l). `plane_type` records which plane
    the source raster was native to ('HK', 'HL' or 'KL'); for that plane a cut
    is a direct array slice, the other two are interpolated (`extract_volume_slice`).
    `H`, `K`, `L` are the Miller indices along each array axis at the raster
    centre line (for a sheared unwarp raster the in-plane index also shifts
    with the other in-plane index, see `volume_affine`).

    `counts` is the number of detector pixels that contribute to each voxel.
    `bin_volume` uses it as the weight of each voxel. `None` gives each
    measured voxel the weight 1.

    `variance` is the variance (sigma^2) of each voxel value. It is float32,
    has the same shape as `intensity` and is NaN where `intensity` is NaN.
    `None` means that the variance is not known. `metadata['variance_model']`
    tells how the variance was calculated:

    * 'poisson_counts' (rawrecon): the raw detector counts are Poisson.
    * 'poisson_stored_units' (unwarp .img): the stored .img values are
      treated as Poisson counts.

    Both models use a floor of 1 count per pixel.

    An unbinned unwarp volume does not store its variance. Its variance is
    max(I, 1), and `voxel_variance` calculates it when it is necessary (see
    `variance_is_derived`).
    """
    intensity: npt.NDArray[Any]          # (nh, nk, nl) — float32; NaN = unmeasured
    H: npt.NDArray[np.floating]          # (nh,) 1D array of h values
    K: npt.NDArray[np.floating]          # (nk,) 1D array of k values
    L: npt.NDArray[np.floating]          # (nl,) 1D array of l values
    plane_type: str                      # native raster plane: 'HK', 'HL', or 'KL'
    metadata: dict[str, Any]             # wavelength, UB, cell params, etc.
    counts: npt.NDArray[Any] | None = None   # (nh, nk, nl) contributing pixels per voxel
    variance: npt.NDArray[Any] | None = None  # (nh, nk, nl) float32 sigma^2; NaN = unmeasured


def variance_is_derived(vol: VolumeData) -> bool:
    """True if the variance of `vol` is max(I, 1) and is not stored.

    This is the case for an unbinned unwarp volume: `variance` and `counts`
    are None, `metadata['variance_model']` is 'poisson_stored_units' and the
    bin factors `bin_xy` and `bin_z` are 1.
    """
    meta = vol.metadata
    return (vol.variance is None and vol.counts is None
            and meta.get('variance_model') == 'poisson_stored_units'
            and int(meta.get('bin_xy', 1)) == 1 and int(meta.get('bin_z', 1)) == 1)


def voxel_variance(vol: VolumeData, index: Any = ()) -> npt.NDArray[np.float32] | None:
    """Variance of the voxels `vol.intensity[index]`, or None if it is not known.

    The function returns `vol.variance[index]` if the variance is stored. For
    a derived variance (`variance_is_derived`) it calculates max(I, 1) as
    float32, NaN where I is NaN. This value is bit-identical to the variance
    that the unwarp loader stores for a binned volume of factor 1.
    """
    if vol.variance is not None:
        return vol.variance[index]
    if not variance_is_derived(vol):
        return None
    x = np.asarray(vol.intensity[index], dtype=np.float32)
    return np.maximum(x, np.float32(1.0))       # NaN stays NaN


# ──────────────────────────────────────────────────────────────────
# Grid geometry
# ──────────────────────────────────────────────────────────────────

def _axis_step(axis: npt.NDArray[np.floating], name: str) -> float:
    """Uniform step of a 1D Miller axis (1.0 for a single point)."""
    axis = np.asarray(axis, dtype=np.float64)
    if len(axis) < 2:
        return 1.0
    steps = np.diff(axis)
    step = float(steps.mean())
    if step == 0 or np.max(np.abs(steps - step)) > 1e-6 * max(1.0, abs(step)):
        raise ValueError(f"{name} axis is not uniformly spaced")
    return step


def raster_shear(vol: VolumeData) -> float:
    """Shear ratio of the native raster: in-plane index-1 shift per unit of
    in-plane index-2 (`M_inv[0,1] / M_inv[1,1]`); 0 for a regular hkl grid."""
    if vol.metadata.get('grid_kind', 'unwarp_raster') != 'unwarp_raster':
        return 0.0
    m_inv = vol.metadata.get('M_inv')
    if m_inv is None:
        ub = vol.metadata.get('ub')
        if ub is None:
            return 0.0
        m_inv = compute_plane_M_inv(np.asarray(ub, dtype=np.float64),
                                    float(vol.metadata.get('wavelength', 1.0)),
                                    vol.plane_type)
    m_inv = np.asarray(m_inv, dtype=np.float64)
    if abs(m_inv[1, 1]) < 1e-15:
        return 0.0
    return float(m_inv[0, 1] / m_inv[1, 1])


def volume_affine(vol: VolumeData) -> tuple[npt.NDArray[np.float64],
                                            npt.NDArray[np.float64]]:
    """Index -> Miller affine map: `hkl = origin + A @ [ih, ik, il]`.

    Built from the 1D axes plus the native raster's shear ratio, so it is exact
    for binned and unbinned unwarp rasters and for regular rawrecon grids.
    """
    dh = _axis_step(vol.H, 'H')
    dk = _axis_step(vol.K, 'K')
    dl = _axis_step(vol.L, 'L')
    A = np.diag([dh, dk, dl]).astype(np.float64)
    origin = np.array([vol.H[0], vol.K[0], vol.L[0]], dtype=np.float64)
    shear = raster_shear(vol)
    if shear != 0.0:
        # native in-plane axes (x, y) of the raster in physical index order
        x_axis, y_axis = {'HK': (0, 1), 'HL': (0, 2), 'KL': (1, 2)}[vol.plane_type]
        axes = [vol.H, vol.K, vol.L]
        y_step = A[y_axis, y_axis]
        A[x_axis, y_axis] = shear * y_step
        origin[x_axis] += shear * float(axes[y_axis][0])
    return origin, A


# ──────────────────────────────────────────────────────────────────
# Fast header reading
# ──────────────────────────────────────────────────────────────────

def _read_header_fast(path: str) -> dict[str, Any]:
    """Read minimal header fields from a CrysAlisPro .img file."""
    with open(path, 'rb') as f:
        raw = f.read(2330)

    def u16(off: int) -> int:
        return struct.unpack_from('<H', raw, off)[0]
    def f64(off: int) -> float:
        return struct.unpack_from('<d', raw, off)[0]

    nx = u16(278)
    ny = u16(280)
    h_fixed, k_fixed, l_fixed = f64(864), f64(872), f64(880)

    h_is_x = abs(f64(896) - 1.0) < 0.01
    k_is_x = abs(f64(904) - 1.0) < 0.01
    k_is_y = abs(f64(936) - 1.0) < 0.01
    l_is_y = abs(f64(944) - 1.0) < 0.01

    if h_is_x and k_is_y:
        plane_type, fixed_value = 'HK', l_fixed
    elif h_is_x and l_is_y:
        plane_type, fixed_value = 'HL', k_fixed
    elif k_is_x and l_is_y:
        plane_type, fixed_value = 'KL', h_fixed
    else:
        # Flagless header: the plane is the one whose fixed Miller value is
        # set (same rule as rsp_reader). Ambiguous headers are an error, not HK.
        detected = [(value, plane) for value, plane in
                    ((l_fixed, 'HK'), (k_fixed, 'HL'), (h_fixed, 'KL'))
                    if abs(value) > 1e-10]
        if len(detected) != 1:
            raise ValueError(
                f"Cannot determine plane type of {path}: axis flags are absent and "
                f"{'no' if not detected else 'several'} fixed Miller values are set")
        fixed_value, plane_type = detected[0]

    ub = np.array([f64(2256 + i * 8) for i in range(9)]).reshape(3, 3)

    return {
        'nx': nx, 'ny': ny, 'plane_type': plane_type,
        'fixed_value': fixed_value, 'd_min': f64(1024),
        'wavelength': f64(2104), 'ub': ub,
    }


def _read_intensity(path: str) -> npt.NDArray[np.int32]:
    """Read intensity data from .img file using fabio (buffered).
    Returns int32 array (native format)."""
    from fabio.OXDimage import OxdImage
    with open(path, 'rb') as f:
        buf = io.BytesIO(f.read())
    img = OxdImage()
    img.read(buf)
    return img.data.astype(np.int32)


# ──────────────────────────────────────────────────────────────────
# Par file reading
# ──────────────────────────────────────────────────────────────────

def find_par_file(unwarp_folder: str) -> str | None:
    """Find the .par file for an unwarp folder.

    Searches in this order:
      1. *_cracker.par in parent directory
      2. *.par in parent directory
      3. *_cracker.par inside the unwarp folder itself
      4. *.par inside the unwarp folder
      5. *_cracker.par two levels up (grandparent)
    """
    import glob
    folder = os.path.normpath(unwarp_folder)
    search_dirs = [
        os.path.dirname(folder),   # parent (most common)
        folder,                     # unwarp folder itself
        os.path.dirname(os.path.dirname(folder)),  # grandparent
    ]
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        pars = glob.glob(os.path.join(d, '*_cracker.par'))
        if pars:
            return pars[0]
        pars = glob.glob(os.path.join(d, '*.par'))
        if pars:
            return pars[0]
    return None


def read_par_cell(par_filename: str) -> dict[str, float] | None:
    """Read unit cell parameters from a CrysAlisPro .par file.

    Primary method: compute from CRYSTALLOGRAPHY UB matrix + wavelength.
    Fallback: parse CELL line directly (if present).

    Returns dict with a, b, c (Angstrom), alpha, beta, gamma (degrees).
    """
    import re

    # Read UB matrix and wavelength from par file
    ub: npt.NDArray[np.float64] | None = None
    wavelength: float | None = None

    def _strip_esd(s: str) -> str:
        return re.sub(r'\([^)]*\)', '', s)

    with open(par_filename, 'r', errors='replace') as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith('\xef\xbf\xbd') or stripped.startswith('?'):
                continue  # skip comment lines

            if 'CRYSTALLOGRAPHY UB ' in line:
                parts = line.split()
                try:
                    idx = parts.index('UB')
                    ub_vals = [float(x) for x in parts[idx + 1:idx + 10]]
                    if len(ub_vals) == 9:
                        ub = np.array(ub_vals).reshape(3, 3)
                except (ValueError, IndexError):
                    pass

            if 'CRYSTALLOGRAPHY WAVELENGTH' in line and not line.lstrip().startswith('\xef'):
                parts = line.split()
                try:
                    idx = parts.index('WAVELENGTH')
                    wavelength = float(parts[idx + 1])
                except (ValueError, IndexError):
                    pass

            # Fallback: direct CELL line
            if 'CELL ' in line and 'NUMOFCELL' not in line and 'INFORMATION' not in line:
                parts = line.split()
                try:
                    idx = parts.index('CELL')
                    cell_strs = parts[idx + 1:idx + 7]
                    if len(cell_strs) == 6:
                        nums = [float(_strip_esd(v)) for v in cell_strs]
                        # Sanity check: a > 1 Angstrom
                        if nums[0] > 1.0:
                            return {
                                'a': nums[0], 'b': nums[1], 'c': nums[2],
                                'alpha': nums[3], 'beta': nums[4], 'gamma': nums[5],
                            }
                except (ValueError, IndexError):
                    pass

    # Compute from UB + wavelength (primary method)
    if ub is not None and wavelength is not None:
        return cell_from_ub(ub, wavelength)

    return None


def cell_from_ub(ub: npt.NDArray[np.float64],
                 wavelength: float) -> dict[str, float]:
    """Compute unit cell parameters from a UB matrix.

    The UB matrix from .img headers includes the lambda factor:
    UB = lambda * [a* b* c*] (reciprocal vectors as columns).

    Returns dict with a, b, c (Angstrom), alpha, beta, gamma (degrees).
    """
    recip = ub / wavelength
    real = np.linalg.inv(recip).T  # real-space vectors as columns
    a = np.linalg.norm(real[:, 0])
    b = np.linalg.norm(real[:, 1])
    c = np.linalg.norm(real[:, 2])
    cos_alpha = np.dot(real[:, 1], real[:, 2]) / (b * c)
    cos_beta  = np.dot(real[:, 0], real[:, 2]) / (a * c)
    cos_gamma = np.dot(real[:, 0], real[:, 1]) / (a * b)
    return {
        'a': a, 'b': b, 'c': c,
        'alpha': np.degrees(np.arccos(np.clip(cos_alpha, -1, 1))),
        'beta':  np.degrees(np.arccos(np.clip(cos_beta, -1, 1))),
        'gamma': np.degrees(np.arccos(np.clip(cos_gamma, -1, 1))),
    }


def compute_1d_axes(
    header: dict[str, Any],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute 1D Miller index axes from .img header.

    Uses raw reciprocal vectors (no projection) matching CrysAlisPro convention.
    Returns h values along center row and k values along center column.
    """
    nx, ny = header['nx'], header['ny']
    s = 2.0 / (header['d_min'] * nx)
    cx = (nx + 1) // 2 + 0.5
    cy = (ny + 1) // 2 + 0.5

    M_inv = compute_plane_M_inv(header['ub'], header['wavelength'],
                                 header['plane_type'])

    ii = np.arange(1, nx + 1) - cx
    jj = np.arange(1, ny + 1) - cy
    return M_inv[0, 0] * ii * s, M_inv[1, 1] * jj * s


# ──────────────────────────────────────────────────────────────────
# Binning utilities
# ──────────────────────────────────────────────────────────────────

def bin_2d_covered(raw: npt.NDArray[Any], mask: npt.NDArray[np.bool_],
                   by: int, bx: int, start_y: int = 0, start_x: int = 0,
                   with_variance: bool = False,
                   ) -> Any:
    """Bin a raw frame by the mean over its *covered* sub-pixels only.

    `mask` is the coverage mask (True = measured). Unmeasured sub-pixels
    (detector `-1`, dead zeros) do not enter the mean, so a partially covered
    cell keeps the mean of its real measurements instead of being pulled
    towards -1/0 (`[[100, -1], [-1, -1]]` -> 100, not 24.25). Cells without
    any covered sub-pixel are NaN. `start_y`/`start_x` leading pixels are
    skipped so the blocks can be aligned to the raster centre (`bin_starts`).

    With `with_variance=True` the function returns `(mean, variance, n_cov)`.
    Each covered pixel value I is a Poisson count with variance max(I, 1),
    so the variance of the mean is sum(max(I, 1)) / n_cov^2 (float32, NaN
    where n_cov = 0). `n_cov` (int32) is the number of covered sub-pixels.
    """
    ny, nx = raw.shape[0] - start_y, raw.shape[1] - start_x
    ny_t = (ny // by) * by
    nx_t = (nx // bx) * bx
    blocks = (ny_t // by, by, nx_t // bx, bx)
    m = mask[start_y:start_y + ny_t, start_x:start_x + nx_t].reshape(blocks)
    values = raw[start_y:start_y + ny_t, start_x:start_x + nx_t].astype(np.int64).reshape(blocks)
    summed = np.where(m, values, 0).sum(axis=(1, 3)).astype(np.float64)
    n_cov = m.sum(axis=(1, 3))
    with np.errstate(invalid='ignore', divide='ignore'):
        out = np.where(n_cov > 0, summed / np.maximum(n_cov, 1), np.nan)
    if not with_variance:
        return out.astype(np.float32)
    ssq = np.where(m, np.maximum(values, 1), 0).sum(axis=(1, 3)).astype(np.float64)
    n = np.maximum(n_cov, 1).astype(np.float64)
    var = np.where(n_cov > 0, ssq / (n * n), np.nan)
    return out.astype(np.float32), var.astype(np.float32), n_cov.astype(np.int32)


MORPH_MAX_KERNEL = 101   # px; ~11x the largest real 'auto' kernel (I19-2: 9 px)


def adaptive_morph_size(s_per_pixel: float, recip_period: float,
                        fraction_of_bragg: float = 0.05,
                        min_morph: int = 3) -> int:
    """Pick a morph kernel that covers `fraction_of_bragg` of one Bragg cell.

    Parameters
    ----------
    s_per_pixel : float
        Cartesian step per detector pixel in 1/A. For an unwarped frame this
        is `2 / (d_min * NX)`.
    recip_period : float
        Length of one reciprocal-lattice period along the in-plane axes,
        in 1/A. For an HK plane that is `mean(|a*|, |b*|)`.
    fraction_of_bragg : float, default 0.05
        Target kernel side length, as a fraction of one Bragg cell. 0.05
        was calibrated on perovskite I19-2 unwarp frames where convergence
        of `n_unmeasured` happened at 0.04-0.05.
    min_morph : int, default 3
        Floor on the returned kernel size.

    Returns
    -------
    int
        Odd integer, >= min_morph. The kernel side length in pixels.
    """
    for name, value in (('s_per_pixel', s_per_pixel), ('recip_period', recip_period),
                        ('fraction_of_bragg', fraction_of_bragg)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive, got {value!r}")
    n = max(min_morph, int(np.ceil(fraction_of_bragg * recip_period / s_per_pixel)))
    if n > MORPH_MAX_KERNEL:
        warnings.warn(
            f"adaptive morph kernel {n} px exceeds the {MORPH_MAX_KERNEL} px ceiling "
            f"(s_per_pixel={s_per_pixel:.3g}, recip_period={recip_period:.3g}); "
            "clamped. Check d_min / UB in the .img header.", RuntimeWarning, stacklevel=2)
        n = MORPH_MAX_KERNEL
    return n if (n % 2) else n + 1   # force odd for symmetric structuring element


def build_coverage_mask(raw: npt.NDArray[np.integer],
                        morph_size: int = 3) -> npt.NDArray[np.bool_]:
    """Return bool mask True=measured / False=unmeasured for a raw .img frame.

    CrysAlisPro .img stores unmeasured pixels as either
      - `-1`   : detector-flagged bad pixel
      - `0`    : large dead region (beamstop, outer mask, edge), but `0` also
                 occurs as a genuine low-count measurement

    A morphological opening with an NxN square structuring element keeps
    'solid' dead blocks while erasing isolated zeros and thin 1-pixel lines
    (treated as real measurements).

    For `morph_size`, see `adaptive_morph_size` to derive a value from the
    detector's Cartesian step and the in-plane reciprocal-lattice period
    (recommended path is `load_unwarp_folder(morph_size='auto')` which
    computes it once from the .img header).
    """
    bad = (raw == -1)
    zero = (raw == 0)
    struct = np.ones((morph_size, morph_size), dtype=bool)
    unmeasured_zero = ndimage.binary_opening(zero, structure=struct)
    unmeasured = unmeasured_zero | bad
    return ~unmeasured


def bin_1d(arr: npt.NDArray[np.floating], b: int, start: int = 0) -> npt.NDArray[np.floating]:
    """Bin a 1D array by averaging groups of b elements (skipping `start` leading ones)."""
    n = ((len(arr) - start) // b) * b
    return arr[start:start + n].reshape(-1, b).mean(axis=1)


def _binned_center(c: float, b: int, start: int = 0) -> float:
    """1-based raster centre after binning by `b` (blocks start at pixel 1 + start)."""
    return (c - 1.0 - start) / b + 1.0 - (b - 1.0) / (2.0 * b)


def _axis_center(axis: npt.NDArray[np.floating]) -> float:
    """1-based position of Miller index 0 along a uniform axis."""
    return 1.0 - float(axis[0]) / _axis_step(axis, 'axis')


def _bin_start(c: float, b: int) -> int | None:
    """Leading pixels to skip so the centre `c` (1-based) sits on a bin
    centre after binning by `b`, or else on a bin edge; None when no integer
    skip achieves either.

    Bin edges sit at pixel coordinate 0.5 + skip + b*j and bin centres half
    a bin further. A bin centre keeps the operations of all Laue groups
    exact and keeps index 0 as a layer of the binned grid. A bin edge keeps
    only the signed-permutation operations exact.
    """
    # Bin centre first, then bin edge. Round to the nearest integer skip
    # (no float modulo, because round-off near a multiple of b/2 changes
    # the result of a float modulo by b/2).
    for x in (c - 0.5 - b / 2.0, c - 0.5):
        k = round(x)
        if abs(x - k) <= 1e-9:
            return int(k) % b
    return None                           # only a fractional skip is possible


def _bin_kind(c: float, b: int, start: int) -> int:
    """0 = centre on a bin edge, 1 = centre on a bin centre (after `start` skip)."""
    return int(round((c - 0.5 - start) / (b / 2.0))) % 2


def bin_starts(axes: list[npt.NDArray[np.floating]], factors: list[int],
               couple: tuple[int, int] | None = None) -> list[int]:
    """Block alignment for binning: leading samples to skip per axis.

    Chosen so that each axis keeps Miller index 0 on a bin centre where
    possible, otherwise on a bin edge (inversion stays an exact index map),
    and, for the two `couple`d axes, so
    that both are of the same kind (90 deg rotations and the hexagonal
    operations, which mix the two in-plane axes, stay exact). Costs at most
    `b` samples at the raster edge. Falls back to 0 where no alignment exists
    (e.g. an odd layer count binned by 2), in which case `symmetrize_volume`
    interpolates the affected operations.
    """
    starts = []
    for axis, b in zip(axes, factors):
        if b == 1 or len(axis) < 2 * b:
            starts.append(0)
            continue
        s = _bin_start(_axis_center(axis), b)
        starts.append(0 if s is None else s)
    if couple is not None:
        i, j = couple
        bi, bj = factors[i], factors[j]
        if bi == bj and bi > 1 and bi % 2 == 0 and len(axes[j]) >= 2 * bj + bj // 2:
            ci, cj = _axis_center(axes[i]), _axis_center(axes[j])
            if (_bin_start(ci, bi) is not None and _bin_start(cj, bj) is not None
                    and _bin_kind(ci, bi, starts[i]) != _bin_kind(cj, bj, starts[j])):
                starts[j] += bj // 2
    return starts


_BIN_SLAB_VOXELS = 1 << 25    # input voxels per slab of bin_volume


def bin_volume(vol: VolumeData, bh: int, bk: int, bl: int) -> VolumeData:
    """Bin the 3D volume by averaging (NaN-aware).

    NaN voxels (unmeasured) are excluded from the bin mean. A binned cell
    becomes NaN only if every sub-cell was NaN.

    Each finite sub-voxel has the weight w = `counts` (1 if `counts` is
    None). The binned value is sum(w x) / sum(w) over the finite sub-voxels.
    For a rawrecon volume this is the value of a reconstruction with the
    larger voxel. Without `counts` the value is the plain `nanmean`, as in
    rspace3d 2.2. The binned `counts` are sum(w) over the finite sub-voxels.
    The binned `variance` is sum(w^2 var) / sum(w)^2 over the finite
    sub-voxels (float32, NaN where no sub-voxel is finite). It is None if
    the input has no variance.

    If the input has no `counts`, the binned `counts` are uint16 (uint32 if
    bh x bk x bl is more than 65535). If the input has `counts`, they are
    int64. A derived input variance (`variance_is_derived`) is calculated
    for one slab at a time. The binned variance is always stored.

    The function works on slabs of output h-layers, so that the temporary
    arrays are much smaller than the volume.

    Blocks are aligned to the grid centre (`bin_starts`)
    so that symmetry operations remain exact index maps on the binned grid.
    An odd layer count binned by an even factor has no such alignment (see
    `bin_starts`). The (fewer than 2b per axis) edge layers that the
    alignment costs are recorded in
    metadata as `bin_dropped_layers_hkl` / `bin_starts_hkl`.
    """
    x_axis, y_axis = {'HK': (0, 1), 'HL': (0, 2), 'KL': (1, 2)}[vol.plane_type]
    factors = [bh, bk, bl]
    bx, by = factors[x_axis], factors[y_axis]
    if bx != by:
        raise ValueError(f"in-plane bin factors must be equal, got {bx} x {by}")
    nh, nk, nl = vol.intensity.shape
    starts = bin_starts([vol.H, vol.K, vol.L], factors, couple=(x_axis, y_axis))
    sh, sk, sl = starts
    nh_t = ((nh - sh) // bh) * bh
    nk_t = ((nk - sk) // bk) * bk
    nl_t = ((nl - sl) // bl) * bl
    dropped = [nh - nh_t, nk - nk_t, nl - nl_t]     # alignment + remainder, < 2b each
    box = (slice(sh, sh + nh_t), slice(sk, sk + nk_t), slice(sl, sl + nl_t))
    out_shape = (nh_t // bh, nk_t // bk, nl_t // bl)
    has_variance = vol.variance is not None or variance_is_derived(vol)
    if vol.counts is not None:
        count_dtype: Any = np.int64
    else:
        count_dtype = np.uint16 if bh * bk * bl <= 0xFFFF else np.uint32
    binned = np.empty(out_shape, np.float32)
    counts = np.empty(out_shape, count_dtype)
    variance = np.empty(out_shape, np.float32) if has_variance else None

    # One slab of output h-layers at a time. Each output voxel uses only its
    # own block, so the result does not depend on the slab size.
    rows = max(1, _BIN_SLAB_VOXELS // max(1, bh * nk_t * nl_t))
    for i0 in range(0, out_shape[0], rows):
        i1 = min(out_shape[0], i0 + rows)
        sub = (slice(sh + i0 * bh, sh + i1 * bh), box[1], box[2])
        blocks = (i1 - i0, bh, out_shape[1], bk, out_shape[2], bl)
        trimmed = vol.intensity[sub].astype(np.float32).reshape(blocks)
        finite = np.isfinite(trimmed)
        weight = None                              # None: weight 1 per finite sub-voxel
        if vol.counts is None:
            with np.errstate(invalid='ignore'), warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)   # all-NaN blocks are fine
                binned[i0:i1] = np.nanmean(trimmed, axis=(1, 3, 5))
            cnt = finite.sum(axis=(1, 3, 5), dtype=np.int64)
        else:
            weight = np.where(finite, vol.counts[sub].astype(np.int64).reshape(blocks), 0)
            cnt = weight.sum(axis=(1, 3, 5))
        counts[i0:i1] = cnt
        has_weight = cnt > 0
        denom = np.maximum(cnt, 1).astype(np.float64)
        if weight is not None:
            wx = np.where(finite, trimmed.astype(np.float64) * weight, 0.0)
            binned[i0:i1] = np.where(has_weight, wx.sum(axis=(1, 3, 5)) / denom, np.nan)
            del wx
        if variance is not None:
            var = voxel_variance(vol, sub).reshape(blocks)
            if weight is None:
                w2v = np.where(finite, var, np.float32(0.0))
            else:
                w2v = np.where(finite, var.astype(np.float64) * (weight * weight), 0.0)
            variance[i0:i1] = np.where(has_weight, w2v.sum(axis=(1, 3, 5), dtype=np.float64)
                                       / (denom * denom), np.nan)
            del w2v, var
        del trimmed, finite, weight

    # Update metadata to reflect the binned raster. The in-plane bin factor
    # of the native raster scales its Cartesian step `s`; the raster centre
    # follows the block layout (blocks start at pixel 1).
    meta = vol.metadata.copy()
    # These values describe the symmetrisation of the unbinned grid
    for key in ('symmetry_diagnostics', 'n_outliers_removed', 'symmetry_max_index_error',
                'noise_model'):
        meta.pop(key, None)
    if 's' in meta:
        meta['s'] = meta['s'] * bx
    if 'cx' in meta:
        meta['cx'] = _binned_center(meta['cx'], bx, starts[x_axis])
    if 'cy' in meta:
        meta['cy'] = _binned_center(meta['cy'], by, starts[y_axis])
    meta['bin_xy'] = meta.get('bin_xy', 1) * bx
    meta['bin_z'] = meta.get('bin_z', 1) * factors[3 - x_axis - y_axis]
    if any(dropped):
        meta['bin_dropped_layers_hkl'] = dropped
        meta['bin_starts_hkl'] = list(starts)

    return VolumeData(
        intensity=binned,
        H=bin_1d(vol.H, bh, sh),
        K=bin_1d(vol.K, bk, sk),
        L=bin_1d(vol.L, bl, sl),
        plane_type=vol.plane_type,
        metadata=meta,
        counts=counts,
        variance=variance,
    )


def bin_native(vol: VolumeData, bin_xy: int, bin_z: int) -> VolumeData:
    """Bin in-plane by `bin_xy` and along the layer stack by `bin_z`.

    Maps the raster-relative factors onto the physical (h, k, l) axes of the
    volume's native plane (the layer axis is l for HK, k for HL, h for KL).
    """
    factors = [bin_xy, bin_xy, bin_xy]
    factors[{'HK': 2, 'HL': 1, 'KL': 0}[vol.plane_type]] = bin_z
    return bin_volume(vol, *factors)


# ──────────────────────────────────────────────────────────────────
# Volume loading
# ──────────────────────────────────────────────────────────────────

def unwarp_source_manifest(folder: str) -> str:
    """Fingerprint of the unwarp inputs: numbered .img names, sizes and
    modification times plus the same for the resolved .par file.

    Stored in the `_raw.h5` cache as `source_manifest`; a cache is reused only
    when the fingerprint of the folder still matches (`raw_cache_mismatch`),
    so added, removed or re-written layers are never silently ignored.
    """
    import hashlib
    h = hashlib.sha1()
    for fname in _filter_numbered_imgs(folder):
        st = os.stat(os.path.join(folder, fname))
        h.update(f"{fname}|{st.st_size}|{st.st_mtime_ns}\n".encode('utf-8'))
    par_path = find_par_file(folder)
    if par_path:
        st = os.stat(par_path)
        h.update(f"par:{os.path.abspath(par_path)}|{st.st_size}|{st.st_mtime_ns}\n"
                 .encode('utf-8'))
    return h.hexdigest()


def raw_cache_mismatch(cached: VolumeData, folder: str, morph_size: int) -> str | None:
    """Why a cached `_raw.h5` cannot stand in for re-reading `folder` (None = reusable)."""
    if cached.metadata.get('morph_size') != morph_size:
        return (f"mask kernel morph={cached.metadata.get('morph_size')} does not match "
                f"the requested morph={morph_size}")
    stamp = cached.metadata.get('source_manifest')
    if not stamp:
        return "the cache predates source fingerprinting (rspace3d < 2.1)"
    # An unbinned cache does not need a stored variance (`variance_is_derived`).
    # A cache with per-frame binning needs its variance and its pixel counts.
    if cached.variance is None and not variance_is_derived(cached):
        return "the cache has no per-voxel variance (rspace3d <= 2.2)"
    if int(cached.metadata.get('bin_xy', 1)) > 1 and cached.counts is None:
        return "the binned cache has no pixel counts (rspace3d <= 2.2)"
    if stamp != unwarp_source_manifest(folder):
        return "the .img files or the .par file changed since the cache was written"
    return None


def _filter_numbered_imgs(folder: str) -> list[str]:
    """Return only numbered .img files sharing the most common prefix.

    Filters to {prefix}_{number}.img pattern, then groups by prefix and
    keeps only the largest group. This excludes stray .img files from
    other runs or tests that happen to be in the same folder.
    """
    import re
    from collections import Counter

    all_numbered = []
    for fname in os.listdir(folder):
        if re.search(r'_\d+\.img$', fname, re.IGNORECASE):
            all_numbered.append(fname)

    if not all_numbered:
        return []

    # Group by prefix (everything before the last _<number>.img)
    prefixes = [f.rsplit('_', 1)[0] for f in all_numbered]
    prefix_counts = Counter(prefixes)
    main_prefix = prefix_counts.most_common(1)[0][0]

    return sorted((f for f in all_numbered if f.rsplit('_', 1)[0] == main_prefix),
                  key=_img_number)


def _img_number(fname: str) -> int:
    """Parse trailing number from an .img filename for numeric sorting.

    e.g. "MAPbI2Br_293K_1_42.img" -> 42. Returns 0 on parse failure.
    """
    try:
        return int(fname.rsplit('_', 1)[1].split('.')[0])
    except (ValueError, IndexError):
        return 0


def resolve_unit_cell(
    folder: str,
    header: dict[str, Any],
) -> tuple[dict[str, float], str | None]:
    """Resolve unit cell from par file (primary) or .img header UB (fallback).

    Parameters
    ----------
    folder : str
        Unwarp folder to search for a .par file.
    header : dict
        Header dict from `_read_header_fast` (needs 'ub' and 'wavelength').

    Returns
    -------
    (cell, par_path)
        `cell` has keys a, b, c, alpha, beta, gamma. `par_path` is the path
        to the .par file that was used (for logging), or None if the cell
        was computed from the .img header UB.
    """
    par_path = find_par_file(folder)
    cell: dict[str, float] | None = None
    if par_path:
        cell = read_par_cell(par_path)
    if cell is None:
        cell = cell_from_ub(header['ub'], header['wavelength'])
        return cell, None
    return cell, par_path


def scan_unwarp_folder(folder: str) -> list[tuple[str, float]]:
    """Scan folder for numbered .img files, return [(path, fixed_value)] sorted.

    Only loads files matching the pattern {prefix}_{number}.img with the
    most common prefix. Ignores files from other runs or tests.
    """
    img_files = _filter_numbered_imgs(folder)
    files = []
    plane_types = set()
    for fname in img_files:
        path = os.path.join(folder, fname)
        hdr = _read_header_fast(path)
        files.append((path, hdr['fixed_value']))
        plane_types.add(hdr['plane_type'])
    if len(plane_types) > 1:
        raise ValueError(f"Mixed plane types in {folder}: {sorted(plane_types)}")
    files.sort(key=lambda x: x[1])
    # The layer stack must be a uniform axis: duplicates or a missing layer
    # would silently corrupt every index map through that axis.
    fixed = np.array([fv for _, fv in files], dtype=np.float64)
    if len(fixed) > 1:
        steps = np.diff(fixed)
        if np.any(steps <= 0):
            dup = [os.path.basename(files[i + 1][0]) for i in np.nonzero(steps <= 0)[0]]
            raise ValueError(
                f"Duplicate layer coordinates in {folder} (e.g. {dup[0]}): "
                "remove the stray .img files")
        step = float(np.median(steps))
        bad = np.nonzero(np.abs(steps - step) > 1e-3 * step)[0]
        if len(bad):
            i = int(bad[0])
            raise ValueError(
                f"Layer spacing is not uniform in {folder}: {fixed[i]:g} -> "
                f"{fixed[i + 1]:g} (expected step {step:g}); a layer may be missing")
    return files


def load_unwarp_folder(folder: str, bin_xy: int = 1, bin_z: int = 1,
                       morph_size: int | str = 'auto',
                       progress_callback: Callable[[int, int], None] | None = None,
                       max_workers: int = 0) -> VolumeData:
    """Load all .img files from unwarp folder into a 3D volume.

    Unmeasured voxels (detector-bad `-1` and solid-zero dead regions) are
    flagged using a per-frame coverage mask (`build_coverage_mask`) and
    encoded as `NaN` in the float32 intensity volume. `symmetrize_volume`
    skips `NaN` voxels, so downstream statistics never mix real zeros with
    unmeasured pixels.

    The stored .img values are treated as Poisson counts in the stored units
    (`metadata['variance_model'] = 'poisson_stored_units'`). The variance of
    a covered pixel is max(I, 1). With `bin_xy > 1` a binned pixel has the
    variance sum(max(I, 1)) / n_cov^2 over its n_cov covered sub-pixels, and
    `counts` holds n_cov (uint16, uint32 if bin_xy^2 is more than 65535), so
    that `bin_volume` can weight each voxel
    correctly. With `bin_xy = 1`, `counts` is None (weight 1 per pixel) and
    the variance is not stored (`variance` is None), because it is max(I, 1).
    `voxel_variance` calculates it when it is necessary. This keeps the
    memory of the volume at one float32 array.

    Parameters
    ----------
    morph_size : int or 'auto', default 'auto'
        Side length of the morphological-opening kernel that decides which
        zero pixels are real measurements vs unmeasured. `'auto'` derives
        it once from the .img header so the kernel covers ~5% of one
        in-plane Bragg cell — adapts to detector distance, pixel count
        and unit-cell size. Pass an integer to override.

    Per-file reads (fabio decode + mask + bin) run on a ThreadPoolExecutor;
    all steps release the GIL, so threading scales well on typical disks.
    `max_workers=0` auto-picks `min(8, cpu_count())`; pass `max_workers=1` to
    disable threading (e.g. on RAM-constrained machines — each worker holds
    one raw frame transiently).
    """
    file_list = scan_unwarp_folder(folder)
    if not file_list:
        raise ValueError(f"No .img files found in {folder}")

    ref_header = _read_header_fast(file_list[0][0])
    nx, ny = ref_header['nx'], ref_header['ny']
    plane_type = ref_header['plane_type']
    axis_x_full, axis_y_full = compute_1d_axes(ref_header)

    # Resolve morph_size='auto' once from header geometry. Track both the
    # mode (auto/manual) and the geometry inputs so they land in metadata.
    morph_recip_period: float | None = None
    morph_s_per_pixel: float | None = None
    if isinstance(morph_size, str):
        if morph_size != 'auto':
            raise ValueError(f"morph_size must be 'auto' or int, got {morph_size!r}")
        morph_mode = 'auto'
        # Cartesian step per pixel (1/A)
        morph_s_per_pixel = 2.0 / (ref_header['d_min'] * nx)
        # In-plane reciprocal-vector columns by plane type
        plane_to_cols = {'HK': (0, 1), 'HL': (0, 2), 'KL': (1, 2)}
        c1, c2 = plane_to_cols[plane_type]
        ub = np.asarray(ref_header['ub']).reshape(3, 3)
        wl = ref_header['wavelength']
        period_1 = float(np.linalg.norm(ub[:, c1] / wl))
        period_2 = float(np.linalg.norm(ub[:, c2] / wl))
        morph_recip_period = 0.5 * (period_1 + period_2)
        morph_size = adaptive_morph_size(morph_s_per_pixel, morph_recip_period)
    elif not isinstance(morph_size, (int, np.integer)) or morph_size < 1:
        raise ValueError(f"morph_size must be 'auto' or positive int, got {morph_size!r}")
    else:
        morph_mode = 'manual'
        morph_size = int(morph_size)

    if bin_xy > 1:
        # blocks aligned to the raster centre so symmetry maps stay exact
        start_x, start_y = bin_starts([axis_x_full, axis_y_full], [bin_xy, bin_xy],
                                      couple=(0, 1))
        axis_x = bin_1d(axis_x_full, bin_xy, start_x)
        axis_y = bin_1d(axis_y_full, bin_xy, start_y)
        nx_bin, ny_bin = len(axis_x), len(axis_y)
    else:
        start_x = start_y = 0
        axis_x, axis_y = axis_x_full, axis_y_full
        nx_bin, ny_bin = nx, ny

    n_files = len(file_list)

    # float32 storage in physical (h, k, l) order. NaN = unmeasured, finite =
    # measured (including real 0). The raster's x/y axes and the layer stack
    # land on the physical axes of the native plane.
    layer_values = np.zeros(n_files, dtype=np.float64)
    # HK: (x, y, layer) = (h, k, l); HL: (x, layer, y); KL: (layer, x, y)
    layer_axis = {'HK': 2, 'HL': 1, 'KL': 0}[plane_type]
    shape = [nx_bin, ny_bin]
    shape.insert(layer_axis, n_files)
    volume = np.full(tuple(shape), np.nan, dtype=np.float32)
    # bin_xy = 1: the variance is max(I, 1) and is not stored (`voxel_variance`)
    variance = np.full(tuple(shape), np.nan, dtype=np.float32) if bin_xy > 1 else None
    count_dtype = np.uint16 if bin_xy * bin_xy <= 0xFFFF else np.uint32
    counts = np.zeros(tuple(shape), dtype=count_dtype) if bin_xy > 1 else None

    def put(target: npt.NDArray[Any], i: int, frame: npt.NDArray[Any]) -> None:
        index: list[Any] = [slice(None), slice(None)]
        index.insert(layer_axis, i)
        target[tuple(index)] = frame

    workers = max_workers if max_workers > 0 else min(8, os.cpu_count() or 1)

    def _load_frame(args):
        idx, path, fixed_val = args
        raw = _read_intensity(path)                           # int32 (ny, nx)
        mask_raw = build_coverage_mask(raw, morph_size=morph_size)  # bool
        n_cov = var_T = None
        if bin_xy > 1:
            # mean over the covered sub-pixels only; NaN where none is covered
            binned, var, n_cov = bin_2d_covered(raw, mask_raw, bin_xy, bin_xy,
                                                start_y, start_x, with_variance=True)
            n_cov = np.ascontiguousarray(n_cov.T)
            var_T = np.ascontiguousarray(var.T)
        else:
            binned = np.where(mask_raw, raw, np.nan).astype(np.float32)
        data_T = np.ascontiguousarray(binned.T)               # (x, y)
        n_unmeasured = int(np.count_nonzero(np.isnan(data_T)))
        return idx, fixed_val, data_T, var_T, n_cov, n_unmeasured

    n_done = 0
    n_unmeasured_per_frame = np.zeros(n_files, dtype=np.int64)
    with ThreadPoolExecutor(max_workers=workers) as exe:
        futs = [exe.submit(_load_frame, (i, path, fv))
                for i, (path, fv) in enumerate(file_list)]
        for fut in as_completed(futs):
            idx, fixed_val, data_T, var_T, n_cov, n_unmeasured = fut.result()
            put(volume, idx, data_T)
            if counts is not None:
                put(variance, idx, var_T)
                put(counts, idx, n_cov)
            layer_values[idx] = fixed_val
            n_unmeasured_per_frame[idx] = n_unmeasured
            n_done += 1
            if progress_callback:
                progress_callback(n_done, n_files)

    # Compute full M_inv and cell from first file header
    ref_layer = read_rsp_layer(file_list[0][0])
    cell = cell_from_ub(ref_header['ub'], ref_header['wavelength'])

    par_path = find_par_file(folder)
    if par_path:
        par_cell = read_par_cell(par_path)
        if par_cell:
            cell = par_cell

    # Coverage-mask diagnostics. The raster geometry (s, cx, cy) describes
    # the volume as stored, i.e. after in-plane binning; bin_z is applied
    # once, below, through bin_volume.
    frame_npix_post_bin = nx_bin * ny_bin
    metadata = {
        'wavelength': ref_header['wavelength'],
        'ub': ref_header['ub'],
        'd_min': ref_header['d_min'],
        'source_folder': folder,
        'source_manifest': unwarp_source_manifest(folder),
        'grid_kind': 'unwarp_raster',
        'bin_xy': bin_xy, 'bin_z': 1,
        'n_files': n_files,
        'cell': cell,
        'M_inv': ref_layer.M_inv,
        's': ref_layer.s * bin_xy,
        'cx': _binned_center(ref_layer.cx, bin_xy, start_x),
        'cy': _binned_center(ref_layer.cy, bin_xy, start_y),
        'morph_size': morph_size,
        'morph_mode': morph_mode,
        'morph_s_per_pixel': morph_s_per_pixel,
        'morph_recip_period': morph_recip_period,
        'n_unmeasured_per_frame_mean': float(n_unmeasured_per_frame.mean()),
        'n_unmeasured_per_frame_min': int(n_unmeasured_per_frame.min()),
        'n_unmeasured_per_frame_max': int(n_unmeasured_per_frame.max()),
        'n_unmeasured_per_frame_pct': float(
            100.0 * n_unmeasured_per_frame.mean() / frame_npix_post_bin),
        'frame_npix': int(frame_npix_post_bin),
        'variance_model': 'poisson_stored_units',
    }

    axes = {'HK': (axis_x, axis_y, layer_values),
            'HL': (axis_x, layer_values, axis_y),
            'KL': (layer_values, axis_x, axis_y)}[plane_type]
    vol = VolumeData(
        intensity=volume, H=axes[0], K=axes[1], L=axes[2],
        plane_type=plane_type, metadata=metadata,
        counts=counts, variance=variance,
    )

    if bin_z > 1:
        factors = [1, 1, 1]
        factors[layer_axis] = bin_z
        vol = bin_volume(vol, *factors)
    return vol


# ──────────────────────────────────────────────────────────────────
# Symmetry operations — all 11 Laue groups
# ──────────────────────────────────────────────────────────────────

def _generate_group(
    generators: list[npt.NDArray[np.int_]],
    max_iter: int = 200,
) -> list[npt.NDArray[np.int_]]:
    """Generate full point group from generator matrices."""
    ops = {tuple(np.eye(3, dtype=int).flatten())}
    queue = [g.astype(int) for g in generators]
    for _ in range(max_iter):
        new_ops = set()
        for g in queue:
            for o_flat in list(ops):
                o = np.array(o_flat, dtype=int).reshape(3, 3)
                for product in [g @ o, o @ g]:
                    key = tuple(product.flatten())
                    if key not in ops:
                        new_ops.add(key)
        if not new_ops:
            break
        ops.update(new_ops)
        queue = [np.array(k, dtype=int).reshape(3, 3) for k in new_ops]
    return [np.array(o, dtype=int).reshape(3, 3) for o in ops]


# Generators act on Miller indices, h' = W h. For the hexagonal setting
# (a = b, gamma = 120 deg) the Miller-index operators are the transposed
# inverses of the familiar direct-space matrices: the 3-fold about c is
# (h, k, l) -> (-h-k, h, l), the 6-fold (h, k, l) -> (-k, h+k, l) and the
# 2-fold along a is (h, k, l) -> (h-k... ) see below. rspace3d <= 2.0.0 used
# the direct-space matrices, which do not preserve |q| on a hexagonal cell.
_INV = -np.eye(3, dtype=int)
_C2a = np.diag([1, -1, -1]).astype(int)
_C2b = np.diag([-1, 1, -1]).astype(int)
_C4c = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=int)
_C3_111 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=int)
_C3_hex = np.array([[-1, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=int)    # (h,k,l)->(-h-k, h, l)
_C6_hex = np.array([[0, -1, 0], [1, 1, 0], [0, 0, 1]], dtype=int)     # (h,k,l)->(-k, h+k, l)
_C2p_hex = np.array([[1, 0, 0], [-1, -1, 0], [0, 0, -1]], dtype=int)  # 2-fold along a

_EXPECTED_ORDERS = {
    '-1': 2, '2/m': 4, 'mmm': 8, '4/m': 8, '4/mmm': 16,
    '-3': 6, '-3m': 12, '6/m': 12, '6/mmm': 24,
    'm-3': 24, 'm-3m': 48,
}

_LAUE_GENERATORS = {
    '-1':     [_INV],
    '2/m':    [_C2b, _INV],
    'mmm':    [_C2a, _C2b, _INV],
    '4/m':    [_C4c, _INV],
    '4/mmm':  [_C4c, _C2a, _INV],
    '-3':     [_C3_hex, _INV],
    '-3m':    [_C3_hex, _C2p_hex, _INV],
    '6/m':    [_C6_hex, _INV],
    '6/mmm':  [_C6_hex, _C2p_hex, _INV],
    'm-3':    [_C3_111, _C2b, _INV],
    'm-3m':   [_C3_111, _C4c, _INV],
}

_LAUE_GROUPS_CACHE: dict[str, list[npt.NDArray[np.int_]]] = {}


def get_symmetry_operations(laue_group: str) -> list[npt.NDArray[np.int_]]:
    """Return list of 3x3 symmetry operation matrices for a Laue group."""
    if laue_group not in _LAUE_GROUPS_CACHE:
        if laue_group not in _LAUE_GENERATORS:
            raise ValueError(f"Unknown Laue group '{laue_group}'. "
                             f"Valid: {list(_LAUE_GENERATORS.keys())}")
        ops = _generate_group(_LAUE_GENERATORS[laue_group])
        if len(ops) != _EXPECTED_ORDERS[laue_group]:
            raise RuntimeError(f"Generated {len(ops)} operations for {laue_group}, "
                               f"expected {_EXPECTED_ORDERS[laue_group]}")
        _LAUE_GROUPS_CACHE[laue_group] = ops
    return _LAUE_GROUPS_CACHE[laue_group]


LAUE_GROUP_NAMES = {
    '-1':     'Triclinic  (-1)',
    '2/m':    'Monoclinic (2/m)',
    'mmm':    'Orthorhombic (mmm)',
    '4/m':    'Tetragonal (4/m)',
    '4/mmm':  'Tetragonal (4/mmm)',
    '-3':     'Trigonal (-3)',
    '-3m':    'Trigonal (-3m)',
    '6/m':    'Hexagonal (6/m)',
    '6/mmm':  'Hexagonal (6/mmm)',
    'm-3':    'Cubic (m-3)',
    'm-3m':   'Cubic (m-3m)',
}


def laue_metric_residual(vol: VolumeData, laue_group: str) -> float | None:
    """Largest relative violation of `W^T G* W = G*` over the group's operations.

    `G* = B^T B` is the reciprocal metric from the volume's UB. Zero for a
    group the cell really has; ~1e-3 for a pseudo-symmetric cell (allowed);
    order 1 when the operations are in the wrong setting or the wrong group
    was chosen. None if the volume carries no UB.
    """
    ub = vol.metadata.get('ub')
    if ub is None:
        return None
    wavelength = float(vol.metadata.get('wavelength') or 1.0)
    recip = np.asarray(ub, dtype=np.float64) / wavelength
    gstar = recip.T @ recip
    scale = float(np.abs(gstar).max())
    worst = 0.0
    for op in get_symmetry_operations(laue_group):
        worst = max(worst, float(np.abs(op.T @ gstar @ op - gstar).max()) / scale)
    return worst


# ──────────────────────────────────────────────────────────────────
# GPU detection
# ──────────────────────────────────────────────────────────────────

def _has_gpu() -> bool:
    """Check if CuPy + CUDA GPU is available."""
    try:
        import cupy as cp
        cp.cuda.Device(0).compute_capability
        return True
    except Exception:
        return False


HAS_GPU = _has_gpu()


# ──────────────────────────────────────────────────────────────────
# Operation maps in array-index space
# ──────────────────────────────────────────────────────────────────

_INDEX_MAP_TOL = 1e-4     # index units; a map is exact when all entries are integers
_PERM_NEGLECT_TOL = 0.25  # voxels; off-permutation terms below this are rounded away


def index_space_ops(vol: VolumeData, laue_group: str) -> list[dict[str, Any]]:
    """Express each Miller-index operation W as an array-index map i' = M i + t.

    With `hkl = origin + A i` (see `volume_affine`):
        M = A^-1 W A,   t = A^-1 (W origin - origin).
    Each entry has 'op' (W), 'M', 't' and 'kind':
      'perm'     M is a signed permutation with integer t: exact 1D gather
      'nearest'  M is a signed permutation up to terms that displace a source
                 position by < `_PERM_NEGLECT_TOL` voxels anywhere in the
                 volume, but its scale/offset are not integers (a raster whose
                 layer step differs from the pixel step, a pseudo-symmetric
                 cell with dh != dk, an unaligned binning): every target voxel
                 takes its nearest source voxel through 1D rounded index
                 arrays, as rspace3d <= 2.0.0 did for all operations (<= 0.5
                 voxel positional error, no smoothing)
      'integer'  M and t integer but not a permutation (hexagonal/trigonal
                 3- and 6-fold operations on a regular grid): exact gather
      'interp'   otherwise (a sheared monoclinic unwarp raster under a
                 2-fold): trilinear, support-normalised interpolation.
                 A unit-scale map with a half-voxel offset (an odd layer
                 count binned by 2) is also 'interp', because nearest-voxel
                 rounding cannot choose between two equal neighbours
    """
    origin, A = volume_affine(vol)
    A_inv = np.linalg.inv(A)
    n = np.array(vol.intensity.shape, dtype=np.float64)
    centre = (n - 1) / 2.0
    corners = np.array([[i, j, k] for i in (0, n[0] - 1) for j in (0, n[1] - 1)
                        for k in (0, n[2] - 1)], dtype=np.float64) - centre
    result = []
    for op in get_symmetry_operations(laue_group):
        M = A_inv @ op @ A
        t = A_inv @ (op @ origin - origin)
        M_int = np.rint(M)
        t_int = np.rint(t)
        exact = (np.abs(M - M_int).max() < _INDEX_MAP_TOL
                 and np.abs(t - t_int).max() < _INDEX_MAP_TOL)
        # dominant entry per row: the candidate signed-permutation structure.
        # The dropped terms are evaluated relative to the volume centre (their
        # value there is absorbed into the offset), so a small raster shear
        # displaces a source position by at most its edge-to-centre effect.
        dominant = np.argmax(np.abs(M), axis=1)
        P = np.zeros_like(M)
        P[np.arange(3), dominant] = M[np.arange(3), dominant]
        is_perm_structure = len(set(dominant.tolist())) == 3
        neglected = np.abs(corners @ (M - P).T).max() if is_perm_structure else np.inf
        t_P = t + (M - P) @ centre
        # A unit-scale map with a half-integer offset puts each target voxel
        # at the midpoint of two source voxels. Example: an odd layer count
        # binned by 2. rint (round half to even) then reads some source
        # layers two times and does not read the other layers. Thus,
        # interpolate this map.
        unit = np.abs(np.abs(P[np.arange(3), dominant]) - 1) < _INDEX_MAP_TOL
        half_tie = np.abs(t_P - np.floor(t_P) - 0.5) < _INDEX_MAP_TOL
        tie = bool(np.any(unit & half_tie))
        if exact and is_perm_structure and np.all(np.abs(M_int).sum(axis=1) == 1):
            result.append({'op': op, 'M': M_int.astype(np.int64),
                           't': t_int.astype(np.int64), 'kind': 'perm'})
        elif exact:
            result.append({'op': op, 'M': M_int.astype(np.int64),
                           't': t_int.astype(np.int64), 'kind': 'integer'})
        elif neglected < _PERM_NEGLECT_TOL and not tie:
            result.append({'op': op, 'M': P, 't': t_P, 'kind': 'nearest'})
        else:
            result.append({'op': op, 'M': M, 't': t, 'kind': 'interp'})
    return result


def reciprocal_metric(vol: VolumeData) -> tuple[npt.NDArray[np.float64], str]:
    """Reciprocal metric tensor G* = B^T B with B = UB / wavelength.

    |q| = sqrt(hkl^T G* hkl) is then 1/d in 1/Angstrom. Without a UB matrix
    the function returns the identity and the unit 'r.l.u.'.
    """
    ub = vol.metadata.get('ub')
    wavelength = float(vol.metadata.get('wavelength', 0.0) or 0.0)
    if ub is None or wavelength <= 0:
        return np.eye(3), 'r.l.u.'
    B = np.asarray(ub, dtype=np.float64) / wavelength
    return B.T @ B, '1/Angstrom'


# ──────────────────────────────────────────────────────────────────
# Combined outlier rejection + symmetrization (per-voxel orbit engine)
# ──────────────────────────────────────────────────────────────────

def orbit_plan(vol: VolumeData, laue_group: str) -> Any:
    """Operation tables of `vol` and `laue_group` for `orbit_engine` (an `OrbitPlan`)."""
    from . import orbit_engine
    ops = index_space_ops(vol, laue_group)
    identity_index = next(i for i, o in enumerate(ops)
                          if np.array_equal(o['op'], np.eye(3, dtype=int)))
    origin, A = volume_affine(vol)
    G, q_unit = reciprocal_metric(vol)
    return orbit_engine.build_plan(vol.intensity.shape, ops, identity_index, origin, A, G,
                                   q_unit, _EDGE_SNAP_TOL)


def symmetrize_volume(vol: VolumeData, laue_group: str,
                      sigma: float | None = 3.0, min_valid: int = 3,
                      progress_callback: Callable[[int, int], None] | None = None,
                      use_gpu: bool | None = None,
                      poisson_floor: bool = True,
                      propagate_variance: bool = True,
                      weighting: str = 'equal',
                      rejection: str | None = None,
                      noise_model: Any = None,
                      grad: np.ndarray | None = None) -> VolumeData:
    """Outlier-reject and symmetry-average in a single orbit gather.

    For each voxel the function:
      1. gathers intensities at all symmetry-equivalent positions (orbit),
      2. computes the orbit median and MAD,
      3. marks any orbit member deviating by > `sigma * scale` from the median
         as `NaN` (outlier removed from the mean, not replaced),
      4. writes the `nanmean` of the surviving orbit members to the voxel.

    One orbit gather per voxel — no separate reject/symmetrize passes.
    Unmeasured voxels (NaN) participate only where a measurement exists; a
    voxel whose whole orbit is unmeasured stays NaN.

    Operations are applied in array-index space through the volume's affine
    grid (`index_space_ops`): exact index maps are gathered directly (fast
    1D maps for signed permutations, index arithmetic for the hexagonal and
    trigonal 3-/6-fold operations); a permutation-structured map whose scale
    or offset is not integer (layer step != pixel step, dh != dk of a
    pseudo-symmetric cell) takes the nearest source voxel, as 2.0.0 did
    (`metadata['symmetry_max_index_error']` records the largest rounding);
    an operation that mixes axes off the grid (a sheared monoclinic unwarp
    raster under a 2-fold) is sampled by trilinear interpolation of the
    measured neighbours. A unit-scale map with an offset of exactly half a
    voxel (an odd layer count binned by 2) is also interpolated.

    The work is done by `orbit_engine`: one CUDA thread per voxel on the GPU.
    On the CPU, the same per-voxel code runs on all cores if numba is
    installed, else a NumPy engine runs. The engines give bit-identical
    intensity, variance and outlier count.

    Variance. If the volume has a variance (stored in `vol.variance`, or
    derived, see `variance_is_derived`), the output has a variance too:
      * a member that reads one voxel has the variance of that voxel,
      * an interpolated member has the variance sum_k (w_k / W)^2 var_k of its
        measured corners (weights w_k, W = sum_k w_k),
      * the output variance is sum_j (c_j / m)^2 var_j over the distinct
        sources j of the m surviving members, where c_j is the number of
        survivors that read source j. A voxel on a special position (for
        example h = 0 under a mirror) reads the same source more than once.
        These reads are not independent, and the formula counts them so.
    The output variance is always stored, also for a derived input variance.
    It is one more float32 array (4 bytes per voxel) in memory, and the
    dataset `/uncertainty` in the HDF5 file (about the size of `/data`).
    Without a variance, or with `propagate_variance=False`, the output
    variance is None and `metadata['variance_model']` is removed.

    Diagnostics. `metadata['symmetry_diagnostics']` holds per-shell sums in
    10 shells of equal width in |q| (1/d in 1/Angstrom from the UB matrix, or
    r.l.u. without one): voxels with members, finite members, rejected
    members, voxels tested for outliers and voxels whose scale is set by the
    Poisson floor, R_sym = sum |x_i - mean| / sum |x_i| over the survivors,
    and (with variance) chi^2 = sum (x_i - mean)^2 / var_i over the distinct
    survivors with var_i > 0, with nu = (number of these survivors) - 1.
    Each voxel of an orbit has the same members. Thus each voxel adds its
    chi^2, nu and R_sym sums divided by the number of distinct sources among
    its finite members, so that an exact orbit counts once. With
    interpolated members this weight is approximate.
    `format_symmetry_diagnostics` prints them as a table.

    Parameters
    ----------
    sigma : float or None
        Multiplier of the robust scale for the outlier threshold. Pass `None`
        to skip outlier rejection (pure symmetric averaging).
    min_valid : int
        A voxel's orbit must contain at least this many finite measurements
        for outlier flagging to apply. A non-integer value is rounded up
        (the rule is n_valid >= min_valid).
    poisson_floor : bool
        The robust scale is `max(1.4826 * MAD, sqrt(max(median, 1)))`: the
        MAD is floored at the Poisson noise of the orbit median, so a tied,
        low-count orbit (MAD = 0) does not reject ordinary counting noise
        (an orbit of seven zeros and a one kept its one; rspace3d <= 2.0.0
        rejected it at any sigma). `False` restores the bare MAD.
    use_gpu : bool or None
        None uses the GPU when CuPy finds one. If the input volume does not
        fit in GPU memory, or if the GPU engine fails, a warning is given and
        the CPU engine is used.
    propagate_variance : bool
        True (default) propagates the variance of the input, if it has one.
        False gives the output of rspace3d 2.2.0: no variance and no chi^2.
        This uses less memory and time. The intensity and the outlier count
        do not change.
    weighting : str
        'equal' (default) gives the mean of rspace3d 2.3.0.
        'counts' weights each member with its pixel count N_i (`vol.counts`
        gathered with the same maps, interpolated for an 'interp' member).
        The interpolated count is used on purpose: the corners of an
        interpolated member also go into the neighbouring voxels, so their
        errors are correlated, and the interpolated count gives them less
        weight than independent members. Without `vol.counts`, N_i = 1.
        'adaptive' weights each member with 1 / v_i (see `noise_model`).
        A weighted mean is the float64 sum of w_i x_i over the survivors,
        divided by the float64 sum of w_i, rounded to float32. With weights,
        the output variance is sum_j (W_j / W)^2 var_j over the distinct
        sources j. W_j is the sum of the weights of the survivors that read
        source j, and W is the sum of all survivor weights.
    rejection : str or None
        None (default) keeps the rule of rspace3d 2.3.0: 'mad' if `sigma`
        is a number, 'none' if `sigma` is None.
        'none' rejects no member. 'mad' is the median and MAD rule above.
        'model' rejects member i if |x_i - med| > sigma * sqrt(v_i), the
        orbit has at least `min_valid` finite members, and member i is not
        the nearest member to the median (the member or members with the
        smallest |x_j - med| always stay). It does not use the MAD or
        `poisson_floor`. 'mad' and 'model' need a `sigma`.
    noise_model : averaging.NoiseModel or None
        The variance of a member, at the orbit median med:
        v_i = gain max(med, 1) / N_i + (rel med)^2 + (kappa G)^2 + floor^2.
        G is the gradient magnitude of a first-pass symmetric average at
        the target voxel (`averaging.first_pass_gradient`). 'adaptive' and
        'model' use it. If it is None, the function fits it with
        `averaging.fit_noise_model`. `metadata['noise_model']` then holds it.
        If that fit fails, the function raises ValueError.
    grad : ndarray or None
        The gradient volume G (shape of `vol.intensity`), for example from
        `averaging.fit_noise_model(..., return_gradient=True)`. 'adaptive'
        and 'model' use it if kappa > 0. None calculates G when it is
        necessary. Give G to calculate it only once for several calls.

    The defaults give an output that is bit-identical to rspace3d 2.3.0.
    """
    from . import orbit_engine

    if use_gpu is None:
        use_gpu = HAS_GPU
    if sigma is not None and (not np.isfinite(sigma) or sigma <= 0):
        raise ValueError(f"sigma must be positive or None, got {sigma!r}")
    rej = rejection if rejection is not None else ('mad' if sigma is not None else 'none')
    if rej != 'none' and sigma is None:
        raise ValueError(f"rejection {rej!r} needs a sigma")
    av = orbit_engine.Averaging(weighting=weighting, rejection=rej)

    residual = laue_metric_residual(vol, laue_group)
    if residual is not None and residual > 0.02:
        warnings.warn(
            f"Laue group {laue_group!r} does not preserve the reciprocal metric of "
            f"this cell (relative residual {residual:.3g}); check the group and the "
            "cell setting. Continuing with the requested projection.",
            RuntimeWarning, stacklevel=2)

    ops = index_space_ops(vol, laue_group)
    n_ops = len(ops)
    kinds = {o['kind'] for o in ops}
    mapping = ('interpolated' if 'interp' in kinds
               else 'nearest' if 'nearest' in kinds else 'exact')

    data = np.ascontiguousarray(vol.intensity, dtype=np.float32)
    variance = (np.ascontiguousarray(vol.variance, dtype=np.float32)
                if vol.variance is not None and propagate_variance else None)
    # A derived variance max(I, 1) is calculated by the engine from the data
    derive = bool(propagate_variance) and variance_is_derived(vol)
    plan = orbit_plan(vol, laue_group)

    # Noise model and gradient volume (only for 'adaptive' and 'model')
    if av.uses_model:
        from . import averaging
        if grad is not None:
            grad = averaging._check_gradient(grad, data.shape)
        fitted = noise_model is None
        if fitted:
            try:
                noise_model, grad = averaging._fit(vol, laue_group, plan, data,
                                                   use_gpu=use_gpu, grad=grad)
            except ValueError as exc:
                raise ValueError(
                    f"weighting {weighting!r} with rejection {rej!r} needs a noise model, "
                    f"and the noise-model fit failed. {exc} Give a noise_model, or use "
                    "weighting 'equal' or 'counts' with rejection 'mad'.") from exc
        elif noise_model.kappa > 0 and grad is None:
            grad = averaging.first_pass_gradient(plan, data, use_gpu)
        av.noise = noise_model.params
        av.grad = grad if noise_model.kappa > 0 else None
    if av.uses_counts and vol.counts is not None:
        av.counts = np.ascontiguousarray(vol.counts, dtype=np.float32)

    # The threshold is float32(sigma) * scale, as in rspace3d <= 2.2.
    sigma_f = None if sigma is None or rej == 'none' else float(sigma)
    # The engines test n_valid >= an integer. ceil keeps the rule
    # n_valid >= min_valid of rspace3d 2.2.0 for a non-integer value.
    min_valid_i = int(np.clip(np.ceil(float(min_valid)), 0, orbit_engine.MAX_OPS + 1))
    res = orbit_engine.run(plan, data, variance, sigma_f, min_valid_i,
                           bool(poisson_floor), bool(use_gpu), progress_callback,
                           derive_variance=derive, averaging=av)
    del av

    meta: dict[str, Any] = {**vol.metadata, 'laue_group': laue_group,
                            'symmetry_ops_applied': n_ops,
                            'symmetry_mapping': mapping,
                            'symmetry_max_index_error': plan.max_index_error,
                            'symmetry_diagnostics': orbit_engine.result_diagnostics(plan, res)}
    for key in ('weighting', 'rejection', 'noise_model', 'averaging_choice'):  # earlier run
        meta.pop(key, None)
    if res.variance is None:
        meta.pop('variance_model', None)
    if residual is not None:
        meta['laue_metric_residual'] = residual
    if weighting != 'equal' or rejection is not None:
        meta['weighting'] = weighting
        meta['rejection'] = rej
    if rej == 'model' or weighting == 'adaptive':
        meta['noise_model'] = {**noise_model.to_dict(), 'fitted': fitted}
    if rej != 'none':
        meta['sigma'] = sigma
        meta['min_valid'] = min_valid
        if rej == 'mad':
            meta['poisson_floor'] = poisson_floor
        else:
            meta.pop('poisson_floor', None)
        meta['n_outliers_removed'] = res.n_outliers_removed
    return VolumeData(
        intensity=res.intensity,
        H=vol.H.copy(), K=vol.K.copy(), L=vol.L.copy(),
        plane_type=vol.plane_type,
        metadata=meta,
        variance=res.variance,
    )


def format_symmetry_diagnostics(diag: dict[str, Any] | None) -> str:
    """Short text table of `metadata['symmetry_diagnostics']`."""
    from .orbit_engine import format_symmetry_diagnostics as fmt
    return fmt(diag)


# ──────────────────────────────────────────────────────────────────
# Volume slice extraction (shared by all viewers)
# ──────────────────────────────────────────────────────────────────

_EDGE_SNAP_TOL = 1e-9   # index units: round-off overshoot past the grid edge is snapped back


def extract_volume_slice(
    vol: VolumeData,
    plane_index: int,
    target_val: float,
    int_range: float = 0.0,
) -> tuple[
    npt.NDArray[np.float32],     # slice_2d
    npt.NDArray[np.floating],    # x_ax
    npt.NDArray[np.floating],    # y_ax
    str,                         # x_label
    str,                         # y_label
    str,                         # fixed_label
    float,                       # actual_val
    int,                         # n_slices
]:
    """Extract a 2D slice from a 3D volume at a constant Miller index.

    The volume's native plane is a direct array slice. The other two planes
    are sampled through the volume's affine grid (`volume_affine`), which
    handles sheared unwarp rasters (monoclinic/triclinic cross-terms) and
    regular rawrecon grids alike: the slab is returned on the physical
    Miller-index axes (x_ax, y_ax) of the requested plane.

    Integration over `int_range` returns the NaN-aware mean over the
    contributing layers times their number (equal to the plain sum when all
    layers are measured); a point stays NaN only where every layer is NaN.

    Parameters
    ----------
    vol : VolumeData
    plane_index : int
        0=HK (fix L), 1=HL (fix K), 2=KL (fix H)
    target_val : float
        Value of the fixed Miller index
    int_range : float
        Integration half-width (0 = single slice)

    Returns
    -------
    (slice_2d, x_ax, y_ax, x_label, y_label, fixed_label, actual_val, n_slices)
    with slice_2d of shape (len(y_ax), len(x_ax)) for imshow.
    """
    cfgs = [
        (vol.H, vol.K, vol.L, 'h', 'k', 'l', 0, 1, 2),  # HK fix L
        (vol.H, vol.L, vol.K, 'h', 'l', 'k', 0, 2, 1),  # HL fix K
        (vol.K, vol.L, vol.H, 'k', 'l', 'h', 1, 2, 0),  # KL fix H
    ]
    x_ax, y_ax, fixed_ax = cfgs[plane_index][0], cfgs[plane_index][1], cfgs[plane_index][2]
    x_label, y_label, fixed_label = cfgs[plane_index][3], cfgs[plane_index][4], cfgs[plane_index][5]
    x_dim, y_dim, fixed_dim = cfgs[plane_index][6], cfgs[plane_index][7], cfgs[plane_index][8]

    fixed_ax = np.asarray(fixed_ax, dtype=np.float64)
    if int_range < 1e-6:
        indices = np.array([int(np.argmin(np.abs(fixed_ax - target_val)))])
    else:
        # Keep a layer that is on an end of the window. Round-off can move
        # the end a small distance inward (0.35 + 0.1 = 0.44999999999999996).
        step = abs(fixed_ax[1] - fixed_ax[0]) if len(fixed_ax) > 1 else 1.0
        tol = _EDGE_SNAP_TOL * step
        lo, hi = target_val - int_range - tol, target_val + int_range + tol
        indices = np.where((fixed_ax >= lo) & (fixed_ax <= hi))[0]
        if len(indices) == 0:
            indices = np.array([int(np.argmin(np.abs(fixed_ax - target_val)))])
    actual_val = float(fixed_ax[indices[len(indices) // 2]])
    n_slices = len(indices)

    plane_types = ['HK', 'HL', 'KL']
    if plane_types[plane_index] == vol.plane_type:
        sl = _extract_native(vol.intensity, fixed_dim, indices)
    elif int_range < 1e-6:
        # a single non-native cut is interpolated at the requested value itself
        actual_val = float(target_val)
        sl = _extract_nonnat(vol, x_dim, y_dim, fixed_dim, [actual_val])
    else:
        sl = _extract_nonnat(vol, x_dim, y_dim, fixed_dim,
                             [float(fixed_ax[i]) for i in indices])

    return sl, x_ax, y_ax, x_label, y_label, fixed_label, actual_val, n_slices


def _integrate(slabs: npt.NDArray[np.floating], axis: int) -> npt.NDArray[np.float32]:
    """NaN-aware layer integration: mean over measured layers x number of layers."""
    n = slabs.shape[axis]
    if n == 1:
        return np.take(slabs, 0, axis=axis).astype(np.float32)
    with np.errstate(invalid='ignore', divide='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)       # all-NaN columns are fine
        mean = np.nanmean(slabs.astype(np.float64), axis=axis)
    return (mean * n).astype(np.float32)


def _extract_native(
    data: npt.NDArray[Any],
    fixed_dim: int,
    indices: npt.NDArray[np.intp],
) -> npt.NDArray[np.float32]:
    """Native plane: direct array slice along the fixed physical axis.

    Returns (len(y_ax), len(x_ax)): the two remaining axes in physical order
    are (x, y) for every plane, so a transpose gives imshow orientation.
    """
    slab = _integrate(np.take(data, indices, axis=fixed_dim), axis=fixed_dim)
    return np.ascontiguousarray(slab.T.astype(np.float32))


def _extract_nonnat(
    vol: VolumeData,
    x_dim: int,
    y_dim: int,
    fixed_dim: int,
    target_vals: list[float],
) -> npt.NDArray[np.float32]:
    """Non-native plane: trilinear sampling through the affine grid.

    Every output point (x, y, fixed) is converted to fractional array indices
    with the inverse of `volume_affine`, then interpolated with support
    normalisation (interpolated intensity / interpolated finite-mask), so a
    single unmeasured neighbour does not erase a sample and a point is NaN
    only where no neighbour is measured. Coordinates that overshoot the grid
    edge by round-off (< `_EDGE_SNAP_TOL`) are snapped onto it.
    """
    data = vol.intensity
    axes = [np.asarray(vol.H, dtype=np.float64), np.asarray(vol.K, dtype=np.float64),
            np.asarray(vol.L, dtype=np.float64)]
    x_ax, y_ax = axes[x_dim], axes[y_dim]
    origin, A = volume_affine(vol)
    A_inv = np.linalg.inv(A)
    shape = np.array(data.shape)

    X, Y = np.meshgrid(x_ax, y_ax, indexing='ij')           # (nx, ny)
    hkl = np.empty((3, X.size), dtype=np.float64)
    hkl[x_dim] = X.ravel()
    hkl[y_dim] = Y.ravel()
    slabs = []
    for tv in target_vals:
        hkl[fixed_dim] = tv
        coords = A_inv @ (hkl - origin[:, None])            # (3, N) fractional indices
        upper = (shape - 1)[:, None].astype(np.float64)
        near = (coords > -_EDGE_SNAP_TOL) & (coords < upper + _EDGE_SNAP_TOL)
        coords = np.where(near, np.clip(coords, 0.0, upper), coords)
        # Only the box of source voxels touched by the plane is materialised.
        lo = np.clip(np.floor(coords.min(axis=1)).astype(int), 0, shape - 1)
        hi = np.clip(np.ceil(coords.max(axis=1)).astype(int), 0, shape - 1)
        box = tuple(slice(int(lo[d]), int(hi[d]) + 1) for d in range(3))
        sub = np.asarray(data[box], dtype=np.float32)
        finite = np.isfinite(sub)
        local = coords - lo[:, None]
        num = map_coordinates(np.where(finite, sub, np.float32(0.0)), local,
                              order=1, mode='constant', cval=0.0)
        den = map_coordinates(finite.astype(np.float32), local,
                              order=1, mode='constant', cval=0.0)
        with np.errstate(invalid='ignore', divide='ignore'):
            slab = np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan)
        slabs.append(slab.reshape(X.shape))
    slab = _integrate(np.stack(slabs, axis=0), axis=0)      # (nx, ny)
    return np.ascontiguousarray(slab.T.astype(np.float32))


# ──────────────────────────────────────────────────────────────────
# Save / load — HDF5 (MATLAB-compatible) and npz
# ──────────────────────────────────────────────────────────────────

# Scalar provenance attributes written by save_volume_h5 (key -> loader cast)
_PROVENANCE_KEYS: dict[str, Callable[[Any], Any]] = {
    'sigma': float, 'min_valid': int, 'poisson_floor': bool,
    'n_outliers_removed': int, 'symmetry_ops_applied': int,
    'symmetry_mapping': str, 'symmetry_max_index_error': float,
    'laue_metric_residual': float,
    'reconstructed_by': str, 'n_measured_voxels': int, 'measured_pct': float,
    'n_files': int, 'd_min': float, 'hot_pixel_cutoff': float,
    'polarization_axis': str, 'geometry_method': str, 'source_manifest': str,
    'variance_model': str, 'weighting': str, 'rejection': str,
}


def save_volume_h5(path: str, vol: VolumeData, compression: str = 'gzip',
                   compression_level: int = 4) -> None:
    """Save volume as HDF5 file, MATLAB-compatible.

    Datasets: /data (3D intensity), /H, /K, /L (1D axes),
    plus metadata attributes.  Uses gzip compression by default.
    Optional datasets: /counts (contributing pixels per voxel) and
    /uncertainty (float32, the standard deviation sqrt(`vol.variance`) of
    each voxel value, NaN where unmeasured, same chunks as /data).
    `load_volume_h5` reads /uncertainty back as `variance = uncertainty^2`.
    The attribute `sigma` is a different value: the outlier threshold of
    the symmetrisation.
    A derived variance (`variance_is_derived`) is not written. The
    attribute `variance_model` and the bin factors are sufficient to
    calculate it again.
    Optional group: /diagnostics (the per-shell sums of
    `metadata['symmetry_diagnostics']`, one 1D dataset per sum).
    Optional attribute: noise_model (`metadata['noise_model']` as a JSON
    text). `averaging.NoiseModel.from_dict` makes the model from the dict.
    Optional attribute: averaging_choice (`metadata['averaging_choice']` as
    a JSON text): the option that the averaging panel or the command line
    applied, the reason and a short score summary (`averaging.choice_record`).

    Can be read in MATLAB with:
        data = h5read('file.h5', '/data');
        H = h5read('file.h5', '/H');
        K = h5read('file.h5', '/K');
        L = h5read('file.h5', '/L');
    """
    import h5py
    comp_opts = {}
    if compression:
        comp_opts = {'compression': compression,
                     'compression_opts': compression_level}

    with h5py.File(path, 'w') as f:
        dset = f.create_dataset('data', data=vol.intensity, **comp_opts)
        f.create_dataset('H', data=vol.H, **comp_opts)
        f.create_dataset('K', data=vol.K, **comp_opts)
        f.create_dataset('L', data=vol.L, **comp_opts)
        if vol.counts is not None:
            f.create_dataset('counts', data=vol.counts, **comp_opts)
        if vol.variance is not None:
            # standard uncertainty of each voxel, chunked like /data
            unc = np.sqrt(np.asarray(vol.variance, dtype=np.float32))
            f.create_dataset('uncertainty', data=unc, chunks=dset.chunks, **comp_opts)
            del unc

        # Metadata as attributes
        f.attrs['plane_type'] = vol.plane_type
        f.attrs['wavelength'] = vol.metadata.get('wavelength', 0.0)
        f.attrs['grid_kind'] = vol.metadata.get('grid_kind', 'unwarp_raster')
        if 'laue_group' in vol.metadata:
            f.attrs['laue_group'] = vol.metadata['laue_group']
        f.attrs['bin_xy'] = vol.metadata.get('bin_xy', 1)
        f.attrs['bin_z'] = vol.metadata.get('bin_z', 1)
        if 'source_folder' in vol.metadata:
            f.attrs['source_folder'] = vol.metadata['source_folder']
        # Processing provenance (rejection, symmetry, reconstruction)
        for key in _PROVENANCE_KEYS:
            value = vol.metadata.get(key)
            if value is not None:
                f.attrs[key] = value
        # Noise model of the averaging (a JSON text)
        if vol.metadata.get('noise_model') is not None:
            import json
            f.attrs['noise_model'] = json.dumps(vol.metadata['noise_model'])
        # Choice of the averaging panel or of the command line (a JSON text)
        if vol.metadata.get('averaging_choice') is not None:
            import json
            f.attrs['averaging_choice'] = json.dumps(vol.metadata['averaging_choice'])

        # Symmetrisation diagnostics per |q| shell
        diag = vol.metadata.get('symmetry_diagnostics')
        if diag:
            from .orbit_engine import diagnostics_to_arrays
            grp = f.create_group('diagnostics')
            grp.attrs['q_unit'] = diag['q_unit']
            grp.attrs['has_variance'] = bool(diag['has_variance'])
            for name, arr in diagnostics_to_arrays(diag).items():
                grp.create_dataset(name, data=arr)

        # Unit cell parameters
        cell = vol.metadata.get('cell')
        if cell:
            f.attrs['cell_a'] = cell['a']
            f.attrs['cell_b'] = cell['b']
            f.attrs['cell_c'] = cell['c']
            f.attrs['cell_alpha'] = cell['alpha']
            f.attrs['cell_beta'] = cell['beta']
            f.attrs['cell_gamma'] = cell['gamma']

        # M_inv matrix (2x2 pixel-to-Miller transform)
        m_inv = vol.metadata.get('M_inv')
        if m_inv is not None:
            f.create_dataset('M_inv', data=m_inv)

        # UB matrix
        ub = vol.metadata.get('ub')
        if ub is not None:
            f.create_dataset('UB', data=ub)

        # Cartesian step and grid center (for Cartesian display)
        if 's' in vol.metadata:
            f.attrs['s'] = vol.metadata['s']
        if 'cx' in vol.metadata:
            f.attrs['cx'] = vol.metadata['cx']
        if 'cy' in vol.metadata:
            f.attrs['cy'] = vol.metadata['cy']

        # Coverage-mask diagnostics
        for key in ('morph_size',
                    'morph_mode',
                    'morph_s_per_pixel',
                    'morph_recip_period',
                    'n_unmeasured_per_frame_mean',
                    'n_unmeasured_per_frame_min',
                    'n_unmeasured_per_frame_max',
                    'n_unmeasured_per_frame_pct',
                    'frame_npix'):
            if key in vol.metadata and vol.metadata[key] is not None:
                f.attrs[key] = vol.metadata[key]


def load_volume_h5(path: str) -> VolumeData:
    """Load a volume from an HDF5 file (2.0.0 files load unchanged)."""
    import h5py
    with h5py.File(path, 'r') as f:
        intensity = np.array(f['data'])
        H = np.array(f['H'])
        K = np.array(f['K'])
        L = np.array(f['L'])
        counts = np.array(f['counts']) if 'counts' in f else None
        variance = None
        if 'uncertainty' in f:
            variance = np.array(f['uncertainty'], dtype=np.float32)
            np.square(variance, out=variance)
        metadata = {
            'wavelength': float(f.attrs.get('wavelength', 0)),
            'laue_group': str(f.attrs.get('laue_group', '')),
            'source_folder': str(f.attrs.get('source_folder', '')),
            'bin_xy': int(f.attrs.get('bin_xy', 1)),
            'bin_z': int(f.attrs.get('bin_z', 1)),
            'grid_kind': str(f.attrs.get('grid_kind', 'unwarp_raster')),
        }
        for key, kind in _PROVENANCE_KEYS.items():
            if key in f.attrs:
                metadata[key] = kind(f.attrs[key])
        if 'noise_model' in f.attrs:
            import json
            metadata['noise_model'] = json.loads(str(f.attrs['noise_model']))
        if 'averaging_choice' in f.attrs:
            import json
            metadata['averaging_choice'] = json.loads(str(f.attrs['averaging_choice']))
        # M_inv, UB, cell, Cartesian step
        if 'M_inv' in f:
            metadata['M_inv'] = np.array(f['M_inv'])
        if 'UB' in f:
            metadata['ub'] = np.array(f['UB'])
        if 's' in f.attrs:
            metadata['s'] = float(f.attrs['s'])
        if 'cx' in f.attrs:
            metadata['cx'] = float(f.attrs['cx'])
        if 'cy' in f.attrs:
            metadata['cy'] = float(f.attrs['cy'])
        if 'cell_a' in f.attrs:
            metadata['cell'] = {
                'a': float(f.attrs['cell_a']),
                'b': float(f.attrs['cell_b']),
                'c': float(f.attrs['cell_c']),
                'alpha': float(f.attrs['cell_alpha']),
                'beta': float(f.attrs['cell_beta']),
                'gamma': float(f.attrs['cell_gamma']),
            }
        # Coverage-mask diagnostics
        int_keys = {'morph_size', 'n_unmeasured_per_frame_min',
                    'n_unmeasured_per_frame_max', 'frame_npix'}
        str_keys = {'morph_mode'}
        for key in ('morph_size',
                    'morph_mode',
                    'morph_s_per_pixel',
                    'morph_recip_period',
                    'n_unmeasured_per_frame_mean',
                    'n_unmeasured_per_frame_min',
                    'n_unmeasured_per_frame_max',
                    'n_unmeasured_per_frame_pct',
                    'frame_npix'):
            if key in f.attrs:
                v = f.attrs[key]
                if key in int_keys:
                    metadata[key] = int(v)
                elif key in str_keys:
                    metadata[key] = str(v)
                else:
                    metadata[key] = float(v)
        if 'diagnostics' in f:
            from .orbit_engine import diagnostics_from_arrays
            grp = f['diagnostics']
            metadata['symmetry_diagnostics'] = diagnostics_from_arrays(
                {name: np.array(grp[name]) for name in grp},
                str(grp.attrs['q_unit']), bool(grp.attrs['has_variance']))
        # A raw unbinned unwarp volume of rspace3d <= 2.2 has no variance
        # model. Its variance is max(I, 1), the same as in a newer file.
        if ('variance_model' not in metadata and variance is None and counts is None
                and metadata['grid_kind'] == 'unwarp_raster' and not metadata['laue_group']
                and 'symmetry_ops_applied' not in metadata
                and metadata['bin_xy'] == 1 and metadata['bin_z'] == 1):
            metadata['variance_model'] = 'poisson_stored_units'
        plane_type = str(f.attrs.get('plane_type', 'HK'))
    return VolumeData(
        intensity=intensity, H=H, K=K, L=L,
        plane_type=plane_type, metadata=metadata, counts=counts,
        variance=variance,
    )


# ──────────────────────────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Validating Laue group generation...")
    for name in _LAUE_GENERATORS:
        ops = get_symmetry_operations(name)
        expected = _EXPECTED_ORDERS[name]
        status = 'OK' if len(ops) == expected else 'FAIL'
        print(f"  {name:8s}: {len(ops):3d} ops (expected {expected}) [{status}]")
        op_set = {tuple(o.flatten()) for o in ops}
        for a in ops:
            for b in ops:
                assert tuple((a @ b).flatten()) in op_set
    print("\nAll Laue groups validated.")
