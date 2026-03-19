"""
volume_builder.py — Build, bin, outlier-reject, and symmetrize
3D reciprocal space volumes from CrysAlisPro unwarp .img files.

Reads .img headers directly for grid computation (no .par file needed).
"""

import numpy as np
import os
import struct
import io
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable, Dict
from scipy.ndimage import map_coordinates


# ──────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────

@dataclass
class VolumeData:
    """3D reciprocal space volume on a regular Miller index grid."""
    intensity: np.ndarray     # (nh, nk, nl) — int32 (raw) or float32 (processed)
    H: np.ndarray             # (nh,) 1D array of h values
    K: np.ndarray             # (nk,) 1D array of k values
    L: np.ndarray             # (nl,) 1D array of l values
    plane_type: str           # 'HK', 'HL', or 'KL'
    metadata: dict            # wavelength, UB, cell params, etc.


# ──────────────────────────────────────────────────────────────────
# Fast header reading
# ──────────────────────────────────────────────────────────────────

def _read_header_fast(path: str) -> dict:
    """Read minimal header fields from a CrysAlisPro .img file."""
    with open(path, 'rb') as f:
        raw = f.read(2330)

    def u16(off):
        return struct.unpack_from('<H', raw, off)[0]
    def f64(off):
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
        plane_type, fixed_value = 'HK', l_fixed

    ub = np.array([f64(2256 + i * 8) for i in range(9)]).reshape(3, 3)

    return {
        'nx': nx, 'ny': ny, 'plane_type': plane_type,
        'fixed_value': fixed_value, 'd_min': f64(1024),
        'wavelength': f64(2104), 'ub': ub,
    }


def _read_intensity(path: str) -> np.ndarray:
    """Read intensity data from .img file using fabio (buffered).
    Returns int32 array (native format)."""
    from fabio.OXDimage import OxdImage
    with open(path, 'rb') as f:
        buf = io.BytesIO(f.read())
    img = OxdImage()
    img.read(buf)
    return img.data.astype(np.int32)


# ──────────────────────────────────────────────────────────────────
# Grid computation (from rsp_reader logic)
# ──────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────
# Par file reading
# ──────────────────────────────────────────────────────────────────

def find_par_file(unwarp_folder: str) -> Optional[str]:
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


def read_par_cell(par_filename: str) -> Optional[dict]:
    """Read unit cell parameters from a CrysAlisPro .par file.

    Primary method: compute from CRYSTALLOGRAPHY UB matrix + wavelength.
    Fallback: parse CELL line directly (if present).

    Returns dict with a, b, c (Angstrom), alpha, beta, gamma (degrees).
    """
    import re

    # Read UB matrix and wavelength from par file
    ub = None
    wavelength = None

    def _strip_esd(s):
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
                    vals = [float(x) for x in parts[idx + 1:idx + 10]]
                    if len(vals) == 9:
                        ub = np.array(vals).reshape(3, 3)
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
                    vals = parts[idx + 1:idx + 7]
                    if len(vals) == 6:
                        nums = [float(_strip_esd(v)) for v in vals]
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


def cell_from_ub(ub, wavelength):
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


def compute_plane_M_inv(ub, wavelength, plane_type):
    """Compute the 2x2 M_inv matrix for any plane type from the UB matrix.

    Uses the raw reciprocal vectors (not projected perpendicular to
    fixed axis) to match CrysAlisPro's pixel grid convention.
    """
    cfg = _PLANE_CONFIG[plane_type]
    recip = ub / wavelength
    v1 = recip[:, cfg['vec1_col']]
    v2 = recip[:, cfg['vec2_col']]

    e_x = v1 / np.linalg.norm(v1)
    v2_perp = v2 - np.dot(v2, e_x) * e_x
    e_y = v2_perp / np.linalg.norm(v2_perp)

    M = np.array([[np.dot(v1, e_x), np.dot(v2, e_x)],
                   [np.dot(v1, e_y), np.dot(v2, e_y)]])
    return np.linalg.inv(M)


_PLANE_CONFIG = {
    'HK': {'vec1_col': 0, 'vec2_col': 1, 'fixed_col': 2,
            'x_label': 'h', 'y_label': 'k', 'fixed_label': 'l'},
    'HL': {'vec1_col': 0, 'vec2_col': 2, 'fixed_col': 1,
            'x_label': 'h', 'y_label': 'l', 'fixed_label': 'k'},
    'KL': {'vec1_col': 1, 'vec2_col': 2, 'fixed_col': 0,
            'x_label': 'k', 'y_label': 'l', 'fixed_label': 'h'},
}


def compute_1d_axes(header: dict) -> Tuple[np.ndarray, np.ndarray]:
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

def bin_2d(data: np.ndarray, by: int, bx: int) -> np.ndarray:
    """Bin 2D array by averaging. Returns int32 (integer average)."""
    ny, nx = data.shape
    ny_t = (ny // by) * by
    nx_t = (nx // bx) * bx
    # Use int64 accumulator to avoid overflow, then back to int32
    trimmed = data[:ny_t, :nx_t].astype(np.int64)
    return (trimmed.reshape(ny_t // by, by, nx_t // bx, bx)
            .sum(axis=(1, 3)) // (by * bx)).astype(np.int32)


def bin_1d(arr: np.ndarray, b: int) -> np.ndarray:
    """Bin a 1D array by averaging groups of b elements."""
    n = (len(arr) // b) * b
    return arr[:n].reshape(-1, b).mean(axis=1)


def bin_volume(vol: VolumeData, bh: int, bk: int, bl: int) -> VolumeData:
    """Bin the 3D volume by averaging."""
    nh, nk, nl = vol.intensity.shape
    nh_t = (nh // bh) * bh
    nk_t = (nk // bk) * bk
    nl_t = (nl // bl) * bl
    trimmed = vol.intensity[:nh_t, :nk_t, :nl_t]

    # Keep int32 if input is integer, else float32
    if np.issubdtype(trimmed.dtype, np.integer):
        acc = trimmed.astype(np.int64).reshape(
            nh_t // bh, bh, nk_t // bk, bk, nl_t // bl, bl)
        binned = (acc.sum(axis=(1, 3, 5)) // (bh * bk * bl)).astype(np.int32)
    else:
        binned = trimmed.reshape(
            nh_t // bh, bh, nk_t // bk, bk, nl_t // bl, bl
        ).mean(axis=(1, 3, 5)).astype(np.float32)

    # Update metadata to reflect binned pixel grid
    meta = vol.metadata.copy()
    nh_new = nh_t // bh
    nk_new = nk_t // bk
    old_bin = meta.get('bin_xy', 1)
    if 's' in meta:
        meta['s'] = meta['s'] * max(bh, bk)  # Cartesian step scales with binning
    meta['cx'] = (nh_new + 1) / 2.0
    meta['cy'] = (nk_new + 1) / 2.0
    meta['bin_xy'] = old_bin * max(bh, bk)
    meta['bin_z'] = meta.get('bin_z', 1) * bl

    return VolumeData(
        intensity=binned,
        H=bin_1d(vol.H[:nh_t], bh),
        K=bin_1d(vol.K[:nk_t], bk),
        L=bin_1d(vol.L[:nl_t], bl),
        plane_type=vol.plane_type,
        metadata=meta,
    )


# ──────────────────────────────────────────────────────────────────
# Volume loading
# ──────────────────────────────────────────────────────────────────

def _filter_numbered_imgs(folder: str) -> List[str]:
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

    return [f for f in all_numbered if f.rsplit('_', 1)[0] == main_prefix]


def scan_unwarp_folder(folder: str) -> List[Tuple[str, float]]:
    """Scan folder for numbered .img files, return [(path, fixed_value)] sorted.

    Only loads files matching the pattern {prefix}_{number}.img with the
    most common prefix. Ignores files from other runs or tests.
    """
    img_files = _filter_numbered_imgs(folder)
    files = []
    for fname in img_files:
        path = os.path.join(folder, fname)
        hdr = _read_header_fast(path)
        files.append((path, hdr['fixed_value']))
    files.sort(key=lambda x: x[1])
    return files


def load_unwarp_folder(folder: str, bin_xy: int = 1, bin_z: int = 1,
                       progress_callback: Optional[Callable] = None) -> VolumeData:
    """Load all .img files from unwarp folder into a 3D volume.

    Data is kept as int32 (native CrysAlisPro format) to save memory.
    Only converted to float during processing (outlier rejection / symmetrization).
    """
    file_list = scan_unwarp_folder(folder)
    if not file_list:
        raise ValueError(f"No .img files found in {folder}")

    ref_header = _read_header_fast(file_list[0][0])
    nx, ny = ref_header['nx'], ref_header['ny']
    plane_type = ref_header['plane_type']
    axis_x_full, axis_y_full = compute_1d_axes(ref_header)

    if bin_xy > 1:
        axis_x = bin_1d(axis_x_full[:(len(axis_x_full) // bin_xy * bin_xy)], bin_xy)
        axis_y = bin_1d(axis_y_full[:(len(axis_y_full) // bin_xy * bin_xy)], bin_xy)
        nx_bin, ny_bin = len(axis_x), len(axis_y)
    else:
        axis_x, axis_y = axis_x_full, axis_y_full
        nx_bin, ny_bin = nx, ny

    n_files = len(file_list)

    # int32 storage — same as MATLAB's int32
    volume = np.zeros((nx_bin, ny_bin, n_files), dtype=np.int32)
    l_values = np.zeros(n_files, dtype=np.float64)

    for i, (path, fixed_val) in enumerate(file_list):
        if progress_callback:
            progress_callback(i, n_files)

        data = _read_intensity(path)  # int32 (ny, nx)
        if bin_xy > 1:
            data = bin_2d(data, bin_xy, bin_xy)  # int32
        volume[:, :, i] = data.T
        l_values[i] = fixed_val

    if progress_callback:
        progress_callback(n_files, n_files)

    # Compute full M_inv and cell from first file header
    from .rsp_reader import read_rsp_layer as _read_layer
    ref_layer = _read_layer(file_list[0][0])
    cell = cell_from_ub(ref_header['ub'], ref_header['wavelength'])

    # Also try par file
    par_path = find_par_file(folder)
    if par_path:
        par_cell = read_par_cell(par_path)
        if par_cell:
            cell = par_cell

    metadata = {
        'wavelength': ref_header['wavelength'],
        'ub': ref_header['ub'],
        'd_min': ref_header['d_min'],
        'source_folder': folder,
        'bin_xy': bin_xy, 'bin_z': bin_z,
        'n_files': n_files,
        'cell': cell,
        'M_inv': ref_layer.M_inv,
        's': ref_layer.s,
        'cx': ref_layer.cx,
        'cy': ref_layer.cy,
    }

    vol = VolumeData(
        intensity=volume, H=axis_x, K=axis_y, L=l_values,
        plane_type=plane_type, metadata=metadata,
    )

    if bin_z > 1:
        vol = bin_volume(vol, 1, 1, bin_z)
    return vol


# ──────────────────────────────────────────────────────────────────
# Symmetry operations — all 11 Laue groups
# ──────────────────────────────────────────────────────────────────

def _generate_group(generators, max_iter=200):
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


_INV = -np.eye(3, dtype=int)
_C2a = np.diag([1, -1, -1]).astype(int)
_C2b = np.diag([-1, 1, -1]).astype(int)
_C4c = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=int)
_C3_111 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=int)
_C3_hex = np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]], dtype=int)
_C6_hex = np.array([[1, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=int)
_C2p_hex = np.array([[1, -1, 0], [0, -1, 0], [0, 0, -1]], dtype=int)

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

_LAUE_GROUPS_CACHE: Dict[str, List[np.ndarray]] = {}


def get_symmetry_operations(laue_group: str) -> List[np.ndarray]:
    """Return list of 3x3 symmetry operation matrices for a Laue group."""
    if laue_group not in _LAUE_GROUPS_CACHE:
        if laue_group not in _LAUE_GENERATORS:
            raise ValueError(f"Unknown Laue group '{laue_group}'. "
                             f"Valid: {list(_LAUE_GENERATORS.keys())}")
        ops = _generate_group(_LAUE_GENERATORS[laue_group])
        assert len(ops) == _EXPECTED_ORDERS[laue_group]
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


# ──────────────────────────────────────────────────────────────────
# Axis mapping helpers
# ──────────────────────────────────────────────────────────────────

def _get_axis_mapping(plane_type: str) -> dict:
    """Map volume axes (H,K,L) to standard (h,k,l).
    HK: H->h, K->k, L->l | HL: H->h, K->l, L->k | KL: H->k, K->l, L->h
    """
    return {'HK': {'H': 'h', 'K': 'k', 'L': 'l'},
            'HL': {'H': 'h', 'K': 'l', 'L': 'k'},
            'KL': {'H': 'k', 'K': 'l', 'L': 'h'},
            }.get(plane_type, {'H': 'h', 'K': 'k', 'L': 'l'})


def _build_axis_permutation(op, axis_map):
    """For a symmetry operation, determine how it permutes volume axes.

    Returns (src_axes, signs) where for each volume dimension i:
        target_dim_i_value = signs[i] * source_axis[src_axes[i]]

    Only valid for signed-permutation operations (one non-zero per row).
    Returns None if the operation is not a signed permutation.
    """
    # axis_map: e.g. {'H': 'h', 'K': 'k', 'L': 'l'} for HK planes
    vol_to_std = {'H': 0, 'K': 1, 'L': 2}  # volume dim -> std axis index
    std_to_vol = {}
    for vol_ax, std_ax in axis_map.items():
        std_idx = {'h': 0, 'k': 1, 'l': 2}[std_ax]
        vol_dim = {'H': 0, 'K': 1, 'L': 2}[vol_ax]
        std_to_vol[std_idx] = vol_dim
        vol_to_std[vol_ax] = std_idx

    # Convert vol_dim -> std_idx mapping
    dim_to_std = [None, None, None]
    for vol_ax in ['H', 'K', 'L']:
        vol_dim = {'H': 0, 'K': 1, 'L': 2}[vol_ax]
        std_idx = {'h': 0, 'k': 1, 'l': 2}[axis_map[vol_ax]]
        dim_to_std[vol_dim] = std_idx

    # For each target volume dim i:
    # target_std[dim_to_std[i]] = sum(op[dim_to_std[i], j] * source_std[j])
    # We need: which source volume dim provides the value, and with what sign
    src_dims = []
    signs = []
    for i in range(3):  # target volume dim
        tgt_std = dim_to_std[i]  # which standard axis this vol dim represents
        # op row tgt_std: target_std[tgt_std] = sum(op[tgt_std, j] * src_std[j])
        row = op[tgt_std, :]
        nonzero = [(j, int(row[j])) for j in range(3) if row[j] != 0]
        if len(nonzero) != 1:
            return None  # Not a signed permutation
        src_std_idx, sign = nonzero[0]
        # Which volume dim has std axis src_std_idx?
        src_vol_dim = std_to_vol[src_std_idx]
        src_dims.append(src_vol_dim)
        signs.append(sign)

    return src_dims, signs


# ──────────────────────────────────────────────────────────────────
# GPU detection
# ──────────────────────────────────────────────────────────────────

def _has_gpu():
    """Check if CuPy + CUDA GPU is available."""
    try:
        import cupy as cp
        cp.cuda.Device(0).compute_capability
        return True
    except Exception:
        return False


HAS_GPU = _has_gpu()


# ──────────────────────────────────────────────────────────────────
# Precomputed operation maps (shared between sym & reject)
# ──────────────────────────────────────────────────────────────────

def _precompute_op_maps(vol, laue_group, xp=None):
    """Precompute 1D index maps for all operations. Works with numpy or cupy."""
    if xp is None:
        xp = np
    ops = get_symmetry_operations(laue_group)
    axes = [vol.H, vol.K, vol.L]
    steps = [axes[i][1] - axes[i][0] if len(axes[i]) > 1 else 1.0 for i in range(3)]
    origins = [axes[i][0] for i in range(3)]
    sizes = [len(vol.H), len(vol.K), len(vol.L)]
    axis_map = _get_axis_mapping(vol.plane_type)

    op_maps = []
    for op in ops:
        perm = _build_axis_permutation(op, axis_map)
        if perm is None:
            op_maps.append(None)  # hex/trig fallback
            continue

        src_dims, signs = perm
        idx_1d = []
        valid_1d = []
        for i in range(3):
            sd = src_dims[i]
            sgn = signs[i]
            src_idx = np.round(
                (sgn * axes[i] - origins[sd]) / steps[sd]
            ).astype(np.intp)
            v = (src_idx >= 0) & (src_idx < sizes[sd])
            idx_1d.append(xp.asarray(np.clip(src_idx, 0, sizes[sd] - 1)))
            valid_1d.append(xp.asarray(v))

        inv_map = [None, None, None]
        for i in range(3):
            inv_map[src_dims[i]] = i

        gather_idx = [idx_1d[inv_map[d]] for d in range(3)]
        gather_valid = [valid_1d[inv_map[d]] for d in range(3)]

        shapes = [[1, 1, 1] for _ in range(3)]
        for d in range(3):
            shapes[d][inv_map[d]] = sizes[inv_map[d]]

        gi = [gather_idx[d].reshape(shapes[d]) for d in range(3)]
        gv = [gather_valid[d].reshape(shapes[d]) for d in range(3)]
        op_maps.append((gi, gv))

    return op_maps


# ──────────────────────────────────────────────────────────────────
# Core symmetrization loop (works with numpy or cupy arrays)
# ──────────────────────────────────────────────────────────────────

def _symmetrize_core(data, op_maps, vol, xp=None):
    """Run symmetrization using precomputed maps. xp = numpy or cupy.

    Memory-efficient: uses float32 accumulator + int16 count, deletes
    temps immediately.  At bin_xy=2 (1.5 GB volume), total GPU usage is
    ~5.5 GB peak, fitting in 8 GB VRAM.
    """
    if xp is None:
        xp = np
    nh, nk, nl = data.shape

    # float32 sum (not float64) — saves 50% memory.
    # Precision: max sum = 48 * 7M = 336M, float32 mantissa = 24 bits
    # (exact up to 16.7M). Worst case ~1e-4 relative error on bright pixels,
    # well within noise. After dividing by count, result is in original range.
    sym_sum = xp.zeros((nh, nk, nl), dtype=xp.float32)
    sym_count = xp.zeros((nh, nk, nl), dtype=xp.int16)  # max 48 ops

    for om in op_maps:
        if om is None:
            continue
        gi, gv = om

        vals = data[gi[0], gi[1], gi[2]]         # (nh,nk,nl) float32 temp
        mask = gv[0] & gv[1] & gv[2]             # broadcast bool, tiny
        mask = mask & (vals != 0)                 # (nh,nk,nl) bool temp
        vals *= mask                              # in-place zero unmeasured
        sym_sum += vals                           # in-place accumulate
        sym_count += mask                         # bool -> int16 in-place
        del vals, mask                            # free immediately

    with np.errstate(invalid='ignore', divide='ignore'):
        count_f = sym_count.astype(xp.float32)
        result = xp.where(count_f > 0, sym_sum / count_f, 0)
    del sym_sum, sym_count, count_f
    return result


# ──────────────────────────────────────────────────────────────────
# FAST symmetrization (auto GPU/CPU)
# ──────────────────────────────────────────────────────────────────

def symmetrize_volume(vol: VolumeData, laue_group: str,
                      progress_callback: Optional[Callable] = None,
                      use_gpu: Optional[bool] = None) -> VolumeData:
    """Symmetrize the 3D volume by averaging over all Laue group operations.

    Uses fast 1D index-map + broadcasting for signed-permutation ops.
    Automatically uses GPU (CuPy) if available, unless use_gpu=False.
    Zero voxels (unmeasured) are excluded from averaging.
    """
    if use_gpu is None:
        use_gpu = HAS_GPU

    if use_gpu:
        import cupy as cp
        xp = cp
        data = cp.asarray(np.nan_to_num(vol.intensity.astype(np.float32), nan=0.0))
        op_maps = _precompute_op_maps(vol, laue_group, xp=cp)
    else:
        xp = np
        data = np.nan_to_num(vol.intensity.astype(np.float32), nan=0.0)
        op_maps = _precompute_op_maps(vol, laue_group, xp=np)

    if progress_callback:
        progress_callback(0, 1)

    result = _symmetrize_core(data, op_maps, vol, xp=xp)

    if use_gpu:
        result = cp.asnumpy(result)
        del data, op_maps
        cp.get_default_memory_pool().free_all_blocks()

    if progress_callback:
        progress_callback(1, 1)

    return VolumeData(
        intensity=result,
        H=vol.H.copy(), K=vol.K.copy(), L=vol.L.copy(),
        plane_type=vol.plane_type,
        metadata={**vol.metadata, 'laue_group': laue_group},
    )


def _to_hkl(H_vals, K_vals, L_vals, axis_map):
    """Convert volume coordinates to standard (h, k, l)."""
    result = [None, None, None]
    for vol_ax, std_ax in axis_map.items():
        idx = {'h': 0, 'k': 1, 'l': 2}[std_ax]
        result[idx] = {'H': H_vals, 'K': K_vals, 'L': L_vals}[vol_ax]
    return result


def _from_hkl(h, k, l, axis_map):
    """Convert standard (h, k, l) back to volume coordinates."""
    std_vals = {'h': h, 'k': k, 'l': l}
    return (std_vals[axis_map['H']],
            std_vals[axis_map['K']],
            std_vals[axis_map['L']])


# ──────────────────────────────────────────────────────────────────
# Outlier rejection
# ──────────────────────────────────────────────────────────────────

def compute_outlier_stats(vol: VolumeData, laue_group: str) -> dict:
    """Compute statistics to help choose sigma threshold.

    Returns dict with percentiles and suggested sigma values.
    """
    vol_sym = symmetrize_volume(vol, laue_group)
    residual = vol.intensity.astype(np.float32) - vol_sym.intensity

    finite = np.isfinite(residual) & (residual != 0)
    r = residual[finite]
    med = np.median(r)
    mad = np.median(np.abs(r - med))
    robust_std = 1.4826 * mad

    abs_r = np.abs(r)
    return {
        'median_residual': float(med),
        'mad': float(mad),
        'robust_std': float(robust_std),
        'percentile_90': float(np.percentile(abs_r, 90)),
        'percentile_95': float(np.percentile(abs_r, 95)),
        'percentile_99': float(np.percentile(abs_r, 99)),
        'percentile_999': float(np.percentile(abs_r, 99.9)),
        'max_abs_residual': float(abs_r.max()),
        'n_above_2sigma': int((abs_r > 2 * robust_std).sum()),
        'n_above_3sigma': int((abs_r > 3 * robust_std).sum()),
        'n_above_5sigma': int((abs_r > 5 * robust_std).sum()),
        'n_total': int(finite.sum()),
        'fraction_above_3sigma': float((abs_r > 3 * robust_std).sum() / len(r)),
    }


def reject_outliers(vol: VolumeData, laue_group: str,
                    sigma: float = 3.0, n_iter: int = 2,
                    progress_callback: Optional[Callable] = None,
                    use_gpu: Optional[bool] = None) -> VolumeData:
    """Reject outlier voxels by comparing to symmetrized volume.

    Approach: symmetrize, compute vectorized MAD per L-slice,
    replace voxels deviating > sigma*robust_std with symmetrized value.
    Automatically uses GPU if available.
    """
    if use_gpu is None:
        use_gpu = HAS_GPU

    if use_gpu:
        import cupy as cp
        xp = cp
        data = cp.asarray(vol.intensity.astype(np.float32))
        op_maps = _precompute_op_maps(vol, laue_group, xp=cp)
    else:
        xp = np
        data = vol.intensity.astype(np.float32).copy()
        op_maps = _precompute_op_maps(vol, laue_group, xp=np)

    n_replaced_total = 0

    for iteration in range(n_iter):
        if progress_callback:
            progress_callback(iteration, n_iter)

        sym_data = _symmetrize_core(data, op_maps, vol, xp=xp)

        # MAD-based outlier detection per L-slice
        # Use a loop to avoid creating multiple volume-sized temps
        nl = data.shape[2]
        n_replaced = 0
        for il in range(nl):
            sl_res = data[:, :, il] - sym_data[:, :, il]
            nz = sl_res != 0
            nz_count = int(nz.sum())
            if nz_count < 10:
                continue
            vals_nz = sl_res[nz]
            med = float(xp.median(vals_nz))
            mad = float(xp.median(xp.abs(vals_nz - med)))
            robust_std = 1.4826 * mad
            if robust_std < 1e-10:
                continue
            outlier = xp.abs(sl_res) > sigma * robust_std
            n_out = int(outlier.sum())
            n_replaced += n_out
            if n_out > 0:
                data[:, :, il] = xp.where(outlier, sym_data[:, :, il], data[:, :, il])
        del sym_data
        n_replaced_total += n_replaced

    if progress_callback:
        progress_callback(n_iter, n_iter)

    if use_gpu:
        data = cp.asnumpy(data)
        del op_maps
        cp.get_default_memory_pool().free_all_blocks()

    return VolumeData(
        intensity=data,
        H=vol.H.copy(), K=vol.K.copy(), L=vol.L.copy(),
        plane_type=vol.plane_type,
        metadata={**vol.metadata, 'n_outliers_replaced': n_replaced_total},
    )


# ──────────────────────────────────────────────────────────────────
# Volume slice extraction (shared by all viewers)
# ──────────────────────────────────────────────────────────────────

def extract_volume_slice(vol, plane_index, target_val, int_range=0.0):
    """Extract a 2D slice from a 3D volume at a constant Miller index.

    Handles non-orthogonal grids (monoclinic) by interpolating at the
    correct constant-Miller-index surface, then regridding onto the
    view plane's Cartesian pixel grid.

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
    """
    # Axis configuration
    cfgs = [
        (vol.H, vol.K, vol.L, 'h', 'k', 'l', 0, 1, 2),  # HK fix L
        (vol.H, vol.L, vol.K, 'h', 'l', 'k', 0, 2, 1),  # HL fix K
        (vol.K, vol.L, vol.H, 'k', 'l', 'h', 1, 2, 0),  # KL fix H
    ]
    x_ax, y_ax, fixed_ax = cfgs[plane_index][0], cfgs[plane_index][1], cfgs[plane_index][2]
    x_label, y_label, fixed_label = cfgs[plane_index][3], cfgs[plane_index][4], cfgs[plane_index][5]
    vol_dim_fixed = cfgs[plane_index][8]

    plane_types = ['HK', 'HL', 'KL']
    is_native = plane_types[plane_index] == vol.plane_type

    if is_native:
        sl, actual_val, n_slices = _extract_native(
            vol.intensity, fixed_ax, target_val, int_range)
    else:
        sl, actual_val, n_slices, x_ax = _extract_nonnat(
            vol, x_ax, y_ax, fixed_ax, vol_dim_fixed,
            target_val, int_range, plane_types[plane_index])

    return sl, x_ax, y_ax, x_label, y_label, fixed_label, actual_val, n_slices


def _extract_native(data, fixed_ax, target_val, int_range):
    """Native plane (HK): direct pixel extraction."""
    if int_range < 1e-6:
        idx = int(np.argmin(np.abs(fixed_ax - target_val)))
        actual_val = float(fixed_ax[idx])
        slab = data[:, :, idx]
        n_slices = 1
    else:
        lo, hi = target_val - int_range, target_val + int_range
        indices = np.where((fixed_ax >= lo) & (fixed_ax <= hi))[0]
        if len(indices) == 0:
            indices = [int(np.argmin(np.abs(fixed_ax - target_val)))]
        actual_val = float(fixed_ax[indices[len(indices) // 2]])
        n_slices = len(indices)
        slab = data[:, :, indices].astype(np.float64).sum(axis=2)

    return slab.T.astype(np.float32), actual_val, n_slices


def _extract_nonnat(vol, x_ax, y_ax, fixed_ax, vol_dim_fixed,
                    target_val, int_range, view_plane):
    """Non-native plane with cross-term correction and regridding."""
    data = vol.intensity.astype(np.float32)
    H, K, L = vol.H, vol.K, vol.L
    nh, nk, nl = data.shape
    dh = H[1] - H[0] if nh > 1 else 1.0
    dk = K[1] - K[0] if nk > 1 else 1.0

    # HK M_inv for cross-term correction
    M_inv_HK = vol.metadata.get('M_inv')
    if M_inv_HK is None:
        ub = vol.metadata.get('ub')
        wl = vol.metadata.get('wavelength', 1.0)
        if ub is not None:
            M_inv_HK = compute_plane_M_inv(ub, wl, vol.plane_type)
        else:
            M_inv_HK = np.eye(2)

    s = vol.metadata.get('s', abs(dh / M_inv_HK[0, 0]))
    cx = vol.metadata.get('cx', (nh + 1) / 2.0)
    cy = vol.metadata.get('cy', (nk + 1) / 2.0)
    h_cross = M_inv_HK[0, 1] * s

    # Integration targets
    if int_range < 1e-6:
        idx = int(np.argmin(np.abs(fixed_ax - target_val)))
        actual_val = float(fixed_ax[idx])
        target_vals = [target_val]
        n_slices = 1
    else:
        lo, hi = target_val - int_range, target_val + int_range
        indices = np.where((fixed_ax >= lo) & (fixed_ax <= hi))[0]
        if len(indices) == 0:
            indices = [int(np.argmin(np.abs(fixed_ax - target_val)))]
        actual_val = float(fixed_ax[indices[len(indices) // 2]])
        target_vals = [fixed_ax[i] for i in indices]
        n_slices = len(target_vals)

    if vol_dim_fixed == 0:
        # KL: fixed H — per-row ih correction
        IK, IL = np.meshgrid(np.arange(nk), np.arange(nl), indexing='ij')
        slab = np.zeros((nk, nl), dtype=np.float64)
        for tv in target_vals:
            ik_offsets = np.arange(nk) + 1 - cy
            ih_frac = (tv - h_cross * ik_offsets - H[0]) / dh
            IH_frac = np.broadcast_to(ih_frac[:, None], (nk, nl))
            slab += map_coordinates(data,
                [IH_frac.ravel(), IK.ravel().astype(float), IL.ravel().astype(float)],
                order=1, mode='constant', cval=0.0).reshape(nk, nl)

    elif vol_dim_fixed == 1:
        # HL: fixed K — interpolate ik, correct h-offset
        IH, IL = np.meshgrid(np.arange(nh), np.arange(nl), indexing='ij')
        slab = np.zeros((nh, nl), dtype=np.float64)
        for tv in target_vals:
            ik_frac = (tv - K[0]) / dk
            IK_frac = np.full((nh, nl), ik_frac, dtype=np.float64)
            slab += map_coordinates(data,
                [IH.ravel().astype(float), IK_frac.ravel(), IL.ravel().astype(float)],
                order=1, mode='constant', cval=0.0).reshape(nh, nl)

        avg_k = np.mean(target_vals)
        ik_avg = (avg_k - K[0]) / dk
        h_offset = h_cross * (ik_avg + 1 - cy)
        x_ax = x_ax + h_offset

    else:
        idx = int(np.argmin(np.abs(fixed_ax - target_val)))
        slab = data[:, :, idx].T
        actual_val = float(fixed_ax[idx])
        return slab.astype(np.float32), actual_val, 1, x_ax

    # Regrid to Cartesian
    ub = vol.metadata.get('ub')
    wl = vol.metadata.get('wavelength', 1.0)
    if ub is not None:
        M_inv_view = compute_plane_M_inv(ub, wl, view_plane)
    else:
        M_inv_view = M_inv_HK

    out_n = len(vol.H)
    cx_out = (out_n + 1) / 2.0
    ii = np.arange(1, out_n + 1) - cx_out
    jj = np.arange(1, out_n + 1) - cx_out
    II, JJ = np.meshgrid(ii, jj)
    # Use the stored s directly — bin_volume already updates it correctly
    s_out = vol.metadata.get('s', s)

    idx1_out = M_inv_view[0, 0] * s_out * II + M_inv_view[0, 1] * s_out * JJ
    idx2_out = M_inv_view[1, 0] * s_out * II + M_inv_view[1, 1] * s_out * JJ

    dx = x_ax[1] - x_ax[0] if len(x_ax) > 1 else 1.0
    dy = y_ax[1] - y_ax[0] if len(y_ax) > 1 else 1.0
    fi = (idx1_out - x_ax[0]) / dx
    fj = (idx2_out - y_ax[0]) / dy

    regridded = map_coordinates(
        slab.astype(np.float32), [fi.ravel(), fj.ravel()],
        order=1, mode='constant', cval=0.0
    ).reshape(out_n, out_n).astype(np.float32)

    return regridded, actual_val, n_slices, x_ax


# ──────────────────────────────────────────────────────────────────
# Save / load — HDF5 (MATLAB-compatible) and npz
# ──────────────────────────────────────────────────────────────────

def save_volume_h5(path: str, vol: VolumeData, compression: str = 'gzip',
                   compression_level: int = 4):
    """Save volume as HDF5 file, MATLAB-compatible.

    Datasets: /data (3D intensity), /H, /K, /L (1D axes),
    plus metadata attributes.  Uses gzip compression by default.

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
        f.create_dataset('data', data=vol.intensity, **comp_opts)
        f.create_dataset('H', data=vol.H, **comp_opts)
        f.create_dataset('K', data=vol.K, **comp_opts)
        f.create_dataset('L', data=vol.L, **comp_opts)

        # Metadata as attributes
        f.attrs['plane_type'] = vol.plane_type
        f.attrs['wavelength'] = vol.metadata.get('wavelength', 0.0)
        if 'laue_group' in vol.metadata:
            f.attrs['laue_group'] = vol.metadata['laue_group']
        f.attrs['bin_xy'] = vol.metadata.get('bin_xy', 1)
        f.attrs['bin_z'] = vol.metadata.get('bin_z', 1)
        if 'source_folder' in vol.metadata:
            f.attrs['source_folder'] = vol.metadata['source_folder']

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


def load_volume_h5(path: str) -> VolumeData:
    """Load a volume from an HDF5 file."""
    import h5py
    with h5py.File(path, 'r') as f:
        intensity = np.array(f['data'])
        H = np.array(f['H'])
        K = np.array(f['K'])
        L = np.array(f['L'])
        metadata = {
            'wavelength': float(f.attrs.get('wavelength', 0)),
            'laue_group': str(f.attrs.get('laue_group', '')),
            'source_folder': str(f.attrs.get('source_folder', '')),
            'bin_xy': int(f.attrs.get('bin_xy', 1)),
            'bin_z': int(f.attrs.get('bin_z', 1)),
        }
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
        plane_type = str(f.attrs.get('plane_type', 'HK'))
    return VolumeData(
        intensity=intensity, H=H, K=K, L=L,
        plane_type=plane_type, metadata=metadata,
    )


def save_volume_npz(path: str, vol: VolumeData):
    """Save volume as compressed .npz file."""
    np.savez_compressed(
        path, intensity=vol.intensity,
        H=vol.H, K=vol.K, L=vol.L,
        plane_type=np.array(vol.plane_type),
        wavelength=vol.metadata.get('wavelength', 0),
        laue_group=np.array(vol.metadata.get('laue_group', '')),
        source_folder=np.array(vol.metadata.get('source_folder', '')),
        bin_xy=vol.metadata.get('bin_xy', 1),
        bin_z=vol.metadata.get('bin_z', 1),
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
