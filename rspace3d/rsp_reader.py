"""
Generalized reader for CrysAlisPro reciprocal space .img files.

Auto-detects cross-section type (HK, HL, KL) from header flags and
computes Miller index grids for any unit cell.

Usage:
    from rsp_reader import read_rsp_layer
    layer = read_rsp_layer("FAPbBr3_230K_H1.5L.img")
    print(layer.plane_type)    # 'HL'
    print(layer.fixed_label)   # 'k'
    print(layer.fixed_value)   # 1.5
    plt.pcolormesh(layer.idx1, layer.idx2, layer.intensity)
"""

import numpy as np
import struct
import io
from dataclasses import dataclass
from fabio.OXDimage import OxdImage


@dataclass
class RSPLayer:
    """Result of reading a reciprocal space layer .img file."""
    intensity: np.ndarray   # (NY, NX) intensity array
    idx1: np.ndarray        # (NY, NX) first free Miller index (x-axis)
    idx2: np.ndarray        # (NY, NX) second free Miller index (y-axis)
    fixed_value: float      # value of the fixed Miller index
    plane_type: str         # 'HK', 'HL', or 'KL'
    x_label: str            # label for x-axis, e.g. 'h'
    y_label: str            # label for y-axis, e.g. 'k'
    fixed_label: str        # label for fixed axis, e.g. 'l'
    thickness: tuple        # (min, max) integration range of fixed index
    M_inv: np.ndarray       # (2,2) pixel-to-Miller-index matrix (full, with cross-terms)
    s: float                # Cartesian step (1/Angstrom per pixel)
    cx: float               # grid center x (1-based pixel)
    cy: float               # grid center y (1-based pixel)
    step_idx1: float        # display step: r.l.u. per pixel along x (no cross-term)
    step_idx2: float        # display step: r.l.u. per pixel along y (no cross-term)


# Header offsets for axis flags and fixed values
_FIXED_OFFSETS = {'h': 864, 'k': 872, 'l': 880}
_THICKNESS_MIN = {'h': 960, 'k': 968, 'l': 976}
_THICKNESS_MAX = {'h': 992, 'k': 1000, 'l': 1008}
_XAXIS_FLAGS = {'h': 896, 'k': 904}       # offset where =1.0 means this index is x-axis
_YAXIS_FLAGS = {'k': 936, 'l': 944}       # offset where =1.0 means this index is y-axis

# Plane configurations: which UB columns map to which axis
_PLANE_CONFIG = {
    'HK': {'vec1_col': 0, 'vec2_col': 1, 'fixed_col': 2,
            'x_label': 'h', 'y_label': 'k', 'fixed_label': 'l'},
    'HL': {'vec1_col': 0, 'vec2_col': 2, 'fixed_col': 1,
            'x_label': 'h', 'y_label': 'l', 'fixed_label': 'k'},
    'KL': {'vec1_col': 1, 'vec2_col': 2, 'fixed_col': 0,
            'x_label': 'k', 'y_label': 'l', 'fixed_label': 'h'},
}


def read_rsp_layer(filename, UB_override=None):
    """Read any CrysAlisPro reciprocal space .img file.

    Auto-detects cross-section type (HK, HL, KL) from header flags.

    Parameters
    ----------
    filename : str
        Path to a CrysAlisPro .img file.
    UB_override : ndarray (3,3), optional
        Use this UB matrix instead of the one in the .img header.

    Returns
    -------
    RSPLayer
        Dataclass with intensity, Miller index grids, and metadata.
    """
    with open(filename, 'rb') as f:
        raw = f.read()
    header = raw[:5120]

    # --- Detect plane type ---
    plane_type = _detect_plane_type(header)
    cfg = _PLANE_CONFIG[plane_type]

    # --- Read header parameters ---
    fixed_label = cfg['fixed_label']
    fixed_value = struct.unpack_from('<d', header, _FIXED_OFFSETS[fixed_label])[0]
    thickness_min = struct.unpack_from('<d', header, _THICKNESS_MIN[fixed_label])[0]
    thickness_max = struct.unpack_from('<d', header, _THICKNESS_MAX[fixed_label])[0]

    wavelength = struct.unpack_from('<d', header, 2104)[0]
    d_min = struct.unpack_from('<d', header, 1024)[0]

    # Image dimensions: use offsets 278/280 (uint16), works for all plane types
    NX = struct.unpack_from('<H', header, 278)[0]
    NY = struct.unpack_from('<H', header, 280)[0]

    # UB matrix
    if UB_override is not None:
        UB = UB_override
    else:
        UB = np.array([
            struct.unpack_from('<d', header, 2256 + i * 8)[0] for i in range(9)
        ]).reshape(3, 3)

    # --- Compute Miller index grids ---
    idx1, idx2, M_inv, s, cx, cy, step_idx1, step_idx2 = _compute_index_grid(
        UB, wavelength, d_min, NX, NY,
        cfg['vec1_col'], cfg['vec2_col'], cfg['fixed_col']
    )

    # --- Read intensity ---
    buf = io.BytesIO(raw)
    img = OxdImage()
    img.read(buf)

    return RSPLayer(
        intensity=img.data,
        idx1=idx1,
        idx2=idx2,
        fixed_value=fixed_value,
        plane_type=plane_type,
        x_label=cfg['x_label'],
        y_label=cfg['y_label'],
        fixed_label=fixed_label,
        thickness=(thickness_min, thickness_max),
        M_inv=M_inv,
        s=s,
        cx=cx,
        cy=cy,
        step_idx1=step_idx1,
        step_idx2=step_idx2,
    )


def _detect_plane_type(header):
    """Detect cross-section type from header axis flags.

    Checks which axis flag offsets are set to 1.0:
        offset 896 = 1.0 → h is x-axis
        offset 904 = 1.0 → k is x-axis
        offset 936 = 1.0 → k is y-axis
        offset 944 = 1.0 → l is y-axis

    Returns 'HK', 'HL', or 'KL'.
    """
    h_is_x = abs(struct.unpack_from('<d', header, 896)[0] - 1.0) < 0.01
    k_is_x = abs(struct.unpack_from('<d', header, 904)[0] - 1.0) < 0.01
    k_is_y = abs(struct.unpack_from('<d', header, 936)[0] - 1.0) < 0.01
    l_is_y = abs(struct.unpack_from('<d', header, 944)[0] - 1.0) < 0.01

    if h_is_x and k_is_y:
        return 'HK'
    elif h_is_x and l_is_y:
        return 'HL'
    elif k_is_x and l_is_y:
        return 'KL'
    else:
        # Fallback: check which fixed-value offset is non-zero
        for label, plane in [('l', 'HK'), ('k', 'HL'), ('h', 'KL')]:
            val = struct.unpack_from('<d', header, _FIXED_OFFSETS[label])[0]
            if abs(val) > 1e-10:
                return plane
        return 'HK'  # default


def _compute_index_grid(UB, wavelength, d_min, NX, NY,
                        vec1_col, vec2_col, fixed_col):
    """Compute Miller index grids for two free axes.

    Builds a 2D orthonormal basis in the plane of the two free
    reciprocal vectors (v1, v2), then constructs the 2x2 transformation
    matrix M that maps Miller indices to Cartesian pixel coordinates.

    CrysAlisPro uses the raw reciprocal vectors (NOT projected
    perpendicular to the fixed axis) to define the image plane.
    This matters for monoclinic/triclinic cells where the fixed
    axis is not perpendicular to the free axes.

    Works for any unit cell including triclinic (handles shear).
    """
    # Reciprocal lattice vectors (standard 1/Angstrom)
    recip = UB / wavelength
    v1 = recip[:, vec1_col]
    v2 = recip[:, vec2_col]

    # Orthonormal basis in the v1-v2 plane (no projection onto fixed axis)
    e_x = v1 / np.linalg.norm(v1)
    v2_perp = v2 - np.dot(v2, e_x) * e_x
    e_y = v2_perp / np.linalg.norm(v2_perp)

    # 2x2 transformation matrix: (x, y) = M @ (idx1, idx2)
    M = np.array([
        [np.dot(v1, e_x), np.dot(v2, e_x)],
        [np.dot(v1, e_y), np.dot(v2, e_y)]
    ])
    M_inv = np.linalg.inv(M)

    # Cartesian step
    s = 2.0 / (d_min * NX)

    # Pixel center: works for both odd and even NX/NY
    cx = (NX + 1) // 2 + 0.5
    cy = (NY + 1) // 2 + 0.5
    ii = np.arange(1, NX + 1) - cx
    jj = np.arange(1, NY + 1) - cy
    I, J = np.meshgrid(ii, jj)

    # Miller index grids using full M_inv (correct h,k,l with cross-terms)
    idx1 = M_inv[0, 0] * I * s + M_inv[0, 1] * J * s
    idx2 = M_inv[1, 0] * I * s + M_inv[1, 1] * J * s

    # Diagonal step sizes for display
    step_idx1 = s / np.linalg.norm(v1)
    step_idx2 = s / abs(np.dot(v2, e_y))

    return idx1, idx2, M_inv, s, cx, cy, step_idx1, step_idx2


def read_par_UB(par_filename):
    """Read the UB matrix from a CrysAlisPro .par file."""
    with open(par_filename, 'r', errors='replace') as f:
        for line in f:
            if line.startswith('CRYSTALLOGRAPHY UB '):
                vals = [float(x) for x in line.split()[2:11]]
                return np.array(vals).reshape(3, 3)
    raise ValueError(f"No CRYSTALLOGRAPHY UB line found in {par_filename}")


if __name__ == '__main__':
    import sys
    import glob

    files = sys.argv[1:] if len(sys.argv) > 1 else glob.glob('*.img')
    for fname in files:
        layer = read_rsp_layer(fname)
        print(f"{fname}:")
        print(f"  Plane: {layer.plane_type}, {layer.fixed_label} = {layer.fixed_value:.4f}"
              f" [{layer.thickness[0]:.4f}, {layer.thickness[1]:.4f}]")
        print(f"  Shape: {layer.intensity.shape}")
        print(f"  {layer.x_label}: [{layer.idx1.min():.4f}, {layer.idx1.max():.4f}]")
        print(f"  {layer.y_label}: [{layer.idx2.min():.4f}, {layer.idx2.max():.4f}]")
        print()
