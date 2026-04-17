"""
rspace3d — Reciprocal Space 3D Processing Toolkit

Process single-crystal X-ray diffuse scattering data from CrysAlisPro
reciprocal space reconstructions into symmetrized 3D volumes with
GPU acceleration.

Works with any unit cell (cubic to triclinic).
"""

__version__ = '0.1.0'

# Rigaku Oxford detector type 8 ships with newer CrysAlisPro experiments but is
# missing from fabio < 2026-04. Patch in place so .img reads don't KeyError.
import fabio.OXDimage as _oxd  # noqa: E402
_oxd.DETECTOR_TYPES.setdefault(8, "Rigaku Oxford detector type 8")
del _oxd

from .rsp_reader import read_rsp_layer, RSPLayer
from .volume_builder import (
    VolumeData,
    load_unwarp_folder,
    bin_volume,
    symmetrize_volume,
    reject_outliers,
    save_volume_h5,
    load_volume_h5,
    extract_volume_slice,
    compute_plane_M_inv,
    cell_from_ub,
    get_symmetry_operations,
    LAUE_GROUP_NAMES,
)
from .make_dcunwarp import generate_dcunwarp
