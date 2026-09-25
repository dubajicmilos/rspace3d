"""rspace3d.rawrecon - raw-CBF reciprocal-space reconstruction.

Isolated subpackage: given a CrysAlisPro rotation dataset (raw Eiger .cbf frames +
a *_cracker.par), map every detector pixel to fractional (h,k,l) and accumulate a
3D reciprocal-space volume, GPU-accelerated (CuPy) with a transparent CPU fallback.
Optional per-pixel solid-angle + polarisation corrections. The result converts to
an rspace3d VolumeData (`reconstruct_to_volumedata`) so the existing
bin/symmetrize/save/view machinery applies unchanged.

Unmeasured voxels are NaN (exact, from the per-voxel pixel count) - no coverage-
mask heuristic needed, unlike the .img/unwarp path.
"""
from .geometry import (FlatDetector, find_crysalis_par, find_cbf_frames,
                       read_crysalis_par, orient_from_frame, index_frame, detect_peaks)
from .corrections import pixel_corrections, pixel_directions
from .engine import (Volume, reconstruct_volume, reconstruct_volume_gpu,
                     reconstruct_dataset, reconstruct_to_volumedata,
                     bragg_registration_rms, has_cupy)
from .tabbin import read_tabbin, find_tabbin
from .calibrate import geometry_from_tabbin, FittedGeometry

__all__ = [
    "FlatDetector", "find_crysalis_par", "find_cbf_frames", "read_crysalis_par",
    "orient_from_frame", "index_frame", "detect_peaks",
    "pixel_corrections", "pixel_directions",
    "Volume", "reconstruct_volume", "reconstruct_volume_gpu", "reconstruct_dataset",
    "reconstruct_to_volumedata", "bragg_registration_rms", "has_cupy",
    "read_tabbin", "find_tabbin", "geometry_from_tabbin", "FittedGeometry",
]
