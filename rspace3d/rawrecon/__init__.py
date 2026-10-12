"""rspace3d.rawrecon - raw-CBF reciprocal-space reconstruction.

Isolated subpackage: given a CrysAlisPro rotation dataset (raw Eiger .cbf frames +
a *_cracker.par), map every detector pixel to fractional (h,k,l) and accumulate a
3D reciprocal-space volume, GPU-accelerated (CuPy) with a transparent CPU fallback.
Optional per-pixel solid-angle + polarisation corrections. The result converts to
an rspace3d VolumeData (`reconstruct_to_volumedata`) so the existing
bin/symmetrize/save/view machinery applies unchanged.

Unmeasured voxels are NaN (exact, from the per-voxel pixel count) - no coverage-
mask heuristic needed, unlike the .img/unwarp path.

Frame tools (`frames`, `rings`, `frame_model`): odd and even half volumes in
one pass, a powder-ring detector and mask, a per-frame model of additive and
scale contamination, and the chain that applies them (`run_frame_chain`).
"""
from .geometry import (FlatDetector, find_crysalis_par, find_cbf_frames,
                       read_crysalis_par, orient_from_frame, index_frame, detect_peaks)
from .corrections import pixel_corrections, pixel_directions
from .engine import (Volume, reconstruct_volume, reconstruct_volume_gpu,
                     reconstruct_dataset, reconstruct_to_volumedata,
                     bragg_registration_rms, has_cupy, ReconSetup, prepare_setup,
                     gpu_memory_bytes)
from .tabbin import read_tabbin, find_tabbin
from .calibrate import geometry_from_tabbin, FittedGeometry
from .rings import (PixelPolar, RingStats, RingDetection, RingMask, ring_statistics,
                    detect_rings)
from .frame_model import FrameModel, ModelOptions, fit_frame_model
from .frames import (FrameCells, FrameStats, FrameMap, FrameChainResult, predict_frame,
                     leave_one_out_median, frame_statistics, frame_map, frame_map_from_stats,
                     frame_function, corrected_reconstruction, run_frame_chain)

__all__ = [
    "FlatDetector", "find_crysalis_par", "find_cbf_frames", "read_crysalis_par",
    "orient_from_frame", "index_frame", "detect_peaks",
    "pixel_corrections", "pixel_directions",
    "Volume", "reconstruct_volume", "reconstruct_volume_gpu", "reconstruct_dataset",
    "reconstruct_to_volumedata", "bragg_registration_rms", "has_cupy",
    "ReconSetup", "prepare_setup", "gpu_memory_bytes",
    "read_tabbin", "find_tabbin", "geometry_from_tabbin", "FittedGeometry",
    "PixelPolar", "RingStats", "RingDetection", "RingMask", "ring_statistics", "detect_rings",
    "FrameModel", "ModelOptions", "fit_frame_model",
    "FrameCells", "FrameStats", "FrameMap", "FrameChainResult", "predict_frame",
    "leave_one_out_median", "frame_statistics", "frame_map", "frame_map_from_stats",
    "frame_function", "corrected_reconstruction", "run_frame_chain",
]
