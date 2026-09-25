"""Raw-CBF -> reciprocal-space reconstruction engine (GPU via CuPy, CPU fallback).

Every detector pixel is mapped to a fractional (h,k,l) and its intensity
accumulated into a voxel grid over all rotation frames:

    hkl = (R_n . UB)^-1 . r_lab ,   R_n = R_osc(sense*(phi_n - phi0)) . R0 ,

with r_lab fixed by the detector and R0 the reference-frame orientation. Each
voxel stores the mean of the contributing pixels (sum / count); a voxel with zero
contributing pixels is UNMEASURED and set to NaN. Because the pixel `counts` are
known exactly, measured-but-zero voxels (0.0) and unmeasured voxels (NaN) are
unambiguous - no coverage-mask heuristic is needed (unlike the .img/unwarp path).

The grid may be anisotropic: independent (min,max) per axis, common voxel size dq.
"""
from __future__ import annotations

import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from .geometry import (FlatDetector, find_crysalis_par, find_cbf_frames,
                       orient_from_frame, _axis_rot)
from .corrections import pixel_corrections


@dataclass
class Volume:
    data: np.ndarray        # (nh,nk,nl) mean intensity, NaN where unmeasured
    H: np.ndarray           # (nh,) axis
    K: np.ndarray           # (nk,)
    L: np.ndarray           # (nl,)
    UB: np.ndarray          # columns a*,b*,c* in 1/d (CrysAlis UB / lambda)
    counts: np.ndarray      # (nh,nk,nl) contributing pixels per voxel
    wavelength: float


# ---------------------------------------------------------------- grid / cell
def _axes(ranges, step):
    """(nh,nk,nl), (H,K,L), (lo_h,lo_k,lo_l) for per-axis (lo,hi) ranges + dq."""
    los, ns, axes = [], [], []
    for lo, hi in ranges:
        n = int(round((hi - lo) / step))
        los.append(float(lo)); ns.append(n)
        axes.append(lo + (np.arange(n) + 0.5) * step)
    return tuple(ns), tuple(axes), tuple(los)


def _reciprocal_cell(recip):
    """Direct cell dict from reciprocal vectors (columns a*,b*,c* in 1/d)."""
    Gstar = recip.T @ recip
    G = np.linalg.inv(Gstar)
    a, b, c = np.sqrt(np.diag(G))

    def ang(i, j, x, y):
        return float(np.degrees(np.arccos(np.clip(G[i, j] / (x * y), -1, 1))))

    return dict(a=float(a), b=float(b), c=float(c),
                alpha=ang(1, 2, b, c), beta=ang(0, 2, a, c), gamma=ang(0, 1, a, b))


# ---------------------------------------------------------------- prefetch / corr
def _prefetch(paths, read_fn, ahead):
    """Yield read_fn(path) in order, keeping up to `ahead` reads in flight."""
    if ahead <= 0:
        for p in paths:
            yield read_fn(p)
        return
    with ThreadPoolExecutor(max_workers=min(ahead, 8)) as ex:
        q = deque()
        i = 0
        for _ in range(min(ahead, len(paths))):
            q.append(ex.submit(read_fn, paths[i])); i += 1
        while q:
            img = q.popleft().result()
            if i < len(paths):
                q.append(ex.submit(read_fn, paths[i])); i += 1
            yield img


def _resolve_corrections(corrections, detector, r_lab):
    if corrections is None or corrections is False:
        return None
    if corrections is True:
        return pixel_corrections(detector, r_lab)
    if isinstance(corrections, dict):
        return pixel_corrections(detector, r_lab, **corrections)
    arr = np.asarray(corrections, np.float64).ravel()
    if arr.shape[0] != r_lab.shape[0]:
        raise ValueError(f"corrections length {arr.shape[0]} != n_pixels {r_lab.shape[0]}")
    return arr


def _default_read_fn():
    import fabio

    def read_fn(p):
        return fabio.open(p).data
    return read_fn


def _validate_series(frames, phis):
    """Frames and angles must pair up one-to-one (no silent zip truncation)."""
    frames = list(frames)
    phis = [float(p) for p in phis]
    if not frames:
        raise ValueError("at least one frame is required")
    if len(frames) != len(phis):
        raise ValueError(f"frames length {len(frames)} != phis length {len(phis)}")
    if not np.all(np.isfinite(phis)):
        raise ValueError("all frame angles must be finite")
    return frames, phis


def has_cupy(device=0):
    """True if CuPy + a usable CUDA device are available."""
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > device
    except Exception:
        return False


# ---------------------------------------------------------------- GPU engine
def reconstruct_volume_gpu(frames, phis, UB, R0, detector, *, phi0,
                           osc_axis=(0, 1, 0), sense=1,
                           ranges=((-6.0, 6.0), (-6.0, 6.0), (-6.0, 6.0)), step=0.025,
                           read_fn=None, hot=None, progress_callback=None,
                           corrections=None, device=0, prefetch=3) -> Volume:
    """GPU reconstruction. Per-frame float64 scatter-add into a resident
    accumulator; disk reads prefetched on a thread pool. Returns a host Volume."""
    import cupy as cp
    import cupyx

    read_fn = read_fn or _default_read_fn()
    cp.cuda.Device(device).use()
    frames, phis = _validate_series(frames, phis)

    ny, nx = detector.shape
    ys, xs = np.mgrid[0:ny, 0:nx]
    px = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float32)
    r_lab = detector.scattering_vectors(px).astype(np.float32)

    (nh, nk, nl), (H, K, L), (lo_h, lo_k, lo_l) = _axes(ranges, step)
    nkl = nk * nl
    nvox = nh * nkl
    corr = _resolve_corrections(corrections, detector, r_lab)
    r_lab_g = cp.asarray(r_lab)
    corr_g = None if corr is None else cp.asarray(corr)
    lo_g = cp.asarray([lo_h, lo_k, lo_l], cp.float32)

    ssum = cp.zeros(nvox, cp.float64)
    scount = cp.zeros(nvox, cp.int32)
    UBinv = np.linalg.inv(UB)

    for i, (phi, img_host) in enumerate(zip(phis, _prefetch(frames, read_fn, prefetch))):
        Rn = _axis_rot(osc_axis, sense * (phi - phi0)) @ R0
        Mn = cp.asarray((UBinv @ Rn.T).astype(np.float32))       # hkl = r_lab @ Mn.T
        img = cp.asarray(img_host.ravel())
        vi = cp.floor((r_lab_g @ Mn.T - lo_g) / step).astype(cp.int32)
        h0, k0, l0 = vi[:, 0], vi[:, 1], vi[:, 2]
        inb = ((img >= 0)
               & (h0 >= 0) & (h0 < nh) & (k0 >= 0) & (k0 < nk) & (l0 >= 0) & (l0 < nl))
        if hot is not None:
            inb &= img < hot
        flat = (h0.astype(cp.int64) * nkl + k0 * nl + l0)[inb]     # int64: any grid size
        w = img[inb].astype(cp.float64)
        if corr_g is not None:
            w = w * corr_g[inb]
        cupyx.scatter_add(ssum, flat, w)
        cupyx.scatter_add(scount, flat, 1)
        if progress_callback:
            progress_callback(i + 1, len(frames))

    ssum_h = cp.asnumpy(ssum)
    scount_h = cp.asnumpy(scount).astype(np.int64)
    del ssum, scount, r_lab_g, corr_g
    cp.get_default_memory_pool().free_all_blocks()

    with np.errstate(invalid="ignore"):
        vol = np.where(scount_h > 0, ssum_h / np.maximum(scount_h, 1), np.nan)
    return Volume(vol.reshape(nh, nk, nl).astype(np.float32), H, K, L,
                  np.asarray(UB), scount_h.reshape(nh, nk, nl), detector.wavelength)


# ---------------------------------------------------------------- CPU fallback
def reconstruct_volume(frames, phis, UB, R0, detector, *, phi0,
                       osc_axis=(0, 1, 0), sense=1,
                       ranges=((-6.0, 6.0), (-6.0, 6.0), (-6.0, 6.0)), step=0.025,
                       read_fn=None, hot=None, progress_callback=None,
                       corrections=None, **_ignored) -> Volume:
    """CPU reconstruction (numpy bincount). Same result as the GPU path."""
    read_fn = read_fn or _default_read_fn()
    frames, phis = _validate_series(frames, phis)

    ny, nx = detector.shape
    ys, xs = np.mgrid[0:ny, 0:nx]
    px = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float32)
    r_lab = detector.scattering_vectors(px).astype(np.float32)

    (nh, nk, nl), (H, K, L), (lo_h, lo_k, lo_l) = _axes(ranges, step)
    nkl = nk * nl
    nvox = nh * nkl
    corr = _resolve_corrections(corrections, detector, r_lab)
    lo = np.asarray([lo_h, lo_k, lo_l], np.float32)
    ssum = np.zeros(nvox, np.float64)
    scount = np.zeros(nvox, np.int64)
    UBinv = np.linalg.inv(UB)

    for i, (phi, img_host) in enumerate(zip(phis, _prefetch(frames, read_fn, 2))):
        Rn = _axis_rot(osc_axis, sense * (phi - phi0)) @ R0
        Mn = (UBinv @ Rn.T).astype(np.float32)
        img = img_host.ravel()
        vi = np.floor((r_lab @ Mn.T - lo) / step).astype(np.int64)
        inb = ((img >= 0)
               & (vi[:, 0] >= 0) & (vi[:, 0] < nh)
               & (vi[:, 1] >= 0) & (vi[:, 1] < nk)
               & (vi[:, 2] >= 0) & (vi[:, 2] < nl))
        if hot is not None:
            inb &= img < hot
        flat = vi[inb, 0] * nkl + vi[inb, 1] * nl + vi[inb, 2]
        w = img[inb].astype(np.float64)
        if corr is not None:
            w = w * corr[inb]
        ssum += np.bincount(flat, weights=w, minlength=nvox)
        scount += np.bincount(flat, minlength=nvox)
        if progress_callback:
            progress_callback(i + 1, len(frames))

    with np.errstate(invalid="ignore"):
        vol = np.where(scount > 0, ssum / np.maximum(scount, 1), np.nan)
    return Volume(vol.reshape(nh, nk, nl).astype(np.float32), H, K, L,
                  np.asarray(UB), scount.reshape(nh, nk, nl), detector.wavelength)


# ---------------------------------------------------------------- adapters
def bragg_registration_rms(vol: Volume, min_count=20, K=300, r_excl=0.6):
    """RMS distance (rlu) of the K brightest well-measured voxels to the nearest
    integer node - a reference-free check that the reconstruction is correctly
    registered to the lattice (should be < the voxel size)."""
    hh, kk, ll = np.meshgrid(vol.H, vol.K, vol.L, indexing="ij")
    d = vol.data.copy()
    d[vol.counts < min_count] = np.nan
    d[(hh ** 2 + kk ** 2 + ll ** 2) < r_excl ** 2] = np.nan
    flat = d.ravel()
    finite_idx = np.flatnonzero(np.isfinite(flat))
    if finite_idx.size == 0:
        return float("nan")
    # K brightest among the finite candidates only (never NaN/-inf fillers)
    order = np.argsort(flat[finite_idx])[::-1][:K]
    idx = finite_idx[order]
    coords = np.stack([hh.ravel()[idx], kk.ravel()[idx], ll.ravel()[idx]], 1)
    dist = np.linalg.norm(coords - np.round(coords), axis=1)
    return float(np.sqrt(np.mean(dist ** 2)))


def reconstruct_to_volumedata(vol: Volume, source_folder=None, extra_meta=None):
    """Wrap a rawrecon Volume as an rspace3d VolumeData (NaN = unmeasured), ready
    for bin_volume / symmetrize_volume / save_volume_h5 / the viewer.

    The grid is a regular fractional-hkl grid (`grid_kind='hkl_regular'`,
    diagonal index->hkl map): it carries no unwarp-raster geometry (`s`, `cx`,
    `cy`, `M_inv`), so the viewer and the non-native cuts use the H/K/L axes
    directly instead of inferring a raster step from |a*| (wrong for a != b)
    or injecting a raster shear that this grid does not have. The per-voxel
    pixel counts travel along as `counts`."""
    from ..volume_builder import VolumeData

    recip = np.asarray(vol.UB, float)
    wl = float(vol.wavelength)
    meta = {
        "wavelength": wl,
        "ub": recip * wl,                      # lambda-scaled (rspace3d/CrysAlis convention)
        "cell": _reciprocal_cell(recip),
        "grid_kind": "hkl_regular",
        "bin_xy": 1, "bin_z": 1,
        "reconstructed_by": "rspace3d.rawrecon",
        "n_measured_voxels": int((vol.counts > 0).sum()),
        "measured_pct": float(100.0 * (vol.counts > 0).mean()),
    }
    if source_folder:
        meta["source_folder"] = source_folder
    if extra_meta:
        meta.update(extra_meta)
    return VolumeData(intensity=vol.data, H=vol.H, K=vol.K, L=vol.L,
                      plane_type="HK", metadata=meta, counts=vol.counts)


# ---------------------------------------------------------------- high level
def reconstruct_dataset(folder, *, ranges=((-6.0, 6.0), (-6.0, 6.0), (-6.0, 6.0)),
                        step=0.025, nframes=None, corrections=True, hot=None,
                        use_gpu=True, device=0, prefetch=4, par_path=None,
                        use_tabbin=False, progress_callback=None, log=None):
    """One call: locate the cracker par + CBF frames, read the UB, get the
    orientation, and reconstruct. Returns (Volume, info_dict).

    use_tabbin : if True, fit the full geometry (detector pose, oscillation axis,
        R0) from the CrysAlisPro peak-hunt tabbin (calibrate.geometry_from_tabbin) -
        instrument-agnostic and robust to a wrong header or powder contamination.
        If False, index frame 1 (orient_from_frame) assuming the header geometry and
        a lab-Y oscillation axis (works for clean, correctly-headered datasets).
    par_path : explicit .par to use (e.g. in a subfolder); else searched in `folder`.
        It is also the par the tabbin calibration uses.
    hot : raw-count cutoff above which pixels are ignored; None (default) keeps
        every pixel, so saturated Bragg pixels are never clipped silently.
    corrections : True applies solid-angle + polarisation corrections with the
        beam polarised perpendicular to the oscillation axis (horizontal for the
        usual vertical rotation axis); a dict is passed to `pixel_corrections`
        (e.g. `horizontal=` to override the polarisation axis); an array is used
        as is.
    `log` is an optional callable(str) for human-readable progress lines.
    """
    import os

    say = log or (lambda *_: None)
    name = os.path.basename(os.path.normpath(folder))
    if par_path is None:
        par_path = find_crysalis_par(folder)
    if par_path is None:
        raise FileNotFoundError(f"no CrysAlisPro .par file found in/under {folder}")
    if not os.path.isfile(par_path):
        raise FileNotFoundError(f"CrysAlisPro parameter file does not exist: {par_path}")
    all_frames, all_nums = find_cbf_frames(folder)
    if not all_frames:
        raise FileNotFoundError(f"no numbered .cbf frames found in {folder}")
    if np.any(np.diff(all_nums) <= 0):
        raise ValueError("discovered CBF frame numbers must be strictly increasing")
    n_total = len(all_frames)
    n = int(nframes) if nframes else n_total
    if n <= 0 or n > n_total:
        raise ValueError(f"nframes must be between 1 and {n_total}, got {nframes}")
    frames, nums = all_frames[:n], all_nums[:n]

    if use_tabbin:
        from .calibrate import geometry_from_tabbin
        fg = geometry_from_tabbin(folder, par_path=par_path, log=say)
        det, UB, R0 = fg.detector, fg.UB, fg.R0
        osc_axis, sense, phi0, incr = fg.osc_axis, fg.sense, fg.phi0, fg.increment
        geom_rms, geom_inl, geom_dev = fg.rms, fg.n_peaks, fg.median_hkl_dev
        say(f"tabbin geometry: beam ({det.beam_center[0]:.1f},{det.beam_center[1]:.1f}) "
            f"dist {det.distance:.1f}mm, osc {tuple(round(x,3) for x in osc_axis)} "
            f"sense {sense:+d}, rms {geom_rms:.4f} 1/A on {geom_inl} peaks")
    else:
        det, ang0, _ = FlatDetector.from_eiger_cbf(frames[0])
        incr = ang0["Angle_increment"]
        det, UB, R0, phi0, idx = orient_from_frame(par_path, frames[0])
        osc_axis, sense = (0, 1, 0), 1
        geom_rms, geom_inl, geom_dev = idx.rms, int(idx.inliers.sum()), None
        say(f"par: {os.path.basename(par_path)}")
        say(f"indexed frame 1: rms={idx.rms:.5f} rlu, {geom_inl} inliers")
    cell = _reciprocal_cell(UB)
    say(f"cell a,b,c = {cell['a']:.3f},{cell['b']:.3f},{cell['c']:.3f} A  "
        f"ang {cell['alpha']:.2f},{cell['beta']:.2f},{cell['gamma']:.2f}")

    # phi tied to the frame NUMBER (robust to gaps): phi = phi0 + (num - num0)*incr
    phis = [phi0 + (num - nums[0]) * incr for num in nums]
    missing = sorted(set(range(nums[0], nums[-1] + 1)) - set(nums))
    if missing:
        say(f"frame-number gaps kept in the angle mapping: {missing[:10]}"
            f"{' ...' if len(missing) > 10 else ''}")
    # Polarisation: E-vector perpendicular to beam and oscillation axis. The
    # tabbin fit's lab frame is only defined up to a rotation about the beam,
    # so neither the detector fast axis nor lab +X is physically anchored;
    # the oscillation axis is.
    polarization_axis = "perpendicular to the oscillation axis"
    if corrections is True:
        e_vec = np.cross(det.beam, np.asarray(osc_axis, float))
        corr = pixel_corrections(det, horizontal=e_vec / np.linalg.norm(e_vec))
    elif isinstance(corrections, dict):
        kw = dict(corrections)
        if kw.get("horizontal") is None:
            e_vec = np.cross(det.beam, np.asarray(osc_axis, float))
            kw["horizontal"] = e_vec / np.linalg.norm(e_vec)
        else:
            polarization_axis = "explicit"
        corr = pixel_corrections(det, **kw)
    elif corrections is None or corrections is False:
        corr, polarization_axis = None, "none"
    else:
        corr, polarization_axis = corrections, "explicit array"

    gpu = use_gpu and has_cupy(device)
    recon = reconstruct_volume_gpu if gpu else reconstruct_volume
    say(f"reconstructing {n}/{n_total} frames on {'GPU' if gpu else 'CPU'}, "
        f"grid {[list(r) for r in ranges]} dq={step} ...")
    t0 = time.time()
    vol = recon(frames, phis, UB, R0, det, phi0=phi0, osc_axis=osc_axis, sense=sense,
                ranges=ranges, step=step, hot=hot, corrections=corr,
                progress_callback=progress_callback, device=device, prefetch=prefetch)
    dt = time.time() - t0

    info = {
        "par": par_path, "name": name, "n_frames": n, "n_total": n_total,
        "phi0": phi0, "increment": incr, "index_rms": geom_rms,
        "index_inliers": geom_inl, "geom_hkl_dev": geom_dev,
        "osc_axis": osc_axis, "sense": sense, "cell": cell, "gpu": gpu,
        "time_s": dt, "ms_per_frame": 1e3 * dt / max(n, 1),
        "measured_voxels": int((vol.counts > 0).sum()),
        "total_voxels": int(vol.data.size),
        "corrections": bool(corr is not None),
        "polarization_axis": polarization_axis,
        "hot_pixel_cutoff": hot,
        "geometry_method": "tabbin" if use_tabbin else "indexed_first_frame",
        "missing_frame_numbers": missing,
    }
    say(f"done in {dt:.1f}s ({info['ms_per_frame']:.1f} ms/frame); measured "
        f"{info['measured_voxels']:,}/{info['total_voxels']:,} voxels "
        f"({100*info['measured_voxels']/info['total_voxels']:.1f}%)")
    return vol, info
