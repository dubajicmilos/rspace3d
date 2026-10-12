"""Raw-CBF -> reciprocal-space reconstruction engine (GPU via CuPy, CPU fallback).

Every detector pixel is mapped to a fractional (h,k,l) and its intensity
accumulated into a voxel grid over all rotation frames:

    hkl = (R_n . UB)^-1 . r_lab ,   R_n = R_osc(sense*(phi_n - phi0)) . R0 ,

with r_lab fixed by the detector and R0 the reference-frame orientation. Each
voxel stores the mean of the contributing pixels (sum / count); a voxel with zero
contributing pixels is UNMEASURED and set to NaN. Because the pixel `counts` are
known exactly, measured-but-zero voxels (0.0) and unmeasured voxels (NaN) are
unambiguous - no coverage-mask heuristic is needed (unlike the .img/unwarp path).

Each voxel also gets a variance (sigma^2 of the voxel mean). A voxel value is
sum_p(g_p r_p) / N, with the raw count r_p and the correction factor g_p of
each contributing pixel p. The raw counts are Poisson, so

    variance = sum_p(g_p^2 max(r_p, 1)) / N^2 .

The floor of 1 count per pixel is the same floor that the outlier rejection
uses (`poisson_floor`). A voxel with N = 0 has a NaN variance.

The grid may be anisotropic: independent (min,max) per axis, common voxel size dq.

Options of the two engine functions (all off by default):

* `groups` gives each frame a group number. The engine then makes one
  partial volume per group in the same pass (for example odd and even
  frames). The full volume is the sum of the partial volumes.
* `frame_fn(i, img)` changes the pixels of frame i before the scatter. It
  returns None (no change) or `(offset, gain)`. The pixel value is then
  g (r - offset) gain and its variance is g^2 gain^2 max(r, 1). A pixel with
  a gain that is not finite is not used (a mask). The raw count r controls
  the validity test, the hot cutoff and the floor of the variance.
* `cancelled()` is called once per frame. If it returns True, the engine
  stops with `averaging.Cancelled`.
"""
from __future__ import annotations

import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

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
    variance: np.ndarray | None = None   # (nh,nk,nl) sigma^2 of data, NaN where unmeasured
    parts: list | None = None            # one Volume per frame group (`groups`), else None


# ---------------------------------------------------------------- grid / cell
def _axes(ranges, step, bins=(1, 1, 1)):
    """(nh,nk,nl), (H,K,L), (lo_h,lo_k,lo_l) for per-axis (lo,hi) ranges + dq.

    Each axis is a sequence of blocks of b voxels (b from `bins`). The block
    centres are the integer multiples of b * step in [lo, hi]. Thus hkl = 0
    is a voxel centre after `bin_volume` with the same factors, and the
    operations of all 11 Laue groups stay exact index maps. lo_* are the
    lower edges of the first voxels, as the forward scatter uses them.
    """
    los, ns, axes = [], [], []
    for (lo, hi), b in zip(ranges, bins):
        j0 = int(np.ceil(lo / (b * step) - 1e-9))
        j1 = int(np.floor(hi / (b * step) + 1e-9))
        n = max(0, j1 - j0 + 1) * b
        first = j0 * b - (b - 1) / 2.0          # first voxel centre, in units of step
        los.append((first - 0.5) * step); ns.append(n)
        axes.append((first + np.arange(n)) * step)
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


def _mean_and_variance(ssum, scount, ssq):
    """Voxel mean sum/N and its variance ssq/N^2 (float32, NaN where N = 0)."""
    measured = scount > 0
    n = np.maximum(scount, 1).astype(np.float64)
    vol = np.where(measured, ssum / n, np.nan).astype(np.float32)
    var = np.where(measured, ssq / (n * n), np.nan).astype(np.float32)
    return vol, var


def frame_matrix(UBinv, R0, osc_axis, sense, phi, phi0):
    """float32 (3, 3) matrix Mn of a frame: hkl = r_lab @ Mn.T."""
    Rn = _axis_rot(osc_axis, sense * (phi - phi0)) @ R0
    return (UBinv @ Rn.T).astype(np.float32)


def _check_groups(groups, n):
    """Validated int64 group number per frame and the number of groups (None: one group)."""
    if groups is None:
        return None, 1
    g = np.asarray(groups)
    if g.shape != (n,) or not np.issubdtype(g.dtype, np.integer) or (n and g.min() < 0):
        raise ValueError(f"groups must be {n} integers >= 0, one for each frame")
    return g.astype(np.int64), int(g.max()) + 1


def _cancel_check(cancelled):
    if cancelled is not None and cancelled():
        from ..averaging import Cancelled
        raise Cancelled("Cancelled.")


def _frame_stream(frames, read_fn, frame_fn, ahead):
    """(image, fix) per frame. fix is the result of frame_fn, or None without frame_fn."""
    if frame_fn is None:
        for img in _prefetch(frames, read_fn, ahead):
            yield img, None
        return

    def read_fix(i):
        img = read_fn(frames[i])
        return img, frame_fn(i, img)
    yield from _prefetch(list(range(len(frames))), read_fix, ahead)


def _assemble(H, K, L, UB, wavelength, shape, sums):
    """Volume from host accumulators. `sums` is one (ssum, scount, ssq) per group.

    One group gives the plain Volume. More groups give a Volume whose
    accumulators are the sums of the groups, with one Volume per group in
    `parts`."""
    def make(ssum, scount, ssq, parts=None):
        vol, var = _mean_and_variance(ssum, scount, ssq)
        return Volume(vol.reshape(shape), H, K, L, np.asarray(UB),
                      scount.reshape(shape), wavelength, var.reshape(shape), parts)

    if len(sums) == 1:
        return make(*sums[0])
    parts = [make(*s) for s in sums]
    ssum, scount, ssq = (a.copy() for a in sums[0])
    for s in sums[1:]:
        ssum += s[0]
        scount += s[1]
        ssq += s[2]
    return make(ssum, scount, ssq, parts)


# ---------------------------------------------------------------- GPU engine
def reconstruct_volume_gpu(frames, phis, UB, R0, detector, *, phi0,
                           osc_axis=(0, 1, 0), sense=1,
                           ranges=((-6.0, 6.0), (-6.0, 6.0), (-6.0, 6.0)), step=0.025,
                           bins=(1, 1, 1),
                           read_fn=None, hot=None, progress_callback=None,
                           corrections=None, device=0, prefetch=3,
                           groups=None, frame_fn=None, cancelled=None) -> Volume:
    """GPU reconstruction. Per-frame float64 scatter-add into a resident
    accumulator; disk reads prefetched on a thread pool. Returns a host Volume.

    `groups`, `frame_fn` and `cancelled` are optional (see the module text).
    Without them the result is the same as in rspace3d 2.5.0."""
    import cupy as cp
    import cupyx

    read_fn = read_fn or _default_read_fn()
    cp.cuda.Device(device).use()
    frames, phis = _validate_series(frames, phis)
    gidx, n_groups = _check_groups(groups, len(frames))

    ny, nx = detector.shape
    ys, xs = np.mgrid[0:ny, 0:nx]
    px = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float32)
    r_lab = detector.scattering_vectors(px).astype(np.float32)

    (nh, nk, nl), (H, K, L), (lo_h, lo_k, lo_l) = _axes(ranges, step, bins)
    nkl = nk * nl
    nvox = nh * nkl
    corr = _resolve_corrections(corrections, detector, r_lab)
    r_lab_g = cp.asarray(r_lab)
    corr_g = None if corr is None else cp.asarray(corr)
    lo_g = cp.asarray([lo_h, lo_k, lo_l], cp.float32)

    ssum = [cp.zeros(nvox, cp.float64) for _ in range(n_groups)]
    scount = [cp.zeros(nvox, cp.int32) for _ in range(n_groups)]
    ssq = [cp.zeros(nvox, cp.float64) for _ in range(n_groups)]   # sum of g^2 * max(r, 1)
    UBinv = np.linalg.inv(UB)

    stream = _frame_stream(frames, read_fn, frame_fn, prefetch)
    for i, (phi, (img_host, fix)) in enumerate(zip(phis, stream)):
        _cancel_check(cancelled)
        Mn = cp.asarray(frame_matrix(UBinv, R0, osc_axis, sense, phi, phi0))  # hkl = r_lab @ Mn.T
        img = cp.asarray(img_host.ravel())
        vi = cp.floor((r_lab_g @ Mn.T - lo_g) / step).astype(cp.int32)
        h0, k0, l0 = vi[:, 0], vi[:, 1], vi[:, 2]
        inb = ((img >= 0)
               & (h0 >= 0) & (h0 < nh) & (k0 >= 0) & (k0 < nk) & (l0 >= 0) & (l0 < nl))
        if hot is not None:
            inb &= img < hot
        if fix is not None:
            off, gain = fix
            if gain is not None:
                gain = cp.asarray(np.asarray(gain).ravel()).astype(cp.float64)
                inb &= cp.isfinite(gain)
        flat = (h0.astype(cp.int64) * nkl + k0 * nl + l0)[inb]     # int64: any grid size
        r = img[inb].astype(cp.float64)
        if fix is None:
            w = r
            w2 = cp.maximum(r, 1.0)
            if corr_g is not None:
                g = corr_g[inb]
                w = r * g
                w2 = w2 * g * g
        else:
            ga = gain[inb] if gain is not None else cp.ones_like(r)
            if corr_g is not None:
                ga = corr_g[inb] * ga
            c = r
            if off is not None:
                c = r - cp.asarray(np.asarray(off).ravel()).astype(cp.float64)[inb]
            w = c * ga
            w2 = cp.maximum(r, 1.0) * ga * ga
        gi = 0 if gidx is None else int(gidx[i])
        cupyx.scatter_add(ssum[gi], flat, w)
        cupyx.scatter_add(scount[gi], flat, 1)
        cupyx.scatter_add(ssq[gi], flat, w2)
        if progress_callback:
            progress_callback(i + 1, len(frames))

    sums = [(cp.asnumpy(ssum[j]), cp.asnumpy(scount[j]).astype(np.int64), cp.asnumpy(ssq[j]))
            for j in range(n_groups)]
    del ssum, scount, ssq, r_lab_g, corr_g
    cp.get_default_memory_pool().free_all_blocks()
    return _assemble(H, K, L, UB, detector.wavelength, (nh, nk, nl), sums)


# ---------------------------------------------------------------- CPU fallback
def reconstruct_volume(frames, phis, UB, R0, detector, *, phi0,
                       osc_axis=(0, 1, 0), sense=1,
                       ranges=((-6.0, 6.0), (-6.0, 6.0), (-6.0, 6.0)), step=0.025,
                       bins=(1, 1, 1),
                       read_fn=None, hot=None, progress_callback=None,
                       corrections=None, groups=None, frame_fn=None, cancelled=None,
                       **_ignored) -> Volume:
    """CPU reconstruction (numpy bincount). Same result as the GPU path.

    `groups`, `frame_fn` and `cancelled` are optional (see the module text).
    Without them the result is the same as in rspace3d 2.5.0."""
    read_fn = read_fn or _default_read_fn()
    frames, phis = _validate_series(frames, phis)
    gidx, n_groups = _check_groups(groups, len(frames))

    ny, nx = detector.shape
    ys, xs = np.mgrid[0:ny, 0:nx]
    px = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float32)
    r_lab = detector.scattering_vectors(px).astype(np.float32)

    (nh, nk, nl), (H, K, L), (lo_h, lo_k, lo_l) = _axes(ranges, step, bins)
    nkl = nk * nl
    nvox = nh * nkl
    corr = _resolve_corrections(corrections, detector, r_lab)
    lo = np.asarray([lo_h, lo_k, lo_l], np.float32)
    ssum = [np.zeros(nvox, np.float64) for _ in range(n_groups)]
    scount = [np.zeros(nvox, np.int64) for _ in range(n_groups)]
    ssq = [np.zeros(nvox, np.float64) for _ in range(n_groups)]   # sum of g^2 * max(r, 1)
    UBinv = np.linalg.inv(UB)

    stream = _frame_stream(frames, read_fn, frame_fn, 2)
    for i, (phi, (img_host, fix)) in enumerate(zip(phis, stream)):
        _cancel_check(cancelled)
        Mn = frame_matrix(UBinv, R0, osc_axis, sense, phi, phi0)
        img = img_host.ravel()
        vi = np.floor((r_lab @ Mn.T - lo) / step).astype(np.int64)
        inb = ((img >= 0)
               & (vi[:, 0] >= 0) & (vi[:, 0] < nh)
               & (vi[:, 1] >= 0) & (vi[:, 1] < nk)
               & (vi[:, 2] >= 0) & (vi[:, 2] < nl))
        if hot is not None:
            inb &= img < hot
        if fix is not None:
            off, gain = fix
            if gain is not None:
                gain = np.asarray(gain).ravel().astype(np.float64)
                inb &= np.isfinite(gain)
        flat = vi[inb, 0] * nkl + vi[inb, 1] * nl + vi[inb, 2]
        r = img[inb].astype(np.float64)
        if fix is None:
            w = r
            w2 = np.maximum(r, 1.0)
            if corr is not None:
                g = corr[inb]
                w = r * g
                w2 = w2 * g * g
        else:
            ga = gain[inb] if gain is not None else np.ones_like(r)
            if corr is not None:
                ga = corr[inb] * ga
            c = r
            if off is not None:
                c = r - np.asarray(off).ravel().astype(np.float64)[inb]
            w = c * ga
            w2 = np.maximum(r, 1.0) * ga * ga
        gi = 0 if gidx is None else int(gidx[i])
        ssum[gi] += np.bincount(flat, weights=w, minlength=nvox)
        scount[gi] += np.bincount(flat, minlength=nvox)
        ssq[gi] += np.bincount(flat, weights=w2, minlength=nvox)
        if progress_callback:
            progress_callback(i + 1, len(frames))

    return _assemble(H, K, L, UB, detector.wavelength, (nh, nk, nl),
                     list(zip(ssum, scount, ssq)))


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
    pixel counts travel along as `counts` and the per-voxel variance as
    `variance` (`metadata['variance_model'] = 'poisson_counts'`)."""
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
    if vol.variance is not None:
        meta["variance_model"] = "poisson_counts"
    if source_folder:
        meta["source_folder"] = source_folder
    if extra_meta:
        meta.update(extra_meta)
    return VolumeData(intensity=vol.data, H=vol.H, K=vol.K, L=vol.L,
                      plane_type="HK", metadata=meta, counts=vol.counts,
                      variance=vol.variance)




# ---------------------------------------------------------------- setup of a dataset
@dataclass
class ReconSetup:
    """Geometry, frames and grid of one raw-frame dataset.

    The fields are the arguments of `reconstruct_volume` and
    `reconstruct_volume_gpu`. Make the setup one time (`prepare_setup`, or
    `reconstruct_dataset(..., keep_setup=True)` in `info['setup']`). Then
    each frame tool makes its passes over the frames without a new geometry
    fit.

    frames : frame paths (or keys of `read_fn`), in scan order.
    numbers : frame numbers. The split into halves uses them.
    phis : rotation angle of each frame in degrees.
    corrections : per-pixel correction array (solid angle and polarisation),
        or None.
    hot : raw-count cutoff. Pixels at or above it are not used. None keeps all.
    read_fn : callable(frame) -> int image. None reads the file with fabio.
    meta : names and text values for the metadata (not used in passes).
    """
    frames: list
    numbers: list
    phis: list
    detector: Any
    UB: np.ndarray
    R0: np.ndarray
    phi0: float
    osc_axis: tuple = (0, 1, 0)
    sense: int = 1
    increment: float = 0.0
    corrections: Any = None
    ranges: tuple = ((-6.0, 6.0), (-6.0, 6.0), (-6.0, 6.0))
    step: float = 0.025
    bins: tuple = (1, 1, 1)
    hot: float | None = None
    read_fn: Callable | None = None
    device: int = 0
    prefetch: int = 4
    meta: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    def reader(self) -> Callable:
        """The function that reads one frame."""
        return self.read_fn or _default_read_fn()

    def halves_groups(self) -> np.ndarray:
        """Group of each frame for the split into halves: 0 = odd frame number, 1 = even."""
        return (np.asarray(self.numbers, np.int64) % 2 == 0).astype(np.int64)

    def reconstruct(self, *, positions=None, groups=None, frame_fn=None, use_gpu=True,
                    progress_callback=None, cancelled=None) -> Volume:
        """Run the engine on all frames, or on the frame `positions` (indices into `frames`).

        `frame_fn(i, img)` gets the position i in `frames`, also with `positions`.
        """
        if positions is None:
            pos = list(range(self.n_frames))
        else:
            pos = [int(p) for p in positions]
        fn = frame_fn
        if frame_fn is not None and positions is not None:
            fn = (lambda j, img: frame_fn(pos[j], img))
        g = None if groups is None else np.asarray(groups)[pos]
        gpu = use_gpu and has_cupy(self.device)
        recon = reconstruct_volume_gpu if gpu else reconstruct_volume
        return recon([self.frames[i] for i in pos], [self.phis[i] for i in pos],
                     self.UB, self.R0, self.detector, phi0=self.phi0,
                     osc_axis=self.osc_axis, sense=self.sense, ranges=self.ranges,
                     step=self.step, bins=self.bins, read_fn=self.read_fn, hot=self.hot,
                     corrections=self.corrections, progress_callback=progress_callback,
                     device=self.device, prefetch=self.prefetch, groups=g, frame_fn=fn,
                     cancelled=cancelled)


def _checked_halves(setup: ReconSetup) -> np.ndarray:
    """`setup.halves_groups()`. Raises ValueError if the frames do not have the two parities."""
    g = setup.halves_groups()
    if np.unique(g).size < 2:
        raise ValueError("The split into halves needs odd and even frame numbers. "
                         "All frames have the same parity.")
    return g


def gpu_memory_bytes(n_voxels: int, halves: bool = False) -> int:
    """GPU accumulator memory of one reconstruction pass in bytes.

    Each frame group needs a float64 sum, an int32 count and a float64 sum of
    squares (20 bytes per voxel). The split into halves uses two groups."""
    return int(n_voxels) * 20 * (2 if halves else 1)


# ---------------------------------------------------------------- high level
def prepare_setup(folder, *, ranges=((-6.0, 6.0), (-6.0, 6.0), (-6.0, 6.0)),
                  step=0.025, bins=(1, 1, 1), nframes=None, corrections=True,
                  hot=None, device=0, prefetch=4, par_path=None, use_tabbin=False,
                  log=None) -> ReconSetup:
    """Find the frames and the par file of `folder`, get the geometry and make a `ReconSetup`.

    The arguments are those of `reconstruct_dataset`. The function does not
    read the frames for the volume."""
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

    meta = {
        "par": par_path, "name": name, "folder": folder, "n_total": n_total,
        "index_rms": geom_rms, "index_inliers": geom_inl, "geom_hkl_dev": geom_dev,
        "cell": cell, "polarization_axis": polarization_axis,
        "geometry_method": "tabbin" if use_tabbin else "indexed_first_frame",
        "missing_frame_numbers": missing,
    }
    return ReconSetup(frames=list(frames), numbers=[int(v) for v in nums], phis=phis,
                      detector=det, UB=UB, R0=R0, phi0=phi0, osc_axis=osc_axis,
                      sense=sense, increment=incr, corrections=corr, ranges=ranges,
                      step=step, bins=bins, hot=hot, read_fn=None, device=device,
                      prefetch=prefetch, meta=meta)


def reconstruct_dataset(folder, *, ranges=((-6.0, 6.0), (-6.0, 6.0), (-6.0, 6.0)),
                        step=0.025, bins=(1, 1, 1), nframes=None, corrections=True,
                        hot=None, use_gpu=True, device=0, prefetch=4, par_path=None,
                        use_tabbin=False, progress_callback=None, log=None,
                        halves=False, cancelled=None, keep_setup=False):
    """One call: locate the cracker par + CBF frames, read the UB, get the
    orientation, and reconstruct. Returns (Volume, info_dict).

    use_tabbin : if True, fit the full geometry (detector pose, oscillation axis,
        R0) from the CrysAlisPro peak-hunt tabbin (calibrate.geometry_from_tabbin) -
        instrument-agnostic and robust to a wrong header or powder contamination.
        If False, index frame 1 (orient_from_frame) assuming the header geometry and
        a lab-Y oscillation axis (works for clean, correctly-headered datasets).
    par_path : explicit .par to use (e.g. in a subfolder); else searched in `folder`.
        It is also the par the tabbin calibration uses.
    bins : (bh, bk, bl) factors that you will give to `bin_volume` before
        symmetrisation. The grid is built so that hkl = 0 is a voxel centre
        after this binning (see `_axes`). Use the default (1, 1, 1) if you do
        not bin.
    hot : raw-count cutoff above which pixels are ignored; None (default) keeps
        every pixel, so saturated Bragg pixels are never clipped silently.
    corrections : True applies solid-angle + polarisation corrections with the
        beam polarised perpendicular to the oscillation axis (horizontal for the
        usual vertical rotation axis); a dict is passed to `pixel_corrections`
        (e.g. `horizontal=` to override the polarisation axis); an array is used
        as is.
    halves : if True, the same pass also makes the odd-frame and the
        even-frame volumes (frame-number parity). They are in `vol.parts`
        (odd first). The GPU needs two times the accumulator memory. The
        frames must have odd and even numbers. Else the function raises
        ValueError.
    cancelled : optional callable. It is called once per frame. If it returns
        True, the pass stops with `averaging.Cancelled`.
    `log` is an optional callable(str) for human-readable progress lines.
    keep_setup : if True, `info['setup']` is the `ReconSetup` of the run.
        Frame tools use it for more passes. It is not a JSON value. Remove it
        before you serialise `info`, for example
        `{k: v for k, v in info.items() if k != 'setup'}`. With the default
        (False), `info` has no 'setup' key.
    """
    say = log or (lambda *_: None)
    setup = prepare_setup(folder, ranges=ranges, step=step, bins=bins, nframes=nframes,
                          corrections=corrections, hot=hot, device=device,
                          prefetch=prefetch, par_path=par_path, use_tabbin=use_tabbin,
                          log=log)
    n, n_total = setup.n_frames, setup.meta["n_total"]
    gpu = use_gpu and has_cupy(device)
    say(f"reconstructing {n}/{n_total} frames on {'GPU' if gpu else 'CPU'}, "
        f"grid {[list(r) for r in ranges]} dq={step} ...")
    t0 = time.time()
    vol = setup.reconstruct(groups=_checked_halves(setup) if halves else None, use_gpu=gpu,
                            progress_callback=progress_callback, cancelled=cancelled)
    dt = time.time() - t0

    m = setup.meta
    info = {
        "par": m["par"], "name": m["name"], "n_frames": n, "n_total": n_total,
        "phi0": setup.phi0, "increment": setup.increment, "index_rms": m["index_rms"],
        "index_inliers": m["index_inliers"], "geom_hkl_dev": m["geom_hkl_dev"],
        "osc_axis": setup.osc_axis, "sense": setup.sense, "cell": m["cell"], "gpu": gpu,
        "time_s": dt, "ms_per_frame": 1e3 * dt / max(n, 1),
        "measured_voxels": int((vol.counts > 0).sum()),
        "total_voxels": int(vol.data.size),
        "corrections": bool(setup.corrections is not None),
        "polarization_axis": m["polarization_axis"],
        "hot_pixel_cutoff": hot,
        "geometry_method": m["geometry_method"],
        "missing_frame_numbers": m["missing_frame_numbers"],
    }
    if keep_setup:
        info["setup"] = setup
    if halves:
        info["halves_rule"] = "frame-number parity: part 0 = odd numbers, part 1 = even numbers"
    say(f"done in {dt:.1f}s ({info['ms_per_frame']:.1f} ms/frame); measured "
        f"{info['measured_voxels']:,}/{info['total_voxels']:,} voxels "
        f"({100*info['measured_voxels']/info['total_voxels']:.1f}%)")
    return vol, info
