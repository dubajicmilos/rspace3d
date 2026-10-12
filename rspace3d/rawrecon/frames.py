"""Frame tools: passes over the raw frames with a model volume.

The functions use an `engine.ReconSetup`, so they need no new geometry fit.

* `predict_frame` samples a model volume at the hkl of each pixel of a frame
  (trilinear between voxel centres) and gives the predicted raw counts.
* `leave_one_out_median` makes the prediction volume: for each voxel the
  median of the other members of its orbit (the voxel's own value is not
  used, except when an operation maps the voxel onto itself).
* `frame_statistics` (pass A) compares each frame with the prediction. For
  each frame, 5-degree sector and 1-pixel radial bin it gives a Huber
  location of the residual (D), the mean prediction (P) and the pixel count
  (N). It also gives the sums of the frame map.
* `frame_map` gives the agreement of each frame and sector with a first
  average (observed / predicted, divided by its median over the frames of
  the sector) and the radial residual profile.
* `corrected_reconstruction` makes a volume with the ring mask and the frame
  model applied to each pixel: g (obs - B) / S.
* `run_frame_chain` runs the frame chain: ring detection and mask,
  3 iterations of the frame model, plain orbit mean. The chain was checked
  with Laue group mmm and 0.2 degree frames.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..averaging import FRAME_MAP_DEFINITION, FRAME_MAP_FLOOR
from .engine import (ReconSetup, Volume, _prefetch, _cancel_check, _checked_halves, frame_matrix,
                     reconstruct_to_volumedata)
from .frame_model import (NSEC, NMIN, MODEL_ERR, REL_B, HUBER_C, FrameModel, ModelOptions,
                          fit_frame_model, frame_scale_of, get_xp, to_host, _free,
                          interp_B, interp_S)
from .rings import (PixelPolar, RingMask, RingDetection, detect_rings,
                    ring_statistics, sector_of_az, q_of_radius)


# ---------------------------------------------------------------- pixel cells of the frame model
class FrameCells:
    """Sector and radial bin of each pixel for the frame model (72 sectors, 1-pixel bins).

    Sector k is centred on the counter-clockwise angle 5 k degrees (y up).
    Radial bin b is centred on r = b + 0.5. `fa` and `fb` are the pixel
    coordinates in cell units (cell centres at integers)."""

    def __init__(self, polar: PixelPolar):
        self.polar = polar
        self.shape = polar.shape
        self.nbin = polar.nbin
        nb = self.nbin
        r = polar.r.astype(np.float64)
        self.rb = np.minimum(np.floor(r).astype(np.int64), nb - 1)
        self.sec = sector_of_az(polar.az, NSEC).astype(np.int64)
        self.seg = self.sec * nb + self.rb
        self.nseg = NSEC * nb
        th = (-polar.az.astype(np.float64)) % 360.0
        fa = th / (360.0 / NSEC)
        a0 = np.floor(fa)
        self.ta = (fa - a0).astype(np.float32)
        self.a0 = (a0.astype(np.int64) % NSEC).astype(np.int32)
        self.a1 = ((a0.astype(np.int64) + 1) % NSEC).astype(np.int32)
        self.fa = fa.astype(np.float32)
        self.fb = (r - 0.5).astype(np.float32)

    def B_image(self, B_row) -> np.ndarray:
        """Additive term of one frame at each pixel (npix,) float32."""
        return interp_B(np.asarray(B_row, np.float32), self.fa, self.fb)

    def S_image(self, a_row) -> np.ndarray:
        """Scale term S = 1 + a of one frame at each pixel (npix,) float32."""
        return interp_S(a_row, self.ta, self.a0, self.a1)


# ---------------------------------------------------------------- prediction
class _Predictor:
    """Device arrays to sample one model volume in every frame."""

    def __init__(self, xp, setup: ReconSetup, model, r_lab, corr32):
        kind = model.metadata.get("grid_kind", "hkl_regular")
        if kind != "hkl_regular":
            raise ValueError(f"the model volume must be on a regular hkl grid, not '{kind}'")
        inten = np.ascontiguousarray(model.intensity, np.float32)
        self.xp = xp
        self.V = xp.asarray(inten).ravel()
        self.shape = inten.shape
        axes = (model.H, model.K, model.L)
        steps = [float(a[1] - a[0]) if len(a) > 1 else float(setup.step) for a in axes]
        self.step = xp.asarray(np.asarray(steps, np.float32))
        self.lo = (xp.asarray([float(a[0]) for a in axes], xp.float32)
                   - xp.asarray(np.asarray(steps, np.float32) / 2))
        self.r_lab = r_lab
        self.corr = corr32
        self.UBinv = np.linalg.inv(setup.UB)
        self.setup = setup

    def __call__(self, i, min_weight=0.5):
        """Predicted raw counts (npix,) float32 of frame position i (NaN where not predicted)."""
        xp, s = self.xp, self.setup
        Mn = xp.asarray(frame_matrix(self.UBinv, s.R0, s.osc_axis, s.sense, s.phis[i], s.phi0))
        hkl = self.r_lab @ Mn.T
        nh, nk, nl = self.shape
        t = (hkl - self.lo) / self.step - 0.5
        i0 = xp.floor(t).astype(xp.int32)
        fr = t - i0
        acc = xp.zeros(t.shape[0], xp.float32)
        ws = xp.zeros(t.shape[0], xp.float32)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    ix, iy, iz = i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz
                    w = ((fr[:, 0] if dx else 1 - fr[:, 0]) * (fr[:, 1] if dy else 1 - fr[:, 1])
                         * (fr[:, 2] if dz else 1 - fr[:, 2]))
                    inb = (ix >= 0) & (ix < nh) & (iy >= 0) & (iy < nk) & (iz >= 0) & (iz < nl)
                    flat = ((xp.clip(ix, 0, nh - 1).astype(xp.int64) * nk + xp.clip(iy, 0, nk - 1))
                            * nl + xp.clip(iz, 0, nl - 1))
                    v = self.V[flat]
                    good = inb & xp.isfinite(v)
                    acc += xp.where(good, w * v, 0)
                    ws += xp.where(good, w, 0)
        pred = xp.where(ws >= min_weight, acc / xp.maximum(ws, 1e-12), xp.nan).astype(xp.float32)
        if self.corr is not None:
            pred = pred / self.corr
        return pred


def _lab_vectors(setup: ReconSetup) -> np.ndarray:
    ny, nx = setup.detector.shape
    ys, xs = np.mgrid[0:ny, 0:nx]
    px = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float32)
    return setup.detector.scattering_vectors(px).astype(np.float32)


def _corrections32(setup: ReconSetup, r_lab) -> np.ndarray | None:
    from .engine import _resolve_corrections
    corr = _resolve_corrections(setup.corrections, setup.detector, r_lab)
    return None if corr is None else np.asarray(corr).astype(np.float32)


def predict_frame(setup: ReconSetup, model, i: int, *, use_gpu: bool = False,
                  min_weight: float = 0.5) -> np.ndarray:
    """Predicted raw counts of frame position i from the model volume (a VolumeData).

    The model holds corrected intensities (the unit of a reconstruction). The
    function samples it trilinearly at the hkl of each pixel (voxel centres
    on the model axes) and divides by the pixel correction. A pixel is NaN if
    less than `min_weight` of the interpolation weight is on finite voxels.
    Returns a (ny, nx) float32 image."""
    xp = get_xp(use_gpu, setup.device)
    r_lab = _lab_vectors(setup)
    corr = _corrections32(setup, r_lab)
    pr = _Predictor(xp, setup, model, xp.asarray(r_lab), None if corr is None else xp.asarray(corr))
    out = to_host(pr(i, min_weight)).reshape(setup.detector.shape)
    _free(xp)
    return out


def leave_one_out_median(vol, laue_group: str) -> np.ndarray:
    """Median of the other orbit members of each voxel. float32, NaN if no member is finite.

    The operations come from `volume_builder.orbit_plan`. The identity is
    not used. When an operation maps a voxel onto itself (for example a
    mirror plane), the voxel's own value is one of the members."""
    from ..volume_builder import orbit_plan
    from .. import orbit_engine as oe
    data = np.ascontiguousarray(vol.intensity, np.float32)
    plan = orbit_plan(vol, laue_group)
    src = oe.make_source(plan, data)
    keep = np.array([o for o in range(plan.n_ops) if o != plan.identity_index])
    nh, nk, nl = data.shape
    out = np.empty(data.size, np.float32)
    for h0, h1, k0, k1 in oe._boxes(data.shape, max(1, oe._BOX_VALUES // plan.n_ops)):
        _, c, dc = oe._ranges(h0, h1, k0, k1, nl)
        shape = (h1 - h0, k1 - k0, nl)
        vals, _, _ = oe.gather_members(plan, src, c, dc, shape)
        others = vals[keep]
        n = np.isfinite(others).sum(axis=0)
        med = oe._row_median(others, n)
        med[n == 0] = np.nan
        start = (h0 * nk + k0) * nl
        out[start:start + med.size] = med
    return out.reshape(data.shape)


# ---------------------------------------------------------------- pass A
def _seg_median(xp, seg, v, n, q=8.0):
    """Median of v per segment: mean of the lower and upper middle, values quantised to 1/q."""
    if seg.size == 0:
        return xp.full(n.shape, xp.nan)
    key = xp.sort((seg << 24) | xp.clip(xp.rint((v + 1000.0) * q), 0, 2 ** 24 - 1).astype(xp.int64))
    val = (key & (2 ** 24 - 1)).astype(xp.float64) / q - 1000.0
    off = xp.cumsum(n) - n
    top = key.size - 1
    lo = xp.clip(off + (n - 1) // 2, 0, top)
    hi = xp.clip(off + n // 2, 0, top)
    return xp.where(n > 0, 0.5 * (val[lo] + val[hi]), xp.nan)


def _frame_cells_stats(xp, g, pred, seg, nseg):
    """D, P (float16, NaN if < 8 pixels) and n of each cell of one frame.

    g is the observed raw count (< 0: not used), pred the predicted raw count.
    A frame with no usable pixel gives NaN for D and P and 0 for n."""
    ok = (g >= 0) & xp.isfinite(pred)
    gv, pv, sv = g[ok], pred[ok], seg[ok]
    if gv.size == 0:
        nan16 = xp.full(nseg, xp.nan, xp.float16)
        return nan16, nan16.copy(), xp.zeros(nseg, xp.int64), ok, gv, pv
    u = gv - pv
    sig2 = xp.maximum(pv, 1.0) + (MODEL_ERR * pv) ** 2
    n = xp.bincount(sv, minlength=nseg)
    Bh = xp.nan_to_num(_seg_median(xp, sv, u, n))
    w = None
    den = None
    for _ in range(2):
        Bp = Bh[sv]
        Bpos = xp.maximum(Bp, 0.0)
        s2 = sig2 + Bpos + (REL_B * Bpos) ** 2
        z = (u - Bp) / xp.sqrt(s2)
        w = xp.minimum(1.0, HUBER_C / xp.maximum(xp.abs(z), 1e-9)) / sig2
        den = xp.bincount(sv, weights=w, minlength=nseg)
        Bh = xp.where(den > 0, xp.bincount(sv, weights=w * u, minlength=nseg)
                      / xp.maximum(den, 1e-30), 0.0)
    P = xp.bincount(sv, weights=w * pv, minlength=nseg) / xp.maximum(den, 1e-30)
    empty = n < NMIN
    D = xp.where(empty, xp.nan, xp.clip(Bh, -6e4, 6e4)).astype(xp.float16)
    P = xp.where(empty, xp.nan, xp.clip(P, 0, 6e4)).astype(xp.float16)
    return D, P, n, ok, gv, pv


@dataclass
class FrameStats:
    """Result of pass A (`frame_statistics`).

    D, P : (n_frames, 72, n_bins) float16. Huber location of obs - pred and
        mean prediction of each cell, in raw counts per pixel. NaN if the
        cell has fewer than 8 pixels.
    N : (n_frames, 72, n_bins) uint16 pixel count of each cell.
    obs_sector, pred_sector : (n_frames, 72) float64 sums of the observed and
        predicted raw counts over the pixels with a prediction.
    resid_radial, pred_radial, n_radial : (n_bins,) sums over all frames of
        obs - pred, of pred and the pixel counts.
    """
    D: np.ndarray
    P: np.ndarray
    N: np.ndarray
    obs_sector: np.ndarray
    pred_sector: np.ndarray
    resid_radial: np.ndarray
    pred_radial: np.ndarray
    n_radial: np.ndarray
    time_s: float = 0.0


def _stream(setup: ReconSetup, mask: RingMask | None):
    read = setup.reader()
    if mask is None:
        def fn(i):
            return read(setup.frames[i])
    else:
        def fn(i):
            return mask.apply(i, read(setup.frames[i]))
    return _prefetch(list(range(setup.n_frames)), fn, setup.prefetch)


def frame_statistics(setup: ReconSetup, model, *, mask: RingMask | None = None,
                     cells: FrameCells | None = None, use_gpu: bool = True,
                     progress: Callable | None = None, cancelled: Callable | None = None,
                     log: Callable | None = None) -> FrameStats:
    """Pass A: compare every frame of `setup` with the model volume.

    model : VolumeData of corrected intensities (for example
        `leave_one_out_median` of the current volume). It is sampled at the
        pixel hkl of each frame (`predict_frame`).
    mask : optional `RingMask`. Masked pixels are not used.
    Pixels < 0 and pixels at or above `setup.hot` are not used.
    Returns a `FrameStats`. Memory: 6 bytes per cell on the host (D, P, N).
    The device holds the model volume, 12 bytes per pixel and the
    temporary arrays of one frame."""
    say = log or (lambda *_: None)
    xp = get_xp(use_gpu, setup.device)
    cells = cells or FrameCells(PixelPolar.of(setup.detector))
    r_lab = _lab_vectors(setup)
    corr = _corrections32(setup, r_lab)
    pr = _Predictor(xp, setup, model, xp.asarray(r_lab), None if corr is None else xp.asarray(corr))
    del r_lab
    seg = xp.asarray(cells.seg)
    sec = xp.asarray(cells.sec)
    rb = xp.asarray(cells.rb)
    nf, nb, nseg = setup.n_frames, cells.nbin, cells.nseg
    D = np.full((nf, NSEC, nb), np.nan, np.float16)
    P = np.full((nf, NSEC, nb), np.nan, np.float16)
    N = np.zeros((nf, NSEC, nb), np.uint16)
    obs_s = np.zeros((nf, NSEC))
    pred_s = np.zeros((nf, NSEC))
    res_r = np.zeros(nb)
    pred_r = np.zeros(nb)
    n_r = np.zeros(nb, np.int64)
    hot = setup.hot
    t0 = time.time()
    for i, img in enumerate(_stream(setup, mask)):
        _cancel_check(cancelled)
        pred = pr(i)
        g = xp.asarray(np.asarray(img).ravel()).astype(xp.float32)
        if hot is not None:
            g = xp.where(g >= hot, -1.0, g)
        Dg, Pg, ng, ok, gv, pv = _frame_cells_stats(xp, g, pred, seg, nseg)
        if gv.size == 0:
            # No usable pixel: D, P stay NaN and the counts and sums stay 0.
            if progress:
                progress(i + 1, nf)
            continue
        D[i] = to_host(Dg).reshape(NSEC, nb)
        P[i] = to_host(Pg).reshape(NSEC, nb)
        N[i] = np.minimum(to_host(ng), 65535).reshape(NSEC, nb)
        s_ok, b_ok = sec[ok], rb[ok]
        gv64, pv64 = gv.astype(xp.float64), pv.astype(xp.float64)
        obs_s[i] = to_host(xp.bincount(s_ok, weights=gv64, minlength=NSEC))
        pred_s[i] = to_host(xp.bincount(s_ok, weights=pv64, minlength=NSEC))
        res_r += to_host(xp.bincount(b_ok, weights=gv64 - pv64, minlength=nb))
        pred_r += to_host(xp.bincount(b_ok, weights=pv64, minlength=nb))
        n_r += to_host(xp.bincount(b_ok, minlength=nb))
        if progress:
            progress(i + 1, nf)
    del pr, seg, sec, rb
    _free(xp)
    dt = time.time() - t0
    say(f"pass A: {nf} frames in {dt:.1f} s")
    return FrameStats(D, P, N, obs_s, pred_s, res_r, pred_r, n_r, dt)


# ---------------------------------------------------------------- frame map
# FRAME_MAP_DEFINITION and FRAME_MAP_FLOOR are in `averaging`. The Averaging
# panel then needs no import of rawrecon at start-up.


@dataclass
class FrameMap:
    """Agreement of the frames with a first average.

    excess : (n_frames, 72) observed / predicted raw counts of each frame and
        5-degree sector (NaN without prediction). These are the sums of
        pass A. They are not normalised.
    r_px, q : (n_bins,) radius (pixels) and |q| (1/Angstrom) of the radial bins.
    resid : (n_bins,) mean of obs - pred over all frames, raw counts per pixel.
    rel_resid : (n_bins,) sum(obs - pred) / sum(pred) of each radial bin.
    rings : ring radii (pixels) of the ring detection, or an empty list.

    `normalised()` gives the display values (see FRAME_MAP_DEFINITION).
    On real data the median of `excess` can differ from 1 by 5 % or more.
    Each effect has a direction. Real changes of the intensity with the
    frame and the sector can lower the ratio. Absorption or a shadow are
    possible causes, not tested. The sector sums contain Bragg peaks. The
    cells at Bragg nodes have a low median ratio and a long upper tail,
    because a peak is in only a few frames. The median of the orbit members
    is lower than their mean when the values have a long upper tail. This
    raises the ratio. The normalised ratio removes the part that is the
    same in all frames.
    """
    excess: np.ndarray
    r_px: np.ndarray
    q: np.ndarray
    resid: np.ndarray
    rel_resid: np.ndarray
    rings: list = field(default_factory=list)

    def sector_median(self) -> np.ndarray:
        """(72,) median of `excess` over the frames of each sector. NaN if no value is above 0."""
        ex = np.where(self.excess > 0, self.excess, np.nan).astype(np.float64)
        out = np.full(ex.shape[1], np.nan)
        ok = np.isfinite(ex).any(axis=0)
        out[ok] = np.nanmedian(ex[:, ok], axis=0)
        return out

    def normalised(self) -> np.ndarray:
        """(n_frames, 72) excess / sector median (FRAME_MAP_DEFINITION).

        A cell without observed counts (ratio 0) gets FRAME_MAP_FLOOR, so an
        empty frame shows as the lowest value. A cell without a prediction,
        or a sector without a median, is NaN."""
        with np.errstate(invalid="ignore", divide="ignore"):
            out = self.excess / self.sector_median()[None, :]
            keep = np.isfinite(out) & (out >= 0)
        return np.where(keep, np.maximum(out, FRAME_MAP_FLOOR), np.nan).astype(np.float32)

    def summary(self) -> dict:
        """Numbers of the frame map (FRAME_MAP_DEFINITION).

        excess_* : median, min and max of the ratio before normalisation.
        n_frames : number of frames (rows of the map).
        frames_off_10pct : frames whose median normalised ratio over the
            sectors is below 0.9 or above 1.1 (of n_frames).
        cell_dev_median : median of |normalised ratio - 1| over all cells
            (the typical scatter of one cell).
        rings_px : ring radii (pixels) of the ring detection (list)."""
        ex = self.excess[np.isfinite(self.excess)]
        nr = self.normalised()
        rows = np.isfinite(nr).any(axis=1)
        fmed = np.full(nr.shape[0], np.nan)
        fmed[rows] = np.nanmedian(nr[rows], axis=1)
        dev = np.abs(nr[np.isfinite(nr)] - 1.0)
        return dict(excess_median=float(np.median(ex)) if ex.size else None,
                    excess_max=float(ex.max()) if ex.size else None,
                    excess_min=float(ex.min()) if ex.size else None,
                    n_frames=int(self.excess.shape[0]),
                    frames_off_10pct=int((np.abs(fmed[rows] - 1.0) > 0.1).sum()),
                    cell_dev_median=float(np.median(dev)) if dev.size else None,
                    rings_px=list(self.rings))


def frame_map_from_stats(setup: ReconSetup, stats: FrameStats, rings=None) -> FrameMap:
    """`FrameMap` from the sums of pass A."""
    with np.errstate(invalid="ignore", divide="ignore"):
        excess = np.where(stats.pred_sector > 0, stats.obs_sector / stats.pred_sector, np.nan)
        resid = np.where(stats.n_radial > 0, stats.resid_radial / np.maximum(stats.n_radial, 1),
                         np.nan)
        rel = np.where(stats.pred_radial > 0, stats.resid_radial / stats.pred_radial, np.nan)
    r_px = np.arange(stats.resid_radial.size) + 0.5
    return FrameMap(excess.astype(np.float32), r_px, q_of_radius(setup.detector, r_px),
                    resid.astype(np.float32), rel.astype(np.float32), list(rings or []))


def frame_map(setup: ReconSetup, volume, laue_group: str, *, mask: RingMask | None = None,
              rings=None, use_gpu: bool = True, progress=None, cancelled=None,
              log=None) -> FrameMap:
    """Frame map of the reconstructed `volume` (VolumeData, not symmetrised).

    The first average is the leave-one-out orbit median of `volume`
    (`leave_one_out_median`). Pass A compares each frame with it."""
    pv = _with_intensity(volume, leave_one_out_median(volume, laue_group))
    stats = frame_statistics(setup, pv, mask=mask, use_gpu=use_gpu, progress=progress,
                             cancelled=cancelled, log=log)
    return frame_map_from_stats(setup, stats, rings)


def _with_intensity(vol, data):
    import dataclasses
    return dataclasses.replace(vol, intensity=data, variance=None, counts=None,
                               metadata=dict(vol.metadata))


# ---------------------------------------------------------------- corrected reconstruction
def frame_function(cells: FrameCells, model: FrameModel | None = None,
                   mask: RingMask | None = None) -> Callable:
    """`frame_fn` for the engine: the ring mask and the frame model of each frame.

    For frame position i it returns None (no change) or (offset, gain) with
    offset = B and gain = 1 / S (NaN on masked pixels). The engine then bins
    g (obs - B) / S with the variance g^2 max(obs, 1) / S^2."""
    fS = set() if model is None else set(model.frames_S().tolist())
    fB = set() if model is None else set(model.frames_B().tolist())

    def fn(i, img):
        m = None if mask is None else mask.pixel_mask(i)
        if m is None and i not in fS and i not in fB:
            return None
        offset = cells.B_image(model.B[i]) if i in fB else None
        gain = 1.0 / cells.S_image(model.a[i]).astype(np.float64) if i in fS else None
        if m is not None:
            gain = np.where(m, np.nan, 1.0 if gain is None else gain)
        return offset, gain
    return fn


def corrected_reconstruction(setup: ReconSetup, *, model: FrameModel | None = None,
                             mask: RingMask | None = None, cells: FrameCells | None = None,
                             groups=None, use_gpu: bool = True, progress=None,
                             cancelled=None) -> Volume:
    """Reconstruction with the ring mask and the frame model applied to each pixel.

    The value of a pixel is g (obs - B) / S. Its variance is
    g^2 max(obs, 1) / S^2: the Poisson variance of the raw count. The error of
    the model itself is not in the variance. Masked pixels are not used.
    `groups` makes partial volumes in the same pass (see `engine`)."""
    cells = cells or FrameCells(PixelPolar.of(setup.detector))
    fn = frame_function(cells, model, mask) if (model is not None or mask is not None) else None
    return setup.reconstruct(groups=groups, frame_fn=fn, use_gpu=use_gpu,
                             progress_callback=progress, cancelled=cancelled)


# ---------------------------------------------------------------- the frame chain
CHAIN_VERSION = "ring mask, frame model (3 iterations), plain orbit mean"


def _jsonable(x):
    """The value with NumPy scalars and arrays changed to Python values (for JSON)."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if np.isfinite(v) else None
    return x


@dataclass
class FrameChainResult:
    """Result of `run_frame_chain`.

    volume : VolumeData, the plain orbit mean of `corrected` (the final result).
    corrected : VolumeData, the corrected volume before the orbit mean.
    halves : [odd, even] corrected VolumeData (same tables), or None.
    masked : VolumeData with the ring mask only (no frame model), or None.
    masked_halves : [odd, even] of `masked`, or None.
    rings : `RingDetection`, or None.
    model : the `FrameModel` of the last iteration (only with keep_tables=True).
    frame_map : `FrameMap` of the first iteration, or None.
    info : JSON values (the same as volume.metadata['frame_tools']).
    """
    volume: Any
    corrected: Any
    halves: list | None = None
    masked: Any = None
    masked_halves: list | None = None
    rings: RingDetection | None = None
    model: FrameModel | None = None
    frame_map: FrameMap | None = None
    info: dict = field(default_factory=dict)


def _correction_name(mask, frame_model) -> str:
    if mask is not None and frame_model:
        return "ring_mask+frame_model"
    if frame_model:
        return "frame_model"
    return "ring_mask" if mask is not None else "none"


def run_frame_chain(setup: ReconSetup, laue_group: str, *, rings: Any = "auto",
                    frame_model: bool = True, iterations: int = 3, halves: bool = False,
                    raw: Any = None, options: ModelOptions | None = None,
                    use_gpu: bool = True, keep_tables: bool = False, keep_masked: bool = True,
                    progress: Callable | None = None, cancelled: Callable | None = None,
                    log: Callable | None = None) -> FrameChainResult:
    """Run the frame chain on the frames of `setup`.

    The frame chain is: ring mask, frame model (3 iterations), plain orbit
    mean. It was checked with Laue group mmm and 0.2 degree frames. Other
    Laue groups and other frame widths were not checked on real data.

    1. Ring detection on the frame statistics (rings='auto'), or the
       `RingMask` that you give, or no mask (rings=None).
    2. The volume with the ring mask.
    3. `iterations` times (default 3): the prediction is the leave-one-out
       orbit median of the current volume. Pass A compares every frame with
       it. The frame model is fitted again from the original observations.
       The corrected volume is made again from the frames. The tables of an
       iteration replace those of the iteration before.
    4. The plain orbit mean (equal weights, no rejection) of the corrected
       volume in `laue_group`.

    If no ring is found, step 2 gives the plain volume and the chain is the
    frame model alone. frame_model=False stops after step 2 (ring mask only).
    halves=True also makes the odd-frame and even-frame volumes of the last
    pass (the tables are not fitted again for each half).
    raw : the plain volume of the same setup (Volume or VolumeData), if you
        have it. The chain uses it when there is no mask and no halves, and
        to measure the masked pixel fraction.
    options : `ModelOptions` of the frame model. If you give options,
        options.frame_scale sets the frame windows of the ring detection and
        of the frame model. Only for options=None is the scale
        0.2 / frame width. info['frame_scale'] is the scale that the chain used.
    halves=True needs odd and even frame numbers. Else the chain raises
        ValueError.
    use_gpu : use the GPU (CuPy) if it is available (default True).
    keep_tables : keep the frame model of the last iteration in .model
        (default False).
    keep_masked : keep the volume with the ring mask only in .masked and
        its halves in .masked_halves (default True). If False, .masked_halves
        is None. .masked is then also None when frame_model is True. With
        frame_model=False, .masked is always kept.
    progress(done, total) counts frames over all passes. cancelled() is
    called once per frame and stops the chain with `averaging.Cancelled`.
    log(msg) receives the status lines.

    Memory: the host holds D, P (float16) and N (uint16) for each frame,
    sector and radial bin (6 bytes per cell, 1.3 GB for 1735 frames and
    1785 bins). The table fit needs more memory than the tables. Some
    volumes of the grid are also kept. Measured case (GPU, 1735 frames,
    1785 radial bins, grid 297 x 297 x 241, keep_tables=True, no raw, no
    halves): the peak resident memory of the process was 6.0 GB above the
    memory before the call. The peak is in the table fit. Between the
    passes the chain held 3.1 to 3.4 GB above that memory.
    The frames are streamed: no pass holds all frames. The GPU holds one
    volume, the accumulators of one pass and the arrays of one frame.
    """
    import dataclasses
    from ..volume_builder import symmetrize_volume
    say = log or (lambda *_: None)
    t_all = time.time()
    nf = setup.n_frames
    opts = options or ModelOptions(frame_scale=frame_scale_of(setup.increment))
    fs = opts.frame_scale
    detect = isinstance(rings, str)
    if detect and rings != "auto":
        raise ValueError("rings must be 'auto', None or a RingMask")
    groups = _checked_halves(setup) if halves else None
    reuse_raw = raw is not None and not halves and not detect and rings is None
    n_pass = int(detect) + int(not reuse_raw) + (2 * int(iterations) if frame_model else 0)
    total = max(1, n_pass * nf)
    done = [0]

    def stage(name):
        say(name)
        base = done[0]
        done[0] += nf

        def cb(i, n):
            if progress:
                progress(base + i, total)
        return cb

    polar = PixelPolar.of(setup.detector)
    cells = FrameCells(polar)
    times: dict[str, float] = {}
    mask, detection, ring_info = None, None, {"mode": "none"}
    if detect:
        t0 = time.time()
        rs = ring_statistics(setup, use_gpu=use_gpu, progress=stage("Ring statistics ..."),
                             cancelled=cancelled, log=log, polar=polar)
        detection = detect_rings(rs, frame_scale=fs)
        ring_info = dict(mode="detected", **detection.summary(rs.n1))
        del rs
        if detection.found:
            mask = RingMask.from_detection(polar, detection)
        times["rings_s"] = time.time() - t0
        say(f"rings: {ring_info['ring_windows']} event(s), {ring_info['ring_frames']} frames, "
            f"{ring_info['ring_entries']} ring entries")
    elif rings is not None:
        mask = rings
        ring_info = dict(mode="given", **mask.summary())
    _cancel_check(cancelled)

    source = setup.meta.get("folder")
    base_meta = {k: setup.meta[k] for k in ("polarization_axis", "geometry_method")
                 if setup.meta.get(k) is not None}
    if setup.hot is not None:
        base_meta["hot_pixel_cutoff"] = float(setup.hot)

    def to_vd(vol, corr_name):
        meta = dict(base_meta, frame_correction=corr_name)
        vd = reconstruct_to_volumedata(vol, source, meta)
        parts = None
        if vol.parts is not None:
            parts = [reconstruct_to_volumedata(p, source, dict(meta)) for p in vol.parts]
        return vd, parts

    t0 = time.time()
    masked_parts = None
    if reuse_raw:
        masked_vd = (to_vd(raw, "none")[0] if isinstance(raw, Volume)
                     else dataclasses.replace(raw, metadata=dict(raw.metadata)))
    else:
        vol0 = corrected_reconstruction(setup, mask=mask, cells=cells, groups=groups,
                                        use_gpu=use_gpu, cancelled=cancelled,
                                        progress=stage("Volume with the ring mask ..."))
        masked_vd, masked_parts = to_vd(vol0, "ring_mask" if mask is not None else "none")
        del vol0
    times["masked_volume_s"] = time.time() - t0
    if raw is not None and mask is not None:
        c_raw = np.asarray(raw.counts, np.float64).sum()
        c_mask = np.asarray(masked_vd.counts, np.float64).sum()
        ring_info["masked_pixel_frac_total"] = float(1.0 - c_mask / c_raw)

    cur, cur_parts, model, fmap, it_rows = masked_vd, masked_parts, None, None, []
    if frame_model:
        ring_r = sorted({float(t["r"]) for _, _, tab in (detection.windows if detection else [])
                         for t in tab})
        for it in range(1, int(iterations) + 1):
            _cancel_check(cancelled)
            t0 = time.time()
            pred_vol = leave_one_out_median(cur, laue_group)
            if not np.isfinite(pred_vol).any():
                raise ValueError(
                    f"The prediction has no finite voxel. In Laue group {laue_group}, no voxel "
                    f"of the grid has a symmetry equivalent with data in the grid. Use a grid "
                    f"that contains the symmetry equivalents.")
            pv = _with_intensity(cur, pred_vol)
            del pred_vol
            t_pred = time.time() - t0
            stats = frame_statistics(setup, pv, mask=mask, cells=cells, use_gpu=use_gpu,
                                     cancelled=cancelled, log=log,
                                     progress=stage(f"Iteration {it}: pass A ..."))
            del pv
            if it == 1:
                fmap = frame_map_from_stats(setup, stats, ring_r)
            t1 = time.time()
            model = fit_frame_model(stats.D, stats.P, stats.N, opts, use_gpu=use_gpu,
                                    device=setup.device, log=log)
            t_fit = time.time() - t1
            t_a = stats.time_s
            del stats
            _cancel_check(cancelled)
            t1 = time.time()
            last = it == int(iterations)
            vol = corrected_reconstruction(setup, model=model, mask=mask, cells=cells,
                                           groups=groups if last else None, use_gpu=use_gpu,
                                           cancelled=cancelled,
                                           progress=stage(f"Iteration {it}: corrected volume ..."))
            cur, cur_parts = to_vd(vol, _correction_name(mask, True))
            del vol
            it_rows.append(dict(iteration=it, prediction_s=t_pred, pass_a_s=t_a, fit_s=t_fit,
                                corrected_s=time.time() - t1, **model.info))
    t0 = time.time()
    final = symmetrize_volume(cur, laue_group, sigma=None, use_gpu=use_gpu)
    times["mean_s"] = time.time() - t0
    times["total_s"] = time.time() - t_all
    corr_name = _correction_name(mask, frame_model)
    info = dict(
        method=CHAIN_VERSION if (frame_model and int(iterations) == 3) else "frame tools",
        correction=corr_name, laue_group=laue_group,
        iterations=int(iterations) if frame_model else 0,
        frame_scale=fs, options=dataclasses.asdict(opts), rings=ring_info,
        prediction="leave-one-out orbit median of the current volume",
        final_average="plain orbit mean (equal weights, no rejection)",
        variance="Poisson variance of the raw counts times (g / S)^2. The model error is not included.",
        iterations_info=it_rows, frame_map=fmap.summary() if fmap is not None else None,
        halves_rule=("frame-number parity: odd numbers, then even numbers" if halves else None),
        n_frames=nf, finite_frac=float(np.isfinite(final.intensity).mean()), times=times)
    info = _jsonable(info)
    for vd in [final, cur] + list(cur_parts or []):
        vd.metadata["frame_correction"] = corr_name
        vd.metadata["frame_tools"] = info
    if frame_model:
        masked_info = _jsonable(dict(correction=masked_vd.metadata.get("frame_correction", "none"),
                                     rings=ring_info))
        for vd in [masked_vd] + list(masked_parts or []):
            vd.metadata["frame_tools"] = masked_info
    say(f"frame chain done in {times['total_s']:.0f} s ({corr_name})")
    return FrameChainResult(
        volume=final, corrected=cur, halves=cur_parts,
        masked=masked_vd if (keep_masked or not frame_model) else None,
        masked_halves=masked_parts if keep_masked else None, rings=detection,
        model=model if keep_tables else None, frame_map=fmap, info=info)
