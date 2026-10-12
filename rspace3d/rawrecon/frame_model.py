"""Frame model: an additive table B and a scale table S for each frame.

The model describes contamination that changes from frame to frame (for
example a wire or a powder ring that is in the beam for some frames, or a
shadow). The model uses frame f, azimuth sector s (5 degrees) and radial bin
b (1 pixel). The model of a raw count is

    obs = S(f, s) * true + B(f, s, b) ,   S = 1 + a .

`fit_frame_model` finds a and B from the residual statistics of pass A
(`frames.frame_statistics`). The thresholds are noise thresholds that the
method measures from the data, so a clean frame gets a = 0 and B = 0.

Steps of the fit (for each cell, "frames" is the axis of the frames):
1. The median over the frames of the residual D is the baseline. The method
   removes it.
2. a(f, s): a weighted median polish over the radial bins. Then a median
   filter, a significance test against the frame scatter, and an opening.
   The method keeps a >= -0.6.
3. B(f, s, b): a radial gate (significant radial bins of the sector median)
   OR connected regions of significant smoothed cells (hysteresis in 3D).
   B is the cell residual inside the gate or the regions, at least 0.
   B is exactly 0 outside them.

The tables are stored as float16 (D, P, B) and float32 (a). The thresholds
act on these values. A float32 table would change which cells pass.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

NSEC = 72            # azimuth sectors (5 degrees)
NMIN = 8             # minimum pixels of a cell
PMIN = 0.5           # minimum mean prediction (raw counts) of a cell for the scale term
HUBER_C = 2.0        # Huber threshold of pass A
MODEL_ERR = 0.05     # relative error of the prediction in the pixel variance
REL_B = 0.10         # relative scatter of an excess inside a cell
A_FLOOR = 0.003      # floor of the frame scatter of a
R_FLOOR = 0.05       # floor of the frame scatter of D (raw counts)


@dataclass(frozen=True)
class ModelOptions:
    """Options of the frame model. The defaults are the values of the frame chain.

    Do not change them for a normal run: they were set on validation data
    (Laue group mmm, 0.2 degree frames).
    frame_scale converts the frame-window sizes (set for 0.2 degree frames)
    to other frame widths: a window of n frames becomes round(n * frame_scale).
    If you give options to `frames.run_frame_chain`, set frame_scale to
    0.2 / frame width yourself. The chain uses it for the ring detection too.
    """
    k_B: float = 5.0          # threshold of the radial gate (frame-MAD units)
    k_a: float = 5.0          # threshold of the scale term
    k_lo_B: float = 5.0       # low threshold of the hysteresis
    k_seed_B: float = 8.0     # seed threshold of the hysteresis
    min_cells_B: int = 2000   # minimum cells of a kept region
    rel_floor: float = 0.25   # model-error floor: sigma^2 = mad^2 + (rel_floor * P)^2
    a_floor: float = -0.6     # a >= a_floor, so S >= 0.4
    frame_scale: float = 1.0

    def frames(self, n: int) -> int:
        """Number of frames of a window of n frames at 0.2 degrees per frame."""
        return max(1, int(round(n * self.frame_scale)))


def frame_scale_of(increment: float | None) -> float:
    """Scale of the frame windows: 0.2 degrees / frame width (1 if the width is unknown)."""
    if not increment:
        return 1.0
    return 0.2 / abs(float(increment))


# ---------------------------------------------------------------- array helpers
def get_xp(use_gpu: bool, device: int = 0):
    """CuPy (on `device`) if use_gpu is True and a GPU is available, else NumPy."""
    if use_gpu:
        from .engine import has_cupy
        if has_cupy(device):
            import cupy as cp
            cp.cuda.Device(device).use()
            return cp
    return np


def to_host(a):
    """NumPy array of a NumPy or CuPy array."""
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _free(xp) -> None:
    if xp is not np:
        xp.get_default_memory_pool().free_all_blocks()


def nanmedian(xp, x, axis):
    """Median along `axis` without NaN: mean of the lower and upper middle values. NaN if no value."""
    xs = xp.sort(x, axis=axis)                       # NaN sorted last
    k = xp.sum(xp.isfinite(x), axis=axis, keepdims=True)
    lo = xp.maximum((k - 1) // 2, 0)
    hi = xp.maximum(k // 2, 0)
    a = xp.take_along_axis(xs, lo.astype(xp.int64), axis)
    b = xp.take_along_axis(xs, hi.astype(xp.int64), axis)
    return xp.where(k > 0, 0.5 * (a + b), xp.nan).squeeze(axis)


def _wmedian(xp, r, w):
    """Lower weighted median along the last axis. NaN values and weights <= 0 are not used."""
    bad = ~xp.isfinite(r) | (w <= 0)
    rr = xp.where(bad, xp.inf, r)
    ww = xp.where(bad, 0.0, w)
    idx = xp.argsort(rr, axis=-1)
    rs = xp.take_along_axis(rr, idx, -1)
    ws = xp.take_along_axis(ww, idx, -1)
    cw = xp.cumsum(ws, axis=-1)
    tot = cw[..., -1:]
    pos = xp.argmax(cw >= 0.5 * tot, axis=-1)[..., None]
    out = xp.take_along_axis(rs, pos, -1)[..., 0]
    return xp.where(tot[..., 0] > 0, out, xp.nan)


def mad_frames(xp, R, step=2048):
    """(1.4826 x MAD, median) over axis 0 (the frames) of R, without NaN. float32."""
    sh = R.shape
    flat = R.reshape(sh[0], -1)
    mad = np.full(flat.shape[1], np.nan, np.float32)
    med = np.full(flat.shape[1], np.nan, np.float32)
    for j in range(0, flat.shape[1], step):
        blk = xp.asarray(flat[:, j:j + step]).astype(xp.float32)
        m = nanmedian(xp, blk, 0)
        med[j:j + step] = to_host(m)
        mad[j:j + step] = to_host(1.4826 * nanmedian(xp, xp.abs(blk - m[None]), 0))
    return mad.reshape(sh[1:]), med.reshape(sh[1:])


def _open_wrap(sig, fr, az):
    """Binary opening of a (frames, sectors) map with an fr x az block. The azimuth wraps."""
    from scipy.ndimage import binary_opening
    pad = np.concatenate([sig[:, -az:], sig, sig[:, :az]], axis=1)
    return binary_opening(pad, structure=np.ones((fr, az), bool))[:, az:-az]


def _medfilt_wrap(a, fr, az):
    """Median filter of a (frames, sectors) map. The azimuth wraps."""
    from scipy.ndimage import median_filter
    pad = np.concatenate([a[:, -az:], a, a[:, :az]], axis=1)
    return median_filter(pad, size=(fr, az), mode="nearest")[:, az:-az]


def _hyst_keep(lo, seed, min_cells, az_axis, pad=2):
    """Connected regions of `lo` (full connectivity, azimuth wraps) with a seed and >= min_cells cells.

    The azimuth axis is padded by `pad` sectors at each end. The cells are
    counted in the padded array (the method was validated so)."""
    from scipy.ndimage import label
    nsec = lo.shape[az_axis]

    def _pad(a):
        first = np.take(a, range(pad), axis=az_axis)
        last = np.take(a, range(nsec - pad, nsec), axis=az_axis)
        return np.concatenate([last, a, first], axis=az_axis)

    lo_p, seed_p = _pad(lo), _pad(seed & lo)
    lab, nl = label(lo_p, structure=np.ones((3,) * lo.ndim, bool))
    size = np.zeros(nl + 1, np.int64)
    for i0 in range(0, lab.shape[0], 64):            # in blocks: bincount copies to int64
        size += np.bincount(lab[i0:i0 + 64].ravel(), minlength=nl + 1)
    nseed = np.bincount(lab[seed_p], minlength=nl + 1)
    ok = (size >= min_cells) & (nseed >= 1)
    ok[0] = False
    keep_p = ok[lab]
    del lab
    sl = [slice(None)] * lo.ndim
    mid = keep_p.take(range(pad, pad + nsec), axis=az_axis)
    sl[az_axis] = slice(0, pad)
    tail = keep_p[tuple(sl)]                           # copy of the last `pad` sectors
    sl[az_axis] = slice(nsec + pad, nsec + 2 * pad)
    head = keep_p[tuple(sl)]                           # copy of the first `pad` sectors
    out = mid.copy()
    idx_last = [slice(None)] * lo.ndim
    idx_last[az_axis] = slice(nsec - pad, nsec)
    out[tuple(idx_last)] |= tail
    idx_first = [slice(None)] * lo.ndim
    idx_first[az_axis] = slice(0, pad)
    out[tuple(idx_first)] |= head
    return out


def _smooth_cells(xp, R, size=(3, 9), chunk=48):
    """Box mean of R (nf, ns, nb) over (3 sectors, 9 bins) without NaN. float16.

    The azimuth wraps. The radial edges are padded with zeros. A cell is NaN
    if less than half of its box is finite."""
    if xp is np:
        from scipy.ndimage import uniform_filter
    else:
        from cupyx.scipy.ndimage import uniform_filter
    nf = R.shape[0]
    out = np.full(R.shape, np.nan, np.float16)
    pa = size[0] // 2
    for i0 in range(0, nf, chunk):
        r = xp.asarray(R[i0:i0 + chunk]).astype(xp.float32)
        r = xp.concatenate([r[:, -pa:], r, r[:, :pa]], axis=1)
        fin = xp.isfinite(r)
        num = uniform_filter(xp.where(fin, r, 0.0), size=(1,) + tuple(size), mode="constant")
        den = uniform_filter(fin.astype(xp.float32), size=(1,) + tuple(size), mode="constant")
        v = xp.where(den >= 0.5, num / xp.maximum(den, 1e-6), xp.nan)
        out[i0:i0 + chunk] = to_host(v[:, pa:-pa]).astype(np.float16)
    return out


# ---------------------------------------------------------------- the fit
@dataclass
class FrameModel:
    """Fitted frame model.

    a : (n_frames, 72) float32. S = 1 + a for each frame and sector.
    B : (n_frames, 72, n_bins) float16. Additive counts per raw pixel, >= 0.
    info : JSON values that describe the fit.
    """
    a: np.ndarray
    B: np.ndarray
    info: dict = field(default_factory=dict)

    def frames_S(self) -> np.ndarray:
        """Positions of the frames with a scale term."""
        return np.flatnonzero((self.a != 0).any(axis=1))

    def frames_B(self) -> np.ndarray:
        """Positions of the frames with an additive term."""
        return np.flatnonzero((self.B != 0).reshape(self.B.shape[0], -1).any(axis=1))


def fit_frame_model(D, P, N, options: ModelOptions | None = None, use_gpu: bool = True,
                    device: int = 0, log=None, chunk: int = 48) -> FrameModel:
    """Fit the tables a and B from the pass-A tables D, P, N (each (n_frames, 72, n_bins)).

    D is the Huber residual, P the mean prediction (both float16, raw
    counts per pixel) and N the pixel count (uint16) of each cell.
    The function returns a `FrameModel`. It does not change D, P or N."""
    from scipy.ndimage import binary_opening, binary_dilation
    o = options or ModelOptions()
    say = log or (lambda *_: None)
    xp = get_xp(use_gpu, device)
    nf, ns, nb = D.shape
    t0 = time.time()
    valid = N >= NMIN
    Dn = np.where(valid, D, np.float16(np.nan))
    del valid
    sig0, D0 = mad_frames(xp, Dn)                    # noise and baseline of each cell
    sig0 = np.maximum(sig0, R_FLOOR)
    D0g, s0g = xp.asarray(D0), xp.asarray(sig0)
    info: dict[str, Any] = dict(baseline_abs_mean=float(np.nanmean(np.abs(D0))),
                                baseline_med=float(np.nanmedian(D0)),
                                cell_sigma_median=float(np.nanmedian(sig0)))
    # ---- scale term a (weighted median polish, median filter, significance, opening)
    a_raw = np.zeros((nf, ns), np.float32)
    for i0 in range(0, nf, chunk):
        d = xp.asarray(Dn[i0:i0 + chunk]).astype(xp.float32) - D0g[None]
        p = xp.asarray(P[i0:i0 + chunk]).astype(xp.float32)
        nn = xp.asarray(N[i0:i0 + chunk]).astype(xp.float32)
        good = xp.isfinite(d) & xp.isfinite(p) & (p >= PMIN) & (nn >= NMIN)
        wgt = xp.where(good, (p / s0g[None]) ** 2, 0.0)
        pp = xp.where(good, p, 1.0)
        a = xp.zeros((d.shape[0], ns), xp.float32)
        for _ in range(3):
            B0 = xp.maximum(nanmedian(xp, d - a[:, :, None] * xp.nan_to_num(p), 1), 0.0)
            a = xp.nan_to_num(_wmedian(xp, (d - B0[:, None, :]) / pp, wgt))
        a_raw[i0:i0 + chunk] = to_host(a)
    del d, p, nn, good, wgt, pp, a, B0
    sig_a, med_a = mad_frames(xp, a_raw)
    a_sm = _medfilt_wrap(a_raw - med_a[None], 2 * o.frames(2) + 1, 3)
    sig_s, _ = mad_frames(xp, a_sm)
    sig_s = np.maximum(sig_s, A_FLOOR)
    za = a_sm / sig_s[None]
    keep = _open_wrap(np.abs(za) > o.k_a, 2 * o.frames(1) + 1, 5)
    a_app = np.where(keep, a_sm, 0.0).astype(np.float32)
    a_app = np.maximum(a_app, np.float32(o.a_floor))
    info["a_sigma_frames_median"] = float(np.median(sig_a))
    info["a_smoothed_sigma_median"] = float(np.median(sig_s))
    # ---- additive term: residual after the scale term, radial gate, sector regions
    Rrad = np.full((nf, nb), np.nan, np.float32)
    R = np.full((nf, ns, nb), np.nan, np.float16)
    for i0 in range(0, nf, chunk):
        d = xp.asarray(Dn[i0:i0 + chunk]).astype(xp.float32) - D0g[None]
        p = xp.nan_to_num(xp.asarray(P[i0:i0 + chunk]).astype(xp.float32))
        r = d - xp.asarray(a_app[i0:i0 + chunk])[:, :, None] * p
        Rrad[i0:i0 + chunk] = to_host(nanmedian(xp, r, 1))
        R[i0:i0 + chunk] = to_host(r).astype(np.float16)
    del Dn, d, p, r, D0g, s0g
    _free(xp)
    sig_r, med_r = mad_frames(xp, Rrad)
    sig_r = np.maximum(sig_r, 0.02)
    zr = np.nan_to_num((Rrad - med_r[None]) / sig_r[None], nan=0.0)
    gate = binary_opening(zr > o.k_B, structure=np.ones((2 * o.frames(1) + 1, 5), bool))
    gate = binary_dilation(gate, structure=np.ones((1, 7), bool))   # (nf, nb) radial bins with an excess
    Rs = _smooth_cells(xp, R)
    sig_c, med_c = mad_frames(xp, Rs)
    sig_c = np.maximum(sig_c, 0.02)
    Ps = _smooth_cells(xp, P)
    seed = np.zeros(R.shape, bool)
    lo = np.zeros(R.shape, bool)
    for i0 in range(0, nf, chunk):
        se = np.sqrt(sig_c[None] ** 2
                     + (o.rel_floor * np.nan_to_num(Ps[i0:i0 + chunk].astype(np.float32))) ** 2)
        z_ = (Rs[i0:i0 + chunk].astype(np.float32) - med_c[None]) / se
        seed[i0:i0 + chunk] = z_ > o.k_seed_B
        lo[i0:i0 + chunk] = z_ > o.k_lo_B
    del Rs, Ps
    region = _hyst_keep(lo, seed, o.min_cells_B, az_axis=1)
    del seed, lo
    B_tab = np.zeros((nf, ns, nb), np.float16)
    for i0 in range(0, nf, chunk):
        g = gate[i0:i0 + chunk, None, :] | region[i0:i0 + chunk]
        Rc_ = R[i0:i0 + chunk].astype(np.float32)
        fill = np.maximum(Rrad[i0:i0 + chunk] - med_r[None], 0.0)[:, None, :]   # cells without data
        B_tab[i0:i0 + chunk] = np.where(g, np.maximum(np.where(np.isfinite(Rc_), Rc_,
                                                               np.nan_to_num(fill)), 0.0),
                                        0.0).astype(np.float16)
    del R
    _free(xp)
    fB = gate.any(1) | region.any((1, 2))
    fS = (a_app != 0).any(1)
    info.update(frames_B=int(fB.sum()), frames_S=int(fS.sum()), gate_bins=int(gate.sum()),
                region_cells=int(region.sum()),
                B_sum_counts=float(B_tab.astype(np.float64).sum()), B_max=float(B_tab.max()),
                a_min=float(a_app.min()), a_max=float(a_app.max()),
                radial_sigma_median=float(np.median(sig_r)), tables_time_s=time.time() - t0)
    say(f"frame model: {info['frames_S']} frames with a scale term, {info['frames_B']} frames "
        f"with an additive term, a in [{info['a_min']:.3f}, {info['a_max']:.3f}]")
    return FrameModel(a_app, B_tab, info)


# ---------------------------------------------------------------- tables -> pixel images
def interp_B(T, fa, fb, pad=3):
    """Value of the cell table T (72, n_bins) at each pixel. float32, >= 0.

    The cells hold means over 1 px x 5 degrees. A 4th-order deconvolution
    f = T - laplacian(T) / 24 turns them into point values. A cubic spline
    then interpolates them (the azimuth wraps). fa and fb are the cell
    coordinates of the pixels (cell centres at integers)."""
    from scipy.ndimage import map_coordinates
    T = np.asarray(T, np.float32)
    Tp = np.concatenate([T[-pad:], T, T[:pad]], axis=0).astype(np.float64)
    f = Tp.copy()
    lap = np.zeros_like(Tp)
    lap[1:-1, :] += Tp[2:] - 2 * Tp[1:-1] + Tp[:-2]
    lap[:, 1:-1] += Tp[:, 2:] - 2 * Tp[:, 1:-1] + Tp[:, :-2]
    f -= lap / 24.0
    v = map_coordinates(f, [fa + pad, fb], order=3, mode="nearest")
    return np.maximum(v, 0.0).astype(np.float32)


def interp_S(a, ta, a0, a1):
    """S = 1 + a at each pixel, linear in azimuth between the sector centres. float32."""
    a = np.asarray(a, np.float32)
    return 1.0 + (1.0 - ta) * a[a0] + ta * a[a1]
