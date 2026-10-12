"""Powder-ring detector and ring mask for raw frames.

A thin powder ring (for example from a wire or a sample holder in the beam
for some frames) is a narrow peak in the radial profile of a frame. The
detector finds such rings from the frame data only:

1. `ring_statistics` makes one pass over the frames. For each frame it
   measures the trimmed mean of each 1-pixel annulus. It also measures the
   trimmed mean of each 30-degree sector of each annulus. The trimmed mean
   uses the lowest 85 % of the pixels. This removes Bragg peaks. The
   function converts the trimmed means to Poisson means (raw counts per
   pixel).
2. `detect_rings` removes the smooth radial background and the slow change
   with time. It finds frames with narrow peaks at 4 or more radii. It groups
   these frames into events. It keeps the ring radii that show in many
   azimuth sectors and that have the time course of the event. A ring must
   show a peak in 5 of the 12 sectors, or in 60 % of the valid sectors if
   that is fewer. The minimum is 2 sectors. For each frame of an event the
   function gives the ring centres and half widths in pixels.
3. `RingMask` gives the pixel mask of each frame: a pixel in a ring band is
   not used.

You can also give the ring bands yourself (`RingMask.from_bands`).
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from .engine import _prefetch
from .frame_model import get_xp, to_host, _free

NSEC_RING = 12       # azimuth sectors of the ring statistics (30 degrees)
TRIM = 0.85          # trimmed fraction of the pixel values
MIN_PIX = 40         # minimum pixels of an annulus
MIN_PIX_SECTOR = 20  # minimum pixels of a sector annulus
LUT_RES = 4          # ring-mask look-up table: 4 steps per pixel of radius


# ---------------------------------------------------------------- pixel radius and azimuth
@dataclass
class PixelPolar:
    """Radius (pixels from the beam centre) and image azimuth (degrees) of each pixel.

    The pixel order is the order of the engine (row-major, slow then fast).
    The azimuth is 0 along +x (fast) and 90 along +y (slow). The detector of
    rawrecon is normal to the beam, so a powder ring is a circle of constant r.
    """
    shape: tuple
    r: np.ndarray        # (npix,) float32
    az: np.ndarray       # (npix,) float32

    @classmethod
    def of(cls, detector) -> "PixelPolar":
        ny, nx = detector.shape
        bcx, bcy = detector.beam_center
        ys, xs = np.mgrid[0:ny, 0:nx]
        r = np.hypot(xs - bcx, ys - bcy).astype(np.float32)
        az = np.degrees(np.arctan2(ys - bcy, xs - bcx)).astype(np.float32)
        return cls((ny, nx), r.ravel(), az.ravel())

    @property
    def nbin(self) -> int:
        """Number of 1-pixel radial bins."""
        return int(np.floor(self.r.max())) + 1


def sector_of_az(az, nsec):
    """Sector of an image azimuth (degrees). Counter-clockwise with y up. Sector k is centred on k * 360 / nsec."""
    th = (-np.asarray(az, float)) % 360.0
    return (np.floor(((th + 180.0 / nsec) % 360.0) / (360.0 / nsec))).astype(np.int64)


def radius_of_q(detector, q):
    """Ring radius in pixels of |q| = 2 sin(theta) / lambda (1/Angstrom)."""
    s = np.asarray(q, float) * detector.wavelength / 2.0
    tth = 2.0 * np.arcsin(np.clip(s, -1.0, 1.0))
    return detector.distance * np.tan(tth) / detector.pixel_size


def q_of_radius(detector, r):
    """|q| = 2 sin(theta) / lambda (1/Angstrom) of a ring radius in pixels."""
    tth = np.arctan2(np.asarray(r, float) * detector.pixel_size, detector.distance)
    return 2.0 * np.sin(tth / 2.0) / detector.wavelength


# ---------------------------------------------------------------- statistics pass
def poisson_trim_table(frac=TRIM, lam_max=400.0, n=900):
    """T(lambda): mean of the lowest `frac` of the probability of Poisson(lambda). Returns (lambda, T)."""
    from scipy.stats import poisson
    lam = np.concatenate([[0.0], np.geomspace(1e-3, lam_max, n)])
    xmax = int(lam_max + 12 * np.sqrt(lam_max) + 20)
    x = np.arange(xmax + 1)
    T = np.zeros_like(lam)
    for i, val in enumerate(lam):
        if val == 0:
            continue
        F = poisson.cdf(x, val)
        Fm = np.concatenate([[0.0], F[:-1]])
        mass = np.clip(np.minimum(F, frac) - Fm, 0.0, None)
        T[i] = float((x * mass).sum() / frac)
    return lam, T


_TAB: dict = {}


def invert_trim(tm, frac=TRIM):
    """Poisson mean of a trimmed mean `tm` (NaN stays NaN)."""
    if frac not in _TAB:
        _TAB[frac] = poisson_trim_table(frac)
    lam, T = _TAB[frac]
    out = np.interp(tm, T, lam)
    hi = tm > T[-1]                     # above the table: slope-one continuation
    if np.any(hi):
        out = np.where(hi, lam[-1] + (tm - T[-1]), out)
    return np.where(np.isfinite(tm), out, np.nan)


def _trim_stats(xp, seg, v, nseg, frac):
    """Trimmed mean (lowest floor(frac n) values) and count n of each segment. Exact integer sums."""
    # cupy.bincount fails on an empty array (a frame with no valid pixel).
    n = xp.bincount(seg, minlength=nseg) if seg.size else xp.zeros(nseg, xp.int64)
    key = xp.sort((seg << 21) | v)
    vals = key & (2 ** 21 - 1)
    cs = xp.concatenate([xp.zeros(1, xp.int64), xp.cumsum(vals)])
    off = xp.cumsum(n) - n
    k = xp.floor(frac * n.astype(xp.float64)).astype(xp.int64)
    tsum = cs[off + k] - cs[off]
    tm = xp.where(k > 0, tsum.astype(xp.float64) / xp.maximum(k, 1), xp.float64("nan"))
    return tm, n


@dataclass
class RingStats:
    """Per-frame annulus statistics of `ring_statistics`.

    lam1 : (n_frames, n_bins) Poisson mean of each 1-pixel annulus (NaN if < 40 pixels).
    lams : (n_frames, 12, n_bins) the same for each 30-degree sector (NaN if < 20 pixels).
    n1 : (n_frames, n_bins) pixel count of each annulus.
    """
    lam1: np.ndarray
    lams: np.ndarray
    n1: np.ndarray
    time_s: float = 0.0

    @property
    def nbin(self) -> int:
        return self.lam1.shape[1]


def ring_statistics(setup, *, use_gpu=True, progress=None, cancelled=None, log=None,
                    polar: PixelPolar | None = None) -> RingStats:
    """One pass over the frames of `setup` (`engine.ReconSetup`): the annulus statistics.

    The pass uses the frames without a mask. Pixels < 0 and pixels at or
    above `setup.hot` are not used. Values are clipped to 2^21 - 1."""
    from .engine import _cancel_check
    say = log or (lambda *_: None)
    xp = get_xp(use_gpu, setup.device)
    polar = polar or PixelPolar.of(setup.detector)
    nbin = polar.nbin
    b1_h = np.minimum(np.floor(polar.r).astype(np.int64), nbin - 1)
    sec_h = sector_of_az(polar.az, NSEC_RING)
    b1 = xp.asarray(b1_h)
    seg1 = xp.asarray(sec_h * nbin + b1_h)
    nf = setup.n_frames
    tm1 = np.full((nf, nbin), np.nan, np.float32)
    n1 = np.zeros((nf, nbin), np.int32)
    tms = np.full((nf, NSEC_RING, nbin), np.nan, np.float32)
    ns1 = np.zeros((nf, NSEC_RING, nbin), np.int32)
    hot = setup.hot
    read = setup.reader()
    t0 = time.time()
    for i, img in enumerate(_prefetch(setup.frames, read, setup.prefetch)):
        _cancel_check(cancelled)
        g = xp.asarray(np.asarray(img).ravel())
        valid = g >= 0
        if hot is not None:
            valid &= g < hot
        v = xp.clip(g[valid].astype(xp.int64), 0, 2 ** 21 - 1)
        a, b = _trim_stats(xp, b1[valid], v, nbin, TRIM)
        tm1[i], n1[i] = to_host(a), to_host(b)
        a, b = _trim_stats(xp, seg1[valid], v, NSEC_RING * nbin, TRIM)
        tms[i], ns1[i] = to_host(a).reshape(NSEC_RING, nbin), to_host(b).reshape(NSEC_RING, nbin)
        if progress:
            progress(i + 1, nf)
    _free(xp)
    dt = time.time() - t0
    say(f"ring statistics: {nf} frames in {dt:.1f} s")
    lam1 = np.where(n1 >= MIN_PIX, invert_trim(tm1), np.nan).astype(np.float32)
    lams = np.where(ns1 >= MIN_PIX_SECTOR, invert_trim(tms), np.nan).astype(np.float32)
    return RingStats(lam1, lams, n1, dt)


# ---------------------------------------------------------------- detection
def _running_median(y, half):
    """Lower running median over axis 0 (window 2 half + 1, shorter at the ends). y has no NaN."""
    nf, nc = y.shape
    w = 2 * half + 1
    out = np.empty((nf, nc), np.float32)
    y = np.asarray(y, np.float32)
    lo_rows = list(range(min(half, nf))) + list(range(max(half, nf - half), nf))
    if nf >= w:
        win = np.lib.stride_tricks.sliding_window_view(y, w, axis=0)   # (nf - w + 1, nc, w)
        ch = max(1, int(2e8 // (nc * w * 4)))
        for a in range(0, win.shape[0], ch):
            part = np.partition(win[a:a + ch], half, axis=-1)
            out[half + a:half + a + part.shape[0]] = part[..., half]
    for f in sorted(set(lo_rows)):
        s = np.sort(y[max(0, f - half):min(nf, f + half + 1)], axis=0)
        out[f] = s[int(np.floor(0.5 * (s.shape[0] - 1)))]
    return out


def _runs(b):
    d = np.diff(np.concatenate([[0], np.asarray(b).astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _filter_runs(b, min_len, bridge=0):
    """1D bool: fill gaps of <= bridge frames, then remove runs shorter than min_len."""
    b = np.asarray(b, bool).copy()
    if bridge > 0:
        for (s, e) in _runs(~b):
            if s > 0 and e < b.size and (e - s) <= bridge:
                b[s:e] = True
    for (s, e) in _runs(b):
        if e - s < min_len:
            b[s:e] = False
    return b


def _gauss(x, a, c, s, o):
    return a * np.exp(-0.5 * ((x - c) / s) ** 2) + o


def _fill_rows(a):
    """NaN along axis 1 replaced by linear interpolation in each row. All-NaN rows become 0."""
    a = np.array(a, np.float32)
    x = np.arange(a.shape[1])
    for i in np.flatnonzero(~np.isfinite(a).all(axis=1)):
        ok = np.isfinite(a[i])
        a[i] = np.interp(x, x[ok], a[i, ok]) if ok.any() else 0.0
    return a


@dataclass
class RingDetection:
    """Result of `detect_rings`.

    masks : {frame position: (centres px, half widths px)} for the frames of a ring event.
    windows : list of (first, end, rings) per kept event (positions, the end is excluded).
        rings is a list of dicts (r px, sigma px, zsum, nfr, nsec, nsec_valid, rho).
    rejected : the candidate rings that failed a test.
    cnt : (n_frames,) number of significant narrow peaks of each frame.
    """
    masks: dict
    windows: list
    rejected: list = field(default_factory=list)
    cnt: np.ndarray | None = None

    @property
    def found(self) -> bool:
        return bool(self.masks)

    def summary(self, n1=None) -> dict:
        """JSON summary: frames, entries, events, ring radii and (with n1) the masked fraction."""
        out: dict[str, Any] = dict(
            ring_frames=int(len(self.masks)),
            ring_entries=int(sum(len(v[0]) for v in self.masks.values())),
            ring_windows=len(self.windows),
            windows=[dict(first=int(a), end=int(b),
                          rings=[{k: _plain(v) for k, v in t.items()} for t in tab])
                     for a, b, tab in self.windows])
        if n1 is not None:
            nf, nb = n1.shape
            cells = ring_cells(self.masks, nf, nb)
            out["px_ring_frac"] = float(n1[cells].sum() / max(float(n1.sum()), 1.0))
        return out


def _plain(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return float(v)
    return v


def ring_cells(masks, nf, nbin, pad=0.0):
    """(n_frames, n_bins) bool: the 1-pixel annuli in a ring band."""
    m = np.zeros((nf, nbin), bool)
    r = np.arange(nbin) + 0.5
    for fi, (c, h) in masks.items():
        for cc, hh in zip(c, h):
            m[fi] |= np.abs(r - cc) <= hh + pad
    return m


def detect_rings(stats: RingStats, *, frame_scale: float = 1.0, half_t=100, med_r=21,
                 zpk=5.0, zclu=4.0, zgate=3.0, nmin=4, min_run=5, bridge=4, margin=2,
                 tol_rel=0.02, hw_pad=1.5, tex=3.0, min_sec=5, z_sec=2.5, rho_time=0.6,
                 hull_dil=3, z_strong=15.0, event_gap=60) -> RingDetection:
    """Find thin powder rings that occur at >= nmin radii in a frame.

    The defaults are the values of the frame chain (`frames.run_frame_chain`,
    checked on I15 and I19-2 data). Frame-window sizes (half_t,
    min_run, bridge, margin, hull_dil, event_gap and the hull windows) are
    for 0.2 degree frames: `frame_scale` = 0.2 / frame width scales them.

    The time background is a running median over 2 half_t + 1 frames. Thus
    the detector needs frames without the rings before or after the ring
    event in the given frames. If all frames have the rings, the running
    median contains the rings and the function finds no ring. The mask
    depends on the frames that you give. Another frame range can give a
    different mask.
    """
    def fr(n):
        return max(1, int(round(n * frame_scale)))

    half_t, min_run, bridge, margin = fr(half_t), fr(min_run), fr(bridge), fr(margin)
    hull_dil, event_gap, k3 = fr(hull_dil), fr(event_gap), fr(3)
    lam = stats.lam1.astype(np.float32)
    lams = stats.lams
    nf, nb = lam.shape
    finite = np.isfinite(lam)
    fill = _fill_rows(lam)
    base = ndi.median_filter(fill, size=(1, med_r), mode="nearest")
    hp = fill - base
    tb = _running_median(hp, half_t)
    hp2 = np.where(finite, hp - tb, 0.0).astype(np.float32)
    mad = 1.4826 * np.median(np.abs(hp2 - np.median(hp2, axis=0)), axis=0)
    sig = np.maximum(ndi.uniform_filter1d(mad, 15, mode="nearest"), 1e-3)
    z = hp2 / sig
    mx = ndi.maximum_filter1d(z, 7, axis=1)
    off = 7
    narrow = (np.roll(hp2, off, 1) < 0.5 * hp2) & (np.roll(hp2, -off, 1) < 0.5 * hp2)
    pk = (z >= mx) & (z > zclu) & narrow
    cnt = (pk & (z > zpk)).sum(axis=1)
    cnt_strong = (pk & (z > z_strong)).sum(axis=1)
    gate = _filter_runs(cnt >= nmin, min_run, bridge) | (cnt_strong >= nmin)
    events: list[list[int]] = []
    for (a_, b_) in _runs(gate):                     # one event: runs closer than event_gap
        if events and a_ - events[-1][1] <= event_gap:
            events[-1][1] = b_
        else:
            events.append([a_, b_])
    masks: dict = {}
    windows: list = []
    rejected: list = []
    for (i0, i1) in events:
        frs = np.arange(i0, i1)
        strong = frs[cnt[frs] >= max(nmin, int(0.4 * cnt[frs].max()))]
        pks = pk[strong]
        rr = np.where(pks)[1]
        ww = z[strong][pks]
        cnth = np.bincount(rr, minlength=nb).astype(float)
        hs = ndi.gaussian_filter1d(cnth, 1.0)
        cs5 = ndi.uniform_filter1d(cnth, 5) * 5
        need = max(3.0, 0.3 * len(strong))
        cand = [j for j in range(2, nb - 2)
                if hs[j] >= hs[j - 1] and hs[j] > hs[j + 1] and cs5[j] >= need]
        used = np.zeros(nb, bool)
        table = []
        for j in sorted(cand, key=lambda k: -hs[k]):
            if used[max(0, j - 4):j + 5].any():
                continue
            sel = (np.abs(rr - j) <= 2)
            rc = float(np.average(rr[sel], weights=ww[sel])) + 0.5
            used[max(0, j - 4):j + 5] = True
            table.append(dict(r=rc, zsum=float(ww[sel].sum()), nfr=int(sel.sum())))
        if not table:
            continue
        table.sort(key=lambda t: t["r"])
        f0, f1 = max(0, i0 - margin), min(nf, i1 + margin)
        ext = np.arange(f0, f1)
        for t in table:                              # amplitude series and azimuth coherence
            j = int(round(t["r"] - 0.5))
            lo, hi = max(0, j - 1), min(nb, j + 2)
            A = hp2[ext, lo:hi].max(axis=1)
            t["A"] = A
            t["z"] = A / sig[j]
            fsel = ext[np.argsort(A)[-3:]]
            nsec_ok, nsec_val = 0, 0
            if lo >= 6 and hi + 6 <= nb:
                seg_ = lams[fsel][:, :, max(0, j - 12):j + 13]
                with np.errstate(all="ignore"), warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)     # all-NaN sectors
                    sec_hp = (np.nanmax(lams[fsel][:, :, lo:hi], axis=2)
                              - np.nanmedian(seg_, axis=2))
                    sec_sig = (np.nanmedian(np.abs(np.diff(seg_, axis=2)), axis=2)
                               * 1.4826 / np.sqrt(2))
                    zs = np.nanmedian(sec_hp / np.maximum(sec_sig, 1e-3), axis=0)
                nsec_ok = int((zs > z_sec).sum())
                nsec_val = int(np.isfinite(zs).sum())
            t["nsec_valid"] = nsec_val
            t["nsec"] = nsec_ok
        amp = np.array([t["A"] for t in table])      # rank-one time course
        nrm = amp / np.maximum(amp.max(axis=1, keepdims=True), 1e-9)
        strong_t = np.array([t["zsum"] for t in table]) >= np.median([t["zsum"] for t in table])
        cons = np.median(nrm[strong_t], axis=0) if strong_t.any() else np.median(nrm, axis=0)
        for k, t in enumerate(table):
            a = nrm[k] - nrm[k].mean()
            c = cons - cons.mean()
            t["rho"] = float((a * c).sum() / max(np.sqrt((a * a).sum() * (c * c).sum()), 1e-12))
        keep = [t for t in table
                if t["nsec"] >= max(2, min(min_sec, 0.6 * t["nsec_valid"])) and t["rho"] >= rho_time]
        rejected += [_ring_row(t, i0, i1) for t in table if not any(t is k for k in keep)]
        table = keep
        if len(table) < nmin:
            rejected += [_ring_row(t, i0, i1) for t in table]
            continue
        for t in table:                              # ring width: fit of the mean profile
            j = int(round(t["r"] - 0.5))
            good = t["z"] > zpk
            t["sigma"] = np.nan
            if good.sum() >= 3:
                dr = np.arange(-12, 13)
                jj = np.clip(j + dr, 0, nb - 1)
                prof = (hp2[ext[good]][:, jj] / t["A"][good][:, None]).mean(axis=0)
                try:
                    from scipy.optimize import curve_fit
                    p, _ = curve_fit(_gauss, dr, prof, p0=(1.0, 0.0, 1.5, 0.0), maxfev=2000)
                    if 0.4 < abs(p[2]) < 8.0:
                        t["sigma"] = abs(p[2])
                except Exception:
                    pass
        sgs = np.array([t["sigma"] for t in table])
        okk = np.isfinite(sgs)
        rr_c = np.array([t["r"] for t in table])
        if okk.sum() >= 3:
            model = np.clip(np.polyval(np.polyfit(rr_c[okk], sgs[okk], 1), rr_c), 0.6, 8.0)
        elif okk.any():
            model = np.full(rr_c.shape, np.median(sgs[okk]))
        else:
            model = np.full(rr_c.shape, 1.5)
        for t, m in zip(table, model):
            if not np.isfinite(t["sigma"]):
                t["sigma"] = float(m)
        windows.append((int(i0), int(i1),
                        [{k: _plain(v) for k, v in t.items() if k not in ("A", "z")} for t in table]))
        hulls = []
        for t in table:                              # mask the hull of the frames above zgate
            good = t["z"] >= zgate
            closed = ndi.binary_closing(np.pad(good, 2 * k3, mode="constant"),
                                        structure=np.ones(2 * k3 + 1, bool))[2 * k3:-2 * k3]
            hulls.append(ndi.binary_dilation(closed, structure=np.ones(2 * hull_dil + 1, bool)))
        for k, fi in enumerate(ext):
            cs, hws = [], []
            for t, hl in zip(table, hulls):
                if not hl[k]:
                    continue
                j = int(round(t["r"] - 0.5))
                lam_bg = max(float(base[fi, min(nb - 1, max(0, j))]), 0.05)
                Aq = max(float(t["A"][max(0, k - k3):k + k3 + 1].max()), 1e-6) * tex
                hw = t["sigma"] * np.sqrt(2.0 * np.log(max(Aq / (tol_rel * lam_bg), np.e))) + hw_pad
                cs.append(t["r"])
                hws.append(hw)
            if cs:
                if fi in masks:
                    c0, h0 = masks[fi]
                    cs = list(c0) + cs
                    hws = list(h0) + hws
                masks[int(fi)] = (np.array(cs, np.float32), np.array(hws, np.float32))
    return RingDetection(masks, windows, rejected, cnt)


def _ring_row(t, i0, i1):
    return {**{k: _plain(v) for k, v in t.items() if k not in ("A", "z")},
            "window": (int(i0), int(i1))}


# ---------------------------------------------------------------- ring mask
class RingMask:
    """Pixel mask of ring bands for each frame position.

    bands : {frame position: (centres px, half widths px)}. Each band (c, hw)
    of a frame masks the pixels of that frame with
    |(floor(4 r) + 0.5) / 4 - c| <= hw. Here r is the pixel radius. The
    mask uses a look-up table with 4 steps per pixel.
    """

    def __init__(self, polar: PixelPolar, bands: dict, source: str = "detected"):
        self.polar = polar
        self.bands = {int(k): (np.asarray(c, np.float32), np.asarray(h, np.float32))
                      for k, (c, h) in bands.items()}
        self.source = source
        rmax = float(polar.r.max()) + 2
        self._rr = (np.arange(int(rmax * LUT_RES) + 2) + 0.5) / LUT_RES
        self._ri = None

    @classmethod
    def from_detection(cls, polar: PixelPolar, det: RingDetection) -> "RingMask":
        return cls(polar, det.masks, "detected")

    @classmethod
    def from_bands(cls, setup, bands, unit: str = "q", positions=None) -> "RingMask":
        """Mask from bands that you give: a list of (centre, half width).

        unit : 'px' (radius in pixels), 'q' (|q| = 1/d in 1/Angstrom) or
            'd' (d-spacing in Angstrom). For 'q' and 'd' the band edges are
            converted to pixel radii with the detector geometry.
        positions : frame positions of the mask (None: all frames).
        """
        det = setup.detector
        cs, hs = [], []
        for c, h in bands:
            if unit == "px":
                lo, hi = c - h, c + h
            elif unit == "q":
                lo, hi = radius_of_q(det, max(c - h, 0.0)), radius_of_q(det, c + h)
            elif unit == "d":
                lo, hi = radius_of_q(det, 1.0 / (c + h)), radius_of_q(det, 1.0 / max(c - h, 1e-9))
            else:
                raise ValueError("unit must be 'px', 'q' or 'd'")
            cs.append(0.5 * (lo + hi))
            hs.append(0.5 * (hi - lo))
        pos = range(setup.n_frames) if positions is None else positions
        bands_px = {int(i): (cs, hs) for i in pos}
        return cls(PixelPolar.of(det), bands_px, f"bands ({unit})")

    def pixel_mask(self, i: int) -> np.ndarray | None:
        """(npix,) bool mask of frame position i (True = not used), or None."""
        b = self.bands.get(int(i))
        if b is None:
            return None
        lut = np.zeros(self._rr.size, bool)
        for cc, hh in zip(*b):
            lut |= np.abs(self._rr - cc) <= hh
        if self._ri is None:
            self._ri = (self.polar.r * LUT_RES).astype(np.int32)
        return lut[self._ri]

    def apply(self, i: int, img):
        """Copy of `img` with the masked pixels set to -1 (the image itself if frame i has no band).

        An unsigned image gives an int64 copy, so that -1 stays negative."""
        m = self.pixel_mask(i)
        if m is None:
            return img
        flat = np.asarray(img).ravel()
        if flat.dtype.kind == "u":
            flat = flat.astype(np.int64)
        return np.where(m, -1, flat).reshape(np.shape(img))

    def masked_fraction(self, n1: np.ndarray) -> float:
        """Approximate fraction of the valid pixels in a band, from the annulus counts n1."""
        cells = ring_cells(self.bands, *n1.shape)
        return float(n1[cells].sum() / max(float(n1.sum()), 1.0))

    def summary(self) -> dict:
        return dict(source=self.source, n_frames=len(self.bands),
                    entries=int(sum(len(v[0]) for v in self.bands.values())))
