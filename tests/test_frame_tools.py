"""Tests of the frame tools (rspace3d 2.6.0): halves, pixel hook, prediction,
ring detector and mask, frame model and the frame chain.

Synthetic, CPU. The GPU test is skipped when CuPy or a GPU is not available.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from rspace3d.rawrecon import engine, frames as ft, rings as rg, frame_model as fm
from rspace3d.rawrecon.geometry import FlatDetector
from rspace3d.volume_builder import (VolumeData, symmetrize_volume, save_volume_h5,
                                     load_volume_h5)


class _SyntheticDetector:
    """Detector whose pixels have fixed scattering vectors (no geometry)."""
    wavelength = 0.5

    def __init__(self, r_lab, shape):
        self.r_lab = np.asarray(r_lab, np.float64)
        self.shape = shape

    def scattering_vectors(self, px):
        assert len(px) == len(self.r_lab)
        return self.r_lab


def _random_scan(seed=5, n=8, shape=(30, 40)):
    rng = np.random.default_rng(seed)
    det = _SyntheticDetector(rng.uniform(-0.6, 0.6, (shape[0] * shape[1], 3)), shape)
    g = rng.uniform(0.5, 2.0, shape[0] * shape[1])
    names = [f'f{i}' for i in range(n)]
    imgs = {f: rng.poisson(4.0, shape).astype(np.int32) for f in names}
    for f in names:
        imgs[f].flat[rng.integers(0, g.size, 5)] = -1
    kw = dict(phi0=0.0, osc_axis=(0.3, 1.0, 0.2), ranges=((-0.5, 0.5),) * 3, step=0.0625,
              read_fn=imgs.__getitem__, corrections=g)
    return det, names, np.arange(n) * 7.0, imgs, g, kw


def _same(a, b):
    return np.array_equal(a, b, equal_nan=True)


# ──────────────────────────────────────────────────────────────────
# Engine: halves and the pixel hook
# ──────────────────────────────────────────────────────────────────

def test_halves_sum_to_full_volume_and_match_separate_reconstructions():
    det, names, phis, _, _, kw = _random_scan()
    ub = np.diag([1.0, 1.25, 0.8])
    groups = np.arange(len(names)) % 2
    vol = engine.reconstruct_volume(names, phis, ub, np.eye(3), det, groups=groups, **kw)
    full = engine.reconstruct_volume(names, phis, ub, np.eye(3), det, **kw)
    assert vol.parts is not None and len(vol.parts) == 2 and full.parts is None
    for j in (0, 1):
        sep = engine.reconstruct_volume(names[j::2], phis[j::2], ub, np.eye(3), det, **kw)
        part = vol.parts[j]
        assert np.array_equal(part.counts, sep.counts)
        assert _same(part.data, sep.data) and _same(part.variance, sep.variance)
    assert np.array_equal(vol.counts, vol.parts[0].counts + vol.parts[1].counts)
    assert np.array_equal(vol.counts, full.counts)
    hit = full.counts > 0
    assert hit.sum() > 100
    assert np.array_equal(np.isnan(vol.data), ~hit)
    assert np.allclose(vol.data[hit], full.data[hit], rtol=1e-6)
    assert np.allclose(vol.variance[hit], full.variance[hit], rtol=1e-6)
    # the total is the count-weighted sum of the parts
    c0, c1 = (np.asarray(p.counts, np.float64) for p in vol.parts)
    s = np.nan_to_num(vol.parts[0].data * c0) + np.nan_to_num(vol.parts[1].data * c1)
    assert np.allclose(vol.data[hit], (s[hit] / (c0 + c1)[hit]), rtol=1e-6)


def test_groups_are_validated_and_setup_splits_by_frame_number():
    det, names, phis, _, _, kw = _random_scan(n=4)
    with pytest.raises(ValueError, match='groups'):
        engine.reconstruct_volume(names, phis, np.eye(3), np.eye(3), det, groups=[0, 1], **kw)
    setup = engine.ReconSetup(frames=names, numbers=[3, 4, 6, 7], phis=list(phis), detector=det,
                              UB=np.eye(3), R0=np.eye(3), phi0=0.0)
    assert setup.halves_groups().tolist() == [0, 1, 1, 0]
    assert engine.gpu_memory_bytes(1000) == 20000
    assert engine.gpu_memory_bytes(1000, halves=True) == 40000


def test_halves_need_odd_and_even_frame_numbers():
    import dataclasses
    det, names, phis, _, _, kw = _random_scan(n=4)
    setup = engine.ReconSetup(frames=names, numbers=[3, 4, 6, 7], phis=list(phis), detector=det,
                              UB=np.eye(3), R0=np.eye(3), phi0=0.0)
    assert engine._checked_halves(setup).tolist() == [0, 1, 1, 0]
    for nums in ([1, 3, 5, 7], [2, 4, 6, 8]):
        one = dataclasses.replace(setup, numbers=nums)
        with pytest.raises(ValueError, match='odd and even frame numbers'):
            engine._checked_halves(one)
        # the chain stops before its first pass
        with pytest.raises(ValueError, match='odd and even frame numbers'):
            ft.run_frame_chain(one, 'mmm', rings=None, halves=True, use_gpu=False,
                               progress=lambda i, n: pytest.fail('no pass expected'))


def test_frame_fn_with_positions_gets_the_position_in_frames():
    det, names, phis, imgs, g, kw = _random_scan()
    ub = np.diag([1.0, 1.25, 0.8])
    kw.pop('read_fn')
    kw.pop('corrections')
    setup = engine.ReconSetup(frames=names, numbers=list(range(1, len(names) + 1)),
                              phis=list(phis), detector=det, UB=ub, R0=np.eye(3),
                              read_fn=imgs.__getitem__, corrections=g, **kw)
    npix = det.shape[0] * det.shape[1]

    def fn(i, img):                       # offset i: depends on the position in `frames`
        return np.full(npix, float(i), np.float32), None
    pos = [2, 5, 6]
    got = setup.reconstruct(positions=pos, frame_fn=fn, use_gpu=False)
    sub = [names[i] for i in pos]
    ref = engine.reconstruct_volume(sub, [phis[i] for i in pos], ub, np.eye(3), det,
                                    read_fn=imgs.__getitem__, corrections=g,
                                    frame_fn=lambda j, img: fn(pos[j], img), **kw)
    assert np.array_equal(got.counts, ref.counts)
    assert _same(got.data, ref.data) and _same(got.variance, ref.variance)
    wrong = engine.reconstruct_volume(sub, [phis[i] for i in pos], ub, np.eye(3), det,
                                      read_fn=imgs.__getitem__, corrections=g, frame_fn=fn, **kw)
    assert not _same(got.data, wrong.data)


def test_pixel_hook_off_or_identity_is_bit_identical():
    det, names, phis, _, _, kw = _random_scan()
    ub = np.diag([1.0, 1.25, 0.8])
    plain = engine.reconstruct_volume(names, phis, ub, np.eye(3), det, **kw)
    ones = np.ones(det.shape[0] * det.shape[1], np.float32)
    for fn in (lambda i, img: None, lambda i, img: (None, None), lambda i, img: (None, ones),
               lambda i, img: (np.zeros_like(ones), ones)):
        v = engine.reconstruct_volume(names, phis, ub, np.eye(3), det, frame_fn=fn, **kw)
        assert np.array_equal(v.counts, plain.counts)
        assert _same(v.data, plain.data) and _same(v.variance, plain.variance)


@pytest.mark.parametrize('hot', [None, 4])
def test_corrected_pixels_value_and_variance_match_the_exact_formula(hot):
    """value = sum g (r - B) a / N, variance = sum g^2 a^2 max(r, 1) / N^2, NaN gain = masked.

    The hot cutoff acts on the raw count r, not on the corrected value."""
    rng = np.random.default_rng(3)
    shape = (6, 7)
    npix = shape[0] * shape[1]
    step = 0.25
    (n, _, _), (H, _, _), _ = engine._axes(((-0.5, 0.5),) * 3, step)
    vox = rng.integers(0, n, (npix, 3))
    det = _SyntheticDetector(H[vox], shape)
    g = rng.uniform(0.5, 2.0, npix)
    imgs, fixes = {}, {}
    for k, f in enumerate('abcd'):
        img = rng.poisson(3.0, shape).astype(np.int32)
        img.flat[rng.integers(0, npix)] = -1
        imgs[f] = img
        off = rng.uniform(0.0, 6.0, npix).astype(np.float32)       # gives negative values
        gain = rng.uniform(0.3, 2.5, npix)
        gain[rng.integers(0, npix, 3)] = np.nan                        # masked pixels
        fixes[k] = (off, gain)
    fixes[3] = None                                                    # a frame without change
    vol = engine.reconstruct_volume(list(imgs), [0.0] * 4, np.eye(3), np.eye(3), det,
                                    phi0=0.0, ranges=((-0.5, 0.5),) * 3, step=step,
                                    read_fn=imgs.__getitem__, corrections=g, hot=hot,
                                    frame_fn=lambda i, img: fixes[i])
    N = np.zeros((n,) * 3); S = np.zeros((n,) * 3); Q = np.zeros((n,) * 3)
    n_hot = 0
    for k, img in enumerate(imgs.values()):
        off, gain = fixes[k] if fixes[k] is not None else (np.zeros(npix), np.ones(npix))
        for p, r in enumerate(img.ravel()):
            if r < 0 or not np.isfinite(gain[p]):
                continue
            if hot is not None and r >= hot:
                n_hot += int(r - float(off[p]) < hot)    # a cutoff on r - B would keep it
                continue
            i = tuple(vox[p])
            N[i] += 1
            S[i] += g[p] * (r - float(off[p])) * gain[p]
            Q[i] += (g[p] * gain[p]) ** 2 * max(r, 1)
    hit = N > 0
    assert np.array_equal(vol.counts, N.astype(np.int64))
    assert (S[hit] < 0).any()
    assert np.allclose(vol.data[hit], S[hit] / N[hit], rtol=1e-5, atol=1e-6)
    assert np.allclose(vol.variance[hit], Q[hit] / N[hit] ** 2, rtol=1e-6)
    if hot is not None:
        # the test has removed pixels whose corrected value is below the cutoff
        assert n_hot > 10


def test_cancel_stops_the_engine():
    from rspace3d.averaging import Cancelled
    det, names, phis, _, _, kw = _random_scan()
    calls = []

    def cancelled():
        calls.append(1)
        return len(calls) > 2
    with pytest.raises(Cancelled):
        engine.reconstruct_volume(names, phis, np.eye(3), np.eye(3), det, cancelled=cancelled, **kw)
    assert len(calls) == 3


# ──────────────────────────────────────────────────────────────────
# A synthetic scan on a flat detector
# ──────────────────────────────────────────────────────────────────

NF = 160
RING_R = (62.5, 78.5, 94.5, 110.5)
RING_FRAMES = (30, 60)
S_FRAMES, S_SECTORS, S_VALUE = (70, 100), (20, 40), 1.6
B_FRAMES, B_BAND, B_VALUE = (120, 140), (100.0, 116.0), 4.0


def _truth(hkl):
    return 12.0 * np.exp(-(hkl ** 2).sum(-1) / 40.0) + 30.0


class _Scene:
    """Detector, frames and truth of a small rotation scan (mmm intensity)."""

    def __init__(self, ring=False, scale=False, add=False, corr=False, seed=1, nf=NF):
        self.det = FlatDetector(distance=150.0, pixel_size=1.0, beam_center=(127.3, 128.6),
                                shape=(256, 256), wavelength=0.5)
        self.UB = np.eye(3) * 0.25
        ny, nx = self.det.shape
        ys, xs = np.mgrid[0:ny, 0:nx]
        px = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float32)
        r_lab = self.det.scattering_vectors(px)
        self.g = (np.random.default_rng(9).uniform(0.8, 1.25, r_lab.shape[0])
                  if corr else None)
        polar = rg.PixelPolar.of(self.det)
        cells = ft.FrameCells(polar)
        phis = np.arange(nf) * (180.0 / nf)
        rng = np.random.default_rng(seed)
        self.imgs, self.clean = {}, {}
        UBinv = np.linalg.inv(self.UB)
        for i in range(nf):
            Mn = engine.frame_matrix(UBinv, np.eye(3), (0, 1, 0), 1, phis[i], 0.0)
            lam = _truth(r_lab @ Mn.T.astype(np.float64))
            if self.g is not None:
                lam = lam / self.g
            self.clean[i] = lam.reshape(ny, nx)
            c = lam.copy()
            if scale and S_FRAMES[0] <= i < S_FRAMES[1]:
                sel = (cells.sec >= S_SECTORS[0]) & (cells.sec < S_SECTORS[1])
                c[sel] *= S_VALUE
            if add and B_FRAMES[0] <= i < B_FRAMES[1]:
                c[(polar.r >= B_BAND[0]) & (polar.r < B_BAND[1])] += B_VALUE
            if ring and RING_FRAMES[0] <= i < RING_FRAMES[1]:
                amp = 6.0 * np.sin(np.pi * (i - RING_FRAMES[0] + 0.5) / (RING_FRAMES[1] - RING_FRAMES[0]))
                for R in RING_R:
                    c += amp * np.exp(-0.5 * ((polar.r - R) / 1.0) ** 2)
            img = rng.poisson(c).astype(np.int32).reshape(ny, nx)
            img[:, 100:103] = -1                                     # a detector gap
            self.imgs[i] = img
        self.polar, self.cells = polar, cells
        self.setup = engine.ReconSetup(
            frames=list(range(nf)), numbers=list(range(1, nf + 1)), phis=list(phis),
            detector=self.det, UB=self.UB, R0=np.eye(3), phi0=0.0, increment=0.2,
            corrections=self.g, ranges=((-6.0, 6.0),) * 3, step=0.25,
            read_fn=self.imgs.__getitem__, prefetch=2)

    def clean_setup(self):
        import dataclasses
        return dataclasses.replace(self.setup, read_fn=self.clean.__getitem__)


@pytest.fixture(scope='module')
def clean_scene():
    return _Scene(corr=True)


@pytest.fixture(scope='module')
def dirty_scene():
    return _Scene(ring=True, scale=True, add=True)


def _vd(vol):
    return engine.reconstruct_to_volumedata(vol)


# ──────────────────────────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────────────────────────

def test_prediction_of_a_constant_model_is_the_constant_over_the_correction(clean_scene):
    s = clean_scene.setup
    (n, _, _), (H, K, L), _ = engine._axes(s.ranges, s.step)
    model = VolumeData(np.full((n, n, n), 7.5, np.float32), H, K, L, 'HK', {})
    pred = ft.predict_frame(s, model, 17).ravel()
    ok = np.isfinite(pred)
    assert ok.mean() > 0.5
    assert np.allclose(pred[ok], 7.5 / clean_scene.g[ok].astype(np.float32), rtol=1e-6)
    # a pixel outside the grid has no prediction
    small = VolumeData(np.full((5, 5, 5), 7.5, np.float32), H[:5] * 0 + np.arange(5) * 0.25,
                       H[:5] * 0 + np.arange(5) * 0.25, H[:5] * 0 + np.arange(5) * 0.25, 'HK', {})
    assert np.isnan(ft.predict_frame(s, small, 17)).mean() > 0.9


def test_leave_one_out_median_is_the_median_of_the_flipped_volumes():
    rng = np.random.default_rng(2)
    n = 9
    ax = (np.arange(n) - n // 2) * 0.25
    data = rng.normal(5, 1, (n, n, n)).astype(np.float32)
    data[rng.random(data.shape) < 0.3] = np.nan
    vol = VolumeData(data, ax, ax, ax, 'HK', {'grid_kind': 'hkl_regular'})
    got = ft.leave_one_out_median(vol, 'mmm')
    others = [data[tuple(slice(None, None, -1) if f else slice(None) for f in fl)]
              for fl in itertools.product((0, 1), repeat=3) if any(fl)]
    st = np.sort(np.stack(others), axis=0)
    k = np.isfinite(st).sum(0)
    lo = np.take_along_axis(st, np.maximum((k - 1) // 2, 0)[None], 0)[0]
    hi = np.take_along_axis(st, np.maximum(k // 2, 0)[None], 0)[0]
    ref = np.where(k > 0, 0.5 * (lo + hi), np.nan).astype(np.float32)
    assert _same(got, ref)


# ──────────────────────────────────────────────────────────────────
# Rings
# ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def ring_scene():
    return _Scene(ring=True)


def test_ring_detector_finds_an_injected_ring_and_nothing_on_clean_frames(ring_scene, clean_scene):
    stats = rg.ring_statistics(ring_scene.setup, use_gpu=False)
    det = rg.detect_rings(stats)
    assert det.found and len(det.windows) == 1
    found = sorted(t['r'] for t in det.windows[0][2])
    assert len(found) == len(RING_R)
    assert np.allclose(found, RING_R, atol=0.01)       # the rings are at bin centres
    masked = sorted(det.masks)
    assert masked[0] <= RING_FRAMES[0] + 3 and masked[-1] >= RING_FRAMES[1] - 4
    assert all(RING_FRAMES[0] - 15 <= i < RING_FRAMES[1] + 15 for i in masked)
    # frame_scale 2 (0.1-degree frames): the frame windows are longer, so the
    # masked frames cover a wider window around the same event
    wide = rg.detect_rings(stats, frame_scale=2.0)
    assert wide.found and len(wide.windows) == 1
    assert np.allclose(sorted(t['r'] for t in wide.windows[0][2]), RING_R, atol=0.01)
    wm = sorted(wide.masks)
    assert set(masked) < set(wm) and wm[0] < masked[0] and wm[-1] > masked[-1]
    clean = rg.detect_rings(rg.ring_statistics(clean_scene.setup, use_gpu=False))
    assert not clean.found and clean.masks == {}


def _xps():
    """NumPy, and CuPy when a GPU is available."""
    out = [np]
    if engine.has_cupy():
        import cupy
        out.append(cupy)
    return out


def test_ring_statistics_of_a_frame_without_valid_pixels():
    sc = _Scene(nf=6)
    sc.imgs[2][:] = -1
    for xp in _xps():
        tm, n = rg._trim_stats(xp, xp.zeros(0, xp.int64), xp.zeros(0, xp.int64), 5, rg.TRIM)
        assert np.isnan(fm.to_host(tm)).all() and (fm.to_host(n) == 0).all()
        stats = rg.ring_statistics(sc.setup, use_gpu=xp is not np)
        assert np.isnan(stats.lam1[2]).all() and np.isnan(stats.lams[2]).all()
        assert np.isfinite(stats.lam1[3]).any()


def test_ring_mask_of_an_unsigned_image_gives_minus_one():
    sc = _Scene(nf=2)
    band = rg.RingMask.from_bands(sc.setup, [(50.0, 2.0)], unit='px', positions=[0])
    m = band.pixel_mask(0)
    for dt in (np.uint16, np.uint32):
        img = np.full(sc.det.shape, 7, dt)
        out = band.apply(0, img)
        assert out.shape == img.shape and out.dtype.kind == 'i'
        assert (out.ravel()[m] == -1).all() and (out.ravel()[~m] == 7).all()


def test_ring_mask_removes_the_ring_band(ring_scene):
    s = ring_scene.setup
    stats = rg.ring_statistics(s, use_gpu=False)
    mask = rg.RingMask.from_detection(ring_scene.polar, rg.detect_rings(stats))
    mid = (RING_FRAMES[0] + RING_FRAMES[1]) // 2
    m = mask.pixel_mask(mid)
    r = ring_scene.polar.r
    for R in RING_R:
        assert m[np.abs(r - R) <= 2.0].all()
    assert not m[np.abs(r - 30.0) <= 2.0].any()
    assert mask.pixel_mask(5) is None
    img = mask.apply(mid, s.read_fn(mid))
    assert (img.ravel()[m] == -1).all() and img.dtype == np.int32
    # user bands in pixels, in |q| and in d give the same band
    band = rg.RingMask.from_bands(s, [(50.0, 2.0)], unit='px', positions=[3])
    assert band.pixel_mask(3)[np.abs(r - 50.0) <= 1.9].all()
    assert not band.pixel_mask(3)[np.abs(r - 50.0) >= 2.1].any()
    q0 = rg.q_of_radius(s.detector, 50.0)
    qb = rg.RingMask.from_bands(s, [(q0, 0.01)], unit='q').pixel_mask(0)
    db = rg.RingMask.from_bands(s, [(1.0 / q0, 0.01 / q0 ** 2)], unit='d').pixel_mask(0)
    assert qb.any() and (qb != db).mean() < 1e-3
    # the masked reconstruction does not use the band pixels
    vol = ft.corrected_reconstruction(s, mask=band, use_gpu=False)
    plain = s.reconstruct(use_gpu=False)
    valid = (s.read_fn(3).ravel() >= 0)
    assert plain.counts.sum() - vol.counts.sum() == (band.pixel_mask(3) & valid).sum()


# ──────────────────────────────────────────────────────────────────
# Frame model
# ──────────────────────────────────────────────────────────────────

def test_frame_model_recovers_known_scale_and_additive_terms():
    sc = _Scene(scale=True, add=True, corr=True, seed=4)
    model = _vd(sc.clean_setup().reconstruct(use_gpu=False))           # the noise-free truth
    stats = ft.frame_statistics(sc.setup, model, cells=sc.cells, use_gpu=False)
    fit = fm.fit_frame_model(stats.D, stats.P, stats.N, use_gpu=False)
    a = fit.a
    f0, f1 = S_FRAMES
    s0, s1 = S_SECTORS
    inner = a[f0 + 3:f1 - 3, s0 + 2:s1 - 2]
    assert abs(np.median(inner) - (S_VALUE - 1)) < 0.05
    outside = np.ones(a.shape, bool)
    outside[f0 - 3:f1 + 3, s0 - 3:s1 + 3] = False
    assert (a[outside] == 0).mean() > 0.99
    B = fit.B.astype(np.float32)
    b0, b1 = B_FRAMES
    band = B[b0 + 2:b1 - 2, :, int(B_BAND[0]) + 2:int(B_BAND[1]) - 2]
    band = band[stats.N[b0 + 2:b1 - 2, :, int(B_BAND[0]) + 2:int(B_BAND[1]) - 2] >= 8]
    # The frame-median baseline of a 9-pixel cell is shifted by the contaminated
    # frames. Measured: median 3.46 against B_VALUE 4.0 (bias 0.54), also 3.45 in the
    # pixel image below. The tolerance 0.6 covers this bias with a margin of about 0.06.
    assert abs(np.median(band) - B_VALUE) < 0.6
    far = np.ones(B.shape[0], bool)
    far[b0 - 3:b1 + 3] = False
    assert (B[far] == 0).mean() > 0.999
    assert fit.info['frames_S'] > 0 and fit.info['frames_B'] > 0
    # the pixel images of the tables
    i = (f0 + f1) // 2
    S_img = sc.cells.S_image(a[i])
    sel = (sc.cells.sec == (s0 + s1) // 2)
    assert np.allclose(np.median(S_img[sel]), S_VALUE, atol=0.05)
    j = (b0 + b1) // 2
    B_img = sc.cells.B_image(fit.B[j])
    rr = sc.polar.r
    assert abs(np.median(B_img[(rr > B_BAND[0] + 3) & (rr < B_BAND[1] - 3)]) - B_VALUE) < 0.6
    assert np.median(B_img[rr < 60]) == 0


def test_pass_a_handles_frames_without_usable_pixels():
    sc = _Scene(nf=6)
    sc.imgs[2][:] = -1                                     # a blank or fully invalid frame
    s = sc.setup
    nseg = sc.cells.nseg
    seg = sc.cells.seg
    npix = seg.size
    for xp in _xps():
        g = xp.full(npix, 5.0, xp.float32)
        pred = xp.full(npix, 5.0, xp.float32)
        for gg, pp in ((g, xp.full(npix, xp.nan, xp.float32)), (-xp.ones_like(g), pred)):
            D, P, n, ok, gv, pv = ft._frame_cells_stats(xp, gg, pp, xp.asarray(seg), nseg)
            assert np.isnan(fm.to_host(D).astype(np.float32)).all()
            assert np.isnan(fm.to_host(P).astype(np.float32)).all()
            assert (fm.to_host(n) == 0).all() and gv.size == 0
        assert np.isnan(fm.to_host(ft._seg_median(xp, xp.zeros(0, xp.int64),
                                                  xp.zeros(0, xp.float64),
                                                  xp.zeros(4, xp.int64)))).all()
        (n_, _, _), (H, K, L), _ = engine._axes(s.ranges, s.step)
        model = VolumeData(np.full((n_, n_, n_), 30.0, np.float32), H, K, L, 'HK', {})
        st = ft.frame_statistics(s, model, cells=sc.cells, use_gpu=xp is not np)
        assert np.isnan(st.D[2].astype(np.float32)).all() and (st.N[2] == 0).all()
        assert (st.obs_sector[2] == 0).all() and (st.pred_sector[2] == 0).all()
        assert np.isfinite(st.D[3].astype(np.float32)).any() and st.N[3].sum() > 0
        # a model without any prediction: every frame is empty, no error
        empty = VolumeData(np.full((n_, n_, n_), np.nan, np.float32), H, K, L, 'HK', {})
        st = ft.frame_statistics(s, empty, cells=sc.cells, use_gpu=xp is not np)
        assert (st.N == 0).all() and st.n_radial.sum() == 0


def test_frame_chain_on_a_grid_without_symmetry_equivalents_gives_a_clear_error():
    import dataclasses
    sc = _Scene(nf=6)
    off = dataclasses.replace(sc.setup, ranges=((0.5, 3.0), (0.5, 3.0), (-3.0, -0.5)))
    with pytest.raises(ValueError, match='no finite voxel'):
        ft.run_frame_chain(off, 'mmm', rings=None, use_gpu=False)


def test_frame_model_is_zero_on_clean_frames(clean_scene):
    model = _vd(clean_scene.clean_setup().reconstruct(use_gpu=False))
    stats = ft.frame_statistics(clean_scene.setup, model, use_gpu=False)
    fit = fm.fit_frame_model(stats.D, stats.P, stats.N, use_gpu=False)
    assert (fit.a == 0).mean() > 0.995
    assert (fit.B == 0).mean() > 0.999
    fmap = ft.frame_map_from_stats(clean_scene.setup, stats)
    assert abs(np.nanmedian(fmap.excess) - 1.0) < 0.01
    assert fmap.excess.shape == (NF, 72) and fmap.resid.shape == fmap.q.shape
    s = fmap.summary()
    assert s['frames_off_10pct'] == 0 and s['n_frames'] == NF


def test_frame_map_normalises_each_sector_and_counts_the_frames_that_differ():
    # Real data: an offset of all ratios (median 0.94), a sector pattern (a shadow) and
    # cell scatter. Only the frames of an event differ from the other frames.
    rng = np.random.default_rng(3)
    nf = 300
    sector = np.linspace(0.45, 1.05, 72)
    ex = 0.94 * sector[None, :] * rng.lognormal(0.0, 0.05, (nf, 72))
    ex[100:130] *= 1.3                       # event: 30 frames with 30 % more counts
    ex[:, 7] = np.nan                        # a sector without prediction
    ex[200] = np.nan                         # a frame without prediction
    raw = ex.astype(np.float32)
    fmap = ft.FrameMap(raw.copy(), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))
    nr = fmap.normalised()
    assert np.array_equal(fmap.excess, raw, equal_nan=True)    # the sums stay unchanged
    assert np.isnan(nr[:, 7]).all() and np.isnan(nr[200]).all()
    ok = np.isfinite(nr)
    assert abs(np.median(nr[ok]) - 1.0) < 0.01
    assert abs(np.median(nr[100:130][ok[100:130]]) - 1.3) < 0.03
    s = fmap.summary()
    assert s['frames_off_10pct'] == 30 and s['n_frames'] == nf
    assert 0.02 < s['cell_dev_median'] < 0.06
    assert abs(s['excess_median'] - float(np.nanmedian(raw))) < 1e-6
    # the old rule (any sector beyond 10 % of 1) counted every frame
    assert int((np.abs(raw - 1.0) > 0.1).any(axis=1).sum()) == nf - 1


def test_frame_map_shows_an_empty_frame_as_off():
    # A frame without counts (for example a closed shutter) has the ratio 0. It must be
    # a finite cell at the bottom of the scale and count as off. A cell without a
    # prediction stays NaN.
    rng = np.random.default_rng(4)
    nf = 400
    ex = rng.lognormal(0.0, 0.03, (nf, 72)).astype(np.float32)
    ex[100] = 0.0
    ex[200] = 1e-4
    ex[:, 7] = np.nan
    fmap = ft.FrameMap(ex, np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))
    nr = fmap.normalised()
    for i in (100, 200):
        assert np.isfinite(np.delete(nr[i], 7)).all()
        assert np.allclose(np.delete(nr[i], 7), ft.FRAME_MAP_FLOOR)
    assert np.isnan(nr[:, 7]).all()
    assert ft.FRAME_MAP_FLOOR < 1 / 1.5                 # the bottom colour of the map
    others = np.delete(nr, [100, 200], axis=0)
    assert abs(np.nanmedian(others) - 1.0) < 0.01       # the empty frame does not move the median
    s = fmap.summary()
    assert s['frames_off_10pct'] == 2 and s['excess_min'] == 0.0


def test_reconstruct_dataset_keeps_the_setup_only_on_request(monkeypatch):
    import dataclasses
    sc = _Scene(nf=6)
    meta = dict(par='x.par', name='x', folder='synthetic', n_total=6, index_rms=0.0,
                index_inliers=0, geom_hkl_dev=0.0, cell=None, polarization_axis='none',
                geometry_method='indexed_first_frame', missing_frame_numbers=[])
    setup = dataclasses.replace(sc.setup, meta=meta)
    monkeypatch.setattr(engine, 'prepare_setup', lambda folder, **kw: setup)
    vol, info = engine.reconstruct_dataset('synthetic', use_gpu=False)
    assert 'setup' not in info                          # the 2.5.0 keys only
    vol2, info2 = engine.reconstruct_dataset('synthetic', use_gpu=False, keep_setup=True)
    assert info2['setup'] is setup and set(info2) - set(info) == {'setup'}
    assert np.array_equal(vol.counts, vol2.counts)


# ──────────────────────────────────────────────────────────────────
# The chain
# ──────────────────────────────────────────────────────────────────

def _err(vol, truth):
    ok = np.isfinite(vol) & np.isfinite(truth) & (truth > 0)
    return float(np.mean(np.abs(vol[ok] - truth[ok]) / truth[ok])), ok.mean()


def test_frame_chain_lowers_the_error_against_the_truth(dirty_scene, tmp_path):
    s = dirty_scene.setup
    truth = symmetrize_volume(_vd(dirty_scene.clean_setup().reconstruct(use_gpu=False)), 'mmm',
                              sigma=None, use_gpu=False).intensity
    raw = s.reconstruct(use_gpu=False)
    plain = symmetrize_volume(_vd(raw), 'mmm', sigma=None, use_gpu=False).intensity
    steps = []
    res = ft.run_frame_chain(s, 'mmm', use_gpu=False, raw=raw, halves=True,
                             progress=lambda i, n: steps.append((i, n)))
    e_plain, _ = _err(plain, truth)
    e_chain, cover = _err(res.volume.intensity, truth)
    e_mask, _ = _err(symmetrize_volume(res.masked, 'mmm', sigma=None, use_gpu=False).intensity,
                     truth)
    # Measured: e_plain 0.0406, e_mask 0.0397, e_chain 0.0058. The frame model does
    # the main correction in this scene, so the ring mask alone gains only about 2 %.
    # test_ring_mask_removes_the_ring_band checks the mask itself.
    assert e_chain < 0.5 * e_plain and e_mask < e_plain
    assert cover > 0.3
    assert res.rings is not None and res.rings.found
    info = res.info
    assert info['correction'] == 'ring_mask+frame_model' and info['iterations'] == 3
    assert info['rings']['ring_windows'] == 1 and 0 < info['rings']['masked_pixel_frac_total'] < 0.1
    assert len(info['iterations_info']) == 3
    assert steps[-1][0] == steps[-1][1] == 8 * NF
    # halves of the last pass: their counts add up to the corrected volume
    assert len(res.halves) == 2 and len(res.masked_halves) == 2
    assert np.array_equal(res.halves[0].counts + res.halves[1].counts, res.corrected.counts)
    assert res.corrected.variance is not None
    # the metadata survive save_volume_h5
    path = str(tmp_path / 'chain.h5')
    save_volume_h5(path, res.volume)
    back = load_volume_h5(path)
    assert back.metadata['frame_correction'] == 'ring_mask+frame_model'
    assert back.metadata['frame_tools'] == res.volume.metadata['frame_tools']
    assert back.metadata['laue_group'] == 'mmm'


def test_frame_chain_without_mask_and_without_a_found_ring(clean_scene):
    s = clean_scene.setup
    raw = s.reconstruct(use_gpu=False)
    res = ft.run_frame_chain(s, 'mmm', rings=None, frame_model=False, raw=raw, use_gpu=False)
    plain = symmetrize_volume(_vd(raw), 'mmm', sigma=None, use_gpu=False)
    assert _same(res.volume.intensity, plain.intensity)
    assert res.info['correction'] == 'none'
    auto = ft.run_frame_chain(s, 'mmm', frame_model=False, use_gpu=False)
    assert auto.rings is not None and not auto.rings.found
    assert _same(auto.volume.intensity, plain.intensity)


def test_frame_chain_with_a_ring_mask_only(ring_scene):
    s = ring_scene.setup
    mask = rg.RingMask.from_detection(ring_scene.polar,
                                      rg.detect_rings(rg.ring_statistics(s, use_gpu=False)))
    raw = s.reconstruct(use_gpu=False)
    res = ft.run_frame_chain(s, 'mmm', rings=mask, frame_model=False, raw=raw, use_gpu=False)
    assert res.info['correction'] == 'ring_mask' and res.info['iterations'] == 0
    assert res.info['rings']['mode'] == 'given'
    ref = ft.corrected_reconstruction(s, mask=mask, use_gpu=False)
    assert np.array_equal(res.corrected.counts, ref.counts)
    assert _same(res.corrected.intensity, _vd(ref).intensity)
    # the masked pixel fraction is the share of the raw counts that the mask removed
    frac = 1.0 - ref.counts.sum() / raw.counts.sum()
    assert res.info['rings']['masked_pixel_frac_total'] == pytest.approx(frac)
    assert 0 < frac < 0.1


def test_frame_chain_takes_the_frame_scale_of_the_options(monkeypatch):
    import dataclasses
    sc = _Scene(nf=6)
    s = dataclasses.replace(sc.setup, increment=0.1)          # 0.1-degree frames: scale 2
    seen = []

    class Stop(Exception):
        pass

    def fake_detect(stats, frame_scale=1.0):
        seen.append(frame_scale)
        raise Stop
    monkeypatch.setattr(ft, 'ring_statistics', lambda *a, **k: None)
    monkeypatch.setattr(ft, 'detect_rings', fake_detect)
    for opts, want in ((None, 2.0), (fm.ModelOptions(k_B=6.0), 1.0),
                       (fm.ModelOptions(frame_scale=3.0), 3.0)):
        with pytest.raises(Stop):
            ft.run_frame_chain(s, 'mmm', options=opts, use_gpu=False)
        assert seen[-1] == want
    monkeypatch.undo()
    res = ft.run_frame_chain(s, 'mmm', rings=None, frame_model=False,
                             options=fm.ModelOptions(frame_scale=3.0), use_gpu=False)
    assert res.info['frame_scale'] == 3.0 and res.info['options']['frame_scale'] == 3.0
    res = ft.run_frame_chain(s, 'mmm', rings=None, frame_model=False, use_gpu=False)
    assert res.info['frame_scale'] == 2.0


# ──────────────────────────────────────────────────────────────────
# GPU = CPU (skipped without CuPy)
# ──────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not engine.has_cupy(), reason='CuPy or a GPU is not available')
def test_gpu_and_cpu_give_the_same_frame_tools(ring_scene):
    s = ring_scene.setup
    stats_c = rg.ring_statistics(s, use_gpu=False)
    stats_g = rg.ring_statistics(s, use_gpu=True)
    assert _same(stats_c.lam1, stats_g.lam1) and _same(stats_c.lams, stats_g.lams)
    det = rg.detect_rings(stats_c)
    mask = rg.RingMask.from_detection(ring_scene.polar, det)
    groups = s.halves_groups()
    vc = ft.corrected_reconstruction(s, mask=mask, groups=groups, use_gpu=False)
    vg = ft.corrected_reconstruction(s, mask=mask, groups=groups, use_gpu=True)
    for a, b in [(vc, vg)] + list(zip(vc.parts, vg.parts)):
        assert np.array_equal(a.counts, b.counts)
        assert np.allclose(a.data, b.data, rtol=1e-6, equal_nan=True)
        assert np.allclose(a.variance, b.variance, rtol=1e-6, equal_nan=True)
    model = _vd(vc)
    pv = ft._with_intensity(model, ft.leave_one_out_median(model, 'mmm'))
    pc = ft.frame_statistics(s, pv, mask=mask, use_gpu=False)
    pg = ft.frame_statistics(s, pv, mask=mask, use_gpu=True)
    assert np.array_equal(pc.N, pg.N)
    fin = np.isfinite(pc.D.astype(np.float32))
    assert np.array_equal(fin, np.isfinite(pg.D.astype(np.float32)))
    d = np.abs(pc.D.astype(np.float32) - pg.D.astype(np.float32))[fin]
    assert (d <= 2e-3 * (1 + np.abs(pc.D.astype(np.float32)[fin]))).mean() > 0.999
    fc = fm.fit_frame_model(pc.D, pc.P, pc.N, use_gpu=False)
    fg = fm.fit_frame_model(pc.D, pc.P, pc.N, use_gpu=True)
    assert np.allclose(fc.a, fg.a, atol=1e-6)
    assert (fc.B != fg.B).mean() < 1e-4
