"""Second fit of `rawrecon.calibrate.geometry_from_tabbin`.

The tests use a synthetic rotation scan with a known geometry. The peak table
has the properties that make the first fit wrong on real low-temperature data:

* the reciprocal vectors of the table come from an instrument model with a
  wrong oscillation axis (as the model of CrysAlisPro can be),
* the strong peaks scatter about their ideal positions (a split crystal),
* peaks at half-integer hkl are present (a superstructure),
* weak entries are at wrong positions. This population is a worst case: all
  weak entries agree with one other axis, so a fit that uses them finds a
  wrong axis.

The first fit selects its peaks one time with the vectors of the table. The
selection keeps the peaks whose scatter cancels the error of the table, so
the oscillation axis of the first fit is wrong. The second fit must find the
true axis.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from rspace3d.rawrecon import calibrate, tabbin
from rspace3d.rawrecon.geometry import FlatDetector

NX, NY, PS, WL, DIST = 420, 400, 0.5, 0.30, 200.0
BC = np.array([210.3, 199.6])                 # beam centre (col, row), 1-based
INCR, NFRAMES = 0.1, 3600
FC, FR, Z = calibrate._X, calibrate._Y, calibrate._Z
UB = np.eye(3) / 6.0                          # cubic, a = 6 A
R0 = Rot.from_rotvec([0.3, -0.5, 0.2]).as_matrix()
AXIS = np.array([0.004, 1.0, -0.0005]) / np.linalg.norm([0.004, 1.0, -0.0005])


def _tilted(axis, deg, about=Z):
    return Rot.from_rotvec(np.radians(deg) * np.asarray(about, float)).apply(axis)


def _nodes(hmax, half=False):
    r = np.arange(-hmax, hmax + 1)
    n = np.array(np.meshgrid(r, r, r, indexing='ij')).reshape(3, -1).T.astype(float)
    return n + 0.5 if half else n[np.any(n != 0, axis=1)]


def _peaks(nodes, axis, rng, mosaic_deg=0.0):
    """Pixel (col, row; 1-based, not rounded) and frame of each crossing of the
    Ewald sphere for the lattice vectors of `nodes`, for a rotation about `axis`."""
    g0 = (R0 @ UB @ nodes.T).T
    if mosaic_deg:
        g0 = Rot.from_rotvec(np.radians(mosaic_deg) * rng.standard_normal((len(g0), 3))).apply(g0)
    s0 = Z / WL
    gpar = (g0 @ axis)[:, None] * axis
    gper = g0 - gpar
    a, b = gper @ s0, np.cross(axis, gper) @ s0
    c = -0.5 * np.sum(g0 ** 2, axis=1) - gpar @ s0
    rad = np.hypot(a, b)
    ok = np.abs(c) < rad
    base, half = np.arctan2(b[ok], a[ok]), np.arccos(c[ok] / rad[ok])
    out = []
    for theta in (base + half, base - half):
        s1 = calibrate._rotate(g0[ok], axis, theta) + s0
        P = s1 * (DIST / s1[:, 2])[:, None]
        col, row = P @ FC / PS + BC[0], P @ FR / PS + BC[1]
        frame = np.floor((np.degrees(theta) % 360.0) / INCR).astype(int) + 1
        m = (col > 3) & (col < NX - 2) & (row > 3) & (row < NY - 2)
        out.append((col[m], row[m], frame[m], nodes[ok][m]))
    return [np.concatenate(x) for x in zip(*out)]


def _params(axis):
    return np.array([BC[0], BC[1], DIST, *Rot.from_matrix(R0).as_rotvec(),
                     np.arccos(axis[1]), np.arctan2(axis[2], axis[0])])


def _to_table(name, col, row):
    """Inverse of calibrate._pixel_map(name, NX, NY): frame pixel -> table pixel."""
    tr, fx, fy = name[0] == 'T', name[1] == 'X', name[2] == 'Y'
    a = NX + 1 - col if fx else col
    b = NY + 1 - row if fy else row
    return (b, a) if tr else (a, b)


def _table(seed=0, cen_sign=1.0, table_axis_tilt=0.25, mosaic=0.10, weak_axis_tilt=0.4,
           rot_sign=1, mapping='TXY'):
    """A peak table as a dict of arrays, in the layout of `read_tabbin`.
    rot_sign = -1 turns the crystal the other way (sense -1 about AXIS)."""
    rng = np.random.default_rng(seed)
    axis = rot_sign * AXIS
    parts = [
        _peaks(_nodes(6), axis, rng, mosaic) + [False],                 # split crystal
        _peaks(_nodes(3, half=True), axis, rng, mosaic) + [False],      # superstructure
        _peaks(_nodes(6), _tilted(axis, -weak_axis_tilt), rng) + [True],  # weak, wrong
    ]
    col = np.concatenate([p[0] for p in parts]); row = np.concatenate([p[1] for p in parts])
    frame = np.concatenate([p[2] for p in parts])
    weak = np.concatenate([np.full(len(p[0]), p[4]) for p in parts])
    inten = np.where(weak, rng.integers(50, 300, len(col)), rng.integers(5000, 200000, len(col)))
    fx, fy = _to_table(mapping, col, row)                 # table pixel with sub-pixel part
    px, py = np.rint(fx), np.rint(fy)
    # reciprocal vectors of the table: an instrument model with a wrong axis
    hkl_tab = calibrate._hkl_of_peaks(_params(_tilted(axis, table_axis_tilt)), col, row,
                                      (frame - 1) * INCR, UB, FC, FR, 1, PS, WL)
    return {'d': (UB * WL @ hkl_tab.T).T, 'intensity': inten, 'px': px.astype(int),
            'py': py.astype(int), 'cenx': cen_sign * (fx - px), 'ceny': cen_sign * (fy - py),
            'frame': frame, 'weak': weak, 'mapping': mapping}


def _axis_error_deg(effective_axis, rot_sign=1):
    return float(np.degrees(np.arccos(np.clip(effective_axis @ (rot_sign * AXIS), -1, 1))))


def _fitted_axis(fg):
    """Effective rotation axis (sense x axis) in the frame of the detector (fast,
    slow, beam). This does not depend on the laboratory frame that the fit selected."""
    d, a = fg.detector, np.asarray(fg.osc_axis)
    return fg.sense * np.array([a @ d.fast, a @ d.slow, a @ d.beam])


class _Frame:
    def __init__(self, data):
        self.data = data


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    """Write a par and a tabbin, and replace the frame functions. Returns the function
    make(**options) -> (par_path, tabbin_path, table)."""
    import fabio

    def make(**kw):
        t = _table(**kw)
        n = len(t['frame'])
        rec = np.zeros(n, tabbin._DTYPE)
        rec['d'], rec['dlen'] = t['d'], np.linalg.norm(t['d'], axis=1)
        rec['intensity'], rec['frame'] = t['intensity'], t['frame']
        rec['px'], rec['py'] = t['px'] - 1, t['py'] - 1           # 0-based in the file
        rec['cenx'], rec['ceny'] = t['cenx'], t['ceny']
        header = bytearray(tabbin._HEADER)
        header[0:4] = n.to_bytes(4, 'little'); header[10:12] = (3).to_bytes(2, 'little')
        tab = tmp_path / 'run_peakhunt.tabbin'
        tab.write_bytes(bytes(header) + rec.tobytes())
        par = tmp_path / 'run_cracker.par'
        par.write_text('CRYSTALLOGRAPHY UB ' + ' '.join(f'{v:.12f}' for v in (UB * WL).ravel())
                       + f'\nCRYSTALLOGRAPHY WAVELENGTH {WL} {WL} {WL}\n', encoding='latin-1')

        icol, irow = calibrate._pixel_map(t['mapping'], NX, NY)(t['px'], t['py'])

        def open_frame(path):
            num = int(str(path)[1:-4])
            img = np.zeros((NY, NX), np.int32)
            m = t['frame'] == num
            img[irow[m] - 1, icol[m] - 1] = t['intensity'][m]
            return _Frame(img)

        det0 = FlatDetector(distance=DIST * 1.03, pixel_size=PS, beam_center=(BC[0] + 4, BC[1] - 3),
                            shape=(NY, NX), wavelength=WL)
        monkeypatch.setattr(fabio, 'open', open_frame)
        monkeypatch.setattr(calibrate, 'find_cbf_frames', lambda folder: (
            [f'f{i}.cbf' for i in range(1, NFRAMES + 1)], list(range(1, NFRAMES + 1))))
        monkeypatch.setattr(FlatDetector, 'from_eiger_cbf', staticmethod(
            lambda path: (det0, {'Phi': -180.0, 'Angle_increment': INCR}, None)))
        return str(par), str(tab), t
    return make


def _run(dataset, refine=True, **kw):
    par, tab, t = dataset(**kw)
    log = []
    fg = calibrate.geometry_from_tabbin('unused', par_path=par, tab_path=tab,
                                        refine=refine, log=log.append)
    return fg, t, log


# ──────────────────────────────────────────────────────────────────
# helper functions of the fit
# ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('name', calibrate._MAP_NAMES)
def test_pixel_map_and_its_inverse_with_fractions(name):
    assert len(set(calibrate._MAP_NAMES)) == 8
    rng = np.random.default_rng(3)
    px, py = rng.uniform(1, NY, 50), rng.uniform(1, NX, 50)
    if name[0] != 'T':
        px, py = rng.uniform(1, NX, 50), rng.uniform(1, NY, 50)
    col, row = calibrate._pixel_map(name, NX, NY)(px, py)
    assert np.all((col >= 0) & (col <= NX + 1) & (row >= 0) & (row <= NY + 1))
    back = _to_table(name, col, row)
    assert np.allclose(back[0], px, rtol=0, atol=1e-12) and np.allclose(back[1], py, rtol=0, atol=1e-12)


def test_pixel_map_values():
    px, py = np.array([10.25]), np.array([3.5])
    assert calibrate._pixel_map('---', 100, 80)(px, py) == (px, py)
    c, r = calibrate._pixel_map('TXY', 100, 80)(px, py)
    assert c[0] == 100 + 1 - 3.5 and r[0] == 80 + 1 - 10.25
    c, r = calibrate._pixel_map('-X-', 100, 80)(px, py)
    assert c[0] == 100 + 1 - 10.25 and r[0] == 3.5


@pytest.mark.parametrize('rot_sign', [1, -1])
def test_hkl_of_peaks_is_the_inverse_of_the_peak_model(rot_sign):
    rng = np.random.default_rng(1)
    col, row, frame, nodes = _peaks(_nodes(4), rot_sign * AXIS, rng)
    hkl = calibrate._hkl_of_peaks(_params(AXIS), col, row, (frame - 1) * INCR,
                                  UB, FC, FR, rot_sign, PS, WL)
    # the frame number gives the angle to 0.1 deg only
    assert np.abs(hkl - nodes).max() < 0.03
    assert np.median(np.abs(hkl - nodes).max(1)) < 0.006


def test_refine_reselect_finds_the_axis_and_ignores_half_integer_peaks():
    rng = np.random.default_rng(2)
    col, row, frame, _ = _peaks(_nodes(6), AXIS, rng, mosaic_deg=0.05)
    dphi = (frame - 1) * INCR
    start = _params(_tilted(AXIS, 0.3))
    args = (UB, FC, FR, 1, PS, WL)
    p, n, rms = calibrate._refine_reselect(start, col, row, dphi, np.ones(len(col), bool), *args)
    assert n > 500 and 0 < rms < 0.003
    assert _axis_error_deg(calibrate._sph(p[6], p[7])) < 0.02
    # more peaks at half-integer hkl do not change the result
    c2, r2, f2, _ = _peaks(_nodes(3, half=True), AXIS, rng)
    col2, row2, dphi2 = np.r_[col, c2], np.r_[row, r2], np.r_[dphi, (f2 - 1) * INCR]
    p2, n2, _ = calibrate._refine_reselect(start, col2, row2, dphi2, np.ones(len(col2), bool), *args)
    assert n2 == n and np.allclose(p2, p, rtol=0, atol=1e-9)
    # not enough peaks: no result
    few = np.zeros(len(col), bool); few[:calibrate.REFINE_MIN_PEAKS - 1] = True
    p3, n3, rms3 = calibrate._refine_reselect(start, col, row, dphi, few, *args)
    assert p3 is None and n3 == 0 and np.isnan(rms3)


def test_strongest_selects_by_rank_also_with_equal_intensities():
    inten = np.array([5, 1, 9, 9, 3, 7, 2, 8, 6, 4])
    assert np.flatnonzero(calibrate._strongest(inten, 0.3)).tolist() == [2, 3, 7]
    equal = np.full(1000, 77)
    m = calibrate._strongest(equal, 0.3)
    assert m.sum() == 300 and m[:300].all()          # equal values keep the table order
    assert calibrate._strongest(np.arange(7), 0.3).sum() == 3      # ceil(2.1)


def test_second_fit_is_better_uses_weights_and_a_fixed_seed():
    rng = np.random.default_rng(4)
    a1 = rng.uniform(0.0, 0.01, 800)
    w = rng.uniform(1e3, 1e5, 800)
    better, m1, m2, frac = calibrate._second_fit_is_better(a1, 0.8 * a1, w)
    assert better and frac == 1.0 and m2 == pytest.approx(0.8 * m1)
    assert calibrate._second_fit_is_better(a1, 0.8 * a1, w) == (better, m1, m2, frac)
    assert not calibrate._second_fit_is_better(a1, a1, w)[0]               # equal: not better
    assert not calibrate._second_fit_is_better(a1, 1.2 * a1, w)[0]         # worse
    noise = a1 + rng.normal(0.0, 0.002, 800)
    assert not calibrate._second_fit_is_better(a1, np.abs(noise), w)[0]   # no clear gain
    # many weak peaks better, one very strong peak much worse: the weighted mean is worse
    a2 = 0.5 * a1
    a2[0], w2 = 0.5, w.copy()
    w2[0] = 1e9
    assert not calibrate._second_fit_is_better(a1, a2, w2)[0]


def test_a_smaller_mean_without_confidence_is_not_better():
    rng = np.random.default_rng(1)
    a1 = rng.uniform(0.0, 0.01, 800)
    w = rng.uniform(1e3, 1e5, 800)
    a2 = np.abs(0.9 * a1 + rng.normal(0.0, 0.003, 800))     # a small gain in noise
    better, m1, m2, frac = calibrate._second_fit_is_better(a1, a2, w)
    assert m2 < m1                                          # the mean is smaller ...
    assert 0.5 < frac < calibrate.REFINE_CONFIDENCE         # ... but not in 95 % of the resamples
    assert not better
    assert calibrate._second_fit_is_better(a1, a2, w, confidence=0.5)[0]
    assert calibrate._second_fit_is_better(a1, a2, w)[3] == frac      # fixed seed


def test_the_bootstrap_keeps_the_pairs():
    rng = np.random.default_rng(6)
    a1 = rng.uniform(0.0, 0.02, 800)                        # large spread between the peaks
    better, _, _, frac = calibrate._second_fit_is_better(a1, a1 - 0.0005, np.ones(800))
    assert better and frac == 1.0                           # the same small gain at each peak


def test_refine_reselect_does_not_use_peaks_at_hkl_zero():
    rng = np.random.default_rng(2)
    col, row, frame, _ = _peaks(_nodes(6), AXIS, rng, mosaic_deg=0.05)
    dphi = (frame - 1) * INCR
    start = _params(_tilted(AXIS, 0.3))
    args = (UB, FC, FR, 1, PS, WL)
    _, n, _ = calibrate._refine_reselect(start, col, row, dphi, np.ones(len(col), bool), *args)
    # 40 entries at the direct beam: hkl = 0 at each rotation angle
    col2, row2 = np.r_[col, np.full(40, BC[0])], np.r_[row, np.full(40, BC[1])]
    dphi2 = np.r_[dphi, np.linspace(0.0, 300.0, 40)]
    _, n2, _ = calibrate._refine_reselect(start, col2, row2, dphi2, np.ones(len(col2), bool), *args)
    assert n2 == n


# ──────────────────────────────────────────────────────────────────
# geometry_from_tabbin with the second fit
# ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('rot_sign, mapping', [(1, 'TXY'), (-1, 'TXY'), (1, '-X-'), (-1, '---')])
def test_second_fit_finds_the_axis_that_the_first_selection_hides(dataset, rot_sign, mapping):
    first, _, _ = _run(dataset, refine=False, rot_sign=rot_sign, mapping=mapping)
    assert not first.refined and not first.centroids and first.n_peaks_refined == 0
    assert first.pixel_mapping == mapping
    err_first = _axis_error_deg(_fitted_axis(first), rot_sign)

    fg, t, log = _run(dataset, rot_sign=rot_sign, mapping=mapping)
    err = _axis_error_deg(_fitted_axis(fg), rot_sign)
    assert fg.refined and fg.centroids
    if rot_sign == -1:
        assert fg.sense == -1                     # the code path for sense -1 is used
    assert err_first > 0.03                       # the first fit is wrong ...
    assert err < 0.015 and err < err_first / 3    # ... and the second fit is correct
    assert fg.axis_change_deg > 0.03
    assert fg.n_peaks == fg.n_peaks_refined > 0
    assert abs(fg.detector.distance - DIST) < 0.2
    assert np.abs(np.asarray(fg.detector.beam_center) + 1 - BC).max() < 0.2
    assert any('The second fit is better.' in line for line in log)
    assert any(line.startswith('second fit: used. The rotation axis was corrected by') for line in log)
    # the weak entries are at wrong positions: the second fit did not use them
    assert fg.n_peaks_refined <= int((~t['weak']).sum())


def test_second_fit_excludes_the_held_out_peaks(dataset, monkeypatch):
    seen = {}
    orig, orig_check = calibrate._refine_reselect, calibrate._second_fit_is_better

    def spy(p, col, row, dphi, usable, *args, **kw):
        res = orig(p, col, row, dphi, usable, *args, **kw)
        seen.update(usable=usable.copy(), rms=res[2])
        return res

    def spy_check(a1, a2, w, **kw):
        seen.update(n_check=len(a1), w_check=np.asarray(w, float).copy())
        return orig_check(a1, a2, w, **kw)
    monkeypatch.setattr(calibrate, '_refine_reselect', spy)
    monkeypatch.setattr(calibrate, '_second_fit_is_better', spy_check)
    fg, t, _ = _run(dataset)
    usable = seen['usable']
    strong = calibrate._strongest(t['intensity'], calibrate.REFINE_STRONG_FRACTION)
    # the check peaks are strong peaks that the second fit did not use
    assert calibrate.REFINE_MIN_PEAKS <= seen['n_check'] <= int((strong & ~usable).sum())
    assert seen['w_check'].min() >= t['intensity'][strong].min()
    # each 5th peak of the table in frame order
    assert not usable[np.argsort(t['frame'], kind='stable')[4::5]].any()
    # the held-out clean peaks of the first fit
    hkl = np.linalg.solve(UB * WL, t['d'].T).T
    keep = np.flatnonzero(np.abs(hkl - np.round(hkl)).max(1) < 0.02)
    held_clean = keep[np.argsort(t['frame'][keep], kind='stable')[4::5]]
    assert not usable[held_clean].any()
    # only strong peaks
    assert not usable[~strong].any()
    # the rms of the result is the rms of the last pass
    assert fg.refined and fg.rms == seen['rms']


def test_centroids_with_the_wrong_sign_are_not_used(dataset):
    first, _, _ = _run(dataset, refine=False)
    fg, _, log = _run(dataset, cen_sign=-1.0)
    assert not fg.centroids
    assert any('sub-pixel centroids: not used' in line for line in log)
    assert _axis_error_deg(_fitted_axis(fg)) <= _axis_error_deg(_fitted_axis(first)) + 0.005


def test_centroids_that_are_not_finite_give_the_integer_pixel(dataset, monkeypatch):
    orig = calibrate.read_tabbin

    def with_nan(path):
        pk = orig(path)
        pk['cenx'] = pk['cenx'].astype(float)
        pk['cenx'][::20] = np.nan
        pk['ceny'] = pk['ceny'].astype(float)
        pk['ceny'][7::20] = np.inf
        return pk
    monkeypatch.setattr(calibrate, 'read_tabbin', with_nan)
    fg, _, log = _run(dataset)
    assert fg.centroids and fg.refined
    assert _axis_error_deg(_fitted_axis(fg)) < 0.015


def test_no_model_error_the_second_fit_does_no_harm(dataset):
    first, _, _ = _run(dataset, refine=False, table_axis_tilt=0.0)
    fg, _, _ = _run(dataset, table_axis_tilt=0.0)
    assert _axis_error_deg(_fitted_axis(fg)) <= _axis_error_deg(_fitted_axis(first)) + 0.005
    assert _axis_error_deg(_fitted_axis(fg)) < 0.03
    assert not fg.refined or fg.axis_change_deg < 0.02


def test_a_second_fit_that_is_not_better_is_not_used(dataset, monkeypatch):
    first, _, _ = _run(dataset, refine=False, table_axis_tilt=0.0)

    def small_turn(p, *args, **kw):                   # turn the axis by 0.015 deg
        q = np.array(p, float)
        q[6] += np.radians(0.015)
        return q, 999, 0.001
    monkeypatch.setattr(calibrate, '_refine_reselect', small_turn)
    fg, _, log = _run(dataset, table_axis_tilt=0.0)
    assert not fg.refined and fg.n_peaks_refined == 0 and fg.axis_change_deg == 0.0
    assert any('The second fit is not better.' in line for line in log)
    assert not any(line.startswith('second fit: used.') for line in log)
    assert _axis_error_deg(_fitted_axis(fg)) == pytest.approx(
        _axis_error_deg(_fitted_axis(first)), abs=0.005)

    monkeypatch.setattr(calibrate, '_refine_reselect', lambda p, *a, **k: (None, 0, float('nan')))
    fg, _, log = _run(dataset)
    assert not fg.refined
    assert any('second fit: not enough peaks' in line for line in log)

    def fails(*a, **k):
        raise ValueError('no convergence')
    monkeypatch.setattr(calibrate, '_refine_reselect', fails)
    fg, _, log = _run(dataset)
    assert not fg.refined
    assert any('second fit: error (ValueError: no convergence)' in line for line in log)


def test_second_fit_that_fails_the_self_check_is_not_used(dataset, monkeypatch):
    def bad(p, *args, **kw):                          # beam centre 40 pixels off
        q = np.array(p, float)
        q[0] += 40.0
        return q, 999, 1.0
    monkeypatch.setattr(calibrate, '_refine_reselect', bad)
    monkeypatch.setattr(calibrate, '_second_fit_is_better', lambda *a, **k: (True, 1.0, 0.5, 1.0))
    fg, _, log = _run(dataset)
    assert not fg.refined and fg.n_peaks_refined == 0
    assert any('fail the self-check' in line for line in log)
    assert not any(line.startswith('second fit: used.') for line in log)
    assert np.abs(np.asarray(fg.detector.beam_center) + 1 - BC).max() < 0.5
    assert fg.median_hkl_dev < calibrate.HKL_DEV_LIMIT
    assert fg.rms < 0.01                              # the rms of the fit before, not 1.0


@pytest.mark.parametrize('centroids, refined, method', [
    (False, False, 'tabbin'), (True, False, 'tabbin_centroids'),
    (False, True, 'tabbin_second_fit'), (True, True, 'tabbin_centroids_second_fit')])
def test_setup_records_the_parts_of_the_geometry_fit(tmp_path, monkeypatch, centroids, refined,
                                                     method):
    from rspace3d.rawrecon import engine
    (tmp_path / 'run_cracker.par').write_text('', encoding='latin-1')
    for i in range(1, 4):
        (tmp_path / f'run_{i:04d}.cbf').write_bytes(b'')
    det = FlatDetector(distance=DIST, pixel_size=PS, beam_center=(10.0, 8.0), shape=(16, 20),
                       wavelength=WL)
    fg = calibrate.FittedGeometry(det, UB, np.eye(3), (0.0, 1.0, 0.0), 1, -180.0, INCR, 0.001,
                                  100, 0.01, centroids=centroids, refined=refined)
    monkeypatch.setattr(calibrate, 'geometry_from_tabbin', lambda *a, **k: fg)
    setup = engine.prepare_setup(str(tmp_path), use_tabbin=True)
    assert setup.meta['geometry_method'] == method
