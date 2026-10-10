"""Tests of the per-voxel orbit engine (CPU), the variance and the diagnostics.

The reference is `orbit_reference` (conftest.py): the rspace3d 2.2.0
algorithm with np.nanmedian, and the variance rule. The NumPy engine and the
numba engine (if numba is installed) must give the same bits.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from rspace3d import orbit_engine
from rspace3d import volume_builder as vb
from rspace3d.volume_builder import (
    _EDGE_SNAP_TOL, VolumeData, bin_volume, index_space_ops, load_volume_h5,
    reciprocal_metric, save_volume_h5, symmetrize_volume, variance_is_derived,
    volume_affine, voxel_variance,
)

BACKENDS = ['numpy'] + (['numba'] if orbit_engine.numba_available() else [])


# ──────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────

def _regular(cell, n, step=0.5):
    """Regular hkl grid, symmetric about the origin (rawrecon style)."""
    a, b, c, al, be, ga = cell
    al, be, ga = np.radians([al, be, ga])
    cz = np.sqrt(max(c ** 2 - (c * np.cos(be)) ** 2
                     - (c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)) ** 2, 0.0))
    direct = np.array([[a, b * np.cos(ga), c * np.cos(be)],
                       [0.0, b * np.sin(ga), c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)],
                       [0.0, 0.0, cz]])
    ub = np.linalg.inv(direct).T * 0.5
    axes = [(np.arange(k) - (k - 1) / 2) * step for k in n]
    meta = {'ub': ub, 'wavelength': 0.5, 'grid_kind': 'hkl_regular'}
    return VolumeData(np.zeros(n, np.float32), axes[0], axes[1], axes[2], 'HK', meta)


def _unwarp(nx, ny, nl, step=0.01, dl=None):
    """HK unwarp raster with CrysAlisPro centring."""
    cx = (nx + 1) // 2 + 0.5
    cy = (ny + 1) // 2 + 0.5
    H = (np.arange(1, nx + 1) - cx) * step
    K = (np.arange(1, ny + 1) - cy) * step
    L = (np.arange(nl) - (nl - 1) / 2) * (dl or step)
    meta = {'M_inv': np.eye(2) / step, 's': step * step, 'cx': cx, 'cy': cy,
            'grid_kind': 'unwarp_raster', 'ub': np.eye(3) * 0.5, 'wavelength': 0.5}
    return VolumeData(np.zeros((nx, ny, nl), np.float32), H, K, L, 'HK', meta)


def _sheared(shear=0.3, n=(11, 9, 7), d=0.5):
    """HK unwarp raster with a monoclinic cross term (2/m needs interpolation)."""
    axes = [(np.arange(k) - (k - 1) / 2) * d for k in n]
    s = 0.1
    meta = {'M_inv': np.array([[d / s, shear * d / s], [0.0, d / s]]), 's': s,
            'cx': (n[0] + 1) / 2, 'cy': (n[1] + 1) / 2, 'grid_kind': 'unwarp_raster',
            'wavelength': 0.5}
    return VolumeData(np.zeros(n, np.float32), axes[0], axes[1], axes[2], 'HK', meta)


def _fill(vol, seed, lam=20.0, nan_frac=0.1, n_hot=6):
    """Poisson data with unmeasured voxels and a few hot voxels."""
    rng = np.random.default_rng(seed)
    shape = vol.intensity.shape
    data = rng.poisson(lam, shape).astype(np.float32)
    data[rng.random(shape) < nan_frac] = np.nan
    hot = tuple(rng.integers(0, s, n_hot) for s in shape)
    data[hot] = 50 * lam
    vol.intensity = data
    return vol


CASES = {
    'perm': (lambda: _fill(_regular((4.0, 4.0, 4.0, 90, 90, 90), (9, 9, 9)), 1), 'm-3m'),
    'nearest': (lambda: _fill(_unwarp(16, 16, 11, dl=0.02), 2), 'm-3m'),
    'integer': (lambda: _fill(_regular((4.0, 4.0, 6.5, 90, 90, 120), (9, 9, 7)), 3), '6/mmm'),
    'interp': (lambda: _fill(_sheared(), 4), '2/m'),
}


def _plan(vol, group):
    ops = index_space_ops(vol, group)
    ident = next(i for i, o in enumerate(ops) if np.array_equal(o['op'], np.eye(3, dtype=int)))
    origin, A = volume_affine(vol)
    G, unit = reciprocal_metric(vol)
    return ops, orbit_engine.build_plan(vol.intensity.shape, ops, ident, origin, A, G, unit,
                                        _EDGE_SNAP_TOL)


def _bits(a):
    return np.asarray(a, np.float32).view(np.int32)


def _same(a, b):
    """Same NaN pattern and the same bits of the finite values."""
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    fa, fb = np.isfinite(a), np.isfinite(b)
    return bool(np.array_equal(fa, fb) and np.array_equal(_bits(a[fa]), _bits(b[fb])))


# ──────────────────────────────────────────────────────────────────
# engine = 2.2.0 reference, for each kind of orbit member
# ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('kind', list(CASES))
def test_engine_matches_2_2_0_reference(kind, backend, orbit_reference):
    make, group = CASES[kind]
    vol = make()
    ops, plan = _plan(vol, group)
    assert kind in {o['kind'] for o in ops}
    variance = np.maximum(vol.intensity, np.float32(1.0))
    ref, ref_var, ref_out = orbit_reference(vol, group, sigma=3.0, variance=variance)
    res = orbit_engine.run_cpu(plan, vol.intensity, variance, 3.0, 3, True, backend=backend)
    assert res.n_outliers_removed == ref_out > 0
    if kind == 'interp':
        assert np.array_equal(np.isnan(res.intensity), np.isnan(ref))
        assert np.allclose(res.intensity, ref, rtol=1e-6, atol=0, equal_nan=True)
    else:
        assert _same(res.intensity, ref)                     # bit for bit
    assert np.array_equal(np.isnan(res.variance), np.isnan(ref_var))
    assert np.allclose(res.variance, ref_var, rtol=1e-6, atol=0, equal_nan=True)
    # without variance: the same intensity, no variance, no chi^2
    plain = orbit_engine.run_cpu(plan, vol.intensity, None, 3.0, 3, True, backend=backend)
    assert _same(plain.intensity, res.intensity) and plain.variance is None
    diag = orbit_engine.result_diagnostics(plan, plain)
    assert diag['total']['chi2_per_nu'] is None and not diag['has_variance']


@pytest.mark.parametrize('kind', list(CASES))
def test_numba_and_numpy_engines_give_the_same_bits(kind):
    if 'numba' not in BACKENDS:
        pytest.skip('numba is not installed')
    make, group = CASES[kind]
    vol = make()
    _, plan = _plan(vol, group)
    stored = np.maximum(vol.intensity, np.float32(1.0))
    for variance, derive in ((None, False), (stored, False), (None, True)):
        a = orbit_engine.run_cpu(plan, vol.intensity, variance, 3.0, 3, True,
                                 derive_variance=derive, backend='numpy')
        b = orbit_engine.run_cpu(plan, vol.intensity, variance, 3.0, 3, True,
                                 derive_variance=derive, backend='numba')
        assert np.array_equal(_bits(a.intensity), _bits(b.intensity))
        if a.variance is None:
            assert b.variance is None
        else:
            assert np.array_equal(_bits(a.variance), _bits(b.variance))
        assert np.array_equal(a.int_sums, b.int_sums)
        assert np.allclose(a.float_sums, b.float_sums, rtol=1e-12, atol=0)
        assert a.engine == 'cpu-numpy' and b.engine == 'cpu-numba'


@pytest.mark.parametrize('kind', ['interp', 'perm'])
def test_small_numpy_boxes_and_slabs_give_the_same_bits(kind, monkeypatch):
    make, group = CASES[kind]
    vol = make()
    _, plan = _plan(vol, group)
    var = np.maximum(vol.intensity, np.float32(1.0))
    a = orbit_engine.run_cpu(plan, vol.intensity, var, 3.0, 3, True, backend='numpy')
    for slab in (1 << 22, 40, 200):     # collision test on slabs of the volume
        monkeypatch.setattr(orbit_engine, '_SLAB_VOXELS', slab)
        for chunk in (7, 30, 100, None):   # boxes of one h-layer and part of the k range
            b = orbit_engine.run_cpu(plan, vol.intensity, var, 3.0, 3, True,
                                     backend='numpy', chunk_voxels=chunk)
            assert np.array_equal(_bits(a.intensity), _bits(b.intensity))
            assert np.array_equal(_bits(a.variance), _bits(b.variance))
            assert np.array_equal(a.int_sums, b.int_sums)
            assert np.allclose(a.float_sums, b.float_sums, rtol=1e-12, atol=0)


# ──────────────────────────────────────────────────────────────────
# variance at special positions, chi^2 and R_sym
# ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('backend', BACKENDS)
def test_variance_at_special_positions_counts_duplicate_sources(backend):
    """mmm on h, k, l = -2..2: a voxel on h = 0 reads each source twice."""
    vol = _regular((4.0, 4.0, 4.0, 90, 90, 90), (5, 5, 5), step=1.0)
    rng = np.random.default_rng(7)
    vol.intensity = rng.uniform(5, 15, (5, 5, 5)).astype(np.float32)
    v = rng.uniform(1, 4, (5, 5, 5)).astype(np.float32)
    _, plan = _plan(vol, 'mmm')
    res = orbit_engine.run_cpu(plan, vol.intensity, v, None, 3, True, backend=backend)

    def expected(h, k, l):
        # distinct sources j with multiplicity c_j, m = 8: sum_j (c_j / 8)^2 v_j
        src = {(sh * h, sk * k, sl * l) for sh in (1, -1) for sk in (1, -1) for sl in (1, -1)}
        c = 8 // len(src)
        return sum((c / 8) ** 2 * float(v[2 + a, 2 + b, 2 + d]) for a, b, d in src)

    for hkl in [(0, 1, 1), (0, 0, 1), (0, 0, 0), (1, 1, 1), (0, 2, -1)]:
        i = tuple(2 + x for x in hkl)
        assert res.variance[i] == pytest.approx(expected(*hkl), rel=1e-6), hkl
    # h = 0, k = 1, l = 1: four sources, each read twice: v_mean / 4, not v_mean / 8
    four = [v[2, 2 + b, 2 + d] for b in (1, -1) for d in (1, -1)]
    assert res.variance[2, 3, 3] == pytest.approx(np.mean(four) / 4, rel=1e-6)
    assert res.variance[2, 2, 2] == pytest.approx(v[2, 2, 2], rel=1e-6)


@pytest.mark.parametrize('backend', BACKENDS)
def test_chi2_per_nu_is_near_one_for_poisson_data_with_the_true_variance(backend):
    vol = _regular((4.0, 4.0, 4.0, 90, 90, 90), (21, 21, 21), step=0.5)
    h, k, l = np.meshgrid(vol.H, vol.K, vol.L, indexing='ij')
    lam = 200 + 80 * np.cos(h) * np.cos(0.7 * k) * np.cos(1.3 * l)      # mmm-invariant
    rng = np.random.default_rng(8)
    vol.intensity = rng.poisson(lam).astype(np.float32)
    _, plan = _plan(vol, 'mmm')
    res = orbit_engine.run_cpu(plan, vol.intensity, lam.astype(np.float32), None, 3, True,
                               backend=backend)
    diag = orbit_engine.result_diagnostics(plan, res)
    assert diag['has_variance']
    assert diag['total']['chi2_per_nu'] == pytest.approx(1.0, abs=0.05)
    # a variance 4x too small gives chi^2 / nu near 4
    small = orbit_engine.run_cpu(plan, vol.intensity, (lam / 4).astype(np.float32), None, 3,
                                 True, backend=backend)
    small_diag = orbit_engine.result_diagnostics(plan, small)
    assert small_diag['total']['chi2_per_nu'] == pytest.approx(4.0, rel=0.05)


@pytest.mark.parametrize('backend', BACKENDS)
def test_r_sym_of_a_hand_example(backend):
    """One mmm orbit with the values 1 .. 8: R_sym = sum|x - 4.5| / sum|x| = 16 / 36."""
    vol = _regular((4.0, 4.0, 4.0, 90, 90, 90), (5, 5, 5), step=1.0)
    data = np.full((5, 5, 5), np.nan, np.float32)
    signs = [(a, b, c) for a in (1, -1) for b in (1, -1) for c in (1, -1)]
    for value, (a, b, c) in zip(range(1, 9), signs):
        data[2 + a, 2 + b, 2 + c] = value
    vol.intensity = data
    _, plan = _plan(vol, 'mmm')
    res = orbit_engine.run_cpu(plan, data, None, None, 3, True, backend=backend)
    t = orbit_engine.result_diagnostics(plan, res)['total']
    assert t['n_voxels'] == 8 and t['n_members'] == 64
    # each of the 8 voxels has the weight 1 / 8 (8 distinct sources): the orbit counts once
    assert t['rsym_num'] == pytest.approx(16.0) and t['rsym_den'] == pytest.approx(36.0)
    assert t['r_sym'] == pytest.approx(16 / 36)
    assert np.all(res.intensity[np.isfinite(data)] == 4.5)


# ──────────────────────────────────────────────────────────────────
# lazy (derived) variance = stored variance
# ──────────────────────────────────────────────────────────────────

def _derived(vol):
    vol.metadata['variance_model'] = 'poisson_stored_units'
    assert variance_is_derived(vol)
    stored = VolumeData(vol.intensity, vol.H, vol.K, vol.L, vol.plane_type, dict(vol.metadata),
                        variance=voxel_variance(vol))
    assert not variance_is_derived(stored)
    return vol, stored


@pytest.mark.parametrize('kind', ['nearest', 'interp'])
def test_lazy_variance_gives_the_same_bits_as_the_stored_variance(kind, monkeypatch):
    make, group = CASES[kind]
    vol, stored = _derived(make())
    v = voxel_variance(vol)
    assert v.dtype == np.float32 and v is not vol.intensity
    assert np.array_equal(np.isnan(v), np.isnan(vol.intensity))
    assert np.all(v[np.isfinite(v)] >= 1.0)
    for backend in BACKENDS:
        monkeypatch.setattr(orbit_engine, 'numba_available', lambda: backend == 'numba')
        a = symmetrize_volume(vol, group, sigma=3.0, use_gpu=False)
        b = symmetrize_volume(stored, group, sigma=3.0, use_gpu=False)
        assert np.array_equal(_bits(a.intensity), _bits(b.intensity))
        assert np.array_equal(_bits(a.variance), _bits(b.variance))
        assert a.metadata['n_outliers_removed'] == b.metadata['n_outliers_removed']
        assert a.metadata['variance_model'] == 'poisson_stored_units'
        assert a.metadata['symmetry_diagnostics']['has_variance']
    # binning: the derived variance is calculated slab by slab
    monkeypatch.setattr(vb, '_BIN_SLAB_VOXELS', 50)
    a = bin_volume(vol, 1, 1, 2)
    b = bin_volume(stored, 1, 1, 2)
    monkeypatch.setattr(vb, '_BIN_SLAB_VOXELS', 1 << 25)
    c = bin_volume(vol, 1, 1, 2)
    for x, y in ((a, b), (a, c)):
        assert np.array_equal(_bits(x.intensity), _bits(y.intensity))
        assert np.array_equal(_bits(x.variance), _bits(y.variance))
        assert np.array_equal(x.counts, y.counts)
    assert a.counts.dtype == np.uint16 and not variance_is_derived(a)


def test_compact_counts_and_their_hdf5_round_trip():
    vol = _fill(_unwarp(16, 16, 12), 9)
    b = bin_volume(vol, 2, 2, 2)
    assert b.counts.dtype == np.uint16 and b.counts.max() <= 8
    assert np.array_equal(b.counts == 0, np.isnan(b.intensity))
    # a second binning weights by these counts (int64 output)
    b2 = bin_volume(b, 1, 1, 2)
    assert b2.counts.dtype == np.int64
    fd, path = tempfile.mkstemp(suffix='.h5'); os.close(fd)
    try:
        save_volume_h5(path, b)
        back = load_volume_h5(path)
    finally:
        os.remove(path)
    assert back.counts.dtype == np.uint16 and np.array_equal(back.counts, b.counts)
    assert np.array_equal(bin_volume(back, 1, 1, 2).intensity, b2.intensity, equal_nan=True)


# ──────────────────────────────────────────────────────────────────
# HDF5: diagnostics, files of rspace3d 2.2.0
# ──────────────────────────────────────────────────────────────────

def test_hdf5_round_trip_keeps_the_diagnostics():
    make, group = CASES['nearest']
    vol, _ = _derived(make())
    sym = symmetrize_volume(vol, group, sigma=3.0, use_gpu=False)
    diag = sym.metadata['symmetry_diagnostics']
    fd, path = tempfile.mkstemp(suffix='.h5'); os.close(fd)
    try:
        save_volume_h5(path, sym)
        back = load_volume_h5(path)
    finally:
        os.remove(path)
    got = back.metadata['symmetry_diagnostics']
    assert got['q_unit'] == diag['q_unit'] and got['has_variance'] == diag['has_variance']
    assert got['shell_edges'] == pytest.approx(diag['shell_edges'])
    for part in ('shells', 'total'):
        for key, value in diag[part].items():
            if isinstance(value, list):
                assert got[part][key] == pytest.approx(value, nan_ok=True), (part, key)
            else:
                assert got[part][key] == pytest.approx(value), (part, key)
    # the symmetrised variance is stored explicitly (as /uncertainty)
    assert back.variance is not None and not variance_is_derived(back)
    assert np.allclose(back.variance, sym.variance, rtol=1e-6, equal_nan=True)
    text = vb.format_symmetry_diagnostics(got)
    assert 'R_sym' in text and 'chi2/nu' in text and len(text.splitlines()) == 14
    assert ';' not in text


def _write_2_2_0_file(path, vol, laue_group=None):
    """An HDF5 file as rspace3d 2.2.0 wrote it: no /uncertainty, no variance model."""
    import h5py
    with h5py.File(path, 'w') as f:
        f.create_dataset('data', data=vol.intensity)
        for name in 'HKL':
            f.create_dataset(name, data=getattr(vol, name))
        f.create_dataset('M_inv', data=vol.metadata['M_inv'])
        f.create_dataset('UB', data=vol.metadata['ub'])
        f.attrs['plane_type'] = vol.plane_type
        f.attrs['wavelength'] = vol.metadata['wavelength']
        f.attrs['grid_kind'] = 'unwarp_raster'
        f.attrs['bin_xy'] = 1
        f.attrs['bin_z'] = 1
        f.attrs['s'] = vol.metadata['s']
        f.attrs['cx'] = vol.metadata['cx']
        f.attrs['cy'] = vol.metadata['cy']
        if laue_group:
            f.attrs['laue_group'] = laue_group
            f.attrs['symmetry_ops_applied'] = 48


def test_a_2_2_0_file_without_variance_loads():
    vol = _fill(_unwarp(16, 16, 11, dl=0.02), 10)
    fd, path = tempfile.mkstemp(suffix='.h5'); os.close(fd)
    try:
        _write_2_2_0_file(path, vol)
        raw = load_volume_h5(path)
        _write_2_2_0_file(path, vol, laue_group='m-3m')
        sym = load_volume_h5(path)
    finally:
        os.remove(path)
    # a raw unbinned unwarp volume: the variance max(I, 1) is derived
    assert raw.variance is None and raw.counts is None
    assert raw.metadata['variance_model'] == 'poisson_stored_units' and variance_is_derived(raw)
    assert np.array_equal(voxel_variance(raw), np.maximum(vol.intensity, 1), equal_nan=True)
    out = symmetrize_volume(raw, 'm-3m', sigma=3.0, use_gpu=False)
    assert out.variance is not None
    # a symmetrised 2.2.0 volume has no known variance
    assert 'variance_model' not in sym.metadata and voxel_variance(sym) is None
    out = symmetrize_volume(sym, 'm-3m', sigma=3.0, use_gpu=False)
    assert out.variance is None and not out.metadata['symmetry_diagnostics']['has_variance']


# ──────────────────────────────────────────────────────────────────
# options of symmetrize_volume, diagnostics weights, GPU failures
# ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('kind', ['nearest', 'interp'])
def test_propagate_variance_false_gives_the_2_2_0_output(kind, orbit_reference):
    make, group = CASES[kind]
    vol, _ = _derived(make())
    full = symmetrize_volume(vol, group, sigma=3.0, use_gpu=False)
    plain = symmetrize_volume(vol, group, sigma=3.0, use_gpu=False, propagate_variance=False)
    assert plain.variance is None and full.variance is not None
    assert 'variance_model' not in plain.metadata
    assert not plain.metadata['symmetry_diagnostics']['has_variance']
    assert plain.metadata['symmetry_diagnostics']['total']['chi2_per_nu'] is None
    assert np.array_equal(_bits(plain.intensity), _bits(full.intensity))
    assert plain.metadata['n_outliers_removed'] == full.metadata['n_outliers_removed']
    ref, _, ref_out = orbit_reference(vol, group, sigma=3.0)
    assert plain.metadata['n_outliers_removed'] == ref_out
    if kind != 'interp':
        assert _same(plain.intensity, ref)


@pytest.mark.parametrize('backend', BACKENDS)
def test_non_integer_min_valid_keeps_the_2_2_0_rule(backend, monkeypatch, orbit_reference):
    monkeypatch.setattr(orbit_engine, 'numba_available', lambda: backend == 'numba')
    vol = _regular((4.0, 4.0, 4.0, 90, 90, 90), (9, 9, 9))
    rng = np.random.default_rng(0)
    data = rng.poisson(20, (9, 9, 9)).astype(np.float32)
    data[rng.random(data.shape) < 0.55] = np.nan          # many orbits with 3 members
    data[rng.random(data.shape) < 0.05] = 900.0
    vol.intensity = data
    counts = []
    for mv in (3, 3.5, 4, 1e12, -2.5):
        new = symmetrize_volume(vol, 'mmm', sigma=3.0, min_valid=mv, use_gpu=False)
        old, _, n_old = orbit_reference(vol, 'mmm', sigma=3.0, min_valid=mv)
        assert new.metadata['n_outliers_removed'] == n_old, mv
        assert _same(new.intensity, old), mv
        counts.append(n_old)
    assert counts[1] == counts[2] < counts[0] and counts[3] == 0


@pytest.mark.parametrize('backend', BACKENDS)
def test_diagnostics_count_an_exact_orbit_once(backend):
    """mmm on h, k, l = -2..2: 27 orbits, an orbit with z zero indices has 2^(3 - z) sources."""
    vol = _regular((4.0, 4.0, 4.0, 90, 90, 90), (5, 5, 5), step=1.0)
    rng = np.random.default_rng(3)
    data = rng.uniform(5, 15, (5, 5, 5)).astype(np.float32)
    _, plan = _plan(vol, 'mmm')
    res = orbit_engine.run_cpu(plan, data, np.ones_like(data), None, 3, True, backend=backend)
    t = orbit_engine.result_diagnostics(plan, res)['total']
    assert t['n_voxels'] == 125 and t['n_members'] == 125 * 8
    # nu = sum over the orbits of (number of distinct sources - 1) = 5^3 - 27
    assert t['nu'] == pytest.approx(98.0, rel=1e-12)
    # R_sym sums: each orbit once
    num = den = chi = 0.0
    seen = set()
    for idx in np.ndindex(5, 5, 5):
        hkl = tuple(abs(i - 2) for i in idx)
        if hkl in seen:
            continue
        seen.add(hkl)
        x = np.array([data[2 + a * hkl[0], 2 + b * hkl[1], 2 + c * hkl[2]]
                      for a in (1, -1) for b in (1, -1) for c in (1, -1)], np.float64)
        mean = float(np.float32(np.float32(x.astype(np.float32).sum(dtype=np.float32)) / 8))
        num += np.abs(x - mean).sum()
        den += np.abs(x).sum()
        chi += ((np.unique(x) - mean) ** 2).sum()
    assert len(seen) == 27
    assert t['rsym_num'] == pytest.approx(num, rel=1e-6)
    assert t['rsym_den'] == pytest.approx(den, rel=1e-12)
    assert t['chi2'] == pytest.approx(chi, rel=1e-6)


def test_bin_volume_drops_the_symmetrisation_diagnostics():
    make, group = CASES['nearest']
    vol, _ = _derived(make())
    sym = symmetrize_volume(vol, group, sigma=3.0, use_gpu=False)
    assert {'symmetry_diagnostics', 'n_outliers_removed',
            'symmetry_max_index_error'} <= set(sym.metadata)
    b = bin_volume(sym, 2, 2, 1)
    for key in ('symmetry_diagnostics', 'n_outliers_removed', 'symmetry_max_index_error'):
        assert key not in b.metadata
    assert b.metadata['laue_group'] == group and 'symmetry_diagnostics' in sym.metadata


def test_gpu_errors_fall_back_to_the_cpu_engine(monkeypatch):
    cp = pytest.importorskip('cupy')
    from cupy.cuda import compiler
    make, group = CASES['interp']
    vol = make()
    _, plan = _plan(vol, group)
    var = np.maximum(vol.intensity, np.float32(1.0))
    ref = orbit_engine.run_cpu(plan, vol.intensity, var, 3.0, 3, True)

    def compile_error():
        raise compiler.CompileException('error: test', 'source', 'orbit.cu', ())

    def driver_error(*args):
        raise cp.cuda.driver.CUDADriverError(700)        # an illegal address in a kernel

    def other_error():
        raise KeyError('not a CUDA error')

    cases = ({'_kernel': compile_error},
             {'_kernel': lambda: None, '_upload': driver_error})
    for patches in cases:
        with monkeypatch.context() as m:
            for name, fn in patches.items():
                m.setattr(orbit_engine, name, fn)
            with pytest.warns(RuntimeWarning, match='GPU engine failed'):
                res = orbit_engine.run(plan, vol.intensity, var, 3.0, 3, True, use_gpu=True)
        assert res.engine.startswith('cpu')
        assert np.array_equal(_bits(res.intensity), _bits(ref.intensity))
        assert np.array_equal(_bits(res.variance), _bits(ref.variance))
    # other errors are not hidden
    with monkeypatch.context() as m:
        m.setattr(orbit_engine, '_kernel', other_error)
        with pytest.raises(KeyError):
            orbit_engine.run(plan, vol.intensity, var, 3.0, 3, True, use_gpu=True)
