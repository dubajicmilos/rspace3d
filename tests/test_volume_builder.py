"""Regression tests for the audit findings ported into rspace3d 2.1.

Synthetic, CPU-only (GPU paths are exercised by the real-data benchmarks).
Run:  python -m pytest tests -q
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from rspace3d import volume_builder as vb
from rspace3d.volume_builder import (
    VolumeData, bin_2d_covered, bin_volume, extract_volume_slice, get_symmetry_operations,
    index_space_ops, laue_metric_residual, load_volume_h5, save_volume_h5,
    symmetrize_volume, volume_affine, _EXPECTED_ORDERS,
)


# ──────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────

def _cell_ub(a, b, c, alpha, beta, gamma, wavelength=0.5):
    """Lambda-scaled UB (rspace3d/CrysAlisPro convention) of a cell, B-matrix setting."""
    al, be, ga = np.radians([alpha, beta, gamma])
    va = np.array([a, 0.0, 0.0])
    vb_ = np.array([b * np.cos(ga), b * np.sin(ga), 0.0])
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    cz = np.sqrt(max(c ** 2 - cx ** 2 - cy ** 2, 0.0))
    direct = np.column_stack([va, vb_, np.array([cx, cy, cz])])
    recip = np.linalg.inv(direct).T          # columns a*, b*, c*
    return recip * wavelength, wavelength


def _regular_volume(cell, n=(9, 9, 7), step=0.5, data=None):
    """rawrecon-style regular hkl grid, symmetric about the origin."""
    ub, wl = _cell_ub(*cell)
    axes = [(np.arange(k) - (k - 1) / 2) * step for k in n]
    if data is None:
        data = np.zeros(n, dtype=np.float32)
    meta = {'ub': ub, 'wavelength': wl, 'grid_kind': 'hkl_regular',
            'cell': dict(zip('a b c alpha beta gamma'.split(), cell))}
    return VolumeData(data.astype(np.float32), axes[0], axes[1], axes[2], 'HK', meta)


def _hkl_grid(vol):
    origin, A = volume_affine(vol)
    ih, ik, il = np.meshgrid(np.arange(len(vol.H)), np.arange(len(vol.K)),
                             np.arange(len(vol.L)), indexing='ij')
    idx = np.stack([ih, ik, il], axis=-1).astype(float)
    return origin + idx @ A.T


def _sheared_unwarp_volume(shear=0.3, n=(11, 9, 5), dh=0.5, dk=0.5, dl=0.5):
    """HK-native unwarp raster with a monoclinic-style cross term h = H[i] + shear*K[j]."""
    H = (np.arange(n[0]) - (n[0] - 1) / 2) * dh
    K = (np.arange(n[1]) - (n[1] - 1) / 2) * dk
    L = (np.arange(n[2]) - (n[2] - 1) / 2) * dl
    # M_inv * s with M_inv[0,1]/M_inv[1,1] = shear; s arbitrary
    s = 0.1
    m_inv = np.array([[dh / s, shear * dk / s], [0.0, dk / s]])
    meta = {'M_inv': m_inv, 's': s, 'cx': (n[0] + 1) / 2, 'cy': (n[1] + 1) / 2,
            'grid_kind': 'unwarp_raster', 'wavelength': 0.5}
    vol = VolumeData(np.zeros(n, np.float32), H, K, L, 'HK', meta)
    return vol


# ──────────────────────────────────────────────────────────────────
# D1 / F1: complete and correct trigonal + hexagonal operations
# ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('group', list(_EXPECTED_ORDERS))
def test_group_closure_and_order(group):
    ops = get_symmetry_operations(group)
    assert len(ops) == _EXPECTED_ORDERS[group]
    keys = {tuple(o.ravel()) for o in ops}
    assert tuple(np.eye(3, dtype=int).ravel()) in keys
    for a in ops:
        assert round(abs(np.linalg.det(a))) == 1
        for b in ops:
            assert tuple((a @ b).ravel()) in keys


@pytest.mark.parametrize('group', ['-3', '-3m', '6/m', '6/mmm'])
def test_hexagonal_operations_preserve_q_on_standard_cell(group):
    """The Miller-index operators must keep |q| on a = b, gamma = 120 (F1)."""
    ub, wl = _cell_ub(4.0, 4.0, 6.5, 90, 90, 120)
    recip = ub / wl
    h = np.array(np.meshgrid(range(-3, 4), range(-3, 4), range(-2, 3))).reshape(3, -1)
    q0 = np.linalg.norm(recip @ h, axis=0)
    for op in get_symmetry_operations(group):
        q1 = np.linalg.norm(recip @ (op @ h), axis=0)
        assert np.allclose(q0, q1, atol=1e-12)
    vol = _regular_volume((4.0, 4.0, 6.5, 90, 90, 120))
    assert laue_metric_residual(vol, group) < 1e-12
    # the 2.0.0 direct-space tables fail this test on the same cell
    old_c3 = np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]])
    assert not np.allclose(q0, np.linalg.norm(recip @ (old_c3 @ h), axis=0))


def test_metric_residual_flags_wrong_group_but_allows_pseudo_symmetry():
    pseudo = _regular_volume((5.925, 5.926, 5.936, 89.98, 90.0, 89.99))
    assert laue_metric_residual(pseudo, 'mmm') < 0.01          # allowed
    assert laue_metric_residual(pseudo, 'm-3m') < 0.01
    tetra = _regular_volume((4.0, 4.0, 4.4, 90, 90, 90))
    assert laue_metric_residual(tetra, 'm-3m') > 0.05           # warned
    with pytest.warns(RuntimeWarning, match='does not preserve'):
        symmetrize_volume(tetra, 'm-3m', sigma=None, use_gpu=False)


@pytest.mark.parametrize('group', list(_EXPECTED_ORDERS))
def test_all_operations_applied_and_invariant_field_unchanged(group):
    """Every operation of every group lands on a symmetric regular grid, and a
    field built as an orbit sum is left unchanged by the projection."""
    cell = {'-3': (4, 4, 6.5, 90, 90, 120), '-3m': (4, 4, 6.5, 90, 90, 120),
            '6/m': (4, 4, 6.5, 90, 90, 120), '6/mmm': (4, 4, 6.5, 90, 90, 120),
            '2/m': (4, 5, 6, 90, 105, 90), '-1': (4, 5, 6, 85, 95, 100)}.get(
        group, (4.0, 4.0, 4.0, 90, 90, 90))
    vol = _regular_volume(cell, n=(9, 9, 9))
    ops = index_space_ops(vol, group)
    assert len(ops) == _EXPECTED_ORDERS[group]
    assert {o['kind'] for o in ops} <= {'perm', 'integer'}
    if group in ('-3', '-3m', '6/m', '6/mmm'):
        assert any(o['kind'] == 'integer' for o in ops)   # the 3-/6-fold maps
    hkl = _hkl_grid(vol)
    rng = np.random.default_rng(3)
    base = lambda p: np.sin(1.3 * p[..., 0]) + 0.5 * np.cos(0.7 * p[..., 1] + p[..., 2]) + \
        0.2 * p[..., 0] * p[..., 1]
    field = np.zeros(vol.intensity.shape)
    for o in ops:
        field += base(hkl @ o['op'].T.astype(float))
    field += 5.0
    vol.intensity = field.astype(np.float32)
    out = symmetrize_volume(vol, group, sigma=None, use_gpu=False)
    assert out.metadata['symmetry_ops_applied'] == _EXPECTED_ORDERS[group]
    assert out.metadata['symmetry_mapping'] == 'exact'
    finite = np.isfinite(out.intensity)
    assert finite.all()
    assert np.max(np.abs(out.intensity - vol.intensity)) < 1e-4 * np.abs(field).max()


def test_symmetrize_averages_orbit_members():
    vol = _regular_volume((4.0, 4.0, 4.0, 90, 90, 90), n=(5, 5, 5))
    data = np.full(vol.intensity.shape, np.nan, np.float32)
    data[3, 2, 2] = 10.0      # (h, k, l) = (0.5, 0, 0)
    data[2, 3, 2] = 30.0      # (0, 0.5, 0): m-3m mate
    vol.intensity = data
    out = symmetrize_volume(vol, 'm-3m', sigma=None, use_gpu=False)
    # every mate of the (100)-type orbit gets the mean of the two measurements
    assert out.intensity[3, 2, 2] == pytest.approx(20.0)
    assert out.intensity[1, 2, 2] == pytest.approx(20.0)
    assert out.intensity[2, 2, 3] == pytest.approx(20.0)


# ──────────────────────────────────────────────────────────────────
# D5: unmeasured orbits stay NaN
# ──────────────────────────────────────────────────────────────────

def test_all_nan_orbit_stays_nan():
    vol = _regular_volume((4.0, 4.0, 4.0, 90, 90, 90), n=(5, 5, 5))
    vol.intensity[:] = np.nan
    vol.intensity[2, 2, 2] = 1.0
    out = symmetrize_volume(vol, 'm-3m', sigma=3.0, use_gpu=False)
    assert np.isfinite(out.intensity).sum() == 1
    assert out.intensity[2, 2, 2] == 1.0


# ──────────────────────────────────────────────────────────────────
# MAD = 0: Poisson floor, unique outlier count
# ──────────────────────────────────────────────────────────────────

def _mmm_orbit_volume(values):
    """Put 8 values on the mmm orbit of (1, 1, 1) in a 5^3 regular grid."""
    vol = _regular_volume((4.0, 4.0, 4.0, 90, 90, 90), n=(5, 5, 5), step=1.0)
    data = np.full(vol.intensity.shape, np.nan, np.float32)
    for v, (sh, sk, sl) in zip(values, [(a, b, c) for a in (1, -1) for b in (1, -1)
                                        for c in (1, -1)]):
        data[2 + sh, 2 + sk, 2 + sl] = v
    vol.intensity = data
    return vol


def test_poisson_floor_keeps_counting_noise_but_rejects_hot_pixel():
    quantised = symmetrize_volume(_mmm_orbit_volume([0] * 7 + [1]), 'mmm',
                                  sigma=3.0, use_gpu=False)
    assert quantised.intensity[3, 3, 3] == pytest.approx(0.125)
    assert quantised.metadata['n_outliers_removed'] == 0

    noisy = symmetrize_volume(_mmm_orbit_volume([3, 4, 3, 3, 3, 4, 3, 5]), 'mmm',
                              sigma=3.0, use_gpu=False)
    assert noisy.intensity[3, 3, 3] == pytest.approx(28 / 8)
    assert noisy.metadata['n_outliers_removed'] == 0

    hot = symmetrize_volume(_mmm_orbit_volume([3] * 7 + [300]), 'mmm',
                            sigma=3.0, use_gpu=False)
    assert hot.intensity[3, 3, 3] == pytest.approx(3.0)
    assert hot.metadata['n_outliers_removed'] == 1     # counted once, not 8 times

    legacy = symmetrize_volume(_mmm_orbit_volume([0] * 7 + [1]), 'mmm', sigma=3.0,
                               use_gpu=False, poisson_floor=False)
    assert legacy.intensity[3, 3, 3] == pytest.approx(0.0)  # 2.0.0 behaviour


# ──────────────────────────────────────────────────────────────────
# D2: sheared (monoclinic) raster under 2/m -> interpolation
# ──────────────────────────────────────────────────────────────────

def test_sheared_raster_two_fold_is_interpolated_and_invariant_field_kept():
    vol = _sheared_unwarp_volume(shear=0.3)
    ops = index_space_ops(vol, '2/m')
    kinds = [o['kind'] for o in ops]
    assert kinds.count('interp') == 2 and kinds.count('perm') == 2  # 2_y, m_y need interpolation
    hkl = _hkl_grid(vol)
    # 2/m-invariant smooth field: depends on h^2, l^2, hl and k^2
    field = (2 + 0.3 * hkl[..., 0] ** 2 + 0.2 * hkl[..., 2] ** 2
             + 0.1 * hkl[..., 0] * hkl[..., 2] + 0.4 * hkl[..., 1] ** 2)
    vol.intensity = field.astype(np.float32)
    out = symmetrize_volume(vol, '2/m', sigma=None, use_gpu=False)
    assert out.metadata['symmetry_mapping'] == 'interpolated'
    inner = out.intensity[2:-2, 1:-1, 1:-1]
    ref = vol.intensity[2:-2, 1:-1, 1:-1]
    assert np.isfinite(inner).all()
    assert np.max(np.abs(inner - ref)) < 0.05 * np.abs(ref).max()   # linear interpolation error


def test_regular_grid_two_fold_is_exact_for_monoclinic_cell():
    vol = _regular_volume((4, 5, 6, 90, 105, 90))
    assert {o['kind'] for o in index_space_ops(vol, '2/m')} == {'perm'}


# ──────────────────────────────────────────────────────────────────
# D4: covered-pixel binning
# ──────────────────────────────────────────────────────────────────

def test_bin_2d_covered_ignores_unmeasured_subpixels():
    raw = np.array([[100, -1], [-1, -1]], dtype=np.int32)
    mask = raw >= 0
    assert bin_2d_covered(raw, mask, 2, 2)[0, 0] == 100.0
    raw = np.array([[0, 0], [0, 1]], dtype=np.int32)
    assert bin_2d_covered(raw, np.ones((2, 2), bool), 2, 2)[0, 0] == pytest.approx(0.25)
    assert np.isnan(bin_2d_covered(raw, np.zeros((2, 2), bool), 2, 2)[0, 0])


# ──────────────────────────────────────────────────────────────────
# D10 / V4 / V5: bin_volume metadata
# ──────────────────────────────────────────────────────────────────

def test_bin_volume_scales_raster_geometry_and_sums_counts():
    vol = _sheared_unwarp_volume(shear=0.0, n=(8, 8, 6))
    vol.metadata.update({'s': 0.5, 'cx': 4.5, 'cy': 4.5, 'bin_xy': 1, 'bin_z': 1})
    vol.intensity[:] = 2.0
    vol.intensity[0, 0, 0] = np.nan
    vol.counts = np.ones(vol.intensity.shape, np.int64)
    b = bin_volume(vol, 2, 2, 2)
    assert b.intensity.shape == (4, 4, 3)
    assert b.metadata['s'] == 1.0 and b.metadata['bin_xy'] == 2 and b.metadata['bin_z'] == 2
    assert b.metadata['cx'] == 2.5 and b.metadata['cy'] == 2.5
    assert b.counts[0, 0, 0] == 8 and b.intensity[0, 0, 0] == 2.0   # NaN excluded
    assert np.allclose(b.H, [-1.5, -0.5, 0.5, 1.5])
    t = bin_volume(vol, 2, 2, 4)
    assert t.metadata['bin_dropped_layers_hkl'] == [0, 0, 2]
    with pytest.raises(ValueError, match='in-plane'):
        bin_volume(vol, 3, 2, 1)
    sheared = _sheared_unwarp_volume(0.3, (8, 8, 6))
    assert np.allclose(volume_affine(bin_volume(sheared, 2, 2, 1))[1][:2, :2] / 2,
                       volume_affine(sheared)[1][:2, :2])       # shear survives binning


def _unwarp_like(nx, ny, nl, step=0.01):
    """HK raster with CrysAlisPro centring: h = 0 at pixel (nx+1)//2 + 0.5."""
    cx = (nx + 1) // 2 + 0.5
    cy = (ny + 1) // 2 + 0.5
    H = (np.arange(1, nx + 1) - cx) * step
    K = (np.arange(1, ny + 1) - cy) * step
    L = (np.arange(nl) - (nl - 1) / 2) * step
    meta = {'M_inv': np.eye(2) / step, 's': step * step, 'cx': cx, 'cy': cy,
            'grid_kind': 'unwarp_raster', 'ub': np.eye(3) * 0.5, 'wavelength': 0.5}
    return VolumeData(np.zeros((nx, ny, nl), np.float32), H, K, L, 'HK', meta)


@pytest.mark.parametrize('b', [2, 4, 8])
def test_binning_keeps_cubic_operations_exact_on_i19_like_raster(b):
    """2162 x 2068 pixel rasters binned by 2 put h = 0 on a bin centre and
    k = 0 on a bin edge (2.0.0 rounded the 90 deg rotations by half a voxel);
    aligned blocks keep every m-3m / 4/mmm operation an exact index map."""
    vol = _unwarp_like(86, 80, 2 * b + 1)     # same parity as 2162 x 2068: nx/2 odd, ny/2 even
    vol.intensity[:] = 1.0
    binned = bin_volume(vol, b, b, 1)
    assert {o['kind'] for o in index_space_ops(binned, '4/mmm')} == {'perm'}
    assert {o['kind'] for o in index_space_ops(binned, 'mmm')} == {'perm'}
    assert binned.metadata['s'] == pytest.approx(vol.metadata['s'] * b)
    # the recorded centre is consistent with the binned axis
    assert binned.metadata['cx'] == pytest.approx(1 - binned.H[0] / (binned.H[1] - binned.H[0]))
    assert binned.metadata['cy'] == pytest.approx(1 - binned.K[0] / (binned.K[1] - binned.K[0]))
    # unaligned blocks (2.0.0 layout) are not exact for this raster: the
    # rotations would fall back to nearest-voxel rounding (half a voxel off)
    naive_H = vb.bin_1d(vol.H, b); naive_K = vb.bin_1d(vol.K, b)
    naive = VolumeData(binned.intensity, naive_H, naive_K, binned.L, 'HK', binned.metadata)
    assert 'nearest' in {o['kind'] for o in index_space_ops(naive, '4/mmm')}


def test_layer_step_differing_from_pixel_step_uses_nearest_voxel_maps():
    """An unwarp raster has dl (layer spacing) != dh (pixel step): the cubic
    3-fold maps l onto h with a non-integer scale. These are permutation-
    structured, so they take the nearest voxel (2.0.0 semantics, no
    interpolation) and the rounding is recorded."""
    vol = _unwarp_like(24, 24, 15, step=0.01)
    vol.L = (np.arange(15) - 7) * 0.02
    kinds = {o['kind'] for o in index_space_ops(vol, 'm-3m')}
    assert kinds == {'perm', 'nearest'}
    vol.intensity[:] = 1.0
    out = symmetrize_volume(vol, 'm-3m', sigma=None, use_gpu=False)
    assert out.metadata['symmetry_mapping'] == 'nearest'
    assert 0 < out.metadata['symmetry_max_index_error'] <= 0.5
    assert np.allclose(out.intensity[np.isfinite(out.intensity)], 1.0)


# ──────────────────────────────────────────────────────────────────
# D3 / D7 / D11 / V1: slice extraction
# ──────────────────────────────────────────────────────────────────

def _linear(p):
    return 4 + 2 * p[..., 0] + 3 * p[..., 1] + 5 * p[..., 2]


@pytest.mark.parametrize('plane_index', [0, 1, 2])
def test_nonnative_slices_of_sheared_raster_match_analytic_field(plane_index):
    vol = _sheared_unwarp_volume(shear=0.3, n=(13, 9, 7))
    vol.intensity = _linear(_hkl_grid(vol)).astype(np.float32)
    targets = {0: 0.5, 1: 0.0, 2: 0.5}
    sl, x_ax, y_ax, xl, yl, fl, fv, n = extract_volume_slice(vol, plane_index, targets[plane_index])
    assert sl.shape == (len(y_ax), len(x_ax)) and n == 1
    X, Y = np.meshgrid(x_ax, y_ax)
    # native HK plane: the raster pixel (i, j) sits at h = H[i] + shear * K[j]
    # (displayed in Cartesian coordinates with tilted grid lines by the viewer);
    # the interpolated HL / KL planes are on the plain Miller axes.
    p = {0: (X + 0.3 * Y, Y, fv), 1: (X, fv, Y), 2: (fv, X, Y)}[plane_index]
    expect = 4 + 2 * p[0] + 3 * p[1] + 5 * p[2]
    ok = np.isfinite(sl)
    assert ok.sum() > 0.5 * sl.size
    assert np.max(np.abs(sl[ok] - expect[ok])) < 1e-4


def test_fully_measured_regular_grid_has_no_boundary_nan_and_exact_values():
    vol = _regular_volume((4.0, 4.0, 4.0, 90, 90, 90), n=(9, 7, 5), step=0.25)
    vol.intensity = _linear(_hkl_grid(vol)).astype(np.float32)
    for plane_index in (1, 2):
        sl, x_ax, y_ax, *_ = extract_volume_slice(vol, plane_index, 0.25)
        assert np.isfinite(sl).all()          # V1: no spurious edge NaN
        X, Y = np.meshgrid(x_ax, y_ax)
        p = (X, 0.25, Y) if plane_index == 1 else (0.25, X, Y)
        assert np.allclose(sl, 4 + 2 * p[0] + 3 * p[1] + 5 * p[2], atol=1e-5)


def test_hl_native_volume_slices_are_dimensionally_consistent():
    """D7: an HL-native raster is stored physically as (h, k, l)."""
    H = np.arange(6) * 0.5; K = np.arange(2) * 1.0; L = np.arange(4) * 0.5
    data = _linear(np.stack(np.meshgrid(H, K, L, indexing='ij'), -1)).astype(np.float32)
    vol = VolumeData(data, H, K, L, 'HL', {'grid_kind': 'hkl_regular'})
    sl, x_ax, y_ax, xl, yl, fl, fv, n = extract_volume_slice(vol, 1, 1.0)   # native HL
    assert (xl, yl, fl) == ('h', 'l', 'k') and sl.shape == (4, 6) and fv == 1.0
    X, Y = np.meshgrid(x_ax, y_ax)
    assert np.allclose(sl, 4 + 2 * X + 3 * 1.0 + 5 * Y)
    sl, x_ax, y_ax, xl, yl, *_ = extract_volume_slice(vol, 0, 0.5)         # non-native HK
    assert sl.shape == (2, 6)
    X, Y = np.meshgrid(x_ax, y_ax)
    assert np.allclose(sl, 4 + 2 * X + 3 * Y + 5 * 0.5)


def test_nonnative_cut_interpolates_at_the_requested_value():
    """k = 0 between two raster rows: the HL cut is their mean, not the nearer row."""
    nx, ny, nl = 8, 6, 3
    cx = (nx + 1) // 2 + 0.5; cy = (ny + 1) // 2 + 0.5
    H = (np.arange(1, nx + 1) - cx) * 0.5; K = (np.arange(1, ny + 1) - cy) * 0.5
    L = (np.arange(nl) - 1) * 0.5
    data = np.zeros((nx, ny, nl), np.float32)
    data[:, 3, :] = 1.0                       # rows 2 and 3 straddle k = 0
    data[:, 3, 1] = 2.0
    data[0, 2, 0] = np.nan                    # one missing neighbour
    vol = VolumeData(data, H, K, L, 'HK', {'M_inv': np.eye(2) / 0.5, 's': 0.25, 'cx': cx,
                                            'cy': cy, 'grid_kind': 'unwarp_raster'})
    sl, x_ax, y_ax, xl, yl, fl, fv, n = extract_volume_slice(vol, 1, 0.0)
    assert fv == 0.0 and sl.shape == (nl, nx)
    assert np.allclose(sl[0, 1:], 0.5) and np.allclose(sl[1], 1.0) and np.allclose(sl[2], 0.5)
    assert sl[0, 0] == 1.0                    # support-normalised: the finite neighbour alone


def test_integration_is_nan_aware():
    vol = _regular_volume((4.0, 4.0, 4.0, 90, 90, 90), n=(3, 3, 3), step=1.0)
    vol.intensity[:] = 1.0
    vol.intensity[:, :, 0] = np.nan
    sl, *_ , n = extract_volume_slice(vol, 0, 0.0, int_range=1.5)   # all three l layers
    assert n == 3
    assert np.allclose(sl, 3.0)                 # mean of the measured layers x n
    vol.intensity[:] = np.nan
    sl, *_ = extract_volume_slice(vol, 0, 0.0, int_range=1.5)
    assert np.isnan(sl).all()


# ──────────────────────────────────────────────────────────────────
# D14: HDF5 provenance round trip
# ──────────────────────────────────────────────────────────────────

def test_hdf5_round_trip_keeps_counts_and_provenance():
    vol = _regular_volume((4.0, 4.0, 4.0, 90, 90, 90), n=(5, 5, 5))
    vol.intensity[:] = np.random.default_rng(0).random(vol.intensity.shape)
    vol.counts = np.arange(125).reshape(5, 5, 5)
    vol.metadata['reconstructed_by'] = 'rspace3d.rawrecon'
    sym = symmetrize_volume(vol, 'mmm', sigma=3.0, use_gpu=False)
    sym.counts = vol.counts
    fd, path = tempfile.mkstemp(suffix='.h5'); os.close(fd)
    try:
        save_volume_h5(path, sym)
        back = load_volume_h5(path)
    finally:
        os.remove(path)
    assert np.array_equal(back.counts, vol.counts)
    assert back.metadata['grid_kind'] == 'hkl_regular'
    for key in ('sigma', 'min_valid', 'n_outliers_removed', 'symmetry_ops_applied',
                'symmetry_mapping', 'reconstructed_by', 'poisson_floor'):
        assert back.metadata[key] == sym.metadata[key], key
    assert np.array_equal(np.isnan(back.intensity), np.isnan(sym.intensity))
