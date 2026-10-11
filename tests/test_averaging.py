"""Tests of the averaging options (weighting, rejection, noise model) and the scores.

All tests run on the CPU. The NumPy engine and the numba engine (if numba is
installed) must give the same bits for every option.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from rspace3d import orbit_engine
from rspace3d.averaging import (
    BIAS_REFERENCE, NoiseModel, automatic_candidates, bias_checks, fit_members,
    fit_noise_model, gradient_magnitude, preselect, sample_orbits, score_candidates, suggest,
)
from rspace3d.volume_builder import (
    VolumeData, load_volume_h5, orbit_plan, save_volume_h5, symmetrize_volume,
)
from test_orbit_engine import BACKENDS, CASES, _bits, _plan, _regular, _same

OPTIONS = [(w, r) for w in ('equal', 'counts', 'adaptive') for r in ('none', 'mad', 'model')]


def _with_counts(vol, seed):
    rng = np.random.default_rng(seed)
    vol.counts = rng.integers(1, 400, vol.intensity.shape).astype(np.uint16)
    return vol


# ──────────────────────────────────────────────────────────────────
# defaults = rspace3d 2.3.0
# ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('kind', list(CASES))
def test_defaults_give_the_2_3_0_result(kind, orbit_reference):
    make, group = CASES[kind]
    vol = _with_counts(make(), 3)                    # counts must not change 'equal'
    ref, _, ref_out = orbit_reference(vol, group, sigma=3.0)
    out = symmetrize_volume(vol, group, use_gpu=False)
    explicit = symmetrize_volume(vol, group, use_gpu=False, weighting='equal', rejection='mad')
    if kind == 'interp':
        assert np.allclose(out.intensity, ref, rtol=1e-6, atol=0, equal_nan=True)
    else:
        assert _same(out.intensity, ref)
    assert out.metadata['n_outliers_removed'] == ref_out
    assert _same(explicit.intensity, out.intensity)
    assert not {'weighting', 'rejection', 'noise_model'} & set(out.metadata)
    # sigma None: no rejection, the same as rejection='none'
    plain = symmetrize_volume(vol, group, sigma=None, use_gpu=False)
    none = symmetrize_volume(vol, group, sigma=3.0, rejection='none', use_gpu=False)
    assert _same(plain.intensity, none.intensity)
    assert 'sigma' not in none.metadata and none.metadata['rejection'] == 'none'


def test_rejection_needs_a_sigma():
    vol = CASES['perm'][0]()
    for rej in ('mad', 'model'):
        with pytest.raises(ValueError):
            symmetrize_volume(vol, 'm-3m', sigma=None, rejection=rej, use_gpu=False)
    with pytest.raises(ValueError):
        symmetrize_volume(vol, 'm-3m', weighting='median', use_gpu=False)


# ──────────────────────────────────────────────────────────────────
# 'counts' and 'adaptive'
# ──────────────────────────────────────────────────────────────────

def _mmm_counts():
    """mmm on h, k, l = -2..2 with random values, variances and counts."""
    vol = _regular((4.0, 4.0, 4.0, 90, 90, 90), (5, 5, 5), step=1.0)
    rng = np.random.default_rng(11)
    vol.intensity = rng.uniform(5, 15, (5, 5, 5)).astype(np.float32)
    vol.variance = rng.uniform(1, 4, (5, 5, 5)).astype(np.float32)
    vol.counts = rng.integers(1, 50, (5, 5, 5)).astype(np.uint16)
    return vol


@pytest.mark.parametrize('backend', BACKENDS)
def test_counts_weighting_equals_a_hand_calculation(backend, monkeypatch):
    monkeypatch.setattr(orbit_engine, 'numba_available', lambda: backend == 'numba')
    vol = _mmm_counts()
    out = symmetrize_volume(vol, 'mmm', sigma=None, weighting='counts', use_gpu=False)
    x = vol.intensity.astype(np.float64)
    v = vol.variance.astype(np.float64)
    n = vol.counts.astype(np.float64)
    for hkl in [(1, 1, 1), (1, -2, 2), (0, 1, 1), (0, 0, 1), (0, 0, 0)]:
        # sources of the 8 operations; a special position reads a source more than once
        src = [(2 + a * hkl[0], 2 + b * hkl[1], 2 + c * hkl[2])
               for a in (1, -1) for b in (1, -1) for c in (1, -1)]
        w = np.array([n[s] for s in src])
        mean = np.float32(sum(n[s] * x[s] for s in src) / w.sum())
        i = tuple(2 + q for q in hkl)
        assert out.intensity[i] == pytest.approx(float(mean), rel=1e-6), hkl
        # sum_j (W_j / W)^2 var_j over the distinct sources j
        var = sum((src.count(s) * n[s] / w.sum()) ** 2 * v[s] for s in set(src))
        assert out.variance[i] == pytest.approx(var, rel=1e-6), hkl


@pytest.mark.parametrize('backend', BACKENDS)
def test_counts_and_adaptive_without_counts(backend, monkeypatch):
    """Without counts N = 1: 'counts' is the plain mean, and 'adaptive' with
    only the gain too (all members of a voxel have the same weight)."""
    monkeypatch.setattr(orbit_engine, 'numba_available', lambda: backend == 'numba')
    vol = _mmm_counts()
    vol.counts = None
    a = symmetrize_volume(vol, 'mmm', sigma=None, weighting='counts', use_gpu=False)
    b = symmetrize_volume(vol, 'mmm', sigma=None, use_gpu=False)
    c = symmetrize_volume(vol, 'mmm', sigma=None, weighting='adaptive', use_gpu=False,
                          noise_model=NoiseModel(gain=3.0))
    for out in (a, c):
        assert np.allclose(out.intensity, b.intensity, rtol=1e-6)
        assert np.allclose(out.variance, b.variance, rtol=1e-6)


@pytest.mark.parametrize('kind', ['perm', 'interp'])
def test_adaptive_with_only_the_gain_equals_counts(kind):
    """rel = kappa = floor = 0: w_i = N_i / (gain max(med, 1)), the counts weights up to a factor."""
    make, group = CASES[kind]
    vol = _with_counts(make(), 5)
    vol.variance = np.maximum(vol.intensity, np.float32(1.0))
    a = symmetrize_volume(vol, group, sigma=None, weighting='adaptive', use_gpu=False,
                          noise_model=NoiseModel(gain=2.5))
    b = symmetrize_volume(vol, group, sigma=None, weighting='counts', use_gpu=False)
    assert np.array_equal(np.isnan(a.intensity), np.isnan(b.intensity))
    assert np.allclose(a.intensity, b.intensity, rtol=1e-6, atol=0, equal_nan=True)
    assert np.allclose(a.variance, b.variance, rtol=1e-5, atol=0, equal_nan=True)
    assert a.metadata['noise_model']['gain'] == 2.5 and not a.metadata['noise_model']['fitted']


# ──────────────────────────────────────────────────────────────────
# 'model' rejection and the gradient guard
# ──────────────────────────────────────────────────────────────────

def _displaced_peaks():
    """mmm grid with sharp peaks near (+-1, +-1, +-1) and one spike.

    Each peak is displaced by the same small vector in all octants, so the
    orbit members sample the peaks at different sub-voxel positions.
    """
    ax = (np.arange(31) - 15) * 0.1
    vol = VolumeData(np.zeros((31, 31, 31), np.float32), ax.copy(), ax.copy(), ax.copy(), 'HK',
                     {'grid_kind': 'hkl_regular'})
    h, k, l = np.meshgrid(ax, ax, ax, indexing='ij')
    mu = 50.0 + 0 * h
    for sh in (1, -1):
        for sk in (1, -1):
            for sl in (1, -1):
                d2 = (h - sh - 0.03) ** 2 + (k - sk - 0.02) ** 2 + (l - sl - 0.04) ** 2
                mu = mu + 5000.0 * np.exp(-d2 / (2 * 0.06 ** 2))
    rng = np.random.default_rng(0)
    x = mu + np.sqrt(mu) * rng.standard_normal(mu.shape)
    spike = (3, 22, 9)
    x[spike] += 2000.0
    vol.intensity = x.astype(np.float32)
    near = ((np.abs(np.abs(h) - 1) <= 0.25) & (np.abs(np.abs(k) - 1) <= 0.25)
            & (np.abs(np.abs(l) - 1) <= 0.25))
    return vol, spike, near


def test_model_rejects_a_spike_and_keeps_a_sharp_peak_that_mad_clips():
    vol, spike, near = _displaced_peaks()
    ref = symmetrize_volume(vol, 'mmm', sigma=None, use_gpu=False)
    mad = symmetrize_volume(vol, 'mmm', sigma=3.0, use_gpu=False)
    guard = NoiseModel(gain=1.0, kappa=2.0)
    model = symmetrize_volume(vol, 'mmm', sigma=3.0, rejection='model', noise_model=guard,
                              use_gpu=False)
    no_guard = symmetrize_volume(vol, 'mmm', sigma=3.0, rejection='model', use_gpu=False,
                                 noise_model=NoiseModel(gain=1.0))

    def ratio(out):
        return float(np.nansum(out.intensity[near]) / np.nansum(ref.intensity[near]))

    assert ratio(mad) < 0.9                      # MAD clips the peak
    assert ratio(model) > 0.97                   # the gradient guard keeps it
    assert ratio(no_guard) < 0.9                 # without the guard the model clips it too
    assert ref.intensity[spike] > 250            # the spike is in the plain mean
    assert abs(model.intensity[spike] - 50.0) < 10    # 'model' rejects it
    assert model.metadata['rejection'] == 'model' and 'poisson_floor' not in model.metadata


def test_model_rejection_keeps_the_members_nearest_the_median():
    """Two groups of 4 members, 100 and 200: the median 150 is far from all members."""
    av = orbit_engine.Averaging('equal', 'model', (1.0, 0.0, 0.0, 0.0))
    for arr, kept in (([100] * 4 + [200] * 4, 8), ([100, 101, 180, 181], 2),
                      ([10, 100, 100, 100, 200, 200, 200, 900], 6)):
        vals = np.array(arr, np.float32)[:, None]
        est = orbit_engine.estimate(vals, None, None, 3.0, 3, True, av)
        assert int(est.m[0]) == kept and np.isfinite(est.out[0]), arr


@pytest.mark.parametrize('backend', BACKENDS)
def test_model_rejection_does_not_empty_a_bimodal_orbit_in_the_engines(backend, monkeypatch):
    monkeypatch.setattr(orbit_engine, 'numba_available', lambda: backend == 'numba')
    vol = _regular((4.0, 4.0, 4.0, 90, 90, 90), (5, 5, 5), step=1.0)
    h = vol.H[:, None, None] + 0 * vol.K[None, :, None] + 0 * vol.L[None, None, :]
    vol.intensity = np.where(h > 0, 200.0, np.where(h < 0, 100.0, 150.0)).astype(np.float32)
    out = symmetrize_volume(vol, 'mmm', sigma=3.0, rejection='model', use_gpu=False,
                            noise_model=NoiseModel(gain=1.0))
    assert np.all(out.intensity == 150.0)
    assert out.metadata['n_outliers_removed'] == 0


def test_gradient_is_positive_at_a_peak_maximum_and_the_backends_agree():
    f = np.full((9, 8, 7), 10.0, np.float32)
    f[4, 3, 3] = 1000.0
    f[3:6, 3, 3] += 300.0
    g = gradient_magnitude(f, backend='numpy')
    assert g[4, 3, 3] > 0                        # a minmod slope is 0 here
    assert g[4, 3, 3] == np.sqrt(np.float32(990.0 ** 2 + 2 * 1290.0 ** 2))
    rng = np.random.default_rng(1)
    r = rng.normal(50, 10, (13, 11, 9)).astype(np.float32)
    r[rng.random(r.shape) < 0.2] = np.nan
    a = gradient_magnitude(r, backend='numpy', slab=4)
    assert np.all(a[np.isnan(r)] == 0) and np.all(np.isfinite(a))
    if 'numba' in BACKENDS:
        assert np.array_equal(_bits(a), _bits(gradient_magnitude(r, backend='numba')))
    assert np.array_equal(_bits(a), _bits(gradient_magnitude(r, backend='numpy', slab=32)))


# ──────────────────────────────────────────────────────────────────
# engines: NumPy = numba for every option
# ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('kind', list(CASES))
def test_numba_and_numpy_give_the_same_bits_for_every_option(kind):
    if 'numba' not in BACKENDS:
        pytest.skip('numba is not installed')
    make, group = CASES[kind]
    vol = _with_counts(make(), 7)
    _, plan = _plan(vol, group)
    counts = vol.counts.astype(np.float32)
    grad = gradient_magnitude(vol.intensity)
    stored = np.maximum(vol.intensity, np.float32(1.0))
    noise = (1.5, 0.05, 0.4, 2.0)
    runs = [(None, False, counts), (stored, False, counts), (None, True, counts),
            (stored, False, None), (None, False, None)]          # None: no counts (N = 1)
    for weighting, rejection in OPTIONS:
        sigma = None if rejection == 'none' else 3.0
        for variance, derive, cnt in runs:
            res = []
            for backend in ('numpy', 'numba'):
                av = orbit_engine.Averaging(weighting, rejection, noise, cnt, grad)
                res.append(orbit_engine.run_cpu(plan, vol.intensity, variance, sigma, 3, True,
                                                derive_variance=derive, backend=backend,
                                                averaging=av))
            a, b = res
            tag = (weighting, rejection, variance is not None, derive, cnt is not None)
            assert np.array_equal(_bits(a.intensity), _bits(b.intensity)), tag
            if a.variance is None:
                assert b.variance is None
            else:
                assert np.array_equal(_bits(a.variance), _bits(b.variance)), tag
            assert np.array_equal(a.int_sums, b.int_sums), tag
            assert np.allclose(a.float_sums, b.float_sums, rtol=1e-12, atol=0), tag


@pytest.mark.parametrize('kind', ['interp', 'perm'])
def test_small_numpy_boxes_give_the_same_bits_with_weights(kind, monkeypatch):
    make, group = CASES[kind]
    vol = _with_counts(make(), 8)
    _, plan = _plan(vol, group)
    var = np.maximum(vol.intensity, np.float32(1.0))
    av = orbit_engine.Averaging('adaptive', 'model', (1.0, 0.05, 0.3, 1.0),
                                vol.counts.astype(np.float32), gradient_magnitude(vol.intensity))
    a = orbit_engine.run_cpu(plan, vol.intensity, var, 3.0, 3, True, backend='numpy',
                             averaging=av)
    monkeypatch.setattr(orbit_engine, '_SLAB_VOXELS', 40)
    for chunk in (7, 100):
        b = orbit_engine.run_cpu(plan, vol.intensity, var, 3.0, 3, True, backend='numpy',
                                 chunk_voxels=chunk, averaging=av)
        assert np.array_equal(_bits(a.intensity), _bits(b.intensity))
        assert np.array_equal(_bits(a.variance), _bits(b.variance))


# ──────────────────────────────────────────────────────────────────
# noise-model fit and scores
# ──────────────────────────────────────────────────────────────────

TRUE = {'gain': 2.0, 'rel': 0.05, 'kappa': 0.3, 'floor': 3.0}


def _synthetic(contamination, seed):
    """4/mmm-invariant truth with peaks, counts, the noise model TRUE and contamination."""
    rng = np.random.default_rng(seed)
    H = (np.arange(61) - 30) * 0.1
    L = (np.arange(31) - 15) * 0.2
    meta = {'ub': np.diag([0.125, 0.125, 1 / 12]), 'wavelength': 0.5,
            'grid_kind': 'hkl_regular'}
    vol = VolumeData(np.zeros((61, 61, 31), np.float32), H.copy(), H.copy(), L, 'HK', meta)
    h, k, l = np.meshgrid(H, H, L, indexing='ij')
    q2 = h * h + k * k + l * l
    mu = 30.0 + 300.0 * np.exp(-q2 / 4.0) + 800.0 * np.exp(-(np.sqrt(q2) - 2.0) ** 2 / 0.5)
    for f, amp, w in ((0.0, 3000.0, 0.07), (0.5, 300.0, 0.06)):
        d2 = sum((a - f - np.rint(a - f)) ** 2 for a in (h, k, l))
        mu += amp * np.exp(-q2 / 8.0) * np.exp(-d2 / (2 * w * w))
    n = np.floor(np.exp(rng.uniform(0.0, np.log(300.0), mu.shape)))
    g = gradient_magnitude(mu).astype(np.float64)
    v = (TRUE['gain'] * np.maximum(mu, 1.0) / n + (TRUE['rel'] * mu) ** 2
         + (TRUE['kappa'] * g) ** 2 + TRUE['floor'] ** 2)
    x = mu + np.sqrt(v) * rng.standard_normal(mu.shape)
    bad = rng.random(mu.shape) < 0.05
    cont = (rng.random(mu.shape) < contamination) & ~bad
    x = np.where(cont, x + np.exp(rng.uniform(np.log(8), np.log(100), mu.shape)) * np.sqrt(v), x)
    x[bad] = np.nan
    vol.intensity = x.astype(np.float32)
    vol.counts = np.where(bad, 0, n).astype(np.uint16)
    return vol


@pytest.mark.parametrize('contamination', [0.0, 0.1])
def test_noise_model_fit_recovers_known_parameters(contamination):
    vol = _synthetic(contamination, seed=3)
    model = fit_noise_model(vol, '4/mmm', use_gpu=False)
    for name, value in TRUE.items():
        assert getattr(model, name) == pytest.approx(value, rel=0.15), name
    assert model.info['n_members'] > 100_000 and model.info['has_counts']


def test_fit_with_only_rel_adds_a_small_floor():
    """Relative scatter only and orbits with med = 0: gain = floor = 0 in the fit."""
    rng = np.random.default_rng(3)
    nv = 20000
    med = np.exp(rng.uniform(np.log(1e3), np.log(1e6), nv))
    med[rng.random(nv) < 0.2] = 0.0
    x = med[None, :] * (1.0 + 0.05 * rng.standard_normal((8, nv)))
    model = fit_members(x.astype(np.float32), None, np.zeros(nv), seed=0)
    assert model.info['floor_added'] and model.gain == 0.0
    assert model.rel == pytest.approx(0.05, rel=0.3)
    v0 = orbit_engine.model_variance(np.zeros(1, np.float32), None, None, model.params)
    assert 0 < v0[0] < 1e-4 * (model.rel * 1e4) ** 2


def test_fit_without_orbits_raises_a_clear_error():
    vol = _regular((4.0, 5.0, 6.0, 90, 90, 90), (7, 7, 7))
    vol.intensity[:] = np.nan
    msg = 'too few orbits with 3 or more members from different sources'
    with pytest.raises(ValueError, match=msg):
        fit_noise_model(vol, 'mmm', use_gpu=False)
    with pytest.raises(ValueError, match=msg):
        symmetrize_volume(vol, 'mmm', sigma=3.0, weighting='adaptive', rejection='model',
                          use_gpu=False)


def test_row_median_does_not_change_an_f_contiguous_input():
    a = np.asfortranarray(np.array([[3, 1], [1, 5], [2, np.nan]], np.float32))
    before = a.copy()
    med = orbit_engine._row_median(a, np.array([3, 2]))
    assert np.array_equal(med, np.array([2, 3], np.float32))
    assert np.array_equal(a, before, equal_nan=True)


def test_scores_use_every_finite_member_of_a_volume_with_nan():
    """Before the fix, the in-place sort moved members away from their finite mask."""
    vol = _with_counts(_regular((4.0, 5.0, 6.0, 90, 90, 90), (15, 15, 15)), 2)
    rng = np.random.default_rng(2)
    vol.intensity = rng.poisson(50, (15, 15, 15)).astype(np.float32)
    vol.intensity[rng.random(vol.intensity.shape) < 0.2] = np.nan
    cands = [{'name': 'equal', 'weighting': 'equal', 'rejection': 'none'},
             {'name': 'counts', 'weighting': 'counts', 'rejection': 'none'}]
    scores = score_candidates(vol, 'mmm', cands, noise_model=NoiseModel(gain=1.0),
                              n_members=20000, seed=1, use_gpu=False)
    smp = sample_orbits(orbit_plan(vol, 'mmm'), np.ascontiguousarray(vol.intensity),
                        vol.counts.astype(np.float32), n_members=20000, seed=1)
    n_fin = int(np.isfinite(smp.vals).sum())
    assert n_fin < smp.vals.size                 # the sample has missing members
    assert scores['n_members_sampled'] == scores['n_members_scored'] == n_fin
    for c in scores['candidates'].values():
        assert c['total']['n_members'] == n_fin and c['n_no_estimate'] == 0


def test_scores_rank_a_robust_estimator_first_on_common_members():
    vol = _synthetic(0.1, seed=4)
    cands = [{'name': 'plain', 'weighting': 'equal', 'rejection': 'none'},
             {'name': 'robust', 'weighting': 'adaptive', 'rejection': 'model', 'sigma': 3.0}]
    model, grad = fit_noise_model(vol, '4/mmm', use_gpu=False, return_gradient=True)
    assert grad.shape == vol.intensity.shape and grad.dtype == np.float32
    scores = score_candidates(vol, '4/mmm', cands, noise_model=model, grad=grad,
                              n_members=200_000, use_gpu=False)
    again = score_candidates(vol, '4/mmm', cands, noise_model=model, n_members=200_000,
                             use_gpu=False)
    assert scores['candidates'] == again['candidates']       # grad = the calculated G
    plain = scores['candidates']['plain']['total']
    robust = scores['candidates']['robust']['total']
    # A contaminated member has a large z with any estimator. Thus the
    # difference is small, but the robust estimator must rank first.
    assert robust['trimmed_mean'] < plain['trimmed_mean']
    assert robust['mean'] < plain['mean']
    # both candidates are scored on the same members
    assert plain['n_members'] == robust['n_members'] == scores['n_members_scored'] > 50_000
    assert scores['candidates']['plain']['n_no_estimate'] == 0
    assert len(scores['candidates']['robust']['shells']['trimmed_mean']) == 10


def test_automatic_candidates_reject_outliers():
    for has_counts in (True, False):
        cands = automatic_candidates(has_counts)
        names = [c['name'] for c in cands]
        assert names[0] == BIAS_REFERENCE == 'equal, mad 3'
        assert ('counts, mad 3' in names) == has_counts and 'adaptive, model 4' in names
        assert all(c['rejection'] != 'none' for c in cands)


def test_suggest_uses_the_mad_reference_and_checks_both_sides():
    vol = _synthetic(0.1, seed=4)
    model, grad = fit_noise_model(vol, '4/mmm', use_gpu=False, return_gradient=True)
    scores = score_candidates(vol, '4/mmm', noise_model=model, grad=grad, n_members=200_000,
                              use_gpu=False)
    names = ['equal, mad 3', 'counts, mad 3', 'adaptive, model 4']
    assert list(scores['candidates']) == names and scores['has_counts']
    outs = {}
    for c in automatic_candidates(True):
        opts = {k: v for k, v in c.items() if k != 'name'}
        outs[c['name']] = symmetrize_volume(vol, '4/mmm', noise_model=model, grad=grad,
                                            use_gpu=False, **opts)
    bias = bias_checks(outs, outs[BIAS_REFERENCE], extra_positions=[(0.5, 0.5, 0.5)])
    ref = bias[BIAS_REFERENCE]
    assert ref['node_sum_ratio'] == 1.0 and ref['diffuse_sum_ratio'] == 1.0
    assert ref['n_lost'] == 0 and '(0.5, 0.5, 0.5)' in ref['extra_sum_ratio']
    choice, reason = suggest(scores, bias)
    assert choice in names and reason.startswith(f"Use '{choice}'")
    b = bias[choice]
    for val in [b['node_sum_ratio'], b['diffuse_sum_ratio'], *b['extra_sum_ratio'].values()]:
        assert abs(val - 1) <= 0.02
    assert b['n_lost'] == 0 and scores['candidates'][choice]['n_no_estimate'] == 0


def _fake(scores_by_name, has_counts=True, n_no=None):
    return {'has_counts': has_counts,
            'candidates': {k: {'total': {'trimmed_mean': s},
                               'n_no_estimate': (n_no or {}).get(k, 0)}
                           for k, s in scores_by_name.items()}}


def _ok(**kw):
    b = {'node_sum_ratio': 1.0, 'extra_sum_ratio': {'(0.5, 0.5, 0.5)': 1.0},
         'diffuse_sum_ratio': 1.0, 'n_lost': 0}
    b.update(kw)
    return b


def test_suggest_rules_and_preselection():
    scores = _fake({'equal, mad 3': 1.2, 'counts, mad 3': 1.1, 'adaptive, model 4': 1.0})
    good = {n: _ok() for n in scores['candidates']}
    assert suggest(scores, good)[0] == 'adaptive, model 4'
    # a ratio above 1 + tol fails too (both sides)
    for kw, text in (({'node_sum_ratio': 1.05}, 'node sum ratio'),
                     ({'node_sum_ratio': 0.95}, 'node sum ratio'),
                     ({'extra_sum_ratio': {'(0.5, 0.5, 0.5)': 1.03}}, 'sum ratio at'),
                     ({'diffuse_sum_ratio': 0.97}, 'diffuse sum ratio'),
                     ({'n_lost': 3}, '3 voxels NaN')):
        bias = dict(good, **{'adaptive, model 4': _ok(**kw)})
        name, reason = suggest(scores, bias)
        assert name == 'counts, mad 3' and text in reason, kw
    # no estimate for some members
    name, reason = suggest(_fake(scores_by_name={'counts, mad 3': 1.1, 'adaptive, model 4': 1.0},
                                 n_no={'adaptive, model 4': 5}), good)
    assert name == 'counts, mad 3' and 'no estimate for 5' in reason
    # nothing passes: the preselection
    bad = {n: _ok(n_lost=1) for n in scores['candidates']}
    name, reason = suggest(scores, bad)
    assert name == 'counts, mad 3' and 'No candidate passes' in reason
    name, reason = suggest(_fake({'equal, mad 3': 1.2, 'adaptive, model 4': 1.0}, False), bad)
    assert name == 'equal, mad 3' and 'no pixel counts' in reason
    assert preselect(True, 'The fit failed.')[0] == 'counts, mad 3'
    assert preselect(False)[0] == 'equal, mad 3'


def test_bias_checks_count_the_lost_voxels():
    vol = _regular((4.0, 5.0, 6.0, 90, 90, 90), (15, 15, 15))
    vol.intensity = np.random.default_rng(6).poisson(50, (15, 15, 15)).astype(np.float32)
    ref = symmetrize_volume(vol, 'mmm', use_gpu=False)
    half = ref.intensity.copy()
    half[1::2] = np.nan                          # the layers with integer h
    b = bias_checks({'ref': ref, 'half': half}, ref, extra_positions=[(0.5, 0.5, 0.5)])
    # the sums do not see the lost voxels: the ratio stays 1
    assert b['ref']['n_lost'] == 0 and b['half']['diffuse_sum_ratio'] == 1.0
    lost = b['half']['lost_voxels']
    assert set(lost) == {'node', '(0.5, 0.5, 0.5)', 'diffuse'}
    assert b['half']['n_lost'] == sum(lost.values()) == int(np.isnan(half).sum())
    assert lost['node'] > 0 and lost['diffuse'] > 0


def test_symmetrize_with_a_given_gradient_gives_the_same_bits():
    vol = _synthetic(0.0, seed=6)
    model, grad = fit_noise_model(vol, '4/mmm', use_gpu=False, return_gradient=True)
    kw = dict(sigma=4.0, weighting='adaptive', rejection='model', noise_model=model,
              use_gpu=False)
    a = symmetrize_volume(vol, '4/mmm', **kw)
    b = symmetrize_volume(vol, '4/mmm', grad=grad, **kw)
    assert _same(a.intensity, b.intensity)
    with pytest.raises(ValueError):
        symmetrize_volume(vol, '4/mmm', grad=grad[:-1], **kw)


# ──────────────────────────────────────────────────────────────────
# HDF5
# ──────────────────────────────────────────────────────────────────

def test_hdf5_round_trip_keeps_the_noise_model():
    vol = _synthetic(0.0, seed=5)
    out = symmetrize_volume(vol, '4/mmm', sigma=3.0, weighting='adaptive', rejection='model',
                            use_gpu=False)
    nm = out.metadata['noise_model']
    assert nm['fitted'] and nm['info']['n_members'] > 0
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'v.h5')
        save_volume_h5(path, out)
        back = load_volume_h5(path)
    assert back.metadata['weighting'] == 'adaptive' and back.metadata['rejection'] == 'model'
    assert back.metadata['noise_model'] == nm
    a, b = NoiseModel.from_dict(nm), NoiseModel.from_dict(back.metadata['noise_model'])
    assert a.params == b.params and a.info == b.info
    assert _same(back.intensity, out.intensity)


def test_noise_model_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        NoiseModel(gain=0.0)
    with pytest.raises(ValueError):
        NoiseModel(gain=1.0, rel=-0.1)
    with pytest.raises(ValueError):
        NoiseModel(gain=float('nan'))
    assert NoiseModel(gain=0.0, floor=1.0).params == (0.0, 0.0, 0.0, 1.0)
