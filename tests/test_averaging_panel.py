"""Tests of the averaging workflow, the Averaging panel, its use in both GUIs and the CLI flags.

All tests run offscreen on the CPU with small synthetic volumes. The panel
runs its jobs synchronously (`panel.synchronous = True`).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from rspace3d import averaging as av
from rspace3d import orbit_engine
from rspace3d.averaging import NoiseModel
from rspace3d.volume_builder import (
    VolumeData, load_volume_h5, save_volume_h5, symmetrize_volume,
)
from test_orbit_engine import _regular, _same


def _small(contamination=0.05, seed=0, counts=True):
    """mmm-invariant volume (31 x 31 x 21) with peaks, pixel counts and noise."""
    rng = np.random.default_rng(seed)
    H = (np.arange(31) - 15) * 0.2
    L = (np.arange(21) - 10) * 0.3
    meta = {'ub': np.diag([0.125, 0.11, 1 / 12]), 'wavelength': 0.5,
            'grid_kind': 'hkl_regular'}
    vol = VolumeData(np.zeros((31, 31, 21), np.float32), H.copy(), H.copy(), L, 'HK', meta)
    h, k, l = np.meshgrid(H, H, L, indexing='ij')
    q2 = h * h + k * k + l * l
    mu = 30.0 + 300.0 * np.exp(-q2 / 4.0)
    d2 = sum((a - np.rint(a)) ** 2 for a in (h, k, l))
    mu += 3000.0 * np.exp(-q2 / 8.0) * np.exp(-d2 / (2 * 0.08 ** 2))
    n = np.floor(np.exp(rng.uniform(0.0, np.log(200.0), mu.shape)))
    v = 2.0 * mu / n + (0.03 * mu) ** 2 + 4.0
    x = mu + np.sqrt(v) * rng.standard_normal(mu.shape)
    bad = rng.random(mu.shape) < 0.05
    cont = (rng.random(mu.shape) < contamination) & ~bad
    x = np.where(cont, x + 30 * np.sqrt(v), x)
    x[bad] = np.nan
    vol.intensity = x.astype(np.float32)
    vol.counts = np.where(bad, 0, n).astype(np.uint16) if counts else None
    return vol


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    """The panel, the GUIs and the workflow use the CPU engines in these tests."""
    from rspace3d import reconstruct_gui, volume_builder, volume_builder_gui
    for mod in (volume_builder, volume_builder_gui, reconstruct_gui):
        monkeypatch.setattr(mod, 'HAS_GPU', False)


@pytest.fixture(scope='module')
def qapp():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def _panel(qapp, vol, laue='mmm'):
    from rspace3d.averaging_panel import AveragingPanel
    lines = []
    panel = AveragingPanel(lambda: (laue, lambda log, progress: vol), log=lines.append)
    panel.synchronous = True
    panel.crop_chk.setChecked(False)
    return panel, lines


# ──────────────────────────────────────────────────────────────────
# workflow functions
# ──────────────────────────────────────────────────────────────────

def test_bias_checks_numba_equals_numpy(monkeypatch):
    if not orbit_engine.numba_available():
        pytest.skip('numba is not installed')
    vol = _small(seed=1)
    ref = symmetrize_volume(vol, 'mmm', use_gpu=False)
    a = symmetrize_volume(vol, 'mmm', sigma=None, use_gpu=False)
    b = a.intensity.copy()
    b[::3, 1::4] = np.nan
    res = {'a': a, 'b': b}
    kw = dict(extra_positions=[(0.5, 0.5, 0.5), (0, 0, 0.5)], radius=0.15)
    fast = av.bias_checks(res, ref, **kw)
    monkeypatch.setattr(orbit_engine, 'numba_available', lambda: False)
    slow = av.bias_checks(res, ref, **kw)
    assert fast.keys() == slow.keys()
    for name in fast:
        assert fast[name]['lost_voxels'] == slow[name]['lost_voxels']
        assert fast[name]['n_lost'] == slow[name]['n_lost'] > (0 if name == 'b' else -1)
        for key in ('sums', 'reference_sums'):
            for reg, val in slow[name][key].items():
                assert fast[name][key][reg] == pytest.approx(val, rel=1e-12, abs=0)
    nb = av.node_sums_by_shell(a, ref, 'mmm')
    monkeypatch.setattr(orbit_engine, 'numba_available', lambda: False)
    npy = av.node_sums_by_shell(a, ref, 'mmm')
    assert nb['result'] == pytest.approx(npy['result'], rel=1e-12)
    assert nb['reference'] == pytest.approx(npy['reference'], rel=1e-12)
    total = av.bias_checks({'a': a}, ref)['a']
    assert sum(nb['result']) == pytest.approx(total['sums']['node'], rel=1e-12)


def test_centred_crop_is_symmetric_about_hkl_zero():
    vol = _small()
    box = av.centred_crop(vol, 11)
    assert box.intensity.shape == (11, 11, 11)
    for ax in (box.H, box.K, box.L):
        assert ax[0] == -ax[-1] and ax[len(ax) // 2] == 0
    assert box.counts.shape == box.intensity.shape
    assert av.centred_crop(vol, 999).intensity.shape == vol.intensity.shape


def test_centred_crop_of_an_axis_without_hkl_zero():
    """An axis without hkl = 0 keeps min(n, length) voxels around its middle."""
    vol = _small()
    vol.H = vol.H - vol.H[0] + 0.2               # 0.2 ... 6.2
    box = av.centred_crop(vol, 15)
    assert box.intensity.shape == (15, 15, 15)
    assert box.H[0] == pytest.approx(vol.H[8]) and box.H[-1] == pytest.approx(vol.H[22])
    assert box.K[7] == 0 and box.L[7] == 0
    assert av.centred_crop(vol, 999).intensity.shape == vol.intensity.shape
    vol.H = vol.H - vol.H[0]                     # 0 ... 6: hkl = 0 at the end of the axis
    box = av.centred_crop(vol, 15)
    assert box.intensity.shape == (15, 15, 15) and box.H[0] == 0


def test_slice_geometry_uses_the_reciprocal_metric():
    vol = _small()
    for plane, diag in ((0, (0.25, 0.22)), (1, (0.25, 1 / 6)), (2, (0.22, 1 / 6))):
        geo = av.slice_geometry(vol, plane, 0.0)
        assert geo['cartesian']
        assert np.allclose(geo['M'], np.diag(diag))
    geo = av.slice_geometry(vol, 0, 0.0)
    assert geo['m0'] == pytest.approx([vol.H[0], vol.K[0]])
    assert geo['mx'] == pytest.approx([0.2, 0.0]) and geo['my'] == pytest.approx([0.0, 0.2])
    vol.metadata.pop('ub')
    geo = av.slice_geometry(vol, 1, 0.0)
    assert not geo['cartesian'] and np.allclose(geo['M'], np.eye(2))


def test_slice_geometry_of_a_sheared_raster():
    """The native plane of a sheared unwarp raster uses the affine grid of the volume."""
    from rspace3d.volume_builder import extract_volume_slice, volume_affine
    from rspace3d.averaging_panel import _slice_mesh
    from test_orbit_engine import _sheared
    vol = _sheared(shear=0.3)
    origin, A = volume_affine(vol)
    assert A[0, 1] != 0
    geo = av.slice_geometry(vol, 0, 0.0)
    il = int(np.argmin(np.abs(vol.L)))
    assert geo['mx'] == pytest.approx(A[[0, 1], 0]) and geo['my'] == pytest.approx(A[[0, 1], 1])
    assert geo['m0'] == pytest.approx((origin + A[:, 2] * il)[[0, 1]])
    assert geo['cartesian']                      # M_inv of the raster (no UB)
    sl = extract_volume_slice(vol, 0, 0.0)[0]
    X, Y, m1, m2 = _slice_mesh(geo, sl.shape)
    assert X.shape == (sl.shape[0] + 1, sl.shape[1] + 1)
    # the pixel centre (ix, iy) has the Miller indices of the volume grid
    ix, iy = 3, 5
    c1 = 0.25 * (m1[iy, ix] + m1[iy + 1, ix] + m1[iy, ix + 1] + m1[iy + 1, ix + 1])
    assert c1 == pytest.approx((origin + A @ np.array([ix, iy, il]))[0])
    # in Cartesian coordinates the raster is rectangular with the step s
    s = vol.metadata['s']
    assert np.allclose(X, X[0][None, :]) and np.allclose(Y, Y[:, :1])
    assert np.allclose(np.diff(X[0]), s) and np.allclose(np.diff(Y[:, 0]), s)


def test_known_candidate_and_kwargs():
    assert av.known_candidate('adaptive, model 4') == av.ADAPTIVE_MODEL
    assert av.known_candidate('equal, none')['rejection'] == 'none'
    assert av.known_candidate('equal, mad') is None and av.known_candidate('foo') is None
    m = NoiseModel(gain=1.0)
    g = np.zeros((2, 2, 2), np.float32)
    assert av.candidate_kwargs(av.EQUAL_MAD, m, g) == {'weighting': 'equal', 'rejection': 'mad',
                                                       'sigma': 3.0}
    kw = av.candidate_kwargs(av.ADAPTIVE_MODEL, m, g)
    assert kw['noise_model'] is m and kw['grad'] is g and kw['sigma'] == 4.0
    assert av.candidate_kwargs(av.make_candidate('counts', 'none'))['sigma'] is None


def test_no_rejection_is_never_suggested(monkeypatch):
    vol = _small(seed=2)
    measured = av.measure_volume(vol, 'mmm', use_gpu=False)
    real = av.score_candidates

    def best_plain(*a, **k):
        out = real(*a, **k)
        out['candidates']['equal, none']['total']['trimmed_mean'] = 0.0
        return out
    monkeypatch.setattr(av, 'score_candidates', best_plain)
    cands = av.automatic_candidates(True) + [av.make_candidate('equal', 'none')]
    c = av.compare_candidates(vol, 'mmm', measured, cands, use_gpu=False)
    assert c['suggestion'] != 'equal, none'
    assert [r['name'] for r in c['rows']][-1] == 'equal, none'      # shown in the table
    assert "never uses the candidates without rejection" in c['reason']


# ──────────────────────────────────────────────────────────────────
# the panel
# ──────────────────────────────────────────────────────────────────

def test_panel_measures_compares_and_suggests(qapp):
    vol = _small(seed=3)
    panel, lines = _panel(qapp, vol)
    assert panel.symmetrize_kwargs(vol, 'mmm') == {}
    assert not panel.compare_btn.isEnabled()
    panel._measure()
    st = panel._state
    m = st['measured']
    assert m['model'] is not None and m['grad'].shape == vol.intensity.shape
    assert 'gain' in panel.model_label.text() and not st['cropped']
    assert panel.shell_table.item(0, 0) is not None
    assert all(panel.shell_table.item(i, 6) is not None for i in range(10))
    assert len(m['node_ratio']) == 10 and m['count_spread']['fraction'][5] is not None
    checked = [c['name'] for c in panel._candidates()]
    assert checked == [c['name'] for c in av.automatic_candidates(True)]
    assert panel.cand_list.count() == len(checked) + len(av.more_candidates(True))
    assert panel.compare_btn.isEnabled()
    # one hand-picked option and an extra position
    panel.add_weight.setCurrentText('equal')
    panel.add_rej.setCurrentText('none')
    panel._add_custom()
    panel.extra_edit.setText('0.5 0.5 0.5')
    panel._compare()
    c = panel._compared
    names = [r['name'] for r in c['rows']]
    assert names == checked + ['equal, none']
    assert panel.result_table.rowCount() == len(names)
    assert c['suggestion'] in checked and c['suggestion'] != 'equal, none'
    assert panel.apply_combo.currentText() == c['suggestion']
    assert all(s.ndim == 2 for s in c['slices'].values())
    assert '(0.5, 0.5, 0.5)' in c['rows'][0]['extra']
    assert not any(isinstance(v, VolumeData) for v in c.values())      # no full volumes
    assert av.LIMITS_TEXT == panel.limits_label.text()
    assert any('Suggestion:' in ln for ln in lines)
    # Apply 'adaptive, model 4'
    panel.apply_combo.setCurrentText('adaptive, model 4')
    panel._apply()
    kw = panel.symmetrize_kwargs(vol, 'mmm')
    assert kw['weighting'] == 'adaptive' and kw['rejection'] == 'model' and kw['sigma'] == 4.0
    assert kw['noise_model'] is m['model'] and kw['grad'] is m['grad']
    rec = panel.choice_metadata()
    assert rec['name'] == 'adaptive, model 4' and rec['suggested'] == c['suggestion']
    assert set(rec['candidates']) == set(names)
    # another volume or group: no options
    assert panel.symmetrize_kwargs(_small(seed=3), 'mmm') == {}
    assert panel.symmetrize_kwargs(vol, '4/mmm') == {}
    panel.clear_applied()
    assert panel.symmetrize_kwargs(vol, 'mmm') == {}
    panel._apply()
    panel.reset('changed')
    assert panel.symmetrize_kwargs(vol, 'mmm') == {} and panel.choice_metadata() is None


def test_panel_compare_error_keeps_the_measure_text(qapp, monkeypatch):
    from rspace3d.averaging_panel import EXTRA_TIP, OLD_ROWS
    vol = _small(seed=15)
    panel, lines = _panel(qapp, vol)
    assert panel.model_label.text() == 'Click Measure to start.'
    panel.reset('The volume settings changed. Click Measure again.')
    assert panel.model_label.text() == 'Click Measure to start.'     # no Measure yet
    panel._measure()
    measure_text = panel.model_label.text()
    panel.extra_edit.setText('0.5 0.5 0.5')
    panel._compare()
    assert panel.result_table.horizontalHeaderItem(3).toolTip().startswith('Sum ratio at (0.5')
    old = panel._compared

    def boom(*a, **k):
        raise RuntimeError('boom')
    monkeypatch.setattr(av, 'compare_candidates', boom)
    panel._compare()
    assert panel.model_label.text() == measure_text
    text = panel.suggest_label.text()
    assert text.startswith('Compare failed: boom') and 'older Compare' in text
    assert 'running' not in text and not panel.is_busy()
    assert panel._compared is old
    assert panel.result_table.item(0, 0).foreground().color() == OLD_ROWS
    monkeypatch.undo()
    # a new Compare without extra positions: normal rows, default tooltip
    panel.extra_edit.setText('')
    panel._compare()
    assert panel.result_table.item(0, 0).foreground().color() != OLD_ROWS
    assert panel.result_table.horizontalHeaderItem(3).toolTip() == EXTRA_TIP
    # a Measure error shows in the Measure label
    monkeypatch.setattr(av, 'measure_volume', boom)
    panel._measure()
    assert panel.model_label.text().startswith('Measure failed: boom')
    assert panel._state is not None and panel._compared is not None


def test_panel_cancel_keeps_the_earlier_state(qapp):
    vol = _small(seed=16)
    panel, lines = _panel(qapp, vol)

    def cancel_at(word):
        def log(msg):
            lines.append(msg)
            if word in msg:
                assert panel.measure_cancel_btn.isEnabled() or panel.compare_cancel_btn.isEnabled()
                panel.cancel()
        panel._log_fn = log
    # Measure cancelled before the first result: no state
    cancel_at('Fit the noise model')
    panel._measure()
    assert panel._state is None and not panel.is_busy()
    assert panel.model_label.text() == 'Cancelled. Click Measure to start.'
    assert panel.measure_btn.isEnabled() and not panel.measure_cancel_btn.isEnabled()
    # a full Measure and Compare
    panel._log_fn = lines.append
    panel._measure()
    panel._compare()
    state, compared = panel._state, panel._compared
    suggest = panel.suggest_label.text()
    # Compare cancelled between two candidates: the earlier comparison stays
    cancel_at("Symmetrise with 'counts")
    panel._compare()
    assert panel._compared is compared and panel._state is state
    assert panel.suggest_label.text() == f'Cancelled. {suggest}'
    assert panel.result_table.rowCount() == len(compared['rows'])
    assert not any("Symmetrise with 'adaptive" in ln for ln in lines[-3:])
    # Measure cancelled after the fit: the earlier measurement stays
    cancel_at('Symmetrise with the default')
    panel._measure()
    assert panel._state is state and panel._compared is compared
    assert panel.model_label.text().startswith('Cancelled. Noise model: gain')


def test_panel_fit_error_is_clear_and_gives_the_preselection(qapp):
    vol = _small(seed=17)
    vol.intensity[np.isfinite(vol.intensity)] = 100.3     # no scatter between members
    vol.counts = None
    panel, _ = _panel(qapp, vol)
    panel._measure()
    m = panel._state['measured']
    assert m['model'] is None and m['fit_error'] == av.FIT_NO_SCATTER
    assert panel.apply_combo.currentText() == 'equal, mad 3'
    with pytest.raises(ValueError, match='no scatter'):
        av.fit_noise_model(vol, 'mmm', use_gpu=False)
    bad = _small(seed=17)
    bad.intensity[:] = 50.5                      # NaN voxels get a value, but their count is 0
    with pytest.raises(ValueError, match='not finite'):
        av.fit_noise_model(bad, 'mmm', use_gpu=False)


def test_spread_column_without_counts_shows_a_dash():
    vol = _small(seed=18, counts=False)
    measured = av.measure_volume(vol, 'mmm', use_gpu=False)
    assert measured['count_spread'] is None
    assert all(row[5] == '-' for row in av.shell_rows(measured))


def test_count_spread_of_known_counts():
    """Equal counts give 0 %. A factor of 20 in some members gives 100 %."""
    vol = _small(seed=19)
    vol.counts = np.where(np.isfinite(vol.intensity), 50, 0).astype(np.uint16)
    assert all(f in (None, 0.0) for f in av.count_spread(vol, 'mmm')['fraction'])
    h = np.asarray(vol.H)[:, None, None]
    vol.counts = np.where(np.isfinite(vol.intensity), np.where(h > 0, 1000, 50), 0)
    fr = av.count_spread(vol, 'mmm')['fraction']
    assert all(f >= 0.99 for f in fr if f is not None)


def test_panel_worker_thread_and_request_errors(qapp):
    from rspace3d.averaging_panel import AveragingPanel
    vol = _small(seed=14)
    busy = []
    panel, lines = _panel(qapp, vol)
    panel.synchronous = False
    panel.busy_changed.connect(busy.append)
    panel._measure()
    assert panel.is_busy() and not panel.measure_btn.isEnabled()
    assert panel._worker.wait(120_000)
    for _ in range(5):
        qapp.processEvents()
    assert busy == [True, False] and panel._state is not None and panel.compare_btn.isEnabled()
    assert any(ln.startswith('Fit the noise model') for ln in lines)

    def not_ready():
        raise ValueError('Reconstruct a volume first.')
    other = AveragingPanel(not_ready)
    other._measure()
    assert other.model_label.text() == 'Reconstruct a volume first.' and other._state is None


def test_panel_offers_the_crop_only_for_the_numpy_engine(qapp, monkeypatch):
    from rspace3d import averaging_panel as ap
    monkeypatch.setattr(ap, 'engine_name', lambda: 'NumPy')
    panel = ap.AveragingPanel(lambda: ('mmm', None))
    assert panel.crop_chk.isChecked() and 'centred crop' in panel.engine_label.text()
    monkeypatch.setattr(ap, 'engine_name', lambda: 'numba')
    panel = ap.AveragingPanel(lambda: ('mmm', None))
    assert not panel.crop_chk.isChecked() and 'full volume' in panel.engine_label.text()


def test_panel_crop_does_not_pass_the_gradient(qapp):
    vol = _small(seed=4)
    panel, _ = _panel(qapp, vol)
    panel.crop_chk.setChecked(True)
    panel.crop_spin.setValue(15)
    panel._measure()
    assert panel._state['cropped'] and panel._state['work'].intensity.shape == (15, 15, 15)
    assert 'centred crop 15 x 15 x 15' in panel.model_label.text()
    panel._compare()
    panel.apply_combo.setCurrentText('adaptive, model 4')
    panel._apply()
    kw = panel.symmetrize_kwargs(vol, 'mmm')
    assert 'grad' not in kw and kw['noise_model'] is panel._state['measured']['model']


def test_panel_failed_fit_gives_the_preselection(qapp):
    vol = _regular((4.0, 5.0, 6.0, 90, 90, 90), (7, 7, 7))
    vol.intensity[:] = np.nan
    panel, _ = _panel(qapp, vol)
    panel._measure()
    m = panel._state['measured']
    assert m['model'] is None and 'too few orbits' in m['fit_error']
    assert 'failed' in panel.model_label.text()
    assert not panel.compare_btn.isEnabled()
    assert panel.apply_combo.count() == 1 and panel.apply_combo.currentText() == 'equal, mad 3'
    panel._apply()
    assert panel.symmetrize_kwargs(vol, 'mmm') == {'weighting': 'equal', 'rejection': 'mad',
                                                   'sigma': 3.0}
    assert 'fit failed' in panel.choice_metadata()['reason']


def test_auto_crop_size_keeps_the_budget():
    from rspace3d.averaging_panel import CROP_BUDGET, auto_crop_size
    n = auto_crop_size((2162, 2068, 61), 16)
    assert 9 <= n and min(n, 2162) * min(n, 2068) * 61 * 16 <= CROP_BUDGET
    assert min(n + 2, 2162) * min(n + 2, 2068) * 61 * 16 > CROP_BUDGET
    assert auto_crop_size((20, 20, 20), 48) == 21


def test_parse_positions():
    from rspace3d.averaging_panel import parse_positions
    assert parse_positions('0.5 0.5 0.5; 0 0 0.5') == [(0.5, 0.5, 0.5), (0.0, 0.0, 0.5)]
    assert parse_positions('  ') == []
    with pytest.raises(ValueError):
        parse_positions('0.5 0.5')


# ──────────────────────────────────────────────────────────────────
# HDF5
# ──────────────────────────────────────────────────────────────────

def test_hdf5_round_trip_of_the_choice(tmp_path):
    vol = _small(seed=5)
    measured = av.measure_volume(vol, 'mmm', use_gpu=False)
    compared = av.compare_candidates(vol, 'mmm', measured, av.automatic_candidates(True),
                                     extra_positions=[(0.5, 0.5, 0.5)], use_gpu=False)
    name = compared['suggestion']
    rec = av.choice_record(name, compared['reason'], measured, compared, 'full volume')
    out = symmetrize_volume(vol, 'mmm', use_gpu=False,
                            **av.candidate_kwargs(compared['specs'][name], measured['model'],
                                                  measured['grad']))
    out.metadata['averaging_choice'] = rec
    path = str(tmp_path / 'v.h5')
    save_volume_h5(path, out)
    back = load_volume_h5(path)
    assert back.metadata['averaging_choice'] == rec
    assert rec['candidates'][name]['n_no_estimate'] == 0 and rec['noise_model']['gain'] > 0
    assert _same(back.intensity, out.intensity)
    # a later symmetrisation does not carry the old record
    again = symmetrize_volume(back, 'mmm', use_gpu=False)
    assert 'averaging_choice' not in again.metadata


# ──────────────────────────────────────────────────────────────────
# the GUIs
# ──────────────────────────────────────────────────────────────────

class _Stub:
    """Records the calls of symmetrize_volume and save_volume_h5."""

    def __init__(self):
        self.sym = []
        self.saved = []

    def symmetrize(self, vol, laue, **kw):
        self.sym.append((vol, laue, kw))
        out = VolumeData(np.ones(vol.intensity.shape, np.float32), vol.H, vol.K, vol.L,
                         vol.plane_type, {'symmetry_diagnostics': None,
                                          'n_outliers_removed': 0, 'symmetry_mapping': 'exact'})
        return out

    def save(self, path, vol, *a, **k):
        self.saved.append((path, vol))
        with open(path, 'wb') as f:
            f.write(b'x')


def _wait(qapp, gui):
    assert gui._worker.wait(120_000)
    for _ in range(5):
        qapp.processEvents()


def _builder(qapp, monkeypatch, tmp_path, vol):
    from rspace3d import volume_builder_gui as vbg
    stub = _Stub()
    monkeypatch.setattr(vbg, 'symmetrize_volume', stub.symmetrize)
    monkeypatch.setattr(vbg, 'save_volume_h5', stub.save)
    gui = vbg.SimpleVolumeGUI()
    gui._folder_path = str(tmp_path)
    gui._dataset_valid = True
    gui.laue_combo.setCurrentIndex(gui.laue_combo.findData('mmm'))
    loads = []

    def fake_load(folder, morph, bin_xy, bin_z, force_reload, log, cb):
        loads.append((bin_xy, bin_z))
        return vol, str(tmp_path / 'r_raw.h5'), 'r'
    monkeypatch.setattr(gui, '_load_binned', fake_load)
    gui.avg_panel.synchronous = True
    gui.avg_panel.crop_chk.setChecked(False)
    return gui, stub, loads


def test_builder_gui_defaults_and_process_all_without_panel(qapp, monkeypatch, tmp_path):
    vol = _small(seed=6)
    gui, stub, loads = _builder(qapp, monkeypatch, tmp_path, vol)
    gui.laue_combo.setCurrentIndex(gui.laue_combo.findData('m-3m'))
    assert (gui.laue_combo.currentData(), gui.sigma_spin.value(), gui.bin_xy_spin.value(),
            gui.bin_z_spin.value(), gui.morph_combo.currentData(),
            gui.force_reload_chk.isChecked()) == ('m-3m', 3.0, 2, 1, 'auto', False)
    assert gui.tabs.tabText(0) == 'Process' and gui.tabs.tabText(1) == 'Averaging'
    gui._process_all()
    _wait(qapp, gui)
    assert loads == [(2, 1)]
    (v, laue, kw), = stub.sym
    assert v is vol and laue == 'm-3m'
    assert set(kw) == {'sigma', 'progress_callback'} and kw['sigma'] == 3.0
    assert 'averaging_choice' not in stub.saved[0][1].metadata


def test_builder_gui_apply_passes_the_options(qapp, monkeypatch, tmp_path):
    vol = _small(seed=7)
    gui, stub, loads = _builder(qapp, monkeypatch, tmp_path, vol)
    gui.bin_xy_spin.setValue(1)
    gui.avg_panel._measure()                     # loads and bins once
    assert loads == [(1, 1)] and gui._current_prepared()['vol'] is vol
    gui.avg_panel._compare()
    gui.avg_panel.apply_combo.setCurrentText('counts, mad 3')
    gui.avg_panel._apply()
    gui._process_all()
    _wait(qapp, gui)
    assert loads == [(1, 1)]                     # Process All used the same volume
    (v, laue, kw), = stub.sym
    assert v is vol and laue == 'mmm'
    assert (kw['weighting'], kw['rejection'], kw['sigma']) == ('counts', 'mad', 3.0)
    assert 'noise_model' not in kw
    assert stub.saved[0][1].metadata['averaging_choice']['name'] == 'counts, mad 3'
    # a changed bin factor drops the volume and the applied option
    gui.bin_z_spin.setValue(2)
    assert gui._current_prepared() is None and gui.avg_panel.applied_name() is None
    gui._process_all()
    _wait(qapp, gui)
    assert loads == [(1, 1), (1, 2)] and set(stub.sym[-1][2]) == {'sigma', 'progress_callback'}


def test_builder_gui_shows_that_the_option_replaces_sigma(qapp, monkeypatch, tmp_path):
    vol = _small(seed=20)
    gui, stub, _ = _builder(qapp, monkeypatch, tmp_path, vol)
    logged = []
    monkeypatch.setattr(gui, '_log', logged.append)
    gui.bin_xy_spin.setValue(1)
    assert gui.avg_panel.model_label.text() == 'Click Measure to start.'
    assert gui.sigma_spin.isEnabled() and gui.sigma_note.text() == ''
    gui.avg_panel._measure()
    gui.avg_panel._compare()
    gui.avg_panel.apply_combo.setCurrentText('counts, mad 3')
    gui.avg_panel._apply()
    assert not gui.sigma_spin.isEnabled() and "'counts, mad 3'" in gui.sigma_note.text()
    gui._process_all()
    _wait(qapp, gui)
    head = [ln for ln in logged if ln.startswith('Processing:')]
    assert head and "averaging option 'counts, mad 3'" in head[-1] and 'sigma=' not in head[-1]
    gui.avg_panel.clear_applied()
    assert gui.sigma_spin.isEnabled() and gui.sigma_note.text() == ''
    gui.avg_panel._apply()
    gui.laue_combo.setCurrentIndex(gui.laue_combo.findData('4/mmm'))   # reset also undoes it
    assert gui.sigma_spin.isEnabled() and gui.sigma_note.text() == ''
    gui._process_all()
    _wait(qapp, gui)
    assert 'sigma=3.0' in [ln for ln in logged if ln.startswith('Processing:')][-1]


def test_builder_gui_apply_with_a_real_symmetrisation(qapp, monkeypatch, tmp_path):
    from rspace3d import volume_builder_gui as vbg
    vol = _small(seed=8)
    gui, stub, _ = _builder(qapp, monkeypatch, tmp_path, vol)
    monkeypatch.setattr(vbg, 'symmetrize_volume', symmetrize_volume)
    monkeypatch.setattr(vbg, 'save_volume_h5', save_volume_h5)
    gui.bin_xy_spin.setValue(1)
    gui.avg_panel._measure()
    gui.avg_panel._compare()
    gui.avg_panel.apply_combo.setCurrentText('adaptive, model 4')
    gui.avg_panel._apply()
    gui._process_all()
    _wait(qapp, gui)
    back = load_volume_h5(str(tmp_path / 'r_sym_mmm.h5'))
    m = gui.avg_panel._state['measured']
    ref = symmetrize_volume(vol, 'mmm', sigma=4.0, weighting='adaptive', rejection='model',
                            noise_model=m['model'], use_gpu=False)
    assert _same(back.intensity, ref.intensity)
    assert back.metadata['weighting'] == 'adaptive' and back.metadata['rejection'] == 'model'
    assert back.metadata['averaging_choice']['name'] == 'adaptive, model 4'


def _recon(qapp, monkeypatch, tmp_path, vol):
    from rspace3d import reconstruct_gui as rg
    stub = _Stub()
    monkeypatch.setattr(rg, 'symmetrize_volume', stub.symmetrize)
    monkeypatch.setattr(rg, 'save_volume_h5', stub.save)
    monkeypatch.setattr(rg.QFileDialog, 'getSaveFileName',
                        staticmethod(lambda *a, **k: (str(tmp_path / 's.h5'), '')))
    gui = rg.ReconstructGUI()
    gui._vd, gui._info, gui._folder = vol, {'name': 'x'}, str(tmp_path)
    gui._set_post_enabled(True)
    gui.laue_combo.setCurrentIndex(gui.laue_combo.findData('mmm'))
    gui.avg_panel.synchronous = True
    gui.avg_panel.crop_chk.setChecked(False)
    return gui, stub


def test_reconstruct_gui_defaults_and_symmetrise_without_panel(qapp, monkeypatch, tmp_path):
    vol = _small(seed=9)
    gui, stub = _recon(qapp, monkeypatch, tmp_path, vol)
    assert (gui.sigma.value(), gui.bin_xy.value(), gui.bin_z.value()) == (3.0, 1, 1)
    assert gui.tabs.tabText(1) == 'Averaging' and gui.frame_tools_group is gui.avg_panel.frame_group
    assert gui.avg_panel.selected_volume() is None and gui.source_note.isHidden()
    gui._symmetrize_save()
    _wait(qapp, gui)
    (v, laue, kw), = stub.sym
    assert v is vol and set(kw) == {'sigma', 'progress_callback'} and kw['sigma'] == 3.0


def test_reconstruct_gui_apply_uses_the_binned_volume(qapp, monkeypatch, tmp_path):
    vol = _small(seed=10)
    gui, stub = _recon(qapp, monkeypatch, tmp_path, vol)
    gui.bin_z.setValue(3)
    gui.avg_panel._measure()
    binned = gui._current_binned()
    assert binned is not None and binned.intensity.shape == (31, 31, 7)
    assert gui.avg_panel._state['volume'] is binned
    gui.avg_panel._compare()
    gui.avg_panel.apply_combo.setCurrentText('adaptive, model 4')
    gui.avg_panel._apply()
    gui._symmetrize_save()
    _wait(qapp, gui)
    (v, laue, kw), = stub.sym
    assert v is binned and kw['weighting'] == 'adaptive' and kw['sigma'] == 4.0
    assert kw['noise_model'] is gui.avg_panel._state['measured']['model']
    assert stub.saved[0][1].metadata['averaging_choice']['name'] == 'adaptive, model 4'
    assert not gui.sigma.isEnabled() and "'adaptive, model 4'" in gui.sigma_note.text()
    gui.laue_combo.setCurrentIndex(gui.laue_combo.findData('4/mmm'))
    assert gui.avg_panel.applied_name() is None
    assert gui.sigma.isEnabled() and gui.sigma_note.text() == ''


def test_gui_log_height_at_the_default_size(qapp):
    from rspace3d.reconstruct_gui import ReconstructGUI
    from rspace3d.volume_builder_gui import SimpleVolumeGUI
    from PyQt6.QtGui import QFont
    old = qapp.font()
    qapp.setFont(QFont('Segoe UI', 10))          # a large desktop font: tall tabs
    try:
        for cls in (SimpleVolumeGUI, ReconstructGUI):
            gui = cls()
            gui.show()
            for _ in range(5):
                qapp.processEvents()
            assert 200 <= gui.split.sizes()[1] <= 250, (cls.__name__, gui.split.sizes())
            gui.close()
    finally:
        qapp.setFont(old)


# ──────────────────────────────────────────────────────────────────
# the CLI
# ──────────────────────────────────────────────────────────────────

def _cli(monkeypatch, tmp_path, vol, *flags):
    from rspace3d import volume_process as vp
    from test_rawrecon_and_gui import _img_header
    for i, l in enumerate([-0.5, 0.0, 0.5]):
        (tmp_path / f'run_{i + 1}.img').write_bytes(_img_header(8, 8, 'HK', l))
    calls = []
    vol.metadata.update({'morph_size': 3, 'n_unmeasured_per_frame_mean': 0.0,
                         'n_unmeasured_per_frame_min': 0, 'n_unmeasured_per_frame_max': 0,
                         'n_unmeasured_per_frame_pct': 0.0})
    monkeypatch.setattr(vp, 'load_unwarp_folder', lambda *a, **k: vol)
    real = vp.symmetrize_volume

    def spy(v, laue, **kw):
        calls.append(kw)
        return real(v, laue, **kw)
    monkeypatch.setattr(vp, 'symmetrize_volume', spy)
    monkeypatch.setattr(sys, 'argv', ['rspace3d-process', str(tmp_path), '--laue', 'mmm',
                                      '--bin', '1', '--binl', '1', '--no-gpu',
                                      '--force-reload', *flags])
    vp._main()
    return calls, tmp_path / 'run_sym_mmm.h5'


def test_cli_without_new_flags_is_unchanged(monkeypatch, tmp_path, capsys):
    vol = _small(seed=11)
    calls, path = _cli(monkeypatch, tmp_path, vol)
    assert calls == [{'use_gpu': False, 'sigma': 3.0}]
    back = load_volume_h5(str(path))
    assert _same(back.intensity, symmetrize_volume(vol, 'mmm', use_gpu=False).intensity)
    assert not {'weighting', 'rejection', 'averaging_choice'} & set(back.metadata)
    assert 'Average:' not in capsys.readouterr().out


def test_cli_explicit_options(monkeypatch, tmp_path):
    vol = _small(seed=12)
    calls, path = _cli(monkeypatch, tmp_path, vol, '--weighting', 'counts', '--rejection', 'mad',
                       '--sigma', '4')
    assert calls == [{'use_gpu': False, 'sigma': 4.0, 'weighting': 'counts', 'rejection': 'mad'}]
    back = load_volume_h5(str(path))
    assert back.metadata['weighting'] == 'counts' and back.metadata['sigma'] == 4.0


def test_cli_report_stops_and_auto_applies(monkeypatch, tmp_path, capsys):
    vol = _small(seed=13)
    calls, path = _cli(monkeypatch, tmp_path, vol, '--report', '--extra-positions',
                       '0.5', '0.5', '0.5')
    out = capsys.readouterr().out
    assert 'Noise model: gain' in out and 'Suggestion:' in out and '(0.5, 0.5, 0.5)' in out
    assert 'saves no volume' in out and not path.exists()
    assert calls == []                           # no symmetrisation of the CLI step 4
    calls, path = _cli(monkeypatch, tmp_path, vol, '--auto')
    rec = load_volume_h5(str(path)).metadata['averaging_choice']
    assert rec['name'] == rec['suggested'] and rec['compared_on'].startswith('full volume')
    assert calls[-1]['weighting'] == rec['weighting'] and 'progress_callback' not in calls[-1]


def test_cli_rejects_model_without_sigma(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        _cli(monkeypatch, tmp_path, _small(), '--rejection', 'model', '--sigma', '0')


def test_cli_extra_positions_need_report_or_auto(monkeypatch, tmp_path, capsys):
    with pytest.raises(SystemExit):
        _cli(monkeypatch, tmp_path, _small(), '--extra-positions', '0.5', '0.5', '0.5')
    assert '--extra-positions needs --report or --auto' in capsys.readouterr().err


def test_cli_auto_with_own_options_prints_a_note(monkeypatch, tmp_path, capsys):
    calls, path = _cli(monkeypatch, tmp_path, _small(seed=21), '--auto', '--sigma', '4')
    out = capsys.readouterr().out
    assert sum('--auto symmetrises with the suggestion' in ln for ln in out.splitlines()) == 1
    assert path.exists()
    calls, path = _cli(monkeypatch, tmp_path, _small(seed=21), '--auto')
    assert '--auto symmetrises' not in capsys.readouterr().out


def test_cli_report_continue_reuses_the_fit(monkeypatch, tmp_path):
    fits = []
    real = av._fit

    def count(*a, **k):
        fits.append(k.get('grad') is not None)
        return real(*a, **k)
    monkeypatch.setattr(av, '_fit', count)
    calls, path = _cli(monkeypatch, tmp_path, _small(seed=22), '--report', 'continue',
                       '--weighting', 'adaptive', '--rejection', 'model', '--sigma', '4')
    assert fits == [False]                       # one fit: the one of Measure
    kw = calls[-1]
    assert kw['weighting'] == 'adaptive' and kw['noise_model'] is not None
    assert kw['grad'] is not None and path.exists()
