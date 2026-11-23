"""Frame selection by frame number and the 4-fold axis along a or b (rspace3d 2.12.0)."""
from __future__ import annotations

import os

import numpy as np
import pytest

from rspace3d.rawrecon import calibrate, engine
from rspace3d.rawrecon.engine import parse_frame_selection, frame_metadata
from rspace3d.rawrecon.geometry import FlatDetector
from rspace3d.volume_builder import (
    VolumeData, symmetrize_volume, get_symmetry_operations, cell_metric_residual,
    fitting_laue_group, laue_key, split_laue_key, laue_display_name, save_volume_h5,
    load_volume_h5, inexact_mapping_note, _EXPECTED_ORDERS, LAUE_GROUP_NAMES,
)

NEW_KEYS = ('4/m_a', '4/m_b', '4/mmm_a', '4/mmm_b')


# ──────────────────────────────────────────────────────────────────
# Parser
# ──────────────────────────────────────────────────────────────────
SCAN = list(range(1, 1736))


def test_parser_all_frames_and_ranges():
    for spec in (None, '', '  ', 'all', 'ALL'):
        pos, text = parse_frame_selection(spec, SCAN)
        assert pos == list(range(1735)) and text == '1-1735'
    pos, text = parse_frame_selection('1-369, 901-1289', SCAN)
    assert len(pos) == 758 and text == '1-369, 901-1289'
    assert pos[:2] == [0, 1] and pos[368] == 368 and pos[369] == 900 and pos[-1] == 1288
    # spaces, single numbers, overlaps and an unsorted order give one normalised text
    assert parse_frame_selection('901 - 905 1-3,3-4 7', SCAN)[1] == '1-4, 7, 901-905'
    assert parse_frame_selection([5, 3, 4, 4], SCAN) == ([2, 3, 4], '3-5')
    assert parse_frame_selection(np.array([10, 12]), SCAN)[1] == '10, 12'


def test_parser_with_gaps_in_the_frame_numbers():
    nums = [1, 2, 3, 5, 6, 9]
    pos, text = parse_frame_selection('1-6', nums)
    assert pos == [0, 1, 2, 3, 4] and text == '1-6'     # the gap 4 is not an error
    assert parse_frame_selection('2-9', nums)[1] == '2-9'
    with pytest.raises(ValueError, match='Frame 4 is not in the scan'):
        parse_frame_selection('4', nums)
    with pytest.raises(ValueError, match='Frame 4 is not in the scan'):
        parse_frame_selection([4], nums)
    with pytest.raises(ValueError, match='empty'):
        parse_frame_selection('7-8', nums)


@pytest.mark.parametrize('spec, match', [
    ('1-369; 901', "not allowed: ';'"),
    ('1-369 x', "not allowed: 'x'"),
    ('1.5', "not allowed: '.'"),
    ('369-1', 'The range 369-1 is reversed'),
    ('2000', 'Frame 2000 is not in the scan. The scan has the frames 1 to 1735.'),
    ('1-2000', 'Frame 2000 is not in the scan'),
    ('0', 'Frame 0 is not in the scan'),
    (',', 'empty'),
    ('1--5', "The part '1--5' is not a frame number or a range"),
    ('5-', "The part '5-' is not a frame number or a range"),
    ([], 'empty'),
    ([1.5], 'not an integer'),
    (5, 'For the first n frames, use nframes'),
    (5.0, 'For the first n frames, use nframes'),
    (object(), 'list of frame numbers'),
])
def test_parser_errors_are_plain(spec, match):
    with pytest.raises(ValueError, match=match.replace('(', r'\(').replace('.', r'\.')):
        parse_frame_selection(spec, SCAN)


# ──────────────────────────────────────────────────────────────────
# Frame subset of the setup
# ──────────────────────────────────────────────────────────────────
INCR, PHI0 = 0.5, -180.0


def _folder(tmp_path, numbers, monkeypatch):
    (tmp_path / 'run_cracker.par').write_text('', encoding='latin-1')
    for i in numbers:
        (tmp_path / f'run_{i:04d}.cbf').write_bytes(b'')
    det = FlatDetector(distance=80.0, pixel_size=0.172, beam_center=(10.0, 8.0),
                       shape=(16, 20), wavelength=0.7)
    fg = calibrate.FittedGeometry(det, np.eye(3) / 6.0, np.eye(3), (0.0, 1.0, 0.0), 1, PHI0,
                                  INCR, 0.001, 100, 0.01)
    monkeypatch.setattr(calibrate, 'geometry_from_tabbin', lambda *a, **k: fg)
    return str(tmp_path)


def test_setup_keeps_only_the_selected_frames_with_their_angles(tmp_path, monkeypatch):
    folder = _folder(tmp_path, range(1, 11), monkeypatch)
    full = engine.prepare_setup(folder, use_tabbin=True)
    s = engine.prepare_setup(folder, use_tabbin=True, frames='2-4, 8')
    assert s.numbers == [2, 3, 4, 8] and s.n_frames == 4
    assert s.phis == [full.phis[i] for i in (1, 2, 3, 7)]
    assert s.phis == [PHI0 + (n - 1) * INCR for n in (2, 3, 4, 8)]
    assert [os.path.basename(f) for f in s.frames] == [
        'run_0002.cbf', 'run_0003.cbf', 'run_0004.cbf', 'run_0008.cbf']
    assert s.meta['frame_selection'] == '2-4, 8' and s.meta['n_frames_used'] == 4
    assert s.meta['n_total'] == 10 and s.meta['missing_frame_numbers'] == []
    assert s.halves_groups().tolist() == [1, 0, 1, 1]          # by frame number
    assert frame_metadata(s.meta) == {'frame_selection': '2-4, 8', 'n_frames_used': 4,
                                      'n_frames_total': 10}
    # the first n frames work as before
    first = engine.prepare_setup(folder, use_tabbin=True, nframes=3)
    assert first.numbers == [1, 2, 3] and first.phis == full.phis[:3]
    assert first.meta['frame_selection'] == '1-3'
    assert full.meta['frame_selection'] == '1-10' and full.meta['n_frames_used'] == 10
    with pytest.raises(ValueError, match='not both'):
        engine.prepare_setup(folder, use_tabbin=True, nframes=3, frames='1-3')
    with pytest.raises(ValueError, match='Frame 11 is not in the scan'):
        engine.prepare_setup(folder, use_tabbin=True, frames='1-11')
    # an empty text or 'all' is no selection: it is not an error with nframes
    for text in ('', 'all', ' ALL '):
        s = engine.prepare_setup(folder, use_tabbin=True, nframes=3, frames=text)
        assert s.numbers == [1, 2, 3]
    assert engine.prepare_setup(folder, use_tabbin=True, frames='all').n_frames == 10


def test_setup_with_a_gap_in_the_scan(tmp_path, monkeypatch):
    folder = _folder(tmp_path, [1, 2, 3, 5, 6], monkeypatch)
    s = engine.prepare_setup(folder, use_tabbin=True, frames='3-6')
    assert s.numbers == [3, 5, 6] and s.phis == [PHI0 + 2 * INCR, PHI0 + 4 * INCR, PHI0 + 5 * INCR]
    assert s.meta['missing_frame_numbers'] == [4]
    s = engine.prepare_setup(folder, use_tabbin=True, frames='5-6')
    assert s.meta['missing_frame_numbers'] == []        # omitted frames are not gaps


def test_reconstruct_dataset_records_the_selection(tmp_path, monkeypatch):
    folder = _folder(tmp_path, range(1, 11), monkeypatch)
    seen = {}

    def fake(self, **kw):
        seen['numbers'] = list(self.numbers)
        z = np.zeros((2, 2, 2))
        return engine.Volume(z, z[0, 0], z[0, 0], z[0, 0], np.eye(3), z.astype(np.int32), 0.7)
    monkeypatch.setattr(engine.ReconSetup, 'reconstruct', fake)
    lines = []
    vol, info = engine.reconstruct_dataset(folder, use_tabbin=True, frames=[1, 2, 9],
                                           log=lines.append)
    assert seen['numbers'] == [1, 2, 9]
    assert info['n_frames'] == 3 and info['n_total'] == 10
    assert info['frame_selection'] == '1-2, 9'
    assert 'Frames: 3 of 10 (1-2, 9).' in lines
    assert frame_metadata(info) == {'frame_selection': '1-2, 9', 'n_frames_used': 3,
                                    'n_frames_total': 10}


# ──────────────────────────────────────────────────────────────────
# 4-fold axis along a or b
# ──────────────────────────────────────────────────────────────────
def _recip(a, b, c):
    return np.diag([1.0 / a, 1.0 / b, 1.0 / c])


def test_new_keys_order_closure_and_the_4_fold_operation():
    assert len(LAUE_GROUP_NAMES) == 11
    for key, order in zip(NEW_KEYS, (8, 8, 16, 16)):
        ops = get_symmetry_operations(key)
        assert len(ops) == order == _EXPECTED_ORDERS[key]
        flat = {tuple(o.ravel()) for o in ops}
        assert all(tuple((x @ y).ravel()) in flat for x in ops for y in ops)
    hkl = np.array([1, 2, 3])
    images_a = {tuple(o @ hkl) for o in get_symmetry_operations('4/m_a')}
    images_b = {tuple(o @ hkl) for o in get_symmetry_operations('4/m_b')}
    assert (1, -3, 2) in images_a and all(abs(v[0]) == 1 for v in images_a)
    assert (3, 2, -1) in images_b and all(abs(v[1]) == 2 for v in images_b)
    for key in ('4/m', '4/mmm'):                    # the axis c stays the axis c
        assert all(abs((o @ hkl)[2]) == 3 for o in get_symmetry_operations(key))


def test_key_helpers_and_file_name_suffix():
    assert laue_key('4/mmm', 'a') == '4/mmm_a' and laue_key('4/m', 'b') == '4/m_b'
    assert laue_key('4/mmm', 'c') == '4/mmm' and laue_key('mmm', 'a') == 'mmm'
    with pytest.raises(ValueError):
        laue_key('4/mmm', 'x')
    for key in NEW_KEYS + tuple(LAUE_GROUP_NAMES):
        assert laue_key(*split_laue_key(key)) == key
    assert laue_display_name('4/mmm_a') == '4/mmm (4-fold axis a)'
    assert laue_display_name('m-3m') == 'm-3m'
    suffix = '4/mmm_a'.replace('/', '').replace('-', 'bar')
    assert suffix == '4mmm_a'


def test_cell_metric_and_the_automatic_choice():
    bc = _recip(4.4, 4.0, 4.0)                       # b = c, angles 90 deg
    assert cell_metric_residual(bc, '4/mmm_a') < 1e-12
    assert cell_metric_residual(bc, '4/mmm') > 0.02
    assert fitting_laue_group(bc, 'm-3m')[1] == '4/mmm_a'
    assert fitting_laue_group(bc, '4/mmm')[1] == '4/mmm_a'
    assert fitting_laue_group(_recip(4.0, 4.4, 4.0), 'm-3m')[1] == '4/mmm_b'
    assert fitting_laue_group(_recip(4.0, 4.0, 4.4), 'm-3m')[1] == '4/mmm'
    pseudo = _recip(5.925, 5.926, 5.936)
    assert fitting_laue_group(pseudo, 'm-3m')[1] == 'm-3m'
    assert fitting_laue_group(pseudo, '4/mmm_a')[1] == '4/mmm_a'
    # on a cubic cell all three settings fit; at equal order the axis c comes first
    order = [g for g in sorted(_EXPECTED_ORDERS, key=lambda g: -_EXPECTED_ORDERS[g])
             if _EXPECTED_ORDERS[g] == 16]
    assert order == ['4/mmm', '4/mmm_a', '4/mmm_b']


def test_allowed_keys_limit_the_automatic_choice():
    bc = _recip(4.4, 4.0, 4.0)
    assert fitting_laue_group(bc, 'm-3m', allowed=LAUE_GROUP_NAMES)[1] == 'mmm'
    assert fitting_laue_group(bc, 'm-3m', allowed=['-1', '2/m'])[1] == '2/m'
    assert fitting_laue_group(bc, 'm-3m', allowed=[])[1] == '-1'
    # a group that fits but is not in `allowed` is not returned
    assert fitting_laue_group(_recip(4.0, 4.0, 4.0), 'm-3m', allowed=['-1'])[1] == '-1'


def test_unwarp_window_does_not_select_a_key_that_its_combo_lacks(qapp):
    from rspace3d.volume_builder_gui import SimpleVolumeGUI
    gui = SimpleVolumeGUI()
    try:
        gui._recip = _recip(4.4, 4.0, 4.0)               # b = c
        gui._check_laue(switch=True)
        assert gui.laue_combo.currentData() == 'mmm'
        assert 'mmm' in gui.laue_note.text()
        gui._on_laue_changed()                           # no exception
        gui._recip = _recip(4.0, 4.0, 4.0)
        gui.laue_combo.setCurrentIndex(gui.laue_combo.findData('4/mmm'))
        gui._check_laue(switch=True)
        assert gui.laue_combo.currentData() == '4/mmm'
    finally:
        gui.close()


def _stepped_volume(step_l):
    """A cubic volume with the l step `step_l` and the steps 0.5 on h and k."""
    vd = _cubic_volume()
    n = vd.L.size
    vd.L = (np.arange(n) - (n - 1) / 2) * step_l
    return vd


def test_unequal_steps_give_a_nearest_mapping_and_one_log_line():
    note = None
    for key, mapping in (('4/mmm', 'exact'), ('4/mmm_a', 'nearest'), ('4/mmm_b', 'nearest'),
                         ('4/m_a', 'nearest'), ('m-3m', 'nearest')):
        sym = symmetrize_volume(_stepped_volume(0.25), key, sigma=None, use_gpu=False)
        assert sym.metadata['symmetry_mapping'] == mapping, key
        note = inexact_mapping_note(sym.metadata['symmetry_mapping'])
        assert (note is None) == (mapping == 'exact')
    assert 'same bin factor' in note and chr(10) not in note
    sym = symmetrize_volume(_stepped_volume(0.5), '4/mmm_a', sigma=None, use_gpu=False)
    assert sym.metadata['symmetry_mapping'] == 'exact'
    assert inexact_mapping_note('interpolated') is None and inexact_mapping_note(None) is None


def _cubic_volume(n=9, seed=0, nan=0.1):
    rng = np.random.default_rng(seed)
    axis = (np.arange(n) - (n - 1) / 2) * 0.5
    data = rng.gamma(3.0, 1.0, (n, n, n)).astype(np.float32)
    data[rng.random(data.shape) < nan] = np.nan
    counts = np.where(np.isfinite(data), rng.integers(1, 9, data.shape), 0).astype(np.int32)
    meta = {'ub': np.eye(3) / 6.0, 'wavelength': 1.0, 'grid_kind': 'hkl_regular'}
    return VolumeData(data, axis.copy(), axis.copy(), axis.copy(), 'HK', meta, counts, None)


def _permuted(vd, order):
    t = lambda a: None if a is None else np.ascontiguousarray(np.transpose(a, order))
    axes = [vd.H, vd.K, vd.L]
    return VolumeData(t(vd.intensity), axes[order[0]], axes[order[1]], axes[order[2]], 'HK',
                      dict(vd.metadata), t(vd.counts), t(vd.variance))


@pytest.mark.parametrize('key, order', [('4/mmm_a', (1, 2, 0)), ('4/mmm_b', (2, 0, 1)),
                                        ('4/m_a', (1, 2, 0)), ('4/m_b', (2, 0, 1))])
@pytest.mark.parametrize('kw', [dict(sigma=None, weighting='equal', rejection='none'),
                                dict(sigma=3.0, weighting='counts', rejection='mad')])
def test_new_key_equals_a_permutation_of_the_axes_and_the_c_axis_group(key, order, kw):
    """'4/mmm_a' on (h, k, l) = '4/mmm' on the axes (k, l, h), with the axes permuted back."""
    vd = _cubic_volume()
    direct = symmetrize_volume(vd, key, use_gpu=False, **kw)
    base = key.split('_')[0]
    via = symmetrize_volume(_permuted(vd, order), base, use_gpu=False, **kw)
    back = np.transpose(via.intensity, np.argsort(order))
    assert direct.metadata['symmetry_mapping'] == 'exact'
    assert direct.metadata['laue_group'] == key
    assert np.array_equal(np.isnan(direct.intensity), np.isnan(back))
    assert np.allclose(direct.intensity, back, rtol=1e-6, atol=1e-6, equal_nan=True)


def _reference_mean(vd, key):
    """Mean of the finite members inside the grid (signed permutations, equal steps)."""
    axes = (vd.H, vd.K, vd.L)
    step = vd.H[1] - vd.H[0]
    shape = vd.intensity.shape
    hkl = np.stack(np.meshgrid(*axes, indexing='ij'), -1)
    acc, n = np.zeros(shape), np.zeros(shape)
    for W in get_symmetry_operations(key):
        m = hkl @ np.asarray(W, float).T
        idx = [np.rint((m[..., d] - axes[d][0]) / step).astype(int) for d in range(3)]
        ok = np.all([(idx[d] >= 0) & (idx[d] < shape[d]) for d in range(3)], axis=0)
        v = np.full(shape, np.nan)
        v[ok] = vd.intensity[idx[0][ok], idx[1][ok], idx[2][ok]]
        f = np.isfinite(v)
        acc[f] += v[f]
        n[f] += 1
    return np.where(n > 0, acc / np.maximum(n, 1), np.nan)


@pytest.mark.parametrize('key', ['m-3m', '4/mmm', '4/mmm_a', '4/mmm_b', '4/m_a'])
def test_unequal_ranges_use_the_members_inside_the_grid(key):
    """h, k from -7.4 to 7.4 and l from -6 to 6: members outside the grid are not used."""
    step = 0.2
    H = np.arange(-37, 38) * step
    L = np.arange(-30, 31) * step
    rng = np.random.default_rng(4)
    data = rng.gamma(4.0, 1.0, (H.size, H.size, L.size)).astype(np.float32)
    data[rng.random(data.shape) < 0.1] = np.nan
    meta = {'ub': np.eye(3) / 6.0, 'wavelength': 1.0, 'grid_kind': 'hkl_regular'}
    vd = VolumeData(data, H.copy(), H.copy(), L.copy(), 'HK', meta)
    out = symmetrize_volume(vd, key, sigma=None, weighting='equal', rejection='none',
                            use_gpu=False)
    ref = _reference_mean(vd, key)
    assert out.metadata['symmetry_mapping'] == 'exact'
    assert np.array_equal(np.isnan(out.intensity), np.isnan(ref))
    assert np.allclose(out.intensity, ref, rtol=1e-5, atol=1e-5, equal_nan=True)


def test_metadata_round_trip_with_a_new_key_and_a_selection(tmp_path):
    z = np.ones((5, 5, 5))
    vol = engine.Volume(z, np.linspace(-1, 1, 5), np.linspace(-1, 1, 5), np.linspace(-1, 1, 5),
                        np.eye(3) / 6.0, np.ones((5, 5, 5), np.int32), 1.0, z.copy())
    meta = {'frame_selection': '1-369, 901-1289', 'n_frames_used': 758,
            'n_frames_total': 1735}
    raw = engine.reconstruct_to_volumedata(vol, extra_meta=meta)
    sym = symmetrize_volume(raw, '4/mmm_a', sigma=None, use_gpu=False)
    for name, v in (('raw.h5', raw), ('x_recon_sym_4mmm_a.h5', sym)):
        save_volume_h5(str(tmp_path / name), v)
        back = load_volume_h5(str(tmp_path / name))
        for k, val in meta.items():
            assert back.metadata[k] == val
    assert load_volume_h5(str(tmp_path / 'x_recon_sym_4mmm_a.h5')).metadata['laue_group'] \
        == '4/mmm_a'


def test_measure_compare_and_split_halves_accept_a_new_key():
    from rspace3d import averaging as av
    vol = _cubic_volume(n=13, seed=5, nan=0.05)
    measured = av.measure_volume(vol, '4/mmm_a', use_gpu=False)
    assert measured['laue_group'] == '4/mmm_a'
    cands = av.automatic_candidates(True)
    c = av.compare_candidates(vol, '4/mmm_a', measured, cands, use_gpu=False)
    assert len(c['rows']) >= len(cands)
    split = av.compare_split_halves(_cubic_volume(n=13, seed=6, nan=0.05),
                                    _cubic_volume(n=13, seed=7, nan=0.05), '4/mmm_a', cands,
                                    use_gpu=False)
    assert split


def test_viewer_shows_the_name_of_a_new_key(qapp, tmp_path, monkeypatch):
    from rspace3d import rsp_viewer
    vol = symmetrize_volume(_cubic_volume(), '4/mmm_a', sigma=None, use_gpu=False)
    path = str(tmp_path / 'x_recon_sym_4mmm_a.h5')
    save_volume_h5(path, vol)
    shown = []

    class Text(rsp_viewer.QTextEdit):
        def setPlainText(self, text):
            shown.append(text)
            super().setPlainText(text)
    monkeypatch.setattr(rsp_viewer, 'QTextEdit', Text)
    monkeypatch.setattr(rsp_viewer.QDialog, 'exec', lambda self: 0)
    win = rsp_viewer.UnifiedViewer()
    try:
        win._load_file(path)
        qapp.processEvents()
        assert win.vol is not None and win.vol.metadata['laue_group'] == '4/mmm_a'
        win._show_info()
        assert any('Laue group: 4/mmm (4-fold axis a)' in t for t in shown)
    finally:
        win.close()


def test_command_line_accepts_the_new_keys():
    import argparse
    from rspace3d import volume_process
    seen = {}

    class Stop(Exception):
        pass

    def parse(self, args=None, namespace=None):
        seen['choices'] = next(a.choices for a in self._actions if a.dest == 'laue')
        raise Stop
    orig = argparse.ArgumentParser.parse_args
    argparse.ArgumentParser.parse_args = parse
    try:
        with pytest.raises(Stop):
            volume_process.main()
    finally:
        argparse.ArgumentParser.parse_args = orig
    assert set(NEW_KEYS) <= set(seen['choices']) and '4/mmm' in seen['choices']


# ──────────────────────────────────────────────────────────────────
# Window
# ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope='module')
def qapp():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def gui(qapp, tmp_path, monkeypatch):
    from rspace3d import reconstruct_gui as rg
    monkeypatch.setattr(rg, 'HAS_GPU', False)
    g = rg.ReconstructGUI()
    g._folder = str(tmp_path / 'run467')
    yield g
    g.close()


def _select(g, group, axis=None):
    g.laue_combo.setCurrentIndex(g.laue_combo.findData(group))
    if axis is not None:
        g.axis_combo.setCurrentIndex(g.axis_combo.findData(axis))


def test_window_axis_choice_shows_only_for_4m_and_4mmm(gui):
    assert gui.laue_combo.count() == 11
    assert gui.laue_combo.currentData() == 'm-3m' and gui.axis_combo.isHidden()
    _select(gui, '4/mmm')
    assert not gui.axis_combo.isHidden() and gui._laue() == '4/mmm'
    assert gui.axis_combo.currentData() == 'c'
    _select(gui, '4/mmm', 'a')
    assert gui._laue() == '4/mmm_a'
    sym, _ = gui._output_paths(gui._laue(), '')
    assert os.path.basename(sym) == 'run467_recon_sym_4mmm_a.h5'
    _select(gui, '4/m')
    assert not gui.axis_combo.isHidden() and gui._laue() == '4/m_a'
    _select(gui, 'mmm')
    assert gui.axis_combo.isHidden() and gui._laue() == 'mmm'
    gui._update_files_label()
    assert gui.files_label.text().startswith('run467_recon_sym_mmm.h5')


def test_window_axis_change_resets_the_panel_and_the_frame_model(gui, monkeypatch):
    calls = []
    monkeypatch.setattr(gui.avg_panel, 'drop_laue_results', lambda msg: calls.append(msg))
    _select(gui, '4/mmm')
    n = len(calls)
    _select(gui, '4/mmm', 'b')
    assert len(calls) == n + 1 and gui._laue() == '4/mmm_b'


def test_window_switches_to_the_axis_that_the_cell_allows(gui):
    gui._recip = _recip(4.4, 4.0, 4.0)
    gui._check_laue(switch=True)
    assert gui._laue() == '4/mmm_a' and not gui.axis_combo.isHidden()
    assert gui.laue_note.kind() == 'warn' and '4/mmm (4-fold axis a)' in gui.laue_note.text()
    gui._check_laue()
    assert gui.laue_note.isHidden()
    _select(gui, '4/mmm', 'c')                        # not allowed: an error note
    assert gui.laue_note.kind() == 'error' and not gui.laue_note.isHidden()
    gui._recip = _recip(5.925, 5.926, 5.936)
    _select(gui, 'm-3m')
    gui._check_laue(switch=True)
    assert gui._laue() == 'm-3m'


def test_window_frames_field_validation_and_count(gui, monkeypatch):
    gui._numbers, gui._n_total = list(range(1, 101)), 100
    gui._par = 'x.par'
    gui._facts = {'Frames': '100 (run_*.cbf)'}
    assert gui._frame_choice() == (None, None, None) and gui.frames_note.isHidden()
    gui.nframes.setText('5')
    assert gui._frame_choice() == (5, None, None) and gui._frames_count() == 5
    assert gui.info_grid._values['Frames'].text() == '100 (run_*.cbf), 5 selected'
    gui.nframes.setText('1-30,  61 - 90')
    assert gui._frame_choice() == (None, '1-30, 61-90', None) and gui._frames_count() == 60
    assert gui.frames_note.isHidden()
    gui.nframes.setText('1-30; 61')
    assert not gui.frames_note.isHidden() and "not allowed: ';'" in gui.frames_note.text()
    assert gui.run_note.text().startswith('The frame selection is not valid')
    assert 'not valid' in gui.info_grid._values['Frames'].text()
    started = []
    monkeypatch.setattr(gui, '_reconstruct', lambda: started.append(1))
    gui._run()
    assert started == []                               # no run with a wrong text
    gui.nframes.setText('200')
    assert 'The folder has 100 frames' in gui.frames_note.text()
    gui.nframes.setText('all')
    assert gui._frame_choice() == (None, None, None) and gui.frames_note.isHidden()
    gui._reset_advanced()
    assert gui.nframes.text() == ''


def test_window_axis_hint_shows_only_with_the_axis_row(gui):
    assert gui.axis_hint.isHidden() and gui.axis_combo.isHidden()
    _select(gui, '4/mmm')
    assert not gui.axis_hint.isHidden() and not gui.axis_combo.isHidden()
    _select(gui, 'mmm')
    assert gui.axis_hint.isHidden() and gui.axis_combo.isHidden()


def test_window_frames_field_ignores_other_digit_characters(gui):
    gui._numbers, gui._n_total = list(range(1, 101)), 100
    for text in ('²', '٣', '5²'):
        gui.nframes.setText(text)                        # no exception
        err = gui._frame_choice()[2]
        assert err and 'not allowed' in err
    gui.nframes.setText('5')
    assert gui._frame_choice() == (5, None, None)
    gui.nframes.setText('5-5')
    assert gui._frame_choice() == (None, '5', None) and gui._frames_count() == 1


def test_window_frame_tools_results_stay_while_the_user_types(gui, monkeypatch):
    gui._numbers, gui._n_total = list(range(1, 101)), 100
    gui._vd = object()
    removed = []
    monkeypatch.setattr(gui.avg_panel, 'clear_frame_results', lambda msg: removed.append(msg))
    monkeypatch.setattr(gui.nframes, 'hasFocus', lambda: True)    # the user is typing
    gui.nframes.setText('1-')
    assert not gui.frames_note.isHidden() and not removed          # the note is live
    gui.nframes.setText('1-30')
    assert gui.frames_note.isHidden() and not removed
    gui.nframes.editingFinished.emit()
    assert len(removed) == 1
    gui.nframes.setText('1-3')                                     # type again ...
    gui.nframes.setText('1-30')                                    # ... and return
    gui.nframes.editingFinished.emit()
    assert len(removed) == 1                                       # same frames: no removal
    gui.nframes.setText('1-29, 30')                                # same frames, other text
    gui.nframes.editingFinished.emit()
    assert len(removed) == 1
    gui.nframes.setText('1-31')
    gui.nframes.editingFinished.emit()
    assert len(removed) == 2


def test_window_frames_change_is_a_setup_change(gui, monkeypatch):
    gui._numbers, gui._n_total = list(range(1, 101)), 100
    gui._vd = object()
    removed = []
    monkeypatch.setattr(gui.avg_panel, 'clear_frame_results', lambda msg: removed.append(msg))
    key = gui._settings_key()
    gui.nframes.setText('1-30, 61-90')
    assert removed and gui._frame_stale and gui._settings_key() != key
    key2 = gui._settings_key()
    gui.nframes.setText('1 - 30 , 61-90')              # the same selection
    assert gui._settings_key() == key2


def test_window_passes_the_selection_to_the_engine(gui, monkeypatch):
    from rspace3d import reconstruct_gui as rg
    seen = {}

    def fake(folder, **kw):
        seen.update(kw)
        info = {'polarization_axis': 'none', 'geometry_method': 'tabbin',
                'frame_selection': '1-30, 61-90', 'n_frames': 60, 'n_total': 100}
        z = np.ones((3, 3, 3))
        return engine.Volume(z, z[0, 0], z[0, 0], z[0, 0], np.eye(3), z.astype(np.int32),
                             1.0), info
    monkeypatch.setattr(rg.rawrecon, 'reconstruct_dataset', fake)
    monkeypatch.setattr(rg.rawrecon, 'bragg_registration_rms', lambda vol: 0.0)
    w = rg.WorkerThread(gui._do_reconstruct, 'f', ((-1, 1),) * 3, 0.5, (1, 1, 1),
                        '1-30, 61-90', False, None, False, 'x.par', True, True)
    out = []
    w.finished.connect(out.append)
    w.start()
    assert w.wait(30_000)
    from PyQt6.QtWidgets import QApplication
    QApplication.processEvents()
    assert seen['frames'] == '1-30, 61-90' and seen['nframes'] is None
    vd = out[0][1]
    assert vd.metadata['frame_selection'] == '1-30, 61-90'
    assert vd.metadata['n_frames_used'] == 60 and vd.metadata['n_frames_total'] == 100


# ──────────────────────────────────────────────────────────────────
# Second fix round: the start size and the name of the group in lines that the user reads
# ──────────────────────────────────────────────────────────────────
def test_window_axis_hint_is_in_the_row_of_the_axis_box(gui):
    """The hint of the 4-fold axis stands beside the box: the row adds no line of its own.

    The offscreen platform has no fonts, so this test checks the structure only. The native
    measurement of the scroll bar (validation/v2.12-frame-selection-axis) checks the height."""
    form = gui.laue_form
    assert form.rowCount() == 2                         # Laue group, 4-fold axis (no hint row)
    assert gui._axis_row.indexOf(gui.axis_combo) >= 0 and gui._axis_row.indexOf(gui.axis_hint) >= 0
    assert 'tetragonal' in gui.axis_hint.text() and 'metric' in gui.axis_hint.text()
    form.invalidate()
    base = form.sizeHint().height()
    _select(gui, '4/mmm', 'a')
    form.invalidate()
    grown = form.sizeHint().height() - base
    assert 0 < grown <= gui.axis_combo.sizeHint().height() + form.verticalSpacing()


def _stop_after_the_symmetrising_line(gui, monkeypatch, laue):
    from PyQt6.QtCore import QThread
    from rspace3d import reconstruct_gui as rg
    lines = []

    class Current:
        _emit_log = staticmethod(lines.append)
        _emit_progress = staticmethod(lambda *a: None)

    class Stop(Exception):
        pass

    def stop(*a, **k):
        raise Stop
    monkeypatch.setattr(QThread, 'currentThread', staticmethod(lambda: Current))
    monkeypatch.setattr(rg, 'symmetrize_volume', stop)
    with pytest.raises(Stop):
        gui._do_sym(object(), laue, {}, 'averaging x', 1, 1, 'p.h5')
    return [x for x in lines if 'symmetrising' in x]


@pytest.mark.parametrize('key, text', [
    ('4/mmm_a', '  symmetrising 4/mmm (4-fold axis a) (16 ops), averaging x ...'),
    ('4/m_b', '  symmetrising 4/m (4-fold axis b) (8 ops), averaging x ...'),
    ('4/mmm', '  symmetrising 4/mmm (16 ops), averaging x ...'),      # the old line, unchanged
    ('mmm', '  symmetrising mmm (8 ops), averaging x ...'),
    ('m-3m', '  symmetrising m-3m (48 ops), averaging x ...'),
])
def test_run_log_shows_the_group_and_the_axis_in_words(gui, monkeypatch, key, text):
    assert _stop_after_the_symmetrising_line(gui, monkeypatch, key) == [text]


def test_command_line_shows_the_group_and_the_axis_in_words(tmp_path, monkeypatch, capsys):
    from rspace3d import volume_process as vp

    class Stop(Exception):
        pass

    def stop(*a, **k):
        raise Stop
    hdr = {'nx': 10, 'ny': 10, 'fixed_value': 0.0, 'plane_type': 'HK', 'wavelength': 0.5}
    cell = dict(a=5.9, b=5.9, c=5.9, alpha=90.0, beta=90.0, gamma=90.0)
    monkeypatch.setattr(vp, '_filter_numbered_imgs', lambda folder: ['run_1.img', 'run_2.img'])
    monkeypatch.setattr(vp, '_read_header_fast', lambda path: hdr)
    monkeypatch.setattr(vp, 'resolve_unit_cell', lambda folder, h: (cell, None))
    monkeypatch.setattr(vp, 'load_unwarp_folder', stop)
    out = {}
    for key in ('4/mmm_a', '4/mmm'):
        monkeypatch.setattr('sys.argv', ['rspace3d-process', str(tmp_path), '--laue', key,
                                         '--morph', '3'])
        with pytest.raises(Stop):
            vp.main()
        out[key] = [x for x in capsys.readouterr().out.splitlines() if x.startswith('  Laue:')]
    assert out['4/mmm_a'] == ['  Laue:    4/mmm (4-fold axis a) (16 ops)']
    assert out['4/mmm'] == ['  Laue:    4/mmm (16 ops)']


def test_isosurface_title_shows_the_group_and_the_axis_in_words(monkeypatch):
    from rspace3d import volume_isosurface as vi
    titles = []
    monkeypatch.setattr(vi, '_plot_plotly', lambda data, H, K, L, iso, cmap, op, title, save:
                        titles.append(title))
    for key in ('4/mmm_a', 'mmm'):
        ax = np.arange(4.0)
        vol = VolumeData(np.ones((4, 4, 4), np.float32), ax, ax.copy(), ax.copy(), 'HK',
                         {'laue_group': key})
        vi.plot_isosurface(vol, isovalue=1.0, method='plotly')
    assert titles == ['Isosurface at I=1 (4/mmm (4-fold axis a))', 'Isosurface at I=1 (mmm)']


def test_frame_chain_error_shows_the_group_and_the_axis_in_words():
    import dataclasses
    from test_frame_tools import _Scene
    from rspace3d.rawrecon import frames as ft
    sc = _Scene(nf=6)
    off = dataclasses.replace(sc.setup, ranges=((0.5, 3.0), (0.5, 1.0), (2.0, 3.0)))
    with pytest.raises(ValueError, match=r'In Laue group 4/mmm \(4-fold axis a\), no voxel'):
        ft.run_frame_chain(off, '4/mmm_a', rings=None, use_gpu=False)


# ──────────────────────────────────────────────────────────────────
# Third fix round (outside review)
# ──────────────────────────────────────────────────────────────────
def test_fitting_laue_group_obeys_allowed_and_takes_the_smallest_residual():
    cell = _recip(4.000, 4.004, 4.001)             # a = c is the best 4-fold setting
    keys = ['-1', 'mmm', '4/mmm', '4/mmm_a', '4/mmm_b']
    # the group asked for is not in `allowed`: search, do not return it
    res, fit = fitting_laue_group(cell, 'm-3m', allowed=keys)
    assert res > 0 and fit == '4/mmm_b'              # residuals 2.0e-3, 1.5e-3, 5.0e-4
    assert fitting_laue_group(_recip(4.0, 4.0, 4.0), 'm-3m', allowed=['-1'])[1] == '-1'
    # a group that fits and is allowed stays
    assert fitting_laue_group(cell, '4/mmm', allowed=keys)[1] == '4/mmm'
    # equal residuals: the order c, a, b
    assert fitting_laue_group(_recip(4.0, 4.0, 4.0), 'm-3m', allowed=keys[:1] + keys[2:])[1] == '4/mmm'
    assert fitting_laue_group(_recip(4.0, 4.0, 4.0), 'm-3m', allowed=['-1', '4/mmm_a', '4/mmm_b'])[1] == '4/mmm_a'
    # no change for cells with one answer
    assert fitting_laue_group(_recip(4.4, 4.0, 4.0), '4/mmm')[1] == '4/mmm_a'
    assert fitting_laue_group(_recip(5.925, 5.926, 5.936), 'm-3m')[1] == 'm-3m'


def test_settings_key_treats_equal_frame_selections_as_equal(gui):
    gui._numbers, gui._n_total = list(range(1, 101)), 100
    gui.nframes.setText('60')
    key = gui._settings_key()
    gui.nframes.setText('1-60')
    assert gui._settings_key() == key
    gui.nframes.setText('1-59')
    assert gui._settings_key() != key
    gui.nframes.setText('1-')
    bad1 = gui._settings_key()
    gui.nframes.setText('1-x')
    assert gui._settings_key() != bad1 and bad1 != key


def test_set_laue_changes_both_boxes_with_one_update(gui, monkeypatch):
    seen = []
    monkeypatch.setattr(gui.avg_panel, 'drop_laue_results', lambda msg: seen.append(gui._laue()))
    gui._set_laue('4/mmm_a')
    assert seen == ['4/mmm_a']                        # one call, never an inconsistent pair
    assert gui._laue() == '4/mmm_a' and not gui.axis_combo.isHidden()
    seen.clear()
    gui._set_laue('mmm')
    assert seen == ['mmm'] and gui.axis_combo.isHidden()
    gui._set_laue('not-a-key')                        # no exception, no change of the group
    assert gui._laue() == 'mmm'


def _fake_folder(monkeypatch, numbers):
    from rspace3d import reconstruct_gui as rg
    monkeypatch.setattr(rg.rawrecon, 'find_cbf_frames',
                        lambda folder: (['x_%05d.cbf' % n for n in numbers], list(numbers)))
    monkeypatch.setattr(rg.rawrecon, 'find_crysalis_par', lambda folder, **kw: None)


def test_new_folder_with_other_frame_numbers_resets_the_frames_field(gui, monkeypatch, tmp_path):
    _fake_folder(monkeypatch, range(1, 101))
    gui._set_folder(str(tmp_path))
    gui.nframes.setText('1-30')
    lines = []
    monkeypatch.setattr(gui, '_log', lambda text: lines.append(text))
    gui._set_folder(str(tmp_path))                    # the same frame numbers: text stays
    assert gui.nframes.text() == '1-30'
    _fake_folder(monkeypatch, range(1, 51))
    gui._set_folder(str(tmp_path))
    assert gui.nframes.text() == ''
    assert sum('Frames to use was reset for the new folder.' in t for t in lines) == 1
