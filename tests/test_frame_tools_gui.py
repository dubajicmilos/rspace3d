"""Tests of the frame tools in the Averaging tab of the reconstruction GUI (rspace3d 2.6.0).

Offscreen, CPU, synthetic frames (the scene of `test_frame_tools`). The panel
runs its jobs synchronously, except in the test of the threaded Cancel.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from rspace3d import averaging as av
from rspace3d.rawrecon import engine, frames as ft
from rspace3d.volume_builder import load_volume_h5
from test_frame_tools import NF, RING_R, _Scene, _same

RING_FRAMES_MID = slice(40, 50)      # frames near the maximum of the injected ring


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    from rspace3d import reconstruct_gui, volume_builder, volume_builder_gui
    for mod in (volume_builder, volume_builder_gui, reconstruct_gui):
        monkeypatch.setattr(mod, 'HAS_GPU', False)


@pytest.fixture(scope='module')
def qapp():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(scope='module')
def scene():
    """Frames with a powder ring, a scale term and extra counts, and the plain volume."""
    sc = _Scene(ring=True, scale=True, add=True)
    sc.setup.meta.update(name='x', folder='synthetic')
    sc.raw = engine.reconstruct_to_volumedata(sc.setup.reconstruct(use_gpu=False))
    return sc


def _gui(qapp, monkeypatch, tmp_path, sc):
    from rspace3d import reconstruct_gui as rg
    monkeypatch.setattr(rg.QFileDialog, 'getSaveFileName',
                        staticmethod(lambda *a, **k: (str(tmp_path / 's.h5'), '')))
    gui = rg.ReconstructGUI()
    gui._vd, gui._info, gui._folder = sc.raw, {'name': 'x', 'setup': sc.setup}, str(tmp_path)
    gui._set_post_enabled(True)
    gui.laue_combo.setCurrentIndex(gui.laue_combo.findData('mmm'))
    gui.avg_panel.synchronous = True
    gui.avg_panel.crop_chk.setChecked(False)
    return gui, gui.avg_panel


def _wait(qapp, worker):
    assert worker.wait(120_000)
    for _ in range(5):
        qapp.processEvents()


def _slots(panel):
    return [panel.volume_combo.itemData(i) for i in range(panel.volume_combo.count())]


def _select(panel, slot):
    panel.volume_combo.setCurrentIndex(_slots(panel).index(slot))


def test_group_hidden_for_unwarp_data_and_shown_for_raw_frames(qapp, monkeypatch, tmp_path):
    from rspace3d.reconstruct_gui import ReconstructGUI
    from rspace3d.volume_builder_gui import SimpleVolumeGUI
    builder = SimpleVolumeGUI()
    assert builder.avg_panel.frame_group.isHidden()
    assert builder.avg_panel._frame_request is None
    gui = ReconstructGUI()
    panel = gui.avg_panel
    assert gui.frame_tools_group is panel.frame_group and not panel.frame_group.isHidden()
    assert _slots(panel) == ['none'] and panel.selected_volume() is None
    assert 'sum|A - B| / sum(|A| + |B|)' in panel.frame_help.text()
    # no reconstruction yet: a message, no job
    panel.ring_btn.click()
    assert 'Reconstruct a volume first' in panel.frame_label.text()
    assert not panel.is_busy() and panel._fvols == {}


def test_ring_mask_and_frame_model_make_selectable_volumes(qapp, monkeypatch, tmp_path, scene):
    gui, panel = _gui(qapp, monkeypatch, tmp_path, scene)
    s, raw = scene.setup, scene.raw
    panel.ring_btn.click()
    assert _slots(panel) == ['none', 'mask']
    mask_vd = panel._fvols['mask']
    assert mask_vd.metadata['frame_correction'] == 'ring_mask'
    rings = mask_vd.metadata['frame_tools']['rings']
    assert rings['mode'] == 'detected' and rings['ring_windows'] == 1
    assert 0 < rings['masked_pixel_frac_total'] < 0.1
    assert np.allclose(sorted(panel._rings['radii_px']), RING_R, atol=0.6)
    assert 'Ring mask: 1 event(s)' in panel.frame_label.text()
    # Frame model keeps the ring detection and the mask volume of the panel (no second
    # ring pass). The chain makes the masked volume again, so the run takes 7 passes.
    passes = []
    panel._progress_fn = lambda i, n: passes.append(n)
    panel.model_btn.click()
    assert _slots(panel) == ['none', 'mask', 'model'] and panel._fvols['mask'] is mask_vd
    assert set(passes) == {7 * NF}                 # masked volume + 3 x (pass A + corrected)
    model_vd = panel._fvols['model']
    info = model_vd.metadata['frame_tools']
    assert model_vd.metadata['frame_correction'] == 'ring_mask+frame_model'
    assert info['iterations'] == 3 and info['rings']['mode'] == 'detected'
    assert panel.volume_combo.itemText(2) == 'ring mask + frame model (Laue group mmm)'
    assert panel._tables is not None and panel._fmap is not None
    assert panel._fmap[0] == 'Frame map of the first iteration (ring-masked volume)'
    assert panel._fmap[2] == 'model'
    assert len(panel.frame_figure.axes) >= 2
    # the GUI result is the result of the core chain with automatic ring detection
    core = ft.run_frame_chain(s, 'mmm', raw=raw, use_gpu=False, keep_masked=True)
    assert _same(model_vd.intensity, core.corrected.intensity)
    assert np.array_equal(model_vd.counts, core.corrected.counts)
    assert _same(mask_vd.intensity, core.masked.intensity)
    # the selection is the volume of the host
    _select(panel, 'model')
    assert gui._source()[0] is model_vd and gui._source()[1] == '_ringmask_framemodel'
    assert not gui.source_note.isHidden()
    _select(panel, 'none')
    assert gui._source()[0] is raw and gui.source_note.isHidden()


def test_split_half_check_measure_compare_and_save(qapp, monkeypatch, tmp_path, scene):
    gui, panel = _gui(qapp, monkeypatch, tmp_path, scene)
    panel.split_btn.click()                       # no Measure yet: disabled, no job
    assert panel._split is None and panel.split_table.isHidden()
    panel._split_check()
    assert 'Click Measure first' in panel.frame_label.text()
    panel.ring_btn.click()
    _select(panel, 'mask')
    mask_vd = panel._fvols['mask']
    panel._measure()
    assert panel._state['volume'] is mask_vd       # bin factors 1: the volume itself
    panel._compare()
    names = [r['name'] for r in panel._compared['rows']]
    assert panel.result_table.columnCount() == len(av.COMPARE_HEADERS)
    panel.split_btn.click()
    split = panel._split
    assert split is not None and set(split['rows']) == set(names)
    halves = panel._halves[1]
    assert panel._halves[0] == 'mask' and len(halves) == 2
    assert np.array_equal(halves[0].counts + halves[1].counts, mask_vd.counts)
    ref = split['rows'][av.BIAS_REFERENCE]
    assert 0 < ref['total'] < 1 and len(ref['shells']) == 10
    # the Results table has the split-half column, the shell table one column per candidate
    n = len(av.COMPARE_HEADERS)
    assert panel.result_table.columnCount() == n + 1
    assert panel.result_table.horizontalHeaderItem(n).text() == 'split-half'
    row = names.index(av.BIAS_REFERENCE)
    assert panel.result_table.item(row, n).text() == f"{ref['total']:.4f}"
    assert panel.split_table.columnCount() == len(names) + 1 and panel.split_table.rowCount() == 13
    assert not panel.split_table.isHidden()
    vt = panel.volumes_table
    assert vt.rowCount() == 2 and vt.item(1, 3).text() == f"{ref['total']:.4f}"
    # a second check reuses the halves (no pass over the frames)
    passes = []
    panel._progress_fn = lambda i, k: passes.append(k)
    panel._split_check()
    assert panel._halves[1] is halves and passes and max(passes) <= len(names)
    # Apply and Symmetrise + Save: the selected volume and the frame-tool metadata
    panel.apply_combo.setCurrentText(av.BIAS_REFERENCE)
    panel._apply()
    gui._symmetrize_save()
    _wait(qapp, gui._worker)
    back = load_volume_h5(str(tmp_path / 's.h5'))
    assert back.metadata['frame_correction'] == 'ring_mask'
    assert back.metadata['frame_tools']['rings']['ring_windows'] == 1
    choice = back.metadata['averaging_choice']
    assert choice['name'] == av.BIAS_REFERENCE
    assert choice['candidates'][av.BIAS_REFERENCE]['split_half']['total'] == pytest.approx(
        ref['total'])
    assert choice['split_half']['halves_rule'].startswith('frame-number parity')
    # another volume releases the halves and the split-half results
    _select(panel, 'none')
    assert panel._halves is None and panel._split is None
    assert panel.result_table.columnCount() == n and panel._state is None
    assert panel.split_table.isHidden()


def test_frame_map_marks_the_detected_rings(qapp, monkeypatch, tmp_path, scene):
    gui, panel = _gui(qapp, monkeypatch, tmp_path, scene)
    panel.map_btn.click()
    title, fmap, slot = panel._fmap
    assert title == "Frame map of 'no correction'" and slot == 'none'
    assert fmap.excess.shape == (NF, 72)
    assert np.allclose(sorted(fmap.rings), RING_R, atol=0.6)
    assert len(panel._rings['radii_q']) == len(RING_R)
    ax = panel.frame_figure.axes[1]
    assert sum(1 for ln in ax.lines if ln.get_linestyle() == '--') == len(RING_R)
    assert '4 ring radii' in panel.frame_label.text()
    # the ring frames have more counts than the prediction in the frame x sector image
    ring = np.nanmedian(fmap.excess[RING_FRAMES_MID])
    assert ring > np.nanmedian(fmap.excess[:20])
    # the image shows the sector-normalised ratio on a log scale centred on 1
    im = panel.frame_figure.axes[0].images[0]
    assert np.allclose(im.get_array().filled(np.nan), fmap.normalised(), equal_nan=True)
    assert im.norm.vmin == pytest.approx(1 / 1.5) and im.norm.vmax == pytest.approx(1.5)
    s = fmap.summary()
    assert f"{s['frames_off_10pct']} of {NF} frames differ" in panel.frame_label.text()


def test_frame_figure_shows_a_hint_before_the_first_map(qapp, monkeypatch, tmp_path, scene):
    from rspace3d import averaging_panel as ap
    gui, panel = _gui(qapp, monkeypatch, tmp_path, scene)
    assert [t.get_text() for t in panel.frame_figure.texts] == [ap.FRAME_MAP_EMPTY]
    assert ap.FRAME_TOOLS_PLACE in panel.engine_label.text()
    assert 'Laue group mmm and 0.2 degree frames' in panel.frame_help.text()
    panel.map_btn.click()
    assert ap.FRAME_MAP_EMPTY not in [t.get_text() for t in panel.frame_figure.texts]
    assert len(panel.frame_figure.axes) >= 2
    gui.laue_combo.setCurrentIndex(gui.laue_combo.findData('4/mmm'))   # removes the map
    assert [t.get_text() for t in panel.frame_figure.texts] == [ap.FRAME_MAP_EMPTY]


def test_no_ring_found_gives_a_hint_and_the_file_tag_follows_the_metadata(qapp, monkeypatch,
                                                                         tmp_path, scene):
    from rspace3d import averaging_panel as ap
    gui, panel = _gui(qapp, monkeypatch, tmp_path, scene)
    # a kept ring detection without a ring (as for a frame subset inside the ring event)
    panel._rings = {'mask': None, 'radii_px': [], 'radii_q': [],
                    'summary': {'mode': 'detected', 'ring_windows': 0, 'ring_frames': 0}}
    panel.model_btn.click()
    assert _slots(panel) == ['none', 'mask', 'model']
    assert ap.NO_RING_HINT in panel.frame_label.text()
    assert panel._fmap[0] == 'Frame map of the first iteration (volume without ring mask)'
    model_vd = panel._fvols['model']
    assert model_vd.metadata['frame_correction'] == 'frame_model'
    _select(panel, 'model')
    assert gui._source() == (model_vd, '_framemodel')
    _select(panel, 'mask')
    assert panel._fvols['mask'].metadata.get('frame_correction', 'none') == 'none'
    assert gui._source() == (panel._fvols['mask'], '')
    panel.ring_btn.click()
    assert ap.NO_RING_HINT in panel.frame_label.text()


def test_cancel_leaves_the_panel_consistent(qapp, monkeypatch, tmp_path, scene):
    gui, panel = _gui(qapp, monkeypatch, tmp_path, scene)
    lines = []
    panel._log_fn = lines.append

    def cancel_at_once(i, n):
        panel.cancel()
    panel._progress_fn = cancel_at_once
    panel.model_btn.click()
    assert not panel.is_busy() and panel._step is None
    assert _slots(panel) == ['none'] and panel._fvols == {} and panel._tables is None
    assert panel.frame_label.text().startswith('Frame model cancelled.')
    assert "keeps the volumes 'no correction'" in panel.frame_label.text()
    assert any('Frame model cancelled' in m for m in lines)
    assert panel.model_btn.isEnabled() and not panel.frame_cancel_btn.isEnabled()
    # a cancelled split-half check keeps the earlier state
    panel._progress_fn = lambda i, n: None
    panel._measure()
    panel._progress_fn = cancel_at_once
    panel._split_check()
    assert panel._halves is None and panel._split is None and panel._state is not None
    assert panel.result_table.columnCount() == len(av.COMPARE_HEADERS)
    # the threaded path (PanelWorker): Cancel stops Ring mask
    panel._progress_fn = lambda i, n: None
    panel.synchronous = False
    panel.ring_btn.click()
    assert panel.is_busy() and panel.frame_cancel_btn.isEnabled()
    assert not gui.recon_tab.isEnabled()
    panel.frame_cancel_btn.click()
    _wait(qapp, panel._worker)
    assert not panel.is_busy() and _slots(panel) == ['none'] and gui.recon_tab.isEnabled()
    # then the step runs to the end
    panel.synchronous = True
    panel.ring_btn.click()
    assert _slots(panel) == ['none', 'mask']


def test_new_grid_frames_or_folder_remove_the_kept_results(qapp, monkeypatch, tmp_path, scene):
    gui, panel = _gui(qapp, monkeypatch, tmp_path, scene)
    panel.ring_btn.click()
    _select(panel, 'mask')
    assert gui._source()[0] is panel._fvols['mask']
    gui.hmax.setValue(gui.hmax.value() + 1.0)
    assert _slots(panel) == ['none'] and panel._fvols == {} and panel._rings is None
    assert panel.selected_slot() == 'none' and gui._source()[0] is scene.raw
    panel.ring_btn.click()
    assert 'Reconstruct again' in panel.frame_label.text() and panel._fvols == {}
    # a new reconstruction makes the tools usable again
    gui._frame_stale = False
    panel.ring_btn.click()
    assert _slots(panel) == ['none', 'mask']
    gui.nframes.setValue(5)
    assert panel._fvols == {}
    gui._frame_stale = False
    panel.ring_btn.click()
    gui._set_folder(str(tmp_path))                 # a folder without frames
    assert panel._fvols == {} and gui._vd is None


def test_defaults_unchanged_when_the_tools_are_not_used(qapp, monkeypatch, tmp_path, scene):
    from rspace3d import reconstruct_gui as rg
    calls = []
    real = rg.symmetrize_volume

    def spy(v, laue, **kw):
        calls.append((v, kw))
        return real(v, laue, **kw)
    monkeypatch.setattr(rg, 'symmetrize_volume', spy)
    gui, panel = _gui(qapp, monkeypatch, tmp_path, scene)
    panel._measure()
    panel._compare()
    assert panel.result_table.columnCount() == len(av.COMPARE_HEADERS)
    panel._apply()
    assert 'split_half' not in panel.choice_metadata()
    panel.clear_applied()
    gui._symmetrize_save()
    _wait(qapp, gui._worker)
    (v, kw), = calls
    assert v is scene.raw and set(kw) == {'sigma', 'progress_callback'} and kw['sigma'] == 3.0
    back = load_volume_h5(str(tmp_path / 's.h5'))
    assert 'frame_correction' not in back.metadata and 'frame_tools' not in back.metadata


def test_cancel_keeps_a_finished_ring_detection_and_frees_the_gpu_pool(qapp, monkeypatch, tmp_path,
                                                                       scene):
    from rspace3d import orbit_engine as oe
    gui, panel = _gui(qapp, monkeypatch, tmp_path, scene)
    frees = []
    monkeypatch.setattr(oe, '_free_gpu_memory', lambda: frees.append(1))
    lines = []
    panel._log_fn = lines.append

    def cancel_after_rings(i, n):            # the chain counts more than NF frames
        if n > NF:
            panel.cancel()
    panel._progress_fn = cancel_after_rings
    panel.model_btn.click()
    assert _slots(panel) == ['none'] and panel._rings is not None
    assert np.allclose(sorted(panel._rings['radii_px']), RING_R, atol=0.6)
    assert frees == [1]
    # the next run uses the kept detection: no ring pass
    passes = []
    panel._progress_fn = lambda i, n: passes.append(n)
    panel.ring_btn.click()
    assert _slots(panel) == ['none', 'mask'] and set(passes) == {NF}
    assert frees == [1]                      # a normal end does not release the pool

    def cancel_at_once(i, n):
        panel.cancel()
    panel._progress_fn = cancel_at_once
    for _ in range(3):
        panel.ring_btn.click()
    assert _slots(panel) == ['none'] and len(frees) == 4
    assert any("removed the earlier 'ring mask' volume before the run" in m for m in lines)
    assert panel.frame_label.text().count('cancelled.') == 1
    # the label names the cancelled step and the kept results, not the text of an earlier step
    assert panel.frame_label.text() == ("Ring mask cancelled. The panel keeps the volumes "
                                        "'no correction', the ring detection (4 ring radii).")


def test_laue_change_removes_the_frame_model_results(qapp, monkeypatch, tmp_path, scene):
    gui, panel = _gui(qapp, monkeypatch, tmp_path, scene)
    panel.model_btn.click()
    assert _slots(panel) == ['none', 'mask', 'model']
    _select(panel, 'model')
    # the frame map of the corrected volume says that the frames are not corrected
    panel.map_btn.click()
    title, _, _ = panel._fmap
    assert title.startswith('Frame map: uncorrected frames (ring mask only)')
    mask_vd, rings = panel._fvols['mask'], panel._rings
    gui.laue_combo.setCurrentIndex(gui.laue_combo.findData('4/mmm'))
    assert _slots(panel) == ['none', 'mask'] and panel.selected_slot() == 'none'
    assert panel._tables is None and panel._fmap is None and panel._halves is None
    assert panel._fvols['mask'] is mask_vd and panel._rings is rings
    assert gui._source()[0] is scene.raw
    assert 'Run Frame model again' in panel.frame_label.text()


def _scores(panel):
    return {s: {k: d.get(k) for k in ('suggestion', 'score', 'split', 'split_diffuse')}
            for s, d in panel._fsummary.items()}


def test_laue_or_bin_change_removes_the_scores_of_the_volume_table(qapp, monkeypatch, tmp_path,
                                                                   scene):
    gui, panel = _gui(qapp, monkeypatch, tmp_path, scene)
    panel.ring_btn.click()
    vals = dict(suggestion='equal, mad 3', score=1.5, split=0.01, split_diffuse=0.02)

    def fill():
        for s in ('none', 'mask'):
            panel._fsummary.setdefault(s, {}).update(vals)
        panel._refresh_volumes()
        assert panel.volumes_table.item(1, 2).text() == '1.5000'
    empty = dict.fromkeys(vals)
    fill()
    gui.bin_z.setValue(gui.bin_z.value() + 1)
    assert _scores(panel) == {'none': empty, 'mask': empty}
    assert [panel.volumes_table.item(i, 2).text() for i in range(2)] == ['-', '-']
    assert _slots(panel) == ['none', 'mask']             # the volumes stay
    fill()
    gui.laue_combo.setCurrentIndex(gui.laue_combo.findData('4/mmm'))
    assert _scores(panel) == {'none': empty, 'mask': empty}
    assert _slots(panel) == ['none', 'mask'] and panel._rings is not None


def test_new_frame_model_run_removes_the_map_of_the_model_volume(qapp, monkeypatch, tmp_path,
                                                                 scene):
    gui, panel = _gui(qapp, monkeypatch, tmp_path, scene)
    panel.model_btn.click()
    _select(panel, 'model')
    panel.map_btn.click()
    assert panel._fmap[2] == 'model'

    def cancel_at_once(i, n):
        panel.cancel()
    panel._progress_fn = cancel_at_once
    panel.model_btn.click()
    assert _slots(panel) == ['none', 'mask'] and panel._fmap is None
    assert 'the frame map' not in panel.frame_label.text()
    from rspace3d import averaging_panel as ap
    assert [t.get_text() for t in panel.frame_figure.texts] == [ap.FRAME_MAP_EMPTY]
    # a map of another volume stays
    panel._progress_fn = lambda i, n: None
    _select(panel, 'mask')
    panel.map_btn.click()
    panel._progress_fn = cancel_at_once
    panel.model_btn.click()
    assert panel._fmap is not None and panel._fmap[2] == 'mask'
    assert 'the frame map' in panel.frame_label.text()


def test_frame_map_image_keeps_short_events_and_shows_cells_without_data_grey(qapp, monkeypatch,
                                                                              tmp_path, scene):
    from matplotlib.colors import to_rgba
    from rspace3d import averaging_panel as ap
    nf = 1735
    ex = np.ones((nf, 72), np.float32)
    ex[1000] = 1.5                           # an event of one frame
    ex[500] = 0.0                            # an empty frame
    ex[:, 7] = np.nan                        # a sector without prediction
    fmap = ft.FrameMap(ex, np.arange(3) + 0.5, np.zeros(3), np.zeros(3), np.zeros(3))
    rows, per = ap.reduce_frame_rows(fmap.normalised())
    assert per == 9 and rows.shape == (193, 72)
    assert np.allclose(np.delete(rows[1000 // 9], 7), 1.5)
    assert np.allclose(np.delete(rows[500 // 9], 7), ft.FRAME_MAP_FLOOR)
    assert np.isnan(rows[:, 7]).all()
    assert np.allclose(np.delete(np.delete(rows, [1000 // 9, 500 // 9], 0), 7, 1), 1.0)
    small = ex[:150]
    assert ap.reduce_frame_rows(small)[0] is small
    gui, panel = _gui(qapp, monkeypatch, tmp_path, scene)
    panel._fmap = ('test', fmap, 'none')
    panel._plot_frame_map()
    ax = panel.frame_figure.axes[0]
    im = ax.images[0]
    arr = im.get_array()
    assert arr.shape == (193, 72) and np.isclose(arr.max(), 1.5)
    assert np.isclose(arr.min(), ft.FRAME_MAP_FLOOR)             # the empty frame
    assert ax.get_ylim() == pytest.approx((-0.5, nf - 0.5))
    assert '9 frames per row' in ax.get_ylabel()
    assert im.cmap.get_bad() == pytest.approx(to_rgba('0.65'))


def test_split_half_check_uses_the_cpu_if_the_gpu_halves_do_not_fit(qapp, monkeypatch, tmp_path,
                                                                     scene):
    from rspace3d import averaging_panel as ap
    assert ap.halves_gpu_note(10 ** 6, None) is None
    assert ap.halves_gpu_note(10 ** 6, 10 ** 9) is None          # 40 MB of 1 GB
    note = ap.halves_gpu_note(10 ** 8, 10 ** 9)                  # 4 GB of 1 GB
    assert note is not None and 'uses the CPU' in note
    gui, panel = _gui(qapp, monkeypatch, tmp_path, scene)
    lines = []
    panel._log_fn = lines.append
    real = panel._frame_request
    panel._frame_request = lambda: dict(real(), use_gpu=True)
    monkeypatch.setattr(ap, '_gpu_free_bytes', lambda device: 1000)
    seen = []

    def fake_split(req, *a, **k):
        seen.append(req)
        raise ValueError('stop')
    monkeypatch.setattr(ap, '_split_job', fake_split)
    panel._state = {'laue': 'mmm', 'measured': {'model': None}}     # enough for the check
    panel._split_check()
    assert seen[0]['halves_gpu'] is False and seen[0]['use_gpu'] is True
    assert any('The halves pass uses the CPU' in m for m in lines)
    monkeypatch.setattr(ap, '_gpu_free_bytes', lambda device: 10 ** 12)
    panel._split_check()
    assert 'halves_gpu' not in seen[1]
