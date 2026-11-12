"""Tests of the viewer window: empty state, loading a volume, the view fit, the plane change,
the line profile panel and the error message. All tests run offscreen on small data."""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from rspace3d import ui
from rspace3d.volume_builder import VolumeData, save_volume_h5

EXAMPLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'example_data', 'example_monoclinic.h5')


@pytest.fixture(scope='module')
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def viewer(qapp):
    from rspace3d.rsp_viewer import UnifiedViewer
    win = UnifiedViewer()
    win.show()
    qapp.processEvents()
    yield win
    win.close()


def _regular_volume():
    """A volume on a regular hkl grid (h and k from -4 to 4, l from -2 to 2)."""
    H = np.linspace(-4, 4, 33)
    L = np.linspace(-2, 2, 17)
    rng = np.random.default_rng(1)
    data = rng.random((33, 33, 17)).astype(np.float32) * 100
    meta = {'ub': np.diag([0.125, 0.125, 0.25]), 'wavelength': 0.5, 'grid_kind': 'hkl_regular'}
    return VolumeData(data, H.copy(), H.copy(), L, 'HK', meta)


def test_empty_state(viewer):
    assert viewer.stack.currentIndex() == 0
    assert not viewer.vol_toolbar.isVisible()
    # one primary button: "Open file..." in the centre of the empty window
    assert viewer.empty_open_btn.property('role') == 'primary'
    assert viewer.open_btn.property('role') is None
    assert not viewer.splitter.widget(1).isVisible()
    assert viewer.minimumWidth() == 1000 and viewer.minimumHeight() == 640


def test_load_volume_file(viewer, qapp):
    viewer._load_file(EXAMPLE)
    qapp.processEvents()
    assert viewer.stack.currentIndex() == 1
    assert viewer.vol_toolbar.isVisible()
    assert viewer.view3d_btn.isVisible() and viewer.info_btn.isVisible()
    assert viewer.open_btn.property('role') is None
    lo, hi = viewer.min_spin.value(), viewer.max_spin.value()
    assert np.isfinite(lo) and np.isfinite(hi) and lo < hi
    assert viewer.file_label.toolTip() == EXAMPLE
    assert viewer.summary_label.text() != ''
    # The range boxes cover the shown data and not much more (one 0.1 step).
    M, e = viewer._current_M_inv, viewer.extent
    ny, nx = viewer._display_data.shape
    dx, dy = (e[1] - e[0]) / nx / 2, (e[3] - e[2]) / ny / 2
    xs = np.array([e[0] + dx, e[1] - dx, e[0] + dx, e[1] - dx])
    ys = np.array([e[2] + dy, e[2] + dy, e[3] - dy, e[3] - dy])
    want = (np.abs(M[0, 0] * xs + M[0, 1] * ys).max(), np.abs(M[1, 0] * xs + M[1, 1] * ys).max())
    for got, w in ((viewer.xrange_spin.value(), want[0]), (viewer.yrange_spin.value(), want[1])):
        assert w - 1e-6 <= got <= w + 0.1 + 1e-6
    assert viewer.xrange_label.text().startswith('h')
    assert viewer.yrange_label.text().startswith('k')


def test_regular_grid_range_is_exact(viewer, tmp_path, qapp):
    path = str(tmp_path / 'regular.h5')
    save_volume_h5(path, _regular_volume())
    viewer._load_file(path)
    qapp.processEvents()
    assert viewer.xrange_spin.value() == pytest.approx(4.0)
    assert viewer.yrange_spin.value() == pytest.approx(4.0)
    xlim = viewer.main_ax.get_xlim()
    assert xlim[0] == pytest.approx(-4.0) and xlim[1] == pytest.approx(4.0)
    # A change of the plane does not refit the view. The fit button does.
    viewer.xrange_spin.setValue(2.0)
    viewer.plane_combo.setCurrentIndex(1)
    assert viewer.xrange_spin.value() == pytest.approx(2.0)
    assert viewer.xrange_label.text().startswith('h')
    assert viewer.yrange_label.text().startswith('l')
    viewer.fit_btn.click()
    assert viewer.xrange_spin.value() == pytest.approx(4.0)
    assert viewer.yrange_spin.value() == pytest.approx(2.0)


def test_plane_change_keeps_the_cut(viewer):
    viewer._load_file(EXAMPLE)
    viewer.slice_spin.setValue(1.0)
    assert viewer._last_cut_value == pytest.approx(1.0)
    viewer.plane_combo.setCurrentIndex(2)
    assert viewer.slice_spin.value() == pytest.approx(1.0)
    assert viewer._last_cut_value == pytest.approx(1.0)
    assert viewer.fixed_label.text() == 'h ='
    assert viewer._display_data is not None and viewer._display_data.ndim == 2


def test_line_profile_panel(viewer):
    viewer._load_file(EXAMPLE)
    right = viewer.splitter.widget(1)
    assert not right.isVisible()
    viewer.line_btn.setChecked(True)
    assert right.isVisible()
    viewer.line_btn.setChecked(False)
    assert not right.isVisible()


def test_unreadable_file_gives_an_error_message(viewer, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(ui, 'show_error', lambda parent, title, text: calls.append((title, text)))
    bad = tmp_path / 'broken.h5'
    bad.write_bytes(b'this is not an hdf5 file')
    viewer._load_file(str(bad))
    assert len(calls) == 1
    assert calls[0][0] == 'This file cannot be opened'
    assert 'broken.h5' in calls[0][1]
    assert viewer.status.currentMessage().startswith('Error')
    assert viewer.stack.currentIndex() == 0


def test_every_control_has_a_tool_tip(viewer):
    for w in (viewer.cmap_combo, viewer.min_spin, viewer.max_spin, viewer.auto_btn,
              viewer.log_check, viewer.xrange_spin, viewer.yrange_spin, viewer.fit_btn,
              viewer.grid_check, viewer.cbar_check, viewer.line_btn, viewer.export_btn,
              viewer.view3d_btn, viewer.info_btn, viewer.plane_combo, viewer.slice_spin,
              viewer.slice_slider, viewer.int_range_spin, viewer.open_btn):
        assert w.toolTip(), w
