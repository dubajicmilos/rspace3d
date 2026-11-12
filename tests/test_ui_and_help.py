"""Tests of the shared look (`rspace3d.ui`), the pages of the windows and the help pages.

All tests run offscreen. The example of the help ("Try the options on one
voxel") must give the values of `symmetrize_volume` for the same orbit.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from rspace3d import averaging as av
from rspace3d import help_content as hc
from rspace3d import orbit_engine as oe
from rspace3d import ui
from rspace3d.averaging import NoiseModel
from rspace3d.volume_builder import VolumeData, symmetrize_volume


@pytest.fixture(scope='module')
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _contrast(a: str, b: str) -> float:
    """WCAG contrast ratio of two '#RRGGBB' colours."""
    def lum(h):
        c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


# ── theme ──

def test_text_and_border_contrast():
    c = ui.COLORS
    for back in ('surface', 'bg', 'surface2', 'tint'):
        assert _contrast(c['text'], c[back]) >= 4.5, back
        assert _contrast(c['muted'], c[back]) >= 4.5, back      # hints are text too
    for kind in ('ok', 'warn', 'error'):
        assert _contrast(c[kind], c[kind + '_bg']) >= 4.5, kind
        assert _contrast(c[kind], c['surface']) >= 4.5, kind
    assert _contrast(c['surface'], c['accent']) >= 4.5           # text of the main button
    assert _contrast(c['accent'], c['surface']) >= 4.5           # links
    assert _contrast(c['border'], c['surface']) >= 3.0           # border of an input
    assert _contrast(c['border'], c['bg']) >= 3.0


def test_style_sheet_has_no_open_token(qapp):
    ui.ensure_theme()
    text = ui.style_sheet()
    assert '@' not in re.sub(r'@2x', '', text)
    assert qapp.property('rspace3d_theme') is True
    assert qapp.styleSheet() == text


def test_wheel_guard_keeps_the_value_of_a_field_without_focus(qapp):
    from PyQt6.QtCore import QPoint, QPointF, Qt
    from PyQt6.QtGui import QWheelEvent
    from PyQt6.QtWidgets import QLineEdit, QSpinBox, QVBoxLayout, QWidget
    box = QWidget()
    lay = QVBoxLayout(box)
    edit = QLineEdit()
    spin = QSpinBox()
    spin.setRange(0, 100)
    spin.setValue(5)
    lay.addWidget(edit)
    lay.addWidget(spin)
    ui.guard_wheel(box)
    box.show()
    box.activateWindow()
    edit.setFocus()
    qapp.processEvents()
    assert spin.focusPolicy() == Qt.FocusPolicy.StrongFocus and not spin.hasFocus()

    def wheel():
        ev = QWheelEvent(QPointF(5, 5), QPointF(spin.mapToGlobal(QPoint(5, 5))), QPoint(0, 0),
                         QPoint(0, 120), Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                         Qt.ScrollPhase.NoScrollPhase, False)
        qapp.sendEvent(spin, ev)

    wheel()
    assert spin.value() == 5                 # no focus: the wheel goes to the page
    spin.setFocus()
    qapp.processEvents()
    if spin.hasFocus():                      # with focus, the wheel changes the value
        wheel()
        assert spin.value() == 6
    box.close()


def test_banner_text_kind_and_visibility(qapp):
    banner = ui.Banner('warn')
    assert banner.isHidden() and banner.text() == ''
    banner.show_message('The cell metric does not allow m-3m (residual 0.55 < 1).', 'error')
    assert not banner.isHidden() and banner.kind() == 'error'
    assert banner.text() == 'The cell metric does not allow m-3m (residual 0.55 < 1).'
    assert 'Problem:' in banner.label.text() and '&lt;' in banner.label.text()
    banner.clear()
    assert banner.isHidden() and banner.text() == ''


# ── pages of the two processing windows ──

@pytest.fixture
def cpu_only(monkeypatch):
    from rspace3d import reconstruct_gui, volume_builder, volume_builder_gui
    for mod in (volume_builder, volume_builder_gui, reconstruct_gui):
        monkeypatch.setattr(mod, 'HAS_GPU', False)


def test_pages_fit_in_the_default_window(qapp, cpu_only):
    """No page needs a scroll bar at 1280 x 920 (the scroll area is a fallback only)."""
    from PyQt6.QtWidgets import QScrollArea
    from rspace3d.reconstruct_gui import ReconstructGUI
    from rspace3d.volume_builder_gui import SimpleVolumeGUI
    for cls in (ReconstructGUI, SimpleVolumeGUI):
        gui = cls()
        gui.resize(1280, 920)
        gui.show()
        for i, name in enumerate(gui.page_names()):
            gui.show_page(i)
            for _ in range(10):
                qapp.processEvents()
            for area in gui.page_widget(i).findChildren(QScrollArea):
                if area.widget() is None or not area.isVisibleTo(gui):
                    continue
                assert area.widget().height() <= area.viewport().height(), (cls.__name__, name)
                assert area.widget().width() <= area.viewport().width(), (cls.__name__, name)
        gui.close()


def test_log_is_open_on_the_run_page_and_closed_on_the_result_pages(qapp, cpu_only):
    from rspace3d.reconstruct_gui import ReconstructGUI
    gui = ReconstructGUI()
    gui.resize(1280, 920)
    gui.show()
    for _ in range(5):
        qapp.processEvents()
    assert gui.log.isVisibleTo(gui.activity)
    gui.show_page('Averaging')
    qapp.processEvents()
    assert not gui.log.isVisibleTo(gui.activity)
    assert gui.split.sizes()[1] < 80                      # the pages got the height of the log
    assert gui.progress.isVisibleTo(gui.activity)         # progress and status stay visible
    gui.activity.toggle_btn.click()                       # the user opens the log on this page
    gui.show_page('Reconstruct')
    gui.show_page('Averaging')
    qapp.processEvents()
    assert gui.log.isVisibleTo(gui.activity)              # the page keeps the choice
    gui.close()


def test_help_page_opens_a_topic_and_goes_back(qapp, cpu_only):
    from rspace3d.reconstruct_gui import ReconstructGUI
    gui = ReconstructGUI()
    gui.show_page('Advanced')
    gui.open_help('averaging')
    assert gui.current_page() == 'Help'
    assert gui._help_view.current_topic() == 'averaging'
    assert 'Advanced' in gui._help_view.back_btn.text() and not gui._help_view.back_btn.isHidden()
    gui._help_view.back_btn.click()
    assert gui.current_page() == 'Advanced'
    gui.close()


def test_run_note_tells_why_the_main_action_is_off(qapp, cpu_only):
    from rspace3d.reconstruct_gui import ReconstructGUI
    from rspace3d.volume_builder_gui import SimpleVolumeGUI
    gui = ReconstructGUI()
    assert not gui.run_btn.isEnabled() and 'Select a dataset folder' in gui.run_note.text()
    assert gui.browse_btn.property('role') == 'primary'    # the first action of a new window
    gui.close()
    builder = SimpleVolumeGUI()
    assert not builder.process_btn.isEnabled()
    assert 'Select an unwarp folder' in builder.run_note.text()
    builder.close()


def test_builder_layer_list_note(qapp, cpu_only):
    from rspace3d.volume_builder_gui import SimpleVolumeGUI
    gui = SimpleVolumeGUI()
    assert gui.dc_note.text().startswith('601 layers in 2 files.')     # -6 to 6, step 0.02
    gui.dc_step.setValue(0.05)
    assert gui.dc_note.text() == '241 layers in 1 file.'
    gui.dc_lmax.setValue(-7.0)                                         # last l below first l
    assert not gui.dc_btn.isEnabled() and 'larger' in gui.dc_note.text()
    gui.close()


# ── help pages ──

def test_each_help_link_goes_to_a_topic(qapp):
    keys = hc.topic_keys()
    assert len(keys) == len(set(keys)) and hc.EXPLORER in keys
    for key in keys:
        for target in re.findall(r'href="topic:([^"]+)"', hc.topic_html(key)):
            assert target in keys, (key, target)
        for fig in re.findall(r'src="fig:([^"]+)"', hc.topic_html(key)):
            from rspace3d import help_view
            assert fig in help_view._FIGURES, (key, fig)


def test_each_help_button_of_the_windows_has_a_topic(qapp, cpu_only):
    from rspace3d.reconstruct_gui import ReconstructGUI
    from rspace3d.volume_builder_gui import SimpleVolumeGUI
    keys = hc.topic_keys()
    for cls in (ReconstructGUI, SimpleVolumeGUI):
        gui = cls()
        topics = {b.topic for b in gui.findChildren(ui.HelpButton)}
        assert topics and topics <= set(keys), (cls.__name__, topics - set(keys))
        gui.close()


def test_help_uses_the_names_and_the_definitions_of_the_program(qapp):
    text = hc.topic_html('averaging')
    for name in (av.WEIGHT_LABELS['equal'], av.WEIGHT_LABELS['counts'], av.WEIGHT_LABELS['adaptive'],
                 av.REJECTION_LABELS['none'], av.REJECTION_LABELS['mad'],
                 av.REJECTION_LABELS['model'], av.COUNTS_MODEL['name'], av.EQUAL_MAD['name']):
        assert name in text, name
    from rspace3d import help_view
    assert 'sum|A - B| / sum(|A| + |B|)' in help_view.document_text('frame-tools')
    assert 'Laue group mmm and 0.2 degree frames' in help_view.document_text('frame-tools')
    assert av.LIMITS_TEXT in help_view.document_text('averaging-page')
    for header in av.COMPARE_HEADERS[1:] + av.SHELL_HEADERS[1:]:
        assert header in help_view.document_text('averaging-page'), header


def test_help_view_shows_text_figure_and_example(qapp):
    from rspace3d import help_view
    view = help_view.HelpView()
    view.show_topic('symmetry')
    assert view.stack.currentIndex() == 0 and 'orbit' in view.browser.toPlainText()
    img = help_view.render_figure('orbit')
    assert img.width() > 300 and img.height() > 100
    view.show_topic(hc.EXPLORER)
    assert view.stack.currentIndex() == 1 and view.explorer is not None
    view.show_topic('no such topic')
    assert view.current_topic() == 'start'
    view.browser.anchorClicked.emit(__import__('PyQt6.QtCore', fromlist=['QUrl']).QUrl('topic:glossary'))
    assert view.current_topic() == 'glossary'
    view.close()


# ── the example of the help follows the program ──

def _orbit_volume(case):
    """A 3 x 3 x 3 volume whose 8 corners are one orbit of the Laue group mmm."""
    axis = np.array([-0.5, 0.0, 0.5])
    data = np.full((3, 3, 3), np.nan, np.float32)
    counts = np.zeros((3, 3, 3), np.uint16)
    corners = [(i, j, k) for i in (0, 2) for j in (0, 2) for k in (0, 2)]
    for (i, j, k), x, n in zip(corners, case['values'], case['counts']):
        data[i, j, k] = x
        counts[i, j, k] = n
    meta = {'ub': np.diag([0.1, 0.11, 0.12]), 'wavelength': 0.5, 'grid_kind': 'hkl_regular'}
    vol = VolumeData(data, axis.copy(), axis.copy(), axis.copy(), 'HK', meta)
    vol.counts = counts
    return vol, corners


@pytest.mark.parametrize('weighting', oe.WEIGHTINGS)
@pytest.mark.parametrize('rejection,sigma', [('none', 3.0), ('mad', 3.0), ('mad', 2.0),
                                             ('model', 4.0), ('model', 2.0)])
def test_help_example_gives_the_values_of_symmetrize_volume(weighting, rejection, sigma):
    from rspace3d import help_view
    gain, rel, kappa, floor = help_view.EXAMPLE_NOISE
    model = NoiseModel(gain=gain, rel=rel, kappa=kappa, floor=floor)
    for case in help_view.CASES:
        res = help_view.explore(case, weighting, rejection, sigma)
        vol, corners = _orbit_volume(case)
        grad = np.full(vol.intensity.shape, case['grad'], np.float32)
        out = symmetrize_volume(vol, 'mmm', sigma=None if rejection == 'none' else sigma,
                                weighting=weighting, rejection=rejection, noise_model=model,
                                grad=grad, use_gpu=False, propagate_variance=False)
        # each voxel of the orbit gets the value of the example (float32 sums in another order)
        for c in corners:
            assert float(out.intensity[c]) == pytest.approx(res['value'], rel=1e-5), case['name']
        # each rejected member is one voxel that the symmetrisation counts as an outlier
        n_out = int(len(case['values']) - res['kept'].sum())
        assert out.metadata.get('n_outliers_removed', 0) == n_out, case['name']


def test_help_example_shows_what_the_text_says():
    """The cases of the example behave as their texts say."""
    from rspace3d import help_view
    clean, spike, frame, peak = help_view.CASES

    def rejected(case, rejection, sigma):
        return int((~help_view.explore(case, 'counts', rejection, sigma)['kept']).sum())

    assert rejected(clean, 'mad', 3.0) == 0 and rejected(clean, 'model', 4.0) == 0
    assert rejected(spike, 'none', 3.0) == 0
    assert rejected(spike, 'mad', 3.0) == 1 and rejected(spike, 'model', 4.0) == 1
    assert rejected(frame, 'mad', 3.0) == 0          # the spread limit keeps the extra intensity
    assert rejected(frame, 'model', 4.0) == 1        # the noise limit removes it
    assert rejected(peak, 'model', 4.0) == 0         # the gradient term keeps the peak members
    assert rejected(peak, 'model', 1.0) > 0          # a low threshold removes good members
    res = help_view.explore(frame, 'counts', 'mad', 3.0)
    assert res['floor_used'] and abs(res['value'] - frame['true']) > 2.0
    res = help_view.explore(frame, 'counts', 'model', 4.0)
    assert abs(res['value'] - frame['true']) < 1.0
