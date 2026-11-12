"""Tests of the 3D view: the geometry and the camera, the statistics for the levels, the ray
march on the CPU, the controls of the window, and the shader against the CPU ray march.

The window tests run offscreen, where there is no OpenGL: the window then shows a message in
place of the image. The shader tests start `gl_selftest.py` in a process with the normal
display. They are skipped on a computer without OpenGL 4.1.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from rspace3d import view3d as v3
from rspace3d.volume_builder import VolumeData, volume_affine

from gl_selftest import make_volume

HERE = os.path.dirname(os.path.abspath(__file__))
CELL = {'a': 6.0, 'b': 7.0, 'c': 9.0, 'alpha': 82.0, 'beta': 104.0, 'gamma': 95.0}


@pytest.fixture(scope='module')
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope='module')
def vol():
    return make_volume()


@pytest.fixture
def window(qapp, vol):
    from rspace3d.view3d_window import View3DWindow
    win = View3DWindow(vol, title='test.h5')
    win.show()
    qapp.processEvents()
    yield win
    win.close()


# ---------------------------------------------------------------- geometry

def test_b_matrix_has_the_metric_of_the_cell():
    a, b, c = CELL['a'], CELL['b'], CELL['c']
    al, be, ga = np.radians([CELL['alpha'], CELL['beta'], CELL['gamma']])
    g = np.array([[a * a, a * b * np.cos(ga), a * c * np.cos(be)],
                  [a * b * np.cos(ga), b * b, b * c * np.cos(al)],
                  [a * c * np.cos(be), b * c * np.cos(al), c * c]])
    bm = v3.b_matrix(CELL)
    assert np.allclose(bm.T @ bm, np.linalg.inv(g), rtol=1e-12, atol=1e-15)
    assert bm[1, 0] == 0 and bm[2, 0] == 0 and bm[2, 1] == 0        # a* along x, b* in the xy plane
    assert np.array_equal(v3.b_matrix(None), np.eye(3))
    assert np.array_equal(v3.b_matrix({'a': 1, 'b': 1, 'c': 1, 'alpha': 60, 'beta': 60, 'gamma': 150}),
                          np.eye(3))                                 # not a possible cell


def test_grid_puts_each_voxel_at_its_miller_indices(vol):
    grid = v3.make_grid(vol)
    origin, a = volume_affine(vol)
    assert a[0, 1] != 0                                              # the raster of the test volume has a shear
    nh, nk, nl = vol.intensity.shape
    rng = np.random.default_rng(0)
    idx = np.stack([rng.integers(0, n, 50) for n in (nh, nk, nl)], axis=1)
    tex = np.stack([(idx[:, 2] + 0.5) / nl, (idx[:, 1] + 0.5) / nk, (idx[:, 0] + 0.5) / nh], axis=1)
    hkl = origin + idx @ a.T
    assert np.allclose(grid.hkl(tex), hkl, atol=1e-12)
    assert np.allclose(grid.world(tex), hkl @ v3.b_matrix(CELL).T, atol=1e-12)
    assert np.allclose(grid.tex(grid.world(tex)), tex, atol=1e-12)
    assert grid.factors == (1, 1, 1) and grid.has_cell


def test_box_ranges_keep_the_voxels_at_both_ends(vol):
    grid = v3.make_grid(vol)
    nh = vol.intensity.shape[0]
    assert grid.tex_range(0, float(vol.H[0]), float(vol.H[-1])) == pytest.approx((0.0, 1.0))
    lo, hi = grid.tex_range(0, float(vol.H[5]), float(vol.H[9]))
    assert lo == pytest.approx(5 / nh) and hi == pytest.approx(10 / nh)
    lo, hi = grid.tex_range(0, float(vol.H[9]), float(vol.H[5]))      # the order does not matter
    assert lo == pytest.approx(5 / nh) and hi == pytest.approx(10 / nh)
    lo, hi = grid.tex_range(0, 100.0, 200.0)                         # outside: the last voxel
    assert hi == pytest.approx(1.0) and hi - lo == pytest.approx(1 / nh)
    box_lo, box_hi = grid.box((float(vol.H[0]), float(vol.H[-1])), (float(vol.K[2]), float(vol.K[2])),
                              (float(vol.L[0]), float(vol.L[-1])))
    assert box_lo[1] == pytest.approx(2 / vol.intensity.shape[1])    # texture order is (l, k, h)
    assert box_hi[1] == pytest.approx(3 / vol.intensity.shape[1])


def test_large_volume_gets_factors_per_axis():
    assert v3.reduce_factors((480, 480, 480)) == (1, 1, 1)
    assert v3.reduce_factors((2162, 2068, 61)) == (2, 2, 1)          # only the long axes
    f = v3.reduce_factors((1500, 1500, 1000))
    assert np.prod([n // k for n, k in zip((1500, 1500, 1000), f)]) <= v3.MAX_TEXTURE_VOXELS
    assert v3.reduce_factors((100, 100, 100), max_voxels=130_000) == (2, 2, 2)
    data = np.arange(4 * 6 * 2, dtype=np.float32).reshape(4, 6, 2)
    data[0, 0, 0] = np.nan
    data[2:, :2, :] = np.nan
    small = v3.reduce_volume(data, (2, 2, 1))
    assert small.shape == (2, 3, 2)
    assert small[0, 0, 0] == pytest.approx(np.mean([data[0, 1, 0], data[1, 0, 0], data[1, 1, 0]]))
    assert np.isnan(small[1, 0, 0]) and small[1, 2, 1] == pytest.approx(data[2:, 4:, 1].mean())
    vol = VolumeData(np.ones((10, 12, 9), np.float32), np.arange(10.0), np.arange(12.0), np.arange(9.0),
                     'HK', {'grid_kind': 'hkl_regular'})
    grid = v3.make_grid(vol, max_voxels=300)
    assert grid.factors == (2, 2, 1) and grid.shape == (5, 6, 9) and grid.span == (10, 12, 9)
    # the centre of the first block of the texture is between the voxels 0 and 1 of the volume
    assert grid.hkl([(0.5 / 9, 0.5 / 6, 0.5 / 5)])[0] == pytest.approx([0.5, 0.5, 0.0])


def test_block_max_is_above_each_sample_of_its_block():
    rng = np.random.default_rng(3)
    data = rng.lognormal(2.0, 1.5, (30, 37, 41)).astype(np.float32)
    data[rng.random(data.shape) < 0.05] = np.nan
    bm = v3.block_max(data)
    assert bm.shape == (4, 5, 6)
    tex = rng.random((4000, 3))
    vals = v3.trilinear(data, tex)
    n = np.array(data.shape)
    block = np.minimum((tex[:, ::-1] * n / v3.BLOCK).astype(int), np.array(bm.shape) - 1)
    assert np.all(vals <= bm[block[:, 0], block[:, 1], block[:, 2]] * (1 + 1e-6))


# ---------------------------------------------------------------- camera

def test_camera_basis_and_projection(vol):
    grid = v3.make_grid(vol)
    cam = v3.Camera(yaw=0.7, pitch=-0.4, zoom=1.7, target=np.array([0.01, -0.02, 0.03]))
    for ortho in (True, False):
        cam.ortho = ortho
        fwd, right, up = cam.basis()
        m = np.stack([fwd, right, up])
        assert np.allclose(m @ m.T, np.eye(3), atol=1e-12)
        pts = grid.world(np.random.default_rng(1).random((20, 3)))
        px = cam.project(grid, pts, 800, 500)
        for p, (x, y) in zip(pts, px):                                # the ray of the pixel goes through the point
            o, d = cam.ray(grid, x, y, 800, 500)
            assert np.linalg.norm(np.cross(p - o, d)) < 1e-9 * (1 + np.linalg.norm(p - o))


@pytest.mark.parametrize('view, first, second', [('hk', 0, 1), ('hl', 0, 2), ('kl', 1, 2)])
def test_plane_views_have_the_first_index_to_the_right(vol, view, first, second):
    grid = v3.make_grid(vol)
    cam = v3.Camera(yaw=1.0, pitch=0.3, zoom=2.0, target=np.ones(3))
    cam.look(grid, view)
    assert cam.zoom == 1.0 and np.array_equal(cam.target, np.zeros(3))
    c = grid.center
    o, a, b = cam.project(grid, [c, c + grid.b[:, first], c + grid.b[:, second]], 800, 600)
    assert a[0] > o[0] and abs(a[1] - o[1]) < 1e-3 * (a[0] - o[0])   # first index: to the right
    assert b[1] < o[1]                                               # second index: up
    third = 3 - first - second
    fwd = cam.basis()[0]
    assert abs(fwd @ grid.b[:, first]) < 1e-3 * np.linalg.norm(grid.b[:, first])
    assert abs(fwd @ grid.b[:, second]) < 1e-3 * np.linalg.norm(grid.b[:, second])
    assert abs(fwd @ grid.b[:, third]) > 0.5 * np.linalg.norm(grid.b[:, third])


def test_zoom_keeps_the_point_below_the_mouse(vol):
    grid = v3.make_grid(vol)
    cam = v3.Camera()
    o, d = cam.ray(grid, 520.0, 130.0, 800, 600)
    point = o + 3.0 * grid.radius * d
    cam.zoom_at(grid, 1.8, 520.0, 130.0, 800, 600)
    assert cam.zoom == pytest.approx(1.8)
    assert cam.project(grid, [point], 800, 600)[0] == pytest.approx([520.0, 130.0], abs=1e-6)
    cam.pan(grid, 30.0, -12.0, 600)
    assert cam.project(grid, [point], 800, 600)[0] == pytest.approx([550.0, 118.0], abs=1e-6)


# ---------------------------------------------------------------- levels

def test_default_level_has_one_percent_of_the_voxels_above():
    rng = np.random.default_rng(5)
    data = rng.lognormal(2.0, 1.0, (60, 60, 60)).astype(np.float32)
    data[:5] = np.nan
    data[5:8] = 0.0
    sample = v3.sample_values(data)
    assert sample.size == np.count_nonzero(np.isfinite(data) & (data > 0))
    level = v3.default_scene(sample).levels[0].value
    assert np.mean(data[np.isfinite(data) & (data > 0)] > level) == pytest.approx(0.01, abs=0.001)
    # counts in steps: the voxels that are equal to the percentile are not inside the surface
    steps = np.round(rng.lognormal(0.0, 0.6, (50, 50, 50)) * 4) / 4
    level = v3.default_scene(v3.sample_values(steps)).levels[0].value
    assert np.mean(steps[steps > 0] >= level) <= 0.01
    # a large volume: a random part of it, and nearly the same level
    big = rng.lognormal(2.0, 1.0, (160, 160, 160)).astype(np.float32)
    part = v3.sample_values(big, target=200_000)
    assert 150_000 < part.size <= 200_000 and np.all(np.diff(part) >= 0)
    assert v3.percentile(part, 99) == pytest.approx(np.percentile(big, 99), rel=0.05)


def test_new_level_is_not_near_an_old_level():
    sample = np.sort(np.random.default_rng(7).lognormal(2.0, 1.2, 100_000))
    first = v3.default_scene(sample).levels[0].value
    second = v3.new_level(sample, [first])
    third = v3.new_level(sample, [first, second])
    assert second == pytest.approx(v3.percentile(sample, 90)) and second < first
    assert third == pytest.approx(v3.percentile(sample, 99.9)) and third > first
    for a, b in ((first, second), (first, third), (second, third)):
        assert max(a, b) / min(a, b) > 1.5
    assert v3.new_level(np.sort(np.full(100, 5.0)), [5.0]) == pytest.approx(2.5)


def test_histogram_is_on_a_logarithmic_axis():
    sample = np.sort(np.random.default_rng(2).lognormal(1.0, 1.0, 50_000))
    edges, counts = v3.histogram(sample, bins=40)
    assert len(edges) == 41 and counts.sum() > 0.99 * sample.size
    assert edges[-1] == pytest.approx(math.log10(sample[-1]))
    edges, counts = v3.histogram(np.array([]))
    assert counts.sum() == 0


# ---------------------------------------------------------------- ray march on the CPU

def _ball(n=48, cell=None, shear=0.0):
    """A volume whose surfaces of equal intensity are spheres around hkl = 0 (cubic cell)."""
    ax = np.linspace(-2.0, 2.0, n)
    h, k, l = np.meshgrid(ax, ax, ax, indexing='ij')
    data = (1000.0 * np.exp(-(h * h + k * k + l * l) / (2 * 0.5 ** 2))).astype(np.float32)
    meta = {'grid_kind': 'hkl_regular', 'cell': cell or {'a': 5.0, 'b': 5.0, 'c': 5.0, 'alpha': 90.0,
                                                          'beta': 90.0, 'gamma': 90.0}}
    return VolumeData(data, ax.copy(), ax.copy(), ax.copy(), 'HK', meta)


def test_first_hit_is_on_the_sphere():
    ball = _ball()
    grid = v3.make_grid(ball)
    level = 100.0
    radius = 0.5 * math.sqrt(2.0 * math.log(1000.0 / level))        # in r.l.u.
    for cam in (v3.Camera(), v3.Camera(yaw=2.0, pitch=-0.9, ortho=False)):
        o, d = cam.ray(grid, 400.0, 300.0, 800, 600)                 # through the centre of the box
        hit = v3.first_hit(ball.intensity, grid, o, d, level, ortho=cam.ortho)
        assert hit is not None and not hit['cap']
        assert np.linalg.norm(hit['hkl']) == pytest.approx(radius, abs=0.02)     # voxel: 0.085
        assert hit['value'] == pytest.approx(level, rel=0.05)
        peak = v3.ray_max(ball.intensity, grid, o, d, ortho=cam.ortho)
        assert peak['value'] == pytest.approx(1000.0, rel=0.02)
        assert np.linalg.norm(peak['hkl']) < 0.1
    o, d = v3.Camera().ray(grid, 5.0, 5.0, 800, 600)                 # a corner of the image: no box
    assert v3.first_hit(ball.intensity, grid, o, d, level) is None
    # a box that cuts the sphere: the ray is above the level where it enters
    cam = v3.Camera()
    cam.look(grid, 'hk')
    o, d = cam.ray(grid, 400.0, 300.0, 800, 600)
    lo, hi = grid.box((-2.0, 2.0), (-2.0, 2.0), (-2.0, 0.0))
    hit = v3.first_hit(ball.intensity, grid, o, d, level, lo, hi)
    assert hit['cap'] and hit['hkl'][2] == pytest.approx(0.0, abs=0.05) and hit['value'] > 900


def test_unmeasured_voxels_are_below_each_level():
    ball = _ball()
    ball.intensity[:, :, 30:] = np.nan                               # no data above l = 0.55
    grid = v3.make_grid(ball)
    cam = v3.Camera()
    cam.look(grid, 'hk')                                             # look down along l
    o, d = cam.ray(grid, 400.0, 300.0, 800, 600)
    hit = v3.first_hit(ball.intensity, grid, o, d, 100.0)
    assert hit is not None and hit['hkl'][2] == pytest.approx(float(ball.L[29]), abs=0.09)
    assert v3.measured(np.array([1.0, 0.0, np.nan, -2.0, np.inf])).tolist() == [True, False, False, True, False]


# ---------------------------------------------------------------- labels

def test_labels_are_along_the_edges_of_the_box(vol):
    grid = v3.make_grid(vol)
    cam = v3.Camera()
    layout = v3.axis_layout(grid, cam, (0, 0, 0), (1, 1, 1), 900, 700)
    assert [e['axis'] for e in layout] == ['h', 'k', 'l']
    for entry in layout:
        values = [float(t[2]) for t in entry['ticks']]
        assert len(values) >= 3 and values == sorted(values)
        steps = np.diff(values)
        assert np.allclose(steps, entry['step']) and entry['step'] in (0.5, 1.0, 2.0)
        assert entry['name'] is not None
        for x, y, _ in entry['ticks']:
            assert -40 < x < 940 and -40 < y < 740
    assert v3.grid_steps(layout) == (1.0, 1.0, 1.0)
    cam.look(grid, 'hk')
    layout = v3.axis_layout(grid, cam, (0, 0, 0), (1, 1, 1), 900, 700)
    assert len(layout[0]['ticks']) >= 3 and len(layout[1]['ticks']) >= 3
    cubic = v3.make_grid(_ball())                                    # c* along the line of sight:
    cam.look(cubic, 'hk')                                            # the l edges are points
    layout = v3.axis_layout(cubic, cam, (0, 0, 0), (1, 1, 1), 900, 700)
    assert layout[2]['ticks'] == [] and layout[2]['name'] is None
    assert [t[2] for t in layout[0]['ticks']] == ['-2.0', '-1.5', '-1.0', '-0.5', '0', '0.5', '1.0',
                                                  '1.5', '2.0']
    assert v3.nice_step(0.004, 46.0) == pytest.approx(0.2)
    assert v3.nice_step(0.3, 12.0) == pytest.approx(5.0)
    assert len(v3.triad(grid, cam)) == 3


def test_colour_map_has_its_weak_end_at_the_background():
    from rspace3d.view3d_gl import colormap_for
    assert colormap_for(v3.Scene(colormap='inferno', dark=True)) == 'inferno'
    assert colormap_for(v3.Scene(colormap='inferno', dark=False)) == 'inferno_r'
    assert colormap_for(v3.Scene(colormap='gray_r', dark=False)) == 'gray_r'
    assert colormap_for(v3.Scene(colormap='gray_r', dark=True)) == 'gray'


# ---------------------------------------------------------------- window (no OpenGL offscreen)

def test_window_without_opengl_says_why(window):
    assert window.canvas is None
    assert 'cannot draw' in window.problem.text() and 'OpenGL' in window.problem.text()
    assert window.problem.kind() == 'error' and window.stack.currentIndex() == 0
    assert not window.save_btn.isEnabled() and not window.copy_btn.isEnabled()
    assert window.image() is None
    assert window.windowTitle() == 'rspace3d 3D view: test.h5'
    assert '40 x 48 x 56 voxels' in window.facts.text()


def test_levels_of_the_window(window, qapp):
    scene = window.scene
    assert scene.mode == 'surface' and len(scene.levels) == 1 and scene.levels[0].opacity == 1.0
    first = scene.levels[0].value
    assert first == pytest.approx(v3.percentile(window.sample, 99), rel=1e-3)
    assert not window._level_rows[0]['remove'].isEnabled()           # the last level stays
    window._add_level()
    window._add_level()
    assert len(scene.levels) == 3 and not window.add_btn.isEnabled()
    values = [lv.value for lv in scene.levels]
    assert len({lv.color for lv in scene.levels}) == 3
    assert [lv.opacity for lv in scene.levels] == [1.0 if v == max(values) else 0.35 for v in values][:3] \
        or scene.levels[0].opacity == 1.0
    window._level_rows[1]['value'].setValue(123.0)
    assert scene.levels[1].value == pytest.approx(123.0)
    window._level_rows[1]['opacity'].setValue(60)
    assert scene.levels[1].opacity == pytest.approx(0.6)
    window._bar_moved(2, 77.0)                                       # a drag on the histogram
    assert scene.levels[2].value == pytest.approx(77.0)
    assert window._level_rows[2]['value'].value() == pytest.approx(77.0)
    window._remove_level(1)
    assert [lv.value for lv in scene.levels] == pytest.approx([first, 77.0])
    window._reset_levels()
    assert len(scene.levels) == 1 and scene.levels[0].value == pytest.approx(first)
    assert [lv.value for lv in scene.sorted_levels()] == sorted(lv.value for lv in scene.levels)


def test_projection_mode_of_the_window(window, qapp):
    window.show()
    window.projection_radio.setChecked(True)
    qapp.processEvents()
    assert window.scene.mode == 'projection'
    assert window.scale_box.isVisible() and not window.levels_box.isVisible()
    assert window.level_title.text() == 'Colour scale'
    lo, hi = window.scene.clim
    assert lo < hi and window.min_spin.value() == pytest.approx(lo, rel=1e-3)
    window.max_spin.setValue(5000.0)
    assert window.scene.clim[1] == pytest.approx(5000.0)
    window._bar_moved(0, 9000.0)                                     # Min cannot go above Max
    assert window.scene.clim[0] < window.scene.clim[1]
    window.cmap_combo.setCurrentText('viridis')
    assert window.scene.colormap == 'viridis'
    window.surface_radio.setChecked(True)
    assert window.scene.mode == 'surface' and window.levels_box.isVisible()


def test_box_and_view_of_the_window(window, vol):
    grid = window.grid
    assert window.scene.lo == pytest.approx((0, 0, 0)) and window.scene.hi == pytest.approx((1, 1, 1))
    window.box_spins[2][0].setValue(0.0)                             # l from 0
    lo, hi = grid.box(*window.box_ranges())
    assert window.scene.lo == pytest.approx(lo) and window.scene.hi == pytest.approx(hi)
    assert 0.4 < window.scene.lo[0] < 0.6 and window.scene.hi[0] == pytest.approx(1.0)
    window._set_box(None)
    assert window.scene.lo == pytest.approx((0, 0, 0))
    window._set_box(((-1.0, 1.0), None, None))
    assert window.box_ranges()[0] == pytest.approx((-1.0, 1.0))
    assert window.box_ranges()[1] == pytest.approx((float(vol.K[0]), float(vol.K[-1])))
    window.persp_check.setChecked(True)
    window.grid_check.setChecked(False)
    window.dark_check.setChecked(True)
    assert not window.camera.ortho and not window.scene.grid and window.scene.dark
    window._look('hk')
    assert window.camera.pitch == pytest.approx(math.pi / 2, abs=0.2)


def test_panel_of_the_window_needs_no_scroll_bar(window, qapp):
    window._add_level()
    window._add_level()
    window.resize(1180, 850)                                         # the start size of the window
    window.show()
    qapp.processEvents()
    view = window.panel_scroll.viewport()
    assert window.panel_scroll.verticalScrollBar().maximum() == 0
    assert window.panel_body.minimumSizeHint().width() <= view.width() - 12   # also with a scroll bar
    hover = {'hkl': np.array([1.0, -0.5, 0.25]), 'value': 1234.5, 'cap': True}
    window._hover(hover)
    assert 'h = 1.000' in window.readout.text() and 'cut by the box' in window.readout.text()
    window._hover(None)
    assert 'Drag' in window.readout.text()


def test_start_values_of_the_window(qapp, vol):
    from rspace3d.view3d_window import View3DWindow
    win = View3DWindow(vol, levels=[50.0, 800.0, 5.0, 1.0], box=((-1.0, 1.0), None, (0.0, 1.0)))
    try:
        assert [lv.value for lv in win.scene.levels] == [800.0, 50.0, 5.0]     # at most three, strongest first
        assert [lv.opacity for lv in win.scene.levels] == [1.0, 0.35, 0.35]
        assert win.box_ranges()[0] == pytest.approx((-1.0, 1.0))
        assert win.box_ranges()[2] == pytest.approx((0.0, 1.0), abs=0.03)
    finally:
        win.close()


def test_viewer_opens_the_3d_window_with_its_ranges(qapp):
    from rspace3d.rsp_viewer import UnifiedViewer
    example = os.path.join(os.path.dirname(HERE), 'example_data', 'example_monoclinic.h5')
    viewer = UnifiedViewer()
    try:
        viewer._load_file(example)
        qapp.processEvents()
        viewer.xrange_spin.setValue(3.0)
        viewer.yrange_spin.setValue(2.0)
        assert viewer._box_3d() == ((-3.0, 3.0), (-2.0, 2.0), None)
        viewer._show_3d()
        qapp.processEvents()
        assert len(viewer._views3d) == 1
        win = viewer._views3d[0]
        assert win.windowTitle().endswith('example_monoclinic.h5')
        assert win.box_ranges()[0] == pytest.approx((-3.0, 3.0))
        assert win.box_ranges()[1] == pytest.approx((-2.0, 2.0))
        assert win.box_ranges()[2] == pytest.approx((float(viewer.vol.L[0]), float(viewer.vol.L[-1])))
        win.close()
    finally:
        viewer.close()


def test_plot_isosurface_opens_the_3d_view(qapp, vol, monkeypatch):
    from rspace3d import view3d_window, volume_isosurface
    seen = {}
    monkeypatch.setattr(view3d_window, 'show_volume',
                        lambda v, levels=None, box=None, title='': seen.update(
                            vol=v, levels=levels, box=box, title=title))
    volume_isosurface.plot_isosurface(vol, isovalue=[5.0, 50.0], h_range=(-1.0, 1.0))
    assert seen['vol'] is vol and seen['levels'] == [5.0, 50.0]
    assert seen['box'] == ((-1.0, 1.0), None, None)
    volume_isosurface.plot_isosurface(vol)
    assert seen['levels'] is None


def test_help_has_the_3d_topic():
    from rspace3d import help_content as hc
    assert 'view3d' in hc.topic_keys()
    text = hc.topic_html('view3d')
    for word in ('Surfaces of equal intensity', 'Projection of the maximum', 'Grey parts', 'OpenGL 4.1'):
        assert word in text
    assert 'topic:view3d' in hc.topic_html('viewer') and 'topic:view3d' in hc.topic_html('start')


# ---------------------------------------------------------------- more cases

def test_cut_face_has_the_values_of_the_voxels_in_the_box():
    rng = np.random.default_rng(11)
    data = (rng.random((12, 14, 16)) * 100 + 1).astype(np.float32)
    vol = VolumeData(data, np.arange(12.0), np.arange(14.0), np.arange(16.0), 'HK',
                     {'grid_kind': 'hkl_regular'})
    grid = v3.make_grid(vol)
    lo, hi = grid.box((0.0, 11.0), (0.0, 13.0), (5.0, 5.0))          # one plane of voxels: l = 5
    assert hi[0] - lo[0] == pytest.approx(1 / 16)
    cam = v3.Camera()
    cam.look(grid, 'hk')
    for x, y in ((400.0, 300.0), (333.0, 251.0), (470.0, 362.0)):
        o, d = cam.ray(grid, x, y, 800, 600)
        peak = v3.ray_max(data, grid, o, d, lo, hi)
        h, k, _ = peak['hkl']
        plane = v3.trilinear(data[:, :, 5:6], [[0.5, (k + 0.5) / 14, (h + 0.5) / 12]])[0]
        assert peak['value'] == pytest.approx(plane, rel=1e-9)       # no part of the planes 4 and 6
        hit = v3.first_hit(data, grid, o, d, 0.5, lo, hi)            # the level is below all data: a cap
        assert hit['cap'] and hit['value'] == pytest.approx(plane, rel=1e-9)
    data[:] = -3.0                                                   # the picture has no value below zero
    o, d = cam.ray(grid, 400.0, 300.0, 800, 600)
    assert v3.ray_max(data, grid, o, d)['value'] == 0.0


def test_default_level_is_not_above_all_voxels():
    assert v3.default_scene(np.sort(np.full(1000, 5.0))).levels[0].value == 5.0
    steps = np.sort(np.array([1.0] * 95 + [2.0] * 5))
    level = v3.default_scene(steps).levels[0].value
    assert level == 2.0 and np.mean(steps >= level) == pytest.approx(0.05)
    assert v3.default_scene(np.array([])).levels[0].value == 1.0
    bad = dict(CELL, a=float('nan'))
    assert np.array_equal(v3.b_matrix(bad), np.eye(3))
    assert np.array_equal(v3.b_matrix(dict(CELL, gamma=float('inf'))), np.eye(3))


def test_numbers_are_at_their_points_in_perspective(vol):
    grid = v3.make_grid(vol)
    cam = v3.Camera(yaw=-0.96, pitch=0.38, ortho=False)
    layout = v3.axis_layout(grid, cam, (0, 0, 0), (1, 1, 1), 900, 700)
    checked = 0
    for ax, entry in enumerate(layout):
        if len(entry['ticks']) < 3:
            continue
        ends_tex = None
        # find the edge of the box whose ends project to the edge of the entry
        for corner in np.array(np.meshgrid([0, 1], [0, 1], [0, 1])).T.reshape(-1, 3).astype(float):
            ta = (2, 1, 0)[ax]
            if corner[ta] != 0:
                continue
            other = corner.copy()
            other[ta] = 1.0
            px = cam.project(grid, grid.world([corner, other]), 900, 700)
            if np.allclose(px, entry['edge'], atol=1e-6):
                ends_tex = (corner, other)
        assert ends_tex is not None
        m0, m1 = grid.hkl(ends_tex[0])[0, ax], grid.hkl(ends_tex[1])[0, ax]
        places = []
        for x, y, text in entry['ticks']:
            point = ends_tex[0] + (float(text) - m0) / (m1 - m0) * (ends_tex[1] - ends_tex[0])
            places.append(cam.project(grid, grid.world([point]), 900, 700)[0])
        got = np.array([(x, y) for x, y, _ in entry['ticks']])
        # the same shift (14 pixels away from the edge) for each number of the axis
        assert np.allclose(got - got[0], np.array(places) - places[0], atol=1e-6)
        assert np.linalg.norm(got[0] - places[0]) == pytest.approx(14.0, abs=1e-6)
        checked += 1
    assert checked >= 2


def test_grid_lines_are_where_the_numbers_are():
    def entry(step, px):
        return {'step': step, 'px_per_unit': px}
    assert v3.grid_steps([entry(5.0, 14.0), entry(2.0, 33.0), entry(0.5, 120.0)]) == (5.0, 1.0, 1.0)
    assert v3.grid_steps([entry(10.0, 6.0), entry(5.0, 25.0), entry(1.0, 3.0)]) == (5.0, 1.0, 1.0)
    assert v3.grid_steps([entry(10.0, 12.0), entry(2.0, 15.0), entry(20.0, 2.0)]) == (2.0, 2.0, 10.0)


def test_intensity_field_keeps_a_typed_number(qapp):
    from rspace3d.view3d_window import NumberBox, intensity_text
    assert [intensity_text(v) for v in (1451.33, 40.81, 0.04, 20982.0, 1.0001, 1.0e9, 1.234e-5, 0.0)] == \
        ['1451', '40.81', '0.04000', '20982', '1.000', '1.000e+09', '1.234e-05', '0']
    box = NumberBox()
    box.setValue(1150.05)
    assert box.text() == '1150' and box.value() == pytest.approx(1150.05)
    seen = []
    box.valueChanged.connect(seen.append)
    for typed, value in (('0.04', 0.04), ('12.7', 12.7), ('2e3', 2000.0), ('0,5', 0.5)):
        box.lineEdit().setText(typed)
        box.interpretText()                                          # as the Return key
        assert box.value() == pytest.approx(value)
    assert seen == pytest.approx([0.04, 12.7, 2000.0, 0.5])
    for typed in ('0', '-3', 'abc'):                                 # not a level: the value stays
        box.lineEdit().setText(typed)
        box.interpretText()
        assert box.value() == pytest.approx(0.5)
    box.stepUp()
    assert box.value() > 0.5


def test_levels_from_python_are_above_zero(qapp, vol):
    from rspace3d.view3d_window import View3DWindow
    win = View3DWindow(vol, levels=[-5.0, 0.0, float('nan'), 30.0])
    try:
        assert [lv.value for lv in win.scene.levels] == [30.0]
        assert win._level_rows[0]['value'].value() == pytest.approx(30.0)
        win._level_rows[0]['value'].lineEdit().setText('0.003')
        win._level_rows[0]['value'].interpretText()
        assert win.scene.levels[0].value == pytest.approx(0.003)     # not rounded to the old digits
    finally:
        win.close()
    win = View3DWindow(vol, levels=[0.0, -1.0])
    try:
        assert len(win.scene.levels) == 1 and win.scene.levels[0].value > 0
    finally:
        win.close()


def test_shift_drag_makes_small_steps_from_the_present_level(qapp):
    from PyQt6.QtCore import QPointF, Qt
    from PyQt6.QtGui import QColor
    from rspace3d.view3d_window import LevelBar

    class Event:
        def __init__(self, x, shift=True):
            self.x = x
            self.shift = shift

        def position(self):
            return QPointF(self.x, 20.0)

        def modifiers(self):
            return Qt.KeyboardModifier.ShiftModifier if self.shift else Qt.KeyboardModifier.NoModifier

    bar = LevelBar()
    bar.resize(240, 88)
    bar.set_histogram(np.linspace(0.0, 4.0, 41), np.ones(40))        # 1 to 1e4
    moves = []
    bar.moved.connect(lambda i, v: moves.append(v))
    for start in (1.0e15, 1.0e-3, 50.0):                             # two of them are outside the histogram
        bar.set_marks([(start, QColor('#000000'))])
        bar._drag, bar._last_x = 0, 100.0
        bar._move(Event(100.0))                                      # Shift + click: no change
        assert moves[-1] == pytest.approx(start)
        bar._drag, bar._last_x = 0, 100.0
        bar._move(Event(110.0))                                      # 10 pixels: one tenth of a normal move
        per_px = 4.0 / bar._plot().width()
        assert moves[-1] == pytest.approx(start * 10 ** (0.1 * 10 * per_px))
    bar._drag, bar._last_x = 0, 100.0
    bar._move(Event(bar._plot().left() + bar._plot().width() / 2, shift=False))
    assert moves[-1] == pytest.approx(100.0)                         # the middle of 1 to 1e4


def test_plot_isosurface_keeps_the_html_export(qapp, vol, monkeypatch):
    from rspace3d import view3d_window, volume_isosurface
    calls = []
    monkeypatch.setattr(view3d_window, 'show_volume', lambda *a, **k: calls.append('gl'))
    monkeypatch.setattr(volume_isosurface, '_plot_plotly',
                        lambda data, H, K, L, iso, cmap, opacity, title, save: calls.append(('plotly', save, opacity)))
    volume_isosurface.plot_isosurface(vol, isovalue=50.0, save_html='out.html')      # as before 2.10
    assert calls == [('plotly', 'out.html', 0.6)]
    volume_isosurface.plot_isosurface(vol, isovalue=50.0)
    assert calls[-1] == 'gl'
    volume_isosurface.plot_isosurface(vol, isovalue=50.0, method='gl', save_html='out.html')
    assert calls[-1] == 'gl'
    with pytest.raises(ValueError):
        volume_isosurface.plot_isosurface(vol, method='vtk')


def test_show_volume_two_times_in_a_script(tmp_path):
    script = tmp_path / 'two_views.py'
    script.write_text('''
import sys
sys.path.insert(0, sys.argv[1])
from PyQt6.QtCore import QTimer
from gl_selftest import make_volume
from rspace3d import view3d_window as w

show = w.View3DWindow.show
def show_and_close(self):
    show(self)
    QTimer.singleShot(150, self.close)
w.View3DWindow.show = show_and_close
vol = make_volume()
first = w.show_volume(vol)
second = w.show_volume(vol, levels=[100.0])
print('RESULT', first, second, len(w._WINDOWS))
''')
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    run = subprocess.run([sys.executable, str(script), HERE], capture_output=True, text=True, env=env, timeout=300)
    assert run.returncode == 0, run.stdout + run.stderr
    assert 'RESULT None None 0' in run.stdout


# ---------------------------------------------------------------- shader (real display)

@pytest.fixture(scope='module')
def gl_report(tmp_path_factory):
    out = tmp_path_factory.mktemp('gl') / 'selftest.json'
    env = {k: v for k, v in os.environ.items() if k != 'QT_QPA_PLATFORM'}
    run = subprocess.run([sys.executable, os.path.join(HERE, 'gl_selftest.py'), str(out)],
                         capture_output=True, text=True, env=env, timeout=600)
    if run.returncode == 77 or run.returncode < 0 or run.returncode > 255:
        pytest.skip('no OpenGL 4.1 on this computer: ' + (run.stdout + run.stderr).strip()[-200:])
    assert run.returncode == 0, run.stdout + run.stderr
    return json.loads(out.read_text())


def test_shader_finds_the_surface_where_the_cpu_finds_it(gl_report):
    worst = gl_report['worst']
    # distance of the first hit along the ray, in voxels, on a triclinic cell with a sheared raster
    assert worst['hit_noskip'] < 0.15 and worst['hit_skip'] < 0.15
    assert worst['mismatch_noskip'] <= 0.005 and worst['mismatch_skip'] <= 0.005
    cases = [c for c in gl_report['cases'] if 'level' in c]
    assert len(cases) == 12 and all(c['hits'] > 20 for c in cases)
    assert np.median([c['median_error_voxels'] for c in cases]) < 0.01


def test_shader_projection_has_the_largest_value(gl_report):
    assert gl_report['worst']['max'] < 0.01                          # in units of the colour scale
    assert all(c['rays'] > 100 for c in gl_report['cases'] if c.get('mode') == 'projection')


def test_picture_has_surface_and_background(gl_report):
    pic = gl_report['picture']
    assert 0.5 < pic['white_fraction'] < 0.95 and pic['blue_fraction'] > 0.005


def test_real_window_draws(gl_report):
    win = gl_report['window']
    assert win['canvas'] and win['problem'] == '' and win['error'] == ''
    assert win['image'][0] > 500 and win['colours_surface'] > 50
    assert win['changed_by_turn'] > 0.01 and win['changed_by_mode'] > 0.01
    # the canvas in another window has a new OpenGL context: the volume is sent again
    assert win['after_new_context']['error'] == '' and win['after_new_context']['not_background'] > 0.01


def test_mouse_turns_moves_and_zooms(gl_report):
    mouse = gl_report['window']['mouse']
    d_yaw, d_pitch, moved = mouse['turn']                        # a drag of (50, 20) pixels
    assert d_yaw == pytest.approx(-0.4) and d_pitch == pytest.approx(0.16) and moved == 0.0
    for key in ('move_right_button', 'move_shift'):              # the view moves and does not turn
        assert mouse[key][0] == pytest.approx(0.0, abs=1e-12) and mouse[key][1] > 1e-4
    assert mouse['zoom_factor'] == pytest.approx(2 ** 0.25)
    assert 'h = ' in mouse['readout_on_surface'] and 'intensity' in mouse['readout_on_surface']
    assert mouse['value_over_level'] == pytest.approx(1.0, rel=0.1)   # the point is on the surface
    assert mouse['readout_off_surface'].startswith('Drag')
