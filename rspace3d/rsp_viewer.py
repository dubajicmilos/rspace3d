"""
rsp_viewer.py — Unified reciprocal space viewer.

Opens .img (single CrysAlisPro frame), .cbf (raw detector frame),
or .h5 (3D volume) files in a single interface with auto-detection.

Usage:
    python rsp_viewer.py
    python rsp_viewer.py file.img
    python rsp_viewer.py volume.h5
"""

import sys
import os
import numpy as np
from scipy.ndimage import map_coordinates

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QToolBar, QStatusBar, QFileDialog, QComboBox,
    QDoubleSpinBox, QCheckBox, QLabel, QPushButton, QSlider,
    QStackedWidget, QDialog, QTextEdit, QDialogButtonBox,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm, Normalize

from .rsp_reader import read_rsp_layer
from .volume_builder import VolumeData, load_volume_h5, compute_plane_M_inv, extract_volume_slice


COLORMAPS = [
    'viridis', 'plasma', 'inferno', 'magma', 'hot', 'afmhot',
    'gray', 'gray_r', 'bone', 'jet', 'turbo', 'coolwarm', 'seismic',
    'gnuplot2', 'cubehelix', 'nipy_spectral', 'cividis',
]


class UnifiedViewer(QMainWindow):
    """Single GUI for .img, .cbf, and .h5 reciprocal space data."""

    def __init__(self, filename=None):
        super().__init__()
        self.setWindowTitle('Reciprocal Space Viewer')
        self.resize(1300, 850)

        # State
        self.layer = None          # single .img
        self.vol = None            # 3D volume
        self._raw_data = None      # raw .cbf
        self._display_data = None
        self._y_flipped = False
        self._mode = None          # 'img', 'cbf', 'vol'

        # Grid/line state
        self.grid_lines = []
        self.line_artist = None
        self._line_start = None
        self._cid_press = self._cid_drag = self._cid_release = None
        self.im = None
        self.cbar = None
        self.extent = None
        self._current_M_inv = None
        self._current_x_label = 'x'
        self._current_y_label = 'y'

        self._build_ui()

        if filename:
            self._load_file(filename)

    # ──────────────────────────────────────────────────────────
    # UI
    # ──────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Toolbar row 1: file + display ──
        tb = QToolBar('Controls')
        tb.setMovable(False)
        self.addToolBar(tb)

        open_act = QAction('Open...', self)
        open_act.setShortcut('Ctrl+O')
        open_act.triggered.connect(self._open_file)
        tb.addAction(open_act)
        tb.addSeparator()

        tb.addWidget(QLabel(' Cmap: '))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(COLORMAPS)
        self.cmap_combo.setCurrentText('gnuplot2')
        self.cmap_combo.currentTextChanged.connect(self._update_cmap)
        tb.addWidget(self.cmap_combo)
        tb.addSeparator()

        tb.addWidget(QLabel(' Min: '))
        self.min_spin = QDoubleSpinBox()
        self.min_spin.setDecimals(1)
        self.min_spin.setRange(-1e8, 1e8)
        self.min_spin.setValue(0)
        self.min_spin.valueChanged.connect(self._update_clim)
        tb.addWidget(self.min_spin)

        tb.addWidget(QLabel(' Max: '))
        self.max_spin = QDoubleSpinBox()
        self.max_spin.setDecimals(1)
        self.max_spin.setRange(-1e8, 1e8)
        self.max_spin.setValue(100)
        self.max_spin.valueChanged.connect(self._update_clim)
        tb.addWidget(self.max_spin)
        tb.addSeparator()

        self.log_check = QCheckBox('Log')
        self.log_check.toggled.connect(self._update_clim)
        tb.addWidget(self.log_check)

        self.grid_check = QCheckBox('Grid')
        self.grid_check.toggled.connect(self._toggle_grid)
        tb.addWidget(self.grid_check)

        self.cbar_check = QCheckBox('Colorbar')
        self.cbar_check.setChecked(True)
        self.cbar_check.toggled.connect(self._toggle_colorbar)
        tb.addWidget(self.cbar_check)
        tb.addSeparator()

        self.line_btn = QPushButton('Line Profile')
        self.line_btn.setCheckable(True)
        self.line_btn.toggled.connect(self._toggle_line_tool)
        tb.addWidget(self.line_btn)

        # ── Toolbar row 2: axis ranges + volume controls ──
        tb2 = QToolBar('Navigation')
        tb2.setMovable(False)
        self.addToolBar(tb2)

        tb2.addWidget(QLabel(' x range: '))
        self.xrange_spin = QDoubleSpinBox()
        self.xrange_spin.setDecimals(1)
        self.xrange_spin.setRange(0.5, 100)
        self.xrange_spin.setValue(8)
        self.xrange_spin.setSingleStep(1.0)
        self.xrange_spin.valueChanged.connect(self._apply_axis_range)
        tb2.addWidget(self.xrange_spin)

        tb2.addWidget(QLabel(' y range: '))
        self.yrange_spin = QDoubleSpinBox()
        self.yrange_spin.setDecimals(1)
        self.yrange_spin.setRange(0.5, 100)
        self.yrange_spin.setValue(8)
        self.yrange_spin.setSingleStep(1.0)
        self.yrange_spin.valueChanged.connect(self._apply_axis_range)
        tb2.addWidget(self.yrange_spin)
        tb2.addSeparator()

        tb2.addSeparator()
        export_btn = QPushButton('Export...')
        export_btn.clicked.connect(self._export)
        tb2.addWidget(export_btn)

        # ── Toolbar row 3: volume controls (hidden for single frames) ──
        self.vol_toolbar = QToolBar('Volume')
        self.vol_toolbar.setMovable(False)
        self.addToolBar(self.vol_toolbar)

        self.vol_toolbar.addWidget(QLabel(' Plane: '))
        self.plane_combo = QComboBox()
        self.plane_combo.addItems(['HK (fix L)', 'HL (fix K)', 'KL (fix H)'])
        self.plane_combo.currentIndexChanged.connect(self._on_plane_changed)
        self.vol_toolbar.addWidget(self.plane_combo)

        self.vol_toolbar.addWidget(QLabel(' Fixed value: '))
        self.slice_spin = QDoubleSpinBox()
        self.slice_spin.setDecimals(3)
        self.slice_spin.setSingleStep(0.02)
        self.slice_spin.valueChanged.connect(self._on_slice_changed)
        self.vol_toolbar.addWidget(self.slice_spin)

        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setMinimumWidth(300)
        self.slice_slider.valueChanged.connect(self._on_slider_moved)
        self.vol_toolbar.addWidget(self.slice_slider)

        self.slice_label = QLabel('')
        self.vol_toolbar.addWidget(self.slice_label)
        self.vol_toolbar.addSeparator()

        self.vol_toolbar.addWidget(QLabel(' Integrate ±: '))
        self.int_range_spin = QDoubleSpinBox()
        self.int_range_spin.setDecimals(2)
        self.int_range_spin.setRange(0, 5)
        self.int_range_spin.setValue(0)
        self.int_range_spin.setSingleStep(0.02)
        self.int_range_spin.valueChanged.connect(self._on_slice_changed)
        self.vol_toolbar.addWidget(self.int_range_spin)
        self.vol_toolbar.addSeparator()

        self.iso_btn = QPushButton('3D Isosurface...')
        self.iso_btn.clicked.connect(self._show_isosurface)
        self.vol_toolbar.addWidget(self.iso_btn)

        self.info_btn = QPushButton('Info...')
        self.info_btn.clicked.connect(self._show_info)
        self.vol_toolbar.addWidget(self.info_btn)

        self.vol_toolbar.setVisible(False)

        # ── Main area ──
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        self.main_fig = Figure(tight_layout=True)
        self.main_ax = self.main_fig.add_subplot(111)
        self.main_canvas = FigureCanvasQTAgg(self.main_fig)
        self.nav_toolbar = NavigationToolbar2QT(self.main_canvas, self)
        ll.addWidget(self.main_canvas)
        ll.addWidget(self.nav_toolbar)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.profile_fig = Figure(tight_layout=True)
        self.profile_ax = self.profile_fig.add_subplot(111)
        self.profile_canvas = FigureCanvasQTAgg(self.profile_fig)
        rl.addWidget(self.profile_canvas)

        self.splitter.addWidget(left)
        self.splitter.addWidget(right)
        self.splitter.setSizes([900, 400])
        right.setVisible(False)
        self.setCentralWidget(self.splitter)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage('Open a .img, .cbf, or .h5 file')

        # Mouse tracking
        self.main_canvas.mpl_connect('motion_notify_event', self._on_mouse_move)

    # ──────────────────────────────────────────────────────────
    # File I/O
    # ──────────────────────────────────────────────────────────

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open', '',
            'All supported (*.img *.cbf *.h5 *.hdf5 *.npz);;'
            'CrysAlisPro (*.img);;CBF (*.cbf);;HDF5 (*.h5 *.hdf5);;All (*)')
        if path:
            self._load_file(path)

    def _load_file(self, path):
        ext = os.path.splitext(path)[1].lower()
        self.layer = None
        self.vol = None
        self._raw_data = None
        self._current_M_inv = None
        self.grid_lines.clear()
        self.line_artist = None
        self.cbar = None
        self.im = None

        try:
            if ext == '.cbf':
                self._load_cbf(path)
            elif ext in ('.h5', '.hdf5', '.npz'):
                self._load_volume(path)
            else:
                self._load_img(path)
        except Exception as e:
            self.status.showMessage(f'Error: {e}')
            import traceback; traceback.print_exc()

    def _load_img(self, path):
        self.layer = read_rsp_layer(path)
        self._mode = 'img'
        self._current_M_inv = self.layer.M_inv
        self._current_x_label = self.layer.x_label
        self._current_y_label = self.layer.y_label
        self.vol_toolbar.setVisible(False)
        self.setWindowTitle(f'RSP Viewer — {os.path.basename(path)}')
        self._display_img()
        self._auto_range()

    def _load_cbf(self, path):
        import fabio
        cbf = fabio.open(path)
        self._raw_data = cbf.data.astype(float)
        self._mode = 'cbf'
        self._current_M_inv = None
        self.vol_toolbar.setVisible(False)
        self.setWindowTitle(f'RSP Viewer — {os.path.basename(path)}')
        self._display_cbf()
        self._auto_range()

    def _load_volume(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.h5', '.hdf5'):
            self.vol = load_volume_h5(path)
        else:
            d = np.load(path, allow_pickle=False)
            self.vol = VolumeData(
                intensity=d['intensity'], H=d['H'], K=d['K'], L=d['L'],
                plane_type=str(d.get('plane_type', 'HK')),
                metadata={'wavelength': float(d.get('wavelength', 0))})

        self._mode = 'vol'
        self.vol_toolbar.setVisible(True)
        self.setWindowTitle(f'RSP Viewer — {os.path.basename(path)}')

        nh, nk, nl = self.vol.intensity.shape
        self.status.showMessage(
            f'Volume: {nh}x{nk}x{nl}  '
            f'H=[{self.vol.H[0]:.2f},{self.vol.H[-1]:.2f}]  '
            f'K=[{self.vol.K[0]:.2f},{self.vol.K[-1]:.2f}]  '
            f'L=[{self.vol.L[0]:.2f},{self.vol.L[-1]:.2f}]')

        self._setup_vol_controls()
        self._update_vol_slice()
        self._auto_range()

    # ──────────────────────────────────────────────────────────
    # Display — single .img
    # ──────────────────────────────────────────────────────────

    def _display_img(self):
        layer = self.layer
        NY, NX = layer.intensity.shape
        s, cx, cy = layer.s, layer.cx, layer.cy

        x_min = (0.5 - cx) * s
        x_max = (NX + 0.5 - cx) * s
        y_first = (0.5 - cy) * s
        y_last = (NY + 0.5 - cy) * s

        cx_int = NX // 2
        self._y_flipped = layer.idx2[-1, cx_int] < layer.idx2[0, cx_int]
        self._display_data = layer.intensity.astype(float)
        if self._y_flipped:
            self._display_data = self._display_data[::-1, :]

        self.extent = [x_min, x_max, y_first, y_last]
        self._show_image(
            title=f'{layer.fixed_label} = {layer.fixed_value:.4f}',
            xlabel=layer.x_label, ylabel=layer.y_label)

    # ──────────────────────────────────────────────────────────
    # Display — raw CBF
    # ──────────────────────────────────────────────────────────

    def _display_cbf(self):
        NY, NX = self._raw_data.shape
        self._display_data = self._raw_data
        self._y_flipped = False
        self.extent = [0, NX, 0, NY]
        self._current_x_label = 'pixel x'
        self._current_y_label = 'pixel y'
        self._show_image(xlabel='pixel x', ylabel='pixel y')

    # ──────────────────────────────────────────────────────────
    # Display — volume slice (extracts volume slices)
    # ──────────────────────────────────────────────────────────

    def _setup_vol_controls(self):
        self._on_plane_changed(0)

    def _on_plane_changed(self, _idx=None):
        if self.vol is None:
            return
        fixed_ax = self._get_vol_fixed_axis()
        self.slice_spin.blockSignals(True)
        self.slice_slider.blockSignals(True)
        self.slice_spin.setRange(float(fixed_ax[0]), float(fixed_ax[-1]))
        target = 0.0 if fixed_ax[0] <= 0 <= fixed_ax[-1] else float(fixed_ax[len(fixed_ax)//2])
        self.slice_spin.setValue(target)
        self.slice_slider.setRange(0, len(fixed_ax) - 1)
        self.slice_slider.setValue(int(np.argmin(np.abs(fixed_ax - target))))
        self.slice_spin.blockSignals(False)
        self.slice_slider.blockSignals(False)
        self._update_vol_slice()

    def _on_slice_changed(self, _val=None):
        if self.vol is None:
            return
        fixed_ax = self._get_vol_fixed_axis()
        idx = int(np.argmin(np.abs(fixed_ax - self.slice_spin.value())))
        self.slice_slider.blockSignals(True)
        self.slice_slider.setValue(idx)
        self.slice_slider.blockSignals(False)
        self._update_vol_slice()

    def _on_slider_moved(self, idx):
        if self.vol is None:
            return
        fixed_ax = self._get_vol_fixed_axis()
        self.slice_spin.blockSignals(True)
        self.slice_spin.setValue(float(fixed_ax[idx]))
        self.slice_spin.blockSignals(False)
        self._update_vol_slice()

    def _get_vol_fixed_axis(self):
        idx = self.plane_combo.currentIndex()
        return [self.vol.L, self.vol.K, self.vol.H][idx]

    def _get_vol_plane_type(self):
        return ['HK', 'HL', 'KL'][self.plane_combo.currentIndex()]

    def _update_vol_slice(self):
        """Extract and display a volume slice."""
        if self.vol is None:
            return

        plane_idx = self.plane_combo.currentIndex()
        target_val = self.slice_spin.value()
        int_range = self.int_range_spin.value()

        sl, x_ax, y_ax, xl, yl, fl, fv, n_sl = extract_volume_slice(
            self.vol, plane_idx, target_val, int_range)

        plane_type = self._get_vol_plane_type()
        ub = self.vol.metadata.get('ub')
        wl = self.vol.metadata.get('wavelength', 1.0)
        if ub is not None:
            self._current_M_inv = compute_plane_M_inv(ub, wl, plane_type)
        else:
            self._current_M_inv = self.vol.metadata.get('M_inv')

        # Y-flip for native plane
        is_native = plane_type == self.vol.plane_type
        if is_native and self._current_M_inv is not None and self._current_M_inv[1, 1] < 0:
            self._y_flipped = True
            sl = sl[::-1, :]
        else:
            self._y_flipped = False

        self._display_data = sl
        self._current_x_label = xl
        self._current_y_label = yl

        # Compute extent
        s = self.vol.metadata.get('s')
        if is_native and s is not None:
            ny_sl, nx_sl = sl.shape
            cx_e = (nx_sl + 1) / 2.0
            cy_e = (ny_sl + 1) / 2.0
            x_min = (0.5 - cx_e) * s
            x_max = (nx_sl + 0.5 - cx_e) * s
            y_first = (0.5 - cy_e) * s
            y_last = (ny_sl + 0.5 - cy_e) * s
            if self._y_flipped:
                y_first, y_last = y_last, y_first
            self.extent = [x_min, x_max, min(y_first, y_last), max(y_first, y_last)]
        elif s is not None:
            # Regridded non-native: uses s directly
            ny_sl, nx_sl = sl.shape
            cx_e = (nx_sl + 1) / 2.0
            cy_e = (ny_sl + 1) / 2.0
            self.extent = [(0.5-cx_e)*s, (nx_sl+0.5-cx_e)*s,
                           (0.5-cy_e)*s, (ny_sl+0.5-cy_e)*s]
        else:
            dx = abs(x_ax[-1]-x_ax[0]) / max(len(x_ax)-1, 1) / 2
            dy = abs(y_ax[-1]-y_ax[0]) / max(len(y_ax)-1, 1) / 2
            self.extent = [x_ax[0]-dx, x_ax[-1]+dx, y_ax[0]-dy, y_ax[-1]+dy]

        title = f'{fl} = {fv:.3f}'
        if n_sl > 1:
            title += f'  (±{self.int_range_spin.value():.2f}, {n_sl} slices)'
        self.slice_label.setText(f'{fl}={fv:.3f}')

        self._show_image(title=title, xlabel=xl, ylabel=yl)

    # ──────────────────────────────────────────────────────────
    # Common display
    # ──────────────────────────────────────────────────────────

    def _show_image(self, title='', xlabel='', ylabel=''):
        self.main_fig.clear()
        self.main_ax = self.main_fig.add_subplot(111)
        self.grid_lines.clear()
        self.line_artist = None
        self.cbar = None

        cmap = self.cmap_combo.currentText()
        self.main_ax.set_autoscale_on(False)
        self.im = self.main_ax.imshow(
            self._display_data, extent=self.extent, origin='lower',
            aspect='equal', cmap=cmap, interpolation='nearest')

        self.main_ax.set_xlabel(xlabel, fontsize=12)
        self.main_ax.set_ylabel(ylabel, fontsize=12)
        self.main_ax.set_title(title, fontsize=13)
        self.main_ax.format_coord = lambda x, y: ''

        if self._current_M_inv is not None:
            self._set_miller_ticks()

        self._apply_axis_range()

        if self.cbar_check.isChecked():
            self.cbar = self.main_fig.colorbar(self.im, ax=self.main_ax,
                                                fraction=0.046, pad=0.04)
        if self.grid_check.isChecked():
            self._draw_grid_lines()

        self._update_clim()
        self.main_canvas.draw()

    def _auto_range(self):
        if self._display_data is None:
            return
        data = self._display_data
        pos = data[data > 0]
        vmin = float(np.percentile(pos, 1)) if len(pos) > 0 else 0
        vmax = float(np.percentile(data, 99.5))
        self.min_spin.blockSignals(True)
        self.max_spin.blockSignals(True)
        self.min_spin.setRange(float(data.min()), float(data.max()))
        self.max_spin.setRange(float(data.min()), float(data.max()))
        self.min_spin.setValue(vmin)
        self.max_spin.setValue(vmax)
        self.min_spin.blockSignals(False)
        self.max_spin.blockSignals(False)
        self._update_clim()

    # ──────────────────────────────────────────────────────────
    # Colormap / clim
    # ──────────────────────────────────────────────────────────

    def _update_cmap(self, name):
        if self.im:
            self.im.set_cmap(name)
            self.main_canvas.draw_idle()

    def _update_clim(self):
        if self.im is None:
            return
        vmin, vmax = self.min_spin.value(), self.max_spin.value()
        if vmax <= vmin:
            vmax = vmin + 1
        if self.log_check.isChecked():
            vmin = max(vmin, 0.1)
            self.im.set_norm(LogNorm(vmin=vmin, vmax=vmax))
        else:
            self.im.set_norm(Normalize(vmin=vmin, vmax=vmax))
        self.main_canvas.draw_idle()

    def _toggle_colorbar(self, state):
        if self.im is None:
            return
        if state and self.cbar is None:
            self.cbar = self.main_fig.colorbar(self.im, ax=self.main_ax,
                                                fraction=0.046, pad=0.04)
        elif not state and self.cbar is not None:
            self.cbar.remove()
            self.cbar = None
        self.main_canvas.draw_idle()

    # ──────────────────────────────────────────────────────────
    # Miller index ticks and grid (from M_inv)
    # ──────────────────────────────────────────────────────────

    def _miller_to_cart(self, i1, i2):
        M = np.linalg.inv(self._current_M_inv)
        return M[0, 0]*i1 + M[0, 1]*i2, M[1, 0]*i1 + M[1, 1]*i2

    def _set_miller_ticks(self):
        M_inv = self._current_M_inv
        if M_inv is None:
            return
        xlim = self.main_ax.get_xlim()
        ylim = self.main_ax.get_ylim()

        if abs(M_inv[0, 0]) > 1e-10:
            x_per = 1.0 / M_inv[0, 0]
            n_min = int(np.ceil(min(xlim) * M_inv[0, 0]))
            n_max = int(np.floor(max(xlim) * M_inv[0, 0]))
            self.main_ax.set_xticks([n * x_per for n in range(n_min, n_max + 1)])
            self.main_ax.set_xticklabels([str(n) for n in range(n_min, n_max + 1)])

        if abs(M_inv[1, 1]) > 1e-10:
            y_per = 1.0 / M_inv[1, 1]
            m_lo, m_hi = min(ylim) * M_inv[1, 1], max(ylim) * M_inv[1, 1]
            if m_lo > m_hi:
                m_lo, m_hi = m_hi, m_lo
            m_min = int(np.ceil(m_lo))
            m_max = int(np.floor(m_hi))
            self.main_ax.set_yticks([m * y_per for m in range(m_min, m_max + 1)])
            self.main_ax.set_yticklabels([str(m) for m in range(m_min, m_max + 1)])

    def _apply_axis_range(self, _val=None):
        if self._display_data is None:
            return
        if self._current_M_inv is not None:
            rx, ry = self.xrange_spin.value(), self.yrange_spin.value()
            corners = [(-rx, -ry), (rx, -ry), (-rx, ry), (rx, ry)]
            cart = [self._miller_to_cart(i1, i2) for i1, i2 in corners]
            self.main_ax.set_xlim(min(c[0] for c in cart), max(c[0] for c in cart))
            self.main_ax.set_ylim(min(c[1] for c in cart), max(c[1] for c in cart))
            self._set_miller_ticks()
        else:
            # CBF: pixel range
            self.main_ax.set_xlim(self.extent[0], self.extent[1])
            self.main_ax.set_ylim(self.extent[2], self.extent[3])
        if self.grid_check.isChecked():
            self._draw_grid_lines()
        self.main_canvas.draw_idle()

    # ──────────────────────────────────────────────────────────
    # Grid overlay (tilted lines from M_inv)
    # ──────────────────────────────────────────────────────────

    def _toggle_grid(self, state):
        if state:
            self._draw_grid_lines()
        else:
            for l in self.grid_lines:
                l.remove()
            self.grid_lines.clear()
        self.main_canvas.draw_idle()

    def _draw_grid_lines(self):
        for l in self.grid_lines:
            l.remove()
        self.grid_lines.clear()

        M_inv = self._current_M_inv
        if M_inv is None:
            return

        xlim = np.array(self.main_ax.get_xlim())
        ylim = np.array(self.main_ax.get_ylim())
        saved = (tuple(xlim), tuple(ylim))

        cx = [xlim[0], xlim[1], xlim[0], xlim[1]]
        cy = [ylim[0], ylim[0], ylim[1], ylim[1]]
        i1 = [M_inv[0, 0]*x + M_inv[0, 1]*y for x, y in zip(cx, cy)]
        i2 = [M_inv[1, 0]*x + M_inv[1, 1]*y for x, y in zip(cx, cy)]

        tilt = dict(color='white', alpha=0.35, lw=0.5, scalex=False, scaley=False)
        straight = dict(color='white', alpha=0.35, lw=0.5)

        for n in range(int(np.floor(min(i1))), int(np.ceil(max(i1))) + 1):
            if abs(M_inv[0, 1]) > 1e-12:
                y0 = (n - M_inv[0, 0]*xlim[0]) / M_inv[0, 1]
                y1 = (n - M_inv[0, 0]*xlim[1]) / M_inv[0, 1]
                line, = self.main_ax.plot(xlim, [y0, y1], **tilt)
            else:
                line = self.main_ax.axvline(n / M_inv[0, 0], **straight)
            self.grid_lines.append(line)

        for m in range(int(np.floor(min(i2))), int(np.ceil(max(i2))) + 1):
            if abs(M_inv[1, 0]) > 1e-12:
                y0 = (m - M_inv[1, 0]*xlim[0]) / M_inv[1, 1]
                y1 = (m - M_inv[1, 0]*xlim[1]) / M_inv[1, 1]
                line, = self.main_ax.plot(xlim, [y0, y1], **tilt)
            else:
                line = self.main_ax.axhline(m / M_inv[1, 1], **straight)
            self.grid_lines.append(line)

        self.main_ax.set_xlim(saved[0])
        self.main_ax.set_ylim(saved[1])

    # ──────────────────────────────────────────────────────────
    # Cursor readout
    # ──────────────────────────────────────────────────────────

    def _on_mouse_move(self, event):
        if event.inaxes != self.main_ax or self._display_data is None:
            return
        x, y = event.xdata, event.ydata
        sl = self._display_data
        ny, nx = sl.shape
        ext = self.extent

        fi = (x - ext[0]) / (ext[1] - ext[0]) * nx - 0.5
        fj = (y - ext[2]) / (ext[3] - ext[2]) * ny - 0.5
        i, j = int(round(fi)), int(round(fj))

        if 0 <= i < nx and 0 <= j < ny:
            inten = sl[j, i]
            if self._current_M_inv is not None:
                M_inv = self._current_M_inv
                v1 = M_inv[0, 0]*x + M_inv[0, 1]*y
                v2 = M_inv[1, 0]*x + M_inv[1, 1]*y
                msg = f'{self._current_x_label}={v1:.4f}  {self._current_y_label}={v2:.4f}'
                if self._mode == 'vol':
                    fl = ['l', 'k', 'h'][self.plane_combo.currentIndex()]
                    msg += f'  {fl}={self.slice_spin.value():.3f}'
                elif self._mode == 'img' and self.layer:
                    msg += f'  {self.layer.fixed_label}={self.layer.fixed_value:.4f}'
                self.status.showMessage(f'{msg}   I={inten:.1f}')
            else:
                self.status.showMessage(f'x={x:.0f}  y={y:.0f}  I={inten:.1f}')

    # ──────────────────────────────────────────────────────────
    # Line profile
    # ──────────────────────────────────────────────────────────

    def _toggle_line_tool(self, active):
        right = self.splitter.widget(1)
        if active:
            self.nav_toolbar.mode = ''
            for a in self.nav_toolbar.actions():
                if a.isCheckable() and a.isChecked():
                    a.setChecked(False)
            self._cid_press = self.main_canvas.mpl_connect('button_press_event', self._lp_press)
            self._cid_drag = self.main_canvas.mpl_connect('motion_notify_event', self._lp_drag)
            self._cid_release = self.main_canvas.mpl_connect('button_release_event', self._lp_release)
            right.setVisible(True)
            self.splitter.setSizes([900, 400])
        else:
            for c in [self._cid_press, self._cid_drag, self._cid_release]:
                if c: self.main_canvas.mpl_disconnect(c)
            self._cid_press = self._cid_drag = self._cid_release = None
            if self.line_artist:
                self.line_artist.remove()
                self.line_artist = None
                self.main_canvas.draw_idle()
            right.setVisible(False)

    def _lp_press(self, e):
        if e.inaxes != self.main_ax or e.button != 1: return
        self._line_start = (e.xdata, e.ydata)
        if self.line_artist:
            self.line_artist.remove()
            self.line_artist = None
            self.main_canvas.draw_idle()

    def _lp_drag(self, e):
        if not self._line_start or e.inaxes != self.main_ax: return
        x0, y0 = self._line_start
        if self.line_artist:
            self.line_artist.set_data([x0, e.xdata], [y0, e.ydata])
        else:
            self.line_artist, = self.main_ax.plot(
                [x0, e.xdata], [y0, e.ydata], 'r-', lw=1.5, alpha=0.8,
                scalex=False, scaley=False)
        self.main_canvas.draw_idle()

    def _lp_release(self, e):
        if not self._line_start or e.inaxes != self.main_ax:
            self._line_start = None
            return
        x0, y0 = self._line_start
        x1, y1 = e.xdata, e.ydata
        self._line_start = None
        if abs(x1-x0) < 0.01 and abs(y1-y0) < 0.01:
            return

        sl = self._display_data
        ny, nx = sl.shape
        ext = self.extent
        num = max(500, int(max(abs(x1-x0), abs(y1-y0)) /
                  min((ext[1]-ext[0])/nx, (ext[3]-ext[2])/ny) * 3))
        t = np.linspace(0, 1, num)
        xl = x0 + t*(x1-x0)
        yl = y0 + t*(y1-y0)
        fi = (xl - ext[0])/(ext[1]-ext[0])*nx - 0.5
        fj = (yl - ext[2])/(ext[3]-ext[2])*ny - 0.5
        valid = (fi >= 0) & (fi < nx-1) & (fj >= 0) & (fj < ny-1)
        profile = np.full(num, np.nan)
        if valid.any():
            profile[valid] = map_coordinates(sl, [fj[valid], fi[valid]], order=1)

        if self._current_M_inv is not None:
            M_inv = self._current_M_inv
            i1 = M_inv[0,0]*xl + M_inv[0,1]*yl
            i2 = M_inv[1,0]*xl + M_inv[1,1]*yl
            if abs(i1[-1]-i1[0]) >= abs(i2[-1]-i2[0]):
                xd, xlab = i1, self._current_x_label
                cv, cl = (i2[0]+i2[-1])/2, self._current_y_label
            else:
                xd, xlab = i2, self._current_y_label
                cv, cl = (i1[0]+i1[-1])/2, self._current_x_label
        else:
            xd, xlab = xl if abs(x1-x0)>=abs(y1-y0) else yl, 'position'
            cv, cl = 0, ''

        self.profile_ax.clear()
        self.profile_ax.plot(xd, profile, 'b-', lw=0.8)
        self.profile_ax.set_xlabel(xlab)
        self.profile_ax.set_ylabel('Intensity')
        if cl:
            self.profile_ax.set_title(f'{cl} \u2248 {cv:.2f}', fontsize=10)
        if self.log_check.isChecked():
            self.profile_ax.set_yscale('log')
        self.profile_canvas.draw()

    # ──────────────────────────────────────────────────────────
    # Export
    # ──────────────────────────────────────────────────────────

    def _export(self):
        if self._display_data is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export', '', 'PNG (*.png);;PDF (*.pdf);;SVG (*.svg)')
        if path:
            self.main_fig.savefig(path, dpi=300, bbox_inches='tight',
                                  pad_inches=0.05, facecolor='white')
            self.status.showMessage(f'Exported: {path}')

    def _show_isosurface(self):
        """Launch 3D isosurface viewer for the loaded volume."""
        if self.vol is None:
            return
        rx = self.xrange_spin.value()
        ry = self.yrange_spin.value()
        try:
            from .volume_isosurface import plot_isosurface
            self.status.showMessage('Launching 3D isosurface...')
            plot_isosurface(
                self.vol, method='pyvista',
                h_range=(-rx, rx), k_range=(-ry, ry),
                l_range=(float(self.vol.L[0]), float(self.vol.L[-1])))
        except Exception as e:
            self.status.showMessage(f'Isosurface error: {e}')

    def _show_info(self):
        """Show volume metadata in a dialog."""
        if self.vol is None:
            return

        import numpy as np
        v = self.vol
        m = v.metadata
        nh, nk, nl = v.intensity.shape

        lines = []
        lines.append('=== Volume Information ===\n')
        lines.append(f'Shape:      {nh} x {nk} x {nl}')
        lines.append(f'Data type:  {v.intensity.dtype}')
        lines.append(f'Memory:     {v.intensity.nbytes / 1e6:.0f} MB')
        lines.append(f'Plane type: {v.plane_type}')
        lines.append(f'')
        lines.append(f'H range:    [{v.H[0]:.4f}, {v.H[-1]:.4f}]  ({len(v.H)} pts, step {v.H[1]-v.H[0]:.6f})')
        lines.append(f'K range:    [{v.K[0]:.4f}, {v.K[-1]:.4f}]  ({len(v.K)} pts, step {v.K[1]-v.K[0]:.6f})')
        lines.append(f'L range:    [{v.L[0]:.4f}, {v.L[-1]:.4f}]  ({len(v.L)} pts, step {v.L[1]-v.L[0]:.6f})')
        lines.append(f'')

        nz = (v.intensity != 0).sum()
        lines.append(f'Nonzero:    {nz:,} / {v.intensity.size:,} ({nz/v.intensity.size*100:.1f}%)')
        lines.append(f'Intensity:  [{v.intensity.min():.1f}, {v.intensity.max():.1f}]')
        lines.append(f'')

        cell = m.get('cell')
        if cell:
            lines.append('=== Unit Cell ===\n')
            lines.append(f'a = {cell["a"]:.5f} A')
            lines.append(f'b = {cell["b"]:.5f} A')
            lines.append(f'c = {cell["c"]:.5f} A')
            lines.append(f'alpha = {cell["alpha"]:.3f} deg')
            lines.append(f'beta  = {cell["beta"]:.3f} deg')
            lines.append(f'gamma = {cell["gamma"]:.3f} deg')
            lines.append(f'')

        wl = m.get('wavelength')
        if wl:
            lines.append(f'Wavelength: {wl:.6f} A')

        lines.append(f'')
        lines.append('=== Processing ===\n')
        laue = m.get('laue_group', '')
        if laue:
            lines.append(f'Laue group: {laue}')
        lines.append(f'Bin HK:     {m.get("bin_xy", 1)}x')
        lines.append(f'Bin L:      {m.get("bin_z", 1)}x')

        s = m.get('s')
        if s:
            lines.append(f'')
            lines.append(f'=== Grid ===\n')
            lines.append(f'Cartesian step (s): {s:.8f} 1/A per pixel')
            cx = m.get('cx'); cy = m.get('cy')
            if cx:
                lines.append(f'Pixel center:       cx={cx:.1f}, cy={cy:.1f}')

        M_inv = m.get('M_inv')
        if M_inv is not None:
            lines.append(f'')
            lines.append(f'M_inv (pixel -> Miller):')
            lines.append(f'  [{M_inv[0,0]:.6f}, {M_inv[0,1]:.6f}]')
            lines.append(f'  [{M_inv[1,0]:.6f}, {M_inv[1,1]:.6f}]')

        ub = m.get('ub')
        if ub is not None:
            lines.append(f'')
            lines.append(f'UB matrix (includes lambda):')
            for row in ub:
                lines.append(f'  [{row[0]:12.8f}, {row[1]:12.8f}, {row[2]:12.8f}]')

        src = m.get('source_folder', '')
        if src:
            lines.append(f'')
            lines.append(f'Source: {src}')

        # Show dialog
        dlg = QDialog(self)
        dlg.setWindowTitle('Volume Information')
        dlg.setMinimumSize(500, 600)
        layout = QVBoxLayout(dlg)

        text = QTextEdit()
        text.setReadOnly(True)
        text.setStyleSheet('font-family: Consolas, monospace; font-size: 12px;')
        text.setPlainText('\n'.join(lines))
        layout.addWidget(text)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(dlg.accept)
        layout.addWidget(buttons)

        dlg.exec()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName('RSP Viewer')
    fname = sys.argv[1] if len(sys.argv) > 1 else None
    viewer = UnifiedViewer(filename=fname)
    viewer.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
