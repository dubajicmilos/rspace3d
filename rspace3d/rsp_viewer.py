"""
rsp_viewer.py — Unified reciprocal space viewer.

Opens .img (single CrysAlisPro frame), .cbf (raw detector frame),
or .h5 (3D volume) files in a single interface with auto-detection.

Usage:
    python rsp_viewer.py
    python rsp_viewer.py file.img
    python rsp_viewer.py volume.h5
"""

from __future__ import annotations

import io
import sys
import os
import numpy as np
import numpy.typing as npt
from scipy.ndimage import map_coordinates

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFrame,
    QSplitter, QStatusBar, QFileDialog, QComboBox, QScrollArea, QStackedWidget,
    QDoubleSpinBox, QCheckBox, QPushButton, QSlider, QSizePolicy,
    QDialog, QTextEdit, QDialogButtonBox, QGridLayout,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFontMetrics, QKeySequence, QPixmap, QShortcut

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm, Normalize

from . import ui
from .rsp_reader import RSPLayer, read_rsp_layer
from .volume_builder import VolumeData, load_volume_h5, compute_plane_M_inv, extract_volume_slice


COLORMAPS = [
    'viridis', 'plasma', 'inferno', 'magma', 'hot', 'afmhot',
    'gray', 'gray_r', 'bone', 'jet', 'turbo', 'coolwarm', 'seismic',
    'gnuplot2', 'cubehelix', 'nipy_spectral', 'cividis',
]


class ViewToolbar(NavigationToolbar2QT):
    """The matplotlib tool bar with the view tools only: home, back, forward, pan and zoom.

    The panel has its own button for the export of the image.
    """
    toolitems = [t for t in NavigationToolbar2QT.toolitems
                 if t[0] in ('Home', 'Back', 'Forward', None, 'Pan', 'Zoom')]


class UnifiedViewer(QMainWindow):
    """Single GUI for .img, .cbf, and .h5 reciprocal space data."""

    def __init__(self, filename: str | None = None) -> None:
        super().__init__()
        self.setWindowTitle('Reciprocal Space Viewer')
        ui.start_size(self, 1300, 850)

        # State
        self.layer: RSPLayer | None = None          # single .img
        self.vol: VolumeData | None = None          # 3D volume
        self._raw_data: npt.NDArray[np.float64] | None = None      # raw .cbf
        self._display_data: npt.NDArray[np.floating] | None = None
        self._y_flipped: bool = False
        self._mode: str | None = None               # 'img', 'cbf', 'vol'

        # Grid/line state
        self.grid_lines: list = []
        self.line_artist = None
        self._line_start: tuple[float, float] | None = None
        self._cid_press: int | None = None
        self._cid_drag: int | None = None
        self._cid_release: int | None = None
        self.im = None
        self.cbar = None
        self.extent: list[float] | None = None
        self._current_M_inv: npt.NDArray[np.float64] | None = None
        self._current_x_label: str = 'x'
        self._current_y_label: str = 'y'
        # Last fixed-axis value used across all planes. Carried over when
        # switching plane so the viewer doesn't reset the cut to 0.
        self._last_cut_value: float | None = None

        self._build_ui()

        if filename:
            self._load_file(filename)

    # ──────────────────────────────────────────────────────────
    # UI
    # ──────────────────────────────────────────────────────────

    def _build_ui(self):
        ui.ensure_theme()
        self.setMinimumSize(1000, 640)

        central = QWidget()
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_panel())

        # The centre: the empty state or the image, and the line profile on the right.
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(self._build_centre())
        self.splitter.addWidget(self._build_profile())
        self.splitter.setSizes([700, 330])
        self.splitter.widget(1).setVisible(False)
        outer.addWidget(self.splitter, 1)

        # Status bar: the cursor readout is the message, the file summary is permanent.
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.summary_label = ui.label('', 'caption')
        self.status.addPermanentWidget(self.summary_label)
        self.status.showMessage('Open a file to start.')

        self.main_canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        QShortcut(QKeySequence('Ctrl+O'), self, self._open_file)
        QShortcut(QKeySequence(QKeySequence.StandardKey.HelpContents), self,
                  lambda: ui.open_help(self, 'viewer'))

        self._sync_mode_widgets()
        ui.guard_wheel(self)

    def _heading(self, lay, text):
        """A heading of the control panel, with 16 px of space above it."""
        lay.addSpacing(ui.SPACE[1])
        lay.addWidget(ui.label(text, 'section-title'))

    @staticmethod
    def _field(text, widget, tip='', width=84):
        """A label and a control in one row. Returns the row and the label."""
        lab = ui.label(text)
        lab.setFixedWidth(width)
        if tip:
            lab.setToolTip(tip)
        lay = ui.row(lab, stretch=False)
        lay.addWidget(widget, 1)
        return lay, lab

    def _build_panel(self):
        S = ui.SPACE
        panel = QFrame()
        panel.setObjectName('sidebar')
        panel.setFixedWidth(272)
        shell = QVBoxLayout(panel)
        shell.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.viewport().setAutoFillBackground(False)
        shell.addWidget(scroll)
        body = QWidget()
        scroll.setWidget(body)
        lay = QVBoxLayout(body)
        lay.setContentsMargins(S[3], S[3], S[3], S[3])
        lay.setSpacing(S[1])

        # 1. Name of the window
        names = QVBoxLayout()
        names.setSpacing(0)
        names.addWidget(ui.label('rspace3d', 'app-title'))
        names.addWidget(ui.label('Viewer', 'caption'))
        head = QHBoxLayout()
        head.setSpacing(S[1])
        head.addWidget(ui.Mark(), 0, Qt.AlignmentFlag.AlignTop)
        head.addLayout(names, 1)
        lay.addLayout(head)
        lay.addSpacing(S[1])

        # 2. File
        self.open_btn = ui.button('Open file...', None,
                                  'Open a volume (.h5), a CrysAlisPro layer (.img) or a raw '
                                  'frame (.cbf). Shortcut: Ctrl+O.', self._open_file)
        lay.addWidget(self.open_btn)
        self.file_label = ui.label('', 'strong')
        self.file_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.facts1 = ui.label('', 'caption', wrap=True)
        self.facts2 = ui.label('', 'caption', wrap=True)
        for w in (self.file_label, self.facts1, self.facts2):
            lay.addWidget(w)

        # 3. Colour scale
        self._heading(lay, 'Colour scale')
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(COLORMAPS)
        self.cmap_combo.setCurrentText('gnuplot2')
        self.cmap_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.cmap_combo.setMinimumContentsLength(8)
        self.cmap_combo.setToolTip('The colours that show the intensity.')
        self.cmap_combo.currentTextChanged.connect(self._update_cmap)
        field, _ = self._field('Colour map', self.cmap_combo)
        lay.addLayout(field)

        self.min_spin = QDoubleSpinBox()
        self.min_spin.setDecimals(1)
        self.min_spin.setRange(-1e8, 1e8)
        self.min_spin.setValue(0)
        self.min_spin.setToolTip('Intensity that has the first colour of the map.')
        self.min_spin.valueChanged.connect(self._update_clim)
        self.max_spin = QDoubleSpinBox()
        self.max_spin.setDecimals(1)
        self.max_spin.setRange(-1e8, 1e8)
        self.max_spin.setValue(100)
        self.max_spin.setToolTip('Intensity that has the last colour of the map.')
        self.max_spin.valueChanged.connect(self._update_clim)
        for spin in (self.min_spin, self.max_spin):
            spin.setMinimumWidth(64)      # the range of the box would make it wider
        mm = QGridLayout()
        mm.setHorizontalSpacing(ui.SPACE[1])
        mm.setVerticalSpacing(ui.SPACE[0])
        mm.addWidget(ui.label('Min'), 0, 0)
        mm.addWidget(ui.label('Max'), 0, 1)
        mm.addWidget(self.min_spin, 1, 0)
        mm.addWidget(self.max_spin, 1, 1)
        lay.addLayout(mm)

        self.auto_btn = ui.button('Auto', tip='Set Min and Max from the slice that is shown now.',
                                  clicked=self._auto_range)
        lay.addLayout(ui.row(self.auto_btn))
        self.log_check = QCheckBox('Log scale')
        self.log_check.setToolTip('Show the intensity on a logarithmic scale. Values at or '
                                  'below zero get the first colour.')
        self.log_check.toggled.connect(self._update_clim)
        lay.addWidget(self.log_check)

        # 4. View
        self._heading(lay, 'View')
        self.xrange_spin = QDoubleSpinBox()
        self.xrange_spin.setDecimals(1)
        self.xrange_spin.setRange(0.5, 100)
        self.xrange_spin.setValue(8)
        self.xrange_spin.setSingleStep(1.0)
        self.xrange_spin.valueChanged.connect(self._apply_axis_range)
        self.yrange_spin = QDoubleSpinBox()
        self.yrange_spin.setDecimals(1)
        self.yrange_spin.setRange(0.5, 100)
        self.yrange_spin.setValue(8)
        self.yrange_spin.setSingleStep(1.0)
        self.yrange_spin.valueChanged.connect(self._apply_axis_range)
        field, self.xrange_label = self._field('x range ±', self.xrange_spin)
        lay.addLayout(field)
        field, self.yrange_label = self._field('y range ±', self.yrange_spin)
        lay.addLayout(field)
        self.fit_btn = ui.button('Fit to data', tip='Set both ranges so that the image fills '
                                 'the axes.', clicked=self._fit_view)
        lay.addLayout(ui.row(self.fit_btn))
        self.grid_check = QCheckBox('Miller-index grid')
        self.grid_check.setToolTip('Draw a line at each integer Miller index.')
        self.grid_check.toggled.connect(self._toggle_grid)
        lay.addWidget(self.grid_check)
        self.cbar_check = QCheckBox('Colour bar')
        self.cbar_check.setChecked(True)
        self.cbar_check.setToolTip('Show the colour bar next to the image.')
        self.cbar_check.toggled.connect(self._toggle_colorbar)
        lay.addWidget(self.cbar_check)

        # 5. Tools
        self._heading(lay, 'Tools')
        self.line_btn = QPushButton('Line profile')
        self.line_btn.setCheckable(True)
        self.line_btn.setToolTip('Show the intensity along a line that you draw on the image.')
        self.line_btn.toggled.connect(self._toggle_line_tool)
        lay.addWidget(self.line_btn)
        self.line_hint = ui.hint('Drag on the image to draw a line.')
        self.line_hint.setVisible(False)             # shown while the tool is on
        lay.addWidget(self.line_hint)
        self.export_btn = ui.button('Export image...', tip='Save the image to a file or copy '
                                    'it to the clipboard.', clicked=self._export)
        lay.addWidget(self.export_btn)
        self.iso_btn = ui.button('3D isosurface...', tip='Show the volume as surfaces of '
                                 'equal intensity in 3D.', clicked=self._show_isosurface)
        lay.addWidget(self.iso_btn)
        self.info_btn = ui.button('File information...', tip='Show the size, the ranges, the '
                                  'unit cell and the processing of the volume.',
                                  clicked=self._show_info)
        lay.addWidget(self.info_btn)

        # 6. Help
        lay.addStretch(1)
        lay.addWidget(ui.button('Help (F1)', 'link', 'Open the help for this window.',
                                lambda: ui.open_help(self, 'viewer')))
        self._data_widgets = [self.auto_btn, self.fit_btn, self.line_btn, self.export_btn]
        return panel

    def _build_centre(self):
        S = ui.SPACE
        self.stack = QStackedWidget()

        # Page 0: empty state
        empty = QWidget()
        el = QVBoxLayout(empty)
        el.setContentsMargins(S[4], S[4], S[4], S[4])
        el.addStretch(1)
        # The texts have their own line breaks. A wrapped label in a centred column
        # gets its height for the width of the window, and its lines are cut.
        el.setSpacing(S[2])
        msg = ui.label('Open a volume (.h5), a CrysAlisPro layer (.img)\nor a raw frame (.cbf).',
                       'page-title')
        msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        el.addWidget(msg, 0, Qt.AlignmentFlag.AlignHCenter)
        what = ui.label('A volume shows one plane at a time. You select the plane and the '
                        'slice.\nA layer shows one plane in Miller indices. A raw frame shows '
                        'the detector image in pixels.', 'hint')
        what.setAlignment(Qt.AlignmentFlag.AlignCenter)
        el.addWidget(what, 0, Qt.AlignmentFlag.AlignHCenter)
        self.empty_open_btn = ui.button('Open file...', 'primary', 'Open a volume, a layer or '
                                        'a raw frame. Shortcut: Ctrl+O.', self._open_file)
        self.empty_open_btn.setMinimumWidth(160)
        el.addWidget(self.empty_open_btn, 0, Qt.AlignmentFlag.AlignHCenter)
        el.addStretch(1)
        self.stack.addWidget(empty)

        # Page 1: canvas, navigation tool bar and slice bar
        page = QWidget()
        pl = QVBoxLayout(page)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(0)
        self.main_fig = Figure(tight_layout=True)
        self.main_ax = self.main_fig.add_subplot(111)
        self.main_canvas = FigureCanvasQTAgg(self.main_fig)
        self.nav_toolbar = ViewToolbar(self.main_canvas, self)
        pl.addWidget(self.main_canvas, 1)
        pl.addWidget(self.nav_toolbar)
        pl.addWidget(self._build_slice_bar())
        self.stack.addWidget(page)
        return self.stack

    def _build_slice_bar(self):
        S = ui.SPACE
        self.vol_toolbar = QWidget()
        lay = QHBoxLayout(self.vol_toolbar)
        lay.setContentsMargins(S[2], S[1], S[3], S[2])
        lay.setSpacing(S[1])

        lay.addWidget(ui.label('Plane'))
        self.plane_combo = QComboBox()
        self.plane_combo.addItems(['hk plane (l fixed)', 'hl plane (k fixed)',
                                   'kl plane (h fixed)'])
        self.plane_combo.setToolTip('The plane of the volume that is shown. The third index '
                                    'is fixed.')
        self.plane_combo.currentIndexChanged.connect(self._on_plane_changed)
        lay.addWidget(self.plane_combo)

        self.fixed_label = ui.label('l =')
        lay.addWidget(self.fixed_label)
        self.slice_spin = QDoubleSpinBox()
        self.slice_spin.setDecimals(3)
        self.slice_spin.setSingleStep(0.02)
        self.slice_spin.setMinimumWidth(96)
        self.slice_spin.setToolTip('Value of the fixed index. It is the position of the slice.')
        self.slice_spin.valueChanged.connect(self._on_slice_changed)
        lay.addWidget(self.slice_spin)

        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setMinimumWidth(120)
        self.slice_slider.setToolTip('Move the slice through the volume.')
        self.slice_slider.valueChanged.connect(self._on_slider_moved)
        lay.addWidget(self.slice_slider, 1)

        lay.addWidget(ui.label('Integrate over ±'))
        self.int_range_spin = QDoubleSpinBox()
        self.int_range_spin.setDecimals(2)
        self.int_range_spin.setRange(0, 5)
        self.int_range_spin.setValue(0)
        self.int_range_spin.setSingleStep(0.02)
        self.int_range_spin.setMinimumWidth(92)
        self.int_range_spin.setToolTip('Add the slices within this distance of the fixed '
                                       'index. Zero shows one slice.')
        self.int_range_spin.valueChanged.connect(self._on_slice_changed)
        lay.addWidget(self.int_range_spin)

        self.slice_label = ui.label('', 'caption')
        lay.addWidget(self.slice_label)
        self.vol_toolbar.setVisible(False)
        return self.vol_toolbar

    def _build_profile(self):
        S = ui.SPACE
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(S[1], S[2], S[2], S[1])
        lay.setSpacing(S[1])
        lay.addWidget(ui.label('Line profile', 'section-title'))
        self.profile_fig = Figure(tight_layout=True)
        self.profile_ax = self.profile_fig.add_subplot(111)
        self.profile_canvas = FigureCanvasQTAgg(self.profile_fig)
        lay.addWidget(self.profile_canvas, 1)
        return panel

    def _sync_mode_widgets(self):
        """Show the controls that fit the open file and the page of the centre."""
        loaded = self._mode is not None
        self.stack.setCurrentIndex(1 if loaded else 0)
        for w in self._data_widgets:
            w.setEnabled(loaded)
        for w in (self.file_label, self.facts1, self.facts2):
            w.setVisible(loaded and (w is not self.facts2 or bool(w.text())))
        self.iso_btn.setVisible(self._mode == 'vol')
        self.info_btn.setVisible(self._mode == 'vol')
        self.vol_toolbar.setVisible(self._mode == 'vol')
        for spin in (self.xrange_spin, self.yrange_spin):
            spin.setEnabled(self._mode in ('img', 'vol'))
        self.fit_btn.setEnabled(self._mode in ('img', 'vol'))
        self._sync_range_labels()

    def _sync_range_labels(self):
        """Name the two range boxes after the axes of the shown plane."""
        if self._mode in ('img', 'vol'):
            xl, yl = self._current_x_label, self._current_y_label
        else:
            xl, yl = 'x', 'y'
        self.xrange_label.setText(f'{xl} range ±' if self._mode != 'cbf' else 'x range')
        self.yrange_label.setText(f'{yl} range ±' if self._mode != 'cbf' else 'y range')
        self.xrange_spin.setToolTip(f'Half width of the view along {xl}, in Miller-index units.')
        self.yrange_spin.setToolTip(f'Half width of the view along {yl}, in Miller-index units.')

    def _data_miller_extent(self):
        """Largest absolute Miller indices of the two axes over the shown data, or None."""
        M = self._current_M_inv
        if M is None or self.extent is None or self._display_data is None:
            return None
        e = self.extent
        ny, nx = self._display_data.shape
        dx, dy = (e[1] - e[0]) / nx / 2, (e[3] - e[2]) / ny / 2     # first and last pixel centres
        xs = np.array([e[0] + dx, e[1] - dx, e[0] + dx, e[1] - dx])
        ys = np.array([e[2] + dy, e[2] + dy, e[3] - dy, e[3] - dy])
        i1 = M[0, 0] * xs + M[0, 1] * ys
        i2 = M[1, 0] * xs + M[1, 1] * ys
        return float(np.abs(i1).max()), float(np.abs(i2).max())

    def _fit_view(self):
        """Set the two ranges to the data extent, rounded up to 0.1."""
        ext = self._data_miller_extent()
        if ext is None:
            return
        vals = [min(max(np.ceil(v * 10 - 1e-9) / 10, 0.5), 100.0) for v in ext]
        for spin, v in zip((self.xrange_spin, self.yrange_spin), vals):
            spin.blockSignals(True)
            spin.setValue(float(v))
            spin.blockSignals(False)
        self._apply_axis_range()

    def _set_summary(self, name, path, line1, line2=''):
        """Fill the file name, the facts in the panel and the summary in the status bar."""
        fm = QFontMetrics(self.file_label.font())
        self.file_label.setText(fm.elidedText(name, Qt.TextElideMode.ElideMiddle, 216))
        self.file_label.setToolTip(path)
        self.facts1.setText(line1)
        self.facts2.setText(line2)
        self.facts2.setVisible(bool(line2))
        self.file_label.setVisible(True)
        self.facts1.setVisible(True)
        self.summary_label.setText(f'{name}  |  {line1}')

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
            # No file is open now: show the empty window, not the image of the file before
            self._mode = None
            self._display_data = None
            self.layer = self.vol = self._raw_data = None
            self.summary_label.setText('')
            self.setWindowTitle('Reciprocal Space Viewer')
            self._sync_mode_widgets()
            ui.show_error(
                self, 'This file cannot be opened',
                f'The file "{os.path.basename(path)}" could not be read. Check that it is a '
                f'volume (.h5), a CrysAlisPro layer (.img) or a raw frame (.cbf).\n'
                f'{type(e).__name__}: {e}')

    def _load_img(self, path):
        self.layer = read_rsp_layer(path)
        self._mode = 'img'
        self._current_M_inv = self.layer.M_inv
        self._current_x_label = self.layer.x_label
        self._current_y_label = self.layer.y_label
        self.setWindowTitle(f'Viewer - {os.path.basename(path)}')
        ny, nx = self.layer.intensity.shape
        self._set_summary(os.path.basename(path), path, f'Layer, {nx} x {ny} pixels',
                          f'{self.layer.fixed_label} = {self.layer.fixed_value:.4f}')
        self._sync_mode_widgets()
        self._display_img()
        self._auto_range()
        self._fit_view()
        self.status.showMessage('Move the cursor over the image to read the indices and '
                                'the intensity.')

    def _load_cbf(self, path):
        import fabio
        cbf = fabio.open(path)
        self._raw_data = cbf.data.astype(float)
        self._mode = 'cbf'
        self._current_M_inv = None
        self.setWindowTitle(f'Viewer - {os.path.basename(path)}')
        ny, nx = self._raw_data.shape
        self._set_summary(os.path.basename(path), path, f'Raw frame, {nx} x {ny} pixels')
        self._sync_mode_widgets()
        self._display_cbf()
        self._auto_range()
        self.status.showMessage('Move the cursor over the image to read the pixel and '
                                'the intensity.')

    def _load_volume(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.h5', '.hdf5'):
            self.vol = load_volume_h5(path)
        else:
            d = np.load(path, allow_pickle=False)
            unc = d.get('uncertainty')
            self.vol = VolumeData(
                intensity=d['intensity'], H=d['H'], K=d['K'], L=d['L'],
                plane_type=str(d.get('plane_type', 'HK')),
                metadata={'wavelength': float(d.get('wavelength', 0))},
                variance=None if unc is None else np.square(unc.astype(np.float32)))

        self._mode = 'vol'
        self._last_cut_value = None
        self.setWindowTitle(f'Viewer - {os.path.basename(path)}')

        nh, nk, nl = self.vol.intensity.shape

        def rng(ax):
            return f'{round(float(ax[0]), 2):g} to {round(float(ax[-1]), 2):g}'

        self._set_summary(
            os.path.basename(path), path, f'Volume, {nh} x {nk} x {nl} voxels',
            f'h {rng(self.vol.H)}, k {rng(self.vol.K)}, l {rng(self.vol.L)}')
        self._sync_mode_widgets()
        self._setup_vol_controls()
        self._update_vol_slice()
        self._auto_range()
        self._fit_view()
        self.status.showMessage('Move the cursor over the image to read the indices and '
                                'the intensity.')

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
        last = self._last_cut_value
        if last is not None and fixed_ax[0] <= last <= fixed_ax[-1]:
            target = float(last)
        else:
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
        self._last_cut_value = target_val

        sl, x_ax, y_ax, xl, yl, fl, fv, n_sl = extract_volume_slice(
            self.vol, plane_idx, target_val, int_range)

        plane_type = self._get_vol_plane_type()
        ub = self.vol.metadata.get('ub')
        wl = self.vol.metadata.get('wavelength', 1.0)
        # Only a CrysAlisPro unwarp raster is displayed in Cartesian coordinates
        # (isotropic pixel step `s`, Miller grid through M_inv). A regular hkl
        # grid (rawrecon) and every non-native cut are already in Miller-index
        # coordinates: straight grid lines, identity M_inv.
        raster = self.vol.metadata.get('grid_kind', 'unwarp_raster') == 'unwarp_raster'
        is_native = plane_type == self.vol.plane_type and raster
        if is_native:
            if ub is not None:
                self._current_M_inv = compute_plane_M_inv(ub, wl, plane_type)
            else:
                self._current_M_inv = self.vol.metadata.get('M_inv')
        else:
            self._current_M_inv = np.eye(2)

        # Y-flip for native plane
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
            # raster centre from the volume geometry (odd rasters are not
            # centred on (n+1)/2; bin_volume keeps cx/cy consistent)
            cx_e = float(self.vol.metadata.get('cx', (nx_sl + 1) / 2.0))
            cy_e = float(self.vol.metadata.get('cy', (ny_sl + 1) / 2.0))
            x_min = (0.5 - cx_e) * s
            x_max = (nx_sl + 0.5 - cx_e) * s
            y_first = (0.5 - cy_e) * s
            y_last = (ny_sl + 0.5 - cy_e) * s
            if self._y_flipped:
                y_first, y_last = y_last, y_first
            self.extent = [x_min, x_max, min(y_first, y_last), max(y_first, y_last)]
        else:
            # Non-native: data in Miller indices — use x_ax/y_ax directly
            dx = abs(x_ax[-1]-x_ax[0]) / max(len(x_ax)-1, 1) / 2
            dy = abs(y_ax[-1]-y_ax[0]) / max(len(y_ax)-1, 1) / 2
            self.extent = [x_ax[0]-dx, x_ax[-1]+dx, y_ax[0]-dy, y_ax[-1]+dy]

        title = f'{fl} = {fv:.3f}'
        if n_sl > 1:
            title += f'  (±{self.int_range_spin.value():.2f}, {n_sl} slices)'
        self.slice_label.setText(f'{fl}={fv:.3f}')
        self.fixed_label.setText(f'{fl} =')

        self._sync_range_labels()
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

        self.main_ax.set_autoscale_on(False)
        self.im = self.main_ax.imshow(
            self._display_data, extent=self.extent, origin='lower',
            aspect='equal', cmap=self._get_cmap(), interpolation='nearest')

        self.main_ax.set_xlabel(xlabel, fontsize=12)
        self.main_ax.set_ylabel(ylabel, fontsize=12)
        self.main_ax.set_title(title, fontsize=13)
        self.main_ax.format_coord = lambda x, y: ''

        # Add colorbar BEFORE applying axis range so the widget aspect
        # reflects the final layout (colorbar steals width from the axes).
        if self.cbar_check.isChecked():
            self.cbar = self.main_fig.colorbar(self.im, ax=self.main_ax,
                                                fraction=0.046, pad=0.04)

        self._apply_axis_range()

        self._update_clim()
        self.main_canvas.draw()

    def _auto_range(self):
        if self._display_data is None:
            return
        data = self._display_data
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            d_lo, d_hi, vmin, vmax = 0.0, 1.0, 0.0, 1.0
        else:
            d_lo = float(finite.min())
            d_hi = float(finite.max())
            pos = finite[finite > 0]
            vmin = float(np.percentile(pos, 1)) if pos.size > 0 else d_lo
            vmax = float(np.percentile(finite, 99.5))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                vmin, vmax = d_lo, max(d_hi, d_lo + 1.0)
        self.min_spin.blockSignals(True)
        self.max_spin.blockSignals(True)
        self.min_spin.setRange(d_lo, d_hi)
        self.max_spin.setRange(d_lo, d_hi)
        self.min_spin.setValue(vmin)
        self.max_spin.setValue(vmax)
        self.min_spin.blockSignals(False)
        self.max_spin.blockSignals(False)
        self._update_clim()

    # ──────────────────────────────────────────────────────────
    # Colormap / clim
    # ──────────────────────────────────────────────────────────

    def _get_cmap(self):
        # Map "bad" values (NaN / non-positive inputs under LogNorm) to the low
        # end of the colormap so unmeasured voxels render the same as in linear
        # mode instead of showing through to the figure background (white).
        cmap = matplotlib.colormaps[self.cmap_combo.currentText()].copy()
        cmap.set_bad(cmap(0.0))
        return cmap

    def _update_cmap(self, name):
        if self.im:
            self.im.set_cmap(self._get_cmap())
            self.main_canvas.draw_idle()

    def _update_clim(self):
        if self.im is None:
            return
        vmin, vmax = self.min_spin.value(), self.max_spin.value()
        if vmax <= vmin:
            vmax = vmin + 1
        if self.log_check.isChecked():
            # LogNorm requires strictly positive vmin. clip=True maps positive
            # values below vmin to the low-end color instead of extrapolating.
            vmin = max(vmin, 1e-10)
            self.im.set_norm(LogNorm(vmin=vmin, vmax=vmax, clip=True))
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
            x_lo, x_hi = min(c[0] for c in cart), max(c[0] for c in cart)
            y_lo, y_hi = min(c[1] for c in cart), max(c[1] for c in cart)
            self.main_ax.set_xlim(x_lo, x_hi)
            self.main_ax.set_ylim(y_lo, y_hi)
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

    def closeEvent(self, event):
        ui.release_help(self)
        super().closeEvent(event)

    def _toggle_line_tool(self, active):
        self.line_hint.setVisible(bool(active))
        right = self.splitter.widget(1)
        if active:
            self.nav_toolbar.mode = ''
            for a in self.nav_toolbar.actions():
                if a.isCheckable() and a.isChecked():
                    a.setChecked(False)
            self._cid_press = self.main_canvas.mpl_connect('button_press_event', self._lp_press)
            self._cid_drag = self.main_canvas.mpl_connect('motion_notify_event', self._lp_drag)
            self._cid_release = self.main_canvas.mpl_connect('button_release_event', self._lp_release)
            if not self.profile_ax.lines:
                self._profile_placeholder()
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

    def _profile_placeholder(self):
        """Tell the user what the empty profile panel will show."""
        ax = self.profile_ax
        ax.clear()
        ax.set_axis_off()
        ax.text(0.5, 0.5, 'Drag on the image\nto draw a line.', ha='center', va='center',
                transform=ax.transAxes, color=ui.COLORS['muted'])
        self.profile_canvas.draw_idle()

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
        self.profile_ax.set_axis_on()
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

        dlg = QDialog(self)
        dlg.setWindowTitle('Export options')
        layout = QVBoxLayout(dlg)
        cb_colorbar = QCheckBox('Include colorbar')
        cb_colorbar.setChecked(self.cbar is not None)
        cb_colorbar.setEnabled(self.cbar is not None)
        cb_title = QCheckBox('Include title')
        cb_title.setChecked(True)
        cb_axes = QCheckBox('Include axis labels and numbers')
        cb_axes.setChecked(True)
        layout.addWidget(cb_colorbar)
        layout.addWidget(cb_title)
        layout.addWidget(cb_axes)

        action = {'mode': None}
        btns = QDialogButtonBox()
        save_btn = btns.addButton('Save to file...',
                                  QDialogButtonBox.ButtonRole.AcceptRole)
        copy_btn = btns.addButton('Copy to clipboard',
                                  QDialogButtonBox.ButtonRole.AcceptRole)
        btns.addButton(QDialogButtonBox.StandardButton.Cancel)

        def pick(mode):
            action['mode'] = mode
            dlg.accept()
        save_btn.clicked.connect(lambda: pick('save'))
        copy_btn.clicked.connect(lambda: pick('copy'))
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)

        if dlg.exec() != QDialog.DialogCode.Accepted or action['mode'] is None:
            return

        path = None
        if action['mode'] == 'save':
            path, _ = QFileDialog.getSaveFileName(
                self, 'Export', '', 'PNG (*.png);;PDF (*.pdf);;SVG (*.svg)')
            if not path:
                return

        saved_title = self.main_ax.get_title()
        hid_cbar = False
        hid_axes = False
        try:
            if not cb_title.isChecked():
                self.main_ax.set_title('')
            if not cb_axes.isChecked():
                self.main_ax.xaxis.set_visible(False)
                self.main_ax.yaxis.set_visible(False)
                hid_axes = True
            if not cb_colorbar.isChecked() and self.cbar is not None:
                self.cbar.ax.set_visible(False)
                hid_cbar = True

            if action['mode'] == 'save':
                self.main_fig.savefig(path, dpi=300, bbox_inches='tight',
                                      pad_inches=0.05, facecolor='white')
                self.status.showMessage(f'Exported: {path}')
            else:
                buf = io.BytesIO()
                self.main_fig.savefig(buf, format='png', dpi=300,
                                      bbox_inches='tight', pad_inches=0.05,
                                      facecolor='white')
                pixmap = QPixmap()
                pixmap.loadFromData(buf.getvalue(), 'PNG')
                QApplication.clipboard().setPixmap(pixmap)
                self.status.showMessage('Copied image to clipboard')
        finally:
            if saved_title:
                self.main_ax.set_title(saved_title)
            if hid_axes:
                self.main_ax.xaxis.set_visible(True)
                self.main_ax.yaxis.set_visible(True)
            if hid_cbar and self.cbar is not None:
                self.cbar.ax.set_visible(True)
            self.main_canvas.draw_idle()

    def _show_isosurface(self):
        """Launch 3D isosurface viewer for the loaded volume."""
        if self.vol is None:
            return
        # The two range boxes are the axes of the shown plane. The third axis is complete.
        full = {'h': (float(self.vol.H[0]), float(self.vol.H[-1])),
                'k': (float(self.vol.K[0]), float(self.vol.K[-1])),
                'l': (float(self.vol.L[0]), float(self.vol.L[-1]))}
        shown = {'HK': ('h', 'k'), 'HL': ('h', 'l'), 'KL': ('k', 'l')}[self._get_vol_plane_type()]
        for axis, spin in zip(shown, (self.xrange_spin, self.yrange_spin)):
            full[axis] = (-spin.value(), spin.value())
        try:
            from .volume_isosurface import plot_isosurface
            self.status.showMessage('Launching 3D isosurface...')
            plot_isosurface(
                self.vol, method='pyvista',
                h_range=full['h'], k_range=full['k'], l_range=full['l'])
        except Exception as e:
            self.status.showMessage(f'Isosurface error: {e}')
            ui.show_error(self, 'The 3D view did not open',
                          f'The 3D isosurface needs PyVista or Plotly (pip install pyvista).\n'
                          f'{type(e).__name__}: {e}')

    def _show_info(self):
        """Show volume metadata in a dialog."""
        if self.vol is None:
            return

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

        finite = np.isfinite(v.intensity)
        n_meas = int(finite.sum())
        lines.append(f'Measured:   {n_meas:,} / {v.intensity.size:,} ({n_meas/v.intensity.size*100:.1f}%)')
        if n_meas > 0:
            lines.append(f'Intensity:  [{np.nanmin(v.intensity):.1f}, {np.nanmax(v.intensity):.1f}]')
        else:
            lines.append(f'Intensity:  (no measured voxels)')
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
        text.setProperty('role', 'log')
        text.setPlainText('\n'.join(lines))
        layout.addWidget(text)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(dlg.accept)
        layout.addWidget(buttons)

        dlg.exec()


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName('RSP Viewer')
    fname = sys.argv[1] if len(sys.argv) > 1 else None
    viewer = UnifiedViewer(filename=fname)
    viewer.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
