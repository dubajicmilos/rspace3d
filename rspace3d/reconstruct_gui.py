"""
reconstruct_gui.py - GUI to reconstruct a 3D reciprocal-space volume directly from
raw CrysAlisPro rotation frames (.cbf) + the *_cracker.par, then bin / symmetrize /
save with the same machinery as the .img volume builder.

Unlike the volume builder (which loads pre-computed unwarp .img layers), this GUI
computes the reconstruction on the GPU from the raw detector frames:
  cracker.par -> UB,  index frame 1 -> R0,  map every pixel -> (h,k,l),  accumulate.
Unmeasured voxels are known exactly (zero contributing pixels -> NaN), so there is
no coverage-mask guesswork; measured-but-zero voxels stay 0.0.

Tabs:
  Reconstruct  folder, Laue group, grid, output and one button that reconstructs,
               symmetrises with the standard averaging and saves.
  Averaging    measure and compare the averaging options, frame tools.
  Advanced     frames, geometry, pixel corrections, bin factors, averaging settings.

Usage:
    python -m rspace3d.reconstruct_gui        (or the `rspace3d-reconstruct` script)
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any, Callable

import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QFormLayout, QPushButton, QLabel, QComboBox, QCheckBox,
    QSpinBox, QDoubleSpinBox, QProgressBar, QFileDialog, QTextEdit, QMessageBox,
    QTabWidget, QScrollArea, QSplitter,
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt

from . import rawrecon
from .averaging import COUNTS_MAD, Cancelled
from .averaging_panel import LOG_HEIGHT, AveragingPanel, AveragingSettings
from .volume_builder import (
    bin_volume, symmetrize_volume, save_volume_h5, format_symmetry_diagnostics,
    fitting_laue_group, LAUE_GROUP_NAMES, METRIC_TOLERANCE, _EXPECTED_ORDERS, HAS_GPU,
)

STANDARD_WHY = (
    "Standard for raw frames: 'counts, mad 3'. Each orbit member has the weight of its "
    'pixel count. A member farther than 3 x the robust spread (MAD, at least the Poisson '
    'value) from the orbit median is rejected. In the rspace3d 2.4 benchmarks this option '
    'gave the lowest diffuse noise on clean data (real I15 data: split-half difference 18 % '
    "below 'equal, mad 3', Bragg sums within 1 %) at the speed of the plain MAD rule. For "
    'data with powder rings or other contamination, use the Averaging tab.')


# ──────────────────────────────────────────────────────────────────
# Worker thread (runs the reconstruction / symmetrisation off the UI thread)
# ──────────────────────────────────────────────────────────────────
class WorkerThread(QThread):
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    cancelled = pyqtSignal()
    log_msg = pyqtSignal(str)

    def __init__(self, func: Callable[..., Any], *args: Any) -> None:
        super().__init__()
        self.func = func
        self.args = args

    def _emit_progress(self, current: int, total: int) -> None:
        self.progress.emit(current, total)

    def _emit_log(self, msg: str) -> None:
        self.log_msg.emit(msg)

    def run(self) -> None:
        stopped = False
        try:
            self.finished.emit(self.func(*self.args))
        except Cancelled:
            stopped = True
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")
        if stopped:
            # The arrays of the stopped pass are free now. Give their GPU memory back.
            from . import orbit_engine as oe
            oe._free_gpu_memory()
            self.cancelled.emit()


def data_reach(detector, cell: dict) -> tuple[float, tuple[float, float, float]]:
    """|q| at the detector corners (1/Angstrom) and the largest |h|, |k|, |l| there.

    The geometry is the frame header (flat detector normal to the beam). A
    Miller index h is q . a, so |h| <= |q|max x a for all measured pixels.
    """
    ny, nx = detector.shape
    corners = np.array([[0, 0], [nx - 1, 0], [0, ny - 1], [nx - 1, ny - 1]], float)
    qmax = float(np.linalg.norm(detector.scattering_vectors(corners), axis=1).max())
    return qmax, (qmax * cell['a'], qmax * cell['b'], qmax * cell['c'])


# ──────────────────────────────────────────────────────────────────
# Main GUI
# ──────────────────────────────────────────────────────────────────
class ReconstructGUI(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('Raw-frame 3D Reconstruction')
        self.resize(1280, 920)
        self._worker: WorkerThread | None = None
        self._folder: str | None = None
        self._par: str | None = None
        self._n_total: int = 0
        self._vol = None        # rawrecon.Volume
        self._vd = None         # VolumeData (raw, NaN = unmeasured)
        self._info: dict | None = None
        self._viewer = None
        self._binned: dict | None = None   # binned volume of the Averaging panel: key, vol
        self._gpu_free_gb = 0.0
        self._frame_stale = False   # True: the grid or the frames changed after the reconstruction
        self._recip = None          # reciprocal basis of the .par (columns a*, b*, c*)
        self._recon_key: str | None = None     # settings of the reconstruction in memory
        self._running_key: str | None = None   # settings of the running reconstruction
        self._pending_save = False  # save after the running reconstruction
        self._cancel = False
        self._busy = False
        self._out_dir: str | None = None       # None: the dataset folder
        self._last_saved: str | None = None    # the last symmetrised file
        self._raw_saved: str | None = None     # the raw file of the selected volume
        self._build_ui()

        if HAS_GPU:
            try:
                import cupy as cp
                props = cp.cuda.runtime.getDeviceProperties(0)
                free, total = cp.cuda.runtime.memGetInfo()
                self._gpu_free_gb = free / 1e9
                nm = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
                self._log(f'GPU: {nm}  ({free/1e9:.1f}/{total/1e9:.1f} GB free)')
            except Exception:
                self._log('GPU detected')
        else:
            self.gpu_chk.setChecked(False)
            self.gpu_chk.setEnabled(False)
            self._log('No GPU - reconstruction will run on CPU (slow).')
        self.statusBar().showMessage('Select a dataset folder (raw .cbf + *_cracker.par)')

    # ── UI ──
    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        # Tabs above, progress and log below. The user can move the border.
        self.split = QSplitter(Qt.Orientation.Vertical)
        outer.addWidget(self.split)
        self.tabs = QTabWidget()
        self.split.addWidget(self.tabs)
        self.recon_tab = QWidget()
        layout = QVBoxLayout(self.recon_tab)
        # A scroll area: the tab does not push the log below its height
        recon_scroll = QScrollArea(); recon_scroll.setWidgetResizable(True)
        recon_scroll.setWidget(self.recon_tab)
        self.tabs.addTab(recon_scroll, 'Reconstruct')

        # 1. Dataset folder
        g1 = QGroupBox('1. Dataset folder  (raw .cbf frames + *_cracker.par)')
        v1 = QVBoxLayout(g1)
        h1 = QHBoxLayout()
        self.folder_label = QLabel('No folder selected'); self.folder_label.setWordWrap(True)
        h1.addWidget(self.folder_label, stretch=1)
        self.browse_btn = QPushButton('Browse...')
        self.browse_btn.clicked.connect(self._browse_folder)
        self.browse_btn.setToolTip('Folder with the raw .cbf rotation frames. The .cbf\n'
                                   'stem need not match the folder name - it is\n'
                                   'auto-detected from the numbered frames present.')
        h1.addWidget(self.browse_btn)
        v1.addLayout(h1)
        self.info_label = QLabel(''); self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet('color:#555; padding:4px;')
        self.info_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v1.addWidget(self.info_label)
        layout.addWidget(g1)
        self._dataset_group = g1

        # 2. Laue group and grid
        g2 = QGroupBox('2. Laue group and grid')
        gl = QGridLayout(g2)
        gl.addWidget(QLabel('Laue group:'), 0, 0)
        self.laue_combo = QComboBox()
        for key, label in LAUE_GROUP_NAMES.items():
            self.laue_combo.addItem(label, key)
        if 'm-3m' in LAUE_GROUP_NAMES:
            self.laue_combo.setCurrentIndex(list(LAUE_GROUP_NAMES).index('m-3m'))
        self.laue_combo.setToolTip(
            'Laue group for the symmetry averaging and the outlier rejection.\n'
            'm-3m = cubic (for example MAPbBr3). Select the group of the phase.\n'
            'An experiment that breaks the symmetry (for example an applied field)\n'
            'needs a lower group. The window checks only that the cell metric\n'
            'allows the group: the metric cannot tell a cubic cell from a cell\n'
            'with a small distortion.')
        gl.addWidget(self.laue_combo, 0, 1, 1, 2)
        self.laue_note = QLabel('')
        self.laue_note.setWordWrap(True)
        self.laue_note.setVisible(False)
        gl.addWidget(self.laue_note, 1, 0, 1, 5)
        mk = self._mk_range_spin
        gl.addWidget(QLabel('min'), 2, 1, alignment=Qt.AlignmentFlag.AlignHCenter)
        gl.addWidget(QLabel('max'), 2, 2, alignment=Qt.AlignmentFlag.AlignHCenter)
        gl.addWidget(QLabel('h:'), 3, 0); self.hmin = mk(-6.0); self.hmax = mk(6.0)
        gl.addWidget(self.hmin, 3, 1); gl.addWidget(self.hmax, 3, 2)
        gl.addWidget(QLabel('k:'), 4, 0); self.kmin = mk(-6.0); self.kmax = mk(6.0)
        gl.addWidget(self.kmin, 4, 1); gl.addWidget(self.kmax, 4, 2)
        gl.addWidget(QLabel('l:'), 5, 0); self.lmin = mk(-6.0); self.lmax = mk(6.0)
        gl.addWidget(self.lmin, 5, 1); gl.addWidget(self.lmax, 5, 2)

        gl.addWidget(QLabel('voxel dq (rlu):'), 3, 3)
        self.dq = QDoubleSpinBox(); self.dq.setRange(0.002, 1.0); self.dq.setDecimals(3)
        self.dq.setValue(0.025); self.dq.setSingleStep(0.005)
        self.dq.setToolTip('Voxel size in reciprocal-lattice units, applied to all\n'
                           'three axes. 0.025 rlu resolves Bragg peaks and diffuse.')
        gl.addWidget(self.dq, 3, 4)
        gl.addWidget(QLabel('Grid:'), 6, 0)
        self.estimate_label = QLabel(''); self.estimate_label.setStyleSheet('color:#337;')
        self.estimate_label.setWordWrap(True)
        gl.addWidget(self.estimate_label, 6, 1, 1, 5)
        gl.setColumnStretch(5, 1)
        layout.addWidget(g2)
        self._grid_group = g2

        # 3. Averaging (the settings are in the Advanced tab)
        g3 = QGroupBox('3. Averaging and outlier rejection')
        v3 = QVBoxLayout(g3)
        self.avg_label = QLabel(''); self.avg_label.setWordWrap(True)
        v3.addWidget(self.avg_label)
        # The volume of the frame tools (Averaging tab), if it is not the uncorrected one
        self.source_note = QLabel('')
        self.source_note.setStyleSheet('color: #070; font-weight: bold;')
        self.source_note.setWordWrap(True)
        self.source_note.setVisible(False)
        v3.addWidget(self.source_note)
        layout.addWidget(g3)

        # 4. Output
        g4 = QGroupBox('4. Output')
        v4 = QVBoxLayout(g4)
        h4 = QHBoxLayout()
        h4.addWidget(QLabel('Folder:'))
        self.out_label = QLabel(''); self.out_label.setWordWrap(True)
        h4.addWidget(self.out_label, stretch=1)
        self.out_btn = QPushButton('Change...')
        self.out_btn.setToolTip('Folder for the output files. The default is the dataset folder.')
        self.out_btn.clicked.connect(self._browse_out_dir)
        h4.addWidget(self.out_btn)
        v4.addLayout(h4)
        self.save_raw_chk = QCheckBox('Also save the raw (unsymmetrised) volume')
        self.save_raw_chk.setChecked(True)
        self.save_raw_chk.setToolTip('The raw volume (NaN = unmeasured) lets you symmetrise again\n'
                                     'later, for example with another Laue group, without a new\n'
                                     'reconstruction. It is larger than the symmetrised volume.')
        v4.addWidget(self.save_raw_chk)
        self.files_label = QLabel(''); self.files_label.setWordWrap(True)
        self.files_label.setStyleSheet('color:#555;')
        self.files_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v4.addWidget(self.files_label)
        layout.addWidget(g4)
        self._out_group = g4

        btns = QHBoxLayout()
        self.run_btn = QPushButton('Reconstruct and save')
        self.run_btn.setEnabled(False); self.run_btn.setMinimumHeight(38)
        self.run_btn.setStyleSheet('font-weight:bold; font-size:13px;')
        self.run_btn.setToolTip('Reconstruct, symmetrise with the averaging of step 3 and save\n'
                                'the files of step 4. If only the Laue group, the averaging or\n'
                                'the output changed, the window uses the reconstruction in\n'
                                'memory and does not reconstruct again.')
        self.run_btn.clicked.connect(self._run)
        btns.addWidget(self.run_btn, stretch=3)
        self.cancel_btn = QPushButton('Cancel')
        self.cancel_btn.setEnabled(False); self.cancel_btn.setMinimumHeight(38)
        self.cancel_btn.setToolTip('Stop the reconstruction after the current frame.\n'
                                   'The window then keeps the earlier reconstruction.')
        self.cancel_btn.clicked.connect(self._cancel_run)
        btns.addWidget(self.cancel_btn, stretch=1)
        self.view_btn = QPushButton('Open in viewer')
        self.view_btn.setEnabled(False); self.view_btn.setMinimumHeight(38)
        self.view_btn.setToolTip('Open the last saved symmetrised volume in the viewer.')
        self.view_btn.clicked.connect(self._open_in_viewer)
        btns.addWidget(self.view_btn, stretch=1)
        layout.addLayout(btns)
        layout.addStretch(1)

        # Averaging panel (works on the volume binned with the factors of the Advanced tab)
        self.avg_panel = AveragingPanel(self._averaging_request, log=self._log,
                                        progress=self._update_progress)
        self.avg_panel.busy_changed.connect(self._on_panel_busy)
        self.avg_panel.applied_changed.connect(self._on_applied_changed)
        # Frame tools: raw frames only (the volume builder keeps the group hidden)
        self.avg_panel.enable_frame_tools(self._frame_request)
        self.avg_panel.volume_changed.connect(self._on_volume_changed)
        self.avg_panel.set_save_action(
            'Symmetrise + Save', self._symmetrize_save,
            'Symmetrise the volume in memory with the applied option (or with\n'
            'the settings of the Advanced tab) and save it. The output folder\n'
            'and the file names are those of the Reconstruct tab.')
        self.frame_tools_group = self.avg_panel.frame_group
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setWidget(self.avg_panel)
        self.tabs.addTab(scroll, 'Averaging')

        self._build_advanced_tab()

        self.laue_combo.currentIndexChanged.connect(self._on_laue_changed)
        for w in (self.hmin, self.hmax, self.kmin, self.kmax, self.lmin, self.lmax, self.dq,
                  self.bin_xy, self.bin_z):
            w.valueChanged.connect(self._update_estimate)
        for w in (self.bin_xy, self.bin_z):
            w.valueChanged.connect(self._invalidate_binned)
        # A change of the grid, the frames or the pixel options makes the frame
        # tools results of the last reconstruction stale
        for w in (self.hmin, self.hmax, self.kmin, self.kmax, self.lmin, self.lmax, self.dq,
                  self.nframes, self.hot, self.pol_frac):
            w.valueChanged.connect(self._on_setup_changed)
        for w in (self.corr_chk, self.tabbin_chk):
            w.toggled.connect(self._on_setup_changed)
        self.save_raw_chk.toggled.connect(self._update_files_label)
        self.avg_settings.changed.connect(self._update_avg_label)

        bottom = QWidget(); bl = QVBoxLayout(bottom); bl.setContentsMargins(0, 0, 0, 0)
        self.progress = QProgressBar(); self.progress.setTextVisible(True)
        bl.addWidget(self.progress)
        self.log = QTextEdit(); self.log.setReadOnly(True)
        self.log.setStyleSheet('font-family:Consolas,monospace; font-size:11px;')
        bl.addWidget(self.log)
        self.split.addWidget(bottom)
        self.split.setStretchFactor(0, 1); self.split.setStretchFactor(1, 0)
        # About 230 px for the progress bar and the log at the default window size
        self.split.setSizes([LOG_HEIGHT * 3, LOG_HEIGHT])
        self._update_estimate()
        self._update_avg_label()
        self._update_out_label()

    def _build_advanced_tab(self):
        self.adv_tab = QWidget()
        layout = QVBoxLayout(self.adv_tab)
        adv_scroll = QScrollArea(); adv_scroll.setWidgetResizable(True)
        adv_scroll.setWidget(self.adv_tab)
        self.tabs.addTab(adv_scroll, 'Advanced')
        intro = QLabel('The defaults are the standard settings. Change them only if you need to.')
        intro.setStyleSheet('color:#555;')
        layout.addWidget(intro)

        ga = QGroupBox('Frames and geometry')
        fa = QFormLayout(ga)
        fa.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        self.nframes = QSpinBox(); self.nframes.setRange(0, 100000); self.nframes.setValue(0)
        self.nframes.setToolTip('Number of rotation frames to use. 0 = all frames in\n'
                                'the folder (recommended for full coverage). Use fewer\n'
                                'frames for a quick test.')
        fa.addRow('Frames (0 = all):', self.nframes)
        self.subfolder_par_chk = QCheckBox('Search subfolders for the .par file')
        self.subfolder_par_chk.setToolTip(
            'If the *_cracker.par is not in the selected folder itself but in a\n'
            'subfolder, tick this to search recursively (nearest match wins).')
        self.subfolder_par_chk.toggled.connect(self._on_subfolder_toggled)
        fa.addRow(self.subfolder_par_chk)
        self.tabbin_chk = QCheckBox('Geometry from CrysAlisPro tabbin (robust, any instrument)')
        self.tabbin_chk.setChecked(True)
        self.tabbin_chk.setToolTip(
            'Fit the full geometry (detector pose, oscillation axis, orientation)\n'
            'from the CrysAlisPro *_peakhunt.tabbin ground truth. Instrument-agnostic\n'
            'and robust to a wrong header distance/beam-centre or powder rings.\n\n'
            'Uncheck to instead index frame 1 directly (only works for a clean,\n'
            'correctly-headered dataset with a lab-Y oscillation axis).')
        fa.addRow(self.tabbin_chk)
        self.gpu_chk = QCheckBox('Use GPU (CuPy)'); self.gpu_chk.setChecked(True)
        self.gpu_chk.setToolTip('Reconstruct on the GPU via CuPy (tens of seconds for\n'
                                'a full scan). Uncheck to force the CPU path.')
        fa.addRow(self.gpu_chk)
        layout.addWidget(ga)

        gb = QGroupBox('Pixel corrections')
        fb = QFormLayout(gb)
        fb.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        self.corr_chk = QCheckBox('Solid-angle + polarisation corrections')
        self.corr_chk.setChecked(True)
        self.corr_chk.setToolTip('Multiply each pixel by the geometric intensity\n'
                                 'correction (1/cos^3(obliquity) x 1/polarisation).\n'
                                 'Lorentz is handled by the per-voxel count normalisation.')
        fb.addRow(self.corr_chk)
        self.pol_frac = QDoubleSpinBox(); self.pol_frac.setRange(0.0, 1.0)
        self.pol_frac.setValue(0.95); self.pol_frac.setSingleStep(0.05); self.pol_frac.setDecimals(2)
        self.pol_frac.setToolTip('Fraction of the beam polarised in the horizontal\n'
                                 '(detector fast) plane. ~0.95 for a synchrotron,\n'
                                 '0.5 = unpolarised.')
        fb.addRow('Polarisation fraction:', self.pol_frac)
        self.hot = QDoubleSpinBox(); self.hot.setRange(0, 1e9); self.hot.setDecimals(0)
        self.hot.setValue(0); self.hot.setSingleStep(1e5)
        self.hot.setToolTip('Ignore pixels with counts above this (zingers /\n'
                            'saturated). 0 = no clip (default): the Eiger at\n'
                            'I19-2 reaches ~1.3e6 counts in Bragg cores.')
        fb.addRow('Hot-pixel clip (0 = off):', self.hot)
        layout.addWidget(gb)

        gc = QGroupBox('Binning of the symmetrised volume')
        fc = QFormLayout(gc)
        fc.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        brow = QHBoxLayout()
        self.bin_xy = QSpinBox(); self.bin_xy.setRange(1, 16); self.bin_xy.setValue(1)
        self.bin_z = QSpinBox(); self.bin_z.setRange(1, 16); self.bin_z.setValue(1)
        brow.addWidget(self.bin_xy); brow.addWidget(self.bin_z); brow.addStretch(1)
        for w in (self.bin_xy, self.bin_z):
            w.setToolTip('Bin factors for the symmetrised volume. Set them before\n'
                         'you reconstruct: the grid puts hkl = 0 on a voxel centre\n'
                         'after this binning, so that the operations of all Laue\n'
                         'groups are exact. The raw volume is not binned.')
        fc.addRow('bin H,K / L:', brow)
        layout.addWidget(gc)

        self.avg_settings = AveragingSettings(COUNTS_MAD, STANDARD_WHY)
        layout.addWidget(self.avg_settings)

        self.reset_btn = QPushButton('Reset to the standard settings')
        self.reset_btn.setToolTip('Set all settings of this tab to the standard values.')
        self.reset_btn.clicked.connect(self._reset_advanced)
        rrow = QHBoxLayout(); rrow.addWidget(self.reset_btn); rrow.addStretch(1)
        layout.addLayout(rrow)
        layout.addStretch(1)

    def _mk_range_spin(self, val):
        s = QDoubleSpinBox(); s.setRange(-50, 50); s.setValue(val)
        s.setDecimals(2); s.setSingleStep(1.0); return s

    def _reset_advanced(self):
        self.nframes.setValue(0)
        self.tabbin_chk.setChecked(True)
        self.gpu_chk.setChecked(bool(HAS_GPU))
        self.corr_chk.setChecked(True)
        self.pol_frac.setValue(0.95)
        self.hot.setValue(0)
        self.bin_xy.setValue(1)
        self.bin_z.setValue(1)
        self.avg_settings.reset()
        self._log('Advanced: the settings are the standard settings again.')

    # ── grid-size estimate ──
    def _ranges(self):
        return ((self.hmin.value(), self.hmax.value()),
                (self.kmin.value(), self.kmax.value()),
                (self.lmin.value(), self.lmax.value()))

    def _bins(self):
        return (self.bin_xy.value(), self.bin_xy.value(), self.bin_z.value())

    def _grid_counts(self):
        """Voxel counts per axis of the grid that `_reconstruct` builds."""
        from .rawrecon.engine import _axes
        return _axes(self._ranges(), self.dq.value(), self._bins())[0]

    def _update_estimate(self):
        ns = self._grid_counts()
        nvox = ns[0] * ns[1] * ns[2]
        # GPU working set during accumulation: float64 sum + int32 count
        # + float64 sum of squares (variance)
        gb = rawrecon.gpu_memory_bytes(nvox) / 1e9
        txt = f'{ns[0]} x {ns[1]} x {ns[2]} = {nvox:,} voxels  (~{gb:.1f} GB GPU)'
        too_big = self._gpu_free_gb and gb > 0.7 * self._gpu_free_gb
        if any(hi <= lo for lo, hi in self._ranges()):
            txt = 'invalid range (max must exceed min)'
            self.estimate_label.setStyleSheet('color:#a00;')
        elif 0 in ns:
            txt = 'range too narrow: it must contain a multiple of dq x bin'
            self.estimate_label.setStyleSheet('color:#a00;')
        else:
            self.estimate_label.setStyleSheet('color:#a00;' if too_big else 'color:#337;')
            if self.bin_xy.value() > 1 or self.bin_z.value() > 1:
                txt += (f'. Bin {self.bin_xy.value()} x {self.bin_xy.value()} x '
                        f'{self.bin_z.value()} for the symmetrised volume (Advanced tab)')
            if too_big:
                txt += '  - may exceed GPU memory'
        self.estimate_label.setText(txt)

    # ── folder ──
    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, 'Select dataset folder', '')
        if folder:
            self._set_folder(folder)

    def _set_folder(self, folder):
        self._folder = folder
        self._par = None
        # A new folder invalidates the previous reconstruction: otherwise the
        # save steps would save/symmetrise the old volume under the new
        # folder's name.
        self._vol = self._vd = self._info = None
        self._recon_key = None
        self._raw_saved = self._last_saved = None
        self._recip = None
        self._frame_stale = False
        self.avg_panel.clear_frame_results('A new folder removed the frame tools results.')
        self._invalidate_binned()
        self.view_btn.setEnabled(False)
        self.folder_label.setText(folder)
        self._log(f'\nFolder: {folder}')
        self._update_out_label()

        # Discover frames (stem is NOT assumed to match the folder name) and the par
        frames, nums = rawrecon.find_cbf_frames(folder)
        self._n_total = len(frames)
        par = rawrecon.find_crysalis_par(folder)
        if par is None:                                  # optionally search subfolders
            par = self._find_par_maybe_subfolders(folder)

        if not frames:
            self._fail_folder('No numbered *.cbf frames found in this folder.')
            return
        if par is None:
            self._fail_folder(
                f'Found {self._n_total} .cbf frames but no *_cracker.par / *.par. '
                f'If the par is in a subfolder, tick "Search subfolders for the .par file" '
                f'in the Advanced tab.')
            return

        try:
            det, ang0, _ = rawrecon.FlatDetector.from_eiger_cbf(frames[0])
            UB, wl = rawrecon.read_crysalis_par(par)
            from .rawrecon.engine import _reciprocal_cell
            cell = _reciprocal_cell(UB)
        except Exception as e:
            self._fail_folder(f'Could not read geometry: {e}')
            return

        self._par = par
        self._recip = np.asarray(UB, float)
        phi0 = ang0['Phi']; incr = ang0['Angle_increment']
        stem = os.path.basename(frames[0]).rsplit('_', 1)[0]
        info = (f'{self._n_total} frames ({stem}_*.cbf)  |  '
                f'{det.shape[1]} x {det.shape[0]} px  |  '
                f'lambda = {det.wavelength:.5f} A  |  dist {det.distance:.1f} mm\n'
                f'phi {phi0:.1f} -> {phi0 + (self._n_total-1)*incr:.1f} deg (step {incr})\n'
                f'cell a={cell["a"]:.4f} b={cell["b"]:.4f} c={cell["c"]:.4f} A  '
                f'alpha={cell["alpha"]:.2f} beta={cell["beta"]:.2f} gamma={cell["gamma"]:.2f}')
        try:
            qmax, reach = data_reach(det, cell)
            info += (f'\ndata range (detector corners, header geometry): d_min = '
                     f'{1.0 / qmax:.3f} A, |h| <= {reach[0]:.2f}, '
                     f'|k| <= {reach[1]:.2f}, |l| <= {reach[2]:.2f}')
        except Exception:
            pass
        self.info_label.setText(info)
        self._log(f'  par: {os.path.relpath(par, folder)}')
        for line in info.split('\n'):
            self._log('  ' + line)
        self.nframes.setMaximum(max(self._n_total, 1))
        self.run_btn.setEnabled(True)
        self._check_laue(switch=True)
        self._update_files_label()

    def _fail_folder(self, msg):
        self.info_label.setText(msg)
        self._log('  ERROR: ' + msg)
        self.run_btn.setEnabled(False)
        self._check_laue()
        self._update_files_label()

    def _find_par_maybe_subfolders(self, folder):
        """Search subfolders for a .par only if the user enabled it (the top
        folder was already searched by find_crysalis_par and came up empty)."""
        if self.subfolder_par_chk.isChecked():
            return rawrecon.find_crysalis_par(folder, recursive=True)
        return None

    def _on_subfolder_toggled(self, _checked):
        if self._folder:
            self._set_folder(self._folder)          # re-detect with the new setting

    def _check_laue(self, switch=False):
        """Show a note if the cell metric does not allow the Laue group.

        With `switch`, select the allowed group of highest order instead."""
        if self._recip is None:
            self.laue_note.setVisible(False)
            return
        laue = self.laue_combo.currentData()
        residual, fit = fitting_laue_group(self._recip, laue)
        if fit == laue:
            self.laue_note.setVisible(False)
            return
        if switch:
            self.laue_combo.setCurrentIndex(self.laue_combo.findData(fit))
            text = (f'The cell metric does not allow {laue} (residual {residual:.2g}). The '
                    f'window selected {fit}: the group of highest order that the cell allows. '
                    'Check the Laue group.')
            self._log('  Laue group: ' + text)
            style = 'color:#a05000; font-weight:bold;'
        else:
            text = (f'The cell metric does not allow {laue} (residual {residual:.2g}, limit '
                    f'{METRIC_TOLERANCE:g}). The group of highest order that the cell allows '
                    f'is {fit}.')
            style = 'color:#a00; font-weight:bold;'
        self.laue_note.setText(text)
        self.laue_note.setStyleSheet(style)
        self.laue_note.setVisible(True)

    # ── output ──
    def _browse_out_dir(self):
        start = self._out_dir or self._folder or ''
        folder = QFileDialog.getExistingDirectory(self, 'Select the output folder', start)
        if folder:
            self._out_dir = folder
            self._raw_saved = None
            self._update_out_label()

    def _update_out_label(self):
        if self._out_dir:
            self.out_label.setText(self._out_dir)
        else:
            self.out_label.setText('the dataset folder' if not self._folder
                                   else f'{self._folder}  (the dataset folder)')
        self._update_files_label()

    def _output_paths(self, laue, tag):
        """(symmetrised file, raw file or None) in the output folder."""
        out = self._out_dir or self._folder
        name = ((self._info or {}).get('name')
                or os.path.basename(os.path.normpath(self._folder)))
        suffix = laue.replace('/', '').replace('-', 'bar')
        sym = os.path.join(out, f'{name}_recon_sym_{suffix}{tag}.h5')
        raw = (os.path.join(out, f'{name}_recon_raw{tag}.h5')
               if self.save_raw_chk.isChecked() else None)
        return sym, raw

    def _update_files_label(self, *_):
        if not self._folder:
            self.files_label.setText('')
            return
        sym, raw = self._output_paths(self.laue_combo.currentData(), self._source()[1])
        names = [os.path.basename(sym)] + ([os.path.basename(raw)] if raw else [])
        self.files_label.setText('Files: ' + ',  '.join(names))

    def _confirm_overwrite(self, paths):
        """True if no file in `paths` exists, or if the user agrees to overwrite them."""
        existing = [p for p in paths if p and os.path.exists(p)]
        if not existing:
            return True
        names = '\n'.join(os.path.basename(p) for p in existing)
        answer = QMessageBox.question(
            self, 'Overwrite files',
            f'These files exist in the output folder:\n{names}\n\nOverwrite them?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        return answer == QMessageBox.StandardButton.Yes

    # ── run: reconstruct (if necessary), symmetrise and save ──
    def _corrections(self):
        return ({'solid_angle': True, 'polarization': True,
                 'polarization_fraction': self.pol_frac.value()}
                if self.corr_chk.isChecked() else False)

    def _settings_key(self):
        """The settings that change the reconstruction (not the GPU switch)."""
        return repr((self._folder, self._par, self._ranges(), self.dq.value(), self._bins(),
                     self.nframes.value(), self._corrections(), self.hot.value(),
                     self.tabbin_chk.isChecked()))

    def _grid_ok(self):
        if any(hi <= lo for lo, hi in self._ranges()):
            QMessageBox.warning(self, 'Invalid grid', 'Each axis max must exceed its min.')
            return False
        if 0 in self._grid_counts():
            QMessageBox.warning(self, 'Invalid grid',
                                'Each axis range must contain a multiple of dq x bin.')
            return False
        return True

    def _run(self):
        """Reconstruct and save. Use the reconstruction in memory if its settings are the same."""
        if self._folder is None or self._par is None or not self._grid_ok():
            return
        if self._vd is not None and self._recon_key == self._settings_key():
            self._log('\nThe reconstruction settings did not change. The window uses the '
                      'reconstruction in memory.')
            self._symmetrize_save()
            return
        sym, raw = self._output_paths(self.laue_combo.currentData(), '')
        if not self._confirm_overwrite([sym, raw]):
            return
        self._pending_save = True
        self._reconstruct()

    def _reconstruct(self):
        r = self._ranges()
        corr = self._corrections()
        hot = self.hot.value() or None
        nframes = self.nframes.value() or None
        use_gpu = self.gpu_chk.isChecked()
        self._cancel = False
        self._running_key = self._settings_key()
        self._set_busy(True, 'Reconstructing...', cancellable=True)
        self._log(f'\n{"="*54}\nReconstructing {self._folder}')
        self._retire_worker()
        self._worker = WorkerThread(self._do_reconstruct, self._folder, r,
                                    self.dq.value(), self._bins(), nframes, corr, hot,
                                    use_gpu, self._par, self.tabbin_chk.isChecked())
        self._worker.progress.connect(self._update_progress)
        self._worker.log_msg.connect(self._log)
        self._worker.finished.connect(self._on_recon_done)
        self._worker.cancelled.connect(self._on_recon_cancelled)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _do_reconstruct(self, folder, ranges, step, bins, nframes, corr, hot, use_gpu,
                        par_path, use_tabbin):
        log = lambda m: QThread.currentThread()._emit_log(m)     # type: ignore[union-attr]
        cb = QThread.currentThread()._emit_progress             # type: ignore[union-attr]
        vol, info = rawrecon.reconstruct_dataset(
            folder, ranges=ranges, step=step, bins=bins, nframes=nframes, corrections=corr,
            hot=hot, use_gpu=use_gpu, par_path=par_path, use_tabbin=use_tabbin,
            progress_callback=cb, log=log, keep_setup=True, cancelled=lambda: self._cancel)
        info['bragg_rms'] = rawrecon.bragg_registration_rms(vol)
        provenance = {k: info[k] for k in ('polarization_axis', 'hot_pixel_cutoff',
                                           'geometry_method') if info.get(k) is not None}
        vd = rawrecon.reconstruct_to_volumedata(vol, source_folder=folder,
                                                extra_meta=provenance)
        return vol, vd, info

    def _cancel_run(self):
        if self._busy:
            self._cancel = True
            self._log('Cancel: the reconstruction stops after the current frame.')
            self.cancel_btn.setEnabled(False)

    def _on_recon_cancelled(self):
        self._pending_save = False
        self._log('Reconstruction cancelled. The window keeps the earlier reconstruction, '
                  'if there is one.')
        self._set_busy(False)

    def _on_recon_done(self, result):
        self._vol, self._vd, self._info = result
        self._recon_key = self._running_key
        self._raw_saved = None
        self._frame_stale = False
        self.avg_panel.clear_frame_results('A new reconstruction removed the frame tools results.')
        self._invalidate_binned()
        i = self._info
        self._log(f'\nReconstruction complete:')
        self._log(f'  {i["time_s"]:.1f} s  ({i["ms_per_frame"]:.1f} ms/frame, '
                  f'{"GPU" if i["gpu"] else "CPU"})')
        self._log(f'  frame-1 index rms {i["index_rms"]:.4f} rlu ({i["index_inliers"]} inliers)')
        self._log(f'  measured {i["measured_voxels"]:,}/{i["total_voxels"]:,} voxels '
                  f'({100*i["measured_voxels"]/i["total_voxels"]:.1f}%)  '
                  f'[unmeasured = NaN]')
        rms = i.get('bragg_rms', float('nan'))
        self._log(f'  Bragg registration RMS {rms:.4f} rlu (voxel {self.dq.value()}) '
                  f'- {"OK, sub-voxel" if rms < self.dq.value() else "check orientation"}')
        self.statusBar().showMessage(
            f'Reconstructed {self._vd.intensity.shape} in {i["time_s"]:.0f} s')
        self._set_busy(False)
        if self._pending_save:
            self._pending_save = False
            self._symmetrize_save(confirmed=True)

    # ── symmetrise + save ──
    def _symmetrize_save(self, confirmed=False):
        """Save the raw volume (if ticked), symmetrise and save the volume in memory."""
        if self._busy:
            return
        if self._vd is None:
            msg = 'Reconstruct a volume first: click Reconstruct and save in the Reconstruct tab.'
            self._log(msg)
            self.statusBar().showMessage(msg)
            return
        if self._recon_key is not None and self._recon_key != self._settings_key():
            msg = ('The grid, the frames, the bin factors or the pixel options changed after the '
                   'reconstruction. Click Reconstruct and save to reconstruct again, or set them '
                   'back.')
            self._log(msg)
            QMessageBox.warning(self, 'Settings changed', msg)
            return
        laue = self.laue_combo.currentData()
        vd, tag = self._source()
        sym_path, raw_path = self._output_paths(laue, tag)
        if raw_path is not None and raw_path == self._raw_saved:
            raw_path = None                      # this volume is in that file already
        if not confirmed and not self._confirm_overwrite([sym_path, raw_path]):
            return
        # The Averaging panel: its binned volume and its applied option, if any
        binned = self._current_binned()
        sym_kw, choice = {}, None
        if binned is not None:
            sym_kw = self.avg_panel.symmetrize_kwargs(binned, laue)
            choice = self.avg_panel.choice_metadata() if sym_kw else None
        if sym_kw:
            rule = f"averaging option '{choice['name'] if choice else '?'}' from the Averaging tab"
        else:
            sym_kw = self.avg_settings.kwargs()
            rule = f'averaging {self.avg_settings.describe()}'
        self._set_busy(True, 'Symmetrising...')
        self._retire_worker()
        self._worker = WorkerThread(self._do_sym, vd, laue, sym_kw, rule, self.bin_xy.value(),
                                    self.bin_z.value(), sym_path, binned, choice, raw_path)
        self._worker.progress.connect(self._update_progress)
        self._worker.log_msg.connect(self._log)
        self._worker.finished.connect(self._on_saved)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _do_sym(self, vd, laue, sym_kw, rule, bin_xy, bin_z, path, binned=None, choice=None,
                raw_path=None):
        log = lambda m: QThread.currentThread()._emit_log(m)     # type: ignore[union-attr]
        cb = QThread.currentThread()._emit_progress             # type: ignore[union-attr]
        if raw_path:
            log(f'  saving the raw volume {os.path.basename(raw_path)} ...')
            t = time.time()
            save_volume_h5(raw_path, vd)
            log(f'  wrote {os.path.getsize(raw_path)/1e6:.0f} MB in {time.time()-t:.0f}s')
        if binned is not None:
            if binned is not vd:
                log(f'  use the binned volume of the Averaging panel {binned.intensity.shape}')
            vd = binned
        elif bin_xy > 1 or bin_z > 1:
            log(f'  binning {bin_xy}x{bin_xy}x{bin_z} ...')
            vd = bin_volume(vd, bin_xy, bin_xy, bin_z)
            log(f'  -> {vd.intensity.shape}')
        n_ops = _EXPECTED_ORDERS[laue]
        t = time.time()
        log(f'  symmetrising {laue} ({n_ops} ops), {rule} ...')
        sym = symmetrize_volume(vd, laue, progress_callback=cb, **sym_kw)
        if choice is not None:
            sym.metadata['averaging_choice'] = choice
        n_out = sym.metadata.get('n_outliers_removed', 0)
        log(f'  done in {time.time()-t:.1f}s ({n_out:,} outliers); '
            f'range [{np.nanmin(sym.intensity):.2f}, {np.nanmax(sym.intensity):.2f}]')
        mapping = sym.metadata.get('symmetry_mapping')
        log(f'  symmetry operations applied as: {mapping} maps')
        if mapping != 'exact':
            log('  Not all operations are exact index maps. If you changed the bin\n'
                '  factors after the reconstruction, reconstruct again with them.')
        log('  symmetry diagnostics per |q| shell:')
        log(format_symmetry_diagnostics(sym.metadata.get('symmetry_diagnostics')))
        t = time.time()
        save_volume_h5(path, sym)
        log(f'  wrote {os.path.getsize(path)/1e6:.0f} MB in {time.time()-t:.0f}s')
        return {'sym': path, 'raw': raw_path}

    def _on_saved(self, result):
        self._last_saved = result['sym']
        if result['raw']:
            self._raw_saved = result['raw']
            self._log(f'  saved {result["raw"]}')
        self._log(f'  saved {result["sym"]}\nDone.')
        self._set_busy(False)
        self.statusBar().showMessage(f'Saved {os.path.basename(result["sym"])}')

    # ── Averaging panel ──
    def _source(self):
        """The volume for the Averaging panel and the save steps, and its file-name tag.

        It is the uncorrected volume unless the user selected a corrected volume
        of the frame tools (Averaging tab). The tag follows the metadata value
        'frame_correction' of the volume (for example no ring found: 'frame_model')."""
        vd = self.avg_panel.selected_volume()
        if vd is None or self._vd is None:
            return self._vd, ''
        return vd, {'ring_mask': '_ringmask', 'frame_model': '_framemodel',
                    'ring_mask+frame_model': '_ringmask_framemodel'}.get(
            vd.metadata.get('frame_correction'), '')

    def _binned_key(self):
        return (id(self._source()[0]), self.bin_xy.value(), self.bin_z.value())

    def _current_binned(self):
        """The binned volume of the Averaging panel if it matches the volume and the bins."""
        if (self._vd is not None and self._binned is not None
                and self._binned['key'] == self._binned_key()):
            return self._binned['vol']
        return None

    def _invalidate_binned(self, *_):
        self._binned = None
        self.avg_panel.reset('The volume or the bin factors changed. Click Measure again.')
        self.avg_panel.clear_scores()

    def _averaging_request(self):
        """Laue group and volume function for the Averaging panel (GUI thread)."""
        if self._vd is None:
            raise ValueError('Reconstruct a volume first (Reconstruct tab).')
        laue = self.laue_combo.currentData()
        vd, key = self._source()[0], self._binned_key()
        bin_xy, bin_z = key[1], key[2]
        ready = self._current_binned()

        def volume_fn(log, progress):
            if ready is not None:
                return ready
            vol = vd
            if bin_xy > 1 or bin_z > 1:
                log(f'Bin the volume {bin_xy}x{bin_xy}x{bin_z}.')
                vol = bin_volume(vd, bin_xy, bin_xy, bin_z)
            self._binned = {'key': key, 'vol': vol}
            return vol
        return laue, volume_fn

    # ── frame tools (Averaging tab) ──
    def _frame_request(self):
        """Setup, uncorrected volume and options for the frame tools (GUI thread)."""
        setup = (self._info or {}).get('setup')
        if self._vd is None or setup is None:
            raise ValueError('Reconstruct a volume first. The frame tools use its frames.')
        if self._frame_stale and self._recon_key != self._settings_key():
            raise ValueError('The grid, the frames or the pixel options changed after the '
                             'reconstruction. Reconstruct again to use the frame tools.')
        bin_xy, bin_z = self.bin_xy.value(), self.bin_z.value()

        def bin_fn(vd):
            if bin_xy > 1 or bin_z > 1:
                return bin_volume(vd, bin_xy, bin_xy, bin_z)
            return vd
        return {'setup': setup, 'raw': self._vd, 'laue': self.laue_combo.currentData(),
                'use_gpu': bool(self.gpu_chk.isChecked() and HAS_GPU), 'bin': bin_fn}

    def _on_volume_changed(self):
        """The frame tools selected another volume: release the binned copy."""
        self._binned = None
        self._raw_saved = None
        slot = self.avg_panel.selected_slot()
        names = {'mask': 'ring mask', 'model': 'ring mask + frame model'}
        self.source_note.setText('' if slot == 'none' else
                                 f"Volume: '{names[slot]}' (frame tools, Averaging tab). The save "
                                 'steps use this volume.')
        self.source_note.setVisible(slot != 'none')
        self._update_files_label()

    def _on_laue_changed(self, *_):
        """The Laue group changed: reset the panel and remove the frame-model results."""
        self.avg_panel.reset('The Laue group changed. Click Measure again.')
        self.avg_panel.drop_laue_results(
            "The Laue group changed. The panel removed the 'ring mask + frame model' volume "
            'and the frame map. They use the Laue group. Run Frame model again.')
        self._check_laue()
        self._update_files_label()

    def _on_setup_changed(self, *_):
        """The settings of the reconstruction changed: the frame tools results are stale."""
        if self._vd is None:
            return
        self._frame_stale = True
        self.avg_panel.clear_frame_results('The grid, the frames or the pixel options changed. '
                                           'The frame tools results were removed. Reconstruct '
                                           'again to use the frame tools.')

    def _on_applied_changed(self):
        """An applied averaging option replaces the settings of the Advanced tab."""
        self.avg_settings.set_override(self.avg_panel.applied_name())
        self._update_avg_label()

    def _update_avg_label(self, *_):
        name = self.avg_panel.applied_name()
        if name is not None:
            text = (f"Averaging: '{name}', applied in the Averaging tab. Click Clear in that tab "
                    'to use the settings of the Advanced tab again.')
            self.avg_label.setStyleSheet('color: #070; font-weight: bold;')
        elif self.avg_settings.is_standard():
            text = (f'Averaging: {self.avg_settings.describe()}. Counts weights and MAD outlier '
                    'rejection at 3 sigma: the best option for clean data in the benchmarks. '
                    'Change it in the Advanced tab, or compare the options on your volume in '
                    'the Averaging tab.')
            self.avg_label.setStyleSheet('')
        else:
            text = (f'Averaging: {self.avg_settings.describe()}, set in the Advanced tab.')
            self.avg_label.setStyleSheet('color: #a05000; font-weight: bold;')
        self.avg_label.setText(text)

    def _on_panel_busy(self, busy):
        self.recon_tab.setEnabled(not busy)
        self.adv_tab.setEnabled(not busy)
        self.statusBar().showMessage('Averaging panel is running...' if busy else 'Ready')

    # ── open in viewer ──
    def _open_in_viewer(self):
        if self._last_saved is None or not os.path.exists(self._last_saved):
            return
        try:
            from .rsp_viewer import UnifiedViewer
            self._viewer = UnifiedViewer()
            self._viewer._load_file(self._last_saved)
            self._viewer.show()
            self._log(f'  opened {os.path.basename(self._last_saved)} in the viewer')
        except Exception as e:
            QMessageBox.warning(self, 'Viewer', f'Could not open the viewer:\n{e}')

    # ── helpers ──
    def _retire_worker(self):
        """Wait until the last worker has left run(). Its result signal comes
        before the end of run(), and a thread that still runs must not be deleted."""
        if self._worker is not None:
            self._worker.wait()

    def _update_progress(self, current, total):
        if total > 0:
            self.progress.setMaximum(total); self.progress.setValue(current)
        else:
            self.progress.setMaximum(0)

    def _set_busy(self, busy, message='', cancellable=False):
        self._busy = busy
        for w in (self._dataset_group, self._grid_group, self._out_group, self.adv_tab):
            w.setEnabled(not busy)
        self.run_btn.setEnabled(not busy and self._par is not None)
        self.cancel_btn.setEnabled(busy and cancellable)
        self.view_btn.setEnabled(not busy and self._last_saved is not None)
        self.avg_panel.set_host_busy(busy)
        if message or not busy:
            self.statusBar().showMessage(message or 'Ready')

    def _on_error(self, msg):
        self._log(f'ERROR: {msg}')
        self._pending_save = False
        self._set_busy(False)
        QMessageBox.critical(self, 'Error', str(msg)[:800])

    def _log(self, msg):
        self.log.append(msg)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName('Raw-frame 3D Reconstruction')
    gui = ReconstructGUI()
    gui.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
