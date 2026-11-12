"""
reconstruct_gui.py - GUI to reconstruct a 3D reciprocal-space volume directly from
raw CrysAlisPro rotation frames (.cbf) + the *_cracker.par, then bin / symmetrize /
save with the same machinery as the .img volume builder.

Unlike the volume builder (which loads pre-computed unwarp .img layers), this GUI
computes the reconstruction on the GPU from the raw detector frames:
  cracker.par -> UB,  index frame 1 -> R0,  map every pixel -> (h,k,l),  accumulate.
Unmeasured voxels are known exactly (zero contributing pixels -> NaN), so there is
no coverage-mask guesswork; measured-but-zero voxels stay 0.0.

Pages (the side bar of the window):
  Reconstruct  folder, Laue group, grid, output and one button that reconstructs,
               symmetrises with the standard averaging and saves.
  Averaging    measure and compare the averaging options.
  Frame tools  ring mask, frame model, frame map and split-half check.
  Advanced     frames, geometry, pixel corrections, bin factors, averaging settings.
  Help         how to use the window and what the options do to the data.

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
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout, QHBoxLayout,
    QLabel, QMessageBox, QSpinBox,
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt

from . import __version__, rawrecon, ui
from .averaging import COUNTS_MAD, COUNTS_MODEL, Cancelled, candidate_kwargs
from .averaging_panel import AveragingPanel, AveragingSettings, engine_name
from .volume_builder import (
    bin_volume, symmetrize_volume, save_volume_h5, format_symmetry_diagnostics,
    fitting_laue_group, LAUE_GROUP_NAMES, METRIC_TOLERANCE, _EXPECTED_ORDERS, HAS_GPU,
)

STANDARD_WHY = (
    f"Standard for raw frames: '{COUNTS_MODEL['name']}'. It removes intensity that is in one "
    'frame only, for example the halo of a strong Bragg reflection. If the noise-model fit '
    f"fails, the app uses '{COUNTS_MAD['name']}'.")
STANDARD_EFFECT = ('Each value has the weight of its pixel count. A value farther than 4 x its '
                   'expected noise from the median of its symmetry-equivalent values is rejected.')
ENGINE_WORDS = {'GPU': 'GPU', 'numba': 'CPU (numba)', 'NumPy': 'CPU (NumPy, slow)'}


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
class ReconstructGUI(ui.Shell):

    def __init__(self) -> None:
        super().__init__('Raw-frame 3D Reconstruction', 'Reconstruction from raw frames')
        ui.start_size(self, 1280, 920)
        self._worker: WorkerThread | None = None
        self._folder: str | None = None
        self._par: str | None = None
        self._n_total: int = 0
        self._vol = None        # rawrecon.Volume
        self._vd = None         # VolumeData (raw, NaN = unmeasured)
        self._info: dict | None = None
        self._viewer = None
        self._viewers: list = []           # the viewer windows that this window opened
        self._grid_note: tuple[str, str] | None = None     # a problem of the grid, for the run note
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

        engine = ENGINE_WORDS[engine_name()]
        if HAS_GPU:
            try:
                import cupy as cp
                props = cp.cuda.runtime.getDeviceProperties(0)
                free, total = cp.cuda.runtime.memGetInfo()
                self._gpu_free_gb = free / 1e9
                nm = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
                self._log(f'GPU: {nm}  ({free/1e9:.1f}/{total/1e9:.1f} GB free)')
                engine = f'GPU ({nm})'
            except Exception:
                self._log('GPU detected')
        else:
            self.gpu_chk.setChecked(False)
            self.gpu_chk.setEnabled(False)
            self._log('No GPU - reconstruction will run on CPU (slow).')
        self.sidebar.footer.setText(f'Engine: {engine}\nrspace3d {__version__}')
        self.set_status('Select a dataset folder (raw .cbf + *_cracker.par)')

    # ── UI ──
    def _build_ui(self):
        self._build_reconstruct_page()
        self._build_averaging_pages()
        self._build_advanced_page()
        self.add_help_page('raw-frames')

        self.laue_combo.currentIndexChanged.connect(self._on_laue_changed)
        for w in (self.hmin, self.hmax, self.kmin, self.kmax, self.lmin, self.lmax, self.dq,
                  self.bin_xy, self.bin_z):
            w.valueChanged.connect(self._update_estimate)
        for w in (self.bin_xy, self.bin_z):
            w.valueChanged.connect(self._invalidate_binned)
            w.valueChanged.connect(self._update_run_note)     # the bins are reconstruction settings
        # A change of the grid, the frames or the pixel options makes the frame
        # tools results of the last reconstruction stale
        for w in (self.hmin, self.hmax, self.kmin, self.kmax, self.lmin, self.lmax, self.dq,
                  self.nframes, self.hot, self.pol_frac):
            w.valueChanged.connect(self._on_setup_changed)
        for w in (self.corr_chk, self.tabbin_chk):
            w.toggled.connect(self._on_setup_changed)
        self.save_raw_chk.toggled.connect(self._update_files_label)
        self.avg_settings.changed.connect(self._update_avg_label)
        ui.guard_wheel(self)
        self._update_estimate()
        self._update_avg_label()
        self._update_out_label()
        self._update_run_note()

    def _build_reconstruct_page(self):
        # 1. Dataset folder
        s1 = ui.Section('Dataset', step=1, help_topic='raw-frames')
        self.folder_label = ui.label('No folder selected', 'strong', wrap=True, selectable=True)
        self.browse_btn = ui.button(
            'Browse...', clicked=self._browse_folder,
            tip='Folder with the raw .cbf rotation frames. The .cbf\n'
                'stem need not match the folder name - it is\n'
                'auto-detected from the numbered frames present.')
        ui.set_role(self.browse_btn, 'primary')      # the first action in a new window
        top = QHBoxLayout()
        top.addWidget(self.folder_label, 1)
        top.addWidget(self.browse_btn, 0, Qt.AlignmentFlag.AlignTop)
        s1.add(top)
        self.dataset_hint = s1.add_hint(
            'Select the folder with the raw .cbf frames. The folder also needs the '
            '*_cracker.par and the *_peakhunt.tabbin of CrysAlisPro.')
        self.dataset_note = s1.add(ui.Banner('error'))
        self.info_grid = ui.FactGrid(['Frames', 'Detector', 'Wavelength', 'Scan', 'Unit cell'])
        self.info_grid.setVisible(False)
        s1.add(self.info_grid)
        self._dataset_group = s1

        # 2. Grid
        s2 = ui.Section('Grid', step=2, help_topic='grid')
        grid = QGridLayout()
        grid.setHorizontalSpacing(ui.SPACE[1])
        grid.setVerticalSpacing(ui.SPACE[1])
        grid.addWidget(ui.label('min', 'caption'), 0, 1)
        grid.addWidget(ui.label('max', 'caption'), 0, 2)
        grid.addWidget(ui.label('data reach', 'caption'), 0, 3)
        self.reach_labels: dict[str, QLabel] = {}
        for i, axis in enumerate('hkl', start=1):
            lo, hi = self._mk_range_spin(-6.0), self._mk_range_spin(6.0)
            lo.setToolTip(f'Smallest {axis} of the grid, in reciprocal-lattice units.')
            hi.setToolTip(f'Largest {axis} of the grid, in reciprocal-lattice units.')
            setattr(self, axis + 'min', lo)
            setattr(self, axis + 'max', hi)
            grid.addWidget(QLabel(axis), i, 0)
            grid.addWidget(lo, i, 1)
            grid.addWidget(hi, i, 2)
            self.reach_labels[axis] = ui.label('', 'caption')
            self.reach_labels[axis].setToolTip(
                'The largest index at the corners of the detector (frame header geometry).\n'
                'A range that is smaller cuts the outer data.')
            grid.addWidget(self.reach_labels[axis], i, 3)
        self.dq = QDoubleSpinBox(); self.dq.setRange(0.002, 1.0); self.dq.setDecimals(3)
        self.dq.setValue(0.025); self.dq.setSingleStep(0.005)
        self.dq.setSuffix(' r.l.u.')
        self.dq.setToolTip('Voxel size in reciprocal-lattice units, applied to all\n'
                           'three axes. 0.025 rlu resolves Bragg peaks and diffuse.')
        grid.addWidget(QLabel('Voxel size'), 4, 0)
        grid.addWidget(self.dq, 4, 1)
        grid.addWidget(ui.label('the same on the three axes', 'caption'), 4, 2, 1, 2)
        self.estimate_label = ui.label('', wrap=True)
        grid.addWidget(ui.label('Grid size', 'caption'), 5, 0, Qt.AlignmentFlag.AlignTop)
        grid.addWidget(self.estimate_label, 5, 1, 1, 4)
        grid.setColumnMinimumWidth(0, 72)
        grid.setColumnStretch(4, 1)
        s2.add(grid)
        s2.add_hint('A range smaller than the data reach cuts the outer data. A smaller voxel '
                    'gives finer detail and a larger volume.')
        self._grid_group = s2

        # 3. Laue group
        s3 = ui.Section('Symmetry', step=3, help_topic='symmetry')
        self.laue_combo = QComboBox()
        for key, name in LAUE_GROUP_NAMES.items():
            self.laue_combo.addItem(name, key)
        if 'm-3m' in LAUE_GROUP_NAMES:
            self.laue_combo.setCurrentIndex(list(LAUE_GROUP_NAMES).index('m-3m'))
        self.laue_combo.setToolTip(
            'Laue group for the symmetry averaging and the outlier rejection.\n'
            'm-3m = cubic (for example MAPbBr3). Select the group of the phase.\n'
            'An experiment that breaks the symmetry (for example an applied field)\n'
            'needs a lower group. The window checks only that the cell metric\n'
            'allows the group: the metric cannot tell a cubic cell from a cell\n'
            'with a small distortion.')
        s3.add(ui.row('Laue group', self.laue_combo))
        s3.add_hint('The average uses this symmetry. Select the group of the phase. An '
                    'experiment that breaks the symmetry needs a lower group.')
        self.laue_note = s3.add(ui.Banner('warn'))
        self._laue_group = s3

        # 4. Averaging (the settings are on the Advanced page)
        s4 = ui.Section('Averaging and outlier rejection', step=4, help_topic='averaging')
        self.avg_label = ui.label('', 'strong', wrap=True)
        s4.add(self.avg_label)
        self.avg_hint = s4.add_hint('')
        s4.add(ui.row(
            ui.button('Change the option', 'link', clicked=lambda: self.show_page('Advanced'),
                      tip='The averaging settings are on the Advanced page.'),
            ui.button('Compare the options on this volume', 'link',
                      clicked=lambda: self.show_page('Averaging'),
                      tip='The Averaging page scores the options on the volume in memory.'),
            spacing=ui.SPACE[3]))
        # The volume of the frame tools, if it is not the uncorrected one
        self.source_note = s4.add(ui.Banner('info'))

        # 5. Output
        s5 = ui.Section('Output', step=5, help_topic='files')
        self.out_label = ui.label('', wrap=True, selectable=True)
        self.out_btn = ui.button('Change...', clicked=self._browse_out_dir,
                                 tip='Folder for the output files. The default is the dataset '
                                     'folder.')
        out = QHBoxLayout()
        out.setSpacing(ui.SPACE[1])
        out.addWidget(ui.label('Folder', 'caption'), 0, Qt.AlignmentFlag.AlignTop)
        out.addWidget(self.out_label, 1)
        out.addWidget(self.out_btn, 0, Qt.AlignmentFlag.AlignTop)
        s5.add(out)
        self.save_raw_chk = QCheckBox('Also save the raw (unsymmetrised) volume')
        self.save_raw_chk.setChecked(True)
        self.save_raw_chk.setToolTip('The raw volume (NaN = unmeasured) lets you symmetrise again\n'
                                     'later, for example with another Laue group, without a new\n'
                                     'reconstruction. It is larger than the symmetrised volume.')
        s5.add(self.save_raw_chk)
        s5.add_hint('With the raw volume you can symmetrise again later without a new '
                    'reconstruction.')
        self.files_label = ui.label('', 'mono', wrap=True, selectable=True)
        s5.add(self.files_label)
        self._out_group = s5

        # The main action, below the page (always visible)
        self.run_bar, bar = ui.footer_bar()
        self.run_note = ui.hint('')
        bar.addWidget(self.run_note, 1)
        self.view_btn = ui.button('Open in viewer', clicked=self._open_in_viewer,
                                  tip='Open the last saved symmetrised volume in the viewer.')
        self.view_btn.setEnabled(False)
        bar.addWidget(self.view_btn)
        self.run_btn = ui.button(
            'Reconstruct and save', 'primary', clicked=self._run,
            tip='Reconstruct, symmetrise with the averaging of step 4 and save\n'
                'the files of step 5. If only the Laue group, the averaging or\n'
                'the output changed, the window uses the reconstruction in\n'
                'memory and does not reconstruct again. (Ctrl+Return)')
        self.run_btn.setEnabled(False)
        self.run_btn.setShortcut('Ctrl+Return')
        bar.addWidget(self.run_btn)
        self.cancel_btn = ui.button('Cancel', clicked=self._cancel_run,
                                    tip='Stop the reconstruction after the current frame.\n'
                                        'The window then keeps the earlier reconstruction.')
        self.cancel_btn.setEnabled(False)
        bar.addWidget(self.cancel_btn)

        self.recon_tab = ui.two_columns([s1, s2], [s3, s4, s5], ratio=(6, 5))
        self.add_page('Reconstruct', 'Reconstruct',
                      'From the raw frames to a symmetry-averaged volume in one step.',
                      self.recon_tab, 'raw-frames', footer=self.run_bar)

    def _build_averaging_pages(self):
        # Averaging panel (works on the volume binned with the factors of the Advanced page)
        self.avg_panel = AveragingPanel(self._averaging_request, log=self._log,
                                        progress=self._update_progress)
        self.avg_panel.busy_changed.connect(self._on_panel_busy)
        self.avg_panel.applied_changed.connect(self._on_applied_changed)
        # Frame tools: raw frames only (the volume builder keeps them hidden)
        self.avg_panel.enable_frame_tools(self._frame_request)
        self.avg_panel.volume_changed.connect(self._on_volume_changed)
        self.avg_panel.set_save_action(
            'Symmetrise and save', self._symmetrize_save,
            'Symmetrise the volume in memory with the applied option (or with\n'
            'the settings of the Advanced page) and save it. The output folder\n'
            'and the file names are those of the Reconstruct page.')
        self.frame_tools_group = self.avg_panel.frame_group
        self.add_page('Averaging', 'Averaging',
                      'Compare the averaging options on the volume in memory, then apply one.',
                      self.avg_panel, 'averaging-page', log=False)
        self.add_page('Frame tools', 'Frame tools',
                      'Find and remove contamination that changes between the frames.',
                      self.frame_tools_group, 'frame-tools', log=False)

    def _build_advanced_page(self):
        ga = ui.Section('Frames and geometry', help_topic='advanced')
        fa = ui.form()
        self.nframes = QSpinBox(); self.nframes.setRange(0, 100000); self.nframes.setValue(0)
        self.nframes.setSpecialValueText('all')
        self.nframes.setToolTip('Number of rotation frames to use. 0 = all frames in\n'
                                'the folder (recommended for full coverage). Use fewer\n'
                                'frames for a quick test.')
        fa.addRow('Frames to use', self.nframes)
        ui.form_hint(fa, 'Use fewer frames for a quick test. 0 means all frames.')
        self.subfolder_par_chk = QCheckBox('Search subfolders for the .par file')
        self.subfolder_par_chk.setToolTip(
            'If the *_cracker.par is not in the selected folder itself but in a\n'
            'subfolder, tick this to search recursively (nearest match wins).')
        self.subfolder_par_chk.toggled.connect(self._on_subfolder_toggled)
        fa.addRow(self.subfolder_par_chk)
        self.tabbin_chk = QCheckBox('Geometry from the CrysAlisPro peak table (recommended)')
        self.tabbin_chk.setChecked(True)
        self.tabbin_chk.setToolTip(
            'Fit the full geometry (detector pose, oscillation axis, orientation)\n'
            'from the CrysAlisPro *_peakhunt.tabbin ground truth. Instrument-agnostic\n'
            'and robust to a wrong header distance/beam-centre or powder rings.\n\n'
            'Uncheck to instead index frame 1 directly (only works for a clean,\n'
            'correctly-headered dataset with a lab-Y oscillation axis).')
        fa.addRow(self.tabbin_chk)
        ui.form_hint(fa, 'The fit uses the peaks of *_peakhunt.tabbin. Untick it only for a '
                         'clean dataset with a correct frame header.')
        self.gpu_chk = QCheckBox('Use the GPU (CuPy)'); self.gpu_chk.setChecked(True)
        self.gpu_chk.setToolTip('Reconstruct on the GPU via CuPy (tens of seconds for\n'
                                'a full scan). Uncheck to force the CPU path.')
        fa.addRow(self.gpu_chk)
        ga.add(fa)

        gb = ui.Section('Pixel corrections', help_topic='advanced')
        fb = ui.form()
        self.corr_chk = QCheckBox('Solid-angle and polarisation corrections')
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
        fb.addRow('Polarisation fraction', self.pol_frac)
        ui.form_hint(fb, 'About 0.95 for a synchrotron, 0.5 for an unpolarised source.')
        self.hot = QDoubleSpinBox(); self.hot.setRange(0, 1e9); self.hot.setDecimals(0)
        self.hot.setValue(0); self.hot.setSingleStep(1e5)
        self.hot.setSpecialValueText('off')
        self.hot.setToolTip('Ignore pixels with counts above this (zingers /\n'
                            'saturated). 0 = no clip (default): the Eiger at\n'
                            'I19-2 reaches ~1.3e6 counts in Bragg cores.')
        fb.addRow('Hot-pixel limit (counts)', self.hot)
        ui.form_hint(fb, 'Pixels with more counts are ignored. A limit also cuts saturated '
                         'Bragg maxima. 0 means no limit.')
        gb.add(fb)

        gc = ui.Section('Binning of the symmetrised volume', help_topic='grid')
        fc = ui.form()
        self.bin_xy = QSpinBox(); self.bin_xy.setRange(1, 16); self.bin_xy.setValue(1)
        self.bin_z = QSpinBox(); self.bin_z.setRange(1, 16); self.bin_z.setValue(1)
        for w in (self.bin_xy, self.bin_z):
            w.setToolTip('Bin factors for the symmetrised volume. Set them before\n'
                         'you reconstruct: the grid puts hkl = 0 on a voxel centre\n'
                         'after this binning, so that the operations of all Laue\n'
                         'groups are exact. The raw volume is not binned.')
        fc.addRow('Bin factor h, k', self.bin_xy)
        fc.addRow('Bin factor l', self.bin_z)
        ui.form_hint(fc, 'Set the factors before you reconstruct. The raw volume is not binned.')
        gc.add(fc)

        self.avg_settings = AveragingSettings(COUNTS_MODEL, STANDARD_WHY)

        self.adv_bar, bar = ui.footer_bar()
        bar.addWidget(ui.hint('The defaults are the standard settings. Change them only if you '
                              'need to.'), 1)
        self.reset_btn = ui.button('Reset to the standard settings', clicked=self._reset_advanced,
                                   tip='Set all settings of this page to the standard values.')
        bar.addWidget(self.reset_btn)
        self.adv_tab = ui.two_columns([ga, gb], [gc, self.avg_settings])
        self.add_page('Advanced', 'Advanced',
                      'Settings that a standard run does not need.', self.adv_tab, 'advanced',
                      footer=self.adv_bar)

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
        problem = True
        self._grid_note = None
        if any(hi <= lo for lo, hi in self._ranges()):
            txt = 'invalid range (max must exceed min)'
            self._grid_note = ('error', 'The grid is not valid: each maximum must be larger '
                               'than its minimum.')
        elif 0 in ns:
            txt = 'range too narrow: it must contain a multiple of dq x bin'
            self._grid_note = ('error', 'The grid is not valid: a range is too narrow for the '
                               'voxel size.')
        else:
            problem = bool(too_big)
            if too_big:
                self._grid_note = ('warn', 'The grid can need more GPU memory than is free. '
                                   'Use narrower ranges or a larger voxel size.')
            if self.bin_xy.value() > 1 or self.bin_z.value() > 1:
                txt += (f'. Bin {self.bin_xy.value()} x {self.bin_xy.value()} x '
                        f'{self.bin_z.value()} for the symmetrised volume (Advanced page)')
            if too_big:
                txt += '  - may exceed GPU memory'
        ui.set_role(self.estimate_label, 'error' if problem else None)
        self.estimate_label.setText(txt)
        self._update_run_note()

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
        self.folder_label.setText(folder.replace('\\', '/'))   # a path with "/" can wrap
        self.folder_label.setToolTip(folder)
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
                f'on the Advanced page.')
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
        phi1 = phi0 + (self._n_total - 1) * incr
        info = (f'{self._n_total} frames ({stem}_*.cbf)  |  '
                f'{det.shape[1]} x {det.shape[0]} px  |  '
                f'lambda = {det.wavelength:.5f} A  |  dist {det.distance:.1f} mm\n'
                f'phi {phi0:.1f} -> {phi1:.1f} deg (step {incr})\n'
                f'cell a={cell["a"]:.4f} b={cell["b"]:.4f} c={cell["c"]:.4f} A  '
                f'alpha={cell["alpha"]:.2f} beta={cell["beta"]:.2f} gamma={cell["gamma"]:.2f}')
        facts = {
            'Frames': f'{self._n_total} ({stem}_*.cbf)',
            'Detector': f'{det.shape[1]} x {det.shape[0]} px, distance {det.distance:.1f} mm',
            'Wavelength': f'{det.wavelength:.5f} Å',
            'Scan': f'phi {phi0:.1f} to {phi1:.1f} deg, step {incr} deg',
            'Unit cell': (f'a = {cell["a"]:.4f}, b = {cell["b"]:.4f}, c = {cell["c"]:.4f} Å\n'
                          f'alpha = {cell["alpha"]:.2f}, beta = {cell["beta"]:.2f}, '
                          f'gamma = {cell["gamma"]:.2f} deg')}
        for lab in self.reach_labels.values():
            lab.setText('')
        try:
            qmax, reach = data_reach(det, cell)
            info += (f'\ndata range (detector corners, header geometry): d_min = '
                     f'{1.0 / qmax:.3f} A, |h| <= {reach[0]:.2f}, '
                     f'|k| <= {reach[1]:.2f}, |l| <= {reach[2]:.2f}')
            facts['Wavelength'] += (f' (d_min = {1.0 / qmax:.3f} Å at the detector '
                                    'corners)')
            for axis, value in zip('hkl', reach):
                self.reach_labels[axis].setText(f'data to ±{value:.2f}')
        except Exception:
            pass
        self.info_grid.set_values(facts)
        self.info_grid.setVisible(True)
        self.dataset_hint.setVisible(False)
        self.dataset_note.clear()
        ui.set_role(self.browse_btn, None)
        self._log(f'  par: {os.path.relpath(par, folder)}')
        for line in info.split('\n'):
            self._log('  ' + line)
        self.nframes.setMaximum(max(self._n_total, 1))
        self.run_btn.setEnabled(True)
        self._check_laue(switch=True)
        self._update_files_label()
        self._update_run_note()
        self.set_status('Dataset folder read. Check the settings, then click Reconstruct and '
                        'save.')

    def _fail_folder(self, msg):
        self.dataset_note.show_message(msg, 'error')
        self.info_grid.setVisible(False)
        self.dataset_hint.setVisible(False)
        for lab in self.reach_labels.values():
            lab.setText('')
        self._log('  ERROR: ' + msg)
        self.run_btn.setEnabled(False)
        self._check_laue()
        self._update_files_label()
        self._update_run_note()

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
            self.laue_note.clear()
            return
        laue = self.laue_combo.currentData()
        residual, fit = fitting_laue_group(self._recip, laue)
        if fit == laue:
            self.laue_note.clear()
            return
        if switch:
            self.laue_combo.setCurrentIndex(self.laue_combo.findData(fit))
            text = (f'The cell metric does not allow {laue} (residual {residual:.2g}). The '
                    f'window selected {fit}: the group of highest order that the cell allows. '
                    'Check the Laue group.')
            self._log('  Laue group: ' + text)
            kind = 'warn'
        else:
            text = (f'The cell metric does not allow {laue} (residual {residual:.2g}, limit '
                    f'{METRIC_TOLERANCE:g}). The group of highest order that the cell allows '
                    f'is {fit}.')
            kind = 'error'
        self.laue_note.show_message(text, kind)

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
            text = self._out_dir
        else:
            text = ('the dataset folder' if not self._folder
                    else f'{self._folder}  (the dataset folder)')
        self.out_label.setText(text.replace('\\', '/'))
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
        self.files_label.setText('\n'.join(names))

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

    def _update_run_note(self, *_):
        """The line next to the main action: why it is off, or what a click does."""
        if self._busy or not hasattr(self, 'run_note'):
            return
        role = 'hint'
        if self._par is None:
            text = 'Select a dataset folder to start.'
        elif self._grid_note is not None:
            role, text = self._grid_note
        elif self._vd is not None and self._recon_key == self._settings_key():
            text = ('The reconstruction is in memory. A click symmetrises and saves without a '
                    'new reconstruction.')
        else:
            text = 'One click reconstructs, symmetrises and saves.'
        ui.set_role(self.run_note, role)
        self.run_note.setText(text)

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
        self._set_busy(False)
        self.set_status(f'Reconstructed {self._vd.intensity.shape} in {i["time_s"]:.0f} s')
        if self._pending_save:
            self._pending_save = False
            self._symmetrize_save(confirmed=True)

    # ── symmetrise + save ──
    def _symmetrize_save(self, confirmed=False):
        """Save the raw volume (if ticked), symmetrise and save the volume in memory."""
        if self._busy:
            return
        if self._vd is None:
            msg = ('Reconstruct a volume first: click Reconstruct and save on the Reconstruct '
                   'page.')
            self._log(msg)
            self.set_status(msg)
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
        sym_kw, choice, standard = {}, None, False
        if binned is not None:
            sym_kw = self.avg_panel.symmetrize_kwargs(binned, laue)
            choice = self.avg_panel.choice_metadata() if sym_kw else None
        if sym_kw:
            rule = (f"averaging option '{choice['name'] if choice else '?'}' from the "
                    'Averaging page')
        else:
            sym_kw = self.avg_settings.kwargs()
            rule = f'averaging {self.avg_settings.describe()}'
            standard = self.avg_settings.is_standard()
        self._set_busy(True, 'Symmetrising...')
        self._retire_worker()
        self._worker = WorkerThread(self._do_sym, vd, laue, sym_kw, rule, self.bin_xy.value(),
                                    self.bin_z.value(), sym_path, binned, choice, raw_path,
                                    standard)
        self._worker.progress.connect(self._update_progress)
        self._worker.log_msg.connect(self._log)
        self._worker.finished.connect(self._on_saved)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _do_sym(self, vd, laue, sym_kw, rule, bin_xy, bin_z, path, binned=None, choice=None,
                raw_path=None, standard=False):
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
        try:
            sym = symmetrize_volume(vd, laue, progress_callback=cb, **sym_kw)
        except ValueError as exc:
            # The standard needs a noise model. Without it, use the spread limit.
            if not standard or 'noise-model fit failed' not in str(exc):
                raise
            log(f"  The noise-model fit failed. Use '{COUNTS_MAD['name']}'. ({exc})")
            sym = symmetrize_volume(vd, laue, progress_callback=cb,
                                    **candidate_kwargs(COUNTS_MAD))
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
        name = os.path.basename(result['sym'])
        self.set_status(f'Saved {name}')
        ui.set_role(self.run_note, 'ok')
        self.run_note.setText(f'Saved {name}. Click Open in viewer to look at it.')

    # ── Averaging panel ──
    def _source(self):
        """The volume for the Averaging panel and the save steps, and its file-name tag.

        It is the uncorrected volume unless the user selected a corrected volume
        of the frame tools. The tag follows the metadata value
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
            raise ValueError('Reconstruct a volume first (Reconstruct page).')
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

    # ── frame tools ──
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
        self.source_note.show_message(
            '' if slot == 'none' else
            f"Volume: '{names[slot]}' (Frame tools page). The save steps use this volume.")
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
        self._update_run_note()
        if self._vd is None:
            return
        self._frame_stale = True
        self.avg_panel.clear_frame_results('The grid, the frames or the pixel options changed. '
                                           'The frame tools results were removed. Reconstruct '
                                           'again to use the frame tools.')

    def _on_applied_changed(self):
        """An applied averaging option replaces the settings of the Advanced page."""
        self.avg_settings.set_override(self.avg_panel.applied_name())
        self._update_avg_label()

    def _update_avg_label(self, *_):
        name = self.avg_panel.applied_name()
        if name is not None:
            text = f"'{name}', applied on the Averaging page"
            note = ('Click Clear on the Averaging page to use the settings of the Advanced page '
                    'again.')
            role = 'ok'
        elif self.avg_settings.is_standard():
            text = self.avg_settings.describe()
            note = STANDARD_EFFECT
            role = 'strong'
        else:
            text = f'{self.avg_settings.describe()}, set on the Advanced page'
            note = 'This is not the standard option for raw frames.'
            role = 'warn'
        ui.set_role(self.avg_label, role)
        self.avg_label.setText(text)
        self.avg_hint.setText(note)

    def _on_panel_busy(self, busy):
        for w in (self.recon_tab, self.adv_tab, self.run_bar, self.adv_bar):
            w.setEnabled(not busy)
        self.set_status('The Averaging panel is running...' if busy else 'Ready')
        if busy:                                    # say why the settings are off
            ui.set_role(self.run_note, 'hint')
            self.run_note.setText('The Averaging or the Frame tools page is running. The '
                                  'settings are off until it stops.')
        else:
            self._update_run_note()

    # ── open in viewer ──
    def _open_in_viewer(self):
        if self._last_saved is None or not os.path.exists(self._last_saved):
            return
        try:
            from .rsp_viewer import UnifiedViewer
            self._viewer = UnifiedViewer()
            self._viewers.append(self._viewer)       # an earlier viewer window stays open
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

    def _set_busy(self, busy, message='', cancellable=False):
        self._busy = busy
        for w in (self._dataset_group, self._grid_group, self._laue_group, self._out_group,
                  self.adv_tab, self.adv_bar):
            w.setEnabled(not busy)
        self.run_btn.setEnabled(not busy and self._par is not None)
        self.cancel_btn.setEnabled(busy and cancellable)
        self.view_btn.setEnabled(not busy and self._last_saved is not None)
        self.avg_panel.set_host_busy(busy)
        if message or not busy:
            self.set_status(message or 'Ready')
        if busy:
            ui.set_role(self.run_note, 'hint')
            self.run_note.setText(message + ' The log shows the steps.')
        else:
            self._update_run_note()

    def _on_error(self, msg):
        self._log(f'ERROR: {msg}')
        self._pending_save = False
        self._set_busy(False)
        ui.set_role(self.run_note, 'error')
        self.run_note.setText('The run stopped. The log has the error text.')
        ui.show_error(self, 'The run stopped', msg)


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName('Raw-frame 3D Reconstruction')
    gui = ReconstructGUI()
    gui.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
