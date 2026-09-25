"""
reconstruct_gui.py - GUI to reconstruct a 3D reciprocal-space volume directly from
raw CrysAlisPro rotation frames (.cbf) + the *_cracker.par, then bin / symmetrize /
save with the same machinery as the .img volume builder.

Unlike the volume builder (which loads pre-computed unwarp .img layers), this GUI
computes the reconstruction on the GPU from the raw detector frames:
  cracker.par -> UB,  index frame 1 -> R0,  map every pixel -> (h,k,l),  accumulate.
Unmeasured voxels are known exactly (zero contributing pixels -> NaN), so there is
no coverage-mask guesswork; measured-but-zero voxels stay 0.0.

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
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt

from . import rawrecon
from .volume_builder import (
    bin_volume, symmetrize_volume, save_volume_h5,
    LAUE_GROUP_NAMES, _EXPECTED_ORDERS, HAS_GPU,
)


# ──────────────────────────────────────────────────────────────────
# Worker thread (runs the reconstruction / symmetrisation off the UI thread)
# ──────────────────────────────────────────────────────────────────
class WorkerThread(QThread):
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
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
        try:
            self.finished.emit(self.func(*self.args))
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")


# ──────────────────────────────────────────────────────────────────
# Main GUI
# ──────────────────────────────────────────────────────────────────
class ReconstructGUI(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('Raw-frame 3D Reconstruction')
        self.resize(760, 820)
        self._worker: WorkerThread | None = None
        self._folder: str | None = None
        self._par: str | None = None
        self._n_total: int = 0
        self._vol = None        # rawrecon.Volume
        self._vd = None         # VolumeData (raw, NaN = unmeasured)
        self._info: dict | None = None
        self._viewer = None
        self._gpu_free_gb = 0.0
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
        layout = QVBoxLayout(central)

        # Dataset folder
        g1 = QGroupBox('1. Dataset folder  (raw .cbf frames + *_cracker.par)')
        v1 = QVBoxLayout()
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
        self.subfolder_par_chk = QCheckBox('Search subfolders for the .par file')
        self.subfolder_par_chk.setToolTip(
            'If the *_cracker.par is not in the selected folder itself but in a\n'
            'subfolder, tick this to search recursively (nearest match wins).')
        self.subfolder_par_chk.toggled.connect(self._on_subfolder_toggled)
        v1.addWidget(self.subfolder_par_chk)
        g1.setLayout(v1); layout.addWidget(g1)
        self.info_label = QLabel(''); self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet('color:#555; padding:4px;')
        layout.addWidget(self.info_label)

        # Reconstruction settings
        g2 = QGroupBox('2. Reconstruction grid')
        gl = QGridLayout()
        mk = self._mk_range_spin
        gl.addWidget(QLabel('min'), 0, 1, alignment=Qt.AlignmentFlag.AlignHCenter)
        gl.addWidget(QLabel('max'), 0, 2, alignment=Qt.AlignmentFlag.AlignHCenter)
        gl.addWidget(QLabel('h:'), 1, 0); self.hmin = mk(-6.0); self.hmax = mk(6.0)
        gl.addWidget(self.hmin, 1, 1); gl.addWidget(self.hmax, 1, 2)
        gl.addWidget(QLabel('k:'), 2, 0); self.kmin = mk(-6.0); self.kmax = mk(6.0)
        gl.addWidget(self.kmin, 2, 1); gl.addWidget(self.kmax, 2, 2)
        gl.addWidget(QLabel('l:'), 3, 0); self.lmin = mk(-6.0); self.lmax = mk(6.0)
        gl.addWidget(self.lmin, 3, 1); gl.addWidget(self.lmax, 3, 2)

        gl.addWidget(QLabel('voxel dq (rlu):'), 1, 3)
        self.dq = QDoubleSpinBox(); self.dq.setRange(0.002, 1.0); self.dq.setDecimals(3)
        self.dq.setValue(0.025); self.dq.setSingleStep(0.005)
        self.dq.setToolTip('Voxel size in reciprocal-lattice units, applied to all\n'
                           'three axes. 0.025 rlu resolves Bragg peaks and diffuse.')
        gl.addWidget(self.dq, 1, 4)
        for w in (self.hmin, self.hmax, self.kmin, self.kmax, self.lmin, self.lmax, self.dq):
            w.valueChanged.connect(self._update_estimate)

        gl.addWidget(QLabel('frames (0=all):'), 2, 3)
        self.nframes = QSpinBox(); self.nframes.setRange(0, 100000); self.nframes.setValue(0)
        self.nframes.setToolTip('Number of rotation frames to use. 0 = all frames in\n'
                                'the folder (recommended for full coverage).')
        gl.addWidget(self.nframes, 2, 4)

        gl.addWidget(QLabel('hot-pixel clip:'), 3, 3)
        self.hot = QDoubleSpinBox(); self.hot.setRange(0, 1e9); self.hot.setDecimals(0)
        self.hot.setValue(0); self.hot.setSingleStep(1e5)
        self.hot.setToolTip('Ignore pixels with counts above this (zingers /\n'
                            'saturated). 0 = no clip (default): the Eiger at\n'
                            'I19-2 reaches ~1.3e6 counts in Bragg cores.')
        gl.addWidget(self.hot, 3, 4)
        g2.setLayout(gl); layout.addWidget(g2)

        # Corrections + device
        g3 = QGroupBox('3. Options')
        f3 = QFormLayout()
        self.corr_chk = QCheckBox('Solid-angle + polarisation corrections')
        self.corr_chk.setChecked(True)
        self.corr_chk.setToolTip('Multiply each pixel by the geometric intensity\n'
                                 'correction (1/cos^3(obliquity) x 1/polarisation).\n'
                                 'Lorentz is handled by the per-voxel count normalisation.')
        f3.addRow(self.corr_chk)
        self.pol_frac = QDoubleSpinBox(); self.pol_frac.setRange(0.0, 1.0)
        self.pol_frac.setValue(0.95); self.pol_frac.setSingleStep(0.05); self.pol_frac.setDecimals(2)
        self.pol_frac.setToolTip('Fraction of the beam polarised in the horizontal\n'
                                 '(detector fast) plane. ~0.95 for a synchrotron,\n'
                                 '0.5 = unpolarised.')
        f3.addRow('Polarisation fraction:', self.pol_frac)
        self.gpu_chk = QCheckBox('Use GPU (CuPy)'); self.gpu_chk.setChecked(True)
        self.gpu_chk.setToolTip('Reconstruct on the GPU via CuPy (tens of seconds for\n'
                                'a full scan). Uncheck to force the CPU path.')
        f3.addRow(self.gpu_chk)
        self.tabbin_chk = QCheckBox('Geometry from CrysAlisPro tabbin (robust, any instrument)')
        self.tabbin_chk.setChecked(True)
        self.tabbin_chk.setToolTip(
            'Fit the full geometry (detector pose, oscillation axis, orientation)\n'
            'from the CrysAlisPro *_peakhunt.tabbin ground truth. Instrument-agnostic\n'
            'and robust to a wrong header distance/beam-centre or powder rings.\n\n'
            'Uncheck to instead index frame 1 directly (only works for a clean,\n'
            'correctly-headered dataset with a lab-Y oscillation axis).')
        f3.addRow(self.tabbin_chk)
        self.estimate_label = QLabel(''); self.estimate_label.setStyleSheet('color:#337;')
        f3.addRow('Grid:', self.estimate_label)
        g3.setLayout(f3); layout.addWidget(g3)

        self.recon_btn = QPushButton('Reconstruct')
        self.recon_btn.setEnabled(False); self.recon_btn.setMinimumHeight(38)
        self.recon_btn.setStyleSheet('font-weight:bold; font-size:13px;')
        self.recon_btn.clicked.connect(self._reconstruct)
        layout.addWidget(self.recon_btn)

        # Post-processing (enabled after a reconstruction)
        g4 = QGroupBox('4. Process reconstructed volume')
        f4 = QFormLayout()
        self.laue_combo = QComboBox()
        for key, label in LAUE_GROUP_NAMES.items():
            self.laue_combo.addItem(label, key)
        if 'm-3m' in LAUE_GROUP_NAMES:
            self.laue_combo.setCurrentIndex(list(LAUE_GROUP_NAMES).index('m-3m'))
        self.laue_combo.setToolTip('Laue group for symmetry averaging + outlier\n'
                                   'rejection. m-3m = cubic (MAPbBr3).')
        f4.addRow('Laue group:', self.laue_combo)
        self.sigma = QDoubleSpinBox(); self.sigma.setRange(1.0, 10.0); self.sigma.setValue(3.0)
        self.sigma.setSingleStep(0.5); self.sigma.setDecimals(1)
        self.sigma.setToolTip('Outlier rejection threshold (MAD sigma) across each\n'
                              'symmetry orbit. 3 = standard.')
        f4.addRow('Outlier sigma:', self.sigma)
        brow = QHBoxLayout()
        brow.addWidget(QLabel('H,K:'))
        self.bin_xy = QSpinBox(); self.bin_xy.setRange(1, 16); self.bin_xy.setValue(1)
        brow.addWidget(self.bin_xy); brow.addWidget(QLabel('L:'))
        self.bin_z = QSpinBox(); self.bin_z.setRange(1, 16); self.bin_z.setValue(1)
        brow.addWidget(self.bin_z); brow.addStretch(1)
        f4.addRow('Bin before symmetrise:', brow)

        btns = QHBoxLayout()
        self.save_raw_btn = QPushButton('Save raw volume')
        self.save_raw_btn.clicked.connect(self._save_raw)
        self.save_raw_btn.setToolTip('Save the unsymmetrised reconstruction (NaN =\n'
                                     'unmeasured) as an rspace3d .h5.')
        btns.addWidget(self.save_raw_btn)
        self.sym_btn = QPushButton('Symmetrise + Save')
        self.sym_btn.clicked.connect(self._symmetrize_save)
        self.sym_btn.setToolTip('Bin (optional), symmetry-average with outlier\n'
                                'rejection, and save with a Laue-group suffix.')
        btns.addWidget(self.sym_btn)
        self.view_btn = QPushButton('Open in Viewer')
        self.view_btn.clicked.connect(self._open_in_viewer)
        self.view_btn.setToolTip('Save the raw volume and open it in the RSP viewer.')
        btns.addWidget(self.view_btn)
        f4.addRow(btns)
        g4.setLayout(f4); layout.addWidget(g4)
        self._post_group = g4
        self._set_post_enabled(False)

        self.progress = QProgressBar(); self.progress.setTextVisible(True)
        layout.addWidget(self.progress)
        self.log = QTextEdit(); self.log.setReadOnly(True)
        self.log.setStyleSheet('font-family:Consolas,monospace; font-size:11px;')
        layout.addWidget(self.log)
        self._update_estimate()

    def _mk_range_spin(self, val):
        s = QDoubleSpinBox(); s.setRange(-50, 50); s.setValue(val)
        s.setDecimals(2); s.setSingleStep(1.0); return s

    # ── grid-size estimate ──
    def _ranges(self):
        return ((self.hmin.value(), self.hmax.value()),
                (self.kmin.value(), self.kmax.value()),
                (self.lmin.value(), self.lmax.value()))

    def _update_estimate(self):
        dq = self.dq.value()
        ns = []
        for lo, hi in self._ranges():
            ns.append(max(0, int(round((hi - lo) / dq))) if hi > lo else 0)
        nvox = ns[0] * ns[1] * ns[2]
        # GPU working set during accumulation: float64 sum + int32 count
        gb = nvox * 12 / 1e9
        txt = f'{ns[0]} x {ns[1]} x {ns[2]} = {nvox:,} voxels  (~{gb:.1f} GB GPU)'
        too_big = self._gpu_free_gb and gb > 0.7 * self._gpu_free_gb
        if ns[0] == 0 or ns[1] == 0 or ns[2] == 0:
            txt = 'invalid range (max must exceed min)'
            self.estimate_label.setStyleSheet('color:#a00;')
        else:
            self.estimate_label.setStyleSheet('color:#a00;' if too_big else 'color:#337;')
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
        # post-processing buttons would save/symmetrise the old volume under
        # the new folder's name.
        self._vol = self._vd = self._info = None
        self._set_post_enabled(False)
        self.folder_label.setText(folder)
        self._log(f'\nFolder: {folder}')

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
                f'If the par is in a subfolder, tick "Search subfolders for .par".')
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
        phi0 = ang0['Phi']; incr = ang0['Angle_increment']
        stem = os.path.basename(frames[0]).rsplit('_', 1)[0]
        info = (f'{self._n_total} frames ({stem}_*.cbf)  |  '
                f'{det.shape[1]} x {det.shape[0]} px  |  '
                f'lambda = {det.wavelength:.5f} A  |  dist {det.distance:.1f} mm\n'
                f'phi {phi0:.1f} -> {phi0 + (self._n_total-1)*incr:.1f} deg (step {incr})\n'
                f'cell a={cell["a"]:.4f} b={cell["b"]:.4f} c={cell["c"]:.4f} A  '
                f'alpha={cell["alpha"]:.2f} beta={cell["beta"]:.2f} gamma={cell["gamma"]:.2f}')
        self.info_label.setText(info)
        self._log(f'  par: {os.path.relpath(par, folder)}')
        for line in info.split('\n'):
            self._log('  ' + line)
        self.nframes.setMaximum(max(self._n_total, 1))
        self.recon_btn.setEnabled(True)

    def _fail_folder(self, msg):
        self.info_label.setText(msg)
        self._log('  ERROR: ' + msg)
        self.recon_btn.setEnabled(False)

    def _find_par_maybe_subfolders(self, folder):
        """Search subfolders for a .par only if the user enabled it (the top
        folder was already searched by find_crysalis_par and came up empty)."""
        if self.subfolder_par_chk.isChecked():
            return rawrecon.find_crysalis_par(folder, recursive=True)
        return None

    def _on_subfolder_toggled(self, _checked):
        if self._folder:
            self._set_folder(self._folder)          # re-detect with the new setting

    # ── reconstruct ──
    def _reconstruct(self):
        r = self._ranges()
        if any(hi <= lo for lo, hi in r):
            QMessageBox.warning(self, 'Invalid grid', 'Each axis max must exceed its min.')
            return
        corr = ({'solid_angle': True, 'polarization': True,
                 'polarization_fraction': self.pol_frac.value()}
                if self.corr_chk.isChecked() else False)
        hot = self.hot.value() or None
        nframes = self.nframes.value() or None
        use_gpu = self.gpu_chk.isChecked()
        self._set_busy(True, 'Reconstructing...')
        self._log(f'\n{"="*54}\nReconstructing {self._folder}')
        self._worker = WorkerThread(self._do_reconstruct, self._folder, r,
                                    self.dq.value(), nframes, corr, hot, use_gpu,
                                    self._par, self.tabbin_chk.isChecked())
        self._worker.progress.connect(self._update_progress)
        self._worker.log_msg.connect(self._log)
        self._worker.finished.connect(self._on_recon_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _do_reconstruct(self, folder, ranges, step, nframes, corr, hot, use_gpu,
                        par_path, use_tabbin):
        log = lambda m: QThread.currentThread()._emit_log(m)     # type: ignore[union-attr]
        cb = QThread.currentThread()._emit_progress             # type: ignore[union-attr]
        vol, info = rawrecon.reconstruct_dataset(
            folder, ranges=ranges, step=step, nframes=nframes, corrections=corr,
            hot=hot, use_gpu=use_gpu, par_path=par_path, use_tabbin=use_tabbin,
            progress_callback=cb, log=log)
        info['bragg_rms'] = rawrecon.bragg_registration_rms(vol)
        provenance = {k: info[k] for k in ('polarization_axis', 'hot_pixel_cutoff',
                                           'geometry_method') if info.get(k) is not None}
        vd = rawrecon.reconstruct_to_volumedata(vol, source_folder=folder,
                                                extra_meta=provenance)
        return vol, vd, info

    def _on_recon_done(self, result):
        self._vol, self._vd, self._info = result
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
        self._set_post_enabled(True)

    # ── save raw ──
    def _save_raw(self):
        if self._vd is None:
            return
        default = os.path.join(self._folder, f'{self._info["name"]}_recon_raw.h5')
        path, _ = QFileDialog.getSaveFileName(self, 'Save raw volume', default, 'HDF5 (*.h5)')
        if not path:
            return
        self._set_busy(True, 'Saving...')
        self._worker = WorkerThread(self._do_save, self._vd, path)
        self._worker.log_msg.connect(self._log)
        self._worker.finished.connect(lambda p: (self._log(f'  saved {p}'), self._set_busy(False)))
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _do_save(self, vd, path):
        t = time.time()
        save_volume_h5(path, vd)
        QThread.currentThread()._emit_log(                       # type: ignore[union-attr]
            f'  wrote {os.path.getsize(path)/1e6:.0f} MB in {time.time()-t:.0f}s')
        return path

    # ── symmetrise + save ──
    def _symmetrize_save(self):
        if self._vd is None:
            return
        laue = self.laue_combo.currentData()
        suffix = laue.replace('/', '').replace('-', 'bar')
        default = os.path.join(self._folder, f'{self._info["name"]}_recon_sym_{suffix}.h5')
        path, _ = QFileDialog.getSaveFileName(self, 'Save symmetrised volume', default, 'HDF5 (*.h5)')
        if not path:
            return
        self._set_busy(True, 'Symmetrising...')
        self._worker = WorkerThread(self._do_sym, self._vd, laue, self.sigma.value(),
                                    self.bin_xy.value(), self.bin_z.value(), path)
        self._worker.progress.connect(self._update_progress)
        self._worker.log_msg.connect(self._log)
        self._worker.finished.connect(lambda p: (self._log(f'  saved {p}\nDone.'),
                                                  self._set_busy(False)))
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _do_sym(self, vd, laue, sigma, bin_xy, bin_z, path):
        log = lambda m: QThread.currentThread()._emit_log(m)     # type: ignore[union-attr]
        cb = QThread.currentThread()._emit_progress             # type: ignore[union-attr]
        if bin_xy > 1 or bin_z > 1:
            log(f'  binning {bin_xy}x{bin_xy}x{bin_z} ...')
            vd = bin_volume(vd, bin_xy, bin_xy, bin_z)
            log(f'  -> {vd.intensity.shape}')
        n_ops = _EXPECTED_ORDERS[laue]
        log(f'  symmetrising {laue} ({n_ops} ops), sigma={sigma} ...')
        t = time.time()
        sym = symmetrize_volume(vd, laue, sigma=sigma, progress_callback=cb)
        n_out = sym.metadata.get('n_outliers_removed', 0)
        log(f'  done in {time.time()-t:.1f}s ({n_out:,} outliers); '
            f'range [{np.nanmin(sym.intensity):.2f}, {np.nanmax(sym.intensity):.2f}]')
        t = time.time()
        save_volume_h5(path, sym)
        log(f'  wrote {os.path.getsize(path)/1e6:.0f} MB in {time.time()-t:.0f}s')
        return path

    # ── open in viewer ──
    def _open_in_viewer(self):
        if self._vd is None:
            return
        try:
            path = os.path.join(self._folder, f'{self._info["name"]}_recon_raw.h5')
            save_volume_h5(path, self._vd)
            from .rsp_viewer import UnifiedViewer
            self._viewer = UnifiedViewer()
            self._viewer._load_file(path)
            self._viewer.show()
            self._log(f'  opened {os.path.basename(path)} in the viewer')
        except Exception as e:
            QMessageBox.warning(self, 'Viewer', f'Could not open the viewer:\n{e}')

    # ── helpers ──
    def _update_progress(self, current, total):
        if total > 0:
            self.progress.setMaximum(total); self.progress.setValue(current)
        else:
            self.progress.setMaximum(0)

    def _set_post_enabled(self, on):
        self._post_group.setEnabled(on)

    def _set_busy(self, busy, message=''):
        self.browse_btn.setEnabled(not busy)
        self.recon_btn.setEnabled(not busy and self._par is not None)
        self._post_group.setEnabled(not busy and self._vd is not None)
        if message or not busy:
            self.statusBar().showMessage(message or 'Ready')

    def _on_error(self, msg):
        self._log(f'ERROR: {msg}')
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
