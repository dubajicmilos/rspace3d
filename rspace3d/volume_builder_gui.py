"""
volume_builder_gui.py — Simplified one-button 3D volume processing GUI.

Select unwarp folder → Process All:
  1. Load all .img files into 3D volume
  2. Save raw volume as .h5
  3. Bin 2x2 (HK), reject outliers, symmetrize
  4. Save processed volume as .h5 with Laue group suffix

Usage:
    python volume_builder_gui.py
"""

import sys
import os
import time
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QFormLayout, QPushButton, QLabel, QComboBox,
    QSpinBox, QDoubleSpinBox, QProgressBar, QFileDialog,
    QStatusBar, QTextEdit, QMessageBox,
)
from PyQt6.QtCore import QThread, pyqtSignal

from .volume_builder import (
    VolumeData, load_unwarp_folder, bin_volume,
    reject_outliers, symmetrize_volume,
    save_volume_h5, _read_header_fast,
    find_par_file, read_par_cell, cell_from_ub, _filter_numbered_imgs,
    LAUE_GROUP_NAMES, _EXPECTED_ORDERS, HAS_GPU,
)
from .make_dcunwarp import generate_dcunwarp


# ──────────────────────────────────────────────────────────────────
# Worker thread
# ──────────────────────────────────────────────────────────────────

class WorkerThread(QThread):
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    log_msg = pyqtSignal(str)

    def __init__(self, func, *args):
        super().__init__()
        self.func = func
        self.args = args

    def _emit_progress(self, current, total):
        self.progress.emit(current, total)

    def _emit_log(self, msg):
        self.log_msg.emit(msg)

    def run(self):
        try:
            result = self.func(*self.args)
            self.finished.emit(result)
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")


# ──────────────────────────────────────────────────────────────────
# Main GUI
# ──────────────────────────────────────────────────────────────────

class SimpleVolumeGUI(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('3D Volume Processor')
        self.resize(700, 700)
        self._worker = None
        self._folder_path = None
        self._build_ui()

        if HAS_GPU:
            try:
                import cupy as cp
                name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
                self._log(f'GPU detected: {name}')
            except Exception:
                self._log('GPU detected')
        else:
            self._log('No GPU — using CPU')

        self.statusBar().showMessage('Select an unwarp folder to begin')

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # ── Folder selection ──
        g1 = QGroupBox('Select Unwarp Folder')
        h1 = QHBoxLayout()
        self.folder_label = QLabel('No folder selected')
        self.folder_label.setWordWrap(True)
        h1.addWidget(self.folder_label, stretch=1)
        self.browse_btn = QPushButton('Browse...')
        self.browse_btn.clicked.connect(self._browse_folder)
        self.browse_btn.setToolTip(
            'Select the folder containing CrysAlisPro unwarp .img files.\n'
            'These are HK reciprocal space reconstructions at different l values.')
        h1.addWidget(self.browse_btn)
        g1.setLayout(h1)
        layout.addWidget(g1)

        self.info_label = QLabel('')
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet('color: #555; padding: 4px;')
        layout.addWidget(self.info_label)

        # ── Generate dcunwarp ──
        g_dc = QGroupBox('Generate dcunwarp file (for CrysAlisPro)')
        g_dc.setToolTip(
            'Generate a .dcunwarp layer list file for CrysAlisPro.\n'
            'Use this BEFORE processing to create HK reconstructions\n'
            'at each l value via the "dc unwarp" command in CrysAlisPro.')
        dc_layout = QHBoxLayout()

        dc_layout.addWidget(QLabel('l min:'))
        self.dc_lmin = QDoubleSpinBox()
        self.dc_lmin.setRange(-50, 50); self.dc_lmin.setValue(-6.0)
        self.dc_lmin.setDecimals(2); self.dc_lmin.setSingleStep(1.0)
        self.dc_lmin.setToolTip('Starting l value (most negative)')
        dc_layout.addWidget(self.dc_lmin)

        dc_layout.addWidget(QLabel('l max:'))
        self.dc_lmax = QDoubleSpinBox()
        self.dc_lmax.setRange(-50, 50); self.dc_lmax.setValue(6.0)
        self.dc_lmax.setDecimals(2); self.dc_lmax.setSingleStep(1.0)
        self.dc_lmax.setToolTip('Ending l value (most positive)')
        dc_layout.addWidget(self.dc_lmax)

        dc_layout.addWidget(QLabel('step:'))
        self.dc_step = QDoubleSpinBox()
        self.dc_step.setRange(0.001, 1.0); self.dc_step.setValue(0.02)
        self.dc_step.setDecimals(3); self.dc_step.setSingleStep(0.01)
        self.dc_step.setToolTip('Step size in l (r.l.u.). Typical: 0.02-0.05')
        dc_layout.addWidget(self.dc_step)

        dc_layout.addWidget(QLabel('res:'))
        self.dc_res = QDoubleSpinBox()
        self.dc_res.setRange(0.1, 5.0); self.dc_res.setValue(0.8)
        self.dc_res.setDecimals(1); self.dc_res.setSingleStep(0.1)
        self.dc_res.setToolTip('Resolution limit d_min in Angstroms.\nSmaller = more data but larger files.')
        dc_layout.addWidget(self.dc_res)

        self.dc_btn = QPushButton('Generate')
        self.dc_btn.clicked.connect(self._generate_dcunwarp)
        self.dc_btn.setToolTip('Create .dcunwarp file in the selected folder')
        dc_layout.addWidget(self.dc_btn)

        g_dc.setLayout(dc_layout)
        layout.addWidget(g_dc)

        # ── Processing settings + one button ──
        g2 = QGroupBox('Process')
        f2 = QFormLayout()

        self.laue_combo = QComboBox()
        for key, label in LAUE_GROUP_NAMES.items():
            self.laue_combo.addItem(label, key)
        self.laue_combo.setCurrentIndex(list(LAUE_GROUP_NAMES.keys()).index('m-3m'))
        self.laue_combo.setToolTip(
            'Laue group for symmetrization and outlier rejection.\n'
            'Determines which symmetry operations are applied to average\n'
            'equivalent reflections in reciprocal space.\n\n'
            'Common choices:\n'
            '  m-3m  — cubic (e.g. MAPbBr3, CsPbCl3)\n'
            '  4/mmm — tetragonal\n'
            '  mmm   — orthorhombic\n'
            '  2/m   — monoclinic\n'
            '  -1    — triclinic (inversion only, no symmetry averaging)')
        f2.addRow('Laue group:', self.laue_combo)

        bin_row = QHBoxLayout()
        bin_row.addWidget(QLabel('H,K:'))
        self.bin_xy_spin = QSpinBox()
        self.bin_xy_spin.setRange(1, 16); self.bin_xy_spin.setValue(2)
        self.bin_xy_spin.setToolTip(
            'Bin factor for H and K axes.\n'
            'Averages NxN pixels to reduce volume size and noise.\n'
            '1 = no binning, 2 = 2x2 (4x smaller), 4 = 4x4 (16x smaller)')
        bin_row.addWidget(self.bin_xy_spin)
        bin_row.addWidget(QLabel('L:'))
        self.bin_z_spin = QSpinBox()
        self.bin_z_spin.setRange(1, 16); self.bin_z_spin.setValue(1)
        self.bin_z_spin.setToolTip(
            'Bin factor for L axis (stacking direction).\n'
            '1 = keep all L slices, 2 = average pairs, etc.')
        bin_row.addWidget(self.bin_z_spin)
        f2.addRow('Binning:', bin_row)

        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setRange(1.0, 10.0)
        self.sigma_spin.setValue(3.0)
        self.sigma_spin.setSingleStep(0.5)
        self.sigma_spin.setDecimals(1)
        self.sigma_spin.setToolTip(
            'Outlier rejection threshold in units of robust standard deviation.\n'
            'Voxels deviating more than N*sigma from the symmetrized mean\n'
            'are replaced with the symmetrized value.\n\n'
            '3.0 = standard (removes ~2-5% of voxels)\n'
            '5.0 = conservative (removes only extreme outliers)\n'
            '2.0 = aggressive (removes more, may affect weak features)')
        f2.addRow('Outlier sigma:', self.sigma_spin)

        self.niter_spin = QSpinBox()
        self.niter_spin.setRange(1, 5)
        self.niter_spin.setValue(1)
        self.niter_spin.setToolTip(
            'Number of outlier rejection iterations.\n'
            'Each iteration: symmetrize, find outliers, replace, repeat.\n'
            '1 = usually sufficient, 2 = for noisy data with many hot pixels')
        f2.addRow('Rejection iterations:', self.niter_spin)

        self.process_btn = QPushButton('Process All')
        self.process_btn.setEnabled(False)
        self.process_btn.setMinimumHeight(40)
        self.process_btn.setStyleSheet('font-weight: bold; font-size: 13px;')
        self.process_btn.clicked.connect(self._process_all)
        self.process_btn.setToolTip(
            'Run the full pipeline:\n'
            '1. Load all .img files into 3D volume\n'
            '2. Save raw volume as .h5\n'
            '3. Bin the volume\n'
            '4. Reject outliers using symmetry equivalents\n'
            '5. Symmetrize (average equivalent reflections)\n'
            '6. Save processed volume as .h5')
        f2.addRow(self.process_btn)

        g2.setLayout(f2)
        layout.addWidget(g2)

        # ── Progress + log ──
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        layout.addWidget(self.progress)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet('font-family: Consolas, monospace; font-size: 11px;')
        layout.addWidget(self.log)

    # ── Folder ──

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, 'Select Unwarp Folder',
            'F:\\' if os.path.exists('F:\\') else '')
        if folder:
            self._set_folder(folder)

    def _set_folder(self, folder):
        self._folder_path = folder
        self.folder_label.setText(folder)
        self._log(f'Folder: {folder}')

        img_files = _filter_numbered_imgs(folder)
        n = len(img_files)
        if n == 0:
            self.info_label.setText('No .img files found.')
            self.process_btn.setEnabled(False)
            return

        def _num(f):
            try: return int(f.rsplit('_', 1)[1].split('.')[0])
            except: return 0
        sorted_f = sorted(img_files, key=_num)

        # Store prefix for output filenames
        self._img_prefix = sorted_f[0].rsplit('_', 1)[0]
        self._log(f'Prefix: {self._img_prefix} ({n} files)')

        hdr = _read_header_fast(os.path.join(folder, sorted_f[0]))
        nx, ny = hdr['nx'], hdr['ny']
        l_min = _read_header_fast(os.path.join(folder, sorted_f[0]))['fixed_value']
        l_max = _read_header_fast(os.path.join(folder, sorted_f[-1]))['fixed_value']
        l_step = (l_max - l_min) / max(n - 1, 1)

        info = (f'{n} files  |  {nx} x {ny} px  |  {hdr["plane_type"]}  |  '
                f'lambda = {hdr["wavelength"]:.5f} A\n'
                f'Fixed axis: {l_min:.3f} to {l_max:.3f} (step {l_step:.4f})')

        # Unit cell: try par file first, fall back to .img header UB
        par_path = find_par_file(folder)
        cell = None
        if par_path:
            self._log(f'Par file: {os.path.basename(par_path)}')
            cell = read_par_cell(par_path)
        if cell is None:
            cell = cell_from_ub(hdr['ub'], hdr['wavelength'])
            self._log('Cell computed from .img header UB matrix')

        info += (f'\nCell: a={cell["a"]:.5f}  b={cell["b"]:.5f}  '
                 f'c={cell["c"]:.5f} A  |  '
                 f'alpha={cell["alpha"]:.3f}  beta={cell["beta"]:.3f}  '
                 f'gamma={cell["gamma"]:.3f} deg')

        self.info_label.setText(info)
        self._log(info)

        # Estimate sizes
        nx2, ny2 = nx // 2, ny // 2
        raw_mb = nx * ny * n * 4 / 1e6
        bin_mb = nx2 * ny2 * n * 4 / 1e6
        self._log(f'Raw volume: {nx} x {ny} x {n} = {raw_mb:.0f} MB')
        self._log(f'Binned 2x2: {nx2} x {ny2} x {n} = {bin_mb:.0f} MB')

        self._file_count = n
        self._frame_nx = nx
        self._frame_ny = ny
        self.process_btn.setEnabled(True)

    # ── Generate dcunwarp ──

    def _generate_dcunwarp(self):
        if self._folder_path:
            out_dir = self._folder_path
        else:
            out_dir = QFileDialog.getExistingDirectory(
                self, 'Select output folder for .dcunwarp file')
            if not out_dir:
                return

        l_min = self.dc_lmin.value()
        l_max = self.dc_lmax.value()
        l_step = self.dc_step.value()
        res = self.dc_res.value()
        n_layers = len(np.arange(l_min, l_max + l_step * 0.5, l_step))
        n_files = int(np.ceil(n_layers / 500))

        self._log(f'Generating dcunwarp: l=[{l_min}, {l_max}], step={l_step}, '
                  f'res={res}, {n_layers} layers')

        old_cwd = os.getcwd()
        try:
            os.chdir(out_dir)
            filenames = generate_dcunwarp(l_min, l_max, l_step, res)
        finally:
            os.chdir(old_cwd)

        for fn in filenames:
            self._log(f'  Created: {os.path.join(out_dir, fn)}')

        if n_files > 1:
            msg = (f'{n_files} files were created because CrysAlisPro can only '
                   f'process 500 layers at a time.\n\n'
                   f'You need to run "dc unwarp" {n_files} times in CrysAlisPro, '
                   f'once with each file.')
            self._log(f'  WARNING: {n_files} files — CrysAlisPro 500-layer limit')
            QMessageBox.warning(self, 'Multiple dcunwarp files', msg)

    # ── Process All ──

    def _process_all(self):
        laue = self.laue_combo.currentData()
        sigma = self.sigma_spin.value()
        niter = self.niter_spin.value()
        bin_xy = self.bin_xy_spin.value()
        bin_z = self.bin_z_spin.value()
        device = 'GPU' if HAS_GPU else 'CPU'

        self._set_busy(True, 'Processing...')
        self._log(f'\n{"="*50}')
        self._log(f'Processing: {laue}, sigma={sigma}, iter={niter}, '
                  f'bin={bin_xy}x{bin_xy}x{bin_z}, {device}')
        self._log(f'{"="*50}')

        self._worker = WorkerThread(
            self._do_process_all,
            self._folder_path, laue, sigma, niter, bin_xy, bin_z)
        self._worker.progress.connect(self._update_progress)
        self._worker.log_msg.connect(self._log)
        self._worker.finished.connect(self._on_process_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _do_process_all(self, folder, laue, sigma, niter, bin_xy, bin_z):
        log = lambda msg: QThread.currentThread()._emit_log(msg)
        cb = QThread.currentThread()._emit_progress
        device = 'GPU' if HAS_GPU else 'CPU'

        # Derive base name from .img filename prefix
        # e.g. "MAPbI2Br_293K_1_1_0_1.img" -> prefix "MAPbI2Br_293K_1_1_0"
        img_files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.img')])
        first_img = img_files[0]
        # Strip trailing _<number>.img
        base = first_img.rsplit('_', 1)[0]

        # ── Step 1: Load raw ──
        log(f'Step 1/4: Loading all .img files...')
        t0 = time.time()
        vol = load_unwarp_folder(folder, bin_xy=1, progress_callback=cb)
        dt = time.time() - t0
        nh, nk, nl = vol.intensity.shape
        log(f'  Loaded: {nh} x {nk} x {nl} ({vol.intensity.dtype}) in {dt:.1f}s')
        log(f'  H=[{vol.H[0]:.3f}, {vol.H[-1]:.3f}]  '
            f'K=[{vol.K[0]:.3f}, {vol.K[-1]:.3f}]  '
            f'L=[{vol.L[0]:.3f}, {vol.L[-1]:.3f}]')

        # ── Step 2: Save raw ──
        raw_path = os.path.join(folder, f'{base}_raw.h5')
        log(f'Step 2/4: Saving raw volume...')
        t0 = time.time()
        save_volume_h5(raw_path, vol)
        dt = time.time() - t0
        size_mb = os.path.getsize(raw_path) / 1e6
        log(f'  Saved: {raw_path}')
        log(f'  {size_mb:.1f} MB on disk (gzip) in {dt:.1f}s')

        # ── Step 3: Bin ──
        if bin_xy > 1 or bin_z > 1:
            log(f'Step 3/4: Binning {bin_xy}x{bin_xy} (HK), {bin_z}x (L)...')
            vol = bin_volume(vol, bin_xy, bin_xy, bin_z)
            nh, nk, nl = vol.intensity.shape
            log(f'  Binned: {nh} x {nk} x {nl}')
        else:
            log(f'Step 3/4: No binning (1x1x1)')

        # ── Step 4: Reject + Symmetrize ──
        n_ops = _EXPECTED_ORDERS[laue]
        log(f'Step 4/4: Outlier rejection + symmetrization ({device})...')
        log(f'  Laue group: {laue} ({n_ops} ops)')
        log(f'  Sigma: {sigma}, iterations: {niter}')

        t0 = time.time()
        log(f'  Rejecting outliers...')
        vol = reject_outliers(vol, laue, sigma=sigma, n_iter=niter,
                              progress_callback=cb)
        dt_rej = time.time() - t0
        n_repl = vol.metadata.get('n_outliers_replaced', 0)
        log(f'  Outlier rejection: {dt_rej:.1f}s, {n_repl:,} voxels replaced')

        t0 = time.time()
        log(f'  Symmetrizing...')
        vol = symmetrize_volume(vol, laue, progress_callback=cb)
        dt_sym = time.time() - t0
        log(f'  Symmetrization: {dt_sym:.1f}s')
        log(f'  Range: [{vol.intensity.min():.1f}, {vol.intensity.max():.1f}]')

        # Save with Laue group suffix
        laue_suffix = laue.replace('/', '').replace('-', 'bar')
        sym_path = os.path.join(folder, f'{base}_sym_{laue_suffix}.h5')
        log(f'  Saving: {sym_path}')
        t0 = time.time()
        save_volume_h5(sym_path, vol)
        dt_save = time.time() - t0
        size_mb = os.path.getsize(sym_path) / 1e6
        log(f'  {size_mb:.1f} MB on disk in {dt_save:.1f}s')

        return {'raw_path': raw_path, 'sym_path': sym_path}

    def _on_process_done(self, result):
        self._log(f'\nDone!')
        self._log(f'  Raw: {result["raw_path"]}')
        self._log(f'  Sym: {result["sym_path"]}')
        self._set_busy(False)

    # ── Helpers ──

    def _update_progress(self, current, total):
        if total > 0:
            self.progress.setMaximum(total)
            self.progress.setValue(current)
        else:
            self.progress.setMaximum(0)

    def _set_busy(self, busy, message=''):
        self.browse_btn.setEnabled(not busy)
        self.process_btn.setEnabled(not busy and self._folder_path is not None)
        self.dc_btn.setEnabled(not busy)
        self.statusBar().showMessage(message if message else ('Ready' if not busy else ''))

    def _on_error(self, msg):
        self._log(f'ERROR: {msg}')
        self._set_busy(False)
        QMessageBox.critical(self, 'Error', str(msg)[:500])

    def _log(self, msg):
        self.log.append(msg)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())


def main():
    app = QApplication(sys.argv)
    app.setApplicationName('3D Volume Processor')
    gui = SimpleVolumeGUI()
    gui.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
