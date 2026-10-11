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

from __future__ import annotations

import sys
import os
import time
from typing import Any, Callable
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QFormLayout, QPushButton, QLabel, QComboBox,
    QCheckBox, QSpinBox, QDoubleSpinBox, QProgressBar, QFileDialog,
    QTextEdit, QMessageBox, QTabWidget, QScrollArea, QSplitter,
)
from PyQt6.QtCore import QThread, Qt, pyqtSignal

from .volume_builder import (
    load_unwarp_folder, bin_volume, symmetrize_volume,
    save_volume_h5, load_volume_h5, adaptive_morph_size, bin_native,
    raw_cache_mismatch, _read_header_fast, resolve_unit_cell, format_symmetry_diagnostics,
    _filter_numbered_imgs, _img_number,
    LAUE_GROUP_NAMES, _EXPECTED_ORDERS, HAS_GPU,
)
from .make_dcunwarp import generate_dcunwarp
from .averaging_panel import LOG_HEIGHT, AveragingPanel


# ──────────────────────────────────────────────────────────────────
# Worker thread
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
            result = self.func(*self.args)
            self.finished.emit(result)
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")


# ──────────────────────────────────────────────────────────────────
# Main GUI
# ──────────────────────────────────────────────────────────────────

class SimpleVolumeGUI(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('3D Volume Processor')
        self.resize(1280, 920)
        self._worker: WorkerThread | None = None
        self._folder_path: str | None = None
        self._dataset_valid = False
        # The binned volume of the Averaging panel: {'key', 'vol', 'raw_path', 'base'}
        self._prepared: dict[str, Any] | None = None
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

        self.statusBar().showMessage('Select an unwarp folder to begin')  # type: ignore[union-attr]

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        # Tabs above, progress and log below. The user can move the border.
        self.split = QSplitter(Qt.Orientation.Vertical)
        outer.addWidget(self.split)
        self.tabs = QTabWidget()
        self.split.addWidget(self.tabs)
        self.process_tab = QWidget()
        layout = QVBoxLayout(self.process_tab)
        # A scroll area: the tab does not push the log below its height
        process_scroll = QScrollArea()
        process_scroll.setWidgetResizable(True)
        process_scroll.setWidget(self.process_tab)
        self.tabs.addTab(process_scroll, 'Process')

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
            'For each voxel, the symmetry-equivalent orbit is gathered and\n'
            'members deviating more than N*sigma*MAD from the orbit median\n'
            'are excluded from the symmetrized mean.\n\n'
            '3.0 = standard\n'
            '5.0 = conservative (removes only extreme outliers)\n'
            '1.5 = aggressive (tighter clipping, use with caution)')
        sigma_row = QHBoxLayout()
        sigma_row.addWidget(self.sigma_spin)
        self.sigma_note = QLabel('')
        self.sigma_note.setStyleSheet('color: #070; font-weight: bold;')
        sigma_row.addWidget(self.sigma_note, stretch=1)
        f2.addRow('Outlier sigma:', sigma_row)

        # Coverage-mask kernel (auto by default; allow odd integer override)
        self.morph_combo = QComboBox()
        self.morph_combo.addItem('auto', 'auto')
        for n in (3, 5, 7, 9, 11, 13, 15):
            self.morph_combo.addItem(str(n), n)
        self.morph_combo.setToolTip(
            'Coverage-mask morphological-opening kernel.\n'
            'Decides which zero pixels are real low-count measurements\n'
            'vs unmeasured detector regions (gaps, beamstop, edges).\n\n'
            "'auto' (default) picks ~5% of one Bragg cell from the\n"
            'detector geometry — adapts to detector distance, pixel\n'
            'count and unit cell size.\n\n'
            'Override with a fixed odd integer if needed (3 = original\n'
            'behaviour; larger = fewer false positives, may shave thin\n'
            'detector gaps).')
        f2.addRow('Mask kernel:', self.morph_combo)

        # Force-reload toggle (default: reuse _raw.h5 if it exists and
        # matches the requested morph_size).
        self.force_reload_chk = QCheckBox('Force re-read .img files')
        self.force_reload_chk.setChecked(False)
        self.force_reload_chk.setToolTip(
            'When unchecked (default): if a {prefix}_raw.h5 already exists\n'
            'in the unwarp folder AND its morph_size matches the request,\n'
            'load from there and skip the .img read + raw save steps.\n\n'
            'Check this to always re-read the .img files and overwrite\n'
            'the cached raw file (e.g. if frames changed on disk).')
        f2.addRow('', self.force_reload_chk)

        self.process_btn = QPushButton('Process All')
        self.process_btn.setEnabled(False)
        self.process_btn.setMinimumHeight(40)
        self.process_btn.setStyleSheet('font-weight: bold; font-size: 13px;')
        self.process_btn.clicked.connect(self._process_all)
        self.process_btn.setToolTip(
            'Run the full pipeline:\n'
            '1. Load all .img files into 3D volume (with per-frame coverage mask)\n'
            '2. Save raw volume as .h5\n'
            '3. Bin the volume\n'
            '4. Single-pass symmetrize + outlier rejection (orbit MAD clip)\n'
            '5. Save processed volume as .h5')
        f2.addRow(self.process_btn)

        g2.setLayout(f2)
        layout.addWidget(g2)
        layout.addStretch(1)

        # ── Averaging panel ──
        self.avg_panel = AveragingPanel(self._averaging_request, log=self._log,
                                        progress=self._update_progress)
        self.avg_panel.busy_changed.connect(self._on_panel_busy)
        self.avg_panel.applied_changed.connect(self._on_applied_changed)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.avg_panel)
        self.tabs.addTab(scroll, 'Averaging')
        self.laue_combo.currentIndexChanged.connect(
            lambda _: self.avg_panel.reset('The Laue group changed. Click Measure again.'))
        for w in (self.bin_xy_spin, self.bin_z_spin):
            w.valueChanged.connect(self._invalidate_prepared)
        self.morph_combo.currentIndexChanged.connect(self._invalidate_prepared)
        self.force_reload_chk.toggled.connect(self._invalidate_prepared)

        # ── Progress + log ──
        bottom = QWidget()
        bl = QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        bl.addWidget(self.progress)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet('font-family: Consolas, monospace; font-size: 11px;')
        bl.addWidget(self.log)
        self.split.addWidget(bottom)
        self.split.setStretchFactor(0, 1)
        self.split.setStretchFactor(1, 0)
        # About 230 px for the progress bar and the log at the default window size
        self.split.setSizes([LOG_HEIGHT * 3, LOG_HEIGHT])

    # ── Folder ──

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, 'Select Unwarp Folder',
            '')
        if folder:
            self._set_folder(folder)

    def _set_folder(self, folder):
        self._folder_path = folder
        self._dataset_valid = False
        self._invalidate_prepared()
        self.folder_label.setText(folder)
        self._log(f'Folder: {folder}')

        img_files = _filter_numbered_imgs(folder)
        n = len(img_files)
        if n == 0:
            self._fail_folder('No .img files found.')
            return

        sorted_f = sorted(img_files, key=_img_number)

        img_prefix = sorted_f[0].rsplit('_', 1)[0]
        self._log(f'Prefix: {img_prefix} ({n} files)')

        try:
            hdr = _read_header_fast(os.path.join(folder, sorted_f[0]))
            l_min = hdr['fixed_value']
            l_max = _read_header_fast(os.path.join(folder, sorted_f[-1]))['fixed_value']
            # Unit cell: try par file first, fall back to .img header UB
            cell, par_path = resolve_unit_cell(folder, hdr)
        except Exception as e:                     # truncated / foreign .img, bad par
            self._fail_folder(f'Could not read the .img headers: {e}')
            return
        nx, ny = hdr['nx'], hdr['ny']
        l_step = (l_max - l_min) / max(n - 1, 1)

        info = (f'{n} files  |  {nx} x {ny} px  |  {hdr["plane_type"]}  |  '
                f'lambda = {hdr["wavelength"]:.5f} A\n'
                f'Fixed axis: {l_min:.3f} to {l_max:.3f} (step {l_step:.4f})')

        if par_path:
            self._log(f'Par file: {os.path.basename(par_path)}')
        else:
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

        self._dataset_valid = True
        self.process_btn.setEnabled(True)

    def _fail_folder(self, msg):
        self._dataset_valid = False
        self.info_label.setText(msg)
        self._log('  ERROR: ' + msg)
        self.process_btn.setEnabled(False)

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
        morph = self.morph_combo.currentData()
        force_reload = self.force_reload_chk.isChecked()
        bin_xy = self.bin_xy_spin.value()
        bin_z = self.bin_z_spin.value()
        device = 'GPU' if HAS_GPU else 'CPU'

        # The Averaging panel: its volume and its applied option, if any
        prepared = self._current_prepared()
        sym_kw: dict[str, Any] = {}
        choice = None
        if prepared is not None:
            sym_kw = self.avg_panel.symmetrize_kwargs(prepared['vol'], laue)
            choice = self.avg_panel.choice_metadata() if sym_kw else None
        if sym_kw:
            rule = f"averaging option '{choice['name'] if choice else '?'}'"
        else:
            rule = f'sigma={sigma}'

        self._set_busy(True, 'Processing...')
        self._log(f'\n{"="*50}')
        self._log(f'Processing: {laue}, {rule}, morph={morph}, '
                  f'bin={bin_xy}x{bin_xy}x{bin_z}, '
                  f'force_reload={force_reload}, {device}')
        self._log(f'{"="*50}')

        self._worker = WorkerThread(
            self._do_process_all,
            self._folder_path, laue, sigma, morph, bin_xy, bin_z,
            force_reload, prepared, sym_kw, choice)
        self._worker.progress.connect(self._update_progress)
        self._worker.log_msg.connect(self._log)
        self._worker.finished.connect(self._on_process_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    # ── Averaging panel ──

    def _volume_key(self) -> tuple[Any, ...]:
        return (self._folder_path, self.morph_combo.currentData(),
                self.force_reload_chk.isChecked(), self.bin_xy_spin.value(),
                self.bin_z_spin.value())

    def _current_prepared(self) -> dict[str, Any] | None:
        """The binned volume of the Averaging panel if it matches the settings."""
        if self._prepared is not None and self._prepared['key'] == self._volume_key():
            return self._prepared
        return None

    def _invalidate_prepared(self, *_: Any) -> None:
        self._prepared = None
        self.avg_panel.reset('The volume settings changed. Click Measure again.')

    def _averaging_request(self) -> tuple[str, Callable[..., Any]]:
        """Laue group and volume function for the Averaging panel (GUI thread)."""
        if not self._dataset_valid or not self._folder_path:
            raise ValueError('Select an unwarp folder with .img files first.')
        laue = self.laue_combo.currentData()
        key = self._volume_key()
        prepared = self._current_prepared()

        def volume_fn(log: Callable[[str], None], progress: Callable[[int, int], None]) -> Any:
            if prepared is not None:
                log('Use the volume that the panel loaded and binned before.')
                return prepared['vol']
            folder, morph, force_reload, bin_xy, bin_z = key
            vol, raw_path, base = self._load_binned(folder, morph, bin_xy, bin_z,
                                                    force_reload, log, progress)
            self._prepared = {'key': key, 'vol': vol, 'raw_path': raw_path, 'base': base}
            log('The panel loaded and binned the volume. Process All uses this volume.')
            return vol
        return laue, volume_fn

    def _on_applied_changed(self) -> None:
        """An applied averaging option replaces the outlier sigma of Process All."""
        name = self.avg_panel.applied_name()
        self.sigma_spin.setEnabled(name is None)
        self.sigma_note.setText('' if name is None
                                else f"Not used. Process All uses '{name}' (Averaging tab).")

    def _on_panel_busy(self, busy: bool) -> None:
        self.process_tab.setEnabled(not busy)
        if busy:
            self.statusBar().showMessage('Averaging panel is running...')  # type: ignore[union-attr]
        else:
            self.statusBar().showMessage('Ready')  # type: ignore[union-attr]

    def _do_process_all(
        self,
        folder: str,
        laue: str,
        sigma: float,
        morph: int | str,
        bin_xy: int,
        bin_z: int,
        force_reload: bool,
        prepared: dict[str, Any] | None = None,
        sym_kw: dict[str, Any] | None = None,
        choice: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        log = lambda msg: QThread.currentThread()._emit_log(msg)  # type: ignore[union-attr]
        cb = QThread.currentThread()._emit_progress  # type: ignore[union-attr]
        device = 'GPU' if HAS_GPU else 'CPU'

        if prepared is not None:
            log('Step 1-3/4: Use the volume that the Averaging panel loaded and binned.')
            vol, raw_path, base = prepared['vol'], prepared['raw_path'], prepared['base']
            nh, nk, nl = vol.intensity.shape
            log(f'  Binned: {nh} x {nk} x {nl}')
        else:
            vol, raw_path, base = self._load_binned(folder, morph, bin_xy, bin_z,
                                                    force_reload, log, cb)

        # ── Step 4: Single-pass symmetrize + outlier rejection ──
        n_ops = _EXPECTED_ORDERS[laue]
        log(f'Step 4/4: Symmetrize + outlier rejection ({device})...')
        t0 = time.time()
        if sym_kw:
            log(f'  Laue group: {laue} ({n_ops} ops), averaging option '
                f'\'{choice["name"] if choice else "?"}\' from the Averaging panel')
            vol = symmetrize_volume(vol, laue, progress_callback=cb, **sym_kw)
            if choice is not None:
                vol.metadata['averaging_choice'] = choice
        else:
            log(f'  Laue group: {laue} ({n_ops} ops), sigma={sigma}')
            vol = symmetrize_volume(vol, laue, sigma=sigma, progress_callback=cb)
        dt_sym = time.time() - t0
        n_removed = vol.metadata.get('n_outliers_removed', 0)
        log(f'  Done in {dt_sym:.1f}s '
            f'({n_removed:,} orbit members flagged as outliers)')
        log(f'  Range: [{np.nanmin(vol.intensity):.3f}, '
            f'{np.nanmax(vol.intensity):.3f}]')
        log('  Symmetry diagnostics per |q| shell:')
        log(format_symmetry_diagnostics(vol.metadata.get('symmetry_diagnostics')))

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

    def _load_binned(
        self,
        folder: str,
        morph: int | str,
        bin_xy: int,
        bin_z: int,
        force_reload: bool,
        log: Callable[[str], None],
        cb: Callable[[int, int], None],
    ) -> tuple[Any, str, str]:
        """Steps 1 to 3 of Process All: load (or the raw cache), save raw, bin.

        Returns (binned volume, raw cache path, file prefix).
        """
        # Derive base name from .img filename prefix
        # e.g. "MAPbI2Br_293K_1_1_0_1.img" -> prefix "MAPbI2Br_293K_1_1_0"
        img_files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.img')])
        first_img = img_files[0]
        # Strip trailing _<number>.img
        base = first_img.rsplit('_', 1)[0]
        raw_path = os.path.join(folder, f'{base}_raw.h5')

        # Compute the morph_size that the loader WOULD apply, so we can
        # validate any cached _raw.h5 before reusing it.
        if morph == 'auto':
            hdr0 = _read_header_fast(os.path.join(folder, first_img))
            s_pp_now = 2.0 / (hdr0['d_min'] * hdr0['nx'])
            plane_to_cols = {'HK': (0, 1), 'HL': (0, 2), 'KL': (1, 2)}
            c1, c2 = plane_to_cols[hdr0['plane_type']]
            ub0 = np.asarray(hdr0['ub']).reshape(3, 3)
            wl0 = hdr0['wavelength']
            recip_now = 0.5 * (float(np.linalg.norm(ub0[:, c1] / wl0))
                               + float(np.linalg.norm(ub0[:, c2] / wl0)))
            morph_now = adaptive_morph_size(s_pp_now, recip_now)
        else:
            morph_now = int(morph)

        # Cache decision: the raw cache must carry the requested mask kernel
        # AND the fingerprint of the current .img/.par files.
        use_cache = False
        cached_vol = None
        if os.path.isfile(raw_path) and not force_reload:
            try:
                cached_vol = load_volume_h5(raw_path)
                reason = raw_cache_mismatch(cached_vol, folder, morph_now)
                if reason is None:
                    use_cache = True
                else:
                    log(f'  Cache invalid: {os.path.basename(raw_path)}: {reason}. '
                        'Regenerating from .img files.')
                    cached_vol = None
            except Exception as e:
                log(f'  Could not read {os.path.basename(raw_path)} '
                    f'({e}); regenerating.')

        if use_cache:
            # ── Steps 1+2: load cached, skip .img read + save ──
            log(f'Step 1+2/4: Loading cached {os.path.basename(raw_path)}...')
            t0 = time.time()
            vol = cached_vol
            dt = time.time() - t0
            nh, nk, nl = vol.intensity.shape
            log(f'  Loaded: {nh} x {nk} x {nl} ({vol.intensity.dtype}) '
                f'in {dt:.1f}s')
            log(f'  H=[{vol.H[0]:.3f}, {vol.H[-1]:.3f}]  '
                f'K=[{vol.K[0]:.3f}, {vol.K[-1]:.3f}]  '
                f'L=[{vol.L[0]:.3f}, {vol.L[-1]:.3f}]')
            ms = vol.metadata.get('morph_size', morph_now)
            mode = vol.metadata.get('morph_mode', 'unknown')
            nu_mean = vol.metadata.get('n_unmeasured_per_frame_mean')
            nu_pct = vol.metadata.get('n_unmeasured_per_frame_pct')
            log(f'  Cached mask: morph={ms}px ({mode})'
                + (f', unmeasured/frame mean={nu_mean:,.0f} '
                   f'({nu_pct:.2f}%)' if nu_mean is not None else ''))
            log(f'  (skipped .img read + raw save; check '
                f'"Force re-read" to regenerate)')
        else:
            # ── Step 1: Load raw ──
            log(f'Step 1/4: Loading all .img files...')
            t0 = time.time()
            vol = load_unwarp_folder(folder, bin_xy=1, morph_size=morph,
                                     progress_callback=cb)
            dt = time.time() - t0
            nh, nk, nl = vol.intensity.shape
            log(f'  Loaded: {nh} x {nk} x {nl} ({vol.intensity.dtype}) '
                f'in {dt:.1f}s')
            log(f'  H=[{vol.H[0]:.3f}, {vol.H[-1]:.3f}]  '
                f'K=[{vol.K[0]:.3f}, {vol.K[-1]:.3f}]  '
                f'L=[{vol.L[0]:.3f}, {vol.L[-1]:.3f}]')
            # Coverage-mask diagnostics from metadata
            ms = vol.metadata.get('morph_size')
            mode = vol.metadata.get('morph_mode', 'manual')
            s_pp = vol.metadata.get('morph_s_per_pixel')
            recip = vol.metadata.get('morph_recip_period')
            nu_mean = vol.metadata.get('n_unmeasured_per_frame_mean')
            nu_min = vol.metadata.get('n_unmeasured_per_frame_min')
            nu_max = vol.metadata.get('n_unmeasured_per_frame_max')
            nu_pct = vol.metadata.get('n_unmeasured_per_frame_pct')
            if mode == 'auto' and s_pp is not None and recip is not None:
                log(f'  Mask kernel: morph={ms}px (auto from '
                    f's={s_pp:.5f} 1/A, |a*|={recip:.4f} 1/A, '
                    f'5% of one Bragg cell)')
            else:
                log(f'  Mask kernel: morph={ms}px (manual)')
            if nu_mean is not None:
                log(f'  Coverage: unmeasured/frame mean={nu_mean:,.0f} '
                    f'({nu_pct:.2f}%) [min {nu_min:,}, max {nu_max:,}]')

            # ── Step 2: Save raw ──
            log(f'Step 2/4: Saving raw volume...')
            t0 = time.time()
            save_volume_h5(raw_path, vol)
            dt = time.time() - t0
            size_mb = os.path.getsize(raw_path) / 1e6
            log(f'  Saved: {raw_path}')
            log(f'  {size_mb:.1f} MB on disk (gzip) in {dt:.1f}s')

        # ── Step 3: Bin ──
        if bin_xy > 1 or bin_z > 1:
            log(f'Step 3/4: Binning {bin_xy}x{bin_xy} (in-plane), {bin_z}x (layers)...')
            vol = bin_native(vol, bin_xy, bin_z)
            nh, nk, nl = vol.intensity.shape
            log(f'  Binned: {nh} x {nk} x {nl}')
        else:
            log(f'Step 3/4: No binning (1x1x1)')
        return vol, raw_path, base

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
        self.process_btn.setEnabled(not busy and self._dataset_valid)
        self.dc_btn.setEnabled(not busy)
        self.avg_panel.set_host_busy(busy)
        self.statusBar().showMessage(message if message else ('Ready' if not busy else ''))

    def _on_error(self, msg):
        self._log(f'ERROR: {msg}')
        self._set_busy(False)
        QMessageBox.critical(self, 'Error', str(msg)[:500])

    def _log(self, msg):
        self.log.append(msg)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName('3D Volume Processor')
    gui = SimpleVolumeGUI()
    gui.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
