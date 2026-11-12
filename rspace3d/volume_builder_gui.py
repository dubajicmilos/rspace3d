"""
volume_builder_gui.py — One-button 3D volume processing GUI for unwarp layers.

Select unwarp folder → Process all:
  1. Load all .img files into 3D volume
  2. Save raw volume as .h5
  3. Bin 2x2 (HK), reject outliers, symmetrize
  4. Save processed volume as .h5 with Laue group suffix

Pages (the side bar of the window):
  Layer list  the .dcunwarp layer list for CrysAlisPro (the step before this window)
  Process     folder, Laue group, binning and the button Process all
  Averaging   measure and compare the averaging options
  Advanced    averaging settings, coverage mask, raw cache
  Help        how to use the window and what the options do to the data

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
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QMessageBox,
    QSpinBox,
)
from PyQt6.QtCore import QThread, Qt, pyqtSignal

from . import __version__, ui
from .volume_builder import (
    load_unwarp_folder, bin_volume, symmetrize_volume,
    save_volume_h5, load_volume_h5, adaptive_morph_size, bin_native,
    raw_cache_mismatch, _read_header_fast, resolve_unit_cell, format_symmetry_diagnostics,
    _filter_numbered_imgs, _img_number, fitting_laue_group,
    LAUE_GROUP_NAMES, METRIC_TOLERANCE, _EXPECTED_ORDERS, HAS_GPU,
)
from .make_dcunwarp import generate_dcunwarp
from .averaging import EQUAL_MAD
from .averaging_panel import AveragingPanel, AveragingSettings, engine_name

STANDARD_WHY = (
    f"Standard for unwarp layers: '{EQUAL_MAD['name']}', the default of the earlier versions. "
    'The layers have no detector pixel counts, so the pixel-count weights give almost no gain '
    'here.')
STANDARD_EFFECT = ('The value is the plain mean of the symmetry-equivalent values. A value '
                   'farther than 3 x the spread of these values from their median is rejected.')
ENGINE_WORDS = {'GPU': 'GPU', 'numba': 'CPU (numba)', 'NumPy': 'CPU (NumPy, slow)'}
LAYER_LIMIT = 500           # layers that CrysAlisPro accepts in one .dcunwarp file


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

class SimpleVolumeGUI(ui.Shell):

    def __init__(self) -> None:
        super().__init__('3D Volume Processor', 'Volume from unwarp layers')
        ui.start_size(self, 1280, 920)
        self._busy = False
        self._worker: WorkerThread | None = None
        self._folder_path: str | None = None
        self._dataset_valid = False
        # The binned volume of the Averaging panel: {'key', 'vol', 'raw_path', 'base'}
        self._prepared: dict[str, Any] | None = None
        self._recip: Any = None     # reciprocal basis of the .img header (columns a*, b*, c*)
        self._prefix: str | None = None             # file prefix of the layers
        self._dims: tuple[int, int, int] | None = None   # pixels x, pixels y, layers
        self._last_saved: str | None = None         # the last symmetrised file
        self._viewer: Any = None
        self._viewers: list[Any] = []               # the viewer windows that this window opened
        self._build_ui()

        engine = ENGINE_WORDS[engine_name()]
        if HAS_GPU:
            try:
                import cupy as cp
                name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
                self._log(f'GPU detected: {name}')
                engine = f'GPU ({name})'
            except Exception:
                self._log('GPU detected')
        else:
            self._log('No GPU — using CPU')
        self.sidebar.footer.setText(f'Engine: {engine}\nrspace3d {__version__}')
        self.set_status('Select an unwarp folder to begin')

    def _build_ui(self):
        self._build_dcunwarp_page()
        self._build_process_page()

        # ── Averaging panel ──
        self.avg_panel = AveragingPanel(self._averaging_request, log=self._log,
                                        progress=self._update_progress)
        self.avg_panel.busy_changed.connect(self._on_panel_busy)
        self.avg_panel.applied_changed.connect(self._on_applied_changed)
        self.avg_panel.set_save_action(
            'Process all', self._process_all,
            'Symmetrise with the applied option (or with the settings of the\n'
            'Advanced page) and save. The same step as Process all on the\n'
            'Process page.')
        self.add_page('Averaging', 'Averaging',
                      'Compare the averaging options on the volume, then apply one.',
                      self.avg_panel, 'averaging-page', log=False)

        self._build_advanced_page()
        self.add_help_page('unwarp')

        self.laue_combo.currentIndexChanged.connect(self._on_laue_changed)
        for w in (self.bin_xy_spin, self.bin_z_spin):
            w.valueChanged.connect(self._invalidate_prepared)
            w.valueChanged.connect(self._update_size_label)
        self.morph_combo.currentIndexChanged.connect(self._invalidate_prepared)
        self.force_reload_chk.toggled.connect(self._invalidate_prepared)
        self.avg_settings.changed.connect(self._update_avg_label)
        for w in (self.dc_lmin, self.dc_lmax, self.dc_step):
            w.valueChanged.connect(self._update_dc_note)
        ui.guard_wheel(self)
        self._update_avg_label()
        self._update_files_label()
        self._update_size_label()
        self._update_dc_note()
        self._update_run_note()
        self.show_page('Process')

    def _build_process_page(self):
        # ── 1. Folder selection ──
        s1 = ui.Section('Unwarp folder', step=1, help_topic='unwarp')
        self.folder_label = ui.label('No folder selected', 'strong', wrap=True, selectable=True)
        self.browse_btn = ui.button(
            'Browse...', clicked=self._browse_folder,
            tip='Select the folder containing CrysAlisPro unwarp .img files.\n'
                'These are HK reciprocal space reconstructions at different l values.')
        ui.set_role(self.browse_btn, 'primary')      # the first action in a new window
        top = QHBoxLayout()
        top.addWidget(self.folder_label, 1)
        top.addWidget(self.browse_btn, 0, Qt.AlignmentFlag.AlignTop)
        s1.add(top)
        self.dataset_hint = s1.add_hint(
            'Select the folder that CrysAlisPro wrote with "dc unwarp" (numbered .img files). '
            'If you have no layers, make the layer list on the Layer list page first.')
        self.dataset_note = s1.add(ui.Banner('error'))
        self.info_grid = ui.FactGrid(['Layers', 'Layer size', 'Fixed index', 'Wavelength',
                                      'Unit cell'])
        self.info_grid.setVisible(False)
        s1.add(self.info_grid)

        # ── 2. Laue group and binning ──
        s2 = ui.Section('Symmetry and binning', step=2, help_topic='symmetry')
        self.laue_combo = QComboBox()
        for key, name in LAUE_GROUP_NAMES.items():
            self.laue_combo.addItem(name, key)
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
            '  -1    — triclinic (inversion only, no symmetry averaging)\n\n'
            'The window checks only that the cell metric allows the group:\n'
            'the metric cannot tell a cubic cell from a cell with a small\n'
            'distortion.')
        form = ui.form()
        form.addRow('Laue group', self.laue_combo)
        ui.form_hint(form, 'The average uses this symmetry. Select the group of the phase. An '
                           'experiment that breaks the symmetry needs a lower group.')
        self.laue_note = ui.Banner('warn')
        form.addRow(self.laue_note)
        self.bin_xy_spin = QSpinBox()
        self.bin_xy_spin.setRange(1, 16); self.bin_xy_spin.setValue(2)
        self.bin_xy_spin.setToolTip(
            'Bin factor for H and K axes.\n'
            'Averages NxN pixels to reduce volume size and noise.\n'
            '1 = no binning, 2 = 2x2 (4x smaller), 4 = 4x4 (16x smaller)')
        self.bin_z_spin = QSpinBox()
        self.bin_z_spin.setRange(1, 16); self.bin_z_spin.setValue(1)
        self.bin_z_spin.setToolTip(
            'Bin factor for L axis (stacking direction).\n'
            '1 = keep all L slices, 2 = average pairs, etc.')
        form.addRow('Bin factor h, k', self.bin_xy_spin)
        form.addRow('Bin factor l', self.bin_z_spin)
        self.size_label = ui.form_hint(form, '')
        s2.add(form)
        self._setup_group = s2

        # ── 3. Averaging (the settings are on the Advanced page) ──
        s3 = ui.Section('Averaging and outlier rejection', step=3, help_topic='averaging')
        self.avg_label = ui.label('', 'strong', wrap=True)
        s3.add(self.avg_label)
        self.avg_hint = s3.add_hint('')
        s3.add(ui.row(
            ui.button('Change the option', 'link', clicked=lambda: self.show_page('Advanced'),
                      tip='The averaging settings are on the Advanced page.'),
            ui.button('Compare the options on this volume', 'link',
                      clicked=lambda: self.show_page('Averaging'),
                      tip='The Averaging page scores the options on your volume.'),
            spacing=ui.SPACE[3]))

        # ── 4. Output ──
        s4 = ui.Section('Output', step=4, help_topic='files')
        s4.add_hint('The files go into the unwarp folder. The raw volume is also a cache: the '
                    'next run reads it and does not read the .img files again.')
        self.files_label = ui.label('', 'mono', wrap=True, selectable=True)
        s4.add(self.files_label)

        # The main action, below the page (always visible)
        self.run_bar, bar = ui.footer_bar()
        self.run_note = ui.hint('')
        bar.addWidget(self.run_note, 1)
        self.view_btn = ui.button('Open in viewer', clicked=self._open_in_viewer,
                                  tip='Open the last saved symmetrised volume in the viewer.')
        self.view_btn.setEnabled(False)
        bar.addWidget(self.view_btn)
        self.process_btn = ui.button(
            'Process all', 'primary', clicked=self._process_all,
            tip='Run the full pipeline:\n'
                '1. Load all .img files into 3D volume (with per-frame coverage mask)\n'
                '2. Save raw volume as .h5\n'
                '3. Bin the volume\n'
                '4. Single-pass symmetrize + outlier rejection\n'
                '5. Save processed volume as .h5  (Ctrl+Return)')
        self.process_btn.setEnabled(False)
        self.process_btn.setShortcut('Ctrl+Return')
        bar.addWidget(self.process_btn)

        self.process_tab = ui.two_columns([s1, s2], [s3, s4], ratio=(6, 5))
        self.add_page('Process', 'Process',
                      'From the unwarp layers to a symmetry-averaged volume in one step.',
                      self.process_tab, 'unwarp', footer=self.run_bar)

    def _build_advanced_page(self):
        self.avg_settings = AveragingSettings(EQUAL_MAD, STANDARD_WHY)

        g = ui.Section('Unwarp layers', help_topic='advanced')
        f = ui.form()
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
        f.addRow('Mask kernel (pixels)', self.morph_combo)
        ui.form_hint(f, 'The size that separates real low counts from regions without a '
                        'measurement (gaps, beam stop, edges). Keep "auto" unless the mask is '
                        'wrong.')

        # Force-reload toggle (default: reuse _raw.h5 if it exists and
        # matches the requested morph_size).
        self.force_reload_chk = QCheckBox('Read the .img files again')
        self.force_reload_chk.setChecked(False)
        self.force_reload_chk.setToolTip(
            'When unchecked (default): if a {prefix}_raw.h5 already exists\n'
            'in the unwarp folder AND its morph_size matches the request,\n'
            'load from there and skip the .img read + raw save steps.\n\n'
            'Check this to always re-read the .img files and overwrite\n'
            'the cached raw file (e.g. if frames changed on disk).')
        f.addRow(self.force_reload_chk)
        ui.form_hint(f, 'Without a tick, the window uses the raw volume of an earlier run if it '
                        'agrees with the folder.')
        g.add(f)

        self.adv_bar, bar = ui.footer_bar()
        bar.addWidget(ui.hint('The defaults are the standard settings. Change them only if you '
                              'need to.'), 1)
        self.reset_btn = ui.button('Reset to the standard settings', clicked=self._reset_advanced,
                                   tip='Set all settings of this page to the standard values.')
        bar.addWidget(self.reset_btn)
        self.adv_tab = ui.two_columns([self.avg_settings], [g])
        self.add_page('Advanced', 'Advanced', 'Settings that a standard run does not need.',
                      self.adv_tab, 'advanced', footer=self.adv_bar)

    def _build_dcunwarp_page(self):
        g_dc = ui.Section('Layer list for CrysAlisPro', help_topic='unwarp')
        g_dc.add_hint(
            'Use this page first if you have no unwarp layers.\n'
            '1. Make the layer list here.\n'
            '2. In CrysAlisPro, run "dc unwarp" and load the list. CrysAlisPro writes the .img '
            'layers into an unwarp folder.\n'
            '3. Select that folder on the Process page.')
        f = ui.form()
        self.dc_lmin = QDoubleSpinBox()
        self.dc_lmin.setRange(-50, 50); self.dc_lmin.setValue(-6.0)
        self.dc_lmin.setDecimals(2); self.dc_lmin.setSingleStep(1.0)
        self.dc_lmin.setToolTip('Starting l value (most negative)')
        f.addRow('First l', self.dc_lmin)

        self.dc_lmax = QDoubleSpinBox()
        self.dc_lmax.setRange(-50, 50); self.dc_lmax.setValue(6.0)
        self.dc_lmax.setDecimals(2); self.dc_lmax.setSingleStep(1.0)
        self.dc_lmax.setToolTip('Ending l value (most positive)')
        f.addRow('Last l', self.dc_lmax)

        self.dc_step = QDoubleSpinBox()
        self.dc_step.setRange(0.001, 1.0); self.dc_step.setValue(0.02)
        self.dc_step.setDecimals(3); self.dc_step.setSingleStep(0.01)
        self.dc_step.setSuffix(' r.l.u.')
        self.dc_step.setToolTip('Step size in l (r.l.u.). Typical: 0.02-0.05')
        f.addRow('Step in l', self.dc_step)

        self.dc_res = QDoubleSpinBox()
        self.dc_res.setRange(0.1, 5.0); self.dc_res.setValue(0.8)
        self.dc_res.setDecimals(1); self.dc_res.setSingleStep(0.1)
        self.dc_res.setSuffix(' Å')
        self.dc_res.setToolTip('Resolution limit d_min in Angstroms.\n'
                               'Smaller = more data but larger files.')
        f.addRow('Resolution limit', self.dc_res)
        for w in (self.dc_lmin, self.dc_lmax, self.dc_step, self.dc_res):
            w.setMinimumWidth(130)
        ui.form_hint(f, 'A smaller resolution limit gives more data and larger layers.')
        g_dc.add(f)
        self.dc_note = ui.label('', wrap=True)
        g_dc.add(self.dc_note)
        self.dc_done = g_dc.add(ui.Banner('ok'))

        self.dc_bar, bar = ui.footer_bar()
        self.dc_target = ui.hint('')
        bar.addWidget(self.dc_target, 1)
        self.dc_btn = ui.button('Generate the layer list', 'primary',
                                clicked=self._generate_dcunwarp,
                                tip='Create the .dcunwarp file. It goes into the selected '
                                    'unwarp folder,\nor into a folder that you select.')
        bar.addWidget(self.dc_btn)
        page = ui.two_columns([g_dc], [], ratio=(6, 5))
        self.add_page('Layer list', 'Layer list',
                      'The list of layers that CrysAlisPro needs for "dc unwarp".',
                      page, 'unwarp', footer=self.dc_bar)

    def _reset_advanced(self) -> None:
        self.avg_settings.reset()
        self.morph_combo.setCurrentIndex(0)
        self.force_reload_chk.setChecked(False)
        self._log('Advanced: the settings are the standard settings again.')

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
        self._recip = None
        self._prefix = None
        self._dims = None
        self._invalidate_prepared()
        self.folder_label.setText(folder.replace('\\', '/'))    # a path with "/" can wrap
        self.folder_label.setToolTip(folder)
        self._log(f'Folder: {folder}')
        self._update_dc_note()

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

        self.info_grid.set_values({
            'Layers': f'{n} files ({img_prefix}_*.img)',
            'Layer size': f'{nx} x {ny} px, plane {hdr["plane_type"]}',
            'Fixed index': f'{l_min:.3f} to {l_max:.3f}, step {l_step:.4f}',
            'Wavelength': f'{hdr["wavelength"]:.5f} Å',
            'Unit cell': (f'a = {cell["a"]:.5f}, b = {cell["b"]:.5f}, c = {cell["c"]:.5f} Å\n'
                          f'alpha = {cell["alpha"]:.3f}, beta = {cell["beta"]:.3f}, '
                          f'gamma = {cell["gamma"]:.3f} deg'
                          + (f'\n(from {os.path.basename(par_path)})' if par_path else
                             '\n(from the UB matrix of the .img header)'))})
        self.info_grid.setVisible(True)
        self.dataset_hint.setVisible(False)
        self.dataset_note.clear()
        ui.set_role(self.browse_btn, None)
        self._log(info)

        # Estimate sizes
        nx2, ny2 = nx // 2, ny // 2
        raw_mb = nx * ny * n * 4 / 1e6
        bin_mb = nx2 * ny2 * n * 4 / 1e6
        self._log(f'Raw volume: {nx} x {ny} x {n} = {raw_mb:.0f} MB')
        self._log(f'Binned 2x2: {nx2} x {ny2} x {n} = {bin_mb:.0f} MB')

        self._dataset_valid = True
        self._prefix = img_prefix
        self._dims = (nx, ny, n)
        self.process_btn.setEnabled(True)
        try:
            self._recip = (np.asarray(hdr['ub'], np.float64).reshape(3, 3)
                           / float(hdr['wavelength']))
        except Exception:
            self._recip = None
        self._check_laue(switch=True)
        self._update_files_label()
        self._update_size_label()
        self._update_run_note()
        self.set_status('Unwarp folder read. Check the settings, then click Process all.')

    def _fail_folder(self, msg):
        self._dataset_valid = False
        self.dataset_note.show_message(msg, 'error')
        self.info_grid.setVisible(False)
        self.dataset_hint.setVisible(False)
        self._log('  ERROR: ' + msg)
        self.process_btn.setEnabled(False)
        self._check_laue()
        self._update_files_label()
        self._update_size_label()
        self._update_run_note()

    def _update_files_label(self, *_: Any) -> None:
        if not self._prefix:
            self.files_label.setText('<sample>_raw.h5\n<sample>_sym_<laue>.h5')
            return
        suffix = self.laue_combo.currentData().replace('/', '').replace('-', 'bar')
        self.files_label.setText(f'{self._prefix}_raw.h5\n{self._prefix}_sym_{suffix}.h5')

    def _update_size_label(self, *_: Any) -> None:
        """The size of the volume after the binning (4 bytes for each voxel)."""
        bxy, bz = self.bin_xy_spin.value(), self.bin_z_spin.value()
        text = ('A factor of 2 for h and k averages 2 x 2 pixels of each layer. This gives 4 x '
                'fewer voxels and less noise.')
        if self._dims is not None:
            nx, ny, n = self._dims
            mx, my, mz = max(nx // bxy, 1), max(ny // bxy, 1), max(n // bz, 1)
            text = (f'Volume after the binning: about {mx} x {my} x {mz} voxels, '
                    f'{mx * my * mz * 4 / 1e9:.1f} GB in memory.')
        self.size_label.setText(text)

    def _update_run_note(self, *_: Any) -> None:
        """The line next to the main action: why it is off, or what a click does."""
        if not self._dataset_valid:
            text = 'Select an unwarp folder to start.'
        else:
            text = 'One click loads the layers, bins, symmetrises and saves.'
        ui.set_role(self.run_note, 'hint')
        self.run_note.setText(text)

    # ── Generate dcunwarp ──

    def _dc_layers(self) -> tuple[int, int]:
        """Number of layers and of .dcunwarp files for the values of the Layer list page."""
        l_min, l_max, l_step = self.dc_lmin.value(), self.dc_lmax.value(), self.dc_step.value()
        n_layers = len(np.arange(l_min, l_max + l_step * 0.5, l_step))
        return n_layers, int(np.ceil(n_layers / LAYER_LIMIT))

    def _update_dc_note(self, *_: Any) -> None:
        n_layers, n_files = self._dc_layers()
        if n_layers <= 0:
            text = 'The last l must be larger than the first l.'
        else:
            text = f'{n_layers} layers in {n_files} file' + ('s' if n_files > 1 else '') + '.'
            if n_files > 1:
                text += (f' CrysAlisPro accepts {LAYER_LIMIT} layers in one file, so you run '
                         f'"dc unwarp" {n_files} times.')
        ui.set_role(self.dc_note, 'error' if n_layers <= 0 else None)
        self.dc_note.setText(text)
        self.dc_btn.setEnabled(n_layers > 0 and not self._busy)
        self.dc_target.setText(
            f'The file goes into the selected unwarp folder: {self._folder_path}'
            if self._folder_path else
            'The window asks for the folder of the file when you click the button.')

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
        n_layers, n_files = self._dc_layers()

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
        self.dc_done.show_message(
            f'Created {", ".join(filenames)} in {out_dir}. Now run "dc unwarp" in CrysAlisPro '
            'and load the file.', 'ok')
        self.set_status(f'Created {len(filenames)} layer list file'
                        + ('s' if len(filenames) > 1 else ''))

        if n_files > 1:
            msg = (f'{n_files} files were created because CrysAlisPro can only '
                   f'process 500 layers at a time.\n\n'
                   f'You need to run "dc unwarp" {n_files} times in CrysAlisPro, '
                   f'once with each file.')
            self._log(f'  WARNING: {n_files} files — CrysAlisPro 500-layer limit')
            QMessageBox.warning(self, 'Multiple dcunwarp files', msg)

    # ── Process all ──

    def _process_all(self):
        if not self._dataset_valid or not self._folder_path:
            msg = 'Select an unwarp folder with .img files first (Process page).'
            self._log(msg)
            self.set_status(msg)
            return
        laue = self.laue_combo.currentData()
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
            rule = (f"averaging option '{choice['name'] if choice else '?'}' from the "
                    'Averaging page')
        else:
            sym_kw = self.avg_settings.kwargs()
            rule = f'averaging {self.avg_settings.describe()}'

        self._set_busy(True, 'Processing...')
        self._log(f'\n{"="*50}')
        self._log(f'Processing: {laue}, {rule}, morph={morph}, '
                  f'bin={bin_xy}x{bin_xy}x{bin_z}, '
                  f'force_reload={force_reload}, {device}')
        self._log(f'{"="*50}')

        self._worker = WorkerThread(
            self._do_process_all,
            self._folder_path, laue, rule, morph, bin_xy, bin_z,
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
            raise ValueError('Select an unwarp folder with .img files first (Process page).')
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
            log('The panel loaded and binned the volume. Process all uses this volume.')
            return vol
        return laue, volume_fn

    def _on_applied_changed(self) -> None:
        """An applied averaging option replaces the settings of the Advanced page."""
        self.avg_settings.set_override(self.avg_panel.applied_name())
        self._update_avg_label()

    def _update_avg_label(self, *_: Any) -> None:
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
            note = 'This is not the standard option for unwarp layers.'
            role = 'warn'
        ui.set_role(self.avg_label, role)
        self.avg_label.setText(text)
        self.avg_hint.setText(note)

    def _on_laue_changed(self, *_: Any) -> None:
        self.avg_panel.reset('The Laue group changed. Click Measure again.')
        self._check_laue()
        self._update_files_label()

    def _check_laue(self, switch: bool = False) -> None:
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
            self._log('Laue group: ' + text)
            kind = 'warn'
        else:
            text = (f'The cell metric does not allow {laue} (residual {residual:.2g}, limit '
                    f'{METRIC_TOLERANCE:g}). The group of highest order that the cell allows '
                    f'is {fit}.')
            kind = 'error'
        self.laue_note.show_message(text, kind)

    def _on_panel_busy(self, busy: bool) -> None:
        for w in (self.process_tab, self.adv_tab, self.run_bar, self.adv_bar):
            w.setEnabled(not busy)
        self.set_status('The Averaging panel is running...' if busy else 'Ready')
        if busy:                                    # say why the settings are off
            ui.set_role(self.run_note, 'hint')
            self.run_note.setText('The Averaging page is running. The settings are off until '
                                  'it stops.')
        else:
            self._update_run_note()

    def _do_process_all(
        self,
        folder: str,
        laue: str,
        rule: str,
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
        log(f'  Laue group: {laue} ({n_ops} ops), {rule}')
        vol = symmetrize_volume(vol, laue, progress_callback=cb, **(sym_kw or {}))
        if choice is not None:
            vol.metadata['averaging_choice'] = choice
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
        """Steps 1 to 3 of Process all: load (or the raw cache), save raw, bin.

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
            log(f'  (skipped .img read + raw save; tick '
                f'"Read the .img files again" to regenerate)')
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
        self._last_saved = result['sym_path']
        self._set_busy(False)
        name = os.path.basename(result['sym_path'])
        self.set_status(f'Saved {name}')
        ui.set_role(self.run_note, 'ok')
        self.run_note.setText(f'Saved {name}. Click Open in viewer to look at it.')

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

    # ── Helpers ──

    def _log(self, msg):
        """Write to the log. A step line also goes to the status line (visible with the log closed)."""
        super()._log(msg)
        if isinstance(msg, str) and msg.startswith('Step '):
            self.set_status(msg.strip())

    def _set_busy(self, busy, message=''):
        self._busy = busy
        self.browse_btn.setEnabled(not busy)
        self._setup_group.setEnabled(not busy)
        self.process_btn.setEnabled(not busy and self._dataset_valid)
        self.view_btn.setEnabled(not busy and self._last_saved is not None)
        self.dc_btn.setEnabled(not busy)
        self.adv_tab.setEnabled(not busy)
        self.adv_bar.setEnabled(not busy)
        self.avg_panel.set_host_busy(busy)
        if not busy:
            self._update_dc_note()          # the button is off for an empty range of l
        self.set_status(message if message else ('Ready' if not busy else ''))
        if busy:
            ui.set_role(self.run_note, 'hint')
            self.run_note.setText(message + ' The log shows the steps.')
        else:
            self._update_run_note()

    def _on_error(self, msg):
        self._log(f'ERROR: {msg}')
        self._set_busy(False)
        ui.set_role(self.run_note, 'error')
        self.run_note.setText('The run stopped. The log has the error text.')
        ui.show_error(self, 'The run stopped', msg)


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName('3D Volume Processor')
    gui = SimpleVolumeGUI()
    gui.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
