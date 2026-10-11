"""
averaging_panel.py — The "Averaging" panel of the volume builder and of the
raw-frame reconstruction GUI.

The panel has three steps:

* Measure: fits the noise model, symmetrises once with the default options
  ('equal, mad 3') and shows the diagnostics per |q| shell.
* Compare: scores a list of averaging options, compares their sums with the
  default and shows a table, a suggestion and one slice.
* Apply: the host GUI uses the selected option in its normal symmetrise
  step. Without Apply, the host uses its own settings.

Cancel stops Measure or Compare between two steps (or two candidates). The
panel then keeps the earlier results.

The host gives the panel a `request` function. The panel calls it in the GUI
thread. It returns (laue_group, volume_fn) or raises ValueError with a
message for the user. The panel calls volume_fn(log, progress) in its worker
thread to get the volume. The host gets the applied options with
`symmetrize_kwargs(vol, laue_group)` and the record for the HDF5 metadata
with `choice_metadata()`.

The calculations are in `rspace3d.averaging` (`measure_volume`,
`compare_candidates`). The panel keeps the reference volume, G and slices.
It keeps no full candidate volume.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton, QSpinBox,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from . import averaging as av
from . import orbit_engine as oe

# Orbit members (voxels x operations) of the automatic crop. With the NumPy
# engine, the slowest candidate ('adaptive, model 4') then takes less than
# about one minute (measured: 5.3 s for 60 million members, m-3m).
CROP_BUDGET = 500_000_000

# Height (px) of the progress bar and the log below the tabs of the host GUIs
LOG_HEIGHT = 230

EXTRA_TIP ='Sum ratio at the extra positions. The last Compare had no extra positions.'
OLD_ROWS = QColor('#999999')


def engine_name() -> str:
    """'GPU', 'numba' or 'NumPy': the engine that the panel uses."""
    from .volume_builder import HAS_GPU
    if HAS_GPU:
        return 'GPU'
    return 'numba' if oe.numba_available() else 'NumPy'


def auto_crop_size(shape: tuple[int, ...], n_ops: int, budget: int = CROP_BUDGET) -> int:
    """The largest n with prod(min(n, shape)) x n_ops <= budget (at least 9)."""
    n = 9
    for m in range(9, max(shape) + 2, 2):
        if np.prod([min(m, s) for s in shape]) * n_ops > budget:
            break
        n = m
    return n


def parse_positions(text: str) -> list[tuple[float, float, float]]:
    """Positions such as '0.5 0.5 0.5; 0 0 0.5'. Raises ValueError for a bad text."""
    out = []
    for part in text.replace('\n', ';').split(';'):
        vals = part.replace(',', ' ').split()
        if not vals:
            continue
        if len(vals) != 3:
            raise ValueError(f"Each position needs three numbers (h k l), got {part.strip()!r}.")
        out.append(tuple(float(v) for v in vals))
    return out


def _slice_mesh(geo: dict[str, Any], shape: tuple[int, ...]) -> tuple[Any, Any, Any, Any]:
    """Pixel corners of a slice for pcolormesh (see `averaging.slice_geometry`).

    Returns X, Y (display coordinates, (ny + 1, nx + 1)) and the in-plane
    Miller indices m1, m2 of the same corners.
    """
    ny, nx = shape
    ii, jj = np.meshgrid(np.arange(nx + 1) - 0.5, np.arange(ny + 1) - 0.5)
    m0, mx, my = (np.asarray(geo[k], np.float64) for k in ('m0', 'mx', 'my'))
    m1 = m0[0] + mx[0] * ii + my[0] * jj
    m2 = m0[1] + mx[1] * ii + my[1] * jj
    M = np.asarray(geo['M'], np.float64)
    return M[0, 0] * m1 + M[0, 1] * m2, M[1, 0] * m1 + M[1, 1] * m2, m1, m2


def _miller_lines(ax: Any, M: Any, m1: Any, m2: Any, colour: str, max_lines: int = 40) -> None:
    """Lines of integer Miller index (m1 = n and m2 = n) in display coordinates."""
    lo1, hi1, lo2, hi2 = m1.min(), m1.max(), m2.min(), m2.max()
    style = {'color': colour, 'lw': 0.5, 'alpha': 0.35}
    n1 = np.arange(np.ceil(lo1), np.floor(hi1) + 1)
    n2 = np.arange(np.ceil(lo2), np.floor(hi2) + 1)
    if n1.size <= max_lines:
        for n in n1:
            p = M @ np.array([[n, n], [lo2, hi2]])
            ax.plot(p[0], p[1], **style)
    if n2.size <= max_lines:
        for n in n2:
            p = M @ np.array([[lo1, hi1], [n, n]])
            ax.plot(p[0], p[1], **style)


class PanelWorker(QThread):
    """Runs one job of the panel in a thread (the pattern of the GUI WorkerThread)."""
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    cancelled = pyqtSignal()
    log_msg = pyqtSignal(str)

    def __init__(self, func: Callable[..., Any], *args: Any) -> None:
        super().__init__()
        self.func = func
        self.args = args

    def run(self) -> None:
        try:
            self.finished.emit(self.func(*self.args, log=self.log_msg.emit,
                                         progress=self.progress.emit))
        except av.Cancelled:
            self.cancelled.emit()
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")


def _measure_job(volume_fn: Callable[..., Any], laue: str, crop: int | str | None,
                 use_gpu: bool, cancelled: Callable[[], bool], log: Callable[[str], None],
                 progress: Callable[[int, int], None]) -> dict[str, Any]:
    """Get the volume, crop it if `crop` is a size or 'auto', and measure it."""
    vol = volume_fn(log, progress)
    av._check_cancel(cancelled)
    work = vol
    region = 'full volume ' + ' x '.join(str(s) for s in vol.intensity.shape)
    if crop == 'auto':
        from .volume_builder import _EXPECTED_ORDERS
        crop = auto_crop_size(vol.intensity.shape, _EXPECTED_ORDERS[laue])
    if crop:
        box = av.centred_crop(vol, int(crop))
        if box.intensity.shape != vol.intensity.shape:
            work = box
            region = ('centred crop ' + ' x '.join(str(s) for s in work.intensity.shape)
                      + ' of ' + ' x '.join(str(s) for s in vol.intensity.shape))
        del box
    log(f'Measure on the {region}, Laue group {laue}.')
    measured = av.measure_volume(work, laue, use_gpu=use_gpu, log=log, progress=progress,
                                 cancelled=cancelled)
    return {'volume': vol, 'work': work, 'cropped': work is not vol, 'region': region,
            'laue': laue, 'measured': measured}


class AveragingPanel(QWidget):
    """Measure, compare and apply averaging options (see the module docstring)."""

    busy_changed = pyqtSignal(bool)
    applied_changed = pyqtSignal()

    def __init__(self, request: Callable[[], tuple[str, Callable[..., Any]]],
                 log: Callable[[str], None] | None = None,
                 progress: Callable[[int, int], None] | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._request = request
        self._log_fn = log or (lambda msg: None)
        self._progress_fn = progress or (lambda cur, tot: None)
        self.synchronous = False         # tests: run the jobs in the calling thread
        self._worker: PanelWorker | None = None
        self._state: dict[str, Any] | None = None     # output of _measure_job
        self._compared: dict[str, Any] | None = None
        self._applied: dict[str, Any] | None = None
        self._suggestion: tuple[str, str] | None = None
        self._busy = False
        self._step: str | None = None            # 'Measure' or 'Compare' while a job runs
        self._cancel_flag = False
        self._saved_text = ''                    # label text before the job, for Cancel
        self._measured_once = False
        self._build_ui()
        self.reset('')

    # ── UI ──

    def _build_ui(self) -> None:
        outer = QHBoxLayout(self)
        left = QVBoxLayout()
        right = QVBoxLayout()
        outer.addLayout(left, stretch=4)
        outer.addLayout(right, stretch=5)

        # Measure
        g1 = QGroupBox('Measure')
        v1 = QVBoxLayout(g1)
        eng = engine_name()
        self.engine_label = QLabel()
        self.engine_label.setWordWrap(True)
        if eng == 'NumPy':
            self.engine_label.setText(
                'Engine: NumPy only. Measure and Compare use a centred crop. Each candidate '
                'then takes less than about one minute. Install numba or CuPy to use the '
                'full volume.')
        else:
            self.engine_label.setText(f'Engine: {eng}. Measure and Compare use the full volume.')
        v1.addWidget(self.engine_label)
        crop_row = QHBoxLayout()
        self.crop_chk = QCheckBox('Use a centred crop')
        self.crop_chk.setChecked(eng == 'NumPy')
        self.crop_chk.setToolTip('Measure and compare on a box at the centre of the volume.\n'
                                 'The box has its centre at hkl = 0. If hkl = 0 is near the\n'
                                 'end of an axis, the box moves inward. On an axis without\n'
                                 'hkl = 0, the box uses the middle of the axis.')
        crop_row.addWidget(self.crop_chk)
        crop_row.addWidget(QLabel('voxels per axis:'))
        self.crop_spin = QSpinBox()
        self.crop_spin.setRange(0, 4096)
        self.crop_spin.setValue(0)
        self.crop_spin.setSpecialValueText('auto')
        self.crop_spin.setToolTip('Size of the crop on each axis. "auto" keeps each candidate\n'
                                  'under about one minute with the NumPy engine.')
        crop_row.addWidget(self.crop_spin)
        crop_row.addStretch(1)
        v1.addLayout(crop_row)
        self.measure_btn = QPushButton('Measure')
        self.measure_btn.setToolTip('Fit the noise model and run the default options\n'
                                    "('equal, mad 3') once. Show the diagnostics per |q| shell.")
        self.measure_btn.clicked.connect(self._measure)
        self.measure_cancel_btn = QPushButton('Cancel')
        self.measure_cancel_btn.setToolTip('Stop Measure after the current step.\n'
                                           'The panel keeps the earlier results.')
        self.measure_cancel_btn.clicked.connect(self.cancel)
        m_row = QHBoxLayout()
        m_row.addWidget(self.measure_btn, stretch=1)
        m_row.addWidget(self.measure_cancel_btn)
        v1.addLayout(m_row)
        self.model_label = QLabel()
        self.model_label.setWordWrap(True)
        self.model_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v1.addWidget(self.model_label)
        self.shell_table = QTableWidget(oe.N_SHELLS, len(av.SHELL_HEADERS))
        self.shell_table.setHorizontalHeaderLabels(av.SHELL_HEADERS)
        self.shell_table.horizontalHeaderItem(4).setToolTip(
            'Voxels where the Poisson floor sets the scale / voxels tested.')
        self.shell_table.horizontalHeaderItem(5).setToolTip(
            'Sampled orbits with max/min member pixel count above 10.\n'
            "A volume without pixel counts shows '-'.")
        self.shell_table.horizontalHeaderItem(6).setToolTip(
            'Node sum with MAD 3 / node sum without rejection.')
        self._setup_table(self.shell_table)
        # all 10 shells without a vertical scroll bar
        self.shell_table.setFixedHeight(20 * oe.N_SHELLS + 30)
        v1.addWidget(self.shell_table)
        left.addWidget(g1)

        # Compare
        g2 = QGroupBox('Compare')
        v2 = QVBoxLayout(g2)
        v2.addWidget(QLabel('Candidates (Compare uses the ticked items. Measure ticks the '
                            'automatic set.):'))
        self.cand_list = QListWidget()
        self.cand_list.setMaximumHeight(130)
        v2.addWidget(self.cand_list)
        add_row = QHBoxLayout()
        self.add_weight = QComboBox()
        self.add_weight.addItems(list(oe.WEIGHTINGS))
        self.add_rej = QComboBox()
        self.add_rej.addItems(['mad', 'model', 'none'])
        self.add_sigma = QDoubleSpinBox()
        self.add_sigma.setRange(1.0, 10.0)
        self.add_sigma.setValue(3.5)
        self.add_sigma.setSingleStep(0.5)
        self.add_sigma.setDecimals(1)
        self.add_btn = QPushButton('Add')
        self.add_btn.setToolTip('Add this option to the list of candidates.')
        self.add_btn.clicked.connect(self._add_custom)
        for w in (QLabel('Add:'), self.add_weight, self.add_rej, QLabel('sigma'),
                  self.add_sigma, self.add_btn):
            add_row.addWidget(w)
        add_row.addStretch(1)
        v2.addLayout(add_row)
        pos_row = QHBoxLayout()
        pos_row.addWidget(QLabel('Extra positions:'))
        self.extra_edit = QLineEdit()
        self.extra_edit.setPlaceholderText('for example 0.5 0.5 0.5; 0 0 0.5')
        self.extra_edit.setToolTip('Offsets from the integer nodes (h k l). Separate positions\n'
                                   'with ";". Compare checks the sums near these positions.')
        pos_row.addWidget(self.extra_edit)
        v2.addLayout(pos_row)
        sl_row = QHBoxLayout()
        sl_row.addWidget(QLabel('Slice:'))
        self.plane_combo = QComboBox()
        self.plane_combo.addItems(['HK (fix l)', 'HL (fix k)', 'KL (fix h)'])
        sl_row.addWidget(self.plane_combo)
        sl_row.addWidget(QLabel('value:'))
        self.slice_spin = QDoubleSpinBox()
        self.slice_spin.setRange(-100.0, 100.0)
        self.slice_spin.setDecimals(3)
        self.slice_spin.setSingleStep(0.5)
        sl_row.addWidget(self.slice_spin)
        sl_row.addStretch(1)
        v2.addLayout(sl_row)
        self.compare_btn = QPushButton('Compare')
        self.compare_btn.setToolTip('Score the ticked candidates, compare their sums with\n'
                                    "'equal, mad 3' and make a suggestion.")
        self.compare_btn.clicked.connect(self._compare)
        self.compare_cancel_btn = QPushButton('Cancel')
        self.compare_cancel_btn.setToolTip('Stop Compare after the current candidate.\n'
                                           'The panel keeps the earlier results.')
        self.compare_cancel_btn.clicked.connect(self.cancel)
        c_row = QHBoxLayout()
        c_row.addWidget(self.compare_btn, stretch=1)
        c_row.addWidget(self.compare_cancel_btn)
        v2.addLayout(c_row)
        left.addWidget(g2)
        left.addStretch(1)

        # Results
        g3 = QGroupBox('Results')
        v3 = QVBoxLayout(g3)
        self.result_table = QTableWidget(0, len(av.COMPARE_HEADERS))
        self.result_table.setHorizontalHeaderLabels(av.COMPARE_HEADERS)
        self.result_table.horizontalHeaderItem(1).setToolTip(
            'Leave-one-member-out score. Lower is better.')
        self.result_table.horizontalHeaderItem(3).setToolTip(EXTRA_TIP)
        self.result_table.horizontalHeaderItem(5).setToolTip(
            "Voxels that are finite in 'equal, mad 3' and NaN with this candidate.")
        self.result_table.horizontalHeaderItem(6).setToolTip(
            'Sampled members for which the candidate gives no estimate.')
        self.result_table.horizontalHeaderItem(7).setToolTip(
            "Time of the symmetrisation. The time of 'equal, mad 3' is from Measure.")
        self._setup_table(self.result_table)
        self.result_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.result_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.result_table.itemSelectionChanged.connect(self._on_row_selected)
        self.result_table.setMaximumHeight(170)
        v3.addWidget(self.result_table)
        self.suggest_label = QLabel()
        self.suggest_label.setWordWrap(True)
        self.suggest_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v3.addWidget(self.suggest_label)
        self.limits_label = QLabel(av.LIMITS_TEXT)
        self.limits_label.setWordWrap(True)
        self.limits_label.setStyleSheet('color: #a05000;')
        v3.addWidget(self.limits_label)
        self.figure = Figure(figsize=(6.4, 2.8), layout='constrained')
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(260)
        v3.addWidget(self.canvas, stretch=1)
        right.addWidget(g3, stretch=1)

        # Apply
        g4 = QGroupBox('Apply')
        v4 = QVBoxLayout(g4)
        a_row = QHBoxLayout()
        a_row.addWidget(QLabel('Use:'))
        self.apply_combo = QComboBox()
        self.apply_combo.setMinimumWidth(180)
        a_row.addWidget(self.apply_combo, stretch=1)
        self.apply_btn = QPushButton('Apply')
        self.apply_btn.setToolTip('The symmetrise step of this window uses this option.')
        self.apply_btn.clicked.connect(self._apply)
        a_row.addWidget(self.apply_btn)
        self.clear_btn = QPushButton('Clear')
        self.clear_btn.setToolTip('Remove the applied option. The window uses its own settings.')
        self.clear_btn.clicked.connect(self.clear_applied)
        a_row.addWidget(self.clear_btn)
        v4.addLayout(a_row)
        self.applied_label = QLabel()
        self.applied_label.setWordWrap(True)
        v4.addWidget(self.applied_label)
        right.addWidget(g4)

    @staticmethod
    def _setup_table(t: QTableWidget) -> None:
        t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        t.verticalHeader().setVisible(False)
        t.verticalHeader().setDefaultSectionSize(20)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        t.setStyleSheet('font-size: 11px;')

    # ── state ──

    def reset(self, message: str = '') -> None:
        """Forget the measurement, the comparison and the applied option.

        The panel shows `message` only after a first Measure. Before that,
        it shows 'Click Measure to start.'.
        """
        had = self._state is not None or self._applied is not None
        was_applied = self._applied is not None
        self._state = self._compared = self._applied = self._suggestion = None
        self.shell_table.clearContents()
        self.result_table.setRowCount(0)
        self.result_table.horizontalHeaderItem(3).setToolTip(EXTRA_TIP)
        self.cand_list.clear()
        self.apply_combo.clear()
        self.figure.clear()
        self.canvas.draw_idle()
        self.suggest_label.setText('')
        self.model_label.setText(message if (message and self._measured_once)
                                 else 'Click Measure to start.')
        self._update_applied_label()
        self._update_enabled()
        if had and message:
            self._log_fn(message)
        if was_applied:
            self.applied_changed.emit()

    def is_busy(self) -> bool:
        return self._busy

    def applied_name(self) -> str | None:
        return self._applied['name'] if self._applied else None

    def symmetrize_kwargs(self, vol: Any, laue_group: str) -> dict[str, Any]:
        """Keywords for `symmetrize_volume`, or {} if no option is applied.

        Also {} (with a log line) if `vol` is not the measured volume or
        `laue_group` is not the measured group. G is given only if Measure
        used the full volume.
        """
        if self._applied is None or self._state is None:
            return {}
        if vol is not self._state['volume'] or laue_group != self._state['laue']:
            self._log_fn('The applied averaging option belongs to another volume or Laue group. '
                         'The window uses its own settings.')
            return {}
        m = self._state['measured']
        grad = None if self._state['cropped'] else m['grad']
        return av.candidate_kwargs(self._applied['spec'], m['model'], grad)

    def choice_metadata(self) -> dict[str, Any] | None:
        """The record of the applied option for `metadata['averaging_choice']`, or None."""
        return dict(self._applied['record']) if self._applied else None

    def set_host_busy(self, busy: bool) -> None:
        """The host calls this while it works. The panel buttons are then off."""
        self.setEnabled(not busy)

    def _update_enabled(self) -> None:
        idle = not self._busy
        measured = self._state is not None
        has_model = measured and self._state['measured']['model'] is not None
        self.measure_btn.setEnabled(idle)
        self.crop_chk.setEnabled(idle)
        self.crop_spin.setEnabled(idle)
        for w in (self.compare_btn, self.add_btn, self.cand_list, self.extra_edit,
                  self.plane_combo, self.slice_spin):
            w.setEnabled(idle and bool(has_model))
        self.apply_btn.setEnabled(idle and self.apply_combo.count() > 0)
        self.clear_btn.setEnabled(idle and self._applied is not None)
        self.measure_cancel_btn.setEnabled(self._busy and self._step == 'Measure')
        self.compare_cancel_btn.setEnabled(self._busy and self._step == 'Compare')

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._update_enabled()
        self.busy_changed.emit(busy)

    def _update_applied_label(self) -> None:
        if self._applied:
            self.applied_label.setText(f"Applied: '{self._applied['name']}'. The symmetrise step "
                                       'of this window uses this option.')
            self.applied_label.setStyleSheet('color: #070; font-weight: bold;')
        else:
            self.applied_label.setText('The panel has no applied option. The symmetrise step '
                                       'uses the settings of this window.')
            self.applied_label.setStyleSheet('color: #555;')

    # ── jobs ──

    def _step_label(self) -> QLabel:
        """The label of the running step: Measure -> model_label, Compare -> suggest_label."""
        return self.suggest_label if self._step == 'Compare' else self.model_label

    def _run(self, step: str, func: Callable[..., Any], args: tuple[Any, ...],
             done: Callable[[Any], None]) -> None:
        self._step = step
        self._cancel_flag = False
        self._saved_text = self._step_label().text()
        self._step_label().setText(f'{step} is running.')
        self._set_busy(True)
        if self.synchronous:
            try:
                result = func(*args, self.is_cancelled, log=self._log_fn,
                              progress=self._progress_fn)
            except av.Cancelled:
                self._on_cancelled()
                return
            except Exception as e:
                self._on_error(str(e))
                return
            self._finish(done, result)
            return
        self._worker = PanelWorker(func, *args, self.is_cancelled)
        self._worker.log_msg.connect(self._log_fn)
        self._worker.progress.connect(self._progress_fn)
        self._worker.finished.connect(lambda r: self._finish(done, r))
        self._worker.error.connect(self._on_error)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.start()

    def _finish(self, done: Callable[[Any], None], result: Any) -> None:
        self._set_busy(False)
        self._step = None
        done(result)

    def cancel(self) -> None:
        """Stop the running Measure or Compare at the next check."""
        if self._busy:
            self._cancel_flag = True
            self._step_label().setText(f'{self._step} stops after the current step.')
            self._log_fn(f'Cancel: {self._step} stops after the current step.')

    def is_cancelled(self) -> bool:
        return self._cancel_flag

    def _on_cancelled(self) -> None:
        label, step = self._step_label(), self._step
        self._set_busy(False)
        self._step = None
        label.setText(f'Cancelled. {self._saved_text}'.strip())
        if step == 'Compare':
            self._mark_rows_old(False)
        self._log_fn(f'{step} cancelled. The panel keeps the earlier results.')
        self._update_enabled()

    def _on_error(self, msg: str) -> None:
        label, step = self._step_label(), self._step or 'Measure'
        self._set_busy(False)
        self._step = None
        first = msg.split('\n', 1)[0]
        self._log_fn(f'Averaging panel error: {step} failed: {msg}')
        text = f'{step} failed: {first}'
        if step == 'Compare' and self.result_table.rowCount():
            text += ' The grey rows show an older Compare.'
        elif step == 'Measure' and self._state is not None:
            text += ' The panel keeps the earlier results.'
        label.setText(text)
        self._update_enabled()

    def _mark_rows_old(self, old: bool) -> None:
        """Show the rows of the Results table in grey (old) or in the normal colour."""
        for i in range(self.result_table.rowCount()):
            for j in range(self.result_table.columnCount()):
                item = self.result_table.item(i, j)
                if item is None:
                    continue
                if old:
                    item.setForeground(QBrush(OLD_ROWS))
                else:
                    item.setData(Qt.ItemDataRole.ForegroundRole, None)

    # ── Measure ──

    def _measure(self) -> None:
        try:
            laue, volume_fn = self._request()
        except ValueError as e:
            self.model_label.setText(str(e))
            self._log_fn(str(e))
            return
        crop: int | str | None = None
        if self.crop_chk.isChecked():
            crop = self.crop_spin.value() or 'auto'
        from .volume_builder import HAS_GPU
        self._run('Measure', _measure_job, (volume_fn, laue, crop, HAS_GPU), self._on_measured)

    def _on_measured(self, state: dict[str, Any]) -> None:
        self._measured_once = True
        self.reset('')
        self._state = state
        m = state['measured']
        text = av.format_measure(m).split('\n', 1)[0]
        self.model_label.setText(f"{text}\nRegion: {state['region']}.")
        self._log_fn(av.format_measure(m))
        for i, row in enumerate(av.shell_rows(m)):
            for j, cell in enumerate(row):
                item = QTableWidgetItem(cell)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.shell_table.setItem(i, j, item)
        unit = m['diagnostics'].get('q_unit', '')
        self.shell_table.horizontalHeaderItem(0).setText(f'|q| ({unit})')
        self.cand_list.clear()
        auto = av.automatic_candidates(m['has_counts'])
        for c in auto:
            self._add_item(c, checked=True, fixed=c['name'] == av.BIAS_REFERENCE)
        for c in av.more_candidates(m['has_counts']):
            self._add_item(c, checked=False)
        self.apply_combo.clear()
        if m['model'] is None:
            name, reason = av.preselect(m['has_counts'], 'The noise-model fit failed. '
                                        + str(m['fit_error']))
            self._suggestion = (name, reason)
            self.suggest_label.setText(f'Suggestion: {name}. {reason}')
            self.apply_combo.addItem(name)
        else:
            self.suggest_label.setText('Click Compare to get a suggestion.')
        self._update_enabled()

    # ── Compare ──

    def _add_item(self, spec: dict[str, Any], checked: bool, fixed: bool = False) -> None:
        label = spec['name'] + ('  (reference)' if fixed else '')
        item = QListWidgetItem(label)
        item.setData(Qt.ItemDataRole.UserRole, spec)
        flags = item.flags() | Qt.ItemFlag.ItemIsUserCheckable
        if fixed:
            flags &= ~Qt.ItemFlag.ItemIsUserCheckable
            item.setToolTip('The reference of the sum checks. Compare always uses it.')
        item.setFlags(flags)
        item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        self.cand_list.addItem(item)

    def _candidates(self) -> list[dict[str, Any]]:
        out = []
        for i in range(self.cand_list.count()):
            item = self.cand_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                out.append(dict(item.data(Qt.ItemDataRole.UserRole)))
        return out

    def _add_custom(self) -> None:
        spec = av.make_candidate(self.add_weight.currentText(), self.add_rej.currentText(),
                                 self.add_sigma.value())
        for i in range(self.cand_list.count()):
            item = self.cand_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole)['name'] == spec['name']:
                item.setCheckState(Qt.CheckState.Checked)
                return
        self._add_item(spec, checked=True)

    def _compare(self) -> None:
        if self._state is None:
            return
        try:
            extra = parse_positions(self.extra_edit.text())
        except ValueError as e:
            self.suggest_label.setText(str(e))
            return
        st = self._state
        from .volume_builder import HAS_GPU
        args = (st['work'], st['laue'], st['measured'], self._candidates(), extra,
                self.plane_combo.currentIndex(), self.slice_spin.value(), HAS_GPU)
        self._mark_rows_old(True)
        self._run('Compare', self._compare_job, args, self._on_compared)

    @staticmethod
    def _compare_job(work: Any, laue: str, measured: dict[str, Any], cands: list[dict[str, Any]],
                     extra: list[tuple[float, float, float]], plane: int, value: float,
                     use_gpu: bool, cancelled: Callable[[], bool], log: Callable[[str], None],
                     progress: Callable[[int, int], None]) -> dict[str, Any]:
        return av.compare_candidates(work, laue, measured, cands, extra, plane, value,
                                     use_gpu=use_gpu, log=log, progress=progress,
                                     cancelled=cancelled)

    def _on_compared(self, compared: dict[str, Any]) -> None:
        self._compared = compared
        self._suggestion = (compared['suggestion'], compared['reason'])
        self._log_fn(av.format_compare(compared))
        rows = av.compare_rows(compared)
        self.result_table.setRowCount(0)
        self.result_table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, cell in enumerate(row):
                item = QTableWidgetItem(cell)
                if j:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                          | Qt.AlignmentFlag.AlignVCenter)
                if row[0] == compared['suggestion']:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                self.result_table.setItem(i, j, item)
        if compared['extra_positions']:
            self.result_table.horizontalHeaderItem(3).setToolTip(
                'Sum ratio at ' + ', '.join(av._offset_label(o)
                                            for o in compared['extra_positions']))
        else:
            self.result_table.horizontalHeaderItem(3).setToolTip(EXTRA_TIP)
        self.suggest_label.setText(f"Suggestion: {compared['suggestion']}. {compared['reason']}")
        self.apply_combo.clear()
        names = [r['name'] for r in compared['rows']]
        if compared['suggestion'] not in names:          # the preselection, not compared
            names.append(compared['suggestion'])
        self.apply_combo.addItems(names)
        self.apply_combo.setCurrentText(compared['suggestion'])
        sel = names.index(compared['suggestion'])
        if sel < self.result_table.rowCount():
            self.result_table.selectRow(sel)
        else:
            self._plot(av.BIAS_REFERENCE)
        self._update_enabled()

    def _on_row_selected(self) -> None:
        rows = self.result_table.selectionModel().selectedRows()
        if rows and self._compared:
            self._plot(self._compared['rows'][rows[0].row()]['name'])

    def _plot(self, name: str) -> None:
        c = self._compared
        if c is None or name not in c['slices']:
            return
        info = c['slice_info']
        img = c['slices'][name]
        diff = img - c['slices'][av.BIAS_REFERENCE]
        xl, yl, fl = info['labels']
        geo = info['geometry']
        X, Y, m1, m2 = _slice_mesh(geo, img.shape)
        self.figure.clear()
        ax1, ax2 = self.figure.subplots(1, 2, sharex=True, sharey=True)
        fin = img[np.isfinite(img)]
        lo, hi = (np.percentile(fin, [1, 99.5]) if fin.size else (0.0, 1.0))
        im1 = ax1.pcolormesh(X, Y, np.ma.masked_invalid(img), cmap='viridis', shading='flat',
                             vmin=lo, vmax=hi if hi > lo else lo + 1)
        ax1.set_title(f'{name}, {fl} = {info["value"]:g}', fontsize=9)
        dfin = np.abs(diff[np.isfinite(diff)])
        lim = float(np.percentile(dfin, 99.5)) if dfin.size else 1.0
        lim = lim if lim > 0 else 1.0
        im2 = ax2.pcolormesh(X, Y, np.ma.masked_invalid(diff), cmap='RdBu_r', shading='flat',
                             vmin=-lim, vmax=lim)
        ax2.set_title(f'{name}\nminus {av.BIAS_REFERENCE}' if name != av.BIAS_REFERENCE
                      else 'difference (this is the reference)', fontsize=9)
        M = np.asarray(geo['M'], np.float64)
        for ax, colour in ((ax1, 'w'), (ax2, 'k')):
            _miller_lines(ax, M, m1, m2, colour)
            ax.set_xlim(float(X.min()), float(X.max()))
            ax.set_ylim(float(Y.min()), float(Y.max()))
            ax.set_aspect('equal')
            if geo['cartesian']:
                ax.set_xlabel(f'q along {xl}* (1/Å)', fontsize=8)
                ax.set_ylabel(f'q normal to {xl}* (1/Å)', fontsize=8)
            else:
                ax.set_xlabel(f'{xl} (hkl units)', fontsize=8)
                ax.set_ylabel(f'{yl} (hkl units)', fontsize=8)
            ax.tick_params(labelsize=8)
        if geo['cartesian']:
            note = f'Reciprocal space with the UB metric. Lines: integer {xl} and {yl}.'
        else:
            note = f'No UB matrix: the axes are in hkl units. Lines: integer {xl} and {yl}.'
        self.figure.suptitle(note, fontsize=8)
        self.figure.colorbar(im1, ax=ax1, shrink=0.9)
        self.figure.colorbar(im2, ax=ax2, shrink=0.9)
        self.canvas.draw_idle()

    # ── Apply ──

    def _apply(self) -> None:
        if self._state is None or self.apply_combo.count() == 0:
            return
        name = self.apply_combo.currentText()
        spec = ((self._compared or {}).get('specs', {}).get(name) or av.known_candidate(name))
        if spec is None:
            return
        sugg, sugg_reason = self._suggestion or (None, '')
        if name == sugg:
            reason = sugg_reason
        else:
            reason = (f"The user selected '{name}'. The suggestion was '{sugg}'."
                      if sugg else f"The user selected '{name}'.")
        record = av.choice_record(name, reason, self._state['measured'], self._compared,
                                  self._state['region'])
        self._applied = {'name': name, 'spec': dict(spec), 'record': record}
        self._update_applied_label()
        self._update_enabled()
        self._log_fn(f"Averaging option applied: '{name}'. {reason}")
        self.applied_changed.emit()

    def clear_applied(self) -> None:
        if self._applied is None:
            return
        self._applied = None
        self._update_applied_label()
        self._update_enabled()
        self._log_fn('The panel removed the applied averaging option. The window uses its own '
                     'settings.')
        self.applied_changed.emit()
