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

Frame tools (raw-frame data only): the group is hidden until the host calls
`enable_frame_tools(frame_request)`. The volume builder does not call it.
`frame_request()` runs in the GUI thread. It returns a dict with these keys:
'setup' (`rawrecon.ReconSetup`), 'raw' (the uncorrected VolumeData), 'laue',
'use_gpu' and 'bin' (a function that bins a volume with the factors of the
host). If the tools cannot run, it raises ValueError with a message for the
user. The group makes
corrected volumes (ring mask, ring mask + frame model), the odd-frame and
even-frame halves of the selected volume and a frame map. The host gets the
selected volume with `selected_volume()` (None = the uncorrected volume) and
the signal `volume_changed` when the selection changes.
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
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure

from . import averaging as av
from . import orbit_engine as oe
from .averaging import FRAME_MAP_DEFINITION

# Orbit members (voxels x operations) of the automatic crop. With the NumPy
# engine, the slowest candidate ('adaptive, model 4') then takes less than
# about one minute (measured: 5.3 s for 60 million members, m-3m).
CROP_BUDGET = 500_000_000

# Height (px) of the progress bar and the log below the tabs of the host GUIs
LOG_HEIGHT = 230

EXTRA_TIP ='Sum ratio at the extra positions. The last Compare had no extra positions.'
OLD_ROWS = QColor('#999999')

# Frame tools: the steps, the volume slots and their names
FRAME_STEPS = ('Split-half check', 'Frame map', 'Ring mask', 'Frame model')
SLOT_NAMES = {'none': 'no correction', 'mask': 'ring mask', 'model': 'ring mask + frame model'}
HALVES_RULE = 'frame-number parity: odd numbers, then even numbers'
SPLIT_HEADER = 'split-half'
SPLIT_TIP = ('Split-half difference (total) of the last Split-half check. Lower is better.\n'
             + av.SPLIT_HALF_DEFINITION)
FRAME_MAP_RANGE = 1.5   # the colour scale of the frame map goes from 1 / 1.5 to 1.5
# Rows of the drawn frame map. The axes are at least about 200 px high (canvas
# 300 px), so each drawn row gets at least one screen pixel.
FRAME_MAP_ROWS = 200
NO_RING_HINT = ('The ring detector needs frames without the rings before or after the ring '
                'event in the selected frames. If all selected frames have the rings, it finds '
                'no ring. The ring mask depends on the selected frames. Check the number of '
                'ring frames after a change of the frame count.')
FRAME_HELP = (
    'Volume: the volume that Measure, Compare and the symmetrise step of this window use. '
    'Ring mask: one pass finds powder rings in the frames. One more pass makes a volume '
    'without the ring pixels. Frame model: the ring mask, then 3 iterations of a model of '
    'the extra counts and the scale of each frame and 5-degree sector (6 more passes). '
    'Frame map: one pass compares each frame with the leave-one-out orbit median of the '
    'selected volume. The first use also needs the ring statistics pass. The frames are not '
    'corrected with the frame model in this pass. ' + FRAME_MAP_DEFINITION + ' The colour '
    'scale is logarithmic. White is 1, red is more counts and blue is fewer counts than in '
    f'the other frames. The scale stops at 1/{FRAME_MAP_RANGE:g} and {FRAME_MAP_RANGE:g}. '
    'Larger deviations get the end colour. Grey cells have no prediction. On a long scan, '
    'one image row shows a block of frames. The row shows the value of the block that is '
    'farthest from 1. ' + NO_RING_HINT + ' Split-half check: one pass '
    'makes the '
    'odd-frame and the even-frame halves of the selected volume. Each candidate is then '
    'applied to each half. '
    'Split-half difference: ' + av.SPLIT_HALF_DEFINITION + ' Lower is better. It does not '
    'see contamination that is the same in the two halves. A new reconstruction, a new '
    'folder, grid or frame count removes the results of the frame tools. A new Laue group '
    'removes the frame model volume and the frame map. Validated scope: Laue group mmm and '
    '0.2 degree frames (I15 and I19-2 data). Other Laue groups and frame widths work, but '
    'they are not validated.')
FRAME_TOOLS_PLACE = ('The Frame tools group is below Measure, Compare and Results. '
                     'Scroll down to see it.')
FRAME_MAP_EMPTY = 'Frame map: click Frame map or Frame model. The map shows here.'


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
        stopped = True
        try:
            self.finished.emit(self.func(*self.args, log=self.log_msg.emit,
                                         progress=self.progress.emit))
            stopped = False
        except av.Cancelled:
            self.cancelled.emit()
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")
        if stopped:
            # The arrays of the stopped job are free now. Give their GPU memory back.
            oe._free_gpu_memory()


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
            'laue': laue, 'measured': measured,
            'crop': int(crop) if work is not vol else None}


# ── frame tools: helpers ──

def reduce_frame_rows(nr: Any, max_rows: int = FRAME_MAP_ROWS) -> tuple[Any, int]:
    """Frame map rows for the display and the number of frames per row.

    If the map has more than `max_rows` frames, each row is a block of
    frames. Per block and sector, the row keeps the finite value with the
    largest |log(value)|. Thus a short event stays visible. A block without
    a finite value stays NaN.
    """
    n, ns = nr.shape
    nb = max(1, -(-n // max_rows))
    if nb == 1:
        return nr, 1
    m = -(-n // nb)
    pad = np.full((m * nb, ns), np.nan, np.float32)
    pad[:n] = nr
    blk = pad.reshape(m, nb, ns)
    with np.errstate(divide='ignore', invalid='ignore'):
        dev = np.abs(np.log(blk))
    dev = np.where(np.isfinite(dev), dev, -1.0)
    idx = dev.argmax(axis=1)
    return np.take_along_axis(blk, idx[:, None, :], axis=1)[:, 0, :], nb


def _gpu_free_bytes(device: int = 0) -> int | None:
    """Free GPU memory in bytes (with the free blocks of the CuPy pool), or None."""
    try:
        import cupy as cp
        with cp.cuda.Device(device):
            free = cp.cuda.runtime.memGetInfo()[0]
            return int(free + cp.get_default_memory_pool().free_bytes())
    except Exception:
        return None


def halves_gpu_note(n_voxels: int, free_bytes: int | None) -> str | None:
    """A note if the GPU accumulators of a halves pass need more than 0.7 x the free memory.

    The same limit as the memory warning of the reconstruction GUI. None if
    they fit or if the free memory is not known."""
    from .rawrecon.engine import gpu_memory_bytes
    need = gpu_memory_bytes(n_voxels, halves=True)
    if free_bytes is None or need <= 0.7 * free_bytes:
        return None
    return (f'Split-half check: the halves pass needs about {need / 1e9:.1f} GB of GPU memory '
            f'(two halves, 40 bytes per voxel). The GPU has {free_bytes / 1e9:.1f} GB free. '
            'The halves pass uses the CPU.')


# ── frame tools: jobs (worker thread, no change to the panel) ──

def _rings_job(req: dict[str, Any], rings: dict[str, Any] | None, cancelled: Callable[[], bool],
               log: Callable[[str], None], progress: Callable[[int, int], None],
               found: dict[str, Any] | None = None) -> dict[str, Any]:
    """The kept ring detection, or a new one (one pass over the frames).

    A new detection also goes into found['rings'] at once. The panel then
    keeps it if a later stage of the job stops."""
    if rings is not None:
        return rings
    from .rawrecon import frames as ft
    from .rawrecon import rings as rg
    from .rawrecon.frame_model import frame_scale_of
    setup = req['setup']
    log('Ring statistics: one pass over the frames.')
    polar = rg.PixelPolar.of(setup.detector)
    stats = rg.ring_statistics(setup, use_gpu=req['use_gpu'], progress=progress,
                               cancelled=cancelled, log=log, polar=polar)
    det = rg.detect_rings(stats, frame_scale=frame_scale_of(setup.increment))
    summary = ft._jsonable(dict(mode='detected', **det.summary(stats.n1)))
    del stats
    radii = sorted({float(t['r']) for _, _, tab in det.windows for t in tab})
    log(f"Rings: {summary['ring_windows']} event(s), {summary['ring_frames']} frames, "
        f"{len(radii)} ring radii.")
    out = {'mask': rg.RingMask.from_detection(polar, det) if det.found else None,
           'summary': summary, 'radii_px': radii,
           'radii_q': [float(rg.q_of_radius(setup.detector, r)) for r in radii]}
    if found is not None:
        found['rings'] = out
    return out


def _ring_record(vols: list[Any], rings: dict[str, Any]) -> None:
    """Put the ring detection into metadata['frame_tools']['rings'] of each volume."""
    for vd in vols:
        info = vd.metadata.get('frame_tools')
        if info is None:
            continue
        rec = dict(rings['summary'])
        frac = (info.get('rings') or {}).get('masked_pixel_frac_total')
        if frac is not None:
            rec['masked_pixel_frac_total'] = frac
        info['rings'] = rec


def _ring_mask_job(req: dict[str, Any], rings: dict[str, Any] | None, found: dict[str, Any],
                   cancelled: Callable[[], bool], log: Callable[[str], None],
                   progress: Callable[[int, int], None]) -> dict[str, Any]:
    """Ring detection (if not kept) and the volume with the ring mask."""
    from .rawrecon import frames as ft
    rings = _rings_job(req, rings, cancelled, log, progress, found)
    res = ft.run_frame_chain(req['setup'], req['laue'], rings=rings['mask'], frame_model=False,
                             raw=req['raw'], use_gpu=req['use_gpu'], progress=progress,
                             cancelled=cancelled, log=log)
    _ring_record([res.corrected], rings)
    return {'rings': rings, 'mask': res.corrected}


def _frame_model_job(req: dict[str, Any], rings: dict[str, Any] | None, found: dict[str, Any],
                     need_mask: bool, cancelled: Callable[[], bool], log: Callable[[str], None],
                     progress: Callable[[int, int], None]) -> dict[str, Any]:
    """Ring detection (if not kept) and the frame chain (ring mask + frame model, 3 iterations)."""
    from .rawrecon import frames as ft
    rings = _rings_job(req, rings, cancelled, log, progress, found)
    res = ft.run_frame_chain(req['setup'], req['laue'], rings=rings['mask'], frame_model=True,
                             iterations=3, raw=req['raw'], use_gpu=req['use_gpu'],
                             keep_tables=True, keep_masked=need_mask, progress=progress,
                             cancelled=cancelled, log=log)
    mask = res.masked if need_mask else None
    _ring_record([res.corrected] + ([mask] if mask is not None else []), rings)
    return {'rings': rings, 'model': res.corrected, 'mask': mask, 'tables': res.model,
            'map': res.frame_map}


def _frame_map_job(req: dict[str, Any], rings: dict[str, Any] | None, found: dict[str, Any],
                   volume: Any, use_mask: bool, cancelled: Callable[[], bool],
                   log: Callable[[str], None],
                   progress: Callable[[int, int], None]) -> dict[str, Any]:
    """Ring detection (if not kept) and the frame map of `volume`."""
    from .rawrecon import frames as ft
    rings = _rings_job(req, rings, cancelled, log, progress, found)
    log('Frame map: one pass compares each frame with the leave-one-out orbit median.')
    fmap = ft.frame_map(req['setup'], volume, req['laue'],
                        mask=rings['mask'] if use_mask else None, rings=rings['radii_px'],
                        use_gpu=req['use_gpu'], progress=progress, cancelled=cancelled, log=log)
    return {'rings': rings, 'map': fmap}


def _split_job(req: dict[str, Any], slot: str, rings: dict[str, Any] | None, tables: Any,
               halves: list[Any] | None, laue: str, crop: int | None,
               cands: list[dict[str, Any]], cancelled: Callable[[], bool],
               log: Callable[[str], None], progress: Callable[[int, int], None]) -> dict[str, Any]:
    """Halves of the volume in `slot` (one pass, if not kept) and their split-half differences."""
    from .rawrecon import frames as ft
    from .rawrecon.engine import reconstruct_to_volumedata
    if halves is None:
        from .rawrecon.engine import _checked_halves
        setup = req['setup']
        groups = _checked_halves(setup)
        log(f"Split halves: one pass makes the odd-frame and the even-frame halves of "
            f"'{SLOT_NAMES[slot]}'.")
        if slot == 'model' and tables is None:
            raise ValueError('The frame model tables are not available. Click Frame model again.')
        mask = rings['mask'] if (rings is not None and slot != 'none') else None
        model = tables if slot == 'model' else None
        vol = ft.corrected_reconstruction(setup, model=model, mask=mask, groups=groups,
                                          use_gpu=req.get('halves_gpu', req['use_gpu']),
                                          progress=progress, cancelled=cancelled)
        halves = [reconstruct_to_volumedata(p) for p in vol.parts]
        del vol
    work = [req['bin'](h) for h in halves]
    if crop:
        work = [av.centred_crop(w, crop) for w in work]
    split = av.compare_split_halves(work[0], work[1], laue, cands, use_gpu=req['use_gpu'],
                                    log=log, progress=progress, cancelled=cancelled)
    split['halves_rule'] = HALVES_RULE
    split['volume'] = SLOT_NAMES[slot]
    return {'halves': halves, 'split': split}


class AveragingPanel(QWidget):
    """Measure, compare and apply averaging options (see the module docstring)."""

    busy_changed = pyqtSignal(bool)
    applied_changed = pyqtSignal()
    volume_changed = pyqtSignal()        # frame tools: the selected volume changed

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
        # Frame tools (see the module docstring). The panel keeps at most the
        # volumes of the slots 'mask' and 'model' and the two halves of the
        # selected slot. The host keeps the uncorrected volume (slot 'none').
        self._frame_request: Callable[[], dict[str, Any]] | None = None
        self._slot = 'none'                      # the selected volume
        self._fvols: dict[str, Any] = {}         # slot -> VolumeData (not binned)
        self._fnotes: dict[str, str] = {}        # slot -> note for the volume list
        self._rings: dict[str, Any] | None = None    # output of _rings_job
        self._found: dict[str, Any] = {}         # ring detection of the running job
        self._released: str | None = None        # slot that the running job removed first
        self._tables: Any = None                 # FrameModel of the slot 'model'
        self._halves: tuple[str, list[Any]] | None = None   # (slot, [odd, even]), not binned
        self._split: dict[str, Any] | None = None    # compare_split_halves of the measured region
        self._fmap: tuple[str, Any, str] | None = None    # (title, FrameMap, source slot)
        self._fsummary: dict[str, dict[str, Any]] = {}   # slot -> values for the volume table
        self._build_ui()
        self.reset('')

    # ── UI ──

    def _build_ui(self) -> None:
        page = QVBoxLayout(self)
        outer = QHBoxLayout()
        page.addLayout(outer, stretch=1)
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
        # 'compressed' keeps the axis labels of the equal-aspect slices inside the
        # canvas after a resize ('constrained' can cut the left label).
        self.figure = Figure(figsize=(6.4, 2.8), layout='compressed')
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

        # Frame tools (raw-frame data only, hidden until enable_frame_tools)
        self.frame_group = self._build_frame_group()
        self.frame_group.setVisible(False)
        page.addWidget(self.frame_group)

    def _build_frame_group(self) -> QGroupBox:
        g = QGroupBox('Frame tools')
        row = QHBoxLayout(g)
        v = QVBoxLayout()
        row.addLayout(v, stretch=4)
        self.frame_help = QLabel(FRAME_HELP)
        self.frame_help.setWordWrap(True)
        self.frame_help.setStyleSheet('color: #555;')
        v.addWidget(self.frame_help)
        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel('Volume:'))
        self.volume_combo = QComboBox()
        self.volume_combo.setMinimumWidth(200)
        self.volume_combo.setToolTip('The volume for Measure, Compare and the symmetrise step.\n'
                                     'A change removes the results of Measure and Compare.')
        self.volume_combo.currentIndexChanged.connect(self._on_volume_selected)
        sel_row.addWidget(self.volume_combo, stretch=1)
        v.addLayout(sel_row)
        b_row = QHBoxLayout()
        self.split_btn = QPushButton('Split-half check')
        self.split_btn.setToolTip('Make the odd-frame and the even-frame halves of the selected\n'
                                  'volume (one pass) and calculate the split-half difference of\n'
                                  'each candidate. Click Measure first.\n'
                                  'The halves pass needs about 120 bytes per voxel of host\n'
                                  'memory at its peak (about 55 bytes for a plain pass).\n'
                                  'On the GPU it needs 40 bytes per voxel (20 bytes for a\n'
                                  'plain pass). If that is more than 70 % of the free GPU\n'
                                  'memory, the halves pass uses the CPU.')
        self.map_btn = QPushButton('Frame map')
        self.map_btn.setToolTip('Compare each frame and 5-degree sector with the leave-one-out\n'
                                'orbit median of the selected volume (one pass). The first use\n'
                                'also runs the ring statistics pass. The frames are not\n'
                                'corrected with the frame model in this pass.')
        self.ring_btn = QPushButton('Ring mask')
        self.ring_btn.setToolTip('Find powder rings and make a volume without the ring pixels.')
        self.model_btn = QPushButton('Frame model')
        self.model_btn.setToolTip('Ring mask and frame model (3 iterations). This makes the\n'
                                  "volume 'ring mask + frame model'.\n"
                                  'Host memory at the peak (measured): 6.0 GB more than before\n'
                                  'the run for 1735 frames, 1785 radial bins and a grid of\n'
                                  '297 x 297 x 241 voxels.')
        self.frame_cancel_btn = QPushButton('Cancel')
        self.frame_cancel_btn.setToolTip('Stop the frame tool after the current frame.\n'
                                         'The panel keeps the earlier results and a finished\n'
                                         'ring detection. A new Ring mask or Frame model run\n'
                                         'removes the earlier volume of that step when it starts.')
        self.split_btn.clicked.connect(self._split_check)
        self.map_btn.clicked.connect(self._frame_map)
        self.ring_btn.clicked.connect(self._ring_mask)
        self.model_btn.clicked.connect(self._frame_model)
        self.frame_cancel_btn.clicked.connect(self.cancel)
        for w in (self.split_btn, self.map_btn, self.ring_btn, self.model_btn,
                  self.frame_cancel_btn):
            b_row.addWidget(w)
        v.addLayout(b_row)
        self.frame_label = QLabel()
        self.frame_label.setWordWrap(True)
        self.frame_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.frame_label)
        self.volumes_table = QTableWidget(0, 5)
        self.volumes_table.setHorizontalHeaderLabels(
            ['volume', 'suggestion', 'score', SPLIT_HEADER, 'split diffuse'])
        ref = av.BIAS_REFERENCE
        for j, tip in ((2, f"Score of '{ref}' in the last Compare of this volume."),
                       (3, f"Split-half difference (total) of '{ref}'."),
                       (4, f"Split-half difference of '{ref}' outside the nodes.")):
            self.volumes_table.horizontalHeaderItem(j).setToolTip(tip)
        self._setup_table(self.volumes_table)
        self.volumes_table.setFixedHeight(20 * 3 + 30)
        v.addWidget(self.volumes_table)
        self.split_table = QTableWidget(0, 0)
        self.split_table.setToolTip(av.SPLIT_HALF_DEFINITION)
        self._setup_table(self.split_table)
        self.split_table.setFixedHeight(20 * (3 + oe.N_SHELLS) + 30)
        self.split_table.setVisible(False)       # shown when a split-half result exists
        v.addWidget(self.split_table)
        self.frame_figure = Figure(figsize=(6.4, 3.0), layout='constrained')
        self.frame_canvas = FigureCanvasQTAgg(self.frame_figure)
        self.frame_canvas.setMinimumHeight(300)
        row.addWidget(self.frame_canvas, stretch=5)
        return g

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
        self._split = None
        self._show_split()
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
        for w in (self.volume_combo, self.map_btn, self.ring_btn, self.model_btn):
            w.setEnabled(idle)
        self.split_btn.setEnabled(idle and measured)
        self.frame_cancel_btn.setEnabled(self._busy and self._step in FRAME_STEPS)

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
        """The label of the running step: Measure -> model_label, Compare -> suggest_label,
        a frame tool -> frame_label."""
        if self._step in FRAME_STEPS:
            return self.frame_label
        return self.suggest_label if self._step == 'Compare' else self.model_label

    def _run(self, step: str, func: Callable[..., Any], args: tuple[Any, ...],
             done: Callable[[Any], None]) -> None:
        self._step = step
        self._cancel_flag = False
        saved = self._step_label().text()
        while saved.startswith('Cancelled.'):           # no repeated 'Cancelled.'
            saved = saved[len('Cancelled.'):].lstrip()
        self._saved_text = saved
        self._step_label().setText(f'{step} is running.')
        self._set_busy(True)
        if self.synchronous:
            was_cancelled, failure = False, None
            try:
                result = func(*args, self.is_cancelled, log=self._log_fn,
                              progress=self._progress_fn)
            except av.Cancelled:
                was_cancelled = True
            except Exception as e:
                failure = str(e)
            if was_cancelled or failure is not None:
                oe._free_gpu_memory()       # the arrays of the stopped job are free now
                if was_cancelled:
                    self._on_cancelled()
                else:
                    self._on_error(failure)
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
        self._found, self._released = {}, None
        done(result)

    def cancel(self) -> None:
        """Stop the running Measure or Compare at the next check."""
        if self._busy:
            self._cancel_flag = True
            self._step_label().setText(f'{self._step} stops after the current step.')
            self._log_fn(f'Cancel: {self._step} stops after the current step.')

    def is_cancelled(self) -> bool:
        return self._cancel_flag

    def _after_stop(self) -> str:
        """Keep a ring detection of the stopped job. Return a note on a removed volume."""
        if self._rings is None and self._found.get('rings') is not None:
            self._rings = self._found['rings']
        self._found = {}
        slot, self._released = self._released, None
        if slot is None:
            return ''
        return (f" The panel removed the earlier '{SLOT_NAMES[slot]}' volume before the run. "
                'Run the step again to make it.')

    def _kept_text(self) -> str:
        """The frame tools results that the panel keeps (for the label after a stop)."""
        slots = ['none'] + [s for s in ('mask', 'model') if s in self._fvols]
        text = 'The panel keeps the volumes ' + ', '.join(f"'{SLOT_NAMES[s]}'" for s in slots)
        if self._rings is not None:
            n = len(self._rings.get('radii_px') or [])
            text += (f', the ring detection ({n} ring radii)' if self._rings.get('mask') is not None
                     else ', the ring detection (no ring found)')
        if self._fmap is not None:
            text += ', the frame map'
        return text + '.'

    def _on_cancelled(self) -> None:
        label, step = self._step_label(), self._step
        self._set_busy(False)
        self._step = None
        note = self._after_stop()
        if step in FRAME_STEPS:
            # The frame label is shared by the frame steps: do not show the text of an
            # earlier step. Say which step stopped and what the panel keeps.
            label.setText(f'{step} cancelled.{note} {self._kept_text()}')
        else:
            label.setText(f'Cancelled. {self._saved_text}'.strip())
        if step == 'Compare':
            self._mark_rows_old(False)
        if note:
            self._log_fn(f'{step} cancelled.{note}')
        else:
            self._log_fn(f'{step} cancelled. The panel keeps the earlier results.')
        self._update_enabled()

    def _on_error(self, msg: str) -> None:
        label, step = self._step_label(), self._step or 'Measure'
        self._set_busy(False)
        self._step = None
        note = self._after_stop()
        first = msg.split('\n', 1)[0]
        self._log_fn(f'Averaging panel error: {step} failed: {msg}')
        text = f'{step} failed: {first}'
        if step == 'Compare' and self.result_table.rowCount():
            text += ' The grey rows show an older Compare.'
        elif step == 'Measure' and self._state is not None:
            text += ' The panel keeps the earlier results.'
        label.setText(text + note)
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
        self._show_split()
        ref = next((r for r in compared['rows'] if r['name'] == av.BIAS_REFERENCE), None)
        self._summarise(suggestion=compared['suggestion'],
                        score=None if ref is None else ref['score'])
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
                                  self._state['region'], split=self._split)
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

    # ── Frame tools: host interface ──

    def enable_frame_tools(self, request: Callable[[], dict[str, Any]]) -> None:
        """Show the frame tools. `request` is the frame request of the host (module docstring)."""
        self._frame_request = request
        self.frame_group.setVisible(True)
        if FRAME_TOOLS_PLACE not in self.engine_label.text():
            self.engine_label.setText(f'{self.engine_label.text()} {FRAME_TOOLS_PLACE}')
        self._refresh_volumes()
        self.frame_label.setText('Reconstruct a volume, then use the frame tools.')
        self._frame_map_hint()

    def _frame_map_hint(self) -> None:
        """Clear the frame map figure and show a short hint in it."""
        self.frame_figure.clear()
        self.frame_figure.text(0.5, 0.5, FRAME_MAP_EMPTY, ha='center', va='center',
                               fontsize=9, color='#555555', wrap=True)
        self.frame_canvas.draw_idle()

    def selected_slot(self) -> str:
        """'none', 'mask' or 'model': the selected volume."""
        return self._slot

    def selected_volume(self) -> Any:
        """The selected corrected volume (not binned), or None for the uncorrected volume."""
        return self._fvols.get(self._slot)

    def clear_frame_results(self, message: str = '') -> None:
        """Remove all results of the frame tools and select the uncorrected volume."""
        if not (self._fvols or self._rings or self._halves or self._fmap or self._fsummary
                or self._split):
            return
        self._fvols.clear()
        self._fnotes.clear()
        self._fsummary.clear()
        self._rings = self._tables = self._halves = self._fmap = self._split = None
        self._show_split()
        self._frame_map_hint()
        self._select('none', message or 'The frame tools results were removed.')
        self._refresh_volumes()
        if message:
            self.frame_label.setText(message)
            self._log_fn(message)

    def drop_laue_results(self, message: str) -> None:
        """Remove the frame tools results that use the Laue group.

        The host calls this when the Laue group changes. The prediction of the
        frame model and of the frame map uses the Laue group. The panel removes
        the 'ring mask + frame model' volume, its tables and halves, and the
        frame map. The ring detection and the 'ring mask' volume stay. The
        values of Measure, Compare and Split-half in the volume table go.
        """
        self.clear_scores()
        if 'model' not in self._fvols and self._fmap is None:
            return
        if self._slot == 'model':
            self._select('none', message)
        if self._halves is not None and self._halves[0] == 'model':
            self._halves = None
        self._fvols.pop('model', None)
        self._fnotes.pop('model', None)
        self._fsummary.pop('model', None)
        self._tables = None
        self._fmap = None
        self._frame_map_hint()
        self._refresh_volumes()
        self.frame_label.setText(message)
        self._log_fn(message)

    def clear_scores(self) -> None:
        """Remove the suggestion, score and split-half values of all volumes in the volume table.

        The host calls this when the Laue group or the bin factors change.
        The volumes, their notes and the ring detection stay. `reset` does not
        call this, because a volume change keeps the values of each volume.
        """
        for d in self._fsummary.values():
            for key in ('suggestion', 'score', 'split', 'split_diffuse'):
                d.pop(key, None)
        if self._frame_request is not None:
            self._refresh_volumes()

    # ── Frame tools: volumes ──

    def _refresh_volumes(self) -> None:
        """Fill the volume list and the volume table from the kept volumes."""
        slots = ['none'] + [s for s in ('mask', 'model') if s in self._fvols]
        self.volume_combo.blockSignals(True)
        self.volume_combo.clear()
        for s in slots:
            note = self._fnotes.get(s, '')
            self.volume_combo.addItem(SLOT_NAMES[s] + (f' ({note})' if note else ''), s)
        self.volume_combo.setCurrentIndex(slots.index(self._slot) if self._slot in slots else 0)
        self.volume_combo.blockSignals(False)
        self.volumes_table.setRowCount(len(slots))
        for i, s in enumerate(slots):
            d = self._fsummary.get(s, {})
            cells = [self.volume_combo.itemText(i), d.get('suggestion') or '-',
                     av._num(d.get('score')), av._num(d.get('split')),
                     av._num(d.get('split_diffuse'))]
            for j, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if j:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                          | Qt.AlignmentFlag.AlignVCenter)
                if s == self._slot:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                self.volumes_table.setItem(i, j, item)

    def _summarise(self, **values: Any) -> None:
        """Keep values of the selected volume for the volume table."""
        if self._frame_request is None:
            return
        self._fsummary.setdefault(self._slot, {}).update(values)
        self._refresh_volumes()

    def _select(self, slot: str, message: str) -> None:
        """Select a volume. A change releases the halves and resets Measure and Compare."""
        if slot == self._slot:
            return
        self._slot = slot
        if self._halves is not None and self._halves[0] != slot:
            self._halves = None
        self.reset(message)
        self._refresh_volumes()
        self.volume_changed.emit()

    def _on_volume_selected(self, index: int) -> None:
        slot = self.volume_combo.itemData(index)
        if slot is not None:
            self._select(slot, f"Volume '{SLOT_NAMES[slot]}' selected. Click Measure again.")

    def _release(self, slot: str) -> None:
        """Remove the kept volume of `slot` (and its halves) before a new run makes it again."""
        if slot not in self._fvols:
            return
        if self._slot == slot:
            self._select('none', f"The volume '{SLOT_NAMES[slot]}' is made again. "
                         "The panel selected 'no correction'.")
        if self._halves is not None and self._halves[0] == slot:
            self._halves = None
        del self._fvols[slot]
        self._fnotes.pop(slot, None)
        self._fsummary.pop(slot, None)
        if slot == 'model':
            self._tables = None
        if self._fmap is not None and self._fmap[2] == slot:
            # The map was made from this volume (or in its run): it goes with it.
            self._fmap = None
            self._frame_map_hint()
        self._released = slot
        self._refresh_volumes()
        self._log_fn(f"The panel removed the earlier '{SLOT_NAMES[slot]}' volume.")

    def _frame_req(self) -> dict[str, Any] | None:
        """The frame request of the host, or None (with a message)."""
        if self._frame_request is None:
            return None
        try:
            return self._frame_request()
        except ValueError as e:
            self.frame_label.setText(str(e))
            self._log_fn(str(e))
            return None

    # ── Frame tools: steps ──

    def _ring_mask(self) -> None:
        req = self._frame_req()
        if req is None:
            return
        self._found, self._released = {}, None
        self._release('mask')
        self._run('Ring mask', _ring_mask_job, (req, self._rings, self._found),
                  self._on_ring_mask)

    def _on_ring_mask(self, out: dict[str, Any]) -> None:
        self._rings = out['rings']
        self._fvols['mask'] = out['mask']
        found = self._rings['mask'] is not None
        self._fnotes['mask'] = '' if found else 'no ring found'
        self._refresh_volumes()
        s = self._rings['summary']
        self.frame_label.setText(
            (f"Ring mask: {s['ring_windows']} event(s), {s['ring_frames']} frames, "
             f"{len(self._rings['radii_px'])} ring radii. " if found else
             'Ring mask: no ring found. The volume is the same as the uncorrected volume. '
             + NO_RING_HINT + ' ')
            + "Select 'ring mask' in Volume to measure it.")
        self._log_fn(self.frame_label.text())
        self._update_enabled()

    def _frame_model(self) -> None:
        req = self._frame_req()
        if req is None:
            return
        self._found, self._released = {}, None
        self._release('model')
        self._run('Frame model', _frame_model_job,
                  (req, self._rings, self._found, 'mask' not in self._fvols),
                  self._on_frame_model)

    def _on_frame_model(self, out: dict[str, Any]) -> None:
        self._rings = out['rings']
        found = self._rings['mask'] is not None
        if out['mask'] is not None:
            self._fvols['mask'] = out['mask']
            self._fnotes['mask'] = '' if found else 'no ring found'
        self._fvols['model'] = out['model']
        info = out['model'].metadata.get('frame_tools') or {}
        # The prediction of the frame model uses the Laue group. The note shows it.
        self._fnotes['model'] = (('' if found else 'no ring found, frame model only, ')
                                 + f"Laue group {info.get('laue_group', '-')}")
        self._tables = out['tables']
        its = info.get('iterations_info') or []
        last = its[-1] if its else {}
        text = (f"Frame model: {info.get('iterations', 0)} iterations. Last iteration: "
                f"{last.get('frames_S', '-')} frames with a scale term, "
                f"{last.get('frames_B', '-')} frames with extra counts. "
                + ('' if found else 'No ring found: the volume has the frame model only. '
                   + NO_RING_HINT + ' ')
                + "Select 'ring mask + frame model' in Volume to measure it.")
        self.frame_label.setText(text)
        self._log_fn(text)
        if out['map'] is not None:
            title = ('Frame map of the first iteration (ring-masked volume)' if found else
                     'Frame map of the first iteration (volume without ring mask)')
            self._fmap = (title, out['map'], 'model')
            self._plot_frame_map()
        self._refresh_volumes()
        self._update_enabled()

    def _frame_map(self) -> None:
        req = self._frame_req()
        if req is None:
            return
        vol = self.selected_volume()
        vol = req['raw'] if vol is None else vol
        use_mask = self._slot != 'none'
        self._found, self._released = {}, None
        self._run('Frame map', _frame_map_job, (req, self._rings, self._found, vol, use_mask),
                  self._on_frame_map)

    def _on_frame_map(self, out: dict[str, Any]) -> None:
        self._rings = out['rings']
        if self._slot == 'model':
            # The pass does not apply the frame model to the frames.
            title = ("Frame map: uncorrected frames (ring mask only) compared with the "
                     f"prediction of '{SLOT_NAMES[self._slot]}'")
        else:
            title = f"Frame map of '{SLOT_NAMES[self._slot]}'"
        self._fmap = (title, out['map'], self._slot)
        self._plot_frame_map()
        s = out['map'].summary()
        text = (f"{self._fmap[0]}: {s['frames_off_10pct']} of {s['n_frames']} frames differ "
                'from the other frames by more than 10 % (median over the sectors of the '
                'normalised ratio). Median deviation of one cell from 1: '
                + ('-' if s['cell_dev_median'] is None else f"{100 * s['cell_dev_median']:.0f} %")
                + '. Observed / predicted before the '
                f"normalisation: median {av._num(s['excess_median'])}, range "
                f"{av._num(s['excess_min'])} to {av._num(s['excess_max'])}. "
                f"{len(self._rings['radii_px'])} ring radii (dashed lines).")
        self.frame_label.setText(text)
        self._log_fn(text)
        self._update_enabled()

    def _split_check(self) -> None:
        if self._state is None:
            self.frame_label.setText('Click Measure first. The split-half check uses the region '
                                     'and the bin factors of Measure.')
            return
        req = self._frame_req()
        if req is None:
            return
        if self._compared is not None:
            cands = [dict(self._compared['specs'][r['name']]) for r in self._compared['rows']]
        else:
            cands = self._candidates()
            if av.BIAS_REFERENCE not in [c['name'] for c in cands]:
                cands.insert(0, dict(av.EQUAL_MAD))
        halves = (self._halves[1] if self._halves is not None and self._halves[0] == self._slot
                  else None)
        if halves is None and req['use_gpu']:
            note = halves_gpu_note(req['raw'].intensity.size,
                                   _gpu_free_bytes(req['setup'].device))
            if note is not None:
                req = dict(req, halves_gpu=False)
                self._log_fn(note)
        st = self._state
        args = (req, self._slot, self._rings, self._tables, halves, st['laue'], st.get('crop'),
                cands)
        self._run('Split-half check', _split_job, args, self._on_split)

    def _on_split(self, out: dict[str, Any]) -> None:
        self._halves = (self._slot, out['halves'])
        self._split = out['split']
        self._show_split()
        ref = self._split['rows'].get(av.BIAS_REFERENCE) or {}
        self._summarise(split=ref.get('total'), split_diffuse=ref.get('diffuse'))
        lines = [f"Split-half difference of '{SLOT_NAMES[self._slot]}' ({HALVES_RULE}):"]
        for name, r in self._split['rows'].items():
            lines.append(f"  {name}: " + ('no estimate (the noise-model fit of a half failed)'
                                          if r is None else
                                          f"total {av._num(r['total'])}, diffuse "
                                          f"{av._num(r['diffuse'])}, node {av._num(r['node'])}"))
        self._log_fn('\n'.join(lines))
        self.frame_label.setText(f"Split-half check of '{SLOT_NAMES[self._slot]}' done. The "
                                 'table below and the column '
                                 f"'{SPLIT_HEADER}' of Results show the values.")
        self._update_enabled()

    # ── Frame tools: display ──

    def _show_split(self) -> None:
        """Show the split-half results in the Results table and in the shell table."""
        n = len(av.COMPARE_HEADERS)
        split = self._split
        if split is None:
            self.result_table.setColumnCount(n)
            self.split_table.setRowCount(0)
            self.split_table.setColumnCount(0)
            self.split_table.setVisible(False)
            return
        rows = split['rows']
        self.result_table.setColumnCount(n + 1)
        head = QTableWidgetItem(SPLIT_HEADER)
        head.setToolTip(SPLIT_TIP)
        self.result_table.setHorizontalHeaderItem(n, head)
        for i in range(self.result_table.rowCount()):
            name_item = self.result_table.item(i, 0)
            r = rows.get(name_item.text()) if name_item is not None else None
            item = QTableWidgetItem(av._num(None if r is None else r['total']))
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.result_table.setItem(i, n, item)
        names = list(rows)
        edges = split['shell_edges']
        labels = (['total', 'diffuse', 'node']
                  + [f"{edges[k]:.3f}-{edges[k + 1]:.3f}" for k in range(len(edges) - 1)])
        self.split_table.setColumnCount(len(names) + 1)
        self.split_table.setHorizontalHeaderLabels([f"|q| ({split['q_unit']})"] + names)
        self.split_table.setRowCount(len(labels))
        for i, lab in enumerate(labels):
            self.split_table.setItem(i, 0, QTableWidgetItem(lab))
            for j, name in enumerate(names):
                r = rows[name]
                if r is None:
                    val = None
                elif i < 3:
                    val = r[lab]
                else:
                    val = r['shells'][i - 3]
                item = QTableWidgetItem(av._num(val))
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.split_table.setItem(i, j + 1, item)
        self.split_table.setVisible(True)

    def _plot_frame_map(self) -> None:
        """The frame x sector image and the radial residual profile of the kept frame map."""
        if self._fmap is None:
            return
        title, fm, _ = self._fmap
        self.frame_figure.clear()
        ax1, ax2 = self.frame_figure.subplots(1, 2)
        nr = fm.normalised()                     # FRAME_MAP_DEFINITION
        n = nr.shape[0]
        rows, per_row = reduce_frame_rows(nr)
        c = FRAME_MAP_RANGE
        cmap = matplotlib.colormaps['RdBu_r'].copy()
        cmap.set_bad('0.65')                     # grey: no prediction
        # Sector k is centred on 5 k degrees: the image extent is -2.5 to 357.5.
        w = 360.0 / nr.shape[1]
        im = ax1.imshow(np.ma.masked_invalid(rows), aspect='auto', origin='lower',
                        interpolation='nearest', cmap=cmap, norm=LogNorm(1.0 / c, c),
                        extent=(-w / 2, 360.0 - w / 2, -0.5, rows.shape[0] * per_row - 0.5))
        ax1.set_ylim(-0.5, n - 0.5)
        ax1.set_xlabel('azimuth sector (deg)', fontsize=8)
        ax1.set_ylabel('frame position in the scan' if per_row == 1 else
                       f'frame position in the scan ({per_row} frames per row)', fontsize=8)
        ax1.set_title('obs / pred, divided by the sector median', fontsize=9)
        cb = self.frame_figure.colorbar(im, ax=ax1, shrink=0.9, extend='both')
        ticks = [t for t in (1 / 1.5, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5) if 1.0 / c <= t <= c]
        cb.set_ticks(ticks, labels=[f'{t:.3g}'.rstrip('0') for t in ticks])
        cb.minorticks_off()
        cb.ax.tick_params(labelsize=8)
        ok = np.isfinite(fm.rel_resid)
        ax2.plot(fm.q[ok], 100.0 * fm.rel_resid[ok], lw=0.8, color='tab:blue')
        ax2.axhline(0.0, color='k', lw=0.5)
        radii_q = (self._rings or {}).get('radii_q', [])
        for i, q in enumerate(radii_q):
            ax2.axvline(q, color='tab:red', lw=0.8, ls='--',
                        label='detected ring' if i == 0 else None)
        if radii_q:
            ax2.legend(fontsize=7, loc='upper right')
        ax2.set_xlabel('|q| (1/Å)', fontsize=8)
        ax2.set_ylabel('(obs - pred) / pred (%)', fontsize=8)
        ax2.set_title('radial residual, all frames', fontsize=9)
        for ax in (ax1, ax2):
            ax.tick_params(labelsize=8)
        self.frame_figure.suptitle(title, fontsize=8)
        self.frame_canvas.draw_idle()
