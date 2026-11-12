"""
view3d_window.py - the window of the 3D view.

The window shows a volume as surfaces of equal intensity or as a projection of
the largest intensity. The controls are in a panel on the left: the levels with
a histogram, the box, the view direction and the export of the image. The
drawing is done by `view3d_gl.View3DCanvas`.

    from rspace3d.view3d_window import show_volume
    show_volume(vol)                       # or a path to a .h5 file
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any

import numpy as np
from PyQt6.QtCore import QEvent, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (QColor, QFont, QImage, QKeySequence, QPainter, QPen, QPolygonF, QShortcut,
                         QValidator)
from PyQt6.QtWidgets import (
    QAbstractButton, QAbstractSpinBox, QApplication, QButtonGroup, QCheckBox, QColorDialog,
    QComboBox, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QMainWindow,
    QPushButton, QRadioButton, QScrollArea, QSpinBox, QStackedWidget, QToolButton, QVBoxLayout,
    QWidget,
)

from . import ui
from . import view3d as v3
from .volume_builder import VolumeData, load_volume_h5

COLORMAPS = ['inferno', 'magma', 'viridis', 'plasma', 'cividis', 'gnuplot2', 'hot', 'gray']
MOUSE_HINT = 'Drag: turn.  Shift + drag or right button: move.  Wheel: zoom.'
CAPTION_PT, SECTION_PT = 9.0, 10.5       # sizes of the type scale of ui.py, for text that is painted
OPACITY_WIDTH = 88                       # of the opacity field of a level ("100" must fit)
REMOVE_WIDTH = 24                        # of the round button that removes a level
SWATCH_SIZE = 28                         # of the colour button of a level
LEVEL_MIN, LEVEL_MAX = 1.0e-9, 1.0e12    # range of an intensity field
_WINDOWS: list[QMainWindow] = []         # windows that `show_volume` opened
_APP: QApplication | None = None         # the application that `show_volume` made, kept alive


def _qcolor(rgb: tuple[float, float, float]) -> QColor:
    return QColor.fromRgbF(float(rgb[0]), float(rgb[1]), float(rgb[2]))


def _decade(n: int) -> str:
    return f'{10.0 ** n:g}' if -2 <= n <= 2 else f'1e{n}'


def intensity_text(value: float) -> str:
    """An intensity with four significant digits, for example 1451, 40.81 or 0.04000."""
    if value == 0 or not math.isfinite(value):
        return '0'
    mag = int(math.floor(math.log10(abs(value))))
    if -4 <= mag < 7:
        return f'{value:.{max(3 - mag, 0)}f}'
    return f'{value:.3e}'


class NumberBox(QDoubleSpinBox):
    """A field for an intensity.

    It shows four significant digits, but it keeps each number as it was typed:
    a QDoubleSpinBox with a fixed number of decimals rounds 0.04 to 0.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setDecimals(12)                 # digits that the value keeps, not the digits on the screen
        self.setRange(LEVEL_MIN, LEVEL_MAX)
        self.setStepType(QAbstractSpinBox.StepType.AdaptiveDecimalStepType)
        self.setKeyboardTracking(False)
        self.setMinimumWidth(72)             # the range of the box would make it much wider

    def textFromValue(self, value: float) -> str:
        return intensity_text(value)

    def valueFromText(self, text: str | None) -> float:
        try:
            return float((text or '').strip().replace(',', '.'))
        except ValueError:
            return self.value()

    def validate(self, text: str | None, pos: int) -> tuple[QValidator.State, str, int]:
        t = (text or '').strip().replace(',', '.').lower()
        try:
            ok = float(t) > 0.0
        except ValueError:
            part = t in ('', '.', '+') or t.endswith(('e', 'e-', 'e+'))
            return (QValidator.State.Intermediate if part else QValidator.State.Invalid, text or '', pos)
        return (QValidator.State.Acceptable if ok else QValidator.State.Intermediate, text or '', pos)


def _set_number(spin: QDoubleSpinBox, value: float) -> None:
    """Show `value` in a field and send no signal."""
    spin.blockSignals(True)
    spin.setValue(value)
    spin.blockSignals(False)


class Swatch(QAbstractButton):
    """A button that shows the colour of a level."""

    def __init__(self, rgb: tuple[float, float, float], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.rgb = rgb
        self.setFixedSize(SWATCH_SIZE, SWATCH_SIZE)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip('Change the colour of this surface.')
        self.setAttribute(Qt.WidgetAttribute.WA_Hover)       # draw again when the mouse comes or goes

    def paintEvent(self, event: Any) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        strong = self.hasFocus() or self.isDown()
        edge = 'text' if strong else ('muted' if self.underMouse() else 'border')
        p.setPen(QPen(ui.color(edge), 2 if strong else 1))
        p.setBrush(_qcolor(self.rgb))
        inset = 2.5 if self.isDown() else 1.5
        p.drawRoundedRect(QRectF(inset, inset, self.width() - 2 * inset, self.height() - 2 * inset), 5, 5)


class LevelBar(QWidget):
    """The histogram of the intensity on a logarithmic axis, with one line for each level.

    Drag a line to change its level. `moved` gives the number of the line and
    the new intensity.
    """

    moved = pyqtSignal(int, float)
    released = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(88)
        self.setMouseTracking(True)
        self.setToolTip('Drag a line to change a level. Hold Shift for small steps.')
        self._edges = np.array([0.0, 1.0])
        self._counts = np.zeros(1)
        self._marks: list[tuple[float, QColor]] = []
        self._drag: int | None = None
        self._last_x = 0.0

    def set_histogram(self, edges: Any, counts: Any) -> None:
        self._edges, self._counts = np.asarray(edges, float), np.asarray(counts, float)
        self.update()

    def set_marks(self, marks: list[tuple[float, QColor]]) -> None:
        self._marks = list(marks)
        self.update()

    def _plot(self) -> QRectF:
        return QRectF(1.0, 12.0, self.width() - 2.0, self.height() - 12.0 - 20.0)

    def _x(self, value: float) -> float:
        r = self._plot()
        lo, hi = float(self._edges[0]), float(self._edges[-1])
        frac = (math.log10(max(value, 1.0e-300)) - lo) / (hi - lo)
        return r.left() + min(max(frac, 0.0), 1.0) * r.width()

    def _value(self, x: float) -> float:
        r = self._plot()
        lo, hi = float(self._edges[0]), float(self._edges[-1])
        frac = min(max((x - r.left()) / max(r.width(), 1.0), 0.0), 1.0)
        return 10.0 ** (lo + frac * (hi - lo))

    def _near(self, x: float) -> int | None:
        best, dist = None, 9.0
        for i, (value, _) in enumerate(self._marks):
            d = abs(self._x(value) - x)
            if d < dist:
                best, dist = i, d
        return best

    def paintEvent(self, event: Any) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self._plot()
        p.setPen(QPen(ui.color('line'), 1))
        p.setBrush(ui.color('surface'))
        p.drawRoundedRect(r.adjusted(-0.5, -0.5, 0.5, 0.5), 4, 4)
        top = float(np.log1p(self._counts).max()) if self._counts.size else 0.0
        if top > 0:
            n = len(self._counts)
            poly = QPolygonF([QPointF(r.left(), r.bottom())])
            for i, c in enumerate(np.log1p(self._counts) / top):
                y = r.bottom() - float(c) * (r.height() - 4.0)
                poly.append(QPointF(r.left() + r.width() * i / n, y))
                poly.append(QPointF(r.left() + r.width() * (i + 1) / n, y))
            poly.append(QPointF(r.right(), r.bottom()))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(ui.color('line2'))
            p.drawPolygon(poly)
        font = QFont(self.font())
        font.setPointSizeF(CAPTION_PT)
        p.setFont(font)
        lo, hi = float(self._edges[0]), float(self._edges[-1])
        decades = list(range(math.ceil(lo - 1.0e-9), math.floor(hi + 1.0e-9) + 1))
        every = 1 if len(decades) <= 6 else 2
        for n in decades:
            x = r.left() + (n - lo) / (hi - lo) * r.width()
            p.setPen(QPen(ui.color('border'), 1))
            p.drawLine(QPointF(x, r.bottom()), QPointF(x, r.bottom() + 4.0))
            if (n - decades[0]) % every == 0:
                p.setPen(ui.color('muted'))
                box = QRectF(x - 30.0, r.bottom() + 4.0, 60.0, 16.0)
                box.moveLeft(min(max(box.left(), 0.0), self.width() - 60.0))     # keep it in the widget
                align = Qt.AlignmentFlag.AlignHCenter
                if x < 30.0:
                    align = Qt.AlignmentFlag.AlignLeft
                elif x > self.width() - 30.0:
                    align = Qt.AlignmentFlag.AlignRight
                p.drawText(box, align | Qt.AlignmentFlag.AlignTop, _decade(n))
        for i, (value, colour) in enumerate(self._marks):
            x = self._x(value)
            p.setPen(QPen(colour, 3 if i == self._drag else 2))
            p.drawLine(QPointF(x, r.top() - 3.0), QPointF(x, r.bottom()))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(colour)
            p.drawPolygon(QPolygonF([QPointF(x - 6.0, r.top() - 11.0), QPointF(x + 6.0, r.top() - 11.0),
                                     QPointF(x, r.top() - 2.0)]))

    def mousePressEvent(self, event: Any) -> None:
        self._drag = self._near(event.position().x())
        self._last_x = event.position().x()
        if self._drag is not None:
            self._move(event)

    def mouseMoveEvent(self, event: Any) -> None:
        if self._drag is None:
            near = self._near(event.position().x()) is not None
            self.setCursor(Qt.CursorShape.SizeHorCursor if near else Qt.CursorShape.ArrowCursor)
            return
        self._move(event)

    def _move(self, event: Any) -> None:
        assert self._drag is not None
        x = event.position().x()
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            # small steps from the present value (which can be outside the histogram)
            per_px = (float(self._edges[-1]) - float(self._edges[0])) / max(self._plot().width(), 1.0)
            value = self._marks[self._drag][0] * 10.0 ** (0.1 * (x - self._last_x) * per_px)
        else:
            value = self._value(x)
        self._last_x = x
        self.moved.emit(self._drag, value)

    def mouseReleaseEvent(self, event: Any) -> None:
        if self._drag is not None:
            self._drag = None
            self.update()
            self.released.emit()


def draw_labels(p: QPainter, canvas: Any, width: float, height: float) -> None:
    """Write the numbers and names of h, k, l and the directions of the axes over an image."""
    grid, cam, scene = canvas.grid, canvas.camera, canvas.scene
    if grid is None:
        return
    from .view3d_gl import DARK, LIGHT
    layout = v3.axis_layout(grid, cam, scene.lo, scene.hi, width, height)
    theme = DARK if scene.dark else LIGHT
    text, soft = QColor(theme['text']), QColor(theme['soft'])
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    small = QFont(p.font())
    small.setPointSizeF(CAPTION_PT)
    bold = QFont(p.font())
    bold.setPointSizeF(SECTION_PT)
    bold.setWeight(QFont.Weight.DemiBold)
    centre = Qt.AlignmentFlag.AlignCenter
    for entry in layout:
        p.setFont(small)
        p.setPen(soft)
        for x, y, label in entry['ticks']:
            p.drawText(QRectF(x - 30.0, y - 9.0, 60.0, 18.0), centre, label)
        if entry['name'] is not None:
            p.setFont(bold)
            p.setPen(text)
            p.drawText(QRectF(entry['name'][0] - 20.0, entry['name'][1] - 10.0, 40.0, 20.0), centre,
                       entry['axis'])
    # directions of the reciprocal axes, far arrows first
    ox, oy, length = 50.0, height - 50.0, 30.0
    names = ('a*', 'b*', 'c*') if grid.has_cell else ('h', 'k', 'l')
    arrows = sorted(zip(names, v3.triad(grid, cam)), key=lambda a: -a[1][3])
    p.setFont(small)
    for name, (_, dx, dy, _) in arrows:
        p.setPen(QPen(text, 1.6))
        p.drawLine(QPointF(ox, oy), QPointF(ox + length * dx, oy + length * dy))
        norm = math.hypot(dx, dy)
        ux, uy = (dx / norm, dy / norm) if norm > 1.0e-6 else (0.0, -1.0)
        tx, ty = ox + length * dx + 11.0 * ux, oy + length * dy + 11.0 * uy
        p.drawText(QRectF(tx - 14.0, ty - 9.0, 28.0, 18.0), centre, name)


class Overlay(QWidget):
    """The labels over the 3D image. The mouse goes through it to the canvas."""

    def __init__(self, canvas: QWidget) -> None:
        super().__init__(canvas)
        self.canvas = canvas
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        canvas.installEventFilter(self)
        self.setGeometry(canvas.rect())

    def eventFilter(self, obj: Any, event: Any) -> bool:
        if obj is self.canvas and event.type() == QEvent.Type.Resize:
            self.setGeometry(self.canvas.rect())
        return False

    def paintEvent(self, event: Any) -> None:
        p = QPainter(self)
        draw_labels(p, self.canvas, self.width(), self.height())


class View3DWindow(QMainWindow):
    """The 3D view of one volume."""

    def __init__(self, vol: VolumeData, title: str = '', levels: list[float] | None = None,
                 box: tuple[Any, Any, Any] | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        ui.ensure_theme()
        self.vol = vol
        # the textures on the graphics card are released when the window is closed
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle(f'rspace3d 3D view: {title}' if title else 'rspace3d 3D view')
        self.setMinimumSize(900, 600)
        ui.start_size(self, 1180, 850)

        self.sample = v3.sample_values(vol.intensity)
        self.grid = v3.make_grid(vol)
        reduced = self.grid.factors != (1, 1, 1)
        self.data = v3.reduce_volume(vol.intensity, self.grid.factors) if reduced else vol.intensity
        self.scene = v3.default_scene(self.sample)
        self.camera = v3.Camera()
        # a level is on a logarithmic scale: only numbers above zero can be levels
        levels = [min(max(float(v), LEVEL_MIN), LEVEL_MAX) for v in (levels or [])
                  if math.isfinite(float(v)) and float(v) > 0.0]
        if levels:
            self.scene.levels = [v3.Level(v, v3.LEVEL_COLORS[i], 1.0 if i == 0 else 0.35)
                                 for i, v in enumerate(sorted(levels, reverse=True)[:v3.MAX_LEVELS])]
        self._level_rows: list[dict[str, Any]] = []
        self.canvas: Any = None
        self.overlay: Overlay | None = None

        central = QWidget()
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_panel(title, reduced))
        outer.addWidget(self._build_centre(), 1)
        QShortcut(QKeySequence(Qt.Key.Key_F1), self, lambda: ui.open_help(self, 'view3d'))

        edges, counts = v3.histogram(self.sample)
        self.level_bar.set_histogram(edges, counts)
        self._set_box(box)
        self._rebuild_levels()
        self._sync_mode()
        ui.guard_wheel(self)
        if self.canvas is not None:
            self.canvas.scene, self.canvas.camera = self.scene, self.camera
            self.canvas.set_data(self.data, self.grid)

    # ---- layout

    @staticmethod
    def _heading(lay: QVBoxLayout, text: str, help_topic: str | None = None) -> None:
        lay.addSpacing(ui.SPACE[1])
        if help_topic is None:
            lay.addWidget(ui.label(text, 'section-title'))
        else:
            lay.addLayout(ui.row(ui.label(text, 'section-title'), 1, ui.HelpButton(help_topic),
                                 stretch=False))

    def _build_panel(self, title: str, reduced: bool) -> QWidget:
        S = ui.SPACE
        panel = QFrame()
        panel.setObjectName('sidebar')
        panel.setFixedWidth(288)
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
        self.panel_scroll = scroll
        lay = QVBoxLayout(body)
        lay.setContentsMargins(S[3], S[3], S[3], S[3])
        lay.setSpacing(S[1])

        names = QVBoxLayout()
        names.setSpacing(0)
        names.addWidget(ui.label('rspace3d', 'app-title'))
        names.addWidget(ui.label('3D view', 'caption'))
        head = QHBoxLayout()
        head.setSpacing(S[1])
        head.addWidget(ui.Mark(), 0, Qt.AlignmentFlag.AlignTop)
        head.addLayout(names, 1)
        lay.addLayout(head)
        nh, nk, nl = self.vol.intensity.shape
        facts = f'{nh} x {nk} x {nl} voxels'
        if reduced:
            fh, fk, fl = self.grid.factors
            facts += f'. Shown with 1 of each {fh} x {fk} x {fl} voxels'
        if not self.grid.has_cell:
            facts += '. No unit cell: the axes are at right angles'
        self.facts = ui.label(facts + '.', 'caption', wrap=True)
        lay.addWidget(self.facts)

        # 1. What to show
        self._heading(lay, 'Show', 'view3d')
        self.surface_radio = QRadioButton('Surfaces of equal intensity')
        self.surface_radio.setToolTip('Draw a surface where the intensity has the value of a level.')
        self.projection_radio = QRadioButton('Projection of the maximum')
        self.projection_radio.setToolTip('Show for each point of the image the largest intensity '
                                         'along the line of sight.')
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.surface_radio)
        self.mode_group.addButton(self.projection_radio)
        self.surface_radio.setChecked(True)
        self.surface_radio.toggled.connect(self._sync_mode)
        lay.addWidget(self.surface_radio)
        lay.addWidget(self.projection_radio)

        # 2. Levels (surfaces) or colour scale (projection)
        lay.addSpacing(S[1])
        self.level_title = ui.label('Levels', 'section-title')
        lay.addWidget(self.level_title)
        self.level_bar = LevelBar()
        self.level_bar.moved.connect(self._bar_moved)
        lay.addWidget(self.level_bar)
        self.bar_hint = ui.label('', 'hint')
        lay.addWidget(self.bar_hint)

        self.levels_box = QWidget()
        lb = QVBoxLayout(self.levels_box)
        lb.setContentsMargins(0, 0, 0, 0)
        lb.setSpacing(S[0])
        names = QHBoxLayout()                # the names of the two columns of numbers
        names.setContentsMargins(SWATCH_SIZE + S[1], 0, 0, 0)
        names.setSpacing(S[1])
        names.addWidget(ui.label('Intensity', 'caption'), 1)
        opacity_name = ui.label('Opacity %', 'caption')
        opacity_name.setFixedWidth(OPACITY_WIDTH + S[1] + REMOVE_WIDTH)
        names.addWidget(opacity_name)
        lb.addLayout(names)
        self.level_rows = QVBoxLayout()
        self.level_rows.setSpacing(S[0])
        lb.addLayout(self.level_rows)
        self.add_btn = ui.button('Add level', tip='Add a second or a third surface.',
                                 clicked=self._add_level)
        self.reset_btn = ui.button('Reset levels', tip='Remove the other surfaces and show one surface '
                                   'at the level that 1 % of the voxels with a positive intensity '
                                   'are above.', clicked=self._reset_levels)
        lb.addSpacing(S[0])
        lb.addLayout(ui.row(self.add_btn, self.reset_btn))
        lay.addWidget(self.levels_box)

        self.scale_box = QWidget()
        sb = QGridLayout(self.scale_box)
        sb.setContentsMargins(0, 0, 0, 0)
        sb.setHorizontalSpacing(S[1])
        sb.setVerticalSpacing(S[0])
        self.min_spin, self.max_spin = NumberBox(), NumberBox()
        self.min_spin.setToolTip('Intensity that has the first colour of the map.')
        self.max_spin.setToolTip('Intensity that has the last colour of the map.')
        self.min_spin.valueChanged.connect(self._clim_edited)
        self.max_spin.valueChanged.connect(self._clim_edited)
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(COLORMAPS)
        self.cmap_combo.setToolTip('The colours that show the intensity.')
        self.cmap_combo.currentTextChanged.connect(self._cmap_changed)
        sb.addWidget(ui.label('Min'), 0, 0)
        sb.addWidget(ui.label('Max'), 0, 1)
        sb.addWidget(self.min_spin, 1, 0)
        sb.addWidget(self.max_spin, 1, 1)
        sb.addWidget(ui.label('Colour map'), 2, 0)
        sb.addWidget(self.cmap_combo, 2, 1)
        lay.addWidget(self.scale_box)

        # 3. Box
        lay.addSpacing(S[1])
        self.full_btn = ui.button('Show all', 'link', 'Show the complete volume again.',
                                  lambda: self._set_box(None))
        lay.addLayout(ui.row(ui.label('Box', 'section-title'), 1, self.full_btn, stretch=False))
        bg = QGridLayout()
        bg.setHorizontalSpacing(S[1])
        bg.setVerticalSpacing(S[0])
        self.box_spins: list[tuple[QDoubleSpinBox, QDoubleSpinBox]] = []
        for i, name in enumerate(v3.AXIS_NAMES):
            ax = self.grid.axes[i]
            lo, hi = float(min(ax[0], ax[-1])), float(max(ax[0], ax[-1]))
            pair = []
            for value in (lo, hi):
                spin = QDoubleSpinBox()
                spin.setDecimals(2)
                spin.setRange(lo, hi)
                spin.setSingleStep(0.25)
                spin.setValue(value)
                spin.setMinimumWidth(92)             # "-15.86" must fit
                spin.setToolTip(f'Limit of the shown part along {name}.')
                spin.valueChanged.connect(self._box_edited)
                pair.append(spin)
            bg.addWidget(ui.label(name), i, 0)
            bg.addWidget(pair[0], i, 1)
            bg.addWidget(ui.label('to'), i, 2)
            bg.addWidget(pair[1], i, 3)
            self.box_spins.append((pair[0], pair[1]))
        bg.setColumnStretch(1, 1)                    # the two fields fill the width of the panel
        bg.setColumnStretch(3, 1)
        lay.addLayout(bg)

        # 4. View
        self._heading(lay, 'View')
        views = QHBoxLayout()
        views.setSpacing(S[0])
        self.view_buttons: dict[str, QPushButton] = {}
        for key, text, tip in (('home', '3D', 'Look at the box from the side and from above.'),
                               ('hk', 'hk', 'Look at the hk plane: h to the right, k up.'),
                               ('hl', 'hl', 'Look at the hl plane: h to the right, l up.'),
                               ('kl', 'kl', 'Look at the kl plane: k to the right, l up.')):
            btn = ui.button(text, tip=tip, clicked=lambda key=key: self._look(key))
            btn.setMinimumWidth(0)
            views.addWidget(btn, 1)
            self.view_buttons[key] = btn
        lay.addLayout(views)
        self.persp_check = QCheckBox('Perspective')
        self.persp_check.setToolTip('Draw near things larger. Without it, equal lengths are '
                                    'equal on the screen.')
        self.persp_check.toggled.connect(self._options_changed)
        self.grid_check = QCheckBox('Miller-index grid')
        self.grid_check.setChecked(True)
        self.grid_check.setToolTip('Draw a line at each whole Miller index on the far faces of the box.')
        self.grid_check.toggled.connect(self._options_changed)
        self.dark_check = QCheckBox('Dark background')
        self.dark_check.toggled.connect(self._options_changed)
        for w in (self.persp_check, self.grid_check, self.dark_check):
            lay.addWidget(w)

        lay.addStretch(1)
        lay.addWidget(ui.button('Help (F1)', 'link', 'Open the help for the 3D view.',
                                lambda: ui.open_help(self, 'view3d')))
        self.panel_body = body
        return panel

    def _build_centre(self) -> QWidget:
        from .view3d_gl import gl_problem
        S = ui.SPACE
        page = QWidget()
        pl = QVBoxLayout(page)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(0)
        self.stack = QStackedWidget()
        pl.addWidget(self.stack, 1)

        problem_page = QWidget()
        pp = QVBoxLayout(problem_page)
        pp.setContentsMargins(S[4], S[4], S[4], S[4])
        self.problem = ui.Banner('error')
        pp.addWidget(self.problem)
        pp.addStretch(1)
        self.stack.addWidget(problem_page)

        problem = gl_problem()
        if problem is None:
            from .view3d_gl import View3DCanvas
            self.canvas = View3DCanvas()
            self.overlay = Overlay(self.canvas)
            self.canvas.overlay = self.overlay
            self.canvas.failed.connect(self._failed)
            self.canvas.hover.connect(self._hover)
            self.stack.addWidget(self.canvas)
            self.stack.setCurrentWidget(self.canvas)
        else:
            self._failed(problem)

        bar = QWidget()
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(S[2], S[1], S[3], S[1])
        bl.setSpacing(S[2])
        self.readout = ui.label(MOUSE_HINT, 'caption')
        bl.addWidget(self.readout, 1)
        self.save_btn = ui.button('Save image...', tip='Save the 3D image with its labels as a PNG file.',
                                  clicked=self._save_image)
        self.copy_btn = ui.button('Copy image', tip='Copy the 3D image with its labels to the clipboard.',
                                  clicked=self._copy_image)
        for btn in (self.save_btn, self.copy_btn):
            btn.setEnabled(self.canvas is not None)
            bl.addWidget(btn)
        pl.addWidget(bar)
        return page

    # ---- state to widgets

    def _refresh(self) -> None:
        if self.canvas is not None:
            self.canvas.refresh()

    def _failed(self, text: str) -> None:
        self.problem.show_message(
            f'The 3D view cannot draw on this computer. {text} The older 3D view is still there '
            "in Python: rspace3d.volume_isosurface.plot_isosurface(volume, method='pyvista').", 'error')
        self.stack.setCurrentIndex(0)
        for name in ('save_btn', 'copy_btn'):            # the buttons are made after the canvas
            if hasattr(self, name):
                getattr(self, name).setEnabled(False)

    def _marks(self) -> list[tuple[float, QColor]]:
        if self.scene.mode == 'projection':
            return [(self.scene.clim[0], ui.color('muted')), (self.scene.clim[1], ui.color('text'))]
        return [(lv.value, _qcolor(lv.color)) for lv in self.scene.levels]

    def _sync_mode(self, *_: Any) -> None:
        surface = self.surface_radio.isChecked()
        self.scene.mode = 'surface' if surface else 'projection'
        self.level_title.setText('Levels' if surface else 'Colour scale')
        self.bar_hint.setText('Drag a line to change its level.' if surface
                              else 'Drag a line to change an end of the scale.')
        self.levels_box.setVisible(surface)
        self.scale_box.setVisible(not surface)
        _set_number(self.min_spin, self.scene.clim[0])
        _set_number(self.max_spin, self.scene.clim[1])
        self.cmap_combo.blockSignals(True)
        self.cmap_combo.setCurrentText(self.scene.colormap)
        self.cmap_combo.blockSignals(False)
        self.level_bar.set_marks(self._marks())
        if self.canvas is not None:
            self.canvas.retune()
        self._refresh()

    def _rebuild_levels(self) -> None:
        """Make one row of controls for each level."""
        while self.level_rows.count():
            item = self.level_rows.takeAt(0)
            old = item.widget()
            if old is not None:
                old.setParent(None)                  # off the panel now, deleted by the event loop
                old.deleteLater()
        self._level_rows = []
        for i, lv in enumerate(self.scene.levels):
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(ui.SPACE[1])
            swatch = Swatch(lv.color)
            swatch.clicked.connect(lambda _=False, i=i: self._pick_colour(i))
            value = NumberBox()
            value.setToolTip('Intensity of this surface.')
            _set_number(value, lv.value)
            value.valueChanged.connect(lambda v, i=i: self._level_edited(i, v))
            opacity = QSpinBox()
            opacity.setRange(10, 100)
            opacity.setSingleStep(10)
            opacity.setValue(int(round(lv.opacity * 100)))
            opacity.setFixedWidth(OPACITY_WIDTH)
            opacity.setToolTip('Opacity of this surface in percent. Below 100 you see the '
                               'surfaces behind it.')
            opacity.valueChanged.connect(lambda v, i=i: self._opacity_edited(i, v))
            remove = QToolButton()
            remove.setText('×')
            remove.setProperty('role', 'mini')
            remove.setToolTip('Remove this surface.')
            remove.setEnabled(len(self.scene.levels) > 1)
            remove.clicked.connect(lambda _=False, i=i: self._remove_level(i))
            rl.addWidget(swatch)
            rl.addWidget(value, 1)
            rl.addWidget(opacity)
            rl.addWidget(remove)
            self.level_rows.addWidget(row)
            self._level_rows.append({'swatch': swatch, 'value': value, 'opacity': opacity, 'remove': remove})
        self.add_btn.setEnabled(len(self.scene.levels) < v3.MAX_LEVELS)
        ui.guard_wheel(self.levels_box)
        self.level_bar.set_marks(self._marks())
        self._refresh()

    # ---- levels

    def _bar_moved(self, index: int, value: float) -> None:
        if self.scene.mode == 'projection':
            lo, hi = self.scene.clim
            lo, hi = (min(value, hi * 0.999), hi) if index == 0 else (lo, max(value, lo * 1.001))
            self.scene.clim = (lo, hi)
            _set_number(self.min_spin, lo)
            _set_number(self.max_spin, hi)
        elif index < len(self.scene.levels):
            self.scene.levels[index].value = value
            _set_number(self._level_rows[index]['value'], value)
        self.level_bar.set_marks(self._marks())
        if self.canvas is not None:
            self.canvas.touch()                  # a drag: slow graphics draw with fewer pixels
        self._refresh()

    def _level_edited(self, index: int, value: float) -> None:
        self.scene.levels[index].value = float(value)
        _set_number(self._level_rows[index]['value'], float(value))
        self.level_bar.set_marks(self._marks())
        self._refresh()

    def _opacity_edited(self, index: int, percent: int) -> None:
        self.scene.levels[index].opacity = percent / 100.0
        self._refresh()

    def _pick_colour(self, index: int) -> None:
        lv = self.scene.levels[index]
        colour = QColorDialog.getColor(_qcolor(lv.color), self, 'Colour of the surface')
        if colour.isValid():
            lv.color = (colour.redF(), colour.greenF(), colour.blueF())
            self._rebuild_levels()

    def _add_level(self) -> None:
        if len(self.scene.levels) >= v3.MAX_LEVELS:
            return
        used = {lv.color for lv in self.scene.levels}
        colour = next((c for c in v3.LEVEL_COLORS if c not in used), v3.LEVEL_COLORS[-1])
        values = [lv.value for lv in self.scene.levels]
        value = v3.new_level(self.sample, values)
        # the surface with the highest intensity is inside the others: only that one is opaque
        self.scene.levels.append(v3.Level(value, colour, 1.0 if value > max(values) else 0.35))
        self._rebuild_levels()

    def _remove_level(self, index: int) -> None:
        if len(self.scene.levels) > 1:
            del self.scene.levels[index]
            self._rebuild_levels()

    def _reset_levels(self) -> None:
        self.scene.levels = v3.default_scene(self.sample).levels
        self._rebuild_levels()

    def _clim_edited(self, *_: Any) -> None:
        lo, hi = self.min_spin.value(), self.max_spin.value()
        if hi <= lo:
            hi = lo * 1.001
        self.scene.clim = (lo, hi)
        _set_number(self.min_spin, lo)
        _set_number(self.max_spin, hi)
        self.level_bar.set_marks(self._marks())
        self._refresh()

    def _cmap_changed(self, name: str) -> None:
        self.scene.colormap = name
        self._refresh()

    # ---- box and view

    def _set_box(self, box: tuple[Any, Any, Any] | None) -> None:
        """Set the three ranges (h, k, l). None is the full volume."""
        for i, (lo_spin, hi_spin) in enumerate(self.box_spins):
            lo, hi = (lo_spin.minimum(), hi_spin.maximum()) if box is None or box[i] is None else box[i]
            for spin, value in ((lo_spin, min(lo, hi)), (hi_spin, max(lo, hi))):
                spin.blockSignals(True)
                spin.setValue(float(value))
                spin.blockSignals(False)
        self._box_edited()

    def _box_edited(self, *_: Any) -> None:
        ranges = []
        for lo_spin, hi_spin in self.box_spins:
            lo, hi = lo_spin.value(), hi_spin.value()
            ranges.append((min(lo, hi), max(lo, hi)))
        self.scene.lo, self.scene.hi = self.grid.box(*ranges)
        self._refresh()

    def box_ranges(self) -> list[tuple[float, float]]:
        return [(a.value(), b.value()) for a, b in self.box_spins]

    def _look(self, view: str) -> None:
        self.camera.look(self.grid, view)
        self._refresh()

    def _options_changed(self, *_: Any) -> None:
        self.camera.ortho = not self.persp_check.isChecked()
        self.scene.grid = self.grid_check.isChecked()
        self.scene.dark = self.dark_check.isChecked()
        self._refresh()

    # ---- read-out and image

    def _hover(self, point: dict[str, Any] | None) -> None:
        if point is None:
            self.readout.setText(MOUSE_HINT)
            return
        h, k, l = point['hkl']
        text = f'h = {h:.3f}    k = {k:.3f}    l = {l:.3f}    intensity = {point["value"]:.4g}'
        if point.get('cap'):
            text += '    (cut by the box)'
        self.readout.setText(text)

    def image(self, scale: float = 2.0) -> QImage | None:
        """The 3D image with its labels. `scale` is the number of image pixels for each screen pixel."""
        if self.canvas is None:
            return None
        w, h = self.canvas.width(), self.canvas.height()
        scale = min(scale, 4096.0 / max(w, h, 1))
        img = self.canvas.grab_image(max(int(w * scale), 1), max(int(h * scale), 1))
        if img is None:
            return None
        img = img.convertToFormat(QImage.Format.Format_RGB32)
        p = QPainter(img)
        p.scale(scale, scale)
        p.setFont(self.font())
        draw_labels(p, self.canvas, w, h)
        p.end()
        return img

    def _save_image(self) -> None:
        img = self.image()
        if img is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Save the 3D image', 'view3d.png', 'PNG image (*.png)')
        if not path:
            return
        if img.save(path):
            self.readout.setText(f'Saved: {path}')
        else:
            ui.show_error(self, 'The image was not saved', f'The file could not be written:\n{path}')

    def _copy_image(self) -> None:
        img = self.image()
        if img is not None:
            QApplication.clipboard().setImage(img)
            self.readout.setText('The image is in the clipboard.')

    def closeEvent(self, event: Any) -> None:
        if self in _WINDOWS:
            _WINDOWS.remove(self)
        ui.release_help(self)
        super().closeEvent(event)


def show_volume(vol: VolumeData | str, levels: list[float] | None = None,
                box: tuple[Any, Any, Any] | None = None, title: str = '') -> View3DWindow | None:
    """Open the 3D view of a volume or of a .h5 file.

    `levels` are the intensities of the surfaces (at most three). `box` is
    ((h_min, h_max), (k_min, k_max), (l_min, l_max)); an entry that is None is
    the full range.

    In a program with a Qt event loop the call returns the window at once. In
    a script without a Qt application the call makes one, returns when the
    window is closed and returns None (the window is deleted then).
    """
    global _APP
    if isinstance(vol, str):
        title = title or os.path.basename(vol)
        vol = load_volume_h5(vol)
    app = QApplication.instance()
    own = app is None
    if own:
        ui.prepare_opengl()
        app = _APP = QApplication(sys.argv[:1])      # kept: a second call uses it again
    win = View3DWindow(vol, title=title, levels=levels, box=box)
    _WINDOWS.append(win)
    win.show()
    if own or app is _APP:                           # no event loop runs: run one until the window is closed
        app.exec()
        return None
    return win
