"""
ui.py - The look of the rspace3d windows.

This module holds the colours, the sizes, the style sheet and the widgets that
the three windows share. A window does not set a colour or a font size of its
own: it gives a widget a role (`set_role`) and the style sheet does the rest.

Parts:
  ensure_theme   the Fusion style, the palette, the font and the style sheet
  Shell          main window with a side bar, pages, and a progress and log panel
  Section        a titled block of a page (optional step number and help button)
  Banner         a status line: 'info', 'ok', 'warn' or 'error'
  FactGrid       rows of "name: value" (for example the dataset summary)
  guard_wheel    spin boxes, combo boxes and sliders ignore the wheel without focus
  show_error     an error dialog with the full text in a details part
"""
from __future__ import annotations

import html
import os
import sys
import tempfile
from typing import Any, Callable

from PyQt6.QtCore import QEvent, QObject, QPointF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor, QKeySequence, QPainter, QPalette, QPen, QPixmap, QPolygonF, QShortcut,
    QSurfaceFormat,
)
from PyQt6.QtWidgets import (
    QAbstractItemView, QAbstractSpinBox, QApplication, QButtonGroup, QComboBox, QFormLayout,
    QFrame, QGridLayout, QHBoxLayout, QLabel, QLayout, QListWidget, QMainWindow, QMessageBox,
    QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSlider, QSplitter, QStackedWidget,
    QTableWidget, QTextEdit, QToolButton, QVBoxLayout, QWidget,
)

# ──────────────────────────────────────────────────────────────────
# Tokens
# ──────────────────────────────────────────────────────────────────
# Contrast on the surface colour: text 15.8, muted 6.6, border 3.5, accent 6.4.
COLORS = {
    'bg': '#F4F6F8',            # window and page background
    'surface': '#FFFFFF',       # sections, inputs, tables
    'surface2': '#EBEEF2',      # side bar, table header
    'hover': '#EEF1F5',
    'pressed': '#E2E6EC',
    'line': '#D9DEE5',          # dividers and section borders
    'line2': '#B6BECA',         # button borders
    'border': '#7E899A',        # input borders
    'text': '#1B2330',
    'muted': '#525E6F',         # hints and secondary text
    'disabled': '#8A93A1',
    'disabled_bg': '#F1F3F6',
    'accent': '#1F5FA8',
    'accent_hover': '#1A528F',
    'accent_down': '#164678',
    'tint': '#E3EEF9',          # selection, information
    'ok': '#17663A',
    'ok_bg': '#E2F3E9',
    'warn': '#7A4A00',
    'warn_bg': '#FFF1D6',
    'error': '#A3241A',
    'error_bg': '#FDE7E4',
}
SPACE = (4, 8, 12, 16, 24, 32)          # the only distances that a layout uses
MONO = '"Consolas", "Cascadia Mono", "DejaVu Sans Mono", monospace'
SIDEBAR_WIDTH = 204
FORM_WIDTH = 560                        # largest useful width of a form column
LOG_HEIGHT = 230                        # progress and log panel at the default window size
_ICON_SET = 'v1'                        # change it when `_draw_icons` changes


def color(name: str) -> QColor:
    return QColor(COLORS[name])


# ──────────────────────────────────────────────────────────────────
# Small images of the style sheet (a style sheet can only use image files)
# ──────────────────────────────────────────────────────────────────

def _chevron(p: QPainter, size: float, up: bool) -> None:
    a, b = 0.25 * size, 0.75 * size
    y0, y1 = (0.64 * size, 0.36 * size) if up else (0.36 * size, 0.64 * size)
    p.drawPolyline(QPolygonF([QPointF(a, y0), QPointF(size / 2, y1), QPointF(b, y0)]))


def _check(p: QPainter, size: float) -> None:
    p.drawPolyline(QPolygonF([QPointF(0.22 * size, 0.52 * size), QPointF(0.43 * size, 0.72 * size),
                              QPointF(0.78 * size, 0.30 * size)]))


def _draw_icons(folder: str) -> None:
    """Write the arrows and the check mark as PNG files, at 1x and 2x."""
    jobs = [('down', 10, 'muted', lambda p, s: _chevron(p, s, False)),
            ('down_off', 10, 'disabled', lambda p, s: _chevron(p, s, False)),
            ('up', 10, 'muted', lambda p, s: _chevron(p, s, True)),
            ('up_off', 10, 'disabled', lambda p, s: _chevron(p, s, True)),
            ('check', 12, 'surface', _check),
            ('check_off', 12, 'disabled_bg', _check)]
    for name, size, col, draw in jobs:
        for scale, tag in ((1, ''), (2, '@2x')):
            pix = QPixmap(size * scale, size * scale)
            pix.fill(Qt.GlobalColor.transparent)
            p = QPainter(pix)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            pen = QPen(color(col), 1.6 * scale)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            p.setPen(pen)
            draw(p, float(size * scale))
            p.end()
            if not pix.save(os.path.join(folder, f'{name}{tag}.png')):
                raise OSError(f'cannot write the image {name}{tag}.png')


def _icon_dir() -> str | None:
    """Folder with the images, or None if the folder cannot be written."""
    folder = os.path.join(tempfile.gettempdir(), f'rspace3d-ui-{_ICON_SET}')
    try:
        if not os.path.isfile(os.path.join(folder, 'check_off@2x.png')):
            os.makedirs(folder, exist_ok=True)
            _draw_icons(folder)
    except OSError:
        return None
    return folder.replace('\\', '/')


# ──────────────────────────────────────────────────────────────────
# Style sheet
# ──────────────────────────────────────────────────────────────────

_QSS = """
* { outline: 0; }
QMainWindow, QDialog { background: @bg; }
QToolTip { background: @text; color: @surface; border: 0; padding: 6px 8px; }

/* text roles */
QLabel[role="app-title"] { font-size: 13pt; font-weight: 600; }
QLabel[role="page-title"] { font-size: 14pt; font-weight: 600; }
QLabel[role="section-title"] { font-size: 10.5pt; font-weight: 600; }
QLabel[role="strong"] { font-weight: 600; }
QLabel[role="hint"], QLabel[role="caption"] { color: @muted; font-size: 9pt; }
QLabel[role="mono"] { font-family: @mono; font-size: 9pt; color: @muted; }
QLabel[role="ok"] { color: @ok; font-weight: 600; }
QLabel[role="warn"] { color: @warn; font-weight: 600; }
QLabel[role="error"] { color: @error; font-weight: 600; }
QLabel[role="step"] {
    background: @tint; color: @accent; border-radius: 10px; font-size: 9pt; font-weight: 600;
    min-width: 20px; max-width: 20px; min-height: 20px; max-height: 20px;
}
QLabel:disabled { color: @disabled; }

/* blocks */
QFrame[role="section"] { background: @surface; border: 1px solid @line; border-radius: 8px; }
QFrame[role="rule"] { background: @line; border: 0; max-height: 1px; min-height: 1px; }
QFrame#sidebar { background: @surface2; border: 0; border-right: 1px solid @line; }
QFrame#activity { background: @surface; border: 0; border-top: 1px solid @line; }
QFrame#footer { background: @surface; border: 0; border-top: 1px solid @line; }
QFrame#footer QPushButton { min-height: 28px; }
QFrame[role="banner"] { border: 0; border-radius: 6px; }
QFrame[role="banner"][kind="info"] { background: @tint; }
QFrame[role="banner"][kind="ok"] { background: @ok_bg; }
QFrame[role="banner"][kind="warn"] { background: @warn_bg; }
QFrame[role="banner"][kind="error"] { background: @error_bg; }
QFrame[role="banner"][kind="ok"] QLabel { color: @ok; }
QFrame[role="banner"][kind="warn"] QLabel { color: @warn; }
QFrame[role="banner"][kind="error"] QLabel { color: @error; }

/* buttons */
QPushButton {
    background: @surface; color: @text; border: 1px solid @line2; border-radius: 6px;
    padding: 5px 14px; min-height: 18px;
}
QPushButton:hover { background: @hover; border-color: @border; }
QPushButton:pressed { background: @pressed; }
QPushButton:focus { border: 2px solid @accent; padding: 4px 13px; }
QPushButton:checked { background: @tint; border-color: @accent; color: @accent; }
QPushButton:disabled { background: @disabled_bg; color: @disabled; border-color: @line; }
QPushButton[role="primary"] {
    background: @accent; color: @surface; border: 1px solid @accent; font-weight: 600;
}
QPushButton[role="primary"]:hover { background: @accent_hover; border-color: @accent_hover; }
QPushButton[role="primary"]:pressed { background: @accent_down; border-color: @accent_down; }
QPushButton[role="primary"]:focus { border: 2px solid @text; }
QPushButton[role="primary"]:disabled { background: @line; color: @muted; border-color: @line; }
QPushButton[role="link"] {
    background: transparent; color: @accent; border: 0; padding: 2px 0; text-align: left;
}
QPushButton[role="link"]:hover, QPushButton[role="link"]:focus {
    color: @accent_down; text-decoration: underline; border: 0; padding: 2px 0;
}
QPushButton[role="link"]:disabled { background: transparent; color: @disabled; }
QPushButton[role="nav"] {
    background: transparent; border: 0; border-radius: 6px; padding: 8px 12px; text-align: left;
}
QPushButton[role="nav"]:hover { background: @pressed; }
QPushButton[role="nav"]:checked { background: @surface; color: @accent; font-weight: 600; }
QPushButton[role="nav"]:focus { border: 2px solid @accent; padding: 6px 10px; }
QPushButton[role="nav"]:disabled { background: transparent; }
QToolButton[role="help"] {
    background: @surface; color: @muted; border: 1px solid @line2; border-radius: 10px;
    font-weight: 600; font-size: 9pt;
    min-width: 18px; max-width: 18px; min-height: 18px; max-height: 18px;
}
QToolButton[role="help"]:hover, QToolButton[role="help"]:focus {
    color: @accent; border-color: @accent;
}
QToolButton[role="mini"] {
    background: @surface; color: @muted; border: 1px solid @line2; border-radius: 12px;
    font-weight: 600;
    min-width: 22px; max-width: 22px; min-height: 22px; max-height: 22px;
}
QToolButton[role="mini"]:hover, QToolButton[role="mini"]:focus {
    color: @accent; border-color: @accent;
}
QToolButton[role="mini"]:disabled { color: @disabled; border-color: @line; background: transparent; }
QToolBar { background: transparent; border: 0; spacing: 2px; }
QToolBar QToolButton { border: 0; border-radius: 4px; padding: 4px; }
QToolBar QToolButton:hover { background: @pressed; }
QToolBar QToolButton:checked { background: @tint; }

/* inputs */
QLineEdit, QComboBox, QAbstractSpinBox {
    background: @surface; color: @text; border: 1px solid @border; border-radius: 6px;
    padding: 4px 8px; min-height: 18px;
    selection-background-color: @accent; selection-color: @surface;
}
QLineEdit:hover, QComboBox:hover, QAbstractSpinBox:hover { border-color: @muted; }
QLineEdit:focus, QComboBox:focus, QAbstractSpinBox:focus {
    border: 2px solid @accent; padding: 3px 7px;
}
QLineEdit:disabled, QComboBox:disabled, QAbstractSpinBox:disabled {
    background: @disabled_bg; color: @disabled; border-color: @line2;
}
QLineEdit:read-only { background: @bg; }
QComboBox { padding-right: 26px; }
QComboBox:focus { padding-right: 25px; }
QComboBox::drop-down {
    subcontrol-origin: padding; subcontrol-position: center right; width: 24px; border: 0;
}
QComboBox QAbstractItemView {
    background: @surface; color: @text; border: 1px solid @border; padding: 2px;
    selection-background-color: @tint; selection-color: @text;
}
QAbstractSpinBox { padding-right: 24px; }
QAbstractSpinBox:focus { padding-right: 23px; }
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {
    subcontrol-origin: border; width: 20px; border: 0; border-left: 1px solid @line;
    background: transparent;
}
QAbstractSpinBox::up-button { subcontrol-position: top right; margin: 2px 2px 0 0; }
QAbstractSpinBox::down-button { subcontrol-position: bottom right; margin: 0 2px 2px 0; }
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover { background: @pressed; }
QCheckBox { spacing: 8px; }
QCheckBox:disabled { color: @disabled; }
QCheckBox::indicator, QAbstractItemView::indicator {
    width: 16px; height: 16px; background: @surface; border: 1px solid @border;
    border-radius: 4px;
}
QCheckBox::indicator:hover, QAbstractItemView::indicator:hover { border-color: @accent; }
QCheckBox::indicator:focus { border: 2px solid @accent; width: 14px; height: 14px; }
QCheckBox::indicator:checked, QAbstractItemView::indicator:checked {
    background: @accent; border-color: @accent;
}
QCheckBox::indicator:disabled, QAbstractItemView::indicator:disabled {
    background: @disabled_bg; border-color: @line2;
}
QCheckBox::indicator:checked:disabled, QAbstractItemView::indicator:checked:disabled {
    background: @disabled; border-color: @disabled;
}
QSlider::groove:horizontal { height: 4px; background: @line2; border-radius: 2px; }
QSlider::sub-page:horizontal { background: @accent; border-radius: 2px; }
QSlider::handle:horizontal {
    width: 14px; height: 14px; margin: -6px 0; border-radius: 8px;
    background: @surface; border: 2px solid @accent;
}
QSlider::handle:horizontal:hover, QSlider::handle:horizontal:focus { background: @tint; }
QSlider::handle:horizontal:disabled { border-color: @line2; }
QSlider::sub-page:horizontal:disabled { background: @line2; }

/* tables, lists, text */
QTableView, QListView {
    background: @surface; color: @text; border: 1px solid @line; border-radius: 6px;
    gridline-color: @line; selection-background-color: @tint; selection-color: @text;
}
QTableView:focus, QListView:focus { border-color: @accent; }
QFrame[role="section"] QTableView, QFrame[role="section"] QListView { border: 0; }
QTableView::item { padding: 1px 6px; }
QTableView::item:selected, QListView::item:selected { background: @tint; color: @text; }
QListView::item { padding: 3px 4px; border-radius: 4px; }
QListView::item:hover { background: @hover; }
QHeaderView { background: @surface2; border: 0; }
QHeaderView::section {
    background: @surface2; color: @text; font-weight: 600; padding: 4px 6px; border: 0;
    border-bottom: 1px solid @line; border-right: 1px solid @line;
}
QTableCornerButton::section { background: @surface2; border: 0; }
QTextEdit, QPlainTextEdit {
    background: @surface; color: @text; border: 1px solid @line; border-radius: 6px;
    selection-background-color: @accent; selection-color: @surface;
}
QTextEdit[role="log"] {
    font-family: @mono; font-size: 9pt; border: 0; border-radius: 0; padding: 0 12px;
}
QTextBrowser[role="help"] { border: 0; border-radius: 0; background: @surface; }
QTabWidget::pane { border: 0; border-top: 1px solid @line; top: -1px; }
QTabBar::tab {
    background: transparent; color: @muted; border: 0; border-bottom: 2px solid transparent;
    padding: 6px 4px; margin-right: 20px;
}
QTabBar::tab:hover { color: @text; }
QTabBar::tab:selected { color: @accent; border-bottom: 2px solid @accent; }
QTabBar::tab:focus { color: @accent_down; }
QProgressBar {
    background: @line; border: 0; border-radius: 3px; min-height: 6px; max-height: 6px;
}
QProgressBar::chunk { background: @accent; border-radius: 3px; }
QMenu { background: @surface; border: 1px solid @line2; padding: 4px; }
QMenu::item { padding: 5px 24px; border-radius: 4px; }
QMenu::item:selected { background: @tint; color: @text; }

/* scroll bars and splitters */
QScrollArea { background: transparent; border: 0; }
QScrollBar:vertical { background: transparent; width: 12px; margin: 2px; }
QScrollBar:horizontal { background: transparent; height: 12px; margin: 2px; }
QScrollBar::handle { background: @line2; border-radius: 4px; }
QScrollBar::handle:vertical { min-height: 32px; }
QScrollBar::handle:horizontal { min-width: 32px; }
QScrollBar::handle:hover { background: @border; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
QSplitter::handle { background: transparent; }
QSplitter::handle:hover { background: @line2; }
"""

_QSS_ICONS = """
QComboBox::down-arrow { image: url("@icons/down.png"); width: 10px; height: 10px; }
QComboBox::down-arrow:disabled { image: url("@icons/down_off.png"); }
QAbstractSpinBox::up-arrow { image: url("@icons/up.png"); width: 10px; height: 10px; }
QAbstractSpinBox::down-arrow { image: url("@icons/down.png"); width: 10px; height: 10px; }
QAbstractSpinBox::up-arrow:disabled, QAbstractSpinBox::up-arrow:off {
    image: url("@icons/up_off.png");
}
QAbstractSpinBox::down-arrow:disabled, QAbstractSpinBox::down-arrow:off {
    image: url("@icons/down_off.png");
}
QCheckBox::indicator:checked, QAbstractItemView::indicator:checked {
    image: url("@icons/check.png");
}
QCheckBox::indicator:checked:disabled, QAbstractItemView::indicator:checked:disabled {
    image: url("@icons/check_off.png");
}
"""


def style_sheet() -> str:
    """The style sheet of the application, with the token values."""
    text = _QSS
    icons = _icon_dir()
    if icons is not None:
        text += _QSS_ICONS.replace('@icons', icons)
    values = dict(COLORS, mono=MONO)
    for name in sorted(values, key=len, reverse=True):      # '@accent_hover' before '@accent'
        text = text.replace('@' + name, values[name])
    return text


def _palette() -> QPalette:
    pal = QPalette()
    roles = {QPalette.ColorRole.Window: 'bg', QPalette.ColorRole.WindowText: 'text',
             QPalette.ColorRole.Base: 'surface', QPalette.ColorRole.AlternateBase: 'bg',
             QPalette.ColorRole.Text: 'text', QPalette.ColorRole.Button: 'surface',
             QPalette.ColorRole.ButtonText: 'text', QPalette.ColorRole.Highlight: 'accent',
             QPalette.ColorRole.HighlightedText: 'surface', QPalette.ColorRole.Link: 'accent',
             QPalette.ColorRole.ToolTipBase: 'text', QPalette.ColorRole.ToolTipText: 'surface',
             QPalette.ColorRole.PlaceholderText: 'muted'}
    for role, name in roles.items():
        pal.setColor(role, color(name))
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text,
                 QPalette.ColorRole.ButtonText):
        pal.setColor(QPalette.ColorGroup.Disabled, role, color('disabled'))
    return pal


def prepare_opengl() -> None:
    """Ask for OpenGL 4.1 (core profile) for the 3D view. Call it before the QApplication is made.

    Qt on macOS gives a core profile only if it is the format of the whole
    application. On the other systems the 3D view also works without this call.
    """
    fmt = QSurfaceFormat()
    fmt.setVersion(4, 1)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    QSurfaceFormat.setDefaultFormat(fmt)


def ensure_theme() -> None:
    """Give the application the rspace3d look. Only the first call for an application works."""
    app = QApplication.instance()
    if app is None or app.property('rspace3d_theme'):
        return
    app.setProperty('rspace3d_theme', True)
    app.setStyle('Fusion')
    app.setPalette(_palette())
    font = app.font()
    if sys.platform == 'win32':
        font.setFamily('Segoe UI')
    if font.pointSizeF() < 10:
        font.setPointSizeF(10)
    app.setFont(font)
    app.setStyleSheet(style_sheet())


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def set_role(widget: QWidget, role: str | None, name: str = 'role') -> None:
    """Set the style role of a widget and draw it again with the new role."""
    if widget.property(name) == role:
        return
    widget.setProperty(name, role)
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


class WrapLabel(QLabel):
    """A label with word wrap whose smallest height is the height at its real width.

    Qt estimates the smallest height of a wrapped label for a narrow width. A
    page with many wrapped lines then gets a scroll bar that it does not need.
    """

    def __init__(self, text: str = '', parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setWordWrap(True)

    def minimumSizeHint(self) -> QSize:
        hint = super().minimumSizeHint()
        if self.width() < 60:                       # no real width yet
            return hint
        return QSize(min(hint.width(), 60), self.heightForWidth(self.width()))

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        if event.size().width() != event.oldSize().width():
            self.updateGeometry()


def label(text: str = '', role: str | None = None, wrap: bool = False,
          selectable: bool = False) -> QLabel:
    lab = WrapLabel(text) if wrap else QLabel(text)
    if role:
        lab.setProperty('role', role)
    if selectable:
        lab.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return lab


def hint(text: str = '') -> QLabel:
    """A short line of help below a control (secondary colour, wraps)."""
    return label(text, 'hint', wrap=True)


def rule() -> QFrame:
    """A thin horizontal line."""
    line = QFrame()
    line.setProperty('role', 'rule')
    return line


def button(text: str, role: str | None = None, tip: str = '',
           clicked: Callable[..., Any] | None = None) -> QPushButton:
    btn = QPushButton(text)
    if role:
        btn.setProperty('role', role)
    if role == 'link':
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
    if role == 'primary':
        btn.setMinimumHeight(40)
    if tip:
        btn.setToolTip(tip)
    if clicked is not None:
        btn.clicked.connect(lambda _=False: clicked())
    return btn


def row(*items: Any, spacing: int = SPACE[1], stretch: bool = True) -> QHBoxLayout:
    """A horizontal layout. An item is a widget, a layout, a text or a number (a stretch)."""
    lay = QHBoxLayout()
    lay.setSpacing(spacing)
    lay.setContentsMargins(0, 0, 0, 0)
    for item in items:
        if isinstance(item, QWidget):
            lay.addWidget(item)
        elif isinstance(item, QLayout):
            lay.addLayout(item)
        elif isinstance(item, str):
            lay.addWidget(QLabel(item))
        else:
            lay.addStretch(int(item))
    if stretch:
        lay.addStretch(1)
    return lay


def form() -> QFormLayout:
    """A form layout: labels on the left, fields at their natural width."""
    lay = QFormLayout()
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setHorizontalSpacing(SPACE[2])
    lay.setVerticalSpacing(SPACE[1])
    lay.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    lay.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
    return lay


def form_hint(lay: QFormLayout, text: str) -> QLabel:
    """A hint line below the last row of a form."""
    lab = hint(text)
    lay.addRow(lab)
    return lab


class _WheelGuard(QObject):
    """A wheel turn over a field without focus goes to the page, not to the field."""

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.Wheel and not obj.hasFocus():     # type: ignore[attr-defined]
            event.ignore()
            return True
        return False


_GUARD: _WheelGuard | None = None


def guard_wheel(root: QWidget) -> None:
    """Stop value changes by the mouse wheel in the fields below `root` without focus."""
    global _GUARD
    if _GUARD is None:
        _GUARD = _WheelGuard(QApplication.instance())
    for kind in (QAbstractSpinBox, QComboBox, QSlider):
        for w in root.findChildren(kind):
            if not w.property('wheel_guard'):
                w.setProperty('wheel_guard', True)
                w.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
                w.installEventFilter(_GUARD)


def fit_rows(view: QAbstractItemView, rows: int | None = None, max_rows: int = 12) -> None:
    """Make a table or a list as high as its rows, so that it does not scroll in its page."""
    if isinstance(view, QTableWidget):
        head = view.horizontalHeader().sizeHint().height()
        if rows is None and 0 < view.rowCount() <= max_rows:
            # the real rows: a row with a wrapped text is higher
            body = sum(view.rowHeight(i) for i in range(view.rowCount()))
            view.setFixedHeight(head + body + 2 * view.frameWidth() + 2)
            return
        n = view.rowCount() if rows is None else rows
        height = view.verticalHeader().defaultSectionSize()
    else:
        n = view.count() if rows is None and isinstance(view, QListWidget) else (rows or 0)
        head = 0
        height = view.sizeHintForRow(0) if n and view.sizeHintForRow(0) > 0 else 26
    view.setFixedHeight(head + height * max(1, min(n, max_rows)) + 2 * view.frameWidth() + 2)


def show_error(parent: QWidget | None, title: str, message: str) -> None:
    """An error dialog: the first line as the message, the full text as details."""
    lines = str(message).strip().split('\n')
    first = lines[0] or title
    if len(first) > 300:
        first = first[:300] + ' ...'
    box = QMessageBox(QMessageBox.Icon.Critical, title, first,
                      QMessageBox.StandardButton.Ok, parent)
    box.setTextFormat(Qt.TextFormat.PlainText)      # a path or "<...>" is text, not HTML
    if len(lines) > 1 or len(lines[0]) > 300:
        box.setInformativeText('The details and the log have the full error text.')
        box.setDetailedText(str(message))
    box.exec()


def release_help(closing: QWidget) -> None:
    """Close the help window if `closing` is the last main window of the application."""
    app = QApplication.instance()
    dlg = getattr(app, '_rspace3d_help', None)
    if dlg is None or not dlg.isVisible():
        return
    others = [w for w in app.topLevelWidgets()
              if isinstance(w, QMainWindow) and w is not closing and w.isVisible()]
    if not others:
        dlg.close()


def start_size(window: QWidget, width: int, height: int) -> None:
    """Give a window its start size, not larger than the free area of the screen."""
    screen = window.screen() or QApplication.primaryScreen()
    if screen is not None:
        free = screen.availableGeometry()
        width = min(width, free.width() - 40)
        height = min(height, free.height() - 60)
    window.resize(max(width, window.minimumWidth()), max(height, window.minimumHeight()))


def open_help(widget: QWidget, topic: str) -> None:
    """Show a help topic: in the window of `widget` if it has help pages, else in a dialog."""
    win = widget.window()
    if hasattr(win, 'open_help'):
        win.open_help(topic)
    else:
        from .help_view import show_help_dialog
        show_help_dialog(topic, win)


# ──────────────────────────────────────────────────────────────────
# Widgets
# ──────────────────────────────────────────────────────────────────

class HelpButton(QToolButton):
    """A small "?" button that opens one help topic."""

    def __init__(self, topic: str, tip: str = 'Open the help for this part',
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.topic = topic
        self.setText('?')
        self.setProperty('role', 'help')
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(tip)
        self.clicked.connect(lambda _=False: open_help(self, self.topic))


class Section(QFrame):
    """A titled block of a page. `body` is the layout for its content."""

    def __init__(self, title: str, step: int | None = None, help_topic: str | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty('role', 'section')
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACE[3], SPACE[2], SPACE[3], SPACE[2])
        outer.setSpacing(SPACE[2])
        self.header = QHBoxLayout()
        self.header.setSpacing(SPACE[1])
        if step is not None:
            badge = label(str(step), 'step')
            badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.header.addWidget(badge)
        self.title_label = label(title, 'section-title')
        self.header.addWidget(self.title_label)
        self.header.addStretch(1)
        if help_topic:
            self.header.addWidget(HelpButton(help_topic))
        outer.addLayout(self.header)
        self.body = QVBoxLayout()
        self.body.setSpacing(SPACE[1])
        outer.addLayout(self.body)

    def add(self, item: QWidget | QLayout, stretch: int = 0) -> Any:
        if isinstance(item, QWidget):
            self.body.addWidget(item, stretch)
        else:
            self.body.addLayout(item, stretch)
        return item

    def add_hint(self, text: str) -> QLabel:
        return self.add(hint(text))


class Banner(QFrame):
    """A status line with a tint. It is hidden while it has no text.

    The kind is 'info', 'ok', 'warn' or 'error'. The first word names the kind,
    so that the colour is not the only sign.
    """

    WORDS = {'info': 'Note', 'ok': 'Done', 'warn': 'Check', 'error': 'Problem'}

    def __init__(self, kind: str = 'info', parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty('role', 'banner')
        self.setProperty('kind', kind)
        self._kind = kind
        self._text = ''
        self.row = QHBoxLayout(self)
        self.row.setContentsMargins(SPACE[2], SPACE[1], SPACE[2], SPACE[1])
        self.row.setSpacing(SPACE[1])
        self.label = label('', wrap=True, selectable=True)
        self.row.addWidget(self.label, 1)
        self.setVisible(False)

    def show_message(self, text: str, kind: str | None = None) -> None:
        if kind is not None and kind != self._kind:
            self._kind = kind
            set_role(self, kind, 'kind')
            for child in self.findChildren(QLabel):          # the label colour follows the kind
                child.style().unpolish(child)
                child.style().polish(child)
        self._text = text
        self.label.setText(f'<b>{self.WORDS[self._kind]}:</b> {html.escape(text)}' if text else '')
        self.setVisible(bool(text))

    def setText(self, text: str) -> None:
        self.show_message(text)

    def text(self) -> str:
        return self._text

    def kind(self) -> str:
        return self._kind

    def clear(self) -> None:
        self.show_message('')

    def add_button(self, btn: QPushButton) -> QPushButton:
        self.row.addWidget(btn, 0, Qt.AlignmentFlag.AlignVCenter)
        return btn


class FactGrid(QWidget):
    """Rows of "name  value". `set_values` fills them, `clear` shows a dash in each row."""

    def __init__(self, names: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(SPACE[3])
        grid.setVerticalSpacing(SPACE[0])
        grid.setColumnStretch(1, 1)
        self._values: dict[str, QLabel] = {}
        for i, name in enumerate(names):
            grid.addWidget(label(name, 'caption'), i, 0, Qt.AlignmentFlag.AlignTop)
            value = label('-', wrap=True, selectable=True)
            grid.addWidget(value, i, 1)
            self._values[name] = value

    def set_values(self, values: dict[str, str]) -> None:
        for name, lab in self._values.items():
            lab.setText(values.get(name, '-'))

    def clear(self) -> None:
        self.set_values({})

    def value(self, name: str) -> str:
        return self._values[name].text()


class Mark(QWidget):
    """The rspace3d mark: a cube with one plane through it (drawn, no image file)."""

    def __init__(self, size: int = 30, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(size, size)

    def paintEvent(self, _event: Any) -> None:
        s = float(self.width())
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        top, mid, low = 0.08 * s, 0.30 * s, 0.70 * s
        left, cx, right = 0.10 * s, 0.50 * s, 0.90 * s
        # the plane, behind the front edges of the cube
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color('accent'))
        p.drawPolygon(QPolygonF([QPointF(0.30 * s, 0.19 * s), QPointF(0.70 * s, 0.41 * s),
                                 QPointF(0.70 * s, 0.81 * s), QPointF(0.30 * s, 0.59 * s)]))
        pen = QPen(color('text'), max(1.2, 0.045 * s))
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPolygon(QPolygonF([QPointF(cx, top), QPointF(right, mid), QPointF(right, low),
                                 QPointF(cx, 0.92 * s), QPointF(left, low), QPointF(left, mid)]))
        p.drawPolyline(QPolygonF([QPointF(left, mid), QPointF(cx, 0.52 * s), QPointF(right, mid)]))
        p.drawLine(QPointF(cx, 0.52 * s), QPointF(cx, 0.92 * s))
        p.end()


class Sidebar(QFrame):
    """The navigation of a window: the name of the window and one button for each page."""

    page_selected = pyqtSignal(int)

    def __init__(self, title: str, subtitle: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName('sidebar')
        self.setFixedWidth(SIDEBAR_WIDTH)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(SPACE[2], SPACE[3], SPACE[2], SPACE[2])
        lay.setSpacing(SPACE[0])
        names = QVBoxLayout()
        names.setSpacing(0)
        names.addWidget(label(title, 'app-title'))
        names.addWidget(label(subtitle, 'caption', wrap=True))
        head = QHBoxLayout()
        head.setSpacing(SPACE[1])
        head.addWidget(Mark(), 0, Qt.AlignmentFlag.AlignTop)
        head.addLayout(names, 1)
        lay.addLayout(head)
        lay.addSpacing(SPACE[3])
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._group.idClicked.connect(self.page_selected.emit)
        self._top = QVBoxLayout()
        self._top.setSpacing(2)
        lay.addLayout(self._top)
        lay.addStretch(1)
        self._bottom = QVBoxLayout()
        self._bottom.setSpacing(2)
        lay.addLayout(self._bottom)
        lay.addSpacing(SPACE[1])
        self.footer = label('', 'caption', wrap=True)
        lay.addWidget(self.footer)

    def add_item(self, text: str, index: int, bottom: bool = False, tip: str = '') -> QPushButton:
        btn = QPushButton(text)
        btn.setProperty('role', 'nav')
        btn.setCheckable(True)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        if tip:
            btn.setToolTip(tip)
        self._group.addButton(btn, index)
        (self._bottom if bottom else self._top).addWidget(btn)
        return btn

    def select(self, index: int) -> None:
        btn = self._group.button(index)
        if btn is not None:
            btn.setChecked(True)


class ActivityPanel(QFrame):
    """The progress bar, the status line and the log below the pages of a window."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName('activity')
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.header = QWidget()
        head = QHBoxLayout(self.header)
        head.setContentsMargins(SPACE[3], SPACE[1], SPACE[3], SPACE[1])
        head.setSpacing(SPACE[2])
        self.toggle_btn = button('Hide log', 'link', 'Show or hide the log. The log has the '
                                 'details of each step.')
        head.addWidget(self.toggle_btn)
        self.status = label('', wrap=False)
        self.status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        head.addWidget(self.status, 1)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedWidth(240)
        head.addWidget(self.progress)
        self.percent = label('', 'caption')
        self.percent.setFixedWidth(40)
        head.addWidget(self.percent)
        lay.addWidget(self.header)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setProperty('role', 'log')
        self.log.setMinimumHeight(60)
        lay.addWidget(self.log, 1)

    def set_status(self, text: str) -> None:
        self.status.setText(text)
        self.status.setToolTip(text)

    def set_progress(self, current: int, total: int) -> None:
        if total > 0:
            self.progress.setMaximum(total)
            self.progress.setValue(current)
            self.percent.setText(f'{100 * current // total} %')
        else:                                       # no total: a busy bar
            self.progress.setMaximum(0)
            self.percent.setText('')

    def append(self, msg: str) -> None:
        self.log.append(msg)
        bar = self.log.verticalScrollBar()
        bar.setValue(bar.maximum())


class Shell(QMainWindow):
    """A window with a side bar, pages, and the progress and log panel below the pages.

    `add_page` puts a widget on a page with a title line. The page content is in
    a scroll area, which shows a scroll bar only if the window is too small. A
    footer of a page (its main action) is below the scroll area, so it is always
    visible.
    """

    def __init__(self, window_title: str, subtitle: str) -> None:
        super().__init__()
        ensure_theme()
        self.setWindowTitle(window_title)
        self.setMinimumSize(1040, 640)
        self._names: list[str] = []
        self._topics: list[str | None] = []
        self._log_open: list[bool] = []         # for each page: is the log open
        self._containers: list[QWidget] = []
        self._help_view: Any = None
        self._help_index: int | None = None
        self._before_help = 0
        self._log_sizes: list[int] | None = None
        central = QWidget()
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.sidebar = Sidebar('rspace3d', subtitle)
        self.sidebar.page_selected.connect(self.show_page)
        outer.addWidget(self.sidebar)
        # Pages above, progress and log below. The user can move the border.
        self.split = QSplitter(Qt.Orientation.Vertical)
        self.split.setChildrenCollapsible(False)
        self.split.setHandleWidth(SPACE[0])
        outer.addWidget(self.split, 1)
        self.pages = QStackedWidget()
        self.pages.setMinimumHeight(240)
        self.split.addWidget(self.pages)
        self.activity = ActivityPanel()
        self.progress = self.activity.progress
        self.log = self.activity.log
        self.activity.toggle_btn.clicked.connect(self._toggle_log)
        self.split.addWidget(self.activity)
        self.split.setStretchFactor(0, 1)
        self.split.setStretchFactor(1, 0)
        self.split.setSizes([LOG_HEIGHT * 3, LOG_HEIGHT])
        QShortcut(QKeySequence(QKeySequence.StandardKey.HelpContents), self, self._help_key)

    # ── pages ──
    def add_page(self, name: str, title: str, purpose: str, content: QWidget,
                 help_topic: str | None = None, scroll: bool = True,
                 footer: QWidget | None = None, bottom: bool = False,
                 log: bool = True) -> int:
        """Add a page and its button in the side bar. Returns the index of the page.

        `log` tells if the log is open when the user comes to the page. A page
        that needs the height for its results starts with the log closed. The
        status line and the progress bar stay visible on each page.
        """
        index = len(self._names)
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        head = QHBoxLayout()
        head.setContentsMargins(SPACE[4], SPACE[3], SPACE[4], SPACE[1])
        head.setSpacing(SPACE[2])
        head.addWidget(label(title, 'page-title'), 0, Qt.AlignmentFlag.AlignBottom)
        about = label(purpose, 'hint')
        about.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        about.setToolTip(purpose)
        head.addWidget(about, 1, Qt.AlignmentFlag.AlignBottom)
        if help_topic:
            head.addWidget(HelpButton(help_topic, 'Open the help for this page (F1)'), 0,
                           Qt.AlignmentFlag.AlignBottom)
        lay.addLayout(head)
        content.setContentsMargins(SPACE[4], SPACE[1], SPACE[4], SPACE[2])
        if scroll:
            area = QScrollArea()
            area.setWidgetResizable(True)
            area.setFrameShape(QFrame.Shape.NoFrame)
            area.setWidget(content)
            lay.addWidget(area, 1)
        else:
            lay.addWidget(content, 1)
        if footer is not None:
            lay.addWidget(footer)
        self.pages.addWidget(box)
        self._names.append(name)
        self._topics.append(help_topic)
        self._log_open.append(log)
        self._containers.append(box)
        self.sidebar.add_item(name, index, bottom=bottom)
        if index == 0:
            self.sidebar.select(0)
        return index

    def add_help_page(self, start_topic: str = 'start') -> int:
        """Add the Help page (the last button of the side bar)."""
        from .help_view import HelpView
        self._help_view = HelpView()
        self._help_view.show_topic(start_topic)
        self._help_index = self.add_page('Help', 'Help', 'How to use this window and what the '
                                         'options do to the data.', self._help_view,
                                         help_topic=None, scroll=False, bottom=True, log=False)
        self._topics[self._help_index] = start_topic
        self._help_view.back_btn.clicked.connect(lambda _=False: self.show_page(self._before_help))
        return self._help_index

    def page_names(self) -> list[str]:
        return list(self._names)

    def page_widget(self, index: int) -> QWidget:
        return self._containers[index]

    def current_page(self) -> str:
        return self._names[self.pages.currentIndex()]

    def show_page(self, page: int | str) -> None:
        index = self._names.index(page) if isinstance(page, str) else int(page)
        if index == self._help_index and self._help_view is not None:
            if self.pages.currentIndex() != index:
                self._before_help = self.pages.currentIndex()
            self._help_view.set_back(f'Back to {self._names[self._before_help]}')
        self.pages.setCurrentIndex(index)
        self.sidebar.select(index)
        self.set_log_visible(self._log_open[index])

    def _help_key(self) -> None:
        """F1: the help for the page that the user sees. On the Help page it does nothing."""
        index = self.pages.currentIndex()
        if index != self._help_index:
            self.open_help(self._topics[index] or 'start')

    def closeEvent(self, event: Any) -> None:
        release_help(self)
        super().closeEvent(event)

    def open_help(self, topic: str) -> None:
        """Show the Help page at `topic`."""
        if self._help_index is None:
            from .help_view import show_help_dialog
            show_help_dialog(topic, self)
            return
        self._help_view.show_topic(topic)
        self.show_page(self._help_index)

    # ── progress, status and log ──
    def set_status(self, text: str) -> None:
        self.activity.set_status(text)

    def _toggle_log(self, _checked: bool = False) -> None:
        """The user opens or closes the log. The page keeps this choice."""
        visible = not self.log.isVisibleTo(self.activity)
        if self._log_open:
            self._log_open[self.pages.currentIndex()] = visible
        self.set_log_visible(visible)

    def set_log_visible(self, visible: bool) -> None:
        """Show the log, or hide it and give its height to the pages."""
        if visible == self.log.isVisibleTo(self.activity):
            return
        head = self.activity.header.sizeHint().height()
        if not visible:
            sizes = self.split.sizes()
            if sizes[1] > head + 40:                # a real height: the window is on screen
                self._log_sizes = sizes
            self.log.setVisible(False)
            self.activity.setMaximumHeight(head)
            if sum(sizes) > head:
                self.split.setSizes([sum(sizes) - head, head])
        else:
            self.activity.setMaximumHeight(16777215)
            self.log.setVisible(True)
            total = sum(self.split.sizes())
            low = (self._log_sizes or [0, LOG_HEIGHT])[1]
            self.split.setSizes([total - low, low] if total > low else [LOG_HEIGHT * 3, low])
        self.activity.toggle_btn.setText('Hide log' if visible else 'Show log')

    def _update_progress(self, current: int, total: int) -> None:
        self.activity.set_progress(current, total)

    def _log(self, msg: str) -> None:
        self.activity.append(msg)


def two_columns(left: list[QWidget], right: list[QWidget], left_width: int | None = None,
                ratio: tuple[int, int] = (1, 1)) -> QWidget:
    """A page content with two columns of widgets. The columns start at the top."""
    page = QWidget()
    lay = QHBoxLayout(page)
    lay.setContentsMargins(0, 0, 0, 0)          # the page has the margins (Shell.add_page)
    lay.setSpacing(SPACE[3])
    for widgets, stretch, width in ((left, ratio[0], left_width), (right, ratio[1], None)):
        col = QVBoxLayout()
        col.setSpacing(SPACE[3])
        for w in widgets:
            col.addWidget(w)
        col.addStretch(1)
        holder = QWidget()
        holder.setLayout(col)
        col.setContentsMargins(0, 0, 0, 0)
        if width is not None:
            holder.setFixedWidth(width)
            lay.addWidget(holder)
        else:
            lay.addWidget(holder, stretch)
    return page


def footer_bar() -> tuple[QFrame, QHBoxLayout]:
    """The bar below a page for its main action. Returns the bar and its layout."""
    bar = QFrame()
    bar.setObjectName('footer')
    lay = QHBoxLayout(bar)
    lay.setContentsMargins(SPACE[4], SPACE[2], SPACE[4], SPACE[2])
    lay.setSpacing(SPACE[2])
    return bar, lay


def big_hint(text: str) -> QLabel:
    """The text of an empty area: what will be there and how to get it."""
    lab = label(text, 'hint', wrap=True)
    lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lab.setMinimumHeight(80)
    return lab



