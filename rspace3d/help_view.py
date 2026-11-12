"""
help_view.py - The help pages of the rspace3d windows.

`HelpView` is a list of topics next to the text of the selected topic
(`help_content.TOPICS`). One topic is not a text: `AveragingExplorer` shows one
voxel with the members of its orbit and applies an averaging option to them.
It calls the estimator of the program (`orbit_engine.estimate`), so the example
follows the same rules as a real symmetrisation.

`show_help_dialog(topic)` shows the help in its own window. The viewer uses
it, and the "Open in a window" button of the processing windows.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QImage, QTextDocument
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QDialog, QDoubleSpinBox, QGridLayout, QHBoxLayout, QListWidget,
    QListWidgetItem, QStackedWidget, QTextBrowser, QVBoxLayout, QWidget,
)

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from . import averaging as av
from . import help_content as hc
from . import orbit_engine as oe
from . import ui

TEXT_WIDTH = 700            # widest text line of a help page (px)

_CSS = f"""
body {{ color: {ui.COLORS['text']}; }}
h1 {{ font-size: 14pt; font-weight: 600; margin-top: 0px; margin-bottom: 10px; }}
h2 {{ font-size: 11.5pt; font-weight: 600; margin-top: 20px; margin-bottom: 6px; }}
p, li {{ line-height: 140%; }}
li {{ margin-bottom: 4px; }}
a {{ color: {ui.COLORS['accent']}; text-decoration: none; }}
code {{ font-family: Consolas, 'Cascadia Mono', 'DejaVu Sans Mono', monospace; }}
table.grid {{ border-collapse: collapse; margin-top: 6px; margin-bottom: 6px; }}
table.grid th {{ background-color: {ui.COLORS['surface2']}; font-weight: 600;
                border: 1px solid {ui.COLORS['line']}; }}
table.grid td {{ border: 1px solid {ui.COLORS['line']}; }}
"""


# ──────────────────────────────────────────────────────────────────
# Figures of the help text (drawn, no image files)
# ──────────────────────────────────────────────────────────────────

def _draw_orbit(fig: Figure) -> None:
    """A voxel and its symmetry-equivalent positions in one plane (Laue group 4/mmm)."""
    ax = fig.add_axes([0.02, 0.04, 0.44, 0.92])
    c = ui.COLORS
    for x0, y0, x1, y1 in ((-1, 0, 1, 0), (0, -1, 0, 1), (-1, -1, 1, 1), (-1, 1, 1, -1)):
        ax.plot([x0, x1], [y0, y1], color=c['line2'], lw=0.8, ls='--')
    pts = [(0.75, 0.3), (0.3, 0.75), (-0.3, 0.75), (-0.75, 0.3),
           (-0.75, -0.3), (-0.3, -0.75), (0.3, -0.75), (0.75, -0.3)]
    for i, (x, y) in enumerate(pts):
        first = i == 0
        ax.plot(x, y, 'o', ms=11, mfc=c['accent'] if first else c['surface'],
                mec=c['accent'], mew=1.6)
        ax.annotate(str(i + 1), (x, y), ha='center', va='center', fontsize=7,
                    color=c['surface'] if first else c['accent'])
    ax.text(0, 0, 'hkl = 0', ha='center', va='center', fontsize=8, color=c['muted'],
            bbox={'fc': c['surface'], 'ec': 'none', 'pad': 1.5})
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.text(0.50, 0.80, 'One voxel (1) and its symmetry-equivalent\npositions (2 to 8) in '
             'one plane, for the\nLaue group 4/mmm. The dashed lines are\nthe mirror lines.',
             fontsize=9, color=c['text'], va='top', linespacing=1.5)
    fig.text(0.50, 0.36, 'The eight positions are the members of the\norbit. The average '
             'gives each of them the\nsame value: the mean of the members\nthat remain after '
             'the outlier rejection.', fontsize=9, color=c['text'], va='top',
             linespacing=1.5)


_FIGURES = {'orbit': ((6.6, 2.5), _draw_orbit)}


def render_figure(name: str, scale: float = 1.0) -> QImage:
    """A figure of the help text as an image. `scale` is the pixel ratio of the screen."""
    size, draw = _FIGURES[name]
    fig = Figure(figsize=size, dpi=100 * scale, facecolor=ui.COLORS['surface'])
    canvas = FigureCanvasAgg(fig)
    draw(fig)
    canvas.draw()
    w, h = canvas.get_width_height()
    img = QImage(canvas.buffer_rgba(), w, h, QImage.Format.Format_RGBA8888).copy()
    img.setDevicePixelRatio(scale)
    return img


class _Browser(QTextBrowser):
    """A text browser that draws the figures of the help text (`<img src="fig:NAME">`)."""

    def loadResource(self, kind: int, url: QUrl) -> Any:
        if url.scheme() == 'fig' and url.path() in _FIGURES:
            return render_figure(url.path(), self.devicePixelRatioF())
        return super().loadResource(kind, url)


# ──────────────────────────────────────────────────────────────────
# Interactive example: one voxel and the members of its orbit
# ──────────────────────────────────────────────────────────────────

# Noise model of the example: gain, rel, kappa, floor (see help topic 'averaging')
EXAMPLE_NOISE = (1.0, 0.02, 0.25, 0.0)
MIN_VALID = 3
# Each case: the true value, the member values, their pixel counts, the gradient term G
CASES: list[dict[str, Any]] = [
    {'name': 'Clean data',
     'true': 100.0,
     'values': [101.2, 98.6, 100.9, 97.9, 104.1, 95.2, 106.8, 91.5],
     'counts': [40, 35, 30, 25, 8, 6, 4, 2], 'grad': 0.0,
     'text': 'The members differ by counting noise only. The members from few pixels (on the '
             'right) scatter more. No member is wrong, so a good option rejects nothing. The '
             'pixel-count weights and the noise weights give the noisy members less weight.'},
    {'name': 'A spike in one member',
     'true': 100.0,
     'values': [100.8, 99.1, 101.5, 98.7, 100.2, 99.6, 101.1, 400.0],
     'counts': [32, 30, 28, 34, 31, 29, 33, 30], 'grad': 0.0,
     'text': 'One member has a zinger or a part of a bright arc. Without rejection, the spike '
             'goes into the mean, and the average puts it at all eight positions. The spread '
             'limit and the noise limit remove it.'},
    {'name': 'Extra intensity in one frame',
     'true': 100.0,
     'values': [100.8, 99.1, 101.5, 98.7, 100.2, 99.6, 101.1, 122.0],
     'counts': [32, 30, 28, 34, 31, 29, 33, 30], 'grad': 0.0,
     'text': 'One member has a small extra intensity, for example from the halo of a strong '
             'Bragg reflection in one frame. The spread limit at 3 sigma keeps it: its limit '
             'cannot go below the counting noise of one pixel. The noise limit knows that '
             'this member is the mean of 30 pixels, so its limit is narrower and the member '
             'goes out.'},
    {'name': 'A sharp Bragg peak',
     'true': 5100.0,
     'values': [5200.0, 4100.0, 6100.0, 4700.0, 5600.0, 3900.0, 6500.0, 5000.0],
     'counts': [32, 30, 28, 34, 31, 29, 33, 30], 'grad': 2400.0,
     'text': 'The voxel is on the side of a sharp Bragg peak. The members sample the peak at '
             'different positions in the voxel, so they scatter much more than the counting '
             'noise. The members are not wrong. The noise limit keeps them, because the '
             'noise model has a term for the change of the intensity from voxel to voxel. A '
             'low threshold removes good members here.'},
]


def explore(case: dict[str, Any], weighting: str, rejection: str,
            sigma: float) -> dict[str, Any]:
    """Apply an averaging option to the members of one case with the estimator of the program.

    Returns the value of the voxel, the mask of the members that remain, the
    median and the limit of each member (None without rejection).
    """
    vals = np.asarray(case['values'], np.float32)[:, None]
    ncnt = np.asarray(case['counts'], np.float64)[:, None]
    grad = np.full(1, case['grad'], np.float32)
    options = oe.Averaging(weighting, rejection, EXAMPLE_NOISE)
    est = oe.estimate(vals, ncnt, grad, None if rejection == 'none' else float(sigma),
                      MIN_VALID, True, options)
    med = float(np.median(vals[:, 0]))
    limits = spread = None
    if rejection == 'mad':
        mad = float(np.median(np.abs(vals[:, 0] - np.float32(med))))
        spread = max(oe.MAD_SCALE * mad, float(np.sqrt(max(med, 1.0))))
        limits = np.full(len(vals), sigma * spread)
    elif rejection == 'model':
        v = oe.model_variance(np.full(1, med, np.float32), ncnt, grad, EXAMPLE_NOISE)
        limits = sigma * np.sqrt(v[:, 0])
    return {'value': float(est.out[0]), 'kept': est.surv[:, 0].copy(), 'median': med,
            'limits': limits, 'spread': spread, 'floor_used': bool(est.floor_used[0])}


def explain(case: dict[str, Any], rejection: str, sigma: float, res: dict[str, Any]) -> str:
    """The result of `explore` in words."""
    n = len(case['values'])
    out = int(n - res['kept'].sum())
    diff = res['value'] - case['true']
    text = (f"Value of the voxel: {res['value']:.1f}. True value: {case['true']:.1f} "
            f"(difference {diff:+.1f}). Rejected: {out} of {n} members. ")
    if rejection == 'none':
        text += 'No rejection: all members go into the mean.'
    elif rejection == 'mad':
        lim = float(res['limits'][0])
        text += (f"Spread limit: a member goes out if it is more than {lim:.1f} from the median "
                 f"({sigma:g} x the spread {res['spread']:.1f}). ")
        text += ('The spread is the counting noise of one pixel here, because the members '
                 'agree better than that.' if res['floor_used'] else
                 'The spread comes from the scatter of the members (1.4826 x MAD).')
    else:
        lo, hi = float(res['limits'].min()), float(res['limits'].max())
        text += (f"Noise limit: each member has its own limit, {sigma:g} x its expected noise "
                 f"({lo:.1f} to {hi:.1f} here). A member from few pixels has a wider limit.")
    return text


class AveragingExplorer(QWidget):
    """One voxel, the members of its orbit and the effect of an averaging option."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(ui.SPACE[2])
        lay.addWidget(ui.label(hc.topic_title(hc.EXPLORER), 'page-title'))
        lay.addWidget(ui.label(hc.EXPLORER_INTRO, wrap=True))
        self.case_combo = QComboBox()
        for case in CASES:
            self.case_combo.addItem(case['name'])
        self.weight_combo = QComboBox()
        for key in oe.WEIGHTINGS:
            self.weight_combo.addItem(av.WEIGHT_LABELS[key], key)
        self.weight_combo.setCurrentIndex(self.weight_combo.findData('counts'))
        self.rej_combo = QComboBox()
        for key in ('none', 'mad', 'model'):
            self.rej_combo.addItem(av.REJECTION_LABELS[key], key)
        self.rej_combo.setCurrentIndex(self.rej_combo.findData('model'))
        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setRange(1.0, 10.0)
        self.sigma_spin.setSingleStep(0.5)
        self.sigma_spin.setDecimals(1)
        self.sigma_spin.setValue(4.0)
        grid = QGridLayout()
        grid.setHorizontalSpacing(ui.SPACE[2])
        grid.setVerticalSpacing(ui.SPACE[0])
        for j, (name, w) in enumerate((('Case', self.case_combo), ('Weights', self.weight_combo),
                                       ('Outlier rejection', self.rej_combo),
                                       ('Threshold (sigma)', self.sigma_spin))):
            grid.addWidget(ui.label(name, 'caption'), 0, j)
            grid.addWidget(w, 1, j)
        grid.setColumnStretch(4, 1)
        lay.addLayout(grid)
        self.case_label = ui.label('', wrap=True)
        lay.addWidget(self.case_label)
        self.figure = Figure(figsize=(6.4, 3.2), layout='constrained',
                             facecolor=ui.COLORS['surface'])
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(280)
        lay.addWidget(self.canvas, 1)
        self.result_label = ui.label('', 'strong', wrap=True, selectable=True)
        lay.addWidget(self.result_label)
        lay.addWidget(ui.hint(
            'The noise model of the example has the values gain 1, rel 0.02 and kappa 0.25. '
            'For your data, the program fits these values to the volume.'))
        self.case_combo.currentIndexChanged.connect(self._update)
        self.weight_combo.currentIndexChanged.connect(self._update)
        self.rej_combo.currentIndexChanged.connect(self._update)
        self.sigma_spin.valueChanged.connect(self._update)
        ui.guard_wheel(self)
        self._update()

    def result(self) -> dict[str, Any]:
        """The result for the selected case and option (see `explore`)."""
        return explore(CASES[self.case_combo.currentIndex()], self.weight_combo.currentData(),
                       self.rej_combo.currentData(), self.sigma_spin.value())

    def _update(self, *_: Any) -> None:
        case = CASES[self.case_combo.currentIndex()]
        rejection = self.rej_combo.currentData()
        sigma = self.sigma_spin.value()
        self.sigma_spin.setEnabled(rejection != 'none')
        res = self.result()
        self.case_label.setText(case['text'])
        self.result_label.setText(explain(case, rejection, sigma, res))
        self._draw(case, res)

    def _draw(self, case: dict[str, Any], res: dict[str, Any]) -> None:
        c = ui.COLORS
        vals = np.asarray(case['values'], float)
        x = np.arange(1, len(vals) + 1)
        kept = res['kept']
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        med = res['median']
        if res['limits'] is not None:
            # the range around the median in which a member stays
            for xi, lim in zip(x, res['limits']):
                ax.fill_between([xi - 0.42, xi + 0.42], med - lim, med + lim, color=c['tint'],
                                lw=0, zorder=1)
            ax.fill_between([], [], [], color=c['tint'], label='a member in this range stays')
        # the three lines stay visible if they have the same value: thick, dashed, dotted
        ax.axhline(res['value'], color=c['accent'], lw=3.0, zorder=2, label='value of the voxel')
        ax.axhline(case['true'], color=c['ok'], lw=1.4, ls=(0, (6, 4)), zorder=3,
                   label='true value')
        ax.axhline(med, color=c['muted'], lw=1.0, ls=':', zorder=3, label='median of the members')
        ax.plot(x[kept], vals[kept], 'o', ms=9, color=c['accent'], zorder=4,
                label='member that stays')
        if (~kept).any():
            ax.plot(x[~kept], vals[~kept], 'x', ms=10, mew=2.2, color=c['error'], zorder=4,
                    label='rejected member')
        ax.set_xticks(x, [f'{i}\nN = {n}' for i, n in zip(x, case['counts'])], fontsize=8)
        ax.set_xlim(0.4, len(vals) + 0.6)
        ax.set_xlabel('member of the orbit (N = its pixel count)', fontsize=8)
        ax.set_ylabel('intensity', fontsize=8)
        ax.tick_params(labelsize=8)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.legend(fontsize=7.5, loc='upper left', ncols=3, frameon=False,
                  bbox_to_anchor=(0.0, 1.16))
        self.canvas.draw_idle()


# ──────────────────────────────────────────────────────────────────
# The help pages
# ──────────────────────────────────────────────────────────────────

class HelpView(QWidget):
    """The list of the help topics and the text of the selected topic."""

    def __init__(self, parent: QWidget | None = None, popout: bool = True) -> None:
        super().__init__(parent)
        ui.ensure_theme()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(ui.SPACE[3])
        self.topics = QListWidget()
        self.topics.setFixedWidth(250)
        for key, title, _ in hc.TOPICS:
            item = QListWidgetItem(title)
            item.setData(Qt.ItemDataRole.UserRole, key)
            self.topics.addItem(item)
        self.topics.currentRowChanged.connect(self._on_row)
        lay.addWidget(self.topics)

        card = ui.Section('')
        card.title_label.setVisible(False)
        self.back_btn = ui.button('Back', 'link', 'Go back to the page that you came from.')
        self.back_btn.setVisible(False)
        card.header.insertWidget(0, self.back_btn)
        self.popout_btn = ui.button('Open in a window', 'link',
                                    'Show the help in its own window, next to your work.',
                                    clicked=self._popout)
        self.popout_btn.setVisible(popout)
        card.header.addWidget(self.popout_btn)
        self.stack = QStackedWidget()
        self.browser = _Browser()
        self.browser.setProperty('role', 'help')
        self.browser.setOpenLinks(False)
        self.browser.anchorClicked.connect(self._on_link)
        self.browser.document().setDefaultStyleSheet(_CSS)
        self.browser.document().setDocumentMargin(0)
        self.browser.setMaximumWidth(TEXT_WIDTH)
        text_page = QWidget()
        tl = QHBoxLayout(text_page)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.addWidget(self.browser, 100)       # the text takes the width up to TEXT_WIDTH
        tl.addStretch(1)
        self.stack.addWidget(text_page)
        self.explorer: AveragingExplorer | None = None      # made at its first use
        self._explorer_page = QWidget()
        el = QHBoxLayout(self._explorer_page)
        el.setContentsMargins(0, 0, 0, 0)
        self.stack.addWidget(self._explorer_page)
        card.add(self.stack, 1)
        lay.addWidget(card, 1)
        self._key = ''
        self.show_topic('start')

    def current_topic(self) -> str:
        return self._key

    def set_back(self, text: str | None) -> None:
        """Show the Back link with `text`, or hide it (None)."""
        self.back_btn.setVisible(bool(text))
        if text:
            self.back_btn.setText('← ' + text)

    def show_topic(self, key: str) -> None:
        """Show a topic. An unknown key shows the first topic."""
        keys = hc.topic_keys()
        if key not in keys:
            key = keys[0]
        self._key = key
        self.topics.blockSignals(True)
        self.topics.setCurrentRow(keys.index(key))
        self.topics.blockSignals(False)
        if key == hc.EXPLORER:
            if self.explorer is None:
                self.explorer = AveragingExplorer()
                self.explorer.setMaximumWidth(TEXT_WIDTH + 140)
                self._explorer_page.layout().addWidget(self.explorer, 100)
                self._explorer_page.layout().addStretch(1)
            self.stack.setCurrentIndex(1)
        else:
            self.browser.setHtml(hc.topic_html(key))
            self.stack.setCurrentIndex(0)

    def _on_row(self, row: int) -> None:
        if row >= 0:
            self.show_topic(self.topics.item(row).data(Qt.ItemDataRole.UserRole))

    def _on_link(self, url: QUrl) -> None:
        if url.scheme() == 'topic':
            self.show_topic(url.path())

    def _popout(self) -> None:
        show_help_dialog(self._key, self.window())


class HelpDialog(QDialog):
    """The help in its own window. It does not block the other windows."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle('rspace3d help')
        self.setWindowFlag(Qt.WindowType.WindowMinMaxButtonsHint, True)
        self.resize(1080, 760)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(ui.SPACE[3], ui.SPACE[3], ui.SPACE[3], ui.SPACE[3])
        self.view = HelpView(popout=False)
        lay.addWidget(self.view)


def show_help_dialog(topic: str, parent: QWidget | None = None) -> HelpDialog:
    """Show the help window at `topic`. An application has one help window."""
    app = QApplication.instance()
    dlg = getattr(app, '_rspace3d_help', None)
    if dlg is None:
        dlg = HelpDialog()
        app._rspace3d_help = dlg            # type: ignore[union-attr]
    dlg.view.show_topic(topic)
    dlg.show()
    dlg.raise_()
    dlg.activateWindow()
    return dlg


def document_text(key: str) -> str:
    """The plain text of a topic (for a search or a test)."""
    doc = QTextDocument()
    doc.setHtml(hc.topic_html(key))
    return doc.toPlainText()
