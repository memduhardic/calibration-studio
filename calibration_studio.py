"""
Calibration Studio — PySide6 desktop tool for hydrological model evaluation.

Left panel : CSV loading, observed / precipitation / model styling, axis
             scaling, legend editing, calibration & validation periods
Right panel: hydrograph with precipitation columns on a secondary axis
             + numerical goodness-of-fit table

Metrics (observed vs. each model output only, per period):
    NSE, KGE, RMSE, MAE, PBIAS, Pearson r, R², Rv (volume ratio)

All styling, names, axis ranges and legend settings are remembered between
sessions (auto-saved to ~/.calibration_studio.json) and can also be saved to
/ loaded from a named settings file.

Requirements:
    pip install PySide6 pandas numpy matplotlib

Run:
    python calibration_studio.py
"""

import sys
import re
import os
import json
import numpy as np
import pandas as pd

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QComboBox, QLineEdit, QCheckBox, QFileDialog,
    QScrollArea, QFrame, QTableWidget, QTableWidgetItem, QHeaderView,
    QColorDialog, QSplitter, QMessageBox, QGroupBox, QSizePolicy, QSpinBox
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.patches import Patch

# -----------------------------------------------------------------------------
# Colours (explicit everywhere so the app looks identical on light AND dark
# Windows themes — nothing inherits from the system palette)
# -----------------------------------------------------------------------------

INK = "#12222B"
INK_SOFT = "#5B6C74"
INK_FAINT = "#8A99A0"
BG = "#F4F6F7"
PANEL = "#FFFFFF"
PANEL_ALT = "#FAFBFC"
LINE = "#DCE4E7"
ACCENT = "#0E7C86"
ACCENT_DARK = "#0A5D65"
CAL_SHADE = "#0E7C86"
VAL_SHADE = "#C97B2E"

APP_STYLESHEET = f"""
QWidget {{
    background: {BG};
    color: {INK};
    font-size: 12px;
}}
QGroupBox {{
    background: {PANEL};
    border: 1px solid {LINE};
    border-radius: 8px;
    margin-top: 10px;
    padding: 10px 8px 8px 8px;
    font-weight: 600;
    color: {INK};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {INK_SOFT};
    background: transparent;
}}
QLabel {{
    background: transparent;
    color: {INK_SOFT};
}}
QPushButton {{
    background: {PANEL};
    color: {INK};
    border: 1px solid {LINE};
    border-radius: 6px;
    padding: 6px 10px;
}}
QPushButton:hover {{
    background: {PANEL_ALT};
    border-color: {INK_FAINT};
}}
QComboBox, QLineEdit, QSpinBox {{
    background: {PANEL};
    color: {INK};
    border: 1px solid {LINE};
    border-radius: 6px;
    padding: 5px 8px;
    selection-background-color: {ACCENT};
    selection-color: white;
}}
QComboBox QAbstractItemView {{
    background: {PANEL};
    color: {INK};
    border: 1px solid {LINE};
    selection-background-color: {ACCENT};
    selection-color: white;
}}
QCheckBox {{
    background: transparent;
    color: {INK};
    spacing: 6px;
}}
QScrollArea {{ border: none; background: transparent; }}
QTableWidget {{
    background: {PANEL};
    alternate-background-color: {PANEL_ALT};
    color: {INK};
    gridline-color: #EAEFF1;
    border: 1px solid {LINE};
    border-radius: 6px;
    font-size: 12px;
}}
QHeaderView::section {{
    background: {INK};
    color: white;
    font-weight: 600;
    padding: 6px;
    border: none;
}}
QTableWidget::item {{ color: {INK}; }}
QToolBar {{ background: {BG}; border: none; }}
QSplitter::handle {{ background: {LINE}; }}
QScrollBar:vertical {{
    background: {BG}; width: 10px; border: none;
}}
QScrollBar::handle:vertical {{
    background: {LINE}; border-radius: 5px; min-height: 24px;
}}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
"""

PALETTE = ["#0E7C86", "#C97B2E", "#7F77DD", "#D85A30",
           "#378ADD", "#639922", "#993556", "#5F5E5A"]

LINE_STYLES = {
    "Solid line":    dict(linestyle="-",  marker=""),
    "Dashed line":   dict(linestyle="--", marker=""),
    "Dotted line":   dict(linestyle=":",  marker=""),
    "Dash-dot line": dict(linestyle="-.", marker=""),
    "Line + points": dict(linestyle="-",  marker="o"),
    "Points only":   dict(linestyle="",   marker="o"),
}

LEGEND_LOCS = {
    "Upper right":   dict(loc="upper right"),
    "Upper left":    dict(loc="upper left"),
    "Lower right":   dict(loc="lower right"),
    "Lower left":    dict(loc="lower left"),
    "Center right":  dict(loc="center right"),
    "Outside right": dict(loc="upper left", bbox_to_anchor=(1.13, 1.0)),
    "Below chart":   dict(loc="upper center", bbox_to_anchor=(0.5, -0.13),
                          ncol=3),
    "Best (auto)":   dict(loc="best"),
}

LEGEND_ORDERS = [
    "Observed → Models → Precip",
    "Models → Observed → Precip",
    "Precip → Observed → Models",
]

PRECIP_AGG = ["Daily (raw)", "Weekly total", "Monthly total"]

TIME_KEYWORDS = ("date", "time", "tarih", "index")
OBS_KEYWORDS = ("observed", "obs", "gözlem", "gozlem", "measured", "gauge")
PRECIP_KEYWORDS = ("precip", "precipitation", "rain", "yağış", "yagis")
PRECIP_DEFAULT_COLOR = "#378ADD"

AUTOSAVE_PATH = os.path.join(os.path.expanduser("~"),
                             ".calibration_studio.json")


def parse_num(text):
    """Return float from a text box, or None if blank / unparseable."""
    try:
        t = str(text).strip()
        return float(t) if t else None
    except (TypeError, ValueError):
        return None


# -----------------------------------------------------------------------------
# Metrics — always observed (reference) vs. one simulated series
# -----------------------------------------------------------------------------

def compute_metrics(obs: np.ndarray, sim: np.ndarray) -> dict | None:
    mask = ~(np.isnan(obs) | np.isnan(sim))
    o, s = obs[mask], sim[mask]
    n = len(o)
    if n < 2:
        return None

    o_mean, s_mean = o.mean(), s.mean()
    o_std, s_std = o.std(), s.std()
    if o_std == 0 or s_std == 0 or o_mean == 0:
        return None

    diff = o - s
    nse = 1.0 - np.sum(diff ** 2) / np.sum((o - o_mean) ** 2)
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    pbias = float(100.0 * np.sum(diff) / np.sum(o))

    r = float(np.corrcoef(o, s)[0, 1])
    alpha = s_std / o_std

    # Rv = volume (water-balance) ratio = mean(sim) / mean(obs).
    # Because obs and sim are paired over the same mask, this is identical to
    # sum(sim)/sum(obs) and therefore satisfies exactly:
    #     Rv == 1 - PBIAS/100
    # It is also the beta term of KGE, so all three stay mutually consistent.
    beta = s_mean / o_mean
    rv = float(beta)

    kge = float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))

    return dict(n=n, nse=float(nse), kge=kge, rmse=rmse, mae=mae,
                pbias=pbias, r=r, r2=float(r ** 2), rv=rv)


# -----------------------------------------------------------------------------
# Per-series row widget
# -----------------------------------------------------------------------------

class SeriesRow(QFrame):
    def __init__(self, name: str, color: str, on_change, on_move):
        super().__init__()
        self.name = name
        self.color = color
        self.on_change = on_change

        self.setStyleSheet(
            f"QFrame{{background:{PANEL_ALT};border:1px solid {LINE};"
            f"border-radius:6px;}}")

        lay = QGridLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setHorizontalSpacing(5)
        lay.setVerticalSpacing(5)

        self.chk = QCheckBox(name)
        self.chk.setChecked(True)
        self.chk.setToolTip(f"Show / hide {name}")
        self.chk.setStyleSheet(
            f"QCheckBox{{border:none;background:transparent;"
            f"color:{INK};font-weight:600;}}")
        self.chk.stateChanged.connect(lambda _: self.on_change())

        up = QPushButton("▲")
        dn = QPushButton("▼")
        for b, delta in ((up, -1), (dn, +1)):
            b.setFixedSize(24, 22)
            b.setToolTip("Move up / down (changes draw and legend order)")
            b.clicked.connect(lambda _=False, d=delta: on_move(self.name, d))

        self.label_edit = QLineEdit(name)
        self.label_edit.setToolTip("Name shown in the chart legend")
        self.label_edit.editingFinished.connect(self.on_change)

        self.color_btn = QPushButton()
        self.color_btn.setFixedSize(26, 24)
        self.color_btn.setToolTip("Series colour")
        self.color_btn.clicked.connect(self.pick_color)
        self._paint_color_btn()

        self.style_cmb = QComboBox()
        self.style_cmb.addItems(LINE_STYLES.keys())
        self.style_cmb.currentIndexChanged.connect(lambda _: self.on_change())

        lay.addWidget(self.chk, 0, 0, 1, 2)
        lay.addWidget(up, 0, 2)
        lay.addWidget(dn, 0, 3)
        lay.addWidget(self.label_edit, 1, 0, 1, 4)
        lay.addWidget(self.color_btn, 2, 0)
        lay.addWidget(self.style_cmb, 2, 1, 1, 3)

    def _paint_color_btn(self):
        self.color_btn.setStyleSheet(
            f"QPushButton{{background:{self.color};border:1px solid #B4B8BC;"
            f"border-radius:5px;}}")

    def pick_color(self):
        c = QColorDialog.getColor(QColor(self.color), self, "Series colour")
        if c.isValid():
            self.color = c.name()
            self._paint_color_btn()
            self.on_change()

    @property
    def visible(self) -> bool:
        return self.chk.isChecked()

    @property
    def label(self) -> str:
        return self.label_edit.text().strip() or self.name

    @property
    def style(self) -> dict:
        return LINE_STYLES[self.style_cmb.currentText()]


# -----------------------------------------------------------------------------
# Main window
# -----------------------------------------------------------------------------

class CalibrationStudio(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Calibration Studio — model evaluation")
        self.resize(1440, 880)

        self.df: pd.DataFrame | None = None
        self.time_col: str | None = None
        self.precip_col: str | None = None
        self.x = None
        self.x_is_dates = False
        self.series_rows: dict[str, SeriesRow] = {}
        self.order: list[str] = []
        self.last_scores: list[dict] = []
        self._loading = False          # suppresses refresh while applying settings

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_sidebar())
        splitter.addWidget(self._build_main())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([350, 1090])
        self.setCentralWidget(splitter)

    # ---------------------------------------------------------------- sidebar
    def _build_sidebar(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(320)
        panel.setMaximumWidth(400)

        inner = QWidget()
        outer = QVBoxLayout(inner)
        outer.setContentsMargins(12, 8, 12, 12)
        outer.setSpacing(10)

        # --- data source
        g_data = QGroupBox("Data source")
        v = QVBoxLayout(g_data)
        self.load_btn = QPushButton("Open CSV…")
        self.load_btn.clicked.connect(self.open_csv)
        self.file_lbl = QLabel("No file loaded")
        self.file_lbl.setStyleSheet(f"color:{INK_FAINT};font-size:11px;")
        self.file_lbl.setWordWrap(True)
        self.time_lbl = QLabel("")
        self.time_lbl.setStyleSheet(
            f"color:{ACCENT};font-size:11px;font-weight:600;")
        v.addWidget(self.load_btn)
        v.addWidget(self.file_lbl)
        v.addWidget(self.time_lbl)
        outer.addWidget(g_data)

        # --- observed series
        self.g_obs = QGroupBox("Observed series")
        v = QVBoxLayout(self.g_obs)
        self.obs_cmb = QComboBox()
        self.obs_cmb.currentIndexChanged.connect(self.on_obs_changed)
        v.addWidget(self.obs_cmb)

        v.addWidget(QLabel("Legend name"))
        self.obs_label_edit = QLineEdit("Observed")
        self.obs_label_edit.editingFinished.connect(self.refresh)
        v.addWidget(self.obs_label_edit)

        self.obs_color = INK
        style_row = QHBoxLayout()
        self.obs_color_btn = QPushButton()
        self.obs_color_btn.setFixedSize(26, 24)
        self.obs_color_btn.setToolTip("Observed colour")
        self.obs_color_btn.clicked.connect(self.pick_obs_color)
        self._paint_obs_color_btn()
        self.obs_style_cmb = QComboBox()
        self.obs_style_cmb.addItems(LINE_STYLES.keys())
        self.obs_style_cmb.currentIndexChanged.connect(lambda _: self.refresh())
        style_row.addWidget(self.obs_color_btn)
        style_row.addWidget(self.obs_style_cmb, stretch=1)
        v.addLayout(style_row)

        self.obs_back_chk = QCheckBox("Draw behind model lines")
        self.obs_back_chk.setChecked(True)
        self.obs_back_chk.setToolTip(
            "Keeps the observed series as a background reference so the "
            "simulated curves stay visible on top of it.")
        self.obs_back_chk.stateChanged.connect(lambda _: self.refresh())
        v.addWidget(self.obs_back_chk)

        note = QLabel("All metrics are computed against this series only.")
        note.setStyleSheet(f"color:{INK_FAINT};font-size:11px;")
        note.setWordWrap(True)
        v.addWidget(note)
        self.g_obs.setVisible(False)
        outer.addWidget(self.g_obs)

        # --- precipitation
        self.g_precip = QGroupBox("Precipitation")
        v = QVBoxLayout(self.g_precip)
        self.precip_lbl = QLabel("")
        self.precip_lbl.setStyleSheet(
            f"color:{ACCENT};font-size:11px;font-weight:600;")
        self.precip_lbl.setWordWrap(True)
        v.addWidget(self.precip_lbl)

        self.precip_chk = QCheckBox("Show as columns (secondary axis)")
        self.precip_chk.setChecked(True)
        self.precip_chk.stateChanged.connect(lambda _: self.refresh())
        v.addWidget(self.precip_chk)

        v.addWidget(QLabel("Aggregation"))
        self.precip_agg_cmb = QComboBox()
        self.precip_agg_cmb.addItems(PRECIP_AGG)
        self.precip_agg_cmb.setToolTip(
            "Daily bars become invisible over long records — "
            "aggregate to weekly or monthly totals for readable columns.")
        self.precip_agg_cmb.currentIndexChanged.connect(lambda _: self.refresh())
        v.addWidget(self.precip_agg_cmb)

        row = QHBoxLayout()
        self.precip_color = PRECIP_DEFAULT_COLOR
        self.precip_color_btn = QPushButton()
        self.precip_color_btn.setFixedSize(26, 24)
        self.precip_color_btn.setToolTip("Precipitation colour")
        self.precip_color_btn.clicked.connect(self.pick_precip_color)
        self._paint_precip_color_btn()
        row.addWidget(self.precip_color_btn)
        row.addWidget(QLabel("Legend name"), stretch=1)
        v.addLayout(row)
        self.precip_name_edit = QLineEdit("Precipitation")
        self.precip_name_edit.editingFinished.connect(self.refresh)
        v.addWidget(self.precip_name_edit)

        v.addWidget(QLabel("Axis label"))
        self.precip_label_edit = QLineEdit("Precipitation (mm)")
        self.precip_label_edit.editingFinished.connect(self.refresh)
        v.addWidget(self.precip_label_edit)

        self.g_precip.setVisible(False)
        outer.addWidget(self.g_precip)

        # --- axis scaling
        self.g_axes = QGroupBox("Axis scaling")
        grid = QGridLayout(self.g_axes)
        grid.setVerticalSpacing(6)

        grid.addWidget(QLabel("Flow min"), 0, 0)
        grid.addWidget(QLabel("Flow max"), 0, 1)
        self.flow_min_edit = QLineEdit()
        self.flow_min_edit.setPlaceholderText("auto")
        self.flow_max_edit = QLineEdit()
        self.flow_max_edit.setPlaceholderText("auto")
        grid.addWidget(self.flow_min_edit, 1, 0)
        grid.addWidget(self.flow_max_edit, 1, 1)

        self.log_chk = QCheckBox("Log scale (flow axis)")
        self.log_chk.setToolTip(
            "Useful when low flows are compressed against the axis.")
        grid.addWidget(self.log_chk, 2, 0, 1, 2)

        grid.addWidget(QLabel("Precip axis max"), 3, 0)
        grid.addWidget(QLabel("Precip band"), 3, 1)
        self.precip_max_edit = QLineEdit()
        self.precip_max_edit.setPlaceholderText("auto")
        self.precip_max_edit.setToolTip(
            "Cap the precipitation axis so typical events fill the band "
            "instead of a few extreme days.")
        self.band_spin = QSpinBox()
        self.band_spin.setRange(10, 60)
        self.band_spin.setValue(30)
        self.band_spin.setSuffix(" %")
        self.band_spin.setToolTip(
            "Share of the plot height reserved for precipitation columns. "
            "The flow curves are scaled into the remaining space.")
        grid.addWidget(self.precip_max_edit, 4, 0)
        grid.addWidget(self.band_spin, 4, 1)

        for w in (self.flow_min_edit, self.flow_max_edit, self.precip_max_edit):
            w.editingFinished.connect(self.refresh)
        self.log_chk.stateChanged.connect(lambda _: self.refresh())
        self.band_spin.valueChanged.connect(lambda _: self.refresh())

        self.g_axes.setVisible(False)
        outer.addWidget(self.g_axes)

        # --- legend
        self.g_legend = QGroupBox("Legend")
        grid = QGridLayout(self.g_legend)
        grid.setVerticalSpacing(6)

        self.legend_chk = QCheckBox("Show legend")
        self.legend_chk.setChecked(True)
        self.legend_frame_chk = QCheckBox("Background box")
        self.legend_frame_chk.setChecked(True)
        grid.addWidget(self.legend_chk, 0, 0, 1, 2)
        grid.addWidget(self.legend_frame_chk, 1, 0, 1, 2)

        grid.addWidget(QLabel("Position"), 2, 0)
        self.legend_loc_cmb = QComboBox()
        self.legend_loc_cmb.addItems(LEGEND_LOCS.keys())
        grid.addWidget(self.legend_loc_cmb, 2, 1)

        grid.addWidget(QLabel("Order"), 3, 0)
        self.legend_order_cmb = QComboBox()
        self.legend_order_cmb.addItems(LEGEND_ORDERS)
        self.legend_order_cmb.setToolTip(
            "Model order within the legend follows the ▲▼ buttons "
            "in Model outputs.")
        grid.addWidget(self.legend_order_cmb, 3, 1)

        grid.addWidget(QLabel("Calibration label"), 4, 0, 1, 2)
        self.cal_label_edit = QLineEdit("Calibration period")
        grid.addWidget(self.cal_label_edit, 5, 0, 1, 2)
        grid.addWidget(QLabel("Validation label"), 6, 0, 1, 2)
        self.val_label_edit = QLineEdit("Validation period")
        grid.addWidget(self.val_label_edit, 7, 0, 1, 2)

        for w in (self.cal_label_edit, self.val_label_edit):
            w.editingFinished.connect(self.refresh)
        for w in (self.legend_chk, self.legend_frame_chk):
            w.stateChanged.connect(lambda _: self.refresh())
        for w in (self.legend_loc_cmb, self.legend_order_cmb):
            w.currentIndexChanged.connect(lambda _: self.refresh())

        self.g_legend.setVisible(False)
        outer.addWidget(self.g_legend)

        # --- periods
        self.g_periods = QGroupBox("Periods")
        grid = QGridLayout(self.g_periods)
        grid.setVerticalSpacing(6)

        cal_hdr = QLabel("Calibration")
        cal_hdr.setStyleSheet(f"color:{CAL_SHADE};font-weight:600;")
        grid.addWidget(cal_hdr, 0, 0, 1, 2)
        grid.addWidget(QLabel("Start"), 1, 0)
        grid.addWidget(QLabel("End"), 1, 1)
        self.cal_start = QLineEdit()
        self.cal_end = QLineEdit()
        grid.addWidget(self.cal_start, 2, 0)
        grid.addWidget(self.cal_end, 2, 1)

        self.val_chk = QCheckBox("Validation period")
        self.val_chk.setChecked(True)
        self.val_chk.setStyleSheet(
            f"QCheckBox{{color:{VAL_SHADE};font-weight:600;}}")
        grid.addWidget(self.val_chk, 3, 0, 1, 2)
        grid.addWidget(QLabel("Start"), 4, 0)
        grid.addWidget(QLabel("End"), 4, 1)
        self.val_start = QLineEdit()
        self.val_end = QLineEdit()
        grid.addWidget(self.val_start, 5, 0)
        grid.addWidget(self.val_end, 5, 1)

        apply_btn = QPushButton("Apply periods")
        apply_btn.clicked.connect(self.refresh)
        grid.addWidget(apply_btn, 6, 0, 1, 2)

        self.period_hint = QLabel("")
        self.period_hint.setStyleSheet(f"color:{INK_FAINT};font-size:11px;")
        self.period_hint.setWordWrap(True)
        grid.addWidget(self.period_hint, 7, 0, 1, 2)

        for w in (self.cal_start, self.cal_end, self.val_start, self.val_end):
            w.editingFinished.connect(self.refresh)
        self.val_chk.stateChanged.connect(lambda _: self.refresh())

        self.g_periods.setVisible(False)
        outer.addWidget(self.g_periods)

        # --- model outputs
        self.g_series = QGroupBox("Model outputs")
        v = QVBoxLayout(self.g_series)
        btn_row = QHBoxLayout()
        b_all = QPushButton("Select all")
        b_none = QPushButton("Clear")
        b_all.clicked.connect(lambda: self._set_all(True))
        b_none.clicked.connect(lambda: self._set_all(False))
        btn_row.addWidget(b_all)
        btn_row.addWidget(b_none)
        v.addLayout(btn_row)

        self.series_container = QWidget()
        self.series_container.setStyleSheet("background:transparent;")
        self.series_layout = QVBoxLayout(self.series_container)
        self.series_layout.setContentsMargins(0, 0, 0, 0)
        self.series_layout.setSpacing(6)
        self.series_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.series_container)
        scroll.setMinimumHeight(200)
        v.addWidget(scroll)
        self.g_series.setVisible(False)
        outer.addWidget(self.g_series, stretch=1)

        # --- chart labels
        self.g_labels = QGroupBox("Chart labels")
        v = QVBoxLayout(self.g_labels)
        v.addWidget(QLabel("Y-axis label"))
        self.y_label_edit = QLineEdit("Discharge (m³/s)")
        self.y_label_edit.editingFinished.connect(self.refresh)
        v.addWidget(self.y_label_edit)
        v.addWidget(QLabel("Chart title"))
        self.title_edit = QLineEdit("")
        self.title_edit.editingFinished.connect(self.refresh)
        v.addWidget(self.title_edit)
        self.g_labels.setVisible(False)
        outer.addWidget(self.g_labels)

        # --- export & settings
        self.export_btn = QPushButton("Export scores (CSV)")
        self.export_btn.setStyleSheet(
            f"QPushButton{{background:{ACCENT};color:white;font-weight:600;"
            f"padding:9px;border-radius:6px;border:none;}}"
            f"QPushButton:hover{{background:{ACCENT_DARK};}}")
        self.export_btn.clicked.connect(self.export_scores)
        self.export_btn.setVisible(False)
        outer.addWidget(self.export_btn)

        self.export_plots_btn = QPushButton("Export model graphs (PNG)")
        self.export_plots_btn.setStyleSheet(
            f"QPushButton{{background:{PANEL};color:{INK};font-weight:600;"
            f"padding:9px;border-radius:6px;border:1px solid {ACCENT};}}"
            f"QPushButton:hover{{background:{PANEL_ALT};}}")
        self.export_plots_btn.clicked.connect(self.export_individual_plots)
        self.export_plots_btn.setVisible(False)
        outer.addWidget(self.export_plots_btn)

        self.g_settings = QGroupBox("Settings")
        row = QHBoxLayout(self.g_settings)
        b_save = QPushButton("Save…")
        b_load = QPushButton("Load…")
        b_save.setToolTip("Save colours, names, axis ranges and legend setup")
        b_load.setToolTip("Load a saved setup")
        b_save.clicked.connect(self.save_settings_dialog)
        b_load.clicked.connect(self.load_settings_dialog)
        row.addWidget(b_save)
        row.addWidget(b_load)
        self.g_settings.setVisible(False)
        outer.addWidget(self.g_settings)

        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setWidget(inner)
        wrap = QVBoxLayout(panel)
        wrap.setContentsMargins(0, 0, 0, 0)
        wrap.addWidget(sidebar_scroll)
        return panel

    # ------------------------------------------------------------------- main
    def _build_main(self) -> QWidget:
        panel = QWidget()
        v = QVBoxLayout(panel)
        v.setContentsMargins(12, 8, 12, 12)
        v.setSpacing(8)

        self.fig = Figure(figsize=(9, 4.5), tight_layout=True)
        self.fig.patch.set_facecolor(BG)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar = NavigationToolbar2QT(self.canvas, panel)

        v.addWidget(self.toolbar)
        v.addWidget(self.canvas, stretch=3)

        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        v.addWidget(self.table, stretch=2)

        self.metrics_note = QLabel("")
        self.metrics_note.setWordWrap(True)
        self.metrics_note.setStyleSheet(f"color:{INK_FAINT};font-size:11px;")
        v.addWidget(self.metrics_note)

        return panel

    # ------------------------------------------------------------- colour UI
    def _paint_obs_color_btn(self):
        self.obs_color_btn.setStyleSheet(
            f"QPushButton{{background:{self.obs_color};border:1px solid #B4B8BC;"
            f"border-radius:5px;}}")

    def pick_obs_color(self):
        c = QColorDialog.getColor(QColor(self.obs_color), self, "Observed colour")
        if c.isValid():
            self.obs_color = c.name()
            self._paint_obs_color_btn()
            self.refresh()

    def _paint_precip_color_btn(self):
        self.precip_color_btn.setStyleSheet(
            f"QPushButton{{background:{self.precip_color};"
            f"border:1px solid #B4B8BC;border-radius:5px;}}")

    def pick_precip_color(self):
        c = QColorDialog.getColor(QColor(self.precip_color), self,
                                  "Precipitation colour")
        if c.isValid():
            self.precip_color = c.name()
            self._paint_precip_color_btn()
            self.refresh()

    # -------------------------------------------------------------- data load
    def open_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open calibration CSV", "", "CSV files (*.csv);;All files (*)")
        if not path:
            return
        try:
            df = pd.read_csv(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Read error", f"Could not read CSV:\n{e}")
            return

        if df.shape[1] < 2:
            QMessageBox.warning(self, "Invalid file",
                                "The CSV needs at least two columns "
                                "(observed + one model output).")
            return

        self.df = df
        self.file_lbl.setText(f"{path}\n{len(df)} rows × {df.shape[1]} columns")
        self.title_edit.setText(os.path.splitext(os.path.basename(path))[0])

        self._detect_time_column()
        self._detect_precip_column()
        self._populate_obs_combo()
        self._init_periods()

        for g in (self.g_obs, self.g_axes, self.g_legend, self.g_periods,
                  self.g_series, self.g_labels, self.g_settings):
            g.setVisible(True)
        self.g_precip.setVisible(self.precip_col is not None)
        self.export_btn.setVisible(True)
        self.export_plots_btn.setVisible(True)

        self.rebuild_series_rows()

        # restore the previous session's look, if it fits this file
        data = self._read_settings_file(AUTOSAVE_PATH)
        if data:
            self._apply_settings(data, restore_periods=False)

        self.refresh()

    def _detect_time_column(self):
        cols = list(self.df.columns)
        candidate = next(
            (c for c in cols if any(k in c.lower() for k in TIME_KEYWORDS)), None)

        if candidate is None:
            for c in cols:
                if self.df[c].dtype == object:
                    parsed = pd.to_datetime(self.df[c], errors="coerce")
                    if parsed.notna().mean() > 0.9:
                        candidate = c
                        break

        self.time_col = candidate
        if candidate is not None:
            parsed = pd.to_datetime(self.df[candidate], errors="coerce")
            if parsed.notna().mean() > 0.9:
                self.x = parsed
                self.x_is_dates = True
            else:
                self.x = self.df[candidate]
                self.x_is_dates = False
            self.time_lbl.setText(f"Time column detected: {candidate}")
        else:
            self.x = pd.Series(np.arange(len(self.df)))
            self.x_is_dates = False
            self.time_lbl.setText("No time column found — using row index")

    def _detect_precip_column(self):
        cols = [c for c in self.df.columns if c != self.time_col]
        candidate = next(
            (c for c in cols if any(k in c.lower() for k in PRECIP_KEYWORDS)),
            None)
        self.precip_col = candidate
        self.precip_lbl.setText(
            f"Column detected: {candidate}" if candidate else "")
        if candidate:
            self.precip_name_edit.setText(candidate)

    def _populate_obs_combo(self):
        cols = [c for c in self.df.columns
                if c not in (self.time_col, self.precip_col)]
        self.obs_cmb.blockSignals(True)
        self.obs_cmb.clear()
        self.obs_cmb.addItems(cols)
        guess = None
        for kw in OBS_KEYWORDS:
            guess = next((c for c in cols if kw in c.lower()), None)
            if guess:
                break
        if guess:
            self.obs_cmb.setCurrentText(guess)
        self.obs_cmb.blockSignals(False)
        self.obs_label_edit.setText(self.obs_cmb.currentText())

    def _init_periods(self):
        n = len(self.df)
        split = int(n * 0.7)

        def fmt(i):
            if self.x_is_dates:
                return self.x.iloc[i].strftime("%Y-%m-%d")
            return str(i)

        self.cal_start.setText(fmt(0))
        self.cal_end.setText(fmt(max(0, split - 1)))
        self.val_start.setText(fmt(split))
        self.val_end.setText(fmt(n - 1))
        self.period_hint.setText(
            "Enter dates as YYYY-MM-DD." if self.x_is_dates
            else f"No dates in file — enter row numbers (0 to {n - 1}).")

    def _period_mask(self, start_txt: str, end_txt: str) -> np.ndarray | None:
        try:
            if self.x_is_dates:
                start = pd.to_datetime(start_txt)
                end = pd.to_datetime(end_txt)
                return ((self.x >= start) & (self.x <= end)).to_numpy()
            start = int(float(start_txt))
            end = int(float(end_txt))
            idx = np.arange(len(self.df))
            return (idx >= start) & (idx <= end)
        except Exception:  # noqa: BLE001
            return None

    def on_obs_changed(self):
        if self.df is None:
            return
        self.obs_label_edit.setText(self.obs_cmb.currentText())
        self.rebuild_series_rows()
        self.refresh()

    # -------------------------------------------------------- series ordering
    def rebuild_series_rows(self):
        for row in self.series_rows.values():
            row.setParent(None)
            row.deleteLater()
        self.series_rows.clear()
        self.order = []

        obs_col = self.obs_cmb.currentText()
        for c in self.df.columns:
            if c in (self.time_col, obs_col, self.precip_col):
                continue
            if pd.to_numeric(self.df[c], errors="coerce").notna().mean() > 0.5:
                self.order.append(c)

        for i, name in enumerate(self.order):
            self.series_rows[name] = SeriesRow(
                name, PALETTE[i % len(PALETTE)], self.refresh, self._move_series)
        self._relayout_series()

    def _relayout_series(self):
        while self.series_layout.count():
            item = self.series_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
        for name in self.order:
            w = self.series_rows[name]
            self.series_layout.addWidget(w)
            w.show()
        self.series_layout.addStretch()

    def _move_series(self, name: str, delta: int):
        if name not in self.order:
            return
        i = self.order.index(name)
        j = i + delta
        if 0 <= j < len(self.order):
            self.order[i], self.order[j] = self.order[j], self.order[i]
            self._relayout_series()
            self.refresh()

    def _set_all(self, value: bool):
        for row in self.series_rows.values():
            row.chk.blockSignals(True)
            row.chk.setChecked(value)
            row.chk.blockSignals(False)
        self.refresh()

    # ------------------------------------------------------------- precip data
    def _precip_series(self):
        """Return (x_values, values, bar_width) honouring the aggregation
        setting, or None when precipitation isn't available/shown."""
        if self.precip_col is None or not self.precip_chk.isChecked():
            return None

        raw = pd.to_numeric(self.df[self.precip_col], errors="coerce")
        if not np.isfinite(raw.to_numpy(float)).any():
            return None

        mode = self.precip_agg_cmb.currentText()
        if self.x_is_dates and mode != "Daily (raw)":
            s = pd.Series(raw.to_numpy(float), index=pd.DatetimeIndex(self.x))
            rule, width = ("W", 7 * 0.85) if mode == "Weekly total" \
                else ("MS", 30 * 0.85)
            agg = s.resample(rule).sum(min_count=1)
            return agg.index, agg.to_numpy(float), width

        if self.x_is_dates:
            diffs = np.diff(self.x.values).astype("timedelta64[s]"
                                                  ).astype(float) / 86400.0
            width = float(np.median(diffs)) * 0.9 if len(diffs) else 1.0
        else:
            width = 0.8
        return self.x, raw.to_numpy(float), width

    # ------------------------------------------------------------ chart engine
    def _draw_chart(self, fig, ax, models, title):
        """Draw observed + the given models (+ precipitation) onto ax.

        `models` is a list of (column_name, SeriesRow). Shared by the on-screen
        chart and the per-model PNG export so both look identical.
        """
        obs_col = self.obs_cmb.currentText()
        obs = pd.to_numeric(self.df[obs_col], errors="coerce").to_numpy(float)

        ax.set_facecolor("white")

        cal_mask = self._period_mask(self.cal_start.text(), self.cal_end.text())
        val_mask = (self._period_mask(self.val_start.text(), self.val_end.text())
                    if self.val_chk.isChecked() else None)

        if cal_mask is not None and cal_mask.any():
            i0, i1 = np.where(cal_mask)[0][[0, -1]]
            ax.axvspan(self.x.iloc[i0], self.x.iloc[i1],
                       color=CAL_SHADE, alpha=0.07, zorder=0)
        if val_mask is not None and val_mask.any():
            i0, i1 = np.where(val_mask)[0][[0, -1]]
            ax.axvspan(self.x.iloc[i0], self.x.iloc[i1],
                       color=VAL_SHADE, alpha=0.10, zorder=0)

        n = len(self.df)

        def marker_every(style_name):
            if style_name == "Points only":
                return 1
            return max(1, n // 400)

        obs_style_name = self.obs_style_cmb.currentText()
        obs_st = LINE_STYLES[obs_style_name]
        obs_behind = self.obs_back_chk.isChecked()
        obs_z = 2 if obs_behind else 20
        obs_handle, = ax.plot(
            self.x, obs, color=self.obs_color, lw=2.0,
            linestyle=obs_st["linestyle"] if obs_st["linestyle"] else "None",
            marker=obs_st["marker"] if obs_st["marker"] else None,
            markersize=3, markevery=marker_every(obs_style_name),
            label=self.obs_label_edit.text().strip() or obs_col, zorder=obs_z)

        flow_arrays = [obs]
        model_handles, model_labels = [], []
        n_models = len(models)
        for i, (name, row) in enumerate(models):
            sim = pd.to_numeric(self.df[name], errors="coerce").to_numpy(float)
            flow_arrays.append(sim)
            st = row.style
            style_name = row.style_cmb.currentText()
            # first model in the sidebar list sits on top of the later ones
            h, = ax.plot(
                self.x, sim, color=row.color, lw=1.7,
                linestyle=st["linestyle"] if st["linestyle"] else "None",
                marker=st["marker"] if st["marker"] else None,
                markersize=3, markevery=marker_every(style_name),
                label=row.label, zorder=3 + (n_models - i))
            model_handles.append(h)
            model_labels.append(row.label)

        ax.set_ylabel(self.y_label_edit.text() or "Value", color=INK_SOFT)
        ax.set_title(title, fontsize=12.5, fontweight="bold", loc="left",
                     color=INK)
        ax.grid(color="#EAEFF1", lw=0.8)
        ax.tick_params(colors=INK_SOFT)
        for spine in ax.spines.values():
            spine.set_color(LINE)
        ax.spines[["top", "right"]].set_visible(False)

        # ---- precipitation on an inverted twin axis ------------------------
        precip_handle = precip_label = None
        ax2 = None
        precip_data = self._precip_series()
        band = 0.0
        if precip_data is not None:
            px, pv, width = precip_data
            ax2 = ax.twinx()
            band = self.band_spin.value() / 100.0
            precip_label = (self.precip_name_edit.text().strip()
                            or self.precip_col)
            bars = ax2.bar(px, pv, width=width, color=self.precip_color,
                           alpha=0.75, linewidth=0, label=precip_label, zorder=1)
            precip_handle = bars

            pmax = parse_num(self.precip_max_edit.text())
            if pmax is None or pmax <= 0:
                finite = pv[np.isfinite(pv)]
                pmax = float(finite.max()) if len(finite) and finite.max() > 0 else 1.0
            ax2.set_ylim(pmax / band, 0)     # 0 at top; bars hang downward

            ax2.set_ylabel(self.precip_label_edit.text() or "Precipitation (mm)",
                           color=INK_SOFT)
            ax2.tick_params(colors=INK_SOFT)
            for spine in ax2.spines.values():
                spine.set_color(LINE)
            ax2.spines[["top", "left"]].set_visible(False)
            ax2.set_zorder(ax.get_zorder() - 1)
            ax.patch.set_visible(False)

        self._apply_flow_limits(ax, flow_arrays, band)

        # ---- legend --------------------------------------------------------
        if self.legend_chk.isChecked():
            groups = {
                "obs": ([obs_handle], [obs_handle.get_label()]),
                "models": (model_handles, model_labels),
                "precip": (([precip_handle], [precip_label])
                           if precip_handle is not None else ([], [])),
            }
            order_key = self.legend_order_cmb.currentText()
            if order_key == LEGEND_ORDERS[1]:
                seq = ["models", "obs", "precip"]
            elif order_key == LEGEND_ORDERS[2]:
                seq = ["precip", "obs", "models"]
            else:
                seq = ["obs", "models", "precip"]

            handles, labels = [], []
            for key in seq:
                h, l = groups[key]
                handles += list(h)
                labels += list(l)

            if cal_mask is not None and cal_mask.any():
                handles.append(Patch(facecolor=CAL_SHADE, alpha=0.15))
                labels.append(self.cal_label_edit.text().strip()
                              or "Calibration period")
            if val_mask is not None and val_mask.any():
                handles.append(Patch(facecolor=VAL_SHADE, alpha=0.20))
                labels.append(self.val_label_edit.text().strip()
                              or "Validation period")

            kwargs = dict(LEGEND_LOCS[self.legend_loc_cmb.currentText()])
            frame = self.legend_frame_chk.isChecked()
            ax.legend(handles, labels, fontsize=9, frameon=frame,
                      facecolor="white" if frame else "none",
                      framealpha=0.85 if frame else 0.0,
                      edgecolor="none", **kwargs)

        if self.x_is_dates:
            fig.autofmt_xdate(rotation=0, ha="center")

        return cal_mask, val_mask

    def _apply_flow_limits(self, ax, flow_arrays, band):
        """Set the flow-axis range, leaving `band` of the height free at the
        top for the precipitation columns."""
        finite = [a[np.isfinite(a)] for a in flow_arrays]
        finite = [a for a in finite if len(a)]
        if not finite:
            return
        allv = np.concatenate(finite)
        data_min, data_max = float(allv.min()), float(allv.max())

        user_min = parse_num(self.flow_min_edit.text())
        user_max = parse_num(self.flow_max_edit.text())

        if self.log_chk.isChecked():
            pos = allv[allv > 0]
            lo = user_min if (user_min and user_min > 0) else (
                float(pos.min()) * 0.8 if len(pos) else 0.001)
            top = user_max if user_max else data_max * 1.1
            if user_max is None and band > 0 and top > lo:
                top = lo * (top / lo) ** (1.0 / (1.0 - band))
            ax.set_yscale("log")
            ax.set_ylim(lo, top)
            return

        ax.set_yscale("linear")
        lo = user_min if user_min is not None else min(0.0, data_min)
        if user_max is not None:
            hi = user_max
        else:
            span = (data_max - lo) * 1.05
            span = span if span > 0 else 1.0
            hi = lo + (span / (1.0 - band) if band > 0 else span)
        ax.set_ylim(lo, hi)

    # ---------------------------------------------------------------- refresh
    def refresh(self):
        if self.df is None or self._loading:
            return

        visible = [(n, self.series_rows[n]) for n in self.order
                   if self.series_rows[n].visible]

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        cal_mask, val_mask = self._draw_chart(
            self.fig, ax, visible, self.title_edit.text() or "Hydrograph")
        self.canvas.draw_idle()

        # ---- scores --------------------------------------------------------
        obs_col = self.obs_cmb.currentText()
        obs = pd.to_numeric(self.df[obs_col], errors="coerce").to_numpy(float)

        self.last_scores = []
        periods = []
        if cal_mask is not None and cal_mask.any():
            periods.append(("Calibration", cal_mask))
        if val_mask is not None and val_mask.any():
            periods.append(("Validation", val_mask))
        if not periods:
            periods.append(("Full record", np.ones(len(self.df), dtype=bool)))

        for name, row in visible:
            sim = pd.to_numeric(self.df[name], errors="coerce").to_numpy(float)
            for period_name, mask in periods:
                m = compute_metrics(obs[mask], sim[mask])
                if m:
                    self.last_scores.append(
                        dict(model=row.label, period=period_name, **m))
        self._fill_table()
        self._update_metrics_note(periods)
        self._autosave()

    def _update_metrics_note(self, periods):
        """Spell out how Rv is defined and exactly which rows each period used,
        so results can be reproduced by hand."""
        parts = []
        for period_name, mask in periods:
            idx = np.where(mask)[0]
            if not len(idx):
                continue
            if self.x_is_dates:
                lo = self.x.iloc[idx[0]].strftime("%Y-%m-%d")
                hi = self.x.iloc[idx[-1]].strftime("%Y-%m-%d")
            else:
                lo, hi = str(idx[0]), str(idx[-1])
            parts.append(f"{period_name}: {lo} → {hi} ({len(idx)} rows)")

        self.metrics_note.setText(
            "Rv = mean(simulated) / mean(observed) over the paired rows, "
            "so Rv = 1 − PBIAS/100 exactly and equals the β term of KGE.  "
            "Rows where the observed value or that model's value is missing "
            "are excluded from every metric (see n).   "
            + "   |   ".join(parts))

    def _fill_table(self):
        headers = ["Model", "Period", "NSE", "KGE", "RMSE", "MAE",
                   "PBIAS (%)", "r", "R²", "Rv", "n"]
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(self.last_scores))

        mono = QFont("Consolas")
        mono.setStyleHint(QFont.Monospace)
        bold = QFont()
        bold.setBold(True)

        def num_item(text):
            it = QTableWidgetItem(text)
            it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            it.setFont(mono)
            it.setForeground(QColor(INK))
            return it

        for i, s in enumerate(self.last_scores):
            name_it = QTableWidgetItem(s["model"])
            name_it.setFont(bold)
            name_it.setForeground(QColor(INK))
            self.table.setItem(i, 0, name_it)

            per_it = QTableWidgetItem(s["period"])
            per_it.setForeground(
                QColor(CAL_SHADE if s["period"] == "Calibration"
                       else VAL_SHADE if s["period"] == "Validation"
                       else INK_SOFT))
            per_it.setFont(bold)
            self.table.setItem(i, 1, per_it)

            self.table.setItem(i, 2, num_item(f"{s['nse']:.3f}"))
            self.table.setItem(i, 3, num_item(f"{s['kge']:.3f}"))
            self.table.setItem(i, 4, num_item(f"{s['rmse']:.3f}"))
            self.table.setItem(i, 5, num_item(f"{s['mae']:.3f}"))
            self.table.setItem(i, 6, num_item(f"{s['pbias']:.2f}"))
            self.table.setItem(i, 7, num_item(f"{s['r']:.3f}"))
            self.table.setItem(i, 8, num_item(f"{s['r2']:.3f}"))
            rv = s.get("rv")
            self.table.setItem(i, 9, num_item(f"{rv:.3f}" if rv is not None else "—"))
            self.table.setItem(i, 10, num_item(str(s["n"])))

        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.Stretch)
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeToContents)

    # ----------------------------------------------------------------- export
    def export_scores(self):
        if not self.last_scores:
            QMessageBox.information(self, "Nothing to export",
                                    "No visible model series with valid scores.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export scores", "calibration_scores.csv", "CSV files (*.csv)")
        if not path:
            return

        rows = [{
            "Model": s["model"],
            "Period": s["period"],
            "NSE": round(s["nse"], 4),
            "KGE": round(s["kge"], 4),
            "RMSE": round(s["rmse"], 4),
            "MAE": round(s["mae"], 4),
            "PBIAS_%": round(s["pbias"], 3),
            "r": round(s["r"], 4),
            "R2": round(s["r2"], 4),
            "Rv": round(s["rv"], 4) if s.get("rv") is not None else "",
            "n": s["n"],
        } for s in self.last_scores]
        pd.DataFrame(rows).to_csv(path, index=False)
        QMessageBox.information(self, "Export complete",
                                f"Scores written to:\n{path}")

    def export_individual_plots(self):
        """One PNG per visible model, using the same styling as the GUI."""
        if self.df is None:
            return
        visible = [(n, self.series_rows[n]) for n in self.order
                   if self.series_rows[n].visible]
        if not visible:
            QMessageBox.information(self, "Nothing to export",
                                    "No visible model series to plot.")
            return

        folder = QFileDialog.getExistingDirectory(
            self, "Choose folder for exported graphs")
        if not folder:
            return

        base_title = self.title_edit.text() or "Hydrograph"
        written = []
        for name, row in visible:
            fig = Figure(figsize=(11, 5.2), tight_layout=True)
            fig.patch.set_facecolor("white")
            ax = fig.add_subplot(111)
            # don't repeat the model name if the chart title already carries it
            title = (base_title if row.label.lower() in base_title.lower()
                     else f"{row.label} — {base_title}")
            self._draw_chart(fig, ax, [(name, row)], title)

            safe = re.sub(r"[^\w\-.]+", "_", row.label).strip("_") or "model"
            out_path = os.path.join(folder, f"{safe}.png")
            fig.savefig(out_path, dpi=200, facecolor="white",
                        bbox_inches="tight")
            written.append(out_path)

        QMessageBox.information(
            self, "Export complete",
            f"{len(written)} graph(s) written to:\n{folder}")

    # --------------------------------------------------------------- settings
    def _collect_settings(self) -> dict:
        return {
            "version": 2,
            "observed": {
                "column": self.obs_cmb.currentText(),
                "label": self.obs_label_edit.text(),
                "color": self.obs_color,
                "style": self.obs_style_cmb.currentText(),
                "behind": self.obs_back_chk.isChecked(),
            },
            "precipitation": {
                "enabled": self.precip_chk.isChecked(),
                "color": self.precip_color,
                "name": self.precip_name_edit.text(),
                "axis_label": self.precip_label_edit.text(),
                "aggregation": self.precip_agg_cmb.currentText(),
            },
            "models": {
                name: {
                    "label": row.label_edit.text(),
                    "color": row.color,
                    "style": row.style_cmb.currentText(),
                    "visible": row.visible,
                    "order": self.order.index(name),
                } for name, row in self.series_rows.items()
            },
            "axes": {
                "flow_min": self.flow_min_edit.text(),
                "flow_max": self.flow_max_edit.text(),
                "precip_max": self.precip_max_edit.text(),
                "band": self.band_spin.value(),
                "log": self.log_chk.isChecked(),
            },
            "legend": {
                "show": self.legend_chk.isChecked(),
                "frame": self.legend_frame_chk.isChecked(),
                "loc": self.legend_loc_cmb.currentText(),
                "order": self.legend_order_cmb.currentText(),
                "cal_label": self.cal_label_edit.text(),
                "val_label": self.val_label_edit.text(),
            },
            "periods": {
                "cal_start": self.cal_start.text(),
                "cal_end": self.cal_end.text(),
                "val_start": self.val_start.text(),
                "val_end": self.val_end.text(),
                "val_enabled": self.val_chk.isChecked(),
            },
            "labels": {"y": self.y_label_edit.text()},
        }

    def _apply_settings(self, data: dict, restore_periods: bool = True):
        self._loading = True
        try:
            obs = data.get("observed", {})
            if obs.get("color"):
                self.obs_color = obs["color"]
                self._paint_obs_color_btn()
            if obs.get("style") in LINE_STYLES:
                self.obs_style_cmb.setCurrentText(obs["style"])
            if "behind" in obs:
                self.obs_back_chk.setChecked(bool(obs["behind"]))
            if obs.get("column") == self.obs_cmb.currentText() and obs.get("label"):
                self.obs_label_edit.setText(obs["label"])

            p = data.get("precipitation", {})
            if "enabled" in p:
                self.precip_chk.setChecked(bool(p["enabled"]))
            if p.get("color"):
                self.precip_color = p["color"]
                self._paint_precip_color_btn()
            if p.get("name"):
                self.precip_name_edit.setText(p["name"])
            if p.get("axis_label"):
                self.precip_label_edit.setText(p["axis_label"])
            if p.get("aggregation") in PRECIP_AGG:
                self.precip_agg_cmb.setCurrentText(p["aggregation"])

            a = data.get("axes", {})
            self.flow_min_edit.setText(a.get("flow_min", ""))
            self.flow_max_edit.setText(a.get("flow_max", ""))
            self.precip_max_edit.setText(a.get("precip_max", ""))
            if a.get("band"):
                self.band_spin.setValue(int(a["band"]))
            self.log_chk.setChecked(bool(a.get("log", False)))

            lg = data.get("legend", {})
            if "show" in lg:
                self.legend_chk.setChecked(bool(lg["show"]))
            if "frame" in lg:
                self.legend_frame_chk.setChecked(bool(lg["frame"]))
            if lg.get("loc") in LEGEND_LOCS:
                self.legend_loc_cmb.setCurrentText(lg["loc"])
            if lg.get("order") in LEGEND_ORDERS:
                self.legend_order_cmb.setCurrentText(lg["order"])
            if lg.get("cal_label"):
                self.cal_label_edit.setText(lg["cal_label"])
            if lg.get("val_label"):
                self.val_label_edit.setText(lg["val_label"])

            if data.get("labels", {}).get("y"):
                self.y_label_edit.setText(data["labels"]["y"])

            models = data.get("models", {})
            for name, row in self.series_rows.items():
                cfg = models.get(name)
                if not cfg:
                    continue
                if cfg.get("color"):
                    row.color = cfg["color"]
                    row._paint_color_btn()
                if cfg.get("style") in LINE_STYLES:
                    row.style_cmb.setCurrentText(cfg["style"])
                if cfg.get("label"):
                    row.label_edit.setText(cfg["label"])
                row.chk.setChecked(bool(cfg.get("visible", True)))
            if models:
                self.order.sort(
                    key=lambda nm: models.get(nm, {}).get("order", 10_000))
                self._relayout_series()

            if restore_periods:
                pr = data.get("periods", {})
                if pr:
                    self.cal_start.setText(pr.get("cal_start", self.cal_start.text()))
                    self.cal_end.setText(pr.get("cal_end", self.cal_end.text()))
                    self.val_start.setText(pr.get("val_start", self.val_start.text()))
                    self.val_end.setText(pr.get("val_end", self.val_end.text()))
                    self.val_chk.setChecked(bool(pr.get("val_enabled", True)))
                    # if the saved dates don't overlap this file, go back to 70/30
                    m = self._period_mask(self.cal_start.text(), self.cal_end.text())
                    if m is None or not m.any():
                        self._init_periods()
        finally:
            self._loading = False

    @staticmethod
    def _read_settings_file(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:  # noqa: BLE001
            return None

    def _autosave(self):
        try:
            with open(AUTOSAVE_PATH, "w", encoding="utf-8") as fh:
                json.dump(self._collect_settings(), fh, indent=2)
        except Exception:  # noqa: BLE001
            pass

    def save_settings_dialog(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save settings", "calibration_studio_settings.json",
            "JSON files (*.json)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(self._collect_settings(), fh, indent=2)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Save failed", str(e))
            return
        QMessageBox.information(self, "Settings saved", f"Written to:\n{path}")

    def load_settings_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load settings", "", "JSON files (*.json)")
        if not path:
            return
        data = self._read_settings_file(path)
        if data is None:
            QMessageBox.critical(self, "Load failed",
                                 "That file could not be read as settings JSON.")
            return
        self._apply_settings(data, restore_periods=True)
        self.refresh()

    def closeEvent(self, event):
        if self.df is not None:
            self._autosave()
        super().closeEvent(event)


# -----------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLESHEET)
    win = CalibrationStudio()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
