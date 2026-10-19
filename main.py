import sys
import os
import json
import math
import queue
import numpy as np
import sounddevice as sd
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QLabel, QGroupBox, QMessageBox,
    QPushButton, QSlider, QSpinBox, QDialog, QColorDialog,
    QGridLayout, QDialogButtonBox, QMenu, QCheckBox, QFrame
)
from PyQt6.QtCore import QTimer, Qt, QRectF, QPoint, QPointF, pyqtSignal
from PyQt6.QtGui import (
    QColor, QCursor, QPainter, QPainterPath, QPen, QBrush,
    QLinearGradient, QPolygonF, QFont, QPixmap, QIcon
)
import pyqtgraph as pg

# Activar aceleración por hardware GPU (OpenGL) para descargar la CPU (Intel UHD Graphics / GPU dedicada)
try:
    import OpenGL  # noqa: F401
    pg.setConfigOptions(useOpenGL=True, enableExperimental=True, antialias=True)
except Exception:
    pg.setConfigOptions(antialias=True)

from fft_analyzer import FFTSpectrumAnalyzer
from eq_dsp import ParametricEQEngine
from eq_ui import ProQ4Overlay, FloatingBandHUD, LiquidGlassButton


DEFAULT_PRESET_NAME = "Gris Puro de Estudio (Default)"
THEME_PRESETS = {
    "Gris Puro de Estudio (Default)": {
        "background": "#141414", "curve": "#B8BDC6", "curve_width": 15,
        "fill": "#6E747E", "fill_alpha": 40,
        "grid": "#2A2A2A", "grid_alpha": 45, "axis_text": "#8C919B",
    },
    "Carbón de Estudio": {
        "background": "#1E1E1E", "curve": "#C5CAD3", "curve_width": 16,
        "fill": "#767C87", "fill_alpha": 42,
        "grid": "#333333", "grid_alpha": 45, "axis_text": "#989EA8",
    },
    "Negro Absoluto": {
        "background": "#080808", "curve": "#00E5FF", "curve_width": 17,
        "fill": "#00ACC1", "fill_alpha": 42,
        "grid": "#222222", "grid_alpha": 45, "axis_text": "#858A94",
    },
    "Azul Estudio": {
        "background": "#0D1017", "curve": "#A0AAB8", "curve_width": 15,
        "fill": "#606B7D", "fill_alpha": 38,
        "grid": "#2D3340", "grid_alpha": 30, "axis_text": "#7A8496",
    },
    "Ámbar Analógico": {
        "background": "#140C02", "curve": "#FFB300", "curve_width": 20,
        "fill": "#FF8F00", "fill_alpha": 50,
        "grid": "#8A6A3A", "grid_alpha": 35, "axis_text": "#D9B98A",
    },
}

BACKGROUND_COLOR_PRESETS = [
    ("Gris Puro #141414", "#141414"),
    ("Carbón #1E1E1E", "#1E1E1E"),
    ("Negro Absoluto #080808", "#080808"),
    ("Azul Estudio #0D1017", "#0D1017"),
]

THEME_NUMERIC_RANGES = {"curve_width": (10, 60), "fill_alpha": (0, 255), "grid_alpha": (5, 100)}
THEME_COLOR_KEYS = ("background", "curve", "fill", "grid", "axis_text")
CONFIG_FILENAME = "config.json"
CURRENT_CONFIG_VERSION = 4


def _make_swatch_icon(hex_color: str, size: int = 12) -> QIcon:
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setPen(QPen(QColor("#888888"), 1.0))
    p.setBrush(QBrush(QColor(hex_color)))
    p.drawRoundedRect(QRectF(1.0, 1.0, size - 2.0, size - 2.0), 2.0, 2.0)
    p.end()
    return QIcon(pix)


def _app_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def _config_candidates():
    paths = [os.path.join(_app_dir(), CONFIG_FILENAME)]
    appdata = os.environ.get("APPDATA")
    if appdata:
        paths.append(os.path.join(appdata, "AnalizadorFFT", CONFIG_FILENAME))
    return paths


def load_full_config() -> dict:
    theme = dict(THEME_PRESETS[DEFAULT_PRESET_NAME])
    raw_data = {}
    for path in _config_candidates():
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                raw_data = loaded
                break
        except (OSError, ValueError):
            continue

    saved_version = int(raw_data.get("version", 1))
    saved_theme = raw_data.get("theme", {})
    if isinstance(saved_theme, dict) and saved_version >= CURRENT_CONFIG_VERSION:
        for key in THEME_COLOR_KEYS:
            value = saved_theme.get(key)
            if isinstance(value, str) and QColor(value).isValid():
                theme[key] = QColor(value).name()
        for key, (vmin, vmax) in THEME_NUMERIC_RANGES.items():
            value = saved_theme.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                theme[key] = int(min(max(value, vmin), vmax))

    raw_data["theme"] = theme
    return raw_data


def save_full_config(config_payload: dict) -> bool:
    for path in _config_candidates():
        try:
            data = {}
            if os.path.isfile(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                    if isinstance(loaded, dict):
                        data = loaded
                except (OSError, ValueError):
                    pass
            data.update(config_payload)
            data["version"] = CURRENT_CONFIG_VERSION
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp_path = path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, path)
            return True
        except OSError:
            continue
    return False


class WaveformEnvelopeBuffer:
    """
    Buffer circular ultraliviano de Envolvente Min/Max (Downsampling en Entrada).
    - Por cada bloque de audio entrante (512 muestras, ~10.6 ms), almacena únicamente
      (min_l, max_l, min_r, max_r, min_m, max_m).
    - Lleva un contador monotónico `total_pushed` para alimentar el Scroll Blitting incremental
      (`QPixmap.scroll(-dx, 0)`) tanto en la vista compacta como en el panel desplegado.
    """

    def __init__(self, sample_rate: int = 48000, block_size: int = 512):
        self.sample_rate = int(sample_rate)
        self.block_size = int(block_size)
        self.duration_sec = 10  # 5 o 10 segundos visibles
        self.total_pushed = 0
        self._allocate_buffers()

    def _allocate_buffers(self):
        blocks_per_sec = max(10.0, self.sample_rate / float(self.block_size))
        self.capacity_10s = int(math.ceil(10.0 * blocks_per_sec))
        self.capacity_5s = max(10, self.capacity_10s // 2)

        self.min_l = np.zeros(self.capacity_10s, dtype=np.float32)
        self.max_l = np.zeros(self.capacity_10s, dtype=np.float32)
        self.min_r = np.zeros(self.capacity_10s, dtype=np.float32)
        self.max_r = np.zeros(self.capacity_10s, dtype=np.float32)
        self.min_m = np.zeros(self.capacity_10s, dtype=np.float32)
        self.max_m = np.zeros(self.capacity_10s, dtype=np.float32)

        self._silent_blocks = self.capacity_10s + 1
        self.is_at_rest = True
        self.total_pushed = 0

    def set_sample_rate(self, sample_rate: int):
        if int(sample_rate) != self.sample_rate:
            self.sample_rate = int(sample_rate)
            self._allocate_buffers()

    def current_capacity(self) -> int:
        return self.capacity_5s if self.duration_sec == 5 else self.capacity_10s

    def push_blocks(self, blocks: list) -> bool:
        """
        Procesa bloques entrantes extrayendo únicamente (min, max) por bloque.
        Retorna True si el historial visible cambió y requiere repintado.
        """
        if not blocks:
            return False

        any_active = False
        for blk in blocks:
            if blk.ndim > 1 and blk.shape[1] > 1:
                ch_l = blk[:, 0]
                ch_r = blk[:, 1]
                mono = (ch_l + ch_r) * 0.5
                l_min, l_max = float(np.min(ch_l)), float(np.max(ch_l))
                r_min, r_max = float(np.min(ch_r)), float(np.max(ch_r))
                m_min, m_max = float(np.min(mono)), float(np.max(mono))
            else:
                ch = blk[:, 0] if blk.ndim > 1 else blk
                l_min = r_min = m_min = float(np.min(ch))
                l_max = r_max = m_max = float(np.max(ch))

            peak = max(abs(l_min), abs(l_max), abs(r_min), abs(r_max))
            if peak < 1e-4:
                self._silent_blocks += 1
            else:
                self._silent_blocks = 0
                any_active = True

            if self._silent_blocks > self.capacity_10s and not any_active:
                if not self.is_at_rest:
                    self.min_l.fill(0.0)
                    self.max_l.fill(0.0)
                    self.min_r.fill(0.0)
                    self.max_r.fill(0.0)
                    self.min_m.fill(0.0)
                    self.max_m.fill(0.0)
                    self.is_at_rest = True
                    self.total_pushed += 1
                    return True
                continue

            self.is_at_rest = False
            self.min_l[:-1] = self.min_l[1:]
            self.max_l[:-1] = self.max_l[1:]
            self.min_r[:-1] = self.min_r[1:]
            self.max_r[:-1] = self.max_r[1:]
            self.min_m[:-1] = self.min_m[1:]
            self.max_m[:-1] = self.max_m[1:]

            self.min_l[-1] = np.clip(l_min, -1.0, 1.0)
            self.max_l[-1] = np.clip(l_max, -1.0, 1.0)
            self.min_r[-1] = np.clip(r_min, -1.0, 1.0)
            self.max_r[-1] = np.clip(r_max, -1.0, 1.0)
            self.min_m[-1] = np.clip(m_min, -1.0, 1.0)
            self.max_m[-1] = np.clip(m_max, -1.0, 1.0)
            self.total_pushed += 1

        return not self.is_at_rest

    def get_visible_slice(self):
        count = self.current_capacity()
        return (
            self.min_l[-count:],
            self.max_l[-count:],
            self.min_r[-count:],
            self.max_r[-count:],
            self.min_m[-count:],
            self.max_m[-count:],
        )


class _CompactScopeCanvas(QWidget):
    """
    Lienzo interno del mini-osciloscopio en la barra superior usando Scroll Blitting (`QPixmap.scroll`):
    - Mantiene `self.wave_canvas` (`QPixmap`) en memoria.
    - En cada frame desplaza con `self.wave_canvas.scroll(-dx, 0, self.wave_canvas.rect())` y pinta
      únicamente la franja derecha nueva de ancho `dx`.
    """

    clicked = pyqtSignal()

    def __init__(self, env_buffer: WaveformEnvelopeBuffer, parent=None):
        super().__init__(parent)
        self.env_buffer = env_buffer
        self.wave_canvas: QPixmap | None = None
        self._last_pushed_seq: int = -1
        self._px_accum: float = 0.0
        self._cached_dur: int = -1
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Clic para desplegar / plegar el Osciloscopio detallado")

    def invalidate_canvas(self):
        self.wave_canvas = None
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.wave_canvas = None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        event.accept()

    def _seed_full_canvas(self):
        w = max(10, self.width())
        h = max(10, self.height())
        self._cached_dur = self.env_buffer.duration_sec
        self._last_pushed_seq = self.env_buffer.total_pushed
        self._px_accum = 0.0

        pix = QPixmap(w, h)
        pix.fill(QColor("#121212"))
        p = QPainter(pix)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        mid_y = h // 2
        p.setPen(QPen(QColor(255, 255, 255, 35), 1))
        p.drawLine(0, mid_y, w - 1, mid_y)

        _, _, _, _, min_m, max_m = self.env_buffer.get_visible_slice()
        n_raw = len(min_m)
        if n_raw >= 2 and not self.env_buffer.is_at_rest:
            amp = (h * 0.5) - 2.0
            n_pts = min(180, max(16, w - 4), n_raw)
            idx = np.linspace(0, n_raw - 1, n_pts, dtype=np.int32)
            xs = np.linspace(1.0, w - 2.0, n_pts)
            y_top = mid_y - max_m[idx] * amp
            y_bot = mid_y - min_m[idx] * amp

            poly = QPolygonF()
            for i in range(n_pts):
                xv = float(xs[i])
                poly.append(QPointF(xv, float(y_top[i])))
                poly.append(QPointF(xv, float(y_bot[i])))
            p.setPen(QPen(QColor("#00E5FF"), 1))
            p.drawPolyline(poly)

        p.end()
        self.wave_canvas = pix

    def _advance_scroll_canvas(self):
        if (
            self.wave_canvas is None
            or self.wave_canvas.size() != self.size()
            or self._cached_dur != self.env_buffer.duration_sec
        ):
            self._seed_full_canvas()
            return

        new_blocks = self.env_buffer.total_pushed - self._last_pushed_seq
        if new_blocks <= 0:
            return
        self._last_pushed_seq = self.env_buffer.total_pushed

        if self.env_buffer.is_at_rest:
            self._seed_full_canvas()
            return

        w = self.wave_canvas.width()
        h = self.wave_canvas.height()
        cap = max(1, self.env_buffer.current_capacity())
        if new_blocks >= cap // 2:
            self._seed_full_canvas()
            return

        exact_px = (new_blocks / float(cap)) * w + self._px_accum
        dx = int(exact_px)
        if dx < 1:
            self._px_accum = exact_px
            return
        self._px_accum = exact_px - dx
        dx = min(dx, w - 2)

        # Scroll Blitting hacia la izquierda en `dx` píxeles
        self.wave_canvas.scroll(-dx, 0, self.wave_canvas.rect())

        p = QPainter(self.wave_canvas)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        x0 = w - dx
        p.fillRect(x0, 0, dx, h, QColor("#121212"))
        mid_y = h // 2
        p.setPen(QPen(QColor(255, 255, 255, 35), 1))
        p.drawLine(x0, mid_y, w - 1, mid_y)

        n_slice = min(len(self.env_buffer.min_m), new_blocks + 1)
        min_seg = self.env_buffer.min_m[-n_slice:]
        max_seg = self.env_buffer.max_m[-n_slice:]
        amp = (h * 0.5) - 2.0

        poly = QPolygonF()
        for col in range(dx):
            frac = col / float(max(1, dx - 1)) if dx > 1 else 1.0
            si = int(round(frac * (len(min_seg) - 1)))
            xv = float(x0 + col)
            yt = float(mid_y - max_seg[si] * amp)
            yb = float(mid_y - min_seg[si] * amp)
            poly.append(QPointF(xv, yt))
            poly.append(QPointF(xv, yb))

        p.setPen(QPen(QColor("#00E5FF"), 1))
        p.drawPolyline(poly)
        p.end()

    def paintEvent(self, event):
        self._advance_scroll_canvas()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.drawPixmap(0, 0, self.wave_canvas)
        painter.setPen(QPen(QColor("#2E2E2E"), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(0, 0, self.width() - 1, self.height() - 1)


class WaveformScopeWidget(QWidget):
    """
    Vista Compacta Integrada en la Top Bar Flotante con estética Apple Liquid Glass.
    """

    toggleExpandedRequested = pyqtSignal()
    durationChanged = pyqtSignal(int)

    def __init__(self, env_buffer: WaveformEnvelopeBuffer, parent=None):
        super().__init__(parent)
        self.env_buffer = env_buffer
        self.setFixedSize(338, 30)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        row = QHBoxLayout(self)
        row.setContentsMargins(7, 3, 6, 3)
        row.setSpacing(4)

        self.canvas = _CompactScopeCanvas(self.env_buffer, self)
        self.canvas.setFixedSize(256, 22)
        self.canvas.clicked.connect(self.toggleExpandedRequested.emit)
        row.addWidget(self.canvas)

        self.dur_btn = LiquidGlassButton("10s", self, pill_radius=10.0)
        self.dur_btn.setFixedSize(36, 22)
        self.dur_btn.setToolTip("Alternar historial del osciloscopio entre 5s y 10s")
        self.dur_btn.clicked.connect(self._toggle_duration)
        row.addWidget(self.dur_btn)

        self.expand_btn = LiquidGlassButton("▾", self, active_tint_color="#00E5FF", pill_radius=10.0)
        self.expand_btn.setFixedSize(24, 22)
        self.expand_btn.setToolTip("Desplegar / Plegar Osciloscopio detallado")
        self.expand_btn.clicked.connect(self.toggleExpandedRequested.emit)
        row.addWidget(self.expand_btn)

    def _toggle_duration(self):
        new_dur = 5 if self.env_buffer.duration_sec == 10 else 10
        self.set_duration(new_dur)
        self.durationChanged.emit(new_dur)

    def set_duration(self, sec: int):
        self.env_buffer.duration_sec = 5 if int(sec) == 5 else 10
        self.dur_btn.setText(f"{self.env_buffer.duration_sec}s")
        self.canvas.invalidate_canvas()

    def set_expanded_indicator(self, expanded: bool):
        self.expand_btn.setText("▴" if expanded else "▾")
        self.expand_btn.setChecked(expanded)

    def refresh_scope(self):
        self.canvas.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(0.75, 0.75, self.width() - 1.5, self.height() - 1.5)

        fill_grad = QLinearGradient(0.0, rect.top(), 0.0, rect.bottom())
        fill_grad.setColorAt(0.0, QColor(58, 62, 72, 175))
        fill_grad.setColorAt(0.45, QColor(34, 36, 42, 165))
        fill_grad.setColorAt(1.0, QColor(24, 26, 30, 185))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(fill_grad))
        painter.drawRoundedRect(rect, 14.0, 14.0)

        spec_grad = QLinearGradient(rect.left(), rect.top(), rect.right(), rect.bottom())
        spec_grad.setColorAt(0.0, QColor(255, 255, 255, 115))
        spec_grad.setColorAt(0.5, QColor(255, 255, 255, 32))
        spec_grad.setColorAt(1.0, QColor(255, 255, 255, 16))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QBrush(spec_grad), 1.2))
        painter.drawRoundedRect(rect, 14.0, 14.0)


class WaveformOverlayWidget(QWidget):
    """
    Panel desplegable del Osciloscopio con Aislamiento OpenGL y Scroll Blitting (`QPixmap.scroll`):
    1. Aislamiento del Widget: `self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)`
       e insertado directamente en el `QVBoxLayout` principal (desplazando el gráfico FFT hacia abajo)
       para no superponerse jamás con transparencia sobre el lienzo `OpenGL`.
    2. Scroll Blitting (`self.wave_canvas.scroll(-dx, 0, self.wave_canvas.rect())`):
       En lugar de redibujar 10 segundos de onda en cada cuadro, desplaza `self.wave_canvas` hacia la
       izquierda `dx` píxeles y pinta únicamente la franja vertical nueva de ancho `dx` en el extremo
       derecho con los valores `min/max` recién llegados, volcando en `paintEvent` con
       `painter.drawPixmap(0, 0, self.wave_canvas)` (< 1% de uso de CPU).
    """

    collapseRequested = pyqtSignal()

    def __init__(self, env_buffer: WaveformEnvelopeBuffer, parent=None):
        super().__init__(parent)
        self.env_buffer = env_buffer
        self.wave_canvas: QPixmap | None = None
        self._hud_overlay_cache: QPixmap | None = None
        self._last_pushed_seq: int = -1
        self._px_accum: float = 0.0
        self._cached_dur: int = -1

        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setFixedHeight(168)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Clic para plegar el osciloscopio de vuelta a la barra superior")
        self.hide()

    def invalidate_cache(self):
        self.wave_canvas = None
        self._hud_overlay_cache = None
        if self.isVisible():
            self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.wave_canvas = None
        self._hud_overlay_cache = None

    def showEvent(self, event):
        super().showEvent(event)
        self.wave_canvas = None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.collapseRequested.emit()
        event.accept()

    def wheelEvent(self, event):
        event.accept()

    def _draw_strip_bg(self, p: QPainter, x0: int, strip_w: int, h: int):
        p.fillRect(x0, 0, strip_w, h, QColor("#151515"))
        pad_t = 22
        pad_b = 20
        plot_h = max(10, h - pad_t - pad_b)
        mid_y = pad_t + plot_h // 2
        half_h = plot_h * 0.5

        p.setPen(QPen(QColor("#252525"), 1))
        for level in (-0.5, 0.5):
            y_g = int(round(mid_y - level * half_h))
            p.drawLine(x0, y_g, x0 + strip_w, y_g)

        p.setPen(QPen(QColor("#444444"), 1))
        p.drawLine(x0, mid_y, x0 + strip_w, mid_y)

    def _rebuild_hud_overlay_cache(self):
        w = max(40, self.width())
        h = max(40, self.height())
        dur = self.env_buffer.duration_sec

        pix = QPixmap(w, h)
        pix.fill(Qt.GlobalColor.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        pad_l = 38
        pad_r = 38
        pad_t = 22
        pad_b = 20
        plot_w = max(10, w - pad_l - pad_r)
        plot_h = max(10, h - pad_t - pad_b)

        # Máscaras laterales opacas y barras superior/inferior de leyenda
        p.fillRect(0, 0, pad_l, h, QColor("#151515"))
        p.fillRect(w - pad_r, 0, pad_r, h, QColor("#151515"))
        p.fillRect(0, 0, w, pad_t, QColor("#151515"))
        p.fillRect(0, h - pad_b, w, pad_b, QColor("#151515"))

        # Borde exterior y marco del área de onda
        p.setPen(QPen(QColor("#2E2E2E"), 1))
        p.drawRect(0, 0, w - 1, h - 1)
        p.drawRect(pad_l, pad_t, plot_w, plot_h)

        # Grilla temporal en segundos (-10s ... -5s ... 0s)
        font_grid = QFont("Segoe UI", 8)
        p.setFont(font_grid)
        for s in range(0, dur + 1):
            frac = 1.0 - (s / float(dur))
            x_s = int(round(pad_l + frac * plot_w))
            p.setPen(QPen(QColor(255, 255, 255, 22), 1))
            p.drawLine(x_s, pad_t, x_s, pad_t + plot_h)

            lbl = "0s" if s == 0 else f"-{s}s"
            p.setPen(QColor("#8E95A4"))
            p.drawText(QRectF(x_s - 18.0, h - 18.0, 36.0, 15.0), Qt.AlignmentFlag.AlignCenter, lbl)

        # Cabecera del panel (Leyenda L / R y aviso de plegado)
        font_hdr = QFont("Segoe UI", 8, QFont.Weight.Bold)
        p.setFont(font_hdr)
        p.setPen(QColor("#00E5FF"))
        p.drawText(QRectF(pad_l, 4.0, 130.0, 15.0), Qt.AlignmentFlag.AlignLeft, "● Canal L (Post-EQ)")
        p.setPen(QColor("#FFB300"))
        p.drawText(QRectF(pad_l + 135.0, 4.0, 130.0, 15.0), Qt.AlignmentFlag.AlignLeft, "● Canal R (Post-EQ)")
        p.setPen(QColor("#9AA0A6"))
        p.drawText(
            QRectF(w - pad_r - 230.0, 4.0, 230.0, 15.0),
            Qt.AlignmentFlag.AlignRight,
            f"Historial {dur}s  ▴ Clic para plegar",
        )
        p.end()
        self._hud_overlay_cache = pix

    def _seed_full_wave_canvas(self):
        w = max(40, self.width())
        h = max(40, self.height())
        self._cached_dur = self.env_buffer.duration_sec
        self._last_pushed_seq = self.env_buffer.total_pushed
        self._px_accum = 0.0

        self._rebuild_hud_overlay_cache()

        pix = QPixmap(w, h)
        p = QPainter(pix)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        self._draw_strip_bg(p, 0, w, h)

        min_l, max_l, min_r, max_r, _, _ = self.env_buffer.get_visible_slice()
        n_raw = len(min_l)
        if n_raw >= 2 and not self.env_buffer.is_at_rest:
            pad_l = 38.0
            pad_r = 38.0
            pad_t = 22.0
            pad_b = 20.0
            plot_w = max(10.0, w - pad_l - pad_r)
            plot_h = max(10.0, h - pad_t - pad_b)
            mid_y = pad_t + plot_h * 0.5
            half_h = plot_h * 0.5

            n_pts = min(360, max(32, int(plot_w) // 2), n_raw)
            idx = np.linspace(0, n_raw - 1, n_pts, dtype=np.int32)
            xs = np.linspace(pad_l, pad_l + plot_w, n_pts)

            yr_top = mid_y - max_r[idx] * half_h
            yr_bot = mid_y - min_r[idx] * half_h
            yl_top = mid_y - max_l[idx] * half_h
            yl_bot = mid_y - min_l[idx] * half_h

            poly_r = QPolygonF()
            poly_l = QPolygonF()
            for i in range(n_pts):
                xv = float(xs[i])
                poly_r.append(QPointF(xv, float(yr_top[i])))
                poly_r.append(QPointF(xv, float(yr_bot[i])))
                poly_l.append(QPointF(xv, float(yl_top[i])))
                poly_l.append(QPointF(xv, float(yl_bot[i])))

            p.setPen(QPen(QColor("#FFB300"), 1))
            p.drawPolyline(poly_r)
            p.setPen(QPen(QColor("#00E5FF"), 1))
            p.drawPolyline(poly_l)

        p.end()
        self.wave_canvas = pix

    def _advance_wave_canvas(self):
        if (
            self.wave_canvas is None
            or self._hud_overlay_cache is None
            or self.wave_canvas.size() != self.size()
            or self._cached_dur != self.env_buffer.duration_sec
        ):
            self._seed_full_wave_canvas()
            return

        new_blocks = self.env_buffer.total_pushed - self._last_pushed_seq
        if new_blocks <= 0:
            return
        self._last_pushed_seq = self.env_buffer.total_pushed

        if self.env_buffer.is_at_rest:
            self._seed_full_wave_canvas()
            return

        w = self.wave_canvas.width()
        h = self.wave_canvas.height()
        pad_l = 38
        pad_r = 38
        pad_t = 22
        pad_b = 20
        plot_w = max(10, w - pad_l - pad_r)
        plot_h = max(10, h - pad_t - pad_b)
        mid_y = pad_t + plot_h * 0.5
        half_h = plot_h * 0.5

        cap = max(1, self.env_buffer.current_capacity())
        if new_blocks >= cap // 2:
            self._seed_full_wave_canvas()
            return

        exact_px = (new_blocks / float(cap)) * plot_w + self._px_accum
        dx = int(exact_px)
        if dx < 1:
            self._px_accum = exact_px
            return
        self._px_accum = exact_px - dx
        dx = min(dx, plot_w - 2)

        # Scroll Blitting: desplaza `self.wave_canvas` hacia la izquierda `dx` píxeles
        self.wave_canvas.scroll(-dx, 0, self.wave_canvas.rect())

        # Pintar únicamente la franja vertical nueva de ancho `dx` en el extremo derecho del plot
        x_right_edge = pad_l + plot_w
        x0 = x_right_edge - dx

        p = QPainter(self.wave_canvas)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        self._draw_strip_bg(p, x0, dx + pad_r, h)

        n_slice = min(len(self.env_buffer.min_l), new_blocks + 1)
        min_l_seg = self.env_buffer.min_l[-n_slice:]
        max_l_seg = self.env_buffer.max_l[-n_slice:]
        min_r_seg = self.env_buffer.min_r[-n_slice:]
        max_r_seg = self.env_buffer.max_r[-n_slice:]

        poly_r = QPolygonF()
        poly_l = QPolygonF()
        for col in range(dx):
            frac = col / float(max(1, dx - 1)) if dx > 1 else 1.0
            si = int(round(frac * (n_slice - 1)))
            xv = float(x0 + col)
            poly_r.append(QPointF(xv, float(mid_y - max_r_seg[si] * half_h)))
            poly_r.append(QPointF(xv, float(mid_y - min_r_seg[si] * half_h)))
            poly_l.append(QPointF(xv, float(mid_y - max_l_seg[si] * half_h)))
            poly_l.append(QPointF(xv, float(mid_y - min_l_seg[si] * half_h)))

        p.setPen(QPen(QColor("#FFB300"), 1))
        p.drawPolyline(poly_r)
        p.setPen(QPen(QColor("#00E5FF"), 1))
        p.drawPolyline(poly_l)
        p.end()

    def paintEvent(self, event):
        self._advance_wave_canvas()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        # Volcar el QPixmap con Scroll Blitting
        painter.drawPixmap(0, 0, self.wave_canvas)
        if self._hud_overlay_cache is not None:
            painter.drawPixmap(0, 0, self._hud_overlay_cache)


# Alias de compatibilidad
ExpandedWaveformOverlay = WaveformOverlayWidget



class SlimStereoMeter(QWidget):
    """
    Vúmetro estéreo L/R ultrafino estilo FabFilter Pro-Q 4.
    Solo invoca `update()` cuando los niveles cambian perceptiblemente (> 0.05 dB),
    evitando repintados innecesarios en IDLE.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(24)
        self.db_min = -60.0
        self.db_max = 0.0
        self.level_l = -60.0
        self.level_r = -60.0
        self.peak_l = -60.0
        self.peak_r = -60.0

    def set_levels(self, db_l: float, db_r: float):
        new_l = max(db_l, self.level_l * 0.78 + db_l * 0.22)
        new_r = max(db_r, self.level_r * 0.78 + db_r * 0.22)
        new_pk_l = max(new_l, max(self.db_min, self.peak_l - 0.18))
        new_pk_r = max(new_r, max(self.db_min, self.peak_r - 0.18))

        changed = (
            abs(new_l - self.level_l) > 0.05
            or abs(new_r - self.level_r) > 0.05
            or abs(new_pk_l - self.peak_l) > 0.05
            or abs(new_pk_r - self.peak_r) > 0.05
        )
        self.level_l = new_l
        self.level_r = new_r
        self.peak_l = new_pk_l
        self.peak_r = new_pk_r
        if changed:
            self.update()

    def _db_to_ratio(self, db: float) -> float:
        return float(np.clip((db - self.db_min) / (self.db_max - self.db_min), 0.0, 1.0))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        w = self.width()
        h = self.height()
        top_pad = 6
        bot_pad = 6
        bar_h = max(10, h - top_pad - bot_pad)

        bar_w = 5
        gap = 3
        x_l = (w - (2 * bar_w + gap)) // 2
        x_r = x_l + bar_w + gap

        bg_color = QColor("#101010")
        border_color = QColor("#2A2A2A")
        painter.fillRect(x_l, top_pad, bar_w, bar_h, bg_color)
        painter.fillRect(x_r, top_pad, bar_w, bar_h, bg_color)

        grad = QLinearGradient(0, top_pad + bar_h, 0, top_pad)
        grad.setColorAt(0.00, QColor("#12A84B"))
        grad.setColorAt(0.65, QColor("#4CD92B"))
        grad.setColorAt(0.86, QColor("#AEEA00"))
        grad.setColorAt(0.95, QColor("#FFD600"))
        grad.setColorAt(0.985, QColor("#FF6D00"))
        grad.setColorAt(1.00, QColor("#FF1744"))

        for x_pos, lvl, pk in ((x_l, self.level_l, self.peak_l), (x_r, self.level_r, self.peak_r)):
            ratio = self._db_to_ratio(lvl)
            fill_h = int(round(ratio * bar_h))
            if fill_h > 0:
                y_top = top_pad + (bar_h - fill_h)
                painter.fillRect(QRectF(x_pos, y_top, bar_w, fill_h), grad)

            pk_ratio = self._db_to_ratio(pk)
            if pk_ratio > 0.02:
                pk_y = top_pad + int(round((1.0 - pk_ratio) * bar_h))
                pk_color = QColor("#FF3B30") if pk > -1.0 else QColor("#FFE047")
                painter.fillRect(x_pos, pk_y, bar_w, 2, pk_color)

            painter.setPen(border_color)
            painter.drawRect(x_pos, top_pad, bar_w - 1, bar_h - 1)


class ColorSettingsDialog(QDialog):
    """Panel no modal de colores."""

    COLOR_ROWS = [
        ("background", "Fondo del gráfico"),
        ("curve", "Línea de espectro Post-EQ"),
        ("fill", "Relleno bajo el espectro"),
        ("grid", "Líneas de grilla / frecuencia"),
        ("axis_text", "Texto de los ejes"),
    ]

    def __init__(self, theme, on_change, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuración de Colores")
        self.setModal(False)
        self.theme = theme
        self.on_change = on_change
        self.swatches = {}
        self.sliders = {}

        layout = QVBoxLayout(self)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("— Personalizado —")
        self.preset_combo.addItems(list(THEME_PRESETS.keys()))
        self.preset_combo.activated.connect(self._on_preset_selected)
        preset_row.addWidget(self.preset_combo, 1)
        layout.addLayout(preset_row)

        colors_group = QGroupBox("Colores")
        colors_grid = QGridLayout(colors_group)
        for row, (key, label) in enumerate(self.COLOR_ROWS):
            colors_grid.addWidget(QLabel(label), row, 0)
            btn = QPushButton()
            btn.setFixedSize(110, 26)
            btn.clicked.connect(lambda _=False, k=key, l=label: self._pick_color(k, l))
            colors_grid.addWidget(btn, row, 1)
            self.swatches[key] = btn
        layout.addWidget(colors_group)

        style_group = QGroupBox("Grosor e Intensidad")
        style_grid = QGridLayout(style_group)
        self._add_slider(style_grid, 0, "Grosor del espectro", "curve_width", 10, 60,
                         lambda v: f"{v / 10:.1f} px")
        self._add_slider(style_grid, 1, "Opacidad del relleno", "fill_alpha", 0, 255,
                         lambda v: f"{round(v * 100 / 255)}%")
        self._add_slider(style_grid, 2, "Intensidad de la grilla", "grid_alpha", 5, 100,
                         lambda v: f"{v}%")
        layout.addWidget(style_group)

        buttons = QDialogButtonBox()
        reset_btn = buttons.addButton("Restaurar Gris Neutro", QDialogButtonBox.ButtonRole.ResetRole)
        close_btn = buttons.addButton("Cerrar", QDialogButtonBox.ButtonRole.AcceptRole)
        reset_btn.clicked.connect(lambda: self._apply_preset(DEFAULT_PRESET_NAME))
        close_btn.clicked.connect(self.close)
        layout.addWidget(buttons)

        self._refresh_widgets()
        self._sync_preset_combo()

    def _add_slider(self, grid, row, label, key, vmin, vmax, fmt):
        grid.addWidget(QLabel(label), row, 0)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(vmin, vmax)
        slider.setMinimumWidth(160)
        val_label = QLabel()
        val_label.setFixedWidth(48)
        slider.valueChanged.connect(lambda v, k=key: self._on_slider_changed(k, v))
        grid.addWidget(slider, row, 1)
        grid.addWidget(val_label, row, 2)
        self.sliders[key] = (slider, val_label, fmt)

    def _on_slider_changed(self, key, value):
        self.theme[key] = value
        slider, val_label, fmt = self.sliders[key]
        val_label.setText(fmt(value))
        self._sync_preset_combo()
        self.on_change()

    def _pick_color(self, key, label):
        original = self.theme[key]
        dlg = QColorDialog(QColor(original), self)
        dlg.setWindowTitle(f"Elegir color: {label}")
        dlg.setOption(QColorDialog.ColorDialogOption.DontUseNativeDialog, True)
        dlg.currentColorChanged.connect(lambda c, k=key: self._set_color(k, c.name()))
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._set_color(key, dlg.selectedColor().name())
        else:
            self._set_color(key, original)

    def _set_color(self, key, hex_color):
        self.theme[key] = hex_color
        self._style_swatch(key)
        self._sync_preset_combo()
        self.on_change()

    def _style_swatch(self, key):
        color = QColor(self.theme[key])
        luminance = 0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()
        text_color = "#000000" if luminance > 140 else "#FFFFFF"
        btn = self.swatches[key]
        btn.setText(color.name().upper())
        btn.setStyleSheet(
            f"QPushButton {{ background-color: {color.name()}; color: {text_color};"
            f" border: 1px solid #555; border-radius: 3px; font-family: Consolas, monospace; }}"
        )

    def _refresh_widgets(self):
        for key in self.swatches:
            self._style_swatch(key)
        for key, (slider, val_label, fmt) in self.sliders.items():
            slider.blockSignals(True)
            slider.setValue(int(self.theme[key]))
            slider.blockSignals(False)
            val_label.setText(fmt(int(self.theme[key])))

    def _sync_preset_combo(self):
        for name, preset in THEME_PRESETS.items():
            if all(str(self.theme.get(k)).lower() == str(v).lower() for k, v in preset.items()):
                self.preset_combo.setCurrentText(name)
                return
        self.preset_combo.setCurrentIndex(0)

    def _on_preset_selected(self, index):
        if index <= 0:
            return
        self._apply_preset(self.preset_combo.itemText(index))

    def _apply_preset(self, name):
        self.theme.update(THEME_PRESETS[name])
        self._refresh_widgets()
        self._sync_preset_combo()
        self.on_change()


class FullBleedGridItem(pg.GraphicsObject):
    """
    Grilla full-bleed (borde a borde desde x=0 hasta x=w y de y=0 hasta y=h)
    dibujada detrás de las curvas FFT y EQ (zValue = -20).
    """

    def __init__(self, app_ref):
        super().__init__()
        self.app_ref = app_ref
        self.setZValue(-20)

    def boundingRect(self) -> QRectF:
        vb = self.app_ref.plot_widget.getPlotItem().vb
        return vb.sceneBoundingRect()

    def paint(self, painter: QPainter, option, widget=None):
        vb = self.app_ref.plot_widget.getPlotItem().vb
        rect = vb.sceneBoundingRect()
        if rect.width() <= 10 or rect.height() <= 10:
            return

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setClipRect(rect)

        grid_base = QColor(self.app_ref.theme.get("grid", "#2A2A2A"))
        alpha_pct = int(self.app_ref.theme.get("grid_alpha", 45))
        major_alpha = int(np.clip(alpha_pct * 2.1, 18, 160))
        minor_alpha = int(np.clip(alpha_pct * 0.95, 8, 90))

        pen_major = QPen(QColor(grid_base.red(), grid_base.green(), grid_base.blue(), major_alpha), 1)
        pen_minor = QPen(QColor(grid_base.red(), grid_base.green(), grid_base.blue(), minor_alpha), 1)

        is_x_log = (self.app_ref.x_scale_combo.currentText() == "Logarítmica")
        nyquist = self.app_ref.current_sr / 2.0

        # Líneas verticales de frecuencia (borde superior a borde inferior)
        if is_x_log:
            minor_freqs = [
                30, 40, 60, 70, 80, 90,
                300, 400, 600, 700, 800, 900,
                3000, 4000, 6000, 7000, 8000, 9000,
                30000, 40000,
            ]
            painter.setPen(pen_minor)
            for f in minor_freqs:
                if f <= nyquist:
                    x_s = vb.mapViewToScene(QPointF(float(np.log10(f)), 0.0)).x()
                    if rect.left() <= x_s <= rect.right():
                        painter.drawLine(QPointF(x_s, rect.top()), QPointF(x_s, rect.bottom()))

            major_freqs = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
            painter.setPen(pen_major)
            for f in major_freqs:
                if f <= nyquist:
                    x_s = vb.mapViewToScene(QPointF(float(np.log10(f)), 0.0)).x()
                    if rect.left() <= x_s <= rect.right():
                        painter.drawLine(QPointF(x_s, rect.top()), QPointF(x_s, rect.bottom()))
        else:
            step = 2000 if nyquist <= 24000 else 5000
            painter.setPen(pen_major)
            for f in range(step, int(nyquist) + 1, step):
                x_s = vb.mapViewToScene(QPointF(float(f), 0.0)).x()
                if rect.left() <= x_s <= rect.right():
                    painter.drawLine(QPointF(x_s, rect.top()), QPointF(x_s, rect.bottom()))

        # Líneas horizontales de dB (desde x=0 hasta el borde derecho)
        is_y_log = (self.app_ref.y_scale_combo.currentText() == "Logarítmica (dB)")
        if is_y_log:
            db_min = self.app_ref.db_min
            db_max = self.app_ref.db_max
            span = db_max - db_min
            step_db = 10 if span <= 140 else 20
            first_tick = int(np.ceil(db_min / step_db) * step_db)
            painter.setPen(pen_major)
            for db_val in range(first_tick, db_max + 1, step_db):
                y_s = vb.mapViewToScene(QPointF(0.0, float(db_val))).y()
                if rect.top() <= y_s <= rect.bottom():
                    painter.drawLine(QPointF(rect.left(), y_s), QPointF(rect.right(), y_s))
        else:
            painter.setPen(pen_major)
            for val in range(1, 10):
                y_s = vb.mapViewToScene(QPointF(0.0, val / 10.0)).y()
                if rect.top() <= y_s <= rect.bottom():
                    painter.drawLine(QPointF(rect.left(), y_s), QPointF(rect.right(), y_s))

        painter.restore()


class FullBleedEdgeMasksAndScalesItem(pg.GraphicsObject):
    """
    Capa frontal del lienzo (zValue = 25, sobre las curvas FFT/EQ y debajo de los nodos interactivos):
    1. Top Gradient Mask (`y = 0..75px`) para atenuar curvas altas bajo la barra flotante Liquid Glass.
    2. Bottom Vignette (`últimos 45px`) para fundir el espectro y los filtros de corte hacia -infinito.
    3. Etiquetas de Frecuencia inferiores (`20, 50, 100 ... 20k`) sobre el fundido inferior.
    4. Escalas Verticales Unificadas a la Derecha:
       - Columna extrema derecha (`#787C85`, 9pt): Escala FFT dBFS (`0, -10, -20 ... -120`).
       - Columna interna adyacente (`#E0B84C`, 9pt bold): Escala de Ganancia EQ (`+9, +6, +3, 0, -3, -6, -9, -12`).
    """

    def __init__(self, app_ref):
        super().__init__()
        self.app_ref = app_ref
        self.setZValue(25)

    def boundingRect(self) -> QRectF:
        vb = self.app_ref.plot_widget.getPlotItem().vb
        return vb.sceneBoundingRect()

    def paint(self, painter: QPainter, option, widget=None):
        vb = self.app_ref.plot_widget.getPlotItem().vb
        rect = vb.sceneBoundingRect()
        if rect.width() <= 10 or rect.height() <= 10:
            return

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setClipRect(rect)

        # 1. Top Gradient Mask (y = 0 a y = 75px) detrás de los controles flotantes Liquid Glass
        top_mask_h = 75.0
        top_grad = QLinearGradient(0.0, rect.top(), 0.0, rect.top() + top_mask_h)
        top_grad.setColorAt(0.0, QColor(18, 18, 20, 225))
        top_grad.setColorAt(0.55, QColor(18, 18, 20, 120))
        top_grad.setColorAt(1.0, QColor(18, 18, 20, 0))
        painter.fillRect(QRectF(rect.left(), rect.top(), rect.width(), top_mask_h), top_grad)

        # 2. Bottom Vignette (últimos 45px inferiores) hacia -infinito
        bot_mask_h = 45.0
        bot_y0 = rect.bottom() - bot_mask_h
        bot_grad = QLinearGradient(0.0, bot_y0, 0.0, rect.bottom())
        bot_grad.setColorAt(0.0, QColor(20, 20, 20, 0))
        bot_grad.setColorAt(0.58, QColor(16, 16, 18, 165))
        bot_grad.setColorAt(1.0, QColor(14, 14, 16, 235))
        painter.fillRect(QRectF(rect.left(), bot_y0, rect.width(), bot_mask_h), bot_grad)

        # 3. Etiquetas de Frecuencia en la franja inferior
        is_x_log = (self.app_ref.x_scale_combo.currentText() == "Logarítmica")
        nyquist = self.app_ref.current_sr / 2.0
        font_freq = QFont("Segoe UI", 8, QFont.Weight.DemiBold)
        painter.setFont(font_freq)
        painter.setPen(QColor("#8C919B"))

        max_freq_x = rect.right() - 80.0
        if is_x_log:
            freq_marks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
            for f in freq_marks:
                if f <= nyquist:
                    x_s = vb.mapViewToScene(QPointF(float(np.log10(f)), 0.0)).x()
                    if rect.left() + 14.0 <= x_s <= max_freq_x:
                        lbl = f"{int(f // 1000)}k" if f >= 1000 else str(f)
                        painter.drawText(
                            QRectF(x_s - 20.0, rect.bottom() - 19.0, 40.0, 16.0),
                            Qt.AlignmentFlag.AlignCenter,
                            lbl,
                        )
        else:
            step = 2000 if nyquist <= 24000 else 5000
            for f in range(step, int(nyquist) + 1, step):
                x_s = vb.mapViewToScene(QPointF(float(f), 0.0)).x()
                if rect.left() + 18.0 <= x_s <= max_freq_x:
                    lbl = f"{int(f // 1000)}k" if f >= 1000 else str(f)
                    painter.drawText(
                        QRectF(x_s - 22.0, rect.bottom() - 19.0, 44.0, 16.0),
                        Qt.AlignmentFlag.AlignCenter,
                        lbl,
                    )

        # 4. Escalas Verticales Unificadas en el Margen Derecho (junto al vúmetro L/R)
        is_y_log = (self.app_ref.y_scale_combo.currentText() == "Logarítmica (dB)")
        if is_y_log:
            # Columna derecha extrema: Nivel de señal FFT dBFS (#787C85, 9pt)
            font_dbfs = QFont("Segoe UI", 9)
            painter.setFont(font_dbfs)
            painter.setPen(QColor("#787C85"))

            db_min = self.app_ref.db_min
            db_max = self.app_ref.db_max
            span = db_max - db_min
            step_db = 10 if span <= 140 else 20
            first_tick = int(np.ceil(db_min / step_db) * step_db)
            x_dbfs_col = rect.right() - 34.0

            for db_val in range(first_tick, db_max + 1, step_db):
                y_s = vb.mapViewToScene(QPointF(0.0, float(db_val))).y()
                if rect.top() + 42.0 <= y_s <= rect.bottom() - 16.0:
                    painter.drawText(
                        QRectF(x_dbfs_col, y_s - 8.0, 29.0, 16.0),
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                        f"{db_val}",
                    )

            # Columna interna adyacente: Escala de Ganancia EQ (#E0B84C, 9pt bold)
            eq_vb = self.app_ref.eq_overlay.eq_vb
            eq_range = self.app_ref.eq_overlay.eq_range_db
            font_eq = QFont("Segoe UI", 9, QFont.Weight.Bold)
            painter.setFont(font_eq)
            painter.setPen(QColor("#E0B84C"))

            if abs(eq_range - 12.0) < 0.1:
                eq_ticks = [9, 6, 3, 0, -3, -6, -9, -12]
            else:
                eq_ticks = [24, 18, 12, 6, 0, -6, -12, -18, -24, -30]

            x_eq_col = rect.right() - 72.0
            for g_val in eq_ticks:
                y_eq = eq_vb.mapViewToScene(QPointF(0.0, float(g_val))).y()
                if rect.top() + 62.0 <= y_eq <= rect.bottom() - 16.0:
                    lbl_eq = f"+{g_val}" if g_val > 0 else f"{g_val}"
                    painter.drawText(
                        QRectF(x_eq_col, y_eq - 8.0, 32.0, 16.0),
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                        lbl_eq,
                    )

        painter.restore()


class GraphStageContainer(QWidget):
    """
    Contenedor full-bleed del lienzo que aloja como hijos directos:
    - `plot_widget` (ocupando el 100% del área desde x=0, y=0)
    - `top_bar` (Controles superiores flotantes Apple Liquid Glass en y=10px)
    - `eq_range_badge` (Badge interactivo `+12 dB` / `+30 dB` en la cabecera de la escala derecha)
    - `scope_overlay` (Osciloscopio desplegable con Scroll Blitting y `WA_OpaquePaintEvent`)
    - `floating_hud` (`FloatingBandHUD` superpuesto en la zona central-inferior)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot_widget = None
        self.floating_hud = None
        self.top_bar = None
        self.scope_overlay = None
        self.eq_range_badge = None

    def set_stage_widgets(
        self,
        plot_widget: QWidget,
        floating_hud: QWidget,
        top_bar: QWidget | None = None,
        scope_overlay: QWidget | None = None,
        eq_range_badge: QWidget | None = None,
    ):
        self.plot_widget = plot_widget
        self.floating_hud = floating_hud
        self.top_bar = top_bar
        self.scope_overlay = scope_overlay
        self.eq_range_badge = eq_range_badge
        self._reposition_children()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reposition_children()

    def _reposition_children(self):
        w = self.width()
        h = self.height()
        if self.plot_widget is not None:
            self.plot_widget.setGeometry(0, 0, w, h)

        if self.top_bar is not None:
            self.top_bar.setGeometry(10, 8, max(200, w - 20), 34)
            self.top_bar.raise_()

        if self.eq_range_badge is not None:
            # Ubicado justo en la cabecera de la columna dorada de ganancia EQ (+12 dB / +30 dB)
            badge_w = self.eq_range_badge.width()
            badge_x = max(10, w - 38 - badge_w)
            badge_y = 46
            self.eq_range_badge.move(badge_x, badge_y)
            self.eq_range_badge.raise_()

        if self.scope_overlay is not None:
            scope_w = max(240, w - 104)
            self.scope_overlay.setGeometry(12, 46, scope_w, 164)
            if self.scope_overlay.isVisible():
                self.scope_overlay.raise_()

        if self.floating_hud is not None:
            hud_w = self.floating_hud.width()
            hud_h = self.floating_hud.height()
            hud_x = max(16, (w - 72 - hud_w) // 2)
            # Posicionado justo encima de las etiquetas de frecuencia inferiores
            hud_y = max(48, h - hud_h - 22)
            self.floating_hud.move(hud_x, hud_y)
            self.floating_hud.raise_()


class AudioAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BDJ-FFT-EQ")
        self.setMinimumSize(860, 520)

        screen = QApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            w = min(1140, int(avail.width() * 0.86))
            h = min(700, int(avail.height() * 0.85))
            self.resize(w, h)
            self.move(
                avail.x() + (avail.width() - w) // 2,
                avail.y() + (avail.height() - h) // 2,
            )
        else:
            self.resize(1120, 680)

        # Estética Gris Puro Neutro de Estudio (#141414 lienzo, #1C1C1C barras/paneles, #2A2A2A bordes/grilla)
        self.setStyleSheet("""
            QMainWindow, QDialog {
                background-color: #141414;
                color: #D8DEE9;
                font-family: "Segoe UI", sans-serif;
            }
            QWidget {
                color: #D8DEE9;
                font-family: "Segoe UI", sans-serif;
            }
            QLabel {
                background: transparent;
                border: none;
            }
            QComboBox, QSpinBox {
                background-color: #242424;
                color: #ECEFF4;
                border: 1px solid #383838;
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 11px;
                font-weight: bold;
            }
            QComboBox:hover, QSpinBox:hover {
                border-color: #525252;
            }
            QPushButton {
                background-color: #242424;
                color: #D8DEE9;
                border: 1px solid #363636;
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #303030;
                color: #FFFFFF;
            }
            QCheckBox {
                background: transparent;
                border: none;
                color: #CBD5E1;
                font-size: 11px;
                spacing: 6px;
            }
        """)

        self.sample_rates = [44100, 48000, 88200, 96000]
        self.current_sr = 48000
        self.audio_block_size = 512
        self.fft_sizes = [512, 1024, 2048, 4096, 8192, 16384]
        self.fft_size = 4096
        self.channels = 2

        self.audio_queue = queue.Queue(maxsize=10)
        self.stream = None
        self._loading_config = True

        # Inicio MUTEADO por seguridad
        self.is_muted = True

        # Rango por defecto del analizador espectral: -120 dB a 0 dB
        self.decay_factor = 0.88
        self.db_min = -120
        self.db_max = 0

        # Flags para no llamar a setData() del espectro ni osciloscopio cuando ya reposan en silencio
        self._pre_drawn_at_rest = False
        self._post_drawn_at_rest = False
        self._scope_tick_counter = 0

        self.saved_config = load_full_config()
        self.theme = self.saved_config["theme"]
        self.color_dialog = None

        self.fft_analyzer = FFTSpectrumAnalyzer(
            sample_rate=self.current_sr, fft_size=self.fft_size, decay_factor=self.decay_factor
        )
        self.eq_engine = ParametricEQEngine(sample_rate=self.current_sr, channels=self.channels)
        self.scope_buffer = WaveformEnvelopeBuffer(sample_rate=self.current_sr, block_size=self.audio_block_size)

        self.config_save_timer = QTimer(self)
        self.config_save_timer.setSingleShot(True)
        self.config_save_timer.setInterval(600)
        self.config_save_timer.timeout.connect(self.save_app_config)

        self._init_ui()
        self._populate_devices()
        self._restore_saved_state()
        self._loading_config = False
        self.restart_audio_stream()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_visuals)
        self.timer.start(16)

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        # LIENZO 100% FULL-BLEED (Borde a Borde sin márgenes)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        center_layout = QHBoxLayout()
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)

        self.graph_container = GraphStageContainer(self)

        self.plot_widget = pg.PlotWidget(self.graph_container)
        self.plot_widget.setFrameShape(QFrame.Shape.NoFrame)
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.hideButtons()
        self.plot_widget.setMenuEnabled(False)
        plot_item = self.plot_widget.getPlotItem()
        plot_item.vb.setMouseEnabled(x=False, y=False)
        for ax_name in ("left", "bottom", "right", "top"):
            self.plot_widget.hideAxis(ax_name)
        plot_item.layout.setContentsMargins(0, 0, 0, 0)
        plot_item.vb.setDefaultPadding(0.0)

        # Grilla full-bleed detrás del espectro
        self.full_bleed_grid = FullBleedGridItem(self)
        self.full_bleed_grid.setZValue(-30)
        plot_item.scene().addItem(self.full_bleed_grid)

        plot_item.vb.setZValue(-10)

        self.pre_curve = self.plot_widget.plot(
            pen=pg.mkPen(color=(150, 160, 178, 95), width=1.1)
        )
        self.pre_curve.setZValue(-20)

        # El piso visual del relleno cae hasta db_min - 12.0 (-132 dB) para fundirse a alpha=0 en el borde inferior
        self.curve = self.plot_widget.plot(fillLevel=float(self.db_min - 12.0))
        self.curve.setZValue(-10)

        self.eq_overlay = ProQ4Overlay(self.plot_widget, self.eq_engine, self.graph_container)

        # Capa de viñetas (Top Gradient Mask + Bottom -inf Vignette) y Escalas Unificadas a la derecha
        self.edge_masks_and_scales = FullBleedEdgeMasksAndScalesItem(self)
        self.edge_masks_and_scales.setZValue(25)
        plot_item.scene().addItem(self.edge_masks_and_scales)

        # --- BARRA SUPERIOR FLOTANTE ESTILO "APPLE LIQUID GLASS" ---
        self.top_bar = QWidget(self.graph_container)
        self.top_bar.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        top_layout = QHBoxLayout(self.top_bar)
        top_layout.setContentsMargins(4, 2, 4, 2)
        top_layout.setSpacing(8)

        brand_label = QLabel(
            "<b style='color:#00E5FF; font-size:13px; letter-spacing:0.8px;'>BDJ</b> "
            "<b style='color:#FFD54F; font-size:13px;'>FFT-EQ</b>"
        )
        brand_label.setContentsMargins(2, 0, 4, 0)
        top_layout.addWidget(brand_label)

        self.mute_btn = LiquidGlassButton("🔇 Muted (Activar)", self.top_bar, tint_color="#FF5252", active_tint_color="#FF5252")
        self.mute_btn.setCheckable(True)
        self.mute_btn.setChecked(True)
        self.mute_btn.clicked.connect(self.toggle_mute)
        self._update_mute_button_style()
        top_layout.addWidget(self.mute_btn)

        # Panel emergente de Dispositivos de Audio y Sample Rate
        self.audio_menu_btn = LiquidGlassButton("🎧 Audio I/O ▾", self.top_bar)
        self.audio_menu_btn.setToolTip("Seleccionar Dispositivo de Entrada, Salida y Sample Rate")
        self.audio_menu_btn.clicked.connect(self._show_audio_popup)
        self._build_audio_io_popup()
        top_layout.addWidget(self.audio_menu_btn)

        # Panel emergente de Analizador FFT (incluye FFT Samples y Color de Fondo)
        self.analyzer_menu_btn = LiquidGlassButton("📊 Analyzer: Pre+Post ▾", self.top_bar)
        self.analyzer_menu_btn.setToolTip("Configurar Analizador FFT, FFT Samples, Escalas, Rango dB y Color de Fondo")
        self.analyzer_menu_btn.clicked.connect(self._show_analyzer_popup)
        self._build_analyzer_popup()
        top_layout.addWidget(self.analyzer_menu_btn)

        top_layout.addStretch(1)

        # --- MINI-OSCILOSCOPIO EN CÁPSULA LIQUID GLASS ---
        self.scope_widget = WaveformScopeWidget(self.scope_buffer, self.top_bar)
        self.scope_widget.toggleExpandedRequested.connect(self.toggle_expanded_scope)
        self.scope_widget.durationChanged.connect(self._on_scope_duration_changed)
        top_layout.addWidget(self.scope_widget)

        top_layout.addStretch(1)

        self.bypass_btn = LiquidGlassButton("Bypass EQ", self.top_bar, active_tint_color="#FF5252")
        self.bypass_btn.setCheckable(True)
        self.bypass_btn.setToolTip("Desactivar temporalmente todas las bandas del ecualizador")
        self.bypass_btn.clicked.connect(self._on_global_bypass_toggled)
        top_layout.addWidget(self.bypass_btn)

        self.reset_eq_btn = LiquidGlassButton("↺ Reset EQ", self.top_bar)
        self.reset_eq_btn.setToolTip("Restaurar las 4 bandas por defecto en 0 dB")
        self.reset_eq_btn.clicked.connect(lambda: self.eq_panel._on_reset_eq())
        top_layout.addWidget(self.reset_eq_btn)

        self.theme_menu_btn = LiquidGlassButton("🎨", self.top_bar)
        self.theme_menu_btn.setFixedWidth(34)
        self.theme_menu_btn.setToolTip("Color de Fondo y Ajustes Visuales del Tema")
        self.theme_menu_btn.clicked.connect(self._show_background_color_menu)
        top_layout.addWidget(self.theme_menu_btn)

        self.fullscreen_btn = LiquidGlassButton("⛶", self.top_bar)
        self.fullscreen_btn.setFixedWidth(34)
        self.fullscreen_btn.setToolTip("Pantalla completa (Atajo: F11)")
        self.fullscreen_btn.clicked.connect(self.toggle_fullscreen)
        top_layout.addWidget(self.fullscreen_btn)

        # --- BADGE INTERACTIVO DE RANGO EQ (+12 dB / +30 dB) EN LA ESCALA DERECHA ---
        self.eq_range_badge = LiquidGlassButton("+12 dB", self.graph_container, tint_color="#E0B84C", active_tint_color="#E0B84C", pill_radius=9.0)
        self.eq_range_badge.setFixedSize(48, 20)
        self.eq_range_badge.setCheckable(True)
        self.eq_range_badge.setChecked(True)
        self.eq_range_badge.setToolTip("Clic para alternar el rango visual de ganancia EQ entre ±12 dB y ±30 dB")
        self.eq_range_badge.clicked.connect(self._on_toggle_eq_range_badge)

        # --- PANEL DESPLEGABLE DEL OSCILOSCOPIO CON SCROLL BLITTING ---
        self.scope_overlay = WaveformOverlayWidget(self.scope_buffer, self.graph_container)
        self.scope_overlay.collapseRequested.connect(self.toggle_expanded_scope)

        # --- HUD FLOTANTE SUPERPUESTO CON `GAIN` EN LA JOROBA ---
        self.eq_panel = FloatingBandHUD(self.eq_engine, self.eq_overlay, self.graph_container)
        self.eq_panel.parameters_changed.connect(self.schedule_config_save)

        self.graph_container.set_stage_widgets(
            self.plot_widget,
            self.eq_panel,
            top_bar=self.top_bar,
            scope_overlay=self.scope_overlay,
            eq_range_badge=self.eq_range_badge,
        )
        center_layout.addWidget(self.graph_container, 1)

        self.stereo_meter = SlimStereoMeter(self)
        center_layout.addWidget(self.stereo_meter)

        main_layout.addLayout(center_layout, 1)

        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

        self.on_hold_changed()
        self.update_axes_config()
        self.apply_theme()

    def _on_toggle_eq_range_badge(self):
        new_range = self.eq_overlay.toggle_eq_range_db()
        self.eq_range_badge.setChecked(True)
        self.eq_range_badge.setText(f"+{int(new_range)} dB")
        self.edge_masks_and_scales.update()
        self.schedule_config_save()

    def toggle_expanded_scope(self):
        visible = not self.scope_overlay.isVisible()
        self.scope_overlay.setVisible(visible)
        if visible:
            self.scope_overlay.raise_()
            self.scope_overlay.update()
        self.scope_widget.set_expanded_indicator(visible)

    def _on_scope_duration_changed(self, _sec: int):
        self.scope_overlay.invalidate_cache()
        self.schedule_config_save()

    def _show_background_color_menu(self):
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #1C1C1C;
                color: #ECEFF4;
                border: 1px solid #363636;
                padding: 4px;
                font-size: 11px;
            }
            QMenu::item { padding: 5px 16px; border-radius: 3px; }
            QMenu::item:selected { background-color: #2E3440; color: #FFFFFF; }
        """)
        hdr = menu.addAction("Color de Fondo")
        hdr.setEnabled(False)
        menu.addSeparator()

        for label, hex_val in BACKGROUND_COLOR_PRESETS:
            act = menu.addAction(_make_swatch_icon(hex_val, 12), label)
            act.setCheckable(True)
            act.setChecked(self.theme.get("background", "").lower() == hex_val.lower())
            act.triggered.connect(lambda _=False, hv=hex_val: self._set_background_color(hv))

        act_custom = menu.addAction("🎨 Personalizado...")
        act_custom.triggered.connect(self._pick_custom_background_color)

        menu.addSeparator()
        act_dlg = menu.addAction("⚙ Configuración Completa de Colores...")
        act_dlg.triggered.connect(self.open_color_settings)

        pos = self.theme_menu_btn.mapToGlobal(QPoint(0, self.theme_menu_btn.height() + 4))
        menu.exec(pos)

    def _set_background_color(self, hex_color: str):
        c = QColor(hex_color)
        if c.isValid():
            self.theme["background"] = c.name()
            self._sync_bg_combo()
            self.on_theme_changed()

    def _pick_custom_background_color(self):
        orig = self.theme.get("background", "#141414")
        dlg = QColorDialog(QColor(orig), self)
        dlg.setWindowTitle("Color de Fondo del Gráfico")
        dlg.setOption(QColorDialog.ColorDialogOption.DontUseNativeDialog, True)
        dlg.currentColorChanged.connect(lambda c: self._set_background_color(c.name()) if c.isValid() else None)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._set_background_color(dlg.selectedColor().name())
        else:
            self._set_background_color(orig)

    def _build_audio_io_popup(self):
        self.audio_popup = QDialog(self, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.audio_popup.setStyleSheet("""
            QDialog {
                background-color: #1C1C1C;
                border: 1px solid #363636;
                border-radius: 6px;
            }
            QLabel {
                background: transparent;
                border: none;
                color: #C5CDD9;
                font-size: 11px;
                font-weight: bold;
            }
        """)
        grid = QGridLayout(self.audio_popup)
        grid.setContentsMargins(12, 10, 12, 10)
        grid.setSpacing(8)

        grid.addWidget(QLabel("Dispositivo de Entrada:"), 0, 0)
        self.in_combo = QComboBox()
        self.in_combo.setMinimumWidth(320)
        self.in_combo.currentIndexChanged.connect(self._on_audio_device_changed)
        grid.addWidget(self.in_combo, 0, 1)

        grid.addWidget(QLabel("Dispositivo de Salida:"), 1, 0)
        self.out_combo = QComboBox()
        self.out_combo.setMinimumWidth(320)
        self.out_combo.currentIndexChanged.connect(self._on_audio_device_changed)
        grid.addWidget(self.out_combo, 1, 1)

        grid.addWidget(QLabel("Sample Rate:"), 2, 0)
        self.sr_combo = QComboBox()
        for sr in self.sample_rates:
            self.sr_combo.addItem(f"{sr} Hz", sr)
        self.sr_combo.setCurrentText("48000 Hz")
        self.sr_combo.currentIndexChanged.connect(self.on_sr_changed)
        grid.addWidget(self.sr_combo, 2, 1)

    def _show_audio_popup(self):
        self.audio_popup.adjustSize()
        pos = self.audio_menu_btn.mapToGlobal(QPoint(0, self.audio_menu_btn.height() + 4))
        self.audio_popup.move(pos)
        self.audio_popup.show()

    def _build_analyzer_popup(self):
        """Panel del Analizador con FFT Samples y Selector de Color de Fondo."""
        self.analyzer_popup = QDialog(self, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.analyzer_popup.setStyleSheet("""
            QDialog {
                background-color: #1C1C1C;
                border: 1px solid #363636;
                border-radius: 6px;
            }
            QLabel {
                background: transparent;
                border: none;
                color: #C5CDD9;
                font-size: 11px;
                font-weight: bold;
            }
        """)
        grid = QGridLayout(self.analyzer_popup)
        grid.setContentsMargins(12, 10, 12, 10)
        grid.setSpacing(8)

        chk_row = QHBoxLayout()
        self.show_pre_chk = QCheckBox("Mostrar Pre-EQ")
        self.show_pre_chk.setChecked(True)
        self.show_pre_chk.toggled.connect(self._update_analyzer_btn_label)
        chk_row.addWidget(self.show_pre_chk)

        self.show_post_chk = QCheckBox("Mostrar Post-EQ")
        self.show_post_chk.setChecked(True)
        self.show_post_chk.toggled.connect(self._update_analyzer_btn_label)
        chk_row.addWidget(self.show_post_chk)
        grid.addLayout(chk_row, 0, 0, 1, 2)

        grid.addWidget(QLabel("FFT Samples:"), 1, 0)
        self.fft_combo = QComboBox()
        for sz in self.fft_sizes:
            self.fft_combo.addItem(f"{sz} samples", sz)
        self.fft_combo.setCurrentText("4096 samples")
        self.fft_combo.currentIndexChanged.connect(self.on_fft_size_changed)
        grid.addWidget(self.fft_combo, 1, 1)

        grid.addWidget(QLabel("Escala Eje X:"), 2, 0)
        self.x_scale_combo = QComboBox()
        self.x_scale_combo.addItems(["Logarítmica", "Lineal"])
        self.x_scale_combo.currentIndexChanged.connect(self.update_axes_config)
        grid.addWidget(self.x_scale_combo, 2, 1)

        grid.addWidget(QLabel("Escala Eje Y:"), 3, 0)
        self.y_scale_combo = QComboBox()
        self.y_scale_combo.addItems(["Logarítmica (dB)", "Lineal (Amplitud)"])
        self.y_scale_combo.currentIndexChanged.connect(self.on_y_scale_changed)
        grid.addWidget(self.y_scale_combo, 3, 1)

        grid.addWidget(QLabel("Rango dB Mín:"), 4, 0)
        self.db_min_spin = QSpinBox()
        self.db_min_spin.setRange(-140, -10)
        self.db_min_spin.setSingleStep(5)
        self.db_min_spin.setValue(self.db_min)
        self.db_min_spin.setSuffix(" dB")
        self.db_min_spin.valueChanged.connect(self.on_db_range_changed)
        grid.addWidget(self.db_min_spin, 4, 1)

        grid.addWidget(QLabel("Rango dB Máx:"), 5, 0)
        self.db_max_spin = QSpinBox()
        self.db_max_spin.setRange(-60, 20)
        self.db_max_spin.setSingleStep(5)
        self.db_max_spin.setValue(self.db_max)
        self.db_max_spin.setSuffix(" dB")
        self.db_max_spin.valueChanged.connect(self.on_db_range_changed)
        grid.addWidget(self.db_max_spin, 5, 1)

        hold_row = QHBoxLayout()
        self.hold_slider = QSlider(Qt.Orientation.Horizontal)
        self.hold_slider.setRange(0, 99)
        self.hold_slider.setValue(85)
        self.hold_slider.setMinimumWidth(130)
        self.hold_slider.valueChanged.connect(self.on_hold_changed)
        self.hold_val_label = QLabel("85%")
        self.hold_val_label.setFixedWidth(36)
        hold_row.addWidget(self.hold_slider)
        hold_row.addWidget(self.hold_val_label)
        grid.addWidget(QLabel("Retención FFT:"), 6, 0)
        grid.addLayout(hold_row, 6, 1)

        grid.addWidget(QLabel("Color de Fondo:"), 7, 0)
        self.bg_color_combo = QComboBox()
        for label, hex_val in BACKGROUND_COLOR_PRESETS:
            self.bg_color_combo.addItem(label, hex_val)
        self.bg_color_combo.addItem("Personalizado...", "custom")
        self.bg_color_combo.activated.connect(self._on_bg_combo_activated)
        grid.addWidget(self.bg_color_combo, 7, 1)

        colors_btn = QPushButton("🎨 Configurar Colores del Espectro...")
        colors_btn.clicked.connect(lambda: (self.analyzer_popup.close(), self.open_color_settings()))
        grid.addWidget(colors_btn, 8, 0, 1, 2)

    def _on_bg_combo_activated(self, index: int):
        val = self.bg_color_combo.itemData(index)
        if val == "custom":
            self.analyzer_popup.close()
            self._pick_custom_background_color()
        elif isinstance(val, str):
            self._set_background_color(val)

    def _sync_bg_combo(self):
        cur_bg = self.theme.get("background", "#141414").lower()
        self.bg_color_combo.blockSignals(True)
        matched = False
        for i in range(self.bg_color_combo.count()):
            data = self.bg_color_combo.itemData(i)
            if isinstance(data, str) and data.lower() == cur_bg:
                self.bg_color_combo.setCurrentIndex(i)
                matched = True
                break
        if not matched:
            self.bg_color_combo.setCurrentIndex(self.bg_color_combo.count() - 1)
        self.bg_color_combo.blockSignals(False)

    def _show_analyzer_popup(self):
        self._sync_bg_combo()
        self.analyzer_popup.adjustSize()
        pos = self.analyzer_menu_btn.mapToGlobal(QPoint(0, self.analyzer_menu_btn.height() + 4))
        self.analyzer_popup.move(pos)
        self.analyzer_popup.show()

    def _update_analyzer_btn_label(self):
        pre = self.show_pre_chk.isChecked()
        post = self.show_post_chk.isChecked()
        if not pre:
            self.pre_curve.setData([], [])
        if not post:
            self.curve.setData([], [])
        self._pre_drawn_at_rest = False
        self._post_drawn_at_rest = False

        if pre and post:
            txt = "📊 Analyzer: Pre+Post ▾"
        elif post:
            txt = "📊 Analyzer: Post ▾"
        elif pre:
            txt = "📊 Analyzer: Pre ▾"
        else:
            txt = "📊 Analyzer: Off ▾"
        self.analyzer_menu_btn.setText(txt)
        self.schedule_config_save()

    def _on_global_bypass_toggled(self, checked: bool):
        with self.eq_engine.lock:
            self.eq_engine.bypass = checked
            self.eq_engine.mark_dirty()
        self.bypass_btn.setChecked(checked)
        self.bypass_btn.update()
        self.eq_overlay.update_eq_curves(force=True)
        self.schedule_config_save()

    def open_color_settings(self):
        if self.color_dialog is None:
            self.color_dialog = ColorSettingsDialog(self.theme, self.on_theme_changed, self)
        self.color_dialog.show()
        self.color_dialog.raise_()
        self.color_dialog.activateWindow()

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def show_context_menu(self, pos):
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #1C1C1C;
                color: #EEE;
                border: 1px solid #363636;
                padding: 4px;
            }
            QMenu::item { padding: 5px 18px; border-radius: 3px; }
            QMenu::item:selected { background-color: #2E3440; color: #FFF; }
        """)

        act_add = menu.addAction("➕ Añadir Banda EQ (Doble clic en gráfico)")
        act_add.triggered.connect(self.eq_panel._on_add_band)

        act_bypass = menu.addAction("🔇 Bypass Global EQ")
        act_bypass.setCheckable(True)
        act_bypass.setChecked(self.eq_engine.bypass)
        act_bypass.triggered.connect(lambda c: (
            self.bypass_btn.setChecked(c),
            self._on_global_bypass_toggled(c),
        ))

        menu.addSeparator()

        act_scope = menu.addAction("📈 Desplegar Osciloscopio sobre FFT")
        act_scope.setCheckable(True)
        act_scope.setChecked(self.scope_overlay.isVisible())
        act_scope.triggered.connect(self.toggle_expanded_scope)

        act_fs = menu.addAction("⛶ Pantalla Completa (F11)")
        act_fs.setCheckable(True)
        act_fs.setChecked(self.isFullScreen())
        act_fs.triggered.connect(self.toggle_fullscreen)

        menu.addSeparator()

        act_mute = menu.addAction("🔇 Salida Física Muteada")
        act_mute.setCheckable(True)
        act_mute.setChecked(self.is_muted)
        act_mute.triggered.connect(lambda: self.toggle_mute(not self.is_muted))

        bg_menu = menu.addMenu("🎨 Color de Fondo")
        for label, hex_val in BACKGROUND_COLOR_PRESETS:
            act_bg = bg_menu.addAction(_make_swatch_icon(hex_val, 12), label)
            act_bg.triggered.connect(lambda _=False, hv=hex_val: self._set_background_color(hv))
        act_bg_custom = bg_menu.addAction("Personalizado...")
        act_bg_custom.triggered.connect(self._pick_custom_background_color)

        act_colors = menu.addAction("⚙ Configurar Colores del Espectro...")
        act_colors.triggered.connect(self.open_color_settings)

        menu.exec(QCursor.pos())

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_F11:
            self.toggle_fullscreen()
            event.accept()
        elif event.key() == Qt.Key.Key_Escape:
            if self.scope_overlay.isVisible():
                self.toggle_expanded_scope()
            elif self.isFullScreen():
                self.showNormal()
            event.accept()
        elif event.key() == Qt.Key.Key_Delete:
            self.eq_panel._on_delete_band()
            event.accept()
        else:
            super().keyPressEvent(event)

    def schedule_config_save(self):
        if not self._loading_config:
            self.config_save_timer.start()

    def on_theme_changed(self):
        self.apply_theme()
        self.schedule_config_save()

    def save_app_config(self):
        payload = {
            "theme": {k: self.theme[k] for k in (*THEME_COLOR_KEYS, *THEME_NUMERIC_RANGES)},
            "audio": {
                "input_device_name": self.in_combo.currentText(),
                "input_device_index": self.in_combo.currentData(),
                "output_device_name": self.out_combo.currentText(),
                "output_device_index": self.out_combo.currentData(),
                "sample_rate": int(self.current_sr),
            },
            "analyzer": {
                "fft_size": int(self.fft_size),
                "show_pre": bool(self.show_pre_chk.isChecked()),
                "show_post": bool(self.show_post_chk.isChecked()),
                "x_scale": self.x_scale_combo.currentText(),
                "y_scale": self.y_scale_combo.currentText(),
                "db_min": int(self.db_min),
                "db_max": int(self.db_max),
                "hold_percent": int(self.hold_slider.value()),
                "scope_duration_sec": int(self.scope_buffer.duration_sec),
                "eq_range_db": float(self.eq_overlay.eq_range_db),
            },
            "eq": {
                **self.eq_engine.to_dict(),
                "selected_band_idx": int(self.eq_overlay.selected_band_idx),
            },
        }
        save_full_config(payload)

    def _restore_saved_state(self):
        saved_version = int(self.saved_config.get("version", 1))
        audio_cfg = self.saved_config.get("audio", {})
        if isinstance(audio_cfg, dict):
            in_name = audio_cfg.get("input_device_name")
            in_idx_saved = audio_cfg.get("input_device_index")
            if isinstance(in_name, str) and self.in_combo.findText(in_name) >= 0:
                self.in_combo.setCurrentIndex(self.in_combo.findText(in_name))
            elif isinstance(in_idx_saved, int) and self.in_combo.findData(in_idx_saved) >= 0:
                self.in_combo.setCurrentIndex(self.in_combo.findData(in_idx_saved))

            out_name = audio_cfg.get("output_device_name")
            out_idx_saved = audio_cfg.get("output_device_index")
            if isinstance(out_name, str) and self.out_combo.findText(out_name) >= 0:
                self.out_combo.setCurrentIndex(self.out_combo.findText(out_name))
            elif isinstance(out_idx_saved, int) and self.out_combo.findData(out_idx_saved) >= 0:
                self.out_combo.setCurrentIndex(self.out_combo.findData(out_idx_saved))

            sr_saved = audio_cfg.get("sample_rate")
            if sr_saved in self.sample_rates:
                self.current_sr = int(sr_saved)
                self.sr_combo.setCurrentIndex(self.sr_combo.findData(self.current_sr))
                self.fft_analyzer.set_sample_rate(self.current_sr)
                self.eq_engine.set_sample_rate(self.current_sr)
                self.scope_buffer.set_sample_rate(self.current_sr)

        analyzer_cfg = self.saved_config.get("analyzer", {})
        if isinstance(analyzer_cfg, dict):
            fft_sz = analyzer_cfg.get("fft_size")
            if fft_sz in self.fft_sizes:
                self.fft_size = int(fft_sz)
                self.fft_combo.setCurrentIndex(self.fft_combo.findData(self.fft_size))
                self.fft_analyzer.set_fft_size(self.fft_size)

            if "show_pre" in analyzer_cfg:
                self.show_pre_chk.setChecked(bool(analyzer_cfg["show_pre"]))
            if "show_post" in analyzer_cfg:
                self.show_post_chk.setChecked(bool(analyzer_cfg["show_post"]))

            x_sc = analyzer_cfg.get("x_scale")
            if x_sc in ("Logarítmica", "Lineal"):
                self.x_scale_combo.setCurrentText(x_sc)

            y_sc = analyzer_cfg.get("y_scale")
            if y_sc in ("Logarítmica (dB)", "Lineal (Amplitud)"):
                self.y_scale_combo.setCurrentText(y_sc)

            if saved_version >= CURRENT_CONFIG_VERSION:
                db_min_s = analyzer_cfg.get("db_min")
                db_max_s = analyzer_cfg.get("db_max")
                if isinstance(db_min_s, int) and isinstance(db_max_s, int) and db_min_s < db_max_s - 5:
                    self.db_min = int(np.clip(db_min_s, -140, -10))
                    self.db_max = int(np.clip(db_max_s, -60, 20))
                    self.db_min_spin.setValue(self.db_min)
                    self.db_max_spin.setValue(self.db_max)

            hold_s = analyzer_cfg.get("hold_percent")
            if isinstance(hold_s, int):
                self.hold_slider.setValue(int(np.clip(hold_s, 0, 99)))

            scope_dur = analyzer_cfg.get("scope_duration_sec")
            if scope_dur in (5, 10):
                self.scope_widget.set_duration(int(scope_dur))

            eq_rng = analyzer_cfg.get("eq_range_db")
            if eq_rng in (12, 12.0, 30, 30.0):
                self.eq_overlay.set_eq_range_db(float(eq_rng))
                self.eq_range_badge.setText(f"+{int(float(eq_rng))} dB")

        eq_cfg = self.saved_config.get("eq", {})
        if isinstance(eq_cfg, dict):
            self.eq_engine.load_from_dict(eq_cfg)
            sel_idx = int(eq_cfg.get("selected_band_idx", 0))
            if 0 <= sel_idx < len(self.eq_engine.bands):
                self.eq_overlay.selected_band_idx = sel_idx
            self.bypass_btn.setChecked(self.eq_engine.bypass)
            self._on_global_bypass_toggled(self.eq_engine.bypass)
            self.eq_panel.refresh_from_engine()

        self._sync_bg_combo()
        self.update_axes_config()
        self.eq_overlay.update_eq_curves(force=True)

    def _update_curve_brush(self):
        t = self.theme
        fill = QColor(t.get("fill", "#6E747E"))
        base_alpha = int(np.clip(int(t.get("fill_alpha", 40)), 0, 255))
        fill.setAlpha(base_alpha)
        self.curve.setBrush(pg.mkBrush(fill))

    def apply_theme(self):
        t = self.theme
        self.plot_widget.setBackground(t["background"])
        self.curve.setPen(pg.mkPen(color=t["curve"], width=t["curve_width"] / 10.0))
        self._update_curve_brush()
        self.plot_widget.showGrid(x=False, y=False)
        self.full_bleed_grid.update()
        self.edge_masks_and_scales.update()

    def _update_mute_button_style(self):
        if self.is_muted:
            self.mute_btn.setText("🔇 Muted (Activar)")
            self.mute_btn.setTintColor("#FF5252")
            self.mute_btn.setActiveTintColor("#FF5252")
        else:
            self.mute_btn.setText("🔊 Audio ON")
            self.mute_btn.setTintColor("#2ECC71")
            self.mute_btn.setActiveTintColor("#2ECC71")
        self.mute_btn.update()

    def toggle_mute(self, checked=None):
        if checked is None:
            self.is_muted = self.mute_btn.isChecked()
        else:
            self.is_muted = bool(checked)
            self.mute_btn.setChecked(self.is_muted)
        self._update_mute_button_style()

    def _on_audio_device_changed(self):
        if not self._loading_config:
            self.restart_audio_stream()
            self.schedule_config_save()

    def on_fft_size_changed(self):
        self.fft_size = self.fft_combo.currentData()
        self.fft_analyzer.set_fft_size(self.fft_size)
        self._pre_drawn_at_rest = False
        self._post_drawn_at_rest = False
        self.schedule_config_save()

    def on_hold_changed(self):
        val = self.hold_slider.value()
        self.hold_val_label.setText(f"{val}%")
        self.decay_factor = 0.20 + (val / 99.0) * 0.792
        self.fft_analyzer.set_decay_factor(self.decay_factor)
        self.schedule_config_save()

    def on_y_scale_changed(self):
        is_y_log = (self.y_scale_combo.currentText() == "Logarítmica (dB)")
        self.db_min_spin.setEnabled(is_y_log)
        self.db_max_spin.setEnabled(is_y_log)
        self.update_axes_config()
        self.schedule_config_save()

    def on_db_range_changed(self):
        min_v = self.db_min_spin.value()
        max_v = self.db_max_spin.value()
        if min_v >= max_v - 5:
            if self.sender() == self.db_min_spin:
                self.db_min_spin.setValue(max_v - 5)
            else:
                self.db_max_spin.setValue(min_v + 5)
            return

        self.db_min = self.db_min_spin.value()
        self.db_max = self.db_max_spin.value()
        self.update_axes_config()
        self.schedule_config_save()

    def _populate_devices(self):
        self.in_combo.blockSignals(True)
        self.out_combo.blockSignals(True)
        self.in_combo.clear()
        self.out_combo.clear()

        devices = sd.query_devices()
        hostapis = sd.query_hostapis()

        for idx, dev in enumerate(devices):
            api_name = hostapis[dev["hostapi"]]["name"]
            dev_label = f"[{api_name}] {dev['name']}"
            if dev["max_input_channels"] > 0:
                self.in_combo.addItem(dev_label, idx)
            if dev["max_output_channels"] > 0:
                self.out_combo.addItem(dev_label, idx)

        default_in, default_out = sd.default.device
        in_idx = self.in_combo.findData(default_in)
        out_idx = self.out_combo.findData(default_out)
        if in_idx >= 0:
            self.in_combo.setCurrentIndex(in_idx)
        if out_idx >= 0:
            self.out_combo.setCurrentIndex(out_idx)

        self.in_combo.blockSignals(False)
        self.out_combo.blockSignals(False)

    def on_sr_changed(self):
        self.current_sr = self.sr_combo.currentData()
        self.fft_analyzer.set_sample_rate(self.current_sr)
        self.eq_engine.set_sample_rate(self.current_sr)
        self.scope_buffer.set_sample_rate(self.current_sr)
        self.update_axes_config()
        if not self._loading_config:
            self.restart_audio_stream()
            self.schedule_config_save()

    def update_axes_config(self):
        is_x_log = (self.x_scale_combo.currentText() == "Logarítmica")
        is_y_log = (self.y_scale_combo.currentText() == "Logarítmica (dB)")

        self.plot_widget.setLogMode(x=is_x_log, y=False)
        for ax_name in ("left", "bottom", "right", "top"):
            self.plot_widget.hideAxis(ax_name)

        nyquist = self.current_sr / 2.0
        if is_x_log:
            self.plot_widget.setXRange(np.log10(10.0), np.log10(nyquist), padding=0.0)
        else:
            self.plot_widget.setXRange(0.0, nyquist, padding=0.0)

        if is_y_log:
            # Extender -12 dB por debajo de db_min (-132 dB) para la zona de fundido a -infinito
            # y +12 dB por encima de db_max (+12 dB) para alojar la barra flotante Liquid Glass
            y_floor = float(self.db_min - 12.0)
            y_ceil = float(self.db_max + 12.0)
            self.plot_widget.setYRange(y_floor, y_ceil, padding=0.0)
            self.curve.setFillLevel(y_floor)
        else:
            self.plot_widget.setYRange(0.0, 1.08, padding=0.0)
            self.curve.setFillLevel(0.0)

        self._update_curve_brush()
        self._pre_drawn_at_rest = False
        self._post_drawn_at_rest = False
        self.eq_overlay.set_axes_mode(is_x_log=is_x_log, sr=self.current_sr)
        self.full_bleed_grid.update()
        self.edge_masks_and_scales.update()
        self.schedule_config_save()

    def audio_callback(self, indata, outdata, frames, time_info, status):
        post_block = self.eq_engine.process_block(indata)

        if self.is_muted:
            outdata.fill(0.0)
        else:
            outdata[:] = post_block

        try:
            self.audio_queue.put_nowait((indata.copy(), post_block.copy()))
        except queue.Full:
            pass

    def restart_audio_stream(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

        in_dev = self.in_combo.currentData()
        out_dev = self.out_combo.currentData()
        if in_dev is None or out_dev is None:
            return

        try:
            self.stream = sd.Stream(
                samplerate=self.current_sr,
                blocksize=self.audio_block_size,
                device=(in_dev, out_dev),
                channels=self.channels,
                dtype="float32",
                callback=self.audio_callback,
            )
            self.stream.start()
        except Exception as e:
            QMessageBox.warning(
                self,
                "Error de Audio",
                f"No se pudo iniciar el stream con los dispositivos/Sample Rate elegidos:\n{e}",
            )

    def update_visuals(self):
        pre_blocks = []
        post_blocks = []
        while not self.audio_queue.empty():
            try:
                pre_blk, post_blk = self.audio_queue.get_nowait()
                pre_blocks.append(pre_blk)
                post_blocks.append(post_blk)
            except queue.Empty:
                break

        # Actualiza curvas EQ (en IDLE retorna en 0 ms gracias a dirty-check)
        self.eq_overlay.update_eq_curves(force=False)
        self.eq_panel.update_live_Readouts()

        if not post_blocks:
            return

        # Actualizar envolvente Min/Max del Osciloscopio con throttling a ~25 FPS (cada 2 ticks)
        scope_changed = self.scope_buffer.push_blocks(post_blocks)
        self._scope_tick_counter = (self._scope_tick_counter + 1) % 2
        if scope_changed and self._scope_tick_counter == 0:
            self.scope_widget.refresh_scope()
            if self.scope_overlay.isVisible():
                self.scope_overlay.update()

        latest_post = post_blocks[-1]
        eps = 1e-10
        peak_l = float(np.max(np.abs(latest_post[:, 0])))
        peak_r = float(np.max(np.abs(latest_post[:, 1]))) if latest_post.shape[1] > 1 else peak_l

        db_l = float(np.clip(20.0 * np.log10(peak_l + eps), -60.0, 0.0))
        db_r = float(np.clip(20.0 * np.log10(peak_r + eps), -60.0, 0.0))
        self.stereo_meter.set_levels(db_l, db_r)

        show_pre = self.show_pre_chk.isChecked()
        show_post = self.show_post_chk.isChecked()
        y_floor_fft = float(self.db_min - 12.0)
        self.fft_analyzer.push_blocks(pre_blocks if show_pre else [], post_blocks if show_post else [])
        self.fft_analyzer.update_spectra(compute_pre=show_pre, compute_post=show_post, db_min=y_floor_fft)

        is_x_log = (self.x_scale_combo.currentText() == "Logarítmica")
        is_y_log = (self.y_scale_combo.currentText() == "Logarítmica (dB)")

        if show_pre:
            if not self.fft_analyzer.pre_at_rest:
                self._pre_drawn_at_rest = False
                x_pre, y_pre = self.fft_analyzer.get_smooth_curve(
                    which="pre", is_x_log=is_x_log, is_y_log=is_y_log,
                    db_min=y_floor_fft, db_max=self.db_max
                )
                self.pre_curve.setData(x_pre, y_pre)
            elif not self._pre_drawn_at_rest:
                self._pre_drawn_at_rest = True
                x_pre, y_pre = self.fft_analyzer.get_smooth_curve(
                    which="pre", is_x_log=is_x_log, is_y_log=is_y_log,
                    db_min=y_floor_fft, db_max=self.db_max
                )
                self.pre_curve.setData(x_pre, y_pre)

        if show_post:
            if not self.fft_analyzer.post_at_rest:
                self._post_drawn_at_rest = False
                x_post, y_post = self.fft_analyzer.get_smooth_curve(
                    which="post", is_x_log=is_x_log, is_y_log=is_y_log,
                    db_min=y_floor_fft, db_max=self.db_max
                )
                self.curve.setData(x_post, y_post)
            elif not self._post_drawn_at_rest:
                self._post_drawn_at_rest = True
                x_post, y_post = self.fft_analyzer.get_smooth_curve(
                    which="post", is_x_log=is_x_log, is_y_log=is_y_log,
                    db_min=y_floor_fft, db_max=self.db_max
                )
                self.curve.setData(x_post, y_post)

    def closeEvent(self, event):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
        if self.config_save_timer.isActive():
            self.config_save_timer.stop()
        self.save_app_config()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioAnalyzerApp()
    window.show()
    sys.exit(app.exec())
