import sys
import os

# En Windows, fijar explícitamente el AppUserModelID al inicio absoluto ANTES de importar
# cualquier subsistema GUI (PyQt6 / OpenGL) o multimedia para que Windows Explorer y
# la barra de tareas asocien siempre el icono personalizado al proceso de la aplicación.
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("bdj.ffteq.audioanalyzer.1.0")
    except Exception:
        pass

import json
import math
import queue
import threading
import numpy as np
import sounddevice as sd
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QLabel, QGroupBox, QMessageBox,
    QPushButton, QSlider, QSpinBox, QDoubleSpinBox, QDialog, QColorDialog,
    QGridLayout, QDialogButtonBox, QMenu, QCheckBox, QFrame
)
from PyQt6.QtCore import QTimer, Qt, QRectF, QPoint, QPointF, pyqtSignal
from PyQt6.QtGui import (
    QColor, QCursor, QPainter, QPainterPath, QPen, QBrush,
    QLinearGradient, QPolygonF, QFont, QFontMetrics, QPixmap, QIcon
)
import pyqtgraph as pg

# Activar aceleración por hardware GPU (OpenGL) para descargar la CPU (Intel UHD Graphics / GPU dedicada)
try:
    import OpenGL  # noqa: F401
    pg.setConfigOptions(useOpenGL=True, enableExperimental=True, antialias=True)
except Exception:
    pg.setConfigOptions(antialias=True)

from eq_ui import (
    ProQ4Overlay, FloatingBandHUD, LiquidGlassButton, style_dark_combobox,
    LimiterControlPopup, LimiterGRMiniWidget, MiniOscilloscopeWidget, CompactScopeGLWidget
)
from fft_analyzer import FFTSpectrumAnalyzer
from eq_dsp import ParametricEQEngine


def get_resource_path(relative_path: str) -> str:
    """Obtiene la ruta absoluta hacia un recurso, compatible con desarrollo y PyInstaller."""
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


def apply_windows_taskbar_icon(hwnd: int, icon_path: str):
    """Asegura que el HWND nativo de Windows reciba explícitamente los iconos grande y pequeño vía Win32 API."""
    if sys.platform != "win32" or not hwnd or not icon_path or not os.path.exists(icon_path):
        return
    try:
        import ctypes
        WM_SETICON = 0x0080
        ICON_SMALL = 0
        ICON_BIG = 1
        IMAGE_ICON = 1
        LR_LOADFROMFILE = 0x00000010
        LR_DEFAULTSIZE = 0x00000040
        abs_path = os.path.abspath(icon_path)

        # Cargar HICON explícito de tamaño nativo/grande (Barra de tareas / Alt-Tab) y 16x16 (Barra de título)
        hicon_big = ctypes.windll.user32.LoadImageW(
            None, abs_path, IMAGE_ICON, 0, 0, LR_LOADFROMFILE | LR_DEFAULTSIZE
        )
        hicon_small = ctypes.windll.user32.LoadImageW(
            None, abs_path, IMAGE_ICON, 16, 16, LR_LOADFROMFILE
        )
        if hicon_big:
            ctypes.windll.user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, hicon_big)
        if hicon_small:
            ctypes.windll.user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, hicon_small)
    except Exception:
        pass


DEFAULT_PRESET_NAME = "Gris Puro de Estudio (Default)"
THEME_PRESETS = {
    "Gris Puro de Estudio (Default)": {
        "background": "#141414", "curve": "#00E5FF", "curve_width": 15,
        "curve_in": "#9E9E9E", "curve_out1": "#00E5FF", "curve_out2": "#00E676",
        "fill": "#6E747E", "fill_alpha": 40,
        "grid": "#2A2A2A", "grid_alpha": 45, "axis_text": "#8C919B",
    },
    "Carbón de Estudio": {
        "background": "#1E1E1E", "curve": "#C5CAD3", "curve_width": 16,
        "curve_in": "#9E9E9E", "curve_out1": "#00E5FF", "curve_out2": "#00E676",
        "fill": "#767C87", "fill_alpha": 42,
        "grid": "#333333", "grid_alpha": 45, "axis_text": "#989EA8",
    },
    "Negro Absoluto": {
        "background": "#080808", "curve": "#00E5FF", "curve_width": 17,
        "curve_in": "#9E9E9E", "curve_out1": "#00E5FF", "curve_out2": "#00E676",
        "fill": "#00ACC1", "fill_alpha": 42,
        "grid": "#222222", "grid_alpha": 45, "axis_text": "#858A94",
    },
    "Azul Estudio": {
        "background": "#0D1017", "curve": "#A0AAB8", "curve_width": 15,
        "curve_in": "#9E9E9E", "curve_out1": "#00E5FF", "curve_out2": "#00E676",
        "fill": "#606B7D", "fill_alpha": 38,
        "grid": "#2D3340", "grid_alpha": 30, "axis_text": "#7A8496",
    },
    "Ámbar Analógico": {
        "background": "#140C02", "curve": "#FFB300", "curve_width": 20,
        "curve_in": "#9E9E9E", "curve_out1": "#FFB300", "curve_out2": "#00E676",
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
THEME_COLOR_KEYS = ("background", "curve", "fill", "grid", "axis_text", "curve_in", "curve_out1", "curve_out2")
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

    if "curve_in" not in theme:
        theme["curve_in"] = "#9E9E9E"
    if "curve_out1" not in theme:
        theme["curve_out1"] = theme.get("curve", "#00E5FF")
    if "curve_out2" not in theme:
        theme["curve_out2"] = "#00E676"

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


SIGNAL_KEYS = [
    "L_in", "R_in", "LR_in",
    "L_outA", "R_outA", "LR_outA",
    "L_outB", "R_outB", "LR_outB",
]

SIGNAL_LABELS = {
    "L_in": "L In (Pre-EQ)",
    "R_in": "R In (Pre-EQ)",
    "LR_in": "L+R In (Pre-EQ)",
    "L_outA": "L Out 1 (Bus A)",
    "R_outA": "R Out 1 (Bus A)",
    "LR_outA": "L+R Out 1 (Bus A)",
    "L_outB": "L Out 2 (Bus B)",
    "R_outB": "R Out 2 (Bus B)",
    "LR_outB": "L+R Out 2 (Bus B)",
}

SHORT_SIGNAL_LABELS = {
    "L_in": "L In",
    "R_in": "R In",
    "LR_in": "L+R In",
    "L_outA": "L Out 1",
    "R_outA": "R Out 1",
    "LR_outA": "L+R Out 1",
    "L_outB": "L Out 2",
    "R_outB": "R Out 2",
    "LR_outB": "L+R Out 2",
}

DEFAULT_SIGNAL_COLORS = {
    "L_in": "#64B5F6",
    "R_in": "#BA68C8",
    "LR_in": "#8E95A4",
    "L_outA": "#00B0FF",
    "R_outA": "#FFB300",
    "LR_outA": "#00E5FF",
    "L_outB": "#69F0AE",
    "R_outB": "#FFD54F",
    "LR_outB": "#00E676",
}

DEFAULT_ACTIVE_SIGNALS = {"LR_in", "LR_outA", "LR_outB"}


class WaveformEnvelopeBuffer:
    """
    Buffer circular ultraliviano de Envolvente Min/Max para 9 canales simultáneos.
    - Por cada bloque de audio entrante (512 muestras, ~10.6 ms), extrae y almacena
      (min, max) para las 9 señales:
      [L_in, R_in, LR_in, L_outA, R_outA, LR_outA, L_outB, R_outB, LR_outB].
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

        self.min_data = np.zeros((9, self.capacity_10s), dtype=np.float32)
        self.max_data = np.zeros((9, self.capacity_10s), dtype=np.float32)
        self.gr_data = np.zeros(self.capacity_10s, dtype=np.float32)

        # Vistas de conveniencia y compatibilidad hacia atrás
        self.min_l = self.min_data[3]
        self.max_l = self.max_data[3]
        self.min_r = self.min_data[4]
        self.max_r = self.max_data[4]
        self.min_m = self.min_data[5]
        self.max_m = self.max_data[5]

        self._silent_blocks = self.capacity_10s + 1
        self.is_at_rest = True
        self.total_pushed = 0

    def set_sample_rate(self, sample_rate: int):
        if int(sample_rate) != self.sample_rate:
            self.sample_rate = int(sample_rate)
            self._allocate_buffers()

    def current_capacity(self) -> int:
        dur = max(1.0, float(self.duration_sec))
        return max(10, int(round((dur / 10.0) * self.capacity_10s)))

    def push_multi_blocks(
        self,
        pre_blocks: list,
        post_blocks: list,
        post2_blocks: list = None,
        gr_db: float = 0.0,
    ) -> bool:
        """
        Procesa bloques entrantes de las 3 fuentes (Pre-EQ, Out 1 / Bus A, Out 2 / Bus B)
        extrayendo la envolvente Min/Max de las 9 señales y el valor de Gain Reduction (GR).
        """
        n_blocks = max(
            len(pre_blocks) if pre_blocks else 0,
            len(post_blocks) if post_blocks else 0,
            len(post2_blocks) if post2_blocks else 0,
        )
        if n_blocks == 0:
            return False

        any_active = False
        for i in range(n_blocks):
            # 1. Pre-EQ (Input)
            if pre_blocks and i < len(pre_blocks) and pre_blocks[i] is not None:
                b_in = pre_blocks[i]
                in_l = b_in[:, 0]
                in_r = b_in[:, 1] if b_in.ndim > 1 and b_in.shape[1] > 1 else in_l
                in_m = (in_l + in_r) * 0.5
            else:
                in_l = in_r = in_m = None

            # 2. Post-EQ 1 (Bus A)
            if post_blocks and i < len(post_blocks) and post_blocks[i] is not None:
                b_out1 = post_blocks[i]
                out1_l = b_out1[:, 0]
                out1_r = b_out1[:, 1] if b_out1.ndim > 1 and b_out1.shape[1] > 1 else out1_l
                out1_m = (out1_l + out1_r) * 0.5
            else:
                out1_l = out1_r = out1_m = None

            # 3. Post-EQ 2 (Bus B)
            if post2_blocks and i < len(post2_blocks) and post2_blocks[i] is not None:
                b_out2 = post2_blocks[i]
                out2_l = b_out2[:, 0]
                out2_r = b_out2[:, 1] if b_out2.ndim > 1 and b_out2.shape[1] > 1 else out2_l
                out2_m = (out2_l + out2_r) * 0.5
            else:
                out2_l = out2_r = out2_m = None

            channels = (in_l, in_r, in_m, out1_l, out1_r, out1_m, out2_l, out2_r, out2_m)
            mins = np.zeros(9, dtype=np.float32)
            maxs = np.zeros(9, dtype=np.float32)
            peak = 0.0

            for ch_idx, ch_arr in enumerate(channels):
                if ch_arr is not None and len(ch_arr) > 0:
                    mn = float(np.min(ch_arr))
                    mx = float(np.max(ch_arr))
                    mins[ch_idx] = mn
                    maxs[ch_idx] = mx
                    p = max(abs(mn), abs(mx))
                    if p > peak:
                        peak = p

            if peak < 1e-4:
                self._silent_blocks += 1
            else:
                self._silent_blocks = 0
                any_active = True

            if self._silent_blocks > self.capacity_10s and not any_active:
                if not self.is_at_rest:
                    self.min_data.fill(0.0)
                    self.max_data.fill(0.0)
                    self.gr_data.fill(0.0)
                    self.is_at_rest = True
                    self.total_pushed += 1
                    return True
                continue

            self.is_at_rest = False
            self.min_data[:, :-1] = self.min_data[:, 1:]
            self.max_data[:, :-1] = self.max_data[:, 1:]
            self.gr_data[:-1] = self.gr_data[1:]

            self.min_data[:, -1] = np.clip(mins, -1.0, 1.0)
            self.max_data[:, -1] = np.clip(maxs, -1.0, 1.0)
            self.gr_data[-1] = float(np.clip(gr_db, -60.0, 0.0))
            self.total_pushed += 1

        return not self.is_at_rest

    def push_blocks(self, blocks: list) -> bool:
        """Compatibilidad con llamadas previas."""
        return self.push_multi_blocks([], blocks, [])

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

    def get_multi_slice(self, count: int | None = None):
        c = self.current_capacity() if count is None else int(count)
        return self.min_data[:, -c:], self.max_data[:, -c:]

    def get_gr_slice(self, count: int | None = None) -> np.ndarray:
        c = self.current_capacity() if count is None else int(count)
        return self.gr_data[-c:]


class StereoDelayLine:
    """
    Línea de retardo circular estéreo de alta precisión y costo cero cuando delay=0 ms.
    Permite compensar latencias de buffers y sincronizar acústicamente múltiples salidas.
    """

    def __init__(self, sample_rate: int = 48000, max_delay_sec: float = 0.250):
        self.sample_rate = max(8000, int(sample_rate))
        self.capacity = int(math.ceil(max_delay_sec * self.sample_rate)) + 8192
        self.buffer = np.zeros((self.capacity, 2), dtype=np.float32)
        self.write_idx = 0
        self.delay_ms = 0.0
        self.delay_samples = 0
        self.lock = threading.Lock()

    def set_delay_ms(self, delay_ms: float) -> None:
        with self.lock:
            self.delay_ms = max(0.0, float(delay_ms))
            self.delay_samples = int(round(self.delay_ms * 0.001 * self.sample_rate))

    def set_sample_rate(self, sample_rate: int) -> None:
        with self.lock:
            sr = max(8000, int(sample_rate))
            if sr != self.sample_rate:
                self.sample_rate = sr
                self.capacity = int(math.ceil(0.250 * self.sample_rate)) + 8192
                self.buffer = np.zeros((self.capacity, 2), dtype=np.float32)
                self.write_idx = 0
                self.delay_samples = int(round(self.delay_ms * 0.001 * self.sample_rate))

    def reset(self) -> None:
        with self.lock:
            self.buffer.fill(0.0)
            self.write_idx = 0

    def process(self, block: np.ndarray) -> np.ndarray:
        if self.delay_samples == 0:
            return block

        with self.lock:
            n_frames = block.shape[0]
            # 1. Escribir el bloque entrante en write_idx
            first_w = min(n_frames, self.capacity - self.write_idx)
            self.buffer[self.write_idx : self.write_idx + first_w] = block[:first_w, :2]
            if n_frames > first_w:
                self.buffer[: n_frames - first_w] = block[first_w:, :2]

            # 2. Leer las muestras retrasadas desde (write_idx - delay_samples)
            read_idx = (self.write_idx - self.delay_samples) % self.capacity
            out = np.empty_like(block)
            first_r = min(n_frames, self.capacity - read_idx)
            out[:first_r] = self.buffer[read_idx : read_idx + first_r]
            if n_frames > first_r:
                out[first_r:] = self.buffer[: n_frames - first_r]

            self.write_idx = (self.write_idx + n_frames) % self.capacity
            return out


class SecondaryOutputRingBuffer:
    """
    Ring buffer continuo thread-safe con colchón anti-jitter y compensación suave de reloj
    (drift correction) para sincronizar un segundo dispositivo de salida de audio en Windows.
    """

    def __init__(self, block_size: int = 512, channels: int = 2):
        self.block_size = max(64, int(block_size))
        self.channels = max(1, int(channels))
        self.capacity_frames = self.block_size * 16
        self.target_fill = self.block_size * 2
        self.buffer = np.zeros((self.capacity_frames, self.channels), dtype=np.float32)
        self.write_ptr = 0
        self.read_ptr = 0
        self.available_frames = 0
        self.is_prebuffering = True
        self.last_frame = np.zeros((1, self.channels), dtype=np.float32)
        self.lock = threading.Lock()

    def reset(self, block_size: int | None = None):
        with self.lock:
            if block_size is not None and block_size > 0:
                self.block_size = int(block_size)
                self.capacity_frames = self.block_size * 16
                self.target_fill = self.block_size * 2
                self.buffer = np.zeros((self.capacity_frames, self.channels), dtype=np.float32)
            else:
                self.buffer.fill(0.0)
            self.write_ptr = 0
            self.read_ptr = 0
            self.available_frames = 0
            self.is_prebuffering = True
            self.last_frame.fill(0.0)

    def write(self, data: np.ndarray) -> None:
        """Escribe un bloque de audio estéreo (N, channels) en el ring buffer."""
        n_frames = data.shape[0]
        if n_frames == 0:
            return

        with self.lock:
            # Si el búfer está lleno (overrun severo), descartar las muestras más viejas
            if self.available_frames + n_frames > self.capacity_frames:
                excess = (self.available_frames + n_frames) - self.capacity_frames
                self.read_ptr = (self.read_ptr + excess) % self.capacity_frames
                self.available_frames -= excess

            ch = min(self.channels, data.shape[1])
            first_chunk = min(n_frames, self.capacity_frames - self.write_ptr)
            second_chunk = n_frames - first_chunk

            self.buffer[self.write_ptr : self.write_ptr + first_chunk, :ch] = data[:first_chunk, :ch]
            if ch < self.channels:
                self.buffer[self.write_ptr : self.write_ptr + first_chunk, ch:] = 0.0

            if second_chunk > 0:
                self.buffer[:second_chunk, :ch] = data[first_chunk:, :ch]
                if ch < self.channels:
                    self.buffer[:second_chunk, ch:] = 0.0

            self.write_ptr = (self.write_ptr + n_frames) % self.capacity_frames
            self.available_frames += n_frames

            # Al acumular el colchón inicial de target_fill, desactivar pre-buffering
            if self.is_prebuffering and self.available_frames >= self.target_fill:
                self.is_prebuffering = False

    def _read_raw(self, count: int) -> np.ndarray:
        """Lee `count` muestras del buffer circular (debe llamarse dentro de self.lock)."""
        chunk = np.empty((count, self.channels), dtype=np.float32)
        first = min(count, self.capacity_frames - self.read_ptr)
        second = count - first

        chunk[:first] = self.buffer[self.read_ptr : self.read_ptr + first]
        if second > 0:
            chunk[first:] = self.buffer[:second]

        self.read_ptr = (self.read_ptr + count) % self.capacity_frames
        self.available_frames -= count
        return chunk

    def read(self, outdata: np.ndarray, frames: int) -> None:
        """Lee y entrega `frames` muestras a `outdata`, aplicando compensación suave de deriva de reloj."""
        with self.lock:
            # 1. Fase de Pre-buffering: silenciar mientras se llena el colchón inicial anti-jitter
            if self.is_prebuffering:
                outdata.fill(0.0)
                return

            # 2. Compensación por Exceso de Reloj (available_frames > block_size * 8):
            #    Leemos frames + 2 muestras y las comprimimos suavemente a `frames` sin cortes ni saltos
            if self.available_frames >= frames + 2 and self.available_frames > self.block_size * 8:
                n_read = frames + 2
                raw = self._read_raw(n_read)
                x_src = np.linspace(0.0, 1.0, n_read, endpoint=True)
                x_dst = np.linspace(0.0, 1.0, frames, endpoint=True)
                ch = min(self.channels, outdata.shape[1])
                for c in range(ch):
                    outdata[:, c] = np.interp(x_dst, x_src, raw[:, c])
                if outdata.shape[1] > ch:
                    outdata[:, ch:] = 0.0
                self.last_frame[:] = outdata[-1:, :self.channels]
                return

            # 3. Lectura Normal (available_frames >= frames)
            if self.available_frames >= frames:
                raw = self._read_raw(frames)
                ch = min(self.channels, outdata.shape[1])
                outdata[:, :ch] = raw[:, :ch]
                if outdata.shape[1] > ch:
                    outdata[:, ch:] = 0.0
                self.last_frame[:] = outdata[-1:, :self.channels]
                return

            # 4. Compensación por Defecto de Reloj / Underrun (available_frames < frames):
            #    NUNCA silenciar con escalón brusco; estirar muestras o atenuar suavemente
            avail = self.available_frames
            if avail > 1:
                raw = self._read_raw(avail)
                x_src = np.linspace(0.0, 1.0, avail, endpoint=True)
                x_dst = np.linspace(0.0, 1.0, frames, endpoint=True)
                ch = min(self.channels, outdata.shape[1])
                for c in range(ch):
                    outdata[:, c] = np.interp(x_dst, x_src, raw[:, c])
                if outdata.shape[1] > ch:
                    outdata[:, ch:] = 0.0
                self.last_frame[:] = outdata[-1:, :self.channels]
            elif avail == 1:
                raw = self._read_raw(1)
                fade = np.linspace(1.0, 0.0, frames, endpoint=True)[:, None]
                ch = min(self.channels, outdata.shape[1])
                outdata[:, :ch] = raw * fade
                if outdata.shape[1] > ch:
                    outdata[:, ch:] = 0.0
                self.last_frame.fill(0.0)
            else:
                fade = np.linspace(0.8, 0.0, frames, endpoint=True)[:, None]
                ch = min(self.channels, outdata.shape[1])
                outdata[:, :ch] = self.last_frame * fade
                if outdata.shape[1] > ch:
                    outdata[:, ch:] = 0.0
                self.last_frame.fill(0.0)
                self.is_prebuffering = True


# Compatibilidad hacia atrás
_CompactScopeCanvas = MiniOscilloscopeWidget
CompactScopeGLWidget = MiniOscilloscopeWidget


class WaveformScopeWidget(QWidget):
    """
    Vista Compacta Integrada en la Top Bar Flotante con estética Apple Liquid Glass.
    Incluye Mini-Osciloscopio a 60 FPS reales y 0% CPU con Scroll Blitting de 1 píxel,
    control de tiempo (1s / 5s / 10s) y resplandor 'LED' de alerta del Limitador Master.
    """

    toggleExpandedRequested = pyqtSignal()
    durationChanged = pyqtSignal(int)

    def __init__(self, env_buffer: WaveformEnvelopeBuffer, parent=None):
        super().__init__(parent)
        self.env_buffer = env_buffer
        self._limit_glow_alpha: float = 0.0
        self.current_duration_sec: int = 10
        self.setFixedSize(338, 30)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        row = QHBoxLayout(self)
        row.setContentsMargins(7, 3, 6, 3)
        row.setSpacing(4)

        self.mini_scope = MiniOscilloscopeWidget(self)
        self.mini_scope.setFixedSize(240, 22)
        self.mini_scope.set_time_window(10.0)
        self.mini_scope.clicked.connect(self.toggleExpandedRequested.emit)
        self.canvas = self.mini_scope  # compat alias
        row.addWidget(self.mini_scope)

        self.dur_btn = LiquidGlassButton("10s", self, pill_radius=10.0)
        self.dur_btn.setFixedSize(36, 22)
        self.dur_btn.setToolTip("Alternar tiempo del osciloscopio: 1s -> 5s -> 10s")
        self.dur_btn.clicked.connect(self._cycle_duration)
        self.btn_scope_time = self.dur_btn  # compat alias
        row.addWidget(self.dur_btn)

        self.expand_btn = LiquidGlassButton("▾", self, active_tint_color="#00E5FF", pill_radius=10.0)
        self.expand_btn.setFixedSize(24, 22)
        self.expand_btn.setToolTip("Desplegar / Plegar Osciloscopio detallado")
        self.expand_btn.clicked.connect(self.toggleExpandedRequested.emit)
        row.addWidget(self.expand_btn)

    def update_gr(self, gr_db: float):
        """Actualiza el nivel de reducción de ganancia y modula el glow del LED analógico."""
        if gr_db < -0.2:
            self._limit_glow_alpha = 1.0
        else:
            self._limit_glow_alpha = max(0.0, self._limit_glow_alpha - 0.08)

        self.mini_scope.set_current_gr(gr_db)
        if self._limit_glow_alpha > 0.0:
            self.update()

    def _cycle_duration(self):
        """Alterna cíclicamente el tiempo entre 1s -> 5s -> 10s -> 1s."""
        cycle_map = {1: 5, 5: 10, 10: 1}
        new_dur = cycle_map.get(self.current_duration_sec, 10)
        self.set_duration(new_dur)
        self.durationChanged.emit(new_dur)

    def set_duration(self, sec: int):
        sec = int(sec)
        if sec not in (1, 3, 5, 10):
            sec = 10
        self.current_duration_sec = sec
        self.dur_btn.setText(f"{sec}s")
        self.mini_scope.set_time_window(float(sec))
        if hasattr(self, "env_buffer") and self.env_buffer is not None:
            self.env_buffer.duration_sec = sec

    def set_expanded_indicator(self, expanded: bool):
        self.expand_btn.setText("▴" if expanded else "▾")
        self.expand_btn.setChecked(expanded)

    def refresh_scope(self):
        self.mini_scope.refresh_scope()
        if self._limit_glow_alpha > 0.0:
            self.update()

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

        # "LED" de Alerta: Resplandor (Glow) en la cápsula Liquid Glass
        if self._limit_glow_alpha > 0.0:
            glow_pen = QPen(QColor(213, 0, 249, int(150 * self._limit_glow_alpha)), 1.8)
            painter.setPen(glow_pen)
            painter.setBrush(QBrush(QColor(213, 0, 249, int(35 * self._limit_glow_alpha))))
            painter.drawRoundedRect(rect, 14.0, 14.0)

        spec_grad = QLinearGradient(rect.left(), rect.top(), rect.right(), rect.bottom())
        spec_grad.setColorAt(0.0, QColor(255, 255, 255, 115))
        spec_grad.setColorAt(0.5, QColor(255, 255, 255, 32))
        spec_grad.setColorAt(1.0, QColor(255, 255, 255, 16))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QBrush(spec_grad), 1.2))
        painter.drawRoundedRect(rect, 14.0, 14.0)


class ScopeSignalSettingsPopup(QWidget):
    """
    Popup flotante no-modal para seleccionar qué señales se visualizan en el osciloscopio desplegado,
    personalizar sus colores (vía QColorDialog) y ajustar la opacidad global de las curvas.
    """

    def __init__(self, overlay_widget: "WaveformOverlayWidget", parent=None):
        super().__init__(parent, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.overlay = overlay_widget
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setFixedWidth(460)
        self.setStyleSheet("""
            QWidget {
                background-color: #181A20;
                color: #ECEFF4;
                font-family: "Segoe UI", sans-serif;
                font-size: 11px;
            }
            QCheckBox {
                color: #D8DEE9;
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 13px;
                height: 13px;
                border: 1px solid #3E4452;
                border-radius: 3px;
                background-color: #21252B;
            }
            QCheckBox::indicator:checked {
                background-color: #00E5FF;
                border: 1px solid #00B0FF;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #2B303C;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #00E5FF;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #FFFFFF;
                border: 1px solid #00B0FF;
                width: 12px;
                margin-top: -4px;
                margin-bottom: -4px;
                border-radius: 6px;
            }
            QPushButton#actionBtn {
                background-color: #262B34;
                color: #C0C5D0;
                border: 1px solid #363C48;
                border-radius: 4px;
                padding: 3px 10px;
                font-size: 10px;
            }
            QPushButton#actionBtn:hover {
                background-color: #323846;
                color: #FFFFFF;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        # Encabezado
        hdr_row = QHBoxLayout()
        title_lbl = QLabel("Configuración de Señales y Opacidad")
        title_lbl.setStyleSheet("font-weight: bold; font-size: 11px; color: #E5E9F0;")
        hdr_row.addWidget(title_lbl)
        hdr_row.addStretch(1)
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(18, 18)
        close_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #8C919B;
                border: none;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                color: #FF5252;
            }
        """)
        close_btn.clicked.connect(self.close)
        hdr_row.addWidget(close_btn)
        layout.addLayout(hdr_row)

        # 3 Columnas: Entrada Pre-EQ, Salida 1 (Bus A), Salida 2 (Bus B)
        cols_row = QHBoxLayout()
        cols_row.setSpacing(10)

        groups_data = [
            ("Entrada (Pre-EQ)", ["L_in", "R_in", "LR_in"]),
            ("Salida 1 (Bus A)", ["L_outA", "R_outA", "LR_outA"]),
            ("Salida 2 (Bus B)", ["L_outB", "R_outB", "LR_outB"]),
        ]

        self.checkboxes = {}
        self.swatches = {}

        for group_title, keys in groups_data:
            col_box = QVBoxLayout()
            col_box.setSpacing(5)
            grp_lbl = QLabel(group_title)
            grp_lbl.setStyleSheet("color: #7E8594; font-weight: bold; font-size: 10px;")
            col_box.addWidget(grp_lbl)

            for key in keys:
                item_row = QHBoxLayout()
                item_row.setSpacing(6)
                chk = QCheckBox(SHORT_SIGNAL_LABELS[key])
                chk.setChecked(key in self.overlay.active_signals)
                chk.toggled.connect(lambda chked, k=key: self._on_chk_toggled(k, chked))
                self.checkboxes[key] = chk
                item_row.addWidget(chk, 1)

                swatch = QPushButton()
                swatch.setFixedSize(20, 15)
                swatch.setCursor(Qt.CursorShape.PointingHandCursor)
                col_hex = self.overlay.signal_colors.get(key, "#FFFFFF")
                swatch.setStyleSheet(f"background-color: {col_hex}; border: 1px solid #444C5C; border-radius: 3px;")
                swatch.clicked.connect(lambda _, k=key, s=swatch: self._on_pick_color(k, s))
                self.swatches[key] = swatch
                item_row.addWidget(swatch)

                col_box.addLayout(item_row)

            cols_row.addLayout(col_box)

        layout.addLayout(cols_row)

        # Separador horizontal
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #2A2E38; margin: 2px 0;")
        layout.addWidget(sep)

        # Fila de Opacidad Global
        opac_row = QHBoxLayout()
        opac_row.setSpacing(8)
        opac_lbl = QLabel("Opacidad de curvas:")
        opac_lbl.setStyleSheet("color: #D8DEE9; font-size: 10px;")
        opac_row.addWidget(opac_lbl)

        self.opac_slider = QSlider(Qt.Orientation.Horizontal)
        self.opac_slider.setRange(15, 100)
        self.opac_slider.setValue(self.overlay.opacity_pct)
        self.opac_val_lbl = QLabel(f"{self.overlay.opacity_pct}%")
        self.opac_val_lbl.setFixedWidth(34)
        self.opac_val_lbl.setStyleSheet("color: #00E5FF; font-weight: bold; font-size: 10px;")

        self.opac_slider.valueChanged.connect(self._on_opacity_slider_changed)
        opac_row.addWidget(self.opac_slider, 1)
        opac_row.addWidget(self.opac_val_lbl)
        layout.addLayout(opac_row)

        # Botones inferiores: Restaurar por defecto y Cerrar
        bottom_row = QHBoxLayout()
        reset_btn = QPushButton("Restaurar por defecto")
        reset_btn.setObjectName("actionBtn")
        reset_btn.clicked.connect(self._reset_defaults)
        bottom_row.addWidget(reset_btn)
        bottom_row.addStretch(1)

        done_btn = QPushButton("Listo")
        done_btn.setObjectName("actionBtn")
        done_btn.clicked.connect(self.close)
        bottom_row.addWidget(done_btn)
        layout.addLayout(bottom_row)

    def _on_chk_toggled(self, key: str, checked: bool):
        if checked:
            self.overlay.active_signals.add(key)
        else:
            self.overlay.active_signals.discard(key)
        self.overlay.invalidate_cache()
        if hasattr(self.overlay, "app_ref") and self.overlay.app_ref:
            self.overlay.app_ref.schedule_config_save()

    def _on_pick_color(self, key: str, swatch_btn: QPushButton):
        curr_hex = self.overlay.signal_colors.get(key, "#FFFFFF")
        dlg = QColorDialog(QColor(curr_hex), self)
        dlg.setOption(QColorDialog.ColorDialogOption.DontUseNativeDialog, True)
        if dlg.exec():
            new_hex = dlg.selectedColor().name()
            self.overlay.signal_colors[key] = new_hex
            swatch_btn.setStyleSheet(f"background-color: {new_hex}; border: 1px solid #444C5C; border-radius: 3px;")
            self.overlay.invalidate_cache()
            if hasattr(self.overlay, "app_ref") and self.overlay.app_ref:
                self.overlay.app_ref.schedule_config_save()

    def _on_opacity_slider_changed(self, val: int):
        self.opac_val_lbl.setText(f"{val}%")
        self.overlay.set_opacity_pct(val)
        if hasattr(self.overlay, "app_ref") and self.overlay.app_ref:
            self.overlay.app_ref.schedule_config_save()

    def _reset_defaults(self):
        self.overlay.reset_signals_to_default()
        for k, chk in self.checkboxes.items():
            chk.blockSignals(True)
            chk.setChecked(k in self.overlay.active_signals)
            chk.blockSignals(False)
        for k, sw in self.swatches.items():
            col = self.overlay.signal_colors.get(k, "#FFFFFF")
            sw.setStyleSheet(f"background-color: {col}; border: 1px solid #444C5C; border-radius: 3px;")
        self.opac_slider.blockSignals(True)
        self.opac_slider.setValue(self.overlay.opacity_pct)
        self.opac_slider.blockSignals(False)
        self.opac_val_lbl.setText(f"{self.overlay.opacity_pct}%")
        self.overlay.invalidate_cache()
        if hasattr(self.overlay, "app_ref") and self.overlay.app_ref:
            self.overlay.app_ref.schedule_config_save()


class WaveformOverlayWidget(QWidget):
    """
    Panel desplegable del Osciloscopio con Aislamiento OpenGL y Scroll Blitting (`QPixmap.scroll`):
    1. Fondo 100% opaco (`#101216`) con `WA_OpaquePaintEvent = True` para evitar solapamientos
       o transparencias costosas sobre el lienzo OpenGL.
    2. Scroll Blitting multi-canal (< 1% CPU): Desplaza `self.wave_canvas` hacia la izquierda `dx`
       píxeles y dibuja únicamente la franja nueva de ancho `dx` para todas las señales activadas.
    3. Selector interactivo `⚙ Señales y Color ▾` y control de opacidad en la cabecera.
    """

    collapseRequested = pyqtSignal()

    def __init__(self, env_buffer: WaveformEnvelopeBuffer, parent=None, app_ref=None):
        super().__init__(parent)
        self.env_buffer = env_buffer
        self.app_ref = app_ref
        self.wave_canvas: QPixmap | None = None
        self._hud_overlay_cache: QPixmap | None = None
        self._last_pushed_seq: int = -1
        self._px_accum: float = 0.0
        self._cached_dur: int = -1

        self.active_signals = set(DEFAULT_ACTIVE_SIGNALS)
        self.signal_colors = dict(DEFAULT_SIGNAL_COLORS)
        self.opacity_pct = 75
        self.opacity_alpha = int(75 * 255 / 100)
        self.settings_popup: ScopeSignalSettingsPopup | None = None
        self.limiter_popup: LimiterControlPopup | None = None

        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setFixedHeight(164)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Clic para plegar el osciloscopio de vuelta a la barra superior")

        self.settings_btn = QPushButton("⚙ Señales y Color ▾", self)
        self.settings_btn.setFixedSize(126, 20)
        self.settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.settings_btn.setStyleSheet("""
            QPushButton {
                background-color: #20242C;
                color: #D8DEE9;
                border: 1px solid #363C48;
                border-radius: 9px;
                padding: 1px 6px;
                font-size: 10px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #2B303C;
                border: 1px solid #4E5666;
                color: #FFFFFF;
            }
            QPushButton:pressed {
                background-color: #1A1D24;
            }
        """)
        self.settings_btn.clicked.connect(self._toggle_settings_popup)

        self.limiter_btn = QPushButton("🛡 Limitador ▾", self)
        self.limiter_btn.setFixedSize(116, 20)
        self.limiter_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.limiter_btn.setStyleSheet("""
            QPushButton {
                background-color: #20242C;
                color: #EA80FC;
                border: 1px solid #7B1FA2;
                border-radius: 9px;
                padding: 1px 6px;
                font-size: 10px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #2E1B38;
                border: 1px solid #D500F9;
                color: #FFFFFF;
            }
            QPushButton:pressed {
                background-color: #1A1D24;
            }
        """)
        self.limiter_btn.clicked.connect(self._toggle_limiter_popup)

        self.gr_meter_widget = LimiterGRMiniWidget(self)
        self.gr_meter_widget.clicked.connect(self._toggle_limiter_popup)

        self.dur_btn = QPushButton(f"{self.env_buffer.duration_sec}s", self)
        self.dur_btn.setFixedSize(38, 20)
        self.dur_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.dur_btn.setToolTip("Alternar ventana de tiempo: 1s -> 5s -> 10s")
        self.dur_btn.setStyleSheet("""
            QPushButton {
                background-color: #20242C;
                color: #00E5FF;
                border: 1px solid #363C48;
                border-radius: 9px;
                padding: 1px 6px;
                font-size: 10px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #2B303C;
                border: 1px solid #00E5FF;
                color: #FFFFFF;
            }
            QPushButton:pressed {
                background-color: #1A1D24;
            }
        """)
        self.dur_btn.clicked.connect(self._cycle_duration)

        self.hide()

    def _cycle_duration(self):
        if self.app_ref and hasattr(self.app_ref, "scope_widget"):
            self.app_ref.scope_widget._cycle_duration()

    def set_opacity_pct(self, pct: int):
        self.opacity_pct = int(np.clip(pct, 15, 100))
        self.opacity_alpha = int(self.opacity_pct * 255 / 100)
        self.invalidate_cache()

    def reset_signals_to_default(self):
        self.active_signals = set(DEFAULT_ACTIVE_SIGNALS)
        self.signal_colors = dict(DEFAULT_SIGNAL_COLORS)
        self.set_opacity_pct(75)

    def set_active_signals(self, signals):
        self.active_signals = set(s for s in signals if s in SIGNAL_KEYS)
        self.invalidate_cache()

    def set_signal_colors(self, colors: dict):
        for k, col in colors.items():
            if k in SIGNAL_KEYS and isinstance(col, str):
                self.signal_colors[k] = col
        self.invalidate_cache()

    def _toggle_settings_popup(self):
        if self.settings_popup is not None and self.settings_popup.isVisible():
            self.settings_popup.close()
            return
        if self.settings_popup is None:
            self.settings_popup = ScopeSignalSettingsPopup(self)
        pos = self.settings_btn.mapToGlobal(QPoint(0, self.settings_btn.height() + 2))
        self.settings_popup.move(pos)
        self.settings_popup.show()
        self.settings_popup.raise_()

    def _toggle_limiter_popup(self):
        if self.limiter_popup is not None and self.limiter_popup.isVisible():
            self.limiter_popup.close()
            return
        if self.limiter_popup is None:
            eq_engine = self.app_ref.eq_engine if (self.app_ref and hasattr(self.app_ref, "eq_engine")) else None
            self.limiter_popup = LimiterControlPopup(eq_engine=eq_engine, overlay_ref=self, app_ref=self.app_ref, parent=self)
        pos = self.limiter_btn.mapToGlobal(QPoint(0, self.limiter_btn.height() + 2))
        self.limiter_popup.move(pos)
        self.limiter_popup.show()
        self.limiter_popup.raise_()

    def update_gr(self, gr_db: float):
        if hasattr(self, "gr_meter_widget"):
            self.gr_meter_widget.set_gr(gr_db)
        if self.limiter_popup is not None and self.limiter_popup.isVisible():
            self.limiter_popup.set_live_gr(gr_db)

    def invalidate_cache(self):
        self.wave_canvas = None
        self._hud_overlay_cache = None
        if self.isVisible():
            self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.wave_canvas = None
        self._hud_overlay_cache = None
        if hasattr(self, "settings_btn"):
            self.settings_btn.move(40, 2)
            self.settings_btn.raise_()
        if hasattr(self, "limiter_btn"):
            self.limiter_btn.move(172, 2)
            self.limiter_btn.raise_()
        if hasattr(self, "gr_meter_widget"):
            self.gr_meter_widget.move(294, 2)
            self.gr_meter_widget.raise_()
        if hasattr(self, "dur_btn"):
            self.dur_btn.move(382, 2)
            self.dur_btn.raise_()

    def showEvent(self, event):
        super().showEvent(event)
        self.wave_canvas = None
        if hasattr(self, "settings_btn"):
            self.settings_btn.move(40, 2)
            self.settings_btn.raise_()
        if hasattr(self, "limiter_btn"):
            self.limiter_btn.move(172, 2)
            self.limiter_btn.raise_()
        if hasattr(self, "gr_meter_widget"):
            self.gr_meter_widget.move(294, 2)
            self.gr_meter_widget.raise_()
        if hasattr(self, "dur_btn"):
            self.dur_btn.move(382, 2)
            self.dur_btn.raise_()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if hasattr(self, "settings_btn") and self.settings_btn.geometry().contains(event.pos()):
                event.accept()
                return
            if hasattr(self, "limiter_btn") and self.limiter_btn.geometry().contains(event.pos()):
                event.accept()
                return
            if hasattr(self, "gr_meter_widget") and self.gr_meter_widget.geometry().contains(event.pos()):
                event.accept()
                return
            if hasattr(self, "dur_btn") and self.dur_btn.geometry().contains(event.pos()):
                event.accept()
                return
            if self.settings_popup and self.settings_popup.isVisible():
                self.settings_popup.close()
            if self.limiter_popup and self.limiter_popup.isVisible():
                self.limiter_popup.close()
            self.collapseRequested.emit()
        event.accept()

    def wheelEvent(self, event):
        event.accept()

    def _draw_strip_bg(self, p: QPainter, x0: int, strip_w: int, h: int):
        p.fillRect(x0, 0, strip_w, h, QColor("#101216"))
        pad_t = 24
        pad_b = 20
        plot_h = max(10, h - pad_t - pad_b)
        mid_y = pad_t + plot_h // 2
        half_h = plot_h * 0.5

        p.setPen(QPen(QColor("#1F232B"), 1))
        for level in (-0.5, 0.5):
            y_g = int(round(mid_y - level * half_h))
            p.drawLine(x0, y_g, x0 + strip_w, y_g)

        p.setPen(QPen(QColor("#3A404D"), 1))
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
        pad_t = 24
        pad_b = 20
        plot_w = max(10, w - pad_l - pad_r)
        plot_h = max(10, h - pad_t - pad_b)
        mid_y = pad_t + plot_h * 0.5
        half_h = plot_h * 0.5

        # Máscaras laterales 100% opacas (#101216) y barras de leyenda
        bg_col = QColor("#101216")
        p.fillRect(0, 0, pad_l, h, bg_col)
        p.fillRect(w - pad_r, 0, pad_r, h, bg_col)
        p.fillRect(0, 0, w, pad_t, bg_col)
        p.fillRect(0, h - pad_b, w, pad_b, bg_col)

        # Borde exterior y marco del área de onda
        p.setPen(QPen(QColor("#2A2E38"), 1))
        p.drawRect(0, 0, w - 1, h - 1)
        p.drawRect(pad_l, pad_t, plot_w, plot_h)

        # Línea punteada horizontal del Threshold estilo Fruity Limiter
        thr_db = 0.0
        limiter_enabled = True
        if self.app_ref and hasattr(self.app_ref, "eq_engine") and hasattr(self.app_ref.eq_engine, "limiter_a"):
            thr_db = getattr(self.app_ref.eq_engine.limiter_a, "threshold_db", 0.0)
            limiter_enabled = getattr(self.app_ref.eq_engine.limiter_a, "enabled", True)

        if limiter_enabled:
            thr_lin = min(1.0, max(0.01, 10.0 ** (thr_db / 20.0)))
            y_thr_top = int(round(mid_y - thr_lin * half_h))
            y_thr_bot = int(round(mid_y + thr_lin * half_h))

            pen_thr = QPen(QColor(213, 0, 249, 140), 1, Qt.PenStyle.DashLine)
            p.setPen(pen_thr)
            p.drawLine(pad_l, y_thr_top, pad_l + plot_w, y_thr_top)
            p.drawLine(pad_l, y_thr_bot, pad_l + plot_w, y_thr_bot)

            font_thr = QFont("Segoe UI", 7, QFont.Weight.DemiBold)
            p.setFont(font_thr)
            p.setPen(QColor("#D500F9"))
            p.drawText(QRectF(pad_l + 4.0, max(float(pad_t), y_thr_top - 12.0), 65.0, 11.0), Qt.AlignmentFlag.AlignLeft, f"THR {thr_db:.1f} dB")

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

        # Badges de señales activas junto a los botones
        cur_x = 428.0
        font_badge = QFont("Segoe UI", 8, QFont.Weight.DemiBold)
        p.setFont(font_badge)
        fm = QFontMetrics(font_badge)
        for sig_key in SIGNAL_KEYS:
            if sig_key in self.active_signals:
                sig_col = QColor(self.signal_colors.get(sig_key, "#FFFFFF"))
                badge_txt = SHORT_SIGNAL_LABELS.get(sig_key, sig_key)
                txt_w = fm.horizontalAdvance(badge_txt) + 16.0
                if cur_x + txt_w > w - pad_r - 210.0:
                    break
                p.setPen(sig_col)
                p.drawText(QRectF(cur_x, 3.0, 10.0, 18.0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, "●")
                p.setPen(QColor("#C0C5D0"))
                p.drawText(QRectF(cur_x + 11.0, 3.0, txt_w - 11.0, 18.0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, badge_txt)
                cur_x += txt_w + 6.0

        # Aviso de plegado a la derecha
        p.setPen(QColor("#8E95A4"))
        p.drawText(
            QRectF(w - pad_r - 210.0, 3.0, 210.0, 18.0),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
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

        min_full, max_full = self.env_buffer.get_multi_slice()
        gr_full = self.env_buffer.get_gr_slice()
        n_raw = min_full.shape[1]
        if n_raw >= 2 and not self.env_buffer.is_at_rest:
            pad_l = 38
            pad_r = 38
            pad_t = 24
            pad_b = 20
            plot_w = max(10, w - pad_l - pad_r)
            plot_h = max(10, h - pad_t - pad_b)
            mid_y = int(pad_t + plot_h * 0.5)
            half_h = plot_h * 0.5

            n_pts = min(360, max(32, int(plot_w) // 2), n_raw)
            idx = np.linspace(0, n_raw - 1, n_pts, dtype=np.int32)
            xs = np.linspace(pad_l, pad_l + plot_w, n_pts)

            # Fruity Limiter Gain Reduction envelope
            thr_db = 0.0
            limiter_enabled = True
            if self.app_ref and hasattr(self.app_ref, "eq_engine") and hasattr(self.app_ref.eq_engine, "limiter_a"):
                thr_db = getattr(self.app_ref.eq_engine.limiter_a, "threshold_db", 0.0)
                limiter_enabled = getattr(self.app_ref.eq_engine.limiter_a, "enabled", True)

            if limiter_enabled and gr_full is not None and len(gr_full) >= n_raw:
                thr_lin = min(1.0, max(0.01, 10.0 ** (thr_db / 20.0)))
                y_thr_top = int(round(mid_y - thr_lin * half_h))
                y_thr_bot = int(round(mid_y + thr_lin * half_h))
                gr_pts = gr_full[idx]

                p.setPen(QPen(QColor(213, 0, 249, 190), 1))
                for i in range(n_pts):
                    gr_val = float(gr_pts[i])
                    if gr_val < -0.05:
                        comp_lin = min(1.0, max(0.001, 10.0 ** ((thr_db + gr_val) / 20.0)))
                        y_ct = int(round(mid_y - comp_lin * half_h))
                        y_cb = int(round(mid_y + comp_lin * half_h))
                        xv = int(round(float(xs[i])))
                        p.drawLine(xv, y_thr_top, xv, y_ct)
                        p.drawLine(xv, y_thr_bot, xv, y_cb)

            if len(self.active_signals) > 0:
                for sig_key in SIGNAL_KEYS:
                    if sig_key not in self.active_signals:
                        continue
                    sig_idx = SIGNAL_KEYS.index(sig_key)
                    min_ch = min_full[sig_idx, idx]
                    max_ch = max_full[sig_idx, idx]

                    col_q = QColor(self.signal_colors.get(sig_key, "#FFFFFF"))
                    col_q.setAlpha(self.opacity_alpha)
                    p.setPen(QPen(col_q, 1))

                    for i in range(n_pts):
                        xv = int(round(float(xs[i])))
                        y_top = int(round(mid_y - max_ch[i] * half_h))
                        y_bot = int(round(mid_y - min_ch[i] * half_h))
                        p.drawLine(xv, y_top, xv, max(y_top + 1, y_bot))

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
            return

        w = self.wave_canvas.width()
        h = self.wave_canvas.height()
        pad_l = 38
        pad_r = 38
        pad_t = 24
        pad_b = 20
        plot_w = max(10, w - pad_l - pad_r)
        plot_h = max(10, h - pad_t - pad_b)
        mid_y = int(pad_t + plot_h * 0.5)
        half_h = plot_h * 0.5

        cap = max(1, self.env_buffer.current_capacity())
        if new_blocks >= cap // 2:
            new_blocks = min(new_blocks, max(1, cap // 8))

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

        n_slice = min(self.env_buffer.min_data.shape[1], new_blocks + 1)
        min_slice = self.env_buffer.min_data[:, -n_slice:]
        max_slice = self.env_buffer.max_data[:, -n_slice:]
        gr_slice = self.env_buffer.get_gr_slice(n_slice)

        # Fruity Limiter Gain Reduction envelope para la franja desplazada
        thr_db = 0.0
        limiter_enabled = True
        if self.app_ref and hasattr(self.app_ref, "eq_engine") and hasattr(self.app_ref.eq_engine, "limiter_a"):
            thr_db = getattr(self.app_ref.eq_engine.limiter_a, "threshold_db", 0.0)
            limiter_enabled = getattr(self.app_ref.eq_engine.limiter_a, "enabled", True)

        if limiter_enabled and gr_slice is not None and len(gr_slice) > 0:
            thr_lin = min(1.0, max(0.01, 10.0 ** (thr_db / 20.0)))
            y_thr_top = int(round(mid_y - thr_lin * half_h))
            y_thr_bot = int(round(mid_y + thr_lin * half_h))
            p.setPen(QPen(QColor(213, 0, 249, 190), 1))

            for col in range(dx):
                frac = col / float(max(1, dx - 1)) if dx > 1 else 1.0
                si = int(round(frac * (n_slice - 1)))
                gr_val = float(gr_slice[si])
                if gr_val < -0.05:
                    comp_lin = min(1.0, max(0.001, 10.0 ** ((thr_db + gr_val) / 20.0)))
                    y_ct = int(round(mid_y - comp_lin * half_h))
                    y_cb = int(round(mid_y + comp_lin * half_h))
                    xv = x0 + col
                    p.drawLine(xv, y_thr_top, xv, y_ct)
                    p.drawLine(xv, y_thr_bot, xv, y_cb)

        if len(self.active_signals) > 0:
            for sig_key in SIGNAL_KEYS:
                if sig_key not in self.active_signals:
                    continue
                sig_idx = SIGNAL_KEYS.index(sig_key)
                min_seg = min_slice[sig_idx]
                max_seg = max_slice[sig_idx]

                col_q = QColor(self.signal_colors.get(sig_key, "#FFFFFF"))
                col_q.setAlpha(self.opacity_alpha)
                p.setPen(QPen(col_q, 1))

                for col in range(dx):
                    frac = col / float(max(1, dx - 1)) if dx > 1 else 1.0
                    si = int(round(frac * (n_slice - 1)))
                    xv = x0 + col
                    y1 = int(round(mid_y - max_seg[si] * half_h))
                    y2 = int(round(mid_y - min_seg[si] * half_h))
                    p.drawLine(xv, y1, xv, max(y1 + 1, y2))

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
        ("curve_in", "Curva Input (Pre-EQ)"),
        ("curve_out1", "Curva Output 1 (Bus A)"),
        ("curve_out2", "Curva Output 2 (Bus B)"),
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
        style_dark_combobox(self.preset_combo)
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
        if key == "curve_out1":
            self.theme["curve"] = hex_color
        self._style_swatch(key)
        self._sync_preset_combo()
        self.on_change()

    def _style_swatch(self, key):
        color = QColor(self.theme.get(key, "#9E9E9E"))
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
                30000, 40000, 60000, 70000, 80000,
            ]
            painter.setPen(pen_minor)
            for f in minor_freqs:
                if f <= nyquist:
                    x_s = vb.mapViewToScene(QPointF(float(np.log10(f)), 0.0)).x()
                    if rect.left() <= x_s <= rect.right():
                        painter.drawLine(QPointF(x_s, rect.top()), QPointF(x_s, rect.bottom()))

            major_freqs = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 90000]
            painter.setPen(pen_major)
            for f in major_freqs:
                if f <= nyquist or (f == 90000 and self.app_ref.current_sr >= 176400):
                    f_draw = min(float(f), nyquist * 0.996)
                    x_s = vb.mapViewToScene(QPointF(float(np.log10(f_draw)), 0.0)).x()
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

        max_freq_x = rect.right() - 10.0
        if is_x_log:
            freq_marks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 90000]
            for f in freq_marks:
                if f <= nyquist or (f == 90000 and self.app_ref.current_sr >= 176400):
                    f_draw = min(float(f), nyquist * 0.996)
                    x_s = vb.mapViewToScene(QPointF(float(np.log10(f_draw)), 0.0)).x()
                    if rect.left() + 14.0 <= x_s <= max_freq_x:
                        lbl = f"{int(f // 1000)}k" if f >= 1000 else str(f)
                        x_text = min(x_s - 20.0, rect.right() - 42.0)
                        painter.drawText(
                            QRectF(x_text, rect.bottom() - 19.0, 40.0, 16.0),
                            Qt.AlignmentFlag.AlignCenter,
                            lbl,
                        )
        else:
            step = 2000 if nyquist <= 24000 else (5000 if nyquist <= 48000 else 10000)
            for f in range(step, int(nyquist) + 1, step):
                x_s = vb.mapViewToScene(QPointF(float(f), 0.0)).x()
                if rect.left() + 18.0 <= x_s <= max_freq_x:
                    lbl = f"{int(f // 1000)}k" if f >= 1000 else str(f)
                    x_text = min(x_s - 22.0, rect.right() - 46.0)
                    painter.drawText(
                        QRectF(x_text, rect.bottom() - 19.0, 44.0, 16.0),
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

            scope_open = (
                hasattr(self.app_ref, "scope_overlay")
                and self.app_ref.scope_overlay is not None
                and self.app_ref.scope_overlay.isVisible()
            )
            min_tick_y = rect.top() + (212.0 if scope_open else 42.0)
            min_eq_tick_y = rect.top() + (216.0 if scope_open else 62.0)

            for db_val in range(first_tick, db_max + 1, step_db):
                y_s = vb.mapViewToScene(QPointF(0.0, float(db_val))).y()
                if min_tick_y <= y_s <= rect.bottom() - 16.0:
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
                if min_eq_tick_y <= y_eq <= rect.bottom() - 16.0:
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
            scope_open = (self.scope_overlay is not None and self.scope_overlay.isVisible())
            badge_y = 216 if scope_open else 46
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

        icon_path = get_resource_path(os.path.join("assets", "icon.ico"))
        if not os.path.exists(icon_path):
            icon_path = get_resource_path(os.path.join("assets", "logo.png"))
        if not os.path.exists(icon_path):
            icon_path = get_resource_path("icon.ico")
        self._icon_path = icon_path
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

    def showEvent(self, event):
        super().showEvent(event)
        if hasattr(self, "_icon_path") and self._icon_path:
            apply_windows_taskbar_icon(int(self.winId()), self._icon_path)

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

        self.sample_rates = [44100, 48000, 88200, 96000, 176400, 192000]
        self.current_sr = 48000
        self.audio_block_size = 512
        self.fft_sizes = [512, 1024, 2048, 4096, 8192, 16384]
        self.fft_size = 4096
        self.channels = 2

        self.audio_queue = queue.Queue(maxsize=10)
        self.ring_buffer_out_b = SecondaryOutputRingBuffer(block_size=self.audio_block_size, channels=2)
        self.stream = None
        self.stream_out_b = None
        self.delay_bus_a = StereoDelayLine(sample_rate=self.current_sr)
        self.delay_bus_b = StereoDelayLine(sample_rate=self.current_sr)
        self._delay_manually_adjusted = False
        self._loading_config = True

        # Inicio MUTEADO por seguridad
        self.is_muted = True

        # Rango por defecto del analizador espectral: -120 dB a 0 dB
        self.decay_factor = 0.88
        self.db_min = -120
        self.db_max = 0

        # Flags para no llamar a setData() del espectro ni osciloscopio cuando ya reposan en silencio
        self._in_drawn_at_rest = False
        self._out1_drawn_at_rest = False
        self._out2_drawn_at_rest = False
        self._pre_drawn_at_rest = False
        self._post_drawn_at_rest = False
        self._post2_drawn_at_rest = False
        self._auto_out2_activated = False
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

        # 1. Curva FFT Input (Pre-EQ)
        c_in = QColor(self.theme.get("curve_in", "#9E9E9E"))
        self.fft_curve_in = self.plot_widget.plot(
            pen=pg.mkPen(color=c_in, width=1.1),
            fillLevel=float(self.db_min - 12.0)
        )
        self.fft_curve_in.setZValue(-20)
        self.pre_curve = self.fft_curve_in  # compat alias

        # 2. Curva FFT Output 2 (Bus B / Salidas 3-4 Post-EQ)
        c_out2 = QColor(self.theme.get("curve_out2", "#00E676"))
        self.fft_curve_out2 = self.plot_widget.plot(
            pen=pg.mkPen(color=c_out2, width=float(self.theme.get("curve_width", 16)) / 10.0),
            fillLevel=float(self.db_min - 12.0)
        )
        self.fft_curve_out2.setZValue(-15)

        # 3. Curva FFT Output 1 (Bus A / Salidas 1-2 Post-EQ)
        c_out1 = QColor(self.theme.get("curve_out1", self.theme.get("curve", "#00E5FF")))
        self.fft_curve_out1 = self.plot_widget.plot(
            pen=pg.mkPen(color=c_out1, width=float(self.theme.get("curve_width", 16)) / 10.0),
            fillLevel=float(self.db_min - 12.0)
        )
        self.fft_curve_out1.setZValue(-10)
        self.curve = self.fft_curve_out1  # compat alias

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

        logo_path = get_resource_path(os.path.join("assets", "logo.png"))
        if os.path.exists(logo_path):
            logo_lbl = QLabel(self.top_bar)
            logo_pix = QPixmap(logo_path).scaled(
                22, 22, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
            logo_lbl.setPixmap(logo_pix)
            logo_lbl.setContentsMargins(2, 0, 0, 0)
            top_layout.addWidget(logo_lbl)

        brand_label = QLabel(
            "<b style='color:#00E5FF; font-size:13px; letter-spacing:0.8px;'>BDJ</b> "
            "<b style='color:#FFD54F; font-size:13px;'>FFT-EQ</b>"
        )
        brand_label.setContentsMargins(0, 0, 4, 0)
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
        self.analyzer_menu_btn = LiquidGlassButton("📊 Analyzer: In+Out1 ▾", self.top_bar)
        self.analyzer_menu_btn.setToolTip("Configurar Analizador FFT, Curvas, Escalas, Rango dB y Colores")
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
        self.scope_overlay = WaveformOverlayWidget(self.scope_buffer, self.graph_container, app_ref=self)
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
        self.scope_widget.set_expanded_indicator(visible)

        # Cambiar el rango de la FFT de forma instantánea sin recálculos bloqueantes ni cortes de audio
        is_y_log = (self.y_scale_combo.currentText() == "Logarítmica (dB)")
        if is_y_log:
            y_floor = float(self.db_min - 12.0)
            if visible:
                self.scope_overlay.raise_()
                h_graph = max(300.0, float(self.graph_container.height() if hasattr(self, "graph_container") else 600.0))
                target_f_top = min(0.48, max(0.32, 232.0 / h_graph))
                y_ceil = float(round((target_f_top / (1.0 - target_f_top)) * (-y_floor)))
            else:
                y_ceil = float(self.db_max + 12.0)

            self.plot_widget.setYRange(y_floor, y_ceil, padding=0.0)
            self.eq_overlay.update_y_range(y_floor, y_ceil)
        else:
            self.plot_widget.setYRange(0.0, 1.08, padding=0.0)
            self.eq_overlay.update_y_range(-132.0, 12.0)

        if visible:
            self.scope_overlay.update()

        if hasattr(self, "full_bleed_grid"):
            self.full_bleed_grid.update()
        if hasattr(self, "edge_masks_and_scales"):
            self.edge_masks_and_scales.update()

        self.graph_container._reposition_children()

    toggle_oscilloscope_panel = toggle_expanded_scope

    def _on_scope_duration_changed(self, _sec: int):
        if hasattr(self, "scope_overlay") and hasattr(self.scope_overlay, "dur_btn"):
            self.scope_overlay.dur_btn.setText(f"{_sec}s")
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
        style_dark_combobox(self.in_combo)
        self.in_combo.setMinimumWidth(320)
        self.in_combo.currentIndexChanged.connect(self._on_audio_device_changed)
        grid.addWidget(self.in_combo, 0, 1)

        grid.addWidget(QLabel("Dispositivo de Salida:"), 1, 0)
        self.out_combo = QComboBox()
        style_dark_combobox(self.out_combo)
        self.out_combo.setMinimumWidth(320)
        self.out_combo.currentIndexChanged.connect(self._on_audio_device_changed)
        grid.addWidget(self.out_combo, 1, 1)

        grid.addWidget(QLabel("Salida Secundaria (Bus B - 3,4):"), 2, 0)
        self.out_b_combo = QComboBox()
        style_dark_combobox(self.out_b_combo)
        self.out_b_combo.setMinimumWidth(320)
        self.out_b_combo.setToolTip(
            "Opcional: envía el Bus B (Out B [3-4]) a un segundo dispositivo de salida independiente."
        )
        self.out_b_combo.currentIndexChanged.connect(self._on_audio_device_changed)
        grid.addWidget(self.out_b_combo, 2, 1)

        grid.addWidget(QLabel("Sample Rate:"), 3, 0)
        self.sr_combo = QComboBox()
        style_dark_combobox(self.sr_combo)
        for sr in self.sample_rates:
            self.sr_combo.addItem(f"{sr} Hz", sr)
        self.sr_combo.setCurrentText("48000 Hz")
        self.sr_combo.currentIndexChanged.connect(self.on_sr_changed)
        grid.addWidget(self.sr_combo, 3, 1)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background-color: rgba(255, 255, 255, 0.12); min-height: 1px; max-height: 1px;")
        grid.addWidget(sep, 4, 0, 1, 2)

        # Fila 5: Delay Out 1 (A) + Botón Auto Sync
        grid.addWidget(QLabel("Delay Out 1 (A):"), 5, 0)
        delay_a_layout = QHBoxLayout()
        delay_a_layout.setSpacing(8)

        self.delay_out1_spin = QDoubleSpinBox()
        self.delay_out1_spin.setRange(0.0, 200.0)
        self.delay_out1_spin.setSingleStep(0.5)
        self.delay_out1_spin.setDecimals(1)
        self.delay_out1_spin.setSuffix(" ms")
        self.delay_out1_spin.setValue(0.0)
        self.delay_out1_spin.setFixedWidth(110)
        self.delay_out1_spin.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #181A20;
                color: #00E5FF;
                border: 1px solid #3A3F4B;
                border-radius: 5px;
                padding: 3px 6px;
                font-size: 11px;
                font-weight: bold;
            }
            QDoubleSpinBox:hover {
                border-color: #00E5FF;
                background-color: #20242D;
            }
        """)
        self.delay_out1_spin.valueChanged.connect(self._on_user_delay_a_changed)
        delay_a_layout.addWidget(self.delay_out1_spin)

        self.auto_sync_btn = QPushButton("⚡ Auto Sync")
        self.auto_sync_btn.setToolTip("Calcula y sincroniza automáticamente la latencia acústica entre Salida 1 y Salida 2")
        self.auto_sync_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_sync_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 229, 255, 0.12);
                color: #00E5FF;
                border: 1px solid #00E5FF;
                border-radius: 5px;
                padding: 3px 10px;
                font-size: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(0, 229, 255, 0.28);
                color: #FFFFFF;
                border-color: #33EBFF;
            }
        """)
        self.auto_sync_btn.clicked.connect(self.auto_sync_delays)
        delay_a_layout.addWidget(self.auto_sync_btn)
        delay_a_layout.addStretch()
        grid.addLayout(delay_a_layout, 5, 1)

        # Fila 6: Delay Out 2 (B)
        grid.addWidget(QLabel("Delay Out 2 (B):"), 6, 0)
        self.delay_out2_spin = QDoubleSpinBox()
        self.delay_out2_spin.setRange(0.0, 200.0)
        self.delay_out2_spin.setSingleStep(0.5)
        self.delay_out2_spin.setDecimals(1)
        self.delay_out2_spin.setSuffix(" ms")
        self.delay_out2_spin.setValue(0.0)
        self.delay_out2_spin.setFixedWidth(110)
        self.delay_out2_spin.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #181A20;
                color: #00E676;
                border: 1px solid #3A3F4B;
                border-radius: 5px;
                padding: 3px 6px;
                font-size: 11px;
                font-weight: bold;
            }
            QDoubleSpinBox:hover {
                border-color: #00E676;
                background-color: #20242D;
            }
        """)
        self.delay_out2_spin.valueChanged.connect(self._on_user_delay_b_changed)
        grid.addWidget(self.delay_out2_spin, 6, 1, alignment=Qt.AlignmentFlag.AlignLeft)

    def _on_user_delay_a_changed(self, val: float):
        if not getattr(self, "_is_auto_syncing", False):
            self._delay_manually_adjusted = True
        self.delay_bus_a.set_delay_ms(val)
        self.schedule_config_save()

    def _on_user_delay_b_changed(self, val: float):
        if not getattr(self, "_is_auto_syncing", False):
            self._delay_manually_adjusted = True
        self.delay_bus_b.set_delay_ms(val)
        self.schedule_config_save()

    def auto_sync_delays(self):
        """Calcula y aplica automáticamente el retraso ideal para sincronizar Salida 1 con Salida 2."""
        has_secondary = (self.stream_out_b is not None) or (
            hasattr(self, "out_b_combo") and self.out_b_combo.currentIndex() > 0
        )
        self._is_auto_syncing = True
        try:
            if not has_secondary:
                self._delay_manually_adjusted = False
                self.delay_out1_spin.setValue(0.0)
                self.delay_out2_spin.setValue(0.0)
                self.delay_bus_a.set_delay_ms(0.0)
                self.delay_bus_b.set_delay_ms(0.0)
                return

            buffer_delay_ms = (2.0 * self.audio_block_size / float(self.current_sr)) * 1000.0

            main_out_lat = 0.0
            sec_out_lat = 0.0
            if self.stream is not None:
                try:
                    lat = self.stream.latency
                    main_out_lat = float(lat[1]) if isinstance(lat, (tuple, list)) and len(lat) > 1 else float(lat)
                except Exception:
                    pass
            if self.stream_out_b is not None:
                try:
                    lat = self.stream_out_b.latency
                    sec_out_lat = float(lat[1]) if isinstance(lat, (tuple, list)) and len(lat) > 1 else float(lat)
                except Exception:
                    pass

            driver_diff_ms = max(-50.0, min(100.0, (sec_out_lat - main_out_lat) * 1000.0))
            auto_delay_ms = round(max(0.0, buffer_delay_ms + driver_diff_ms), 1)

            self._delay_manually_adjusted = False
            self.delay_out1_spin.setValue(auto_delay_ms)
            self.delay_out2_spin.setValue(0.0)
            self.delay_bus_a.set_delay_ms(auto_delay_ms)
            self.delay_bus_b.set_delay_ms(0.0)
        finally:
            self._is_auto_syncing = False
        self.schedule_config_save()

    def _show_audio_popup(self):
        self.audio_popup.adjustSize()
        pos = self.audio_menu_btn.mapToGlobal(QPoint(0, self.audio_menu_btn.height() + 4))
        self.audio_popup.move(pos)
        self.audio_popup.show()

    def _build_analyzer_popup(self):
        """Panel del Analizador con FFT Samples, Curvas FFT (In, Out 1, Out 2) y Colores."""
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
            QPushButton {
                background-color: #262930;
                color: #ECEFF4;
                border: 1px solid #3E4350;
                border-radius: 4px;
                padding: 5px 8px;
                font-size: 11px;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #323742;
                border-color: #00E5FF;
                color: #FFFFFF;
            }
        """)
        grid = QGridLayout(self.analyzer_popup)
        grid.setContentsMargins(12, 10, 12, 10)
        grid.setSpacing(8)

        chk_row = QHBoxLayout()
        self.show_in_chk = QCheckBox("Input (Pre)")
        self.show_in_chk.setChecked(True)
        self.show_in_chk.toggled.connect(self._update_analyzer_btn_label)
        chk_row.addWidget(self.show_in_chk)
        self.show_pre_chk = self.show_in_chk

        self.show_out1_chk = QCheckBox("Output 1 (Bus A)")
        self.show_out1_chk.setChecked(True)
        self.show_out1_chk.toggled.connect(self._update_analyzer_btn_label)
        chk_row.addWidget(self.show_out1_chk)
        self.show_post_chk = self.show_out1_chk

        self.show_out2_chk = QCheckBox("Output 2 (Bus B)")
        self.show_out2_chk.setChecked(False)
        self.show_out2_chk.toggled.connect(self._update_analyzer_btn_label)
        chk_row.addWidget(self.show_out2_chk)
        self.show_post2_chk = self.show_out2_chk

        grid.addLayout(chk_row, 0, 0, 1, 2)

        grid.addWidget(QLabel("FFT Samples:"), 1, 0)
        self.fft_combo = QComboBox()
        style_dark_combobox(self.fft_combo)
        for sz in self.fft_sizes:
            self.fft_combo.addItem(f"{sz} samples", sz)
        self.fft_combo.setCurrentText("4096 samples")
        self.fft_combo.currentIndexChanged.connect(self.on_fft_size_changed)
        grid.addWidget(self.fft_combo, 1, 1)

        grid.addWidget(QLabel("Escala Eje X:"), 2, 0)
        self.x_scale_combo = QComboBox()
        style_dark_combobox(self.x_scale_combo)
        self.x_scale_combo.addItems(["Logarítmica", "Lineal"])
        self.x_scale_combo.currentIndexChanged.connect(self.update_axes_config)
        grid.addWidget(self.x_scale_combo, 2, 1)

        grid.addWidget(QLabel("Escala Eje Y:"), 3, 0)
        self.y_scale_combo = QComboBox()
        style_dark_combobox(self.y_scale_combo)
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
        style_dark_combobox(self.bg_color_combo)
        for label, hex_val in BACKGROUND_COLOR_PRESETS:
            self.bg_color_combo.addItem(label, hex_val)
        self.bg_color_combo.addItem("Personalizado...", "custom")
        self.bg_color_combo.activated.connect(self._on_bg_combo_activated)
        grid.addWidget(self.bg_color_combo, 7, 1)

        # Botones de personalización de color de las curvas FFT
        colors_layout = QVBoxLayout()
        colors_layout.setSpacing(4)

        btn_color_in = QPushButton("🎨 Color Curva Input (Pre)...")
        btn_color_in.clicked.connect(lambda: (self.analyzer_popup.close(), self._pick_curve_color("curve_in", "Curva Input (Pre)")))
        colors_layout.addWidget(btn_color_in)

        btn_color_out1 = QPushButton("🎨 Color Curva Output 1 (Bus A)...")
        btn_color_out1.clicked.connect(lambda: (self.analyzer_popup.close(), self._pick_curve_color("curve_out1", "Curva Output 1 (Bus A)")))
        colors_layout.addWidget(btn_color_out1)

        btn_color_out2 = QPushButton("🎨 Color Curva Output 2 (Bus B)...")
        btn_color_out2.clicked.connect(lambda: (self.analyzer_popup.close(), self._pick_curve_color("curve_out2", "Curva Output 2 (Bus B)")))
        colors_layout.addWidget(btn_color_out2)

        colors_btn = QPushButton("⚙ Configuración Completa de Colores...")
        colors_btn.clicked.connect(lambda: (self.analyzer_popup.close(), self.open_color_settings()))
        colors_layout.addWidget(colors_btn)

        grid.addLayout(colors_layout, 8, 0, 1, 2)

    def _pick_curve_color(self, key: str, label: str):
        original = self.theme.get(key, "#00E5FF")
        dlg = QColorDialog(QColor(original), self)
        dlg.setWindowTitle(f"Elegir color: {label}")
        dlg.setOption(QColorDialog.ColorDialogOption.DontUseNativeDialog, True)

        def _apply(c):
            if c.isValid():
                self.theme[key] = c.name()
                if key == "curve_out1":
                    self.theme["curve"] = c.name()
                self.apply_theme()
                self.schedule_config_save()

        dlg.currentColorChanged.connect(_apply)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            _apply(dlg.selectedColor())
        else:
            self.theme[key] = original
            if key == "curve_out1":
                self.theme["curve"] = original
            self.apply_theme()

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
        show_in = self.show_in_chk.isChecked()
        show_out1 = self.show_out1_chk.isChecked()
        show_out2 = self.show_out2_chk.isChecked()
        if not show_in:
            self.fft_curve_in.setData([], [])
        if not show_out1:
            self.fft_curve_out1.setData([], [])
        if not show_out2:
            self.fft_curve_out2.setData([], [])
        self._in_drawn_at_rest = False
        self._out1_drawn_at_rest = False
        self._out2_drawn_at_rest = False
        self._pre_drawn_at_rest = False
        self._post_drawn_at_rest = False

        active = []
        if show_in:
            active.append("In")
        if show_out1:
            active.append("Out1")
        if show_out2:
            active.append("Out2")

        if active:
            txt = f"📊 Analyzer: {'+'.join(active)} ▾"
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
                "output_b_device_name": self.out_b_combo.currentText(),
                "output_b_device_index": self.out_b_combo.currentData(),
                "sample_rate": int(self.current_sr),
                "delay_out1_ms": float(self.delay_out1_spin.value()) if hasattr(self, "delay_out1_spin") else 0.0,
                "delay_out2_ms": float(self.delay_out2_spin.value()) if hasattr(self, "delay_out2_spin") else 0.0,
                "delay_manually_adjusted": bool(getattr(self, "_delay_manually_adjusted", False)),
            },
            "analyzer": {
                "fft_size": int(self.fft_size),
                "show_pre": bool(self.show_in_chk.isChecked()),
                "show_post": bool(self.show_out1_chk.isChecked()),
                "show_post2": bool(self.show_out2_chk.isChecked()),
                "x_scale": self.x_scale_combo.currentText(),
                "y_scale": self.y_scale_combo.currentText(),
                "db_min": int(self.db_min),
                "db_max": int(self.db_max),
                "hold_percent": int(self.hold_slider.value()),
                "scope_duration_sec": int(self.scope_buffer.duration_sec),
                "scope_active_signals": list(self.scope_overlay.active_signals),
                "scope_signal_colors": dict(self.scope_overlay.signal_colors),
                "scope_opacity_pct": int(self.scope_overlay.opacity_pct),
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

            out_b_name = audio_cfg.get("output_b_device_name")
            out_b_idx_saved = audio_cfg.get("output_b_device_index")
            if isinstance(out_b_name, str) and self.out_b_combo.findText(out_b_name) >= 0:
                self.out_b_combo.setCurrentIndex(self.out_b_combo.findText(out_b_name))
            elif isinstance(out_b_idx_saved, int) and self.out_b_combo.findData(out_b_idx_saved) >= 0:
                self.out_b_combo.setCurrentIndex(self.out_b_combo.findData(out_b_idx_saved))

            sr_saved = audio_cfg.get("sample_rate")
            if sr_saved in self.sample_rates:
                self.current_sr = int(sr_saved)
                self.sr_combo.setCurrentIndex(self.sr_combo.findData(self.current_sr))
                self.fft_analyzer.set_sample_rate(self.current_sr)
                self.eq_engine.set_sample_rate(self.current_sr)
                self.scope_buffer.set_sample_rate(self.current_sr)

            if hasattr(self, "delay_out1_spin") and "delay_out1_ms" in audio_cfg:
                d1 = float(audio_cfg["delay_out1_ms"])
                self.delay_out1_spin.blockSignals(True)
                self.delay_out1_spin.setValue(d1)
                self.delay_out1_spin.blockSignals(False)
                self.delay_bus_a.set_delay_ms(d1)
            if hasattr(self, "delay_out2_spin") and "delay_out2_ms" in audio_cfg:
                d2 = float(audio_cfg["delay_out2_ms"])
                self.delay_out2_spin.blockSignals(True)
                self.delay_out2_spin.setValue(d2)
                self.delay_out2_spin.blockSignals(False)
                self.delay_bus_b.set_delay_ms(d2)
            if "delay_manually_adjusted" in audio_cfg:
                self._delay_manually_adjusted = bool(audio_cfg["delay_manually_adjusted"])

        analyzer_cfg = self.saved_config.get("analyzer", {})
        if isinstance(analyzer_cfg, dict):
            fft_sz = analyzer_cfg.get("fft_size")
            if fft_sz in self.fft_sizes:
                self.fft_size = int(fft_sz)
                self.fft_combo.setCurrentIndex(self.fft_combo.findData(self.fft_size))
                self.fft_analyzer.set_fft_size(self.fft_size)

            if "show_pre" in analyzer_cfg:
                self.show_in_chk.setChecked(bool(analyzer_cfg["show_pre"]))
            if "show_post" in analyzer_cfg:
                self.show_out1_chk.setChecked(bool(analyzer_cfg["show_post"]))
            if "show_post2" in analyzer_cfg:
                self.show_out2_chk.setChecked(bool(analyzer_cfg["show_post2"]))

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
            if scope_dur in (1, 3, 5, 10):
                self.scope_widget.set_duration(int(scope_dur))
                if hasattr(self, "scope_overlay") and hasattr(self.scope_overlay, "dur_btn"):
                    self.scope_overlay.dur_btn.setText(f"{int(scope_dur)}s")

            scope_active = analyzer_cfg.get("scope_active_signals")
            if isinstance(scope_active, list):
                self.scope_overlay.set_active_signals(scope_active)

            scope_colors = analyzer_cfg.get("scope_signal_colors")
            if isinstance(scope_colors, dict):
                self.scope_overlay.set_signal_colors(scope_colors)

            scope_opac = analyzer_cfg.get("scope_opacity_pct")
            if isinstance(scope_opac, (int, float)):
                self.scope_overlay.set_opacity_pct(int(scope_opac))

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
        # 1. Input (Pre-EQ): gris/blanco tenue (alpha=25)
        c_in = QColor(t.get("curve_in", "#9E9E9E"))
        a_in = int(t.get("curve_in_alpha", 25))
        c_in_fill = QColor(c_in)
        c_in_fill.setAlpha(a_in)
        self.fft_curve_in.setBrush(pg.mkBrush(c_in_fill))

        # 2. Output 1 (Bus A): color principal (alpha=55)
        c_out1 = QColor(t.get("curve_out1", t.get("curve", "#00E5FF")))
        a_out1 = int(t.get("curve_out1_alpha", t.get("fill_alpha", 55)))
        c_out1_fill = QColor(c_out1)
        c_out1_fill.setAlpha(a_out1)
        self.fft_curve_out1.setBrush(pg.mkBrush(c_out1_fill))

        # 3. Output 2 (Bus B): color secundario verde esmeralda (alpha=45)
        c_out2 = QColor(t.get("curve_out2", "#00E676"))
        a_out2 = int(t.get("curve_out2_alpha", 45))
        c_out2_fill = QColor(c_out2)
        c_out2_fill.setAlpha(a_out2)
        self.fft_curve_out2.setBrush(pg.mkBrush(c_out2_fill))

    def apply_theme(self):
        t = self.theme
        self.plot_widget.setBackground(t["background"])

        c_in = QColor(t.get("curve_in", "#9E9E9E"))
        self.fft_curve_in.setPen(pg.mkPen(color=c_in, width=1.1))

        c_out1 = QColor(t.get("curve_out1", t.get("curve", "#00E5FF")))
        w_curve = float(t.get("curve_width", 16)) / 10.0
        self.fft_curve_out1.setPen(pg.mkPen(color=c_out1, width=w_curve))

        c_out2 = QColor(t.get("curve_out2", "#00E676"))
        self.fft_curve_out2.setPen(pg.mkPen(color=c_out2, width=w_curve))

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
        self._post2_drawn_at_rest = False
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
        self.out_b_combo.blockSignals(True)
        self.in_combo.clear()
        self.out_combo.clear()
        self.out_b_combo.clear()
        self.out_b_combo.addItem("— Ninguna (Desactivada) —", None)

        devices = sd.query_devices()
        hostapis = sd.query_hostapis()

        for idx, dev in enumerate(devices):
            api_name = hostapis[dev["hostapi"]]["name"]
            dev_label = f"[{api_name}] {dev['name']}"
            if dev["max_input_channels"] > 0:
                self.in_combo.addItem(dev_label, idx)
            if dev["max_output_channels"] > 0:
                self.out_combo.addItem(dev_label, idx)
                self.out_b_combo.addItem(dev_label, idx)

        default_in, default_out = sd.default.device
        in_idx = self.in_combo.findData(default_in)
        out_idx = self.out_combo.findData(default_out)
        if in_idx >= 0:
            self.in_combo.setCurrentIndex(in_idx)
        if out_idx >= 0:
            self.out_combo.setCurrentIndex(out_idx)
        self.out_b_combo.setCurrentIndex(0)

        self.in_combo.blockSignals(False)
        self.out_combo.blockSignals(False)
        self.out_b_combo.blockSignals(False)

    def on_sr_changed(self):
        self.current_sr = self.sr_combo.currentData()
        self.fft_analyzer.set_sample_rate(self.current_sr)
        self.eq_engine.set_sample_rate(self.current_sr)
        self.scope_buffer.set_sample_rate(self.current_sr)
        self.eq_panel.refresh_from_engine()
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
            if hasattr(self, "scope_overlay") and self.scope_overlay is not None and self.scope_overlay.isVisible():
                h_graph = max(300.0, float(self.graph_container.height() if hasattr(self, "graph_container") else 600.0))
                # El osciloscopio desplegado termina en y = 210px. Con margen de respiro, target en y = 232px.
                target_f_top = min(0.48, max(0.32, 232.0 / h_graph))
                y_ceil = float(round((target_f_top / (1.0 - target_f_top)) * (-y_floor)))
            else:
                y_ceil = float(self.db_max + 12.0)

            self.plot_widget.setYRange(y_floor, y_ceil, padding=0.0)
            self.fft_curve_in.setFillLevel(y_floor)
            self.fft_curve_out1.setFillLevel(y_floor)
            self.fft_curve_out2.setFillLevel(y_floor)
            self.eq_overlay.update_y_range(y_floor, y_ceil)
        else:
            self.plot_widget.setYRange(0.0, 1.08, padding=0.0)
            self.fft_curve_in.setFillLevel(0.0)
            self.fft_curve_out1.setFillLevel(0.0)
            self.fft_curve_out2.setFillLevel(0.0)
            self.eq_overlay.update_y_range(-132.0, 12.0)

        self._update_curve_brush()
        self._pre_drawn_at_rest = False
        self._post_drawn_at_rest = False
        self._post2_drawn_at_rest = False
        self.eq_overlay.set_axes_mode(is_x_log=is_x_log, sr=self.current_sr)
        self.full_bleed_grid.update()
        self.edge_masks_and_scales.update()
        self.schedule_config_save()

    def audio_callback(self, indata, outdata, frames, time_info, status):
        out_A, out_B = self.eq_engine.process_dual_block(indata)

        if self.is_muted:
            outdata.fill(0.0)
            blk_b = np.zeros_like(out_B)
        else:
            out_A_delayed = self.delay_bus_a.process(out_A)
            out_B_delayed = self.delay_bus_b.process(out_B)

            n_out_ch = outdata.shape[1]
            if n_out_ch >= 4:
                outdata[:, 0:2] = out_A_delayed[:, 0:2]
                outdata[:, 2:4] = out_B_delayed[:, 0:2]
                if n_out_ch > 4:
                    outdata[:, 4:] = 0.0
            elif n_out_ch >= 2:
                outdata[:, 0:2] = out_A_delayed[:, 0:2]
            elif n_out_ch == 1:
                outdata[:, 0] = 0.5 * (out_A_delayed[:, 0] + out_A_delayed[:, 1])

            blk_b = out_B_delayed.copy()

        if self.stream_out_b is not None:
            self.ring_buffer_out_b.write(blk_b)

        try:
            self.audio_queue.put_nowait((indata.copy(), out_A.copy(), out_B.copy()))
        except queue.Full:
            pass

    def audio_callback_out_b(self, outdata, frames, time_info, status):
        if self.is_muted:
            outdata.fill(0.0)
            return
        self.ring_buffer_out_b.read(outdata, frames)

    def restart_audio_stream(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

        if self.stream_out_b is not None:
            try:
                self.stream_out_b.stop()
                self.stream_out_b.close()
            except Exception:
                pass
            self.stream_out_b = None

        # Ajuste dinámico de blocksize según el sample rate para mantener latencia estable de ~10.6 ms
        if self.current_sr >= 176400:
            self.audio_block_size = 2048
        elif self.current_sr >= 88200:
            self.audio_block_size = 1024
        else:
            self.audio_block_size = 512

        self.ring_buffer_out_b.reset(block_size=self.audio_block_size)
        self.scope_buffer.sample_rate = self.current_sr
        self.scope_buffer.block_size = self.audio_block_size
        self.scope_buffer._allocate_buffers()

        in_dev = self.in_combo.currentData()
        out_dev = self.out_combo.currentData()
        out_b_dev = self.out_b_combo.currentData() if hasattr(self, "out_b_combo") else None
        if in_dev is None or out_dev is None:
            return

        # Detectar si el dispositivo de salida principal soporta 4 o más canales (Out A [1-2] + Out B [3-4])
        out_max_ch = 2
        try:
            dev_info = sd.query_devices(out_dev)
            out_max_ch = int(dev_info.get("max_output_channels", 2))
        except Exception:
            out_max_ch = 2

        primary_opened = False
        if out_max_ch >= 4:
            try:
                self.stream = sd.Stream(
                    samplerate=self.current_sr,
                    blocksize=self.audio_block_size,
                    device=(in_dev, out_dev),
                    channels=(self.channels, 4),
                    dtype="float32",
                    callback=self.audio_callback,
                )
                self.stream.start()
                primary_opened = True
            except Exception:
                self.stream = None
                primary_opened = False

        if not primary_opened:
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
                return

        # Si el usuario seleccionó una Salida Secundaria (Bus B - 3,4), abrir OutputStream secundario no bloqueante
        if out_b_dev is not None and out_b_dev != out_dev:
            try:
                self.stream_out_b = sd.OutputStream(
                    samplerate=self.current_sr,
                    blocksize=self.audio_block_size,
                    device=out_b_dev,
                    channels=2,
                    dtype="float32",
                    callback=self.audio_callback_out_b,
                )
                self.stream_out_b.start()
            except Exception as e:
                self.stream_out_b = None
                QMessageBox.warning(
                    self,
                    "Aviso de Salida Secundaria (Bus B)",
                    f"No se pudo iniciar la Salida Secundaria (Bus B) en el dispositivo seleccionado:\n{e}",
                )

        # Actualizar frecuencia de muestreo de las líneas de retardo estéreo
        self.delay_bus_a.set_sample_rate(self.current_sr)
        self.delay_bus_b.set_sample_rate(self.current_sr)

        if hasattr(self, "delay_out1_spin"):
            if self.stream_out_b is not None:
                # Caso B: Salida 1 y Salida 2 activas simultáneamente
                if not getattr(self, "_delay_manually_adjusted", False):
                    self.auto_sync_delays()
                else:
                    self.delay_bus_a.set_delay_ms(self.delay_out1_spin.value())
                    self.delay_bus_b.set_delay_ms(self.delay_out2_spin.value())
            else:
                # Caso A: Solo Salida 1 activa (sin Salida 2)
                if not getattr(self, "_delay_manually_adjusted", False):
                    self.delay_out1_spin.setValue(0.0)
                    self.delay_out2_spin.setValue(0.0)
                self.delay_bus_a.set_delay_ms(self.delay_out1_spin.value())
                self.delay_bus_b.set_delay_ms(self.delay_out2_spin.value())

    def update_visuals(self):
        pre_blocks = []
        post_blocks = []
        post2_blocks = []
        while not self.audio_queue.empty():
            try:
                item = self.audio_queue.get_nowait()
                if len(item) == 3:
                    pre_blk, post_blk, post2_blk = item
                    pre_blocks.append(pre_blk)
                    post_blocks.append(post_blk)
                    post2_blocks.append(post2_blk)
                else:
                    pre_blk, post_blk = item
                    pre_blocks.append(pre_blk)
                    post_blocks.append(post_blk)
            except queue.Empty:
                break

        # Actualiza curvas EQ (en IDLE retorna en 0 ms gracias a dirty-check)
        self.eq_overlay.update_eq_curves(force=False)
        self.eq_overlay.update_ghost_tick()
        self.eq_panel.update_live_Readouts()

        gr_db = float(min(self.eq_engine.limiter_a.last_gr_db, self.eq_engine.limiter_b.last_gr_db))

        # 1. Mini-Osciloscopio a 60 FPS reales y 0% CPU (Acumulación O(1) de picos sin bucles pesados)
        if post_blocks:
            vmin = 0.0
            vmax = 0.0
            for blk in post_blocks:
                mono = (blk[:, 0] + blk[:, 1]) * 0.5 if blk.ndim > 1 and blk.shape[1] > 1 else blk[:, 0]
                mn = float(np.min(mono))
                mx = float(np.max(mono))
                if mn < vmin: vmin = mn
                if mx > vmax: vmax = mx
        else:
            vmin = vmax = 0.0

        if hasattr(self, "scope_widget"):
            self.scope_widget.update_gr(gr_db)
            self.scope_widget.mini_scope.push_audio_peaks(vmin, vmax, gr_db)
            self.scope_widget.mini_scope.tick_render(QColor("#00E5FF"))

        # 2. Regla de Oro de Rendimiento: Solo actualizar el osciloscopio grande si está visible
        if self.scope_overlay.isVisible() and post_blocks:
            self.scope_overlay.update_gr(gr_db)
            scope_changed = self.scope_buffer.push_multi_blocks(pre_blocks, post_blocks, post2_blocks, gr_db=gr_db)
            self._scope_tick_counter = (self._scope_tick_counter + 1) % 2
            if scope_changed and self._scope_tick_counter == 0:
                self.scope_overlay.update()

        if not post_blocks:
            return

        latest_post = post_blocks[-1]
        eps = 1e-10
        peak_l = float(np.max(np.abs(latest_post[:, 0])))
        peak_r = float(np.max(np.abs(latest_post[:, 1]))) if latest_post.shape[1] > 1 else peak_l

        db_l = float(np.clip(20.0 * np.log10(peak_l + eps), -60.0, 0.0))
        db_r = float(np.clip(20.0 * np.log10(peak_r + eps), -60.0, 0.0))
        self.stereo_meter.set_levels(db_l, db_r)

        # Activación inteligente por defecto de Output 2:
        # Se activa automáticamente si el usuario configura una salida secundaria o si alguna banda está ruteada a B o A+B
        has_b_routed = any(b.target_output in ("B", "A+B") for b in self.eq_engine.bands if b.enabled)
        has_secondary_device = (self.stream_out_b is not None) or (
            hasattr(self, "out_b_combo") and self.out_b_combo.currentIndex() > 0
        )
        if (has_b_routed or has_secondary_device) and not getattr(self, "_auto_out2_activated", False):
            self.show_out2_chk.setChecked(True)
            self._auto_out2_activated = True

        show_in = self.show_in_chk.isChecked()
        show_out1 = self.show_out1_chk.isChecked()
        show_out2 = self.show_out2_chk.isChecked()
        y_floor_fft = float(self.db_min - 12.0)

        self.fft_analyzer.push_blocks(
            pre_blocks if show_in else [],
            post_blocks if show_out1 else [],
            post2_blocks if show_out2 else []
        )
        self.fft_analyzer.update_spectra(
            compute_in=show_in,
            compute_out1=show_out1,
            compute_out2=show_out2,
            db_min=y_floor_fft
        )

        is_x_log = (self.x_scale_combo.currentText() == "Logarítmica")
        is_y_log = (self.y_scale_combo.currentText() == "Logarítmica (dB)")

        # Curva 1: Input (Pre-EQ)
        if show_in:
            if not self.fft_analyzer.pre_at_rest:
                self._in_drawn_at_rest = False
                x_in, y_in = self.fft_analyzer.get_smooth_curve(
                    which="in", is_x_log=is_x_log, is_y_log=is_y_log,
                    db_min=y_floor_fft, db_max=self.db_max
                )
                self.fft_curve_in.setData(x_in, y_in)
            elif not self._in_drawn_at_rest:
                self._in_drawn_at_rest = True
                x_in, y_in = self.fft_analyzer.get_smooth_curve(
                    which="in", is_x_log=is_x_log, is_y_log=is_y_log,
                    db_min=y_floor_fft, db_max=self.db_max
                )
                self.fft_curve_in.setData(x_in, y_in)
        else:
            self.fft_curve_in.setData([], [])
            self._in_drawn_at_rest = False

        # Curva 2: Output 1 (Bus A)
        if show_out1:
            if not self.fft_analyzer.post_at_rest:
                self._out1_drawn_at_rest = False
                x_out1, y_out1 = self.fft_analyzer.get_smooth_curve(
                    which="out1", is_x_log=is_x_log, is_y_log=is_y_log,
                    db_min=y_floor_fft, db_max=self.db_max
                )
                self.fft_curve_out1.setData(x_out1, y_out1)
            elif not self._out1_drawn_at_rest:
                self._out1_drawn_at_rest = True
                x_out1, y_out1 = self.fft_analyzer.get_smooth_curve(
                    which="out1", is_x_log=is_x_log, is_y_log=is_y_log,
                    db_min=y_floor_fft, db_max=self.db_max
                )
                self.fft_curve_out1.setData(x_out1, y_out1)
        else:
            self.fft_curve_out1.setData([], [])
            self._out1_drawn_at_rest = False

        # Curva 3: Output 2 (Bus B)
        if show_out2:
            if not self.fft_analyzer.post2_at_rest:
                self._out2_drawn_at_rest = False
                x_out2, y_out2 = self.fft_analyzer.get_smooth_curve(
                    which="out2", is_x_log=is_x_log, is_y_log=is_y_log,
                    db_min=y_floor_fft, db_max=self.db_max
                )
                self.fft_curve_out2.setData(x_out2, y_out2)
            elif not self._out2_drawn_at_rest:
                self._out2_drawn_at_rest = True
                x_out2, y_out2 = self.fft_analyzer.get_smooth_curve(
                    which="out2", is_x_log=is_x_log, is_y_log=is_y_log,
                    db_min=y_floor_fft, db_max=self.db_max
                )
                self.fft_curve_out2.setData(x_out2, y_out2)
        else:
            self.fft_curve_out2.setData([], [])
            self._out2_drawn_at_rest = False

    def closeEvent(self, event):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
        if self.stream_out_b is not None:
            try:
                self.stream_out_b.stop()
                self.stream_out_b.close()
            except Exception:
                pass
        if self.config_save_timer.isActive():
            self.config_save_timer.stop()
        self.save_app_config()
        event.accept()


if __name__ == "__main__":
    # Asegurar AppUserModelID para Windows
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("bdj.ffteq.audioanalyzer.1.0")
        except Exception:
            pass

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    dark_stylesheet = """
    QWidget {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    QMenu {
        background-color: #181A20;
        color: #F0F2F5;
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 8px;
        padding: 6px;
    }
    QMenu::item {
        background-color: transparent;
        padding: 6px 22px 6px 12px;
        border-radius: 4px;
        color: #D8DEE9;
        font-size: 11px;
        font-weight: 500;
    }
    QMenu::item:selected {
        background-color: #2B303C;
        color: #00E5FF;
    }
    QMenu::separator {
        height: 1px;
        background: rgba(255, 255, 255, 0.1);
        margin: 4px 8px;
    }
    QToolTip {
        background-color: #181A20;
        color: #F0F2F5;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 4px;
        padding: 4px 8px;
        font-size: 11px;
    }
    QComboBox {
        background-color: rgba(26, 29, 36, 230);
        color: #F0F2F5;
        border: 1px solid #3A3F4B;
        border-radius: 5px;
        padding: 2px 8px;
        font-size: 11px;
    }
    QComboBox:hover {
        border-color: #00E5FF;
        background-color: rgba(35, 40, 50, 245);
    }
    QComboBox::drop-down {
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 18px;
        border-left: none;
    }
    QComboBox QAbstractItemView {
        background-color: #181A20;
        color: #F0F2F5;
        selection-background-color: #2B303C;
        selection-color: #00E5FF;
        border: 1px solid #3A3F4B;
        border-radius: 6px;
        padding: 4px;
        outline: none;
    }
    QScrollBar:vertical {
        background: #181A20;
        width: 8px;
        margin: 2px 0 2px 0;
        border-radius: 4px;
    }
    QScrollBar::handle:vertical {
        background: #3A3F4B;
        min-height: 20px;
        border-radius: 4px;
    }
    QScrollBar::handle:vertical:hover {
        background: #00E5FF;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }
    """
    app.setStyleSheet(dark_stylesheet)

    icon_path = get_resource_path(os.path.join("assets", "icon.ico"))
    if not os.path.exists(icon_path):
        icon_path = get_resource_path(os.path.join("assets", "logo.png"))
    if not os.path.exists(icon_path):
        icon_path = get_resource_path("icon.ico")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    window = AudioAnalyzerApp()
    window.show()
    apply_windows_taskbar_icon(int(window.winId()), icon_path)
    sys.exit(app.exec())
