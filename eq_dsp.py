import threading
import numpy as np
from scipy.signal import sosfilt


FILTER_TYPES = [
    "Bell",
    "Low Shelf",
    "High Shelf",
    "Low Cut",
    "High Cut",
    "Notch",
    "Band Pass",
]

SLOPE_OPTIONS = [12, 24, 36, 48]

CHANNEL_MODES = ["LR", "L", "R", "Mid", "Side"]

TARGET_OUTPUTS = ["A+B", "A", "B"]

BAND_COLORS = [
    "#00E5FF",  # Cian
    "#00E676",  # Verde Esmeralda
    "#FFB300",  # Ámbar
    "#FF5252",  # Coral
    "#7C4DFF",  # Violeta
    "#2979FF",  # Azul Eléctrico
    "#FF4081",  # Rosa
    "#FFD54F",  # Dorado
]


def eval_sos_mag_db_fast(sos: np.ndarray, z1: np.ndarray, z2: np.ndarray) -> np.ndarray:
    """
    Evalúa la respuesta en frecuencia en dB de una matriz SOS de forma 100% vectorizada en NumPy
    usando los fasores precalculados z^-1 = exp(-j*w) y z^-2 = exp(-2j*w), evitando el overhead
    de llamar a scipy.signal.sosfreqz en bucles de 60 FPS.
    """
    b0 = sos[:, 0:1]
    b1 = sos[:, 1:2]
    b2 = sos[:, 2:3]
    a1 = sos[:, 4:5]
    a2 = sos[:, 5:6]

    num = b0 + b1 * z1 + b2 * z2
    den = 1.0 + a1 * z1 + a2 * z2
    h_sections = np.abs(num / den)
    mag = np.prod(h_sections, axis=0) if sos.shape[0] > 1 else h_sections[0]
    return 20.0 * np.log10(np.maximum(mag, 1e-6))


def design_biquad_sos(
    filter_type: str,
    freq: float,
    gain_db: float,
    q: float,
    sr: float,
    slope: int = 12,
) -> np.ndarray:
    """
    Diseña secciones de segundo orden (SOS) en numpy puro usando las ecuaciones
    del Audio EQ Cookbook (Robert Bristow-Johnson).
    """
    sr = float(max(sr, 8000.0))
    f0 = float(np.clip(freq, 10.0, 0.495 * sr))
    q_val = float(np.clip(q, 0.1, 24.0))
    g_db = float(np.clip(gain_db, -40.0, 40.0))

    w0 = 2.0 * np.pi * f0 / sr
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / (2.0 * q_val)
    A = 10.0 ** (g_db / 40.0)

    if filter_type == "Bell":
        b0 = 1.0 + alpha * A
        b1 = -2.0 * cos_w0
        b2 = 1.0 - alpha * A
        a0 = 1.0 + alpha / A
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha / A
        return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)

    elif filter_type == "Low Shelf":
        two_sqrt_A_alpha = 2.0 * np.sqrt(A) * alpha
        b0 = A * ((A + 1.0) - (A - 1.0) * cos_w0 + two_sqrt_A_alpha)
        b1 = 2.0 * A * ((A - 1.0) - (A + 1.0) * cos_w0)
        b2 = A * ((A + 1.0) - (A - 1.0) * cos_w0 - two_sqrt_A_alpha)
        a0 = (A + 1.0) + (A - 1.0) * cos_w0 + two_sqrt_A_alpha
        a1 = -2.0 * ((A - 1.0) + (A + 1.0) * cos_w0)
        a2 = (A + 1.0) + (A - 1.0) * cos_w0 - two_sqrt_A_alpha
        return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)

    elif filter_type == "High Shelf":
        two_sqrt_A_alpha = 2.0 * np.sqrt(A) * alpha
        b0 = A * ((A + 1.0) + (A - 1.0) * cos_w0 + two_sqrt_A_alpha)
        b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cos_w0)
        b2 = A * ((A + 1.0) + (A - 1.0) * cos_w0 - two_sqrt_A_alpha)
        a0 = (A + 1.0) - (A - 1.0) * cos_w0 + two_sqrt_A_alpha
        a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cos_w0)
        a2 = (A + 1.0) - (A - 1.0) * cos_w0 - two_sqrt_A_alpha
        return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)

    elif filter_type in ("Low Cut", "High Cut"):
        n_sections = max(1, min(4, int(slope) // 12))
        order = 2 * n_sections
        sections = []
        q_user_scale = q_val / 0.70710678
        for k in range(n_sections):
            theta = np.pi * (2 * k + 1) / (2.0 * order)
            q_butter = 1.0 / (2.0 * np.cos(theta))
            q_sec = float(np.clip(q_butter * (q_user_scale if k == n_sections - 1 else 1.0), 0.1, 24.0))
            alpha_k = sin_w0 / (2.0 * q_sec)
            if filter_type == "Low Cut":
                b0 = (1.0 + cos_w0) * 0.5
                b1 = -(1.0 + cos_w0)
                b2 = (1.0 + cos_w0) * 0.5
            else:
                b0 = (1.0 - cos_w0) * 0.5
                b1 = 1.0 - cos_w0
                b2 = (1.0 - cos_w0) * 0.5
            a0 = 1.0 + alpha_k
            a1 = -2.0 * cos_w0
            a2 = 1.0 - alpha_k
            sections.append([b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0])
        return np.array(sections, dtype=np.float64)

    elif filter_type == "Notch":
        b0 = 1.0
        b1 = -2.0 * cos_w0
        b2 = 1.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha
        return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)

    elif filter_type == "Band Pass":
        b0 = alpha
        b1 = 0.0
        b2 = -alpha
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha
        return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)

    return np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float64)


def design_isolate_bandpass_sos(
    freq: float,
    q: float,
    sr: float,
    cascade_sections: int = 2,
) -> np.ndarray:
    """
    Diseña en tiempo real un filtro Band-Pass Biquad (2do orden o 4to orden en cascada)
    centrado exactamente en `freq` (f0) y con el factor de calidad `q` (Q) para Isolate:
      w0 = 2*pi*f0 / fs, alpha = sin(w0) / (2*Q)
      b0 = alpha, b1 = 0, b2 = -alpha
      a0 = 1 + alpha, a1 = -2*cos(w0), a2 = 1 - alpha
    """
    sr_f = float(max(sr, 8000.0))
    f0 = float(np.clip(freq, 10.0, 0.495 * sr_f))
    q_val = float(np.clip(q, 0.1, 24.0))

    w0 = 2.0 * np.pi * f0 / sr_f
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / (2.0 * q_val)

    b0 = alpha
    b1 = 0.0
    b2 = -alpha
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha

    row = [b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]
    n_sec = max(1, int(cascade_sections))
    return np.array([row for _ in range(n_sec)], dtype=np.float64)


def _apply_sos_channel_mode(
    sos: np.ndarray,
    block: np.ndarray,
    zi: np.ndarray,
    channel_mode: str,
    zi_mid: np.ndarray | None = None,
    zi_side: np.ndarray | None = None,
) -> None:
    """
    Aplica `sosfilt` in-place sobre un bloque estéreo `(N, 2)` según `channel_mode`:
    - "LR": filtra ambos canales `[L, R]` con `zi[:, :, 0:2]`.
    - "L": filtra únicamente `block[:, 0]` con `zi[:, :, 0]`, dejando `block[:, 1]` intacto.
    - "R": filtra únicamente `block[:, 1]` con `zi[:, :, 1]`, dejando `block[:, 0]` intacto.
    - "Mid": codifica `mid = 0.5*(L+R)` y `side = 0.5*(L-R)`, filtra `mid` con `zi_mid` (o `zi[:, :, 0]`),
      y reconstruye `L = mid + side`, `R = mid - side`.
    - "Side": codifica `mid = 0.5*(L+R)` y `side = 0.5*(L-R)`, filtra `side` con `zi_side` (o `zi[:, :, 1]`),
      y reconstruye `L = mid + side`, `R = mid - side`.
    """
    if block.ndim == 1:
        block[:], zi[:, :, 0] = sosfilt(sos, block, zi=zi[:, :, 0])
        return
    if block.shape[1] < 2:
        block[:, 0], zi[:, :, 0] = sosfilt(sos, block[:, 0], zi=zi[:, :, 0])
        return

    if channel_mode == "L":
        block[:, 0], zi[:, :, 0] = sosfilt(sos, block[:, 0], zi=zi[:, :, 0])
    elif channel_mode == "R":
        block[:, 1], zi[:, :, 1] = sosfilt(sos, block[:, 1], zi=zi[:, :, 1])
    elif channel_mode == "Mid":
        ch_l = block[:, 0]
        ch_r = block[:, 1]
        mid = 0.5 * (ch_l + ch_r)
        side = 0.5 * (ch_l - ch_r)
        if zi_mid is not None:
            mid_f, zi_mid[:] = sosfilt(sos, mid, zi=zi_mid)
        else:
            mid_f, zi[:, :, 0] = sosfilt(sos, mid, zi=zi[:, :, 0])
        block[:, 0] = mid_f + side
        block[:, 1] = mid_f - side
    elif channel_mode == "Side":
        ch_l = block[:, 0]
        ch_r = block[:, 1]
        mid = 0.5 * (ch_l + ch_r)
        side = 0.5 * (ch_l - ch_r)
        if zi_side is not None:
            side_f, zi_side[:] = sosfilt(sos, side, zi=zi_side)
        else:
            side_f, zi[:, :, 1] = sosfilt(sos, side, zi=zi[:, :, 1])
        block[:, 0] = mid + side_f
        block[:, 1] = mid - side_f
    else:  # "LR"
        block[:, 0:2], zi[:, :, 0:2] = sosfilt(sos, block[:, 0:2], axis=0, zi=zi[:, :, 0:2])


def _apply_isolate_channel_mode(
    sos: np.ndarray,
    indata_f64: np.ndarray,
    zi: np.ndarray,
    channel_mode: str,
    zi_mid: np.ndarray | None = None,
    zi_side: np.ndarray | None = None,
) -> np.ndarray:
    """
    Aplica el filtro Band-Pass de Isolate respetando `channel_mode` y devolviendo únicamente
    la porción aislada del espectro (muteando el resto de frecuencias y canales no afectados).
    """
    out = np.zeros_like(indata_f64)
    if indata_f64.ndim == 1 or indata_f64.shape[1] < 2:
        col = indata_f64 if indata_f64.ndim == 1 else indata_f64[:, 0]
        filt, zi[:, :, 0] = sosfilt(sos, col, zi=zi[:, :, 0])
        if out.ndim == 1:
            out[:] = filt
        else:
            out[:, 0] = filt
        return out

    if channel_mode == "L":
        out[:, 0], zi[:, :, 0] = sosfilt(sos, indata_f64[:, 0], zi=zi[:, :, 0])
        out[:, 1] = 0.0
    elif channel_mode == "R":
        out[:, 0] = 0.0
        out[:, 1], zi[:, :, 1] = sosfilt(sos, indata_f64[:, 1], zi=zi[:, :, 1])
    elif channel_mode == "Mid":
        mid = 0.5 * (indata_f64[:, 0] + indata_f64[:, 1])
        if zi_mid is not None:
            mid_iso, zi_mid[:] = sosfilt(sos, mid, zi=zi_mid)
        else:
            mid_iso, zi[:, :, 0] = sosfilt(sos, mid, zi=zi[:, :, 0])
        out[:, 0] = mid_iso
        out[:, 1] = mid_iso
    elif channel_mode == "Side":
        side = 0.5 * (indata_f64[:, 0] - indata_f64[:, 1])
        if zi_side is not None:
            side_iso, zi_side[:] = sosfilt(sos, side, zi=zi_side)
        else:
            side_iso, zi[:, :, 1] = sosfilt(sos, side, zi=zi[:, :, 1])
        out[:, 0] = side_iso
        out[:, 1] = -side_iso
    else:  # "LR"
        out[:, 0:2], zi[:, :, 0:2] = sosfilt(sos, indata_f64[:, 0:2], axis=0, zi=zi[:, :, 0:2])
    return out


class EQBand:
    """
    Representa una banda paramétrica y dinámica individual del motor BDJ-FFT-EQ.
    Soporta:
    - Procesamiento por canal (`channel_mode`: "LR", "L", "R", "Mid", "Side")
    - Asignación de bus de salida (`target_output`: "A", "B", "A+B") con estados `zi_A` y `zi_B` independientes
    - Filtro Band-Pass de Isolate en tiempo real con estados `iso_zi_A` e `iso_zi_B`
    - Caché de respuesta en frecuencia estática (`_cached_static_db`) para evitar recálculos en IDLE.
    """

    def __init__(
        self,
        freq: float = 1000.0,
        gain_db: float = 0.0,
        q: float = 1.0,
        filter_type: str = "Bell",
        slope: int = 12,
        enabled: bool = True,
        dyn_enabled: bool = False,
        dyn_range_db: float = 0.0,
        threshold_db: float = -24.0,
        attack_ms: float = 15.0,
        release_ms: float = 120.0,
        color: str = "#00E5FF",
        sr: float = 48000.0,
        channels: int = 2,
        channel_mode: str = "LR",
        target_output: str = "A+B",
    ):
        self.sr = float(max(sr, 8000.0))
        self.channels = max(2, int(channels))

        self.enabled = bool(enabled)
        self.solo = False
        self.filter_type = filter_type if filter_type in FILTER_TYPES else "Bell"
        max_nyq = max(24000.0, self.sr * 0.495)
        self.freq = float(np.clip(freq, 10.0, max_nyq))
        self.gain_db = float(np.clip(gain_db, -24.0, 24.0))
        self.q = float(np.clip(q, 0.1, 24.0))
        self.slope = int(slope) if int(slope) in SLOPE_OPTIONS else 12
        self.color = color

        self.channel_mode = channel_mode if channel_mode in CHANNEL_MODES else "LR"
        self.target_output = target_output if target_output in TARGET_OUTPUTS else "A+B"

        # Parámetros Dynamic EQ
        self.dyn_enabled = bool(dyn_enabled)
        self.dyn_range_db = float(np.clip(dyn_range_db, -30.0, 30.0))
        self.threshold_db = float(np.clip(threshold_db, -70.0, 0.0))
        self.attack_ms = float(np.clip(attack_ms, 0.5, 500.0))
        self.release_ms = float(np.clip(release_ms, 5.0, 2000.0))

        # Estado dinámico en tiempo real
        self.env_db = -90.0
        self.current_dyn_offset_db = 0.0

        self.sos_static = design_biquad_sos(
            self.filter_type, self.freq, self.gain_db, self.q, self.sr, self.slope
        )
        self.sos_inst = self.sos_static.copy()
        self.sc_sos = design_biquad_sos("Band Pass", self.freq, 0.0, self.q, self.sr)
        self.iso_sos = design_isolate_bandpass_sos(self.freq, self.q, self.sr, cascade_sections=2)

        # Estados de filtro independientes para Bus A (1-2) y Bus B (3-4)
        n_sec = self.sos_static.shape[0]
        self.zi_A = np.zeros((n_sec, 2, self.channels), dtype=np.float64)
        self.zi_B = np.zeros((n_sec, 2, self.channels), dtype=np.float64)
        self.zi_mid_A = np.zeros((n_sec, 2), dtype=np.float64)
        self.zi_side_A = np.zeros((n_sec, 2), dtype=np.float64)
        self.zi_mid_B = np.zeros((n_sec, 2), dtype=np.float64)
        self.zi_side_B = np.zeros((n_sec, 2), dtype=np.float64)

        iso_sec = self.iso_sos.shape[0]
        self.iso_zi_A = np.zeros((iso_sec, 2, self.channels), dtype=np.float64)
        self.iso_zi_B = np.zeros((iso_sec, 2, self.channels), dtype=np.float64)
        self.iso_zi_mid_A = np.zeros((iso_sec, 2), dtype=np.float64)
        self.iso_zi_side_A = np.zeros((iso_sec, 2), dtype=np.float64)
        self.iso_zi_mid_B = np.zeros((iso_sec, 2), dtype=np.float64)
        self.iso_zi_side_B = np.zeros((iso_sec, 2), dtype=np.float64)

        self.sc_zi = np.zeros((1, 2), dtype=np.float64)

        # Caché de respuesta en frecuencia para no recalcular en IDLE
        self._cache_valid = False
        self._cached_static_db: np.ndarray | None = None
        self._cached_max_dyn_db: np.ndarray | None = None

    @property
    def zi(self) -> np.ndarray:
        return self.zi_A

    @zi.setter
    def zi(self, val: np.ndarray):
        self.zi_A = val

    @property
    def zi_a(self) -> np.ndarray:
        return self.zi_A

    @zi_a.setter
    def zi_a(self, val: np.ndarray):
        self.zi_A = val

    @property
    def zi_b(self) -> np.ndarray:
        return self.zi_B

    @zi_b.setter
    def zi_b(self, val: np.ndarray):
        self.zi_B = val

    @property
    def zi_mid_a(self) -> np.ndarray:
        return self.zi_mid_A

    @zi_mid_a.setter
    def zi_mid_a(self, val: np.ndarray):
        self.zi_mid_A = val

    @property
    def zi_side_a(self) -> np.ndarray:
        return self.zi_side_A

    @zi_side_a.setter
    def zi_side_a(self, val: np.ndarray):
        self.zi_side_A = val

    @property
    def zi_mid_b(self) -> np.ndarray:
        return self.zi_mid_B

    @zi_mid_b.setter
    def zi_mid_b(self, val: np.ndarray):
        self.zi_mid_B = val

    @property
    def zi_side_b(self) -> np.ndarray:
        return self.zi_side_B

    @zi_side_b.setter
    def zi_side_b(self, val: np.ndarray):
        self.zi_side_B = val

    @property
    def is_dynamic_active(self) -> bool:
        return self.dyn_enabled and abs(self.dyn_range_db) >= 0.05

    @property
    def supports_gain(self) -> bool:
        return self.filter_type in ("Bell", "Low Shelf", "High Shelf")

    @property
    def supports_slope(self) -> bool:
        return self.filter_type in ("Low Cut", "High Cut")

    def invalidate_cache(self):
        self._cache_valid = False

    def update_coefficients(self, sr: float | None = None, channels: int | None = None):
        if sr is not None and sr != self.sr:
            self.sr = float(max(sr, 8000.0))
        if channels is not None and channels != self.channels:
            self.channels = max(2, int(channels))

        max_nyq = max(24000.0, self.sr * 0.495)
        self.freq = float(np.clip(self.freq, 10.0, max_nyq))

        new_sos = design_biquad_sos(
            self.filter_type, self.freq, self.gain_db, self.q, self.sr, self.slope
        )
        req_shape = (new_sos.shape[0], 2, self.channels)
        mid_shape = (new_sos.shape[0], 2)
        if self.zi_A.shape != req_shape:
            self.zi_A = np.zeros(req_shape, dtype=np.float64)
        if self.zi_B.shape != req_shape:
            self.zi_B = np.zeros(req_shape, dtype=np.float64)
        if self.zi_mid_A.shape != mid_shape:
            self.zi_mid_A = np.zeros(mid_shape, dtype=np.float64)
        if self.zi_side_A.shape != mid_shape:
            self.zi_side_A = np.zeros(mid_shape, dtype=np.float64)
        if self.zi_mid_B.shape != mid_shape:
            self.zi_mid_B = np.zeros(mid_shape, dtype=np.float64)
        if self.zi_side_B.shape != mid_shape:
            self.zi_side_B = np.zeros(mid_shape, dtype=np.float64)

        self.sos_static = new_sos
        if not self.is_dynamic_active:
            self.sos_inst = new_sos.copy()
            self.current_dyn_offset_db = 0.0
        else:
            self._update_instant_sos()

        self.sc_sos = design_biquad_sos("Band Pass", self.freq, 0.0, self.q, self.sr)
        if self.sc_zi.shape != (1, 2):
            self.sc_zi = np.zeros((1, 2), dtype=np.float64)

        self.iso_sos = design_isolate_bandpass_sos(self.freq, self.q, self.sr, cascade_sections=2)
        iso_shape = (self.iso_sos.shape[0], 2, self.channels)
        iso_mid_shape = (self.iso_sos.shape[0], 2)
        if self.iso_zi_A.shape != iso_shape:
            self.iso_zi_A = np.zeros(iso_shape, dtype=np.float64)
        if self.iso_zi_B.shape != iso_shape:
            self.iso_zi_B = np.zeros(iso_shape, dtype=np.float64)
        if self.iso_zi_mid_A.shape != iso_mid_shape:
            self.iso_zi_mid_A = np.zeros(iso_mid_shape, dtype=np.float64)
        if self.iso_zi_side_A.shape != iso_mid_shape:
            self.iso_zi_side_A = np.zeros(iso_mid_shape, dtype=np.float64)
        if self.iso_zi_mid_B.shape != iso_mid_shape:
            self.iso_zi_mid_B = np.zeros(iso_mid_shape, dtype=np.float64)
        if self.iso_zi_side_B.shape != iso_mid_shape:
            self.iso_zi_side_B = np.zeros(iso_mid_shape, dtype=np.float64)

        self._cache_valid = False

    def _update_instant_sos(self):
        g_inst = self.gain_db + self.current_dyn_offset_db
        if self.supports_gain:
            self.sos_inst = design_biquad_sos(
                self.filter_type, self.freq, g_inst, self.q, self.sr, self.slope
            )
        else:
            q_mod = self.q * (2.0 ** (self.current_dyn_offset_db / 18.0))
            self.sos_inst = design_biquad_sos(
                self.filter_type, self.freq, 0.0, q_mod, self.sr, self.slope
            )

    def reset_state(self):
        self.zi_A.fill(0.0)
        self.zi_B.fill(0.0)
        self.zi_mid_A.fill(0.0)
        self.zi_side_A.fill(0.0)
        self.zi_mid_B.fill(0.0)
        self.zi_side_B.fill(0.0)
        self.iso_zi_A.fill(0.0)
        self.iso_zi_B.fill(0.0)
        self.iso_zi_mid_A.fill(0.0)
        self.iso_zi_side_A.fill(0.0)
        self.iso_zi_mid_B.fill(0.0)
        self.iso_zi_side_B.fill(0.0)
        self.sc_zi.fill(0.0)
        self.env_db = -90.0
        self.current_dyn_offset_db = 0.0

    def get_static_freq_response_db(self, z1: np.ndarray, z2: np.ndarray) -> np.ndarray:
        if not self._cache_valid or self._cached_static_db is None or len(self._cached_static_db) != len(z1):
            self._cached_static_db = eval_sos_mag_db_fast(self.sos_static, z1, z2)
            self._cached_max_dyn_db = None
            self._cache_valid = True
        return self._cached_static_db

    def get_dynamic_freq_response_db(self, z1: np.ndarray, z2: np.ndarray) -> np.ndarray:
        return eval_sos_mag_db_fast(self.sos_inst, z1, z2)

    def get_isolate_freq_response_db(self, z1: np.ndarray, z2: np.ndarray) -> np.ndarray:
        return eval_sos_mag_db_fast(self.iso_sos, z1, z2)

    def get_max_dynamic_freq_response_db(self, z1: np.ndarray, z2: np.ndarray) -> np.ndarray:
        if self._cache_valid and self._cached_max_dyn_db is not None and len(self._cached_max_dyn_db) == len(z1):
            return self._cached_max_dyn_db
        g_max = self.gain_db + self.dyn_range_db
        if self.supports_gain:
            sos_max = design_biquad_sos(
                self.filter_type, self.freq, g_max, self.q, self.sr, self.slope
            )
        else:
            q_mod = self.q * (2.0 ** (self.dyn_range_db / 18.0))
            sos_max = design_biquad_sos(
                self.filter_type, self.freq, 0.0, q_mod, self.sr, self.slope
            )
        self._cached_max_dyn_db = eval_sos_mag_db_fast(sos_max, z1, z2)
        return self._cached_max_dyn_db

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "solo": self.solo,
            "filter_type": self.filter_type,
            "freq": round(self.freq, 2),
            "gain_db": round(self.gain_db, 2),
            "q": round(self.q, 3),
            "slope": self.slope,
            "channel_mode": self.channel_mode,
            "target_output": self.target_output,
            "dyn_enabled": self.dyn_enabled,
            "dyn_range_db": round(self.dyn_range_db, 2),
            "threshold_db": round(self.threshold_db, 2),
            "attack_ms": round(self.attack_ms, 1),
            "release_ms": round(self.release_ms, 1),
            "color": self.color,
        }

    @classmethod
    def from_dict(cls, data: dict, sr: float = 48000.0, channels: int = 2, fallback_color: str = "#00E5FF"):
        b = cls(
            freq=float(data.get("freq", 1000.0)),
            gain_db=float(data.get("gain_db", 0.0)),
            q=float(data.get("q", 1.0)),
            filter_type=str(data.get("filter_type", "Bell")),
            slope=int(data.get("slope", 12)),
            enabled=bool(data.get("enabled", True)),
            dyn_enabled=bool(data.get("dyn_enabled", False)),
            dyn_range_db=float(data.get("dyn_range_db", 0.0)),
            threshold_db=float(data.get("threshold_db", -24.0)),
            attack_ms=float(data.get("attack_ms", 15.0)),
            release_ms=float(data.get("release_ms", 120.0)),
            color=str(data.get("color", fallback_color)),
            sr=sr,
            channels=channels,
            channel_mode=str(data.get("channel_mode", "LR")),
            target_output=str(data.get("target_output", "A+B")),
        )
        b.solo = bool(data.get("solo", False))
        return b


class MasterLimiter:
    """
    Limitador Master de Picos en Tiempo Real estilo 'Fruity Limiter'.
    - Previene clipping digital asegurando que la salida no supere `ceiling_db`.
    - Envelope follower con constantes de tiempo de Attack y Release.
    - Expone `last_gr_db` y `last_gr_linear` para la visualización en el osciloscopio y vúmetro de reducción de ganancia.
    """

    def __init__(
        self,
        sr: float = 48000.0,
        ceiling_db: float = -0.1,
        threshold_db: float = 0.0,
        attack_ms: float = 2.0,
        release_ms: float = 50.0,
        enabled: bool = True,
    ):
        self.sr = float(sr)
        self.ceiling_db = float(ceiling_db)
        self.threshold_db = float(threshold_db)
        self.attack_ms = float(attack_ms)
        self.release_ms = float(release_ms)
        self.enabled = bool(enabled)

        self.current_gain_linear = 1.0
        self.last_gr_db = 0.0
        self.last_gr_linear = 1.0

        self._update_coefficients()

    def set_sample_rate(self, sr: float):
        self.sr = max(1000.0, float(sr))
        self._update_coefficients()

    def _update_coefficients(self):
        self.thresh_lin = float(10.0 ** (self.threshold_db / 20.0))
        self.ceil_lin = float(10.0 ** (self.ceiling_db / 20.0))

        # Sub-bloque de 8 muestras para suavizado ultrarrápido sin aliasing
        self.sub_size = 8
        dt = self.sub_size / self.sr
        att_sec = max(1e-5, self.attack_ms / 1000.0)
        rel_sec = max(1e-4, self.release_ms / 1000.0)
        self.alpha_att = float(np.exp(-dt / att_sec))
        self.alpha_rel = float(np.exp(-dt / rel_sec))

    def set_params(
        self,
        ceiling_db: float | None = None,
        threshold_db: float | None = None,
        attack_ms: float | None = None,
        release_ms: float | None = None,
        enabled: bool | None = None,
    ):
        if ceiling_db is not None:
            self.ceiling_db = float(ceiling_db)
        if threshold_db is not None:
            self.threshold_db = float(threshold_db)
        if attack_ms is not None:
            self.attack_ms = float(attack_ms)
        if release_ms is not None:
            self.release_ms = float(release_ms)
        if enabled is not None:
            self.enabled = bool(enabled)
        self._update_coefficients()

    def process(self, block: np.ndarray) -> np.ndarray:
        if not self.enabled or block.size == 0:
            self.last_gr_db = 0.0
            self.last_gr_linear = 1.0
            return block

        frames = block.shape[0]
        # Pico absoluto por muestra en canales L y R
        if block.ndim > 1:
            peak_env = np.max(np.abs(block[:, 0:2]), axis=1)
        else:
            peak_env = np.abs(block)

        max_peak = float(np.max(peak_env))
        # Si la señal está por debajo del umbral y la ganancia ya se recuperó completamente:
        if max_peak <= self.thresh_lin and self.current_gain_linear >= 0.999:
            self.current_gain_linear = 1.0
            self.last_gr_db = 0.0
            self.last_gr_linear = 1.0
            return block

        # Ganancia objetivo instantánea
        target_lin = np.where(
            peak_env > self.thresh_lin,
            np.minimum(1.0, self.ceil_lin / (peak_env + 1e-12)),
            1.0,
        )

        sub_sz = self.sub_size
        n_sub = frames // sub_sz
        if n_sub > 0:
            rem = frames % sub_sz
            main_len = n_sub * sub_sz
            target_sub = np.min(target_lin[:main_len].reshape(n_sub, sub_sz), axis=1)

            gr_sub = np.empty(n_sub, dtype=np.float32)
            g = self.current_gain_linear
            a_att = self.alpha_att
            a_rel = self.alpha_rel

            for k in range(n_sub):
                tgt = target_sub[k]
                if tgt < g:
                    g = a_att * g + (1.0 - a_att) * tgt
                else:
                    g = a_rel * g + (1.0 - a_rel) * tgt
                gr_sub[k] = g

            self.current_gain_linear = g
            gr_linear = np.repeat(gr_sub, sub_sz)
            if rem > 0:
                gr_linear = np.concatenate((gr_linear, np.full(rem, g, dtype=np.float32)))
        else:
            dt_s = 1.0 / self.sr
            a_att_s = float(np.exp(-dt_s / max(1e-5, self.attack_ms / 1000.0)))
            a_rel_s = float(np.exp(-dt_s / max(1e-4, self.release_ms / 1000.0)))
            gr_linear = np.empty(frames, dtype=np.float32)
            g = self.current_gain_linear
            for k in range(frames):
                tgt = target_lin[k]
                if tgt < g:
                    g = a_att_s * g + (1.0 - a_att_s) * tgt
                else:
                    g = a_rel_s * g + (1.0 - a_rel_s) * tgt
                gr_linear[k] = g
            self.current_gain_linear = g

        out_block = block * gr_linear[:, None] if block.ndim > 1 else block * gr_linear
        # Garantía absoluta anti-clipping: jamás supera el ceiling configurado
        np.clip(out_block, -self.ceil_lin, self.ceil_lin, out=out_block)

        min_g = float(np.min(gr_linear))
        self.last_gr_linear = min_g
        self.last_gr_db = float(20.0 * np.log10(max(min_g, 1e-6)))

        return out_block

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "ceiling_db": round(self.ceiling_db, 2),
            "threshold_db": round(self.threshold_db, 2),
            "attack_ms": round(self.attack_ms, 2),
            "release_ms": round(self.release_ms, 2),
        }

    def load_from_dict(self, data: dict):
        if not isinstance(data, dict):
            return
        self.enabled = bool(data.get("enabled", True))
        self.ceiling_db = float(np.clip(data.get("ceiling_db", -0.1), -12.0, 0.0))
        self.threshold_db = float(np.clip(data.get("threshold_db", 0.0), -30.0, 0.0))
        self.attack_ms = float(np.clip(data.get("attack_ms", 2.0), 0.1, 100.0))
        self.release_ms = float(np.clip(data.get("release_ms", 50.0), 1.0, 500.0))
        self._update_coefficients()


class ParametricEQEngine:
    """
    Motor DSP de Ecualización Paramétrica y Dinámica en Tiempo Real con:
    - Procesamiento por canal (`LR`, `L`, `R`, `Mid`, `Side`)
    - Función Isolate (`isolated_band_index`) con filtro Band-Pass Biquad en tiempo real
    - Doble Bus de Salida Estéreo Independiente (`Out A [1-2]` y `Out B [3-4]`)
    """

    def __init__(self, sample_rate: int = 48000, channels: int = 2):
        self.sr = float(sample_rate)
        self.channels = max(2, int(channels))
        self.bypass = False
        self.output_gain_db = 0.0
        self.isolated_band_index: int | None = None
        self.lock = threading.RLock()
        self.static_version = 1
        self.bands: list[EQBand] = []
        self.init_default_bands()
        self.limiter_a = MasterLimiter(sr=self.sr)
        self.limiter_b = MasterLimiter(sr=self.sr)

    def mark_dirty(self):
        self.static_version += 1

    def set_isolated_band(self, index: int | None):
        """Activa o desactiva el modo Isolate (Solo de Frecuencia Band-Pass) sobre la banda `index`."""
        with self.lock:
            if index is not None and 0 <= int(index) < len(self.bands):
                idx = int(index)
                self.isolated_band_index = idx
                for i, b in enumerate(self.bands):
                    b.solo = (i == idx)
            else:
                self.isolated_band_index = None
                for b in self.bands:
                    b.solo = False
            self.static_version += 1

    def _resolve_isolated_band_index(self) -> int | None:
        if self.isolated_band_index is not None:
            if 0 <= self.isolated_band_index < len(self.bands) and self.bands[self.isolated_band_index].enabled:
                return self.isolated_band_index
        for i, b in enumerate(self.bands):
            if b.enabled and b.solo:
                return i
        return None

    def to_dict(self) -> dict:
        with self.lock:
            return {
                "bypass": self.bypass,
                "output_gain_db": round(self.output_gain_db, 2),
                "bands": [b.to_dict() for b in self.bands],
                "limiter": self.limiter_a.to_dict(),
            }

    def load_from_dict(self, data: dict):
        if not isinstance(data, dict):
            return
        with self.lock:
            self.bypass = bool(data.get("bypass", False))
            self.output_gain_db = float(np.clip(data.get("output_gain_db", 0.0), -24.0, 24.0))
            self.isolated_band_index = None
            if "limiter" in data and isinstance(data["limiter"], dict):
                self.limiter_a.load_from_dict(data["limiter"])
                self.limiter_b.load_from_dict(data["limiter"])
            bands_raw = data.get("bands")
            if isinstance(bands_raw, list) and len(bands_raw) > 0:
                loaded_bands = []
                for i, b_dict in enumerate(bands_raw):
                    if isinstance(b_dict, dict):
                        color = BAND_COLORS[i % len(BAND_COLORS)]
                        b = EQBand.from_dict(b_dict, sr=self.sr, channels=self.channels, fallback_color=color)
                        if b.solo and self.isolated_band_index is None:
                            self.isolated_band_index = i
                        else:
                            b.solo = False
                        loaded_bands.append(b)
                if loaded_bands:
                    self.bands = loaded_bands
            self.static_version += 1

    def init_default_bands(self):
        with self.lock:
            self.isolated_band_index = None
            self.bands = [
                EQBand(
                    freq=80.0, gain_db=0.0, q=0.80, filter_type="Low Shelf",
                    color=BAND_COLORS[0], sr=self.sr, channels=self.channels
                ),
                EQBand(
                    freq=320.0, gain_db=0.0, q=1.20, filter_type="Bell",
                    color=BAND_COLORS[1], sr=self.sr, channels=self.channels
                ),
                EQBand(
                    freq=2500.0, gain_db=0.0, q=1.20, filter_type="Bell",
                    color=BAND_COLORS[2], sr=self.sr, channels=self.channels
                ),
                EQBand(
                    freq=10000.0, gain_db=0.0, q=0.80, filter_type="High Shelf",
                    color=BAND_COLORS[3], sr=self.sr, channels=self.channels
                ),
            ]
            self.static_version += 1

    def set_sample_rate(self, sr: int):
        with self.lock:
            self.sr = float(sr)
            for b in self.bands:
                b.update_coefficients(sr=self.sr, channels=self.channels)
                b.reset_state()
            self.limiter_a.set_sample_rate(self.sr)
            self.limiter_b.set_sample_rate(self.sr)
            self.static_version += 1

    def add_band(
        self,
        freq: float = 1000.0,
        gain_db: float = 0.0,
        q: float = 1.0,
        filter_type: str | None = None,
        dyn_enabled: bool = False,
        dyn_range_db: float = 0.0,
        channel_mode: str = "LR",
        target_output: str = "A+B",
    ) -> EQBand:
        with self.lock:
            if filter_type is None:
                if freq < 45.0 and abs(gain_db) < 3.0:
                    filter_type = "Low Cut"
                elif freq < 110.0:
                    filter_type = "Low Shelf"
                elif freq > 16000.0 and abs(gain_db) < 3.0:
                    filter_type = "High Cut"
                elif freq > 9000.0:
                    filter_type = "High Shelf"
                else:
                    filter_type = "Bell"

            color = BAND_COLORS[len(self.bands) % len(BAND_COLORS)]
            band = EQBand(
                freq=freq,
                gain_db=gain_db,
                q=q,
                filter_type=filter_type,
                enabled=True,
                dyn_enabled=dyn_enabled,
                dyn_range_db=dyn_range_db,
                color=color,
                sr=self.sr,
                channels=self.channels,
                channel_mode=channel_mode,
                target_output=target_output,
            )
            self.bands.append(band)
            self.static_version += 1
            return band

    def remove_band(self, index: int):
        with self.lock:
            if 0 <= index < len(self.bands):
                self.bands.pop(index)
                if self.isolated_band_index == index:
                    self.isolated_band_index = None
                elif self.isolated_band_index is not None and self.isolated_band_index > index:
                    self.isolated_band_index -= 1
                self.static_version += 1

    def clear_bands(self):
        with self.lock:
            self.bands.clear()
            self.isolated_band_index = None
            self.static_version += 1

    def get_active_bands(self, target_bus: str | None = None) -> list[EQBand]:
        iso_idx = self._resolve_isolated_band_index()
        if iso_idx is not None:
            b = self.bands[iso_idx]
            if target_bus == "A" and b.target_output not in ("A", "A+B"):
                return []
            if target_bus == "B" and b.target_output not in ("B", "A+B"):
                return []
            return [b]

        if target_bus == "A":
            return [b for b in self.bands if b.enabled and b.target_output in ("A", "A+B")]
        if target_bus == "B":
            return [b for b in self.bands if b.enabled and b.target_output in ("B", "A+B")]
        return [b for b in self.bands if b.enabled]

    def process_dual_block(self, indata: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Procesa `indata` generando dos bloques estéreo independientes `(out_A, out_B)`:
        - `out_A`: procesado con todas las bandas activas cuyo `target_output` sea "A" o "A+B" (usando `zi_A`).
        - `out_B`: procesado con todas las bandas activas cuyo `target_output` sea "B" o "A+B" (usando `zi_B`).
        Soporta Isolate (`isolated_band_index`) y procesamiento por canal (`LR`, `L`, `R`, `Mid`, `Side`).
        """
        if indata.ndim == 1:
            stereo_in = np.column_stack((indata, indata))
        elif indata.shape[1] == 1:
            stereo_in = np.repeat(indata, 2, axis=1)
        else:
            stereo_in = indata[:, 0:2]

        if self.bypass:
            out_f32 = stereo_in.astype(np.float32, copy=True)
            return out_f32, out_f32.copy()

        with self.lock:
            iso_idx = self._resolve_isolated_band_index()

            # --- MODO ISOLATE (SOLO DE FRECUENCIA BAND-PASS) ---
            if iso_idx is not None:
                iso_band = self.bands[iso_idx]
                in_f64 = stereo_in.astype(np.float64, copy=True)

                # Escalar suavemente según la ganancia de monitoreo y de la banda
                band_boost = min(12.0, max(-12.0, iso_band.gain_db * 0.5)) if iso_band.supports_gain else 0.0
                mon_gain_db = float(np.clip(band_boost + self.output_gain_db, -24.0, 12.0))
                gain_scale = 10.0 ** (mon_gain_db / 20.0)

                if iso_band.target_output in ("A", "A+B"):
                    out_A = _apply_isolate_channel_mode(
                        iso_band.iso_sos, in_f64, iso_band.iso_zi_A, iso_band.channel_mode,
                        zi_mid=iso_band.iso_zi_mid_A, zi_side=iso_band.iso_zi_side_A
                    ) * gain_scale
                else:
                    out_A = np.zeros_like(in_f64)

                if iso_band.target_output in ("B", "A+B"):
                    out_B = _apply_isolate_channel_mode(
                        iso_band.iso_sos, in_f64, iso_band.iso_zi_B, iso_band.channel_mode,
                        zi_mid=iso_band.iso_zi_mid_B, zi_side=iso_band.iso_zi_side_B
                    ) * gain_scale
                else:
                    out_B = np.zeros_like(in_f64)

                out_A = self.limiter_a.process(out_A)
                out_B = self.limiter_b.process(out_B)
                return out_A.astype(np.float32), out_B.astype(np.float32)

            # --- MODO NORMAL DUAL BUS (OUT A + OUT B) ---
            enabled_bands = [b for b in self.bands if b.enabled]
            any_filtering = any(
                b.is_dynamic_active or abs(b.gain_db) > 0.01 or not b.supports_gain
                for b in enabled_bands
            )
            if not any_filtering and abs(self.output_gain_db) < 0.01:
                for b in self.bands:
                    if abs(b.current_dyn_offset_db) > 0.001:
                        b.current_dyn_offset_db *= 0.85
                    else:
                        b.current_dyn_offset_db = 0.0
                out_A = self.limiter_a.process(stereo_in.astype(np.float64, copy=True))
                out_B = self.limiter_b.process(stereo_in.astype(np.float64, copy=True))
                return out_A.astype(np.float32), out_B.astype(np.float32)

            out_A = stereo_in.astype(np.float64, copy=True)
            out_B = stereo_in.astype(np.float64, copy=True)
            frames = out_A.shape[0]
            block_duration = frames / self.sr

            mono_in = 0.5 * (stereo_in[:, 0] + stereo_in[:, 1])
            side_in = 0.5 * (stereo_in[:, 0] - stereo_in[:, 1])

            for band in self.bands:
                if not band.enabled:
                    band.current_dyn_offset_db = (
                        0.0 if abs(band.current_dyn_offset_db) < 0.01 else band.current_dyn_offset_db * 0.85
                    )
                    continue

                if band.is_dynamic_active:
                    if band.channel_mode == "L":
                        sc_in = stereo_in[:, 0]
                    elif band.channel_mode == "R":
                        sc_in = stereo_in[:, 1]
                    elif band.channel_mode == "Side":
                        sc_in = side_in
                    else:
                        sc_in = mono_in

                    band_signal, band.sc_zi = sosfilt(band.sc_sos, sc_in, zi=band.sc_zi)
                    rms = np.sqrt(np.mean(band_signal ** 2))
                    level_db = 20.0 * np.log10(rms + 1e-10)

                    att_sec = max(band.attack_ms / 1000.0, 1e-4)
                    rel_sec = max(band.release_ms / 1000.0, 1e-4)
                    alpha_att = np.exp(-block_duration / att_sec)
                    alpha_rel = np.exp(-block_duration / rel_sec)

                    if level_db > band.env_db:
                        base_env = max(band.env_db, band.threshold_db - 15.0, -70.0)
                        band.env_db = alpha_att * base_env + (1.0 - alpha_att) * level_db
                    else:
                        band.env_db = max(-90.0, alpha_rel * band.env_db + (1.0 - alpha_rel) * level_db)

                    over_db = band.env_db - band.threshold_db
                    knee_w = 6.0
                    if over_db <= -0.5 * knee_w:
                        eff_over = 0.0
                    elif over_db < 0.5 * knee_w:
                        eff_over = ((over_db + 0.5 * knee_w) ** 2) / (2.0 * knee_w)
                    else:
                        eff_over = over_db

                    if eff_over > 0.0:
                        norm_excursion = 1.0 - np.exp(-eff_over / 12.0)
                        band.current_dyn_offset_db = float(band.dyn_range_db * norm_excursion)
                    else:
                        band.current_dyn_offset_db = 0.0

                    band._update_instant_sos()
                else:
                    band.current_dyn_offset_db = 0.0
                    band.sos_inst = band.sos_static

                if band.is_dynamic_active or abs(band.gain_db) > 0.01 or not band.supports_gain:
                    if band.target_output in ("A", "A+B"):
                        _apply_sos_channel_mode(
                            band.sos_inst, out_A, band.zi_A, band.channel_mode,
                            zi_mid=band.zi_mid_A, zi_side=band.zi_side_A
                        )
                    if band.target_output in ("B", "A+B"):
                        _apply_sos_channel_mode(
                            band.sos_inst, out_B, band.zi_B, band.channel_mode,
                            zi_mid=band.zi_mid_B, zi_side=band.zi_side_B
                        )

            if abs(self.output_gain_db) >= 0.01:
                gain_lin = 10.0 ** (self.output_gain_db / 20.0)
                out_A *= gain_lin
                out_B *= gain_lin

            out_A = self.limiter_a.process(out_A)
            out_B = self.limiter_b.process(out_B)
            return out_A.astype(np.float32), out_B.astype(np.float32)

    def process_block(self, indata: np.ndarray) -> np.ndarray:
        """Procesa un bloque devolviendo la salida principal `out_A` (compatible con llamadas de 1 bus)."""
        out_A, _ = self.process_dual_block(indata)
        if indata.ndim == 1:
            return out_A[:, 0]
        if indata.shape[1] == 1:
            return out_A[:, 0:1]
        return out_A

    def get_combined_responses_db(
        self,
        z1: np.ndarray,
        z2: np.ndarray,
        target_bus: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        with self.lock:
            n = len(z1)
            total_static = np.zeros(n, dtype=np.float64)
            total_dyn = np.zeros(n, dtype=np.float64)

            if self.bypass:
                return total_static, total_dyn

            iso_idx = self._resolve_isolated_band_index()
            if iso_idx is not None:
                iso_b = self.bands[iso_idx]
                if target_bus == "A" and iso_b.target_output not in ("A", "A+B"):
                    return np.full(n, -120.0, dtype=np.float64), np.full(n, -120.0, dtype=np.float64)
                if target_bus == "B" and iso_b.target_output not in ("B", "A+B"):
                    return np.full(n, -120.0, dtype=np.float64), np.full(n, -120.0, dtype=np.float64)
                iso_resp = iso_b.get_isolate_freq_response_db(z1, z2) + self.output_gain_db
                return iso_resp, iso_resp.copy()

            active_bands = self.get_active_bands(target_bus=target_bus)
            for band in active_bands:
                static_db = band.get_static_freq_response_db(z1, z2)
                total_static += static_db
                if band.is_dynamic_active and abs(band.current_dyn_offset_db) > 0.02:
                    dyn_db = band.get_dynamic_freq_response_db(z1, z2)
                    total_dyn += dyn_db
                else:
                    total_dyn += static_db

            if abs(self.output_gain_db) >= 0.01:
                total_static += self.output_gain_db
                total_dyn += self.output_gain_db

            return total_static, total_dyn
