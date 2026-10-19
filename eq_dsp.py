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


class EQBand:
    """
    Representa una banda paramétrica y dinámica individual estilo FabFilter Pro-Q 4.
    Cachea su respuesta en frecuencia estática (`_cached_static_db`) para evitar
    recálculos redundantes en el loop de 60 FPS cuando la banda no fue modificada.
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
    ):
        self.enabled = bool(enabled)
        self.solo = False
        self.filter_type = filter_type if filter_type in FILTER_TYPES else "Bell"
        self.freq = float(np.clip(freq, 10.0, 24000.0))
        self.gain_db = float(np.clip(gain_db, -24.0, 24.0))
        self.q = float(np.clip(q, 0.1, 24.0))
        self.slope = int(slope) if int(slope) in SLOPE_OPTIONS else 12
        self.color = color

        # Parámetros Dynamic EQ
        self.dyn_enabled = bool(dyn_enabled)
        self.dyn_range_db = float(np.clip(dyn_range_db, -30.0, 30.0))
        self.threshold_db = float(np.clip(threshold_db, -70.0, 0.0))
        self.attack_ms = float(np.clip(attack_ms, 0.5, 500.0))
        self.release_ms = float(np.clip(release_ms, 5.0, 2000.0))

        # Estado dinámico en tiempo real
        self.env_db = -90.0
        self.current_dyn_offset_db = 0.0

        self.sr = float(sr)
        self.channels = int(channels)

        self.sos_static = design_biquad_sos(
            self.filter_type, self.freq, self.gain_db, self.q, self.sr, self.slope
        )
        self.sos_inst = self.sos_static.copy()
        self.sc_sos = design_biquad_sos("Band Pass", self.freq, 0.0, self.q, self.sr)

        self.zi = np.zeros((self.sos_static.shape[0], 2, self.channels), dtype=np.float64)
        self.sc_zi = np.zeros((1, 2), dtype=np.float64)

        # Caché de respuesta en frecuencia para no recalcular en IDLE
        self._cache_valid = False
        self._cached_static_db: np.ndarray | None = None
        self._cached_max_dyn_db: np.ndarray | None = None

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
            self.sr = float(sr)
        if channels is not None and channels != self.channels:
            self.channels = int(channels)

        new_sos = design_biquad_sos(
            self.filter_type, self.freq, self.gain_db, self.q, self.sr, self.slope
        )
        if self.zi.shape != (new_sos.shape[0], 2, self.channels):
            self.zi = np.zeros((new_sos.shape[0], 2, self.channels), dtype=np.float64)

        self.sos_static = new_sos
        if not self.is_dynamic_active:
            self.sos_inst = new_sos.copy()
            self.current_dyn_offset_db = 0.0
        else:
            self._update_instant_sos()

        self.sc_sos = design_biquad_sos("Band Pass", self.freq, 0.0, self.q, self.sr)
        if self.sc_zi.shape != (1, 2):
            self.sc_zi = np.zeros((1, 2), dtype=np.float64)

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
        self.zi.fill(0.0)
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
        )
        b.solo = bool(data.get("solo", False))
        return b


class ParametricEQEngine:
    """
    Motor DSP de Ecualización Paramétrica y Dinámica en Tiempo Real.
    Incluye contador `static_version` para que la GUI únicamente recalcule y redibuje
    las curvas estáticas cuando el usuario modifica algún parámetro.
    """

    def __init__(self, sample_rate: int = 48000, channels: int = 2):
        self.sr = float(sample_rate)
        self.channels = int(channels)
        self.bypass = False
        self.output_gain_db = 0.0
        self.lock = threading.RLock()
        self.static_version = 1
        self.bands: list[EQBand] = []
        self.init_default_bands()

    def mark_dirty(self):
        self.static_version += 1

    def to_dict(self) -> dict:
        with self.lock:
            return {
                "bypass": self.bypass,
                "output_gain_db": round(self.output_gain_db, 2),
                "bands": [b.to_dict() for b in self.bands],
            }

    def load_from_dict(self, data: dict):
        if not isinstance(data, dict):
            return
        with self.lock:
            self.bypass = bool(data.get("bypass", False))
            self.output_gain_db = float(np.clip(data.get("output_gain_db", 0.0), -24.0, 24.0))
            bands_raw = data.get("bands")
            if isinstance(bands_raw, list) and len(bands_raw) > 0:
                loaded_bands = []
                for i, b_dict in enumerate(bands_raw):
                    if isinstance(b_dict, dict):
                        color = BAND_COLORS[i % len(BAND_COLORS)]
                        loaded_bands.append(
                            EQBand.from_dict(b_dict, sr=self.sr, channels=self.channels, fallback_color=color)
                        )
                if loaded_bands:
                    self.bands = loaded_bands
            self.static_version += 1

    def init_default_bands(self):
        with self.lock:
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
            self.static_version += 1

    def add_band(
        self,
        freq: float = 1000.0,
        gain_db: float = 0.0,
        q: float = 1.0,
        filter_type: str | None = None,
        dyn_enabled: bool = False,
        dyn_range_db: float = 0.0,
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
            )
            self.bands.append(band)
            self.static_version += 1
            return band

    def remove_band(self, index: int):
        with self.lock:
            if 0 <= index < len(self.bands):
                self.bands.pop(index)
                self.static_version += 1

    def clear_bands(self):
        with self.lock:
            self.bands.clear()
            self.static_version += 1

    def get_active_bands(self) -> list[EQBand]:
        solo_bands = [b for b in self.bands if b.enabled and b.solo]
        if solo_bands:
            return solo_bands
        return [b for b in self.bands if b.enabled]

    def process_block(self, indata: np.ndarray) -> np.ndarray:
        if self.bypass:
            return indata.copy()

        with self.lock:
            active_bands = self.get_active_bands()
            # Si ninguna banda activa altera el audio ni tiene Dynamic EQ activo, evita copiar/filtrar en vano
            any_filtering = any(
                b.is_dynamic_active or abs(b.gain_db) > 0.01 or not b.supports_gain
                for b in active_bands
            )
            if not any_filtering and abs(self.output_gain_db) < 0.01:
                for b in self.bands:
                    if abs(b.current_dyn_offset_db) > 0.001:
                        b.current_dyn_offset_db *= 0.85
                    else:
                        b.current_dyn_offset_db = 0.0
                return indata.copy()

            out = indata.astype(np.float64, copy=True)
            frames = out.shape[0]
            block_duration = frames / self.sr
            mono_in = (out[:, 0] + out[:, 1]) * 0.5 if (out.ndim > 1 and out.shape[1] > 1) else (out[:, 0] if out.ndim > 1 else out)

            for band in self.bands:
                if not band.enabled or (band not in active_bands):
                    band.current_dyn_offset_db = 0.0 if abs(band.current_dyn_offset_db) < 0.01 else band.current_dyn_offset_db * 0.85
                    continue

                if band.is_dynamic_active:
                    band_signal, band.sc_zi = sosfilt(band.sc_sos, mono_in, zi=band.sc_zi)
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
                    out, band.zi = sosfilt(band.sos_inst, out, axis=0, zi=band.zi)

            if abs(self.output_gain_db) >= 0.01:
                out *= 10.0 ** (self.output_gain_db / 20.0)

            np.clip(out, -4.0, 4.0, out=out)
            return out.astype(np.float32)

    def get_combined_responses_db(self, z1: np.ndarray, z2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with self.lock:
            n = len(z1)
            total_static = np.zeros(n, dtype=np.float64)
            total_dyn = np.zeros(n, dtype=np.float64)

            if self.bypass:
                return total_static, total_dyn

            active_bands = self.get_active_bands()
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
