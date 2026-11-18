"""
audio_sync.py
=============
Ring buffer con remuestreo adaptativo (ASRC) para unir dos relojes de audio distintos
(ej. Output B en otro dispositivo, o Input 2 en otra placa) SIN saltos ni aceleraciones.

Problema que resuelve
---------------------
Dos dispositivos nunca corren exactamente a la misma frecuencia (diferencias típicas de
20–300 ppm, mucho más con Bluetooth / cables virtuales). Con un ring buffer simple, el nivel
de llenado deriva hasta que se vacía (hueco = "paquete perdido") o se llena (hay que tirar o
comprimir audio de golpe = "se acelera por un instante").

Solución
--------
- El lector consume a una tasa `step = nominal * (1 + corr)` donde `corr` es una corrección
  minúscula (por defecto máx ±0.3 %) que un lazo de control proporcional ajusta de forma
  continua para mantener el llenado alrededor del objetivo. La deriva se absorbe con cambios
  de tono inaudibles (< 5 cents) en lugar de saltos.
- La interpolación fraccionaria usa un sinc enventanado (Kaiser) de 32 taps con tabla
  polifásica: respuesta plana a cualquier fase, sin el "brillo que respira" de la
  interpolación lineal/cúbica.
- Underrun: fade-out corto, re-prebuffer y el objetivo de colchón crece automáticamente
  (jitter buffer adaptativo) para que no vuelva a ocurrir.
- Exceso grande tras un cuelgue del sistema: salto con crossfade en un solo bloque.
"""

import threading
import time
import numpy as np


class AdaptiveResamplingRingBuffer:
    TAPS = 32
    PHASES = 2048

    def __init__(
        self,
        capacity_frames: int = 32768,
        channels: int = 2,
        target_fill: int = 256,
        max_target_fill: int = 8192,
        max_correction: float = 0.003,
        kp: float = 5.0e-6,
        ema_tau_frames: float = 36000.0,
        jump_excess_frames: int = 6144,
        sample_rate: float = 48000.0,
    ):
        self.channels = max(1, int(channels))
        self.max_correction = float(max_correction)
        self.kp = float(kp)
        self.ema_tau_frames = float(ema_tau_frames)
        self.jump_excess_frames = int(jump_excess_frames)
        self.max_target_fill = int(max_target_fill)
        self.sample_rate = float(sample_rate)
        self._clock = time.perf_counter
        self._lock = threading.Lock()
        self.nominal = 1.0
        self._tap_off = np.arange(self.TAPS, dtype=np.int64)
        self._half = self.TAPS // 2
        self._arange_cache = {}
        self._build_kernel()
        self._allocate(int(capacity_frames))
        self._base_target = float(max(1, int(target_fill)))
        self._reset_state()
        # Estadísticas (leídas desde el hilo GUI)
        self.underruns = 0
        self.overflows = 0
        self.jumps = 0

    # ------------------------------------------------------------------ setup
    def _allocate(self, capacity: int):
        self.capacity = max(4096, int(capacity))
        # Canal-primero + espejo de TAPS muestras al final: las ventanas del sinc nunca
        # necesitan módulo por tap (sólo por posición), ~2x más rápido.
        self.buffer = np.zeros((self.channels, self.capacity + self.TAPS), dtype=np.float32)

    def _build_kernel(self):
        T, P = self.TAPS, self.PHASES
        fc = 0.96 * min(1.0, 1.0 / max(1e-6, self.nominal))
        m = np.arange(T, dtype=np.float64) - (self._half - 1)
        f = np.arange(P, dtype=np.float64) / P
        x = m[None, :] - f[:, None]
        beta = 8.6
        r = np.clip(x / float(self._half), -1.0, 1.0)
        w = np.i0(beta * np.sqrt(np.maximum(0.0, 1.0 - r * r))) / np.i0(beta)
        h = fc * np.sinc(fc * x) * w
        h /= np.sum(h, axis=1, keepdims=True)
        self._kern = h.astype(np.float32)

    def _reset_state(self):
        self.buffer.fill(0.0)
        self._w_total = 0
        self._r_int = 0
        self._r_frac = 0.0
        self._target_out = self._base_target
        self._fill_ema = 0.0
        self._corr = 0.0
        self._prebuffering = True
        self._fade_in = False
        self._last_out = np.zeros(self.channels, dtype=np.float32)
        self._t_last_w = None
        self._last_w_n = 0

    def reset(self, target_fill: int | None = None, capacity_frames: int | None = None,
              sample_rate: float | None = None):
        with self._lock:
            if target_fill is not None:
                self._base_target = float(max(1, int(target_fill)))
            if sample_rate is not None and sample_rate > 0:
                self.sample_rate = float(sample_rate)
            if capacity_frames is not None and int(capacity_frames) != self.capacity:
                self._allocate(int(capacity_frames))
            self._reset_state()

    def set_nominal_ratio(self, ratio: float):
        """Relación de muestreo entrada/salida (ej. 44100/48000). 1.0 = mismo sample rate."""
        ratio = float(ratio) if ratio and ratio > 0 else 1.0
        with self._lock:
            if abs(ratio - self.nominal) > 1e-9:
                self.nominal = ratio
                self._build_kernel()
                self._reset_state()

    # ------------------------------------------------------------ properties
    @property
    def target_fill(self) -> int:
        """Colchón objetivo en frames de salida (crece solo si hubo underruns)."""
        return int(round(self._target_out))

    @property
    def latency_frames(self) -> float:
        """Latencia total introducida (colchón + medio filtro sinc) en frames de salida."""
        return float(self._target_out) + self._half / max(1e-6, self.nominal)

    @property
    def current_correction_ppm(self) -> float:
        return float(self._corr) * 1e6

    # ----------------------------------------------------------------- write
    def write(self, data: np.ndarray):
        n = int(data.shape[0])
        if n <= 0:
            return
        if data.ndim == 1:
            frames = np.repeat(data[:, None], self.channels, axis=1)
        elif data.shape[1] < self.channels:
            frames = np.repeat(data[:, :1], self.channels, axis=1)
        else:
            frames = data[:, : self.channels]

        with self._lock:
            cap = self.capacity
            if n > cap // 2:
                frames = frames[-(cap // 2):]
                n = frames.shape[0]
            fill = self._w_total - self._r_int
            if fill + n > cap - (self.TAPS + 16):
                # Desborde severo (lector detenido mucho tiempo): reubicar el lector
                keep = max(int(self._target_out * self.nominal), self.TAPS + 16)
                self._r_int = self._w_total + n - keep
                self._r_frac = 0.0
                self._fill_ema = float(keep) / self.nominal
                self.overflows += 1

            w = self._w_total % cap
            first = min(n, cap - w)
            self.buffer[:, w:w + first] = frames[:first].T
            if n > first:
                self.buffer[:, : n - first] = frames[first:].T
            if w < self.TAPS or n > first:
                self.buffer[:, cap:cap + self.TAPS] = self.buffer[:, :self.TAPS]
            self._w_total += n
            self._t_last_w = self._clock()
            self._last_w_n = n

    # ------------------------------------------------------------------ read
    def _arange(self, n: int) -> np.ndarray:
        a = self._arange_cache.get(n)
        if a is None:
            a = np.arange(n, dtype=np.float64)
            self._arange_cache[n] = a
        return a

    def _render(self, r_int: int, r_frac: float, n: int, step: float):
        pos = r_frac + self._arange(n) * step
        ipos = np.floor(pos).astype(np.int64)
        phase = ((pos - ipos) * self.PHASES).astype(np.int64)
        np.minimum(phase, self.PHASES - 1, out=phase)
        start = (r_int + ipos - (self._half - 1)) % self.capacity
        idx = start[:, None] + self._tap_off[None, :]
        k = self._kern[phase]                       # (n, T)
        y = np.empty((n, self.channels), dtype=np.float32)
        for c in range(self.channels):
            y[:, c] = np.einsum("nt,nt->n", k, self.buffer[c][idx])
        new_pos = r_frac + n * step
        adv = int(np.floor(new_pos))
        return y, r_int + adv, new_pos - adv

    def _frames_needed(self, r_frac: float, n: int, step: float) -> int:
        # índice relativo máximo leído (desde r_int) + 1
        return int(np.floor(r_frac + (n - 1) * step)) + self._half + 1

    def read_into(self, out: np.ndarray) -> bool:
        """Llena `out` (n, ch). Devuelve False si no había audio (prebuffer/underrun)."""
        n = int(out.shape[0])
        if n <= 0:
            return False
        ch = min(self.channels, out.shape[1]) if out.ndim > 1 else 1
        with self._lock:
            nominal = self.nominal
            fill_in = float(self._w_total - self._r_int) - self._r_frac
            fill_out = fill_in / nominal

            # Llenado "continuo": suma las muestras que el productor ya generó desde su
            # última entrega. Elimina el diente de sierra de cuantización por bloques
            # (que si no, el lazo vería como error y modularía el tono).
            fill_ctl = fill_out
            if self._t_last_w is not None:
                elapsed = max(0.0, self._clock() - self._t_last_w)
                fill_ctl += min(elapsed * self.sample_rate, self._last_w_n / nominal)

            setpoint = self._target_out + self._half + 0.5 * self._last_w_n / nominal

            if self._prebuffering:
                if fill_out >= self._target_out + self._half:
                    self._prebuffering = False
                    self._fade_in = True
                    self._fill_ema = fill_ctl
                    self._corr = 0.0
                else:
                    out.fill(0.0)
                    return False

            # --- Lazo de control de deriva (proporcional + EMA + slew) ---
            alpha = min(1.0, n / self.ema_tau_frames)
            self._fill_ema += alpha * (fill_ctl - self._fill_ema)
            err = self._fill_ema - setpoint
            target_corr = max(-self.max_correction, min(self.max_correction, err * self.kp))
            slew = 2.0e-6 * n
            if target_corr > self._corr + slew:
                self._corr += slew
            elif target_corr < self._corr - slew:
                self._corr -= slew
            else:
                self._corr = target_corr
            step = nominal * (1.0 + self._corr)

            avail = self._w_total - self._r_int
            # Tras retroceder 'half-1' muestras hace falta historia; siempre está en buffer.
            if self._frames_needed(self._r_frac, n, step) > avail:
                # Underrun: fade-out suave desde la última muestra, re-prebuffer y colchón mayor
                fade = np.linspace(1.0, 0.0, n, dtype=np.float32)[:, None]
                if out.ndim > 1:
                    out[:, :ch] = self._last_out[None, :ch] * fade
                    if out.shape[1] > ch:
                        out[:, ch:] = 0.0
                else:
                    out[:] = self._last_out[0] * fade[:, 0]
                self._last_out[:] = 0.0
                self._prebuffering = True
                self.underruns += 1
                self._target_out = min(float(self.max_target_fill), self._target_out + max(float(n), 0.5 * self._target_out))
                return False

            y, new_int, new_frac = self._render(self._r_int, self._r_frac, n, step)

            # Exceso enorme (cuelgue del sistema): saltar al objetivo con crossfade de un bloque
            if err > self.jump_excess_frames:
                keep_in = setpoint * nominal
                j_int = self._w_total - int(np.ceil(keep_in))
                if j_int > new_int and self._frames_needed(0.0, n, step) <= self._w_total - j_int:
                    y2, new_int, new_frac = self._render(j_int, 0.0, n, step)
                    xf = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
                    y = y * (1.0 - xf) + y2 * xf
                    self._fill_ema = setpoint
                    self._corr = 0.0
                    self.jumps += 1

            self._r_int, self._r_frac = new_int, new_frac

            if self._fade_in:
                y = y * np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
                self._fade_in = False

            if out.ndim > 1:
                out[:, :ch] = y[:, :ch]
                if out.shape[1] > ch:
                    out[:, ch:] = 0.0
            else:
                out[:] = y[:, 0]
            self._last_out[:] = y[-1]
            return True
