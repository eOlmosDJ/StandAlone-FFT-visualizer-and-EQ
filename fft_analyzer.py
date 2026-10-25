import numpy as np


class FFTSpectrumAnalyzer:
    """
    Motor de Análisis Espectral FFT optimizado para bajo consumo de CPU.
    - Precalcula grillas logarítmicas/lineales y vectores de ventana.
    - Detecta estado de silencio/reposo (`is_at_rest`) para omitir `rfft`.
    - Interpolación vectorizada ultra-rápida (np.interp) con 0% sobrecarga de CPU.
    """

    def __init__(self, sample_rate: int = 48000, fft_size: int = 4096, decay_factor: float = 0.88):
        self.sample_rate = int(sample_rate)
        self.fft_size = int(fft_size)
        self.decay_factor = float(decay_factor)

        self.pre_ring_buffer = np.zeros(self.fft_size, dtype=np.float32)
        self.post_ring_buffer = np.zeros(self.fft_size, dtype=np.float32)
        self.post2_ring_buffer = np.zeros(self.fft_size, dtype=np.float32)

        self.held_pre_fft = None
        self.held_post_fft = None
        self.held_post2_fft = None

        self.pre_at_rest = False
        self.post_at_rest = False
        self.post2_at_rest = False
        self._last_pre_silent = True
        self._last_post_silent = True
        self._last_post2_silent = True

        self.num_points = 360
        self._rebuild_window_and_freqs()

    def _rebuild_window_and_freqs(self):
        self.window = np.hanning(self.fft_size).astype(np.float32)
        self.window_scale = float(2.0 / np.sum(self.window))
        self.freqs = np.fft.rfftfreq(self.fft_size, 1.0 / self.sample_rate)

        self.valid_mask = self.freqs >= 25.0
        self.f_real = self.freqs[self.valid_mask]

        if len(self.f_real) > 4:
            # Precalcular vectores de interpolación logarítmica y lineal una sola vez
            self.f_full_log = np.concatenate([np.array([10.0, 15.0]), self.f_real])
            self.log_f_full = np.log10(self.f_full_log)
            self.log_f_smooth = np.linspace(self.log_f_full[0], self.log_f_full[-1], self.num_points)
            self.x_smooth_log = 10.0 ** self.log_f_smooth

            self.f_full_lin = np.concatenate([np.array([0.0, 12.0]), self.f_real])
            self.x_smooth_lin = np.linspace(0.0, self.f_full_lin[-1], self.num_points)

        self.pre_at_rest = False
        self.post_at_rest = False

    def set_sample_rate(self, sample_rate: int):
        self.sample_rate = int(sample_rate)
        self.reset_buffers()
        self._rebuild_window_and_freqs()

    def set_fft_size(self, fft_size: int):
        self.fft_size = int(fft_size)
        self.reset_buffers()
        self._rebuild_window_and_freqs()

    def set_decay_factor(self, decay_factor: float):
        self.decay_factor = float(np.clip(decay_factor, 0.0, 0.999))

    def reset_buffers(self):
        self.pre_ring_buffer = np.zeros(self.fft_size, dtype=np.float32)
        self.post_ring_buffer = np.zeros(self.fft_size, dtype=np.float32)
        self.post2_ring_buffer = np.zeros(self.fft_size, dtype=np.float32)
        self.held_pre_fft = None
        self.held_post_fft = None
        self.held_post2_fft = None
        self.pre_at_rest = False
        self.post_at_rest = False
        self.post2_at_rest = False
        self._last_pre_silent = True
        self._last_post_silent = True
        self._last_post2_silent = True

    @property
    def in_ring_buffer(self):
        return self.pre_ring_buffer

    @property
    def out1_ring_buffer(self):
        return self.post_ring_buffer

    @property
    def out2_ring_buffer(self):
        return self.post2_ring_buffer

    @property
    def in_at_rest(self):
        return self.pre_at_rest

    @property
    def out1_at_rest(self):
        return self.post_at_rest

    @property
    def out2_at_rest(self):
        return self.post2_at_rest

    @staticmethod
    def _push_to_ring(ring: np.ndarray, blocks: list) -> tuple[np.ndarray, bool]:
        fft_size = len(ring)
        all_silent = True
        for blk in blocks:
            mono_blk = (blk[:, 0] + blk[:, 1]) * 0.5 if (blk.ndim > 1 and blk.shape[1] > 1) else (blk[:, 0] if blk.ndim > 1 else blk)
            if all_silent and np.max(np.abs(mono_blk)) > 1e-5:
                all_silent = False
            n = len(mono_blk)
            if n >= fft_size:
                ring[:] = mono_blk[-fft_size:]
            else:
                ring[:-n] = ring[n:]
                ring[-n:] = mono_blk
        return ring, all_silent

    def push_blocks(self, pre_blocks: list, post_blocks: list, post2_blocks: list = None):
        """Acumula bloques de audio en los 3 Ring Buffers (Input, Output 1, Output 2)."""
        if pre_blocks:
            self.pre_ring_buffer, self._last_pre_silent = self._push_to_ring(self.pre_ring_buffer, pre_blocks)
            if not self._last_pre_silent:
                self.pre_at_rest = False
        else:
            self._last_pre_silent = True

        if post_blocks:
            self.post_ring_buffer, self._last_post_silent = self._push_to_ring(self.post_ring_buffer, post_blocks)
            if not self._last_post_silent:
                self.post_at_rest = False
        else:
            self._last_post_silent = True

        if post2_blocks:
            self.post2_ring_buffer, self._last_post2_silent = self._push_to_ring(self.post2_ring_buffer, post2_blocks)
            if not self._last_post2_silent:
                self.post2_at_rest = False
        else:
            self._last_post2_silent = True

    def _compute_held_spectrum(self, ring: np.ndarray, held_prev: np.ndarray | None, is_silent: bool) -> np.ndarray:
        if is_silent and held_prev is not None:
            # Si el bloque entrante es silencio puro, solo decaemos el vector retenido sin ejecutar rfft
            return held_prev * self.decay_factor

        fft_mag = np.abs(np.fft.rfft(ring * self.window)) * self.window_scale
        if held_prev is None or len(held_prev) != len(fft_mag):
            return fft_mag
        decayed = held_prev * self.decay_factor + fft_mag * (1.0 - self.decay_factor)
        return np.maximum(fft_mag, decayed)

    def update_spectra(
        self,
        compute_pre: bool = True,
        compute_post: bool = True,
        compute_post2: bool = False,
        db_min: float = -90.0,
        compute_in: bool = None,
        compute_out1: bool = None,
        compute_out2: bool = None,
    ):
        """Actualiza las envolventes FFT omitiendo cálculos si la señal ya reposa bajo el piso."""
        if compute_in is not None:
            compute_pre = compute_in
        if compute_out1 is not None:
            compute_post = compute_out1
        if compute_out2 is not None:
            compute_post2 = compute_out2

        floor_mag = 10.0 ** ((db_min - 1.0) / 20.0)

        if compute_pre and not self.pre_at_rest:
            self.held_pre_fft = self._compute_held_spectrum(
                self.pre_ring_buffer, self.held_pre_fft, self._last_pre_silent
            )
            if self._last_pre_silent and self.held_pre_fft is not None and np.max(self.held_pre_fft) <= floor_mag:
                self.pre_at_rest = True

        if compute_post and not self.post_at_rest:
            self.held_post_fft = self._compute_held_spectrum(
                self.post_ring_buffer, self.held_post_fft, self._last_post_silent
            )
            if self._last_post_silent and self.held_post_fft is not None and np.max(self.held_post_fft) <= floor_mag:
                self.post_at_rest = True

        if compute_post2 and not self.post2_at_rest:
            self.held_post2_fft = self._compute_held_spectrum(
                self.post2_ring_buffer, self.held_post2_fft, self._last_post2_silent
            )
            if self._last_post2_silent and self.held_post2_fft is not None and np.max(self.held_post2_fft) <= floor_mag:
                self.post2_at_rest = True

    def get_smooth_curve(
        self,
        which: str = "post",
        is_x_log: bool = True,
        is_y_log: bool = True,
        db_min: float = -90.0,
        db_max: float = 0.0,
    ):
        """
        Devuelve (x_smooth, y_smooth) usando grillas precalculadas y omitiendo `make_interp_spline`
        cuando el espectro se encuentra en reposo en el piso (`y_floor`).
        Soporta which="pre" / "in", which="post" / "out1", which="post2" / "out2".
        """
        w = which.lower()
        if w in ("post2", "out2", "output2", "b"):
            held = self.held_post2_fft
            at_rest = self.post2_at_rest
        elif w in ("pre", "in", "input"):
            held = self.held_pre_fft
            at_rest = self.pre_at_rest
        else:
            held = self.held_post_fft
            at_rest = self.post_at_rest

        if held is None:
            return np.array([]), np.array([])

        y_floor = float(db_min) if is_y_log else 0.0
        y_ceil = float(db_max) if is_y_log else 1.0

        if len(self.f_real) > 4:
            x_smooth = self.x_smooth_log if is_x_log else self.x_smooth_lin
            if at_rest:
                return x_smooth, np.full_like(x_smooth, y_floor)

            eps = 1e-10
            mag_real = held[self.valid_mask]
            if is_y_log:
                y_real = np.clip(20.0 * np.log10(mag_real + eps), y_floor, y_ceil)
            else:
                y_real = np.clip(mag_real, y_floor, y_ceil)

            first_y = y_real[0]
            anchor_y = np.array([y_floor, y_floor + 0.35 * (first_y - y_floor)])
            y_full = np.concatenate([anchor_y, y_real])

            if is_x_log:
                y_smooth = np.interp(self.log_f_smooth, self.log_f_full, y_full)
            else:
                y_smooth = np.interp(x_smooth, self.f_full_lin, y_full)

            return x_smooth, np.clip(y_smooth, y_floor, y_ceil)

        return self.f_real, np.full_like(self.f_real, y_floor)
