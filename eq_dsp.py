import math
import threading
from audio_sync import AdaptiveResamplingRingBuffer as _AdaptiveResamplingRingBuffer
import numpy as np
from scipy.signal import sosfilt, butter, sosfreqz, find_peaks

try:
    from scipy.signal._signaltools import _sosfilt as _c_sosfilt
except ImportError:
    try:
        from scipy.signal._sosfilt import _sosfilt as _c_sosfilt
    except ImportError:
        _c_sosfilt = None


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
    max_f = 0.495 * sr
    f0 = 10.0 if freq < 10.0 else (max_f if freq > max_f else float(freq))
    q_val = 0.1 if q < 0.1 else (24.0 if q > 24.0 else float(q))
    g_db = -40.0 if gain_db < -40.0 else (40.0 if gain_db > 40.0 else float(gain_db))

    w0 = 2.0 * math.pi * f0 / sr
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    alpha = sin_w0 / (2.0 * q_val)
    A = 10.0 ** (g_db * 0.025)

    if filter_type == "Bell":
        b0 = 1.0 + alpha * A
        b1 = -2.0 * cos_w0
        b2 = 1.0 - alpha * A
        inv_a0 = 1.0 / (1.0 + alpha / A)
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha / A
        return np.array([[b0 * inv_a0, b1 * inv_a0, b2 * inv_a0, 1.0, a1 * inv_a0, a2 * inv_a0]], dtype=np.float64)

    elif filter_type == "Low Shelf":
        two_sqrt_A_alpha = 2.0 * math.sqrt(A) * alpha
        b0 = A * ((A + 1.0) - (A - 1.0) * cos_w0 + two_sqrt_A_alpha)
        b1 = 2.0 * A * ((A - 1.0) - (A + 1.0) * cos_w0)
        b2 = A * ((A + 1.0) - (A - 1.0) * cos_w0 - two_sqrt_A_alpha)
        inv_a0 = 1.0 / ((A + 1.0) + (A - 1.0) * cos_w0 + two_sqrt_A_alpha)
        a1 = -2.0 * ((A - 1.0) + (A + 1.0) * cos_w0)
        a2 = (A + 1.0) + (A - 1.0) * cos_w0 - two_sqrt_A_alpha
        return np.array([[b0 * inv_a0, b1 * inv_a0, b2 * inv_a0, 1.0, a1 * inv_a0, a2 * inv_a0]], dtype=np.float64)

    elif filter_type == "High Shelf":
        two_sqrt_A_alpha = 2.0 * math.sqrt(A) * alpha
        b0 = A * ((A + 1.0) + (A - 1.0) * cos_w0 + two_sqrt_A_alpha)
        b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cos_w0)
        b2 = A * ((A + 1.0) + (A - 1.0) * cos_w0 - two_sqrt_A_alpha)
        inv_a0 = 1.0 / ((A + 1.0) - (A - 1.0) * cos_w0 + two_sqrt_A_alpha)
        a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cos_w0)
        a2 = (A + 1.0) - (A - 1.0) * cos_w0 - two_sqrt_A_alpha
        return np.array([[b0 * inv_a0, b1 * inv_a0, b2 * inv_a0, 1.0, a1 * inv_a0, a2 * inv_a0]], dtype=np.float64)

    elif filter_type in ("Low Cut", "High Cut"):
        n_sections = max(1, min(4, int(slope) // 12))
        order = 2 * n_sections
        sections = []
        q_user_scale = q_val / 0.70710678
        for k in range(n_sections):
            theta = math.pi * (2 * k + 1) / (2.0 * order)
            q_butter = 1.0 / (2.0 * math.cos(theta))
            q_raw = q_butter * (q_user_scale if k == n_sections - 1 else 1.0)
            q_sec = 0.1 if q_raw < 0.1 else (24.0 if q_raw > 24.0 else q_raw)
            alpha_k = sin_w0 / (2.0 * q_sec)
            if filter_type == "Low Cut":
                b0 = (1.0 + cos_w0) * 0.5
                b1 = -(1.0 + cos_w0)
                b2 = (1.0 + cos_w0) * 0.5
            else:
                b0 = (1.0 - cos_w0) * 0.5
                b1 = 1.0 - cos_w0
                b2 = (1.0 - cos_w0) * 0.5
            inv_a0 = 1.0 / (1.0 + alpha_k)
            a1 = -2.0 * cos_w0
            a2 = 1.0 - alpha_k
            sections.append([b0 * inv_a0, b1 * inv_a0, b2 * inv_a0, 1.0, a1 * inv_a0, a2 * inv_a0])
        return np.array(sections, dtype=np.float64)

    elif filter_type == "Notch":
        b0 = 1.0
        b1 = -2.0 * cos_w0
        b2 = 1.0
        inv_a0 = 1.0 / (1.0 + alpha)
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha
        return np.array([[b0 * inv_a0, b1 * inv_a0, b2 * inv_a0, 1.0, a1 * inv_a0, a2 * inv_a0]], dtype=np.float64)

    elif filter_type == "Band Pass":
        b0 = alpha
        b1 = 0.0
        b2 = -alpha
        inv_a0 = 1.0 / (1.0 + alpha)
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha
        return np.array([[b0 * inv_a0, b1 * inv_a0, b2 * inv_a0, 1.0, a1 * inv_a0, a2 * inv_a0]], dtype=np.float64)

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


def _apply_sos_channel_first_inplace(
    sos: np.ndarray,
    buf_2xN: np.ndarray,
    zi: np.ndarray,
    channel_mode: str,
    zi_mid: np.ndarray | None = None,
    zi_side: np.ndarray | None = None,
    scratch_2xN: np.ndarray | None = None,
) -> None:
    """
    Aplica filtrado SOS in-place directamente sobre un buffer C-contiguous channel-first
    `(2, N)` en float64 con cero transposición y cero asignaciones en estado estable.
    """
    n_sec = sos.shape[0]
    frames = buf_2xN.shape[1]

    if _c_sosfilt is None:
        blk = np.ascontiguousarray(buf_2xN.T, dtype=np.float64)
        _apply_sos_channel_mode(sos, blk, zi, channel_mode, zi_mid=zi_mid, zi_side=zi_side)
        buf_2xN[0] = blk[:, 0]
        buf_2xN[1] = blk[:, 1]
        return

    if channel_mode == "LR":
        if zi.ndim == 3 and zi.shape[0] >= 2 and zi.shape[1] == n_sec:
            _c_sosfilt(sos, buf_2xN, zi[0:2])
        else:
            z_tmp = np.zeros((2, n_sec, 2), dtype=np.float64)
            _c_sosfilt(sos, buf_2xN, z_tmp)
        return

    if channel_mode == "L":
        if zi.ndim == 3 and zi.shape[0] >= 1 and zi.shape[1] == n_sec:
            _c_sosfilt(sos, buf_2xN[0:1], zi[0:1])
        else:
            z_tmp = np.zeros((1, n_sec, 2), dtype=np.float64)
            _c_sosfilt(sos, buf_2xN[0:1], z_tmp)
        return

    if channel_mode == "R":
        if zi.ndim == 3 and zi.shape[0] >= 2 and zi.shape[1] == n_sec:
            _c_sosfilt(sos, buf_2xN[1:2], zi[1:2])
        else:
            z_tmp = np.zeros((1, n_sec, 2), dtype=np.float64)
            _c_sosfilt(sos, buf_2xN[1:2], z_tmp)
        return

    if scratch_2xN is None or scratch_2xN.shape != (2, frames):
        scratch_2xN = np.empty((2, frames), dtype=np.float64)

    ch_l = buf_2xN[0]
    ch_r = buf_2xN[1]
    mid_1d = scratch_2xN[0]
    side_1d = scratch_2xN[1]
    np.add(ch_l, ch_r, out=mid_1d)
    mid_1d *= 0.5
    np.subtract(ch_l, ch_r, out=side_1d)
    side_1d *= 0.5

    if channel_mode == "Mid":
        if zi_mid is not None and zi_mid.ndim == 3 and zi_mid.shape[0] == 1 and zi_mid.shape[1] == n_sec:
            z_arr = zi_mid
        elif zi.ndim == 3 and zi.shape[0] >= 1 and zi.shape[1] == n_sec and zi_mid is None:
            z_arr = zi[0:1]
        else:
            z_arr = np.zeros((1, n_sec, 2), dtype=np.float64)
        _c_sosfilt(sos, scratch_2xN[0:1], z_arr)
    else:  # "Side"
        if zi_side is not None and zi_side.ndim == 3 and zi_side.shape[0] == 1 and zi_side.shape[1] == n_sec:
            z_arr = zi_side
        elif zi.ndim == 3 and zi.shape[0] >= 2 and zi.shape[1] == n_sec and zi_side is None:
            z_arr = zi[1:2]
        else:
            z_arr = np.zeros((1, n_sec, 2), dtype=np.float64)
        _c_sosfilt(sos, scratch_2xN[1:2], z_arr)

    np.add(mid_1d, side_1d, out=ch_l)
    np.subtract(mid_1d, side_1d, out=ch_r)


def _apply_sos_channel_mode(
    sos: np.ndarray,
    block: np.ndarray,
    zi: np.ndarray,
    channel_mode: str,
    zi_mid: np.ndarray | None = None,
    zi_side: np.ndarray | None = None,
) -> None:
    """
    Aplica filtrado SOS in-place sobre un bloque estéreo `(N, 2)` según `channel_mode`.
    Utiliza el kernel directo en C `_c_sosfilt` para máxima velocidad sin asignaciones dinámicas.
    Soporta formato nativo C-contiguous channel-first `(channels, n_sec, 2)` con cero copias.
    """
    is_channel_first = (zi.ndim == 3 and zi.shape[0] == 2)

    if block.ndim == 1 or block.shape[1] < 2:
        col = block if block.ndim == 1 else block[:, 0]
        if _c_sosfilt is not None:
            c_arr = np.ascontiguousarray(col, dtype=np.float64)[np.newaxis, :]
            z_arr = zi[0:1] if is_channel_first else np.ascontiguousarray(zi[:, :, 0], dtype=np.float64)[np.newaxis, :, :]
            _c_sosfilt(sos, c_arr, z_arr)
            col[:] = c_arr[0]
            if not is_channel_first:
                zi[:, :, 0] = z_arr[0]
        else:
            z_in = zi[0] if is_channel_first else zi[:, :, 0]
            filt, z_out = sosfilt(sos, col, zi=z_in)
            col[:] = filt
            if is_channel_first:
                zi[0] = z_out
            else:
                zi[:, :, 0] = z_out
        return

    if channel_mode == "L":
        if _c_sosfilt is not None:
            c_arr = np.ascontiguousarray(block[:, 0], dtype=np.float64)[np.newaxis, :]
            z_arr = zi[0:1] if is_channel_first else np.ascontiguousarray(zi[:, :, 0], dtype=np.float64)[np.newaxis, :, :]
            _c_sosfilt(sos, c_arr, z_arr)
            block[:, 0] = c_arr[0]
            if not is_channel_first:
                zi[:, :, 0] = z_arr[0]
        else:
            z_in = zi[0] if is_channel_first else zi[:, :, 0]
            filt, z_out = sosfilt(sos, block[:, 0], zi=z_in)
            block[:, 0] = filt
            if is_channel_first:
                zi[0] = z_out
            else:
                zi[:, :, 0] = z_out
    elif channel_mode == "R":
        if _c_sosfilt is not None:
            c_arr = np.ascontiguousarray(block[:, 1], dtype=np.float64)[np.newaxis, :]
            z_arr = zi[1:2] if is_channel_first else np.ascontiguousarray(zi[:, :, 1], dtype=np.float64)[np.newaxis, :, :]
            _c_sosfilt(sos, c_arr, z_arr)
            block[:, 1] = c_arr[0]
            if not is_channel_first:
                zi[:, :, 1] = z_arr[0]
        else:
            z_in = zi[1] if is_channel_first else zi[:, :, 1]
            filt, z_out = sosfilt(sos, block[:, 1], zi=z_in)
            block[:, 1] = filt
            if is_channel_first:
                zi[1] = z_out
            else:
                zi[:, :, 1] = z_out
    elif channel_mode == "Mid":
        ch_l = block[:, 0]
        ch_r = block[:, 1]
        mid = 0.5 * (ch_l + ch_r)
        side = 0.5 * (ch_l - ch_r)
        if _c_sosfilt is not None:
            if zi_mid is not None and zi_mid.ndim == 3 and zi_mid.shape[0] == 1:
                z_arr = zi_mid
            elif is_channel_first and zi_mid is None:
                z_arr = zi[0:1]
            else:
                target_zi = zi_mid if zi_mid is not None else zi[:, :, 0]
                z_arr = np.ascontiguousarray(target_zi, dtype=np.float64)[np.newaxis, :, :]
            m_arr = np.ascontiguousarray(mid, dtype=np.float64)[np.newaxis, :]
            _c_sosfilt(sos, m_arr, z_arr)
            mid_f = m_arr[0]
            if not is_channel_first and zi_mid is None:
                zi[:, :, 0] = z_arr[0]
            elif zi_mid is not None and zi_mid.ndim == 2:
                zi_mid[:] = z_arr[0]
        else:
            z_in = zi_mid[0] if (zi_mid is not None and zi_mid.ndim == 3) else (zi_mid if zi_mid is not None else (zi[0] if is_channel_first else zi[:, :, 0]))
            mid_f, z_out = sosfilt(sos, mid, zi=z_in)
            if zi_mid is not None:
                if zi_mid.ndim == 3:
                    zi_mid[0] = z_out
                else:
                    zi_mid[:] = z_out
            elif is_channel_first:
                zi[0] = z_out
            else:
                zi[:, :, 0] = z_out
        block[:, 0] = mid_f + side
        block[:, 1] = mid_f - side
    elif channel_mode == "Side":
        ch_l = block[:, 0]
        ch_r = block[:, 1]
        mid = 0.5 * (ch_l + ch_r)
        side = 0.5 * (ch_l - ch_r)
        if _c_sosfilt is not None:
            if zi_side is not None and zi_side.ndim == 3 and zi_side.shape[0] == 1:
                z_arr = zi_side
            elif is_channel_first and zi_side is None:
                z_arr = zi[1:2]
            else:
                target_zi = zi_side if zi_side is not None else zi[:, :, 1]
                z_arr = np.ascontiguousarray(target_zi, dtype=np.float64)[np.newaxis, :, :]
            s_arr = np.ascontiguousarray(side, dtype=np.float64)[np.newaxis, :]
            _c_sosfilt(sos, s_arr, z_arr)
            side_f = s_arr[0]
            if not is_channel_first and zi_side is None:
                zi[:, :, 1] = z_arr[0]
            elif zi_side is not None and zi_side.ndim == 2:
                zi_side[:] = z_arr[0]
        else:
            z_in = zi_side[0] if (zi_side is not None and zi_side.ndim == 3) else (zi_side if zi_side is not None else (zi[1] if is_channel_first else zi[:, :, 1]))
            side_f, z_out = sosfilt(sos, side, zi=z_in)
            if zi_side is not None:
                if zi_side.ndim == 3:
                    zi_side[0] = z_out
                else:
                    zi_side[:] = z_out
            elif is_channel_first:
                zi[1] = z_out
            else:
                zi[:, :, 1] = z_out
        block[:, 0] = mid + side_f
        block[:, 1] = mid - side_f
    else:  # "LR"
        if _c_sosfilt is not None:
            x_c = np.ascontiguousarray(block[:, 0:2].T, dtype=np.float64)
            if is_channel_first:
                _c_sosfilt(sos, x_c, zi[0:2])
            else:
                zi_c = np.ascontiguousarray(np.moveaxis(zi[:, :, 0:2], -1, 0), dtype=np.float64)
                _c_sosfilt(sos, x_c, zi_c)
                zi[:, :, 0:2] = np.moveaxis(zi_c, 0, -1)
            block[:, 0:2] = x_c.T
        else:
            if is_channel_first:
                z_sc = np.ascontiguousarray(np.moveaxis(zi[0:2], 0, -1))
                filt, z_sc_out = sosfilt(sos, block[:, 0:2], axis=0, zi=z_sc)
                block[:, 0:2] = filt
                zi[0:2] = np.moveaxis(z_sc_out, -1, 0)
            else:
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
    Aplica el filtro Band-Pass de Isolate respetando `channel_mode` con kernel directo en C.
    Soporta formato nativo C-contiguous channel-first `(channels, n_sec, 2)` con cero copias.
    """
    out = np.zeros_like(indata_f64)
    is_channel_first = (zi.ndim == 3 and zi.shape[0] == 2)

    if indata_f64.ndim == 1 or indata_f64.shape[1] < 2:
        col = indata_f64 if indata_f64.ndim == 1 else indata_f64[:, 0]
        if _c_sosfilt is not None:
            c_arr = np.ascontiguousarray(col, dtype=np.float64)[np.newaxis, :]
            z_arr = zi[0:1] if is_channel_first else np.ascontiguousarray(zi[:, :, 0], dtype=np.float64)[np.newaxis, :, :]
            _c_sosfilt(sos, c_arr, z_arr)
            filt = c_arr[0]
            if not is_channel_first:
                zi[:, :, 0] = z_arr[0]
        else:
            z_in = zi[0] if is_channel_first else zi[:, :, 0]
            filt, z_out = sosfilt(sos, col, zi=z_in)
            if is_channel_first:
                zi[0] = z_out
            else:
                zi[:, :, 0] = z_out
        if out.ndim == 1:
            out[:] = filt
        else:
            out[:, 0] = filt
        return out

    if channel_mode == "L":
        if _c_sosfilt is not None:
            c_arr = np.ascontiguousarray(indata_f64[:, 0], dtype=np.float64)[np.newaxis, :]
            z_arr = zi[0:1] if is_channel_first else np.ascontiguousarray(zi[:, :, 0], dtype=np.float64)[np.newaxis, :, :]
            _c_sosfilt(sos, c_arr, z_arr)
            out[:, 0] = c_arr[0]
            if not is_channel_first:
                zi[:, :, 0] = z_arr[0]
        else:
            z_in = zi[0] if is_channel_first else zi[:, :, 0]
            filt, z_out = sosfilt(sos, indata_f64[:, 0], zi=z_in)
            out[:, 0] = filt
            if is_channel_first:
                zi[0] = z_out
            else:
                zi[:, :, 0] = z_out
        out[:, 1] = 0.0
    elif channel_mode == "R":
        out[:, 0] = 0.0
        if _c_sosfilt is not None:
            c_arr = np.ascontiguousarray(indata_f64[:, 1], dtype=np.float64)[np.newaxis, :]
            z_arr = zi[1:2] if is_channel_first else np.ascontiguousarray(zi[:, :, 1], dtype=np.float64)[np.newaxis, :, :]
            _c_sosfilt(sos, c_arr, z_arr)
            out[:, 1] = c_arr[0]
            if not is_channel_first:
                zi[:, :, 1] = z_arr[0]
        else:
            z_in = zi[1] if is_channel_first else zi[:, :, 1]
            filt, z_out = sosfilt(sos, indata_f64[:, 1], zi=z_in)
            out[:, 1] = filt
            if is_channel_first:
                zi[1] = z_out
            else:
                zi[:, :, 1] = z_out
    elif channel_mode == "Mid":
        mid = 0.5 * (indata_f64[:, 0] + indata_f64[:, 1])
        if _c_sosfilt is not None:
            if zi_mid is not None and zi_mid.ndim == 3 and zi_mid.shape[0] == 1:
                z_arr = zi_mid
            elif is_channel_first and zi_mid is None:
                z_arr = zi[0:1]
            else:
                target_zi = zi_mid if zi_mid is not None else zi[:, :, 0]
                z_arr = np.ascontiguousarray(target_zi, dtype=np.float64)[np.newaxis, :, :]
            m_arr = np.ascontiguousarray(mid, dtype=np.float64)[np.newaxis, :]
            _c_sosfilt(sos, m_arr, z_arr)
            mid_iso = m_arr[0]
            if not is_channel_first and zi_mid is None:
                zi[:, :, 0] = z_arr[0]
            elif zi_mid is not None and zi_mid.ndim == 2:
                zi_mid[:] = z_arr[0]
        else:
            z_in = zi_mid[0] if (zi_mid is not None and zi_mid.ndim == 3) else (zi_mid if zi_mid is not None else (zi[0] if is_channel_first else zi[:, :, 0]))
            mid_iso, z_out = sosfilt(sos, mid, zi=z_in)
            if zi_mid is not None:
                if zi_mid.ndim == 3:
                    zi_mid[0] = z_out
                else:
                    zi_mid[:] = z_out
            elif is_channel_first:
                zi[0] = z_out
            else:
                zi[:, :, 0] = z_out
        out[:, 0] = mid_iso
        out[:, 1] = mid_iso
    elif channel_mode == "Side":
        side = 0.5 * (indata_f64[:, 0] - indata_f64[:, 1])
        if _c_sosfilt is not None:
            if zi_side is not None and zi_side.ndim == 3 and zi_side.shape[0] == 1:
                z_arr = zi_side
            elif is_channel_first and zi_side is None:
                z_arr = zi[1:2]
            else:
                target_zi = zi_side if zi_side is not None else zi[:, :, 1]
                z_arr = np.ascontiguousarray(target_zi, dtype=np.float64)[np.newaxis, :, :]
            s_arr = np.ascontiguousarray(side, dtype=np.float64)[np.newaxis, :]
            _c_sosfilt(sos, s_arr, z_arr)
            side_iso = s_arr[0]
            if not is_channel_first and zi_side is None:
                zi[:, :, 1] = z_arr[0]
            elif zi_side is not None and zi_side.ndim == 2:
                zi_side[:] = z_arr[0]
        else:
            z_in = zi_side[0] if (zi_side is not None and zi_side.ndim == 3) else (zi_side if zi_side is not None else (zi[1] if is_channel_first else zi[:, :, 1]))
            side_iso, z_out = sosfilt(sos, side, zi=z_in)
            if zi_side is not None:
                if zi_side.ndim == 3:
                    zi_side[0] = z_out
                else:
                    zi_side[:] = z_out
            elif is_channel_first:
                zi[1] = z_out
            else:
                zi[:, :, 1] = z_out
        out[:, 0] = side_iso
        out[:, 1] = -side_iso
    else:  # "LR"
        if _c_sosfilt is not None:
            x_c = np.ascontiguousarray(indata_f64[:, 0:2].T, dtype=np.float64)
            if is_channel_first:
                _c_sosfilt(sos, x_c, zi[0:2])
            else:
                zi_c = np.ascontiguousarray(np.moveaxis(zi[:, :, 0:2], -1, 0), dtype=np.float64)
                _c_sosfilt(sos, x_c, zi_c)
                zi[:, :, 0:2] = np.moveaxis(zi_c, 0, -1)
            out[:, 0:2] = x_c.T
        else:
            if is_channel_first:
                z_sc = np.ascontiguousarray(np.moveaxis(zi[0:2], 0, -1))
                filt, z_sc_out = sosfilt(sos, indata_f64[:, 0:2], axis=0, zi=z_sc)
                out[:, 0:2] = filt
                zi[0:2] = np.moveaxis(z_sc_out, -1, 0)
            else:
                out[:, 0:2], zi[:, :, 0:2] = sosfilt(sos, indata_f64[:, 0:2], axis=0, zi=zi[:, :, 0:2])
    return out


class EQBand:
    """
    Representa una banda paramétrica y dinámica individual del motor Heaviside DSP.
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

        # Estado dinámico en tiempo real (telemetría lock-free compartida con GUI 60 Hz)
        self.env_db = -90.0
        self.current_dyn_offset_db = 0.0
        self.live_env_db = -90.0
        self.live_dyn_delta_db = 0.0

        self.sos_static = design_biquad_sos(
            self.filter_type, self.freq, self.gain_db, self.q, self.sr, self.slope
        )
        self.sos_inst = self.sos_static.copy()
        self.live_sos = self.sos_static
        self.sc_sos = design_biquad_sos("Band Pass", self.freq, 0.0, self.q, self.sr)
        self.iso_sos = design_isolate_bandpass_sos(self.freq, self.q, self.sr, cascade_sections=2)

        # Estados de filtro independientes para Bus A (1-2) y Bus B (3-4) en formato channel-first (C-contiguous)
        n_sec = self.sos_static.shape[0]
        self.zi_A = np.zeros((self.channels, n_sec, 2), dtype=np.float64)
        self.zi_B = np.zeros((self.channels, n_sec, 2), dtype=np.float64)
        self.zi_mid_A = np.zeros((1, n_sec, 2), dtype=np.float64)
        self.zi_side_A = np.zeros((1, n_sec, 2), dtype=np.float64)
        self.zi_mid_B = np.zeros((1, n_sec, 2), dtype=np.float64)
        self.zi_side_B = np.zeros((1, n_sec, 2), dtype=np.float64)

        iso_sec = self.iso_sos.shape[0]
        self.iso_zi_A = np.zeros((self.channels, iso_sec, 2), dtype=np.float64)
        self.iso_zi_B = np.zeros((self.channels, iso_sec, 2), dtype=np.float64)
        self.iso_zi_mid_A = np.zeros((1, iso_sec, 2), dtype=np.float64)
        self.iso_zi_side_A = np.zeros((1, iso_sec, 2), dtype=np.float64)
        self.iso_zi_mid_B = np.zeros((1, iso_sec, 2), dtype=np.float64)
        self.iso_zi_side_B = np.zeros((1, iso_sec, 2), dtype=np.float64)

        self.sc_zi = np.zeros((1, 1, 2), dtype=np.float64)

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
    def gain(self) -> float:
        return self.gain_db

    @gain.setter
    def gain(self, val: float):
        self.gain_db = float(val)

    @property
    def output_bus(self) -> str:
        return self.target_output

    @output_bus.setter
    def output_bus(self, val: str):
        self.target_output = str(val)

    @property
    def dyn_threshold(self) -> float:
        return self.threshold_db

    @dyn_threshold.setter
    def dyn_threshold(self, val: float):
        self.threshold_db = float(val)

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
        req_shape = (self.channels, new_sos.shape[0], 2)
        mid_shape = (1, new_sos.shape[0], 2)
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
            self.live_dyn_delta_db = 0.0
        else:
            self._update_instant_sos()
        self.live_sos = self.sos_inst

        new_sc_sos = design_biquad_sos("Band Pass", self.freq, 0.0, self.q, self.sr)
        if self.sc_zi.shape != (1, 1, 2):
            self.sc_zi = np.zeros((1, 1, 2), dtype=np.float64)
        self.sc_sos = new_sc_sos

        new_iso_sos = design_isolate_bandpass_sos(self.freq, self.q, self.sr, cascade_sections=2)
        iso_shape = (self.channels, new_iso_sos.shape[0], 2)
        iso_mid_shape = (1, new_iso_sos.shape[0], 2)
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
        self.iso_sos = new_iso_sos

        self._cache_valid = False

    def smoothed_sos(self, k: float = 0.3) -> np.ndarray:
        """Slews the coefficients actually run by the filter toward `sos_inst` (once per block).
        Linear interpolation of biquad (a1, a2) between two stable filters stays stable
        (the stability region is convex), and removes the zipper/click noise of hard
        coefficient jumps while a node is being dragged."""
        target = self.sos_inst
        cur = getattr(self, "_sos_run", None)
        if cur is target:
            return cur
        if cur is None or cur.shape != target.shape:
            self._sos_run = target.copy()
            return self._sos_run
        diff = target - cur
        if float(np.max(np.abs(diff))) < 1e-7:
            self._sos_run = target
            return self._sos_run
        self._sos_run = cur + k * diff
        return self._sos_run

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
        self._sos_run = None
        self.env_db = -90.0
        self.current_dyn_offset_db = 0.0
        self.live_env_db = -90.0
        self.live_dyn_delta_db = 0.0
        self.live_sos = self.sos_static

    def get_static_freq_response_db(self, z1: np.ndarray, z2: np.ndarray) -> np.ndarray:
        if not self._cache_valid or self._cached_static_db is None or len(self._cached_static_db) != len(z1):
            self._cached_static_db = eval_sos_mag_db_fast(self.sos_static, z1, z2)
            self._cached_max_dyn_db = None
            self._cache_valid = True
        return self._cached_static_db

    def get_dynamic_freq_response_db(self, z1: np.ndarray, z2: np.ndarray) -> np.ndarray:
        sos = self.live_sos
        return eval_sos_mag_db_fast(sos, z1, z2)

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

        if not block.flags.writeable:
            block = block.copy()

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

            g = self.current_gain_linear
            a_att = self.alpha_att
            a_rel = self.alpha_rel
            one_m_att = 1.0 - a_att
            one_m_rel = 1.0 - a_rel

            gr_list = []
            for tgt in target_sub.tolist():
                if tgt < g:
                    g = a_att * g + one_m_att * tgt
                else:
                    g = a_rel * g + one_m_rel * tgt
                gr_list.append(g)

            self.current_gain_linear = g
            min_g = min(gr_list)
            gr_sub = np.array(gr_list, dtype=np.float32)
            gr_linear = np.repeat(gr_sub, sub_sz)
            if rem > 0:
                gr_linear = np.concatenate((gr_linear, np.full(rem, g, dtype=np.float32)))
        else:
            dt_s = 1.0 / self.sr
            a_att_s = float(np.exp(-dt_s / max(1e-5, self.attack_ms / 1000.0)))
            a_rel_s = float(np.exp(-dt_s / max(1e-4, self.release_ms / 1000.0)))
            one_m_att_s = 1.0 - a_att_s
            one_m_rel_s = 1.0 - a_rel_s
            gr_list = []
            g = self.current_gain_linear
            for tgt in target_lin.tolist():
                if tgt < g:
                    g = a_att_s * g + one_m_att_s * tgt
                else:
                    g = a_rel_s * g + one_m_rel_s * tgt
                gr_list.append(g)
            self.current_gain_linear = g
            min_g = min(gr_list)
            gr_linear = np.array(gr_list, dtype=np.float32)

        if block.ndim > 1:
            np.multiply(block, gr_linear[:, None], out=block)
        else:
            np.multiply(block, gr_linear, out=block)
        # Garantía absoluta anti-clipping: jamás supera el ceiling configurado
        np.clip(block, -self.ceil_lin, self.ceil_lin, out=block)

        self.last_gr_linear = float(min_g)
        self.last_gr_db = float(20.0 * np.log10(max(min_g, 1e-6)))

        return block

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
        self._tls = threading.local()
        self.static_version = 1
        self.bands: list[EQBand] = []
        self.autogain_enabled = False
        self.target_autogain_a_db = 0.0
        self.target_autogain_b_db = 0.0
        self.current_autogain_a_db = 0.0
        self.current_autogain_b_db = 0.0
        self.target_autogain_db = 0.0
        self.current_autogain_db = 0.0
        self.init_default_bands()
        self.limiter_a = MasterLimiter(sr=self.sr)
        self.limiter_b = MasterLimiter(sr=self.sr)

    def mark_dirty(self):
        self.static_version += 1
        if getattr(self, "autogain_enabled", False):
            self.recalculate_autogain_target()

    def compute_combined_eq_db_at_freqs(self, freqs: np.ndarray) -> np.ndarray:
        with self.lock:
            w = 2.0 * np.pi * freqs / self.sr
            z1 = np.exp(-1j * w)
            z2 = np.exp(-2j * w)
            total_db = np.zeros(len(freqs), dtype=np.float64)
            for b in self.bands:
                if b.enabled:
                    total_db += eval_sos_mag_db_fast(b.sos_static, z1, z2)
            return total_db

    def compute_single_band_db_at_freqs(self, band: EQBand, freqs: np.ndarray) -> np.ndarray:
        w = 2.0 * np.pi * freqs / self.sr
        z1 = np.exp(-1j * w)
        z2 = np.exp(-2j * w)
        return eval_sos_mag_db_fast(band.sos_static, z1, z2)

    def recalculate_autogain_target(self):
        if not getattr(self, "autogain_enabled", False) or getattr(self, "bypass", False) or getattr(self, "bypass_eq", False):
            self.target_autogain_a_db = 0.0
            self.target_autogain_b_db = 0.0
            self.target_autogain_db = 0.0
            return

        freqs = np.geomspace(60.0, 12000.0, 32)
        db_curve_a = np.zeros_like(freqs, dtype=float)
        db_curve_b = np.zeros_like(freqs, dtype=float)

        with self.lock:
            w = 2.0 * np.pi * freqs / self.sr
            z1 = np.exp(-1j * w)
            z2 = np.exp(-2j * w)
            for b in self.bands:
                if not getattr(b, "enabled", True):
                    continue
                bus = str(getattr(b, "output_bus", getattr(b, "target_output", "A+B"))).upper()
                band_resp_db = eval_sos_mag_db_fast(b.sos_static, z1, z2)
                if "A" in bus:
                    db_curve_a += band_resp_db
                if "B" in bus:
                    db_curve_b += band_resp_db

        mean_a = float(np.mean(db_curve_a))
        mean_b = float(np.mean(db_curve_b))
        self.target_autogain_a_db = float(np.clip(-mean_a * 0.85, -12.0, 12.0))
        self.target_autogain_b_db = float(np.clip(-mean_b * 0.85, -12.0, 12.0))
        self.target_autogain_db = self.target_autogain_a_db

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
            if getattr(self, "autogain_enabled", False):
                self.recalculate_autogain_target()

    def set_sample_rate(self, sr: int):
        with self.lock:
            self.sr = float(sr)
            for b in self.bands:
                b.update_coefficients(sr=self.sr, channels=self.channels)
                b.reset_state()
            self.limiter_a.set_sample_rate(self.sr)
            self.limiter_b.set_sample_rate(self.sr)
            self.static_version += 1
            if getattr(self, "autogain_enabled", False):
                self.recalculate_autogain_target()

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

    def process_dual_block(
        self,
        indata: np.ndarray,
        apply_limiter: bool = True,
        apply_autogain: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Procesa `indata` generando dos bloques estéreo independientes `(out_A, out_B)`:
        - `out_A`: procesado con todas las bandas activas cuyo `target_output` sea "A" o "A+B" (usando `zi_A`).
        - `out_B`: procesado con todas las bandas activas cuyo `target_output` sea "B" o "A+B" (usando `zi_B`).
        Soporta Isolate (`isolated_band_index`) y procesamiento por canal (`LR`, `L`, `R`, `Mid`, `Side`).
        Lock-free en el hilo de audio con buffers channel-first prealocados y ruta rápida single-pass para A+B.
        """
        frames = indata.shape[0] if indata.ndim > 0 else 0
        if frames == 0:
            empty = np.zeros((0, 2), dtype=np.float32)
            return empty, empty.copy()

        bands = list(self.bands)
        bypass = self.bypass
        out_gain_db = self.output_gain_db
        autogain_en = getattr(self, "autogain_enabled", False)
        bypass_eq = getattr(self, "bypass_eq", False)

        if bypass:
            if indata.ndim == 1:
                stereo_in = np.column_stack((indata, indata)).astype(np.float32, copy=False)
            elif indata.shape[1] == 1:
                stereo_in = np.repeat(indata, 2, axis=1).astype(np.float32, copy=False)
            else:
                stereo_in = indata[:, 0:2].astype(np.float32, copy=True)
            return stereo_in, stereo_in.copy()

        iso_idx = self.isolated_band_index
        if iso_idx is not None:
            if not (0 <= iso_idx < len(bands) and bands[iso_idx].enabled):
                iso_idx = None
        if iso_idx is None:
            for i, b in enumerate(bands):
                if b.enabled and b.solo:
                    iso_idx = i
                    break

        # --- MODO ISOLATE (SOLO DE FRECUENCIA BAND-PASS) ---
        if iso_idx is not None:
            iso_band = bands[iso_idx]
            if indata.ndim == 1:
                in_f64 = np.column_stack((indata, indata)).astype(np.float64, copy=False)
            elif indata.shape[1] == 1:
                in_f64 = np.repeat(indata, 2, axis=1).astype(np.float64, copy=False)
            else:
                in_f64 = indata[:, 0:2].astype(np.float64, copy=True)

            band_boost = min(12.0, max(-12.0, iso_band.gain_db * 0.5)) if iso_band.supports_gain else 0.0
            mon_gain_db = float(np.clip(band_boost + out_gain_db, -24.0, 12.0))
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

            if apply_limiter:
                out_A = self.limiter_a.process(out_A)
                out_B = self.limiter_b.process(out_B)
            return out_A.astype(np.float32), out_B.astype(np.float32)

        # --- MODO NORMAL DUAL BUS (OUT A + OUT B) ---
        any_filtering = False
        shared_ab = True
        any_dyn = False
        for b in bands:
            if b.enabled:
                if b.target_output != "A+B":
                    shared_ab = False
                if b.is_dynamic_active:
                    any_dyn = True
                    any_filtering = True
                elif abs(b.gain_db) > 0.01 or not b.supports_gain:
                    any_filtering = True

        if not any_filtering and abs(out_gain_db) < 0.01 and not autogain_en and abs(self.current_autogain_db) < 0.01:
            for b in bands:
                if abs(b.current_dyn_offset_db) > 0.001:
                    b.current_dyn_offset_db *= 0.85
                else:
                    b.current_dyn_offset_db = 0.0
                b.live_dyn_delta_db = float(b.current_dyn_offset_db)
                b.live_env_db = float(b.env_db)
                b.live_sos = b.sos_static
            if indata.ndim == 1:
                out_A = np.column_stack((indata, indata)).astype(np.float32, copy=False)
            elif indata.shape[1] == 1:
                out_A = np.repeat(indata, 2, axis=1).astype(np.float32, copy=False)
            else:
                out_A = indata[:, 0:2].astype(np.float32, copy=True)
            out_B = out_A.copy()
            if apply_limiter:
                out_A = self.limiter_a.process(out_A)
                out_B = self.limiter_b.process(out_B)
            return out_A, out_B

        tls = self._tls
        if getattr(tls, "work_frames", -1) != frames:
            tls.work_frames = frames
            tls.work_a = np.empty((2, frames), dtype=np.float64)
            tls.work_b = np.empty((2, frames), dtype=np.float64)
            tls.ms_scratch = np.empty((2, frames), dtype=np.float64)
            tls.dyn_scratch = np.empty((2, frames), dtype=np.float64)
            tls.sc_work = np.empty((1, frames), dtype=np.float64)

        work_a = tls.work_a
        work_b = tls.work_b
        ms_scratch = tls.ms_scratch
        dyn_scratch = tls.dyn_scratch

        if indata.ndim == 1:
            work_a[0] = indata
            work_a[1] = indata
            in_ch0 = indata
            in_ch1 = indata
        elif indata.shape[1] == 1:
            col = indata[:, 0]
            work_a[0] = col
            work_a[1] = col
            in_ch0 = col
            in_ch1 = col
        else:
            in_ch0 = indata[:, 0]
            in_ch1 = indata[:, 1]
            work_a[0] = in_ch0
            work_a[1] = in_ch1

        if not shared_ab:
            np.copyto(work_b, work_a)

        block_duration = frames / self.sr
        if any_dyn:
            mono_in = dyn_scratch[0]
            side_in = dyn_scratch[1]
            np.add(work_a[0], work_a[1], out=mono_in)
            mono_in *= 0.5
            np.subtract(work_a[0], work_a[1], out=side_in)
            side_in *= 0.5

        for band in bands:
            if not band.enabled:
                band.current_dyn_offset_db = (
                    0.0 if abs(band.current_dyn_offset_db) < 0.01 else band.current_dyn_offset_db * 0.85
                )
                band.live_dyn_delta_db = float(band.current_dyn_offset_db)
                band.live_env_db = float(band.env_db)
                band.live_sos = band.sos_static
                continue

            if band.is_dynamic_active:
                sc_work = tls.sc_work
                if band.channel_mode == "L":
                    sc_work[0] = in_ch0
                elif band.channel_mode == "R":
                    sc_work[0] = in_ch1
                elif band.channel_mode == "Side":
                    np.copyto(sc_work[0], side_in)
                else:
                    np.copyto(sc_work[0], mono_in)

                if _c_sosfilt is not None:
                    sc_zi = band.sc_zi
                    if sc_zi.ndim == 3 and sc_zi.shape == (1, 1, 2):
                        _c_sosfilt(band.sc_sos, sc_work, sc_zi)
                    else:
                        sc_zi_arr = np.zeros((1, 1, 2), dtype=np.float64)
                        _c_sosfilt(band.sc_sos, sc_work, sc_zi_arr)
                    band_signal = sc_work[0]
                else:
                    band_signal, band.sc_zi = sosfilt(band.sc_sos, sc_work[0], zi=band.sc_zi)

                rms = math.sqrt(float(np.dot(band_signal, band_signal)) / frames)
                level_db = 20.0 * math.log10(rms + 1e-10)

                att_sec = max(band.attack_ms * 0.001, 1e-4)
                rel_sec = max(band.release_ms * 0.001, 1e-4)
                alpha_att = math.exp(-block_duration / att_sec)
                alpha_rel = math.exp(-block_duration / rel_sec)

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

                prev_offset = band.current_dyn_offset_db
                if eff_over > 0.0:
                    norm_excursion = 1.0 - math.exp(-eff_over / 12.0)
                    new_offset = float(band.dyn_range_db * norm_excursion)
                else:
                    new_offset = 0.0

                band.current_dyn_offset_db = new_offset
                band.live_env_db = float(band.env_db)
                band.live_dyn_delta_db = float(new_offset)
                if new_offset == 0.0 and prev_offset == 0.0 and band.sos_inst is band.sos_static:
                    pass
                elif abs(new_offset - prev_offset) > 1e-4 or band.sos_inst is band.sos_static:
                    if new_offset == 0.0:
                        band.sos_inst = band.sos_static
                    else:
                        band._update_instant_sos()
            else:
                band.current_dyn_offset_db = 0.0
                band.live_dyn_delta_db = 0.0
                band.live_env_db = -90.0
                band.sos_inst = band.sos_static
                band.live_sos = band.sos_static

            if band.is_dynamic_active or abs(band.gain_db) > 0.01 or not band.supports_gain:
                sos_run = band.smoothed_sos()
                band.live_sos = sos_run
                if shared_ab:
                    _apply_sos_channel_first_inplace(
                        sos_run, work_a, band.zi_A, band.channel_mode,
                        zi_mid=band.zi_mid_A, zi_side=band.zi_side_A,
                        scratch_2xN=ms_scratch,
                    )
                    if band.zi_B.shape == band.zi_A.shape:
                        np.copyto(band.zi_B, band.zi_A)
                    if band.zi_mid_B.shape == band.zi_mid_A.shape:
                        np.copyto(band.zi_mid_B, band.zi_mid_A)
                    if band.zi_side_B.shape == band.zi_side_A.shape:
                        np.copyto(band.zi_side_B, band.zi_side_A)
                else:
                    if band.target_output in ("A", "A+B"):
                        _apply_sos_channel_first_inplace(
                            sos_run, work_a, band.zi_A, band.channel_mode,
                            zi_mid=band.zi_mid_A, zi_side=band.zi_side_A,
                            scratch_2xN=ms_scratch,
                        )
                    if band.target_output in ("B", "A+B"):
                        _apply_sos_channel_first_inplace(
                            sos_run, work_b, band.zi_B, band.channel_mode,
                            zi_mid=band.zi_mid_B, zi_side=band.zi_side_B,
                            scratch_2xN=ms_scratch,
                        )

        if shared_ab:
            np.copyto(work_b, work_a)

        if abs(out_gain_db) >= 0.01:
            gain_lin = 10.0 ** (out_gain_db / 20.0)
            work_a *= gain_lin
            work_b *= gain_lin

        # --- AUTO-GAIN (Compensación de Loudness Post-EQ / Pre-Limitador) ---
        if apply_autogain:
            if autogain_en and not bypass_eq:
                peak_a = float(np.max(np.abs(work_a))) if work_a.size else 0.0
                peak_b = float(np.max(np.abs(work_b))) if work_b.size else 0.0
                target_a = self.target_autogain_a_db if peak_a > 0.001 else 0.0
                target_b = self.target_autogain_b_db if peak_b > 0.001 else 0.0
            else:
                target_a = 0.0
                target_b = 0.0

            self.current_autogain_a_db += 0.12 * (target_a - self.current_autogain_a_db)
            self.current_autogain_b_db += 0.12 * (target_b - self.current_autogain_b_db)
            self.current_autogain_db = self.current_autogain_a_db

            if abs(self.current_autogain_a_db) > 0.02:
                work_a *= (10.0 ** (self.current_autogain_a_db / 20.0))
            if abs(self.current_autogain_b_db) > 0.02:
                work_b *= (10.0 ** (self.current_autogain_b_db / 20.0))

        out_A = np.empty((frames, 2), dtype=np.float32)
        out_A[:, 0] = work_a[0]
        out_A[:, 1] = work_a[1]
        out_B = np.empty((frames, 2), dtype=np.float32)
        out_B[:, 0] = work_b[0]
        out_B[:, 1] = work_b[1]

        if apply_limiter:
            out_A = self.limiter_a.process(out_A)
            out_B = self.limiter_b.process(out_B)
        return out_A, out_B

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
        n = len(z1)
        total_static = np.zeros(n, dtype=np.float64)
        total_dyn = np.zeros(n, dtype=np.float64)

        if self.bypass:
            return total_static, total_dyn

        bands = list(self.bands)
        iso_idx = self.isolated_band_index
        if iso_idx is not None:
            if not (0 <= iso_idx < len(bands) and bands[iso_idx].enabled):
                iso_idx = None
        if iso_idx is None:
            for i, b in enumerate(bands):
                if b.enabled and b.solo:
                    iso_idx = i
                    break

        if iso_idx is not None:
            iso_b = bands[iso_idx]
            if target_bus == "A" and iso_b.target_output not in ("A", "A+B"):
                return np.full(n, -120.0, dtype=np.float64), np.full(n, -120.0, dtype=np.float64)
            if target_bus == "B" and iso_b.target_output not in ("B", "A+B"):
                return np.full(n, -120.0, dtype=np.float64), np.full(n, -120.0, dtype=np.float64)
            iso_resp = iso_b.get_isolate_freq_response_db(z1, z2) + self.output_gain_db
            return iso_resp, iso_resp.copy()

        if target_bus == "A":
            active_bands = [b for b in bands if b.enabled and b.target_output in ("A", "A+B")]
        elif target_bus == "B":
            active_bands = [b for b in bands if b.enabled and b.target_output in ("B", "A+B")]
        else:
            active_bands = [b for b in bands if b.enabled]

        for band in active_bands:
            static_db = band.get_static_freq_response_db(z1, z2)
            total_static += static_db
            if band.is_dynamic_active and abs(band.live_dyn_delta_db) > 0.02:
                dyn_db = band.get_dynamic_freq_response_db(z1, z2)
                total_dyn += dyn_db
            else:
                total_dyn += static_db

        out_gain = self.output_gain_db
        if abs(out_gain) >= 0.01:
            total_static += out_gain
            total_dyn += out_gain

        return total_static, total_dyn


class DualOutputCrossover:
    """
    Crossover Digital de Salidas para dividir frecuencias entre Out A y Out B
    utilizando filtros Linkwitz-Riley o Butterworth de orden 2, 4 u 8 (12, 24, 48 dB/oct).
    """

    def __init__(self, sample_rate=48000):
        self.sample_rate = float(sample_rate)
        self.enabled = False
        self.freq_hz = 80.0
        self.filter_type = "Linkwitz-Riley"  # "Linkwitz-Riley" o "Butterworth"
        self.slope_db = 24                   # 12, 24, 48 dB/oct
        self.routing_mode = "A_Tops_B_Sub"   # "A_Tops_B_Sub", "A_Sub_B_Tops", "A_Full_B_Sub"
        self._sos_lp = None
        self._sos_hp = None
        self._zi_a = None
        self._zi_b = None
        self.update_filters()

    def get_state_dict(self) -> dict:
        return {
            "enabled": bool(self.enabled),
            "freq_hz": float(self.freq_hz),
            "filter_type": str(self.filter_type),
            "slope_db": int(self.slope_db),
            "routing_mode": str(self.routing_mode),
        }

    def set_config(self, state_dict: dict, sample_rate=None):
        if sample_rate is not None:
            self.sample_rate = float(sample_rate)
        self.enabled = bool(state_dict.get("enabled", self.enabled))
        self.freq_hz = float(np.clip(state_dict.get("freq_hz", self.freq_hz), 20.0, 20000.0))
        self.filter_type = str(state_dict.get("filter_type", self.filter_type))
        self.slope_db = int(state_dict.get("slope_db", self.slope_db))
        self.routing_mode = str(state_dict.get("routing_mode", self.routing_mode))
        self.update_filters()

    def update_filters(self):
        nyq = 0.5 * self.sample_rate
        norm_f = float(np.clip(self.freq_hz / nyq, 0.0005, 0.98))
        order_total = max(2, int(self.slope_db // 6))

        if self.filter_type == "Linkwitz-Riley":
            # Linkwitz-Riley (LR2, LR4, LR8) = dos Butterworth de mitad de orden en cascada
            half_order = max(1, order_total // 2)
            bw_lp = butter(half_order, norm_f, btype='low', output='sos')
            bw_hp = butter(half_order, norm_f, btype='high', output='sos')
            new_sos_lp = np.vstack([bw_lp, bw_lp])
            new_sos_hp = np.vstack([bw_hp, bw_hp])
        else:
            new_sos_lp = butter(order_total, norm_f, btype='low', output='sos')
            new_sos_hp = butter(order_total, norm_f, btype='high', output='sos')

        # Si el orden de secciones no cambia, conservar el estado zi para evitar chasquidos de fase
        new_n_sec = new_sos_lp.shape[0]
        if self._zi_a is None or self._zi_a.shape != (2, new_n_sec, 2):
            self._zi_a = np.zeros((2, new_n_sec, 2), dtype=np.float64)
            self._zi_b = np.zeros((2, new_n_sec, 2), dtype=np.float64)

        self._sos_lp = new_sos_lp
        self._sos_hp = new_sos_hp

    def _fast_filter_block(self, sos: np.ndarray, blk: np.ndarray, zi: np.ndarray) -> np.ndarray:
        frames = blk.shape[0] if blk.ndim > 0 else 0
        if frames == 0:
            return blk
        if _c_sosfilt is not None:
            n_ch = blk.shape[1] if blk.ndim > 1 else 1
            if blk.ndim == 1 or n_ch == 1:
                col = np.ascontiguousarray(blk.flatten(), dtype=np.float64)[np.newaxis, :]
                _c_sosfilt(sos, col, zi[0:1])
                return col[0].astype(np.float32, copy=False)
            else:
                if not hasattr(self, "_buf_f64") or self._buf_f64.shape != (2, frames):
                    self._buf_f64 = np.zeros((2, frames), dtype=np.float64)
                    self._buf_f32 = np.zeros((frames, 2), dtype=np.float32)

                self._buf_f64[0, :] = blk[:, 0]
                self._buf_f64[1, :] = blk[:, 1]
                _c_sosfilt(sos, self._buf_f64, zi)
                self._buf_f32[:, 0] = self._buf_f64[0]
                self._buf_f32[:, 1] = self._buf_f64[1]
                return self._buf_f32.copy()
        else:
            zi_sc = np.moveaxis(zi, 0, -1)
            proc, zi_out = sosfilt(sos, blk, axis=0, zi=zi_sc)
            zi[:] = np.moveaxis(zi_out, -1, 0)
            return proc.astype(np.float32, copy=False)

    def process_outputs(self, out_a: np.ndarray, out_b: np.ndarray):
        if not self.enabled or self._sos_lp is None or self._sos_hp is None:
            return out_a, out_b

        sos_lp = self._sos_lp
        sos_hp = self._sos_hp
        n_sec = sos_lp.shape[0]

        if self._zi_a is None or self._zi_a.shape != (2, n_sec, 2):
            self._zi_a = np.zeros((2, n_sec, 2), dtype=np.float64)
            self._zi_b = np.zeros((2, n_sec, 2), dtype=np.float64)

        # Detección ultra-rápida de silencio para eludir sosfilt en reposo y evitar gasto de CPU (0.001 ms)
        is_silent_a = float(np.max(np.abs(out_a))) < 1e-7
        is_silent_b = float(np.max(np.abs(out_b))) < 1e-7

        if self.routing_mode == "A_Tops_B_Sub":
            if is_silent_a:
                proc_a = out_a
                self._zi_a.fill(0.0)
            else:
                proc_a = self._fast_filter_block(sos_hp, out_a, self._zi_a)

            if is_silent_b:
                proc_b = out_b
                self._zi_b.fill(0.0)
            else:
                proc_b = self._fast_filter_block(sos_lp, out_b, self._zi_b)

        elif self.routing_mode == "A_Sub_B_Tops":
            if is_silent_a:
                proc_a = out_a
                self._zi_a.fill(0.0)
            else:
                proc_a = self._fast_filter_block(sos_lp, out_a, self._zi_a)

            if is_silent_b:
                proc_b = out_b
                self._zi_b.fill(0.0)
            else:
                proc_b = self._fast_filter_block(sos_hp, out_b, self._zi_b)

        else:  # "A_Full_B_Sub"
            proc_a = out_a
            if is_silent_b:
                proc_b = out_b
                self._zi_b.fill(0.0)
            else:
                proc_b = self._fast_filter_block(sos_lp, out_b, self._zi_b)

        return proc_a.astype(np.float32, copy=False), proc_b.astype(np.float32, copy=False)

    def get_magnitude_db(self, freqs_hz: np.ndarray):
        if not self.enabled or self._sos_lp is None or self._sos_hp is None:
            zeros = np.zeros_like(freqs_hz, dtype=float)
            return zeros, zeros
        _, h_lp = sosfreqz(self._sos_lp, worN=freqs_hz, fs=self.sample_rate)
        _, h_hp = sosfreqz(self._sos_hp, worN=freqs_hz, fs=self.sample_rate)
        db_lp = 20.0 * np.log10(np.maximum(np.abs(h_lp), 1e-6))
        db_hp = 20.0 * np.log10(np.maximum(np.abs(h_hp), 1e-6))
        if self.routing_mode == "A_Tops_B_Sub":
            return db_hp, db_lp
        elif self.routing_mode == "A_Sub_B_Tops":
            return db_lp, db_hp
        else:
            return np.zeros_like(freqs_hz, dtype=float), db_lp


class AsyncInputRingBuffer(_AdaptiveResamplingRingBuffer):
    """
    Ring buffer para Input 2 (Aux / Mic, otro reloj de hardware) con remuestreo adaptativo
    continuo (ASRC): la deriva de reloj se absorbe con correcciones de ppm inaudibles en lugar
    de descartar bloques (saltos) o insertar silencios. Ver audio_sync.py.
    """

    def __init__(self, capacity_frames: int = 32768, channels: int = 2):
        super().__init__(capacity_frames=capacity_frames, channels=channels, target_fill=256)


def smooth_spectrum_12th_octave(freqs_hz: np.ndarray, mag_db: np.ndarray) -> np.ndarray:
    """
    Aplica suavizado de 1/12 de octava sobre un espectro de frecuencias en dB
    mediante interpolación logarítmica y filtro Gaussiano sin alterar los extremos.
    """
    valid = (freqs_hz >= 20.0) & np.isfinite(mag_db)
    if not np.any(valid):
        return mag_db.copy()

    f_valid = freqs_hz[valid]
    m_valid = mag_db[valid]
    log_f = np.log2(f_valid)

    n_grid = 1000
    log_min, log_max = log_f[0], log_f[-1]
    if log_max <= log_min:
        return mag_db.copy()

    grid_log = np.linspace(log_min, log_max, n_grid)
    grid_mag = np.interp(grid_log, log_f, m_valid)

    octave_step = (log_max - log_min) / float(n_grid - 1)
    pts_per_oct = 1.0 / (12.0 * octave_step)
    sigma = max(1.0, pts_per_oct / 2.355)

    try:
        from scipy.ndimage import gaussian_filter1d
        smoothed_grid = gaussian_filter1d(grid_mag, sigma=sigma, mode="nearest")
    except Exception:
        # Fallback si scipy.ndimage no estuviera disponible
        k_w = int(max(3, round(pts_per_oct)))
        if k_w % 2 == 0:
            k_w += 1
        kernel = np.hanning(k_w)
        kernel /= np.sum(kernel)
        smoothed_grid = np.convolve(grid_mag, kernel, mode="same")

    smoothed = mag_db.copy()
    smoothed[valid] = np.interp(log_f, grid_log, smoothed_grid)
    return smoothed


def detect_resonances(
    freqs_hz: np.ndarray,
    smoothed_mag_db: np.ndarray,
    n_peaks: int = 3,
    min_freq: float = 25.0,
    max_freq: float = 16000.0,
    prominence: float = 3.0,
) -> list[tuple[float, float, float]]:
    """
    Detecta los picos de resonancia más notorios usando scipy.signal.find_peaks
    en el rango acotado (por defecto 25 Hz - 16 kHz).
    Retorna lista de tuplas (frecuencia_hz, ganancia_db, prominencia_db).
    """
    mask = (freqs_hz >= min_freq) & (freqs_hz <= max_freq) & np.isfinite(smoothed_mag_db)
    if not np.any(mask):
        return []

    f_sub = freqs_hz[mask]
    m_sub = smoothed_mag_db[mask]

    peaks, props = find_peaks(m_sub, prominence=prominence)
    if len(peaks) == 0:
        return []

    proms = props.get("prominences", np.ones(len(peaks)))
    order = np.argsort(proms)[::-1][:n_peaks]

    results = []
    for p in order:
        idx = peaks[p]
        results.append((float(f_sub[idx]), float(m_sub[idx]), float(proms[p])))
    return results


MB_MODES = ["Compress", "Expand", "Limit"]
MB_CHARACTER_MODES = ("Off", "Tube", "Diode", "Bright")
MB_MODE_COLORS = {
    "Compress": "#00E5FF",
    "Expand": "#00E676",
    "Limit": "#FF6D00",
}
MAX_MB_BANDS = 6


def eval_sos_complex_fast(sos: np.ndarray, z1: np.ndarray, z2: np.ndarray) -> np.ndarray:
    """Evalúa la respuesta compleja H(z) de una matriz SOS usando fasores z^-1 y z^-2."""
    b0 = sos[:, 0:1]
    b1 = sos[:, 1:2]
    b2 = sos[:, 2:3]
    a1 = sos[:, 4:5]
    a2 = sos[:, 5:6]
    h_sec = (b0 + b1 * z1 + b2 * z2) / (1.0 + a1 * z1 + a2 * z2)
    return np.prod(h_sec, axis=0) if sos.shape[0] > 1 else h_sec[0]


def design_lr4_and_ap2(fc_hz: float, sr: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Diseña el par complementario Linkwitz-Riley de 4to orden (LR4: 24 dB/oct) junto
    con su filtro All-Pass de 2do orden emparejado en fase (AP2):
        H_LP4(z) + H_HP4(z) == H_AP2(z)  con error < 1e-11 dB.
    """
    sr_f = float(max(sr, 8000.0))
    fc = float(np.clip(fc_hz, 10.0, 0.48 * sr_f))
    K = math.tan(math.pi * fc / sr_f)
    K2 = K * K
    sqrt2_K = math.sqrt(2.0) * K
    inv_d0 = 1.0 / (1.0 + sqrt2_K + K2)
    a1 = 2.0 * (K2 - 1.0) * inv_d0
    a2 = (1.0 - sqrt2_K + K2) * inv_d0

    lp_row = [K2 * inv_d0, 2.0 * K2 * inv_d0, K2 * inv_d0, 1.0, a1, a2]
    hp_row = [inv_d0, -2.0 * inv_d0, inv_d0, 1.0, a1, a2]
    ap_row = [a2, a1, 1.0, 1.0, a1, a2]

    sos_lp4 = np.array([lp_row, lp_row], dtype=np.float64)
    sos_hp4 = np.array([hp_row, hp_row], dtype=np.float64)
    sos_ap2 = np.array([ap_row], dtype=np.float64)
    return sos_lp4, sos_hp4, sos_ap2


class DynamicsBand:
    """
    Banda individual de Dinámica Multibanda (Compress, Expand o Limit por banda de frecuencia).
    Expone telemetría lock-free `live_level_db` y `live_gr_db` para la UI a 60 Hz.
    """

    def __init__(
        self,
        f_low: float = 100.0,
        f_high: float = 1000.0,
        enabled: bool = True,
        solo: bool = False,
        bypass: bool = False,
        bus_target: str = "A+B",
        mode: str = "Compress",
        threshold_db: float = -24.0,
        range_db: float = -12.0,
        ratio: float = 4.0,
        attack_ms: float = 10.0,
        release_ms: float = 120.0,
        knee_db: float = 6.0,
        output_gain_db: float = 0.0,
        color: str | None = None,
        character_mode: str = "Off",
        character_drive: float = 0.35,
        character_post: bool = True,
        custom_color: bool = False,
    ):
        self.f_low = float(np.clip(f_low, 10.0, 22000.0))
        self.f_high = float(np.clip(f_high, self.f_low * 1.12, 24000.0))
        self.enabled = bool(enabled)
        self.solo = bool(solo)
        self.bypass = bool(bypass)
        self.bus_target = bus_target if bus_target in TARGET_OUTPUTS else "A+B"
        self.mode = mode if mode in MB_MODES else "Compress"
        self.threshold_db = float(np.clip(threshold_db, -70.0, 0.0))
        self.range_db = float(np.clip(range_db, -30.0, 12.0))
        self.ratio = float(np.clip(ratio, 1.0, 20.0))
        self.attack_ms = float(np.clip(attack_ms, 0.1, 500.0))
        self.release_ms = float(np.clip(release_ms, 5.0, 2000.0))
        self.knee_db = float(np.clip(knee_db, 0.0, 24.0))
        self.output_gain_db = float(np.clip(output_gain_db, -24.0, 24.0))
        self.custom_color = bool(custom_color)
        self.color = str(color) if color else MB_MODE_COLORS.get(self.mode, "#00E5FF")
        self.character_mode = character_mode if character_mode in MB_CHARACTER_MODES else "Off"
        self.character_drive = float(np.clip(character_drive, 0.0, 1.0))
        self.character_post = bool(character_post)

        # Telemetría lock-free compartida con la UI
        self.live_level_db: float = -90.0
        self.live_gr_db: float = 0.0

        # Estado de envolvente por bus (A y B)
        self._gr_env_a: float = 0.0
        self._gr_env_b: float = 0.0
        self._prev_glin_a: float = 1.0
        self._prev_glin_b: float = 1.0
        self._lvl_env_a: float = -90.0
        self._lvl_env_b: float = -90.0
        self._bright_z_a: list[float] = [0.0, 0.0]
        self._bright_z_b: list[float] = [0.0, 0.0]

    def reset_state(self):
        self.live_level_db = -90.0
        self.live_gr_db = 0.0
        self._gr_env_a = 0.0
        self._gr_env_b = 0.0
        self._prev_glin_a = 1.0
        self._prev_glin_b = 1.0
        self._lvl_env_a = -90.0
        self._lvl_env_b = -90.0
        self._bright_z_a = [0.0, 0.0]
        self._bright_z_b = [0.0, 0.0]

    def to_dict(self) -> dict:
        return {
            "f_low": round(self.f_low, 2),
            "f_high": round(self.f_high, 2),
            "enabled": bool(self.enabled),
            "solo": bool(self.solo),
            "bypass": bool(self.bypass),
            "bus_target": str(self.bus_target),
            "mode": str(self.mode),
            "threshold_db": round(self.threshold_db, 2),
            "range_db": round(self.range_db, 2),
            "ratio": round(self.ratio, 2),
            "attack_ms": round(self.attack_ms, 2),
            "release_ms": round(self.release_ms, 1),
            "knee_db": round(self.knee_db, 2),
            "output_gain_db": round(self.output_gain_db, 2),
            "color": str(self.color),
            "custom_color": bool(self.custom_color),
            "character_mode": str(self.character_mode),
            "character_drive": round(self.character_drive, 3),
            "character_post": bool(self.character_post),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DynamicsBand":
        mode = str(data.get("mode", "Compress"))
        if mode not in MB_MODES:
            mode = "Compress"
        default_col = MB_MODE_COLORS.get(mode, "#00E5FF")
        char_mode = str(data.get("character_mode", "Off"))
        if char_mode not in MB_CHARACTER_MODES:
            char_mode = "Off"
        return cls(
            f_low=float(data.get("f_low", 100.0)),
            f_high=float(data.get("f_high", 1000.0)),
            enabled=bool(data.get("enabled", True)),
            solo=bool(data.get("solo", False)),
            bypass=bool(data.get("bypass", False)),
            bus_target=str(data.get("bus_target", "A+B")),
            mode=mode,
            threshold_db=float(data.get("threshold_db", -24.0)),
            range_db=float(data.get("range_db", -12.0)),
            ratio=float(data.get("ratio", 4.0)),
            attack_ms=float(data.get("attack_ms", 10.0)),
            release_ms=float(data.get("release_ms", 120.0)),
            knee_db=float(data.get("knee_db", 6.0)),
            output_gain_db=float(data.get("output_gain_db", 0.0)),
            color=str(data.get("color", default_col)),
            character_mode=char_mode,
            character_drive=float(data.get("character_drive", 0.35)),
            character_post=bool(data.get("character_post", True)),
            custom_color=bool(data.get("custom_color", False)),
        )


class _MBSplitPlan:
    """Plan atómico inmutable para división de bandas LR4 + AP2 en tiempo real sin locks."""

    def __init__(
        self,
        any_active: bool,
        active_bands: tuple[DynamicsBand, ...],
        split_freqs: tuple[float, ...],
        lp4_sos: tuple[np.ndarray, ...],
        hp4_sos: tuple[np.ndarray, ...],
        ap_sos: tuple[np.ndarray | None, ...],
        region_band_a: tuple[DynamicsBand | None, ...],
        region_band_b: tuple[DynamicsBand | None, ...],
        shared_ab: bool,
        any_solo: bool,
        zi_lp_a: list[np.ndarray],
        zi_hp_a: list[np.ndarray],
        zi_ap_a: list[np.ndarray | None],
        zi_lp_b: list[np.ndarray],
        zi_hp_b: list[np.ndarray],
        zi_ap_b: list[np.ndarray | None],
    ):
        self.any_active = any_active
        self.active_bands = active_bands
        self.split_freqs = split_freqs
        self.lp4_sos = lp4_sos
        self.hp4_sos = hp4_sos
        self.ap_sos = ap_sos
        self.region_band_a = region_band_a
        self.region_band_b = region_band_b
        self.shared_ab = shared_ab
        self.any_solo = any_solo
        self.zi_lp_a = zi_lp_a
        self.zi_hp_a = zi_hp_a
        self.zi_ap_a = zi_ap_a
        self.zi_lp_b = zi_lp_b
        self.zi_hp_b = zi_hp_b
        self.zi_ap_b = zi_ap_b


class MultibandDynamicsProcessor:
    """
    Procesador Dinámico Multibanda (Compressor / Expander / Per-Frequency Limiter) estilo Pro-MB:
    - División de bandas libres sin solapamiento mediante cascada LR4 (24 dB/oct) con compensación
      All-Pass AP2 emparejada en fase (suma plana exacta de 0.00 dB a 0 dB de GR).
    - Detección y envolvente en sub-bloques de 8 muestras en buffers float64 `(2, N)` prealocados.
    - Intercambio atómico de plan `_plan` para ejecución 100% lock-free en el callback RT.
    """

    def __init__(self, sample_rate: float = 48000.0):
        self.sr = float(max(8000.0, sample_rate))
        self.lock = threading.RLock()
        self.bypass = False
        self.bands: list[DynamicsBand] = []
        self.static_version: int = 0
        self._tls = threading.local()
        self._plan = _MBSplitPlan(
            any_active=False, active_bands=(), split_freqs=(),
            lp4_sos=(), hp4_sos=(), ap_sos=(),
            region_band_a=(), region_band_b=(),
            shared_ab=True, any_solo=False,
            zi_lp_a=[], zi_hp_a=[], zi_ap_a=[],
            zi_lp_b=[], zi_hp_b=[], zi_ap_b=[],
        )

    def mark_dirty(self):
        with self.lock:
            self.static_version += 1
            self._rebuild_plan()

    def set_sample_rate(self, sr: float):
        with self.lock:
            new_sr = float(max(8000.0, sr))
            if abs(new_sr - self.sr) > 1.0:
                self.sr = new_sr
                self.static_version += 1
                self._rebuild_plan()

    def _max_band_freq(self) -> float:
        return min(22000.0, 0.47 * self.sr)

    def _clamp_non_overlapping(self):
        """Ordena por f_low y garantiza que ninguna banda se solape con sus vecinas."""
        max_f = self._max_band_freq()
        self.bands.sort(key=lambda b: b.f_low)
        for i, b in enumerate(self.bands):
            lo_bound = 15.0 if i == 0 else max(15.0, self.bands[i - 1].f_high)
            hi_bound = max_f if i == len(self.bands) - 1 else min(max_f, self.bands[i + 1].f_low)
            b.f_low = float(np.clip(b.f_low, lo_bound, max(lo_bound, max_f / 1.15)))
            min_hi = min(max_f, b.f_low * 1.15)
            b.f_high = float(np.clip(b.f_high, min_hi, max(min_hi, hi_bound)))

    def get_band_edge_limits(self, idx: int) -> tuple[float, float]:
        max_f = self._max_band_freq()
        if not (0 <= idx < len(self.bands)):
            return 15.0, max_f
        lo = 15.0 if idx == 0 else self.bands[idx - 1].f_high
        hi = max_f if idx == len(self.bands) - 1 else self.bands[idx + 1].f_low
        return float(lo), float(hi)

    def add_band(
        self,
        f_low: float | None = None,
        f_high: float | None = None,
        mode: str = "Compress",
        threshold_db: float = -24.0,
        range_db: float | None = None,
        ratio: float = 4.0,
        attack_ms: float | None = None,
        release_ms: float = 120.0,
        knee_db: float = 6.0,
        output_gain_db: float = 0.0,
        bus_target: str = "A+B",
        color: str | None = None,
        character_mode: str = "Off",
        character_drive: float = 0.35,
        character_post: bool = True,
        custom_color: bool = False,
    ) -> DynamicsBand | None:
        with self.lock:
            if len(self.bands) >= MAX_MB_BANDS:
                return None
            max_f = self._max_band_freq()
            existing = sorted(self.bands, key=lambda b: b.f_low)

            if f_low is None or f_high is None:
                # Buscar el mayor hueco logarítmico disponible
                edges = [20.0]
                for eb in existing:
                    edges.append(eb.f_low)
                    edges.append(eb.f_high)
                edges.append(min(18000.0, max_f))
                best_gap = None
                best_span = 0.0
                for k in range(0, len(edges), 2):
                    g_lo, g_hi = edges[k], edges[k + 1]
                    if g_hi > g_lo * 1.25:
                        span = math.log10(g_hi / g_lo)
                        if span > best_span:
                            best_span = span
                            best_gap = (g_lo, g_hi)
                if best_gap is None:
                    return None
                g_lo, g_hi = best_gap
                f_low = g_lo * ((g_hi / g_lo) ** 0.15)
                f_high = g_lo * ((g_hi / g_lo) ** 0.85)
            else:
                f_low = float(np.clip(f_low, 15.0, max_f / 1.15))
                f_high = float(np.clip(f_high, f_low * 1.15, max_f))
                # Ajustar dentro del hueco que contiene el centro geométrico para evitar solapamientos
                fc = math.sqrt(f_low * f_high)
                lo_lim, hi_lim = 15.0, max_f
                for eb in existing:
                    if eb.f_high <= fc:
                        lo_lim = max(lo_lim, eb.f_high)
                    elif eb.f_low >= fc:
                        hi_lim = min(hi_lim, eb.f_low)
                        break
                    else:
                        # Dentro de una banda existente: buscar el hueco libre más amplio
                        return None
                if hi_lim < lo_lim * 1.15:
                    return None
                f_low = max(lo_lim, min(f_low, hi_lim / 1.15))
                f_high = min(hi_lim, max(f_high, f_low * 1.15))

            if mode not in MB_MODES:
                mode = "Compress"
            if range_db is None:
                range_db = -30.0 if mode == "Limit" else (-12.0 if mode == "Compress" else 6.0)
            if attack_ms is None:
                attack_ms = 0.5 if mode == "Limit" else 10.0
            if mode == "Limit":
                ratio = 20.0

            band = DynamicsBand(
                f_low=f_low,
                f_high=f_high,
                enabled=True,
                bus_target=bus_target,
                mode=mode,
                threshold_db=threshold_db,
                range_db=range_db,
                ratio=ratio,
                attack_ms=attack_ms,
                release_ms=release_ms,
                knee_db=knee_db,
                output_gain_db=output_gain_db,
                color=color,
                character_mode=character_mode,
                character_drive=character_drive,
                character_post=character_post,
                custom_color=custom_color,
            )
            self.bands.append(band)
            self._clamp_non_overlapping()
            self.static_version += 1
            self._rebuild_plan()
            return band

    def remove_band(self, index: int):
        with self.lock:
            if 0 <= index < len(self.bands):
                self.bands.pop(index)
                self.static_version += 1
                self._rebuild_plan()

    def set_band_edges(self, index: int, f_low: float, f_high: float):
        with self.lock:
            if not (0 <= index < len(self.bands)):
                return
            lo_lim, hi_lim = self.get_band_edge_limits(index)
            if hi_lim <= lo_lim * 1.15:
                return
            new_lo = float(np.clip(f_low, lo_lim, hi_lim / 1.15))
            new_hi = float(np.clip(f_high, new_lo * 1.15, hi_lim))
            b = self.bands[index]
            b.f_low = new_lo
            b.f_high = new_hi
            self.static_version += 1
            self._rebuild_plan()

    def set_solo_band(self, index: int | None, solo: bool = True):
        with self.lock:
            for i, b in enumerate(self.bands):
                if index is not None and i == index:
                    b.solo = bool(solo)
                elif solo:
                    b.solo = False
            self.static_version += 1
            self._rebuild_plan()

    def _rebuild_plan(self):
        active = tuple(
            b for b in sorted(self.bands, key=lambda x: x.f_low)
            if b.enabled and not b.bypass
        )
        if not active:
            self._plan = _MBSplitPlan(
                any_active=False, active_bands=(), split_freqs=(),
                lp4_sos=(), hp4_sos=(), ap_sos=(),
                region_band_a=(), region_band_b=(),
                shared_ab=True, any_solo=False,
                zi_lp_a=[], zi_hp_a=[], zi_ap_a=[],
                zi_lp_b=[], zi_hp_b=[], zi_ap_b=[],
            )
            return

        lo_open = 12.0
        hi_open = self._max_band_freq() * 0.995
        raw_splits: list[float] = []
        for b in active:
            if b.f_low > lo_open:
                raw_splits.append(b.f_low)
            if b.f_high < hi_open:
                raw_splits.append(b.f_high)
        raw_splits.sort()

        splits: list[float] = []
        for f in raw_splits:
            if not splits or abs(f - splits[-1]) / max(1.0, splits[-1]) > 0.002:
                splits.append(float(f))

        K = len(splits)
        lp_list: list[np.ndarray] = []
        hp_list: list[np.ndarray] = []
        ap_single: list[np.ndarray] = []
        for fc in splits:
            lp, hp, ap = design_lr4_and_ap2(fc, self.sr)
            lp_list.append(lp)
            hp_list.append(hp)
            ap_single.append(ap)

        ap_packed: list[np.ndarray | None] = []
        for i in range(K):
            if i + 1 < K:
                ap_packed.append(np.ascontiguousarray(np.vstack(ap_single[i + 1:]), dtype=np.float64))
            else:
                ap_packed.append(None)

        # Conservar estados zi cuando el número de puntos de cruce no cambia (arrastre suave sin clics)
        old = getattr(self, "_plan", None)
        can_reuse_zi = (old is not None and len(old.split_freqs) == K)

        zi_lp_a, zi_hp_a, zi_ap_a = [], [], []
        zi_lp_b, zi_hp_b, zi_ap_b = [], [], []
        for i in range(K):
            if can_reuse_zi:
                zi_lp_a.append(old.zi_lp_a[i].copy())
                zi_hp_a.append(old.zi_hp_a[i].copy())
                zi_lp_b.append(old.zi_lp_b[i].copy())
                zi_hp_b.append(old.zi_hp_b[i].copy())
            else:
                zi_lp_a.append(np.zeros((2, 2, 2), dtype=np.float64))
                zi_hp_a.append(np.zeros((2, 2, 2), dtype=np.float64))
                zi_lp_b.append(np.zeros((2, 2, 2), dtype=np.float64))
                zi_hp_b.append(np.zeros((2, 2, 2), dtype=np.float64))

            ap_s = ap_packed[i]
            if ap_s is not None:
                n_sec = ap_s.shape[0]
                if can_reuse_zi and old.zi_ap_a[i] is not None and old.zi_ap_a[i].shape == (2, n_sec, 2):
                    zi_ap_a.append(old.zi_ap_a[i].copy())
                    zi_ap_b.append(old.zi_ap_b[i].copy())
                else:
                    zi_ap_a.append(np.zeros((2, n_sec, 2), dtype=np.float64))
                    zi_ap_b.append(np.zeros((2, n_sec, 2), dtype=np.float64))
            else:
                zi_ap_a.append(None)
                zi_ap_b.append(None)

        reg_edges = [10.0] + splits + [self.sr * 0.49]
        reg_band_a: list[DynamicsBand | None] = []
        reg_band_b: list[DynamicsBand | None] = []
        for r in range(K + 1):
            fc_r = math.sqrt(max(10.0, reg_edges[r]) * max(10.0, reg_edges[r + 1]))
            owner = None
            for b in active:
                if b.f_low <= fc_r <= b.f_high:
                    owner = b
                    break
            if owner is not None:
                reg_band_a.append(owner if owner.bus_target in ("A", "A+B") else None)
                reg_band_b.append(owner if owner.bus_target in ("B", "A+B") else None)
            else:
                reg_band_a.append(None)
                reg_band_b.append(None)

        shared_ab = all(b.bus_target == "A+B" for b in active)
        any_solo = any(b.solo for b in active)

        self._plan = _MBSplitPlan(
            any_active=True,
            active_bands=active,
            split_freqs=tuple(splits),
            lp4_sos=tuple(lp_list),
            hp4_sos=tuple(hp_list),
            ap_sos=tuple(ap_packed),
            region_band_a=tuple(reg_band_a),
            region_band_b=tuple(reg_band_b),
            shared_ab=shared_ab,
            any_solo=any_solo,
            zi_lp_a=zi_lp_a, zi_hp_a=zi_hp_a, zi_ap_a=zi_ap_a,
            zi_lp_b=zi_lp_b, zi_hp_b=zi_hp_b, zi_ap_b=zi_ap_b,
        )

    def _ensure_tls(self, frames: int):
        tls = self._tls
        if getattr(tls, "frames", -1) != frames:
            tls.frames = frames
            tls.work_a = np.empty((2, frames), dtype=np.float64)
            tls.work_b = np.empty((2, frames), dtype=np.float64)
            tls.rem_buf = np.empty((2, frames), dtype=np.float64)
            tls.reg_buf = np.empty((2, frames), dtype=np.float64)
            tls.sum_buf = np.empty((2, frames), dtype=np.float64)
            tls.abs_l = np.empty(frames, dtype=np.float64)
            tls.abs_r = np.empty(frames, dtype=np.float64)
            tls.gain_1d = np.empty(frames, dtype=np.float64)
            tls.ramp_unit = np.arange(1, 9, dtype=np.float64) / 8.0
            tls.sat_tmp = np.empty((2, frames), dtype=np.float64)
            tls.out_a_f32 = np.empty((frames, 2), dtype=np.float32)
            tls.out_b_f32 = np.empty((frames, 2), dtype=np.float32)
        return tls

    @staticmethod
    def _apply_character_inplace(
        buf_2xN: np.ndarray,
        band: DynamicsBand,
        is_bus_b: bool,
        sat_tmp: np.ndarray,
    ):
        cm = band.character_mode
        drv = band.character_drive
        if cm == "Off" or drv <= 1e-4:
            return
        frames = buf_2xN.shape[1]
        if frames == 0:
            return

        if cm == "Tube":
            k = 1.0 + 3.5 * drv
            bias = 0.12 * drv
            tb = math.tanh(bias)
            norm = 1.0 / (k * max(1e-6, 1.0 - tb * tb))
            np.multiply(buf_2xN, k, out=sat_tmp)
            sat_tmp += bias
            np.tanh(sat_tmp, out=sat_tmp)
            sat_tmp -= tb
            np.multiply(sat_tmp, norm, out=buf_2xN)
        elif cm == "Diode":
            k = 1.0 + 5.0 * drv
            inv_k = 1.0 / k
            np.multiply(buf_2xN, k, out=sat_tmp)
            np.arctan(sat_tmp, out=sat_tmp)
            np.multiply(sat_tmp, inv_k, out=buf_2xN)
        elif cm == "Bright":
            z = band._bright_z_b if is_bus_b else band._bright_z_a
            for ch in (0, 1):
                x = buf_2xN[ch]
                d = sat_tmp[ch]
                d[0] = x[0] - 0.85 * z[ch]
                if frames > 1:
                    np.multiply(x[:-1], 0.85, out=d[1:])
                    np.subtract(x[1:], d[1:], out=d[1:])
                z[ch] = float(x[-1])
            bias_b = 0.14 * drv
            tb_b = math.tanh(bias_b)
            sat_tmp *= (8.0 + 22.0 * drv)
            sat_tmp += bias_b
            np.tanh(sat_tmp, out=sat_tmp)
            sat_tmp -= tb_b
            sat_tmp *= (0.28 * drv)
            buf_2xN += sat_tmp

    def _process_band_inplace(self, buf_2xN: np.ndarray, band: DynamicsBand, is_bus_b: bool, tls, write_telemetry: bool):
        frames = buf_2xN.shape[1]
        if frames == 0:
            return
        if not band.character_post:
            self._apply_character_inplace(buf_2xN, band, is_bus_b, tls.sat_tmp)
        ch0 = buf_2xN[0]
        ch1 = buf_2xN[1]
        abs_l = tls.abs_l
        abs_r = tls.abs_r
        gain_1d = tls.gain_1d
        ramp_unit = tls.ramp_unit

        mode = band.mode
        thresh_db = band.threshold_db
        range_db = band.range_db
        ratio = max(1.0, band.ratio)
        kw = max(0.0, band.knee_db)
        out_gain_db = band.output_gain_db
        att_sec = max(0.0001, band.attack_ms * 0.001)
        rel_sec = max(0.0005, band.release_ms * 0.001)
        sr = self.sr

        if mode == "Limit":
            np.abs(ch0, out=abs_l)
            np.abs(ch1, out=abs_r)

        gr_env = band._gr_env_b if is_bus_b else band._gr_env_a
        prev_glin = band._prev_glin_b if is_bus_b else band._prev_glin_a

        peak_lvl_db = -90.0
        ext_gr_db = 0.0
        sub_sz = 8

        for start in range(0, frames, sub_sz):
            end = start + sub_sz
            if end > frames:
                end = frames
            n_sub = end - start
            dt = n_sub / sr
            a_att = math.exp(-dt / att_sec)
            a_rel = math.exp(-dt / rel_sec)

            if mode == "Limit":
                pk = max(float(np.max(abs_l[start:end])), float(np.max(abs_r[start:end])))
                lvl_db = 20.0 * math.log10(pk + 1e-10)
                over = lvl_db - thresh_db
                gr_target = -over if over > 0.0 else 0.0
                if gr_target < -60.0:
                    gr_target = -60.0
            else:
                s0 = ch0[start:end]
                s1 = ch1[start:end]
                e = 0.5 * (float(np.dot(s0, s0)) + float(np.dot(s1, s1))) / n_sub
                lvl_db = 10.0 * math.log10(e + 1e-20)

                if mode == "Compress":
                    over = lvl_db - thresh_db
                    if kw <= 1e-4:
                        eff_over = over if over > 0.0 else 0.0
                    elif over <= -0.5 * kw:
                        eff_over = 0.0
                    elif over < 0.5 * kw:
                        t = over + 0.5 * kw
                        eff_over = (t * t) / (2.0 * kw)
                    else:
                        eff_over = over
                    gr_target = -eff_over * (1.0 - 1.0 / ratio)
                    max_cut = -abs(range_db)
                    if gr_target < max_cut:
                        gr_target = max_cut
                    elif gr_target > 0.0:
                        gr_target = 0.0
                else:  # "Expand"
                    if range_db >= 0.0:
                        over = lvl_db - thresh_db
                        if kw <= 1e-4:
                            eff = over if over > 0.0 else 0.0
                        elif over <= -0.5 * kw:
                            eff = 0.0
                        elif over < 0.5 * kw:
                            t = over + 0.5 * kw
                            eff = (t * t) / (2.0 * kw)
                        else:
                            eff = over
                        gr_target = eff * (1.0 - 1.0 / ratio)
                        if gr_target > range_db:
                            gr_target = range_db
                        elif gr_target < 0.0:
                            gr_target = 0.0
                    else:
                        under = thresh_db - lvl_db
                        if kw <= 1e-4:
                            eff = under if under > 0.0 else 0.0
                        elif under <= -0.5 * kw:
                            eff = 0.0
                        elif under < 0.5 * kw:
                            t = under + 0.5 * kw
                            eff = (t * t) / (2.0 * kw)
                        else:
                            eff = under
                        gr_target = -eff * max(0.0, ratio - 1.0)
                        if gr_target < range_db:
                            gr_target = range_db
                        elif gr_target > 0.0:
                            gr_target = 0.0

            is_attack = abs(gr_target) > abs(gr_env)
            alpha = a_att if is_attack else a_rel
            gr_env = alpha * gr_env + (1.0 - alpha) * gr_target
            if abs(gr_env) < 1e-4 and abs(gr_target) < 1e-4:
                gr_env = 0.0

            g_lin = 10.0 ** ((gr_env + out_gain_db) * 0.05)
            dg = g_lin - prev_glin
            if abs(dg) < 1e-6:
                gain_1d[start:end] = g_lin
            elif n_sub == 8:
                np.multiply(ramp_unit, dg, out=gain_1d[start:end])
                gain_1d[start:end] += prev_glin
            else:
                inv_n = 1.0 / float(n_sub)
                for k in range(n_sub):
                    gain_1d[start + k] = prev_glin + dg * ((k + 1) * inv_n)
            prev_glin = g_lin

            if lvl_db > peak_lvl_db:
                peak_lvl_db = lvl_db
            if abs(gr_env) > abs(ext_gr_db):
                ext_gr_db = gr_env

        ch0 *= gain_1d
        ch1 *= gain_1d

        if band.character_post:
            self._apply_character_inplace(buf_2xN, band, is_bus_b, tls.sat_tmp)

        if mode == "Limit":
            ceil_lin = 10.0 ** ((thresh_db + out_gain_db) * 0.05)
            np.clip(ch0, -ceil_lin, ceil_lin, out=ch0)
            np.clip(ch1, -ceil_lin, ceil_lin, out=ch1)

        if is_bus_b:
            band._gr_env_b = gr_env
            band._prev_glin_b = prev_glin
            band._lvl_env_b = peak_lvl_db
        else:
            band._gr_env_a = gr_env
            band._prev_glin_a = prev_glin
            band._lvl_env_a = peak_lvl_db

        if write_telemetry:
            band.live_level_db = float(peak_lvl_db)
            band.live_gr_db = float(ext_gr_db)

    def _process_single_bus(
        self,
        in_2xN: np.ndarray,
        plan: _MBSplitPlan,
        is_bus_b: bool,
        tls,
    ) -> np.ndarray:
        frames = in_2xN.shape[1]
        rem = tls.rem_buf
        reg = tls.reg_buf
        sum_buf = tls.sum_buf
        np.copyto(rem, in_2xN)
        sum_buf.fill(0.0)

        zi_lp = plan.zi_lp_b if is_bus_b else plan.zi_lp_a
        zi_hp = plan.zi_hp_b if is_bus_b else plan.zi_hp_a
        zi_ap = plan.zi_ap_b if is_bus_b else plan.zi_ap_a
        reg_bands = plan.region_band_b if is_bus_b else plan.region_band_a
        any_solo = plan.any_solo
        K = len(plan.split_freqs)

        for i in range(K):
            np.copyto(reg, rem)
            _apply_sos_channel_first_inplace(plan.lp4_sos[i], reg, zi_lp[i], "LR")
            _apply_sos_channel_first_inplace(plan.hp4_sos[i], rem, zi_hp[i], "LR")
            ap_s = plan.ap_sos[i]
            if ap_s is not None and zi_ap[i] is not None:
                _apply_sos_channel_first_inplace(ap_s, reg, zi_ap[i], "LR")

            owner = reg_bands[i]
            if owner is not None:
                write_tel = (not is_bus_b) or (owner.bus_target == "B")
                self._process_band_inplace(reg, owner, is_bus_b, tls, write_tel)
            if not any_solo or (owner is not None and owner.solo):
                sum_buf += reg

        # Región superior restante
        owner_top = reg_bands[K]
        if owner_top is not None:
            write_tel = (not is_bus_b) or (owner_top.bus_target == "B")
            self._process_band_inplace(rem, owner_top, is_bus_b, tls, write_tel)
        if not any_solo or (owner_top is not None and owner_top.solo):
            sum_buf += rem

        return sum_buf

    def process_dual_block(self, out_a: np.ndarray, out_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Procesa en paralelo phase-coherent out_a y out_b en tiempo real sin adquirir locks.
        Utiliza ruta rápida single-pass cuando todas las bandas activas apuntan a 'A+B'
        y out_a / out_b son idénticos.
        """
        plan = self._plan
        frames = out_a.shape[0] if out_a.ndim > 0 else 0
        if self.bypass or not plan.any_active or frames == 0:
            for b in plan.active_bands:
                b.live_gr_db = 0.0
                b.live_level_db = -90.0
            return out_a, out_b

        tls = self._ensure_tls(frames)
        work_a = tls.work_a
        work_a[0] = out_a[:, 0]
        work_a[1] = out_a[:, 1]

        single_pass = plan.shared_ab and (out_a is out_b or np.array_equal(out_a, out_b))
        sum_a = self._process_single_bus(work_a, plan, is_bus_b=False, tls=tls)
        out_a_f32 = tls.out_a_f32
        out_a_f32[:, 0] = sum_a[0]
        out_a_f32[:, 1] = sum_a[1]

        if single_pass:
            K = len(plan.split_freqs)
            for i in range(K):
                np.copyto(plan.zi_lp_b[i], plan.zi_lp_a[i])
                np.copyto(plan.zi_hp_b[i], plan.zi_hp_a[i])
                if plan.zi_ap_b[i] is not None and plan.zi_ap_a[i] is not None:
                    np.copyto(plan.zi_ap_b[i], plan.zi_ap_a[i])
            for b in plan.active_bands:
                b._gr_env_b = b._gr_env_a
                b._prev_glin_b = b._prev_glin_a
                b._lvl_env_b = b._lvl_env_a
                b._bright_z_b[0] = b._bright_z_a[0]
                b._bright_z_b[1] = b._bright_z_a[1]
            out_b_f32 = tls.out_b_f32
            np.copyto(out_b_f32, out_a_f32)
            return out_a_f32, out_b_f32

        work_b = tls.work_b
        work_b[0] = out_b[:, 0]
        work_b[1] = out_b[:, 1]
        sum_b = self._process_single_bus(work_b, plan, is_bus_b=True, tls=tls)
        out_b_f32 = tls.out_b_f32
        out_b_f32[:, 0] = sum_b[0]
        out_b_f32[:, 1] = sum_b[1]
        return out_a_f32, out_b_f32

    def get_band_response_curves_db(
        self, z1: np.ndarray, z2: np.ndarray, target_bus: str = "A"
    ) -> tuple[dict[int, np.ndarray], np.ndarray]:
        """
        Devuelve `(band_curves_by_idx, net_db)` evaluando de forma exacta la cascada LR4+AP2:
        - Cada banda `idx` recibe su curva individual analítica (0 dB fuera de sus cruces y suave LR4 dentro)
          para que la sombra de banda y la curva maestra coincidan píxel por píxel.
        """
        n = len(z1)
        zeros = np.zeros(n, dtype=np.float64)
        plan = self._plan
        if self.bypass or not plan.any_active:
            return {}, zeros

        is_b = (target_bus == "B")
        reg_bands = plan.region_band_b if is_b else plan.region_band_a
        K = len(plan.split_freqs)
        any_solo = plan.any_solo

        rem_h = np.ones(n, dtype=np.complex128)
        tot_h = np.zeros(n, dtype=np.complex128)
        ap_total_h = np.zeros(n, dtype=np.complex128)
        owner_reg_h: dict[int, np.ndarray] = {}
        id_to_band: dict[int, DynamicsBand] = {}

        for i in range(K):
            h_lp = eval_sos_complex_fast(plan.lp4_sos[i], z1, z2)
            h_hp = eval_sos_complex_fast(plan.hp4_sos[i], z1, z2)
            reg_h = rem_h * h_lp
            ap_s = plan.ap_sos[i]
            if ap_s is not None:
                reg_h = reg_h * eval_sos_complex_fast(ap_s, z1, z2)
            ap_total_h += reg_h

            owner = reg_bands[i]
            if owner is not None:
                oid = id(owner)
                id_to_band[oid] = owner
                if oid in owner_reg_h:
                    owner_reg_h[oid] = owner_reg_h[oid] + reg_h
                else:
                    owner_reg_h[oid] = reg_h
            if not any_solo or (owner is not None and owner.solo):
                g_db = (owner.output_gain_db + owner.live_gr_db) if owner is not None else 0.0
                tot_h += reg_h * (10.0 ** (g_db * 0.05))
            rem_h = rem_h * h_hp

        ap_total_h += rem_h
        owner_top = reg_bands[K]
        if owner_top is not None:
            oid = id(owner_top)
            id_to_band[oid] = owner_top
            if oid in owner_reg_h:
                owner_reg_h[oid] = owner_reg_h[oid] + rem_h
            else:
                owner_reg_h[oid] = rem_h
        if not any_solo or (owner_top is not None and owner_top.solo):
            g_db = (owner_top.output_gain_db + owner_top.live_gr_db) if owner_top is not None else 0.0
            tot_h += rem_h * (10.0 ** (g_db * 0.05))

        band_curves: dict[int, np.ndarray] = {}
        for idx, b in enumerate(self.bands):
            oid = id(b)
            h_b = owner_reg_h.get(oid)
            if h_b is None or (any_solo and not b.solo):
                band_curves[idx] = zeros
                continue
            g_db = b.output_gain_db + b.live_gr_db
            if abs(g_db) < 0.02:
                band_curves[idx] = zeros
            else:
                g_lin = 10.0 ** (g_db * 0.05)
                h_single = ap_total_h + (g_lin - 1.0) * h_b
                band_curves[idx] = 20.0 * np.log10(np.maximum(np.abs(h_single), 1e-6))

        net_db = 20.0 * np.log10(np.maximum(np.abs(tot_h), 1e-6))
        return band_curves, net_db

    def get_net_response_db(self, z1: np.ndarray, z2: np.ndarray, target_bus: str = "A") -> np.ndarray:
        """Calcula la respuesta neta analítica LR4+AP2 (ganancia de salida + GR en vivo)."""
        _, net_db = self.get_band_response_curves_db(z1, z2, target_bus=target_bus)
        return net_db

    def to_dict(self) -> dict:
        with self.lock:
            return {
                "bypass": bool(self.bypass),
                "bands": [b.to_dict() for b in self.bands],
            }

    def load_from_dict(self, data: dict):
        if not isinstance(data, dict):
            return
        with self.lock:
            self.bypass = bool(data.get("bypass", False))
            loaded: list[DynamicsBand] = []
            raw_bands = data.get("bands", [])
            if isinstance(raw_bands, list):
                for item in raw_bands[:MAX_MB_BANDS]:
                    if isinstance(item, dict):
                        loaded.append(DynamicsBand.from_dict(item))
            self.bands = loaded
            self._clamp_non_overlapping()
            self.static_version += 1
            self._rebuild_plan()
