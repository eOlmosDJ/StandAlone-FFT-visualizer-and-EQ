import math
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QFrame, QMenu, QColorDialog, QDialog, QToolTip
)
from PyQt6.QtCore import Qt, QEvent, QPoint, QPointF, QRectF, pyqtSignal
from PyQt6.QtGui import (
    QColor, QFont, QPainter, QPainterPath, QPen, QBrush,
    QLinearGradient, QPolygonF, QPixmap, QIcon
)
import pyqtgraph as pg

from eq_dsp import ParametricEQEngine, EQBand, FILTER_TYPES, SLOPE_OPTIONS


STUDIO_BAND_COLORS = [
    ("Cian", "#00E5FF"),
    ("Verde Esmeralda", "#00E676"),
    ("Ámbar", "#FFB300"),
    ("Coral", "#FF5252"),
    ("Violeta", "#7C4DFF"),
    ("Azul Eléctrico", "#2979FF"),
    ("Rosa", "#FF4081"),
    ("Dorado", "#FFD54F"),
]


def _brighten_color(hex_or_qcolor, ratio: float = 0.25) -> QColor:
    """Mezcla un color con blanco en `ratio` (ej. 25%) para darle brillo alto."""
    c = QColor(hex_or_qcolor)
    r = min(255, int(round(c.red() * (1.0 - ratio) + 255 * ratio)))
    g = min(255, int(round(c.green() * (1.0 - ratio) + 255 * ratio)))
    b = min(255, int(round(c.blue() * (1.0 - ratio) + 255 * ratio)))
    return QColor(r, g, b)


def _make_color_icon(hex_color: str, size: int = 12) -> QIcon:
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setPen(QPen(QColor("#E0E4EC"), 1.0))
    p.setBrush(QBrush(QColor(hex_color)))
    p.drawEllipse(QRectF(1.0, 1.0, size - 2.0, size - 2.0))
    p.end()
    return QIcon(pix)


class LiquidGlassButton(QPushButton):
    """
    Cápsula translúcida estilo Apple Liquid Glass dibujada con QPainter:
    - Relleno Vítreo: QLinearGradient(0, 0, 0, height) desde rgba(255, 255, 255, 36) arriba
      a rgba(255, 255, 255, 10) en el centro y rgba(15, 17, 22, 185) en la base.
    - Borde Especular: Trazo de 1.2px con QLinearGradient diagonal (rgba(255,255,255,145) ->
      rgba(255,255,255,25) -> rgba(255,255,255,65)).
    - Hover & Active: En hover sube la luminancia a rgba(255, 255, 255, 55); activo toma
      un tinte translúcido del color de estado manteniendo el reflejo especular blanco.
    """

    def __init__(
        self,
        text: str = "",
        parent=None,
        tint_color: str | None = None,
        active_tint_color: str | None = "#00E5FF",
        pill_radius: float = 13.0,
    ):
        super().__init__(text, parent)
        self._tint_color = QColor(tint_color) if tint_color else None
        self._active_tint_color = QColor(active_tint_color) if active_tint_color else QColor("#00E5FF")
        self._pill_radius = float(pill_radius)
        self._hovered = False
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(28)

    def setTintColor(self, hex_color: str | None):
        self._tint_color = QColor(hex_color) if hex_color else None
        self.update()

    def setActiveTintColor(self, hex_color: str | None):
        if hex_color:
            self._active_tint_color = QColor(hex_color)
            self.update()

    def enterEvent(self, event):
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        w = float(self.width())
        h = float(self.height())
        rect = QRectF(1.0, 1.0, w - 2.0, h - 2.0)
        r = min(self._pill_radius, (h - 2.0) * 0.5)

        is_active = (self.isCheckable() and self.isChecked()) or (self._tint_color is not None)
        tint = self._active_tint_color if (self.isCheckable() and self.isChecked()) else self._tint_color

        # 1. Relleno Vítreo Vertical
        fill_grad = QLinearGradient(0.0, 0.0, 0.0, h)
        if is_active and tint is not None and self.isEnabled():
            top_a = 155 if self._hovered else 130
            mid_a = 110 if self._hovered else 85
            bot_a = 195
            c_top = _brighten_color(tint, 0.32 if self._hovered else 0.22)
            c_top.setAlpha(top_a)
            c_mid = QColor(tint)
            c_mid.setAlpha(mid_a)
            c_bot = QColor(
                max(0, int(tint.red() * 0.35)),
                max(0, int(tint.green() * 0.35)),
                max(0, int(tint.blue() * 0.35)),
                bot_a,
            )
            fill_grad.setColorAt(0.0, c_top)
            fill_grad.setColorAt(0.48, c_mid)
            fill_grad.setColorAt(1.0, c_bot)
        else:
            top_white_a = 55 if (self._hovered and self.isEnabled()) else 36
            mid_white_a = 22 if (self._hovered and self.isEnabled()) else 10
            fill_grad.setColorAt(0.0, QColor(255, 255, 255, top_white_a))
            fill_grad.setColorAt(0.45, QColor(255, 255, 255, mid_white_a))
            fill_grad.setColorAt(1.0, QColor(15, 17, 22, 185))

        # 2. Borde Especular Diagonal Liquid Glass (1.2px)
        spec_grad = QLinearGradient(0.0, 0.0, w, h)
        spec_grad.setColorAt(0.0, QColor(255, 255, 255, 165 if self._hovered else 145))
        if is_active and tint is not None and self.isEnabled():
            mid_spec = _brighten_color(tint, 0.25)
            mid_spec.setAlpha(95)
            bot_spec = _brighten_color(tint, 0.15)
            bot_spec.setAlpha(135)
            spec_grad.setColorAt(0.5, mid_spec)
            spec_grad.setColorAt(1.0, bot_spec)
        else:
            spec_grad.setColorAt(0.5, QColor(255, 255, 255, 25))
            spec_grad.setColorAt(1.0, QColor(255, 255, 255, 65))

        painter.setPen(QPen(QBrush(spec_grad), 1.2))
        painter.setBrush(QBrush(fill_grad))
        painter.drawRoundedRect(rect, r, r)

        # Reflejo superior interno de cristal (arco especular fino)
        if w > 16.0:
            hi_pen = QPen(QColor(255, 255, 255, 65 if self._hovered else 42), 0.9)
            painter.setPen(hi_pen)
            painter.drawLine(QPointF(r * 0.85, 2.1), QPointF(w - r * 0.85, 2.1))

        # 3. Texto nítido centrado
        font = QFont("Segoe UI", 8, QFont.Weight.Bold)
        painter.setFont(font)
        if not self.isEnabled():
            txt_c = QColor(120, 126, 138, 150)
        elif is_active and tint is not None:
            lum = 0.299 * tint.red() + 0.587 * tint.green() + 0.114 * tint.blue()
            txt_c = QColor("#0E1015") if lum > 175 else QColor("#FFFFFF")
        else:
            txt_c = QColor("#FFFFFF") if self._hovered else QColor("#E5E9F0")

        painter.setPen(txt_c)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self.text())


class PowerToggleButton(QPushButton):
    """
    Botón Power (On/Off) de 24x24 px dibujado con QPainter (Antialiasing) estilo FabFilter Pro-Q 4.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(True)
        self.setFixedSize(24, 24)
        self._accent_color = QColor("#00E5FF")
        self._hovered = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def setAccentColor(self, hex_color: str):
        c = QColor(hex_color)
        if c.isValid() and c != self._accent_color:
            self._accent_color = c
            self.update()

    def enterEvent(self, event):
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        w = float(self.width())
        h = float(self.height())
        rect = QRectF(1.0, 1.0, w - 2.0, h - 2.0)

        if self.isChecked() and self.isEnabled():
            bg = QColor(self._accent_color)
            bg.setAlpha(65 if self._hovered else 42)
            border_c = _brighten_color(self._accent_color, 0.2) if self._hovered else QColor(self._accent_color)
            icon_c = QColor("#FFFFFF")
        else:
            bg = QColor(255, 255, 255, 26 if self._hovered else 14)
            border_c = QColor(255, 255, 255, 65 if self._hovered else 38)
            icon_c = QColor("#787E8C")

        painter.setPen(QPen(border_c, 1.1))
        painter.setBrush(QBrush(bg))
        painter.drawRoundedRect(rect, 6.0, 6.0)

        cx = w * 0.5
        cy = h * 0.5 + 0.5
        r = 5.6
        arc_rect = QRectF(cx - r, cy - r, 2.0 * r, 2.0 * r)
        pen = QPen(icon_c, 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawArc(arc_rect, 120 * 16, -300 * 16)
        painter.drawLine(QPointF(cx, cy - r - 1.3), QPointF(cx, cy - 0.5))


class ColorSwatchButton(QPushButton):
    """
    Selector circular de color (16x16 px visual) para cambiar el color del nodo y de la banda.
    """

    colorSelected = pyqtSignal(str)

    def __init__(self, initial_color: str = "#00E5FF", parent=None):
        super().__init__(parent)
        self._color = QColor(initial_color)
        self.setFixedSize(22, 24)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Cambiar color del nodo, curvas, sombreado y perillas de esta banda")
        self.clicked.connect(self._open_palette_menu)

    def colorName(self) -> str:
        return self._color.name()

    def setColor(self, hex_color: str):
        c = QColor(hex_color)
        if c.isValid() and c != self._color:
            self._color = c
            self.update()

    def _open_palette_menu(self):
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #1A1C22;
                color: #ECEFF4;
                border: 1px solid #383C47;
                border-radius: 6px;
                padding: 4px;
                font-size: 11px;
            }
            QMenu::item { padding: 5px 14px; border-radius: 4px; }
            QMenu::item:selected { background-color: #2E3440; color: #FFFFFF; }
        """)
        for label, hex_val in STUDIO_BAND_COLORS:
            act = menu.addAction(_make_color_icon(hex_val, 12), label)
            act.triggered.connect(lambda _=False, hv=hex_val: self._emit_color(hv))

        menu.addSeparator()
        act_custom = menu.addAction("🎨 Personalizado...")
        act_custom.triggered.connect(self._pick_custom_color)
        menu.exec(self.mapToGlobal(QPoint(0, self.height() + 2)))

    def _emit_color(self, hex_val: str):
        self.setColor(hex_val)
        self.colorSelected.emit(self._color.name())

    def _pick_custom_color(self):
        original = self._color.name()
        dlg = QColorDialog(self._color, self)
        dlg.setWindowTitle("Color del Nodo / Banda EQ")
        dlg.setOption(QColorDialog.ColorDialogOption.DontUseNativeDialog, True)
        dlg.currentColorChanged.connect(lambda c: self._emit_color(c.name()) if c.isValid() else None)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._emit_color(dlg.selectedColor().name())
        else:
            self._emit_color(original)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        cx = self.width() * 0.5
        cy = self.height() * 0.5
        r = 7.2
        painter.setPen(QPen(QColor(255, 255, 255, 210), 1.3))
        painter.setBrush(QBrush(self._color if self.isEnabled() else QColor("#555555")))
        painter.drawEllipse(QPointF(cx, cy), r, r)


class ProQKnob(QWidget):
    """
    Perilla de estudio estilo FabFilter Pro-Q 4 con geometría fiel:
    - Capuchón Central 3D: Círculo relleno con gradiente vertical (#2E3138 a #1F2126),
      sombra interior oscura de 2px alrededor del capuchón para separarlo de la pista,
      y línea indicadora radial (tick) corta y nítida cerca del borde del capuchón.
    - Arco de Valor Principal: Pegado al perímetro del capuchón con un grosor limpio
      de 3.5px y puntas rectas (Qt.PenCapStyle.FlatCap).
    - Filas inferiores independientes con altura fija (val_label y title_label).
    """

    valueChanged = pyqtSignal(float)

    def __init__(
        self,
        title: str,
        min_val: float,
        max_val: float,
        default_val: float,
        fmt_func=None,
        log_scale: bool = False,
        bipolar: bool = False,
        accent_color: str = "#00E5FF",
        diameter: int = 58,
        parent=None,
    ):
        super().__init__(parent)
        self.title = title
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.default_val = float(default_val)
        self._value = float(default_val)
        self.fmt_func = fmt_func or (lambda v: f"{v:.1f}")
        self.log_scale = bool(log_scale)
        self.bipolar = bool(bipolar)
        self.accent_color = QColor(accent_color)
        self.diameter = int(diameter)

        self._dragging = False
        self._last_mouse_y = 0.0

        is_compact = self.diameter <= 46
        w = max(54 if is_compact else 74, self.diameter + (10 if is_compact else 16))
        h = self.diameter + 31
        self.setFixedSize(w, h)
        self.setCursor(Qt.CursorShape.SizeVerCursor)
        self.setToolTip(
            f"{self.title}: Arrastrar verticalmente | Shift = Ajuste fino | Rueda = Paso suave | Doble clic = Reset"
        )

        self.val_label = QLabel(self)
        self.val_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.val_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.val_label.setGeometry(0, self.diameter + 2, w, 14)

        self.title_label = QLabel(self.title, self)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.title_label.setGeometry(0, self.diameter + 16, w, 13)

        self._refresh_labels()

    def _refresh_labels(self):
        val_c = "#ECEFF4" if self.isEnabled() else "#686C75"
        tit_c = "#8E95A4" if self.isEnabled() else "#555962"
        self.val_label.setText(self.fmt_func(self._value))
        self.val_label.setStyleSheet(
            f"QLabel {{ color: {val_c}; font-family: 'Segoe UI'; font-size: 10px; font-weight: 700; background: transparent; }}"
        )
        self.title_label.setStyleSheet(
            f"QLabel {{ color: {tit_c}; font-family: 'Segoe UI'; font-size: 9px; font-weight: 600; background: transparent; }}"
        )

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.EnabledChange:
            self._refresh_labels()

    def value(self) -> float:
        return self._value

    def setValue(self, val: float, emit_signal: bool = False):
        clamped = float(np.clip(val, self.min_val, self.max_val))
        if abs(clamped - self._value) > 1e-6:
            self._value = clamped
            self._refresh_labels()
            self.update()
            if emit_signal:
                self.valueChanged.emit(self._value)
        else:
            self._refresh_labels()

    def setAccentColor(self, color_str: str):
        c = QColor(color_str)
        if c.isValid() and c != self.accent_color:
            self.accent_color = c
            self.update()

    def _val_to_norm(self, val: float) -> float:
        v = float(np.clip(val, self.min_val, self.max_val))
        if self.log_scale and self.min_val > 0:
            return float(np.log(v / self.min_val) / np.log(self.max_val / self.min_val))
        return float((v - self.min_val) / (self.max_val - self.min_val))

    def _norm_to_val(self, norm: float) -> float:
        n = float(np.clip(norm, 0.0, 1.0))
        if self.log_scale and self.min_val > 0:
            return float(self.min_val * ((self.max_val / self.min_val) ** n))
        return float(self.min_val + n * (self.max_val - self.min_val))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._last_mouse_y = event.position().y()
            event.accept()
        else:
            event.accept()

    def mouseMoveEvent(self, event):
        if self._dragging:
            y = event.position().y()
            dy = self._last_mouse_y - y
            self._last_mouse_y = y

            fine = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            sensitivity = 950.0 if fine else 165.0
            norm = self._val_to_norm(self._value) + (dy / sensitivity)
            new_val = self._norm_to_val(norm)
            self.setValue(new_val, emit_signal=True)
            event.accept()
        else:
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
        event.accept()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setValue(self.default_val, emit_signal=True)
        event.accept()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta != 0:
            fine = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            step = 0.008 if fine else 0.035
            direction = 1.0 if delta > 0 else -1.0
            norm = self._val_to_norm(self._value) + direction * step
            self.setValue(self._norm_to_val(norm), emit_signal=True)
        event.accept()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        w = float(self.width())
        d = float(self.diameter)
        is_compact = d <= 46.0
        track_w = 3.5

        cx = w * 0.5
        cy = d * 0.5 + 1.0
        r_track = (d - track_w - 2.0) * 0.5
        # Capuchón pegado al arco principal con separación de sombra oscura de 2px
        cap_r = r_track - (track_w * 0.5 + 2.0)

        arc_rect = QRectF(cx - r_track, cy - r_track, 2.0 * r_track, 2.0 * r_track)

        # 1. Pista Exterior (Track Arc): 270° con puntas rectas (FlatCap)
        track_pen = QPen(QColor("#111216"), track_w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap)
        painter.setPen(track_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawArc(arc_rect, 225 * 16, -270 * 16)

        # 2. Arco de Valor Principal pegado al perímetro con grosor 3.5px y FlatCap
        norm = self._val_to_norm(self._value)
        active_color = self.accent_color if self.isEnabled() else QColor("#4A4D57")
        val_pen = QPen(active_color, track_w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap)
        painter.setPen(val_pen)

        if self.bipolar:
            zero_norm = self._val_to_norm(0.0)
            start_deg = 225.0 - zero_norm * 270.0
            span_deg = -(norm - zero_norm) * 270.0
            painter.drawArc(arc_rect, int(round(start_deg * 16)), int(round(span_deg * 16)))
        else:
            span_deg = -norm * 270.0
            painter.drawArc(arc_rect, 225 * 16, int(round(span_deg * 16)))

        # 3. Sombra oscura de 2px alrededor del capuchón para separarlo de la pista
        painter.setPen(QPen(QColor("#0B0C0F"), 2.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(cx, cy), cap_r + 1.0, cap_r + 1.0)

        # 4. Capuchón Central 3D con gradiente vertical (#2E3138 a #1F2126)
        cap_grad = QLinearGradient(cx, cy - cap_r, cx, cy + cap_r)
        if self.isEnabled():
            cap_grad.setColorAt(0.0, QColor("#2E3138"))
            cap_grad.setColorAt(1.0, QColor("#1F2126"))
            bevel_color = QColor("#434750")
        else:
            cap_grad.setColorAt(0.0, QColor("#24262B"))
            cap_grad.setColorAt(1.0, QColor("#1A1B1F"))
            bevel_color = QColor("#32343A")

        painter.setPen(QPen(bevel_color, 1.0))
        painter.setBrush(QBrush(cap_grad))
        painter.drawEllipse(QPointF(cx, cy), cap_r, cap_r)

        # 5. Línea indicadora radial (tick) corta y nítida cerca del borde del capuchón
        angle_deg = 225.0 - norm * 270.0
        angle_rad = math.radians(angle_deg)
        r_in = cap_r * (0.60 if is_compact else 0.64)
        r_out = cap_r * 0.88
        x1 = cx + r_in * math.cos(angle_rad)
        y1 = cy - r_in * math.sin(angle_rad)
        x2 = cx + r_out * math.cos(angle_rad)
        y2 = cy - r_out * math.sin(angle_rad)

        tick_color = QColor("#F8FAFC") if self.isEnabled() else QColor("#686C75")
        painter.setPen(QPen(tick_color, 1.9 if is_compact else 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))


class GainDynProQKnob(ProQKnob):
    """
    Perilla grande de GAIN (diámetro 88px) encastrada en el notch superior del HUD estilo FabFilter Pro-Q 4:
    - Capuchón Central 3D (#2E3138 a #1F2126) con sombra interior de 2px y tick radial corto.
    - Arco de Valor Principal de 3.5px con FlatCap pegado al capuchón.
    - Anillo Exterior de Ganancia Dinámica (Dynamic Range) MUCHO MÁS ANCHO (10px de grosor):
      - Pista visible en rgba(255, 255, 255, 28) cuando está en 0 e iluminada con `band.color` cuando != 0.
      - Pestaña / manija brillante (handle radial) en el extremo del anillo dinámico para agarrarlo fácilmente.
      - Cursor `PointingHandCursor` al hacer hover sobre el anillo exterior.
      - Arrastrar con Clic Derecho o `Alt + Clic Izquierdo` en CUALQUIER parte de la perilla ajusta
        directamente `Dynamic Range` con tooltip instantáneo en vivo (`Dyn Range: -6.0 dB`).
    """

    dynRangeChanged = pyqtSignal(float)

    def __init__(self, diameter: int = 88, parent=None):
        self._dyn_range_db = 0.0
        self._dyn_offset_db = 0.0
        self._dyn_enabled = False
        self._dragging_outer = False
        self._hover_outer = False

        super().__init__(
            title="GAIN",
            min_val=-24.0,
            max_val=24.0,
            default_val=0.0,
            fmt_func=lambda v: f"{v:+.1f} dB",
            log_scale=False,
            bipolar=True,
            accent_color="#00E5FF",
            diameter=diameter,
            parent=parent,
        )
        w = 108
        h = self.diameter + 43  # 88 + 43 = 131px
        self.setFixedSize(w, h)
        self.setMouseTracking(True)

        # Tres filas independientes con altura fija y sin superposición
        self.val_label.setGeometry(0, self.diameter + 1, w, 14)

        self.dyn_label = QLabel("DYN: 0.0 dB", self)
        self.dyn_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.dyn_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.dyn_label.setGeometry(0, self.diameter + 15, w, 14)

        self.title_label.setGeometry(0, self.diameter + 29, w, 13)
        self._refresh_labels()

        self.setToolTip(
            "Capuchón Central (Clic izq): Ganancia Estática (-24 a +24 dB)\n"
            "Anillo Exterior Ancho (o Clic Derecho / Alt + Clic Izq en cualquier parte): Rango Dinámico (-30 a +30 dB)\n"
            "Shift = Ajuste fino | Doble clic = Reset"
        )

    def _refresh_labels(self):
        super()._refresh_labels()
        if not hasattr(self, "dyn_label"):
            return
        if abs(self._dyn_range_db) >= 0.05 or self._dyn_enabled:
            dyn_c = _brighten_color(self.accent_color, 0.20).name()
            dyn_txt = f"DYN: {self._dyn_range_db:+.1f} dB"
        else:
            dyn_c = "#747B8A"
            dyn_txt = "DYN: 0.0 dB"
        self.dyn_label.setText(dyn_txt)
        self.dyn_label.setStyleSheet(
            f"QLabel {{ color: {dyn_c}; font-family: 'Segoe UI'; font-size: 10px; font-weight: 700; background: transparent; }}"
        )

    def setAccentColor(self, color_str: str):
        super().setAccentColor(color_str)
        self._refresh_labels()

    def dynRangeValue(self) -> float:
        return self._dyn_range_db

    def setDynRangeValue(self, val: float, emit_signal: bool = False):
        clamped = float(np.clip(val, -30.0, 30.0))
        if abs(clamped - self._dyn_range_db) > 1e-6:
            self._dyn_range_db = clamped
            self._refresh_labels()
            self.update()
            if emit_signal:
                self.dynRangeChanged.emit(self._dyn_range_db)
        else:
            self._refresh_labels()

    def setDynEnabled(self, enabled: bool):
        if bool(enabled) != self._dyn_enabled:
            self._dyn_enabled = bool(enabled)
            self._refresh_labels()
            self.update()

    def setLiveDynOffset(self, offset_db: float):
        clamped = float(np.clip(offset_db, -30.0, 30.0))
        if abs(clamped - self._dyn_offset_db) > 0.04:
            self._dyn_offset_db = clamped
            self.update()

    def _geometry_radii(self):
        w = float(self.width())
        d = float(self.diameter)
        cx = w * 0.5
        cy = d * 0.5 + 1.0
        outer_w = 10.0                        # Anillo exterior ancho de 10px
        outer_r = (d - outer_w - 2.0) * 0.5   # Radio central del anillo exterior
        track_w = 3.5                         # Arco principal de 3.5px
        track_r = outer_r - (outer_w * 0.5 + track_w * 0.5 + 1.5)
        cap_r = track_r - (track_w * 0.5 + 2.0)
        return cx, cy, outer_r, outer_w, track_r, track_w, cap_r

    def _is_on_outer_handle(self, pos: QPointF) -> bool:
        cx, cy, outer_r, outer_w, _, _, _ = self._geometry_radii()
        dx = pos.x() - cx
        dy = cy - pos.y()  # Invertir ya que el eje Y de Qt crece hacia abajo
        dist = math.hypot(dx, dy)
        if not ((outer_r - outer_w * 0.5 - 4.0) <= dist <= (outer_r + outer_w * 0.5 + 6.0)):
            return False

        angle_deg = math.degrees(math.atan2(dy, dx))
        if angle_deg < 0:
            angle_deg += 360.0

        norm_gain = self._val_to_norm(self._value)
        gain_deg = (225.0 - norm_gain * 270.0) % 360.0
        dyn_span_deg = -(self._dyn_range_db / 30.0) * 135.0
        handle_deg = (gain_deg + dyn_span_deg) % 360.0

        diff = abs(angle_deg - handle_deg)
        if diff > 180.0:
            diff = 360.0 - diff
        return diff <= 22.0

    def leaveEvent(self, event):
        if self._hover_outer:
            self._hover_outer = False
            self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        is_dyn = (
            event.button() == Qt.MouseButton.RightButton
            or (
                event.button() == Qt.MouseButton.LeftButton
                and (
                    bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
                    or self._is_on_outer_handle(event.position())
                )
            )
        )
        if is_dyn:
            self._dragging_outer = True
            self._dragging = False
            self._last_mouse_y = event.position().y()
            QToolTip.showText(
                event.globalPosition().toPoint(),
                f"Dyn Range: {self._dyn_range_db:+.1f} dB",
                self,
            )
            event.accept()
        elif event.button() == Qt.MouseButton.LeftButton:
            self._dragging_outer = False
            super().mousePressEvent(event)
        else:
            event.accept()

    def mouseMoveEvent(self, event):
        if self._dragging_outer:
            y = event.position().y()
            dy = self._last_mouse_y - y
            self._last_mouse_y = y
            fine = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            step = (dy / 900.0 if fine else dy / 150.0) * 60.0
            self.setDynRangeValue(self._dyn_range_db + step, emit_signal=True)
            QToolTip.showText(
                event.globalPosition().toPoint(),
                f"Dyn Range: {self._dyn_range_db:+.1f} dB",
                self,
            )
            event.accept()
        elif self._dragging:
            super().mouseMoveEvent(event)
        else:
            on_handle = self._is_on_outer_handle(event.position())
            if on_handle != self._hover_outer:
                self._hover_outer = on_handle
                self.setCursor(
                    Qt.CursorShape.PointingHandCursor if on_handle else Qt.CursorShape.SizeVerCursor
                )
                self.update()
            event.accept()

    def mouseReleaseEvent(self, event):
        if self._dragging_outer:
            self._dragging_outer = False
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton or self._is_on_outer_handle(event.position()):
            self.setDynRangeValue(0.0, emit_signal=True)
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event):
        if bool(event.modifiers() & Qt.KeyboardModifier.AltModifier) or self._is_on_outer_handle(event.position()):
            delta = event.angleDelta().y()
            if delta != 0:
                fine = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
                step = 0.25 if fine else 1.0
                direction = 1.0 if delta > 0 else -1.0
                self.setDynRangeValue(self._dyn_range_db + direction * step, emit_signal=True)
                QToolTip.showText(
                    event.globalPosition().toPoint(),
                    f"Dyn Range: {self._dyn_range_db:+.1f} dB",
                    self,
                )
                event.accept()
                return
        super().wheelEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        cx, cy, outer_r, outer_w, track_r, track_w, cap_r = self._geometry_radii()

        outer_rect = QRectF(cx - outer_r, cy - outer_r, 2.0 * outer_r, 2.0 * outer_r)
        track_rect = QRectF(cx - track_r, cy - track_r, 2.0 * track_r, 2.0 * track_r)

        # 1. Pista Exterior Ancha (10px) de Ganancia Dinámica claramente visible (rgba(255, 255, 255, 28))
        ring_bg_alpha = 45 if self._hover_outer else 28
        painter.setPen(QPen(QColor(255, 255, 255, ring_bg_alpha), outer_w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawArc(outer_rect, 225 * 16, -270 * 16)

        # 2. Pista Principal de Ganancia Estática (3.5px, FlatCap)
        painter.setPen(QPen(QColor("#111216"), track_w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap))
        painter.drawArc(track_rect, 225 * 16, -270 * 16)

        # 3. Arco de Valor Activo de Ganancia Estática (3.5px, FlatCap, pegado al capuchón)
        norm_gain = self._val_to_norm(self._value)
        gain_deg = 225.0 - norm_gain * 270.0
        zero_deg = 90.0
        span_gain_deg = gain_deg - zero_deg

        band_c = self.accent_color if self.isEnabled() else QColor("#4A4D57")
        painter.setPen(QPen(band_c, track_w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap))
        painter.drawArc(track_rect, int(round(zero_deg * 16)), int(round(span_gain_deg * 16)))

        # 4. Arco del Límite de Rango Dinámico Configurado en el anillo ancho de 10px
        dyn_span_deg = -(self._dyn_range_db / 30.0) * 135.0
        if abs(self._dyn_range_db) >= 0.05:
            lim_color = QColor(self.accent_color)
            lim_color.setAlpha(185 if self._dyn_enabled else 105)
            painter.setPen(QPen(lim_color, outer_w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap))
            painter.drawArc(outer_rect, int(round(gain_deg * 16)), int(round(dyn_span_deg * 16)))

        # 5. Excursión Dinámica Instantánea en Vivo
        if self._dyn_enabled and abs(self._dyn_offset_db) > 0.04:
            inst_span_deg = -(self._dyn_offset_db / 30.0) * 135.0
            bright_c = _brighten_color(self.accent_color, 0.35)
            bright_c.setAlpha(230)
            painter.setPen(QPen(bright_c, outer_w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap))
            painter.drawArc(outer_rect, int(round(gain_deg * 16)), int(round(inst_span_deg * 16)))

        # 6. Pestaña / Manija Brillante (Handle Radial) en el extremo del anillo dinámico
        handle_deg = gain_deg + dyn_span_deg
        handle_rad = math.radians(handle_deg)
        hr_in = outer_r - outer_w * 0.5 - 0.5
        hr_out = outer_r + outer_w * 0.5 + 2.2
        hx1 = cx + hr_in * math.cos(handle_rad)
        hy1 = cy - hr_in * math.sin(handle_rad)
        hx2 = cx + hr_out * math.cos(handle_rad)
        hy2 = cy - hr_out * math.sin(handle_rad)

        handle_color = QColor("#FFFFFF") if (self._hover_outer or self._dragging_outer or abs(self._dyn_range_db) >= 0.05) else _brighten_color(self.accent_color, 0.45)
        painter.setPen(QPen(QColor("#0B0C0F"), 4.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(QPointF(hx1, hy1), QPointF(hx2, hy2))
        painter.setPen(QPen(handle_color, 2.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(QPointF(hx1, hy1), QPointF(hx2, hy2))

        # 7. Sombra interior oscura de 2px alrededor del capuchón
        painter.setPen(QPen(QColor("#0B0C0F"), 2.2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(cx, cy), cap_r + 1.0, cap_r + 1.0)

        # 8. Capuchón Central 3D (#2E3138 a #1F2126)
        cap_grad = QLinearGradient(cx, cy - cap_r, cx, cy + cap_r)
        if self.isEnabled():
            cap_grad.setColorAt(0.0, QColor("#2E3138"))
            cap_grad.setColorAt(1.0, QColor("#1F2126"))
            bevel_color = QColor("#454852")
        else:
            cap_grad.setColorAt(0.0, QColor("#25262B"))
            cap_grad.setColorAt(1.0, QColor("#1B1C20"))
            bevel_color = QColor("#32343A")

        painter.setPen(QPen(bevel_color, 1.0))
        painter.setBrush(QBrush(cap_grad))
        painter.drawEllipse(QPointF(cx, cy), cap_r, cap_r)

        # 9. Línea indicadora radial (tick) corta y nítida cerca del borde del capuchón
        angle_rad = math.radians(gain_deg)
        r_in = cap_r * 0.64
        r_out = cap_r * 0.88
        x1 = cx + r_in * math.cos(angle_rad)
        y1 = cy - r_in * math.sin(angle_rad)
        x2 = cx + r_out * math.cos(angle_rad)
        y2 = cy - r_out * math.sin(angle_rad)
        painter.setPen(QPen(QColor("#FFFFFF") if self.isEnabled() else QColor("#686C75"), 2.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))


class DynThresholdMeter(QWidget):
    """
    Slider vertical de Threshold (`THR`) de 68px de alto de barra con un handle triangular grande
    y fácil de arrastrar (`14px` de ancho), ubicado a la derecha inmediata de `GAIN`.
    """

    thresholdChanged = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.db_min = -70.0
        self.db_max = 0.0
        self._threshold_db = -24.0
        self._level_db = -70.0
        self._accent_color = QColor("#00E5FF")
        self._dragging = False
        self.setFixedSize(44, 102)
        self.setCursor(Qt.CursorShape.SizeVerCursor)
        self.setToolTip(
            "Umbral Dinámico (Threshold): Arrastrar el triángulo verticalmente (-70 a 0 dB)\n"
            "Doble clic = Reset a -24 dB"
        )

    def setAccentColor(self, color_str: str):
        c = QColor(color_str)
        if c.isValid() and c != self._accent_color:
            self._accent_color = c
            self.update()

    def thresholdValue(self) -> float:
        return self._threshold_db

    def setThresholdValue(self, db: float, emit_signal: bool = False):
        clamped = float(np.clip(db, self.db_min, self.db_max))
        if abs(clamped - self._threshold_db) > 1e-5:
            self._threshold_db = clamped
            self.update()
            if emit_signal:
                self.thresholdChanged.emit(self._threshold_db)

    def setLiveLevel(self, env_db: float):
        clamped = float(np.clip(env_db, self.db_min, self.db_max))
        if abs(clamped - self._level_db) > 0.15:
            self._level_db = clamped
            self.update()

    def _db_to_y(self, db: float, top_y: float, bar_h: float) -> float:
        ratio = float(np.clip((db - self.db_min) / (self.db_max - self.db_min), 0.0, 1.0))
        return top_y + (1.0 - ratio) * bar_h

    def _y_to_db(self, y: float, top_y: float, bar_h: float) -> float:
        ratio = float(np.clip(1.0 - (y - top_y) / max(1.0, bar_h), 0.0, 1.0))
        return self.db_min + ratio * (self.db_max - self.db_min)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._update_from_mouse_y(event.position().y())
        event.accept()

    def mouseMoveEvent(self, event):
        if self._dragging:
            self._update_from_mouse_y(event.position().y())
        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
        event.accept()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setThresholdValue(-24.0, emit_signal=True)
        event.accept()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta != 0:
            step = 0.5 if (event.modifiers() & Qt.KeyboardModifier.ShiftModifier) else 1.5
            direction = 1.0 if delta > 0 else -1.0
            self.setThresholdValue(self._threshold_db + direction * step, emit_signal=True)
        event.accept()

    def _update_from_mouse_y(self, y: float):
        top_y = 3.0
        bar_h = 66.0
        self.setThresholdValue(self._y_to_db(y, top_y, bar_h), emit_signal=True)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        w = float(self.width())
        top_y = 3.0
        bar_h = 66.0
        bar_w = 8.0
        bar_x = 10.0

        painter.setPen(QPen(QColor("#2E3138"), 1.0))
        painter.setBrush(QBrush(QColor("#111215")))
        painter.drawRoundedRect(QRectF(bar_x, top_y, bar_w, bar_h), 2.5, 2.5)

        if self.isEnabled() and self._level_db > self.db_min + 0.5:
            y_lvl = self._db_to_y(self._level_db, top_y, bar_h)
            fill_h = (top_y + bar_h) - y_lvl
            grad = QLinearGradient(0, top_y + bar_h, 0, top_y)
            c_base = QColor(self._accent_color)
            c_bright = _brighten_color(self._accent_color, 0.35)
            grad.setColorAt(0.0, c_base)
            grad.setColorAt(1.0, c_bright)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(grad))
            painter.drawRoundedRect(QRectF(bar_x + 1.0, y_lvl, bar_w - 2.0, fill_h), 1.5, 1.5)

        y_thr = self._db_to_y(self._threshold_db, top_y, bar_h)
        tri_color = self._accent_color if self.isEnabled() else QColor("#5A5E68")

        painter.setPen(QPen(tri_color, 1.6))
        painter.drawLine(QPointF(bar_x - 1.5, y_thr), QPointF(bar_x + bar_w + 2.0, y_thr))

        handle_x0 = bar_x + bar_w + 2.0
        handle_w = 14.0
        tri = QPolygonF([
            QPointF(handle_x0, y_thr),
            QPointF(handle_x0 + handle_w, y_thr - 5.5),
            QPointF(handle_x0 + handle_w, y_thr + 5.5),
        ])
        painter.setPen(QPen(QColor("#111215"), 1.0))
        painter.setBrush(QBrush(tri_color))
        painter.drawPolygon(tri)

        font_val = QFont("Segoe UI", 8, QFont.Weight.Bold)
        painter.setFont(font_val)
        painter.setPen(QColor("#ECEFF4") if self.isEnabled() else QColor("#686C75"))
        painter.drawText(QRectF(0, top_y + bar_h + 2.0, w, 13.0), Qt.AlignmentFlag.AlignCenter, f"{self._threshold_db:.0f} dB")

        font_title = QFont("Segoe UI", 7, QFont.Weight.DemiBold)
        painter.setFont(font_title)
        painter.setPen(QColor("#8E95A4") if self.isEnabled() else QColor("#555962"))
        painter.drawText(QRectF(0, top_y + bar_h + 15.0, w, 12.0), Qt.AlignmentFlag.AlignCenter, "THR")


class ProQ4Overlay(QWidget):
    """
    Capa visual interactiva estilo FabFilter Pro-Q 4 Full-Bleed:
    - Oculta los ejes de pyqtgraph para que las curvas lleguen hasta x = 0 y el borde superior/inferior.
    - Soporta rango de ganancia visual conmutable (`±12 dB` o `±30 dB`) alineado con las líneas de grilla.
    - Todas las curvas dinámicas, sombreados y anillos usan el color propio de cada banda (`band.color`).
    """

    band_selected = pyqtSignal(int)
    band_modified = pyqtSignal()
    eq_range_changed = pyqtSignal(float)

    def __init__(self, plot_widget: pg.PlotWidget, eq_engine: ParametricEQEngine, parent=None):
        super().__init__(parent)
        self.plot_widget = plot_widget
        self._viewport = plot_widget.viewport()
        self.plot_item = plot_widget.getPlotItem()
        self.eq_engine = eq_engine

        self.is_x_log = True
        self.nyquist = eq_engine.sr / 2.0
        self.eq_range_db = 12.0  # Conmutable entre 12.0 dB y 30.0 dB

        self.selected_band_idx = 0 if eq_engine.bands else -1
        self.hovered_band_idx = -1
        self.dragging_band_idx = -1
        self._drag_start_q = 1.0
        self._drag_start_scene_y = 0.0

        self._last_rendered_version = -1
        self._was_dyn_moving = False

        self._rebuild_eval_freqs()
        self._init_secondary_viewbox()
        self._init_curves_and_nodes()

        self._viewport.installEventFilter(self)
        self._viewport.setMouseTracking(True)

    def _effective_eq_vb_limit(self) -> float:
        """
        Dado que el ViewBox principal va de +12 dBFS (arriba, y=0) a -132 dBFS (abajo, y=h)
        con centro en -60 dBFS, escalamos eq_vb por 144/120 = 1.20 para que:
        +eq_range_db coincida exactamente con 0 dBFS y -eq_range_db coincida con -120 dBFS,
        dejando que las curvas de corte (Low/High Cut) bajen hasta -132 dBFS (-infty).
        """
        return self.eq_range_db * 1.20

    def set_eq_range_db(self, range_db: float):
        new_r = 30.0 if float(range_db) >= 20.0 else 12.0
        if abs(new_r - self.eq_range_db) > 1e-3:
            self.eq_range_db = new_r
            lim = self._effective_eq_vb_limit()
            self.eq_vb.setYRange(-lim, lim, padding=0.0)
            self.update_eq_curves(force=True)
            self.eq_range_changed.emit(self.eq_range_db)

    def toggle_eq_range_db(self):
        self.set_eq_range_db(30.0 if self.eq_range_db <= 12.0 else 12.0)

    def _rebuild_eval_freqs(self):
        self.nyquist = max(1000.0, self.eq_engine.sr / 2.0)
        n_pts = 320
        if self.is_x_log:
            self.eval_freqs = np.logspace(np.log10(10.0), np.log10(self.nyquist), n_pts)
            self.eval_x = np.log10(self.eval_freqs)
        else:
            self.eval_freqs = np.linspace(10.0, self.nyquist, n_pts)
            self.eval_x = self.eval_freqs.copy()

        w = 2.0 * np.pi * self.eval_freqs / self.eq_engine.sr
        self.eval_z1 = np.exp(-1j * w)
        self.eval_z2 = self.eval_z1 * self.eval_z1
        with self.eq_engine.lock:
            for b in self.eq_engine.bands:
                b.invalidate_cache()
            self.eq_engine.mark_dirty()

    def _init_secondary_viewbox(self):
        # Ocultamos el eje derecho nativo de pyqtgraph porque las dos escalas (FFT + EQ)
        # se dibujan condensadas y alineadas en el margen derecho del lienzo Full-Bleed.
        self.plot_item.hideAxis("right")
        self.plot_item.hideAxis("left")
        self.plot_item.hideAxis("top")
        self.plot_item.hideAxis("bottom")
        self.plot_item.layout.setContentsMargins(0, 0, 0, 0)

        self.eq_vb = pg.ViewBox(enableMenu=False)
        self.eq_vb.setZValue(10)
        self.eq_vb.setMouseEnabled(x=False, y=False)
        self.plot_item.scene().addItem(self.eq_vb)
        self.eq_vb.setXLink(self.plot_item.vb)
        lim = self._effective_eq_vb_limit()
        self.eq_vb.setYRange(-lim, lim, padding=0.0)

        self.plot_item.vb.sigResized.connect(self._update_vb_geometry)
        self._update_vb_geometry()

    def _update_vb_geometry(self):
        self.eq_vb.setGeometry(self.plot_item.vb.sceneBoundingRect())
        self.eq_vb.linkedViewChanged(self.plot_item.vb, self.eq_vb.XAxis)

    def _init_curves_and_nodes(self):
        self.zero_line = pg.PlotDataItem(
            pen=pg.mkPen(color=(255, 255, 255, 42), width=1.0)
        )
        self.zero_line.setZValue(5)
        self.eq_vb.addItem(self.zero_line)

        self.band_fill_curves: list[pg.PlotDataItem] = []
        self.band_dyn_curves: list[pg.PlotDataItem] = []
        self.band_dyn_fills: list[pg.FillBetweenItem] = []

        self.dyn_limit_curve = pg.PlotDataItem(
            pen=pg.mkPen(color=(0, 229, 255, 175), width=1.4, style=Qt.PenStyle.DotLine)
        )
        self.dyn_limit_curve.setZValue(18)
        self.eq_vb.addItem(self.dyn_limit_curve)

        self.static_master_curve = pg.PlotDataItem(
            pen=pg.mkPen(color="#FFD54F", width=2.2)
        )
        self.static_master_curve.setZValue(20)
        self.eq_vb.addItem(self.static_master_curve)

        self.dyn_nodes_scatter = pg.ScatterPlotItem(size=22, pxMode=True)
        self.dyn_nodes_scatter.setZValue(35)
        self.eq_vb.addItem(self.dyn_nodes_scatter)

        self.nodes_scatter = pg.ScatterPlotItem(size=18, pxMode=True)
        self.nodes_scatter.setZValue(40)
        self.eq_vb.addItem(self.nodes_scatter)

        self.node_labels: list[pg.TextItem] = []

        self.floating_tooltip = pg.TextItem(anchor=(0.5, 1.35), color="#FFFFFF")
        self.floating_tooltip.setZValue(100)
        self.eq_vb.addItem(self.floating_tooltip)
        self.floating_tooltip.hide()

    def set_axes_mode(self, is_x_log: bool, sr: float):
        self.is_x_log = bool(is_x_log)
        self.eq_engine.sr = float(sr)
        self._rebuild_eval_freqs()
        lim = self._effective_eq_vb_limit()
        self.eq_vb.setYRange(-lim, lim, padding=0.0)
        self._update_vb_geometry()
        self.update_eq_curves(force=True)

    def freq_to_x(self, freq: float) -> float:
        f = float(np.clip(freq, 10.0, self.nyquist))
        return float(np.log10(f)) if self.is_x_log else f

    def x_to_freq(self, x_val: float) -> float:
        if self.is_x_log:
            f = 10.0 ** float(x_val)
        else:
            f = float(x_val)
        return float(np.clip(f, 10.0, self.nyquist * 0.99))

    def _band_node_y(self, band: EQBand) -> float:
        lim = self.eq_range_db
        return float(np.clip(band.gain_db, -lim, lim)) if band.supports_gain else 0.0

    def _find_band_near_scene_pos(self, scene_pos: QPointF, max_dist_px: float = 18.0) -> int:
        best_idx = -1
        best_dist = max_dist_px
        with self.eq_engine.lock:
            for idx, band in enumerate(self.eq_engine.bands):
                vx = self.freq_to_x(band.freq)
                vy = self._band_node_y(band)
                node_scene = self.eq_vb.mapViewToScene(QPointF(vx, vy))
                dist = float(np.hypot(node_scene.x() - scene_pos.x(), node_scene.y() - scene_pos.y()))
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
        return best_idx

    def eventFilter(self, obj, event):
        if obj is not self._viewport:
            return super().eventFilter(obj, event)

        etype = event.type()

        if etype == QEvent.Type.MouseButtonDblClick:
            if event.button() == Qt.MouseButton.LeftButton:
                scene_pos = self.plot_widget.mapToScene(event.position().toPoint())
                if not self.plot_item.vb.sceneBoundingRect().contains(scene_pos):
                    return False

                hit_idx = self._find_band_near_scene_pos(scene_pos)
                if hit_idx >= 0:
                    with self.eq_engine.lock:
                        band = self.eq_engine.bands[hit_idx]
                        band.gain_db = 0.0
                        band.update_coefficients()
                        self.eq_engine.mark_dirty()
                    self.selected_band_idx = hit_idx
                    self.band_selected.emit(hit_idx)
                    self.band_modified.emit()
                    self.update_eq_curves(force=True)
                    return True
                else:
                    view_pt = self.eq_vb.mapSceneToView(scene_pos)
                    freq = self.x_to_freq(view_pt.x())
                    gain_db = float(np.clip(view_pt.y(), -24.0, 24.0))
                    self.eq_engine.add_band(freq=freq, gain_db=gain_db, q=1.2)
                    new_idx = len(self.eq_engine.bands) - 1
                    self.selected_band_idx = new_idx
                    self.dragging_band_idx = new_idx
                    self.band_selected.emit(new_idx)
                    self.band_modified.emit()
                    self.update_eq_curves(force=True)
                    return True

        elif etype == QEvent.Type.MouseButtonPress:
            scene_pos = self.plot_widget.mapToScene(event.position().toPoint())
            if not self.plot_item.vb.sceneBoundingRect().contains(scene_pos):
                return False

            hit_idx = self._find_band_near_scene_pos(scene_pos)
            if event.button() == Qt.MouseButton.LeftButton:
                if hit_idx >= 0:
                    self.selected_band_idx = hit_idx
                    self.dragging_band_idx = hit_idx
                    with self.eq_engine.lock:
                        self._drag_start_q = self.eq_engine.bands[hit_idx].q
                    self._drag_start_scene_y = scene_pos.y()
                    self.band_selected.emit(hit_idx)
                    self.update_eq_curves(force=True)
                    return True
            elif event.button() == Qt.MouseButton.RightButton:
                if hit_idx >= 0:
                    self.selected_band_idx = hit_idx
                    self.band_selected.emit(hit_idx)
                    self._show_node_context_menu(hit_idx, event.globalPosition().toPoint())
                    return True

        elif etype == QEvent.Type.MouseMove:
            scene_pos = self.plot_widget.mapToScene(event.position().toPoint())
            if self.dragging_band_idx >= 0:
                view_pt = self.eq_vb.mapSceneToView(scene_pos)
                new_freq = self.x_to_freq(view_pt.x())
                new_gain = float(np.clip(view_pt.y(), -24.0, 24.0))

                with self.eq_engine.lock:
                    if 0 <= self.dragging_band_idx < len(self.eq_engine.bands):
                        band = self.eq_engine.bands[self.dragging_band_idx]
                        band.freq = new_freq
                        if band.supports_gain:
                            band.gain_db = new_gain
                        else:
                            dy = self._drag_start_scene_y - scene_pos.y()
                            band.q = float(np.clip(self._drag_start_q * (1.015 ** dy), 0.1, 24.0))
                        band.update_coefficients()
                        self.eq_engine.mark_dirty()

                self.band_modified.emit()
                self.update_eq_curves(force=True)
                return True
            else:
                hit_idx = self._find_band_near_scene_pos(scene_pos)
                if hit_idx != self.hovered_band_idx:
                    self.hovered_band_idx = hit_idx
                    if hit_idx >= 0:
                        self._viewport.setCursor(Qt.CursorShape.SizeAllCursor)
                    else:
                        self._viewport.setCursor(Qt.CursorShape.CrossCursor)
                    self.update_eq_curves(force=True)

        elif etype == QEvent.Type.MouseButtonRelease:
            if event.button() == Qt.MouseButton.LeftButton and self.dragging_band_idx >= 0:
                self.dragging_band_idx = -1
                self.update_eq_curves(force=True)
                return True

        elif etype == QEvent.Type.Wheel:
            scene_pos = self.plot_widget.mapToScene(event.position().toPoint())
            if self.plot_item.vb.sceneBoundingRect().contains(scene_pos):
                target_idx = self._find_band_near_scene_pos(scene_pos, max_dist_px=28.0)
                if target_idx < 0:
                    target_idx = self.selected_band_idx
                if 0 <= target_idx < len(self.eq_engine.bands):
                    delta = event.angleDelta().y()
                    if delta != 0:
                        factor = 1.12 if delta > 0 else (1.0 / 1.12)
                        with self.eq_engine.lock:
                            band = self.eq_engine.bands[target_idx]
                            band.q = float(np.clip(band.q * factor, 0.1, 24.0))
                            band.update_coefficients()
                            self.eq_engine.mark_dirty()
                        self.selected_band_idx = target_idx
                        self.band_selected.emit(target_idx)
                        self.band_modified.emit()
                        self.update_eq_curves(force=True)
                        return True

        return super().eventFilter(obj, event)

    def _show_node_context_menu(self, band_idx: int, global_pos):
        with self.eq_engine.lock:
            if not (0 <= band_idx < len(self.eq_engine.bands)):
                return
            band = self.eq_engine.bands[band_idx]

        menu = QMenu(self.plot_widget)
        menu.setStyleSheet("""
            QMenu {
                background-color: #1C1C1C;
                color: #ECEFF4;
                border: 1px solid #383838;
                padding: 4px;
                font-size: 11px;
            }
            QMenu::item { padding: 5px 16px; border-radius: 3px; }
            QMenu::item:selected { background-color: #2E3440; color: #FFFFFF; }
        """)

        header = menu.addAction(f"Band {band_idx + 1} ({band.filter_type} @ {band.freq:.0f} Hz)")
        header.setEnabled(False)
        menu.addSeparator()

        act_enable = menu.addAction("✔ Banda Activa")
        act_enable.setCheckable(True)
        act_enable.setChecked(band.enabled)
        act_enable.triggered.connect(lambda checked: self._toggle_band_enabled(band_idx, checked))

        act_dyn = menu.addAction("⚡ Activar Dynamic EQ")
        act_dyn.setCheckable(True)
        act_dyn.setChecked(band.dyn_enabled)
        act_dyn.triggered.connect(lambda checked: self._toggle_band_dyn(band_idx, checked))

        act_solo = menu.addAction("🎧 Solo Banda")
        act_solo.setCheckable(True)
        act_solo.setChecked(band.solo)
        act_solo.triggered.connect(lambda checked: self._toggle_band_solo(band_idx, checked))

        menu.addSeparator()
        type_menu = menu.addMenu("Tipo de Filtro")
        for ft in FILTER_TYPES:
            act_ft = type_menu.addAction(ft)
            act_ft.setCheckable(True)
            act_ft.setChecked(band.filter_type == ft)
            act_ft.triggered.connect(lambda _=False, t=ft: self._set_band_type(band_idx, t))

        if band.supports_slope:
            slope_menu = menu.addMenu("Pendiente (Slope)")
            for s in SLOPE_OPTIONS:
                act_s = slope_menu.addAction(f"{s} dB/oct")
                act_s.setCheckable(True)
                act_s.setChecked(band.slope == s)
                act_s.triggered.connect(lambda _=False, val=s: self._set_band_slope(band_idx, val))

        color_menu = menu.addMenu("🎨 Color de Banda")
        for label, hex_val in STUDIO_BAND_COLORS:
            act_c = color_menu.addAction(_make_color_icon(hex_val, 12), label)
            act_c.triggered.connect(lambda _=False, hv=hex_val: self._set_band_color(band_idx, hv))
        color_menu.addSeparator()
        act_custom_c = color_menu.addAction("Personalizado...")
        act_custom_c.triggered.connect(lambda: self._pick_custom_band_color(band_idx))

        menu.addSeparator()
        act_del = menu.addAction("🗑 Eliminar Banda")
        act_del.triggered.connect(lambda: self._delete_band(band_idx))

        menu.exec(global_pos)

    def _set_band_slope(self, idx: int, slope: int):
        with self.eq_engine.lock:
            if 0 <= idx < len(self.eq_engine.bands):
                b = self.eq_engine.bands[idx]
                b.slope = int(slope)
                b.update_coefficients()
                self.eq_engine.mark_dirty()
        self.band_modified.emit()
        self.update_eq_curves(force=True)

    def _set_band_color(self, idx: int, hex_color: str):
        with self.eq_engine.lock:
            if 0 <= idx < len(self.eq_engine.bands):
                self.eq_engine.bands[idx].color = QColor(hex_color).name()
                self.eq_engine.mark_dirty()
        self.band_modified.emit()
        self.update_eq_curves(force=True)

    def _pick_custom_band_color(self, idx: int):
        with self.eq_engine.lock:
            if not (0 <= idx < len(self.eq_engine.bands)):
                return
            orig = self.eq_engine.bands[idx].color
        dlg = QColorDialog(QColor(orig), self.plot_widget)
        dlg.setWindowTitle(f"Color de Band {idx + 1}")
        dlg.setOption(QColorDialog.ColorDialogOption.DontUseNativeDialog, True)
        dlg.currentColorChanged.connect(lambda c: self._set_band_color(idx, c.name()) if c.isValid() else None)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._set_band_color(idx, dlg.selectedColor().name())
        else:
            self._set_band_color(idx, orig)

    def _toggle_band_enabled(self, idx: int, checked: bool):
        with self.eq_engine.lock:
            if 0 <= idx < len(self.eq_engine.bands):
                self.eq_engine.bands[idx].enabled = checked
                self.eq_engine.mark_dirty()
        self.band_modified.emit()
        self.update_eq_curves(force=True)

    def _toggle_band_dyn(self, idx: int, checked: bool):
        with self.eq_engine.lock:
            if 0 <= idx < len(self.eq_engine.bands):
                b = self.eq_engine.bands[idx]
                b.dyn_enabled = checked
                if checked and abs(b.dyn_range_db) < 0.1:
                    b.dyn_range_db = -6.0
                b.update_coefficients()
                self.eq_engine.mark_dirty()
        self.band_modified.emit()
        self.update_eq_curves(force=True)

    def _toggle_band_solo(self, idx: int, checked: bool):
        with self.eq_engine.lock:
            if 0 <= idx < len(self.eq_engine.bands):
                self.eq_engine.bands[idx].solo = checked
                self.eq_engine.mark_dirty()
        self.band_modified.emit()
        self.update_eq_curves(force=True)

    def _set_band_type(self, idx: int, filter_type: str):
        with self.eq_engine.lock:
            if 0 <= idx < len(self.eq_engine.bands):
                b = self.eq_engine.bands[idx]
                b.filter_type = filter_type
                b.update_coefficients()
                self.eq_engine.mark_dirty()
        self.band_modified.emit()
        self.update_eq_curves(force=True)

    def _delete_band(self, idx: int):
        self.eq_engine.remove_band(idx)
        if self.selected_band_idx >= len(self.eq_engine.bands):
            self.selected_band_idx = len(self.eq_engine.bands) - 1
        self.band_selected.emit(self.selected_band_idx)
        self.band_modified.emit()
        self.update_eq_curves(force=True)

    def update_eq_curves(self, force: bool = False):
        """
        Actualiza las curvas estáticas y dinámicas por banda permitiendo que las curvas de corte
        (Low Cut / High Cut) caigan hasta el fondo real del ViewBox (-infty) sin cortes abruptos.
        """
        with self.eq_engine.lock:
            bands = list(self.eq_engine.bands)
            cur_version = self.eq_engine.static_version
            any_dyn_moving = any(
                b.enabled and b.is_dynamic_active and abs(b.current_dyn_offset_db) > 0.02
                for b in bands
            )

            if not force and cur_version == self._last_rendered_version and not any_dyn_moving and not self._was_dyn_moving:
                return

            self._last_rendered_version = cur_version
            self._was_dyn_moving = any_dyn_moving

            x = self.eval_x
            z1 = self.eval_z1
            z2 = self.eval_z2
            clip_lim = max(36.0, self._effective_eq_vb_limit() * 1.15)

            self.zero_line.setData([x[0], x[-1]], [0.0, 0.0])

            total_static_db, _ = self.eq_engine.get_combined_responses_db(z1, z2)

            while len(self.band_fill_curves) < len(bands):
                st_item = pg.PlotDataItem(fillLevel=0.0)
                st_item.setZValue(10)
                self.eq_vb.addItem(st_item)
                self.band_fill_curves.append(st_item)

                dyn_item = pg.PlotDataItem()
                dyn_item.setZValue(22)
                self.eq_vb.addItem(dyn_item)
                self.band_dyn_curves.append(dyn_item)

                fill_between = pg.FillBetweenItem(st_item, dyn_item, brush=pg.mkBrush(0, 0, 0, 0))
                fill_between.setZValue(19)
                self.eq_vb.addItem(fill_between)
                self.band_dyn_fills.append(fill_between)

            while len(self.band_fill_curves) > len(bands):
                fill_b = self.band_dyn_fills.pop()
                self.eq_vb.removeItem(fill_b)
                dyn_c = self.band_dyn_curves.pop()
                self.eq_vb.removeItem(dyn_c)
                st_c = self.band_fill_curves.pop()
                self.eq_vb.removeItem(st_c)

            for idx, b in enumerate(bands):
                st_curve = self.band_fill_curves[idx]
                dyn_curve = self.band_dyn_curves[idx]
                dyn_fill = self.band_dyn_fills[idx]

                is_sel = (idx == self.selected_band_idx)
                if not b.enabled and not is_sel:
                    st_curve.setData([], [])
                    dyn_curve.setData([], [])
                    dyn_fill.setBrush(pg.mkBrush(0, 0, 0, 0))
                    continue

                resp_static = np.clip(b.get_static_freq_response_db(z1, z2), -clip_lim, clip_lim)
                c = QColor(b.color)

                if is_sel or abs(b.gain_db) > 0.08 or not b.supports_gain or b.is_dynamic_active:
                    pen_alpha = 210 if is_sel else (120 if b.enabled else 50)
                    fill_alpha = 48 if is_sel else (22 if b.enabled else 10)
                    st_curve.setPen(pg.mkPen(color=(c.red(), c.green(), c.blue(), pen_alpha), width=1.5 if is_sel else 1.0))
                    st_curve.setBrush(pg.mkBrush(c.red(), c.green(), c.blue(), fill_alpha))
                    st_curve.setData(x, resp_static)
                else:
                    st_curve.setData([], [])

                if b.enabled and b.is_dynamic_active and abs(b.current_dyn_offset_db) > 0.02 and not self.eq_engine.bypass:
                    resp_dyn = np.clip(b.get_dynamic_freq_response_db(z1, z2), -clip_lim, clip_lim)
                    bright_c = _brighten_color(c, 0.25)
                    dyn_curve.setPen(pg.mkPen(color=bright_c, width=2.0))
                    dyn_curve.setData(x, resp_dyn)
                    dyn_fill.setBrush(pg.mkBrush(c.red(), c.green(), c.blue(), 75))
                else:
                    dyn_curve.setData([], [])
                    dyn_fill.setBrush(pg.mkBrush(0, 0, 0, 0))

            if 0 <= self.selected_band_idx < len(bands):
                sel_band = bands[self.selected_band_idx]
                if sel_band.is_dynamic_active:
                    sel_c = _brighten_color(sel_band.color, 0.15)
                    sel_c.setAlpha(185)
                    self.dyn_limit_curve.setPen(pg.mkPen(color=sel_c, width=1.4, style=Qt.PenStyle.DotLine))
                    max_dyn_resp = sel_band.get_max_dynamic_freq_response_db(z1, z2)
                    self.dyn_limit_curve.setData(x, np.clip(max_dyn_resp, -clip_lim, clip_lim))
                else:
                    self.dyn_limit_curve.setData([], [])
            else:
                self.dyn_limit_curve.setData([], [])

            static_clipped = np.clip(total_static_db, -clip_lim, clip_lim)
            self.static_master_curve.setData(x, static_clipped)

            node_spots = []
            dyn_spots = []

            while len(self.node_labels) < len(bands):
                lbl = pg.TextItem(anchor=(0.5, 0.5), color="#0B0D12")
                font = QFont("Segoe UI", 8, QFont.Weight.Bold)
                lbl.setFont(font)
                lbl.setZValue(50)
                self.eq_vb.addItem(lbl)
                self.node_labels.append(lbl)
            while len(self.node_labels) > len(bands):
                lbl = self.node_labels.pop()
                self.eq_vb.removeItem(lbl)

            for idx, band in enumerate(bands):
                nx = self.freq_to_x(band.freq)
                ny = self._band_node_y(band)
                c = QColor(band.color)
                if not band.enabled:
                    c = QColor("#555D6E")

                is_sel = (idx == self.selected_band_idx)
                is_hov = (idx == self.hovered_band_idx)
                size = 20 if (is_sel or is_hov) else 16
                border_pen = pg.mkPen("#FFFFFF", width=2.2) if is_sel else pg.mkPen("#12141C", width=1.3)

                node_spots.append({
                    "pos": (nx, ny),
                    "size": size,
                    "pen": border_pen,
                    "brush": pg.mkBrush(c),
                })

                lbl = self.node_labels[idx]
                lbl.setText(str(idx + 1))
                lbl.setPos(nx, ny)
                lbl.show()

                if band.is_dynamic_active and band.enabled:
                    dyn_y = float(np.clip(ny + band.current_dyn_offset_db, -self.eq_range_db, self.eq_range_db))
                    ring_c = _brighten_color(band.color, 0.25)
                    dyn_spots.append({
                        "pos": (nx, dyn_y),
                        "size": size + 8,
                        "pen": pg.mkPen(color=ring_c, width=2.0),
                        "brush": pg.mkBrush(0, 0, 0, 0),
                    })

            self.nodes_scatter.setData(node_spots)
            self.dyn_nodes_scatter.setData(dyn_spots)

            active_idx = self.dragging_band_idx if self.dragging_band_idx >= 0 else self.hovered_band_idx
            if 0 <= active_idx < len(bands):
                b = bands[active_idx]
                nx = self.freq_to_x(b.freq)
                ny = self._band_node_y(b)
                freq_str = f"{b.freq / 1000.0:.2f} kHz" if b.freq >= 1000.0 else f"{b.freq:.0f} Hz"
                dyn_str = f" | ΔG: {b.current_dyn_offset_db:+.1f} dB" if b.is_dynamic_active else ""
                html = (
                    f"<div style='background-color: rgba(24, 24, 28, 235); "
                    f"border: 1px solid {b.color}; border-radius: 5px; padding: 2px 7px; "
                    f"font-family: Segoe UI, sans-serif; font-size: 10px; color: #ECEFF4;'>"
                    f"<b style='color:{b.color};'>Band {active_idx + 1}</b> "
                    f"{freq_str} &bull; {b.gain_db:+.1f} dB &bull; Q {b.q:.2f}{dyn_str}</div>"
                )
                self.floating_tooltip.setHtml(html)
                self.floating_tooltip.setPos(nx, min(ny + 1.5, self.eq_range_db * 0.88))
                self.floating_tooltip.show()
            else:
                self.floating_tooltip.hide()


class FloatingBandHUD(QWidget):
    """
    HUD Flotante Superpuesto estilo FabFilter Pro-Q 4 + Apple Liquid Glass Oscuro:
    - La perilla grande de `GAIN` (diámetro 88px) está encastrada directamente en el notch
      curvo superior ("joroba"), despejando por completo la zona central superior.
    - El navegador de bandas (`<`, `Band 1`, `>`, `+`, `✕`, `⚡ DYN`, `🎧 Solo`) vive en una
      fila inferior limpia en la base del HUD, debajo de las etiquetas `FREQ`, `GAIN`, `Q`.
    - Cuerpo pintado con degradado oscuro profundo (#1E2028 a #14151A, alpha = 235) y borde
      especular Liquid Glass combinado con un resplandor sutil del color de la banda activa.
    """

    parameters_changed = pyqtSignal()

    def __init__(self, eq_engine: ParametricEQEngine, overlay: ProQ4Overlay, parent=None):
        super().__init__(parent)
        self.eq_engine = eq_engine
        self.overlay = overlay
        self._updating_ui = False
        self._active_band_color = QColor("#00E5FF")

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFixedSize(700, 196)

        self.setStyleSheet("""
            QWidget {
                background: transparent;
            }
            QLabel {
                background: transparent;
                border: none;
                color: #B0B6C2;
                font-size: 11px;
                font-weight: bold;
            }
            QComboBox {
                background-color: rgba(30, 33, 40, 220);
                color: #ECEFF4;
                border: 1px solid #3C404C;
                border-radius: 5px;
                padding: 2px 7px;
                font-size: 11px;
                font-weight: bold;
            }
            QComboBox:hover {
                border-color: #646B7D;
            }
            QPushButton {
                background-color: rgba(32, 35, 42, 210);
                color: #D8DEE9;
                border: 1px solid #3C404C;
                border-radius: 5px;
                padding: 1px 6px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(50, 55, 66, 235);
                color: #FFFFFF;
                border-color: #5E6575;
            }
        """)

        self._build_ui()
        self.overlay.band_selected.connect(self.select_band)
        self.overlay.band_modified.connect(self._on_overlay_band_modified)
        self.refresh_from_engine()

    def mousePressEvent(self, event):
        event.accept()

    def mouseMoveEvent(self, event):
        event.accept()

    def mouseReleaseEvent(self, event):
        event.accept()

    def mouseDoubleClickEvent(self, event):
        event.accept()

    def wheelEvent(self, event):
        event.accept()

    @staticmethod
    def _fmt_freq(v: float) -> str:
        return f"{v / 1000.0:.2f} kHz" if v >= 1000.0 else f"{v:.1f} Hz"

    def _build_ui(self):
        # Posicionamiento determinístico para encastrar GAIN (88px) dentro de la joroba superior (y_peak=3, y_top=34)
        # 1. Columna Izquierda: Power + Color Swatch en esquina superior izquierda del ala, Filter Shape y Slope debajo
        self.band_enable_btn = PowerToggleButton(self)
        self.band_enable_btn.setToolTip("Activar / Desactivar banda seleccionada")
        self.band_enable_btn.move(20, 44)
        self.band_enable_btn.clicked.connect(self._on_band_enable_toggled)

        self.color_swatch = ColorSwatchButton("#00E5FF", self)
        self.color_swatch.move(50, 44)
        self.color_swatch.colorSelected.connect(self._on_swatch_color_selected)

        self.type_combo = QComboBox(self)
        self.type_combo.addItems(FILTER_TYPES)
        self.type_combo.setFixedSize(112, 27)
        self.type_combo.move(20, 76)
        self.type_combo.setToolTip("Forma del filtro (Filter Shape)")
        self.type_combo.currentTextChanged.connect(self._on_type_or_slope_changed)

        self.slope_combo = QComboBox(self)
        self.slope_combo.setFixedSize(112, 27)
        self.slope_combo.move(20, 110)
        self.slope_combo.setToolTip("Pendiente del filtro (Slope)")
        for s in SLOPE_OPTIONS:
            self.slope_combo.addItem(f"{s} dB/oct", s)
        self.slope_combo.currentIndexChanged.connect(self._on_type_or_slope_changed)

        # 2. Bloque Central: FREQ (60px), GAIN (88px encastrado en la joroba en y=14), THR y Q (60px)
        self.freq_knob = ProQKnob(
            title="FREQ", min_val=10.0, max_val=24000.0, default_val=1000.0,
            fmt_func=self._fmt_freq, log_scale=True, bipolar=False,
            accent_color="#00E5FF", diameter=60, parent=self
        )
        self.freq_knob.move(146, 48)
        self.freq_knob.valueChanged.connect(self._on_knob_param_changed)

        # GAIN de 88px alojada en el notch curvo superior (su mitad superior sube hasta y=14, dentro de la joroba!)
        self.gain_dyn_knob = GainDynProQKnob(diameter=88, parent=self)
        self.gain_dyn_knob.move(232, 14)
        self.gain_dyn_knob.valueChanged.connect(self._on_knob_param_changed)
        self.gain_dyn_knob.dynRangeChanged.connect(self._on_dyn_range_knob_changed)

        self.thresh_meter = DynThresholdMeter(parent=self)
        self.thresh_meter.move(346, 44)
        self.thresh_meter.thresholdChanged.connect(self._on_threshold_meter_changed)

        self.q_knob = ProQKnob(
            title="Q", min_val=0.10, max_val=24.0, default_val=1.00,
            fmt_func=lambda v: f"{v:.2f}", log_scale=True, bipolar=False,
            accent_color="#00E5FF", diameter=60, parent=self
        )
        self.q_knob.move(398, 48)
        self.q_knob.valueChanged.connect(self._on_knob_param_changed)

        # 3. Bloque Derecho: Perillas auxiliares A, R y OUT (diámetro 46px) con espaciado uniforme
        self.attack_knob = ProQKnob(
            title="A", min_val=0.5, max_val=500.0, default_val=15.0,
            fmt_func=lambda v: f"{v:.1f} ms" if v < 10 else f"{v:.0f} ms",
            log_scale=True, bipolar=False, accent_color="#00E5FF", diameter=46, parent=self
        )
        self.attack_knob.move(492, 56)
        self.attack_knob.valueChanged.connect(self._on_knob_param_changed)

        self.release_knob = ProQKnob(
            title="R", min_val=5.0, max_val=2000.0, default_val=120.0,
            fmt_func=lambda v: f"{v:.0f} ms",
            log_scale=True, bipolar=False, accent_color="#00E5FF", diameter=46, parent=self
        )
        self.release_knob.move(556, 56)
        self.release_knob.valueChanged.connect(self._on_knob_param_changed)

        self.out_gain_knob = ProQKnob(
            title="OUT", min_val=-24.0, max_val=24.0, default_val=0.0,
            fmt_func=lambda v: f"{v:+.1f} dB", log_scale=False, bipolar=True,
            accent_color="#FFD54F", diameter=46, parent=self
        )
        self.out_gain_knob.move(620, 56)
        self.out_gain_knob.valueChanged.connect(self._on_global_changed)

        # 4. Fila Inferior Limpia en la Base del HUD (debajo de FREQ, GAIN, Q: y = 158..184)
        self.bottom_bar = QWidget(self)
        self.bottom_bar.setGeometry(20, 156, 660, 28)
        bot_row = QHBoxLayout(self.bottom_bar)
        bot_row.setContentsMargins(0, 0, 0, 0)
        bot_row.setSpacing(6)

        self.prev_band_btn = QPushButton("‹", self.bottom_bar)
        self.prev_band_btn.setFixedSize(24, 24)
        self.prev_band_btn.setToolTip("Banda anterior")
        self.prev_band_btn.clicked.connect(lambda: self._step_band(-1))
        bot_row.addWidget(self.prev_band_btn)

        self.band_combo = QComboBox(self.bottom_bar)
        self.band_combo.setFixedSize(96, 24)
        self.band_combo.currentIndexChanged.connect(self._on_band_combo_changed)
        bot_row.addWidget(self.band_combo)

        self.next_band_btn = QPushButton("›", self.bottom_bar)
        self.next_band_btn.setFixedSize(24, 24)
        self.next_band_btn.setToolTip("Banda siguiente")
        self.next_band_btn.clicked.connect(lambda: self._step_band(1))
        bot_row.addWidget(self.next_band_btn)

        self.add_band_btn = QPushButton("+", self.bottom_bar)
        self.add_band_btn.setFixedSize(24, 24)
        self.add_band_btn.setToolTip("Añadir nueva banda")
        self.add_band_btn.clicked.connect(self._on_add_band)
        bot_row.addWidget(self.add_band_btn)

        self.del_band_btn = QPushButton("✕", self.bottom_bar)
        self.del_band_btn.setFixedSize(24, 24)
        self.del_band_btn.setToolTip("Eliminar banda seleccionada (Supr)")
        self.del_band_btn.clicked.connect(self._on_delete_band)
        bot_row.addWidget(self.del_band_btn)

        bot_row.addStretch(1)

        self.dyn_toggle_btn = LiquidGlassButton("⚡ DYN", self.bottom_bar, pill_radius=11.0)
        self.dyn_toggle_btn.setCheckable(True)
        self.dyn_toggle_btn.setFixedSize(68, 24)
        self.dyn_toggle_btn.setToolTip("Activar / Desactivar modo Dynamic EQ en esta banda")
        self.dyn_toggle_btn.clicked.connect(self._on_dyn_toggle_clicked)
        bot_row.addWidget(self.dyn_toggle_btn)

        self.band_solo_btn = LiquidGlassButton("🎧 Solo", self.bottom_bar, active_tint_color="#00E5FF", pill_radius=11.0)
        self.band_solo_btn.setCheckable(True)
        self.band_solo_btn.setFixedSize(68, 24)
        self.band_solo_btn.setToolTip("Escuchar únicamente esta banda")
        self.band_solo_btn.clicked.connect(self._on_band_solo_toggled)
        bot_row.addWidget(self.band_solo_btn)

    def _gain_center_x(self) -> float:
        return float(self.gain_dyn_knob.x() + self.gain_dyn_knob.width() * 0.5)

    def paintEvent(self, event):
        """
        Cuerpo del HUD estilo Liquid Glass Oscuro (#1E2028 a #14151A, alpha = 235) con la joroba
        superior ("notch") alojando la mitad superior de la perilla grande de GAIN (88px).
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        w = float(self.width())
        h = float(self.height())
        pad = 2.5
        r_corner = 18.0
        y_top = 34.0     # Alas planas a y=34 para que GAIN (y=14) sobresalga dentro de la joroba
        y_peak = 2.5     # Cima del arco convexo superior sobre GAIN
        y_bot = h - pad
        x_left = pad
        x_right = w - pad

        arch_cx = self._gain_center_x()
        arch_half_w = 104.0
        ax1 = max(x_left + r_corner + 10.0, arch_cx - arch_half_w)
        ax2 = min(x_right - r_corner - 10.0, arch_cx + arch_half_w)

        path = QPainterPath()
        path.moveTo(x_left + r_corner, y_top)
        path.lineTo(ax1, y_top)

        # Elevación curva convexa ("joroba") simétrica que envuelve la mitad superior de GAIN
        path.cubicTo(
            QPointF(ax1 + 38.0, y_top),
            QPointF(arch_cx - 42.0, y_peak),
            QPointF(arch_cx, y_peak),
        )
        path.cubicTo(
            QPointF(arch_cx + 42.0, y_peak),
            QPointF(ax2 - 38.0, y_top),
            QPointF(ax2, y_top),
        )

        path.lineTo(x_right - r_corner, y_top)
        path.quadTo(QPointF(x_right, y_top), QPointF(x_right, y_top + r_corner))
        path.lineTo(x_right, y_bot - r_corner)
        path.quadTo(QPointF(x_right, y_bot), QPointF(x_right - r_corner, y_bot))
        path.lineTo(x_left + r_corner, y_bot)
        path.quadTo(QPointF(x_left, y_bot), QPointF(x_left, y_bot - r_corner))
        path.lineTo(x_left, y_top + r_corner)
        path.quadTo(QPointF(x_left, y_top), QPointF(x_left + r_corner, y_top))
        path.closeSubpath()

        # 1. Relleno Liquid Glass Oscuro (#1E2028 a #14151A, alpha = 235)
        body_grad = QLinearGradient(0.0, y_peak, 0.0, y_bot)
        body_grad.setColorAt(0.0, QColor(34, 37, 46, 238))
        body_grad.setColorAt(0.25, QColor(30, 32, 40, 235))
        body_grad.setColorAt(1.0, QColor(20, 21, 26, 242))
        painter.fillPath(path, QBrush(body_grad))

        # 2. Resplandor exterior sutil del color de la banda activa
        glow_c = QColor(self._active_band_color)
        glow_c.setAlpha(45)
        painter.setPen(QPen(glow_c, 3.2))
        painter.drawPath(path)

        # 3. Borde especular fino (combina reflejo blanco superior Liquid Glass + color de banda)
        spec_border = QLinearGradient(0.0, y_peak, 0.0, y_bot)
        top_spec = _brighten_color(self._active_band_color, 0.65)
        top_spec.setAlpha(220)
        mid_spec = QColor(self._active_band_color)
        mid_spec.setAlpha(155)
        bot_spec = QColor(self._active_band_color)
        bot_spec.setAlpha(195)
        spec_border.setColorAt(0.0, top_spec)
        spec_border.setColorAt(0.45, mid_spec)
        spec_border.setColorAt(1.0, bot_spec)

        painter.setPen(QPen(QBrush(spec_border), 1.35))
        painter.drawPath(path)

        # 4. Separador sutil sobre la fila inferior de navegación de bandas
        painter.setPen(QPen(QColor(255, 255, 255, 20), 1.0))
        painter.drawLine(QPointF(20.0, 151.0), QPointF(w - 20.0, 151.0))

    def _step_band(self, delta: int):
        with self.eq_engine.lock:
            n = len(self.eq_engine.bands)
        if n == 0:
            return
        new_idx = (self.overlay.selected_band_idx + delta) % n
        self.select_band(new_idx)

    def _on_swatch_color_selected(self, hex_color: str):
        idx = self.overlay.selected_band_idx
        with self.eq_engine.lock:
            if 0 <= idx < len(self.eq_engine.bands):
                self.eq_engine.bands[idx].color = hex_color
                self.eq_engine.mark_dirty()
        self.refresh_from_engine()
        self.overlay.update_eq_curves(force=True)
        self.parameters_changed.emit()

    def _on_overlay_band_modified(self):
        self.refresh_from_engine()
        self.parameters_changed.emit()

    def select_band(self, index: int):
        self.overlay.selected_band_idx = index
        self.refresh_from_engine()
        self.overlay.update_eq_curves(force=True)

    def refresh_from_engine(self):
        self._updating_ui = True
        try:
            with self.eq_engine.lock:
                bands = self.eq_engine.bands
                idx = self.overlay.selected_band_idx
                if idx < 0 and bands:
                    idx = 0
                    self.overlay.selected_band_idx = 0

                self.band_combo.blockSignals(True)
                self.band_combo.clear()
                for i, b in enumerate(bands):
                    dyn_tag = " ⚡" if b.is_dynamic_active else ""
                    self.band_combo.addItem(f"Band {i + 1}{dyn_tag}", i)
                if 0 <= idx < len(bands):
                    self.band_combo.setCurrentIndex(idx)
                self.band_combo.blockSignals(False)

                has_band = 0 <= idx < len(bands)
                for w in (
                    self.del_band_btn, self.band_enable_btn, self.prev_band_btn,
                    self.next_band_btn, self.color_swatch, self.band_solo_btn,
                    self.type_combo, self.slope_combo, self.freq_knob,
                    self.gain_dyn_knob, self.thresh_meter, self.q_knob,
                    self.dyn_toggle_btn, self.attack_knob, self.release_knob,
                ):
                    w.setEnabled(has_band)

                if has_band:
                    b = bands[idx]
                    self._active_band_color = QColor(b.color)
                    self.color_swatch.setColor(b.color)
                    self.band_enable_btn.setAccentColor(b.color)

                    self.band_enable_btn.setChecked(b.enabled)
                    self.band_solo_btn.setChecked(b.solo)
                    self.band_solo_btn.setActiveTintColor(b.color)
                    self.type_combo.setCurrentText(b.filter_type)

                    slope_idx = self.slope_combo.findData(b.slope)
                    if slope_idx >= 0:
                        self.slope_combo.setCurrentIndex(slope_idx)
                    self.slope_combo.setVisible(b.supports_slope)
                    self.slope_combo.setEnabled(b.supports_slope)
                    self.gain_dyn_knob.setEnabled(b.supports_gain)

                    self.freq_knob.setAccentColor(b.color)
                    self.gain_dyn_knob.setAccentColor(b.color)
                    self.thresh_meter.setAccentColor(b.color)
                    self.q_knob.setAccentColor(b.color)
                    self.attack_knob.setAccentColor(b.color)
                    self.release_knob.setAccentColor(b.color)

                    self.freq_knob.setValue(b.freq)
                    self.gain_dyn_knob.setValue(b.gain_db)
                    self.gain_dyn_knob.setDynRangeValue(b.dyn_range_db)
                    self.gain_dyn_knob.setDynEnabled(b.dyn_enabled)
                    self.q_knob.setValue(b.q)

                    self.dyn_toggle_btn.setChecked(b.dyn_enabled)
                    self.dyn_toggle_btn.setActiveTintColor(b.color)

                    dyn_active = b.dyn_enabled or abs(b.dyn_range_db) >= 0.05
                    self.thresh_meter.setEnabled(dyn_active)
                    self.attack_knob.setEnabled(dyn_active)
                    self.release_knob.setEnabled(dyn_active)

                    self.thresh_meter.setThresholdValue(b.threshold_db)
                    self.attack_knob.setValue(b.attack_ms)
                    self.release_knob.setValue(b.release_ms)
                else:
                    self.slope_combo.setVisible(False)

                self.out_gain_knob.setValue(self.eq_engine.output_gain_db)
            self.update()
        finally:
            self._updating_ui = False

    def _on_band_combo_changed(self, index: int):
        if self._updating_ui or index < 0:
            return
        self.select_band(index)

    def _on_add_band(self):
        self.eq_engine.add_band(freq=1000.0, gain_db=0.0, q=1.0, filter_type="Bell")
        new_idx = len(self.eq_engine.bands) - 1
        self.select_band(new_idx)
        self.parameters_changed.emit()

    def _on_delete_band(self):
        idx = self.overlay.selected_band_idx
        if 0 <= idx < len(self.eq_engine.bands):
            self.eq_engine.remove_band(idx)
            new_idx = min(idx, len(self.eq_engine.bands) - 1)
            self.select_band(new_idx)
            self.parameters_changed.emit()

    def _on_band_enable_toggled(self, checked: bool):
        if self._updating_ui:
            return
        idx = self.overlay.selected_band_idx
        with self.eq_engine.lock:
            if 0 <= idx < len(self.eq_engine.bands):
                self.eq_engine.bands[idx].enabled = checked
                self.eq_engine.mark_dirty()
        self.refresh_from_engine()
        self.overlay.update_eq_curves(force=True)
        self.parameters_changed.emit()

    def _on_band_solo_toggled(self, checked: bool):
        if self._updating_ui:
            return
        idx = self.overlay.selected_band_idx
        with self.eq_engine.lock:
            if 0 <= idx < len(self.eq_engine.bands):
                self.eq_engine.bands[idx].solo = checked
                self.eq_engine.mark_dirty()
        self.overlay.update_eq_curves(force=True)
        self.parameters_changed.emit()

    def _on_type_or_slope_changed(self):
        if self._updating_ui:
            return
        idx = self.overlay.selected_band_idx
        with self.eq_engine.lock:
            if 0 <= idx < len(self.eq_engine.bands):
                b = self.eq_engine.bands[idx]
                b.filter_type = self.type_combo.currentText()
                b.slope = int(self.slope_combo.currentData())
                b.update_coefficients()
                self.eq_engine.mark_dirty()
        self.refresh_from_engine()
        self.overlay.update_eq_curves(force=True)
        self.parameters_changed.emit()

    def _on_knob_param_changed(self, _val: float = 0.0):
        if self._updating_ui:
            return
        idx = self.overlay.selected_band_idx
        with self.eq_engine.lock:
            if 0 <= idx < len(self.eq_engine.bands):
                b = self.eq_engine.bands[idx]
                b.freq = self.freq_knob.value()
                b.gain_db = self.gain_dyn_knob.value()
                b.q = self.q_knob.value()
                b.attack_ms = self.attack_knob.value()
                b.release_ms = self.release_knob.value()
                b.update_coefficients()
                self.eq_engine.mark_dirty()
        self.overlay.update_eq_curves(force=True)
        self.parameters_changed.emit()

    def _on_threshold_meter_changed(self, val: float):
        if self._updating_ui:
            return
        idx = self.overlay.selected_band_idx
        with self.eq_engine.lock:
            if 0 <= idx < len(self.eq_engine.bands):
                b = self.eq_engine.bands[idx]
                b.threshold_db = float(val)
                b.update_coefficients()
                self.eq_engine.mark_dirty()
        self.overlay.update_eq_curves(force=True)
        self.parameters_changed.emit()

    def _on_dyn_range_knob_changed(self, val: float):
        if self._updating_ui:
            return
        idx = self.overlay.selected_band_idx
        with self.eq_engine.lock:
            if 0 <= idx < len(self.eq_engine.bands):
                b = self.eq_engine.bands[idx]
                b.dyn_range_db = float(val)
                if abs(val) >= 0.1 and not b.dyn_enabled:
                    b.dyn_enabled = True
                b.update_coefficients()
                self.eq_engine.mark_dirty()
        self.refresh_from_engine()
        self.overlay.update_eq_curves(force=True)
        self.parameters_changed.emit()

    def _on_dyn_toggle_clicked(self, checked: bool):
        if self._updating_ui:
            return
        idx = self.overlay.selected_band_idx
        with self.eq_engine.lock:
            if 0 <= idx < len(self.eq_engine.bands):
                b = self.eq_engine.bands[idx]
                b.dyn_enabled = checked
                if checked and abs(b.dyn_range_db) < 0.1:
                    b.dyn_range_db = -6.0
                b.update_coefficients()
                self.eq_engine.mark_dirty()
        self.refresh_from_engine()
        self.overlay.update_eq_curves(force=True)
        self.parameters_changed.emit()

    def _on_global_changed(self, _val: float = 0.0):
        if self._updating_ui:
            return
        with self.eq_engine.lock:
            self.eq_engine.output_gain_db = self.out_gain_knob.value()
            self.eq_engine.mark_dirty()
        self.overlay.update_eq_curves(force=True)
        self.parameters_changed.emit()

    def _on_reset_eq(self):
        self.eq_engine.init_default_bands()
        self.select_band(0)
        self.parameters_changed.emit()

    def update_live_Readouts(self):
        idx = self.overlay.selected_band_idx
        with self.eq_engine.lock:
            if 0 <= idx < len(self.eq_engine.bands):
                b = self.eq_engine.bands[idx]
                self.gain_dyn_knob.setLiveDynOffset(b.current_dyn_offset_db if b.is_dynamic_active else 0.0)
                self.thresh_meter.setLiveLevel(b.env_db if b.is_dynamic_active else -70.0)


# Alias de compatibilidad
EQControlPanel = FloatingBandHUD
