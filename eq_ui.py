import math
import time
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QFrame, QMenu, QColorDialog, QDialog, QToolTip,
    QGraphicsOpacityEffect, QListView, QSlider, QGraphicsPathItem, QGraphicsItem,
    QApplication, QCheckBox, QDoubleSpinBox, QGridLayout, QWidgetAction
)
from PyQt6.QtCore import Qt, QEvent, QPoint, QPointF, QRectF, pyqtSignal, QTimer
from PyQt6.QtGui import (
    QColor, QFont, QPainter, QPainterPath, QPen, QBrush,
    QLinearGradient, QPolygonF, QPixmap, QImage, QIcon, QCursor
)
import pyqtgraph as pg

from eq_dsp import (
    ParametricEQEngine,
    EQBand,
    FILTER_TYPES,
    SLOPE_OPTIONS,
    CHANNEL_MODES,
    TARGET_OUTPUTS,
)


def style_dark_combobox(combo: QComboBox):
    """Fuerza a PyQt6 a renderizar la lista desplegable internamente con estética oscura de alto contraste."""
    list_view = QListView(combo)
    list_view.setUniformItemSizes(True)
    list_view.setSpacing(2)
    list_view.setStyleSheet("""
        QListView {
            background-color: #181A20;
            color: #F0F2F5;
            border: 1px solid #3A3F4B;
            border-radius: 6px;
            padding: 4px;
            outline: none;
        }
        QListView::item {
            min-height: 24px;
            padding: 4px 10px;
            border-radius: 4px;
            color: #F0F2F5;
            background-color: transparent;
        }
        QListView::item:hover {
            background-color: #2B303C;
            color: #FFFFFF;
        }
        QListView::item:selected {
            background-color: #00E5FF;
            color: #0D0F14;
            font-weight: bold;
        }
    """)
    combo.setView(list_view)
    try:
        combo.view().window().setWindowFlags(
            Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint | Qt.WindowType.NoDropShadowWindowHint
        )
        combo.view().window().setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
    except Exception:
        pass


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
    Botón translúcido con la estética del HUD (fondo oscuro translúcido de baja opacidad
    y bordes coloreados):
    - Inactivo: Relleno oscuro translúcido rgba(16, 18, 24, 120) con borde sutil rgba(62, 68, 82, 155).
    - Hover: Sube sutilmente la luminancia del relleno y tiñe el borde con el color de acento.
    - Activo / Presionado: Fondo translúcido teñido con baja opacidad del color de acento,
      resplandor exterior suave y borde coloreado nítido (1.25px), indicando claramente el estado presionado.
    """

    def __init__(
        self,
        text: str = "",
        parent=None,
        tint_color: str | None = None,
        active_tint_color: str | None = "#00E5FF",
        pill_radius: float = 7.0,
    ):
        super().__init__(text, parent)
        self._tint_color = QColor(tint_color) if tint_color else None
        self._active_tint_color = QColor(active_tint_color) if active_tint_color else QColor("#00E5FF")
        if self._tint_color is not None and active_tint_color == "#00E5FF":
            self._active_tint_color = QColor(self._tint_color)
        self._pill_radius = min(float(pill_radius), 7.0)
        self._hovered = False
        self._force_active = False
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(28)

    def setTintColor(self, hex_color: str | None):
        self._tint_color = QColor(hex_color) if hex_color else None
        if self._tint_color is not None:
            self._active_tint_color = QColor(self._tint_color)
        self.update()

    def setActiveTintColor(self, hex_color: str | None):
        if hex_color:
            self._active_tint_color = QColor(hex_color)
            self.update()

    def setForceActive(self, active: bool):
        if self._force_active != bool(active):
            self._force_active = bool(active)
            self.update()

    def enterEvent(self, event):
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.update()

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.update()

    def paintEvent(self, event):
        w = int(self.width())
        h = int(self.height())
        if w <= 0 or h <= 0:
            return

        dpr = float(self.devicePixelRatioF()) if hasattr(self, "devicePixelRatioF") else 1.0
        is_active = (
            (self.isCheckable() and self.isChecked())
            or self.isDown()
            or getattr(self, "_force_active", False)
        )
        accent = self._active_tint_color if self._active_tint_color is not None else (
            self._tint_color if self._tint_color is not None else QColor("#00E5FF")
        )
        acc_val = accent.rgba()
        cache_key = (w, h, dpr, self._hovered, is_active, acc_val, self.text(), self.isEnabled())

        if getattr(self, "_cached_key", None) != cache_key or getattr(self, "_cached_pixmap", None) is None:
            self._cached_key = cache_key
            pw = max(1, int(round(w * dpr)))
            ph = max(1, int(round(h * dpr)))
            pix = QPixmap(pw, ph)
            pix.setDevicePixelRatio(dpr)
            pix.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pix)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            self._draw_hud_button(painter, float(w), float(h), is_active, accent)
            painter.end()
            self._cached_pixmap = pix

        p = QPainter(self)
        p.drawPixmap(0, 0, self._cached_pixmap)

    def _draw_hud_button(self, painter: QPainter, w: float, h: float, is_active: bool, accent: QColor):
        rect = QRectF(1.2, 1.2, w - 2.4, h - 2.4)
        r = min(self._pill_radius, (h - 2.4) * 0.45)

        # 1. Fondo translúcido oscuro estilo HUD
        base_alpha = 142 if is_active else (138 if self._hovered else 118)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(16, 18, 24, base_alpha))
        painter.drawRoundedRect(rect, r, r)

        if is_active and self.isEnabled():
            # Tinte translúcido de baja opacidad del color contextual
            tint_fill = QColor(accent)
            tint_fill.setAlpha(62 if self._hovered else 46)
            painter.setBrush(tint_fill)
            painter.drawRoundedRect(rect, r, r)

            # Resplandor exterior suave estilo HUD
            glow = QColor(accent)
            glow.setAlpha(44 if self._hovered else 34)
            painter.setPen(QPen(glow, 2.4))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(rect, r, r)

            # Borde coloreado nítido que indica estado presionado/activo
            border = _brighten_color(accent, 0.18 if self._hovered else 0.08)
            border.setAlpha(235 if self._hovered else 210)
            painter.setPen(QPen(border, 1.25))
            painter.drawRoundedRect(rect, r, r)
        else:
            # Estado no presionado: borde discreto, se tiñe suavemente en hover
            if self._hovered and self.isEnabled():
                hover_tint = QColor(accent)
                hover_tint.setAlpha(18)
                painter.setBrush(hover_tint)
                painter.drawRoundedRect(rect, r, r)

                border = QColor(accent)
                border.setAlpha(135)
                painter.setPen(QPen(border, 1.15))
            else:
                border = QColor(62, 68, 82, 155 if self.isEnabled() else 85)
                painter.setPen(QPen(border, 1.05))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(rect, r, r)

        # 2. Texto nítido centrado
        font = QFont("Segoe UI", 8, QFont.Weight.Bold)
        painter.setFont(font)
        if not self.isEnabled():
            txt_c = QColor(115, 122, 136, 135)
        elif is_active:
            txt_c = _brighten_color(accent, 0.38)
        else:
            txt_c = QColor("#F0F4F8") if self._hovered else QColor("#C5CDD9")

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
        act_custom = menu.addAction("Personalizado...")
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


class CalloutIconButton(QPushButton):
    """Botón compacto con íconos vectoriales anti-aliased para el callout del nodo estilo Pro-Q 4."""

    def __init__(self, icon_type: str, parent=None):
        super().__init__(parent)
        self.icon_type = icon_type
        self._hovered = False
        self._color = QColor("#8E95A5")
        self._active = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def setColor(self, color_val, active: bool = False):
        self._color = QColor(color_val)
        self._active = bool(active)
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
        cx = w * 0.5
        cy = h * 0.5

        if self._hovered:
            bg = QColor(255, 255, 255, 25)
            if self.icon_type == "close":
                bg = QColor(255, 82, 82, 35)
            elif self.icon_type == "solo":
                bg = QColor(0, 229, 255, 35)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(bg))
            painter.drawRoundedRect(QRectF(1.0, 1.0, w - 2.0, h - 2.0), 4.0, 4.0)

        c = self._color if not (self._hovered and self.icon_type == "close") else QColor("#FF5252")
        if self._hovered and self.icon_type in ("power", "solo", "menu"):
            c = _brighten_color(c, 0.3)

        if self.icon_type == "power":
            r = 5.4
            pen = QPen(c, 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawArc(QRectF(cx - r, cy - r + 0.5, 2.0 * r, 2.0 * r), 120 * 16, -300 * 16)
            painter.drawLine(QPointF(cx, cy - r - 0.5), QPointF(cx, cy + 0.5))

        elif self.icon_type == "solo":
            font = QFont("Segoe UI", 8, QFont.Weight.Bold)
            painter.setFont(font)
            painter.setPen(c)
            painter.drawText(QRectF(0.0, 0.0, w, h), Qt.AlignmentFlag.AlignCenter, "S")

        elif self.icon_type == "close":
            pen = QPen(c, 1.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            d = 3.8
            painter.drawLine(QPointF(cx - d, cy - d), QPointF(cx + d, cy + d))
            painter.drawLine(QPointF(cx + d, cy - d), QPointF(cx - d, cy + d))

        elif self.icon_type == "menu":
            pen = QPen(c, 1.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            path = QPainterPath()
            path.moveTo(cx - 3.5, cy - 1.5)
            path.lineTo(cx, cy + 2.0)
            path.lineTo(cx + 3.5, cy - 1.5)
            painter.drawPath(path)


class FilterShapeIconWidget(QWidget):
    """Pinta la silueta vectorial de la forma del filtro de banda."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._filter_type = "Bell"
        self._color = QColor("#00E5FF")

    def setShape(self, filter_type: str, color_hex: str):
        self._filter_type = filter_type
        self._color = QColor(color_hex)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w = float(self.width())
        h = float(self.height())
        cx = w * 0.5
        cy = h * 0.5

        pen = QPen(self._color, 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        path = QPainterPath()
        ft = self._filter_type
        if ft == "Bell":
            path.moveTo(2.0, cy + 3.0)
            path.cubicTo(cx - 3.0, cy + 3.0, cx - 2.0, cy - 5.0, cx, cy - 5.0)
            path.cubicTo(cx + 2.0, cy - 5.0, cx + 3.0, cy + 3.0, w - 2.0, cy + 3.0)
        elif ft == "Low Shelf":
            path.moveTo(2.0, cy - 4.0)
            path.lineTo(cx - 2.0, cy - 4.0)
            path.cubicTo(cx, cy - 4.0, cx + 1.0, cy + 3.0, cx + 4.0, cy + 3.0)
            path.lineTo(w - 2.0, cy + 3.0)
        elif ft == "High Shelf":
            path.moveTo(2.0, cy + 3.0)
            path.lineTo(cx - 4.0, cy + 3.0)
            path.cubicTo(cx - 1.0, cy + 3.0, cx, cy - 4.0, cx + 2.0, cy - 4.0)
            path.lineTo(w - 2.0, cy - 4.0)
        elif ft == "Low Cut":
            path.moveTo(2.0, cy + 5.0)
            path.cubicTo(cx - 2.0, cy + 4.0, cx, cy - 4.0, w - 2.0, cy - 4.0)
        elif ft == "High Cut":
            path.moveTo(2.0, cy - 4.0)
            path.cubicTo(cx, cy - 4.0, cx + 2.0, cy + 4.0, w - 2.0, cy + 5.0)
        elif ft == "Notch":
            path.moveTo(2.0, cy - 4.0)
            path.lineTo(cx - 3.0, cy - 4.0)
            path.lineTo(cx, cy + 4.0)
            path.lineTo(cx + 3.0, cy - 4.0)
            path.lineTo(w - 2.0, cy - 4.0)
        else:  # Band Pass
            path.moveTo(2.0, cy + 4.0)
            path.lineTo(cx - 2.0, cy - 4.0)
            path.lineTo(cx + 2.0, cy - 4.0)
            path.lineTo(w - 2.0, cy + 4.0)

        painter.drawPath(path)


FILTER_SHAPE_ICONS = {
    "Bell": "⋀",
    "Low Shelf": "⎍",
    "High Shelf": "⎎",
    "Low Cut": "╱",
    "High Cut": "╲",
    "Notch": "∨",
    "Band Pass": "⋂",
}


class BandNodeCallout(QWidget):
    """
    Menú flotante / callout sobre el nodo activo estilo FabFilter Pro-Q 4:
    - Dimensiones: 185px de ancho por 82px de alto total (con flecha inferior hacia el nodo).
    - Fondo oscuro #1A1D24 (alpha=235), borde redondeado de 8px con flecha inferior.
    - Tipografía clara de 10pt bold.
    - Controles interactivos:
      - Arriba a la izquierda: Botón ⏻ (Power de la banda, pintado con band.color).
      - Centro (3 renglones legibles): Frecuencia (4545.9 Hz), Ganancia (+5.91 dB) y Factor Q (Q: 1.000).
      - Arriba a la derecha: Botón Isolate de frecuencia y botón ✕ para borrar la banda.
      - Abajo: Ícono de forma de filtro, indicador canal/salida (LR/M/S • A/B/A+B) y flecha ⌄ para cambio rápido.
    """

    def __init__(self, overlay, parent=None):
        super().__init__(parent)
        self.overlay = overlay
        self.setFixedSize(185, 82)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._band_color = "#00E5FF"
        self._building = True

        self._opacity_effect = QGraphicsOpacityEffect(self)
        self._opacity_effect.setOpacity(1.0)
        self.setGraphicsEffect(self._opacity_effect)
        self._current_opacity = 1.0

        self.setStyleSheet("""
            QWidget {
                background: transparent;
            }
            QLabel {
                background: transparent;
                border: none;
            }
        """)

        # 1. Fila Superior: Power (⏻), Frecuencia (Bold 10pt), Solo (Isolate) y Eliminar (✕)
        self.power_btn = CalloutIconButton("power", self)
        self.power_btn.setGeometry(7, 5, 22, 22)
        self.power_btn.setToolTip("Activar / Desactivar banda")
        self.power_btn.clicked.connect(self._on_power_clicked)

        self.freq_label = QLabel(self)
        self.freq_label.setGeometry(30, 5, 98, 20)
        self.freq_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.freq_label.setStyleSheet("color: #FFFFFF; font-family: 'Segoe UI'; font-size: 10pt; font-weight: bold;")

        self.solo_btn = CalloutIconButton("solo", self)
        self.solo_btn.setGeometry(132, 5, 22, 22)
        self.solo_btn.setToolTip("Aislar frecuencia (Isolate)")
        self.solo_btn.clicked.connect(self._on_solo_clicked)

        self.close_btn = CalloutIconButton("close", self)
        self.close_btn.setGeometry(156, 5, 22, 22)
        self.close_btn.setToolTip("Eliminar banda (Supr)")
        self.close_btn.clicked.connect(self._on_close_clicked)

        # 2. Renglón Medio: Ganancia (Bold 10pt)
        self.gain_label = QLabel(self)
        self.gain_label.setGeometry(15, 27, 155, 20)
        self.gain_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gain_label.setStyleSheet("color: #ECEFF4; font-family: 'Segoe UI'; font-size: 10pt; font-weight: bold;")

        # 3. Renglón Inferior: Forma, Q (Bold 10pt), Canal/Salida y Flecha ⌄
        self.shape_icon = FilterShapeIconWidget(self)
        self.shape_icon.setGeometry(8, 49, 22, 22)

        self.q_label = QLabel(self)
        self.q_label.setGeometry(32, 50, 68, 20)
        self.q_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.q_label.setStyleSheet("color: #B0B6C2; font-family: 'Segoe UI'; font-size: 10pt; font-weight: bold;")

        self.mode_badge = QLabel(self)
        self.mode_badge.setGeometry(100, 52, 58, 17)
        self.mode_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mode_badge.setStyleSheet("color: #8C92A4; font-family: 'Segoe UI'; font-size: 8.5pt; font-weight: bold;")

        self.menu_btn = CalloutIconButton("menu", self)
        self.menu_btn.setGeometry(162, 50, 16, 20)
        self.menu_btn.setToolTip("Menú rápido de opciones de banda")
        self.menu_btn.clicked.connect(self._on_menu_clicked)

        self._building = False
        self.hide()

    def mousePressEvent(self, event):
        event.accept()

    def mouseDoubleClickEvent(self, event):
        event.accept()

    def _on_power_clicked(self):
        idx = self.overlay.selected_band_idx
        with self.overlay.eq_engine.lock:
            if 0 <= idx < len(self.overlay.eq_engine.bands):
                b = self.overlay.eq_engine.bands[idx]
                new_en = not b.enabled
            else:
                return
        self.overlay._toggle_band_enabled(idx, new_en)

    def _on_solo_clicked(self):
        idx = self.overlay.selected_band_idx
        is_cur_iso = (self.overlay.eq_engine.isolated_band_index == idx)
        self.overlay._toggle_band_solo(idx, not is_cur_iso)

    def _on_close_clicked(self):
        idx = self.overlay.selected_band_idx
        self.overlay._delete_band(idx)

    def _on_menu_clicked(self):
        idx = self.overlay.selected_band_idx
        pos = self.menu_btn.mapToGlobal(QPoint(0, self.menu_btn.height() + 2))
        self.overlay._show_node_context_menu(idx, pos)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        w = float(self.width())
        body_h = 75.0
        caret_h = 6.0
        cx = w * 0.5

        body_rect = QRectF(1.0, 1.0, w - 2.0, body_h)
        path = QPainterPath()
        path.addRoundedRect(body_rect, 8.0, 8.0)

        caret_path = QPainterPath()
        caret_path.moveTo(cx - 7.0, body_h)
        caret_path.lineTo(cx, body_h + caret_h)
        caret_path.lineTo(cx + 7.0, body_h)
        caret_path.closeSubpath()

        full_path = path.united(caret_path)

        # Fondo oscuro #1A1D24 con alpha=235
        painter.fillPath(full_path, QBrush(QColor(26, 29, 36, 235)))

        # Resplandor exterior sutil del color de la banda
        glow_c = QColor(self._band_color)
        glow_c.setAlpha(45)
        painter.setPen(QPen(glow_c, 3.0))
        painter.drawPath(full_path)

        # Borde redondeado nítido
        border_c = QColor(self._band_color)
        border_c.setAlpha(180)
        painter.setPen(QPen(border_c, 1.2))
        painter.drawPath(full_path)

    def update_callout(self):
        if self._building:
            return
        idx = self.overlay.selected_band_idx
        with self.overlay.eq_engine.lock:
            bands = self.overlay.eq_engine.bands
            if not (0 <= idx < len(bands)):
                self.hide()
                return
            b = bands[idx]
            freq = b.freq
            gain = b.gain_db
            q = b.q
            enabled = b.enabled
            color = b.color
            ftype = b.filter_type
            cm = b.channel_mode
            to_val = b.target_output
            supports_gain = b.supports_gain
            is_iso = bool(b.solo or self.overlay.eq_engine.isolated_band_index == idx)

        self._band_color = color

        # Frecuencia
        if freq >= 10000.0:
            self.freq_label.setText(f"{freq / 1000.0:.2f} kHz")
        elif freq >= 1000.0:
            self.freq_label.setText(f"{freq / 1000.0:.2f} kHz")
        else:
            self.freq_label.setText(f"{freq:.1f} Hz")

        # Ganancia
        if supports_gain:
            self.gain_label.setText(f"{gain:+.2f} dB")
        else:
            self.gain_label.setText("0.00 dB")

        # Q
        self.q_label.setText(f"Q: {q:.3f}")

        # Silueta vectorial de la forma de filtro
        self.shape_icon.setShape(ftype, color)

        # Indicador Modo de Canal y Salida
        ch_short = {"LR": "LR", "L": "L", "R": "R", "Mid": "M", "Side": "S"}.get(cm, cm)
        to_short = {"A+B": "A+B", "A": "A", "B": "B"}.get(to_val, to_val)
        self.mode_badge.setText(f"{ch_short} | {to_short}")

        # Botones vectoriales
        p_c = color if enabled else "#686C75"
        self.power_btn.setColor(p_c)
        self.solo_btn.setColor("#00E5FF" if is_iso else "#8E95A5", active=is_iso)
        self.close_btn.setColor("#8E95A5")
        self.menu_btn.setColor("#8E95A5")

        self.reposition()
        self.show()
        self.raise_()
        self.update()

    def set_callout_opacity(self, op: float):
        self._current_opacity = float(np.clip(op, 0.0, 1.0))
        if hasattr(self, "_opacity_effect"):
            self._opacity_effect.setOpacity(self._current_opacity)

    def show_callout(self):
        self.set_callout_opacity(1.0)
        self.update_callout()
        self.show()
        self.raise_()

    def reposition(self):
        idx = self.overlay.selected_band_idx
        with self.overlay.eq_engine.lock:
            bands = self.overlay.eq_engine.bands
            if not (0 <= idx < len(bands)):
                self.hide()
                return
            b = bands[idx]
            vx = self.overlay.freq_to_x(b.freq)
            vy = self.overlay._band_node_y(b)

        scene_pt = self.overlay.eq_vb.mapViewToScene(QPointF(vx, vy))
        view_pt = self.overlay.plot_widget.mapFromScene(scene_pt)

        w = self.width()
        h = self.height()
        target_x = int(view_pt.x() - w * 0.5)
        # Flecha inferior apoyada 9px sobre el centro del nodo de la banda
        target_y = int(view_pt.y() - h - 9)

        pw = self.overlay.plot_widget
        clamped_x = max(10, min(pw.width() - w - 10, target_x))
        if target_y < 46:
            target_y = int(view_pt.y() + 18)

        self.move(clamped_x, target_y)


class GhostCurveItem(pg.GraphicsObject):
    """
    Ítem gráfico para la curva propuesta con degradado horizontal y bordes transparentes.
    Dibuja su trazo mediante QPainter estándar para soportar QLinearGradient en el QPen
    sin depender de OpenGL ni generar artefactos visuales.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.path = QPainterPath()
        self.bounds = QRectF()
        self.pen = QPen(Qt.PenStyle.NoPen)
        self._is_visible = False

    def boundingRect(self) -> QRectF:
        return self.bounds

    def setData(self, x, y, pen: QPen | None = None):
        if len(x) < 2:
            self.clear()
            return

        new_path = QPainterPath()
        new_path.moveTo(float(x[0]), float(y[0]))
        for xi, yi in zip(x[1:], y[1:]):
            new_path.lineTo(float(xi), float(yi))

        self.prepareGeometryChange()
        self.path = new_path
        if pen is not None:
            pen.setCosmetic(True)
            self.pen = pen
        elif self.pen.style() == Qt.PenStyle.NoPen:
            default_pen = QPen(QColor(245, 195, 85, 140), 1.6)
            default_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            default_pen.setCosmetic(True)
            self.pen = default_pen
        else:
            self.pen.setCosmetic(True)
        self.bounds = new_path.boundingRect().adjusted(-0.5, -2.0, 0.5, 2.0)
        self._is_visible = True
        self.update()

    def clear(self):
        if self._is_visible or not self.path.isEmpty():
            self.prepareGeometryChange()
            self.path = QPainterPath()
            self.bounds = QRectF()
            self.pen = QPen(Qt.PenStyle.NoPen)
            self._is_visible = False
            self.update()

    def isVisible(self) -> bool:
        return self._is_visible

    def paint(self, painter: QPainter, option, widget=None):
        if not self._is_visible or self.path.isEmpty() or self.pen.style() == Qt.PenStyle.NoPen:
            return
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(self.pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(self.path)
        painter.restore()


class CrossoverLineItem(pg.GraphicsObject):
    """
    Línea vertical interactiva y badge estilizado para el Crossover Digital de Salidas,
    renderizado en el ViewBox de EQ con 0% impacto de CPU/GPU (60 FPS estables).
    """
    sigPositionChanged = pyqtSignal(float)
    sigPositionChangeFinished = pyqtSignal()

    def __init__(self, freq_hz: float = 80.0, is_x_log: bool = True, parent=None):
        super().__init__(parent)
        self.freq_hz = float(freq_hz)
        self.is_x_log = bool(is_x_log)
        self.is_dragging = False
        self.is_hovered = False
        self._clip_lim = 36.0
        self.setZValue(30)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._pen_normal = QPen(QColor(255, 171, 64, 180), 1.5, Qt.PenStyle.SolidLine)
        self._pen_normal.setCosmetic(True)
        self._pen_hover = QPen(QColor(255, 213, 79, 255), 2.0, Qt.PenStyle.SolidLine)
        self._pen_hover.setCosmetic(True)
        self._font = QFont("Segoe UI", 9)
        self._font.setBold(True)
        from PyQt6.QtGui import QFontMetricsF
        self._fm = QFontMetricsF(self._font)
        self._cached_txt = ""
        self._cached_badge_w = 40.0
        self._cached_badge_h = 20.0
        self._badge_pixmap = None
        self._badge_pixmap_hover = None
        self._update_label_cache()
        self._update_pos()

    def _update_label_cache(self):
        self._cached_txt = f"XOVER: {int(round(self.freq_hz))} Hz" if self.freq_hz < 1000.0 else f"XOVER: {self.freq_hz/1000.0:.1f} kHz"
        tw = self._fm.horizontalAdvance(self._cached_txt)
        th = self._fm.height()
        self._cached_badge_w = tw + 14.0
        self._cached_badge_h = th + 6.0

    def set_config(self, freq_hz: float, is_x_log: bool = True, clip_lim: float = 36.0):
        self.freq_hz = float(np.clip(freq_hz, 20.0, 20000.0))
        self.is_x_log = bool(is_x_log)
        self._clip_lim = float(clip_lim)
        self._update_label_cache()
        self.prepareGeometryChange()
        self._update_pos()
        self.update()

    def _update_pos(self):
        x = float(np.log10(self.freq_hz)) if self.is_x_log else float(self.freq_hz)
        self.setPos(x, 0.0)

    def boundingRect(self) -> QRectF:
        return QRectF(-0.12, -self._clip_lim, 0.24, 2.0 * self._clip_lim)

    def paint(self, painter: QPainter, option, widget=None):
        if not self.isVisible():
            return

        pen = self._pen_hover if (self.is_hovered or self.is_dragging) else self._pen_normal
        painter.setPen(pen)
        painter.drawLine(QPointF(0.0, -self._clip_lim), QPointF(0.0, self._clip_lim))

        # Badge vectorial nativo en coordenadas de pantalla con métricas cacheadas (nitidez total en cualquier DPI)
        painter.save()
        tr = painter.transform()
        painter.resetTransform()

        center_scene_pt = tr.map(QPointF(0.0, 0.0))
        px = center_scene_pt.x()
        py = 52.0  # Posición vertical elegante y siempre visible justo debajo de la barra superior

        painter.setFont(self._font)
        badge_w = self._cached_badge_w
        badge_h = self._cached_badge_h
        badge_rect = QRectF(px - badge_w * 0.5, py - badge_h * 0.5, badge_w, badge_h)

        border_color = QColor(255, 213, 79, 230) if (self.is_hovered or self.is_dragging) else QColor(255, 171, 64, 180)
        painter.setPen(QPen(border_color, 1.2))
        painter.setBrush(QBrush(QColor(20, 24, 32, 230)))
        painter.drawRoundedRect(badge_rect, 5.0, 5.0)

        painter.setPen(border_color)
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, self._cached_txt)
        painter.restore()

    def hoverEnterEvent(self, ev):
        self.is_hovered = True
        self.setCursor(Qt.CursorShape.SizeHorCursor)
        self.update()

    def hoverLeaveEvent(self, ev):
        self.is_hovered = False
        self.unsetCursor()
        self.update()

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = True
            ev.accept()

    def mouseMoveEvent(self, ev):
        if self.is_dragging:
            scene_pos = ev.scenePos()
            vb = self.getViewBox()
            if vb is not None:
                view_pt = vb.mapSceneToView(scene_pos)
                view_x = view_pt.x()
            else:
                view_x = scene_pos.x()
            new_f = float(10.0 ** view_x if self.is_x_log else view_x)
            new_f = float(np.clip(new_f, 20.0, 20000.0))
            self.freq_hz = new_f
            self._update_label_cache()
            self._update_pos()
            self.sigPositionChanged.emit(new_f)
            self.update()
            ev.accept()

    def mouseReleaseEvent(self, ev):
        if self.is_dragging:
            self.is_dragging = False
            self.sigPositionChangeFinished.emit()
            self.update()
            ev.accept()


class FFTInspectorItem(QGraphicsItem):
    """
    Mini-inspector flotante dibujado DENTRO de la escena OpenGL de la FFT (zValue alto):
    línea 1 = Hz + dBFS instantáneo de la FFT; línea 2 opcional = ±dB del filtro propuesto.
    """

    def __init__(self):
        super().__init__()
        self.setZValue(1000)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._line1 = ""
        self._line2 = ""
        self._line2_color = QColor("#00E5FF")
        self._font = QFont("Segoe UI", 9)
        self._font.setBold(True)
        self._w = 120.0
        self._h = 22.0

    def set_content(self, line1: str, line2: str = "", line2_color: QColor | None = None):
        if line1 == self._line1 and line2 == self._line2 and (line2_color is None or line2_color == self._line2_color):
            return
        self.prepareGeometryChange()
        self._line1 = line1
        self._line2 = line2
        if line2_color is not None:
            self._line2_color = QColor(line2_color)
        from PyQt6.QtGui import QFontMetricsF
        fm = QFontMetricsF(self._font)
        w = fm.horizontalAdvance(line1)
        if line2:
            w = max(w, fm.horizontalAdvance(line2))
        self._w = w + 18.0
        self._h = (fm.height() * (2 if line2 else 1)) + 10.0
        self.update()

    def size(self):
        return self._w, self._h

    def boundingRect(self):
        return QRectF(0.0, 0.0, self._w, self._h)

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r = QRectF(0.5, 0.5, self._w - 1.0, self._h - 1.0)
        painter.setPen(QPen(QColor(255, 255, 255, 46), 1.0))
        painter.setBrush(QBrush(QColor(14, 16, 22, 185)))
        painter.drawRoundedRect(r, 6.0, 6.0)
        painter.setFont(self._font)
        from PyQt6.QtGui import QFontMetricsF
        lh = QFontMetricsF(self._font).height()
        painter.setPen(QColor("#E8ECF2"))
        painter.drawText(QRectF(9.0, 5.0, self._w - 18.0, lh), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self._line1)
        if self._line2:
            painter.setPen(self._line2_color)
            painter.drawText(QRectF(9.0, 5.0 + lh, self._w - 18.0, lh), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self._line2)


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
        self.hud_container = parent
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

        # Variables para control y animación de la curva propuesta (Ghost Curve)
        self._last_ghost_mouse_pos = QPoint(-9999, -9999)
        self._current_ghost_mouse_pos = QPoint(-9999, -9999)
        self._current_ghost_scene_pos = QPointF(-9999, -9999)
        self._last_mouse_move_time = time.perf_counter()
        self._mouse_velocity_px = 0.0
        self._ghost_still_since = time.perf_counter()
        self._ghost_unlocked = False
        self._ghost_fade = 0.0
        self._ghost_log_f = 3.0
        self._ghost_db = 0.0
        self._ghost_has_pos = False
        self._proposed_band = None

        self._current_y_floor = -132.0
        self._current_y_ceil = 12.0

        # Variables para auto-cierre del menú de nodo por distancia (> 55px por > 0.3s)
        self._callout_away_since = None

        self._rebuild_eval_freqs()
        self._init_secondary_viewbox()
        self._init_curves_and_nodes()

        # Mini-inspector FFT (Hz + dBFS + ±dB filtro propuesto) renderizado en la escena OpenGL
        self.fft_curves = []
        self.fft_inspector = FFTInspectorItem()
        self.plot_item.scene().addItem(self.fft_inspector)
        self.fft_inspector.hide()

        self.node_callout = BandNodeCallout(self, self.plot_widget)

        self._viewport.installEventFilter(self)
        self.setMouseTracking(True)
        self._viewport.setMouseTracking(True)

    def _effective_eq_vb_limit(self) -> float:
        """
        Dado que el ViewBox principal va de +12 dBFS (arriba, y=0) a -132 dBFS (abajo, y=h)
        con centro en -60 dBFS, escalamos eq_vb por 144/120 = 1.20 para que:
        +eq_range_db coincida exactamente con 0 dBFS y -eq_range_db coincida con -120 dBFS,
        dejando que las curvas de corte (Low/High Cut) bajen hasta -132 dBFS (-infty).
        """
        return self.eq_range_db * 1.20

    def update_y_range(self, y_floor: float, y_ceil: float):
        """
        Sincroniza el ViewBox de EQ (eq_vb) con el ViewBox principal de la FFT.
        Mantiene la correspondencia visual matemática exacta:
        v_EQ = (v_FFT + 60.0) * (self.eq_range_db / 60.0).
        """
        self._current_y_floor = float(y_floor)
        self._current_y_ceil = float(y_ceil)
        scale = self.eq_range_db / 60.0
        ymin = (self._current_y_floor + 60.0) * scale
        ymax = (self._current_y_ceil + 60.0) * scale
        self.eq_vb.setYRange(ymin, ymax, padding=0.0)
        self._update_vb_geometry()
        if hasattr(self, "zero_line"):
            self.update_eq_curves(force=True)

    def set_eq_range_db(self, range_db: float):
        new_r = 30.0 if float(range_db) >= 20.0 else 12.0
        if abs(new_r - self.eq_range_db) > 1e-3:
            self.eq_range_db = new_r
            self.update_y_range(self._current_y_floor, self._current_y_ceil)
            self.eq_range_changed.emit(self.eq_range_db)

    def toggle_eq_range_db(self):
        self.set_eq_range_db(30.0 if self.eq_range_db <= 12.0 else 12.0)
        return self.eq_range_db

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
        self.update_y_range(self._current_y_floor, self._current_y_ceil)

        self.plot_item.vb.sigResized.connect(self._update_vb_geometry)
        self._update_vb_geometry()

    def _update_vb_geometry(self):
        self.eq_vb.setGeometry(self.plot_item.vb.sceneBoundingRect())
        self.eq_vb.linkedViewChanged(self.plot_item.vb, self.eq_vb.XAxis)
        if hasattr(self, "node_callout"):
            self.node_callout.reposition()

    def _init_curves_and_nodes(self):
        self.zero_line = pg.PlotDataItem(
            pen=pg.mkPen(color=(255, 255, 255, 40), width=1.0)
        )
        self.zero_line.setZValue(5)
        self.eq_vb.addItem(self.zero_line)

        # Curva fantasma preliminar ("Ghost Preview Curve") con degradado horizontal y bordes transparentes
        self.ghost_curve = GhostCurveItem()
        self.ghost_curve.setZValue(12)
        self.eq_vb.addItem(self.ghost_curve)
        self._proposed_band = None

        self.band_fill_curves: list[pg.PlotDataItem] = []
        self.band_dyn_curves: list[pg.PlotDataItem] = []
        self.band_dyn_fills: list[pg.FillBetweenItem] = []

        self.dyn_limit_curve = pg.PlotDataItem(
            pen=pg.mkPen(color=(0, 229, 255, 175), width=1.4, style=Qt.PenStyle.DotLine),
            connect='finite',
        )
        self.dyn_limit_curve.setZValue(18)
        self.eq_vb.addItem(self.dyn_limit_curve)

        # Curva Maestra EQ Salida A (1-2) en dorado (#FFD54F por defecto)
        self._color_eq_a = "#FFD54F"
        self._color_eq_b = "#00E5FF"
        self.static_master_curve = pg.PlotDataItem(
            pen=pg.mkPen(color=self._color_eq_a, width=2.2),
            connect='finite',
        )
        self.static_master_curve.setZValue(20)
        self.eq_vb.addItem(self.static_master_curve)

        # Curva Maestra EQ Salida B (3-4) en cian eléctrico (#00E5FF por defecto)
        self.static_bus_b_curve = pg.PlotDataItem(
            pen=pg.mkPen(color=self._color_eq_b, width=2.0, style=Qt.PenStyle.DashLine),
            connect='finite',
        )
        self.static_bus_b_curve.setZValue(21)
        self.eq_vb.addItem(self.static_bus_b_curve)
        self._alt_isolate_dragging = False

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

    def _clear_ghost_curve(self):
        self._ghost_unlocked = False
        self._ghost_still_since = time.perf_counter()
        self._ghost_fade = 0.0
        self._ghost_has_pos = False
        if hasattr(self, "ghost_curve"):
            self.ghost_curve.clear()

    def _update_proposed_band_info(self, scene_pos: QPointF):
        rect = self.plot_item.vb.sceneBoundingRect()
        if not rect.contains(scene_pos):
            self._proposed_band = None
            return

        hit_idx = self._find_band_near_scene_pos(scene_pos, max_dist_px=22.0)
        if hit_idx >= 0:
            self._proposed_band = None
            return

        view_pt = self.eq_vb.mapSceneToView(scene_pos)
        f_cursor = self.x_to_freq(view_pt.x())
        db_cursor = float(np.clip(view_pt.y(), -24.0, 24.0))

        if f_cursor < 45.0:
            hypo_type = "Low Cut" if abs(db_cursor) < 3.0 else "Low Shelf"
            hypo_q = 0.707
        elif f_cursor > 10000.0:
            hypo_type = "High Cut" if abs(db_cursor) < 3.0 else "High Shelf"
            hypo_q = 0.707
        else:
            hypo_type = "Notch" if db_cursor < -10.0 else "Bell"
            hypo_q = 1.0

        self._proposed_band = {
            "type": hypo_type,
            "freq": f_cursor,
            "gain": db_cursor,
            "q": hypo_q,
        }

    def _render_ghost_curve(self, scene_pos: QPointF):
        self.update_ghost_tick()

    def _update_ghost_curve(self, scene_pos: QPointF, local_pt: QPoint):
        """Método de actualización forzada directa (usado en tests y previews)."""
        self._current_ghost_scene_pos = scene_pos
        self._current_ghost_mouse_pos = local_pt
        view_pt = self.eq_vb.mapSceneToView(scene_pos)
        target_f = self.x_to_freq(view_pt.x())
        self._ghost_log_f = float(np.log10(target_f)) if self.is_x_log else target_f
        self._ghost_db = float(np.clip(view_pt.y(), -24.0, 24.0))
        self._ghost_has_pos = True
        self._ghost_unlocked = True
        self._ghost_still_since = time.perf_counter() - 2.0
        self._ghost_fade = 1.0
        self._last_mouse_move_time = time.perf_counter() - 1.0
        self._mouse_velocity_px = 0.0
        self.update_ghost_tick()

    def _cursor_in_blocked_zone(self, local_pt: QPoint) -> bool:
        if hasattr(self, "hud_container") and self.hud_container is not None:
            hud = getattr(self.hud_container, "floating_hud", None)
            if hud is not None and hud.isVisible() and hud.geometry().contains(local_pt):
                return True
            top_bar = getattr(self.hud_container, "top_bar", None)
            if top_bar is not None and top_bar.isVisible() and top_bar.geometry().contains(local_pt):
                return True
            scope = getattr(self.hud_container, "scope_overlay", None)
            if scope is not None and scope.isVisible() and scope.geometry().contains(local_pt):
                return True
        if hasattr(self, "node_callout") and self.node_callout.isVisible() and self.node_callout.geometry().contains(local_pt):
            return True
        return False

    def _update_fft_inspector(self):
        """Actualiza el mini-inspector OpenGL junto al cursor: Hz + dBFS de la FFT (+ ±dB del filtro propuesto)."""
        insp = getattr(self, "fft_inspector", None)
        if insp is None:
            return
        scene_pos = self._current_ghost_scene_pos
        local_pt = self._current_ghost_mouse_pos
        rect = self.plot_item.vb.sceneBoundingRect()
        if (
            local_pt.x() < -9000
            or self.dragging_band_idx >= 0
            or not rect.contains(scene_pos)
            or self._find_band_near_scene_pos(scene_pos, max_dist_px=22.0) >= 0
            or self._cursor_in_blocked_zone(local_pt)
        ):
            if insp.isVisible():
                insp.hide()
            return

        main_pt = self.plot_item.vb.mapSceneToView(scene_pos)
        freq_hz = float(10.0 ** main_pt.x()) if self.is_x_log else float(main_pt.x())
        freq_hz = max(1.0, freq_hz)
        f_txt = f"{freq_hz / 1000.0:.2f} kHz" if freq_hz >= 1000.0 else f"{freq_hz:.1f} Hz"

        fft_txt = "-- dBFS"
        for curve in self.fft_curves:
            try:
                if not curve.isVisible():
                    continue
                xd, yd = curve.getData()
                if xd is None or yd is None or len(xd) < 2:
                    continue
                val = float(np.interp(main_pt.x(), xd, yd))
                fft_txt = f"{val:.1f} dBFS"
                break
            except Exception:
                continue
        line1 = f"{f_txt}  |  {fft_txt}"

        line2 = ""
        col = None
        if self._ghost_unlocked and self._ghost_fade > 0.05:
            g_db = float(np.clip(self._ghost_db, -24.0, 24.0))
            g_f = float(10.0 ** self._ghost_log_f) if self.is_x_log else float(self._ghost_log_f)
            if g_f < 45.0:
                g_type = "Low Cut" if abs(g_db) < 3.0 else "Low Shelf"
            elif g_f > 10000.0:
                g_type = "High Cut" if abs(g_db) < 3.0 else "High Shelf"
            else:
                g_type = "Notch" if g_db < -10.0 else "Bell"

            from eq_dsp import BAND_COLORS
            with self.eq_engine.lock:
                nb = len(self.eq_engine.bands)
            col = QColor(BAND_COLORS[nb % len(BAND_COLORS)])

            if abs(g_db) < 0.05:
                line2 = f"{g_type}: 0.0 dB"
            elif g_db > 0:
                line2 = f"{g_type}: +{g_db:.1f} dB"
            else:
                line2 = f"{g_type}: {g_db:.1f} dB"

        insp.set_content(line1, line2, col)
        w, h = insp.size()
        x = scene_pos.x() + 14.0
        y = scene_pos.y() - 24.0 - h
        if y < rect.top() + 4.0:
            y = scene_pos.y() + 18.0
        if x + w > rect.right() - 4.0:
            x = scene_pos.x() - w - 14.0
        if x < rect.left() + 4.0:
            x = rect.left() + 4.0
        if y + h > rect.bottom() - 4.0:
            y = rect.bottom() - 4.0 - h
        insp.setPos(x, y)
        if not insp.isVisible():
            insp.show()

    def update_ghost_tick(self):
        self._update_fft_inspector()
        # --- 1. ZONA SEGURA AMPLIA PARA EL MENÚ DE NODO (Nodo + Callout + Margen 45px) ---
        if hasattr(self, "node_callout") and self.node_callout.isVisible():
            node_pt = None
            with self.eq_engine.lock:
                bands = self.eq_engine.bands
                idx = self.selected_band_idx
                if 0 <= idx < len(bands):
                    b = bands[idx]
                    vx = self.freq_to_x(b.freq)
                    vy = self._band_node_y(b)
                    scene_pt = self.eq_vb.mapViewToScene(QPointF(vx, vy))
                    node_pt = self.plot_widget.mapFromScene(scene_pt)

            if node_pt is not None:
                # Unión Geométrica: círculo del nodo (radio 35px) + rectángulo del callout + margen 45px
                node_rect = QRectF(node_pt.x() - 35.0, node_pt.y() - 35.0, 70.0, 70.0)
                callout_rect = QRectF(self.node_callout.geometry())
                combined_rect = callout_rect.united(node_rect)
                safe_zone_rect = combined_rect.adjusted(-45.0, -45.0, 45.0, 45.0)

                # Comprobar si el cursor está dentro de safe_zone_rect
                inside_safe = False
                cursor_pt = self._current_ghost_mouse_pos
                if cursor_pt.x() > -9000 and safe_zone_rect.contains(QPointF(cursor_pt)):
                    inside_safe = True

                if not inside_safe:
                    try:
                        global_pos = QCursor.pos()
                        widget_cursor = QPointF(self.plot_widget.mapFromGlobal(global_pos))
                        if safe_zone_rect.contains(widget_cursor):
                            inside_safe = True
                    except Exception:
                        pass

                # Condición de Permanencia
                if inside_safe:
                    self._callout_away_since = None
                    if self.node_callout._current_opacity < 1.0:
                        self.node_callout.set_callout_opacity(1.0)
                else:
                    if self._callout_away_since is None:
                        self._callout_away_since = time.perf_counter()
                    elapsed = time.perf_counter() - self._callout_away_since
                    if elapsed >= 0.35:
                        new_op = max(0.0, self.node_callout._current_opacity - 0.14)
                        self.node_callout.set_callout_opacity(new_op)
                        if new_op <= 0.0:
                            self.node_callout.hide()
                            self._callout_away_since = None
            else:
                self.node_callout.hide()
                self._callout_away_since = None

        # --- 2. GHOST CURVE ANIMATION & 2-PHASE STATE MACHINE ---
        if self.dragging_band_idx >= 0:
            if self._ghost_fade > 0.0 or self.ghost_curve.isVisible():
                self._clear_ghost_curve()
            return

        scene_pos = self._current_ghost_scene_pos
        local_pt = self._current_ghost_mouse_pos

        rect = self.plot_item.vb.sceneBoundingRect()
        invalid_zone = False

        if not rect.contains(scene_pos):
            invalid_zone = True
        else:
            hit_idx = self._find_band_near_scene_pos(scene_pos, max_dist_px=22.0)
            if hit_idx >= 0:
                invalid_zone = True
            elif hasattr(self, "hud_container") and self.hud_container is not None:
                hud = getattr(self.hud_container, "floating_hud", None)
                if hud is not None and hud.isVisible() and hud.geometry().contains(local_pt):
                    invalid_zone = True
                top_bar = getattr(self.hud_container, "top_bar", None)
                if top_bar is not None and top_bar.isVisible() and top_bar.geometry().contains(local_pt):
                    invalid_zone = True
            if not invalid_zone and hasattr(self, "node_callout") and self.node_callout.isVisible() and self.node_callout.geometry().contains(local_pt):
                invalid_zone = True

        if invalid_zone:
            if self._ghost_unlocked:
                self._ghost_fade = max(0.0, self._ghost_fade - 0.12)
                if self._ghost_fade <= 0.01:
                    self._ghost_unlocked = False
                    self._ghost_fade = 0.0
                    self._ghost_still_since = time.perf_counter()
                    if hasattr(self, "ghost_curve") and self.ghost_curve.isVisible():
                        self.ghost_curve.clear()
                    return
            else:
                self._ghost_still_since = time.perf_counter()
                self._ghost_fade = 0.0
                if hasattr(self, "ghost_curve") and self.ghost_curve.isVisible():
                    self.ghost_curve.clear()
                return

        # Coordenadas destino de cursor
        view_pt = self.eq_vb.mapSceneToView(scene_pos)
        target_f = self.x_to_freq(view_pt.x())
        target_log_f = float(np.log10(target_f)) if self.is_x_log else target_f
        target_db = float(np.clip(view_pt.y(), -24.0, 24.0))

        # Detección de velocidad efectiva del mouse
        time_since_move = time.perf_counter() - self._last_mouse_move_time
        if time_since_move > 0.07:
            v = 0.0
        else:
            v = self._mouse_velocity_px

        # Máquina de estados de 2 fases
        if not self._ghost_unlocked:
            # Fase 1: Estado Oculto (self._ghost_unlocked == False)
            self._ghost_fade = 0.0
            if v > 2.5:
                self._ghost_still_since = time.perf_counter()

            elapsed_still = time.perf_counter() - self._ghost_still_since
            if elapsed_still >= 1.0:
                # Condición de Activación: 1.0 segundo continuo quieto (v <= 2.5)
                self._ghost_unlocked = True
                self._ghost_log_f = target_log_f
                self._ghost_db = target_db
                self._ghost_has_pos = True
                self._ghost_fade = 0.06
            else:
                if hasattr(self, "ghost_curve") and self.ghost_curve.isVisible():
                    self.ghost_curve.clear()
                return
        else:
            # Fase 2: Estado Visible / Activo (self._ghost_unlocked == True)
            if v <= 2.0:
                # Si permanece quieto (v <= 2.0): fade-in suave hasta 1.0
                self._ghost_fade = min(1.0, self._ghost_fade + 0.06)
                lerp_speed = 0.28
                self._ghost_log_f += (target_log_f - self._ghost_log_f) * lerp_speed
                self._ghost_db += (target_db - self._ghost_db) * lerp_speed
            elif 2.0 < v <= 14.0:
                # Si mueve el cursor lentamente (2.0 < v <= 14.0): permanece a media opacidad (~0.40) y migra suavemente
                target_crawl = 0.40
                if self._ghost_fade > target_crawl:
                    self._ghost_fade = max(target_crawl, self._ghost_fade - 0.05)
                else:
                    self._ghost_fade = min(target_crawl, self._ghost_fade + 0.05)
                lerp_speed = min(0.42, 0.26 + 0.008 * v)
                self._ghost_log_f += (target_log_f - self._ghost_log_f) * lerp_speed
                self._ghost_db += (target_db - self._ghost_db) * lerp_speed
            else:
                # Movimiento rápido (v > 14.0): reduce la opacidad rápidamente
                self._ghost_fade = max(0.0, self._ghost_fade - 0.12)
                if self._ghost_fade <= 0.01:
                    self._ghost_unlocked = False
                    self._ghost_fade = 0.0
                    self._ghost_still_since = time.perf_counter()
                    if hasattr(self, "ghost_curve") and self.ghost_curve.isVisible():
                        self.ghost_curve.clear()
                    return
                else:
                    lerp_speed = 0.35
                    self._ghost_log_f += (target_log_f - self._ghost_log_f) * lerp_speed
                    self._ghost_db += (target_db - self._ghost_db) * lerp_speed

        if self._ghost_fade <= 0.005:
            if hasattr(self, "ghost_curve") and self.ghost_curve.isVisible():
                self.ghost_curve.clear()
            return

        # Renderizar usando coordenadas suavizadas
        current_f = float(10.0 ** self._ghost_log_f) if self.is_x_log else float(self._ghost_log_f)
        current_f = float(np.clip(current_f, 10.0, self.nyquist * 0.99))
        current_db = float(np.clip(self._ghost_db, -24.0, 24.0))

        if current_f < 45.0:
            hypo_type = "Low Cut" if abs(current_db) < 3.0 else "Low Shelf"
            hypo_q = 0.707
        elif current_f > 10000.0:
            hypo_type = "High Cut" if abs(current_db) < 3.0 else "High Shelf"
            hypo_q = 0.707
        else:
            hypo_type = "Notch" if current_db < -10.0 else "Bell"
            hypo_q = 1.0

        self._proposed_band = {
            "type": hypo_type,
            "freq": current_f,
            "gain": current_db,
            "q": hypo_q,
        }

        temp_b = EQBand(
            filter_type=hypo_type,
            freq=current_f,
            gain_db=current_db,
            q=hypo_q,
            sr=self.eq_engine.sr,
        )
        clip_lim = max(36.0, self._effective_eq_vb_limit() * 1.15)
        resp = np.clip(temp_b.get_static_freq_response_db(self.eval_z1, self.eval_z2), -clip_lim, clip_lim)

        if self.is_x_log:
            u_c = (self._ghost_log_f - np.log10(10.0)) / (np.log10(self.nyquist) - np.log10(10.0))
            x_min_view = float(np.log10(10.0))
            x_max_view = float(np.log10(self.nyquist))
        else:
            u_c = current_f / self.nyquist
            x_min_view = 0.0
            x_max_view = float(self.nyquist)

        u_c = float(np.clip(u_c, 0.05, 0.95))

        peak_alpha = int(145 * self._ghost_fade)
        mid_alpha = int(45 * self._ghost_fade)

        grad = QLinearGradient(x_min_view, 0.0, x_max_view, 0.0)
        s1 = float(np.clip(u_c - 0.28, 0.005, 0.95))
        s2 = float(np.clip(u_c - 0.10, s1 + 0.005, 0.96))
        s3 = float(np.clip(u_c, s2 + 0.005, 0.97))
        s4 = float(np.clip(u_c + 0.10, s3 + 0.005, 0.98))
        s5 = float(np.clip(u_c + 0.28, s4 + 0.005, 0.99))

        grad.setColorAt(0.00, QColor(180, 180, 190, 0))
        grad.setColorAt(s1, QColor(180, 180, 190, 0))
        grad.setColorAt(s2, QColor(200, 200, 210, mid_alpha))
        grad.setColorAt(s3, QColor(245, 195, 85, peak_alpha))
        grad.setColorAt(s4, QColor(200, 200, 210, mid_alpha))
        grad.setColorAt(s5, QColor(180, 180, 190, 0))
        grad.setColorAt(1.00, QColor(180, 180, 190, 0))

        pen = QPen(QBrush(grad), 1.6)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setCosmetic(True)

        self.ghost_curve.setData(self.eval_x, resp, pen)

    def set_axes_mode(self, is_x_log: bool, sr: float):
        self.is_x_log = bool(is_x_log)
        self.eq_engine.sr = float(sr)
        self._rebuild_eval_freqs()
        self.update_y_range(self._current_y_floor, self._current_y_ceil)

    def set_curve_colors(self, color_a: str = "#FFD54F", color_b: str = "#00E5FF"):
        """Permite personalizar interactivamente los colores de las curvas maestras de EQ (Out A y Out B)."""
        self._color_eq_a = str(color_a)
        self._color_eq_b = str(color_b)
        if hasattr(self, "static_master_curve"):
            self.static_master_curve.setPen(pg.mkPen(color=self._color_eq_a, width=2.2))
        if hasattr(self, "static_bus_b_curve"):
            self.static_bus_b_curve.setPen(pg.mkPen(color=self._color_eq_b, width=2.0, style=Qt.PenStyle.DashLine))

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
                pos_pt = event.position().toPoint()
                scene_pos = self.plot_widget.mapToScene(pos_pt)
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
                    if self._proposed_band is not None:
                        p_type = self._proposed_band["type"]
                        p_freq = self._proposed_band["freq"]
                        p_gain = self._proposed_band["gain"]
                        p_q = self._proposed_band["q"]
                    else:
                        view_pt = self.eq_vb.mapSceneToView(scene_pos)
                        p_freq = self.x_to_freq(view_pt.x())
                        p_gain = float(np.clip(view_pt.y(), -24.0, 24.0))
                        p_type = "Bell"
                        p_q = 1.0

                    self._clear_ghost_curve()
                    self.eq_engine.add_band(freq=p_freq, gain_db=p_gain, q=p_q, filter_type=p_type)
                    new_idx = len(self.eq_engine.bands) - 1
                    self.selected_band_idx = new_idx
                    self.dragging_band_idx = new_idx
                    self.band_selected.emit(new_idx)
                    self.band_modified.emit()
                    self.update_eq_curves(force=True)
                    if hasattr(self, "node_callout"):
                        self.node_callout.show_callout()
                    return True

        elif etype == QEvent.Type.MouseButtonPress:
            pos_pt = event.position().toPoint()
            scene_pos = self.plot_widget.mapToScene(pos_pt)
            self._clear_ghost_curve()
            if not self.plot_item.vb.sceneBoundingRect().contains(scene_pos):
                return False

            hit_idx = self._find_band_near_scene_pos(scene_pos)
            if event.button() == Qt.MouseButton.LeftButton:
                if hit_idx >= 0:
                    self.selected_band_idx = hit_idx
                    self.dragging_band_idx = hit_idx
                    with self.eq_engine.lock:
                        self._drag_start_q = self.eq_engine.bands[hit_idx].q
                    if event.modifiers() & Qt.KeyboardModifier.AltModifier:
                        self._alt_isolate_dragging = True
                        self.eq_engine.set_isolated_band(hit_idx)
                    self._drag_start_scene_y = scene_pos.y()
                    self.band_selected.emit(hit_idx)
                    self.update_eq_curves(force=True)
                    if hasattr(self, "node_callout"):
                        self.node_callout.show_callout()
                    return True
                else:
                    if hasattr(self, "node_callout") and self.node_callout.isVisible():
                        self.node_callout.hide()
                        self._callout_away_since = None
            elif event.button() == Qt.MouseButton.RightButton:
                if hit_idx >= 0:
                    self.selected_band_idx = hit_idx
                    self.band_selected.emit(hit_idx)
                    self._show_node_context_menu(hit_idx, event.globalPosition().toPoint())
                    return True

        elif etype == QEvent.Type.MouseMove:
            pos_pt = event.position().toPoint()
            scene_pos = self.plot_widget.mapToScene(pos_pt)

            # Medición de velocidad del cursor en px/frame
            if self._last_ghost_mouse_pos.x() > -9000:
                dx = pos_pt.x() - self._last_ghost_mouse_pos.x()
                dy = pos_pt.y() - self._last_ghost_mouse_pos.y()
                self._mouse_velocity_px = math.hypot(dx, dy)
            else:
                self._mouse_velocity_px = 0.0

            if self._mouse_velocity_px > 2.5:
                self._ghost_still_since = time.perf_counter()

            self._last_ghost_mouse_pos = pos_pt
            self._last_mouse_move_time = time.perf_counter()
            self._current_ghost_mouse_pos = pos_pt
            self._current_ghost_scene_pos = scene_pos

            if self.dragging_band_idx >= 0:
                self._clear_ghost_curve()
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
                if hasattr(self, "node_callout"):
                    self.node_callout.reposition()
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

                if hit_idx < 0:
                    self._update_proposed_band_info(scene_pos)
                else:
                    self._proposed_band = None
                    self._clear_ghost_curve()
                self._update_fft_inspector()

        elif etype == QEvent.Type.Leave:
            self._current_ghost_mouse_pos = QPoint(-9999, -9999)
            self._current_ghost_scene_pos = QPointF(-9999, -9999)
            self._clear_ghost_curve()
            if hasattr(self, "fft_inspector") and self.fft_inspector.isVisible():
                self.fft_inspector.hide()

        elif etype == QEvent.Type.MouseButtonRelease:
            if event.button() == Qt.MouseButton.LeftButton and self.dragging_band_idx >= 0:
                self.dragging_band_idx = -1
                if self._alt_isolate_dragging:
                    self._alt_isolate_dragging = False
                    self.eq_engine.set_isolated_band(None)
                    self.band_modified.emit()
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

        act_enable = menu.addAction("Banda Activa")
        act_enable.setCheckable(True)
        act_enable.setChecked(band.enabled)
        act_enable.triggered.connect(lambda checked: self._toggle_band_enabled(band_idx, checked))

        act_dyn = menu.addAction("Activar Dynamic EQ")
        act_dyn.setCheckable(True)
        act_dyn.setChecked(band.dyn_enabled)
        act_dyn.triggered.connect(lambda checked: self._toggle_band_dyn(band_idx, checked))

        act_solo = menu.addAction("Isolate (Solo de Frecuencia)")
        act_solo.setCheckable(True)
        act_solo.setChecked(band.solo or self.eq_engine.isolated_band_index == band_idx)
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

        ch_menu = menu.addMenu("Modo de Canal (LR / L / R / Mid / Side)")
        ch_labels = {
            "LR": "Stereo (LR)",
            "L": "Left (L)",
            "R": "Right (R)",
            "Mid": "Mid (M)",
            "Side": "Side (S)",
        }
        for cm in CHANNEL_MODES:
            act_cm = ch_menu.addAction(ch_labels.get(cm, cm))
            act_cm.setCheckable(True)
            act_cm.setChecked(band.channel_mode == cm)
            act_cm.triggered.connect(lambda _=False, val=cm: self._set_band_channel_mode(band_idx, val))

        out_menu = menu.addMenu("Bus de Salida (Out A / Out B)")
        out_labels = {
            "A+B": "A+B (Ambas Salidas 1-4)",
            "A": "Solo Salida A (Canales 1-2)",
            "B": "Solo Salida B (Canales 3-4)",
        }
        for to_val in TARGET_OUTPUTS:
            act_to = out_menu.addAction(out_labels.get(to_val, to_val))
            act_to.setCheckable(True)
            act_to.setChecked(band.target_output == to_val)
            act_to.triggered.connect(lambda _=False, val=to_val: self._set_band_target_output(band_idx, val))

        color_menu = menu.addMenu("Color de Banda")
        for label, hex_val in STUDIO_BAND_COLORS:
            act_c = color_menu.addAction(_make_color_icon(hex_val, 12), label)
            act_c.triggered.connect(lambda _=False, hv=hex_val: self._set_band_color(band_idx, hv))
        color_menu.addSeparator()
        act_custom_c = color_menu.addAction("Personalizado...")
        act_custom_c.triggered.connect(lambda: self._pick_custom_band_color(band_idx))

        menu.addSeparator()
        act_del = menu.addAction("Eliminar Banda")
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
        self.eq_engine.set_isolated_band(idx if checked else None)
        self.band_modified.emit()
        self.update_eq_curves(force=True)

    def _set_band_channel_mode(self, idx: int, mode: str):
        with self.eq_engine.lock:
            if 0 <= idx < len(self.eq_engine.bands) and mode in CHANNEL_MODES:
                b = self.eq_engine.bands[idx]
                if b.channel_mode != mode:
                    b.channel_mode = mode
                    b.reset_state()
                    self.eq_engine.mark_dirty()
        self.band_modified.emit()
        self.update_eq_curves(force=True)

    def _set_band_target_output(self, idx: int, target_out: str):
        with self.eq_engine.lock:
            if 0 <= idx < len(self.eq_engine.bands) and target_out in TARGET_OUTPUTS:
                b = self.eq_engine.bands[idx]
                if b.target_output != target_out:
                    b.target_output = target_out
                    b.reset_state()
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

                is_active = is_sel or abs(b.gain_db) > 0.05 or not b.supports_gain or b.is_dynamic_active

                if is_active:
                    pen_alpha = 220 if is_sel else (140 if b.enabled else 60)
                    fill_alpha = 48 if is_sel else (24 if b.enabled else 10)
                    pen_w = 2.0 if is_sel else 1.4
                    st_curve.setPen(pg.mkPen(color=(c.red(), c.green(), c.blue(), pen_alpha), width=pen_w))
                    st_curve.setBrush(pg.mkBrush(c.red(), c.green(), c.blue(), fill_alpha))
                else:
                    # Inactiva (0 dB): trazo muy delgado (1.0px) y opacidad extremadamente baja (alpha=40)
                    pen_alpha = 40 if b.enabled else 20
                    st_curve.setPen(pg.mkPen(color=(c.red(), c.green(), c.blue(), pen_alpha), width=1.0))
                    st_curve.setBrush(pg.mkBrush(0, 0, 0, 0))

                # Curva 100% continua de punta a punta (sin recortes ni np.nan)
                st_curve.setData(x, resp_static)

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
                    max_dyn_resp = np.clip(sel_band.get_max_dynamic_freq_response_db(z1, z2), -clip_lim, clip_lim)
                    self.dyn_limit_curve.setData(x, max_dyn_resp)
                else:
                    self.dyn_limit_curve.setData([], [])
            else:
                self.dyn_limit_curve.setData([], [])

            static_a_db, _ = self.eq_engine.get_combined_responses_db(z1, z2, target_bus="A")
            static_b_db, _ = self.eq_engine.get_combined_responses_db(z1, z2, target_bus="B")

            # Independencia absoluta de la perilla OUT:
            # Nunca se suma al cálculo de los dB visuales en el gráfico.
            # Las curvas pasan exactamente por el centro físico de cada nodo y se anclan en 0.0 dB.
            if not self.eq_engine.bypass:
                out_gain = self.eq_engine.output_gain_db
                static_a_db = static_a_db - out_gain
                static_b_db = static_b_db - out_gain

            static_clipped_a = np.clip(static_a_db, -clip_lim, clip_lim)
            static_clipped_b = np.clip(static_b_db, -clip_lim, clip_lim)

            # Curva Maestra Bus A (Amarilla): 100% continua de punta a punta, sin np.nan
            self.static_master_curve.setData(x, static_clipped_a)

            # Curva Maestra Bus B (Cian): continua de punta a punta si difiere de Bus A o si hay bandas asignadas a Bus B
            b_differs = np.any(np.abs(static_clipped_b - static_clipped_a) > 0.02)
            has_b_routed = any(getattr(b, 'target_output', 'A') in ('B', 'A+B') for b in bands)
            if b_differs or has_b_routed:
                self.static_bus_b_curve.setData(x, static_clipped_b)
            else:
                self.static_bus_b_curve.setData([], [])

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
                ch_str = f" &bull; [{b.channel_mode}]" if b.channel_mode != "LR" else ""
                bus_str = f" &bull; [Out {b.target_output}]" if b.target_output != "A+B" else ""
                iso_str = " &bull; ISOLATE" if (b.solo or self.eq_engine.isolated_band_index == active_idx) else ""
                html = (
                    f"<div style='background-color: rgba(24, 24, 28, 235); "
                    f"border: 1px solid {b.color}; border-radius: 5px; padding: 2px 7px; "
                    f"font-family: Segoe UI, sans-serif; font-size: 10px; color: #ECEFF4;'>"
                    f"<b style='color:{b.color};'>Band {active_idx + 1}</b> "
                    f"{freq_str} &bull; {b.gain_db:+.1f} dB &bull; Q {b.q:.2f}{ch_str}{bus_str}{dyn_str}{iso_str}</div>"
                )
                self.floating_tooltip.hide()
            else:
                self.floating_tooltip.hide()

            if hasattr(self, "node_callout"):
                if 0 <= self.selected_band_idx < len(bands):
                    if force or self.node_callout.isVisible():
                        self.node_callout.update_callout()
                else:
                    self.node_callout.hide()


class HUDOpenGLBackgroundItem(QGraphicsItem):
    """
    Fondo translúcido del HUD con Gaussian Blur en tiempo real (1/8-scale separable blur)
    renderizado en la escena OpenGL de la FFT sin bloqueos glReadPixels:
    - Rasteriza la región bajo el HUD en un mini-buffer (1/8 resolución) y aplica un kernel
      Gaussiano 1D separable en NumPy (< 0.05 ms).
    - Superpone el vidrio oscuro translúcido rgba(14, 16, 22, 138) y el borde/resplandor
      coloreado de la banda activa.
    """

    _GAUSS_KERNEL = np.array([0.0545, 0.2442, 0.4026, 0.2442, 0.0545], dtype=np.float32)

    def __init__(self, overlay=None, plot_widget=None):
        super().__init__()
        self.setZValue(25)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.overlay = overlay
        self.plot_widget = plot_widget
        self._rect = QRectF(0.0, 0.0, 700.0, 196.0)
        self._is_minimized = False
        self._active_color = QColor("#00E5FF")
        self._gain_center_x = 276.0
        self._cached_path = QPainterPath()
        self._blur_pixmap = None
        self._last_blur_time = 0.0
        self._rebuild_path()

    def boundingRect(self) -> QRectF:
        return self._rect.adjusted(-6.0, -6.0, 6.0, 6.0)

    def set_state(self, scene_pos: QPointF, width: float, height: float, is_minimized: bool, active_color: QColor, gain_cx: float):
        pos_changed = (self.pos() - scene_pos).manhattanLength() > 0.5
        self.setPos(scene_pos)
        c = QColor(active_color) if active_color else QColor("#00E5FF")
        geom_changed = (
            abs(self._rect.width() - width) > 0.5
            or abs(self._rect.height() - height) > 0.5
            or self._is_minimized != is_minimized
            or abs(self._gain_center_x - gain_cx) > 0.5
        )
        if geom_changed:
            self.prepareGeometryChange()
            self._rect = QRectF(0.0, 0.0, width, height)
            self._is_minimized = is_minimized
            self._gain_center_x = gain_cx
            self._active_color = c
            self._rebuild_path()
            self._last_blur_time = 0.0
            self.update()
        elif self._active_color != c or pos_changed:
            self._active_color = c
            if pos_changed:
                self._last_blur_time = 0.0
            self.update()

    def _rebuild_path(self):
        self._cached_path = QPainterPath()
        w = float(self._rect.width())
        h = float(self._rect.height())
        pad = 2.0
        r_corner = 10.0 if self._is_minimized else 18.0
        x_left = pad
        x_right = w - pad
        y_bot = h - pad

        if self._is_minimized:
            self._cached_path.addRoundedRect(QRectF(x_left, pad, w - 2 * pad, h - 2 * pad), r_corner, r_corner)
            return

        y_top = 34.0
        y_peak = 2.5
        arch_cx = self._gain_center_x
        arch_half_w = 104.0
        ax1 = max(x_left + r_corner + 10.0, arch_cx - arch_half_w)
        ax2 = min(x_right - r_corner - 10.0, arch_cx + arch_half_w)

        p = self._cached_path
        p.moveTo(x_left + r_corner, y_top)
        p.lineTo(ax1, y_top)
        p.cubicTo(
            QPointF(ax1 + 38.0, y_top),
            QPointF(arch_cx - 42.0, y_peak),
            QPointF(arch_cx, y_peak),
        )
        p.cubicTo(
            QPointF(arch_cx + 42.0, y_peak),
            QPointF(ax2 - 38.0, y_top),
            QPointF(ax2, y_top),
        )
        p.lineTo(x_right - r_corner, y_top)
        p.quadTo(QPointF(x_right, y_top), QPointF(x_right, y_top + r_corner))
        p.lineTo(x_right, y_bot - r_corner)
        p.quadTo(QPointF(x_right, y_bot), QPointF(x_right - r_corner, y_bot))
        p.lineTo(x_left + r_corner, y_bot)
        p.quadTo(QPointF(x_left, y_bot), QPointF(x_left, y_bot - r_corner))
        p.lineTo(x_left, y_top + r_corner)
        p.quadTo(QPointF(x_left, y_top), QPointF(x_left + r_corner, y_top))
        p.closeSubpath()

    @staticmethod
    def _separable_gaussian_blur(arr: np.ndarray, k: np.ndarray) -> np.ndarray:
        # Dos pasadas separables X + Y para lograr un radio efectivo amplio con costo casi nulo
        for _ in range(2):
            p = np.pad(arr, ((0, 0), (2, 2), (0, 0)), mode="edge")
            arr = (
                k[0] * p[:, 0:-4, :]
                + k[1] * p[:, 1:-3, :]
                + k[2] * p[:, 2:-2, :]
                + k[3] * p[:, 3:-1, :]
                + k[4] * p[:, 4:, :]
            )
            p = np.pad(arr, ((2, 2), (0, 0), (0, 0)), mode="edge")
            arr = (
                k[0] * p[0:-4, :, :]
                + k[1] * p[1:-3, :, :]
                + k[2] * p[2:-2, :, :]
                + k[3] * p[3:-1, :, :]
                + k[4] * p[4:, :, :]
            )
        return arr

    def _update_gaussian_blur_texture(self):
        now = time.perf_counter()
        if self._blur_pixmap is not None and (now - self._last_blur_time) < 0.032:
            return
        self._last_blur_time = now

        w = float(self._rect.width())
        h = float(self._rect.height())
        if w <= 4.0 or h <= 4.0 or self.overlay is None or self.plot_widget is None:
            return

        bw = max(16, int(round(w / 8.0)))
        bh = max(6, int(round(h / 8.0)))

        img = QImage(bw, bh, QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(QColor(14, 16, 22, 255))

        ip = QPainter(img)
        ip.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        sx = float(bw) / w
        sy = float(bh) / h
        hud_pos = self.pos()
        hx = float(hud_pos.x())
        hy = float(hud_pos.y())

        try:
            vb = self.plot_widget.plotItem.vb
            vb_rect = vb.sceneBoundingRect()
            vx = float(vb_rect.left())
            vy = float(vb_rect.top())
            vw = max(1.0, float(vb_rect.width()))
            vh = max(1.0, float(vb_rect.height()))
            view_range = vb.viewRange()
            xmin, xmax = float(view_range[0][0]), float(view_range[0][1])
            ymin, ymax = float(view_range[1][0]), float(view_range[1][1])
            dx = max(1e-6, xmax - xmin)
            dy = max(1e-6, ymax - ymin)

            # 1. Líneas de grilla bajo el HUD
            grid_pen = QPen(QColor(55, 62, 78, 120), 1.0)
            ip.setPen(grid_pen)
            for db_grid in (-18.0, -36.0, -54.0, -72.0, -90.0, -108.0):
                if ymin <= db_grid <= ymax:
                    gy_scene = vy + (ymax - db_grid) / dy * vh
                    ly = (gy_scene - hy) * sy
                    if 0.0 <= ly <= bh:
                        ip.drawLine(QPointF(0.0, ly), QPointF(float(bw), ly))

            # 2. Curvas FFT activas bajo el HUD
            for entry in getattr(self.overlay, "fft_curves", []):
                if isinstance(entry, dict):
                    curve_item = entry.get("item")
                    c_hex = entry.get("color", "#00E5FF")
                else:
                    curve_item = entry
                    c_hex = "#00E5FF"
                    try:
                        pen_obj = curve_item.opts.get("pen")
                        if pen_obj is not None:
                            c_hex = pg.mkPen(pen_obj).color()
                    except Exception:
                        pass
                if curve_item is None or not curve_item.isVisible():
                    continue
                xd, yd = curve_item.getData()
                if xd is None or yd is None or len(xd) < 4:
                    continue
                step = max(1, len(xd) // 72)
                xs = xd[::step]
                ys = yd[::step]
                bx_arr = ((vx + (xs - xmin) / dx * vw) - hx) * sx
                by_arr = ((vy + (ymax - ys) / dy * vh) - hy) * sy
                c_fft = QColor(c_hex) if isinstance(c_hex, str) else QColor(c_hex)
                if not c_fft.isValid():
                    c_fft = QColor("#00E5FF")

                poly = QPolygonF()
                poly.append(QPointF(float(bx_arr[0]), float(bh + 4)))
                for px_v, py_v in zip(bx_arr, by_arr):
                    poly.append(QPointF(float(px_v), float(py_v)))
                poly.append(QPointF(float(bx_arr[-1]), float(bh + 4)))

                fill_c = QColor(c_fft)
                fill_c.setAlpha(95)
                stroke_c = _brighten_color(c_fft, 0.25)
                stroke_c.setAlpha(240)
                ip.setBrush(QBrush(fill_c))
                ip.setPen(QPen(stroke_c, 2.0))
                ip.drawPolygon(poly)

            # 3. Curvas EQ bajo el HUD
            eq_vb = getattr(self.overlay, "eq_vb", None)
            if eq_vb is not None:
                eq_range = eq_vb.viewRange()
                eymin, eymax = float(eq_range[1][0]), float(eq_range[1][1])
                edy = max(1e-6, eymax - eymin)
                y0_local = ((vy + (eymax - 0.0) / edy * vh) - hy) * sy

                for b_idx, st_c in enumerate(getattr(self.overlay, "band_fill_curves", [])):
                    if not st_c.isVisible():
                        continue
                    xd, yd = st_c.getData()
                    if xd is None or yd is None or len(xd) < 4:
                        continue
                    step = max(1, len(xd) // 64)
                    xs = xd[::step]
                    ys = yd[::step]
                    bx_arr = ((vx + (xs - xmin) / dx * vw) - hx) * sx
                    by_arr = ((vy + (eymax - ys) / edy * vh) - hy) * sy

                    b_col = QColor(self._active_color)
                    if b_idx < len(self.overlay.eq_engine.bands):
                        b_col = QColor(self.overlay.eq_engine.bands[b_idx].color)

                    poly = QPolygonF()
                    poly.append(QPointF(float(bx_arr[0]), float(y0_local)))
                    for px_v, py_v in zip(bx_arr, by_arr):
                        poly.append(QPointF(float(px_v), float(py_v)))
                    poly.append(QPointF(float(bx_arr[-1]), float(y0_local)))

                    fc = QColor(b_col)
                    fc.setAlpha(90)
                    sc = QColor(b_col)
                    sc.setAlpha(220)
                    ip.setBrush(QBrush(fc))
                    ip.setPen(QPen(sc, 1.8))
                    ip.drawPolygon(poly)

                for m_curve, m_col_fb in (
                    (getattr(self.overlay, "static_master_curve", None), getattr(self.overlay, "curve_eq_a_color", "#FFD54F")),
                    (getattr(self.overlay, "static_bus_b_curve", None), getattr(self.overlay, "curve_eq_b_color", "#00E5FF")),
                ):
                    if m_curve is None or not m_curve.isVisible():
                        continue
                    xd, yd = m_curve.getData()
                    if xd is None or yd is None or len(xd) < 4:
                        continue
                    step = max(1, len(xd) // 64)
                    xs = xd[::step]
                    ys = yd[::step]
                    bx_arr = ((vx + (xs - xmin) / dx * vw) - hx) * sx
                    by_arr = ((vy + (eymax - ys) / edy * vh) - hy) * sy
                    poly = QPolygonF([QPointF(float(px_v), float(py_v)) for px_v, py_v in zip(bx_arr, by_arr)])
                    pen_c = QColor(m_col_fb)
                    pen_c.setAlpha(245)
                    ip.setBrush(Qt.BrushStyle.NoBrush)
                    ip.setPen(QPen(pen_c, 2.2))
                    ip.drawPolyline(poly)

                # 4. Curvas Crossover bajo el HUD
                app_win = self.plot_widget.window() if self.plot_widget is not None else None
                for x_curve, x_col_fb in (
                    (getattr(app_win, "xover_curve_a", None), "#00E5FF"),
                    (getattr(app_win, "xover_curve_b", None), "#FFAB40"),
                ):
                    if x_curve is None or not x_curve.isVisible():
                        continue
                    xd, yd = x_curve.getData()
                    if xd is None or yd is None or len(xd) < 4:
                        continue
                    step = max(1, len(xd) // 64)
                    xs = xd[::step]
                    ys = yd[::step]
                    bx_arr = ((vx + (xs - xmin) / dx * vw) - hx) * sx
                    by_arr = ((vy + (eymax - ys) / edy * vh) - hy) * sy
                    poly = QPolygonF([QPointF(float(px_v), float(py_v)) for px_v, py_v in zip(bx_arr, by_arr)])
                    pen_c = QColor(x_col_fb)
                    pen_c.setAlpha(220)
                    ip.setBrush(Qt.BrushStyle.NoBrush)
                    ip.setPen(QPen(pen_c, 1.8))
                    ip.drawPolyline(poly)
        except Exception:
            pass
        finally:
            ip.end()

        # Aplicar Gaussian Blur separable rápido en NumPy
        ptr = img.bits()
        ptr.setsize(bh * bw * 4)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((bh, bw, 4)).astype(np.float32)
        blurred = np.clip(self._separable_gaussian_blur(arr, self._GAUSS_KERNEL), 0.0, 255.0).astype(np.uint8)
        self._blur_bytes = blurred.tobytes()
        qimg_blur = QImage(self._blur_bytes, bw, bh, bw * 4, QImage.Format.Format_ARGB32_Premultiplied)
        self._blur_pixmap = QPixmap.fromImage(qimg_blur)

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        self._update_gaussian_blur_texture()

        if self._blur_pixmap is not None and not self._blur_pixmap.isNull():
            painter.save()
            painter.setClipPath(self._cached_path)
            painter.drawPixmap(self._rect.toRect(), self._blur_pixmap)
            # Capa translúcida esmerilada oscura sobre el Gaussian Blur
            painter.fillRect(self._rect, QColor(14, 16, 22, 136))
            # Sutil tinte contextual de baja opacidad
            tint = QColor(self._active_color)
            tint.setAlpha(14)
            painter.fillRect(self._rect, tint)
            painter.restore()
        else:
            painter.fillPath(self._cached_path, QBrush(QColor(16, 18, 24, 130)))

        glow = QColor(self._active_color)
        glow.setAlpha(38)
        painter.setPen(QPen(glow, 2.8))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(self._cached_path)

        border = QColor(self._active_color)
        border.setAlpha(185)
        painter.setPen(QPen(border, 1.25))
        painter.drawPath(self._cached_path)


class FloatingBandHUD(QWidget):
    """
    HUD Flotante Superpuesto estilo FabFilter Pro-Q 4 + Apple Liquid Glass Oscuro:
    - Fondo semitransparente composicionado en la GPU vía QGraphicsItem en OpenGL (0% CPU).
    - Arrastrable libremente (Draggable) por todo el lienzo con memoria de posición.
    - Modo minimizado compacto (36px) con controles esenciales en 1 fila + Auto-Expand con cursor.
    """

    parameters_changed = pyqtSignal()

    def __init__(self, eq_engine: ParametricEQEngine, overlay: ProQ4Overlay, parent=None):
        super().__init__(parent)
        self.eq_engine = eq_engine
        self.overlay = overlay
        self._updating_ui = False
        self._active_band_color = QColor("#00E5FF")
        self._is_minimized = False
        self._auto_expand = False
        self._user_custom_pos = None
        self._drag_active = False
        self._drag_start_cursor = QPoint()
        self._drag_start_pos = QPoint()
        self._bg_gl_item = None
        self.plot_widget = None

        self._leave_timer = QTimer(self)
        self._leave_timer.setSingleShot(True)
        self._leave_timer.timeout.connect(self._on_leave_timeout)

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFixedSize(700, 196)

        self._apply_hud_stylesheet(self._active_band_color)

        self._build_ui()
        self.overlay.band_selected.connect(self.select_band)
        self.overlay.band_modified.connect(self._on_overlay_band_modified)
        self.refresh_from_engine()

        self.installEventFilter(self)
        self.expanded_container.installEventFilter(self)
        self.compact_container.installEventFilter(self)
        self.drag_grip_btn.installEventFilter(self)
        self.compact_drag_grip.installEventFilter(self)

    def _apply_hud_stylesheet(self, color: QColor):
        c = QColor(color) if color else QColor("#00E5FF")
        r, g, b = c.red(), c.green(), c.blue()
        hex_c = c.name()
        bright_hex = _brighten_color(c, 0.35).name()
        self.setStyleSheet(f"""
            QWidget {{
                background: transparent;
            }}
            QLabel {{
                background: transparent;
                border: none;
                color: #B0B6C2;
                font-size: 11px;
                font-weight: bold;
            }}
            QComboBox {{
                background-color: rgba(16, 18, 24, 155);
                color: #ECEFF4;
                border: 1px solid rgba(62, 68, 82, 165);
                border-radius: 5px;
                padding: 2px 7px;
                font-size: 11px;
                font-weight: bold;
            }}
            QComboBox:hover {{
                border-color: rgba({r}, {g}, {b}, 150);
            }}
            QPushButton {{
                background-color: rgba(16, 18, 24, 130);
                color: #C5CDD9;
                border: 1px solid rgba(62, 68, 82, 160);
                border-radius: 5px;
                padding: 1px 6px;
                font-size: 11px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: rgba(24, 28, 36, 165);
                color: #FFFFFF;
                border-color: rgba({r}, {g}, {b}, 145);
            }}
            QPushButton:pressed, QPushButton:checked {{
                background-color: rgba({r}, {g}, {b}, 52);
                color: {bright_hex};
                border: 1px solid {hex_c};
            }}
        """)
        if hasattr(self, "auto_expand_btn"):
            self._update_auto_expand_style()

    def _update_auto_expand_style(self):
        c = QColor(self._active_band_color) if self._active_band_color else QColor("#00E5FF")
        r, g, b = c.red(), c.green(), c.blue()
        hex_c = c.name()
        bright_hex = _brighten_color(c, 0.35).name()
        active_css = (
            f"background-color: rgba({r}, {g}, {b}, 56); color: {bright_hex}; "
            f"border: 1px solid {hex_c}; border-radius: 5px; font-size: 10px; font-weight: bold;"
        )
        idle_css = (
            f"background-color: rgba(16, 18, 24, 130); color: #C5CDD9; "
            f"border: 1px solid rgba(62, 68, 82, 160); border-radius: 5px; font-size: 10px; font-weight: bold;"
        )
        self.auto_expand_btn.setStyleSheet(active_css if self._auto_expand else idle_css)
        if hasattr(self, "compact_auto_btn"):
            self.compact_auto_btn.setStyleSheet(active_css if self._auto_expand else idle_css)

    def attach_gl_background(self, plot_widget: pg.PlotWidget):
        self.plot_widget = plot_widget
        scene = plot_widget.plotItem.vb.scene()
        if self._bg_gl_item is None:
            self._bg_gl_item = HUDOpenGLBackgroundItem(overlay=self.overlay, plot_widget=plot_widget)
            scene.addItem(self._bg_gl_item)
        else:
            self._bg_gl_item.overlay = self.overlay
            self._bg_gl_item.plot_widget = plot_widget
        self.sync_bg_item()

    def sync_bg_item(self):
        if self._bg_gl_item is None:
            return
        if not self.isVisible():
            self._bg_gl_item.hide()
            return
        if hasattr(self, "plot_widget") and self.plot_widget is not None:
            scene_pos = self.plot_widget.mapToScene(self.pos())
            self._bg_gl_item.set_state(
                scene_pos,
                float(self.width()),
                float(self.height()),
                self._is_minimized,
                self._active_band_color,
                self._gain_center_x(),
            )
            if not self._bg_gl_item.isVisible():
                self._bg_gl_item.show()

    def showEvent(self, event):
        super().showEvent(event)
        self.sync_bg_item()

    def hideEvent(self, event):
        super().hideEvent(event)
        if self._bg_gl_item is not None:
            self._bg_gl_item.hide()

    def moveEvent(self, event):
        super().moveEvent(event)
        self.sync_bg_item()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.sync_bg_item()

    def set_minimized(self, minimized: bool):
        if self._is_minimized == bool(minimized):
            return
        self._is_minimized = bool(minimized)
        if self._is_minimized:
            self.setFixedSize(700, 36)
            self.expanded_container.hide()
            self.compact_container.setGeometry(0, 0, 700, 36)
            self.compact_container.show()
        else:
            self.setFixedSize(700, 196)
            self.compact_container.hide()
            self.expanded_container.setGeometry(0, 0, 700, 196)
            self.expanded_container.show()
            p = self.parent()
            if p is not None:
                max_y = p.height() - 196 - 10
                if self.y() > max_y:
                    self.move(self.x(), max(44, max_y))
                    if self._user_custom_pos is not None:
                        self._user_custom_pos = self.pos()
        self.refresh_from_engine()
        self.sync_bg_item()

    def toggle_auto_expand(self, enabled: bool):
        self._auto_expand = bool(enabled)
        self.auto_expand_btn.setChecked(self._auto_expand)
        self.compact_auto_btn.setChecked(self._auto_expand)
        self.auto_expand_btn.setText("Auto ON" if self._auto_expand else "Auto")
        self._update_auto_expand_style()

    def enterEvent(self, event):
        super().enterEvent(event)
        self._leave_timer.stop()
        if self._is_minimized and self._auto_expand:
            self.set_minimized(False)

    def leaveEvent(self, event):
        super().leaveEvent(event)
        if not self._is_minimized and self._auto_expand:
            if not self._has_active_popup():
                self._leave_timer.start(400)

    def _has_active_popup(self) -> bool:
        focus_w = QApplication.focusWidget()
        if focus_w is not None and (self.isAncestorOf(focus_w) or isinstance(focus_w, QListView)):
            return True
        for cb in self.findChildren(QComboBox):
            if cb.view() is not None and cb.view().isVisible():
                return True
        return False

    def _on_leave_timeout(self):
        if not self._is_minimized and self._auto_expand:
            if not self._has_active_popup():
                cursor_pos = self.mapFromGlobal(QCursor.pos())
                if not self.rect().contains(cursor_pos):
                    self.set_minimized(True)

    def eventFilter(self, obj, event):
        if obj in (self, self.expanded_container, self.compact_container, getattr(self, "drag_grip_btn", None), getattr(self, "compact_drag_grip", None)):
            etype = event.type()
            if etype == QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.LeftButton:
                    self._drag_active = True
                    self._drag_start_cursor = event.globalPosition().toPoint()
                    self._drag_start_pos = self.pos()
                    return True
            elif etype == QEvent.Type.MouseMove:
                if getattr(self, "_drag_active", False):
                    delta = event.globalPosition().toPoint() - self._drag_start_cursor
                    new_pos = self._drag_start_pos + delta
                    p = self.parent()
                    pw = p.width() if p is not None else 1000
                    ph = p.height() if p is not None else 700
                    new_x = max(0, min(pw - self.width(), new_pos.x()))
                    new_y = max(40, min(ph - self.height(), new_pos.y()))
                    self._user_custom_pos = QPoint(new_x, new_y)
                    self.move(new_x, new_y)
                    self.sync_bg_item()
                    return True
            elif etype == QEvent.Type.MouseButtonRelease:
                if event.button() == Qt.MouseButton.LeftButton and getattr(self, "_drag_active", False):
                    self._drag_active = False
                    return True
        return super().eventFilter(obj, event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_active = True
            self._drag_start_cursor = event.globalPosition().toPoint()
            self._drag_start_pos = self.pos()
            event.accept()
            return
        event.accept()

    def mouseMoveEvent(self, event):
        if getattr(self, "_drag_active", False):
            delta = event.globalPosition().toPoint() - self._drag_start_cursor
            new_pos = self._drag_start_pos + delta
            p = self.parent()
            pw = p.width() if p is not None else 1000
            ph = p.height() if p is not None else 700
            new_x = max(0, min(pw - self.width(), new_pos.x()))
            new_y = max(40, min(ph - self.height(), new_pos.y()))
            self._user_custom_pos = QPoint(new_x, new_y)
            self.move(new_x, new_y)
            self.sync_bg_item()
            event.accept()
            return
        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and getattr(self, "_drag_active", False):
            self._drag_active = False
            event.accept()
            return
        event.accept()

    def mouseDoubleClickEvent(self, event):
        event.accept()

    def wheelEvent(self, event):
        event.accept()

    @staticmethod
    def _fmt_freq(v: float) -> str:
        return f"{v / 1000.0:.2f} kHz" if v >= 1000.0 else f"{v:.1f} Hz"

    def _build_ui(self):
        # =====================================================================
        # 1. CONTENEDOR EXPANDIDO (Altura 196px)
        # =====================================================================
        self.expanded_container = QWidget(self)
        self.expanded_container.setGeometry(0, 0, 700, 196)

        # Botón Minimizar, Auto-Expand y Agarre de Arrastre en la esquina superior derecha
        self.drag_grip_btn = QLabel("⠿", self.expanded_container)
        self.drag_grip_btn.setFixedSize(18, 18)
        self.drag_grip_btn.move(672, 38)
        self.drag_grip_btn.setCursor(Qt.CursorShape.SizeAllCursor)
        self.drag_grip_btn.setToolTip("Arrastrar HUD")
        self.drag_grip_btn.setStyleSheet("color: #727B8E; font-size: 13px; font-weight: bold;")

        self.minimize_btn = QPushButton("—", self.expanded_container)
        self.minimize_btn.setFixedSize(22, 18)
        self.minimize_btn.move(646, 38)
        self.minimize_btn.setToolTip("Minimizar HUD (Barra Compacta)")
        self.minimize_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.minimize_btn.clicked.connect(lambda: self.set_minimized(True))

        self.auto_expand_btn = QPushButton("Auto", self.expanded_container)
        self.auto_expand_btn.setCheckable(True)
        self.auto_expand_btn.setFixedSize(56, 18)
        self.auto_expand_btn.move(586, 38)
        self.auto_expand_btn.setToolTip("Auto-expandir al pasar el cursor (vuelve a minimizar tras 0.4s)")
        self.auto_expand_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_expand_btn.clicked.connect(self.toggle_auto_expand)

        # Columna Izquierda: Power + Color Swatch, Filter Shape y Slope
        self.band_enable_btn = PowerToggleButton(self.expanded_container)
        self.band_enable_btn.setToolTip("Activar / Desactivar banda seleccionada")
        self.band_enable_btn.move(20, 44)
        self.band_enable_btn.clicked.connect(self._on_band_enable_toggled)

        self.color_swatch = ColorSwatchButton("#00E5FF", self.expanded_container)
        self.color_swatch.move(50, 44)
        self.color_swatch.colorSelected.connect(self._on_swatch_color_selected)

        self.type_combo = QComboBox(self.expanded_container)
        style_dark_combobox(self.type_combo)
        self.type_combo.addItems(FILTER_TYPES)
        self.type_combo.setFixedSize(112, 27)
        self.type_combo.move(20, 76)
        self.type_combo.setToolTip("Forma del filtro (Filter Shape)")
        self.type_combo.currentTextChanged.connect(self._on_type_or_slope_changed)

        self.slope_combo = QComboBox(self.expanded_container)
        style_dark_combobox(self.slope_combo)
        self.slope_combo.setFixedSize(112, 27)
        self.slope_combo.move(20, 110)
        self.slope_combo.setToolTip("Pendiente del filtro (Slope)")
        for s in SLOPE_OPTIONS:
            self.slope_combo.addItem(f"{s} dB/oct", s)
        self.slope_combo.currentIndexChanged.connect(self._on_type_or_slope_changed)

        # Bloque Central: FREQ, GAIN encastrada en la joroba, THR, Q
        self.freq_knob = ProQKnob(
            title="FREQ", min_val=10.0, max_val=24000.0, default_val=1000.0,
            fmt_func=self._fmt_freq, log_scale=True, bipolar=False,
            accent_color="#00E5FF", diameter=60, parent=self.expanded_container
        )
        self.freq_knob.move(146, 48)
        self.freq_knob.valueChanged.connect(self._on_knob_param_changed)

        self.gain_dyn_knob = GainDynProQKnob(diameter=88, parent=self.expanded_container)
        self.gain_dyn_knob.move(232, 14)
        self.gain_dyn_knob.valueChanged.connect(self._on_knob_param_changed)
        self.gain_dyn_knob.dynRangeChanged.connect(self._on_dyn_range_knob_changed)

        self.thresh_meter = DynThresholdMeter(parent=self.expanded_container)
        self.thresh_meter.move(346, 44)
        self.thresh_meter.thresholdChanged.connect(self._on_threshold_meter_changed)

        self.q_knob = ProQKnob(
            title="Q", min_val=0.10, max_val=24.0, default_val=1.00,
            fmt_func=lambda v: f"{v:.2f}", log_scale=True, bipolar=False,
            accent_color="#00E5FF", diameter=60, parent=self.expanded_container
        )
        self.q_knob.move(398, 48)
        self.q_knob.valueChanged.connect(self._on_knob_param_changed)

        # Bloque Derecho: A, R, OUT
        self.attack_knob = ProQKnob(
            title="A", min_val=0.5, max_val=500.0, default_val=15.0,
            fmt_func=lambda v: f"{v:.1f} ms" if v < 10 else f"{v:.0f} ms",
            log_scale=True, bipolar=False, accent_color="#00E5FF", diameter=46, parent=self.expanded_container
        )
        self.attack_knob.move(492, 56)
        self.attack_knob.valueChanged.connect(self._on_knob_param_changed)

        self.release_knob = ProQKnob(
            title="R", min_val=5.0, max_val=2000.0, default_val=120.0,
            fmt_func=lambda v: f"{v:.0f} ms",
            log_scale=True, bipolar=False, accent_color="#00E5FF", diameter=46, parent=self.expanded_container
        )
        self.release_knob.move(556, 56)
        self.release_knob.valueChanged.connect(self._on_knob_param_changed)

        self.out_gain_knob = ProQKnob(
            title="OUT", min_val=-24.0, max_val=24.0, default_val=0.0,
            fmt_func=lambda v: f"{v:+.1f} dB", log_scale=False, bipolar=True,
            accent_color="#FFD54F", diameter=46, parent=self.expanded_container
        )
        self.out_gain_knob.move(620, 56)
        self.out_gain_knob.valueChanged.connect(self._on_global_changed)

        # Fila Inferior
        self.bottom_bar = QWidget(self.expanded_container)
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
        style_dark_combobox(self.band_combo)
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

        self.channel_mode_combo = QComboBox(self.bottom_bar)
        style_dark_combobox(self.channel_mode_combo)
        self.channel_mode_combo.setFixedSize(110, 24)
        self.channel_mode_combo.setToolTip("Modo de Canal: Stereo (LR), Left (L), Right (R), Mid (Mono) o Side (Stereo)")
        for cm, lbl in (
            ("LR", "Stereo (LR)"),
            ("L", "Left (L)"),
            ("R", "Right (R)"),
            ("Mid", "Mid (Mono)"),
            ("Side", "Side (Stereo)"),
        ):
            self.channel_mode_combo.addItem(lbl, cm)
        self.channel_mode_combo.currentIndexChanged.connect(self._on_channel_or_output_changed)
        bot_row.addWidget(self.channel_mode_combo)

        self.target_out_combo = QComboBox(self.bottom_bar)
        style_dark_combobox(self.target_out_combo)
        self.target_out_combo.setFixedSize(108, 24)
        self.target_out_combo.setToolTip("Asignar Banda a Bus de Salida: A+B (Ambas), Out A (1-2) o Out B (3-4)")
        for to_val, lbl in (
            ("A+B", "A+B (Ambas)"),
            ("A", "Out A (1-2)"),
            ("B", "Out B (3-4)"),
        ):
            self.target_out_combo.addItem(lbl, to_val)
        self.target_out_combo.currentIndexChanged.connect(self._on_channel_or_output_changed)
        bot_row.addWidget(self.target_out_combo)

        self.dyn_toggle_btn = LiquidGlassButton("DYN", self.bottom_bar, pill_radius=6.0)
        self.dyn_toggle_btn.setCheckable(True)
        self.dyn_toggle_btn.setFixedSize(58, 24)
        self.dyn_toggle_btn.setToolTip("Activar / Desactivar modo Dynamic EQ en esta banda")
        self.dyn_toggle_btn.clicked.connect(self._on_dyn_toggle_clicked)
        bot_row.addWidget(self.dyn_toggle_btn)

        self.band_solo_btn = LiquidGlassButton("Isolate", self.bottom_bar, active_tint_color="#00E5FF", pill_radius=6.0)
        self.band_solo_btn.setCheckable(True)
        self.band_solo_btn.setFixedSize(68, 24)
        self.band_solo_btn.setToolTip("Aislar únicamente el rango de frecuencias de esta banda (Band-Pass Isolate)")
        self.band_solo_btn.clicked.connect(self._on_band_solo_toggled)
        bot_row.addWidget(self.band_solo_btn)

        # =====================================================================
        # 2. CONTENEDOR COMPACTO (Altura 36px)
        # =====================================================================
        self.compact_container = QWidget(self)
        self.compact_container.setGeometry(0, 0, 700, 36)
        self.compact_container.hide()
        self.compact_widget = self.compact_container

        comp_row = QHBoxLayout(self.compact_container)
        comp_row.setContentsMargins(10, 4, 10, 4)
        comp_row.setSpacing(6)

        self.compact_drag_grip = QLabel("⠿", self.compact_container)
        self.compact_drag_grip.setFixedSize(14, 24)
        self.compact_drag_grip.setCursor(Qt.CursorShape.SizeAllCursor)
        self.compact_drag_grip.setToolTip("Arrastrar HUD")
        self.compact_drag_grip.setStyleSheet("color: #727B8E; font-size: 13px; font-weight: bold;")
        comp_row.addWidget(self.compact_drag_grip)

        # 1. Power ON/OFF + Color Swatch
        self.compact_power_btn = PowerToggleButton(self.compact_container)
        self.compact_power_btn.setToolTip("Activar / Desactivar banda")
        self.compact_power_btn.clicked.connect(self._on_band_enable_toggled)
        comp_row.addWidget(self.compact_power_btn)

        self.compact_color_swatch = ColorSwatchButton("#00E5FF", self.compact_container)
        self.compact_color_swatch.colorSelected.connect(self._on_swatch_color_selected)
        comp_row.addWidget(self.compact_color_swatch)

        # 2. Navegador < Band X > + y ✕
        self.compact_prev_btn = QPushButton("‹", self.compact_container)
        self.compact_prev_btn.setFixedSize(22, 22)
        self.compact_prev_btn.setToolTip("Banda anterior")
        self.compact_prev_btn.clicked.connect(lambda: self._step_band(-1))
        comp_row.addWidget(self.compact_prev_btn)

        self.compact_band_combo = QComboBox(self.compact_container)
        style_dark_combobox(self.compact_band_combo)
        self.compact_band_combo.setFixedSize(88, 22)
        self.compact_band_combo.currentIndexChanged.connect(self._on_compact_band_combo_changed)
        comp_row.addWidget(self.compact_band_combo)

        self.compact_next_btn = QPushButton("›", self.compact_container)
        self.compact_next_btn.setFixedSize(22, 22)
        self.compact_next_btn.setToolTip("Banda siguiente")
        self.compact_next_btn.clicked.connect(lambda: self._step_band(1))
        comp_row.addWidget(self.compact_next_btn)

        self.compact_add_btn = QPushButton("+", self.compact_container)
        self.compact_add_btn.setFixedSize(22, 22)
        self.compact_add_btn.setToolTip("Añadir nueva banda")
        self.compact_add_btn.clicked.connect(self._on_add_band)
        comp_row.addWidget(self.compact_add_btn)

        self.compact_del_btn = QPushButton("✕", self.compact_container)
        self.compact_del_btn.setFixedSize(22, 22)
        self.compact_del_btn.setToolTip("Eliminar banda (Supr)")
        self.compact_del_btn.clicked.connect(self._on_delete_band)
        comp_row.addWidget(self.compact_del_btn)

        comp_row.addSpacing(4)

        # 3. Lectura rápida interactiva: Freq | Gain | Q
        self.compact_readout = QLabel("1.00 kHz  |  0.0 dB  |  Q 1.00", self.compact_container)
        self.compact_readout.setStyleSheet("""
            color: #00E5FF;
            font-size: 11px;
            font-weight: bold;
            padding: 0 4px;
        """)
        self.compact_readout.setToolTip("Lectura de parámetros de la banda activa")
        comp_row.addWidget(self.compact_readout)

        comp_row.addStretch(1)

        # Selector rápido Out (A/B/A+B)
        self.compact_target_out = QComboBox(self.compact_container)
        style_dark_combobox(self.compact_target_out)
        self.compact_target_out.setFixedSize(84, 22)
        self.compact_target_out.setToolTip("Asignar a Bus de Salida")
        for to_val, lbl in (
            ("A+B", "A+B"),
            ("A", "Out A"),
            ("B", "Out B"),
        ):
            self.compact_target_out.addItem(lbl, to_val)
        self.compact_target_out.currentIndexChanged.connect(self._on_compact_output_changed)
        comp_row.addWidget(self.compact_target_out)

        # 4. Botón Isolate + Auto + Expandir
        self.compact_solo_btn = LiquidGlassButton("ISO", self.compact_container, pill_radius=6.0, active_tint_color="#00E5FF")
        self.compact_solo_btn.setCheckable(True)
        self.compact_solo_btn.setFixedSize(36, 22)
        self.compact_solo_btn.setToolTip("Aislar banda (Solo Isolate)")
        self.compact_solo_btn.clicked.connect(self._on_band_solo_toggled)
        comp_row.addWidget(self.compact_solo_btn)

        self.compact_auto_btn = QPushButton("Auto", self.compact_container)
        self.compact_auto_btn.setCheckable(True)
        self.compact_auto_btn.setFixedSize(40, 22)
        self.compact_auto_btn.setToolTip("Auto-expandir al pasar el cursor")
        self.compact_auto_btn.clicked.connect(self.toggle_auto_expand)
        comp_row.addWidget(self.compact_auto_btn)

        self.compact_expand_btn = QPushButton("▲ Expandir", self.compact_container)
        self.compact_expand_btn.setFixedSize(78, 22)
        self.compact_expand_btn.setToolTip("Expandir panel HUD completo")
        self.compact_expand_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.compact_expand_btn.clicked.connect(lambda: self.set_minimized(False))
        comp_row.addWidget(self.compact_expand_btn)

    def _gain_center_x(self) -> float:
        return float(self.gain_dyn_knob.x() + self.gain_dyn_knob.width() * 0.5)

    def paintEvent(self, event):
        # Si el fondo OpenGL está activo, el 100% de la transparencia se mezcla en la GPU
        if self._bg_gl_item is not None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w = float(self.width())
        h = float(self.height())
        pad = 2.5
        r_corner = 10.0 if self._is_minimized else 18.0
        y_bot = h - pad
        x_left = pad
        x_right = w - pad

        path = QPainterPath()
        if self._is_minimized:
            path.addRoundedRect(QRectF(x_left, pad, w - 2 * pad, h - 2 * pad), r_corner, r_corner)
        else:
            y_top = 34.0
            y_peak = 2.5
            arch_cx = self._gain_center_x()
            arch_half_w = 104.0
            ax1 = max(x_left + r_corner + 10.0, arch_cx - arch_half_w)
            ax2 = min(x_right - r_corner - 10.0, arch_cx + arch_half_w)

            path.moveTo(x_left + r_corner, y_top)
            path.lineTo(ax1, y_top)
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

        body_grad = QLinearGradient(0.0, 0.0, 0.0, y_bot)
        body_grad.setColorAt(0.0, QColor(34, 37, 46, 238))
        body_grad.setColorAt(0.25, QColor(30, 32, 40, 235))
        body_grad.setColorAt(1.0, QColor(20, 21, 26, 242))
        painter.fillPath(path, QBrush(body_grad))

        glow_c = QColor(self._active_band_color)
        glow_c.setAlpha(45)
        painter.setPen(QPen(glow_c, 3.2))
        painter.drawPath(path)

        spec_border = QLinearGradient(0.0, 0.0, 0.0, y_bot)
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

        if not self._is_minimized:
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
        if hasattr(self.overlay, "node_callout"):
            self.overlay.node_callout.show_callout()

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
                    dyn_tag = " [D]" if b.is_dynamic_active else ""
                    self.band_combo.addItem(f"Band {i + 1}{dyn_tag}", i)
                if 0 <= idx < len(bands):
                    self.band_combo.setCurrentIndex(idx)
                self.band_combo.blockSignals(False)

                if hasattr(self, "compact_band_combo"):
                    self.compact_band_combo.blockSignals(True)
                    self.compact_band_combo.clear()
                    for i, b in enumerate(bands):
                        dyn_tag = " [D]" if b.is_dynamic_active else ""
                        self.compact_band_combo.addItem(f"Band {i + 1}{dyn_tag}", i)
                    if 0 <= idx < len(bands):
                        self.compact_band_combo.setCurrentIndex(idx)
                    self.compact_band_combo.blockSignals(False)

                has_band = 0 <= idx < len(bands)
                for w in (
                    self.del_band_btn, self.band_enable_btn, self.prev_band_btn,
                    self.next_band_btn, self.color_swatch, self.band_solo_btn,
                    self.channel_mode_combo, self.target_out_combo,
                    self.type_combo, self.slope_combo, self.freq_knob,
                    self.gain_dyn_knob, self.thresh_meter, self.q_knob,
                    self.dyn_toggle_btn, self.attack_knob, self.release_knob,
                    getattr(self, "compact_del_btn", None),
                    getattr(self, "compact_power_btn", None),
                    getattr(self, "compact_prev_btn", None),
                    getattr(self, "compact_next_btn", None),
                    getattr(self, "compact_color_swatch", None),
                    getattr(self, "compact_solo_btn", None),
                    getattr(self, "compact_target_out", None),
                ):
                    if w is not None:
                        w.setEnabled(has_band)

                self.freq_knob.max_val = max(24000.0, self.eq_engine.sr * 0.495)

                if has_band:
                    b = bands[idx]
                    self._active_band_color = QColor(b.color)
                    self._apply_hud_stylesheet(self._active_band_color)
                    self.color_swatch.setColor(b.color)
                    self.band_enable_btn.setAccentColor(b.color)

                    if hasattr(self, "compact_color_swatch"):
                        self.compact_color_swatch.setColor(b.color)
                    if hasattr(self, "compact_power_btn"):
                        self.compact_power_btn.setAccentColor(b.color)
                        self.compact_power_btn.setChecked(b.enabled)

                    self.band_enable_btn.setChecked(b.enabled)
                    is_iso = bool(b.solo or self.eq_engine.isolated_band_index == idx)
                    self.band_solo_btn.setChecked(is_iso)
                    self.band_solo_btn.setActiveTintColor(b.color)
                    if hasattr(self, "compact_solo_btn"):
                        self.compact_solo_btn.setChecked(is_iso)
                        self.compact_solo_btn.setActiveTintColor(b.color)

                    self.type_combo.setCurrentText(b.filter_type)

                    cm_idx = self.channel_mode_combo.findData(b.channel_mode)
                    if cm_idx >= 0:
                        self.channel_mode_combo.blockSignals(True)
                        self.channel_mode_combo.setCurrentIndex(cm_idx)
                        self.channel_mode_combo.blockSignals(False)

                    to_idx = self.target_out_combo.findData(b.target_output)
                    if to_idx >= 0:
                        self.target_out_combo.blockSignals(True)
                        self.target_out_combo.setCurrentIndex(to_idx)
                        self.target_out_combo.blockSignals(False)
                        if hasattr(self, "compact_target_out"):
                            self.compact_target_out.blockSignals(True)
                            c_to = self.compact_target_out.findData(b.target_output)
                            if c_to >= 0:
                                self.compact_target_out.setCurrentIndex(c_to)
                            self.compact_target_out.blockSignals(False)

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

                    if hasattr(self, "compact_readout"):
                        gain_txt = f"{b.gain_db:+.1f} dB" if b.supports_gain else b.filter_type
                        self.compact_readout.setText(f"{self._fmt_freq(b.freq)}  |  {gain_txt}  |  Q {b.q:.2f}")
                        self.compact_readout.setStyleSheet(f"color: {b.color}; font-size: 11px; font-weight: bold; padding: 0 4px;")

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
                    if hasattr(self, "compact_readout"):
                        self.compact_readout.setText("Sin bandas activas")
                        self.compact_readout.setStyleSheet("color: #727B8E; font-size: 11px; font-weight: bold; padding: 0 4px;")

                self.out_gain_knob.setValue(self.eq_engine.output_gain_db)
            self.update()
            self.sync_bg_item()
        finally:
            self._updating_ui = False

    def _on_band_combo_changed(self, index: int):
        if self._updating_ui or index < 0:
            return
        self.select_band(index)

    def _on_compact_band_combo_changed(self, index: int):
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
        self.eq_engine.set_isolated_band(idx if checked else None)
        self.refresh_from_engine()
        self.overlay.update_eq_curves(force=True)
        self.parameters_changed.emit()

    def _on_channel_or_output_changed(self):
        if self._updating_ui:
            return
        idx = self.overlay.selected_band_idx
        with self.eq_engine.lock:
            if 0 <= idx < len(self.eq_engine.bands):
                b = self.eq_engine.bands[idx]
                cm = str(self.channel_mode_combo.currentData())
                to_val = str(self.target_out_combo.currentData())
                if cm in CHANNEL_MODES and b.channel_mode != cm:
                    b.channel_mode = cm
                    b.reset_state()
                if to_val in TARGET_OUTPUTS and b.target_output != to_val:
                    b.target_output = to_val
                    b.reset_state()
                self.eq_engine.mark_dirty()
        self.refresh_from_engine()
        self.overlay.update_eq_curves(force=True)
        self.parameters_changed.emit()

    def _on_compact_output_changed(self):
        if self._updating_ui:
            return
        idx = self.overlay.selected_band_idx
        with self.eq_engine.lock:
            if 0 <= idx < len(self.eq_engine.bands):
                b = self.eq_engine.bands[idx]
                to_val = str(self.compact_target_out.currentData())
                if to_val in TARGET_OUTPUTS and b.target_output != to_val:
                    b.target_output = to_val
                    b.reset_state()
                    self.eq_engine.mark_dirty()
        self.refresh_from_engine()
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
                if hasattr(self, "compact_readout"):
                    gain_txt = f"{b.gain_db:+.1f} dB" if b.supports_gain else b.filter_type
                    self.compact_readout.setText(f"{self._fmt_freq(b.freq)}  |  {gain_txt}  |  Q {b.q:.2f}")
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
                if hasattr(self, "compact_readout"):
                    gain_txt = f"{b.gain_db:+.1f} dB" if b.supports_gain else b.filter_type
                    self.compact_readout.setText(f"{self._fmt_freq(b.freq)}  |  {gain_txt}  |  Q {b.q:.2f}")
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
                if self._is_minimized and hasattr(self, "compact_readout"):
                    gain_txt = f"{b.gain_db:+.1f} dB" if b.supports_gain else b.filter_type
                    dyn_txt = f" (Δ {b.current_dyn_offset_db:+.1f})" if (b.is_dynamic_active and abs(b.current_dyn_offset_db) >= 0.1) else ""
                    self.compact_readout.setText(f"{self._fmt_freq(b.freq)}  |  {gain_txt}{dyn_txt}  |  Q {b.q:.2f}")
        if self._bg_gl_item is not None and self._bg_gl_item.isVisible():
            self._bg_gl_item.update()


# Alias de compatibilidad
EQControlPanel = FloatingBandHUD


class LimiterGRMiniWidget(QWidget):
    """
    Mini-widget indicador de Reducción de Ganancia (GR) en tiempo real estilo Fruity Limiter.
    Ubicado en la cabecera del osciloscopio junto a los controles del limitador.
    """

    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(82, 20)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Reducción de Ganancia (GR) en tiempo real del Limitador Master")
        self.gr_db = 0.0

    def set_gr(self, gr_db: float):
        new_gr = float(min(0.0, gr_db))
        if abs(new_gr - self.gr_db) > 0.04:
            self.gr_db = new_gr
            self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        event.accept()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w = self.width()
        h = self.height()

        # Fondo del widget padre (#101216) para WA_OpaquePaintEvent
        painter.fillRect(0, 0, w, h, QColor("#101216"))

        # Fondo cápsula
        bg_col = QColor("#16181E")
        border_col = QColor("#D500F9" if self.gr_db < -0.2 else "#2B303C")
        painter.setPen(QPen(border_col, 1))
        painter.setBrush(QBrush(bg_col))
        painter.drawRoundedRect(QRectF(0.5, 0.5, w - 1, h - 1), 4.0, 4.0)

        # Barra de atenuación (Rango 0 dB a -18 dB)
        ratio = float(np.clip(-self.gr_db / 18.0, 0.0, 1.0))
        if ratio > 0.01:
            bar_w = int(round(ratio * (w - 6)))
            bar_grad = QLinearGradient(3, 0, 3 + bar_w, 0)
            bar_grad.setColorAt(0.0, QColor("#AA00FF"))
            bar_grad.setColorAt(1.0, QColor("#D500F9"))
            painter.fillRect(QRectF(3, 2, bar_w, h - 4), bar_grad)

        # Texto GR
        font = QFont("Segoe UI", 8, QFont.Weight.Bold)
        painter.setFont(font)
        text_col = QColor("#FFFFFF" if self.gr_db < -0.2 else "#787C85")
        painter.setPen(text_col)
        txt = f"GR {self.gr_db:.1f}" if self.gr_db < -0.05 else "GR 0.0"
        painter.drawText(QRectF(0, 0, w, h), Qt.AlignmentFlag.AlignCenter, txt)


class LimiterControlPopup(QWidget):
    """
    Panel flotante no-modal de controles para el Limitador Master estilo 'Fruity Limiter'.
    Permite ajustar Threshold, Ceiling, Attack, Release y visualizar la reducción de ganancia en vivo.
    """

    def __init__(self, eq_engine, overlay_ref=None, app_ref=None, parent=None):
        super().__init__(parent, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.eq_engine = eq_engine
        self.overlay_ref = overlay_ref
        self.app_ref = app_ref
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setFixedWidth(340)

        self.setStyleSheet("""
            QWidget {
                background-color: #16181E;
                color: #ECEFF4;
                font-family: "Segoe UI", sans-serif;
                font-size: 11px;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #252830;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #D500F9;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #FFFFFF;
                border: 1px solid #D500F9;
                width: 12px;
                margin-top: -4px;
                margin-bottom: -4px;
                border-radius: 6px;
            }
            QPushButton#toggleBtn {
                background-color: #D500F9;
                color: #FFFFFF;
                font-weight: bold;
                border: 1px solid #EA80FC;
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 11px;
            }
            QPushButton#toggleBtn:checked {
                background-color: #D500F9;
                color: #FFFFFF;
            }
            QPushButton#actionBtn {
                background-color: #242730;
                color: #B0B5C0;
                border: 1px solid #363C48;
                border-radius: 4px;
                padding: 3px 10px;
                font-size: 10px;
            }
            QPushButton#actionBtn:hover {
                background-color: #2F3340;
                color: #FFFFFF;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        # Encabezado
        hdr_row = QHBoxLayout()
        title_lbl = QLabel("Limitador Master (Fruity Limiter)")
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

        # Fila de Activación y Medidor GR en vivo
        act_row = QHBoxLayout()
        self.toggle_btn = QPushButton("Limitador: ACTIVO")
        self.toggle_btn.setObjectName("toggleBtn")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setChecked(self.eq_engine.limiter_a.enabled)
        self.toggle_btn.clicked.connect(self._on_toggle_clicked)
        act_row.addWidget(self.toggle_btn)
        act_row.addStretch(1)

        self.gr_lbl = QLabel("GR: 0.0 dB")
        self.gr_lbl.setStyleSheet("font-size: 12px; font-weight: bold; color: #8C919B;")
        act_row.addWidget(self.gr_lbl)
        layout.addLayout(act_row)

        # Barra visual de GR
        self.meter_bar = QWidget()
        self.meter_bar.setFixedHeight(8)
        self.meter_bar.setStyleSheet("background-color: #21242C; border-radius: 3px;")
        layout.addWidget(self.meter_bar)

        # Separador
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #262A34; margin: 1px 0;")
        layout.addWidget(sep)

        # 4 Sliders de parámetros
        # 1. THR (Threshold): -30.0 dB a 0.0 dB
        self.thr_slider, self.thr_val_lbl = self._create_param_row(
            layout, "THR (Umbral):",
            vmin=-300, vmax=0,
            init_val=int(round(self.eq_engine.limiter_a.threshold_db * 10)),
            fmt_fn=lambda v: f"{v / 10.0:.1f} dB",
            callback=self._on_thr_changed,
        )

        # 2. CEIL (Ceiling): -12.0 dB a 0.0 dB
        self.ceil_slider, self.ceil_val_lbl = self._create_param_row(
            layout, "CEIL (Techo):",
            vmin=-120, vmax=0,
            init_val=int(round(self.eq_engine.limiter_a.ceiling_db * 10)),
            fmt_fn=lambda v: f"{v / 10.0:.1f} dB",
            callback=self._on_ceil_changed,
        )

        # 3. ATT (Attack): 0.1 ms a 100.0 ms
        self.att_slider, self.att_val_lbl = self._create_param_row(
            layout, "ATT (Ataque):",
            vmin=1, vmax=1000,
            init_val=int(round(self.eq_engine.limiter_a.attack_ms * 10)),
            fmt_fn=lambda v: f"{v / 10.0:.1f} ms",
            callback=self._on_att_changed,
        )

        # 4. REL (Release): 1.0 ms a 500.0 ms
        self.rel_slider, self.rel_val_lbl = self._create_param_row(
            layout, "REL (Relajación):",
            vmin=1, vmax=500,
            init_val=int(round(self.eq_engine.limiter_a.release_ms)),
            fmt_fn=lambda v: f"{v} ms",
            callback=self._on_rel_changed,
        )

        # Botones inferiores: Restaurar y Listo
        bot_row = QHBoxLayout()
        reset_btn = QPushButton("Restaurar por defecto")
        reset_btn.setObjectName("actionBtn")
        reset_btn.clicked.connect(self._reset_defaults)
        bot_row.addWidget(reset_btn)
        bot_row.addStretch(1)

        done_btn = QPushButton("Listo")
        done_btn.setObjectName("actionBtn")
        done_btn.clicked.connect(self.close)
        bot_row.addWidget(done_btn)
        layout.addLayout(bot_row)

        self._update_toggle_style()

    def _create_param_row(self, parent_layout, label_text, vmin, vmax, init_val, fmt_fn, callback):
        row = QHBoxLayout()
        lbl = QLabel(label_text)
        lbl.setFixedWidth(110)
        lbl.setStyleSheet("color: #D8DEE9; font-size: 10px;")
        row.addWidget(lbl)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(vmin, vmax)
        slider.setValue(init_val)
        row.addWidget(slider, 1)

        val_lbl = QLabel(fmt_fn(init_val))
        val_lbl.setFixedWidth(55)
        val_lbl.setStyleSheet("color: #D500F9; font-weight: bold; font-size: 10px;")
        val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(val_lbl)

        slider.valueChanged.connect(lambda v: (val_lbl.setText(fmt_fn(v)), callback(v)))
        parent_layout.addLayout(row)
        return slider, val_lbl

    def _update_toggle_style(self):
        enabled = self.eq_engine.limiter_a.enabled
        if enabled:
            self.toggle_btn.setText("Limitador: ACTIVO")
            self.toggle_btn.setStyleSheet("""
                background-color: #D500F9;
                color: #FFFFFF;
                font-weight: bold;
                border: 1px solid #EA80FC;
                border-radius: 4px;
                padding: 4px 10px;
            """)
        else:
            self.toggle_btn.setText("Limitador: APAGADO")
            self.toggle_btn.setStyleSheet("""
                background-color: #262932;
                color: #7E8594;
                font-weight: bold;
                border: 1px solid #3E4350;
                border-radius: 4px;
                padding: 4px 10px;
            """)

    def _on_toggle_clicked(self, checked: bool):
        self.eq_engine.limiter_a.enabled = checked
        self.eq_engine.limiter_b.enabled = checked
        self._update_toggle_style()
        if self.overlay_ref:
            self.overlay_ref.invalidate_cache()
        if self.app_ref:
            self.app_ref.schedule_config_save()

    def _on_thr_changed(self, v: int):
        val_db = v / 10.0
        self.eq_engine.limiter_a.threshold_db = val_db
        self.eq_engine.limiter_b.threshold_db = val_db
        self.eq_engine.limiter_a._update_coefficients()
        self.eq_engine.limiter_b._update_coefficients()
        if self.overlay_ref:
            self.overlay_ref.invalidate_cache()
        if self.app_ref:
            self.app_ref.schedule_config_save()

    def _on_ceil_changed(self, v: int):
        val_db = v / 10.0
        self.eq_engine.limiter_a.ceiling_db = val_db
        self.eq_engine.limiter_b.ceiling_db = val_db
        self.eq_engine.limiter_a._update_coefficients()
        self.eq_engine.limiter_b._update_coefficients()
        if self.app_ref:
            self.app_ref.schedule_config_save()

    def _on_att_changed(self, v: int):
        val_ms = v / 10.0
        self.eq_engine.limiter_a.attack_ms = val_ms
        self.eq_engine.limiter_b.attack_ms = val_ms
        self.eq_engine.limiter_a._update_coefficients()
        self.eq_engine.limiter_b._update_coefficients()
        if self.app_ref:
            self.app_ref.schedule_config_save()

    def _on_rel_changed(self, v: int):
        val_ms = float(v)
        self.eq_engine.limiter_a.release_ms = val_ms
        self.eq_engine.limiter_b.release_ms = val_ms
        self.eq_engine.limiter_a._update_coefficients()
        self.eq_engine.limiter_b._update_coefficients()
        if self.app_ref:
            self.app_ref.schedule_config_save()

    def set_live_gr(self, gr_db: float):
        if not self.isVisible():
            return
        if gr_db < -0.05:
            self.gr_lbl.setText(f"GR: {gr_db:.1f} dB")
            self.gr_lbl.setStyleSheet("font-size: 12px; font-weight: bold; color: #D500F9;")
            ratio = float(np.clip(-gr_db / 18.0, 0.0, 1.0))
            self.meter_bar.setStyleSheet(f"""
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #AA00FF, stop:{ratio} #D500F9, stop:{min(1.0, ratio + 0.001)} #21242C, stop:1 #21242C);
                border-radius: 3px;
            """)
        else:
            self.gr_lbl.setText("GR: 0.0 dB")
            self.gr_lbl.setStyleSheet("font-size: 12px; font-weight: bold; color: #787C85;")
            self.meter_bar.setStyleSheet("background-color: #21242C; border-radius: 3px;")

    update_meter = set_live_gr

    def _reset_defaults(self):
        self.toggle_btn.setChecked(True)
        self._on_toggle_clicked(True)

        self.thr_slider.blockSignals(True)
        self.thr_slider.setValue(0)
        self.thr_val_lbl.setText("0.0 dB")
        self.thr_slider.blockSignals(False)
        self._on_thr_changed(0)

        self.ceil_slider.blockSignals(True)
        self.ceil_slider.setValue(-1)
        self.ceil_val_lbl.setText("-0.1 dB")
        self.ceil_slider.blockSignals(False)
        self._on_ceil_changed(-1)

        self.att_slider.blockSignals(True)
        self.att_slider.setValue(20)
        self.att_val_lbl.setText("2.0 ms")
        self.att_slider.blockSignals(False)
        self._on_att_changed(20)

        self.rel_slider.blockSignals(True)
        self.rel_slider.setValue(50)
        self.rel_val_lbl.setText("50 ms")
        self.rel_slider.blockSignals(False)
        self._on_rel_changed(50)


class MiniOscilloscopeWidget(QWidget):
    """
    Mini-Osciloscopio a 60 FPS reales y 0% CPU con Scroll Blitting de 1 píxel.
    Acumula picos O(1) entre frames visuales sin bucles en Python ni sobrecarga de GPU.
    """
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(240, 22)
        self._pixmap = QPixmap(self.size())
        self._pixmap.fill(Qt.GlobalColor.transparent)
        self._pending_min = 0.0
        self._pending_max = 0.0
        self._pending_gr_db = 0.0
        self._has_new_data = False
        self._accum_steps = 0.0
        self._step_per_frame = 1.0  # Se ajusta según 1s, 5s, 10s
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Clic para desplegar / plegar el Osciloscopio detallado")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()

    def set_time_window(self, seconds: float, fps: float = 45.0):
        # Cuántos píxeles debe avanzar por frame para cubrir el ancho en 'seconds'
        total_frames = max(1.0, float(seconds) * fps)
        self._step_per_frame = self.width() / total_frames

    def push_audio_peaks(self, vmin: float, vmax: float, gr_db: float):
        # Acumula el pico más extremo entre frames visuales (O(1), sin np.roll)
        if not self._has_new_data:
            self._pending_min = vmin
            self._pending_max = vmax
            self._pending_gr_db = gr_db
            self._has_new_data = True
        else:
            if vmin < self._pending_min: self._pending_min = vmin
            if vmax > self._pending_max: self._pending_max = vmax
            if gr_db < self._pending_gr_db: self._pending_gr_db = gr_db

    def tick_render(self, wave_color: QColor):
        if not self._has_new_data:
            return
        self._accum_steps += self._step_per_frame
        dx = int(self._accum_steps)
        if dx < 1:
            return
        self._accum_steps -= dx

        w, h = self.width(), self.height()
        mid_y = h * 0.5

        # 1. Desplazar imagen existente dx píxeles a la izquierda (0% CPU)
        self._pixmap.scroll(-dx, 0, self._pixmap.rect())

        p = QPainter(self._pixmap)
        # 2. Limpiar solo la franja derecha nueva de ancho dx
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        p.fillRect(w - dx, 0, dx, h, Qt.GlobalColor.transparent)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

        # 3. Dibujar onda en la franja nueva
        y1 = int(np.clip(mid_y - self._pending_max * (mid_y - 2), 1, h - 1))
        y2 = int(np.clip(mid_y - self._pending_min * (mid_y - 2), 1, h - 1))
        p.setPen(QPen(wave_color, 1))
        for x in range(w - dx, w):
            p.drawLine(x, y1, x, max(y1 + 1, y2))

        # 4. Dibujar Reducción de Ganancia (Limitador) en púrpura desde el techo
        if self._pending_gr_db < -0.1:
            gr_h = int(min(h * 0.6, abs(self._pending_gr_db) * 1.2))
            p.fillRect(w - dx, 0, dx, max(2, gr_h), QColor(213, 0, 249, 210))

        p.end()
        self._has_new_data = False
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        clip = QPainterPath()
        clip.addRoundedRect(QRectF(self.rect()), 9.0, 9.0)
        p.setClipPath(clip)
        p.drawPixmap(0, 0, self._pixmap)

    def invalidate_canvas(self):
        self._pixmap.fill(Qt.GlobalColor.transparent)
        self.update()

    def set_current_gr(self, gr_db: float):
        self._pending_gr_db = float(gr_db)

    def refresh_scope(self):
        self.tick_render(QColor("#00E5FF"))


# Aliases de compatibilidad
CompactScopeGLWidget = MiniOscilloscopeWidget
_CompactScopeCanvas = MiniOscilloscopeWidget
OverlayMaskWidget = MiniOscilloscopeWidget


class AutoGainMiniBar(QWidget):
    toggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(78, 26)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Auto-Gain Independiente (Out A | Out B)\nClic para Activar / Desactivar")
        self._enabled = False
        self._hovered = False
        self._ag_a_db = 0.0
        self._ag_b_db = 0.0
        self._accent_color = QColor("#00E5FF")

    def setAccentColor(self, hex_color: str):
        if hex_color:
            self._accent_color = QColor(hex_color)
            self.update()

    def enterEvent(self, event):
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def set_state(self, enabled: bool, ag_a_db: float, ag_b_db: float = 0.0):
        changed = (
            (self._enabled != enabled)
            or (abs(self._ag_a_db - ag_a_db) > 0.05)
            or (abs(self._ag_b_db - ag_b_db) > 0.05)
        )
        self._enabled = enabled
        self._ag_a_db = float(np.clip(ag_a_db, -12.0, 12.0))
        self._ag_b_db = float(np.clip(ag_b_db, -12.0, 12.0))
        if changed:
            self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._enabled = not self._enabled
            self.toggled.emit(self._enabled)
            self.update()

    def _draw_bipolar_rail(self, p: QPainter, x: float, y: float, w: float, h: float, val_db: float):
        mid_y = y + h * 0.5
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(10, 12, 16, 200))
        p.drawRoundedRect(QRectF(x, y, w, h), 1.5, 1.5)

        if self._enabled and abs(val_db) > 0.05:
            norm = min(1.0, abs(val_db) / 12.0)
            fill_h = max(2.0, norm * (h * 0.5))
            if val_db < 0:
                p.setBrush(self._accent_color)
                p.drawRoundedRect(QRectF(x, mid_y, w, fill_h), 1.5, 1.5)
            else:
                p.setBrush(QColor(0, 230, 118, 230))
                p.drawRoundedRect(QRectF(x, mid_y - fill_h, w, fill_h), 1.5, 1.5)

        p.setPen(QPen(QColor(255, 255, 255, 120), 1.0))
        p.drawLine(QPointF(x - 0.5, mid_y), QPointF(x + w + 0.5, mid_y))

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = float(self.width()), float(self.height())
        rect = QRectF(1.2, 1.2, w - 2.4, h - 2.4)
        r = 6.0

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(16, 18, 24, 140 if self._enabled else (135 if self._hovered else 118)))
        p.drawRoundedRect(rect, r, r)

        if self._enabled:
            tint = QColor(self._accent_color)
            tint.setAlpha(48 if not self._hovered else 62)
            p.setBrush(tint)
            p.drawRoundedRect(rect, r, r)

            glow = QColor(self._accent_color)
            glow.setAlpha(36)
            p.setPen(QPen(glow, 2.2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(rect, r, r)

            border = _brighten_color(self._accent_color, 0.12)
            border.setAlpha(215)
            p.setPen(QPen(border, 1.25))
            p.drawRoundedRect(rect, r, r)
        else:
            if self._hovered:
                border = QColor(self._accent_color)
                border.setAlpha(135)
                p.setPen(QPen(border, 1.15))
            else:
                p.setPen(QPen(QColor(62, 68, 82, 155), 1.05))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(rect, r, r)

        # Dos rieles paralelos: Izquierdo (Out A) y Derecho (Out B)
        self._draw_bipolar_rail(p, 5.0, 4.0, 4.5, h - 8.0, self._ag_a_db)
        self._draw_bipolar_rail(p, 11.5, 4.0, 4.5, h - 8.0, self._ag_b_db)

        font = p.font()
        font.setPixelSize(8)
        font.setBold(True)
        p.setFont(font)

        if not self._enabled:
            p.setPen(QColor("#F0F4F8") if self._hovered else QColor("#C5CDD9"))
            p.drawText(QRectF(18, 0, w - 20, h), Qt.AlignmentFlag.AlignCenter, "AG OFF")
        else:
            acc_bright = _brighten_color(self._accent_color, 0.25)
            p.setPen(acc_bright if self._ag_a_db <= 0 else QColor(0, 230, 118))
            p.drawText(QRectF(18, 2, w - 20, 11), Qt.AlignmentFlag.AlignCenter, f"A:{self._ag_a_db:+.1f}")
            p.setPen(acc_bright if self._ag_b_db <= 0 else QColor(0, 230, 118))
            p.drawText(QRectF(18, 13, w - 20, 11), Qt.AlignmentFlag.AlignCenter, f"B:{self._ag_b_db:+.1f}")


class CrossoverMenuWidget(QWidget):
    """
    Panel de control para el Crossover Digital de Salidas (Out A / Out B)
    para alojar dentro de un QMenu con QWidgetAction con estética oscura (#14171F).
    """
    configChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(290)
        self.setStyleSheet("""
            QWidget {
                background-color: #14171F;
                color: #ECEFF4;
                font-family: "Segoe UI", sans-serif;
                font-size: 11px;
            }
            QCheckBox {
                font-size: 11px;
                font-weight: 600;
                color: #FFB300;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 15px;
                height: 15px;
                border: 1px solid #4A5263;
                border-radius: 3px;
                background-color: #1B1E28;
            }
            QCheckBox::indicator:checked {
                background-color: #FFB300;
                border: 1px solid #FFD54F;
            }
            QLabel {
                color: #9AA2B2;
                font-size: 10px;
                font-weight: 600;
            }
            QDoubleSpinBox {
                background-color: #1B1E28;
                color: #ECEFF4;
                border: 1px solid #333A4A;
                border-radius: 4px;
                padding: 3px 6px;
                font-size: 11px;
                font-weight: 600;
            }
            QDoubleSpinBox:focus {
                border: 1px solid #FFB300;
            }
            QComboBox {
                background-color: #1B1E28;
                color: #ECEFF4;
                border: 1px solid #333A4A;
                border-radius: 4px;
                padding: 3px 6px;
                font-size: 10px;
                font-weight: 500;
            }
            QComboBox:focus {
                border: 1px solid #FFB300;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        # 1. Header / Checkbox principal
        self.chk_enabled = QCheckBox("Activar Crossover (Out A / Out B)", self)
        self.chk_enabled.setCursor(Qt.CursorShape.PointingHandCursor)
        layout.addWidget(self.chk_enabled)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background-color: #242936; min-height: 1px; max-height: 1px;")
        layout.addWidget(sep)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)

        # 2. Frecuencia de Cruce (Hz)
        lbl_freq = QLabel("Frecuencia de Cruce:", self)
        self.spin_freq = QDoubleSpinBox(self)
        self.spin_freq.setRange(20.0, 20000.0)
        self.spin_freq.setDecimals(1)
        self.spin_freq.setSuffix(" Hz")
        self.spin_freq.setKeyboardTracking(False)
        self.spin_freq.setValue(80.0)
        grid.addWidget(lbl_freq, 0, 0)
        grid.addWidget(self.spin_freq, 0, 1)

        # 3. Tipo de Filtro
        lbl_type = QLabel("Tipo de Filtro:", self)
        self.combo_type = QComboBox(self)
        self.combo_type.addItems(["Linkwitz-Riley", "Butterworth"])
        style_dark_combobox(self.combo_type)
        grid.addWidget(lbl_type, 1, 0)
        grid.addWidget(self.combo_type, 1, 1)

        # 4. Pendiente (Slope)
        lbl_slope = QLabel("Pendiente (Slope):", self)
        self.combo_slope = QComboBox(self)
        self.combo_slope.addItem("12 dB/oct", 12)
        self.combo_slope.addItem("24 dB/oct", 24)
        self.combo_slope.addItem("48 dB/oct", 48)
        style_dark_combobox(self.combo_slope)
        grid.addWidget(lbl_slope, 2, 0)
        grid.addWidget(self.combo_slope, 2, 1)

        # 5. Asignación de Vías
        lbl_routing = QLabel("Asignación de Vías:", self)
        self.combo_routing = QComboBox(self)
        self.combo_routing.addItem("Out A: Tops (HP)  |  Out B: Sub (LP)", "A_Tops_B_Sub")
        self.combo_routing.addItem("Out A: Sub (LP)   |  Out B: Tops (HP)", "A_Sub_B_Tops")
        self.combo_routing.addItem("Out A: Full Range |  Out B: Sub (LP)", "A_Full_B_Sub")
        style_dark_combobox(self.combo_routing)
        grid.addWidget(lbl_routing, 3, 0)
        grid.addWidget(self.combo_routing, 3, 1)

        layout.addLayout(grid)

        # Connect internal signals
        self.chk_enabled.toggled.connect(self._on_control_changed)
        self.spin_freq.valueChanged.connect(self._on_control_changed)
        self.combo_type.currentIndexChanged.connect(self._on_control_changed)
        self.combo_slope.currentIndexChanged.connect(self._on_control_changed)
        self.combo_routing.currentIndexChanged.connect(self._on_control_changed)

    def _on_control_changed(self):
        self.configChanged.emit()

    def get_state(self) -> dict:
        return {
            "enabled": bool(self.chk_enabled.isChecked()),
            "freq_hz": float(self.spin_freq.value()),
            "filter_type": str(self.combo_type.currentText()),
            "slope_db": int(self.combo_slope.currentData() if self.combo_slope.currentData() is not None else 24),
            "routing_mode": str(self.combo_routing.currentData() if self.combo_routing.currentData() is not None else "A_Tops_B_Sub"),
        }

    def set_state(self, state: dict):
        self.chk_enabled.blockSignals(True)
        self.spin_freq.blockSignals(True)
        self.combo_type.blockSignals(True)
        self.combo_slope.blockSignals(True)
        self.combo_routing.blockSignals(True)

        self.chk_enabled.setChecked(bool(state.get("enabled", False)))
        self.spin_freq.setValue(float(state.get("freq_hz", 80.0)))
        
        ft = str(state.get("filter_type", "Linkwitz-Riley"))
        idx_t = self.combo_type.findText(ft)
        if idx_t >= 0:
            self.combo_type.setCurrentIndex(idx_t)

        sl = int(state.get("slope_db", 24))
        idx_s = self.combo_slope.findData(sl)
        if idx_s >= 0:
            self.combo_slope.setCurrentIndex(idx_s)

        rm = str(state.get("routing_mode", "A_Tops_B_Sub"))
        idx_r = self.combo_routing.findData(rm)
        if idx_r >= 0:
            self.combo_routing.setCurrentIndex(idx_r)

        self.chk_enabled.blockSignals(False)
        self.spin_freq.blockSignals(False)
        self.combo_type.blockSignals(False)
        self.combo_slope.blockSignals(False)
        self.combo_routing.blockSignals(False)


class EditInputSelector(QWidget):
    """
    Selector segmentado con estética HUD para alternar el banco de EQ activo:
    EQ: [ In 1 | In 2 ]
    """
    inputChanged = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFixedHeight(28)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 0, 2, 0)
        layout.setSpacing(4)

        lbl = QLabel("EQ:", self)
        lbl.setToolTip("Banco de EQ activo para editar (Entrada 1 o 2)")
        lbl.setStyleSheet("color: #9AA2B2; font-size: 10px; font-weight: bold; background: transparent;")
        layout.addWidget(lbl)

        self.btn_in1 = LiquidGlassButton("In 1", self, active_tint_color="#00E5FF", pill_radius=6.0)
        self.btn_in1.setCheckable(True)
        self.btn_in1.setChecked(True)
        self.btn_in1.setFixedWidth(44)
        self.btn_in1.setFixedHeight(24)
        self.btn_in1.setToolTip("Editar banco de EQ para Entrada 1 (Principal)")
        self.btn_in1.clicked.connect(lambda: self._select_input(1))
        layout.addWidget(self.btn_in1)

        self.btn_in2 = LiquidGlassButton("In 2", self, active_tint_color="#BA68C8", pill_radius=6.0)
        self.btn_in2.setCheckable(True)
        self.btn_in2.setChecked(False)
        self.btn_in2.setFixedWidth(44)
        self.btn_in2.setFixedHeight(24)
        self.btn_in2.setToolTip("Editar banco de EQ para Entrada 2 (Aux / Mic)")
        self.btn_in2.clicked.connect(lambda: self._select_input(2))
        layout.addWidget(self.btn_in2)

        self._active_input = 1

    def _select_input(self, inp: int):
        self._active_input = inp
        self.btn_in1.blockSignals(True)
        self.btn_in2.blockSignals(True)
        self.btn_in1.setChecked(inp == 1)
        self.btn_in2.setChecked(inp == 2)
        self.btn_in1.blockSignals(False)
        self.btn_in2.blockSignals(False)
        self.btn_in1.update()
        self.btn_in2.update()
        self.inputChanged.emit(inp)

    def set_active_input(self, inp: int):
        if inp in (1, 2) and inp != self._active_input:
            self._select_input(inp)
