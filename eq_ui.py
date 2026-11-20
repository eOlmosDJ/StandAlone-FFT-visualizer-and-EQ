import math
import os
import re
import sys
import time
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QFrame, QMenu, QColorDialog, QDialog, QToolTip,
    QGraphicsOpacityEffect, QListView, QSlider, QGraphicsPathItem, QGraphicsItem,
    QApplication, QCheckBox, QDoubleSpinBox, QSpinBox, QGridLayout, QWidgetAction,
    QSizePolicy
)
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import (
    Qt, QEvent, QObject, QPoint, QPointF, QRectF, QSize, QByteArray,
    pyqtSignal, QTimer, QVariantAnimation, QEasingCurve
)
from PyQt6.QtGui import (
    QColor, QFont, QFontMetricsF, QPainter, QPainterPath, QPen, QBrush,
    QLinearGradient, QRadialGradient, QPolygonF, QPixmap, QImage, QIcon, QCursor
)
from PyQt6.QtSvg import QSvgRenderer
import pyqtgraph as pg

from eq_dsp import (
    ParametricEQEngine,
    EQBand,
    FILTER_TYPES,
    SLOPE_OPTIONS,
    CHANNEL_MODES,
    TARGET_OUTPUTS,
    MB_MODES,
    MB_CHARACTER_MODES,
    MB_MODE_COLORS,
    MAX_MB_BANDS,
    DynamicsBand,
    MultibandDynamicsProcessor,
)


def _resolve_app_and_plot(widget: QWidget | None):
    """Localiza el AudioAnalyzerApp principal, su plot_widget OpenGL y el eq_overlay desde cualquier widget."""
    cur = widget
    visited = set()
    while cur is not None and id(cur) not in visited:
        visited.add(id(cur))
        if hasattr(cur, "plot_widget") and hasattr(cur, "eq_overlay") and getattr(cur, "plot_widget", None) is not None:
            return cur, cur.plot_widget, cur.eq_overlay
        if hasattr(cur, "overlay") and getattr(cur, "overlay", None) is not None:
            ov = cur.overlay
            if hasattr(ov, "plot_widget") and hasattr(ov, "eq_engine"):
                return getattr(ov.plot_widget, "window", lambda: None)(), ov.plot_widget, ov
            if hasattr(ov, "app_ref") and getattr(ov, "app_ref", None) is not None:
                app_r = ov.app_ref
                return app_r, getattr(app_r, "plot_widget", None), getattr(app_r, "eq_overlay", None)
        if hasattr(cur, "app_ref") and getattr(cur, "app_ref", None) is not None:
            app_r = cur.app_ref
            return app_r, getattr(app_r, "plot_widget", None), getattr(app_r, "eq_overlay", None)
        try:
            win = cur.window()
            if win is not None and win is not cur and hasattr(win, "plot_widget") and hasattr(win, "eq_overlay"):
                return win, win.plot_widget, win.eq_overlay
        except Exception:
            pass
        cur = cur.parentWidget() if hasattr(cur, "parentWidget") else None

    for top_w in QApplication.topLevelWidgets():
        if hasattr(top_w, "plot_widget") and hasattr(top_w, "eq_overlay"):
            return top_w, top_w.plot_widget, top_w.eq_overlay
    return None, None, None


class _HUDPopupGLHelper:
    """
    Provee pintado HUD Apple Liquid Glass con Gaussian Blur compartido a 60 FPS,
    refracción óptica en bordes y halo difuso coloreado para menús y popups desplegables,
    manteniendo registro activo para repintado animado en tiempo real sin crear ítems OpenGL redundantes.
    """

    _active_helpers: set["_HUDPopupGLHelper"] = set()

    def __init__(self, widget: QWidget, accent_color: str = "#00E5FF", corner_radius: float = 10.0):
        self.widget = widget
        self.accent_color = QColor(accent_color) if accent_color else QColor("#00E5FF")
        self.corner_radius = float(corner_radius)
        self._bg_gl_item = None
        self._blur_pixmap = None
        self._blur_frame_id = -1
        self._blur_geom_key = None
        self._last_blur_time = 0.0

    def set_accent_color(self, hex_or_color):
        c = QColor(hex_or_color)
        if c.isValid() and c != self.accent_color:
            self.accent_color = c
            self.sync_gl_item()
            if self.widget is not None:
                self.widget.update()

    def build_rounded_path(self, w: float, h: float) -> QPainterPath:
        pad = 2.5
        path = QPainterPath()
        path.addRoundedRect(
            QRectF(pad, pad, max(2.0, w - 2.0 * pad), max(2.0, h - 2.0 * pad)),
            self.corner_radius,
            self.corner_radius,
        )
        return path

    def sync_gl_item(self):
        if self._bg_gl_item is not None:
            self._bg_gl_item.hide()
        if self.widget is not None:
            try:
                vis = self.widget.isVisible()
            except RuntimeError:
                _HUDPopupGLHelper._active_helpers.discard(self)
                return
            if vis:
                _HUDPopupGLHelper._active_helpers.add(self)
                self.widget.update()
            else:
                _HUDPopupGLHelper._active_helpers.discard(self)
        else:
            _HUDPopupGLHelper._active_helpers.discard(self)

    def hide_gl_item(self):
        _HUDPopupGLHelper._active_helpers.discard(self)
        if self._bg_gl_item is not None:
            self._bg_gl_item.hide()

    @classmethod
    def repaint_active_popups(cls):
        updated_widgets = set()
        for h in list(cls._active_helpers):
            w = h.widget
            if w is None:
                cls._active_helpers.discard(h)
                continue
            try:
                if not w.isVisible():
                    cls._active_helpers.discard(h)
                    continue
                updated_widgets.add(id(w))
                w.update()
                for lv in w.findChildren(QListView):
                    vp = lv.viewport()
                    if vp is not None:
                        vp.update()
            except RuntimeError:
                cls._active_helpers.discard(h)

        active_pop = QApplication.activePopupWidget()
        if active_pop is not None and id(active_pop) not in updated_widgets:
            try:
                if active_pop.isVisible():
                    active_pop.update()
                    for lv in active_pop.findChildren(QListView):
                        vp = lv.viewport()
                        if vp is not None:
                            vp.update()
            except RuntimeError:
                pass

    def paint_hud_background(self, painter: QPainter):
        if self.widget is not None:
            try:
                if self.widget.isVisible():
                    _HUDPopupGLHelper._active_helpers.add(self)
            except RuntimeError:
                return

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        w = float(self.widget.width())
        h = float(self.widget.height())
        path = self.build_rounded_path(w, h)

        _, pw, ov = _resolve_app_and_plot(self.widget)
        if pw is not None and ov is not None:
            global_tl = self.widget.mapToGlobal(QPoint(0, 0))
            local_in_pw = pw.mapFromGlobal(global_tl)
            scene_pt = pw.mapToScene(local_in_pw)
            sx = float(scene_pt.x())
            sy = float(scene_pt.y())
            now = time.perf_counter()
            cur_fid = HUDOpenGLBackgroundItem._scene_blur_frame_id
            geom_key = (int(round(sx)), int(round(sy)), int(round(w)), int(round(h)), round(self.corner_radius, 1))
            if (
                self._blur_pixmap is None
                or self._blur_frame_id != cur_fid
                or self._blur_geom_key != geom_key
                or (now - HUDOpenGLBackgroundItem._scene_blur_last_time) >= 0.036
            ):
                pix = HUDOpenGLBackgroundItem.render_scene_blur_pixmap(
                    ov, pw, sx, sy, w, h, self.accent_color, corner_radius=self.corner_radius
                )
                if pix is not None:
                    self._blur_pixmap = pix
                    self._blur_frame_id = HUDOpenGLBackgroundItem._scene_blur_frame_id
                    self._blur_geom_key = geom_key

        if self._blur_pixmap is not None and not self._blur_pixmap.isNull():
            painter.save()
            painter.setClipPath(path)
            painter.drawPixmap(self.widget.rect(), self._blur_pixmap)
            painter.fillRect(self.widget.rect(), QColor(10, 12, 18, 96))
            tint = QColor(self.accent_color)
            tint.setAlpha(12)
            painter.fillRect(self.widget.rect(), tint)
            painter.restore()
        else:
            painter.fillPath(path, QBrush(QColor(10, 12, 18, 180)))
            tint = QColor(self.accent_color)
            tint.setAlpha(14)
            painter.fillPath(path, QBrush(tint))

        draw_liquid_glass_border_glow(painter, path, QRectF(0.0, 0.0, w, h), self.accent_color)


class HUDPopupDialog(QDialog):
    """
    Diálogo desplegable sin bordes con el diseño del HUD (fondo translúcido de baja opacidad +
    Gaussian Blur en tiempo real + borde coloreado y sincronización OpenGL en GPU).
    """

    def __init__(self, parent=None, accent_color: str = "#00E5FF", corner_radius: float = 10.0):
        super().__init__(
            parent,
            Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint | Qt.WindowType.NoDropShadowWindowHint,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._hud_gl = _HUDPopupGLHelper(self, accent_color=accent_color, corner_radius=corner_radius)

    def setAccentColor(self, hex_color: str):
        self._hud_gl.set_accent_color(hex_color)

    def showEvent(self, event):
        super().showEvent(event)
        self._hud_gl._last_blur_time = 0.0
        self._hud_gl._blur_geom_key = None
        self._hud_gl.sync_gl_item()

    def moveEvent(self, event):
        super().moveEvent(event)
        self._hud_gl.sync_gl_item()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._hud_gl._last_blur_time = 0.0
        self._hud_gl._blur_geom_key = None
        self._hud_gl.sync_gl_item()

    def hideEvent(self, event):
        super().hideEvent(event)
        self._hud_gl.hide_gl_item()

    def paintEvent(self, event):
        painter = QPainter(self)
        self._hud_gl.paint_hud_background(painter)


class HUDPopupMenu(QMenu):
    """
    Menú desplegable QMenu con el diseño del HUD (fondo translúcido de baja opacidad +
    Gaussian Blur en tiempo real + borde coloreado y sincronización OpenGL en GPU).
    """

    def __init__(self, title_or_parent=None, parent=None, accent_color: str = "#00E5FF", corner_radius: float = 9.0):
        if isinstance(title_or_parent, str):
            super().__init__(title_or_parent, parent)
        else:
            super().__init__(title_or_parent)
        self.setWindowFlags(
            Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint | Qt.WindowType.NoDropShadowWindowHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._hud_gl = _HUDPopupGLHelper(self, accent_color=accent_color, corner_radius=corner_radius)
        self._apply_menu_qss(accent_color)

    def _apply_menu_qss(self, accent_hex: str):
        c = QColor(accent_hex) if accent_hex else QColor("#00E5FF")
        r, g, b = c.red(), c.green(), c.blue()
        bright_hex = _brighten_color(c, 0.35).name()
        self.setStyleSheet(f"""
            QMenu {{
                background-color: transparent;
                color: #ECEFF4;
                border: none;
                padding: 6px 5px;
                font-family: "Segoe UI", sans-serif;
                font-size: 11px;
                font-weight: 600;
            }}
            QMenu::item {{
                background-color: transparent;
                padding: 5px 22px 5px 12px;
                margin: 1px 4px;
                border-radius: 5px;
                color: #D8DEE9;
            }}
            QMenu::item:selected {{
                background-color: rgba({r}, {g}, {b}, 52);
                color: {bright_hex};
                border: 1px solid rgba({r}, {g}, {b}, 185);
            }}
            QMenu::item:disabled {{
                color: #6C7486;
            }}
            QMenu::separator {{
                height: 1px;
                background-color: rgba(255, 255, 255, 28);
                margin: 4px 8px;
            }}
        """)

    def setAccentColor(self, hex_color: str):
        self._hud_gl.set_accent_color(hex_color)
        self._apply_menu_qss(hex_color)

    def addMenu(self, title: str) -> "HUDPopupMenu":
        sub = HUDPopupMenu(title, self, accent_color=self._hud_gl.accent_color.name(), corner_radius=self._hud_gl.corner_radius)
        super().addMenu(sub)
        return sub

    def showEvent(self, event):
        super().showEvent(event)
        self._hud_gl._last_blur_time = 0.0
        self._hud_gl._blur_geom_key = None
        self._hud_gl.sync_gl_item()

    def moveEvent(self, event):
        super().moveEvent(event)
        self._hud_gl.sync_gl_item()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._hud_gl._last_blur_time = 0.0
        self._hud_gl._blur_geom_key = None
        self._hud_gl.sync_gl_item()

    def hideEvent(self, event):
        super().hideEvent(event)
        self._hud_gl.hide_gl_item()

    def paintEvent(self, event):
        painter = QPainter(self)
        self._hud_gl.paint_hud_background(painter)
        painter.end()
        super().paintEvent(event)


class _ComboPopupFilter(QObject):
    """Sincroniza y pinta el contenedor desplegable de cualquier QComboBox con diseño HUD + Liquid Glass."""

    def __init__(self, combo: QComboBox, accent_color: str = "#00E5FF"):
        super().__init__(combo)
        self.combo = combo
        self.accent_color = accent_color
        self._helper = None

    def set_accent_color(self, accent_color: str):
        self.accent_color = str(accent_color) if accent_color else "#00E5FF"
        if self._helper is not None:
            self._helper.set_accent_color(self.accent_color)

    def eventFilter(self, obj, event):
        etype = event.type()
        if etype in (QEvent.Type.Show, QEvent.Type.Move, QEvent.Type.Resize):
            if self._helper is None:
                self._helper = _HUDPopupGLHelper(obj, accent_color=self.accent_color, corner_radius=7.0)
            if etype in (QEvent.Type.Show, QEvent.Type.Resize):
                self._helper._blur_geom_key = None
            self._helper.sync_gl_item()
        elif etype == QEvent.Type.Hide:
            if self._helper is not None:
                self._helper.hide_gl_item()
        elif etype == QEvent.Type.Paint:
            if self._helper is None:
                self._helper = _HUDPopupGLHelper(obj, accent_color=self.accent_color, corner_radius=7.0)
            p = QPainter(obj)
            self._helper.paint_hud_background(p)
            p.end()
            return True
        return False


def style_dark_combobox(combo: QComboBox, accent_color: str = "#00E5FF"):
    """Estiliza la lista desplegable de un QComboBox con el diseño Liquid Glass HUD en tiempo real."""
    c = QColor(accent_color) if accent_color else QColor("#00E5FF")
    r, g, b = c.red(), c.green(), c.blue()
    bright_hex = _brighten_color(c, 0.35).name()
    qss = f"""
        QListView {{
            background-color: transparent;
            color: #F0F2F5;
            border: none;
            border-radius: 7px;
            padding: 5px;
            outline: none;
        }}
        QListView::item {{
            min-height: 22px;
            padding: 3px 10px;
            border-radius: 4px;
            color: #D8DEE9;
            background-color: transparent;
        }}
        QListView::item:hover {{
            background-color: rgba({r}, {g}, {b}, 36);
            color: #FFFFFF;
        }}
        QListView::item:selected {{
            background-color: rgba({r}, {g}, {b}, 62);
            color: {bright_hex};
            border: 1px solid rgba({r}, {g}, {b}, 185);
            font-weight: bold;
        }}
    """
    existing_flt = getattr(combo, "_hud_popup_filter", None)
    existing_view = combo.view()
    if existing_flt is not None and isinstance(existing_view, QListView):
        existing_view.setStyleSheet(qss)
        existing_flt.set_accent_color(c.name())
        return

    list_view = QListView(combo)
    list_view.setUniformItemSizes(True)
    list_view.setSpacing(2)
    list_view.setFrameShape(QFrame.Shape.NoFrame)
    list_view.setAutoFillBackground(False)
    list_view.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
    vp = list_view.viewport()
    if vp is not None:
        vp.setAutoFillBackground(False)
        vp.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
    list_view.setStyleSheet(qss)
    combo.setView(list_view)
    try:
        popup_win = combo.view().window()
        popup_win.setWindowFlags(
            Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint | Qt.WindowType.NoDropShadowWindowHint
        )
        popup_win.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        popup_win.setAutoFillBackground(False)
        if hasattr(popup_win, "setFrameShape"):
            popup_win.setFrameShape(QFrame.Shape.NoFrame)
        popup_win.setStyleSheet("QFrame { background: transparent; border: none; }")
        flt = _ComboPopupFilter(combo, accent_color=c.name())
        combo._hud_popup_filter = flt
        popup_win.installEventFilter(flt)
    except Exception:
        pass


class _SpinBoxArrowOverlay(QWidget):
    """
     Superposición ligera a la derecha del QSpinBox / QDoubleSpinBox que dibuja
    flechas vectoriales nítidas ▴ (subir) y ▾ (bajar) con estética HUD.
    """

    def __init__(self, spinbox, accent_color: str = "#00E5FF"):
        super().__init__(spinbox)
        self.spinbox = spinbox
        self.accent_color = QColor(accent_color) if accent_color else QColor("#00E5FF")
        self._hover_part = None  # "up", "down", or None
        self._pressed_part = None
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._sync_geometry()

    def _sync_geometry(self):
        btn_w = 20
        self.setGeometry(max(0, self.spinbox.width() - btn_w), 0, btn_w, self.spinbox.height())
        self.raise_()

    def mouseMoveEvent(self, event):
        part = "up" if event.position().y() < self.height() * 0.5 else "down"
        if part != self._hover_part:
            self._hover_part = part
            self.update()
        event.accept()

    def leaveEvent(self, event):
        self._hover_part = None
        self._pressed_part = None
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.spinbox.isEnabled():
            part = "up" if event.position().y() < self.height() * 0.5 else "down"
            self._pressed_part = part
            if part == "up":
                self.spinbox.stepUp()
            else:
                self.spinbox.stepDown()
            self.update()
        event.accept()

    def mouseReleaseEvent(self, event):
        self._pressed_part = None
        self.update()
        event.accept()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w = float(self.width())
        h = float(self.height())
        mid_y = h * 0.5

        # Línea divisoria vertical e intermedia sutil
        p.setPen(QPen(QColor(255, 255, 255, 28), 1.0))
        p.drawLine(QPointF(0.5, 3.0), QPointF(0.5, h - 3.0))
        p.drawLine(QPointF(2.0, mid_y), QPointF(w - 2.0, mid_y))

        for part, cy, is_up in (("up", mid_y * 0.5 + 0.5, True), ("down", mid_y * 1.5 - 0.5, False)):
            hovered = (self._hover_part == part) and self.spinbox.isEnabled()
            pressed = (self._pressed_part == part) and self.spinbox.isEnabled()
            if pressed or hovered:
                sub_rect = QRectF(1.5, 1.5 if is_up else mid_y + 0.5, w - 3.0, mid_y - 2.0)
                bg = QColor(self.accent_color)
                bg.setAlpha(68 if pressed else 35)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(bg)
                p.drawRoundedRect(sub_rect, 3.0, 3.0)

            cx = w * 0.5
            aw = 3.6
            ah = 2.4
            tri = QPolygonF()
            if is_up:
                tri.append(QPointF(cx, cy - ah))
                tri.append(QPointF(cx - aw, cy + ah))
                tri.append(QPointF(cx + aw, cy + ah))
            else:
                tri.append(QPointF(cx, cy + ah))
                tri.append(QPointF(cx - aw, cy - ah))
                tri.append(QPointF(cx + aw, cy - ah))

            if not self.spinbox.isEnabled():
                col = QColor(110, 116, 128, 110)
            elif pressed:
                col = _brighten_color(self.accent_color, 0.45)
            elif hovered:
                col = _brighten_color(self.accent_color, 0.25)
            else:
                col = QColor("#D8DEE9")

            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(col))
            p.drawPolygon(tri)


class _SpinBoxResizeFilter(QObject):
    def __init__(self, spinbox, overlay: _SpinBoxArrowOverlay):
        super().__init__(spinbox)
        self.overlay = overlay

    def eventFilter(self, obj, event):
        if event.type() in (QEvent.Type.Resize, QEvent.Type.Show):
            self.overlay._sync_geometry()
        return False


def style_hud_spinbox(spinbox, accent_color: str = "#00E5FF"):
    """
    Aplica la estética del HUD a un QSpinBox / QDoubleSpinBox y añade flechas
    vectoriales explícitas ▴ (subir) y ▾ (bajar) en los botones laterales.
    """
    c = QColor(accent_color) if accent_color else QColor("#00E5FF")
    r, g, b = c.red(), c.green(), c.blue()
    hex_c = c.name()
    spinbox.setStyleSheet(f"""
        QSpinBox, QDoubleSpinBox {{
            background-color: rgba(16, 18, 24, 145);
            color: {hex_c};
            border: 1px solid rgba(62, 68, 82, 170);
            border-radius: 5px;
            padding: 3px 22px 3px 6px;
            font-size: 11px;
            font-weight: bold;
        }}
        QSpinBox:hover, QDoubleSpinBox:hover, QSpinBox:focus, QDoubleSpinBox:focus {{
            border: 1px solid {hex_c};
            background-color: rgba({r}, {g}, {b}, 26);
        }}
        QSpinBox::up-button, QDoubleSpinBox::up-button,
        QSpinBox::down-button, QDoubleSpinBox::down-button {{
            width: 0px;
            border: none;
            background: transparent;
        }}
    """)
    if not hasattr(spinbox, "_hud_arrow_overlay"):
        ov = _SpinBoxArrowOverlay(spinbox, accent_color=hex_c)
        flt = _SpinBoxResizeFilter(spinbox, ov)
        spinbox._hud_arrow_overlay = ov
        spinbox._hud_resize_filter = flt
        spinbox.installEventFilter(flt)
        ov._sync_geometry()
        ov.show()
    else:
        spinbox._hud_arrow_overlay.accent_color = c
        spinbox._hud_arrow_overlay._sync_geometry()
        spinbox._hud_arrow_overlay.update()


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


def draw_liquid_glass_border_glow(painter: QPainter, path: QPainterPath, rect: QRectF, accent_color: QColor):
    """Dibuja brillo interior difuso (inward bleed), halo exterior suave y bisel especular direccional."""
    c = QColor(accent_color) if (accent_color is not None and QColor(accent_color).isValid()) else QColor("#00E5FF")
    r, g, b = c.red(), c.green(), c.blue()

    # 1. Sangrado interior suave hacia el vidrio esmerilado y brillo especular superior
    painter.save()
    painter.setClipPath(path)

    sheen_h = min(22.0, rect.height() * 0.35)
    sheen_grad = QLinearGradient(0.0, rect.top(), 0.0, rect.top() + sheen_h)
    sheen_grad.setColorAt(0.0, QColor(255, 255, 255, 16))
    sheen_grad.setColorAt(1.0, QColor(255, 255, 255, 0))
    painter.fillRect(QRectF(rect.left(), rect.top(), rect.width(), sheen_h), sheen_grad)

    painter.setBrush(Qt.BrushStyle.NoBrush)
    for alpha, width in ((15, 12.0), (28, 6.0), (48, 2.8)):
        painter.setPen(QPen(QColor(r, g, b, alpha), width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.drawPath(path)
    painter.restore()

    # 2. Halo exterior difuso
    painter.setBrush(Qt.BrushStyle.NoBrush)
    for alpha, width in ((18, 5.6), (38, 3.0)):
        painter.setPen(QPen(QColor(r, g, b, alpha), width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.drawPath(path)

    # 3. Bisel especular direccional + borde nítido (1.25 px)
    rim_grad = QLinearGradient(rect.left(), rect.top(), rect.left() + rect.width() * 0.45, rect.bottom())
    c0 = _brighten_color(c, 0.45)
    c0.setAlpha(230)
    c1 = _brighten_color(c, 0.12)
    c1.setAlpha(200)
    c2 = QColor(r, g, b, 165)
    c3 = _brighten_color(c, 0.20)
    c3.setAlpha(195)
    rim_grad.setColorAt(0.0, c0)
    rim_grad.setColorAt(0.35, c1)
    rim_grad.setColorAt(0.75, c2)
    rim_grad.setColorAt(1.0, c3)
    painter.setPen(QPen(QBrush(rim_grad), 1.25, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    painter.drawPath(path)


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


# =========================================================================
# MOTOR DE ICONOS VECTORIALES SVG CON ILUMINACIÓN NEÓN Y CACHÉ QPIXMAP
# =========================================================================

_SVG_RAW_CACHE: dict[str, str] = {}
_SVG_PIXMAP_CACHE: dict[tuple, QPixmap] = {}

_SVG_PRIMARY_HEXES = (
    "#35C6E8", "#648BFF", "#B18CFF", "#55D6A3",
    "#F078C8", "#FFA94D", "#F4D35E", "#FF6B78",
)
_SVG_SECONDARY_HEXES = ("#8290A5", "#DCE4EF")

FILTER_TYPE_SVGS: dict[str, str] = {
    "Bell": "bell.svg",
    "Low Shelf": "low_shelf.svg",
    "High Shelf": "high_shelf.svg",
    "Low Cut": "low_cut.svg",
    "High Cut": "high_cut.svg",
    "Notch": "notch.svg",
    "Band Pass": "band_pass.svg",
    "Tilt Shelf": "tilt_shelf.svg",
    "All Pass": "all_pass.svg",
}

CHANNEL_MODE_SVGS: dict[str, str] = {
    "LR": "stereo.svg",
    "L": "left.svg",
    "R": "right.svg",
    "Mid": "mid.svg",
    "Side": "side.svg",
}

TARGET_OUT_SVGS: dict[str, str] = {
    "A+B": "ab.svg",
    "A": "a.svg",
    "B": "b.svg",
}

MB_MODE_SVGS: dict[str, str] = {
    "Compress": "comp_mode.svg",
    "Expand": "exp_mode.svg",
    "Limit": "lim_mode.svg",
}


def _get_svg_icons_dir() -> str:
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    cand = os.path.join(base, "dynamic_eq_svg_icons")
    if os.path.isdir(cand):
        return cand
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "dynamic_eq_svg_icons")


def _load_svg_raw(svg_name: str) -> str:
    fname = svg_name if svg_name.endswith(".svg") else f"{svg_name}.svg"
    if fname in _SVG_RAW_CACHE:
        return _SVG_RAW_CACHE[fname]
    path = os.path.join(_get_svg_icons_dir(), fname)
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
    except Exception:
        raw = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 48 48" fill="none">'
            '<circle cx="24" cy="24" r="14" stroke="#35C6E8" stroke-width="2.5"/></svg>'
        )
    _SVG_RAW_CACHE[fname] = raw
    return raw


def _scale_svg_strokes(svg_xml: str, factor: float) -> str:
    def _repl(m):
        try:
            val = float(m.group(1)) * factor
            return f'stroke-width="{val:.2f}"'
        except Exception:
            return m.group(0)
    return re.sub(r'stroke-width="([0-9.]+)"', _repl, svg_xml)


def _recolor_svg_xml(svg_xml: str, primary_hex: str | None, secondary_hex: str | None, preserve_off_red: bool = False) -> str:
    out = svg_xml
    if primary_hex:
        for hx in _SVG_PRIMARY_HEXES:
            if preserve_off_red and hx.upper() == "#FF6B78":
                out = re.sub(hx, "#D45D6A", out, flags=re.IGNORECASE)
            else:
                out = re.sub(hx, primary_hex, out, flags=re.IGNORECASE)
    if secondary_hex:
        for hx in _SVG_SECONDARY_HEXES:
            out = re.sub(hx, secondary_hex, out, flags=re.IGNORECASE)
    return out


def _render_svg_pixmap(
    svg_name: str,
    w: int,
    h: int,
    dpr: float = 1.0,
    state: str = "off",
    accent_hex: str | None = None,
    recolor_primary: bool = True,
) -> QPixmap:
    """
    Renderiza un icono SVG de dynamic_eq_svg_icons con 3 estados de iluminación:
    - 'disabled': gris tenue apagado
    - 'off': gris metálico mate sin resplandor (inactivo)
    - 'hover': pre-encendido suave con halo discreto
    - 'on' / 'lit': encendido neón multicapa (halo ancho + halo medio + núcleo brillante)
    - 'semantic': mantiene la paleta semántica multicolor original del SVG con iluminación limpia
    """
    fname = svg_name if svg_name.endswith(".svg") else f"{svg_name}.svg"
    key = (fname, int(w), int(h), round(float(dpr), 2), state, str(accent_hex or ""), bool(recolor_primary))
    cached = _SVG_PIXMAP_CACHE.get(key)
    if cached is not None:
        return cached

    pw = max(1, int(round(w * dpr)))
    ph = max(1, int(round(h * dpr)))
    pix = QPixmap(pw, ph)
    pix.setDevicePixelRatio(dpr)
    pix.fill(Qt.GlobalColor.transparent)

    raw = _load_svg_raw(fname)
    acc = QColor(accent_hex) if (accent_hex and QColor(accent_hex).isValid()) else QColor("#35C6E8")
    target_rect = QRectF(0.0, 0.0, float(w), float(h))

    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

    if state == "disabled":
        xml_dis = _recolor_svg_xml(raw, "#484F5E", "#363B46")
        QSvgRenderer(QByteArray(xml_dis.encode("utf-8"))).render(p, target_rect)
    elif state == "off":
        is_off_icon = fname == "off.svg"
        xml_off = _recolor_svg_xml(raw, "#788296", "#4F5768", preserve_off_red=is_off_icon)
        QSvgRenderer(QByteArray(xml_off.encode("utf-8"))).render(p, target_rect)
    elif state == "semantic":
        xml_glow = _scale_svg_strokes(raw, 2.0)
        p.setOpacity(0.28)
        QSvgRenderer(QByteArray(xml_glow.encode("utf-8"))).render(p, target_rect)
        p.setOpacity(1.0)
        QSvgRenderer(QByteArray(raw.encode("utf-8"))).render(p, target_rect)
    elif state == "hover":
        prim = _brighten_color(acc, 0.16).name() if recolor_primary else None
        sec = "#AAB6CA"
        xml_hov = _recolor_svg_xml(raw, prim, sec)
        xml_glow = _scale_svg_strokes(_recolor_svg_xml(raw, acc.name() if recolor_primary else None, acc.name()), 2.1)
        p.setOpacity(0.26)
        QSvgRenderer(QByteArray(xml_glow.encode("utf-8"))).render(p, target_rect)
        p.setOpacity(0.95)
        QSvgRenderer(QByteArray(xml_hov.encode("utf-8"))).render(p, target_rect)
        p.setOpacity(1.0)
    else:  # 'on' / 'lit' — Iluminación neón completa al activarse
        glow_outer_c = acc.name() if recolor_primary else None
        glow_mid_c = _brighten_color(acc, 0.22).name() if recolor_primary else None
        core_c = _brighten_color(acc, 0.44).name() if recolor_primary else None
        sec_c = _brighten_color(acc, 0.08).name() if recolor_primary else "#C6D2E4"

        xml_outer = _scale_svg_strokes(_recolor_svg_xml(raw, glow_outer_c, glow_outer_c), 2.8)
        xml_mid = _scale_svg_strokes(_recolor_svg_xml(raw, glow_mid_c, sec_c), 1.65)
        xml_core = _recolor_svg_xml(raw, core_c, "#DCE6F4")

        p.setOpacity(0.32)
        QSvgRenderer(QByteArray(xml_outer.encode("utf-8"))).render(p, target_rect)
        p.setOpacity(0.58)
        QSvgRenderer(QByteArray(xml_mid.encode("utf-8"))).render(p, target_rect)
        p.setOpacity(1.0)
        QSvgRenderer(QByteArray(xml_core.encode("utf-8"))).render(p, target_rect)

    p.end()
    if len(_SVG_PIXMAP_CACHE) > 512:
        _SVG_PIXMAP_CACHE.clear()
    _SVG_PIXMAP_CACHE[key] = pix
    return pix


def _make_svg_qicon(
    svg_name: str,
    size: int = 20,
    accent_hex: str | None = None,
    recolor_primary: bool = False,
) -> QIcon:
    state = "on" if recolor_primary else "semantic"
    pix = _render_svg_pixmap(svg_name, size, size, dpr=2.0, state=state, accent_hex=accent_hex, recolor_primary=recolor_primary)
    return QIcon(pix)


class PowerToggleButton(QPushButton):
    """
    Selector gráfico ON/OFF basado en on.svg / off.svg de dynamic_eq_svg_icons (sin borde exterior: el SVG es el botón).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(True)
        self.setFixedSize(26, 26)
        self._accent_color = QColor("#55D6A3")
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
        dpr = float(self.devicePixelRatioF()) if hasattr(self, "devicePixelRatioF") else 1.0
        enabled = self.isEnabled()
        checked = self.isChecked() and enabled

        svg_file = "on.svg" if self.isChecked() else "off.svg"
        st = "disabled" if not enabled else ("on" if checked else ("hover" if self._hovered else "off"))
        isz = max(16, int(min(w, h) - 2))
        pix = _render_svg_pixmap(
            svg_file, isz, isz, dpr=dpr, state=st,
            accent_hex=self._accent_color.name(), recolor_primary=self.isChecked()
        )
        painter.drawPixmap(int((w - isz) * 0.5), int((h - isz) * 0.5), pix)


class SvgToggleButton(QPushButton):
    """
    Elemento gráfico SVG puro (para DYN, ISOLATE, SOLO, BYPASS, X-OVER):
    sin borde exterior de botón — el propio SVG es el botón y se ilumina al activarse.
    """

    def __init__(
        self,
        svg_name: str,
        label_fallback: str = "",
        parent=None,
        active_tint_color: str = "#00E5FF",
        pill_radius: float = 6.0,
        checkable_on_click: bool = True,
        off_svg_name: str | None = None,
    ):
        super().__init__(label_fallback, parent)
        self._svg_name = svg_name
        self._off_svg_name = off_svg_name
        self._active_color = QColor(active_tint_color)
        self._pill_radius = float(pill_radius)
        self._hovered = False
        self._force_active = False
        self.setCheckable(bool(checkable_on_click))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)

    def setForceActive(self, active: bool):
        a = bool(active)
        if a != self._force_active:
            self._force_active = a
            self.update()

    def setActiveTintColor(self, hex_color: str):
        c = QColor(hex_color)
        if c.isValid() and c != self._active_color:
            self._active_color = c
            self.update()

    def setAccentColor(self, hex_color: str):
        self.setActiveTintColor(hex_color)

    def enterEvent(self, event):
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w = float(self.width())
        h = float(self.height())
        dpr = float(self.devicePixelRatioF()) if hasattr(self, "devicePixelRatioF") else 1.0
        enabled = self.isEnabled()
        active = enabled and (self.isChecked() or self._force_active or self.isDown())

        st = "disabled" if not enabled else ("on" if active else ("hover" if self._hovered else "off"))
        icon_name = self._svg_name if (active or not self._off_svg_name) else self._off_svg_name
        isz = max(16, int(min(w, h) - 1))
        pix = _render_svg_pixmap(
            icon_name, isz, isz, dpr=dpr, state=st,
            accent_hex=self._active_color.name(), recolor_primary=True
        )
        p.drawPixmap(int((w - isz) * 0.5), int((h - isz) * 0.5), pix)


class SvgActionButton(QPushButton):
    """
    Botón de acción puramente gráfico basado en SVG (Prev, Next, Add, Delete, +MB, ←EQ):
    sin borde exterior — el propio SVG es el botón y se ilumina al pasar el cursor o presionar.
    """

    def __init__(
        self,
        svg_name: str,
        label_fallback: str = "",
        parent=None,
        active_tint_color: str = "#35C6E8",
        semantic_idle: bool = False,
    ):
        super().__init__(label_fallback, parent)
        self._svg_name = svg_name
        self._active_color = QColor(active_tint_color)
        self._semantic_idle = bool(semantic_idle)
        self._hovered = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)

    def setActiveTintColor(self, hex_color: str):
        c = QColor(hex_color)
        if c.isValid() and c != self._active_color:
            self._active_color = c
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
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w = float(self.width())
        h = float(self.height())
        dpr = float(self.devicePixelRatioF()) if hasattr(self, "devicePixelRatioF") else 1.0
        enabled = self.isEnabled()
        lit = enabled and (self._hovered or self.isDown())

        if not enabled:
            st = "disabled"
            recol = True
        elif lit:
            st = "on"
            recol = not self._semantic_idle
        else:
            st = "semantic" if self._semantic_idle else "off"
            recol = not self._semantic_idle

        isz = max(16, int(min(w, h) - 1))
        pix = _render_svg_pixmap(
            self._svg_name, isz, isz, dpr=dpr, state=st,
            accent_hex=self._active_color.name(), recolor_primary=recol
        )
        p.drawPixmap(int((w - isz) * 0.5), int((h - isz) * 0.5), pix)


_DROPDOWN_SEP = object()


class SvgDropdownButton(QPushButton):
    """
    Selector desplegable gráfico con íconos SVG iluminados y menú Liquid Glass (HUDPopupMenu) sin caja exterior.
    Reemplaza los QComboBox nativos de Python/Qt manteniendo compatibilidad completa con la API
    de QComboBox (addItem, addItems, currentIndex, setCurrentIndex, currentText, currentData, findData, etc.).
    """

    currentIndexChanged = pyqtSignal(int)
    currentTextChanged = pyqtSignal(str)
    itemReselected = pyqtSignal(int)
    actionTriggered = pyqtSignal(object)

    def __init__(
        self,
        parent=None,
        icon_map: dict[str, str] | None = None,
        default_svg: str = "bell.svg",
        show_text: bool = False,
        recolor_to_accent: bool = True,
        accent_color: str = "#00E5FF",
        short_text_func=None,
    ):
        super().__init__(parent)
        self._items: list[tuple[str, object]] = []
        self._current_index: int = -1
        self._icon_map: dict[str, str] = dict(icon_map) if icon_map else {}
        self._item_accents: dict[str, str] = {}
        self._action_items: list[tuple[str, object, str, str]] = []
        self._default_svg = default_svg
        self._show_text = bool(show_text)
        self._recolor_to_accent = bool(recolor_to_accent)
        self._accent_color = QColor(accent_color)
        self._short_text_func = short_text_func
        self._hovered = False
        self._active_menu: HUDPopupMenu | None = None
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.clicked.connect(self._open_dropdown_menu)

    def setAccentColor(self, hex_color: str):
        c = QColor(hex_color)
        if c.isValid() and c != self._accent_color:
            self._accent_color = c
            self._cached_btn_pixmap = None
            self.update()

    def setActiveTintColor(self, hex_color: str):
        self.setAccentColor(hex_color)

    def set_icon_map(self, icon_map: dict[str, str]):
        if self._icon_map != icon_map:
            self._icon_map = dict(icon_map)
            self._cached_btn_pixmap = None
            self.update()

    def set_item_accent_map(self, accent_map: dict[str, str]):
        if self._item_accents != accent_map:
            self._item_accents = dict(accent_map)
            self._cached_btn_pixmap = None
            self.update()

    def is_menu_visible(self) -> bool:
        return self._active_menu is not None and self._active_menu.isVisible()

    def view(self):
        return self._active_menu

    def clear(self):
        self._items.clear()
        self._action_items.clear()
        self._item_accents.clear()
        self._current_index = -1
        self._cached_btn_pixmap = None
        self.setText("")
        if not self.signalsBlocked():
            self.update()

    def count(self) -> int:
        return len(self._items)

    def addSeparator(self):
        self._items.append(("", _DROPDOWN_SEP))

    def addActionItem(self, text: str, data, svg_name: str, accent_hex: str):
        self._action_items.append((str(text), data, str(svg_name), str(accent_hex)))

    def addItem(self, text: str, userData=None):
        self._items.append((str(text), userData))
        if self._current_index < 0 and userData is not _DROPDOWN_SEP:
            self._current_index = len(self._items) - 1
            self.setText(str(text))
        if not self.signalsBlocked():
            self.update()

    def addItems(self, texts):
        for t in texts:
            self.addItem(str(t), str(t))

    def currentIndex(self) -> int:
        return self._current_index

    def currentText(self) -> str:
        if 0 <= self._current_index < len(self._items) and self._items[self._current_index][1] is not _DROPDOWN_SEP:
            return self._items[self._current_index][0]
        return ""

    def currentData(self):
        if 0 <= self._current_index < len(self._items):
            txt, data = self._items[self._current_index]
            if data is _DROPDOWN_SEP:
                return None
            return data if data is not None else txt
        return None

    def itemText(self, index: int) -> str:
        if 0 <= index < len(self._items) and self._items[index][1] is not _DROPDOWN_SEP:
            return self._items[index][0]
        return ""

    def itemData(self, index: int):
        if 0 <= index < len(self._items):
            d = self._items[index][1]
            return None if d is _DROPDOWN_SEP else d
        return None

    def findData(self, data) -> int:
        for i, (_, d) in enumerate(self._items):
            if d is _DROPDOWN_SEP:
                continue
            if d == data:
                return i
        return -1

    def findText(self, text: str) -> int:
        t_str = str(text)
        for i, (t, d) in enumerate(self._items):
            if d is _DROPDOWN_SEP:
                continue
            if t == t_str:
                return i
        return -1

    def setCurrentIndex(self, index: int):
        if not (0 <= index < len(self._items)):
            return
        if self._items[index][1] is _DROPDOWN_SEP:
            return
        changed = (index != self._current_index)
        self._current_index = index
        txt = self._items[index][0]
        self.setText(txt)
        self.update()
        if changed and not self.signalsBlocked():
            self.currentIndexChanged.emit(index)
            self.currentTextChanged.emit(txt)

    def setCurrentText(self, text: str):
        idx = self.findText(text)
        if idx >= 0:
            self.setCurrentIndex(idx)

    def _svg_for_item(self, text: str, data) -> str:
        if data is not None and data is not _DROPDOWN_SEP and str(data) in self._icon_map:
            return self._icon_map[str(data)]
        if text in self._icon_map:
            return self._icon_map[text]
        return self._default_svg

    def _open_dropdown_menu(self):
        if not self.isEnabled() or (not self._items and not self._action_items):
            return
        menu = HUDPopupMenu(self, accent_color=self._accent_color.name(), corner_radius=8.0)
        self._active_menu = menu
        has_normal = False
        for i, (txt, data) in enumerate(self._items):
            if data is _DROPDOWN_SEP:
                menu.addSeparator()
                continue
            has_normal = True
            svg_file = self._svg_for_item(txt, data)
            item_acc = self._item_accents.get(
                str(data), self._item_accents.get(txt, self._accent_color.name())
            )
            recol = bool(self._recolor_to_accent)
            ic = _make_svg_qicon(
                svg_file, 20,
                accent_hex=item_acc,
                recolor_primary=(recol and (i == self._current_index or str(data) in self._item_accents)),
            )
            act = menu.addAction(ic, txt)
            act.setCheckable(True)
            act.setChecked(i == self._current_index)
            act.triggered.connect(lambda _=False, idx=i: self._on_menu_item_triggered(idx))
        if self._action_items:
            if has_normal:
                menu.addSeparator()
            for a_txt, a_data, a_svg, a_acc in self._action_items:
                ic_a = _make_svg_qicon(a_svg, 18, accent_hex=a_acc, recolor_primary=True)
                act_a = menu.addAction(ic_a, a_txt)
                act_a.triggered.connect(lambda _=False, ad=a_data: self.actionTriggered.emit(ad))
        pos = self.mapToGlobal(QPoint(0, self.height() + 2))
        menu.exec(pos)
        self.update()

    def _on_menu_item_triggered(self, index: int):
        if 0 <= index < len(self._items) and self._items[index][1] is not _DROPDOWN_SEP:
            prev = self._current_index
            self._current_index = index
            txt = self._items[index][0]
            self.setText(txt)
            self.update()
            if index != prev:
                self.currentIndexChanged.emit(index)
                self.currentTextChanged.emit(txt)
            else:
                self.itemReselected.emit(index)

    def enterEvent(self, event):
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event):
        w = int(self.width())
        h = int(self.height())
        if w <= 0 or h <= 0:
            return
        dpr = float(self.devicePixelRatioF()) if hasattr(self, "devicePixelRatioF") else 1.0
        enabled = self.isEnabled()
        menu_open = self.is_menu_visible()
        lit = enabled and (self._hovered or menu_open)

        txt = self.currentText()
        data = self.currentData()
        svg_file = self._svg_for_item(txt, data)
        disp_txt = (
            (self._short_text_func(txt, data) if self._short_text_func is not None else txt)
            if self._show_text
            else ""
        )
        if self._show_text and not disp_txt and not self._items:
            disp_txt = "—"

        cache_key = (
            w, h, round(dpr, 2), enabled, lit, self._current_index, disp_txt,
            self._accent_color.rgba(), self._recolor_to_accent, self._show_text, svg_file,
        )
        cached_pix = getattr(self, "_cached_btn_pixmap", None)
        if cached_pix is None or getattr(self, "_cached_btn_key", None) != cache_key:
            pw = max(1, int(round(w * dpr)))
            ph = max(1, int(round(h * dpr)))
            pix = QPixmap(pw, ph)
            pix.setDevicePixelRatio(dpr)
            pix.fill(Qt.GlobalColor.transparent)
            bp = QPainter(pix)
            bp.setRenderHint(QPainter.RenderHint.Antialiasing, True)

            if not enabled:
                st = "disabled"
            elif self._recolor_to_accent:
                st = "on" if lit else "hover"
            else:
                st = "on" if lit else "semantic"

            if self._show_text:
                isz = max(16, int(h - 4))
                icon_pix = _render_svg_pixmap(
                    svg_file, isz, isz, dpr=dpr, state=st,
                    accent_hex=self._accent_color.name(), recolor_primary=self._recolor_to_accent
                )
                bp.drawPixmap(2, int((h - isz) * 0.5), icon_pix)
                bp.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
                tc = _brighten_color(self._accent_color, 0.38) if (enabled and lit) else (QColor("#DCE4F0") if enabled else QColor("#5E6472"))
                bp.setPen(tc)
                bp.drawText(QRectF(float(isz + 5), 0.0, float(w) - float(isz + 6), float(h)), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, disp_txt)
            else:
                isz = max(18, int(min(w, h) - 1))
                icon_pix = _render_svg_pixmap(
                    svg_file, isz, isz, dpr=dpr, state=st,
                    accent_hex=self._accent_color.name(), recolor_primary=self._recolor_to_accent
                )
                bp.drawPixmap(int((w - isz) * 0.5), int((h - isz) * 0.5), icon_pix)
            bp.end()
            self._cached_btn_key = cache_key
            self._cached_btn_pixmap = pix
            cached_pix = pix

        p = QPainter(self)
        p.drawPixmap(0, 0, cached_pix)


class RangeDbSvgButton(QPushButton):
    """Botón SVG puro sin caja para alternar rango vertical ±12 dB / ±30 dB."""

    def __init__(self, range_db: float = 12.0, parent=None, active_tint_color: str = "#FFD54F"):
        super().__init__(parent)
        self._range_db = 30.0 if float(range_db) >= 20.0 else 12.0
        self._active_color = QColor(active_tint_color)
        self._hovered = False
        self.setFixedSize(44, 24)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)

    def setRangeDb(self, range_db: float):
        nr = 30.0 if float(range_db) >= 20.0 else 12.0
        if abs(nr - self._range_db) > 1e-3:
            self._range_db = nr
            self.update()

    def setText(self, text: str):
        super().setText(text)
        if "30" in str(text):
            self.setRangeDb(30.0)
        elif "12" in str(text):
            self.setRangeDb(12.0)

    def setActiveTintColor(self, hex_color: str):
        c = QColor(hex_color)
        if c.isValid() and c != self._active_color:
            self._active_color = c
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
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = float(self.width()), float(self.height())
        dpr = float(self.devicePixelRatioF()) if hasattr(self, "devicePixelRatioF") else 1.0
        svg_file = "range_30db.svg" if self._range_db >= 20.0 else "range_12db.svg"
        st = "on" if (self._hovered or self.isDown()) else "hover"
        isz = max(20, int(min(w, h)))
        pix = _render_svg_pixmap(
            svg_file, isz, isz, dpr=dpr, state=st,
            accent_hex=self._active_color.name(), recolor_primary=True
        )
        p.drawPixmap(int((w - isz) * 0.5), int((h - isz) * 0.5), pix)


class TimeWindowSvgButton(QPushButton):
    """Botón SVG puro sin caja para alternar la ventana de tiempo del Osciloscopio (1s / 5s / 10s)."""

    def __init__(self, duration_sec: int = 10, parent=None, active_tint_color: str = "#00E5FF"):
        super().__init__(parent)
        self._duration_sec = int(duration_sec) if int(duration_sec) in (1, 5, 10) else 10
        self._active_color = QColor(active_tint_color)
        self._hovered = False
        self.setFixedSize(26, 22)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)

    def set_duration(self, sec: int):
        s = int(sec)
        if s not in (1, 5, 10):
            s = 10
        if s != self._duration_sec:
            self._duration_sec = s
            self.update()

    def setText(self, text: str):
        super().setText(text)
        t = str(text).strip().lower().rstrip("s")
        try:
            self.set_duration(int(t))
        except Exception:
            pass

    def setActiveTintColor(self, hex_color: str):
        c = QColor(hex_color)
        if c.isValid() and c != self._active_color:
            self._active_color = c
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
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = float(self.width()), float(self.height())
        dpr = float(self.devicePixelRatioF()) if hasattr(self, "devicePixelRatioF") else 1.0
        svg_map = {1: "time_1s.svg", 5: "time_5s.svg", 10: "time_10s.svg"}
        svg_file = svg_map.get(self._duration_sec, "time_10s.svg")
        st = "on" if (self._hovered or self.isDown()) else "hover"
        isz = max(18, int(min(w, h)))
        pix = _render_svg_pixmap(
            svg_file, isz, isz, dpr=dpr, state=st,
            accent_hex=self._active_color.name(), recolor_primary=True
        )
        p.drawPixmap(int((w - isz) * 0.5), int((h - isz) * 0.5), pix)


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
        menu = HUDPopupMenu(self, accent_color=self._color.name(), corner_radius=8.0)
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
    Perilla de estudio con arco de valor e indicador circular integrados dentro del bisel:
    - Base estática cacheada en QPixmap (bisel exterior sutil, capuchón mate oscuro hundido,
      surco interior de 270° y marcas de límite mín/máx en los extremos del arco).
    - Título centrado en la cara interior del potenciómetro y lectura en vivo en texto curvo
      siguiendo el arco inferior de 90° donde la perilla no gira.
    - Arco de progreso luminoso + punto indicador brillante dibujados dinámicamente dentro del borde.
    - Soporte opcional de anillo de señal en tiempo real (meter_mode), marcador de Threshold,
      y arco de Reducción de Ganancia (gr_mode) con dirty-check para evitar repintados en reposo.
    """

    valueChanged = pyqtSignal(float)

    @staticmethod
    def _format_edge_limit(val: float, bipolar: bool = False) -> str:
        v = float(val)
        if abs(v) >= 1000.0:
            kv = v / 1000.0
            s = f"{kv:.1f}k" if abs(kv - round(kv)) >= 0.05 else f"{int(round(kv))}k"
            return f"+{s}" if (bipolar and v > 0) else s
        if abs(v - round(v)) < 1e-4:
            iv = int(round(v))
            return f"+{iv}" if (bipolar and iv > 0) else f"{iv}"
        s = f"{v:.1f}".rstrip("0").rstrip(".")
        return f"+{s}" if (bipolar and v > 0) else s

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
        reserve_outer_ring: bool = False,
        min_label: str | None = None,
        max_label: str | None = None,
        center_subtext: str = "",
        center_subtext_color: str = "#FFB74D",
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
        self._reserve_outer_ring = bool(reserve_outer_ring)

        self._min_label: str = min_label if min_label is not None else self._format_edge_limit(self.min_val, self.bipolar)
        self._max_label: str = max_label if max_label is not None else self._format_edge_limit(self.max_val, self.bipolar)
        self._center_subtext: str = str(center_subtext or "")
        self._center_subtext_color: str = str(center_subtext_color or "#FFB74D")
        self._reveal_alpha: float = 1.0

        self._dragging = False
        self._hovered = False
        self._last_mouse_y = 0.0

        # Caché estática de la base del potenciómetro (solo resize/enabled/tema/ring)
        self._static_cache_pix: QPixmap | None = None
        self._static_cache_key = None
        self._theme_version: int = 0
        self._static_cache_builds: int = 0

        # Medición de señal en tiempo real + marcador de Threshold
        self._meter_mode: bool = False
        self._meter_min_db: float = float(min_val) if (min_val < 0.0 and max_val <= 0.0) else -70.0
        self._meter_max_db: float = float(max_val) if (min_val < 0.0 and max_val <= 0.0) else 0.0
        self._meter_level_db: float = self._meter_min_db
        self._meter_level_norm: float = 0.0
        self._meter_uses_db: bool = False
        self._meter_dirty_eps_db: float = 0.3
        self._threshold_marker_enabled: bool = False
        self._threshold_marker_db: float = -24.0
        self._threshold_marker_norm: float = self._db_to_meter_norm(-24.0)

        # Modo opcional de Reducción de Ganancia (GR) para Limitador / Dinámica Multibanda
        self._gr_mode: bool = False
        self._gr_max_db: float = 24.0
        self._gr_db: float = 0.0
        self._gr_color: QColor | None = None

        w = max(76, self.diameter + 30)
        h = self.diameter + (42 if self._center_subtext else 32)
        self.setFixedSize(w, h)
        self.setCursor(Qt.CursorShape.SizeVerCursor)
        # TabFocus preserva navegación por teclado sin atrapar focus al hacer clic (seguro para Auto-Expand)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.setToolTip(
            f"{self.title}: Arrastrar verticalmente | Shift = Ajuste fino | Rueda = Paso suave | Doble clic = Reset"
        )

        # Etiquetas ocultas conservadas únicamente por compatibilidad interna
        self.val_label = QLabel(self)
        self.val_label.hide()
        self.title_label = QLabel(self.title, self)
        self.title_label.hide()

        self._refresh_labels()

    def setMinMaxLabels(self, min_lbl: str, max_lbl: str):
        mn = str(min_lbl)
        mx = str(max_lbl)
        if mn != self._min_label or mx != self._max_label:
            self._min_label = mn
            self._max_label = mx
            self._static_cache_pix = None
            self._full_cache_pix = None
            self.update()

    def setCenterSubtext(self, text: str, color: str | None = None):
        txt = str(text or "")
        col = str(color) if color is not None else self._center_subtext_color
        if txt != self._center_subtext or col != self._center_subtext_color:
            self._center_subtext = txt
            self._center_subtext_color = col
            self.update()

    def setRevealAlpha(self, alpha: float):
        a = float(np.clip(alpha, 0.0, 1.0))
        if abs(a - self._reveal_alpha) > 0.01:
            self._reveal_alpha = a
            self.update()

    def _refresh_labels(self):
        txt = self.fmt_func(self._value)
        if getattr(self, "_last_val_txt", None) != txt:
            self._last_val_txt = txt
            self.update()

    def invalidate_static_cache(self):
        self._theme_version += 1
        self._static_cache_pix = None
        self._full_cache_pix = None
        self.update()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.EnabledChange:
            self._refresh_labels()
            self.update()

    def enterEvent(self, event):
        if not self._hovered:
            self._hovered = True
            self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        if self._hovered:
            self._hovered = False
            self.update()
        super().leaveEvent(event)

    def focusInEvent(self, event):
        super().focusInEvent(event)
        self.update()

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        self.update()

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

    def setAccentColor(self, color_str: str):
        c = QColor(color_str)
        if c.isValid() and c != self.accent_color:
            self.accent_color = c
            self.update()

    def _db_to_meter_norm(self, db: float) -> float:
        span = max(1e-6, self._meter_max_db - self._meter_min_db)
        return float(np.clip((float(db) - self._meter_min_db) / span, 0.0, 1.0))

    def set_meter_range(self, min_db: float, max_db: float):
        mn = float(min_db)
        mx = float(max_db)
        if mx <= mn:
            return
        if abs(mn - self._meter_min_db) < 1e-6 and abs(mx - self._meter_max_db) < 1e-6:
            return
        self._meter_min_db = mn
        self._meter_max_db = mx
        self._meter_level_db = float(np.clip(self._meter_level_db, mn, mx))
        self._meter_level_norm = self._db_to_meter_norm(self._meter_level_db)
        self._threshold_marker_db = float(np.clip(self._threshold_marker_db, mn, mx))
        self._threshold_marker_norm = self._db_to_meter_norm(self._threshold_marker_db)
        if self._meter_mode:
            self.update()

    def set_meter_mode(self, enabled: bool, min_db: float | None = None, max_db: float | None = None):
        new_en = bool(enabled)
        range_changed = False
        if min_db is not None and max_db is not None:
            mn = float(min_db)
            mx = float(max_db)
            if mx > mn and (abs(mn - self._meter_min_db) >= 1e-6 or abs(mx - self._meter_max_db) >= 1e-6):
                self._meter_min_db = mn
                self._meter_max_db = mx
                self._meter_level_db = float(np.clip(self._meter_level_db, mn, mx))
                self._meter_level_norm = self._db_to_meter_norm(self._meter_level_db)
                self._threshold_marker_db = float(np.clip(self._threshold_marker_db, mn, mx))
                self._threshold_marker_norm = self._db_to_meter_norm(self._threshold_marker_db)
                range_changed = True
        if new_en != self._meter_mode or range_changed:
            self._meter_mode = new_en
            self.update()

    def set_meter_level(self, norm_or_db: float, is_normalized: bool | None = None):
        val = float(norm_or_db)
        if is_normalized is True:
            norm = float(np.clip(val, 0.0, 1.0))
            db = self._meter_min_db + norm * (self._meter_max_db - self._meter_min_db)
        elif is_normalized is False:
            self._meter_uses_db = True
            db = float(np.clip(val, self._meter_min_db, self._meter_max_db))
            norm = self._db_to_meter_norm(db)
        else:
            if val < 0.0 or val > 1.0 or self._meter_uses_db:
                if val < 0.0 or val > 1.0:
                    self._meter_uses_db = True
                db = float(np.clip(val, self._meter_min_db, self._meter_max_db))
                norm = self._db_to_meter_norm(db)
            else:
                norm = float(np.clip(val, 0.0, 1.0))
                db = self._meter_min_db + norm * (self._meter_max_db - self._meter_min_db)

        at_floor = db <= (self._meter_min_db + 0.05)
        was_at_floor = self._meter_level_db <= (self._meter_min_db + 0.05)
        if at_floor and was_at_floor:
            return
        if not (at_floor != was_at_floor) and abs(db - self._meter_level_db) < self._meter_dirty_eps_db:
            return

        self._meter_level_db = self._meter_min_db if at_floor else db
        self._meter_level_norm = 0.0 if at_floor else norm
        if self._meter_mode:
            self.update()

    def set_threshold_marker(self, norm_or_db: float, is_normalized: bool | None = None):
        val = float(norm_or_db)
        if is_normalized is True:
            norm = float(np.clip(val, 0.0, 1.0))
            db = self._meter_min_db + norm * (self._meter_max_db - self._meter_min_db)
        elif is_normalized is False or val <= 0.0 or val > 1.0:
            db = float(np.clip(val, self._meter_min_db, self._meter_max_db))
            norm = self._db_to_meter_norm(db)
        else:
            norm = float(np.clip(val, 0.0, 1.0))
            db = self._meter_min_db + norm * (self._meter_max_db - self._meter_min_db)

        if self._threshold_marker_enabled and abs(db - self._threshold_marker_db) < 0.05:
            return
        self._threshold_marker_enabled = True
        self._threshold_marker_db = db
        self._threshold_marker_norm = norm
        if self._meter_mode:
            self.update()

    def clear_threshold_marker(self):
        if self._threshold_marker_enabled:
            self._threshold_marker_enabled = False
            self.update()

    def set_gr_mode(self, enabled: bool, max_gr_db: float = 24.0, color: str | QColor | None = None):
        new_en = bool(enabled)
        new_max = max(1.0, float(max_gr_db))
        new_col = QColor(color) if color is not None else None
        if new_col is not None and not new_col.isValid():
            new_col = None
        if (
            new_en != self._gr_mode
            or abs(new_max - self._gr_max_db) >= 1e-6
            or new_col != self._gr_color
        ):
            self._gr_mode = new_en
            self._gr_max_db = new_max
            self._gr_color = new_col
            self.update()

    def set_gr_level(self, gr_db: float):
        val = float(gr_db)
        gr_neg = float(np.clip(val if val <= 0.0 else -val, -self._gr_max_db, 0.0))
        at_zero = gr_neg > -0.05
        was_at_zero = self._gr_db > -0.05
        if at_zero and was_at_zero:
            return
        if not (at_zero != was_at_zero) and abs(gr_neg - self._gr_db) < 0.25:
            return
        self._gr_db = 0.0 if at_zero else gr_neg
        if self._gr_mode:
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
            self.update()
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
            self.update()
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

    def _has_outer_ring(self) -> bool:
        return bool(self._reserve_outer_ring or self._meter_mode or self._gr_mode)

    def _knob_geometry(self):
        w = float(self.width())
        d = float(self.diameter)
        has_outer = self._has_outer_ring()
        cx = w * 0.5
        cy = d * 0.5 + 15.0
        if has_outer:
            meter_w = 3.4 if d >= 68.0 else 2.6
            r_meter = d * 0.5 - 2.0
            r_cap = max(8.0, r_meter - meter_w * 0.5 - 1.6)
            r_outer = r_cap
        else:
            meter_w = 2.4
            r_cap = max(8.0, d * 0.5 - 1.8)
            r_outer = r_cap
            r_meter = r_cap + 2.5
        arc_w = 2.8 if d >= 64.0 else 2.3
        dot_r = 2.1 if d >= 64.0 else 1.75
        r_arc = max(4.0, r_cap - (4.0 if d >= 64.0 else 3.4))
        return cx, cy, r_outer, r_cap, r_meter, meter_w, r_arc, arc_w, dot_r

    def _paint_knob_housing_to(
        self,
        painter: QPainter,
        cx: float,
        cy: float,
        r_outer: float,
        r_cap: float,
        r_arc: float,
        arc_w: float,
    ):
        enabled = self.isEnabled()

        # 1. Cara circular mate lisa estilo FabFilter Pro-C 2 (sin surcos ni sombras en min/max)
        cap_grad = QLinearGradient(cx - r_cap * 0.35, cy - r_cap, cx + r_cap * 0.35, cy + r_cap)
        if enabled:
            cap_grad.setColorAt(0.0, QColor("#2B2D34"))
            cap_grad.setColorAt(1.0, QColor("#1D1F25"))
        else:
            cap_grad.setColorAt(0.0, QColor("#1F2127"))
            cap_grad.setColorAt(1.0, QColor("#16171C"))

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(cap_grad))
        painter.drawEllipse(QPointF(cx, cy), r_cap, r_cap)

        # 2. Borde metálico fino únicamente en el barrido activo de 270° (225° a -45°, zona inferior libre)
        cap_rect = QRectF(cx - r_cap, cy - r_cap, 2.0 * r_cap, 2.0 * r_cap)
        rim_c = QColor(142, 150, 162, 185) if enabled else QColor(76, 80, 90, 95)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(rim_c, 1.25, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap))
        painter.drawArc(cap_rect, 225 * 16, -270 * 16)

        # 3. Etiquetas compactas de límite Mín/Máx en los extremos izquierdo (214°) y derecho (326°)
        w = float(self.width())
        lim_r = float(self.diameter) * 0.5 + 1.5
        lim_font = QFont("Segoe UI", 7, QFont.Weight.DemiBold)
        lim_font.setPointSizeF(6.5 if self.diameter < 72 else 7.0)
        painter.setFont(lim_font)
        painter.setPen(QColor("#828B9C") if enabled else QColor("#4A4F5B"))

        ang_l = math.radians(214.0)
        lx_edge = cx + lim_r * math.cos(ang_l)
        ly = cy - lim_r * math.sin(ang_l)
        if lx_edge > 6.0 and self._min_label:
            painter.drawText(
                QRectF(0.0, ly - 7.0, max(6.0, lx_edge - 1.0), 14.0),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                self._min_label,
            )

        ang_r = math.radians(326.0)
        rx_edge = cx + lim_r * math.cos(ang_r)
        ry = cy - lim_r * math.sin(ang_r)
        if w - rx_edge > 6.0 and self._max_label:
            painter.drawText(
                QRectF(rx_edge + 1.0, ry - 7.0, max(6.0, w - rx_edge - 1.0), 14.0),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                self._max_label,
            )

    def _draw_top_title(self, painter: QPainter, cx: float, cy: float):
        enabled = self.isEnabled()
        w = float(self.width())
        tit_c = QColor("#C6CEDB") if enabled else QColor("#5B616E")
        base_pt = 8.0 if self.diameter >= 72 else (7.4 if self.diameter >= 58 else 7.0)
        tit_font = QFont("Segoe UI", 8, QFont.Weight.Bold)
        tit_font.setPointSizeF(base_pt)
        safe_w = max(36.0, w - 4.0)
        fm = QFontMetricsF(tit_font)
        if fm.horizontalAdvance(self.title) > safe_w:
            tit_font.setPointSizeF(max(5.8, base_pt * (safe_w / max(1.0, fm.horizontalAdvance(self.title)))))
        painter.setFont(tit_font)
        painter.setPen(tit_c)
        painter.drawText(
            QRectF(0.0, 0.0, w, 14.0),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            self.title,
        )

    def _draw_curved_bottom_value(
        self,
        painter: QPainter,
        cx: float,
        cy: float,
        r_text: float,
        val_txt: str,
        enabled: bool,
        text_color: QColor | None = None,
        font_scale: float = 1.0,
    ):
        if not val_txt:
            return
        max_r = max(8.0, float(self.height()) - cy - 5.2)
        r_t = min(float(r_text), max_r)
        base_pt = (8.4 if self.diameter >= 80 else (7.6 if self.diameter >= 58 else 7.2)) * float(font_scale)
        max_arc_deg = 84.0
        letter_gap = 0.35

        font = QFont("Segoe UI", 8, QFont.Weight.Bold)
        font.setPointSizeF(base_pt)
        fm = QFontMetricsF(font)
        advs = [fm.horizontalAdvance(ch) + letter_gap for ch in val_txt]
        total_px = sum(advs) - letter_gap
        total_deg = math.degrees(total_px / max(1.0, r_t))
        if total_deg > max_arc_deg:
            scale = max_arc_deg / total_deg
            font.setPointSizeF(max(5.4, base_pt * scale))
            fm = QFontMetricsF(font)
            advs = [fm.horizontalAdvance(ch) + letter_gap for ch in val_txt]
            total_px = sum(advs) - letter_gap
            total_deg = math.degrees(total_px / max(1.0, r_t))

        painter.setFont(font)
        if not enabled:
            pen_c = QColor("#686C75")
        elif text_color is not None and text_color.isValid():
            pen_c = text_color
        else:
            pen_c = QColor("#ECEFF4")
        painter.setPen(pen_c)
        ch_h = fm.height() + 2.0

        theta = 270.0 - total_deg * 0.5
        for ch, adv in zip(val_txt, advs):
            char_w = adv - letter_gap
            da_char = math.degrees(char_w / max(1.0, r_t))
            da_step = math.degrees(adv / max(1.0, r_t))
            mid_theta = theta + da_char * 0.5
            rad = math.radians(mid_theta)
            px = cx + r_t * math.cos(rad)
            py = cy - r_t * math.sin(rad)

            painter.save()
            painter.translate(px, py)
            painter.rotate(270.0 - mid_theta)
            box_w = max(char_w + 4.0, 8.0)
            painter.drawText(
                QRectF(-box_w * 0.5, -ch_h * 0.5, box_w, ch_h),
                Qt.AlignmentFlag.AlignCenter,
                ch,
            )
            painter.restore()
            theta += da_step

    def _paint_static_base_to(self, painter: QPainter):
        cx, cy, r_outer, r_cap, _, _, r_arc, arc_w, _ = self._knob_geometry()
        self._paint_knob_housing_to(painter, cx, cy, r_outer, r_cap, r_arc, arc_w)

    def _draw_internal_arc_and_dot(
        self,
        painter: QPainter,
        cx: float,
        cy: float,
        r_arc: float,
        arc_w: float,
        dot_r: float,
        norm: float,
        active_color: QColor,
    ):
        enabled = self.isEnabled()
        arc_rect = QRectF(cx - r_arc, cy - r_arc, 2.0 * r_arc, 2.0 * r_arc)
        angle_deg = 225.0 - norm * 270.0

        if self.bipolar:
            zero_norm = self._val_to_norm(0.0)
            start_deg = 225.0 - zero_norm * 270.0
            full_span_deg = angle_deg - start_deg
        else:
            start_deg = 225.0
            full_span_deg = -norm * 270.0

        # Recortar el arco justo antes del punto indicador (sin pista ni sombra en valores no barridos)
        trim_deg = math.degrees((dot_r + arc_w * 0.35 + 1.5) / max(1.0, r_arc))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        if abs(full_span_deg) > trim_deg:
            arc_span_deg = (1.0 if full_span_deg > 0.0 else -1.0) * (abs(full_span_deg) - trim_deg)
            start_16 = int(round(start_deg * 16.0))
            span_16 = int(round(arc_span_deg * 16.0))
            if span_16 != 0:
                if enabled:
                    glow_mid = QColor(active_color)
                    glow_mid.setAlpha(65)
                    painter.setPen(QPen(glow_mid, arc_w + 1.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap))
                    painter.drawArc(arc_rect, start_16, span_16)

                    core_c = _brighten_color(active_color, 0.18)
                    core_c.setAlpha(248)
                    painter.setPen(QPen(core_c, arc_w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap))
                    painter.drawArc(arc_rect, start_16, span_16)
                else:
                    muted_c = QColor("#4A4D57")
                    muted_c.setAlpha(175)
                    painter.setPen(QPen(muted_c, arc_w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap))
                    painter.drawArc(arc_rect, start_16, span_16)

        # Punto indicador circular nítido en la punta activa
        angle_rad = math.radians(angle_deg)
        dot_x = cx + r_arc * math.cos(angle_rad)
        dot_y = cy - r_arc * math.sin(angle_rad)
        if enabled:
            halo_r = dot_r * 1.9
            halo_grad = QRadialGradient(dot_x, dot_y, halo_r)
            c0 = QColor(active_color)
            c0.setAlpha(115)
            c1 = QColor(active_color)
            c1.setAlpha(0)
            halo_grad.setColorAt(0.0, c0)
            halo_grad.setColorAt(1.0, c1)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(halo_grad))
            painter.drawEllipse(QPointF(dot_x, dot_y), halo_r, halo_r)

            dot_c = _brighten_color(active_color, 0.32)
            painter.setBrush(QBrush(dot_c))
            painter.drawEllipse(QPointF(dot_x, dot_y), dot_r, dot_r)
        else:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor("#686C75")))
            painter.drawEllipse(QPointF(dot_x, dot_y), dot_r * 0.9, dot_r * 0.9)

    def _draw_meter_and_gr_overlays(
        self,
        painter: QPainter,
        cx: float,
        cy: float,
        r_cap: float,
        r_meter: float,
        meter_w: float,
        active_color: QColor,
    ):
        enabled = self.isEnabled()
        meter_rect = QRectF(cx - r_meter, cy - r_meter, 2.0 * r_meter, 2.0 * r_meter)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        # Un único arco exterior continuo (sin dividir ni chocar dos barras en el mismo anillo)
        if self._meter_mode:
            if enabled and self._meter_level_norm > 0.004:
                lvl_norm = self._meter_level_norm
                sig_c = _brighten_color(active_color, 0.22)
                sig_c.setAlpha(210)
                painter.setPen(QPen(sig_c, meter_w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap))
                painter.drawArc(meter_rect, 225 * 16, int(round(-lvl_norm * 270.0 * 16.0)))
        elif self._gr_mode and enabled and self._gr_db < -0.05:
            gr_norm = min(1.0, abs(self._gr_db) / max(1.0, self._gr_max_db))
            gr_col = self._gr_color if (self._gr_color is not None and self._gr_color.isValid()) else QColor("#D500F9")
            span_16 = int(round(gr_norm * 270.0 * 16.0))
            if span_16 != 0:
                gr_core = _brighten_color(gr_col, 0.18)
                gr_core.setAlpha(235)
                painter.setPen(QPen(gr_core, meter_w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap))
                painter.drawArc(meter_rect, -45 * 16, span_16)

    def _paint_dynamic_overlay(self, painter: QPainter):
        cx, cy, r_outer, r_cap, r_meter, meter_w, r_arc, arc_w, dot_r = self._knob_geometry()
        enabled = self.isEnabled()
        active_color = self.accent_color if enabled else QColor("#4A4D57")

        if enabled and (self._dragging or self._hovered or self.hasFocus()):
            ring_c = QColor(active_color)
            ring_c.setAlpha(170 if self._dragging else (140 if self.hasFocus() else 100))
            cap_rect = QRectF(cx - r_cap, cy - r_cap, 2.0 * r_cap, 2.0 * r_cap)
            painter.setPen(QPen(ring_c, 1.35, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawArc(cap_rect, 225 * 16, -270 * 16)

        self._draw_meter_and_gr_overlays(painter, cx, cy, r_cap, r_meter, meter_w, active_color)
        self._draw_internal_arc_and_dot(painter, cx, cy, r_arc, arc_w, dot_r, self._val_to_norm(self._value), active_color)

        self._draw_top_title(painter, cx, cy)
        r_text = (r_meter if self._has_outer_ring() else r_cap) + 6.4
        self._draw_curved_bottom_value(painter, cx, cy, r_text, self.fmt_func(self._value), enabled)
        if self._center_subtext:
            sub_c = QColor(self._center_subtext_color) if enabled else QColor("#686C75")
            self._draw_curved_bottom_value(
                painter, cx, cy, r_text + 10.5, self._center_subtext, enabled,
                text_color=sub_c, font_scale=0.86,
            )

    def paintEvent(self, event):
        w = int(self.width())
        h = int(self.height())
        if w <= 0 or h <= 0 or self._reveal_alpha <= 0.005:
            return
        dpr = float(self.devicePixelRatioF()) if hasattr(self, "devicePixelRatioF") else 1.0
        cache_key = (
            w, h, round(dpr, 2), self.isEnabled(), self.diameter,
            self._theme_version, self._has_outer_ring(),
            self._min_label, self._max_label,
        )
        if self._static_cache_key != cache_key or self._static_cache_pix is None:
            self._static_cache_key = cache_key
            pw = max(1, int(round(w * dpr)))
            ph = max(1, int(round(h * dpr)))
            pix = QPixmap(pw, ph)
            pix.setDevicePixelRatio(dpr)
            pix.fill(Qt.GlobalColor.transparent)
            bp = QPainter(pix)
            bp.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            self._paint_static_base_to(bp)
            bp.end()
            self._static_cache_pix = pix
            self._static_cache_builds += 1
            self._full_cache_pix = None

        val_txt = self.fmt_func(self._value)
        full_key = (
            cache_key,
            round(self._value, 5),
            val_txt,
            self._center_subtext,
            self._center_subtext_color,
            self.accent_color.rgba(),
            self._hovered,
            self._dragging,
            self.hasFocus(),
            self._meter_mode,
            round(self._meter_level_norm, 3) if self._meter_mode else 0.0,
            self._threshold_marker_enabled,
            round(self._threshold_marker_norm, 3) if self._threshold_marker_enabled else 0.0,
            self._gr_mode,
            round(self._gr_db, 2) if self._gr_mode else 0.0,
            self._gr_color.rgba() if self._gr_color is not None else 0,
        )
        full_pix = getattr(self, "_full_cache_pix", None)
        if full_pix is None or getattr(self, "_full_cache_key", None) != full_key:
            pw = max(1, int(round(w * dpr)))
            ph = max(1, int(round(h * dpr)))
            fpix = QPixmap(pw, ph)
            fpix.setDevicePixelRatio(dpr)
            fpix.fill(Qt.GlobalColor.transparent)
            fp = QPainter(fpix)
            fp.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            fp.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            fp.drawPixmap(0, 0, self._static_cache_pix)
            self._paint_dynamic_overlay(fp)
            fp.end()
            self._full_cache_key = full_key
            self._full_cache_pix = fpix
            full_pix = fpix

        p = QPainter(self)
        if self._reveal_alpha < 0.999:
            p.setOpacity(self._reveal_alpha)
        p.drawPixmap(0, 0, full_pix)


class GainDynProQKnob(ProQKnob):
    """
    Perilla grande de GAIN (diámetro 92px) encastrada en el notch superior del HUD:
    - Base estática cacheada en QPixmap (capuchón mate oscuro hundido, bisel sutil, marcas -24 / +24).
    - Título GAIN + lectura DYN centrados dentro del capuchón; ganancia actual en arco curvo inferior.
    - Anillo de señal en tiempo real + marcador de Threshold opcional en el borde interior.
    - Anillo Exterior de Ganancia Dinámica (Dynamic Range) con excursión en vivo y manija radial.
    """

    dynRangeChanged = pyqtSignal(float)

    def __init__(self, diameter: int = 84, parent=None):
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
            min_label="-24",
            max_label="+24",
            parent=parent,
        )
        w = max(122, self.diameter + 30)
        h = self.diameter + 52
        self.setFixedSize(w, h)
        self.setMouseTracking(True)

        self.dyn_label = QLabel("DYN: 0.0 dB", self)
        self.dyn_label.hide()
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
        if self._dyn_enabled:
            dyn_txt = f"DYN {self._dyn_range_db:+.1f} dB"
        else:
            dyn_txt = ""
        if getattr(self, "_last_dyn_txt", None) != dyn_txt:
            self._last_dyn_txt = dyn_txt
            self.dyn_label.setText(dyn_txt)
            self.update()

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

    def setDynEnabled(self, enabled: bool):
        if bool(enabled) != self._dyn_enabled:
            self._dyn_enabled = bool(enabled)
            self._refresh_labels()
            self.update()

    def setLiveDynOffset(self, offset_db: float):
        clamped = float(np.clip(offset_db, -30.0, 30.0))
        at_zero = abs(clamped) <= 0.04
        was_at_zero = abs(self._dyn_offset_db) <= 0.04
        if at_zero and was_at_zero:
            return
        if not (at_zero != was_at_zero) and abs(clamped - self._dyn_offset_db) <= 0.04:
            return
        self._dyn_offset_db = 0.0 if at_zero else clamped
        self.update()

    def _geometry_radii(self):
        w = float(self.width())
        d = float(self.diameter)
        cx = w * 0.5
        cy = d * 0.5 + 15.0
        outer_w = 3.8
        outer_r = (d - outer_w - 2.4) * 0.5
        r_outer = outer_r - outer_w * 0.5 - 1.6
        r_cap = r_outer
        r_meter = outer_r
        meter_w = 3.4
        r_arc = r_cap - 4.4
        arc_w = 3.0
        dot_r = 2.2
        return cx, cy, outer_r, outer_w, r_outer, r_cap, r_meter, meter_w, r_arc, arc_w, dot_r

    def _is_on_outer_handle(self, pos: QPointF) -> bool:
        cx, cy, outer_r, outer_w, _, _, _, _, _, _, _ = self._geometry_radii()
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
            self.update()
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
            self.update()
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

    def _paint_dyn_static_base_to(self, painter: QPainter):
        cx, cy, _, _, r_outer, r_cap, _, _, r_arc, arc_w, _ = self._geometry_radii()
        self._paint_knob_housing_to(painter, cx, cy, r_outer, r_cap, r_arc, arc_w)

    def _paint_dyn_dynamic_overlay(self, painter: QPainter):
        cx, cy, outer_r, outer_w, r_outer, r_cap, r_meter, meter_w, r_arc, arc_w, dot_r = self._geometry_radii()
        enabled = self.isEnabled()
        outer_rect = QRectF(cx - outer_r, cy - outer_r, 2.0 * outer_r, 2.0 * outer_r)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        norm_gain = self._val_to_norm(self._value)
        gain_deg = 225.0 - norm_gain * 270.0
        band_c = self.accent_color if enabled else QColor("#4A4D57")

        # 1. Arco del Límite de Rango Dinámico Configurado en el perímetro exterior (solo con DYN activo)
        dyn_span_deg = -(self._dyn_range_db / 30.0) * 135.0
        if self._dyn_enabled and abs(self._dyn_range_db) >= 0.05:
            lim_color = QColor(band_c)
            lim_color.setAlpha(175 if enabled else 80)
            painter.setPen(QPen(lim_color, outer_w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap))
            painter.drawArc(outer_rect, int(round(gain_deg * 16.0)), int(round(dyn_span_deg * 16.0)))

        # 2. Excursión Dinámica Instantánea en Vivo (disminución o aumento aplicado)
        if enabled and self._dyn_enabled and abs(self._dyn_offset_db) > 0.04:
            inst_span_deg = -(self._dyn_offset_db / 30.0) * 135.0
            bright_c = _brighten_color(band_c, 0.35)
            bright_c.setAlpha(235)
            painter.setPen(QPen(bright_c, outer_w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap))
            painter.drawArc(outer_rect, int(round(gain_deg * 16.0)), int(round(inst_span_deg * 16.0)))

        # 3. Manija Radial solo cuando Dynamic Range está activo o bajo hover/arrastre
        if self._dyn_enabled or self._hover_outer or self._dragging_outer:
            handle_deg = gain_deg + dyn_span_deg
            handle_rad = math.radians(handle_deg)
            hr_in = r_cap + 0.5
            hr_out = outer_r + outer_w * 0.5 + 1.5
            hx1 = cx + hr_in * math.cos(handle_rad)
            hy1 = cy - hr_in * math.sin(handle_rad)
            hx2 = cx + hr_out * math.cos(handle_rad)
            hy2 = cy - hr_out * math.sin(handle_rad)

            handle_color = (
                QColor("#FFFFFF")
                if (enabled and (self._hover_outer or self._dragging_outer))
                else (_brighten_color(band_c, 0.38) if enabled else QColor("#686C75"))
            )
            painter.setPen(QPen(QColor("#0B0C0F"), 3.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(QPointF(hx1, hy1), QPointF(hx2, hy2))
            painter.setPen(QPen(handle_color, 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(QPointF(hx1, hy1), QPointF(hx2, hy2))

        # 4. Realce sutil únicamente en el borde de 270° durante hover/drag/foco (sin sombra en min/max)
        if enabled and (self._dragging or self._hovered or self.hasFocus()):
            ring_c = QColor(band_c)
            ring_c.setAlpha(170 if self._dragging else 110)
            cap_rect = QRectF(cx - r_cap, cy - r_cap, 2.0 * r_cap, 2.0 * r_cap)
            painter.setPen(QPen(ring_c, 1.35, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawArc(cap_rect, 225 * 16, -270 * 16)

        # 5. Arco interno bipolar de GAIN + punto luminoso
        self._draw_internal_arc_and_dot(painter, cx, cy, r_arc, arc_w, dot_r, norm_gain, band_c)

        # 6. Título GAIN arriba en "0" (12 en punto) y lecturas curvas debajo: Gain, DYN y GR en vivo
        self._draw_top_title(painter, cx, cy)
        r_line1 = outer_r + 6.5
        self._draw_curved_bottom_value(painter, cx, cy, r_line1, self.fmt_func(self._value), enabled)
        if self._dyn_enabled:
            r_line2 = r_line1 + 10.8
            dyn_col = _brighten_color(band_c, 0.25) if enabled else QColor("#686C75")
            self._draw_curved_bottom_value(
                painter, cx, cy, r_line2, f"DYN {self._dyn_range_db:+.1f} dB", enabled,
                text_color=dyn_col, font_scale=0.86,
            )
            r_line3 = r_line2 + 10.8
            if abs(self._dyn_offset_db) >= 0.05:
                gr_txt = f"GR {self._dyn_offset_db:+.1f} dB"
                gr_col = QColor("#FFB74D") if self._dyn_offset_db < 0.0 else QColor("#69F0AE")
            else:
                gr_txt = "GR 0.0 dB"
                gr_col = QColor("#828B9C")
            if not enabled:
                gr_col = QColor("#686C75")
            self._draw_curved_bottom_value(
                painter, cx, cy, r_line3, gr_txt, enabled,
                text_color=gr_col, font_scale=0.84,
            )

    def paintEvent(self, event):
        w = int(self.width())
        h = int(self.height())
        if w <= 0 or h <= 0:
            return
        dpr = float(self.devicePixelRatioF()) if hasattr(self, "devicePixelRatioF") else 1.0
        cache_key = (
            w, h, round(dpr, 2), self.isEnabled(), self.diameter,
            self._theme_version, self._min_label, self._max_label,
        )
        if self._static_cache_key != cache_key or self._static_cache_pix is None:
            self._static_cache_key = cache_key
            pw = max(1, int(round(w * dpr)))
            ph = max(1, int(round(h * dpr)))
            pix = QPixmap(pw, ph)
            pix.setDevicePixelRatio(dpr)
            pix.fill(Qt.GlobalColor.transparent)
            bp = QPainter(pix)
            bp.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            self._paint_dyn_static_base_to(bp)
            bp.end()
            self._static_cache_pix = pix
            self._static_cache_builds += 1
            self._full_cache_pix = None

        full_key = (
            cache_key,
            round(self._value, 5),
            self.accent_color.rgba(),
            self._hovered,
            self._dragging,
            self.hasFocus(),
            self._dyn_enabled,
            round(self._dyn_range_db, 3),
            round(self._dyn_offset_db, 3),
            self._dragging_outer,
            self._hover_outer,
            self._meter_mode,
            round(self._meter_level_norm, 3) if self._meter_mode else 0.0,
            self._threshold_marker_enabled,
            round(self._threshold_marker_norm, 3) if self._threshold_marker_enabled else 0.0,
        )
        full_pix = getattr(self, "_full_cache_pix", None)
        if full_pix is None or getattr(self, "_full_cache_key", None) != full_key:
            pw = max(1, int(round(w * dpr)))
            ph = max(1, int(round(h * dpr)))
            fpix = QPixmap(pw, ph)
            fpix.setDevicePixelRatio(dpr)
            fpix.fill(Qt.GlobalColor.transparent)
            fp = QPainter(fpix)
            fp.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            fp.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            fp.drawPixmap(0, 0, self._static_cache_pix)
            self._paint_dyn_dynamic_overlay(fp)
            fp.end()
            self._full_cache_key = full_key
            self._full_cache_pix = fpix
            full_pix = fpix

        p = QPainter(self)
        p.drawPixmap(0, 0, full_pix)


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
        c = QColor(color_val)
        a = bool(active)
        if c != self._color or a != self._active:
            self._color = c
            self._active = a
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
        c = QColor(color_hex)
        if filter_type != self._filter_type or c != self._color:
            self._filter_type = filter_type
            self._color = c
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
        self._opacity_effect.setEnabled(False)
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
        # Si el fondo OpenGL está activo, el 100% de la transparencia y Gaussian Blur se mezcla en la GPU
        if getattr(self, "_bg_gl_item", None) is not None and self._bg_gl_item.isVisible():
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        w = float(self.width())
        h = float(self.height())
        full_path = self._build_callout_path(w)

        pw = getattr(self.overlay, "plot_widget", None)
        if pw is not None:
            scene_pt = pw.mapToScene(self.pos())
            sx = float(scene_pt.x())
            sy = float(scene_pt.y())
            now = time.perf_counter()
            cur_fid = HUDOpenGLBackgroundItem._scene_blur_frame_id
            geom_key = (int(round(sx)), int(round(sy)), int(round(w)), int(round(h)))
            if (
                getattr(self, "_blur_pixmap", None) is None
                or getattr(self, "_blur_frame_id", -1) != cur_fid
                or getattr(self, "_blur_geom_key", None) != geom_key
                or (now - HUDOpenGLBackgroundItem._scene_blur_last_time) >= 0.036
            ):
                pix = HUDOpenGLBackgroundItem.render_scene_blur_pixmap(
                    self.overlay, pw, sx, sy, w, h, QColor(self._band_color), corner_radius=8.0
                )
                if pix is not None:
                    self._blur_pixmap = pix
                    self._blur_frame_id = HUDOpenGLBackgroundItem._scene_blur_frame_id
                    self._blur_geom_key = geom_key
        blur_pix = getattr(self, "_blur_pixmap", None)

        if blur_pix is not None and not blur_pix.isNull():
            painter.save()
            painter.setClipPath(full_path)
            painter.drawPixmap(self.rect(), blur_pix)
            painter.fillRect(self.rect(), QColor(10, 12, 18, 96))
            tint = QColor(self._band_color)
            tint.setAlpha(12)
            painter.fillRect(self.rect(), tint)
            painter.restore()
        else:
            painter.fillPath(full_path, QBrush(QColor(10, 12, 18, 180)))
            tint = QColor(self._band_color)
            tint.setAlpha(14)
            painter.fillPath(full_path, QBrush(tint))

        draw_liquid_glass_border_glow(painter, full_path, QRectF(0.0, 0.0, w, h), QColor(self._band_color))

    _CALLOUT_PATH_CACHE: dict[float, QPainterPath] = {}

    @classmethod
    def _build_callout_path(cls, w: float) -> QPainterPath:
        wk = round(float(w), 1)
        cached = cls._CALLOUT_PATH_CACHE.get(wk)
        if cached is not None:
            return cached
        body_h = 75.0
        caret_h = 6.0
        cx = wk * 0.5
        body_rect = QRectF(1.0, 1.0, wk - 2.0, body_h)
        path = QPainterPath()
        path.addRoundedRect(body_rect, 8.0, 8.0)

        caret_path = QPainterPath()
        caret_path.moveTo(cx - 7.0, body_h)
        caret_path.lineTo(cx, body_h + caret_h)
        caret_path.lineTo(cx + 7.0, body_h)
        caret_path.closeSubpath()
        united = path.united(caret_path)
        cls._CALLOUT_PATH_CACHE[wk] = united
        return united

    def _sync_gl_bg(self):
        pw = getattr(self.overlay, "plot_widget", None)
        if pw is None:
            return
        if getattr(self, "_bg_gl_item", None) is None:
            self._bg_gl_item = HUDOpenGLBackgroundItem(overlay=self.overlay, plot_widget=pw)
            self._bg_gl_item.setZValue(26)
            pw.plotItem.vb.scene().addItem(self._bg_gl_item)
        if not self.isVisible() or self._current_opacity <= 0.02:
            self._bg_gl_item.hide()
            return
        scene_pt = pw.mapToScene(self.pos())
        w = float(self.width())
        h = float(self.height())
        c = QColor(self._band_color)
        self._bg_gl_item.setOpacity(self._current_opacity)
        self._bg_gl_item.set_custom_shape(scene_pt, w, h, self._build_callout_path(w), c)
        if not self._bg_gl_item.isVisible():
            self._bg_gl_item.show()

    def hideEvent(self, event):
        super().hideEvent(event)
        if getattr(self, "_bg_gl_item", None) is not None:
            self._bg_gl_item.hide()

    def update_callout(self):
        if self._building:
            return
        idx = self.overlay.selected_band_idx
        bands = list(self.overlay.eq_engine.bands)
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
        if freq >= 1000.0:
            f_txt = f"{freq / 1000.0:.2f} kHz"
        else:
            f_txt = f"{freq:.1f} Hz"
        if getattr(self, "_last_freq_txt", None) != f_txt:
            self._last_freq_txt = f_txt
            self.freq_label.setText(f_txt)

        # Ganancia
        g_txt = f"{gain:+.2f} dB" if supports_gain else "0.00 dB"
        if getattr(self, "_last_gain_txt", None) != g_txt:
            self._last_gain_txt = g_txt
            self.gain_label.setText(g_txt)

        # Q
        q_txt = f"Q: {q:.3f}"
        if getattr(self, "_last_q_txt", None) != q_txt:
            self._last_q_txt = q_txt
            self.q_label.setText(q_txt)

        # Silueta vectorial de la forma de filtro
        self.shape_icon.setShape(ftype, color)

        # Indicador Modo de Canal y Salida
        ch_short = {"LR": "LR", "L": "L", "R": "R", "Mid": "M", "Side": "S"}.get(cm, cm)
        to_short = {"A+B": "A+B", "A": "A", "B": "B"}.get(to_val, to_val)
        m_txt = f"{ch_short} | {to_short}"
        if getattr(self, "_last_mode_txt", None) != m_txt:
            self._last_mode_txt = m_txt
            self.mode_badge.setText(m_txt)

        # Botones vectoriales
        p_c = color if enabled else "#686C75"
        self.power_btn.setColor(p_c)
        self.solo_btn.setColor("#00E5FF" if is_iso else "#8E95A5", active=is_iso)
        self.close_btn.setColor("#8E95A5")
        self.menu_btn.setColor("#8E95A5")

        self.reposition()
        if not self.isVisible():
            self.show()
            self.raise_()

    def set_callout_opacity(self, op: float):
        self._current_opacity = float(np.clip(op, 0.0, 1.0))
        if hasattr(self, "_opacity_effect"):
            if self._current_opacity >= 0.999:
                if self._opacity_effect.isEnabled():
                    self._opacity_effect.setEnabled(False)
            else:
                if not self._opacity_effect.isEnabled():
                    self._opacity_effect.setEnabled(True)
                self._opacity_effect.setOpacity(self._current_opacity)
        self._sync_gl_bg()

    def show_callout(self):
        if hasattr(self.overlay, "_callout_away_since"):
            self.overlay._callout_away_since = None
        self.set_callout_opacity(1.0)
        self.update_callout()
        if not self.isVisible():
            self.show()
        self.raise_()
        self._sync_gl_bg()

    def reposition(self):
        idx = self.overlay.selected_band_idx
        bands = list(self.overlay.eq_engine.bands)
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
        min_y = 46
        hud_c = getattr(self.overlay, "hud_container", None)
        scope_ov = getattr(hud_c, "scope_overlay", None) if hud_c is not None else None
        if scope_ov is not None and scope_ov.isVisible():
            min_y = max(min_y, scope_ov.y() + scope_ov.height() + 6)
        if target_y < min_y:
            target_y = int(view_pt.y() + 18)

        if self.x() != clamped_x or self.y() != target_y:
            self.move(clamped_x, target_y)
        self._sync_gl_bg()


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


class GradientBandCurveItem(pg.GraphicsObject):
    """
    Ítem gráfico ligero compatible con OpenGL y software para cada banda EQ:
    - Dibuja el sombreado vertical QLinearGradient anclado en 0 dB (boosts y cuts simétricos)
      y el trazo suavizado con QPainterPath construido vectorialmente via pg.arrayToQPath.
    - Expone getData() para el rasterizador 1/8-scale del Gaussian Blur HUD.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._stroke_path = QPainterPath()
        self._fill_path = QPainterPath()
        self._bounds = QRectF()
        self._pen = QPen(Qt.PenStyle.NoPen)
        self._brush = QBrush(Qt.BrushStyle.NoBrush)

    def getData(self):
        return self._x, self._y

    def setPen(self, pen):
        if pen is None:
            self._pen = QPen(Qt.PenStyle.NoPen)
        else:
            p = QPen(pen)
            p.setCosmetic(True)
            self._pen = p
        self.update()

    def setBrush(self, brush):
        self._brush = QBrush(brush) if brush is not None else QBrush(Qt.BrushStyle.NoBrush)
        self.update()

    def setData(self, x, y, pen=None, brush=None):
        if x is None or y is None or len(x) < 2 or len(y) < 2:
            if self._x is not None or not self._stroke_path.isEmpty():
                self.prepareGeometryChange()
                self._x = None
                self._y = None
                self._stroke_path = QPainterPath()
                self._fill_path = QPainterPath()
                self._bounds = QRectF()
                self.update()
            return

        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        n = min(len(x_arr), len(y_arr))
        x_arr = x_arr[:n]
        y_arr = y_arr[:n]

        stroke_path = pg.arrayToQPath(x_arr, y_arr, connect="all", finiteCheck=False)
        x_poly = np.empty(n + 2, dtype=np.float64)
        x_poly[0] = x_arr[0]
        x_poly[1:-1] = x_arr
        x_poly[-1] = x_arr[-1]
        y_poly = np.empty(n + 2, dtype=np.float64)
        y_poly[0] = 0.0
        y_poly[1:-1] = y_arr
        y_poly[-1] = 0.0
        fill_path = pg.arrayToQPath(x_poly, y_poly, connect="all", finiteCheck=False)
        fill_path.closeSubpath()

        x_min = float(np.min(x_arr))
        x_max = float(np.max(x_arr))
        y_min = min(0.0, float(np.min(y_arr)))
        y_max = max(0.0, float(np.max(y_arr)))

        self.prepareGeometryChange()
        self._x = x_arr
        self._y = y_arr
        self._stroke_path = stroke_path
        self._fill_path = fill_path
        self._bounds = QRectF(x_min, y_min, max(1e-6, x_max - x_min), max(1e-6, y_max - y_min)).adjusted(-0.5, -2.0, 0.5, 2.0)
        if pen is not None:
            p = QPen(pen)
            p.setCosmetic(True)
            self._pen = p
        if brush is not None:
            self._brush = QBrush(brush)
        self.update()

    def boundingRect(self) -> QRectF:
        return self._bounds

    def paint(self, painter: QPainter, option, widget=None):
        if self._stroke_path.isEmpty():
            return
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self._brush.style() != Qt.BrushStyle.NoBrush and not self._fill_path.isEmpty():
            painter.fillPath(self._fill_path, self._brush)
        if self._pen.style() != Qt.PenStyle.NoPen:
            painter.setPen(self._pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(self._stroke_path)
        painter.restore()


class DynamicRangeArrowsItem(pg.GraphicsObject):
    """
    Indicadores vectoriales de ventana de excursión dinámica (▲ / ▼) sobre los nodos
    de Dynamic EQ para mostrar el límite máximo de rango y la excursión activa en vivo.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._entries: list[tuple] = []
        self._last_state = None
        self._bounds = QRectF()
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    def set_entries(self, entries: list[tuple], eq_range_db: float):
        rounded = tuple(
            (
                round(float(e[0]), 4),
                round(float(e[1]), 2),
                round(float(e[2]), 2),
                round(float(e[3]), 2),
                str(e[4]),
                bool(e[5]),
                round(float(e[6]), 2),
            )
            for e in entries
        )
        state_key = (rounded, round(float(eq_range_db), 2))
        if state_key == self._last_state:
            return
        self._last_state = state_key
        self._entries = list(entries)
        self.prepareGeometryChange()
        if not self._entries:
            self._bounds = QRectF()
        else:
            xs = [e[0] for e in self._entries]
            ys = [v for e in self._entries for v in (e[1], e[2], e[3])]
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            self._bounds = QRectF(x0 - 0.3, y0 - 2.5, max(0.6, (x1 - x0) + 0.6), max(5.0, (y1 - y0) + 5.0))
        self.update()

    def boundingRect(self) -> QRectF:
        return self._bounds

    def paint(self, painter: QPainter, option, widget=None):
        if not self._entries:
            return
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        tr = painter.transform()
        painter.resetTransform()

        for x_val, y_static, y_target, y_live, color_hex, is_selected, dyn_range_db in self._entries:
            if abs(dyn_range_db) < 0.05:
                continue
            c = QColor(color_hex)
            pt_s = tr.map(QPointF(x_val, y_static))
            pt_t = tr.map(QPointF(x_val, y_target))
            pt_l = tr.map(QPointF(x_val, y_live))

            stem_c = QColor(c.red(), c.green(), c.blue(), 145 if is_selected else 82)
            painter.setPen(QPen(stem_c, 1.5 if is_selected else 1.15, Qt.PenStyle.DotLine))
            painter.drawLine(pt_s, pt_t)

            # En coordenadas de pantalla: +dB es hacia arriba (dy < 0), -dB hacia abajo (dy > 0)
            dir_y = -1.0 if dyn_range_db > 0.0 else 1.0
            tip_y = pt_t.y()
            base_y = tip_y - dir_y * 6.0
            tri_target = QPolygonF([
                QPointF(pt_t.x(), tip_y),
                QPointF(pt_t.x() - 4.2, base_y),
                QPointF(pt_t.x() + 4.2, base_y),
            ])
            fill_c = QColor(c.red(), c.green(), c.blue(), 225 if is_selected else 150)
            painter.setPen(QPen(QColor(10, 12, 18, 200), 0.9))
            painter.setBrush(QBrush(fill_c))
            painter.drawPolygon(tri_target)

            if abs(y_live - y_static) > 0.15:
                live_c = _brighten_color(c, 0.35)
                live_c.setAlpha(240)
                ly_tip = pt_l.y()
                ly_base = ly_tip - dir_y * 5.0
                chev = QPolygonF([
                    QPointF(pt_l.x(), ly_tip),
                    QPointF(pt_l.x() - 3.8, ly_base),
                    QPointF(pt_l.x() + 3.8, ly_base),
                ])
                painter.setPen(QPen(QColor(8, 10, 14, 210), 0.9))
                painter.setBrush(QBrush(live_c))
                painter.drawPolygon(chev)

        painter.restore()


class MultibandOverlayItem(pg.GraphicsObject):
    """
    Capa interactiva Pro-MB en eq_vb:
    - Dibuja zonas de frecuencia con desvanecimiento horizontal suave hacia los límites X (eje X),
      curva LR4+AP2 exacta con sombreado vertical anclado en 0 dB que abraza píxel por píxel
      la curva comprimida/expandida, línea de Threshold, cinta de vista previa (Ghost Band),
      nodos centrales M1..M6 y badges de modo (COMP / EXP / LIM).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._entries: list[tuple] = []
        self._ghost_entry = None
        self._eq_range_db: float = 12.0
        self._vx_min: float = 1.0
        self._vx_max: float = 4.4
        self._visible: bool = True
        self._last_state = None
        self._bounds = QRectF()
        self._cache_pixmap: QPixmap | None = None
        self._cache_render_key = None
        self._badge_font = QFont("Segoe UI", 7, QFont.Weight.Bold)
        self._node_font = QFont("Segoe UI", 8, QFont.Weight.Bold)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    def set_state(
        self,
        entries: list[tuple],
        eq_range_db: float,
        vx_min: float,
        vx_max: float,
        visible: bool = True,
        ghost_entry=None,
    ):
        def _curve_sig(ct):
            if ct is None:
                return None
            _, ya = ct
            if ya is None or len(ya) == 0:
                return None
            idx_pk = int(np.argmax(np.abs(ya)))
            return (round(float(ya[idx_pk]), 2), len(ya))

        rounded = tuple(
            (
                int(e[0]),
                round(float(e[1]), 4),
                round(float(e[2]), 4),
                round(float(e[3]), 2),
                round(float(e[4]), 2),
                round(float(e[5]), 2),
                str(e[6]),
                str(e[7]),
                bool(e[8]),
                bool(e[9]),
                bool(e[10]),
                bool(e[11]),
                str(e[12]),
                str(e[13]),
                round(float(e[14]), 2),
                _curve_sig(e[15]) if len(e) > 15 else None,
                str(e[16]) if len(e) > 16 else "Off",
            )
            for e in entries
        )
        ghost_sig = (
            (
                round(float(ghost_entry[0]), 4),
                round(float(ghost_entry[1]), 4),
                round(float(ghost_entry[2]), 2),
                str(ghost_entry[3]),
                str(ghost_entry[4]),
            )
            if ghost_entry is not None
            else None
        )
        key = (rounded, ghost_sig, round(float(eq_range_db), 2), round(float(vx_min), 4), round(float(vx_max), 4), bool(visible))
        if key == self._last_state:
            return
        self._last_state = key
        self._entries = list(entries)
        self._ghost_entry = ghost_entry
        self._eq_range_db = float(eq_range_db)
        self._vx_min = float(vx_min)
        self._vx_max = float(vx_max)
        self._visible = bool(visible)
        self.setVisible(self._visible)
        self._cache_pixmap = None
        self._cache_render_key = None
        self.prepareGeometryChange()
        if not self._visible or (not self._entries and self._ghost_entry is None):
            self._bounds = QRectF()
        else:
            lim = max(120.0, self._eq_range_db * 4.0)
            self._bounds = QRectF(self._vx_min, -lim, max(1e-3, self._vx_max - self._vx_min), 2.0 * lim)
        self.update()

    def boundingRect(self) -> QRectF:
        return self._bounds

    def paint(self, painter: QPainter, option, widget=None):
        if not self._visible or (not self._entries and self._ghost_entry is None):
            return
        tr = painter.transform()
        dpr = float(widget.devicePixelRatioF()) if (widget is not None and hasattr(widget, "devicePixelRatioF")) else 1.0
        vw = int(widget.width()) if widget is not None else 0
        vh = int(widget.height()) if widget is not None else 0

        if vw > 10 and vh > 10:
            render_key = (
                self._last_state,
                vw,
                vh,
                round(dpr, 2),
                round(tr.m11(), 4),
                round(tr.m22(), 4),
                round(tr.dx(), 1),
                round(tr.dy(), 1),
            )
            if self._cache_pixmap is None or self._cache_render_key != render_key:
                pw = max(1, int(round(vw * dpr)))
                ph = max(1, int(round(vh * dpr)))
                pix = QPixmap(pw, ph)
                pix.setDevicePixelRatio(dpr)
                pix.fill(Qt.GlobalColor.transparent)
                pm_painter = QPainter(pix)
                self._render_overlay_contents(pm_painter, tr, dpr, vh)
                pm_painter.end()
                self._cache_pixmap = pix
                self._cache_render_key = render_key
            painter.save()
            painter.resetTransform()
            painter.drawPixmap(0, 0, self._cache_pixmap)
            painter.restore()
            return

        painter.save()
        painter.resetTransform()
        vp_h = int(painter.viewport().height()) if painter.viewport().height() > 10 else vh
        self._render_overlay_contents(painter, tr, dpr, vp_h)
        painter.restore()

    def _render_overlay_contents(self, painter: QPainter, tr, dpr: float, view_h: int = 0):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        eq_top_px = tr.map(QPointF(self._vx_min, self._eq_range_db)).y()
        eq_bot_px = tr.map(QPointF(self._vx_min, -self._eq_range_db)).y()
        zero_px = tr.map(QPointF(self._vx_min, 0.0)).y()
        canvas_top_px = 0.0
        canvas_bot_px = float(view_h) if view_h > 10 else max(eq_bot_px + 200.0, 900.0)
        zone_h = max(10.0, canvas_bot_px - canvas_top_px)
        clip_lim = max(60.0, self._eq_range_db * 3.0)

        for entry in self._entries:
            (
                idx, x_lo, x_hi, y_thr, y_out, y_gr_end,
                mode, color_hex, enabled, bypass, solo, is_sel,
                hover_part, drag_part, live_gr_db,
            ) = entry[:15]
            curve_tuple = entry[15] if len(entry) > 15 else None
            char_mode = str(entry[16]) if len(entry) > 16 else "Off"

            x0_px = tr.map(QPointF(x_lo, 0.0)).x()
            x1_px = tr.map(QPointF(x_hi, 0.0)).x()
            w_px = max(6.0, x1_px - x0_px)
            xm_px = 0.5 * (x0_px + x1_px)
            thr_px = tr.map(QPointF(0.0, y_thr)).y()
            out_px = tr.map(QPointF(0.0, y_out)).y()

            active = enabled and not bypass
            base_c = QColor(color_hex) if active else QColor("#555D6E")
            r, g, b = base_c.red(), base_c.green(), base_c.blue()

            # 1. Fondo sombreado continuo hasta los bordes superior e inferior del lienzo (Y infinito)
            zone_alpha = 28 if (is_sel or hover_part != "") else (16 if active else 7)
            zone_rect = QRectF(x0_px, canvas_top_px, w_px, zone_h)
            zone_grad = QLinearGradient(x0_px, 0.0, x1_px, 0.0)
            zone_grad.setColorAt(0.00, QColor(r, g, b, 0))
            zone_grad.setColorAt(0.16, QColor(r, g, b, zone_alpha))
            zone_grad.setColorAt(0.84, QColor(r, g, b, zone_alpha))
            zone_grad.setColorAt(1.00, QColor(r, g, b, 0))
            painter.fillRect(zone_rect, QBrush(zone_grad))

            # 2. Sombreado exacto siguiendo la curva LR4+AP2 (anclado a 0 dB, cero inconsistencia)
            if active and curve_tuple is not None:
                x_arr, y_db_arr = curve_tuple
                if x_arr is not None and y_db_arr is not None and len(x_arr) >= 2:
                    y_cl = np.clip(y_db_arr, -clip_lim, clip_lim)
                    if np.max(np.abs(y_cl)) > 0.03:
                        view_curve = pg.arrayToQPath(x_arr, y_cl, connect="all")
                        px_curve = tr.map(view_curve)
                        fill_view = QPainterPath()
                        fill_view.moveTo(float(x_arr[0]), 0.0)
                        fill_view.connectPath(view_curve)
                        fill_view.lineTo(float(x_arr[-1]), 0.0)
                        fill_view.closeSubpath()
                        px_fill = tr.map(fill_view)

                        br = px_curve.boundingRect()
                        y_min_px = br.top()
                        y_max_px = br.bottom()
                        has_pos = (y_min_px < zero_px - 0.5)
                        has_neg = (y_max_px > zero_px + 0.5)
                        if has_pos and has_neg:
                            s_grad = QLinearGradient(0.0, y_min_px, 0.0, y_max_px)
                            zt = float(np.clip((zero_px - y_min_px) / max(1e-3, y_max_px - y_min_px), 0.02, 0.98))
                            s_grad.setColorAt(0.0, QColor(r, g, b, 128))
                            s_grad.setColorAt(zt, QColor(r, g, b, 0))
                            s_grad.setColorAt(1.0, QColor(r, g, b, 128))
                        elif has_neg:
                            s_grad = QLinearGradient(0.0, zero_px, 0.0, y_max_px)
                            s_grad.setColorAt(0.0, QColor(r, g, b, 0))
                            s_grad.setColorAt(0.55, QColor(r, g, b, 68))
                            s_grad.setColorAt(1.0, QColor(r, g, b, 132))
                        else:
                            s_grad = QLinearGradient(0.0, zero_px, 0.0, y_min_px)
                            s_grad.setColorAt(0.0, QColor(r, g, b, 0))
                            s_grad.setColorAt(0.55, QColor(r, g, b, 68))
                            s_grad.setColorAt(1.0, QColor(r, g, b, 132))

                        painter.fillPath(px_fill, QBrush(s_grad))

                        edge_c = _brighten_color(base_c, 0.25)
                        stroke_grad = QLinearGradient(x0_px - 18.0, 0.0, x1_px + 18.0, 0.0)
                        c_out = QColor(edge_c.red(), edge_c.green(), edge_c.blue(), 0)
                        c_in = QColor(edge_c.red(), edge_c.green(), edge_c.blue(), 220 if is_sel else 175)
                        span_s = max(1e-3, (x1_px + 18.0) - (x0_px - 18.0))
                        t_l = min(0.45, 18.0 / span_s)
                        stroke_grad.setColorAt(0.0, c_out)
                        stroke_grad.setColorAt(t_l, c_in)
                        stroke_grad.setColorAt(1.0 - t_l, c_in)
                        stroke_grad.setColorAt(1.0, c_out)
                        painter.setBrush(Qt.BrushStyle.NoBrush)
                        painter.setPen(QPen(QBrush(stroke_grad), 1.7))
                        painter.drawPath(px_curve)

            # 3. Líneas divisorias verticales continuas hasta el infinito Y + grips al hacer hover
            for edge_name, ex in (("left_edge", x0_px), ("right_edge", x1_px)):
                edge_active = (drag_part == edge_name or hover_part == edge_name)
                e_alpha = 230 if edge_active else (130 if is_sel else 62)
                e_w = 2.0 if edge_active else (1.3 if is_sel else 1.0)
                e_col = _brighten_color(base_c, 0.35) if edge_active else QColor(r, g, b)
                e_grad = QLinearGradient(ex, canvas_top_px, ex, canvas_bot_px)
                e_grad.setColorAt(0.0, QColor(e_col.red(), e_col.green(), e_col.blue(), 0))
                e_grad.setColorAt(0.06, QColor(e_col.red(), e_col.green(), e_col.blue(), e_alpha))
                e_grad.setColorAt(0.94, QColor(e_col.red(), e_col.green(), e_col.blue(), e_alpha))
                e_grad.setColorAt(1.0, QColor(e_col.red(), e_col.green(), e_col.blue(), 0))
                painter.setPen(QPen(QBrush(e_grad), e_w))
                painter.drawLine(QPointF(ex, canvas_top_px), QPointF(ex, canvas_bot_px))

                if edge_active or is_sel:
                    grip_col = QColor(255, 255, 255, 210) if edge_active else QColor(r, g, b, 175)
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QBrush(grip_col))
                    painter.drawRoundedRect(QRectF(ex - 1.6, zero_px - 9.0, 3.2, 18.0), 1.5, 1.5)

            # 4. Línea horizontal de Threshold / Ceiling con extremos desvanecidos
            thr_active = (drag_part == "threshold" or hover_part == "threshold")
            thr_col = QColor("#FF6D00") if (mode == "Limit" and active) else (_brighten_color(base_c, 0.25) if thr_active else QColor(r, g, b))
            t_alpha = 240 if thr_active else (185 if is_sel else 125)
            t_grad = QLinearGradient(x0_px, thr_px, x1_px, thr_px)
            t_grad.setColorAt(0.0, QColor(thr_col.red(), thr_col.green(), thr_col.blue(), 0))
            t_grad.setColorAt(0.08, QColor(thr_col.red(), thr_col.green(), thr_col.blue(), t_alpha))
            t_grad.setColorAt(0.92, QColor(thr_col.red(), thr_col.green(), thr_col.blue(), t_alpha))
            t_grad.setColorAt(1.0, QColor(thr_col.red(), thr_col.green(), thr_col.blue(), 0))
            t_style = Qt.PenStyle.SolidLine if (mode == "Limit" and active) else Qt.PenStyle.DashLine
            painter.setPen(QPen(QBrush(t_grad), 1.9 if thr_active else 1.35, t_style))
            painter.drawLine(QPointF(x0_px, thr_px), QPointF(x1_px, thr_px))

            # Indicadores SVG puros renderizados en la escena OpenGL (Modo + Character + Solo, sin cajas)
            icons = [MB_MODE_SVGS.get(mode, "comp_mode.svg")]
            if char_mode != "Off":
                icons.append("character.svg")
            if solo:
                icons.append("isolate.svg")

            isz = 19
            gap = 3
            max_fit = max(0, int((w_px - 8.0 + gap) // (isz + gap)))
            icons = icons[:max_fit]
            if icons:
                total_w = len(icons) * isz + max(0, len(icons) - 1) * gap
                bx = x1_px - total_w - 5.0
                by = max(50.0, eq_top_px + 4.0)
                st_svg = "off" if not active else ("on" if (is_sel or hover_part != "" or solo) else "hover")
                for k_ic, svg_name in enumerate(icons):
                    pix = _render_svg_pixmap(
                        svg_name, isz, isz, dpr=dpr, state=st_svg,
                        accent_hex=color_hex, recolor_primary=True
                    )
                    painter.drawPixmap(QPointF(bx + k_ic * (isz + gap), by), pix)

            # 5. Nodo central interactivo (cuadrado redondeado M1..M6)
            node_active = (is_sel or hover_part == "node" or drag_part == "node")
            nsz = 21.0 if node_active else 18.0
            nrect = QRectF(xm_px - nsz * 0.5, out_px - nsz * 0.5, nsz, nsz)
            if is_sel:
                halo = QColor(r, g, b, 65)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(halo))
                painter.drawRoundedRect(nrect.adjusted(-3.5, -3.5, 3.5, 3.5), 7.5, 7.5)
            painter.setPen(QPen(QColor("#FFFFFF") if is_sel else QColor("#101218"), 2.0 if is_sel else 1.2))
            painter.setBrush(QBrush(base_c))
            painter.drawRoundedRect(nrect, 5.0, 5.0)
            painter.setFont(self._node_font)
            painter.setPen(QColor("#0B0D12"))
            painter.drawText(nrect, Qt.AlignmentFlag.AlignCenter, f"M{idx + 1}")

        # 6. Banda fantasma de vista previa siguiendo al cursor (Creación intuitiva Pro-MB)
        if self._ghost_entry is not None:
            gx_lo, gx_hi, gy_thr, _, gcol_hex = self._ghost_entry[:5]
            gx0_px = tr.map(QPointF(gx_lo, 0.0)).x()
            gx1_px = tr.map(QPointF(gx_hi, 0.0)).x()
            gw_px = max(8.0, gx1_px - gx0_px)
            gxm_px = 0.5 * (gx0_px + gx1_px)
            gthr_px = tr.map(QPointF(0.0, gy_thr)).y()
            gc = QColor(gcol_hex)
            gr, gg, gb = gc.red(), gc.green(), gc.blue()

            gzone_rect = QRectF(gx0_px, canvas_top_px, gw_px, zone_h)
            g_grad = QLinearGradient(gx0_px, 0.0, gx1_px, 0.0)
            g_grad.setColorAt(0.00, QColor(gr, gg, gb, 0))
            g_grad.setColorAt(0.18, QColor(gr, gg, gb, 34))
            g_grad.setColorAt(0.82, QColor(gr, gg, gb, 34))
            g_grad.setColorAt(1.00, QColor(gr, gg, gb, 0))
            painter.fillRect(gzone_rect, QBrush(g_grad))

            # Guías verticales punteadas continuas en Y
            ge_grad = QLinearGradient(0.0, canvas_top_px, 0.0, canvas_bot_px)
            ge_grad.setColorAt(0.0, QColor(gr, gg, gb, 0))
            ge_grad.setColorAt(0.08, QColor(gr, gg, gb, 145))
            ge_grad.setColorAt(0.92, QColor(gr, gg, gb, 145))
            ge_grad.setColorAt(1.0, QColor(gr, gg, gb, 0))
            painter.setPen(QPen(QBrush(ge_grad), 1.3, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(gx0_px, canvas_top_px), QPointF(gx0_px, canvas_bot_px))
            painter.drawLine(QPointF(gx1_px, canvas_top_px), QPointF(gx1_px, canvas_bot_px))

            # Línea punteada de Threshold sugerido
            painter.setPen(QPen(QColor(gr, gg, gb, 185), 1.4, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(gx0_px, gthr_px), QPointF(gx1_px, gthr_px))

            # Nodo fantasma "M+" en la línea de 0 dB
            gn_rect = QRectF(gxm_px - 9.0, zero_px - 9.0, 18.0, 18.0)
            painter.setPen(QPen(QColor(255, 255, 255, 190), 1.3))
            painter.setBrush(QBrush(QColor(gr, gg, gb, 170)))
            painter.drawRoundedRect(gn_rect, 5.0, 5.0)
            painter.setFont(self._node_font)
            painter.setPen(QColor("#0B0D12"))
            painter.drawText(gn_rect, Qt.AlignmentFlag.AlignCenter, "M+")


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
    Mini-inspector flotante dibujado DENTRO de la escena OpenGL de la FFT (zValue alto)
    siguiendo el diseño del HUD (fondo translúcido de baja opacidad + Gaussian Blur + borde coloreado):
    línea 1 = Hz + dBFS instantáneo de la FFT; línea 2 opcional = ±dB del filtro propuesto.
    """

    def __init__(self, overlay=None, plot_widget=None):
        super().__init__()
        self.setZValue(1000)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.overlay = overlay
        self.plot_widget = plot_widget
        self._line1 = ""
        self._line2 = ""
        self._line2_color = QColor("#00E5FF")
        self._font = QFont("Segoe UI", 9)
        self._font.setBold(True)
        self._w = 120.0
        self._h = 22.0
        self._blur_pixmap = None
        self._blur_frame_id = -1
        self._blur_geom_key = None
        self._last_blur_time = 0.0

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
        return QRectF(-8.0, -8.0, self._w + 16.0, self._h + 16.0)

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        r = QRectF(1.0, 1.0, self._w - 2.0, self._h - 2.0)
        clip = QPainterPath()
        clip.addRoundedRect(r, 7.0, 7.0)

        accent = QColor(self._line2_color) if (self._line2_color and self._line2_color.isValid()) else QColor("#00E5FF")

        if self.overlay is not None and self.plot_widget is not None:
            pos = self.pos()
            px = float(pos.x())
            py = float(pos.y())
            now = time.perf_counter()
            cur_fid = HUDOpenGLBackgroundItem._scene_blur_frame_id
            geom_key = (int(round(px)), int(round(py)), int(round(self._w)), int(round(self._h)))
            if (
                self._blur_pixmap is None
                or self._blur_frame_id != cur_fid
                or self._blur_geom_key != geom_key
                or (now - HUDOpenGLBackgroundItem._scene_blur_last_time) >= 0.036
            ):
                pix = HUDOpenGLBackgroundItem.render_scene_blur_pixmap(
                    self.overlay, self.plot_widget, px, py, float(self._w), float(self._h), accent, corner_radius=7.0
                )
                if pix is not None:
                    self._blur_pixmap = pix
                    self._blur_frame_id = HUDOpenGLBackgroundItem._scene_blur_frame_id
                    self._blur_geom_key = geom_key

        if self._blur_pixmap is not None and not self._blur_pixmap.isNull():
            painter.save()
            painter.setClipPath(clip)
            painter.drawPixmap(QRectF(0.0, 0.0, self._w, self._h).toRect(), self._blur_pixmap)
            painter.fillRect(r, QColor(10, 12, 18, 96))
            tint = QColor(accent)
            tint.setAlpha(14 if self._line2 else 12)
            painter.fillRect(r, tint)
            painter.restore()
        else:
            painter.fillPath(clip, QBrush(QColor(10, 12, 18, 180)))
            tint = QColor(accent)
            tint.setAlpha(16 if self._line2 else 14)
            painter.fillPath(clip, QBrush(tint))

        draw_liquid_glass_border_glow(painter, clip, QRectF(0.0, 0.0, self._w, self._h), accent)

        painter.setFont(self._font)
        from PyQt6.QtGui import QFontMetricsF
        lh = QFontMetricsF(self._font).height()
        painter.setPen(QColor("#F0F4F8"))
        painter.drawText(QRectF(9.0, 5.0, self._w - 18.0, lh), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self._line1)
        if self._line2:
            painter.setPen(_brighten_color(accent, 0.25))
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
    mb_band_selected = pyqtSignal(int)
    mb_band_modified = pyqtSignal()
    eq_range_changed = pyqtSignal(float)

    def __init__(self, plot_widget: pg.PlotWidget, eq_engine: ParametricEQEngine, parent=None):
        super().__init__(parent)
        self.hud_container = parent
        self.plot_widget = plot_widget
        self._viewport = plot_widget.viewport()
        self.plot_item = plot_widget.getPlotItem()
        self.eq_engine = eq_engine
        self.multiband_dyn: MultibandDynamicsProcessor | None = None
        self.canvas_mode: str = "BOTH"  # "EQ" | "MB" | "BOTH" | "NONE"
        self.selected_mb_idx: int = -1
        self._hovered_mb: tuple[int, str] = (-1, "")
        self._dragging_mb: tuple[int, str] = (-1, "")
        self._mb_drag_init = None
        self._alt_mb_solo_dragging = False
        self._suppress_eq_callout: bool = False
        self._mb_placement_armed: bool = False
        self._mb_anchor_freq: float | None = None
        self._mb_anchor_scene_x: float = 0.0
        self._mb_ghost_data = None
        self._mb_ghost_half_ratio: float = 1.5
        self._last_rendered_mb_ghost = None
        self._last_mb_rendered_version: int = -1
        self._was_mb_moving: bool = False

        self.is_x_log = True
        self.nyquist = eq_engine.sr / 2.0
        self.eq_range_db = 12.0  # Conmutable entre 12.0 dB y 30.0 dB
        self.target_eq_range_db = 12.0
        self._range_anim: QVariantAnimation | None = None

        self.selected_band_idx = 0 if eq_engine.bands else -1
        self.hovered_band_idx = -1
        self.dragging_band_idx = -1
        self._drag_start_q = 1.0
        self._drag_start_scene_y = 0.0
        self._last_drag_ui_time = 0.0

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
        self.fft_inspector = FFTInspectorItem(overlay=self, plot_widget=self.plot_widget)
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

    def _on_range_anim_step(self, val):
        self.eq_range_db = float(val)
        self.update_y_range(self._current_y_floor, self._current_y_ceil)
        self.eq_range_changed.emit(self.eq_range_db)

    def set_eq_range_db(self, range_db: float, animate: bool = False):
        new_r = 30.0 if float(range_db) >= 20.0 else 12.0
        self.target_eq_range_db = new_r
        if self._range_anim is not None and self._range_anim.state() == QVariantAnimation.State.Running:
            self._range_anim.stop()
        if not animate or abs(new_r - self.eq_range_db) <= 1e-3:
            if abs(new_r - self.eq_range_db) > 1e-3:
                self.eq_range_db = new_r
                self.update_y_range(self._current_y_floor, self._current_y_ceil)
                self.eq_range_changed.emit(self.eq_range_db)
            return
        if self._range_anim is None:
            self._range_anim = QVariantAnimation(self)
            self._range_anim.setDuration(240)
            self._range_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
            self._range_anim.valueChanged.connect(self._on_range_anim_step)
        self._range_anim.setStartValue(float(self.eq_range_db))
        self._range_anim.setEndValue(float(new_r))
        self._range_anim.start()

    def toggle_eq_range_db(self, animate: bool = True):
        target = 30.0 if self.target_eq_range_db <= 12.0 else 12.0
        self.set_eq_range_db(target, animate=animate)
        return target

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
        vb_rect = self.plot_item.vb.sceneBoundingRect()
        self.eq_vb.setGeometry(vb_rect)
        self.eq_vb.linkedViewChanged(self.plot_item.vb, self.eq_vb.XAxis)
        if hasattr(self, "node_callout"):
            self.node_callout.reposition()
        cur_sz = (int(round(vb_rect.width())), int(round(vb_rect.height())))
        if getattr(self, "_last_vb_size", None) != cur_sz:
            self._last_vb_size = cur_sz
            if hasattr(self, "zero_line"):
                self.update_eq_curves(force=True)

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

        self.band_fill_curves: list[GradientBandCurveItem] = []
        self.band_static_outlines: list[pg.PlotDataItem] = []
        self.band_dyn_curves: list[pg.PlotDataItem] = []
        self.band_dyn_fills: list[pg.FillBetweenItem] = []
        self._cached_static_master_key = None
        self._cached_static_master_a: np.ndarray | None = None
        self._cached_static_master_b: np.ndarray | None = None

        self.dyn_limit_curve = pg.PlotDataItem(
            pen=pg.mkPen(color=(0, 229, 255, 175), width=1.4, style=Qt.PenStyle.DotLine),
            connect='finite',
        )
        self.dyn_limit_curve.setZValue(18)
        self.eq_vb.addItem(self.dyn_limit_curve)

        self.dyn_arrows_item = DynamicRangeArrowsItem()
        self.dyn_arrows_item.setZValue(36)
        self.eq_vb.addItem(self.dyn_arrows_item)

        self.mb_overlay_item = MultibandOverlayItem()
        self.mb_overlay_item.setZValue(16)
        self.eq_vb.addItem(self.mb_overlay_item)

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
            or (self.canvas_mode in ("EQ", "BOTH") and self._find_band_near_scene_pos(scene_pos, max_dist_px=22.0) >= 0)
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
        active_fft_col = QColor("#00E5FF")
        for curve in self.fft_curves:
            try:
                if not curve.isVisible():
                    continue
                xd, yd = curve.getData()
                if xd is None or yd is None or len(xd) < 2:
                    continue
                val = float(np.interp(main_pt.x(), xd, yd))
                fft_txt = f"{val:.1f} dBFS"
                try:
                    pen_obj = curve.opts.get("pen")
                    if pen_obj is not None:
                        active_fft_col = QColor(pg.mkPen(pen_obj).color())
                except Exception:
                    pass
                break
            except Exception:
                continue
        line1 = f"{f_txt}  |  {fft_txt}"

        line2 = ""
        col = active_fft_col
        if self._mb_ghost_data is not None:
            _, _, _, gm_mode, gcol_hex, gf_lo, gf_hi, g_thr, _ = self._mb_ghost_data
            col = QColor(gcol_hex)
            fl_s = f"{gf_lo / 1000.0:.1f}k" if gf_lo >= 1000.0 else f"{int(round(gf_lo))}"
            fh_s = f"{gf_hi / 1000.0:.1f}k" if gf_hi >= 1000.0 else f"{int(round(gf_hi))}"
            line2 = f"+MB {gm_mode[:4].upper()}: {fl_s}–{fh_s} Hz | Thr {g_thr:.0f} dB"
        elif self._ghost_unlocked and self._ghost_fade > 0.05:
            g_db = float(np.clip(self._ghost_db, -24.0, 24.0))
            g_f = float(10.0 ** self._ghost_log_f) if self.is_x_log else float(self._ghost_log_f)
            if g_f < 45.0:
                g_type = "Low Cut" if abs(g_db) < 3.0 else "Low Shelf"
            elif g_f > 10000.0:
                g_type = "High Cut" if abs(g_db) < 3.0 else "High Shelf"
            else:
                g_type = "Notch" if g_db < -10.0 else "Bell"

            from eq_dsp import BAND_COLORS
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
            bands = list(self.eq_engine.bands)
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
        if self.dragging_band_idx >= 0 or self._dragging_mb[0] >= 0 or self.canvas_mode not in ("EQ", "BOTH"):
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
            if hit_idx >= 0 or self._hovered_mb[1] in ("node", "left_edge", "right_edge", "threshold"):
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

    def set_multiband_processor(self, mb_proc: MultibandDynamicsProcessor | None):
        self.multiband_dyn = mb_proc
        if mb_proc is not None and mb_proc.bands:
            if not (0 <= self.selected_mb_idx < len(mb_proc.bands)):
                self.selected_mb_idx = 0
        else:
            self.selected_mb_idx = -1
        self.update_eq_curves(force=True)

    def set_canvas_mode(self, mode: str):
        m = mode if mode in ("EQ", "MB", "BOTH", "NONE") else "BOTH"
        if m != self.canvas_mode:
            self.canvas_mode = m
            eq_vis = m in ("EQ", "BOTH")
            mb_vis = m in ("MB", "BOTH")
            if not eq_vis:
                self._suppress_eq_callout = True
                self._clear_ghost_curve()
                if hasattr(self, "node_callout") and self.node_callout.isVisible():
                    self.node_callout.hide()
            else:
                self._suppress_eq_callout = False
            if not mb_vis:
                self._mb_placement_armed = False
                self._mb_anchor_freq = None
                self._mb_ghost_data = None
            self.update_eq_curves(force=True)

    def start_mb_placement(self):
        """Arma la creación MB: al hacer clic en una coordenada X de la FFT se ancla y sigue al puntero."""
        mb = self.multiband_dyn
        if mb is None or len(mb.bands) >= MAX_MB_BANDS:
            return
        self._mb_placement_armed = True
        self._mb_anchor_freq = None
        self._mb_ghost_data = None
        self._clear_ghost_curve()
        self._suppress_eq_callout = True
        if hasattr(self, "node_callout") and self.node_callout.isVisible():
            self.node_callout.hide()
        self._viewport.setCursor(Qt.CursorShape.CrossCursor)
        self.update_eq_curves(force=True)

    def cancel_mb_placement(self):
        if self._mb_placement_armed or self._mb_anchor_freq is not None or self._mb_ghost_data is not None:
            self._mb_placement_armed = False
            self._mb_anchor_freq = None
            self._mb_ghost_data = None
            if hasattr(self, "fft_inspector") and self.fft_inspector.isVisible():
                self.fft_inspector.hide()
            self.update_eq_curves(force=True)

    def _compute_mb_ghost_at(self, scene_pos: QPointF, alt_held: bool = False):
        mb = self.multiband_dyn
        if mb is None:
            return None
        with mb.lock:
            existing = list(mb.bands)
        if len(existing) >= MAX_MB_BANDS:
            return None
        view_pt = self.eq_vb.mapSceneToView(scene_pos)
        fc = self.x_to_freq(view_pt.x())
        ref_f = float(self._mb_anchor_freq) if self._mb_anchor_freq is not None else fc
        max_f = min(22000.0, self.nyquist * 0.94)
        lo_lim = 15.0
        hi_lim = max_f
        for eb in existing:
            if eb.f_low <= ref_f <= eb.f_high:
                return None
            if eb.f_high < ref_f:
                lo_lim = max(lo_lim, eb.f_high)
            elif eb.f_low > ref_f:
                hi_lim = min(hi_lim, eb.f_low)
        if hi_lim < lo_lim * 1.15:
            return None

        if self._mb_anchor_freq is not None and abs(scene_pos.x() - self._mb_anchor_scene_x) >= 12.0:
            f_cur = float(np.clip(fc, lo_lim, hi_lim))
            f_lo = max(lo_lim, min(ref_f, f_cur))
            f_hi = min(hi_lim, max(ref_f, f_cur))
            if f_hi < f_lo * 1.15:
                if f_cur >= ref_f:
                    f_hi = min(hi_lim, f_lo * 1.15)
                    f_lo = max(lo_lim, f_hi / 1.15)
                else:
                    f_lo = max(lo_lim, f_hi / 1.15)
                    f_hi = min(hi_lim, f_lo * 1.15)
        else:
            hr = float(np.clip(self._mb_ghost_half_ratio, 1.12, 4.0))
            f_lo = max(lo_lim, min(ref_f / hr, hi_lim / 1.15))
            f_hi = min(hi_lim, max(ref_f * hr, f_lo * 1.15))
            if f_hi < f_lo * 1.15:
                f_lo = max(lo_lim, f_hi / 1.15)

        thr_db = float(np.clip(self._y_to_mb_threshold(view_pt.y()), -55.0, -4.0))
        y_thr = self._mb_y_threshold(thr_db)
        mb_mode = "Limit" if alt_held else "Compress"
        col_hex = MB_MODE_COLORS.get(mb_mode, "#00E5FF")
        return (self.freq_to_x(f_lo), self.freq_to_x(f_hi), y_thr, mb_mode, col_hex, f_lo, f_hi, thr_db, fc)

    def _commit_mb_ghost(self, ghost) -> bool:
        mb = self.multiband_dyn
        if ghost is None or mb is None:
            self.cancel_mb_placement()
            return False
        _, _, _, gm_mode, _, gf_lo, gf_hi, g_thr, _ = ghost
        created = mb.add_band(f_low=gf_lo, f_high=gf_hi, mode=gm_mode, threshold_db=g_thr)
        self._mb_placement_armed = False
        self._mb_anchor_freq = None
        self._mb_ghost_data = None
        if hasattr(self, "fft_inspector") and self.fft_inspector.isVisible():
            self.fft_inspector.hide()
        if created is None:
            self.update_eq_curves(force=True)
            return False
        try:
            new_idx = mb.bands.index(created)
        except ValueError:
            new_idx = len(mb.bands) - 1
        self._suppress_eq_callout = True
        if hasattr(self, "node_callout") and self.node_callout.isVisible():
            self.node_callout.hide()
        self.selected_mb_idx = new_idx
        self.mb_band_selected.emit(new_idx)
        self.mb_band_modified.emit()
        self.update_eq_curves(force=True)
        return True

    def _mb_y_threshold(self, threshold_db: float) -> float:
        lim = self._effective_eq_vb_limit()
        return float(np.clip((float(threshold_db) + 60.0) * (self.eq_range_db / 60.0), -lim, lim))

    def _y_to_mb_threshold(self, y_eq: float) -> float:
        return float(np.clip((float(y_eq) * (60.0 / max(1e-6, self.eq_range_db))) - 60.0, -70.0, 0.0))

    def _find_mb_target(self, scene_pos: QPointF) -> tuple[int, str]:
        mb = self.multiband_dyn
        if mb is None or self.canvas_mode not in ("MB", "BOTH"):
            return -1, ""
        bands = list(mb.bands)
        if not bands:
            return -1, ""

        # 1. Nodos centrales M1..M6
        best_idx = -1
        best_dist = 16.0
        for idx, b in enumerate(bands):
            xm = self.freq_to_x(math.sqrt(max(10.0, b.f_low) * max(10.0, b.f_high)))
            ym = float(np.clip(b.output_gain_db, -self.eq_range_db, self.eq_range_db))
            n_scene = self.eq_vb.mapViewToScene(QPointF(xm, ym))
            dist = float(np.hypot(n_scene.x() - scene_pos.x(), n_scene.y() - scene_pos.y()))
            if dist <= best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx >= 0:
            return best_idx, "node"

        # 2. Bordes izquierdo / derecho y línea horizontal de Threshold
        for idx, b in enumerate(bands):
            x0_sc = self.eq_vb.mapViewToScene(QPointF(self.freq_to_x(b.f_low), 0.0)).x()
            x1_sc = self.eq_vb.mapViewToScene(QPointF(self.freq_to_x(b.f_high), 0.0)).x()
            if abs(scene_pos.x() - x0_sc) <= 8.0:
                return idx, "left_edge"
            if abs(scene_pos.x() - x1_sc) <= 8.0:
                return idx, "right_edge"
            if x0_sc <= scene_pos.x() <= x1_sc:
                y_thr = self._mb_y_threshold(b.threshold_db)
                thr_sc_y = self.eq_vb.mapViewToScene(QPointF(0.0, y_thr)).y()
                if abs(scene_pos.y() - thr_sc_y) <= 8.0:
                    return idx, "threshold"

        # 3. Dentro de la zona sombreada de la banda
        for idx, b in enumerate(bands):
            x0_sc = self.eq_vb.mapViewToScene(QPointF(self.freq_to_x(b.f_low), 0.0)).x()
            x1_sc = self.eq_vb.mapViewToScene(QPointF(self.freq_to_x(b.f_high), 0.0)).x()
            if x0_sc <= scene_pos.x() <= x1_sc:
                return idx, "zone"
        return -1, ""

    def _create_mb_band_at(self, view_pt: QPointF, mode: str = "Compress") -> bool:
        mb = self.multiband_dyn
        if mb is None:
            return False
        fc = self.x_to_freq(view_pt.x())
        f_lo = fc / 1.5
        f_hi = fc * 1.5
        thr_db = float(np.clip(self._y_to_mb_threshold(view_pt.y()), -55.0, -4.0))
        created = mb.add_band(f_low=f_lo, f_high=f_hi, mode=mode, threshold_db=thr_db)
        if created is None:
            return False
        try:
            new_idx = mb.bands.index(created)
        except ValueError:
            new_idx = len(mb.bands) - 1
        self._suppress_eq_callout = True
        if hasattr(self, "node_callout") and self.node_callout.isVisible():
            self.node_callout.hide()
        self.selected_mb_idx = new_idx
        self.mb_band_selected.emit(new_idx)
        self.mb_band_modified.emit()
        self.update_eq_curves(force=True)
        return True

    def _find_band_near_scene_pos(self, scene_pos: QPointF, max_dist_px: float = 18.0) -> int:
        best_idx = -1
        best_dist = max_dist_px
        for idx, band in enumerate(list(self.eq_engine.bands)):
            vx = self.freq_to_x(band.freq)
            vy = self._band_node_y(band)
            node_scene = self.eq_vb.mapViewToScene(QPointF(vx, vy))
            dist = float(np.hypot(node_scene.x() - scene_pos.x(), node_scene.y() - scene_pos.y()))
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        return best_idx

    def eventFilter(self, obj, event):
        vp = getattr(self, "_viewport", None)
        if vp is None or obj is not vp:
            return super().eventFilter(obj, event)

        etype = event.type()
        mode = self.canvas_mode
        eq_vis = mode in ("EQ", "BOTH")
        mb_vis = mode in ("MB", "BOTH")
        mb = self.multiband_dyn

        if etype == QEvent.Type.MouseButtonDblClick:
            if event.button() == Qt.MouseButton.LeftButton:
                pos_pt = event.position().toPoint()
                scene_pos = self.plot_widget.mapToScene(pos_pt)
                if not self.plot_item.vb.sceneBoundingRect().contains(scene_pos):
                    return False

                mods = event.modifiers()
                shift_held = bool(mods & Qt.KeyboardModifier.ShiftModifier)
                alt_held = bool(mods & Qt.KeyboardModifier.AltModifier)

                if mode == "MB" and mb is not None:
                    mb_idx, mb_part = self._find_mb_target(scene_pos)
                    if mb_part == "node" and 0 <= mb_idx < len(mb.bands):
                        with mb.lock:
                            mb.bands[mb_idx].output_gain_db = 0.0
                            mb.mark_dirty()
                        self.selected_mb_idx = mb_idx
                        self.mb_band_selected.emit(mb_idx)
                        self.mb_band_modified.emit()
                        self.update_eq_curves(force=True)
                        return True
                    view_pt = self.eq_vb.mapSceneToView(scene_pos)
                    mb_mode = "Limit" if shift_held else "Compress"
                    self._create_mb_band_at(view_pt, mode=mb_mode)
                    return True

                hit_idx = self._find_band_near_scene_pos(scene_pos) if eq_vis else -1
                if mode == "BOTH" and shift_held and mb is not None and hit_idx < 0:
                    self._clear_ghost_curve()
                    view_pt = self.eq_vb.mapSceneToView(scene_pos)
                    mb_mode = "Limit" if alt_held else "Compress"
                    if self._create_mb_band_at(view_pt, mode=mb_mode):
                        return True

                if hit_idx >= 0:
                    with self.eq_engine.lock:
                        band = self.eq_engine.bands[hit_idx]
                        band.gain_db = 0.0
                        band.update_coefficients()
                        self.eq_engine.mark_dirty()
                    self._suppress_eq_callout = False
                    self.selected_band_idx = hit_idx
                    self.band_selected.emit(hit_idx)
                    self.band_modified.emit()
                    self.update_eq_curves(force=True)
                    return True

                if mode == "BOTH" and mb is not None:
                    mb_idx, mb_part = self._find_mb_target(scene_pos)
                    if mb_part == "node" and 0 <= mb_idx < len(mb.bands):
                        with mb.lock:
                            mb.bands[mb_idx].output_gain_db = 0.0
                            mb.mark_dirty()
                        self._suppress_eq_callout = True
                        if hasattr(self, "node_callout") and self.node_callout.isVisible():
                            self.node_callout.hide()
                        self.selected_mb_idx = mb_idx
                        self.mb_band_selected.emit(mb_idx)
                        self.mb_band_modified.emit()
                        self.update_eq_curves(force=True)
                        return True

                if eq_vis:
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
                    self._suppress_eq_callout = False
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

            if event.button() == Qt.MouseButton.RightButton and (
                self._mb_placement_armed or self._mb_anchor_freq is not None
            ):
                self.cancel_mb_placement()
                return True

            # Si ya hay un ancla activa (1er clic realizado), el 2do clic confirma y crea la banda MB
            if event.button() == Qt.MouseButton.LeftButton and self._mb_anchor_freq is not None:
                alt_h = bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
                ghost = self._compute_mb_ghost_at(scene_pos, alt_held=alt_h) or self._mb_ghost_data
                self._commit_mb_ghost(ghost)
                return True

            hit_idx = self._find_band_near_scene_pos(scene_pos, max_dist_px=16.0) if (
                eq_vis and not self._mb_placement_armed
            ) else -1
            if event.button() == Qt.MouseButton.LeftButton:
                if hit_idx >= 0:
                    self._suppress_eq_callout = False
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

                mb_idx, mb_part = (-1, "")
                if mb_vis and mb is not None and not self._mb_placement_armed:
                    mb_idx, mb_part = self._find_mb_target(scene_pos)
                    if mb_idx >= 0:
                        if hasattr(self, "node_callout") and self.node_callout.isVisible():
                            self.node_callout.hide()
                            self._callout_away_since = None
                        if mb_part in ("node", "left_edge", "right_edge", "threshold") or (
                            mode == "MB" and mb_part == "zone"
                        ):
                            self._suppress_eq_callout = True
                            self.selected_mb_idx = mb_idx
                            if mb_part != "zone":
                                self._dragging_mb = (mb_idx, mb_part)
                                view_pt = self.eq_vb.mapSceneToView(scene_pos)
                                b_mb = mb.bands[mb_idx]
                                self._mb_drag_init = (
                                    view_pt.x(), view_pt.y(),
                                    b_mb.f_low, b_mb.f_high,
                                    b_mb.output_gain_db, b_mb.range_db, b_mb.threshold_db,
                                )
                                if mb_part == "node" and (event.modifiers() & Qt.KeyboardModifier.AltModifier):
                                    self._alt_mb_solo_dragging = True
                                    mb.set_solo_band(mb_idx, True)
                            self.mb_band_selected.emit(mb_idx)
                            self.update_eq_curves(force=True)
                            return True

                # 1er clic en una coordenada X libre de la FFT: inicia la banda sugerida que sigue al puntero
                hud_in_mb = getattr(getattr(self.hud_container, "floating_hud", None), "_hud_mode", "eq") == "mb"
                shift_held = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
                can_start_mb_anchor = (
                    mb is not None
                    and len(mb.bands) < MAX_MB_BANDS
                    and not self._cursor_in_blocked_zone(pos_pt)
                    and (
                        self._mb_placement_armed
                        or (mode == "MB" and mb_idx < 0)
                        or (mode == "BOTH" and (shift_held or hud_in_mb) and hit_idx < 0 and mb_idx < 0)
                    )
                )
                if can_start_mb_anchor:
                    view_pt = self.eq_vb.mapSceneToView(scene_pos)
                    self._mb_anchor_freq = self.x_to_freq(view_pt.x())
                    self._mb_anchor_scene_x = float(scene_pos.x())
                    alt_h = bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
                    ghost = self._compute_mb_ghost_at(scene_pos, alt_held=alt_h)
                    if ghost is not None:
                        self._mb_ghost_data = ghost
                        self._proposed_band = None
                        self._clear_ghost_curve()
                        self._suppress_eq_callout = True
                        if hasattr(self, "node_callout") and self.node_callout.isVisible():
                            self.node_callout.hide()
                        self.update_eq_curves(force=True)
                        self._update_fft_inspector()
                        return True
                    else:
                        self._mb_anchor_freq = None
                        if self._mb_placement_armed:
                            self.cancel_mb_placement()
                            return True

                if hasattr(self, "node_callout") and self.node_callout.isVisible():
                    self.node_callout.hide()
                    self._callout_away_since = None

            elif event.button() == Qt.MouseButton.RightButton:
                if hit_idx >= 0:
                    self._suppress_eq_callout = False
                    self.selected_band_idx = hit_idx
                    self.band_selected.emit(hit_idx)
                    self._show_node_context_menu(hit_idx, event.globalPosition().toPoint())
                    return True
                if mb_vis and mb is not None:
                    mb_idx, mb_part = self._find_mb_target(scene_pos)
                    if mb_idx >= 0 and (mode == "MB" or mb_part != "zone"):
                        self._suppress_eq_callout = True
                        if hasattr(self, "node_callout") and self.node_callout.isVisible():
                            self.node_callout.hide()
                        self.selected_mb_idx = mb_idx
                        self.mb_band_selected.emit(mb_idx)
                        self._show_mb_context_menu(mb_idx, event.globalPosition().toPoint())
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

            if self._dragging_mb[0] >= 0 and mb is not None:
                self._clear_ghost_curve()
                mb_idx, mb_part = self._dragging_mb
                if 0 <= mb_idx < len(mb.bands) and self._mb_drag_init is not None:
                    view_pt = self.eq_vb.mapSceneToView(scene_pos)
                    x0_v, y0_v, lo0, hi0, out0, rng0, _ = self._mb_drag_init
                    if mb_part == "left_edge":
                        new_lo = self.x_to_freq(view_pt.x())
                        mb.set_band_edges(mb_idx, new_lo, hi0)
                    elif mb_part == "right_edge":
                        new_hi = self.x_to_freq(view_pt.x())
                        mb.set_band_edges(mb_idx, lo0, new_hi)
                    elif mb_part == "threshold":
                        with mb.lock:
                            mb.bands[mb_idx].threshold_db = self._y_to_mb_threshold(view_pt.y())
                            mb.mark_dirty()
                    elif mb_part == "node":
                        lo_lim, hi_lim = mb.get_band_edge_limits(mb_idx)
                        if self.is_x_log:
                            factor = 10.0 ** (view_pt.x() - x0_v)
                            new_lo = lo0 * factor
                            new_hi = hi0 * factor
                            span_ratio = hi0 / max(1e-6, lo0)
                            if new_lo < lo_lim:
                                new_lo = lo_lim
                                new_hi = min(hi_lim, new_lo * span_ratio)
                            if new_hi > hi_lim:
                                new_hi = hi_lim
                                new_lo = max(lo_lim, new_hi / span_ratio)
                        else:
                            dx_f = view_pt.x() - x0_v
                            new_lo = lo0 + dx_f
                            new_hi = hi0 + dx_f
                            span_f = hi0 - lo0
                            if new_lo < lo_lim:
                                new_lo = lo_lim
                                new_hi = min(hi_lim, new_lo + span_f)
                            if new_hi > hi_lim:
                                new_hi = hi_lim
                                new_lo = max(lo_lim, new_hi - span_f)
                        mb.set_band_edges(mb_idx, new_lo, new_hi)
                        with mb.lock:
                            b_m = mb.bands[mb_idx]
                            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                                b_m.range_db = float(np.clip(rng0 + (view_pt.y() - y0_v), -30.0, 12.0))
                            else:
                                b_m.output_gain_db = float(np.clip(view_pt.y(), -24.0, 24.0))
                            mb.mark_dirty()

                now = time.perf_counter()
                if (now - self._last_drag_ui_time) >= 0.011:
                    self._last_drag_ui_time = now
                    self.mb_band_modified.emit()
                    self.update_eq_curves(force=True)
                return True

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

                now = time.perf_counter()
                if (now - self._last_drag_ui_time) >= 0.011:
                    self._last_drag_ui_time = now
                    self.band_modified.emit()
                    self.update_eq_curves(force=True)
                return True
            else:
                hit_idx = self._find_band_near_scene_pos(scene_pos) if eq_vis else -1
                mb_hov = self._find_mb_target(scene_pos) if (mb_vis and hit_idx < 0) else (-1, "")
                if hit_idx != self.hovered_band_idx or mb_hov != self._hovered_mb:
                    self.hovered_band_idx = hit_idx
                    self._hovered_mb = mb_hov
                    if hit_idx >= 0 or mb_hov[1] == "node":
                        self._viewport.setCursor(Qt.CursorShape.SizeAllCursor)
                    elif mb_hov[1] in ("left_edge", "right_edge"):
                        self._viewport.setCursor(Qt.CursorShape.SizeHorCursor)
                    elif mb_hov[1] == "threshold":
                        self._viewport.setCursor(Qt.CursorShape.SizeVerCursor)
                    else:
                        self._viewport.setCursor(Qt.CursorShape.CrossCursor)
                    self.update_eq_curves(force=True)

                if hit_idx < 0 and eq_vis and mb_hov[1] not in ("node", "left_edge", "right_edge", "threshold"):
                    self._update_proposed_band_info(scene_pos)
                else:
                    self._proposed_band = None
                    self._clear_ghost_curve()

                alt_held = bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
                want_mb_ghost = (
                    self._mb_anchor_freq is not None
                    and not self._cursor_in_blocked_zone(pos_pt)
                )
                if want_mb_ghost:
                    self._mb_ghost_data = self._compute_mb_ghost_at(scene_pos, alt_held=alt_held)
                    if self._mb_ghost_data is not None:
                        self._proposed_band = None
                        self._clear_ghost_curve()
                else:
                    self._mb_ghost_data = None
                self.update_eq_curves(force=False)
                self._update_fft_inspector()

        elif etype == QEvent.Type.Leave:
            self._current_ghost_mouse_pos = QPoint(-9999, -9999)
            self._current_ghost_scene_pos = QPointF(-9999, -9999)
            self._clear_ghost_curve()
            had_mb_state = (self._hovered_mb != (-1, "")) or (self._mb_ghost_data is not None)
            self._hovered_mb = (-1, "")
            self._mb_anchor_freq = None
            self._mb_ghost_data = None
            if had_mb_state:
                self.update_eq_curves(force=True)
            if hasattr(self, "fft_inspector") and self.fft_inspector.isVisible():
                self.fft_inspector.hide()

        elif etype == QEvent.Type.MouseButtonRelease:
            if event.button() == Qt.MouseButton.LeftButton and self._mb_anchor_freq is not None:
                scene_pos = self.plot_widget.mapToScene(event.position().toPoint())
                if abs(scene_pos.x() - self._mb_anchor_scene_x) >= 18.0:
                    alt_h = bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
                    ghost = self._compute_mb_ghost_at(scene_pos, alt_held=alt_h) or self._mb_ghost_data
                    self._commit_mb_ghost(ghost)
                    return True
            if event.button() == Qt.MouseButton.LeftButton and self._dragging_mb[0] >= 0:
                self._dragging_mb = (-1, "")
                self._mb_drag_init = None
                if self._alt_mb_solo_dragging and mb is not None:
                    self._alt_mb_solo_dragging = False
                    mb.set_solo_band(None, False)
                self.mb_band_modified.emit()
                self.update_eq_curves(force=True)
                return True
            if event.button() == Qt.MouseButton.LeftButton and self.dragging_band_idx >= 0:
                self.dragging_band_idx = -1
                if self._alt_isolate_dragging:
                    self._alt_isolate_dragging = False
                    self.eq_engine.set_isolated_band(None)
                self.band_modified.emit()
                self.update_eq_curves(force=True)
                if hasattr(self, "node_callout") and self.node_callout.isVisible():
                    self.node_callout.update_callout()
                return True

        elif etype == QEvent.Type.Wheel:
            scene_pos = self.plot_widget.mapToScene(event.position().toPoint())
            if self.plot_item.vb.sceneBoundingRect().contains(scene_pos):
                delta = event.angleDelta().y()
                if delta != 0 and self._mb_ghost_data is not None:
                    scale = 1.12 if delta > 0 else (1.0 / 1.12)
                    self._mb_ghost_half_ratio = float(np.clip(self._mb_ghost_half_ratio * scale, 1.12, 4.0))
                    alt_h = bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
                    self._mb_ghost_data = self._compute_mb_ghost_at(scene_pos, alt_held=alt_h)
                    self.update_eq_curves(force=False)
                    self._update_fft_inspector()
                    return True
                if delta != 0 and mb is not None and mb_vis:
                    mb_idx, mb_part = self._find_mb_target(scene_pos)
                    if (mode == "MB" and mb_idx >= 0) or (
                        mode == "BOTH" and mb_part in ("node", "left_edge", "right_edge", "threshold")
                    ):
                        b_m = mb.bands[mb_idx]
                        fc = math.sqrt(max(10.0, b_m.f_low) * max(10.0, b_m.f_high))
                        half_r = math.sqrt(b_m.f_high / max(1.0, b_m.f_low))
                        scale = 1.12 if delta > 0 else (1.0 / 1.12)
                        new_half = max(1.08, half_r * scale)
                        mb.set_band_edges(mb_idx, fc / new_half, fc * new_half)
                        self._suppress_eq_callout = True
                        if hasattr(self, "node_callout") and self.node_callout.isVisible():
                            self.node_callout.hide()
                        self.selected_mb_idx = mb_idx
                        self.mb_band_selected.emit(mb_idx)
                        self.mb_band_modified.emit()
                        self.update_eq_curves(force=True)
                        return True

                if eq_vis:
                    target_idx = self._find_band_near_scene_pos(scene_pos, max_dist_px=28.0)
                    if target_idx < 0:
                        target_idx = self.selected_band_idx
                    if 0 <= target_idx < len(self.eq_engine.bands):
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

    def _show_mb_context_menu(self, mb_idx: int, global_pos):
        mb = self.multiband_dyn
        if mb is None:
            return
        with mb.lock:
            if not (0 <= mb_idx < len(mb.bands)):
                return
            b = mb.bands[mb_idx]

        menu = HUDPopupMenu(self.plot_widget, accent_color=b.color, corner_radius=9.0)
        hdr = menu.addAction(f"MB Band {mb_idx + 1} ({b.f_low:.0f}–{b.f_high:.0f} Hz)")
        hdr.setEnabled(False)
        menu.addSeparator()

        for m_opt in MB_MODES:
            act_m = menu.addAction(f"Modo: {m_opt}")
            act_m.setCheckable(True)
            act_m.setChecked(b.mode == m_opt)
            act_m.triggered.connect(lambda _=False, mv=m_opt: self._set_mb_mode(mb_idx, mv))

        menu.addSeparator()
        act_solo = menu.addAction("Solo Banda MB")
        act_solo.setCheckable(True)
        act_solo.setChecked(b.solo)
        act_solo.triggered.connect(lambda chk: self._set_mb_solo(mb_idx, chk))

        act_byp = menu.addAction("Bypass Banda MB")
        act_byp.setCheckable(True)
        act_byp.setChecked(b.bypass)
        act_byp.triggered.connect(lambda chk: self._set_mb_bypass(mb_idx, chk))

        bus_menu = menu.addMenu("Bus de Salida")
        for to_val in TARGET_OUTPUTS:
            act_to = bus_menu.addAction(to_val)
            act_to.setCheckable(True)
            act_to.setChecked(b.bus_target == to_val)
            act_to.triggered.connect(lambda _=False, tv=to_val: self._set_mb_bus(mb_idx, tv))

        color_menu = menu.addMenu("Color de Banda")
        for label, hex_val in STUDIO_BAND_COLORS:
            act_c = color_menu.addAction(_make_color_icon(hex_val, 12), label)
            act_c.triggered.connect(lambda _=False, hv=hex_val: self._set_mb_color(mb_idx, hv))
        color_menu.addSeparator()
        act_custom_c = color_menu.addAction("Personalizado...")
        act_custom_c.triggered.connect(lambda: self._pick_custom_mb_color(mb_idx))

        menu.addSeparator()
        act_del = menu.addAction("Eliminar Banda MB")
        act_del.triggered.connect(lambda: self._delete_mb_band(mb_idx))
        menu.exec(global_pos)

    def _set_mb_mode(self, idx: int, mode_val: str):
        mb = self.multiband_dyn
        if mb is None:
            return
        with mb.lock:
            if 0 <= idx < len(mb.bands):
                b = mb.bands[idx]
                b.mode = mode_val
                if not getattr(b, "custom_color", False):
                    b.color = MB_MODE_COLORS.get(mode_val, b.color)
                if mode_val == "Limit":
                    if b.attack_ms > 2.0:
                        b.attack_ms = 0.5
                mb.mark_dirty()
        self.mb_band_modified.emit()
        self.update_eq_curves(force=True)

    def _set_mb_color(self, idx: int, hex_color: str):
        mb = self.multiband_dyn
        if mb is None:
            return
        with mb.lock:
            if 0 <= idx < len(mb.bands):
                b = mb.bands[idx]
                b.color = str(hex_color)
                b.custom_color = True
                mb.mark_dirty()
        self.mb_band_modified.emit()
        self.update_eq_curves(force=True)

    def _pick_custom_mb_color(self, idx: int):
        mb = self.multiband_dyn
        if mb is None or not (0 <= idx < len(mb.bands)):
            return
        col = QColorDialog.getColor(QColor(mb.bands[idx].color), self.plot_widget, "Color de Banda Multibanda")
        if col.isValid():
            self._set_mb_color(idx, col.name())

    def _set_mb_solo(self, idx: int, checked: bool):
        mb = self.multiband_dyn
        if mb is None:
            return
        mb.set_solo_band(idx, checked)
        self.mb_band_modified.emit()
        self.update_eq_curves(force=True)

    def _set_mb_bypass(self, idx: int, checked: bool):
        mb = self.multiband_dyn
        if mb is None:
            return
        with mb.lock:
            if 0 <= idx < len(mb.bands):
                mb.bands[idx].bypass = bool(checked)
                mb.mark_dirty()
        self.mb_band_modified.emit()
        self.update_eq_curves(force=True)

    def _set_mb_bus(self, idx: int, bus_val: str):
        mb = self.multiband_dyn
        if mb is None:
            return
        with mb.lock:
            if 0 <= idx < len(mb.bands):
                mb.bands[idx].bus_target = bus_val
                mb.mark_dirty()
        self.mb_band_modified.emit()
        self.update_eq_curves(force=True)

    def _delete_mb_band(self, idx: int):
        mb = self.multiband_dyn
        if mb is None:
            return
        mb.remove_band(idx)
        if self.selected_mb_idx >= len(mb.bands):
            self.selected_mb_idx = len(mb.bands) - 1
        self.mb_band_selected.emit(self.selected_mb_idx)
        self.mb_band_modified.emit()
        self.update_eq_curves(force=True)

    def _show_node_context_menu(self, band_idx: int, global_pos):
        with self.eq_engine.lock:
            if not (0 <= band_idx < len(self.eq_engine.bands)):
                return
            band = self.eq_engine.bands[band_idx]

        menu = HUDPopupMenu(self.plot_widget, accent_color=band.color, corner_radius=9.0)

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

    def _make_zero_anchored_band_brush(self, color: QColor, y_vals: np.ndarray, peak_alpha: int) -> QBrush:
        if peak_alpha <= 0 or y_vals is None or y_vals.size == 0:
            return pg.mkBrush(0, 0, 0, 0)
        y_hi = min(float(np.max(y_vals)), float(self.eq_range_db))
        y_lo = max(float(np.min(y_vals)), -float(self.eq_range_db))
        r, g, b = color.red(), color.green(), color.blue()
        c_peak = QColor(r, g, b, peak_alpha)
        c_mid = QColor(r, g, b, int(peak_alpha * 0.38))
        c_zero = QColor(r, g, b, 0)

        if y_hi > 0.05 and y_lo >= -0.05:
            grad = QLinearGradient(0.0, y_hi, 0.0, 0.0)
            grad.setColorAt(0.0, c_peak)
            grad.setColorAt(0.55, c_mid)
            grad.setColorAt(1.0, c_zero)
            return QBrush(grad)
        elif y_lo < -0.05 and y_hi <= 0.05:
            grad = QLinearGradient(0.0, y_lo, 0.0, 0.0)
            grad.setColorAt(0.0, c_peak)
            grad.setColorAt(0.55, c_mid)
            grad.setColorAt(1.0, c_zero)
            return QBrush(grad)
        elif y_hi > 0.05 and y_lo < -0.05:
            span = max(1e-6, y_hi - y_lo)
            t0 = float(np.clip(y_hi / span, 0.02, 0.98))
            grad = QLinearGradient(0.0, y_hi, 0.0, y_lo)
            grad.setColorAt(0.0, c_peak)
            grad.setColorAt(t0 * 0.55, c_mid)
            grad.setColorAt(t0, c_zero)
            grad.setColorAt(t0 + (1.0 - t0) * 0.45, c_mid)
            grad.setColorAt(1.0, c_peak)
            return QBrush(grad)
        return pg.mkBrush(0, 0, 0, 0)

    def update_eq_curves(self, force: bool = False):
        """
        Actualiza las curvas estáticas, dinámicas (morphing en vivo sobre el sombreado degradado
        anclado en 0 dB), las zonas Pro-MB y la curva maestra dinámica usando solo deltas activos.
        """
        bands = list(self.eq_engine.bands)
        cur_version = self.eq_engine.static_version
        bypass = self.eq_engine.bypass
        out_gain = self.eq_engine.output_gain_db
        any_dyn_moving = any(
            b.enabled and b.is_dynamic_active and abs(b.live_dyn_delta_db) > 0.03
            for b in bands
        )

        mb = self.multiband_dyn
        mb_bands = list(mb.bands) if mb is not None else []
        mb_version = mb.static_version if mb is not None else 0
        mb_bypass = bool(mb.bypass) if mb is not None else False
        any_mb_moving = bool(
            mb is not None
            and not mb_bypass
            and any(b.enabled and not b.bypass and abs(b.live_gr_db) > 0.03 for b in mb_bands)
        )

        cur_mb_ghost = self._mb_ghost_data
        if (
            not force
            and cur_version == self._last_rendered_version
            and mb_version == self._last_mb_rendered_version
            and cur_mb_ghost == self._last_rendered_mb_ghost
            and not any_dyn_moving
            and not self._was_dyn_moving
            and not any_mb_moving
            and not self._was_mb_moving
        ):
            return False

        eq_vis = self.canvas_mode in ("EQ", "BOTH")
        mb_vis = self.canvas_mode in ("MB", "BOTH")

        self._last_rendered_version = cur_version
        self._last_mb_rendered_version = mb_version
        self._last_rendered_mb_ghost = cur_mb_ghost
        self._was_dyn_moving = bool(any_dyn_moving and eq_vis)
        self._was_mb_moving = bool(any_mb_moving and mb_vis)

        x = self.eval_x
        z1 = self.eval_z1
        z2 = self.eval_z2
        clip_lim = max(36.0, self._effective_eq_vb_limit() * 1.15)

        self.zero_line.setVisible(eq_vis or mb_vis)
        if getattr(self, "_zero_line_bounds", None) != (float(x[0]), float(x[-1])):
            self._zero_line_bounds = (float(x[0]), float(x[-1]))
            self.zero_line.setData([x[0], x[-1]], [0.0, 0.0])

        iso_idx = self.eq_engine._resolve_isolated_band_index()

        band_curves_a: dict[int, np.ndarray] = {}
        band_curves_b: dict[int, np.ndarray] = {}
        net_a = None
        net_b = None
        if mb is not None and not mb_bypass and mb._plan.any_active and (not eq_vis or iso_idx is None):
            band_curves_a, net_a = mb.get_band_response_curves_db(z1, z2, target_bus="A")
            band_curves_b, net_b = mb.get_band_response_curves_db(z1, z2, target_bus="B")

        if hasattr(self, "mb_overlay_item"):
            mb_visible = (mb is not None and mb_vis) or (self._mb_ghost_data is not None)
            mb_entries = []
            if mb is not None and mb_vis:
                hov_i, hov_p = self._hovered_mb
                drg_i, drg_p = self._dragging_mb
                for i_mb, b_mb in enumerate(mb_bands):
                    x_lo = self.freq_to_x(b_mb.f_low)
                    x_hi = self.freq_to_x(b_mb.f_high)
                    y_thr = self._mb_y_threshold(b_mb.threshold_db)
                    y_out = float(np.clip(b_mb.output_gain_db, -self.eq_range_db, self.eq_range_db))
                    gr_val = 0.0 if mb_bypass else float(b_mb.live_gr_db)
                    y_gr_end = float(np.clip(b_mb.output_gain_db + gr_val, -self.eq_range_db, self.eq_range_db))
                    y_curve_db = band_curves_b.get(i_mb) if b_mb.bus_target == "B" else band_curves_a.get(i_mb)
                    curve_tup = (x, y_curve_db) if y_curve_db is not None else None
                    mb_entries.append((
                        i_mb, x_lo, x_hi, y_thr, y_out, y_gr_end,
                        b_mb.mode, b_mb.color, b_mb.enabled, (b_mb.bypass or mb_bypass), b_mb.solo,
                        i_mb == self.selected_mb_idx,
                        hov_p if hov_i == i_mb else "",
                        drg_p if drg_i == i_mb else "",
                        gr_val,
                        curve_tup,
                        str(getattr(b_mb, "character_mode", "Off")),
                    ))
            self.mb_overlay_item.set_state(
                mb_entries,
                self.eq_range_db,
                float(x[0]),
                float(x[-1]),
                visible=mb_visible,
                ghost_entry=self._mb_ghost_data,
            )

        if not eq_vis:
            for st_c in self.band_fill_curves:
                st_c.setData([], [])
                st_c.setVisible(False)
            for st_out in self.band_static_outlines:
                st_out.setData([], [])
                st_out.setVisible(False)
            for dyn_c in self.band_dyn_curves:
                dyn_c.setData([], [])
                dyn_c.setVisible(False)
            for fill_b in self.band_dyn_fills:
                fill_b.setBrush(pg.mkBrush(0, 0, 0, 0))
            self.dyn_limit_curve.setData([], [])
            self.dyn_limit_curve.setVisible(False)
            if hasattr(self, "dyn_arrows_item"):
                self.dyn_arrows_item.set_entries([], self.eq_range_db)
                self.dyn_arrows_item.setVisible(False)
            self.static_master_curve.setData([], [])
            self.static_master_curve.setVisible(False)
            self.static_bus_b_curve.setData([], [])
            self.static_bus_b_curve.setVisible(False)
            self.nodes_scatter.setData([])
            self.nodes_scatter.setVisible(False)
            self.dyn_nodes_scatter.setData([])
            self.dyn_nodes_scatter.setVisible(False)
            for lbl in self.node_labels:
                if lbl.isVisible():
                    lbl.hide()
            self.floating_tooltip.hide()
            if hasattr(self, "node_callout") and self.node_callout.isVisible():
                self.node_callout.hide()
            if force:
                HUDOpenGLBackgroundItem.invalidate_scene_blur_cache(self, getattr(self, "plot_widget", None))
            return True

        if not self.nodes_scatter.isVisible():
            for st_c in self.band_fill_curves:
                st_c.setVisible(True)
            for st_out in self.band_static_outlines:
                st_out.setVisible(True)
            for dyn_c in self.band_dyn_curves:
                dyn_c.setVisible(True)
            self.dyn_limit_curve.setVisible(True)
            if hasattr(self, "dyn_arrows_item"):
                self.dyn_arrows_item.setVisible(True)
            self.static_master_curve.setVisible(True)
            self.static_bus_b_curve.setVisible(True)
            self.nodes_scatter.setVisible(True)
            self.dyn_nodes_scatter.setVisible(True)

        while len(self.band_fill_curves) < len(bands):
            st_item = GradientBandCurveItem()
            st_item.setZValue(10)
            self.eq_vb.addItem(st_item)
            self.band_fill_curves.append(st_item)

            st_outline = pg.PlotDataItem(connect='finite')
            st_outline.setZValue(13)
            self.eq_vb.addItem(st_outline)
            self.band_static_outlines.append(st_outline)

            dyn_item = pg.PlotDataItem(connect='finite')
            dyn_item.setZValue(14)
            self.eq_vb.addItem(dyn_item)
            self.band_dyn_curves.append(dyn_item)

            fill_between = pg.FillBetweenItem(st_outline, dyn_item, brush=pg.mkBrush(0, 0, 0, 0))
            fill_between.setZValue(11)
            self.eq_vb.addItem(fill_between)
            self.band_dyn_fills.append(fill_between)

        while len(self.band_fill_curves) > len(bands):
            fill_b = self.band_dyn_fills.pop()
            self.eq_vb.removeItem(fill_b)
            dyn_c = self.band_dyn_curves.pop()
            self.eq_vb.removeItem(dyn_c)
            st_out = self.band_static_outlines.pop()
            self.eq_vb.removeItem(st_out)
            st_c = self.band_fill_curves.pop()
            self.eq_vb.removeItem(st_c)

        arrow_entries: list[tuple] = []
        moving_dyn_bands: list[EQBand] = []

        for idx, b in enumerate(bands):
            st_curve = self.band_fill_curves[idx]
            st_outline = self.band_static_outlines[idx]
            dyn_curve = self.band_dyn_curves[idx]
            dyn_fill = self.band_dyn_fills[idx]

            is_sel = (idx == self.selected_band_idx)
            if not b.enabled and not is_sel:
                st_curve.setData([], [])
                st_outline.setData([], [])
                dyn_curve.setData([], [])
                dyn_fill.setBrush(pg.mkBrush(0, 0, 0, 0))
                continue

            resp_static = np.clip(b.get_static_freq_response_db(z1, z2), -clip_lim, clip_lim)
            c = QColor(b.color)

            is_moving_dyn = b.enabled and b.is_dynamic_active and abs(b.live_dyn_delta_db) > 0.03 and not bypass
            if is_moving_dyn:
                moving_dyn_bands.append(b)
                resp_live = np.clip(b.get_dynamic_freq_response_db(z1, z2), -clip_lim, clip_lim)
            else:
                resp_live = resp_static

            is_active = is_sel or abs(b.gain_db) > 0.05 or not b.supports_gain or b.is_dynamic_active
            if is_active:
                pen_alpha = 230 if (is_sel or is_moving_dyn) else (145 if b.enabled else 60)
                peak_alpha = 105 if is_sel else (78 if is_moving_dyn else (56 if b.enabled else 22))
                pen_w = 2.0 if (is_sel or is_moving_dyn) else 1.4
                pen_c = _brighten_color(c, 0.18) if is_moving_dyn else QColor(c)
                pen_c.setAlpha(pen_alpha)
                st_curve.setData(
                    x,
                    resp_live,
                    pen=pg.mkPen(color=pen_c, width=pen_w),
                    brush=self._make_zero_anchored_band_brush(c, resp_live, peak_alpha),
                )
            else:
                pen_alpha = 40 if b.enabled else 20
                st_curve.setData(
                    x,
                    resp_static,
                    pen=pg.mkPen(color=(c.red(), c.green(), c.blue(), pen_alpha), width=1.0),
                    brush=pg.mkBrush(0, 0, 0, 0),
                )

            if b.is_dynamic_active and (b.enabled or is_sel) and abs(b.dyn_range_db) >= 0.05:
                st_outline.setPen(pg.mkPen(color=(c.red(), c.green(), c.blue(), 145 if is_sel else 85), width=1.2))
                st_outline.setData(x, resp_static)

                resp_target = np.clip(b.get_max_dynamic_freq_response_db(z1, z2), -clip_lim, clip_lim)
                tgt_c = _brighten_color(c, 0.15)
                tgt_c.setAlpha(185 if is_sel else 82)
                dyn_curve.setPen(pg.mkPen(color=tgt_c, width=1.3, style=Qt.PenStyle.DotLine))
                dyn_curve.setData(x, resp_target)
                dyn_fill.setBrush(pg.mkBrush(c.red(), c.green(), c.blue(), 26 if is_sel else 12))

                nx_a = self.freq_to_x(b.freq)
                y_st = self._band_node_y(b)
                y_tg = (
                    float(np.clip(y_st + b.dyn_range_db, -self.eq_range_db, self.eq_range_db))
                    if b.supports_gain
                    else float(np.clip(b.dyn_range_db * 0.5, -self.eq_range_db, self.eq_range_db))
                )
                y_lv = float(
                    np.clip(
                        y_st + (b.live_dyn_delta_db if is_moving_dyn else 0.0),
                        -self.eq_range_db,
                        self.eq_range_db,
                    )
                )
                arrow_entries.append((nx_a, y_st, y_tg, y_lv, b.color, is_sel, b.dyn_range_db))
            else:
                st_outline.setData([], [])
                dyn_curve.setData([], [])
                dyn_fill.setBrush(pg.mkBrush(0, 0, 0, 0))

        self.dyn_limit_curve.setData([], [])
        if hasattr(self, "dyn_arrows_item"):
            self.dyn_arrows_item.set_entries(arrow_entries, self.eq_range_db)

        master_key = (cur_version, bypass, len(z1), iso_idx)
        if getattr(self, "_cached_static_master_key", None) != master_key or self._cached_static_master_a is None:
            static_a_db, _ = self.eq_engine.get_combined_responses_db(z1, z2, target_bus="A")
            static_b_db, _ = self.eq_engine.get_combined_responses_db(z1, z2, target_bus="B")
            if not bypass:
                static_a_db = static_a_db - out_gain
                static_b_db = static_b_db - out_gain
            self._cached_static_master_a = static_a_db
            self._cached_static_master_b = static_b_db
            self._cached_static_master_key = master_key

        master_a = self._cached_static_master_a.copy()
        master_b = self._cached_static_master_b.copy()

        if not bypass and iso_idx is None and moving_dyn_bands:
            for mb_b in moving_dyn_bands:
                delta_db = mb_b.get_dynamic_freq_response_db(z1, z2) - mb_b.get_static_freq_response_db(z1, z2)
                if mb_b.target_output in ("A", "A+B"):
                    master_a += delta_db
                if mb_b.target_output in ("B", "A+B"):
                    master_b += delta_db

        if net_a is not None and net_b is not None and iso_idx is None:
            master_a += net_a
            master_b += net_b

        static_clipped_a = np.clip(master_a, -clip_lim, clip_lim)
        static_clipped_b = np.clip(master_b, -clip_lim, clip_lim)

        # Curva Maestra Bus A (Amarilla): 100% continua de punta a punta, sin np.nan
        self.static_master_curve.setData(x, static_clipped_a)

        # Curva Maestra Bus B (Cian): continua de punta a punta si difiere de Bus A o si hay bandas asignadas a Bus B
        b_differs = np.any(np.abs(static_clipped_b - static_clipped_a) > 0.02)
        has_b_routed = any(getattr(b, 'target_output', 'A') in ('B', 'A+B') for b in bands) or any(
            b.enabled and not b.bypass and b.bus_target in ('B', 'A+B') for b in mb_bands
        )
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
            lbl_txt = str(idx + 1)
            if getattr(lbl, "_last_txt", None) != lbl_txt:
                lbl._last_txt = lbl_txt
                lbl.setText(lbl_txt)
            lbl.setPos(nx, ny)
            if not lbl.isVisible():
                lbl.show()

            if band.is_dynamic_active and band.enabled:
                dyn_y = float(np.clip(ny + band.live_dyn_delta_db, -self.eq_range_db, self.eq_range_db))
                ring_c = _brighten_color(band.color, 0.25)
                dyn_spots.append({
                    "pos": (nx, dyn_y),
                    "size": size + 8,
                    "pen": pg.mkPen(color=ring_c, width=2.0),
                    "brush": pg.mkBrush(0, 0, 0, 0),
                })

        self.nodes_scatter.setData(node_spots)
        self.dyn_nodes_scatter.setData(dyn_spots)
        self.floating_tooltip.hide()

        if hasattr(self, "node_callout"):
            if eq_vis and not self._suppress_eq_callout and 0 <= self.selected_band_idx < len(bands):
                if force or self.node_callout.isVisible():
                    self.node_callout.update_callout()
            else:
                self.node_callout.hide()

        if force:
            HUDOpenGLBackgroundItem.invalidate_scene_blur_cache(self, getattr(self, "plot_widget", None))
        return True


class HUDOpenGLBackgroundItem(QGraphicsItem):
    """
    Fondo translúcido Apple Liquid Glass con Gaussian Blur compartido a 60 FPS (1/8-scale separable blur)
    y halo difuso coloreado:
    - Rasteriza el lienzo completo una sola vez por fotograma (60 FPS) en un buffer compartido
      (0 bucles Python QPointF usando pg.arrayToQPath) y todos los widgets/popups muestrean de él.
    - Superpone vidrio oscuro esmerilado preservando contraste y colores saturados del fondo.
    """

    _GAUSS_KERNEL = np.array([0.0545, 0.2442, 0.4026, 0.2442, 0.0545], dtype=np.float32)
    _scene_blur_frame_id: int = 0
    _scene_blur_last_time: float = 0.0
    _scene_blur_cache_key = None
    _scene_blur_arr: np.ndarray | None = None
    _scene_blur_pixmap: QPixmap | None = None
    _scene_blur_dims: tuple[float, float, float, float, int, int] = (0.0, 0.0, 1.0, 1.0, 1, 1)

    def __init__(self, overlay=None, plot_widget=None):
        super().__init__()
        self.setZValue(25)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.overlay = overlay
        self.plot_widget = plot_widget
        self._rect = QRectF(0.0, 0.0, 520.0, 208.0)
        self._is_minimized = False
        self._active_color = QColor("#00E5FF")
        self._gain_center_x = 276.0
        self._custom_path = None
        self._cached_path = QPainterPath()
        self._blur_pixmap = None
        self._blur_frame_id = -1
        self._blur_geom_key = None
        self._last_blur_time = 0.0
        self._border_overlay_pixmap: QPixmap | None = None
        self._border_overlay_key = None
        self._rebuild_path()

    def boundingRect(self) -> QRectF:
        return self._rect.adjusted(-10.0, -10.0, 10.0, 10.0)

    def set_custom_shape(self, scene_pos: QPointF, width: float, height: float, custom_path: QPainterPath, active_color: QColor):
        pos_changed = (self.pos() - scene_pos).manhattanLength() > 0.5
        self.setPos(scene_pos)
        c = QColor(active_color) if active_color else QColor("#00E5FF")
        geom_changed = (
            abs(self._rect.width() - width) > 0.5
            or abs(self._rect.height() - height) > 0.5
            or self._custom_path != custom_path
        )
        if geom_changed:
            self.prepareGeometryChange()
            self._rect = QRectF(0.0, 0.0, width, height)
            self._custom_path = custom_path
            self._cached_path = QPainterPath(custom_path)
            self._active_color = c
            self._last_blur_time = 0.0
            self._blur_geom_key = None
            self.update()
        elif self._active_color != c or pos_changed:
            self._active_color = c
            self.update()

    def set_state(self, scene_pos: QPointF, width: float, height: float, is_minimized: bool, active_color: QColor, gain_cx: float):
        pos_changed = (self.pos() - scene_pos).manhattanLength() > 0.5
        self.setPos(scene_pos)
        c = QColor(active_color) if active_color else QColor("#00E5FF")
        geom_changed = (
            abs(self._rect.width() - width) > 0.5
            or abs(self._rect.height() - height) > 0.5
            or self._is_minimized != is_minimized
            or abs(self._gain_center_x - gain_cx) > 0.5
            or self._custom_path is not None
        )
        if geom_changed:
            self.prepareGeometryChange()
            self._rect = QRectF(0.0, 0.0, width, height)
            self._is_minimized = is_minimized
            self._gain_center_x = gain_cx
            self._custom_path = None
            self._active_color = c
            self._rebuild_path()
            self._last_blur_time = 0.0
            self._blur_geom_key = None
            self.update()
        elif self._active_color != c or pos_changed:
            self._active_color = c
            self.update()

    def _rebuild_path(self):
        if self._custom_path is not None:
            self._cached_path = QPainterPath(self._custom_path)
            return
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
        arch_half_w = min(104.0, max(68.0, arch_cx - (x_left + r_corner + 8.0)))
        ax1 = max(x_left + r_corner + 6.0, arch_cx - arch_half_w)
        ax2 = min(x_right - r_corner - 6.0, arch_cx + arch_half_w)
        h_a = arch_half_w * 0.365
        h_b = arch_half_w * 0.404

        p = self._cached_path
        p.moveTo(x_left + r_corner, y_top)
        p.lineTo(ax1, y_top)
        p.cubicTo(
            QPointF(ax1 + h_a, y_top),
            QPointF(arch_cx - h_b, y_peak),
            QPointF(arch_cx, y_peak),
        )
        p.cubicTo(
            QPointF(arch_cx + h_b, y_peak),
            QPointF(ax2 - h_a, y_top),
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

    @classmethod
    def _separable_gaussian_blur(cls, arr: np.ndarray, k: np.ndarray) -> np.ndarray:
        # Dos pasadas separables X + Y ultra-rápidas mediante vistas continuas (sin sobrecosto de np.pad)
        bh, bw = arr.shape[0], arr.shape[1]
        for _ in range(2):
            px = np.empty((bh, bw + 4, 4), dtype=np.float32)
            px[:, 2:-2, :] = arr
            px[:, :2, :] = arr[:, 0:1, :]
            px[:, -2:, :] = arr[:, -1:, :]
            arr = (
                k[0] * px[:, :-4, :]
                + k[1] * px[:, 1:-3, :]
                + k[2] * px[:, 2:-2, :]
                + k[3] * px[:, 3:-1, :]
                + k[4] * px[:, 4:, :]
            )
            py = np.empty((bh + 4, bw, 4), dtype=np.float32)
            py[2:-2, :, :] = arr
            py[:2, :, :] = arr[0:1, :, :]
            py[-2:, :, :] = arr[-1:, :, :]
            arr = (
                k[0] * py[:-4, :, :]
                + k[1] * py[1:-3, :, :]
                + k[2] * py[2:-2, :, :]
                + k[3] * py[3:-1, :, :]
                + k[4] * py[4:, :, :]
            )
        return arr

    @classmethod
    def invalidate_scene_blur_cache(cls, overlay=None, plot_widget=None):
        cls._scene_blur_frame_id += 1
        cls._scene_blur_cache_key = None
        _HUDPopupGLHelper.repaint_active_popups()
        if overlay is not None:
            insp = getattr(overlay, "fft_inspector", None)
            if insp is not None and insp.isVisible():
                insp.update()
            nc = getattr(overlay, "node_callout", None)
            if nc is not None and nc.isVisible():
                nc_bg = getattr(nc, "_bg_gl_item", None)
                if nc_bg is not None and nc_bg.isVisible():
                    nc_bg.update()
                else:
                    nc.update()
            hud_c = getattr(overlay, "hud_container", None)
            hud = getattr(hud_c, "floating_hud", None) if hud_c is not None else None
            if hud is not None and hud.isVisible():
                hud_bg = getattr(hud, "_bg_gl_item", None)
                if hud_bg is not None and hud_bg.isVisible():
                    hud_bg.update()
                else:
                    hud.update()

    @staticmethod
    def _fast_fill_and_stroke_paths(bx_arr: np.ndarray, by_arr: np.ndarray, y_base: float | None = None):
        bx = np.asarray(bx_arr, dtype=np.float64)
        by = np.asarray(by_arr, dtype=np.float64)
        if not (np.all(np.isfinite(bx)) and np.all(np.isfinite(by))):
            m = np.isfinite(bx) & np.isfinite(by)
            bx = bx[m]
            by = by[m]
        if len(bx) < 2:
            return None, None
        stroke_path = pg.arrayToQPath(bx, by, connect="all", finiteCheck=False)
        if y_base is None:
            return stroke_path, None
        n = len(bx)
        x_fill = np.empty(n + 2, dtype=np.float64)
        y_fill = np.empty(n + 2, dtype=np.float64)
        x_fill[0] = bx[0]
        x_fill[1:-1] = bx
        x_fill[-1] = bx[-1]
        y_fill[0] = float(y_base)
        y_fill[1:-1] = by
        y_fill[-1] = float(y_base)
        fill_path = pg.arrayToQPath(x_fill, y_fill, connect="all", finiteCheck=False)
        fill_path.closeSubpath()
        return stroke_path, fill_path

    @classmethod
    def _ensure_scene_blur_buffer(cls, overlay, plot_widget):
        if overlay is None or plot_widget is None:
            return None

        vb = plot_widget.plotItem.vb
        vb_rect = vb.sceneBoundingRect()
        vx = float(vb_rect.left())
        vy = float(vb_rect.top())
        vw = max(1.0, float(vb_rect.width()))
        vh = max(1.0, float(vb_rect.height()))

        now = time.perf_counter()
        if cls._scene_blur_cache_key is not None and (now - cls._scene_blur_last_time) >= 0.036:
            cls._scene_blur_frame_id += 1
            cls._scene_blur_cache_key = None

        cache_key = (cls._scene_blur_frame_id, id(plot_widget), round(vw, 1), round(vh, 1))
        if cls._scene_blur_cache_key == cache_key and cls._scene_blur_pixmap is not None:
            return cls._scene_blur_pixmap, cls._scene_blur_dims

        full_bw = max(32, int(round(vw / 8.0)))
        full_bh = max(16, int(round(vh / 8.0)))

        app_win = plot_widget.window() if plot_widget is not None else None
        bg_hex = "#141414"
        if app_win is not None and hasattr(app_win, "theme") and isinstance(app_win.theme, dict):
            bg_hex = app_win.theme.get("background", "#141414")
        bg_base = QColor(bg_hex)
        if not bg_base.isValid():
            bg_base = QColor(14, 16, 22, 255)
        bg_dark = QColor(
            int(round(bg_base.red() * 0.90)),
            int(round(bg_base.green() * 0.90)),
            int(round(bg_base.blue() * 0.90)),
            255,
        )

        img = QImage(full_bw, full_bh, QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(bg_dark)

        ip = QPainter(img)
        ip.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        fbw = float(full_bw)
        fbh = float(full_bh)

        try:
            view_range = vb.viewRange()
            xmin, xmax = float(view_range[0][0]), float(view_range[0][1])
            ymin, ymax = float(view_range[1][0]), float(view_range[1][1])
            dx = max(1e-6, xmax - xmin)
            dy = max(1e-6, ymax - ymin)

            # 1. Líneas de grilla sutiles bajo el vidrio
            ip.setPen(QPen(QColor(60, 66, 82, 50), 0.55))
            for db_grid in (-18.0, -36.0, -54.0, -72.0, -90.0, -108.0):
                if ymin <= db_grid <= ymax:
                    ly = (ymax - db_grid) / dy * fbh
                    if 0.0 <= ly <= fbh:
                        ip.drawLine(QPointF(0.0, ly), QPointF(fbw, ly))

            # 2. Curvas FFT activas (In 1, In 2, Out 1, Out 2 y RTA) sin sobreiluminación ni bloat
            fft_list = list(getattr(overlay, "fft_curves", []))
            if app_win is not None:
                for attr_name in ("fft_curve_in", "fft_curve_in2", "fft_curve_out2", "fft_curve_out1", "rta_curve_item"):
                    c_obj = getattr(app_win, attr_name, None)
                    if c_obj is not None and c_obj not in fft_list:
                        fft_list.append(c_obj)

            c_in_item = getattr(app_win, "fft_curve_in", None) if app_win is not None else None
            c_in2_item = getattr(app_win, "fft_curve_in2", None) if app_win is not None else None
            c_out2_item = getattr(app_win, "fft_curve_out2", None) if app_win is not None else None
            c_rta_item = getattr(app_win, "rta_curve_item", None) if app_win is not None else None

            for entry in fft_list:
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
                bx_arr = ((xs - xmin) / dx) * fbw
                by_arr = ((ymax - ys) / dy) * fbh
                c_fft = QColor(c_hex) if isinstance(c_hex, str) else QColor(c_hex)
                if not c_fft.isValid():
                    c_fft = QColor("#00E5FF")

                if curve_item is c_rta_item:
                    fill_alpha = 0
                elif curve_item is c_in_item:
                    fill_alpha = 38
                elif curve_item is c_in2_item or curve_item is c_out2_item:
                    fill_alpha = 68
                else:
                    fill_alpha = 82

                stroke_path, fill_path = cls._fast_fill_and_stroke_paths(
                    bx_arr, by_arr, y_base=(fbh + 4.0) if fill_alpha > 0 else None
                )
                if stroke_path is None:
                    continue

                if fill_path is not None and fill_alpha > 0:
                    fill_c = QColor(c_fft)
                    fill_c.setAlpha(fill_alpha)
                    ip.setPen(Qt.PenStyle.NoPen)
                    ip.setBrush(QBrush(fill_c))
                    ip.drawPath(fill_path)

                stroke_c = QColor(c_fft)
                stroke_c.setAlpha(230)
                ip.setBrush(Qt.BrushStyle.NoBrush)
                ip.setPen(QPen(stroke_c, 1.35))
                ip.drawPath(stroke_path)

            # 3. Curvas EQ estáticas y dinámicas bajo la transparencia
            c_mode = getattr(overlay, "canvas_mode", "BOTH")
            eq_vb = getattr(overlay, "eq_vb", None)
            if eq_vb is not None:
                eq_range = eq_vb.viewRange()
                eymin, eymax = float(eq_range[1][0]), float(eq_range[1][1])
                edy = max(1e-6, eymax - eymin)
                y0_local = ((eymax - 0.0) / edy) * fbh

                if c_mode in ("EQ", "BOTH"):
                    band_fill_list = getattr(overlay, "band_fill_curves", [])
                    band_dyn_list = getattr(overlay, "band_dyn_curves", [])

                    bands_list = getattr(overlay.eq_engine, "bands", ())
                    sel_b_idx = getattr(overlay, "selected_band_idx", -1)
                    for b_idx, st_c in enumerate(band_fill_list):
                        if not st_c.isVisible():
                            continue
                        if b_idx < len(bands_list):
                            b_obj = bands_list[b_idx]
                            if not b_obj.enabled or (
                                b_idx != sel_b_idx
                                and abs(b_obj.gain_db) <= 0.05
                                and b_obj.supports_gain
                                and not b_obj.is_dynamic_active
                            ):
                                continue
                        xd, yd = st_c.getData()
                        if xd is None or yd is None or len(xd) < 4:
                            continue
                        step = max(1, len(xd) // 64)
                        xs = xd[::step]
                        ys = yd[::step]
                        bx_arr = ((xs - xmin) / dx) * fbw
                        by_arr = ((eymax - ys) / edy) * fbh

                        b_col = QColor("#00E5FF")
                        if b_idx < len(bands_list):
                            b_col = QColor(bands_list[b_idx].color)

                        stroke_path, fill_path = cls._fast_fill_and_stroke_paths(bx_arr, by_arr, y_base=y0_local)
                        if stroke_path is not None:
                            if fill_path is not None:
                                fc = QColor(b_col)
                                fc.setAlpha(62 if b_idx == sel_b_idx else 36)
                                ip.setPen(Qt.PenStyle.NoPen)
                                ip.setBrush(QBrush(fc))
                                ip.drawPath(fill_path)

                            sc = QColor(b_col)
                            sc.setAlpha(215)
                            ip.setBrush(Qt.BrushStyle.NoBrush)
                            ip.setPen(QPen(sc, 1.15))
                            ip.drawPath(stroke_path)

                        # Límite de rango dinámico en trazado punteado ligero (el cuerpo principal ya refleja la curva en vivo)
                        if b_idx < len(band_dyn_list):
                            dyn_c = band_dyn_list[b_idx]
                            if dyn_c is not None and dyn_c.isVisible():
                                dxd, dyd = dyn_c.getData()
                                if dxd is not None and dyd is not None and len(dxd) >= 4:
                                    dstep = max(1, len(dxd) // 64)
                                    dxs = dxd[::dstep]
                                    dys = dyd[::dstep]
                                    dbx_arr = ((dxs - xmin) / dx) * fbw
                                    dby_arr = ((eymax - dys) / edy) * fbh
                                    dyn_stroke_path, _ = cls._fast_fill_and_stroke_paths(dbx_arr, dby_arr, y_base=None)
                                    if dyn_stroke_path is not None:
                                        dsc = QColor(b_col)
                                        dsc.setAlpha(175 if b_idx == sel_b_idx else 95)
                                        ip.setBrush(Qt.BrushStyle.NoBrush)
                                        ip.setPen(QPen(dsc, 0.95, Qt.PenStyle.DotLine))
                                        ip.drawPath(dyn_stroke_path)

                    # Curva de límite máximo de rango dinámico (dyn_limit_curve)
                    lim_c = getattr(overlay, "dyn_limit_curve", None)
                    if lim_c is not None and lim_c.isVisible():
                        lxd, lyd = lim_c.getData()
                        if lxd is not None and lyd is not None and len(lxd) >= 4:
                            lstep = max(1, len(lxd) // 64)
                            lxs = lxd[::lstep]
                            lys = lyd[::lstep]
                            lbx = ((lxs - xmin) / dx) * fbw
                            lby = ((eymax - lys) / edy) * fbh
                            lpath, _ = cls._fast_fill_and_stroke_paths(lbx, lby, y_base=None)
                            if lpath is not None:
                                lcol = QColor("#00E5FF")
                                if 0 <= sel_b_idx < len(bands_list):
                                    lcol = QColor(bands_list[sel_b_idx].color)
                                lcol.setAlpha(190)
                                ip.setBrush(Qt.BrushStyle.NoBrush)
                                ip.setPen(QPen(lcol, 0.95, Qt.PenStyle.DotLine))
                                ip.drawPath(lpath)

                    # Curva Fantasma sugerida (ghost_curve)
                    ghost_c = getattr(overlay, "ghost_curve", None)
                    if ghost_c is not None and ghost_c.isVisible() and not ghost_c.path.isEmpty():
                        ex = getattr(overlay, "eval_x", None)
                        if ex is not None and len(ex) >= 4 and getattr(overlay, "_proposed_band", None) is not None:
                            try:
                                pb = overlay._proposed_band
                                tb = EQBand(filter_type=pb["type"], freq=pb["freq"], gain_db=pb["gain"], q=pb["q"], sr=overlay.eq_engine.sr)
                                gy = tb.get_static_freq_response_db(overlay.eval_z1, overlay.eval_z2)
                                gstep = max(1, len(ex) // 48)
                                gxs = ex[::gstep]
                                gys = gy[::gstep]
                                gbx = ((gxs - xmin) / dx) * fbw
                                gby = ((eymax - gys) / edy) * fbh
                                gpath, _ = cls._fast_fill_and_stroke_paths(gbx, gby, y_base=None)
                                if gpath is not None:
                                    ip.setBrush(Qt.BrushStyle.NoBrush)
                                    ip.setPen(QPen(QColor(245, 195, 85, 180), 1.05))
                                    ip.drawPath(gpath)
                            except Exception:
                                pass

                    for m_curve, m_col_fb in (
                        (getattr(overlay, "static_master_curve", None), getattr(overlay, "_color_eq_a", "#FFD54F")),
                        (getattr(overlay, "static_bus_b_curve", None), getattr(overlay, "_color_eq_b", "#00E5FF")),
                    ):
                        if m_curve is None or not m_curve.isVisible():
                            continue
                        xd, yd = m_curve.getData()
                        if xd is None or yd is None or len(xd) < 4:
                            continue
                        step = max(1, len(xd) // 64)
                        xs = xd[::step]
                        ys = yd[::step]
                        bx_arr = ((xs - xmin) / dx) * fbw
                        by_arr = ((eymax - ys) / edy) * fbh
                        mpath, _ = cls._fast_fill_and_stroke_paths(bx_arr, by_arr, y_base=None)
                        if mpath is not None:
                            pen_c = QColor(m_col_fb)
                            pen_c.setAlpha(235)
                            ip.setBrush(Qt.BrushStyle.NoBrush)
                            ip.setPen(QPen(pen_c, 1.35))
                            ip.drawPath(mpath)

                # 3.1 Zonas Pro-MB ligeras bajo el vidrio esmerilado
                mb_proc = getattr(overlay, "multiband_dyn", None)
                if mb_proc is not None and c_mode in ("MB", "BOTH"):
                    for b_mb in list(mb_proc.bands):
                        if not b_mb.enabled or b_mb.bypass or mb_proc.bypass:
                            continue
                        x0_mb = ((overlay.freq_to_x(b_mb.f_low) - xmin) / dx) * fbw
                        x1_mb = ((overlay.freq_to_x(b_mb.f_high) - xmin) / dx) * fbw
                        mc = QColor(b_mb.color)
                        mc.setAlpha(24)
                        ip.fillRect(QRectF(x0_mb, 0.0, max(2.0, x1_mb - x0_mb), fbh), mc)

                # 4. Curvas Crossover bajo la transparencia
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
                    bx_arr = ((xs - xmin) / dx) * fbw
                    by_arr = ((eymax - ys) / edy) * fbh
                    xpath, _ = cls._fast_fill_and_stroke_paths(bx_arr, by_arr, y_base=None)
                    if xpath is not None:
                        pen_c = QColor(x_col_fb)
                        pen_c.setAlpha(220)
                        ip.setBrush(Qt.BrushStyle.NoBrush)
                        ip.setPen(QPen(pen_c, 1.20))
                        ip.drawPath(xpath)
        except Exception:
            pass
        finally:
            ip.end()

        # Aplicar Gaussian Blur separable rápido en NumPy + ligera atenuación de luminancia (0.92x)
        ptr = img.bits()
        ptr.setsize(full_bh * full_bw * 4)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((full_bh, full_bw, 4)).astype(np.float32)
        blurred = cls._separable_gaussian_blur(arr, cls._GAUSS_KERNEL)
        blurred[..., :3] *= 0.92
        blurred[..., 3] = 255.0
        out_u8 = np.ascontiguousarray(np.clip(blurred, 0.0, 255.0).astype(np.uint8))
        qimg_scene = QImage(out_u8.data, full_bw, full_bh, full_bw * 4, QImage.Format.Format_ARGB32_Premultiplied)
        cls._scene_blur_arr = blurred
        cls._scene_blur_pixmap = QPixmap.fromImage(qimg_scene.copy())
        cls._scene_blur_dims = (vx, vy, vw, vh, full_bw, full_bh)
        cls._scene_blur_last_time = time.perf_counter()
        cls._scene_blur_cache_key = cache_key
        return cls._scene_blur_pixmap, cls._scene_blur_dims

    @classmethod
    def render_scene_blur_pixmap(
        cls,
        overlay,
        plot_widget,
        hx: float,
        hy: float,
        w: float,
        h: float,
        fallback_color: QColor | None = None,
        corner_radius: float = 12.0,
    ) -> QPixmap | None:
        if w <= 4.0 or h <= 4.0 or overlay is None or plot_widget is None:
            return None

        buf_res = cls._ensure_scene_blur_buffer(overlay, plot_widget)
        if buf_res is None:
            return None
        scene_pix, (vx, vy, vw, vh, full_bw, full_bh) = buf_res

        bw = max(16, int(round(w / 8.0)))
        bh = max(6, int(round(h / 8.0)))

        sx_scale = float(full_bw) / max(1.0, vw)
        sy_scale = float(full_bh) / max(1.0, vh)
        src_x = float(np.clip((float(hx) - vx) * sx_scale, 0.0, max(0.0, float(full_bw - 1))))
        src_y = float(np.clip((float(hy) - vy) * sy_scale, 0.0, max(0.0, float(full_bh - 1))))
        src_w = max(1.0, min(float(full_bw) - src_x, float(w) * sx_scale))
        src_h = max(1.0, min(float(full_bh) - src_y, float(h) * sy_scale))

        sub_pix = QPixmap(bw, bh)
        sub_pix.fill(Qt.GlobalColor.transparent)
        sp = QPainter(sub_pix)
        sp.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        sp.drawPixmap(QRectF(0.0, 0.0, float(bw), float(bh)), scene_pix, QRectF(src_x, src_y, src_w, src_h))
        sp.end()
        return sub_pix

    def _update_gaussian_blur_texture(self):
        w = float(self._rect.width())
        h = float(self._rect.height())
        hud_pos = self.pos()
        hx = float(hud_pos.x())
        hy = float(hud_pos.y())
        corner_r = 8.0 if self._custom_path is not None else (10.0 if self._is_minimized else 18.0)
        now = time.perf_counter()
        cur_fid = self._scene_blur_frame_id
        geom_key = (int(round(hx)), int(round(hy)), int(round(w)), int(round(h)), self._is_minimized, round(corner_r, 1))
        if (
            self._blur_pixmap is not None
            and self._blur_frame_id == cur_fid
            and self._blur_geom_key == geom_key
            and (now - self._scene_blur_last_time) < 0.036
        ):
            return

        pix = self.render_scene_blur_pixmap(
            self.overlay,
            self.plot_widget,
            hx,
            hy,
            w,
            h,
            self._active_color,
            corner_radius=corner_r,
        )
        if pix is not None:
            self._blur_pixmap = pix
            self._blur_frame_id = self._scene_blur_frame_id
            self._blur_geom_key = geom_key

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        self._update_gaussian_blur_texture()

        dpr = float(widget.devicePixelRatioF()) if (widget is not None and hasattr(widget, "devicePixelRatioF")) else 1.0
        w = float(self._rect.width())
        h = float(self._rect.height())
        bkey = (
            int(round(w)),
            int(round(h)),
            round(dpr, 2),
            self._is_minimized,
            round(self._gain_center_x, 1),
            self._active_color.rgba(),
            id(self._custom_path) if self._custom_path is not None else 0,
        )
        if self._border_overlay_pixmap is None or self._border_overlay_key != bkey:
            pw = max(1, int(round((w + 20.0) * dpr)))
            ph = max(1, int(round((h + 20.0) * dpr)))
            bpix = QPixmap(pw, ph)
            bpix.setDevicePixelRatio(dpr)
            bpix.fill(Qt.GlobalColor.transparent)
            bp = QPainter(bpix)
            bp.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            bp.translate(10.0, 10.0)
            bp.save()
            bp.setClipPath(self._cached_path)
            bp.fillRect(self._rect, QColor(10, 12, 18, 96))
            tint = QColor(self._active_color)
            tint.setAlpha(12)
            bp.fillRect(self._rect, tint)
            bp.restore()
            draw_liquid_glass_border_glow(bp, self._cached_path, self._rect, self._active_color)
            bp.end()
            self._border_overlay_pixmap = bpix
            self._border_overlay_key = bkey

        if self._blur_pixmap is not None and not self._blur_pixmap.isNull():
            painter.save()
            painter.setClipPath(self._cached_path)
            painter.drawPixmap(self._rect.toRect(), self._blur_pixmap)
            painter.restore()
        else:
            painter.fillPath(self._cached_path, QBrush(QColor(10, 12, 18, 180)))

        painter.drawPixmap(QPointF(-10.0, -10.0), self._border_overlay_pixmap)


class ArcMiniSlider(QWidget):
    """
    Deslizador horizontal recto y limpio (para KNEE y RANGE):
    - Traza una pista horizontal recta, segmento activo iluminado + punto indicador y etiqueta.
    - Arrastrar horizontal/vertical (Shift = ajuste fino), rueda y doble clic para reset.
    """

    valueChanged = pyqtSignal(float)

    def __init__(
        self,
        title: str,
        min_val: float,
        max_val: float,
        default_val: float,
        fmt_func=None,
        bipolar: bool = False,
        accent_color: str = "#00E5FF",
        parent=None,
    ):
        super().__init__(parent)
        self.title = title
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.default_val = float(default_val)
        self._value = float(default_val)
        self.fmt_func = fmt_func or (lambda v: f"{v:.1f}")
        self.bipolar = bool(bipolar)
        self.accent_color = QColor(accent_color)
        self._dragging = False
        self._hovered = False
        self._last_pos = QPointF()
        self.setFixedSize(124, 24)
        self.setCursor(Qt.CursorShape.SizeHorCursor)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.setToolTip(f"{self.title}: Arrastrar | Shift = Ajuste fino | Rueda | Doble clic = Reset")

    def value(self) -> float:
        return self._value

    def setValue(self, val: float, emit_signal: bool = False):
        clamped = float(np.clip(val, self.min_val, self.max_val))
        if abs(clamped - self._value) > 1e-6:
            self._value = clamped
            self.update()
            if emit_signal:
                self.valueChanged.emit(self._value)

    def setAccentColor(self, color_str: str):
        c = QColor(color_str)
        if c.isValid() and c != self.accent_color:
            self.accent_color = c
            self.update()

    def _norm(self, val: float) -> float:
        span = max(1e-6, self.max_val - self.min_val)
        return float(np.clip((val - self.min_val) / span, 0.0, 1.0))

    def enterEvent(self, event):
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._last_pos = event.position()
            self.update()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._dragging:
            dx = event.position().x() - self._last_pos.x()
            dy = self._last_pos.y() - event.position().y()
            self._last_pos = event.position()
            delta_px = dx if abs(dx) >= abs(dy) else dy
            fine = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            dn = delta_px / (500.0 if fine else 130.0)
            new_n = float(np.clip(self._norm(self._value) + dn, 0.0, 1.0))
            self.setValue(self.min_val + new_n * (self.max_val - self.min_val), emit_signal=True)
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            self.update()
            event.accept()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setValue(self.default_val, emit_signal=True)
            event.accept()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta != 0:
            fine = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            dn = (0.01 if fine else 0.04) * (1.0 if delta > 0 else -1.0)
            new_n = float(np.clip(self._norm(self._value) + dn, 0.0, 1.0))
            self.setValue(self.min_val + new_n * (self.max_val - self.min_val), emit_signal=True)
        event.accept()

    def paintEvent(self, event):
        w = int(self.width())
        h = int(self.height())
        if w <= 0 or h <= 0:
            return
        dpr = float(self.devicePixelRatioF()) if hasattr(self, "devicePixelRatioF") else 1.0
        enabled = self.isEnabled()
        cache_key = (
            w, h, round(dpr, 2), enabled, round(self._value, 4),
            self.accent_color.rgba(), self._hovered, self._dragging,
        )
        full_pix = getattr(self, "_full_cache_pix", None)
        if full_pix is None or getattr(self, "_full_cache_key", None) != cache_key:
            pw = max(1, int(round(w * dpr)))
            ph = max(1, int(round(h * dpr)))
            pix = QPixmap(pw, ph)
            pix.setDevicePixelRatio(dpr)
            pix.fill(Qt.GlobalColor.transparent)
            p = QPainter(pix)
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            wf, hf = float(w), float(h)

            x0 = 8.0
            x1 = max(x0 + 10.0, wf - 8.0)
            span_w = x1 - x0
            y_bar = 5.5
            track_rect = QRectF(x0, y_bar - 2.0, span_w, 4.0)
            p.setPen(QPen(QColor(68, 74, 88, 175 if enabled else 90), 0.9))
            p.setBrush(QBrush(QColor(16, 18, 24, 210 if enabled else 130)))
            p.drawRoundedRect(track_rect, 2.0, 2.0)

            n_val = self._norm(self._value)
            col = self.accent_color if enabled else QColor("#4E535E")
            x_val = x0 + n_val * span_w
            x_anchor = (x0 + self._norm(0.0) * span_w) if self.bipolar else x0
            xa = min(x_anchor, x_val)
            xb = max(x_anchor, x_val)
            if (xb - xa) > 0.8:
                glow_c = QColor(col)
                glow_c.setAlpha(70 if enabled else 30)
                p.setPen(QPen(glow_c, 4.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                p.drawLine(QPointF(xa, y_bar), QPointF(xb, y_bar))

                core_c = _brighten_color(col, 0.20) if enabled else col
                core_c.setAlpha(240 if enabled else 140)
                p.setPen(QPen(core_c, 2.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                p.drawLine(QPointF(xa, y_bar), QPointF(xb, y_bar))

            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(_brighten_color(col, 0.42) if enabled else QColor("#686C75")))
            p.drawEllipse(QPointF(x_val, y_bar), 2.5, 2.5)

            txt = f"{self.title}  {self.fmt_func(self._value)}"
            p.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            txt_c = QColor("#E2E6EE") if (self._hovered or self._dragging) else (QColor("#AEB6C4") if enabled else QColor("#585D68"))
            p.setPen(txt_c)
            p.drawText(QRectF(0.0, 10.5, wf, hf - 10.5), Qt.AlignmentFlag.AlignCenter, txt)
            p.end()
            self._full_cache_key = cache_key
            self._full_cache_pix = pix
            full_pix = pix

        QPainter(self).drawPixmap(0, 0, full_pix)


class CharacterResistorButton(QPushButton):
    """
    Elemento gráfico SVG puro de resistencia (character.svg) sin caja ni borde de botón:
    permanece apagado en gris metálico cuando Character == Off y se ilumina con resplandor
    de filamento ámbar cálido al activar Tube / Diode / Bright, desplegando el menú al presionarlo.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._char_mode: str = "Off"
        self.setFixedSize(44, 30)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.setToolTip("Character (Saturación armónica por banda: Off / Tube / Diode / Bright)")

    def setCharacterState(self, mode: str):
        m = mode if mode in MB_CHARACTER_MODES else "Off"
        if m != self._char_mode:
            self._char_mode = m
            self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = float(self.width()), float(self.height())
        dpr = float(self.devicePixelRatioF()) if hasattr(self, "devicePixelRatioF") else 1.0
        enabled = self.isEnabled()
        active = enabled and (self._char_mode != "Off")
        hov = self.underMouse()

        st = "disabled" if not enabled else ("on" if active else ("hover" if hov else "off"))
        isz = max(20, int(min(w, h) - 1))
        pix = _render_svg_pixmap(
            "character.svg", isz, isz, dpr=dpr, state=st,
            accent_hex="#FFA94D", recolor_primary=True
        )
        p.drawPixmap(int((w - isz) * 0.5), int((h - isz) * 0.5), pix)


class MBModeSvgButton(QPushButton):
    """
    Selector gráfico SVG puro de curva de dinámica Multibanda (Compress / Expand / Limit):
    sin caja ni borde exterior — la propia curva SVG iluminada es el botón y despliega el menú al presionarlo.
    """

    def __init__(self, parent=None, active_tint_color: str = "#00E5FF"):
        super().__init__("Compress ▾", parent)
        self._mode: str = "Compress"
        self._active_color = QColor(active_tint_color)
        self.setFixedSize(44, 30)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)

    def setActiveTintColor(self, hex_color: str):
        c = QColor(hex_color)
        if c.isValid() and c != self._active_color:
            self._active_color = c
            self.update()

    def setText(self, text: str):
        super().setText(text)
        t = str(text)
        if t.startswith("Limit"):
            self._mode = "Limit"
        elif t.startswith("Expand"):
            self._mode = "Expand"
        else:
            self._mode = "Compress"
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = float(self.width()), float(self.height())
        dpr = float(self.devicePixelRatioF()) if hasattr(self, "devicePixelRatioF") else 1.0
        enabled = self.isEnabled()
        hov = self.underMouse()

        svg_file = MB_MODE_SVGS.get(self._mode, "comp_mode.svg")
        st = "disabled" if not enabled else ("on" if hov else "hover")
        isz = max(20, int(min(w, h) - 1))
        pix = _render_svg_pixmap(
            svg_file, isz, isz, dpr=dpr, state=st,
            accent_hex=self._active_color.name(), recolor_primary=True
        )
        p.drawPixmap(int((w - isz) * 0.5), int((h - isz) * 0.5), pix)


class _DriveBarSlider(QWidget):
    """Barra horizontal naranja estilo FabFilter Pro-C para DRIVE (0..100%)."""

    valueChanged = pyqtSignal(float)

    def __init__(self, default_val: float = 0.35, parent=None):
        super().__init__(parent)
        self._value = float(default_val)
        self._default = float(default_val)
        self._dragging = False
        self.setFixedSize(134, 28)
        self.setCursor(Qt.CursorShape.SizeHorCursor)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.setToolTip("DRIVE: Intensidad de saturación armónica (Doble clic = Reset)")

    def value(self) -> float:
        return self._value

    def setValue(self, val: float):
        v = float(np.clip(val, 0.0, 1.0))
        if abs(v - self._value) > 1e-4:
            self._value = v
            self.update()

    def _update_from_x(self, x: float):
        pad = 6.0
        w = max(10.0, float(self.width()) - 2.0 * pad)
        v = float(np.clip((x - pad) / w, 0.0, 1.0))
        if abs(v - self._value) > 1e-4:
            self._value = v
            self.update()
            self.valueChanged.emit(self._value)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._update_from_x(event.position().x())
            event.accept()

    def mouseMoveEvent(self, event):
        if self._dragging:
            self._update_from_x(event.position().x())
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            event.accept()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._value = self._default
            self.update()
            self.valueChanged.emit(self._value)
            event.accept()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta != 0:
            step = 0.05 * (1.0 if delta > 0 else -1.0)
            v = float(np.clip(self._value + step, 0.0, 1.0))
            if abs(v - self._value) > 1e-4:
                self._value = v
                self.update()
                self.valueChanged.emit(self._value)
        event.accept()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w = float(self.width())
        bar_rect = QRectF(6.0, 3.0, w - 12.0, 7.0)
        p.setPen(QPen(QColor(68, 74, 88, 180), 0.9))
        p.setBrush(QBrush(QColor(16, 18, 24, 210)))
        p.drawRoundedRect(bar_rect, 3.5, 3.5)

        if self._value > 0.005:
            fw = max(4.0, (w - 14.0) * self._value)
            fill_rect = QRectF(7.0, 4.0, fw, 5.0)
            grad = QLinearGradient(7.0, 0.0, 7.0 + fw, 0.0)
            grad.setColorAt(0.0, QColor("#E65100"))
            grad.setColorAt(1.0, QColor("#FFB74D"))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(grad))
            p.drawRoundedRect(fill_rect, 2.5, 2.5)

        p.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        p.setPen(QColor("#C8CEDB"))
        p.drawText(QRectF(0.0, 12.0, w, 15.0), Qt.AlignmentFlag.AlignCenter, f"DRIVE  {int(round(self._value * 100))}%")


class MBCharacterPopup(HUDPopupDialog):
    """
    Popup flotante estilo FabFilter Pro-C CHARACTER (Off / Tube / Diode / Bright + DRIVE + POST)
    conservando la estética Liquid Glass con blur GPU de toda la interfaz.
    """

    characterChanged = pyqtSignal(str, float, bool)

    def __init__(self, parent=None):
        super().__init__(parent=parent, accent_color="#FF9100", corner_radius=10.0)
        self.setFixedSize(244, 108)
        self._mode = "Off"
        self._drive = 0.35
        self._post = True

        self.setStyleSheet("""
            QWidget {
                background: transparent;
                color: #ECEFF4;
                font-family: "Segoe UI", sans-serif;
            }
            QPushButton {
                background-color: rgba(16, 18, 24, 145);
                color: #B8C0CE;
                border: 1px solid rgba(68, 74, 90, 165);
                border-radius: 5px;
                font-size: 10px;
                font-weight: bold;
                padding: 2px 6px;
            }
            QPushButton:hover {
                color: #FFFFFF;
                border-color: rgba(255, 145, 0, 175);
            }
            QPushButton:checked {
                background-color: rgba(255, 145, 0, 56);
                color: #FFD180;
                border: 1px solid #FF9100;
            }
        """)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 10)
        lay.setSpacing(7)

        hdr = QLabel("CHARACTER", self)
        hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hdr.setStyleSheet("color: #DDE3EE; font-size: 10px; font-weight: bold; letter-spacing: 0.8px;")
        lay.addWidget(hdr)

        mode_row = QHBoxLayout()
        mode_row.setContentsMargins(0, 0, 0, 0)
        mode_row.setSpacing(4)
        self._mode_btns: dict[str, QPushButton] = {}
        for m in MB_CHARACTER_MODES:
            btn = QPushButton(m, self)
            btn.setCheckable(True)
            btn.setFixedHeight(23)
            btn.clicked.connect(lambda _=False, mv=m: self._on_mode_btn(mv))
            mode_row.addWidget(btn)
            self._mode_btns[m] = btn
        lay.addLayout(mode_row)

        bot_row = QHBoxLayout()
        bot_row.setContentsMargins(0, 0, 0, 0)
        bot_row.setSpacing(8)
        self.drive_slider = _DriveBarSlider(default_val=0.35, parent=self)
        self.drive_slider.valueChanged.connect(self._on_drive_changed)
        bot_row.addWidget(self.drive_slider)

        self.post_btn = QPushButton("• POST", self)
        self.post_btn.setCheckable(True)
        self.post_btn.setFixedSize(68, 24)
        self.post_btn.setToolTip("Ubicación de saturación: POST (después de dinámica) / PRE (antes)")
        self.post_btn.clicked.connect(self._on_post_toggled)
        bot_row.addWidget(self.post_btn)
        lay.addLayout(bot_row)

    def set_state(self, mode: str, drive: float, post: bool):
        self._mode = mode if mode in MB_CHARACTER_MODES else "Off"
        self._drive = float(np.clip(drive, 0.0, 1.0))
        self._post = bool(post)
        for m, btn in self._mode_btns.items():
            btn.blockSignals(True)
            btn.setChecked(m == self._mode)
            btn.blockSignals(False)
        self.drive_slider.setValue(self._drive)
        self.post_btn.blockSignals(True)
        self.post_btn.setChecked(self._post)
        self.post_btn.setText("• POST" if self._post else "• PRE")
        self.post_btn.blockSignals(False)

    def _emit_state(self):
        self.characterChanged.emit(self._mode, self._drive, self._post)

    def _on_mode_btn(self, mode_val: str):
        self.set_state(mode_val, self._drive, self._post)
        self._emit_state()

    def _on_drive_changed(self, val: float):
        self._drive = float(val)
        if self._mode == "Off" and self._drive > 0.02:
            self.set_state("Tube", self._drive, self._post)
        self._emit_state()

    def _on_post_toggled(self, checked: bool):
        self._post = bool(checked)
        self.post_btn.setText("• POST" if self._post else "• PRE")
        self._emit_state()


class FloatingBandHUD(QWidget):
    """
    HUD Flotante Superpuesto estilo FabFilter Pro-Q 4 + Apple Liquid Glass Oscuro:
    - Fondo semitransparente composicionado en la GPU vía QGraphicsItem en OpenGL (0% CPU).
    - Arrastrable libremente (Draggable) por todo el lienzo con memoria de posición.
    - Modo EQ compacto (520px) que se ensancha suavemente (768px) al activar DYN para revelar THR, A y R.
    - Modo Multibanda Pro-MB simétrico (768px) con THRESHOLD centrado bajo la joroba y 3+3 perillas espejo.
    - Modo minimizado compacto (40px) anclado a la base inferior + Auto-Expand con cursor.
    """

    HUD_EXPANDED_H = 208
    HUD_COMPACT_H = 40
    HUD_COMPACT_W = 720
    HUD_EQ_BASE_W = 520
    HUD_EQ_DYN_W = 768
    HUD_MB_W = 768

    parameters_changed = pyqtSignal()
    active_accent_changed = pyqtSignal(str)
    ensure_layer_requested = pyqtSignal(str)

    def __init__(self, eq_engine: ParametricEQEngine, overlay: ProQ4Overlay, parent=None):
        super().__init__(parent)
        self.eq_engine = eq_engine
        self.overlay = overlay
        self.multiband_dyn: MultibandDynamicsProcessor | None = None
        self._hud_mode: str = "eq"  # "eq" | "mb"
        self._mb_ratio_is_limit: bool = False
        self._last_mb_gr_txt: str | None = None
        self._last_emitted_accent_hex: str | None = None
        self._mb_char_popup: MBCharacterPopup | None = None
        self._mb_mode_menu: HUDPopupMenu | None = None
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

        self._help_force_dyn_open = False
        self._help_force_slope_visible = False
        self._dyn_drawer_open = False
        self._dyn_anim_t = 0.0
        self._anim_anchor_cx: float | None = None
        self._dyn_width_anim = QVariantAnimation(self)
        self._dyn_width_anim.setDuration(240)
        self._dyn_width_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._dyn_width_anim.valueChanged.connect(self._on_dyn_width_anim_step)
        self._dyn_width_anim.finished.connect(self._on_dyn_width_anim_finished)

        self._leave_timer = QTimer(self)
        self._leave_timer.setSingleShot(True)
        self._leave_timer.timeout.connect(self._on_leave_timeout)

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFixedSize(self.HUD_EQ_BASE_W, self.HUD_EXPANDED_H)

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
        """)
        self._apply_hud_stylesheet(self._active_band_color)

        self._build_ui()
        self.overlay.band_selected.connect(self.select_band)
        self.overlay.band_modified.connect(self._on_overlay_band_modified)
        self.overlay.mb_band_selected.connect(self.select_mb_band)
        self.overlay.mb_band_modified.connect(self._on_overlay_mb_modified)
        self.refresh_from_engine()

        self.installEventFilter(self)
        self.expanded_container.installEventFilter(self)
        self.compact_container.installEventFilter(self)
        self.mb_container.installEventFilter(self)
        self.drag_grip_btn.installEventFilter(self)
        self.compact_drag_grip.installEventFilter(self)
        self.mb_drag_grip_btn.installEventFilter(self)

    def _apply_hud_stylesheet(self, color: QColor):
        c = QColor(color) if color else QColor("#00E5FF")
        hex_c = c.name()
        if getattr(self, "_last_hud_qss_color", None) == hex_c:
            return
        self._last_hud_qss_color = hex_c
        if hasattr(self, "auto_expand_btn"):
            self._update_auto_expand_style()

    def _update_auto_expand_style(self):
        c = QColor(self._active_band_color) if self._active_band_color else QColor("#00E5FF")
        hex_c = c.name()
        for attr in ("auto_expand_btn", "compact_auto_btn", "mb_auto_expand_btn"):
            btn = getattr(self, attr, None)
            if btn is not None:
                btn.setActiveTintColor(hex_c)
                btn.setChecked(self._auto_expand)
                btn.update()
        for attr in ("minimize_btn", "mb_minimize_btn", "compact_expand_btn"):
            btn = getattr(self, attr, None)
            if btn is not None and hasattr(btn, "setActiveTintColor"):
                btn.setActiveTintColor(hex_c)

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

    def _target_expanded_width(self) -> int:
        if self._hud_mode == "mb":
            return self.HUD_MB_W
        return int(round(self.HUD_EQ_BASE_W + self._dyn_anim_t * (self.HUD_EQ_DYN_W - self.HUD_EQ_BASE_W)))

    def _layout_eq_left_controls(self, has_slope: bool):
        if not hasattr(self, "band_enable_btn"):
            return
        if has_slope:
            self.band_enable_btn.move(16, 52)
            self.color_swatch.move(50, 55)
            self.type_combo.move(16, 88)
            self.slope_combo.move(16, 126)
        else:
            self.band_enable_btn.move(16, 66)
            self.color_swatch.move(50, 69)
            self.type_combo.move(16, 104)

    def _apply_eq_dyn_layout(self, t: float, reposition_window: bool = True):
        t = float(np.clip(t, 0.0, 1.0))
        self._dyn_anim_t = t
        cur_w = int(round(self.HUD_EQ_BASE_W + t * (self.HUD_EQ_DYN_W - self.HUD_EQ_BASE_W)))
        out_x = int(round(420.0 + t * (670.0 - 420.0)))
        self.out_gain_knob.move(out_x, 64)

        if t <= 0.02:
            for knob in (self.thresh_knob, self.attack_knob, self.release_knob):
                knob.setRevealAlpha(0.0)
                if not knob.isHidden():
                    knob.hide()
        else:
            slide_dx = int(round((1.0 - t) * -18.0))
            alpha = float(np.clip((t - 0.10) / 0.90, 0.0, 1.0))
            for knob, tx in ((self.thresh_knob, 418), (self.attack_knob, 502), (self.release_knob, 586)):
                knob.move(tx + slide_dx, 64)
                knob.setRevealAlpha(alpha)
                if knob.isHidden():
                    knob.show()

        self.auto_expand_btn.move(cur_w - 94, 40)
        self.minimize_btn.move(cur_w - 62, 40)
        self.drag_grip_btn.move(cur_w - 30, 40)
        self.bottom_bar.setGeometry(18, 170, cur_w - 36, 28)

        if reposition_window and not self._is_minimized and self._hud_mode == "eq":
            self.expanded_container.setGeometry(0, 0, cur_w, self.HUD_EXPANDED_H)
            cx = self._anim_anchor_cx if self._anim_anchor_cx is not None else (self.x() + self.width() * 0.5)
            p = self.parent()
            pw = p.width() if p is not None else 1280
            new_x = int(round(cx - cur_w * 0.5))
            new_x = max(0, min(max(0, pw - cur_w), new_x))
            self.setFixedSize(cur_w, self.HUD_EXPANDED_H)
            self.move(new_x, self.y())
            if self._user_custom_pos is not None:
                self._user_custom_pos = QPoint(new_x, self.y())
            self.sync_bg_item()

    def _on_dyn_width_anim_step(self, val):
        self._apply_eq_dyn_layout(float(val), reposition_window=True)

    def _on_dyn_width_anim_finished(self):
        target_t = 1.0 if self._dyn_drawer_open else 0.0
        self._apply_eq_dyn_layout(target_t, reposition_window=(not self._is_minimized and self._hud_mode == "eq"))
        self._anim_anchor_cx = None

    def _set_dyn_drawer_open(self, open_dyn: bool, animate: bool = True):
        target_open = bool(open_dyn)
        target_t = 1.0 if target_open else 0.0
        if (
            target_open == self._dyn_drawer_open
            and abs(self._dyn_anim_t - target_t) < 1e-3
            and self._dyn_width_anim.state() != QVariantAnimation.State.Running
        ):
            return
        self._dyn_drawer_open = target_open
        if self._dyn_width_anim.state() == QVariantAnimation.State.Running:
            self._dyn_width_anim.stop()
        self._anim_anchor_cx = self.x() + self.width() * 0.5

        if not animate or self._is_minimized or self._hud_mode != "eq" or not self.isVisible():
            self._apply_eq_dyn_layout(target_t, reposition_window=(not self._is_minimized and self._hud_mode == "eq"))
            self._anim_anchor_cx = None
            return

        self._dyn_width_anim.setStartValue(float(self._dyn_anim_t))
        self._dyn_width_anim.setEndValue(float(target_t))
        self._dyn_width_anim.start()

    def set_help_preview_dyn(self, enabled: bool):
        self._help_force_dyn_open = bool(enabled)
        bands = list(self.eq_engine.bands)
        idx = self.overlay.selected_band_idx
        actual = False
        if 0 <= idx < len(bands):
            b = bands[idx]
            actual = bool(b.dyn_enabled)
        self._set_dyn_drawer_open(bool(self._help_force_dyn_open or actual), animate=True)

    def set_help_preview_slope(self, enabled: bool):
        self._help_force_slope_visible = bool(enabled)
        bands = list(self.eq_engine.bands)
        idx = self.overlay.selected_band_idx
        actual_slope = bool(0 <= idx < len(bands) and bands[idx].supports_slope)
        show_slope = bool(actual_slope or self._help_force_slope_visible)
        if hasattr(self, "slope_combo"):
            self.slope_combo.setVisible(show_slope)
            self._layout_eq_left_controls(show_slope)

    def _sync_hud_mode_visibility(self):
        if self._is_minimized:
            self.expanded_container.hide()
            if hasattr(self, "mb_container"):
                self.mb_container.hide()
            self.compact_container.setGeometry(0, 0, self.HUD_COMPACT_W, self.HUD_COMPACT_H)
            self.compact_container.show()
        else:
            self.compact_container.hide()
            if self._hud_mode == "mb" and hasattr(self, "mb_container"):
                self.expanded_container.hide()
                self.mb_container.setGeometry(0, 0, self.HUD_MB_W, self.HUD_EXPANDED_H)
                self.mb_container.show()
            else:
                if hasattr(self, "mb_container"):
                    self.mb_container.hide()
                cur_eq_w = int(round(self.HUD_EQ_BASE_W + self._dyn_anim_t * (self.HUD_EQ_DYN_W - self.HUD_EQ_BASE_W)))
                self.expanded_container.setGeometry(0, 0, cur_eq_w, self.HUD_EXPANDED_H)
                self.expanded_container.show()

    def set_multiband_processor(self, mb_proc: MultibandDynamicsProcessor | None):
        self.multiband_dyn = mb_proc
        self.refresh_mb_from_engine()
        if self._hud_mode == "eq":
            self.refresh_from_engine()

    def set_hud_mode(self, mode: str):
        m = "mb" if mode == "mb" else "eq"
        if m == self._hud_mode:
            return
        if self._dyn_width_anim.state() == QVariantAnimation.State.Running:
            self._dyn_width_anim.stop()
        self._anim_anchor_cx = None
        self._apply_eq_dyn_layout(1.0 if self._dyn_drawer_open else 0.0, reposition_window=False)

        self._hud_mode = m
        if not self._is_minimized:
            old_cx = self.x() + self.width() * 0.5
            new_w = self._target_expanded_width()
            p = self.parent()
            pw = p.width() if p is not None else 1280
            new_x = max(0, min(max(0, pw - new_w), int(round(old_cx - new_w * 0.5))))
            self.setFixedSize(new_w, self.HUD_EXPANDED_H)
            self.move(new_x, self.y())
            if self._user_custom_pos is not None:
                self._user_custom_pos = QPoint(new_x, self.y())

        self._sync_hud_mode_visibility()
        if m == "mb":
            self.refresh_mb_from_engine()
        else:
            self.refresh_from_engine()
        self.sync_bg_item()

    def set_minimized(self, minimized: bool):
        if self._is_minimized == bool(minimized):
            return
        if self._dyn_width_anim.state() == QVariantAnimation.State.Running:
            self._dyn_width_anim.stop()
        self._anim_anchor_cx = None
        self._apply_eq_dyn_layout(1.0 if self._dyn_drawer_open else 0.0, reposition_window=False)

        old_x, old_y = self.x(), self.y()
        old_w, old_h = self.width(), self.height()
        bottom_y = old_y + old_h
        center_x = old_x + old_w * 0.5

        self._is_minimized = bool(minimized)
        if self._is_minimized:
            new_w, new_h = self.HUD_COMPACT_W, self.HUD_COMPACT_H
        else:
            new_w, new_h = self._target_expanded_width(), self.HUD_EXPANDED_H

        p = self.parent()
        pw = p.width() if p is not None else 1280
        ph = p.height() if p is not None else 720
        new_x = max(0, min(max(0, pw - new_w), int(round(center_x - new_w * 0.5))))
        new_y = max(44, min(max(44, ph - new_h), bottom_y - new_h))

        self.setFixedSize(new_w, new_h)
        self.move(new_x, new_y)
        if self._user_custom_pos is not None:
            self._user_custom_pos = QPoint(new_x, new_y)

        self._sync_hud_mode_visibility()
        if self._hud_mode == "mb":
            self.refresh_mb_from_engine()
        else:
            self.refresh_from_engine()
        self.sync_bg_item()

    def toggle_auto_expand(self, enabled: bool):
        self._auto_expand = bool(enabled)
        self.auto_expand_btn.setChecked(self._auto_expand)
        self.compact_auto_btn.setChecked(self._auto_expand)
        if hasattr(self, "mb_auto_expand_btn"):
            self.mb_auto_expand_btn.setChecked(self._auto_expand)
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
        if getattr(self, "_mb_char_popup", None) is not None and self._mb_char_popup.isVisible():
            return True
        if getattr(self, "_mb_mode_menu", None) is not None and self._mb_mode_menu.isVisible():
            return True
        focus_w = QApplication.focusWidget()
        if focus_w is not None and (self.isAncestorOf(focus_w) or isinstance(focus_w, QListView)):
            return True
        for cb in self.findChildren(QComboBox):
            if cb.view() is not None and cb.view().isVisible():
                return True
        for dd in self.findChildren(SvgDropdownButton):
            if dd.is_menu_visible():
                return True
        return False

    def _on_leave_timeout(self):
        if not self._is_minimized and self._auto_expand:
            if not self._has_active_popup():
                cursor_pos = self.mapFromGlobal(QCursor.pos())
                if not self.rect().contains(cursor_pos):
                    self.set_minimized(True)

    def eventFilter(self, obj, event):
        if obj in (
            self,
            self.expanded_container,
            self.compact_container,
            getattr(self, "mb_container", None),
            getattr(self, "drag_grip_btn", None),
            getattr(self, "compact_drag_grip", None),
            getattr(self, "mb_drag_grip_btn", None),
        ):
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
        # 1. CONTENEDOR EXPANDIDO EQ (520x208 base -> 768x208 con DYN activo)
        # =====================================================================
        self.expanded_container = QWidget(self)
        self.expanded_container.setGeometry(0, 0, self.HUD_EQ_BASE_W, self.HUD_EXPANDED_H)

        # Botón Minimizar, Auto-Expand y Agarre de Arrastre en la esquina superior derecha
        self.drag_grip_btn = QLabel("⠿", self.expanded_container)
        self.drag_grip_btn.setFixedSize(22, 24)
        self.drag_grip_btn.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drag_grip_btn.move(self.HUD_EQ_BASE_W - 30, 40)
        self.drag_grip_btn.setCursor(Qt.CursorShape.SizeAllCursor)
        self.drag_grip_btn.setToolTip("Arrastrar HUD")
        self.drag_grip_btn.setStyleSheet("color: #727B8E; font-size: 15px; font-weight: bold;")

        self.minimize_btn = SvgActionButton(
            "minimize.svg", "—", self.expanded_container, active_tint_color="#00E5FF"
        )
        self.minimize_btn.setFixedSize(28, 24)
        self.minimize_btn.move(self.HUD_EQ_BASE_W - 62, 40)
        self.minimize_btn.setToolTip("Minimizar HUD (Barra Compacta)")
        self.minimize_btn.clicked.connect(lambda: self.set_minimized(True))

        self.auto_expand_btn = SvgToggleButton(
            "auto_expand.svg", "Auto", self.expanded_container, active_tint_color="#00E5FF"
        )
        self.auto_expand_btn.setFixedSize(28, 24)
        self.auto_expand_btn.move(self.HUD_EQ_BASE_W - 94, 40)
        self.auto_expand_btn.setToolTip("Auto-expandir al pasar el cursor (vuelve a minimizar tras 0.4s)")
        self.auto_expand_btn.clicked.connect(self.toggle_auto_expand)

        # Columna Izquierda Gráfica SVG: Power (on.svg/off.svg) + Color Swatch, Curva de Filtro SVG y Pendiente SVG
        self.band_enable_btn = PowerToggleButton(self.expanded_container)
        self.band_enable_btn.setFixedSize(28, 28)
        self.band_enable_btn.setToolTip("Activar / Desactivar banda seleccionada (ON / OFF)")
        self.band_enable_btn.move(16, 52)
        self.band_enable_btn.clicked.connect(self._on_band_enable_toggled)

        self.color_swatch = ColorSwatchButton("#00E5FF", self.expanded_container)
        self.color_swatch.move(50, 55)
        self.color_swatch.colorSelected.connect(self._on_swatch_color_selected)

        self.type_combo = SvgDropdownButton(
            self.expanded_container,
            icon_map=FILTER_TYPE_SVGS,
            default_svg="bell.svg",
            show_text=False,
            recolor_to_accent=True,
        )
        self.type_combo.addItems(FILTER_TYPES)
        self.type_combo.setFixedSize(64, 34)
        self.type_combo.move(16, 88)
        self.type_combo.setToolTip("Forma de curva del filtro (Clic para desplegar menú de curvas)")
        self.type_combo.currentTextChanged.connect(self._on_type_or_slope_changed)

        self.slope_combo = SvgDropdownButton(
            self.expanded_container,
            default_svg="slope.svg",
            show_text=True,
            recolor_to_accent=True,
            short_text_func=lambda txt, d: f"{d} dB" if d is not None else txt,
        )
        self.slope_combo.setFixedSize(76, 26)
        self.slope_combo.move(16, 126)
        self.slope_combo.setToolTip("Pendiente del filtro (Slope dB/oct)")
        for s in SLOPE_OPTIONS:
            self.slope_combo.addItem(f"{s} dB/oct", s)
        self.slope_combo.currentIndexChanged.connect(self._on_type_or_slope_changed)

        # Bloque Central Fijo: FREQ | GAIN (joroba) | Q
        self.freq_knob = ProQKnob(
            title="FREQ", min_val=10.0, max_val=24000.0, default_val=1000.0,
            fmt_func=self._fmt_freq, log_scale=True, bipolar=False,
            accent_color="#00E5FF", diameter=60, min_label="10", max_label="24k",
            parent=self.expanded_container
        )
        self.freq_knob.move(102, 62)
        self.freq_knob.valueChanged.connect(self._on_knob_param_changed)

        self.gain_dyn_knob = GainDynProQKnob(diameter=84, parent=self.expanded_container)
        self.gain_dyn_knob.move(202, 10)
        self.gain_dyn_knob.valueChanged.connect(self._on_knob_param_changed)
        self.gain_dyn_knob.dynRangeChanged.connect(self._on_dyn_range_knob_changed)

        self.q_knob = ProQKnob(
            title="Q", min_val=0.10, max_val=24.0, default_val=1.00,
            fmt_func=lambda v: f"{v:.2f}", log_scale=True, bipolar=False,
            accent_color="#00E5FF", diameter=60, min_label="0.1", max_label="24",
            parent=self.expanded_container
        )
        self.q_knob.move(326, 62)
        self.q_knob.valueChanged.connect(self._on_knob_param_changed)

        # Bloque Dinámico Desplegable (oculto hasta activar DYN): THR, A, R + OUT deslizante
        self.thresh_knob = ProQKnob(
            title="THR", min_val=-70.0, max_val=0.0, default_val=-24.0,
            fmt_func=lambda v: f"{v:.0f} dB", log_scale=False, bipolar=False,
            accent_color="#00E5FF", diameter=56, reserve_outer_ring=True,
            min_label="-70", max_label="0", parent=self.expanded_container
        )
        self.thresh_knob.set_meter_mode(True, min_db=-70.0, max_db=0.0)
        self.thresh_knob.set_threshold_marker(-24.0, is_normalized=False)
        self.thresh_knob.move(418, 64)
        self.thresh_knob.valueChanged.connect(self._on_threshold_knob_changed)

        self.attack_knob = ProQKnob(
            title="A", min_val=0.5, max_val=500.0, default_val=15.0,
            fmt_func=lambda v: f"{v:.1f} ms" if v < 10 else f"{v:.0f} ms",
            log_scale=True, bipolar=False, accent_color="#00E5FF", diameter=56,
            min_label="0.5", max_label="500", parent=self.expanded_container
        )
        self.attack_knob.move(502, 64)
        self.attack_knob.valueChanged.connect(self._on_knob_param_changed)

        self.release_knob = ProQKnob(
            title="R", min_val=5.0, max_val=2000.0, default_val=120.0,
            fmt_func=lambda v: f"{v:.0f} ms",
            log_scale=True, bipolar=False, accent_color="#00E5FF", diameter=56,
            min_label="5", max_label="2k", parent=self.expanded_container
        )
        self.release_knob.move(586, 64)
        self.release_knob.valueChanged.connect(self._on_knob_param_changed)

        self.out_gain_knob = ProQKnob(
            title="OUT", min_val=-24.0, max_val=24.0, default_val=0.0,
            fmt_func=lambda v: f"{v:+.1f} dB", log_scale=False, bipolar=True,
            accent_color="#FFD54F", diameter=56,
            min_label="-24", max_label="+24", parent=self.expanded_container
        )
        self.out_gain_knob.move(420, 64)
        self.out_gain_knob.valueChanged.connect(self._on_global_changed)

        # Fila Inferior Gráfica SVG
        self.bottom_bar = QWidget(self.expanded_container)
        self.bottom_bar.setGeometry(18, 170, self.HUD_EQ_BASE_W - 36, 28)
        bot_row = QHBoxLayout(self.bottom_bar)
        bot_row.setContentsMargins(0, 0, 0, 0)
        bot_row.setSpacing(6)

        self.prev_band_btn = SvgActionButton("previous.svg", "‹", self.bottom_bar, active_tint_color="#35C6E8")
        self.prev_band_btn.setFixedSize(28, 24)
        self.prev_band_btn.setToolTip("Nodo / Banda anterior (EQ y COMP)")
        self.prev_band_btn.clicked.connect(lambda: self._step_unified_band(-1))
        bot_row.addWidget(self.prev_band_btn)

        self.band_combo = SvgDropdownButton(
            self.bottom_bar,
            default_svg="bell.svg",
            show_text=True,
            recolor_to_accent=True,
        )
        self.band_combo.setFixedSize(102, 24)
        self.band_combo.setToolTip("Seleccionar nodo EQ o banda COMP activa")
        self.band_combo.currentIndexChanged.connect(lambda _idx, c=self.band_combo: self._on_unified_combo_changed(c))
        self.band_combo.itemReselected.connect(lambda _idx, c=self.band_combo: self._on_unified_combo_changed(c))
        self.band_combo.actionTriggered.connect(self._on_unified_combo_action)
        bot_row.addWidget(self.band_combo)

        self.next_band_btn = SvgActionButton("next.svg", "›", self.bottom_bar, active_tint_color="#35C6E8")
        self.next_band_btn.setFixedSize(28, 24)
        self.next_band_btn.setToolTip("Nodo / Banda siguiente (EQ y COMP)")
        self.next_band_btn.clicked.connect(lambda: self._step_unified_band(1))
        bot_row.addWidget(self.next_band_btn)

        self.add_band_btn = SvgActionButton("add.svg", "+", self.bottom_bar, active_tint_color="#55D6A3")
        self.add_band_btn.setFixedSize(28, 24)
        self.add_band_btn.setToolTip("Añadir nueva banda EQ")
        self.add_band_btn.clicked.connect(self._on_add_band)
        bot_row.addWidget(self.add_band_btn)

        self.del_band_btn = SvgActionButton("delete.svg", "✕", self.bottom_bar, active_tint_color="#FF6B78")
        self.del_band_btn.setFixedSize(28, 24)
        self.del_band_btn.setToolTip("Eliminar banda seleccionada (Supr)")
        self.del_band_btn.clicked.connect(self._on_delete_band)
        bot_row.addWidget(self.del_band_btn)

        self.add_mb_hud_btn = SvgActionButton("multiband.svg", "+ MB", self.bottom_bar, active_tint_color="#FFA94D", semantic_idle=True)
        self.add_mb_hud_btn.setFixedSize(36, 24)
        self.add_mb_hud_btn.setToolTip("Crear banda de Compresor / Limitador Multibanda en el espectro FFT")
        self.add_mb_hud_btn.clicked.connect(self._on_hud_add_mb_clicked)
        bot_row.addWidget(self.add_mb_hud_btn)

        bot_row.addStretch(1)

        self.channel_mode_combo = SvgDropdownButton(
            self.bottom_bar,
            icon_map=CHANNEL_MODE_SVGS,
            default_svg="stereo.svg",
            show_text=False,
            recolor_to_accent=False,
        )
        self.channel_mode_combo.setFixedSize(40, 24)
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

        self.target_out_combo = SvgDropdownButton(
            self.bottom_bar,
            icon_map=TARGET_OUT_SVGS,
            default_svg="ab.svg",
            show_text=False,
            recolor_to_accent=False,
        )
        self.target_out_combo.setFixedSize(40, 24)
        self.target_out_combo.setToolTip("Asignar Banda a Bus de Salida: A+B (Ambas), Out A (1-2) o Out B (3-4)")
        for to_val, lbl in (
            ("A+B", "A+B (Ambas)"),
            ("A", "Out A (1-2)"),
            ("B", "Out B (3-4)"),
        ):
            self.target_out_combo.addItem(lbl, to_val)
        self.target_out_combo.currentIndexChanged.connect(self._on_channel_or_output_changed)
        bot_row.addWidget(self.target_out_combo)

        self.dyn_toggle_btn = SvgToggleButton("dyn.svg", "DYN", self.bottom_bar, pill_radius=6.0)
        self.dyn_toggle_btn.setFixedSize(36, 24)
        self.dyn_toggle_btn.setToolTip("Dynamic EQ (Se ilumina al activar compresión/expansión dinámica)")
        self.dyn_toggle_btn.clicked.connect(self._on_dyn_toggle_clicked)
        bot_row.addWidget(self.dyn_toggle_btn)

        self.band_solo_btn = SvgToggleButton("isolate.svg", "Isolate", self.bottom_bar, active_tint_color="#00E5FF", pill_radius=6.0)
        self.band_solo_btn.setFixedSize(36, 24)
        self.band_solo_btn.setToolTip("Isolate / Escucha aislada (Se ilumina al aislar el rango de esta banda)")
        self.band_solo_btn.clicked.connect(self._on_band_solo_toggled)
        bot_row.addWidget(self.band_solo_btn)

        # =====================================================================
        # 2. CONTENEDOR COMPACTO (720x40px, anclado abajo)
        # =====================================================================
        self.compact_container = QWidget(self)
        self.compact_container.setGeometry(0, 0, self.HUD_COMPACT_W, self.HUD_COMPACT_H)
        self.compact_container.hide()
        self.compact_widget = self.compact_container

        comp_row = QHBoxLayout(self.compact_container)
        comp_row.setContentsMargins(10, 5, 10, 5)
        comp_row.setSpacing(6)

        self.compact_drag_grip = QLabel("⠿", self.compact_container)
        self.compact_drag_grip.setFixedSize(18, 26)
        self.compact_drag_grip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.compact_drag_grip.setCursor(Qt.CursorShape.SizeAllCursor)
        self.compact_drag_grip.setToolTip("Arrastrar HUD")
        self.compact_drag_grip.setStyleSheet("color: #727B8E; font-size: 15px; font-weight: bold;")
        comp_row.addWidget(self.compact_drag_grip)

        # 1. Power ON/OFF + Color Swatch
        self.compact_power_btn = PowerToggleButton(self.compact_container)
        self.compact_power_btn.setFixedSize(24, 24)
        self.compact_power_btn.setToolTip("Activar / Desactivar banda")
        self.compact_power_btn.clicked.connect(self._on_compact_power_toggled)
        comp_row.addWidget(self.compact_power_btn)

        self.compact_color_swatch = ColorSwatchButton("#00E5FF", self.compact_container)
        self.compact_color_swatch.colorSelected.connect(self._on_compact_color_selected)
        comp_row.addWidget(self.compact_color_swatch)

        # 2. Navegador < Band X > + y ✕
        self.compact_prev_btn = SvgActionButton("previous.svg", "‹", self.compact_container)
        self.compact_prev_btn.setFixedSize(24, 22)
        self.compact_prev_btn.setToolTip("Nodo / Banda anterior (EQ y COMP)")
        self.compact_prev_btn.clicked.connect(lambda: self._step_unified_band(-1))
        comp_row.addWidget(self.compact_prev_btn)

        self.compact_band_combo = SvgDropdownButton(
            self.compact_container,
            default_svg="bell.svg",
            show_text=True,
            recolor_to_accent=True,
        )
        self.compact_band_combo.setFixedSize(98, 22)
        self.compact_band_combo.setToolTip("Seleccionar nodo EQ o banda COMP activa")
        self.compact_band_combo.currentIndexChanged.connect(lambda _idx, c=self.compact_band_combo: self._on_unified_combo_changed(c))
        self.compact_band_combo.itemReselected.connect(lambda _idx, c=self.compact_band_combo: self._on_unified_combo_changed(c))
        self.compact_band_combo.actionTriggered.connect(self._on_unified_combo_action)
        comp_row.addWidget(self.compact_band_combo)

        self.compact_next_btn = SvgActionButton("next.svg", "›", self.compact_container)
        self.compact_next_btn.setFixedSize(24, 22)
        self.compact_next_btn.setToolTip("Nodo / Banda siguiente (EQ y COMP)")
        self.compact_next_btn.clicked.connect(lambda: self._step_unified_band(1))
        comp_row.addWidget(self.compact_next_btn)

        self.compact_add_btn = SvgActionButton("add.svg", "+", self.compact_container, active_tint_color="#55D6A3")
        self.compact_add_btn.setFixedSize(24, 22)
        self.compact_add_btn.setToolTip("Añadir nueva banda")
        self.compact_add_btn.clicked.connect(self._on_compact_add_clicked)
        comp_row.addWidget(self.compact_add_btn)

        self.compact_del_btn = SvgActionButton("delete.svg", "✕", self.compact_container, active_tint_color="#FF6B78")
        self.compact_del_btn.setFixedSize(24, 22)
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
        self.compact_target_out = SvgDropdownButton(
            self.compact_container,
            icon_map=TARGET_OUT_SVGS,
            default_svg="ab.svg",
            show_text=False,
            recolor_to_accent=False,
        )
        self.compact_target_out.setFixedSize(36, 22)
        self.compact_target_out.setToolTip("Asignar a Bus de Salida")
        for to_val, lbl in (
            ("A+B", "A+B"),
            ("A", "Out A"),
            ("B", "Out B"),
        ):
            self.compact_target_out.addItem(lbl, to_val)
        self.compact_target_out.currentIndexChanged.connect(self._on_compact_output_changed)
        comp_row.addWidget(self.compact_target_out)

        # 4. Botón Isolate + Auto + Expandir (agrandados para clic cómodo)
        self.compact_solo_btn = SvgToggleButton("isolate.svg", "ISO", self.compact_container, pill_radius=6.0, active_tint_color="#00E5FF")
        self.compact_solo_btn.setFixedSize(34, 24)
        self.compact_solo_btn.setToolTip("Aislar banda (Solo Isolate)")
        self.compact_solo_btn.clicked.connect(self._on_compact_solo_toggled)
        comp_row.addWidget(self.compact_solo_btn)

        self.compact_auto_btn = SvgToggleButton(
            "auto_expand.svg", "Auto", self.compact_container, active_tint_color="#00E5FF"
        )
        self.compact_auto_btn.setFixedSize(28, 26)
        self.compact_auto_btn.setToolTip("Auto-expandir al pasar el cursor")
        self.compact_auto_btn.clicked.connect(self.toggle_auto_expand)
        comp_row.addWidget(self.compact_auto_btn)

        self.compact_expand_btn = SvgActionButton(
            "expand.svg", "▲", self.compact_container, active_tint_color="#00E5FF"
        )
        self.compact_expand_btn.setFixedSize(28, 26)
        self.compact_expand_btn.setToolTip("Expandir panel HUD completo")
        self.compact_expand_btn.clicked.connect(lambda: self.set_minimized(False))
        comp_row.addWidget(self.compact_expand_btn)

        # =====================================================================
        # 3. CONTENEDOR MULTIBANDA PRO-MB SIMÉTRICO (768x208px)
        # =====================================================================
        self.mb_container = QWidget(self)
        self.mb_container.setGeometry(0, 0, self.HUD_MB_W, self.HUD_EXPANDED_H)
        self.mb_container.hide()

        self.mb_drag_grip_btn = QLabel("⠿", self.mb_container)
        self.mb_drag_grip_btn.setFixedSize(22, 24)
        self.mb_drag_grip_btn.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mb_drag_grip_btn.move(self.HUD_MB_W - 30, 40)
        self.mb_drag_grip_btn.setCursor(Qt.CursorShape.SizeAllCursor)
        self.mb_drag_grip_btn.setToolTip("Arrastrar HUD")
        self.mb_drag_grip_btn.setStyleSheet("color: #727B8E; font-size: 15px; font-weight: bold;")

        self.mb_minimize_btn = SvgActionButton(
            "minimize.svg", "—", self.mb_container, active_tint_color="#00E5FF"
        )
        self.mb_minimize_btn.setFixedSize(28, 24)
        self.mb_minimize_btn.move(self.HUD_MB_W - 62, 40)
        self.mb_minimize_btn.setToolTip("Minimizar HUD (Barra Compacta)")
        self.mb_minimize_btn.clicked.connect(lambda: self.set_minimized(True))

        self.mb_auto_expand_btn = SvgToggleButton(
            "auto_expand.svg", "Auto", self.mb_container, active_tint_color="#00E5FF"
        )
        self.mb_auto_expand_btn.setFixedSize(28, 24)
        self.mb_auto_expand_btn.move(self.HUD_MB_W - 94, 40)
        self.mb_auto_expand_btn.setToolTip("Auto-expandir al pasar el cursor (vuelve a minimizar tras 0.4s)")
        self.mb_auto_expand_btn.clicked.connect(self.toggle_auto_expand)

        # Columna Izquierda Gráfica SVG: Power (on/off) + Color Swatch centrados verticalmente
        self.mb_enable_btn = PowerToggleButton(self.mb_container)
        self.mb_enable_btn.setFixedSize(28, 28)
        self.mb_enable_btn.setToolTip("Activar / Desactivar banda multibanda seleccionada (ON / OFF)")
        self.mb_enable_btn.move(16, 86)
        self.mb_enable_btn.clicked.connect(self._on_mb_enable_toggled)

        self.mb_color_swatch = ColorSwatchButton("#00E5FF", self.mb_container)
        self.mb_color_swatch.move(50, 89)
        self.mb_color_swatch.setToolTip("Elegir color de la banda multibanda")
        self.mb_color_swatch.colorSelected.connect(self._on_mb_color_selected)

        # Ala Izquierda Simétrica (3 perillas uniformes centradas verticalmente): RANGE, RATIO, KNEE
        self.mb_range_knob = ProQKnob(
            title="RANGE", min_val=-30.0, max_val=12.0, default_val=-12.0,
            fmt_func=lambda v: f"{v:+.1f} dB", log_scale=False, bipolar=True,
            accent_color="#00E5FF", diameter=56, min_label="-30", max_label="+12",
            parent=self.mb_container,
        )
        self.mb_range_knob.move(84, 64)
        self.mb_range_knob.valueChanged.connect(self._on_mb_knob_changed)
        self.mb_range_arc = self.mb_range_knob

        self.mb_ratio_knob = ProQKnob(
            title="RATIO", min_val=1.0, max_val=20.0, default_val=4.0,
            fmt_func=lambda v: "∞:1" if self._mb_ratio_is_limit else f"{v:.1f}:1",
            log_scale=True, bipolar=False,
            accent_color="#00E5FF", diameter=56, min_label="1:1", max_label="20:1",
            parent=self.mb_container,
        )
        self.mb_ratio_knob.move(176, 64)
        self.mb_ratio_knob.valueChanged.connect(self._on_mb_knob_changed)

        self.mb_knee_knob = ProQKnob(
            title="KNEE", min_val=0.0, max_val=24.0, default_val=6.0,
            fmt_func=lambda v: f"{v:.1f} dB", log_scale=False, bipolar=False,
            accent_color="#00E5FF", diameter=56, min_label="0", max_label="24",
            parent=self.mb_container,
        )
        self.mb_knee_knob.move(268, 64)
        self.mb_knee_knob.valueChanged.connect(self._on_mb_knob_changed)
        self.mb_knee_arc = self.mb_knee_knob

        # Perilla Hero Central encastrada bajo la joroba superior (cx=418.0, GR curvo debajo)
        self.mb_thr_knob = ProQKnob(
            title="THRESHOLD", min_val=-70.0, max_val=0.0, default_val=-18.0,
            fmt_func=lambda v: f"{v:.1f} dB", log_scale=False, bipolar=False,
            accent_color="#00E5FF", diameter=74, reserve_outer_ring=True,
            min_label="-70", max_label="0", center_subtext="GR 0.0 dB",
            center_subtext_color="#FFB74D", parent=self.mb_container,
        )
        self.mb_thr_knob.set_meter_mode(True, min_db=-70.0, max_db=0.0)
        self.mb_thr_knob.set_threshold_marker(-18.0, is_normalized=False)
        self.mb_thr_knob.set_gr_mode(False)
        self.mb_thr_knob.move(366, 8)
        self.mb_thr_knob.valueChanged.connect(self._on_mb_knob_changed)

        # Selectores Curva Modo y Resistencia Character centrados debajo de THRESHOLD
        self.mb_mode_btn = MBModeSvgButton(self.mb_container, active_tint_color="#00E5FF")
        self.mb_mode_btn.setFixedSize(44, 28)
        self.mb_mode_btn.move(369, 128)
        self.mb_mode_btn.setToolTip("Curva de dinámica de banda (Clic para elegir Compress / Expand / Limit)")
        self.mb_mode_btn.clicked.connect(self._show_mb_mode_menu)
        self.mb_mode_btn.raise_()

        self.mb_char_btn = CharacterResistorButton(self.mb_container)
        self.mb_char_btn.setFixedSize(44, 28)
        self.mb_char_btn.move(423, 128)
        self.mb_char_btn.clicked.connect(self._toggle_mb_character_popup)
        self.mb_char_btn.raise_()

        self.mb_gr_lbl = QLabel("GR 0.0 dB", self.mb_container)
        self.mb_gr_lbl.hide()

        # Ala Derecha Simétrica (3 perillas uniformes centradas verticalmente): ATTACK, RELEASE, OUT
        self.mb_attack_knob = ProQKnob(
            title="ATTACK", min_val=0.1, max_val=200.0, default_val=10.0,
            fmt_func=lambda v: f"{v:.1f} ms" if v < 10.0 else f"{v:.0f} ms",
            log_scale=True, bipolar=False,
            accent_color="#00E5FF", diameter=56, min_label="0.1", max_label="200",
            parent=self.mb_container,
        )
        self.mb_attack_knob.move(482, 64)
        self.mb_attack_knob.valueChanged.connect(self._on_mb_knob_changed)

        self.mb_release_knob = ProQKnob(
            title="RELEASE", min_val=5.0, max_val=2000.0, default_val=100.0,
            fmt_func=lambda v: f"{v:.0f} ms",
            log_scale=True, bipolar=False,
            accent_color="#00E5FF", diameter=56, min_label="5", max_label="2k",
            parent=self.mb_container,
        )
        self.mb_release_knob.move(574, 64)
        self.mb_release_knob.valueChanged.connect(self._on_mb_knob_changed)

        self.mb_output_knob = ProQKnob(
            title="OUT", min_val=-24.0, max_val=24.0, default_val=0.0,
            fmt_func=lambda v: f"{v:+.1f} dB",
            log_scale=False, bipolar=True,
            accent_color="#FFD54F", diameter=56, min_label="-24", max_label="+24",
            parent=self.mb_container,
        )
        self.mb_output_knob.move(666, 64)
        self.mb_output_knob.valueChanged.connect(self._on_mb_knob_changed)

        # Barra Inferior MB Gráfica SVG (Solo reubicado aquí igual que en el HUD de EQ)
        self.mb_bottom_bar = QWidget(self.mb_container)
        self.mb_bottom_bar.setGeometry(18, 170, self.HUD_MB_W - 36, 28)
        mb_bot_row = QHBoxLayout(self.mb_bottom_bar)
        mb_bot_row.setContentsMargins(0, 0, 0, 0)
        mb_bot_row.setSpacing(6)

        self.mb_prev_btn = SvgActionButton("previous.svg", "‹", self.mb_bottom_bar, active_tint_color="#35C6E8")
        self.mb_prev_btn.setFixedSize(28, 24)
        self.mb_prev_btn.setToolTip("Nodo / Banda anterior (EQ y COMP)")
        self.mb_prev_btn.clicked.connect(lambda: self._step_unified_band(-1))
        mb_bot_row.addWidget(self.mb_prev_btn)

        self.mb_band_combo = SvgDropdownButton(
            self.mb_bottom_bar,
            default_svg="comp_mode.svg",
            show_text=True,
            recolor_to_accent=True,
        )
        self.mb_band_combo.setFixedSize(102, 24)
        self.mb_band_combo.setToolTip("Seleccionar nodo EQ o banda COMP activa")
        self.mb_band_combo.currentIndexChanged.connect(lambda _idx, c=self.mb_band_combo: self._on_unified_combo_changed(c))
        self.mb_band_combo.itemReselected.connect(lambda _idx, c=self.mb_band_combo: self._on_unified_combo_changed(c))
        self.mb_band_combo.actionTriggered.connect(self._on_unified_combo_action)
        mb_bot_row.addWidget(self.mb_band_combo)

        self.mb_next_btn = SvgActionButton("next.svg", "›", self.mb_bottom_bar, active_tint_color="#35C6E8")
        self.mb_next_btn.setFixedSize(28, 24)
        self.mb_next_btn.setToolTip("Nodo / Banda siguiente (EQ y COMP)")
        self.mb_next_btn.clicked.connect(lambda: self._step_unified_band(1))
        mb_bot_row.addWidget(self.mb_next_btn)

        self.mb_add_btn = SvgActionButton("add.svg", "+", self.mb_bottom_bar, active_tint_color="#55D6A3")
        self.mb_add_btn.setFixedSize(28, 24)
        self.mb_add_btn.setToolTip("Crear banda multibanda en el espectro FFT (o clic en coordenada X)")
        self.mb_add_btn.clicked.connect(self._on_hud_add_mb_clicked)
        mb_bot_row.addWidget(self.mb_add_btn)

        self.mb_del_btn = SvgActionButton("delete.svg", "✕", self.mb_bottom_bar, active_tint_color="#FF6B78")
        self.mb_del_btn.setFixedSize(28, 24)
        self.mb_del_btn.setToolTip("Eliminar banda MB seleccionada (Supr)")
        self.mb_del_btn.clicked.connect(self._on_mb_delete_band)
        mb_bot_row.addWidget(self.mb_del_btn)

        self.mb_freq_lbl = QLabel("—", self.mb_bottom_bar)
        self.mb_freq_lbl.setFixedSize(104, 24)
        self.mb_freq_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mb_freq_lbl.setToolTip("Rango de frecuencias [f_low – f_high] de la banda LR4")
        self.mb_freq_lbl.setStyleSheet(
            "QLabel { color: #AEB6C4; font-size: 10px; font-weight: bold; "
            "background: transparent; border: none; padding: 1px 4px; }"
        )
        mb_bot_row.addWidget(self.mb_freq_lbl)

        mb_bot_row.addStretch(1)

        self.mb_bus_combo = SvgDropdownButton(
            self.mb_bottom_bar,
            icon_map=TARGET_OUT_SVGS,
            default_svg="ab.svg",
            show_text=False,
            recolor_to_accent=False,
        )
        self.mb_bus_combo.setFixedSize(40, 24)
        self.mb_bus_combo.setToolTip("Asignar Banda MB a Bus de Salida: A+B (Ambas), Out A o Out B")
        for to_val, lbl in (
            ("A+B", "A+B (Ambas)"),
            ("A", "Out A (1-2)"),
            ("B", "Out B (3-4)"),
        ):
            self.mb_bus_combo.addItem(lbl, to_val)
        self.mb_bus_combo.currentIndexChanged.connect(self._on_mb_bus_changed)
        mb_bot_row.addWidget(self.mb_bus_combo)

        self.mb_solo_btn = SvgToggleButton("isolate.svg", "S", self.mb_bottom_bar, pill_radius=6.0, active_tint_color="#FFD54F")
        self.mb_solo_btn.setFixedSize(36, 24)
        self.mb_solo_btn.setToolTip("Solo / Isolate en banda multibanda (Se ilumina al aislar esta región LR4)")
        self.mb_solo_btn.clicked.connect(self._on_mb_solo_toggled)
        mb_bot_row.addWidget(self.mb_solo_btn)

        self.mb_global_bypass_btn = SvgToggleButton("bypass.svg", "MB Bypass", self.mb_bottom_bar, pill_radius=6.0, active_tint_color="#FFA94D")
        self.mb_global_bypass_btn.setFixedSize(36, 24)
        self.mb_global_bypass_btn.setToolTip("Bypass global del procesador Multiband Dynamics (Se ilumina al activar Bypass)")
        self.mb_global_bypass_btn.clicked.connect(self._on_mb_global_bypass_toggled)
        mb_bot_row.addWidget(self.mb_global_bypass_btn)

        # Estado inicial compacto de EQ (DYN oculto hasta activarse)
        self._apply_eq_dyn_layout(0.0, reposition_window=False)

    @staticmethod
    def _fmt_mb_short(v: float) -> str:
        return f"{v / 1000.0:.1f}k" if v >= 1000.0 else f"{int(round(v))}"

    def _gain_center_x(self) -> float:
        if self._hud_mode == "mb" and hasattr(self, "mb_thr_knob"):
            return float(self.mb_thr_knob.x() + self.mb_thr_knob.width() * 0.5)
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
            arch_half_w = min(104.0, max(68.0, arch_cx - (x_left + r_corner + 8.0)))
            ax1 = max(x_left + r_corner + 6.0, arch_cx - arch_half_w)
            ax2 = min(x_right - r_corner - 6.0, arch_cx + arch_half_w)
            h_a = arch_half_w * 0.365
            h_b = arch_half_w * 0.404

            path.moveTo(x_left + r_corner, y_top)
            path.lineTo(ax1, y_top)
            path.cubicTo(
                QPointF(ax1 + h_a, y_top),
                QPointF(arch_cx - h_b, y_peak),
                QPointF(arch_cx, y_peak),
            )
            path.cubicTo(
                QPointF(arch_cx + h_b, y_peak),
                QPointF(ax2 - h_a, y_top),
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
            painter.drawLine(QPointF(18.0, h - 44.0), QPointF(w - 18.0, h - 44.0))

    def _update_nav_arrow_colors(self, hex_color: str):
        c = str(hex_color) if hex_color else "#35C6E8"
        for attr in (
            "prev_band_btn", "next_band_btn",
            "compact_prev_btn", "compact_next_btn",
            "mb_prev_btn", "mb_next_btn",
        ):
            btn = getattr(self, attr, None)
            if btn is not None and hasattr(btn, "setActiveTintColor"):
                btn.setActiveTintColor(c)

    def _notify_active_accent(self, hex_color: str):
        norm = str(hex_color or "")
        if self._last_emitted_accent_hex != norm:
            self._last_emitted_accent_hex = norm
            self.active_accent_changed.emit(norm)

    def _build_unified_items(self) -> list[tuple[str, int, str, str, str]]:
        """Retorna lista unificada de nodos EQ y bandas COMP: (kind, idx, label, svg_name, color_hex)."""
        items: list[tuple[str, int, str, str, str]] = []
        with self.eq_engine.lock:
            eq_bands = list(self.eq_engine.bands)
        for i, b in enumerate(eq_bands):
            dyn_tag = " [D]" if b.is_dynamic_active else ""
            lbl = f"Band {i + 1}{dyn_tag}"
            svg = FILTER_TYPE_SVGS.get(b.filter_type, "bell.svg")
            items.append(("eq", i, lbl, svg, str(b.color)))

        mb = self.multiband_dyn
        if mb is not None:
            with mb.lock:
                mb_bands = list(mb.bands)
            for i, b_m in enumerate(mb_bands):
                m_tag = "LIM" if b_m.mode == "Limit" else ("EXP" if b_m.mode == "Expand" else "CMP")
                lbl = f"MB {i + 1} [{m_tag}]"
                svg = MB_MODE_SVGS.get(b_m.mode, "comp_mode.svg")
                items.append(("mb", i, lbl, svg, str(b_m.color)))
        return items

    def _sync_unified_combos(self):
        items = self._build_unified_items()
        eq_items = [it for it in items if it[0] == "eq"]
        mb_items = [it for it in items if it[0] == "mb"]

        icon_map: dict[str, str] = {}
        accent_map: dict[str, str] = {}
        for kind, idx, lbl, svg, col in items:
            d_key = str((kind, idx))
            icon_map[d_key] = svg
            icon_map[lbl] = svg
            accent_map[d_key] = col
            accent_map[lbl] = col

        cur_kind = "mb" if self._hud_mode == "mb" else "eq"
        cur_idx = getattr(self.overlay, "selected_mb_idx", -1) if cur_kind == "mb" else self.overlay.selected_band_idx
        target_data = (cur_kind, cur_idx)
        active_col = accent_map.get(str(target_data), self._active_band_color.name() if self._active_band_color else "#00E5FF")

        mb = self.multiband_dyn
        can_add_mb = (mb is not None) and (len(mb_items) < MAX_MB_BANDS)
        items_sig = (tuple(items), bool(can_add_mb))
        rebuild_items = getattr(self, "_last_unified_items_sig", None) != items_sig
        if rebuild_items:
            self._last_unified_items_sig = items_sig

        for combo_attr in ("band_combo", "compact_band_combo", "mb_band_combo"):
            combo = getattr(self, combo_attr, None)
            if combo is None:
                continue
            combo.blockSignals(True)
            if rebuild_items:
                combo.clear()
                for kind, idx, lbl, _svg, _col in eq_items:
                    combo.addItem(lbl, (kind, idx))
                if eq_items and mb_items:
                    combo.addSeparator()
                for kind, idx, lbl, _svg, _col in mb_items:
                    combo.addItem(lbl, (kind, idx))
                combo.addActionItem("+ Nueva Banda EQ", "add_eq", "add.svg", "#55D6A3")
                if can_add_mb:
                    combo.addActionItem("+ Nueva Banda COMP", "add_mb", "multiband.svg", "#FFA94D")
                combo.set_icon_map(icon_map)
                combo.set_item_accent_map(accent_map)
            combo.setAccentColor(active_col)
            sel_pos = combo.findData(target_data)
            if sel_pos >= 0:
                combo.setCurrentIndex(sel_pos)
            combo.setEnabled(True)
            combo.blockSignals(False)

        has_any = len(items) > 0
        for arrow_attr in (
            "prev_band_btn", "next_band_btn",
            "compact_prev_btn", "compact_next_btn",
            "mb_prev_btn", "mb_next_btn",
        ):
            btn = getattr(self, arrow_attr, None)
            if btn is not None:
                btn.setEnabled(has_any)

    def _step_unified_band(self, delta: int):
        items = self._build_unified_items()
        if not items:
            return
        cur_kind = "mb" if self._hud_mode == "mb" else "eq"
        cur_idx = getattr(self.overlay, "selected_mb_idx", -1) if cur_kind == "mb" else self.overlay.selected_band_idx
        pos = -1
        for k, (kind, idx, _, _, _) in enumerate(items):
            if kind == cur_kind and idx == cur_idx:
                pos = k
                break
        if pos < 0:
            next_pos = 0 if delta >= 0 else (len(items) - 1)
        else:
            next_pos = (pos + delta) % len(items)
        t_kind, t_idx = items[next_pos][0], items[next_pos][1]
        if t_kind == "mb":
            self.select_mb_band(t_idx, from_hud=True)
        else:
            self.select_band(t_idx, from_hud=True)

    def _on_unified_combo_changed(self, combo: SvgDropdownButton):
        if self._updating_ui or combo is None:
            return
        data = combo.currentData()
        if not isinstance(data, tuple) or len(data) != 2:
            return
        kind, idx = data
        if kind == "mb":
            self.select_mb_band(int(idx), from_hud=True)
        elif kind == "eq":
            self.select_band(int(idx), from_hud=True)

    def _on_unified_combo_action(self, action_id):
        if action_id == "add_eq":
            self._on_add_band()
        elif action_id == "add_mb":
            self._on_hud_add_mb_clicked()

    def _step_band(self, delta: int):
        self._step_unified_band(delta)

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
        self.sync_selected_band_parameters()
        self.parameters_changed.emit()

    def select_band(self, index: int, from_hud: bool = False):
        if hasattr(self.overlay, "_suppress_eq_callout"):
            self.overlay._suppress_eq_callout = False
        self.overlay.selected_band_idx = index
        if from_hud:
            self.ensure_layer_requested.emit("EQ")
        if self._hud_mode != "eq":
            self.set_hud_mode("eq")
        else:
            self._sync_hud_mode_visibility()
            self.refresh_from_engine()
        self.overlay.update_eq_curves(force=True)
        if hasattr(self.overlay, "node_callout") and getattr(self.overlay, "canvas_mode", "BOTH") in ("EQ", "BOTH"):
            self.overlay.node_callout.show_callout()

    def select_mb_band(self, index: int, from_hud: bool = False):
        if hasattr(self.overlay, "_suppress_eq_callout"):
            self.overlay._suppress_eq_callout = True
        if hasattr(self.overlay, "node_callout") and self.overlay.node_callout.isVisible():
            self.overlay.node_callout.hide()
        self.overlay.selected_mb_idx = index
        if from_hud:
            self.ensure_layer_requested.emit("MB")
        if self._hud_mode != "mb":
            self.set_hud_mode("mb")
        else:
            self._sync_hud_mode_visibility()
            self.refresh_mb_from_engine()
        self.overlay.update_eq_curves(force=True)

    def _on_overlay_mb_modified(self):
        self.sync_selected_mb_band_parameters()
        self.parameters_changed.emit()

    def sync_selected_mb_band_parameters(self):
        """Fast path for MB band/node drag: updates only sliders, knobs and labels without rebuilding combos."""
        mb = self.multiband_dyn
        if mb is None or not hasattr(self, "mb_container"):
            self.refresh_mb_from_engine()
            return
        with mb.lock:
            bands = list(mb.bands)
            mb_bypass = bool(mb.bypass)
        idx = getattr(self.overlay, "selected_mb_idx", -1)
        if not (0 <= idx < len(bands)):
            self.refresh_mb_from_engine()
            return
        b = bands[idx]
        cur_struct = (
            len(bands), idx, b.mode, b.enabled, b.bypass, b.solo,
            b.bus_target, b.color, b.character_mode, mb_bypass,
        )
        if getattr(self, "_last_mb_synced_struct", None) != cur_struct:
            self.refresh_mb_from_engine()
            return
        self._updating_ui = True
        try:
            self.mb_freq_lbl.setText(f"{self._fmt_mb_short(b.f_low)}–{self._fmt_mb_short(b.f_high)} Hz")
            self.mb_range_knob.setValue(b.range_db)
            self.mb_thr_knob.setValue(b.threshold_db)
            self.mb_thr_knob.set_threshold_marker(b.threshold_db, is_normalized=False)
            self.mb_ratio_knob.setValue(b.ratio)
            self.mb_attack_knob.setValue(b.attack_ms)
            self.mb_release_knob.setValue(b.release_ms)
            self.mb_knee_knob.setValue(b.knee_db)
            self.mb_output_knob.setValue(b.output_gain_db)
            if self._hud_mode == "mb" and hasattr(self, "compact_readout"):
                mb_txt = (
                    f"MB{idx + 1} {b.mode[:4].upper()} | "
                    f"{self._fmt_mb_short(b.f_low)}–{self._fmt_mb_short(b.f_high)} Hz | "
                    f"THR {b.threshold_db:.0f} dB"
                )
                self._last_compact_live_txt = mb_txt
                self.compact_readout.setText(mb_txt)
        finally:
            self._updating_ui = False

    def sync_selected_band_parameters(self):
        """Fast path for node drag: updates only knob values and compact readout without rebuilding combos or QSS."""
        bands = list(self.eq_engine.bands)
        idx = self.overlay.selected_band_idx
        if not (0 <= idx < len(bands)):
            self.refresh_from_engine()
            return
        b = bands[idx]
        is_iso = bool(b.solo or self.eq_engine.isolated_band_index == idx)
        cur_struct = (
            len(bands), idx, b.filter_type, b.slope, b.enabled, b.dyn_enabled,
            b.is_dynamic_active, b.channel_mode, b.target_output, b.color, is_iso,
        )
        if getattr(self, "_last_synced_struct", None) != cur_struct:
            self.refresh_from_engine()
            return
        self._updating_ui = True
        try:
            self.freq_knob.setValue(b.freq)
            self.gain_dyn_knob.setValue(b.gain_db)
            self.gain_dyn_knob.setDynRangeValue(b.dyn_range_db)
            self.q_knob.setValue(b.q)
            self.thresh_knob.setValue(b.threshold_db)
            self.thresh_knob.set_threshold_marker(b.threshold_db, is_normalized=False)
            self.attack_knob.setValue(b.attack_ms)
            self.release_knob.setValue(b.release_ms)
            self.out_gain_knob.setValue(self.eq_engine.output_gain_db)
            if hasattr(self, "compact_readout"):
                gain_txt = f"{b.gain_db:+.1f} dB" if b.supports_gain else b.filter_type
                self.compact_readout.setText(f"{self._fmt_freq(b.freq)}  |  {gain_txt}  |  Q {b.q:.2f}")
        finally:
            self._updating_ui = False

    def refresh_from_engine(self):
        self._updating_ui = True
        try:
            bands = list(self.eq_engine.bands)
            idx = self.overlay.selected_band_idx
            if idx < 0 and bands:
                idx = 0
                self.overlay.selected_band_idx = 0

            has_band = 0 <= idx < len(bands)
            eq_widgets = [
                self.del_band_btn, self.band_enable_btn,
                self.color_swatch, self.band_solo_btn,
                self.channel_mode_combo, self.target_out_combo,
                self.type_combo, self.slope_combo, self.freq_knob,
                self.gain_dyn_knob, self.thresh_knob, self.q_knob,
                self.dyn_toggle_btn, self.attack_knob, self.release_knob,
            ]
            if self._hud_mode == "eq":
                eq_widgets.extend([
                    getattr(self, "compact_del_btn", None),
                    getattr(self, "compact_power_btn", None),
                    getattr(self, "compact_color_swatch", None),
                    getattr(self, "compact_solo_btn", None),
                    getattr(self, "compact_target_out", None),
                ])
            for w in eq_widgets:
                if w is not None:
                    w.setEnabled(has_band)
            if self._hud_mode == "eq" and hasattr(self, "compact_add_btn"):
                self.compact_add_btn.setEnabled(True)

            self.freq_knob.max_val = max(24000.0, self.eq_engine.sr * 0.495)

            if has_band:
                b = bands[idx]
                if self._hud_mode == "eq":
                    self._active_band_color = QColor(b.color)
                    self._apply_hud_stylesheet(self._active_band_color)
                    self._update_nav_arrow_colors(b.color)
                    self._notify_active_accent(b.color)

                self.color_swatch.setColor(b.color)
                self.band_enable_btn.setAccentColor(b.color)
                self.band_enable_btn.setChecked(b.enabled)
                is_iso = bool(b.solo or self.eq_engine.isolated_band_index == idx)
                self.band_solo_btn.setChecked(is_iso)
                self.band_solo_btn.setActiveTintColor(b.color)

                if self._hud_mode == "eq":
                    if hasattr(self, "compact_color_swatch"):
                        self.compact_color_swatch.setColor(b.color)
                    if hasattr(self, "compact_power_btn"):
                        self.compact_power_btn.setAccentColor(b.color)
                        self.compact_power_btn.setChecked(b.enabled)
                    if hasattr(self, "compact_solo_btn"):
                        self.compact_solo_btn.setChecked(is_iso)
                        self.compact_solo_btn.setActiveTintColor(b.color)

                self.type_combo.setAccentColor(b.color)
                self.type_combo.setCurrentText(b.filter_type)
                self.slope_combo.setAccentColor(b.color)
                self.channel_mode_combo.setAccentColor(b.color)
                self.target_out_combo.setAccentColor(b.color)

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
                    if self._hud_mode == "eq" and hasattr(self, "compact_target_out"):
                        self.compact_target_out.blockSignals(True)
                        c_to = self.compact_target_out.findData(b.target_output)
                        if c_to >= 0:
                            self.compact_target_out.setCurrentIndex(c_to)
                        self.compact_target_out.blockSignals(False)

                slope_idx = self.slope_combo.findData(b.slope)
                if slope_idx >= 0:
                    self.slope_combo.setCurrentIndex(slope_idx)
                show_slope = bool(b.supports_slope or getattr(self, "_help_force_slope_visible", False))
                self.slope_combo.setVisible(show_slope)
                self.slope_combo.setEnabled(b.supports_slope)
                self._layout_eq_left_controls(show_slope)
                self.gain_dyn_knob.setEnabled(b.supports_gain)

                self.freq_knob.setAccentColor(b.color)
                self.gain_dyn_knob.setAccentColor(b.color)
                self.thresh_knob.setAccentColor(b.color)
                self.q_knob.setAccentColor(b.color)
                self.attack_knob.setAccentColor(b.color)
                self.release_knob.setAccentColor(b.color)

                self.freq_knob.setValue(b.freq)
                self.gain_dyn_knob.setValue(b.gain_db)
                self.gain_dyn_knob.setDynRangeValue(b.dyn_range_db)
                self.gain_dyn_knob.setDynEnabled(b.dyn_enabled)
                self.q_knob.setValue(b.q)

                if self._hud_mode == "eq" and hasattr(self, "compact_readout"):
                    gain_txt = f"{b.gain_db:+.1f} dB" if b.supports_gain else b.filter_type
                    self.compact_readout.setText(f"{self._fmt_freq(b.freq)}  |  {gain_txt}  |  Q {b.q:.2f}")
                    if getattr(self, "_last_compact_readout_color", None) != b.color:
                        self._last_compact_readout_color = b.color
                        self.compact_readout.setStyleSheet(f"color: {b.color}; font-size: 11px; font-weight: bold; padding: 0 4px;")

                self.dyn_toggle_btn.setChecked(b.dyn_enabled)
                self.dyn_toggle_btn.setActiveTintColor(b.color)

                dyn_active = bool(b.dyn_enabled)
                self.thresh_knob.setEnabled(dyn_active)
                self.attack_knob.setEnabled(dyn_active)
                self.release_knob.setEnabled(dyn_active)
                self._set_dyn_drawer_open(bool(dyn_active or getattr(self, "_help_force_dyn_open", False)), animate=True)

                self.thresh_knob.setValue(b.threshold_db)
                self.thresh_knob.set_meter_mode(dyn_active, -70.0, 0.0)
                self.thresh_knob.set_threshold_marker(b.threshold_db, is_normalized=False)
                if not dyn_active:
                    self.thresh_knob.set_meter_level(-70.0, is_normalized=False)
                    self.gain_dyn_knob.setLiveDynOffset(0.0)
                self.attack_knob.setValue(b.attack_ms)
                self.release_knob.setValue(b.release_ms)

                self._last_synced_struct = (
                    len(bands), idx, b.filter_type, b.slope, b.enabled, b.dyn_enabled,
                    b.is_dynamic_active, b.channel_mode, b.target_output, b.color, is_iso,
                )
            else:
                self._last_synced_struct = None
                show_slope = bool(getattr(self, "_help_force_slope_visible", False))
                self.slope_combo.setVisible(show_slope)
                self._layout_eq_left_controls(show_slope)
                self._set_dyn_drawer_open(bool(getattr(self, "_help_force_dyn_open", False)), animate=True)
                self.thresh_knob.set_meter_mode(False)
                self.gain_dyn_knob.setLiveDynOffset(0.0)
                if self._hud_mode == "eq":
                    self._active_band_color = QColor("#00E5FF")
                    self._apply_hud_stylesheet(self._active_band_color)
                    self._update_nav_arrow_colors("#35C6E8")
                    self._notify_active_accent("")
                    if hasattr(self, "compact_readout"):
                        self.compact_readout.setText("Sin bandas activas")
                        if getattr(self, "_last_compact_readout_color", None) != "#727B8E":
                            self._last_compact_readout_color = "#727B8E"
                            self.compact_readout.setStyleSheet("color: #727B8E; font-size: 11px; font-weight: bold; padding: 0 4px;")

            self.out_gain_knob.setValue(self.eq_engine.output_gain_db)
            self._sync_unified_combos()
            if self._bg_gl_item is None:
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
        self.select_band(new_idx, from_hud=True)
        self.parameters_changed.emit()

    def _on_delete_band(self):
        if self._hud_mode == "mb":
            self._on_mb_delete_band()
            return
        idx = self.overlay.selected_band_idx
        if 0 <= idx < len(self.eq_engine.bands):
            self.eq_engine.remove_band(idx)
            if (
                len(self.eq_engine.bands) == 0
                and self.multiband_dyn is not None
                and len(self.multiband_dyn.bands) > 0
            ):
                self.overlay.selected_band_idx = -1
                self.select_mb_band(0, from_hud=True)
            else:
                new_idx = min(idx, len(self.eq_engine.bands) - 1)
                self.select_band(new_idx)
            self.parameters_changed.emit()

    def refresh_mb_from_engine(self):
        if not hasattr(self, "mb_container"):
            return
        mb = self.multiband_dyn
        if mb is None:
            return
        self._updating_ui = True
        try:
            with mb.lock:
                bands = list(mb.bands)
                mb_bypass = bool(mb.bypass)
            idx = getattr(self.overlay, "selected_mb_idx", -1)
            if idx < 0 and bands:
                idx = 0
                self.overlay.selected_mb_idx = 0
            elif idx >= len(bands):
                idx = len(bands) - 1
                self.overlay.selected_mb_idx = idx

            self.mb_global_bypass_btn.setChecked(mb_bypass)

            has_band = 0 <= idx < len(bands)
            mb_widgets = [
                self.mb_enable_btn, self.mb_color_swatch, self.mb_solo_btn,
                self.mb_mode_btn, self.mb_char_btn,
                self.mb_range_arc, self.mb_thr_knob, self.mb_ratio_knob,
                self.mb_attack_knob, self.mb_release_knob, self.mb_knee_arc,
                self.mb_output_knob, self.mb_del_btn, self.mb_bus_combo,
            ]
            if self._hud_mode == "mb":
                mb_widgets.extend([
                    getattr(self, "compact_del_btn", None),
                    getattr(self, "compact_power_btn", None),
                    getattr(self, "compact_color_swatch", None),
                    getattr(self, "compact_solo_btn", None),
                    getattr(self, "compact_target_out", None),
                ])
            for w in mb_widgets:
                if w is not None:
                    w.setEnabled(has_band)
            self.mb_add_btn.setEnabled(len(bands) < MAX_MB_BANDS)
            if self._hud_mode == "mb" and hasattr(self, "compact_add_btn"):
                self.compact_add_btn.setEnabled(len(bands) < MAX_MB_BANDS)

            if has_band:
                b = bands[idx]
                if self._hud_mode == "mb":
                    self._active_band_color = QColor(b.color)
                    self._apply_hud_stylesheet(self._active_band_color)
                    self._update_nav_arrow_colors(b.color)
                    self._notify_active_accent(b.color)

                self.mb_enable_btn.setAccentColor(b.color)
                self.mb_enable_btn.setChecked(b.enabled)
                self.mb_color_swatch.setColor(b.color)
                self.mb_solo_btn.setChecked(b.solo)
                self.mb_solo_btn.setActiveTintColor(b.color)

                self.mb_mode_btn.setText(f"{b.mode} ▾")
                self.mb_mode_btn.setActiveTintColor(b.color)
                self.mb_bus_combo.setAccentColor(b.color)
                self.mb_char_btn.setCharacterState(b.character_mode)
                if self._mb_char_popup is not None and self._mb_char_popup.isVisible():
                    self._mb_char_popup.set_state(b.character_mode, b.character_drive, b.character_post)

                self.mb_freq_lbl.setText(f"{self._fmt_mb_short(b.f_low)}–{self._fmt_mb_short(b.f_high)} Hz")

                for kn in (
                    self.mb_range_arc, self.mb_thr_knob, self.mb_ratio_knob,
                    self.mb_attack_knob, self.mb_release_knob, self.mb_knee_arc,
                ):
                    kn.setAccentColor(b.color)

                self._mb_ratio_is_limit = (b.mode == "Limit")
                self.mb_ratio_knob.setEnabled(not self._mb_ratio_is_limit)
                self.mb_ratio_knob.setValue(b.ratio)
                self.mb_ratio_knob._refresh_labels()

                self.mb_range_knob.setValue(b.range_db)
                self.mb_thr_knob.setValue(b.threshold_db)
                self.mb_thr_knob.set_threshold_marker(b.threshold_db, is_normalized=False)
                self.mb_thr_knob.set_gr_mode(False)
                self.mb_attack_knob.setValue(b.attack_ms)
                self.mb_release_knob.setValue(b.release_ms)
                self.mb_knee_knob.setValue(b.knee_db)
                self.mb_output_knob.setValue(b.output_gain_db)

                b_idx = self.mb_bus_combo.findData(b.bus_target)
                if b_idx >= 0:
                    self.mb_bus_combo.blockSignals(True)
                    self.mb_bus_combo.setCurrentIndex(b_idx)
                    self.mb_bus_combo.blockSignals(False)

                if self._hud_mode == "mb":
                    if hasattr(self, "compact_color_swatch"):
                        self.compact_color_swatch.setColor(b.color)
                    if hasattr(self, "compact_power_btn"):
                        self.compact_power_btn.setAccentColor(b.color)
                        self.compact_power_btn.setChecked(b.enabled)
                    if hasattr(self, "compact_solo_btn"):
                        self.compact_solo_btn.setChecked(b.solo)
                        self.compact_solo_btn.setActiveTintColor(b.color)
                    if hasattr(self, "compact_target_out"):
                        self.compact_target_out.blockSignals(True)
                        c_mb_to = self.compact_target_out.findData(b.bus_target)
                        if c_mb_to >= 0:
                            self.compact_target_out.setCurrentIndex(c_mb_to)
                        self.compact_target_out.blockSignals(False)
                    if hasattr(self, "compact_readout"):
                        mb_txt = (
                            f"MB{idx + 1} {b.mode[:4].upper()} | "
                            f"{self._fmt_mb_short(b.f_low)}–{self._fmt_mb_short(b.f_high)} Hz | "
                            f"THR {b.threshold_db:.0f} dB"
                        )
                        self._last_compact_live_txt = mb_txt
                        self.compact_readout.setText(mb_txt)
                        if getattr(self, "_last_compact_readout_color", None) != b.color:
                            self._last_compact_readout_color = b.color
                            self.compact_readout.setStyleSheet(
                                f"color: {b.color}; font-size: 11px; font-weight: bold; padding: 0 4px;"
                            )
                self._last_mb_synced_struct = (
                    len(bands), idx, b.mode, b.enabled, b.bypass, b.solo,
                    b.bus_target, b.color, b.character_mode, mb_bypass,
                )
            else:
                self._last_mb_synced_struct = None
                self._mb_ratio_is_limit = False
                self.mb_mode_btn.setText("Compress ▾")
                self.mb_char_btn.setCharacterState("Off")
                self.mb_freq_lbl.setText("Sin bandas MB")
                self.mb_thr_knob.set_meter_level(-70.0, is_normalized=False)
                self.mb_thr_knob.set_gr_level(0.0)
                self._last_mb_gr_txt = "GR 0.0 dB"
                self.mb_thr_knob.setCenterSubtext("GR 0.0 dB", "#828B9C")
                if self._hud_mode == "mb":
                    self._active_band_color = QColor("#FF6D00")
                    self._apply_hud_stylesheet(self._active_band_color)
                    self._update_nav_arrow_colors("#FF6D00")
                    self._notify_active_accent("")
                    if hasattr(self, "compact_readout"):
                        self._last_compact_live_txt = "Sin bandas MB"
                        self.compact_readout.setText("Sin bandas MB")
                        if getattr(self, "_last_compact_readout_color", None) != "#727B8E":
                            self._last_compact_readout_color = "#727B8E"
                            self.compact_readout.setStyleSheet(
                                "color: #727B8E; font-size: 11px; font-weight: bold; padding: 0 4px;"
                            )

            self._sync_unified_combos()
            if self._hud_mode == "mb":
                if self._bg_gl_item is None:
                    self.update()
                self.sync_bg_item()
        finally:
            self._updating_ui = False

    def _on_hud_add_mb_clicked(self):
        if self.multiband_dyn is None:
            return
        if len(self.multiband_dyn.bands) >= MAX_MB_BANDS:
            self.select_mb_band(max(0, len(self.multiband_dyn.bands) - 1), from_hud=True)
            return
        if getattr(self.overlay, "_mb_placement_armed", False):
            self.overlay.cancel_mb_placement()
            self._on_mb_add_band()
            return
        if getattr(self.overlay, "canvas_mode", "BOTH") not in ("MB", "BOTH"):
            self.ensure_layer_requested.emit("MB")
        self.overlay.start_mb_placement()

    def _on_mb_color_selected(self, hex_color: str):
        if self._updating_ui or self.multiband_dyn is None:
            return
        idx = getattr(self.overlay, "selected_mb_idx", -1)
        if idx >= 0:
            self.overlay._set_mb_color(idx, hex_color)
            self.refresh_mb_from_engine()
            self.parameters_changed.emit()

    def _show_mb_mode_menu(self):
        if self.multiband_dyn is None:
            return
        idx = getattr(self.overlay, "selected_mb_idx", -1)
        with self.multiband_dyn.lock:
            if not (0 <= idx < len(self.multiband_dyn.bands)):
                return
            b = self.multiband_dyn.bands[idx]
        menu = HUDPopupMenu(self.mb_mode_btn, accent_color=b.color, corner_radius=8.0)
        self._mb_mode_menu = menu
        labels = {
            "Compress": "Compress (Compresor)",
            "Expand": "Expand (Expansor / Gate)",
            "Limit": "Limit (Limitador ∞:1)",
        }
        for m_opt in MB_MODES:
            ic = _make_svg_qicon(
                MB_MODE_SVGS.get(m_opt, "comp_mode.svg"), 20,
                accent_hex=b.color, recolor_primary=(b.mode == m_opt)
            )
            act = menu.addAction(ic, labels.get(m_opt, m_opt))
            act.setCheckable(True)
            act.setChecked(b.mode == m_opt)
            act.triggered.connect(lambda _=False, mv=m_opt: self._on_mb_mode_clicked(mv))
        pos = self.mb_mode_btn.mapToGlobal(QPoint(0, self.mb_mode_btn.height() + 2))
        menu.exec(pos)

    def _toggle_mb_character_popup(self):
        if self._mb_char_popup is not None and self._mb_char_popup.isVisible():
            self._mb_char_popup.close()
            return
        if self.multiband_dyn is None:
            return
        idx = getattr(self.overlay, "selected_mb_idx", -1)
        with self.multiband_dyn.lock:
            if not (0 <= idx < len(self.multiband_dyn.bands)):
                return
            b = self.multiband_dyn.bands[idx]
            st = (b.character_mode, b.character_drive, b.character_post)
        if self._mb_char_popup is None:
            self._mb_char_popup = MBCharacterPopup(parent=self)
            self._mb_char_popup.characterChanged.connect(self._on_mb_character_changed)
        self._mb_char_popup.set_state(*st)
        bx = self.mb_char_btn.mapToGlobal(QPoint(self.mb_char_btn.width() // 2, self.mb_char_btn.height() + 4))
        self._mb_char_popup.move(bx.x() - self._mb_char_popup.width() // 2, bx.y())
        self._mb_char_popup.show()
        self._mb_char_popup.raise_()

    def _on_mb_character_changed(self, char_mode: str, drive: float, post: bool):
        if self._updating_ui or self.multiband_dyn is None:
            return
        idx = getattr(self.overlay, "selected_mb_idx", -1)
        with self.multiband_dyn.lock:
            if 0 <= idx < len(self.multiband_dyn.bands):
                b = self.multiband_dyn.bands[idx]
                b.character_mode = char_mode
                b.character_drive = float(drive)
                b.character_post = bool(post)
                self.multiband_dyn.mark_dirty()
        self.mb_char_btn.setCharacterState(char_mode)
        self.overlay.update_eq_curves(force=True)
        self.parameters_changed.emit()

    def _on_mb_knob_changed(self, _val: float = 0.0):
        if self._updating_ui or self.multiband_dyn is None:
            return
        idx = getattr(self.overlay, "selected_mb_idx", -1)
        with self.multiband_dyn.lock:
            if 0 <= idx < len(self.multiband_dyn.bands):
                b = self.multiband_dyn.bands[idx]
                b.range_db = self.mb_range_arc.value()
                b.threshold_db = self.mb_thr_knob.value()
                if not self._mb_ratio_is_limit:
                    b.ratio = self.mb_ratio_knob.value()
                b.attack_ms = self.mb_attack_knob.value()
                b.release_ms = self.mb_release_knob.value()
                b.knee_db = self.mb_knee_arc.value()
                b.output_gain_db = self.mb_output_knob.value()
                self.multiband_dyn.mark_dirty()
                self.mb_thr_knob.set_threshold_marker(b.threshold_db, is_normalized=False)
        self.overlay.update_eq_curves(force=True)
        self.parameters_changed.emit()

    def _on_mb_mode_clicked(self, mode_val: str):
        if self._updating_ui or self.multiband_dyn is None:
            return
        idx = getattr(self.overlay, "selected_mb_idx", -1)
        with self.multiband_dyn.lock:
            if 0 <= idx < len(self.multiband_dyn.bands):
                b = self.multiband_dyn.bands[idx]
                b.mode = mode_val
                if not getattr(b, "custom_color", False):
                    b.color = MB_MODE_COLORS.get(mode_val, b.color)
                if mode_val == "Limit":
                    if b.attack_ms > 2.0:
                        b.attack_ms = 0.5
                self.multiband_dyn.mark_dirty()
        self.refresh_mb_from_engine()
        self.overlay.update_eq_curves(force=True)
        self.parameters_changed.emit()

    def _on_mb_enable_toggled(self, checked: bool):
        if self._updating_ui or self.multiband_dyn is None:
            return
        idx = getattr(self.overlay, "selected_mb_idx", -1)
        with self.multiband_dyn.lock:
            if 0 <= idx < len(self.multiband_dyn.bands):
                self.multiband_dyn.bands[idx].enabled = bool(checked)
                self.multiband_dyn.mark_dirty()
        self.refresh_mb_from_engine()
        self.overlay.update_eq_curves(force=True)
        self.parameters_changed.emit()

    def _on_mb_bypass_toggled(self, checked: bool):
        if self._updating_ui or self.multiband_dyn is None:
            return
        idx = getattr(self.overlay, "selected_mb_idx", -1)
        with self.multiband_dyn.lock:
            if 0 <= idx < len(self.multiband_dyn.bands):
                self.multiband_dyn.bands[idx].bypass = bool(checked)
                self.multiband_dyn.mark_dirty()
        self.refresh_mb_from_engine()
        self.overlay.update_eq_curves(force=True)
        self.parameters_changed.emit()

    def _on_mb_solo_toggled(self, checked: bool):
        if self._updating_ui or self.multiband_dyn is None:
            return
        idx = getattr(self.overlay, "selected_mb_idx", -1)
        self.multiband_dyn.set_solo_band(idx, bool(checked))
        self.refresh_mb_from_engine()
        self.overlay.update_eq_curves(force=True)
        self.parameters_changed.emit()

    def _on_mb_bus_changed(self):
        if self._updating_ui or self.multiband_dyn is None:
            return
        idx = getattr(self.overlay, "selected_mb_idx", -1)
        to_val = str(self.mb_bus_combo.currentData())
        with self.multiband_dyn.lock:
            if 0 <= idx < len(self.multiband_dyn.bands) and to_val in TARGET_OUTPUTS:
                self.multiband_dyn.bands[idx].bus_target = to_val
                self.multiband_dyn.mark_dirty()
        self.refresh_mb_from_engine()
        self.overlay.update_eq_curves(force=True)
        self.parameters_changed.emit()

    def _on_mb_global_bypass_toggled(self, checked: bool):
        if self._updating_ui or self.multiband_dyn is None:
            return
        with self.multiband_dyn.lock:
            self.multiband_dyn.bypass = bool(checked)
            self.multiband_dyn.mark_dirty()
        self.refresh_mb_from_engine()
        self.overlay.update_eq_curves(force=True)
        self.parameters_changed.emit()

    def _on_mb_add_band(self):
        if self.multiband_dyn is None:
            return
        created = self.multiband_dyn.add_band()
        if created is None:
            return
        try:
            new_idx = self.multiband_dyn.bands.index(created)
        except ValueError:
            new_idx = len(self.multiband_dyn.bands) - 1
        self.select_mb_band(new_idx, from_hud=True)
        self.parameters_changed.emit()

    def _on_mb_delete_band(self):
        if self.multiband_dyn is None:
            return
        idx = getattr(self.overlay, "selected_mb_idx", -1)
        if 0 <= idx < len(self.multiband_dyn.bands):
            self.multiband_dyn.remove_band(idx)
            if len(self.multiband_dyn.bands) == 0 and len(self.eq_engine.bands) > 0:
                self.overlay.selected_mb_idx = -1
                self.select_band(0, from_hud=True)
            else:
                new_idx = min(idx, len(self.multiband_dyn.bands) - 1)
                self.select_mb_band(new_idx)
            self.parameters_changed.emit()

    def _step_mb_band(self, delta: int):
        self._step_unified_band(delta)

    def _on_mb_band_combo_changed(self, index: int):
        if self._updating_ui or index < 0:
            return
        self.select_mb_band(index)

    def _on_compact_power_toggled(self, checked: bool):
        if self._hud_mode == "mb":
            self._on_mb_enable_toggled(checked)
        else:
            self._on_band_enable_toggled(checked)

    def _on_compact_color_selected(self, hex_color: str):
        if self._hud_mode == "mb":
            self._on_mb_color_selected(hex_color)
        else:
            self._on_swatch_color_selected(hex_color)

    def _on_compact_solo_toggled(self, checked: bool):
        if self._hud_mode == "mb":
            self._on_mb_solo_toggled(checked)
        else:
            self._on_band_solo_toggled(checked)

    def _on_compact_add_clicked(self):
        if self._hud_mode == "mb":
            self._on_hud_add_mb_clicked()
        else:
            self._on_add_band()

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
        to_val = str(self.compact_target_out.currentData())
        if self._hud_mode == "mb":
            if self.multiband_dyn is None:
                return
            idx = getattr(self.overlay, "selected_mb_idx", -1)
            with self.multiband_dyn.lock:
                if 0 <= idx < len(self.multiband_dyn.bands) and to_val in TARGET_OUTPUTS:
                    self.multiband_dyn.bands[idx].bus_target = to_val
                    self.multiband_dyn.mark_dirty()
            self.refresh_mb_from_engine()
            self.overlay.update_eq_curves(force=True)
            self.parameters_changed.emit()
            return
        idx = self.overlay.selected_band_idx
        with self.eq_engine.lock:
            if 0 <= idx < len(self.eq_engine.bands):
                b = self.eq_engine.bands[idx]
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

    def _on_threshold_knob_changed(self, val: float):
        if self._updating_ui:
            return
        idx = self.overlay.selected_band_idx
        with self.eq_engine.lock:
            if 0 <= idx < len(self.eq_engine.bands):
                b = self.eq_engine.bands[idx]
                b.threshold_db = float(val)
                b.update_coefficients()
                self.eq_engine.mark_dirty()
        self.thresh_knob.set_threshold_marker(val, is_normalized=False)
        self.overlay.update_eq_curves(force=True)
        self.parameters_changed.emit()

    _on_threshold_meter_changed = _on_threshold_knob_changed

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
        if self._hud_mode == "mb":
            mb = self.multiband_dyn
            if mb is None:
                return
            mb_bands = list(mb.bands)
            mb_idx = getattr(self.overlay, "selected_mb_idx", -1)
            if 0 <= mb_idx < len(mb_bands):
                b_mb = mb_bands[mb_idx]
                active = bool(b_mb.enabled and not b_mb.bypass and not mb.bypass)
                live_lvl = float(b_mb.live_level_db) if active else -70.0
                live_gr = float(b_mb.live_gr_db) if active else 0.0
                if not self._is_minimized and self.isVisible():
                    self.mb_thr_knob.set_meter_level(live_lvl, is_normalized=False)
                    self.mb_thr_knob.set_gr_level(min(0.0, live_gr) if active else 0.0)
                    if active and b_mb.mode == "Expand" and live_gr > 0.05:
                        gr_txt = f"EXP +{live_gr:.1f} dB"
                        gr_col = "#69F0AE"
                    elif active and abs(live_gr) >= 0.05:
                        gr_txt = f"GR {live_gr:.1f} dB"
                        gr_col = "#FFB74D"
                    else:
                        gr_txt = "GR 0.0 dB"
                        gr_col = "#828B9C"
                    if self._last_mb_gr_txt != gr_txt or getattr(self, "_last_mb_gr_col", None) != gr_col:
                        self._last_mb_gr_txt = gr_txt
                        self._last_mb_gr_col = gr_col
                        self.mb_thr_knob.setCenterSubtext(gr_txt, gr_col)
                elif self._is_minimized and hasattr(self, "compact_readout"):
                    new_txt = (
                        f"MB{mb_idx + 1} {b_mb.mode[:4].upper()} | "
                        f"{self._fmt_mb_short(b_mb.f_low)}–{self._fmt_mb_short(b_mb.f_high)} Hz | "
                        f"{live_gr:+.1f} dB"
                    )
                    if getattr(self, "_last_compact_live_txt", None) != new_txt or self.compact_readout.text() != new_txt:
                        self._last_compact_live_txt = new_txt
                        self.compact_readout.setText(new_txt)
            return

        idx = self.overlay.selected_band_idx
        bands = list(self.eq_engine.bands)
        if 0 <= idx < len(bands):
            b = bands[idx]
            if not self._is_minimized and self.isVisible():
                dyn_on = b.is_dynamic_active
                dyn_delta = float(getattr(b, "live_dyn_delta_db", b.current_dyn_offset_db))
                self.gain_dyn_knob.setLiveDynOffset(dyn_delta if dyn_on else 0.0)
                live_env = float(getattr(b, "live_env_db", b.env_db)) if dyn_on else -70.0
                self.thresh_knob.set_meter_level(live_env, is_normalized=False)
            if self._is_minimized and hasattr(self, "compact_readout"):
                gain_txt = f"{b.gain_db:+.1f} dB" if b.supports_gain else b.filter_type
                dyn_delta = float(getattr(b, "live_dyn_delta_db", b.current_dyn_offset_db))
                dyn_txt = f" (Δ {dyn_delta:+.1f})" if (b.is_dynamic_active and abs(dyn_delta) >= 0.1) else ""
                new_txt = f"{self._fmt_freq(b.freq)}  |  {gain_txt}{dyn_txt}  |  Q {b.q:.2f}"
                if getattr(self, "_last_compact_live_txt", None) != new_txt or self.compact_readout.text() != new_txt:
                    self._last_compact_live_txt = new_txt
                    self.compact_readout.setText(new_txt)


# Alias de compatibilidad
EQControlPanel = FloatingBandHUD


class LimiterGRMiniWidget(QWidget):
    """
    Indicador textual sin caja de Reducción de Ganancia (GR) en tiempo real del Limitador Master.
    Permanece en texto limpio grisáceo en reposo y pulsa en resplandor púrpura cuando el limitador actúa.
    """

    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(68, 22)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Reducción de Ganancia (GR) en tiempo real del Limitador Master (Clic para configurar)")
        self.gr_db = 0.0
        self._pulse_phase = 0.0
        self._glow_level = 0.0

    def set_gr(self, gr_db: float):
        new_gr = float(min(0.0, gr_db))
        prev_glow = self._glow_level
        if new_gr < -0.05:
            self._pulse_phase = (self._pulse_phase + 0.34) % (2.0 * math.pi)
            target_g = min(1.0, 0.45 + abs(new_gr) / 10.0)
            self._glow_level = max(self._glow_level, target_g)
            changed = True
        else:
            self._glow_level = max(0.0, self._glow_level - 0.12)
            changed = (abs(new_gr - self.gr_db) > 0.04) or (abs(self._glow_level - prev_glow) > 0.01)
        self.gr_db = new_gr
        if changed:
            self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        event.accept()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w = float(self.width())
        h = float(self.height())
        rect = QRectF(0.0, 0.0, w, h)

        font = QFont("Segoe UI", 8, QFont.Weight.Bold)
        painter.setFont(font)
        txt = f"GR {self.gr_db:.1f}" if self.gr_db < -0.05 else "GR 0.0"

        if self._glow_level > 0.02:
            p_mod = 0.65 + 0.35 * math.sin(self._pulse_phase)
            g_eff = float(np.clip(self._glow_level * p_mod, 0.0, 1.0))
            halo_c = QColor(213, 0, 249, int(115 * g_eff))
            painter.setPen(halo_c)
            for dx, dy in ((-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)):
                painter.drawText(rect.translated(dx, dy), Qt.AlignmentFlag.AlignCenter, txt)

            r = int(round(213 + (248 - 213) * p_mod))
            g = int(round(0 + (185 - 0) * p_mod))
            b = int(round(249 + (255 - 249) * p_mod))
            painter.setPen(QColor(r, g, b))
        else:
            painter.setPen(QColor("#7E889B"))

        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, txt)


class LimiterControlPopup(HUDPopupDialog):
    """
    Panel flotante no-modal de controles para el Limitador Master estilo 'Fruity Limiter'
    con botones gráficos SVG puros sin cajas, perillas ProQKnob (medidor GR integrado),
    pico de GR reiniciable y diseño translúcido HUD + Gaussian Blur en GPU con OpenGL.
    """

    def __init__(self, eq_engine, overlay_ref=None, app_ref=None, parent=None):
        super().__init__(parent=parent, accent_color="#D500F9", corner_radius=10.0)
        self.eq_engine = eq_engine
        self.overlay_ref = overlay_ref
        self.app_ref = app_ref
        self._peak_gr_db = 0.0
        self._last_gr_txt = None
        self._last_gr_active = None
        self._last_peak_txt = None
        self.setFixedWidth(392)

        self.setStyleSheet("""
            QWidget {
                background: transparent;
                color: #ECEFF4;
                font-family: "Segoe UI", sans-serif;
                font-size: 11px;
            }
            QPushButton#peakBtn {
                background: transparent;
                color: #9AA2B2;
                border: none;
                padding: 2px 4px;
                font-size: 10px;
                font-weight: bold;
            }
            QPushButton#peakBtn:hover {
                color: #F3B8FF;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 12)
        layout.setSpacing(8)

        # Encabezado con iconos SVG de Reset y Cerrar
        hdr_row = QHBoxLayout()
        hdr_row.setSpacing(6)
        title_lbl = QLabel("Limitador Master")
        title_lbl.setStyleSheet("font-weight: bold; font-size: 11px; color: #E5E9F0;")
        hdr_row.addWidget(title_lbl)
        hdr_row.addStretch(1)

        self.reset_btn = SvgActionButton("reset.svg", "Reset", self, active_tint_color="#D500F9")
        self.reset_btn.setFixedSize(22, 22)
        self.reset_btn.setToolTip("Restaurar parámetros por defecto del Limitador")
        self.reset_btn.clicked.connect(self._reset_defaults)
        hdr_row.addWidget(self.reset_btn)

        close_btn = SvgActionButton("close.svg", "✕", self, active_tint_color="#FF6B78")
        close_btn.setFixedSize(22, 22)
        close_btn.setToolTip("Cerrar panel")
        close_btn.clicked.connect(self.close)
        hdr_row.addWidget(close_btn)
        layout.addLayout(hdr_row)

        # Fila de Activación (PowerToggleButton SVG puro) y Medidor GR + Pico GR reiniciable
        act_row = QHBoxLayout()
        act_row.setSpacing(8)
        self.toggle_btn = PowerToggleButton(self)
        self.toggle_btn.setFixedSize(26, 26)
        self.toggle_btn.setAccentColor("#D500F9")
        self.toggle_btn.setToolTip("Activar / Desactivar Limitador Master")
        self.toggle_btn.setChecked(self.eq_engine.limiter_a.enabled)
        self.toggle_btn.clicked.connect(self._on_toggle_clicked)
        act_row.addWidget(self.toggle_btn)

        self.state_lbl = QLabel("ACTIVO" if self.eq_engine.limiter_a.enabled else "APAGADO")
        self.state_lbl.setStyleSheet("font-size: 10px; font-weight: bold; color: #F3B8FF;")
        act_row.addWidget(self.state_lbl)
        act_row.addStretch(1)

        self.gr_lbl = QLabel("GR: 0.0 dB")
        self.gr_lbl.setStyleSheet("font-size: 11px; font-weight: bold; color: #8C919B;")
        act_row.addWidget(self.gr_lbl)

        self.peak_gr_btn = QPushButton("Peak: 0.0 dB")
        self.peak_gr_btn.setObjectName("peakBtn")
        self.peak_gr_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.peak_gr_btn.setToolTip("Pico máximo de Reducción de Ganancia (Clic para reiniciar)")
        self.peak_gr_btn.clicked.connect(self._reset_peak_gr)
        act_row.addWidget(self.peak_gr_btn)
        layout.addLayout(act_row)

        # Separador
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background-color: rgba(255, 255, 255, 24); min-height: 1px; max-height: 1px;")
        layout.addWidget(sep)

        # Fila alineada de 4 perillas ProQKnob
        knobs_row = QHBoxLayout()
        knobs_row.setContentsMargins(0, 2, 0, 2)
        knobs_row.setSpacing(6)

        self.thr_knob = ProQKnob(
            title="THR", min_val=-30.0, max_val=0.0, default_val=0.0,
            fmt_func=lambda v: f"{v:.1f} dB", log_scale=False, bipolar=False,
            accent_color="#D500F9", diameter=54, reserve_outer_ring=True,
            min_label="-30", max_label="0", parent=self
        )
        self.thr_knob.set_gr_mode(True, max_gr_db=18.0, color="#D500F9")
        self.thr_knob.valueChanged.connect(self._on_thr_changed)
        knobs_row.addWidget(self.thr_knob)

        self.ceil_knob = ProQKnob(
            title="CEIL", min_val=-12.0, max_val=0.0, default_val=-0.1,
            fmt_func=lambda v: f"{v:.1f} dB", log_scale=False, bipolar=False,
            accent_color="#FF4081", diameter=54, reserve_outer_ring=True,
            min_label="-12", max_label="0", parent=self
        )
        self.ceil_knob.set_gr_mode(True, max_gr_db=18.0, color="#D500F9")
        self.ceil_knob.valueChanged.connect(self._on_ceil_changed)
        knobs_row.addWidget(self.ceil_knob)

        self.att_knob = ProQKnob(
            title="ATT", min_val=0.1, max_val=100.0, default_val=2.0,
            fmt_func=lambda v: f"{v:.1f} ms" if v < 10.0 else f"{v:.0f} ms",
            log_scale=True, bipolar=False, accent_color="#D500F9", diameter=54,
            min_label="0.1", max_label="100", parent=self
        )
        self.att_knob.valueChanged.connect(self._on_att_changed)
        knobs_row.addWidget(self.att_knob)

        self.rel_knob = ProQKnob(
            title="REL", min_val=1.0, max_val=500.0, default_val=50.0,
            fmt_func=lambda v: f"{v:.0f} ms",
            log_scale=True, bipolar=False, accent_color="#D500F9", diameter=54,
            min_label="1", max_label="500", parent=self
        )
        self.rel_knob.valueChanged.connect(self._on_rel_changed)
        knobs_row.addWidget(self.rel_knob)

        layout.addLayout(knobs_row)

        self.sync_from_engine()

    def showEvent(self, event):
        self.sync_from_engine()
        super().showEvent(event)

    def sync_from_engine(self):
        lim = self.eq_engine.limiter_a
        self.toggle_btn.blockSignals(True)
        self.toggle_btn.setChecked(bool(lim.enabled))
        self.toggle_btn.blockSignals(False)
        self._update_toggle_style()

        for knob, val in (
            (self.thr_knob, lim.threshold_db),
            (self.ceil_knob, lim.ceiling_db),
            (self.att_knob, lim.attack_ms),
            (self.rel_knob, lim.release_ms),
        ):
            knob.blockSignals(True)
            knob.setValue(val, emit_signal=False)
            knob.blockSignals(False)

    def _reset_peak_gr(self):
        self._peak_gr_db = 0.0
        self._last_peak_txt = "Peak: 0.0 dB"
        self.peak_gr_btn.setText(self._last_peak_txt)

    def _update_toggle_style(self):
        enabled = bool(self.eq_engine.limiter_a.enabled)
        self.toggle_btn.update()
        if hasattr(self, "state_lbl"):
            self.state_lbl.setText("ACTIVO" if enabled else "APAGADO")
            self.state_lbl.setStyleSheet(
                f"font-size: 10px; font-weight: bold; color: {'#F3B8FF' if enabled else '#7E8594'};"
            )
        if self.overlay_ref is not None and hasattr(self.overlay_ref, "limiter_btn"):
            self.overlay_ref.limiter_btn.setForceActive(enabled)
        for k in (self.thr_knob, self.ceil_knob, self.att_knob, self.rel_knob):
            k.setEnabled(enabled)

    def _on_toggle_clicked(self, checked: bool):
        self.eq_engine.limiter_a.enabled = bool(checked)
        self.eq_engine.limiter_b.enabled = bool(checked)
        self._update_toggle_style()
        if self.overlay_ref:
            self.overlay_ref.invalidate_cache()
        if self.app_ref:
            self.app_ref.schedule_config_save()

    def _on_thr_changed(self, val_db: float):
        v = float(val_db)
        self.eq_engine.limiter_a.threshold_db = v
        self.eq_engine.limiter_b.threshold_db = v
        self.eq_engine.limiter_a._update_coefficients()
        self.eq_engine.limiter_b._update_coefficients()
        if self.overlay_ref:
            self.overlay_ref.invalidate_cache()
        if self.app_ref:
            self.app_ref.schedule_config_save()

    def _on_ceil_changed(self, val_db: float):
        v = float(val_db)
        self.eq_engine.limiter_a.ceiling_db = v
        self.eq_engine.limiter_b.ceiling_db = v
        self.eq_engine.limiter_a._update_coefficients()
        self.eq_engine.limiter_b._update_coefficients()
        if self.app_ref:
            self.app_ref.schedule_config_save()

    def _on_att_changed(self, val_ms: float):
        v = float(val_ms)
        self.eq_engine.limiter_a.attack_ms = v
        self.eq_engine.limiter_b.attack_ms = v
        self.eq_engine.limiter_a._update_coefficients()
        self.eq_engine.limiter_b._update_coefficients()
        if self.app_ref:
            self.app_ref.schedule_config_save()

    def _on_rel_changed(self, val_ms: float):
        v = float(val_ms)
        self.eq_engine.limiter_a.release_ms = v
        self.eq_engine.limiter_b.release_ms = v
        self.eq_engine.limiter_a._update_coefficients()
        self.eq_engine.limiter_b._update_coefficients()
        if self.app_ref:
            self.app_ref.schedule_config_save()

    def set_live_gr(self, gr_db: float):
        if not self.isVisible():
            return
        gr = float(min(0.0, gr_db))
        self.ceil_knob.set_gr_level(gr)
        self.thr_knob.set_gr_level(gr)

        is_active = gr < -0.05
        txt = f"GR: {gr:.1f} dB" if is_active else "GR: 0.0 dB"
        if txt != self._last_gr_txt:
            self._last_gr_txt = txt
            self.gr_lbl.setText(txt)
        if is_active != self._last_gr_active:
            self._last_gr_active = is_active
            self.gr_lbl.setStyleSheet(
                f"font-size: 11px; font-weight: bold; color: {'#D500F9' if is_active else '#8C919B'};"
            )

        if gr < self._peak_gr_db - 0.05:
            self._peak_gr_db = gr
            ptxt = f"Peak: {self._peak_gr_db:.1f} dB"
            if ptxt != self._last_peak_txt:
                self._last_peak_txt = ptxt
                self.peak_gr_btn.setText(ptxt)

    update_meter = set_live_gr

    def _reset_defaults(self):
        self.toggle_btn.setChecked(True)
        self._on_toggle_clicked(True)
        self._reset_peak_gr()

        for knob, val, slot in (
            (self.thr_knob, 0.0, self._on_thr_changed),
            (self.ceil_knob, -0.1, self._on_ceil_changed),
            (self.att_knob, 2.0, self._on_att_changed),
            (self.rel_knob, 50.0, self._on_rel_changed),
        ):
            knob.blockSignals(True)
            knob.setValue(val, emit_signal=False)
            knob.blockSignals(False)
            slot(val)


class MiniOscilloscopeWidget(QWidget):
    """
    Mini-Osciloscopio a 60 FPS reales y 0% CPU con Scroll Blitting de 1 píxel.
    Acumula picos O(1) entre frames visuales sin bucles en Python ni sobrecarga de GPU.
    """
    clicked = pyqtSignal()
    colorPickRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(32)
        self.setMinimumWidth(110)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._pixmap: QPixmap | None = None
        self._scratch_pixmap: QPixmap | None = None
        self._pending_min = 0.0
        self._pending_max = 0.0
        self._pending_gr_db = 0.0
        self._has_new_data = False
        self._accum_steps = 0.0
        self._time_window_sec = 10.0
        self._fps = 60.0
        self._step_per_frame = 0.4
        self._wave_color = QColor("#00E5FF")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Clic izq: desplegar / plegar Osciloscopio detallado | Clic der: cambiar color")

    def set_wave_color(self, color: str | QColor):
        c = QColor(color) if color else QColor("#00E5FF")
        if c.isValid() and c != self._wave_color:
            self._wave_color = c
            self.update()

    def _recalc_step(self):
        w = max(68, int(self.width()))
        total_frames = max(1.0, float(self._time_window_sec) * self._fps)
        self._step_per_frame = max(0.25, float(w) / total_frames)

    def _ensure_pixmaps(self) -> bool:
        w, h = int(self.width()), int(self.height())
        if w <= 1 or h <= 1:
            return False
        if (
            self._pixmap is None
            or self._pixmap.isNull()
            or self._pixmap.width() != w
            or self._pixmap.height() != h
        ):
            new_pix = QPixmap(w, h)
            new_pix.fill(Qt.GlobalColor.transparent)
            if self._pixmap is not None and not self._pixmap.isNull():
                old_w = self._pixmap.width()
                cp = QPainter(new_pix)
                cp.drawPixmap(w - old_w, 0, self._pixmap)
                cp.end()
            self._pixmap = new_pix
            scr = QPixmap(w, h)
            scr.fill(Qt.GlobalColor.transparent)
            self._scratch_pixmap = scr
            self._recalc_step()
        return True

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._ensure_pixmaps()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()
        elif event.button() == Qt.MouseButton.RightButton:
            self.colorPickRequested.emit()
            event.accept()

    def set_time_window(self, seconds: float, fps: float = 60.0):
        self._time_window_sec = max(0.5, float(seconds))
        self._fps = max(10.0, float(fps))
        self._recalc_step()

    def push_audio_peaks(self, vmin: float, vmax: float, gr_db: float):
        # Acumula el pico más extremo entre frames visuales (O(1), sin np.roll)
        if not self._has_new_data:
            self._pending_min = vmin
            self._pending_max = vmax
            self._pending_gr_db = gr_db
            self._has_new_data = True
        else:
            if vmin < self._pending_min:
                self._pending_min = vmin
            if vmax > self._pending_max:
                self._pending_max = vmax
            if gr_db < self._pending_gr_db:
                self._pending_gr_db = gr_db

    def tick_render(self, wave_color: QColor):
        if not self._has_new_data:
            return
        if not self._ensure_pixmaps():
            return

        self._accum_steps += self._step_per_frame
        dx = int(self._accum_steps)
        if dx < 1:
            return
        self._accum_steps -= dx

        w, h = self.width(), self.height()
        dx = min(w, dx)
        mid_y = h * 0.5

        # 1. Copiar imagen previa desplazada hacia la izquierda sobre scratch pixmap transparente (flotante sobre el lienzo)
        self._scratch_pixmap.fill(Qt.GlobalColor.transparent)
        p = QPainter(self._scratch_pixmap)
        if dx < w:
            p.drawPixmap(-dx, 0, self._pixmap)

        lim_active = self._pending_gr_db < -0.1

        # 2. Línea base central tenue (pulsa púrpura sutil cuando el limitador actúa)
        base_col = QColor(213, 0, 249, 130) if lim_active else QColor(255, 255, 255, 26)
        p.setPen(QPen(base_col, 1))
        p.drawLine(w - dx, int(mid_y), w, int(mid_y))

        # 3. Dibujar envolvente de onda en la franja nueva
        amp = max(abs(self._pending_max), abs(self._pending_min))
        if amp > 0.002:
            y_top = int(np.clip(mid_y - self._pending_max * (mid_y - 2.0), 1, h - 2))
            y_bot = int(np.clip(mid_y - self._pending_min * (mid_y - 2.0), 1, h - 2))
            if y_top > y_bot:
                y_top, y_bot = y_bot, y_top
            col_draw = QColor("#D500F9") if lim_active else wave_color
            p.setPen(QPen(col_draw, 1))
            for x in range(w - dx, w):
                p.drawLine(x, y_top, x, max(y_top + 1, y_bot))

        # 4. Dibujar Reducción de Ganancia (Limitador) en púrpura desde el techo
        if lim_active:
            gr_h = int(min(h * 0.6, abs(self._pending_gr_db) * (h / 18.0)))
            p.fillRect(w - dx, 0, dx, max(2, gr_h), QColor(213, 0, 249, 210))

        p.end()
        self._pixmap, self._scratch_pixmap = self._scratch_pixmap, self._pixmap
        self._has_new_data = False
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self._pixmap is not None and not self._pixmap.isNull():
            p.drawPixmap(0, 0, self._pixmap)
        else:
            mid_y = int(self.height() * 0.5)
            p.setPen(QPen(QColor(255, 255, 255, 28), 1))
            p.drawLine(0, mid_y, self.width(), mid_y)

    def invalidate_canvas(self):
        if self._pixmap is not None:
            self._pixmap.fill(Qt.GlobalColor.transparent)
        self.update()

    def set_current_gr(self, gr_db: float):
        self._pending_gr_db = float(gr_db)

    def refresh_scope(self):
        self.tick_render(self._wave_color)


# Aliases de compatibilidad
CompactScopeGLWidget = MiniOscilloscopeWidget
_CompactScopeCanvas = MiniOscilloscopeWidget
OverlayMaskWidget = MiniOscilloscopeWidget


class AutoGainMiniBar(QWidget):
    """
    Botón gráfico SVG puro para Auto-Gain (auto_gain.svg de dynamic_eq_svg_icons):
    sin caja ni borde de botón Python — el propio medidor SVG es el botón y se ilumina al activarse.
    """
    toggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(28, 26)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Auto-Gain Independiente (Out A | Out B)\nClic para Activar / Desactivar")
        self._enabled = False
        self._hovered = False
        self._ag_a_db = 0.0
        self._ag_b_db = 0.0
        self._accent_color = QColor("#F4D35E")

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
            if self._enabled:
                self.setToolTip(f"Auto-Gain: ON (Out A: {self._ag_a_db:+.1f} dB | Out B: {self._ag_b_db:+.1f} dB)")
            else:
                self.setToolTip("Auto-Gain Independiente (Out A | Out B)\nClic para Activar / Desactivar")
            self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._enabled = not self._enabled
            self.toggled.emit(self._enabled)
            self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = float(self.width()), float(self.height())
        dpr = float(self.devicePixelRatioF()) if hasattr(self, "devicePixelRatioF") else 1.0
        st = "on" if self._enabled else ("hover" if self._hovered else "off")
        isz = max(20, int(min(w, h) - 1))
        pix = _render_svg_pixmap("auto_gain.svg", isz, isz, dpr=dpr, state=st, accent_hex="#F4D35E", recolor_primary=True)
        p.drawPixmap(int((w - isz) * 0.5), int((h - isz) * 0.5), pix)


class AutoGainScopeMeter(QWidget):
    """
    Medidor gráfico apilado de Auto-Gain independiente para Bus A y Bus B (-12 a +12 dB)
    para la cabecera del Osciloscopio expandido (solo lectura gráfica, sin botón duplicado).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(190, 22)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setToolTip("Ajuste gráfico de Auto-Gain (Arriba: Bus A | Abajo: Bus B)")
        self._enabled = False
        self._ag_a_db = 0.0
        self._ag_b_db = 0.0

    def set_state(self, enabled: bool, ag_a_db: float, ag_b_db: float = 0.0):
        en = bool(enabled)
        a = float(np.clip(ag_a_db, -12.0, 12.0))
        b = float(np.clip(ag_b_db, -12.0, 12.0))
        changed = (
            (self._enabled != en)
            or (abs(self._ag_a_db - a) > 0.04)
            or (abs(self._ag_b_db - b) > 0.04)
        )
        self._enabled = en
        self._ag_a_db = a
        self._ag_b_db = b
        if changed:
            if en:
                self.setToolTip(f"Auto-Gain Activo — Out A: {a:+.1f} dB | Out B: {b:+.1f} dB")
            else:
                self.setToolTip("Auto-Gain Desactivado (Activar desde la barra superior)")
            self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        h = float(self.height())

        bar_x = 2.0
        bar_w = 72.0
        bar_h = 4.5
        y_a = 4.0
        y_b = 13.0
        cx = bar_x + bar_w * 0.5

        for y_row, val_db, col_hex in (
            (y_a, self._ag_a_db, "#FFD54F"),
            (y_b, self._ag_b_db, "#00E5FF"),
        ):
            track_rect = QRectF(bar_x, y_row, bar_w, bar_h)
            track_col = QColor(255, 255, 255, 24 if self._enabled else 12)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(track_col)
            p.drawRoundedRect(track_rect, 2.0, 2.0)

            p.setPen(QPen(QColor(255, 255, 255, 75 if self._enabled else 30), 1.0))
            p.drawLine(QPointF(cx, y_row - 0.5), QPointF(cx, y_row + bar_h + 0.5))

            if self._enabled and abs(val_db) > 0.05:
                frac = float(np.clip(abs(val_db) / 12.0, 0.0, 1.0))
                fill_w = max(2.0, frac * (bar_w * 0.5))
                fx = cx if val_db >= 0.0 else (cx - fill_w)
                c_fill = QColor(col_hex)
                c_fill.setAlpha(225)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(c_fill)
                p.drawRoundedRect(QRectF(fx, y_row, fill_w, bar_h), 2.0, 2.0)

        p.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        if self._enabled:
            p.setPen(QColor("#FFD54F"))
            p.drawText(QRectF(80.0, 0.0, 52.0, h), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, f"A:{self._ag_a_db:+.1f}")
            p.setPen(QColor("#00E5FF"))
            p.drawText(QRectF(134.0, 0.0, 54.0, h), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, f"B:{self._ag_b_db:+.1f}")
        else:
            p.setPen(QColor("#788296"))
            p.drawText(QRectF(80.0, 0.0, 108.0, h), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, "AG: OFF")


class CrossoverMenuWidget(QWidget):
    """
    Panel de control gráfico SVG para el Crossover Digital de Salidas (Out A / Out B)
    para alojar dentro de un HUDPopupMenu con QWidgetAction y estética HUD translúcida.
    """
    configChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(320)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setStyleSheet("""
            QWidget {
                background-color: transparent;
                color: #ECEFF4;
                font-family: "Segoe UI", sans-serif;
                font-size: 11px;
            }
            QLabel {
                color: #C5CDD9;
                font-size: 10px;
                font-weight: 600;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        # 1. Header con PowerToggleButton SVG puro sincronizado
        hdr_row = QHBoxLayout()
        hdr_row.setSpacing(8)
        self.chk_enabled = PowerToggleButton(self)
        self.chk_enabled.setFixedSize(26, 26)
        self.chk_enabled.setChecked(False)
        self.chk_enabled.setAccentColor("#FFB300")
        self.chk_enabled.setToolTip("Activar / Desactivar Crossover Digital (Out A / Out B)")
        hdr_row.addWidget(self.chk_enabled)

        lbl_hdr = QLabel("Crossover Digital (Out A / Out B)", self)
        lbl_hdr.setStyleSheet("color: #FFB300; font-size: 11px; font-weight: bold;")
        hdr_row.addWidget(lbl_hdr)
        hdr_row.addStretch(1)
        layout.addLayout(hdr_row)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background-color: rgba(255, 255, 255, 28); min-height: 1px; max-height: 1px;")
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
        style_hud_spinbox(self.spin_freq, accent_color="#FFB300")
        grid.addWidget(lbl_freq, 0, 0)
        grid.addWidget(self.spin_freq, 0, 1)

        # 3. Tipo de Filtro (SvgDropdownButton sin caja)
        lbl_type = QLabel("Tipo de Filtro:", self)
        self.combo_type = SvgDropdownButton(
            self,
            icon_map={"Linkwitz-Riley": "lr_filter.svg", "Butterworth": "bw_filter.svg"},
            default_svg="lr_filter.svg",
            show_text=True,
            recolor_to_accent=True,
            accent_color="#FFB300",
        )
        self.combo_type.setFixedHeight(24)
        self.combo_type.addItems(["Linkwitz-Riley", "Butterworth"])
        grid.addWidget(lbl_type, 1, 0)
        grid.addWidget(self.combo_type, 1, 1)

        # 4. Pendiente (Slope SVG sin caja)
        lbl_slope = QLabel("Pendiente (Slope):", self)
        self.combo_slope = SvgDropdownButton(
            self,
            default_svg="slope.svg",
            show_text=True,
            recolor_to_accent=True,
            accent_color="#FFB300",
        )
        self.combo_slope.setFixedHeight(24)
        self.combo_slope.addItem("12 dB/oct", 12)
        self.combo_slope.addItem("24 dB/oct", 24)
        self.combo_slope.addItem("48 dB/oct", 48)
        grid.addWidget(lbl_slope, 2, 0)
        grid.addWidget(self.combo_slope, 2, 1)

        # 5. Asignación de Vías (SvgDropdownButton sin caja)
        lbl_routing = QLabel("Asignación de Vías:", self)
        self.combo_routing = SvgDropdownButton(
            self,
            default_svg="xover_routing.svg",
            show_text=True,
            recolor_to_accent=True,
            accent_color="#FFB300",
            short_text_func=lambda txt, d: (
                "A:Tops | B:Sub" if d == "A_Tops_B_Sub"
                else ("A:Sub | B:Tops" if d == "A_Sub_B_Tops" else "A:Full | B:Sub")
            ),
        )
        self.combo_routing.setFixedHeight(24)
        self.combo_routing.addItem("Out A: Tops (HP)  |  Out B: Sub (LP)", "A_Tops_B_Sub")
        self.combo_routing.addItem("Out A: Sub (LP)   |  Out B: Tops (HP)", "A_Sub_B_Tops")
        self.combo_routing.addItem("Out A: Full Range |  Out B: Sub (LP)", "A_Full_B_Sub")
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
        self.chk_enabled.update()
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
    Selector SVG sin bordes para alternar el banco de EQ activo:
    [ In 1 | In 2 ]
    """
    inputChanged = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFixedSize(64, 28)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 0, 2, 0)
        layout.setSpacing(4)

        self.btn_in1 = SvgToggleButton(
            "in1.svg", "In 1", self, active_tint_color="#00E5FF", pill_radius=6.0
        )
        self.btn_in1.setChecked(True)
        self.btn_in1.setFixedSize(28, 26)
        self.btn_in1.setToolTip("Editar banco de EQ para Entrada 1 (Principal)")
        self.btn_in1.clicked.connect(lambda: self._select_input(1))
        layout.addWidget(self.btn_in1)

        self.btn_in2 = SvgToggleButton(
            "in2.svg", "In 2", self, active_tint_color="#BA68C8", pill_radius=6.0
        )
        self.btn_in2.setChecked(False)
        self.btn_in2.setFixedSize(28, 26)
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


class CanvasModeSelector(QWidget):
    """
    Selector SVG de capas activas del lienzo:
    permite iluminar solo EQ, solo MB, ambos ("BOTH") o apagar ambos ("NONE" — solo FFT).
    """
    modeChanged = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFixedSize(64, 28)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 0, 2, 0)
        layout.setSpacing(4)

        self.btn_eq = SvgToggleButton(
            "eq_layer.svg", "EQ", self, active_tint_color="#00E5FF", pill_radius=6.0
        )
        self.btn_eq.setChecked(True)
        self.btn_eq.setFixedSize(28, 26)
        self.btn_eq.setToolTip("Mostrar/ocultar capa Ecualizador Paramétrico (ambos apagados = Solo FFT)")
        self.btn_eq.clicked.connect(lambda: self._toggle_layer("EQ"))
        layout.addWidget(self.btn_eq)

        self.btn_mb = SvgToggleButton(
            "mb_layer.svg", "MB", self, active_tint_color="#FF6D00", pill_radius=6.0
        )
        self.btn_mb.setChecked(True)
        self.btn_mb.setFixedSize(28, 26)
        self.btn_mb.setToolTip("Mostrar/ocultar capa Multiband Compressor / Limiter (ambos apagados = Solo FFT)")
        self.btn_mb.clicked.connect(lambda: self._toggle_layer("MB"))
        layout.addWidget(self.btn_mb)

        self._mode = "BOTH"

    def mode(self) -> str:
        return self._mode

    def _apply_buttons(self, m: str):
        self._mode = m
        eq_on = m in ("EQ", "BOTH")
        mb_on = m in ("MB", "BOTH")
        for btn, checked in ((self.btn_eq, eq_on), (self.btn_mb, mb_on)):
            btn.blockSignals(True)
            btn.setChecked(checked)
            btn.blockSignals(False)
            btn.update()

    def _toggle_layer(self, layer: str):
        eq_on = self.btn_eq.isChecked()
        mb_on = self.btn_mb.isChecked()
        if eq_on and mb_on:
            new_mode = "BOTH"
        elif eq_on:
            new_mode = "EQ"
        elif mb_on:
            new_mode = "MB"
        else:
            new_mode = "NONE"
        self._select_mode(new_mode)

    def _select_mode(self, mode: str):
        m = mode if mode in ("EQ", "MB", "BOTH", "NONE") else "BOTH"
        self._apply_buttons(m)
        self.modeChanged.emit(m)

    def set_mode(self, mode: str, emit_signal: bool = False):
        m = mode if mode in ("EQ", "MB", "BOTH", "NONE") else "BOTH"
        if m != self._mode:
            self._apply_buttons(m)
            if emit_signal:
                self.modeChanged.emit(m)
