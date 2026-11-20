from dataclasses import dataclass
from typing import Callable

from PyQt6.QtCore import Qt, QObject, QEvent, QPoint, QPointF, QRect, QRectF, QTimer
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QBrush, QPolygonF
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
)

from eq_ui import LimiterControlPopup


@dataclass
class HelpStep:
    id: str
    section: str
    title: str
    description: str
    widget_getter: Callable[[], QWidget | None]
    prepare: Callable[[], None] | None = None
    spotlight_rect_fn: Callable[[], QRect] | None = None


class HelpSpotlightOverlay(QWidget):
    """
    Capa translúcida sobre el área central que atenúa el entorno y recorta un
    foco luminoso redondeado alrededor del control explicado actualmente.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._spotlight_rect = QRect()
        self._accent = QColor("#00E5FF")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.hide()

    def set_spotlight(self, local_rect: QRect, accent_color: str = "#00E5FF"):
        c = QColor(accent_color) if accent_color else QColor("#00E5FF")
        if not c.isValid():
            c = QColor("#00E5FF")
        self._spotlight_rect = QRect(local_rect)
        self._accent = c
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        full_rect = QRectF(self.rect())
        path = QPainterPath()
        path.setFillRule(Qt.FillRule.OddEvenFill)
        path.addRect(full_rect)

        spot = self._spotlight_rect
        if spot.isValid() and not spot.isEmpty():
            r_spot = QRectF(spot.adjusted(-4, -4, 4, 4))
            path.addRoundedRect(r_spot, 9.0, 9.0)
            painter.fillPath(path, QColor(6, 8, 13, 138))

            halo = QColor(self._accent)
            halo.setAlpha(72)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(halo, 5.0))
            painter.drawRoundedRect(r_spot, 9.0, 9.0)

            core = QColor(self._accent)
            core.setAlpha(235)
            painter.setPen(QPen(core, 1.8))
            painter.drawRoundedRect(r_spot, 9.0, 9.0)
        else:
            painter.fillPath(path, QColor(6, 8, 13, 138))


class HelpPopupCard(QWidget):
    """
    Tarjeta flotante explicativa estilo Liquid Glass oscuro con flecha indicadora,
    navegación paso a paso (Anterior / Siguiente / Saltar sección / Cerrar) y contador.
    """

    def __init__(self, tour: "InteractiveHelpTour", parent: QWidget | None = None):
        super().__init__(
            parent,
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowDoesNotAcceptFocus,
        )
        self.tour = tour
        self._accent = QColor("#00E5FF")
        self._arrow_dir = "none"  # "top" | "bottom" | "left" | "right" | "none"
        self._arrow_offset = 40.0
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setFixedWidth(338)

        self.setStyleSheet("""
            QWidget {
                background: transparent;
                color: #ECEFF4;
                font-family: "Segoe UI", sans-serif;
            }
            QPushButton#helpNavBtn {
                background-color: rgba(24, 28, 38, 210);
                color: #DCE2EE;
                border: 1px solid rgba(92, 102, 124, 175);
                border-radius: 6px;
                padding: 4px 9px;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton#helpNavBtn:hover:enabled {
                background-color: rgba(42, 48, 64, 235);
                border-color: #00E5FF;
                color: #FFFFFF;
            }
            QPushButton#helpNavBtn:disabled {
                color: #5E6575;
                border-color: rgba(65, 72, 88, 95);
            }
            QPushButton#helpPrimaryBtn {
                background-color: rgba(0, 229, 255, 42);
                color: #E8FBFF;
                border: 1px solid #00E5FF;
                border-radius: 6px;
                padding: 4px 11px;
                font-size: 11px;
                font-weight: 700;
            }
            QPushButton#helpPrimaryBtn:hover {
                background-color: rgba(0, 229, 255, 76);
                color: #FFFFFF;
            }
            QPushButton#helpCloseBtn {
                background: transparent;
                color: #8E95A5;
                border: none;
                font-size: 13px;
                font-weight: bold;
                padding: 0 4px;
            }
            QPushButton#helpCloseBtn:hover {
                color: #FF6B78;
            }
        """)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(7)

        hdr = QHBoxLayout()
        hdr.setSpacing(6)
        self.section_lbl = QLabel("SECCIÓN", self)
        self.section_lbl.setStyleSheet(
            "background-color: rgba(0, 229, 255, 32); color: #7EEFFF; "
            "border: 1px solid rgba(0, 229, 255, 115); border-radius: 4px; "
            "padding: 2px 7px; font-size: 9.5px; font-weight: bold;"
        )
        hdr.addWidget(self.section_lbl)

        self.counter_lbl = QLabel("1 / 1", self)
        self.counter_lbl.setStyleSheet("color: #8E95A5; font-size: 10px; font-weight: bold;")
        hdr.addWidget(self.counter_lbl)
        hdr.addStretch(1)

        self.close_x_btn = QPushButton("✕", self)
        self.close_x_btn.setObjectName("helpCloseBtn")
        self.close_x_btn.setFixedSize(22, 20)
        self.close_x_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.close_x_btn.setToolTip("Cerrar ayuda (Esc o clic en ?)")
        self.close_x_btn.clicked.connect(self.tour.stop)
        hdr.addWidget(self.close_x_btn)
        lay.addLayout(hdr)

        self.title_lbl = QLabel("", self)
        self.title_lbl.setWordWrap(True)
        self.title_lbl.setStyleSheet("color: #FFFFFF; font-size: 13px; font-weight: bold;")
        lay.addWidget(self.title_lbl)

        self.desc_lbl = QLabel("", self)
        self.desc_lbl.setWordWrap(True)
        self.desc_lbl.setStyleSheet("color: #CBD3E1; font-size: 11px; font-weight: 500;")
        lay.addWidget(self.desc_lbl)

        self.tip_lbl = QLabel("Tip: hacé clic en cualquier botón o perilla para inspeccionarlo.", self)
        self.tip_lbl.setWordWrap(True)
        self.tip_lbl.setStyleSheet("color: #7D8799; font-size: 9.5px; font-style: italic;")
        lay.addWidget(self.tip_lbl)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 4, 0, 0)
        btn_row.setSpacing(6)

        self.prev_btn = QPushButton("‹ Anterior", self)
        self.prev_btn.setObjectName("helpNavBtn")
        self.prev_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.prev_btn.clicked.connect(self.tour.prev_step)
        btn_row.addWidget(self.prev_btn)

        self.skip_sec_btn = QPushButton("Saltar sección", self)
        self.skip_sec_btn.setObjectName("helpNavBtn")
        self.skip_sec_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.skip_sec_btn.clicked.connect(self.tour.next_section)
        btn_row.addWidget(self.skip_sec_btn)

        btn_row.addStretch(1)

        self.next_btn = QPushButton("Siguiente ›", self)
        self.next_btn.setObjectName("helpPrimaryBtn")
        self.next_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.next_btn.clicked.connect(self.tour.next_step)
        btn_row.addWidget(self.next_btn)

        lay.addLayout(btn_row)

    def set_content(self, step: HelpStep, index: int, total: int, accent_hex: str):
        c = QColor(accent_hex) if accent_hex else QColor("#00E5FF")
        if not c.isValid():
            c = QColor("#00E5FF")
        self._accent = c
        hx = c.name()
        r, g, b = c.red(), c.green(), c.blue()

        self.section_lbl.setText(step.section.upper())
        self.section_lbl.setStyleSheet(
            f"background-color: rgba({r}, {g}, {b}, 34); color: #F0F8FF; "
            f"border: 1px solid rgba({r}, {g}, {b}, 145); border-radius: 4px; "
            "padding: 2px 7px; font-size: 9.5px; font-weight: bold;"
        )
        self.counter_lbl.setText(f"{index + 1} / {total}")
        self.title_lbl.setText(step.title)
        self.desc_lbl.setText(step.description)

        self.prev_btn.setEnabled(index > 0)
        is_last = (index >= total - 1)
        self.next_btn.setText("Finalizar ✓" if is_last else "Siguiente ›")
        self.next_btn.setStyleSheet(
            f"QPushButton {{ background-color: rgba({r}, {g}, {b}, 48); color: #FFFFFF; "
            f"border: 1px solid {hx}; border-radius: 6px; padding: 4px 11px; font-size: 11px; font-weight: 700; }}"
            f"QPushButton:hover {{ background-color: rgba({r}, {g}, {b}, 88); }}"
        )
        self.adjustSize()

    def position_near(self, target_global: QRect, host_global: QRect):
        cw = self.width()
        ch = self.height()
        gap = 12

        # Preferencia: debajo -> arriba -> derecha -> izquierda
        candidates = [
            ("top", QPoint(target_global.center().x() - cw // 2, target_global.bottom() + gap)),
            ("bottom", QPoint(target_global.center().x() - cw // 2, target_global.top() - ch - gap)),
            ("left", QPoint(target_global.right() + gap, target_global.center().y() - ch // 2)),
            ("right", QPoint(target_global.left() - cw - gap, target_global.center().y() - ch // 2)),
        ]

        chosen_dir = "none"
        chosen_pos = QPoint(
            host_global.center().x() - cw // 2,
            host_global.center().y() - ch // 2,
        )

        min_x = host_global.left() + 8
        max_x = max(min_x, host_global.right() - cw - 8)
        min_y = host_global.top() + 8
        max_y = max(min_y, host_global.bottom() - ch - 8)

        for arrow_d, raw_pt in candidates:
            cx = max(min_x, min(max_x, raw_pt.x()))
            cy = max(min_y, min(max_y, raw_pt.y()))
            cand_rect = QRect(cx, cy, cw, ch)
            if not cand_rect.intersects(target_global.adjusted(-4, -4, 4, 4)):
                chosen_dir = arrow_d
                chosen_pos = QPoint(cx, cy)
                break

        self._arrow_dir = chosen_dir
        if chosen_dir in ("top", "bottom"):
            self._arrow_offset = float(
                max(26, min(cw - 26, target_global.center().x() - chosen_pos.x()))
            )
        elif chosen_dir in ("left", "right"):
            self._arrow_offset = float(
                max(26, min(ch - 26, target_global.center().y() - chosen_pos.y()))
            )

        self.move(chosen_pos)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        pad = 8.0
        w = float(self.width())
        h = float(self.height())
        box = QRectF(pad, pad, w - 2.0 * pad, h - 2.0 * pad)

        path = QPainterPath()
        path.addRoundedRect(box, 11.0, 11.0)

        if self._arrow_dir != "none":
            tri = QPolygonF()
            aw = 8.0
            ah = 7.5
            if self._arrow_dir == "top":
                ax = min(box.right() - 16.0, max(box.left() + 16.0, self._arrow_offset))
                tri << QPointF(ax - aw, box.top()) << QPointF(ax, box.top() - ah) << QPointF(ax + aw, box.top())
            elif self._arrow_dir == "bottom":
                ax = min(box.right() - 16.0, max(box.left() + 16.0, self._arrow_offset))
                tri << QPointF(ax - aw, box.bottom()) << QPointF(ax, box.bottom() + ah) << QPointF(ax + aw, box.bottom())
            elif self._arrow_dir == "left":
                ay = min(box.bottom() - 16.0, max(box.top() + 16.0, self._arrow_offset))
                tri << QPointF(box.left(), ay - aw) << QPointF(box.left() - ah, ay) << QPointF(box.left(), ay + aw)
            elif self._arrow_dir == "right":
                ay = min(box.bottom() - 16.0, max(box.top() + 16.0, self._arrow_offset))
                tri << QPointF(box.right(), ay - aw) << QPointF(box.right() + ah, ay) << QPointF(box.right(), ay + aw)
            if not tri.isEmpty():
                tp = QPainterPath()
                tp.addPolygon(tri)
                path = path.united(tp)

        p.fillPath(path, QBrush(QColor(18, 21, 29, 244)))

        glow = QColor(self._accent)
        glow.setAlpha(60)
        p.setPen(QPen(glow, 3.6))
        p.drawPath(path)

        border = QColor(self._accent)
        border.setAlpha(220)
        p.setPen(QPen(border, 1.35))
        p.drawPath(path)


class InteractiveHelpTour(QObject):
    """
    Recorrido guiado interactivo completo sobre todos los botones y perillas de la aplicación.
    Permite avanzar paso a paso, saltar de sección o hacer clic directo sobre cualquier control
    para inspeccionar qué hace sin alterar ningún parámetro DSP; se cierra volviendo a pulsar '?' o Esc.
    """

    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self._active = False
        self._current_idx = 0
        self._steps: list[HelpStep] = []
        self._snapshot: dict | None = None
        self._tour_opened_scope = False
        self._tour_opened_limiter = False
        self._saved_limiter_flags = None

        self.overlay = HelpSpotlightOverlay(app.centralWidget())
        self.card = HelpPopupCard(self, app)
        self.card.hide()

    def is_active(self) -> bool:
        return self._active

    def _build_steps(self) -> list[HelpStep]:
        app = self.app
        hud = app.eq_panel
        sc = app.scope_overlay

        def prep_eq_base():
            self._ensure_limiter_popup(False)
            hud.set_help_preview_slope(False)
            hud.set_help_preview_dyn(False)
            hud.set_minimized(False)
            hud.set_hud_mode("eq")

        def prep_eq_slope():
            self._ensure_limiter_popup(False)
            hud.set_minimized(False)
            hud.set_hud_mode("eq")
            hud.set_help_preview_slope(True)
            hud.set_help_preview_dyn(False)

        def prep_eq_dyn():
            self._ensure_limiter_popup(False)
            hud.set_help_preview_slope(False)
            hud.set_minimized(False)
            hud.set_hud_mode("eq")
            hud.set_help_preview_dyn(True)

        def prep_mb():
            self._ensure_limiter_popup(False)
            hud.set_help_preview_slope(False)
            hud.set_help_preview_dyn(False)
            hud.set_minimized(False)
            hud.set_hud_mode("mb")

        def prep_scope():
            self._ensure_limiter_popup(False)
            if not sc.isVisible():
                self._tour_opened_scope = True
                app.toggle_expanded_scope()

        def prep_limiter():
            if not sc.isVisible():
                self._tour_opened_scope = True
                app.toggle_expanded_scope()
            self._ensure_limiter_popup(True)

        def lim_w(attr: str) -> QWidget | None:
            lp = getattr(sc, "limiter_popup", None)
            return getattr(lp, attr, None) if lp is not None else None

        steps: list[HelpStep] = [
            # 1. BARRA SUPERIOR
            HelpStep(
                "top_mute", "Barra Superior", "Mute / Activar Audio",
                "Silencia o habilita de inmediato ambas salidas de audio (Bus A y Bus B) por seguridad.",
                lambda: app.mute_btn,
            ),
            HelpStep(
                "top_audio", "Barra Superior", "Audio I/O y Latencia",
                "Configura el driver (WASAPI/ASIO/DirectSound), dispositivos Input 1 / Input 2, ruteo hacia Bus A/B, Sample Rate, Buffer y retardo de sincronización.",
                lambda: app.audio_menu_btn,
            ),
            HelpStep(
                "top_fft", "Barra Superior", "Analizador FFT y Medición RTA",
                "Elige qué curvas espectrales ver (In 1, In 2, Out 1, Out 2, RTA), resolución FFT, escalas X/Y, rango dB, retención y acceso a la configuración completa de colores.",
                lambda: app.analyzer_menu_btn,
            ),
            HelpStep(
                "top_in1", "Barra Superior", "Banco de Edición Input 1",
                "Selecciona el ecualizador y compresor multibanda independientes de la Entrada 1 (Principal) para editarlos en el lienzo y el HUD.",
                lambda: app.edit_input_selector.btn_in1,
            ),
            HelpStep(
                "top_in2", "Barra Superior", "Banco de Edición Input 2",
                "Cambia la edición al ecualizador y compresor multibanda independientes de la Entrada 2 (Auxiliar / Micrófono).",
                lambda: app.edit_input_selector.btn_in2,
            ),
            HelpStep(
                "top_layer_eq", "Barra Superior", "Capa de Ecualización (EQ)",
                "Muestra u oculta las curvas y nodos del ecualizador paramétrico sobre el espectro. Si apagás EQ y MB, ves solo el analizador FFT.",
                lambda: app.canvas_mode_selector.btn_eq,
            ),
            HelpStep(
                "top_layer_mb", "Barra Superior", "Capa Multibanda (MB)",
                "Muestra u oculta las bandas de dinámica multibanda (Compresor / Expansor / Limitador por frecuencia) en el lienzo.",
                lambda: app.canvas_mode_selector.btn_mb,
            ),
            HelpStep(
                "top_miniscope", "Barra Superior", "Mini-Osciloscopio",
                "Muestra la envolvente de salida en tiempo real. Clic izquierdo abre el Osciloscopio detallado; clic derecho permite elegir su color.",
                lambda: app.scope_widget.mini_scope,
            ),
            HelpStep(
                "top_scope_expand", "Barra Superior", "Desplegar / Plegar Osciloscopio",
                "Expande o pliega la vista detallada del osciloscopio multiseñal con controles de señales, limitador, crossover y ventana de tiempo.",
                lambda: app.scope_widget.expand_btn,
            ),
            HelpStep(
                "top_autogain", "Barra Superior", "Auto-Gain Independiente (A | B)",
                "Compensa automáticamente el cambio de sonoridad producido por las bandas de EQ de forma independiente para Out A y Out B.",
                lambda: app.autogain_mini_bar,
            ),
            HelpStep(
                "top_xover", "Barra Superior", "Crossover Digital (Out A / Out B)",
                "Activa o abre el menú del divisor de frecuencias entre Bus A (Tops) y Bus B (Sub) con pendiente, tipo de filtro y línea arrastrable en el lienzo.",
                lambda: app.crossover_btn,
            ),
            HelpStep(
                "top_bypass", "Barra Superior", "Bypass Global EQ",
                "Desactiva temporalmente todas las bandas del ecualizador activo para comparar la señal original contra la ecualizada.",
                lambda: app.bypass_btn,
            ),
            HelpStep(
                "top_presets", "Barra Superior", "Presets y Reset",
                "Guarda, carga, elimina presets, fija el preset predeterminado de inicio o resetea el EQ actual o todo el procesador.",
                lambda: app.presets_btn,
            ),
            HelpStep(
                "top_fullscreen", "Barra Superior", "Pantalla Completa (F11)",
                "Alterna el modo de pantalla completa para maximizar el área de análisis y edición espectral.",
                lambda: app.fullscreen_btn,
            ),
            HelpStep(
                "top_help", "Barra Superior", "Modo Ayuda Interactiva (?)",
                "Activa o cierra este tour guiado. Podés recorrer cada control con Siguiente o hacer clic directo sobre cualquier botón o perilla.",
                lambda: app.help_btn,
            ),

            # 2. LIENZO Y MEDIDORES
            HelpStep(
                "canvas_range", "Lienzo y Medidores", "Rango Visual de Ganancia (±12 / ±30 dB)",
                "Alterna la escala vertical de las curvas de EQ y dinámica entre ±12 dB (precisión fina) y ±30 dB (cortes y boosts amplios).",
                lambda: app.eq_range_badge,
            ),
            HelpStep(
                "canvas_meter", "Lienzo y Medidores", "Vúmetro Estéreo L / R",
                "Indica el nivel de pico instantáneo y retención de picos (peak hold) de la señal de salida estéreo en dBFS.",
                lambda: app.stereo_meter,
            ),
            HelpStep(
                "canvas_plot", "Lienzo y Medidores", "Lienzo Espectral Interactivo",
                "Doble clic crea una banda EQ; arrastrar mueve frecuencia y ganancia; rueda del ratón ajusta el Q; Shift permite ajuste fino y clic derecho abre opciones del nodo.",
                lambda: app.plot_widget,
                spotlight_rect_fn=lambda: QRect(
                    app.plot_widget.width() // 2 - 140,
                    app.plot_widget.height() // 2 - 70,
                    280,
                    120,
                ),
            ),

            # 3. HUD DE ECUALIZACIÓN (ESTÁTICO Y DINÁMICO)
            HelpStep(
                "eq_enable", "HUD de EQ", "Activar / Apagar Banda",
                "Habilita o deshabilita el procesamiento de la banda EQ seleccionada sin eliminarla.",
                lambda: hud.band_enable_btn,
                prepare=prep_eq_base,
            ),
            HelpStep(
                "eq_color", "HUD de EQ", "Color de Banda",
                "Permite elegir un color identificador personalizado para la banda activa, su nodo y el borde del HUD.",
                lambda: hud.color_swatch,
                prepare=prep_eq_base,
            ),
            HelpStep(
                "eq_type", "HUD de EQ", "Tipo de Filtro",
                "Selecciona la forma de la curva: Bell, Low/High Shelf, Low/High Cut, Notch, Band Pass, Tilt Shelf o All Pass.",
                lambda: hud.type_combo,
                prepare=prep_eq_base,
            ),
            HelpStep(
                "eq_slope", "HUD de EQ", "Pendiente del Filtro (Slope)",
                "Ajusta la caída en dB/octava (de 6 a 96 dB/oct) en filtros de corte, shelf o pasabanda.",
                lambda: hud.slope_combo,
                prepare=prep_eq_slope,
            ),
            HelpStep(
                "eq_freq", "HUD de EQ", "Perilla FREQ (Frecuencia)",
                "Define la frecuencia central o de corte de la banda seleccionada (10 Hz a 24 kHz). Arrastrá verticalmente o usá la rueda; doble clic resetea.",
                lambda: hud.freq_knob,
                prepare=prep_eq_base,
            ),
            HelpStep(
                "eq_gain", "HUD de EQ", "Perilla Principal GAIN / Rango Dinámico",
                "El capuchón central ajusta la ganancia estática (±24 dB). El anillo exterior (o clic derecho / Alt+clic) fija el rango dinámico (±30 dB).",
                lambda: hud.gain_dyn_knob,
                prepare=prep_eq_base,
            ),
            HelpStep(
                "eq_q", "HUD de EQ", "Perilla Q (Ancho de Banda)",
                "Controla qué tan estrecha o ancha es la campana o resonancia del filtro (0.10 a 24.00).",
                lambda: hud.q_knob,
                prepare=prep_eq_base,
            ),
            HelpStep(
                "eq_thr", "HUD de EQ (Dinámico)", "Perilla THR (Threshold Dinámico)",
                "Nivel en dB a partir del cual se dispara la excursión dinámica de la banda. Aparece automáticamente cuando DYN está activo.",
                lambda: hud.thresh_knob,
                prepare=prep_eq_dyn,
            ),
            HelpStep(
                "eq_attack", "HUD de EQ (Dinámico)", "Perilla A (Attack Dinámico)",
                "Velocidad con la que el filtro dinámico reacciona cuando la señal supera el umbral (0.5 ms a 500 ms).",
                lambda: hud.attack_knob,
                prepare=prep_eq_dyn,
            ),
            HelpStep(
                "eq_release", "HUD de EQ (Dinámico)", "Perilla R (Release Dinámico)",
                "Tiempo que tarda el filtro dinámico en volver a su ganancia estática cuando la señal cae bajo el umbral (5 ms a 2000 ms).",
                lambda: hud.release_knob,
                prepare=prep_eq_dyn,
            ),
            HelpStep(
                "eq_out", "HUD de EQ", "Perilla OUT (Ganancia de Salida EQ)",
                "Ajusta el nivel maestro de salida del banco de EQ activo (-24 a +24 dB).",
                lambda: hud.out_gain_knob,
                prepare=prep_eq_base,
            ),
            HelpStep(
                "eq_prev", "HUD de EQ", "Banda Anterior",
                "Navega al nodo EQ o banda Multibanda anterior en la lista unificada.",
                lambda: hud.prev_band_btn,
                prepare=prep_eq_base,
            ),
            HelpStep(
                "eq_combo", "HUD de EQ", "Selector de Banda / Nodo",
                "Lista todas las bandas de EQ y compresión multibanda disponibles para saltar a cualquiera o crear una nueva.",
                lambda: hud.band_combo,
                prepare=prep_eq_base,
            ),
            HelpStep(
                "eq_next", "HUD de EQ", "Banda Siguiente",
                "Avanza al siguiente nodo EQ o banda Multibanda.",
                lambda: hud.next_band_btn,
                prepare=prep_eq_base,
            ),
            HelpStep(
                "eq_add", "HUD de EQ", "Añadir Banda EQ",
                "Crea una nueva banda Bell en 1 kHz para el banco de entrada activo.",
                lambda: hud.add_band_btn,
                prepare=prep_eq_base,
            ),
            HelpStep(
                "eq_del", "HUD de EQ", "Eliminar Banda",
                "Elimina permanentemente la banda seleccionada (equivalente a la tecla Supr).",
                lambda: hud.del_band_btn,
                prepare=prep_eq_base,
            ),
            HelpStep(
                "eq_add_mb", "HUD de EQ", "Crear Banda Multibanda (+ MB)",
                "Activa la colocación interactiva en el espectro para crear una nueva banda de compresión/expansión multibanda.",
                lambda: hud.add_mb_hud_btn,
                prepare=prep_eq_base,
            ),
            HelpStep(
                "eq_ch_mode", "HUD de EQ", "Modo de Canal (LR / L / R / Mid / Side)",
                "Define si la banda procesa todo el estéreo, solo canal izquierdo/derecho, centro (Mid) o laterales (Side).",
                lambda: hud.channel_mode_combo,
                prepare=prep_eq_base,
            ),
            HelpStep(
                "eq_target_out", "HUD de EQ", "Bus de Salida de Banda (A+B / A / B)",
                "Asigna si esta banda de EQ actúa sobre ambas salidas (A+B) o exclusivamente sobre Out A u Out B.",
                lambda: hud.target_out_combo,
                prepare=prep_eq_base,
            ),
            HelpStep(
                "eq_dyn_btn", "HUD de EQ", "Activar Dinámica (DYN)",
                "Enciende o apaga la ecualización dinámica en la banda actual, desplegando las perillas THR, A y R.",
                lambda: hud.dyn_toggle_btn,
                prepare=prep_eq_base,
            ),
            HelpStep(
                "eq_solo", "HUD de EQ", "Isolate (Solo de Frecuencia)",
                "Aísla la escucha únicamente al rango de frecuencias cubierto por la banda seleccionada.",
                lambda: hud.band_solo_btn,
                prepare=prep_eq_base,
            ),
            HelpStep(
                "hud_auto", "HUD Flotante", "Auto-Expandir HUD",
                "Cuando está activo, el HUD se minimiza al apartar el puntero y se expande automáticamente al pasar el cursor encima.",
                lambda: hud.auto_expand_btn,
                prepare=prep_eq_base,
            ),
            HelpStep(
                "hud_minimize", "HUD Flotante", "Minimizar HUD",
                "Pliega el HUD a una barra compacta anclada abajo sin perder el acceso rápido a lectura, bus de salida e Isolate.",
                lambda: hud.minimize_btn,
                prepare=prep_eq_base,
            ),
            HelpStep(
                "hud_drag", "HUD Flotante", "Mover / Arrastrar HUD",
                "Agarrá desde aquí (o desde el fondo del panel) para ubicar el HUD libremente en cualquier parte del gráfico.",
                lambda: hud.drag_grip_btn,
                prepare=prep_eq_base,
            ),

            # 4. HUD MULTIBANDA (PRO-MB)
            HelpStep(
                "mb_enable", "HUD Multibanda", "Activar / Apagar Banda MB",
                "Activa o desactiva la banda de dinámica multibanda seleccionada en el compresor de la entrada actual.",
                lambda: hud.mb_enable_btn,
                prepare=prep_mb,
            ),
            HelpStep(
                "mb_color", "HUD Multibanda", "Color de Banda MB",
                "Personaliza el color de la región sombreada, nodo y controles de la banda multibanda.",
                lambda: hud.mb_color_swatch,
                prepare=prep_mb,
            ),
            HelpStep(
                "mb_mode", "HUD Multibanda", "Modo de Dinámica (Compress / Expand / Limit)",
                "Elige si la banda comprime picos, expande/atenúa señal baja (Gate/Expander) o limita con relación ∞:1.",
                lambda: hud.mb_mode_btn,
                prepare=prep_mb,
            ),
            HelpStep(
                "mb_char", "HUD Multibanda", "Character (Saturación Armónica)",
                "Agrega coloración analógica (Tube, Diode o Bright) con control de Drive y ubicación Pre o Post dinámica.",
                lambda: hud.mb_char_btn,
                prepare=prep_mb,
            ),
            HelpStep(
                "mb_range", "HUD Multibanda", "Perilla RANGE",
                "Limita la máxima atenuación o expansión permitida en la banda (-30 dB a +12 dB).",
                lambda: hud.mb_range_knob,
                prepare=prep_mb,
            ),
            HelpStep(
                "mb_ratio", "HUD Multibanda", "Perilla RATIO",
                "Define la relación de compresión o expansión (1.0:1 a 20.0:1; fija en ∞:1 en modo Limit).",
                lambda: hud.mb_ratio_knob,
                prepare=prep_mb,
            ),
            HelpStep(
                "mb_knee", "HUD Multibanda", "Perilla KNEE",
                "Suaviza la transición alrededor del umbral (0 dB hard knee a 24 dB soft knee).",
                lambda: hud.mb_knee_knob,
                prepare=prep_mb,
            ),
            HelpStep(
                "mb_thr", "HUD Multibanda", "Perilla Central THRESHOLD + Medidor GR",
                "Umbral de activación en dB con anillo de nivel de entrada en vivo y lectura de Reducción de Ganancia (GR) en el centro.",
                lambda: hud.mb_thr_knob,
                prepare=prep_mb,
            ),
            HelpStep(
                "mb_attack", "HUD Multibanda", "Perilla ATTACK",
                "Rapidez con que actúa el compresor/limitador al superar el umbral (0.1 ms a 200 ms).",
                lambda: hud.mb_attack_knob,
                prepare=prep_mb,
            ),
            HelpStep(
                "mb_release", "HUD Multibanda", "Perilla RELEASE",
                "Tiempo de recuperación tras descender la señal bajo el umbral (5 ms a 2000 ms).",
                lambda: hud.mb_release_knob,
                prepare=prep_mb,
            ),
            HelpStep(
                "mb_out", "HUD Multibanda", "Perilla OUT (Makeup Gain)",
                "Ganancia de salida independiente de esta banda multibanda (-24 a +24 dB).",
                lambda: hud.mb_output_knob,
                prepare=prep_mb,
            ),
            HelpStep(
                "mb_prev", "HUD Multibanda", "Banda / Nodo Anterior (MB)",
                "Navega a la banda multibanda o nodo EQ anterior en la lista unificada.",
                lambda: hud.mb_prev_btn,
                prepare=prep_mb,
            ),
            HelpStep(
                "mb_nav", "HUD Multibanda", "Selector de Banda MB",
                "Permite alternar entre nodos EQ y bandas Multibanda o seleccionar directamente desde la lista desplegable.",
                lambda: hud.mb_band_combo,
                prepare=prep_mb,
            ),
            HelpStep(
                "mb_next", "HUD Multibanda", "Banda / Nodo Siguiente (MB)",
                "Avanza a la siguiente banda multibanda o nodo EQ en la lista unificada.",
                lambda: hud.mb_next_btn,
                prepare=prep_mb,
            ),
            HelpStep(
                "mb_add", "HUD Multibanda", "Crear Banda MB",
                "Inicia la creación interactiva de una nueva banda de frecuencia LR4 en el lienzo.",
                lambda: hud.mb_add_btn,
                prepare=prep_mb,
            ),
            HelpStep(
                "mb_del", "HUD Multibanda", "Eliminar Banda MB",
                "Elimina la banda multibanda seleccionada actualmente.",
                lambda: hud.mb_del_btn,
                prepare=prep_mb,
            ),
            HelpStep(
                "mb_freq_lbl", "HUD Multibanda", "Rango de Frecuencias de Banda",
                "Muestra los cruces inferior y superior [f_low – f_high] ajustables arrastrando los bordes verticales en el gráfico.",
                lambda: hud.mb_freq_lbl,
                prepare=prep_mb,
            ),
            HelpStep(
                "mb_bus", "HUD Multibanda", "Bus de Salida MB (A+B / A / B)",
                "Determina si esta banda multibanda procesa ambas salidas o solo Out A / Out B.",
                lambda: hud.mb_bus_combo,
                prepare=prep_mb,
            ),
            HelpStep(
                "mb_solo", "HUD Multibanda", "Solo / Isolate Multibanda",
                "Aísla la escucha a la banda de frecuencia LR4 seleccionada (Alt+clic sobre el nodo en el gráfico).",
                lambda: hud.mb_solo_btn,
                prepare=prep_mb,
            ),
            HelpStep(
                "mb_bypass", "HUD Multibanda", "Bypass Global Multibanda",
                "Desactiva todas las bandas del compresor multibanda de la entrada activa para comparar.",
                lambda: hud.mb_global_bypass_btn,
                prepare=prep_mb,
            ),

            # 5. OSCILOSCOPIO EXPANDIDO Y LIMITADOR MASTER
            HelpStep(
                "scope_signals", "Osciloscopio Expandido", "Señales y Opacidad",
                "Activa qué formas de onda dibujar (L/R/L+R de Entrada, Bus A y Bus B), sus colores individuales y su opacidad.",
                lambda: sc.settings_btn,
                prepare=prep_scope,
            ),
            HelpStep(
                "scope_limiter", "Osciloscopio Expandido", "Abrir Limitador Master",
                "Abre o cierra el panel del Limitador Master protector final para ambas salidas.",
                lambda: sc.limiter_btn,
                prepare=prep_scope,
            ),
            HelpStep(
                "scope_gr", "Osciloscopio Expandido", "Indicador GR del Limitador",
                "Muestra en tiempo real la reducción de ganancia aplicada por el Limitador Master; clic abre sus controles.",
                lambda: sc.gr_meter_widget,
                prepare=prep_scope,
            ),
            HelpStep(
                "scope_xover", "Osciloscopio Expandido", "Acceso Rápido a Crossover",
                "Despliega el menú de configuración del Crossover Digital entre Bus A y Bus B.",
                lambda: sc.xover_scope_btn,
                prepare=prep_scope,
            ),
            HelpStep(
                "scope_ag", "Osciloscopio Expandido", "Medidor Auto-Gain (A y B)",
                "Barra gráfica dual que indica cuántos dB de compensación Auto-Gain se aplican a Bus A y Bus B.",
                lambda: sc.ag_scope_meter,
                prepare=prep_scope,
            ),
            HelpStep(
                "scope_dur", "Osciloscopio Expandido", "Ventana Temporal (1s / 5s / 10s)",
                "Cambia la duración histórica visible en el Osciloscopio y Mini-Osciloscopio.",
                lambda: sc.dur_btn,
                prepare=prep_scope,
            ),
            HelpStep(
                "lim_power", "Limitador Master", "Activar / Desactivar Limitador",
                "Habilita o apaga el limitador master final que protege contra saturación digital (clipping).",
                lambda: lim_w("toggle_btn"),
                prepare=prep_limiter,
            ),
            HelpStep(
                "lim_thr", "Limitador Master", "Perilla THR (Threshold)",
                "Nivel a partir del cual el limitador comienza a contener los picos de la mezcla final (-30 a 0 dB).",
                lambda: lim_w("thr_knob"),
                prepare=prep_limiter,
            ),
            HelpStep(
                "lim_ceil", "Limitador Master", "Perilla CEIL (Techo Máximo)",
                "Nivel máximo absoluto que jamás superará la salida final (-12 a 0 dBFS).",
                lambda: lim_w("ceil_knob"),
                prepare=prep_limiter,
            ),
            HelpStep(
                "lim_att", "Limitador Master", "Perilla ATT (Attack)",
                "Tiempo de respuesta del limitador ante transitorios rápidos (0.1 a 100 ms).",
                lambda: lim_w("att_knob"),
                prepare=prep_limiter,
            ),
            HelpStep(
                "lim_rel", "Limitador Master", "Perilla REL (Release)",
                "Tiempo de recuperación del limitador después de limitar un pico (1 a 500 ms).",
                lambda: lim_w("rel_knob"),
                prepare=prep_limiter,
            ),
            HelpStep(
                "lim_peak", "Limitador Master", "Pico Máximo de GR (Reset)",
                "Registra la mayor reducción de ganancia ocurrida. Hacé clic sobre el valor para reiniciarlo.",
                lambda: lim_w("peak_gr_btn"),
                prepare=prep_limiter,
            ),
            HelpStep(
                "lim_reset", "Limitador Master", "Restaurar Limitador",
                "Restablece todos los parámetros del Limitador Master a sus valores iniciales de seguridad.",
                lambda: lim_w("reset_btn"),
                prepare=prep_limiter,
            ),
        ]
        return steps

    def _ensure_limiter_popup(self, show: bool):
        sc = getattr(self.app, "scope_overlay", None)
        if sc is None:
            return
        lp = getattr(sc, "limiter_popup", None)
        if show:
            if lp is None:
                lp = LimiterControlPopup(
                    eq_engine=self.app.eq_engine,
                    overlay_ref=sc,
                    app_ref=self.app,
                    parent=sc,
                )
                sc.limiter_popup = lp
            if self._saved_limiter_flags is None:
                self._saved_limiter_flags = lp.windowFlags()
            if not lp.isVisible():
                self._tour_opened_limiter = True
            lp.setWindowFlags(
                Qt.WindowType.Tool
                | Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.NoDropShadowWindowHint
                | Qt.WindowType.WindowDoesNotAcceptFocus
            )
            lp.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
            pos = sc.limiter_btn.mapToGlobal(QPoint(0, sc.limiter_btn.height() + 4))
            lp.move(pos)
            lp.show()
            lp.raise_()
        else:
            if lp is not None:
                was_open_before = bool(self._snapshot and self._snapshot.get("limiter_open"))
                if self._saved_limiter_flags is not None:
                    was_vis = lp.isVisible()
                    lp.setWindowFlags(self._saved_limiter_flags)
                    self._saved_limiter_flags = None
                    if was_vis and not self._tour_opened_limiter and was_open_before:
                        lp.show()
                if self._tour_opened_limiter and lp.isVisible():
                    lp.close()
                self._tour_opened_limiter = False

    def start(self):
        if self._active:
            self.stop()
            return
        app = self.app
        hud = app.eq_panel
        sc = app.scope_overlay
        lp = getattr(sc, "limiter_popup", None)

        self._snapshot = {
            "hud_mode": getattr(hud, "_hud_mode", "eq"),
            "hud_minimized": bool(getattr(hud, "_is_minimized", False)),
            "scope_open": bool(sc.isVisible()),
            "limiter_open": bool(lp is not None and lp.isVisible()),
        }
        self._tour_opened_scope = False
        self._tour_opened_limiter = False
        self._saved_limiter_flags = None

        self._steps = self._build_steps()
        self._active = True
        self._current_idx = 0

        app.help_btn.blockSignals(True)
        app.help_btn.setChecked(True)
        app.help_btn.blockSignals(False)
        app.help_btn.update()

        cw = app.centralWidget()
        if cw is not None:
            self.overlay.setParent(cw)
            self.overlay.setGeometry(cw.rect())
            self.overlay.show()
            self.overlay.raise_()

        qapp = QApplication.instance()
        if qapp is not None:
            qapp.installEventFilter(self)

        self.show_step(0)

    def stop(self):
        if not self._active:
            return
        self._active = False

        qapp = QApplication.instance()
        if qapp is not None:
            qapp.removeEventFilter(self)

        self.card.hide()
        self.overlay.hide()
        self._ensure_limiter_popup(False)

        hud = self.app.eq_panel
        hud.set_help_preview_slope(False)
        hud.set_help_preview_dyn(False)

        if self._snapshot is not None:
            if self._tour_opened_scope and self.app.scope_overlay.isVisible() and not self._snapshot["scope_open"]:
                self.app.toggle_expanded_scope()
            hud.set_hud_mode(self._snapshot["hud_mode"])
            hud.set_minimized(self._snapshot["hud_minimized"])
            self._snapshot = None

        self._tour_opened_scope = False
        self._tour_opened_limiter = False

        self.app.help_btn.blockSignals(True)
        self.app.help_btn.setChecked(False)
        self.app.help_btn.blockSignals(False)
        self.app.help_btn.update()

    def show_step(self, index: int):
        if not self._active or not self._steps:
            return
        index = max(0, min(len(self._steps) - 1, int(index)))
        self._current_idx = index
        step = self._steps[index]

        if step.prepare is not None:
            step.prepare()

        QApplication.processEvents()
        self._reposition_current()

        hud = self.app.eq_panel
        if getattr(hud, "_dyn_width_anim", None) is not None and hud._dyn_width_anim.state():
            QTimer.singleShot(260, self._reposition_current)

    def _reposition_current(self):
        if not self._active or not (0 <= self._current_idx < len(self._steps)):
            return
        app = self.app
        cw = app.centralWidget()
        if cw is not None:
            self.overlay.setGeometry(cw.rect())
            self.overlay.raise_()

        step = self._steps[self._current_idx]
        w = step.widget_getter()
        acc = app._get_active_hud_accent_hex() if hasattr(app, "_get_active_hud_accent_hex") else "#00E5FF"
        self.card.set_content(step, self._current_idx, len(self._steps), acc)

        if w is not None and w.isVisible():
            if step.spotlight_rect_fn is not None:
                local_r = step.spotlight_rect_fn()
                tl = w.mapToGlobal(local_r.topLeft())
                global_rect = QRect(tl, local_r.size())
            else:
                tl = w.mapToGlobal(QPoint(0, 0))
                global_rect = QRect(tl, w.size())

            ov_tl = self.overlay.mapFromGlobal(global_rect.topLeft())
            ov_rect = QRect(ov_tl, global_rect.size())
            self.overlay.set_spotlight(ov_rect, acc)

            host_tl = app.mapToGlobal(QPoint(0, 0))
            host_rect = QRect(host_tl, app.size())
            self.card.position_near(global_rect, host_rect)
        else:
            self.overlay.set_spotlight(QRect(), acc)
            host_tl = app.mapToGlobal(QPoint(0, 0))
            host_rect = QRect(host_tl, app.size())
            self.card.move(
                host_rect.center().x() - self.card.width() // 2,
                host_rect.center().y() - self.card.height() // 2,
            )

        self.card.show()
        self.card.raise_()

    def next_step(self):
        if not self._active:
            return
        if self._current_idx >= len(self._steps) - 1:
            self.stop()
        else:
            self.show_step(self._current_idx + 1)

    def prev_step(self):
        if not self._active:
            return
        if self._current_idx > 0:
            self.show_step(self._current_idx - 1)

    def next_section(self):
        if not self._active or not self._steps:
            return
        cur_sec = self._steps[self._current_idx].section
        for i in range(self._current_idx + 1, len(self._steps)):
            if self._steps[i].section != cur_sec:
                self.show_step(i)
                return
        self.show_step(0)

    def _find_step_for_widget(self, w: QWidget | None) -> int | None:
        if w is None:
            return None
        # Mapa rápido widget -> primer índice de paso
        targets: list[tuple[int, QWidget]] = []
        for i, st in enumerate(self._steps):
            tw = st.widget_getter()
            if tw is not None:
                targets.append((i, tw))

        cur: QWidget | None = w
        visited = set()
        hud = self.app.eq_panel
        alias_map = {
            id(hud.mb_auto_expand_btn): "hud_auto",
            id(hud.mb_minimize_btn): "hud_minimize",
            id(hud.mb_drag_grip_btn): "hud_drag",
            id(hud.compact_auto_btn): "hud_auto",
            id(hud.compact_expand_btn): "hud_minimize",
            id(hud.compact_drag_grip): "hud_drag",
        }
        id_by_step = {st.id: i for i, st in enumerate(self._steps)}
        while cur is not None and id(cur) not in visited:
            visited.add(id(cur))
            if id(cur) in alias_map and alias_map[id(cur)] in id_by_step:
                return id_by_step[alias_map[id(cur)]]
            for idx, tw in targets:
                if cur is tw:
                    return idx
            cur = cur.parentWidget()
        return None

    def eventFilter(self, obj, event):
        if not self._active:
            return super().eventFilter(obj, event)

        etype = event.type()

        if etype == QEvent.Type.Resize and obj is self.app:
            QTimer.singleShot(0, self._reposition_current)
            return False

        if etype == QEvent.Type.KeyPress:
            key = event.key()
            if key == Qt.Key.Key_Escape:
                QTimer.singleShot(0, self.stop)
                return True
            if key in (Qt.Key.Key_Right, Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
                QTimer.singleShot(0, self.next_step)
                return True
            if key == Qt.Key.Key_Left:
                QTimer.singleShot(0, self.prev_step)
                return True

        if isinstance(obj, QWidget):
            # Permitir interacción normal con los botones de la tarjeta de ayuda
            if obj is self.card or self.card.isAncestorOf(obj):
                return False

            # Solo interceptar widgets pertenecientes a nuestra app (y sus popups hijos)
            if obj.window() is self.app or (
                getattr(self.app.scope_overlay, "limiter_popup", None) is not None
                and obj.window() is self.app.scope_overlay.limiter_popup
            ):
                if etype in (
                    QEvent.Type.MouseButtonPress,
                    QEvent.Type.MouseButtonRelease,
                    QEvent.Type.MouseButtonDblClick,
                    QEvent.Type.Wheel,
                ):
                    if obj is self.app.help_btn or self.app.help_btn.isAncestorOf(obj):
                        if etype == QEvent.Type.MouseButtonPress:
                            QTimer.singleShot(0, self.stop)
                        return True

                    if etype == QEvent.Type.MouseButtonPress:
                        hit_idx = self._find_step_for_widget(obj)
                        if hit_idx is not None:
                            QTimer.singleShot(0, lambda i=hit_idx: self.show_step(i))
                    return True

        return super().eventFilter(obj, event)
