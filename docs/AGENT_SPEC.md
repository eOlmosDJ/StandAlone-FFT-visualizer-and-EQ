# Especificación Técnica y Reglas de Arquitectura: Ecualizador Paramétrico & Dinámico Standalone (Estilo FabFilter Pro-Q 4) + Analizador FFT

## 1. Descripción del Sistema y Propuesta de Valor
Aplicación de escritorio standalone en Python para Windows que combina un **Analizador de Espectro FFT de alta precisión** con un **Ecualizador Paramétrico y Dinámico en Tiempo Real inspirado en FabFilter Pro-Q 4**, capaz de procesar todo el audio del sistema sin necesidad de abrir un DAW.
- Recibe la señal desde un dispositivo de entrada (ej. cable virtual en Windows) y la envía procesada con latencia ultra baja hacia los altavoces físicos.
- Permite ecualización estática y **ecualización dinámica por banda (Dynamic EQ)** (comprimiendo o expandiendo rangos de frecuencia específicos según un `Threshold`, `Dynamic Range`, `Attack` y `Release`).
- Visualiza simultáneamente el espectro **Pre-EQ** y **Post-EQ** (con curva suavizada por splines cúbicos y efecto "tira elástica" anclada en 10 Hz) junto con las curvas de respuesta estática y la excursión dinámica instantánea en tiempo real sobre una interfaz oscura de alta precisión (`#0D0F14`).

## 2. Stack Tecnológico Obligatorio
- **Interfaz Gráfica:** `PyQt6` (Prohibido usar Tkinter o Matplotlib por rendimiento/FPS).
- **Renderizado 2D Vectorial:** `pyqtgraph` (60 FPS con antialiasing, doble eje Y mediante `pg.ViewBox` superpuesto vinculado en X y estética oscura `#0D0F14`).
- **Motor de Audio en Tiempo Real:** `sounddevice` (PortAudio, `float32`, baja latencia con `blocksize = 512`).
- **Procesamiento Matemático y DSP:**
  - `numpy` (cálculo vectorial de coeficientes Biquad del *Audio EQ Cookbook*, envolventes RMS y `rfft`).
  - `scipy.signal.sosfilt` (filtrado IIR en cascada de secciones de segundo orden con preservación continua del vector de estado `zi` entre bloques).
  - `scipy.signal.sosfreqz` (evaluación de la respuesta en frecuencia compleja para renderizado de curvas estáticas y dinámicas a 60 FPS).
  - `scipy.interpolate.make_interp_spline` (interpolación cúbica del espectro FFT).
- **Empaquetado:** `PyInstaller` (`--noconsole --onefile`).

## 3. Arquitectura Modular y Paralela
El proyecto está estructurado en cuatro módulos desacoplados que operan en paralelo sin degradar el motor de audio ni el analizador espectral:
1. **`fft_analyzer.py` (Motor de Análisis Espectral Pre/Post-EQ):**
   - Gestiona los buffers circulares (`Ring Buffer`) independientes para las señales **Pre-EQ** y **Post-EQ** (`512` a `16384` muestras), desacoplados del `blocksize = 512` de audio.
   - Calcula `np.fft.rfft` con ventana Hanning, detector de envolvente asimétrico (ataque instantáneo + decaimiento regulable) e interpolación `make_interp_spline` con anclaje de "tira elástica" en `10 Hz` y `15 Hz`.
2. **`eq_dsp.py` (Motor Matemático DSP, Filtros Biquad y Dynamic EQ en Tiempo Real):**
   - Diseña filtros digitales en formato SOS (*Second-Order Sections*) mediante las fórmulas cerradas del *Audio EQ Cookbook* en `numpy`.
   - Soporta los tipos de filtro: **Bell (Campana)**, **Low Shelf**, **High Shelf**, **Low Cut / High Pass** (`12, 24, 36, 48 dB/oct`), **High Cut / Low Pass** (`12, 24, 36, 48 dB/oct`), **Notch** y **Band Pass**.
   - Ejecuta el algoritmo de **Dynamic EQ por banda**: filtrado sidechain Band-Pass auxiliar, medición de nivel RMS en dBFS, seguidor de envolvente exponencial (`Attack` / `Release`), cálculo de ganancia instantánea sobre el `Threshold` con rodilla suave (*soft knee*), actualización de coeficientes Biquad sin reiniciar el estado `zi`, y procesamiento del bloque `indata -> outdata`.
3. **`eq_ui.py` (Interfaz Interactiva y Visualizador de Curvas estilo FabFilter Pro-Q 4):**
   - Gestiona el `pg.ViewBox` secundario superpuesto sobre el `PlotWidget`, vinculado al eje X principal (`setXLink`) y asociado al eje Y derecho (`-24 dB` a `+24 dB`).
   - Dibuja la curva maestra estática (amarillo/dorado brillante), las curvas por banda, el límite punteado de rango dinámico ($G_{\text{static}} + \Delta G_{\text{max}}$) y la **curva dinámica activa en tiempo real** ($G_{\text{inst}} = G_{\text{static}} + \Delta G_{\text{current}}$) con área sombreada translúcida (`pg.FillBetweenItem`) que muestra cuántos dB comprime o expande cada banda frame a frame.
   - Maneja los nodos arrastrables en el gráfico (clic y arrastrar para ajustar $f_0$ y $G_{\text{static}}$, doble clic para crear bandas, rueda del mouse sobre un nodo para ajustar el factor $Q$) y el panel inferior de controles paramétricos y dinámicos estilo Pro-Q 4.
4. **`main.py` (Orquestador de la Aplicación, Audio Router y Ventana Principal):**
   - Integra el stream de `sounddevice`, las colas no bloqueantes Pre/Post-EQ, los vúmetros L/R (Entrada/Salida), el sistema de temas en `config.json`, los modos de ventana (Modo Mini, Fijar al Frente, Pantalla Completa) y la coordinación a 60 FPS entre `fft_analyzer.py`, `eq_dsp.py` y `eq_ui.py`.

## 4. Reglas Inquebrantables para Agentes y Modelos (NO VIOLAR)
1. **Eliminación Total del Modo Retro VFD:**
   - Queda estrictamente prohibido incluir código, imports, botones o archivos del antiguo modo retro VFD (`retro_vfd.py`). Toda la interfaz sigue una única estética moderna y oscura inspirada en **FabFilter Pro-Q 4** (fondo `#0D0F14`).
2. **Aislamiento y Rendimiento del Hilo de Audio (`audio_callback`):**
   - El callback de `sounddevice` corre en un hilo C de tiempo real. **PROHIBIDO** ejecutar `rfft`, `sosfreqz`, `make_interp_spline` o manipular widgets de `PyQt6` dentro del callback.
   - En el callback únicamente se ejecuta `eq_engine.process_block(indata)` (basado en `scipy.signal.sosfilt` y álgebra escalar/vectorial ligera en `numpy`), se respeta el estado de Mute y se envían copias `(pre_block, post_block)` a la cola no bloqueante (`put_nowait`).
3. **Filtrado Continuo Libre de Chasquidos (Preservación de `zi`):**
   - Cada banda mantiene su matriz de estados internos `zi` para el filtro de audio principal y `sc_zi` para su detector sidechain.
   - Cuando cambian $f_0$, $G$, $Q$ o la ganancia dinámica instantánea $G_{\text{inst}}$, se recalculan únicamente los 6 coeficientes Biquad $[b_0, b_1, b_2, 1, a_1, a_2]$ sin poner a cero `zi` (salvo que cambie el número de secciones SOS por un cambio de pendiente `12/24/36/48 dB/oct` o el número de canales).
4. **Seguridad de Monitoreo (Inicio en MUTE por Defecto):**
   - La aplicación arranca siempre con `self.is_muted = True` para prevenir acoples o retroalimentación accidental.
   - El motor DSP procesa el bloque antes de la compuerta de Mute para que tanto el analizador Post-EQ como las envolventes del Dynamic EQ puedan visualizarse en tiempo real incluso con la salida silenciada.
5. **Latencia Mínima Desacoplada de la Resolución FFT:**
   - `audio_block_size = 512` muestras fijas para mínima latencia de procesamiento EQ.
   - El analizador FFT opera sobre `RingBuffer` de hasta `16384` muestras alimentado desde la cola no bloqueante.
6. **Escala Logarítmica 1-2-5, "Tira Elástica" a 10 Hz y Doble Eje Y:**
   - Eje X logarítmico desde `10 Hz` (`log10(10) = 1.0`) hasta Nyquist con marcas 1-2-5 (`20, 50, 100, 200, 500, 1k, 2k, 5k, 10k, 20k`).
   - Eje Y izquierdo: nivel de espectro FFT en `dBFS` (`db_min` a `db_max`).
   - Eje Y derecho: ganancia del Ecualizador en `dB` (`-24 dB` a `+24 dB`) mediante `ViewBox` superpuesto con `setXLink`.
   - Navegación de ejes bloqueada al zoom/paneo por mouse (`setMouseEnabled(False, False)`), reservando el arrastre y la rueda del mouse para la manipulación directa de los nodos del EQ ($f_0$, $G$, $Q$).
7. **Persistencia en `config.json`:**
   - Almacena de forma segura y atómica (`.tmp` + `os.replace`) el tema visual y el estado de las bandas del ecualizador si corresponde, con validación tolerante a fallos.
