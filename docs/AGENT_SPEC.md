# Especificación Técnica y Reglas de Arquitectura: BDJ-FFT-EQ & Limiter Standalone

## 1. Descripción del Sistema y Propuesta de Valor
Aplicación de escritorio standalone en Python para Windows que combina un **Analizador de Espectro FFT multi-resolución a 60+ FPS**, un **Ecualizador Paramétrico y Dinámico en Tiempo Real (BDJ-FFT-EQ)**, una **Matriz de Crossover Digital Linkwitz-Riley**, un **Sistema Inteligente de Auto-Gain**, un **Limitador de Picos Master con visualización de Reducción de Ganancia (GR)** y un **Osciloscopio acelerado por hardware OpenGL**, capaz de procesar todo el audio del sistema sin necesidad de abrir un DAW.
- Recibe la señal desde un dispositivo de entrada (ej. cable virtual en Windows) y la envía procesada con latencia ultra baja hacia altavoces físicos individuales o matrices de doble bus (`Bus A` y `Bus B`).
- Permite ecualización estática y **ecualización dinámica por banda (Dynamic EQ)** (comprimiendo o expandiendo rangos de frecuencia específicos según un `Threshold`, `Dynamic Range`, `Attack` y `Release`).
- Integra **Crossover Digital** con pendientes Linkwitz-Riley de 12, 24 y 48 dB/oct, línea de corte interactiva en el gráfico y exportador de imágenes de alta resolución en PNG.
- Integra **Compensación Automática de Ganancia (Auto-Gain)** con estimación de sonoridad perceptual ponderada (ITU-R BS.1770 / RMS) y limitación de tasa de cambio (slew-rate) para evitar saltos y bombeo acústico.
- Visualiza simultáneamente el espectro **Pre-EQ**, **Post-EQ Bus A** y **Post-EQ Bus B** (con curvas suavizadas por splines cúbicos y efecto "tira elástica" anclada en 10 Hz) junto con las curvas de respuesta estática, excursión dinámica instantánea y osciloscopio en tiempo real sobre una interfaz oscura Apple Liquid Glass (`#0D0F14`).

---

## 2. Stack Tecnológico Obligatorio
- **Interfaz Gráfica:** `PyQt6` (Prohibido usar Tkinter o Matplotlib por rendimiento/FPS).
- **Renderizado 2D Vectorial y OpenGL:** `pyqtgraph` con `useOpenGL=True`, soporte de `QOpenGLWidget` (`MiniScopeGL`) y desacople de VSync DWM (`QSurfaceFormat.setSwapInterval(0)` y `wglSwapIntervalEXT(0)` en 64 bits).
- **Motor de Audio en Tiempo Real:** `sounddevice` (PortAudio, `float32`, latencia ultra baja con bloques de `512` a `2048` muestras).
- **Procesamiento Matemático y DSP:**
  - `numpy` (cálculo vectorial de coeficientes Biquad del *Audio EQ Cookbook*, envolventes de pico/RMS, detección True Peak y `rfft`).
  - `scipy.signal.sosfilt` (filtrado IIR en cascada de secciones de segundo orden con preservación continua del vector de estado `zi` entre bloques).
  - `scipy.signal.sosfreqz` (evaluación de la respuesta en frecuencia compleja para renderizado de curvas estáticas y dinámicas a 60+ FPS).
  - `scipy.interpolate.make_interp_spline` (interpolación cúbica del espectro FFT).
- **Empaquetado y Distribución:**
  - `PyInstaller` con especificación `FFTEQ&LIM.spec` (`--noconsole --onefile`).
  - `Inno Setup 6 / 7` (`setup.iss`, `setup.ino`, `build_installer.iss`) con compresión `lzma2/ultra64` y soporte bilingüe (Español / Inglés).

---

## 3. Arquitectura Modular y Multihilo

El proyecto está estructurado en componentes fuertemente desacoplados que operan en paralelo para garantizar **60+ FPS de refresco visual** y **0 dropouts de audio**:

### 1. `fft_analyzer.py` (Motor de Análisis Espectral Pre/Post-EQ a 60+ FPS)
- **Worker Autónomo Hop-Paced (`_hop_worker_thread`):** Opera a ~90 Hz de forma autónoma, tomando rebanadas de FFT superpuestas desde buffers circulares continuos (`RingBuffer`), desacoplado completamente de la frecuencia y tamaño de bloque del callback de audio.
- Resuelve definitivamente el problema de caída prematura a -120 dBFS (*decay starvation*) cuando el buffer de audio es pequeño (64/128/256/512) o grande (1024/2048).
- Gestiona 3 flujos simultáneos: `Input (Pre-EQ)`, `Output 1 (Bus A)` y `Output 2 (Bus B)`.
- Calcula `np.fft.rfft` con ventana Hanning, detector de envolvente asimétrico (ataque instantáneo + decaimiento configurable) e interpolación `make_interp_spline` con anclaje elástico en 10 Hz y Nyquist.

### 2. `eq_dsp.py` (Motor Matemático DSP, Filtros Biquad, Dynamic EQ, Crossover, AutoGain & Limiter)
- **Filtros Paramétricos Biquad:** Diseña filtros digitales en formato SOS (*Second-Order Sections*) mediante fórmulas cerradas del *Audio EQ Cookbook*. Soporta: Bell, Low Shelf, High Shelf, Low Cut (12-48 dB/oct), High Cut (12-48 dB/oct), Notch y Band Pass.
- **Dynamic EQ:** Sidechain Band-Pass auxiliar por banda, medición de nivel RMS en dBFS, seguidor de envolvente exponencial con ataque/relajación, cálculo de ganancia instantánea con rodilla suave (*soft knee*) y actualización de coeficientes sin reiniciar `zi`.
- **Crossover Digital Linkwitz-Riley (`DualOutputCrossover`):** Filtros paso-bajos y paso-altos acoplados con pendientes de 12, 24 y 48 dB/oct, suma acústica plana en la frecuencia de corte, inversión de polaridad ($180^\circ$) y ganancia independiente por salida.
- **Compensación Automática de Ganancia (`AutoGain`):** Seguimiento continuo de sonoridad diferencial (Pre vs Post) con filtrado perceptual, actualización suave mediante slew-rate de ganancia en dB y bypass instantáneo.
- **Limitador de Picos Master (`MasterLimiter`):** Seguidor de envolvente de ataque ultra rápido (lookahead) y relajación regulable sobre ambos buses de salida para garantizar 0 clipping digital. Emite la señal instantánea de Reducción de Ganancia (GR) en dB.

### 3. `eq_ui.py` (Interfaz Interactiva, Liquid Glass HUD, MiniScopeGL & Overlays)
- **Renderizado por Hardware OpenGL:** ViewBox de pyqtgraph integrado sobre contexto OpenGL.
- **Mini-Osciloscopio Hardware (`MiniScopeGL`):** Subclase directa de `QOpenGLWidget` que dibuja las formas de onda L/R y la envolvente magenta de reducción de ganancia (GR) en la barra superior. Al ser OpenGL nativo, erradica la serialización de repintado raster con DWM, eliminando la penalización de VSync a 30 FPS.
- **Línea de Corte de Crossover Interactiva:** Marcador vertical interactivo con arrastre horizontal, etiqueta flotante en Hz y ajuste en tiempo real.
- **Exportador de Imágenes de Crossover:** Renderizado fuera de pantalla en alta resolución (1920x1080) con escalas en dB, etiquetas de frecuencia y marca de agua institucional.
- **Apple Liquid Glass HUD:** Cápsulas translúcidas con efecto de desenfoque Gaussiano, borde especular y menús emergentes acelerados por GPU.

### 4. `main.py` (Orquestador de Aplicación, PortAudio Stream & Router de Doble Bus)
- Integra los streams de `sounddevice`, buffers circulares anti-jitter (`SecondaryOutputRingBuffer`), líneas de retardo estéreo (`StereoDelayLine`), sincronización automática de latencia (`⚡ Auto Sync`), y bucle principal con temporizador a 60+ FPS sin caídas de framerate.

---

## 4. Reglas Inquebrantables para Agentes y Desarrolladores (NO VIOLAR)

1. **Aislamiento Absoluto del Hilo de Audio (`audio_callback`):**
   - El callback de `sounddevice` corre en un hilo de tiempo real del driver de audio. **ESTRICTAMENTE PROHIBIDO** ejecutar `rfft`, `sosfreqz`, `make_interp_spline`, crear objetos Qt o manipular widgets de PyQt6 dentro del callback.
   - En el callback únicamente se ejecutan operaciones vectorizadas optimizadas en NumPy (`eq_engine.process_block`, retardo circular y enrutamiento a colas lock-free).
2. **Preservación Continua del Vector de Estado `zi`:**
   - Cada banda y filtro de crossover mantiene su matriz de estados internos `zi`.
   - Modificaciones de frecuencia, ganancia o Q recalculan los coeficientes Biquad $[b_0, b_1, b_2, 1, a_1, a_2]$ **sin reiniciar a cero** el vector `zi`, previniendo clics o transitorios audibles.
3. **Mantenimiento del Desacople de VSync y Renderizado a 60+ FPS:**
   - La ventana principal debe inicializar `QSurfaceFormat.setSwapInterval(0)` y llamar `wglSwapIntervalEXT(0)` en `showEvent`.
   - Los widgets superpuestos sobre el canvas de `pyqtgraph` con actualizaciones de alta frecuencia deben ser `QOpenGLWidget` (como `MiniScopeGL`), nunca `QWidget` con `QPainter` de software que fuercen al Desktop Window Manager (DWM) a reducir la tasa de refresco a la mitad (30 FPS).
   - Los desenfoques Gaussianos de fondo deben ser cacheados y redibujados solo por eventos de redimensionado o cambio de estado, no en cada frame del temporizador.
4. **Seguridad de Monitoreo (Inicio en MUTE):**
   - La aplicación arranca siempre con `self.is_muted = True` para evitar acoples acústicos.
   - El motor DSP procesa el bloque antes de la compuerta de Mute para que las curvas, el espectro FFT y los medidores dinámicos permanezcan activos en pantalla.
5. **Estética Visual Unificada:**
   - Toda la interfaz sigue la estética oscura profesional `#0D0F14` con componentes Liquid Glass translúcidos. Queda prohibido cualquier estilo de controles nativos claros de Windows o el antiguo modo retro VFD.
6. **Compilación de Instaladores y Versionado:**
   - El script oficial de Inno Setup es `setup.iss` (con alias `setup.ino` y `build_installer.iss`), configurado para generar `setup.exe` con compresión `lzma2/ultra64` y modo 64 bits (`x64compatible`).
7. **Compatibilidad con UAC y Directorios Protegidos (Program Files):**
   - Queda estrictamente prohibido usar llamadas relativas directas del tipo `os.makedirs("presets")` o asumir que `_app_dir()` es escribible.
   - Todo guardado de datos mutables (presets, `config.json`, archivos temporales) debe verificar `_is_writable_dir(_app_dir())`: si es escribible (entorno de desarrollo o modo portable), se utiliza la ruta local; si la aplicación está instalada en `C:\Program Files` u otra ruta protegida por UAC en Windows, los datos se deben escribir y cargar desde `%APPDATA%\BDJ_FFT_EQ\`.
8. **Arquitectura Dual-Input e Input 2 (Aux / Mic):**
   - El stream secundario (`stream_in2`) opera en un hilo independiente de PortAudio desacoplado del stream maestro mediante `AsyncInputRingBuffer` con protección de concurrencia (`threading.Lock()`).
   - El hilo maestro `audio_callback` debe vaciar y procesar `in2_block` continuamente siempre que `stream_in2` esté activo para alimentar el analizador espectral FFT y la medición acústica RTA, independientemente de si el ruteo acústico hacia los altavoces (`route_in2_to_a` / `route_in2_to_b`) está silenciado.
   - La apertura de Input 2 debe implementar tolerancia de fallos multinivel (baja latencia -> buffer por defecto -> sample rate nativo) y remuestreo dinámico en tiempo real si la frecuencia de la entrada difiere de `current_sr`.
