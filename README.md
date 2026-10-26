<div align="center">
  <img src="assets/logo.png" width="160" alt="BDJ-FFT-EQ Logo" />
  <h1>BDJ-FFT-EQ & Limiter</h1>
  <p><strong>Professional 60+ FPS Real-Time Parametric & Dynamic Equalizer, Dual-Bus Routing, Digital Crossover Matrix, Intelligent Auto-Gain, Master Peak Limiter, 192 kHz DSP Engine, Multi-Resolution FFT Spectrum Analyzer & Hardware-Accelerated Oscilloscope for Windows</strong></p>
</div>

![Platform](https://img.shields.io/badge/Platform-Windows%2010%20%7C%2011-0078D6?logo=windows)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python)
![GUI](https://img.shields.io/badge/GUI-PyQt6-41CD52?logo=qt)
![Graphics](https://img.shields.io/badge/Graphics-pyqtgraph%20%7C%20OpenGL-00E5FF)
![DSP](https://img.shields.io/badge/DSP-NumPy%20%7C%20SciPy-013243)
![Sample Rate](https://img.shields.io/badge/Sample%20Rate-Up%20to%20192%20kHz-FF6F00)
![FPS](https://img.shields.io/badge/FPS-60%2B%20Rock--Solid-brightgreen)
![License](https://img.shields.io/badge/License-MIT-F5A623)

A professional, studio-grade standalone Parametric & Dynamic Equalizer with integrated real-time Windows audio routing, 60+ FPS hop-paced multi-resolution FFT spectrum analyzer, dual-bus output matrix, digital Linkwitz-Riley crossover with high-res PNG export, intelligent auto-gain loudness compensation, master peak limiter with live gain reduction envelope, and hardware-accelerated OpenGL oscilloscope.

Process, sculpt, crossover, and equalize Windows system audio in real time with ultra-low latency—**no DAW or plugin host required**.

---

## Key Features

### 1. 60+ FPS Hop-Paced Multi-Resolution FFT & Zero-Stutter OpenGL Engine
- **Autonomous 90 Hz FIFO Hop-Worker:** Completely decouples FFT spectral computation from soundcard driver buffer sizes (`64` to `2048` frames). High-frequency overlapping hop-slicing prevents curve starvation and artificial decay dips.
- **Hardware-Accelerated OpenGL Rendering:** Full GPU acceleration (`pyqtgraph.setConfigOptions(useOpenGL=True)`) delivering silky-smooth 60+ FPS visual rendering across any display refresh rate.
- **DWM VSync Decoupling:** Employs `QSurfaceFormat.setSwapInterval(0)` and 64-bit `wglSwapIntervalEXT(0)` to prevent Windows Desktop Window Manager from locking or halving framerates to 30 FPS.
- **Triple-Stream Real-Time Spectrum:**
  - `Input (Pre-EQ)`: White/gray spectral trace of unprocessed incoming audio.
  - `Output 1 (Bus A)`: Cyan spectral trace with custom fill showing post-EQ Bus A.
  - `Output 2 (Bus B)`: Emerald green spectral trace showing post-EQ Bus B.
- **Spline Smoothing & Elastic-Band Pinning:** High-fidelity cubic splines (`scipy.interpolate.make_interp_spline`) with elastic-band pinning at 10 Hz and Nyquist.
- **Zero-CPU Idle Optimization:** Shuts off FFT computation and graphic repaints during silence, dropping CPU usage to near 0%.

---

### 2. Digital Crossover Matrix & High-Res PNG Image Exporter
- **Linkwitz-Riley Filter Topologies:** Selectable `12 dB/oct` (2nd order), `24 dB/oct` (4th order), and `48 dB/oct` (8th order) crossover slopes providing flat acoustic summation at cutoff.
- **Interactive Graph Cutoff:** Visual vertical crossover marker line on the main EQ canvas. Click and drag horizontally to smoothly adjust the split frequency with real-time frequency readout and magnetic snapping.
- **Dual-Bus Split (Sub / Tops):** Route Low-Pass energy to Bus A (Subwoofers) and High-Pass energy to Bus B (Tops/Satellites) or vice versa.
- **Phase Inversion ($180^\circ$):** Individual phase flip toggles per bus to correct acoustic cancellations in real rooms.
- **High-Resolution PNG Exporter (`📸 Exportar PNG`):** Generates clean, studio-grade 1920x1080 graphic snapshots of the active crossover response curves with labeled frequency grid, dB scales, and metadata watermarks for documentation and sound system tuning reports.

---

### 3. Intelligent Auto-Gain System (`AutoGain`)
- **Dual-Bus Loudness Balancing:** Real-time loudness tracking across Bus A and Bus B comparing Pre-EQ input energy with Post-EQ filtered output.
- **Perceptual Weighting:** Hybrid RMS and ITU-R BS.1770-style psychoacoustic weighting curve ensuring accurate perceived loudness matching rather than naive peak normalization.
- **Slew-Rate Limiting:** Smooth gain transition curves eliminate audible pumping, zipper noise, or abrupt jumps during aggressive EQ boosting or cutting.
- **Liquid Glass Top-Bar Pill:** Dedicated status indicator with live compensation readouts (`±X.X dB`), calibration button, and instant bypass toggle.

---

### 4. Studio Parametric & Dynamic EQ Engine
- **Vectorized Biquad DSP:** Pure NumPy implementation of Second-Order Sections (SOS) using Audio EQ Cookbook (Robert Bristow-Johnson) filter topologies evaluated directly with complex phasors ($z^{-1}, z^{-2}$).
- **High-Resolution Sample Rate Support:** Native real-time DSP at `44.1 kHz`, `48 kHz`, `88.2 kHz`, `96 kHz`, `176.4 kHz`, and `192 kHz` with dynamic Nyquist axis auto-scaling up to $96\text{ kHz}$ ($50\text{k}$ and $90\text{k}$ logarithmic ticks).
- **Dynamic Block Size Scaling:** Automatically scales processing blocks (`512` at 44.1/48k, `1024` at 88.2/96k, `2048` at 176.4/192k) to maintain a constant ~10.6 ms real-time buffer latency across all resolutions.
- **7 Filter Shapes:**
  - `Bell` (parametric peak)
  - `Low Shelf` & `High Shelf`
  - `Low Cut` (High Pass) & `High Cut` (Low Pass) with slopes: `12 dB/oct`, `24 dB/oct`, `36 dB/oct`, `48 dB/oct`
  - `Notch` (band reject) & `Band Pass`
- **Per-Channel Processing Modes:**
  - `Stereo (LR)`: Symmetrical filtering across both stereo channels.
  - `Left (L)` / `Right (R)`: Selective filtering on one channel while preserving the other.
  - `Mid (M)` / `Side (S)`: Mathematically exact Mid/Side encoding ($M = 0.5(L+R), S = 0.5(L-R)$) for spatial acoustic sculpting.
- **Per-Band Dynamic EQ:**
  - Independent **Threshold** (`THR`, $-70$ to $0\text{ dBFS}$) with live signal meter.
  - **Dynamic Range** ($-30$ to $+30\text{ dB}$) relative to static gain.
  - Adjustable **Attack** ($0.5$ to $500\text{ ms}$) and **Release** ($5$ to $2000\text{ ms}$) ballistics.
  - Animated dynamic excursion curve and shaded area fill (`FillBetweenItem`) showing instantaneous compression or expansion.
- **Interactive Ghost Preview Curve:** Gradual fade-in curve preview on mouse hover; double-clicking automatically instantiates an EQ band matching the preview.
- **Ergonomic Node Callout & Floating HUD:** Attached quick-action callout and floating Apple Liquid Glass HUD with convex GAIN knob arch, outer dynamic range ring, and radial handle tabs.

---

### 5. Master Peak Limiter & Hardware-Accelerated Oscilloscope
- **Studio Peak Limiter (`eq_dsp.py`):** Vectorized NumPy envelope follower with adjustable `Ceiling` (-0.1 dB), `Threshold` (0.0 dB), `Attack` (2.0 ms), and `Release` (50.0 ms) applied independently to both output buses (`Out A` and `Out B`) to prevent clipping.
- **Live Gain Reduction (GR) Envelope:** Real-time purple/magenta gain reduction trace rendered directly in the oscilloscope waveform.
- **`MiniScopeGL` Hardware Capsule:** True `QOpenGLWidget` mini oscilloscope in the top bar displaying live waveforms and limiter gain reduction without triggering DWM VSync penalties. Features adaptive luminous purple glow when limiting is active.
- **Expanded Analysis Panel with Auto-Headroom:** Clean oscilloscope panel displaying Left (cyan) and Right (gold) channels. Opening the panel smoothly expands FFT Y-axis headroom to `+18 dBFS` without cutting or glitching audio playback.
- **Time Window Selector:** Instant switching between `1s`, `5s`, and `10s` history buffers.

---

### 6. Dual-Bus Routing & Latency Auto-Compensation
- **Anti-Jitter Circular Buffer (`SecondaryOutputRingBuffer`):** Continuous 16x block size circular buffer absorbing Windows timing jitter between distinct soundcards.
- **Smooth Clock Drift Compensation:** Automatically compensates clock drift between two physical audio devices via gentle cubic/linear interpolation without audible pops or drops.
- **Acoustic Latency Auto-Compensation (`StereoDelayLine`):** Sample-accurate stereo delay line with circular buffer up to $250\text{ ms}$.
- **`⚡ Auto Sync` Button:** Automatically detects buffer latency and driver differentials, pre-populating Delay Out 1 (A) so both physical soundcards fire in perfect acoustic phase alignment.

---

### 7. Full-Bleed Studio UI & Apple Liquid Glass Aesthetics
- **100% Full-Bleed Canvas:** Zero margins (`0px`) on left, top, bottom, and right edges.
- **Unified Dual Right dB Scales:**
  - Far-Right Column (`#787C85`): Signal level scale ($0$ to $-120\text{ dBFS}$).
  - Inner-Adjacent Column (`#E0B84C` bold): EQ Gain scale aligned with horizontal grid lines.
  - Scale Toggle Badge: Interactive Apple Liquid Glass badge in the scale header toggling between $\pm 12\text{ dB}$ and $\pm 30\text{ dB}$.
- **Natural $-\infty$ Bottom Fade:** Canvas extends down to $-132\text{ dB}$ with a dark gradient vignette dissolving filter tails into the frame.
- **GPU OpenGL Translucent Menus:** Hardware-accelerated popup dialogs with Gaussian blur underlay, specular edge glow, and dark glass styling.

---

## Keyboard Shortcuts & Mouse Gestures

| Action | Control / Gesture |
| :--- | :--- |
| **Add EQ Band** | Double-click canvas or double-click ghost preview curve |
| **Delete Selected Band** | `Delete` key, `✕` in HUD, or `✕` in node callout |
| **Adjust Frequency & Gain** | Click and drag any band node on the graph |
| **Adjust Band Q** | Mouse wheel over node or drag `Q` knob |
| **Fine Adjustment** | Hold `Shift` while dragging any knob, slider, or cutoff line |
| **Reset Knob to Default** | Double-click any knob |
| **Adjust Dynamic Range** | Right-click + drag on `GAIN` knob, or drag the outer radial handle tab |
| **Toggle Dynamic EQ Mode** | Click `⚡ DYN` in HUD, right-click band node, or toggle in callout |
| **Solo Active Band** | Click `🎧 Solo` in HUD or callout |
| **Toggle Channel Mode (LR / L / R / M / S)** | Dropdown in HUD |
| **Toggle Output Routing (Out A / Out B / A+B)** | Dropdown in HUD |
| **Adjust Crossover Frequency** | Click and drag the vertical crossover cutoff marker line |
| **Export Crossover Response** | Click `📸 Exportar PNG` inside the `XOVER` menu |
| **Toggle Auto-Gain** | Click `AUTO-GAIN` pill in top bar |
| **Recalibrate Auto-Gain** | Click `Calibrar` inside the Auto-Gain popup menu |
| **Auto-Sync Outputs Latency** | Click `⚡ Auto Sync` in Audio I/O menu |
| **Toggle EQ Range ($\pm 12$ / $\pm 30$ dB)** | Click the `+12 dB` / `+30 dB` badge in the top-right scale column |
| **Toggle Oscilloscope** | Click the mini-scope canvas or `▾` button |
| **Fullscreen** | `F11` (or click `⛶`) |
| **Exit Fullscreen / Collapse Scope** | `Escape` |
| **Context Menu** | Right-click on background or band nodes |

---

## Project Structure

```
Analyzer/
├── assets/
│   ├── icon.ico                  # Multi-resolution application Windows icon
│   └── logo.png                  # Application emblem and header badge
├── build.bat                     # Automated one-click PyInstaller + Inno Setup build script
├── build_installer.iss           # Inno Setup compilation script (Instalador/setup.exe)
├── setup.iss                     # Root Inno Setup script (setup.exe)
├── setup.ino                     # Inno Setup script alias (setup.ino)
├── FFTEQ&LIM.spec                # PyInstaller build specification
├── config.json                   # User preferences, saved band states, delays & crossover settings
├── eq_dsp.py                     # Vectorized DSP biquad filter, Dynamic EQ, Crossover, AutoGain & Limiter
├── eq_ui.py                      # Liquid Glass UI overlay, MiniScopeGL, HUD, Crossover line & Callout
├── fft_analyzer.py               # 60+ FPS hop-paced 3-stream FFT spectrum analyzer engine
├── main.py                       # Application orchestrator, audio router, delay lines & OpenGL window
├── requirements.txt              # Python package dependencies
└── README.md                     # Comprehensive project documentation
```

---

## Installation & Running from Source

### Prerequisites
- **Operating System:** Windows 10 or Windows 11 (64-bit)
- **Python:** Python 3.10, 3.11, 3.12, 3.13, or 3.14

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/eOlmosDJ/StandAlone-FFT-visualizer-and-EQ.git
   cd StandAlone-FFT-visualizer-and-EQ
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch the application:
   ```bash
   python main.py
   ```

---

## Building Standalone Executable & Installer

### 1. One-Click Build Script (`build.bat`)
Run the provided automated build script:
```cmd
build.bat
```
This script will:
1. Validate Python dependencies.
2. Compile the standalone executable `dist\FFTEQ&LIM.exe` using PyInstaller.
3. Automatically search for Inno Setup 6 / 7 (`ISCC.exe`) and compile `setup.exe`.

### 2. Manual PyInstaller Compilation
```bash
pyinstaller "FFTEQ&LIM.spec" --noconfirm
```
The standalone single-file executable will be generated at `dist/FFTEQ&LIM.exe`.

### 3. Compiling the Windows Installer (`setup.exe`)
With [Inno Setup 6 or 7](https://jrsoftware.org/isdl.php) installed:
```bash
"C:\Program Files\Inno Setup 7\ISCC.exe" setup.iss
```
Or open `setup.iss` (or `setup.ino`) in the Inno Setup IDE and press **Compile (Ctrl+F9)**. The installer `setup.exe` will be generated in the root directory.

---

## Audio Routing Tip (Virtual Audio Cable)

To equalize your entire Windows system audio without a DAW:
1. Install a virtual audio device (such as **VB-Audio Virtual Cable**).
2. Set Windows Default Output Device to **CABLE Input (VB-Audio Virtual Cable)**.
3. In BDJ-FFT-EQ, click **🎧 Audio I/O** and select:
   - **Input Device:** `[MME/WASAPI] CABLE Output (VB-Audio Virtual Cable)`
   - **Output Device:** Your primary physical speakers or headphones (`Bus A`).
   - **Secondary Output (Optional):** A second soundcard or audio interface (`Bus B`).
4. Click `⚡ Auto Sync` to automatically align output latency across devices.
5. Click **🔇 Muted (Activar)** to switch to **🔊 Audio ON**.

---

## License

This project is licensed under the MIT License - see the [LICENSE](file:///c:/Users/gauna/Desktop/Analyzer/LICENSE) file for details.
