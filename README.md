<div align="center">
  <img src="assets/logo.png" width="160" alt="BDJ-FFT-EQ Logo" />
  <h1>BDJ-FFT-EQ</h1>
  <p><strong>Real-Time Parametric & Dynamic Equalizer, Dual-Bus Routing, 192 kHz DSP Engine, FFT Spectrum Analyzer & Oscilloscope for Windows</strong></p>
</div>

![Platform](https://img.shields.io/badge/Platform-Windows%2010%20%7C%2011-0078D6?logo=windows)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python)
![GUI](https://img.shields.io/badge/GUI-PyQt6-41CD52?logo=qt)
![Graphics](https://img.shields.io/badge/Graphics-pyqtgraph%20%7C%20OpenGL-00E5FF)
![DSP](https://img.shields.io/badge/DSP-NumPy%20%7C%20SciPy-013243)
![Sample Rate](https://img.shields.io/badge/Sample%20Rate-Up%20to%20192%20kHz-FF6F00)
![License](https://img.shields.io/badge/License-MIT-F5A623)

A professional, studio-grade standalone Parametric & Dynamic Equalizer with integrated real-time system audio routing, high-resolution triple-stream FFT spectrum analyzer, dual-bus output matrix, latency auto-compensation, and scroll-blitted oscilloscope.

Process, sculpt, and equalize Windows system audio in real time with ultra-low latency—**no DAW or plugin host required**.

---

## Key Features

### 1. Studio-Grade Parametric & Dynamic EQ Engine
- **Vectorized Biquad DSP:** Pure NumPy implementation of Second-Order Sections (SOS) using Audio EQ Cookbook (Robert Bristow-Johnson) filter topologies evaluated directly with complex phasors ($z^{-1}, z^{-2}$).
- **High-Resolution Sample Rate Support:** Native real-time DSP at `44.1 kHz`, `48 kHz`, `88.2 kHz`, `96 kHz`, `176.4 kHz`, and `192 kHz` with dynamic Nyquist axis auto-scaling up to $96\text{ kHz}$ ($50\text{k}$ and $90\text{k}$ logarithmic ticks).
- **Dynamic Block Size Scaling:** Automatically scales processing blocks (`512` at 44.1/48k, `1024` at 88.2/96k, `2048` at 176.4/192k) to maintain a constant ~10.6 ms real-time buffer latency across all resolutions.
- **7 Filter Shapes:**
  - `Bell` (parametric peak)
  - `Low Shelf` & `High Shelf`
  - `Low Cut` (High Pass) & `High Cut` (Low Pass) with selectable slopes: `12 dB/oct`, `24 dB/oct`, `36 dB/oct`, `48 dB/oct`
  - `Notch` (band reject) & `Band Pass`
- **Per-Channel Processing Modes:**
  - `Stereo (LR)`: Filters both channels symmetrically.
  - `Left (L)` / `Right (R)`: Filters exclusively Left or Right channel, leaving the opposite channel pristine.
  - `Mid (M)` / `Side (S)`: Mathematically exact Mid/Side encoding ($M = 0.5(L+R), S = 0.5(L-R)$), filtering the targeted component with isolated filter states and reconstruction.
- **Dual-Bus Routing (Bus A / Bus B / Bus A+B):**
  - Assign individual bands to `Out A (1-2)`, `Out B (3-4)`, or `Out A+B`.
  - Supports 4-channel audio interfaces or routing to two distinct physical audio devices simultaneously.
  - **100% Isolated State Matrices:** Completely decoupled state arrays (`zi_A`, `zi_B`, `zi_mid_A`, `zi_side_A`, etc.) preventing any cross-contamination or phase distortion between outputs.
- **Full Dynamic EQ per Band:**
  - Independent **Threshold** (`THR`, $-70$ to $0\text{ dBFS}$) with live signal meter.
  - **Dynamic Range** ($-30$ to $+30\text{ dB}$) relative to static gain.
  - Adjustable **Attack** ($0.5$ to $500\text{ ms}$) and **Release** ($5$ to $2000\text{ ms}$) ballistics.
  - Real-time animated dynamic excursion curve and shaded area fill (`FillBetweenItem`) showing instantaneous compression or expansion.
- **Per-Band Controls:** On/Off toggle, Solo listening mode (`🎧 Solo`), custom band color swatches, filter shape selector, and direct node dragging on the canvas.
- **Master Controls:** Master Output Gain ($-24$ to $+24\text{ dB}$), global **Bypass EQ**, and one-click **Reset EQ**.

---

### 2. Output 2 Anti-Jitter Ring Buffer & Latency Auto-Compensation
- **Anti-Jitter Circular Buffer (`SecondaryOutputRingBuffer`):**
  - Continuous 16x block size circular buffer with thread-safe lock protection.
  - Initial pre-buffering cushion (`target_fill = 2 * block_size`, ~21 ms at 48k / ~10 ms at 96k) absorbing Windows timing jitter.
  - **Smooth Clock Drift Compensation:** If clocks drift between two soundcards, the buffer gently compresses (`np.interp` by 2 frames per block on excess) or stretches available samples on deficit without abrupt zero-filling or audible pops.
- **Acoustic Latency Auto-Compensation (`StereoDelayLine`):**
  - Sample-accurate stereo delay line with circular buffer up to $250\text{ ms}$ (zero CPU cost when delay is $0\text{ ms}$).
  - **`⚡ Auto Sync` Button:** Automatically detects buffer latency (`2 * block_size / sr`) and hardware driver differential (`sec_latency - main_latency`), pre-populating **Delay Out 1 (A)** so both physical outputs play in perfect acoustic phase alignment.
  - **Fine Delay Adjustment:** Dedicated controls in the Audio I/O menu for **Delay Out 1 (A)** and **Delay Out 2 (B)** ($0.0$ to $200.0\text{ ms}$ with $0.5\text{ ms}$ step).

---

### 3. Triple-Stream FFT Spectrum Analyzer & Master Curves
- **3 Simultaneous FFT Streams:**
  - `Input (Pre-EQ)`: Faint gray/white spectral trace showing original unprocessed incoming audio.
  - `Output 1 (Bus A)`: Cyan spectral trace with custom fill showing post-EQ Bus A.
  - `Output 2 (Bus B)`: Emerald green spectral trace showing post-EQ Bus B.
- **Smart Auto-Activation:** Output 2 curve automatically activates whenever a secondary output device is configured or any EQ band is routed to Bus B.
- **Color Customization:** Direct color pickers for all 3 spectrum curves and background presets located conveniently inside the **📊 Analyzer** menu.
- **Decoupled Ring Buffer:** Configurable FFT resolution from $512$ to $16,384$ samples (default $4096$) without affecting audio I/O latency.
- **Hardware-Accelerated 2D Rendering:** Hardware GPU acceleration (`OpenGL`) maintaining fluid 60 FPS.
- **Zero-CPU Idle Optimization:** Dynamic dirty-flag architecture stops computing FFT, cubic splines, and redraws when the audio signal is at rest in silence.

---

### 4. Interactive UX, Ghost Curve & Node Callout
- **Interactive Ghost Preview Curve:**
  - Appears smoothly with gradual fade-in when the mouse hovers still for ~1 second over the canvas.
  - Smoothly interpolates coordinates and fades out when moving or approaching existing nodes.
  - Double-clicking the ghost curve instantly creates a new EQ band matching the previewed shape and frequency.
- **Node Callout Menu:**
  - Ergonomic popup directly attached above the active band node with filter type silhouette icon, slope, frequency, gain, Q, power toggle, and delete button.
  - Unified geometric safe-zone margin ($45\text{ px}$) preventing accidental closure when navigating to controls.
- **Floating Band HUD:**
  - Large $88\text{ px}$ `GAIN` knob inside the top convex arch ("joroba"), wide $10\text{ px}$ outer Dynamic Range ring with radial handle tab, and clean bottom navigation bar.
- **Custom Dark Controls:**
  - 100% custom-styled dark comboboxes and popups (`#181A20` background, `#00E5FF` selection, frameless window lists) using Qt Fusion theme, completely eliminating white OS-native styling.

---

### 5. Ultra-Low CPU Oscilloscope (Scroll-Blitting)
- **Compact View:** Sleek Liquid Glass capsule in the top bar displaying the live waveform.
- **Expanded View:** $164\text{ px}$ deep analysis panel showing Left (cyan) and Right (gold) post-EQ channels with second grid markers.
- **Hardware Scroll-Blitting (`QPixmap.scroll`):** Instead of redrawing 10 seconds of waveform each frame, the offscreen canvas scrolls left by $dx$ pixels and only paints the newest vertical slice ($< 1\%$ CPU utilization).
- **Time Window Toggle:** Quick toggle between `5s` and `10s` history.

---

### 6. Full-Bleed Studio UI & Apple Liquid Glass Aesthetics
- **100% Full-Bleed Canvas:** Zero margins (`0px`) on left, top, bottom, and right. FFT and EQ curves seamlessly touch $x = 0$ on the left and the stereo meter on the right.
- **Unified Dual Right dB Scales:**
  - **Far-Right Column (`#787C85`):** Signal level scale ($0, -10, -20 \dots -120\text{ dBFS}$).
  - **Inner-Adjacent Column (`#E0B84C` bold):** EQ Gain scale aligned with horizontal grid lines.
  - **Scale Toggle Badge:** Interactive Apple Liquid Glass badge in the scale header toggling the visual gain range between $\pm 12\text{ dB}$ and $\pm 30\text{ dB}$.
- **Natural $-\infty$ Bottom Fade:** Canvas extends down to $-132\text{ dB}$ with a $45\text{ px}$ dark gradient vignette so filter slopes and FFT energy smoothly dissolve into the bottom edge.
- **Floating Top Bar:** Translucent vitreous pills with specular borders floating over the canvas, shielded by a top gradient mask.

---

## Keyboard Shortcuts & Mouse Gestures

| Action | Control / Gesture |
| :--- | :--- |
| **Add EQ Band** | Double-click canvas or double-click ghost curve |
| **Delete Selected Band** | `Delete` key, `✕` in HUD, or `✕` in node callout |
| **Adjust Frequency & Gain** | Click and drag any band node on the graph |
| **Adjust Band Q** | Mouse wheel over node or drag `Q` knob |
| **Fine Adjustment** | Hold `Shift` while dragging any knob or slider |
| **Reset Knob to Default** | Double-click any knob |
| **Adjust Dynamic Range** | Right-click + drag anywhere on `GAIN` knob, or drag the outer radial handle tab |
| **Toggle Dynamic EQ Mode** | Click `⚡ DYN` in HUD, right-click band node, or toggle in callout |
| **Solo Active Band** | Click `🎧 Solo` in HUD or callout |
| **Toggle Channel Mode (LR / L / R / M / S)** | Dropdown in HUD |
| **Toggle Output Routing (Out A / Out B / A+B)** | Dropdown in HUD |
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
│   ├── icon.ico                  # Application multi-resolution Windows icon
│   └── logo.png                  # Application emblem and visual badge
├── build.bat                     # Single-click PyInstaller build script
├── config.json                   # Auto-saved user preferences, delays, and band states
├── eq_dsp.py                     # Vectorized DSP biquad filter, Mid/Side & dynamic EQ engine
├── eq_ui.py                      # BDJ-FFT-EQ overlay, Liquid Glass widgets, node callout & HUD
├── fft_analyzer.py               # 3-stream rolling ring buffer FFT spectrum analyzer
├── main.py                       # Main application window, delay lines & PortAudio stream manager
├── FIRST RELEASE.spec            # PyInstaller build specification
├── requirements.txt              # Python package dependencies
└── README.md                     # Project documentation
```

---

## Installation & Running

### Prerequisites
- **Operating System:** Windows 10 or Windows 11 (64-bit)
- **Python:** Python 3.10, 3.11, 3.12, 3.13, or 3.14

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/eOlmosDJ/StandAlone-FFT-visualizer-and-EQ.git
   cd StandAlone-FFT-visualizer-and-EQ
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch the application:
   ```bash
   python main.py
   ```

---

## Building Standalone Executable (.exe)

To build a standalone, single-file Windows executable (`dist/FIRST RELEASE.exe`):

```cmd
build.bat
```

Or run PyInstaller directly:
```bash
pyinstaller "FIRST RELEASE.spec" --noconfirm
```

The resulting executable will be created in the `dist/` directory.

---

## Audio Routing Tip (Virtual Audio Cable)

To equalize your entire Windows system audio:
1. Install a virtual audio device (such as **VB-Audio Virtual Cable**).
2. Set Windows Default Output Device to **CABLE Input (VB-Audio Virtual Cable)**.
3. In this application, click **🎧 Audio I/O** and select:
   - **Input Device:** `[MME/WASAPI] CABLE Output (VB-Audio Virtual Cable)`
   - **Output Device:** Your primary physical speakers or headphones (`Bus A`).
   - **Secondary Output (Optional):** A second soundcard or audio interface (`Bus B`).
4. Click `⚡ Auto Sync` to align output latency across devices.
5. Click **🔇 Muted (Activar)** to enable **🔊 Audio ON**.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
