# SynXS Data Processing Suite

**SynXS** — An integrated desktop toolkit for **SAXS/WAXS data reduction, visualization, and analysis**, built with Python (Tkinter + Matplotlib + pyFAI).

**SynXS 数据处理套件** —— 面向小角/广角 X 射线散射（SAXS/WAXS）数据的一站式桌面工具，集二维图像积分、自动化绘图与多功能数据处理于一体。

---

## ✨ Features / 功能一览

### 1. Integration Tool — SAXS Batch Processing (pyFAI)
- Batch integrate 2D detector images (`.tif` / `.edf` / `.cbf`) into 1D curves
- Geometry files: pyFAI `.poni`, or a JSON dictionary with the same pyFAI keys (`.json` / `.cfg`). A `.cif` describes a crystal structure, not a detector geometry, and is **not** accepted as one — it can be loaded as a *calibrant reference* in the peak-finding tab instead
- Background subtraction (scale + shift) with **real-time shift preview**
- Optional mask file, sector (azimuth-limited) integration
- Optional third column with the **Poisson uncertainty σ** (`error_model='poisson'`)
- Every output `.dat` carries a **self-describing header** (geometry, mask, background parameters, binning, azimuth range, error model, version, timestamp)
- Multi-processing engine — isolated worker prevents memory leaks
- Optional per-file PNG plot export + full processing log
- **Integration Preview**: overlay integrated curves of selected images
- `benchmark.py` — per-frame timing breakdown (read / integrate / write / PNG), the numbers reported in the paper

### 2. Plotting Tool — Data Automation
- Batch import `.dat` files with custom prefix naming
- **2D waterfall plots**: multi-segment Y spacing, linear/log offsets, 5 color modes (incl. gradient-by-value with colorbar), fly-in animation, interactive curve replacement & annotation
- **3D plots**: line waterfall & surface (Matplotlib), or interactive **PyVista** engine with opacity slider
- **Filled contour plots**: fixed colorbar, progressive left-to-right animation, global color scale for cross-plot consistency
- **Curve slideshow**: auto/manual paging with adjustable interval
- Vector export: SVG / PDF / EPS

### 3. Data Processing Tool
- **Q → 2θ conversion** (custom wavelength, auto-skip out-of-range rows)
- **Automatic peak finding**: `scipy.signal.find_peaks` with prominence/width control, optional log-Y mode, manual point annotation, and **one-click PDF report** (cover + file list + per-file annotated plots & tables). A **crystallographic `.cif`** of a standard sample (LaB₆, CeO₂, Si, AgBh …) can be overlaid as reference: the allowed reflections are computed from the unit cell and Bravais centring and drawn with their *(hkl)* labels, in q or 2θ
- **WAXS crystallinity index**: Gaussian deconvolution of one amorphous halo + N crystalline Bragg peaks (`scipy.optimize.curve_fit`); automatic initial guesses with halo-zone exclusion; single-file preview or batch CI (%) for exposure/dose series; CSV + PDF reports
- **Degradation kinetics**: first-order decay fitting `y(t) = A·exp(−k·t) + C`, returning rate constant *k*, half-life *t*₁/₂ and R²; CSV or manual input
- **Multi-function batch processing**: drop third column, **Kratky** (`q²·I`) or **Porod** (`q⁴·I`) transform, remove `{}` characters — single file or whole directory

## 🚀 Installation / 安装

Requires **Python 3.8+**.

```bash
pip install -r requirements.txt
```

Optional (interactive 3D plots):

```bash
pip install pyvista
```

## ▶️ Usage / 使用

```bash
python synxs.py
```

A launcher window appears with three buttons:

| Button / 按钮 | Tool / 工具 |
|---|---|
| 🟢 Integration Tool (SAXS batch) | pyFAI-based batch image integration |
| 🔵 Plotting Tool (Data Automation) | Waterfall / 3D / contour / slideshow |
| 🟠 Data Processing | Q→2θ / Peak finder / Crystallinity / Kinetics / Batch utilities |

> **Calibration first**: use `pyfai-calib2` (launched from inside the Integration Tool) to generate a `.poni` geometry file before integrating images.
> **积分前请先用 pyfai-calib2 标定几何文件（.poni）。**

📖 Full manuals: [`SynXS_1.0_Manual_EN.docx`](SynXS_1.0_Manual_EN.docx) | [`SynXS_1.0_Manual_CN.docx`](SynXS_1.0_Manual_CN.docx)

## 📦 Dependencies / 依赖

| Package | Purpose |
|---|---|
| numpy | Array math |
| matplotlib | 2D/3D plotting |
| scipy | Peak finding, interpolation, curve fitting |
| fabio | Detector image I/O (.tif/.edf/.cbf) |
| pyFAI | Azimuthal integration |
| pyvista *(optional)* | Interactive 3D rendering |

## 📁 Project Structure / 项目结构

```
SynXS/
├── synxs.py            # Main program (all three tools)
├── requirements.txt    # Python dependencies
├── LICENSE             # MIT License
├── .gitignore
└── docs/
    ├── SynXS_1.0_Manual_EN.docx
    └── SynXS_1.0_Manual_CN.docx
```

## 📝 Notes / 说明

- OpenCL acceleration in pyFAI is deliberately **disabled** for GPU stability; integration runs on CPU.
- On Windows, run with `python synxs.py`; no build step required.
- The program checks dependencies at startup and offers copy-paste `pip install` commands for anything missing.

## 📄 License

Released under the [MIT License](LICENSE).

---

*SynXS Data Processing Suite · v1.1*
