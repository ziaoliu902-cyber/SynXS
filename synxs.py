#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SynXS Data Processing Suite
===========================
Contains three independent tools:
1. Integration Tool (SAXS batch processing) – integrate 2D images using pyFAI
2. Plotting Tool (data automation) – draw 2D waterfall, 3D, filled contour plots, etc.
3. Data Processing Tool – Q→2θ conversion, automatic peak finding report, multi-function data processing
Shows a selection interface at startup to enter each tool separately.

Supports geometry calibration file formats: .poni, .json, .cfg, .cif
"""

import sys
import tkinter as tk
from tkinter import messagebox, ttk, filedialog, simpledialog, colorchooser
import webbrowser
import os
import threading
import subprocess
import glob
import gc
import multiprocessing as mp
from datetime import datetime
import traceback
import json

# ==================== Set Matplotlib backend (must be before importing pyplot) ====================
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid GUI conflicts and memory leaks

# ==================== Dependency check ====================
def check_dependencies():
    required_packages = {
        'numpy': ('numpy', 'https://pypi.org/project/numpy/', 'NumPy'),
        'matplotlib': ('matplotlib', 'https://pypi.org/project/matplotlib/', 'Matplotlib'),
        'fabio': ('fabio', 'https://pypi.org/project/fabio/', 'FabIO'),
        'pyFAI': ('pyfai', 'https://pypi.org/project/pyfai/', 'pyFAI'),
        'scipy': ('scipy', 'https://pypi.org/project/scipy/', 'SciPy')
    }
    missing = []
    for import_name, (pip_name, url, display_name) in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append((import_name, pip_name, url, display_name))
    return missing, required_packages

class MissingDependenciesDialog:
    def __init__(self, parent, missing_list):
        self.parent = parent
        self.missing = missing_list
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Missing Dependencies")
        self.dialog.geometry("650x300")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self.dialog.protocol("WM_DELETE_WINDOW", self.on_close)

        tk.Label(self.dialog, text="The following required libraries are not installed, the program cannot run:",
                 font=('Arial', 12, 'bold')).pack(pady=10)

        frame = tk.Frame(self.dialog)
        frame.pack(fill='both', expand=True, padx=10, pady=5)

        tk.Label(frame, text="Library", font=('Arial', 10, 'bold')).grid(row=0, column=0, padx=5, pady=2, sticky='w')
        tk.Label(frame, text="Install command", font=('Arial', 10, 'bold')).grid(row=0, column=1, padx=5, pady=2, sticky='w')
        tk.Label(frame, text="Action", font=('Arial', 10, 'bold')).grid(row=0, column=2, padx=5, pady=2)

        for i, (import_name, pip_name, url, display_name) in enumerate(self.missing, start=1):
            tk.Label(frame, text=display_name).grid(row=i, column=0, padx=5, pady=2, sticky='w')
            cmd = f"pip install {pip_name}"
            tk.Label(frame, text=cmd, fg="blue", cursor="hand2").grid(row=i, column=1, padx=5, pady=2, sticky='w')
            btn = tk.Button(frame, text="Open PyPI", command=lambda u=url: webbrowser.open(u))
            btn.grid(row=i, column=2, padx=5, pady=2)

        btn_frame = tk.Frame(self.dialog)
        btn_frame.pack(pady=10)

        def copy_all():
            commands = "\n".join([f"pip install {p}" for _, p, _, _ in self.missing])
            self.dialog.clipboard_clear()
            self.dialog.clipboard_append(commands)
            messagebox.showinfo("Copy successful", "All installation commands have been copied to clipboard.")

        tk.Button(btn_frame, text="Copy all install commands", command=copy_all, width=20).pack(side='left', padx=5)
        tk.Button(btn_frame, text="Exit", command=self.on_close, width=10, bg="lightcoral").pack(side='left', padx=5)

        tk.Label(self.dialog, text="Please install the missing libraries and restart the program.", fg="gray").pack(pady=5)
        self.dialog.wait_window()

    def on_close(self):
        self.dialog.destroy()
        self.parent.quit()

# ==================== Import third-party libraries after dependency check ====================
missing, _ = check_dependencies()
if missing:
    root = tk.Tk()
    root.withdraw()
    MissingDependenciesDialog(root, missing)
    root.destroy()
    sys.exit(1)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button, Slider
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
import fabio
import pyFAI
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# ===== Disable PyFAI OpenCL acceleration to prevent GPU instability and crashes =====
pyFAI.use_opencl = False
os.environ['PYFAI_OPENCL'] = 'False'

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy.interpolate import interp1d
from matplotlib.animation import FuncAnimation

# Try to import pyvista (optional, for interactive 3D plots)
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False

# ==================== Generic geometry loader (supports .poni / .json / .cfg / .cif) ====================
def load_geometry(file_path):
    """
    Load geometry calibration file, supports .poni / .json / .cfg / .cif etc.
    First try pyFAI.load(), then try manual JSON parsing or build Geometry.
    """
    # 1. First try to load with pyFAI directly (supports .poni and some .json)
    try:
        ai = pyFAI.load(file_path)
        return ai
    except Exception:
        pass

    # 2. Try to parse as JSON (for .json / .cfg pure JSON files)
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        # If JSON contains typical pyFAI geometry fields, try to build directly
        ai = pyFAI.AzimuthalIntegrator(**data)
        return ai
    except Exception:
        pass

    # 3. Try as CIF file (using pyFAI's CIF support or pycifrw)
    try:
        # pyFAI may have built-in CIF support
        ai = pyFAI.load(file_path)  # try again, some pyFAI versions may support
        return ai
    except Exception:
        pass

    try:
        from pycifrw import CifFile
        cf = CifFile(file_path)
        # Try to extract typical geometry parameters, provide a simple example here
        # In practice, mapping depends on the CIF content
        # If parsing fails, raise exception
        raise NotImplementedError("CIF parsing requires custom mapping, refer to pyFAI documentation")
    except Exception:
        pass

    # 4. If all fail, raise clear error
    raise Exception(f"Unable to parse geometry file: {file_path}\n"
                     "Supported formats: pyFAI .poni, .json, pure JSON dictionary (.cfg), and some .cif")

# ==================== Plotting tool global variables ====================
data_waves = {}
last_prefix = None
last_count = 0
matrices = {}

# ==================== Plotting tool helper functions ====================
def ask_number(title, prompt, default=None, vartype=float):
    if vartype == int:
        return simpledialog.askinteger(title, prompt, initialvalue=default)
    else:
        return simpledialog.askfloat(title, prompt, initialvalue=default)

def ask_choice(title, prompt, choices, default_index=0):
    dialog = tk.Toplevel()
    dialog.title(title)
    dialog.transient()
    dialog.grab_set()
    dialog.geometry("300x150")
    dialog.resizable(False, False)

    tk.Label(dialog, text=prompt).pack(pady=10)
    var = tk.StringVar(value=choices[default_index])
    combo = ttk.Combobox(dialog, textvariable=var, values=choices, state="readonly")
    combo.pack(pady=5)

    result = None

    def on_ok():
        nonlocal result
        result = var.get()
        dialog.destroy()

    def on_cancel():
        dialog.destroy()

    btn_frame = tk.Frame(dialog)
    btn_frame.pack(pady=10)
    tk.Button(btn_frame, text="OK", command=on_ok, width=10).pack(side='left', padx=5)
    tk.Button(btn_frame, text="Cancel", command=on_cancel, width=10).pack(side='left', padx=5)
    dialog.wait_window()
    return result

def show_mode_dialog():
    dialog = tk.Toplevel()
    dialog.title("Select import mode")
    dialog.transient()
    dialog.grab_set()
    dialog.geometry("300x150")
    dialog.resizable(False, False)
    tk.Label(dialog, text="Please select import mode:").pack(pady=10)
    mode_var = tk.StringVar(value="folder")
    combo = ttk.Combobox(dialog, textvariable=mode_var, values=["Folder mode", "File mode"], state="readonly")
    combo.pack(pady=5)
    result = None
    def on_ok():
        nonlocal result
        result = "folder" if mode_var.get() == "Folder mode" else "files"
        dialog.destroy()
    def on_cancel():
        dialog.destroy()
    btn_frame = tk.Frame(dialog)
    btn_frame.pack(pady=10)
    tk.Button(btn_frame, text="OK", command=on_ok, width=10).pack(side='left', padx=5)
    tk.Button(btn_frame, text="Cancel", command=on_cancel, width=10).pack(side='left', padx=5)
    dialog.wait_window()
    return result

def get_segment_colors(values, cmap_name='viridis'):
    norm = plt.Normalize(vmin=np.min(values), vmax=np.max(values))
    sm = cm.ScalarMappable(norm=norm, cmap=cmap_name)
    return sm.to_rgba(values)

# ==================== Plotting tool functions ====================
def load_data():
    global last_prefix, last_count, data_waves
    prefix = simpledialog.askstring("Batch import", "Enter data prefix:", initialvalue="10")
    if not prefix:
        return
    mode = show_mode_dialog()
    if mode is None:
        return
    file_list = []
    if mode == "folder":
        folder = filedialog.askdirectory(title="Select folder containing .dat files")
        if not folder:
            return
        file_list = glob.glob(os.path.join(folder, "*.dat"))
        if not file_list:
            messagebox.showerror("Error", "No .dat files found in folder")
            return
        file_list.sort()
    else:
        files = filedialog.askopenfilenames(title="Select .dat files",
                                            filetypes=[("DAT files", "*.dat"), ("All files", "*.*")])
        if not files:
            return
        file_list = list(files)
        file_list.sort()

    num_files = len(file_list)
    for i, filepath in enumerate(file_list):
        try:
            data = np.loadtxt(filepath)
            if data.ndim != 2 or data.shape[1] < 2:
                continue
            x = data[:, 0]
            y = data[:, 1]
            idx = i + 1
            data_waves[f"{prefix}-{idx}x"] = x
            data_waves[f"{prefix}-{idx}y"] = y
        except Exception as e:
            print(f"Failed to read file {filepath}: {e}")
            continue
    last_prefix = prefix
    last_count = num_files
    messagebox.showinfo("Complete", f"Import completed. Prefix: {prefix} | Number of files: {num_files}")

def draw_2d():
    global data_waves, last_prefix
    if not data_waves:
        messagebox.showerror("Error", "No data available, please import data first")
        return

    # Temporarily switch backend to support interactive plotting
    plt.switch_backend('TkAgg')
    
    default_prefix = last_prefix if last_prefix else "10"
    prefix = simpledialog.askstring("Basic settings", "Data prefix:", initialvalue=default_prefix)
    if not prefix: 
        plt.switch_backend('Agg')
        return
    startL = ask_number("Basic settings", "Start index:", default=1, vartype=int)
    if startL is None: 
        plt.switch_backend('Agg')
        return
    endL = ask_number("Basic settings", "End index:", default=121, vartype=int)
    if endL is None: 
        plt.switch_backend('Agg')
        return
    numS = ask_number("Basic settings", "How many segments to plot? (1-10):", default=1, vartype=int)
    if numS is None: 
        plt.switch_backend('Agg')
        return
    numS = max(1, min(10, numS))
    xOff = ask_number("Basic settings", "X-axis offset (horizontal):", default=0.01, vartype=float)
    if xOff is None: 
        plt.switch_backend('Agg')
        return
    graph_name = simpledialog.askstring("Basic settings", "Window title (no spaces/hyphens):", initialvalue="Waterfall_1")
    if not graph_name: 
        plt.switch_backend('Agg')
        return

    bounds = [0] * numS
    steps = [0] * numS
    for i in range(numS):
        if i < numS - 1:
            bound = ask_number(f"Segment settings {i+1}", f"End index for segment {i+1}:", default=100, vartype=int)
            if bound is None: 
                plt.switch_backend('Agg')
                return
            step = ask_number(f"Segment settings {i+1}", f"Y spacing for segment {i+1}:", default=1, vartype=float)
            if step is None: 
                plt.switch_backend('Agg')
                return
            bounds[i] = bound
            steps[i] = step
        else:
            step = ask_number("Last segment settings", "Y spacing for last segment:", default=1, vartype=float)
            if step is None: 
                plt.switch_backend('Agg')
                return
            steps[i] = step
            bounds[i] = float('inf')

    color_choice = ask_choice("Appearance settings", "Select color mode:", 
                              ["All black", "Cool to warm", "Warm to cool", "High-contrast cycle", "Gradient (by Y value)"], default_index=3)
    if color_choice is None: 
        plt.switch_backend('Agg')
        return
    cMode_map = {"All black":0, "Cool to warm":1, "Warm to cool":2, "High-contrast cycle":3, "Gradient (by Y value)":4}
    cMode = cMode_map[color_choice]

    initial_linewidth = 1.0

    offset_mode_choice = ask_choice("Offset mode", "Select offset type:",
                                    ["Linear offset (addition)", "Logarithmic offset (multiplication)"], default_index=0)
    if offset_mode_choice is None: 
        plt.switch_backend('Agg')
        return
    offset_mode = 0 if offset_mode_choice == "Linear offset (addition)" else 1

    anim_choice = ask_choice("Appearance settings", "Enable fly-in animation?", ["No", "Yes"], default_index=1)
    if anim_choice is None: 
        plt.switch_backend('Agg')
        return
    isAnim = 1 if anim_choice == "Yes" else 0
    if isAnim == 1:
        dTime = ask_number("Appearance settings", "Delay time (seconds):", default=0.03, vartype=float)
        if dTime is None: 
            plt.switch_backend('Agg')
            return
    else:
        dTime = 0.0

    set_axis_choice = ask_choice("Axis presets", "Set axis properties now?", ["No", "Yes"], default_index=0)
    if set_axis_choice is None: 
        plt.switch_backend('Agg')
        return
    set_axis = 1 if set_axis_choice == "Yes" else 0

    axis_settings = {}
    if set_axis == 1:
        xmin = simpledialog.askfloat("X-axis range", "X min (Cancel to keep auto):", initialvalue=None)
        if xmin is not None:
            xmax = simpledialog.askfloat("X-axis range", "X max:", initialvalue=None)
            if xmax is not None:
                axis_settings['xlim'] = (xmin, xmax)
        ymin = simpledialog.askfloat("Y-axis range", "Y min (Cancel to keep auto):", initialvalue=None)
        if ymin is not None:
            ymax = simpledialog.askfloat("Y-axis range", "Y max:", initialvalue=None)
            if ymax is not None:
                axis_settings['ylim'] = (ymin, ymax)
        xscale_choice = ask_choice("X-axis scale", "X-axis scale type:\n(Note: log scale requires data > 0)", 
                                   ["Linear", "Log"], default_index=0)
        if xscale_choice is not None:
            if xscale_choice == "Log":
                axis_settings['xscale'] = 'log'
                if any(np.min(data_waves[key]) <= 0 for key in data_waves if key.endswith('x')):
                    messagebox.showwarning("Data warning", 
                        "Some X data ≤0 detected, log scale may not display correctly.\n"
                        "Suggestions:\n1. Ensure lower limit >0 when setting axis range\n"
                        "2. Pre-process data by shifting")
        yscale_choice = ask_choice("Y-axis scale", "Y-axis scale type:\n(Note: log scale requires data > 0)", 
                                   ["Linear", "Log"], default_index=0)
        if yscale_choice is not None:
            if yscale_choice == "Log":
                axis_settings['yscale'] = 'log'
                if any(np.min(data_waves[key]) <= 0 for key in data_waves if key.endswith('y')):
                    messagebox.showwarning("Data warning", 
                        "Some Y data ≤0 detected, log scale may not display correctly.\n"
                        "Suggestions:\n1. Ensure lower limit >0 when setting axis range\n"
                        "2. Pre-process data by shifting")

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.canvas.manager.set_window_title(graph_name)

    if 'xlim' in axis_settings:
        ax.set_xlim(*axis_settings['xlim'])
    if 'ylim' in axis_settings:
        ax.set_ylim(*axis_settings['ylim'])
    if 'xscale' in axis_settings:
        ax.set_xscale(axis_settings['xscale'])
    if 'yscale' in axis_settings:
        ax.set_yscale(axis_settings['yscale'])

    seg_ends = []
    seg_steps = []
    for k in range(numS):
        if k < numS - 1:
            end = bounds[k]
        else:
            end = endL
        seg_ends.append(end)
        seg_steps.append(steps[k])
    cum_offsets = [0] * (numS + 1)
    for k in range(numS):
        if k == 0:
            seg_start = startL
        else:
            seg_start = seg_ends[k-1] + 1
        num_curves_in_seg = seg_ends[k] - seg_start + 1
        cum_offsets[k+1] = cum_offsets[k] + seg_steps[k] * num_curves_in_seg

    valid_curves = []
    for i in range(startL, endL + 1):
        x_key = f"{prefix}-{i}x"
        y_key = f"{prefix}-{i}y"
        if x_key not in data_waves or y_key not in data_waves:
            continue
        y = data_waves[y_key]
        seg_idx = 0
        for k in range(numS):
            if i <= seg_ends[k]:
                seg_idx = k
                break
        if seg_idx == 0:
            seg_start = startL
        else:
            seg_start = seg_ends[seg_idx-1] + 1
        offset = cum_offsets[seg_idx] + (i - seg_start) * seg_steps[seg_idx]
        y_max = np.max(y)
        valid_curves.append((i, offset, y_max))

    if not valid_curves:
        messagebox.showerror("Error", "No valid curves to plot")
        plt.switch_backend('Agg')
        return

    N = len(valid_curves)
    offsets_linear = [item[1] for item in valid_curves]
    total_linear_offset = max(offsets_linear)
    maxY = max([item[2] for item in valid_curves])

    if offset_mode == 1:
        if maxY <= 0:
            messagebox.showerror("Error", "Data contains non-positive values, cannot use log offset")
            plt.switch_backend('Agg')
            return
        target_factor_last = 1 + total_linear_offset / maxY
        if target_factor_last <= 1:
            target_factor_last = 10.0
        if total_linear_offset > 0:
            k_log = np.log10(target_factor_last) / total_linear_offset
        else:
            k_log = 0.0
    else:
        k_log = None

    def get_custom_norm_and_ticks(data_values):
        answer = simpledialog.askstring(
            "Custom colorbar",
            "Enter min, max and tick values (comma-separated)\n"
            "Example: 0,1           # only set range, ticks auto\n"
            "Example: 0,1,0.2,0.4,0.6,0.8  # set range and specify ticks\n"
            "Leave empty for default",
            parent=fig.canvas.manager.window
        )
        if not answer:
            return None
        parts = answer.replace(' ', '').split(',')
        try:
            nums = [float(x) for x in parts]
        except ValueError:
            messagebox.showerror("Error", "Invalid input format, using default")
            return None
        if len(nums) < 2:
            messagebox.showerror("Error", "Need at least min and max")
            return None
        vmin, vmax = nums[0], nums[1]
        if vmin >= vmax:
            messagebox.showerror("Error", "Min must be less than max")
            return None
        ticks = nums[2:] if len(nums) > 2 else None
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        return norm, ticks

    def add_colorbar_with_customization(data_values):
        custom = get_custom_norm_and_ticks(data_values)
        if custom:
            norm, ticks = custom
        else:
            norm = plt.Normalize(vmin=np.min(data_values), vmax=np.max(data_values))
            ticks = None
        sm = cm.ScalarMappable(norm=norm, cmap='viridis')
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
        if ticks is not None:
            cbar.set_ticks(ticks)
        cbar.set_label('Y Value')
        return cbar

    lines = []
    high_contrast_colors = [(1,0,0), (0,0,1), (0,0.8,0), (1,0.65,0), (0.75,0,0.75), (0,0.75,0.75)]
    current_cmode = cMode
    cbar = None
    original_single_mode = cMode if cMode != 4 else 0
    all_y_shifted_for_cbar = []

    for idx, (i, offset_linear, _) in enumerate(valid_curves):
        x_key = f"{prefix}-{i}x"
        y_key = f"{prefix}-{i}y"
        x = data_waves[x_key]
        y = data_waves[y_key]

        if offset_mode == 1:
            factor = 10.0 ** (k_log * offset_linear)
            y_shifted = y * factor
        else:
            y_shifted = y + offset_linear

        x_shifted = x + idx * xOff

        if current_cmode == 4:
            points = np.array([x_shifted, y_shifted]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            color_values = y_shifted[:-1]
            colors = get_segment_colors(color_values, cmap_name='viridis')
            lc = LineCollection(segments, colors=colors, linewidth=initial_linewidth,
                                label=f'Curve {i}', picker=True, pickradius=5)
            ax.add_collection(lc)
            lines.append(lc)
            all_y_shifted_for_cbar.extend(y_shifted)
        else:
            line, = ax.plot(x_shifted, y_shifted, label=f'Curve {i}', picker=True, pickradius=5, linewidth=initial_linewidth)
            if current_cmode == 0:
                color = (0,0,0)
            elif current_cmode == 1:
                ratio = (i - startL) / (endL - startL + 1)
                color = (ratio, 0, 1-ratio)
            elif current_cmode == 2:
                ratio = (i - startL) / (endL - startL + 1)
                color = (1-ratio, 0, ratio)
            else:
                col_idx = (i - startL) % 6
                color = high_contrast_colors[col_idx]
            line.set_color(color)
            lines.append(line)

    if current_cmode == 4 and all_y_shifted_for_cbar:
        cbar = add_colorbar_with_customization(all_y_shifted_for_cbar)

    if isAnim == 1:
        for line in lines:
            line.set_visible(False)
        fig.canvas.draw_idle()

    ax.set_xlabel("X")
    ax.set_ylabel("Y (shifted)")
    ax.set_title(f"Waterfall Plot (prefix {prefix})")
    if current_cmode == 4:
        ax.relim()
        ax.autoscale_view()

    texts = []
    selected_text = None
    selected_line = None
    adding_text_mode = False
    replace_mode = False
    text_bbox = None
    status_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top', ha='left',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def update_status():
        if adding_text_mode:
            status_text.set_text('Mode: Add Text (click on plot)')
        elif replace_mode:
            status_text.set_text('Mode: Replace Curve (click on a curve to replace)')
        elif selected_text:
            status_text.set_text(f'Selected Text: "{selected_text.get_text()}"  (use arrows to move, Del to delete, right-click to edit)')
        elif selected_line:
            if current_cmode == 4:
                status_text.set_text(f'Selected Curve {selected_line.get_label()} (cannot move Y in gradient mode)')
            else:
                status_text.set_text(f'Selected: {selected_line.get_label()}  (use up/down arrows to adjust Y)')
        else:
            status_text.set_text('')
        fig.canvas.draw_idle()

    def update_linewidth(val):
        for line in lines:
            try:
                line.set_linewidth(val)
            except AttributeError:
                pass
        fig.canvas.draw_idle()

    def enable_add_text(event):
        nonlocal adding_text_mode
        adding_text_mode = True
        fig.canvas.set_cursor(2)
        update_status()

    def batch_replace_curves():
        nonlocal lines, cbar, valid_curves
        global data_waves
        prefix_input = simpledialog.askstring("Batch replace", "Enter data prefix:", parent=fig.canvas.manager.window)
        if not prefix_input:
            return
        keys = [k for k in data_waves if k.startswith(f"{prefix_input}-")]
        if not keys:
            if not messagebox.askyesno("Confirm", f"No data for prefix '{prefix_input}', new curves will be created. Continue?", parent=fig.canvas.manager.window):
                return

        start_str = simpledialog.askstring("Batch replace", "Start index:", parent=fig.canvas.manager.window)
        if not start_str:
            return
        end_str = simpledialog.askstring("Batch replace", "End index:", parent=fig.canvas.manager.window)
        if not end_str:
            end_str = start_str
        try:
            start = int(start_str)
            end = int(end_str)
        except ValueError:
            messagebox.showerror("Error", "Index must be integer", parent=fig.canvas.manager.window)
            return
        if start > end:
            messagebox.showerror("Error", "Start index cannot be greater than end index", parent=fig.canvas.manager.window)
            return

        num_needed = end - start + 1
        files = filedialog.askopenfilenames(
            title=f"Select {num_needed} .dat files (count must match)",
            filetypes=[("DAT files", "*.dat"), ("All files", "*.*")],
            parent=fig.canvas.manager.window
        )
        if not files:
            return
        file_list = list(files)

        if len(file_list) != num_needed:
            if not messagebox.askyesno("Count mismatch",
                                       f"You selected {len(file_list)} files, but {num_needed} needed.\n"
                                       "Continue? (will fill in order, extra files ignored, insufficient skipped)",
                                       parent=fig.canvas.manager.window):
                return
            num_actual = min(len(file_list), num_needed)
            file_list = file_list[:num_actual]
            end = start + num_actual - 1
            messagebox.showinfo("Adjusted", f"Will process indices {start} to {end}", parent=fig.canvas.manager.window)

        file_list.sort()

        success_count = 0
        fail_count = 0
        for i, filepath in zip(range(start, end + 1), file_list):
            try:
                data = np.loadtxt(filepath)
                if data.ndim != 2 or data.shape[1] < 2:
                    raise Exception("File needs at least two columns")
                x = data[:, 0]
                y = data[:, 1]
                data_waves[f"{prefix_input}-{i}x"] = x
                data_waves[f"{prefix_input}-{i}y"] = y
                success_count += 1
            except Exception as e:
                print(f"Replace file {filepath} failed: {e}")
                fail_count += 1
                continue

        messagebox.showinfo("Complete",
                            f"Batch replace completed. Success: {success_count}, Failed: {fail_count}\n"
                            "Updating plot...",
                            parent=fig.canvas.manager.window)

        if prefix_input != prefix:
            messagebox.showinfo("Info", f"Data updated (prefix {prefix_input}), but current plot shows prefix {prefix}, plot unchanged.",
                                parent=fig.canvas.manager.window)
            return

        new_valid_curves = []
        for i in range(startL, endL + 1):
            x_key = f"{prefix}-{i}x"
            y_key = f"{prefix}-{i}y"
            if x_key not in data_waves or y_key not in data_waves:
                continue
            y = data_waves[y_key]
            seg_idx = 0
            for k in range(numS):
                if i <= seg_ends[k]:
                    seg_idx = k
                    break
            if seg_idx == 0:
                seg_start = startL
            else:
                seg_start = seg_ends[seg_idx-1] + 1
            offset = cum_offsets[seg_idx] + (i - seg_start) * seg_steps[seg_idx]
            y_max = np.max(y)
            new_valid_curves.append((i, offset, y_max))

        if not new_valid_curves:
            messagebox.showerror("Error", "No valid curves after update", parent=fig.canvas.manager.window)
            return

        for line in lines:
            try:
                line.remove()
            except:
                pass
        lines.clear()
        if cbar is not None:
            cbar.remove()
            cbar = None

        all_y_shifted_for_cbar = []
        for idx, (i, offset_linear, _) in enumerate(new_valid_curves):
            x_key = f"{prefix}-{i}x"
            y_key = f"{prefix}-{i}y"
            x = data_waves[x_key]
            y = data_waves[y_key]
            if offset_mode == 1:
                factor = 10.0 ** (k_log * offset_linear)
                y_shifted = y * factor
            else:
                y_shifted = y + offset_linear
            x_shifted = x + idx * xOff

            if current_cmode == 4:
                points = np.array([x_shifted, y_shifted]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                color_values = y_shifted[:-1]
                colors = get_segment_colors(color_values, cmap_name='viridis')
                lc = LineCollection(segments, colors=colors, linewidth=initial_linewidth,
                                    label=f'Curve {i}', picker=True, pickradius=5)
                ax.add_collection(lc)
                lines.append(lc)
                all_y_shifted_for_cbar.extend(y_shifted)
            else:
                line, = ax.plot(x_shifted, y_shifted, label=f'Curve {i}', picker=True, pickradius=5, linewidth=initial_linewidth)
                if current_cmode == 0:
                    color = (0,0,0)
                elif current_cmode == 1:
                    ratio = (i - startL) / (endL - startL + 1)
                    color = (ratio, 0, 1-ratio)
                elif current_cmode == 2:
                    ratio = (i - startL) / (endL - startL + 1)
                    color = (1-ratio, 0, ratio)
                else:
                    col_idx = (i - startL) % 6
                    color = high_contrast_colors[col_idx]
                line.set_color(color)
                lines.append(line)

        if current_cmode == 4 and all_y_shifted_for_cbar:
            cbar = add_colorbar_with_customization(all_y_shifted_for_cbar)

        if isAnim == 1:
            for line in lines:
                line.set_visible(False)

        valid_curves = new_valid_curves
        fig.canvas.draw_idle()

    def enable_replace(event):
        nonlocal replace_mode, adding_text_mode, selected_line, selected_text
        choice = ask_choice("Replace mode", "Select replace type:", ["Single replace", "Batch replace"], default_index=0)
        if choice is None:
            return
        if choice == "Single replace":
            adding_text_mode = False
            replace_mode = True
            if selected_line:
                try:
                    selected_line.set_linewidth(initial_linewidth)
                except AttributeError:
                    pass
                selected_line = None
            clear_text_selection()
            fig.canvas.set_cursor(2)
            update_status()
        else:
            batch_replace_curves()

    def create_text_at(x, y):
        try:
            fig.canvas.manager.window.tk.call('tk', 'windowingsystem')
        except:
            pass
        text_content = simpledialog.askstring("Add annotation", "Enter text content:", parent=fig.canvas.manager.window)
        if text_content:
            txt = ax.text(x, y, text_content, fontsize=10,
                          bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                          picker=True)
            texts.append(txt)
            fig.canvas.draw_idle()
        nonlocal adding_text_mode
        adding_text_mode = False
        fig.canvas.set_cursor(0)
        update_status()

    def clear_text_selection():
        nonlocal selected_text, text_bbox
        if text_bbox:
            text_bbox.remove()
            text_bbox = None
        selected_text = None
        update_status()

    def highlight_text(txt):
        nonlocal text_bbox
        if text_bbox:
            text_bbox.remove()
        bb = txt.get_window_extent(renderer=fig.canvas.get_renderer())
        inv = ax.transData.inverted()
        bbox_data = inv.transform(bb)
        x0, y0 = bbox_data[0]
        x1, y1 = bbox_data[1]
        width = x1 - x0
        height = y1 - y0
        rect = plt.Rectangle((x0, y0), width, height, linewidth=1, edgecolor='red', facecolor='none', linestyle='--')
        ax.add_patch(rect)
        text_bbox = rect
        fig.canvas.draw_idle()

    pick_happened = False

    def on_pick(event):
        nonlocal selected_line, selected_text, pick_happened, adding_text_mode, replace_mode

        if replace_mode:
            artist = event.artist
            if artist in lines:
                label = artist.get_label()
                try:
                    i = int(label.split()[1])
                except:
                    messagebox.showerror("Error", "Unable to identify curve index")
                    replace_mode = False
                    fig.canvas.set_cursor(0)
                    update_status()
                    return

                filepath = filedialog.askopenfilename(
                    title="Select .dat file to replace",
                    filetypes=[("DAT files", "*.dat"), ("All files", "*.*")],
                    parent=fig.canvas.manager.window
                )
                if not filepath:
                    replace_mode = False
                    fig.canvas.set_cursor(0)
                    update_status()
                    return

                try:
                    new_data = np.loadtxt(filepath)
                    if new_data.ndim != 2 or new_data.shape[1] < 2:
                        raise Exception("File needs at least two columns")
                    new_x = new_data[:, 0]
                    new_y = new_data[:, 1]
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to read file: {e}")
                    replace_mode = False
                    fig.canvas.set_cursor(0)
                    update_status()
                    return

                orig_x_key = f"{prefix}-{i}x"
                orig_y_key = f"{prefix}-{i}y"
                if orig_x_key not in data_waves or orig_y_key not in data_waves:
                    messagebox.showerror("Error", f"Original data for curve {i} not found")
                    replace_mode = False
                    fig.canvas.set_cursor(0)
                    update_status()
                    return
                orig_x = data_waves[orig_x_key]

                if len(new_x) != len(orig_x) or not np.allclose(new_x, orig_x, rtol=1e-5):
                    try:
                        f = interp1d(new_x, new_y, kind='linear', bounds_error=False, fill_value=0.0)
                        new_y_interp = f(orig_x)
                        if np.any(np.isnan(new_y_interp)):
                            messagebox.showwarning("Warning", "Interpolation produced NaN, filled with 0")
                            new_y_interp = np.nan_to_num(new_y_interp, 0)
                    except Exception as e:
                        messagebox.showerror("Error", f"Interpolation failed: {e}")
                        replace_mode = False
                        fig.canvas.set_cursor(0)
                        update_status()
                        return
                else:
                    new_y_interp = new_y

                data_waves[orig_y_key] = new_y_interp

                idx_in_valid = None
                offset = None
                for idx, (curve_i, off, _) in enumerate(valid_curves):
                    if curve_i == i:
                        idx_in_valid = idx
                        offset = off
                        break
                if idx_in_valid is None:
                    messagebox.showerror("Error", f"Curve {i} not in current plot range")
                    replace_mode = False
                    fig.canvas.set_cursor(0)
                    update_status()
                    return

                if offset_mode == 1:
                    factor = 10.0 ** (k_log * offset)
                    y_shifted = new_y_interp * factor
                else:
                    y_shifted = new_y_interp + offset
                x_shifted = orig_x + idx_in_valid * xOff

                if isinstance(artist, matplotlib.lines.Line2D):
                    artist.set_data(x_shifted, y_shifted)
                elif isinstance(artist, matplotlib.collections.LineCollection):
                    points = np.array([x_shifted, y_shifted]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    colors = get_segment_colors(y_shifted[:-1], cmap_name='viridis')
                    artist.set_segments(segments)
                    artist.set_color(colors)
                else:
                    messagebox.showerror("Error", "Unsupported curve type")
                    replace_mode = False
                    fig.canvas.set_cursor(0)
                    update_status()
                    return

                if current_cmode == 4:
                    messagebox.showinfo("Info", "Curve replaced. If colorbar range is inappropriate, click the Gradient button to regenerate.")

                fig.canvas.draw_idle()
                replace_mode = False
                fig.canvas.set_cursor(0)
                update_status()
                return
            else:
                messagebox.showinfo("Info", "Please click on a curve")
                replace_mode = False
                fig.canvas.set_cursor(0)
                update_status()
                return

        if adding_text_mode:
            return
        pick_happened = True
        artist = event.artist
        mouse_event = event.mouseevent
        if mouse_event is None:
            return
        button = mouse_event.button
        if artist in lines:
            clear_text_selection()
            if button == 1:
                if selected_line and selected_line != artist:
                    try:
                        selected_line.set_linewidth(initial_linewidth)
                    except AttributeError:
                        pass
                selected_line = artist
                try:
                    selected_line.set_linewidth(initial_linewidth * 2.5)
                except AttributeError:
                    pass
                update_status()
            elif button == 3:
                if current_cmode != 4:
                    color = colorchooser.askcolor(color=artist.get_color() if hasattr(artist, 'get_color') else None,
                                                  title=f"Choose color for {artist.get_label()}")[1]
                    if color:
                        if hasattr(artist, 'set_color'):
                            artist.set_color(color)
                            fig.canvas.draw_idle()
                else:
                    messagebox.showinfo("Info", "Gradient mode does not allow changing individual curve color")
        elif artist in texts:
            if selected_line:
                try:
                    selected_line.set_linewidth(initial_linewidth)
                except AttributeError:
                    pass
                selected_line = None
            if button == 1:
                clear_text_selection()
                selected_text = artist
                highlight_text(artist)
                update_status()
            elif button == 3:
                choice = tk.messagebox.askquestion("Text operation", "Edit content? (select No to delete)", 
                                                   icon='question', type='yesnocancel')
                if choice == 'yes':
                    new_content = simpledialog.askstring("Edit text", "New content:", 
                                                         initialvalue=artist.get_text())
                    if new_content is not None:
                        artist.set_text(new_content)
                        if selected_text is artist:
                            clear_text_selection()
                            selected_text = artist
                            highlight_text(artist)
                        fig.canvas.draw_idle()
                elif choice == 'no':
                    artist.remove()
                    texts.remove(artist)
                    if selected_text is artist:
                        clear_text_selection()
                    fig.canvas.draw_idle()

    def on_button_press(event):
        nonlocal selected_line, selected_text, pick_happened, adding_text_mode
        if event.inaxes is ax:
            if adding_text_mode:
                create_text_at(event.xdata, event.ydata)
            elif event.button == 1 and not pick_happened:
                if selected_line:
                    try:
                        selected_line.set_linewidth(initial_linewidth)
                    except AttributeError:
                        pass
                    selected_line = None
                clear_text_selection()
                update_status()
        pick_happened = False

    def on_key_press(event):
        nonlocal selected_line, selected_text
        if selected_text:
            step_x = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01
            step_y = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01
            x0, y0 = selected_text.get_position()
            if event.key == 'up':
                selected_text.set_position((x0, y0 + step_y))
            elif event.key == 'down':
                selected_text.set_position((x0, y0 - step_y))
            elif event.key == 'left':
                selected_text.set_position((x0 - step_x, y0))
            elif event.key == 'right':
                selected_text.set_position((x0 + step_x, y0))
            elif event.key == 'delete':
                selected_text.remove()
                texts.remove(selected_text)
                clear_text_selection()
            else:
                return
            if selected_text:
                clear_text_selection()
                selected_text = selected_text
                highlight_text(selected_text)
            fig.canvas.draw_idle()
        elif selected_line:
            if current_cmode == 4 or not hasattr(selected_line, 'get_ydata'):
                return
            if event.key not in ['up', 'down']:
                return
            ydata = selected_line.get_ydata()
            yscale = ax.get_yscale()
            ylim = ax.get_ylim()
            height = ylim[1] - ylim[0]
            if yscale == 'linear':
                delta = height * 0.01
                if event.key == 'up':
                    new_ydata = ydata + delta
                else:
                    new_ydata = ydata - delta
            else:
                factor = 1.01 if event.key == 'up' else 0.99
                new_ydata = ydata * factor
                if np.any(new_ydata <= 0):
                    messagebox.showwarning("Warning", "Adjusted data contains non-positive values, operation cancelled.")
                    return
            selected_line.set_ydata(new_ydata)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('pick_event', on_pick)
    fig.canvas.mpl_connect('button_press_event', on_button_press)
    fig.canvas.mpl_connect('key_press_event', on_key_press)

    plt.subplots_adjust(bottom=0.35)

    ax_xlog = fig.add_axes([0.10, 0.10, 0.12, 0.04])
    ax_ylog = fig.add_axes([0.25, 0.10, 0.12, 0.04])
    ax_auto = fig.add_axes([0.40, 0.10, 0.12, 0.04])
    ax_setx = fig.add_axes([0.55, 0.10, 0.12, 0.04])
    ax_sety = fig.add_axes([0.70, 0.10, 0.12, 0.04])

    btn_xlog = Button(ax_xlog, 'X Log')
    btn_ylog = Button(ax_ylog, 'Y Log')
    btn_auto = Button(ax_auto, 'Auto')
    btn_setx = Button(ax_setx, 'Set X')
    btn_sety = Button(ax_sety, 'Set Y')

    def toggle_xlog(event):
        current_xscale = ax.get_xscale()
        if current_xscale == 'linear':
            xlim = ax.get_xlim()
            if xlim[0] <= 0:
                all_x_positive = []
                for curve in valid_curves:
                    x_vals = data_waves[f"{prefix}-{curve[0]}x"]
                    pos_x = x_vals[x_vals > 0]
                    if len(pos_x) > 0:
                        all_x_positive.extend(pos_x)
                if all_x_positive:
                    new_min = min(all_x_positive) * 0.9
                    new_max = max(all_x_positive) * 1.1
                    ax.set_xlim(new_min, new_max)
                    messagebox.showinfo("X-axis adjustment", 
                        f"X-axis range adjusted to [{new_min:.4f}, {new_max:.4f}] to support log scale")
                else:
                    messagebox.showerror("Error", "No positive X data, cannot use log scale")
                    return
            ax.set_xscale('log')
        else:
            ax.set_xscale('linear')
        fig.canvas.draw()

    def toggle_ylog(event):
        current_yscale = ax.get_yscale()
        target_scale = 'linear' if current_yscale == 'log' else 'log'
        if target_scale == 'log':
            has_non_positive = False
            for line in lines:
                if hasattr(line, 'get_ydata'):
                    if np.any(line.get_ydata() <= 0):
                        has_non_positive = True
                        break
                elif hasattr(line, 'get_segments'):
                    messagebox.showerror("Error", "Gradient mode does not support switching Y-axis log scale")
                    return
            if has_non_positive:
                messagebox.showerror("Error", "Data contains non-positive values, cannot switch to log scale.")
                return
            ylim = ax.get_ylim()
            if ylim[0] <= 0:
                all_y = np.concatenate([line.get_ydata() for line in lines if hasattr(line, 'get_ydata')])
                pos_y = all_y[all_y > 0]
                if len(pos_y) > 0:
                    new_min = np.min(pos_y) * 0.9
                    new_max = np.max(pos_y) * 1.1
                    ax.set_ylim(new_min, new_max)
        ax.set_yscale(target_scale)
        fig.canvas.draw_idle()

    def set_xrange(event):
        current = ax.get_xlim()
        xmin = simpledialog.askfloat("Set X axis", "X min:", initialvalue=current[0])
        if xmin is not None:
            xmax = simpledialog.askfloat("Set X axis", "X max:", initialvalue=current[1])
            if xmax is not None:
                ax.set_xlim(xmin, xmax)
                fig.canvas.draw()

    def set_yrange(event):
        current = ax.get_ylim()
        ymin = simpledialog.askfloat("Set Y axis", "Y min:", initialvalue=current[0])
        if ymin is not None:
            ymax = simpledialog.askfloat("Set Y axis", "Y max:", initialvalue=current[1])
            if ymax is not None:
                ax.set_ylim(ymin, ymax)
                fig.canvas.draw()

    def auto_scale(event):
        ax.autoscale()
        fig.canvas.draw()

    btn_xlog.on_clicked(toggle_xlog)
    btn_ylog.on_clicked(toggle_ylog)
    btn_auto.on_clicked(auto_scale)
    btn_setx.on_clicked(set_xrange)
    btn_sety.on_clicked(set_yrange)

    ax_switch = fig.add_axes([0.85, 0.10, 0.12, 0.04])
    btn_switch = Button(ax_switch, 'Gradient' if current_cmode != 4 else 'Single')

    def toggle_color_mode(event):
        nonlocal lines, current_cmode, cbar, selected_line, selected_text, original_single_mode
        for line in lines:
            try:
                line.remove()
            except:
                pass
        lines.clear()
        if cbar is not None:
            cbar.remove()
            cbar = None
        if current_cmode == 4:
            new_mode = original_single_mode
            btn_switch.label.set_text('Gradient')
        else:
            new_mode = 4
            btn_switch.label.set_text('Single')
        current_cmode = new_mode
        all_y_shifted_for_cbar = []
        for idx, (i, offset_linear, _) in enumerate(valid_curves):
            x_key = f"{prefix}-{i}x"
            y_key = f"{prefix}-{i}y"
            x = data_waves[x_key]
            y = data_waves[y_key]
            if offset_mode == 1:
                factor = 10.0 ** (k_log * offset_linear)
                y_shifted = y * factor
            else:
                y_shifted = y + offset_linear
            x_shifted = x + idx * xOff
            if current_cmode == 4:
                points = np.array([x_shifted, y_shifted]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                color_values = y_shifted[:-1]
                colors = get_segment_colors(color_values, cmap_name='viridis')
                lc = LineCollection(segments, colors=colors, linewidth=initial_linewidth,
                                    label=f'Curve {i}', picker=True, pickradius=5)
                ax.add_collection(lc)
                lines.append(lc)
                all_y_shifted_for_cbar.extend(y_shifted)
            else:
                line, = ax.plot(x_shifted, y_shifted, label=f'Curve {i}', picker=True, pickradius=5, linewidth=initial_linewidth)
                if new_mode == 0:
                    color = (0,0,0)
                elif new_mode == 1:
                    ratio = (i - startL) / (endL - startL + 1)
                    color = (ratio, 0, 1-ratio)
                elif new_mode == 2:
                    ratio = (i - startL) / (endL - startL + 1)
                    color = (1-ratio, 0, ratio)
                else:
                    col_idx = (i - startL) % 6
                    color = high_contrast_colors[col_idx]
                line.set_color(color)
                lines.append(line)
        if current_cmode == 4 and all_y_shifted_for_cbar:
            cbar = add_colorbar_with_customization(all_y_shifted_for_cbar)
        if current_cmode == 4:
            ax.relim()
            ax.autoscale_view()
        if selected_line:
            selected_line = None
        if selected_text:
            clear_text_selection()

        if isAnim == 1:
            for line in lines:
                line.set_visible(False)

        update_status()
        fig.canvas.draw_idle()

    btn_switch.on_clicked(toggle_color_mode)

    ax_slider = fig.add_axes([0.10, 0.05, 0.35, 0.03])
    slider = Slider(ax=ax_slider, label='Line Width', valmin=0.5, valmax=5.0, valinit=initial_linewidth, valstep=0.1)
    slider.on_changed(update_linewidth)

    ax_textbtn = fig.add_axes([0.55, 0.05, 0.12, 0.04])
    btn_text = Button(ax_textbtn, 'Add Text')
    btn_text.on_clicked(enable_add_text)

    ax_replacebtn = fig.add_axes([0.68, 0.05, 0.12, 0.04])
    btn_replace = Button(ax_replacebtn, 'Replace')
    btn_replace.on_clicked(enable_replace)

    if isAnim == 1:
        ax_start = fig.add_axes([0.81, 0.05, 0.12, 0.04])
        btn_start = Button(ax_start, 'Fly-in')
        def start_animation(event):
            for line in lines:
                line.set_visible(False)
            fig.canvas.draw_idle()
            for line in lines:
                line.set_visible(True)
                fig.canvas.draw_idle()
                plt.pause(dTime)
        btn_start.on_clicked(start_animation)

    update_status()
    plt.show()
    
    # Restore backend
    plt.switch_backend('Agg')

def make_matrix(is_log):
    global data_waves, matrices, last_prefix
    if not data_waves:
        messagebox.showerror("Error", "No data available, please import data first")
        return
    default_prefix = last_prefix if last_prefix else "10"
    prefix = simpledialog.askstring("Generate matrix", "Enter prefix:", initialvalue=default_prefix)
    if not prefix:
        return
    y_keys = [key for key in data_waves if key.startswith(prefix + "-") and key.endswith("y")]
    y_keys.sort(key=lambda s: int(s.split('-')[1][:-1]))
    if not y_keys:
        messagebox.showerror("Error", f"No waveform data found for prefix {prefix}")
        return
    first_y = data_waves[y_keys[0]]
    n_points = len(first_y)
    n_curves = len(y_keys)
    matrix = np.zeros((n_points, n_curves))
    for j, y_key in enumerate(y_keys):
        y = data_waves[y_key]
        if len(y) != n_points:
            min_len = min(n_points, len(y))
            matrix[:min_len, j] = y[:min_len]
        else:
            if is_log == 1:
                matrix[:, j] = np.log(np.maximum(y, 1e-10))
            else:
                matrix[:, j] = y
    x_key = f"{prefix}-1x"
    if x_key in data_waves:
        x_coords = data_waves[x_key]
        if len(x_coords) != n_points:
            x_coords = x_coords[:n_points]
    else:
        x_coords = np.arange(n_points)
    matrices[prefix] = {'matrix': matrix, 'x': x_coords}
    msg = f"Matrix generation completed. Size: {n_points} x {n_curves}"
    if is_log:
        msg = "Log " + msg
    messagebox.showinfo("Complete", msg)

def show_status():
    global last_prefix, last_count
    if last_prefix:
        messagebox.showinfo("Import record", f"Last import prefix: {last_prefix}\nNumber of files: {last_count}")
    else:
        messagebox.showinfo("Import record", "No import record.")

def kill_all():
    global data_waves, matrices, last_prefix, last_count
    if messagebox.askyesno("Confirm", "Are you sure you want to clear all data?"):
        data_waves.clear()
        matrices.clear()
        last_prefix = None
        last_count = 0
        messagebox.showinfo("Complete", "All data cleared.")

# ==================== PyVista 3D plotting functions ====================
def draw_3d_waterfall_pyvista(curves, x_label, z_label, title, y_offset_base, x_offset,
                              segment_bounds=None, segment_steps=None):
    """
    Use PyVista to draw 3D line waterfall plot (no grid, only X and Z axes shown)
    """
    # Calculate Y positions (depth) for each curve
    y_positions = []
    cum_offset = 0
    current_segment = 0

    if segment_bounds and segment_steps:
        for idx, (i, x, y) in enumerate(curves):
            while current_segment < len(segment_bounds) and i > segment_bounds[current_segment]:
                current_segment += 1
            step = segment_steps[current_segment] if current_segment < len(segment_steps) else segment_steps[-1]
            y_positions.append(cum_offset)
            cum_offset += step
    else:
        for idx, (i, x, y) in enumerate(curves):
            y_positions.append(idx * y_offset_base)

    # Create plotter
    plotter = pv.Plotter(title=title, window_size=(900, 700))
    plotter.background_color = 'white'
    plotter.show_grid(visible=False)  # hide grid lines
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    for idx, (i, x, y) in enumerate(curves):
        y_pos = y_positions[idx]
        x_shifted = x + idx * x_offset
        
        points = np.c_[x_shifted, np.full_like(x, y_pos), y]
        poly = pv.lines_from_points(points)
        color = colors[idx % len(colors)]
        plotter.add_mesh(poly, color=color, line_width=3, label=f'Curve {i}')

    # Custom axes: only show X and Z axes, hide Y axis
    plotter.add_axes(
        xlabel=x_label,
        ylabel='',          # hide Y axis label
        zlabel=z_label,
        line_width=2,
        color='black',
        x_color='black',
        y_color='white',    # make Y axis white (hidden)
        z_color='black',
    )
    
    plotter.view_xz(negative=True)
    plotter.show()
    return plotter

def draw_3d_surface_pyvista(X, Y, Z, x_label, y_label, z_label, title):
    """
    Use PyVista to draw 3D surface plot with custom axis scaling and real-time opacity slider
    """
    # Ask user for axis scaling factors
    scale_x = simpledialog.askfloat("Axis scaling", "Enter X-axis visual scale:", initialvalue=1.0, minvalue=0.1, maxvalue=5.0)
    if scale_x is None: scale_x = 1.0
    scale_y = simpledialog.askfloat("Axis scaling", "Enter Y-axis visual scale:", initialvalue=1.0, minvalue=0.1, maxvalue=5.0)
    if scale_y is None: scale_y = 1.0
    scale_z = simpledialog.askfloat("Axis scaling", "Enter Z-axis visual scale:", initialvalue=1.0, minvalue=0.1, maxvalue=5.0)
    if scale_z is None: scale_z = 1.0

    # Create structured grid
    grid = pv.StructuredGrid(X, Y, Z)
    
    plotter = pv.Plotter(title=title, window_size=(900, 700))
    plotter.background_color = 'white'
    
    # Add surface
    surf = plotter.add_mesh(grid, scalars=Z.ravel(), cmap='viridis', 
                            show_scalar_bar=True, scalar_bar_args={'title': z_label},
                            opacity=1.0)
    
    # Add axes
    plotter.add_axes(xlabel=x_label, ylabel=y_label, zlabel=z_label,
                     line_width=2, color='black')
    plotter.show_grid(xlabel=x_label, ylabel=y_label, zlabel=z_label)
    
    # Set axis scale
    plotter.set_scale(xscale=scale_x, yscale=scale_y, zscale=scale_z)
    
    # Add opacity slider
    def set_opacity(value):
        surf.GetProperty().SetOpacity(value)
        plotter.render()
    
    plotter.add_slider_widget(
        callback=set_opacity,
        rng=[0.0, 1.0],
        value=1.0,
        title='Opacity',
        pointa=(0.1, 0.1),
        pointb=(0.4, 0.1),
        style='modern',
    )
    
    plotter.view_isometric()
    plotter.show()
    return plotter

# ==================== Modified 3D plot function ====================
def draw_3d():
    """Draw 3D plot: choose between line waterfall or surface (requires matrix generation first)"""
    global data_waves, matrices, last_prefix

    # Check available engines
    engine_choices = ["Matplotlib (static)"]
    if PYVISTA_AVAILABLE:
        engine_choices.append("PyVista (interactive)")
    
    if len(engine_choices) == 1:
        messagebox.showinfo("Info", "PyVista not installed, using Matplotlib. For interactive plots, install: pip install pyvista")
        use_pyvista = False
    else:
        engine_choice = ask_choice("Plot engine", "Select plotting engine:", engine_choices, default_index=0)
        if engine_choice is None:
            return
        use_pyvista = (engine_choice == "PyVista (interactive)")

    # Select plot type
    plot_type_choice = ask_choice("3D plot type", "Select plot type:",
                                  ["Line waterfall (requires imported data)", "Surface (requires generated matrix)"], default_index=0)
    if plot_type_choice is None:
        return
    is_waterfall = (plot_type_choice == "Line waterfall (requires imported data)")

    if is_waterfall:
        if not data_waves:
            messagebox.showerror("Error", "No data available, please import data first")
            return
        default_prefix = last_prefix if last_prefix else "10"
        prefix = simpledialog.askstring("3D waterfall", "Data prefix:", initialvalue=default_prefix)
        if not prefix: return
        startL = ask_number("3D waterfall", "Start index:", default=1, vartype=int)
        if startL is None: return
        endL = ask_number("3D waterfall", "End index:", default=121, vartype=int)
        if endL is None: return
        
        x_offset = ask_number("3D waterfall", "X-axis offset (horizontal shift):", default=0.0, vartype=float)
        if x_offset is None: return

        # Custom axis labels
        x_label = simpledialog.askstring("Axis labels", "Enter X-axis label:", initialvalue="q (nm⁻¹)")
        if x_label is None:
            x_label = "q (nm⁻¹)"
        z_label = simpledialog.askstring("Axis labels", "Enter Z-axis label:", initialvalue="Intensity (a.u.)")
        if z_label is None:
            z_label = "Intensity (a.u.)"

        # Z-axis log option
        logz_choice = ask_choice("Z-axis data", "Apply log10 to Z (intensity)?", ["No", "Yes"], default_index=0)
        if logz_choice is None: return
        use_log_z = (logz_choice == "Yes")
        if use_log_z:
            z_label = z_label + " (log10)"

        # Collect curve data
        curves = []
        for i in range(startL, endL + 1):
            x_key = f"{prefix}-{i}x"
            y_key = f"{prefix}-{i}y"
            if x_key in data_waves and y_key in data_waves:
                x = data_waves[x_key].copy()
                y = data_waves[y_key].copy()
                if use_log_z:
                    y = np.maximum(y, 1e-10)
                    y = np.log10(y)
                curves.append((i, x, y))
        if not curves:
            messagebox.showerror("Error", f"No waveform data found for prefix {prefix}")
            return

        title = f'3D Waterfall (prefix {prefix})'

        # Ask whether to use segmented Y spacing
        use_segments = messagebox.askyesno("Segment settings", "Use segmented Y-axis spacing?\n(Similar to 2D waterfall, different spacings for segments)")
        
        segment_bounds = None
        segment_steps = None
        y_offset = 1.0
        
        if use_segments:
            num_segments = ask_number("Segment settings", "How many segments? (1-10):", default=1, vartype=int)
            if num_segments is None:
                num_segments = 1
            num_segments = max(1, min(10, num_segments))
            
            segment_bounds = []
            segment_steps = []
            
            for seg_idx in range(num_segments):
                if seg_idx < num_segments - 1:
                    default_bound = min(startL + (seg_idx+1) * ((endL - startL + 1) // num_segments), endL)
                    bound = ask_number(f"Segment {seg_idx+1}", f"End index for segment {seg_idx+1}:", 
                                      default=default_bound, vartype=int)
                    if bound is None:
                        bound = default_bound
                    step = ask_number(f"Segment {seg_idx+1}", f"Y spacing for segment {seg_idx+1}:", 
                                     default=1.0, vartype=float)
                    if step is None:
                        step = 1.0
                    segment_bounds.append(bound)
                    segment_steps.append(step)
                else:
                    step = ask_number("Last segment", "Y spacing for last segment:", default=1.0, vartype=float)
                    if step is None:
                        step = 1.0
                    segment_steps.append(step)
                    segment_bounds.append(float('inf'))
        else:
            y_offset = ask_number("3D waterfall", "Y-direction offset (curve spacing):", default=1.0, vartype=float)
            if y_offset is None:
                y_offset = 1.0

        if use_pyvista:
            plotter = draw_3d_waterfall_pyvista(
                curves, x_label, z_label, title,
                y_offset if not use_segments else 1.0,
                x_offset,
                segment_bounds if use_segments else None,
                segment_steps if use_segments else None
            )
            messagebox.showinfo("Info", "PyVista window opened.")
        
        else:
            # Temporarily switch backend to support interactive plotting
            plt.switch_backend('TkAgg')
            
            # Matplotlib plotting (unchanged)
            use_segments_mpl = use_segments
            
            if use_segments_mpl:
                num_segments = len(segment_bounds)
                segment_bounds_mpl = segment_bounds
                segment_steps_mpl = segment_steps
            else:
                y_offset_mpl = y_offset

            fig = plt.figure(figsize=(8, 6))
            fig.subplots_adjust(bottom=0.2, left=0, right=1, top=1)
            ax = fig.add_subplot(111, projection='3d')
            ax.grid(False)

            init_size = 0.6
            left = (1 - init_size) / 2
            bottom = (1 - init_size) / 2
            ax.set_position([left, bottom, init_size, init_size])

            # Calculate Y positions
            if use_segments_mpl:
                y_positions = []
                cum_offset = 0
                current_segment = 0
                for idx, (i, x, y) in enumerate(curves):
                    while current_segment < len(segment_bounds_mpl) and i > segment_bounds_mpl[current_segment]:
                        current_segment += 1
                    step = segment_steps_mpl[current_segment] if current_segment < len(segment_steps_mpl) else segment_steps_mpl[-1]
                    y_positions.append(cum_offset)
                    cum_offset += step
            else:
                y_positions = [idx * y_offset_mpl for idx in range(len(curves))]

            for idx, (i, x, y) in enumerate(curves):
                y_pos = y_positions[idx]
                x_shifted = x + idx * x_offset
                ax.plot(x_shifted, np.full_like(x, y_pos), y, label=f'Curve {i}')

            ax.set_xlabel(x_label)
            ax.set_yticks([])
            ax.set_ylabel('')
            ax.yaxis.line.set_color('none')
            ax.zaxis.line.set_color('none')
            ax.zaxis.set_tick_params(size=0)
            ax.set_zticklabels([])
            ax.set_zlabel(z_label)

            zlim = ax.get_zlim()
            if zlim[0] <= 0 <= zlim[1]:
                ax.set_zlim(bottom=0)
            else:
                ax.set_zlim(bottom=zlim[0])

            ax.set_title(title)
            ax.view_init(elev=0, azim=-90)

            # Sliders and export controls
            ax_size_slider = fig.add_axes([0.2, 0.05, 0.6, 0.03])
            size_slider = Slider(ax_size_slider, 'Figure Size', 0.2, 0.9, valinit=init_size, valstep=0.01)
            def update_size(val):
                size = val
                new_left = (1 - size) / 2
                new_bottom = (1 - size) / 2
                ax.set_position([new_left, new_bottom, size, size])
                fig.canvas.draw_idle()
            size_slider.on_changed(update_size)

            ax_yscale = fig.add_axes([0.2, 0.12, 0.6, 0.03])
            slider_yscale = Slider(ax_yscale, 'Y Scale', 0.1, 5.0, valinit=1.0)
            def update_yscale(val):
                current_aspect = ax.get_box_aspect()
                ax.set_box_aspect([current_aspect[0], val, current_aspect[2]])
                fig.canvas.draw_idle()
            slider_yscale.on_changed(update_yscale)

            ax_btn = fig.add_axes([0.8, 0.05, 0.15, 0.04])
            btn_export = Button(ax_btn, 'Export Figure')
            def export_figure(event):
                to_hide = [ax_size_slider, ax_yscale, ax_btn]
                orig_visible = [ax.get_visible() for ax in to_hide]
                for ax_hide in to_hide:
                    ax_hide.set_visible(False)
                fig.canvas.draw_idle()
                temp_root = tk.Tk()
                temp_root.withdraw()
                file_path = filedialog.asksaveasfilename(
                    defaultextension=".png",
                    filetypes=[("PNG image", "*.png"), ("PDF vector", "*.pdf"),
                               ("SVG vector", "*.svg"), ("EPS vector", "*.eps")],
                    title="Save 3D figure", parent=temp_root)
                temp_root.destroy()
                if file_path:
                    ext = os.path.splitext(file_path)[1].lower()
                    fmt = ext[1:] if ext else 'png'
                    try:
                        fig.savefig(file_path, dpi=300, bbox_inches='tight', format=fmt)
                        messagebox.showinfo("Success", f"Figure saved to:\n{file_path}")
                    except Exception as e:
                        messagebox.showerror("Error", f"Save failed: {e}")
                for ax_hide, vis in zip(to_hide, orig_visible):
                    ax_hide.set_visible(vis)
                fig.canvas.draw_idle()
            btn_export.on_clicked(export_figure)

            plt.show()
            plt.switch_backend('Agg')  # restore backend

    else:  # Surface mode
        if not matrices:
            messagebox.showerror("Error", "Please generate a matrix first (menu 3 or 4)")
            return
        prefix_list = list(matrices.keys())
        if len(prefix_list) == 1:
            prefix = prefix_list[0]
        else:
            prefix = simpledialog.askstring("Select matrix", f"Available prefixes: {', '.join(prefix_list)}",
                                            initialvalue=prefix_list[0])
            if prefix not in matrices:
                messagebox.showerror("Error", "Prefix not found")
                return
        mat_info = matrices[prefix]
        Z = mat_info['matrix']
        x_coords = mat_info['x']
        n_curves = Z.shape[1]
        y_inds = np.arange(n_curves)

        # Custom axis labels
        x_label = simpledialog.askstring("Axis labels", "Enter X-axis label:", initialvalue="q (nm⁻¹)")
        if x_label is None:
            x_label = "q (nm⁻¹)"
        y_label = simpledialog.askstring("Axis labels", "Enter Y-axis label:", initialvalue="Frame")
        if y_label is None:
            y_label = "Frame"
        z_label = simpledialog.askstring("Axis labels", "Enter Z-axis label:", initialvalue="Intensity (a.u.)")
        if z_label is None:
            z_label = "Intensity (a.u.)"

        title = f'3D Surface (prefix: {prefix})'

        if use_pyvista:
            X, Y = np.meshgrid(x_coords, y_inds, indexing='ij')
            plotter = draw_3d_surface_pyvista(X, Y, Z, x_label, y_label, z_label, title)
            messagebox.showinfo("Info", "PyVista window opened.\nYou can adjust opacity using the slider at the bottom.")
        
        else:
            # Temporarily switch backend to support interactive plotting
            plt.switch_backend('TkAgg')
            
            # Matplotlib surface
            elev = 30
            azim = -60
            roll = 0

            fig = plt.figure(figsize=(8, 6))
            fig.subplots_adjust(bottom=0.2, left=0, right=1, top=1)
            ax = fig.add_subplot(111, projection='3d')
            ax.grid(False)

            init_size = 0.6
            left = (1 - init_size) / 2
            bottom = (1 - init_size) / 2
            ax.set_position([left, bottom, init_size, init_size])

            X, Y = np.meshgrid(x_coords, y_inds, indexing='ij')
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', shade=True, edgecolor='none')

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_zlabel(z_label)
            ax.set_title(title + '\nLeft drag: rotate, Right drag: pan, Scroll: zoom')
            ax.view_init(elev=elev, azim=azim, roll=roll)

            ax_size_slider = fig.add_axes([0.2, 0.05, 0.6, 0.03])
            size_slider = Slider(ax_size_slider, 'Figure Size', 0.2, 0.9, valinit=init_size, valstep=0.01)
            def update_size(val):
                size = val
                new_left = (1 - size) / 2
                new_bottom = (1 - size) / 2
                ax.set_position([new_left, new_bottom, size, size])
                fig.canvas.draw_idle()
            size_slider.on_changed(update_size)

            ax_alpha = fig.add_axes([0.2, 0.12, 0.6, 0.03])
            slider_alpha = Slider(ax_alpha, 'Alpha', 0, 1, valinit=1.0)
            def update_alpha(val):
                surf.set_alpha(val)
                fig.canvas.draw_idle()
            slider_alpha.on_changed(update_alpha)

            ax_yscale = fig.add_axes([0.2, 0.19, 0.6, 0.03])
            slider_yscale = Slider(ax_yscale, 'Y Scale', 0.1, 5.0, valinit=1.0)
            def update_yscale(val):
                current_aspect = ax.get_box_aspect()
                ax.set_box_aspect([current_aspect[0], val, current_aspect[2]])
                fig.canvas.draw_idle()
            slider_yscale.on_changed(update_yscale)

            ax_btn = fig.add_axes([0.8, 0.05, 0.15, 0.04])
            btn_export = Button(ax_btn, 'Export Figure')
            def export_figure(event):
                to_hide = [ax_size_slider, ax_alpha, ax_yscale, ax_btn]
                orig_visible = [ax.get_visible() for ax in to_hide]
                for ax_hide in to_hide:
                    ax_hide.set_visible(False)
                fig.canvas.draw_idle()
                temp_root = tk.Tk()
                temp_root.withdraw()
                file_path = filedialog.asksaveasfilename(
                    defaultextension=".png",
                    filetypes=[("PNG image", "*.png"), ("PDF vector", "*.pdf"),
                               ("SVG vector", "*.svg"), ("EPS vector", "*.eps")],
                    title="Save 3D figure", parent=temp_root)
                temp_root.destroy()
                if file_path:
                    ext = os.path.splitext(file_path)[1].lower()
                    fmt = ext[1:] if ext else 'png'
                    try:
                        fig.savefig(file_path, dpi=300, bbox_inches='tight', format=fmt)
                        messagebox.showinfo("Success", f"Figure saved to:\n{file_path}")
                    except Exception as e:
                        messagebox.showerror("Error", f"Save failed: {e}")
                for ax_hide, vis in zip(to_hide, orig_visible):
                    ax_hide.set_visible(vis)
                fig.canvas.draw_idle()
            btn_export.on_clicked(export_figure)

            plt.show()
            plt.switch_backend('Agg')  # restore backend

def export_vector():
    figs = [plt.figure(num) for num in plt.get_fignums()]
    if not figs:
        messagebox.showerror("Error", "No open figure windows")
        return
    if len(figs) > 1:
        names = [f"Figure {i+1}" for i in range(len(figs))]
        choice = simpledialog.askinteger("Export vector graphic", "Select figure to export:\n" + "\n".join([f"{i+1}: {name}" for i, name in enumerate(names)]),
                                        initialvalue=1, minvalue=1, maxvalue=len(figs))
        if choice is None: return
        fig = figs[choice-1]
    else:
        fig = figs[0]
    filetypes = [("SVG vector", "*.svg"), ("PDF document", "*.pdf"), ("EPS vector", "*.eps")]
    filename = filedialog.asksaveasfilename(defaultextension=".svg", filetypes=filetypes,
                                            title="Save vector graphic")
    if not filename:
        return
    try:
        fig.savefig(filename, format=filename.split('.')[-1], dpi=300, bbox_inches='tight')
        messagebox.showinfo("Complete", f"Vector graphic saved to:\n{filename}")
    except Exception as e:
        messagebox.showerror("Error", f"Save failed: {e}")

# ==================== Plotting tool main window (Toplevel) ====================
class Application(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("SynXS - Data Automation Tool")
        self.geometry("300x780")
        self.resizable(False, False)

        self.global_vmin = None
        self.global_vmax = None
        self.global_levels = 20
        self.use_custom_color = False

        btn1 = ttk.Button(self, text="1. Batch import and name", command=load_data)
        btn1.pack(pady=5, fill='x', padx=20)

        btn2 = ttk.Button(self, text="2. Draw 2D waterfall", command=draw_2d)
        btn2.pack(pady=5, fill='x', padx=20)

        btn3 = ttk.Button(self, text="3. Generate normal matrix", command=lambda: make_matrix(0))
        btn3.pack(pady=5, fill='x', padx=20)

        btn4 = ttk.Button(self, text="4. Generate log matrix", command=lambda: make_matrix(1))
        btn4.pack(pady=5, fill='x', padx=20)

        btn5 = ttk.Button(self, text="5. View all import records", command=show_status)
        btn5.pack(pady=5, fill='x', padx=20)

        btn6 = ttk.Button(self, text="6. Clear all data", command=kill_all)
        btn6.pack(pady=5, fill='x', padx=20)

        btn7 = ttk.Button(self, text="7. Draw 3D plot", command=draw_3d)
        btn7.pack(pady=5, fill='x', padx=20)

        btn9 = ttk.Button(self, text="8. Export vector graphic (for AI editing)", command=export_vector)
        btn9.pack(pady=5, fill='x', padx=20)

        btn_color_cfg = tk.Button(self, text="Global color scale settings", 
                                  command=self.set_global_color_scale,
                                  bg="#FF9800", fg="white", font=('Arial', 10, 'bold'))
        btn_color_cfg.pack(pady=5, fill='x', padx=20)

        btn_contourf = ttk.Button(self, text="9. Draw filled contour plot", command=self.draw_contourf)
        btn_contourf.pack(pady=5, fill='x', padx=20)

        btn_slideshow = ttk.Button(self, text="10. Curve slideshow", command=self.play_slideshow)
        btn_slideshow.pack(pady=5, fill='x', padx=20)

        btn_exit = ttk.Button(self, text="Exit", command=self.destroy)
        btn_exit.pack(pady=20, fill='x', padx=20)

        self.status = tk.Label(self, text="Ready", bd=1, relief='sunken', anchor='w')
        self.status.pack(side='bottom', fill='x')

    def set_global_color_scale(self):
        result = messagebox.askyesno("Global color scale settings", 
                                     "Do you want to customize the global color range?\n(If 'No', each plot will auto-calculate range)\n\n"
                                     "Note: After setting, it will apply to all contour plots for consistent color scaling.")
        if result:
            vmin = simpledialog.askfloat("Settings", "Enter minimum value (Min):", initialvalue=0.0)
            vmax = simpledialog.askfloat("Settings", "Enter maximum value (Max):", initialvalue=100.0)
            levels = simpledialog.askinteger("Settings", "Enter number of contour levels:", 
                                             initialvalue=20, minvalue=5, maxvalue=200)
            if vmin is not None and vmax is not None and levels is not None:
                if vmin < vmax:
                    self.global_vmin = vmin
                    self.global_vmax = vmax
                    self.global_levels = levels
                    self.use_custom_color = True
                    self.status.config(text=f"Global range: [{vmin}, {vmax}], levels: {levels}")
                    messagebox.showinfo("Success", f"Global range set: {vmin} to {vmax}, {levels} levels\n"
                                        "Subsequent contour plots will use this unified scale.")
                else:
                    messagebox.showerror("Error", "Minimum must be less than maximum")
                    self.use_custom_color = False
            else:
                self.use_custom_color = False
        else:
            self.use_custom_color = False
            self.status.config(text="Auto color range mode")
            messagebox.showinfo("Info", "Switched back to auto range mode, each plot will auto-calculate range.")

    def draw_contourf(self):
        """Draw filled contour plot (with animation option, left-to-right progressive, fixed colorbar)"""
        global matrices, last_prefix
        
        # Temporarily switch backend to support interactive plotting
        plt.switch_backend('TkAgg')
        
        if not matrices:
            messagebox.showerror("Error", "Please generate a matrix first (menu 3 or 4)")
            plt.switch_backend('Agg')
            return

        prefix_list = list(matrices.keys())
        if len(prefix_list) == 1:
            prefix = prefix_list[0]
        else:
            prefix = simpledialog.askstring("Select matrix", f"Available prefixes: {', '.join(prefix_list)}",
                                            initialvalue=prefix_list[0])
            if prefix not in matrices:
                messagebox.showerror("Error", "Prefix not found")
                plt.switch_backend('Agg')
                return

        mat_info = matrices[prefix]
        Z = mat_info['matrix']
        x = mat_info['x']
        n_points, n_curves = Z.shape

        r_min = 0.0
        r_max = float(n_curves - 1)

        y_label = simpledialog.askstring("Y-axis label", "Enter Y-axis name (English):\ne.g. Temperature, Time, Frame Number",
                                         initialvalue="Curve Index")
        if y_label is None:
            y_label = "Curve Index"

        anim_choice = ask_choice("Animation effect", "Enable progressive display animation (left-to-right)?", ["No", "Yes"], default_index=1)
        if anim_choice is None:
            plt.switch_backend('Agg')
            return
        enable_animation = (anim_choice == "Yes")
        if enable_animation:
            duration = ask_number("Animation settings", "Total time for full display (seconds):", default=3.0, vartype=float)
            if duration is None or duration <= 0:
                enable_animation = False
        else:
            duration = 0.0

        levels_input = ask_number("Contour settings", "Number of filled levels (recommended 50-200):", 
                                  default=self.global_levels if self.use_custom_color else 100, 
                                  vartype=int)
        if levels_input is None:
            plt.switch_backend('Agg')
            return
        levels = max(10, min(500, levels_input))

        add_contour = ask_choice("Contour lines", "Overlay black contour lines?", ["Yes", "No"], default_index=0)
        if add_contour is None:
            plt.switch_backend('Agg')
            return
        add_contour = (add_contour == "Yes")

        cmap_choice = ask_choice("Color map", "Select color map:",
                                  ["jet", "viridis", "plasma", "inferno", "hot", "cool", "rainbow"],
                                  default_index=0)
        if cmap_choice is None:
            plt.switch_backend('Agg')
            return
        cmap = cmap_choice

        y = np.linspace(r_min, r_max, n_curves)
        X, Y = np.meshgrid(x, y, indexing='ij')

        if self.use_custom_color:
            levels_array = np.linspace(self.global_vmin, self.global_vmax, self.global_levels)
            vmin, vmax = self.global_vmin, self.global_vmax
            status_msg = f"Global range: [{self.global_vmin:.2f}, {self.global_vmax:.2f}]"
        else:
            levels_array = levels
            vmin = np.min(Z)
            vmax = np.max(Z)
            status_msg = f"Auto range: [{vmin:.2f}, {vmax:.2f}]"

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.canvas.manager.set_window_title(f"Contourf - {prefix}")

        ax.set_xlabel('q (nm⁻¹)')
        ax.set_ylabel(y_label)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(r_min, r_max)
        ax.tick_params(direction='in', top=True, right=True)
        
        if enable_animation:
            if n_points < 2:
                messagebox.showwarning("Warning", "Too few points in X direction, cannot animate. Drawing full plot.")
                enable_animation = False
            else:
                if self.use_custom_color:
                    norm = plt.Normalize(vmin=self.global_vmin, vmax=self.global_vmax)
                else:
                    norm = plt.Normalize(vmin=vmin, vmax=vmax)
                sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
                sm.set_array([])
                
                cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
                cbar.set_label('Intensity')
                
                if self.use_custom_color:
                    tick_locations = np.linspace(self.global_vmin, self.global_vmax, 8)
                    cbar.set_ticks(tick_locations)
                    cbar.set_ticklabels([f"{t:.2f}" for t in tick_locations])
                
                current_contourf = None
                contour_lines = []
                ax.set_title(f'{prefix} | {status_msg} | Animation')
                
                num_frames = n_points - 1
                interval = (duration * 1000) / num_frames
                
                def update(frame_idx):
                    nonlocal current_contourf, contour_lines
                    
                    if current_contourf is not None:
                        current_contourf.remove()
                    for cl in contour_lines:
                        cl.remove()
                    contour_lines.clear()
                    
                    current_max_idx = frame_idx + 2
                    Z_partial = Z.copy()
                    Z_partial[:, current_max_idx:] = np.nan
                    
                    if self.use_custom_color:
                        current_contourf = ax.contourf(X, Y, Z_partial, levels=levels_array, 
                                                       cmap=cmap, vmin=vmin, vmax=vmax)
                    else:
                        current_contourf = ax.contourf(X, Y, Z_partial, levels=levels_array, cmap=cmap)
                    
                    if add_contour:
                        if self.use_custom_color:
                            cl = ax.contour(X, Y, Z_partial, levels=levels_array, colors='black', 
                                            linewidths=0.5, alpha=0.6)
                        else:
                            cl = ax.contour(X, Y, Z_partial, levels=10, colors='black', 
                                            linewidths=0.5, alpha=0.6)
                        contour_lines.append(cl)
                    
                    fig.canvas.draw_idle()
                    return [current_contourf]
                
                ani = FuncAnimation(fig, update, frames=num_frames, interval=interval, repeat=False)
                plt.tight_layout()
                plt.show()
                self.status.config(text=f"Contour plot displayed (animated) | {duration}s | {status_msg} | Y-axis: {y_label}")
                plt.switch_backend('Agg')
                return
        
        if self.use_custom_color:
            cf = ax.contourf(X, Y, Z, levels=levels_array, cmap=cmap, 
                             vmin=vmin, vmax=vmax, extend='both')
        else:
            cf = ax.contourf(X, Y, Z, levels=levels_array, cmap=cmap)
        
        if add_contour:
            if self.use_custom_color:
                ax.contour(X, Y, Z, levels=levels_array, colors='black', 
                           linewidths=0.5, alpha=0.6)
            else:
                ax.contour(X, Y, Z, levels=10, colors='black', 
                           linewidths=0.5, alpha=0.6)
        
        cbar = fig.colorbar(cf, ax=ax, shrink=0.8)
        cbar.set_label('Intensity')
        if self.use_custom_color:
            tick_locations = np.linspace(self.global_vmin, self.global_vmax, 8)
            cbar.set_ticks(tick_locations)
            cbar.set_ticklabels([f"{t:.2f}" for t in tick_locations])
        
        ax.set_title(f'{prefix} | {status_msg}')
        plt.tight_layout()
        plt.show()
        self.status.config(text=f"Contour plot drawn | {status_msg} | Y-axis: {y_label}")
        
        # Restore backend
        plt.switch_backend('Agg')

    def play_slideshow(self):
        """Play imported curve sequence with auto and manual paging, Y-axis log support"""
        global data_waves, last_prefix
        
        # Temporarily switch backend to support interactive plotting
        plt.switch_backend('TkAgg')
        
        if not data_waves:
            messagebox.showerror("Error", "No data available, please import data first")
            plt.switch_backend('Agg')
            return

        prefix = simpledialog.askstring("Slideshow", "Enter data prefix:", initialvalue=last_prefix if last_prefix else "10")
        if not prefix:
            plt.switch_backend('Agg')
            return

        numbers = []
        for key in data_waves:
            if key.startswith(f"{prefix}-") and key.endswith("x"):
                try:
                    num = int(key.split('-')[1][:-1])
                    numbers.append(num)
                except:
                    pass
        numbers = sorted(set(numbers))
        if not numbers:
            messagebox.showerror("Error", f"No waveform data found for prefix {prefix}")
            plt.switch_backend('Agg')
            return

        start_num = ask_number("Slideshow", f"Start index (min {numbers[0]}):", default=numbers[0], vartype=int)
        if start_num is None or start_num not in numbers:
            start_num = numbers[0]
        end_num = ask_number("Slideshow", f"End index (max {numbers[-1]}):", default=numbers[-1], vartype=int)
        if end_num is None or end_num not in numbers:
            end_num = numbers[-1]
        if start_num > end_num:
            start_num, end_num = end_num, start_num

        curves = []
        for n in range(start_num, end_num + 1):
            x_key = f"{prefix}-{n}x"
            y_key = f"{prefix}-{n}y"
            if x_key in data_waves and y_key in data_waves:
                curves.append((n, data_waves[x_key], data_waves[y_key]))
        if not curves:
            messagebox.showerror("Error", "No valid data in selected range")
            plt.switch_backend('Agg')
            return

        speed = ask_number("Slideshow", "Display time per image (seconds):", default=1.0, vartype=float)
        if speed is None:
            speed = 1.0
        speed = max(0.1, speed)

        logy_choice = ask_choice("Y-axis scale", "Use log Y-axis?", ["No", "Yes"], default_index=0)
        if logy_choice is None:
            plt.switch_backend('Agg')
            return
        logy = (logy_choice == "Yes")

        self.slideshow_win = tk.Toplevel(self)
        self.slideshow_win.title("Curve Slideshow")
        self.slideshow_win.geometry("800x600")
        self.slideshow_win.protocol("WM_DELETE_WINDOW", lambda: self._close_slideshow(plt))

        fig = Figure(figsize=(6, 4), dpi=100)
        self.slideshow_fig = fig
        ax = fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig, master=self.slideshow_win)
        self.slideshow_canvas = canvas
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        toolbar = NavigationToolbar2Tk(canvas, self.slideshow_win)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        control_frame = tk.Frame(self.slideshow_win)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

        self.slideshow_index = 0
        self.slideshow_curves = curves
        self.slideshow_speed = speed
        self.slideshow_auto = False
        self.slideshow_after_id = None
        self.slideshow_logy = logy

        self.slideshow_info = tk.Label(control_frame, text="")
        self.slideshow_info.pack(side=tk.LEFT, padx=10)

        tk.Label(control_frame, text="Interval (s):").pack(side=tk.LEFT, padx=5)
        self.speed_var = tk.DoubleVar(value=speed)
        speed_spin = tk.Spinbox(control_frame, from_=0.1, to=10.0, increment=0.1,
                                 textvariable=self.speed_var, width=5, command=self._update_speed)
        speed_spin.pack(side=tk.LEFT, padx=5)

        btn_prev = tk.Button(control_frame, text="Previous", command=self._slideshow_prev)
        btn_prev.pack(side=tk.LEFT, padx=5)
        btn_next = tk.Button(control_frame, text="Next", command=self._slideshow_next)
        btn_next.pack(side=tk.LEFT, padx=5)
        self.btn_auto = tk.Button(control_frame, text="Auto play", command=self._toggle_auto)
        self.btn_auto.pack(side=tk.LEFT, padx=5)

        self._update_slideshow_display()

    def _update_speed(self):
        try:
            self.slideshow_speed = self.speed_var.get()
        except:
            pass
        if self.slideshow_auto:
            self._stop_auto()
            self._start_auto()

    def _update_slideshow_display(self):
        if not hasattr(self, 'slideshow_curves') or not self.slideshow_curves:
            return
        idx = self.slideshow_index
        num, x, y = self.slideshow_curves[idx]
        fig = self.slideshow_fig
        ax = fig.axes[0]
        ax.clear()
        ax.plot(x, y, 'b-', linewidth=1.5)
        ax.set_xlabel('q (nm⁻¹)')
        ax.set_ylabel('Intensity')
        if self.slideshow_logy:
            if np.any(y <= 0):
                messagebox.showwarning("Warning", "Data contains non-positive values, log scale may not display correctly.")
            ax.set_yscale('log')
        else:
            ax.set_yscale('linear')
        ax.set_title(f'Curve {num}')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.canvas.draw()
        total = len(self.slideshow_curves)
        self.slideshow_info.config(text=f"Image {idx+1} / {total}")

    def _slideshow_prev(self):
        if self.slideshow_auto:
            self._stop_auto()
            self.btn_auto.config(text="Auto play")
        self.slideshow_index = (self.slideshow_index - 1) % len(self.slideshow_curves)
        self._update_slideshow_display()

    def _slideshow_next(self):
        if self.slideshow_auto:
            self._stop_auto()
            self.btn_auto.config(text="Auto play")
        self.slideshow_index = (self.slideshow_index + 1) % len(self.slideshow_curves)
        self._update_slideshow_display()

    def _toggle_auto(self):
        if self.slideshow_auto:
            self._stop_auto()
            self.btn_auto.config(text="Auto play")
        else:
            self._start_auto()
            self.btn_auto.config(text="Pause")

    def _start_auto(self):
        self.slideshow_auto = True
        self._auto_play_step()

    def _stop_auto(self):
        self.slideshow_auto = False
        if self.slideshow_after_id:
            self.slideshow_win.after_cancel(self.slideshow_after_id)
            self.slideshow_after_id = None

    def _auto_play_step(self):
        if not self.slideshow_auto:
            return
        self.slideshow_index = (self.slideshow_index + 1) % len(self.slideshow_curves)
        self._update_slideshow_display()
        self.slideshow_after_id = self.slideshow_win.after(int(self.slideshow_speed * 1000), self._auto_play_step)

    def _close_slideshow(self, plt_module=None):
        if hasattr(self, 'slideshow_win'):
            if self.slideshow_auto:
                self._stop_auto()
            self.slideshow_win.destroy()
            del self.slideshow_win
            del self.slideshow_canvas
            del self.slideshow_fig
            del self.slideshow_curves
            del self.slideshow_info
            del self.btn_auto
        # Restore backend
        if plt_module:
            plt_module.switch_backend('Agg')

# ==================== Integration tool helper classes ====================
class BackgroundShiftPreview:
    def __init__(self, parent, config, initial_shift, callback):
        self.parent = parent
        self.config = config
        self.callback = callback
        self.shift = tk.DoubleVar(value=initial_shift)
        self.yscale = tk.StringVar(value='linear')
        self.sample_file = None
        self.bg_file = None
        self.sample_data = None
        self.bg_data = None
        self.ai = None
        if config.get('poni_file') and os.path.exists(config['poni_file']):
            try:
                self.ai = load_geometry(config['poni_file'])
            except:
                self.ai = None
        self.create_window()

    def create_window(self):
        self.window = tk.Toplevel(self.parent)
        self.window.title("Background Shift Preview (supports raw images)")
        self.window.geometry("800x700")
        self.window.transient(self.parent)
        self.window.grab_set()

        style = ttk.Style()
        style.theme_use('clam')

        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.pack(fill='both', expand=True)

        file_frame = ttk.LabelFrame(main_frame, text="File selection", padding="5")
        file_frame.pack(fill='x', pady=5)

        ttk.Label(file_frame, text="Sample curve/image:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.sample_entry = ttk.Entry(file_frame, width=50)
        self.sample_entry.grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        ttk.Button(file_frame, text="Browse", command=self.load_sample).grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(file_frame, text="Background curve/image:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.bg_entry = ttk.Entry(file_frame, width=50)
        self.bg_entry.grid(row=1, column=1, padx=5, pady=5, sticky='ew')
        ttk.Button(file_frame, text="Browse", command=self.load_bg).grid(row=1, column=2, padx=5, pady=5)

        info_label = ttk.Label(file_frame, text="Supports .dat files or .tif/.edf/.cbf images (auto-integrated)", foreground="gray")
        info_label.grid(row=2, column=1, sticky='w', padx=5)

        file_frame.columnconfigure(1, weight=1)

        shift_frame = ttk.Frame(main_frame)
        shift_frame.pack(fill='x', pady=5)
        ttk.Label(shift_frame, text="Background shift:").pack(side='left', padx=5)
        self.shift_scale = tk.Scale(shift_frame, from_=-1000, to=1000, orient='horizontal',
                                     variable=self.shift, resolution=1,
                                     command=self.update_plot, length=400)
        self.shift_scale.pack(side='left', fill='x', expand=True, padx=5)
        self.shift_label = ttk.Label(shift_frame, textvariable=self.shift, width=6)
        self.shift_label.pack(side='left', padx=5)

        scale_frame = ttk.Frame(main_frame)
        scale_frame.pack(fill='x', pady=5)
        ttk.Label(scale_frame, text="Y-axis:").pack(side='left', padx=5)
        rb_linear = ttk.Radiobutton(scale_frame, text="Linear", variable=self.yscale,
                                    value='linear', command=self.update_plot)
        rb_linear.pack(side='left', padx=5)
        rb_log = ttk.Radiobutton(scale_frame, text="Log", variable=self.yscale,
                                value='log', command=self.update_plot)
        rb_log.pack(side='left', padx=5)

        self.fig = Figure(figsize=(8, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, pady=5)

        toolbar = NavigationToolbar2Tk(self.canvas, self.window)
        toolbar.update()
        toolbar.pack(side='bottom', fill='x')

        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=5)
        ttk.Button(btn_frame, text="Apply", command=self.apply_shift).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Close", command=self.window.destroy).pack(side='left', padx=5)

    def load_sample(self):
        filename = filedialog.askopenfilename(
            title="Select sample file (.dat or image)",
            filetypes=[("All supported", "*.dat *.tif *.edf *.cbf"), ("DAT files", "*.dat"), ("TIFF", "*.tif"), ("EDF", "*.edf"), ("CBF", "*.cbf"), ("All files", "*.*")]
        )
        if filename:
            self.sample_entry.delete(0, tk.END)
            self.sample_entry.insert(0, filename)
            self.sample_file = filename
            self.load_data()

    def load_bg(self):
        filename = filedialog.askopenfilename(
            title="Select background file (.dat or image)",
            filetypes=[("All supported", "*.dat *.tif *.edf *.cbf"), ("DAT files", "*.dat"), ("TIFF", "*.tif"), ("EDF", "*.edf"), ("CBF", "*.cbf"), ("All files", "*.*")]
        )
        if filename:
            self.bg_entry.delete(0, tk.END)
            self.bg_entry.insert(0, filename)
            self.bg_file = filename
            self.load_data()

    def integrate_image(self, file_path):
        if self.ai is None:
            raise Exception("No geometry file loaded, cannot integrate image")
        with fabio.open(file_path) as fimg:
            img = fimg.data
        nbins = self.config.get('nbins', 2000)
        unit = self.config.get('unit', 'q_nm^-1')
        use_sector = self.config.get('use_sector', False)
        azimuth_range = self.config.get('azimuth_range', None)
        if use_sector and azimuth_range is not None:
            start, end = azimuth_range
            if start > end:
                start, end = end, start
            x, y = self.ai.integrate1d(img, nbins, unit=unit, azimuth_range=(start, end))
        else:
            x, y = self.ai.integrate1d(img, nbins, unit=unit)
        return x, y

    def load_data(self):
        if not (self.sample_file and self.bg_file):
            return
        try:
            def load_file(file_path):
                ext = os.path.splitext(file_path)[1].lower()
                if ext in ['.tif', '.edf', '.cbf']:
                    return self.integrate_image(file_path)
                else:
                    data = np.loadtxt(file_path, comments='#')
                    if data.ndim != 2 or data.shape[1] < 2:
                        raise Exception("File format incorrect, need at least two columns")
                    return data[:, 0], data[:, 1]

            x_sample, y_sample = load_file(self.sample_file)
            x_bg, y_bg = load_file(self.bg_file)

            if not np.array_equal(x_sample, x_bg):
                interp = interp1d(x_sample, y_sample, kind='linear', bounds_error=False, fill_value=0)
                y_sample_interp = interp(x_bg)
                self.sample_data = (x_bg, y_sample_interp)
                self.bg_data = (x_bg, y_bg)
                self.ax.set_title("Note: Sample x-axis differs from background, interpolated to align")
            else:
                self.sample_data = (x_sample, y_sample)
                self.bg_data = (x_bg, y_bg)

            self.update_plot()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load files: {e}")

    def update_plot(self, event=None):
        if self.sample_data is None or self.bg_data is None:
            return
        self.ax.clear()
        x, y_sample = self.sample_data
        x_bg, y_bg = self.bg_data
        shift = self.shift.get()

        self.ax.plot(x, y_sample, 'b-', label='Sample', linewidth=2)
        self.ax.plot(x_bg, y_bg, 'g--', label='Background (original)', alpha=0.7)
        self.ax.plot(x_bg, y_bg + shift, 'r-', label=f'Background shifted ({shift:+.1f})', linewidth=2)

        self.ax.set_yscale(self.yscale.get())
        self.ax.set_xlabel('q / nm⁻¹')
        self.ax.set_ylabel('Intensity')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()

    def apply_shift(self):
        self.callback(self.shift.get())
        self.window.destroy()


class IntegrationPreview:
    def __init__(self, parent, config):
        self.parent = parent
        self.config = config
        self.file_paths = []
        self.ai = None
        if config.get('poni_file') and os.path.exists(config['poni_file']):
            try:
                self.ai = load_geometry(config['poni_file'])
            except:
                self.ai = None
        self.yscale = tk.StringVar(value='linear')
        self.create_window()

    def create_window(self):
        self.window = tk.Toplevel(self.parent)
        self.window.title("Integration Preview")
        self.window.geometry("800x700")
        self.window.transient(self.parent)
        self.window.grab_set()

        style = ttk.Style()
        style.theme_use('clam')

        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.pack(fill='both', expand=True)

        file_frame = ttk.Frame(main_frame)
        file_frame.pack(fill='x', pady=5)
        ttk.Label(file_frame, text="Select image files (multiple):").pack(side='left', padx=5)
        self.file_btn = ttk.Button(file_frame, text="Browse...", command=self.select_files)
        self.file_btn.pack(side='left', padx=5)
        self.file_label = ttk.Label(file_frame, text="No files selected", foreground="gray")
        self.file_label.pack(side='left', padx=5)

        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill='x', pady=5)
        ttk.Button(btn_frame, text="Integration Preview", command=self.integrate_and_plot).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Close", command=self.window.destroy).pack(side='left', padx=5)

        scale_frame = ttk.Frame(main_frame)
        scale_frame.pack(fill='x', pady=5)
        ttk.Label(scale_frame, text="Y-axis:").pack(side='left', padx=5)
        rb_linear = ttk.Radiobutton(scale_frame, text="Linear", variable=self.yscale,
                                    value='linear', command=self.update_plot)
        rb_linear.pack(side='left', padx=5)
        rb_log = ttk.Radiobutton(scale_frame, text="Log", variable=self.yscale,
                                value='log', command=self.update_plot)
        rb_log.pack(side='left', padx=5)

        self.fig = Figure(figsize=(8, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, pady=5)

        toolbar = NavigationToolbar2Tk(self.canvas, self.window)
        toolbar.update()
        toolbar.pack(side='bottom', fill='x')

        self.results = []

    def select_files(self):
        initial_dir = self.config.get('initial_dir', os.path.expanduser("~"))
        files = filedialog.askopenfilenames(
            title="Select image files (multiple)",
            initialdir=initial_dir,
            filetypes=[("Image files", "*.tif *.edf *.cbf"), ("All files", "*.*")]
        )
        if files:
            self.file_paths = list(files)
            self.file_label.config(text=f"Selected {len(self.file_paths)} files")
        else:
            self.file_paths = []
            self.file_label.config(text="No files selected")

    def integrate_and_plot(self):
        if not self.file_paths:
            messagebox.showerror("Error", "Please select files first")
            return
        if self.ai is None:
            messagebox.showerror("Error", "Invalid geometry file, please check")
            return

        self.results.clear()
        self.ax.clear()
        self.fig.suptitle("Integration Preview (processing...)")
        self.canvas.draw()

        nbins = self.config.get('nbins', 2000)
        unit = self.config.get('unit', 'q_nm^-1')
        use_sector = self.config.get('use_sector', False)
        azimuth_range = self.config.get('azimuth_range', None)

        for file_path in self.file_paths:
            try:
                with fabio.open(file_path) as fimg:
                    img = fimg.data.astype(np.float32)

                if use_sector and azimuth_range is not None:
                    start, end = azimuth_range
                    if start > end:
                        start, end = end, start
                    x, y = self.ai.integrate1d(img, nbins, unit=unit, azimuth_range=(start, end))
                else:
                    x, y = self.ai.integrate1d(img, nbins, unit=unit)

                label = os.path.basename(file_path)
                self.results.append((x, y, label))
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process file {os.path.basename(file_path)}: {e}")
                return

        self.update_plot()

    def update_plot(self, event=None):
        self.ax.clear()
        for x, y, label in self.results:
            self.ax.plot(x, y, label=label)
        self.ax.set_xlabel(self.config.get('unit', 'q_nm^-1'))
        self.ax.set_ylabel('Intensity')
        self.ax.set_yscale(self.yscale.get())
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()


class DependencyStatusDialog:
    def __init__(self, parent):
        self.parent = parent
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Dependency Status")
        self.dialog.geometry("600x300")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        self.packages = [
            ('numpy', 'numpy', 'https://pypi.org/project/numpy/', 'NumPy'),
            ('matplotlib', 'matplotlib', 'https://pypi.org/project/matplotlib/', 'Matplotlib'),
            ('fabio', 'fabio', 'https://pypi.org/project/fabio/', 'FabIO'),
            ('pyFAI', 'pyfai', 'https://pypi.org/project/pyfai/', 'pyFAI'),
            ('scipy', 'scipy', 'https://pypi.org/project/scipy/', 'SciPy')
        ]
        self.status = []
        for import_name, pip_name, url, display_name in self.packages:
            try:
                __import__(import_name)
                installed = True
            except ImportError:
                installed = False
            self.status.append((display_name, installed, pip_name, url))

        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.dialog, text="Required Library Status", font=('Arial', 12, 'bold')).pack(pady=10)

        frame = tk.Frame(self.dialog)
        frame.pack(fill='both', expand=True, padx=10, pady=5)

        tk.Label(frame, text="Status", font=('Arial', 10, 'bold')).grid(row=0, column=0, padx=5, pady=2)
        tk.Label(frame, text="Library", font=('Arial', 10, 'bold')).grid(row=0, column=1, padx=5, pady=2, sticky='w')
        tk.Label(frame, text="Install command", font=('Arial', 10, 'bold')).grid(row=0, column=2, padx=5, pady=2, sticky='w')

        missing_commands = []
        for i, (display_name, installed, pip_name, url) in enumerate(self.status, start=1):
            if installed:
                status_text = "OK"
                color = "green"
                cmd_display = "Installed"
            else:
                status_text = "Missing"
                color = "red"
                cmd_display = f"pip install {pip_name}"
                missing_commands.append(cmd_display)

            lbl_status = tk.Label(frame, text=status_text, fg=color, font=('Arial', 12, 'bold'))
            lbl_status.grid(row=i, column=0, padx=5, pady=2)

            tk.Label(frame, text=display_name).grid(row=i, column=1, padx=5, pady=2, sticky='w')

            if installed:
                tk.Label(frame, text=cmd_display, fg="gray").grid(row=i, column=2, padx=5, pady=2, sticky='w')
            else:
                cmd_frame = tk.Frame(frame)
                cmd_frame.grid(row=i, column=2, padx=5, pady=2, sticky='w')
                tk.Label(cmd_frame, text=cmd_display, fg="blue", cursor="hand2").pack(side='left')
                btn = tk.Button(cmd_frame, text="PyPI", command=lambda u=url: webbrowser.open(u), font=('Arial', 8))
                btn.pack(side='left', padx=5)

        btn_frame = tk.Frame(self.dialog)
        btn_frame.pack(pady=10)

        if missing_commands:
            def copy_missing():
                self.dialog.clipboard_clear()
                self.dialog.clipboard_append("\n".join(missing_commands))
                messagebox.showinfo("Copy successful", "Install commands for missing libraries copied to clipboard.")
            tk.Button(btn_frame, text="Copy missing install commands", command=copy_missing, width=20).pack(side='left', padx=5)
        else:
            tk.Label(btn_frame, text="All required libraries installed!", fg="green").pack(side='left', padx=5)

        tk.Button(btn_frame, text="Close", command=self.dialog.destroy, width=10).pack(side='left', padx=5)


# ==================== Integration tool - multi-process safe worker function (outside class) ====================
def process_images_worker(config, file_list, log_queue, progress_queue, cancel_event):
    """
    Independent worker function running in a subprocess, contains no Tkinter objects.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        import fabio
        import pyFAI
        import os
        import gc
        import traceback
        
        # Disable OpenCL
        pyFAI.use_opencl = False
        os.environ['PYFAI_OPENCL'] = 'False'
        
        # Unpack config
        poni = config['poni']
        output_dir = config['output_dir']
        bg_file = config.get('bg_file')
        mask_file = config.get('mask_file')
        nbins = config['nbins']
        unit = config['unit']
        use_sector = config['use_sector']
        azimuth_start = config['azimuth_start']
        azimuth_end = config['azimuth_end']
        use_log = config['use_log']
        save_plot = config['save_plot']
        bg_scale = config['bg_scale']
        bg_shift = config['bg_shift']
        allow_negative = config['allow_negative']
        
        log_queue.put("Loading geometry file...")
        ai = load_geometry(poni)
        
        bg_corrected = None
        if bg_file and os.path.exists(bg_file):
            log_queue.put("Loading background file...")
            with fabio.open(bg_file) as fbg:
                bg_img = fbg.data.astype(np.float32)
            bg_corrected = bg_scale * bg_img + bg_shift
            log_queue.put(f"Background scale factor: {bg_scale}, background shift: {bg_shift}")
            del bg_img
        
        mask = None
        if mask_file and os.path.exists(mask_file):
            log_queue.put("Loading mask file...")
            with fabio.open(mask_file) as fmask:
                mask = fmask.data
        
        total = len(file_list)
        log_queue.put(f"Total {total} files to process")
        progress_queue.put({'max': total, 'value': 0})
        
        for idx, file_path in enumerate(file_list, 1):
            if cancel_event.is_set():
                log_queue.put("Cancel detected, stopping processing.")
                break
            
            filename = os.path.basename(file_path)
            name, ext = os.path.splitext(filename)
            log_queue.put(f"Processing ({idx}/{total}): {filename}")
            
            try:
                with fabio.open(file_path) as fimg:
                    img = fimg.data.astype(np.float32)
                
                if bg_corrected is not None:
                    if bg_corrected.shape == img.shape:
                        img -= bg_corrected
                        if not allow_negative:
                            np.clip(img, 0, None, out=img)
                    else:
                        log_queue.put(f"Warning: background shape {bg_corrected.shape} != image shape {img.shape}, skipping background subtraction")
                
                if mask is not None:
                    if mask.shape == img.shape:
                        ai.mask = mask
                    else:
                        log_queue.put(f"Warning: mask shape {mask.shape} != image shape {img.shape}, skipping mask")
                        ai.mask = None
                else:
                    ai.mask = None
                
                if use_sector:
                    start, end = sorted([azimuth_start, azimuth_end])
                    x, I = ai.integrate1d(img, nbins, unit=unit, azimuth_range=(start, end))
                else:
                    x, I = ai.integrate1d(img, nbins, unit=unit)
                
                out_file = os.path.join(output_dir, name + ".dat")
                np.savetxt(out_file, np.column_stack((x, I)),
                           header=f"# {unit}\tIntensity", comments='')
                
                if save_plot:
                    plt.figure(figsize=(6,4))
                    plt.plot(x, I)
                    if use_log:
                        plt.yscale("log")
                    plt.xlabel(unit)
                    plt.ylabel("Intensity")
                    plt.title(name)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, name + ".png"), dpi=150)
                    plt.close('all')
                    gc.collect()
                
                del img, x, I
                gc.collect()
                
                progress_queue.put({'value': idx})
                
            except Exception as e:
                log_queue.put(f"File {filename} processing failed: {e}")
                log_queue.put(traceback.format_exc())
        
        if not cancel_event.is_set():
            log_queue.put("All processing completed!")
        progress_queue.put({'done': True})
        
    except Exception as e:
        log_queue.put(f"Critical error during processing: {str(e)}")
        log_queue.put(traceback.format_exc())
    finally:
        gc.collect()


# ==================== Integration tool main window (Toplevel) ====================
class SAXSBatchGUI(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.title("SynXS - SAXS Batch Processing Tool (pyFAI)")
        self.geometry("900x850")
        self.resizable(True, True)

        style = ttk.Style()
        style.theme_use('clam')
        default_font = ('Helvetica', 10)
        self.option_add('*Font', default_font)

        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.poni_file = tk.StringVar()
        self.bg_file = tk.StringVar()
        self.mask_file = tk.StringVar()
        self.nbins = tk.IntVar(value=2000)
        self.unit = tk.StringVar(value="q_nm^-1")
        self.use_sector = tk.BooleanVar(value=False)
        self.azimuth_start = tk.DoubleVar(value=0.0)
        self.azimuth_end = tk.DoubleVar(value=360.0)
        self.use_log = tk.BooleanVar(value=True)
        self.save_plot = tk.BooleanVar(value=True)
        self.bg_scale = tk.DoubleVar(value=1.0)
        self.bg_shift = tk.DoubleVar(value=0.0)
        self.allow_negative = tk.BooleanVar(value=False)

        self.input_mode = tk.StringVar(value="folder")
        self.selected_files = []
        self.selected_count = tk.StringVar(value="Not selected")

        self.calibrated = False
        self.cancel_event = None
        self.log_file = None
        self.process = None
        self.log_queue = None
        self.progress_queue = None

        self.create_widgets()
        self.start_btn.config(state='disabled')
        self.cancel_btn.config(state='disabled')
        self.after(100, self.show_calibration_dialog)

    def create_widgets(self):
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self.v_scrollbar = ttk.Scrollbar(self, orient='vertical', command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set)
        self.canvas.grid(row=0, column=0, sticky='nsew')
        self.v_scrollbar.grid(row=0, column=1, sticky='ns')
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.inner_frame = ttk.Frame(self.canvas, padding="10")
        self.canvas_window = self.canvas.create_window((0, 0), window=self.inner_frame, anchor='nw')
        self.inner_frame.bind('<Configure>', self._on_frame_configure)
        self.canvas.bind('<Configure>', self._on_canvas_configure)

        self.inner_frame.columnconfigure(1, weight=1)

        title_label = ttk.Label(self.inner_frame, text="SAXS Batch Processing Tool (pyFAI)", font=('Helvetica', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0,10))

        mode_frame = ttk.LabelFrame(self.inner_frame, text="Input mode", padding="5")
        mode_frame.grid(row=1, column=0, columnspan=3, sticky='ew', pady=5)
        folder_rb = ttk.Radiobutton(mode_frame, text="Scan folder", variable=self.input_mode,
                                    value="folder", command=self.toggle_input_mode)
        folder_rb.pack(side='left', padx=10)
        files_rb = ttk.Radiobutton(mode_frame, text="Select files", variable=self.input_mode,
                                   value="files", command=self.toggle_input_mode)
        files_rb.pack(side='left', padx=10)

        aux_frame = ttk.Frame(self.inner_frame)
        aux_frame.grid(row=2, column=0, columnspan=3, sticky='e', pady=5)

        self.help_btn = tk.Button(aux_frame, text="Help", command=self.show_help,
                                  bg="#9E9E9E", fg="white", font=('Helvetica', 9, 'bold'),
                                  height=1, width=10, relief='raised', bd=2)
        self.help_btn.pack(side='left', padx=5)

        self.dep_btn = tk.Button(aux_frame, text="Dependency status", command=self.open_dependency_status,
                                 bg="#FF9800", fg="white", font=('Helvetica', 9, 'bold'),
                                 height=1, width=10, relief='raised', bd=2)
        self.dep_btn.pack(side='left', padx=5)

        input_frame = ttk.Frame(self.inner_frame)
        input_frame.grid(row=3, column=0, columnspan=3, sticky='ew', pady=5)
        input_frame.columnconfigure(1, weight=1)
        ttk.Label(input_frame, text="Input image folder:").grid(row=0, column=0, sticky='w', padx=5)
        self.input_dir_entry = ttk.Entry(input_frame, textvariable=self.input_dir)
        self.input_dir_entry.grid(row=0, column=1, sticky='ew', padx=5)
        self.input_dir_btn = ttk.Button(input_frame, text="Browse", command=self.browse_input)
        self.input_dir_btn.grid(row=0, column=2, padx=5)

        self.file_select_frame = ttk.Frame(self.inner_frame)
        self.file_select_frame.grid(row=4, column=0, columnspan=3, sticky='ew', pady=5)
        self.file_select_frame.columnconfigure(1, weight=1)
        ttk.Label(self.file_select_frame, text="Specify files:").grid(row=0, column=0, sticky='w', padx=5)
        self.select_btn = ttk.Button(self.file_select_frame, text="Select files...", command=self.select_files)
        self.select_btn.grid(row=0, column=1, sticky='w', padx=5)
        ttk.Label(self.file_select_frame, text="(multiple: Ctrl/Shift+click)", foreground="gray").grid(row=0, column=2, sticky='w', padx=5)
        self.file_status = ttk.Label(self.file_select_frame, textvariable=self.selected_count, foreground="blue")
        self.file_status.grid(row=0, column=3, sticky='w', padx=5)

        output_frame = ttk.Frame(self.inner_frame)
        output_frame.grid(row=5, column=0, columnspan=3, sticky='ew', pady=5)
        output_frame.columnconfigure(1, weight=1)
        ttk.Label(output_frame, text="Output folder:").grid(row=0, column=0, sticky='w', padx=5)
        ttk.Entry(output_frame, textvariable=self.output_dir).grid(row=0, column=1, sticky='ew', padx=5)
        ttk.Button(output_frame, text="Browse", command=self.browse_output).grid(row=0, column=2, padx=5)

        # Geometry file selection (supports multiple formats)
        poni_frame = ttk.Frame(self.inner_frame)
        poni_frame.grid(row=6, column=0, columnspan=3, sticky='ew', pady=5)
        poni_frame.columnconfigure(1, weight=1)
        ttk.Label(poni_frame, text="Geometry file (.poni/.json/.cfg/.cif):").grid(row=0, column=0, sticky='w', padx=5)
        ttk.Entry(poni_frame, textvariable=self.poni_file).grid(row=0, column=1, sticky='ew', padx=5)
        ttk.Button(poni_frame, text="Browse", command=self.browse_poni).grid(row=0, column=2, padx=5)

        bg_frame = ttk.Frame(self.inner_frame)
        bg_frame.grid(row=7, column=0, columnspan=3, sticky='ew', pady=5)
        bg_frame.columnconfigure(1, weight=1)
        ttk.Label(bg_frame, text="Background file (optional):").grid(row=0, column=0, sticky='w', padx=5)
        ttk.Entry(bg_frame, textvariable=self.bg_file).grid(row=0, column=1, sticky='ew', padx=5)
        ttk.Button(bg_frame, text="Browse", command=self.browse_bg).grid(row=0, column=2, padx=5)

        mask_frame = ttk.Frame(self.inner_frame)
        mask_frame.grid(row=8, column=0, columnspan=3, sticky='ew', pady=5)
        mask_frame.columnconfigure(1, weight=1)
        ttk.Label(mask_frame, text="Mask file (optional):").grid(row=0, column=0, sticky='w', padx=5)
        ttk.Entry(mask_frame, textvariable=self.mask_file).grid(row=0, column=1, sticky='ew', padx=5)
        ttk.Button(mask_frame, text="Browse", command=self.browse_mask).grid(row=0, column=2, padx=5)

        integrate_frame = ttk.LabelFrame(self.inner_frame, text="Integration settings", padding="5")
        integrate_frame.grid(row=9, column=0, columnspan=3, sticky='ew', pady=10)
        integrate_frame.columnconfigure(1, weight=1)

        ttk.Label(integrate_frame, text="Number of bins:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        ttk.Entry(integrate_frame, textvariable=self.nbins, width=10).grid(row=0, column=1, sticky='w', padx=5, pady=5)

        ttk.Label(integrate_frame, text="Output unit:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        unit_combo = ttk.Combobox(integrate_frame, textvariable=self.unit,
                                   values=["q_nm^-1", "q_A^-1", "2theta_deg"], state="readonly", width=15)
        unit_combo.grid(row=1, column=1, sticky='w', padx=5, pady=5)

        self.sector_check = ttk.Checkbutton(integrate_frame, text="Sector integration (limit azimuthal range)",
                                           variable=self.use_sector, command=self.toggle_sector_inputs)
        self.sector_check.grid(row=2, column=0, columnspan=2, sticky='w', padx=5, pady=5)

        self.azimuth_frame = ttk.Frame(integrate_frame)
        self.azimuth_frame.grid(row=3, column=0, columnspan=2, sticky='w', padx=20, pady=5)
        ttk.Label(self.azimuth_frame, text="Start angle (deg):").pack(side='left')
        self.start_entry = ttk.Entry(self.azimuth_frame, textvariable=self.azimuth_start, width=8, state='disabled')
        self.start_entry.pack(side='left', padx=5)
        ttk.Label(self.azimuth_frame, text="End angle (deg):").pack(side='left')
        self.end_entry = ttk.Entry(self.azimuth_frame, textvariable=self.azimuth_end, width=8, state='disabled')
        self.end_entry.pack(side='left', padx=5)
        ttk.Label(self.azimuth_frame, text="(0~360)").pack(side='left', padx=5)

        self.advanced_frame = ttk.LabelFrame(self.inner_frame, text="Advanced settings", padding="5")
        self.advanced_frame.grid(row=10, column=0, columnspan=3, sticky='ew', pady=10)
        self.advanced_frame.columnconfigure(1, weight=1)

        ttk.Label(self.advanced_frame, text="Background scale factor:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        ttk.Entry(self.advanced_frame, textvariable=self.bg_scale, width=10).grid(row=0, column=1, sticky='w', padx=5, pady=5)

        shift_row = ttk.Frame(self.advanced_frame)
        shift_row.grid(row=1, column=0, columnspan=3, sticky='w', pady=5)
        ttk.Label(shift_row, text="Background shift:").pack(side='left', padx=5)
        ttk.Entry(shift_row, textvariable=self.bg_shift, width=10).pack(side='left', padx=5)
        self.preview_btn = ttk.Button(shift_row, text="Preview shift", command=self.open_preview)
        self.preview_btn.pack(side='left', padx=5)
        self.integrate_preview_btn = ttk.Button(shift_row, text="Integration preview", command=self.open_integrate_preview)
        self.integrate_preview_btn.pack(side='left', padx=5)

        ttk.Checkbutton(self.advanced_frame, text="Allow negative intensity (do not clip to 0)", variable=self.allow_negative).grid(row=2, column=0, columnspan=2, sticky='w', padx=5, pady=5)

        options_frame = ttk.Frame(self.inner_frame)
        options_frame.grid(row=11, column=0, columnspan=3, sticky='ew', pady=5)
        ttk.Checkbutton(options_frame, text="Use log scale (Y-axis)", variable=self.use_log).pack(side='left', padx=10)
        ttk.Checkbutton(options_frame, text="Save integration curve plot (PNG)", variable=self.save_plot).pack(side='left', padx=10)

        self.progress = ttk.Progressbar(self.inner_frame, orient='horizontal', length=600, mode='determinate')
        self.progress.grid(row=12, column=0, columnspan=3, sticky='ew', padx=10, pady=10)

        log_frame = ttk.LabelFrame(self.inner_frame, text="Processing log", padding="5")
        log_frame.grid(row=13, column=0, columnspan=3, sticky='nsew', padx=10, pady=5)
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log = tk.Text(log_frame, height=10, state='normal', wrap='word', font=('Consolas', 9))
        self.log.grid(row=0, column=0, sticky='nsew')
        scrollbar = ttk.Scrollbar(log_frame, command=self.log.yview)
        scrollbar.grid(row=0, column=1, sticky='ns')
        self.log['yscrollcommand'] = scrollbar.set

        self.btn_frame = ttk.Frame(self.inner_frame)
        self.btn_frame.grid(row=14, column=0, columnspan=3, pady=10)

        self.start_btn = tk.Button(self.btn_frame, text="Start processing", command=self.start_processing,
                                    bg="#4CAF50", fg="white", font=('Helvetica', 12, 'bold'),
                                    height=2, width=15, relief='raised', bd=3)
        self.start_btn.pack(side='left', padx=20)

        self.cancel_btn = tk.Button(self.btn_frame, text="Cancel", command=self.cancel_processing,
                                     bg="#f44336", fg="white", font=('Helvetica', 12, 'bold'),
                                     height=2, width=15, relief='raised', bd=3, state='disabled')
        self.cancel_btn.pack(side='left', padx=20)

        self.toggle_input_mode()
        self.inner_frame.rowconfigure(13, weight=1)

    def _on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))

    def _on_canvas_configure(self, event):
        canvas_width = event.width
        self.canvas.itemconfig(self.canvas_window, width=canvas_width)

    def toggle_input_mode(self):
        if self.input_mode.get() == "folder":
            self.input_dir_entry.config(state='normal')
            self.input_dir_btn.config(state='normal')
            self.select_btn.config(state='disabled')
        else:
            self.input_dir_entry.config(state='disabled')
            self.input_dir_btn.config(state='disabled')
            self.select_btn.config(state='normal')

    def toggle_sector_inputs(self):
        state = 'normal' if self.use_sector.get() else 'disabled'
        self.start_entry.config(state=state)
        self.end_entry.config(state=state)

    def browse_input(self):
        dirname = filedialog.askdirectory(title="Select folder containing raw images")
        if dirname:
            self.input_dir.set(dirname)

    def browse_output(self):
        dirname = filedialog.askdirectory(title="Select output folder")
        if dirname:
            self.output_dir.set(dirname)

    def browse_poni(self):
        filename = filedialog.askopenfilename(
            title="Select geometry calibration file",
            filetypes=[("Geometry files", "*.poni *.json *.cfg *.cif"), ("All files", "*.*")]
        )
        if filename:
            self.poni_file.set(filename)

    def browse_bg(self):
        filename = filedialog.askopenfilename(title="Select background file", filetypes=[("Image files", "*.tif *.edf *.cbf"), ("All files", "*.*")])
        if filename:
            self.bg_file.set(filename)

    def browse_mask(self):
        filename = filedialog.askopenfilename(title="Select mask file", filetypes=[("Mask files", "*.edf *.tif *.cbf"), ("All files", "*.*")])
        if filename:
            self.mask_file.set(filename)

    def select_files(self):
        initial_dir = self.input_dir.get() if self.input_dir.get() else os.path.expanduser("~")
        filetypes = [("Image files", "*.tif *.edf *.cbf"), ("All files", "*.*")]
        files = filedialog.askopenfilenames(
            title="Select image files to process (multiple: Ctrl/Shift+click)",
            initialdir=initial_dir,
            filetypes=filetypes
        )
        if files:
            self.selected_files = list(files)
            count = len(self.selected_files)
            self.selected_count.set(f"Selected {count} files")
            self._safe_log(f"User selected {count} files")
        else:
            self.selected_files = []
            self.selected_count.set("Not selected")

    def _safe_log(self, msg):
        self.after(0, self._insert_log, msg)

    def _insert_log(self, msg):
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)
        if hasattr(self, 'log_file') and self.log_file:
            try:
                self.log_file.write(msg + "\n")
                self.log_file.flush()
            except:
                pass
        if float(self.log.index('end-1c').split('.')[0]) > 1000:
            self.log.delete(1.0, "end-1000l")

    def _update_progress(self, value):
        self.progress['value'] = value

    def _close_log_file(self):
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()
            self.log_file = None

    def show_calibration_dialog(self):
        dialog = tk.Toplevel(self)
        dialog.title("Calibration reminder")
        dialog.geometry("450x200")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()

        label = tk.Label(dialog, text="Before integration, please calibrate using the calibration tool to generate a geometry file (.poni).\n"
                                       "You can launch the calibration tool (pyFAI-calib2) using the button below.\n"
                                       "After calibration, confirm and proceed to the main interface.",
                         justify='left', padx=10, pady=10)
        label.pack(pady=10)

        btn_frame = tk.Frame(dialog)
        btn_frame.pack(pady=10)

        def launch_calib():
            try:
                subprocess.Popen('start cmd /k pyfai-calib2', shell=True)
            except Exception as e:
                messagebox.showerror("Error", f"Unable to launch calibration tool: {e}")

        btn_launch = tk.Button(btn_frame, text="Launch calibration tool", command=launch_calib, width=15)
        btn_launch.pack(side='left', padx=10)

        def confirm_calibration():
            self.calibrated = True
            self.start_btn.config(state='normal')
            dialog.destroy()

        btn_confirm = tk.Button(btn_frame, text="I have calibrated, enter integrator", command=confirm_calibration, width=25, bg="lightgreen")
        btn_confirm.pack(side='left', padx=10)

        def on_closing():
            if messagebox.askyesno("Exit", "Calibration not confirmed, are you sure you want to exit?"):
                self.destroy()

        dialog.protocol("WM_DELETE_WINDOW", on_closing)
        self.wait_window(dialog)

    def open_preview(self):
        if not self.poni_file.get() or not os.path.exists(self.poni_file.get()):
            messagebox.showerror("Error", "Please set a valid geometry file first")
            return
        if self.use_sector.get():
            azimuth_range = (self.azimuth_start.get(), self.azimuth_end.get())
        else:
            azimuth_range = None
        config = {
            'poni_file': self.poni_file.get(),
            'nbins': self.nbins.get(),
            'unit': self.unit.get(),
            'use_sector': self.use_sector.get(),
            'azimuth_range': azimuth_range
        }
        preview = BackgroundShiftPreview(self, config, self.bg_shift.get(), self.set_bg_shift)

    def open_integrate_preview(self):
        if not self.poni_file.get() or not os.path.exists(self.poni_file.get()):
            messagebox.showerror("Error", "Please set a valid geometry file first")
            return
        if self.use_sector.get():
            azimuth_range = (self.azimuth_start.get(), self.azimuth_end.get())
        else:
            azimuth_range = None
        config = {
            'poni_file': self.poni_file.get(),
            'nbins': self.nbins.get(),
            'unit': self.unit.get(),
            'use_sector': self.use_sector.get(),
            'azimuth_range': azimuth_range,
            'initial_dir': self.input_dir.get() if self.input_dir.get() else os.path.expanduser("~")
        }
        preview = IntegrationPreview(self, config)

    def set_bg_shift(self, value):
        self.bg_shift.set(value)
        self._safe_log(f"Background shift set to: {value}")

    def show_help(self):
        help_text = """SynXS - SAXS Batch Processing Tool User Guide

1. Calibration: Must use pyFAI-calib2 on a standard sample image to generate geometry file.
   Supported formats: .poni, .json, .cfg, .cif

2. Set parameters:
   - Input mode: Scan folder processes all .tif/.edf/.cbf files in the folder;
                Select files allows manual selection of multiple files.
   - Output folder: Stores resulting .dat files and optional plots.
   - Geometry file: Select the calibration geometry file.
   - Background file (optional): Select a background image; program will subtract with scale and shift.
   - Mask file (optional): Select a mask image to exclude bad pixels or beamstop.
   - Number of bins: Data points in output curve.
   - Output unit: q_nm^-1, q_A^-1, 2theta_deg.
   - Sector integration: Check to limit azimuthal range (0~360 degrees).
   - Advanced settings: Background scale factor, shift, allow negative intensity.
   - Log scale: Y-axis log when plotting.
   - Save integration curve plot: Generate PNG for each file.

3. Preview (optional):
   - Background shift preview: Load sample and background files, adjust shift in real time.
   - Integration preview: Select multiple images to overlay their integrated curves for comparison.

4. Start processing: Click Start, program will process all images and save results.
   Progress shown in log area; can cancel at any time.

Optimization features:
   - PyFAI OpenCL acceleration disabled for stability.
   - Multi-processing isolates tasks, preventing memory leaks in main program.
   - Enhanced exception handling, all errors logged.
   - Supports .poni, .json, .cfg, .cif geometry files.

For issues, contact the author."""
        messagebox.showinfo("Help", help_text)

    def open_dependency_status(self):
        DependencyStatusDialog(self)

    def start_processing(self):
        if self.input_mode.get() == "folder":
            if not self.input_dir.get():
                messagebox.showerror("Error", "Please select input folder")
                return
        else:
            if not self.selected_files:
                messagebox.showerror("Error", "Please select files to process")
                return

        if not self.output_dir.get():
            messagebox.showerror("Error", "Please select output folder")
            return
        if not self.poni_file.get():
            messagebox.showerror("Error", "Please select geometry file")
            return
        if not os.path.exists(self.poni_file.get()):
            messagebox.showerror("Error", "Geometry file does not exist")
            return

        os.makedirs(self.output_dir.get(), exist_ok=True)

        log_filename = f"processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.log_file_path = os.path.join(self.output_dir.get(), log_filename)
        try:
            self.log_file = open(self.log_file_path, 'w', encoding='utf-8')
        except Exception as e:
            messagebox.showerror("Error", f"Unable to create log file: {e}")
            return

        config = {
            'poni': self.poni_file.get(),
            'output_dir': self.output_dir.get(),
            'bg_file': self.bg_file.get() if self.bg_file.get() else None,
            'mask_file': self.mask_file.get() if self.mask_file.get() else None,
            'nbins': self.nbins.get(),
            'unit': self.unit.get(),
            'use_sector': self.use_sector.get(),
            'azimuth_start': self.azimuth_start.get(),
            'azimuth_end': self.azimuth_end.get(),
            'use_log': self.use_log.get(),
            'save_plot': self.save_plot.get(),
            'bg_scale': self.bg_scale.get(),
            'bg_shift': self.bg_shift.get(),
            'allow_negative': self.allow_negative.get(),
        }

        if self.input_mode.get() == "folder":
            image_files = glob.glob(os.path.join(self.input_dir.get(), "*.tif")) + \
                          glob.glob(os.path.join(self.input_dir.get(), "*.edf")) + \
                          glob.glob(os.path.join(self.input_dir.get(), "*.cbf"))
            image_files = sorted(image_files)
        else:
            image_files = self.selected_files

        if not image_files:
            messagebox.showerror("Error", "No image files found!")
            self._close_log_file()
            return

        self.cancel_event = mp.Event()
        self.cancel_event.clear()

        self.start_btn.config(state='disabled')
        self.cancel_btn.config(state='normal')
        self.progress['value'] = 0
        self.log.delete(1.0, tk.END)

        self.log_queue = mp.Queue()
        self.progress_queue = mp.Queue()

        self.process = mp.Process(
            target=process_images_worker,
            args=(config, image_files, self.log_queue, self.progress_queue, self.cancel_event),
            daemon=True
        )
        self.process.start()

        self.after(100, self._check_queues)

    def _check_queues(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self._insert_log(msg)
                if self.log_file:
                    self.log_file.write(msg + "\n")
                    self.log_file.flush()
        except:
            pass

        try:
            while True:
                prog = self.progress_queue.get_nowait()
                if 'max' in prog:
                    self.progress['maximum'] = prog['max']
                if 'value' in prog:
                    self.progress['value'] = prog['value']
                if prog.get('done'):
                    self._on_process_finished()
                    return
        except:
            pass

        if self.process and not self.process.is_alive():
            self._on_process_finished()
            return

        self.after(100, self._check_queues)

    def _on_process_finished(self):
        self.start_btn.config(state='normal')
        self.cancel_btn.config(state='disabled')
        self.progress.configure(value=0)
        self._close_log_file()
        if self.process:
            self.process.join(timeout=1)
            self.process = None
        gc.collect()

    def cancel_processing(self):
        if self.cancel_event:
            self.cancel_event.set()
        if self.process and self.process.is_alive():
            self.process.terminate()
        self._safe_log("User cancelled, stopping...")
        self._on_process_finished()


# ==================== Data Processing Tool (integrated from data_processing.py, font adjusted to 10pt) ====================

# ====================== Helper functions ======================
def q_to_2theta(q, wavelength):
    """
    Convert Q value (Å⁻¹) to 2θ angle (degrees).
    Formula: 2θ = 2 * arcsin( Q * λ / (4π) ), result in degrees.
    If argument out of [-1,1], return NaN.
    """
    arg = q * wavelength / (4 * np.pi)
    if arg > 1:
        if arg - 1 < 1e-12:
            arg = 1.0
        else:
            return np.nan
    elif arg < -1:
        if -1 - arg < 1e-12:
            arg = -1.0
        else:
            return np.nan
    theta_rad = np.arcsin(arg)
    two_theta_deg = 2 * np.degrees(theta_rad)
    return two_theta_deg

# ====================== Tab 1: Q→2θ conversion ======================
class Q2ThetaTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        # Input file
        row1 = ttk.Frame(self)
        row1.pack(fill=tk.X, pady=5)
        ttk.Label(row1, text="Input file (*.dat):").pack(side=tk.LEFT, padx=5)
        self.q_input_path = tk.StringVar()
        ttk.Entry(row1, textvariable=self.q_input_path, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(row1, text="Browse...", command=self.browse_input).pack(side=tk.LEFT)

        # Wavelength
        row2 = ttk.Frame(self)
        row2.pack(fill=tk.X, pady=5)
        ttk.Label(row2, text="X-ray wavelength (Å):").pack(side=tk.LEFT, padx=5)
        self.q_wavelength = tk.StringVar(value="1.5406")
        ttk.Entry(row2, textvariable=self.q_wavelength, width=15).pack(side=tk.LEFT, padx=5)

        # Output file
        row3 = ttk.Frame(self)
        row3.pack(fill=tk.X, pady=5)
        ttk.Label(row3, text="Output file (*.asc):").pack(side=tk.LEFT, padx=5)
        self.q_output_path = tk.StringVar()
        ttk.Entry(row3, textvariable=self.q_output_path, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(row3, text="Browse...", command=self.browse_output).pack(side=tk.LEFT)

        # Process button
        ttk.Button(self, text="Start conversion", command=self.process).pack(pady=10)

        # Log area (font 10pt)
        log_frame = ttk.LabelFrame(self, text="Conversion log", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.q_log_text = tk.Text(log_frame, wrap=tk.WORD, height=8, font=('Helvetica', 10))
        self.q_log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.q_log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.q_log_text.config(yscrollcommand=scrollbar.set)

    def browse_input(self):
        filename = filedialog.askopenfilename(
            title="Select input DAT file",
            filetypes=[("DAT files", "*.dat"), ("All files", "*.*")]
        )
        if filename:
            self.q_input_path.set(filename)
            if not self.q_output_path.get():
                base, _ = os.path.splitext(filename)
                self.q_output_path.set(base + ".asc")

    def browse_output(self):
        filename = filedialog.asksaveasfilename(
            title="Save converted ASC file",
            defaultextension=".asc",
            filetypes=[("ASC files", "*.asc"), ("All files", "*.*")]
        )
        if filename:
            self.q_output_path.set(filename)

    def log(self, message):
        self.q_log_text.insert(tk.END, message + "\n")
        self.q_log_text.see(tk.END)
        self.update_idletasks()

    def process(self):
        self.q_log_text.delete(1.0, tk.END)
        input_file = self.q_input_path.get().strip()
        output_file = self.q_output_path.get().strip()
        wavelength_str = self.q_wavelength.get().strip()

        if not input_file:
            messagebox.showerror("Error", "Please select input file.")
            return
        if not output_file:
            messagebox.showerror("Error", "Please specify output file.")
            return
        if not os.path.isfile(input_file):
            messagebox.showerror("Error", f"Input file does not exist: {input_file}")
            return
        try:
            wavelength = float(wavelength_str)
        except ValueError:
            messagebox.showerror("Error", "Wavelength must be a number.")
            return

        try:
            data = np.loadtxt(input_file)
        except Exception as e:
            self.log(f"Read error: {e}")
            messagebox.showerror("Read error", f"Unable to read file:\n{e}")
            return

        if data.ndim != 2 or data.shape[1] < 2:
            self.log("Error: Input file must contain at least two columns of numeric data.")
            messagebox.showerror("Format error", "Input file must contain at least two columns of numeric data.")
            return

        q = data[:, 0]
        y = data[:, 1]

        two_theta = np.array([q_to_2theta(val, wavelength) for val in q])
        invalid_mask = np.isnan(two_theta)
        if np.any(invalid_mask):
            reply = messagebox.askyesno(
                "Warning",
                f"Found {np.sum(invalid_mask)} Q values out of convertible range (Qλ/(4π) > 1).\n"
                "These rows will be skipped. Continue?"
            )
            if not reply:
                return
            valid_mask = ~invalid_mask
            two_theta = two_theta[valid_mask]
            y = y[valid_mask]
            if len(two_theta) == 0:
                self.log("Error: No valid data to save.")
                messagebox.showerror("Error", "No valid data to save.")
                return
            self.log(f"Skipped {np.sum(invalid_mask)} invalid rows.")

        try:
            np.savetxt(output_file, np.column_stack((two_theta, y)),
                       fmt='%.6f', delimiter='\t', header='2Theta(deg)\tIntensity')
        except Exception as e:
            self.log(f"Save error: {e}")
            messagebox.showerror("Save error", f"Unable to save file:\n{e}")
            return

        self.log(f"Conversion complete! Output file: {output_file}")
        messagebox.showinfo("Success", f"Conversion complete!\nOutput file: {output_file}")

# ====================== Tab 2: Automatic peak finding + report export ======================
class PeakFinderTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        self.file_paths = []
        self.current_data = None
        self.current_file = ""
        self.log_mode = tk.BooleanVar(value=False)
        self.add_point_mode = tk.BooleanVar(value=False)

        self.peak_results = {}
        self.manual_points = {}

        self.init_ui()
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)

    def init_ui(self):
        # Parameter area
        top_frame = ttk.LabelFrame(self, text="Peak finding parameters", padding=10)
        top_frame.pack(fill=tk.X, padx=10, pady=5)

        self.log_check = ttk.Checkbutton(
            top_frame, text="Log Y-axis (Log10 Y)", variable=self.log_mode,
            command=self.on_log_mode_toggle
        )
        self.log_check.grid(row=0, column=0, columnspan=2, sticky=tk.W)

        ttk.Label(top_frame, text="Minimum prominence:").grid(row=1, column=0, sticky=tk.W, pady=(10,0))
        self.prom_entry = ttk.Entry(top_frame, width=10)
        self.prom_entry.grid(row=1, column=1, padx=5, pady=(10,0))
        self.prom_entry.insert(0, "0.2")

        ttk.Label(top_frame, text="Minimum width (points):").grid(row=1, column=2, sticky=tk.W, padx=(20,0), pady=(10,0))
        self.width_entry = ttk.Entry(top_frame, width=10)
        self.width_entry.grid(row=1, column=3, padx=5, pady=(10,0))
        self.width_entry.insert(0, "1")

        ttk.Label(top_frame, text="*Prominence = height difference between peak and lowest contour; for log mode calculated on log10(Y)",
                  foreground="gray").grid(row=2, column=0, columnspan=4, sticky=tk.W, pady=(5,0))

        # Main layout: left file list + right plot
        main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        left_frame = ttk.Frame(main_pane, width=250)
        main_pane.add(left_frame, weight=1)

        right_frame = ttk.Frame(main_pane)
        main_pane.add(right_frame, weight=3)

        # Left controls
        ttk.Label(left_frame, text="Imported files:").pack(anchor=tk.W)
        # font 10pt
        self.file_listbox = tk.Listbox(left_frame, height=14, font=('Helvetica', 10))
        self.file_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        self.file_listbox.bind('<<ListboxSelect>>', self.on_file_select)

        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="Import files", command=self.import_files).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Remove selected", command=self.remove_selected).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Clear list", command=self.clear_list).pack(side=tk.LEFT, padx=2)

        ttk.Button(left_frame, text="Preview current file", command=self.preview_peaks).pack(fill=tk.X, pady=2)
        ttk.Button(left_frame, text="One-click peak find (all files)", command=self.batch_find_peaks).pack(fill=tk.X, pady=2)
        ttk.Button(left_frame, text="Export peak plots to PDF", command=self.export_peaks_to_pdf).pack(fill=tk.X, pady=2)

        add_point_frame = ttk.LabelFrame(left_frame, text="Manual point addition", padding=5)
        add_point_frame.pack(fill=tk.X, pady=5)
        self.add_check = ttk.Checkbutton(
            add_point_frame, text="Add point mode (click on plot to add annotation)", variable=self.add_point_mode
        )
        self.add_check.pack(anchor=tk.W)
        ttk.Button(add_point_frame, text="Clear current file manual points", command=self.clear_current_manual_points).pack(fill=tk.X, pady=2)
        ttk.Button(add_point_frame, text="Clear all manual points", command=self.clear_all_manual_points).pack(fill=tk.X, pady=2)

        self.status_label = ttk.Label(left_frame, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(fill=tk.X, pady=5)

        # Right plot
        self.figure, self.ax = plt.subplots(figsize=(7, 5))
        self.canvas = FigureCanvasTkAgg(self.figure, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ax.set_title("Please import a DAT file and preview")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.figure.tight_layout()

    # ---------- Data preparation ----------
    def on_log_mode_toggle(self):
        if self.current_data is not None:
            self.draw_current_file()

    def get_plot_data(self, y_original):
        if self.log_mode.get():
            if np.any(y_original <= 0):
                messagebox.showwarning("Log warning", "Y data has ≤0 values, log will produce breaks.")
                y = np.where(y_original > 0, y_original, np.nan)
                return np.log10(y), "log10(Y)"
            return np.log10(y_original), "log10(Y)"
        return y_original, "Y"

    def prepare_y_for_analysis(self, y_original):
        if self.log_mode.get():
            y = y_original.copy()
            y[y <= 0] = 1e-12
            return np.log10(y), "log10(Y)"
        return y_original, "Y"

    # ---------- File list operations ----------
    def import_files(self):
        files = filedialog.askopenfilenames(title="Select DAT files", filetypes=[("DAT files", "*.dat"), ("All files", "*.*")])
        for f in files:
            if f not in self.file_paths:
                self.file_paths.append(f)
                self.file_listbox.insert(tk.END, os.path.basename(f))
        if files:
            self.status_label.config(text=f"Imported {len(files)} files")
            if self.file_listbox.size() > 0 and not self.file_listbox.curselection():
                self.file_listbox.selection_set(0)
                self.on_file_select()

    def remove_selected(self):
        selection = self.file_listbox.curselection()
        if selection:
            index = selection[0]
            self.file_listbox.delete(index)
            removed_path = self.file_paths.pop(index)
            if removed_path in self.peak_results:
                del self.peak_results[removed_path]
            if removed_path in self.manual_points:
                del self.manual_points[removed_path]
            self.status_label.config(text="File removed")
            if self.file_listbox.size() == 0:
                self.clear_plot()
        else:
            messagebox.showwarning("Warning", "Please select a file in the list first")

    def clear_list(self):
        self.file_paths.clear()
        self.file_listbox.delete(0, tk.END)
        self.peak_results.clear()
        self.manual_points.clear()
        self.clear_plot()
        self.status_label.config(text="List cleared, peak results and manual points removed")

    def clear_plot(self):
        self.ax.clear()
        self.ax.set_title("Please import a DAT file and preview")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("log10(Y)" if self.log_mode.get() else "Y")
        self.canvas.draw()

    def on_file_select(self, event=None):
        selection = self.file_listbox.curselection()
        if not selection:
            return
        index = selection[0]
        file_path = self.file_paths[index]
        try:
            data = np.loadtxt(file_path)
            if data.ndim != 2 or data.shape[1] < 2:
                raise ValueError("Data must have at least two columns")
            x, y_orig = data[:, 0], data[:, 1]
            self.current_data = (x, y_orig)
            self.current_file = file_path
            self.draw_current_file()
        except Exception as e:
            messagebox.showerror("Read error", f"Unable to read file:\n{file_path}\nError: {e}")
            self.status_label.config(text="File read failed")

    # ---------- Plotting ----------
    def draw_current_file(self):
        if self.current_data is None:
            return
        x, y_orig = self.current_data
        y_plot, ylabel = self.get_plot_data(y_orig)
        file_path = self.current_file

        if file_path in self.peak_results:
            res = self.peak_results[file_path]
            self.plot_data_with_annotations(
                res['x'], y_plot, res['peaks'], res['properties'], ylabel, res['file_name']
            )
            self.status_label.config(text=f"Showing peaks: {res['file_name']} ({len(res['peaks'])} peaks)")
        else:
            title = f"File: {os.path.basename(file_path)}"
            self.plot_data(x, y_plot, title=title, ylabel=ylabel)
            self.status_label.config(text=f"Loaded: {os.path.basename(file_path)} (no peaks)")

    def plot_data(self, x, y, peaks=None, title="Data", ylabel="Y"):
        self.ax.clear()
        self.ax.plot(x, y, 'b-', linewidth=1)
        if peaks is not None and len(peaks) > 0:
            self.ax.plot(x[peaks], y[peaks], 'ro', markersize=5, label='Peaks')
        # title font 10pt
        self.ax.set_title(title, fontsize=10)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel(ylabel)
        if peaks is not None and len(peaks) > 0:
            self.ax.legend(loc='best')
        self.draw_manual_points()
        self.figure.tight_layout()
        self.canvas.draw()

    def plot_data_with_annotations(self, x, y_plot, peaks, properties, ylabel, file_name):
        self.ax.clear()
        self.ax.plot(x, y_plot, 'b-', linewidth=1)
        self.ax.plot(x[peaks], y_plot[peaks], 'ro', markersize=5, label='Peaks')

        prominences = properties['prominences']
        widths_x = self.compute_widths_in_x(x, properties)

        for i, idx in enumerate(peaks):
            px = x[idx]
            py = y_plot[idx]
            prom = prominences[i]
            w = widths_x[i]
            label = f"X:{px:.2f}  Y:{py:.2f}\nProm:{prom:.2f}\nFW:{w:.2f}"
            # annotation font 9pt
            self.ax.annotate(label, (px, py),
                             textcoords="offset points", xytext=(10, 10),
                             fontsize=9, color='darkred',
                             arrowprops=dict(arrowstyle='->', color='gray', lw=0.5),
                             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

        # title font 10pt
        self.ax.set_title(f"File: {file_name}  -  {len(peaks)} peaks found", fontsize=10)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel(ylabel)
        self.ax.legend(loc='best')
        self.draw_manual_points()
        self.figure.tight_layout()
        self.canvas.draw()

    def draw_manual_points(self):
        if self.current_file not in self.manual_points:
            return
        pts = self.manual_points[self.current_file]
        if not pts:
            return
        pts_arr = np.array(pts)
        self.ax.plot(pts_arr[:,0], pts_arr[:,1], 'gs', markersize=6, label='Manual')
        for xp, yp in pts:
            self.ax.annotate(f"X:{xp:.2f}  Y:{yp:.2f}", (xp, yp),
                             textcoords="offset points", xytext=(5, -15),
                             fontsize=9, color='green',
                             bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7))

    def compute_widths_in_x(self, x, properties):
        widths = properties['widths']
        left_ips = properties['left_ips']
        right_ips = properties['right_ips']
        try:
            x_left = np.interp(left_ips, np.arange(len(x)), x)
            x_right = np.interp(right_ips, np.arange(len(x)), x)
            return x_right - x_left
        except:
            return widths

    def on_canvas_click(self, event):
        if not self.add_point_mode.get():
            return
        if event.inaxes != self.ax:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        if self.current_file not in self.manual_points:
            self.manual_points[self.current_file] = []
        self.manual_points[self.current_file].append((x, y))
        self.draw_current_file()

    def clear_current_manual_points(self):
        if self.current_file in self.manual_points:
            self.manual_points[self.current_file].clear()
            self.draw_current_file()
            self.status_label.config(text=f"Cleared manual points for {os.path.basename(self.current_file)}")
        else:
            self.status_label.config(text="No manual points for current file")

    def clear_all_manual_points(self):
        self.manual_points.clear()
        self.draw_current_file()
        self.status_label.config(text="Cleared all manual points")

    def get_peak_params(self):
        try:
            prominence = float(self.prom_entry.get())
        except ValueError:
            messagebox.showerror("Parameter error", "Minimum prominence must be a number")
            return None
        try:
            width = float(self.width_entry.get())
        except ValueError:
            messagebox.showerror("Parameter error", "Minimum width must be a number")
            return None
        return {'prominence': prominence, 'width': width, 'rel_height': 0.5}

    def preview_peaks(self):
        if self.current_data is None:
            messagebox.showwarning("No data", "Please select a DAT file first")
            return
        params = self.get_peak_params()
        if params is None:
            return
        x, y_orig = self.current_data
        y_analysis, _ = self.prepare_y_for_analysis(y_orig)
        y_plot, ylabel = self.get_plot_data(y_orig)

        try:
            peaks, properties = find_peaks(y_analysis, **params)
        except Exception as e:
            messagebox.showerror("Peak finding error", f"Error during peak finding:\n{e}")
            return

        file_name = os.path.basename(self.current_file)
        self.plot_data_with_annotations(x, y_plot, peaks, properties, ylabel, file_name)
        self.status_label.config(text=f"Preview: found {len(peaks)} peaks in {file_name}")

    def batch_find_peaks(self):
        if not self.file_paths:
            messagebox.showwarning("No files", "Please import DAT files first")
            return

        params = self.get_peak_params()
        if params is None:
            return

        self.peak_results.clear()
        total_peaks = 0
        summary = []
        last_result = None

        for file_path in self.file_paths:
            try:
                data = np.loadtxt(file_path)
                x, y_orig = data[:, 0], data[:, 1]
            except Exception:
                self.status_label.config(text=f"Skipped file {os.path.basename(file_path)}: read error")
                continue

            y_analysis, _ = self.prepare_y_for_analysis(y_orig)
            y_plot, ylabel = self.get_plot_data(y_orig)

            peaks, properties = find_peaks(y_analysis, **params)

            self.peak_results[file_path] = {
                'x': x,
                'y_plot': y_plot,
                'y_orig': y_orig,
                'y_analysis': y_analysis,
                'peaks': peaks,
                'properties': properties,
                'ylabel': ylabel,
                'file_name': os.path.basename(file_path)
            }

            total_peaks += len(peaks)
            summary.append(f"  {os.path.basename(file_path)}: {len(peaks)} peaks")
            self.status_label.config(text=f"Peak processing: {os.path.basename(file_path)} ...")
            last_result = self.peak_results[file_path]

        self.status_label.config(text=f"One-click peak find completed! Total {total_peaks} peaks (click files to view)")
        summary_text = f"Files processed: {len(self.file_paths)}\nTotal peaks: {total_peaks}\n\n" + "\n".join(summary)
        messagebox.showinfo("Peak results", summary_text)

        if last_result:
            last_index = len(self.file_paths) - 1
            self.file_listbox.selection_clear(0, tk.END)
            self.file_listbox.selection_set(last_index)
            self.file_listbox.see(last_index)
            self.current_file = self.file_paths[last_index]
            self.current_data = (last_result['x'], last_result['y_orig'])
            self.draw_current_file()

    def export_peaks_to_pdf(self):
        if not self.peak_results:
            messagebox.showwarning("No peak results", "Please run 'One-click peak find' first, then export.")
            return

        save_path = filedialog.asksaveasfilename(
            title="Save PDF report",
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")]
        )
        if not save_path:
            return

        include_manual = messagebox.askyesno("Manual points", "Include manually added points in the PDF?")

        try:
            pdf = PdfPages(save_path)
        except Exception as e:
            messagebox.showerror("PDF creation failed", str(e))
            return

        # Cover page (fonts within PDF remain unchanged)
        fig_cover = plt.figure(figsize=(8.27, 11.69))
        fig_cover.text(0.5, 0.95, "Peak Finding Report", ha='center', va='top',
                       fontsize=24, fontweight='bold')
        fig_cover.text(0.5, 0.90, "DAT File Peak Analysis", ha='center', va='top',
                       fontsize=14, color='gray')
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig_cover.text(0.5, 0.84, f"Report generated: {now}", ha='center', fontsize=10)

        total_files = len(self.peak_results)
        total_peaks = sum(len(res['peaks']) for res in self.peak_results.values())

        fig_cover.text(0.1, 0.77, "Summary", fontsize=16, fontweight='bold')
        fig_cover.text(0.1, 0.73, f"Total files processed : {total_files}")
        fig_cover.text(0.1, 0.70, f"Total peaks detected  : {total_peaks}")

        fig_cover.text(0.1, 0.63, "Peak Finding Parameters", fontsize=16, fontweight='bold')
        prominence = self.prom_entry.get()
        width = self.width_entry.get()
        log_status = "Enabled (log10)" if self.log_mode.get() else "Disabled"
        fig_cover.text(0.1, 0.59, f"Minimum prominence (Prominence) : {prominence}")
        fig_cover.text(0.1, 0.56, f"Minimum width (data points)    : {width}")
        fig_cover.text(0.1, 0.53, f"Logarithmic Y scale            : {log_status}")
        fig_cover.text(0.1, 0.50, "Algorithm : scipy.signal.find_peaks", fontsize=9, style='italic')
        fig_cover.text(0.1, 0.47,
                       "Peak definition: prominence = vertical distance from peak to its lowest contour line;",
                       fontsize=8)
        fig_cover.text(0.1, 0.45,
                       "                 width = full width at half maximum (FWHM) relative to the detected baseline.",
                       fontsize=8)

        fig_cover.text(0.1, 0.38, "Legend", fontsize=16, fontweight='bold')
        fig_cover.text(0.1, 0.34, "Blue solid line  : Original data curve", fontsize=10)
        fig_cover.text(0.1, 0.31, "Red circle       : Detected peak (automatic)", fontsize=10)
        fig_cover.text(0.1, 0.28, "Green square     : User-added manual point", fontsize=10)
        fig_cover.text(0.1, 0.25, "Annotations show : X coordinate, Y coordinate, Prom (prominence), FWHM (width)",
                       fontsize=10)

        fig_cover.text(0.5, 0.08, "File list starts on next page →", ha='center', fontsize=9, color='gray')
        pdf.savefig(fig_cover)
        plt.close(fig_cover)

        # File list page (two columns)
        file_list = []
        for fp in self.file_paths:
            if fp in self.peak_results:
                res = self.peak_results[fp]
                file_list.append((res['file_name'], len(res['peaks'])))

        rows_per_column = 35
        fontsize_list = 14
        y_top = 0.88
        y_step = 0.020
        x_left_col = 0.05
        x_right_col = 0.52

        if file_list:
            total = len(file_list)
            items_per_page = rows_per_column * 2
            num_pages = -(-total // items_per_page)

            for page_idx in range(num_pages):
                fig_list = plt.figure(figsize=(8.27, 11.69))
                fig_list.text(0.5, 0.95, "File List", ha='center', fontsize=20, fontweight='bold')
                if num_pages > 1:
                    fig_list.text(0.5, 0.91, f"Page {page_idx+1} / {num_pages}", ha='center', fontsize=10, color='gray')

                start_idx = page_idx * items_per_page
                end_idx = min(start_idx + items_per_page, total)
                page_items = file_list[start_idx:end_idx]

                left_items = page_items[:rows_per_column]
                right_items = page_items[rows_per_column:]

                y_pos = y_top
                for i, (fname, pcount) in enumerate(left_items):
                    global_idx = start_idx + i + 1
                    line = f"{global_idx}. {fname}   ({pcount} peaks)"
                    fig_list.text(x_left_col, y_pos, line, fontsize=fontsize_list, va='top')
                    y_pos -= y_step

                y_pos = y_top
                for i, (fname, pcount) in enumerate(right_items):
                    global_idx = start_idx + rows_per_column + i + 1
                    line = f"{global_idx}. {fname}   ({pcount} peaks)"
                    fig_list.text(x_right_col, y_pos, line, fontsize=fontsize_list, va='top')
                    y_pos -= y_step

                pdf.savefig(fig_list)
                plt.close(fig_list)
        else:
            fig_list = plt.figure(figsize=(8.27, 11.69))
            fig_list.text(0.5, 0.5, "No files to list", ha='center', fontsize=14, color='red')
            pdf.savefig(fig_list)
            plt.close(fig_list)

        # Detailed pages for each file
        processed = 0
        for file_path, res in self.peak_results.items():
            x = res['x']
            y_plot = res['y_plot']
            peaks = res['peaks']
            properties = res['properties']
            ylabel = res['ylabel']
            file_name = res['file_name']
            prominences = properties['prominences']
            widths_x = self.compute_widths_in_x(x, properties) if len(peaks) > 0 else []

            fig = plt.figure(figsize=(8.27, 11.69))
            ax_plot = fig.add_axes([0.1, 0.40, 0.8, 0.55])
            ax_table_peaks = fig.add_axes([0.05, 0.18, 0.9, 0.18])
            ax_table_peaks.axis('off')
            ax_table_manual = fig.add_axes([0.05, 0.03, 0.9, 0.12])
            ax_table_manual.axis('off')

            ax_plot.plot(x, y_plot, 'b-', linewidth=1)
            if len(peaks) > 0:
                ax_plot.plot(x[peaks], y_plot[peaks], 'ro', markersize=5, label='Peaks')
                for i, idx in enumerate(peaks):
                    px = x[idx]
                    py = y_plot[idx]
                    prom = prominences[i]
                    w = widths_x[i]
                    label = f"X:{px:.2f}  Y:{py:.2f}\nProm:{prom:.2f}\nFW:{w:.2f}"
                    ax_plot.annotate(label, (px, py),
                                     textcoords="offset points", xytext=(10, 10),
                                     fontsize=9, color='darkred',
                                     arrowprops=dict(arrowstyle='->', color='gray', lw=0.5),
                                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
            if include_manual and file_path in self.manual_points and self.manual_points[file_path]:
                manual_pts = np.array(self.manual_points[file_path])
                ax_plot.plot(manual_pts[:,0], manual_pts[:,1], 'gs', markersize=6, label='Manual')
                for xp, yp in self.manual_points[file_path]:
                    ax_plot.annotate(f"X:{xp:.2f}  Y:{yp:.2f}", (xp, yp),
                                     textcoords="offset points", xytext=(5, -15),
                                     fontsize=9, color='green',
                                     bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7))
            ax_plot.set_title(f"File: {file_name}  -  {len(peaks)} peaks found")
            ax_plot.set_xlabel("X")
            ax_plot.set_ylabel(ylabel)
            if len(peaks) > 0:
                ax_plot.legend(loc='best')

            if len(peaks) > 0:
                col_labels = ['Peak', 'X', 'Y', 'Prominence', 'FWHM']
                cell_text = []
                for i, idx in enumerate(peaks):
                    cell_text.append([
                        i+1,
                        f"{x[idx]:.4f}",
                        f"{y_plot[idx]:.4f}",
                        f"{prominences[i]:.4f}",
                        f"{widths_x[i]:.4f}"
                    ])
                table = ax_table_peaks.table(cellText=cell_text, colLabels=col_labels,
                                             loc='center', cellLoc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1, 1.2)
                for (row, col), cell in table.get_celld().items():
                    cell.set_edgecolor('gray')
                    if row == 0:
                        cell.get_text().set_fontweight('bold')
                        cell.set_facecolor('#f0f0f0')
            else:
                ax_table_peaks.text(0.5, 0.5, "No peaks found", ha='center', va='center',
                                    transform=ax_table_peaks.transAxes, fontsize=12, color='red')

            if include_manual and file_path in self.manual_points and self.manual_points[file_path]:
                manual_data = self.manual_points[file_path]
                m_col_labels = ['Point', 'X', 'Y']
                m_cell_text = [[i+1, f"{xp:.4f}", f"{yp:.4f}"] for i, (xp, yp) in enumerate(manual_data)]
                table_m = ax_table_manual.table(cellText=m_cell_text, colLabels=m_col_labels,
                                                loc='center', cellLoc='center')
                table_m.auto_set_font_size(False)
                table_m.set_fontsize(7)
                table_m.scale(1, 1.2)
                for (row, col), cell in table_m.get_celld().items():
                    cell.set_edgecolor('gray')
                    if row == 0:
                        cell.get_text().set_fontweight('bold')
                        cell.set_facecolor('#f0f0f0')
                ax_table_manual.set_title("Manual Points", fontsize=9, pad=2)
            else:
                ax_table_manual.set_visible(False)

            pdf.savefig(fig)
            plt.close(fig)
            processed += 1
            self.status_label.config(text=f"Exporting PDF: {processed}/{total_files} pages...")

        pdf.close()
        self.status_label.config(text=f"PDF export complete! {total_files} files, saved to {os.path.basename(save_path)}")
        messagebox.showinfo("Export successful", f"Peak report saved to:\n{save_path}")

# ====================== Tab 3: Data processing (multi-function) ======================
class DataProcessTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        # Operation type
        self.operation = tk.StringVar(value="drop")   # "drop", "calc", "remove_braces"
        # Mode
        self.mode = tk.StringVar(value="single")      # "single" or "batch"
        # Input columns
        self.input_columns = tk.IntVar(value=3)       # 2 or 3
        # File paths
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.batch_dir = tk.StringVar()
        self.batch_output_dir = tk.StringVar()
        self.output_suffix = tk.StringVar(value="_out")

        self.init_ui()

    def init_ui(self):
        # ========== Operation type ==========
        op_frame = ttk.LabelFrame(self, text="Operation type", padding="5")
        op_frame.pack(fill=tk.X, pady=5)

        ttk.Radiobutton(op_frame, text="Drop third column (keep first two)", variable=self.operation,
                        value="drop").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Radiobutton(op_frame, text="Calculate new column (first column² × second)", variable=self.operation,
                        value="calc").grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Radiobutton(op_frame, text="Remove braces (delete all { } characters)", variable=self.operation,
                        value="remove_braces").grid(row=0, column=2, sticky=tk.W, padx=5)

        # ========== Operation mode ==========
        mode_frame = ttk.LabelFrame(self, text="Operation mode", padding="5")
        mode_frame.pack(fill=tk.X, pady=5)

        ttk.Radiobutton(mode_frame, text="Process single file", variable=self.mode,
                        value="single", command=self.on_mode_change).grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Radiobutton(mode_frame, text="Batch process all .dat files in directory", variable=self.mode,
                        value="batch", command=self.on_mode_change).grid(row=1, column=0, sticky=tk.W, padx=5)

        # ========== Input columns ==========
        col_frame = ttk.LabelFrame(self, text="Input file columns (only for drop third and calculate new column)", padding="5")
        col_frame.pack(fill=tk.X, pady=5)

        ttk.Radiobutton(col_frame, text="Two-column data", variable=self.input_columns,
                        value=2).grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Radiobutton(col_frame, text="Three-column data", variable=self.input_columns,
                        value=3).grid(row=0, column=1, sticky=tk.W, padx=5)

        # ========== Input settings (dynamic) ==========
        self.input_frame = ttk.LabelFrame(self, text="Input settings", padding="5")
        self.input_frame.pack(fill=tk.X, pady=5)
        self.update_input_frame()

        # ========== Log area (font 10pt) ==========
        log_frame = ttk.LabelFrame(self, text="Processing log", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.log_text = tk.Text(log_frame, wrap=tk.WORD, height=10, font=('Helvetica', 10))
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)

        # ========== Process button ==========
        self.process_btn = ttk.Button(self, text="Start processing", command=self.process)
        self.process_btn.pack(pady=10)

    def on_mode_change(self):
        """Recreate input area when mode changes"""
        self.input_frame.destroy()
        self.input_frame = ttk.LabelFrame(self, text="Input settings", padding="5")
        self.input_frame.pack(fill=tk.X, pady=5)
        self.update_input_frame()

    def update_input_frame(self):
        """Update input controls according to current mode"""
        if self.mode.get() == "single":
            # Single file mode
            ttk.Label(self.input_frame, text="Input file:").grid(row=0, column=0, sticky=tk.W, pady=2)
            ttk.Entry(self.input_frame, textvariable=self.input_path, width=50).grid(row=0, column=1, padx=5)
            ttk.Button(self.input_frame, text="Browse...", command=self.browse_input_file).grid(row=0, column=2)

            ttk.Label(self.input_frame, text="Output file:").grid(row=1, column=0, sticky=tk.W, pady=2)
            ttk.Entry(self.input_frame, textvariable=self.output_path, width=50).grid(row=1, column=1, padx=5)
            ttk.Button(self.input_frame, text="Browse...", command=self.browse_output_file).grid(row=1, column=2)

        else:
            # Batch mode
            ttk.Label(self.input_frame, text="Input directory:").grid(row=0, column=0, sticky=tk.W, pady=2)
            ttk.Entry(self.input_frame, textvariable=self.batch_dir, width=50).grid(row=0, column=1, padx=5)
            ttk.Button(self.input_frame, text="Browse...", command=self.browse_batch_dir).grid(row=0, column=2)

            ttk.Label(self.input_frame, text="Output directory:").grid(row=1, column=0, sticky=tk.W, pady=2)
            ttk.Entry(self.input_frame, textvariable=self.batch_output_dir, width=50).grid(row=1, column=1, padx=5)
            ttk.Button(self.input_frame, text="Browse...", command=self.browse_batch_output_dir).grid(row=1, column=2)

            ttk.Label(self.input_frame, text="Output filename suffix:").grid(row=2, column=0, sticky=tk.W, pady=2)
            ttk.Entry(self.input_frame, textvariable=self.output_suffix, width=20).grid(row=2, column=1, sticky=tk.W, padx=5)
            ttk.Label(self.input_frame, text="(e.g., _out will produce originalname_out.dat)").grid(row=2, column=1, columnspan=2, sticky=tk.W, padx=(120,5))

    def browse_input_file(self):
        filename = filedialog.askopenfilename(
            title="Select input DAT file",
            filetypes=[("DAT files", "*.dat"), ("All files", "*.*")]
        )
        if filename:
            self.input_path.set(filename)
            if not self.output_path.get():
                base, ext = os.path.splitext(filename)
                self.output_path.set(base + self.output_suffix.get() + ext)

    def browse_output_file(self):
        filename = filedialog.asksaveasfilename(
            title="Save output file",
            defaultextension=".dat",
            filetypes=[("DAT files", "*.dat"), ("All files", "*.*")]
        )
        if filename:
            self.output_path.set(filename)

    def browse_batch_dir(self):
        directory = filedialog.askdirectory(title="Select input directory containing .dat files")
        if directory:
            self.batch_dir.set(directory)

    def browse_batch_output_dir(self):
        directory = filedialog.askdirectory(title="Select output directory")
        if directory:
            self.batch_output_dir.set(directory)

    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.update_idletasks()

    def process_file(self, input_path, output_path):
        """Process a single file according to operation type, return success flag"""
        try:
            with open(input_path, 'r', encoding='utf-8') as fin, \
                 open(output_path, 'w', encoding='utf-8') as fout:

                operation = self.operation.get()

                for line_num, line in enumerate(fin, 1):
                    # Remove braces: preserve original format, only delete {} characters
                    if operation == "remove_braces":
                        new_line = line.rstrip('\n').replace('{', '').replace('}', '')
                        if new_line:  # write non-empty lines
                            fout.write(new_line + "\n")
                        continue  # skip column processing

                    # Other operations (drop/calc) need column splitting
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    expected_cols = self.input_columns.get()
                    if len(parts) < expected_cols:
                        self.log(f"Warning: file {input_path} line {line_num} has {len(parts)} columns, expected {expected_cols}, skipped")
                        continue

                    if operation == "drop":
                        # Drop third column: keep first two
                        new_line = f"{parts[0]} {parts[1]}"
                    else:  # calc
                        try:
                            col1 = float(parts[0])
                            col2 = float(parts[1])
                        except ValueError:
                            self.log(f"Error: file {input_path} line {line_num} first two columns contain non-numeric data, skipped")
                            continue
                        # Calculate new value: first column squared × second column
                        new_val = (col1 ** 2) * col2
                        new_line = f"{parts[0]} {new_val}"

                    fout.write(new_line + "\n")

            self.log(f"Success: {input_path} -> {output_path}")
            return True
        except Exception as e:
            self.log(f"Error: processing {input_path} failed: {e}")
            return False

    def process(self):
        """Process button callback"""
        self.log_text.delete(1.0, tk.END)
        mode = self.mode.get()

        if mode == "single":
            input_file = self.input_path.get().strip()
            output_file = self.output_path.get().strip()

            if not input_file:
                messagebox.showerror("Error", "Please select input file.")
                return
            if not output_file:
                messagebox.showerror("Error", "Please specify output file.")
                return
            if not os.path.isfile(input_file):
                messagebox.showerror("Error", f"Input file does not exist: {input_file}")
                return

            self.process_file(input_file, output_file)
            messagebox.showinfo("Complete", "Processing completed!")

        else:  # batch mode
            batch_dir = self.batch_dir.get().strip()
            batch_output_dir = self.batch_output_dir.get().strip()
            suffix = self.output_suffix.get().strip()

            if not batch_dir:
                messagebox.showerror("Error", "Please select input directory.")
                return
            if not os.path.isdir(batch_dir):
                messagebox.showerror("Error", f"Input directory does not exist: {batch_dir}")
                return

            if not batch_output_dir:
                batch_output_dir = batch_dir
                self.log(f"Output directory not specified, using input directory: {batch_output_dir}")
            else:
                if not os.path.exists(batch_output_dir):
                    try:
                        os.makedirs(batch_output_dir)
                        self.log(f"Created output directory: {batch_output_dir}")
                    except Exception as e:
                        messagebox.showerror("Error", f"Unable to create output directory: {batch_output_dir}\n{e}")
                        return
                elif not os.path.isdir(batch_output_dir):
                    messagebox.showerror("Error", f"Output path exists but is not a directory: {batch_output_dir}")
                    return

            pattern = os.path.join(batch_dir, "*.dat")
            dat_files = glob.glob(pattern)

            if not dat_files:
                self.log(f"No .dat files found in {batch_dir}.")
                messagebox.showinfo("Info", "No .dat files found.")
                return

            self.log(f"Found {len(dat_files)} .dat files, starting processing...")
            success_count = 0
            for input_path in dat_files:
                base, ext = os.path.splitext(input_path)
                filename_base = os.path.basename(base)
                output_path = os.path.join(batch_output_dir, filename_base + suffix + ext)

                if self.process_file(input_path, output_path):
                    success_count += 1

            self.log(f"Batch processing complete: {success_count} succeeded, {len(dat_files)-success_count} failed.")
            messagebox.showinfo("Complete", f"Batch processing complete! {success_count} files succeeded.")

# ====================== Data processing tool main window (Toplevel) ======================
class CrystallinityTab(ttk.Frame):
    """
    WAXS Crystallinity Index Tool.

    Deconvolves one broad amorphous halo plus N sharper crystalline Bragg
    peaks from a 1D WAXS I(q) curve (Gaussian components fitted with
    scipy.optimize.curve_fit) and reports the crystallinity index
    CI (%) = area(crystalline peaks) / area(total). Supports single-file
    preview and batch processing across a file series (e.g. a UV-exposure
    time series), with CSV + PDF report export.
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.file_paths = []
        self.current_data = None      # (q, I)
        self.current_file = ""
        self.results = {}             # filepath -> dict(ci, popt, q, I)
        self.init_ui()

    def init_ui(self):
        top_frame = ttk.LabelFrame(self, text="Crystallinity fit parameters", padding=10)
        top_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(top_frame, text="Number of crystalline peaks:").grid(row=0, column=0, sticky=tk.W)
        self.npeaks_entry = ttk.Entry(top_frame, width=6)
        self.npeaks_entry.grid(row=0, column=1, padx=5)
        self.npeaks_entry.insert(0, "2")

        ttk.Label(top_frame, text="Amorphous halo center (q):").grid(row=0, column=2, sticky=tk.W, padx=(20, 0))
        self.halo_center_entry = ttk.Entry(top_frame, width=8)
        self.halo_center_entry.grid(row=0, column=3, padx=5)
        self.halo_center_entry.insert(0, "1.4")

        ttk.Label(top_frame, text="Amorphous halo width guess:").grid(row=0, column=4, sticky=tk.W, padx=(20, 0))
        self.halo_width_entry = ttk.Entry(top_frame, width=8)
        self.halo_width_entry.grid(row=0, column=5, padx=5)
        self.halo_width_entry.insert(0, "0.4")

        ttk.Label(top_frame,
                  text="*Fits amorphous halo + N crystalline Gaussian peaks; CI = crystalline area / total area",
                  foreground="gray").grid(row=1, column=0, columnspan=6, sticky=tk.W, pady=(5, 0))

        main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        left_frame = ttk.Frame(main_pane, width=250)
        main_pane.add(left_frame, weight=1)
        right_frame = ttk.Frame(main_pane)
        main_pane.add(right_frame, weight=3)

        ttk.Label(left_frame, text="Imported WAXS files:").pack(anchor=tk.W)
        self.file_listbox = tk.Listbox(left_frame, height=14, font=('Helvetica', 10))
        self.file_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        self.file_listbox.bind('<<ListboxSelect>>', self.on_file_select)

        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="Import files", command=self.import_files).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Clear list", command=self.clear_list).pack(side=tk.LEFT, padx=2)

        ttk.Button(left_frame, text="Preview fit (current file)", command=self.preview_fit).pack(fill=tk.X, pady=2)
        ttk.Button(left_frame, text="One-click CI (all files)", command=self.batch_compute).pack(fill=tk.X, pady=2)
        ttk.Button(left_frame, text="Export results (CSV + PDF report)",
                   command=self.export_report).pack(fill=tk.X, pady=2)

        self.status_label = ttk.Label(left_frame, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(fill=tk.X, pady=5)

        self.figure, self.ax = plt.subplots(figsize=(7, 5))
        self.canvas = FigureCanvasTkAgg(self.figure, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ax.set_title("Import a WAXS .dat file and preview")
        self.ax.set_xlabel("q (Å⁻¹)")
        self.ax.set_ylabel("Intensity (a.u.)")
        self.figure.tight_layout()

    # ---------- file I/O ----------
    def import_files(self):
        paths = filedialog.askopenfilenames(title="Select WAXS .dat files",
                                             filetypes=[("Data files", "*.dat *.txt *.csv"), ("All files", "*.*")])
        for p in paths:
            if p not in self.file_paths:
                self.file_paths.append(p)
                self.file_listbox.insert(tk.END, os.path.basename(p))
        self.status_label.config(text=f"{len(self.file_paths)} file(s) loaded")

    def clear_list(self):
        self.file_paths = []
        self.file_listbox.delete(0, tk.END)
        self.results = {}
        self.status_label.config(text="Ready")

    def load_curve(self, path):
        arr = np.loadtxt(path, comments=('#', '"'))
        q, I = arr[:, 0], arr[:, 1]
        order = np.argsort(q)
        return q[order], I[order]

    def on_file_select(self, event):
        sel = self.file_listbox.curselection()
        if not sel:
            return
        path = self.file_paths[sel[0]]
        self.current_file = path
        try:
            self.current_data = self.load_curve(path)
            self.draw_current(fit=None)
        except Exception as e:
            messagebox.showerror("Load error", f"Could not read {path}:\n{e}")

    # ---------- fitting ----------
    def _get_params(self):
        n_peaks = int(self.npeaks_entry.get())
        halo_c = float(self.halo_center_entry.get())
        halo_w = float(self.halo_width_entry.get())
        return n_peaks, halo_c, halo_w

    def fit_crystallinity(self, q, I, n_cryst_peaks, halo_center_guess, halo_width_guess,
                          warn=True):
        """
        Fit amorphous halo + N crystalline Gaussians.

        Initial-value strategy:
        - Candidate crystalline peaks come from find_peaks, EXCLUDING peaks
          that fall inside the amorphous halo zone (within ~1 halo width of
          the halo center), so the halo itself is not mistaken for a Bragg
          peak.
        - If fewer distinct candidates than requested peaks are found (e.g.
          a fully amorphous sample), placeholder components start with small
          amplitude spread evenly over the q range (instead of stacked at
          q.mean(), which made the fit degenerate), and a warning is raised.
        Returns (ci_percent, popt, n_candidates_found).
        """
        peak_idx, _ = find_peaks(I, prominence=(I.max() - I.min()) * 0.05)
        if len(peak_idx):
            halo_zone = np.abs(q[peak_idx] - halo_center_guess) < max(halo_width_guess, 0.05)
            cryst_idx = peak_idx[~halo_zone]
        else:
            cryst_idx = peak_idx
        if len(cryst_idx) > n_cryst_peaks:
            cryst_idx = cryst_idx[np.argsort(I[cryst_idx])[-n_cryst_peaks:]]
        cryst_idx = np.sort(cryst_idx)

        if len(cryst_idx) < n_cryst_peaks and warn:
            messagebox.showwarning(
                "Few crystalline peaks detected",
                f"Only {len(cryst_idx)} candidate crystalline peak(s) found away from the "
                f"amorphous halo (requested {n_cryst_peaks}).\n\n"
                "The sample may be fully amorphous, or try reducing 'Number of crystalline peaks'.\n"
                "Remaining components start near zero amplitude - check the preview fit carefully.")

        p0 = [I.max() * 0.5, halo_center_guess, halo_width_guess]
        lo = [0, q.min(), 0.01]
        hi = [I.max() * 2, q.max(), (q.max() - q.min())]
        for idx in list(cryst_idx[:n_cryst_peaks]):
            p0 += [I[idx], q[idx], 0.03]
            lo += [0, q.min(), 0.005]
            hi += [I.max() * 2, q.max(), 0.3]
        # Placeholder components for missing candidates: spread evenly over
        # the q range with small amplitude (near-zero lower bound lets the
        # fit collapse them harmlessly instead of diverging).
        n_fill = n_cryst_peaks - len(cryst_idx[:n_cryst_peaks])
        if n_fill > 0:
            span = q.max() - q.min()
            for j in range(n_fill):
                frac = (j + 1) / (n_fill + 1)
                p0 += [I.max() * 0.05, q.min() + frac * span, max(0.03, span * 0.02)]
                lo += [0, q.min(), 0.005]
                hi += [I.max() * 2, q.max(), 0.3]

        popt, _ = curve_fit(self._multi_gaussian, q, I, p0=p0, bounds=(lo, hi), maxfev=20000)
        dq = np.mean(np.diff(q))
        total_area = 0.0
        crystalline_area = 0.0
        for i in range(0, len(popt), 3):
            area = self._gaussian(q, *popt[i:i + 3]).sum() * dq
            total_area += area
            if i > 0:
                crystalline_area += area
        ci = 100.0 * crystalline_area / total_area
        return ci, popt, len(cryst_idx)

    @staticmethod
    def _gaussian(x, amp, cen, wid):
        return amp * np.exp(-0.5 * ((x - cen) / wid) ** 2)

    def _multi_gaussian(self, x, *params):
        y = np.zeros_like(x)
        for i in range(0, len(params), 3):
            y += self._gaussian(x, params[i], params[i + 1], params[i + 2])
        return y

    def draw_current(self, fit):
        self.ax.clear()
        q, I = self.current_data
        self.ax.plot(q, I, 'k.', ms=3, label="data")
        if fit is not None:
            ci, popt = fit
            self.ax.plot(q, self._gaussian(q, *popt[0:3]), '--', color='tab:blue', label="amorphous halo")
            total = self._gaussian(q, *popt[0:3]).copy()
            for i in range(3, len(popt), 3):
                comp = self._gaussian(q, *popt[i:i + 3])
                total += comp
                self.ax.plot(q, comp, '--', color='tab:red', alpha=0.7)
            self.ax.plot(q, total, '-', color='tab:green', label="total fit")
            self.ax.set_title(f"{os.path.basename(self.current_file)}   CI = {ci:.1f}%")
        else:
            self.ax.set_title(os.path.basename(self.current_file) if self.current_file else "")
        self.ax.set_xlabel("q (Å⁻¹)")
        self.ax.set_ylabel("Intensity (a.u.)")
        self.ax.legend(fontsize=8)
        self.figure.tight_layout()
        self.canvas.draw()

    def preview_fit(self):
        if self.current_data is None:
            messagebox.showwarning("No file", "Select a file first.")
            return
        try:
            n_peaks, halo_c, halo_w = self._get_params()
            q, I = self.current_data
            ci, popt, _ = self.fit_crystallinity(q, I, n_peaks, halo_c, halo_w)
            self.draw_current(fit=(ci, popt))
            self.status_label.config(text=f"CI = {ci:.1f}%")
        except Exception as e:
            messagebox.showerror("Fit error", str(e))

    def batch_compute(self):
        if not self.file_paths:
            messagebox.showwarning("No files", "Import files first.")
            return
        n_peaks, halo_c, halo_w = self._get_params()
        self.results = {}
        failed = []
        few_candidates = []
        for i, path in enumerate(self.file_paths):
            try:
                q, I = self.load_curve(path)
                ci, popt, ncand = self.fit_crystallinity(q, I, n_peaks, halo_c, halo_w,
                                                         warn=False)
                if ncand < n_peaks:
                    few_candidates.append(os.path.basename(path))
                self.results[path] = {'ci': ci, 'popt': popt, 'q': q, 'I': I}
            except Exception as e:
                failed.append((path, str(e)))
            self.status_label.config(text=f"Processing {i + 1}/{len(self.file_paths)}...")
            self.update_idletasks()
        msg = f"Done: {len(self.results)} succeeded"
        if failed:
            msg += f", {len(failed)} failed"
        self.status_label.config(text=msg)
        if few_candidates:
            preview = ", ".join(few_candidates[:5])
            more = f" ... (+{len(few_candidates) - 5} more)" if len(few_candidates) > 5 else ""
            messagebox.showwarning(
                "Few crystalline peaks in some files",
                f"{len(few_candidates)} file(s) had fewer crystalline peak candidates than "
                f"requested ({n_peaks});\nthey may be fully amorphous and their CI values "
                f"should be treated with caution:\n{preview}{more}")
        if failed:
            messagebox.showwarning("Some files failed", "\n".join(f"{os.path.basename(p)}: {e}" for p, e in failed))

    def export_report(self):
        if not self.results:
            messagebox.showwarning("No results", "Run batch CI computation first.")
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".pdf",
                                                   filetypes=[("PDF report", "*.pdf")])
        if not save_path:
            return
        csv_path = os.path.splitext(save_path)[0] + "_CI_summary.csv"
        with open(csv_path, 'w') as f:
            f.write("filename,crystallinity_index_percent\n")
            for path, r in self.results.items():
                f.write(f"{os.path.basename(path)},{r['ci']:.3f}\n")

        pdf = PdfPages(save_path)
        try:
            for path, r in self.results.items():
                fig, ax = plt.subplots(figsize=(7, 5))
                q, I, popt, ci = r['q'], r['I'], r['popt'], r['ci']
                ax.plot(q, I, 'k.', ms=3, label="data")
                total = self._gaussian(q, *popt[0:3]).copy()
                ax.plot(q, total, '--', color='tab:blue', label="amorphous halo")
                for i in range(3, len(popt), 3):
                    comp = self._gaussian(q, *popt[i:i + 3])
                    total = total + comp
                    ax.plot(q, comp, '--', color='tab:red', alpha=0.7)
                ax.plot(q, total, '-', color='tab:green', label="total fit")
                ax.set_title(f"{os.path.basename(path)}   CI = {ci:.1f}%")
                ax.set_xlabel("q (Å⁻¹)")
                ax.set_ylabel("Intensity (a.u.)")
                ax.legend(fontsize=8)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

            names = [os.path.basename(p) for p in self.results]
            cis = [r['ci'] for r in self.results.values()]
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.bar(range(len(names)), cis, color='tab:blue')
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
            ax.set_ylabel("Crystallinity index (%)")
            ax.set_title("Crystallinity index summary")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        finally:
            pdf.close()
        messagebox.showinfo("Export complete", f"Saved:\n{save_path}\n{csv_path}")


class KineticsTab(ttk.Frame):
    """
    Structural Degradation Kinetics Tool.

    Fits a time (or dose) series of a structural parameter -- typically the
    crystallinity index from the Crystallinity tab, but any two-column
    time/value series is accepted -- to first-order decay kinetics
    y(t) = A * exp(-k*t) + C, returning the rate constant k, half-life, and
    R^2 of the fit.
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.t_values = []
        self.y_values = []
        self.fit_result = None
        self.init_ui()

    def init_ui(self):
        top_frame = ttk.LabelFrame(self, text="Time series input", padding=10)
        top_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(top_frame, text="Load CSV (time, value)",
                   command=self.load_csv).grid(row=0, column=0, padx=5)
        ttk.Button(top_frame, text="Add row manually",
                   command=self.add_row_dialog).grid(row=0, column=1, padx=5)
        ttk.Button(top_frame, text="Clear all",
                   command=self.clear_all).grid(row=0, column=2, padx=5)
        ttk.Label(top_frame, text="Model: y(t) = A·exp(−k·t) + C",
                  foreground="gray").grid(row=0, column=3, padx=(20, 0))

        main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        left_frame = ttk.Frame(main_pane, width=250)
        main_pane.add(left_frame, weight=1)
        right_frame = ttk.Frame(main_pane)
        main_pane.add(right_frame, weight=3)

        ttk.Label(left_frame, text="Data points (time, value):").pack(anchor=tk.W)
        self.data_listbox = tk.Listbox(left_frame, height=14, font=('Helvetica', 10))
        self.data_listbox.pack(fill=tk.BOTH, expand=True, pady=5)

        ttk.Button(left_frame, text="Run kinetics fit", command=self.run_fit).pack(fill=tk.X, pady=2)
        ttk.Button(left_frame, text="Export plot + summary",
                   command=self.export_results).pack(fill=tk.X, pady=2)

        self.result_text = tk.Text(left_frame, height=8, font=('Helvetica', 10), wrap=tk.WORD)
        self.result_text.pack(fill=tk.X, pady=5)
        self.result_text.insert(tk.END, "Fit results will appear here.")
        self.result_text.config(state=tk.DISABLED)

        self.status_label = ttk.Label(left_frame, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(fill=tk.X, pady=5)

        self.figure, self.ax = plt.subplots(figsize=(7, 5))
        self.canvas = FigureCanvasTkAgg(self.figure, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ax.set_title("Load or enter a time series")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Structural parameter")
        self.figure.tight_layout()

    def load_csv(self, path=None):
        if path is None:
            path = filedialog.askopenfilename(title="Select CSV (time,value)",
                                               filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            # Auto-detect header: skip first row only if it is non-numeric
            # (previously a fixed skip_header=1 silently dropped the first
            #  data row of header-less CSV files).
            with open(path, 'r', errors='replace') as f:
                first_line = f.readline()
            fields = [p.strip() for p in first_line.split(',')]
            try:
                float(fields[0])
                float(fields[1] if len(fields) > 1 else fields[0])
                has_header = False
            except (ValueError, IndexError):
                has_header = True
            arr = np.genfromtxt(path, delimiter=',',
                                skip_header=1 if has_header else 0)
            if arr.ndim != 2 or arr.shape[1] < 2:
                raise ValueError("CSV must contain at least two columns (time, value).")
            arr = arr[~np.isnan(arr).any(axis=1)]
            if len(arr) == 0:
                raise ValueError("No numeric data rows found.")
            self.t_values = list(arr[:, 0])
            self.y_values = list(arr[:, 1])
            self._refresh_list()
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def add_row_dialog(self):
        t = simpledialog.askfloat("Time", "Enter time/dose value:")
        if t is None:
            return
        y = simpledialog.askfloat("Value", "Enter structural parameter value:")
        if y is None:
            return
        self.t_values.append(t)
        self.y_values.append(y)
        self._refresh_list()

    def clear_all(self):
        self.t_values = []
        self.y_values = []
        self.fit_result = None
        self._refresh_list()

    def _refresh_list(self):
        self.data_listbox.delete(0, tk.END)
        for t, y in zip(self.t_values, self.y_values):
            self.data_listbox.insert(tk.END, f"t={t:g}   y={y:g}")
        self._draw()

    @staticmethod
    def _model(t, A, k, C):
        return A * np.exp(-k * t) + C

    def run_fit(self):
        if len(self.t_values) < 4:
            messagebox.showwarning("Not enough data", "Need at least 4 points to fit 3 parameters.")
            return
        t = np.array(self.t_values, dtype=float)
        y = np.array(self.y_values, dtype=float)
        order = np.argsort(t)
        t, y = t[order], y[order]
        A0 = y[0] - y[-1]
        C0 = y[-1]
        k0 = 1.0 / (t[-1] - t[0] + 1e-6)
        try:
            popt, pcov = curve_fit(self._model, t, y, p0=[A0, k0, C0], maxfev=20000)
        except Exception as e:
            messagebox.showerror("Fit error", str(e))
            return
        A, k, C = popt
        half_life = np.log(2) / k if k > 0 else np.inf
        y_pred = self._model(t, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        self.fit_result = {'t': t, 'y': y, 'popt': popt, 'half_life': half_life, 'r2': r2}

        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete('1.0', tk.END)
        self.result_text.insert(tk.END,
            f"A = {A:.4g}\nk = {k:.4g} (per time unit)\nC = {C:.4g}\n"
            f"Half-life t1/2 = {half_life:.4g}\nR² = {r2:.4f}")
        self.result_text.config(state=tk.DISABLED)
        self._draw()
        self.status_label.config(text="Fit complete")

    def _draw(self):
        self.ax.clear()
        if self.t_values:
            self.ax.plot(self.t_values, self.y_values, 'ko', ms=6, label="data")
        if self.fit_result is not None:
            t, y, popt = self.fit_result['t'], self.fit_result['y'], self.fit_result['popt']
            t_fine = np.linspace(t.min(), t.max(), 200)
            self.ax.plot(t_fine, self._model(t_fine, *popt), '-', color='tab:red',
                         label=f"fit (k={popt[1]:.3g}, R²={self.fit_result['r2']:.3f})")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Structural parameter")
        self.ax.legend(fontsize=8)
        self.figure.tight_layout()
        self.canvas.draw()

    def export_results(self):
        if self.fit_result is None:
            messagebox.showwarning("No fit", "Run the kinetics fit first.")
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".pdf",
                                                   filetypes=[("PDF", "*.pdf"), ("PNG", "*.png")])
        if not save_path:
            return
        self.figure.savefig(save_path, dpi=300)
        txt_path = os.path.splitext(save_path)[0] + "_summary.txt"
        A, k, C = self.fit_result['popt']
        with open(txt_path, 'w') as f:
            f.write("First-order kinetics fit: y(t) = A*exp(-k*t) + C\n")
            f.write(f"A = {A:.6g}\nk = {k:.6g}\nC = {C:.6g}\n")
            f.write(f"Half-life = {self.fit_result['half_life']:.6g}\n")
            f.write(f"R^2 = {self.fit_result['r2']:.6f}\n")
        messagebox.showinfo("Export complete", f"Saved:\n{save_path}\n{txt_path}")


class DataProcessorApp(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.title("SynXS - Data Processing Toolset")
        self.geometry("1100x800")
        self.resizable(True, True)

        # Set global font to 10pt
        self.option_add('*Font', ('Helvetica', 10))
        style = ttk.Style()
        style.configure('.', font=('Helvetica', 10))
        plt.rcParams['font.size'] = 10

        # Create notebook (tabs)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Add three tabs
        self.tab_q2theta = Q2ThetaTab(self.notebook)
        self.tab_peak = PeakFinderTab(self.notebook)
        self.tab_process = DataProcessTab(self.notebook)
        self.tab_crystallinity = CrystallinityTab(self.notebook)
        self.tab_kinetics = KineticsTab(self.notebook)

        self.notebook.add(self.tab_q2theta, text="Q→2θ Conversion")
        self.notebook.add(self.tab_peak, text="Auto Peak Finder + Report")
        self.notebook.add(self.tab_process, text="Data Processing (Multi-function)")
        self.notebook.add(self.tab_crystallinity, text="WAXS Crystallinity Index")
        self.notebook.add(self.tab_kinetics, text="Degradation Kinetics")

        # On close, return to main window
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        self.destroy()
        if self.master:
            self.master.deiconify()


# ==================== Startup interface ====================
class StartWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SynXS")
        self.geometry("400x350")          # 适当加宽
        self.resizable(False, False)

        label = tk.Label(self, text="SynXS Data Processing Suite", font=('Arial', 14, 'bold'))
        label.pack(pady=30)

        btn_frame = tk.Frame(self)
        btn_frame.pack(expand=True, fill='x', padx=20)   # 横向撑满

        btn_integ = tk.Button(btn_frame, text="Integration Tool (SAXS batch)", 
                              command=self.launch_integration, height=2,
                              bg="#4CAF50", fg="white", font=('Arial', 11))
        btn_integ.pack(fill='x', pady=5)

        btn_plot = tk.Button(btn_frame, text="Plotting Tool (Data Automation)", 
                             command=self.launch_plotting, height=2,
                             bg="#2196F3", fg="white", font=('Arial', 11))
        btn_plot.pack(fill='x', pady=5)

        btn_process = tk.Button(btn_frame, text="Data Processing (Q→2θ / Peak / Multi)", 
                                command=self.launch_processing, height=2,
                                bg="#FF9800", fg="white", font=('Arial', 11))
        btn_process.pack(fill='x', pady=5)

        self.protocol("WM_DELETE_WINDOW", self.quit)

    def launch_integration(self):
        self.withdraw()
        tool = SAXSBatchGUI(self)
        tool.protocol("WM_DELETE_WINDOW", lambda: self.on_tool_close(tool))
        tool.focus()

    def launch_plotting(self):
        self.withdraw()
        tool = Application(self)
        tool.protocol("WM_DELETE_WINDOW", lambda: self.on_tool_close(tool))
        tool.focus()

    def launch_processing(self):
        self.withdraw()
        tool = DataProcessorApp(self)
        tool.focus()

    def on_tool_close(self, tool):
        tool.destroy()
        self.deiconify()


# ==================== Program entry point ====================
if __name__ == "__main__":
    mp.freeze_support()
    app = StartWindow()
    app.mainloop()